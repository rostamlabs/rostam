// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

// mutateRecordCfg is the dense config every engine-level test in this file
// uses (the brief's fixture: dim 4, L2, seeded so the graph is deterministic).
func mutateRecordCfg() Config {
	return Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1}
}

// seedMutateHNSW returns a dense engine holding point 1 with a "session"
// record (rc=7) — the single point almost every case below mutates.
func seedMutateHNSW(t *testing.T) *hnsw {
	t.Helper()
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	seedMutatePoint(t, h, 1)
	return h
}

// seedMutatePoint inserts id carrying a "session" record with rc=7 into any
// engine, so the dense and IVF tables run against identical state.
func seedMutatePoint(t *testing.T, ix VectorIndex, id uint64) {
	t.Helper()
	vec := make([]float32, ix.Dim())
	vec[0] = float32(id)
	if _, _, err := ix.Insert(id, vec, 0, Metadata{"session": NewRecord(sessionRecordBytes(t))}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert(%d): %v", id, err)
	}
}

// storeFn is a mutator that unconditionally stores rec.
func storeFn(rec []byte) RecordMutator {
	return func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return rec, RecordStore, nil }
}

// stampedRecordBytes is a dynamic-mode record {at: I64 = stamp}. It stands in
// for an op-list containing STAMP: the value is baked in by the CALLER's
// pinned clock, so a replay that re-ran the op-list under a different clock
// would produce different bytes.
func stampedRecordBytes(t *testing.T, stamp int64) []byte {
	t.Helper()
	enc := (&wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "at", Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(stamp)}},
	}}).Encode()
	if enc == nil {
		t.Fatal("stamped record failed to encode")
	}
	return enc
}

// storedMeta returns the metadata the arena actually holds for id — the state
// Get normalizes away (it reports nil for any empty payload) and the state a WAL
// replay has to reproduce.
//
// It takes the INTERFACE and switches on the concrete engine so the shared table
// can assert the empty-payload-stores-nil normalisation on the IVF copy too, not
// only on the dense one it was first written against.
func storedMeta(t *testing.T, ix VectorIndex, id uint64) Metadata {
	t.Helper()
	switch e := ix.(type) {
	case *hnsw:
		e.mu.RLock()
		defer e.mu.RUnlock()
		slot, ok := e.arena.Slot(id)
		if !ok {
			t.Fatalf("id %d has no slot", id)
		}
		return e.arena.Metadata(slot)
	case *ivf:
		e.mu.RLock()
		defer e.mu.RUnlock()
		slot, ok := e.arena.Slot(id)
		if !ok {
			t.Fatalf("id %d has no slot", id)
		}
		return e.arena.Metadata(slot)
	default:
		t.Fatalf("storedMeta: unhandled engine %T — add it, or the nil normalisation goes unchecked", ix)
		return nil
	}
}

// pointVersion reads id's current version through the public read path.
func pointVersion(t *testing.T, ix VectorIndex, id uint64) uint64 {
	t.Helper()
	_, _, _, _, v, ok := ix.Get(id)
	if !ok {
		t.Fatalf("Get(%d): point is not live", id)
	}
	return v
}

// recordUnder returns the record bytes stored under key in id's payload.
func recordUnder(t *testing.T, ix VectorIndex, id uint64, key string) ([]byte, bool) {
	t.Helper()
	_, meta, _, _, _, ok := ix.Get(id)
	if !ok {
		t.Fatalf("Get(%d): point is not live", id)
	}
	v, present := meta[key]
	if !present {
		return nil, false
	}
	if v.Kind != ValueRecord {
		t.Fatalf("payload key %q holds kind %d, want a record", key, v.Kind)
	}
	return v.Rec, true
}

// ---------------------------------------------------------------------------
// engine-level behaviour (dense)
// ---------------------------------------------------------------------------

func TestMutateRecordStoresAndBumps(t *testing.T) {
	h := seedMutateHNSW(t)
	before := pointVersion(t, h, 1)
	want := sessionRecordBytesRC(t, 8)

	var sawOld []byte
	var sawExists bool
	meta, ke, version, changed, err := h.MutatePayloadRecord(1, "session", func(old []byte, exists bool) ([]byte, RecordMutation, error) {
		sawOld = append([]byte(nil), old...)
		sawExists = exists
		return want, RecordStore, nil
	}, CASCond{})
	if err != nil {
		t.Fatalf("MutatePayloadRecord: %v", err)
	}
	if !changed {
		t.Fatal("changed=false for a RecordStore — a stored record is a change")
	}
	if version != before+1 {
		t.Fatalf("version = %d, want %d (exactly one bump)", version, before+1)
	}
	if !sawExists {
		t.Error("the mutator saw exists=false for a key that holds a record")
	}
	if !bytes.Equal(sawOld, sessionRecordBytesRC(t, 7)) {
		t.Errorf("the mutator saw %d old bytes, want the stored rc=7 record", len(sawOld))
	}
	if got := meta["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, want) {
		t.Error("the returned payload does not carry the new record bytes")
	}
	if len(ke) != 0 {
		t.Errorf("resulting per-key deadlines = %v, want none", ke)
	}
	got, present := recordUnder(t, h, 1, "session")
	if !present || !bytes.Equal(got, want) {
		t.Fatal("Get(1) does not show the new record bytes")
	}
	if v := pointVersion(t, h, 1); v != version {
		t.Fatalf("Get(1) version = %d, want the returned %d", v, version)
	}
}

func TestMutateRecordCreatesUnderAbsentKey(t *testing.T) {
	h := seedMutateHNSW(t)
	want := sessionRecordBytesRC(t, 8)

	var sawOld []byte
	var sawExists, called bool
	_, _, _, changed, err := h.MutatePayloadRecord(1, "fresh", func(old []byte, exists bool) ([]byte, RecordMutation, error) {
		called, sawOld, sawExists = true, old, exists
		return want, RecordStore, nil
	}, CASCond{})
	if err != nil {
		t.Fatalf("MutatePayloadRecord: %v", err)
	}
	if !called {
		t.Fatal("the mutator was never called for an absent key")
	}
	if sawOld != nil || sawExists {
		t.Errorf("the mutator saw (%v, %v) for an absent key, want (nil, false)", sawOld, sawExists)
	}
	if !changed {
		t.Fatal("changed=false after creating a record under an absent key")
	}
	fresh, ok := recordUnder(t, h, 1, "fresh")
	if !ok || !bytes.Equal(fresh, want) {
		t.Fatal("the new key does not hold the stored record")
	}
	session, ok := recordUnder(t, h, 1, "session")
	if !ok || !bytes.Equal(session, sessionRecordBytes(t)) {
		t.Fatal("the untouched key changed — a mutation must only touch its own key")
	}
}

func TestMutateRecordUnchangedIsANoOp(t *testing.T) {
	t.Run("engine", func(t *testing.T) {
		h := seedMutateHNSW(t)
		before := pointVersion(t, h, 1)
		meta, ke, version, changed, err := h.MutatePayloadRecord(1, "session",
			func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordUnchanged, nil }, CASCond{})
		if err != nil {
			t.Fatalf("MutatePayloadRecord: %v", err)
		}
		if changed {
			t.Fatal("changed=true for RecordUnchanged")
		}
		if version != before {
			t.Fatalf("version = %d after a no-op, want the unbumped %d", version, before)
		}
		if meta != nil || ke != nil {
			t.Errorf("a no-op returned a payload/deadline map (%v, %v) — nothing was applied", meta, ke)
		}
		got, ok := recordUnder(t, h, 1, "session")
		if !ok || !bytes.Equal(got, sessionRecordBytes(t)) {
			t.Fatal("a no-op changed the stored bytes")
		}
	})

	t.Run("wal", func(t *testing.T) {
		_, c := seedMutateWALCollection(t)
		before := pointVersion(t, c.idx, 1)
		seqBefore := walWriteSeq(c)
		version, err := c.MutatePayloadRecordCAS(1, "session",
			func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordUnchanged, nil }, CASCond{})
		if err != nil {
			t.Fatalf("MutatePayloadRecordCAS: %v", err)
		}
		if version != before {
			t.Fatalf("version = %d after a no-op, want the unbumped %d", version, before)
		}
		if got := walWriteSeq(c); got != seqBefore {
			t.Fatalf("the WAL advanced %d -> %d on a no-op — a failed CHECK must not cost a log record", seqBefore, got)
		}
	})
}

func TestMutateRecordDelete(t *testing.T) {
	h := seedMutateHNSW(t)
	before := pointVersion(t, h, 1)

	meta, _, version, changed, err := h.MutatePayloadRecord(1, "session",
		func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordDelete, nil }, CASCond{})
	if err != nil {
		t.Fatalf("MutatePayloadRecord (delete): %v", err)
	}
	if !changed || version != before+1 {
		t.Fatalf("delete of a present key: changed=%v version=%d, want true/%d", changed, version, before+1)
	}
	if _, present := meta["session"]; present {
		t.Error("the returned payload still holds the deleted key")
	}
	if _, ok := recordUnder(t, h, 1, "session"); ok {
		t.Fatal("the payload key survived a RecordDelete")
	}
	// Deleting the only key empties the payload, and an empty payload is STORED
	// as nil — the same normalization DeletePayloadKeys applies. Get cannot see
	// the difference (it returns nil for any empty payload), so this asserts on
	// the arena: an empty non-nil map there would differ from the state the same
	// WAL record rebuilds on replay, which decodes an absent payload as nil.
	if got := storedMeta(t, h, 1); got != nil {
		t.Errorf("stored payload = %v after deleting the only key, want nil", got)
	}

	// A second delete of the now-absent key is a deliberate no-op.
	_, _, version2, changed2, err := h.MutatePayloadRecord(1, "session",
		func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordDelete, nil }, CASCond{})
	if err != nil {
		t.Fatalf("MutatePayloadRecord (second delete): %v", err)
	}
	if changed2 {
		t.Error("changed=true deleting an already-absent key")
	}
	if version2 != version {
		t.Fatalf("version = %d after deleting an absent key, want the unbumped %d", version2, version)
	}
}

func TestMutateRecordNonRecordValueIsAnError(t *testing.T) {
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0},
		0, Metadata{"session": NewRecord(sessionRecordBytes(t)), "country": NewString("DE")}, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	before := pointVersion(t, h, 1)

	var called bool
	_, _, _, changed, err := h.MutatePayloadRecord(1, "country", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
		called = true
		return sessionRecordBytes(t), RecordStore, nil
	}, CASCond{})
	if !errors.Is(err, ErrPayloadKeyNotRecord) {
		t.Fatalf("err = %v, want ErrPayloadKeyNotRecord", err)
	}
	if called {
		t.Error("the mutator ran against a non-record value — the key must be rejected before fn")
	}
	if changed {
		t.Error("changed=true on a rejected call")
	}
	_, meta, _, _, version, _ := h.Get(1)
	if got := meta["country"]; !got.Equal(NewString("DE")) {
		t.Errorf("the non-record value changed: %+v", got)
	}
	if version != before {
		t.Fatalf("version = %d after a rejected call, want %d", version, before)
	}
}

func TestMutateRecordOversizeRejected(t *testing.T) {
	h := seedMutateHNSW(t)
	before := pointVersion(t, h, 1)

	_, _, _, changed, err := h.MutatePayloadRecord(1, "session", storeFn(make([]byte, maxRecordValueBytes+1)), CASCond{})
	if !errors.Is(err, ErrRecordTooLarge) {
		t.Fatalf("err = %v, want ErrRecordTooLarge", err)
	}
	if changed {
		t.Error("changed=true on a rejected oversize store")
	}
	got, ok := recordUnder(t, h, 1, "session")
	if !ok || !bytes.Equal(got, sessionRecordBytes(t)) {
		t.Fatal("the point changed despite the oversize rejection")
	}
	if v := pointVersion(t, h, 1); v != before {
		t.Fatalf("version = %d after a rejected store, want %d", v, before)
	}
	// The invariant the cap exists for: the snapshot writer must still be able
	// to encode every stored value.
	if err := h.Snapshot(io.Discard); err != nil {
		t.Fatalf("Snapshot after a rejected oversize store: %v", err)
	}
}

func TestMutateRecordMutatorErrorLeavesPointUnchanged(t *testing.T) {
	h := seedMutateHNSW(t)
	before := pointVersion(t, h, 1)

	_, _, _, changed, err := h.MutatePayloadRecord(1, "session",
		func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordStore, wire.ErrOperateRecord }, CASCond{})
	if !errors.Is(err, wire.ErrOperateRecord) {
		t.Fatalf("err = %v, want the mutator's error verbatim", err)
	}
	if changed {
		t.Error("changed=true after a mutator error")
	}
	got, ok := recordUnder(t, h, 1, "session")
	if !ok || !bytes.Equal(got, sessionRecordBytes(t)) {
		t.Fatal("the payload changed after a mutator error")
	}
	if v := pointVersion(t, h, 1); v != before {
		t.Fatalf("version = %d after a mutator error, want %d", v, before)
	}
}

func TestMutateRecordCASConflict(t *testing.T) {
	h := seedMutateHNSW(t)
	before := pointVersion(t, h, 1)

	var called bool
	_, _, _, changed, err := h.MutatePayloadRecord(1, "session", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
		called = true
		return sessionRecordBytesRC(t, 8), RecordStore, nil
	}, CASCond{Expected: 99, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("err = %v, want ErrVersionConflict", err)
	}
	if called {
		t.Error("the mutator ran under a failed CAS")
	}
	if changed {
		t.Error("changed=true under a failed CAS")
	}
	if v := pointVersion(t, h, 1); v != before {
		t.Fatalf("version = %d after a CAS conflict, want %d", v, before)
	}
}

func TestMutateRecordDeadPointNotFound(t *testing.T) {
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatal(err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }
	seedMutatePoint(t, h, 1)
	// Point 2 carries a 1s point TTL, so it expires by the wall clock the
	// non-At form reads.
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, time.Second,
		Metadata{"session": NewRecord(sessionRecordBytes(t))}, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.Delete(1, CASCond{}); err != nil {
		t.Fatal(err)
	}
	fakeNow += 5000

	for _, id := range []uint64{1, 2, 404} {
		var called bool
		_, _, _, changed, err := h.MutatePayloadRecord(id, "session", func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
			called = true
			return sessionRecordBytesRC(t, 8), RecordStore, nil
		}, CASCond{})
		if !errors.Is(err, ErrIDNotFound) {
			t.Errorf("id %d: err = %v, want ErrIDNotFound", id, err)
		}
		if called {
			t.Errorf("id %d: the mutator ran against a dead point", id)
		}
		if changed {
			t.Errorf("id %d: changed=true against a dead point", id)
		}
	}
}

func TestMutateRecordAtUsesTheStampedClock(t *testing.T) {
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatal(err)
	}
	// Pin the wall clock at 4000 for the seeding SetPayload so the "session"
	// key gets an ABSOLUTE deadline of 5000, then move it far away: every
	// judgement below must come from the explicit stamp, not from here.
	var fakeNow int64 = 4000
	h.now = func() int64 { return fakeNow }
	seedMutatePoint(t, h, 1)
	if _, ke, _, err := h.SetPayload(1, Metadata{"session": NewRecord(sessionRecordBytes(t))},
		map[string]int64{"session": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	} else if ke["session"] != 5000 {
		t.Fatalf("seeded deadline = %d, want 5000", ke["session"])
	}
	fakeNow = 9_000_000 // far past every stamp used below

	// One millisecond before the deadline the key is still live.
	var sawExists bool
	if _, _, _, _, err := h.MutatePayloadRecordAt(1, "session", func(_ []byte, exists bool) ([]byte, RecordMutation, error) {
		sawExists = exists
		return nil, RecordUnchanged, nil
	}, CASCond{}, 4999); err != nil {
		t.Fatalf("MutatePayloadRecordAt(4999): %v", err)
	}
	if !sawExists {
		t.Fatal("at stamp 4999 the mutator saw the key as absent — the wall clock was consulted")
	}

	// At the deadline the key reads as ABSENT, and storing under it must drop
	// the stale deadline, or the record just written would be invisible.
	want := sessionRecordBytesRC(t, 8)
	_, ke, _, changed, err := h.MutatePayloadRecordAt(1, "session", func(_ []byte, exists bool) ([]byte, RecordMutation, error) {
		sawExists = exists
		return want, RecordStore, nil
	}, CASCond{}, 5000)
	if err != nil {
		t.Fatalf("MutatePayloadRecordAt(5000): %v", err)
	}
	if sawExists {
		t.Fatal("at stamp 5000 the mutator saw the expired key as present")
	}
	if !changed {
		t.Fatal("changed=false storing under an expired key")
	}
	if _, present := ke["session"]; present {
		t.Fatalf("the stale deadline survived: %v", ke)
	}
	fakeNow = 6000
	_, meta, _, _, _, ok := h.GetProjected(1, false, true)
	if !ok {
		t.Fatal("GetProjected(1): point not live")
	}
	got, present := meta["session"]
	if !present || !bytes.Equal(got.Rec, want) {
		t.Fatal("the freshly stored record is invisible at now=6000 — the stale deadline was not dropped")
	}
}

// ---------------------------------------------------------------------------
// auto-index exactness (phase-1 invariant, restated after a mutation)
// ---------------------------------------------------------------------------

// assertRecordPostingsExact re-proves the phase-1 exactness invariant over the
// given ids: for the synthetic field, the index's postings agree EXACTLY with
// what lookupPath resolves from each live slot's live metadata.
func assertRecordPostingsExact(t *testing.T, h *hnsw, ids []uint64, field string) {
	t.Helper()
	h.mu.RLock()
	defer h.mu.RUnlock()
	now := uint64(h.now()) //nolint:gosec // test clock is non-negative
	posted := make(map[uint32]scalarKey)
	for key, slots := range h.payloadIdx.fields[field] {
		for slot := range slots {
			if prev, dup := posted[slot]; dup {
				t.Errorf("slot %d is posted under two keys for %q: %+v and %+v", slot, field, prev, key)
			}
			posted[slot] = key
		}
	}
	for _, id := range ids {
		slot, ok := h.arena.Slot(id)
		if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
			continue
		}
		v, resolved := lookupPath(h.liveMeta(slot, now), field)
		key, scalar := scalarKeyOf(v)
		gotKey, gotPosted := posted[slot]
		delete(posted, slot)
		if resolved && scalar {
			if !gotPosted {
				t.Errorf("id %d resolves %q = %+v but has no posting", id, field, key)
			} else if gotKey != key {
				t.Errorf("id %d resolves %q = %+v but is posted under %+v", id, field, key, gotKey)
			}
			continue
		}
		if gotPosted {
			t.Errorf("id %d does not resolve %q but is posted under %+v", id, field, gotKey)
		}
	}
	for slot, key := range posted {
		t.Errorf("%q has a posting for slot %d under %+v that no live id explains", field, slot, key)
	}
}

func TestMutateRecordIndexExactAfterMutation(t *testing.T) {
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, 0, 12)
	for i := 1; i <= 12; i++ {
		id := uint64(i)
		vec := []float32{float32(i), float32(i % 5), 0, 1}
		if _, _, err := h.Insert(id, vec, 0,
			Metadata{"session": NewRecord(sessionRecordBytesRC(t, uint8(i%5)))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	// Point 1 goes to rc=7 and then to rc=8, so the mutation must move it.
	if _, _, _, _, err := h.MutatePayloadRecord(1, "session", storeFn(sessionRecordBytesRC(t, 7)), CASCond{}); err != nil {
		t.Fatal(err)
	}
	assertRecordPostingsExact(t, h, ids, "session/rc")
	if _, _, _, _, err := h.MutatePayloadRecord(1, "session", storeFn(sessionRecordBytesRC(t, 8)), CASCond{}); err != nil {
		t.Fatal(err)
	}

	slot, ok := h.arena.Slot(1)
	if !ok {
		t.Fatal("point 1 has no slot")
	}
	if recordSlotIn(h.payloadIdx, "session/rc", intKey(7), slot) {
		t.Error("fields[session/rc][7] still holds the mutated slot — stale posting after a record mutation")
	}
	if !recordSlotIn(h.payloadIdx, "session/rc", intKey(8), slot) {
		t.Error("fields[session/rc][8] is missing the mutated slot")
	}
	assertRecordPostingsExact(t, h, ids, "session/rc")
}

func TestMutateRecordReindexesBadRecordPoison(t *testing.T) {
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatal(err)
	}
	// A mode byte followed by a torn schema blob: IndexEntries cannot
	// enumerate it, so the payload key is poisoned and nothing under it
	// accelerates. Seeded through the REPLAY entry because the public ones now
	// refuse a malformed record (ErrRecordMalformed) — see insertViaReplay.
	insertViaReplay(t, h, 1, []float32{1, 0, 0, 0}, Metadata{"session": NewRecord([]byte{0x01, 0xFF})})
	if !h.payloadIdx.badRecords.poisoned("session") {
		t.Fatal("the malformed record did not poison the payload key — the fixture is wrong")
	}

	want := sessionRecordBytes(t)
	if _, _, _, changed, err := h.MutatePayloadRecord(1, "session", storeFn(want), CASCond{}); err != nil {
		t.Fatalf("MutatePayloadRecord over a malformed record: %v", err)
	} else if !changed {
		t.Fatal("changed=false replacing a malformed record")
	}
	if h.payloadIdx.badRecords.poisoned("session") {
		t.Fatal("the payload key stayed poisoned after the malformed record was replaced")
	}
	slot, _ := h.arena.Slot(1)
	if !recordSlotIn(h.payloadIdx, "session/rc", intKey(7), slot) {
		t.Error("the repaired record did not regain acceleration (no session/rc posting)")
	}
}

// ---------------------------------------------------------------------------
// WAL: one record per applied mutation, replayed verbatim
// ---------------------------------------------------------------------------

// walWriteSeq reads the WAL's monotonic record counter under its own lock.
func walWriteSeq(c *Collection) uint64 {
	c.wal.syncMu.Lock()
	defer c.wal.syncMu.Unlock()
	return c.wal.writeSeq
}

// seedMutateWALCollection opens a WAL-mode store holding point 1 with a
// "session" record and returns both, with the store closed on cleanup.
func seedMutateWALCollection(t *testing.T) (*CollectionStore, *Collection) {
	t.Helper()
	cs, c := openMutateWALCollection(t, t.TempDir())
	t.Cleanup(func() { _ = cs.Close() })
	return cs, c
}

// openMutateWALCollection opens (or creates) the WAL collection in dir. The
// caller owns Close — the reopen test closes it deliberately, without a Flush.
func openMutateWALCollection(t *testing.T, dir string) (*CollectionStore, *Collection) {
	t.Helper()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	vec := make([]float32, 16)
	vec[0] = 1
	if err := c.Insert(1, vec, 0, Metadata{"session": NewRecord(sessionRecordBytes(t))}, nil); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	return cs, c
}

func TestMutateRecordSurvivesReopen(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(t *testing.T, c *Collection) (version uint64)
		verify func(t *testing.T, c2 *Collection, version uint64)
	}{
		{
			name: "store",
			mutate: func(t *testing.T, c *Collection) uint64 {
				v, err := c.MutatePayloadRecordCAS(1, "session", storeFn(sessionRecordBytesRC(t, 8)), CASCond{})
				if err != nil {
					t.Fatalf("MutatePayloadRecordCAS: %v", err)
				}
				return v
			},
			verify: func(t *testing.T, c2 *Collection, version uint64) {
				_, meta, _, _, v, ok := c2.Get(1)
				if !ok {
					t.Fatal("point 1 lost across reopen")
				}
				if v != version {
					t.Fatalf("version = %d after replay, want the acked %d", v, version)
				}
				if got := meta["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, sessionRecordBytesRC(t, 8)) {
					t.Fatal("the mutated record did not replay byte-identically")
				}
			},
		},
		{
			name: "delete",
			mutate: func(t *testing.T, c *Collection) uint64 {
				v, err := c.MutatePayloadRecordCAS(1, "session",
					func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordDelete, nil }, CASCond{})
				if err != nil {
					t.Fatalf("MutatePayloadRecordCAS (delete): %v", err)
				}
				return v
			},
			verify: func(t *testing.T, c2 *Collection, version uint64) {
				_, meta, _, _, v, ok := c2.Get(1)
				if !ok {
					t.Fatal("point 1 lost across reopen")
				}
				if v != version {
					t.Fatalf("version = %d after replay, want the acked %d", v, version)
				}
				if _, present := meta["session"]; present {
					t.Fatal("the deleted key came back on replay — the WAL record is a REPLACE, not a merge")
				}
				h, ok := c2.idx.(*hnsw)
				if !ok {
					t.Fatal("the reopened collection is not dense")
				}
				if got := storedMeta(t, h, 1); got != nil {
					t.Errorf("replayed stored payload = %v, want nil (the pre-restart state)", got)
				}
				h.mu.RLock()
				defer h.mu.RUnlock()
				if len(h.payloadIdx.fields["session/rc"]) != 0 {
					t.Fatalf("the rebuilt index still posts session/rc: %v", h.payloadIdx.fields["session/rc"])
				}
			},
		},
		{
			name: "stamped op-list",
			mutate: func(t *testing.T, c *Collection) uint64 {
				const stamp int64 = 1_700_000_000_000
				v, err := c.MutatePayloadRecordCASAt(1, "session", storeFn(stampedRecordBytes(t, stamp)), CASCond{}, stamp)
				if err != nil {
					t.Fatalf("MutatePayloadRecordCASAt: %v", err)
				}
				return v
			},
			verify: func(t *testing.T, c2 *Collection, version uint64) {
				const stamp int64 = 1_700_000_000_000
				_, meta, _, _, v, ok := c2.Get(1)
				if !ok {
					t.Fatal("point 1 lost across reopen")
				}
				if v != version {
					t.Fatalf("version = %d after replay, want the acked %d", v, version)
				}
				if got := meta["session"]; !bytes.Equal(got.Rec, stampedRecordBytes(t, stamp)) {
					t.Fatal("the stamped record changed on replay — the op-list must never be re-run")
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cs, c := openMutateWALCollection(t, dir)
			version := tc.mutate(t, c)
			// NO Flush: the WAL alone must carry the mutation.
			if err := cs.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			reopened, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatalf("reopen: %v", err)
			}
			t.Cleanup(func() { _ = reopened.Close() })
			c2, ok := reopened.Get("docs")
			if !ok {
				t.Fatal("collection missing after reopen")
			}
			tc.verify(t, c2, version)
		})
	}
}

// ---------------------------------------------------------------------------
// cross-engine parity (dense, IVF, named, multi-vector)
// ---------------------------------------------------------------------------

// mutateHarness is ONE engine behind the shared record-mutation table: a seeded
// point 1 holding a "session" record, plus the four accessors every case needs.
// The dense, IVF, named and multi-vector implementations are four copies of the
// same body, so they are held to ONE table (TestMutateRecordNamedAndMVMatchDense
// runs all four) and a divergence in any of them fails loudly.
//
// mutate targets the ENGINE-level entry (which reports changed), not the
// collection wrapper, so the table sees the no-op flag directly on every family.
type mutateHarness struct {
	name string
	// insert seeds an ADDITIONAL point carrying meta (point 1 is already seeded
	// with an rc=7 "session" record when the harness is built).
	insert func(t *testing.T, id uint64, meta Metadata)
	// setNow pins the engine's wall clock, so the non-At entries and every read
	// below are judged against an explicit, deterministic time.
	setNow func(ms int64)
	// setPayload is the family's set_payload, used only to seed a per-key TTL.
	setPayload func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64)
	mutate     func(id uint64, key string, fn RecordMutator, cas CASCond) (version uint64, changed bool, err error)
	mutateAt   func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (version uint64, changed bool, err error)
	// mutateAtKE is mutateAt keeping the RESULTING per-key deadline map, which
	// mutateAt drops. It is normalised to int64 unix-millis (the slot-keyed
	// engines report uint64, the id-keyed ones int64) so one table can assert on
	// all four engines.
	mutateAtKE func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (map[string]int64, uint64, bool, error)
	// payload is the live payload as a reader sees it at the pinned clock
	// (per-key-expired keys already dropped); ok=false for a dead/absent point.
	payload func(id uint64) (Metadata, bool)
	// version is the point's current version, 0 for a dead/absent point.
	version func(id uint64) uint64
	// stored is the RAW stored payload, before the nil normalisation Get hides.
	stored func(t *testing.T, id uint64) Metadata
}

// record reads the record bytes under key from the harness's live payload.
func (h mutateHarness) record(t *testing.T, id uint64, key string) ([]byte, bool) {
	t.Helper()
	meta, ok := h.payload(id)
	if !ok {
		t.Fatalf("%s: point %d is not live", h.name, id)
	}
	v, present := meta[key]
	if !present {
		return nil, false
	}
	if v.Kind != ValueRecord {
		t.Fatalf("%s: payload key %q holds kind %d, want a record", h.name, key, v.Kind)
	}
	return v.Rec, true
}

// mutateHarnessBase is the clock every harness is pinned to before a case runs,
// so "now" is explicit rather than whatever time.Now happened to return.
const mutateHarnessBase int64 = 1_000_000_000_000

// denseMutateHarness seeds a dense engine holding point 1 with an rc=7 record.
func denseMutateHarness(t *testing.T) mutateHarness {
	t.Helper()
	h, err := newHNSW(mutateRecordCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	h.now = func() int64 { return mutateHarnessBase }
	seedMutatePoint(t, h, 1)
	return vectorIndexHarness("hnsw", h, func(ms int64) { h.now = func() int64 { return ms } })
}

// ivfMutateHarness seeds an IVF engine with the identical point.
func ivfMutateHarness(t *testing.T) mutateHarness {
	t.Helper()
	ix, err := newIVF(ivfTestConfig(4))
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	ix.now = func() int64 { return mutateHarnessBase }
	seedMutatePoint(t, ix, 1)
	return vectorIndexHarness("ivf", ix, func(ms int64) { ix.now = func() int64 { return ms } })
}

// vectorIndexHarness adapts either slot-keyed engine to the shared table.
func vectorIndexHarness(name string, ix VectorIndex, setNow func(int64)) mutateHarness {
	return mutateHarness{
		name:   name,
		setNow: setNow,
		insert: func(t *testing.T, id uint64, meta Metadata) {
			t.Helper()
			vec := make([]float32, ix.Dim())
			vec[0] = float32(id)
			if _, _, err := ix.Insert(id, vec, 0, meta, nil, nil, CASCond{}); err != nil {
				t.Fatalf("%s: Insert(%d): %v", name, id, err)
			}
		},
		setPayload: func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64) {
			t.Helper()
			if _, _, _, err := ix.SetPayload(id, meta, keyTTLMs, CASCond{}); err != nil {
				t.Fatalf("%s: SetPayload(%d): %v", name, id, err)
			}
		},
		mutate: func(id uint64, key string, fn RecordMutator, cas CASCond) (uint64, bool, error) {
			_, _, version, changed, err := ix.MutatePayloadRecord(id, key, fn, cas)
			return version, changed, err
		},
		mutateAt: func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, bool, error) {
			_, _, version, changed, err := ix.MutatePayloadRecordAt(id, key, fn, cas, nowMs)
			return version, changed, err
		},
		mutateAtKE: func(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (map[string]int64, uint64, bool, error) {
			_, ke, version, changed, err := ix.MutatePayloadRecordAt(id, key, fn, cas, nowMs)
			return deadlinesInt64(ke), version, changed, err
		},
		payload: func(id uint64) (Metadata, bool) {
			_, meta, _, _, _, ok := ix.Get(id)
			return meta, ok
		},
		version: func(id uint64) uint64 {
			_, _, _, _, v, ok := ix.Get(id)
			if !ok {
				return 0
			}
			return v
		},
		stored: func(t *testing.T, id uint64) Metadata {
			t.Helper()
			return storedMeta(t, ix, id)
		},
	}
}

// deadlinesInt64 converts a slot-keyed engine's absolute per-key deadline map to
// the int64 unix-millis the id-keyed engines already report, so the shared table
// compares like with like.
func deadlinesInt64(ke map[string]uint64) map[string]int64 {
	if ke == nil {
		return nil
	}
	out := make(map[string]int64, len(ke))
	for k, v := range ke {
		out[k] = int64(v) //nolint:gosec // absolute unix-millis deadlines fit int64
	}
	return out
}

// mutateCase is one engine-level scenario, run against every engine so no copy
// of the body can drift from the others.
type mutateCase struct {
	name        string
	key         string
	fn          RecordMutator
	cas         CASCond
	wantErr     error
	wantChanged bool
	wantBump    bool
	wantRec     func(t *testing.T) ([]byte, bool) // resulting bytes under "session"
	// wantStoredNil asserts the RAW stored payload is nil afterwards — the
	// empty-payload normalisation Get hides (it reports nil for any empty
	// payload) and a WAL replay has to reproduce.
	wantStoredNil bool
}

func mutateCases(t *testing.T) []mutateCase {
	t.Helper()
	return []mutateCase{
		{
			name: "store", key: "session", fn: storeFn(sessionRecordBytesRC(t, 8)),
			wantChanged: true, wantBump: true,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytesRC(t, 8), true },
		},
		{
			name: "unchanged", key: "session",
			fn:      func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordUnchanged, nil },
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			name: "delete", key: "session",
			fn:          func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordDelete, nil },
			wantChanged: true, wantBump: true,
			wantRec: func(*testing.T) ([]byte, bool) { return nil, false },
			// "session" is the point's ONLY key, so deleting it empties the payload
			// — which every engine must store as nil, or the live state differs
			// from the state the same WAL record rebuilds on replay.
			wantStoredNil: true,
		},
		{
			name: "cas conflict", key: "session", fn: storeFn(sessionRecordBytesRC(t, 8)),
			cas: CASCond{Expected: 99, Has: true}, wantErr: ErrVersionConflict,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			name: "mutator error", key: "session",
			fn: func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
				return nil, RecordStore, wire.ErrOperateRecord
			},
			wantErr: wire.ErrOperateRecord,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			name: "oversize", key: "session", fn: storeFn(make([]byte, maxRecordValueBytes+1)),
			wantErr: ErrRecordTooLarge,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			name: "unknown mutation", key: "session",
			fn:      func(_ []byte, _ bool) ([]byte, RecordMutation, error) { return nil, RecordMutation(9), nil },
			wantErr: ErrRecordMutation,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			name: "empty key", key: "", fn: storeFn(sessionRecordBytesRC(t, 8)),
			wantErr: ErrPayloadKeyNotRecord,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			// The reserved document-content key never holds a record: rejected on
			// the same guard as the empty key, before the point is even read.
			name: "content field", key: contentField, fn: storeFn(sessionRecordBytesRC(t, 8)),
			wantErr: ErrPayloadKeyNotRecord,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
		{
			// A nil mutator is a caller bug, caught by the same guard rather than
			// panicking inside the write-lock critical section.
			name: "nil mutator", key: "session", fn: nil,
			wantErr: ErrPayloadKeyNotRecord,
			wantRec: func(t *testing.T) ([]byte, bool) { return sessionRecordBytes(t), true },
		},
	}
}

// runMutateCase applies one case to a freshly seeded engine and asserts the
// outcome, so every engine is held to the same table.
func runMutateCase(t *testing.T, h mutateHarness, tc mutateCase) {
	t.Helper()
	before := h.version(1)
	version, changed, err := h.mutate(1, tc.key, tc.fn, tc.cas)
	if tc.wantErr != nil {
		if !errors.Is(err, tc.wantErr) {
			t.Fatalf("%s: err = %v, want %v", tc.name, err, tc.wantErr)
		}
	} else if err != nil {
		t.Fatalf("%s: unexpected err %v", tc.name, err)
	}
	if changed != tc.wantChanged {
		t.Errorf("%s: changed = %v, want %v", tc.name, changed, tc.wantChanged)
	}
	if tc.wantErr == nil {
		want := before
		if tc.wantBump {
			want = before + 1
		}
		if version != want {
			t.Errorf("%s: returned version = %d, want %d", tc.name, version, want)
		}
	}
	wantBytes, wantPresent := tc.wantRec(t)
	got, present := h.record(t, 1, "session")
	if present != wantPresent {
		t.Fatalf("%s: session present = %v, want %v", tc.name, present, wantPresent)
	}
	if present && !bytes.Equal(got, wantBytes) {
		t.Errorf("%s: stored bytes differ from the expected result", tc.name)
	}
	if tc.wantStoredNil {
		if stored := h.stored(t, 1); stored != nil {
			t.Errorf("%s: raw stored payload = %v, want nil (an emptied payload stores nil)", tc.name, stored)
		}
	}
	live := h.version(1)
	wantLive := before
	if tc.wantBump {
		wantLive = before + 1
	}
	if live != wantLive {
		t.Errorf("%s: point version = %d, want %d", tc.name, live, wantLive)
	}
}

func TestMutateRecordIVFMatchesHNSW(t *testing.T) {
	for _, tc := range mutateCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("hnsw", func(t *testing.T) { runMutateCase(t, denseMutateHarness(t), tc) })
			t.Run("ivf", func(t *testing.T) { runMutateCase(t, ivfMutateHarness(t), tc) })
		})
	}
}

// ---------------------------------------------------------------------------
// the stamped/unstamped asymmetry on a key that carries a deadline
// ---------------------------------------------------------------------------

// TestMutateRecordUnstampedKeepsTheKeyDeadline pins the rule that makes an
// unstamped replicated apply deterministic: on the UNSTAMPED path the mutator
// must not consult the clock for the payload key's deadline at all.
//
// Without it, two replicas applying the same entry with wall clocks either side
// of the deadline disagree about whether the record EXISTS — one creates a fresh
// record, the other increments the stored one — and they hold different bytes
// forever. So the unstamped path treats a deadline-passed record as present,
// mutates it in place, and passes the deadline through untouched; only the
// stamped path (whose clock is the leader's, identical everywhere) may expire
// the key. See recordKeyPastDeadline (vector/record_mutate.go).
//
// Runs on all four engines, because the rule lives in four bodies.
func TestMutateRecordUnstampedKeepsTheKeyDeadline(t *testing.T) {
	const deadline = mutateHarnessBase + 1000

	t.Run("unstamped mutates in place and keeps the deadline", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				// Seed an absolute deadline of base+1000 on the record key.
				h.setNow(mutateHarnessBase)
				h.setPayload(t, 1, Metadata{"session": NewRecord(sessionRecordBytes(t))},
					map[string]int64{"session": 1000})

				// Move this "replica's" wall clock well past the deadline.
				h.setNow(deadline + 4000)

				want := sessionRecordBytesRC(t, 8)
				var sawOld []byte
				var sawExists bool
				version, changed, err := h.mutate(1, "session", func(old []byte, exists bool) ([]byte, RecordMutation, error) {
					sawOld, sawExists = append([]byte(nil), old...), exists
					return want, RecordStore, nil
				}, CASCond{})
				if err != nil {
					t.Fatalf("%s: unstamped mutate past the deadline: %v", h.name, err)
				}
				if !sawExists {
					t.Fatalf("%s: the unstamped mutator saw the record as ABSENT — a wall clock decided whether it exists, "+
						"which is exactly what makes two replicas store different bytes", h.name)
				}
				if !bytes.Equal(sawOld, sessionRecordBytes(t)) {
					t.Fatalf("%s: the unstamped mutator got old=%x, want the stored record %x",
						h.name, sawOld, sessionRecordBytes(t))
				}
				if !changed || version == 0 {
					t.Fatalf("%s: changed=%v version=%d, want a real applied mutation", h.name, changed, version)
				}

				// The deadline survived: BEFORE it the freshly written record is
				// visible, AFTER it the key is gone again. A dropped deadline would
				// leave the key visible at both clocks.
				h.setNow(deadline - 1)
				if rec, ok := h.record(t, 1, "session"); !ok || !bytes.Equal(rec, want) {
					t.Fatalf("%s: before the deadline the mutated record is not readable (ok=%v)", h.name, ok)
				}
				h.setNow(deadline)
				if _, ok := h.record(t, 1, "session"); ok {
					t.Fatalf("%s: the key is still live at its deadline — the unstamped mutation DROPPED the deadline "+
						"instead of passing it through", h.name)
				}
			})
		}
	})

	// The control: the stamped path is the one that may expire the key, and it
	// still does. A stamp past the deadline reads the record as absent and drops
	// the stale deadline, so the record it writes is immediately visible.
	t.Run("stamped past the deadline still reads absent", func(t *testing.T) {
		for _, newH := range allMutateHarnesses(t) {
			h := newH(t)
			t.Run(h.name, func(t *testing.T) {
				h.setNow(mutateHarnessBase)
				h.setPayload(t, 1, Metadata{"session": NewRecord(sessionRecordBytes(t))},
					map[string]int64{"session": 1000})
				// The wall clock stays BEFORE the deadline: anything that comes out
				// "expired" below was decided by the stamp, not by the clock.
				h.setNow(mutateHarnessBase)

				want := sessionRecordBytesRC(t, 9)
				var sawExists bool
				ke, version, changed, err := h.mutateAtKE(1, "session", func(_ []byte, exists bool) ([]byte, RecordMutation, error) {
					sawExists = exists
					return want, RecordStore, nil
				}, CASCond{}, deadline)
				if err != nil {
					t.Fatalf("%s: stamped mutate at the deadline: %v", h.name, err)
				}
				if sawExists {
					t.Fatalf("%s: the stamped mutator saw the deadline-passed record as PRESENT — the stamp was ignored", h.name)
				}
				if !changed || version == 0 {
					t.Fatalf("%s: changed=%v version=%d, want a real applied mutation", h.name, changed, version)
				}
				if _, present := ke["session"]; present {
					t.Fatalf("%s: the stale deadline survived a stamped mutation: %v", h.name, ke)
				}
				// The record just written is visible at a clock past the old deadline.
				h.setNow(deadline + 4000)
				if rec, ok := h.record(t, 1, "session"); !ok || !bytes.Equal(rec, want) {
					t.Fatalf("%s: the record stored by the stamped mutation is invisible (ok=%v) — the stale deadline was not dropped",
						h.name, ok)
				}
			})
		}
	})
}
