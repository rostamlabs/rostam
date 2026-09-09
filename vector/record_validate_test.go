// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// Phase 1 stored record values without ever looking inside them, so a plain
// set_payload could put arbitrary bytes under a payload key. The consequence
// was not local to that point: a record IndexEntries cannot enumerate but
// Resolve still answers from POISONS the key for the whole collection (every
// path under it stops accelerating, for every point), and a later
// vector_operate on the same key fails with wire.ErrOperateRecord.
//
// checkRecordValues now runs record.Validate — wire.DecodeRecord's acceptance
// set — at every WIRE-REACHABLE ingest entry, so no newly stored record can be
// in that state. The replay bodies keep the size-only check: they carry bytes
// this store already accepted, and refusing to replay one would discard an
// acked write.

// malformedRecord is a schema mode byte followed by a torn schema blob: it
// decodes far enough to be recognisably a record attempt and no further, which
// is the shape a truncated or forged payload value really arrives in.
func malformedRecord() Metadata {
	return Metadata{"session": NewRecord([]byte{0x01, 0xFF})}
}

func wantMalformed(t *testing.T, what string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: got nil error, want ErrRecordMalformed", what)
	}
	if !errors.Is(err, ErrRecordMalformed) {
		t.Fatalf("%s: err = %v, want ErrRecordMalformed", what, err)
	}
}

// TestMalformedRecordRejectedAtIngestDense covers the dense and IVF entry-point
// family: insert, insert-if-absent, upsert, set-payload, overwrite and the bulk
// stage/build path. Each rejects with ErrRecordMalformed and leaves the point
// exactly as it was.
func TestMalformedRecordRejectedAtIngestDense(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{{"hnsw", oversizeDenseConfig()}, {"ivf", oversizeIVFConfig()}} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("default/malformed", tc.cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			vec := []float32{1, 0, 0, 0}
			good := Metadata{"rc": NewInt(1)}
			if err := c.Insert(1, vec, 0, good, nil); err != nil {
				t.Fatalf("baseline Insert: %v", err)
			}

			wantMalformed(t, "Insert", c.Insert(2, vec, 0, malformedRecord(), nil))
			if _, _, _, _, _, ok := c.Get(2); ok {
				t.Fatal("rejected Insert still created point 2")
			}

			_, err = c.InsertIfAbsent(3, vec, 0, malformedRecord(), nil)
			wantMalformed(t, "InsertIfAbsent", err)
			if _, _, _, _, _, ok := c.Get(3); ok {
				t.Fatal("rejected InsertIfAbsent still created point 3")
			}

			wantMalformed(t, "Upsert", c.Upsert(1, vec, "doc", 0, malformedRecord(), nil))
			// Upsert is delete-then-insert: the point must survive a rejection.
			_, meta, _, _, _, ok := c.Get(1)
			if !ok {
				t.Fatal("rejected Upsert deleted the existing point 1")
			}
			if got, gok := meta["rc"]; !gok || !got.Equal(NewInt(1)) {
				t.Fatalf("point 1 payload changed after a rejected Upsert: %v", meta)
			}

			wantMalformed(t, "SetPayload", c.SetPayload(1, malformedRecord(), nil))
			wantMalformed(t, "OverwritePayload", c.OverwritePayload(1, malformedRecord(), nil))
			_, meta, _, _, _, _ = c.Get(1)
			if got, gok := meta["rc"]; !gok || !got.Equal(NewInt(1)) {
				t.Fatalf("point 1 payload changed after a rejected payload op: %v", meta)
			}
			if _, gok := meta["session"]; gok {
				t.Fatal("rejected payload op still stored the malformed record")
			}

			// Bulk: the stage refuses the batch outright, so nothing is queued.
			bulk, berr := NewCollection("default/malformed-bulk", tc.cfg)
			if berr != nil {
				t.Fatalf("NewCollection bulk: %v", berr)
			}
			wantMalformed(t, "StageBulkPayloads", bulk.StageBulkPayloads(
				[]uint64{9}, [][]float32{vec}, []Metadata{malformedRecord()}))
			wantMalformed(t, "BuildConcurrentMeta", bulk.BuildConcurrentMeta(
				[]uint64{9}, [][]float32{vec}, []Metadata{malformedRecord()}, 1))
			if n := bulk.Stats().Size; n != 0 {
				t.Fatalf("rejected bulk build left %d points", n)
			}
		})
	}
}

// TestMalformedRecordRejectedAtIngestNamed covers the named-vector family.
func TestMalformedRecordRejectedAtIngestNamed(t *testing.T) {
	nc, err := NewNamedCollection("default/named-malformed", namedTestConfig())
	if err != nil {
		t.Fatalf("NewNamedCollection: %v", err)
	}
	vecs := map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}
	good := Metadata{"rc": NewInt(1)}
	if err := nc.Insert(1, vecs, good, 0); err != nil {
		t.Fatalf("baseline Insert: %v", err)
	}

	wantMalformed(t, "NamedInsert", nc.Insert(2, vecs, malformedRecord(), 0))
	if _, _, _, _, ok := nc.Get(2); ok {
		t.Fatal("rejected named Insert still created point 2")
	}
	wantMalformed(t, "NamedSetPayload", nc.SetPayload(1, malformedRecord(), nil))
	wantMalformed(t, "NamedOverwritePayload", nc.OverwritePayload(1, malformedRecord(), nil))
	_, payload, _, _, ok := nc.Get(1)
	if !ok {
		t.Fatal("point 1 disappeared")
	}
	if got, gok := payload["rc"]; !gok || !got.Equal(NewInt(1)) {
		t.Fatalf("named payload changed after a rejected op: %v", payload)
	}
	if _, gok := payload["session"]; gok {
		t.Fatal("rejected named payload op still stored the malformed record")
	}
}

// TestMalformedRecordRejectedAtIngestMultiVector covers the multi-vector
// family, including the versioned if-absent path that does NOT delegate to
// AddIfAbsent and the wire-reachable MultiRestoreAdd pair.
func TestMalformedRecordRejectedAtIngestMultiVector(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	tokens := [][]float32{{1, 0, 0, 0}}
	good := Metadata{"rc": NewInt(1)}
	if err := m.Add(1, tokens, good); err != nil {
		t.Fatalf("baseline Add: %v", err)
	}

	wantMalformed(t, "MVAdd", m.Add(2, tokens, malformedRecord()))
	if _, _, _, ok := m.Get(2); ok {
		t.Fatal("rejected MV Add still created doc 2")
	}
	_, err = m.AddIfAbsent(3, tokens, malformedRecord())
	wantMalformed(t, "MVAddIfAbsent", err)
	if _, _, _, ok := m.Get(3); ok {
		t.Fatal("rejected MV AddIfAbsent still created doc 3")
	}
	_, err = m.MultiAddIfAbsentVersion(4, tokens, malformedRecord(), nil, 5)
	wantMalformed(t, "MVMultiAddIfAbsentVersion", err)
	if _, _, _, ok := m.Get(4); ok {
		t.Fatal("rejected MV MultiAddIfAbsentVersion still created doc 4")
	}

	wantMalformed(t, "MVSetPayload", m.SetPayload(1, malformedRecord(), nil))
	wantMalformed(t, "MVOverwritePayload", m.OverwritePayload(1, malformedRecord(), nil))
	_, payload, _, ok := m.Get(1)
	if !ok {
		t.Fatal("doc 1 disappeared")
	}
	if got, gok := payload["rc"]; !gok || !got.Equal(NewInt(1)) {
		t.Fatalf("MV payload changed after a rejected op: %v", payload)
	}
	if _, gok := payload["session"]; gok {
		t.Fatal("rejected MV payload op still stored the malformed record")
	}
}

// TestMalformedRecordRejectedAtWireReachableRestore is the half the phase-2
// pre-flight scan added. The "restore" family is NOT all replay: ops/builtin.go
// routes an ordinary WIRE insert to Collection.RestoreInsert/RestoreInsertAt
// whenever the decoded args carry a non-zero version (the version-preserving
// reinsert), and ops/multivector.go + ops/mv_batch.go route one to
// MultiRestoreAddSparse. Those entries carry the CALLER's metadata, so they are
// ingest and must apply the shape check like any other ingest entry.
//
// The engine-level bodies underneath them (idx.RestoreInsert, idx.RestorePayload,
// the MV restoreAdd) stay size-only — TestRestorePathsSkipShapeValidation is the
// other side of this ruling.
func TestMalformedRecordRejectedAtWireReachableRestore(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{{"hnsw", oversizeDenseConfig()}, {"ivf", oversizeIVFConfig()}} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("default/malformed-restore", tc.cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			vec := []float32{1, 0, 0, 0}
			wantMalformed(t, "RestoreInsert", c.RestoreInsert(2, vec, 0, malformedRecord(), nil, nil, 5))
			if _, _, _, _, _, ok := c.Get(2); ok {
				t.Fatal("rejected RestoreInsert still created point 2")
			}
			wantMalformed(t, "RestoreInsertAt", c.RestoreInsertAt(3, vec, 0, malformedRecord(), nil, nil, 5, 1_000))
			if _, _, _, _, _, ok := c.Get(3); ok {
				t.Fatal("rejected RestoreInsertAt still created point 3")
			}
		})
	}

	t.Run("multivector", func(t *testing.T) {
		m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			t.Fatalf("NewMultiVectorIndex: %v", err)
		}
		tokens := [][]float32{{1, 0, 0, 0}}
		wantMalformed(t, "MultiRestoreAdd", m.MultiRestoreAdd(1, tokens, malformedRecord(), nil, 5))
		if _, _, _, ok := m.Get(1); ok {
			t.Fatal("rejected MultiRestoreAdd still created doc 1")
		}
		wantMalformed(t, "MultiRestoreAddSparse", m.MultiRestoreAddSparse(2, tokens, malformedRecord(), nil, 5, nil))
		if _, _, _, ok := m.Get(2); ok {
			t.Fatal("rejected MultiRestoreAddSparse still created doc 2")
		}
	})
}

// TestWellFormedRecordStillAccepted is the guard against an over-broad gate:
// the same table with the canonical session record must go through everywhere.
// Without it, a Validate that rejected everything would pass every test above.
func TestWellFormedRecordStillAccepted(t *testing.T) {
	rec := func() Metadata { return Metadata{"session": NewRecord(sessionRecordBytes(t))} }

	for _, tc := range []struct {
		name string
		cfg  Config
	}{{"hnsw", oversizeDenseConfig()}, {"ivf", oversizeIVFConfig()}} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("default/wellformed", tc.cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			vec := []float32{1, 0, 0, 0}
			if err := c.Insert(1, vec, 0, rec(), nil); err != nil {
				t.Fatalf("Insert: %v", err)
			}
			if _, err := c.InsertIfAbsent(2, vec, 0, rec(), nil); err != nil {
				t.Fatalf("InsertIfAbsent: %v", err)
			}
			if err := c.Upsert(1, vec, "doc", 0, rec(), nil); err != nil {
				t.Fatalf("Upsert: %v", err)
			}
			if err := c.SetPayload(1, rec(), nil); err != nil {
				t.Fatalf("SetPayload: %v", err)
			}
			if err := c.OverwritePayload(1, rec(), nil); err != nil {
				t.Fatalf("OverwritePayload: %v", err)
			}
			if err := c.RestoreInsert(4, vec, 0, rec(), nil, nil, 5); err != nil {
				t.Fatalf("RestoreInsert: %v", err)
			}
			if err := c.RestoreInsertAt(5, vec, 0, rec(), nil, nil, 5, 1_000); err != nil {
				t.Fatalf("RestoreInsertAt: %v", err)
			}

			bulk, berr := NewCollection("default/wellformed-bulk", tc.cfg)
			if berr != nil {
				t.Fatalf("NewCollection bulk: %v", berr)
			}
			if err := bulk.StageBulkPayloads([]uint64{9}, [][]float32{vec}, []Metadata{rec()}); err != nil {
				t.Fatalf("StageBulkPayloads: %v", err)
			}
			if err := bulk.BuildConcurrentMeta([]uint64{10}, [][]float32{vec}, []Metadata{rec()}, 1); err != nil {
				t.Fatalf("BuildConcurrentMeta: %v", err)
			}
		})
	}

	t.Run("named", func(t *testing.T) {
		nc, err := NewNamedCollection("default/named-wellformed", namedTestConfig())
		if err != nil {
			t.Fatalf("NewNamedCollection: %v", err)
		}
		vecs := map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}
		if err := nc.Insert(1, vecs, rec(), 0); err != nil {
			t.Fatalf("named Insert: %v", err)
		}
		if err := nc.SetPayload(1, rec(), nil); err != nil {
			t.Fatalf("named SetPayload: %v", err)
		}
		if err := nc.OverwritePayload(1, rec(), nil); err != nil {
			t.Fatalf("named OverwritePayload: %v", err)
		}
		if err := nc.RestoreInsert(2, vecs, nil, rec(), 0, nil, 5); err != nil {
			t.Fatalf("named RestoreInsert: %v", err)
		}
	})

	t.Run("multivector", func(t *testing.T) {
		m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			t.Fatalf("NewMultiVectorIndex: %v", err)
		}
		tokens := [][]float32{{1, 0, 0, 0}}
		if err := m.Add(1, tokens, rec()); err != nil {
			t.Fatalf("MV Add: %v", err)
		}
		if _, err := m.AddIfAbsent(2, tokens, rec()); err != nil {
			t.Fatalf("MV AddIfAbsent: %v", err)
		}
		if _, err := m.MultiAddIfAbsentVersion(3, tokens, rec(), nil, 5); err != nil {
			t.Fatalf("MV MultiAddIfAbsentVersion: %v", err)
		}
		if err := m.SetPayload(1, rec(), nil); err != nil {
			t.Fatalf("MV SetPayload: %v", err)
		}
		if err := m.OverwritePayload(1, rec(), nil); err != nil {
			t.Fatalf("MV OverwritePayload: %v", err)
		}
		if err := m.MultiRestoreAdd(4, tokens, rec(), nil, 5); err != nil {
			t.Fatalf("MV MultiRestoreAdd: %v", err)
		}
	})
}

// TestRestorePathsSkipShapeValidation is the other half of the pre-flight
// ruling. The REPLAY bodies — the engine-level RestoreInsert/RestorePayload, the
// named RestoreInsert the WAL replays through, the MV restoreAdd — must accept a
// malformed record, because they replay bytes this store already ACKED. A shape
// check there would refuse to rebuild an acknowledged write, which is worse than
// the poison it would prevent: the poison is a lost acceleration, this would be
// lost data. They keep the size-only check, which cannot refuse anything the WAL
// or a snapshot can actually hold.
func TestRestorePathsSkipShapeValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{{"hnsw", oversizeDenseConfig()}, {"ivf", oversizeIVFConfig()}} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("default/replay", tc.cfg)
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			vec := []float32{1, 0, 0, 0}
			if err := c.idx.RestoreInsert(1, vec, 0, malformedRecord(), nil, nil, 5); err != nil {
				t.Fatalf("idx.RestoreInsert on a malformed record: %v, want nil (replay must not refuse acked data)", err)
			}
			if err := c.idx.RestoreInsertAt(2, vec, 0, malformedRecord(), nil, nil, 5, 1_000); err != nil {
				t.Fatalf("idx.RestoreInsertAt on a malformed record: %v, want nil", err)
			}
			if err := c.idx.RestorePayload(1, malformedRecord(), nil, 6); err != nil {
				t.Fatalf("idx.RestorePayload on a malformed record: %v, want nil", err)
			}
			_, meta, _, _, _, ok := c.Get(1)
			if !ok {
				t.Fatal("the replayed point is not live")
			}
			if got := meta["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, []byte{0x01, 0xFF}) {
				t.Fatalf("the replayed record was not stored verbatim: %+v", got)
			}
		})
	}

	t.Run("named", func(t *testing.T) {
		nc, err := NewNamedCollection("default/named-replay", namedTestConfig())
		if err != nil {
			t.Fatalf("NewNamedCollection: %v", err)
		}
		vecs := map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}
		if err := nc.RestoreInsert(1, vecs, nil, malformedRecord(), 0, nil, 5); err != nil {
			t.Fatalf("named RestoreInsert on a malformed record: %v, want nil", err)
		}
		nc.restorePayload(1, malformedRecord(), nil, 6)
		_, payload, _, _, ok := nc.Get(1)
		if !ok {
			t.Fatal("the replayed named point is not live")
		}
		if got := payload["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, []byte{0x01, 0xFF}) {
			t.Fatalf("the replayed named record was not stored verbatim: %+v", got)
		}
	})

	t.Run("multivector", func(t *testing.T) {
		m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			t.Fatalf("NewMultiVectorIndex: %v", err)
		}
		tokens := [][]float32{{1, 0, 0, 0}}
		if err := m.restoreAdd(1, tokens, malformedRecord(), nil, 5, nil); err != nil {
			t.Fatalf("MV restoreAdd on a malformed record: %v, want nil", err)
		}
		m.restorePayload(1, malformedRecord(), nil, 6)
		_, payload, _, ok := m.Get(1)
		if !ok {
			t.Fatal("the replayed MV doc is not live")
		}
		if got := payload["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, []byte{0x01, 0xFF}) {
			t.Fatalf("the replayed MV record was not stored verbatim: %+v", got)
		}
	})
}

// TestWALReplayRestoresAMalformedRecord is the end-to-end statement of the same
// ruling: a WAL holding a record this store accepted BEFORE the gate existed
// still replays on reopen. The log record is written directly (the public entry
// no longer produces one), then the store is closed and reopened, and replay
// must rebuild the point rather than drop it.
func TestWALReplayRestoresAMalformedRecord(t *testing.T) {
	dir := t.TempDir()
	cs, c := openMutateWALCollection(t, dir)

	seq, err := c.wal.appendSetPayloadStaged(1, malformedRecord(), nil, 2)
	if err != nil {
		t.Fatalf("appendSetPayloadStaged: %v", err)
	}
	if err := c.wal.commitWaitStaged(seq); err != nil {
		t.Fatalf("commitWaitStaged: %v", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	_, meta, _, _, version, ok := c2.Get(1)
	if !ok {
		t.Fatal("replay dropped the point holding a malformed record — an acked write was lost")
	}
	if version != 2 {
		t.Fatalf("replayed version = %d, want 2", version)
	}
	if got := meta["session"]; got.Kind != ValueRecord || !bytes.Equal(got.Rec, []byte{0x01, 0xFF}) {
		t.Fatalf("replay did not restore the record verbatim: %+v", got)
	}
}

// BenchmarkCheckRecordValues puts the write-path cost of the shape check on the
// record: a payload with no ValueRecord at all (the overwhelmingly common case,
// which must stay free), the session fixture, and a table-heavy 64 KiB record,
// which is where DecodeRecord's per-row allocation shows up.
func BenchmarkCheckRecordValues(b *testing.B) {
	sess := benchVectorSessionRecord(b)
	large := benchVectorTableRecord(b, 64<<10)

	for _, c := range []struct {
		name string
		m    Metadata
	}{
		{"no-record", Metadata{"country": NewString("DE"), "rc": NewInt(7)}},
		{"session", Metadata{"session": NewRecord(sess)}},
		{"table-64KiB", Metadata{"session": NewRecord(large)}},
	} {
		b.Run("shape/"+c.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := checkRecordValues(c.m); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run("size-only/"+c.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := checkRecordValuesSize(c.m); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// benchVectorSessionRecord is the canonical session record for a benchmark
// (sessionRecordBytes needs a *testing.T).
func benchVectorSessionRecord(b *testing.B) []byte {
	b.Helper()
	s := &wire.Schema{Version: 3, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "rc", Type: wire.OperateTypeU8},
		{Name: "bal", Type: wire.OperateTypeI32},
		{Name: "tag", Type: wire.OperateTypeBytes},
	}}
	if err := s.Validate(); err != nil {
		b.Fatalf("schema invalid: %v", err)
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
	}}).Encode()
	if enc == nil {
		b.Fatal("bench session record failed to encode")
	}
	return enc
}

// benchVectorTableRecord builds a schema-mode record whose single TABLE field
// holds enough rows to reach roughly size bytes.
func benchVectorTableRecord(b *testing.B, size int) []byte {
	b.Helper()
	s := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "t", Type: wire.OperateTypeTable, Table: &wire.TableDef{
			KeyType: wire.OperateTypeU64,
			Cols: []wire.ColumnDef{
				{Name: "hi", Type: wire.OperateTypeU32},
				{Name: "lo", Type: wire.OperateTypeU32},
			},
		}},
	}}
	if err := s.Validate(); err != nil {
		b.Fatalf("schema invalid: %v", err)
	}
	const rowWidth = 16 // 8-byte key + two U32 columns
	rows := make([]wire.Row, 0, size/rowWidth)
	for i := 0; i < size/rowWidth; i++ {
		key := make([]byte, 8)
		for j := 0; j < 8; j++ {
			key[j] = byte(uint64(i) >> (8 * j)) //nolint:gosec // bounded loop counter
		}
		rows = append(rows, wire.Row{Key: key, Cols: []wire.Col{
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(i)}},     //nolint:gosec // bounded
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: uint64(i) + 3}}, //nolint:gosec // bounded
		}})
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: rows}},
	}}).Encode()
	if enc == nil {
		b.Fatal("bench table record failed to encode")
	}
	return enc
}
