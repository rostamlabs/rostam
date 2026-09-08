// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// tornDynamicRecord is a dynamic-mode record {a: I64 = 5, z: BYTES "hello"}
// with its LAST BYTE removed. It is the exact asymmetry this file is about:
// IndexEntries walks every field and fails on the damaged tail, while Resolve
// reads only the bytes on its own path and answers "a" perfectly well. An
// index that treats "IndexEntries failed" as "this record contributes nothing"
// therefore has no posting for a field the predicate still matches.
func tornDynamicRecord(t *testing.T) []byte {
	t.Helper()
	enc := (&wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "a", Cell: wire.Cell{Type: wire.OperateTypeI64, U: 5}},
		{Name: "z", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hello")}},
	}}).Encode()
	if enc == nil {
		t.Fatal("dynamic record failed to encode")
	}
	return enc[:len(enc)-1]
}

// goodDynamicRecord is tornDynamicRecord's intact twin, with a caller-chosen
// value for "a" so a corpus can vary it.
func goodDynamicRecord(t *testing.T, a int64) []byte {
	t.Helper()
	enc := (&wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "a", Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(a)}},
		{Name: "z", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hello")}},
	}}).Encode()
	if enc == nil {
		t.Fatal("dynamic record failed to encode")
	}
	return enc
}

// tornSchemaRecord is the schema-mode route to the same asymmetry, and it is a
// DIFFERENT route: walkSchemaFields fails on the damaged variable-length tail
// ("z"), but schemaFieldOffset hands back the FIXED field "a"'s offset without
// walking the tail at all, so "a" still resolves.
func tornSchemaRecord(t *testing.T) []byte {
	t.Helper()
	s := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "a", Type: wire.OperateTypeI64},
		{Name: "z", Type: wire.OperateTypeBytes},
	}}
	if err := s.Validate(); err != nil {
		t.Fatalf("torn schema invalid: %v", err)
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeI64, U: 5}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hello")}},
	}}).Encode()
	if enc == nil {
		t.Fatal("torn schema record failed to encode")
	}
	return enc[:len(enc)-1]
}

// namelessSchemaRecord is a schema-mode record whose schema stores NO names, so
// IndexEntries labels its fields positionally ("#0") and a by-name path cannot
// resolve at all. It is the other half of the positional-decline argument: the
// corpus must contain records of both spellings, or "positional paths are not
// indexed" is only ever tested against records that name their fields.
func namelessSchemaRecord(t *testing.T, a int64) []byte {
	t.Helper()
	s := &wire.Schema{Version: 1, StoreNames: false, Fields: []wire.FieldDef{
		{Type: wire.OperateTypeI64},
	}}
	if err := s.Validate(); err != nil {
		t.Fatalf("nameless schema invalid: %v", err)
	}
	enc := (&wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(a)}},
	}}).Encode()
	if enc == nil {
		t.Fatal("nameless schema record failed to encode")
	}
	return enc
}

// TestRecordTornFixturesReproduceTheAsymmetry pins the PRECONDITION of every
// test below: these bytes really do fail IndexEntries while still resolving a
// field. If a future change to sdk/record makes IndexEntries partial (or makes
// Resolve fail alongside it), the fail-closed machinery is being tested against
// a fixture that no longer reproduces anything, and this test says so directly
// instead of letting the others pass vacuously.
func TestRecordTornFixturesReproduceTheAsymmetry(t *testing.T) {
	for _, c := range []struct {
		name string
		rec  []byte
	}{
		{"dynamic", tornDynamicRecord(t)},
		{"schema fixed field after a damaged tail", tornSchemaRecord(t)},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := recordResolver.IndexEntries(c.rec); err == nil {
				t.Fatal("IndexEntries succeeded — the fixture is not damaged and proves nothing")
			}
			v, ok := lookupPath(Metadata{"torn": NewRecord(c.rec)}, "torn/a")
			if !ok || !v.Equal(NewInt(5)) {
				t.Fatalf("lookupPath(torn/a) = (%+v, %v), want (5, true) — the fixture does not reproduce the asymmetry", v, ok)
			}
		})
	}
}

// poisonCorpus inserts n points carrying an intact record under "torn" (a = i%9)
// and replaces ONE of them (id badID) with the damaged record supplied, so the
// payload key holds exactly one unindexable record among many good ones.
func poisonCorpus(t *testing.T, n int, badID uint64, bad []byte) (*hnsw, map[uint64][]float32, map[uint64]Metadata) {
	t.Helper()
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		id := uint64(i)
		v := []float32{float32(i), float32(i % 7), 0, 1}
		m := Metadata{"torn": NewRecord(goodDynamicRecord(t, int64(i%9)))} //nolint:gosec // bounded
		if id == badID {
			m = Metadata{"torn": NewRecord(bad)}
		}
		corpus[id] = v
		metas[id] = m
		if _, _, err := h.Insert(id, v, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	return h, corpus, metas
}

// TestRecordPoisonedKeyFailsClosed is the fix-round-1 regression test. A record
// that IndexEntries cannot fully enumerate contributes NO postings, but its
// intact fields still resolve — so an accelerated query over any path under
// that payload key would drop the row. The index must therefore fail CLOSED:
// while any live slot under a payload key holds an unindexable record, every
// path under that key declines to narrow and falls back to the predicate.
//
// It drives both damage routes and both consumers (SearchFiltered and the
// matchingIDs selection delete-by-filter and scroll use), and then checks the
// key regains acceleration once the bad record is replaced — fail-closed must
// be a state, not a one-way latch.
func TestRecordPoisonedKeyFailsClosed(t *testing.T) {
	const badID = 7
	routes := []struct {
		name string
		rec  []byte
	}{
		{"dynamic", tornDynamicRecord(t)},
		{"schema fixed field after a damaged tail", tornSchemaRecord(t)},
	}
	for _, route := range routes {
		t.Run(route.name, func(t *testing.T) {
			h, corpus, metas := poisonCorpus(t, 60, badID, route.rec)
			f := Filter{Op: FilterEq, Field: "torn/a", Value: NewInt(5)}
			pred := compileOrFail(t, f)

			want := bruteMatchIDs(metas, pred)
			if _, ok := want[badID]; !ok {
				t.Fatalf("the damaged point %d does not satisfy the predicate — the fixture proves nothing", badID)
			}

			gotIDs, err := h.matchingIDs(f, pred)
			if err != nil {
				t.Fatalf("matchingIDs: %v", err)
			}
			got := make(map[uint64]struct{}, len(gotIDs))
			for _, id := range gotIDs {
				got[id] = struct{}{}
			}
			for id := range want {
				if _, ok := got[id]; !ok {
					t.Errorf("matchingIDs is missing id %d (the damaged record's row is dropped; delete-by-filter would under-delete)", id)
				}
			}
			if len(got) != len(want) {
				t.Errorf("matchingIDs returned %d ids, want %d", len(got), len(want))
			}

			res, err := h.SearchFiltered([]float32{float32(badID), float32(badID % 7), 0, 1}, len(want), f)
			if err != nil {
				t.Fatalf("SearchFiltered: %v", err)
			}
			if !eqUint64(resultIDs(res), bruteForceFiltered(corpus, metas, []float32{float32(badID), float32(badID % 7), 0, 1}, len(want), pred)) {
				t.Errorf("SearchFiltered ids %v != brute force", resultIDs(res))
			}

			// The mechanism, asserted directly: while the key is poisoned the
			// planner must DECLINE, and no gate may be graded exact.
			func() {
				h.mu.RLock()
				defer h.mu.RUnlock()
				limit := h.effectiveFilterFirstLimit(h.arena.Size())
				if cands, ok := h.payloadIdx.candidates(f, limit); ok {
					t.Errorf("the index narrowed to %d candidates while the payload key holds an unindexable record", len(cands))
				}
				if _, ok := h.payloadIdx.eqSet("torn/a", NewInt(5)); ok {
					t.Error("eqSet answered for a path under a poisoned payload key")
				}
				// The column sidecar is built FROM the posting map, so it
				// inherits the same hole and must decline the same way.
				rangeF := Filter{Op: FilterGt, Field: "torn/a", Value: NewInt(3)}
				if _, ok := h.payloadIdx.collectColumnTerms(rangeF, h.arena.Capacity(), -1, nil); ok {
					t.Error("a range under a poisoned payload key is column-expressible — the column would be built from postings that are missing the damaged slot")
				}
			}()
			if armed, exact := gateExactFor(t, h, f); armed && exact {
				t.Error("an EXACT gate armed over a poisoned payload key — it would skip the predicate re-check and lose the row")
			}

			// Replacing the damaged record un-poisons the key: acceleration
			// must come back, and still agree with the predicate.
			good := Metadata{"torn": NewRecord(goodDynamicRecord(t, 5))}
			if _, _, _, err := h.SetPayload(badID, good, nil, CASCond{}); err != nil {
				t.Fatalf("SetPayload: %v", err)
			}
			metas[badID] = good
			func() {
				h.mu.RLock()
				defer h.mu.RUnlock()
				limit := h.effectiveFilterFirstLimit(h.arena.Size())
				cands, ok := h.payloadIdx.candidates(f, limit)
				if !ok {
					t.Fatal("the index still declines after the damaged record was replaced — fail-closed latched instead of tracking state")
				}
				if len(cands) == 0 || len(cands) >= len(metas) {
					t.Fatalf("candidate set of %d over %d points is not a proper narrowing", len(cands), len(metas))
				}
			}()
			gotIDs, err = h.matchingIDs(f, compileOrFail(t, f))
			if err != nil {
				t.Fatalf("matchingIDs after repair: %v", err)
			}
			want = bruteMatchIDs(metas, compileOrFail(t, f))
			if len(gotIDs) != len(want) {
				t.Errorf("after repair matchingIDs returned %d ids, want %d", len(gotIDs), len(want))
			}
		})
	}
}

// TestRecordIndexSelectsEverySlotLookupPathResolves is the consistency
// invariant stated as a test rather than as an argument: for every indexed
// shape the index claims it can narrow, every live slot whose lookupPath
// resolves to a MATCHING value must appear in the candidate set. Under-
// selection is the failure mode that loses rows, so it is checked directly,
// slot by slot, over a corpus that deliberately mixes intact records, a
// damaged one, records under a names-less schema, non-record payload values
// and absent keys.
func TestRecordIndexSelectsEverySlotLookupPathResolves(t *testing.T) {
	h, _, _ := recordCorpus(t, 500, 8)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	now := uint64(h.now())

	filters := []Filter{
		{Op: FilterEq, Field: "session/rc", Value: NewInt(5)},
		{Op: FilterIn, Field: "session/rc", Value: NewInts([]int64{3, 7})},
		{Op: FilterGt, Field: "session/rc", Value: NewInt(10)},
		{Op: FilterEq, Field: "session/tag", Value: NewString("de")},
		{Op: FilterEq, Field: "session/b#count", Value: NewInt(2)},
		{Op: FilterEq, Field: "session/#0", Value: NewInt(5)},
		{Op: FilterEq, Field: "torn/a", Value: NewInt(5)},
		{Op: FilterEq, Field: "nameless/#0", Value: NewInt(3)},
	}
	for _, f := range filters {
		pred := compileOrFail(t, f)
		cands, ok := h.payloadIdx.candidates(f, limit)
		if !ok {
			continue // declined: the predicate answers, which the brute-force test covers
		}
		in := make(map[uint32]struct{}, len(cands))
		for _, s := range cands {
			in[s] = struct{}{}
		}
		matched := 0
		for slot := 0; slot < h.arena.Capacity(); slot++ {
			u := uint32(slot) //nolint:gosec // bounded by capacity
			if h.tombstoned[u] || h.isExpiredAt(u, now) {
				continue
			}
			if !pred(h.liveMeta(u, now)) {
				continue
			}
			matched++
			if _, ok := in[u]; !ok {
				t.Errorf("%+v: slot %d resolves to a matching value through lookupPath but is not in the candidate set", f, slot)
			}
		}
		if matched == 0 {
			t.Errorf("%+v: narrowed but no slot matches — the check is vacuous", f)
		}
	}
}

// TestRecordPoisonSurvivesRestoreAndMirrorsToTheIDIndex covers the two places
// the fail-closed state could quietly go missing: a restored collection (whose
// index is REBUILT from metadata rather than maintained incrementally) and the
// id-keyed index used by named vectors and multivectors, which has its own
// reindex and its own counters.
func TestRecordPoisonSurvivesRestoreAndMirrorsToTheIDIndex(t *testing.T) {
	t.Run("restore", func(t *testing.T) {
		src, _, _ := poisonCorpus(t, 40, 7, tornDynamicRecord(t))
		var buf bytes.Buffer
		if err := src.Snapshot(&buf); err != nil {
			t.Fatal(err)
		}
		dst, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		if err := dst.Restore(&buf); err != nil {
			t.Fatal(err)
		}
		dst.mu.RLock()
		defer dst.mu.RUnlock()
		if !dst.payloadIdx.badRecords.poisoned("torn") {
			t.Fatal("the restored index does not know the payload key holds an unindexable record — rebuild lost the fail-closed state")
		}
		f := Filter{Op: FilterEq, Field: "torn/a", Value: NewInt(5)}
		if cands, ok := dst.payloadIdx.candidates(f, dst.effectiveFilterFirstLimit(dst.arena.Size())); ok {
			t.Errorf("the restored index narrowed to %d candidates under a poisoned payload key", len(cands))
		}
	})

	t.Run("id-keyed mirror", func(t *testing.T) {
		p := newPayloadIndexID()
		p.reindex(1, Metadata{"torn": NewRecord(goodDynamicRecord(t, 5))})
		p.reindex(2, Metadata{"torn": NewRecord(tornDynamicRecord(t))})
		if !p.badRecords.poisoned("torn") {
			t.Fatal("the id index did not poison the payload key")
		}
		if _, ok := p.eqSet("torn/a", NewInt(5)); ok {
			t.Error("the id index answered eqSet under a poisoned payload key")
		}
		// Repairing id 2 clears the key and acceleration returns.
		p.reindex(2, Metadata{"torn": NewRecord(goodDynamicRecord(t, 5))})
		if p.badRecords.poisoned("torn") {
			t.Fatal("the id index stayed poisoned after the damaged record was replaced")
		}
		set, ok := p.eqSet("torn/a", NewInt(5))
		if !ok {
			t.Fatal("the id index still declines after repair")
		}
		if len(set) != 2 {
			t.Fatalf("eqSet(torn/a == 5) holds %d ids, want 2", len(set))
		}
		// Clearing the payload of the only bad id is the other way out.
		p.reindex(2, Metadata{"torn": NewRecord(tornDynamicRecord(t))})
		p.reindex(2, nil)
		if p.badRecords.poisoned("torn") {
			t.Fatal("clearing the payload left the key poisoned")
		}
		if len(p.badIDKeys) != 0 {
			t.Errorf("badIDKeys leaked: %v", p.badIDKeys)
		}
	})
}
