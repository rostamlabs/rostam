// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// A RECORD STORED UNDER $content.
//
// contentField is the one metadata key reindex refuses to index, and it skips
// the entry BEFORE the ValueRecord branch — so a payload value of kind record
// stored under "$content" contributes no synthetic postings at all, not even
// for its top-level scalar fields. The compiled predicate reads it perfectly
// well: lookupPath splits "$content/rc" like any other field string and
// resolves into the record.
//
// indexNarrowable used to special-case only the BARE "$content" string, so
// "$content/rc" split, parsed, and satisfied both recordShapeIndexed and
// recordOpIndexed. The planner then graded a permanently empty posting set as
// EXACT, filter-first brute-forced zero candidates, and every matching row was
// dropped — silently, and without a predicate re-check to catch it. Nothing
// reserves "$content": handleVectorUpsert takes it as an ordinary metadata key
// straight off the wire, so this is wire-reachable rather than theoretical.
//
// These tests pin the fix (indexNarrowable declines any field whose SPLIT
// payload key is contentField) at both levels: the planner predicate itself,
// and the answers the two consumers give against a brute-force oracle.

// contentRecordCorpus builds an index whose points carry a session record under
// $content — the shape that used to narrow to empty — plus an ordinary literal
// field so the "declining must not disable the siblings" case is testable too.
func contentRecordCorpus(t *testing.T, n int) (*hnsw, map[uint64][]float32, map[uint64]Metadata) {
	t.Helper()
	const dim = 8
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(20260909))
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		m := Metadata{"tag": NewString("a")}
		// rc cycles 0..12, so an eq on 7 has both true and false points, and
		// the last few points carry no record at all.
		if i <= n-5 {
			m[contentField] = NewRecord(sessionRecordBytesRC(t, uint8(i%13))) //nolint:gosec // i%13 fits uint8
		}
		corpus[id] = v
		metas[id] = m
		if _, _, err := h.Insert(id, v, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	return h, corpus, metas
}

// TestContentRecordPathIsNeverIndexNarrowable pins the fix at the planner's own
// level, so a refactor that reintroduces the hole fails here with a precise
// message rather than downstream as a count mismatch.
func TestContentRecordPathIsNeverIndexNarrowable(t *testing.T) {
	paths := []string{
		contentField + "/rc",      // an indexed SHAPE, the dangerous case
		contentField + "/b#count", // the other indexed shape
		contentField + "/b/42",    // a row
		contentField + "/b/42/hi", // a cell
		contentField + "/#0",      // positional
	}
	ops := []FilterOp{FilterEq, FilterIn, FilterGt, FilterGte, FilterLt, FilterContains, FilterMatch}
	for _, field := range paths {
		for _, op := range ops {
			if indexNarrowable(field, op, nil) {
				t.Errorf("indexNarrowable(%q, %v) = true — reindex writes no posting for any path under $content", field, op)
			}
		}
	}
	// The guard must be about $content specifically, not about record paths in
	// general: an ordinary record path with an indexed shape still narrows.
	if !indexNarrowable("session/rc", FilterEq, nil) {
		t.Error("an ordinary record path stopped narrowing — the fix over-reached")
	}

	// filterIndexExact and columnExpressible route through the same helper, so
	// the exact-gate and the column planner must inherit the decline. A leaf
	// the gate calls EXACT is one whose empty set is taken as proof.
	f := Filter{Op: FilterEq, Field: contentField + "/rc", Value: NewInt(7)}
	if filterIndexExact(f, nil) {
		t.Error("filterIndexExact graded a $content record path EXACT — an empty posting set would read as proven-empty")
	}
	if columnExpressible(f, nil) {
		t.Error("columnExpressible accepted a $content record path")
	}
}

// TestContentRecordPathFilterFirstMatchesBruteForce drives the two consumers
// that consult the planner — SearchFiltered (filter-first KNN) and matchingIDs
// (the selection delete-by-filter and scroll perform) — against a brute-force
// evaluation of the SAME compiled predicate. A regression here is a wrong
// ANSWER, not a slow one.
func TestContentRecordPathFilterFirstMatchesBruteForce(t *testing.T) {
	const (
		n   = 60
		dim = 8
		k   = 12
	)
	h, corpus, metas := contentRecordCorpus(t, n)
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(j) / 10
	}

	filters := []Filter{
		{Op: FilterEq, Field: contentField + "/rc", Value: NewInt(7)},
		{Op: FilterGt, Field: contentField + "/rc", Value: NewInt(10)},
		{Op: FilterIn, Field: contentField + "/rc", Value: NewInts([]int64{3, 7})},
		{Op: FilterGte, Field: contentField + "/b#count", Value: NewInt(1)},
		{Op: FilterRowExists, Field: contentField + "/b/42"},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "tag", Value: NewString("a")},
			{Op: FilterEq, Field: contentField + "/rc", Value: NewInt(7)},
		}},
	}

	nonEmpty := false
	for _, f := range filters {
		pred := compileOrFail(t, f)

		got, err := h.SearchFiltered(q, k, f)
		if err != nil {
			t.Fatalf("filter %+v: SearchFiltered: %v", f, err)
		}
		want := bruteForceFiltered(corpus, metas, q, k, pred)
		if !eqUint64(resultIDs(got), want) {
			t.Errorf("filter %+v: SearchFiltered ids %v != brute-force %v", f, resultIDs(got), want)
		}

		gotIDs, err := h.matchingIDs(f, pred)
		if err != nil {
			t.Fatalf("filter %+v: matchingIDs: %v", f, err)
		}
		wantSet := bruteMatchIDs(metas, pred)
		if len(wantSet) > 0 {
			nonEmpty = true
		}
		gotSet := make(map[uint64]struct{}, len(gotIDs))
		for _, id := range gotIDs {
			gotSet[id] = struct{}{}
		}
		if len(gotSet) != len(wantSet) {
			t.Errorf("filter %+v: matchingIDs set size %d != brute-force %d", f, len(gotSet), len(wantSet))
			continue
		}
		for id := range wantSet {
			if _, ok := gotSet[id]; !ok {
				t.Errorf("filter %+v: matchingIDs missing true match id %d (delete-by-filter would under-delete)", f, id)
			}
		}
	}
	if !nonEmpty {
		t.Fatal("every filter's brute-force match set was empty; test proves nothing — fixture is broken")
	}
}

// TestContentRecordPathCandidatesDecline pins the posting-set level: every
// narrowing entry point must DECLINE (ok=false) for a path under $content
// rather than answer the empty set, which reads as "no matches" to the planner
// and the exact gate alike.
func TestContentRecordPathCandidatesDecline(t *testing.T) {
	h, _, _ := contentRecordCorpus(t, 30)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	field := contentField + "/rc"

	if _, ok := h.payloadIdx.eqSet(field, NewInt(7)); ok {
		t.Error("eqSet answered for a $content record path instead of declining")
	}
	if _, ok := h.payloadIdx.inSet(field, NewInts([]int64{7}), limit); ok {
		t.Error("inSet answered for a $content record path instead of declining")
	}
	if _, ok := h.payloadIdx.orderingSet(field, FilterGt, NewInt(1), limit); ok {
		t.Error("orderingSet answered for a $content record path instead of declining")
	}
	if _, ok := collectEqTerms(Filter{Op: FilterEq, Field: field, Value: NewInt(7)}, nil); ok {
		t.Error("collectEqTerms accepted a $content record path")
	}
	if _, ok := h.payloadIdx.candidates(Filter{Op: FilterEq, Field: field, Value: NewInt(7)}, limit); ok {
		t.Error("candidates narrowed a $content record path — its posting set is permanently empty")
	}
	// The sibling conjunct must still narrow: declining is per leaf.
	if _, ok := h.payloadIdx.candidates(Filter{Op: FilterEq, Field: "tag", Value: NewString("a")}, limit); !ok {
		t.Error("an ordinary field stopped narrowing — the fix over-reached")
	}
}

// TestContentRecordPathIDIndexDeclines is the named/multivector mirror. The
// id-keyed payload index skips contentField in the same place for the same
// reason, so it had the same hole; no shipped named/MV write path routes a
// record into $content today, but "$content" is just a map key a caller can
// set and the failure mode is silent wrong results.
func TestContentRecordPathIDIndexDeclines(t *testing.T) {
	p := newPayloadIndexID()
	for id := uint64(1); id <= 20; id++ {
		p.reindex(id, Metadata{
			"tag":        NewString("a"),
			contentField: NewRecord(sessionRecordBytesRC(t, uint8(id%13))), //nolint:gosec // id%13 fits uint8
		})
	}
	const limit = 1000
	field := contentField + "/rc"
	if _, ok := p.eqSet(field, NewInt(7)); ok {
		t.Error("id eqSet answered for a $content record path")
	}
	if _, ok := p.inSet(field, NewInts([]int64{7}), limit); ok {
		t.Error("id inSet answered for a $content record path")
	}
	if _, ok := p.orderingSet(field, FilterGt, NewInt(1), limit); ok {
		t.Error("id orderingSet answered for a $content record path")
	}
	if _, ok := p.candidatesCapped(Filter{Op: FilterEq, Field: field, Value: NewInt(7)}, limit, limit); ok {
		t.Error("the id index narrowed a $content record path")
	}
	if _, ok := p.candidatesCapped(Filter{Op: FilterEq, Field: "tag", Value: NewString("a")}, limit, limit); !ok {
		t.Error("an ordinary field stopped narrowing on the id index")
	}
}
