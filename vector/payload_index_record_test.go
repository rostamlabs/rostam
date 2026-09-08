// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// recordSlotIn reports whether slot carries a posting for (field, key) in the
// dense payload index — the assertion every synthetic-field test makes.
func recordSlotIn(p *payloadIndex, field string, key scalarKey, slot uint32) bool {
	vals := p.fields[field]
	if vals == nil {
		return false
	}
	set := vals[key]
	if set == nil {
		return false
	}
	_, ok := set[slot]
	return ok
}

func intKey(i int64) scalarKey  { return scalarKey{kind: ValueInt, i: i} }
func strKey(s string) scalarKey { return scalarKey{kind: ValueString, str: s} }

// TestReindexRecordSyntheticFields pins the indexing half of the contract: a
// ValueRecord payload key posts one synthetic eq entry per top-level scalar
// field and per table "#count", under "<payloadKey>/<field>", and NOTHING for
// the shapes the planner refuses to accelerate (a table cell, a positional
// "#N", the table field itself). It also pins the reverse-map discipline —
// rewriting the record drops the old postings, clearing the payload drops all
// of them — and the exact-key precedence lookupPath applies: a LITERAL payload
// key spelled "session/rc" wins over the record's rc, on both sides.
func TestReindexRecordSyntheticFields(t *testing.T) {
	p := newPayloadIndex()
	rec := sessionRecordBytes(t) // rc=7 bc=3 hist=1234 bal=-5 tag="de" b={42,99}
	p.reindex(3, Metadata{"session": NewRecord(rec)})

	present := []struct {
		field string
		key   scalarKey
	}{
		{"session/rc", intKey(7)},
		{"session/bc", intKey(3)},
		{"session/hist", intKey(1234)},
		{"session/bal", intKey(-5)},
		{"session/tag", strKey("de")},
		{"session/b#count", intKey(2)},
	}
	for _, c := range present {
		if !recordSlotIn(p, c.field, c.key, 3) {
			t.Errorf("fields[%q][%+v] is missing slot 3 — the synthetic field was not indexed", c.field, c.key)
		}
	}
	// posts must count each (slot, field) ONCE, or fieldTotalNumeric's
	// complement-gate proof (posts >= liveCount ⇒ every live slot covered)
	// stops being an upper bound on totality.
	if got := p.posts["session/rc"].num; got != 1 {
		t.Errorf("posts[session/rc].num = %d, want 1 (one slot, one key)", got)
	}
	if got := p.posts["session/tag"].str; got != 1 {
		t.Errorf("posts[session/tag].str = %d, want 1", got)
	}

	// Shapes this phase deliberately does not post: a table cell, a table row,
	// the table field itself, a positional field (IndexEntries names a
	// names-carrying schema's fields by NAME, so "#0" has no posting and the
	// planner must not treat its absence as proof), and "#count" on a scalar.
	for _, field := range []string{"session/b/42/hi", "session/b/42", "session/b", "session/#0", "session/rc#count", "session/zz"} {
		if len(p.fields[field]) != 0 {
			t.Errorf("fields[%q] has postings %v — this shape must not be indexed", field, p.fields[field])
		}
	}

	// Rewriting the record (SetPayload) drops the old key and adds the new.
	p.reindex(3, Metadata{"session": NewRecord(sessionRecordBytesRC(t, 8))})
	if recordSlotIn(p, "session/rc", intKey(7), 3) {
		t.Error("fields[session/rc][7] still holds slot 3 after the record was rewritten to rc=8 — stale posting")
	}
	if !recordSlotIn(p, "session/rc", intKey(8), 3) {
		t.Error("fields[session/rc][8] is missing slot 3 after the rewrite")
	}

	// Clearing the payload drops every synthetic posting and its reverse entry.
	p.reindex(3, nil)
	for _, c := range present {
		if len(p.fields[c.field]) != 0 {
			t.Errorf("fields[%q] survived ClearPayload: %v", c.field, p.fields[c.field])
		}
	}
	if _, ok := p.slotKeys[3]; ok {
		t.Error("slotKeys[3] survived ClearPayload — the reverse map leaked")
	}
	if len(p.posts) != 0 {
		t.Errorf("posts survived ClearPayload: %v", p.posts)
	}

	// Exact-key precedence: a literal payload key spelled like a record path
	// shadows the record's field, and the index must post the LITERAL value
	// only — one key per (slot, field), the same one lookupPath resolves.
	q := newPayloadIndex()
	shadow := Metadata{"session": NewRecord(rec), "session/rc": NewInt(99)}
	q.reindex(5, shadow)
	if !recordSlotIn(q, "session/rc", intKey(99), 5) {
		t.Error("fields[session/rc][99] is missing slot 5 — the literal key was not indexed")
	}
	if recordSlotIn(q, "session/rc", intKey(7), 5) {
		t.Error("fields[session/rc][7] holds slot 5 — the record's rc was indexed even though a literal key shadows it")
	}
	if got := q.posts["session/rc"].num; got != 1 {
		t.Errorf("posts[session/rc].num = %d with a shadowing literal key, want 1", got)
	}
	if v, ok := lookupPath(shadow, "session/rc"); !ok || !v.Equal(NewInt(99)) {
		t.Fatalf("lookupPath disagrees with the index: got (%+v, %v), want (99, true)", v, ok)
	}
}

// TestReindexRecordSyntheticFieldsID mirrors the above for the id-keyed index
// (named vectors / multivector), which has its own reindex and would otherwise
// silently post nothing while indexNarrowable says record paths are provable.
func TestReindexRecordSyntheticFieldsID(t *testing.T) {
	p := newPayloadIndexID()
	p.reindex(11, Metadata{"session": NewRecord(sessionRecordBytes(t))})
	if _, ok := p.fields["session/rc"][intKey(7)][11]; !ok {
		t.Fatalf("id index: fields[session/rc][7] is missing id 11 (have %v)", p.fields["session/rc"])
	}
	if _, ok := p.fields["session/b#count"][intKey(2)][11]; !ok {
		t.Fatal("id index: fields[session/b#count][2] is missing id 11")
	}
	if len(p.fields["session/#0"]) != 0 {
		t.Error("id index: a positional shape must not be indexed")
	}
	p.reindex(11, nil)
	if len(p.fields["session/rc"]) != 0 {
		t.Errorf("id index: fields[session/rc] survived a payload clear: %v", p.fields["session/rc"])
	}
}

// recordCorpus builds the shared 500-point fixture the brute-force tests use:
// most points carry a "session" record whose rc varies, some carry a plain
// STRING under "session" (so a record path must reject a non-record payload
// key), some carry no session at all, some carry a LITERAL payload key spelled
// "session/rc" (which shadows the record's rc on both sides), and some carry a
// literal "a/b" that parses as a record path but names no record.
func recordCorpus(t *testing.T, n, dim int) (*hnsw, map[uint64][]float32, map[uint64]Metadata) {
	t.Helper()
	rng := rand.New(rand.NewSource(20260908))
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		m := Metadata{}
		if i%2 == 0 {
			m["country"] = NewString("DE")
		} else {
			m["country"] = NewString("US")
		}
		switch {
		case i%7 == 0:
			m["session"] = NewString("not a record at all")
		case i%11 == 0: // no session key at all
		default:
			m["session"] = NewRecord(sessionRecordBytesRC(t, uint8(i%13))) //nolint:gosec // i%13 fits uint8
		}
		if i%23 == 0 {
			m["session/rc"] = NewInt(99) // literal key; shadows the record's rc
		}
		if i%29 == 0 {
			m["a/b"] = NewString("lit") // literal key that parses as a record path
		}
		corpus[id] = v
		metas[id] = m
		if _, _, err := h.Insert(id, v, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	return h, corpus, metas
}

// recordFilterCases is the filter matrix both the search and the delete/scroll
// entry point are checked against. wantEmpty marks the one case whose
// brute-force match set is legitimately empty (contains over a scalar string
// can never hold), so every other case is asserted NON-empty and cannot pass
// vacuously.
type recordFilterCase struct {
	name      string
	f         Filter
	wantEmpty bool
}

func recordFilterCases() []recordFilterCase {
	and := func(name string, f Filter) recordFilterCase {
		return recordFilterCase{name: "and(country,+" + name + ")", f: Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "country", Value: NewString("DE")}, f,
		}}}
	}
	base := []recordFilterCase{
		{name: "eq scalar", f: Filter{Op: FilterEq, Field: "session/rc", Value: NewInt(5)}},
		{name: "in scalar", f: Filter{Op: FilterIn, Field: "session/rc", Value: NewInts([]int64{3, 7})}},
		{name: "gt scalar", f: Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(10)}},
		{name: "lte scalar", f: Filter{Op: FilterLte, Field: "session/rc", Value: NewInt(4)}},
		{name: "eq string scalar", f: Filter{Op: FilterEq, Field: "session/tag", Value: NewString("de")}},
		{name: "eq count", f: Filter{Op: FilterEq, Field: "session/b#count", Value: NewInt(2)}},
		{name: "gte count", f: Filter{Op: FilterGte, Field: "session/b#count", Value: NewInt(2)}},
		{name: "lt count", f: Filter{Op: FilterLt, Field: "session/b#count", Value: NewInt(3)}},
		{name: "eq positional", f: Filter{Op: FilterEq, Field: "session/#0", Value: NewInt(5)}},
		{name: "in positional", f: Filter{Op: FilterIn, Field: "session/#0", Value: NewInts([]int64{3, 7})}},
		{name: "gt positional", f: Filter{Op: FilterGt, Field: "session/#0", Value: NewInt(10)}},
		{name: "eq column path", f: Filter{Op: FilterEq, Field: "session/b/42/hi", Value: NewInt(500)}},
		{name: "gt column path", f: Filter{Op: FilterGt, Field: "session/b/42/hi", Value: NewInt(100)}},
		{name: "match record string", f: Filter{Op: FilterMatch, Field: "session/tag", Value: NewString("de")}},
		{name: "contains record string", f: Filter{Op: FilterContains, Field: "session/tag", Value: NewString("de")}, wantEmpty: true},
		{name: "row_exists", f: Filter{Op: FilterRowExists, Field: "session/b/42"}},
		{name: "row_absent", f: Filter{Op: FilterRowAbsent, Field: "session/b/7"}},
		{name: "eq literal shadowing key", f: Filter{Op: FilterEq, Field: "session/rc", Value: NewInt(99)}},
		{name: "eq literal path-shaped key", f: Filter{Op: FilterEq, Field: "a/b", Value: NewString("lit")}},
	}
	out := make([]recordFilterCase, 0, 2*len(base))
	for _, c := range base {
		out = append(out, c)
		conj := and(c.name, c.f)
		conj.wantEmpty = c.wantEmpty
		out = append(out, conj)
	}
	return out
}

// TestRecordFilterFirstEqualsBruteForce is the whole point of the task: for
// EVERY record-path shape and op — the ones the index now accelerates and the
// ones it still declines — the filter-first search path and the
// delete-by-filter / scroll selection must return exactly what a brute-force
// evaluation of the SAME compiled predicate returns. An over-narrowed posting
// set loses rows; an under-narrowed one returns rows the predicate rejects;
// both fail here.
func TestRecordFilterFirstEqualsBruteForce(t *testing.T) {
	const (
		n   = 500
		dim = 8
		k   = 12
	)
	h, corpus, metas := recordCorpus(t, n, dim)
	q := make([]float32, dim)
	rng := rand.New(rand.NewSource(7))
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	for _, c := range recordFilterCases() {
		t.Run(c.name, func(t *testing.T) {
			pred := compileOrFail(t, c.f)

			wantSet := bruteMatchIDs(metas, pred)
			if len(wantSet) == 0 != c.wantEmpty {
				t.Fatalf("brute-force match set size %d contradicts wantEmpty=%v — the fixture proves nothing", len(wantSet), c.wantEmpty)
			}

			// Consumer 1: filter-first KNN.
			got, err := h.SearchFiltered(q, k, c.f)
			if err != nil {
				t.Fatalf("SearchFiltered: %v", err)
			}
			want := bruteForceFiltered(corpus, metas, q, k, pred)
			if !eqUint64(resultIDs(got), want) {
				t.Errorf("SearchFiltered ids %v != brute-force %v", resultIDs(got), want)
			}

			// Consumer 2: the id selection matchingIDsAt performs for
			// delete-by-filter and non-vector scroll.
			gotIDs, err := h.matchingIDs(c.f, pred)
			if err != nil {
				t.Fatalf("matchingIDs: %v", err)
			}
			gotSet := make(map[uint64]struct{}, len(gotIDs))
			for _, id := range gotIDs {
				gotSet[id] = struct{}{}
			}
			for id := range wantSet {
				if _, ok := gotSet[id]; !ok {
					t.Errorf("matchingIDs is missing true match id %d (delete-by-filter would under-delete)", id)
				}
			}
			for id := range gotSet {
				if _, ok := wantSet[id]; !ok {
					t.Errorf("matchingIDs returned id %d the predicate rejects (delete-by-filter would over-delete)", id)
				}
			}
		})
	}
}

// TestRecordColumnFastPathEqualsPredicate drives the numeric column sidecar
// over a record path: a range on "session/rc" must be column-expressible,
// build through ensureColumn from the synthetic postings, and agree with the
// predicate SLOT BY SLOT — the comparison the search-level test can only
// sample.
func TestRecordColumnFastPathEqualsPredicate(t *testing.T) {
	h, _, _ := recordCorpus(t, 400, 8)
	h.mu.RLock()
	defer h.mu.RUnlock()
	now := uint64(h.now())
	capacity := h.arena.Capacity()

	cases := []struct {
		name string
		f    Filter
	}{
		{"gt rc", Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(5)}},
		{"lte rc", Filter{Op: FilterLte, Field: "session/rc", Value: NewInt(4)}},
		{"gte count", Filter{Op: FilterGte, Field: "session/b#count", Value: NewInt(2)}},
		{"and rc + plain", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGt, Field: "session/rc", Value: NewInt(3)},
			{Op: FilterLt, Field: "session/hist", Value: NewInt(2000)},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pred := compileOrFail(t, tc.f)
			terms, ok := h.payloadIdx.collectColumnTerms(tc.f, capacity, -1, nil)
			if !ok {
				t.Fatalf("%s: not column-expressible — the record path never reached ensureColumn", tc.name)
			}
			var g admitGate
			g.armColumns(terms)
			checked, accepted := 0, 0
			for slot := 0; slot < capacity; slot++ {
				u := uint32(slot) //nolint:gosec // bounded by capacity
				if h.tombstoned[u] || h.isExpiredAt(u, now) {
					continue
				}
				want := pred(h.liveMeta(u, now))
				got := g.testCols(u)
				checked++
				if want {
					accepted++
				}
				if got != want {
					t.Fatalf("slot %d: column=%v predicate=%v (meta %v)", slot, got, want, h.liveMeta(u, now))
				}
			}
			if checked == 0 || accepted == 0 || accepted == checked {
				t.Fatalf("%d/%d accepted — the comparison is degenerate", accepted, checked)
			}
		})
	}
}

// TestRecordIndexRebuildAfterRestore proves the synthetic postings are rebuilt
// from metadata on Restore (rebuild -> reindex per slot), not merely maintained
// incrementally on the insert path — a restored collection that lost them would
// answer every record-path filter with the empty set.
func TestRecordIndexRebuildAfterRestore(t *testing.T) {
	src, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	want := map[uint64]bool{}
	metas := make(map[uint64]Metadata, 40)
	for i := 1; i <= 40; i++ {
		rc := uint8(i % 4) //nolint:gosec // bounded by 4
		m := Metadata{"session": NewRecord(sessionRecordBytesRC(t, rc))}
		if rc == 2 {
			want[uint64(i)] = true
		}
		metas[uint64(i)] = m
		if _, _, err := src.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	dst, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatal(err)
	}
	if len(dst.payloadIdx.fields["session/rc"][intKey(2)]) != len(want) {
		t.Fatalf("restored fields[session/rc][2] holds %d slots, want %d — rebuild did not reconstruct the synthetic postings",
			len(dst.payloadIdx.fields["session/rc"][intKey(2)]), len(want))
	}
	if len(dst.payloadIdx.fields["session/b#count"][intKey(2)]) != 40 {
		t.Fatalf("restored fields[session/b#count][2] holds %d slots, want 40",
			len(dst.payloadIdx.fields["session/b#count"][intKey(2)]))
	}
	f := Filter{Op: FilterEq, Field: "session/rc", Value: NewInt(2)}
	got, err := dst.SearchFiltered([]float32{20, 0, 0, 0}, 40, f)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("restored search returned %d ids, want %d", len(got), len(want))
	}
	for _, r := range got {
		if !want[r.ID] {
			t.Fatalf("restored search returned non-matching id %d", r.ID)
		}
	}
}

// TestRecordFilterFirstActuallyNarrows is what keeps
// TestRecordFilterFirstEqualsBruteForce honest. Equality between filter-first
// and brute force is trivially true when the planner declines EVERYTHING and
// falls back to graph traversal — which is exactly what Task 5 did and exactly
// what this task changes. So: the accelerated shapes must produce a real,
// PROPER-SUBSET candidate set from the index, and the declined ones must still
// decline outright rather than answer with an empty posting map.
func TestRecordFilterFirstActuallyNarrows(t *testing.T) {
	h, _, metas := recordCorpus(t, 500, 8)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())

	narrowing := []struct {
		name string
		f    Filter
	}{
		{"eq scalar", Filter{Op: FilterEq, Field: "session/rc", Value: NewInt(5)}},
		{"in scalar", Filter{Op: FilterIn, Field: "session/rc", Value: NewInts([]int64{3, 7})}},
		{"gt scalar", Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(10)}},
		{"eq string scalar", Filter{Op: FilterEq, Field: "session/tag", Value: NewString("de")}},
		{"eq count", Filter{Op: FilterEq, Field: "session/b#count", Value: NewInt(2)}},
	}
	for _, c := range narrowing {
		cands, ok := h.payloadIdx.candidates(c.f, limit)
		if !ok {
			t.Errorf("%s: the index declined to narrow — the acceleration this task adds is not being exercised", c.name)
			continue
		}
		if len(cands) == 0 || len(cands) >= len(metas) {
			t.Errorf("%s: candidate set of %d over a %d-point corpus is not a proper narrowing", c.name, len(cands), len(metas))
		}
	}

	// The shapes that stay on the predicate must DECLINE (ok=false). Answering
	// with the empty sentinel is the Task 5 bug: it reads as "no matches" to
	// the planner and silently returns nothing.
	declining := []struct {
		name string
		f    Filter
	}{
		{"positional", Filter{Op: FilterEq, Field: "session/#0", Value: NewInt(5)}},
		{"column path", Filter{Op: FilterEq, Field: "session/b/42/hi", Value: NewInt(500)}},
		{"match on a record string", Filter{Op: FilterMatch, Field: "session/tag", Value: NewString("de")}},
		{"contains on a record string", Filter{Op: FilterContains, Field: "session/tag", Value: NewString("de")}},
		{"row_exists", Filter{Op: FilterRowExists, Field: "session/b/42"}},
	}
	for _, c := range declining {
		if cands, ok := h.payloadIdx.candidates(c.f, limit); ok {
			t.Errorf("%s: the index narrowed to %d candidates for a shape it never posts — a wrong answer waiting to happen", c.name, len(cands))
		}
	}

	// The per-entry-point guards, at the level the planner consumes them.
	if _, ok := h.payloadIdx.matchSet("session/tag", NewString("de"), limit); ok {
		t.Error("matchSet answered for a record path: record strings get no token postings")
	}
	if _, ok := h.payloadIdx.containsSet("session/tag", NewString("de"), limit); ok {
		t.Error("containsSet answered for a record path: record strings get no contains postings")
	}
	if _, ok := h.payloadIdx.eqSet("session/#0", NewInt(5)); ok {
		t.Error("eqSet answered for a positional record path, which IndexEntries does not name for a names-carrying schema")
	}
	if _, ok := h.payloadIdx.eqSet("session/rc", NewInt(5)); !ok {
		t.Error("eqSet declined an indexed record path")
	}
}

// recordGateCorpus is recordCorpus's all-records sibling: EVERY point carries a
// session record, so posts[field].num reaches arena.Size() and the complement
// gate's totality precondition (fieldTotalNumeric) can actually hold for a
// synthetic field. Without that, the complement path silently declines and a
// test over it proves nothing.
func recordGateCorpus(t *testing.T, n int) *hnsw {
	t.Helper()
	rng := rand.New(rand.NewSource(4242))
	h, err := newHNSW(Config{Dim: 8, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= n; i++ {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		m := Metadata{"session": NewRecord(sessionRecordBytesRC(t, uint8(i%13)))} //nolint:gosec // i%13 fits uint8
		if i%2 == 0 {
			m["country"] = NewString("DE")
		} else {
			m["country"] = NewString("US")
		}
		if _, _, err := h.Insert(uint64(i), v, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	return h
}

// TestRecordAdmitGateMatchesPredicate drives the traversal-time admission gate
// — the ONE consumer that is allowed to skip the predicate re-check — over
// record paths. Two properties, and the second is the whole reason the
// exactness invariant has to hold rather than merely be plausible:
//
//   - SAFETY (every gate): a slot the predicate accepts must never be rejected
//     by the gate. A gate narrower than the predicate loses rows silently.
//   - EXACTNESS (a gate graded exact): the gate must equal the predicate slot
//     for slot, because nothing downstream will re-check it.
func TestRecordAdmitGateMatchesPredicate(t *testing.T) {
	h := recordGateCorpus(t, 400)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	now := uint64(h.now())

	cases := []struct {
		name      string
		f         Filter
		wantExact bool
	}{
		{"eq scalar", Filter{Op: FilterEq, Field: "session/rc", Value: NewInt(5)}, true},
		{"in scalar", Filter{Op: FilterIn, Field: "session/rc", Value: NewInts([]int64{3, 7})}, true},
		{"gt scalar", Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(6)}, true},
		{"eq string scalar", Filter{Op: FilterEq, Field: "session/tag", Value: NewString("de")}, true},
		{"eq count", Filter{Op: FilterEq, Field: "session/b#count", Value: NewInt(2)}, true},
		{"and plain + record", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "country", Value: NewString("DE")},
			{Op: FilterEq, Field: "session/rc", Value: NewInt(5)},
		}}, true},
	}

	prev := admitGateMode
	defer func() { admitGateMode = prev }()

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pred := compileOrFail(t, tc.f)
			admitGateMode = gateForce
			s := getLayerScratch()
			defer layerScratchPool.Put(s)
			plan, ok := h.payloadIdx.collectNarrowSets(tc.f, limit)
			if !ok {
				t.Fatalf("%s: not narrowable — the gate is not being exercised", tc.name)
			}
			h.buildAdmitGate(s, tc.f, plan, 10)
			defer s.gate.disable()
			if !s.gate.active() {
				t.Fatalf("%s: no gate armed", tc.name)
			}
			if s.gate.exact != tc.wantExact {
				t.Fatalf("%s: gate exact = %v, want %v", tc.name, s.gate.exact, tc.wantExact)
			}
			accepted, admitted := 0, 0
			for slot := 0; slot < h.arena.Capacity(); slot++ {
				u := uint32(slot) //nolint:gosec // bounded by capacity
				if h.tombstoned[u] || h.isExpiredAt(u, now) {
					continue
				}
				want := pred(h.liveMeta(u, now))
				got := s.gate.test(u)
				if want {
					accepted++
					if !got {
						t.Fatalf("%s slot %d: the predicate accepts it but the gate rejects it — rows would be lost", tc.name, slot)
					}
				}
				if got {
					admitted++
					if s.gate.exact && !want {
						t.Fatalf("%s slot %d: an EXACT gate admits a slot the predicate rejects, and nothing re-checks it", tc.name, slot)
					}
				}
			}
			if accepted == 0 || admitted == 0 {
				t.Fatalf("%s: %d accepted / %d admitted — the comparison is degenerate", tc.name, accepted, admitted)
			}
		})
	}

	// The COMPLEMENT gate marks what a range REJECTS and admits everything else,
	// which is only sound when the index knows about every live slot. Every
	// point here carries the record, so a synthetic field's numeric postings
	// cover the whole arena and the precondition genuinely holds.
	t.Run("complement over a synthetic field", func(t *testing.T) {
		if !h.payloadIdx.fieldTotalNumeric("session/rc", h.arena.Size()) {
			t.Fatal("session/rc is not numerically total over an all-records corpus — the synthetic postings do not cover every slot")
		}
		f := Filter{Op: FilterGte, Field: "session/rc", Value: NewInt(6)}
		pred := compileOrFail(t, f)
		admitGateMode = gateForceComplement
		s := getLayerScratch()
		defer layerScratchPool.Put(s)
		h.buildComplementGate(s, f, 10)
		defer s.gate.disable()
		if !s.gate.active() {
			t.Fatal("no complement gate armed over a total synthetic field")
		}
		accepted := 0
		for slot := 0; slot < h.arena.Capacity(); slot++ {
			u := uint32(slot) //nolint:gosec // bounded by capacity
			if h.tombstoned[u] || h.isExpiredAt(u, now) {
				continue
			}
			want := pred(h.liveMeta(u, now))
			if want {
				accepted++
			}
			if got := s.gate.test(u); got != want {
				t.Fatalf("slot %d: complement gate=%v predicate=%v", slot, got, want)
			}
		}
		if accepted == 0 {
			t.Fatal("no slot satisfies the range — the comparison is degenerate")
		}
	})
}
