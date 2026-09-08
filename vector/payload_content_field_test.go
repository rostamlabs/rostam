// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// $content AND THE PAYLOAD INDEX.
//
// contentField ($content) is the one metadata key the payload index refuses to
// index: reindex skips it, because a document body is not a filter key and
// equality-indexing megabytes of text would be absurd. The compiled predicate,
// however, reads it perfectly well — content is stored as an ordinary entry in
// the slot's metadata map, and lookupPath finds it like any other key. Only the
// RETURN path strips it (fetchDocs / docForLocked).
//
// That asymmetry used to be a silent wrong-results bug. Every narrowing path
// answers a missing field with the empty set, on the sound reasoning that a
// field carrying no postings can have no matches — but for $content the postings
// are absent because nothing ever tried to write them. The planner took the
// empty set as a candidate superset, brute-forced zero candidates, and returned
// nothing for filters that every row satisfied.
//
// These tests pin the fix (indexNarrowable: $content is never index-narrowable,
// so such a filter falls back to graph traversal + predicate) against the ONLY
// oracle that matters — the compiled predicate itself.

// contentCorpus builds an index where every point carries the same $content and
// the same tag, so a filter that reads $content correctly must return them ALL,
// and a filter that reads it through the index used to return NONE. The gap
// between those two answers is the whole test.
func contentCorpus(t testing.TB, n int) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{Dim: 8, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 0; i < n; i++ {
		v := make([]float32, 8)
		v[i%8] = float32(i)
		meta := withContent(Metadata{"tag": NewString("a")}, "the quick brown fox")
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	return h
}

// predicateMatchCount is the oracle: how many live slots the COMPILED PREDICATE
// accepts. Every assertion below compares the search against this rather than
// against a hard-coded number, so a test that drifts out of sync with the corpus
// fails loudly instead of quietly agreeing with a broken search.
func predicateMatchCount(t testing.TB, h *hnsw, f Filter) int {
	t.Helper()
	pred, err := CompileFilter(f)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	now := uint64(h.now())
	n := 0
	for slot := 0; slot < h.arena.Capacity(); slot++ {
		s := uint32(slot) //nolint:gosec // capacity fits uint32
		if h.tombstoned[s] || h.isExpiredAt(s, now) {
			continue
		}
		if pred(h.liveMeta(s, now)) {
			n++
		}
	}
	return n
}

// TestFilterFirstDoesNotDropContentFieldRows is the regression test for the
// reviewer's exact shape: And(Eq("tag","a"), Match("$content","quick")) over 50
// points that ALL match returned 0 results.
//
// The mechanism, for whoever hits this next: collectMatchSets asked matchSet for
// the $content token postings, matchSet found tokens["$content"] == nil and
// returned the empty sentinel with ok=true, the eq branch folded that empty set
// into the plan, and the intersection came back empty. Zero candidates is under
// any maxCand, so the planner chose filter-first, brute-forced nothing, and
// returned nothing. The predicate was never consulted.
func TestFilterFirstDoesNotDropContentFieldRows(t *testing.T) {
	const n = 50
	h := contentCorpus(t, n)
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "tag", Value: NewString("a")},
		{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
	}}

	if want := predicateMatchCount(t, h, f); want != n {
		t.Fatalf("precondition: the predicate should accept all %d points, got %d — "+
			"the corpus no longer exercises the bug", n, want)
	}
	res, err := h.SearchFiltered(make([]float32, 8), n, f)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res) != n {
		t.Fatalf("And(Eq, Match($content)) returned %d of %d matching points — the payload index "+
			"treated $content's empty posting set as a candidate superset", len(res), n)
	}
}

// TestContentFieldOpsAgreeWithThePredicate sweeps EVERY op that can name a
// field, because the hole was never Match-specific: it belonged to the
// empty-set-means-no-matches inference, which every narrowing path makes. Eq,
// In and the range family were wrong in exactly the same way and were not in the
// original report.
//
// Each case asserts the search agrees with the compiled predicate. Cases where
// the predicate legitimately matches nothing (Contains needs an array kind;
// $content is a string) are kept deliberately: they pin that the fix did not
// swing the other way and start over-returning.
func TestContentFieldOpsAgreeWithThePredicate(t *testing.T) {
	const n = 50
	h := contentCorpus(t, n)
	const body = "the quick brown fox"

	cases := []struct {
		name string
		f    Filter
	}{
		{"match", Filter{Op: FilterMatch, Field: contentField, Value: NewString("quick")}},
		{"match/multi-token", Filter{Op: FilterMatch, Field: contentField, Value: NewString("quick fox")}},
		{"eq", Filter{Op: FilterEq, Field: contentField, Value: NewString(body)}},
		{"ne", Filter{Op: FilterNe, Field: contentField, Value: NewString("other")}},
		{"in", Filter{Op: FilterIn, Field: contentField, Value: NewStrings([]string{body, "other"})}},
		{"contains", Filter{Op: FilterContains, Field: contentField, Value: NewString("quick")}},
		{"gt", Filter{Op: FilterGt, Field: contentField, Value: NewString("a")}},
		{"gte", Filter{Op: FilterGte, Field: contentField, Value: NewString(body)}},
		{"lt", Filter{Op: FilterLt, Field: contentField, Value: NewString("zzz")}},
		{"lte", Filter{Op: FilterLte, Field: contentField, Value: NewString(body)}},
		{"regex", Filter{Op: FilterRegex, Field: contentField, Value: NewString("^the quick")}},
		{"isnull", Filter{Op: FilterIsNull, Field: contentField}},
		{"isempty", Filter{Op: FilterIsEmpty, Field: contentField}},
		{"and/eq+match", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "tag", Value: NewString("a")},
			{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
		}}},
		{"and/eq+eq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "tag", Value: NewString("a")},
			{Op: FilterEq, Field: contentField, Value: NewString(body)},
		}}},
		{"and/eq+in", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "tag", Value: NewString("a")},
			{Op: FilterIn, Field: contentField, Value: NewStrings([]string{body})},
		}}},
		{"and/eq+range", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "tag", Value: NewString("a")},
			{Op: FilterGt, Field: contentField, Value: NewString("a")},
		}}},
		{"and/content-only-pair", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
			{Op: FilterEq, Field: contentField, Value: NewString(body)},
		}}},
		{"or/eq+content", Filter{Op: FilterOr, Or: []Filter{
			{Op: FilterEq, Field: "tag", Value: NewString("nope")},
			{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
		}}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := predicateMatchCount(t, h, tc.f)
			res, err := h.SearchFiltered(make([]float32, 8), n, tc.f)
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			if len(res) != want {
				t.Fatalf("returned %d points, predicate accepts %d", len(res), want)
			}
		})
	}
}

// TestContentFieldIsNeverIndexNarrowable pins the fix at its own level rather
// than only through search results, so a refactor that reintroduces a $content
// posting-set lookup fails here with a precise message instead of somewhere
// downstream with a count mismatch.
func TestContentFieldIsNeverIndexNarrowable(t *testing.T) {
	for _, op := range []FilterOp{FilterEq, FilterIn, FilterGt, FilterContains, FilterMatch} {
		if indexNarrowable(contentField, op) {
			t.Fatalf("contentField must never be index-narrowable (op %v) — reindex does not index it", op)
		}
		if !indexNarrowable("tag", op) {
			t.Fatalf("ordinary fields must stay narrowable (op %v)", op)
		}
	}

	h := contentCorpus(t, 30)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())

	// Every posting-set entry point must DECLINE (ok=false), not answer empty.
	// Returning (emptySet, true) is the bug: it reads as "no matches" to every
	// caller, and both the planner and the gate believe it.
	if _, ok := h.payloadIdx.eqSet(contentField, NewString("x")); ok {
		t.Error("eqSet answered for $content instead of declining")
	}
	if _, ok := h.payloadIdx.matchSet(contentField, NewString("quick"), limit); ok {
		t.Error("matchSet answered for $content instead of declining")
	}
	if _, ok := h.payloadIdx.containsSet(contentField, NewString("x"), limit); ok {
		t.Error("containsSet answered for $content instead of declining")
	}
	if _, ok := h.payloadIdx.inSet(contentField, NewStrings([]string{"x"}), limit); ok {
		t.Error("inSet answered for $content instead of declining")
	}
	if _, ok := h.payloadIdx.orderingSet(contentField, FilterGt, NewString("a"), limit); ok {
		t.Error("orderingSet answered for $content instead of declining")
	}
	if _, ok := h.payloadIdx.geoSet(contentField, FilterGeoRadius,
		&GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: 100}, limit); ok {
		t.Error("geoSet answered for $content instead of declining")
	}

	// collectEqTerms is the one that does NOT go through a *Set function — the
	// eq branch indexes p.fields inline — so it needs its own guard and its own
	// check.
	if _, ok := collectEqTerms(Filter{Op: FilterEq, Field: contentField, Value: NewString("x")}); ok {
		t.Error("collectEqTerms accepted a bare $content Eq")
	}
	terms, ok := collectEqTerms(Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "tag", Value: NewString("a")},
		{Op: FilterEq, Field: contentField, Value: NewString("x")},
	}})
	if !ok || len(terms) != 1 || terms[0].field != "tag" {
		t.Errorf("collectEqTerms should keep the tag term and drop the $content one, got ok=%v terms=%v", ok, terms)
	}

	// And the whole-filter view: a filter narrowable ONLY through $content must
	// report not-narrowable, so the caller falls back to graph traversal.
	for _, f := range []Filter{
		{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
		{Op: FilterEq, Field: contentField, Value: NewString("x")},
		{Op: FilterIn, Field: contentField, Value: NewStrings([]string{"x"})},
		{Op: FilterGt, Field: contentField, Value: NewString("a")},
	} {
		if _, ok := h.payloadIdx.collectNarrowSets(f, limit); ok {
			t.Errorf("collectNarrowSets narrowed a $content-only filter (op %v)", f.Op)
		}
	}
}

// TestContentFieldIDIndexIsNeverNarrowable is the named/multivector mirror. The
// id-keyed payload index skips contentField in exactly the same place for
// exactly the same reason, so it had exactly the same hole. No shipped named/MV
// write path routes content into a payload today — but "$content" is just a map
// key a caller can set, and the failure mode is silent wrong results, so the
// guard is symmetric rather than dense-only.
func TestContentFieldIDIndexIsNeverNarrowable(t *testing.T) {
	p := newPayloadIndexID()
	for id := uint64(1); id <= 20; id++ {
		p.reindex(id, withContent(Metadata{"tag": NewString("a")}, "the quick brown fox"))
	}
	const limit = 1000
	if _, ok := p.eqSet(contentField, NewString("x")); ok {
		t.Error("id eqSet answered for $content")
	}
	if _, ok := p.matchSet(contentField, NewString("quick"), limit); ok {
		t.Error("id matchSet answered for $content")
	}
	if _, ok := p.containsSet(contentField, NewString("x"), limit); ok {
		t.Error("id containsSet answered for $content")
	}
	if _, ok := p.inSet(contentField, NewStrings([]string{"x"}), limit); ok {
		t.Error("id inSet answered for $content")
	}
	if _, ok := p.orderingSet(contentField, FilterGt, NewString("a"), limit); ok {
		t.Error("id orderingSet answered for $content")
	}
	if _, ok := p.geoSet(contentField, FilterGeoRadius,
		&GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: 100}, limit); ok {
		t.Error("id geoSet answered for $content")
	}
	// The tag conjunct still narrows; only $content is declined.
	if _, ok := p.candidatesCapped(Filter{Op: FilterEq, Field: "tag", Value: NewString("a")}, limit, limit); !ok {
		t.Error("an ordinary field stopped narrowing on the id index")
	}
	if _, ok := p.candidatesCapped(Filter{Op: FilterMatch, Field: contentField, Value: NewString("quick")}, limit, limit); ok {
		t.Error("the id index narrowed a $content-only filter")
	}
}

// TestContentFieldStillNarrowsOnItsSiblings guards the obvious over-correction:
// declining $content must not disable narrowing for the conjuncts beside it.
// A filter that pairs $content with a real indexed field should still take the
// index path on that field — the whole point of declining rather than bailing.
func TestContentFieldStillNarrowsOnItsSiblings(t *testing.T) {
	h := contentCorpus(t, 40)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())

	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "tag", Value: NewString("a")},
		{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
	}}
	sets, ok := h.payloadIdx.collectNarrowSets(f, limit)
	if !ok {
		t.Fatal("And(Eq(tag), Match($content)) should still narrow on the tag conjunct")
	}
	if len(sets) != 1 {
		t.Fatalf("expected exactly the tag posting set in the plan, got %d sets", len(sets))
	}
	if n := len(sets[0].set); n != 40 {
		t.Fatalf("tag posting set holds %d slots, want 40", n)
	}
}

// TestContentFieldRAGSearchDocsUnaffected checks the fix through the surface
// that actually stores content, so the regression is pinned where a RAG user
// would meet it rather than only through the raw index.
func TestContentFieldRAGSearchDocsUnaffected(t *testing.T) {
	const n = 30
	h := contentCorpus(t, n)
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "tag", Value: NewString("a")},
		{Op: FilterMatch, Field: contentField, Value: NewString("brown fox")},
	}}
	res, err := h.SearchFiltered(make([]float32, 8), n, f)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	docs := h.fetchDocs(res)
	if len(docs) != n {
		t.Fatalf("content-filtered search yielded %d of %d matching documents", len(docs), n)
	}
	for _, d := range docs {
		if d.Content != "the quick brown fox" {
			t.Fatalf("document %d content = %q", d.ID, d.Content)
		}
		if _, leaked := d.Metadata[contentField]; leaked {
			t.Fatalf("document %d leaked %s into user metadata", d.ID, contentField)
		}
	}
}
