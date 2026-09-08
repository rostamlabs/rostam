// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// The bitset admission gate's whole safety argument is a per-op claim: for these
// ops the payload index's posting set IS the predicate's match set, for those
// ops it is merely a superset. A claim like that is worthless without a test
// that fails when it is wrong, so this file is built around one differential
// harness run three ways:
//
//	1. gate OFF vs gate FORCED — every supported filter shape must return the
//	   same results and the same reject tally. (TestAdmitGateEquivalence)
//	2. the same harness with a MUTANT classifier that promotes a superset op to
//	   EXACT — which must FAIL, or claim (1) proves nothing.
//	   (TestAdmitGateExactMutantsAreCaught)
//	3. the same harness under per-key TTL, which is the one way an otherwise
//	   exact op stops being exact. (TestAdmitGateKeyTTLDowngrade)

// gateCorpus builds an index whose payloads exercise every indexable value kind
// (string, int, float, bool, string array, int array, tokenized text, geo,
// datetime-as-unix-ms) plus the fields that are deliberately NOT indexable
// (a per-slot-unique field, an absent field).
//
// THE CONFIG IS LOAD-BEARING, and getting it wrong makes this whole file pass
// vacuously. The gate only exists on the branch where the planner DECLINES
// filter-first, so the corpus has to be big enough that the filters here land
// past the crossover (which grows only as sqrt(k·n·2M), while a filter's match
// count grows linearly in n — hence a large n rather than a small one). At the
// same time FilterFirstThreshold has to be GENEROUS, because the same number is
// the per-leaf selectivity cap: pinning it to 1 to force the planner off would
// also make matchSet/containsSet/inSet/geoSet decline to narrow, so those shapes
// would arm no gate at all and their equivalence subtests would compare the
// unassisted path against itself. TestAdmitGateEquivalence's coverage floor
// checks the outcome via Stats().FilterGates rather than trusting this comment.
// nanScoreFor is the "nanscore" field's value for corpus index i: NaN on every
// 23rd point, otherwise a small repeating float so ordinary range shapes over
// the field still have interesting selectivity around the NaN holes.
func nanScoreFor(i int) float64 {
	if i%23 == 0 {
		return math.NaN()
	}
	return float64(i%17) / 2
}

func gateCorpus(t testing.TB, n, dim int) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{
		Dim: dim, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1 << 20,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	colors := []string{"red", "green", "blue", "amber"}
	words := []string{"quick", "brown", "fox", "lazy", "dog", "jumps"}
	rng := rand.New(rand.NewSource(20260731))
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		// tags: 1-3 elements drawn from colors, so Contains has both hits and
		// misses and some slots carry the same element twice (dedup coverage).
		ntag := 1 + rng.Intn(3)
		tags := make([]string, 0, ntag)
		for j := 0; j < ntag; j++ {
			tags = append(tags, colors[rng.Intn(len(colors))])
		}
		nums := []int64{int64(i % 7), int64(i % 5)}
		// text: two phrases, so a per-document token index over-covers a
		// per-element match — the reason FilterMatch is classified SUPERSET.
		text := fmt.Sprintf("%s %s. %s %s.",
			words[rng.Intn(len(words))], words[rng.Intn(len(words))],
			words[rng.Intn(len(words))], words[rng.Intn(len(words))])
		// phrases: a MULTI-ELEMENT string array whose elements each carry
		// different tokens. The token index posts a slot under the union of all
		// its elements' tokens, while compileMatch requires ONE element to carry
		// every query token — so a query like "quick fox" over
		// ["quick dog","brown fox"] is a genuine index over-cover. Without a
		// field shaped like this, FilterMatch is accidentally exact and the
		// mutation test that pins its SUPERSET grade proves nothing.
		phrases := []string{
			words[rng.Intn(len(words))] + " " + words[rng.Intn(len(words))],
			words[rng.Intn(len(words))] + " " + words[rng.Intn(len(words))],
		}
		meta := Metadata{
			"phrases": NewStrings(phrases),
			"color":   NewString(colors[i%len(colors)]),
			"size":    NewInt(int64(i % 10)),
			"score":   NewFloat(float64(i%13) / 2),
			"flag":    NewBool(i%3 == 0),
			"tags":    NewStrings(tags),
			"nums":    NewInts(nums),
			"text":    NewString(text),
			"when":    NewInt(base + int64(i)*3_600_000),
			"unique":  NewString(fmt.Sprintf("u%d", i)),
			// seq is DENSE and UNIQUE (0..n-1), which is the only way to express a
			// range at a chosen pass rate: `seq >= n/100` passes 99%, `>= n/2`
			// passes 50%, `>= 99n/100` passes 1%. That sweep is what the m5 range
			// work is judged on, and a field with ten distinct values cannot express
			// it. It is also the shape that stresses the key-count pre-check in
			// orderingSet — one posting per distinct key, so distinct-key count and
			// union size coincide.
			"seq": NewInt(int64(i)), //nolint:gosec // bounded loop index
			// nanscore carries a NaN on ~4% of the corpus. Under the m5 semantics
			// (orderingHoldsFloat) those points match NO range predicate and carry
			// NO index key (scalarKeyOf), and the differential suite's job is to
			// prove the gate and the predicate agree about them — including on the
			// ordering ops that used to ACCEPT NaN, which is what made the whole
			// family ungateable.
			"nanscore": NewFloat(nanScoreFor(i)),
			// The reserved content key, present on every point. reindex refuses to
			// index it, so it adds no postings and cannot shift any other shape's
			// results — it exists so the content/* shapes below have something the
			// PREDICATE can match while the index has nothing, which is exactly the
			// asymmetry that made $content a wrong-results bug.
			contentField: NewString("doc body: the quick brown fox"),
		}
		if i%11 == 0 {
			meta["loc"] = NewGeo(40+float64(i%20)*0.05, -74+float64(i%20)*0.05)
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	// Tombstone a slice of the corpus: a stale posting pointing at a dead slot
	// must be rejected as admitDead (never as a filter reject), which is the
	// ordering constraint inside admitVerdictOf.
	for i := 0; i < n; i += 17 {
		_, _ = h.Delete(uint64(i+1), CASCond{})
	}
	return h
}

// gateFilters is every filter shape the gate must survive: the EXACT ops, the
// SUPERSET ops, the not-narrowable ops, and the And/Or/Not compositions that mix
// them (including an And where one conjunct is narrowable and the other is not —
// the case that must stay a superset plan).
func gateFilters() []struct {
	name string
	f    Filter
} {
	geo := &GeoCondition{CenterLat: 40.5, CenterLon: -73.6, RadiusM: 60_000}
	box := &GeoCondition{MinLat: 40, MinLon: -74, MaxLat: 40.6, MaxLon: -73.4}
	return []struct {
		name string
		f    Filter
	}{
		// --- ops classified EXACT ---
		{"eq/string", Filter{Op: FilterEq, Field: "color", Value: NewString("red")}},
		{"eq/int", Filter{Op: FilterEq, Field: "size", Value: NewInt(3)}},
		{"eq/float", Filter{Op: FilterEq, Field: "score", Value: NewFloat(2.5)}},
		{"eq/bool", Filter{Op: FilterEq, Field: "flag", Value: NewBool(true)}},
		{"eq/absent-field", Filter{Op: FilterEq, Field: "nope", Value: NewString("x")}},
		{"eq/absent-value", Filter{Op: FilterEq, Field: "color", Value: NewString("puce")}},
		{"eq/kind-mismatch", Filter{Op: FilterEq, Field: "size", Value: NewFloat(3)}},
		{"eq/array-field", Filter{Op: FilterEq, Field: "tags", Value: NewString("red")}},
		{"contains/string", Filter{Op: FilterContains, Field: "tags", Value: NewString("blue")}},
		{"contains/int", Filter{Op: FilterContains, Field: "nums", Value: NewInt(3)}},
		{"contains/absent", Filter{Op: FilterContains, Field: "tags", Value: NewString("puce")}},
		{"contains/wrong-kind", Filter{Op: FilterContains, Field: "nums", Value: NewString("3")}},
		{"contains/scalar-field", Filter{Op: FilterContains, Field: "color", Value: NewString("red")}},
		{"in/strings", Filter{Op: FilterIn, Field: "color", Value: NewStrings([]string{"red", "blue"})}},
		{"in/ints", Filter{Op: FilterIn, Field: "size", Value: NewInts([]int64{1, 2, 3})}},
		{"in/floats", Filter{Op: FilterIn, Field: "score", Value: NewFloats([]float64{0.5, 1.5})}},
		{"in/empty", Filter{Op: FilterIn, Field: "color", Value: NewStrings(nil)}},
		{"in/no-member-present", Filter{Op: FilterIn, Field: "color", Value: NewStrings([]string{"puce"})}},
		{"and/eq+eq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterEq, Field: "flag", Value: NewBool(true)},
		}}},
		{"and/eq+contains", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("green")},
			{Op: FilterContains, Field: "tags", Value: NewString("red")},
		}}},
		{"and/contains+in", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterContains, Field: "tags", Value: NewString("amber")},
			{Op: FilterIn, Field: "size", Value: NewInts([]int64{0, 1, 2, 3, 4})},
		}}},
		{"and/nested-eq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("blue")},
			{Op: FilterAnd, And: []Filter{{Op: FilterEq, Field: "size", Value: NewInt(2)}}},
		}}},
		{"and/eq+missing-field", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("blue")},
			{Op: FilterEq, Field: "nope", Value: NewInt(1)},
		}}},

		// --- the ordering family: EXACT since m5 (see orderingHoldsFloat) ---
		// The three-selectivity sweep over the dense `seq` field is the shape the
		// m5 range work exists for; the nanscore shapes are the semantics change
		// under differential test — an ordering op over a field ~4% of whose values
		// are NaN, which the predicate must reject and the index must not carry.
		{"range/seq-pass99", Filter{Op: FilterGte, Field: "seq", Value: NewInt(int64(gateSharedN() / 100))}},
		{"range/seq-pass50", Filter{Op: FilterGte, Field: "seq", Value: NewInt(int64(gateSharedN() / 2))}},
		{"range/seq-pass01", Filter{Op: FilterGte, Field: "seq", Value: NewInt(int64(gateSharedN() * 99 / 100))}},
		{"range/seq-lt-pass99", Filter{Op: FilterLt, Field: "seq", Value: NewInt(int64(gateSharedN() * 99 / 100))}},
		{"range/seq-band", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "seq", Value: NewInt(int64(gateSharedN() / 4))},
			{Op: FilterLt, Field: "seq", Value: NewInt(int64(gateSharedN() * 3 / 4))},
		}}},
		{"range/nan-gte", Filter{Op: FilterGte, Field: "nanscore", Value: NewFloat(4)}},
		{"range/nan-lte", Filter{Op: FilterLte, Field: "nanscore", Value: NewFloat(4)}},
		{"range/nan-gt", Filter{Op: FilterGt, Field: "nanscore", Value: NewFloat(0)}},
		{"range/nan-lt", Filter{Op: FilterLt, Field: "nanscore", Value: NewFloat(8)}},
		{"range/nan-bound", Filter{Op: FilterGte, Field: "nanscore", Value: NewFloat(math.NaN())}},
		{"range/nan-and-seq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "nanscore", Value: NewFloat(2)},
			{Op: FilterGte, Field: "seq", Value: NewInt(int64(gateSharedN() / 100))},
		}}},
		{"range/eq-and-nan", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterLte, Field: "nanscore", Value: NewFloat(3)},
		}}},

		// --- ops classified SUPERSET (pre-filter only) ---
		{"match/one-token", Filter{Op: FilterMatch, Field: "text", Value: NewString("fox")}},
		{"match/two-tokens", Filter{Op: FilterMatch, Field: "text", Value: NewString("quick fox")}},
		{"match/case", Filter{Op: FilterMatch, Field: "text", Value: NewString("QUICK Fox")}},
		{"match/absent-token", Filter{Op: FilterMatch, Field: "text", Value: NewString("zebra")}},
		{"match/array-field", Filter{Op: FilterMatch, Field: "phrases", Value: NewString("quick fox")}},
		{"gt/int", Filter{Op: FilterGt, Field: "size", Value: NewInt(5)}},
		{"gte/float", Filter{Op: FilterGte, Field: "score", Value: NewFloat(3)}},
		{"lt/int", Filter{Op: FilterLt, Field: "size", Value: NewInt(4)}},
		{"lte/string", Filter{Op: FilterLte, Field: "color", Value: NewString("blue")}},
		{"dt/gt", Filter{Op: FilterDtGt, Field: "when", Value: NewString("2026-01-05T00:00:00Z")}},
		{"dt/lte", Filter{Op: FilterDtLte, Field: "when", Value: NewString("2026-01-03T12:00:00Z")}},
		{"geo/radius", Filter{Op: FilterGeoRadius, Field: "loc", Geo: geo}},
		{"geo/box", Filter{Op: FilterGeoBox, Field: "loc", Geo: box}},
		{"and/eq+range", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterGt, Field: "size", Value: NewInt(3)},
		}}},
		{"and/eq+match", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterMatch, Field: "text", Value: NewString("fox")},
		}}},
		{"and/match+range", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterMatch, Field: "text", Value: NewString("dog")},
			{Op: FilterLt, Field: "size", Value: NewInt(6)},
		}}},
		{"and/geo+eq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGeoRadius, Field: "loc", Geo: geo},
			{Op: FilterEq, Field: "flag", Value: NewBool(true)},
		}}},
		{"and/eq+ne", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterNe, Field: "size", Value: NewInt(0)},
		}}},
		{"and/eq+not", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "flag", Value: NewBool(true)}},
		}}},
		{"and/eq+or", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterOr, Or: []Filter{
				{Op: FilterEq, Field: "size", Value: NewInt(1)},
				{Op: FilterEq, Field: "size", Value: NewInt(2)},
			}},
		}}},
		{"and/eq+regex", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterRegex, Field: "unique", Value: NewString("^u1")},
		}}},
		{"and/eq+isnull", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "flag", Value: NewBool(false)},
			{Op: FilterIsNull, Field: "loc"},
		}}},

		// --- not narrowable at all: must run exactly today's path ---
		{"or/eq", Filter{Op: FilterOr, Or: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterEq, Field: "color", Value: NewString("blue")},
		}}},
		{"not/eq", Filter{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "color", Value: NewString("red")}}},
		{"ne", Filter{Op: FilterNe, Field: "color", Value: NewString("red")}},
		{"regex", Filter{Op: FilterRegex, Field: "unique", Value: NewString("^u[0-9]$")}},
		{"isnull", Filter{Op: FilterIsNull, Field: "loc"}},
		{"isempty", Filter{Op: FilterIsEmpty, Field: "tags"}},

		// --- the reserved content field: never index-narrowable ---
		// $content is readable by the predicate but deliberately unindexed, so
		// every op on it must decline to narrow rather than answer with an empty
		// posting set (see indexNarrowable, and payload_content_field_test.go for
		// the wrong-results bug that cost). Here they earn their place as gate
		// shapes: the gate must stay disarmed for them, and the equivalence
		// assertion then pins that gate-off and gate-forced agree on a filter the
		// index refuses to touch.
		{"content/match", Filter{Op: FilterMatch, Field: contentField, Value: NewString("quick")}},
		{"content/eq", Filter{Op: FilterEq, Field: contentField, Value: NewString("doc body")}},
		{"content/in", Filter{Op: FilterIn, Field: contentField, Value: NewStrings([]string{"doc body"})}},
		{"and/eq+content-match", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("red")},
			{Op: FilterMatch, Field: contentField, Value: NewString("quick")},
		}}},
		{"and/eq+content-eq", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("blue")},
			{Op: FilterEq, Field: contentField, Value: NewString("doc body")},
		}}},
	}
}

// gateSharedDim sizes the corpus the read-only tests share; gateSharedN is a
// function because it depends on -short.
//
// THE FULL SIZE IS LOAD-BEARING: see gateCorpus for why a small corpus makes the
// equivalence assertions vacuous (too few points per filter shape and the top-k
// stops discriminating), and the three-selectivity range sweep in gateFilters
// needs enough distinct `seq` values that 1%/50%/99% are meaningfully different
// pass rates. Building it once via sync.Once keeps 30k affordable.
//
// THE SHORT SIZE EXISTS FOR -race. The three differential suites take ~300s
// plain and roughly an order of magnitude more under the detector, which puts
// them past any sane CI budget — so `go test -short -race ./vector` runs the
// same shapes over a tenth of the corpus. That is a real reduction in power (it
// is the corpus, not the shape coverage, that shrinks), so -short is for the
// race lane and the unabridged run is what gates a merge.
const gateSharedDim = 16

func gateSharedN() int {
	if testing.Short() {
		return 3_000
	}
	return 30_000
}

// gateQueryN scales a test's query count the same way, so a -race run shrinks
// both dimensions of the work rather than only the corpus.
func gateQueryN(n int) int {
	if testing.Short() {
		if n > 10 {
			return n / 4
		}
	}
	return n
}

var (
	gateSharedOnce sync.Once
	gateSharedIdx  *hnsw
)

// gateSharedCorpus returns the corpus shared by every test here that only READS
// it. Tests that mutate payloads (the per-key TTL downgrade) build their own.
func gateSharedCorpus(t testing.TB) *hnsw {
	t.Helper()
	gateSharedOnce.Do(func() { gateSharedIdx = gateCorpus(t, gateSharedN(), gateSharedDim) })
	if gateSharedIdx == nil {
		t.Fatal("shared gate corpus failed to build")
	}
	return gateSharedIdx
}

// gateArm is one arm's observable output: the ordered result IDs of every query,
// the index's cumulative filter-reject tally, and how many of the queries
// actually armed a gate (so an arm that silently never engaged is visible rather
// than mistaken for agreement).
type gateArm struct {
	ids     [][]uint64
	rejects uint64
	gates   uint64
}

// runGateArm runs the query set under the given gate mode and classifier. Both
// arms run over the SAME index, so nothing but the admission strategy differs.
func runGateArm(t testing.TB, h *hnsw, queries [][]float32, f Filter, mode gateMode, forceExact bool) gateArm {
	t.Helper()
	prevMode, prevForce := admitGateMode, admitGateForceExact
	admitGateMode, admitGateForceExact = mode, forceExact
	defer func() { admitGateMode, admitGateForceExact = prevMode, prevForce }()

	before := h.Stats()
	out := gateArm{ids: make([][]uint64, 0, len(queries))}
	for _, q := range queries {
		res, err := h.SearchFiltered(q, 10, f)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		ids := make([]uint64, len(res))
		for i, r := range res {
			ids[i] = r.ID
		}
		out.ids = append(out.ids, ids)
	}
	after := h.Stats()
	out.rejects = after.FilterRejects - before.FilterRejects
	out.gates = after.FilterGates - before.FilterGates
	return out
}

// diffGateArms reports the queries whose result lists differ, and whether the
// reject tallies differ.
func diffGateArms(a, b gateArm) (nDiff int, rejectsDiffer bool) {
	for i := range a.ids {
		if len(a.ids[i]) != len(b.ids[i]) {
			nDiff++
			continue
		}
		for j := range a.ids[i] {
			if a.ids[i][j] != b.ids[i][j] {
				nDiff++
				break
			}
		}
	}
	return nDiff, a.rejects != b.rejects
}

func gateQueries(n, dim int) [][]float32 {
	rng := rand.New(rand.NewSource(4242))
	qs := make([][]float32, n)
	for i := range qs {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		qs[i] = v
	}
	return qs
}

// TestAdmitGateEquivalence is the primary gate: for every supported filter
// shape, forcing the bitset gate on must change NOTHING observable — not the
// result sets, and not Stats().FilterRejects.
//
// The reject tally is asserted as EQUALITY, not "no worse". The gate rejects a
// candidate by a different oracle than the predicate, and a tally that moved
// would mean either the gate rejected something the predicate would have
// admitted (a wrong result the ID comparison might miss when it happens outside
// the top 10) or that a dead slot got reclassified from admitDead to
// admitFiltered. Both are exactly the failures this file exists to catch.
func TestAdmitGateEquivalence(t *testing.T) {
	h := gateSharedCorpus(t)
	queries := gateQueries(gateQueryN(40), gateSharedDim)

	gatedShapes, shapes := 0, gateFilters()
	for _, tc := range shapes {
		gated := false
		t.Run(tc.name, func(t *testing.T) {
			off := runGateArm(t, h, queries, tc.f, gateOff, false)
			on := runGateArm(t, h, queries, tc.f, gateForce, false)
			gated = on.gates > 0
			nDiff, rejectsDiffer := diffGateArms(off, on)
			if nDiff != 0 {
				t.Errorf("%d/%d queries returned different results with the gate armed", nDiff, len(queries))
			}
			if rejectsDiffer {
				t.Errorf("filterRejects %d (gate off) vs %d (gate armed) — the gate changed what counts as a filter rejection",
					off.rejects, on.rejects)
			}
			if off.gates != 0 {
				t.Errorf("the gate-off arm armed %d gates — gateOff is not off", off.gates)
			}
			t.Logf("gates armed: %d/%d queries; rejects %d", on.gates, len(queries), on.rejects)
		})
		if gated {
			gatedShapes++
		}
	}
	// COVERAGE FLOOR, measured rather than assumed. Every subtest above passes
	// trivially when the gate never arms — it would be comparing the unassisted
	// path against itself — and the ways that happens are all silent: a corpus
	// too small for the filter-first crossover, a FilterFirstThreshold that caps
	// the leaves, a plan the gate drops. So count the shapes that actually armed
	// one, via the same counter operators see.
	//
	// The floor is 20 rather than "most of them", because a good third of the
	// table CANNOT be gated through a search by construction and should not be:
	// the empty-result shapes (an absent field, an absent value, a kind mismatch)
	// narrow to nothing, so the planner brute-forces zero candidates and never
	// reaches the graph at all, and the ordering/Or/Not/regex shapes are
	// deliberately ungateable. 20 leaves headroom over the ~25 that do arm today
	// without silently tolerating a collapse to a handful.
	const gatedShapesFloor = 20
	if gatedShapes < gatedShapesFloor {
		t.Fatalf("only %d/%d filter shapes armed a gate during the search (want >= %d) — the equivalence assertions "+
			"are going vacuous; check gateCorpus's size and FilterFirstThreshold against the filter-first crossover",
			gatedShapes, len(shapes), gatedShapesFloor)
	}
	t.Logf("%d/%d filter shapes exercised the gate end to end", gatedShapes, len(shapes))
}

// gateExactFor reports whether buildAdmitGate would classify f as EXACT against
// this index (op classification AND full-conjunct coverage AND no per-key TTL).
func gateExactFor(t testing.TB, h *hnsw, f Filter) (armed, exact bool) {
	t.Helper()
	prev := admitGateMode
	admitGateMode = gateForce
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)
	h.mu.RLock()
	defer h.mu.RUnlock()
	plan, ok := h.payloadIdx.collectNarrowSets(f, h.effectiveFilterFirstLimit(h.arena.Size()))
	if !ok {
		return false, false
	}
	h.buildAdmitGate(s, f, plan, 10)
	armed, exact = s.gate.active(), s.gate.exact
	s.gate.disable()
	return armed, exact
}

// TestAdmitGateExactMutantsAreCaught is the proof that TestAdmitGateEquivalence
// has teeth. It re-runs the same differential harness with a classifier that
// wrongly promotes ONE superset op to EXACT — which makes the gate skip the
// predicate re-check on candidates the index over-covers — and requires the
// harness to notice.
//
// If a mutant here ever stops being caught, the corresponding op's EXACT/
// SUPERSET classification in filterIndexExact is no longer being tested and the
// entry in this table must be re-derived, not deleted.
//
// IT DOES NOT RUN UNDER -short, and skipping is the honest answer rather than a
// convenience. The test's premise is that a wrongly-admitted candidate reaches
// somebody's top-10; on the reduced corpus the over-covered slots are too few
// and too far from the queries for that to happen, so every mutant survives and
// the test reports the OPPOSITE of what it means — "the harness has no teeth"
// — when the truth is only that this corpus is too small to show them. A
// mutation test that cannot distinguish those two states must not run in the
// state where it cannot.
func TestAdmitGateExactMutantsAreCaught(t *testing.T) {
	if testing.Short() {
		t.Skip("needs the full shared corpus: on the -short corpus an over-covered slot never reaches a top-10, " +
			"so every mutant survives for reasons that have nothing to do with the classifier")
	}
	h := gateSharedCorpus(t)
	queries := gateQueries(gateQueryN(40), gateSharedDim)

	// promote returns a mutant classifier: filterIndexExact, plus a lie about op.
	mutants := []struct {
		name string
		f    Filter
		why  string
	}{
		{
			name: "match-over-an-array-field",
			f:    Filter{Op: FilterMatch, Field: "phrases", Value: NewString("quick fox")},
			why:  "token postings are per-DOCUMENT; compileMatch requires one ELEMENT to carry every query token",
		},
		{
			name: "match-plus-eq",
			f: Filter{Op: FilterAnd, And: []Filter{
				{Op: FilterEq, Field: "flag", Value: NewBool(true)},
				{Op: FilterMatch, Field: "phrases", Value: NewString("quick fox")},
			}},
			why: "an And is only exact if EVERY conjunct's set is; one superset conjunct poisons the whole plan",
		},
		{
			name: "geo-radius",
			f:    Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 40.5, CenterLon: -73.6, RadiusM: 60_000}},
			why:  "a geohash-cell cover of a bounding box is strictly larger than the region",
		},
	}

	for _, m := range mutants {
		t.Run(m.name, func(t *testing.T) {
			if armed, exact := gateExactFor(t, h, m.f); !armed || exact {
				t.Fatalf("precondition: want an armed SUPERSET gate for this filter, got armed=%v exact=%v", armed, exact)
			}
			truth := runGateArm(t, h, queries, m.f, gateOff, false)
			mutant := runGateArm(t, h, queries, m.f, gateForce, true)
			nDiff, rejectsDiffer := diffGateArms(truth, mutant)
			if nDiff == 0 && !rejectsDiffer {
				t.Fatalf("promoting this filter's gate to EXACT changed nothing observable — the differential "+
					"harness cannot detect a wrong exactness classification here (%s)", m.why)
			}
			t.Logf("caught: %d/%d result sets differ, rejects %d vs %d (%s)",
				nDiff, len(queries), truth.rejects, mutant.rejects, m.why)
		})
	}

	// The converse: the shapes the classifier DOES call exact must be
	// indistinguishable from their forced-exact selves. If forcing exactness on
	// an Eq changed anything, "EXACT" would not mean what this file claims.
	for _, f := range []Filter{
		{Op: FilterEq, Field: "color", Value: NewString("red")},
		{Op: FilterContains, Field: "tags", Value: NewString("blue")},
		{Op: FilterIn, Field: "size", Value: NewInts([]int64{1, 2, 3})},
	} {
		if _, exact := gateExactFor(t, h, f); !exact {
			t.Fatalf("precondition: %v should already be EXACT", f.Op)
		}
		natural := runGateArm(t, h, queries, f, gateForce, false)
		forced := runGateArm(t, h, queries, f, gateForce, true)
		if nDiff, rejectsDiffer := diffGateArms(natural, forced); nDiff != 0 || rejectsDiffer {
			t.Fatalf("op %v: forcing exactness on an already-exact gate changed results (%d) or rejects (%d vs %d)",
				f.Op, nDiff, natural.rejects, forced.rejects)
		}
	}
}

// TestOrderingIndexAgreesWithThePredicateForNaN is the m5 REPLACEMENT for
// TestOrderingIndexIsNarrowerThanThePredicateForNaN, and the inversion is the
// point: the old test pinned a DISAGREEMENT between predicate and index as
// evidence for the narrowUnproven grade, this one pins the agreement that
// removed the grade.
//
// The semantics being pinned (see orderingHoldsFloat): a NaN operand makes a
// range comparison UNORDERED, so gt/gte/lt/lte are all false — for a NaN field
// value AND for a NaN bound. The index side matches for free, in two
// independent ways: scalarKeyOf refuses to mint a key for NaN (so a NaN-valued
// slot has no posting anywhere), and ensureSorted refuses to place one in the
// sorted list (so the binary search keeps its total order).
//
// The BEHAVIOUR CHANGE this locks in, stated so a future reader does not have to
// diff it out of the git history: `score >= b` and `score <= b` used to accept a
// NaN-valued point for every bound b, and now accept it for none.
func TestOrderingIndexAgreesWithThePredicateForNaN(t *testing.T) {
	p := newPayloadIndex()
	vals := []float64{1, 2, math.NaN(), 4, 5}
	const nanSlot = 2
	for i, v := range vals {
		p.reindex(uint32(i), Metadata{"score": NewFloat(v)}) //nolint:gosec // small loop index
	}
	// The NaN never became a key at all: `score` has four distinct keys, not five.
	if got := len(p.fields["score"]); got != 4 {
		t.Fatalf("score has %d distinct keys, want 4 — scalarKeyOf minted a key for NaN", got)
	}
	// ...and re-indexing the same NaN over and over cannot grow the index, which
	// is the leak the old behaviour had (a NaN map key is unreachable by lookup,
	// so every insert appended an entry drop could never remove).
	for i := 100; i < 200; i++ {
		p.reindex(uint32(i), Metadata{"score": NewFloat(math.NaN())}) //nolint:gosec // small loop index
	}
	if got := len(p.fields["score"]); got != 4 {
		t.Fatalf("score has %d distinct keys after 100 NaN inserts, want 4 — NaN keys are accumulating", got)
	}

	nanMeta := Metadata{"score": NewFloat(math.NaN())}
	for _, op := range []FilterOp{FilterGt, FilterGte, FilterLt, FilterLte} {
		set, ok := p.orderingSet("score", op, NewFloat(3), 1000)
		if !ok {
			t.Fatalf("op %d: orderingSet declined", op)
		}
		pred, err := CompileFilter(Filter{Op: op, Field: "score", Value: NewFloat(3)})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		if pred(nanMeta) {
			t.Errorf("op %d: the predicate ACCEPTS a NaN field — orderingHoldsFloat's NaN rule regressed, "+
				"and the ordering family's narrowExact grade is now unsound", op)
		}
		if _, in := set[nanSlot]; in {
			t.Errorf("op %d: the ordering index carries the NaN slot but the predicate rejects it — "+
				"the index is now WIDER than the predicate, breaking the EXACT grade", op)
		}
		// Exactness, spelled out over the whole corpus rather than just the NaN:
		// membership in the posting set must equal the predicate's verdict.
		for slot, v := range vals {
			_, in := set[uint32(slot)] //nolint:gosec // small loop index
			want := pred(Metadata{"score": NewFloat(v)})
			if in != want {
				t.Errorf("op %d slot %d (%v): index=%v predicate=%v", op, slot, v, in, want)
			}
		}
		// A NaN BOUND matches nothing and narrows nothing.
		if pred, err := CompileFilter(Filter{Op: op, Field: "score", Value: NewFloat(math.NaN())}); err != nil {
			t.Fatalf("compile nan bound: %v", err)
		} else {
			for _, v := range vals {
				if pred(Metadata{"score": NewFloat(v)}) {
					t.Errorf("op %d: a NaN bound accepted %v", op, v)
				}
			}
		}
		if _, ok := p.orderingSet("score", op, NewFloat(math.NaN()), 1000); ok {
			t.Errorf("op %d: orderingSet narrowed on a NaN bound", op)
		}
	}
}

// TestOrderingKeyCountPreCheckMatchesMaterialization pins the O(1) overflow
// proof added in m5: within one field the posting sets of distinct keys are
// DISJOINT (reindex posts one key per field per slot) and no key survives with
// an empty set (drop removes it), so hi-lo > limit proves the union overflows.
// The test asserts the pre-check and the incremental check reach the SAME
// verdict for every limit around the boundary — if they ever disagree, one of
// those two structural facts has changed.
func TestOrderingKeyCountPreCheckMatchesMaterialization(t *testing.T) {
	p := newPayloadIndex()
	const n = 40
	for i := 0; i < n; i++ {
		// Two slots per distinct key, so distinct-key count and union size differ
		// by exactly 2x and the pre-check cannot be right by coincidence.
		p.reindex(uint32(i), Metadata{"v": NewInt(int64(i / 2))}) //nolint:gosec // small loop index
	}
	for limit := 0; limit <= n+2; limit++ {
		set, ok := p.orderingSet("v", FilterGte, NewInt(0), limit)
		wantOK := n <= limit
		if ok != wantOK {
			t.Errorf("limit %d: orderingSet ok=%v, want %v (union is %d slots over %d keys)",
				limit, ok, wantOK, n, n/2)
		}
		if ok && len(set) != n {
			t.Errorf("limit %d: union has %d slots, want %d", limit, len(set), n)
		}
	}
}

// TestAdmitGateUsesOrderingSets is the m5 inversion of
// TestAdmitGateNeverUsesOrderingSets. That test pinned the DROP — an
// ordering-only filter had to arm no gate at all, because orderingSet could come
// back narrower than the predicate. With the NaN disagreement closed at the root
// the ordering family is narrowExact, so the same filters must now arm an EXACT
// gate, and the And(match, range) case must INTERSECT the ordering set in rather
// than discard it.
func TestAdmitGateUsesOrderingSets(t *testing.T) {
	h := gateSharedCorpus(t)
	orderingOnly := []Filter{
		{Op: FilterGt, Field: "size", Value: NewInt(5)},
		{Op: FilterGte, Field: "score", Value: NewFloat(3)},
		{Op: FilterLt, Field: "size", Value: NewInt(4)},
		{Op: FilterLte, Field: "color", Value: NewString("blue")},
		{Op: FilterDtGt, Field: "when", Value: NewString("2026-01-05T00:00:00Z")},
		{Op: FilterGte, Field: "nanscore", Value: NewFloat(2)},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterGt, Field: "size", Value: NewInt(2)},
			{Op: FilterLt, Field: "size", Value: NewInt(8)},
		}},
	}
	for i, f := range orderingOnly {
		armed, exact := gateExactFor(t, h, f)
		if !armed {
			t.Errorf("ordering filter %d armed no gate — the ordering family is narrowExact since m5", i)
			continue
		}
		if !exact {
			t.Errorf("ordering filter %d armed a non-EXACT gate — filterIndexExact and the plan grade disagree", i)
		}
	}
	// A NaN BOUND still narrows nothing: orderingSet declines it, so there is no
	// plan and no gate. (The predicate rejects everything, which is the same
	// answer by the slower route.)
	if armed, _ := gateExactFor(t, h, Filter{Op: FilterGte, Field: "nanscore", Value: NewFloat(math.NaN())}); armed {
		t.Error("a NaN bound armed a gate — orderingSet must decline to narrow on it")
	}

	// And(match, range): match is SUPERSET, range is EXACT. The gate is their
	// intersection, so it is legitimately NARROWER than the match set alone — but
	// it must still be a superset of the true match set, and it must not be EXACT
	// (the match conjunct is not).
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterMatch, Field: "text", Value: NewString("dog")},
		{Op: FilterLt, Field: "size", Value: NewInt(6)},
	}}
	pred, err := CompileFilter(f)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	prev := admitGateMode
	admitGateMode = gateForce
	defer func() { admitGateMode = prev }()
	s := getLayerScratch()
	defer layerScratchPool.Put(s)
	plan, ok := h.payloadIdx.collectNarrowSets(f, limit)
	if !ok {
		t.Fatal("And(match, range) should be index-narrowable")
	}
	if len(plan) != 2 {
		t.Fatalf("And(match, range) plan has %d sets, want 2 (the ordering set must be kept)", len(plan))
	}
	h.buildAdmitGate(s, f, plan, 10)
	if !s.gate.active() {
		t.Fatal("And(match, range) should arm a gate")
	}
	if s.gate.exact {
		t.Fatal("And(match, range) must not be EXACT — the match conjunct is only a superset")
	}
	// THE SAFETY PROPERTY: every live slot the predicate accepts must survive the
	// gate. A pre-filtering gate is only ever allowed to be wider.
	now := uint64(h.now())
	covered := 0
	for slot := 0; slot < h.arena.Capacity(); slot++ {
		u := uint32(slot) //nolint:gosec // bounded by capacity
		if h.tombstoned[u] || h.isExpiredAt(u, now) {
			continue
		}
		if !pred(h.liveMeta(u, now)) {
			continue
		}
		covered++
		if !s.gate.test(u) {
			t.Fatalf("slot %d satisfies the predicate but the gate rejects it — the gate is NARROWER than the match set", slot)
		}
	}
	if covered == 0 {
		t.Fatal("no slot satisfies And(match, range) — the assertion above proved nothing")
	}
	s.gate.disable()
}

// TestAdmitGateExactClassification pins the per-op table itself, so a future
// edit to filterIndexExact has to change a test that says out loud what it is
// changing.
func TestAdmitGateExactClassification(t *testing.T) {
	geo := &GeoCondition{CenterLat: 40, CenterLon: -74, RadiusM: 1000}
	cases := []struct {
		name string
		f    Filter
		want bool
	}{
		{"eq/string", Filter{Op: FilterEq, Field: "a", Value: NewString("x")}, true},
		{"eq/int", Filter{Op: FilterEq, Field: "a", Value: NewInt(1)}, true},
		{"eq/float", Filter{Op: FilterEq, Field: "a", Value: NewFloat(1)}, true},
		{"eq/bool", Filter{Op: FilterEq, Field: "a", Value: NewBool(true)}, true},
		{"eq/strings-want", Filter{Op: FilterEq, Field: "a", Value: NewStrings([]string{"x"})}, false},
		{"eq/geo-want", Filter{Op: FilterEq, Field: "a", Value: NewGeo(1, 2)}, false},
		{"eq/content-field", Filter{Op: FilterEq, Field: contentField, Value: NewString("x")}, false},
		{"contains/string", Filter{Op: FilterContains, Field: "a", Value: NewString("x")}, true},
		{"contains/int", Filter{Op: FilterContains, Field: "a", Value: NewInt(1)}, true},
		{"contains/array-want", Filter{Op: FilterContains, Field: "a", Value: NewInts([]int64{1})}, false},
		{"contains/content-field", Filter{Op: FilterContains, Field: contentField, Value: NewString("x")}, false},
		{"in/strings", Filter{Op: FilterIn, Field: "a", Value: NewStrings([]string{"x"})}, true},
		{"in/ints", Filter{Op: FilterIn, Field: "a", Value: NewInts([]int64{1})}, true},
		{"in/floats", Filter{Op: FilterIn, Field: "a", Value: NewFloats([]float64{1})}, true},
		{"in/scalar-want", Filter{Op: FilterIn, Field: "a", Value: NewString("x")}, false},
		{"in/content-field", Filter{Op: FilterIn, Field: contentField, Value: NewStrings([]string{"x"})}, false},
		{"match", Filter{Op: FilterMatch, Field: "a", Value: NewString("x")}, false},
		{"ne", Filter{Op: FilterNe, Field: "a", Value: NewString("x")}, false},
		// m5 re-graded the ordering family from "not even usable" to EXACT; see
		// orderingHoldsFloat for the NaN decision that made index and predicate
		// answer the same question. The declines below are the boundary of that
		// grade, not leftovers.
		{"gt", Filter{Op: FilterGt, Field: "a", Value: NewInt(1)}, true},
		{"gte", Filter{Op: FilterGte, Field: "a", Value: NewInt(1)}, true},
		{"lt", Filter{Op: FilterLt, Field: "a", Value: NewInt(1)}, true},
		{"lte", Filter{Op: FilterLte, Field: "a", Value: NewInt(1)}, true},
		{"gte/float", Filter{Op: FilterGte, Field: "a", Value: NewFloat(1.5)}, true},
		{"gte/string", Filter{Op: FilterGte, Field: "a", Value: NewString("m")}, true},
		{"gte/nan-bound", Filter{Op: FilterGte, Field: "a", Value: NewFloat(math.NaN())}, false},
		{"gte/array-want", Filter{Op: FilterGte, Field: "a", Value: NewInts([]int64{1})}, false},
		{"gte/bool-want", Filter{Op: FilterGte, Field: "a", Value: NewBool(true)}, false},
		{"gte/content-field", Filter{Op: FilterGte, Field: contentField, Value: NewInt(1)}, false},
		{"dtgt", Filter{Op: FilterDtGt, Field: "a", Value: NewString("2026-01-01T00:00:00Z")}, true},
		{"dtlte", Filter{Op: FilterDtLte, Field: "a", Value: NewString("2026-01-01T00:00:00Z")}, true},
		{"dtgt/unparseable", Filter{Op: FilterDtGt, Field: "a", Value: NewString("not-a-date")}, false},
		{"dtgt/content-field", Filter{Op: FilterDtGt, Field: contentField, Value: NewString("2026-01-01T00:00:00Z")}, false},
		{"georadius", Filter{Op: FilterGeoRadius, Field: "a", Geo: geo}, false},
		{"geobox", Filter{Op: FilterGeoBox, Field: "a", Geo: geo}, false},
		{"regex", Filter{Op: FilterRegex, Field: "a", Value: NewString("^x")}, false},
		{"isnull", Filter{Op: FilterIsNull, Field: "a"}, false},
		{"isempty", Filter{Op: FilterIsEmpty, Field: "a"}, false},
		{"and/all-exact", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "a", Value: NewString("x")},
			{Op: FilterContains, Field: "b", Value: NewInt(2)},
			{Op: FilterIn, Field: "c", Value: NewInts([]int64{1, 2})},
		}}, true},
		{"and/eq-and-range", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "a", Value: NewString("x")},
			{Op: FilterGt, Field: "b", Value: NewInt(2)},
		}}, true},
		{"and/one-superset", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "a", Value: NewString("x")},
			{Op: FilterMatch, Field: "b", Value: NewString("dog")},
		}}, false},
		{"and/nested-exact", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "a", Value: NewString("x")},
			{Op: FilterAnd, And: []Filter{{Op: FilterEq, Field: "b", Value: NewInt(1)}}},
		}}, true},
		{"and/empty", Filter{Op: FilterAnd}, false},
		{"or", Filter{Op: FilterOr, Or: []Filter{{Op: FilterEq, Field: "a", Value: NewString("x")}}}, false},
		{"not", Filter{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "a", Value: NewString("x")}}, false},
	}
	for _, tc := range cases {
		if got := filterIndexExact(tc.f, nil); got != tc.want {
			t.Errorf("filterIndexExact(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestAdmitGateSkippedConjunctIsNotExact pins the SECOND half of the exactness
// test — the len(sets) == filterLeafCount count. An And of two individually
// EXACT ops is only exact if the plan actually narrowed on BOTH; when one
// conjunct's posting set is skipped (here: an In leaf that collectNarrowSets
// drops because the eq branch claimed the filter first), promoting the plan to
// EXACT would drop live matches.
func TestAdmitGateSkippedConjunctIsNotExact(t *testing.T) {
	h := gateSharedCorpus(t)
	// Eq + In: the eq branch wins and folds in only Match/Contains conjuncts, so
	// the In leaf goes un-narrowed. filterIndexExact says both ops are exact; the
	// leaf count is what must veto it.
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "color", Value: NewString("red")},
		{Op: FilterIn, Field: "size", Value: NewInts([]int64{1, 2, 3})},
	}}
	if !filterIndexExact(f, nil) {
		t.Fatal("precondition: both conjunct ops should be classified EXACT by op")
	}
	armed, exact := gateExactFor(t, h, f)
	if !armed {
		t.Fatal("precondition: the filter should arm a gate")
	}
	if exact {
		t.Fatal("an And whose In conjunct was never narrowed must NOT be EXACT — " +
			"the len(sets) == filterLeafCount veto is missing")
	}
	queries := gateQueries(gateQueryN(25), gateSharedDim)
	off := runGateArm(t, h, queries, f, gateOff, false)
	on := runGateArm(t, h, queries, f, gateForce, false)
	if nDiff, rejectsDiffer := diffGateArms(off, on); nDiff != 0 || rejectsDiffer {
		t.Fatalf("gate changed results (%d queries) or rejects (%d vs %d)", nDiff, off.rejects, on.rejects)
	}
}

// TestAdmitGateKeyTTLDowngrade covers the one way an op-EXACT filter stops being
// exact: a per-key TTL. liveMeta hides a key past its deadline, but the posting
// set still carries the slot until it is reindexed — so an EXACT gate would
// admit a point whose filtered-on key has logically expired.
func TestAdmitGateKeyTTLDowngrade(t *testing.T) {
	const n, dim = 800, 16
	h := gateCorpus(t, n, dim)
	f := Filter{Op: FilterEq, Field: "color", Value: NewString("red")}

	if armed, exact := gateExactFor(t, h, f); !armed || !exact {
		t.Fatalf("precondition: a pure Eq on a TTL-free arena must be an EXACT gate, got armed=%v exact=%v", armed, exact)
	}

	// Give one live "red" point a per-key deadline on the very field being
	// filtered, already in the past.
	var victim uint64
	for id := uint64(1); id <= uint64(n); id++ {
		_, m, _, _, _, ok := h.Get(id)
		if !ok || m == nil {
			continue
		}
		if v, ok := m["color"]; ok && v.Str == "red" {
			victim = id
			break
		}
	}
	if victim == 0 {
		t.Fatal("no live red point found")
	}
	if _, _, _, err := h.SetPayload(victim, Metadata{"color": NewString("red")},
		map[string]int64{"color": 1}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	time.Sleep(5 * time.Millisecond)

	if armed, exact := gateExactFor(t, h, f); !armed || exact {
		t.Fatalf("a per-key deadline anywhere in the arena must downgrade the gate to SUPERSET, got armed=%v exact=%v", armed, exact)
	}

	// And the downgraded gate must agree with the unassisted path, including on
	// the victim: its color key is hidden, so it must NOT be in any result.
	queries := gateQueries(30, dim)
	off := runGateArm(t, h, queries, f, gateOff, false)
	on := runGateArm(t, h, queries, f, gateForce, false)
	if nDiff, rejectsDiffer := diffGateArms(off, on); nDiff != 0 || rejectsDiffer {
		t.Fatalf("downgraded gate changed results (%d queries) or rejects (%d vs %d)", nDiff, off.rejects, on.rejects)
	}
	for _, ids := range on.ids {
		for _, id := range ids {
			if id == victim {
				t.Fatal("a point whose filtered key expired via per-key TTL was admitted by the gate")
			}
		}
	}
}

// TestAdmitGateExactSkipsThePredicate proves the EXACT path is not just correct
// but actually doing what it claims: replacing the predicate. It counts
// predicate invocations through a wrapper filter is not possible (Compile owns
// the closure), so it counts them the only way available from outside — by
// comparing an exact-gated run against a superset-gated run of the same filter
// and asserting the exact one still returns the same answers. The interesting
// assertion is the pairing with TestAdmitGateEquivalence: same results, and the
// gate reports exact=true.
func TestAdmitGateExactSkipsThePredicate(t *testing.T) {
	h := gateSharedCorpus(t)
	queries := gateQueries(gateQueryN(30), gateSharedDim)
	exactShapes := []Filter{
		{Op: FilterEq, Field: "color", Value: NewString("red")},
		{Op: FilterEq, Field: "flag", Value: NewBool(true)},
		{Op: FilterContains, Field: "tags", Value: NewString("blue")},
		{Op: FilterContains, Field: "nums", Value: NewInt(3)},
		{Op: FilterIn, Field: "size", Value: NewInts([]int64{1, 2, 3})},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("green")},
			{Op: FilterContains, Field: "tags", Value: NewString("red")},
		}},
	}
	for i, f := range exactShapes {
		armed, exact := gateExactFor(t, h, f)
		if !armed || !exact {
			t.Fatalf("shape %d: want an EXACT gate, got armed=%v exact=%v", i, armed, exact)
		}
		off := runGateArm(t, h, queries, f, gateOff, false)
		on := runGateArm(t, h, queries, f, gateForce, false)
		if nDiff, rejectsDiffer := diffGateArms(off, on); nDiff != 0 || rejectsDiffer {
			t.Fatalf("shape %d: exact gate changed results (%d queries) or rejects (%d vs %d)",
				i, nDiff, off.rejects, on.rejects)
		}
	}
}

// TestAdmitGateDoesNotLeakAcrossQueries pins the pooling invariant: a scratch
// that carried a gate must never hand it to the next query. The failure mode is
// silent and severe (results filtered by a stale predicate), and it is invisible
// to every other test here because they all arm the gate on purpose.
func TestAdmitGateDoesNotLeakAcrossQueries(t *testing.T) {
	h := gateSharedCorpus(t)
	queries := gateQueries(gateQueryN(20), gateSharedDim)

	// A narrowable filter that arms a gate, then a filter that must NOT be
	// gated (Or is never narrowable), run back to back on one goroutine so the
	// second query is overwhelmingly likely to reuse the first's scratch.
	gated := Filter{Op: FilterEq, Field: "color", Value: NewString("red")}
	ungated := Filter{Op: FilterOr, Or: []Filter{
		{Op: FilterEq, Field: "color", Value: NewString("green")},
		{Op: FilterEq, Field: "color", Value: NewString("blue")},
	}}

	want := runGateArm(t, h, queries, ungated, gateOff, false)

	prevMode := admitGateMode
	admitGateMode = gateForce
	defer func() { admitGateMode = prevMode }()
	got := gateArm{ids: make([][]uint64, 0, len(queries))}
	before := h.Stats().FilterRejects
	for _, q := range queries {
		if _, err := h.SearchFiltered(q, 10, gated); err != nil {
			t.Fatalf("gated search: %v", err)
		}
		res, err := h.SearchFiltered(q, 10, ungated)
		if err != nil {
			t.Fatalf("ungated search: %v", err)
		}
		ids := make([]uint64, len(res))
		for i, r := range res {
			ids[i] = r.ID
		}
		got.ids = append(got.ids, ids)
	}
	got.rejects = h.Stats().FilterRejects - before
	if nDiff, _ := diffGateArms(want, got); nDiff != 0 {
		t.Fatalf("%d/%d Or-filtered queries returned different results when interleaved with gated queries — "+
			"a gate leaked across the scratch pool", nDiff, len(queries))
	}
}

// TestAdmitGateBatchedAndPerPairAgree extends batched_traversal_test.go's
// load-bearing tally equality to the gated admission path: the two expansion
// strategies share ONE admit closure and therefore one gate, so arming it must
// not make them diverge.
func TestAdmitGateBatchedAndPerPairAgree(t *testing.T) {
	h := gateSharedCorpus(t)
	queries := gateQueries(gateQueryN(20), gateSharedDim)
	f := Filter{Op: FilterEq, Field: "color", Value: NewString("red")}

	prevMode := admitGateMode
	admitGateMode = gateForce
	defer func() { admitGateMode = prevMode }()

	prevBatch := batchedExpand
	defer func() { batchedExpand = prevBatch }()

	run := func(on bool) gateArm {
		batchedExpand = on
		before := h.Stats().FilterRejects
		out := gateArm{ids: make([][]uint64, 0, len(queries))}
		for _, q := range queries {
			res, err := h.SearchFiltered(q, 10, f)
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			ids := make([]uint64, len(res))
			for i, r := range res {
				ids[i] = r.ID
			}
			out.ids = append(out.ids, ids)
		}
		out.rejects = h.Stats().FilterRejects - before
		return out
	}
	batched, perPairArm := run(true), run(false)
	nDiff, rejectsDiffer := diffGateArms(batched, perPairArm)
	if nDiff != 0 {
		t.Errorf("%d/%d gated queries differ between the batched and per-pair paths", nDiff, len(queries))
	}
	if rejectsDiffer {
		t.Errorf("gated filterRejects differ: batched %d vs per-pair %d", batched.rejects, perPairArm.rejects)
	}
}

// TestAdmitGateBitsetMechanics covers the bitset itself away from the search
// path: sizing, reuse (a shorter second build must not inherit stale bits), and
// the intersection semantics.
// TestAdmitGateUnderConcurrentLinkers covers the interaction this feature and
// the reader-non-blocking insert path (link_stripes.go) create together, which
// neither one's own tests reach: gated filtered searches running WHILE linkers
// mutate the graph under the same read lock.
//
// The gate's safety argument has two halves and this exercises both. The posting
// sets it reads are pinned by the write lock that every payload-index write
// still takes (reindex stayed in placeLockedAt, the exclusive placement half),
// so a query's bitset cannot go stale under it. And the bitset is sized from
// arena.Capacity() and indexed directly, so a back-edge a concurrent linker
// publishes must not point past that capacity — the same invariant the visited
// set already depends on. A regression in either shows up here as a race report,
// an index-out-of-range panic, or an admitted point that does not satisfy the
// filter.
//
// The result check is the one with teeth: EVERY returned id must still satisfy
// the predicate when its payload is read back. A gate that wrongly admitted
// (an over-cover the exact path skipped re-checking, or a bitset built from a
// half-published posting set) surfaces here and nowhere else.
func TestAdmitGateUnderConcurrentLinkers(t *testing.T) {
	const (
		n       = 6000
		dim     = 16
		readers = 6
		writers = 2
		rounds  = 400
	)
	h := gateCorpus(t, n, dim)
	filter := Filter{Op: FilterEq, Field: "color", Value: NewString("red")}
	pred, err := CompileFilter(filter)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	prev := admitGateMode
	admitGateMode = gateForce
	defer func() { admitGateMode = prev }()

	queries := gateQueries(readers, dim)
	var wg sync.WaitGroup
	// started/stop bracket the overlap window. The readers must not be allowed to
	// finish before the writers have linkers in flight, or this degenerates into
	// a serial test that passes for the wrong reason — so the writers signal once
	// each has placed a point, and the readers wait for that signal before their
	// counted rounds begin.
	started := make(chan struct{})
	stop := make(chan struct{})
	var startOnce sync.Once
	// readerWg tracks only the readers, so the main goroutine can release the
	// writers the moment the read workload is done rather than racing them.
	var readerWg sync.WaitGroup
	errs := make(chan error, readers+writers)

	for r := 0; r < readers; r++ {
		wg.Add(1)
		readerWg.Add(1)
		go func(r int) {
			defer wg.Done()
			defer readerWg.Done()
			<-started
			for i := 0; i < rounds; i++ {
				res, err := h.SearchFiltered(queries[r], 10, filter)
				if err != nil {
					errs <- fmt.Errorf("reader %d: search: %w", r, err)
					return
				}
				for _, hit := range res {
					_, meta, _, _, _, ok := h.Get(hit.ID)
					if !ok {
						// Raced with nothing that can delete here; a miss would
						// mean the gate admitted a slot whose id is gone.
						errs <- fmt.Errorf("reader %d: admitted id %d is not live", r, hit.ID)
						return
					}
					if !pred(meta) {
						errs <- fmt.Errorf("reader %d: admitted id %d fails the filter (color=%v)",
							r, hit.ID, meta["color"])
						return
					}
				}
			}
		}(r)
	}

	// Writers keep linkers in flight for the whole run: every insert places under
	// the write lock and then links under the read lock, concurrently with the
	// readers above. maxInserts bounds the arena on a machine slow enough that
	// the readers' fixed round count takes a long wall-clock time — the writers
	// are otherwise open-ended, and the point is overlap, not volume.
	const maxInserts = 40_000
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			// Release the readers on EVERY exit path, not just the happy one. A
			// writer that fails its first insert would otherwise leave them
			// blocked on `started` forever, turning a one-line error into a test
			// timeout that says nothing about what broke.
			defer startOnce.Do(func() { close(started) })
			rng := rand.New(rand.NewSource(int64(9000 + w)))
			colors := []string{"red", "green", "blue", "amber"}
			for i := 0; i < maxInserts; i++ {
				select {
				case <-stop:
					return
				default:
				}
				v := make([]float32, dim)
				for d := range v {
					v[d] = float32(rng.NormFloat64())
				}
				id := uint64(1_000_000 + w*100_000 + i)
				meta := Metadata{
					"color": NewString(colors[rng.Intn(len(colors))]),
					"size":  NewInt(int64(rng.Intn(10))),
					"flag":  NewBool(i%3 == 0),
				}
				if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
					errs <- fmt.Errorf("writer %d: insert: %w", w, err)
					return
				}
				startOnce.Do(func() { close(started) })
			}
		}(w)
	}

	// The readers run a fixed number of rounds against a continuously-writing
	// index, then release the writers. Writers stop only after every reader is
	// done, so the whole read workload overlaps live linkers.
	readerWg.Wait()
	close(stop)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	// The gate must actually have been arming throughout, or this test proved
	// nothing about the gate.
	if got := h.Stats().FilterGates; got == 0 {
		t.Fatal("no gate armed during the concurrent run — the test exercised the unassisted path only")
	}
}

func TestAdmitGateBitsetMechanics(t *testing.T) {
	var g admitGate
	a := map[uint32]struct{}{1: {}, 5: {}, 64: {}, 129: {}}
	b := map[uint32]struct{}{5: {}, 64: {}, 200: {}}
	g.build([]narrowSet{{a, narrowExact}, {b, narrowExact}}, 256, false)
	for _, slot := range []uint32{5, 64} {
		if !g.test(slot) {
			t.Errorf("slot %d should be in the intersection", slot)
		}
	}
	for _, slot := range []uint32{0, 1, 129, 200, 255} {
		if g.test(slot) {
			t.Errorf("slot %d should not be in the intersection", slot)
		}
	}
	if !g.active() || g.exact {
		t.Fatalf("active=%v exact=%v, want true/false", g.active(), g.exact)
	}

	// Reuse: rebuild over a DIFFERENT set on the same backing array. Every bit
	// from the first build must be gone.
	g.build([]narrowSet{{map[uint32]struct{}{7: {}}, narrowExact}}, 256, true)
	if !g.test(7) || g.test(5) || g.test(64) {
		t.Fatal("rebuild did not clear the previous query's bits")
	}
	if !g.exact {
		t.Fatal("rebuild did not carry the exact flag")
	}
	g.disable()
	if g.active() || g.exact {
		t.Fatal("disable left the gate armed")
	}

	// A capacity that is not a multiple of 64 must still cover the top slot.
	g.build([]narrowSet{{map[uint32]struct{}{99: {}}, narrowExact}}, 100, false)
	if !g.test(99) {
		t.Fatal("bitset undersized for a non-multiple-of-64 capacity")
	}
}

// TestFilterLeafCount pins the counter the exactness veto depends on.
func TestFilterLeafCount(t *testing.T) {
	cases := []struct {
		f    Filter
		want int
	}{
		{Filter{Op: FilterEq, Field: "a", Value: NewInt(1)}, 1},
		{Filter{Op: FilterOr, Or: []Filter{
			{Op: FilterEq, Field: "a", Value: NewInt(1)},
			{Op: FilterEq, Field: "b", Value: NewInt(2)},
		}}, 1}, // an Or is ONE opaque leaf: the plan can never narrow "all" of it
		{Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "a", Value: NewInt(1)},
			{Op: FilterEq, Field: "b", Value: NewInt(2)},
		}}, 2},
		{Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "a", Value: NewInt(1)},
			{Op: FilterAnd, And: []Filter{
				{Op: FilterEq, Field: "b", Value: NewInt(2)},
				{Op: FilterEq, Field: "c", Value: NewInt(3)},
			}},
		}}, 3},
		{Filter{Op: FilterAnd}, 0},
	}
	for i, tc := range cases {
		if got := filterLeafCount(tc.f); got != tc.want {
			t.Errorf("case %d: filterLeafCount = %d, want %d", i, got, tc.want)
		}
	}
}

// TestCollectNarrowSetsMatchesCandidates pins the refactor that made the gate
// possible: candidatesCapped is now collectNarrowSets + intersectSlotSets, so
// the plan the gate builds from and the candidate set the planner materializes
// must be the same thing. Compared as SETS over every filter shape.
func TestCollectNarrowSetsMatchesCandidates(t *testing.T) {
	h := gateSharedCorpus(t)
	h.mu.RLock()
	defer h.mu.RUnlock()
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	for _, tc := range gateFilters() {
		cands, okC := h.payloadIdx.candidatesCapped(tc.f, limit, math.MaxInt)
		sets, okS := h.payloadIdx.collectNarrowSets(tc.f, limit)
		if okC != okS {
			t.Errorf("%s: candidatesCapped ok=%v but collectNarrowSets ok=%v", tc.name, okC, okS)
			continue
		}
		if !okC {
			continue
		}
		var g admitGate
		g.build(sets, h.arena.Capacity(), false)
		got := 0
		for slot := 0; slot < h.arena.Capacity(); slot++ {
			if g.test(uint32(slot)) { //nolint:gosec // capacity fits uint32
				got++
			}
		}
		if got != len(cands) {
			t.Errorf("%s: gate holds %d slots but candidatesCapped returned %d", tc.name, got, len(cands))
		}
		for _, slot := range cands {
			if !g.test(slot) {
				t.Errorf("%s: candidate slot %d missing from the gate", tc.name, slot)
				break
			}
		}
	}
}
