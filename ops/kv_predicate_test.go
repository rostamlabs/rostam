// SPDX-License-Identifier: Apache-2.0

package ops

// Tests for the two pure pieces the kv_query leaf is built from: the filter
// rewrite that makes a KV record filter evaluable by the vector filter
// compiler (BuildKVPredicate), and the candidate-selector extraction that
// decides which leaf of that filter an index can drive (SelectorFor).
//
// Both are total functions of their input, so they are tested without a cache,
// a store or a dispatcher.

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// --- fixtures -------------------------------------------------------------

// kvRec encodes a two-field dynamic-mode record {rc: i64, tier: string}.
// Dynamic mode carries the field names in the record, so no schema has to be
// carried alongside the fixture. The fields are written in ascending name
// order, which is the order dynamic mode stores them in.
func kvRec(rc int64, tier string) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(rc)}},
		{Name: "tier", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte(tier)}},
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

// kvMeta is the synthesized metadata a kv_query predicate is evaluated
// against: the live value under the reserved record alias, and nothing else.
func kvMeta(rec []byte) vector.Metadata {
	return vector.Metadata{KVRecordAlias: {Kind: vtypes.ValueRecord, Rec: rec}}
}

// leaf builds a single-leaf filter.
func leaf(op vtypes.FilterOp, field string, v vtypes.Value) vtypes.Filter {
	return vtypes.Filter{Op: op, Field: field, Value: v}
}

func mustPredicate(t *testing.T, f vtypes.Filter) vector.Predicate {
	t.Helper()
	p, err := BuildKVPredicate(f)
	if err != nil {
		t.Fatalf("BuildKVPredicate(%+v): %v", f, err)
	}
	if p == nil {
		t.Fatalf("BuildKVPredicate(%+v): nil predicate for a non-empty filter", f)
	}
	return p
}

// --- BuildKVPredicate -----------------------------------------------------

func TestBuildKVPredicateRewritesFields(t *testing.T) {
	// A bare record path resolves against the record under the alias.
	p := mustPredicate(t, leaf(vtypes.FilterGt, "rc", vtypes.NewInt(100)))
	if !p(kvMeta(kvRec(200, "gold"))) {
		t.Fatal("gt 100 should be true at rc = 200")
	}
	if p(kvMeta(kvRec(50, "gold"))) {
		t.Fatal("gt 100 should be false at rc = 50")
	}

	// Nested and/or/not: every leaf at every depth must be rewritten, or the
	// unrewritten one silently evaluates false and the whole tree is wrong.
	nested := vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
		leaf(vtypes.FilterGte, "rc", vtypes.NewInt(10)),
		{Op: vtypes.FilterOr, Or: []vtypes.Filter{
			leaf(vtypes.FilterEq, "tier", vtypes.NewString("gold")),
			leaf(vtypes.FilterEq, "tier", vtypes.NewString("silver")),
		}},
		{Op: vtypes.FilterNot, Not: ptrFilter(leaf(vtypes.FilterEq, "rc", vtypes.NewInt(13)))},
	}}
	np := mustPredicate(t, nested)
	for _, tc := range []struct {
		rc   int64
		tier string
		want bool
	}{
		{20, "gold", true},
		{20, "silver", true},
		{20, "bronze", false}, // fails the or
		{5, "gold", false},    // fails the gte
		{13, "gold", false},   // caught by the not
	} {
		if got := np(kvMeta(kvRec(tc.rc, tc.tier))); got != tc.want {
			t.Errorf("nested predicate at rc=%d tier=%q: got %v, want %v", tc.rc, tc.tier, got, tc.want)
		}
	}
}

func TestBuildKVPredicateDoesNotMutateItsInput(t *testing.T) {
	// The caller's filter is also what SelectorFor reads, and the selector is
	// matched against the definition's BARE path. A rewrite in place would
	// leave every field prefixed and no selector would ever match again.
	f := vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
		leaf(vtypes.FilterEq, "rc", vtypes.NewInt(1)),
		{Op: vtypes.FilterNot, Not: ptrFilter(leaf(vtypes.FilterEq, "tier", vtypes.NewString("x")))},
	}}
	if _, err := BuildKVPredicate(f); err != nil {
		t.Fatalf("BuildKVPredicate: %v", err)
	}
	if f.And[0].Field != "rc" {
		t.Fatalf("input leaf rewritten in place: field is %q, want %q", f.And[0].Field, "rc")
	}
	if f.And[1].Not.Field != "tier" {
		t.Fatalf("input not-child rewritten in place: field is %q, want %q", f.And[1].Not.Field, "tier")
	}
}

func TestBuildKVPredicateRejectsBadFields(t *testing.T) {
	for _, tc := range []struct {
		name  string
		field string
	}{
		{"reserved alias", KVRecordAlias},
		{"already prefixed", KVRecordAlias + "/rc"},
		{"other reserved", "$content"},
		{"too many segments", "a/b/c/d"},
		{"empty", ""},
		{"empty segment", "a//b"},
		{"count not last", "a#count/b"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildKVPredicate(leaf(vtypes.FilterEq, tc.field, vtypes.NewInt(1)))
			if err == nil {
				t.Fatalf("BuildKVPredicate(field %q): want an error, got nil", tc.field)
			}
			// The MARKER, not just an error: the coordinator classifies a
			// permanent client error by it, and an unmarked one is retried
			// across every replica before failing identically.
			if !errors.Is(err, ErrKVQueryFilter) {
				t.Fatalf("BuildKVPredicate(field %q): %v is not marked ErrKVQueryFilter", tc.field, err)
			}
		})
	}
}

func TestBuildKVPredicateEmptyFilterIsNil(t *testing.T) {
	// A zero filter means "match all", and the leaf skips evaluation entirely
	// for a nil predicate — the same contract vector.CompileFilter has.
	p, err := BuildKVPredicate(vtypes.Filter{})
	if err != nil {
		t.Fatalf("BuildKVPredicate(zero): %v", err)
	}
	if p != nil {
		t.Fatal("a zero filter must compile to a nil predicate")
	}
}

func TestBuildKVPredicateRefusesAnOversizeTree(t *testing.T) {
	// The wire decoder checks this too, but BuildKVPredicate is exported and
	// the coordinator can reach it with a filter that never went through the
	// args decoder, so it re-checks rather than trusting its caller.
	deep := leaf(vtypes.FilterEq, "rc", vtypes.NewInt(1))
	for i := 0; i < wire.KVQueryMaxFilterDepth+2; i++ {
		deep = vtypes.Filter{Op: vtypes.FilterNot, Not: ptrFilter(deep)}
	}
	_, err := BuildKVPredicate(deep)
	if err == nil {
		t.Fatal("a filter deeper than KVQueryMaxFilterDepth must be refused")
	}
	if !errors.Is(err, ErrKVQueryFilter) {
		t.Fatalf("an over-budget tree must be marked ErrKVQueryFilter, got %v", err)
	}
	// And the inner sentinel stays inspectable through the wrap.
	if !errors.Is(err, wire.ErrKVFilterBudget) {
		t.Fatalf("the budget error must stay inspectable through the wrap, got %v", err)
	}
}

func TestBuildKVPredicateMarksCompilerRejections(t *testing.T) {
	// vector.CompileFilter's own rejections are permanent client errors too, so
	// they carry the same marker. An `in` whose value is not a list is the
	// cheapest one to reach.
	for _, tc := range []struct {
		name string
		f    vtypes.Filter
	}{
		{"in over a non-list", leaf(vtypes.FilterIn, "rc", vtypes.NewInt(1))},
		{"unknown op", vtypes.Filter{Op: vtypes.FilterOp(250), Field: "rc"}},
		{"not with no child", vtypes.Filter{Op: vtypes.FilterNot}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildKVPredicate(tc.f)
			if err == nil {
				t.Fatalf("BuildKVPredicate(%+v): want an error, got nil", tc.f)
			}
			if !errors.Is(err, ErrKVQueryFilter) {
				t.Fatalf("%v is not marked ErrKVQueryFilter", err)
			}
		})
	}
}

func TestBuildKVPredicateNeverSeesTheRawRecord(t *testing.T) {
	// The alias holds a ValueRecord. If a filter could address it directly it
	// would compare against the whole blob — so the alias itself, and every
	// other reserved "$" name, is refused at compile time. The check is on the
	// leading byte, so this holds for names this build has never heard of.
	for _, field := range []string{"$rec", "$anything", "$"} {
		if _, err := BuildKVPredicate(leaf(vtypes.FilterEq, field, vtypes.NewInt(1))); err == nil {
			t.Fatalf("field %q must be refused", field)
		}
	}
}

func ptrFilter(f vtypes.Filter) *vtypes.Filter { return &f }

// --- SelectorFor ----------------------------------------------------------

func TestSelectorFor(t *testing.T) {
	d := wiringDef(t, "rc") // PathText == "rc", prefix "u:"

	eqRC := leaf(vtypes.FilterEq, "rc", vtypes.NewInt(7))
	gtRC := leaf(vtypes.FilterGt, "rc", vtypes.NewInt(7))
	gteRC := leaf(vtypes.FilterGte, "rc", vtypes.NewInt(7))
	eqOther := leaf(vtypes.FilterEq, "tier", vtypes.NewString("gold"))
	gtOther := leaf(vtypes.FilterGt, "tier", vtypes.NewString("gold"))

	for _, tc := range []struct {
		name   string
		filter vtypes.Filter
		wantOK bool
		wantOp vtypes.FilterOp
	}{
		{"bare eq on the path", eqRC, true, vtypes.FilterEq},
		{"bare range on the path", gtRC, true, vtypes.FilterGt},
		{"bare leaf on another path", eqOther, false, 0},
		{"and: eq first", vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{eqRC, gtOther}}, true, vtypes.FilterEq},
		{"and: range after an unindexed leaf", vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{gtOther, gteRC}}, true, vtypes.FilterGte},
		{"and: eq preferred over a range", vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{gtRC, eqRC}}, true, vtypes.FilterEq},
		{"or is never a selector", vtypes.Filter{Op: vtypes.FilterOr, Or: []vtypes.Filter{eqRC, eqOther}}, false, 0},
		{"not is never a selector", vtypes.Filter{Op: vtypes.FilterNot, Not: ptrFilter(eqRC)}, false, 0},
		{"and does not descend into or", vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
			{Op: vtypes.FilterOr, Or: []vtypes.Filter{eqRC}},
		}}, false, 0},
		{"and does not descend into not", vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
			{Op: vtypes.FilterNot, Not: ptrFilter(eqRC)},
		}}, false, 0},
		{"ne is not positive", leaf(vtypes.FilterNe, "rc", vtypes.NewInt(7)), false, 0},
		{"is_empty is not positive", leaf(vtypes.FilterIsEmpty, "rc", vtypes.Value{}), false, 0},
		{"contains is not a scalar equality", leaf(vtypes.FilterContains, "rc", vtypes.NewInt(7)), false, 0},
		{"regex is not positive", leaf(vtypes.FilterRegex, "rc", vtypes.NewString("^7$")), false, 0},
		{"eq with no value", leaf(vtypes.FilterEq, "rc", vtypes.Value{}), false, 0},
		{"nested and is not flattened", vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
			{Op: vtypes.FilterAnd, And: []vtypes.Filter{eqRC}},
		}}, false, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sel, ok := SelectorFor(tc.filter, d)
			if ok != tc.wantOK {
				t.Fatalf("SelectorFor: ok = %v, want %v (selector %+v)", ok, tc.wantOK, sel)
			}
			if !ok {
				return
			}
			if sel.Op != tc.wantOp {
				t.Fatalf("SelectorFor: op = %v, want %v", sel.Op, tc.wantOp)
			}
			if sel.Def.Name != d.Name {
				t.Fatalf("SelectorFor: def = %q, want %q", sel.Def.Name, d.Name)
			}
		})
	}
}

func TestSelectorForInExpandsItsList(t *testing.T) {
	d := wiringDef(t, "rc")
	f := leaf(vtypes.FilterIn, "rc", vtypes.NewInts([]int64{3, 5, 7}))
	sel, ok := SelectorFor(f, d)
	if !ok {
		t.Fatal("an `in` on the indexed path must be a selector")
	}
	if sel.Op != vtypes.FilterIn {
		t.Fatalf("op = %v, want in", sel.Op)
	}
	// kvindex.Selector carries N SCALAR values for `in`, never one list value:
	// its posting lookup keys on a scalar.
	if len(sel.Values) != 3 {
		t.Fatalf("values = %+v, want 3 scalars", sel.Values)
	}
	for i, want := range []int64{3, 5, 7} {
		if sel.Values[i].Kind != vtypes.ValueInt || sel.Values[i].Int != want {
			t.Fatalf("value %d = %+v, want int %d", i, sel.Values[i], want)
		}
	}

	// An `in` with no list value narrows nothing and must be declined rather
	// than turned into a selector with an empty value set.
	if _, ok := SelectorFor(leaf(vtypes.FilterIn, "rc", vtypes.NewInt(3)), d); ok {
		t.Fatal("an `in` whose value is not a list must not be a selector")
	}
	if _, ok := SelectorFor(leaf(vtypes.FilterIn, "rc", vtypes.NewInts(nil)), d); ok {
		t.Fatal("an `in` over an empty list must not be a selector")
	}
}

func TestSelectorForAcceptsOnlyScalarBounds(t *testing.T) {
	d := wiringDef(t, "rc")
	// A record path resolves to a scalar or to nothing, so only a scalar bound
	// has a posting-set proof behind it. Declining the rest is always safe: it
	// costs a scan-required refusal, never a lost row.
	for _, v := range []vtypes.Value{
		{}, // ValueNone
		vtypes.NewStrings([]string{"a"}),
		vtypes.NewFloats([]float64{1}),
	} {
		if sel, ok := SelectorFor(leaf(vtypes.FilterEq, "rc", v), d); ok {
			t.Fatalf("eq against kind %d must not be a selector (got %+v)", v.Kind, sel)
		}
	}
	for _, v := range []vtypes.Value{
		vtypes.NewInt(1), vtypes.NewFloat(1.5), vtypes.NewString("x"), vtypes.NewBool(true),
	} {
		if _, ok := SelectorFor(leaf(vtypes.FilterEq, "rc", v), d); !ok {
			t.Fatalf("eq against scalar kind %d must be a selector", v.Kind)
		}
	}
}

func TestSelectorForMatchesTheDefinitionsPathExactly(t *testing.T) {
	// A count definition's path is "b#count"; a filter leaf on "b" addresses a
	// different thing and must not drive it.
	d := wiringDef(t, "rc")
	if _, ok := SelectorFor(leaf(vtypes.FilterEq, "rc#count", vtypes.NewInt(1)), d); ok {
		t.Fatal("a #count leaf must not select a scalar definition on the same field")
	}
	cd, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        "by-rows",
		KeyPrefix:   []byte("u:"),
		PayloadPath: "rows#count",
		Kind:        wire.KVIndexKindCount,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom(count): %v", err)
	}
	if _, ok := SelectorFor(leaf(vtypes.FilterEq, "rows", vtypes.NewInt(1)), cd); ok {
		t.Fatal("a scalar leaf must not select a #count definition on the same field")
	}
	if _, ok := SelectorFor(leaf(vtypes.FilterEq, "rows#count", vtypes.NewInt(1)), cd); !ok {
		t.Fatal("a #count leaf must select the #count definition")
	}
}
