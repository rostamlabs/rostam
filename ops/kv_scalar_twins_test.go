// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/record"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// The scalar-key twins, cross-checked.
//
// ops/kvindex carries its own copies of vector's scalarKey/scalarKeyOf,
// numericValue and orderingHolds/orderingHoldsFloat. The duplication is
// deliberate — kvindex is engine-free and the KV write path calls it on every
// Put, so pulling vector in to share five fields would drag the whole engine
// into that path — and each side's comment names the other as its twin. What a
// comment cannot do is FAIL when they drift, and drift here is not cosmetic: if
// the index answers a comparison differently from the predicate, the candidate
// set stops being a superset of the matches and rows are lost silently.
//
// THE TWINS ARE UNEXPORTED ON BOTH SIDES, so this cannot be a call-for-call
// comparison; an export_test.go seam is visible only to its own package's
// tests, and neither package can reach into the other. It is therefore a
// BEHAVIOURAL cross-check, run from `ops` — the one package that imports both —
// over a value corpus:
//
//	kvindex side: post one key whose record field holds V, then ask
//	              Candidates for the selector (op, bound). Is the key in?
//	vector side:  compile the same leaf and evaluate it against
//	              Metadata{field: V}. Does it match?
//
// Membership and match must agree exactly. Every twin is on that path:
// scalarKeyOf decides what V is posted under and what an eq bound probes for,
// numericValue/numericKey decides which comparisons are numeric, and
// orderingHolds* decides every range answer.
//
// WHEN VECTOR REFUSES TO COMPILE, the requirement is one-directional instead:
// kvindex must yield no candidates. A bound the predicate cannot even be built
// for (an ordering against a bool, a list, a record) is a query error long
// before candidates matter, and the index producing keys for it would be a
// superset of nothing.

// kvTwinCase is one field value in the corpus: the vtypes.Value a record path
// resolves to, and the record bytes that produce it.
type kvTwinCase struct {
	name string
	val  vtypes.Value
	rec  []byte
}

// kvTwinField is the record field (and index path) every case in the corpus
// uses. Its spelling is irrelevant to the twins; it just has to be the same on
// both sides of the comparison.
const kvTwinField = "f"

// kvTwinFieldValues is the corpus of stored values.
//
// It is limited to the three kinds a record path can actually resolve to —
// record.CellValue maps every cell to ValueInt, ValueFloat or ValueString, and
// a #count to ValueInt — because a value that cannot be STORED cannot expose a
// disagreement about how it is INDEXED. ValueBool is therefore unreachable as a
// field here; it is covered from the bound side below, where a bool bound
// against every stored kind exercises the same scalarKey arm.
func kvTwinFieldValues(t *testing.T) []kvTwinCase {
	t.Helper()
	num := func(name string, v int64) kvTwinCase {
		return kvTwinCase{name, vtypes.NewInt(v), kvTwinIntRec(t, v)}
	}
	flt := func(name string, f float64) kvTwinCase {
		return kvTwinCase{name, vtypes.NewFloat(f), kvTwinFloatRec(t, f)}
	}
	str := func(name, s string) kvTwinCase {
		return kvTwinCase{name, vtypes.NewString(s), kvTwinStrRec(t, s)}
	}
	return []kvTwinCase{
		num("int zero", 0),
		num("int one", 1),
		num("int negative", -1),
		num("int max", math.MaxInt64),
		num("int min", math.MinInt64),
		flt("float zero", 0),
		flt("float one", 1),
		flt("float negative zero", math.Copysign(0, -1)),
		flt("float fractional", 1.5),
		flt("float max", math.MaxFloat64),
		flt("float smallest nonzero", math.SmallestNonzeroFloat64),
		flt("float +inf", math.Inf(1)),
		flt("float -inf", math.Inf(-1)),
		flt("float nan", math.NaN()),
		str("string empty", ""),
		str("string a", "a"),
		str("string ab", "ab"),
		str("string abc", "abc"),
		str("string B uppercase", "B"),
		str("string high byte", "\xff"),
	}
}

// kvTwinBounds is the corpus of filter bounds. It deliberately reaches OUTSIDE
// what a record can store — bools, lists, a geo point, the zero Value — because
// a bound is whatever the caller sent, and the interesting disagreements are
// exactly at the kind boundaries (an int posting against a float bound, a
// scalar posting against a bool bound).
func kvTwinBounds() []struct {
	name string
	val  vtypes.Value
} {
	return []struct {
		name string
		val  vtypes.Value
	}{
		{"int 0", vtypes.NewInt(0)},
		{"int 1", vtypes.NewInt(1)},
		{"int -1", vtypes.NewInt(-1)},
		{"int max", vtypes.NewInt(math.MaxInt64)},
		{"int min", vtypes.NewInt(math.MinInt64)},
		{"float 0", vtypes.NewFloat(0)},
		{"float 1", vtypes.NewFloat(1)},
		{"float 1.5", vtypes.NewFloat(1.5)},
		{"float -0", vtypes.NewFloat(math.Copysign(0, -1))},
		{"float +inf", vtypes.NewFloat(math.Inf(1))},
		{"float -inf", vtypes.NewFloat(math.Inf(-1))},
		{"float nan", vtypes.NewFloat(math.NaN())},
		{"string empty", vtypes.NewString("")},
		{"string a", vtypes.NewString("a")},
		{"string ab", vtypes.NewString("ab")},
		{"string abc", vtypes.NewString("abc")},
		{"string B uppercase", vtypes.NewString("B")},
		{"string high byte", vtypes.NewString("\xff")},
		{"bool true", vtypes.Value{Kind: vtypes.ValueBool, Bool: true}},
		{"bool false", vtypes.Value{Kind: vtypes.ValueBool, Bool: false}},
		{"none", vtypes.Value{}},
		{"ints list", vtypes.Value{Kind: vtypes.ValueInts, Ints: []int64{0, 1}}},
		{"strs list", vtypes.Value{Kind: vtypes.ValueStrings, Strs: []string{"a", "ab"}}},
		{"geo", vtypes.Value{Kind: vtypes.ValueGeo, Lat: 1, Lon: 2}},
	}
}

// TestKVIndexScalarTwinsAgreeWithVector is the cross-check itself: over every
// (stored value, bound, op) triple, the index's candidate decision and the
// predicate's match decision are the same decision.
func TestKVIndexScalarTwinsAgreeWithVector(t *testing.T) {
	filterOps := []vtypes.FilterOp{
		vtypes.FilterEq,
		vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte,
		vtypes.FilterIn,
	}
	for _, fc := range kvTwinFieldValues(t) {
		set, key := kvTwinSet(t, kvTwinField, fc.rec)
		// The precondition the whole comparison rests on: the bytes really do
		// resolve to the value the case claims. If Encode/CellValue ever stop
		// round-tripping one of these, the case is testing nothing.
		kvTwinAssertStored(t, fc)

		for _, bc := range kvTwinBounds() {
			for _, op := range filterOps {
				name := fmt.Sprintf("%s/%s/%s", fc.name, kvOpName(op), bc.name)
				t.Run(name, func(t *testing.T) {
					wantMatch, compiled := kvTwinVectorMatch(t, op, bc.val, fc.val)
					gotCandidate := kvTwinIsCandidate(t, set, key, op, bc.val)

					if !compiled {
						// vector refused the leaf outright. The index must then
						// produce nothing: there is no predicate for its
						// candidates to be a superset of.
						if gotCandidate {
							t.Fatalf("vector refuses to compile %s against %s, but kvindex still offers the key as a candidate",
								kvOpName(op), bc.name)
						}
						return
					}
					if gotCandidate != wantMatch {
						t.Fatalf("twins disagree: kvindex candidate=%v, vector match=%v (field %s = %+v, %s bound %s = %+v)",
							gotCandidate, wantMatch, fc.name, fc.val, kvOpName(op), bc.name, bc.val)
					}
				})
			}
		}
	}
}

// TestKVIndexCountTwinAgreesWithVector runs the same comparison for a #count
// definition, whose posting comes from record.ResultValue's Count arm rather
// than from CellValue — a different route to the same scalarKey, and one a
// definition can legally name.
func TestKVIndexCountTwinAgreesWithVector(t *testing.T) {
	for _, rows := range []int{0, 1, 3} {
		rec := kvTwinTableRec(t, rows)
		set, key := kvTwinSet(t, kvTwinField+"#count", rec)
		stored := vtypes.NewInt(int64(rows))
		for _, bound := range []vtypes.Value{vtypes.NewInt(0), vtypes.NewInt(1), vtypes.NewInt(3), vtypes.NewFloat(1)} {
			for _, op := range []vtypes.FilterOp{vtypes.FilterEq, vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte} {
				wantMatch, compiled := kvTwinVectorMatch(t, op, bound, stored)
				gotCandidate := kvTwinIsCandidate(t, set, key, op, bound)
				if !compiled {
					if gotCandidate {
						t.Fatalf("rows=%d: vector refuses %s but kvindex offers a candidate", rows, kvOpName(op))
					}
					continue
				}
				if gotCandidate != wantMatch {
					t.Fatalf("rows=%d, %s bound %+v: kvindex candidate=%v, vector match=%v",
						rows, kvOpName(op), bound, gotCandidate, wantMatch)
				}
			}
		}
	}
}

// kvOpName renders an op for a failure message. vtypes.FilterOp has no
// String method, only MarshalText, so a %s verb on it is a vet error and a %d
// would make a failure name a number nobody can read.
func kvOpName(op vtypes.FilterOp) string {
	if b, err := op.MarshalText(); err == nil {
		return string(b)
	}
	return fmt.Sprintf("op(%d)", uint8(op))
}

// --- the two sides ---------------------------------------------------------

// kvTwinIsCandidate asks the index whether key is a candidate for this leaf.
//
// An `in` leaf is given the bound as a one-element list where the bound is a
// scalar, mirroring what SelectorFor's expandKVInValues produces, and passed
// through unchanged when it already IS a list.
func kvTwinIsCandidate(t *testing.T, set *kvindex.Set, key string, op vtypes.FilterOp, bound vtypes.Value) bool {
	t.Helper()
	def, ok := set.Lookup("twin")
	if !ok {
		t.Fatal("the twin definition is not installed")
	}
	values := []vtypes.Value{bound}
	if op == vtypes.FilterIn {
		values = expandKVInValues(bound)
		if len(values) == 0 {
			// A non-list `in`: SelectorFor declines it and vector.compileIn
			// errors, so there is no candidate set to build. Reported as "not a
			// candidate", which is what the caller compares against.
			return false
		}
	}
	cands, err := set.Candidates(kvindex.Selector{Def: def, Op: op, Values: values}, nil, 1<<20)
	if err != nil {
		t.Fatalf("Candidates(%s): %v", kvOpName(op), err)
	}
	for _, c := range cands {
		if string(c) == key {
			return true
		}
	}
	return false
}

// kvTwinVectorMatch compiles the same leaf on the vector side and evaluates it
// against a metadata map holding the stored value. compiled=false means
// CompileFilter refused the leaf, which is a fact about the bound and not about
// the stored value.
func kvTwinVectorMatch(t *testing.T, op vtypes.FilterOp, bound, stored vtypes.Value) (match, compiled bool) {
	t.Helper()
	pred, err := vector.CompileFilter(vtypes.Filter{Op: op, Field: kvTwinField, Value: bound})
	if err != nil {
		return false, false
	}
	if pred == nil {
		t.Fatalf("CompileFilter(%s) returned no predicate and no error", kvOpName(op))
	}
	return pred(vector.Metadata{kvTwinField: stored}), true
}

// --- fixtures --------------------------------------------------------------

// kvTwinSet installs one definition on path and posts one key holding rec.
//
// MarkReady is the raw override its doc warns about, and it is right here:
// there is exactly one key and it has just been indexed, so the posting really
// does cover every live key — the condition a real backfill establishes by
// walking a cache this test does not need.
func kvTwinSet(t *testing.T, path string, rec []byte) (*kvindex.Set, string) {
	t.Helper()
	kind := uint8(wire.KVIndexKindScalar)
	if strings.HasSuffix(path, "#count") {
		kind = wire.KVIndexKindCount
	}
	def, err := kvindex.DefFrom(wire.KVIndexDef{
		Name: "twin", PayloadPath: path, Kind: kind, Enabled: true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom(%q): %v", path, err)
	}
	set := kvindex.New(0)
	set.Install([]kvindex.Def{def})
	const key = "k"
	set.Reindex([]byte(key), rec)
	set.MarkReady("twin")
	return set, key
}

// kvTwinAssertStored checks the precondition the whole comparison rests on:
// the fixture bytes really do resolve to the value the case claims, through
// the SAME record.ResultValue call both sides consult. Without it a broken
// encoder would make every triple agree trivially, on a value neither side
// ever saw.
func kvTwinAssertStored(t *testing.T, fc kvTwinCase) {
	t.Helper()
	p, err := record.ParsePath(kvTwinField)
	if err != nil {
		t.Fatalf("ParsePath(%q): %v", kvTwinField, err)
	}
	res, err := record.NewResolver(0).Resolve(fc.rec, p)
	if err != nil {
		t.Fatalf("%s: Resolve: %v", fc.name, err)
	}
	got, ok := record.ResultValue(res)
	if !ok {
		t.Fatalf("%s: the fixture record resolves to no value", fc.name)
	}
	if got.Kind != fc.val.Kind {
		t.Fatalf("%s: stored kind %v, case claims %v", fc.name, got.Kind, fc.val.Kind)
	}
	switch got.Kind {
	case vtypes.ValueInt:
		if got.Int != fc.val.Int {
			t.Fatalf("%s: stored int %d, case claims %d", fc.name, got.Int, fc.val.Int)
		}
	case vtypes.ValueString:
		if got.Str != fc.val.Str {
			t.Fatalf("%s: stored string %q, case claims %q", fc.name, got.Str, fc.val.Str)
		}
	case vtypes.ValueFloat:
		// Compared by BITS, not by ==, so -0 is checked rather than waved
		// through by 0 == -0: negative zero is one of the two float values the
		// twins treat specially. NaN is the other, and it is compared as
		// "both are NaN" instead, because a NaN's payload bits are not carried
		// through the encode/decode round trip (math.NaN's 0x…01 comes back as
		// the quiet 0x…00) and nothing in the index or the predicate reads
		// them: scalarKeyOf declines every NaN and orderingHoldsFloat answers
		// false for every NaN operand, whatever its payload.
		bothNaN := got.Flt != got.Flt && fc.val.Flt != fc.val.Flt
		if !bothNaN && math.Float64bits(got.Flt) != math.Float64bits(fc.val.Flt) {
			t.Fatalf("%s: stored float %v (bits %#x), case claims %v (bits %#x)",
				fc.name, got.Flt, math.Float64bits(got.Flt), fc.val.Flt, math.Float64bits(fc.val.Flt))
		}
	default:
		t.Fatalf("%s: unexpected stored kind %v", fc.name, got.Kind)
	}
}

// kvTwinIntRec, kvTwinFloatRec, kvTwinStrRec and kvTwinTableRec build the
// one-field dynamic-mode records the corpus stores. Dynamic mode keeps them
// readable: the field name rides in the record, so no schema has to be carried
// alongside.
func kvTwinIntRec(t *testing.T, v int64) []byte {
	t.Helper()
	return kvTwinRec(t, wire.Field{Name: kvTwinField, Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(v)}})
}

func kvTwinFloatRec(t *testing.T, f float64) []byte {
	t.Helper()
	return kvTwinRec(t, wire.Field{Name: kvTwinField, Cell: wire.Cell{Type: wire.OperateTypeF64, F: f}})
}

func kvTwinStrRec(t *testing.T, s string) []byte {
	t.Helper()
	return kvTwinRec(t, wire.Field{Name: kvTwinField, Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte(s)}})
}

func kvTwinTableRec(t *testing.T, rows int) []byte {
	t.Helper()
	tbl := &wire.Table{}
	for i := 0; i < rows; i++ {
		tbl.Rows = append(tbl.Rows, wire.Row{
			Key:  []byte{byte(i), 0, 0, 0, 0, 0, 0, 0},
			Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU16, U: uint64(i)}}},
		})
	}
	return kvTwinRec(t, wire.Field{Name: kvTwinField, Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: tbl})
}

func kvTwinRec(t *testing.T, f wire.Field) []byte {
	t.Helper()
	b := (&wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{f}}).Encode()
	if b == nil {
		t.Fatal("twin fixture record failed to encode")
	}
	return b
}
