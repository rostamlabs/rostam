// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/record"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- small fixtures -------------------------------------------------------

// dynRec encodes a one-field dynamic-mode record. Dynamic mode keeps the
// fixtures readable: the field name is in the record, so no schema has to be
// carried alongside it.
func dynRec(name string, c wire.Cell) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{{Name: name, Cell: c}}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

func intRec(name string, v int64) []byte {
	return dynRec(name, wire.Cell{Type: wire.OperateTypeI64, U: su64(v)})
}

func floatRec(name string, f float64) []byte {
	return dynRec(name, wire.Cell{Type: wire.OperateTypeF64, F: f})
}

func strRec(name, s string) []byte {
	return dynRec(name, wire.Cell{Type: wire.OperateTypeBytes, B: []byte(s)})
}

func tableRec(name string, rows int) []byte {
	t := &wire.Table{}
	for i := 0; i < rows; i++ {
		key := make([]byte, 4)
		key[0] = byte(i)
		t.Rows = append(t.Rows, wire.Row{
			Key:  key,
			Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU16, U: uint64(i)}}},
		})
	}
	rec := &wire.Record{
		Mode:   wire.OperateModeDynamic,
		Fields: []wire.Field{{Name: name, Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: t}},
	}
	b := rec.Encode()
	if b == nil {
		panic("fixture table record failed to encode")
	}
	return b
}

// mustDef builds a Def or fails the test.
func mustDef(t *testing.T, name, prefix, path string, kind uint8) Def {
	t.Helper()
	d, err := DefFrom(wire.KVIndexDef{
		Name:        name,
		KeyPrefix:   []byte(prefix),
		PayloadPath: path,
		Kind:        kind,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom(%q, %q): %v", name, path, err)
	}
	return d
}

// readySet installs defs and marks them all ready, which is what a completed
// backfill does. Tests that care about readiness itself drive it explicitly.
func readySet(defs ...Def) *Set {
	s := New(1024)
	s.Install(defs)
	for _, d := range defs {
		s.MarkReady(d.Name)
	}
	return s
}

func keyStrings(keys [][]byte) []string {
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, string(k))
	}
	return out
}

func mustCandidates(t *testing.T, s *Set, sel Selector, after []byte, budget int) []string {
	t.Helper()
	got, err := s.Candidates(sel, after, budget)
	if err != nil {
		t.Fatalf("Candidates: unexpected error: %v", err)
	}
	return keyStrings(got)
}

const bigBudget = 1 << 20

// --- reindex / drop -------------------------------------------------------

func TestReindexAndCandidatesEq(t *testing.T) {
	d := mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)
	s := readySet(d)

	s.Reindex([]byte("u:a"), intRec("rc", 7))
	s.Reindex([]byte("u:b"), intRec("rc", 7))
	s.Reindex([]byte("u:c"), intRec("rc", 9))

	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(7)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"u:a", "u:b"}) {
		t.Fatalf("eq 7: got %v, want [u:a u:b]", got)
	}

	// `in` unions the per-value sets, and a repeated value must not duplicate
	// a key.
	sel.Op = vtypes.FilterIn
	sel.Values = []vtypes.Value{vtypes.NewInt(9), vtypes.NewInt(7), vtypes.NewInt(9)}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"u:a", "u:b", "u:c"}) {
		t.Fatalf("in {7,9}: got %v", got)
	}

	// A value nothing was posted under is an empty answer, not an error.
	sel.Op = vtypes.FilterEq
	sel.Values = []vtypes.Value{vtypes.NewInt(1234)}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 0 {
		t.Fatalf("eq 1234: got %v, want none", got)
	}

	// eq is KIND-STRICT, exactly as vtypes.Value.Equal is: an int 7 posting is
	// not reachable by a float 7 bound (the predicate would reject it too).
	sel.Values = []vtypes.Value{vtypes.NewFloat(7)}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 0 {
		t.Fatalf("eq float 7 against int postings: got %v, want none", got)
	}
}

func TestReindexReplacesOldPosting(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)

	s.Reindex([]byte("k"), intRec("rc", 1))
	s.Reindex([]byte("k"), intRec("rc", 2))

	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 0 {
		t.Fatalf("old value still posted: %v", got)
	}
	sel.Values = []vtypes.Value{vtypes.NewInt(2)}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"k"}) {
		t.Fatalf("new value not posted: %v", got)
	}

	// No EMPTY leftovers: the vacated scalar key must be gone from vals, not
	// left behind as an empty set. An empty set is invisible to a query but
	// makes the range walk (which is O(distinct values)) grow without bound.
	keys, distinct := s.Stats("by-rc")
	if keys != 1 || distinct != 1 {
		t.Fatalf("Stats after replace = (%d keys, %d distinct), want (1, 1)", keys, distinct)
	}
}

func TestDropRemovesFromEveryDef(t *testing.T) {
	a := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	b := mustDef(t, "by-rc2", "", "rc", wire.KVIndexKindScalar)
	s := readySet(a, b)

	s.Reindex([]byte("k"), intRec("rc", 5))
	s.Reindex([]byte("j"), intRec("rc", 5))
	s.Drop([]byte("k"))

	for _, d := range []Def{a, b} {
		sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(5)}}
		if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"j"}) {
			t.Fatalf("%s after Drop: got %v, want [j]", d.Name, got)
		}
		keys, distinct := s.Stats(d.Name)
		if keys != 1 || distinct != 1 {
			t.Fatalf("%s Stats = (%d, %d), want (1, 1)", d.Name, keys, distinct)
		}
	}

	// Dropping a key that was never indexed is a no-op, not a panic.
	s.Drop([]byte("never-seen"))
}

func TestNonRecordValueHasNoPosting(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)

	truncated := intRec("rc", 3)
	truncated = truncated[:len(truncated)-1]

	for _, v := range [][]byte{[]byte("plain"), nil, {}, truncated, {wire.OperateModeSchema}} {
		s.Reindex([]byte("k"), v)
		if keys, distinct := s.Stats("by-rc"); keys != 0 || distinct != 0 {
			t.Fatalf("value %q left a posting: (%d keys, %d distinct)", v, keys, distinct)
		}
	}

	// And a non-record OVERWRITING a good record must clear the old posting,
	// not leave the stale one behind under a key whose value no longer
	// resolves. (A stale posting is only ever a wasted lookup, but a posting
	// that outlives every rewrite of its key is an unbounded leak.)
	s.Reindex([]byte("k"), intRec("rc", 3))
	if keys, _ := s.Stats("by-rc"); keys != 1 {
		t.Fatalf("good record not posted (%d keys)", keys)
	}
	s.Reindex([]byte("k"), []byte("plain"))
	if keys, distinct := s.Stats("by-rc"); keys != 0 || distinct != 0 {
		t.Fatalf("non-record overwrite left (%d keys, %d distinct), want (0, 0)", keys, distinct)
	}
}

func TestNaNIsNotPosted(t *testing.T) {
	d := mustDef(t, "by-fl", "", "fl", wire.KVIndexKindScalar)
	s := readySet(d)

	s.Reindex([]byte("nan"), floatRec("fl", math.NaN()))
	if keys, distinct := s.Stats("by-fl"); keys != 0 || distinct != 0 {
		t.Fatalf("NaN was posted: (%d keys, %d distinct)", keys, distinct)
	}

	// Repeating it must not grow anything: a NaN scalarKey never compares
	// equal to itself, so a posted NaN would append a new unreachable entry on
	// every write (the vector/payload_index.go leak this decline exists for).
	for i := 0; i < 10; i++ {
		s.Reindex([]byte(fmt.Sprintf("nan%d", i)), floatRec("fl", math.NaN()))
	}
	if keys, distinct := s.Stats("by-fl"); keys != 0 || distinct != 0 {
		t.Fatalf("repeated NaN grew the index: (%d keys, %d distinct)", keys, distinct)
	}

	// A NaN BOUND selects nothing, because nothing satisfies it.
	for _, op := range selectorOps {
		sel := Selector{Def: d, Op: op, Values: []vtypes.Value{vtypes.NewFloat(math.NaN())}}
		if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 0 {
			t.Fatalf("NaN bound with op %v selected %v", op, got)
		}
	}
}

func TestKeyOutsidePrefixIsNotIndexed(t *testing.T) {
	d := mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)
	s := readySet(d)

	s.Reindex([]byte("u:in"), intRec("rc", 4))
	s.Reindex([]byte("o:out"), intRec("rc", 4))
	s.Reindex([]byte("u"), intRec("rc", 4)) // a proper prefix of the prefix
	s.Reindex([]byte(""), intRec("rc", 4))

	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(4)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"u:in"}) {
		t.Fatalf("prefix scoping broken: got %v, want [u:in]", got)
	}
	if keys, _ := s.Stats("by-rc"); keys != 1 {
		t.Fatalf("out-of-prefix keys were stored: %d keys", keys)
	}

	// An empty prefix covers the whole keyspace.
	all := mustDef(t, "all", "", "rc", wire.KVIndexKindScalar)
	s2 := readySet(all)
	s2.Reindex([]byte("u:in"), intRec("rc", 4))
	s2.Reindex([]byte("o:out"), intRec("rc", 4))
	sel = Selector{Def: all, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(4)}}
	if got := mustCandidates(t, s2, sel, nil, bigBudget); !equalStrings(got, []string{"o:out", "u:in"}) {
		t.Fatalf("empty prefix: got %v", got)
	}
}

// --- ranges ---------------------------------------------------------------

func TestCandidatesRange(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	for i := 1; i <= 20; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%02d", i)), intRec("rc", int64(i)))
	}

	sel := Selector{Def: d, Op: vtypes.FilterGt, Values: []vtypes.Value{vtypes.NewInt(15)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 5 {
		t.Fatalf("gt 15: got %d keys (%v), want 5", len(got), got)
	}

	// CROSS-KIND: a float bound over integer postings, compared as float64 —
	// the compileOrdering semantics. 1, 2 and 3 are <= 3.0, and the int 3 is
	// among them even though its scalar key is int-kinded, not float-kinded.
	sel.Op = vtypes.FilterLte
	sel.Values = []vtypes.Value{vtypes.NewFloat(3.0)}
	got := mustCandidates(t, s, sel, nil, bigBudget)
	if !equalStrings(got, []string{"k01", "k02", "k03"}) {
		t.Fatalf("lte 3.0: got %v, want [k01 k02 k03]", got)
	}

	// And the mirror: a float-kinded posting reachable by an integer bound.
	s.Reindex([]byte("f"), floatRec("rc", 2.5))
	sel.Values = []vtypes.Value{vtypes.NewInt(3)}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"f", "k01", "k02", "k03"}) {
		t.Fatalf("lte int 3 with a float posting: got %v", got)
	}

	// A STRING bound admits string postings only, lexicographically, and never
	// a numeric one (compileOrdering's string path needs both sides string).
	sd := mustDef(t, "by-sc", "", "sc", wire.KVIndexKindScalar)
	ss := readySet(sd)
	for _, w := range []string{"alpha", "beta", "gamma"} {
		ss.Reindex([]byte("s-"+w), strRec("sc", w))
	}
	ss.Reindex([]byte("n"), intRec("sc", 5))
	ssel := Selector{Def: sd, Op: vtypes.FilterGte, Values: []vtypes.Value{vtypes.NewString("beta")}}
	if g := mustCandidates(t, ss, ssel, nil, bigBudget); !equalStrings(g, []string{"s-beta", "s-gamma"}) {
		t.Fatalf("gte \"beta\": got %v", g)
	}
	ssel.Values = []vtypes.Value{vtypes.NewInt(0)}
	if g := mustCandidates(t, ss, ssel, nil, bigBudget); !equalStrings(g, []string{"n"}) {
		t.Fatalf("numeric bound over mixed postings: got %v, want [n]", g)
	}

	// A bound of a kind no ordering can use selects nothing.
	ssel.Values = []vtypes.Value{vtypes.NewBool(true)}
	if g := mustCandidates(t, ss, ssel, nil, bigBudget); len(g) != 0 {
		t.Fatalf("bool bound selected %v", g)
	}

	// A "#count" definition posts an integer and ranges over it like any other.
	cd := mustDef(t, "by-cnt", "", "tb#count", wire.KVIndexKindCount)
	cs := readySet(cd)
	for i := 0; i <= 4; i++ {
		cs.Reindex([]byte(fmt.Sprintf("c%d", i)), tableRec("tb", i))
	}
	csel := Selector{Def: cd, Op: vtypes.FilterGte, Values: []vtypes.Value{vtypes.NewInt(3)}}
	if g := mustCandidates(t, cs, csel, nil, bigBudget); !equalStrings(g, []string{"c3", "c4"}) {
		t.Fatalf("#count gte 3: got %v", g)
	}
}

// --- cursor and budget ----------------------------------------------------

func TestCandidatesAfterCursor(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	for i := 0; i < 6; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%d", i)), intRec("rc", 1))
	}

	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	all := mustCandidates(t, s, sel, nil, bigBudget)
	if len(all) != 6 {
		t.Fatalf("no cursor: got %v", all)
	}
	// The cursor is EXCLUSIVE: the key it names is never returned again.
	got := mustCandidates(t, s, sel, []byte("k2"), bigBudget)
	if !equalStrings(got, []string{"k3", "k4", "k5"}) {
		t.Fatalf("after k2: got %v", got)
	}
	// A cursor past the end empties the page rather than wrapping.
	if got := mustCandidates(t, s, sel, []byte("z"), bigBudget); len(got) != 0 {
		t.Fatalf("after z: got %v", got)
	}
	// A cursor that is not itself a key still splits the keyspace correctly.
	if got := mustCandidates(t, s, sel, []byte("k2!"), bigBudget); !equalStrings(got, []string{"k3", "k4", "k5"}) {
		t.Fatalf("after k2!: got %v", got)
	}
	// Output is key-ordered, so paging over it terminates.
	if !sort.StringsAreSorted(all) {
		t.Fatalf("Candidates output is not key-ordered: %v", all)
	}
}

func TestCandidatesBudget(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	for i := 0; i < 10; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%d", i)), intRec("rc", 1))
	}
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}

	got, err := s.Candidates(sel, nil, 5)
	if !errors.Is(err, ErrCandidateBudget) {
		t.Fatalf("budget 5 over 10 keys: err = %v, want ErrCandidateBudget", err)
	}
	// NEVER a truncated set: the refusal must come before a single key is
	// copied, so a caller cannot mistake a partial page for a complete one.
	if got != nil {
		t.Fatalf("budget refusal returned %d keys, want none", len(got))
	}

	// Exactly at the budget is fine.
	if g := mustCandidates(t, s, sel, nil, 10); len(g) != 10 {
		t.Fatalf("budget 10 over 10 keys: got %d", len(g))
	}

	// The cursor does NOT lower the cost: keys below it were still examined,
	// so a set too large to examine is refused on every page, not silently on
	// the first one only.
	if _, err := s.Candidates(sel, []byte("k8"), 5); !errors.Is(err, ErrCandidateBudget) {
		t.Fatalf("budget with a cursor: err = %v, want ErrCandidateBudget", err)
	}

	// A range walk charges every key it unions, across distinct values.
	for i := 0; i < 10; i++ {
		s.Reindex([]byte(fmt.Sprintf("j%d", i)), intRec("rc", int64(i)))
	}
	rsel := Selector{Def: d, Op: vtypes.FilterGte, Values: []vtypes.Value{vtypes.NewInt(0)}}
	if _, err := s.Candidates(rsel, nil, 5); !errors.Is(err, ErrCandidateBudget) {
		t.Fatalf("range budget: err = %v, want ErrCandidateBudget", err)
	}
}

func TestCandidatesUnknownAndBuildingIndexes(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := New(1024)
	s.Install([]Def{d})

	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if _, err := s.Candidates(sel, nil, bigBudget); !errors.Is(err, ErrIndexBuilding) {
		t.Fatalf("not-ready index: err = %v, want ErrIndexBuilding", err)
	}
	s.MarkReady("by-rc")
	if _, err := s.Candidates(sel, nil, bigBudget); err != nil {
		t.Fatalf("ready index: %v", err)
	}

	missing := sel
	missing.Def.Name = "nope"
	if _, err := s.Candidates(missing, nil, bigBudget); !errors.Is(err, ErrNoSuchIndex) {
		t.Fatalf("unknown index: err = %v, want ErrNoSuchIndex", err)
	}
	if s.IsReady("nope") {
		t.Fatal("IsReady lied about an unknown index")
	}
	if keys, distinct := s.Stats("nope"); keys != 0 || distinct != 0 {
		t.Fatalf("Stats on an unknown index = (%d, %d)", keys, distinct)
	}

	// A non-positive leaf op is never a candidate selector: refusing is the
	// only safe answer, since a `ne`/`not` posting set is not a superset of
	// anything.
	bad := sel
	bad.Op = vtypes.FilterNe
	if _, err := s.Candidates(bad, nil, bigBudget); err == nil {
		t.Fatal("a 'ne' selector was accepted")
	}
}

// --- the oracle property --------------------------------------------------

func TestCandidatesMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(20260910))
	defs := oracleDefs()

	spaces := 200
	if testing.Short() {
		spaces = 25
	}
	var totalSel, totalGot, totalWant, totalExtra int

	for iter := 0; iter < spaces; iter++ {
		ks := randomKeyspace(rng)
		bi := buildBruteIndex(defs, ks)

		s := New(1024)
		s.Install(defs)
		_ = s.Rebuild(walkOf(ks))

		for q := 0; q < 10; q++ {
			sel := randomSelector(rng, defs)
			var after []byte
			if rng.Intn(3) == 0 {
				after = []byte(ks.keys[rng.Intn(len(ks.keys))])
			}

			got, err := s.Candidates(sel, after, bigBudget)
			if err != nil {
				t.Fatalf("iter %d q %d: Candidates: %v", iter, q, err)
			}
			want := bruteCandidates(bi, sel, after)
			totalSel++
			totalGot += len(got)
			totalWant += len(want)
			totalExtra += len(got) - len(want)

			gotSet := make(map[string]struct{}, len(got))
			for i, k := range got {
				ks2 := string(k)
				if _, dup := gotSet[ks2]; dup {
					t.Fatalf("iter %d q %d: duplicate candidate %q", iter, q, ks2)
				}
				gotSet[ks2] = struct{}{}
				// SUBSET of the live keys, within the prefix, above the cursor.
				if !ks.live(ks2) {
					t.Fatalf("iter %d q %d: candidate %q is not a live key", iter, q, ks2)
				}
				if !bytes.HasPrefix(k, sel.Def.Prefix) {
					t.Fatalf("iter %d q %d: candidate %q is outside prefix %q", iter, q, ks2, sel.Def.Prefix)
				}
				if len(after) > 0 && ks2 <= string(after) {
					t.Fatalf("iter %d q %d: candidate %q is not above the cursor %q", iter, q, ks2, after)
				}
				if i > 0 && bytes.Compare(got[i-1], k) >= 0 {
					t.Fatalf("iter %d q %d: candidates are not key-ordered at %d", iter, q, i)
				}
			}
			// SUPERSET of the true matches.
			for _, w := range want {
				if _, ok := gotSet[string(w)]; !ok {
					t.Fatalf("iter %d q %d: op %v def %q lost key %q (true match missing from candidates)",
						iter, q, sel.Op, sel.Def.Name, w)
				}
			}
		}
	}
	t.Logf("oracle: %d keyspaces x 10 selectors = %d selectors; %d candidates for %d true matches (%d extra)",
		spaces, totalSel, totalGot, totalWant, totalExtra)
}

// --- definitions ----------------------------------------------------------

func TestDefFromValidates(t *testing.T) {
	ok, err := DefFrom(wire.KVIndexDef{
		Name: "good", KeyPrefix: []byte("u:"), PayloadPath: "rc",
		Kind: wire.KVIndexKindScalar, Enabled: true,
	}, 42)
	if err != nil {
		t.Fatalf("valid definition rejected: %v", err)
	}
	if ok.Name != "good" || ok.PathText != "rc" || ok.MetaIndex != 42 || string(ok.Prefix) != "u:" {
		t.Fatalf("Def = %+v", ok)
	}
	if len(ok.Path.Segs) != 1 || ok.Path.Segs[0].Kind != record.SegField || ok.Path.Segs[0].Name != "rc" {
		t.Fatalf("Path = %+v", ok.Path)
	}

	// The prefix is COPIED: a caller reusing its buffer must not be able to
	// re-scope a live definition.
	buf := []byte("u:")
	cp, err := DefFrom(wire.KVIndexDef{Name: "cp", KeyPrefix: buf, PayloadPath: "rc"}, 1)
	if err != nil {
		t.Fatalf("DefFrom: %v", err)
	}
	buf[0] = 'z'
	if string(cp.Prefix) != "u:" {
		t.Fatalf("prefix aliased the caller's buffer: %q", cp.Prefix)
	}

	count, err := DefFrom(wire.KVIndexDef{Name: "c", PayloadPath: "tb#count", Kind: wire.KVIndexKindCount}, 1)
	if err != nil {
		t.Fatalf("#count definition rejected: %v", err)
	}
	if count.Path.Segs[0].Kind != record.SegCount {
		t.Fatalf("#count path parsed as %v", count.Path.Segs[0].Kind)
	}

	bad := []struct {
		name string
		def  wire.KVIndexDef
	}{
		{"empty name", wire.KVIndexDef{PayloadPath: "rc"}},
		{"bad charset", wire.KVIndexDef{Name: "a b", PayloadPath: "rc"}},
		{"long name", wire.KVIndexDef{Name: strings.Repeat("n", wire.KVIndexMaxNameLen+1), PayloadPath: "rc"}},
		{"empty path", wire.KVIndexDef{Name: "n"}},
		{"row path", wire.KVIndexDef{Name: "n", PayloadPath: "tb/3"}},
		{"long prefix", wire.KVIndexDef{Name: "n", PayloadPath: "rc", KeyPrefix: bytes.Repeat([]byte("p"), wire.KVIndexMaxPrefixLen+1)}},
		{"long path", wire.KVIndexDef{Name: "n", PayloadPath: strings.Repeat("p", wire.KVIndexMaxPathLen+1)}},
		{"kind disagrees (count)", wire.KVIndexDef{Name: "n", PayloadPath: "rc", Kind: wire.KVIndexKindCount}},
		{"kind disagrees (scalar)", wire.KVIndexDef{Name: "n", PayloadPath: "tb#count", Kind: wire.KVIndexKindScalar}},
		{"unknown kind", wire.KVIndexDef{Name: "n", PayloadPath: "rc", Kind: 9}},
		// Shape-valid for sdk/wire (no '/'), but not a path record.ParsePath
		// accepts. These are the ones DefFrom exists to catch: sdk/wire cannot
		// import record, so the full grammar is checked HERE.
		{"quote in field name", wire.KVIndexDef{Name: "n", PayloadPath: `a"b`}},
		{"non-canonical position", wire.KVIndexDef{Name: "n", PayloadPath: "#0001"}},
		{"position too large", wire.KVIndexDef{Name: "n", PayloadPath: "#99999"}},
		{"lone #count", wire.KVIndexDef{Name: "n", PayloadPath: "#count", Kind: wire.KVIndexKindCount}},
		{"double count", wire.KVIndexDef{Name: "n", PayloadPath: "a#count#count", Kind: wire.KVIndexKindCount}},
	}
	for _, tc := range bad {
		if _, err := DefFrom(tc.def, 1); err == nil {
			t.Errorf("%s: accepted %+v", tc.name, tc.def)
		}
	}
}

// --- lifecycle ------------------------------------------------------------

func TestInstallKeepsUnchangedDefs(t *testing.T) {
	a := mustDef(t, "keep", "u:", "rc", wire.KVIndexKindScalar)
	b := mustDef(t, "drop", "u:", "sc", wire.KVIndexKindScalar)
	s := readySet(a, b)
	s.Reindex([]byte("u:k"), intRec("rc", 3))
	if keys, _ := s.Stats("keep"); keys != 1 {
		t.Fatalf("setup: keep has %d keys", keys)
	}

	// Same name/prefix/path/kind, a NEWER meta index: the posting and its
	// readiness survive, because nothing about what is posted changed.
	a2 := a
	a2.MetaIndex = 99
	// A definition whose PATH changed is a different index: it must start
	// empty and not-ready, or a query would read postings for the old path.
	c := mustDef(t, "keep2", "u:", "rc", wire.KVIndexKindScalar)
	c2 := mustDef(t, "keep2", "u:", "sc", wire.KVIndexKindScalar)
	// A definition whose PREFIX changed is likewise a different index.
	e := mustDef(t, "keep3", "u:", "rc", wire.KVIndexKindScalar)
	e2 := mustDef(t, "keep3", "o:", "rc", wire.KVIndexKindScalar)

	s.Install([]Def{a2, c})
	s.MarkReady("keep2")
	s.Reindex([]byte("u:k2"), strRec("sc", "x"))

	if !s.IsReady("keep") {
		t.Fatal("an unchanged definition lost its readiness")
	}
	if keys, _ := s.Stats("keep"); keys != 1 {
		t.Fatalf("an unchanged definition lost its postings (%d keys)", keys)
	}
	if got, ok := s.Lookup("keep"); !ok || got.MetaIndex != 99 {
		t.Fatalf("Lookup(keep) = %+v, %v; want the NEW meta index", got, ok)
	}
	if _, ok := s.Lookup("drop"); ok {
		t.Fatal("a definition removed by Install is still installed")
	}
	if keys, _ := s.Stats("drop"); keys != 0 {
		t.Fatalf("a removed definition kept %d keys", keys)
	}
	if names := defNames(s.Defs()); !equalStrings(names, []string{"keep", "keep2"}) {
		t.Fatalf("Defs() = %v", names)
	}

	s.Install([]Def{a2, c2})
	if s.IsReady("keep2") {
		t.Fatal("a definition whose path changed kept its readiness")
	}
	if keys, _ := s.Stats("keep2"); keys != 0 {
		t.Fatalf("a definition whose path changed kept %d postings", keys)
	}

	s.Install([]Def{e})
	s.MarkReady("keep3")
	s.Reindex([]byte("u:k3"), intRec("rc", 1))
	s.Install([]Def{e2})
	if s.IsReady("keep3") {
		t.Fatal("a definition whose prefix changed kept its readiness")
	}
	if keys, _ := s.Stats("keep3"); keys != 0 {
		t.Fatalf("a definition whose prefix changed kept %d postings", keys)
	}

	// Defs() hands back a copy: mutating it must not reach the Set.
	s.Install([]Def{a2})
	ds := s.Defs()
	ds[0].Name = "clobbered"
	if got, ok := s.Lookup("keep"); !ok || got.Name != "keep" {
		t.Fatalf("Defs() aliased the Set's own slice: %+v", got)
	}
}

func TestResetKeepsReadyAndClearsPostings(t *testing.T) {
	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	s.Reindex([]byte("k"), intRec("rc", 1))

	// A definition that has never been walked is installed alongside it: after
	// the flush BOTH are exact, because the keyspace they cover is empty.
	fresh := mustDef(t, "fresh", "", "sc", wire.KVIndexKindScalar)
	s.Install([]Def{d, fresh})
	if s.IsReady("fresh") {
		t.Fatal("setup: an unwalked definition must not be ready")
	}

	s.Reset()

	// Flush emptied the cache, so an EMPTY posting set is the EXACT answer for
	// the (now empty) keyspace — for every definition, walked or not. Leaving
	// one not-ready would strand it in `building` until an unrelated meta write
	// moved the observer, turning every kv_query into ErrIndexBuilding for no
	// reason.
	if !s.IsReady("by-rc") {
		t.Fatal("Reset cleared readiness")
	}
	if !s.IsReady("fresh") {
		t.Fatal("Reset left an unwalked definition building over an empty cache")
	}
	if keys, distinct := s.Stats("by-rc"); keys != 0 || distinct != 0 {
		t.Fatalf("Reset left (%d keys, %d distinct)", keys, distinct)
	}
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 0 {
		t.Fatalf("Reset left candidates: %v", got)
	}

	// The Set is still usable afterwards.
	s.Reindex([]byte("k2"), intRec("rc", 1))
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"k2"}) {
		t.Fatalf("post-Reset reindex: got %v", got)
	}
}

// TestResetDuringWalkPublishesTheEmptyIndex pins the one way a flush could
// strand a definition in `building` forever.
//
// Reset bumps every posting's generation, which invalidates any walk in
// flight. A definition that was mid-BACKFILL when the flush landed therefore
// never publishes through its own walk — the grant is refused — and nothing
// re-drives it: it would sit failing every kv_query with ErrIndexBuilding
// until an unrelated meta write happened to move the observer. So Reset
// publishes readiness itself, which is also the truth: the keyspace the
// backfill was being built over no longer exists, and an empty posting set
// over an empty cache is exact.
func TestResetDuringWalkPublishesTheEmptyIndex(t *testing.T) {
	d := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	s := New(1024)
	s.Install([]Def{d})
	if s.IsReady("ix") {
		t.Fatal("setup: a freshly installed definition must not be ready")
	}

	ks := keyspace{keys: []string{"k1", "k2", "k3", "k4"}, vals: map[string][]byte{
		"k1": intRec("rc", 1), "k2": intRec("rc", 2),
		"k3": intRec("rc", 3), "k4": intRec("rc", 4),
	}}
	var flushed bool
	_ = s.Backfill("ix", func(fn func(key, value []byte) bool) error {
		for i, k := range ks.keys {
			if i == 2 {
				s.Reset() // the flush lands mid-backfill
				flushed = true
			}
			if !fn([]byte(k), ks.vals[k]) {
				return nil
			}
		}
		return nil
	})
	if !flushed {
		t.Fatal("the flush did not run")
	}

	// The definition is ready over an empty index: exact, because the cache it
	// was being built over was emptied.
	if !s.IsReady("ix") {
		t.Fatal("a flush mid-backfill left the definition building forever")
	}
	// The stale walk published nothing and resurrected nothing: the keys it had
	// already read from the pre-flush cache must not be posted back.
	if keys, distinct := s.Stats("ix"); keys != 0 || distinct != 0 {
		t.Fatalf("a stale walk wrote (%d keys, %d distinct) into the flushed index", keys, distinct)
	}
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 0 {
		t.Fatalf("flushed index answered %v", got)
	}

	// And it is a normal, usable index afterwards.
	s.Reindex([]byte("k9"), intRec("rc", 1))
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"k9"}) {
		t.Fatalf("post-flush write: got %v, want [k9]", got)
	}

	// Readiness returns to false only at the start of a NEW walk, which will
	// publish its own result.
	_ = s.Backfill("ix", func(fn func(key, value []byte) bool) error {
		if s.IsReady("ix") {
			t.Error("the index reported ready DURING its backfill")
		}
		return walkOf(ks)(fn)
	})
	if !s.IsReady("ix") || func() int { k, _ := s.Stats("ix"); return k }() != 4 {
		t.Fatal("the definition did not come back from its own backfill")
	}
}

func TestRebuildFromWalk(t *testing.T) {
	a := mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)
	b := mustDef(t, "by-cnt", "", "tb#count", wire.KVIndexKindCount)
	s := New(1024)
	s.Install([]Def{a, b})

	// Stale state from before the rebuild must not survive it.
	s.Reindex([]byte("u:gone"), intRec("rc", 1))

	ks := keyspace{
		keys: []string{"o:t", "u:x", "u:y"},
		vals: map[string][]byte{
			"u:x": intRec("rc", 1),
			"u:y": intRec("rc", 2),
			"o:t": tableRec("tb", 3),
		},
	}
	_ = s.Rebuild(walkOf(ks))

	if !s.IsReady("by-rc") || !s.IsReady("by-cnt") {
		t.Fatal("Rebuild did not mark every definition ready")
	}
	sel := Selector{Def: a, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"u:x"}) {
		t.Fatalf("after Rebuild: got %v, want [u:x] (stale key not cleared?)", got)
	}
	csel := Selector{Def: b, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(3)}}
	if got := mustCandidates(t, s, csel, nil, bigBudget); !equalStrings(got, []string{"o:t"}) {
		t.Fatalf("#count after Rebuild: got %v", got)
	}

	// A rebuild is not ready UNTIL it finishes: a query served from a
	// half-filled posting set would be a proper SUBSET of the truth, which is
	// the one thing this index may never return.
	seen := 0
	s.Install([]Def{a})
	s.MarkReady("by-rc")
	_ = s.Rebuild(func(fn func(key, value []byte) bool) error {
		for _, k := range ks.keys {
			if s.IsReady("by-rc") {
				t.Error("the index reported ready DURING a rebuild")
			}
			seen++
			if !fn([]byte(k), ks.vals[k]) {
				return nil
			}
		}
		return nil
	})
	if seen != len(ks.keys) {
		t.Fatalf("walk visited %d keys, want %d", seen, len(ks.keys))
	}
	if !s.IsReady("by-rc") {
		t.Fatal("not ready after the rebuild finished")
	}
}

func TestBackfillOneDef(t *testing.T) {
	a := mustDef(t, "old", "", "rc", wire.KVIndexKindScalar)
	s := New(1024)
	s.Install([]Def{a})
	s.MarkReady("old")
	s.Reindex([]byte("k"), intRec("rc", 1))

	b := mustDef(t, "new", "", "rc", wire.KVIndexKindScalar)
	s.Install([]Def{a, b})
	if s.IsReady("new") {
		t.Fatal("a freshly installed definition is ready before its backfill")
	}

	ks := keyspace{keys: []string{"k", "k2"}, vals: map[string][]byte{
		"k":  intRec("rc", 1),
		"k2": intRec("rc", 2),
	}}
	_ = s.Backfill("new", walkOf(ks))

	if !s.IsReady("new") {
		t.Fatal("Backfill did not mark its definition ready")
	}
	if keys, _ := s.Stats("new"); keys != 2 {
		t.Fatalf("backfilled index has %d keys, want 2", keys)
	}
	// The OTHER definition is untouched — same postings, same readiness.
	if !s.IsReady("old") {
		t.Fatal("Backfill cleared another definition's readiness")
	}
	if keys, _ := s.Stats("old"); keys != 1 {
		t.Fatalf("Backfill changed another definition's postings (%d keys)", keys)
	}

	// Backfilling a name that is not installed is a no-op, not a panic.
	_ = s.Backfill("nope", walkOf(ks))
}

// TestInstallDuringWalkLeavesTheNewDefBuilding pins the one way a rebuild
// could publish a proper SUBSET of the truth: an Install lands mid-walk,
// swapping in a fresh posting for a definition whose path moved, and the walk
// then marks THAT one ready having filled only the tail of the keyspace.
// Readiness is granted only to the posting the walk actually filled.
func TestInstallDuringWalkLeavesTheNewDefBuilding(t *testing.T) {
	a := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	moved := mustDef(t, "ix", "", "sc", wire.KVIndexKindScalar)
	ks := keyspace{keys: []string{"k1", "k2", "k3", "k4"}, vals: map[string][]byte{
		"k1": strRec("sc", "a"), "k2": strRec("sc", "b"),
		"k3": strRec("sc", "c"), "k4": strRec("sc", "d"),
	}}

	for _, tc := range []struct {
		name string
		run  func(s *Set, walk Walker)
		// wantEmpty: whether the replacement definition must be left with no
		// postings at all. Backfill's walk is pinned to the posting it started
		// on, so it writes nothing into a replacement. Rebuild's walk goes
		// through Reindex, which deliberately targets the LIVE definition set —
		// that is what makes an ordinary write racing a rebuild get recorded —
		// so the replacement may collect a few tail-of-the-keyspace postings.
		// Harmless: postings are hints, and the replacement's own backfill
		// clears them before it is ever published as ready.
		wantEmpty bool
	}{
		{"Rebuild", func(s *Set, w Walker) { _ = s.Rebuild(w) }, false},
		{"Backfill", func(s *Set, w Walker) { _ = s.Backfill("ix", w) }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := New(1024)
			s.Install([]Def{a})
			n := 0
			tc.run(s, func(fn func(key, value []byte) bool) error {
				for _, k := range ks.keys {
					if n == 2 {
						s.Install([]Def{moved}) // the definition is replaced mid-walk
					}
					n++
					if !fn([]byte(k), ks.vals[k]) {
						return nil
					}
				}
				return nil
			})
			if s.IsReady("ix") {
				t.Fatal("a definition replaced mid-walk was published as ready with a partial posting set")
			}
			if keys, _ := s.Stats("ix"); tc.wantEmpty && keys != 0 {
				t.Fatalf("a stale walk wrote %d keys into the replacement definition", keys)
			}
		})
	}
}

// TestOverlappingWalksNeverPublishPartial is the deterministic version of the
// interleaving a pointer-identity guard cannot see.
//
// Rebuild and Backfill both reset their posting IN PLACE, so during an overlap
// s.posts[name] is the same *posting for both walks and a pointer check still
// holds. The damage: a backfill fills k:000-k:049, a rebuild then empties the
// posting and refills only k:000-k:019, and the backfill — which started first
// and is still holding a valid pointer — finishes and grants ready over the
// rebuild's partial refill. The index is then published holding neither walk's
// result, and an eq query for a value only a lost key holds returns nothing
// while that key is live. That is the missing-candidate class this package
// exists to make impossible, so readiness is granted against the GENERATION,
// not the pointer.
//
// The two walks are stepped by hand here — no goroutines, no timing.
func TestOverlappingWalksNeverPublishPartial(t *testing.T) {
	const n = 100
	ks := keyspace{vals: make(map[string][]byte, n)}
	for i := 0; i < n; i++ {
		k := fmt.Sprintf("k:%03d", i)
		ks.keys = append(ks.keys, k)
		ks.vals[k] = intRec("rc", int64(i))
	}
	d := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	s := New(1024)
	s.Install([]Def{d})

	// stepped hands out a walk that emits ks.keys[from:to] when driven.
	stepped := func(from, to int) Walker {
		return func(fn func(key, value []byte) bool) error {
			for _, k := range ks.keys[from:to] {
				if !fn([]byte(k), ks.vals[k]) {
					return nil
				}
			}
			return nil
		}
	}

	// The backfill's walk runs in two halves, with a whole rebuild in between.
	var rebuildDone bool
	_ = s.Backfill("ix", func(fn func(key, value []byte) bool) error {
		_ = stepped(0, 50)(fn)        // the backfill fills k:000-k:049
		_ = s.Rebuild(stepped(0, 20)) // a rebuild empties it and refills k:000-k:019
		rebuildDone = true
		return stepped(50, 100)(fn) // the backfill's tail, into a generation it no longer owns
	})
	if !rebuildDone {
		t.Fatal("the overlapping rebuild did not run")
	}

	// The rebuild owns the current generation, so IT published — and what it
	// published is exactly what it walked, with nothing the stale walk added.
	if !s.IsReady("ix") {
		t.Fatal("the rebuild that owns the generation did not publish its result")
	}
	if keys, _ := s.Stats("ix"); keys != 20 {
		t.Fatalf("index holds %d keys, want the rebuild's 20 — a stale walk wrote into a generation it does not own", keys)
	}
	// A published index must not be missing a candidate for anything it covers.
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(19)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); !equalStrings(got, []string{"k:019"}) {
		t.Fatalf("eq 19 over the published index: got %v, want [k:019]", got)
	}
}

func TestRebuildWithNilWalkPublishesNothing(t *testing.T) {
	d := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	s := New(1024)
	s.Install([]Def{d})

	// A nil walker walked nothing, so it is not evidence of an empty keyspace
	// and may not publish an empty posting set as exact.
	s.Rebuild(nil)
	if s.IsReady("ix") {
		t.Fatal("Rebuild(nil) published an unwalked index as ready")
	}
	s.Backfill("ix", nil)
	if s.IsReady("ix") {
		t.Fatal("Backfill(nil) published an unwalked index as ready")
	}
}

func TestCandidatesRejectsAStaleDef(t *testing.T) {
	old := mustDef(t, "ix", "u:", "rc", wire.KVIndexKindScalar)
	s := readySet(old)
	s.Reindex([]byte("u:k"), intRec("rc", 1))

	sel := Selector{Def: old, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if _, err := s.Candidates(sel, nil, bigBudget); err != nil {
		t.Fatalf("current definition: %v", err)
	}

	// A re-issued, byte-identical definition at a newer meta index is the SAME
	// index and must still answer.
	reissued := old
	reissued.MetaIndex = 77
	s.Install([]Def{reissued})
	if _, err := s.Candidates(sel, nil, bigBudget); err != nil {
		t.Fatalf("re-issued identical definition: %v", err)
	}

	// A definition whose PATH moved is a different index wearing the same name:
	// its postings answer a different question, so they are not a superset of
	// the caller's predicate and must not be served to it.
	moved := mustDef(t, "ix", "u:", "sc", wire.KVIndexKindScalar)
	s.Install([]Def{moved})
	s.MarkReady("ix")
	if _, err := s.Candidates(sel, nil, bigBudget); !errors.Is(err, ErrIndexChanged) {
		t.Fatalf("moved path: err = %v, want ErrIndexChanged", err)
	}

	// Same for a definition whose PREFIX moved.
	rescoped := mustDef(t, "ix", "o:", "rc", wire.KVIndexKindScalar)
	s.Install([]Def{rescoped})
	s.MarkReady("ix")
	if _, err := s.Candidates(sel, nil, bigBudget); !errors.Is(err, ErrIndexChanged) {
		t.Fatalf("moved prefix: err = %v, want ErrIndexChanged", err)
	}
}

// TestCandidatesRangeChargesDistinctValues pins the cost a range walk pays
// even when it matches nothing. The walk is O(distinct values) whatever the
// answer size is, and it runs under the read lock, so an uncharged walk over a
// high-cardinality index blocks every write for its duration while reporting
// "0 candidates, well inside budget". Charging the examinations turns that
// into an honest typed refusal.
func TestCandidatesRangeChargesDistinctValues(t *testing.T) {
	d := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	const distinct = 500
	for i := 0; i < distinct; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%04d", i)), intRec("rc", int64(i)))
	}
	if _, got := s.Stats("ix"); got != distinct {
		t.Fatalf("setup: %d distinct values, want %d", got, distinct)
	}

	// Matches NOTHING (every value is >= 0), so no key is ever charged — the
	// examinations are the whole cost, and they must be refused.
	empty := Selector{Def: d, Op: vtypes.FilterLt, Values: []vtypes.Value{vtypes.NewInt(0)}}
	if _, err := s.Candidates(empty, nil, distinct/2); !errors.Is(err, ErrCandidateBudget) {
		t.Fatalf("zero-match range over %d distinct values with budget %d: err = %v, want ErrCandidateBudget",
			distinct, distinct/2, err)
	}
	// With room for the walk it answers normally, empty.
	if got := mustCandidates(t, s, empty, nil, distinct+1); len(got) != 0 {
		t.Fatalf("zero-match range: got %v", got)
	}
	// A budget that covers the examinations but not the keys they admit is
	// still a refusal.
	all := Selector{Def: d, Op: vtypes.FilterGte, Values: []vtypes.Value{vtypes.NewInt(0)}}
	if _, err := s.Candidates(all, nil, distinct+10); !errors.Is(err, ErrCandidateBudget) {
		t.Fatalf("full-match range: err = %v, want ErrCandidateBudget", err)
	}
	if got := mustCandidates(t, s, all, nil, 2*distinct); len(got) != distinct {
		t.Fatalf("full-match range with room: got %d keys, want %d", len(got), distinct)
	}

	// An eq selector pays only for the keys it unions, unchanged: the walk
	// there is one map probe, not a scan of the distinct values.
	eq := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(3)}}
	if got := mustCandidates(t, s, eq, nil, 1); !equalStrings(got, []string{"k0003"}) {
		t.Fatalf("eq with a budget of exactly 1 key: got %v", got)
	}

	// A bound no ordering can use examines nothing at all, so it answers even
	// with a budget of zero.
	boolBound := Selector{Def: d, Op: vtypes.FilterGt, Values: []vtypes.Value{vtypes.NewBool(true)}}
	if got := mustCandidates(t, s, boolBound, nil, 0); len(got) != 0 {
		t.Fatalf("unusable bound with budget 0: got %v", got)
	}
}

// TestCandidatesSortsOutsideTheLock pins that a big page's sort does not block
// writers. Sorting dominates the call (n log n byte comparisons against one
// map walk), and Reindex/Drop both want the write lock — Drop while holding a
// cache shard's write lock — so a sort inside the read lock stalls the shard
// for its whole duration.
//
// Timing-tolerant by construction: the assertion is a RATIO (one write must
// complete in well under the time the whole call takes), it is retried, and it
// fails deterministically when the sort moves back inside the lock, where a
// write blocks for essentially the entire call.
func TestCandidatesSortsOutsideTheLock(t *testing.T) {
	n := 250_000
	if testing.Short() {
		n = 100_000
	}
	d := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	val := intRec("rc", 1) // one value, so the whole keyspace is one answer
	for i := 0; i < n; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%07d", i)), val)
	}
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	other := intRec("rc", 2)

	for attempt := 1; attempt <= 3; attempt++ {
		var (
			wg       sync.WaitGroup
			total    time.Duration
			maxWrite time.Duration
			writes   int
		)
		done := make(chan struct{})
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			got, err := s.Candidates(sel, nil, 4*n)
			total = time.Since(start)
			close(done)
			if err != nil || len(got) != n {
				t.Errorf("Candidates: %d keys, err %v", len(got), err)
			}
		}()

		for i := 0; ; i++ {
			select {
			case <-done:
			default:
				w := time.Now()
				s.Reindex([]byte(fmt.Sprintf("w%07d", i)), other)
				if el := time.Since(w); el > maxWrite {
					maxWrite = el
				}
				writes++
				continue
			}
			break
		}
		wg.Wait()

		if writes == 0 {
			t.Fatal("no write raced the query; the fixture is too small to prove anything")
		}
		// With the sort outside the lock a write waits only for the copy loop,
		// which is a fraction of the call. With it inside, a write waits for
		// essentially all of it.
		if maxWrite < total/2 {
			t.Logf("attempt %d: Candidates %v, %d concurrent writes, slowest %v", attempt, total, writes, maxWrite)
			return
		}
		t.Logf("attempt %d: Candidates %v, slowest write %v (>= half the call) — retrying", attempt, total, maxWrite)
	}
	t.Fatal("a concurrent write was blocked for at least half of every Candidates call: the sort is holding the read lock")
}

// TestSetNeverCallsBack proves the Set never reaches into a store on its own.
// It matters because Drop runs UNDER a cache shard's write lock (the onRemove
// hook): a Set method that called back into the cache while holding Set.mu
// would close the lock cycle and deadlock the node. The walk function is only
// ever an argument to Rebuild/Backfill; the Set never stores one.
func TestSetNeverCallsBack(t *testing.T) {
	var allowed atomic.Bool
	var calls atomic.Int64
	ks := keyspace{keys: []string{"k"}, vals: map[string][]byte{"k": intRec("rc", 1)}}
	walk := func(fn func(key, value []byte) bool) error {
		if !allowed.Load() {
			t.Error("the Set called the store walker outside Rebuild/Backfill")
			return nil
		}
		calls.Add(1)
		return walkOf(ks)(fn)
	}

	d := mustDef(t, "by-rc", "", "rc", wire.KVIndexKindScalar)
	s := New(1024)
	s.Install([]Def{d})
	s.MarkReady("by-rc")
	s.Reindex([]byte("k"), intRec("rc", 1))
	s.Drop([]byte("k"))
	s.Reindex([]byte("k"), intRec("rc", 1))
	s.Reset()
	_ = s.Defs()
	_, _ = s.Lookup("by-rc")
	_ = s.IsReady("by-rc")
	s.MarkReady("by-rc")
	_, _ = s.Stats("by-rc")
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(1)}}
	if _, err := s.Candidates(sel, nil, bigBudget); err != nil {
		t.Fatalf("Candidates: %v", err)
	}
	s.Install([]Def{d})

	allowed.Store(true)
	_ = s.Rebuild(walk)
	_ = s.Backfill("by-rc", walk)
	if got := calls.Load(); got != 2 {
		t.Fatalf("walker called %d times, want 2", got)
	}
}

func TestReindexIsRaceFree(t *testing.T) {
	dur := time.Second
	if testing.Short() {
		dur = 150 * time.Millisecond
	}
	a := mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)
	b := mustDef(t, "by-cnt", "", "tb#count", wire.KVIndexKindCount)
	s := readySet(a, b)

	deadline := time.Now().Add(dur)
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			for time.Now().Before(deadline) {
				k := []byte(fmt.Sprintf("u:%d", rng.Intn(64)))
				switch rng.Intn(4) {
				case 0:
					s.Reindex(k, intRec("rc", int64(rng.Intn(8))))
				case 1:
					s.Reindex(k, tableRec("tb", rng.Intn(3)))
				case 2:
					s.Drop(k)
				default:
					sel := Selector{Def: a, Op: vtypes.FilterGte, Values: []vtypes.Value{vtypes.NewInt(int64(rng.Intn(8)))}}
					if _, err := s.Candidates(sel, nil, bigBudget); err != nil {
						t.Errorf("Candidates: %v", err)
						return
					}
				}
			}
		}(int64(w) + 1)
	}
	// A concurrent Reset and Install must be safe too — flush and the meta
	// observer both run while writes are in flight.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for time.Now().Before(deadline) {
			s.Reset()
			s.Install([]Def{a, b})
			s.MarkReady("by-rc")
			s.MarkReady("by-cnt")
			time.Sleep(time.Millisecond)
		}
	}()
	wg.Wait()
}

// --- helpers --------------------------------------------------------------

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func defNames(defs []Def) []string {
	out := make([]string, 0, len(defs))
	for _, d := range defs {
		out = append(out, d.Name)
	}
	return out
}

// --- benchmarks -----------------------------------------------------------

func BenchmarkReindexSchemaRecord(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	d, err := DefFrom(wire.KVIndexDef{Name: "by-rc", PayloadPath: "rc"}, 1)
	if err != nil {
		b.Fatal(err)
	}
	s := New(1024)
	s.Install([]Def{d})
	s.MarkReady("by-rc")
	val := schemaRecord(rng)
	key := []byte("u:benchmark-key")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Reindex(key, val)
	}
}

// BenchmarkReindexMovingValue is the other half of the write path: a rewrite
// that MOVES the field's value, so the posting is actually removed from one
// set and inserted into another (the one allocation per insert, shared by
// both maps).
func BenchmarkReindexMovingValue(b *testing.B) {
	d, err := DefFrom(wire.KVIndexDef{Name: "by-rc", PayloadPath: "rc"}, 1)
	if err != nil {
		b.Fatal(err)
	}
	s := New(1024)
	s.Install([]Def{d})
	s.MarkReady("by-rc")
	vals := [][]byte{intRec("rc", 1), intRec("rc", 2)}
	key := []byte("u:benchmark-key")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Reindex(key, vals[i&1])
	}
}

func BenchmarkCandidatesEq100k(b *testing.B) {
	d, err := DefFrom(wire.KVIndexDef{Name: "by-rc", PayloadPath: "rc"}, 1)
	if err != nil {
		b.Fatal(err)
	}
	s := New(1024)
	s.Install([]Def{d})
	s.MarkReady("by-rc")
	// 100k keys over 1000 distinct values: 100 keys per equality answer.
	for i := 0; i < 100_000; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%06d", i)), intRec("rc", int64(i%1000)))
	}
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(7)}}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := s.Candidates(sel, nil, bigBudget)
		if err != nil || len(got) != 100 {
			b.Fatalf("got %d keys, err %v", len(got), err)
		}
	}
}

// A walk that does not finish has read a PROPER SUBSET of the keyspace.
// Publishing that as ready would make every query answer from the subset and
// call it exact — silently missing rows, the one failure this index may not
// have. Both fillers must return the walk's error and mark nothing.
func TestBackfillDoesNotPublishOnAWalkError(t *testing.T) {
	ks := keyspace{keys: []string{"k0", "k1", "k2", "k3"}, vals: map[string][]byte{
		"k0": intRec("rc", 0), "k1": intRec("rc", 1),
		"k2": intRec("rc", 2), "k3": intRec("rc", 3),
	}}
	cut := errors.New("the shard went away mid-walk")

	s := New(1024)
	s.Install([]Def{mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)})
	err := s.Backfill("ix", func(fn func(key, value []byte) bool) error {
		for _, k := range ks.keys[:2] {
			fn([]byte(k), ks.vals[k])
		}
		return cut
	})
	if !errors.Is(err, cut) {
		t.Fatalf("Backfill error = %v, want the walk's error", err)
	}
	if s.IsReady("ix") {
		t.Fatal("a half-finished backfill published the index as ready")
	}

	// And the same for a whole-set rebuild, which must not leave ANY definition
	// ready off a walk that stopped early.
	s2 := New(1024)
	s2.Install([]Def{mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)})
	s2.MarkReady("ix") // it was ready before; a failed rebuild must revoke that
	if err := s2.Rebuild(func(fn func(key, value []byte) bool) error {
		fn([]byte("k0"), ks.vals["k0"])
		return cut
	}); !errors.Is(err, cut) {
		t.Fatalf("Rebuild error = %v, want the walk's error", err)
	}
	if s2.IsReady("ix") {
		t.Fatal("a half-finished rebuild left the index published as ready")
	}

	// A walk that DOES finish still publishes, so the guard is not just "never
	// ready".
	s3 := New(1024)
	s3.Install([]Def{mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)})
	if err := s3.Backfill("ix", walkOf(ks)); err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if !s3.IsReady("ix") {
		t.Fatal("a completed backfill did not publish the index")
	}
}
