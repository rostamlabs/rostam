// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strings"
	"testing"
)

// A record-path filter leaf is compiled ONCE and evaluated once per point, so
// anything it re-derives per point is multiplied by the scan width. It used to
// call lookupPath with the raw field string, which re-ran SplitField+ParsePath
// for every row — a constant answer, recomputed, allocating a split slice and a
// Segment slice each time. compileLeaf now builds a fieldLookup at compile time
// and the closures resolve through it.
//
// This test pins the outcome at zero: an ordinary numeric record-path predicate
// must allocate NOTHING per point once the resolver's schema cache is warm. It
// is deliberately an absolute bound, not a comparison — a budget of "fewer than
// before" would drift back up one allocation at a time.
func TestRecordPathPredicateAllocFree(t *testing.T) {
	meta := Metadata{
		"session": NewRecord(sessionRecordBytes(t)),
		"plain":   NewInt(3),
	}

	for _, tc := range []struct {
		name  string
		f     Filter
		want  bool
		field string
	}{
		{"gt record path", Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(5)}, true, "session/rc"},
		{"eq record path", Filter{Op: FilterEq, Field: "session/bc", Value: NewInt(3)}, true, "session/bc"},
		{"count record path", Filter{Op: FilterEq, Field: "session/b#count", Value: NewInt(2)}, true, "session/b#count"},
		{"gt plain key", Filter{Op: FilterGt, Field: "plain", Value: NewInt(1)}, true, "plain"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pred, err := CompileFilter(tc.f)
			if err != nil {
				t.Fatalf("CompileFilter(%s): %v", tc.field, err)
			}
			// Warm the schema cache and confirm the predicate is actually
			// answering — an alloc count of 0 on a predicate that returns false
			// for the wrong reason would prove nothing.
			if got := pred(meta); got != tc.want {
				t.Fatalf("predicate on %s = %v, want %v", tc.field, got, tc.want)
			}
			if n := testing.AllocsPerRun(200, func() { _ = pred(meta) }); n != 0 {
				t.Fatalf("predicate on %s allocated %.1f times per point, want 0", tc.field, n)
			}
		})
	}
}

// TestCompiledLookupMatchesLookupPath is the equivalence half of the same
// change: the compiled fieldLookup and the bare-string lookupPath must answer
// IDENTICALLY for every field shape, because both are live — lookupPath still
// serves order_by extraction and the payload index's reindex resolve, and a
// disagreement between them is exactly how an accelerated query would drop a
// row the predicate accepts.
func TestCompiledLookupMatchesLookupPath(t *testing.T) {
	rec := sessionRecordBytes(t)
	metas := []Metadata{
		nil,
		{},
		{"session": NewRecord(rec)},
		{"session": NewInt(1)},               // payload key is not a record
		{"session/rc": NewString("literal")}, // a literal key that looks like a path
		{"session": NewRecord(rec), "session/rc": NewString("shadow")}, // literal shadows the path
		{"": NewRecord(rec)},                       // the empty payload key is legal
		{"session": NewRecord([]byte{0xFF, 0xFF})}, // malformed record bytes
	}
	fields := []string{
		"session/rc", "session/bc", "session/bal", "session/tag",
		"session/b#count", "session/b/42/hi", "session/#0",
		"session", "plain", "no-slash",
		"/rc",             // empty payload key
		"session/",        // trailing slash: not a valid path
		"session/2024/q1", // parses no further than the row segment: not a path
		"a/b/c/d",         // too many segments
		`session/"x"/hi`,  // quoted row key
	}
	for _, m := range metas {
		for _, f := range fields {
			wantV, wantOK := lookupPath(m, f)
			gotV, gotOK := newFieldLookup(f).get(m)
			if gotOK != wantOK || (wantOK && !gotV.Equal(wantV)) {
				t.Fatalf("field %q on %v: compiled = (%v,%v), lookupPath = (%v,%v)",
					f, m, gotV, gotOK, wantV, wantOK)
			}
		}
	}
}

// A quoted row key in a filter field must be bounded before it can allocate:
// the parse now refuses it at compile time instead of copying megabytes per
// point. The compiled leaf degrades to an exact-key lookup, which is what
// lookupPath answers for an unparseable path too.
func TestHugeQuotedRowKeyFieldIsCheap(t *testing.T) {
	field := `session/"` + strings.Repeat("x", 1<<20) + `"/hi`
	meta := Metadata{"session": NewRecord(sessionRecordBytes(t))}

	pred, err := CompileFilter(Filter{Op: FilterEq, Field: field, Value: NewInt(1)})
	if err != nil {
		t.Fatalf("CompileFilter: %v", err)
	}
	if pred(meta) {
		t.Fatal("predicate on an unparseable path matched; want false (field absent)")
	}
	if n := testing.AllocsPerRun(50, func() { _ = pred(meta) }); n != 0 {
		t.Fatalf("predicate allocated %.1f times per point on a 1 MiB quoted key, want 0", n)
	}
}
