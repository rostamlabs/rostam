// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// su64 reinterprets a signed value as the uint64 bit pattern Cell.U stores
// for a signed type. A constant expression like uint64(int64(-5)) does not
// compile (constant overflow), so negative fixtures go through this helper —
// mirrors sdk/record/resolve_test.go's su64.
func su64(i int64) uint64 { return uint64(i) }

// sessionRecordBytes builds a schema-mode record with the same shape as
// sdk/record's own session example (§2.2), so both packages' tests exercise
// the same fixture without vector importing record's internal test file
// (record's tests live in package record and are not importable):
//
//	#0 rc   U8      = 7
//	#1 bc   U8      = 3
//	#2 hist U32     = 1234
//	#3 bal  I32     = -5
//	#4 tag  BYTES   = "de"
//	#5 b    TABLE(key U64; c0 "hi" U32, c1 "lo" U32) = {42: (500, 1), 99: (7, 2)}
func sessionRecordBytes(t *testing.T) []byte {
	t.Helper()
	return sessionRecordBytesRC(t, 7)
}

// sessionRecordBytesRC is sessionRecordBytes with a caller-chosen rc value
// (bc/hist/bal/tag/table b stay fixed) — used by tests that need many
// distinct records differing only in rc, e.g. to populate a corpus for a
// filter-first-vs-brute-force equivalence check.
func sessionRecordBytesRC(t *testing.T, rc uint8) []byte {
	t.Helper()
	leKey := func(v uint64) []byte {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, v)
		return b
	}
	s := &wire.Schema{
		Version:    1,
		StoreNames: true,
		Fields: []wire.FieldDef{
			{Name: "rc", Type: wire.OperateTypeU8},
			{Name: "bc", Type: wire.OperateTypeU8},
			{Name: "hist", Type: wire.OperateTypeU32},
			{Name: "bal", Type: wire.OperateTypeI32},
			{Name: "tag", Type: wire.OperateTypeBytes},
			{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{
				KeyType: wire.OperateTypeU64,
				Cols: []wire.ColumnDef{
					{Name: "hi", Type: wire.OperateTypeU32},
					{Name: "lo", Type: wire.OperateTypeU32},
				},
			}},
		},
	}
	if err := s.Validate(); err != nil {
		t.Fatalf("session schema invalid: %v", err)
	}
	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: uint64(rc)}},
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 3}},
		{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1234}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: leKey(42), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			}},
			{Key: leKey(99), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 2}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("session record failed to encode")
	}
	return enc
}

// TestLookupPathRecord covers lookupPath's record-path seam directly: exact
// payload key precedence, resolving into a session record's scalar, table
// cell, and #count, absence, and declining a payload key that is not a
// record.
func TestLookupPathRecord(t *testing.T) {
	enc := sessionRecordBytes(t)
	m := Metadata{
		"session": NewRecord(enc),
		// A flat key that happens to LOOK like a payloadKey/path string. Its
		// presence must win outright — lookupPath must never even attempt to
		// split it as "payloadKey a, path b" (which would resolve against a
		// record named "a" that does not exist in m at all).
		"a/b": NewString("literal-wins"),
		// A payload key present but not a record: any path under it must
		// decline, not panic or misinterpret its bytes as record bytes.
		"notrecord": NewString("plain string"),
	}

	cases := []struct {
		name  string
		field string
		want  Value
		ok    bool
	}{
		{"exact key precedence over path shape", "a/b", NewString("literal-wins"), true},
		{"scalar field", "session/rc", NewInt(7), true},
		{"table cell by row+col", "session/b/42/hi", NewInt(500), true},
		{"table row count", "session/b#count", NewInt(2), true},
		{"absent field", "session/zz", Value{}, false},
		{"absent row", "session/b/7", Value{}, false},
		{"table field itself has no scalar value", "session/b", Value{}, false},
		{"a present row has no scalar value either", "session/b/42", Value{}, false},
		{"payload key not a record", "notrecord/rc", Value{}, false},
		{"payload key missing entirely", "missing/rc", Value{}, false},
		{"malformed record path", "session/rc#count", Value{}, false}, // #count on a non-table field
		{"no slash, no exact key", "nonexistent", Value{}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := lookupPath(m, c.field)
			if ok != c.ok {
				t.Fatalf("lookupPath(%q) ok = %v, want %v (got %+v)", c.field, ok, c.ok, got)
			}
			if ok && !got.Equal(c.want) {
				t.Errorf("lookupPath(%q) = %+v, want %+v", c.field, got, c.want)
			}
		})
	}
}

// TestCompileFilterRecordPaths exercises the full CompileFilter path over a
// metadata map holding a record-typed payload key: scalar comparators
// (gt/eq/gte/in) reading through a record path, row_exists/row_absent (both
// directions, plus the row_absent asymmetry against a "cannot apply"
// outcome), and is_empty/is_null on the record field itself.
func TestCompileFilterRecordPaths(t *testing.T) {
	enc := sessionRecordBytes(t)
	m := Metadata{"session": NewRecord(enc)}
	emptyM := Metadata{"session": NewRecord(nil)}

	cases := []struct {
		name string
		f    Filter
		m    Metadata
		want bool
	}{
		{"gt over a scalar record field", Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(5)}, m, true},
		{"gt false case over the same field", Filter{Op: FilterGt, Field: "session/rc", Value: NewInt(100)}, m, false},
		{"eq over a table cell", Filter{Op: FilterEq, Field: "session/b/42/hi", Value: NewInt(500)}, m, true},
		{"eq mismatch over a table cell", Filter{Op: FilterEq, Field: "session/b/42/hi", Value: NewInt(1)}, m, false},
		{"gte over a table row count", Filter{Op: FilterGte, Field: "session/b#count", Value: NewInt(1)}, m, true},
		{"in over a field addressed by position", Filter{Op: FilterIn, Field: "session/#0", Value: NewInts([]int64{7, 8})}, m, true},
		{"in miss over a field addressed by position", Filter{Op: FilterIn, Field: "session/#0", Value: NewInts([]int64{1, 2})}, m, false},

		{"row_exists on a present row", Filter{Op: FilterRowExists, Field: "session/b/42"}, m, true},
		{"row_exists on an absent row", Filter{Op: FilterRowExists, Field: "session/b/7"}, m, false},
		{"row_absent on an absent row", Filter{Op: FilterRowAbsent, Field: "session/b/7"}, m, true},
		{"row_absent on a present row", Filter{Op: FilterRowAbsent, Field: "session/b/42"}, m, false},
		// Asymmetry: a row segment on a SCALAR field cannot apply (Resolve
		// errors ErrPath). row_exists is false for the same reason it would
		// be for any non-RowPresent outcome; row_absent must ALSO be false
		// here, not true — there was never a table to search, so "the row is
		// absent" cannot be asserted about it.
		{"row_exists cannot-apply (row segment on a scalar field)", Filter{Op: FilterRowExists, Field: "session/rc/1"}, m, false},
		{"row_absent cannot-apply (row segment on a scalar field)", Filter{Op: FilterRowAbsent, Field: "session/rc/1"}, m, false},
		// Same asymmetry when the payload key itself cannot apply: missing,
		// or present but not a record.
		{"row_absent on a missing payload key", Filter{Op: FilterRowAbsent, Field: "missing/b/7"}, m, false},
		{"row_absent on an empty/malformed record", Filter{Op: FilterRowAbsent, Field: "session/b/7"}, emptyM, false},

		{"is_empty on a non-empty record field", Filter{Op: FilterIsEmpty, Field: "session"}, m, false},
		{"is_empty on an empty record field", Filter{Op: FilterIsEmpty, Field: "session"}, emptyM, true},
		{"is_null on a present record field", Filter{Op: FilterIsNull, Field: "session"}, m, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := compileOrFail(t, c.f)
			if got := p(c.m); got != c.want {
				t.Errorf("CompileFilter(%+v)(m) = %v, want %v", c.f, got, c.want)
			}
		})
	}
}

// TestCompileFilterRowPresenceCompileErrors pins compileRowPresence's
// compile-time validation: the field must split into payloadKey/path, the
// path must parse, and the parsed path must name exactly a table row
// (field+row, two segments) — never a bare field, a #count, or a row+column.
func TestCompileFilterRowPresenceCompileErrors(t *testing.T) {
	cases := []struct {
		name  string
		field string
	}{
		{"no slash at all", "session"},
		{"malformed path (empty segment)", "session/"},
		{"path names only a field, no row", "session/rc"},
		{"path names a field's count, not a row", "session/b#count"},
		{"path names a row AND a column", "session/b/42/hi"},
	}
	for _, op := range []FilterOp{FilterRowExists, FilterRowAbsent} {
		for _, c := range cases {
			t.Run(mustOpName(op)+"/"+c.name, func(t *testing.T) {
				if _, err := CompileFilter(Filter{Op: op, Field: c.field}); err == nil {
					t.Errorf("CompileFilter(op=%s, field=%q) succeeded, want a compile error", mustOpName(op), c.field)
				}
			})
		}
	}
}

// TestRowPresenceRefusesUnorderedTable is the vector-side half of the proven-
// absence fix. Record bytes are not validated at ingest in this phase, so a
// caller can store a table whose rows are NOT in the ascending key order the
// format requires. A schema-mode row lookup is a binary search, which cannot
// see the disorder, so it answers Absent for a row that IS stored — and
// row_absent used to read that as a positive match. A point could then satisfy
// a filter it does not satisfy, chosen by the bytes the caller stored.
//
// Both ops must answer FALSE on such a table: row_exists because the search
// finds nothing, row_absent because absence is now PROVED by walking the table
// (record.Resolver.RowAbsentProven) and the walk sees the disorder.
//
// The record is patched at the byte level — the encoder sorts rows and
// DecodeRecord refuses the result — which is the test-only stand-in for the
// hostile writer this defends against.
func TestRowPresenceRefusesUnorderedTable(t *testing.T) {
	enc := sessionRecordBytes(t)
	ordered := Metadata{"session": NewRecord(append([]byte(nil), enc...))}

	// The two stored rows of table b, byte for byte: an 8-byte little-endian
	// key then two little-endian u32 columns. They are the last 32 bytes of the
	// record (b is the last field), and swapping them leaves every offset and
	// length in the record unchanged — only the ORDER becomes illegal.
	row42 := []byte{42, 0, 0, 0, 0, 0, 0, 0, 0xF4, 0x01, 0, 0, 1, 0, 0, 0}
	row99 := []byte{99, 0, 0, 0, 0, 0, 0, 0, 0x07, 0, 0, 0, 2, 0, 0, 0}
	at := bytes.Index(enc, row42)
	if at < 0 || !bytes.Equal(enc[at+16:], row99) {
		t.Fatalf("fixture changed: rows 42/99 are not the last 32 bytes of the record (%x)", enc)
	}
	copy(enc[at:], row99)
	copy(enc[at+16:], row42)
	unordered := Metadata{"session": NewRecord(enc)}

	for _, c := range []struct {
		field                  string
		wantExists, wantAbsent bool // on the ORDERED record
	}{
		{"session/b/42", true, false}, // a present row
		{"session/b/99", true, false}, // the other present row
		{"session/b/7", false, true},  // a row that really is absent
	} {
		t.Run(c.field, func(t *testing.T) {
			for _, tc := range []struct {
				op      FilterOp
				ordered bool
			}{{FilterRowExists, c.wantExists}, {FilterRowAbsent, c.wantAbsent}} {
				p := compileOrFail(t, Filter{Op: tc.op, Field: c.field})
				// The control: on the well-formed record the op answers normally.
				if got := p(ordered); got != tc.ordered {
					t.Errorf("%s on the ordered record = %v, want %v", mustOpName(tc.op), got, tc.ordered)
				}
				// The fix: on the damaged one, neither op asserts anything.
				if got := p(unordered); got {
					t.Errorf("%s on an out-of-order table = true, want false", mustOpName(tc.op))
				}
			}
		})
	}
}

// TestRowPresenceCompileErrorIsBounded pins the SIZE of a row-presence compile
// error. The field string is caller-supplied and bounded only by the 32 MiB
// route body cap, and both rejection branches used to quote it whole — with %q
// that is up to four bytes of escapes per input byte, so refusing a hostile
// filter cost another large copy of it, the very cost record.ParsePath's
// pre-split length bound was added to avoid.
//
// The message now carries a 64-byte prefix and the length, so it stays a couple
// of hundred bytes whatever the input, and rejecting stays flat in the size of
// the field.
func TestRowPresenceCompileErrorIsBounded(t *testing.T) {
	const big = 1 << 20
	huge := strings.Repeat("x", big)

	for _, c := range []struct {
		name  string
		field string
	}{
		// No '/' at all: rejected by SplitField, quoting f.Field.
		{"no slash", huge},
		// A path that parses to the wrong shape: rejected after ParsePath,
		// quoting the path.
		{"path too long", "session/" + huge},
		// A row-and-column path built from an over-long column name: parses far
		// enough to reach the shape check.
		{"wrong shape", "session/b/42/" + huge},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := CompileFilter(Filter{Op: FilterRowExists, Field: c.field})
			if err == nil {
				t.Fatal("CompileFilter succeeded, want a compile error")
			}
			if n := len(err.Error()); n >= 256 {
				t.Errorf("error message is %d bytes, want < 256", n)
			}
			if strings.Contains(err.Error(), strings.Repeat("x", 128)) {
				t.Error("error message echoed a long run of the field")
			}
			f := Filter{Op: FilterRowExists, Field: c.field}
			if n := testing.AllocsPerRun(20, func() { _, _ = CompileFilter(f) }); n > 20 {
				t.Errorf("rejecting a %d-byte field allocated %.1f times, want a small constant", big, n)
			}
		})
	}
}

// TestRecordPathFilterFirstMatchesBruteForce is the fix-round-1 regression
// test: the payload-index planner must NEVER claim a record path is
// index-narrowable (it has no postings — reindex only posts literal scalar
// keys, and scalarKeyOf declines ValueRecord outright), because "field has
// no postings" is the planner's proof that "field never matches" — and for a
// record path that inference is false, not just unproven. Before the
// indexNarrowable/filterIndexExact fix, `p.fields["session/rc"]` was always
// nil, so the planner graded an Eq/Gt/In/row-presence leaf over a record path
// as an EXACT, proven-empty set — searchIntoWith intersected the empty set
// and returned zero results (bypassing graph fallback), and matchingIDsAt
// took the same fast path and silently under-deleted. This test drives BOTH
// consumers (SearchFiltered and matchingIDs, the selection matchingIDsAt
// performs for delete-by-filter/scroll) over a real corpus and checks they
// agree with a brute-force evaluation of the SAME compiled predicate — so a
// regression here is a wrong ANSWER, not just a slow one.
func TestRecordPathFilterFirstMatchesBruteForce(t *testing.T) {
	const (
		n   = 60
		dim = 8
		k   = 12
	)
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
		// Most points carry a "session" record (rc varies 0..12 so Eq/Gt/In
		// each have some true and some false points); the rest carry none, so
		// every record-path filter also has to correctly reject points with
		// no such payload key at all.
		if i <= 45 {
			m["session"] = NewRecord(sessionRecordBytesRC(t, uint8(i%13))) //nolint:gosec // i%13 fits uint8
		}
		corpus[id] = v
		metas[id] = m
		if _, _, err := h.Insert(id, v, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	filters := []Filter{
		{Op: FilterEq, Field: "session/rc", Value: NewInt(5)},
		{Op: FilterGt, Field: "session/rc", Value: NewInt(10)},
		{Op: FilterIn, Field: "session/rc", Value: NewInts([]int64{3, 7})},
		{Op: FilterRowExists, Field: "session/b/42"},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "country", Value: NewString("DE")},
			{Op: FilterEq, Field: "session/rc", Value: NewInt(5)},
		}},
	}

	for _, f := range filters {
		pred := compileOrFail(t, f)

		// Consumer 1: SearchFiltered (filter-first KNN through
		// hnsw.searchIntoWith), compared against a brute-force top-k over the
		// SAME compiled predicate.
		got, err := h.SearchFiltered(q, k, f)
		if err != nil {
			t.Fatalf("filter %+v: SearchFiltered: %v", f, err)
		}
		want := bruteForceFiltered(corpus, metas, q, k, pred)
		if !eqUint64(resultIDs(got), want) {
			t.Errorf("filter %+v: SearchFiltered ids %v != brute-force %v", f, resultIDs(got), want)
		}

		// Consumer 2: matchingIDs, the id-selection matchingIDsAt performs for
		// delete-by-filter and non-vector scroll, compared against a
		// brute-force full-corpus evaluation of the same predicate (not
		// capped to k, unlike the search path above).
		gotIDs, err := h.matchingIDs(f, pred)
		if err != nil {
			t.Fatalf("filter %+v: matchingIDs: %v", f, err)
		}
		wantSet := bruteMatchIDs(metas, pred)
		gotSet := make(map[uint64]struct{}, len(gotIDs))
		for _, id := range gotIDs {
			gotSet[id] = struct{}{}
		}
		if len(gotSet) != len(wantSet) {
			t.Errorf("filter %+v: matchingIDs set size %d != brute-force %d (got=%v want=%v)", f, len(gotSet), len(wantSet), gotSet, wantSet)
			continue
		}
		for id := range wantSet {
			if _, ok := gotSet[id]; !ok {
				t.Errorf("filter %+v: matchingIDs missing true match id %d (under-selects, so delete-by-filter would under-delete)", f, id)
			}
		}
	}

	// A control assertion pinning the bug's shape directly: at least one of
	// the filters above must have a NON-EMPTY brute-force match set, or this
	// whole test would pass vacuously (both sides empty) without ever
	// exercising the planner's wrong "proven empty" claim.
	nonEmpty := false
	for _, f := range filters {
		if len(bruteMatchIDs(metas, compileOrFail(t, f))) > 0 {
			nonEmpty = true
			break
		}
	}
	if !nonEmpty {
		t.Fatal("every filter's brute-force match set was empty; test proves nothing — fixture is broken")
	}
}

// TestRowPresenceExactPayloadKeyWins pins the exact-payload-key-first rule for
// row_exists / row_absent.
//
// A filter's field string is tried as an EXACT payload key first, and only then
// split at the first '/'; lookupPath and fieldLookup.get both do this, and the
// rule exists so two seams never disagree about one field string. The row
// presence closure used to skip the exact lookup entirely, so a point holding
// BOTH a record under "session" AND a literal "session/b/42" key answered
// row_exists true while lookupPath on the same field returned the literal
// string. Both ops must answer false: the field named a value, not a table, and
// "the row is absent" presupposes a table to search.
func TestRowPresenceExactPayloadKeyWins(t *testing.T) {
	enc := sessionRecordBytes(t)
	// "session/b/42" is a PRESENT row in the record and "session/b/7" is an
	// absent one, so the two literal keys pin both directions: without the fix
	// the first answers row_exists true and the second answers row_absent true.
	m := Metadata{
		"session":      NewRecord(enc),
		"session/b/42": NewString("literal-wins"),
		"session/b/7":  NewString("literal-wins-too"),
	}

	// The control: lookupPath resolves both fields to the LITERAL value, which
	// is the answer row presence has to agree with.
	for _, field := range []string{"session/b/42", "session/b/7"} {
		got, ok := lookupPath(m, field)
		if !ok || got.Kind != ValueString {
			t.Fatalf("precondition: lookupPath(%q) = (%+v, %v), want the literal string", field, got, ok)
		}
	}

	for _, c := range []struct {
		name  string
		f     Filter
		want  bool
		alone bool // the answer when the literal key is NOT present
	}{
		{"row_exists over a literal key shadowing a present row",
			Filter{Op: FilterRowExists, Field: "session/b/42"}, false, true},
		{"row_absent over a literal key shadowing a present row",
			Filter{Op: FilterRowAbsent, Field: "session/b/42"}, false, false},
		{"row_exists over a literal key shadowing an absent row",
			Filter{Op: FilterRowExists, Field: "session/b/7"}, false, false},
		{"row_absent over a literal key shadowing an absent row",
			Filter{Op: FilterRowAbsent, Field: "session/b/7"}, false, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := compileOrFail(t, c.f)
			if got := p(m); got != c.want {
				t.Errorf("%s over a shadowed field = %v, want %v — the literal payload key must win",
					mustOpName(c.f.Op), got, c.want)
			}
			// Drop the literal key and the SAME filter resolves through the
			// record again, so the fix shadows only what it should.
			bare := Metadata{"session": NewRecord(enc)}
			if got := p(bare); got != c.alone {
				t.Errorf("%s with no literal key = %v, want %v — record resolution must be unaffected",
					mustOpName(c.f.Op), got, c.alone)
			}
		})
	}
}

// TestRowAbsentRequiresATableAtTheField pins row_absent's contract against the
// case that used to slip through it: a field the record does not have AT ALL.
//
// resolveSchema answers Kind == Absent for a field name that is not in the
// record's schema, and for an out-of-range positional segment, BEFORE it looks
// for a row — so "session/zzz/42" reached the closure as Absent and read as "the
// row is missing from that table". That contradicts the documented contract
// ("the row is absent" presupposes a table to search) and it disagreed with the
// neighbouring case: a scalar at the field errors with ErrPath and correctly
// answers false, though both are "no table at that field".
//
// row_absent is now true only when the field itself resolves to a TABLE. The
// row_exists column is asserted alongside it in every case, because "both false"
// is precisely the outcome the contract calls for when the question cannot be
// asked.
func TestRowAbsentRequiresATableAtTheField(t *testing.T) {
	m := Metadata{"session": NewRecord(sessionRecordBytes(t))}
	// A dynamic-mode record has no schema at all; a field it does not carry
	// must answer the same way a schema-mode unknown field does.
	dyn := Metadata{"session": NewRecord(goodDynamicRecord(t, 5))}

	cases := []struct {
		name       string
		m          Metadata
		field      string
		wantExists bool
		wantAbsent bool
	}{
		// The regression: the schema has no "zzz" field whatsoever. There is no
		// table to search, so neither op may assert anything.
		{"field absent from the schema", m, "session/zzz/42", false, false},
		// Same shape by position: #9 is past the end of a 6-field schema.
		{"positional segment out of range", m, "session/#9/42", false, false},
		// A dynamic record that simply has no such field.
		{"field absent from a dynamic record", dyn, "session/zzz/42", false, false},
		// The neighbouring case that was always right, kept here so the two are
		// asserted side by side: a SCALAR at the field errors with ErrPath.
		{"scalar at the field", m, "session/rc/1", false, false},
		{"scalar at the field, dynamic", dyn, "session/a/1", false, false},
		// The genuine cases must be unaffected: a real table with the row
		// missing, and one with the row present.
		{"table present, row missing", m, "session/b/7", false, true},
		{"table present, row present", m, "session/b/42", true, false},
		// Positional addressing of a REAL table still works — the fix is about
		// what the field resolves to, not how it is spelled.
		{"table addressed by position, row missing", m, "session/#5/7", false, true},
		{"table addressed by position, row present", m, "session/#5/42", true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, tc := range []struct {
				op   FilterOp
				want bool
			}{{FilterRowExists, c.wantExists}, {FilterRowAbsent, c.wantAbsent}} {
				p := compileOrFail(t, Filter{Op: tc.op, Field: c.field})
				if got := p(c.m); got != tc.want {
					t.Errorf("%s(%q) = %v, want %v", mustOpName(tc.op), c.field, got, tc.want)
				}
			}
		})
	}
}
