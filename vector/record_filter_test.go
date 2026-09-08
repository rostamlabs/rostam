// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/binary"
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
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
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
