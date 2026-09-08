// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// fuzzPaths is the fixed set of paths FuzzResolve checks against every
// record wire.DecodeRecord accepts out of the fuzzer's corpus. It is drawn
// from the two seed fixtures' own field/row/column names — the session
// example (rc, bc, hist, bal, tag, b/hi/lo) and the dynamic example
// (cnt, raw, t/hi/lo) — plus deliberate near-misses: an absent name (zz), an
// out-of-range position (#9), a #count on a field that may not be a table,
// a row on a field that may not be one, and a row+column on a field that may
// not have a row at that key. The set is FIXED rather than randomly drawn
// (contrast oracle_test.go's randomPath) because the fuzzer already varies
// the record bytes; adding a second axis of randomness would make a
// discovered crasher harder to reproduce from its saved corpus entry.
var fuzzPaths = mustParsePaths([]string{
	"rc", "bc", "hist", "bal", "tag", "b", "cnt", "raw", "t", "zz",
	"#0", "#1", "#5", "#9",
	"b#count", "t#count", "rc#count", "zz#count",
	"b/42", "b/99", "b/7", `b/"DE"`, `b/"XX"`,
	"t/42", "t/7", `t/"DE"`, `t/"XX"`,
	"b/42/hi", "b/42/lo", "b/42/#0", "b/42/#9", "b/42/zz",
	"t/42/hi", "t/42/lo", `t/"DE"/hi`, `t/"DE"/lo`,
	"#5/42/#0", "#0/42",
})

// mustParsePaths parses every path string in ss, panicking on the first
// failure. Called only from a package-level var initializer with a fixed,
// hand-checked literal list, so a panic here is a bug in this file, not
// something a test run needs to recover from.
func mustParsePaths(ss []string) []Path {
	paths := make([]Path, 0, len(ss))
	for _, s := range ss {
		p, err := ParsePath(s)
		if err != nil {
			panic("fuzzPaths: " + s + ": " + err.Error())
		}
		paths = append(paths, p)
	}
	return paths
}

// fuzzLEKey encodes v as an 8-byte little-endian table row key — what
// client.KeyU64 (and a decimal row-key segment, in both schema and dynamic
// mode) produce — for the seed fixtures below.
func fuzzLEKey(v uint64) []byte {
	b := make([]byte, 8)
	for i := range b {
		b[i] = byte(v >> (8 * uint(i)))
	}
	return b
}

// fuzzSessionSeed builds the same schema-mode session record
// resolve_test.go's sessionRecord(t, true) and record_filter_test.go's
// sessionRecordBytes build, without needing a *testing.T (f.Add seeding runs
// before any subtest exists):
//
//	#0 rc   U8      = 7
//	#1 bc   U8      = 3
//	#2 hist U32     = 1234
//	#3 bal  I32     = -5
//	#4 tag  BYTES   = "de"
//	#5 b    TABLE(key U64; c0 "hi" U32, c1 "lo" U32) = {42: (500, 1), 99: (7, 2)}
func fuzzSessionSeed(f *testing.F) []byte {
	s := &wire.Schema{
		Version:    3,
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
		f.Fatalf("session seed schema invalid: %v", err)
	}
	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 3}},
		{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1234}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: fuzzLEKey(42), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			}},
			{Key: fuzzLEKey(99), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 2}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		f.Fatal("session seed failed to encode")
	}
	return enc
}

// fuzzDynamicSeed builds the same dynamic-mode record resolve_test.go's
// dynamicRecord(t) builds, without needing a *testing.T:
//
//	cnt U32   = 9
//	raw BYTES = "hi"
//	t   TABLE: {LE(42): (hi=500, lo=1), "DE": (hi=7)}
func fuzzDynamicSeed(f *testing.F) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "cnt", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 9}},
		{Name: "raw", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hi")}},
		{Name: "t", Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: fuzzLEKey(42), Cols: []wire.Col{
				{Name: "hi", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Name: "lo", Cell: wire.Cell{Type: wire.OperateTypeU8, U: 1}},
			}},
			{Key: []byte("DE"), Cols: []wire.Col{
				{Name: "hi", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		f.Fatal("dynamic seed failed to encode")
	}
	return enc
}

// FuzzResolve is the hostile-input fuzz target for the record resolver.
// Two properties hold for EVERY input, well-formed or not — the same ones
// TestResolveHostileNeverPanics pins with a hand-built corpus:
//
//   - Resolve never panics (the fuzzing engine itself catches that) and
//     never returns an error outside ErrPath/ErrRecord;
//   - Resolve never mutates its input.
//
// A third property holds only when the fuzzed bytes really are a
// well-formed record (wire.DecodeRecord accepts them): for every path in
// fuzzPaths, Resolve must agree with the tree oracle (resolveTree,
// oracle_test.go) — the same Result, or the same error class (ErrPath vs
// ErrRecord, via errors.Is/errClass) — because on a well-formed record the
// byte-level resolver and the decode-then-walk oracle are two
// implementations of the identical rule and must never diverge.
//
// Seeded with the session (schema-mode) and dynamic-mode example records
// (the same fixtures resolve_test.go's own tests use) plus a handful of
// single-byte flips and truncations of each, so the corpus starts from
// bytes already close to well-formed instead of making the mutator
// rediscover record shape from nothing.
func FuzzResolve(f *testing.F) {
	sessEnc := fuzzSessionSeed(f)
	dynEnc := fuzzDynamicSeed(f)
	f.Add(sessEnc)
	f.Add(dynEnc)

	for _, base := range [][]byte{sessEnc, dynEnc} {
		for _, off := range []int{0, 1, len(base) / 2, len(base) - 1} {
			if off < 0 || off >= len(base) {
				continue
			}
			mut := append([]byte(nil), base...)
			mut[off] ^= 0xFF
			f.Add(mut)
		}
		if len(base) > 1 {
			f.Add(base[:len(base)-1])
			f.Add(base[:len(base)/2])
		}
	}

	r := NewResolver(16)
	f.Fuzz(func(t *testing.T, b []byte) {
		before := append([]byte(nil), b...)

		for _, p := range fuzzPaths {
			res, err := r.Resolve(b, p)
			if err != nil && errClass(err) == "other" {
				t.Fatalf("Resolve returned an error outside ErrPath/ErrRecord: %v", err)
			}
			if err != nil && !isZeroResult(res) {
				t.Fatalf("error %v returned a non-zero result %+v", err, res)
			}
		}
		if !bytes.Equal(before, b) {
			t.Fatal("Resolve mutated its input")
		}

		dec, derr := wire.DecodeRecord(b)
		if derr != nil {
			return // not a well-formed record; the oracle has nothing to check.
		}
		for _, p := range fuzzPaths {
			got, gerr := r.Resolve(b, p)
			want, werr := resolveTree(dec, p)
			if errClass(gerr) != errClass(werr) {
				t.Fatalf("path %+v: error class %s (%v), oracle %s (%v)", p, errClass(gerr), gerr, errClass(werr), werr)
			}
			if gerr == nil && !resultsEqual(got, want) {
				t.Fatalf("path %+v: got %+v, oracle %+v", p, got, want)
			}
		}
	})
}
