// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// valuesEqual compares two vtypes.Value the way resultsEqual compares two
// Results: NaN floats are equal to each other (reflect/== would not agree).
func valuesEqual(a, b vtypes.Value) bool {
	if a.Kind != b.Kind {
		return false
	}
	switch a.Kind {
	case vtypes.ValueFloat:
		if math.IsNaN(a.Flt) || math.IsNaN(b.Flt) {
			return math.IsNaN(a.Flt) && math.IsNaN(b.Flt)
		}
		return a.Flt == b.Flt
	default:
		return a.Equal(b)
	}
}

// TestCellValueMapping is the table for the value-mapping ruling: every
// OperateType* tag, exercised at values that pin the interesting boundaries
// (the U64/UVarint MaxInt64 edge, a negative signed value, empty and
// multi-byte BYTES/FIXED, and UNSET/TABLE both declining).
func TestCellValueMapping(t *testing.T) {
	cases := []struct {
		name string
		cell wire.Cell
		want vtypes.Value
		ok   bool
	}{
		{"U8", wire.Cell{Type: wire.OperateTypeU8, U: 7}, vtypes.NewInt(7), true},
		{"U8 max", wire.Cell{Type: wire.OperateTypeU8, U: 255}, vtypes.NewInt(255), true},
		{"U16", wire.Cell{Type: wire.OperateTypeU16, U: 65535}, vtypes.NewInt(65535), true},
		{"U32", wire.Cell{Type: wire.OperateTypeU32, U: 4294967295}, vtypes.NewInt(4294967295), true},
		{"U64 within int64", wire.Cell{Type: wire.OperateTypeU64, U: math.MaxInt64}, vtypes.NewInt(math.MaxInt64), true},
		{"U64 above int64", wire.Cell{Type: wire.OperateTypeU64, U: math.MaxInt64 + 1}, vtypes.NewFloat(float64(uint64(math.MaxInt64) + 1)), true},
		{"U64 max", wire.Cell{Type: wire.OperateTypeU64, U: math.MaxUint64}, vtypes.NewFloat(float64(uint64(math.MaxUint64))), true},
		{"UVarint within int64", wire.Cell{Type: wire.OperateTypeUVarint, U: math.MaxInt64}, vtypes.NewInt(math.MaxInt64), true},
		{"UVarint above int64", wire.Cell{Type: wire.OperateTypeUVarint, U: math.MaxInt64 + 1}, vtypes.NewFloat(float64(uint64(math.MaxInt64) + 1)), true},
		{"I8 negative", wire.Cell{Type: wire.OperateTypeI8, U: su64(-5)}, vtypes.NewInt(-5), true},
		{"I8 positive", wire.Cell{Type: wire.OperateTypeI8, U: su64(5)}, vtypes.NewInt(5), true},
		{"I16 negative", wire.Cell{Type: wire.OperateTypeI16, U: su64(-1234)}, vtypes.NewInt(-1234), true},
		{"I32 negative", wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}, vtypes.NewInt(-5), true},
		{"I64 negative", wire.Cell{Type: wire.OperateTypeI64, U: su64(math.MinInt64)}, vtypes.NewInt(math.MinInt64), true},
		{"I64 positive", wire.Cell{Type: wire.OperateTypeI64, U: su64(math.MaxInt64)}, vtypes.NewInt(math.MaxInt64), true},
		{"IVarint negative", wire.Cell{Type: wire.OperateTypeIVarint, U: su64(-99)}, vtypes.NewInt(-99), true},
		{"F32", wire.Cell{Type: wire.OperateTypeF32, F: 1.5}, vtypes.NewFloat(1.5), true},
		{"F32 NaN", wire.Cell{Type: wire.OperateTypeF32, F: math.NaN()}, vtypes.NewFloat(math.NaN()), true},
		{"F32 Inf", wire.Cell{Type: wire.OperateTypeF32, F: math.Inf(1)}, vtypes.NewFloat(math.Inf(1)), true},
		{"F64", wire.Cell{Type: wire.OperateTypeF64, F: -2.25}, vtypes.NewFloat(-2.25), true},
		{"F64 NaN", wire.Cell{Type: wire.OperateTypeF64, F: math.NaN()}, vtypes.NewFloat(math.NaN()), true},
		{"F64 -Inf", wire.Cell{Type: wire.OperateTypeF64, F: math.Inf(-1)}, vtypes.NewFloat(math.Inf(-1)), true},
		{"BYTES", wire.Cell{Type: wire.OperateTypeBytes, B: []byte("hi")}, vtypes.NewString("hi"), true},
		{"BYTES empty", wire.Cell{Type: wire.OperateTypeBytes, B: []byte{}}, vtypes.NewString(""), true},
		{"BYTES nil", wire.Cell{Type: wire.OperateTypeBytes, B: nil}, vtypes.NewString(""), true},
		{"FIXED", wire.Cell{Type: wire.OperateTypeFixed, N: 3, B: []byte("abc")}, vtypes.NewString("abc"), true},
		{"FIXED binary", wire.Cell{Type: wire.OperateTypeFixed, N: 2, B: []byte{0x00, 0xFF}}, vtypes.NewString(string([]byte{0x00, 0xFF})), true},
		{"Unset", wire.Cell{Type: wire.OperateTypeUnset}, vtypes.Value{}, false},
		{"Table", wire.Cell{Type: wire.OperateTypeTable}, vtypes.Value{}, false},
		{"unknown type", wire.Cell{Type: wire.OperateTypeCount}, vtypes.Value{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := CellValue(tc.cell)
			if ok != tc.ok {
				t.Fatalf("CellValue(%+v) ok = %v, want %v", tc.cell, ok, tc.ok)
			}
			if ok && !valuesEqual(got, tc.want) {
				t.Fatalf("CellValue(%+v) = %+v, want %+v", tc.cell, got, tc.want)
			}
		})
	}
}

// TestCellValueEveryOperateType checks every declared OperateType* tag
// (0..OperateTypeCount-1) is handled by CellValue without panicking, and
// that the type-dispatch table has no silent gap for a tag this package
// does not yet know by name.
func TestCellValueEveryOperateType(t *testing.T) {
	scalarWidth := map[uint8]int{
		wire.OperateTypeU8: 1, wire.OperateTypeU16: 2, wire.OperateTypeU32: 4, wire.OperateTypeU64: 8,
		wire.OperateTypeI8: 1, wire.OperateTypeI16: 2, wire.OperateTypeI32: 4, wire.OperateTypeI64: 8,
	}
	for typ := uint8(0); typ < wire.OperateTypeCount; typ++ {
		c := wire.Cell{Type: typ, N: 1, B: []byte{1}}
		if _, ok := scalarWidth[typ]; ok {
			c.U = 1
		}
		got, ok := CellValue(c)
		switch typ {
		case wire.OperateTypeUnset, wire.OperateTypeTable:
			if ok {
				t.Errorf("type %d: CellValue ok = true, want false", typ)
			}
		default:
			if !ok {
				t.Errorf("type %d: CellValue ok = false, want true", typ)
			}
			_ = got
		}
	}
}

// TestResultValue covers ResultValue for every Kind.
func TestResultValue(t *testing.T) {
	cases := []struct {
		name string
		res  Result
		want vtypes.Value
		ok   bool
	}{
		{"Absent", Result{Kind: Absent}, vtypes.Value{}, false},
		{"Scalar int", Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeU32, U: 42}}, vtypes.NewInt(42), true},
		{"Scalar unset", Result{Kind: Scalar, Cell: wire.Cell{Type: wire.OperateTypeUnset}}, vtypes.Value{}, false},
		{"Count", Result{Kind: Count, Count: 5}, vtypes.NewInt(5), true},
		{"Count zero", Result{Kind: Count, Count: 0}, vtypes.NewInt(0), true},
		{"RowPresent", Result{Kind: RowPresent}, vtypes.Value{}, false},
		{"Table", Result{Kind: Table}, vtypes.Value{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ResultValue(tc.res)
			if ok != tc.ok {
				t.Fatalf("ResultValue(%+v) ok = %v, want %v", tc.res, ok, tc.ok)
			}
			if ok && !valuesEqual(got, tc.want) {
				t.Fatalf("ResultValue(%+v) = %+v, want %+v", tc.res, got, tc.want)
			}
		})
	}
}

// entryFields extracts just the Field names, in order, for a compact
// comparison against the expected slice.
func entryFields(es []Entry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.Field
	}
	return out
}

func fieldsEqual(a, b []string) bool {
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

// TestIndexEntriesSessionRecord pins IndexEntries on the shared session
// record fixture (sessionRecord from resolve_test.go): rc, bc, hist, bal,
// tag are top-level scalars and b is a table, so schema field order puts
// b's "#count" entry at position 5, last.
func TestIndexEntriesSessionRecord(t *testing.T) {
	enc, _ := sessionRecord(t, true)
	r := NewResolver(8)
	entries, err := r.IndexEntries(enc)
	if err != nil {
		t.Fatalf("IndexEntries: %v", err)
	}
	wantFields := []string{"rc", "bc", "hist", "bal", "tag", "b#count"}
	if !fieldsEqual(entryFields(entries), wantFields) {
		t.Fatalf("IndexEntries fields = %v, want %v", entryFields(entries), wantFields)
	}
	want := map[string]vtypes.Value{
		"rc":      vtypes.NewInt(7),
		"bc":      vtypes.NewInt(3),
		"hist":    vtypes.NewInt(1234),
		"bal":     vtypes.NewInt(-5),
		"tag":     vtypes.NewString("de"),
		"b#count": vtypes.NewInt(2),
	}
	for _, e := range entries {
		w, ok := want[e.Field]
		if !ok {
			t.Fatalf("unexpected entry field %q", e.Field)
		}
		if !valuesEqual(e.Value, w) {
			t.Errorf("entry %q = %+v, want %+v", e.Field, e.Value, w)
		}
	}
}

// TestIndexEntriesSessionRecordWithoutNames is the same record with
// StoreNames false: field labels fall back to "#pos".
func TestIndexEntriesSessionRecordWithoutNames(t *testing.T) {
	enc, _ := sessionRecord(t, false)
	r := NewResolver(8)
	entries, err := r.IndexEntries(enc)
	if err != nil {
		t.Fatalf("IndexEntries: %v", err)
	}
	wantFields := []string{"#0", "#1", "#2", "#3", "#4", "#5#count"}
	if !fieldsEqual(entryFields(entries), wantFields) {
		t.Fatalf("IndexEntries fields = %v, want %v", entryFields(entries), wantFields)
	}
	want := map[string]vtypes.Value{
		"#0":       vtypes.NewInt(7),
		"#1":       vtypes.NewInt(3),
		"#2":       vtypes.NewInt(1234),
		"#3":       vtypes.NewInt(-5),
		"#4":       vtypes.NewString("de"),
		"#5#count": vtypes.NewInt(2),
	}
	for _, e := range entries {
		w, ok := want[e.Field]
		if !ok {
			t.Fatalf("unexpected entry field %q", e.Field)
		}
		if !valuesEqual(e.Value, w) {
			t.Errorf("entry %q = %+v, want %+v", e.Field, e.Value, w)
		}
	}
}

// TestIndexEntriesDynamicRecord covers the dynamic example (dynamicRecord
// from resolve_test.go): cnt, raw, t (a table), stored in ascending name
// order ("cnt" < "raw" < "t").
func TestIndexEntriesDynamicRecord(t *testing.T) {
	enc, _ := dynamicRecord(t)
	entries, err := (&Resolver{}).IndexEntries(enc)
	if err != nil {
		t.Fatalf("IndexEntries: %v", err)
	}
	wantFields := []string{"cnt", "raw", "t#count"}
	if !fieldsEqual(entryFields(entries), wantFields) {
		t.Fatalf("IndexEntries fields = %v, want %v", entryFields(entries), wantFields)
	}
	want := map[string]vtypes.Value{
		"cnt":     vtypes.NewInt(9),
		"raw":     vtypes.NewString("hi"),
		"t#count": vtypes.NewInt(2),
	}
	for _, e := range entries {
		w, ok := want[e.Field]
		if !ok {
			t.Fatalf("unexpected entry field %q", e.Field)
		}
		if !valuesEqual(e.Value, w) {
			t.Errorf("entry %q = %+v, want %+v", e.Field, e.Value, w)
		}
	}
}

// TestIndexEntriesSkipsNaN checks a NaN float field contributes no entry,
// in both modes, while its non-NaN siblings still do.
func TestIndexEntriesSkipsNaN(t *testing.T) {
	t.Run("schema", func(t *testing.T) {
		s := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
			{Name: "a", Type: wire.OperateTypeU32},
			{Name: "f", Type: wire.OperateTypeF64},
		}}
		if err := s.Validate(); err != nil {
			t.Fatalf("schema invalid: %v", err)
		}
		rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: []wire.Field{
			{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			{Cell: wire.Cell{Type: wire.OperateTypeF64, F: math.NaN()}},
		}}
		enc := rec.Encode()
		if enc == nil {
			t.Fatal("record failed to encode")
		}
		entries, err := NewResolver(4).IndexEntries(enc)
		if err != nil {
			t.Fatalf("IndexEntries: %v", err)
		}
		if !fieldsEqual(entryFields(entries), []string{"a"}) {
			t.Fatalf("IndexEntries fields = %v, want [a]", entryFields(entries))
		}
	})
	t.Run("dynamic", func(t *testing.T) {
		rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
			{Name: "a", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			{Name: "f", Cell: wire.Cell{Type: wire.OperateTypeF32, F: math.NaN()}},
		}}
		enc := rec.Encode()
		if enc == nil {
			t.Fatal("record failed to encode")
		}
		entries, err := NewResolver(4).IndexEntries(enc)
		if err != nil {
			t.Fatalf("IndexEntries: %v", err)
		}
		if !fieldsEqual(entryFields(entries), []string{"a"}) {
			t.Fatalf("IndexEntries fields = %v, want [a]", entryFields(entries))
		}
	})
}

// TestIndexEntriesSkipsUnset checks a dynamic-mode UNSET field (legal in
// dynamic mode, per doc.go) contributes no entry.
func TestIndexEntriesSkipsUnset(t *testing.T) {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "a", Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
		{Name: "u", Cell: wire.Cell{Type: wire.OperateTypeUnset}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("record failed to encode")
	}
	entries, err := NewResolver(4).IndexEntries(enc)
	if err != nil {
		t.Fatalf("IndexEntries: %v", err)
	}
	if !fieldsEqual(entryFields(entries), []string{"a"}) {
		t.Fatalf("IndexEntries fields = %v, want [a]", entryFields(entries))
	}
}

// TestIndexEntriesMatchesOracle checks IndexEntries against a from-scratch
// oracle built directly off the decoded tree, over the same random-record
// generators resolve.go's oracle test uses: same field set, same order, same
// values (NaN skip included), on every random record in both modes.
func TestIndexEntriesMatchesOracle(t *testing.T) {
	records := 500
	if testing.Short() {
		records = 100
	}
	rng := rand.New(rand.NewSource(20260908))
	r := NewResolver(32)
	for i := 0; i < records; i++ {
		var enc []byte
		var dec *wire.Record
		var err error
		if i%2 == 0 {
			enc, dec, err = randomSchemaRecord(rng)
		} else {
			enc, dec, err = randomDynamicRecord(rng)
		}
		if err != nil {
			t.Fatalf("generator: %v", err)
		}
		got, gerr := r.IndexEntries(enc)
		if gerr != nil {
			t.Fatalf("record %d: IndexEntries: %v", i, gerr)
		}
		want := oracleIndexEntries(t, dec)
		if len(got) != len(want) {
			t.Fatalf("record %d: got %d entries %v, want %d %v", i, len(got), entryFields(got), len(want), entryFields(want))
		}
		for j := range got {
			if got[j].Field != want[j].Field {
				t.Fatalf("record %d entry %d: field %q, want %q (got %v, want %v)", i, j, got[j].Field, want[j].Field, entryFields(got), entryFields(want))
			}
			if !valuesEqual(got[j].Value, want[j].Value) {
				t.Fatalf("record %d entry %q: value %+v, want %+v", i, got[j].Field, got[j].Value, want[j].Value)
			}
		}
	}
}

// oracleIndexEntries computes IndexEntries' expected output directly off
// the decoded tree, independent of resolve.go's byte-walking code.
func oracleIndexEntries(t *testing.T, rec *wire.Record) []Entry {
	t.Helper()
	var entries []Entry
	label := func(storeNames bool, name string, pos int) string {
		if storeNames {
			return name
		}
		return "#" + itoa(pos)
	}
	switch rec.Mode {
	case wire.OperateModeSchema:
		s := rec.Schema
		for i := range s.Fields {
			fd := &s.Fields[i]
			name := label(s.StoreNames, fd.Name, i)
			if fd.Type == wire.OperateTypeTable {
				entries = append(entries, Entry{Field: name + "#count", Value: vtypes.NewInt(int64(len(rec.Fields[i].Table.Rows)))})
				continue
			}
			v, ok := CellValue(rec.Fields[i].Cell)
			if !ok || (v.Kind == vtypes.ValueFloat && math.IsNaN(v.Flt)) {
				continue
			}
			entries = append(entries, Entry{Field: name, Value: v})
		}
	case wire.OperateModeDynamic:
		for i := range rec.Fields {
			f := &rec.Fields[i]
			if f.Cell.Type == wire.OperateTypeTable {
				entries = append(entries, Entry{Field: f.Name + "#count", Value: vtypes.NewInt(int64(len(f.Table.Rows)))})
				continue
			}
			v, ok := CellValue(f.Cell)
			if !ok || (v.Kind == vtypes.ValueFloat && math.IsNaN(v.Flt)) {
				continue
			}
			entries = append(entries, Entry{Field: f.Name, Value: v})
		}
	}
	return entries
}

// itoa avoids importing strconv twice in the test file's oracle helper.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

// TestIndexEntriesHostileNeverPanics drives IndexEntries with every prefix
// and every single-byte mutation of both example records, plus random
// bytes: never panic, never mutate the input, and any error must wrap
// ErrRecord (a malformed record) or ErrPath (never — IndexEntries takes no
// path — so in practice only ErrRecord).
func TestIndexEntriesHostileNeverPanics(t *testing.T) {
	sess, _ := sessionRecord(t, true)
	dyn, _ := dynamicRecord(t)
	r := NewResolver(16)

	run := func(name string, in []byte) {
		before := append([]byte(nil), in...)
		entries, err := r.IndexEntries(in)
		if err != nil {
			if errClass(err) == "other" {
				t.Fatalf("%s: error outside ErrPath/ErrRecord: %v", name, err)
			}
			if entries != nil {
				t.Fatalf("%s: error %v returned non-nil entries %v", name, err, entries)
			}
		}
		if !bytes.Equal(before, in) {
			t.Fatalf("%s: IndexEntries mutated its input", name)
		}
	}

	for _, base := range [][]byte{sess, dyn} {
		for n := 0; n <= len(base); n++ {
			run("prefix", base[:n:n])
		}
		mut := append([]byte(nil), base...)
		for i := range base {
			orig := mut[i]
			for v := 0; v < 256; v++ {
				mut[i] = byte(v)
				run("mutation", mut)
			}
			mut[i] = orig
		}
	}

	rng := rand.New(rand.NewSource(11))
	n := 4000
	if testing.Short() {
		n = 500
	}
	for i := 0; i < n; i++ {
		buf := make([]byte, rng.Intn(40))
		for j := range buf {
			buf[j] = byte(rng.Intn(256))
		}
		run("random", buf)
	}
}

// TestIndexEntriesEmptyRecord checks the one-byte-short and empty cases
// IndexEntries must reject cleanly.
func TestIndexEntriesEmptyRecord(t *testing.T) {
	r := NewResolver(4)
	if _, err := r.IndexEntries(nil); !bytesIsRecordErr(err) {
		t.Fatalf("IndexEntries(nil) error = %v, want ErrRecord", err)
	}
	if _, err := r.IndexEntries([]byte{}); !bytesIsRecordErr(err) {
		t.Fatalf("IndexEntries([]) error = %v, want ErrRecord", err)
	}
	if _, err := r.IndexEntries([]byte{0xFF}); !bytesIsRecordErr(err) {
		t.Fatalf("IndexEntries(unknown mode) error = %v, want ErrRecord", err)
	}
}

func bytesIsRecordErr(err error) bool {
	return err != nil && errClass(err) == "record"
}

// TestIndexEntriesNoNamesTableColumnsNotIndexed pins the indexing ruling
// directly: table columns are never indexed, only the top-level "#count".
func TestIndexEntriesNoNamesTableColumnsNotIndexed(t *testing.T) {
	enc, _ := sessionRecord(t, true)
	entries, err := NewResolver(4).IndexEntries(enc)
	if err != nil {
		t.Fatalf("IndexEntries: %v", err)
	}
	for _, e := range entries {
		if e.Field == "hi" || e.Field == "lo" || e.Field == "b/42/hi" {
			t.Fatalf("table column leaked into index entries: %q", e.Field)
		}
	}
}
