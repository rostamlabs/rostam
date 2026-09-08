// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func sessionRecord() *Record {
	s := sessionSchema()
	r := &Record{Mode: OperateModeSchema, Schema: s, Fields: make([]Field, 4)}
	r.Fields[0].Cell = Cell{Type: OperateTypeU8, U: 7}
	r.Fields[1].Cell = Cell{Type: OperateTypeU8, U: 1}
	r.Fields[2].Cell = Cell{Type: OperateTypeU32, U: 0b1011}
	r.Fields[3].Cell = Cell{Type: OperateTypeTable}
	r.Fields[3].Table = &Table{Cap: 1024, Policy: OperatePolicyMinCol, ByCol: 2}
	for _, k := range []uint64{900, 3, 42} {
		key := make([]byte, 8)
		binary.LittleEndian.PutUint64(key, k)
		r.Fields[3].Table.Rows = append(r.Fields[3].Table.Rows, Row{Key: key, Cols: []Col{
			{Cell: Cell{Type: OperateTypeU16, U: uint64(k)}}, {Cell: Cell{Type: OperateTypeI64, U: su64(-1)}}, {Cell: Cell{Type: OperateTypeU32, U: 1000 + uint64(k)}}}})
	}
	return r
}

func TestRecordSchemaModeRoundtripAndSize(t *testing.T) {
	r := sessionRecord()
	b := r.Encode()
	schemaLen := len(sessionSchema().Encode())
	// 1 mode + schema + 1+1+4 fixed + nRows(1) + 3 rows × (8 key + 2 + 8 + 4)
	if want := 1 + schemaLen + 6 + 1 + 3*22; len(b) != want {
		t.Fatalf("encoded %d bytes, want %d", len(b), want)
	}
	got, err := DecodeRecord(b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Encode(), b) {
		t.Fatal("decode∘encode unstable")
	}
	rows := got.Fields[3].Table.Rows
	if len(rows) != 3 || binary.LittleEndian.Uint64(rows[0].Key) != 3 {
		t.Fatal("rows not sorted by key on encode")
	}
}

func TestRecordDynamicModeRoundtrip(t *testing.T) {
	r := &Record{Mode: OperateModeDynamic, Fields: []Field{
		{Name: "name", Cell: Cell{Type: OperateTypeBytes, B: []byte("vahid")}},
		{Name: "hits", Cell: Cell{Type: OperateTypeI64, U: 5}},
		{Name: "ev", Cell: Cell{Type: OperateTypeTable}, Table: &Table{Cap: 100, Policy: OperatePolicyMinKey, Rows: []Row{
			{Key: []byte("k2"), Cols: []Col{{Name: "p", Cell: Cell{Type: OperateTypeFixed, N: 2, B: []byte("xy")}}}},
			{Key: []byte("k1"), Cols: []Col{{Name: "z", Cell: Cell{Type: OperateTypeU8, U: 1}}, {Name: "a", Cell: Cell{Type: OperateTypeUVarint, U: 999}}}},
		}}},
	}}
	b := r.Encode()
	got, err := DecodeRecord(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Fields[0].Name != "ev" || got.Fields[2].Name != "name" {
		t.Fatal("fields not sorted by name")
	}
	if rows := got.Fields[0].Table.Rows; string(rows[0].Key) != "k1" || rows[0].Cols[0].Name != "a" {
		t.Fatal("rows/cols not sorted")
	}
	if !bytes.Equal(got.Encode(), b) {
		t.Fatal("unstable")
	}
}

func TestRecordEncodeDeterministic(t *testing.T) {
	want := sessionRecord().Encode()
	for i := 0; i < 20; i++ {
		r := sessionRecord()
		rows := r.Fields[3].Table.Rows
		rows[0], rows[2] = rows[2], rows[0]
		if !bytes.Equal(r.Encode(), want) {
			t.Fatal("row order leaked into bytes")
		}
	}
}

func TestDecodeRecordHostile(t *testing.T) {
	cases := [][]byte{{}, {0}, {3}, {1}, {2}, {1, 1, 0, 0, 1, OperateTypeU8}, {2, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F},
		append([]byte{1}, append(sessionSchema().Encode(), 1, 2, 3, 4, 5, 6, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F)...), // nRows huge
		{2, 1, 1, 'x', OperateTypeBytes, 0xFF, 0xFF, 0x03},
		{2, 1, 1, 't', OperateTypeTable, 7, 0, 0, 0, 1, 2, 0, 0}} // dynamic row with a zero-length key
	for _, b := range cases {
		if _, err := DecodeRecord(b); err == nil {
			t.Errorf("%x accepted", b)
		}
	}
}

// TestRecordSchemaZeroValueUsesSchemaType guards against Cell{}'s zero
// value (Type == OperateTypeU8, which is also Go's zero uint8) being
// mistaken for the schema's type when a tree's Fields slice is shorter than
// its schema: every never-touched fixed-width field and column must encode
// at its own declared width, not as a 1-byte U8 zero.
func TestRecordSchemaZeroValueUsesSchemaType(t *testing.T) {
	s := &Schema{Version: 1, Fields: []FieldDef{
		{Type: OperateTypeU8}, {Type: OperateTypeU32}, {Type: OperateTypeI64}, {Type: OperateTypeF64},
		{Type: OperateTypeFixed, N: 3},
		{Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeU8, Cols: []ColumnDef{
			{Type: OperateTypeU16}, {Type: OperateTypeF32},
		}}},
	}}
	// Only field 0 is touched; Fields is shorter than the schema, so fields
	// 1-5 fall back to Cell{} (the Go zero value), which has Type ==
	// OperateTypeU8 despite the schema saying U32, I64, F64, FIXED(3), TABLE.
	r := &Record{Mode: OperateModeSchema, Schema: s, Fields: []Field{{Cell: Cell{Type: OperateTypeU8, U: 9}}}}

	b := r.Encode()
	blob := s.Encode()
	if want := 1 + len(blob) + (1 + 4 + 8 + 8 + 3) + 1; len(b) != want { // +1 for the empty table's nRows=0
		t.Fatalf("encoded %d bytes, want %d", len(b), want)
	}

	got, err := DecodeRecord(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Fields[0].Cell.Type != OperateTypeU8 || got.Fields[0].Cell.U != 9 {
		t.Fatalf("field0 = %+v", got.Fields[0].Cell)
	}
	if got.Fields[1].Cell.Type != OperateTypeU32 || got.Fields[1].Cell.U != 0 {
		t.Fatalf("field1 = %+v", got.Fields[1].Cell)
	}
	if got.Fields[2].Cell.Type != OperateTypeI64 || got.Fields[2].Cell.U != 0 {
		t.Fatalf("field2 = %+v", got.Fields[2].Cell)
	}
	if got.Fields[3].Cell.Type != OperateTypeF64 || got.Fields[3].Cell.F != 0 {
		t.Fatalf("field3 = %+v", got.Fields[3].Cell)
	}
	if got.Fields[4].Cell.Type != OperateTypeFixed || len(got.Fields[4].Cell.B) != 3 {
		t.Fatalf("field4 = %+v", got.Fields[4].Cell)
	}
	for _, bb := range got.Fields[4].Cell.B {
		if bb != 0 {
			t.Fatalf("FIXED field not zero-filled: %+v", got.Fields[4].Cell)
		}
	}
	if got.Fields[5].Table == nil || len(got.Fields[5].Table.Rows) != 0 {
		t.Fatalf("table = %+v", got.Fields[5].Table)
	}
	if !bytes.Equal(got.Encode(), b) {
		t.Fatal("decode∘encode unstable")
	}
}

// TestDecodeRecordRejectsNonCanonicalVarint and
// TestDecodeRecordRejectsNonCanonicalBytesLength are regression tests for
// the bug FuzzDecodeRecord found: DecodeCellData accepts any length-legal
// LEB128 encoding (via binary.Uvarint), not only the minimal one
// AppendCellData produces, so a non-canonical encoding would decode
// successfully yet re-encode to different, shorter bytes.
func TestDecodeRecordRejectsNonCanonicalVarint(t *testing.T) {
	// mode=dynamic, 1 field, name "", type UVARINT, value = over-long
	// two-byte encoding of 1 ([0x81, 0x00]; canonical is one byte, 0x01).
	b := []byte{OperateModeDynamic, 1, 0, OperateTypeUVarint, 0x81, 0x00}
	if _, err := DecodeRecord(b); err == nil {
		t.Fatal("non-canonical UVARINT accepted")
	}
}

func TestDecodeRecordRejectsNonCanonicalBytesLength(t *testing.T) {
	// mode=dynamic, 1 field, name "", type BYTES, length = over-long
	// two-byte encoding of 0 ([0x80, 0x00]; canonical is one byte, 0x00).
	b := []byte{OperateModeDynamic, 1, 0, OperateTypeBytes, 0x80, 0x00}
	if _, err := DecodeRecord(b); err == nil {
		t.Fatal("non-canonical BYTES length accepted")
	}
}

// TestDecodeRecordRejectsNonCanonicalF32 is a regression test for a second
// bug fuzzing found: a schema-mode record whose sole F32 field holds a raw
// SIGNALING NaN bit pattern (exponent all-ones, mantissa nonzero, quiet bit
// clear). Decoding widens it through float64 and re-encoding narrows back;
// on common hardware that round trip sets the quiet bit, so the re-encoded
// bytes differ from the stored ones. A correctly canonicalizing writer never
// produces a signaling NaN (design doc §2.5), so DecodeRecord must reject
// one rather than accept it and silently mutate it on the next round trip.
func TestDecodeRecordRejectsNonCanonicalF32(t *testing.T) {
	s := &Schema{Version: 1, Fields: []FieldDef{{Type: OperateTypeF32}}}
	b := append([]byte{OperateModeSchema}, s.Encode()...)
	b = append(b, 0x30, 0x30, 0x80, 0xff) // LE bits 0xff803030: signaling NaN
	if _, err := DecodeRecord(b); err == nil {
		t.Fatal("non-canonical (signaling NaN) F32 accepted")
	}
}

func FuzzDecodeRecord(f *testing.F) {
	f.Add(sessionRecord().Encode())
	f.Add((&Record{Mode: OperateModeDynamic, Fields: []Field{{Name: "a", Cell: Cell{Type: OperateTypeU8, U: 1}}}}).Encode())
	f.Fuzz(func(t *testing.T, b []byte) {
		r, err := DecodeRecord(b)
		if err != nil {
			return
		}
		if !bytes.Equal(r.Encode(), b) {
			t.Fatal("decode∘encode not identity on accepted input")
		}
	})
}
