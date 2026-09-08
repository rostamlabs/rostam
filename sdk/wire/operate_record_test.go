// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
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
		{2, 1, 1, 't', OperateTypeTable, 7, 0, 0, 0, 1, 2, 0, 0}, // dynamic row with a zero-length key
		{2, 1, 0}, // dynamic field with a zero-length name
		(&Record{Mode: OperateModeDynamic, Fields: []Field{{Name: "t", Cell: Cell{Type: OperateTypeTable}, Table: &Table{
			Rows: []Row{{Key: []byte("k"), Cols: []Col{{Name: "", Cell: Cell{Type: OperateTypeU8}}}}}}}}}).Encode()} // dynamic column with a zero-length name
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
//
// Both build their input with a ONE-BYTE field name. An earlier revision
// used a zero-length name, which decodeDynamicRecord rejects at its nlen
// check before it ever reads the value — so the tests passed without the
// canonical check existing at all. Each one now asserts the canonically
// spelled twin decodes, which is what proves the decoder reaches the value.
func TestDecodeRecordRejectsNonCanonicalVarint(t *testing.T) {
	// mode=dynamic, 1 field, name "a", type UVARINT, value = over-long
	// two-byte encoding of 1 ([0x81, 0x00]; canonical is one byte, 0x01).
	b := []byte{OperateModeDynamic, 1, 1, 'a', OperateTypeUVarint, 0x81, 0x00}
	if _, err := DecodeRecord(b); !errors.Is(err, ErrOperateRecord) {
		t.Fatalf("non-canonical UVARINT: err = %v, want ErrOperateRecord", err)
	}
	canonical := []byte{OperateModeDynamic, 1, 1, 'a', OperateTypeUVarint, 0x01}
	r, err := DecodeRecord(canonical)
	if err != nil {
		t.Fatalf("the canonical spelling must decode, or the case above never reaches the value: %v", err)
	}
	if len(r.Fields) != 1 || r.Fields[0].Cell.U != 1 {
		t.Fatalf("canonical UVARINT decoded to %+v", r.Fields)
	}
}

func TestDecodeRecordRejectsNonCanonicalBytesLength(t *testing.T) {
	// mode=dynamic, 1 field, name "a", type BYTES, length = over-long
	// two-byte encoding of 0 ([0x80, 0x00]; canonical is one byte, 0x00).
	b := []byte{OperateModeDynamic, 1, 1, 'a', OperateTypeBytes, 0x80, 0x00}
	if _, err := DecodeRecord(b); !errors.Is(err, ErrOperateRecord) {
		t.Fatalf("non-canonical BYTES length: err = %v, want ErrOperateRecord", err)
	}
	canonical := []byte{OperateModeDynamic, 1, 1, 'a', OperateTypeBytes, 0x00}
	r, err := DecodeRecord(canonical)
	if err != nil {
		t.Fatalf("the canonical spelling must decode, or the case above never reaches the value: %v", err)
	}
	if len(r.Fields) != 1 || len(r.Fields[0].Cell.B) != 0 {
		t.Fatalf("canonical BYTES decoded to %+v", r.Fields)
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
	if _, err := DecodeRecord(b); !errors.Is(err, ErrOperateRecord) {
		t.Fatalf("non-canonical (signaling NaN) F32: err = %v, want ErrOperateRecord", err)
	}
	// The canonical quiet NaN (F32 bits 0x7FC00000) is the one spelling that
	// must survive, which is also what proves the case above reaches the
	// value rather than failing earlier in the frame.
	ok := append([]byte{OperateModeSchema}, s.Encode()...)
	ok = binary.LittleEndian.AppendUint32(ok, 0x7FC00000)
	got, err := DecodeRecord(ok)
	if err != nil {
		t.Fatalf("canonical quiet-NaN F32 rejected: %v", err)
	}
	if !math.IsNaN(got.Fields[0].Cell.F) || !bytes.Equal(got.Encode(), ok) {
		t.Fatalf("canonical NaN did not round-trip: %+v", got.Fields[0].Cell)
	}
}

// TestDecodeRecordRejectsNonCanonicalF64 is the F64 half of the same rule:
// AppendCellData canonicalizes every NaN (design doc §2.5), so a stored
// quiet NaN with a non-zero payload decodes fine yet re-encodes to the
// canonical pattern — DecodeRecord must reject it rather than silently
// mutate it on the next round trip.
func TestDecodeRecordRejectsNonCanonicalF64(t *testing.T) {
	s := &Schema{Version: 1, Fields: []FieldDef{{Type: OperateTypeF64}}}
	bad := append([]byte{OperateModeSchema}, s.Encode()...)
	bad = binary.LittleEndian.AppendUint64(bad, 0x7FF8000000000001) // quiet NaN, payload 1
	if _, err := DecodeRecord(bad); !errors.Is(err, ErrOperateRecord) {
		t.Fatalf("non-canonical F64 NaN: err = %v, want ErrOperateRecord", err)
	}
	ok := append([]byte{OperateModeSchema}, s.Encode()...)
	ok = binary.LittleEndian.AppendUint64(ok, 0x7FF8000000000000)
	got, err := DecodeRecord(ok)
	if err != nil {
		t.Fatalf("canonical quiet-NaN F64 rejected: %v", err)
	}
	if !math.IsNaN(got.Fields[0].Cell.F) || !bytes.Equal(got.Encode(), ok) {
		t.Fatalf("canonical NaN did not round-trip: %+v", got.Fields[0].Cell)
	}
	// A finite F64 keeps its bits verbatim: adding F64 to the canonical check
	// must not cost the ordinary values anything.
	fin := append([]byte{OperateModeSchema}, s.Encode()...)
	fin = binary.LittleEndian.AppendUint64(fin, math.Float64bits(-0.5))
	if _, err := DecodeRecord(fin); err != nil {
		t.Fatalf("finite F64 rejected: %v", err)
	}
}

// TestDecodeRecordRejectsFixedZeroWidth covers FIXED(0), which is not a
// legal width (design doc §2.1: FIXED(n) is 1-255 bytes). A zero-width cell
// would be a second spelling of UNSET — it encodes and decodes as no bytes
// at all — so the codec rejects it wherever a width is stored: a dynamic
// field header, a dynamic column header, and a bare tagged cell.
func TestDecodeRecordRejectsFixedZeroWidth(t *testing.T) {
	field := []byte{OperateModeDynamic, 1, 1, 'a', OperateTypeFixed, 0}
	if _, err := DecodeRecord(field); !errors.Is(err, ErrOperateRecord) {
		t.Fatalf("FIXED(0) field: err = %v, want ErrOperateRecord", err)
	}
	// The same record with a width of 1 and one data byte must decode, so the
	// case above is failing on the width and not on the frame's shape.
	okField := []byte{OperateModeDynamic, 1, 1, 'a', OperateTypeFixed, 1, 0x7A}
	if _, err := DecodeRecord(okField); err != nil {
		t.Fatalf("FIXED(1) field rejected: %v", err)
	}

	// A dynamic table holding one row with one FIXED(0) column:
	// [byteLen][cap 0][policy 0][byColLen 0][nRows 1][rowLen][klen 1]['k']
	// [nCols 1][nlen 1]['c'][type FIXED][n 0].
	row := []byte{1, 'k', 1, 1, 'c', OperateTypeFixed, 0}
	inner := append([]byte{0, 0, 0, 1, byte(len(row))}, row...)
	col := append([]byte{OperateModeDynamic, 1, 1, 't', OperateTypeTable, byte(len(inner))}, inner...)
	if _, err := DecodeRecord(col); !errors.Is(err, ErrOperateRecord) {
		t.Fatalf("FIXED(0) column: err = %v, want ErrOperateRecord", err)
	}
	okRow := []byte{1, 'k', 1, 1, 'c', OperateTypeFixed, 1, 0x7A}
	okInner := append([]byte{0, 0, 0, 1, byte(len(okRow))}, okRow...)
	okCol := append([]byte{OperateModeDynamic, 1, 1, 't', OperateTypeTable, byte(len(okInner))}, okInner...)
	if _, err := DecodeRecord(okCol); err != nil {
		t.Fatalf("FIXED(1) column rejected: %v", err)
	}

	if _, _, err := DecodeTaggedCell([]byte{OperateTypeFixed, 0}); !errors.Is(err, ErrOperateType) {
		t.Fatalf("DecodeTaggedCell FIXED(0): err = %v, want ErrOperateType", err)
	}
	if _, _, err := DecodeCellData(OperateTypeFixed, 0, []byte{1, 2, 3}); !errors.Is(err, ErrOperateType) {
		t.Fatalf("DecodeCellData FIXED(0): err = %v, want ErrOperateType", err)
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
