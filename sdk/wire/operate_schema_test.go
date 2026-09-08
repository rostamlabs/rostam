// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
)

func sessionSchema() *Schema {
	return &Schema{Version: 1, Fields: []FieldDef{
		{Name: "rc", Type: OperateTypeU8}, {Name: "bc", Type: OperateTypeU8}, {Name: "hist", Type: OperateTypeU32},
		{Name: "b", Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeU64, Cols: []ColumnDef{
			{Name: "c", Type: OperateTypeU16}, {Name: "hi", Type: OperateTypeI64}, {Name: "t", Type: OperateTypeU32}},
			Cap: 1024, Policy: OperatePolicyMinCol, ByCol: 2}},
	}}
}

func TestSchemaRoundtripAndSize(t *testing.T) {
	s := sessionSchema()
	b := s.Encode()
	if len(b) > 20 {
		t.Fatalf("session schema is %d bytes, expected ≤ 20", len(b))
	}
	got, n, err := DecodeSchema(b)
	if err != nil || n != len(b) {
		t.Fatal(err, n)
	}
	if got.Version != 1 || len(got.Fields) != 4 || got.Fields[3].Table.Cap != 1024 || got.Fields[3].Table.ByCol != 2 {
		t.Fatalf("%+v", got)
	}
	if got.Fields[0].Name != "" {
		t.Fatal("names stored without StoreNames")
	}
	s.StoreNames = true
	got, _, _ = DecodeSchema(s.Encode())
	if got.Fields[3].Table.Cols[2].Name != "t" {
		t.Fatal("names not stored")
	}
}

func TestSchemaLayout(t *testing.T) {
	s := &Schema{Version: 1, Fields: []FieldDef{
		{Type: OperateTypeU8}, {Type: OperateTypeBytes}, {Type: OperateTypeU32}, {Type: OperateTypeFixed, N: 3},
		{Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeFixed, KeyN: 2, Cols: []ColumnDef{{Type: OperateTypeU16}, {Type: OperateTypeF64}}}},
		{Type: OperateTypeUVarint},
	}}
	l, err := s.Layout()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(l.FixedOff, []int{0, -1, 1, 5, -1, -1}) || l.FixedLen != 8 {
		t.Fatalf("%+v", l)
	}
	if !reflect.DeepEqual(l.VarTail, []int{1, 4, 5}) {
		t.Fatalf("tail %v", l.VarTail)
	}
	if tl := l.Tables[4]; tl.KeyWidth != 2 || tl.RowWidth != 12 || !reflect.DeepEqual(tl.ColOff, []int{2, 4}) {
		t.Fatalf("%+v", tl)
	}
}

func TestSchemaValidate(t *testing.T) {
	bad := []*Schema{
		{Fields: []FieldDef{{Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeU64, Cols: []ColumnDef{{Type: OperateTypeBytes}}}}}}, // variable column
		{Fields: []FieldDef{{Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeBytes}}}},                                            // variable key
		{Fields: []FieldDef{{Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeU8, Policy: OperatePolicyMinCol, ByCol: 3}}}},        // byCol out of range
		{Fields: []FieldDef{{Type: OperateTypeFixed, N: 0}}},                                                                                   // FIXED(0)
		{Fields: []FieldDef{{Type: OperateTypeUnset}}},                                                                                         // UNSET declared
		{Fields: []FieldDef{{Type: OperateTypeTable, Table: &TableDef{KeyType: OperateTypeU8, Cap: OperateMaxRows + 1}}}},                      // cap above hard max
	}
	for i, s := range bad {
		if err := s.Validate(); err == nil {
			t.Errorf("case %d accepted", i)
		}
	}
	if err := sessionSchema().Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestSchemaExtends(t *testing.T) {
	old := sessionSchema()
	ok := sessionSchema()
	ok.Version = 2
	ok.Fields = append(ok.Fields, FieldDef{Name: "x", Type: OperateTypeI16})
	ok.Fields[3].Table.Cols = append(ok.Fields[3].Table.Cols, ColumnDef{Name: "n", Type: OperateTypeU8})
	ok.Fields[3].Table.Cap = 512
	if err := old.Extends(ok); err != nil {
		t.Fatal(err)
	}
	for i, mut := range []func(s *Schema){
		func(s *Schema) { s.Fields[0].Type = OperateTypeU16 },          // widened type
		func(s *Schema) { s.Fields = s.Fields[:3] },                    // dropped field
		func(s *Schema) { s.Fields[3].Table.KeyType = OperateTypeU32 }, // key type change
		func(s *Schema) { s.Version = 1 },                              // version not increased
	} {
		n := sessionSchema()
		n.Version = 2
		mut(n)
		if err := old.Extends(n); err == nil {
			t.Errorf("mutation %d accepted", i)
		}
	}
}

func TestDecodeSchemaHostile(t *testing.T) {
	for _, b := range [][]byte{{}, {1, 0}, {1, 0, 0, 0xFF}, {1, 0, 0, 1, OperateTypeTable, OperateTypeU8, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F}} {
		if _, _, err := DecodeSchema(b); err == nil {
			t.Errorf("%x accepted", b)
		}
	}
}

func FuzzDecodeSchema(f *testing.F) {
	f.Add(sessionSchema().Encode())
	s := sessionSchema()
	s.StoreNames = true
	f.Add(s.Encode())
	f.Fuzz(func(t *testing.T, b []byte) {
		s, n, err := DecodeSchema(b)
		if err != nil {
			return
		}
		if n > len(b) {
			t.Fatal("over-read")
		}
		if err := s.Validate(); err != nil {
			t.Fatalf("decoded schema fails Validate: %v", err)
		}
		if _, err := s.Layout(); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(s.Encode(), b[:n]) {
			t.Fatal("re-encode differs")
		}
	})
}

// TestDecodeSchemaHostileCounts covers the field and column counts against
// the schema-blob budget. DecodeSchema is handed the whole stored record —
// up to maxRecordBytes — not just the blob, so CountFitsIn against the bytes
// remaining is not on its own a bound: a declared 65535 fields inside a
// megabyte-long record passes it and sizes a multi-megabyte slice. Every
// field and column costs at least one byte of a blob that can never legally
// exceed OperateMaxSchemaBytes, so the count is bounded by that too, before
// the allocation.
func TestDecodeSchemaHostileCounts(t *testing.T) {
	// [version u16][flags 0][nFields = 65535][65535+ filler bytes], every
	// filler byte a valid U8 field type, so nothing but the budget check
	// stops this from being parsed in full.
	blob := []byte{1, 0, 0}
	blob = binary.AppendUvarint(blob, OperateMaxFields)
	blob = append(blob, make([]byte, OperateMaxFields+16)...)
	if _, _, err := DecodeSchema(blob); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("hostile nFields: err = %v, want ErrOperateSchema", err)
	}
	if n := testing.AllocsPerRun(20, func() { _, _, _ = DecodeSchema(blob) }); n > 2 {
		t.Fatalf("hostile nFields allocated %v times; the count must be bounded before the make", n)
	}

	// The same shape one level down: one TABLE field declaring 65535
	// columns, with enough filler to satisfy CountFitsIn.
	cols := []byte{1, 0, 0, 1, OperateTypeTable, OperateTypeU8}
	cols = binary.AppendUvarint(cols, OperateMaxCols)
	cols = append(cols, make([]byte, OperateMaxCols+16)...)
	if _, _, err := DecodeSchema(cols); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("hostile nCols: err = %v, want ErrOperateSchema", err)
	}
	if n := testing.AllocsPerRun(20, func() { _, _, _ = DecodeSchema(cols) }); n > 3 {
		t.Fatalf("hostile nCols allocated %v times; the count must be bounded before the make", n)
	}

	// A schema at the far side of the budget still decodes: the bound is the
	// blob budget, not a smaller number picked to make the test pass.
	s := &Schema{Version: 3, Fields: make([]FieldDef, 1000)}
	for i := range s.Fields {
		s.Fields[i].Type = OperateTypeU8
	}
	enc := s.Encode()
	if len(enc) > OperateMaxSchemaBytes {
		t.Fatalf("fixture schema is %d bytes, over the %d-byte budget", len(enc), OperateMaxSchemaBytes)
	}
	got, n, err := DecodeSchema(enc)
	if err != nil || n != len(enc) || len(got.Fields) != 1000 {
		t.Fatalf("a legal 1000-field schema must decode: %v (n=%d)", err, n)
	}
}
