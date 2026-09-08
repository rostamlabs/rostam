// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
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
