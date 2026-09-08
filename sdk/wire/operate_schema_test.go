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

// declaredTablesBlob builds a schema blob header that declares nTables TABLE
// fields, each declaring nCols columns, with enough trailing filler that
// every CountFitsIn bound is satisfied. It is not a legal schema — it is the
// shape a hostile STORED record can carry, since the 4 KiB blob cap is only
// checked by Validate, after the whole thing has been decoded.
func declaredTablesBlob(nTables, nCols int) []byte {
	b := []byte{1, 0, 0} // version 1, no flags
	b = binary.AppendUvarint(b, uint64(nTables))
	for i := 0; i < nTables; i++ {
		b = append(b, OperateTypeTable, OperateTypeU8) // field type, key type
		b = binary.AppendUvarint(b, uint64(nCols))
		b = append(b, make([]byte, nCols)...) // nCols U8 column type bytes
		b = append(b, 0, 0, 0)                // cap, policy, byCol
	}
	return append(b, 0) // nameBytes
}

// TestDecodeSchemaAggregateBudget covers the schema-blob budget IN
// AGGREGATE. A per-declaration bound is not a bound at all here: every one of
// many table fields can declare a column count that passes its own check
// while the sum sizes far more ColumnDef than a 4 KiB blob could ever hold.
// DecodeSchema is handed the whole stored record, up to maxRecordBytes, and
// the blob cap is only checked by Validate once decoding is done, so the
// budget has to be spent as it goes.
func TestDecodeSchemaAggregateBudget(t *testing.T) {
	// Three tables of 2000 columns each: 2000 passes any per-table bound
	// (it is under both OperateMaxCols and OperateMaxSchemaBytes), while
	// 3 + 6000 declared bytes is well past the 4096-byte blob.
	blob := declaredTablesBlob(3, 2000)
	if _, _, err := DecodeSchema(blob); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("3 tables x 2000 columns: err = %v, want ErrOperateSchema", err)
	}

	// The same shape at scale: 200 tables of 4000 columns is 800,000 declared
	// columns, ~19 MB of ColumnDef if every count were honoured. 4000 is
	// under both OperateMaxCols and OperateMaxSchemaBytes, so no
	// per-declaration bound stops it — only the running budget does, and it
	// does so at the FIRST table (200 fields leave 3896 bytes, less than the
	// 4000 that table asks for), before the make. Three allocations reach
	// that point: the Schema, the field slice, and the first TableDef.
	big := declaredTablesBlob(200, 4000)
	if _, _, err := DecodeSchema(big); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("200 tables x 4000 columns: err = %v, want ErrOperateSchema", err)
	}
	if n := testing.AllocsPerRun(10, func() { _, _, _ = DecodeSchema(big) }); n > 4 {
		t.Fatalf("declared 800000 columns allocated %v times; the budget must be spent before each make", n)
	}

	// The aggregate bound must not cost a legal schema anything. This one
	// spends most of the blob on columns spread over several tables, which is
	// exactly the shape the budget charges.
	s := &Schema{Version: 1}
	for i := 0; i < 4; i++ {
		td := &TableDef{KeyType: OperateTypeU64}
		for c := 0; c < 300; c++ {
			td.Cols = append(td.Cols, ColumnDef{Type: OperateTypeU8})
		}
		s.Fields = append(s.Fields, FieldDef{Type: OperateTypeTable, Table: td})
	}
	enc := s.Encode()
	if len(enc) > OperateMaxSchemaBytes {
		t.Fatalf("fixture blob is %d bytes, over the %d-byte cap", len(enc), OperateMaxSchemaBytes)
	}
	got, n, err := DecodeSchema(enc)
	if err != nil || n != len(enc) || len(got.Fields) != 4 || len(got.Fields[3].Table.Cols) != 300 {
		t.Fatalf("a legal 4-table, 1200-column schema must decode: %v (n=%d)", err, n)
	}
}

// TestExtendsComparesWidthOnlyForFixed covers Extends's width rule. N is the
// FIXED(n) width and is ignored — not even encoded — for every other type,
// so comparing it unconditionally refused an evolution over a byte nothing
// reads.
func TestExtendsComparesWidthOnlyForFixed(t *testing.T) {
	base := &Schema{Version: 1, Fields: []FieldDef{
		{Name: "n", Type: OperateTypeU32, N: 0},
		{Name: "t", Type: OperateTypeTable, Table: &TableDef{
			KeyType: OperateTypeU64, KeyN: 0,
			Cols: []ColumnDef{{Name: "c", Type: OperateTypeU16, N: 0}},
		}},
	}}
	// Identical but for three N values none of these types reads.
	newer := &Schema{Version: 2, Fields: []FieldDef{
		{Name: "n", Type: OperateTypeU32, N: 7},
		{Name: "t", Type: OperateTypeTable, Table: &TableDef{
			KeyType: OperateTypeU64, KeyN: 9,
			Cols: []ColumnDef{{Name: "c", Type: OperateTypeU16, N: 5}},
		}},
	}}
	if err := base.Extends(newer); err != nil {
		t.Fatalf("an ignored N blocked an otherwise identical evolution: %v", err)
	}
	// The blobs are in fact identical, which is why the widths cannot matter.
	if !bytes.Equal(base.Encode()[2:], newer.Encode()[2:]) { // [2:] skips the version
		t.Fatal("fixture schemas do not encode identically; the test would prove nothing")
	}

	// Where N does mean something, a change is still refused.
	fixedBase := &Schema{Version: 1, Fields: []FieldDef{{Name: "f", Type: OperateTypeFixed, N: 4}}}
	fixedNewer := &Schema{Version: 2, Fields: []FieldDef{{Name: "f", Type: OperateTypeFixed, N: 5}}}
	if err := fixedBase.Extends(fixedNewer); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("a changed FIXED width: err = %v, want ErrOperateSchema", err)
	}
	keyBase := &Schema{Version: 1, Fields: []FieldDef{{Name: "t", Type: OperateTypeTable,
		Table: &TableDef{KeyType: OperateTypeFixed, KeyN: 4}}}}
	keyNewer := &Schema{Version: 2, Fields: []FieldDef{{Name: "t", Type: OperateTypeTable,
		Table: &TableDef{KeyType: OperateTypeFixed, KeyN: 5}}}}
	if err := keyBase.Extends(keyNewer); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("a changed FIXED key width: err = %v, want ErrOperateSchema", err)
	}
	colBase := &Schema{Version: 1, Fields: []FieldDef{{Name: "t", Type: OperateTypeTable,
		Table: &TableDef{KeyType: OperateTypeU8, Cols: []ColumnDef{{Type: OperateTypeFixed, N: 4}}}}}}
	colNewer := &Schema{Version: 2, Fields: []FieldDef{{Name: "t", Type: OperateTypeTable,
		Table: &TableDef{KeyType: OperateTypeU8, Cols: []ColumnDef{{Type: OperateTypeFixed, N: 5}}}}}}
	if err := colBase.Extends(colNewer); !errors.Is(err, ErrOperateSchema) {
		t.Fatalf("a changed FIXED column width: err = %v, want ErrOperateSchema", err)
	}
}
