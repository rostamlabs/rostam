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
		{2, 1, 1, 'x', OperateTypeBytes, 0xFF, 0xFF, 0x03}}
	for _, b := range cases {
		if _, err := DecodeRecord(b); err == nil {
			t.Errorf("%x accepted", b)
		}
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
