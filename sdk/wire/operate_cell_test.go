// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestCellWidth(t *testing.T) {
	cases := map[[2]uint8]int{
		{OperateTypeU8, 0}: 1, {OperateTypeI16, 0}: 2, {OperateTypeU32, 0}: 4, {OperateTypeI64, 0}: 8,
		{OperateTypeF32, 0}: 4, {OperateTypeF64, 0}: 8, {OperateTypeFixed, 16}: 16, {OperateTypeUnset, 0}: 0,
		{OperateTypeBytes, 0}: -1, {OperateTypeUVarint, 0}: -1, {OperateTypeIVarint, 0}: -1, {OperateTypeTable, 0}: -1,
	}
	for k, want := range cases {
		if got := CellWidth(k[0], k[1]); got != want {
			t.Errorf("CellWidth(%d,%d)=%d want %d", k[0], k[1], got, want)
		}
	}
}

// su64 reinterprets a signed int64 as its uint64 bit pattern at RUNTIME. It
// exists only so the literals below (-5, math.MinInt32, -300) don't become
// constant expressions: uint64(int64(-5)) is a compile error ("constant -5
// overflows uint64") because Go's constant-conversion rule checks
// representability at each step, even though the identical conversion of a
// non-constant int64 wraps as intended (the convention Cell.U itself uses).
func su64(i int64) uint64 { return uint64(i) }

func TestCellDataRoundtrip(t *testing.T) {
	cells := []Cell{
		{Type: OperateTypeU8, U: 255}, {Type: OperateTypeI8, U: su64(-5)}, {Type: OperateTypeI32, U: su64(math.MinInt32)},
		{Type: OperateTypeU64, U: math.MaxUint64}, {Type: OperateTypeF32, F: 1.5}, {Type: OperateTypeF64, F: -2.25},
		{Type: OperateTypeUVarint, U: 300}, {Type: OperateTypeIVarint, U: su64(-300)},
		{Type: OperateTypeBytes, B: []byte("hello")}, {Type: OperateTypeFixed, N: 2, B: []byte("DE")},
	}
	for _, c := range cells {
		enc := AppendCellData(nil, c)
		got, n, err := DecodeCellData(c.Type, c.N, enc)
		if err != nil || n != len(enc) {
			t.Fatalf("%+v: err=%v n=%d len=%d", c, err, n, len(enc))
		}
		if got.U != c.U || got.F != c.F || !bytes.Equal(got.B, c.B) {
			t.Fatalf("roundtrip %+v → %+v", c, got)
		}
		tagged := AppendTaggedCell(nil, c)
		got2, n2, err := DecodeTaggedCell(tagged)
		if err != nil || n2 != len(tagged) || got2.Type != c.Type || got2.N != c.N {
			t.Fatalf("tagged %+v: %+v %v", c, got2, err)
		}
	}
}

func TestCellDataTruncationAndBadTag(t *testing.T) {
	if _, _, err := DecodeCellData(OperateTypeU32, 0, []byte{1, 2}); err == nil {
		t.Fatal("truncated U32 accepted")
	}
	if _, _, err := DecodeCellData(OperateTypeFixed, 4, []byte{1, 2}); err == nil {
		t.Fatal("truncated FIXED accepted")
	}
	if _, _, err := DecodeCellData(OperateTypeBytes, 0, []byte{0x05, 1}); err == nil {
		t.Fatal("truncated BYTES accepted")
	}
	if _, _, err := DecodeCellData(OperateTypeUVarint, 0, []byte{0x80}); err == nil {
		t.Fatal("truncated varint accepted")
	}
	if _, _, err := DecodeTaggedCell([]byte{OperateTypeCount}); err == nil {
		t.Fatal("unknown tag accepted")
	}
	if _, _, err := DecodeTaggedCell([]byte{OperateTypeTable}); err == nil {
		t.Fatal("table tag accepted as a cell")
	}
}

func TestCanonFloat(t *testing.T) {
	if CanonFloat(OperateTypeF32, 0.1) != float64(float32(0.1)) {
		t.Fatal("F32 not rounded")
	}
	nan := math.Float64frombits(0x7FF8DEADBEEF0001)
	if math.Float64bits(CanonFloat(OperateTypeF64, nan)) != 0x7FF8000000000000 {
		t.Fatal("NaN not canonical")
	}
	if math.Float32bits(float32(CanonFloat(OperateTypeF32, nan))) != 0x7FC00000 {
		t.Fatal("F32 NaN not canonical")
	}
}

// TestAppendCellDataCanonicalizesNaN pins design doc §2.5 on the ENCODE
// side: whatever NaN bit pattern a caller hands in, the bytes that go to
// storage are the canonical quiet NaN (and an F32 is narrowed through
// float32 on the way). Without this, a payload NaN reaching AppendCellData
// from a hand-built Cell would be stored verbatim and then rejected by
// DecodeRecord's canonical-bytes check as a malformed record.
func TestAppendCellDataCanonicalizesNaN(t *testing.T) {
	payload := math.Float64frombits(0x7FF8000000000123) // quiet NaN, payload 0x123
	signaling := math.Float64frombits(0x7FF0000000000001)

	for _, f := range []float64{payload, signaling, math.NaN()} {
		got := AppendCellData(nil, Cell{Type: OperateTypeF64, F: f})
		if len(got) != 8 || binary.LittleEndian.Uint64(got) != 0x7FF8000000000000 {
			t.Fatalf("F64 NaN encoded as %x, want the canonical quiet NaN", got)
		}
		got32 := AppendCellData(nil, Cell{Type: OperateTypeF32, F: f})
		if len(got32) != 4 || binary.LittleEndian.Uint32(got32) != 0x7FC00000 {
			t.Fatalf("F32 NaN encoded as %x, want the canonical quiet NaN", got32)
		}
	}

	// The canonical pattern round-trips: decode it and it re-encodes to the
	// same bytes, which is what keeps DecodeRecord's identity intact.
	canon := AppendCellData(nil, Cell{Type: OperateTypeF64, F: math.NaN()})
	c, n, err := DecodeCellData(OperateTypeF64, 0, canon)
	if err != nil || n != 8 || !math.IsNaN(c.F) {
		t.Fatalf("decode canonical NaN: %+v n=%d err=%v", c, n, err)
	}
	if !bytes.Equal(AppendCellData(nil, c), canon) {
		t.Fatal("canonical NaN did not re-encode to itself")
	}

	// Finite floats are untouched, negative zero included.
	for _, f := range []float64{0, math.Copysign(0, -1), 1.5, -3.25, math.Inf(1), math.Inf(-1)} {
		got := AppendCellData(nil, Cell{Type: OperateTypeF64, F: f})
		if binary.LittleEndian.Uint64(got) != math.Float64bits(f) {
			t.Fatalf("finite F64 %v encoded as %x", f, got)
		}
	}
}
