// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"errors"
	"math"
	"testing"
)

// f64 reinterprets a float64 as its IEEE bit pattern at RUNTIME, mirroring
// su64 in operate_cell_test.go (same package): math.Float64bits(x) is not a
// constant expression, so this needs no special-casing, but the helper
// keeps call sites uniform with su64.
func f64(x float64) int64 { return int64(math.Float64bits(x)) } //nolint:gosec // reinterpret bits

func TestApplyScalarSaturates(t *testing.T) {
	cases := []struct {
		typ   uint8
		start uint64
		op    uint8
		a     int64
		want  uint64
	}{
		{OperateTypeU8, 250, OperateOpADD, 10, 255},
		{OperateTypeU8, 3, OperateOpADD, -10, 0},
		{OperateTypeI8, 120, OperateOpADD, 10, 127},
		{OperateTypeI8, su64(-120), OperateOpADD, -10, su64(-128)},
		{OperateTypeU64, math.MaxUint64 - 1, OperateOpADD, 5, math.MaxUint64},
		{OperateTypeI64, math.MaxInt64, OperateOpADD, 1, math.MaxInt64},
		{OperateTypeU16, 60000, OperateOpMUL, 2, 65535},
		{OperateTypeI16, su64(-30000), OperateOpMUL, 2, su64(-32768)},
		{OperateTypeU32, 7, OperateOpMAX, 9, 9}, {OperateTypeU32, 7, OperateOpMAX, -9, 7},
		{OperateTypeI32, 7, OperateOpMIN, -9, su64(-9)},
		{OperateTypeU8, 0b1010, OperateOpAND, 0b0110, 0b0010}, {OperateTypeU8, 0b1010, OperateOpOR, 0b0101, 0b1111}, {OperateTypeU8, 0b1010, OperateOpXOR, 0b1111, 0b0101},
		{OperateTypeU8, 0b1000_0001, OperateOpSHL, 1, 0b0000_0010}, // masked to 8 bits
		{OperateTypeI8, su64(-128), OperateOpSHR, 1, su64(-64)},    // arithmetic
		{OperateTypeU8, 0x80, OperateOpSHR, 1, 0x40},               // logical
		{OperateTypeU8, 1, OperateOpSHL, 64, 0},
		{OperateTypeI8, su64(-5), OperateOpSHR, 64, su64(-1)}, // negative signed, full shift-out -> -1
		{OperateTypeUVarint, 1 << 40, OperateOpADD, 1, 1<<40 + 1},
	}
	for _, c := range cases {
		got, err := ApplyScalar(c.op, Cell{Type: c.typ, U: c.start}, true, Operand{A: c.a})
		if err != nil || got.U != c.want {
			t.Errorf("%+v: got %d (%v) want %d", c, got.U, err, c.want)
		}
	}
}

func TestApplyScalarFloatsAndBytes(t *testing.T) {
	got, _ := ApplyScalar(OperateOpADD, Cell{Type: OperateTypeF32, F: 1.5}, true, Operand{A: f64(0.25)})
	if got.F != 1.75 {
		t.Fatal(got.F)
	}
	got, _ = ApplyScalar(OperateOpMUL, Cell{Type: OperateTypeF64, F: 8}, true, Operand{A: f64(0.5)})
	if got.F != 4 {
		t.Fatal(got.F)
	}
	got, _ = ApplyScalar(OperateOpMAX, Cell{Type: OperateTypeF64, F: 8}, true, Operand{A: f64(math.NaN())})
	if got.F != 8 {
		t.Fatal("NaN replaced a value")
	}
	got, _ = ApplyScalar(OperateOpMAX, Cell{Type: OperateTypeBytes, B: []byte("abc")}, true, Operand{Bytes: []byte("abd")})
	if string(got.B) != "abd" {
		t.Fatal(string(got.B))
	}
	got, _ = ApplyScalar(OperateOpSET, Cell{Type: OperateTypeFixed, N: 2, B: []byte("DE")}, true, Operand{Bytes: []byte("FR")})
	if string(got.B) != "FR" {
		t.Fatal(string(got.B))
	}
	if _, err := ApplyScalar(OperateOpSET, Cell{Type: OperateTypeFixed, N: 2}, true, Operand{Bytes: []byte("FRA")}); err == nil {
		t.Fatal("wrong FIXED length accepted")
	}
}

func TestApplyScalarRejects(t *testing.T) {
	bad := []struct{ typ, op uint8 }{
		{OperateTypeF32, OperateOpSHL}, {OperateTypeF64, OperateOpAND}, {OperateTypeUVarint, OperateOpSHL},
		{OperateTypeBytes, OperateOpADD},
	}
	for _, b := range bad {
		if _, err := ApplyScalar(b.op, ZeroCell(b.typ, 0), true, Operand{A: 1}); err == nil {
			t.Errorf("%+v accepted", b)
		}
	}
	if _, err := ApplyScalar(OperateOpSHL, Cell{Type: OperateTypeU8}, true, Operand{A: -1}); err == nil {
		t.Fatal("negative shift accepted")
	}
	// An unknown opcode is ErrOperateOpcode in EVERY domain, not just int —
	// it must never be reported as ErrOperateType just because the target
	// field happens to be a float or bytes cell.
	for _, typ := range []uint8{OperateTypeU8, OperateTypeF64, OperateTypeBytes} {
		if _, err := ApplyScalar(99, ZeroCell(typ, 0), true, Operand{A: 1}); !errors.Is(err, ErrOperateOpcode) {
			t.Errorf("type %d: unknown opcode: got %v want ErrOperateOpcode", typ, err)
		}
	}
}

func TestApplyScalarStamp(t *testing.T) {
	got, _ := ApplyScalar(OperateOpSTAMP, ZeroCell(OperateTypeU32, 0), false, Operand{Aux: OperateStampS, StampMs: 1_700_000_000_123})
	if got.U != 1_700_000_000 {
		t.Fatal(got.U)
	}
	got, _ = ApplyScalar(OperateOpSTAMP, ZeroCell(OperateTypeU16, 0), false, Operand{Aux: OperateStampMs, StampMs: 1_700_000_000_123})
	if got.U != 65535 {
		t.Fatal("stamp did not saturate")
	}
	if _, err := ApplyScalar(OperateOpSTAMP, ZeroCell(OperateTypeU32, 0), false, Operand{Aux: 99, StampMs: 1}); !errors.Is(err, ErrOperateType) {
		t.Fatalf("bad stamp aux: got %v want ErrOperateType", err)
	}
}

func TestCompareCell(t *testing.T) {
	c := Cell{Type: OperateTypeI32, U: su64(-3)}
	for cmp, want := range map[uint8]bool{OperateCmpLT: true, OperateCmpGE: false, OperateCmpEQ: false, OperateCmpNE: true, OperateCmpExists: true, OperateCmpAbsent: false} {
		if got, _ := CompareCell(cmp, c, true, Operand{A: 0}); got != want {
			t.Errorf("cmp %d: %v", cmp, got)
		}
	}
	// absent numeric compares as zero
	if got, _ := CompareCell(OperateCmpLT, ZeroCell(OperateTypeU32, 0), false, Operand{A: 100}); !got {
		t.Fatal("absent<100 should hold")
	}
	if got, _ := CompareCell(OperateCmpExists, ZeroCell(OperateTypeU32, 0), false, Operand{}); got {
		t.Fatal("absent exists")
	}
	if got, _ := CompareCell(OperateCmpGT, Cell{Type: OperateTypeBytes, B: []byte("b")}, true, Operand{Bytes: []byte("a")}); !got {
		t.Fatal("bytes cmp")
	}
	if got, _ := CompareCount(OperateCmpGE, 100, true, Operand{A: 100}); !got {
		t.Fatal("count cmp")
	}
	if _, err := CompareCell(99, c, true, Operand{}); err == nil {
		t.Fatal("bad cmp accepted")
	}
}
