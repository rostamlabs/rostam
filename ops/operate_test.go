// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// callOperate applies an operate op-list under a leader stamp of nowMs and returns
// the decoded return-field values. It stamps the tx so touchMs and the eviction
// order are the deterministic leader-clock path (not the wall clock).
func callOperate(t *testing.T, tx *TxContext, nowMs int64, key []byte, maxEntries uint16, ops []wire.OperateOp, ret []wire.OperateRet) []int64 {
	t.Helper()
	tx.SetApplyStamp(uint64(nowMs), true)
	defer tx.SetApplyStamp(0, false)
	res, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, maxEntries, ops, ret))
	if err != nil {
		t.Fatalf("handleOperate: %v", err)
	}
	vals, err := wire.DecodeOperateResult(res)
	if err != nil {
		t.Fatalf("decode result: %v", err)
	}
	return vals
}

func gOp(field uint16, typ, op uint8, arg, arg2 int64) wire.OperateOp {
	return wire.OperateOp{Target: wire.OperateTargetGlobal, FieldIdx: field, Opcode: op, Type: typ, Arg: arg, Arg2: arg2}
}
func eOp(key uint64, field uint16, typ, op uint8, arg, arg2 int64) wire.OperateOp {
	return wire.OperateOp{Target: wire.OperateTargetEntry, EntryKey: key, FieldIdx: field, Opcode: op, Type: typ, Arg: arg, Arg2: arg2}
}
func gRet(field uint16) wire.OperateRet {
	return wire.OperateRet{Target: wire.OperateTargetGlobal, FieldIdx: field}
}
func eRet(key uint64, field uint16) wire.OperateRet {
	return wire.OperateRet{Target: wire.OperateTargetEntry, EntryKey: key, FieldIdx: field}
}

func f32arg(x float32) int64 { return int64(uint64(math.Float32bits(x))) }
func f64arg(x float64) int64 { return int64(math.Float64bits(x)) }

func TestOperateOpcodes(t *testing.T) {
	tests := []struct {
		name string
		ops  []wire.OperateOp
		ret  []wire.OperateRet
		want []int64
	}{
		{
			name: "INCR i64 accumulates",
			ops:  []wire.OperateOp{gOp(0, wire.OperateTypeI64, wire.OperateOpINCR, 5, 0), gOp(0, wire.OperateTypeI64, wire.OperateOpINCR, -2, 0)},
			ret:  []wire.OperateRet{gRet(0)},
			want: []int64{3},
		},
		{
			name: "SETMAX i32 keeps the larger",
			ops:  []wire.OperateOp{gOp(2, wire.OperateTypeI32, wire.OperateOpSETMAX, 10, 0), gOp(2, wire.OperateTypeI32, wire.OperateOpSETMAX, 4, 0), gOp(2, wire.OperateTypeI32, wire.OperateOpSETMAX, 12, 0)},
			ret:  []wire.OperateRet{gRet(2)},
			want: []int64{12},
		},
		{
			name: "SHIFTOR u32 builds a bitfield",
			ops:  []wire.OperateOp{gOp(3, wire.OperateTypeU32, wire.OperateOpSHIFTOR, 4, 0x1), gOp(3, wire.OperateTypeU32, wire.OperateOpSHIFTOR, 4, 0x2)},
			ret:  []wire.OperateRet{gRet(3)},
			want: []int64{0x12},
		},
		{
			name: "HALVE_GRP halves when guard >= threshold",
			ops: []wire.OperateOp{
				gOp(0, wire.OperateTypeU16, wire.OperateOpINCR, 300, 0),
				gOp(1, wire.OperateTypeU16, wire.OperateOpINCR, 80, 0),
				gOp(0, wire.OperateTypeU16, wire.OperateOpHALVEGRP, 255, 2),
			},
			ret:  []wire.OperateRet{gRet(0), gRet(1)},
			want: []int64{150, 40},
		},
		{
			name: "HALVE_GRP no-op when guard < threshold",
			ops: []wire.OperateOp{
				gOp(0, wire.OperateTypeU16, wire.OperateOpINCR, 100, 0),
				gOp(1, wire.OperateTypeU16, wire.OperateOpINCR, 80, 0),
				gOp(0, wire.OperateTypeU16, wire.OperateOpHALVEGRP, 255, 2),
			},
			ret:  []wire.OperateRet{gRet(0), gRet(1)},
			want: []int64{100, 80},
		},
		{
			name: "entry fields are independent of globals",
			ops:  []wire.OperateOp{gOp(0, wire.OperateTypeU8, wire.OperateOpINCR, 1, 0), eOp(7, 0, wire.OperateTypeU8, wire.OperateOpINCR, 42, 0)},
			ret:  []wire.OperateRet{gRet(0), eRet(7, 0), eRet(7, 5)},
			want: []int64{1, 42, 0}, // absent entry field reads 0
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, tx := newTestSetup(t)
			got := callOperate(t, tx, 1000, []byte(tc.name), 0, tc.ops, tc.ret)
			if len(got) != len(tc.want) {
				t.Fatalf("ret len = %d, want %d", len(got), len(tc.want))
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("ret[%d] = %d, want %d (all=%v)", i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

func TestOperateUnsignedSaturation(t *testing.T) {
	cases := []struct {
		name string
		typ  uint8
		max  uint64
	}{
		{"u8", wire.OperateTypeU8, math.MaxUint8},
		{"u16", wire.OperateTypeU16, math.MaxUint16},
		{"u32", wire.OperateTypeU32, math.MaxUint32},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, tx := newTestSetup(t)
			// Add far past the type max in two steps; must cap, not wrap.
			got := callOperate(t, tx, 1, []byte(c.name), 0, []wire.OperateOp{
				gOp(0, c.typ, wire.OperateOpINCR, int64(c.max), 0),
				gOp(0, c.typ, wire.OperateOpINCR, 1000, 0),
			}, []wire.OperateRet{gRet(0)})
			if uint64(got[0]) != c.max {
				t.Fatalf("%s saturated to %d, want %d", c.name, uint64(got[0]), c.max)
			}
			// Decrement below zero must floor at 0, not underflow.
			got = callOperate(t, tx, 2, []byte(c.name), 0, []wire.OperateOp{
				gOp(0, c.typ, wire.OperateOpINCR, math.MinInt64, 0),
			}, []wire.OperateRet{gRet(0)})
			if got[0] != 0 {
				t.Fatalf("%s underflow to %d, want 0", c.name, got[0])
			}
		})
	}
}

func TestOperateU64Saturation(t *testing.T) {
	_, tx := newTestSetup(t)
	got := callOperate(t, tx, 1, []byte("u64"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeU64, wire.OperateOpINCR, math.MaxInt64, 0),
		gOp(0, wire.OperateTypeU64, wire.OperateOpINCR, math.MaxInt64, 0),
		gOp(0, wire.OperateTypeU64, wire.OperateOpINCR, 100, 0),
	}, []wire.OperateRet{gRet(0)})
	if uint64(got[0]) != math.MaxUint64 {
		t.Fatalf("u64 saturated to %d, want MaxUint64", uint64(got[0]))
	}
}

func TestOperateSignedSaturation(t *testing.T) {
	_, tx := newTestSetup(t)
	// i8 caps at 127 going up and -128 going down.
	got := callOperate(t, tx, 1, []byte("i8hi"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeI8, wire.OperateOpINCR, 200, 0),
	}, []wire.OperateRet{gRet(0)})
	if got[0] != 127 {
		t.Fatalf("i8 up = %d, want 127", got[0])
	}
	got = callOperate(t, tx, 2, []byte("i8lo"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeI8, wire.OperateOpINCR, -200, 0),
	}, []wire.OperateRet{gRet(0)})
	if got[0] != -128 {
		t.Fatalf("i8 down = %d, want -128", got[0])
	}
	// i64 caps at the int64 boundary instead of wrapping.
	got = callOperate(t, tx, 3, []byte("i64"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeI64, wire.OperateOpINCR, math.MaxInt64, 0),
		gOp(0, wire.OperateTypeI64, wire.OperateOpINCR, 10, 0),
	}, []wire.OperateRet{gRet(0)})
	if got[0] != math.MaxInt64 {
		t.Fatalf("i64 up = %d, want MaxInt64", got[0])
	}
}

func TestOperateFloatArithmetic(t *testing.T) {
	_, tx := newTestSetup(t)
	// F64 INCRF adds IEEE doubles.
	got := callOperate(t, tx, 1, []byte("f64"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeF64, wire.OperateOpINCRF, f64arg(1.5), 0),
		gOp(0, wire.OperateTypeF64, wire.OperateOpINCRF, f64arg(2.5), 0),
	}, []wire.OperateRet{gRet(0)})
	if v := math.Float64frombits(uint64(got[0])); v != 4.0 {
		t.Fatalf("f64 = %v, want 4.0", v)
	}
	// F32 INCRF + SETMAX, read back through the low-32 bit pattern.
	got = callOperate(t, tx, 2, []byte("f32"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeF32, wire.OperateOpINCRF, f32arg(1.5), 0),
		gOp(0, wire.OperateTypeF32, wire.OperateOpINCRF, f32arg(2.25), 0),
		gOp(1, wire.OperateTypeF32, wire.OperateOpSETMAX, f32arg(9.5), 0),
		gOp(1, wire.OperateTypeF32, wire.OperateOpSETMAX, f32arg(3.0), 0),
	}, []wire.OperateRet{gRet(0), gRet(1)})
	if v := math.Float32frombits(uint32(got[0])); v != 3.75 {
		t.Fatalf("f32 sum = %v, want 3.75", v)
	}
	if v := math.Float32frombits(uint32(got[1])); v != 9.5 {
		t.Fatalf("f32 max = %v, want 9.5", v)
	}
}

func TestOperateShiftOrMasksToWidth(t *testing.T) {
	_, tx := newTestSetup(t)
	// Three shift-4/OR-0xF ops: a u8 field masks to 0xFF (old bits fall off), while
	// a u32 field keeps growing (0xFFF), proving the mask is per-type-width.
	got := callOperate(t, tx, 1, []byte("mask"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeU8, wire.OperateOpSHIFTOR, 4, 0xF),
		gOp(0, wire.OperateTypeU8, wire.OperateOpSHIFTOR, 4, 0xF),
		gOp(0, wire.OperateTypeU8, wire.OperateOpSHIFTOR, 4, 0xF),
		gOp(1, wire.OperateTypeU32, wire.OperateOpSHIFTOR, 4, 0xF),
		gOp(1, wire.OperateTypeU32, wire.OperateOpSHIFTOR, 4, 0xF),
		gOp(1, wire.OperateTypeU32, wire.OperateOpSHIFTOR, 4, 0xF),
	}, []wire.OperateRet{gRet(0), gRet(1)})
	if got[0] != 0xFF {
		t.Fatalf("u8 SHIFTOR = %#x, want 0xFF (masked to 8 bits)", got[0])
	}
	if got[1] != 0xFFF {
		t.Fatalf("u32 SHIFTOR = %#x, want 0xFFF (unmasked at 32 bits)", got[1])
	}
}

func TestOperateShiftOrRejectedOnVarintAndFloat(t *testing.T) {
	for _, typ := range []uint8{wire.OperateTypeUVARINT, wire.OperateTypeIVARINT, wire.OperateTypeF32, wire.OperateTypeF64} {
		_, tx := newTestSetup(t)
		tx.SetApplyStamp(1, true)
		_, err := handleOperate(tx, wire.EncodeOperateArgs([]byte("s"), 0, 0,
			[]wire.OperateOp{gOp(0, typ, wire.OperateOpSHIFTOR, 1, 0)}, nil))
		tx.SetApplyStamp(0, false)
		if err == nil {
			t.Fatalf("SHIFTOR on type %d accepted, want error", typ)
		}
	}
}

func TestOperateVarintRoundTripAndGrow(t *testing.T) {
	_, tx := newTestSetup(t)
	// UVARINT crosses the 1→2 byte boundary (127 → 300) and keeps the exact value.
	got := callOperate(t, tx, 1, []byte("uv"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeUVARINT, wire.OperateOpINCR, 100, 0),
		gOp(0, wire.OperateTypeUVARINT, wire.OperateOpINCR, 200, 0),
	}, []wire.OperateRet{gRet(0)})
	if uint64(got[0]) != 300 {
		t.Fatalf("uvarint = %d, want 300", uint64(got[0]))
	}
	// A large UVARINT (multi-byte) round-trips exactly.
	got = callOperate(t, tx, 2, []byte("uvbig"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeUVARINT, wire.OperateOpINCR, 1<<40, 0),
	}, []wire.OperateRet{gRet(0)})
	if uint64(got[0]) != 1<<40 {
		t.Fatalf("uvarint big = %d, want %d", uint64(got[0]), uint64(1<<40))
	}
	// IVARINT holds negatives (zigzag) and round-trips.
	got = callOperate(t, tx, 3, []byte("iv"), 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeIVARINT, wire.OperateOpINCR, -5, 0),
		gOp(0, wire.OperateTypeIVARINT, wire.OperateOpINCR, -1000, 0),
	}, []wire.OperateRet{gRet(0)})
	if got[0] != -1005 {
		t.Fatalf("ivarint = %d, want -1005", got[0])
	}
}

func TestOperateVarintEncodingGrows(t *testing.T) {
	// Directly assert the stored encoding grows across a varint byte boundary.
	small := operateRec{entries: map[uint64]*operateEntry{}, globals: []operateField{{typ: wire.OperateTypeUVARINT, u: 127}}}
	big := operateRec{entries: map[uint64]*operateEntry{}, globals: []operateField{{typ: wire.OperateTypeUVARINT, u: 128}}}
	if len(encodeRec(big)) <= len(encodeRec(small)) {
		t.Fatalf("uvarint 128 should encode longer than 127: %d vs %d", len(encodeRec(big)), len(encodeRec(small)))
	}
}

func TestOperateStoredTypeWins(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("stw")
	// Create field 0 as U8.
	callOperate(t, tx, 1, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeU8, wire.OperateOpINCR, 10, 0)}, nil)
	// A later op DECLARES I64 but the stored U8 wins → still saturates at 255.
	got := callOperate(t, tx, 2, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeI64, wire.OperateOpINCR, 300, 0)}, []wire.OperateRet{gRet(0)})
	if got[0] != 255 {
		t.Fatalf("stored-type-wins: got %d, want 255 (U8 saturation held)", got[0])
	}
}

func TestOperateTypedRecordShrinks(t *testing.T) {
	// A record of narrow typed fields encodes materially smaller than the same
	// logical record stored all-i64.
	narrow := operateRec{entries: map[uint64]*operateEntry{}, globals: []operateField{
		{typ: wire.OperateTypeU8, u: 200},
		{typ: wire.OperateTypeU16, u: 5000},
		{typ: wire.OperateTypeF32, f: 1.5},
	}}
	wide := operateRec{entries: map[uint64]*operateEntry{}, globals: []operateField{
		{typ: wire.OperateTypeI64, u: 200},
		{typ: wire.OperateTypeI64, u: 5000},
		{typ: wire.OperateTypeI64, u: 1},
	}}
	nn, wn := len(encodeRec(narrow)), len(encodeRec(wide))
	if nn >= wn {
		t.Fatalf("typed record (%d bytes) not smaller than all-i64 (%d bytes)", nn, wn)
	}
	t.Logf("typed=%d bytes, all-i64=%d bytes (%.0f%% smaller)", nn, wn, 100*(1-float64(nn)/float64(wn)))
}

func TestOperatePersistsAcrossCalls(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("counter")
	callOperate(t, tx, 100, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeU32, wire.OperateOpINCR, 1, 0)}, nil)
	callOperate(t, tx, 200, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeU32, wire.OperateOpINCR, 1, 0)}, nil)
	got := callOperate(t, tx, 300, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeU32, wire.OperateOpINCR, 1, 0)}, []wire.OperateRet{gRet(0)})
	if got[0] != 3 {
		t.Fatalf("accumulated = %d, want 3", got[0])
	}
}

func TestOperateCapEvictsDeterministicLRU(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("sess")
	callOperate(t, tx, 100, key, 2, []wire.OperateOp{eOp(1, 0, wire.OperateTypeU8, wire.OperateOpINCR, 11, 0)}, nil)
	callOperate(t, tx, 200, key, 2, []wire.OperateOp{eOp(2, 0, wire.OperateTypeU8, wire.OperateOpINCR, 22, 0)}, nil)
	got := callOperate(t, tx, 300, key, 2,
		[]wire.OperateOp{eOp(3, 0, wire.OperateTypeU8, wire.OperateOpINCR, 33, 0)},
		[]wire.OperateRet{eRet(1, 0), eRet(2, 0), eRet(3, 0)})
	if got[0] != 0 {
		t.Fatalf("entry 1 should be evicted (0), got %d", got[0])
	}
	if got[1] != 22 || got[2] != 33 {
		t.Fatalf("entry 2/3 = %d/%d, want 22/33", got[1], got[2])
	}
}

func TestOperateCapEvictionTieBreaksBySmallestKey(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("tie")
	callOperate(t, tx, 100, key, 2, []wire.OperateOp{
		eOp(9, 0, wire.OperateTypeU8, wire.OperateOpINCR, 99, 0),
		eOp(5, 0, wire.OperateTypeU8, wire.OperateOpINCR, 55, 0),
	}, nil)
	got := callOperate(t, tx, 100, key, 2,
		[]wire.OperateOp{eOp(7, 0, wire.OperateTypeU8, wire.OperateOpINCR, 77, 0)},
		[]wire.OperateRet{eRet(5, 0), eRet(9, 0), eRet(7, 0)})
	if got[0] != 0 {
		t.Fatalf("smallest-key entry 5 should be evicted (0), got %d", got[0])
	}
	if got[1] != 99 || got[2] != 77 {
		t.Fatalf("entry 9/7 = %d/%d, want 99/77", got[1], got[2])
	}
}

func TestOperateBadOpcodeAbortsAtomically(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("atomic")
	callOperate(t, tx, 100, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeU16, wire.OperateOpINCR, 5, 0)}, nil)
	tx.SetApplyStamp(200, true)
	_, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeU16, wire.OperateOpINCR, 100, 0),
		{Target: wire.OperateTargetGlobal, FieldIdx: 0, Opcode: 250 /* unknown */, Type: wire.OperateTypeU16},
	}, nil))
	tx.SetApplyStamp(0, false)
	if err == nil {
		t.Fatal("bad opcode accepted, want error")
	}
	got := callOperate(t, tx, 300, key, 0, nil, []wire.OperateRet{gRet(0)})
	if got[0] != 5 {
		t.Fatalf("record mutated by aborted op-list: got %d, want 5", got[0])
	}
}

func TestOperateNegativeShiftRejected(t *testing.T) {
	_, tx := newTestSetup(t)
	tx.SetApplyStamp(1, true)
	defer tx.SetApplyStamp(0, false)
	_, err := handleOperate(tx, wire.EncodeOperateArgs([]byte("s"), 0, 0,
		[]wire.OperateOp{gOp(0, wire.OperateTypeU32, wire.OperateOpSHIFTOR, -1, 0)}, nil))
	if err == nil {
		t.Fatal("negative shift accepted, want error")
	}
}

func TestOperateFieldIndexOverflowRejected(t *testing.T) {
	_, tx := newTestSetup(t)
	tx.SetApplyStamp(1, true)
	defer tx.SetApplyStamp(0, false)
	_, err := handleOperate(tx, wire.EncodeOperateArgs([]byte("o"), 0, 0,
		[]wire.OperateOp{gOp(65535, wire.OperateTypeU8, wire.OperateOpINCR, 1, 0)}, nil))
	if err == nil {
		t.Fatal("overflowing field index accepted, want error")
	}
}

func TestOperateHalveGroupHostileLengthRejected(t *testing.T) {
	// A huge/overflowing HALVE_GRP group length must be rejected (no idx+len overflow,
	// no index-out-of-range panic on the replicated apply path), leaving the record
	// unchanged.
	for _, arg2 := range []int64{math.MaxInt64, math.MinInt64, -1, 0, int64(maxOperateFields) + 1} {
		_, tx := newTestSetup(t)
		key := []byte("hg")
		callOperate(t, tx, 1, key, 0, []wire.OperateOp{gOp(0, wire.OperateTypeU8, wire.OperateOpINCR, 5, 0)}, nil)
		tx.SetApplyStamp(2, true)
		_, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, 0,
			[]wire.OperateOp{gOp(0, wire.OperateTypeU8, wire.OperateOpHALVEGRP, 1, arg2)}, nil))
		tx.SetApplyStamp(0, false)
		if err == nil {
			t.Fatalf("HALVE_GRP arg2=%d accepted, want error", arg2)
		}
		// Record unchanged: field 0 still 5.
		got := callOperate(t, tx, 3, key, 0, nil, []wire.OperateRet{gRet(0)})
		if got[0] != 5 {
			t.Fatalf("record mutated by rejected HALVE_GRP arg2=%d: got %d, want 5", arg2, got[0])
		}
	}
}

// TestOperateHoleTypeOrderIndependent: creating a high field index first must NOT
// force the lower (hole) indices to a narrow type — a later write to a hole adopts
// ITS declared type, independent of order (no silent truncation).
func TestOperateHoleTypeOrderIndependent(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("holes")
	// Create field 5 (U8) first, which materializes 0..4 as holes.
	callOperate(t, tx, 1, key, 0, []wire.OperateOp{gOp(5, wire.OperateTypeU8, wire.OperateOpINCR, 1, 0)}, nil)
	// Now write field 0 as U32 with a value that would truncate under U8.
	got := callOperate(t, tx, 2, key, 0,
		[]wire.OperateOp{gOp(0, wire.OperateTypeU32, wire.OperateOpINCR, 100000, 0)},
		[]wire.OperateRet{gRet(0), gRet(5)})
	if got[0] != 100000 {
		t.Fatalf("hole field 0 = %d, want 100000 (U32, not truncated to U8)", got[0])
	}
	if got[1] != 1 {
		t.Fatalf("field 5 = %d, want 1", got[1])
	}
	// The other holes (1..4) read as 0 and are still untyped: writing field 2 as
	// IVARINT with a negative value must hold the negative, not a U8 clamp.
	got = callOperate(t, tx, 3, key, 0,
		[]wire.OperateOp{gOp(2, wire.OperateTypeIVARINT, wire.OperateOpINCR, -7, 0)},
		[]wire.OperateRet{gRet(2)})
	if got[0] != -7 {
		t.Fatalf("hole field 2 = %d, want -7 (IVARINT adopted)", got[0])
	}
}

func TestOperateRecordTooLargeRejected(t *testing.T) {
	old := maxOperateRecordBytes
	maxOperateRecordBytes = 8 // tiny cap so a couple of fields exceed it
	defer func() { maxOperateRecordBytes = old }()

	_, tx := newTestSetup(t)
	key := []byte("big")
	tx.SetApplyStamp(1, true)
	_, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeU64, wire.OperateOpINCR, 1, 0),
		gOp(1, wire.OperateTypeU64, wire.OperateOpINCR, 1, 0),
	}, nil))
	tx.SetApplyStamp(0, false)
	if err != errOperateRecordTooLarge {
		t.Fatalf("oversized record: err = %v, want errOperateRecordTooLarge", err)
	}
	// Atomic: nothing was written.
	got := callOperate(t, tx, 2, key, 0, nil, []wire.OperateRet{gRet(0)})
	if got[0] != 0 {
		t.Fatalf("record written despite size rejection: got %d, want 0", got[0])
	}
}

// TestOperateUnstampedEvictionIsDeterministicLowestKey: on the UNSTAMPED path
// touchMs is 0 for every entry (never a per-replica wall clock), so cap-eviction
// deterministically evicts the smallest key. This is the documented trade-off:
// determinism over LRU quality when apply-stamping is off.
func TestOperateUnstampedEvictionIsDeterministicLowestKey(t *testing.T) {
	_, tx := newTestSetup(t) // NewTxContext ⇒ unstamped
	key := []byte("unstamped")
	do := func(ops []wire.OperateOp, ret []wire.OperateRet) []int64 {
		res, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, 2, ops, ret))
		if err != nil {
			t.Fatalf("handleOperate: %v", err)
		}
		vals, _ := wire.DecodeOperateResult(res)
		return vals
	}
	// Touch entry 9 first, then entry 1 (later) — but unstamped, both touchMs=0.
	do([]wire.OperateOp{eOp(9, 0, wire.OperateTypeU8, wire.OperateOpINCR, 90, 0)}, nil)
	do([]wire.OperateOp{eOp(1, 0, wire.OperateTypeU8, wire.OperateOpINCR, 10, 0)}, nil)
	// Insert entry 5 over the cap of 2: tie on touchMs=0 ⇒ smallest KEY (1) evicted.
	got := do([]wire.OperateOp{eOp(5, 0, wire.OperateTypeU8, wire.OperateOpINCR, 50, 0)},
		[]wire.OperateRet{eRet(1, 0), eRet(9, 0), eRet(5, 0)})
	if got[0] != 0 {
		t.Fatalf("unstamped eviction: smallest key 1 should be evicted, got %d", got[0])
	}
	if got[1] != 90 || got[2] != 50 {
		t.Fatalf("unstamped eviction: entry 9/5 = %d/%d, want 90/50", got[1], got[2])
	}
}

// TestOperateUnstampedReplicaReapplyMatches: two independent "replicas" (separate
// caches) applying the SAME committed op-list on the UNSTAMPED path produce
// byte-identical stored records — the regression guard for the wall-clock
// divergence bug (a per-replica wall clock would write different touchMs).
func TestOperateUnstampedReplicaReapplyMatches(t *testing.T) {
	opsList := []wire.OperateOp{
		gOp(0, wire.OperateTypeU16, wire.OperateOpINCR, 7, 0),
		eOp(100, 0, wire.OperateTypeUVARINT, wire.OperateOpINCR, 1, 0),
		eOp(5, 0, wire.OperateTypeU8, wire.OperateOpINCR, 3, 0),
		eOp(100, 1, wire.OperateTypeI64, wire.OperateOpSETMAX, 9, 0),
	}
	key := []byte("repl")
	stored := func() []byte {
		_, tx := newTestSetup(t) // unstamped, fresh cache = one "replica"
		if _, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, 0, opsList, nil)); err != nil {
			t.Fatalf("handleOperate: %v", err)
		}
		v, err := tx.Get(key)
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		out := make([]byte, len(v))
		copy(out, v)
		return out
	}
	if a, b := stored(), stored(); !bytes.Equal(a, b) {
		t.Fatal("two unstamped replicas produced divergent record bytes")
	}
}

// TestOperateAmplificationDoSRejected: an op-list that tries to materialize more
// than the whole-record field budget is rejected INCREMENTALLY (before allocating
// gigabytes). Each op grows a distinct entry's array to the per-array max; the op
// that would push the record's total field count past maxOperateTotalFields fails,
// aborting the whole op-list with nothing written.
func TestOperateAmplificationDoSRejected(t *testing.T) {
	_, tx := newTestSetup(t)
	// Each op grows entry k's array to maxOperateFields; after ~16 of them the total
	// crosses maxOperateTotalFields (1<<20). Build a few more than needed.
	nOps := maxOperateTotalFields/maxOperateFields + 3
	ops := make([]wire.OperateOp, nOps)
	for k := range ops {
		ops[k] = eOp(uint64(k), uint16(maxOperateFields-1), wire.OperateTypeU8, wire.OperateOpINCR, 1, 0)
	}
	tx.SetApplyStamp(1, true)
	_, err := handleOperate(tx, wire.EncodeOperateArgs([]byte("dos"), 0, 0, ops, nil))
	tx.SetApplyStamp(0, false)
	if err != errOperateFieldRange {
		t.Fatalf("amplification op-list: err = %v, want errOperateFieldRange", err)
	}
	// Atomic: nothing written.
	got := callOperate(t, tx, 2, []byte("dos"), 0, nil, []wire.OperateRet{eRet(0, uint16(maxOperateFields-1))})
	if got[0] != 0 {
		t.Fatalf("record written despite DoS rejection: got %d", got[0])
	}
}

// TestOperateHalveGroupTypesUnsetSlots: a HALVE_GRP that first materializes a field
// must fix that field's type to the group's declared type, so a LATER wider write
// cannot adopt a different type and bypass the group's saturation.
func TestOperateHalveGroupTypesUnsetSlots(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("hgtype")
	// HALVE_GRP declares U8 for fields [0,2); it creates them as U8 (threshold 0
	// fires, halving 0→0). A subsequent INCR declaring U32 must NOT re-type field 0:
	// the U8 saturation must hold (300 → 255).
	callOperate(t, tx, 1, key, 0, []wire.OperateOp{
		gOp(0, wire.OperateTypeU8, wire.OperateOpHALVEGRP, 0, 2),
	}, nil)
	got := callOperate(t, tx, 2, key, 0,
		[]wire.OperateOp{gOp(0, wire.OperateTypeU32, wire.OperateOpINCR, 300, 0)},
		[]wire.OperateRet{gRet(0)})
	if got[0] != 255 {
		t.Fatalf("HALVE_GRP-typed field = %d, want 255 (U8 saturation held, not re-typed to U32)", got[0])
	}
}

// TestOperateEncodeDeterministic: encodeRec is a pure function of record contents,
// independent of Go map iteration order, and stable across a decode/encode cycle.
func TestOperateEncodeDeterministic(t *testing.T) {
	build := func() operateRec {
		rec := operateRec{entries: map[uint64]*operateEntry{}}
		for _, k := range []uint64{50, 3, 900, 1, 42, 7} {
			applyOp(&rec, eOp(k, 0, wire.OperateTypeU16, wire.OperateOpINCR, int64(k), 0), 1234, 0) //nolint:errcheck // fixed valid ops
			applyOp(&rec, eOp(k, 1, wire.OperateTypeUVARINT, wire.OperateOpINCR, int64(k*2), 0), 1234, 0)
		}
		applyOp(&rec, gOp(0, wire.OperateTypeF64, wire.OperateOpINCRF, f64arg(1.25), 0), 1234, 0)
		applyOp(&rec, gOp(2, wire.OperateTypeI32, wire.OperateOpINCR, 222, 0), 1234, 0)
		return rec
	}
	want := encodeRec(build())
	for range 20 {
		if got := encodeRec(build()); !bytes.Equal(got, want) {
			t.Fatal("encodeRec is not deterministic across map layouts")
		}
	}
	dec, err := decodeRec(want)
	if err != nil {
		t.Fatalf("decodeRec: %v", err)
	}
	if got := encodeRec(dec); !bytes.Equal(got, want) {
		t.Fatal("decode∘encode is not stable")
	}
}

// TestOperateReplicaReapplyMatches: two independent decode→apply→encode passes of
// the SAME committed op-list (leader vs follower) produce byte-identical state,
// including typed + varint + float fields. This is the RF>1 determinism contract.
func TestOperateReplicaReapplyMatches(t *testing.T) {
	opsList := []wire.OperateOp{
		gOp(0, wire.OperateTypeU16, wire.OperateOpINCR, 7, 0),
		eOp(100, 0, wire.OperateTypeUVARINT, wire.OperateOpINCR, 1, 0),
		eOp(5, 2, wire.OperateTypeU32, wire.OperateOpSHIFTOR, 3, 1),
		eOp(100, 1, wire.OperateTypeI64, wire.OperateOpSETMAX, 9, 0),
		gOp(1, wire.OperateTypeF32, wire.OperateOpINCRF, f32arg(2.5), 0),
		gOp(2, wire.OperateTypeU8, wire.OperateOpHALVEGRP, 0, 2), // threshold 0 always fires
	}
	const nowMs = 987654
	apply := func(start []byte) []byte {
		rec, err := decodeRec(start)
		if err != nil {
			t.Fatalf("decodeRec: %v", err)
		}
		for _, o := range opsList {
			if err := applyOp(&rec, o, nowMs, 8); err != nil {
				t.Fatalf("applyOp: %v", err)
			}
		}
		return encodeRec(rec)
	}
	leader := apply(nil)
	follower := apply(nil)
	if !bytes.Equal(leader, follower) {
		t.Fatal("leader and follower diverged on the same op-list")
	}
	if !bytes.Equal(apply(leader), apply(follower)) {
		t.Fatal("second-round apply diverged")
	}
}

// TestDecodeRecHostile: decodeRec reads whatever bytes are stored under the key (a
// plain put can set arbitrary bytes), so it must never panic on hostile input,
// including unknown type tags and truncated/over-long varints.
func TestDecodeRecHostile(t *testing.T) {
	seeds := [][]byte{
		nil,
		{},
		{1},
		{0xFF, 0xFF}, // nGlobals=65535, no data
		func() []byte { // one global field with an unknown type tag
			b := binary.LittleEndian.AppendUint16(nil, 1)
			return append(b, 200) // type 200 (unknown)
		}(),
		func() []byte { // one global UVARINT that never terminates (all continuation bits)
			b := binary.LittleEndian.AppendUint16(nil, 1)
			b = append(b, wire.OperateTypeUVARINT)
			for range 12 {
				b = append(b, 0x80) // continuation bit set, no terminator → over-long
			}
			return b
		}(),
		func() []byte { // nGlobals sane, nEntries hostile
			b := binary.LittleEndian.AppendUint16(nil, 0)
			b = binary.LittleEndian.AppendUint32(b, 0xFFFFFFFF)
			return b
		}(),
		func() []byte { // one entry with hostile nFields
			b := binary.LittleEndian.AppendUint16(nil, 0) // nGlobals
			b = binary.LittleEndian.AppendUint32(b, 1)    // nEntries
			b = binary.LittleEndian.AppendUint64(b, 1)    // key
			b = binary.LittleEndian.AppendUint64(b, 0)    // touchMs
			b = binary.LittleEndian.AppendUint16(b, 0xFFFF)
			return b
		}(),
	}
	// The contract is simply that decodeRec RETURNS (never panics) on each hostile
	// seed — reaching the end of the loop is the assertion.
	for _, s := range seeds {
		_, _ = decodeRec(s)
	}
}

func FuzzDecodeRec(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0})
	// A valid multi-type record as a structured seed.
	neg := int64(-4)
	rec := operateRec{entries: map[uint64]*operateEntry{
		7: {touchMs: 1, fields: []operateField{{typ: wire.OperateTypeUVARINT, u: 300}, {typ: wire.OperateTypeF32, f: 1.5}}},
	}, globals: []operateField{{typ: wire.OperateTypeU8, u: 9}, {typ: wire.OperateTypeIVARINT, u: uint64(neg)}}}
	f.Add(encodeRec(rec))
	f.Fuzz(func(t *testing.T, b []byte) {
		_, _ = decodeRec(b) // contract: never panic
	})
}
