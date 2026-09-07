// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"math"
	"testing"
	"time"
)

func opsRoundtripEqual(t *testing.T, a, b []OperateOp) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("ops len = %d, want %d", len(b), len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("op[%d] = %+v, want %+v", i, b[i], a[i])
		}
	}
}

func retRoundtripEqual(t *testing.T, a, b []OperateRet) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("ret len = %d, want %d", len(b), len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("ret[%d] = %+v, want %+v", i, b[i], a[i])
		}
	}
}

func TestOperateArgsRoundtrip(t *testing.T) {
	key := []byte("session:42")
	ttl := 90 * time.Second
	maxEntries := uint16(1024)
	ops := []OperateOp{
		{Target: OperateTargetGlobal, FieldIdx: 0, Opcode: OperateOpINCR, Type: OperateTypeU8, Arg: 1},
		{Target: OperateTargetGlobal, FieldIdx: 3, Opcode: OperateOpINCRF, Type: OperateTypeF32, Arg: 12345},
		{Target: OperateTargetGlobal, FieldIdx: 2, Opcode: OperateOpSETMAX, Type: OperateTypeI16, Arg: -7},
		{Target: OperateTargetEntry, EntryKey: 0xDEADBEEF, FieldIdx: 1, Opcode: OperateOpSHIFTOR, Type: OperateTypeU32, Arg: 4, Arg2: 0xF},
		{Target: OperateTargetEntry, EntryKey: 99, FieldIdx: 0, Opcode: OperateOpHALVEGRP, Type: OperateTypeUVARINT, Arg: 255, Arg2: 2},
	}
	ret := []OperateRet{
		{Target: OperateTargetGlobal, FieldIdx: 0},
		{Target: OperateTargetEntry, EntryKey: 0xDEADBEEF, FieldIdx: 1},
	}

	enc := EncodeOperateArgs(key, ttl, maxEntries, ops, ret)
	gotKey, gotTTL, gotMax, gotOps, gotRet, err := DecodeOperateArgs(enc)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if string(gotKey) != string(key) {
		t.Fatalf("key = %q, want %q", gotKey, key)
	}
	if gotTTL != ttl {
		t.Fatalf("ttl = %v, want %v", gotTTL, ttl)
	}
	if gotMax != maxEntries {
		t.Fatalf("maxEntries = %d, want %d", gotMax, maxEntries)
	}
	opsRoundtripEqual(t, ops, gotOps)
	retRoundtripEqual(t, ret, gotRet)
}

func TestOperateArgsEmptyOpsAndRet(t *testing.T) {
	enc := EncodeOperateArgs([]byte("k"), 0, 0, nil, nil)
	k, ttl, max, ops, ret, err := DecodeOperateArgs(enc)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if string(k) != "k" || ttl != 0 || max != 0 || len(ops) != 0 || len(ret) != 0 {
		t.Fatalf("unexpected: k=%q ttl=%v max=%d ops=%d ret=%d", k, ttl, max, len(ops), len(ret))
	}
}

func TestOperateResultRoundtrip(t *testing.T) {
	vals := []int64{0, -1, math.MaxInt64, math.MinInt64, 42}
	got, err := DecodeOperateResult(EncodeOperateResult(vals))
	if err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if len(got) != len(vals) {
		t.Fatalf("len = %d, want %d", len(got), len(vals))
	}
	for i := range vals {
		if got[i] != vals[i] {
			t.Fatalf("val[%d] = %d, want %d", i, got[i], vals[i])
		}
	}
}

func TestOperateResultRejectsRagged(t *testing.T) {
	if _, err := DecodeOperateResult([]byte{0, 0, 0}); err == nil {
		t.Fatal("ragged result accepted, want error")
	}
}

func TestDecodeOperateArgsTruncation(t *testing.T) {
	full := EncodeOperateArgs([]byte("abc"), time.Second, 8,
		[]OperateOp{{Target: OperateTargetEntry, EntryKey: 5, FieldIdx: 1, Opcode: OperateOpINCR, Type: OperateTypeU16, Arg: 1}},
		[]OperateRet{{Target: OperateTargetEntry, EntryKey: 5, FieldIdx: 1}})
	// Every prefix short of the whole frame must be rejected, never panic.
	for n := 0; n < len(full); n++ {
		if _, _, _, _, _, err := DecodeOperateArgs(full[:n]); err == nil {
			t.Fatalf("prefix len %d accepted, want error", n)
		}
	}
}

// buildOperateFrame assembles an args frame with one global op carrying the given
// target/opcode/type bytes, for the malformed-tag tests.
func buildOperateFrame(target, opcode, ftype uint8) []byte {
	frame := make([]byte, 0)
	frame = binary.BigEndian.AppendUint16(frame, 0) // keyLen=0
	frame = binary.BigEndian.AppendUint64(frame, 0) // ttlMs
	frame = binary.BigEndian.AppendUint16(frame, 0) // maxEntries
	frame = binary.BigEndian.AppendUint32(frame, 1) // nOps=1
	frame = append(frame, target)
	frame = binary.BigEndian.AppendUint16(frame, 0) // fieldIdx
	frame = append(frame, opcode)
	frame = append(frame, ftype)
	frame = binary.BigEndian.AppendUint64(frame, 0) // arg
	frame = binary.BigEndian.AppendUint64(frame, 0) // arg2
	frame = binary.BigEndian.AppendUint16(frame, 0) // nRet=0
	return frame
}

func TestDecodeOperateArgsBadTarget(t *testing.T) {
	if _, _, _, _, _, err := DecodeOperateArgs(buildOperateFrame(2, OperateOpINCR, OperateTypeU8)); err != ErrBadOperateTarget {
		t.Fatalf("bad target: err = %v, want ErrBadOperateTarget", err)
	}
}

func TestDecodeOperateArgsBadType(t *testing.T) {
	if _, _, _, _, _, err := DecodeOperateArgs(buildOperateFrame(OperateTargetGlobal, OperateOpINCR, OperateTypeCount)); err != ErrBadOperateType {
		t.Fatalf("bad type: err = %v, want ErrBadOperateType", err)
	}
	if _, _, _, _, _, err := DecodeOperateArgs(buildOperateFrame(OperateTargetGlobal, OperateOpINCR, 200)); err != ErrBadOperateType {
		t.Fatalf("bad type 200: err = %v, want ErrBadOperateType", err)
	}
}

func TestDecodeOperateArgsHostileNOps(t *testing.T) {
	// A huge declared nOps with a tiny body must be rejected by CountFitsIn before
	// any allocation — never a panic or OOM.
	frame := make([]byte, 0)
	frame = binary.BigEndian.AppendUint16(frame, 0) // keyLen=0
	frame = binary.BigEndian.AppendUint64(frame, 0) // ttlMs
	frame = binary.BigEndian.AppendUint16(frame, 0) // maxEntries
	frame = binary.BigEndian.AppendUint32(frame, 0xFFFFFFFF)
	if _, _, _, _, _, err := DecodeOperateArgs(frame); err == nil {
		t.Fatal("hostile nOps accepted, want error")
	}
}

func TestDecodeOperateArgsTTLOverflow(t *testing.T) {
	frame := make([]byte, 0)
	frame = binary.BigEndian.AppendUint16(frame, 0)              // keyLen=0
	frame = binary.BigEndian.AppendUint64(frame, math.MaxUint64) // ttlMs overflow
	frame = binary.BigEndian.AppendUint16(frame, 0)              // maxEntries
	frame = binary.BigEndian.AppendUint32(frame, 0)              // nOps=0
	frame = binary.BigEndian.AppendUint16(frame, 0)              // nRet=0
	if _, _, _, _, _, err := DecodeOperateArgs(frame); err != ErrTTLOutOfRange {
		t.Fatalf("ttl overflow: err = %v, want ErrTTLOutOfRange", err)
	}
}

func FuzzDecodeOperateArgs(f *testing.F) {
	f.Add(EncodeOperateArgs([]byte("k"), time.Second, 4,
		[]OperateOp{{Target: OperateTargetGlobal, FieldIdx: 0, Opcode: OperateOpINCR, Type: OperateTypeU8, Arg: 1}},
		[]OperateRet{{Target: OperateTargetGlobal, FieldIdx: 0}}))
	// One seed per field type so the corpus exercises every tag.
	for ty := uint8(0); ty < OperateTypeCount; ty++ {
		f.Add(EncodeOperateArgs([]byte("t"), 0, 2,
			[]OperateOp{{Target: OperateTargetEntry, EntryKey: 1, FieldIdx: 0, Opcode: OperateOpINCR, Type: ty, Arg: 3}},
			[]OperateRet{{Target: OperateTargetEntry, EntryKey: 1, FieldIdx: 0}}))
	}
	f.Add([]byte{})
	f.Add([]byte{0, 1})
	f.Fuzz(func(t *testing.T, b []byte) {
		// Contract: never panic on arbitrary bytes.
		_, _, _, _, _, _ = DecodeOperateArgs(b)
	})
}
