// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
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

func gOp(field uint16, op uint8, arg, arg2 int64) wire.OperateOp {
	return wire.OperateOp{Target: wire.OperateTargetGlobal, FieldIdx: field, Opcode: op, Arg: arg, Arg2: arg2}
}
func eOp(key uint64, field uint16, op uint8, arg, arg2 int64) wire.OperateOp {
	return wire.OperateOp{Target: wire.OperateTargetEntry, EntryKey: key, FieldIdx: field, Opcode: op, Arg: arg, Arg2: arg2}
}
func gRet(field uint16) wire.OperateRet {
	return wire.OperateRet{Target: wire.OperateTargetGlobal, FieldIdx: field}
}
func eRet(key uint64, field uint16) wire.OperateRet {
	return wire.OperateRet{Target: wire.OperateTargetEntry, EntryKey: key, FieldIdx: field}
}

func TestOperateOpcodes(t *testing.T) {
	tests := []struct {
		name string
		ops  []wire.OperateOp
		ret  []wire.OperateRet
		want []int64
	}{
		{
			name: "INCR accumulates",
			ops:  []wire.OperateOp{gOp(0, wire.OperateOpINCR, 5, 0), gOp(0, wire.OperateOpINCR, -2, 0)},
			ret:  []wire.OperateRet{gRet(0)},
			want: []int64{3},
		},
		{
			name: "INCRF is integer add",
			ops:  []wire.OperateOp{gOp(1, wire.OperateOpINCRF, 1500, 0), gOp(1, wire.OperateOpINCRF, 250, 0)},
			ret:  []wire.OperateRet{gRet(1)},
			want: []int64{1750},
		},
		{
			name: "SETMAX keeps the larger",
			ops:  []wire.OperateOp{gOp(2, wire.OperateOpSETMAX, 10, 0), gOp(2, wire.OperateOpSETMAX, 4, 0), gOp(2, wire.OperateOpSETMAX, 12, 0)},
			ret:  []wire.OperateRet{gRet(2)},
			want: []int64{12},
		},
		{
			name: "SHIFTOR builds a bitfield",
			ops:  []wire.OperateOp{gOp(3, wire.OperateOpSHIFTOR, 4, 0x1), gOp(3, wire.OperateOpSHIFTOR, 4, 0x2)},
			ret:  []wire.OperateRet{gRet(3)},
			want: []int64{0x12},
		},
		{
			name: "SHIFTOR shift>=64 yields arg2",
			ops:  []wire.OperateOp{gOp(0, wire.OperateOpINCR, 999, 0), gOp(0, wire.OperateOpSHIFTOR, 64, 7)},
			ret:  []wire.OperateRet{gRet(0)},
			want: []int64{7},
		},
		{
			name: "HALVE_GRP halves when guard >= threshold",
			ops: []wire.OperateOp{
				gOp(0, wire.OperateOpINCR, 300, 0), // guard field 0 = 300
				gOp(1, wire.OperateOpINCR, 80, 0),  // field 1 = 80
				gOp(0, wire.OperateOpHALVEGRP, 255, 2),
			},
			ret:  []wire.OperateRet{gRet(0), gRet(1)},
			want: []int64{150, 40},
		},
		{
			name: "HALVE_GRP no-op when guard < threshold",
			ops: []wire.OperateOp{
				gOp(0, wire.OperateOpINCR, 100, 0),
				gOp(1, wire.OperateOpINCR, 80, 0),
				gOp(0, wire.OperateOpHALVEGRP, 255, 2),
			},
			ret:  []wire.OperateRet{gRet(0), gRet(1)},
			want: []int64{100, 80},
		},
		{
			name: "entry fields are independent of globals",
			ops:  []wire.OperateOp{gOp(0, wire.OperateOpINCR, 1, 0), eOp(7, 0, wire.OperateOpINCR, 42, 0)},
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

func TestOperatePersistsAcrossCalls(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("counter")
	callOperate(t, tx, 100, key, 0, []wire.OperateOp{gOp(0, wire.OperateOpINCR, 1, 0)}, nil)
	callOperate(t, tx, 200, key, 0, []wire.OperateOp{gOp(0, wire.OperateOpINCR, 1, 0)}, nil)
	got := callOperate(t, tx, 300, key, 0, []wire.OperateOp{gOp(0, wire.OperateOpINCR, 1, 0)}, []wire.OperateRet{gRet(0)})
	if got[0] != 3 {
		t.Fatalf("accumulated = %d, want 3", got[0])
	}
}

func TestOperateCapEvictsDeterministicLRU(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("sess")
	// Create entry 1 at t=100, entry 2 at t=200 (cap 2). Mark each so a read
	// distinguishes present (marker) from evicted (0).
	callOperate(t, tx, 100, key, 2, []wire.OperateOp{eOp(1, 0, wire.OperateOpINCR, 11, 0)}, nil)
	callOperate(t, tx, 200, key, 2, []wire.OperateOp{eOp(2, 0, wire.OperateOpINCR, 22, 0)}, nil)
	// Insert entry 3 at t=300 over the cap → smallest touchMs (entry 1) evicted.
	got := callOperate(t, tx, 300, key, 2,
		[]wire.OperateOp{eOp(3, 0, wire.OperateOpINCR, 33, 0)},
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
	// Two entries created in the SAME op-list share one touchMs; the tie must break
	// toward the smallest key (5 evicted, not 9) when a third arrives at the cap.
	callOperate(t, tx, 100, key, 2, []wire.OperateOp{
		eOp(9, 0, wire.OperateOpINCR, 99, 0),
		eOp(5, 0, wire.OperateOpINCR, 55, 0),
	}, nil)
	got := callOperate(t, tx, 100, key, 2,
		[]wire.OperateOp{eOp(7, 0, wire.OperateOpINCR, 77, 0)},
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
	callOperate(t, tx, 100, key, 0, []wire.OperateOp{gOp(0, wire.OperateOpINCR, 5, 0)}, nil)
	// An op-list whose second op is invalid must leave the record at 5 (the first
	// op's mutation is discarded — the whole operate is all-or-nothing).
	tx.SetApplyStamp(200, true)
	_, err := handleOperate(tx, wire.EncodeOperateArgs(key, 0, 0, []wire.OperateOp{
		gOp(0, wire.OperateOpINCR, 100, 0),
		{Target: wire.OperateTargetGlobal, FieldIdx: 0, Opcode: 250 /* unknown */},
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
		[]wire.OperateOp{gOp(0, wire.OperateOpSHIFTOR, -1, 0)}, nil))
	if err == nil {
		t.Fatal("negative shift accepted, want error")
	}
}

func TestOperateFieldIndexOverflowRejected(t *testing.T) {
	_, tx := newTestSetup(t)
	tx.SetApplyStamp(1, true)
	defer tx.SetApplyStamp(0, false)
	// fieldIdx 65535 needs a globals length of 65536, above maxOperateFields.
	_, err := handleOperate(tx, wire.EncodeOperateArgs([]byte("o"), 0, 0,
		[]wire.OperateOp{gOp(65535, wire.OperateOpINCR, 1, 0)}, nil))
	if err == nil {
		t.Fatal("overflowing field index accepted, want error")
	}
}

// TestOperateEncodeDeterministic: encodeRec is a pure function of record contents,
// independent of Go map iteration order, and stable across a decode/encode cycle.
func TestOperateEncodeDeterministic(t *testing.T) {
	build := func() operateRec {
		rec := operateRec{entries: map[uint64]*operateEntry{}}
		// Insert in different orders each call to perturb map layout.
		for _, k := range []uint64{50, 3, 900, 1, 42, 7} {
			applyOp(&rec, eOp(k, 0, wire.OperateOpINCR, int64(k), 0), 1234, 0) //nolint:errcheck // fixed valid ops
			applyOp(&rec, eOp(k, 1, wire.OperateOpINCR, int64(k*2), 0), 1234, 0)
		}
		applyOp(&rec, gOp(0, wire.OperateOpINCR, 111, 0), 1234, 0)
		applyOp(&rec, gOp(2, wire.OperateOpINCR, 222, 0), 1234, 0)
		return rec
	}
	want := encodeRec(build())
	for range 20 {
		if got := encodeRec(build()); !bytes.Equal(got, want) {
			t.Fatal("encodeRec is not deterministic across map layouts")
		}
	}
	// Decode/encode round-trips to the identical bytes.
	dec, err := decodeRec(want)
	if err != nil {
		t.Fatalf("decodeRec: %v", err)
	}
	if got := encodeRec(dec); !bytes.Equal(got, want) {
		t.Fatal("decode∘encode is not stable")
	}
}

// TestOperateReplicaReapplyMatches: two independent decode→apply→encode passes of
// the SAME committed op-list (the leader vs a follower) produce byte-identical
// state. This is the RF>1 determinism contract.
func TestOperateReplicaReapplyMatches(t *testing.T) {
	opsList := []wire.OperateOp{
		gOp(0, wire.OperateOpINCR, 7, 0),
		eOp(100, 0, wire.OperateOpINCR, 1, 0),
		eOp(5, 2, wire.OperateOpSHIFTOR, 3, 1),
		eOp(100, 1, wire.OperateOpSETMAX, 9, 0),
		gOp(1, wire.OperateOpHALVEGRP, 0, 2), // threshold 0 always fires
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
	// Applying the committed op-list again on top of a prior committed state also
	// stays identical between replicas.
	if !bytes.Equal(apply(leader), apply(follower)) {
		t.Fatal("second-round apply diverged")
	}
}

// TestDecodeRecHostile: decodeRec reads whatever bytes are stored under the key
// (a plain put can set arbitrary bytes), so it must never panic on hostile input.
func TestDecodeRecHostile(t *testing.T) {
	seeds := [][]byte{
		nil,
		{},
		{1},
		{0xFF, 0xFF}, // nGlobals=65535, no data
		func() []byte { // nGlobals sane, nEntries hostile
			b := make([]byte, 0)
			b = binary.LittleEndian.AppendUint16(b, 0)
			b = binary.LittleEndian.AppendUint32(b, 0xFFFFFFFF)
			return b
		}(),
		func() []byte { // one entry with hostile nFields
			b := make([]byte, 0)
			b = binary.LittleEndian.AppendUint16(b, 0) // nGlobals
			b = binary.LittleEndian.AppendUint32(b, 1) // nEntries
			b = binary.LittleEndian.AppendUint64(b, 1) // key
			b = binary.LittleEndian.AppendUint64(b, 0) // touchMs
			b = binary.LittleEndian.AppendUint16(b, 0xFFFF)
			return b
		}(),
	}
	for i, s := range seeds {
		if _, err := decodeRec(s); err == nil && len(s) > 4 {
			// Some seeds are legitimately decodable; we only require NO PANIC. The
			// call above returning is the assertion.
			_ = i
		}
	}
}

func FuzzDecodeRec(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, b []byte) {
		_, _ = decodeRec(b) // contract: never panic
	})
}
