// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// resultIsZero reports whether r is the zero result — what applyRecordBytes
// returns alongside an error. OperateResult holds a slice, so it is not
// comparable with ==.
func resultIsZero(r wire.OperateResult) bool {
	return r.Status == 0 && r.FailedOp == 0 && r.Values == nil
}

// applyRecordBytes must cost exactly the ONE allocation the design doc's §2.6
// budget allows on the in-place path: the private record copy openRecord makes
// (copyRecord). Everything else on the call is pooled or stack-held — the
// engines come from schemaEnginePool/dynamicEnginePool, and a call with no
// return specs builds no values slice. Pool reliance is why this is skipped
// under -race, matching TestSchemaEngineAllocs and TestDynamicEngineAllocs.
//
// The result used to be a second allocation: returned as
// &wire.OperateResult{...}, one 32-byte heap object per call, it was the
// largest allocation site by OBJECT count on a production node (29% of all
// objects, against 1.4% of bytes, since the struct is small). Returning it by
// value leaves it on the caller's stack. A regression to a pointer return — or
// anything else that escapes — pushes this back over budget.
func TestApplyRecordBytesInPlaceUpdateCostsOneAlloc(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool.Put randomly drops items under -race by design, which defeats this allocation budget; see race_detect_test.go")
	}
	s := &wire.Schema{Version: 1, Fields: []wire.FieldDef{{Name: "rc", Type: wire.OperateTypeU16}}}
	create := &wire.OperateArgs{
		Create: wire.OperateCreateSchema,
		Schema: s.Encode(),
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}},
	}
	rec, _, _, err := applyRecordBytes(nil, create, 1)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}

	upd := &wire.OperateArgs{
		Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}},
	}
	// Warm the engine pools so the first call's pool fill is not counted.
	if _, _, _, err := applyRecordBytes(rec, upd, 2); err != nil {
		t.Fatalf("warm: %v", err)
	}

	got := testing.AllocsPerRun(200, func() {
		out, deleted, res, aerr := applyRecordBytes(rec, upd, 3)
		if aerr != nil || deleted || out == nil || res.Status != wire.OperateStatusOK {
			t.Fatalf("apply: err=%v deleted=%v out=%x status=%d", aerr, deleted, out, res.Status)
		}
	})
	// The record copy, and nothing else.
	if got != 1 {
		t.Errorf("applyRecordBytes allocated %.1f objects per in-place scalar update; want 1 (the record copy alone)", got)
	}
}

// With the caller's scratch supplying the record copy, an in-place update
// allocates nothing at all: the engines come from pools, the copy reuses the
// scratch array, and no return specs means no values slice. This is the budget
// handleOperate actually runs at on a warm pool.
func TestApplyRecordBytesIntoWarmScratchIsZeroAlloc(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool.Put randomly drops items under -race by design, which defeats this allocation budget; see race_detect_test.go")
	}
	rec, upd := seedScalarRecord(t)
	// Sized from the SAME expression copyRecord uses, so the budget keeps
	// pinning the formula: guessing "comfortably larger" would let a regression
	// to flat +64 slack pass with zero allocations and reopen the leak.
	scratch := make([]byte, 0, copyRecordNeed(len(rec)))
	if _, _, _, err := applyRecordBytesInto(scratch, rec, upd, 2); err != nil {
		t.Fatalf("warm: %v", err)
	}

	got := testing.AllocsPerRun(200, func() {
		out, _, _, aerr := applyRecordBytesInto(scratch, rec, upd, 3)
		if aerr != nil || out == nil {
			t.Fatalf("apply: err=%v out=%x", aerr, out)
		}
	})
	if got != 0 {
		t.Errorf("applyRecordBytesInto allocated %.1f objects per in-place update with a warm scratch; want 0", got)
	}
}

// out ALIASES the caller's scratch whenever scratch's ARRAY had room, and that
// is what handleOperate's recycling turns on. Growth does not decide it:
// insertGap extends in place while `cap(buf)-old >= n` holds, so a record that
// grew can still be sitting in the caller's array. Only exhausting the capacity
// produces a new one. The three cases below are fits / grows-within-capacity /
// too-small, and the middle one is the reason handleOperate must recycle what
// out points at rather than what it passed in — neither growth nor its absence
// tells it which array is live.
func TestApplyRecordBytesIntoOutAliasesScratchWhileCapacityLasts(t *testing.T) {
	rec, upd := seedScalarRecord(t)

	// Compare backing arrays, not contents. Guarded on CAP, not len: the
	// scratch buffers here are len 0 with spare capacity, and reslicing one to
	// [:1] within its capacity is legal and addresses the same array.
	sameArray := func(a, b []byte) bool {
		return cap(a) > 0 && cap(b) > 0 && &a[:1][0] == &b[:1][0]
	}

	// 1. In-place update into a scratch with room: aliases.
	roomy := make([]byte, 0, copyRecordNeed(len(rec)))
	out, _, _, err := applyRecordBytesInto(roomy, rec, upd, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !sameArray(out, roomy) {
		t.Error("in-place update did not reuse a scratch big enough to hold it")
	}

	// 2. A row insert GROWS the record (insertGap). With slack to grow into it
	//    still aliases — the case the old version of this test never reached.
	grower := sessionArgsFor(keyU64(7))
	grower.Create = wire.OperateCreateSchema
	base, _, _, err := applyRecordBytes(nil, sessionArgsFor(keyU64(1)), 1)
	if err != nil {
		t.Fatal(err)
	}
	roomyGrow := make([]byte, 0, len(base)+4096) // plenty of slack for a new row
	out, _, _, err = applyRecordBytesInto(roomyGrow, base, grower, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) <= len(base) {
		t.Fatalf("the record did not grow: %d bytes from %d", len(out), len(base))
	}
	if !sameArray(out, roomyGrow) {
		t.Error("a record that grew WITHIN the scratch's capacity should still alias it")
	}

	// 3. Too small for the initial copy: a fresh array, and the caller's buffer
	//    is untouched.
	tiny := make([]byte, 0, 1)
	out, _, _, err = applyRecordBytesInto(tiny, rec, upd, 3)
	if err != nil {
		t.Fatal(err)
	}
	if sameArray(out, tiny) {
		t.Error("out aliases a scratch too small for the record")
	}
}

// The budget handleOperate itself runs at, which is what this PR changes. The
// test above supplies a scratch by hand and so still passes if handleOperate
// stops using the pool, or stops keeping the returned buffer; this one fails.
// The single remaining allocation is the reply frame EncodeOperateResult builds.
func TestOperateHandlerWarmCallCostsOneAlloc(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool.Put randomly drops items under -race by design, which defeats this allocation budget; see race_detect_test.go")
	}
	_, tx := newTestSetup(t)
	args, err := wire.EncodeOperateArgs(addField0([]byte("warm-budget-key")))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ { // create, then warm both pools
		if _, err := handleOperate(tx, args); err != nil {
			t.Fatal(err)
		}
	}

	got := testing.AllocsPerRun(200, func() {
		if _, err := handleOperate(tx, args); err != nil {
			t.Fatal(err)
		}
	})
	if got != 1 {
		t.Errorf("handleOperate allocated %.1f objects per warm call; want 1 (the reply frame alone)", got)
	}
}

// Two keys alternating through the handler, each with its own value, driven by
// the real pooled buffer. TestOperatePersistsAcrossCalls already loops one key
// with one op, which a buffer mix-up would survive: the bytes are identical
// every call, so recycling the wrong array is invisible. Interleaving two keys
// whose stored records differ is the case that catches it.
func TestOperateHandlerInterleavedKeysKeepSeparateRecords(t *testing.T) {
	_, tx := newTestSetup(t)
	a, b := []byte("pool-key-a"), []byte("pool-key-b")

	// 3 calls on a, 1 on b, interleaved so b's write lands between a's.
	for i, key := range [][]byte{a, a, b, a} {
		if res := callOperate(t, tx, 0, addField0(key)); res.Status != wire.OperateStatusOK {
			t.Fatalf("call %d on %s: status %v", i, key, res.Status)
		}
	}

	for _, tc := range []struct {
		key  []byte
		want uint64
	}{{a, 3}, {b, 1}} {
		v, err := tx.Get(tc.key)
		if err != nil {
			t.Fatalf("Get %s: %v", tc.key, err)
		}
		rec, err := wire.DecodeRecord(v)
		if err != nil {
			t.Fatalf("DecodeRecord %s: %v", tc.key, err)
		}
		if got := rec.Fields[0].Cell.U; got != tc.want {
			t.Errorf("%s field 0 = %d, want %d", tc.key, got, tc.want)
		}
	}
}

// seedScalarRecord builds a one-U16-field schema record plus the ADD args that
// update it in place.
func seedScalarRecord(t *testing.T) (rec []byte, upd *wire.OperateArgs) {
	t.Helper()
	s := &wire.Schema{Version: 1, Fields: []wire.FieldDef{{Name: "rc", Type: wire.OperateTypeU16}}}
	rec, _, _, err := applyRecordBytes(nil, &wire.OperateArgs{
		Create: wire.OperateCreateSchema,
		Schema: s.Encode(),
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}},
	}, 1)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	return rec, &wire.OperateArgs{
		Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}},
	}
}
