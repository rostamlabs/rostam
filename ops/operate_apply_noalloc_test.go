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
// return specs builds no values slice.
//
// The result used to be a second allocation: returned as
// &wire.OperateResult{...}, one 32-byte heap object per call, it was the
// largest allocation site by OBJECT count on a production node (29% of all
// objects, against 1.4% of bytes, since the struct is small). Returning it by
// value leaves it on the caller's stack. A regression to a pointer return — or
// anything else that escapes — pushes this back over budget.
func TestApplyRecordBytesInPlaceUpdateCostsOneAlloc(t *testing.T) {
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
