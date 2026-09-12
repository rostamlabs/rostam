// SPDX-License-Identifier: Apache-2.0

package server

// The cross-package invariant between wire.OperateMaxRetBytes and
// MaxFrameSize: a return list the operate byte budget ACCEPTS must encode to a
// reply this transport can actually send.
//
// The two constants live in different packages and neither can reference the
// other — wire is below server, and ops (which enforces the budget) must not
// import server at all — so nothing but a test holds them together. Without
// this, the budget could be raised to exactly MaxFrameSize again and the
// largest accepted call would die at the edge as "response exceeds
// MaxFrameSize": the same refusal, arriving after the work instead of before
// it, and naming the wrong cause.

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// budgetFullResult is an OperateResult carrying exactly wire.OperateMaxRetBytes
// of values, spread over as many 4096-byte entries as that divides into — the
// same shape ops.TestOperateRetBytesBudgetBoundary pins as the largest accepted
// call.
func budgetFullResult(t *testing.T, status uint8) *wire.OperateResult {
	t.Helper()
	const valueSize = 4096
	if wire.OperateMaxRetBytes%valueSize != 0 {
		t.Fatalf("budget %d is no longer a multiple of %d", wire.OperateMaxRetBytes, valueSize)
	}
	n := wire.OperateMaxRetBytes / valueSize
	if n > wire.OperateMaxRet {
		t.Fatalf("%d values is past the count cap %d", n, wire.OperateMaxRet)
	}
	one := bytes.Repeat([]byte{0xCD}, valueSize)
	vals := make([][]byte, n)
	for i := range vals {
		vals[i] = one
	}
	// FailedOp is only written for CHECK_FAILED, which is the point of running
	// this for both statuses: that shape is two bytes longer.
	return &wire.OperateResult{Status: status, FailedOp: 7, Values: vals}
}

// TestOperateRetBudgetFramesWithinMaxFrameSize is the guard the
// wire.OperateMaxRetBytes doc comment names. A reply at exactly the budget must
// survive clampResponse untouched, on both result statuses and on the vector
// wrapper, which adds a further 13 bytes.
func TestOperateRetBudgetFramesWithinMaxFrameSize(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status uint8
	}{
		{"OK", wire.OperateStatusOK},
		// CHECK_FAILED writes the extra failedOp field, so it is the larger of
		// the two envelopes and the one the headroom has to cover.
		{"CHECK_FAILED (carries failedOp)", wire.OperateStatusCheckFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload, err := wire.EncodeOperateResult(budgetFullResult(t, tc.status))
			if err != nil {
				t.Fatalf("EncodeOperateResult: %v", err)
			}
			assertFrames(t, "operate", payload)

			vpayload, err := wire.EncodeVectorOperateResult(true, budgetFullResult(t, tc.status), 1)
			if err != nil {
				t.Fatalf("EncodeVectorOperateResult: %v", err)
			}
			if len(vpayload) <= len(payload) {
				t.Fatalf("the vector wrapper should be the larger frame: %d vs %d", len(vpayload), len(payload))
			}
			assertFrames(t, "vector_operate", vpayload)
		})
	}
}

// assertFrames pins that clampResponse passes payload through unchanged. It
// reports the headroom left either way, because that number is the whole
// subject: a failure here means the budget and the frame limit have drifted
// into each other, and the margin says by how much.
func assertFrames(t *testing.T, what string, payload []byte) {
	t.Helper()
	frame := 1 + 4 + len(payload) // exactly what clampResponse bounds
	gotStatus, gotPayload := clampResponse(StatusOK, payload)
	if gotStatus != StatusOK {
		msg, _ := DecodeErrorPayload(gotPayload)
		t.Fatalf("%s reply at exactly the %d-byte return budget did not frame: %q\n"+
			"  frame %d > MaxFrameSize %d, over by %d — wire.OperateMaxRetBytes and server.MaxFrameSize have drifted",
			what, wire.OperateMaxRetBytes, msg, frame, MaxFrameSize, frame-MaxFrameSize)
	}
	if len(gotPayload) != len(payload) {
		t.Fatalf("%s: clampResponse rewrote the payload (%d bytes to %d)", what, len(payload), len(gotPayload))
	}
	t.Logf("%s reply at the full return budget: %d-byte frame, %d bytes of MaxFrameSize headroom left",
		what, frame, MaxFrameSize-frame)
}
