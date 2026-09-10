// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestWCEnvelopeInnerOpMatchesTheStoreWire closes the loop on the retry-guard
// fix from BOTH ends, in the one module that can see both.
//
// The client decides whether an ambiguous failure may be retried by reading the
// inner op name back out of the envelope (wire.WCEnvelopeInnerOp). That is only
// sound if the name it reads is the name this store actually put on the wire, so
// this pins wcWire's output — the real producer, through the real
// ops.EncodeWCEnvelope — against both the peek and ops' own full decoder.
//
// Without this, the envelope layout could drift in ops and the client's guard
// would quietly start reading garbage, which fails OPEN (a wrapped
// vector_operate would look replayable again).
func TestWCEnvelopeInnerOpMatchesTheStoreWire(t *testing.T) {
	active := WriteOpts{WriteConsistencyFactor: 2}
	for _, op := range []string{"vector_operate", "vector_named_operate", "vector_mv_operate", "operate", "vector_insert"} {
		t.Run(op, func(t *testing.T) {
			wireOp, wireArgs := wcWire(op, []byte("inner-args"), active)
			if wireOp != wire.WCEnvelopeOp {
				t.Fatalf("wcWire op = %q, want %q", wireOp, wire.WCEnvelopeOp)
			}
			peeked, ok := wire.WCEnvelopeInnerOp(wireArgs)
			if !ok {
				t.Fatalf("WCEnvelopeInnerOp could not read the frame wcWire built")
			}
			if peeked != op {
				t.Fatalf("WCEnvelopeInnerOp = %q, want %q", peeked, op)
			}
			_, _, decoded, innerArgs, err := ops.DecodeWCEnvelope(wireArgs)
			if err != nil {
				t.Fatalf("DecodeWCEnvelope: %v", err)
			}
			if decoded != peeked {
				t.Fatalf("peeked name %q disagrees with DecodeWCEnvelope's %q", peeked, decoded)
			}
			if string(innerArgs) != "inner-args" {
				t.Fatalf("inner args = %q, want inner-args", innerArgs)
			}
		})
	}
}

// TestWCEnvelopeOpConstantsAgree pins the two names to one value: ops aliases
// the wire constant, and the client compares against the wire one.
func TestWCEnvelopeOpConstantsAgree(t *testing.T) {
	if ops.WCEnvelopeOp != wire.WCEnvelopeOp {
		t.Fatalf("ops.WCEnvelopeOp = %q, wire.WCEnvelopeOp = %q; the two must be the same name", ops.WCEnvelopeOp, wire.WCEnvelopeOp)
	}
	if wire.WCEnvelopeOp != "__wc__" {
		t.Fatalf("WCEnvelopeOp = %q, want __wc__ (a rename is a wire break)", wire.WCEnvelopeOp)
	}
}

// TestWCInactiveOptsSendThePlainOp is the guard's other half: with inactive
// opts no envelope is built at all, so the plain op name reaches the retry
// decision unchanged and the pre-existing classification still applies.
func TestWCInactiveOptsSendThePlainOp(t *testing.T) {
	wireOp, wireArgs := wcWire("vector_operate", []byte("inner-args"), WriteOpts{})
	if wireOp != "vector_operate" {
		t.Fatalf("wcWire op = %q, want the plain vector_operate", wireOp)
	}
	if string(wireArgs) != "inner-args" {
		t.Fatalf("wcWire args = %q, want the byte-identical inner args", wireArgs)
	}
}
