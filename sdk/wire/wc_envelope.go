// SPDX-License-Identifier: Apache-2.0

package wire

// WCEnvelopeOp is the reserved name of the write-consistency envelope
// virtual-op. It is declared in this leaf package, not beside its codec in
// ops, because THREE layers have to agree on it and only two of them can see
// ops: the server-side fanout dispatcher (root module), the store that builds
// the envelope (root module), and the standalone client module, which imports
// this package and nothing else of the tree.
//
// ops.WCEnvelopeOp is this same constant — see ops/write_consistency.go for the
// envelope's full wire layout and its codec.
const WCEnvelopeOp = "__wc__"

// WCEnvelopeInnerOp peeks the inner op NAME out of a write-consistency envelope
// without decoding the rest of the frame. It exists for the retry decision in
// the client: a write wrapped in the envelope is sent under the op name
// "__wc__", so a guard keyed on the op name alone stops seeing the write it is
// meant to protect — a non-replayable conditional write would be retried across
// an ambiguous post-transmission failure and could apply twice.
//
// The layout it reads is the head of ops.EncodeWCEnvelope's frame:
//
//	[wcf u8][wait u8][nameLen u8][name...][argsLen u32][args...]
//
// Only the first three bytes and the name are touched; the inner args are not
// examined, so this costs one bounds check and one string header on a path that
// runs once per failed call. ok is false for any buffer too short to hold the
// declared name — a frame this client did not build — and the caller decides
// what an unreadable envelope means (client.nonReplayableCall treats it as
// non-replayable, which can only cost a retry).
//
// Keep in step with ops.DecodeWCEnvelope: TestWCEnvelopeInnerOpMatchesOpsCodec
// (root module, which can see both) pins the two against each other.
func WCEnvelopeInnerOp(args []byte) (name string, ok bool) {
	if len(args) < 3 {
		return "", false
	}
	nameLen := int(args[2])
	if len(args)-3 < nameLen {
		return "", false
	}
	return string(args[3 : 3+nameLen]), true
}
