// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// WCEnvelopeOp is the reserved name of the write-consistency envelope virtual-op.
// It is NOT registered in builtin.go and is NOT shard-routed: the fanout
// dispatcher intercepts it (mirroring alias_batch / reshard coordinator ops),
// decodes the wrapped inner write op, dispatches that inner op through the
// normal routing/Raft path 100% unchanged, then runs the post-commit barrier.
// Because the inner op's name + args are byte-identical to a plain write, no
// existing data-op codec, FSM handler, decoder, or routing path is touched.
//
// The constant itself lives in sdk/wire (wire.WCEnvelopeOp) because the
// standalone client module has to agree on it too — it must recognise its own
// envelope to classify a wrapped write's retryability — and that module can see
// wire but not ops. This alias keeps every server-side reference reading as
// ops.WCEnvelopeOp.
const WCEnvelopeOp = wire.WCEnvelopeOp

// errWCEnvelopeTruncated is returned by DecodeWCEnvelope when the args bytes are
// shorter than the layout requires (fail-loud, mirroring errAliasArgsTruncated /
// errVectorArgsTruncated). It is returned for any short/oversized/truncated
// buffer so a malformed envelope never panics and never silently mis-decodes.
var errWCEnvelopeTruncated = errors.New("ops: write-consistency envelope truncated")

// EncodeWCEnvelope frames a write-consistency envelope wrapping a single inner
// write op. Wire layout (big-endian, matching the alias/vector codec
// conventions in this package):
//
//	[wcf:u8][wait:u8][nameLen:u8][name...][argsLen:u32][args...]
//
// The inner op name uses a u8 length prefix: op names are capped at
// maxOpNameLen (=255) by the registry (registry.go), so every valid inner op
// name fits. The inner args use a u32 length prefix so large writes (>64KB)
// frame unambiguously. The encoded inner name+args are byte-identical to what a
// plain (non-enveloped) write would carry — the envelope is unwrapped before
// the inner op is routed/logged, so the Raft log entry is unchanged.
//
// innerArgs longer than the u32 length prefix can express (4 GiB) cannot occur
// for any real write op; callers never pass such buffers.
func EncodeWCEnvelope(wcf, wait uint8, innerName string, innerArgs []byte) []byte {
	// A name longer than the u8 length prefix can express would silently wrap
	// (256 → 0) and corrupt the frame. Every registered op name is ≤ maxOpNameLen
	// by construction, so a longer name here is a programming error — fail loud.
	if len(innerName) > maxOpNameLen {
		panic(fmt.Sprintf("ops: EncodeWCEnvelope: inner name %q exceeds maxOpNameLen (%d)", innerName, maxOpNameLen))
	}
	// 1 (wcf) + 1 (wait) + 1 (nameLen) + len(name) + 4 (argsLen) + len(args)
	buf := make([]byte, 0, 1+1+1+len(innerName)+4+len(innerArgs))
	buf = append(buf, wcf, wait)
	buf = append(buf, byte(len(innerName))) //nolint:gosec // name length bounded by maxOpNameLen (255)
	buf = append(buf, innerName...)
	var argsHdr [4]byte
	binary.BigEndian.PutUint32(argsHdr[:], uint32(len(innerArgs))) //nolint:gosec // args length bounded
	buf = append(buf, argsHdr[:]...)
	buf = append(buf, innerArgs...)
	return buf
}

// DecodeWCEnvelope reads an envelope produced by EncodeWCEnvelope. Every field
// is length-checked (fail-loud): on any short, oversized, or truncated buffer it
// returns errWCEnvelopeTruncated and never panics.
//
// The returned innerArgs is a sub-slice that aliases the input args (no copy) —
// it is byte-identical to the inner args that were encoded. The fanout
// dispatcher forwards these bytes straight into f.Call without mutating them, so
// aliasing is safe; a caller that intends to retain or mutate innerArgs beyond
// the lifetime of args must copy it.
func DecodeWCEnvelope(args []byte) (wcf, wait uint8, innerName string, innerArgs []byte, err error) {
	// [wcf:u8][wait:u8][nameLen:u8] header.
	if len(args) < 3 {
		return 0, 0, "", nil, errWCEnvelopeTruncated
	}
	wcf = args[0]
	wait = args[1]
	nameLen := int(args[2])
	off := 3
	// [name...]
	if off+nameLen > len(args) {
		return 0, 0, "", nil, errWCEnvelopeTruncated
	}
	innerName = string(args[off : off+nameLen])
	off += nameLen
	// [argsLen:u32]
	if off+4 > len(args) {
		return 0, 0, "", nil, errWCEnvelopeTruncated
	}
	argsLen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	// [args...] — argsLen is safe on 64-bit (uint32 fits in int); the < 0 check
	// guards 32-bit builds where int(0xffffffff) wraps negative. The comparison is
	// written as argsLen > len(args)-off (never off+argsLen) so a positive argsLen
	// near MaxInt32 cannot overflow off+argsLen back into range on 32-bit.
	if argsLen < 0 || argsLen > len(args)-off {
		return 0, 0, "", nil, errWCEnvelopeTruncated
	}
	innerArgs = args[off : off+argsLen]
	return wcf, wait, innerName, innerArgs, nil
}
