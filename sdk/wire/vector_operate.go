// SPDX-License-Identifier: Apache-2.0

package wire

import "encoding/binary"

// EncodeVectorOperateArgs serializes a vector_operate / vector_named_operate /
// vector_mv_operate request (one wire shape, the OP NAME selects the family).
//
// Wire: [colLen u8][col][id u64][pkLen u16][payloadKey]
//
//	[casPresent u8][?expectedVersion u64][argsLen u32][operateArgs]
//
// The collection sits at offset 0 and the point id immediately after it, which
// is what makes this the At1 routing layout (VectorKeyColAt1) and lets
// PointIDFor read the id without a full decode.
//
// a.Key must be EMPTY and a.TTLMode must be OperateTTLKeep with a.TTL == 0:
// the target is named once, by (collection, id, payloadKey), and a point's TTL
// is the point's, never a record's (brief §3). Both are ErrOperateArgs here and
// again in the decoder, so a hand-built frame cannot smuggle either past the
// encoder.
//
// The wrapped inner frame therefore always carries a u16 zero key length and a
// u64 zero ttlMs — ten bytes that are structurally always zero. That is
// DELIBERATE: reusing EncodeOperateArgs verbatim keeps exactly one operate-args
// encoder in the tree. Do not "optimise" it into a second, divergent shape.
func EncodeVectorOperateArgs(collection string, id uint64, payloadKey string, a *OperateArgs, expectedVersion uint64, hasExpected bool) ([]byte, error) {
	if a == nil {
		return nil, ErrOperateArgs
	}
	if len(collection) == 0 || len(collection) > 255 {
		return nil, ErrOperateCap
	}
	if len(payloadKey) == 0 || len(payloadKey) > 0xFFFF {
		return nil, ErrOperateCap
	}
	if len(a.Key) != 0 || a.TTLMode != OperateTTLKeep || a.TTL != 0 {
		return nil, ErrOperateArgs
	}
	inner, err := EncodeOperateArgs(a)
	if err != nil {
		return nil, err
	}

	n := 1 + len(collection) + 8 + 2 + len(payloadKey) + 1
	if hasExpected {
		n += 8
	}
	n += 4 + len(inner)

	buf := make([]byte, 0, n)
	buf = append(buf, byte(len(collection))) //nolint:gosec // bounded by the check above
	buf = append(buf, collection...)
	buf = binary.BigEndian.AppendUint64(buf, id)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(payloadKey))) //nolint:gosec // bounded by the check above
	buf = append(buf, payloadKey...)
	if hasExpected {
		buf = append(buf, 1)
		buf = binary.BigEndian.AppendUint64(buf, expectedVersion)
	} else {
		buf = append(buf, 0)
	}
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(inner))) //nolint:gosec // inner is bounded well under 4G by OperateMaxOps/Rets/Bytes/Schema caps
	buf = append(buf, inner...)
	return buf, nil
}

// DecodeVectorOperateArgs reads args produced by EncodeVectorOperateArgs.
// Decoder discipline, in order, every read truncation-checked before it
// happens: colLen (an outer frame the encoder never produces with colLen==0,
// so a hostile/corrupt frame is the only source — ErrVectorArgsTruncated),
// col, id, pkLen (0 is ErrOperateArgs — a well-formed-but-empty declared
// name), payloadKey, casPresent (a value outside {0,1} is ErrOperateArgs),
// the optional 8-byte expectedVersion, then argsLen.
//
// argsLen is read as a raw uint32 and compared against the remaining length
// BEFORE any conversion to int: a value like 0xFFFFFFFF or 0x80000000 must
// never reach an int conversion first, because on a 32-bit build int is
// itself 32 bits and the conversion can produce a negative count that then
// slips past a naive "< 0" guard's sibling check. Comparing as uint64 up
// front is correct on every GOARCH and costs nothing extra on 64-bit.
//
// The sliced inner blob is handed to the already-hardened DecodeOperateArgs
// verbatim — this function adds no second parser for that shape. After it
// decodes, the same three post-conditions the encoder enforces are checked
// again (Key empty, TTLMode == OperateTTLKeep, TTL == 0), because a
// hand-built frame can carry an inner blob the encoder would have rejected.
// Finally a trailing-bytes check (off+argsLen != len(args)) rejects a frame
// with slack after the declared inner blob — it is not the frame it claims
// to be.
func DecodeVectorOperateArgs(args []byte) (collection string, id uint64, payloadKey string, a *OperateArgs, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 1 {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if colLen == 0 {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	off := 1
	if len(args)-off < colLen {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	collection = string(args[off : off+colLen])
	off += colLen

	if len(args)-off < 8 {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	id = binary.BigEndian.Uint64(args[off:])
	off += 8

	if len(args)-off < 2 {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	pkLen := int(binary.BigEndian.Uint16(args[off:]))
	off += 2
	if pkLen == 0 {
		return "", 0, "", nil, 0, false, ErrOperateArgs
	}
	if len(args)-off < pkLen {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	payloadKey = string(args[off : off+pkLen])
	off += pkLen

	if len(args)-off < 1 {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	casPresent := args[off]
	off++
	if casPresent > 1 {
		return "", 0, "", nil, 0, false, ErrOperateArgs
	}
	if casPresent == 1 {
		if len(args)-off < 8 {
			return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
		}
		expectedVersion = binary.BigEndian.Uint64(args[off:])
		hasExpected = true
		off += 8
	}

	if len(args)-off < 4 {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	rawArgsLen := binary.BigEndian.Uint32(args[off:])
	off += 4
	if uint64(rawArgsLen) > uint64(len(args)-off) {
		return "", 0, "", nil, 0, false, ErrVectorArgsTruncated
	}
	argsLen := int(rawArgsLen)

	inner, derr := DecodeOperateArgs(args[off : off+argsLen])
	if derr != nil {
		return "", 0, "", nil, 0, false, derr
	}
	if len(inner.Key) != 0 || inner.TTLMode != OperateTTLKeep || inner.TTL != 0 {
		return "", 0, "", nil, 0, false, ErrOperateArgs
	}
	off += argsLen
	if off != len(args) {
		return "", 0, "", nil, 0, false, ErrOperateArgs
	}

	return collection, id, payloadKey, inner, expectedVersion, hasExpected, nil
}

// EncodeVectorOperateResult encodes a vector_operate result frame.
// Wire: [found u8] | [1][resLen u32][operateResult][version u64]. found=false
// writes just the zero byte and r and version are ignored entirely — there is
// nothing else to say about a call that found no record, and an absent point has
// no version. found=true with r == nil is ErrOperateArgs: a caller claiming a
// result exists must supply one.
//
// version is the point's version AFTER the call: bumped when the op-list
// applied, and the CURRENT unbumped one when it was a deliberate no-op (a failed
// CHECK). It exists so a CAS loop can feed the next call's expectedVersion
// straight from this result instead of re-reading the point — the re-read is
// both a round trip and a race, since another writer can land between it and the
// retry.
//
// APPEND-ONLY. The version is written LAST, after the inner blob the previous
// shape ended with, and DecodeVectorOperateResult reads it only if it is there.
// A frame from before this field decodes as version 0. Any future field goes at
// the end too, read the same way: this frame has an exact trailing-bytes check,
// so a field inserted anywhere else is a wire break, not an extension.
func EncodeVectorOperateResult(found bool, r *OperateResult, version uint64) ([]byte, error) {
	if !found {
		return []byte{0}, nil
	}
	if r == nil {
		return nil, ErrOperateArgs
	}
	inner, err := EncodeOperateResult(r)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, 0, 1+4+len(inner)+8)
	buf = append(buf, 1)
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(inner))) //nolint:gosec // inner is bounded well under 4G by OperateMaxRet
	buf = append(buf, inner...)
	buf = binary.BigEndian.AppendUint64(buf, version)
	return buf, nil
}

// DecodeVectorOperateResult reads a frame produced by EncodeVectorOperateResult,
// with the same truncation and trailing-bytes discipline as
// DecodeVectorOperateArgs: found is read and validated against {0,1} first,
// resLen is a raw-uint32-before-int-conversion length check, the inner blob is
// handed to DecodeOperateResult verbatim, and a frame with bytes left over
// after the declared inner blob and the optional trailing version is
// ErrOperateArgs.
//
// The trailing version is the frame's ONE optional field and it is read the only
// way an append-only field can be: exactly 8 bytes after the inner blob, or
// nothing at all. Zero remaining bytes is a pre-version frame and yields
// version 0; any other remainder — 1..7 bytes, or 9+ — is a frame that is not
// what it claims to be, so it is rejected rather than silently truncated. That
// keeps the trailing-bytes guarantee the rest of this codec relies on.
func DecodeVectorOperateResult(b []byte) (found bool, r *OperateResult, version uint64, err error) {
	if len(b) < 1 {
		return false, nil, 0, ErrVectorArgsTruncated
	}
	flag := b[0]
	if flag > 1 {
		return false, nil, 0, ErrOperateArgs
	}
	if flag == 0 {
		if len(b) != 1 {
			return false, nil, 0, ErrOperateArgs
		}
		return false, nil, 0, nil
	}

	off := 1
	if len(b)-off < 4 {
		return false, nil, 0, ErrVectorArgsTruncated
	}
	rawResLen := binary.BigEndian.Uint32(b[off:])
	off += 4
	if uint64(rawResLen) > uint64(len(b)-off) {
		return false, nil, 0, ErrVectorArgsTruncated
	}
	resLen := int(rawResLen)

	res, derr := DecodeOperateResult(b[off : off+resLen])
	if derr != nil {
		return false, nil, 0, derr
	}
	off += resLen
	switch len(b) - off {
	case 0:
		return true, res, 0, nil // a frame written before the version field existed
	case 8:
		return true, res, binary.BigEndian.Uint64(b[off:]), nil
	default:
		return false, nil, 0, ErrOperateArgs
	}
}
