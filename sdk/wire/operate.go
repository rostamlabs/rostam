// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"math"
	"time"
)

// ErrOperateArgs marks an operate call's parameters as structurally invalid:
// an out-of-range ttlMode/create byte, an out-of-range path kind or segment
// kind byte, or a Row/Col path with an empty key. It is returned both by
// EncodeOperateArgs (a caller building an invalid call) and by
// DecodeOperateArgs (a wire frame carrying one) — see design doc §3.5.
var ErrOperateArgs = errors.New("wire: operate call arguments invalid")

// IsOperateArgsMessage reports whether s is the EXACT serialised form of an
// ErrOperateArgs error — the same exact-form matching ops.IsVectorRecordAbsent-
// Message and vector.IsRecordTooLargeMessage do for their sentinels, and for the
// same reason: an operate handler decodes its frame INSIDE the FSM apply, so on
// a cluster the error comes back through shard.decodePBResult rebuilt with
// errors.New, errors.Is identity is gone, and a sentinel-only classifier
// redacts a client protocol mistake to "internal error".
//
// This sentinel has exactly ONE serialised shape: bare, no detail suffix and no
// wrapper. Every server-side producer returns it unadorned — the wire decoders
// and encoders in this package (DecodeOperateArgs, DecodeVectorOperateArgs and
// their encoders) and ops (operate_apply.go's three guards,
// checkVectorOperateArgs) all `return ..., ErrOperateArgs`. So exact equality is
// the whole matcher, and it is deliberately not a strings.Contains: a bare
// substring check would make any internal fault that merely mentions the
// sentinel text client-facing.
//
// The ONE %w wrap in the tree, client.DecodeOperateValue's "N bytes left after
// the tagged cell", is raised in the CLIENT process on a reply it is decoding.
// It never reaches a server classifier, so it is correctly not matched here; if
// a future server-side site does wrap the sentinel with context, add that exact
// shape here, anchored, rather than loosening this to a substring.
func IsOperateArgsMessage(s string) bool {
	return s == ErrOperateArgs.Error()
}

// OperateSeg addresses one hop of an OperatePath (design doc §2.4/§3.5): a
// field or column identified by its schema position, or, for a record that
// stores names, by name.
type OperateSeg struct {
	ByName bool
	Pos    uint32
	Name   string
}

// OperatePath addresses a node inside an operate record: the record itself,
// one of its fields, a row of a table field (by key), or one column of that
// row (design doc §2.4).
type OperatePath struct {
	Kind  uint8
	Field OperateSeg
	Key   []byte
	Col   OperateSeg
}

// OperateOp is one instruction in a call's op list: a scalar/table op, a
// control op (IF/CHECK), or a MIGRATE/CONFIG/TRIM (design doc §3.1-§3.3).
type OperateOp struct {
	Opcode, Type, Aux uint8
	Path              OperatePath
	A, B              int64
	Bytes             []byte
}

// OperateRet is one return spec, evaluated after all ops apply, or against
// the original record on CHECK_FAILED (design doc §3.4).
type OperateRet struct {
	Mode uint8
	Path OperatePath
}

// OperateArgs is a decoded operate call (design doc §3.5).
type OperateArgs struct {
	Key     []byte
	TTL     time.Duration
	TTLMode uint8
	Create  uint8
	Schema  []byte
	Ops     []OperateOp
	Rets    []OperateRet
}

// OperateResult is a decoded operate result frame (design doc §3.4). Values
// holds one tagged-encoded node per return spec, in the same order; an
// absent node is []byte{OperateTypeUnset}.
type OperateResult struct {
	Status   uint8
	FailedOp uint16
	Values   [][]byte
}

// appendSeg appends one fseg/cseg (design doc §3.5): [0][pos uvarint] when
// addressed by schema position, or [1][len u8][name] when addressed by name.
func appendSeg(buf []byte, seg OperateSeg) ([]byte, error) {
	if seg.ByName {
		if len(seg.Name) > OperateMaxNameLen {
			return nil, ErrOperateCap
		}
		buf = append(buf, 1, byte(len(seg.Name))) //nolint:gosec // bounded by the check above
		buf = append(buf, seg.Name...)
		return buf, nil
	}
	buf = append(buf, 0)
	buf = binary.AppendUvarint(buf, uint64(seg.Pos))
	return buf, nil
}

// decodeSeg reads one fseg/cseg from the front of b, returning the segment
// and the number of bytes consumed. An unknown seg-kind byte, or a position
// past the uint32 an OperateSeg addresses with, is ErrOperateArgs — the
// frame is complete and well-formed, its content is out of range, which is
// what ErrOperateArgs means. A truncated frame is ErrShortArgs.
func decodeSeg(b []byte) (OperateSeg, int, error) {
	if len(b) < 1 {
		return OperateSeg{}, 0, ErrShortArgs
	}
	switch b[0] {
	case 0:
		v, n := binary.Uvarint(b[1:])
		if n <= 0 {
			return OperateSeg{}, 0, ErrShortArgs
		}
		if v > math.MaxUint32 {
			return OperateSeg{}, 0, ErrOperateArgs
		}
		return OperateSeg{Pos: uint32(v)}, 1 + n, nil
	case 1:
		if len(b) < 2 {
			return OperateSeg{}, 0, ErrShortArgs
		}
		nlen := int(b[1])
		if len(b)-2 < nlen {
			return OperateSeg{}, 0, ErrShortArgs
		}
		return OperateSeg{ByName: true, Name: string(b[2 : 2+nlen])}, 2 + nlen, nil
	default:
		return OperateSeg{}, 0, ErrOperateArgs
	}
}

// appendPath appends [kind u8] and, per kind, the fseg/kseg/cseg that follow
// it (design doc §2.4/§3.5). A Row or Col path requires a non-empty key.
func appendPath(buf []byte, p OperatePath) ([]byte, error) {
	if p.Kind > OperatePathCol {
		return nil, ErrOperateArgs
	}
	buf = append(buf, p.Kind)
	if p.Kind == OperatePathRecord {
		return buf, nil
	}
	var err error
	buf, err = appendSeg(buf, p.Field)
	if err != nil {
		return nil, err
	}
	if p.Kind == OperatePathField {
		return buf, nil
	}
	if len(p.Key) == 0 {
		return nil, ErrOperateArgs
	}
	if len(p.Key) > OperateMaxKeyLen {
		return nil, ErrOperateCap
	}
	buf = append(buf, byte(len(p.Key))) //nolint:gosec // bounded by the check above
	buf = append(buf, p.Key...)
	if p.Kind == OperatePathRow {
		return buf, nil
	}
	buf, err = appendSeg(buf, p.Col)
	if err != nil {
		return nil, err
	}
	return buf, nil
}

// decodePath reads one path from the front of b, returning the path and the
// number of bytes consumed. An out-of-range kind byte, or a Row/Col path
// with an empty key, is ErrOperateArgs; a truncated frame is ErrShortArgs.
func decodePath(b []byte) (OperatePath, int, error) {
	if len(b) < 1 {
		return OperatePath{}, 0, ErrShortArgs
	}
	kind := b[0]
	if kind > OperatePathCol {
		return OperatePath{}, 0, ErrOperateArgs
	}
	off := 1
	if kind == OperatePathRecord {
		return OperatePath{Kind: kind}, off, nil
	}
	fseg, n, err := decodeSeg(b[off:])
	if err != nil {
		return OperatePath{}, 0, err
	}
	off += n
	p := OperatePath{Kind: kind, Field: fseg}
	if kind == OperatePathField {
		return p, off, nil
	}
	if len(b)-off < 1 {
		return OperatePath{}, 0, ErrShortArgs
	}
	klen := int(b[off])
	off++
	if len(b)-off < klen {
		return OperatePath{}, 0, ErrShortArgs
	}
	if klen == 0 {
		return OperatePath{}, 0, ErrOperateArgs
	}
	p.Key = b[off : off+klen]
	off += klen
	if kind == OperatePathRow {
		return p, off, nil
	}
	cseg, n2, err := decodeSeg(b[off:])
	if err != nil {
		return OperatePath{}, 0, err
	}
	off += n2
	p.Col = cseg
	return p, off, nil
}

// appendOp appends one op: [opcode u8][type u8][aux u8][path][a i64][b
// i64][blen u16][bytes] (design doc §3.5).
func appendOp(buf []byte, op OperateOp) ([]byte, error) {
	if len(op.Bytes) > OperateMaxBytesLen {
		return nil, ErrOperateCap
	}
	buf = append(buf, op.Opcode, op.Type, op.Aux)
	var err error
	buf, err = appendPath(buf, op.Path)
	if err != nil {
		return nil, err
	}
	buf = binary.BigEndian.AppendUint64(buf, uint64(op.A))          //nolint:gosec // reinterpret i64 as u64 for binary write
	buf = binary.BigEndian.AppendUint64(buf, uint64(op.B))          //nolint:gosec // reinterpret i64 as u64 for binary write
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(op.Bytes))) //nolint:gosec // bounded by the check above
	buf = append(buf, op.Bytes...)
	return buf, nil
}

// decodeOp reads one op from the front of b, returning the op and the
// number of bytes consumed. op.Bytes and op.Path.Key/Name may alias b:
// callers only read them during apply, never retain them past the call.
func decodeOp(b []byte) (OperateOp, int, error) {
	if len(b) < 3 {
		return OperateOp{}, 0, ErrShortArgs
	}
	opcode, typ, aux := b[0], b[1], b[2]
	off := 3
	path, n, err := decodePath(b[off:])
	if err != nil {
		return OperateOp{}, 0, err
	}
	off += n
	if len(b)-off < 8+8+2 { // a(8) + b(8) + blen(2)
		return OperateOp{}, 0, ErrShortArgs
	}
	a := int64(binary.BigEndian.Uint64(b[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64 for binary read
	off += 8
	bv := int64(binary.BigEndian.Uint64(b[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64 for binary read
	off += 8
	blen := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	if len(b)-off < blen {
		return OperateOp{}, 0, ErrShortArgs
	}
	var opBytes []byte
	if blen > 0 {
		opBytes = b[off : off+blen]
	}
	off += blen
	return OperateOp{Opcode: opcode, Type: typ, Aux: aux, Path: path, A: a, B: bv, Bytes: opBytes}, off, nil
}

// appendRet appends one ret: [mode u8][path] (design doc §3.4/§3.5). A mode
// outside OperateRet* is ErrOperateArgs: the apply engine has no branch for
// one, so encoding it would ship a frame that can only be rejected server
// side.
func appendRet(buf []byte, ret OperateRet) ([]byte, error) {
	if ret.Mode > OperateRetCount {
		return nil, ErrOperateArgs
	}
	buf = append(buf, ret.Mode)
	return appendPath(buf, ret.Path)
}

// decodeRet reads one ret from the front of b, returning the ret and the
// number of bytes consumed. A mode byte outside OperateRet* is
// ErrOperateArgs, checked here rather than left to the apply engine, so a
// decoded OperateArgs always re-encodes (the FuzzDecodeOperateArgs identity)
// and every OperateRet a caller sees is one it has a branch for.
func decodeRet(b []byte) (OperateRet, int, error) {
	if len(b) < 1 {
		return OperateRet{}, 0, ErrShortArgs
	}
	if b[0] > OperateRetCount {
		return OperateRet{}, 0, ErrOperateArgs
	}
	path, n, err := decodePath(b[1:])
	if err != nil {
		return OperateRet{}, 0, err
	}
	return OperateRet{Mode: b[0], Path: path}, 1 + n, nil
}

// minOpBytes is the smallest an encoded op can be: a record-path op with no
// operand bytes (opcode+type+aux(3) + path kind byte(1) + a+b(16) +
// blen(2) = 22). DecodeOperateArgs bounds the declared op count against it
// before allocating (v1 discipline: CountFitsIn before any reservation).
const minOpBytes = 1 + 1 + 1 + 1 + 8 + 8 + 2

// minRetBytes is the smallest an encoded ret can be: a record-path ret
// (mode(1) + path kind byte(1) = 2).
const minRetBytes = 1 + 1

// EncodeOperateArgs encodes an operate call (design doc §3.5):
// [keyLen u16][key][ttlMs u64][ttlMode u8][create u8][schemaLen
// u16][schema][nOps u16]{op}*[nRet u16]{ret}*. It returns an error, rather
// than silently truncating or wrapping, for any field that cannot be
// represented on the wire or that exceeds a cap (design doc §2.7).
func EncodeOperateArgs(a *OperateArgs) ([]byte, error) {
	return AppendOperateArgs(nil, a)
}

// AppendOperateArgs is EncodeOperateArgs appending into dst (reusing its
// capacity when large enough), for a hot-loop caller that pools the buffer —
// the same pair EncodeKeyArgs/AppendKeyArgs already form for point ops.
// Passing dst=nil reproduces EncodeOperateArgs's bytes exactly. The returned
// slice may alias dst.
//
// A caller that pools dst across calls allocates nothing here once the buffer
// has grown to the largest op list it builds: an operate call's encoding is
// otherwise dominated by appendOp growing a fresh buffer per call.
func AppendOperateArgs(dst []byte, a *OperateArgs) ([]byte, error) {
	if len(a.Key) > 0xFFFF {
		return nil, ErrOperateCap
	}
	if len(a.Schema) > OperateMaxSchemaBytes {
		return nil, ErrOperateCap
	}
	if len(a.Ops) > OperateMaxOps {
		return nil, ErrOperateCap
	}
	if len(a.Rets) > OperateMaxRet {
		return nil, ErrOperateCap
	}
	if a.TTLMode > OperateTTLCreateOnly {
		return nil, ErrOperateArgs
	}
	if a.Create > OperateCreateDynamic {
		return nil, ErrOperateArgs
	}
	// ttlMs rides the wire as an unsigned 64-bit count of milliseconds, so a
	// negative duration has no encoding: converting it would wrap to a
	// ~584-million-year TTL rather than the expiry the caller asked for.
	if a.TTL < 0 {
		return nil, ErrOperateArgs
	}

	n := 2 + len(a.Key) + 8 + 1 + 1 + 2 + len(a.Schema) + 2 + 2
	buf := dst[:0]
	if cap(buf) < n {
		buf = make([]byte, 0, n)
	}
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(a.Key))) //nolint:gosec // bounded by the check above
	buf = append(buf, a.Key...)
	buf = binary.BigEndian.AppendUint64(buf, uint64(a.TTL/time.Millisecond)) //nolint:gosec // duration to milliseconds always non-negative
	buf = append(buf, a.TTLMode, a.Create)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(a.Schema))) //nolint:gosec // bounded by the check above
	buf = append(buf, a.Schema...)

	buf = binary.BigEndian.AppendUint16(buf, uint16(len(a.Ops))) //nolint:gosec // bounded by the check above
	for _, op := range a.Ops {
		var err error
		buf, err = appendOp(buf, op)
		if err != nil {
			return nil, err
		}
	}
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(a.Rets))) //nolint:gosec // bounded by the check above
	for _, ret := range a.Rets {
		var err error
		buf, err = appendRet(buf, ret)
		if err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// DecodeOperateArgs reads args produced by EncodeOperateArgs, applying v1
// decode discipline throughout: every read is truncation-checked before it
// happens, and every declared count (nOps, nRet) is bounded against its cap
// (ErrOperateCap) and with CountFitsIn against the smallest honest encoding
// of that kind of element (ErrShortArgs), BEFORE any slice sized by it is
// allocated. The two bounds report different errors because they mean
// different things: over the cap is a call asking for more than the protocol
// allows, over the byte budget is a truncated or lying frame. Decoded
// Key/Schema/Bytes/Name/path-Key fields may alias b: the apply path only
// reads them during one call and never retains them past it.
//
// TRAILING BYTES ARE REJECTED. A frame that decodes and leaves bytes over is not
// the frame it claims to be, and the slack is not inert: the vector families
// carry this blob inside a length-declared envelope
// (DecodeVectorOperateArgs's [argsLen u32][args...]), so a caller could declare
// a longer inner blob than it actually wrote and smuggle bytes past every
// structural check into a frame the server otherwise accepted. Following
// raft/logstore's decodeInto, the check is exact consumption rather than a
// bound, and it closes the KV operate path (where args is the whole request
// body) at the same time.
func DecodeOperateArgs(b []byte) (*OperateArgs, error) {
	a := new(OperateArgs)
	if err := DecodeOperateArgsInto(a, b); err != nil {
		return nil, err
	}
	return a, nil
}

// DecodeOperateArgsInto is DecodeOperateArgs decoding into dst, reusing the
// capacity of dst.Ops and dst.Rets instead of allocating a fresh slice per
// call. It is for a hot-path caller that pools the struct: the server decodes
// one OperateArgs per operate request, and that op slice is the single largest
// allocation on the apply path (an op list of n bidders x ~4 ops per bidder).
//
// Every field of dst is overwritten, so a pooled dst carries nothing from its
// previous use. A dst that already has capacity KEEPS it, so a frame carrying
// no ops decodes to an empty-but-non-nil dst.Ops where DecodeOperateArgs
// would return nil - read len(dst.Ops), never dst.Ops == nil. A fresh dst
// (nil slices) decodes identically to DecodeOperateArgs.
//
// On error dst is reset to empty rather than left untouched - see the error
// branch for why.
//
// A path addressed by NAME still allocates that name's string per segment;
// schema mode addresses by position and so decodes with no allocation at all
// once dst has grown.
//
// The same aliasing rule applies as for DecodeOperateArgs: Key, Schema, Bytes,
// Name and path-Key fields may point into b, so dst must not outlive b.
func DecodeOperateArgsInto(dst *OperateArgs, b []byte) error {
	n, err := decodeOperateArgsN(dst, b)
	if err == nil && n != len(b) {
		err = ErrOperateArgs
	}
	if err != nil {
		// dst is left EMPTY, not untouched: the decoder appends into dst's
		// backing array as it goes, so a call that fails partway has already
		// overwritten elements that dst's old length still covers. Resetting
		// is the only cheap state a caller can rely on, and clearing releases
		// the failed call's buffer.
		ops, rets := dst.Ops[:0], dst.Rets[:0]
		clear(ops[:cap(ops)])
		clear(rets[:cap(rets)])
		*dst = OperateArgs{Ops: ops, Rets: rets}
		return err
	}
	return nil
}

// decodeOperateArgsN is DecodeOperateArgs' body, reporting how many bytes it
// consumed so the exported wrapper can insist that be all of them.
func decodeOperateArgsN(dst *OperateArgs, b []byte) (int, error) {
	if len(b) < 2 {
		return 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(b[0:2]))
	off := 2
	if len(b)-off < klen {
		return 0, ErrShortArgs
	}
	var key []byte
	if klen > 0 {
		key = b[off : off+klen]
	}
	off += klen

	if len(b)-off < 8+1+1+2 { // ttlMs(8) + ttlMode(1) + create(1) + schemaLen(2)
		return 0, ErrShortArgs
	}
	ttl, err := ttlFromMs(binary.BigEndian.Uint64(b[off : off+8]))
	if err != nil {
		return 0, err
	}
	off += 8
	ttlMode := b[off]
	off++
	create := b[off]
	off++
	if ttlMode > OperateTTLCreateOnly {
		return 0, ErrOperateArgs
	}
	if create > OperateCreateDynamic {
		return 0, ErrOperateArgs
	}

	schemaLen := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	if len(b)-off < schemaLen {
		return 0, ErrShortArgs
	}
	var schema []byte
	if schemaLen > 0 {
		schema = b[off : off+schemaLen]
	}
	off += schemaLen

	if len(b)-off < 2 {
		return 0, ErrShortArgs
	}
	nOps := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	// Two distinct failures, reported as two distinct errors: a count over
	// the design doc §2.7 cap is ErrOperateCap (the frame asked for more ops
	// than a call may carry), while a count the remaining bytes cannot
	// possibly hold is a truncated frame.
	if nOps > OperateMaxOps {
		return 0, ErrOperateCap
	}
	if !CountFitsIn(nOps, len(b)-off, minOpBytes) {
		return 0, ErrShortArgs
	}
	// dst.Ops[:0] on a nil slice is still nil, so a fresh dst keeps
	// DecodeOperateArgs's "nil when the frame carries none" shape while a
	// pooled one reuses whatever capacity it already had.
	oldOps := len(dst.Ops)
	ops := dst.Ops[:0]
	reusedOps := true
	if nOps > 0 {
		if cap(ops) < nOps {
			ops = make([]OperateOp, 0, nOps)
			reusedOps = false
		}
		for i := 0; i < nOps; i++ {
			op, n, oerr := decodeOp(b[off:])
			if oerr != nil {
				return 0, oerr
			}
			ops = append(ops, op)
			off += n
		}
	}

	if len(b)-off < 2 {
		return 0, ErrShortArgs
	}
	nRet := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	if nRet > OperateMaxRet {
		return 0, ErrOperateCap
	}
	if !CountFitsIn(nRet, len(b)-off, minRetBytes) {
		return 0, ErrShortArgs
	}
	oldRets := len(dst.Rets)
	rets := dst.Rets[:0]
	reusedRets := true
	if nRet > 0 {
		if cap(rets) < nRet {
			rets = make([]OperateRet, 0, nRet)
			reusedRets = false
		}
		for i := 0; i < nRet; i++ {
			ret, n, rerr := decodeRet(b[off:])
			if rerr != nil {
				return 0, rerr
			}
			rets = append(rets, ret)
			off += n
		}
	}

	// A reused slice that shrank still holds the previous call's ops past its
	// new length, and those keep that call's request buffer alive. Clearing
	// just the shrink delta is enough: by induction everything past the old
	// length was already cleared by the decode that shrank it.
	if reusedOps && len(ops) < oldOps {
		clear(ops[len(ops):oldOps])
	}
	if reusedRets && len(rets) < oldRets {
		clear(rets[len(rets):oldRets])
	}

	*dst = OperateArgs{
		Key:     key,
		TTL:     ttl,
		TTLMode: ttlMode,
		Create:  create,
		Schema:  schema,
		Ops:     ops,
		Rets:    rets,
	}
	return off, nil
}

// EncodeOperateResult encodes an operate result frame (design doc §3.4):
// [status u8][failedOp u16 iff status==OperateStatusCheckFailed][nRet
// u16]{[vlen u32][value]}*. failedOp is written only when the status is
// OperateStatusCheckFailed; every other status omits it entirely.
//
// It returns an error rather than truncating: nRet is a uint16 on the wire,
// so more than OperateMaxRet values could not be counted honestly, and a
// silent narrowing would ship a frame whose declared count does not match
// the values behind it. A status outside OperateStatus* is rejected for the
// same reason — DecodeOperateResult would not accept it back.
func EncodeOperateResult(r *OperateResult) ([]byte, error) {
	if len(r.Values) > OperateMaxRet {
		return nil, ErrOperateCap
	}
	if r.Status != OperateStatusOK && r.Status != OperateStatusCheckFailed {
		return nil, ErrOperateArgs
	}
	buf := make([]byte, 0, 1+2+2+len(r.Values)*4)
	buf = append(buf, r.Status)
	if r.Status == OperateStatusCheckFailed {
		buf = binary.BigEndian.AppendUint16(buf, r.FailedOp)
	}
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(r.Values))) //nolint:gosec // bounded by OperateMaxRet above
	for _, v := range r.Values {
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(v))) //nolint:gosec // bounded by the apply engine upstream
		buf = append(buf, v...)
	}
	return buf, nil
}

// DecodeOperateResult reads a frame produced by EncodeOperateResult, with
// the same truncation and bounded-count discipline as DecodeOperateArgs, and
// the same exact-consumption rule: a frame that decodes and leaves bytes over
// is rejected. DecodeVectorOperateResult carries this blob inside a declared
// [resLen u32], so without the rule a result frame could over-declare its inner
// length and carry bytes no decoder ever looks at. Decoded values may alias b.
func DecodeOperateResult(b []byte) (*OperateResult, error) {
	r, n, err := decodeOperateResultN(b)
	if err != nil {
		return nil, err
	}
	if n != len(b) {
		return nil, ErrOperateArgs
	}
	return r, nil
}

// decodeOperateResultN is DecodeOperateResult's body, reporting how many bytes
// it consumed so the exported wrapper can insist that be all of them.
func decodeOperateResultN(b []byte) (*OperateResult, int, error) {
	if len(b) < 1 {
		return nil, 0, ErrShortArgs
	}
	status := b[0]
	if status != OperateStatusOK && status != OperateStatusCheckFailed {
		// An unknown status byte changes the frame's own shape (only
		// CHECK_FAILED carries failedOp), so accepting one would mean
		// guessing at the layout of the bytes behind it.
		return nil, 0, ErrOperateArgs
	}
	off := 1
	var failedOp uint16
	if status == OperateStatusCheckFailed {
		if len(b)-off < 2 {
			return nil, 0, ErrShortArgs
		}
		failedOp = binary.BigEndian.Uint16(b[off : off+2])
		off += 2
	}

	if len(b)-off < 2 {
		return nil, 0, ErrShortArgs
	}
	nRet := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	// Split for the same reason DecodeOperateArgs splits its two: over the
	// cap is a frame declaring more values than a call may return,
	// over the byte budget is a truncated or lying frame.
	if nRet > OperateMaxRet {
		return nil, 0, ErrOperateCap
	}
	if !CountFitsIn(nRet, len(b)-off, 4) {
		return nil, 0, ErrShortArgs
	}
	var values [][]byte
	if nRet > 0 {
		values = make([][]byte, 0, nRet)
		for i := 0; i < nRet; i++ {
			if len(b)-off < 4 {
				return nil, 0, ErrShortArgs
			}
			vlen := int(binary.BigEndian.Uint32(b[off : off+4]))
			off += 4
			if vlen < 0 || len(b)-off < vlen {
				return nil, 0, ErrShortArgs
			}
			var val []byte
			if vlen > 0 {
				val = b[off : off+vlen]
			}
			values = append(values, val)
			off += vlen
		}
	}

	return &OperateResult{Status: status, FailedOp: failedOp, Values: values}, off, nil
}
