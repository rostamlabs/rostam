// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"time"
)

// operate is a generic atomic multi-field update op: the client sends a list of
// pure-integer ops (INCR/INCRF/SETMAX/SHIFTOR/HALVE_GRP) against one Rostam value
// modelled as a small globals array + a dynamic map of entry sub-records, and the
// server applies them all atomically under the shard write lock in one round-trip
// (no client CAS-retry). This file carries only the WIRE codec (args + result);
// the value model (decodeRec/encodeRec) and the arithmetic (applyOp) live in the
// ops package, which is the sole authority on opcode semantics.

// Operate opcodes (the op-list entry's opcode byte). The decoder passes the
// opcode through verbatim; the ops-package handler validates it and is the
// authority on its arithmetic.
const (
	OperateOpINCR     uint8 = 0 // f += arg (i64)
	OperateOpINCRF    uint8 = 1 // fixed-point add: f += arg (same integer add; distinct for app clarity)
	OperateOpSETMAX   uint8 = 2 // f = max(f, arg)
	OperateOpSHIFTOR  uint8 = 3 // f = (f << arg) | arg2  (arg = shift bits, arg2 = value)
	OperateOpHALVEGRP uint8 = 4 // if f[fieldIdx] >= arg, halve fields [fieldIdx, fieldIdx+arg2)
)

// Operate targets (the op / return-spec target byte). A target byte outside this
// set is a malformed frame (the decoder cannot know whether an entryKey follows),
// so the decoder rejects it rather than guessing.
const (
	OperateTargetGlobal uint8 = 0 // fieldIdx indexes the globals array
	OperateTargetEntry  uint8 = 1 // (entryKey, fieldIdx) indexes a field inside an entry sub-record
)

// ErrBadOperateTarget indicates an operate op or return spec whose target byte is
// neither Global nor Entry — a malformed frame, since the target byte determines
// whether an 8-byte entryKey follows.
var ErrBadOperateTarget = errors.New("wire: operate target byte invalid")

// OperateOp is one op-list entry. For a global target EntryKey is ignored; for an
// entry target it selects the sub-record. Arg/Arg2 carry the per-opcode operands
// (INCR/INCRF/SETMAX use only Arg; SHIFTOR uses Arg=shift, Arg2=value; HALVE_GRP
// uses Arg=threshold, Arg2=group length).
type OperateOp struct {
	Target   uint8
	EntryKey uint64
	FieldIdx uint16
	Opcode   uint8
	Arg      int64
	Arg2     int64
}

// OperateRet is one return spec: the field whose post-apply i64 value the op
// should read back (so a single round-trip can update and read like Aerospike's
// Operate returning the record). EntryKey is ignored for a global target.
type OperateRet struct {
	Target   uint8
	EntryKey uint64
	FieldIdx uint16
}

// operateOpWireLen is the encoded size of one op: [tgt u8][entryKey u64 iff
// entry][fieldIdx u16][opcode u8][arg i64][arg2 i64]. A global op omits the
// entryKey (20 bytes); an entry op includes it (28 bytes). operateMinOpBytes is
// the smallest honest per-op size, used to bound the declared nOps before any
// allocation (CountFitsIn), exactly like the batch decoders.
const (
	operateMinOpBytes  = 20 // global op: 1 + 2 + 1 + 8 + 8
	operateMinRetBytes = 3  // global return spec: 1 + 2
)

func operateOpWireLen(o OperateOp) int {
	n := operateMinOpBytes
	if o.Target == OperateTargetEntry {
		n += 8
	}
	return n
}

func operateRetWireLen(r OperateRet) int {
	n := operateMinRetBytes
	if r.Target == OperateTargetEntry {
		n += 8
	}
	return n
}

// EncodeOperateArgs encodes the operate args:
//
//	[keyLen u16][key][ttlMs u64][maxEntries u16][nOps u32]
//	  per op: [tgt u8][entryKey u64 iff tgt=1][fieldIdx u16][opcode u8][arg i64][arg2 i64]
//	[nReturn u16][ per return: [tgt u8][entryKey u64 iff tgt=1][fieldIdx u16] ]
//
// maxEntries==0 means the entry map is unbounded; ttl==0 stores the record with
// no expiry (ttl is applied to the whole record on every call, like put).
func EncodeOperateArgs(key []byte, ttl time.Duration, maxEntries uint16, ops []OperateOp, ret []OperateRet) []byte {
	total := 2 + len(key) + 8 + 2 + 4
	for _, o := range ops {
		total += operateOpWireLen(o)
	}
	total += 2
	for _, r := range ret {
		total += operateRetWireLen(r)
	}
	buf := make([]byte, 0, total)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(key))) //nolint:gosec // bounded by upstream key length limits
	buf = append(buf, key...)
	buf = binary.BigEndian.AppendUint64(buf, uint64(ttl/time.Millisecond)) //nolint:gosec // duration to milliseconds always positive
	buf = binary.BigEndian.AppendUint16(buf, maxEntries)
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(ops))) //nolint:gosec // caller-supplied op count
	for _, o := range ops {
		buf = append(buf, o.Target)
		if o.Target == OperateTargetEntry {
			buf = binary.BigEndian.AppendUint64(buf, o.EntryKey)
		}
		buf = binary.BigEndian.AppendUint16(buf, o.FieldIdx)
		buf = append(buf, o.Opcode)
		buf = binary.BigEndian.AppendUint64(buf, uint64(o.Arg))  //nolint:gosec // reinterpret i64 as u64 for binary write
		buf = binary.BigEndian.AppendUint64(buf, uint64(o.Arg2)) //nolint:gosec // reinterpret i64 as u64 for binary write
	}
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(ret))) //nolint:gosec // caller-supplied return count (<= 65535)
	for _, r := range ret {
		buf = append(buf, r.Target)
		if r.Target == OperateTargetEntry {
			buf = binary.BigEndian.AppendUint64(buf, r.EntryKey)
		}
		buf = binary.BigEndian.AppendUint16(buf, r.FieldIdx)
	}
	return buf
}

// DecodeOperateArgs reads args produced by EncodeOperateArgs. Every length/count
// is bounds-checked BEFORE use in the 32-bit-safe `len(args)-off < need` form:
// nOps (a u32, the real overflow hazard) is bounded by CountFitsIn against the
// smallest honest op size before the ops slice is reserved, so a hostile count
// cannot drive an out-of-memory reservation; each op/return spec is then read
// with per-field truncation checks. A malformed frame returns an error — never a
// panic or an over-read (see ops.TestNoDecoderPanicsOnHostileBytes). ttlMs is
// rejected via ttlFromMs if it would overflow the time.Duration.
func DecodeOperateArgs(args []byte) (key []byte, ttl time.Duration, maxEntries uint16, ops []OperateOp, ret []OperateRet, err error) {
	if len(args) < 2 {
		return nil, 0, 0, nil, nil, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	off := 2
	if len(args)-off < klen+8+2+4 { // key + ttlMs(8) + maxEntries(2) + nOps(4)
		return nil, 0, 0, nil, nil, ErrShortArgs
	}
	key = args[off : off+klen]
	off += klen
	ttl, err = ttlFromMs(binary.BigEndian.Uint64(args[off : off+8]))
	if err != nil {
		return nil, 0, 0, nil, nil, err
	}
	off += 8
	maxEntries = binary.BigEndian.Uint16(args[off : off+2])
	off += 2
	nOps := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	// Bound nOps before reserving: the smallest honest op is operateMinOpBytes, so a
	// count above remaining/that cannot be honest (and CountFitsIn also rejects the
	// 32-bit-negative widening of the u32).
	if !CountFitsIn(nOps, len(args)-off, operateMinOpBytes) {
		return nil, 0, 0, nil, nil, ErrShortArgs
	}
	ops = make([]OperateOp, 0, nOps)
	for range nOps {
		var o OperateOp
		if len(args)-off < 1 {
			return nil, 0, 0, nil, nil, ErrShortArgs
		}
		o.Target = args[off]
		off++
		switch o.Target {
		case OperateTargetGlobal:
			// no entryKey
		case OperateTargetEntry:
			if len(args)-off < 8 {
				return nil, 0, 0, nil, nil, ErrShortArgs
			}
			o.EntryKey = binary.BigEndian.Uint64(args[off : off+8])
			off += 8
		default:
			return nil, 0, 0, nil, nil, ErrBadOperateTarget
		}
		if len(args)-off < 2+1+8+8 { // fieldIdx(2) + opcode(1) + arg(8) + arg2(8)
			return nil, 0, 0, nil, nil, ErrShortArgs
		}
		o.FieldIdx = binary.BigEndian.Uint16(args[off : off+2])
		off += 2
		o.Opcode = args[off]
		off++
		o.Arg = int64(binary.BigEndian.Uint64(args[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64
		off += 8
		o.Arg2 = int64(binary.BigEndian.Uint64(args[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64
		off += 8
		ops = append(ops, o)
	}
	if len(args)-off < 2 {
		return nil, 0, 0, nil, nil, ErrShortArgs
	}
	nRet := int(binary.BigEndian.Uint16(args[off : off+2]))
	off += 2
	// nRet is a u16 (<= 65535) so it cannot widen negative, but bound it anyway
	// against the smallest honest return-spec size before reserving.
	if !CountFitsIn(nRet, len(args)-off, operateMinRetBytes) {
		return nil, 0, 0, nil, nil, ErrShortArgs
	}
	ret = make([]OperateRet, 0, nRet)
	for range nRet {
		var r OperateRet
		if len(args)-off < 1 {
			return nil, 0, 0, nil, nil, ErrShortArgs
		}
		r.Target = args[off]
		off++
		switch r.Target {
		case OperateTargetGlobal:
			// no entryKey
		case OperateTargetEntry:
			if len(args)-off < 8 {
				return nil, 0, 0, nil, nil, ErrShortArgs
			}
			r.EntryKey = binary.BigEndian.Uint64(args[off : off+8])
			off += 8
		default:
			return nil, 0, 0, nil, nil, ErrBadOperateTarget
		}
		if len(args)-off < 2 {
			return nil, 0, 0, nil, nil, ErrShortArgs
		}
		r.FieldIdx = binary.BigEndian.Uint16(args[off : off+2])
		off += 2
		ret = append(ret, r)
	}
	return key, ttl, maxEntries, ops, ret, nil
}

// EncodeOperateResult encodes the operate result: the requested field values as
// i64 BE in request order (one 8-byte slot per return spec).
func EncodeOperateResult(vals []int64) []byte {
	buf := make([]byte, 0, len(vals)*8)
	for _, v := range vals {
		buf = binary.BigEndian.AppendUint64(buf, uint64(v)) //nolint:gosec // reinterpret i64 as u64 for binary write
	}
	return buf
}

// DecodeOperateResult reads a result produced by EncodeOperateResult back into a
// slice of i64 (one per return spec). The result must be a whole number of 8-byte
// slots.
func DecodeOperateResult(b []byte) ([]int64, error) {
	if len(b)%8 != 0 {
		return nil, ErrShortArgs
	}
	out := make([]int64, len(b)/8)
	for i := range out {
		out[i] = int64(binary.BigEndian.Uint64(b[i*8 : i*8+8])) //nolint:gosec // reinterpret stored u64 as i64
	}
	return out, nil
}
