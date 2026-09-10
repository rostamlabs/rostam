// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// ErrVectorArgsTruncated is returned by Decode* when the args bytes are
// shorter than the layout requires.
var ErrVectorArgsTruncated = errors.New("ops: vector args truncated")

// IsVectorArgsTruncatedMessage reports whether s is the EXACT serialised form of
// an ErrVectorArgsTruncated error. It is IsOperateArgsMessage's twin, for the
// same reason and with the same shape: the operate handlers decode their frame
// INSIDE the FSM apply, so on a cluster shard.decodePBResult rebuilds the error
// with errors.New, errors.Is identity is gone, and a classifier matching only
// the sentinel redacts a caller's framing mistake to "internal error".
//
// The two sentinels are raised by the SAME decoder, side by side:
// DecodeVectorOperateArgs answers ErrVectorArgsTruncated for a frame shorter
// than its declared fields and ErrOperateArgs for a frame that is complete but
// structurally wrong. Both are the caller's mistake, so both belong in the same
// classification bucket on every transport.
//
// This sentinel has exactly ONE serialised shape as a server classifier sees it:
// bare, no detail suffix and no wrapper. Every producer in sdk/wire returns it
// unadorned, and no SERVER-SIDE site wraps it with context — the %w wraps that
// exist are in tests, constructing the shape a wrapped error would have so the
// negative controls can prove it is declined. So exact equality is the whole
// matcher, and it is deliberately not a strings.Contains: a bare substring check
// would make any internal fault that merely mentions the sentinel text
// client-facing. If a future server-side site does wrap it, add that exact
// shape here, anchored, rather than loosening this to a substring — the same
// rule IsOperateArgsMessage states for its own sentinel.
func IsVectorArgsTruncatedMessage(s string) bool {
	return s == ErrVectorArgsTruncated.Error()
}

// ErrMalformedPayloadJSON marks a per-point payload blob that is well-FRAMED —
// its length prefix is consistent with the body, so the transport streamed it
// through without looking inside — but is not a metadata object.
//
// It is exported and wrapped rather than left anonymous because of WHERE the
// mistake is caught. On the inline batch route the transport unmarshals each
// payload itself and answers 400 directly. The bulk staging route deliberately
// does NOT: it streams the payload section off the socket straight into the op
// args with no per-point JSON round-trip, which is most of why it is fast. The
// first thing to look inside a blob is therefore the op decoder, on the far side
// of the dispatcher — and without a sentinel a caller's typo comes back as a
// redacted 500. A malformed payload is a client mistake with an obvious remedy,
// on both routes.
var ErrMalformedPayloadJSON = errors.New("ops: malformed per-point payload JSON")

// CountFitsIn reports whether a DECLARED element count read off the wire is
// representable by the bytes that remain. Every element of a given kind costs at
// least minBytesPerElem on the wire — a floor derived from the ENCODER, taking the
// smallest form each element can be written in — so a count above
// remaining/minBytesPerElem cannot be honest and the frame is malformed.
//
// Call it BEFORE any reservation sized by that count (make, slices.Grow, a map
// size hint). The per-element truncation checks inside a decode loop cannot stand
// in for it: they run after the reservation has already been made. Sizing a
// reservation straight from an unvalidated u32 turns a corrupt or hostile frame
// into an out-of-memory abort, which is not an error a caller can reject the frame
// on — it takes the process down, from any transport that decodes that frame.
//
// minBytesPerElem must be >= 1; a zero floor would divide by zero, and an element
// that can legitimately occupy no bytes at all cannot be bounded this way (see
// DecodeMVGetResultAt, which handles its zero-width case explicitly).
//
// Exported because the HTTP binary bulk transport bounds its own declared counts
// the same way. It calls THIS function rather than restating the arithmetic: a
// hand-copied "same discipline" check in another package drifts silently, and the
// failure mode of a drifted bound is an OOM, not a test failure.
func CountFitsIn(n, remaining, minBytesPerElem int) bool {
	if n < 0 || remaining < 0 || minBytesPerElem < 1 {
		return false
	}
	return n <= remaining/minBytesPerElem
}

// Vector op arg flags (bit field at args[0]).
const (
	vecFlagTTL      uint8 = 1 << 0 // vector_insert: ttlMs present
	vecFlagMetadata uint8 = 1 << 1 // vector_insert: metadata JSON present
	vecFlagSparse   uint8 = 1 << 2 // vector_insert: sparse vector present
	// vecFlagVersion: vector_insert carries a trailing [version:u64] AND requests a
	// VERSION-PRESERVING reinsert (the handler restores the exact version verbatim
	// instead of bumping to 1). Used ONLY by the reshard/resplit backfill so copied
	// points keep their per-point CAS version. Additive: byte-identical to a plain
	// insert when unset.
	vecFlagVersion uint8 = 1 << 3
	// vecFlagExpectedVersion: vector_insert/upsert carries a trailing
	// [expectedVersion:u64] CAS PRECONDITION (the version the caller expects the
	// point to currently have; 0 = expect-absent). Distinct from vecFlagVersion,
	// which SETS the version verbatim for the reshard backfill — this one is the
	// compare-and-set guard the handler turns into a CASCond{Has:true}. Additive:
	// byte-identical to a plain insert when unset. The trailing field, when both
	// flags are present, follows the version field (version first, then expected).
	vecFlagExpectedVersion uint8 = 1 << 4
	// vecFlagKeyTTL: vector_insert/upsert carries a trailing
	// [keyTtlLen:u32][keyTtlJSON] block — a per-key payload TTL map (key ->
	// RELATIVE ms; the engine computes the ABSOLUTE deadline now+ttl at insert,
	// exactly like set_payload). It rides AFTER expectedVersion in flag order.
	// Additive: byte-identical to a plain insert when the map is empty (flag unset,
	// no trailing bytes).
	vecFlagKeyTTL uint8 = 1 << 5
	// vecFlagKeyExpiresAbs: vector_insert/insert_if_absent carries a trailing
	// [n:u32]{[kLen:u32][k][deadline:u64]×n} block of ABSOLUTE per-key payload TTL
	// deadlines (unix-millis), applied VERBATIM (NOT recomputed now+ttl). DISTINCT
	// from vecFlagKeyTTL (which is RELATIVE ms the engine turns into now+ttl). Used
	// ONLY by the reshard/resplit backfill so a copied point keeps its original
	// absolute key deadlines time-stable. It rides LAST in flag order (after the
	// relative keyTTL block). Additive: byte-identical to a plain insert when the
	// map is empty (flag unset, no trailing bytes).
	vecFlagKeyExpiresAbs uint8 = 1 << 6
	VecFlagFilter        uint8 = 1 << 0 // vector_search: filter JSON present
	vecFlagSearchOpts    uint8 = 1 << 1 // vector_search: consistency opts trailer present

	hybridFlagFilter uint8 = 1 << 0 // vector_hybrid_search: filter JSON present
	hybridFlagSparse uint8 = 1 << 1 // vector_hybrid_search: sparse query present
	hybridFlagOpts   uint8 = 1 << 2 // vector_hybrid_search: consistency opts trailer present

	// Get-op trailing-flags byte (shared by dense/named/MV get args). It rides at
	// the END of the get args ([colLen][col][id:u64][getFlags:u8]) so the leading
	// [colLen][col] layout stays compatible with VectorKeyColAt1 routing. The
	// default (both projections on) is the byte 0x03 — a caller wanting both
	// vector(s) AND payload sets both bits. The encoders always emit the flags byte
	// (never absent), so a missing trailing byte is a truncation error.
	GetFlagWithVector  uint8 = 1 << 0 // include the vector(s)/tokens in the result
	GetFlagWithPayload uint8 = 1 << 1 // include the payload (+ sparse, dense only) in the result
)

// GetFlagsBoth is the default get projection: include both the vector(s) and the
// payload. Callers pass it to Encode*GetArgs for the common "fetch everything" case.
const GetFlagsBoth = GetFlagWithVector | GetFlagWithPayload

// writeSparse appends [nnz:u32]{[dim:u32][value:f32]} to buf at off, returning
// the new offset. Caller sizes buf to include 4 + 8*nnz bytes.
func writeSparse(buf []byte, off int, sv vtypes.SparseVector) int {
	binary.BigEndian.PutUint32(buf[off:], uint32(len(sv.Indices)))
	off += 4
	for i, dim := range sv.Indices {
		binary.BigEndian.PutUint32(buf[off:], dim)
		off += 4
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(sv.Values[i]))
		off += 4
	}
	return off
}

// writeSparseAppend APPENDS [nnz:u32]{[dim:u32][value:f32]} to buf and returns the
// grown slice — the append-style counterpart of writeSparse (which writes into a
// pre-sized buffer at an offset). Used by the named insert sparse sub-block, which
// builds its output incrementally.
func writeSparseAppend(buf []byte, sv vtypes.SparseVector) []byte {
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(len(sv.Indices))) //nolint:gosec
	buf = append(buf, u32[:]...)
	for i, dim := range sv.Indices {
		binary.BigEndian.PutUint32(u32[:], dim)
		buf = append(buf, u32[:]...)
		binary.BigEndian.PutUint32(u32[:], math.Float32bits(sv.Values[i]))
		buf = append(buf, u32[:]...)
	}
	return buf
}

// readSparse decodes [nnz:u32]{[dim:u32][value:f32]} at args[off:], returning
// the sparse vector and the new offset. Returns ErrVectorArgsTruncated if short.
func readSparse(args []byte, off int) (vtypes.SparseVector, int, error) {
	if len(args) < off+4 {
		return vtypes.SparseVector{}, off, ErrVectorArgsTruncated
	}
	nnz := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// Each term costs 8 bytes ([dim:u32][value:f32]); the divide-form bound rejects a
	// widened-negative or absurd nnz before the makes (8*nnz cannot overflow here).
	if !CountFitsIn(nnz, len(args)-off, 8) {
		return vtypes.SparseVector{}, off, ErrVectorArgsTruncated
	}
	sv := vtypes.SparseVector{
		Indices: make([]uint32, nnz),
		Values:  make([]float32, nnz),
	}
	for i := 0; i < nnz; i++ {
		sv.Indices[i] = binary.BigEndian.Uint32(args[off:])
		off += 4
		sv.Values[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	return sv, off, nil
}

// EncodeVectorInsertArgs serializes vector_insert args with no TTL or metadata
// (flags=0). Backwards-compatible entry point for the simple insert path.
func EncodeVectorInsertArgs(collection string, id uint64, vec []float32) []byte {
	return EncodeVectorInsertArgsExt(collection, id, vec, 0, nil, vtypes.SparseVector{})
}

// EncodeVectorInsertArgsExt serializes vector_insert args, optionally carrying
// a TTL, metadata, and/or sparse vector. Wire:
//
//	[flags:u8][colLen:u8][col][id:u64][dim:u32][vec][?ttlMs:u64][?metaLen:u32][?metaJSON][?nnz:u32{dim,val}]
func EncodeVectorInsertArgsExt(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector) []byte {
	return encodeVectorInsertArgs(collection, id, vec, ttl, meta, sparse, 0, 0, false, nil, nil)
}

// EncodeVectorInsertArgsKeyTTL serializes vector_insert/upsert args carrying an
// OPTIONAL per-key payload TTL map (key -> RELATIVE ms). When keyTTLMs is
// empty/nil the output is BYTE-IDENTICAL to EncodeVectorInsertArgsExt (the
// vecFlagKeyTTL bit stays unset and no trailing bytes are appended). The engine
// turns each relative ms into an ABSOLUTE deadline now+ttl at insert, mirroring
// set_payload's per-key TTL. Shared by the dense insert and upsert wire paths.
func EncodeVectorInsertArgsKeyTTL(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, keyTTLMs map[string]int64) []byte {
	return encodeVectorInsertArgs(collection, id, vec, ttl, meta, sparse, 0, 0, false, keyTTLMs, nil)
}

// EncodeVectorInsertArgsCAS serializes vector_insert/upsert args carrying an
// optimistic-CAS precondition: the trailing [expectedVersion:u64] + the
// vecFlagExpectedVersion bit. The handler turns it into a CASCond{Expected:
// expectedVersion, Has:true}, so the mutation applies ONLY when the point's
// current version matches (expectedVersion 0 = expect-absent). When hasExpected
// is false the output is BYTE-IDENTICAL to EncodeVectorInsertArgsExt (no flag, no
// trailing field). Shared by the dense insert and upsert wire paths.
func EncodeVectorInsertArgsCAS(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, expectedVersion uint64, hasExpected bool) []byte {
	return encodeVectorInsertArgs(collection, id, vec, ttl, meta, sparse, 0, expectedVersion, hasExpected, nil, nil)
}

// EncodeVectorInsertArgsCASKeyTTL is EncodeVectorInsertArgsCAS plus an OPTIONAL
// per-key payload TTL map (key -> RELATIVE ms). The keyTTL block rides AFTER the
// expectedVersion field in flag order, so the two trailers coexist. When both
// keyTTLMs is empty AND hasExpected is false the output is BYTE-IDENTICAL to
// EncodeVectorInsertArgsExt. Shared by the dense insert and upsert wire paths.
func EncodeVectorInsertArgsCASKeyTTL(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64) []byte {
	return encodeVectorInsertArgs(collection, id, vec, ttl, meta, sparse, 0, expectedVersion, hasExpected, keyTTLMs, nil)
}

// EncodeVectorInsertArgsVersioned serializes vector_insert args carrying a
// VERSION-PRESERVING reinsert request: the trailing [version:u64] + vecFlagVersion
// so the handler restores the EXACT version (verbatim, not bumped). Used only by
// the reshard/resplit backfill so copied points keep their per-point CAS version.
// A version of 0 is byte-identical to EncodeVectorInsertArgsExt (no flag, no
// trailing field).
func EncodeVectorInsertArgsVersioned(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, version uint64) []byte {
	return encodeVectorInsertArgs(collection, id, vec, ttl, meta, sparse, version, 0, false, nil, nil)
}

// EncodeVectorInsertArgsVersionedKeyExpires is EncodeVectorInsertArgsVersioned
// plus an OPTIONAL ABSOLUTE per-key payload TTL map (key -> ABSOLUTE unix-millis
// deadline) carried by the trailing vecFlagKeyExpiresAbs block. Used ONLY by the
// dense reshard/resplit copy passes so a copied point keeps BOTH its per-point CAS
// version AND its original absolute key deadlines (restored VERBATIM, NOT
// recomputed now+ttl — DISTINCT from the relative vecFlagKeyTTL block). When both
// version==0 AND keyExpires is empty the output is BYTE-IDENTICAL to
// EncodeVectorInsertArgsExt (no flags, no trailing fields).
func EncodeVectorInsertArgsVersionedKeyExpires(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, version uint64, keyExpires map[string]uint64) []byte {
	return encodeVectorInsertArgs(collection, id, vec, ttl, meta, sparse, version, 0, false, nil, keyExpires)
}

func encodeVectorInsertArgs(collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, keyExpiresAbs map[string]uint64) []byte {
	var flags uint8
	var ttlMs uint64
	if ttl > 0 {
		flags |= vecFlagTTL
		ttlMs = uint64(ttl.Milliseconds())
	}
	var metaJSON []byte
	if len(meta) > 0 {
		flags |= vecFlagMetadata
		metaJSON, _ = json.Marshal(meta) // map[string]Value always marshals
	}
	if !sparse.IsZero() {
		flags |= vecFlagSparse
	}
	if version != 0 {
		flags |= vecFlagVersion
	}
	if hasExpected {
		flags |= vecFlagExpectedVersion
	}
	var keyTTLJSON []byte
	if len(keyTTLMs) > 0 {
		flags |= vecFlagKeyTTL
		keyTTLJSON, _ = json.Marshal(keyTTLMs)
	}
	if len(keyExpiresAbs) > 0 {
		flags |= vecFlagKeyExpiresAbs
	}

	n := 1 + 1 + len(collection) + 8 + 4 + 4*len(vec)
	if flags&vecFlagTTL != 0 {
		n += 8
	}
	if flags&vecFlagMetadata != 0 {
		n += 4 + len(metaJSON)
	}
	if flags&vecFlagSparse != 0 {
		n += 4 + 8*len(sparse.Indices)
	}
	if flags&vecFlagVersion != 0 {
		n += 8
	}
	if flags&vecFlagExpectedVersion != 0 {
		n += 8
	}
	if flags&vecFlagKeyTTL != 0 {
		n += 4 + len(keyTTLJSON)
	}
	if flags&vecFlagKeyExpiresAbs != 0 {
		n += 4 // n:u32
		for k := range keyExpiresAbs {
			n += 4 + len(k) + 8 // kLen:u32 + key + deadline:u64
		}
	}
	buf := make([]byte, n)
	buf[0] = flags
	buf[1] = byte(len(collection))
	off := 2 + copy(buf[2:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	off += 8
	binary.BigEndian.PutUint32(buf[off:], uint32(len(vec)))
	off += 4
	for _, f := range vec {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
		off += 4
	}
	if flags&vecFlagTTL != 0 {
		binary.BigEndian.PutUint64(buf[off:], ttlMs)
		off += 8
	}
	if flags&vecFlagMetadata != 0 {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metaJSON)))
		off += 4
		off += copy(buf[off:], metaJSON)
	}
	if flags&vecFlagSparse != 0 {
		off = writeSparse(buf, off, sparse)
	}
	if flags&vecFlagVersion != 0 {
		binary.BigEndian.PutUint64(buf[off:], version)
		off += 8
	}
	if flags&vecFlagExpectedVersion != 0 {
		binary.BigEndian.PutUint64(buf[off:], expectedVersion)
		off += 8
	}
	// Per-key payload TTL block, in flag order AFTER expectedVersion (so it coexists
	// with the CAS/version trailers). Emitted ONLY when the flag is set, so an empty
	// map appends nothing and the output stays byte-identical to a plain insert.
	if flags&vecFlagKeyTTL != 0 {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(keyTTLJSON)))
		off += 4
		off += copy(buf[off:], keyTTLJSON)
	}
	// ABSOLUTE per-key payload TTL block, LAST in flag order. Each deadline is an
	// ABSOLUTE unix-millis value written VERBATIM (NOT relative, NOT recomputed).
	// Emitted ONLY when the flag is set, so an empty map appends nothing and the
	// output stays byte-identical to a plain insert.
	if flags&vecFlagKeyExpiresAbs != 0 {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(keyExpiresAbs))) //nolint:gosec
		off += 4
		for k, deadline := range keyExpiresAbs {
			binary.BigEndian.PutUint32(buf[off:], uint32(len(k))) //nolint:gosec
			off += 4
			off += copy(buf[off:], k)
			binary.BigEndian.PutUint64(buf[off:], deadline)
			off += 8
		}
	}
	return buf
}

// DecodeVectorInsertArgs reads vector_insert args (flags byte + optional TTL,
// metadata, sparse, version, expected_version). sparse is nil when the sparse
// flag is unset. version is the version-preserving reinsert version (0 = none →
// a normal insert; only the reshard backfill sets it via
// EncodeVectorInsertArgsVersioned). hasExpected reports whether a CAS precondition
// trailer was present; expectedVersion is the precondition (0 = expect-absent)
// when hasExpected is true.
func DecodeVectorInsertArgs(args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, err error) {
	collection, id, vec, ttl, meta, sparse, version, _, _, err = DecodeVectorInsertArgsCAS(args)
	return collection, id, vec, ttl, meta, sparse, version, err
}

// DecodeVectorInsertArgsCAS is DecodeVectorInsertArgs plus the optimistic-CAS
// precondition: hasExpected reports whether a vecFlagExpectedVersion trailer was
// present, expectedVersion is its value (0 = expect-absent). When the flag is
// unset hasExpected is false and expectedVersion is 0 (a non-CAS write).
func DecodeVectorInsertArgsCAS(args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, err error) {
	return decodeVectorInsertArgsCASInto(nil, args)
}

// decodeVectorInsertArgsCASInto is DecodeVectorInsertArgsCAS that decodes the
// dense vector INTO the caller-owned scratch dst (passed as dst[:0]) when dst has
// the capacity, instead of allocating a fresh []float32 each call. dst == nil (or
// too small for dim) reproduces the allocating path byte-for-byte, so the public
// wrapper is unchanged. The single-insert handler passes a pooled buffer here and
// returns it to the pool once Insert has COPIED the vector into the arena.
func decodeVectorInsertArgsCASInto(dst []float32, args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 2 {
		return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	if len(args) < 2+colLen+8+4 {
		return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
	}
	collection = string(args[2 : 2+colLen])
	off := 2 + colLen
	id = binary.BigEndian.Uint64(args[off:])
	off += 8
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// Via CountFitsIn, not `len(args) < off+4*dim`, which this replaced: that
	// form is defeated TWICE over by a hostile dim. On a 32-bit build a declared
	// dim above MaxInt32 widens negative, and 4*MinInt32 overflows to exactly 0,
	// so the comparison reads `len(args) < off` and passes — after which dst[:dim]
	// panics on a negative bound. CountFitsIn rejects dim < 0 and does its bound
	// by DIVISION, which cannot overflow.
	if !CountFitsIn(dim, len(args)-off, 4) {
		return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
	}
	// Reuse the caller's scratch backing when it fits (zero-alloc on the pooled
	// single-insert path); fall back to a fresh allocation otherwise so dst == nil
	// is byte-identical to the legacy decode.
	if cap(dst) >= dim {
		vec = dst[:dim]
	} else {
		vec = make([]float32, dim)
	}
	for i := 0; i < dim; i++ {
		vec[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	if flags&vecFlagTTL != 0 {
		if len(args) < off+8 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
		}
		ttl = time.Duration(binary.BigEndian.Uint64(args[off:])) * time.Millisecond
		off += 8
	}
	if flags&vecFlagMetadata != 0 {
		if len(args) < off+4 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
		}
		mlen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if mlen < 0 || len(args)-off < mlen {
			return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
		}
		meta = make(vtypes.Metadata)
		if err := json.Unmarshal(args[off:off+mlen], &meta); err != nil {
			return "", 0, nil, 0, nil, nil, 0, 0, false, fmt.Errorf("ops: decode metadata: %w", err)
		}
		off += mlen
	}
	if flags&vecFlagSparse != 0 {
		sv, noff, serr := readSparse(args, off)
		if serr != nil {
			return "", 0, nil, 0, nil, nil, 0, 0, false, serr
		}
		sparse = &sv
		off = noff
	}
	if flags&vecFlagVersion != 0 {
		if len(args) < off+8 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
		}
		version = binary.BigEndian.Uint64(args[off:])
		off += 8
	}
	if flags&vecFlagExpectedVersion != 0 {
		if len(args) < off+8 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, ErrVectorArgsTruncated
		}
		expectedVersion = binary.BigEndian.Uint64(args[off:])
		hasExpected = true
	}
	return collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, nil
}

// DecodeVectorInsertArgsKeyTTL is DecodeVectorInsertArgsCAS plus the OPTIONAL
// per-key payload TTL map (key -> RELATIVE ms) carried by the trailing
// vecFlagKeyTTL block, which rides AFTER the expectedVersion field. keyTTLMs is
// nil when the flag is unset (a plain insert) — byte-identical to the legacy wire.
// A present-but-truncated block is fail-loud. The handlers use this decoder so an
// insert/upsert can set per-key payload TTLs at write time (the engine turns each
// relative ms into an absolute deadline now+ttl).
func DecodeVectorInsertArgsKeyTTL(args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, err error) {
	return decodeVectorInsertArgsKeyTTLInto(nil, args)
}

// decodeVectorInsertArgsKeyTTLInto is DecodeVectorInsertArgsKeyTTL that threads
// the caller's scratch dst down to the dense-vector decode (zero-alloc on reuse).
// dst == nil reproduces the public wrapper byte-for-byte.
func decodeVectorInsertArgsKeyTTLInto(dst []float32, args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, err error) {
	collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, err = decodeVectorInsertArgsCASInto(dst, args)
	if err != nil {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, err
	}
	if len(args) < 1 || args[0]&vecFlagKeyTTL == 0 {
		return collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, nil, nil
	}
	// Recompute the trailing offset to locate the keyTTL block (it sits last, after
	// every other flag-gated field). The fields before it were validated above.
	flags := args[0]
	colLen := int(args[1])
	off := 2 + colLen
	off += 8 // id
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	off += 4 * dim
	if flags&vecFlagTTL != 0 {
		off += 8
	}
	if flags&vecFlagMetadata != 0 {
		mlen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4 + mlen
	}
	if flags&vecFlagSparse != 0 {
		_, noff, serr := readSparse(args, off)
		if serr != nil {
			return "", 0, nil, 0, nil, nil, 0, 0, false, nil, serr
		}
		off = noff
	}
	if flags&vecFlagVersion != 0 {
		off += 8
	}
	if flags&vecFlagExpectedVersion != 0 {
		off += 8
	}
	if len(args) < off+4 {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, ErrVectorArgsTruncated
	}
	klen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if klen < 0 || len(args)-off < klen {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, ErrVectorArgsTruncated
	}
	km := make(map[string]int64)
	if uerr := json.Unmarshal(args[off:off+klen], &km); uerr != nil {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, fmt.Errorf("ops: decode key_ttl_ms: %w", uerr)
	}
	if len(km) > 0 {
		keyTTLMs = km
	}
	return collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, keyTTLMs, nil
}

// vectorInsertTrailerOffset returns the byte offset just AFTER the expectedVersion
// field — i.e. the start of the optional trailing blocks (relative keyTTL, then
// the absolute keyExpires). The leading fields must already be validated by a
// prior DecodeVectorInsertArgsCAS. It also returns the flags byte for convenience.
func vectorInsertTrailerOffset(args []byte) (flags uint8, off int) {
	flags = args[0]
	colLen := int(args[1])
	off = 2 + colLen
	off += 8 // id
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	off += 4 * dim
	if flags&vecFlagTTL != 0 {
		off += 8
	}
	if flags&vecFlagMetadata != 0 {
		mlen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4 + mlen
	}
	if flags&vecFlagSparse != 0 {
		_, noff, _ := readSparse(args, off)
		off = noff
	}
	if flags&vecFlagVersion != 0 {
		off += 8
	}
	if flags&vecFlagExpectedVersion != 0 {
		off += 8
	}
	return flags, off
}

// DecodeVectorInsertArgsKeyExpires is DecodeVectorInsertArgsKeyTTL plus the
// OPTIONAL ABSOLUTE per-key payload TTL map (key -> ABSOLUTE unix-millis deadline)
// carried by the trailing vecFlagKeyExpiresAbs block, which rides LAST (after the
// relative keyTTL block). keyExpiresAbs is nil when the flag is unset — so the
// output of a non-reshard insert is byte-identical to the legacy wire. The reshard
// reinsert handlers use this decoder so a copied point's ABSOLUTE key deadlines are
// applied VERBATIM (NOT recomputed now+ttl). A present-but-truncated block is
// fail-loud.
func DecodeVectorInsertArgsKeyExpires(args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, keyExpiresAbs map[string]uint64, err error) {
	return DecodeVectorInsertArgsKeyExpiresInto(nil, args)
}

// DecodeVectorInsertArgsKeyExpiresInto is DecodeVectorInsertArgsKeyExpires that
// decodes the dense vector INTO the caller-owned scratch dst (passed as dst[:0])
// when it has the capacity, instead of allocating a fresh []float32 each call —
// the in-ops analogue of vector.GetInto on the read side. dst == nil reproduces
// the allocating public wrapper byte-for-byte. handleVectorInsert passes a pooled
// buffer here and returns it once the engine has COPIED the vector into the arena
// (arena.Insert always copies), so the scratch is never retained past the op.
func DecodeVectorInsertArgsKeyExpiresInto(dst []float32, args []byte) (collection string, id uint64, vec []float32, ttl time.Duration, meta vtypes.Metadata, sparse *vtypes.SparseVector, version uint64, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, keyExpiresAbs map[string]uint64, err error) {
	collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, keyTTLMs, err = decodeVectorInsertArgsKeyTTLInto(dst, args)
	if err != nil {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, nil, err
	}
	if len(args) < 1 || args[0]&vecFlagKeyExpiresAbs == 0 {
		return collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, keyTTLMs, nil, nil
	}
	flags, off := vectorInsertTrailerOffset(args)
	// Skip the relative keyTTL block if present (the absolute block rides after it).
	if flags&vecFlagKeyTTL != 0 {
		if len(args) < off+4 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, nil, nil, ErrVectorArgsTruncated
		}
		klen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4 + klen
	}
	if len(args) < off+4 {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, nil, ErrVectorArgsTruncated
	}
	cnt := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// An entry costs >= 12 bytes ([klen:u32] with an empty key + [ttl:u64]).
	if !CountFitsIn(cnt, len(args)-off, 12) {
		return "", 0, nil, 0, nil, nil, 0, 0, false, nil, nil, ErrVectorArgsTruncated
	}
	ke := make(map[string]uint64, cnt)
	for j := 0; j < cnt; j++ {
		if len(args) < off+4 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, nil, nil, ErrVectorArgsTruncated
		}
		kl := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if kl < 0 || kl > len(args)-off-8 {
			return "", 0, nil, 0, nil, nil, 0, 0, false, nil, nil, ErrVectorArgsTruncated
		}
		key := string(args[off : off+kl])
		off += kl
		ke[key] = binary.BigEndian.Uint64(args[off:])
		off += 8
	}
	if len(ke) > 0 {
		keyExpiresAbs = ke
	}
	return collection, id, vec, ttl, meta, sparse, version, expectedVersion, hasExpected, keyTTLMs, keyExpiresAbs, nil
}

// EncodeVectorSearchArgs serializes vector_search args with no filter (flags=0).
func EncodeVectorSearchArgs(collection string, k int, query []float32) []byte {
	return AppendVectorSearchArgsExt(nil, collection, k, query, vtypes.Filter{})
}

// AppendVectorSearchArgs is EncodeVectorSearchArgs appending into dst (reusing
// its capacity when large enough), for a hot-loop caller that pools the buffer.
// The returned slice may alias dst; the caller must store it back.
func AppendVectorSearchArgs(dst []byte, collection string, k int, query []float32) []byte {
	return AppendVectorSearchArgsExt(dst, collection, k, query, vtypes.Filter{})
}

// EncodeVectorSearchArgsExt serializes vector_search args, optionally carrying a
// metadata filter. Wire:
//
//	[flags:u8][colLen:u8][col][k:u32][dim:u32][query][?filterLen:u32][?filterJSON]
func EncodeVectorSearchArgsExt(collection string, k int, query []float32, filter vtypes.Filter) []byte {
	return AppendVectorSearchArgsExt(nil, collection, k, query, filter)
}

// AppendVectorSearchArgsExt is EncodeVectorSearchArgsExt appending into dst,
// reusing dst's capacity when it is large enough and allocating only when it
// must grow. Passing dst=nil yields the exact bytes EncodeVectorSearchArgsExt
// returned before, so the wire format is unchanged. The returned slice may
// alias dst; a caller that pools dst must store the result back.
func AppendVectorSearchArgsExt(dst []byte, collection string, k int, query []float32, filter vtypes.Filter) []byte {
	var flags uint8
	var filterJSON []byte
	if !filter.IsZero() {
		flags |= VecFlagFilter
		filterJSON, _ = json.Marshal(filter)
	}

	n := 1 + 1 + len(collection) + 4 + 4 + 4*len(query)
	if flags&VecFlagFilter != 0 {
		n += 4 + len(filterJSON)
	}
	var buf []byte
	if cap(dst) >= n {
		buf = dst[:n]
	} else {
		buf = make([]byte, n)
	}
	buf[0] = flags
	buf[1] = byte(len(collection))
	off := 2 + copy(buf[2:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(k))
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(len(query)))
	off += 4
	for _, f := range query {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
		off += 4
	}
	if flags&VecFlagFilter != 0 {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(filterJSON)))
		off += 4
		copy(buf[off:], filterJSON)
	}
	return buf
}

// DecodeVectorSearchArgs reads vector_search args (v3 layout with flags byte).
// The query is freshly allocated; for the hot server path use
// DecodeVectorSearchArgsInto with a reused buffer.
func DecodeVectorSearchArgs(args []byte) (collection string, k int, query []float32, filter vtypes.Filter, err error) {
	return DecodeVectorSearchArgsInto(args, nil)
}

// DecodeVectorSearchArgsInto is DecodeVectorSearchArgs but decodes the query
// into dst when cap(dst) is large enough, allocating only when it must grow.
// The returned query aliases dst (or a fresh slice); callers that pool dst must
// store the returned slice back. The no-filter path allocates nothing beyond
// the (small) collection string.
func DecodeVectorSearchArgsInto(args []byte, dst []float32) (collection string, k int, query []float32, filter vtypes.Filter, err error) {
	if len(args) < 2 {
		return "", 0, nil, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	if len(args) < 2+colLen+4+4 {
		return "", 0, nil, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	collection = string(args[2 : 2+colLen])
	off := 2 + colLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// CountFitsIn, not `len(args) < off+4*dim` — see DecodeVectorInsertArgs: a
	// negative dim (32-bit widening) makes 4*dim overflow to 0 and the comparison
	// vacuous, after which the slice bound below panics.
	if !CountFitsIn(dim, len(args)-off, 4) {
		return "", 0, nil, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	if cap(dst) >= dim {
		query = dst[:dim]
	} else {
		query = make([]float32, dim)
	}
	for i := 0; i < dim; i++ {
		query[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	if flags&VecFlagFilter != 0 {
		if len(args) < off+4 {
			return "", 0, nil, vtypes.Filter{}, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if flen < 0 || len(args)-off < flen {
			return "", 0, nil, vtypes.Filter{}, ErrVectorArgsTruncated
		}
		if err := json.Unmarshal(args[off:off+flen], &filter); err != nil {
			return "", 0, nil, vtypes.Filter{}, fmt.Errorf("ops: decode filter: %w", err)
		}
	}
	return collection, k, query, filter, nil
}

// EncodeVectorSearchArgsOpts serializes vector_search args with an optional
// metadata filter AND optional cross-shard consistency opts. When both
// readConsistency and onPartitionUnavailable are zero the output is
// byte-identical to EncodeVectorSearchArgsExt (backward-compatible). Non-zero
// opts append two bytes after the filter block and set vecFlagSearchOpts in
// the flags byte. Wire (when opts present):
//
//	[flags:u8][colLen:u8][col][k:u32][dim:u32][query]
//	  [?filterLen:u32][?filterJSON]        ← present when VecFlagFilter set
//	  [readConsistency:u8][onPartitionUnavailable:u8]  ← present when vecFlagSearchOpts set
//
// readConsistency: 0=AnyReplica (default), 1=LeaderOnly.
// onPartitionUnavailable: 0=Partial (default, return partial results), 1=Fail.
// Both fields apply only to the clustered backend (Partitions>1); single-shard
// deployments ignore them.
func EncodeVectorSearchArgsOpts(collection string, k int, query []float32, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	base := EncodeVectorSearchArgsExt(collection, k, query, filter) // single source of truth for the leading block
	if readConsistency == 0 && onPartitionUnavailable == 0 {
		return base // byte-identical to the legacy form
	}
	base[0] |= vecFlagSearchOpts
	out := append(base, readConsistency, onPartitionUnavailable)
	return appendBoundTail(out, readConsistency, bound) // 8 bound bytes ride ONLY when rc==BoundedStaleness
}

// DecodeVectorSearchArgsOpts decodes vector_search args that may carry
// cross-shard consistency opts (vecFlagSearchOpts). It is backward-compatible:
// args encoded by the legacy EncodeVectorSearchArgs / EncodeVectorSearchArgsExt
// (without the opts trailer) decode with readConsistency=0, onPartitionUnavailable=0.
func DecodeVectorSearchArgsOpts(args []byte) (collection string, k int, query []float32, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	if len(args) < 2 {
		return "", 0, nil, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	if len(args) < 2+colLen+4+4 {
		return "", 0, nil, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
	}
	collection = string(args[2 : 2+colLen])
	off := 2 + colLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// See DecodeVectorInsertArgs: 4*dim overflows to 0 for a negative dim, and
	// make([]float32, dim) then panics rather than erroring.
	if !CountFitsIn(dim, len(args)-off, 4) {
		return "", 0, nil, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
	}
	query = make([]float32, dim)
	for i := 0; i < dim; i++ {
		query[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	if flags&VecFlagFilter != 0 {
		if len(args) < off+4 {
			return "", 0, nil, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if flen < 0 || len(args)-off < flen {
			return "", 0, nil, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &filter); uerr != nil {
			return "", 0, nil, vtypes.Filter{}, 0, 0, 0, fmt.Errorf("ops: decode filter: %w", uerr)
		}
		off += flen
	}
	if flags&vecFlagSearchOpts != 0 {
		// The flag being set is the contract that the 2-byte opts trailer
		// follows; a missing trailer means corruption/truncation. Fail loud
		// rather than silently downgrading to AnyReplica/Partial, which would
		// quietly weaken an explicit LeaderOnly/Fail request.
		if len(args) < off+2 {
			return "", 0, nil, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
		bound, _, err = readBoundTail(args, off, readConsistency)
		if err != nil {
			return "", 0, nil, vtypes.Filter{}, 0, 0, 0, err
		}
	}
	return collection, k, query, filter, readConsistency, onPartitionUnavailable, bound, nil
}

func EncodeVectorSearchResults(results []vtypes.Result) []byte {
	n := 4 + len(results)*(8+4)
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(results)))
	off := 4
	for _, r := range results {
		binary.BigEndian.PutUint64(buf[off:], r.ID)
		off += 8
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(r.Distance))
		off += 4
	}
	return buf
}

func DecodeVectorSearchResults(body []byte) ([]vtypes.Result, error) {
	return DecodeVectorSearchResultsInto(body, nil)
}

// DecodeVectorSearchResultsInto decodes search results into dst (reused when
// cap allows, else a fresh slice), returning the populated slice. Pair it with
// client.CallFunc so a hot-loop caller reusing dst decodes the wire response
// with zero allocations (no defensive payload copy, no result-slice alloc).
func DecodeVectorSearchResultsInto(body []byte, dst []vtypes.Result) ([]vtypes.Result, error) {
	if len(body) < 4 {
		return nil, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	// See above: the multiplied form overflows for a negative count.
	if !CountFitsIn(count, len(body)-4, 8+4) {
		return nil, ErrVectorArgsTruncated
	}
	var results []vtypes.Result
	if cap(dst) >= count {
		results = dst[:count]
	} else {
		results = make([]vtypes.Result, count)
	}
	off := 4
	for i := 0; i < count; i++ {
		results[i].ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		results[i].Distance = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
		results[i].Score = 0
	}
	return results, nil
}

// appendDegradedTrailer appends [degraded:u8][missingCount:u16]{[partID:u16]} to
// a self-delimiting result body. When !degraded && len(missing)==0 it returns
// body unchanged, so legacy decoders that read only the base block see
// byte-identical output. Partition IDs and the count are u16 (supports up to
// 65535 partitions, far above realistic counts).
func appendDegradedTrailer(body []byte, degraded bool, missing []uint16) []byte {
	if !degraded && len(missing) == 0 {
		return body
	}
	n := len(body) + 1 + 2 + 2*len(missing)
	buf := make([]byte, n)
	copy(buf, body)
	off := len(body)
	if degraded {
		buf[off] = 1
	}
	off++
	binary.BigEndian.PutUint16(buf[off:], uint16(len(missing))) //nolint:gosec // bounded by partition count
	off += 2
	for _, pid := range missing {
		binary.BigEndian.PutUint16(buf[off:], pid)
		off += 2
	}
	return buf
}

// readDegradedTrailer parses the degraded trailer from body[offset:], where
// offset is the byte position at which the base result decoder stopped. It is
// backward-compatible: when no trailer bytes (or only a partial trailer) are
// present it returns (false, nil, nil) so legacy bodies decode cleanly.
func readDegradedTrailer(body []byte, offset int) (degraded bool, missing []uint16, err error) {
	// Legacy bytes end at offset; no trailer present.
	if len(body) <= offset {
		return false, nil, nil
	}
	// Need at least [degraded:u8][missingCount:u16]; tolerate a partial trailer.
	if len(body) < offset+3 {
		return false, nil, nil
	}
	degraded = body[offset] != 0
	missingCount := int(binary.BigEndian.Uint16(body[offset+1:]))
	trailerEnd := offset + 3 + 2*missingCount
	if len(body) < trailerEnd {
		// Truncated missing list — tolerate and return what we have.
		return degraded, nil, nil
	}
	if missingCount > 0 {
		missing = make([]uint16, missingCount)
		off := offset + 3
		for i := 0; i < missingCount; i++ {
			missing[i] = binary.BigEndian.Uint16(body[off:])
			off += 2
		}
	}
	return degraded, missing, nil
}

// readDegradedTrailerN is readDegradedTrailer plus the byte offset at which the
// trailer ends (so a caller can read a tail that follows it, e.g. the scroll
// next_cursor). When no (or only a partial) trailer is present it returns the
// unchanged offset, so a legacy body leaves the cursor-tail check to find no
// further bytes ⇒ empty cursor.
func readDegradedTrailerN(body []byte, offset int) (degraded bool, missing []uint16, end int, err error) {
	if len(body) <= offset {
		return false, nil, offset, nil
	}
	if len(body) < offset+3 {
		return false, nil, offset, nil
	}
	degraded = body[offset] != 0
	missingCount := int(binary.BigEndian.Uint16(body[offset+1:]))
	trailerEnd := offset + 3 + 2*missingCount
	if len(body) < trailerEnd {
		return degraded, nil, offset, nil
	}
	if missingCount > 0 {
		missing = make([]uint16, missingCount)
		off := offset + 3
		for i := 0; i < missingCount; i++ {
			missing[i] = binary.BigEndian.Uint16(body[off:])
			off += 2
		}
	}
	return degraded, missing, trailerEnd, nil
}

// EncodeVectorSearchResultsDegraded encodes search results with an optional
// degraded-partition trailer. When degraded is false and missing is empty the
// output is byte-identical to EncodeVectorSearchResults (backward-compatible).
// Wire:
//
//	[count:u32]{[id:u64][distance:f32]}    ← existing block (unchanged)
//	[degraded:u8][missingCount:u16]{[partitionID:u16]}  ← trailer, omitted when not degraded
//
// partitionID values in missing identify which shard partitions returned no
// results (e.g. due to unavailability). Legacy decoders reading only the base
// block stop before the trailer; DecodeVectorSearchResultsDegraded tolerates
// its absence (→ degraded=false, missing=nil).
// Partition IDs and the count are encoded as u16 (supports up to 65535 partitions, far above realistic counts).
func EncodeVectorSearchResultsDegraded(results []vtypes.Result, degraded bool, missing []uint16) []byte {
	return appendDegradedTrailer(EncodeVectorSearchResults(results), degraded, missing)
}

// DecodeVectorSearchResultsDegraded decodes search results and the optional
// degraded-partition trailer. It is backward-compatible: bytes produced by the
// legacy EncodeVectorSearchResults (no trailer) decode with degraded=false and
// missing=nil. The base result block is read exactly (count*12 bytes after the
// count u32); any remaining bytes are interpreted as the degraded trailer.
func DecodeVectorSearchResultsDegraded(body []byte) (results []vtypes.Result, degraded bool, missing []uint16, err error) {
	if len(body) < 4 {
		return nil, false, nil, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	// See decodeHybridResultsN: the multiplied form overflows for a negative
	// count, so the bound is taken by division first and baseEnd computed only
	// once count is known to fit.
	if !CountFitsIn(count, len(body)-4, 8+4) {
		return nil, false, nil, ErrVectorArgsTruncated
	}
	baseEnd := 4 + count*(8+4)
	results = make([]vtypes.Result, count)
	off := 4
	for i := 0; i < count; i++ {
		results[i].ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		results[i].Distance = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
	}
	degraded, missing, err = readDegradedTrailer(body, baseEnd)
	return results, degraded, missing, err
}

// EncodeHybridSearchArgs serializes vector_hybrid_search args. Wire:
//
//	[flags:u8]                       bit0=HAS_FILTER, bit1=HAS_SPARSE
//	[colLen:u8][col]
//	[k:u32]
//	[method:u8][alpha:f64][rrfK:u32][denseK:u32][sparseK:u32]
//	[dim:u32][dense: f32×dim]
//	if HAS_SPARSE: [nnz:u32]{[dim:u32][value:f32]}
//	if HAS_FILTER: [filterLen:u32][filterJSON]
func EncodeHybridSearchArgs(collection string, dense []float32, k int, sparse vtypes.SparseVector, opts vtypes.HybridOpts) []byte {
	var flags uint8
	var filterJSON []byte
	if !opts.Filter.IsZero() {
		flags |= hybridFlagFilter
		filterJSON, _ = json.Marshal(opts.Filter)
	}
	if !sparse.IsZero() {
		flags |= hybridFlagSparse
	}

	n := 1 + 1 + len(collection) + 4 + (1 + 8 + 4 + 4 + 4) + 4 + 4*len(dense)
	if flags&hybridFlagSparse != 0 {
		n += 4 + 8*len(sparse.Indices)
	}
	if flags&hybridFlagFilter != 0 {
		n += 4 + len(filterJSON)
	}
	buf := make([]byte, n)
	buf[0] = flags
	buf[1] = byte(len(collection))
	off := 2 + copy(buf[2:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(k))
	off += 4
	buf[off] = byte(opts.Method)
	off++
	binary.BigEndian.PutUint64(buf[off:], math.Float64bits(opts.Alpha))
	off += 8
	binary.BigEndian.PutUint32(buf[off:], uint32(opts.RRFK))
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(opts.DenseK))
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(opts.SparseK))
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(len(dense)))
	off += 4
	for _, f := range dense {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
		off += 4
	}
	if flags&hybridFlagSparse != 0 {
		off = writeSparse(buf, off, sparse)
	}
	if flags&hybridFlagFilter != 0 {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(filterJSON)))
		off += 4
		copy(buf[off:], filterJSON)
	}
	return buf
}

// DecodeHybridSearchArgs reads vector_hybrid_search args. opts.Filter is the
// zero filter when absent; sparse is the zero SparseVector when absent.
func DecodeHybridSearchArgs(args []byte) (collection string, dense []float32, k int, sparse vtypes.SparseVector, opts vtypes.HybridOpts, err error) {
	collection, dense, k, sparse, opts, _, err = decodeHybridSearchArgsN(args)
	return collection, dense, k, sparse, opts, err
}

// decodeHybridSearchArgsN decodes the hybrid-search base block and returns the
// number of bytes consumed (so DecodeHybridSearchArgsOpts can read a trailing
// opts block). Trailing bytes beyond the base block are ignored here.
func decodeHybridSearchArgsN(args []byte) (collection string, dense []float32, k int, sparse vtypes.SparseVector, opts vtypes.HybridOpts, n int, err error) {
	if len(args) < 2 {
		return "", nil, 0, sparse, opts, 0, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	// fixed: flags(1)+colLen(1)+col+k(4)+method(1)+alpha(8)+rrfK(4)+denseK(4)+sparseK(4)+dim(4)
	if len(args) < 2+colLen+4+1+8+4+4+4+4 {
		return "", nil, 0, sparse, opts, 0, ErrVectorArgsTruncated
	}
	collection = string(args[2 : 2+colLen])
	off := 2 + colLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.Method = vtypes.FusionMethod(args[off])
	off++
	opts.Alpha = math.Float64frombits(binary.BigEndian.Uint64(args[off:]))
	off += 8
	opts.RRFK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.DenseK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.SparseK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// See DecodeVectorInsertArgs: 4*dim overflows to 0 for a negative dim, and
	// make([]float32, dim) then panics rather than erroring.
	if !CountFitsIn(dim, len(args)-off, 4) {
		return "", nil, 0, sparse, opts, 0, ErrVectorArgsTruncated
	}
	dense = make([]float32, dim)
	for i := 0; i < dim; i++ {
		dense[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	if flags&hybridFlagSparse != 0 {
		sv, noff, serr := readSparse(args, off)
		if serr != nil {
			return "", nil, 0, sparse, opts, 0, serr
		}
		sparse = sv
		off = noff
	}
	if flags&hybridFlagFilter != 0 {
		if len(args) < off+4 {
			return "", nil, 0, sparse, opts, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if flen < 0 || len(args)-off < flen {
			return "", nil, 0, sparse, opts, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &opts.Filter); uerr != nil {
			return "", nil, 0, sparse, opts, 0, fmt.Errorf("ops: decode filter: %w", uerr)
		}
		off += flen
	}
	return collection, dense, k, sparse, opts, off, nil
}

// EncodeHybridSearchArgsOpts serializes vector_hybrid_search args with an
// optional cross-shard consistency opts trailer. When both readConsistency and
// onPartitionUnavailable are zero the output is byte-identical to
// EncodeHybridSearchArgs (backward-compatible). Non-zero opts append two bytes
// after the existing sparse/filter blocks and set hybridFlagOpts in the flags
// byte. readConsistency: 0=AnyReplica, 1=LeaderOnly. onPartitionUnavailable:
// 0=Partial, 1=Fail.
func EncodeHybridSearchArgsOpts(collection string, dense []float32, k int, sparse vtypes.SparseVector, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	base := EncodeHybridSearchArgs(collection, dense, k, sparse, opts) // single source of truth for the leading blocks
	if readConsistency == 0 && onPartitionUnavailable == 0 {
		return base // byte-identical to the legacy form
	}
	base[0] |= hybridFlagOpts
	out := append(base, readConsistency, onPartitionUnavailable)
	return appendBoundTail(out, readConsistency, bound) // 8 bound bytes ride ONLY when rc==BoundedStaleness
}

// DecodeHybridSearchArgsOpts decodes vector_hybrid_search args that may carry
// cross-shard consistency opts (hybridFlagOpts). It is backward-compatible: args
// encoded by the legacy EncodeHybridSearchArgs (without the opts trailer) decode
// with readConsistency=0, onPartitionUnavailable=0.
func DecodeHybridSearchArgsOpts(args []byte) (collection string, dense []float32, k int, sparse vtypes.SparseVector, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, dense, k, sparse, opts, n, err := decodeHybridSearchArgsN(args)
	if err != nil {
		return "", nil, 0, sparse, opts, 0, 0, 0, err
	}
	if len(args) > 0 && args[0]&hybridFlagOpts != 0 {
		// The flag being set is the contract that the 2-byte opts trailer
		// follows; a missing trailer means corruption/truncation. Fail loud
		// rather than silently downgrading an explicit LeaderOnly/Fail request.
		if len(args) < n+2 {
			return "", nil, 0, sparse, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[n]
		onPartitionUnavailable = args[n+1]
		bound, _, err = readBoundTail(args, n+2, readConsistency)
		if err != nil {
			return "", nil, 0, sparse, opts, 0, 0, 0, err
		}
	}
	return collection, dense, k, sparse, opts, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeHybridResults serializes fused results carrying both the dense
// distance and the fusion score. Wire: [count:u32]{[id:u64][distance:f32][score:f32]}.
func EncodeHybridResults(results []vtypes.Result) []byte {
	n := 4 + len(results)*(8+4+4)
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(results)))
	off := 4
	for _, r := range results {
		binary.BigEndian.PutUint64(buf[off:], r.ID)
		off += 8
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(r.Distance))
		off += 4
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(r.Score))
		off += 4
	}
	return buf
}

// decodeHybridResultsN decodes one hybrid-results block and returns the results
// plus the number of bytes consumed (so callers can decode back-to-back blocks).
func decodeHybridResultsN(body []byte) ([]vtypes.Result, int, error) {
	if len(body) < 4 {
		return nil, 0, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	// CountFitsIn rather than `len(body) < 4+count*16`: a negative count (32-bit
	// widening) makes that product overflow and the comparison vacuous. The
	// division form cannot overflow and rejects the sign.
	if !CountFitsIn(count, len(body)-4, 8+4+4) {
		return nil, 0, ErrVectorArgsTruncated
	}
	// Safe to compute only AFTER the bound above: count is now known non-negative
	// and small enough that this product cannot overflow.
	need := 4 + count*(8+4+4)
	results := make([]vtypes.Result, count)
	off := 4
	for i := 0; i < count; i++ {
		results[i].ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		results[i].Distance = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
		results[i].Score = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
	}
	return results, need, nil
}

// DecodeHybridResults reads results produced by EncodeHybridResults.
func DecodeHybridResults(body []byte) ([]vtypes.Result, error) {
	r, _, err := decodeHybridResultsN(body)
	return r, err
}

// EncodeHybridResultsDegraded encodes hybrid results with an optional
// degraded-partition trailer (same wire format as the search trailer). Hybrid
// results carry a Score field (8+4+4 per row) so they are NOT byte-compatible
// with EncodeVectorSearchResults and need their own degraded codec. When
// degraded is false and missing is empty the output is byte-identical to
// EncodeHybridResults.
func EncodeHybridResultsDegraded(results []vtypes.Result, degraded bool, missing []uint16) []byte {
	return appendDegradedTrailer(EncodeHybridResults(results), degraded, missing)
}

// DecodeHybridResultsDegraded decodes hybrid results and the optional degraded
// trailer. Backward-compatible with legacy EncodeHybridResults bytes.
func DecodeHybridResultsDegraded(body []byte) (results []vtypes.Result, degraded bool, missing []uint16, err error) {
	results, off, err := decodeHybridResultsN(body)
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, off)
	return results, degraded, missing, err
}

// EncodeHybridLanesResult serializes the two pre-fusion hybrid lanes back-to-back
// (dense lane first, then sparse lane), each using the hybrid-result wire format.
// Wire: <EncodeHybridResults(dense)><EncodeHybridResults(sparse)>
func EncodeHybridLanesResult(dense, sparse []vtypes.Result) []byte {
	return append(EncodeHybridResults(dense), EncodeHybridResults(sparse)...)
}

// DecodeHybridLanesResult decodes the dense lane then the sparse lane.
func DecodeHybridLanesResult(body []byte) (dense, sparse []vtypes.Result, err error) {
	dense, n, err := decodeHybridResultsN(body)
	if err != nil {
		return nil, nil, err
	}
	sparse, _, err = decodeHybridResultsN(body[n:])
	if err != nil {
		return nil, nil, err
	}
	return dense, sparse, nil
}

func EncodeVectorDeleteArgs(collection string, id uint64) []byte {
	n := 1 + len(collection) + 8
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	return buf
}

// EncodeVectorDeleteArgsCAS serializes a vector_delete request with an optional
// optimistic-CAS precondition. Wire:
//
//	[colLen:u8][col][id:u64]{[casPresent:u8][?expectedVersion:u64]}
//
// When hasExpected is false NOTHING is appended after the id, so the output is
// BYTE-IDENTICAL to EncodeVectorDeleteArgs (the legacy 2-tuple decoder still
// works). When present the trailing [1][expectedVersion:u64] is the CAS guard:
// the delete applies ONLY when the point's current version matches.
func EncodeVectorDeleteArgsCAS(collection string, id uint64, expectedVersion uint64, hasExpected bool) []byte {
	if !hasExpected {
		return EncodeVectorDeleteArgs(collection, id)
	}
	n := 1 + len(collection) + 8 + 1 + 8
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	off += 8
	buf[off] = 1
	off++
	binary.BigEndian.PutUint64(buf[off:], expectedVersion)
	return buf
}

func DecodeVectorDeleteArgs(args []byte) (string, uint64, error) {
	col, id, _, _, err := DecodeVectorDeleteArgsCAS(args)
	return col, id, err
}

// DecodeVectorDeleteArgsCAS reads args produced by EncodeVectorDeleteArgsCAS (or
// the legacy EncodeVectorDeleteArgs — the trailing CAS block is optional).
// hasExpected reports whether a CAS precondition trailer was present;
// expectedVersion is its value (0 = expect-absent) when hasExpected is true. An
// absent (legacy) trailer decodes to hasExpected=false; a present-but-truncated
// trailer is fail-loud.
func DecodeVectorDeleteArgsCAS(args []byte) (collection string, id uint64, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 1 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen + 8
	id = binary.BigEndian.Uint64(args[1+colLen:])
	// Optional trailing CAS block. A legacy encoder stops right after the id.
	if off >= len(args) {
		return collection, id, 0, false, nil
	}
	present := args[off]
	off++
	if present == 0 {
		return collection, id, 0, false, nil
	}
	if len(args) < off+8 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	expectedVersion = binary.BigEndian.Uint64(args[off:])
	return collection, id, expectedVersion, true, nil
}

// EncodeIfAbsentResult serializes the result of an insert-if-absent / add-if-absent
// op: a single byte, 1 if the record was inserted, 0 if it was a no-op (the id was
// already live). Shared by vector_insert_if_absent and vector_mv_add_if_absent. The
// write side reuses the existing insert/add arg codecs (the op name selects dense vs MV).
func EncodeIfAbsentResult(inserted bool) []byte {
	if inserted {
		return []byte{1}
	}
	return []byte{0}
}

// DecodeIfAbsentResult reads the byte produced by EncodeIfAbsentResult.
func DecodeIfAbsentResult(body []byte) (bool, error) {
	if len(body) < 1 {
		return false, ErrVectorArgsTruncated
	}
	return body[0] == 1, nil
}

// EncodeExistsArgs serializes a vector_exists request. Wire: [colLen:u8][col][id:u64]
// — byte-identical to the delete-args shape (VectorKeyColAt1 reads the leading
// collection name for routing). Used by the dense liveness probe.
func EncodeExistsArgs(collection string, id uint64) []byte {
	return EncodeVectorDeleteArgs(collection, id)
}

// DecodeExistsArgs reads args produced by EncodeExistsArgs.
func DecodeExistsArgs(args []byte) (string, uint64, error) {
	return DecodeVectorDeleteArgs(args)
}

// EncodeExistsResult serializes a liveness probe result: 1 if the id is live, 0
// otherwise. Shared by vector_exists and vector_mv_exists.
func EncodeExistsResult(exists bool) []byte {
	if exists {
		return []byte{1}
	}
	return []byte{0}
}

// DecodeExistsResult reads the byte produced by EncodeExistsResult.
func DecodeExistsResult(body []byte) (bool, error) {
	if len(body) < 1 {
		return false, ErrVectorArgsTruncated
	}
	return body[0] == 1, nil
}

// EncodeCreateCollectionArgs serializes a create-collection request. Wire:
//
//	[nameLen:1][name][dim:4][metric:1][M:4][efC:4][efS:4][seed:8]
//	  [quant:1][persistent:1][rescoreFactor:4]      (quant/persist extension)
//	  [extendCandidates:1][extendCandidatesMax:4]   (graph-quality extension)
//	  [level0FullDegree:1][quantizedBuild:1][partitions:4]
//	  [indexType:1][ivfNlist:4][ivfNprobe:4]        (IVF extension, optional)
//
// All extensions are appended (not versioned): DecodeCreateCollectionArgs reads
// each only when present, so a pre-extension client still decodes (with the
// extension fields at their zero values). The IVF extension is written ONLY when
// non-default (IndexType != HNSW || IVFNlist != 0 || IVFNprobe != 0), so a plain
// HNSW create stays BYTE-IDENTICAL to the pre-IVF encoder. Mmap/graph paths are
// intentionally NOT on the wire — the server's CollectionStore derives them for
// Persistent collections.
//
// The trained-scalar/product-residual quant params ride at the VERY END, after
// the FullText trailer: [?sqBits:4][?prqLayers:4]. They are written only when
// non-zero (PRQLayers forces SQBits forces the FullText presence anchor), so a
// create with SQBits==0 && PRQLayers==0 is BYTE-IDENTICAL to the pre-feature
// encoder. The QuantSQ/QuantPRQ enum values themselves ride the existing
// [quant:1] byte, so selecting the new modes adds no bytes beyond these params.
func EncodeCreateCollectionArgs(name string, cfg vtypes.Config) []byte {
	persistent := byte(0)
	if cfg.Persistent {
		persistent = 1
	}
	extend := byte(0)
	if cfg.ExtendCandidates {
		extend = 1
	}
	l0full := byte(0)
	if cfg.Level0FullDegree {
		l0full = 1
	}
	qbuild := byte(0)
	if cfg.QuantizedBuild {
		qbuild = 1
	}
	// IVFTrainThreshold (a 4-byte word) rides at the VERY END of the trailer, but
	// the decoder's upstream optional blocks are read with greedy length guards
	// (the IVF header needs >=9 bytes, the PQ sub-block >=6, QuantPQM >=4): a bare
	// 4-byte threshold tail would be mis-consumed by whichever greedy read fires
	// first. To keep the trailer unambiguous, a non-zero threshold FORCES every
	// upstream optional block to be present (the IVF header, the IVF-PQ sub-block,
	// and the QuantPQM word), so each greedy guard lands on a real, semantically
	// zero-valued word and the threshold sits alone at the tail. This mirrors the
	// existing idiom where PQDropVecs forces the OPQ slot. All forced blocks carry
	// the config's true values (zeros for an HNSW/non-PQ create), so the decode
	// round-trips. Byte-identical when IVFTrainThreshold==0 (forces nothing).
	// Drift-retrain trailer (ivfDriftRetrain:1 + ivfDriftGrowthFactor:8 f64 +
	// ivfDriftFactor:8 f64) rides at the VERY END, AFTER the IVFTrainThreshold word.
	// Like the threshold, it is appended only when non-default and, because it sits
	// past every greedy-guarded read, a non-default drift config FORCES the
	// IVFTrainThreshold word present too (which transitively forces all upstream
	// optional blocks), so the threshold's 4-byte read anchors this trailing block.
	// Byte-identical when all three drift fields are zero/false.
	// FilterFirstRelativeBP (a 4-byte word) rides at the VERY END of the trailer,
	// AFTER the drift block. Like the threshold/drift tail, it is appended only when
	// non-zero and, because it sits past every greedy-guarded read AND the drift
	// block, a non-zero relativeBP FORCES the drift block present (which transitively
	// forces the IVFTrainThreshold word and every upstream optional block), so the
	// relativeBP's 4-byte read anchors unambiguously at the tail. Byte-identical when
	// FilterFirstRelativeBP==0 (forces nothing — the common case incl. every existing
	// collection).
	// OPQIters (a 4-byte word) rides at the VERY END of the trailer, AFTER the
	// FilterFirstRelativeBP word. Like every trailing block it is appended only when
	// non-zero and, because it sits past every greedy-guarded read AND the relativeBP
	// word, a non-zero OPQIters FORCES the relativeBP word present (which transitively
	// forces the drift block, the IVFTrainThreshold word, and every upstream optional
	// block), so the OPQIters 4-byte read anchors unambiguously at the very tail.
	// Byte-identical when OPQIters==0 (forces nothing — the common case incl. every
	// existing collection and the default 0→1 behavior).
	// FullText (BM25) trailer ([fullText:1][analyzerLen:1][analyzer][k1:4][b:4],
	// the analyzer/k1/b present only when fullText==1) rides at the VERY END of the
	// trailer, AFTER the OPQIters word. Like every trailing block it is appended only
	// when non-nil and, because it sits past every greedy-guarded read AND the
	// OPQIters word, a non-nil FullText FORCES the OPQIters word present (which
	// transitively forces the relativeBP word, the drift block, the IVFTrainThreshold
	// word, and every upstream optional block), so the FullText presence byte anchors
	// unambiguously at the very tail. Byte-identical when FullText==nil (forces
	// nothing — the common case incl. every existing collection).
	// SQBits / PRQLayers ride at the VERY END of the trailer, AFTER the FullText
	// block. Each is a 4-byte word appended only when non-zero. Because they sit
	// past the (variable-length) FullText block, a non-zero PRQLayers FORCES SQBits
	// present, and a non-zero SQBits FORCES the FullText PRESENCE byte present (with
	// presence=0 when FullText is nil) so the decoder has a fixed anchor: it always
	// consumes the FullText presence byte before reading these two words. The new
	// Quant enum values (QuantSQ/QuantPRQ) ride the existing Quant byte, so no new
	// bytes there. Byte-identical when SQBits==0 && PRQLayers==0 (forces nothing —
	// the common case incl. every existing collection and every non-SQ/PRQ create).
	// VamanaR / VamanaL / VamanaAlpha ride at the VERY END of the trailer, AFTER
	// the SQBits/PRQLayers words. VamanaR is a 4-byte word, VamanaL a 4-byte word,
	// VamanaAlpha a 4-byte float32 word (same encoding as the other float params).
	// Each is appended only when set; because they sit past the SQBits/PRQLayers
	// words, a non-zero VamanaL or VamanaAlpha FORCES VamanaR present, and a
	// non-zero VamanaR FORCES PRQLayers present (which forces SQBits, which forces
	// the FullText presence anchor), so the decoder has a fixed forcing chain and
	// always consumes PRQLayers before reading these three words. IndexVamana
	// itself rides the existing IndexType byte (the IVF extension), so selecting it
	// adds no bytes beyond these params. Byte-identical when VamanaR==0 &&
	// VamanaL==0 && VamanaAlpha==0 (forces nothing — every existing collection and
	// every non-Vamana create).
	// AnisotropicEta / SOAR / SOARLambda / PQNBits ride at the VERY END of the
	// trailer, AFTER the VamanaR/VamanaL/VamanaAlpha words. AnisotropicEta is a
	// 4-byte float32 word, SOAR a 1-byte flag, SOARLambda a 4-byte float32 word,
	// PQNBits a 4-byte word. Each is appended only when non-default; because they
	// sit past the Vamana words, any non-default among them FORCES VamanaR present
	// (which forces PRQLayers → SQBits → the FullText presence anchor → every
	// upstream optional block), so the decoder has a fixed forcing chain and always
	// consumes VamanaAlpha before reading these four words. PQNBits is "default"
	// when 0 OR 8 (both ⇒ 8-bit PQ): only PQNBits==4 forces. The forcing order is
	// AnisotropicEta first, then SOAR, then SOARLambda, then PQNBits, so each later
	// param forces the earlier slot present as an anchor. Byte-identical when
	// AnisotropicEta==0 && !SOAR && SOARLambda==0 && PQNBits∈{0,8} (forces
	// nothing — every existing collection and every non-ScaNN create).
	pqNBits := cfg.PQNBits == 4
	soarLambda := cfg.SOARLambda != 0 || pqNBits
	soar := cfg.SOAR || soarLambda
	anisotropicEta := cfg.AnisotropicEta != 0 || soar
	vamanaR := cfg.VamanaR != 0 || cfg.VamanaL != 0 || cfg.VamanaAlpha != 0 || anisotropicEta
	vamanaL := vamanaR
	vamanaAlpha := vamanaR
	prqLayers := cfg.PRQLayers != 0 || vamanaR
	sqBits := cfg.SQBits != 0 || prqLayers
	fullText := cfg.FullText != nil
	// fullTextSlot governs whether the FullText presence byte is on the wire: when
	// FullText is set (presence=1 + body) OR when SQBits forces the slot (presence=0,
	// no body — just the anchor byte).
	fullTextSlot := fullText || sqBits
	opqIters := cfg.OPQIters != 0 || fullTextSlot
	relBP := cfg.FilterFirstRelativeBP != 0 || opqIters
	drift := cfg.IVFDriftRetrain || cfg.IVFDriftGrowthFactor != 0 || cfg.IVFDriftFactor != 0 || relBP
	threshold := cfg.IVFTrainThreshold != 0 || drift

	// The IVF extension (indexType:1 + ivfNlist:4 + ivfNprobe:4 = 9 bytes) is
	// appended only when non-default, so a plain HNSW create is byte-identical to
	// the pre-IVF encoder.
	ivfpq := cfg.IVFPQ || cfg.IVFRerank || threshold
	ivf := cfg.IndexType != vtypes.IndexHNSW || cfg.IVFNlist != 0 || cfg.IVFNprobe != 0 || ivfpq
	n := 1 + len(name) + 4 + 1 + 4 + 4 + 4 + 8 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 4
	if ivf {
		n += 1 + 4 + 4
	}
	// PQ sub-block (ivfPQ:1 + ivfPQM:4 + ivfRerank:1 = 6 bytes) appended after the
	// IVF block ONLY when PQ is requested, so a non-PQ IVF (and HNSW) create stays
	// byte-identical to the pre-PQ encoder.
	if ivfpq {
		n += 1 + 4 + 1
	}
	// PQ-HNSW quant sub-quantizer count (quantPQM:4) appended at the very end
	// ONLY when non-zero, so any create with QuantPQM == 0 (the common case, incl.
	// every IVF create) stays byte-identical to the pre-QuantPQM encoder. Quant
	// itself already rides in the quant/persist extension above.
	//
	// A non-zero IVFTrainThreshold ALSO forces this 4-byte slot (see the threshold
	// note above): it anchors the decoder's greedy QuantPQM read to a real word.
	quantpqm := cfg.QuantPQM != 0 || threshold
	if quantpqm {
		n += 4
	}
	// OPQ flag (opq:1) appended at the very end (after any QuantPQM) ONLY when
	// true, so a create with OPQ=false (the common case) stays byte-identical to
	// the pre-OPQ encoder. OPQ requires a PQ mode (validated on decode/create).
	//
	// PQDropVecs flag (pqDropVecs:1) appended after the OPQ slot ONLY when true.
	// So that the decode stays unambiguous (the OPQ slot must exist to anchor the
	// PQDropVecs byte after it), PQDropVecs=true forces the OPQ slot to be written
	// even when OPQ=false. A create with PQDropVecs=false (the common case) is
	// therefore byte-identical to the pre-PQDropVecs encoder: when OPQ=false too,
	// nothing trails; when OPQ=true, exactly the single OPQ byte trails as before.
	//
	// IVFTrainThreshold (ivfTrainThreshold:4) is appended at the very END of the
	// trailer ONLY when non-zero. Because the decoder reads every upstream optional
	// block with a GREEDY length guard (IVF header >=9, PQ sub-block >=6, QuantPQM
	// >=4), a non-zero threshold FORCES all of them present (see the `ivfpq` /
	// `quantpqm` / `threshold` flags above and the OPQ/PQDropVecs slots below) so
	// each guard lands on a real word and the threshold sits unambiguously alone at
	// the tail. A create with IVFTrainThreshold==0 (the common case, incl. every
	// existing collection) forces nothing and is BYTE-IDENTICAL to the
	// pre-threshold encoder.
	if cfg.OPQ || cfg.PQDropVecs || threshold {
		n++ // OPQ slot
	}
	if cfg.PQDropVecs || threshold {
		n++ // PQDropVecs slot
	}
	if threshold {
		n += 4 // IVFTrainThreshold word
	}
	if drift {
		n += 1 + 8 + 8 // ivfDriftRetrain:1 + ivfDriftGrowthFactor:8 + ivfDriftFactor:8
	}
	if relBP {
		n += 4 // FilterFirstRelativeBP word
	}
	if opqIters {
		n += 4 // OPQIters word
	}
	// FullText trailer: presence byte, then (when FullText != nil) analyzerLen:1 +
	// the analyzer name + k1:4 + b:4. The presence byte is appended when FullText is
	// set OR when SQBits forces the slot (as a 0-presence anchor for the trailing
	// SQBits/PRQLayers words). A non-full-text, non-SQ/PRQ create writes nothing here
	// (byte-identical to the pre-feature encoder).
	if fullTextSlot {
		n++ // presence byte
		if fullText {
			n += 1 + len(cfg.FullText.Analyzer) + 4 + 4
		}
	}
	// SQBits / PRQLayers words: each appended only when set (PRQLayers forces SQBits;
	// SQBits forces the FullText presence anchor above). Byte-identical when both 0.
	if sqBits {
		n += 4 // SQBits word
	}
	if prqLayers {
		n += 4 // PRQLayers word
	}
	// VamanaR / VamanaL / VamanaAlpha words: each appended only when set (VamanaL /
	// VamanaAlpha force VamanaR; VamanaR forces PRQLayers above). Byte-identical
	// when all three are zero.
	if vamanaR {
		n += 4 // VamanaR word
	}
	if vamanaL {
		n += 4 // VamanaL word
	}
	if vamanaAlpha {
		n += 4 // VamanaAlpha float32 word
	}
	// AnisotropicEta / SOAR / SOARLambda / PQNBits words: each appended only when
	// set (SOAR/SOARLambda/PQNBits force AnisotropicEta; PQNBits forces SOARLambda
	// forces SOAR — see the forcing chain above). Byte-identical when all default.
	if anisotropicEta {
		n += 4 // AnisotropicEta float32 word
	}
	if soar {
		n++ // SOAR flag
	}
	if soarLambda {
		n += 4 // SOARLambda float32 word
	}
	if pqNBits {
		n += 4 // PQNBits word
	}
	buf := make([]byte, n)
	buf[0] = byte(len(name))
	off := 1 + copy(buf[1:], name)
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.Dim))
	off += 4
	buf[off] = byte(cfg.Metric)
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.M))
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.EfConstruction))
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.EfSearch))
	off += 4
	binary.BigEndian.PutUint64(buf[off:], uint64(cfg.Seed))
	off += 8
	buf[off] = byte(cfg.Quant)
	off++
	buf[off] = persistent
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.RescoreFactor))
	off += 4
	buf[off] = extend
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.ExtendCandidatesMax))
	off += 4
	buf[off] = l0full
	off++
	buf[off] = qbuild
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.Partitions))
	off += 4
	if ivf {
		buf[off] = byte(cfg.IndexType)
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFNlist))
		off += 4
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFNprobe))
		off += 4
	}
	if ivfpq {
		if cfg.IVFPQ {
			buf[off] = 1
		}
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFPQM))
		off += 4
		if cfg.IVFRerank {
			buf[off] = 1
		}
		off++
	}
	if quantpqm {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.QuantPQM))
		off += 4
	}
	if cfg.OPQ || cfg.PQDropVecs || threshold {
		if cfg.OPQ {
			buf[off] = 1
		}
		off++
	}
	if cfg.PQDropVecs || threshold {
		if cfg.PQDropVecs {
			buf[off] = 1
		}
		off++
	}
	if threshold {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFTrainThreshold)) //nolint:gosec
		off += 4
	}
	if drift {
		if cfg.IVFDriftRetrain {
			buf[off] = 1
		}
		off++
		binary.BigEndian.PutUint64(buf[off:], math.Float64bits(cfg.IVFDriftGrowthFactor))
		off += 8
		binary.BigEndian.PutUint64(buf[off:], math.Float64bits(cfg.IVFDriftFactor))
		off += 8
	}
	if relBP {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.FilterFirstRelativeBP)) //nolint:gosec
		off += 4
	}
	if opqIters {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.OPQIters)) //nolint:gosec
		off += 4
	}
	if fullTextSlot {
		if fullText {
			buf[off] = 1 // presence
			off++
			buf[off] = byte(len(cfg.FullText.Analyzer))
			off++
			off += copy(buf[off:], cfg.FullText.Analyzer)
			binary.BigEndian.PutUint32(buf[off:], math.Float32bits(cfg.FullText.K1))
			off += 4
			binary.BigEndian.PutUint32(buf[off:], math.Float32bits(cfg.FullText.B))
			off += 4
		} else {
			buf[off] = 0 // presence anchor (FullText disabled, SQBits/PRQLayers follow)
			off++
		}
	}
	if sqBits {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.SQBits)) //nolint:gosec
		off += 4
	}
	if prqLayers {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.PRQLayers)) //nolint:gosec
		off += 4
	}
	if vamanaR {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.VamanaR)) //nolint:gosec
		off += 4
	}
	if vamanaL {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.VamanaL)) //nolint:gosec
		off += 4
	}
	if vamanaAlpha {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(cfg.VamanaAlpha))
		off += 4
	}
	if anisotropicEta {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(cfg.AnisotropicEta))
		off += 4
	}
	if soar {
		if cfg.SOAR {
			buf[off] = 1
		}
		off++
	}
	if soarLambda {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(cfg.SOARLambda))
		off += 4
	}
	if pqNBits {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.PQNBits)) //nolint:gosec
	}
	return buf
}

func DecodeCreateCollectionArgs(args []byte) (string, vtypes.Config, error) {
	if len(args) < 1 {
		return "", vtypes.Config{}, ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	need := 1 + nameLen + 4 + 1 + 4 + 4 + 4 + 8
	if len(args) < need {
		return "", vtypes.Config{}, ErrVectorArgsTruncated
	}
	name := string(args[1 : 1+nameLen])
	off := 1 + nameLen
	cfg := vtypes.Config{}
	cfg.Dim = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.Metric = vtypes.Metric(args[off])
	off++
	cfg.M = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.EfConstruction = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.EfSearch = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.Seed = int64(binary.BigEndian.Uint64(args[off:]))
	off += 8
	// Quant/persist extension — present iff the encoder included it.
	if len(args) >= off+1+1+4 {
		cfg.Quant = vtypes.QuantMode(args[off])
		off++
		cfg.Persistent = args[off] == 1
		off++
		cfg.RescoreFactor = int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		// Graph-quality extension — present iff the encoder included it too.
		if len(args) >= off+1+4 {
			cfg.ExtendCandidates = args[off] == 1
			off++
			cfg.ExtendCandidatesMax = int(binary.BigEndian.Uint32(args[off:]))
			off += 4
			// Level-0 full-degree flag — present iff the encoder included it.
			if len(args) >= off+1 {
				cfg.Level0FullDegree = args[off] == 1
				off++
				// Quantized-build flag — present iff the encoder included it.
				if len(args) >= off+1 {
					cfg.QuantizedBuild = args[off] == 1
					off++
					// Partitions — present iff the encoder included it.
					if len(args) >= off+4 {
						cfg.Partitions = int(binary.BigEndian.Uint32(args[off:]))
						off += 4
						// IVF extension — present iff the encoder included it
						// (written only for a non-default IVF config).
						if len(args) >= off+1+4+4 {
							cfg.IndexType = vtypes.IndexType(args[off])
							off++
							cfg.IVFNlist = int(binary.BigEndian.Uint32(args[off:]))
							off += 4
							cfg.IVFNprobe = int(binary.BigEndian.Uint32(args[off:]))
							off += 4
							// PQ sub-block — present iff the encoder included it
							// (written only when IVFPQ/IVFRerank requested).
							if len(args) >= off+1+4+1 {
								cfg.IVFPQ = args[off] == 1
								off++
								cfg.IVFPQM = int(binary.BigEndian.Uint32(args[off:]))
								off += 4
								cfg.IVFRerank = args[off] == 1
								off++
							}
						}
						// PQ-HNSW quant sub-quantizer count — appended at the very
						// end iff non-zero (so a QuantPQM == 0 create never wrote it).
						if len(args) >= off+4 {
							cfg.QuantPQM = int(binary.BigEndian.Uint32(args[off:]))
							off += 4
						}
						// OPQ flag — appended after QuantPQM iff true (so OPQ=false
						// never wrote it). A lone trailing byte (QuantPQM==0) is read
						// here; with QuantPQM written the 4-byte read above consumed
						// it first, leaving this 1 byte. Unambiguous: QuantPQM is 4
						// bytes, OPQ is 1, both append-only when set.
						if len(args) >= off+1 {
							cfg.OPQ = args[off] == 1
							off++
							// PQDropVecs flag — appended after the OPQ slot iff true (so
							// PQDropVecs=false never wrote it). The OPQ slot is always
							// present when PQDropVecs is set, anchoring this read: the
							// lone trailing byte here is unambiguously PQDropVecs.
							if len(args) >= off+1 {
								cfg.PQDropVecs = args[off] == 1
								off++
								// IVFTrainThreshold — a 4-byte word at the very end iff
								// non-zero. A non-zero threshold forces every upstream
								// optional block (IVF header, IVF-PQ sub-block,
								// QuantPQM, OPQ, PQDropVecs) to be present, so all the
								// greedy reads above consumed real words and any 4
								// bytes remaining here are unambiguously the threshold.
								if len(args) >= off+4 {
									cfg.IVFTrainThreshold = int(binary.BigEndian.Uint32(args[off:]))
									off += 4
									// Drift-retrain trailer (ivfDriftRetrain:1 +
									// ivfDriftGrowthFactor:8 + ivfDriftFactor:8 = 17 bytes)
									// at the very end iff any drift field was non-default. A
									// non-default drift config forces the IVFTrainThreshold
									// word above (which forces every upstream block), so the
									// 4-byte read just landed and any 17 bytes remaining here
									// are unambiguously the drift block. Absent => zero/false
									// (byte-identical to the pre-drift wire).
									if len(args) >= off+1+8+8 {
										cfg.IVFDriftRetrain = args[off] == 1
										off++
										cfg.IVFDriftGrowthFactor = math.Float64frombits(binary.BigEndian.Uint64(args[off:]))
										off += 8
										cfg.IVFDriftFactor = math.Float64frombits(binary.BigEndian.Uint64(args[off:]))
										off += 8
										// FilterFirstRelativeBP — a 4-byte word at the very
										// end iff non-zero. A non-zero relativeBP forces the
										// drift block above (which forces the IVFTrainThreshold
										// word and every upstream block), so the 17-byte drift
										// read just landed and any 4 bytes remaining here are
										// unambiguously the relativeBP. Absent => 0 (off,
										// byte-identical to the pre-feature wire).
										if len(args) >= off+4 {
											cfg.FilterFirstRelativeBP = int(binary.BigEndian.Uint32(args[off:]))
											off += 4
											// OPQIters — a 4-byte word at the very end iff
											// non-zero. A non-zero OPQIters forces the relativeBP
											// word above (which forces every upstream block), so
											// the 4-byte read just landed and any 4 bytes
											// remaining here are unambiguously OPQIters. Absent
											// => 0 (= 1 = v1 behavior, byte-identical to the
											// pre-feature wire).
											if len(args) >= off+4 {
												cfg.OPQIters = int(binary.BigEndian.Uint32(args[off:]))
												off += 4
												// FullText trailer — a presence byte at the very
												// end iff FullText != nil OR SQBits forces the slot.
												// A non-nil FullText OR a non-zero SQBits forces the
												// OPQIters word above (which forces every upstream
												// block), so the 4-byte read just landed and any byte
												// remaining here is unambiguously the presence flag.
												// presence==1: [analyzerLen:1][analyzer][k1:4 f32][b:4 f32]
												// follow (full text enabled). presence==0: the slot is a
												// bare anchor for the trailing SQBits/PRQLayers words
												// (full text disabled). Absent => nil (byte-identical to
												// the pre-feature wire).
												if len(args) >= off+1 {
													present := args[off]
													off++
													if present == 1 {
														if len(args) >= off+1 {
															alen := int(args[off])
															off++
															if len(args) >= off+alen+4+4 {
																analyzer := string(args[off : off+alen])
																off += alen
																k1 := math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
																off += 4
																b := math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
																off += 4
																cfg.FullText = &vtypes.FullTextConfig{Analyzer: analyzer, K1: k1, B: b}
															}
														}
													}
													// SQBits — a 4-byte word after the FullText slot iff
													// non-zero. A non-zero SQBits forced the presence byte
													// above (the anchor just consumed), so any 4 bytes
													// remaining here are unambiguously SQBits. Absent => 0
													// (⇒ 8 for QuantSQ; byte-identical to the pre-feature wire).
													if len(args) >= off+4 {
														cfg.SQBits = int(binary.BigEndian.Uint32(args[off:]))
														off += 4
														// PRQLayers — a 4-byte word after SQBits iff non-zero.
														// A non-zero PRQLayers forced the SQBits word above, so
														// any 4 bytes remaining here are unambiguously PRQLayers.
														// Absent => 0 (⇒ defaultPRQLayers for QuantPRQ).
														if len(args) >= off+4 {
															cfg.PRQLayers = int(binary.BigEndian.Uint32(args[off:]))
															off += 4
															// VamanaR — a 4-byte word after PRQLayers iff
															// non-zero. A non-zero VamanaR forced the PRQLayers
															// word above (which forced SQBits + the FullText
															// anchor + every upstream block), so any 4 bytes
															// remaining here are unambiguously VamanaR. Absent
															// => 0 (⇒ defaultVamanaR for IndexVamana).
															if len(args) >= off+4 {
																cfg.VamanaR = int(binary.BigEndian.Uint32(args[off:]))
																off += 4
																// VamanaL — a 4-byte word after VamanaR iff
																// non-zero. A non-zero VamanaL forced the VamanaR
																// word above, so any 4 bytes remaining here are
																// unambiguously VamanaL. Absent => 0 (⇒
																// defaultVamanaL for IndexVamana).
																if len(args) >= off+4 {
																	cfg.VamanaL = int(binary.BigEndian.Uint32(args[off:]))
																	off += 4
																	// VamanaAlpha — a 4-byte float32 word after
																	// VamanaL iff non-zero. A non-zero VamanaAlpha
																	// forced the VamanaR word above, so any 4 bytes
																	// remaining here are unambiguously VamanaAlpha.
																	// Absent => 0 (⇒ defaultVamanaAlpha for
																	// IndexVamana).
																	if len(args) >= off+4 {
																		cfg.VamanaAlpha = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
																		off += 4
																		// AnisotropicEta — a 4-byte float32 word after VamanaAlpha
																		// iff non-default. Any non-default ScaNN param forced VamanaR
																		// above (which forced the whole upstream chain), so any 4
																		// bytes remaining here are unambiguously AnisotropicEta.
																		// Absent => 0 (⇒ isotropic; byte-identical to pre-feature).
																		if len(args) >= off+4 {
																			cfg.AnisotropicEta = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
																			off += 4
																			// SOAR — a 1-byte flag after AnisotropicEta iff SOAR/
																			// SOARLambda/PQNBits forced the AnisotropicEta word above.
																			if len(args) >= off+1 {
																				cfg.SOAR = args[off] == 1
																				off++
																				// SOARLambda — a 4-byte float32 word after the SOAR
																				// flag iff SOARLambda/PQNBits forced the flag above.
																				if len(args) >= off+4 {
																					cfg.SOARLambda = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
																					off += 4
																					// PQNBits — a 4-byte word after SOARLambda iff
																					// PQNBits==4 forced the SOARLambda word above.
																					// MUST remain the LAST trailer word: this greedy
																					// decode reads PQNBits as the final 4 bytes of the
																					// args, so appending any new trailer after it would
																					// be misread as PQNBits. A future trailer must add
																					// its OWN forcing/length discipline ahead of this.
																					if len(args) >= off+4 {
																						cfg.PQNBits = int(binary.BigEndian.Uint32(args[off:]))
																					}
																				}
																			}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return name, cfg, nil
}

func EncodeDropCollectionArgs(name string) []byte {
	buf := make([]byte, 1+len(name))
	buf[0] = byte(len(name))
	copy(buf[1:], name)
	return buf
}

func DecodeDropCollectionArgs(args []byte) (string, error) {
	if len(args) < 1 {
		return "", ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	if len(args) < 1+nameLen {
		return "", ErrVectorArgsTruncated
	}
	return string(args[1 : 1+nameLen]), nil
}

// EncodeResplitArgs serializes an offline resplit request. Wire:
//
//	[colLen:u8][col][newP:u32]
//
// Shared by vector_resplit and vector_mv_resplit — the op NAME selects dense vs
// MV; the wire shape is identical for both.
func EncodeResplitArgs(collection string, newP int) []byte {
	buf := make([]byte, 1+len(collection)+4)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	// newP is range-validated server-side (embedded.VectorResplit/VectorMVResplit reject
	// newP <= 1 or > maxResplitPartitions) and client-side (networkedStore guards), so a
	// wrapped negative value cannot drive the resplit loop; the cast is safe.
	binary.BigEndian.PutUint32(buf[off:], uint32(newP)) //nolint:gosec
	return buf
}

// DecodeResplitArgs reads args produced by EncodeResplitArgs.
func DecodeResplitArgs(args []byte) (string, int, error) {
	if len(args) < 1 {
		return "", 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4 {
		return "", 0, ErrVectorArgsTruncated
	}
	collection := string(args[1 : 1+colLen])
	newP := int(binary.BigEndian.Uint32(args[1+colLen:]))
	return collection, newP, nil
}

// EncodeReshardArgs serializes an online reshard request. The wire shape is
// identical to a resplit ([colLen:u8][col][newP:u32]) — the op NAME selects
// dense (vector_reshard) vs MV (vector_mv_reshard); it is a thin alias so the
// reshard call sites read clearly and stay decoupled from the resplit codec.
func EncodeReshardArgs(collection string, newP int) []byte {
	return EncodeResplitArgs(collection, newP)
}

// DecodeReshardArgs reads args produced by EncodeReshardArgs.
func DecodeReshardArgs(args []byte) (string, int, error) {
	return DecodeResplitArgs(args)
}

// EncodeReshardAbortArgs serializes an online reshard-abort request (name-only).
// Wire: [colLen:u8][col]. Shared by vector_reshard_abort and
// vector_mv_reshard_abort (the op name selects dense vs MV). The result is an
// empty ack — abort errors propagate as op errors.
func EncodeReshardAbortArgs(collection string) []byte {
	return EncodeDropCollectionArgs(collection)
}

// DecodeReshardAbortArgs reads args produced by EncodeReshardAbortArgs.
func DecodeReshardAbortArgs(args []byte) (string, error) {
	return DecodeDropCollectionArgs(args)
}

// EncodeResplitCleanupArgs serializes a resplit-cleanup request (name-only).
// Wire: [colLen:u8][col] — identical to the drop-collection arg shape. Shared by
// vector_resplit_cleanup and vector_mv_resplit_cleanup (the op name selects type).
func EncodeResplitCleanupArgs(collection string) []byte {
	return EncodeDropCollectionArgs(collection)
}

// DecodeResplitCleanupArgs reads args produced by EncodeResplitCleanupArgs.
func DecodeResplitCleanupArgs(args []byte) (string, error) {
	return DecodeDropCollectionArgs(args)
}

// EncodeResplitCleanupResult serializes the count of dropped orphan partitions
// returned by a resplit cleanup. Wire: [dropped:u32] (4-byte big-endian),
// mirroring the delete-by-filter count result.
func EncodeResplitCleanupResult(dropped int) []byte {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, uint32(dropped)) //nolint:gosec // count >= 0
	return buf
}

// DecodeResplitCleanupResult reads the count returned by a resplit cleanup.
func DecodeResplitCleanupResult(body []byte) (int, error) {
	if len(body) < 4 {
		return 0, ErrVectorArgsTruncated
	}
	return int(binary.BigEndian.Uint32(body[0:4])), nil
}

// EncodeVectorDocs serializes RAG documents (SearchDocs results). Wire:
//
//	[count:u32] then per doc:
//	  [id:u64][distBits:u32][scoreBits:u32][contentLen:u32][content][metaLen:u32][metaJSON]
func EncodeVectorDocs(docs []vtypes.Document) []byte {
	n := 4
	metas := make([][]byte, len(docs))
	for i, d := range docs {
		if len(d.Metadata) > 0 {
			metas[i], _ = json.Marshal(d.Metadata)
		}
		n += 8 + 4 + 4 + 4 + len(d.Content) + 4 + len(metas[i])
	}
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(docs))) //nolint:gosec
	off := 4
	for i, d := range docs {
		binary.BigEndian.PutUint64(buf[off:], d.ID)
		off += 8
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(d.Distance))
		off += 4
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(d.Score))
		off += 4
		binary.BigEndian.PutUint32(buf[off:], uint32(len(d.Content))) //nolint:gosec
		off += 4
		off += copy(buf[off:], d.Content)
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metas[i]))) //nolint:gosec
		off += 4
		off += copy(buf[off:], metas[i])
	}
	return buf
}

// DecodeVectorDocs reads documents produced by EncodeVectorDocs.
func DecodeVectorDocs(body []byte) ([]vtypes.Document, error) {
	docs, _, err := decodeVectorDocsN(body)
	return docs, err
}

// DecodeVectorDocsRaw reads documents produced by EncodeVectorDocs WITHOUT
// decoding each hit's metadata: every field except metadata is decoded exactly as
// DecodeVectorDocs decodes it, and metadata is handed back as the JSON bytes the
// wire already carries.
//
// USE IT ONLY WHEN THE DESTINATION IS JSON. The result wire stores metadata as
// json.Marshal of the same map DecodeVectorDocs would rebuild, so re-emitting
// those bytes emits what marshalling that map would have produced — which makes a
// vector.RawDocument response byte-identical to the vector.Document one (see
// vector.RawDocument, and TestDocsRawRendersIdentically for the battery). A caller
// that needs to READ metadata fields wants DecodeVectorDocs instead.
//
// The returned metadata ALIASES body; the documents must not outlive it.
func DecodeVectorDocsRaw(body []byte) ([]vtypes.RawDocument, error) {
	docs, _, err := decodeVectorDocsRawN(body)
	return docs, err
}

// frameVectorDocsN walks one document block, decoding every field except each
// hit's metadata, which it returns as an UNVALIDATED window into body. It is the
// single framing implementation behind both document decoders — decodeVectorDocsN
// (typed: unmarshals each metadata window) and decodeVectorDocsRawN (raw:
// validates each window and keeps the bytes) — so the two can never disagree
// about where a document ends, how many there are, or which bytes are its
// metadata. Only the metadata step differs, which is the whole point.
//
// The returned metadata windows alias body and are NOT known to be well-formed
// JSON; every caller must either unmarshal them (which validates) or validate
// them. It is unexported for exactly that reason.
//
// COST OF THE SHARING, measured: routing the typed decoder through here costs it
// one extra staging slice per call — 1.1% on the rich-payload shapes, 5.5% on
// scalar/k10 (BenchmarkDocsJSONCost/*/decode-docs-wire). That is deliberate. A
// second hand-written walker would be a place for the two decoders to drift on
// bounds checks or offsets, which is the failure this whole design is built to
// make impossible; a percent on a decode the response path no longer performs is
// the cheaper side of that trade.
func frameVectorDocsN(body []byte) ([]vtypes.RawDocument, int, error) {
	if len(body) < 4 {
		return nil, 0, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// A document costs >= 20 bytes ([id:u64][dist:u32][score:u32][contentLen:u32]).
	if !CountFitsIn(count, len(body)-off, 20) {
		return nil, 0, ErrVectorArgsTruncated
	}
	docs := make([]vtypes.RawDocument, 0, count)
	for i := 0; i < count; i++ {
		if len(body) < off+8+4+4+4 {
			return nil, 0, ErrVectorArgsTruncated
		}
		var d vtypes.RawDocument
		d.ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		d.Distance = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
		d.Score = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
		clen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if clen < 0 || clen > len(body)-off-4 {
			return nil, 0, ErrVectorArgsTruncated
		}
		d.Content = string(body[off : off+clen])
		off += clen
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if mlen < 0 || len(body)-off < mlen {
			return nil, 0, ErrVectorArgsTruncated
		}
		if mlen > 0 {
			d.Metadata = body[off : off+mlen]
			off += mlen
		}
		docs = append(docs, d)
	}
	return docs, off, nil
}

// decodeVectorDocsN decodes one document block and returns the documents plus
// the number of bytes consumed (so callers can read a trailing degraded block).
func decodeVectorDocsN(body []byte) ([]vtypes.Document, int, error) {
	raw, off, err := frameVectorDocsN(body)
	if err != nil {
		return nil, 0, err
	}
	docs := make([]vtypes.Document, len(raw))
	for i, r := range raw {
		docs[i] = vtypes.Document{ID: r.ID, Distance: r.Distance, Score: r.Score, Content: r.Content}
		if len(r.Metadata) > 0 {
			m := make(vtypes.Metadata)
			// Unmarshalling IS the validation of the metadata window; a malformed or
			// non-Metadata-shaped payload fails here exactly as it always has.
			if err := json.Unmarshal(r.Metadata, &m); err != nil {
				return nil, 0, fmt.Errorf("ops: decode doc metadata: %w", err)
			}
			docs[i].Metadata = m
		}
	}
	return docs, off, nil
}

// decodeVectorDocsRawN is decodeVectorDocsN's raw twin: same framing, with each
// metadata window checked (and where necessary canonicalized) instead of
// unmarshalled. See checkRawMetadataJSON for what those two steps are and why
// they are exactly what makes the verbatim splice byte-identical.
func decodeVectorDocsRawN(body []byte) ([]vtypes.RawDocument, int, error) {
	docs, off, err := frameVectorDocsN(body)
	if err != nil {
		return nil, 0, err
	}
	for i := range docs {
		meta, err := checkRawMetadataJSON(docs[i].Metadata)
		if err != nil {
			return nil, 0, fmt.Errorf("ops: decode doc metadata: %w", err)
		}
		docs[i].Metadata = meta
	}
	return docs, off, nil
}

// checkRawMetadataJSON vets one metadata (or group-key) window before it may be
// spliced verbatim into a JSON response, and returns the bytes to splice.
//
// Splicing is only sound because the window IS json.Marshal's output for the very
// value the typed decoder would rebuild, so re-emitting it emits what re-marshaling
// that value produces. Two things can break that identity, and this handles both:
//
//   - NOT WELL-FORMED JSON. The typed decoder rejects such a body; so does this,
//     rather than emitting a malformed response. json.Valid is a scan instead of a
//     decode, which is where the win comes from.
//
//   - THE \ufffd ESCAPE — the one sequence that does not survive
//     unmarshal→marshal unchanged. encoding/json escapes an INVALID UTF-8 byte as
//     the six characters `\ufffd`, but emits a VALID U+FFFD rune as its three raw
//     UTF-8 bytes. So a window carrying `\ufffd` renders one way through the typed
//     path (raw bytes) and another way here (the escape) — same string, different
//     bytes. Every other escape encoding/json can emit (< > & from
//     HTML escaping, the U+2028/U+2029 separators, the \u00xx controls, and the
//     \" \\ \n \r \t short forms) survives the round-trip byte-for-byte, which
//     is why this is the only case. Such a window is round-tripped HERE, through
//     the same unmarshal+marshal the typed path would do, so the spliced bytes
//     are literally the bytes the typed path would have produced.
//     Reachable only when a stored payload holds invalid UTF-8, so the slow path is
//     effectively never taken; the check that finds it deliberately over-matches
//     (an escaped backslash before a literal "ufffd" trips it too) because a false
//     positive costs one round-trip and nothing else.
//
// An empty window is passed through untouched: it means "no metadata", which both
// paths render by omitting the field.
func checkRawMetadataJSON(window []byte) ([]byte, error) {
	if len(window) == 0 {
		return window, nil
	}
	if !json.Valid(window) {
		return nil, errInvalidDocMetadataJSON
	}
	if !hasReplacementEscape(window) {
		return window, nil
	}
	var m vtypes.Metadata
	if err := json.Unmarshal(window, &m); err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// checkRawGroupKeyJSON is checkRawMetadataJSON for a group key: same two hazards,
// same handling, decoded as the single vector.Value a key is rather than a map. A
// key window is never empty — EncodeGroups always marshals a Value, and the zero
// Value marshals to an object.
func checkRawGroupKeyJSON(window []byte) ([]byte, error) {
	if !json.Valid(window) {
		return nil, errInvalidDocMetadataJSON
	}
	if !hasReplacementEscape(window) {
		return window, nil
	}
	var v vtypes.Value
	if err := json.Unmarshal(window, &v); err != nil {
		return nil, err
	}
	return json.Marshal(v)
}

// hasReplacementEscape reports whether b contains a `\ufffd` escape (any hex
// case). See checkRawMetadataJSON for why that one sequence matters. It scans for
// the backslash rather than parsing escapes, so an escaped backslash followed by
// the literal text "ufffd" also matches — a deliberate over-match.
// It hunts the BACKSLASH with bytes.IndexByte rather than walking byte by byte:
// the overwhelmingly common metadata window contains no backslash at all, and
// IndexByte settles that in one vectorized pass instead of len(window) compares.
func hasReplacementEscape(b []byte) bool {
	for {
		i := bytes.IndexByte(b, '\\')
		if i < 0 || i+6 > len(b) {
			return false
		}
		if b[i+1] == 'u' && bytes.EqualFold(b[i+2:i+6], replacementHex) {
			return true
		}
		b = b[i+1:]
	}
}

// replacementHex is the hex body of the U+FFFD escape hasReplacementEscape hunts.
var replacementHex = []byte("fffd")

// errInvalidDocMetadataJSON reports a metadata window that is not well-formed
// JSON, found by the raw decoder's scan (json.Valid yields no error value of its
// own). The typed decoder surfaces encoding/json's own message instead.
//
// The raw check and the typed unmarshal do NOT agree outside the encoder's range.
// json.Valid is a SYNTAX check, while json.Unmarshal into a struct additionally
// demands an object-of-objects AND silently drops fields the struct does not
// declare — so `[1,2]` is rejected by the typed path and accepted here, and
// `{"a":{"zzz":1}}` is accepted by both but re-marshals as {"kind":"none"} there
// while splicing verbatim here. Neither is reachable: EncodeVectorDocs is the only
// writer of these bytes and it marshals a vector.Metadata. A corrupt peer gets a
// well-formed JSON response either way, never a malformed one — and
// TestDocsRawKnownShapeDivergence pins both cases so the gap stays deliberate.
var errInvalidDocMetadataJSON = errors.New("not valid JSON")

// EncodeVectorDocsDegraded encodes documents with an optional degraded-partition
// trailer (same wire format as the search trailer). When degraded is false and
// missing is empty the output is byte-identical to EncodeVectorDocs. Used by
// both search_docs and scroll (both return []vector.Document).
func EncodeVectorDocsDegraded(docs []vtypes.Document, degraded bool, missing []uint16) []byte {
	return appendDegradedTrailer(EncodeVectorDocs(docs), degraded, missing)
}

// DecodeVectorDocsDegraded decodes documents and the optional degraded trailer.
// Backward-compatible: legacy EncodeVectorDocs bytes decode with degraded=false,
// missing=nil.
func DecodeVectorDocsDegraded(body []byte) (docs []vtypes.Document, degraded bool, missing []uint16, err error) {
	docs, off, err := decodeVectorDocsN(body)
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, off)
	return docs, degraded, missing, err
}

// DecodeVectorDocsDegradedRaw is DecodeVectorDocsDegraded with raw (undecoded)
// metadata — see DecodeVectorDocsRaw for when that is the right choice and for
// the aliasing rule. The trailer is read by the same helper, so the degraded flag
// and missing list are identical to the typed decoder's.
func DecodeVectorDocsDegradedRaw(body []byte) (docs []vtypes.RawDocument, degraded bool, missing []uint16, err error) {
	docs, off, err := decodeVectorDocsRawN(body)
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, off)
	return docs, degraded, missing, err
}

// EncodeScrollResult encodes a scroll page plus the server-authoritative
// next_cursor on the result wire. Layout:
//
//	[docs]                                  ← EncodeVectorDocs body (unchanged)
//	[degraded:u8][missingCount:u16]{[partitionID:u16]}  ← degraded trailer (ALWAYS present)
//	[cursorLen:u32][cursorBytes]            ← next_cursor tail (ALWAYS present, may be 0-len)
//
// Backward-compat is exact and byte-tested (TestScrollResultOldDecoderIgnoresCursor
// + TestScrollResultNewDecoderToleratesOldBody):
//
//   - An OLD DecodeVectorDocs reader parses the docs and ignores everything after.
//   - An OLD DecodeVectorDocsDegraded reader parses docs + degraded/missing and
//     STOPS at the end of the missing trailer (readDegradedTrailer reads exactly
//     trailerEnd = off+3+2*missingCount bytes), so it ignores the trailing cursor.
//     The degraded trailer is emitted UNCONDITIONALLY here (unlike
//     EncodeVectorDocsDegraded, which omits it when not degraded) PRECISELY so that
//     a non-degraded result still positions the cursor tail AFTER a real 3-byte
//     trailer — otherwise the old decoder would misread the cursor's first bytes
//     as a degraded trailer.
//
// The named (non-degraded) path uses EncodeScrollResult(docs, false, nil, cursor).
func EncodeScrollResult(docs []vtypes.Document, degraded bool, missing []uint16, nextCursor string) []byte {
	body := EncodeVectorDocs(docs)
	// ALWAYS emit the degraded trailer so the cursor tail is unambiguously past it.
	n := len(body) + 1 + 2 + 2*len(missing) + 4 + len(nextCursor)
	buf := make([]byte, n)
	off := copy(buf, body)
	if degraded {
		buf[off] = 1
	}
	off++
	binary.BigEndian.PutUint16(buf[off:], uint16(len(missing))) //nolint:gosec // bounded by partition count
	off += 2
	for _, pid := range missing {
		binary.BigEndian.PutUint16(buf[off:], pid)
		off += 2
	}
	binary.BigEndian.PutUint32(buf[off:], uint32(len(nextCursor))) //nolint:gosec // cursor is a short base64 token
	off += 4
	copy(buf[off:], nextCursor)
	return buf
}

// DecodeScrollResult reads a scroll result produced by EncodeScrollResult. It is
// forward-compatible with an OLD server: a body with NO cursor tail (a legacy
// EncodeVectorDocs / EncodeVectorDocsDegraded body) decodes with nextCursor="".
//
//   - docs + degraded/missing are read via the shared decodeVectorDocsN /
//     readDegradedTrailer helpers (legacy-tolerant).
//   - The cursor tail is read ONLY if a full [cursorLen:u32] is present after the
//     degraded trailer AND the declared cursor bytes fit; otherwise nextCursor="".
//     (An old server's body ends at/inside the degraded region, so the trailer
//     position has no u32 length there ⇒ empty cursor, no error.)
func DecodeScrollResult(body []byte) (docs []vtypes.Document, degraded bool, missing []uint16, nextCursor string, err error) {
	docs, off, err := decodeVectorDocsN(body)
	if err != nil {
		return nil, false, nil, "", err
	}
	degraded, missing, nextCursor, err = readScrollTail(body, off)
	if err != nil {
		return nil, false, nil, "", err
	}
	return docs, degraded, missing, nextCursor, nil
}

// DecodeScrollResultRaw is DecodeScrollResult with raw (undecoded) metadata — see
// DecodeVectorDocsRaw for when that is the right choice and for the aliasing rule.
// The trailer and cursor tail come from the same readScrollTail the typed decoder
// uses, so everything after the documents is identical.
func DecodeScrollResultRaw(body []byte) (docs []vtypes.RawDocument, degraded bool, missing []uint16, nextCursor string, err error) {
	docs, off, err := decodeVectorDocsRawN(body)
	if err != nil {
		return nil, false, nil, "", err
	}
	degraded, missing, nextCursor, err = readScrollTail(body, off)
	if err != nil {
		return nil, false, nil, "", err
	}
	return docs, degraded, missing, nextCursor, nil
}

// readScrollTail reads everything a scroll result carries AFTER its document
// block: the degraded trailer and the optional next-cursor tail. Shared by the
// typed and raw scroll decoders so the legacy-tolerance rules below live in one
// place.
func readScrollTail(body []byte, off int) (degraded bool, missing []uint16, nextCursor string, err error) {
	degraded, missing, off, err = readDegradedTrailerN(body, off)
	if err != nil {
		return false, nil, "", err
	}
	// Cursor tail: present only on a new-server body. Need [cursorLen:u32].
	if len(body) < off+4 {
		return degraded, missing, "", nil
	}
	clen := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	if clen == 0 {
		return degraded, missing, "", nil
	}
	if clen < 0 || len(body)-off < clen {
		// Truncated cursor — tolerate as empty (defensive; a well-formed new body
		// never hits this). The check is written as len(body)-off (never off+clen)
		// so a widened-negative or near-MaxInt32 clen cannot overflow past it.
		return degraded, missing, "", nil
	}
	return degraded, missing, string(body[off : off+clen]), nil
}

// EncodeScanVectorsArgs serializes a vector_scan_vectors request (name-only).
// Wire: [colLen:u8][col].
func EncodeScanVectorsArgs(collection string) []byte {
	buf := make([]byte, 1+len(collection))
	buf[0] = byte(len(collection))
	copy(buf[1:], collection)
	return buf
}

// DecodeScanVectorsArgs reads args produced by EncodeScanVectorsArgs.
func DecodeScanVectorsArgs(args []byte) (string, error) {
	if len(args) < 1 {
		return "", ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen {
		return "", ErrVectorArgsTruncated
	}
	return string(args[1 : 1+colLen]), nil
}

// EncodeScanVectorsResult serializes live ScanRecords (the resplit read
// primitive). The wire format carries exactly what vector_insert needs (id, vec,
// ttl, metadata, sparse) so resplit can feed each record straight back in. Wire:
//
//	[count:u32] then per record:
//	  [id:u64][dim:u32][vec f32×dim][ttlMs:u64][metaLen:u32][metaJSON][nnz:u32{[dim:u32][val:f32]}][version:u64][keyExpires]
//
// ttlMs is the remaining TTL in milliseconds (0 = no expiry); metaLen 0 = no
// metadata; nnz 0 = no sparse vector. version is the point's per-point CAS
// version, carried so the reshard backfill reinserts version-preserving. It is a
// TRAILING u64: a decoder reading an OLD blob (no trailing version) tolerates its
// absence per-record (version → 0, the backfill then defaults a fresh insert to 1).
//
// keyExpires is the per-record ABSOLUTE per-key payload TTL trailer riding AFTER
// the version: [present:u8]{[n:u32][kLen:u32 k deadline:u64]×n} (present=0 ⇒ no
// per-key TTL, no further bytes). It carries ABSOLUTE unix-millis deadlines
// VERBATIM (NOT relative ms; NOT recomputed) so the reshard backfill restores the
// point's original key deadlines time-stable. A decoder reading an OLD blob (no
// keyExpires byte at all) tolerates its absence per-record (keyExpires → nil),
// mirroring the version trailer's EOF tolerance — scans are transient (never a
// stored artifact), so the new encoder always writes the present byte.
func EncodeScanVectorsResult(recs []vtypes.ScanRecord) []byte {
	n := 4
	metas := make([][]byte, len(recs))
	for i, r := range recs {
		if len(r.Metadata) > 0 {
			metas[i], _ = json.Marshal(r.Metadata)
		}
		nnz := 0
		if r.Sparse != nil {
			nnz = len(r.Sparse.Indices)
		}
		n += 8 + 4 + 4*len(r.Vec) + 8 + 4 + len(metas[i]) + 4 + 8*nnz + 8 // +8: trailing version
		n++                                                               // +1: keyExpires present byte
		if len(r.KeyExpires) > 0 {
			n += 4 // n:u32
			for k := range r.KeyExpires {
				n += 4 + len(k) + 8 // kLen:u32 + key + deadline:u64
			}
		}
	}
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(recs))) //nolint:gosec // count >= 0
	off := 4
	for i, r := range recs {
		binary.BigEndian.PutUint64(buf[off:], r.ID)
		off += 8
		binary.BigEndian.PutUint32(buf[off:], uint32(len(r.Vec))) //nolint:gosec
		off += 4
		for _, f := range r.Vec {
			binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
		binary.BigEndian.PutUint64(buf[off:], uint64(r.TTL.Milliseconds())) //nolint:gosec // TTL >= 0
		off += 8
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metas[i]))) //nolint:gosec
		off += 4
		off += copy(buf[off:], metas[i])
		if r.Sparse != nil {
			off = writeSparse(buf, off, *r.Sparse)
		} else {
			binary.BigEndian.PutUint32(buf[off:], 0)
			off += 4
		}
		binary.BigEndian.PutUint64(buf[off:], r.Version) // trailing per-point CAS version
		off += 8
		// ABSOLUTE per-key payload TTL trailer (present byte gated). present=0 when the
		// point has no per-key TTL — no further bytes. Deadlines are ABSOLUTE unix-ms,
		// written verbatim so the reshard reinsert restores them time-stable.
		if len(r.KeyExpires) > 0 {
			buf[off] = 1
			off++
			binary.BigEndian.PutUint32(buf[off:], uint32(len(r.KeyExpires))) //nolint:gosec
			off += 4
			for k, deadline := range r.KeyExpires {
				binary.BigEndian.PutUint32(buf[off:], uint32(len(k))) //nolint:gosec
				off += 4
				off += copy(buf[off:], k)
				binary.BigEndian.PutUint64(buf[off:], deadline)
				off += 8
			}
		} else {
			buf[off] = 0
			off++
		}
	}
	return buf
}

// DecodeScanVectorsResult reads records produced by EncodeScanVectorsResult.
func DecodeScanVectorsResult(body []byte) ([]vtypes.ScanRecord, error) {
	if len(body) < 4 {
		return nil, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// A record costs >= 12 bytes ([id:u64][dim:u32]).
	if !CountFitsIn(count, len(body)-off, 12) {
		return nil, ErrVectorArgsTruncated
	}
	recs := make([]vtypes.ScanRecord, 0, count)
	for i := 0; i < count; i++ {
		if len(body) < off+8+4 {
			return nil, ErrVectorArgsTruncated
		}
		var r vtypes.ScanRecord
		r.ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		dim := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		// CountFitsIn (divide-form) instead of len(body) < off+4*dim+8+4: on 386
		// a hostile dim near MaxInt32 makes 4*dim widen negative/zero and slip the
		// additive check, then make([]float32, dim) panics. -(8+4) reserves the
		// trailing ttl(u64)+next(u32) exactly as the old check did.
		if !CountFitsIn(dim, len(body)-off-(8+4), 4) {
			return nil, ErrVectorArgsTruncated
		}
		r.Vec = make([]float32, dim)
		for j := 0; j < dim; j++ {
			r.Vec[j] = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
			off += 4
		}
		ttlMs := binary.BigEndian.Uint64(body[off:])
		off += 8
		r.TTL = time.Duration(ttlMs) * time.Millisecond
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if mlen < 0 || len(body)-off < mlen {
			return nil, ErrVectorArgsTruncated
		}
		if mlen > 0 {
			m := make(vtypes.Metadata)
			if err := json.Unmarshal(body[off:off+mlen], &m); err != nil {
				return nil, fmt.Errorf("ops: decode scan metadata: %w", err)
			}
			r.Metadata = m
			off += mlen
		}
		sv, noff, err := readSparse(body, off)
		if err != nil {
			return nil, err
		}
		off = noff
		if !sv.IsZero() {
			cp := sv
			r.Sparse = &cp
		}
		// Trailing per-point CAS version. It is always written by the current encoder
		// (scan results are transient, never a stored artifact), so it is required.
		if len(body) < off+8 {
			return nil, ErrVectorArgsTruncated
		}
		r.Version = binary.BigEndian.Uint64(body[off:])
		off += 8
		// ABSOLUTE per-key payload TTL trailer (present byte gated). The current encoder
		// always writes the present byte; an OLD blob (no byte at all) tolerates its
		// absence per-record (keyExpires → nil), mirroring the historical version-trailer
		// EOF tolerance. present!=0 ⇒ a [n:u32]{kLen,k,deadline:u64} block follows.
		if off < len(body) {
			present := body[off]
			off++
			if present != 0 {
				if len(body) < off+4 {
					return nil, ErrVectorArgsTruncated
				}
				cnt := int(binary.BigEndian.Uint32(body[off:]))
				off += 4
				// An entry costs >= 12 bytes ([klen:u32] empty key + [ttl:u64]).
				if !CountFitsIn(cnt, len(body)-off, 12) {
					return nil, ErrVectorArgsTruncated
				}
				ke := make(map[string]uint64, cnt)
				for j := 0; j < cnt; j++ {
					if len(body) < off+4 {
						return nil, ErrVectorArgsTruncated
					}
					klen := int(binary.BigEndian.Uint32(body[off:]))
					off += 4
					if klen < 0 || klen > len(body)-off-8 {
						return nil, ErrVectorArgsTruncated
					}
					key := string(body[off : off+klen])
					off += klen
					ke[key] = binary.BigEndian.Uint64(body[off:])
					off += 8
				}
				if len(ke) > 0 {
					r.KeyExpires = ke
				}
			}
		}
		recs = append(recs, r)
	}
	return recs, nil
}

// EncodeGetConfigArgs serializes a vector_get_config request (name-only).
// Wire: [colLen:u8][col].
func EncodeGetConfigArgs(collection string) []byte {
	buf := make([]byte, 1+len(collection))
	buf[0] = byte(len(collection))
	copy(buf[1:], collection)
	return buf
}

// DecodeGetConfigArgs reads args produced by EncodeGetConfigArgs. Trailing bytes
// beyond the base [colLen][col] block (the rc/opa opts trailer) are IGNORED, so
// the single-shard get_config handler stays backward-compatible with rc-carrying
// args.
func DecodeGetConfigArgs(args []byte) (string, error) {
	col, _, err := decodeNameArgsN(args)
	return col, err
}

// EncodeGetConfigArgsOpts is EncodeGetConfigArgs plus the self-delimiting
// [marker][rc][opa] opts trailer. Byte-identical to EncodeGetConfigArgs when
// rc==0 && opa==0 (AnyReplica default unchanged); a non-zero rc rides the trailer
// so a Linearizable get_config arms the shard barrier (via ops.ReadConsistencyOf).
func EncodeGetConfigArgsOpts(collection string, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return AppendReadOptsTrailerBounded(EncodeGetConfigArgs(collection), readConsistency, onPartitionUnavailable, bound)
}

// DecodeGetConfigArgsOpts decodes a get_config request that may carry the rc/opa
// opts trailer. Backward-compatible (legacy args ⇒ rc=0,opa=0); a present marker
// with a truncated rc/opa block is corruption — fail loud.
func DecodeGetConfigArgsOpts(args []byte) (collection string, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	col, n, err := decodeNameArgsN(args)
	if err != nil {
		return "", 0, 0, 0, err
	}
	readConsistency, onPartitionUnavailable, bound, err = DecodeReadOptsTrailerBounded(args, n)
	if err != nil {
		return "", 0, 0, 0, err
	}
	return col, readConsistency, onPartitionUnavailable, bound, nil
}

// decodeNameArgsN decodes the shared single-name base block ([nameLen:u8][name])
// used by get_config / named-name / mv-get-config args, returning the bytes
// consumed so the *Opts decoders can read a trailing self-delimiting opts block.
func decodeNameArgsN(args []byte) (name string, n int, err error) {
	if len(args) < 1 {
		return "", 0, ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	if len(args) < 1+nameLen {
		return "", 0, ErrVectorArgsTruncated
	}
	return string(args[1 : 1+nameLen]), 1 + nameLen, nil
}

// EncodeGetConfigResult serializes a collection's Config as JSON (the Config is
// JSON-serializable; see vector.writeConfig). Resplit reads it to create the
// new-generation partitions with the same configuration.
func EncodeGetConfigResult(cfg vtypes.Config) []byte {
	data, _ := json.Marshal(cfg)
	return data
}

// DecodeGetConfigResult reads a Config produced by EncodeGetConfigResult.
func DecodeGetConfigResult(body []byte) (vtypes.Config, error) {
	var cfg vtypes.Config
	if err := json.Unmarshal(body, &cfg); err != nil {
		return vtypes.Config{}, fmt.Errorf("ops: decode config: %w", err)
	}
	return cfg, nil
}

// EncodeScrollArgs serializes a vector_scroll request.
// Wire: [colLen:u8][col][limit:u32][filterLen:u32][filterJSON] (filterLen 0 = no filter).
func EncodeScrollArgs(collection string, filter vtypes.Filter, limit int) []byte {
	var fj []byte
	if !filter.IsZero() {
		fj, _ = json.Marshal(filter)
	}
	buf := make([]byte, 1+len(collection)+4+4+len(fj))
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(limit)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(len(fj))) //nolint:gosec
	off += 4
	copy(buf[off:], fj)
	return buf
}

// DecodeScrollArgs reads args produced by EncodeScrollArgs. Trailing bytes
// beyond the base block (e.g. an opts trailer from EncodeScrollArgsOpts) are
// ignored, so the single-shard handler stays backward-compatible with
// opts-carrying args.
func DecodeScrollArgs(args []byte) (string, vtypes.Filter, int, error) {
	collection, filter, limit, _, err := decodeScrollArgsN(args)
	return collection, filter, limit, err
}

// decodeScrollArgsN decodes the scroll base block and returns the number of
// bytes consumed (so DecodeScrollArgsOpts can read a trailing opts block).
// Scroll args carry no flags byte, so the Opts trailer is self-delimiting
// (see EncodeScrollArgsOpts).
func decodeScrollArgsN(args []byte) (collection string, filter vtypes.Filter, limit int, n int, err error) {
	if len(args) < 1 {
		return "", vtypes.Filter{}, 0, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4+4 {
		return "", vtypes.Filter{}, 0, 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	limit = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	flen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if flen < 0 || len(args)-off < flen {
		return "", vtypes.Filter{}, 0, 0, ErrVectorArgsTruncated
	}
	if flen > 0 {
		if uerr := json.Unmarshal(args[off:off+flen], &filter); uerr != nil {
			return "", vtypes.Filter{}, 0, 0, fmt.Errorf("ops: decode filter: %w", uerr)
		}
	}
	off += flen
	return collection, filter, limit, off, nil
}

// EncodeScrollArgsOpts serializes a vector_scroll request plus an optional
// cross-shard consistency opts trailer. Scroll args carry no flags byte, so the
// trailer is self-delimiting: [optsPresent:u8][rc:u8][opa:u8], appended ONLY
// when readConsistency!=0 || onPartitionUnavailable!=0. When both are zero the
// output is byte-identical to EncodeScrollArgs (backward-compatible) and the
// plain DecodeScrollArgs (used by the single-shard handler) ignores trailing bytes.
func EncodeScrollArgsOpts(collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8) []byte {
	return EncodeScrollArgsCursor(collection, filter, limit, readConsistency, onPartitionUnavailable, 0, false)
}

// EncodeScrollArgsCursor serializes a vector_scroll request with the optional
// cross-shard consistency opts trailer AND an optional resume-after-id cursor
// (the Task-2 partition fan-out passes the SAME global afterID to every
// partition). It extends the self-delimiting opts trailer:
//
//	[optsPresent:u8=1][rc:u8][opa:u8][cursorPresent:u8][afterID:u64 BE iff cursorPresent]
//
// The whole trailer is appended ONLY when readConsistency!=0 ||
// onPartitionUnavailable!=0 || hasAfter — so the no-opts/no-cursor default is
// byte-identical to EncodeScrollArgs (backward-compatible: an old encoder/decoder
// is unaffected, and the plain DecodeScrollArgs ignores any trailing bytes). The
// afterID:u64 tail (with its cursorPresent byte) is appended only when hasAfter,
// so an opts-only-no-cursor arg stays byte-identical to the legacy
// EncodeScrollArgsOpts trailer ([1][rc][opa]) — see DecodeScrollArgsCursor for
// the read side.
func EncodeScrollArgsCursor(collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool) []byte {
	return EncodeScrollArgsCursorBounded(collection, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, 0)
}

// EncodeScrollArgsCursorBounded is EncodeScrollArgsCursor plus the optional 8-byte
// staleness bound, which rides at the VERY END of the trailer (after the cursor
// block) and ONLY when rc==ConsistencyBoundedStaleness — so it never shifts the
// cursor offset and stays byte-identical for rc∈{0,1,2}. See DecodeScrollArgsCursor
// for the symmetric read.
func EncodeScrollArgsCursorBounded(collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, bound uint64) []byte {
	base := EncodeScrollArgs(collection, filter, limit)
	if readConsistency == 0 && onPartitionUnavailable == 0 && !hasAfter {
		return base // byte-identical to the legacy form
	}
	// Opts trailer (byte-identical to the legacy EncodeScrollArgsOpts form):
	// [optsPresent:u8=1][rc:u8][opa:u8]. The bound (8 bytes) rides immediately
	// after [rc][opa] ONLY when rc==3 — at the SAME position both DecodeScrollArgsCursor
	// and DecodeScrollArgsOrder read it, BEFORE the cursor + order blocks, so it never
	// shifts the order block decode. The cursor tail ([cursorPresent:u8=1][afterID:u64])
	// is appended ONLY when hasAfter, so an opts-only-no-cursor arg stays byte-identical
	// to the pre-cursor encoder for rc∈{0,1,2}.
	out := append(base, 1, readConsistency, onPartitionUnavailable)
	out = appendBoundTail(out, readConsistency, bound)
	if hasAfter {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], afterID)
		out = append(append(out, 1), idb[:]...)
	}
	return out
}

// DecodeScrollArgsOpts decodes a vector_scroll request that may carry a
// self-delimiting consistency opts (+ optional cursor) trailer. Backward-compatible:
// legacy args (no trailer) decode with readConsistency=0, onPartitionUnavailable=0.
func DecodeScrollArgsOpts(args []byte) (collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, err error) {
	collection, filter, limit, readConsistency, onPartitionUnavailable, _, _, _, err = DecodeScrollArgsCursor(args)
	return collection, filter, limit, readConsistency, onPartitionUnavailable, err
}

// DecodeScrollArgsCursor decodes a vector_scroll request that may carry the
// opts (+ optional resume-after-id cursor) trailer written by
// EncodeScrollArgsCursor. Backward-compatible:
//
//   - no trailer (legacy / no-opts-no-cursor): rc=opa=0, hasAfter=false.
//   - opts-only trailer ([1][rc][opa]) WITHOUT the cursorPresent byte (the
//     pre-cursor EncodeScrollArgsOpts form): rc/opa read, hasAfter=false.
//   - full trailer ([1][rc][opa][cursorPresent][afterID?]): all read.
//
// A present-flag with a missing/truncated trailer is corruption — fail loud.
func DecodeScrollArgsCursor(args []byte) (collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, bound uint64, err error) {
	collection, filter, limit, n, err := decodeScrollArgsN(args)
	if err != nil {
		return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, err
	}
	if len(args) > n && args[n] != 0 {
		// Presence flag set is the contract that [rc:u8][opa:u8] follows; a
		// missing trailer means corruption/truncation. Fail loud.
		if len(args) < n+3 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[n+1]
		onPartitionUnavailable = args[n+2]
		off := n + 3
		// The bound rides immediately after [rc][opa] ONLY when rc==3, BEFORE the
		// cursor block (symmetric with EncodeScrollArgsCursorBounded).
		bound, off, err = readBoundTail(args, off, readConsistency)
		if err != nil {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, err
		}
		// Optional cursorPresent byte (absent in the pre-cursor opts-only form).
		if len(args) > off && args[off] != 0 {
			if len(args) < off+1+8 {
				return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, ErrVectorArgsTruncated
			}
			afterID = binary.BigEndian.Uint64(args[off+1:])
			hasAfter = true
		}
	}
	return collection, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, bound, nil
}

// ScrollOrder carries an order_by scroll's ordering across the args wire. It is the
// ops mirror of vector.OrderBy plus the resume value (the resume id rides the
// existing afterID/hasAfter cursor fields):
//
//   - Key/Desc/IsDatetime/StartFrom/HasStart  mirror vector.OrderBy.
//   - Kind                                     the order value-kind (numeric/datetime/
//     string), mirroring vector.OrderBy.Kind. OrderString switches the leaf onto the
//     lexicographic (stringValue, id) path + the v3 string resume below; the
//     numeric/datetime float64 path is unchanged (Kind defaults to OrderNumeric).
//   - ResumeKey/HasResume                     the NUMERIC/DATETIME resume cursor's order
//     VALUE; paired with the args' afterID it is the (value, id) lower bound the leaf
//     seeks past. HasResume == hasAfter for a v2 cursor; page 1 has neither.
//   - ResumeStr/HasResumeStr                   the STRING (Kind==OrderString) resume
//     cursor's order STRING value (the v3 analogue of ResumeKey); paired with afterID
//     it is the (stringValue, id) lower bound. HasResumeStr == hasAfter for a v3 cursor.
//
// A nil *ScrollOrder means the plain id-ascending scroll (no order_by) — the wire
// is then BYTE-IDENTICAL to EncodeScrollArgsCursor (see EncodeScrollArgsOrder).
type ScrollOrder struct {
	Key          string
	Desc         bool
	IsDatetime   bool
	Kind         vtypes.OrderKind
	StartFrom    float64
	HasStart     bool
	ResumeKey    float64
	HasResume    bool
	ResumeStr    string
	HasResumeStr bool
	// Tail carries the MULTI-KEY secondary/tertiary sort keys (Tail[0] is the 2nd key,
	// etc.); the ScrollOrder itself is the PRIMARY (key[0]). EMPTY ⇒ the single-key path
	// (byte/behaviour-identical: v2/v3 cursor, the existing single-key block format). A
	// non-empty Tail switches onto the v4 tuple cursor + the multi-key tail block (the
	// per-key specs + the resume tuple ride ResumeKeys below). Each tail key carries its
	// own Key/Desc/IsDatetime/Kind; StartFrom/Resume* are primary-only (the tuple resume
	// is ResumeKeys).
	Tail []ScrollOrderKey
	// ResumeKeys is the v4 multi-key resume TUPLE (one ScrollOrderVal per key, index 0 the
	// primary). Only meaningful when len(Tail)>0 && HasResumeKeys; the page resumes
	// strictly after the (k1,…,kN, afterID) tuple position. Page 1 has neither.
	ResumeKeys    []ScrollOrderVal
	HasResumeKeys bool
}

// ScrollOrderKey is one SECONDARY/tertiary sort key's spec in a multi-key ScrollOrder.Tail:
// the per-key field name + direction + kind (the tail analogue of the primary's
// Key/Desc/IsDatetime/Kind fields). Tail keys carry no StartFrom/Resume — the resume
// tuple is ScrollOrder.ResumeKeys.
type ScrollOrderKey struct {
	Key        string
	Desc       bool
	IsDatetime bool
	Kind       vtypes.OrderKind
}

// ScrollOrderVal is one typed resume value in ScrollOrder.ResumeKeys (the v4 cursor's
// resume tuple element): a float64 (Num) for a numeric/datetime key or a string (Str)
// for a string key, tagged by Kind. The ops-args analogue of vector.OrderVal /
// ops.OrderKeyVal so the multi-key resume tuple rides the scroll args wire alongside the
// per-key specs in Tail.
type ScrollOrderVal struct {
	Num  float64
	Str  string
	Kind vtypes.OrderKind
}

// scrollOrderFlagDesc / scrollOrderFlagDatetime / scrollOrderFlagString are the order
// block's flags byte bits (bit0=desc, bit1=isDatetime, bit2=Kind==OrderString). bit2
// is purely additive: an old (pre-string) block never sets it, so a numeric/datetime
// block stays byte-identical and decodes with Kind=OrderNumeric/OrderDatetime.
const (
	scrollOrderFlagDesc     byte = 1 << 0
	scrollOrderFlagDatetime byte = 1 << 1
	scrollOrderFlagString   byte = 1 << 2
	// scrollOrderFlagMultiKey (bit3) marks the PRIMARY block of a MULTI-KEY order: the
	// additive tail block (the secondary key specs + the resume tuple) follows the
	// single-key block. bit3 is purely additive — a single-key order never sets it, so a
	// single-key block is byte-identical to the pre-multi-key codec and stops after the
	// (string) resume tail. Set ONLY on the primary's flags byte.
	scrollOrderFlagMultiKey byte = 1 << 3
)

// scrollOrderValKindString is the per-key kind byte for a string resume value in the
// multi-key tail block (mirrors vector.OrderString == 2). Numeric (0) / datetime (1)
// values carry a float64.
const scrollOrderValKindString = byte(vtypes.OrderString)

// EncodeScrollArgsOrder is EncodeScrollArgsCursor with an ADDITIVE order_by block.
//
//	... (EncodeScrollArgsCursor bytes) [orderPresent:u8]
//	  iff orderPresent==1:
//	    [keyLen:u32][key][flags:u8 bit0=desc bit1=isDatetime]
//	    [startPresent:u8][start:f64 iff startPresent]
//	    [resumePresent:u8][resumeKey:f64 iff resumePresent]
//
// When order == nil the order block is OMITTED ENTIRELY (not even a zero byte), so
// the output is BYTE-IDENTICAL to EncodeScrollArgsCursor(...) — the no-order_by path
// is zero-overhead on the wire (asserted in tests). When order != nil the opts+cursor
// trailer is FORCED present (even if rc=opa=0 and no cursor) so the order block has an
// unambiguous, self-delimiting start position after the cursor trailer.
func EncodeScrollArgsOrder(collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, order *ScrollOrder) []byte {
	return EncodeScrollArgsOrderBounded(collection, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, order, 0)
}

// EncodeScrollArgsOrderBounded is EncodeScrollArgsOrder plus the optional 8-byte
// staleness bound, which rides immediately after [rc][opa] (BEFORE the cursor +
// order blocks) ONLY when rc==ConsistencyBoundedStaleness — the SAME position
// EncodeScrollArgsCursorBounded uses, so DecodeScrollArgsCursor / DecodeScrollArgsOrder
// read it identically and the order block decode is undisturbed.
func EncodeScrollArgsOrderBounded(collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, order *ScrollOrder, bound uint64) []byte {
	if order == nil {
		return EncodeScrollArgsCursorBounded(collection, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, bound)
	}
	// FORCE the opts+cursor trailer present so the order block's start is unambiguous:
	// [optsPresent=1][rc][opa](+[bound:u64 iff rc==3])[cursorPresent][afterID?]. The
	// bound rides right after [rc][opa] (before the cursor byte), matching the cursor codec.
	base := EncodeScrollArgs(collection, filter, limit)
	out := append(base, 1, readConsistency, onPartitionUnavailable)
	out = appendBoundTail(out, readConsistency, bound)
	if hasAfter {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], afterID)
		out = append(append(out, 1), idb[:]...)
	} else {
		out = append(out, 0) // cursorPresent=0
	}
	return appendScrollOrderBlock(out, order)
}

// appendScrollOrderBlock appends the self-delimiting order_by block to out:
//
//	[orderPresent:u8]
//	  iff orderPresent==1:
//	    [keyLen:u32][key][flags:u8 bit0=desc bit1=isDatetime bit2=string]
//	    [startPresent:u8][start:f64 iff startPresent]
//	    [resumePresent:u8][resumeKey:f64 iff resumePresent]
//	    iff flags bit2 (Kind==OrderString):  // ADDITIVE string-resume tail
//	      [resumeStrPresent:u8][strLen:u32][strBytes...] iff resumeStrPresent
//
// The string-resume tail is written ONLY when Kind==OrderString (flags bit2). A
// numeric/datetime block never sets bit2 and never writes the tail, so its bytes are
// IDENTICAL to the pre-string codec (asserted in tests). order == nil writes nothing
// at all (the caller decides whether an explicit orderPresent=0 marker is needed).
// This is the SHARED block written by the dense (EncodeScrollArgsOrder), named
// (EncodeNamedScrollArgsOrder) and MV (EncodeMVScrollArgsOrder) codecs so all three
// families agree on the wire byte-for-byte.
func appendScrollOrderBlock(out []byte, order *ScrollOrder) []byte {
	if order == nil {
		return out
	}
	out = append(out, 1) // orderPresent=1
	var kl [4]byte
	binary.BigEndian.PutUint32(kl[:], uint32(len(order.Key))) //nolint:gosec // key is a short field name
	out = append(out, kl[:]...)
	out = append(out, order.Key...)
	var flags byte
	if order.Desc {
		flags |= scrollOrderFlagDesc
	}
	if order.IsDatetime {
		flags |= scrollOrderFlagDatetime
	}
	if order.Kind == vtypes.OrderString {
		flags |= scrollOrderFlagString
	}
	if len(order.Tail) > 0 {
		flags |= scrollOrderFlagMultiKey
	}
	out = append(out, flags)
	if order.HasStart {
		var sb [8]byte
		binary.BigEndian.PutUint64(sb[:], math.Float64bits(order.StartFrom))
		out = append(append(out, 1), sb[:]...)
	} else {
		out = append(out, 0)
	}
	if order.HasResume {
		var rb [8]byte
		binary.BigEndian.PutUint64(rb[:], math.Float64bits(order.ResumeKey))
		out = append(append(out, 1), rb[:]...)
	} else {
		out = append(out, 0)
	}
	// ADDITIVE string-resume tail, ONLY for OrderString (bit2). Numeric/datetime
	// blocks stop above ⇒ their bytes are unchanged.
	if order.Kind == vtypes.OrderString {
		if order.HasResumeStr {
			var sl [4]byte
			binary.BigEndian.PutUint32(sl[:], uint32(len(order.ResumeStr))) //nolint:gosec // bounded by the cursor codec's strLen cap
			out = append(append(out, 1), sl[:]...)
			out = append(out, order.ResumeStr...)
		} else {
			out = append(out, 0)
		}
	}
	// ADDITIVE multi-key tail block, present ONLY when flags bit3 (len(Tail)>0). A
	// single-key block never reaches here ⇒ its bytes are unchanged. Layout:
	//
	//	[numTail:u8]
	//	  per tail key: [keyLen:u32][key][flags:u8 bit0=desc bit1=datetime bit2=string]
	//	[resumeTuplePresent:u8]
	//	  iff present: per FULL key (primary + numTail): [kind:u8][value]
	//	    kind==string (2): [strLen:u32][strBytes...]
	//	    else (numeric/datetime): [float64 BE]
	if len(order.Tail) > 0 {
		out = append(out, byte(len(order.Tail))) //nolint:gosec // tail arity bounded by the cursor codec's max keys
		for _, tk := range order.Tail {
			var kl2 [4]byte
			binary.BigEndian.PutUint32(kl2[:], uint32(len(tk.Key))) //nolint:gosec // key is a short field name
			out = append(out, kl2[:]...)
			out = append(out, tk.Key...)
			var tf byte
			if tk.Desc {
				tf |= scrollOrderFlagDesc
			}
			if tk.IsDatetime {
				tf |= scrollOrderFlagDatetime
			}
			if tk.Kind == vtypes.OrderString {
				tf |= scrollOrderFlagString
			}
			out = append(out, tf)
		}
		if order.HasResumeKeys {
			out = append(out, 1)
			for _, rv := range order.ResumeKeys {
				out = append(out, byte(rv.Kind))
				if rv.Kind == vtypes.OrderString {
					var sl [4]byte
					binary.BigEndian.PutUint32(sl[:], uint32(len(rv.Str))) //nolint:gosec // bounded by the cursor codec's strLen cap
					out = append(out, sl[:]...)
					out = append(out, rv.Str...)
				} else {
					var vb [8]byte
					binary.BigEndian.PutUint64(vb[:], math.Float64bits(rv.Num))
					out = append(out, vb[:]...)
				}
			}
		} else {
			out = append(out, 0)
		}
	}
	return out
}

// readScrollOrderBlock reads the order_by block written by appendScrollOrderBlock,
// starting at args[off] (which must be the orderPresent byte). It returns the decoded
// *ScrollOrder (nil when orderPresent==0 or off is past the end), the new offset, and
// an error on a truncated block. The SHARED read side for all three scroll families.
func readScrollOrderBlock(args []byte, off int) (order *ScrollOrder, newOff int, err error) {
	if len(args) <= off || args[off] == 0 {
		// No order block (off==len, or an explicit orderPresent=0 marker).
		if len(args) > off {
			off++ // consume the orderPresent=0 byte
		}
		return nil, off, nil
	}
	off++ // orderPresent=1
	o := &ScrollOrder{}
	if len(args) < off+4 {
		return nil, off, ErrVectorArgsTruncated
	}
	kl := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if kl < 0 || kl > len(args)-off-1 {
		return nil, off, ErrVectorArgsTruncated
	}
	o.Key = string(args[off : off+kl])
	off += kl
	flags := args[off]
	off++
	o.Desc = flags&scrollOrderFlagDesc != 0
	o.IsDatetime = flags&scrollOrderFlagDatetime != 0
	isString := flags&scrollOrderFlagString != 0
	isMultiKey := flags&scrollOrderFlagMultiKey != 0
	// Kind tracks ONLY the string vs float64 distinction on the wire: bit2 ⇒
	// OrderString, else OrderNumeric (the zero value). The numeric/datetime float64
	// path is byte/behaviour-identical to the pre-string codec — datetime-ness is
	// carried by IsDatetime as before, NOT by Kind, so a numeric/datetime block
	// decodes with Kind==OrderNumeric exactly as an old block did (the engine branches
	// on OrderString only; OrderDatetime is informational on the request side).
	if isString {
		o.Kind = vtypes.OrderString
	}
	// startPresent[+start]
	if len(args) < off+1 {
		return nil, off, ErrVectorArgsTruncated
	}
	if args[off] != 0 {
		if len(args) < off+1+8 {
			return nil, off, ErrVectorArgsTruncated
		}
		o.StartFrom = math.Float64frombits(binary.BigEndian.Uint64(args[off+1:]))
		o.HasStart = true
		off += 1 + 8
	} else {
		off++
	}
	// resumePresent[+resumeKey]
	if len(args) < off+1 {
		return nil, off, ErrVectorArgsTruncated
	}
	if args[off] != 0 {
		if len(args) < off+1+8 {
			return nil, off, ErrVectorArgsTruncated
		}
		o.ResumeKey = math.Float64frombits(binary.BigEndian.Uint64(args[off+1:]))
		o.HasResume = true
		off += 1 + 8
	} else {
		off++
	}
	// ADDITIVE string-resume tail, present ONLY for OrderString (bit2). A
	// numeric/datetime block stops above; off==len there and this branch is skipped.
	if isString {
		if len(args) < off+1 {
			return nil, off, ErrVectorArgsTruncated
		}
		if args[off] != 0 {
			off++
			if len(args) < off+4 {
				return nil, off, ErrVectorArgsTruncated
			}
			sl := int(binary.BigEndian.Uint32(args[off:]))
			off += 4
			if sl < 0 || len(args)-off < sl {
				return nil, off, ErrVectorArgsTruncated
			}
			o.ResumeStr = string(args[off : off+sl])
			o.HasResumeStr = true
			off += sl
		} else {
			off++
		}
	}
	// ADDITIVE multi-key tail block, present ONLY when flags bit3 (isMultiKey). A
	// single-key block never sets bit3; off==len there and this branch is skipped. The
	// reader mirrors appendScrollOrderBlock's tail exactly and is FAIL-LOUD on any
	// truncation, an out-of-range arity, or an oversized strLen (no panic / no OOB).
	if isMultiKey {
		if len(args) < off+1 {
			return nil, off, ErrVectorArgsTruncated
		}
		numTail := int(args[off])
		off++
		// Full key count = primary + numTail; bound by the cursor codec's max-keys cap.
		if numTail < 1 || 1+numTail > ScrollCursorMaxKeys {
			return nil, off, ErrVectorArgsTruncated
		}
		o.Tail = make([]ScrollOrderKey, numTail)
		for i := 0; i < numTail; i++ {
			if len(args) < off+4 {
				return nil, off, ErrVectorArgsTruncated
			}
			kl2 := int(binary.BigEndian.Uint32(args[off:]))
			off += 4
			if kl2 < 0 || kl2 > len(args)-off-1 {
				return nil, off, ErrVectorArgsTruncated
			}
			tk := ScrollOrderKey{Key: string(args[off : off+kl2])}
			off += kl2
			tf := args[off]
			off++
			tk.Desc = tf&scrollOrderFlagDesc != 0
			tk.IsDatetime = tf&scrollOrderFlagDatetime != 0
			if tf&scrollOrderFlagString != 0 {
				tk.Kind = vtypes.OrderString
			} else if tk.IsDatetime {
				tk.Kind = vtypes.OrderDatetime
			}
			o.Tail[i] = tk
		}
		// resumeTuplePresent[+ per-FULL-key kind+value]
		if len(args) < off+1 {
			return nil, off, ErrVectorArgsTruncated
		}
		if args[off] != 0 {
			off++
			nKeys := 1 + numTail
			o.ResumeKeys = make([]ScrollOrderVal, nKeys)
			for i := 0; i < nKeys; i++ {
				if len(args) < off+1 {
					return nil, off, ErrVectorArgsTruncated
				}
				kind := args[off]
				off++
				if kind == scrollOrderValKindString {
					if len(args) < off+4 {
						return nil, off, ErrVectorArgsTruncated
					}
					sl := int(binary.BigEndian.Uint32(args[off:]))
					off += 4
					if sl < 0 || sl > scrollCursorStringMaxLen || len(args)-off < sl {
						return nil, off, ErrVectorArgsTruncated
					}
					o.ResumeKeys[i] = ScrollOrderVal{Str: string(args[off : off+sl]), Kind: vtypes.OrderString}
					off += sl
				} else {
					if len(args) < off+8 {
						return nil, off, ErrVectorArgsTruncated
					}
					o.ResumeKeys[i] = ScrollOrderVal{Num: math.Float64frombits(binary.BigEndian.Uint64(args[off:])), Kind: vtypes.OrderKind(kind)}
					off += 8
				}
			}
			o.HasResumeKeys = true
		} else {
			off++
		}
	}
	return o, off, nil
}

// DecodeScrollArgsOrder decodes args that MAY carry the additive order_by block
// written by EncodeScrollArgsOrder. It is a superset of DecodeScrollArgsCursor: it
// reads the same base + opts + cursor trailer, then (if present) the order block.
//
//   - No order block (legacy args, or order==nil at encode time): order is nil and
//     the result is identical to DecodeScrollArgsCursor.
//   - Order block present: order is non-nil with Key/Desc/IsDatetime/Start/Resume set.
//
// A present-flag with a missing/truncated order block is corruption — fail loud.
func DecodeScrollArgsOrder(args []byte) (collection string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, order *ScrollOrder, err error) {
	collection, filter, limit, n, err := decodeScrollArgsN(args)
	if err != nil {
		return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, err
	}
	off := n
	if len(args) > off && args[off] != 0 {
		if len(args) < off+3 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
		readConsistency = args[off+1]
		onPartitionUnavailable = args[off+2]
		off += 3
		// The bound rides right after [rc][opa] ONLY when rc==3 (symmetric with the
		// encoder); consume it so the cursor + order offsets stay correct.
		_, off, err = readBoundTail(args, off, readConsistency)
		if err != nil {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, err
		}
		// Optional cursorPresent byte (absent in the pre-cursor opts-only form).
		if len(args) > off && args[off] != 0 {
			if len(args) < off+1+8 {
				return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, ErrVectorArgsTruncated
			}
			afterID = binary.BigEndian.Uint64(args[off+1:])
			hasAfter = true
			off += 1 + 8
		} else if len(args) > off {
			off++ // consume the cursorPresent=0 byte
		}
		// Optional order block (shared format; see readScrollOrderBlock).
		order, _, err = readScrollOrderBlock(args, off)
		if err != nil {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, err
		}
	}
	return collection, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, order, nil
}

// EncodeDeleteByFilterArgs serializes a vector_delete_by_filter request.
// Wire: [colLen:u8][col][filterLen:u32][filterJSON].
func EncodeDeleteByFilterArgs(collection string, filter vtypes.Filter) []byte {
	fj, _ := json.Marshal(filter)
	buf := make([]byte, 1+len(collection)+4+len(fj))
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(len(fj))) //nolint:gosec
	off += 4
	copy(buf[off:], fj)
	return buf
}

// DecodeDeleteByFilterArgs reads args produced by EncodeDeleteByFilterArgs.
func DecodeDeleteByFilterArgs(args []byte) (string, vtypes.Filter, error) {
	if len(args) < 1 {
		return "", vtypes.Filter{}, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4 {
		return "", vtypes.Filter{}, ErrVectorArgsTruncated
	}
	collection := string(args[1 : 1+colLen])
	off := 1 + colLen
	flen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if flen < 0 || len(args)-off < flen {
		return "", vtypes.Filter{}, ErrVectorArgsTruncated
	}
	var filter vtypes.Filter
	if err := json.Unmarshal(args[off:off+flen], &filter); err != nil {
		return "", vtypes.Filter{}, fmt.Errorf("ops: decode filter: %w", err)
	}
	return collection, filter, nil
}

// EncodeVectorUpsertArgs serializes a vector_upsert request. It reuses the
// insert-args wire shape, folding the document content into the metadata's
// reserved content field so no separate wire field is needed.
func EncodeVectorUpsertArgs(collection string, id uint64, vec []float32, content string, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector) []byte {
	return EncodeVectorInsertArgsExt(collection, id, vec, ttl, vtypes.WithContent(meta, content), sparse)
}

// EncodeVectorUpsertArgsCAS serializes vector_upsert args carrying an
// optimistic-CAS precondition (see EncodeVectorInsertArgsCAS). When hasExpected
// is false the output is BYTE-IDENTICAL to EncodeVectorUpsertArgs.
func EncodeVectorUpsertArgsCAS(collection string, id uint64, vec []float32, content string, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, expectedVersion uint64, hasExpected bool) []byte {
	return EncodeVectorInsertArgsCAS(collection, id, vec, ttl, vtypes.WithContent(meta, content), sparse, expectedVersion, hasExpected)
}

// EncodeVectorUpsertArgsCASKeyTTL is EncodeVectorUpsertArgsCAS plus an OPTIONAL
// per-key payload TTL map (key -> RELATIVE ms). Byte-identical to
// EncodeVectorUpsertArgs when keyTTLMs is empty and hasExpected is false.
func EncodeVectorUpsertArgsCASKeyTTL(collection string, id uint64, vec []float32, content string, ttl time.Duration, meta vtypes.Metadata, sparse vtypes.SparseVector, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64) []byte {
	return EncodeVectorInsertArgsCASKeyTTL(collection, id, vec, ttl, vtypes.WithContent(meta, content), sparse, expectedVersion, hasExpected, keyTTLMs)
}

// MaxCollectionNameWire is the longest collection name the op wire can carry:
// the name's length is encoded in a SINGLE byte, so a 256-byte name would wrap
// modulo 256 and silently mis-decode or mis-route. Exported so a transport can
// reject an over-long name itself instead of relying on a guard in some other
// package.
const MaxCollectionNameWire = 255

// BulkStageRowLen is the on-wire size of one staged row for a dim-dimensional
// collection: [id u64][dim × f32]. Exported so a transport that receives rows
// already in this layout (the HTTP binary bulk path) can size and bound its
// reads from the same constant the encoder uses.
func BulkStageRowLen(dim int) int { return 8 + dim*4 }

// BulkStageArgsHeader returns JUST the header of a vector_bulk_stage args
// buffer — [colLen u8][col][dim u32][count u32] — for a request that will carry
// `count` rows of `dim` floats. It allocates only the header (tens of bytes),
// never anything sized by count.
//
// It exists so a transport that already holds the rows in exactly the op's
// layout — the HTTP binary bulk-ingest framing, which is big-endian for
// precisely this reason — can emit this header and then stream the rows in
// after it, with no per-float conversion and no intermediate [][]float32.
// EncodeBulkStageArgs uses it too, so the two paths CANNOT drift: there is one
// owner of the layout.
//
// PRECONDITION: len(collection) <= MaxCollectionNameWire. Callers that take a
// name from untrusted input must check it first (the HTTP path does, and does
// not rely on middleware in another package to have done it).
func BulkStageArgsHeader(collection string, dim, count int) []byte {
	if len(collection) > MaxCollectionNameWire {
		// Unreachable from any validated caller. Truncating the length byte here
		// would silently retarget the write at a DIFFERENT collection, so refuse
		// to produce a corrupt header at all.
		panic("ops: collection name exceeds the single-byte wire length")
	}
	buf := make([]byte, 1+len(collection)+4+4)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(dim)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(count)) //nolint:gosec
	return buf
}

// EncodeBulkStageArgs serializes a vector_bulk_stage request: a batch of
// (id, vector) pairs for one collection, every vector of the collection's
// dimension. Vectors-only (no TTL/metadata/sparse) — the bulk-load fast path.
// Wire: [colLen u8][col][dim u32][count u32]{ [id u64][dim×f32] }×count.
// The wire carries ONE dim for the whole batch, so a caller's vectors must all
// have that length. Three things went wrong when this was assumed rather than
// checked, all reachable from an untrusted JSON body:
//
//   - The buffer was sized len(ids)*(8+dim*4) with dim from vecs[0] and len(ids)
//     from the point array — INDEPENDENT numbers. One long vector plus many
//     vector-less points multiplied out to a buffer no body could justify: a
//     289 KB request allocated 1.61 GB, and the same shape scaled inside the body
//     cap reached runtime.throw, which no recover can catch.
//   - A later vector LONGER than vecs[0] wrote past the buffer and panicked.
//   - A MIDDLE SHORT vector left the offset behind and shifted every following
//     row, so ids were reconstructed out of vector bytes: encoding
//     ids [1 2 3] with vecs [[1 1 1 1] [9 9] [7 7 7 7]] decoded back as
//     ids [1 2 4674736414298996736], no error — a point stored under an id
//     nobody sent.
//
// The last one is why this check belongs HERE and cannot be delegated
// downstream: DecodeBulkStageArgs always materializes make([]float32, dim) per
// row, so anything past this function only ever sees uniform vectors.
// Raggedness is destroyed by the encoding itself, and a validator further down
// has nothing left to detect.
func EncodeBulkStageArgs(collection string, ids []uint64, vecs [][]float32) ([]byte, error) {
	if len(ids) != len(vecs) {
		return nil, fmt.Errorf("ops: bulk stage has %d ids and %d vectors", len(ids), len(vecs))
	}
	dim := 0
	if len(vecs) > 0 {
		dim = len(vecs[0])
	}
	// Raggedness is rejected BEFORE anything is sized. Skipping this was how
	// len(ids) and len(vecs[0]) — two numbers a JSON body chooses independently —
	// came to size one buffer: an anonymous 289 KB request reached 1.61 GB here,
	// and the same shape at body-cap scale reached runtime.throw.
	for i, v := range vecs {
		if len(v) != dim {
			return nil, fmt.Errorf("ops: bulk stage vector %d has length %d, batch dim is %d: %w",
				i, len(v), dim, vtypes.ErrDimMismatch)
		}
	}
	// Every vector is now known to hold dim floats, so len(ids)*perRow is bounded
	// by memory the caller already holds and can no longer be conjured from two
	// unrelated numbers. What remains is integer overflow, and CountFitsIn is
	// called rather than restated so this bound cannot drift from the one on the
	// op wire.
	//
	// BulkStageArgsHeader owns the layout; the rows are appended behind it, in the
	// same order the HTTP binary wire delivers them.
	hdr := BulkStageArgsHeader(collection, dim, len(ids))
	perRow := BulkStageRowLen(dim)
	if !CountFitsIn(len(ids), math.MaxInt-len(hdr), perRow) {
		return nil, fmt.Errorf("ops: bulk stage batch too large: %d rows of %d bytes", len(ids), perRow)
	}
	buf := make([]byte, len(hdr)+len(ids)*perRow)
	off := copy(buf, hdr)
	for i, id := range ids {
		binary.BigEndian.PutUint64(buf[off:], id)
		off += 8
		for _, f := range vecs[i] {
			binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
	}
	return buf, nil
}

// DecodeBulkStageArgs reads args produced by EncodeBulkStageArgs.
func DecodeBulkStageArgs(args []byte) (collection string, ids []uint64, vecs [][]float32, err error) {
	collection, ids, vecs, _, err = decodeBulkStageRows(args)
	return collection, ids, vecs, err
}

// decodeBulkStageRows reads the header + row region shared by vector_bulk_stage
// and vector_bulk_stage_payload, returning the offset just past the last row so
// the payload-bearing decoder can continue from there. There is ONE copy of the
// row bounds for the same reason there is one copy of the header layout: a
// second, restated bound drifts silently, and a drifted bound fails as an OOM
// rather than as a test.
func decodeBulkStageRows(args []byte) (collection string, ids []uint64, vecs [][]float32, off int, err error) {
	if len(args) < 1 {
		return "", nil, nil, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8 {
		return "", nil, nil, 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off = 1 + colLen
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	count := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// Bound count and the per-row size against the remaining buffer using only
	// division/subtraction so the check can never overflow. A crafted dim/count
	// near 2^32 would wrap the naive product count*(8+dim*4) to a small/negative
	// value, defeating the guard and forcing a multi-GB allocation or an
	// out-of-bounds read. Each row is [id u64][dim × f32] = 8 + dim*4 bytes, and
	// every row must fit in remaining, so reject when perRow > remaining/count.
	remaining := len(args) - off
	if dim < 0 || count < 0 {
		return "", nil, nil, 0, ErrVectorArgsTruncated
	}
	if count > 0 {
		// dim ≤ (remaining-8)/4 keeps perRow from overflowing before countFitsIn
		// can divide by it.
		if dim > (remaining-8)/4 {
			return "", nil, nil, 0, ErrVectorArgsTruncated
		}
		// CountFitsIn, not a fourth hand-rolled copy of it. A restated bound drifts
		// from the original silently, and a drifted bound fails as an OOM rather
		// than as a test.
		if !CountFitsIn(count, remaining, BulkStageRowLen(dim)) {
			return "", nil, nil, 0, ErrVectorArgsTruncated
		}
	}
	ids = make([]uint64, count)
	vecs = make([][]float32, count)
	for i := 0; i < count; i++ {
		ids[i] = binary.BigEndian.Uint64(args[off:])
		off += 8
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v[d] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
			off += 4
		}
		vecs[i] = v
	}
	return collection, ids, vecs, off, nil
}

// EncodeBulkStagePayloadArgs serializes a vector_bulk_stage_payload request: the
// EXACT vector_bulk_stage wire, plus a trailing per-point payload section.
//
//	[colLen u8][col][dim u32][count u32]
//	{ [id u64][dim×f32] } × count                 ← identical to vector_bulk_stage
//	{ [payLen u32][payLen bytes of JSON] } × count
//
// payLen 0 means "this point carries no payload", so a mixed batch costs 4 bytes
// per payload-less point and nothing else.
//
// WHY A SEPARATE OP RATHER THAN A FLAG ON vector_bulk_stage. Both are
// Raft-replicated, so the args are replayed by whatever binary a follower or a
// restarted node is running. DecodeBulkStageArgs stops at the last row and does
// not reject trailing bytes, so a payload section appended to the EXISTING op
// would be silently DISCARDED by any node that predates this change — the whole
// filter case loaded, replicated, and quietly unfilterable. An unknown op name
// fails loudly instead, which is the same trade binary_bulk.go makes for an
// unknown framing flag.
//
// The payload section's layout is byte-for-byte the HTTP binary framing's
// (binary_bulk.go, RVB1), for the same reason the row region is: /points/bulk
// streams both regions off the socket straight in behind the op header, with no
// re-encoding and no intermediate materialization of the corpus.
func EncodeBulkStagePayloadArgs(collection string, ids []uint64, vecs [][]float32, metas []vtypes.Metadata) ([]byte, error) {
	if len(metas) != len(ids) {
		return nil, fmt.Errorf("ops: bulk stage has %d ids and %d payloads", len(ids), len(metas))
	}
	// The row region is produced by the one owner of that layout, which also runs
	// the raggedness and overflow checks — see EncodeBulkStageArgs for the three
	// distinct ways a ragged batch corrupted this wire.
	rows, err := EncodeBulkStageArgs(collection, ids, vecs)
	if err != nil {
		return nil, err
	}
	blobs := make([][]byte, len(metas))
	total := 0
	for i, m := range metas {
		if len(m) == 0 {
			continue // len-0 span; nothing to marshal
		}
		b, merr := json.Marshal(m)
		if merr != nil {
			return nil, fmt.Errorf("ops: encode bulk stage payload %d: %w", i, merr)
		}
		blobs[i] = b
		if total > math.MaxInt-len(b) {
			return nil, fmt.Errorf("ops: bulk stage payload section too large")
		}
		total += len(b)
	}
	// Every payload also costs its 4-byte length prefix. Checked against the
	// remaining int budget rather than assumed, so the append below cannot be sized
	// by an overflowed sum.
	if !CountFitsIn(len(metas), math.MaxInt-len(rows)-total, 4) {
		return nil, fmt.Errorf("ops: bulk stage payload section too large")
	}
	buf := make([]byte, len(rows), len(rows)+total+4*len(metas))
	copy(buf, rows)
	var lenBuf [4]byte
	for _, b := range blobs {
		binary.BigEndian.PutUint32(lenBuf[:], uint32(len(b))) //nolint:gosec // bounded above
		buf = append(buf, lenBuf[:]...)
		buf = append(buf, b...)
	}
	return buf, nil
}

// DecodeBulkStagePayloadArgs reads args produced by EncodeBulkStagePayloadArgs.
// metas has one entry per staged point, nil where the point carries no payload.
func DecodeBulkStagePayloadArgs(args []byte) (collection string, ids []uint64, vecs [][]float32, metas []vtypes.Metadata, err error) {
	collection, ids, vecs, off, err := decodeBulkStageRows(args)
	if err != nil {
		return "", nil, nil, nil, err
	}
	count := len(ids)
	// Every point's payload costs AT LEAST its 4-byte length prefix, so the
	// section cannot be shorter than 4*count however small the blobs are. Checking
	// that before make() means the per-point slice is sized by bytes that were
	// actually delivered, not by the count word in the header.
	//
	// THIS CHECK DOES NOT SUBSUME THE PER-ITERATION ONE BELOW, AND THE ASYMMETRY
	// WITH THE ROW REGION IS THE TRAP. A row is a FIXED BulkStageRowLen(dim), so
	// `count*perRow <= remaining` proves every row read is in bounds and one
	// up-front check is genuinely enough. A payload is a VARIABLE 4+blobLen, so
	// the same arithmetic proves nothing about iteration i: one oversized blob
	// legitimately consumes the budget the later blobs' prefixes were counted
	// against. Concretely, count=2 with an 8-byte section holding [len=4]["null"]
	// satisfies CountFitsIn(2, 8, 4) exactly, and i=1 then read a 4-byte prefix
	// out of an EMPTY remainder.
	//
	// That was a panic, and a panic here is not a 500. shard/store.go proposes a
	// read-write op to raft WITHOUT decoding it; the frame is COMMITTED and then
	// decoded for the first time inside the FSM apply goroutine, where no recover
	// exists (see the note on ApplyBatch in shard/fsm.go) — so every replica dies
	// on the same durable entry and dies again on replay. One malformed frame from
	// any client holding an ordinary write scope is a permanent cluster-wide crash
	// loop. Both checks stay: this one guards the allocation, the one below guards
	// the read.
	if !CountFitsIn(count, len(args)-off, 4) {
		return "", nil, nil, nil, ErrVectorArgsTruncated
	}
	metas = make([]vtypes.Metadata, count)
	for i := 0; i < count; i++ {
		// The length prefix itself must be proven present before it is READ — see
		// the asymmetry note above. This is the check whose absence panicked.
		if len(args)-off < 4 {
			return "", nil, nil, nil, ErrVectorArgsTruncated
		}
		blobLen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		// blobLen is validated against what REMAINS before it is used to slice.
		// int(uint32) is only negative on a 32-bit build, where a length near 2^32
		// lands negative and would sail past the `>` bound.
		if blobLen < 0 || blobLen > len(args)-off {
			return "", nil, nil, nil, ErrVectorArgsTruncated
		}
		if blobLen == 0 {
			continue // no payload for this point
		}
		m := make(vtypes.Metadata)
		if uerr := json.Unmarshal(args[off:off+blobLen], &m); uerr != nil {
			return "", nil, nil, nil, fmt.Errorf("%w: payload %d: %v", ErrMalformedPayloadJSON, i, uerr)
		}
		// A blob of `null` unmarshals into a map with NO error and leaves it nil,
		// so without this it would be accepted as a payload that is neither absent
		// (len 0, the one canonical spelling) nor an object. Rejecting it keeps
		// ErrMalformedPayloadJSON's contract honest — "framed correctly but not a
		// metadata object" — and leaves exactly one way to say "no payload".
		if m == nil {
			return "", nil, nil, nil, fmt.Errorf("%w: payload %d: JSON null is not a metadata object "+
				"(use a zero length for a point with no payload)", ErrMalformedPayloadJSON, i)
		}
		metas[i] = m
		off += blobLen
	}
	// Trailing bytes mean the sender framed the request differently than it was
	// just read, so accepting the prefix would stage points against payloads that
	// belong to other points. The HTTP binary framing is equally strict (expectEOF).
	if off != len(args) {
		return "", nil, nil, nil, ErrVectorArgsTruncated
	}
	return collection, ids, vecs, metas, nil
}

// EncodeBulkBuildArgs serializes a vector_bulk_build request.
// Wire: [colLen u8][col][workers u32] (workers 0 = GOMAXPROCS).
func EncodeBulkBuildArgs(collection string, workers int) []byte {
	buf := make([]byte, 1+len(collection)+4)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(workers)) //nolint:gosec
	return buf
}

// DecodeBulkBuildArgs reads args produced by EncodeBulkBuildArgs.
func DecodeBulkBuildArgs(args []byte) (collection string, workers int, err error) {
	if len(args) < 1 {
		return "", 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4 {
		return "", 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	workers = int(binary.BigEndian.Uint32(args[1+colLen:])) //nolint:gosec
	return collection, workers, nil
}

// DecodeDeleteByFilterResult reads the count returned by vector_delete_by_filter.
func DecodeDeleteByFilterResult(body []byte) (int, error) {
	if len(body) < 4 {
		return 0, ErrVectorArgsTruncated
	}
	return int(binary.BigEndian.Uint32(body[0:4])), nil
}

// EncodeGroupSearchArgs serializes a vector_search_groups request. Wire:
//
//	[colLen:u8][col][k:u32][groupSize:u32][fetchK:u32]
//	[groupByLen:u16][groupBy][dim:u32][query][filterLen:u32][filterJSON]
//
// filterLen 0 means no filter (so no flags byte is needed).
func EncodeGroupSearchArgs(collection string, k int, query []float32, opts vtypes.GroupOpts) []byte {
	var filterJSON []byte
	if !opts.Filter.IsZero() {
		filterJSON, _ = json.Marshal(opts.Filter)
	}
	n := 1 + len(collection) + 4 + 4 + 4 + 2 + len(opts.GroupBy) + 4 + 4*len(query) + 4 + len(filterJSON)
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(k)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(opts.GroupSize)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(opts.FetchK)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint16(buf[off:], uint16(len(opts.GroupBy))) //nolint:gosec
	off += 2
	off += copy(buf[off:], opts.GroupBy)
	binary.BigEndian.PutUint32(buf[off:], uint32(len(query))) //nolint:gosec
	off += 4
	for _, f := range query {
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
		off += 4
	}
	binary.BigEndian.PutUint32(buf[off:], uint32(len(filterJSON))) //nolint:gosec
	off += 4
	copy(buf[off:], filterJSON)
	return buf
}

// DecodeGroupSearchArgs reads args produced by EncodeGroupSearchArgs. Trailing
// bytes beyond the base block (e.g. an opts trailer from
// EncodeGroupSearchArgsOpts) are ignored, so the single-shard handler stays
// backward-compatible with opts-carrying args.
func DecodeGroupSearchArgs(args []byte) (collection string, k int, query []float32, opts vtypes.GroupOpts, err error) {
	collection, k, query, opts, _, err = decodeGroupSearchArgsN(args)
	return collection, k, query, opts, err
}

// decodeGroupSearchArgsN decodes the group-search base block and returns the
// number of bytes consumed (so DecodeGroupSearchArgsOpts can read a trailing
// opts block). Group args carry no flags byte, so the Opts trailer is
// self-delimiting (see EncodeGroupSearchArgsOpts).
func decodeGroupSearchArgsN(args []byte) (collection string, k int, query []float32, opts vtypes.GroupOpts, n int, err error) {
	fail := func() (string, int, []float32, vtypes.GroupOpts, int, error) {
		return "", 0, nil, vtypes.GroupOpts{}, 0, ErrVectorArgsTruncated
	}
	if len(args) < 1 {
		return fail()
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4+4+4+2 {
		return fail()
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.GroupSize = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.FetchK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	gbLen := int(binary.BigEndian.Uint16(args[off:]))
	off += 2
	if gbLen < 0 || gbLen > len(args)-off-4 {
		return fail()
	}
	opts.GroupBy = string(args[off : off+gbLen])
	off += gbLen
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if !CountFitsIn(dim, len(args)-off, 4) {
		return fail()
	}
	query = make([]float32, dim)
	for i := 0; i < dim; i++ {
		query[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	if len(args) < off+4 {
		return fail()
	}
	flen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if flen < 0 || len(args)-off < flen {
		return fail()
	}
	if flen > 0 {
		if uerr := json.Unmarshal(args[off:off+flen], &opts.Filter); uerr != nil {
			return "", 0, nil, vtypes.GroupOpts{}, 0, fmt.Errorf("ops: decode filter: %w", uerr)
		}
	}
	off += flen
	return collection, k, query, opts, off, nil
}

// EncodeGroupSearchArgsOpts serializes vector_search_groups args plus an optional
// cross-shard consistency opts trailer. Group args carry no flags byte, so the
// trailer is self-delimiting: [optsPresent:u8][rc:u8][opa:u8], appended ONLY
// when readConsistency!=0 || onPartitionUnavailable!=0. When both are zero the
// output is byte-identical to EncodeGroupSearchArgs (backward-compatible) and
// the plain DecodeGroupSearchArgs (used by the single-shard handler) ignores any
// trailing bytes.
func EncodeGroupSearchArgsOpts(collection string, k int, query []float32, opts vtypes.GroupOpts, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	base := EncodeGroupSearchArgs(collection, k, query, opts)
	if readConsistency == 0 && onPartitionUnavailable == 0 {
		return base // byte-identical to the legacy form
	}
	out := append(base, 1, readConsistency, onPartitionUnavailable)
	return appendBoundTail(out, readConsistency, bound) // 8 bound bytes ride ONLY when rc==BoundedStaleness
}

// DecodeGroupSearchArgsOpts decodes vector_search_groups args that may carry a
// self-delimiting consistency opts trailer. Backward-compatible: legacy args
// (no trailer) decode with readConsistency=0, onPartitionUnavailable=0.
func DecodeGroupSearchArgsOpts(args []byte) (collection string, k int, query []float32, opts vtypes.GroupOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, k, query, opts, n, err := decodeGroupSearchArgsN(args)
	if err != nil {
		return "", 0, nil, vtypes.GroupOpts{}, 0, 0, 0, err
	}
	if len(args) > n && args[n] != 0 {
		// Presence flag set is the contract that [rc:u8][opa:u8] follows; a
		// missing trailer means corruption/truncation. Fail loud rather than
		// silently downgrading an explicit LeaderOnly/Fail request.
		if len(args) < n+3 {
			return "", 0, nil, vtypes.GroupOpts{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[n+1]
		onPartitionUnavailable = args[n+2]
		bound, _, err = readBoundTail(args, n+3, readConsistency)
		if err != nil {
			return "", 0, nil, vtypes.GroupOpts{}, 0, 0, 0, err
		}
	}
	return collection, k, query, opts, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeGroups serializes group-by-document results. Wire:
//
//	[count:u32] then per group: [keyLen:u32][keyJSON][docsLen:u32][docsBlob]
//
// where docsBlob is the EncodeVectorDocs encoding of the group's hits and
// keyJSON is the JSON-marshaled group key (a vector.Value).
func EncodeGroups(groups []vtypes.Group) []byte {
	keys := make([][]byte, len(groups))
	docs := make([][]byte, len(groups))
	n := 4
	for i, g := range groups {
		keys[i], _ = json.Marshal(g.Key)
		docs[i] = EncodeVectorDocs(g.Hits)
		n += 4 + len(keys[i]) + 4 + len(docs[i])
	}
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(groups))) //nolint:gosec
	off := 4
	for i := range groups {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(keys[i]))) //nolint:gosec
		off += 4
		off += copy(buf[off:], keys[i])
		binary.BigEndian.PutUint32(buf[off:], uint32(len(docs[i]))) //nolint:gosec
		off += 4
		off += copy(buf[off:], docs[i])
	}
	return buf
}

// DecodeGroups reads groups produced by EncodeGroups.
func DecodeGroups(body []byte) ([]vtypes.Group, error) {
	groups, _, err := decodeGroupsN(body)
	return groups, err
}

// decodeGroupsN decodes one groups block and returns the groups plus the number
// of bytes consumed (so callers can read a trailing degraded block).
func decodeGroupsN(body []byte) ([]vtypes.Group, int, error) {
	raw, off, err := frameGroupsN(body)
	if err != nil {
		return nil, 0, err
	}
	groups := make([]vtypes.Group, len(raw))
	for i, r := range raw {
		// Unmarshalling IS the validation of the key window, exactly as before.
		if err := json.Unmarshal(r.Key, &groups[i].Key); err != nil {
			return nil, 0, fmt.Errorf("ops: decode group key: %w", err)
		}
		hits, err := DecodeVectorDocs(r.hits)
		if err != nil {
			return nil, 0, err
		}
		groups[i].Hits = hits
	}
	return groups, off, nil
}

// decodeGroupsRawN is decodeGroupsN's raw twin: the same framing, with each group
// key checked for well-formedness instead of unmarshalled and each group's hits
// decoded by decodeVectorDocsRawN. See decodeVectorDocsRawN for the one shape
// difference a validity scan has against a full decode.
func decodeGroupsRawN(body []byte) ([]vtypes.RawGroup, int, error) {
	raw, off, err := frameGroupsN(body)
	if err != nil {
		return nil, 0, err
	}
	groups := make([]vtypes.RawGroup, len(raw))
	for i, r := range raw {
		// A group key is the json.Marshal of a single vector.Value, so it needs the
		// same vetting a metadata window does — including the U+FFFD escape case,
		// which a key holding invalid UTF-8 hits exactly as a payload does.
		key, err := checkRawGroupKeyJSON(r.Key)
		if err != nil {
			return nil, 0, fmt.Errorf("ops: decode group key: %w", err)
		}
		hits, _, err := decodeVectorDocsRawN(r.hits)
		if err != nil {
			return nil, 0, err
		}
		groups[i] = vtypes.RawGroup{Key: key, Hits: hits}
	}
	return groups, off, nil
}

// framedGroup is one group as it sits on the wire: its key's JSON bytes and its
// hits' document block, both windows INTO the source buffer and neither yet
// validated. frameGroupsN produces these; the typed and raw group decoders differ
// only in what they then do with the two windows.
type framedGroup struct {
	Key  []byte
	hits []byte
}

// frameGroupsN walks one groups block, returning each group's key and hits as
// windows into body. It is the single framing implementation behind both group
// decoders, for the same reason frameVectorDocsN is: the two can then never
// disagree about group boundaries, only about how a key and its hits are decoded.
func frameGroupsN(body []byte) ([]framedGroup, int, error) {
	if len(body) < 4 {
		return nil, 0, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// A group costs >= 4 bytes (its own leading length field).
	if !CountFitsIn(count, len(body)-off, 4) {
		return nil, 0, ErrVectorArgsTruncated
	}
	groups := make([]framedGroup, 0, count)
	for i := 0; i < count; i++ {
		if len(body) < off+4 {
			return nil, 0, ErrVectorArgsTruncated
		}
		klen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if klen < 0 || klen > len(body)-off-4 {
			return nil, 0, ErrVectorArgsTruncated
		}
		var g framedGroup
		g.Key = body[off : off+klen]
		off += klen
		dlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if dlen < 0 || len(body)-off < dlen {
			return nil, 0, ErrVectorArgsTruncated
		}
		g.hits = body[off : off+dlen]
		off += dlen
		groups = append(groups, g)
	}
	return groups, off, nil
}

// EncodeGroupsDegraded encodes groups with an optional degraded-partition
// trailer (same wire format as the search trailer). When degraded is false and
// missing is empty the output is byte-identical to EncodeGroups.
func EncodeGroupsDegraded(groups []vtypes.Group, degraded bool, missing []uint16) []byte {
	return appendDegradedTrailer(EncodeGroups(groups), degraded, missing)
}

// DecodeGroupsDegraded decodes groups and the optional degraded trailer.
// Backward-compatible with legacy EncodeGroups bytes.
func DecodeGroupsDegraded(body []byte) (groups []vtypes.Group, degraded bool, missing []uint16, err error) {
	groups, off, err := decodeGroupsN(body)
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, off)
	return groups, degraded, missing, err
}

// DecodeGroupsDegradedRaw is DecodeGroupsDegraded with raw (undecoded) group keys
// and hit metadata — see DecodeVectorDocsRaw for when that is the right choice and
// for the aliasing rule.
func DecodeGroupsDegradedRaw(body []byte) (groups []vtypes.RawGroup, degraded bool, missing []uint16, err error) {
	groups, off, err := decodeGroupsRawN(body)
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, off)
	return groups, degraded, missing, err
}

// --- get + payload-update codecs (dense) ---
//
// These wire the point-retrieve and in-place payload-mutation ops. ALL args lead
// with [colLen:u8][col][id:u64] so VectorKeyColAt1 routes them by collection name
// (the dense vector_delete routing template). The named/MV families mirror these
// shapes in named.go / multivector.go.

// EncodeVectorGetArgs serializes a vector_get / vector_named_get / vector_mv_get
// request. Wire: [colLen:u8][col][id:u64][getFlags:u8] where getFlags is a bit
// field (GetFlagWithVector|GetFlagWithPayload). Pass GetFlagsBoth for the common
// "fetch vector(s)+payload" case. The flags byte is ALWAYS present (the leading
// [colLen][col] keeps VectorKeyColAt1 routing intact).
func EncodeVectorGetArgs(collection string, id uint64, flags uint8) []byte {
	buf := make([]byte, 1+len(collection)+8+1)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	off += 8
	buf[off] = flags
	return buf
}

// DecodeVectorGetArgs reads args produced by EncodeVectorGetArgs (shared by all
// three get families — the wire shape is identical). Trailing bytes beyond the
// base block (the rc/opa opts trailer added by EncodeVectorGetArgsOpts) are
// IGNORED, so a single-shard get handler stays backward-compatible with
// rc-carrying args. Fails loud on truncation of the base block.
func DecodeVectorGetArgs(args []byte) (collection string, id uint64, flags uint8, err error) {
	collection, id, flags, _, err = decodeVectorGetArgsN(args)
	return collection, id, flags, err
}

// decodeVectorGetArgsN decodes the get base block and returns the number of bytes
// it consumed, so DecodeVectorGetArgsOpts can read a trailing self-delimiting opts
// block ([marker][rc][opa]). The base block — [colLen:u8][col][id:u64][flags:u8] —
// is fixed-length, so a trailing marker byte is unambiguous.
func decodeVectorGetArgsN(args []byte) (collection string, id uint64, flags uint8, n int, err error) {
	if len(args) < 1 {
		return "", 0, 0, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8+1 {
		return "", 0, 0, 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	id = binary.BigEndian.Uint64(args[off:])
	off += 8
	flags = args[off]
	off++
	return collection, id, flags, off, nil
}

// EncodeVectorGetArgsOpts is EncodeVectorGetArgs plus the self-delimiting
// [marker][rc][opa] opts trailer (read consistency). Shared by all three get
// families (dense / named / MV) exactly like EncodeVectorGetArgs. When
// rc==0 && opa==0 it is BYTE-IDENTICAL to EncodeVectorGetArgs (the AnyReplica
// default path is wire-unchanged); a non-zero rc rides the trailer so the shard
// can arm the readIndex barrier (via ops.ReadConsistencyOf) for a Linearizable get.
func EncodeVectorGetArgsOpts(collection string, id uint64, flags, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return AppendReadOptsTrailerBounded(EncodeVectorGetArgs(collection, id, flags), readConsistency, onPartitionUnavailable, bound)
}

// DecodeVectorGetArgsOpts decodes a get request that may carry the rc/opa opts
// trailer. Backward-compatible: legacy args (no trailer) decode with
// readConsistency=0, onPartitionUnavailable=0. A present marker with a truncated
// rc/opa block is corruption — fail loud (never a silent drop, so a Linearizable
// get never silently degrades to stale).
func DecodeVectorGetArgsOpts(args []byte) (collection string, id uint64, flags, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, id, flags, n, err := decodeVectorGetArgsN(args)
	if err != nil {
		return "", 0, 0, 0, 0, 0, err
	}
	readConsistency, onPartitionUnavailable, bound, err = DecodeReadOptsTrailerBounded(args, n)
	if err != nil {
		return "", 0, 0, 0, 0, 0, err
	}
	return collection, id, flags, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeVectorGetResult serializes a dense vector_get result. Wire:
//
//	[found:u8] then if found==1:
//	  [dim:u32][vec f32×dim]        ← dim 0 (no floats) when with_vector was off
//	  [ttlMs:u64]                   ← remaining TTL (0 = no expiry)
//	  [metaPresent:u8][?metaLen:u32][?metaJSON]   ← metaPresent 0 when no payload / payload off
//	  [sparsePresent:u8][?nnz:u32{[dim:u32][val:f32]}]  ← sparsePresent 0 when no sparse / payload off
//	  [?verPresent:u8][?version:u64]  ← OPTIONAL; written only by EncodeVectorGetResultV with version != 0
//
// not-found is the found=0 FLAG (NEVER an op error) so the fan-out layer can treat
// "absent in this partition" as expected. withVector/withPayload gate the vec and
// the meta+sparse projections respectively (mirroring the get flags).
func EncodeVectorGetResult(found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, withVector, withPayload bool) []byte {
	return EncodeVectorGetResultV(found, vec, meta, ttl, sparse, withVector, withPayload, 0)
}

// EncodeVectorGetResultV is EncodeVectorGetResult plus the point's per-point CAS
// version. The version rides in a trailing [verPresent:u8][?version:u64] block
// AFTER the sparse block. A version of 0 (an absent/unknown version) writes
// verPresent=0 and NO version field, so the output is BYTE-IDENTICAL to the
// pre-version EncodeVectorGetResult; a live version (>=1) writes verPresent=1 +
// the u64. decodeGetResultAt tolerates the block's absence (legacy bodies).
func EncodeVectorGetResultV(found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, withVector, withPayload bool, version uint64) []byte {
	return appendVectorGetResultV(nil, found, vec, meta, ttl, sparse, withVector, withPayload, version)
}

// appendVectorGetResultV appends the get-result record (same wire layout as
// EncodeVectorGetResultV) to dst and returns the grown slice. It sizes the record
// up front and grows dst ONCE, so it is allocation-free when dst already has the
// capacity — which is what lets the batch encoder serialize many rows into a
// single buffer without a throwaway slice per row. EncodeVectorGetResultV is the
// nil-dst single-get wrapper (one presized alloc, byte-identical to before).
func appendVectorGetResultV(dst []byte, found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, withVector, withPayload bool, version uint64) []byte {
	if !found {
		return append(dst, 0)
	}
	var metaJSON []byte
	if withPayload && len(meta) > 0 {
		metaJSON, _ = json.Marshal(meta)
	}
	includeVec := withVector && len(vec) > 0
	nnz := 0
	includeSparse := withPayload && sparse != nil && !sparse.IsZero()
	if includeSparse {
		nnz = len(sparse.Indices)
	}
	n := 1 + 4
	if includeVec {
		n += 4 * len(vec)
	}
	n += 8 + 1
	if len(metaJSON) > 0 {
		n += 4 + len(metaJSON)
	}
	n++
	if includeSparse {
		n += 4 + 8*nnz
	}
	if version != 0 {
		n += 1 + 8 // version present byte + u64
	}
	start := len(dst)
	dst = slices.Grow(dst, n)[:start+n]
	buf := dst[start:]
	buf[0] = 1
	off := 1
	if includeVec {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(vec))) //nolint:gosec
		off += 4
		for _, f := range vec {
			binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
	} else {
		binary.BigEndian.PutUint32(buf[off:], 0)
		off += 4
	}
	binary.BigEndian.PutUint64(buf[off:], uint64(ttl.Milliseconds())) //nolint:gosec // TTL >= 0
	off += 8
	if len(metaJSON) > 0 {
		buf[off] = 1
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metaJSON))) //nolint:gosec
		off += 4
		off += copy(buf[off:], metaJSON)
	} else {
		buf[off] = 0
		off++
	}
	if includeSparse {
		buf[off] = 1
		off++
		off = writeSparse(buf, off, *sparse)
	} else {
		buf[off] = 0
		off++
	}
	// Optional trailing version block: present (1 + u64) ONLY when version != 0,
	// so a zero/absent version is BYTE-IDENTICAL to the pre-version encoder.
	if version != 0 {
		buf[off] = 1
		off++
		binary.BigEndian.PutUint64(buf[off:], version)
	}
	return dst
}

// DecodeVectorGetResult reads a result produced by EncodeVectorGetResult. found is
// false for an absent/tombstoned/expired point (the not-found flag). vec/sparse are
// nil and meta is nil when their projection was off or empty.
func DecodeVectorGetResult(body []byte) (found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, err error) {
	found, vec, meta, ttl, sparse, _, _, err = decodeGetResultAt(body, 0, false)
	return found, vec, meta, ttl, sparse, err
}

// DecodeVectorGetResultV is DecodeVectorGetResult plus the point's per-point CAS
// version (0 when the result carries no version block — a legacy/no-version body
// or a not-found result).
func DecodeVectorGetResultV(body []byte) (found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, version uint64, err error) {
	found, vec, meta, ttl, sparse, version, _, err = decodeGetResultAt(body, 0, false)
	return found, vec, meta, ttl, sparse, version, err
}

// decodeGetResultAt decodes a single get-result record (the [found:u8]+body shape
// produced by EncodeVectorGetResultV) starting at body[off], returning the decoded
// fields and the offset just past the record. Shared by DecodeVectorGetResult (the
// single-get path) and DecodeVectorGetBatchResult (which reads one such record per
// row after the row's id) so the per-row wire layout stays defined in one place.
//
// versionFramed selects how the trailing version block is read:
//   - false (single get): the body holds exactly ONE record, so the OPTIONAL
//     version block is read only when bytes REMAIN after the sparse block. A
//     legacy/no-version body leaves no trailing bytes ⇒ version 0 (byte-identical).
//   - true (batch): each row's version block is ALWAYS framed ([verPresent:u8]
//     [?version:u64]) so the record self-delimits and the next row's id is found
//     unambiguously. The batch encoder always emits the present byte.
func decodeGetResultAt(body []byte, off int, versionFramed bool) (found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, version uint64, next int, err error) {
	found, vec, meta, ttl, sparse, version, next, _, err = decodeGetResultAtArena(body, off, versionFramed, nil)
	return found, vec, meta, ttl, sparse, version, next, err
}

// decodeGetResultAtArena is decodeGetResultAt with caller-owned storage for the
// decoded vector: the record's floats are appended to arena and vec is the
// just-appended window (a full three-index slice, so cap==len and a caller's
// append can never scribble into the next row's floats). A nil arena decodes
// exactly like the old per-record make([]float32, dim) — one allocation for this
// one vector — so the single-get path is unchanged.
//
// The RETURNED arena must be stored back by the caller: it may have been grown
// (reallocated) by this call. Records decoded earlier into a since-grown arena
// keep pointing at the OLD backing array, which still holds their (correct)
// floats — growth copies, it does not invalidate — so a mid-batch grow costs a
// stranded array, never wrong data.
//
// The returned vec ALIASES arena. Callers therefore must not reuse an arena
// while any vector decoded into it is still live (see
// DecodeVectorGetBatchResultInto).
func decodeGetResultAtArena(body []byte, off int, versionFramed bool, arena []float32) (found bool, vec []float32, meta vtypes.Metadata, ttl time.Duration, sparse *vtypes.SparseVector, version uint64, next int, arenaOut []float32, err error) {
	if len(body) < off+1 {
		return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
	}
	if body[off] == 0 {
		return false, nil, nil, 0, nil, 0, off + 1, arena, nil
	}
	off++
	if len(body) < off+4 {
		return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
	}
	dim := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	// See DecodeVectorInsertArgs: 4*dim overflows for a negative dim. The 9 is
	// the fixed ttl+metaPresent tail this record still needs after the vector.
	if !CountFitsIn(dim, len(body)-off-9, 4) {
		return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
	}
	if dim > 0 {
		base := len(arena)
		arena = slices.Grow(arena, dim)[:base+dim]
		vec = arena[base : base+dim : base+dim]
		for i := 0; i < dim; i++ {
			vec[i] = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
			off += 4
		}
	}
	ttl = time.Duration(binary.BigEndian.Uint64(body[off:])) * time.Millisecond //nolint:gosec
	off += 8
	metaPresent := body[off]
	off++
	if metaPresent == 1 {
		if len(body) < off+4 {
			return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
		}
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if mlen < 0 || len(body)-off < mlen {
			return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
		}
		m := make(vtypes.Metadata)
		if err := json.Unmarshal(body[off:off+mlen], &m); err != nil {
			return false, nil, nil, 0, nil, 0, off, arena, fmt.Errorf("ops: decode get metadata: %w", err)
		}
		meta = m
		off += mlen
	}
	if len(body) < off+1 {
		return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
	}
	sparsePresent := body[off]
	off++
	if sparsePresent == 1 {
		sv, soff, serr := readSparse(body, off)
		if serr != nil {
			return false, nil, nil, 0, nil, 0, off, arena, serr
		}
		off = soff
		if !sv.IsZero() {
			cp := sv
			sparse = &cp
		}
	}
	// Trailing version block.
	if versionFramed {
		if len(body) < off+1 {
			return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
		}
		verPresent := body[off]
		off++
		if verPresent == 1 {
			if len(body) < off+8 {
				return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
			}
			version = binary.BigEndian.Uint64(body[off:])
			off += 8
		}
	} else if off < len(body) {
		// Single-get: an OPTIONAL block — read it only when bytes remain.
		verPresent := body[off]
		off++
		if verPresent == 1 {
			if len(body) < off+8 {
				return false, nil, nil, 0, nil, 0, off, arena, ErrVectorArgsTruncated
			}
			version = binary.BigEndian.Uint64(body[off:])
			off += 8
		}
	}
	return true, vec, meta, ttl, sparse, version, off, arena, nil
}

// GetBatchRow is one row of a vector_get_batch result: the requested id plus the
// same projected fields a single vector_get carries (Found is the not-found FLAG,
// never an error). For a not-found id only ID/Found are meaningful. Vec/Meta/Sparse
// follow the with_vector/with_payload projection that was applied at fetch time.
type GetBatchRow struct {
	ID      uint64
	Found   bool
	Vec     []float32
	Meta    vtypes.Metadata
	TTLMs   uint64
	Sparse  *vtypes.SparseVector
	Version uint64 // per-point CAS version (>=1 for a found point; 0 = absent/unknown)
}

// EncodeVectorGetBatchArgs serializes a vector_get_batch request. Wire:
// [colLen:u8][col][flags:u8][n:u32][id:u64 × n] where flags is the same
// GetFlagWithVector|GetFlagWithPayload bit field as single get. The leading
// [colLen][col] keeps VectorKeyColAt1 routing/auth intact; the flags byte sits
// before the id list so it is at a fixed offset regardless of n.
func EncodeVectorGetBatchArgs(collection string, ids []uint64, flags uint8) []byte {
	buf := make([]byte, 1+len(collection)+1+4+8*len(ids))
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	buf[off] = flags
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(len(ids))) //nolint:gosec
	off += 4
	for _, id := range ids {
		binary.BigEndian.PutUint64(buf[off:], id)
		off += 8
	}
	return buf
}

// DecodeVectorGetBatchArgs reads args produced by EncodeVectorGetBatchArgs. Fails
// loud on truncation or a declared count n that overruns the buffer. A zero-id
// request (n=0) is valid and yields an empty ids slice.
func DecodeVectorGetBatchArgs(args []byte) (collection string, ids []uint64, flags uint8, err error) {
	if len(args) < 1 {
		return "", nil, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+1+4 {
		return "", nil, 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	flags = args[off]
	off++
	n := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// Each id costs 8 bytes; the divide-form bound rejects a widened-negative or
	// absurd n before the make (8*n cannot overflow int here).
	if !CountFitsIn(n, len(args)-off, 8) {
		return "", nil, 0, ErrVectorArgsTruncated
	}
	ids = make([]uint64, n)
	for i := 0; i < n; i++ {
		ids[i] = binary.BigEndian.Uint64(args[off:])
		off += 8
	}
	return collection, ids, flags, nil
}

// EncodeVectorGetBatchResult serializes a per-partition vector_get_batch result.
// Wire: [n:u32] then for each row [id:u64] followed by the SAME [found:u8]+body
// record EncodeVectorGetResult produces (so a batch row is just id + a single-get
// result). Rows preserve the order the handler was given. A not-found id is a
// found=0 record (NEVER an op error) — the coordinator derives the global missing
// set from absent ids.
func EncodeVectorGetBatchResult(rows []GetBatchRow) []byte {
	// Presize: count header + per-row (id + record). The estimate is exact for
	// every field except the meta JSON (whose marshaled length isn't known without
	// marshaling); meta is small and any shortfall just triggers a normal append
	// regrowth. This collapses the old ~log(rows) buffer reallocations + a
	// throwaway slice per row into a single allocation in the common case.
	est := 4
	for i := range rows {
		est += 8 + estimateGetRowSize(&rows[i])
	}
	buf := slices.Grow([]byte(nil), est)
	buf = append(buf, 0, 0, 0, 0)
	binary.BigEndian.PutUint32(buf, uint32(len(rows))) //nolint:gosec
	var idbuf [8]byte
	for i := range rows {
		binary.BigEndian.PutUint64(idbuf[:], rows[i].ID)
		buf = append(buf, idbuf[:]...)
		buf = appendGetResultRowFramed(buf, &rows[i])
	}
	return buf
}

// estimateGetRowSize returns a near-exact upper-ish estimate of a batch row's
// encoded size (every field except the meta JSON, which is unknown without
// marshaling). Used only to presize the batch buffer.
func estimateGetRowSize(r *GetBatchRow) int {
	if !r.Found {
		return 1
	}
	n := 1 + 4 + 8 + 1 + 1 + 1 + 8 // found, dimHdr, ttl, metaPresent, sparsePresent, verPresent, version
	n += 4 * len(r.Vec)
	if r.Sparse != nil {
		n += 4 + 8*len(r.Sparse.Indices)
	}
	return n
}

// appendGetResultRowFramed appends one batch row's get-result record to dst with a
// version block that is ALWAYS framed ([verPresent:u8][?version:u64]) so the
// record self-delimits and the next row's id is found unambiguously. A found row
// with version>=1 writes [1][version]; a version of 0 (incl. a not-found row,
// whose record is the single found=0 byte and carries no version block at all)
// writes [0]. This differs from EncodeVectorGetResultV (single-get), which OMITS
// the present byte when version==0 to stay byte-identical to the legacy encoder.
func appendGetResultRowFramed(dst []byte, r *GetBatchRow) []byte {
	if !r.Found {
		return append(dst, 0)
	}
	// version=0 here so the base record carries NO version block; the framed
	// present byte (and optional u64) is appended after it.
	dst = appendVectorGetResultV(dst, true, r.Vec, r.Meta, time.Duration(r.TTLMs)*time.Millisecond, r.Sparse, true, true, 0)
	if r.Version == 0 {
		return append(dst, 0)
	}
	dst = append(dst, 1)
	var v [8]byte
	binary.BigEndian.PutUint64(v[:], r.Version)
	return append(dst, v[:]...)
}

// DecodeVectorGetBatchResult reads a result produced by EncodeVectorGetBatchResult.
// Fails loud on truncation or a declared row count that overruns the buffer. A
// zero-row result yields an empty (non-nil) slice. Each row's projected fields are
// exactly what the encoder carried (the projection was applied at fetch time).
func DecodeVectorGetBatchResult(body []byte) ([]GetBatchRow, error) {
	rows, _, err := DecodeVectorGetBatchResultInto(body, nil, nil)
	return rows, err
}

// DecodeVectorGetBatchResultInto is DecodeVectorGetBatchResult with caller-owned
// storage, the batch-get analogue of DecodeVectorSearchArgsInto. It is the SAME
// decoder (DecodeVectorGetBatchResult delegates to it with nil storage) — there is
// no second wire path — it just stops allocating one []float32 per row:
//
//   - rows is truncated to zero length and reused when it has the capacity;
//   - every found row's Vec is carved out of ONE arena slice, pre-sized from the
//     first row's dim, so a 100-row batch costs one vector allocation instead of
//     100.
//
// BOTH reservations are bounded by what the body could actually encode before
// either is made — the row count by (len(body)-4)/9 (a row costs >= 9 bytes) and
// the arena by len(body)/4 (a float costs 4) — so a hostile count or dim is a
// decode error, never an oversized allocation.
//
// BOTH returned slices must be stored back by the caller — either may have been
// grown. LIFETIME: each returned row's Vec ALIASES arena, so an arena may only be
// reused once every row decoded into it is dead. That makes it a per-call scratch,
// NOT a cross-request pool: batch-get rows are handed to the caller (they become
// the result's points), so reusing an arena across requests would rewrite vectors
// a previous caller still holds.
func DecodeVectorGetBatchResultInto(body []byte, rows []GetBatchRow, arena []float32) ([]GetBatchRow, []float32, error) {
	if len(body) < 4 {
		return nil, arena, ErrVectorArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(body))
	off := 4
	// Bound the DECLARED row count before reserving anything for it. The smallest a
	// row can encode is 9 bytes ([id:u64][found=0:u8]), so no body of this length can
	// hold more than (len(body)-4)/9 rows. Without this bound a hostile count
	// (0xFFFFFFFF ⇒ ~3.8e9 rows) reaches the reservation below as a multi-hundred-GB
	// request, and an out-of-memory abort is NOT a recoverable decode error the
	// caller can reject the frame on — it takes the process down. The per-row
	// truncation checks alone are too late: they run after the reservation.
	// DecodeVectorGetBatchArgs bounds its id count the same way (len(args) >= 8n).
	if !CountFitsIn(n, len(body)-off, 9) {
		return nil, arena, ErrVectorArgsTruncated
	}
	rows = slices.Grow(rows[:0], n)
	if rows == nil {
		// A zero-row body with no caller storage: keep the documented empty
		// (non-nil) result rather than handing back nil.
		rows = []GetBatchRow{}
	}
	arena = growArenaForBatch(body, n, arena)
	for i := 0; i < n; i++ {
		if len(body) < off+8 {
			return nil, arena, ErrVectorArgsTruncated
		}
		id := binary.BigEndian.Uint64(body[off:])
		off += 8
		found, vec, meta, ttl, sparse, version, next, grown, err := decodeGetResultAtArena(body, off, true, arena)
		arena = grown
		if err != nil {
			return nil, arena, err
		}
		off = next
		rows = append(rows, GetBatchRow{
			ID:      id,
			Found:   found,
			Vec:     vec,
			Meta:    meta,
			TTLMs:   uint64(ttl.Milliseconds()), //nolint:gosec // TTL >= 0
			Sparse:  sparse,
			Version: version,
		})
	}
	return rows, arena, nil
}

// growArenaForBatch reserves the vector arena for an n-row batch-get body in ONE
// grow, so the per-row decode never reallocates on a uniform-dim batch (the
// universal case: a collection has one dim). It peeks the FIRST row's dim without
// decoding — [n:u32][id:u64][found:u8][dim:u32] — and reserves n*dim floats,
// clamped to len(body)/4: a found row spends at least 4 bytes of body per float,
// so no valid body can carry more floats than that, and a hostile n/dim therefore
// cannot turn into an oversized allocation. A not-found or dimensionless first row
// reserves nothing and the per-row appends grow the arena as they go (correct, just
// not single-shot: an earlier row's vector keeps pointing at the pre-grow array,
// which still holds its floats).
func growArenaForBatch(body []byte, n int, arena []float32) []float32 {
	const firstDimAt = 4 + 8 + 1 // [n:u32][id:u64][found:u8]
	if n <= 0 || len(body) < firstDimAt+4 || body[4+8] != 1 {
		return arena
	}
	dim := int(binary.BigEndian.Uint32(body[firstDimAt:]))
	if dim <= 0 {
		return arena
	}
	want := int64(n) * int64(dim)
	if maxFloats := int64(len(body) / 4); want > maxFloats {
		want = maxFloats
	}
	return slices.Grow(arena, int(want))
}

// appendKeyTTLBlock appends an OPTIONAL self-delimiting per-key payload TTL block
// to buf and returns the grown slice. The block is [keyTtlPresent:u8][?len:u32]
// [?keyTtlJSON] (key -> RELATIVE ms). It is the named/MV insert/add analogue of the
// set_payload trailer and obeys the same byte-compat discipline:
//
//   - When keyTTLMs is empty AND no later trailer follows (followedByTrailer ==
//     false) NOTHING is appended, so the output stays BYTE-IDENTICAL to the legacy
//     (pre-per-key-TTL) wire and the legacy decoder still works.
//   - When a later trailer follows (a CAS / version block), the present byte is
//     ALWAYS emitted (0 when no map) so the following trailer sits at a
//     deterministic offset — exactly the set_payload-CAS interpose.
//
// Read back with readKeyTTLBlock.
func appendKeyTTLBlock(buf []byte, keyTTLMs map[string]int64, followedByTrailer bool) []byte {
	if len(keyTTLMs) == 0 {
		if !followedByTrailer {
			return buf // byte-identical to the legacy wire
		}
		return append(buf, 0) // present=0 so the following trailer is at a fixed offset
	}
	keyTTLJSON, _ := json.Marshal(keyTTLMs)
	out := append(buf, 1)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(len(keyTTLJSON))) //nolint:gosec
	out = append(out, u32[:]...)
	return append(out, keyTTLJSON...)
}

// readKeyTTLBlock reads the OPTIONAL self-delimiting per-key TTL block (written by
// appendKeyTTLBlock) at offset off in args, returning the decoded RELATIVE-ms map
// (nil when absent/empty) and the offset just past the block. An OLD blob that ends
// before off (no block) yields (nil, off, nil) — back-compat. A present-but-
// truncated block, or bad JSON, is fail-loud.
func readKeyTTLBlock(args []byte, off int) (keyTTLMs map[string]int64, next int, err error) {
	if off >= len(args) {
		return nil, off, nil // OLD wire: no block at all
	}
	present := args[off]
	off++
	if present == 0 {
		return nil, off, nil
	}
	if len(args) < off+4 {
		return nil, off, ErrVectorArgsTruncated
	}
	klen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if klen < 0 || len(args)-off < klen {
		return nil, off, ErrVectorArgsTruncated
	}
	km := make(map[string]int64)
	if uerr := json.Unmarshal(args[off:off+klen], &km); uerr != nil {
		return nil, off, fmt.Errorf("ops: decode key_ttl_ms: %w", uerr)
	}
	off += klen
	if len(km) > 0 {
		keyTTLMs = km
	}
	return keyTTLMs, off, nil
}

// EncodeSetPayloadArgs serializes a vector_set_payload / vector_overwrite_payload
// request with NO per-key TTL (byte-identical to the pre-per-key-TTL encoder).
// Shared by all three families; the per-key-TTL-aware dense path uses
// EncodeSetPayloadArgsOpts.
func EncodeSetPayloadArgs(collection string, id uint64, meta vtypes.Metadata) []byte {
	return EncodeSetPayloadArgsOpts(collection, id, meta, nil)
}

// EncodeSetPayloadArgsOpts serializes a vector_set_payload / vector_overwrite_payload
// request (the wire shape is identical; the OP NAME selects merge vs replace).
// Wire: [colLen:u8][col][id:u64][metaLen:u32][metaJSON]{[keyTtlPresent:u8]
//
//	[?keyTtlLen:u32][?keyTtlJSON]}
//
// keyTTLMs is an OPTIONAL per-key payload TTL map (key -> RELATIVE ms; the engine
// computes the absolute deadline so the WAL stays time-stable). When it is
// empty/nil NOTHING is appended after metaJSON, so the output is BYTE-IDENTICAL to
// the legacy (pre-per-key-TTL) encoder and the legacy 4-tuple decoder still works.
// metaLen 0 = empty payload (a valid overwrite-to-empty / merge-nothing). Shared
// by all three families.
func EncodeSetPayloadArgsOpts(collection string, id uint64, meta vtypes.Metadata, keyTTLMs map[string]int64) []byte {
	var metaJSON []byte
	if len(meta) > 0 {
		metaJSON, _ = json.Marshal(meta)
	}
	var keyTTLJSON []byte
	if len(keyTTLMs) > 0 {
		keyTTLJSON, _ = json.Marshal(keyTTLMs)
	}
	n := 1 + len(collection) + 8 + 4 + len(metaJSON)
	if len(keyTTLJSON) > 0 {
		n += 1 + 4 + len(keyTTLJSON) // present=1 + len + JSON
	}
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	off += 8
	binary.BigEndian.PutUint32(buf[off:], uint32(len(metaJSON))) //nolint:gosec
	off += 4
	off += copy(buf[off:], metaJSON)
	// Optional trailing per-key-TTL block. Absent map => append NOTHING (byte-
	// identical to the legacy encoder); present => [1][len:u32][JSON].
	if len(keyTTLJSON) > 0 {
		buf[off] = 1
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(len(keyTTLJSON))) //nolint:gosec
		off += 4
		copy(buf[off:], keyTTLJSON)
	}
	return buf
}

// EncodeSetPayloadArgsCAS serializes a vector_set_payload / vector_overwrite_payload
// request carrying an optimistic-CAS precondition. When hasExpected is false the
// output is BYTE-IDENTICAL to EncodeSetPayloadArgsOpts (no CAS block). When
// present the per-key-TTL present-byte is ALWAYS emitted (0 when no map) so the
// trailing [casPresent:u8][expectedVersion:u64] block sits at a deterministic
// offset; the legacy DecodeSetPayloadArgsOpts reads the keyTTL block and stops
// before the CAS block (it never consults it), and DecodeSetPayloadArgsCAS reads
// the CAS guard. The mutation applies ONLY when the point's current version
// matches expectedVersion.
func EncodeSetPayloadArgsCAS(collection string, id uint64, meta vtypes.Metadata, keyTTLMs map[string]int64, expectedVersion uint64, hasExpected bool) []byte {
	if !hasExpected {
		return EncodeSetPayloadArgsOpts(collection, id, meta, keyTTLMs)
	}
	var metaJSON []byte
	if len(meta) > 0 {
		metaJSON, _ = json.Marshal(meta)
	}
	var keyTTLJSON []byte
	if len(keyTTLMs) > 0 {
		keyTTLJSON, _ = json.Marshal(keyTTLMs)
	}
	// keyTTL present-byte is ALWAYS emitted here (0 when no map) + the CAS block.
	n := 1 + len(collection) + 8 + 4 + len(metaJSON) + 1
	if len(keyTTLJSON) > 0 {
		n += 4 + len(keyTTLJSON)
	}
	n += 1 + 8 // casPresent + expectedVersion
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	off += 8
	binary.BigEndian.PutUint32(buf[off:], uint32(len(metaJSON))) //nolint:gosec
	off += 4
	off += copy(buf[off:], metaJSON)
	if len(keyTTLJSON) > 0 {
		buf[off] = 1
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(len(keyTTLJSON))) //nolint:gosec
		off += 4
		off += copy(buf[off:], keyTTLJSON)
	} else {
		buf[off] = 0
		off++
	}
	buf[off] = 1
	off++
	binary.BigEndian.PutUint64(buf[off:], expectedVersion)
	return buf
}

// DecodeSetPayloadArgs reads args produced by EncodeSetPayloadArgs. A bad payload
// JSON is a hard error (fail-loud); an empty payload decodes to a nil meta. The
// per-key TTL block (if any) is decoded and discarded — callers that need it use
// DecodeSetPayloadArgsOpts.
func DecodeSetPayloadArgs(args []byte) (collection string, id uint64, meta vtypes.Metadata, err error) {
	collection, id, meta, _, err = DecodeSetPayloadArgsOpts(args)
	return collection, id, meta, err
}

// DecodeSetPayloadArgsOpts reads args produced by EncodeSetPayloadArgsOpts (or the
// legacy EncodeSetPayloadArgs — the trailing per-key-TTL block is optional). A bad
// payload/key-ttl JSON is a hard error (fail-loud); an empty payload decodes to a
// nil meta; an absent or empty per-key-TTL block decodes to a nil keyTTLMs (the
// relative-ms map, exactly as encoded). A truncated trailing block is fail-loud.
func DecodeSetPayloadArgsOpts(args []byte) (collection string, id uint64, meta vtypes.Metadata, keyTTLMs map[string]int64, err error) {
	if len(args) < 1 {
		return "", 0, nil, nil, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8+4 {
		return "", 0, nil, nil, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	id = binary.BigEndian.Uint64(args[off:])
	off += 8
	mlen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if mlen < 0 || len(args)-off < mlen {
		return "", 0, nil, nil, ErrVectorArgsTruncated
	}
	if mlen > 0 {
		m := make(vtypes.Metadata)
		if err := json.Unmarshal(args[off:off+mlen], &m); err != nil {
			return "", 0, nil, nil, fmt.Errorf("ops: decode payload: %w", err)
		}
		meta = m
	}
	off += mlen
	// Optional per-key-TTL block. An OLD encoder stops right after metaJSON; treat
	// the absence of the present byte as "no per-key TTL" (back-compat).
	if off >= len(args) {
		return collection, id, meta, nil, nil
	}
	present := args[off]
	off++
	if present == 0 {
		return collection, id, meta, nil, nil
	}
	if len(args) < off+4 {
		return "", 0, nil, nil, ErrVectorArgsTruncated
	}
	klen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if klen < 0 || len(args)-off < klen {
		return "", 0, nil, nil, ErrVectorArgsTruncated
	}
	km := make(map[string]int64)
	if err := json.Unmarshal(args[off:off+klen], &km); err != nil {
		return "", 0, nil, nil, fmt.Errorf("ops: decode key_ttl_ms: %w", err)
	}
	if len(km) > 0 {
		keyTTLMs = km
	}
	return collection, id, meta, keyTTLMs, nil
}

// DecodeSetPayloadArgsCAS reads args produced by EncodeSetPayloadArgsCAS (or the
// legacy EncodeSetPayloadArgs/Opts — the CAS block is optional). It decodes the
// payload + per-key-TTL block exactly like DecodeSetPayloadArgsOpts, then reads an
// OPTIONAL trailing [casPresent:u8][expectedVersion:u64] block. hasExpected
// reports whether the CAS guard was present; expectedVersion is its value (0 =
// expect-absent). A legacy blob (no CAS block) decodes to hasExpected=false; a
// present-but-truncated CAS block is fail-loud.
func DecodeSetPayloadArgsCAS(args []byte) (collection string, id uint64, meta vtypes.Metadata, keyTTLMs map[string]int64, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 1 {
		return "", 0, nil, nil, 0, false, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8+4 {
		return "", 0, nil, nil, 0, false, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	id = binary.BigEndian.Uint64(args[off:])
	off += 8
	mlen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if mlen < 0 || len(args)-off < mlen {
		return "", 0, nil, nil, 0, false, ErrVectorArgsTruncated
	}
	if mlen > 0 {
		m := make(vtypes.Metadata)
		if uerr := json.Unmarshal(args[off:off+mlen], &m); uerr != nil {
			return "", 0, nil, nil, 0, false, fmt.Errorf("ops: decode payload: %w", uerr)
		}
		meta = m
	}
	off += mlen
	// Per-key-TTL block. A legacy encoder stops right after metaJSON.
	if off >= len(args) {
		return collection, id, meta, nil, 0, false, nil
	}
	present := args[off]
	off++
	if present == 1 {
		if len(args) < off+4 {
			return "", 0, nil, nil, 0, false, ErrVectorArgsTruncated
		}
		klen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if klen < 0 || len(args)-off < klen {
			return "", 0, nil, nil, 0, false, ErrVectorArgsTruncated
		}
		km := make(map[string]int64)
		if uerr := json.Unmarshal(args[off:off+klen], &km); uerr != nil {
			return "", 0, nil, nil, 0, false, fmt.Errorf("ops: decode key_ttl_ms: %w", uerr)
		}
		if len(km) > 0 {
			keyTTLMs = km
		}
		off += klen
	}
	// Optional trailing CAS block (present iff the CAS encoder wrote it).
	if off >= len(args) {
		return collection, id, meta, keyTTLMs, 0, false, nil
	}
	casPresent := args[off]
	off++
	if casPresent == 0 {
		return collection, id, meta, keyTTLMs, 0, false, nil
	}
	if len(args) < off+8 {
		return "", 0, nil, nil, 0, false, ErrVectorArgsTruncated
	}
	expectedVersion = binary.BigEndian.Uint64(args[off:])
	return collection, id, meta, keyTTLMs, expectedVersion, true, nil
}

// EncodeDeletePayloadKeysArgs serializes a vector_delete_payload_keys request.
// Wire: [colLen:u8][col][id:u64][nKeys:u32]{[keyLen:u16][key]}. Shared by all three
// families. An empty key list is a valid no-op delete.
func EncodeDeletePayloadKeysArgs(collection string, id uint64, keys []string) []byte {
	n := 1 + len(collection) + 8 + 4
	for _, k := range keys {
		n += 2 + len(k)
	}
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint64(buf[off:], id)
	off += 8
	binary.BigEndian.PutUint32(buf[off:], uint32(len(keys))) //nolint:gosec
	off += 4
	for _, k := range keys {
		binary.BigEndian.PutUint16(buf[off:], uint16(len(k))) //nolint:gosec
		off += 2
		off += copy(buf[off:], k)
	}
	return buf
}

// EncodeDeletePayloadKeysArgsCAS serializes a vector_delete_payload_keys request
// carrying an optional optimistic-CAS precondition. When hasExpected is false the
// output is BYTE-IDENTICAL to EncodeDeletePayloadKeysArgs; when present a trailing
// [1][expectedVersion:u64] block follows the (deterministic) key list.
func EncodeDeletePayloadKeysArgsCAS(collection string, id uint64, keys []string, expectedVersion uint64, hasExpected bool) []byte {
	base := EncodeDeletePayloadKeysArgs(collection, id, keys)
	if !hasExpected {
		return base
	}
	buf := make([]byte, len(base)+1+8)
	off := copy(buf, base)
	buf[off] = 1
	off++
	binary.BigEndian.PutUint64(buf[off:], expectedVersion)
	return buf
}

// DecodeDeletePayloadKeysArgs reads args produced by EncodeDeletePayloadKeysArgs.
func DecodeDeletePayloadKeysArgs(args []byte) (collection string, id uint64, keys []string, err error) {
	collection, id, keys, _, _, err = DecodeDeletePayloadKeysArgsCAS(args)
	return collection, id, keys, err
}

// DecodeDeletePayloadKeysArgsCAS reads args produced by
// EncodeDeletePayloadKeysArgsCAS (or the legacy encoder — the CAS block is
// optional). hasExpected reports whether the CAS guard was present; an absent
// (legacy) trailer decodes to hasExpected=false; a present-but-truncated trailer
// is fail-loud.
func DecodeDeletePayloadKeysArgsCAS(args []byte) (collection string, id uint64, keys []string, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 1 {
		return "", 0, nil, 0, false, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8+4 {
		return "", 0, nil, 0, false, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	id = binary.BigEndian.Uint64(args[off:])
	off += 8
	nKeys := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// Each key costs >= 2 bytes ([keyLen:u16] with an empty key); the divide-form
	// bound rejects a widened-negative or absurd nKeys before the make cap.
	if !CountFitsIn(nKeys, len(args)-off, 2) {
		return "", 0, nil, 0, false, ErrVectorArgsTruncated
	}
	if nKeys > 0 {
		keys = make([]string, 0, nKeys)
	}
	for i := 0; i < nKeys; i++ {
		if len(args) < off+2 {
			return "", 0, nil, 0, false, ErrVectorArgsTruncated
		}
		klen := int(binary.BigEndian.Uint16(args[off:]))
		off += 2
		if klen < 0 || len(args)-off < klen {
			return "", 0, nil, 0, false, ErrVectorArgsTruncated
		}
		keys = append(keys, string(args[off:off+klen]))
		off += klen
	}
	// Optional trailing CAS block.
	if off >= len(args) {
		return collection, id, keys, 0, false, nil
	}
	present := args[off]
	off++
	if present == 0 {
		return collection, id, keys, 0, false, nil
	}
	if len(args) < off+8 {
		return "", 0, nil, 0, false, ErrVectorArgsTruncated
	}
	expectedVersion = binary.BigEndian.Uint64(args[off:])
	return collection, id, keys, expectedVersion, true, nil
}

// EncodeClearPayloadArgs serializes a vector_clear_payload request. Wire:
// [colLen:u8][col][id:u64] — byte-identical to the delete-args shape. Shared by all
// three families (the op name selects the family).
func EncodeClearPayloadArgs(collection string, id uint64) []byte {
	return EncodeVectorDeleteArgs(collection, id)
}

// EncodeClearPayloadArgsCAS serializes a vector_clear_payload request with an
// optional optimistic-CAS precondition (same wire shape as the CAS delete args).
// Byte-identical to EncodeClearPayloadArgs when hasExpected is false.
func EncodeClearPayloadArgsCAS(collection string, id uint64, expectedVersion uint64, hasExpected bool) []byte {
	return EncodeVectorDeleteArgsCAS(collection, id, expectedVersion, hasExpected)
}

// DecodeClearPayloadArgs reads args produced by EncodeClearPayloadArgs.
func DecodeClearPayloadArgs(args []byte) (string, uint64, error) {
	return DecodeVectorDeleteArgs(args)
}

// DecodeClearPayloadArgsCAS reads args produced by EncodeClearPayloadArgsCAS (or
// the legacy encoder). See DecodeVectorDeleteArgsCAS.
func DecodeClearPayloadArgsCAS(args []byte) (collection string, id uint64, expectedVersion uint64, hasExpected bool, err error) {
	return DecodeVectorDeleteArgsCAS(args)
}

// EncodePayloadResult serializes a payload-mutation result: a single byte, 1 if the
// point existed and the payload was applied, 0 if the point was absent/tombstoned/
// expired (the not-found FLAG). NOT-found is a flag, never an op error, so a
// point-op fan-out can route to ONE partition and treat applied=0 as "not in this
// partition". Mirrors EncodeExistsResult / the vector_delete existed-byte.
func EncodePayloadResult(applied bool) []byte {
	if applied {
		return []byte{1}
	}
	return []byte{0}
}

// DecodePayloadResult reads the byte produced by EncodePayloadResult.
func DecodePayloadResult(body []byte) (bool, error) {
	if len(body) < 1 {
		return false, ErrVectorArgsTruncated
	}
	return body[0] == 1, nil
}
