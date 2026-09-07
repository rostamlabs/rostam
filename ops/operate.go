// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"errors"
	"math"
	"sort"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// operate is the generic atomic multi-field update op (the native alternative to a
// client CAS-retry loop on a HOT key). One Rostam value is modelled as:
//
//   - globals: a small array of TYPED counters.
//   - entries: a map[u64]*operateEntry of dynamic sub-records, each a typed field
//     array plus a leader-stamped touchMs used for DETERMINISTIC LRU eviction.
//
// Fields are TYPED for memory efficiency: each field is stored self-describing as
// [type u8][data], where data is the native fixed width (1/2/4/8 bytes for
// U8..I64), 4/8 bytes for F32/F64, or LEB128 for UVARINT/IVARINT (zigzag). A u8
// counter costs 2 bytes instead of 8. The app owns which type each field is; the
// type rides the op-entry so a field is created on first touch, and the STORED
// type wins for a field that already exists (the op's type is ignored then).
//
// Per-type arithmetic (all internal math in i64/u64/f64, stored back per type):
//   - INCR/INCRF/SETMAX on a fixed-width INT type SATURATE at that type's min/max
//     (never wrap); on a float type they use IEEE math; UVARINT/IVARINT saturate
//     only at the u64/i64 boundary (effectively uncapped, byte length may change).
//   - SHIFTOR on a fixed-width int MASKS to the field's bit width (rolling window,
//     old bits fall off); it is rejected on varint and float types.
//   - HALVE_GRP halves each field in the group per that field's own type.
//
// DETERMINISM (RF>1 safety): every mutation is integer/float-only; touchMs and the
// TTL come solely from tx.applyStamp() (the leader-stamped nowMs); float encodings
// are IEEE-deterministic; the record encodes entries sorted by key and fields in
// index order; eviction picks the smallest touchMs (ties → smallest key). A
// follower re-applying the same committed op-list recomputes byte-identical state.

// maxOperateFields caps a globals/entry-fields array LENGTH. The record encoding
// prefixes each array with a u16 count, so an array holds at most 65535 fields; a
// fieldIdx that would grow it past that is rejected. 64K counters per array is
// ample for a session record.
const maxOperateFields = 65535

// minOperateEntryBytes is the smallest honest encoded entry: key(8) + touchMs(8) +
// nFields(2), with zero fields. minOperateFieldBytes is the smallest field: a
// type tag + one data byte (U8/I8, or a 1-byte varint). Both bound declared counts
// in decodeRec before any allocation (CountFitsIn), the wire batch-decoder
// discipline.
const (
	minOperateEntryBytes = 18
	minOperateFieldBytes = 2
)

var (
	errOperateOpcode     = errors.New("ops: operate unknown opcode")
	errOperateShift      = errors.New("ops: operate SHIFTOR negative shift")
	errOperateShiftType  = errors.New("ops: operate SHIFTOR requires a fixed-width int field")
	errOperateGroup      = errors.New("ops: operate HALVE_GRP group length < 1")
	errOperateFieldRange = errors.New("ops: operate field index out of range")
	errOperateType       = errors.New("ops: operate unknown field type")
)

// operateField is one typed field. Integer/varint values live in u (unsigned types
// hold the value; signed types hold the sign-extended two's-complement, read via
// int64(u)); float values live in f. typ selects which and how it encodes.
type operateField struct {
	typ uint8
	u   uint64
	f   float64
}

// operateEntry is one dynamic sub-record: a typed field array plus the
// leader-stamped last-touch time driving deterministic LRU eviction.
type operateEntry struct {
	touchMs int64
	fields  []operateField
}

// operateRec is the decoded value: the globals array and the entry map.
type operateRec struct {
	globals []operateField
	entries map[uint64]*operateEntry
}

// handleOperate applies a client op-list to one record atomically under the shard
// write lock. On any op error (unknown opcode/type, negative shift, SHIFTOR on a
// non-int field, out-of-range field) it returns BEFORE the Put, so a rejected
// op-list leaves the record unchanged — the whole operate is all-or-nothing.
// Registered OpReadWrite.
func handleOperate(tx *TxContext, args []byte) ([]byte, error) {
	key, ttl, maxEntries, opsList, ret, err := wire.DecodeOperateArgs(args)
	if err != nil {
		return nil, err
	}
	cur, err := tx.Get(key)
	if err != nil && err != cache.ErrNotFound {
		return nil, err
	}
	// cur aliases the page but decodeRec copies every value out into fresh fields, so
	// rec retains no alias; the Put below writes an independently-built buffer.
	rec, err := decodeRec(cur)
	if err != nil {
		return nil, err
	}
	nowMs, _ := tx.applyStamp() // leader-stamped ⇒ deterministic on followers (0 on the unstamped Direct path)
	for _, o := range opsList {
		if err := applyOp(&rec, o, nowMs, int(maxEntries)); err != nil {
			return nil, err
		}
	}
	if err := tx.Put(key, encodeRec(rec), ttl); err != nil {
		return nil, err
	}
	return encodeReturn(&rec, ret), nil
}

// applyOp applies one op to rec. Global ops act on rec.globals; entry ops upsert
// (and, on a new key over the cap, deterministically evict) the sub-record and
// stamp its touchMs before acting on its fields. The target array is grown as
// needed (bounded by maxOperateFields), creating new fields with the op's type.
func applyOp(rec *operateRec, o wire.OperateOp, nowMs int64, maxEntries int) error {
	arrp, err := rec.targetArray(o, nowMs, maxEntries)
	if err != nil {
		return err
	}
	switch o.Opcode {
	case wire.OperateOpINCR, wire.OperateOpINCRF, wire.OperateOpSETMAX, wire.OperateOpSHIFTOR:
		idx := int(o.FieldIdx)
		arr, ok := ensureLen(*arrp, idx+1, o.Type)
		if !ok {
			return errOperateFieldRange
		}
		*arrp = arr
		return applyScalar(&arr[idx], o.Opcode, o.Arg, o.Arg2)
	case wire.OperateOpHALVEGRP:
		// HALVE_GRP(thr=Arg, len=Arg2): the group is the contiguous field range
		// [FieldIdx, FieldIdx+len); FieldIdx is the guard field. If the guard is >= the
		// threshold, halve every field in the group per its own type. (The fixed op
		// wire carries one fieldIdx + two i64 args, so a group is a contiguous range
		// rather than an explicit field list — the design's `HALVE_GRP(thr, f…)` with
		// f… the range starting at the guard.)
		if o.Arg2 < 1 {
			return errOperateGroup
		}
		idx := int(o.FieldIdx)
		end := idx + int(o.Arg2)
		arr, ok := ensureLen(*arrp, end, o.Type)
		if !ok {
			return errOperateFieldRange
		}
		*arrp = arr
		if guardTriggered(&arr[idx], o.Arg) {
			for i := idx; i < end; i++ {
				halveField(&arr[i])
			}
		}
		return nil
	default:
		return errOperateOpcode
	}
}

// applyScalar applies a single-field op (INCR/INCRF/SETMAX/SHIFTOR) to f, routing
// on the field's stored type.
func applyScalar(f *operateField, opcode uint8, arg, arg2 int64) error {
	if typeIsFloat(f.typ) {
		return applyScalarFloat(f, opcode, arg)
	}
	return applyScalarInt(f, opcode, arg, arg2)
}

func applyScalarInt(f *operateField, opcode uint8, arg, arg2 int64) error {
	switch opcode {
	case wire.OperateOpINCR, wire.OperateOpINCRF:
		// INCRF (fixed-point add) is the same integer add as INCR; the fixed-point
		// scale lives in the app's interpretation.
		if typeIsUnsigned(f.typ) {
			f.u = addSatUnsigned(f.u, arg, typeUMax(f.typ))
		} else {
			lo, hi := typeSignedRange(f.typ)
			f.u = uint64(addSatSigned(int64(f.u), arg, lo, hi)) //nolint:gosec // two's-complement store
		}
	case wire.OperateOpSETMAX:
		if typeIsUnsigned(f.typ) {
			if arg >= 0 { // a negative candidate is below every unsigned value
				cand := uint64(arg)
				if m := typeUMax(f.typ); cand > m {
					cand = m
				}
				if cand > f.u {
					f.u = cand
				}
			}
		} else {
			lo, hi := typeSignedRange(f.typ)
			cand := arg
			if cand < lo {
				cand = lo
			} else if cand > hi {
				cand = hi
			}
			if cand > int64(f.u) { //nolint:gosec // signed field value
				f.u = uint64(cand) //nolint:gosec // two's-complement store
			}
		}
	case wire.OperateOpSHIFTOR:
		// A rolling bitfield: shift then OR in the new bits then MASK to the field
		// width so old bits fall off. Only defined for a fixed-width int (a varint has
		// no width). A negative shift panics in Go, so reject it.
		if !typeIsFixedInt(f.typ) {
			return errOperateShiftType
		}
		if arg < 0 {
			return errOperateShift
		}
		bits := typeBits(f.typ)
		mask := widthMask(bits)
		var shifted uint64
		if arg < 64 {
			shifted = f.u << uint(arg)
		}
		res := (shifted | uint64(arg2)) & mask //nolint:gosec // bit pattern
		if !typeIsUnsigned(f.typ) {
			res = uint64(signExtend(res, bits)) //nolint:gosec // canonicalize signed value
		}
		f.u = res
	default:
		return errOperateOpcode
	}
	return nil
}

func applyScalarFloat(f *operateField, opcode uint8, arg int64) error {
	// A float operand rides Arg as its IEEE bit pattern (F32 = low 32 bits).
	val := floatFromArg(f.typ, arg)
	switch opcode {
	case wire.OperateOpINCR, wire.OperateOpINCRF:
		f.f = canonFloat(f.typ, f.f+val)
	case wire.OperateOpSETMAX:
		if val > f.f { // IEEE compare; NaN never replaces, deterministic
			f.f = canonFloat(f.typ, val)
		}
	case wire.OperateOpSHIFTOR:
		return errOperateShiftType
	default:
		return errOperateOpcode
	}
	return nil
}

// guardTriggered reports whether f's value is >= the HALVE_GRP threshold arg,
// compared in f's own domain.
func guardTriggered(f *operateField, arg int64) bool {
	if typeIsFloat(f.typ) {
		return f.f >= float64(arg)
	}
	if typeIsUnsigned(f.typ) {
		if arg < 0 {
			return true // every unsigned value exceeds a negative threshold
		}
		return f.u >= uint64(arg)
	}
	return int64(f.u) >= arg //nolint:gosec // signed field value
}

// halveField integer/float-halves f in place, per its type (integer division
// rounds toward zero for signed, which is deterministic).
func halveField(f *operateField) {
	switch {
	case typeIsFloat(f.typ):
		f.f = canonFloat(f.typ, f.f/2)
	case typeIsUnsigned(f.typ):
		f.u /= 2
	default:
		f.u = uint64(int64(f.u) / 2) //nolint:gosec // signed field value
	}
}

// targetArray returns a pointer to the field array an op mutates. For a global
// target that is &rec.globals; for an entry target it upserts the sub-record
// (evicting deterministically if a NEW key would exceed the cap), stamps its
// touchMs, and returns &entry.fields. Any op referencing an entry counts as a
// touch.
func (rec *operateRec) targetArray(o wire.OperateOp, nowMs int64, maxEntries int) (*[]operateField, error) {
	if o.Target != wire.OperateTargetEntry {
		return &rec.globals, nil
	}
	e := rec.upsertEntry(o.EntryKey, nowMs, maxEntries)
	e.touchMs = nowMs
	return &e.fields, nil
}

// upsertEntry returns the entry for key, creating an empty one (stamped
// touchMs=nowMs) if absent. When creating would exceed maxEntries (>0) it first
// evicts the deterministic LRU victim, so the cap holds under concurrency without
// app coordination.
func (rec *operateRec) upsertEntry(key uint64, nowMs int64, maxEntries int) *operateEntry {
	if e, ok := rec.entries[key]; ok {
		return e
	}
	if maxEntries > 0 && len(rec.entries) >= maxEntries {
		rec.evictOne()
	}
	e := &operateEntry{touchMs: nowMs}
	rec.entries[key] = e
	return e
}

// evictOne removes the entry with the smallest touchMs, ties broken by the
// smallest key — a deterministic function of the (already deterministic) map
// contents, so every replica evicts the same entry regardless of map order.
func (rec *operateRec) evictOne() {
	var (
		victim uint64
		bestMs int64
		found  bool
	)
	for k, e := range rec.entries {
		if !found || e.touchMs < bestMs || (e.touchMs == bestMs && k < victim) {
			victim, bestMs, found = k, e.touchMs, true
		}
	}
	if found {
		delete(rec.entries, victim)
	}
}

// ensureLen grows arr to at least length n, filling holes below the top with a U8
// zero and creating the top slot (index n-1) with newType. It returns arr
// unchanged when already long enough (the STORED field type wins for an existing
// index). ok=false when n exceeds maxOperateFields or a to-be-created top slot's
// newType is unknown.
func ensureLen(arr []operateField, n int, newType uint8) ([]operateField, bool) {
	if n > maxOperateFields {
		return arr, false
	}
	if n <= len(arr) {
		return arr, true
	}
	if newType >= wire.OperateTypeCount {
		return arr, false
	}
	for len(arr) < n {
		if len(arr) == n-1 {
			arr = append(arr, operateField{typ: newType}) // the addressed field
		} else {
			arr = append(arr, operateField{typ: wire.OperateTypeU8}) // a hole
		}
	}
	return arr, true
}

// encodeReturn reads each requested field's post-apply raw bits (0 if the index /
// entry / entry field is absent) and encodes them i64 BE in request order.
func encodeReturn(rec *operateRec, ret []wire.OperateRet) []byte {
	if len(ret) == 0 {
		return nil
	}
	vals := make([]int64, len(ret))
	for i, r := range ret {
		vals[i] = rec.readField(r.Target, r.EntryKey, r.FieldIdx)
	}
	return wire.EncodeOperateResult(vals)
}

func (rec *operateRec) readField(target uint8, entryKey uint64, fieldIdx uint16) int64 {
	var arr []operateField
	if target == wire.OperateTargetEntry {
		e, ok := rec.entries[entryKey]
		if !ok {
			return 0
		}
		arr = e.fields
	} else {
		arr = rec.globals
	}
	if int(fieldIdx) < len(arr) {
		return fieldRaw(arr[fieldIdx])
	}
	return 0
}

// fieldRaw returns a field's value as raw i64 bits: the integer value for
// int/varint types, or the IEEE bit pattern for floats (F32 in the low 32 bits).
func fieldRaw(f operateField) int64 {
	switch f.typ {
	case wire.OperateTypeF64:
		return int64(math.Float64bits(f.f)) //nolint:gosec // raw bits
	case wire.OperateTypeF32:
		return int64(uint64(math.Float32bits(float32(f.f)))) //nolint:gosec // raw bits (low 32)
	default:
		return int64(f.u) //nolint:gosec // raw value bits
	}
}

// --- type property helpers ---------------------------------------------------

func typeIsFloat(t uint8) bool { return t == wire.OperateTypeF32 || t == wire.OperateTypeF64 }

func typeIsUnsigned(t uint8) bool {
	switch t {
	case wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64, wire.OperateTypeUVARINT:
		return true
	default:
		return false
	}
}

// typeIsFixedInt reports the fixed-width integer types (U8..I64) — the ones
// SHIFTOR's width mask is defined for.
func typeIsFixedInt(t uint8) bool {
	return t <= wire.OperateTypeI64 // U8..I64 are contiguous 0..7
}

func typeUMax(t uint8) uint64 {
	switch t {
	case wire.OperateTypeU8:
		return math.MaxUint8
	case wire.OperateTypeU16:
		return math.MaxUint16
	case wire.OperateTypeU32:
		return math.MaxUint32
	default: // U64, UVARINT
		return math.MaxUint64
	}
}

func typeSignedRange(t uint8) (lo, hi int64) {
	switch t {
	case wire.OperateTypeI8:
		return math.MinInt8, math.MaxInt8
	case wire.OperateTypeI16:
		return math.MinInt16, math.MaxInt16
	case wire.OperateTypeI32:
		return math.MinInt32, math.MaxInt32
	default: // I64, IVARINT
		return math.MinInt64, math.MaxInt64
	}
}

func typeBits(t uint8) int {
	switch t {
	case wire.OperateTypeU8, wire.OperateTypeI8:
		return 8
	case wire.OperateTypeU16, wire.OperateTypeI16:
		return 16
	case wire.OperateTypeU32, wire.OperateTypeI32:
		return 32
	default:
		return 64
	}
}

func widthMask(bits int) uint64 {
	if bits >= 64 {
		return math.MaxUint64
	}
	return (uint64(1) << uint(bits)) - 1
}

func signExtend(v uint64, bits int) int64 {
	if bits >= 64 {
		return int64(v) //nolint:gosec // full width
	}
	shift := uint(64 - bits)
	return int64(v<<shift) >> shift //nolint:gosec // arithmetic shift sign-extends
}

func floatFromArg(t uint8, arg int64) float64 {
	if t == wire.OperateTypeF32 {
		return float64(math.Float32frombits(uint32(arg))) //nolint:gosec // low 32 bits are the F32 pattern
	}
	return math.Float64frombits(uint64(arg)) //nolint:gosec // 64-bit IEEE pattern
}

func canonFloat(t uint8, x float64) float64 {
	if t == wire.OperateTypeF32 {
		return float64(float32(x)) // round to F32 precision so storage round-trips exactly
	}
	return x
}

// addSatUnsigned adds a signed delta to an unsigned value, saturating at [0, max]
// (never wrapping). uint64(-delta) yields the correct magnitude even for MinInt64.
func addSatUnsigned(cur uint64, delta int64, max uint64) uint64 {
	if delta >= 0 {
		d := uint64(delta)
		if d > max-cur {
			return max
		}
		return cur + d
	}
	d := uint64(-delta) //nolint:gosec // magnitude; correct even at MinInt64
	if d > cur {
		return 0
	}
	return cur - d
}

// addSatSigned adds delta to cur, saturating at [lo, hi] and detecting the int64
// wrap first (so I64 saturates at the int64 boundary instead of wrapping).
func addSatSigned(cur, delta, lo, hi int64) int64 {
	sum := cur + delta
	switch {
	case delta > 0 && sum < cur:
		return hi // overflowed positive
	case delta < 0 && sum > cur:
		return lo // overflowed negative
	case sum < lo:
		return lo
	case sum > hi:
		return hi
	default:
		return sum
	}
}

// --- record encode / decode --------------------------------------------------

// decodeRec parses a stored operate value:
//
//	[nGlobals u16][ field × nGlobals ]
//	[nEntries u32]
//	  per entry: [key u64][touchMs i64][nFields u16][ field × nFields ]
//	  field:     [type u8][data]  (data: native width; F32/F64 4/8 bytes; U/IVARINT LEB128)
//
// Integers/floats are LITTLE-ENDIAN (the value model's own encoding; the arg wire
// is big-endian, independently). nil/empty bytes decode to an empty record.
// Because operate reads whatever value is stored under the key — which a plain put
// could have set to arbitrary bytes — this decoder is HARDENED like the wire
// decoders: every count is CountFitsIn-bounded before allocation, each field type
// tag is validated, each varint is read with binary.Uvarint (bounded to 10 bytes,
// no over-read), and every read is truncation-checked, so a malformed value
// returns an error, never a panic (which on the replicated apply path would take
// down every replica).
func decodeRec(b []byte) (operateRec, error) {
	rec := operateRec{entries: make(map[uint64]*operateEntry)}
	if len(b) == 0 {
		return rec, nil
	}
	off := 0
	if len(b)-off < 2 {
		return operateRec{}, wire.ErrShortArgs
	}
	nGlobals := int(binary.LittleEndian.Uint16(b[off : off+2]))
	off += 2
	if !wire.CountFitsIn(nGlobals, len(b)-off, minOperateFieldBytes) {
		return operateRec{}, wire.ErrShortArgs
	}
	if nGlobals > 0 {
		rec.globals = make([]operateField, nGlobals)
		for i := range rec.globals {
			f, n, err := decodeField(b[off:])
			if err != nil {
				return operateRec{}, err
			}
			rec.globals[i] = f
			off += n
		}
	}
	if len(b)-off < 4 {
		return operateRec{}, wire.ErrShortArgs
	}
	nEntries := int(binary.LittleEndian.Uint32(b[off : off+4]))
	off += 4
	if !wire.CountFitsIn(nEntries, len(b)-off, minOperateEntryBytes) {
		return operateRec{}, wire.ErrShortArgs
	}
	for range nEntries {
		if len(b)-off < minOperateEntryBytes {
			return operateRec{}, wire.ErrShortArgs
		}
		key := binary.LittleEndian.Uint64(b[off : off+8])
		off += 8
		touchMs := int64(binary.LittleEndian.Uint64(b[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64
		off += 8
		nFields := int(binary.LittleEndian.Uint16(b[off : off+2]))
		off += 2
		if !wire.CountFitsIn(nFields, len(b)-off, minOperateFieldBytes) {
			return operateRec{}, wire.ErrShortArgs
		}
		e := &operateEntry{touchMs: touchMs}
		if nFields > 0 {
			e.fields = make([]operateField, nFields)
			for i := range e.fields {
				f, n, err := decodeField(b[off:])
				if err != nil {
					return operateRec{}, err
				}
				e.fields[i] = f
				off += n
			}
		}
		rec.entries[key] = e
	}
	return rec, nil
}

// decodeField reads one [type u8][data] field from the front of b, returning the
// field and the number of bytes consumed. It rejects an unknown type tag and any
// truncation, and reads varints with binary.Uvarint (self-bounding, no over-read).
func decodeField(b []byte) (operateField, int, error) {
	if len(b) < 1 {
		return operateField{}, 0, wire.ErrShortArgs
	}
	t := b[0]
	if t >= wire.OperateTypeCount {
		return operateField{}, 0, errOperateType
	}
	off := 1
	f := operateField{typ: t}
	need := func(n int) bool { return len(b)-off >= n }
	switch t {
	case wire.OperateTypeU8:
		if !need(1) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(b[off])
		off++
	case wire.OperateTypeI8:
		if !need(1) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(int64(int8(b[off]))) //nolint:gosec // sign-extend
		off++
	case wire.OperateTypeU16:
		if !need(2) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(binary.LittleEndian.Uint16(b[off : off+2]))
		off += 2
	case wire.OperateTypeI16:
		if !need(2) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(int64(int16(binary.LittleEndian.Uint16(b[off : off+2])))) //nolint:gosec // sign-extend
		off += 2
	case wire.OperateTypeU32:
		if !need(4) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(binary.LittleEndian.Uint32(b[off : off+4]))
		off += 4
	case wire.OperateTypeI32:
		if !need(4) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(int64(int32(binary.LittleEndian.Uint32(b[off : off+4])))) //nolint:gosec // sign-extend
		off += 4
	case wire.OperateTypeU64, wire.OperateTypeI64:
		if !need(8) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = binary.LittleEndian.Uint64(b[off : off+8])
		off += 8
	case wire.OperateTypeF32:
		if !need(4) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.f = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[off : off+4])))
		off += 4
	case wire.OperateTypeF64:
		if !need(8) {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.f = math.Float64frombits(binary.LittleEndian.Uint64(b[off : off+8]))
		off += 8
	case wire.OperateTypeUVARINT:
		v, n := binary.Uvarint(b[off:])
		if n <= 0 { // 0 = truncated, <0 = overflow (> 10 bytes / > 64 bits)
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = v
		off += n
	case wire.OperateTypeIVARINT:
		v, n := binary.Uvarint(b[off:])
		if n <= 0 {
			return operateField{}, 0, wire.ErrShortArgs
		}
		f.u = uint64(unzigzag(v)) //nolint:gosec // store signed value bits
		off += n
	default:
		return operateField{}, 0, errOperateType
	}
	return f, off, nil
}

// encodeRec serializes rec in the decodeRec layout, LITTLE-ENDIAN, with entries
// emitted in ASCENDING key order and fields in index order so the bytes are a
// deterministic function of the record's contents (independent of Go map iteration
// order) — the property a follower's byte-identical re-apply relies on.
func encodeRec(rec operateRec) []byte {
	keys := make([]uint64, 0, len(rec.entries))
	for k := range rec.entries {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	buf := make([]byte, 0, 6+len(rec.globals)*3+len(keys)*minOperateEntryBytes)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(rec.globals))) //nolint:gosec // bounded by maxOperateFields
	for i := range rec.globals {
		buf = encodeField(buf, rec.globals[i])
	}
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(keys))) //nolint:gosec // entry count
	for _, k := range keys {
		e := rec.entries[k]
		buf = binary.LittleEndian.AppendUint64(buf, k)
		buf = binary.LittleEndian.AppendUint64(buf, uint64(e.touchMs))     //nolint:gosec // reinterpret i64 as u64
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(e.fields))) //nolint:gosec // bounded by maxOperateFields
		for i := range e.fields {
			buf = encodeField(buf, e.fields[i])
		}
	}
	return buf
}

// encodeField appends one [type u8][data] field: native fixed width for U8..I64,
// 4/8 bytes for F32/F64, or LEB128 for U/IVARINT (IVARINT zigzagged).
func encodeField(buf []byte, f operateField) []byte {
	buf = append(buf, f.typ)
	switch f.typ {
	case wire.OperateTypeU8, wire.OperateTypeI8:
		buf = append(buf, byte(f.u))
	case wire.OperateTypeU16, wire.OperateTypeI16:
		buf = binary.LittleEndian.AppendUint16(buf, uint16(f.u)) //nolint:gosec // low 16 bits
	case wire.OperateTypeU32, wire.OperateTypeI32:
		buf = binary.LittleEndian.AppendUint32(buf, uint32(f.u)) //nolint:gosec // low 32 bits
	case wire.OperateTypeU64, wire.OperateTypeI64:
		buf = binary.LittleEndian.AppendUint64(buf, f.u)
	case wire.OperateTypeF32:
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(float32(f.f)))
	case wire.OperateTypeF64:
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(f.f))
	case wire.OperateTypeUVARINT:
		buf = binary.AppendUvarint(buf, f.u)
	case wire.OperateTypeIVARINT:
		buf = binary.AppendUvarint(buf, zigzag(int64(f.u))) //nolint:gosec // signed value bits
	}
	return buf
}

func zigzag(i int64) uint64   { return uint64((i << 1) ^ (i >> 63)) } //nolint:gosec // standard zigzag
func unzigzag(u uint64) int64 { return int64(u>>1) ^ -int64(u&1) }    //nolint:gosec // standard zigzag
