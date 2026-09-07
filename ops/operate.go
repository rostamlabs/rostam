// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"errors"
	"sort"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// operate is the generic atomic multi-field update op (the native alternative to a
// client CAS-retry loop on a HOT key). One Rostam value is modelled as:
//
//   - globals: a small fixed []i64 array of counters.
//   - entries: a map[u64]*operateEntry of dynamic sub-records, each a []i64 plus a
//     leader-stamped touchMs used for DETERMINISTIC LRU eviction under a cap.
//
// The client sends an op-list (INCR / INCRF / SETMAX / SHIFTOR / HALVE_GRP); the
// handler decodes the value, applies every op under the shard write lock (pure
// integer arithmetic, the ONLY external input being the leader-stamped clock), and
// writes it back in one round-trip. Rostam stays schema-agnostic: the app owns
// which index means what, so a new counter is an app-side change with no rebuild.
//
// DETERMINISM (RF>1 safety): every mutation is integer-only; touchMs and the TTL
// come solely from tx.applyStamp() (the leader-stamped nowMs), so a follower
// re-applying the same committed op-list recomputes byte-identical state. The
// encoded record sorts entries by key, and eviction picks the smallest touchMs
// (ties → smallest key), so neither the stored bytes nor the evicted victim depend
// on Go map iteration order.

// maxOperateFields caps a globals/entry-fields array LENGTH. The record encoding
// prefixes each array with a u16 count (nGlobals / nFields), so an array can hold
// at most 65535 fields; a fieldIdx that would grow it past that is rejected. A
// field index is a u16 on the wire, so the only rejected index is 65535 (which
// would need length 65536) — 64K counters per array is ample for a session record.
const maxOperateFields = 65535

// minOperateEntryBytes is the smallest honest encoded entry: key(8) + touchMs(8) +
// nFields(2), with zero fields. Used to bound the declared entry count in
// decodeRec before any allocation (CountFitsIn), the same discipline the wire
// batch decoders use.
const minOperateEntryBytes = 18

var (
	errOperateOpcode     = errors.New("ops: operate unknown opcode")
	errOperateShift      = errors.New("ops: operate SHIFTOR negative shift")
	errOperateGroup      = errors.New("ops: operate HALVE_GRP group length < 1")
	errOperateFieldRange = errors.New("ops: operate field index out of range")
)

// operateEntry is one dynamic sub-record: a small []i64 plus the leader-stamped
// last-touch time driving deterministic LRU eviction.
type operateEntry struct {
	touchMs int64
	fields  []int64
}

// operateRec is the decoded value: the globals array and the entry map.
type operateRec struct {
	globals []int64
	entries map[uint64]*operateEntry
}

// handleOperate applies a client op-list to one record atomically under the shard
// write lock. On any op error (unknown opcode, negative shift, out-of-range field)
// it returns BEFORE the Put, so a rejected op-list leaves the record unchanged —
// the whole operate is all-or-nothing. Registered OpReadWrite.
func handleOperate(tx *TxContext, args []byte) ([]byte, error) {
	key, ttl, maxEntries, opsList, ret, err := wire.DecodeOperateArgs(args)
	if err != nil {
		return nil, err
	}
	cur, err := tx.Get(key)
	if err != nil && err != cache.ErrNotFound {
		return nil, err
	}
	// cur aliases the page but decodeRec copies every int out into fresh slices, so
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
// needed (bounded by maxOperateFields).
func applyOp(rec *operateRec, o wire.OperateOp, nowMs int64, maxEntries int) error {
	arr, err := rec.targetArray(o, nowMs, maxEntries)
	if err != nil {
		return err
	}
	switch o.Opcode {
	case wire.OperateOpINCR, wire.OperateOpINCRF:
		// INCRF is fixed-point add — arithmetically the same i64 addition; the
		// fixed-point scale lives in the app's interpretation, not here.
		a, ok := growI64(*arr, int(o.FieldIdx)+1)
		if !ok {
			return errOperateFieldRange
		}
		a[o.FieldIdx] += o.Arg
		*arr = a
	case wire.OperateOpSETMAX:
		a, ok := growI64(*arr, int(o.FieldIdx)+1)
		if !ok {
			return errOperateFieldRange
		}
		if o.Arg > a[o.FieldIdx] {
			a[o.FieldIdx] = o.Arg
		}
		*arr = a
	case wire.OperateOpSHIFTOR:
		// A negative shift count panics in Go, so reject it deterministically. A
		// count >= 64 shifts every bit out (f<<64 == 0), so the result is just arg2 —
		// computed explicitly to stay off the wrap path.
		if o.Arg < 0 {
			return errOperateShift
		}
		a, ok := growI64(*arr, int(o.FieldIdx)+1)
		if !ok {
			return errOperateFieldRange
		}
		if o.Arg >= 64 {
			a[o.FieldIdx] = o.Arg2
		} else {
			a[o.FieldIdx] = (a[o.FieldIdx] << uint(o.Arg)) | o.Arg2
		}
		*arr = a
	case wire.OperateOpHALVEGRP:
		// HALVE_GRP(thr=Arg, len=Arg2): the group is the contiguous field range
		// [FieldIdx, FieldIdx+len); FieldIdx is the guard field. If the guard is >=
		// the threshold, halve every field in the group. (The fixed op wire carries
		// one fieldIdx + two i64 args, so a group is expressed as a contiguous range
		// rather than an explicit field list — the design's `HALVE_GRP(thr, f…)` with
		// f… the range starting at the guard.)
		if o.Arg2 < 1 {
			return errOperateGroup
		}
		end := int(o.FieldIdx) + int(o.Arg2)
		a, ok := growI64(*arr, end)
		if !ok {
			return errOperateFieldRange
		}
		if a[o.FieldIdx] >= o.Arg {
			for i := int(o.FieldIdx); i < end; i++ {
				a[i] /= 2 // integer halving; deterministic (rounds toward zero for negatives)
			}
		}
		*arr = a
	default:
		return errOperateOpcode
	}
	return nil
}

// targetArray returns a pointer to the []i64 an op mutates. For a global target
// that is &rec.globals; for an entry target it upserts the sub-record (evicting
// deterministically if a NEW key would exceed the cap), stamps its touchMs, and
// returns &entry.fields. Any op referencing an entry counts as a touch.
func (rec *operateRec) targetArray(o wire.OperateOp, nowMs int64, maxEntries int) (*[]int64, error) {
	if o.Target != wire.OperateTargetEntry {
		return &rec.globals, nil
	}
	e := rec.upsertEntry(o.EntryKey, nowMs, maxEntries)
	e.touchMs = nowMs
	return &e.fields, nil
}

// upsertEntry returns the entry for key, creating an all-zero one (stamped
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
// smallest key. The winner is a deterministic function of the (already
// deterministic) map contents, so it does not depend on Go's map iteration order —
// every replica evicts the same entry.
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

// growI64 returns s grown to at least length n (zero-filled), or s unchanged when
// it is already long enough. It returns ok=false when n exceeds maxOperateFields,
// so a hostile field index cannot drive an unbounded allocation.
func growI64(s []int64, n int) ([]int64, bool) {
	if n > maxOperateFields {
		return s, false
	}
	if len(s) >= n {
		return s, true
	}
	ns := make([]int64, n)
	copy(ns, s)
	return ns, true
}

// encodeReturn reads each requested field's post-apply value (0 if the global
// index / entry / entry field is absent) and encodes them i64 BE in request order.
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
	var arr []int64
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
		return arr[fieldIdx]
	}
	return 0
}

// decodeRec parses a stored operate value:
//
//	[nGlobals u16][global i64 × nGlobals]
//	[nEntries u32]
//	  per entry: [key u64][touchMs i64][nFields u16][field i64 × nFields]
//
// All fields are LITTLE-ENDIAN (the value model's own encoding; the arg wire is
// big-endian, independently). nil/empty bytes decode to an empty record. Because
// operate reads whatever value is stored under the key — which a plain put could
// have set to arbitrary bytes — this decoder is HARDENED like the wire decoders:
// every count is CountFitsIn-bounded before allocation and every read is
// truncation-checked, so a malformed value returns an error, never a panic (which
// on the replicated apply path would take down every replica).
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
	if !wire.CountFitsIn(nGlobals, len(b)-off, 8) {
		return operateRec{}, wire.ErrShortArgs
	}
	if nGlobals > 0 {
		rec.globals = make([]int64, nGlobals)
		for i := range rec.globals {
			rec.globals[i] = int64(binary.LittleEndian.Uint64(b[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64
			off += 8
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
		if !wire.CountFitsIn(nFields, len(b)-off, 8) {
			return operateRec{}, wire.ErrShortArgs
		}
		e := &operateEntry{touchMs: touchMs}
		if nFields > 0 {
			e.fields = make([]int64, nFields)
			for i := range e.fields {
				e.fields[i] = int64(binary.LittleEndian.Uint64(b[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64
				off += 8
			}
		}
		rec.entries[key] = e
	}
	return rec, nil
}

// encodeRec serializes rec in the decodeRec layout, LITTLE-ENDIAN, with entries
// emitted in ASCENDING key order so the bytes are a deterministic function of the
// record's contents (independent of Go map iteration order) — the property a
// follower's byte-identical re-apply relies on. Trailing operations never shrink
// the globals array, so its length is stable across a committed op-list.
func encodeRec(rec operateRec) []byte {
	total := 2 + len(rec.globals)*8 + 4
	keys := make([]uint64, 0, len(rec.entries))
	for k := range rec.entries {
		keys = append(keys, k)
		total += minOperateEntryBytes + len(rec.entries[k].fields)*8
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	buf := make([]byte, 0, total)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(rec.globals))) //nolint:gosec // bounded by maxOperateFields
	for _, g := range rec.globals {
		buf = binary.LittleEndian.AppendUint64(buf, uint64(g)) //nolint:gosec // reinterpret i64 as u64
	}
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(keys))) //nolint:gosec // entry count
	for _, k := range keys {
		e := rec.entries[k]
		buf = binary.LittleEndian.AppendUint64(buf, k)
		buf = binary.LittleEndian.AppendUint64(buf, uint64(e.touchMs))     //nolint:gosec // reinterpret i64 as u64
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(e.fields))) //nolint:gosec // bounded by maxOperateFields
		for _, f := range e.fields {
			buf = binary.LittleEndian.AppendUint64(buf, uint64(f)) //nolint:gosec // reinterpret i64 as u64
		}
	}
	return buf
}
