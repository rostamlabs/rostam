// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"io"
	"time"
)

// Named-collection WAL record codec, layered on the family-agnostic framed-log
// core (wal.appendFramed / replayFramed in wal.go). It mirrors the dense codec
// (appendInsertStaged/appendDeleteStaged/appendSetPayloadStaged + replayRecord) but encodes the
// named family's shape: an Insert carries a MAP of per-space vectors (a point may
// populate any subset of the configured spaces), and the shared per-point payload
// is logged op-agnostically as the RESULTING full payload (set/overwrite/
// delete-keys/clear all collapse to one namedSetPayload record — exactly the
// dense resulting-payload discipline).
//
// Per-key TTL deadlines are ABSOLUTE unix-millis (named's keyTTL is absolute), so
// they are logged and replayed VERBATIM — replay never recomputes now+ttl, which
// keeps a pending per-key deadline time-stable across a crash (advancing the
// clock after recovery still expires the key at its original absolute time).
//
// The record tags are an INDEPENDENT namespace from the dense walRecType (named
// WALs and dense WALs are never the same file), so they restart at 1.

type namedWALRecType uint8

const (
	namedInsertRec namedWALRecType = 1
	namedDeleteRec namedWALRecType = 2
	// namedSetPayloadRec logs a payload mutation as the RESULTING full shared
	// payload (op-agnostic: SetPayload-merge / Overwrite / DeleteKeys / Clear all
	// reduce to "this is the new payload"), plus the resulting ABSOLUTE per-key
	// deadlines. Replay = restorePayload (verbatim, no recompute).
	namedSetPayloadRec namedWALRecType = 3
)

// appendNamedInsertStaged logs a successful named Insert: the point id, its TTL in
// milliseconds (replayed TTLs are restored ABSOLUTE via the snapshot/restore
// path; the relative ms here lets replay recompute a remaining deadline measured
// from recovery time, a bounded inaccuracy after a crash — matching dense
// appendInsertStaged semantics for the point TTL), the per-space vectors, the shared
// payload, and the absolute per-key deadlines captured at insert time (always
// empty for Insert, which drops prior per-key TTL — encoded for symmetry/forward
// use). The vectors map is framed [count][ (name, dim, floats) ...].
//
// WRITE phase only (see wal.appendFramedStaged): it returns the assigned commit
// sequence instead of blocking on the fsync, so insertCASKeyTTLBody can release
// opMu before waiting and group-commit with concurrent writers. The caller MUST
// commitWaitStaged(seq). There is deliberately no blocking wrapper — every named
// insert record goes through the staged pair (see appendSetPayloadStaged).
func (w *wal) appendNamedInsertStaged(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, ttl time.Duration, payload Metadata, keyExpires map[string]uint64, version uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(namedInsertRec))
	_ = writeU64(&buf, id)
	_ = writeU64(&buf, uint64(ttl.Milliseconds())) //nolint:gosec
	_ = writeU32(&buf, uint32(len(vectors)))       //nolint:gosec
	for name, vec := range vectors {
		_ = writeString(&buf, name)
		_ = writeU32(&buf, uint32(len(vec))) //nolint:gosec
		for _, f := range vec {
			_ = writeF32(&buf, f)
		}
	}
	if err := writeOptMeta(&buf, payload); err != nil {
		return 0, err
	}
	writeOptKeyExpires(&buf, keyExpires)
	// Trailing per-point CAS version block (byte-identical when 0; an old record
	// without it replays as version 0 -> a fresh insert defaults to 1). Reuses the
	// shared dense writeOptVersion codec.
	writeOptVersion(&buf, version)
	// Trailing optional SPARSE block, AFTER the version. ENTIRELY OMITTED (zero
	// bytes) when there are no sparse values, so a dense-only named WAL record is
	// BYTE-IDENTICAL to the pre-sparse format (the per-record framing isolates the
	// tail; replay's readOptSparseVectors tolerates EOF → nil). Only written when a
	// sparse space carries a value for this point.
	writeNamedSparseVectors(&buf, sparseVectors)
	return w.appendFramedStaged(buf.Bytes())
}

// writeNamedSparseVectors appends the optional per-space sparse-value block. To
// keep a dense-only record byte-identical it writes NOTHING when sparseVectors is
// empty (no flag byte at all — the framing + EOF-tolerant reader cover the
// absence). Otherwise it writes a present flag (1), a u32 space count, then per
// space [string name][sparsevec], where sparsevec is the shared framing
// [nIdx:u32]{idx:u32...}[nVal:u32]{val:f32...} (writeSparseVecFrame). Spaces whose
// value is nil/zero are skipped (they carry no sparse value).
func writeNamedSparseVectors(buf *bytes.Buffer, sparseVectors map[string]*SparseVector) {
	n := 0
	for _, sv := range sparseVectors {
		if sv != nil && !sv.IsZero() {
			n++
		}
	}
	if n == 0 {
		return // dense-only: byte-identical, no trailing block
	}
	buf.WriteByte(1)
	_ = writeU32(buf, uint32(n)) //nolint:gosec
	for name, sv := range sparseVectors {
		if sv == nil || sv.IsZero() {
			continue
		}
		_ = writeString(buf, name)
		writeSparseVecFrame(buf, sv)
	}
}

// readNamedSparseVectors is the inverse of writeNamedSparseVectors. It tolerates
// EOF (a dense-only record with no trailing block) by returning (nil, true). A
// present flag with a truncated body returns (nil, false) so replay stops at the
// durability boundary.
func readNamedSparseVectors(r *bytes.Reader) (map[string]*SparseVector, bool) {
	var flag [1]byte
	if _, err := io.ReadFull(r, flag[:]); err != nil {
		return nil, true // EOF: dense-only record, no sparse block
	}
	if flag[0] != 1 {
		return nil, false
	}
	n, err := readU32(r)
	if err != nil {
		return nil, false
	}
	out := make(map[string]*SparseVector, n)
	for i := uint32(0); i < n; i++ {
		name, err := readString(r)
		if err != nil {
			return nil, false
		}
		sv, ok := readSparseVecFrame(r)
		if !ok {
			return nil, false
		}
		out[name] = sv
	}
	return out, true
}

// appendNamedDeleteStaged logs a successful named Delete, WRITE phase only (see
// wal.appendFramedStaged): it returns the assigned commit sequence instead of
// blocking on the fsync so DeleteCAS can wait outside opMu. The caller MUST
// commitWaitStaged(seq).
func (w *wal) appendNamedDeleteStaged(id uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(namedDeleteRec))
	_ = writeU64(&buf, id)
	return w.appendFramedStaged(buf.Bytes())
}

// appendNamedSetPayloadStaged logs a payload mutation as the resulting full
// payload (meta) plus the resulting ABSOLUTE per-key deadlines (keyExpires: key
// -> unix-millis). The encoding reuses the dense writeOptMeta/writeOptKeyExpires
// codec so a nil meta encodes a cleared payload and an empty deadline map costs a
// single flag byte.
//
// WRITE phase only (see wal.appendFramedStaged): it returns the assigned commit
// sequence instead of blocking on the fsync so logPayloadOp can wait outside
// opMu. The caller MUST commitWaitStaged(seq).
func (w *wal) appendNamedSetPayloadStaged(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(namedSetPayloadRec))
	_ = writeU64(&buf, id)
	if err := writeOptMeta(&buf, meta); err != nil {
		return 0, err
	}
	writeOptKeyExpires(&buf, keyExpires)
	// Trailing per-point CAS version (the resulting version), restored VERBATIM by
	// replay. Byte-identical when 0 (an old record replays as version 0 -> leaves
	// the existing version untouched).
	writeOptVersion(&buf, version)
	return w.appendFramedStaged(buf.Bytes())
}

// replayNamedWAL replays a named WAL's records onto the (just-restored)
// NamedCollection via the shared replayFramed core. The per-record apply switches
// on the named tag and re-calls the idempotent mutator with VERBATIM absolute
// per-key deadlines (restorePayload, not SetPayload — replay must not recompute
// now+ttl). It stops at the first malformed/unknown record (returning false),
// which is the same torn-tail/durability boundary the framed core enforces for a
// truncated final record. A missing file replays nothing.
func replayNamedWAL(path string, nc *NamedCollection) error {
	return replayFramed(path, func(rec []byte) error {
		return replayNamedRecord(rec, nc)
	})
}

// replayNamedRecord decodes one named record payload and re-applies it onto nc.
// Returns false on a malformed body so replay stops at the durability boundary
// (mirrors dense replayRecord). Apply errors from the mutators are swallowed —
// replay is idempotent on top of the snapshot checkpoint.
func replayNamedRecord(payload []byte, nc *NamedCollection) error {
	r := bytes.NewReader(payload)
	t, err := r.ReadByte()
	if err != nil {
		return errStopReplay
	}
	switch namedWALRecType(t) {
	case namedDeleteRec:
		id, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		_, _ = nc.Delete(id)
		return nil
	case namedSetPayloadRec:
		id, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		meta, ok := readOptMeta(r)
		if !ok {
			return errStopReplay
		}
		ke, ok := readOptKeyExpires(r)
		if !ok {
			return errStopReplay
		}
		version, ok := readOptVersion(r)
		if !ok {
			return errStopReplay
		}
		nc.restorePayload(id, meta, ke, version) // verbatim version (NOT re-bumped)
		return nil
	case namedInsertRec:
		id, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		ttlMs, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		nSpaces, err := readU32(r)
		if err != nil {
			return errStopReplay
		}
		vectors := make(map[string][]float32, nSpaces)
		for i := uint32(0); i < nSpaces; i++ {
			name, err := readString(r)
			if err != nil {
				return errStopReplay
			}
			dim, err := readU32(r)
			if err != nil || dim > uint32(maxWALRecord/4) {
				return errStopReplay
			}
			vec := make([]float32, dim)
			for j := range vec {
				if vec[j], err = readF32(r); err != nil {
					return errStopReplay
				}
			}
			vectors[name] = vec
		}
		meta, ok := readOptMeta(r)
		if !ok {
			return errStopReplay
		}
		// Restore the logged ABSOLUTE per-key deadlines VERBATIM (NOT recomputed
		// now+ttl) so a pending insert-time per-key TTL survives a crash time-stable.
		// An OLD record (no keyExpires block, or an Insert that carried none) reads an
		// empty map → RestoreInsert clears the per-key deadlines (the no-key_ttl path).
		keyExpires, ok := readOptKeyExpires(r)
		if !ok {
			return errStopReplay
		}
		version, ok := readOptVersion(r)
		if !ok {
			return errStopReplay
		}
		// Trailing optional sparse block (after the version). A dense-only record has
		// none → readNamedSparseVectors returns (nil, true) at EOF, byte-identical to
		// the pre-sparse format.
		sparseVectors, ok := readNamedSparseVectors(r)
		if !ok {
			return errStopReplay
		}
		// RestoreInsert sets the version VERBATIM (NOT re-bumped); a 0 version (an old
		// record predating the version block) defaults a fresh insert to 1.
		_ = nc.RestoreInsert(id, vectors, sparseVectors, meta, time.Duration(ttlMs)*time.Millisecond, keyExpires, version) //nolint:gosec
		return nil
	default:
		return errUnknownWALRecord
	}
}
