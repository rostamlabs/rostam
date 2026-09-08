// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"io"
)

// Multi-vector (late-interaction) WAL record codec, layered on the family-agnostic
// framed-log core (wal.appendFramed / replayFramed in wal.go). It mirrors the
// named codec (appendNamed* / replayNamedWAL) but encodes the MV family's shape:
// an Add carries a TOKEN MATRIX (a document is many token vectors, all of the
// same dim), and the shared per-document payload is logged op-agnostically as the
// RESULTING full payload (set/overwrite/delete-keys/clear all collapse to one
// mvSetPayload record — exactly the dense/named resulting-payload discipline).
//
// Per-key TTL deadlines are ABSOLUTE unix-millis (MV's keyTTL is absolute), so
// they are logged and replayed VERBATIM — replay never recomputes now+ttl, which
// keeps a pending per-key deadline time-stable across a crash (advancing the
// clock after recovery still expires the key at its original absolute time).
//
// The record tags are an INDEPENDENT namespace from the dense walRecType and the
// named tags (MV WALs are never the same file as dense or named WALs), so they
// restart at 1.

type mvWALRecType uint8

const (
	mvAddRec    mvWALRecType = 1
	mvDeleteRec mvWALRecType = 2
	// mvSetPayloadRec logs a payload mutation as the RESULTING full payload
	// (op-agnostic: SetPayload-merge / Overwrite / DeleteKeys / Clear all reduce to
	// "this is the new payload"), plus the resulting ABSOLUTE per-key deadlines.
	// Replay = restorePayload (verbatim, no recompute).
	mvSetPayloadRec mvWALRecType = 3
)

// appendMVAddStaged logs a successful MV Add: the document id, its token matrix
// (each row a token vector, all of dim Dim), the shared payload, and the absolute
// per-key deadlines captured at add time (always empty for Add, which drops prior
// per-key TTL — encoded for symmetry/forward use). The token matrix is framed
// [count][ (dim, floats) ... ].
//
// WRITE phase only (see wal.appendFramedStaged): it returns the assigned commit
// sequence instead of blocking on the fsync, so the caller can release opMu before
// waiting and group-commit with concurrent writers. The caller MUST
// commitWaitStaged(seq). There is deliberately no blocking wrapper — every MV add
// record goes through the staged pair (see appendSetPayloadStaged for the rule).
func (w *wal) appendMVAddStaged(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(mvAddRec))
	_ = writeU64(&buf, docID)
	_ = writeU32(&buf, uint32(len(tokens))) //nolint:gosec
	for _, t := range tokens {
		_ = writeU32(&buf, uint32(len(t))) //nolint:gosec
		for _, f := range t {
			_ = writeF32(&buf, f)
		}
	}
	if err := writeOptMeta(&buf, meta); err != nil {
		return 0, err
	}
	writeOptKeyExpires(&buf, keyExpires)
	// Trailing per-doc CAS version block (byte-identical when 0; an old record
	// without it replays as version 0 -> a fresh add defaults to 1). Reuses the
	// shared dense writeOptVersion codec.
	writeOptVersion(&buf, version)
	// Trailing OPTIONAL doc-level sparse block, AFTER the version. ENTIRELY OMITTED
	// (zero bytes) when there is no sparse vector, so a dense-only MV WAL record is
	// BYTE-IDENTICAL to the pre-sparse format (the per-record framing isolates the
	// tail; replay's readOptMVSparse tolerates EOF → nil).
	writeOptMVSparse(&buf, sparse)
	return w.appendFramedStaged(buf.Bytes())
}

// writeOptMVSparse appends the optional doc-level sparse block. To keep a
// dense-only record byte-identical it writes NOTHING when sparse is nil/zero (no
// flag byte at all — the framing + EOF-tolerant reader cover the absence).
// Otherwise it writes a present flag (1) then the shared sparse framing via
// writeSparseVecFrame.
func writeOptMVSparse(buf *bytes.Buffer, sparse *SparseVector) {
	if sparse == nil || sparse.IsZero() {
		return // dense-only: byte-identical, no trailing block
	}
	buf.WriteByte(1)
	writeSparseVecFrame(buf, sparse)
}

// readOptMVSparse is the inverse of writeOptMVSparse. It tolerates EOF (a
// dense-only record with no trailing block) by returning (nil, true). A present
// flag with a truncated body returns (nil, false) so replay stops at the
// durability boundary.
func readOptMVSparse(r *bytes.Reader) (*SparseVector, bool) {
	var flag [1]byte
	if _, err := io.ReadFull(r, flag[:]); err != nil {
		return nil, true // EOF: dense-only record, no sparse block
	}
	if flag[0] != 1 {
		return nil, false
	}
	sv, ok := readSparseVecFrame(r)
	if !ok {
		return nil, false
	}
	return sv, true
}

// appendMVDeleteStaged logs a successful MV Delete, WRITE phase only (see
// wal.appendFramedStaged): it returns the assigned commit sequence instead of
// blocking on the fsync so DeleteCAS can wait outside opMu. The caller MUST
// commitWaitStaged(seq).
func (w *wal) appendMVDeleteStaged(docID uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(mvDeleteRec))
	_ = writeU64(&buf, docID)
	return w.appendFramedStaged(buf.Bytes())
}

// appendMVSetPayloadStaged logs a payload mutation as the resulting full payload
// (meta) plus the resulting ABSOLUTE per-key deadlines (keyExpires: key ->
// unix-millis). The encoding reuses the shared writeOptMeta/writeOptKeyExpires
// codec so a nil meta encodes a cleared payload and an empty deadline map costs a
// single flag byte.
//
// WRITE phase only (see wal.appendFramedStaged): it returns the assigned commit
// sequence instead of blocking on the fsync so logPayloadOp can wait outside
// opMu. The caller MUST commitWaitStaged(seq).
func (w *wal) appendMVSetPayloadStaged(docID uint64, meta Metadata, keyExpires map[string]uint64, version uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(mvSetPayloadRec))
	_ = writeU64(&buf, docID)
	if err := writeOptMeta(&buf, meta); err != nil {
		return 0, err
	}
	writeOptKeyExpires(&buf, keyExpires)
	// Trailing per-doc CAS version (the resulting version), restored VERBATIM by
	// replay. Byte-identical when 0.
	writeOptVersion(&buf, version)
	return w.appendFramedStaged(buf.Bytes())
}

// replayMVWAL replays an MV WAL's records onto the (just-restored)
// MultiVectorIndex via the shared replayFramed core. The per-record apply switches
// on the MV tag and re-calls the idempotent mutator with VERBATIM absolute per-key
// deadlines (restorePayload, not SetPayload — replay must not recompute now+ttl).
// It stops at the first malformed/unknown record (returning false), the same
// torn-tail/durability boundary the framed core enforces for a truncated final
// record. A missing file replays nothing.
func replayMVWAL(path string, m *MultiVectorIndex) error {
	return replayFramed(path, func(rec []byte) error {
		return replayMVRecord(rec, m)
	})
}

// replayMVRecord decodes one MV record payload and re-applies it onto m. Returns
// false on a malformed body so replay stops at the durability boundary (mirrors
// named replayNamedRecord). Apply errors from the mutators are swallowed — replay
// is idempotent on top of the snapshot checkpoint.
func replayMVRecord(payload []byte, m *MultiVectorIndex) error {
	r := bytes.NewReader(payload)
	t, err := r.ReadByte()
	if err != nil {
		return errStopReplay
	}
	switch mvWALRecType(t) {
	case mvDeleteRec:
		docID, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		_ = m.Delete(docID)
		return nil
	case mvSetPayloadRec:
		docID, err := readU64(r)
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
		m.restorePayload(docID, meta, ke, version) // verbatim version (NOT re-bumped)
		return nil
	case mvAddRec:
		docID, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		nTokens, err := readU32(r)
		if err != nil || nTokens > uint32(maxWALRecord/4) {
			return errStopReplay
		}
		// Bound nTokens against the framed record length before pre-sizing the
		// slice headers: each token contributes at least a 4-byte dim field, so a
		// record claiming more than len(payload)/4 tokens is corrupt. Without this
		// a tiny record could pre-allocate millions of empty slice headers.
		if uint64(nTokens)*4 > uint64(len(payload)) {
			return errStopReplay
		}
		tokens := make([][]float32, nTokens)
		for i := uint32(0); i < nTokens; i++ {
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
			tokens[i] = vec
		}
		meta, ok := readOptMeta(r)
		if !ok {
			return errStopReplay
		}
		// Restore the logged ABSOLUTE per-key deadlines VERBATIM (NOT recomputed
		// now+ttl) so a pending add-time per-key TTL survives a crash time-stable. An
		// OLD record (no keyExpires block, or an Add that carried none) reads an empty
		// map → restoreAdd sets no per-key deadlines (the no-key_ttl path).
		keyExpires, ok := readOptKeyExpires(r)
		if !ok {
			return errStopReplay
		}
		version, ok := readOptVersion(r)
		if !ok {
			return errStopReplay
		}
		// Optional trailing doc-level sparse block (AFTER the version). An OLD record
		// (no block) reads nil → the dense-only path; a present-but-truncated block
		// stops replay at the durability boundary.
		sparse, ok := readOptMVSparse(r)
		if !ok {
			return errStopReplay
		}
		// restoreAdd sets the version VERBATIM (NOT re-bumped); a 0 version (an old
		// record predating the version block) defaults a fresh add to 1. The optional
		// doc sparse is restored verbatim too.
		_ = m.restoreAdd(docID, tokens, meta, keyExpires, version, sparse)
		return nil
	default:
		return errUnknownWALRecord
	}
}
