// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

// Persistence for the mmap-backed (Persistent) MultiVectorIndex. The inner
// token-vector index is saved with the same instant-restart sidecar single-
// vector Persistent collections use (mmap vectors + graph come back zero-copy);
// a parallel "maps" sidecar holds the doc<->token bookkeeping that lives only in
// the MultiVectorIndex (nextToken, docTokens, docMeta). tokenDoc is rebuilt by
// inverting docTokens on open.

const mvMapsMagic = "RMV1"

// ErrMultiVectorNotPersistent is returned by Flush on an in-memory index.
var ErrMultiVectorNotPersistent = errors.New("vector: multi-vector index is not Persistent (nothing to flush)")

// Flush persists a Persistent multi-vector index: it syncs the mmap-backed inner
// index via its instant-restart sidecar and writes the doc<->token maps sidecar.
func (m *MultiVectorIndex) Flush() error {
	if !m.persistent {
		return ErrMultiVectorNotPersistent
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if err := m.idx.SavePersist(m.metaPath); err != nil {
		return err
	}
	return m.writeMapsSidecar()
}

// snapshot writes a self-contained image (inner hnsw snapshot + doc<->token
// maps) to w for inclusion in a cluster FSM snapshot. The inner blob is
// length-prefixed so restore can split it from the maps.
func (m *MultiVectorIndex) snapshot(w io.Writer) error {
	// Refused while the log is poisoned: memory may be ahead of it (see
	// wal.refuseCheckpoint).
	if err := m.wal.refuseCheckpoint(); err != nil {
		return err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	var inner bytes.Buffer
	if err := m.idx.Snapshot(&inner); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, uint32(inner.Len())); err != nil { //nolint:gosec
		return err
	}
	if _, err := w.Write(inner.Bytes()); err != nil {
		return err
	}
	return m.encodeMaps(w) // magic + body
}

// restore rebuilds a fresh index from a snapshot written by snapshot.
func (m *MultiVectorIndex) restore(r io.Reader) error {
	var innerLen uint32
	if err := binary.Read(r, binary.BigEndian, &innerLen); err != nil {
		return err
	}
	innerBuf := make([]byte, innerLen)
	if _, err := io.ReadFull(r, innerBuf); err != nil {
		return err
	}
	if err := m.idx.Restore(bytes.NewReader(innerBuf)); err != nil {
		return err
	}
	magic := make([]byte, len(mvMapsMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return err
	}
	if string(magic) != mvMapsMagic {
		return fmt.Errorf("vector: bad multi-vector snapshot maps magic")
	}
	if err := m.decodeMaps(r); err != nil {
		return err
	}
	// Rebuild the filter-first index from the restored payload (never serialized —
	// the rebuild-on-load path, mirroring named snapshot Restore + dense). decodeMaps
	// writes m.docMeta directly (NOT via the reindex hooks), so the index needs one
	// rebuild here. WAL replay on top (WAL-mode load) mutates incrementally via the
	// per-op reindex hooks; loadMultiVector rebuilds once more after replay as a
	// belt-and-suspenders guarantee.
	m.payloadIdx.rebuild(m.docMeta)
	// Rebuild the doc-level sparse inverted index from the restored docSparse (never
	// serialized — the rebuild-on-load path, mirroring payloadIdx.rebuild). A
	// dense-only restore leaves docSparse empty → an empty sparseIdx.
	m.sparseIdx.rebuild(m.docSparse)
	// Wholesale state replacement: invalidate any cached order snapshots from a prior
	// life of this index (orderSnaps may have been built before this restore).
	m.bumpData()
	return nil
}

// openPersistentMultiVector reopens a Persistent multi-vector index by mapping
// its inner files (instant restart) and reading the maps sidecar. cfg must carry
// the store-managed paths.
//
// This is the mmap instant-restart path, which is HNSW-inner-only: an IVF inner
// index has no mmap sidecar (innerConfig forces its Persistent=false; SavePersist
// is unsupported). An IVF inner index is snapshot-only and restored via the
// generic m.idx.Restore (cluster snapshot / WAL replay), never this mmap path, so
// the store never combines single-node mmap-Persistent with an IVF inner index.
// Guard it fail-loud rather than mis-reopen as the wrong index type.
func openPersistentMultiVector(cfg MultiVectorConfig) (*MultiVectorIndex, error) {
	if cfg.IndexType == IndexIVF {
		return nil, ErrInvalidIVFPersistent
	}
	idx, err := openPersist(mvInnerConfig(cfg), mvMetaPath(cfg))
	if err != nil {
		return nil, err
	}
	m := newMultiShell(cfg, idx)
	if err := m.readMapsSidecar(); err != nil {
		_ = idx.Close()
		return nil, err
	}
	return m, nil
}

// writeMapsSidecar atomically writes the doc<->token bookkeeping (tmp + rename).
// Caller holds m.mu (at least RLock).
func (m *MultiVectorIndex) writeMapsSidecar() error {
	tmp := m.mapsPath + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	if err := m.encodeMaps(w); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := w.Flush(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return renameDurable(tmp, m.mapsPath)
}

func (m *MultiVectorIndex) encodeMaps(w io.Writer) error {
	if _, err := w.Write([]byte(mvMapsMagic)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, m.nextToken); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, uint32(len(m.docTokens))); err != nil { //nolint:gosec
		return err
	}
	for docID, tokens := range m.docTokens {
		if err := binary.Write(w, binary.BigEndian, docID); err != nil {
			return err
		}
		if err := binary.Write(w, binary.BigEndian, uint32(len(tokens))); err != nil { //nolint:gosec
			return err
		}
		for _, tid := range tokens {
			if err := binary.Write(w, binary.BigEndian, tid); err != nil {
				return err
			}
		}
		var metaJSON []byte
		if meta := m.docMeta[docID]; len(meta) > 0 {
			metaJSON, _ = json.Marshal(meta)
		}
		if err := binary.Write(w, binary.BigEndian, uint32(len(metaJSON))); err != nil { //nolint:gosec
			return err
		}
		if _, err := w.Write(metaJSON); err != nil {
			return err
		}
	}

	// Per-key payload TTL block (appended after the doc loop). An OLD reader stops
	// right after the doc loop and never sees this; the NEW decodeMaps probes for a
	// 1-byte present marker (EOF => old blob, keyTTL empty — backward compatible).
	// Marker 0 => no per-key TTL (still byte-cheap); marker 1 => the block follows:
	// [count u32] then per doc [docID u64][entries u32]{[key string][deadline i64]}.
	// Deadlines are ABSOLUTE unix-ms, preserved verbatim so a pending key TTL is
	// time-stable across restore. Only docs with a non-empty deadline map are written.
	withKeyTTL := make([]uint64, 0, len(m.keyTTL))
	for docID, ke := range m.keyTTL {
		if len(ke) > 0 {
			withKeyTTL = append(withKeyTTL, docID)
		}
	}
	if len(withKeyTTL) == 0 {
		// Marker 0: no per-key TTL. (Kept explicit rather than omitting so a future
		// reader can always distinguish "old blob" (EOF) from "new, none".) The
		// version block ALWAYS follows the keyTTL block — never early-return here.
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
	} else {
		if _, err := w.Write([]byte{1}); err != nil {
			return err
		}
		if err := binary.Write(w, binary.BigEndian, uint32(len(withKeyTTL))); err != nil { //nolint:gosec
			return err
		}
		for _, docID := range withKeyTTL {
			ke := m.keyTTL[docID]
			if err := binary.Write(w, binary.BigEndian, docID); err != nil {
				return err
			}
			if err := binary.Write(w, binary.BigEndian, uint32(len(ke))); err != nil { //nolint:gosec
				return err
			}
			for key, dl := range ke {
				if err := writeString(w, key); err != nil {
					return err
				}
				if err := binary.Write(w, binary.BigEndian, dl); err != nil {
					return err
				}
			}
		}
	}

	// Per-doc CAS version block (appended AFTER the keyTTL block). Same present-marker
	// scheme as keyTTL so the trailing layout stays self-delimiting and forward-
	// extensible: marker 0 => no versions; marker 1 => [count u32] then per doc
	// [docID u64][version u64]. An OLD blob (no version block) ends right after the
	// keyTTL block; the new decodeMaps probes for this marker and treats EOF as "old
	// blob" (versions default to 1 for live docs). Only docs with a non-zero version
	// are written; versions are preserved verbatim (NOT re-bumped on restore).
	withVersion := make([]uint64, 0, len(m.version))
	for docID, v := range m.version {
		if v != 0 {
			withVersion = append(withVersion, docID)
		}
	}
	if len(withVersion) == 0 {
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
	} else {
		if _, err := w.Write([]byte{1}); err != nil {
			return err
		}
		if err := binary.Write(w, binary.BigEndian, uint32(len(withVersion))); err != nil { //nolint:gosec
			return err
		}
		for _, docID := range withVersion {
			if err := binary.Write(w, binary.BigEndian, docID); err != nil {
				return err
			}
			if err := binary.Write(w, binary.BigEndian, m.version[docID]); err != nil {
				return err
			}
		}
	}

	// Per-doc OPTIONAL sparse block (appended AFTER the version block). To keep a
	// dense-only MV blob BYTE-IDENTICAL to the pre-sparse format, the block is
	// ENTIRELY OMITTED (no marker byte at all) when no doc carries a sparse vector —
	// the EOF-tolerant decode probe covers the absence, exactly like the MV WAL
	// sparse trailer. When at least one doc has a sparse vector the block is a present
	// marker (1), [count u32], then per doc [docID u64][sparsevec via
	// writeSparseVecFrame]. An OLD blob (predating this feature) ends right after the
	// version block; the new decodeMaps probes for the marker and treats EOF as "no
	// doc sparse". The sparseIdx is NOT serialized — it is rebuilt-on-load from
	// docSparse (mirroring payloadIdx.rebuild).
	withSparse := make([]uint64, 0, len(m.docSparse))
	for docID, sv := range m.docSparse {
		if sv != nil && !sv.IsZero() {
			withSparse = append(withSparse, docID)
		}
	}
	if len(withSparse) == 0 {
		return nil // dense-only: byte-identical, no trailing sparse block
	}
	if _, err := w.Write([]byte{1}); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, uint32(len(withSparse))); err != nil { //nolint:gosec
		return err
	}
	var fb bytes.Buffer
	for _, docID := range withSparse {
		if err := binary.Write(w, binary.BigEndian, docID); err != nil {
			return err
		}
		fb.Reset()
		writeSparseVecFrame(&fb, m.docSparse[docID])
		if _, err := w.Write(fb.Bytes()); err != nil {
			return err
		}
	}
	return nil
}

// readMapsSidecar populates nextToken/docTokens/docMeta from the sidecar and
// rebuilds tokenDoc. A missing sidecar (created-but-never-flushed) leaves the
// index empty, which is correct.
func (m *MultiVectorIndex) readMapsSidecar() error {
	raw, err := os.ReadFile(m.mapsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if len(raw) < len(mvMapsMagic) || string(raw[:len(mvMapsMagic)]) != mvMapsMagic {
		return fmt.Errorf("vector: bad multi-vector maps sidecar magic")
	}
	if err := m.decodeMaps(bytes.NewReader(raw[len(mvMapsMagic):])); err != nil {
		return err
	}
	// mmap-Persistent instant restart: rebuild the filter-first index from the
	// decoded payload (rebuild-on-load; the index is never serialized).
	m.payloadIdx.rebuild(m.docMeta)
	// Rebuild the doc-level sparse inverted index from the decoded docSparse
	// (rebuild-on-load; never serialized). Empty for a dense-only index.
	m.sparseIdx.rebuild(m.docSparse)
	return nil
}

// decodeMaps reads the doc<->token bookkeeping body (after the magic) from r and
// rebuilds tokenDoc.
func (m *MultiVectorIndex) decodeMaps(r io.Reader) error {
	if err := binary.Read(r, binary.BigEndian, &m.nextToken); err != nil {
		return err
	}
	var numDocs uint32
	if err := binary.Read(r, binary.BigEndian, &numDocs); err != nil {
		return err
	}
	for i := uint32(0); i < numDocs; i++ {
		var docID uint64
		if err := binary.Read(r, binary.BigEndian, &docID); err != nil {
			return err
		}
		var nTok uint32
		if err := binary.Read(r, binary.BigEndian, &nTok); err != nil {
			return err
		}
		tokens := make([]uint64, nTok)
		for j := uint32(0); j < nTok; j++ {
			if err := binary.Read(r, binary.BigEndian, &tokens[j]); err != nil {
				return err
			}
			m.tokenDoc[tokens[j]] = docID
		}
		m.docTokens[docID] = tokens
		var mlen uint32
		if err := binary.Read(r, binary.BigEndian, &mlen); err != nil {
			return err
		}
		if mlen > 0 {
			buf := make([]byte, mlen)
			if _, err := io.ReadFull(r, buf); err != nil {
				return err
			}
			meta := make(Metadata)
			if err := json.Unmarshal(buf, &meta); err != nil {
				return err
			}
			m.docMeta[docID] = meta
		}
	}

	// Per-key payload TTL block (optional trailing). Probe for the 1-byte present
	// marker: an OLD blob ends here (io.EOF) and leaves keyTTL empty (no per-key
	// TTL — backward compatible). A bufio reader wraps an os.ReadFile / bytes
	// reader, so a clean EOF means the old format. Any OTHER read error is fail-loud.
	//
	// CRUCIAL: the keyTTL probe must NOT early-return — the version block ALWAYS
	// follows it (when present). An EOF here means the blob predates BOTH features,
	// so default versions for live docs and stop. A marker==0 means "no keyTTL"; we
	// still fall through to the version probe.
	var marker [1]byte
	if _, err := io.ReadFull(r, marker[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			m.defaultVersions() // old blob (no keyTTL + no version block)
			return nil
		}
		return err
	}
	if marker[0] == 1 {
		var numKT uint32
		if err := binary.Read(r, binary.BigEndian, &numKT); err != nil {
			return err
		}
		for i := uint32(0); i < numKT; i++ {
			var docID uint64
			if err := binary.Read(r, binary.BigEndian, &docID); err != nil {
				return err
			}
			var entries uint32
			if err := binary.Read(r, binary.BigEndian, &entries); err != nil {
				return err
			}
			ke := make(map[string]int64, entries)
			for j := uint32(0); j < entries; j++ {
				key, err := readString(r)
				if err != nil {
					return err
				}
				var dl int64
				if err := binary.Read(r, binary.BigEndian, &dl); err != nil {
					return err
				}
				ke[key] = dl
			}
			if len(ke) > 0 {
				m.keyTTL[docID] = ke
			}
		}
	}

	// Per-doc CAS version block (optional trailing, AFTER the keyTTL block). Probe
	// for its own 1-byte present marker. An EOF here means a blob that HAS the keyTTL
	// block but predates the version block (written after the keyTTL feature, before
	// this one) — default versions for live docs. marker 0 => no versions (default
	// for live docs); marker 1 => read [count][docID][version] verbatim.
	if _, err := io.ReadFull(r, marker[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			m.defaultVersions() // old blob: no version block
			return nil
		}
		return err
	}
	if marker[0] == 0 {
		m.defaultVersions() // new blob, no versions stored: default live docs to 1
		return m.decodeDocSparse(r)
	}
	var numV uint32
	if err := binary.Read(r, binary.BigEndian, &numV); err != nil {
		return err
	}
	for i := uint32(0); i < numV; i++ {
		var docID uint64
		if err := binary.Read(r, binary.BigEndian, &docID); err != nil {
			return err
		}
		var v uint64
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return err
		}
		if v != 0 {
			m.version[docID] = v
		}
	}
	// Any live doc the version block omitted (shouldn't happen for a current blob,
	// but be defensive across format edges) defaults to 1 so its CAS version is
	// never 0 while present.
	m.defaultVersions()
	return m.decodeDocSparse(r)
}

// decodeDocSparse reads the OPTIONAL trailing doc-level sparse block (after the
// version block) from r and populates m.docSparse. It probes for the 1-byte present
// marker: a clean EOF (an old blob, or a freshly written dense-only blob that omits
// the block) leaves docSparse empty (backward compatible). marker 0 is also "no doc
// sparse" (defensive; the encoder omits the block rather than writing 0). marker 1
// => [count u32] then per doc [docID u64][sparsevec via readSparseVecFrame]. Any
// other read error or a present-but-truncated frame is fail-loud. The sparseIdx is
// rebuilt-on-load by the caller (restore / readMapsSidecar / loadMultiVector) from
// the populated docSparse — never serialized.
func (m *MultiVectorIndex) decodeDocSparse(r io.Reader) error {
	var marker [1]byte
	if _, err := io.ReadFull(r, marker[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil // old / dense-only blob: no doc sparse block
		}
		return err
	}
	if marker[0] == 0 {
		return nil // no doc sparse (defensive; encoder omits the block)
	}
	var numS uint32
	if err := binary.Read(r, binary.BigEndian, &numS); err != nil {
		return err
	}
	for i := uint32(0); i < numS; i++ {
		var docID uint64
		if err := binary.Read(r, binary.BigEndian, &docID); err != nil {
			return err
		}
		sv, ok := readSparseVecFrame(r)
		if !ok {
			return fmt.Errorf("vector: truncated multi-vector doc-sparse frame")
		}
		if sv != nil && !sv.IsZero() {
			m.docSparse[docID] = sv
		}
	}
	return nil
}

// defaultVersions assigns version 1 to every live doc (one with a token-set
// entry) that has no version yet. Used on the backward-compat paths (old maps
// blobs with no version block) so a restored doc always has a non-zero CAS
// version, and as a defensive backstop for a partial version block. Idempotent.
func (m *MultiVectorIndex) defaultVersions() {
	for docID := range m.docTokens {
		if m.version[docID] == 0 {
			m.version[docID] = 1
		}
	}
}
