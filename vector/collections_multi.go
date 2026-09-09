// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Multi-vector (late-interaction) collection management on CollectionStore.
// These indexes live in a registry parallel to the single-vector collections
// and share the store's name canonicalization.
//
// A Persistent multi-vector collection is mmap-backed (its float32 token vectors
// live off-heap in <col>.mv.vecs; the level-0 graph in <col>.mv.graph) and
// durable across restart via an instant-restart sidecar (<col>.mv.meta) plus a
// doc<->token maps sidecar (<col>.mv.maps); a tiny <col>.mvcfg marker records its
// config so the store reloads it on open. Non-persistent collections are
// in-memory only.

// mvPaths returns the on-disk paths the store manages for a multi-vector
// collection. Caller must already canonicalize.
func (s *CollectionStore) mvPaths(canonical string) (cfgPath, vecsPath, graphPath string) {
	tenant, col, _ := splitTenant(canonical)
	base := filepath.Join(s.dir, "vectors", tenant)
	return filepath.Join(base, col+".mvcfg"), filepath.Join(base, col+".mv.vecs"), filepath.Join(base, col+".mv.graph")
}

// mvWALPaths returns the on-disk paths the store manages for a single-node
// WAL-mode (heap-checkpoint) multi-vector collection: the heap snapshot checkpoint
// (<col>.mvwsnap) and the write-ahead log (<col>.mvwal). These are DISTINCT from
// the mmap Persistent sidecars (.mv.vecs/.mv.graph/.mv.meta/.mv.maps) — WAL mode is
// heap-backed, mutually exclusive with Persistent. The <col>.mvcfg marker (shared
// with the Persistent path, carrying the WAL flag) tells the store which to load.
// Caller must already canonicalize.
func (s *CollectionStore) mvWALPaths(canonical string) (snapPath, walPath string) {
	tenant, col, _ := splitTenant(canonical)
	base := filepath.Join(s.dir, "vectors", tenant)
	return filepath.Join(base, col+".mvwsnap"), filepath.Join(base, col+".mvwal")
}

// withMVPaths fills the store-managed mmap paths for a Persistent config.
func (s *CollectionStore) withMVPaths(canonical string, cfg MultiVectorConfig) MultiVectorConfig {
	if !cfg.Persistent {
		return cfg
	}
	_, vecs, graph := s.mvPaths(canonical)
	cfg.MmapPath = vecs
	cfg.GraphMmapPath = graph
	return cfg
}

// mvClusterPaths returns the generation-suffixed multi-vector mmap paths for the
// persistent-cluster policy (mirrors clusterMmapPaths for single-vector).
func (s *CollectionStore) mvClusterPaths(canonical string, gen uint64) (vecs, graph string) {
	tenant, col, _ := splitTenant(canonical)
	base := filepath.Join(s.dir, "vectors", tenant)
	suffix := fmt.Sprintf(".mv.g%d", gen)
	return filepath.Join(base, col+suffix+".vecs"), filepath.Join(base, col+suffix+".graph")
}

// effectiveClusterMVConfig backs a multi-vector index with generation-suffixed
// mmap files for the persistent-cluster policy. The inner index only mmaps token
// vectors when quantized (Persistent requires Quant != QuantNone — mmap needs
// codes), so an unquantized multi-vector collection stays heap-resident; Raft is
// the durability authority either way (no sidecar is ever flushed in cluster).
func (s *CollectionStore) effectiveClusterMVConfig(canonical string, gen uint64, cfg MultiVectorConfig) MultiVectorConfig {
	vecs, graph := s.mvClusterPaths(canonical, gen)
	cfg.MmapPath = vecs
	cfg.GraphMmapPath = graph
	cfg.Persistent = cfg.Quant != QuantNone
	// Raft is the durability + determinism authority: disable the wall-clock per-key
	// TTL sweeper so its physical removal never diverges committed state across
	// replicas (mirror dense effectiveClusterConfig; #4 B3a analog).
	cfg.SuppressSweep = true
	return cfg
}

// AcquireMulti returns the named multi-vector index with a reference held, so a
// concurrent cluster RestoreAll / DropMultiVector won't unmap it mid-operation.
// Pair with exactly one (*MultiVectorIndex).Release.
func (s *CollectionStore) AcquireMulti(name string) (*MultiVectorIndex, bool) {
	canonical, err := canonicalName(name)
	if err != nil {
		return nil, false
	}
	s.mu.RLock()
	idx, ok := s.multi[canonical]
	if ok {
		idx.inuse.Add(1)
	}
	s.mu.RUnlock()
	return idx, ok
}

// removeClusterMVFiles deletes a cluster-persistent multi-vector index's
// generation files (from retire, after Close).
func removeClusterMVFiles(cfg MultiVectorConfig) {
	if cfg.MmapPath == "" {
		return
	}
	for _, p := range []string{cfg.MmapPath, cfg.GraphMmapPath, mvMetaPath(cfg), mvMapsPath(cfg)} {
		_ = os.Remove(p)
	}
}

// CreateMultiVector registers a new late-interaction index under name. A
// Persistent config is backed by store-managed mmap files and a config marker is
// written so it reloads on restart. Returns an error if the name is taken or the
// config is invalid.
func (s *CollectionStore) CreateMultiVector(name string, cfg MultiVectorConfig) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.multi[canonical]; ok {
		return fmt.Errorf("vector: multi-vector collection %q already exists", canonical)
	}
	if _, ok := s.collections[canonical]; ok {
		return ErrCollectionExists
	}
	if _, ok := s.named[canonical]; ok {
		return ErrCollectionExists
	}
	cfgPath, _, _ := s.mvPaths(canonical)
	// Single-node WAL: heap-checkpoint durability (mutually exclusive with the mmap
	// Persistent mode, enforced by NewMultiVectorIndex). FORCED OFF on the cluster
	// path — Raft/SnapshotAll is the durability authority there (mirror dense
	// effectiveClusterConfig / CreateNamedConfig). Cleared before the cluster eff
	// rewrite below (which may set Persistent), so WAL && Persistent never arises.
	wal := cfg.WAL && !s.persistentCluster
	// Cluster-persistent: back the index with generation-suffixed mmap files
	// (off-heap when quantized) regardless of the user's Persistent flag, and
	// always write the .mvcfg marker so a restart reloads it (Raft repopulates).
	eff := s.withMVPaths(canonical, cfg)
	eff.WAL = wal
	writeMarker := cfg.Persistent || wal
	if s.persistentCluster {
		eff = s.effectiveClusterMVConfig(canonical, s.gen.Load(), cfg)
		eff.WAL = false // cluster forces WAL off (Raft is the authority)
		writeMarker = true
	}
	if eff.Persistent || wal || s.persistentCluster {
		if err := os.MkdirAll(filepath.Dir(cfgPath), 0o750); err != nil {
			return err
		}
	}
	idx, err := NewMultiVectorIndex(eff)
	if err != nil {
		return err
	}
	if writeMarker {
		if err := writeMVConfig(cfgPath, cfg); err != nil {
			_ = idx.Close()
			// The marker is the load-time source of truth (loadMultiVector rebuilds the
			// index from it), and the durable publish can fail with the rename already
			// landed — its parent-directory fsync runs after it. Creation is reporting
			// failure and nothing is registered, so a surviving marker would resurrect
			// an empty index on restart. Best-effort remove it.
			_ = os.Remove(cfgPath)
			return err
		}
	}
	if wal {
		_, walPath := s.mvWALPaths(canonical)
		w, werr := openWAL(walPath, cfg.WALNoSync)
		if werr != nil {
			_ = idx.Close()
			return werr
		}
		idx.wal = w
	}
	s.multi[canonical] = idx
	return nil
}

// loadMultiVector reads a multi-vector config marker and registers the index,
// instant-restarting from its sidecar when one exists (else a fresh mmap index).
func (s *CollectionStore) loadMultiVector(canonical, cfgPath string) error {
	cfg, err := readMVConfig(cfgPath)
	if err != nil {
		return fmt.Errorf("vector: load multi-vector %s: %w", canonical, err)
	}

	// Cluster-persistent: data files were wiped at open; build an empty
	// mmap-backed index (current generation) and let Raft repopulate it. No
	// instant-restart sidecar — Raft is the durability authority.
	if s.persistentCluster {
		eff := s.effectiveClusterMVConfig(canonical, s.gen.Load(), cfg)
		idx, nerr := NewMultiVectorIndex(eff)
		if nerr != nil {
			return fmt.Errorf("vector: cluster open multi-vector %s: %w", canonical, nerr)
		}
		s.multi[canonical] = idx
		return nil
	}

	// Single-node WAL (heap-checkpoint) mode: build an empty heap index, restore the
	// heap snapshot checkpoint (if any), replay the WAL tail on top, then reopen the
	// WAL for appends — the MV analogue of loadNamed / dense loadCollection's WAL
	// branch. DISTINCT from the mmap Persistent restart below (no mmap sidecar).
	if cfg.WAL {
		idx, nerr := NewMultiVectorIndex(cfg) // heap-backed (no mmap paths); WAL set below
		if nerr != nil {
			return fmt.Errorf("vector: open multi-vector %s: %w", canonical, nerr)
		}
		snapPath, walPath := s.mvWALPaths(canonical)
		if f, oerr := os.Open(snapPath); oerr == nil { //nolint:gosec // store-managed path
			rerr := idx.restore(f)
			_ = f.Close()
			if rerr != nil {
				_ = idx.Close()
				return fmt.Errorf("vector: restore multi-vector %s: %w", canonical, rerr)
			}
		}
		if rerr := replayMVWAL(walPath, idx); rerr != nil {
			_ = idx.Close()
			return fmt.Errorf("vector: wal replay multi-vector %s: %w", canonical, rerr)
		}
		// Rebuild the filter-first index from the fully-replayed payload so a
		// restored+replayed index has a correct index regardless of replay order
		// (mirrors loadNamed's post-replay rebuild).
		idx.rebuildPayloadIdx()
		// Likewise rebuild the doc-level sparse inverted index from the fully-replayed
		// docSparse (rebuild-on-load; never serialized).
		idx.rebuildSparseIdx()
		w, werr := openWAL(walPath, cfg.WALNoSync)
		if werr != nil {
			_ = idx.Close()
			return fmt.Errorf("vector: wal open multi-vector %s: %w", canonical, werr)
		}
		idx.wal = w
		s.multi[canonical] = idx
		return nil
	}

	eff := s.withMVPaths(canonical, cfg)
	var idx *MultiVectorIndex
	if _, statErr := os.Stat(mvMetaPath(eff)); statErr == nil {
		idx, err = openPersistentMultiVector(eff)
	} else {
		idx, err = NewMultiVectorIndex(eff)
	}
	if err != nil {
		return fmt.Errorf("vector: open multi-vector %s: %w", canonical, err)
	}
	s.multi[canonical] = idx
	return nil
}

// GetMultiVector returns the named late-interaction index.
func (s *CollectionStore) GetMultiVector(name string) (*MultiVectorIndex, bool) {
	canonical, err := canonicalName(name)
	if err != nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	idx, ok := s.multi[canonical]
	return idx, ok
}

// FlushMultiVector persists a Persistent multi-vector collection (no-op error
// for an in-memory one).
func (s *CollectionStore) FlushMultiVector(name string) error {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.Flush()
}

// FlushMVWAL checkpoints a WAL-mode (heap-checkpoint) multi-vector collection:
// under the index's opMu (so it never races an in-flight apply-then-log mutator) it
// writes a consistent heap snapshot to <col>.mvwsnap atomically, then truncates the
// WAL — the checkpoint now subsumes the log (mirror dense Flush / FlushNamed). This
// is the WAL-mode analogue of FlushMultiVector (which targets the mmap Persistent
// sidecar); the two are distinct durability modes. A no-op for a non-WAL index.
//
// Serializing outside opMu was evaluated and rejected for the same reason as dense
// Flush, and MV is the worst case of the three: MultiVectorIndex.snapshot holds
// m.mu.RLock while serializing the ENTIRE inner HNSW into a buffer
// (vector/multivector_persist.go:45-49), and every mutator needs m.mu.Lock, so the
// stall would only move to another lock. See the note on CollectionStore.Flush.
func (s *CollectionStore) FlushMVWAL(name string) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	if idx.wal == nil {
		return nil // not WAL-mode: nothing to checkpoint
	}
	snapPath, _ := s.mvWALPaths(canonical)
	idx.opMu.Lock()
	defer idx.opMu.Unlock()
	if err := s.writeMVWALSnapshotFile(idx, snapPath); err != nil {
		return err
	}
	return idx.wal.truncate()
}

// writeMVWALSnapshotFile atomically writes idx's heap snapshot (inner hnsw blob +
// doc<->token maps + per-key TTL) to path (tmp + fsync + rename), mirroring
// writeNamedSnapshotFile so a crash mid-write leaves the prior good checkpoint
// intact. It uses the MV snapshot/restore HEAP codec (not the mmap SavePersist
// sidecar) — the destination that makes WAL mode distinct from Persistent.
func (s *CollectionStore) writeMVWALSnapshotFile(idx *MultiVectorIndex, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // store-managed path
	if err != nil {
		return err
	}
	if err := idx.snapshot(f); err != nil {
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
	// rename + parent-dir fsync: the MV Flush truncates the WAL on the
	// strength of this checkpoint (see writeSnapshotFile in collections.go).
	return renameDurable(tmp, path)
}

// DropMultiVector removes the named index, frees it, and deletes its files.
func (s *CollectionStore) DropMultiVector(name string) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	s.mu.Lock()
	idx, ok := s.multi[canonical]
	if ok {
		delete(s.multi, canonical)
	}
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w %q", ErrNoMultiVector, canonical)
	}
	cfgPath, vecs, graph := s.mvPaths(canonical)
	wsnap, wwal := s.mvWALPaths(canonical)
	eff := s.withMVPaths(canonical, MultiVectorConfig{Persistent: true, MmapPath: vecs})
	idxCfg := idx.cfg
	// Drain in-flight users before closing (no unmap under a reader), then delete
	// the single-node files (mmap Persistent + WAL heap-checkpoint) and the cluster
	// generation files (no-ops if absent — a heap-only collection has none).
	idx.retire(func() {
		for _, p := range []string{cfgPath, vecs, graph, mvMetaPath(eff), mvMapsPath(eff), wsnap, wwal} {
			_ = os.Remove(p)
		}
		removeClusterMVFiles(idxCfg)
	})
	return nil
}

// MultiAdd inserts or replaces a document's token vectors in the named index.
func (s *CollectionStore) MultiAdd(name string, docID uint64, tokens [][]float32, meta Metadata) error {
	_, err := s.MultiAddCAS(name, docID, tokens, meta, CASCond{})
	return err
}

// MultiAddCAS is MultiAdd with an optimistic-CAS precondition (CASCond{} = no
// precondition). Returns the resulting per-document version; ErrVersionConflict
// (no mutation) on a mismatch. See MultiVectorIndex.AddCAS.
func (s *CollectionStore) MultiAddCAS(name string, docID uint64, tokens [][]float32, meta Metadata, cas CASCond) (uint64, error) {
	return s.MultiAddCASKeyTTL(name, docID, tokens, meta, nil, cas)
}

// MultiAddCASKeyTTL is MultiAddCAS carrying an OPTIONAL per-key payload TTL map
// (key -> RELATIVE ms) set on the fresh document. Empty/nil = no per-key TTL (the
// zero-overhead path). See MultiVectorIndex.AddCASKeyTTL.
func (s *CollectionStore) MultiAddCASKeyTTL(name string, docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return s.MultiAddCASKeyTTLSparse(name, docID, tokens, meta, keyTTLMs, nil, cas)
}

// MultiAddCASKeyTTLSparse is MultiAddCASKeyTTL carrying an OPTIONAL doc-level sparse
// vector. Nil/zero sparse is byte/behaviour-identical to MultiAddCASKeyTTL. See
// MultiVectorIndex.AddCASKeyTTLSparse.
func (s *CollectionStore) MultiAddCASKeyTTLSparse(name string, docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, sparse *SparseVector, cas CASCond) (uint64, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.AddCASKeyTTLSparse(docID, tokens, meta, keyTTLMs, sparse, cas)
}

// MultiAddCASKeyTTLSparseAt is MultiAddCASKeyTTLSparse computing the per-key payload
// deadline against the EXPLICIT leader apply stamp nowMs — the replicated-apply
// variant the MV add handler takes under a stamp (#4 vector TTL determinism).
func (s *CollectionStore) MultiAddCASKeyTTLSparseAt(name string, docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, sparse *SparseVector, cas CASCond, nowMs int64) (uint64, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.AddCASKeyTTLSparseAt(docID, tokens, meta, keyTTLMs, sparse, cas, nowMs)
}

// MVGetSparse returns docID's optional doc-level sparse vector (deep-copied), or
// (nil, false) when absent / dense-only. See MultiVectorIndex.GetSparse.
func (s *CollectionStore) MVGetSparse(name string, docID uint64) (*SparseVector, bool, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, false, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	sv, has := idx.GetSparse(docID)
	return sv, has, nil
}

// MVGet retrieves a live document by id: its token matrix (deep-copied) + payload.
// ok is false for an absent document (the MV index has no tombstones/TTL). A pure
// read. See MultiVectorIndex.Get.
func (s *CollectionStore) MVGet(name string, docID uint64) (tokens [][]float32, payload Metadata, ok bool, err error) {
	tokens, payload, _, ok, err = s.MVGetVersion(name, docID)
	return tokens, payload, ok, err
}

// MVGetVersion is MVGet that also returns the document's per-document CAS version
// (0 for an absent document). See MultiVectorIndex.Get.
func (s *CollectionStore) MVGetVersion(name string, docID uint64) (tokens [][]float32, payload Metadata, version uint64, ok bool, err error) {
	idx, iok := s.AcquireMulti(name)
	if !iok {
		return nil, nil, 0, false, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	tokens, payload, version, ok = idx.Get(docID)
	return tokens, payload, version, ok, nil
}

// MVSetPayload merges patch into docID's payload (no reindex, no WAL). keyTTLMs sets
// per-key relative TTLs (key -> ms; ttl<=0 clears a key's deadline). applied=false
// for an absent document (not an error). See MultiVectorIndex.SetPayload.
func (s *CollectionStore) MVSetPayload(name string, docID uint64, patch Metadata, keyTTLMs map[string]int64) (applied bool, err error) {
	applied, _, err = s.MVSetPayloadCAS(name, docID, patch, keyTTLMs, CASCond{})
	return applied, err
}

// MVSetPayloadCAS is MVSetPayload with an optimistic-CAS precondition. Returns
// (applied, version, err); a CAS mismatch surfaces ErrVersionConflict. See
// MultiVectorIndex.SetPayloadCAS.
func (s *CollectionStore) MVSetPayloadCAS(name string, docID uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.SetPayloadCAS(docID, patch, keyTTLMs, cas))
}

// MVSetPayloadCASAt is MVSetPayloadCAS computing the per-key deadline against the
// leader apply stamp nowMs (#4 vector TTL determinism).
func (s *CollectionStore) MVSetPayloadCASAt(name string, docID uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.SetPayloadCASAt(docID, patch, keyTTLMs, cas, nowMs))
}

// MVMutatePayloadRecordCAS applies fn to the operate record under key in the
// multi-vector collection's document docID. See
// MultiVectorIndex.MutatePayloadRecordCAS. An absent document is the not-found
// FLAG (applied=false, err=nil), like every other payload dispatcher here.
func (s *CollectionStore) MVMutatePayloadRecordCAS(name string, docID uint64, key string, fn RecordMutator, cas CASCond) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.MutatePayloadRecordCAS(docID, key, fn, cas))
}

// MVMutatePayloadRecordCASAt is MVMutatePayloadRecordCAS under the leader apply
// stamp nowMs. See MultiVectorIndex.MutatePayloadRecordCASAt.
func (s *CollectionStore) MVMutatePayloadRecordCASAt(name string, docID uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.MutatePayloadRecordCASAt(docID, key, fn, cas, nowMs))
}

// MVOverwritePayload replaces docID's entire payload with meta. keyTTLMs sets the
// per-key relative TTLs on the new payload. applied=false for an absent document
// (not an error). See MultiVectorIndex.OverwritePayload.
func (s *CollectionStore) MVOverwritePayload(name string, docID uint64, meta Metadata, keyTTLMs map[string]int64) (applied bool, err error) {
	applied, _, err = s.MVOverwritePayloadCAS(name, docID, meta, keyTTLMs, CASCond{})
	return applied, err
}

// MVOverwritePayloadCAS is MVOverwritePayload with an optimistic-CAS precondition.
// See MultiVectorIndex.OverwritePayloadCAS.
func (s *CollectionStore) MVOverwritePayloadCAS(name string, docID uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.OverwritePayloadCAS(docID, meta, keyTTLMs, cas))
}

// MVOverwritePayloadCASAt is MVOverwritePayloadCAS computing the per-key deadline
// against the leader apply stamp nowMs (#4 vector TTL determinism).
func (s *CollectionStore) MVOverwritePayloadCASAt(name string, docID uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.OverwritePayloadCASAt(docID, meta, keyTTLMs, cas, nowMs))
}

// MVDeletePayloadKeys removes the listed keys from docID's payload. applied=false
// for an absent document (not an error). See MultiVectorIndex.DeletePayloadKeys.
func (s *CollectionStore) MVDeletePayloadKeys(name string, docID uint64, keys []string) (applied bool, err error) {
	applied, _, err = s.MVDeletePayloadKeysCAS(name, docID, keys, CASCond{})
	return applied, err
}

// MVDeletePayloadKeysCAS is MVDeletePayloadKeys with an optimistic-CAS precondition.
// See MultiVectorIndex.DeletePayloadKeysCAS.
func (s *CollectionStore) MVDeletePayloadKeysCAS(name string, docID uint64, keys []string, cas CASCond) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.DeletePayloadKeysCAS(docID, keys, cas))
}

// MVClearPayload removes all of docID's payload. applied=false for an absent
// document (not an error). See MultiVectorIndex.ClearPayload.
func (s *CollectionStore) MVClearPayload(name string, docID uint64) (applied bool, err error) {
	applied, _, err = s.MVClearPayloadCAS(name, docID, CASCond{})
	return applied, err
}

// MVClearPayloadCAS is MVClearPayload with an optimistic-CAS precondition. See
// MultiVectorIndex.ClearPayloadCAS.
func (s *CollectionStore) MVClearPayloadCAS(name string, docID uint64, cas CASCond) (applied bool, version uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return payloadAppliedV(idx.ClearPayloadCAS(docID, cas))
}

// MultiAddIfAbsent adds a document to the named index ONLY if it is not already
// present, reporting whether it inserted (no-op returning false when present).
// The atomic online-copy primitive for the MV path (Race A); mirrors MultiAdd.
func (s *CollectionStore) MultiAddIfAbsent(name string, docID uint64, tokens [][]float32, meta Metadata) (bool, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.AddIfAbsent(docID, tokens, meta)
}

// MultiAddIfAbsentVersion is MultiAddIfAbsent that, on a REAL add, sets the
// document's version VERBATIM to `version` (version==0 ⇒ the plain if-absent,
// fresh add → version 1) and the doc's ABSOLUTE per-key payload deadlines
// (keyExpires) VERBATIM. The version-preserving online MV reshard copy primitive
// (Race A); the MV analog of CollectionStore.InsertIfAbsentVersion.
func (s *CollectionStore) MultiAddIfAbsentVersion(name string, docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64) (bool, error) {
	return s.MultiAddIfAbsentVersionSparse(name, docID, tokens, meta, keyExpires, version, nil)
}

// MultiAddIfAbsentVersionSparse is MultiAddIfAbsentVersion carrying the doc's
// OPTIONAL sparse vector VERBATIM (online reshard copy). Nil/zero is byte-identical.
// See MultiVectorIndex.MultiAddIfAbsentVersionSparse.
func (s *CollectionStore) MultiAddIfAbsentVersionSparse(name string, docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) (bool, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.MultiAddIfAbsentVersionSparse(docID, tokens, meta, keyExpires, version, sparse)
}

// MultiRestoreAdd is a verbatim-version replace-add (version==0 ⇒ the normal bump).
// keyExpires carries the doc's ABSOLUTE per-key payload deadlines (set verbatim).
// The version-preserving OFFLINE MV resplit backfill primitive; the MV analog of
// the dense EncodeVectorInsertArgsVersioned reinsert. See MultiVectorIndex.MultiRestoreAdd.
func (s *CollectionStore) MultiRestoreAdd(name string, docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64) error {
	return s.MultiRestoreAddSparse(name, docID, tokens, meta, keyExpires, version, nil)
}

// MultiRestoreAddSparse is MultiRestoreAdd carrying the doc's OPTIONAL sparse vector
// VERBATIM (offline resplit backfill). Nil/zero is byte-identical. See
// MultiVectorIndex.MultiRestoreAddSparse.
func (s *CollectionStore) MultiRestoreAddSparse(name string, docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) error {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.MultiRestoreAddSparse(docID, tokens, meta, keyExpires, version, sparse)
}

// MultiBulkBuild bulk-loads recs into the named MV index in one concurrent pass
// when that index is EMPTY, returning whether it built. Used by the offline MV
// resplit copy: the new-generation partitions are fresh (empty), so this replaces
// one inner-graph insert per token with a single multi-core BuildConcurrent.
// Returns false (no mutation) when the index is non-empty, so the caller falls
// back to per-record MultiRestoreAddSparse. Mirrors the dense bulk_build path.
func (s *CollectionStore) MultiBulkBuild(name string, recs []MultiScanRecord, workers int) (bool, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.MultiBulkBuild(recs, workers)
}

// MultiExists reports whether docID is currently live in the named index — the
// O(1) liveness probe the MV resurrection guard uses (Race B). Mirrors MultiDelete.
func (s *CollectionStore) MultiExists(name string, docID uint64) (bool, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.Exists(docID), nil
}

// MultiSearch runs a late-interaction (MaxSim) search against the named index.
func (s *CollectionStore) MultiSearch(name string, query [][]float32, k int, opts MultiSearchOpts) ([]MultiResult, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.Search(query, k, opts)
}

// MultiHybrid fuses the named MV index's MaxSim lane and its doc-level sparse lane
// into the top-k (cross-modality hybrid). Mirrors MultiSearch's acquire/release.
// See MultiVectorIndex.MVHybrid.
func (s *CollectionStore) MultiHybrid(name string, query [][]float32, sparseQ *SparseVector, k int, opts HybridOpts) ([]Result, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	return idx.MVHybrid(query, sparseQ, k, opts)
}

// MultiQuery executes a unified Query API spec (MaxSim + doc-sparse FUSION or
// RERANK) against the multi-vector index: it acquires the MultiVectorIndex
// (holding a ref so a concurrent DropMultiVector can't close it mid-query) and
// runs (*MultiVectorIndex).Query. No leaf may carry a Space (an MV collection has
// no named spaces — fail loud). The store-level entry point the vector_mv_query op
// handler calls (mirroring NamedQuery / MultiHybrid). Returns a mode-tagged
// QueryResult.
func (s *CollectionStore) MultiQuery(name string, spec QuerySpec) (QueryResult, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return QueryResult{}, fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	return idx.Query(spec)
}

// MultiQueryTreeLanes is the UNFUSED tree-lanes variant of MultiQuery (the per-
// partition emit for a spec containing a nested MULTI-lane FUSION node): it acquires
// the MultiVectorIndex (ref held) and runs (*MultiVectorIndex).QueryTreeLanes,
// returning the node-expanded pre-order lanes the coordinator folds over the global
// union (P>1==P1). The whole nested recursion runs under the index's single
// lock+clock snapshot (see QueryTreeLanes). The store-level entry point the
// vector_mv_query op handler calls when vector.SpecHasNestedFusion(spec).
func (s *CollectionStore) MultiQueryTreeLanes(name string, spec QuerySpec) ([][]Result, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	return idx.QueryTreeLanes(spec)
}

// MultiHybridLanes builds the MV hybrid's UNFUSED MaxSim + sparse lanes (the
// partition-exact fan-out leaf). Mirrors MultiSearch's acquire/release. See
// MultiVectorIndex.MVHybridLanes.
func (s *CollectionStore) MultiHybridLanes(name string, query [][]float32, sparseQ *SparseVector, k int, opts HybridOpts) (dense, sparse []Result, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, nil, fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	return idx.MVHybridLanes(query, sparseQ, k, opts)
}

// MVScrollPage is the cursor-aware multi-vector scroll: up to limit live
// documents (id + payload) matching filter, id-ASCENDING, strictly after afterID
// when hasAfter. Returns docs, nextAfter (largest id returned), and hasMore.
// Mirrors NamedScrollPage's acquire/release. See MultiVectorIndex.ScrollDocsPage.
func (s *CollectionStore) MVScrollPage(name string, filter Filter, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, 0, false, fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	return idx.ScrollDocsPage(filter, afterID, hasAfter, limit)
}

// MVScrollPageOrder is the order_by-aware MVScrollPage: up to limit live documents
// matching filter ordered by the order_by field's (value, id) total order, resuming
// strictly after (afterKey, afterID). Missing-field points are EXCLUDED. order == nil
// falls back to the id-ascending MVScrollPage path. Mirrors Collection.ScrollDocsPageOrder.
func (s *CollectionStore) MVScrollPageOrder(name string, filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, 0, false, fmt.Errorf("%w %q", ErrNoMultiVector, name)
	}
	defer idx.Release()
	return idx.ScrollDocsPageOrder(filter, order, afterID, afterKey, hasAfter, limit)
}

// MultiScanDocuments enumerates every live document of the named index as a
// self-contained MultiScanRecord (the read primitive an offline MV resplit
// uses). Mirrors MultiSearch's acquire/release.
func (s *CollectionStore) MultiScanDocuments(name string) ([]MultiScanRecord, error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return nil, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.ScanDocuments(), nil
}

// MultiDelete removes a document from the named index, returning whether it
// existed.
func (s *CollectionStore) MultiDelete(name string, docID uint64) (bool, error) {
	removed, _, err := s.MultiDeleteCAS(name, docID, CASCond{})
	return removed, err
}

// MultiDeleteCAS is MultiDelete with an optimistic-CAS precondition (CASCond{} =
// no precondition). On a mismatch returns ErrVersionConflict and removed=false.
// See MultiVectorIndex.DeleteCAS.
func (s *CollectionStore) MultiDeleteCAS(name string, docID uint64, cas CASCond) (removed bool, prevVersion uint64, err error) {
	idx, ok := s.AcquireMulti(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no multi-vector collection %q", name)
	}
	defer idx.Release()
	return idx.DeleteCAS(docID, cas)
}

// mvConfigMarker is the relocatable subset of MultiVectorConfig persisted to
// <col>.mvcfg; mmap paths are re-derived on open so the store stays movable.
type mvConfigMarker struct {
	Dim            int       `json:"dim"`
	M              int       `json:"m"`
	EfConstruction int       `json:"ef_construction"`
	EfSearch       int       `json:"ef_search"`
	Seed           int64     `json:"seed"`
	Quant          QuantMode `json:"quant"`
	RescoreFactor  int       `json:"rescore_factor"`
	Persistent     bool      `json:"persistent"`
	Partitions     int       `json:"partitions,omitempty"`
	WAL            bool      `json:"wal,omitempty"`
	WALNoSync      bool      `json:"wal_no_sync,omitempty"`
	// IndexType + IVF knobs must persist so a reopen reconstructs the inner index
	// as IVF (not HNSW). All omitempty ⇒ an HNSW MV collection's marker JSON is
	// byte-identical to the pre-IVF marker (no keys written), preserving
	// back-compat with existing on-disk markers (they decode to IndexHNSW/zero).
	IndexType         IndexType `json:"index_type,omitempty"`
	IVFNlist          int       `json:"ivf_nlist,omitempty"`
	IVFNprobe         int       `json:"ivf_nprobe,omitempty"`
	IVFPQ             bool      `json:"ivf_pq,omitempty"`
	IVFPQM            int       `json:"ivf_pq_m,omitempty"`
	IVFRerank         bool      `json:"ivf_rerank,omitempty"`
	OPQ               bool      `json:"opq,omitempty"`
	IVFTrainThreshold int       `json:"ivf_train_threshold,omitempty"`
}

func writeMVConfig(path string, cfg MultiVectorConfig) error {
	b, err := json.Marshal(mvConfigMarker{
		Dim: cfg.Dim, M: cfg.M, EfConstruction: cfg.EfConstruction, EfSearch: cfg.EfSearch,
		Seed: cfg.Seed, Quant: cfg.Quant, RescoreFactor: cfg.RescoreFactor, Persistent: cfg.Persistent,
		Partitions: cfg.Partitions, WAL: cfg.WAL, WALNoSync: cfg.WALNoSync,
		IndexType: cfg.IndexType, IVFNlist: cfg.IVFNlist, IVFNprobe: cfg.IVFNprobe,
		IVFPQ: cfg.IVFPQ, IVFPQM: cfg.IVFPQM, IVFRerank: cfg.IVFRerank, OPQ: cfg.OPQ,
		IVFTrainThreshold: cfg.IVFTrainThreshold,
	})
	if err != nil {
		return err
	}
	return atomicWriteFile(path, b)
}

func readMVConfig(path string) (MultiVectorConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return MultiVectorConfig{}, err
	}
	var m mvConfigMarker
	if err := json.Unmarshal(b, &m); err != nil {
		return MultiVectorConfig{}, err
	}
	return MultiVectorConfig{
		Dim: m.Dim, M: m.M, EfConstruction: m.EfConstruction, EfSearch: m.EfSearch,
		Seed: m.Seed, Quant: m.Quant, RescoreFactor: m.RescoreFactor, Persistent: m.Persistent,
		Partitions: m.Partitions, WAL: m.WAL, WALNoSync: m.WALNoSync,
		IndexType: m.IndexType, IVFNlist: m.IVFNlist, IVFNprobe: m.IVFNprobe,
		IVFPQ: m.IVFPQ, IVFPQM: m.IVFPQM, IVFRerank: m.IVFRerank, OPQ: m.OPQ,
		IVFTrainThreshold: m.IVFTrainThreshold,
	}, nil
}
