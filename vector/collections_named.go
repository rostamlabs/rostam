// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Named-vector (NamedCollection) management on CollectionStore. These
// collections live in a registry parallel to the single-vector collections and
// the multi-vector indexes, sharing the store's name canonicalization. A name
// belongs to exactly one family — dense XOR multi-vector XOR named — so create
// rejects a name already taken by ANY family.
//
// v1 named collections are in-memory only (single-node). Snapshot/restore
// durability, codecs, transports and fan-out are separate concerns.

// ErrNoNamed is the sentinel wrapped by named-vector operations that target an
// absent collection. Callers use errors.Is to distinguish "absent" from real
// failures (mirrors ErrNoMultiVector).
var ErrNoNamed = errors.New("vector: no named-vector collection")

// AcquireNamed returns the named-vector collection with a reference held, so a
// concurrent DropNamed won't close its sub-indexes mid-operation. Pair with
// exactly one (*NamedCollection).Release. Mirrors AcquireMulti.
func (s *CollectionStore) AcquireNamed(name string) (*NamedCollection, bool) {
	canonical, err := canonicalName(name)
	if err != nil {
		return nil, false
	}
	s.mu.RLock()
	nc, ok := s.named[canonical]
	if ok {
		nc.inuse.Add(1)
	}
	s.mu.RUnlock()
	return nc, ok
}

// namedPaths returns the on-disk paths the store manages for a named-vector
// collection: a tiny config marker (<col>.ncfg, mirroring MV's .mvcfg), the
// snapshot checkpoint (<col>.nsnap), and the write-ahead log (<col>.nwal). The
// marker records the spaces + WAL flags so the store reloads the collection on
// open. Caller must already canonicalize.
func (s *CollectionStore) namedPaths(canonical string) (cfgPath, snapPath, walPath string) {
	tenant, col, _ := splitTenant(canonical)
	base := filepath.Join(s.dir, "vectors", tenant)
	return filepath.Join(base, col+".ncfg"), filepath.Join(base, col+".nsnap"), filepath.Join(base, col+".nwal")
}

// CreateNamed registers a new HEAP-ONLY named-vector collection under name (no
// single-node disk durability — the historical in-memory behavior the wire/op
// create path uses, since the named create op carries no WAL flag in v1). For a
// WAL-mode single-node collection use the cfg-only path: CreateCollection with a
// vector.Config that sets NamedVectors + WAL (it threads the carrier to
// CreateNamedConfig). Returns ErrCollectionExists if the name is taken by ANY
// family (dense, multi-vector, or named).
func (s *CollectionStore) CreateNamed(name string, cfg map[string]NamedVectorParams) error {
	return s.CreateNamedConfig(name, NamedConfig{Spaces: cfg})
}

// CreateNamedConfig registers a new named-vector collection under name from the
// carrier (spaces + single-node WAL flags). The config must define at least one
// space, each with a non-empty reserved-char-free name and a valid per-space
// index config (Dim > 0). Returns ErrCollectionExists if the name is taken by ANY
// family.
//
// Durability:
//   - Cluster (persistentCluster): WAL is FORCED OFF — durability is Raft /
//     SnapshotAll, exactly like dense effectiveClusterConfig. Heap-only, no marker.
//   - Single-node WAL: opens a <col>.nwal, writes a <col>.ncfg marker so a restart
//     reloads it, and stores the wal on the NamedCollection (apply-then-log).
//   - Single-node heap-only: in-memory, no files (historical behavior).
func (s *CollectionStore) CreateNamedConfig(name string, cfg NamedConfig) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	// Validate before taking the lock so a bad config never partially registers.
	if verr := validateNamedVectors(cfg.Spaces); verr != nil {
		return verr
	}
	// Cluster mode: Raft/SnapshotAll is the durability authority — never enable a
	// single-node WAL (mirror dense effectiveClusterConfig forcing WAL=false). DO
	// NOT touch the cluster snapshot/Raft path.
	wal := cfg.WAL && !s.persistentCluster

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.collections[canonical]; ok {
		return ErrCollectionExists
	}
	if _, ok := s.multi[canonical]; ok {
		return ErrCollectionExists
	}
	if _, ok := s.named[canonical]; ok {
		return ErrCollectionExists
	}
	nc, err := NewNamedCollection(canonical, cfg.Spaces)
	if err != nil {
		return err
	}
	// Cluster/replicated: disable the wall-clock per-key sweeper (Raft is the
	// determinism authority; expired keys are filtered lazily at read time) — the
	// named analog of the dense/MV SuppressSweep policy (#4 B3a).
	nc.suppressSweep = s.persistentCluster
	if wal {
		cfgPath, _, walPath := s.namedPaths(canonical)
		if err := os.MkdirAll(filepath.Dir(cfgPath), 0o750); err != nil {
			_ = nc.Close()
			return err
		}
		if err := writeNamedConfig(cfgPath, cfg); err != nil {
			_ = nc.Close()
			// The marker is the load-time source of truth (loadNamed rebuilds the
			// collection from it), and the durable publish can fail with the rename
			// already landed — its parent-directory fsync runs after it. Creation is
			// reporting failure and nothing is registered, so a surviving marker would
			// resurrect an empty collection on restart. Best-effort remove it.
			_ = os.Remove(cfgPath)
			return err
		}
		w, werr := openWAL(walPath, cfg.WALNoSync)
		if werr != nil {
			_ = nc.Close()
			return werr
		}
		nc.wal = w
	}
	s.named[canonical] = nc
	return nil
}

// loadNamed reads a named config marker and registers the collection, restoring
// its snapshot checkpoint + replaying the WAL tail before reopening the WAL for
// appends (the single-node named lifecycle, mirroring dense loadCollection's WAL
// branch). Called from OpenCollectionStorePersistent's tenant pass for each
// <col>.ncfg. Cluster mode leaves the named family heap-only (no marker written).
func (s *CollectionStore) loadNamed(canonical, cfgPath string) error {
	cfg, err := readNamedConfig(cfgPath)
	if err != nil {
		return fmt.Errorf("vector: load named %s: %w", canonical, err)
	}
	nc, err := NewNamedCollection(canonical, cfg.Spaces)
	if err != nil {
		return fmt.Errorf("vector: rebuild named %s: %w", canonical, err)
	}
	nc.suppressSweep = s.persistentCluster // cluster: sweeper off (#4 B3a; see CreateNamedConfig)
	_, snapPath, walPath := s.namedPaths(canonical)
	// Restore the consistent checkpoint (if any), then replay the WAL tail on top.
	if f, oerr := os.Open(snapPath); oerr == nil { //nolint:gosec // store-managed path
		rerr := nc.Restore(f)
		_ = f.Close()
		if rerr != nil {
			_ = nc.Close()
			return fmt.Errorf("vector: restore named %s: %w", canonical, rerr)
		}
	}
	if rerr := replayNamedWAL(walPath, nc); rerr != nil {
		_ = nc.Close()
		return fmt.Errorf("vector: wal replay named %s: %w", canonical, rerr)
	}
	// Rebuild the filter-first index from the fully-replayed payload so a
	// restored+replayed collection has a correct index regardless of replay order.
	nc.rebuildPayloadIdx()
	w, werr := openWAL(walPath, cfg.WALNoSync)
	if werr != nil {
		_ = nc.Close()
		return fmt.Errorf("vector: wal open named %s: %w", canonical, werr)
	}
	nc.wal = w
	s.named[canonical] = nc
	return nil
}

// FlushNamed checkpoints a WAL-mode named collection: under the collection's opMu
// (so it never races an in-flight apply-then-log mutator) it writes a consistent
// Snapshot to <col>.nsnap atomically, then truncates the WAL — the checkpoint now
// subsumes the log (mirror dense Flush). A no-op for a heap-only named collection.
//
// Serializing outside opMu was evaluated and rejected for the same reason as dense
// Flush: NamedCollection.Snapshot holds nc.mu.RLock across the whole serialization
// (vector/named.go:1991) and every mutator needs nc.mu.Lock, so the stall would
// only move to another lock. See the note on CollectionStore.Flush.
func (s *CollectionStore) FlushNamed(name string) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	if nc.wal == nil {
		return nil // heap-only: nothing to checkpoint
	}
	_, snapPath, _ := s.namedPaths(canonical)
	nc.opMu.Lock()
	defer nc.opMu.Unlock()
	if err := s.writeNamedSnapshotFile(nc, snapPath); err != nil {
		return err
	}
	return nc.wal.truncate()
}

// writeNamedSnapshotFile atomically writes nc's snapshot to path (tmp + fsync +
// rename), mirroring writeSnapshotFile so a crash mid-write leaves the prior good
// checkpoint intact.
func (s *CollectionStore) writeNamedSnapshotFile(nc *NamedCollection, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // store-managed path
	if err != nil {
		return err
	}
	if err := nc.Snapshot(f); err != nil {
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
	// rename + parent-dir fsync: the named Flush truncates the WAL on the
	// strength of this checkpoint (see writeSnapshotFile in collections.go).
	return renameDurable(tmp, path)
}

// writeNamedConfig writes the named config marker (spaces + WAL flags) atomically
// (tmp + rename), mirroring writeMVConfig. The marker is the relocatable record
// the store reloads on open.
func writeNamedConfig(path string, cfg NamedConfig) error {
	b, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	return atomicWriteFile(path, b)
}

// readNamedConfig reads a named config marker.
func readNamedConfig(path string) (NamedConfig, error) {
	b, err := os.ReadFile(path) //nolint:gosec // store-managed path
	if err != nil {
		return NamedConfig{}, err
	}
	var cfg NamedConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		return NamedConfig{}, err
	}
	return cfg, nil
}

// GetNamed returns the named-vector collection (canonicalized for tenant lookup).
func (s *CollectionStore) GetNamed(name string) (*NamedCollection, bool) {
	canonical, err := canonicalName(name)
	if err != nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	nc, ok := s.named[canonical]
	return nc, ok
}

// DropNamed removes the named-vector collection and frees its sub-indexes. Drop
// of an unknown name returns ErrNoNamed (callers use errors.Is for idempotence).
func (s *CollectionStore) DropNamed(name string) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	s.mu.Lock()
	nc, ok := s.named[canonical]
	if ok {
		delete(s.named, canonical)
	}
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w %q", ErrNoNamed, canonical)
	}
	cfgPath, snapPath, walPath := s.namedPaths(canonical)
	// Drain in-flight users, close the wal + sub-indexes, then delete the
	// single-node files (no-ops if absent — a heap-only collection has none).
	nc.retire(func() {
		_ = os.Remove(cfgPath)
		_ = os.Remove(snapPath)
		_ = os.Remove(walPath)
	})
	return nil
}

// NamedInsert upserts point id (a map of named vectors + shared payload + ttl)
// into the named collection. See NamedCollection.Insert.
func (s *CollectionStore) NamedInsert(name string, id uint64, vectors map[string][]float32, payload Metadata, ttl time.Duration) error {
	_, err := s.NamedInsertCAS(name, id, vectors, payload, ttl, CASCond{})
	return err
}

// NamedInsertCAS is NamedInsert with an optimistic-CAS precondition (CASCond{} =
// no precondition). Returns the resulting per-point version; ErrVersionConflict
// (no mutation) on a mismatch. See NamedCollection.InsertCAS.
func (s *CollectionStore) NamedInsertCAS(name string, id uint64, vectors map[string][]float32, payload Metadata, ttl time.Duration, cas CASCond) (uint64, error) {
	return s.NamedInsertCASKeyTTL(name, id, vectors, payload, ttl, nil, cas)
}

// NamedInsertCASKeyTTL is NamedInsertCAS carrying an OPTIONAL per-key payload TTL
// map (key -> RELATIVE ms) set on the fresh point. Empty/nil = no per-key TTL (the
// zero-overhead path). See NamedCollection.InsertCASKeyTTL.
func (s *CollectionStore) NamedInsertCASKeyTTL(name string, id uint64, vectors map[string][]float32, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return s.NamedInsertSparseCASKeyTTL(name, id, vectors, nil, payload, ttl, keyTTLMs, cas)
}

// NamedInsertSparseCASKeyTTL is NamedInsertCASKeyTTL carrying per-space SPARSE
// values (sparseVectors[space] is the *SparseVector for a sparse space) alongside
// the dense vectors. A dense-only call passes nil sparseVectors, byte-identical to
// the pre-sparse path. A dense value for a sparse space (or vice versa) fails loud
// with ErrSpaceModalityMismatch. See NamedCollection.InsertCASKeyTTL.
func (s *CollectionStore) NamedInsertSparseCASKeyTTL(name string, id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.InsertCASKeyTTL(id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas)
}

// NamedInsertSparseCASKeyTTLAt is NamedInsertSparseCASKeyTTL stamping every TTL
// deadline against the leader apply stamp nowMs — the replicated-apply variant the
// named insert handler takes under a stamp (#4 vector TTL determinism).
func (s *CollectionStore) NamedInsertSparseCASKeyTTLAt(name string, id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.InsertCASKeyTTLAt(id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas, nowMs)
}

// NamedGet retrieves a live point by id: its per-space vectors (deep-copied) +
// shared payload + remaining TTL. ok is false for an absent/expired point. A pure
// read. See NamedCollection.Get.
func (s *CollectionStore) NamedGet(name string, id uint64) (vectors map[string][]float32, payload Metadata, ttl time.Duration, ok bool, err error) {
	vectors, payload, ttl, _, ok, err = s.NamedGetVersion(name, id)
	return vectors, payload, ttl, ok, err
}

// NamedGetVersion is NamedGet that also returns the point's per-point CAS version
// (0 for an absent/expired point). See NamedCollection.Get.
func (s *CollectionStore) NamedGetVersion(name string, id uint64) (vectors map[string][]float32, payload Metadata, ttl time.Duration, version uint64, ok bool, err error) {
	nc, nok := s.AcquireNamed(name)
	if !nok {
		return nil, nil, 0, 0, false, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	vectors, payload, ttl, version, ok = nc.Get(id)
	return vectors, payload, ttl, version, ok, nil
}

// NamedSetPayload merges patch into id's shared payload (no reindex, no WAL).
// keyTTLMs sets per-key relative TTLs (key -> ms; ttl<=0 clears a key's deadline).
// applied=false for an absent/expired point (not an error). See
// NamedCollection.SetPayload.
func (s *CollectionStore) NamedSetPayload(name string, id uint64, patch Metadata, keyTTLMs map[string]int64) (applied bool, err error) {
	applied, _, err = s.NamedSetPayloadCAS(name, id, patch, keyTTLMs, CASCond{})
	return applied, err
}

// NamedSetPayloadCAS is NamedSetPayload with an optimistic-CAS precondition.
// Returns (applied, version, err): an absent point is applied=false (a FLAG); a
// CAS mismatch surfaces ErrVersionConflict. See NamedCollection.SetPayloadCAS.
func (s *CollectionStore) NamedSetPayloadCAS(name string, id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.SetPayloadCAS(id, patch, keyTTLMs, cas))
}

// NamedSetPayloadCASAt is NamedSetPayloadCAS stamping the per-key deadline + liveness
// gate against the leader apply stamp nowMs (#4 vector TTL determinism).
func (s *CollectionStore) NamedSetPayloadCASAt(name string, id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.SetPayloadCASAt(id, patch, keyTTLMs, cas, nowMs))
}

// NamedMutatePayloadRecordCAS applies fn to the operate record under key in the
// named collection's point id. See NamedCollection.MutatePayloadRecordCAS. An
// absent point is the not-found FLAG (applied=false, err=nil), like every other
// payload dispatcher here.
func (s *CollectionStore) NamedMutatePayloadRecordCAS(name string, id uint64, key string, fn RecordMutator, cas CASCond) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.MutatePayloadRecordCAS(id, key, fn, cas))
}

// NamedMutatePayloadRecordCASAt is NamedMutatePayloadRecordCAS under the leader
// apply stamp nowMs. See NamedCollection.MutatePayloadRecordCASAt.
func (s *CollectionStore) NamedMutatePayloadRecordCASAt(name string, id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.MutatePayloadRecordCASAt(id, key, fn, cas, nowMs))
}

// NamedOverwritePayload replaces id's entire shared payload with meta. keyTTLMs sets
// the per-key relative TTLs on the new payload. applied=false for an absent/expired
// point (not an error). See NamedCollection.OverwritePayload.
func (s *CollectionStore) NamedOverwritePayload(name string, id uint64, meta Metadata, keyTTLMs map[string]int64) (applied bool, err error) {
	applied, _, err = s.NamedOverwritePayloadCAS(name, id, meta, keyTTLMs, CASCond{})
	return applied, err
}

// NamedOverwritePayloadCAS is NamedOverwritePayload with an optimistic-CAS
// precondition. See NamedCollection.OverwritePayloadCAS.
func (s *CollectionStore) NamedOverwritePayloadCAS(name string, id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.OverwritePayloadCAS(id, meta, keyTTLMs, cas))
}

// NamedOverwritePayloadCASAt is NamedOverwritePayloadCAS stamping the per-key
// deadline + liveness gate against the leader apply stamp nowMs (#4 vector TTL
// determinism).
func (s *CollectionStore) NamedOverwritePayloadCASAt(name string, id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.OverwritePayloadCASAt(id, meta, keyTTLMs, cas, nowMs))
}

// NamedDeletePayloadKeys removes the listed keys from id's shared payload.
// applied=false for an absent/expired point (not an error). See
// NamedCollection.DeletePayloadKeys.
func (s *CollectionStore) NamedDeletePayloadKeys(name string, id uint64, keys []string) (applied bool, err error) {
	applied, _, err = s.NamedDeletePayloadKeysCAS(name, id, keys, CASCond{})
	return applied, err
}

// NamedDeletePayloadKeysCAS is NamedDeletePayloadKeys with an optimistic-CAS
// precondition. See NamedCollection.DeletePayloadKeysCAS.
func (s *CollectionStore) NamedDeletePayloadKeysCAS(name string, id uint64, keys []string, cas CASCond) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.DeletePayloadKeysCAS(id, keys, cas))
}

// NamedDeletePayloadKeysCASAt is NamedDeletePayloadKeysCAS judging the liveness gate
// against the leader apply stamp nowMs (#4 vector TTL determinism).
func (s *CollectionStore) NamedDeletePayloadKeysCASAt(name string, id uint64, keys []string, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.DeletePayloadKeysCASAt(id, keys, cas, nowMs))
}

// NamedClearPayload removes all of id's shared payload. applied=false for an
// absent/expired point (not an error). See NamedCollection.ClearPayload.
func (s *CollectionStore) NamedClearPayload(name string, id uint64) (applied bool, err error) {
	applied, _, err = s.NamedClearPayloadCAS(name, id, CASCond{})
	return applied, err
}

// NamedClearPayloadCAS is NamedClearPayload with an optimistic-CAS precondition.
// See NamedCollection.ClearPayloadCAS.
func (s *CollectionStore) NamedClearPayloadCAS(name string, id uint64, cas CASCond) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.ClearPayloadCAS(id, cas))
}

// NamedClearPayloadCASAt is NamedClearPayloadCAS judging the liveness gate against
// the leader apply stamp nowMs (#4 vector TTL determinism).
func (s *CollectionStore) NamedClearPayloadCASAt(name string, id uint64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return payloadAppliedV(nc.ClearPayloadCASAt(id, cas, nowMs))
}

// NamedSearch runs a filtered KNN search against the named space of the named
// collection. See NamedCollection.SearchNamed.
func (s *CollectionStore) NamedSearch(name, vectorName string, query []float32, k int, filter Filter) ([]Result, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.SearchNamed(vectorName, query, k, filter)
}

// NamedSearchDocs is NamedSearch returning Documents enriched with the shared
// per-point payload. See NamedCollection.SearchNamedDocs.
func (s *CollectionStore) NamedSearchDocs(name, vectorName string, query []float32, k int, filter Filter) ([]Document, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.SearchNamedDocs(vectorName, query, k, filter)
}

// NamedSearchSparse runs a filtered sparse-dot-product top-k search against the
// SPARSE named space of the named collection. See NamedCollection.SearchNamedSparse.
func (s *CollectionStore) NamedSearchSparse(name, space string, query *SparseVector, k int, filter Filter) ([]Result, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.SearchNamedSparse(space, query, k, filter)
}

// NamedSearchSparseDocs is NamedSearchSparse returning Documents enriched with the
// shared per-point payload. See NamedCollection.SearchNamedSparseDocs.
func (s *CollectionStore) NamedSearchSparseDocs(name, space string, query *SparseVector, k int, filter Filter) ([]Document, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.SearchNamedSparseDocs(space, query, k, filter)
}

// NamedHybrid fuses a DENSE named space and a SPARSE named space into the top-k
// for the named collection (cross-space hybrid). See NamedCollection.NamedHybrid.
func (s *CollectionStore) NamedHybrid(name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ *SparseVector, k int, opts HybridOpts) ([]Result, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.NamedHybrid(denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
}

// NamedHybridLanes returns the two UNFUSED candidate lanes (dense ascending by
// Distance, sparse descending by Score) for a named cross-space hybrid, so a
// cross-partition coordinator can union and fuse once. See
// NamedCollection.NamedHybridLanes.
func (s *CollectionStore) NamedHybridLanes(name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ *SparseVector, k int, opts HybridOpts) (dense, sparse []Result, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.NamedHybridLanes(denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
}

// NamedQuery executes a unified Query API spec (multi-space N-lane FUSION or
// RERANK) against the named collection: it acquires the NamedCollection (holding
// a ref so a concurrent DropNamed can't close its sub-indexes mid-query) and runs
// (*NamedCollection).Query. Every leaf must target a configured named space (fail
// loud otherwise). The store-level entry point the vector_named_query op handler
// calls (mirroring NamedHybrid). Returns a mode-tagged QueryResult.
func (s *CollectionStore) NamedQuery(name string, spec QuerySpec) (QueryResult, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return QueryResult{}, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.Query(spec)
}

// NamedQueryTreeLanes is the UNFUSED tree-lanes variant of NamedQuery (the per-
// partition emit for a spec containing a nested MULTI-lane FUSION node): it acquires
// the NamedCollection (ref held) and runs (*NamedCollection).QueryTreeLanes, returning
// the node-expanded pre-order lanes the coordinator folds over the global union
// (P>1==P1). The store-level entry point the vector_named_query op handler calls when
// vector.SpecHasNestedFusion(spec) (mirroring the dense QueryTreeLanes path).
func (s *CollectionStore) NamedQueryTreeLanes(name string, spec QuerySpec) ([][]Result, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.QueryTreeLanes(spec)
}

// NamedDelete removes a point from every named space + the shared payload,
// returning whether it existed. See NamedCollection.Delete.
func (s *CollectionStore) NamedDelete(name string, id uint64) (bool, error) {
	removed, _, err := s.NamedDeleteCAS(name, id, CASCond{})
	return removed, err
}

// NamedDeleteCAS is NamedDelete with an optimistic-CAS precondition (CASCond{} =
// no precondition). On a mismatch returns ErrVersionConflict and removed=false.
// See NamedCollection.DeleteCAS.
func (s *CollectionStore) NamedDeleteCAS(name string, id uint64, cas CASCond) (removed bool, prevVersion uint64, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return false, 0, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.DeleteCAS(id, cas)
}

// NamedScroll lists live points (+ shared payload) of the named collection
// matching filter, up to limit. See NamedCollection.ScrollDocs.
func (s *CollectionStore) NamedScroll(name string, filter Filter, limit int) ([]Document, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.ScrollDocs(filter, limit)
}

// NamedScrollPage is the cursor-aware NamedScroll: up to limit live points (with
// shared payload) matching filter, id-ASCENDING, strictly after afterID when
// hasAfter. Returns docs, nextAfter (largest id returned), and hasMore. See
// NamedCollection.ScrollDocsPage.
func (s *CollectionStore) NamedScrollPage(name string, filter Filter, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, 0, false, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.ScrollDocsPage(filter, afterID, hasAfter, limit)
}

// NamedScrollPageOrder is the order_by-aware NamedScrollPage: up to limit live points
// matching filter ordered by the order_by field's (value, id) total order, resuming
// strictly after (afterKey, afterID). Missing-field points are EXCLUDED. order == nil
// falls back to the id-ascending NamedScrollPage path. Mirrors Collection.ScrollDocsPageOrder.
func (s *CollectionStore) NamedScrollPageOrder(name string, filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, 0, false, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.ScrollDocsPageOrder(filter, order, afterID, afterKey, hasAfter, limit)
}

// NamedConfig returns the configured named spaces of the named collection (the
// introspection accessor). See NamedCollection.Config.
func (s *CollectionStore) NamedConfig(name string) (map[string]NamedVectorParams, error) {
	nc, ok := s.AcquireNamed(name)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrNoNamed, name)
	}
	defer nc.Release()
	return nc.Config(), nil
}
