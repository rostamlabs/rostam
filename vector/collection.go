// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"
)

// Collection is a named vector index. It wraps a VectorIndex implementation
// with a name and exposes the same VectorIndex API so callers can hold one
// handle per logical collection. CollectionStore manages the registry of these.
// The concrete index (HNSW today; IVF-Flat in future) is chosen at construction
// time and addressed only through the VectorIndex interface thereafter.
type Collection struct {
	name string
	cfg  Config
	idx  VectorIndex

	// Write-ahead log (nil unless Config.WAL). opMu serializes {apply + WAL
	// append} against {checkpoint + WAL rotate} so a Flush never truncates an op
	// it didn't capture. Only taken when wal != nil (zero overhead otherwise).
	wal  *wal
	opMu sync.Mutex

	// sweeper lifecycle
	sweepStop  chan struct{}
	sweepDone  chan struct{}
	sweepStart sync.Once

	// inuse counts in-flight operations holding this collection (taken by
	// CollectionStore.Acquire, dropped by Release). retire() drains it to zero
	// before unmapping, so a concurrent RestoreAll/Drop never munmaps mmap-backed
	// storage under an in-flight reader (a fatal SIGBUS; benign-but-guarded for
	// heap collections).
	inuse atomic.Int64

	// lastAccess is the injected-clock unix-nanos timestamp of the most recent
	// resolve (CollectionStore.Acquire) or promote, driving the idle cold-tier
	// sweeper (CollectionStore.SweepCold). 0 = never stamped (a fresh/never-resolved
	// collection), which the sweeper treats as first-sight. Stored lock-free from
	// the read-locked Acquire hot path and read by the write-locked sweeper, so it
	// must be atomic — the per-collection atomic keeps the shared last-access map
	// off the hot path entirely.
	lastAccess atomic.Int64

	// Bulk-load staging: StageBulk accumulates (id, vec) pairs cheaply (concurrent
	// appends under stageMu); BuildStaged then builds the whole set into the empty
	// index in one concurrent pass (hnsw.BuildConcurrent — multi-core). This is
	// the fast initial-load path: serialized single inserts can't parallelize the
	// HNSW build, but a staged bulk build can. Only valid on an empty index.
	stageMu   sync.Mutex
	stageIDs  []uint64
	stageVecs [][]float32
	// stageMetas is the OPTIONAL payload column of the staging buffer, and it is
	// LAZY: nil for as long as every staged batch has been vectors-only, so the
	// vector-only load pays nothing (not even one nil map per point — at 1M points
	// that column alone is 8 MB). The first payload-bearing batch backfills nils
	// for everything already staged and from then on the column is kept exactly
	// len(stageIDs) long. Keeping it dense once it exists is what lets a mixed
	// load — some batches with payloads, some without — stay aligned; a sparse or
	// per-batch representation would put point i's payload on point j.
	stageMetas []Metadata
}

// NewCollection constructs a Collection backed by an in-memory index. The
// concrete implementation is selected by newIndex (HNSW today; a future
// IndexType switch can pick IVF-Flat). Returns the same config-validation
// errors as the chosen implementation.
func NewCollection(name string, cfg Config) (*Collection, error) {
	cfg = applyHNSWDefaults(cfg)
	idx, err := newIndex(cfg)
	if err != nil {
		return nil, err
	}
	return &Collection{name: name, cfg: cfg, idx: idx}, nil
}

// applyHNSWDefaults fills the standard graph parameters when a caller leaves
// them zero, so an HNSW collection built from just Dim and Metric works —
// matching the "default 16" the Config.M comment has always promised, and the
// defaults the HTTP and Python layers already apply before the engine sees the
// config. Previously only those outer layers defaulted and the engine rejected
// zero, so the same create succeeded over the wire and failed from the Go
// library. Vamana resolves its own geometry in applyVamanaDefaults; IVF requires
// its parameters and validates them, so both are left untouched — only the
// zero-value (HNSW) IndexType is filled here. Non-positive is treated as unset,
// matching the HTTP layer, so a negative value still becomes the default rather
// than a surprise ErrInvalidM.
func applyHNSWDefaults(cfg Config) Config {
	if cfg.IndexType != IndexHNSW {
		return cfg
	}
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	if cfg.EfSearch <= 0 {
		cfg.EfSearch = 64
	}
	return cfg
}

// newIndex builds the in-memory VectorIndex implementation for cfg, dispatching
// on cfg.IndexType. IndexHNSW (the zero value) builds the graph index; IndexIVF
// builds an IVF-Flat index. This is the single construction dispatch point — both
// returned types satisfy VectorIndex.
func newIndex(cfg Config) (VectorIndex, error) {
	switch cfg.IndexType {
	case IndexIVF:
		return newIVF(cfg)
	case IndexVamana:
		return newVamana(cfg)
	case IndexGPU:
		return newGPUIndex(cfg)
	default:
		return newHNSW(cfg)
	}
}

// openIndex reopens the VectorIndex implementation for cfg, dispatching on
// cfg.IndexType. IndexHNSW reopens its persistent (mmap-backed) sidecar via
// openPersist. A Persistent IndexIVF reopens its instant-restart mmap sidecar via
// openPersistIVF (re-maps the vecs file, reads the .meta sidecar — no full float
// re-read); a non-persistent IndexIVF is snapshot-only, so openIVF returns a fresh
// empty index the caller then Restores. Mirrors newIndex as the single reopen
// dispatch point.
func openIndex(cfg Config, metaPath string) (VectorIndex, error) {
	switch cfg.IndexType {
	case IndexIVF:
		if cfg.Persistent {
			return openPersistIVF(cfg, metaPath)
		}
		return openIVF(cfg, metaPath)
	case IndexVamana:
		// A Persistent Vamana instant-restarts via the single-layer hnsw sidecar
		// (SavePersist / openPersist): the level-0 slab (stride VamanaR) lives in the
		// GraphMmapPath, the float vectors in the MmapPath, and the ids + medoid entry
		// point in the .meta sidecar. newPersistShell re-pins the single-layer geometry
		// (mL=0, m0=VamanaR, vamana=true, pruneAlpha) from cfg. A non-persistent Vamana
		// is snapshot-only: openVamana returns a fresh empty index the caller Restores.
		if cfg.Persistent {
			return openPersist(cfg, metaPath)
		}
		return openVamana(cfg, metaPath)
	case IndexGPU:
		return openGPUIndex(cfg, metaPath)
	default:
		return openPersist(cfg, metaPath)
	}
}

// openPersistentCollection reopens a persistent (mmap-backed) collection by
// mapping its files — instant restart, no graph rebuild — instead of the
// snapshot replay NewCollection+Restore does. cfg must carry the store-managed
// QuantStorage/MmapPath/GraphMmapPath (see CollectionStore.effectiveConfig).
func openPersistentCollection(name string, cfg Config, metaPath string) (*Collection, error) {
	idx, err := openIndex(cfg, metaPath)
	if err != nil {
		return nil, err
	}
	return &Collection{name: name, cfg: cfg, idx: idx}, nil
}

// Name returns the collection name.
func (c *Collection) Name() string { return c.name }

// Config returns the collection's configuration (immutable after construction).
func (c *Collection) Config() Config { return c.cfg }

// Release drops a reference taken by CollectionStore.Acquire. Pair every Acquire
// with exactly one Release (typically via defer).
func (c *Collection) Release() { c.inuse.Add(-1) }

// retire waits for in-flight users (Acquire holders) to drain, then stops the
// sweeper, closes the index (unmapping any mmap files), and runs cleanup (e.g.
// deleting this generation's mmap files). It must be called only AFTER the
// collection has been removed from the store map, so no new Acquire can observe
// it; existing holders finish first. The brief spin is fine — retire is a
// structural-change path (RestoreAll / Drop), not the hot path.
func (c *Collection) retire(cleanup func()) {
	for c.inuse.Load() > 0 {
		time.Sleep(200 * time.Microsecond)
	}
	c.Stop()
	_ = c.Close()
	if cleanup != nil {
		cleanup()
	}
}

// Insert delegates to the underlying index with the given TTL (0 = no expiry),
// metadata (nil = none), and sparse vector (nil = none). It starts the
// background sweeper on the first call.
func (c *Collection) Insert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) error {
	_, err := c.InsertCAS(id, vec, ttl, meta, sparse, CASCond{})
	return err
}

// InsertKeyTTL is Insert carrying an OPTIONAL per-key payload TTL map (key ->
// RELATIVE ms); the engine computes the absolute deadline now+ttl at insert and
// the WAL logs the resulting absolute map (so replay restores it verbatim).
// Empty/nil keyTTLMs behaves exactly like Insert (zero-overhead).
func (c *Collection) InsertKeyTTL(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64) error {
	_, err := c.InsertCASKeyTTL(id, vec, ttl, meta, sparse, keyTTLMs, CASCond{})
	return err
}

// InsertCAS is Insert with an optimistic-CAS precondition (CASCond{} = no
// precondition, an unconditional write that still bumps the version). It returns
// the resulting version on success; on a CAS mismatch it returns
// ErrVersionConflict with no mutation (and nothing WAL-logged). The check+bump and
// the WAL append are serialized under opMu so engine + WAL agree.
func (c *Collection) InsertCAS(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, cas CASCond) (uint64, error) {
	return c.InsertCASKeyTTL(id, vec, ttl, meta, sparse, nil, cas)
}

// InsertCASKeyTTL is InsertCAS carrying an OPTIONAL per-key payload TTL map (key
// -> RELATIVE ms). The engine computes the resulting ABSOLUTE deadline map and
// returns it, so the WAL logs the absolute deadlines (replay restores them
// verbatim, time-stable). Empty/nil keyTTLMs is the zero-overhead path
// (byte-identical WAL record).
func (c *Collection) InsertCASKeyTTL(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	c.startSweeper()
	if c.wal == nil {
		version, _, err := c.idx.Insert(id, vec, ttl, meta, sparse, keyTTLMs, cas)
		return version, err
	}
	// Apply, then log on success (an op is durable iff its fsync'd WAL record
	// exists; a crash before the append loses an un-acked op — consistent). The
	// resulting version AND the resulting absolute per-key deadlines are logged so
	// replay restores them verbatim.
	//
	// opMu is held across {apply + WAL WRITE} only (preserving apply order and the
	// log's byte order), scoped to a closure so a panic inside idx.Insert or the
	// WAL append still unlocks opMu via the deferred Unlock instead of leaking it
	// forever — server/handlers.go recovers per-request panics (crafted frames can
	// panic in arg decoders / index code), so an unrecovered lock would silently
	// deadlock every later write on this collection. The closure returns before
	// the durability wait so concurrent writers overlap in commitWaitStaged and
	// the leader-fsync actually batches them (the group-commit machinery in wal.go
	// otherwise degenerates to one fsync per op).
	version, seq, err := func() (uint64, uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		version, keyExpires, err := c.idx.Insert(id, vec, ttl, meta, sparse, keyTTLMs, cas)
		if err != nil {
			return 0, 0, err
		}
		seq, err := c.wal.appendInsertStaged(id, vec, ttl, meta, sparse, keyExpires, version)
		return version, seq, err
	}()
	if err != nil {
		return 0, err
	}
	if err := c.wal.commitWaitStaged(seq); err != nil {
		return 0, err
	}
	return version, nil
}

// InsertCASKeyTTLAt is InsertCASKeyTTL whose point/per-key TTL deadlines and CAS
// liveness are stamped against the EXPLICIT leader clock nowMs (unix millis)
// instead of the wall clock — the replicated-apply variant the vector op handler
// uses under an apply stamp so every replica computes byte-identical absolute
// deadlines (#4 vector TTL determinism, mirroring cache.PutAt via TxContext.Put).
// The WAL (single-node only; off under cluster replication) still logs the
// resulting ABSOLUTE deadlines verbatim.
func (c *Collection) InsertCASKeyTTLAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	c.startSweeper()
	if c.wal == nil {
		version, _, err := c.idx.InsertAt(id, vec, ttl, meta, sparse, keyTTLMs, cas, nowMs)
		return version, err
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
	// InsertCASKeyTTL) so a panic inside idx.InsertAt or the WAL append still
	// unlocks opMu via the deferred Unlock instead of leaking it forever, then
	// released BEFORE the durability wait so concurrent writers overlap in
	// commitWaitStaged and the leader-fsync actually batches them. ops/builtin.go
	// routes every replicated-apply insert through this path (tx.applyStamped); it
	// only reaches the WAL branch below on a collection actually configured with
	// one (c.wal != nil) — a no-op skip otherwise, unchanged from before.
	version, seq, err := func() (uint64, uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		version, keyExpires, err := c.idx.InsertAt(id, vec, ttl, meta, sparse, keyTTLMs, cas, nowMs)
		if err != nil {
			return 0, 0, err
		}
		seq, err := c.wal.appendInsertStaged(id, vec, ttl, meta, sparse, keyExpires, version)
		return version, seq, err
	}()
	if err != nil {
		return 0, err
	}
	if err := c.wal.commitWaitStaged(seq); err != nil {
		return 0, err
	}
	return version, nil
}

// RestoreInsert inserts id with an EXACT per-point version set verbatim (NOT
// bumped) — the version-preserving primitive used by WAL replay and the
// reshard/resplit backfill so a copied point keeps its version. keyExpires is the
// ABSOLUTE per-key payload deadline map restored VERBATIM (NOT recomputed now+ttl)
// — replay logs and restores it time-stable; the reshard copy pass passes nil
// today (a follow-up could carry the copied point's key deadlines). WAL-logs the
// insert with that version + keyExpires so a later replay restores them too. Starts
// the sweeper like Insert.
func (c *Collection) RestoreInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error {
	// Checked HERE as well as in the engine body, because THIS layer owns the
	// {apply, then append} ordering below: idx.RestoreInsert mutates the index
	// before the WAL append runs, so a payload the codec cannot encode has to be
	// refused before either step, not between them.
	if err := checkRecordValues(meta); err != nil {
		return err
	}
	c.startSweeper()
	if c.wal == nil {
		return c.idx.RestoreInsert(id, vec, ttl, meta, sparse, keyExpires, version)
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure so a panic
	// still unlocks it (see InsertCASKeyTTL), then released BEFORE the durability
	// wait. A restore/reshard backfill is a HIGH-VOLUME writer: holding opMu across
	// the fsync would make it the permanent fsync leader and serialize every
	// concurrent live writer behind it, exactly the group-commit collapse the
	// staged pattern exists to avoid.
	seq, err := func() (uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		if err := c.idx.RestoreInsert(id, vec, ttl, meta, sparse, keyExpires, version); err != nil {
			return 0, err
		}
		return c.wal.appendInsertStaged(id, vec, ttl, meta, sparse, keyExpires, version)
	}()
	if err != nil {
		return err
	}
	return c.wal.commitWaitStaged(seq)
}

// RestoreInsertAt is RestoreInsert stamping the POINT ttl deadline against the
// EXPLICIT leader clock nowMs (keyExpires installed VERBATIM) — the replicated
// version-preserving insert path (reshard/resplit backfill) under an apply stamp
// (#4 vector TTL determinism).
func (c *Collection) RestoreInsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) error {
	// See RestoreInsert: this layer applies before it appends.
	if err := checkRecordValues(meta); err != nil {
		return err
	}
	c.startSweeper()
	if c.wal == nil {
		return c.idx.RestoreInsertAt(id, vec, ttl, meta, sparse, keyExpires, version, nowMs)
	}
	// Staged commit (see RestoreInsert): opMu across {apply + WAL WRITE} only.
	seq, err := func() (uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		if err := c.idx.RestoreInsertAt(id, vec, ttl, meta, sparse, keyExpires, version, nowMs); err != nil {
			return 0, err
		}
		return c.wal.appendInsertStaged(id, vec, ttl, meta, sparse, keyExpires, version)
	}()
	if err != nil {
		return err
	}
	return c.wal.commitWaitStaged(seq)
}

// InsertIfAbsent delegates to the underlying index's atomic insert-if-absent
// (no-op returning inserted=false when id is already live). On a real insert
// with a WAL configured the op is logged as a plain insert (replay is idempotent
// and if-absent collapses to the same stored state). Starts the sweeper like Insert.
func (c *Collection) InsertIfAbsent(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) (bool, error) {
	return c.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, nil, 0)
}

// InsertIfAbsentVersion is InsertIfAbsent that, on a real insert, sets the
// point's version VERBATIM to `version` (0 → fresh insert version 1) — the
// version-PRESERVING online-copy primitive used by the reshard copy pass. The
// WAL logs the actual stored version (the supplied version, or 1 when version==0)
// so a later replay restores it verbatim. keyExpires is the ABSOLUTE per-key
// payload deadline map set + WAL-logged VERBATIM (NOT recomputed now+ttl) so the
// online reshard copy keeps the point's original key deadlines time-stable; nil
// keyExpires (a plain if-absent) is byte-identical to today's behavior.
func (c *Collection) InsertIfAbsentVersion(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) (bool, error) {
	c.startSweeper()
	if c.wal == nil {
		return c.idx.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, keyExpires, version)
	}
	// Staged commit (see RestoreInsert): opMu across {apply + WAL WRITE} only, so
	// the reshard copy pass never becomes the fsync leader holding opMu. A no-op
	// (already live) writes nothing and returns seq 0 — commitWaitStaged(0) is a
	// no-op wait, so the contract "every staged seq is waited on" still holds.
	inserted, seq, err := func() (bool, uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		inserted, err := c.idx.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, keyExpires, version)
		if err != nil || !inserted {
			return inserted, 0, err
		}
		// Log the actual stored version so replay restores it verbatim: the supplied
		// version when version-preserving, else 1 (a fresh if-absent insert; HNSW has
		// no in-place vector update).
		logVersion := version
		if logVersion == 0 {
			logVersion = 1
		}
		// Log the ABSOLUTE key deadlines verbatim (nil for a plain if-absent) so a later
		// replay restores them time-stable, mirroring RestoreInsert.
		seq, err := c.wal.appendInsertStaged(id, vec, ttl, meta, sparse, keyExpires, logVersion)
		return inserted, seq, err
	}()
	if err != nil {
		return inserted, err
	}
	return inserted, c.wal.commitWaitStaged(seq)
}

// InsertIfAbsentVersionAt is InsertIfAbsentVersion whose liveness OUTCOME, reclaim,
// and point-TTL deadline are judged against the EXPLICIT leader clock nowMs — the
// replicated-apply variant the vector op handler uses under an apply stamp so
// skewed replicas agree on resurrection and stamp identical deadlines (#4 vector
// TTL determinism).
func (c *Collection) InsertIfAbsentVersionAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) (bool, error) {
	c.startSweeper()
	if c.wal == nil {
		return c.idx.InsertIfAbsentVersionAt(id, vec, ttl, meta, sparse, keyExpires, version, nowMs)
	}
	// Staged commit (see InsertIfAbsentVersion).
	inserted, seq, err := func() (bool, uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		inserted, err := c.idx.InsertIfAbsentVersionAt(id, vec, ttl, meta, sparse, keyExpires, version, nowMs)
		if err != nil || !inserted {
			return inserted, 0, err
		}
		logVersion := version
		if logVersion == 0 {
			logVersion = 1
		}
		seq, err := c.wal.appendInsertStaged(id, vec, ttl, meta, sparse, keyExpires, logVersion)
		return inserted, seq, err
	}()
	if err != nil {
		return inserted, err
	}
	return inserted, c.wal.commitWaitStaged(seq)
}

// SetNowFunc propagates a test/advanced wall-clock override (unix millis) to this
// collection's index. nil restores the real clock. Production never calls it. See
// VectorIndex.SetNowFunc.
func (c *Collection) SetNowFunc(fn func() int64) { c.idx.SetNowFunc(fn) }

// Exists delegates to the underlying index's O(1) liveness probe (true for live;
// false for deleted, expired, or never-inserted).
func (c *Collection) Exists(id uint64) bool { return c.idx.Exists(id) }

// startSweeper launches a goroutine that periodically scans the index
// for expired entries and tombstones them. It is called once on first
// insert, gated by sweepStart. Safe to call concurrently.
//
// When Config.SuppressSweep is set (the persistent-cluster / Raft-replicated
// policy) it is a permanent no-op: the wall-clock sweeper's physical removal would
// diverge committed state across replicas at skewed clocks, so expired entries are
// filtered lazily at read time instead (client staleness only). The sweepStart
// Once is still consumed so a later call stays a no-op and lifecycle teardown
// (which nil-checks sweepStop) is unaffected (#4 vector TTL determinism, B3a).
func (c *Collection) startSweeper() {
	if c.cfg.SuppressSweep {
		c.sweepStart.Do(func() {})
		return
	}
	c.sweepStart.Do(func() {
		c.sweepStop = make(chan struct{})
		c.sweepDone = make(chan struct{})
		interval := c.cfg.SweepInterval
		if interval <= 0 {
			interval = defaultSweepInterval
		}
		go c.runSweeper(interval)
	})
}

func (c *Collection) runSweeper(interval time.Duration) {
	defer close(c.sweepDone)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-c.sweepStop:
			return
		case <-t.C:
			c.idx.sweepOnce()
		}
	}
}

// Stop halts the background sweeper. Idempotent; safe to call even if
// startSweeper was never invoked.
func (c *Collection) Stop() {
	if c.sweepStop == nil {
		return
	}
	select {
	case <-c.sweepStop:
		// already closed
	default:
		close(c.sweepStop)
		<-c.sweepDone
	}
}

// Search delegates to the underlying index.
func (c *Collection) Search(query []float32, k int) ([]Result, error) {
	return c.idx.Search(query, k)
}

// SearchFiltered delegates to the underlying index, applying a metadata filter.
func (c *Collection) SearchFiltered(query []float32, k int, filter Filter) ([]Result, error) {
	return c.idx.SearchFiltered(query, k, filter)
}

// SearchInto delegates to the underlying index, appending results onto dst for
// a zero-allocation hot path (see hnsw.SearchInto).
func (c *Collection) SearchInto(dst []Result, query []float32, k int, filter Filter) ([]Result, error) {
	return c.idx.SearchInto(dst, query, k, filter)
}

// HybridSearch delegates to the underlying index, fusing dense + sparse lanes.
func (c *Collection) HybridSearch(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, error) {
	return c.idx.HybridSearch(dense, sparse, k, opts)
}

// HybridLanes delegates to the underlying index, returning the dense and sparse
// candidate lanes unfused for cross-partition fan-out (see hnsw.HybridLanes).
func (c *Collection) HybridLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, []Result, error) {
	return c.idx.HybridLanes(dense, sparse, k, opts)
}

// SearchText runs a BM25 full-text search over the indexed $content and returns
// the top-k documents (content + metadata), each carrying its BM25 relevance in
// Score. The filter is honored via the same admit/filter machinery as SearchDocs.
// Returns ErrFullTextDisabled when the collection was not created with FullText.
// Full text is an HNSW-only v1 capability, so a non-HNSW index also reports it
// disabled.
func (c *Collection) SearchText(query string, k int, filter Filter) ([]Document, error) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return nil, ErrFullTextDisabled
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	now := uint64(h.now()) // one clock read for the whole text-lane admission scan
	admit := func(slot uint32) bool { return h.admits(slot, pred, now) }
	res := h.SearchText(query, k, admit)
	return h.fetchDocs(res), nil
}

// HybridText fuses a dense KNN lane with a BM25 full-text lane into the top-k.
// The raw query text is tokenized server-side. opts mirrors HybridSearch (fusion
// method, alpha, RRFK, per-lane pool sizes, filter). Returns ErrFullTextDisabled
// when the collection has no full-text lane.
func (c *Collection) HybridText(dense []float32, query string, k int, opts HybridOpts) ([]Result, error) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return nil, ErrFullTextDisabled
	}
	return h.HybridText(dense, query, k, opts)
}

// HybridTextLanes returns the dense and BM25-text candidate lanes UNFUSED for
// cross-partition fan-out (see hnsw.HybridTextLanes). Returns ErrFullTextDisabled
// when the collection has no full-text lane.
func (c *Collection) HybridTextLanes(dense []float32, query string, k int, opts HybridOpts) ([]Result, []Result, error) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return nil, nil, ErrFullTextDisabled
	}
	return h.HybridTextLanes(dense, query, k, opts)
}

// CorpusStats returns this collection's CORPUS-WIDE BM25 statistics for query's
// terms — the phase-0 reader of the global-DF (dfs_query_then_fetch) fan-out: the
// live document count n, the total live token count tokenTotal (for avgdl), and the
// per-query-term document frequency df. Returns zero/nil (no error) when full text
// is disabled — a partition without a BM25 lane contributes nothing to the summed
// global stats. See hnsw.CorpusStats.
func (c *Collection) CorpusStats(query string) (n int, tokenTotal uint64, df map[uint32]int) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return 0, 0, nil
	}
	return h.CorpusStats(query)
}

// SearchTextGlobal runs a BM25 full-text search scored with coordinator-supplied
// GLOBAL corpus stats g (phase 1 of the global-DF fan-out): this shard's local
// postings, the global IDF. Returns the top-k Results (id + Score). Returns
// ErrFullTextDisabled when the collection has no full-text lane. See
// hnsw.SearchTextGlobal.
func (c *Collection) SearchTextGlobal(query string, k int, filter Filter, g BM25GlobalStats) ([]Result, error) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return nil, ErrFullTextDisabled
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	now := uint64(h.now()) // one clock read for the whole text-lane admission scan
	admit := func(slot uint32) bool { return h.admits(slot, pred, now) }
	return h.SearchTextGlobal(query, k, admit, g), nil
}

// SearchTextGlobalDocs is SearchTextGlobal returning ENRICHED documents (content +
// metadata + the global BM25 Score), mirroring SearchText's enrichment so the
// vector_search_text handler's global-DF phase-1 reply is wire-identical in shape
// to its local reply (both EncodeVectorDocs). The text lane scores this shard's
// LOCAL postings with the injected GLOBAL stats g. Returns ErrFullTextDisabled when
// the collection has no full-text lane.
func (c *Collection) SearchTextGlobalDocs(query string, k int, filter Filter, g BM25GlobalStats) ([]Document, error) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return nil, ErrFullTextDisabled
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	now := uint64(h.now()) // one clock read for the whole text-lane admission scan
	admit := func(slot uint32) bool { return h.admits(slot, pred, now) }
	res := h.SearchTextGlobal(query, k, admit, g)
	return h.fetchDocs(res), nil
}

// HybridTextLanesGlobal returns the dense and BM25-text candidate lanes UNFUSED,
// with the text lane scored against the coordinator-supplied GLOBAL stats g (the
// global-DF fan-out's phase 1). Returns ErrFullTextDisabled when the collection has
// no full-text lane. See hnsw.HybridTextLanesGlobal.
func (c *Collection) HybridTextLanesGlobal(dense []float32, query string, k int, opts HybridOpts, g BM25GlobalStats) ([]Result, []Result, error) {
	h, ok := c.idx.(*hnsw)
	if !ok || !h.FullTextEnabled() {
		return nil, nil, ErrFullTextDisabled
	}
	denseRes, textRes, _, err := h.HybridTextLanesGlobal(dense, query, k, opts, g)
	return denseRes, textRes, err
}

// SearchMMR delegates to the underlying index, re-ranking results for diversity
// via Maximal Marginal Relevance.
func (c *Collection) SearchMMR(query []float32, k int, opts MMROpts) ([]Result, error) {
	return c.idx.SearchMMR(query, k, opts)
}

// Recommend delegates to the underlying index, searching from a query
// synthesized out of positive/negative example ids.
func (c *Collection) Recommend(k int, opts RecommendOpts) ([]Result, error) {
	return c.idx.Recommend(k, opts)
}

// Discover delegates to the underlying index, steering results with context
// pairs.
func (c *Collection) Discover(k int, opts DiscoverOpts) ([]Result, error) {
	return c.idx.Discover(k, opts)
}

// DiscoverVecs delegates to the underlying index's resolved-vectors discovery
// path (the Query API leaf form): the target + context-pair example vectors are
// already resolved (the coordinator embeds them), so the index runs the discover
// scorer over its candidates without per-call id resolution.
func (c *Collection) DiscoverVecs(k int, opts DiscoverVecsOpts) ([]Result, error) {
	return c.idx.DiscoverVecs(k, opts)
}

// RecommendVecs delegates to the underlying index's BEST_SCORE recommend path (the
// Query API leaf form): the positive/negative example vectors are already resolved
// (the coordinator embeds them), so the index seeds the pool + runs the bestScore
// scorer over its candidates without per-call id resolution.
func (c *Collection) RecommendVecs(k int, opts RecommendVecsOpts) ([]Result, error) {
	return c.idx.RecommendVecs(k, opts)
}

// BuildConcurrent bulk-loads an empty collection from ids/vecs using `workers`
// goroutines (0 = GOMAXPROCS), exploiting multiple cores for the HNSW build.
// See hnsw.BuildConcurrent for constraints (empty index, heap storage, no
// TTL/metadata/sparse).
func (c *Collection) BuildConcurrent(ids []uint64, vecs [][]float32, workers int) error {
	return c.idx.BuildConcurrent(ids, vecs, workers)
}

// BuildConcurrentMeta is BuildConcurrent carrying an OPTIONAL per-point payload
// (nil/empty metas, or exactly one entry per id). See hnsw.BuildConcurrentMeta.
func (c *Collection) BuildConcurrentMeta(ids []uint64, vecs [][]float32, metas []Metadata, workers int) error {
	return c.idx.BuildConcurrentMeta(ids, vecs, metas, workers)
}

// StageBulk appends (id, vec) pairs to the bulk-load staging buffer. Cheap and
// concurrency-safe (many uploaders can stage in parallel); nothing is indexed
// until BuildStaged. vecs are retained, so the caller must not mutate them.
//
// Every vector must have the collection's Dim; a mismatch returns
// ErrDimMismatch and stages NOTHING from the batch. The check lives here, at the
// shard that owns the config, rather than in a transport: this is the only place
// with the authority to answer it, so every caller of the staging op gets it —
// the REST JSON body, the REST binary wire, the native TCP wire — without any of
// them paying a round-trip to look the dimension up.
//
// BuildConcurrent enforces the same rule, but only when the build finally runs:
// a wrong-dim vector accepted here would be discovered after the whole corpus
// had been staged, failing the entire load instead of the one bad request.
//
// SCOPE — this catches a UNIFORM batch whose dim is wrong for the collection. It
// does NOT catch a RAGGED batch (vectors of differing lengths within one call),
// even though the loop below would reject one, because no remote request can
// deliver a ragged batch here: ops.DecodeBulkStageArgs materializes
// make([]float32, dim) per row from a single wire dim, so anything arriving
// through a transport is uniform by construction. Raggedness is rejected in
// ops.EncodeBulkStageArgs instead, before it can be encoded into shifted rows
// and fabricated ids. (This corrects an earlier claim to the contrary.)
//
// The two checks are deliberately split by what each layer knows: the encoder
// owns "is this batch self-consistent", which needs only the batch; this method
// owns "does it fit this collection", which needs the config and nothing else.
func (c *Collection) StageBulk(ids []uint64, vecs [][]float32) error {
	return c.StageBulkPayloads(ids, vecs, nil)
}

// StageBulkPayloads is StageBulk with an OPTIONAL per-point payload: metas is
// either nil (vectors only — byte-identical to StageBulk, and the staging buffer
// grows no payload column at all) or exactly len(ids) long, with a nil entry for
// each point that carries no payload. Retained like vecs, so the caller must not
// mutate them.
//
// The payloads are applied by the BUILD, in its single-threaded placement pass
// (see hnsw.BuildConcurrentMeta) — which is what lets a filtered workload use the
// multi-core bulk path at all instead of falling back to one indexed insert per
// point. Everything StageBulk documents about dimension checking applies here
// unchanged.
func (c *Collection) StageBulkPayloads(ids []uint64, vecs [][]float32, metas []Metadata) error {
	// THE FULL GATE, HERE, EVEN THOUGH BuildConcurrentMeta RUNS IT AGAIN.
	// BuildStaged takes the staged slices and CLEARS the buffer before it calls
	// the engine, so a record that got past staging is discovered with the
	// caller's whole staged batch already gone. Rejecting at stage time keeps the
	// buffer intact and names the offending row while the caller can still fix it
	// — the same reason the dimension check is here rather than only in the
	// builder. See TestStagedBatchSurvivesARejectedRecord.
	if err := checkRecordValuesAll(metas); err != nil {
		return err
	}
	// NILNESS here, LENGTH in BuildConcurrentMeta, and the difference is deliberate.
	// This function has to decide whether to MATERIALIZE the payload column, and
	// only nil can mean "this caller has no payload column at all" — an empty
	// non-nil slice paired with a non-empty batch is a caller that meant to send
	// payloads and sent the wrong number, which is worth an error rather than a
	// silent vectors-only stage. The builder has no column to materialize and only
	// has to decide whether to INDEX metas[i], for which empty and nil are the same
	// answer; guarding on nilness there would index metas[0] of a zero-length slice
	// (see TestBulkPayloadEmptyColumnIsVectorsOnly).
	if metas != nil && len(metas) != len(ids) {
		return ErrBuildMetaLenMismatch
	}
	dim := c.Config().Dim
	for i, v := range vecs {
		if len(v) != dim {
			return fmt.Errorf("vector: staged vector %d has length %d, collection Dim is %d: %w",
				i, len(v), dim, ErrDimMismatch)
		}
	}
	c.stageMu.Lock()
	// The payload column is materialized on first use and kept dense from then on
	// — see stageMetas. Backfilling here (rather than at build time) keeps the
	// alignment invariant local to the one function that can break it.
	if metas != nil && c.stageMetas == nil && len(c.stageIDs) > 0 {
		c.stageMetas = make([]Metadata, len(c.stageIDs))
	}
	c.stageIDs = append(c.stageIDs, ids...)
	c.stageVecs = append(c.stageVecs, vecs...)
	switch {
	case metas != nil:
		c.stageMetas = append(c.stageMetas, metas...)
	case c.stageMetas != nil:
		// A vectors-only batch arriving after a payload-bearing one still has to
		// occupy its own rows in the column, or every later payload lands on the
		// wrong point.
		c.stageMetas = append(c.stageMetas, make([]Metadata, len(ids))...)
	}
	c.stageMu.Unlock()
	return nil
}

// BuildStaged builds everything staged so far into the index in one concurrent
// pass (workers=0 → GOMAXPROCS) and clears the buffer, applying any staged
// payloads as part of the build. The index must be empty (bulk load is the
// initial-load path); see hnsw.BuildConcurrentMeta. A no-op when nothing is
// staged.
func (c *Collection) BuildStaged(workers int) error {
	c.stageMu.Lock()
	ids, vecs, metas := c.stageIDs, c.stageVecs, c.stageMetas
	c.stageIDs, c.stageVecs, c.stageMetas = nil, nil, nil
	c.stageMu.Unlock()
	if len(ids) == 0 {
		return nil
	}
	return c.idx.BuildConcurrentMeta(ids, vecs, metas, workers)
}

// Upsert inserts or replaces the record for id with the given vector, document
// content, TTL, metadata, and sparse vector — the RAG-store write path. Content
// is retrievable via SearchDocs and survives persistence. Replace is
// delete-then-insert (HNSW has no in-place vector update); both ops are WAL-
// logged when a WAL is configured.
func (c *Collection) Upsert(id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector) error {
	_, err := c.UpsertCAS(id, vec, content, ttl, meta, sparse, CASCond{})
	return err
}

// UpsertKeyTTL is Upsert carrying an OPTIONAL per-key payload TTL map (key ->
// RELATIVE ms) set on the fresh point. Empty/nil behaves exactly like Upsert.
func (c *Collection) UpsertKeyTTL(id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64) error {
	_, err := c.UpsertCASKeyTTL(id, vec, content, ttl, meta, sparse, keyTTLMs, CASCond{})
	return err
}

// UpsertCAS is Upsert with an optimistic-CAS precondition. Because Upsert is
// delete-then-insert (HNSW has no in-place vector update), the version would reset
// across the delete; so the precondition is checked against the point's CURRENT
// version (before the delete) under opMu, then the replace runs unconditionally
// and the resulting version is returned. A mismatch returns ErrVersionConflict
// with NO mutation. The fresh-insert version is always 1.
func (c *Collection) UpsertCAS(id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector, cas CASCond) (uint64, error) {
	return c.UpsertCASKeyTTL(id, vec, content, ttl, meta, sparse, nil, cas)
}

// UpsertCASKeyTTL is UpsertCAS carrying an OPTIONAL per-key payload TTL map (key
// -> RELATIVE ms) set on the fresh point. The engine computes the absolute
// deadlines; the WAL logs them so replay restores them verbatim. Empty/nil
// keyTTLMs is the zero-overhead path.
func (c *Collection) UpsertCASKeyTTL(id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	// THE FULL GATE, HERE, EVEN THOUGH insertBody RUNS IT AGAIN. Upsert is
	// delete-then-insert, and the delete below is irreversible: downgrade this to
	// checkRecordValuesSize and a malformed record deletes the existing point and
	// only THEN fails inside c.idx.Insert, so a REFUSED write destroys the data it
	// was replacing. That is the one shape of this bug that is worse than the
	// poison the gate prevents, so this path decodes the record twice on purpose.
	// See checkRecordValues' doc and TestRecordValidationCountPerWrite.
	if err := checkRecordValues(meta); err != nil {
		return 0, err
	}
	c.startSweeper()
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
	// InsertCASKeyTTL) so a panic inside idx.Get/Delete/Insert or a WAL append
	// still unlocks opMu via the deferred Unlock instead of leaking it forever.
	//
	// waitSeq is the seq the caller must wait on AFTER the closure returns: the
	// insert's seq on the happy path (which also covers the earlier staged
	// delete — same file, same opMu, so a Sync covering the insert covers the
	// delete too), or the delete's OWN seq if the insert never got a seq of its
	// own (a CAS mismatch, dim mismatch, quota rejection, or the append itself
	// failing) — an already-staged delete write is never made durable by
	// anything else, so it must still be waited on before the error surfaces (0
	// = nothing was staged, a no-op wait).
	version, waitSeq, err := func() (uint64, uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		if cas.Has {
			_, _, _, _, cur, _ := c.idx.Get(id) // current version (0 if absent)
			if cur != cas.Expected {
				return 0, 0, ErrVersionConflict
			}
		}
		// Unconditional delete-then-insert; the CAS was already enforced above
		// (passing the engine cas here would re-check against the post-delete
		// version of 0). A failed delete-WAL-append fails the upsert (delSeq is 0
		// on error, so nothing is left to wait on).
		delSeq, derr := c.idxDeleteLogged(id)
		if derr != nil {
			return 0, 0, derr
		}
		version, keyExpires, err := c.idx.Insert(id, vec, ttl, withContent(meta, content), sparse, keyTTLMs, CASCond{})
		if err != nil {
			return 0, delSeq, err
		}
		if c.wal == nil {
			return version, 0, nil
		}
		// Only the LAST record (the insert) needs the durability wait: the delete's
		// bytes were staged (written, not waited-on) earlier under this same opMu, so
		// a Sync covering the insert's seq also covers the delete's — one fsync per
		// upsert instead of two.
		seq, err := c.wal.appendInsertStaged(id, vec, ttl, withContent(meta, content), sparse, keyExpires, version)
		if err != nil {
			return 0, delSeq, err
		}
		return version, seq, nil
	}()
	if err != nil {
		if waitSeq != 0 {
			// Ensure the already-staged delete is durable before surfacing the
			// error (best-effort, mirrors idxDeleteLogged's own discipline): the
			// in-memory delete already happened, so its WAL record must not be
			// left un-waited (appendFramedStaged's contract).
			_ = c.wal.commitWaitStaged(waitSeq)
		}
		return 0, err
	}
	if waitSeq != 0 {
		if werr := c.wal.commitWaitStaged(waitSeq); werr != nil {
			return 0, werr
		}
	}
	return version, nil
}

// idxDeleteLogged deletes id (unconditionally) and, on a WAL-mode collection,
// stages the delete's WAL write. Caller holds opMu. Returns the assigned staged
// sequence (0 when nothing was staged: no WAL, or the delete was a no-op) and
// the WAL append error, if any.
//
// The append error is PROPAGATED, not swallowed: idempotent replay tolerates a
// torn/duplicate delete record, but it must not mask a durability FAILURE — the
// caller fails the whole upsert so the client is never acked on a replace whose
// tombstone never reached disk.
//
// Does NOT wait on this write's own sequence — the upsert replace path folds it
// into the FOLLOWING insert's single commit-wait (fix for the double-fsync-per-
// upsert: the delete's bytes precede the insert's in the file, written under the
// same opMu, so a Sync covering the insert's seq covers this delete's too) — or,
// if the insert never reaches its own append, the CALLER must explicitly wait on
// the returned seq before returning (see UpsertCASKeyTTL): a staged write is
// never itself durable without a later wait on its seq or a larger one.
func (c *Collection) idxDeleteLogged(id uint64) (uint64, error) {
	ok, _ := c.idx.Delete(id, CASCond{})
	if !ok || c.wal == nil {
		return 0, nil
	}
	return c.wal.appendDeleteStaged(id)
}

// SearchDocs runs a filtered KNN search and returns each hit enriched with its
// stored content and user metadata — one call instead of search-then-fetch.
func (c *Collection) SearchDocs(query []float32, k int, filter Filter) ([]Document, error) {
	res, err := c.idx.SearchFiltered(query, k, filter)
	if err != nil {
		return nil, err
	}
	return c.idx.fetchDocs(res), nil
}

// SearchGroups runs a group-by-document search: it collapses KNN hits sharing a
// metadata field value into groups, returning the top-k groups (best member
// first) with up to opts.GroupSize hits each. See hnsw.SearchGroups.
func (c *Collection) SearchGroups(query []float32, k int, opts GroupOpts) ([]Group, error) {
	return c.idx.SearchGroups(query, k, opts)
}

// GroupCandidates returns the top-(opts.FetchK) candidate documents for a
// group-by-document search WITHOUT grouping — used by the cross-partition
// coordinator for exact group fan-out. See hnsw.GroupCandidates.
func (c *Collection) GroupCandidates(query []float32, opts GroupOpts) ([]Document, error) {
	return c.idx.GroupCandidates(query, opts)
}

// ScrollDocs lists live documents matching filter (zero filter = all), enriched
// with content + metadata, up to limit (0 = no cap). A query-less listing used
// by framework adapters. See hnsw.scrollDocs.
func (c *Collection) ScrollDocs(filter Filter, limit int) ([]Document, error) {
	return c.idx.scrollDocs(filter, limit)
}

// ScrollDocsPage is the cursor-aware scroll: it returns up to limit live docs
// matching filter (zero filter = all) whose id is strictly greater than afterID
// when hasAfter (otherwise from the smallest id), id-ASCENDING. It returns the
// docs, nextAfter (the largest id returned — the next cursor's lower bound), and
// hasMore (true iff the page filled limit AND a further matching-candidate id
// remains). The deep-pagination resume-after-id primitive backing the partition
// fan-out. See hnsw.scrollPage.
func (c *Collection) ScrollDocsPage(filter Filter, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, 0, false, err // fail loud on a malformed filter (matches ScrollDocs)
	}
	docs, nextAfter, hasMore = c.idx.scrollPage(filter, pred, nil, nil, afterID, 0, hasAfter, limit)
	return docs, nextAfter, hasMore, nil
}

// ScrollDocsPageOrder is the order_by-aware cursor scroll: it returns up to limit
// live docs matching filter, ordered by the order_by field's (value, id) total order
// (see OrderBy / OrderLess). Points whose order field is missing/non-numeric are
// EXCLUDED. afterKey/afterID is the resume cursor's (value, id) position (the page
// returns rows strictly after it); on page 1 (hasAfter=false) order.StartFrom (when
// set) is the inclusive starting value bound. Returns docs, the last collected id
// (nextAfter), and hasMore. The order field travels in each Document's Metadata so
// the coordinator can read the last doc's order value for the v2 next-cursor.
//
// order == nil falls back to the plain id-ascending ScrollDocsPage path (so callers
// can pass a possibly-nil order uniformly).
func (c *Collection) ScrollDocsPageOrder(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, 0, false, err
	}
	docs, nextAfter, hasMore = c.idx.scrollPage(filter, pred, nil, order, afterID, afterKey, hasAfter, limit)
	return docs, nextAfter, hasMore, nil
}

// ScanVectors returns every live record (id, vec, remaining TTL, metadata,
// sparse) as self-contained, deep-copied ScanRecords — the read primitive an
// offline resplit uses to re-insert each vector into a re-hashed generation.
// Liveness matches ScrollDocs (tombstoned/expired records are excluded). See
// hnsw.scanVectors.
func (c *Collection) ScanVectors() []ScanRecord { return c.idx.scanVectors() }

// DeleteByFilter deletes every record whose metadata matches filter (e.g. all
// chunks of a document), returning the count removed. The deletes are WAL-logged.
// A zero/match-all filter is rejected (ErrEmptyFilter).
func (c *Collection) DeleteByFilter(filter Filter) (int, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return 0, err
	}
	if filter.IsZero() || pred == nil {
		return 0, ErrEmptyFilter
	}
	// Gather matching ids from the index, then delete via the WAL-aware path.
	ids, err := c.idx.matchingIDs(filter, pred)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, id := range ids {
		// Use the error-returning delete: a durability failure mid-batch (WAL
		// append/fsync error, or a poisoned WAL) must surface, not report a clean
		// batch. Return the count removed so far plus the first error.
		removed, derr := c.DeleteCAS(id, CASCond{})
		if derr != nil {
			return n, derr
		}
		if removed {
			n++
		}
	}
	return n, nil
}

// Delete delegates to the underlying index, logging the delete to the WAL on
// success when one is configured.
func (c *Collection) Delete(id uint64) bool {
	ok, _ := c.DeleteCAS(id, CASCond{})
	return ok
}

// DeleteCAS is Delete with an optimistic-CAS precondition (CASCond{} = no
// precondition). On a mismatch it returns ErrVersionConflict and removed=false
// with no mutation; otherwise removed reports whether id was live. The version is
// dropped with the point (a later reinsert starts at 1).
func (c *Collection) DeleteCAS(id uint64, cas CASCond) (bool, error) {
	if c.wal == nil {
		return c.idx.Delete(id, cas)
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
	// InsertCASKeyTTL) so a panic inside idx.Delete or the WAL append still
	// unlocks opMu via the deferred Unlock instead of leaking it forever. The
	// durability wait runs OUTSIDE opMu: a deleting writer used to become the
	// fsync leader while holding the collection's op lock, which stalled every
	// staged writer behind it AND serialized delete-heavy workloads into one
	// fsync per delete. Now concurrent deletes overlap in commitWaitStaged and
	// batch into a shared flight exactly like inserts.
	var ok bool
	var seq uint64
	err := func() error {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		var derr error
		ok, derr = c.idx.Delete(id, cas)
		if derr != nil {
			return derr
		}
		if ok {
			// Propagate the WAL append error (mirrors InsertCASKeyTTL). Replay is
			// idempotent — a torn/duplicate delete record on replay is tolerated —
			// but that is NOT a licence to swallow a durability FAILURE: a client
			// acked on a delete whose record never reached disk would see the point
			// resurrect after a crash. A real append error fails the delete.
			var serr error
			seq, serr = c.wal.appendDeleteStaged(id)
			if serr != nil {
				return serr
			}
		}
		return nil
	}()
	if err != nil {
		return false, err
	}
	// Wait outside opMu. seq == 0 means nothing was staged (no-op delete: id was
	// not live) — commitWaitStaged(0) is a legit no-op returning nil, so a
	// not-removed delete stays a success. A real fsync failure propagates.
	if werr := c.wal.commitWaitStaged(seq); werr != nil {
		return false, werr
	}
	return ok, nil
}

// Get retrieves the live record for id (deep-copied vector + payload + sparse +
// remaining TTL); ok is false for an absent/tombstoned/expired point. A pure read
// — never WAL-logged. See hnsw.Get (incl. the cosine-normalized-vector caveat).
func (c *Collection) Get(id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	return c.idx.Get(id)
}

// GetInto is Get that appends the vector into the caller-owned scratch dst
// (passed as dst[:0]) instead of allocating a fresh []float32 each call — the
// dense-read analogue of cache.GetInto. A hot-loop caller reusing one buffer
// pays zero allocations for the vector copy. See VectorIndex.GetInto for the
// aliasing and PQDropVecs caveats.
func (c *Collection) GetInto(dst []float32, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	return c.idx.GetInto(dst, id)
}

// projectedGetter is the optional projection-aware get an index may implement so a
// batch reader can skip the dense/meta/sparse copies for projections it will
// discard. hnsw and ivf implement it; GetProjected type-asserts and falls back to a
// full Get for any index that does not (e.g. gpuIndex).
type projectedGetter interface {
	GetProjected(id uint64, withVec, withPayload bool) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool)
}

// GetProjected is Get that skips the copies for unrequested projections: with
// withVec=false the dense vector is not copied, with withPayload=false the meta map
// and sparse vector are not cloned. It is the batch-read fast path (a
// vector_get_batch with with_vector=false no longer allocates a per-point vector
// copy). Indexes that implement projectedGetter (hnsw, ivf) do the skip at the
// source; any other index falls back to a full Get with the unrequested projections
// dropped after the fact (correct, just not allocation-optimal). Semantics otherwise
// match Get.
func (c *Collection) GetProjected(id uint64, withVec, withPayload bool) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	if pg, okIface := c.idx.(projectedGetter); okIface {
		return pg.GetProjected(id, withVec, withPayload)
	}
	vec, meta, ttl, sparse, version, ok = c.idx.Get(id)
	if !withVec {
		vec = nil
	}
	if !withPayload {
		meta, sparse = nil, nil
	}
	return vec, meta, ttl, sparse, version, ok
}

// SetPayload merges patch into id's payload (patch keys overwrite/add). The
// engine computes the resulting full payload (reindexing the payload index
// atomically), and on a WAL-mode collection that exact resulting payload is
// logged — so engine and WAL agree on what was applied. Returns ErrIDNotFound
// for an absent/tombstoned/expired point. Does not change the vector or TTL.
func (c *Collection) SetPayload(id uint64, patch Metadata, keyTTLMs map[string]int64) error {
	_, err := c.SetPayloadCAS(id, patch, keyTTLMs, CASCond{})
	return err
}

// SetPayloadCAS is SetPayload with an optimistic-CAS precondition; it returns the
// resulting version on success and ErrVersionConflict (no mutation) on a mismatch.
func (c *Collection) SetPayloadCAS(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.SetPayload(id, patch, keyTTLMs, cc)
	}, id)
}

// OverwritePayload replaces id's entire payload with meta. WAL-logs the resulting
// payload on a WAL-mode collection. Returns ErrIDNotFound for a dead point.
func (c *Collection) OverwritePayload(id uint64, meta Metadata, keyTTLMs map[string]int64) error {
	_, err := c.OverwritePayloadCAS(id, meta, keyTTLMs, CASCond{})
	return err
}

// OverwritePayloadCAS is OverwritePayload with an optimistic-CAS precondition.
func (c *Collection) OverwritePayloadCAS(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.OverwritePayload(id, meta, keyTTLMs, cc)
	}, id)
}

// DeletePayloadKeys removes the listed keys from id's payload (absent keys =
// no-op). WAL-logs the resulting payload. Returns ErrIDNotFound for a dead point.
func (c *Collection) DeletePayloadKeys(id uint64, keys []string) error {
	_, err := c.DeletePayloadKeysCAS(id, keys, CASCond{})
	return err
}

// DeletePayloadKeysCAS is DeletePayloadKeys with an optimistic-CAS precondition.
func (c *Collection) DeletePayloadKeysCAS(id uint64, keys []string, cas CASCond) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.DeletePayloadKeys(id, keys, cc)
	}, id)
}

// ClearPayload removes all of id's payload. WAL-logs the resulting (empty)
// payload. Returns ErrIDNotFound for a dead point.
func (c *Collection) ClearPayload(id uint64) error {
	_, err := c.ClearPayloadCAS(id, CASCond{})
	return err
}

// ClearPayloadCAS is ClearPayload with an optimistic-CAS precondition.
func (c *Collection) ClearPayloadCAS(id uint64, cas CASCond) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.ClearPayload(id, cc)
	}, id)
}

// SetPayloadCASAt / OverwritePayloadCASAt / DeletePayloadKeysCASAt /
// ClearPayloadCASAt are the ...At variants the replicated apply path uses under a
// leader apply stamp: the engine judges the per-key deadline computation (Set/
// Overwrite) AND the dead-point liveness gate against nowMs, so every replica
// stamps byte-identical per-key deadlines and agrees on liveness (#4 vector TTL
// determinism). WAL logging (single-node only) is unchanged.
func (c *Collection) SetPayloadCASAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.SetPayloadAt(id, patch, keyTTLMs, cc, nowMs)
	}, id)
}

func (c *Collection) OverwritePayloadCASAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.OverwritePayloadAt(id, meta, keyTTLMs, cc, nowMs)
	}, id)
}

func (c *Collection) DeletePayloadKeysCASAt(id uint64, keys []string, cas CASCond, nowMs int64) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.DeletePayloadKeysAt(id, keys, cc, nowMs)
	}, id)
}

func (c *Collection) ClearPayloadCASAt(id uint64, cas CASCond, nowMs int64) (uint64, error) {
	return c.payloadOpCAS(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, error) {
		return c.idx.ClearPayloadAt(id, cc, nowMs)
	}, id)
}

// MutatePayloadRecordCAS applies fn to the operate record stored under key in
// id's payload, atomically with the CAS check and the version bump, and WAL-logs
// the RESULTING payload (a replace, not a merge) exactly as every other payload
// mutation does. A deliberate no-op — fn reporting RecordUnchanged (a failed
// CHECK), or a delete of an already-absent key — writes no WAL record and
// returns the current, unbumped version. Returns ErrIDNotFound for a dead point,
// ErrVersionConflict on a CAS mismatch, ErrPayloadKeyNotRecord when the key
// holds a non-record value, ErrRecordTooLarge when the resulting record exceeds
// the storage cap, and fn's own error verbatim — in every case with the point
// left exactly as it was.
func (c *Collection) MutatePayloadRecordCAS(id uint64, key string, fn RecordMutator, cas CASCond) (uint64, error) {
	return c.payloadOpCASChanged(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, bool, error) {
		return c.idx.MutatePayloadRecord(id, key, fn, cc)
	}, id)
}

// MutatePayloadRecordCASAt is MutatePayloadRecordCAS under a leader apply stamp:
// the engine judges the dead-point liveness gate, the per-key deadline check and
// the stale-deadline drop against nowMs, so every replica reads the same record
// and stores the same bytes (#4 vector TTL determinism).
func (c *Collection) MutatePayloadRecordCASAt(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (uint64, error) {
	return c.payloadOpCASChanged(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, bool, error) {
		return c.idx.MutatePayloadRecordAt(id, key, fn, cc, nowMs)
	}, id)
}

// DeleteCASAt is DeleteCAS judging the dead-slot liveness gate against the leader
// apply stamp nowMs, so replicas agree on already-dead vs live and keep identical
// tombstone sets (#4 vector TTL determinism).
func (c *Collection) DeleteCASAt(id uint64, cas CASCond, nowMs int64) (bool, error) {
	if c.wal == nil {
		return c.idx.DeleteAt(id, cas, nowMs)
	}
	// Staged commit-wait, identical in shape to DeleteCAS above: {apply + WAL
	// WRITE} under opMu in a panic-safe closure, durability wait outside so a
	// stamped delete never holds opMu across an fsync.
	var ok bool
	var seq uint64
	err := func() error {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		var derr error
		ok, derr = c.idx.DeleteAt(id, cas, nowMs)
		if derr != nil {
			return derr
		}
		if ok {
			// Propagate the WAL append error (mirrors DeleteCAS / InsertCASKeyTTL):
			// idempotent replay tolerates torn duplicates, but a durability failure
			// must fail the delete, not silently ack it.
			var serr error
			seq, serr = c.wal.appendDeleteStaged(id)
			if serr != nil {
				return serr
			}
		}
		return nil
	}()
	if err != nil {
		return false, err
	}
	// seq == 0 (no-op delete) makes commitWaitStaged a no-op returning nil; a real
	// fsync failure propagates and fails the delete.
	if werr := c.wal.commitWaitStaged(seq); werr != nil {
		return false, werr
	}
	return ok, nil
}

// DeleteByFilterAt is DeleteByFilter judging BOTH the candidate-selection admission
// gate (matchingIDsAt) and each delete against the leader apply stamp nowMs, so a
// replicated delete-by-filter selects and tombstones the SAME id set on every
// replica (#4 vector TTL determinism).
func (c *Collection) DeleteByFilterAt(filter Filter, nowMs int64) (int, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return 0, err
	}
	if filter.IsZero() || pred == nil {
		return 0, ErrEmptyFilter
	}
	ids, err := c.idx.matchingIDsAt(filter, pred, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
	if err != nil {
		return 0, err
	}
	n := 0
	for _, id := range ids {
		// Surface the first durability failure mid-batch instead of silently
		// reporting a clean batch (mirrors DeleteByFilter).
		ok, derr := c.DeleteCASAt(id, CASCond{}, nowMs)
		if derr != nil {
			return n, derr
		}
		if ok {
			n++
		}
	}
	return n, nil
}

// UpsertCASKeyTTLAt is UpsertCASKeyTTL under a leader apply stamp: it enforces the
// CAS precondition against the STAMP-live version and tombstones the prior point via
// DeleteAt (a CAS-enforcing stamped delete — the At analog of the wall-clock Get
// precheck + unconditional delete), then re-inserts via InsertAt so the point/per-key
// deadlines are stamped identically on every replica (#4 vector TTL determinism).
func (c *Collection) UpsertCASKeyTTLAt(id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	// The full gate before the tombstone, for the reason UpsertCASKeyTTL states:
	// this path deletes before it inserts, so a size-only check here would let a
	// malformed record destroy the point it was refused from replacing.
	if err := checkRecordValues(meta); err != nil {
		return 0, err
	}
	c.startSweeper()
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
	// UpsertCASKeyTTL) so a panic inside idx.DeleteAt/InsertAt or a WAL append
	// still unlocks opMu via the deferred Unlock instead of leaking it forever.
	// waitSeq follows the same contract as UpsertCASKeyTTL's: the insert's seq on
	// the happy path (covers the earlier staged delete too), or the delete's own
	// seq if the insert never got one of its own — 0 means nothing was staged.
	version, waitSeq, err := func() (uint64, uint64, error) {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		// CAS-enforcing stamped delete: on a mismatch it returns ErrVersionConflict
		// with no mutation (like the non-At Get precheck); with CASCond{} it is the
		// unconditional replace-delete. Judges liveness on the stamp.
		removed, err := c.idx.DeleteAt(id, cas, nowMs)
		if err != nil {
			return 0, 0, err
		}
		// Stage the delete's WAL write (no wait of its own): its durability is
		// folded into the following insert's single commit-wait on the happy path,
		// or explicitly waited on by the caller below otherwise.
		var delSeq uint64
		if removed && c.wal != nil {
			// Propagate the delete-WAL-append error (mirrors UpsertCASKeyTTL): a
			// durability failure fails the upsert rather than silently acking a
			// replace whose tombstone never reached disk. delSeq stays 0 on error.
			var derr error
			delSeq, derr = c.wal.appendDeleteStaged(id)
			if derr != nil {
				return 0, delSeq, derr
			}
		}
		version, keyExpires, err := c.idx.InsertAt(id, vec, ttl, withContent(meta, content), sparse, keyTTLMs, CASCond{}, nowMs)
		if err != nil {
			return 0, delSeq, err
		}
		if c.wal == nil {
			return version, 0, nil
		}
		// Only the LAST record (the insert) needs the durability wait: the delete's
		// bytes were staged earlier under this same opMu, so a Sync covering the
		// insert's seq also covers the delete's — one fsync per upsert instead of two.
		seq, err := c.wal.appendInsertStaged(id, vec, ttl, withContent(meta, content), sparse, keyExpires, version)
		if err != nil {
			return 0, delSeq, err
		}
		return version, seq, nil
	}()
	if err != nil {
		if waitSeq != 0 {
			// Ensure the already-staged delete is durable before surfacing the
			// error: the in-memory delete already happened, so its WAL record must
			// not be left un-waited (appendFramedStaged's contract).
			_ = c.wal.commitWaitStaged(waitSeq)
		}
		return 0, err
	}
	if waitSeq != 0 {
		if werr := c.wal.commitWaitStaged(waitSeq); werr != nil {
			return 0, werr
		}
	}
	return version, nil
}

// payloadOpCASChanged is payloadOpCAS for an op that may legitimately apply
// NOTHING (a record mutation whose CHECK failed): when op reports changed=false
// the version it returns is the current, unbumped one, and no WAL record is
// staged — a no-op must not cost a log entry or move a version a CAS loop is
// watching.
//
// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
// InsertCASKeyTTL) so a panic inside op or the WAL append still unlocks opMu via
// the deferred Unlock instead of leaking it forever. The durability wait runs
// outside opMu so concurrent payload writers group-commit instead of each paying
// a serialized fsync under the collection's op lock.
func (c *Collection) payloadOpCASChanged(cas CASCond, op func(CASCond) (Metadata, map[string]uint64, uint64, bool, error), id uint64) (uint64, error) {
	if c.wal == nil {
		_, _, version, _, err := op(cas)
		return version, err
	}
	var seq, version uint64
	var changed bool
	err := func() error {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		resulting, ke, v, ch, err := op(cas)
		if err != nil {
			return err
		}
		version, changed = v, ch
		if !ch {
			return nil
		}
		seq, err = c.wal.appendSetPayloadStaged(id, resulting, ke, v)
		return err
	}()
	if err != nil {
		return 0, err
	}
	if !changed {
		return version, nil
	}
	if err := c.wal.commitWaitStaged(seq); err != nil {
		return 0, err
	}
	return version, nil
}

// payloadOpCAS is the shared CAS-aware payload-mutation core: it runs the engine
// op (which enforces the CAS precondition + bumps the version) and, on a WAL-mode
// collection, logs the resulting payload + per-key deadlines + version under opMu.
// Returns the resulting version; ErrVersionConflict (or ErrIDNotFound) propagate
// from the engine with no WAL append.
//
// It is the always-changed form every payload mutation other than the record
// mutator takes. Behaviour-preserving: payloadOpCASChanged tests err before it
// reads changed, so the hard-coded true is never consulted on an error path.
func (c *Collection) payloadOpCAS(cas CASCond, op func(CASCond) (Metadata, map[string]uint64, uint64, error), id uint64) (uint64, error) {
	return c.payloadOpCASChanged(cas, func(cc CASCond) (Metadata, map[string]uint64, uint64, bool, error) {
		m, ke, v, err := op(cc)
		return m, ke, v, true, err
	}, id)
}

// replayWAL re-applies a WAL's records onto the (just-restored) index. Apply
// errors are ignored — replay is meant to be idempotent on top of the checkpoint.
func (c *Collection) replayWAL(path string) error {
	return replayWAL(path,
		// Replay an insert via the version-preserving RestoreInsert so the logged
		// version is restored VERBATIM (NOT re-bumped). An old WAL record (no version
		// block) replays version 0 → RestoreInsert defaults a fresh insert to 1.
		func(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) {
			// Restore the logged absolute per-key deadlines VERBATIM (NOT recomputed
			// now+ttl) so pending key deadlines survive a crash time-stable.
			_ = c.idx.RestoreInsert(id, vec, ttl, meta, sparse, keyExpires, version)
		},
		func(id uint64) { _, _ = c.idx.Delete(id, CASCond{}) },
		// Replay a payload mutation by restoring the logged resulting payload AND
		// its absolute per-key deadlines AND its version verbatim (RestorePayload
		// reindexes the payload index and does NOT recompute now+ttl or re-bump, so
		// pending key deadlines + the version survive a crash time-stable).
		// Idempotent on top of the checkpoint.
		func(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) {
			_ = c.idx.RestorePayload(id, meta, keyExpires, version)
		},
	)
}

// SavePersist writes the collection's native instant-restart sidecar to
// metaPath (the mmap-backed vectors/graph are flushed in place). Requires a
// Persistent collection; see hnsw.SavePersist for the v1 constraints.
func (c *Collection) SavePersist(metaPath string) error { return c.idx.SavePersist(metaPath) }

// Snapshot delegates to the underlying index.
func (c *Collection) Snapshot(w io.Writer) error { return c.idx.Snapshot(w) }

// Restore delegates to the underlying index.
func (c *Collection) Restore(r io.Reader) error { return c.idx.Restore(r) }

// Stats delegates to the underlying index.
func (c *Collection) Stats() Stats { return c.idx.Stats() }

// WritePrometheus writes the collection's stats to w in the Prometheus text
// exposition format, labeled with this collection's name.
func (c *Collection) WritePrometheus(w io.Writer) error {
	return c.idx.Stats().WritePrometheus(w, c.name)
}

// Close delegates to the underlying index and closes the WAL, if any.
func (c *Collection) Close() error {
	// Join the sweeper on every teardown/error path (idempotent, safe before start) so a
	// load-error Close after startSweeper doesn't leak the goroutine. Close holds no lock,
	// so this can't deadlock with the sweeper.
	c.Stop()
	err := c.idx.Close()
	if c.wal != nil {
		if werr := c.wal.close(); werr != nil && err == nil {
			err = werr
		}
	}
	return err
}

// Reclaim physically removes tombstoned vectors and prunes dangling edges.
// Returns the number of slots reclaimed.
func (c *Collection) Reclaim() int { return c.idx.Reclaim() }

// TombstoneRatio returns the fraction of arena slots that are tombstoned.
// Used by CollectionStore.MaybeReclaim to decide when reclaim is worth running.
func (c *Collection) TombstoneRatio() float64 {
	st := c.idx.Stats()
	total := st.Size + st.Tombstoned
	if total == 0 {
		return 0
	}
	return float64(st.Tombstoned) / float64(total)
}
