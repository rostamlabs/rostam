// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Multi-vector (late-interaction / ColBERT-style) retrieval. A document is
// represented by many token-level vectors instead of one; relevance is the
// MaxSim score — for each query token, the best similarity to any document
// token, summed over query tokens:
//
//	score(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} cos(q, d)
//
// This captures fine-grained term interactions a single pooled vector loses, at
// the cost of storing N vectors per document. Retrieval is two-stage, the
// standard late-interaction recipe:
//
//  1. First stage (approximate): every document token is a node in one HNSW
//     index; each query token runs an ANN search there, and the parent documents
//     of the nearest token nodes form a candidate set.
//  2. Second stage (exact): each candidate is scored by full MaxSim over its
//     stored token vectors, and the top k by score are returned.
//
// Vectors are L2-normalized (the index is Cosine), so MaxSim's per-pair
// similarity is a single dot product on the SIMD kernel.

// MultiVectorConfig (the MultiVectorIndex configuration) now lives in the
// engine-free vtypes leaf package and is re-exported from vtypes_aliases.go. Its
// engine-coupled derivations (metaPath / mapsPath / mvInnerConfig) are the free
// functions mvMetaPath / mvMapsPath / mvInnerConfig below.

// MultiSearchOpts tunes a multi-vector search.
type MultiSearchOpts struct {
	// CandidatesPerToken is how many nearest token nodes each query token
	// contributes to the first-stage candidate set. 0 = max(4*k, 50). Larger
	// values raise recall (more documents reach the exact MaxSim rerank) at the
	// cost of more first-stage work.
	CandidatesPerToken int

	// ReadConsistency and OnPartitionUnavailable are cross-shard routing knobs
	// consumed by the clustered fan-out coordinator; the single-node engine
	// ignores them. 0 = AnyReplica / Partial (defaults); 1 = LeaderOnly / Fail.
	ReadConsistency        uint8
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's committed
	// frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64

	// Filter restricts results to documents whose payload metadata matches. The
	// zero value matches everything (compiles to a nil predicate) and leaves the
	// search path byte/behaviour-identical to no-filter. Reuses the dense filter
	// eval verbatim (Compile -> Predicate(Metadata)bool), evaluated against the
	// same docMeta map the rerank already reads.
	Filter Filter
}

// MultiResult (one scored document from a multi-vector search) now lives in the
// engine-free vtypes leaf package and is re-exported from vtypes_aliases.go.

// MultiVectorIndex is a late-interaction index: many vectors per document, MaxSim
// scoring. Safe for concurrent use.
type MultiVectorIndex struct {
	mu  sync.RWMutex
	idx VectorIndex // one node per document token vector (HNSW or IVF inner per cfg.IndexType)

	nextToken uint64              // monotonic synthetic token-node id (never 0)
	tokenDoc  map[uint64]uint64   // token-node id -> owning document id
	docTokens map[uint64][]uint64 // document id -> its token-node ids
	docMeta   map[uint64]Metadata // document id -> metadata (nil if none)
	dim       int

	// keyTTL is the per-key payload TTL side-structure: docID -> key -> ABSOLUTE
	// unix-millis deadline. A key is logically expired iff its deadline is non-zero
	// and <= now (mirrors named/dense per-key TTL). MV has no point TTL, so per-key
	// TTL is the only expiry here. A doc with no per-key TTL has no entry (the cheap
	// common case). Lazy-only: expired keys are dropped on every read (Get / scroll
	// / the Stage-2 filter pred view + the returned MultiResult payload); no sweep.
	// Snapshot-durable (no WAL); old snapshots restore it empty.
	keyTTL map[uint64]map[string]int64

	// version is the per-document optimistic-CAS version side-structure: docID ->
	// monotonic uint64 version (the MV-family analogue of the dense arena versions
	// side-array). A new Add sets it to 1, every in-place mutate (replace-Add / any
	// payload op) bumps it +1, Delete drops it (an absent doc reads version 0), and
	// a re-Add of a previously-deleted docID starts at 1 again. Stored ALONGSIDE
	// docMeta/keyTTL (mirroring keyTTL's storage / persist / WAL pattern EXACTLY) so
	// it persists verbatim. The check + bump are atomic under m.mu (the FSM-Apply
	// serialization point), so the result is deterministic across Raft replicas. A
	// doc with no version has no entry (version 0 = absent). Persist-durable (maps
	// sidecar + snapshot) + WAL-logged; old artifacts restore it from a sane default.
	version map[uint64]uint64

	// now is the injectable clock for the per-key TTL deadlines (both the absolute
	// deadline computation on set_payload and the lazy-drop expiry check). Defaults
	// to time.Now().UnixMilli when nil; tests override it to age deterministically.
	// Mirrors hnsw.now / NamedCollection.now. nil-default keeps wall-clock behavior.
	now func() int64

	// Persistence (set only in Persistent mode). The inner mmap-backed index is
	// saved via its instant-restart sidecar (metaPath); the doc<->token maps go
	// in a parallel sidecar (mapsPath).
	persistent bool
	metaPath   string
	mapsPath   string

	// cfg retains the construction config (used to re-create the index when
	// restoring a cluster snapshot).
	cfg MultiVectorConfig

	// Write-ahead log (nil unless WAL-mode, single-node only). opMu serializes
	// {apply + WAL append} against {checkpoint + WAL rotate} (FlushMVWAL) so a Flush
	// never truncates an op it didn't capture — exactly the dense/named discipline
	// (see collection.go / named.go). Only taken when wal != nil; a heap-only or
	// mmap-Persistent (non-WAL) index pays nothing. opMu is the OUTER lock: a mutator
	// takes opMu, then m.mu for the actual state change.
	//
	// WAL mode is a HEAP-checkpoint mode (open = restore heap snapshot file + replay
	// WAL; Flush = write heap snapshot file + truncate WAL), DISTINCT from and
	// mutually exclusive with the mmap Persistent mode (open = mmap instant-restart;
	// Flush = SavePersist sidecar). They share the snapshot/restore inner-blob+maps
	// codec but write to DIFFERENT destinations.
	wal  *wal
	opMu sync.Mutex

	// sweeper lifecycle (per-key TTL background reclaim). Mirrors the dense
	// Collection sweeper (collection.go): startSweeper launches a single ticker
	// goroutine on the first add (gated by sweepStart), runSweeper calls
	// sweepKeyTTLOnce each tick, and Stop joins the goroutine (closes sweepStop,
	// waits sweepDone). An index with no per-key TTL still runs the ticker but each
	// sweepKeyTTLOnce is a cheap no-op (matches dense + named). nil sweepStop =
	// never started, so Stop is a safe no-op.
	sweepStop  chan struct{}
	sweepDone  chan struct{}
	sweepStart sync.Once
	// suppressSweep permanently no-ops startSweeper (the replicated/persistent-cluster
	// policy — see MultiVectorConfig.SuppressSweep). Expired per-key entries are then
	// filtered lazily at read time only; no wall-clock physical removal that could
	// diverge committed state across replicas (#4 vector TTL determinism, B3a).
	suppressSweep bool

	// inuse counts in-flight operations holding this index (taken by
	// CollectionStore.AcquireMulti, dropped by Release). retire drains it before
	// unmapping, so a concurrent cluster RestoreAll / DropMultiVector never
	// munmaps mmap-backed storage under a reader. Mirrors Collection.inuse.
	inuse atomic.Int64

	// docSparse holds the OPTIONAL doc-level sparse vector for each document (the
	// authoritative copy, snapshot/WAL-durable). A document MAY omit it (the common
	// dense-only MV case): an absent entry means "no sparse lane for this doc", so a
	// dense-only MV carries no docSparse entries and stays byte/behaviour-identical
	// (the persist block is OMITTED and the add wire trailer is absent). It is the MV
	// analogue of NamedCollection's namedSparseSpace.vecs — id-keyed, since each MV
	// doc holds at most ONE doc-level sparse vector. Guarded by m.mu EXACTLY like
	// docMeta (every add/replace/delete mutates it under the same lock). The
	// MaxSim/inner-HNSW path is UNTOUCHED — this is a standalone parallel structure.
	docSparse map[uint64]*SparseVector

	// sparseIdx is the id-keyed inverted index over the doc-level sparse vectors (the
	// reused sparseIndexID, the same structure NamedCollection's sparse spaces use).
	// It is kept in sync with docSparse incrementally under m.mu (every docSparse
	// add/replace/delete has a matching sparseIdx.add/remove) and is rebuilt-on-load
	// from docSparse on EVERY load path (snapshot restore + mmap reopen + WAL replay),
	// mirroring payloadIdx.rebuild — it is never serialized. searchTopK(query, k,
	// admit) is the sparse-lane engine surface the MVHybrid calls; the MaxSim
	// lane reuses the existing Search/maxSimLocked untouched.
	sparseIdx *sparseIndexID

	// payloadIdx is the id-keyed inverted index over the per-doc payload (the
	// id-keyed analogue of the dense hnsw.payloadIdx and NamedCollection.payloadIdx).
	// It accelerates SELECTIVE filtered MaxSim searches via the filter-first path:
	// candidates() -> a candidate-doc SUPERSET -> a BRUTE-FORCE exact MaxSim rerank
	// over ONLY those docs (skipping the token-HNSW Stage-1 gather), with the
	// existing Stage-2 live-meta predicate RE-CHECK. It is an ACCELERATION ONLY: the
	// adaptive-over-fetch Stage-1 + Stage-2 post-filter path remains the correct
	// fallback for no-filter / non-accelerable / non-selective filters, and the
	// re-check rejects any over-cover. Rebuilt-on-load (never serialized) from
	// m.docMeta. Guarded by m.mu EXACTLY like m.docMeta (every payload-set/add/delete
	// reindexes under the same lock).
	payloadIdx *payloadIndexID

	// dataVersion is the order_by snapshot's invalidation counter (the MV-family
	// analogue of hnsw.dataVersion). MV has no idSetVersion; this is a fresh counter
	// that bumps on the UNION of doc-set mutations {add/restore-add/if-absent/delete}
	// AND payload mutations {set/overwrite/deleteKeys/clear/restorePayload}: a cached
	// (field, direction) sorted snapshot is payload-VALUE-keyed, so a payload write to
	// the order field MUST invalidate it. Guarded by m.mu; bumped via bumpData() under
	// the write lock at each mutator's innermost *Locked body (exactly once per
	// mutation — a replace-Add calls removeLocked then bumps ONCE in addLocked, not
	// in removeLocked, to avoid a double-bump). Starts at 1 (the zero-valued
	// orderSnap.ver is reserved for "never built").
	dataVersion uint64

	// orderSnaps caches per-(field, direction) sorted snapshots for the order_by
	// scroll, version-stamped with dataVersion. Bounded at orderCacheCap with
	// oldest-built eviction. Guarded by m.mu (warm read under RLock of an immutable
	// rows slice; cold rebuild under Lock, double-checked). See order.go orderSnap.
	orderSnaps map[orderCacheKey]*orderSnap
	// orderSeq is a monotonic stamp assigned to each freshly built orderSnap so the
	// cap eviction can pick the oldest-built entry. Guarded by m.mu.
	orderSeq uint64
	// orderRebuilds counts order-snapshot rebuilds — a test hook (assert warm reuse
	// / correct invalidation). Guarded by m.mu.
	orderRebuilds uint64
}

// bumpData advances dataVersion, invalidating every cached order snapshot (a later
// scroll re-tests snap.ver == dataVersion and misses). Called under m.mu (WRITE) at
// every doc-set mutation AND every payload mutation, exactly once per logical
// mutation (in the innermost *Locked body each public method funnels through; NOT in
// removeLocked, which is shared by delete + replace-Add).
func (m *MultiVectorIndex) bumpData() { m.dataVersion++ }

// nowMs returns the current wall-millis via the injectable clock (time.Now when
// m.now is nil). Mirrors hnsw / NamedCollection.nowMs.
func (m *MultiVectorIndex) nowMs() int64 {
	if m.now == nil {
		return time.Now().UnixMilli()
	}
	return m.now()
}

// SetNowFunc overrides the wall-clock source (unix millis) this index's non-apply
// expiry sites consult (sweeper + read/query filter + the wall-clock branch of the
// write paths). nil restores the real clock. TEST/advanced seam mirroring
// cache.Cache.SetNowFunc; production never calls it (byte-identical default). Takes
// the write lock.
func (m *MultiVectorIndex) SetNowFunc(fn func() int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.now = fn
}

// sweepKeyTTLOnce physically reclaims expired per-key payload TTL keys: under the
// write lock it scans every docID that has a keyTTL entry and drops keys whose
// ABSOLUTE deadline has passed (the SAME predicate the lazy liveMetaMap read path
// uses — deadline != 0 && <= now — so sweep and lazy-drop can never diverge) from
// BOTH m.keyTTL[docID] AND the per-doc payload map m.docMeta[docID], then reindexes
// the id-keyed payloadIdx so the dropped key's stale token/field postings physically
// go (mirroring dense ttl.go's per-key pass). An emptied keyTTL[docID] entry is
// deleted. If ANY key was dropped it bumps dataVersion (a sweep is a payload
// mutation, so the order_by snapshot must invalidate — matching the payload
// mutators). Returns the number of keys dropped. Logs nothing: the absolute
// deadlines are durable, so a swept key is re-derived (re-dropped lazily /
// re-swept) after a restart, never resurrected. Must be called WITHOUT holding m.mu
// (it takes the write lock for the duration). Uses the injectable clock (m.nowMs).
// An index with no per-key TTL is a cheap no-op.
func (m *MultiVectorIndex) sweepKeyTTLOnce() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := uint64(m.nowMs()) //nolint:gosec // unix-millis is non-negative
	dropped := 0
	for docID, ke := range m.keyTTL {
		if len(ke) == 0 {
			continue
		}
		meta := m.docMeta[docID]
		// Fast path: deadlines exist but none reached / none still present → no
		// allocation, no mutation.
		var expiredAny bool
		for k, dl := range ke {
			if keyExpired(uint64(dl), now) { //nolint:gosec // i64 deadlines are non-negative unix-millis
				if _, present := meta[k]; present {
					expiredAny = true
					break
				}
			}
		}
		if !expiredAny {
			continue
		}
		// Clone-on-write (mirror the payload mutators): build the survivor payload +
		// pruned deadline map without aliasing the live maps, drop the expired keys
		// with the shared predicate, then reindex the filter-first index.
		newMeta := cloneMeta(meta)
		newKe := cloneKeyTTL(ke)
		for k, dl := range newKe {
			if keyExpired(uint64(dl), now) { //nolint:gosec
				if _, present := newMeta[k]; present {
					delete(newMeta, k)
					dropped++
				}
			}
		}
		if len(newMeta) == 0 {
			newMeta = nil
		}
		newKe = pruneKeyTTL(newKe, newMeta)
		if newMeta == nil {
			delete(m.docMeta, docID)
		} else {
			m.docMeta[docID] = newMeta
		}
		if newKe == nil {
			delete(m.keyTTL, docID)
		} else {
			m.keyTTL[docID] = newKe
		}
		m.payloadIdx.reindex(docID, newMeta) // stale token/field postings physically go
	}
	if dropped > 0 {
		m.bumpData() // payload changed → invalidate the order_by snapshot
	}
	return dropped
}

// startSweeper launches the background per-key-TTL sweeper goroutine once (gated by
// sweepStart), called on the first add. Mirrors dense Collection.startSweeper.
func (m *MultiVectorIndex) startSweeper() {
	if m.suppressSweep {
		m.sweepStart.Do(func() {}) // consume the Once; sweepStop stays nil (Stop is a safe no-op)
		return
	}
	m.sweepStart.Do(func() {
		m.sweepStop = make(chan struct{})
		m.sweepDone = make(chan struct{})
		go m.runSweeper(defaultSweepInterval)
	})
}

// runSweeper ticks at interval, calling sweepKeyTTLOnce until sweepStop is closed.
func (m *MultiVectorIndex) runSweeper(interval time.Duration) {
	defer close(m.sweepDone)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-m.sweepStop:
			return
		case <-t.C:
			m.sweepKeyTTLOnce()
		}
	}
}

// Stop halts the background sweeper, joining its goroutine (no leak). Idempotent
// and safe to call even if startSweeper was never invoked (sweepStop == nil) or if
// already stopped (the channel is already closed). Mirrors dense Collection.Stop.
func (m *MultiVectorIndex) Stop() {
	if m.sweepStop == nil {
		return
	}
	select {
	case <-m.sweepStop:
		// already closed
	default:
		close(m.sweepStop)
		<-m.sweepDone
	}
}

// Release drops a reference taken by CollectionStore.AcquireMulti.
func (m *MultiVectorIndex) Release() { m.inuse.Add(-1) }

// retire waits for in-flight users to drain, then closes the index (unmapping
// any mmap files) and runs cleanup (deleting this generation's files). Called
// only after the index has been removed from the store map.
func (m *MultiVectorIndex) retire(cleanup func()) {
	for m.inuse.Load() > 0 {
		time.Sleep(200 * time.Microsecond)
	}
	m.Stop() // join the sweeper goroutine before closing (no leak; mirrors dense)
	_ = m.Close()
	if cleanup != nil {
		cleanup()
	}
}

// mvMetaPath / mvMapsPath derive the instant-restart sidecar and the doc<->token
// maps sidecar from the store-managed vectors path (which ends in ".vecs").
// Empty in non-persistent mode. They are free functions (not methods) because
// MultiVectorConfig is a data-only type in the engine-free vtypes leaf.
func mvMetaPath(cfg MultiVectorConfig) string {
	if cfg.MmapPath == "" {
		return ""
	}
	return strings.TrimSuffix(cfg.MmapPath, ".vecs") + ".meta"
}

func mvMapsPath(cfg MultiVectorConfig) string {
	if cfg.MmapPath == "" {
		return ""
	}
	return strings.TrimSuffix(cfg.MmapPath, ".vecs") + ".maps"
}

// mvInnerConfig maps a MultiVectorConfig onto the Config of the inner token-vector
// HNSW, filling HNSW defaults and the quantization / mmap backing. In Persistent
// mode the float32 vectors are mmap-backed (off-heap) and quantization defaults
// to SQ8. A free function (not a method) because MultiVectorConfig is a data-only
// type in the engine-free vtypes leaf.
func mvInnerConfig(cfg MultiVectorConfig) Config {
	m, efc, efs := cfg.M, cfg.EfConstruction, cfg.EfSearch
	if m <= 0 {
		m = 16
	}
	if efc <= 0 {
		efc = 200
	}
	if efs <= 0 {
		efs = 64
	}
	quant := cfg.Quant
	if cfg.Persistent && quant == QuantNone {
		quant = QuantSQ8 // mmap backing needs codes
	}
	c := Config{
		Dim:                   cfg.Dim,
		Metric:                Cosine,
		M:                     m,
		EfConstruction:        efc,
		EfSearch:              efs,
		Seed:                  cfg.Seed,
		Quant:                 quant,
		RescoreFactor:         cfg.RescoreFactor,
		FilterFirstRelativeBP: cfg.FilterFirstRelativeBP,
	}
	// IVF / IVF-PQ inner index. IndexHNSW (the zero value) leaves these zero and
	// the inner Config is byte-identical to before. The Config is Validated by
	// newIndex (newHNSW/newIVF both call ValidateConfig(cfg)), so a bad IVF param fails
	// loud at NewMultiVectorIndex.
	c.IndexType = cfg.IndexType
	c.IVFNlist = cfg.IVFNlist
	c.IVFNprobe = cfg.IVFNprobe
	c.IVFPQ = cfg.IVFPQ
	c.IVFPQM = cfg.IVFPQM
	c.IVFRerank = cfg.IVFRerank
	c.OPQ = cfg.OPQ
	c.OPQIters = cfg.OPQIters
	c.IVFTrainThreshold = cfg.IVFTrainThreshold
	c.IVFDriftRetrain = cfg.IVFDriftRetrain
	c.IVFDriftGrowthFactor = cfg.IVFDriftGrowthFactor
	c.IVFDriftFactor = cfg.IVFDriftFactor
	// PQDropVecs (HNSW-PQ only) folds the float-drop into the inner index's
	// incremental auto-train. Validated QuantPQ-only by the inner Config Validate
	// in newIndex (ErrInvalidPQDropVecs otherwise — e.g. set on an IVF inner index
	// or a non-PQ HNSW inner index).
	c.PQDropVecs = cfg.PQDropVecs
	if cfg.IndexType == IndexIVF {
		// IVF is snapshot-only: it rejects mmap-Persistent (ErrInvalidIVFPersistent)
		// and the MV inner index is snapshot-persisted regardless (m.idx.Snapshot;
		// the Persistent mmap sidecar is for the doc/token MAPS, not the inner
		// index). Force the inner Config off the mmap path so an IVF inner index
		// coexists with an MV-Persistent (maps mmap) collection. The maps sidecar is
		// independent and unaffected.
		c.Persistent = false
		c.QuantStorage = QuantInRAM
		c.MmapPath = ""
		c.GraphMmapPath = ""
		return c
	}
	if cfg.Persistent {
		c.QuantStorage = QuantMmap
		c.MmapPath = cfg.MmapPath
		c.GraphMmapPath = cfg.GraphMmapPath
	}
	return c
}

// newMultiShell builds an empty MultiVectorIndex around an inner VectorIndex
// (HNSW or IVF per cfg.IndexType).
func newMultiShell(cfg MultiVectorConfig, idx VectorIndex) *MultiVectorIndex {
	return &MultiVectorIndex{
		idx:        idx,
		tokenDoc:   make(map[uint64]uint64),
		docTokens:  make(map[uint64][]uint64),
		docMeta:    make(map[uint64]Metadata),
		keyTTL:     make(map[uint64]map[string]int64),
		version:    make(map[uint64]uint64),
		docSparse:  make(map[uint64]*SparseVector),
		sparseIdx:  newSparseIndexID(),
		payloadIdx: newPayloadIndexID(),
		// dataVersion starts at 1 so the zero-valued orderSnap.ver (0 = "never built")
		// forces the first scroll to rebuild even for a restored/bulk-loaded index.
		dataVersion:   1,
		orderSnaps:    make(map[orderCacheKey]*orderSnap),
		dim:           cfg.Dim,
		persistent:    cfg.Persistent,
		suppressSweep: cfg.SuppressSweep,
		metaPath:      mvMetaPath(cfg),
		mapsPath:      mvMapsPath(cfg),
		cfg:           cfg,
	}
}

// NewMultiVectorIndex builds an empty late-interaction index over Dim-dimensional
// token vectors. With cfg.Persistent the index is mmap-backed (the store fills
// the file paths) and durable via Flush; otherwise it is heap-backed/in-memory.
func NewMultiVectorIndex(cfg MultiVectorConfig) (*MultiVectorIndex, error) {
	// WAL (heap-checkpoint) and Persistent (mmap instant-restart) are mutually
	// exclusive durability modes (mirror the dense Config rule). Reject the
	// combination loudly rather than silently picking one.
	if cfg.WAL && cfg.Persistent {
		return nil, ErrInvalidWAL
	}
	// Construct the inner index via newIndex so the token index can select
	// IndexIVF (IVF / IVF-PQ — compressing the dominant MV memory cost). IndexHNSW
	// (the zero value) builds the historical inner graph index (byte/behaviour-
	// identical). newIndex validates the inner Config (newHNSW/newIVF both call
	// ValidateConfig(cfg)), so a bad inner IVF param fails loud here. mvInnerConfig forces
	// the inner Persistent=false for an IVF inner index (snapshot-only).
	idx, err := newIndex(mvInnerConfig(cfg))
	if err != nil {
		return nil, err
	}
	return newMultiShell(cfg, idx), nil
}

// Add inserts or replaces document docID with its token vectors (each of length
// Dim). Re-adding an existing id replaces its vectors and metadata. meta is
// optional. Returns ErrDimMismatch if any token has the wrong length, or
// ErrEmptyDocument if tokens is empty.
func (m *MultiVectorIndex) Add(docID uint64, tokens [][]float32, meta Metadata) error {
	_, err := m.AddCAS(docID, tokens, meta, CASCond{})
	return err
}

// AddCAS is Add with an optimistic-CAS precondition (CASCond{} = no precondition,
// an unconditional replace-add that still bumps the version). It returns the
// document's resulting version on success: 1 for a fresh docID, current+1 for an
// in-place replace. On a CAS mismatch it returns ErrVersionConflict with no
// mutation and nothing WAL-logged. The check + bump and the WAL append are
// serialized under opMu so engine + WAL agree (mirrors NamedCollection.InsertCAS).
func (m *MultiVectorIndex) AddCAS(docID uint64, tokens [][]float32, meta Metadata, cas CASCond) (uint64, error) {
	return m.AddCASKeyTTL(docID, tokens, meta, nil, cas)
}

// AddCASKeyTTL is AddCAS carrying an OPTIONAL per-key payload TTL map (key ->
// RELATIVE ms). The engine computes the resulting ABSOLUTE deadline map and the
// WAL logs it so replay restores it VERBATIM (time-stable). Empty/nil keyTTLMs is
// the zero-overhead path (no per-key deadlines, byte-identical WAL record).
func (m *MultiVectorIndex) AddCASKeyTTL(docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return m.AddCASKeyTTLSparse(docID, tokens, meta, keyTTLMs, nil, cas)
}

// AddCASKeyTTLSparse is AddCASKeyTTL carrying an OPTIONAL doc-level sparse vector.
// A non-nil/non-zero sparse is Validate()d, deep-cloned into docSparse, and indexed
// in sparseIdx (a replace drops the prior sparse first via removeLocked). A nil/zero
// sparse is the dense-only path — byte/behaviour-identical to AddCASKeyTTL (no
// docSparse entry, no sparse WAL trailer). The MaxSim/inner-HNSW path is UNTOUCHED.
func (m *MultiVectorIndex) AddCASKeyTTLSparse(docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, sparse *SparseVector, cas CASCond) (uint64, error) {
	return m.addCASKeyTTLSparseBody(docID, tokens, meta, keyTTLMs, sparse, cas, false, 0)
}

// AddCASKeyTTLSparseAt is AddCASKeyTTLSparse computing the per-key payload deadline
// against the EXPLICIT leader-stamped clock nowMs — the replicated-apply variant so
// every replica stamps byte-identical per-key deadlines (#4 vector TTL determinism).
func (m *MultiVectorIndex) AddCASKeyTTLSparseAt(docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, sparse *SparseVector, cas CASCond, nowMs int64) (uint64, error) {
	return m.addCASKeyTTLSparseBody(docID, tokens, meta, keyTTLMs, sparse, cas, true, nowMs)
}

func (m *MultiVectorIndex) addCASKeyTTLSparseBody(docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, sparse *SparseVector, cas CASCond, stamped bool, nowMs int64) (uint64, error) {
	if err := checkRecordValues(meta); err != nil {
		return 0, err
	}
	if len(tokens) == 0 {
		return 0, ErrEmptyDocument
	}
	for _, t := range tokens {
		if len(t) != m.dim {
			return 0, ErrDimMismatch
		}
	}
	if sparse != nil {
		if err := sparse.Validate(); err != nil {
			return 0, err
		}
	}
	now := nowMs
	if !stamped {
		now = m.nowMs()
	}
	m.startSweeper() // launch the per-key-TTL sweeper on first add (mirrors dense)
	if m.wal == nil {
		version, _, err := m.addLockedAt(docID, tokens, meta, keyTTLMs, sparse, cas, now)
		return version, err
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure so a panic
	// inside addLockedAt or the WAL append still unlocks opMu via the deferred
	// Unlock instead of leaking it forever — server/handlers.go recovers
	// per-request panics, so an unrecovered lock would silently deadlock every
	// later write on this collection. The closure returns before the durability
	// wait so concurrent writers overlap in commitWaitStaged and the leader-fsync
	// actually batches them — mirroring dense Collection.InsertCASKeyTTL. This
	// covers BOTH the wall-clock and stamped (...At) callers, which share this
	// body.
	version, seq, err := func() (uint64, uint64, error) {
		m.opMu.Lock()
		defer m.opMu.Unlock()
		version, keyExpires, err := m.addLockedAt(docID, tokens, meta, keyTTLMs, sparse, cas, now)
		if err != nil {
			return 0, 0, err
		}
		// Log the RESULTING absolute per-key deadlines (not nil) so replay restores
		// them verbatim (time-stable). keyTTLToU64 returns nil for an empty map
		// (cheap path). The optional doc sparse rides as a trailing block (omitted
		// when nil/zero ⇒ a dense-only record stays byte-identical).
		seq, err := m.wal.appendMVAddStaged(docID, tokens, meta, keyTTLToU64(keyExpires), version, sparse)
		return version, seq, err
	}()
	if err != nil {
		return 0, err
	}
	if err := m.wal.commitWaitStaged(seq); err != nil {
		return 0, err
	}
	return version, nil
}

// addLockedAt is addLocked computing the per-key payload deadline against the
// caller-supplied `now` (unix millis) instead of the wall clock, so a replicated MV
// add stamps byte-identical per-key deadlines on every replica (#4 vector TTL
// determinism). MV has no doc/point TTL and the token sub-index inserts carry ttl=0,
// so `now` gates only the per-key deadline.
func (m *MultiVectorIndex) addLockedAt(docID uint64, tokens [][]float32, meta Metadata, keyTTLMs map[string]int64, sparse *SparseVector, cas CASCond, now int64) (uint64, map[string]int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// CAS precondition (read current version, 0 if absent; expected==0+Has =
	// add-if-absent). Capture prevVersion BEFORE removeLocked, which drops it.
	prevVersion := m.version[docID]
	if err := cas.check(prevVersion); err != nil {
		return 0, nil, err
	}
	// A replace clears the doc's prior per-key deadlines (removeLocked deletes
	// m.keyTTL[docID]); the fresh add then sets new deadlines from keyTTLMs below.
	if _, exists := m.docTokens[docID]; exists {
		m.removeLocked(docID)
	}
	ids := make([]uint64, len(tokens))
	for i, t := range tokens {
		m.nextToken++
		tid := m.nextToken
		// hnsw normalizes on insert (Cosine); token ids are unique so no
		// duplicate-id path is hit. No-CAS on the token sub-index (the MV family's
		// per-doc version lives at the MultiVectorCollection level).
		if _, _, err := m.idx.Insert(tid, t, 0, nil, nil, nil, CASCond{}); err != nil {
			return 0, nil, err
		}
		m.tokenDoc[tid] = docID
		ids[i] = tid
	}
	m.docTokens[docID] = ids
	if len(meta) > 0 {
		m.docMeta[docID] = meta
	}
	m.payloadIdx.reindex(docID, m.docMeta[docID]) // keep the filter-first index in sync (under m.mu)
	// Optional doc-level sparse vector: store + index it (removeLocked above already
	// dropped any prior sparse on a replace). Nil/zero ⇒ the dense-only no-op.
	m.applyDocSparseLocked(docID, sparse)
	// Insert-time per-key TTL: compute absolute deadlines (now+ttl) against the fresh
	// payload, pruned to its present keys (mirrors named insertLocked / set_payload).
	// Empty/nil ⇒ no per-key TTL (removeLocked already cleared any prior map), so the
	// no-key_ttl path stays byte-identical.
	ke := pruneKeyTTL(applyKeyTTLMs(nil, keyTTLMs, meta, now), meta)
	if ke != nil {
		m.keyTTL[docID] = ke
	}
	// Bump the per-doc version (fresh 0 -> 1; in-place replace prev -> +1).
	v := prevVersion + 1
	m.version[docID] = v
	m.bumpData() // doc-set + payload change: invalidate the order_by snapshot (once; not in removeLocked)
	return v, ke, nil
}

// restoreAdd adds docID with an EXACT per-document version set VERBATIM (NOT
// bumped) — the WAL-replay analogue used so a replayed Add restores the exact
// version the original op produced (mirrors NamedCollection.RestoreInsert). It
// does NOT WAL-log. A version of 0 falls back to the normal bump (an old record
// predating the version block defaults a fresh add to 1).
func (m *MultiVectorIndex) restoreAdd(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) error {
	if err := checkRecordValues(meta); err != nil {
		return err
	}
	if len(tokens) == 0 {
		return ErrEmptyDocument
	}
	for _, t := range tokens {
		if len(t) != m.dim {
			return ErrDimMismatch
		}
	}
	m.startSweeper() // launch the per-key-TTL sweeper on first (replayed) add
	m.mu.Lock()
	defer m.mu.Unlock()
	prevVersion := m.version[docID]
	if _, exists := m.docTokens[docID]; exists {
		m.removeLocked(docID)
	}
	ids := make([]uint64, len(tokens))
	for i, t := range tokens {
		m.nextToken++
		tid := m.nextToken
		if _, _, err := m.idx.Insert(tid, t, 0, nil, nil, nil, CASCond{}); err != nil {
			return err
		}
		m.tokenDoc[tid] = docID
		ids[i] = tid
	}
	m.docTokens[docID] = ids
	if len(meta) > 0 {
		m.docMeta[docID] = meta
	}
	m.payloadIdx.reindex(docID, m.docMeta[docID]) // keep the filter-first index in sync (replay path)
	// Optional doc-level sparse vector carried VERBATIM by the WAL replay / reshard
	// copy (mirrors how keyExpires rides). Nil/zero ⇒ the dense-only no-op.
	m.applyDocSparseLocked(docID, sparse)
	// keyExpires is the ABSOLUTE per-key deadline map the WAL logged — restored
	// VERBATIM (NOT recomputed now+ttl) so a pending add-time per-key TTL survives a
	// crash time-stable. Empty ⇒ the no-key_ttl path (removeLocked already cleared
	// any prior map). The reshard copy passes nil.
	if len(keyExpires) > 0 {
		ke := make(map[string]int64, len(keyExpires))
		for k, dl := range keyExpires {
			ke[k] = int64(dl) //nolint:gosec
		}
		m.keyTTL[docID] = ke
	}
	if version == 0 {
		version = prevVersion + 1
	}
	m.version[docID] = version
	m.bumpData() // WAL-replayed doc-set + payload change: invalidate the order_by snapshot
	return nil
}

// AddIfAbsent inserts document docID with its token vectors ONLY if docID is not
// already present, reporting whether it inserted. When docID already exists it is
// a no-op (returns inserted=false) and the stored document is left untouched — it
// never clobbers a concurrent write. A previously-deleted docID counts as absent
// (the MV index has no tombstones; Delete removes the bookkeeping outright), so
// if-absent re-adds it.
//
// ATOMICITY (Race A): the existence check and the add run inside ONE write-lock
// critical section — no check-then-act gap. Combined with Raft serialization
// (vector_mv_add_if_absent is a single OpReadWrite op on the partition's log),
// when if-absent races a concurrent replace-Add on the same docID the live value
// always wins (mirror of hnsw.InsertIfAbsent). Returns ErrDimMismatch / ErrEmptyDocument
// on a malformed document, exactly as Add.
func (m *MultiVectorIndex) AddIfAbsent(docID uint64, tokens [][]float32, meta Metadata) (inserted bool, err error) {
	if err := checkRecordValues(meta); err != nil {
		return false, err
	}
	if len(tokens) == 0 {
		return false, ErrEmptyDocument
	}
	for _, t := range tokens {
		if len(t) != m.dim {
			return false, ErrDimMismatch
		}
	}
	m.startSweeper() // launch the per-key-TTL sweeper on first add (mirrors dense)
	if m.wal == nil {
		return m.addIfAbsentLocked(docID, tokens, meta, nil)
	}
	// Apply under opMu so the existence-check + add + WAL WRITE are one atomic unit
	// against a concurrent FlushMVWAL — but opMu is released BEFORE the durability
	// wait (staged commit, mirroring dense Collection.InsertCASKeyTTL) so the online
	// MV reshard copy never holds opMu across the fsync and starves live writers.
	// The closure also guarantees a panic inside the apply still unlocks opMu.
	// A no-op (docID already live) writes nothing and returns seq 0;
	// commitWaitStaged(0) is a no-op wait.
	inserted, seq, err := func() (bool, uint64, error) {
		m.opMu.Lock()
		defer m.opMu.Unlock()
		inserted, err := m.addIfAbsentLocked(docID, tokens, meta, nil)
		if err != nil || !inserted {
			return inserted, 0, err
		}
		// Log the resulting Add (only when it actually inserted; a no-op writes nothing).
		// A successful if-absent add is always a fresh doc → version 1.
		seq, err := m.wal.appendMVAddStaged(docID, tokens, meta, nil, 1, nil)
		return inserted, seq, err
	}()
	if err != nil {
		return inserted, err
	}
	return inserted, m.wal.commitWaitStaged(seq)
}

// addIfAbsentLocked performs the if-absent state change under m.mu, reporting
// whether it inserted. Split out so the WAL append happens after m.mu is released
// but still under opMu. The optional doc-level sparse is stored on a real add
// (nil/zero ⇒ the dense-only no-op).
func (m *MultiVectorIndex) addIfAbsentLocked(docID uint64, tokens [][]float32, meta Metadata, sparse *SparseVector) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.docTokens[docID]; exists {
		return false, nil // no-op: docID already live, never clobber
	}
	// Insert under the same lock (cannot call the public Add: it would re-lock).
	ids := make([]uint64, len(tokens))
	for i, t := range tokens {
		m.nextToken++
		tid := m.nextToken
		if _, _, ierr := m.idx.Insert(tid, t, 0, nil, nil, nil, CASCond{}); ierr != nil {
			return false, ierr
		}
		m.tokenDoc[tid] = docID
		ids[i] = tid
	}
	m.docTokens[docID] = ids
	if len(meta) > 0 {
		m.docMeta[docID] = meta
	}
	m.payloadIdx.reindex(docID, m.docMeta[docID]) // keep the filter-first index in sync (under m.mu)
	m.applyDocSparseLocked(docID, sparse)         // optional doc-level sparse on a real add
	// A successful if-absent add is always a fresh doc → version 1 (a re-add of a
	// previously-deleted docID counts as absent; Delete drops the version).
	m.version[docID] = 1
	m.bumpData() // doc-set change: invalidate the order_by snapshot
	return true, nil
}

// MultiAddIfAbsentVersion is AddIfAbsent that, on a REAL add, sets the document's
// per-document version VERBATIM to `version` (NOT 1). version==0 reproduces the
// AddIfAbsent semantics (fresh add → version 1). Used by the ONLINE MV reshard
// copy pass to carry a copied document's exact per-document CAS version while
// still never clobbering a concurrent live dual-write (Race A) — the MV analog of
// dense hnsw.InsertIfAbsentVersion. WAL-logs the resulting (verbatim) version, so
// replay restores it without a re-bump. keyExpires is the doc's ABSOLUTE per-key
// payload deadline map (key -> unix-millis) set VERBATIM (NOT recomputed) on a
// real add so the ONLINE MV reshard copy keeps the doc's original key deadlines
// time-stable; nil keyExpires (a plain if-absent) is byte-identical to today's behavior.
func (m *MultiVectorIndex) MultiAddIfAbsentVersion(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64) (inserted bool, err error) {
	return m.MultiAddIfAbsentVersionSparse(docID, tokens, meta, keyExpires, version, nil)
}

// MultiAddIfAbsentVersionSparse is MultiAddIfAbsentVersion carrying the doc's
// OPTIONAL sparse vector VERBATIM (for an online reshard copy). Nil/zero sparse +
// version==0 + empty keyExpires degrades to the plain AddIfAbsent path
// (byte-identical). The sparse is stored on a real add and WAL-logged verbatim.
func (m *MultiVectorIndex) MultiAddIfAbsentVersionSparse(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) (inserted bool, err error) {
	if version == 0 && len(keyExpires) == 0 && (sparse == nil || sparse.IsZero()) {
		return m.AddIfAbsent(docID, tokens, meta) // byte-for-byte the existing path (logs version 1)
	}
	if err := checkRecordValues(meta); err != nil {
		return false, err
	}
	if len(tokens) == 0 {
		return false, ErrEmptyDocument
	}
	for _, t := range tokens {
		if len(t) != m.dim {
			return false, ErrDimMismatch
		}
	}
	if sparse != nil {
		if err := sparse.Validate(); err != nil {
			return false, err
		}
	}
	m.startSweeper() // launch the per-key-TTL sweeper on first add (mirrors dense)
	if m.wal == nil {
		inserted, _, err := m.addIfAbsentVersionLocked(docID, tokens, meta, keyExpires, version, sparse)
		return inserted, err
	}
	// Apply under opMu so the existence-check + add + WAL WRITE are one atomic unit
	// against a concurrent FlushMVWAL (mirror AddIfAbsent), released BEFORE the
	// durability wait (staged commit) so the online MV reshard copy pass does not
	// hold opMu across the fsync.
	inserted, seq, err := func() (bool, uint64, error) {
		m.opMu.Lock()
		defer m.opMu.Unlock()
		inserted, stored, err := m.addIfAbsentVersionLocked(docID, tokens, meta, keyExpires, version, sparse)
		if err != nil || !inserted {
			return inserted, 0, err
		}
		// Log the resulting Add with its VERBATIM version (replay restores it via
		// restoreAdd without re-bumping; see replayMVRecord) AND the ABSOLUTE per-key
		// deadlines verbatim (nil for a plain copy) so replay restores them time-stable.
		// The optional doc sparse rides verbatim too (omitted when nil/zero).
		seq, err := m.wal.appendMVAddStaged(docID, tokens, meta, keyExpires, stored, sparse)
		return inserted, seq, err
	}()
	if err != nil {
		return inserted, err
	}
	return inserted, m.wal.commitWaitStaged(seq)
}

// addIfAbsentVersionLocked is addIfAbsentLocked that sets the version VERBATIM
// (version==0 ⇒ a fresh add → version 1) and, on a real add, the doc's ABSOLUTE
// per-key payload deadlines + optional doc sparse VERBATIM. Returns the version
// actually stored so the WAL logs it (replay restores all verbatim). Split out so
// the WAL append happens after m.mu is released but still under opMu.
func (m *MultiVectorIndex) addIfAbsentVersionLocked(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) (inserted bool, stored uint64, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.docTokens[docID]; exists {
		return false, 0, nil // no-op: docID already live, never clobber
	}
	ids := make([]uint64, len(tokens))
	for i, t := range tokens {
		m.nextToken++
		tid := m.nextToken
		if _, _, ierr := m.idx.Insert(tid, t, 0, nil, nil, nil, CASCond{}); ierr != nil {
			return false, 0, ierr
		}
		m.tokenDoc[tid] = docID
		ids[i] = tid
	}
	m.docTokens[docID] = ids
	if len(meta) > 0 {
		m.docMeta[docID] = meta
	}
	m.payloadIdx.reindex(docID, m.docMeta[docID]) // keep the filter-first index in sync (reshard copy)
	m.applyDocSparseLocked(docID, sparse)         // optional doc-level sparse VERBATIM (reshard copy)
	// keyExpires (ABSOLUTE unix-ms deadlines) set VERBATIM — NOT recomputed now+ttl —
	// so an online reshard copy keeps the doc's original key deadlines time-stable.
	if len(keyExpires) > 0 {
		ke := make(map[string]int64, len(keyExpires))
		for k, dl := range keyExpires {
			ke[k] = int64(dl) //nolint:gosec
		}
		m.keyTTL[docID] = ke
	}
	if version == 0 {
		version = 1 // fresh add → version 1 (the plain if-absent semantics)
	}
	m.version[docID] = version // VERBATIM
	m.bumpData()               // doc-set change: invalidate the order_by snapshot
	return true, version, nil
}

// MultiRestoreAdd is a verbatim-version, non-if-absent replace-add: it adds (or
// replaces) docID with the per-document version set EXACTLY to `version` (NOT
// bumped), then WAL-logs the verbatim version so replay restores it without a
// re-bump. version==0 falls back to the normal bump (a fresh add → 1), matching
// restoreAdd. Used by the OFFLINE MV resplit backfill so copied documents keep
// their per-document CAS version (the MV analog of dense EncodeVectorInsertArgsVersioned).
// keyExpires is the document's ABSOLUTE per-key payload deadline map (key ->
// unix-millis), applied VERBATIM by restoreAdd (NOT recomputed now+ttl) and
// WAL-logged so an OFFLINE MV resplit copy keeps the doc's original key deadlines
// time-stable; nil keyExpires (a plain reinsert) is byte-identical to today's behavior.
func (m *MultiVectorIndex) MultiRestoreAdd(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64) error {
	return m.MultiRestoreAddSparse(docID, tokens, meta, keyExpires, version, nil)
}

// MultiRestoreAddSparse is MultiRestoreAdd carrying the doc's OPTIONAL sparse vector
// VERBATIM (for an offline resplit backfill). Nil/zero sparse is byte-identical to
// MultiRestoreAdd. restoreAdd sets the sparse verbatim (no recompute); the WAL logs
// it verbatim too (omitted when nil/zero).
func (m *MultiVectorIndex) MultiRestoreAddSparse(docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) error {
	if len(tokens) == 0 {
		return ErrEmptyDocument
	}
	for _, t := range tokens {
		if len(t) != m.dim {
			return ErrDimMismatch
		}
	}
	if sparse != nil {
		if err := sparse.Validate(); err != nil {
			return err
		}
	}
	// keyExpires carries the doc's ABSOLUTE per-key payload deadlines (from the scan
	// trailer); restoreAdd sets them VERBATIM (NOT recomputed). nil ⇒ the no-key-TTL
	// path (byte-identical to today's reinsert). The optional sparse rides verbatim too.
	if m.wal == nil {
		return m.restoreAdd(docID, tokens, meta, keyExpires, version, sparse)
	}
	// Staged commit: opMu spans {apply + WAL WRITE} only (closure ⇒ panic-safe
	// unlock), then is released before the durability wait. The OFFLINE MV resplit
	// backfill drives this in a tight loop; holding opMu across the fsync would make
	// it the permanent fsync leader and serialize every concurrent live writer.
	seq, err := func() (uint64, error) {
		m.opMu.Lock()
		defer m.opMu.Unlock()
		if err := m.restoreAdd(docID, tokens, meta, keyExpires, version, sparse); err != nil {
			return 0, err
		}
		// Re-read the resulting version under the lock: restoreAdd bumps when version==0,
		// so the WAL must log the version restoreAdd actually stored (verbatim otherwise).
		m.mu.RLock()
		stored := m.version[docID]
		m.mu.RUnlock()
		// Log the ABSOLUTE key deadlines verbatim (nil for a plain reinsert) so a later
		// replay restores them time-stable, mirroring restoreAdd / dense RestoreInsert.
		return m.wal.appendMVAddStaged(docID, tokens, meta, keyExpires, stored, sparse)
	}()
	if err != nil {
		return err
	}
	return m.wal.commitWaitStaged(seq)
}

// MultiBulkBuild bulk-loads recs into an EMPTY index in one concurrent pass — the
// MV analogue of the dense bulk_stage+bulk_build path, used by the offline MV
// resplit copy to avoid one inner-graph insert per token. It assigns every token
// id and populates the doc bookkeeping (tokenDoc/docTokens/docMeta/payloadIdx/
// docSparse/keyTTL/version) EXACTLY as restoreAdd would (same id order ⇒ same
// per-doc result), then builds the inner token graph for ALL tokens at once via
// m.idx.BuildConcurrent (multi-core) instead of serial per-token inserts.
//
// Returns (false, nil) WITHOUT mutating when the index is non-empty — the caller
// then falls back to the per-record restore-add path (BuildConcurrent requires an
// empty index, like dense BuildStaged). Like dense BuildConcurrent it does NOT
// WAL-log: durability rides the Raft-replicated batch op that re-executes this on
// replay (mirrors the dense bulk path), so callers must drive it through an op.
func (m *MultiVectorIndex) MultiBulkBuild(recs []MultiScanRecord, workers int) (bool, error) {
	for i, r := range recs {
		if len(r.Tokens) == 0 {
			return false, ErrEmptyDocument
		}
		for _, t := range r.Tokens {
			if len(t) != m.dim {
				return false, ErrDimMismatch
			}
		}
		if r.Sparse != nil {
			if err := r.Sparse.Validate(); err != nil {
				return false, err
			}
		}
		// THIS ENTRY IS WIRE-REACHABLE and it is NOT a replay body, so it runs the
		// record ingest gate. ops/mv_batch.go decodes a batch straight off the wire
		// and calls it as the FAST PATH whenever the target index is empty,
		// falling through to the gated MultiRestoreAddSparse only once documents
		// exist — so without this the same op is gated or ungated depending on the
		// target's emptiness, and the ungated case (a fresh partition) is exactly
		// the one an offline MV resplit drives. The loop below writes r.Metadata
		// into docMeta and reindexes it directly, with nothing else in front.
		//
		// The WHOLE batch is validated before m.mu is taken, so one bad record
		// refuses every record rather than leaving a half-built index that can
		// never use the fast path again. The offending row is named, exactly as
		// checkRecordValuesAll names it for the dense bulk path.
		if err := checkRecordValues(r.Metadata); err != nil {
			return false, fmt.Errorf("payload %d: %w", i, err)
		}
	}
	m.startSweeper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.docTokens) != 0 {
		return false, nil // non-empty: caller uses the incremental restore-add path
	}
	allTids := make([]uint64, 0, len(recs))
	allVecs := make([][]float32, 0, len(recs))
	for _, r := range recs {
		ids := make([]uint64, len(r.Tokens))
		for i, t := range r.Tokens {
			m.nextToken++
			tid := m.nextToken
			m.tokenDoc[tid] = r.ID
			ids[i] = tid
			allTids = append(allTids, tid)
			allVecs = append(allVecs, t)
		}
		m.docTokens[r.ID] = ids
		if len(r.Metadata) > 0 {
			m.docMeta[r.ID] = r.Metadata
		}
		m.payloadIdx.reindex(r.ID, m.docMeta[r.ID])
		m.applyDocSparseLocked(r.ID, r.Sparse)
		if len(r.KeyExpires) > 0 {
			ke := make(map[string]int64, len(r.KeyExpires))
			for k, dl := range r.KeyExpires {
				ke[k] = int64(dl) //nolint:gosec // absolute unix-ms deadline
			}
			m.keyTTL[r.ID] = ke
		}
		version := r.Version
		if version == 0 {
			version = 1 // fresh index ⇒ prevVersion 0; mirrors restoreAdd's bump
		}
		m.version[r.ID] = version
	}
	// One concurrent build of the whole token set (vs serial per-token Insert).
	if err := m.idx.BuildConcurrent(allTids, allVecs, workers); err != nil {
		return false, err
	}
	m.bumpData()
	return true, nil
}

// Exists reports whether docID is currently present (live) in the index. The MV
// index has no tombstones or TTL — a docID is live iff it has a token-set entry —
// so this is an O(1) map lookup under the read lock. The cheap liveness probe the
// online-copy resurrection guard uses to re-check the source generation (Race B).
func (m *MultiVectorIndex) Exists(docID uint64) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.docTokens[docID]
	return ok
}

// Delete removes document docID and all its token vectors. Returns whether it
// was present.
func (m *MultiVectorIndex) Delete(docID uint64) bool {
	removed, _, _ := m.DeleteCAS(docID, CASCond{})
	return removed
}

// DeleteCAS is Delete with an optimistic-CAS precondition (CASCond{} = no
// precondition). On a mismatch it returns ErrVersionConflict and removed=false
// with no mutation. A no-CAS Delete drops the document and its version (an absent
// doc reads version 0; a fresh re-add restarts at 1). The check is atomic under
// m.mu (mirrors NamedCollection.DeleteCAS).
func (m *MultiVectorIndex) DeleteCAS(docID uint64, cas CASCond) (removed bool, prevVersion uint64, err error) {
	if m.wal == nil {
		return m.deleteLocked(docID, cas)
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
	// AddCASKeyTTLSparse) so a panic inside deleteLocked or the WAL append still
	// unlocks opMu via the deferred Unlock. The durability wait runs OUTSIDE
	// opMu so a deleting writer never becomes the fsync leader while holding the
	// index's op lock — concurrent deletes group-commit like adds.
	var seq uint64
	err = func() error {
		m.opMu.Lock()
		defer m.opMu.Unlock()
		var derr error
		removed, prevVersion, derr = m.deleteLocked(docID, cas)
		if derr != nil {
			return derr
		}
		if removed {
			// Propagate the WAL append error (mirrors dense DeleteCAS): idempotent
			// replay tolerates torn duplicates, but a durability failure must fail
			// the delete rather than falsely ack a doc that may resurrect on crash.
			var serr error
			seq, serr = m.wal.appendMVDeleteStaged(docID)
			if serr != nil {
				return serr
			}
		}
		return nil
	}()
	if err != nil {
		return false, 0, err
	}
	// seq == 0 (no-op delete) makes commitWaitStaged a no-op returning nil; a real
	// fsync failure propagates and fails the delete.
	if werr := m.wal.commitWaitStaged(seq); werr != nil {
		return false, 0, werr
	}
	return removed, prevVersion, nil
}

// deleteLocked removes docID + its token nodes under m.mu, returning whether it
// was present and its prior version. The CAS check (against the current version,
// 0 if absent) is atomic with the removal. Split out so the WAL append runs after
// m.mu is released but still under opMu.
func (m *MultiVectorIndex) deleteLocked(docID uint64, cas CASCond) (bool, uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	prev := m.version[docID]
	if err := cas.check(prev); err != nil {
		return false, 0, err // CAS mismatch: no mutation
	}
	if _, exists := m.docTokens[docID]; !exists {
		return false, 0, nil
	}
	m.removeLocked(docID)
	m.bumpData() // live doc set shrank: invalidate the order_by snapshot (removeLocked does NOT bump)
	return true, prev, nil
}

// removeLocked drops a document's token nodes and bookkeeping. Caller holds mu.
func (m *MultiVectorIndex) removeLocked(docID uint64) {
	for _, tid := range m.docTokens[docID] {
		_, _ = m.idx.Delete(tid, CASCond{})
		delete(m.tokenDoc, tid)
	}
	delete(m.docTokens, docID)
	delete(m.docMeta, docID)
	m.payloadIdx.reindex(docID, nil) // nil meta = pure removal from the filter-first index
	// Drop the doc's optional sparse vector + its inverted-index postings (a
	// replace re-adds via the fresh sparse below; a delete leaves it gone). A doc
	// with no sparse vector has no entry here (the dense-only fast path is a no-op:
	// delete-of-absent + remove-of-absent both no-op).
	delete(m.docSparse, docID)
	m.sparseIdx.remove(docID)
	delete(m.keyTTL, docID)  // a removed/replaced doc carries no per-key deadlines
	delete(m.version, docID) // ...and no CAS version (a re-add restarts at 1)
}

// applyDocSparseLocked stores an OPTIONAL doc-level sparse vector for docID and
// indexes it in sparseIdx. Caller holds m.mu (write) and MUST have already run
// removeLocked for a replace (so the prior sparse postings are gone). A nil/zero
// sparse is the dense-only no-op (no docSparse entry, no posting — byte-identical
// to the pre-sparse behavior). The caller is responsible for having validated the
// sparse vector. The stored copy is deep-cloned so it never aliases caller slices.
func (m *MultiVectorIndex) applyDocSparseLocked(docID uint64, sparse *SparseVector) {
	if sparse == nil || sparse.IsZero() {
		return
	}
	sv := cloneSparse(sparse)
	m.docSparse[docID] = sv
	m.sparseIdx.add(docID, sv)
}

// Get retrieves a live document by id: its token matrix (each row a DEEP-COPIED
// normalized token vector) and its metadata (a map-level copy — see cloneMeta
// and vtypes.Metadata for the slice-ownership contract), plus ok. ok is false
// when docID is absent — the MV index has no tombstones or TTL, so a docID is live
// iff it has a token-set entry (mirror Exists/ScanDocuments). The returned
// tokens are owned by the caller, and so is the payload MAP (adding or removing
// keys never corrupts docMeta); a slice-backed Value inside it still points at
// stored bytes and must not be written through. Lock order: m.mu (read) outer, then the inner index's own mu
// (taken internally by vecsForIDs) — the same outer→inner order Search/ScanDocuments use.
func (m *MultiVectorIndex) Get(docID uint64) (tokens [][]float32, payload Metadata, version uint64, ok bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	tokenIDs, exists := m.docTokens[docID]
	if !exists {
		return nil, nil, 0, false
	}
	version = m.version[docID]
	// Resolve the doc's token vectors via the interface: vecsForIDs locks the
	// inner index's own mu internally and returns COPIES (exact arena floats when
	// present, reconstructed from the PQ code + centroid for an IVF-PQ inner index
	// with dropped floats — approximate). An absent token id is omitted. Iterate
	// tokenIDs (not the map) to preserve token order.
	vm := m.idx.vecsForIDs(tokenIDs)
	tokens = make([][]float32, 0, len(tokenIDs))
	for _, tid := range tokenIDs {
		if v, ok := vm[tid]; ok {
			tokens = append(tokens, v) // already a fresh copy owned by us
		}
	}
	// Drop per-key-TTL-expired keys, then copy the map so the caller owns it
	// (liveMetaMap may alias m.docMeta on the no-expiry fast path). Slice-backed
	// Values inside still point at stored bytes — vtypes.Metadata's contract.
	payload = cloneMeta(liveMetaMap(m.docMeta[docID], m.keyTTL[docID], m.nowMs()))
	return tokens, payload, version, true
}

// GetSparse returns a DEEP COPY of docID's optional doc-level sparse vector, or
// (nil, false) when the doc is absent or carries no sparse vector. The returned
// vector is owned by the caller (mutating it never corrupts docSparse). A pure read
// under the read lock. The sparse-lane analogue of Get's payload return.
func (m *MultiVectorIndex) GetSparse(docID uint64) (*SparseVector, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	sv, ok := m.docSparse[docID]
	if !ok || sv == nil {
		return nil, false
	}
	return cloneSparse(sv), true
}

// SearchSparse runs the doc-level sparse lane: a dot-product top-k over the doc
// sparse vectors via the id-keyed inverted index, gated by the live admit predicate
// (per-key TTL is irrelevant to the sparse vector itself; the doc is live iff it has
// a token-set entry, mirroring Exists). Returns the k highest-scoring docs
// descending. This is the engine surface the MVHybrid sparse lane reuses; the
// MaxSim lane reuses Search untouched. nil query / k<=0 ⇒ nil.
func (m *MultiVectorIndex) SearchSparse(query *SparseVector, k int) []Result {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.sparseIdx.searchTopK(query, k, func(id uint64) bool {
		_, live := m.docTokens[id]
		return live
	})
}

// sparseAdmitLocked builds the admit predicate the sparse lane shares with the
// MaxSim lane: a doc is admitted iff it is live (has a token-set entry) AND, when a
// filter is present, its per-key-TTL-live payload matches. The MaxSim lane (Search)
// applies the SAME live + per-key-TTL + filter gate, so a filtered/expired doc is
// excluded from BOTH lanes consistently. pred==nil (no filter) admits every live
// doc — byte-identical to SearchSparse. now is one clock snapshot shared with the
// MaxSim lane's Stage-2 (passed in so both lanes see one clock). Caller holds m.mu.
func (m *MultiVectorIndex) sparseAdmitLocked(pred Predicate, now int64) func(id uint64) bool {
	return func(id uint64) bool {
		if _, live := m.docTokens[id]; !live {
			return false
		}
		if pred == nil {
			return true
		}
		return pred(liveMetaMap(m.docMeta[id], m.keyTTL[id], now))
	}
}

// mvHybridK returns the per-lane candidate-pool size from a HybridOpts knob
// (DenseK / SparseK), defaulting to max(k, 50) — the same lane sizing the named
// hybrid (namedHybridK) and the dense oracle use, so the MV hybrid's lanes match
// the hand-fused ground truth.
func mvHybridK(knob, k int) int {
	if knob > 0 {
		return knob
	}
	if k < 50 {
		return 50
	}
	return k
}

// mvHybridLanesLocked builds the MaxSim (dense) and sparse candidate lanes for the
// MV cross-modality hybrid, UNFUSED, under m.mu (read). BOTH lanes are score-desc
// (the MaxSim lane is a DESCENDING relevance score, NOT an ascending distance —
// unlike the dense hnsw lane). Each lane applies the SAME live / per-key-TTL /
// filter admit gate against ONE clock snapshot (the MaxSim lane via
// maxSimSearchLocked's Stage-2; the sparse lane via sparseAdmitLocked → searchTopK),
// so a filtered/expired doc is excluded from both. An empty query token matrix
// yields a nil MaxSim lane (sparse-only); an empty/absent sparse query yields a nil
// sparse lane (MaxSim-only) — the caller collapses the single-lane case. Each lane
// is pooled to DenseK / SparseK (default max(k,50)). Caller holds m.mu (read). The
// MV analogue of namedHybridLanesLocked.
func (m *MultiVectorIndex) mvHybridLanesLocked(query [][]float32, sparseQ *SparseVector, k int, opts HybridOpts) (denseRes, sparseRes []Result, err error) {
	if k <= 0 {
		return nil, nil, nil
	}
	for _, q := range query {
		if len(q) != m.dim {
			return nil, nil, ErrDimMismatch
		}
	}
	if sparseQ != nil {
		if verr := sparseQ.Validate(); verr != nil {
			return nil, nil, verr
		}
	}
	pred, cerr := CompileFilter(opts.Filter)
	if cerr != nil {
		return nil, nil, cerr
	}
	now := m.nowMs() // one clock snapshot shared by BOTH lanes' TTL view

	// MaxSim lane: the existing Stage-1/Stage-2 MaxSim body (untouched), pooled to
	// denseK, converted to score-desc []Result. Skipped (nil) for an empty query.
	if len(query) > 0 {
		denseK := mvHybridK(opts.DenseK, k)
		// candidatesPerToken 0 = the default adaptive over-fetch (HybridOpts carries
		// no per-token knob; the MaxSim lane uses its standard widening).
		mr, serr := m.maxSimSearchLockedNow(query, denseK, opts.Filter, pred, 0, now)
		if serr != nil {
			return nil, nil, serr
		}
		denseRes = make([]Result, len(mr))
		for i, r := range mr {
			// MaxSim Score rides Result.Score (the lane is score-desc); Distance 0.
			denseRes[i] = Result{ID: r.ID, Score: r.Score}
		}
	}

	// Sparse lane: the id-keyed inverted index top-k, gated by the SAME live /
	// per-key-TTL / filter admit rule the MaxSim lane used (one clock snapshot).
	// Skipped (nil) for an empty query.
	if sparseQ != nil && !sparseQ.IsZero() {
		sparseK := mvHybridK(opts.SparseK, k)
		sparseRes = m.sparseIdx.searchTopK(sparseQ, sparseK, m.sparseAdmitLocked(pred, now))
	}
	return denseRes, sparseRes, nil
}

// MVHybridLanes builds the MaxSim and sparse candidate lanes for the MV hybrid
// WITHOUT fusing them, so a cross-partition coordinator can union the per-partition
// lanes and fuse ONCE globally (exact partition fan-out, mirror NamedHybridLanes).
// BOTH lanes are descending by Score. Lane sizing (DenseK/SparseK) + the shared
// admit gate are identical to MVHybrid (they share mvHybridLanesLocked); only the
// final fusion is deferred to the coordinator.
func (m *MultiVectorIndex) MVHybridLanes(query [][]float32, sparseQ *SparseVector, k int, opts HybridOpts) (dense, sparse []Result, err error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.mvHybridLanesLocked(query, sparseQ, k, opts)
}

// MVHybrid fuses the MaxSim (late-interaction dense) lane and the doc-level sparse
// lane into the top-k. The MaxSim lane reuses the existing Search Stage-1/Stage-2
// body (UNTOUCHED) — a descending relevance score; the sparse lane is a
// dot-product top-k over the per-doc sparse vectors; BOTH under the SAME live /
// per-key-TTL / filter admit gate (one clock snapshot, one lock acquisition for a
// consistent view). They are combined with the reused fusion (FuseScoreLanes —
// both lanes are SCORE-desc so the weighted path normalizes both as scores instead
// of inverting the MaxSim lane the way Fuse's dense slot would). Single-lane
// degradation mirrors NamedHybrid: an empty/absent sparse query returns the MaxSim
// lane alone; an empty query token matrix returns the sparse lane alone (each
// truncated to k). The MV cross-modality analogue of NamedHybrid.
func (m *MultiVectorIndex) MVHybrid(query [][]float32, sparseQ *SparseVector, k int, opts HybridOpts) ([]Result, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	denseRes, sparseRes, err := m.mvHybridLanesLocked(query, sparseQ, k, opts)
	if err != nil {
		return nil, err
	}
	// Single-lane degradation (mirror NamedHybrid). A lane with no query is empty;
	// return the other lane truncated to k instead of fusing.
	sparseEmpty := sparseQ == nil || sparseQ.IsZero()
	switch {
	case sparseEmpty:
		if len(denseRes) > k {
			denseRes = denseRes[:k]
		}
		return denseRes, nil
	case len(query) == 0:
		if len(sparseRes) > k {
			sparseRes = sparseRes[:k]
		}
		return sparseRes, nil
	}
	// BOTH lanes are score-desc → FuseScoreLanes (RRF rank-only; weighted
	// normalizes both as scores, no MaxSim inversion).
	return FuseScoreLanes(denseRes, sparseRes, opts.Method, opts.Alpha, opts.RRFK, k), nil
}

// SetPayload MERGES patch into docID's payload (patch keys overwrite or add, other
// keys retained) and stores the result in m.docMeta[docID]. The MV family stores
// payload only in the outer docMeta map (token nodes carry nil metadata, the inner
// payloadIdx is always empty) and has NO WAL (snapshot-durable by design) — this
// is a pure map update under the write lock, with NO reindex. Returns
// ErrIDNotFound for an absent document. Does not change any token vector.
func (m *MultiVectorIndex) SetPayload(docID uint64, patch Metadata, keyTTLMs map[string]int64) error {
	_, err := m.SetPayloadCAS(docID, patch, keyTTLMs, CASCond{})
	return err
}

// SetPayloadCAS is SetPayload with an optimistic-CAS precondition. Returns the
// resulting version; ErrVersionConflict (no mutation) on a mismatch; ErrIDNotFound
// for an absent document.
func (m *MultiVectorIndex) SetPayloadCAS(docID uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return m.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return m.setPayloadLocked(docID, patch, keyTTLMs, cas)
	}, docID)
}

// SetPayloadCASAt is SetPayloadCAS computing the per-key payload deadline against the
// EXPLICIT leader-stamped clock nowMs (#4 vector TTL determinism).
func (m *MultiVectorIndex) SetPayloadCASAt(docID uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return m.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return m.setPayloadLockedAt(docID, patch, keyTTLMs, cas, nowMs)
	}, docID)
}

func (m *MultiVectorIndex) setPayloadLocked(docID uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	return m.setPayloadLockedAt(docID, patch, keyTTLMs, cas, m.nowMs())
}

func (m *MultiVectorIndex) setPayloadLockedAt(docID uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, now int64) (Metadata, map[string]int64, uint64, error) {
	if err := checkRecordValues(patch); err != nil {
		return nil, nil, 0, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.docTokens[docID]; !exists {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(m.version[docID]); err != nil {
		return nil, nil, 0, err
	}
	merged := cloneMeta(m.docMeta[docID])
	if len(patch) > 0 {
		if merged == nil {
			merged = make(Metadata, len(patch))
		}
		for k, v := range patch {
			merged[k] = v
		}
	}
	// Per-key TTL: clone-on-write the deadline map, set/clear deadlines for the
	// patched keys (relative ms -> absolute now+ttl), then prune deadlines for keys
	// no longer in the payload. A nil result stores nil (the cheap path).
	ke := applyKeyTTLMs(cloneKeyTTL(m.keyTTL[docID]), keyTTLMs, merged, now)
	ke = pruneKeyTTL(ke, merged)
	m.docMeta[docID] = merged
	m.payloadIdx.reindex(docID, merged) // resulting payload may add/remove indexed fields
	if ke == nil {
		delete(m.keyTTL, docID)
	} else {
		m.keyTTL[docID] = ke
	}
	v := m.version[docID] + 1
	m.version[docID] = v
	m.bumpData() // payload-value change: invalidate the order_by snapshot
	return merged, ke, v, nil
}

// logPayloadOp runs a payload mutator (apply, which captures the RESULTING full
// payload + absolute per-key deadlines under m.mu) and, on a WAL-mode index,
// appends ONE op-agnostic mvSetPayload record with that resulting state — the
// dense/named resulting-payload collapse (set/overwrite/delete-keys/clear all
// reduce to "this is the new payload"). The whole apply+append runs under opMu so
// a concurrent FlushMVWAL can't interleave. The deadlines are logged VERBATIM
// (absolute unix-ms) so replay is time-stable.
func (m *MultiVectorIndex) logPayloadOp(apply func() (Metadata, map[string]int64, uint64, error), docID uint64) (uint64, error) {
	if m.wal == nil {
		_, _, version, err := apply()
		return version, err
	}
	// {apply + WAL WRITE} under opMu in a panic-safe closure; the durability wait
	// runs outside so concurrent payload writers group-commit instead of each
	// paying a serialized fsync under the index's op lock.
	var seq, version uint64
	err := func() error {
		m.opMu.Lock()
		defer m.opMu.Unlock()
		meta, ke, v, err := apply()
		if err != nil {
			return err
		}
		version = v
		seq, err = m.wal.appendMVSetPayloadStaged(docID, meta, keyTTLToU64(ke), v)
		return err
	}()
	if err != nil {
		// Mirrors the pre-split return: the applied version rides along even when
		// only the WAL write failed.
		return version, err
	}
	return version, m.wal.commitWaitStaged(seq)
}

// rebuildPayloadIdx discards and reconstructs the filter-first index from the
// live per-doc payload (m.docMeta). The rebuild-on-load entry point used after a
// sidecar/snapshot restore and after WAL replay (mirrors named's
// rebuildPayloadIdx). Takes m.mu so it is safe to call after the load/replay path
// has released its per-op locks.
func (m *MultiVectorIndex) rebuildPayloadIdx() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.payloadIdx == nil {
		m.payloadIdx = newPayloadIndexID()
	}
	m.payloadIdx.rebuild(m.docMeta)
}

// rebuildSparseIdx discards and reconstructs the doc-level sparse inverted index
// from the live per-doc sparse vectors (m.docSparse). The rebuild-on-load entry
// point used after a sidecar/snapshot restore and after WAL replay (the sparse
// analogue of rebuildPayloadIdx — restore/replay populate docSparse, this rebuilds
// the search structure). Takes m.mu so it is safe to call after the load/replay
// path has released its per-op locks. A dense-only index rebuilds an empty index.
func (m *MultiVectorIndex) rebuildSparseIdx() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sparseIdx == nil {
		m.sparseIdx = newSparseIndexID()
	}
	m.sparseIdx.rebuild(m.docSparse)
}

// restorePayload sets docID's payload to meta and its per-key deadlines to
// keyExpires VERBATIM (absolute unix-ms), with NO recompute — the WAL-replay
// analogue of named restorePayload. SetPayload recomputes relative TTL to now+ttl;
// replay must instead restore the exact absolute deadlines the original op
// produced, so a pending per-key TTL survives a crash time-stable. Idempotent on
// top of the snapshot checkpoint; it does NOT gate on liveness (replay re-applies a
// payload op that the snapshot may or may not already reflect).
func (m *MultiVectorIndex) restorePayload(docID uint64, meta Metadata, keyExpires map[string]uint64, version uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.docMeta[docID] = meta
	m.payloadIdx.reindex(docID, meta) // keep the filter-first index in sync (replay path)
	m.bumpData()                      // WAL-replayed payload-value change: invalidate the order_by snapshot
	// Restore the version VERBATIM (the WAL logged the resulting version). A 0
	// version (an old record predating the version block) leaves the existing
	// version untouched, so a prior Add's version is not clobbered.
	if version != 0 {
		m.version[docID] = version
	}
	if len(keyExpires) == 0 {
		delete(m.keyTTL, docID)
		return
	}
	ke := make(map[string]int64, len(keyExpires))
	for k, dl := range keyExpires {
		ke[k] = int64(dl) //nolint:gosec
	}
	m.keyTTL[docID] = ke
}

// OverwritePayload REPLACES docID's entire payload with meta (a nil/empty meta
// clears it). Pure map update — no reindex; logs to the WAL (resulting payload) in WAL mode. Returns ErrIDNotFound for an
// absent document. Does not change any token vector.
func (m *MultiVectorIndex) OverwritePayload(docID uint64, meta Metadata, keyTTLMs map[string]int64) error {
	_, err := m.OverwritePayloadCAS(docID, meta, keyTTLMs, CASCond{})
	return err
}

// OverwritePayloadCAS is OverwritePayload with an optimistic-CAS precondition.
// Returns the resulting version; ErrVersionConflict (no mutation) on a mismatch.
func (m *MultiVectorIndex) OverwritePayloadCAS(docID uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return m.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return m.overwritePayloadLocked(docID, meta, keyTTLMs, cas)
	}, docID)
}

// OverwritePayloadCASAt is OverwritePayloadCAS computing the per-key payload deadline
// against the EXPLICIT leader-stamped clock nowMs (#4 vector TTL determinism).
func (m *MultiVectorIndex) OverwritePayloadCASAt(docID uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return m.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return m.overwritePayloadLockedAt(docID, meta, keyTTLMs, cas, nowMs)
	}, docID)
}

func (m *MultiVectorIndex) overwritePayloadLocked(docID uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	return m.overwritePayloadLockedAt(docID, meta, keyTTLMs, cas, m.nowMs())
}

func (m *MultiVectorIndex) overwritePayloadLockedAt(docID uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, now int64) (Metadata, map[string]int64, uint64, error) {
	if err := checkRecordValues(meta); err != nil {
		return nil, nil, 0, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.docTokens[docID]; !exists {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(m.version[docID]); err != nil {
		return nil, nil, 0, err
	}
	newMeta := cloneMeta(meta)
	// Overwrite REPLACES the per-key deadline set: deadlines come only from the new
	// keyTTLMs (recomputed against now); any prior deadline is dropped.
	ke := pruneKeyTTL(applyKeyTTLMs(nil, keyTTLMs, newMeta, now), newMeta)
	m.docMeta[docID] = newMeta
	m.payloadIdx.reindex(docID, newMeta) // overwrite replaces the indexed fields wholesale
	if ke == nil {
		delete(m.keyTTL, docID)
	} else {
		m.keyTTL[docID] = ke
	}
	v := m.version[docID] + 1
	m.version[docID] = v
	m.bumpData() // payload-value change: invalidate the order_by snapshot
	return newMeta, ke, v, nil
}

// DeletePayloadKeys removes the listed keys from docID's payload (absent keys =
// no-op). Pure map update — no reindex; logs to the WAL (resulting payload) in WAL mode. Returns ErrIDNotFound for an
// absent document. Does not change any token vector.
func (m *MultiVectorIndex) DeletePayloadKeys(docID uint64, keys []string) error {
	_, err := m.DeletePayloadKeysCAS(docID, keys, CASCond{})
	return err
}

// DeletePayloadKeysCAS is DeletePayloadKeys with an optimistic-CAS precondition.
// Returns the resulting version; ErrVersionConflict (no mutation) on a mismatch.
func (m *MultiVectorIndex) DeletePayloadKeysCAS(docID uint64, keys []string, cas CASCond) (uint64, error) {
	return m.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return m.deletePayloadKeysLocked(docID, keys, cas)
	}, docID)
}

func (m *MultiVectorIndex) deletePayloadKeysLocked(docID uint64, keys []string, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.docTokens[docID]; !exists {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(m.version[docID]); err != nil {
		return nil, nil, 0, err
	}
	newMeta := cloneMeta(m.docMeta[docID])
	ke := cloneKeyTTL(m.keyTTL[docID])
	for _, k := range keys {
		delete(newMeta, k)
		if ke != nil {
			delete(ke, k) // a deleted key carries no deadline
		}
	}
	if len(newMeta) == 0 {
		newMeta = nil
	}
	ke = pruneKeyTTL(ke, newMeta)
	m.docMeta[docID] = newMeta
	m.payloadIdx.reindex(docID, newMeta) // dropped keys may remove indexed fields
	if ke == nil {
		delete(m.keyTTL, docID)
	} else {
		m.keyTTL[docID] = ke
	}
	v := m.version[docID] + 1
	m.version[docID] = v
	m.bumpData() // payload-value change: invalidate the order_by snapshot
	return newMeta, ke, v, nil
}

// ClearPayload removes ALL of docID's payload (payload → nil). Pure map update —
// no reindex; logs to the WAL (resulting payload) in WAL mode. Returns ErrIDNotFound for an absent document. Does not
// change any token vector.
func (m *MultiVectorIndex) ClearPayload(docID uint64) error {
	_, err := m.ClearPayloadCAS(docID, CASCond{})
	return err
}

// ClearPayloadCAS is ClearPayload with an optimistic-CAS precondition. Returns
// the resulting version; ErrVersionConflict (no mutation) on a mismatch.
func (m *MultiVectorIndex) ClearPayloadCAS(docID uint64, cas CASCond) (uint64, error) {
	return m.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return m.clearPayloadLocked(docID, cas)
	}, docID)
}

func (m *MultiVectorIndex) clearPayloadLocked(docID uint64, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.docTokens[docID]; !exists {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(m.version[docID]); err != nil {
		return nil, nil, 0, err
	}
	m.docMeta[docID] = nil
	m.payloadIdx.reindex(docID, nil) // cleared payload carries no indexed fields
	delete(m.keyTTL, docID)          // clearing the payload clears all per-key deadlines
	v := m.version[docID] + 1
	m.version[docID] = v
	m.bumpData() // payload-value change: invalidate the order_by snapshot
	return nil, nil, v, nil
}

// mvFilterFirstCands tries the index-accelerated filter-first plan for a filtered
// MaxSim search. It returns (candidateDocIDs, true) — a SUPERSET of the filter's
// matching docs to brute-force the exact MaxSim rerank over — when the path
// applies, or (nil, false) to signal the caller to fall back to the token-HNSW
// Stage-1 gather + Stage-2 post-filter path VERBATIM. Caller holds m.mu (read).
//
// It mirrors the named/dense planner gate: consult the id-keyed payload index for
// a candidate superset; apply the path only when the filter is index-narrowable
// (ok) AND the candidate set is small enough (<= threshold) AND the cost model
// prefers the brute-force rerank over the token-HNSW gather (preferFilterFirst on
// the inner token index). Otherwise fall back (zero regression for no-filter /
// non-accelerable / non-selective filters — candidates returns ok=false or
// too-many ids, or preferFilterFirst says the graph gather is cheaper). The
// Stage-2 live-meta predicate RE-CHECK in Search rejects any over-cover, so
// correctness never depends on the superset being exact.
func (m *MultiVectorIndex) mvFilterFirstCands(pred Predicate, filter Filter, k int) ([]uint64, bool) {
	if pred == nil || m.payloadIdx == nil {
		return nil, false // no filter -> existing token-HNSW path unchanged
	}
	// liveCount is the live DOCUMENT count (m.payloadIdx is document-keyed); the
	// inner m.idx holds TOKEN vectors so its arena size would over-count. Caller
	// holds m.mu (read), so reading m.docTokens is safe.
	threshold := m.idx.effectiveFilterFirstLimit(len(m.docTokens))
	// maxCand is the largest candidate count the planner would still choose the
	// brute-force rerank for; candidatesCapped abandons a superset that grows past
	// it rather than finishing a materialization the next line would discard. Same
	// decision as the old materialize-then-check (see the crossover tests).
	maxCand := m.idx.filterFirstCrossover(k, threshold)
	cands, ok := m.payloadIdx.candidatesCapped(filter, threshold, maxCand)
	if !ok {
		return nil, false // not narrowable / non-selective / graph cheaper -> fall back
	}
	return cands, true
}

// Search returns the top-k documents for the multi-vector query, ranked by
// descending MaxSim. query is the list of query token vectors (each length Dim).
// k <= 0 returns nil; an empty query returns nil.
func (m *MultiVectorIndex) Search(query [][]float32, k int, opts MultiSearchOpts) ([]MultiResult, error) {
	if k <= 0 || len(query) == 0 {
		return nil, nil
	}
	for _, q := range query {
		if len(q) != m.dim {
			return nil, ErrDimMismatch
		}
	}
	// Compile the payload filter once. Fail loud on a malformed filter (bad
	// regex/datetime), mirroring how dense surfaces it (hnsw.go:1062). A zero
	// filter compiles to a nil predicate: the hot path below stays
	// byte/behaviour-identical to no-filter (pred == nil gates both the
	// candidate-budget widen and the per-candidate check).
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.maxSimSearchLocked(query, k, opts.Filter, pred, opts.CandidatesPerToken)
}

// maxSimSearchLocked is the Stage-1/Stage-2 MaxSim core of Search, factored out so
// the MV hybrid (MVHybridLanes) can run the MaxSim lane under the SAME single
// m.mu.RLock acquisition that builds the sparse lane (one consistent snapshot). The
// caller holds m.mu (read) and has already validated query dims and compiled pred.
// query is the RAW (un-normalized) query token matrix; this normalizes it once,
// exactly as Search did inline. Behaviour is byte-identical to the former inline
// body (the MaxSim/inner-HNSW/arena path is untouched). Returns score-desc results.
func (m *MultiVectorIndex) maxSimSearchLocked(query [][]float32, k int, filter Filter, pred Predicate, candidatesPerToken int) ([]MultiResult, error) {
	return m.maxSimSearchLockedNow(query, k, filter, pred, candidatesPerToken, m.nowMs())
}

// maxSimSearchLockedNow is maxSimSearchLocked with an injected clock snapshot, so
// the MV hybrid can share ONE now() across the MaxSim and sparse lanes' per-key-TTL
// views (consistent expiry). nowMs is the unix-ms snapshot used for the Stage-2
// liveMetaMap. Caller holds m.mu (read).
func (m *MultiVectorIndex) maxSimSearchLockedNow(query [][]float32, k int, filter Filter, pred Predicate, candidatesPerToken int, nowMs int64) ([]MultiResult, error) {
	candPerToken := candidatesPerToken
	if candPerToken <= 0 {
		if candPerToken = 4 * k; candPerToken < 50 {
			candPerToken = 50
		}
	}
	// Adaptive over-fetch: a selective filter can leave < k candidates after the
	// Stage-2 post-filter, so widen the Stage-1 budget when a predicate is active.
	//
	// Mirrors dense's ef-widening (hnsw.go:1227-1235): dense's baseline ef is
	// max(EfSearch, k) (a ~k-scale pool), and a predicate DOUBLES it to a 2*k
	// floor, capped at MaxEfSearch (default 1024). The proportional MV analog is a
	// DOUBLING of MV's own baseline — which is already max(4*k, 50), since Stage-1
	// over-fetches per-token before the doc union shrinks the pool. A literal 2*k
	// floor (dense's constant) would be a no-op here: it never exceeds the 4*k
	// baseline. So we floor the per-token budget at 2*baseline = max(8*k, 100),
	// capped at defaultMaxEfSearch (1024) — the same cap dense uses. pred == nil
	// leaves candPerToken EXACTLY as today (zero added work on the no-filter path).
	if pred != nil {
		w := 8 * k
		if w < 100 {
			w = 100
		}
		if candPerToken < w {
			candPerToken = w
		}
		if candPerToken > defaultMaxEfSearch {
			candPerToken = defaultMaxEfSearch
		}
	}

	// Normalize query tokens once: the stored doc vectors are unit-length, so
	// MaxSim's similarity is a plain dot product.
	norm := make([][]float32, len(query))
	for i, q := range query {
		nq := make([]float32, len(q))
		copy(nq, q)
		normalize(nq)
		norm[i] = nq
	}

	// Gather the candidate document set. When a selective, index-
	// narrowable filter is present, the id-keyed payload index yields a candidate
	// SUPERSET directly and we BRUTE-FORCE the exact MaxSim rerank over ONLY those
	// docs, skipping the token-HNSW gather entirely (filter-first). Otherwise we
	// fall back VERBATIM to today's adaptive-over-fetch token-HNSW gather. The
	// Stage-2 rerank below (live-meta pred re-check + MaxSim + top-k) is IDENTICAL
	// for both paths, so the filter-first result equals the post-filter result (the
	// re-check rejects any over-cover).
	var cands []uint64
	if ff, ok := m.mvFilterFirstCands(pred, filter, k); ok {
		cands = ff
	} else {
		// Stage 1 (fallback): union the parent documents of each query token's
		// nearest token nodes. seen preserves first-encounter order only for
		// determinism of the candidate list; final ranking is by MaxSim.
		seen := make(map[uint64]struct{})
		for _, nq := range norm {
			hits, err := m.idx.Search(nq, candPerToken)
			if err != nil {
				return nil, err
			}
			for _, h := range hits {
				doc, ok := m.tokenDoc[h.ID]
				if !ok {
					continue
				}
				if _, dup := seen[doc]; !dup {
					seen[doc] = struct{}{}
					cands = append(cands, doc)
				}
			}
		}
	}
	if len(cands) == 0 {
		return nil, nil
	}

	// Exact MaxSim rerank. Resolve EVERY candidate doc's token vectors in
	// ONE vecsForIDs call (one inner-index lock, returns copies), then score each
	// doc's MaxSim from the returned map with NO lock held (pure compute). With the
	// token floats present (HNSW / IVF-Flat / IVF inner with IVFRerank) the dot
	// products are byte-identical to the former direct-arena read; with an IVF-PQ
	// inner index that dropped its floats the tokens are RECONSTRUCTED ⇒ the MaxSim
	// is approximate (the documented footprint tradeoff).
	now := nowMs // one clock snapshot for the whole rerank's per-key-TTL view (injected so the hybrid shares it with the sparse lane)
	// Stage 2 exact MaxSim rerank, scoring against arena VIEWS under one read lock —
	// NO per-token-vector copy. The former path resolved every candidate doc's token
	// vectors via vecsForIDs into a map[uint64][]float32 (one fresh []float32 per
	// token id: tens of thousands of allocations and megabytes copied per query —
	// the MV search hotspot). withVecAccess instead lends arena views for the
	// duration of fn; we materialize each doc's token views into a REUSED docVecs
	// buffer (slice headers only, no float copy) and score. Byte-identical to the old
	// maxSimFromMap path: same token order, same skip-absent, same dotProduct.
	scored := make([]MultiResult, 0, len(cands))
	m.idx.withVecAccess(func(get func(id uint64) ([]float32, bool)) {
		docVecs := make([][]float32, 0, 64) // reused across docs; holds views, not copies
		for _, doc := range cands {
			// Per-key-TTL live view: drop expired keys before BOTH the predicate eval
			// and the emitted payload. liveMetaMap short-circuits (no clone) for a doc
			// with no per-key TTL — zero overhead on the common path.
			live := liveMetaMap(m.docMeta[doc], m.keyTTL[doc], now)
			// Post-filter: drop non-matching docs before scoring/appending. pred reads
			// docMeta (held under m.mu.RLock). nil short-circuits (no-filter path).
			if pred != nil && !pred(live) {
				continue
			}
			docVecs = docVecs[:0]
			for _, tid := range m.docTokens[doc] {
				if v, ok := get(tid); ok {
					docVecs = append(docVecs, v)
				}
			}
			scored = append(scored, MultiResult{
				ID:       doc,
				Score:    maxSimScore(norm, docVecs),
				Metadata: live,
			})
		}
	})

	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].ID < scored[j].ID // stable tie-break
	})
	if len(scored) > k {
		scored = scored[:k]
	}
	return scored, nil
}

// maxSimScore computes Σ_q max_d dot(q, d) over a document's already-materialized
// token vectors docVecs (in token order). Shared by maxSimFromMap (map-resolved
// copies) and the view-based Stage-2 rerank (arena views via withVecAccess), so
// both score bit-identically. query tokens are normalized and stored doc vectors
// are unit-length, so each pair similarity is a plain dot product.
func maxSimScore(query [][]float32, docVecs [][]float32) float32 {
	var score float32
	for _, q := range query {
		var best float32
		for di, d := range docVecs {
			s := dotProduct(q, d)
			if di == 0 || s > best {
				best = s
			}
		}
		score += best
	}
	return score
}

// Config returns the construction config of this index. Used by the
// vector_mv_get_config introspection op (e.g. so an offline resplit can
// re-create new-generation partitions with the same configuration).
func (m *MultiVectorIndex) Config() MultiVectorConfig { return m.cfg }

// MultiScanRecord (a complete, live MV document exported by ScanDocuments) now
// lives in the engine-free vtypes leaf package and is re-exported from
// vtypes_aliases.go.

// ScanDocuments enumerates every LIVE document as a self-contained
// MultiScanRecord. A docID present in docTokens is live (the index has no
// tombstones — Delete removes the bookkeeping outright). Each token vector is
// DEEP-COPIED off arena storage (arena.Vec returns a view), and metadata is
// rebuilt, so the result is safe to retain. Mirrors hnsw.scanVectors for the
// dense path.
func (m *MultiVectorIndex) ScanDocuments() []MultiScanRecord {
	m.mu.RLock()
	defer m.mu.RUnlock()
	// Resolve EVERY live doc's token vectors in ONE vecsForIDs call (one inner-index
	// lock, returns copies). For an IVF-PQ inner index with dropped floats the tokens
	// are reconstructed; HNSW / IVF-Flat / IVFRerank return the exact stored floats.
	allTokenIDs := make([]uint64, 0)
	for _, tokenIDs := range m.docTokens {
		allTokenIDs = append(allTokenIDs, tokenIDs...)
	}
	tokVecs := m.idx.vecsForIDs(allTokenIDs)
	recs := make([]MultiScanRecord, 0, len(m.docTokens))
	for docID, tokenIDs := range m.docTokens {
		rec := MultiScanRecord{ID: docID, Tokens: make([][]float32, 0, len(tokenIDs)), Version: m.version[docID]}
		for _, tid := range tokenIDs {
			if v, ok := tokVecs[tid]; ok {
				rec.Tokens = append(rec.Tokens, v) // already an owned copy from vecsForIDs
			}
		}
		if meta := m.docMeta[docID]; len(meta) > 0 {
			out := make(Metadata, len(meta))
			for k, v := range meta {
				out[k] = v
			}
			rec.Metadata = out
		}
		if ke := m.keyTTL[docID]; len(ke) > 0 {
			// CLONE: m.keyTTL[docID] is live index storage (the set paths clone-on-write,
			// so the scan copy must not retain it). i64 -> u64 absolute unix-ms deadlines,
			// carried verbatim so the reshard reinsert restores them time-stable.
			out := make(map[string]uint64, len(ke))
			for k, dl := range ke {
				out[k] = uint64(dl) //nolint:gosec
			}
			rec.KeyExpires = out
		}
		// CLONE the doc-level sparse vector (m.docSparse[docID] is live index storage;
		// cloneSparse returns nil for a nil/zero entry). Carried verbatim so the reshard
		// reinsert restores it — otherwise per-doc sparse is silently dropped on reshard.
		if sv := cloneSparse(m.docSparse[docID]); sv != nil {
			rec.Sparse = sv
		}
		recs = append(recs, rec)
	}
	return recs
}

// ScrollDocsPage is the cursor-aware MV scroll: up to limit live documents (id +
// payload) matching filter whose id is strictly greater than afterID when
// hasAfter (otherwise from the smallest id), id-ASCENDING. Returns docs,
// nextAfter (largest id returned), and hasMore (page filled limit AND a further
// id remains). The MV-family analogue of NamedCollection.ScrollDocsPage. Compiles
// the filter so a malformed filter fails loud at the edge. See scrollPage.
func (m *MultiVectorIndex) ScrollDocsPage(filter Filter, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, 0, false, err // fail loud on a malformed filter
	}
	docs, nextAfter, hasMore = m.scrollPage(filter, pred, nil, afterID, 0, hasAfter, limit)
	return docs, nextAfter, hasMore, nil
}

// ScrollDocsPageOrder is the order_by-aware MV cursor scroll: up to limit live
// documents matching filter, ordered by the order_by field's (value, id) total order
// (see OrderBy / OrderLess). Points whose order field is missing/non-numeric are
// EXCLUDED. afterKey/afterID is the resume cursor's (value, id) position (the page
// returns rows strictly after it); on page 1 (hasAfter=false) order.StartFrom (when
// set) is the inclusive starting value bound. The order field travels in each
// Document's Metadata so the coordinator can read the last doc's order value for the
// v2 next-cursor. order == nil falls back to the id-ascending ScrollDocsPage path.
// The MV-family analogue of Collection.ScrollDocsPageOrder.
func (m *MultiVectorIndex) ScrollDocsPageOrder(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, 0, false, err // fail loud on a malformed filter
	}
	docs, nextAfter, hasMore = m.scrollPage(filter, pred, order, afterID, afterKey, hasAfter, limit)
	return docs, nextAfter, hasMore, nil
}

// scrollPage is the MV family's deterministic id-ASCENDING scroll primitive, the
// analogue of NamedCollection.scrollPage for the multi-vector doc id set. MV keeps
// its authoritative live doc set in docTokens (a docID present there is live — MV
// has no tombstones), so the ordered walk sorts the docTokens keys ascending and
// applies the payload filter pred against docMeta (the same map MV search filters
// on). pred is the precompiled filter predicate (nil ⇒ match all); the caller
// compiles it so a malformed filter fails loud at the edge.
//
// There is NO TTL gate (MV has no TTL, unlike named) and NO version snapshot (MV
// has no idSetVersion counter). The sort is per-call — always reading the current
// docTokens, so a doc deleted between pages simply disappears from the next page.
// A cached sorted-id snapshot + an MV idSetVersion counter is a documented
// follow-up if MV scroll volume warrants it (the same follow-up named carries).
//
// Returns the collected docs, nextAfter (largest id collected), and hasMore (true
// iff stopped at limit with a further id remaining). limit <= 0 means no cap.
//
// When order != nil this swaps the id-ascending comparator for the (value, id) total
// order over the SAME per-key-TTL-gated docMeta view, EXCLUDES docs whose order field
// is missing/non-numeric (Qdrant default), seeks STRICTLY PAST the (afterKey, afterID)
// cursor — or past order.StartFrom on page 1 — and walks limit. nil order ⇒ the
// existing id-ascending path VERBATIM (zero overhead). The MV doc set already sorts
// per-call, so order_by only swaps the comparator + adds the (value, id) seek.
func (m *MultiVectorIndex) scrollPage(filter Filter, pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	if order != nil {
		// Filter-first order narrowing: when a filter is present and the id-keyed
		// payload index narrows it to a selective candidate SUPERSET, build the
		// value-sorted order rows over THOSE candidate ids (∩ live) FRESH (never cached
		// — the cache key is filter-independent) instead of the full N-row snapshot,
		// then collectOrderedLocked seeks + predicate-rechecks + pages identically. The
		// narrowed rows are a superset of the matches in the SAME value-order, so the
		// emitted docs / nextAfter / cursor are byte-identical to the predicate-eval
		// order page; hasMore is i+1<len(narrowedRows) (the candidate superset), which
		// can skip a trailing EMPTY page the full path would emit — invisible on the
		// wire (the leaf discards hasMore; the coordinator derives next_cursor from
		// len(docs)==limit).
		m.mu.RLock()
		if rows, ok := m.filterFirstOrderRowsLocked(filter, pred, order); ok {
			docs, nextAfter, hasMore = m.collectOrderedLocked(rows, pred, order, afterID, afterKey, hasAfter, limit, m.nowMs())
			m.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		// Warm path: a cached (field, direction) snapshot fresh at the current
		// dataVersion ⇒ walk under the read lock (rows immutable; a rebuild replaces
		// the pointer wholesale). Miss ⇒ relock for the cold rebuild (double-checked).
		if snap := m.orderSnapWarmLocked(order); snap != nil {
			docs, nextAfter, hasMore = m.collectOrderedLocked(snap.rows, pred, order, afterID, afterKey, hasAfter, limit, m.nowMs())
			m.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		m.mu.RUnlock()
		m.mu.Lock()
		defer m.mu.Unlock()
		// Re-test filter-first under the write lock; build narrowed fresh, else cached full.
		if rows, ok := m.filterFirstOrderRowsLocked(filter, pred, order); ok {
			return m.collectOrderedLocked(rows, pred, order, afterID, afterKey, hasAfter, limit, m.nowMs())
		}
		snap := m.orderSnapLocked(order)
		return m.collectOrderedLocked(snap.rows, pred, order, afterID, afterKey, hasAfter, limit, m.nowMs())
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	now := m.nowMs()
	// The full authoritative live doc set, ascending. hasMore is ALWAYS computed
	// against this set so the boundary is byte-identical whether or not filter-first
	// narrowed the walked set (a filtered page can be followed by a trailing empty
	// page in BOTH paths).
	fullIDs := make([]uint64, 0, len(m.docTokens))
	for id := range m.docTokens {
		fullIDs = append(fullIDs, id)
	}
	sort.Slice(fullIDs, func(i, j int) bool { return fullIDs[i] < fullIDs[j] })
	// Filter-first narrowing: when a filter is present and the id-keyed payload index
	// narrows it to a selective candidate SUPERSET, walk that id-sorted superset
	// (per-key-TTL + predicate rechecked below) instead of the full set. The recheck +
	// the full-set hasMore make the page byte-identical to the full walk; ok=false ⇒
	// the full docTokens walk (no-filter / non-accelerable / non-selective).
	ids := fullIDs
	usedFF := false
	if cands, ok := m.filterFirstScrollCandsLocked(filter, pred); ok {
		ids = cands
		usedFF = true
	}
	start := 0
	if hasAfter {
		start = sort.Search(len(ids), func(i int) bool { return ids[i] > afterID })
	}
	for i := start; i < len(ids); i++ {
		id := ids[i]
		// Drop per-key-TTL-expired keys before BOTH the predicate eval and the
		// emitted payload (so a filter on an expired key never matches).
		payload := liveMetaMap(m.docMeta[id], m.keyTTL[id], now)
		if pred != nil && !pred(payload) {
			continue
		}
		docs = append(docs, Document{ID: id, Metadata: payload})
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			if usedFF {
				hasMore = moreBeyond(fullIDs, nextAfter)
			} else {
				hasMore = i+1 < len(ids)
			}
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// filterFirstScrollCandsLocked consults the id-keyed payload index for an
// id-ASCENDING candidate superset for filter, returning (sortedIDs, true) only when
// the filter is index-narrowable AND selective (bounded by defaultFilterFirstThreshold).
// Returns (nil, false) to fall back to the full docTokens predicate-eval walk
// (no-filter / non-accelerable / non-selective). The candidate ids are intersected
// with the AUTHORITATIVE live set (m.docTokens) so a stale index posting can never
// emit a non-live doc — exactly the ids the full walk would consider, just narrowed.
// The per-key-TTL + predicate recheck (in the caller) drops over-cover, so the page is
// byte-identical to the full walk. Must hold m.mu (read).
func (m *MultiVectorIndex) filterFirstScrollCandsLocked(filter Filter, pred Predicate) ([]uint64, bool) {
	if pred == nil || m.payloadIdx == nil {
		return nil, false // no filter / no index -> full walk unchanged
	}
	threshold := defaultFilterFirstThreshold
	cands, ok := m.payloadIdx.candidates(filter, threshold)
	if !ok || len(cands) > threshold {
		return nil, false
	}
	ids := make([]uint64, 0, len(cands))
	for _, id := range cands {
		if _, live := m.docTokens[id]; live {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids, true
}

// orderSnapWarmLocked returns the cached MV order snapshot for this (field,
// direction) IF it exists and is fresh at the current dataVersion; nil otherwise
// (caller falls to the cold rebuild). Must hold m.mu (R or W). The returned
// *orderSnap's rows slice is immutable (a rebuild replaces the pointer), so a warm
// RLock reader is race-safe.
func (m *MultiVectorIndex) orderSnapWarmLocked(order *OrderBy) *orderSnap {
	if snap, ok := m.orderSnaps[orderSnapCacheKey(order)]; ok && snap.ver == m.dataVersion {
		return snap
	}
	return nil
}

// orderSnapLocked returns a fresh MV order snapshot for (field, direction),
// rebuilding it if stale or absent. Double-checked: the warm path may have lost the
// version race and relocked, but another goroutine may have rebuilt in the gap, so
// re-test the version before rebuilding. Must hold m.mu (WRITE). The snapshot is
// FILTER-INDEPENDENT — it caches EVERY live doc that HAS the order field, sorted by
// (value, id); the per-query filter + per-key-TTL gate run later in
// collectOrderedLocked.
func (m *MultiVectorIndex) orderSnapLocked(order *OrderBy) *orderSnap {
	key := orderSnapCacheKey(order)
	if snap, ok := m.orderSnaps[key]; ok && snap.ver == m.dataVersion {
		return snap // rebuilt by another goroutine in the unlock/relock gap
	}
	rows := m.buildOrderRowsLocked(order)
	m.orderSeq++
	snap := &orderSnap{ver: m.dataVersion, seq: m.orderSeq, rows: rows}
	m.orderSnaps[key] = snap
	if len(m.orderSnaps) > orderCacheCap {
		evictOldestOrderSnap(m.orderSnaps)
	}
	m.orderRebuilds++
	return snap
}

// buildOrderRowsLocked collects every live doc that HAS the order field into a
// (value, id)-sorted slice. MV has no point TTL/tombstone — a docID in docTokens is
// live (a delete bumps dataVersion ⇒ rebuild), so the only lazy expiry is per-key
// TTL, which the order value already reflects via liveMetaMap. The per-query FILTER
// is deliberately NOT applied here so the snapshot is reusable across different
// filters. Missing / non-numeric order field ⇒ EXCLUDED (Qdrant default). Must hold
// m.mu (WRITE).
func (m *MultiVectorIndex) buildOrderRowsLocked(order *OrderBy) []OrderedID {
	now := m.nowMs()
	rows := make([]OrderedID, 0, len(m.docTokens))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for id := range m.docTokens {
			payload := liveMetaMap(m.docMeta[id], m.keyTTL[id], now)
			vals, ok := orderTupleKeys(payload, keys)
			if !ok {
				continue // EXCLUDE: some order key absent or wrong-type
			}
			rows = append(rows, OrderedID{ID: id, Keys: vals})
		}
		SortOrderedIDsTuple(rows, keys)
		return rows
	}
	str := order.Kind == OrderString
	for id := range m.docTokens {
		payload := liveMetaMap(m.docMeta[id], m.keyTTL[id], now)
		if str {
			sk, kok := OrderStringKey(payload, order.Key)
			if !kok {
				continue // EXCLUDE: order field absent or non-string
			}
			rows = append(rows, OrderedID{StrKey: sk, ID: id})
			continue
		}
		key, kok := OrderKey(payload, order.Key, order.IsDatetime)
		if !kok {
			continue // EXCLUDE: order field absent or non-numeric
		}
		rows = append(rows, OrderedID{Key: key, ID: id})
	}
	if str {
		SortOrderedIDsStr(rows, order.Desc)
	} else {
		SortOrderedIDs(rows, order.Desc)
	}
	return rows
}

// filterFirstOrderRowsLocked builds the value-sorted order rows ONLY over the id-keyed
// payload-index candidate SUPERSET (∩ live m.docTokens) for filter, or (nil, false) when
// filter-first order narrowing does not apply (no filter / no index / non-accelerable /
// non-selective). The candidate ids are a superset of the matching ids; the per-row
// field-presence EXCLUDE here + the per-row per-key-TTL + predicate recheck in
// collectOrderedLocked make the narrowed rows EXACTLY the field-present matches in the
// SAME value-order as the full snapshot, so the page (docs / nextAfter / cursor) is
// byte-identical to the predicate-eval order page. NOT cached (the orderSnaps key is
// filter-independent; a narrowed snapshot must never be stored there). Must hold m.mu
// (R or W). The MV analogue of named.filterFirstOrderRowsLocked (id-keyed candidates).
func (m *MultiVectorIndex) filterFirstOrderRowsLocked(filter Filter, pred Predicate, order *OrderBy) ([]OrderedID, bool) {
	if pred == nil || m.payloadIdx == nil {
		return nil, false
	}
	threshold := defaultFilterFirstThreshold
	cands, ok := m.payloadIdx.candidates(filter, threshold)
	if !ok || len(cands) > threshold {
		return nil, false
	}
	now := m.nowMs()
	rows := make([]OrderedID, 0, len(cands))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for _, id := range cands {
			if _, live := m.docTokens[id]; !live {
				continue // stale index posting -> not live
			}
			payload := liveMetaMap(m.docMeta[id], m.keyTTL[id], now)
			vals, vok := orderTupleKeys(payload, keys)
			if !vok {
				continue // EXCLUDE: some order key absent or wrong-type
			}
			rows = append(rows, OrderedID{ID: id, Keys: vals})
		}
		SortOrderedIDsTuple(rows, keys)
		return rows, true
	}
	str := order.Kind == OrderString
	for _, id := range cands {
		if _, live := m.docTokens[id]; !live {
			continue // stale index posting -> not live
		}
		payload := liveMetaMap(m.docMeta[id], m.keyTTL[id], now)
		if str {
			sk, kok := OrderStringKey(payload, order.Key)
			if !kok {
				continue // EXCLUDE: order field absent or non-string
			}
			rows = append(rows, OrderedID{StrKey: sk, ID: id})
			continue
		}
		key, kok := OrderKey(payload, order.Key, order.IsDatetime)
		if !kok {
			continue // EXCLUDE: order field absent or non-numeric
		}
		rows = append(rows, OrderedID{Key: key, ID: id})
	}
	if str {
		SortOrderedIDsStr(rows, order.Desc)
	} else {
		SortOrderedIDs(rows, order.Desc)
	}
	return rows, true
}

// collectOrderedLocked is the MV family's order_by walk: it seeks the cached
// (value, id) sorted rows past the (afterKey, afterID) cursor — or past
// order.StartFrom on page 1 — then walks forward RE-READING live m.docMeta/m.keyTTL
// (the snapshot is filter-independent, so the per-query FILTER + per-key-TTL gate run
// HERE over the live payload), materializing up to limit Documents. The emitted
// Metadata is the live per-key-TTL-gated payload. Must hold m.mu (R or W). rows is
// immutable (never mutated in place).
func (m *MultiVectorIndex) collectOrderedLocked(rows []OrderedID, pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int, now int64) (docs []Document, nextAfter uint64, hasMore bool) {
	start := orderSeekStart(rows, order, afterID, afterKey, hasAfter)
	str := order.Kind == OrderString
	multi := isMultiKey(order)
	var keys []OrderBy
	if multi {
		keys = orderKeyList(order)
	}
	for i := start; i < len(rows); i++ {
		id := rows[i].ID
		if _, ok := m.docTokens[id]; !ok {
			continue // deleted since the snapshot was built (benign; bump rebuilds next)
		}
		// Re-read the live per-key-TTL-gated payload for BOTH the predicate eval and
		// the emitted payload (a filter on an expired key never matches; the scrolled
		// payload omits it).
		payload := liveMetaMap(m.docMeta[id], m.keyTTL[id], now)
		// Re-check the order field(s) are STILL present + the right type in the live view:
		// a per-key TTL on an order field may have lazily expired since the snapshot was
		// built (no bump), so EXCLUDE it (Qdrant missing-field policy) — identical to a
		// per-call sort over the live payload. Multi-key requires EVERY key present;
		// single-key checks the one field by its kind.
		switch {
		case multi:
			if _, kok := orderTupleKeys(payload, keys); !kok {
				continue
			}
		case str:
			if _, kok := OrderStringKey(payload, order.Key); !kok {
				continue
			}
		default:
			if _, kok := OrderKey(payload, order.Key, order.IsDatetime); !kok {
				continue
			}
		}
		if pred != nil && !pred(payload) {
			continue
		}
		docs = append(docs, Document{ID: id, Metadata: payload})
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			hasMore = i+1 < len(rows)
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// NumDocs returns the number of documents in the index.
func (m *MultiVectorIndex) NumDocs() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.docTokens)
}

// NumVectors returns the total number of token vectors across all documents.
func (m *MultiVectorIndex) NumVectors() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.tokenDoc)
}

// Close releases the underlying index and, on a WAL-mode index, closes the WAL
// file (the durable records remain on disk for replay on the next open).
func (m *MultiVectorIndex) Close() error {
	// Stop the sweeper BEFORE taking m.mu — the sweeper goroutine acquires m.mu, so
	// stopping under the lock would deadlock. Idempotent + safe before startSweeper ran,
	// so Close on any teardown/error path joins the goroutine (no leak).
	m.Stop()
	m.mu.Lock()
	defer m.mu.Unlock()
	err := m.idx.Close()
	if m.wal != nil {
		if werr := m.wal.close(); werr != nil && err == nil {
			err = werr
		}
	}
	return err
}
