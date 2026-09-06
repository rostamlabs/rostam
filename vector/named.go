// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Named vectors (Qdrant-style multi-vector-space collections). A NamedCollection
// holds a MAP of named dense vector spaces — each its own HNSW sub-index — that
// all share ONE per-point payload and ONE point-id namespace:
//
//	indexes := {"title": <hnsw dim 384 cosine>, "image": <hnsw dim 512 dot>}
//	meta    := id -> payload   (ONE payload per point, shared across all spaces)
//
// This is a NEW collection family alongside the single-vector Collection and the
// multi-vector MultiVectorIndex (its DIRECT structural template — see
// multivector.go). Insert supplies a map of named vectors for a point (a point
// MAY omit some configured spaces); search names which space to query. The
// shared payload mirrors MultiVectorIndex.docMeta: the per-point payload lives in
// the OUTER meta map, NOT in the sub-index arenas — each sub-arena's metadata is
// nil. A filtered named search therefore evaluates the predicate against the
// shared payload via the Task-1 injectable metadata provider
// (hnsw.SearchFilteredWith / scrollDocsWith), which is predicate-eval (graph
// traversal), not index-accelerated. The single-vector + MV paths are untouched.

// NamedCollection is a named-vector collection: one HNSW sub-index per named
// space, a shared per-point payload + ttl, and a live-point set. Safe for
// concurrent use.
type NamedCollection struct {
	mu   sync.RWMutex
	name string
	cfg  map[string]NamedVectorParams // configured spaces (name -> per-space params)

	indexes map[string]VectorIndex // one sub-index per DENSE named space (sub-arena metadata = nil); HNSW or IVF per the space's IndexType

	// sparseSpaces holds one entry per SPARSE named space (cfg[name].Sparse), the
	// id-keyed parallel to indexes. A sparse space is backed by a standalone
	// id-keyed inverted index (sparseIndexID), NOT an HNSW: vecs[id] stores the
	// point's *SparseVector and idx is the inverted index searchTopK queries. The
	// dense slot-keyed sparse index (hnsw.sparseIdx) is UNTOUCHED — this mirrors it
	// id-keyed for the named family. Empty for a dense-only collection (no sparse
	// blob anywhere). Guarded by nc.mu exactly like indexes.
	sparseSpaces map[string]*namedSparseSpace

	// Shared per-point state (the authoritative payload, mirroring
	// MultiVectorIndex.docMeta). meta[id] is the ONE payload shared across every
	// named space; ttl[id] is the absolute unix-millis deadline (0 = no expiry);
	// ids is the live point set.
	meta map[uint64]Metadata
	ttl  map[uint64]int64
	ids  map[uint64]struct{}

	// keyTTL is the per-key payload TTL side-structure: id -> key -> ABSOLUTE
	// unix-millis deadline. A key is logically expired iff its deadline is
	// non-zero and <= now (mirrors point TTL). Independent of the point ttl above
	// (the point gate fires first — a dead point drops all its keys). An id with
	// no per-key TTL has no entry here (the cheap common case). Lazy-only: expired
	// keys are dropped on every read (Get / scroll / the filter metadata view);
	// there is no sweep. Snapshot-durable (no WAL); old snapshots restore it nil.
	keyTTL map[uint64]map[string]int64

	// version is the per-point optimistic-CAS version side-structure: id ->
	// monotonic uint64 version (the named-family analogue of the dense arena
	// versions side-array). A new insert sets it to 1, every in-place mutate
	// (upsert / any payload op) bumps it +1, delete drops it (an absent point
	// reads version 0), and a reinsert of a previously-deleted id starts at 1
	// again. Stored ALONGSIDE meta/ttl/keyTTL (mirroring keyTTL's storage /
	// snapshot / WAL pattern EXACTLY) so it persists verbatim. The check + bump
	// are atomic under nc.mu (the FSM-Apply serialization point), so the result
	// is deterministic across Raft replicas. An id with no version has no entry
	// (version 0 = absent). Snapshot-durable + WAL-logged; old artifacts restore
	// it from a sane default (see Restore / replay).
	version map[uint64]uint64

	// inuse counts in-flight operations holding this collection (taken by
	// CollectionStore.AcquireNamed, dropped by Release), so a concurrent
	// DropNamed never closes sub-indexes under a reader. Mirrors
	// MultiVectorIndex.inuse / Collection.inuse.
	inuse atomic.Int64

	// Write-ahead log (nil unless WAL-mode, single-node only). opMu serializes
	// {apply + WAL append} against {checkpoint + WAL rotate} (Flush) so a Flush
	// never truncates an op it didn't capture — exactly the dense Collection
	// discipline (see collection.go). Only taken when wal != nil; a heap-only
	// (non-WAL) named collection pays nothing. opMu is the OUTER lock: a mutator
	// takes opMu, then nc.mu for the actual state change.
	wal  *wal
	opMu sync.Mutex

	// sweeper lifecycle (per-key TTL background reclaim). Mirrors the dense
	// Collection sweeper (collection.go): startSweeper launches a single ticker
	// goroutine on the first insert (gated by sweepStart), runSweeper calls
	// sweepKeyTTLOnce each tick, and Stop joins the goroutine (closes sweepStop,
	// waits sweepDone). A collection with no per-key TTL still runs the ticker but
	// each sweepKeyTTLOnce is a cheap no-op (matches dense). nil sweepStop = never
	// started, so Stop is a safe no-op.
	sweepStop  chan struct{}
	sweepDone  chan struct{}
	sweepStart sync.Once
	// suppressSweep permanently no-ops startSweeper (the replicated/persistent-cluster
	// policy): under Raft the wall-clock per-key sweeper would diverge committed state
	// across replicas, so it is off and expired keys are filtered lazily at read time
	// only (#4 vector TTL determinism, B3a analog). Set by the store on the cluster
	// build path; default false = single-node behavior (sweeper on).
	suppressSweep bool

	// now is the clock used for the shared-payload TTL deadlines (both the
	// insert-time deadline computation and the scroll expiry check). Defaults to
	// time.Now().UnixMilli when nil; tests can override to age deterministically.
	// Mirrors hnsw.now. nil-default keeps behavior identical to wall-clock time.
	now func() int64

	// payloadIdx is the id-keyed inverted index over the shared payload (the
	// id-keyed analogue of the dense hnsw.payloadIdx). It accelerates SELECTIVE
	// filtered named searches via the filter-first path (candidates() -> a
	// candidate-id SUPERSET -> brute-force scoring in the target space with the
	// existing live-meta predicate RE-CHECK). It is an ACCELERATION ONLY: the
	// predicate-eval graph path remains the correct fallback for no-filter /
	// non-accelerable / non-selective filters, and the re-check rejects any
	// over-cover. Rebuilt-on-load (never serialized) from nc.meta. Guarded by
	// nc.mu EXACTLY like nc.meta (every payload-set/insert/delete reindexes under
	// the same lock).
	payloadIdx *payloadIndexID

	// dataVersion is the order_by snapshot's invalidation counter (the named-family
	// analogue of hnsw.dataVersion). The named family has no idSetVersion; this is a
	// fresh counter that bumps on the UNION of id-set mutations
	// {insert/restore-insert/delete} AND payload mutations
	// {set/overwrite/deleteKeys/clear/restorePayload}: a cached (field, direction)
	// sorted snapshot is payload-VALUE-keyed, so a payload write to the order field
	// MUST invalidate it. Guarded by nc.mu; bumped via bumpData() under the write
	// lock at each mutator's innermost *Locked body (exactly once per mutation).
	// Starts at 1 (the zero-valued orderSnap.ver is reserved for "never built").
	dataVersion uint64

	// orderSnaps caches per-(field, direction) sorted snapshots for the order_by
	// scroll, version-stamped with dataVersion. Bounded at orderCacheCap with
	// oldest-built eviction. Guarded by nc.mu (warm read under RLock of an immutable
	// rows slice; cold rebuild under Lock, double-checked). See order.go orderSnap.
	orderSnaps map[orderCacheKey]*orderSnap
	// orderSeq is a monotonic stamp assigned to each freshly built orderSnap so the
	// cap eviction can pick the oldest-built entry. Guarded by nc.mu.
	orderSeq uint64
	// orderRebuilds counts order-snapshot rebuilds — a test hook (assert warm reuse
	// / correct invalidation). Guarded by nc.mu.
	orderRebuilds uint64
}

// namedSparseSpace is the per-space storage for a SPARSE named space: vecs maps a
// point id to its stored *SparseVector (the authoritative copy, snapshot-durable),
// and idx is the id-keyed inverted index rebuilt from vecs on load and kept in
// sync incrementally on insert/delete. The two are always consistent under nc.mu:
// every vecs mutation has a matching idx.add/idx.remove. It is the sparse analogue
// of a dense space's *hnsw (which holds both the stored vectors and the search
// structure).
type namedSparseSpace struct {
	vecs map[uint64]*SparseVector
	idx  *sparseIndexID
}

func newNamedSparseSpace() *namedSparseSpace {
	return &namedSparseSpace{vecs: make(map[uint64]*SparseVector), idx: newSparseIndexID()}
}

// store records v as id's sparse vector, dropping any prior vector first (upsert:
// the stale postings must go before the new ones are added). A nil v removes id.
func (s *namedSparseSpace) store(id uint64, v *SparseVector) {
	if _, had := s.vecs[id]; had {
		s.idx.remove(id)
		delete(s.vecs, id)
	}
	if v == nil {
		return
	}
	s.vecs[id] = v
	s.idx.add(id, v)
}

// drop removes id from the space (vecs + the inverted index). A no-op if absent.
func (s *namedSparseSpace) drop(id uint64) {
	if _, had := s.vecs[id]; !had {
		return
	}
	s.idx.remove(id)
	delete(s.vecs, id)
}

// bumpData advances dataVersion, invalidating every cached order snapshot (a later
// scroll re-tests snap.ver == dataVersion and misses). Called under nc.mu (WRITE) at
// every id-set mutation AND every payload mutation, exactly once per logical
// mutation (in the innermost *Locked body each public method funnels through).
func (nc *NamedCollection) bumpData() { nc.dataVersion++ }

// nowMs returns the current wall-millis via the injectable clock (time.Now when
// nc.now is nil). Mirrors hnsw's now-hook pattern.
func (nc *NamedCollection) nowMs() int64 {
	if nc.now == nil {
		return time.Now().UnixMilli()
	}
	return nc.now()
}

// SetNowFunc overrides the wall-clock source (unix millis) this collection's
// non-apply expiry sites consult (sweeper + read/query filter + the wall-clock
// branch of the write paths). nil restores the real clock. TEST/advanced seam
// mirroring cache.Cache.SetNowFunc; production never calls it (byte-identical
// default). Takes the write lock.
func (nc *NamedCollection) SetNowFunc(fn func() int64) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.now = fn
}

// sweepKeyTTLOnce physically reclaims expired per-key payload TTL keys: under the
// write lock it scans every id that has a keyTTL entry and drops keys whose
// ABSOLUTE deadline has passed (the SAME predicate the lazy liveMetaMap read path
// uses — deadline != 0 && <= now — so sweep and lazy-drop can never diverge) from
// BOTH nc.keyTTL[id] AND the shared payload map nc.meta[id], then reindexes the
// id-keyed payloadIdx so the dropped key's stale token/field postings physically
// go (mirroring dense ttl.go's per-key pass). An emptied keyTTL[id] entry is
// deleted. If ANY key was dropped it bumps dataVersion (a sweep is a payload
// mutation, so the order_by snapshot must invalidate — matching how the payload
// mutators bump). Returns the number of keys dropped. Logs nothing: the absolute
// deadlines are durable in the snapshot, so a swept key is re-derived (re-dropped
// lazily / re-swept) after a restart, never resurrected. Must be called WITHOUT
// holding nc.mu (it takes the write lock for the duration). Uses the injectable
// clock (nc.nowMs). A collection with no per-key TTL is a cheap no-op.
func (nc *NamedCollection) sweepKeyTTLOnce() int {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	now := uint64(nc.nowMs()) //nolint:gosec // unix-millis is non-negative
	dropped := 0
	for id, ke := range nc.keyTTL {
		if len(ke) == 0 {
			continue
		}
		meta := nc.meta[id]
		// First check whether any still-present key is actually expired (fast path:
		// deadlines exist but none reached / none present → no allocation, no mutation).
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
			delete(nc.meta, id)
		} else {
			nc.meta[id] = newMeta
		}
		if newKe == nil {
			delete(nc.keyTTL, id)
		} else {
			nc.keyTTL[id] = newKe
		}
		nc.payloadIdx.reindex(id, newMeta) // stale token/field postings physically go
	}
	if dropped > 0 {
		nc.bumpData() // payload changed → invalidate the order_by snapshot
	}
	return dropped
}

// startSweeper launches the background per-key-TTL sweeper goroutine once (gated by
// sweepStart), called on the first insert. Mirrors dense Collection.startSweeper.
func (nc *NamedCollection) startSweeper() {
	if nc.suppressSweep {
		nc.sweepStart.Do(func() {}) // consume the Once; sweepStop stays nil (Stop is a safe no-op)
		return
	}
	nc.sweepStart.Do(func() {
		nc.sweepStop = make(chan struct{})
		nc.sweepDone = make(chan struct{})
		go nc.runSweeper(defaultSweepInterval)
	})
}

// runSweeper ticks at interval, calling sweepKeyTTLOnce until sweepStop is closed.
func (nc *NamedCollection) runSweeper(interval time.Duration) {
	defer close(nc.sweepDone)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-nc.sweepStop:
			return
		case <-t.C:
			nc.sweepKeyTTLOnce()
		}
	}
}

// Stop halts the background sweeper, joining its goroutine (no leak). Idempotent
// and safe to call even if startSweeper was never invoked (sweepStop == nil) or if
// already stopped (the channel is already closed). Mirrors dense Collection.Stop.
func (nc *NamedCollection) Stop() {
	if nc.sweepStop == nil {
		return
	}
	select {
	case <-nc.sweepStop:
		// already closed
	default:
		close(nc.sweepStop)
		<-nc.sweepDone
	}
}

// reservedVectorNameChars are rejected in a named-vector space name: they
// collide with the collection-name path separator and the ops/transport key
// conventions (mirrors the collection-name guards).
const reservedVectorNameChars = "#@/"

// validateNamedVectors checks a NamedVectors config map: at least one space,
// each name non-empty + free of reserved chars, each space Dim > 0 (via the
// sub-config Validate). Returns the first failure (fail loud).
func validateNamedVectors(cfg map[string]NamedVectorParams) error {
	if len(cfg) == 0 {
		return ErrEmptyNamedVectors
	}
	for name, p := range cfg {
		if name == "" {
			return ErrEmptyVectorName
		}
		if strings.ContainsAny(name, reservedVectorNameChars) {
			return ErrReservedVectorName
		}
		// A sparse space is backed by an id-keyed inverted index (NOT an HNSW); the
		// dense-only knobs (Dim/M/Ef…/Quant) do not apply and are not validated.
		if p.Sparse {
			continue
		}
		if err := ValidateConfig(namedToConfig(p)); err != nil {
			return err
		}
	}
	return nil
}

// NewNamedCollection builds an empty named-vector collection: one heap-backed
// HNSW sub-index per configured space, plus the shared payload/ttl/live-set
// maps. Returns an error if the config is invalid (no spaces, empty/reserved
// name, bad per-space params).
func NewNamedCollection(name string, cfg map[string]NamedVectorParams) (*NamedCollection, error) {
	if err := validateNamedVectors(cfg); err != nil {
		return nil, err
	}
	nc := &NamedCollection{
		name:         name,
		cfg:          make(map[string]NamedVectorParams, len(cfg)),
		indexes:      make(map[string]VectorIndex, len(cfg)),
		sparseSpaces: make(map[string]*namedSparseSpace),
		meta:         make(map[uint64]Metadata),
		ttl:          make(map[uint64]int64),
		ids:          make(map[uint64]struct{}),
		keyTTL:       make(map[uint64]map[string]int64),
		version:      make(map[uint64]uint64),
		payloadIdx:   newPayloadIndexID(),
		// dataVersion starts at 1 so the zero-valued orderSnap.ver (0 = "never built")
		// forces the first scroll to rebuild even for a restored/bulk-loaded collection.
		dataVersion: 1,
		orderSnaps:  make(map[orderCacheKey]*orderSnap),
	}
	for sname, p := range cfg {
		if p.Sparse {
			// A sparse space is backed by an id-keyed inverted index, NOT an HNSW.
			nc.sparseSpaces[sname] = newNamedSparseSpace()
			nc.cfg[sname] = p
			continue
		}
		// Construct via newIndex so a space can select IndexIVF (IVF / IVF-PQ);
		// IndexHNSW (the zero value) builds the historical per-space graph index
		// (byte/behaviour-identical). newIndex validates the per-space Config
		// (newHNSW/newIVF both call ValidateConfig(cfg)), so a bad IVF param fails loud.
		idx, err := newIndex(namedToConfig(p))
		if err != nil {
			// Roll back any sub-indexes already built.
			for _, b := range nc.indexes {
				_ = b.Close()
			}
			return nil, err
		}
		nc.indexes[sname] = idx
		nc.cfg[sname] = p
	}
	return nc, nil
}

// Release drops a reference taken by CollectionStore.AcquireNamed.
func (nc *NamedCollection) Release() { nc.inuse.Add(-1) }

// retire waits for in-flight users to drain, then closes every sub-index and
// runs cleanup. Called only after the collection has been removed from the
// store map. Mirrors MultiVectorIndex.retire.
func (nc *NamedCollection) retire(cleanup func()) {
	for nc.inuse.Load() > 0 {
		time.Sleep(200 * time.Microsecond)
	}
	nc.Stop() // join the sweeper goroutine before closing (no leak; mirrors dense)
	_ = nc.Close()
	if cleanup != nil {
		cleanup()
	}
}

// Config returns the configured named spaces (name -> per-space params).
func (nc *NamedCollection) Config() map[string]NamedVectorParams {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	out := make(map[string]NamedVectorParams, len(nc.cfg))
	for k, v := range nc.cfg {
		out[k] = v
	}
	return out
}

// Insert upserts point id: for each provided named space it inserts the vector
// into that sub-index, and it (re)sets the shared payload + ttl for the point.
// Validation (fail loud): every name in vectors MUST be a configured space
// (ErrUnknownVectorName) and each vector's length MUST equal that space's Dim
// (ErrDimMismatch). A point MAY omit configured spaces. Re-inserting an id
// REPLACES the vectors for the provided spaces and the payload + ttl (upsert);
// spaces NOT named in this call retain their prior vector. The payload is stored
// once in the shared meta map (NOT in the sub-arenas — they get nil metadata).
// ttl == 0 means no expiry.
func (nc *NamedCollection) Insert(id uint64, vectors map[string][]float32, payload Metadata, ttl time.Duration) error {
	_, err := nc.InsertCAS(id, vectors, payload, ttl, CASCond{})
	return err
}

// InsertSparse is Insert carrying per-space SPARSE values alongside the dense
// vectors: sparseVectors[name] is the *SparseVector for a sparse space. It is the
// Task-1 engine entry point for sparse inserts (the wire carriage of sparse values
// is not yet wired; until then ops/store pass nil sparseVectors, keeping dense callers
// byte-identical). vectors and sparseVectors are disjoint by space modality (a
// dense value for a sparse space, or vice versa, is ErrSpaceModalityMismatch).
func (nc *NamedCollection) InsertSparse(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration) error {
	_, err := nc.InsertCASKeyTTL(id, vectors, sparseVectors, payload, ttl, nil, CASCond{})
	return err
}

// InsertCAS is Insert with an optimistic-CAS precondition (CASCond{} = no
// precondition, an unconditional upsert that still bumps the version). It returns
// the point's resulting version on success: 1 for a fresh id, current+1 for an
// in-place upsert. On a CAS mismatch it returns ErrVersionConflict with no
// mutation and nothing WAL-logged. The check + bump and the WAL append are
// serialized under opMu so engine + WAL agree (mirrors dense Collection.InsertCAS).
func (nc *NamedCollection) InsertCAS(id uint64, vectors map[string][]float32, payload Metadata, ttl time.Duration, cas CASCond) (uint64, error) {
	return nc.InsertCASKeyTTL(id, vectors, nil, payload, ttl, nil, cas)
}

// InsertCASKeyTTL is InsertCAS carrying an OPTIONAL per-key payload TTL map (key
// -> RELATIVE ms). The engine computes the resulting ABSOLUTE deadline map and
// the WAL logs it so replay restores it VERBATIM (time-stable). Empty/nil keyTTLMs
// is the zero-overhead path (no per-key deadlines, byte-identical WAL record).
func (nc *NamedCollection) InsertCASKeyTTL(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return nc.insertCASKeyTTLBody(id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas, false, 0)
}

// InsertCASKeyTTLAt is InsertCASKeyTTL stamping every point/per-key TTL deadline in
// the op against the EXPLICIT leader apply stamp nowMs — the replicated-apply
// variant so every replica stamps byte-identical state (#4 vector TTL determinism).
func (nc *NamedCollection) InsertCASKeyTTLAt(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return nc.insertCASKeyTTLBody(id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas, true, nowMs)
}

func (nc *NamedCollection) insertCASKeyTTLBody(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs int64) (uint64, error) {
	// Validate every name/dim/modality up front so a malformed request mutates
	// nothing.
	if err := nc.validateInsertSpaces(vectors, sparseVectors); err != nil {
		return 0, err
	}
	nc.startSweeper() // launch the per-key-TTL sweeper on first insert (mirrors dense)

	// Apply, then log on success under opMu (so a concurrent Flush's
	// snapshot+truncate can't interleave between the apply and the append). Only
	// taken when WAL-mode; heap-only named collections skip it (zero overhead).
	// opMu is the OUTER lock; nc.mu guards only the actual state change.
	if nc.wal == nil {
		now := nowMs
		if !stamped {
			now = nc.nowMs()
		}
		version, _, err := nc.insertLockedAt(id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas, now)
		return version, err
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure so a panic
	// inside insertLockedAt or the WAL append still unlocks opMu via the deferred
	// Unlock instead of leaking it forever — server/handlers.go recovers
	// per-request panics, so an unrecovered lock would silently deadlock every
	// later write on this collection. The closure returns before the durability
	// wait so concurrent writers overlap in commitWaitStaged and the leader-fsync
	// actually batches them — mirroring dense Collection.InsertCASKeyTTL. The
	// wall-clock read happens INSIDE the lock (not hoisted above it): two
	// concurrent inserts must have their `now` reads ordered the same as their
	// applies, or their WAL-logged absolute deadlines could disagree with actual
	// apply order under contention.
	version, seq, err := func() (uint64, uint64, error) {
		nc.opMu.Lock()
		defer nc.opMu.Unlock()
		now := nowMs
		if !stamped {
			now = nc.nowMs()
		}
		version, keyExpires, err := nc.insertLockedAt(id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas, now)
		if err != nil {
			return 0, 0, err
		}
		// Log the RESULTING absolute per-key deadlines (not nil) so replay restores
		// them verbatim (time-stable). keyTTLToU64 returns nil for an empty map
		// (cheap path).
		seq, err := nc.wal.appendNamedInsertStaged(id, vectors, sparseVectors, ttl, payload, keyTTLToU64(keyExpires), version)
		return version, seq, err
	}()
	if err != nil {
		return 0, err
	}
	if err := nc.wal.commitWaitStaged(seq); err != nil {
		return 0, err
	}
	return version, nil
}

// validateInsertSpaces checks every named value against its space's configured
// modality, failing loud before any mutation: a dense vector's name MUST be a
// configured DENSE space with a matching Dim (ErrUnknownVectorName / ErrDimMismatch
// / ErrSpaceModalityMismatch for a sparse space); a sparse value's name MUST be a
// configured SPARSE space (ErrUnknownVectorName / ErrSpaceModalityMismatch for a
// dense space) with a valid SparseVector. A point MAY omit configured spaces.
func (nc *NamedCollection) validateInsertSpaces(vectors map[string][]float32, sparseVectors map[string]*SparseVector) error {
	for name, vec := range vectors {
		idx, ok := nc.indexes[name]
		if !ok {
			if _, isSparse := nc.sparseSpaces[name]; isSparse {
				return ErrSpaceModalityMismatch // dense value for a sparse space
			}
			return ErrUnknownVectorName
		}
		if len(vec) != idx.Dim() {
			return ErrDimMismatch
		}
	}
	for name, sv := range sparseVectors {
		if _, ok := nc.sparseSpaces[name]; !ok {
			if _, isDense := nc.indexes[name]; isDense {
				return ErrSpaceModalityMismatch // sparse value for a dense space
			}
			return ErrUnknownVectorName
		}
		if sv != nil {
			if err := sv.Validate(); err != nil {
				return err
			}
		}
	}
	return nil
}

// insertLockedAt is insertLocked stamping EVERY TTL deadline in the op against the
// caller-supplied `now` (unix millis): the shared point deadline (nc.ttl), each
// sub-space's arena point deadline (idx.InsertAt) + its upsert-reclaim liveness
// (idx.DeleteAt), and the per-key payload deadline. A replicated named insert then
// stamps byte-identical state on every replica (#4 vector TTL determinism).
func (nc *NamedCollection) insertLockedAt(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyTTLMs map[string]int64, cas CASCond, now int64) (uint64, map[string]int64, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	// CAS precondition: read the current version (0 if absent) and bail with NO
	// mutation on a mismatch. expected==0+Has = insert-if-absent.
	if err := cas.check(nc.version[id]); err != nil {
		return 0, nil, err
	}

	// Insert into each provided space with nil arena metadata (the shared meta map
	// is authoritative); ttl rides into the sub-index so per-arena expiry works. A
	// re-insert tombstones the prior slot first so the live id resurrects with the
	// new vector (upsert; a fresh id's Delete is a no-op) — hnsw.Insert collides on
	// a still-live id otherwise.
	//
	// Sub-index Insert here can only fail on a dim mismatch, which is rejected by
	// the pre-loop validation; toConfig() sets no per-space quota/rate-limit, so a
	// mid-loop failure (which would leave earlier spaces updated but the shared
	// meta/ids stale) is unreachable. If per-space quotas are ever added, add
	// rollback of the already-updated spaces here.
	for name, vec := range vectors {
		idx := nc.indexes[name]
		// No-CAS (CASCond{}) on the per-space sub-index; the named family's own
		// per-point version lives at the NamedCollection level, not here.
		// Stamp the sub-space reclaim + arena point deadline against `now` so every
		// replica agrees on the reclaim and stamps an identical per-space expiry.
		_, _ = idx.DeleteAt(id, CASCond{}, now)
		if _, _, err := idx.InsertAt(id, vec, ttl, nil, nil, nil, CASCond{}, now); err != nil {
			return 0, nil, err
		}
	}
	// Sparse spaces: store the *SparseVector (cloned so we never alias caller-owned
	// slices) into the per-space vecs + inverted index. store() drops any prior
	// vector first (upsert: stale postings removed before the new ones are added). A
	// nil/zero sparse value drops the id from that space (store handles nil).
	for name, sv := range sparseVectors {
		nc.sparseSpaces[name].store(id, cloneSparse(sv))
	}
	nc.meta[id] = payload
	if ttl > 0 {
		nc.ttl[id] = uint64ToInt64(uint64(now) + uint64(ttl.Milliseconds())) //nolint:gosec // unix-millis is non-negative
	} else {
		nc.ttl[id] = 0
	}
	// An upsert REPLACES the payload, so any prior per-key deadlines are stale. The
	// insert-time per-key TTL (keyTTLMs: key -> RELATIVE ms) sets the FRESH point's
	// deadlines: compute absolute (now+ttl) against the new payload, pruned to its
	// present keys (mirrors set_payload / computeInsertKeyExpires). Empty/nil ⇒ the
	// map is cleared (no per-key TTL), byte-identical to the no-key_ttl path.
	ke := pruneKeyTTL(applyKeyTTLMs(nil, keyTTLMs, payload, now), payload)
	if ke == nil {
		delete(nc.keyTTL, id)
	} else {
		nc.keyTTL[id] = ke
	}
	nc.ids[id] = struct{}{}
	nc.payloadIdx.reindex(id, payload) // keep the filter-first index in sync (under nc.mu)
	// Bump the per-point version (fresh id 0 -> 1; in-place upsert current -> +1).
	v := nc.version[id] + 1
	nc.version[id] = v
	nc.bumpData() // id-set + payload change: invalidate the order_by snapshot
	return v, ke, nil
}

// RestoreInsert inserts id with an EXACT per-point version set VERBATIM (NOT
// bumped) — the WAL-replay analogue used so a replayed insert restores the exact
// version the original op produced (mirrors dense Collection.RestoreInsert). It
// does NOT WAL-log (replay is reading the log). A version of 0 falls back to the
// normal bump (an old record predating the version block defaults a fresh insert
// to 1).
func (nc *NamedCollection) RestoreInsert(id uint64, vectors map[string][]float32, sparseVectors map[string]*SparseVector, payload Metadata, ttl time.Duration, keyExpires map[string]uint64, version uint64) error {
	if err := nc.validateInsertSpaces(vectors, sparseVectors); err != nil {
		return err
	}
	nc.startSweeper() // launch the per-key-TTL sweeper on first (replayed) insert
	nc.mu.Lock()
	defer nc.mu.Unlock()
	for name, vec := range vectors {
		idx := nc.indexes[name]
		_, _ = idx.Delete(id, CASCond{})
		if _, _, err := idx.Insert(id, vec, ttl, nil, nil, nil, CASCond{}); err != nil {
			return err
		}
	}
	// Sparse spaces (WAL replay / restore): store the *SparseVector verbatim (cloned
	// to own the slices), dropping any prior vector. nil drops the id from the space.
	for name, sv := range sparseVectors {
		nc.sparseSpaces[name].store(id, cloneSparse(sv))
	}
	nc.meta[id] = payload
	if ttl > 0 {
		nc.ttl[id] = uint64ToInt64(uint64(nc.nowMs()) + uint64(ttl.Milliseconds()))
	} else {
		nc.ttl[id] = 0
	}
	// keyExpires is the ABSOLUTE per-key deadline map the WAL logged — restored
	// VERBATIM (NOT recomputed now+ttl) so a pending key deadline survives a crash
	// time-stable. Empty ⇒ clear (the no-key_ttl path). Reshard copy passes nil.
	if len(keyExpires) == 0 {
		delete(nc.keyTTL, id)
	} else {
		ke := make(map[string]int64, len(keyExpires))
		for k, dl := range keyExpires {
			ke[k] = int64(dl) //nolint:gosec
		}
		nc.keyTTL[id] = ke
	}
	nc.ids[id] = struct{}{}
	nc.payloadIdx.reindex(id, payload) // keep the filter-first index in sync (replay path)
	if version == 0 {
		// Old record without a version block: default a fresh insert to 1, else bump.
		version = nc.version[id] + 1
	}
	nc.version[id] = version
	nc.bumpData() // WAL-replayed id-set + payload change: invalidate the order_by snapshot
	return nil
}

// uint64ToInt64 converts a unix-millis deadline to the stored int64 form
// (deadlines are always within int64 range).
func uint64ToInt64(v uint64) int64 { return int64(v) }

// metaOf returns the shared-payload provider for the named sub-indexes: a point
// id maps to its single shared payload with per-key-TTL-EXPIRED keys DROPPED, so
// the filtered-search / scroll predicate evaluates against the live view (a filter
// on an expired key never matches). Caller holds nc.mu (read) for the duration of
// the search; the now() snapshot is taken once so the whole search sees one clock.
func (nc *NamedCollection) metaOf() func(id uint64) Metadata {
	now := nc.nowMs()
	return func(id uint64) Metadata { return liveMetaMap(nc.meta[id], nc.keyTTL[id], now) }
}

// rebuildPayloadIdx discards and reconstructs the filter-first index from the
// live shared payload (nc.meta). The rebuild-on-load entry point used after WAL
// replay (Restore rebuilds inline). Takes nc.mu so it is safe to call after
// replay has released its per-op locks.
func (nc *NamedCollection) rebuildPayloadIdx() {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if nc.payloadIdx == nil {
		nc.payloadIdx = newPayloadIndexID()
	}
	nc.payloadIdx.rebuild(nc.meta)
}

// searchNamedFilterFirst tries the index-accelerated filter-first path for a
// filtered named search. It returns (results, true) when the path applied, or
// (nil, false) to signal the caller to fall back to the predicate-eval graph
// search VERBATIM. Caller holds nc.mu (read). It mirrors the dense planner gate
// in hnsw.searchIntoWith: consult the id-keyed payload index for a candidate
// SUPERSET; if the filter is index-narrowable (ok) AND the candidate set is
// small enough (<= threshold) AND the cost model prefers brute force over graph
// traversal, score those candidate ids exactly in the target space with the
// live-meta predicate re-check. Otherwise fall back (zero regression for
// no-filter / non-accelerable / non-selective filters — candidates returns
// ok=false or too-many ids, or preferFilterFirst says graph is cheaper).
func (nc *NamedCollection) searchNamedFilterFirst(idx VectorIndex, query []float32, k int, filter Filter, pred Predicate) ([]Result, bool, error) {
	if pred == nil || nc.payloadIdx == nil {
		return nil, false, nil // no filter -> existing path unchanged
	}
	threshold := idx.effectiveFilterFirstLimit(idx.Stats().Size)
	// maxCand is the largest candidate count the planner would still choose
	// filter-first for (filterFirstCrossover, computed from the inner index's own
	// cost model). Materializing the whole superset only to discover
	// preferFilterFirst rejects it is pure waste, so candidatesCapped abandons the
	// build the moment the set exceeds maxCand. The decision is identical to the
	// old materialize-then-check — see TestIVFFilterFirstCrossoverMatchesUncappedDecision.
	maxCand := idx.filterFirstCrossover(k, threshold)
	cands, ok := nc.payloadIdx.candidatesCapped(filter, threshold, maxCand)
	if !ok {
		return nil, false, nil // not narrowable / non-selective / graph cheaper -> fall back
	}
	res := idx.filterFirstByID(nil, cands, query, k, pred, nc.metaOf())
	return res, true, nil
}

// SearchNamed returns the top-k point ids nearest to query in the named space,
// optionally filtered. name MUST be a configured space (ErrUnknownVectorName).
// When a filter is present and selective, an id-keyed payload index narrows the
// search to a candidate superset that is brute-force-scored exactly (filter-
// first) with the live-meta predicate re-check; otherwise the filter is
// evaluated against the SHARED per-point payload via the injectable provider
// (predicate-eval graph traversal) — the correct fallback for no-filter /
// non-accelerable / non-selective filters.
func (nc *NamedCollection) SearchNamed(name string, query []float32, k int, filter Filter) ([]Result, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	idx, ok := nc.indexes[name]
	if !ok {
		return nil, ErrUnknownVectorName
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	if res, applied, ferr := nc.searchNamedFilterFirst(idx, query, k, filter, pred); applied {
		return res, ferr
	}
	return idx.SearchFilteredWith(nil, query, k, filter, nc.metaOf())
}

// SearchNamedDocs is SearchNamed returning Documents carrying the SHARED
// per-point payload (the docs/scroll result shape). The named spaces store no
// content/metadata in their arenas, so each Document's Content is empty and its
// Metadata is the shared payload; Distance is the named-space distance.
func (nc *NamedCollection) SearchNamedDocs(name string, query []float32, k int, filter Filter) ([]Document, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	idx, ok := nc.indexes[name]
	if !ok {
		return nil, ErrUnknownVectorName
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	var res []Result
	if ffRes, applied, ferr := nc.searchNamedFilterFirst(idx, query, k, filter, pred); applied {
		if ferr != nil {
			return nil, ferr
		}
		res = ffRes
	} else {
		res, err = idx.SearchFilteredWith(nil, query, k, filter, nc.metaOf())
		if err != nil {
			return nil, err
		}
	}
	now := nc.nowMs()
	docs := make([]Document, 0, len(res))
	for _, r := range res {
		docs = append(docs, Document{
			ID:       r.ID,
			Distance: r.Distance,
			Score:    r.Score,
			// Drop per-key-TTL-expired keys from the returned payload (the live view).
			Metadata: liveMetaMap(nc.meta[r.ID], nc.keyTTL[r.ID], now),
		})
	}
	return docs, nil
}

// SearchNamedSparse returns the top-k point ids by sparse dot-product in the
// SPARSE named space, optionally filtered. space MUST be a configured SPARSE
// space (ErrUnknownVectorName for an absent space; ErrSpaceModalityMismatch for a
// DENSE space — fail loud, the dense lane has its own SearchNamed). The admit gate
// applied to every candidate is EXACTLY the one SearchNamed uses: the live shared
// payload (per-key-TTL-expired keys dropped) is point-TTL-gated and the compiled
// filter predicate is evaluated against it, so a filtered/expired point is never
// scored. Score is the sparse dot product; results are descending by score (ties
// by lower id). This is the Task-2 sparse lane Task-3's NamedHybrid reuses.
func (nc *NamedCollection) SearchNamedSparse(space string, query *SparseVector, k int, filter Filter) ([]Result, error) {
	if query != nil {
		if err := query.Validate(); err != nil {
			return nil, err
		}
	}
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	sp, ok := nc.sparseSpaces[space]
	if !ok {
		if _, isDense := nc.indexes[space]; isDense {
			return nil, ErrSpaceModalityMismatch // a sparse search against a dense space
		}
		return nil, ErrUnknownVectorName
	}
	admit, err := nc.sparseAdmitLocked(filter)
	if err != nil {
		return nil, err
	}
	return sp.idx.searchTopK(query, k, admit), nil
}

// SearchNamedSparseDocs is SearchNamedSparse returning Documents carrying the
// SHARED per-point payload (the docs/scroll result shape), mirroring
// SearchNamedDocs. Distance is left 0 (sparse search ranks by Score, not a
// distance metric); Score is the sparse dot product.
func (nc *NamedCollection) SearchNamedSparseDocs(space string, query *SparseVector, k int, filter Filter) ([]Document, error) {
	if query != nil {
		if err := query.Validate(); err != nil {
			return nil, err
		}
	}
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	sp, ok := nc.sparseSpaces[space]
	if !ok {
		if _, isDense := nc.indexes[space]; isDense {
			return nil, ErrSpaceModalityMismatch
		}
		return nil, ErrUnknownVectorName
	}
	admit, err := nc.sparseAdmitLocked(filter)
	if err != nil {
		return nil, err
	}
	res := sp.idx.searchTopK(query, k, admit)
	now := nc.nowMs()
	docs := make([]Document, 0, len(res))
	for _, r := range res {
		docs = append(docs, Document{
			ID:       r.ID,
			Distance: r.Distance,
			Score:    r.Score,
			Metadata: liveMetaMap(nc.meta[r.ID], nc.keyTTL[r.ID], now),
		})
	}
	return docs, nil
}

// sparseAdmitLocked compiles filter and returns the per-id admit predicate the
// sparse searchTopK gates each candidate by: the point must be live (present +
// not point-TTL-expired) AND its live shared payload (per-key-TTL-expired keys
// dropped) must satisfy the compiled filter. It is the sparse-lane analogue of
// the live-meta predicate gate hnsw.SearchFilteredWith applies in the dense lane,
// reusing nc.metaOf() so both lanes see ONE clock snapshot per search. A nil
// (zero) filter yields a liveness-only admit. Caller holds nc.mu (read).
func (nc *NamedCollection) sparseAdmitLocked(filter Filter) (func(id uint64) bool, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	live := nc.metaOf() // one now() snapshot for the whole search (mirrors SearchNamed)
	return func(id uint64) bool {
		if !nc.liveLocked(id) {
			return false
		}
		if pred == nil {
			return true
		}
		return pred(live(id))
	}, nil
}

// namedHybridK returns the per-lane candidate-pool size from a HybridOpts knob
// (DenseK / SparseK), defaulting to max(k, 50) exactly like the dense
// hnsw.buildLanes — so the named hybrid's lane sizing matches the dense oracle.
func namedHybridK(knob, k int) int {
	if knob > 0 {
		return knob
	}
	if k < 50 {
		return 50
	}
	return k
}

// namedHybridLanesLocked builds the dense and sparse candidate lanes for a named
// cross-space hybrid, UNFUSED, under nc.mu (read). It validates the modality of
// BOTH spaces up front (fail loud) and applies the SAME live-meta/TTL/filter admit
// gate to each lane (the dense lane via SearchNamed's filter-first / SearchFilteredWith
// path; the sparse lane via sparseAdmitLocked → searchTopK), so a filtered/expired
// point is consistently excluded from both. An empty/absent dense query yields a nil
// dense lane (sparse-only); an empty/absent sparse query yields a nil sparse lane
// (dense-only) — the caller (NamedHybrid / the fan-out coordinator) collapses the
// single-lane case. denseRes is ascending by Distance; sparseRes is descending by
// Score, each pooled to DenseK/SparseK (default max(k,50)). Caller holds nc.mu (read).
func (nc *NamedCollection) namedHybridLanesLocked(denseSpace string, denseQ []float32, sparseSpace string, sparseQ *SparseVector, k int, opts HybridOpts) (denseRes, sparseRes []Result, err error) {
	if k <= 0 {
		return nil, nil, nil
	}
	// Validate denseSpace IS dense and sparseSpace IS sparse (fail loud), even when
	// that lane's query is empty — a mistyped space name is a request error, not a
	// silent single-lane degradation.
	denseIdx, ok := nc.indexes[denseSpace]
	if !ok {
		if _, isSparse := nc.sparseSpaces[denseSpace]; isSparse {
			return nil, nil, ErrSpaceModalityMismatch // dense lane points at a sparse space
		}
		return nil, nil, ErrUnknownVectorName
	}
	sp, ok := nc.sparseSpaces[sparseSpace]
	if !ok {
		if _, isDense := nc.indexes[sparseSpace]; isDense {
			return nil, nil, ErrSpaceModalityMismatch // sparse lane points at a dense space
		}
		return nil, nil, ErrUnknownVectorName
	}
	if sparseQ != nil {
		if verr := sparseQ.Validate(); verr != nil {
			return nil, nil, verr
		}
	}

	// Dense lane: reuse the SearchNamed path (filter-first acceleration, else the
	// predicate-eval graph search), pooled to denseK. Skipped (nil) for an empty query.
	if len(denseQ) > 0 {
		denseK := namedHybridK(opts.DenseK, k)
		pred, cerr := CompileFilter(opts.Filter)
		if cerr != nil {
			return nil, nil, cerr
		}
		if ffRes, applied, ferr := nc.searchNamedFilterFirst(denseIdx, denseQ, denseK, opts.Filter, pred); applied {
			if ferr != nil {
				return nil, nil, ferr
			}
			denseRes = ffRes
		} else {
			denseRes, err = denseIdx.SearchFilteredWith(nil, denseQ, denseK, opts.Filter, nc.metaOf())
			if err != nil {
				return nil, nil, err
			}
		}
	}

	// Sparse lane: the id-keyed inverted index top-k, gated by the SAME live-meta /
	// TTL / filter admit rule the dense lane used (sparseAdmitLocked reuses nc.metaOf()
	// so both lanes see one clock snapshot). Skipped (nil) for an empty query.
	if sparseQ != nil && !sparseQ.IsZero() {
		sparseK := namedHybridK(opts.SparseK, k)
		admit, aerr := nc.sparseAdmitLocked(opts.Filter)
		if aerr != nil {
			return nil, nil, aerr
		}
		sparseRes = sp.idx.searchTopK(sparseQ, sparseK, admit)
	}
	return denseRes, sparseRes, nil
}

// NamedHybrid fuses a DENSE named space and a SPARSE named space into the top-k.
// denseSpace MUST be a configured dense space and sparseSpace a configured sparse
// space (else ErrUnknownVectorName / ErrSpaceModalityMismatch — fail loud). It runs
// the dense KNN lane (denseSpace, via the SearchNamed path) and the sparse
// dot-product lane (sparseSpace, via searchTopK), BOTH under the SAME live-meta /
// TTL / filter admit gate, then combines them with the reused vector.Fuse
// (RRF default / weighted by opts.Method) truncated to k. Fusion is NOT
// reimplemented here — vector.Fuse is the single source of truth (shared with the
// dense hybrid + the cross-partition coordinator). Single-lane degradation mirrors
// the dense hnsw.HybridSearch: an empty/absent sparse query returns the dense lane
// alone; an empty/absent dense query returns the sparse lane alone (each truncated
// to k). The cross-space analogue of hnsw.HybridSearch.
func (nc *NamedCollection) NamedHybrid(denseSpace string, denseQ []float32, sparseSpace string, sparseQ *SparseVector, k int, opts HybridOpts) ([]Result, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	denseRes, sparseRes, err := nc.namedHybridLanesLocked(denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
	if err != nil {
		return nil, err
	}
	// Single-lane degradation (mirror hnsw.HybridSearch:1748-1759). A lane with no
	// query is empty; return the other lane truncated to k instead of fusing.
	sparseEmpty := sparseQ == nil || sparseQ.IsZero()
	switch {
	case sparseEmpty:
		if len(denseRes) > k {
			denseRes = denseRes[:k]
		}
		return denseRes, nil
	case len(denseQ) == 0:
		if len(sparseRes) > k {
			sparseRes = sparseRes[:k]
		}
		return sparseRes, nil
	}
	return Fuse(denseRes, sparseRes, opts.Method, opts.Alpha, opts.RRFK, k), nil
}

// NamedHybridLanes builds the dense and sparse candidate lanes for a named
// cross-space hybrid WITHOUT fusing them, so a cross-partition coordinator can
// union the per-partition lanes and fuse ONCE globally (exact partition fan-out,
// mirror hnsw.HybridLanes). denseRes is ascending by Distance; sparseRes is
// descending by Score. Modality validation + the shared admit gate + the DenseK/
// SparseK lane sizing are identical to NamedHybrid (they share
// namedHybridLanesLocked); only the final fusion is deferred to the coordinator.
func (nc *NamedCollection) NamedHybridLanes(denseSpace string, denseQ []float32, sparseSpace string, sparseQ *SparseVector, k int, opts HybridOpts) (dense, sparse []Result, err error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return nc.namedHybridLanesLocked(denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
}

// Delete removes point id from EVERY named sub-index and from the shared
// payload/ttl/live-set, returning whether it existed.
func (nc *NamedCollection) Delete(id uint64) (bool, error) {
	removed, _, err := nc.DeleteCAS(id, CASCond{})
	return removed, err
}

// DeleteCAS is Delete with an optimistic-CAS precondition (CASCond{} = no
// precondition). On a mismatch it returns ErrVersionConflict and removed=false
// with no mutation. A no-CAS Delete drops the point and its version (an absent
// point reads version 0; a fresh reinsert restarts at 1). The check is atomic
// under nc.mu (mirrors dense Collection.DeleteCAS).
func (nc *NamedCollection) DeleteCAS(id uint64, cas CASCond) (removed bool, prevVersion uint64, err error) {
	if nc.wal == nil {
		return nc.deleteLocked(id, cas)
	}
	// opMu is held across {apply + WAL WRITE} only, scoped to a closure (see
	// InsertCASKeyTTL) so a panic inside deleteLocked or the WAL append still
	// unlocks opMu via the deferred Unlock. The durability wait runs OUTSIDE
	// opMu so a deleting writer never becomes the fsync leader while holding the
	// collection's op lock — concurrent deletes group-commit like inserts.
	var seq uint64
	err = func() error {
		nc.opMu.Lock()
		defer nc.opMu.Unlock()
		var derr error
		removed, prevVersion, derr = nc.deleteLocked(id, cas)
		if derr != nil {
			return derr
		}
		if removed {
			// Propagate the WAL append error (mirrors dense DeleteCAS): idempotent
			// replay tolerates torn duplicates, but a durability failure must fail
			// the delete rather than falsely ack a point that may resurrect on crash.
			var serr error
			seq, serr = nc.wal.appendNamedDeleteStaged(id)
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
	if werr := nc.wal.commitWaitStaged(seq); werr != nil {
		return false, 0, werr
	}
	return removed, prevVersion, nil
}

// deleteLocked removes id from every space + the shared maps under nc.mu,
// returning whether it was a live id and its prior version. The CAS check
// (against the current version, 0 if absent) is atomic with the removal. Split
// out so the WAL append runs after nc.mu is released but still under opMu.
func (nc *NamedCollection) deleteLocked(id uint64, cas CASCond) (bool, uint64, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	prev := nc.version[id]
	if err := cas.check(prev); err != nil {
		return false, 0, err // CAS mismatch: no mutation
	}
	_, existed := nc.ids[id]
	for _, idx := range nc.indexes {
		_, _ = idx.Delete(id, CASCond{}) // best-effort across spaces; absent in a space is a no-op
	}
	for _, sp := range nc.sparseSpaces {
		sp.drop(id) // drop from every sparse space's vecs + inverted index (no-op if absent)
	}
	delete(nc.meta, id)
	delete(nc.ttl, id)
	delete(nc.keyTTL, id)
	delete(nc.ids, id)
	delete(nc.version, id)         // delete drops the version (a reinsert restarts at 1)
	nc.payloadIdx.reindex(id, nil) // nil meta = pure removal from the filter-first index
	if existed {
		nc.bumpData() // live id set shrank: invalidate the order_by snapshot
	}
	return existed, prev, nil
}

// liveLocked reports whether id is a live point in the shared namespace: present
// in the authoritative id set AND not TTL-expired (mirror the ScrollDocs/Insert
// liveness gate). Caller holds nc.mu (read or write). The named family has no
// per-space tombstones — the shared ids/ttl maps are authoritative.
func (nc *NamedCollection) liveLocked(id uint64) bool {
	return nc.liveLockedAt(id, nc.nowMs())
}

// liveLockedAt is liveLocked judging point-TTL expiry against the caller-supplied
// `now` (unix millis), so a replicated write's liveness gate is decided on the
// leader apply stamp identically on every replica (#4 vector TTL determinism).
// Caller holds nc.mu.
func (nc *NamedCollection) liveLockedAt(id uint64, now int64) bool {
	if _, ok := nc.ids[id]; !ok {
		return false
	}
	if dl := nc.ttl[id]; dl != 0 && dl <= now {
		return false
	}
	return true
}

// Get retrieves a live point by id: a map of its per-space DEEP-COPIED vectors
// (only the spaces the point actually populated appear — an omitted space is
// absent from the map), the shared per-point payload (deep-copied), plus the
// remaining TTL. ok is false for an absent or TTL-expired point (mirror the
// ScrollDocs liveness gate; the named family has no tombstones — the shared ids
// set is authoritative). The returned vectors/payload are owned by the caller
// (mutating them never corrupts the sub-arenas or the shared meta map). For a
// cosine-metric space the returned vector is the NORMALIZED vector (Insert
// normalizes on the way in). ttl is the remaining duration to the shared
// deadline (0 = no expiry). Lock order: nc.mu (read) outer, then each sub-index's
// idx.mu (read) for the arena read — the same outer→inner order SearchNamed /
// ScrollDocs use.
func (nc *NamedCollection) Get(id uint64) (vectors map[string][]float32, payload Metadata, ttl time.Duration, version uint64, ok bool) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	if !nc.liveLocked(id) {
		return nil, nil, 0, 0, false
	}
	version = nc.version[id]
	vectors = make(map[string][]float32)
	for name, idx := range nc.indexes {
		// Resolve the point's vector in this space via the interface (locks the
		// sub-index's own mu internally, returns a COPY — exact arena floats when
		// present, reconstructed for an IVF-PQ space with dropped floats). A point
		// may omit a space; vecsForIDs simply omits the id then.
		if vm := idx.vecsForIDs([]uint64{id}); len(vm) > 0 {
			if v, present := vm[id]; present {
				vectors[name] = v
			}
		}
	}
	// Drop per-key-TTL-expired keys, then deep-copy so the caller owns the payload
	// (liveMetaMap may alias nc.meta on the no-expiry fast path).
	payload = cloneMeta(liveMetaMap(nc.meta[id], nc.keyTTL[id], nc.nowMs()))
	if dl := nc.ttl[id]; dl != 0 {
		if now := nc.nowMs(); dl > now {
			ttl = time.Duration(dl-now) * time.Millisecond
		}
	}
	return vectors, payload, ttl, version, true
}

// SetPayload MERGES patch into id's shared payload (patch keys overwrite or add,
// other keys retained) and stores the result in nc.meta[id]. The named family has
// NO payload index (filtering is predicate-eval against the shared payload) and NO
// WAL (snapshot-durable by design) — this is a pure map update under the write
// lock, with NO reindex. Returns ErrIDNotFound for an absent/expired point. Does
// not change any vector or the TTL.
func (nc *NamedCollection) SetPayload(id uint64, patch Metadata, keyTTLMs map[string]int64) error {
	_, err := nc.SetPayloadCAS(id, patch, keyTTLMs, CASCond{})
	return err
}

// SetPayloadCAS is SetPayload with an optimistic-CAS precondition (CASCond{} = no
// precondition). It returns the resulting version on success and ErrVersionConflict
// (no mutation) on a mismatch. ErrIDNotFound for an absent/expired point.
func (nc *NamedCollection) SetPayloadCAS(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.setPayloadLocked(id, patch, keyTTLMs, cas)
	}, id)
}

// SetPayloadCASAt is SetPayloadCAS judging the per-key deadline computation AND the
// dead-point liveness gate against the EXPLICIT leader apply stamp nowMs (#4 vector
// TTL determinism).
func (nc *NamedCollection) SetPayloadCASAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.setPayloadLockedAt(id, patch, keyTTLMs, cas, nowMs)
	}, id)
}

func (nc *NamedCollection) setPayloadLocked(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	return nc.setPayloadLockedAt(id, patch, keyTTLMs, cas, nc.nowMs())
}

func (nc *NamedCollection) setPayloadLockedAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, now int64) (Metadata, map[string]int64, uint64, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if !nc.liveLockedAt(id, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(nc.version[id]); err != nil {
		return nil, nil, 0, err // CAS mismatch: no mutation, no bump
	}
	merged := cloneMeta(nc.meta[id])
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
	ke := applyKeyTTLMs(cloneKeyTTL(nc.keyTTL[id]), keyTTLMs, merged, now)
	ke = pruneKeyTTL(ke, merged)
	nc.meta[id] = merged
	nc.payloadIdx.reindex(id, merged) // resulting payload may add/remove indexed fields
	if ke == nil {
		delete(nc.keyTTL, id)
	} else {
		nc.keyTTL[id] = ke
	}
	v := nc.version[id] + 1
	nc.version[id] = v
	nc.bumpData() // payload-value change: invalidate the order_by snapshot
	return merged, ke, v, nil
}

// logPayloadOp runs a payload mutator (apply, which captures the RESULTING full
// payload + absolute per-key deadlines under nc.mu) and, on a WAL-mode
// collection, appends ONE op-agnostic namedSetPayload record with that resulting
// state — the dense resulting-payload collapse (set/overwrite/delete-keys/clear
// all reduce to "this is the new payload"). The whole apply+append runs under
// opMu so a concurrent Flush can't interleave. The deadlines are logged VERBATIM
// (absolute unix-ms) so replay is time-stable.
func (nc *NamedCollection) logPayloadOp(apply func() (Metadata, map[string]int64, uint64, error), id uint64) (uint64, error) {
	if nc.wal == nil {
		_, _, version, err := apply()
		return version, err
	}
	// {apply + WAL WRITE} under opMu in a panic-safe closure; the durability wait
	// runs outside so concurrent payload writers group-commit instead of each
	// paying a serialized fsync under the collection's op lock.
	var seq, version uint64
	err := func() error {
		nc.opMu.Lock()
		defer nc.opMu.Unlock()
		meta, ke, v, err := apply()
		if err != nil {
			return err
		}
		version = v
		seq, err = nc.wal.appendNamedSetPayloadStaged(id, meta, keyTTLToU64(ke), v)
		return err
	}()
	if err != nil {
		// The apply either failed (version still 0) or succeeded and only the WAL
		// write failed — the pre-split code returned the applied version alongside
		// the append error, so keep returning it.
		return version, err
	}
	return version, nc.wal.commitWaitStaged(seq)
}

// keyTTLToU64 converts named's absolute int64 unix-ms deadlines to the uint64
// form the shared WAL keyExpires codec uses (deadlines are non-negative). Returns
// nil for an empty map (the cheap append path).
func keyTTLToU64(ke map[string]int64) map[string]uint64 {
	if len(ke) == 0 {
		return nil
	}
	out := make(map[string]uint64, len(ke))
	for k, dl := range ke {
		out[k] = uint64(dl) //nolint:gosec
	}
	return out
}

// restorePayload sets id's shared payload to meta and its per-key deadlines to
// keyExpires VERBATIM (absolute unix-ms), with NO recompute — the WAL-replay
// analogue of dense hnsw.RestorePayload. SetPayload recomputes relative TTL to
// now+ttl; replay must instead restore the exact absolute deadlines the original
// op produced, so a pending per-key TTL survives a crash time-stable. Idempotent
// on top of the snapshot checkpoint; it does NOT gate on liveness (replay
// re-applies a payload op that the snapshot may or may not already reflect).
func (nc *NamedCollection) restorePayload(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	// A payload op only ever targeted a live point; if replay reaches it before the
	// point's Insert record (impossible in a well-formed log) the id-set entry is
	// absent — restore the payload anyway so a later Insert/replay converges.
	nc.meta[id] = meta
	nc.payloadIdx.reindex(id, meta) // keep the filter-first index in sync (replay path)
	nc.bumpData()                   // WAL-replayed payload-value change: invalidate the order_by snapshot
	// Restore the version VERBATIM (the WAL logged the resulting version). A 0
	// version (an old record predating the version block) leaves the existing
	// version untouched, so a prior Insert's version is not clobbered.
	if version != 0 {
		nc.version[id] = version
	}
	if len(keyExpires) == 0 {
		delete(nc.keyTTL, id)
		return
	}
	ke := make(map[string]int64, len(keyExpires))
	for k, dl := range keyExpires {
		ke[k] = int64(dl) //nolint:gosec
	}
	nc.keyTTL[id] = ke
}

// OverwritePayload REPLACES id's entire shared payload with meta (a nil/empty meta
// clears it). Pure map update — no reindex, no WAL. Returns ErrIDNotFound for an
// absent/expired point. Does not change any vector or the TTL.
func (nc *NamedCollection) OverwritePayload(id uint64, meta Metadata, keyTTLMs map[string]int64) error {
	_, err := nc.OverwritePayloadCAS(id, meta, keyTTLMs, CASCond{})
	return err
}

// OverwritePayloadCAS is OverwritePayload with an optimistic-CAS precondition.
// Returns the resulting version; ErrVersionConflict (no mutation) on a mismatch.
func (nc *NamedCollection) OverwritePayloadCAS(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.overwritePayloadLocked(id, meta, keyTTLMs, cas)
	}, id)
}

// OverwritePayloadCASAt is OverwritePayloadCAS judging the per-key deadline
// computation AND the dead-point liveness gate against the leader apply stamp nowMs
// (#4 vector TTL determinism).
func (nc *NamedCollection) OverwritePayloadCASAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.overwritePayloadLockedAt(id, meta, keyTTLMs, cas, nowMs)
	}, id)
}

func (nc *NamedCollection) overwritePayloadLocked(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	return nc.overwritePayloadLockedAt(id, meta, keyTTLMs, cas, nc.nowMs())
}

func (nc *NamedCollection) overwritePayloadLockedAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, now int64) (Metadata, map[string]int64, uint64, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if !nc.liveLockedAt(id, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(nc.version[id]); err != nil {
		return nil, nil, 0, err
	}
	newMeta := cloneMeta(meta)
	// Overwrite REPLACES the per-key deadline set: deadlines come only from the new
	// keyTTLMs (recomputed against now); any prior deadline is dropped.
	ke := pruneKeyTTL(applyKeyTTLMs(nil, keyTTLMs, newMeta, now), newMeta)
	nc.meta[id] = newMeta
	nc.payloadIdx.reindex(id, newMeta) // overwrite replaces the indexed fields wholesale
	if ke == nil {
		delete(nc.keyTTL, id)
	} else {
		nc.keyTTL[id] = ke
	}
	v := nc.version[id] + 1
	nc.version[id] = v
	nc.bumpData() // payload-value change: invalidate the order_by snapshot
	return newMeta, ke, v, nil
}

// DeletePayloadKeys removes the listed keys from id's shared payload (absent keys
// = no-op). Pure map update, no reindex. WAL-logged (op-agnostic resulting
// payload) on a WAL-mode collection. Returns ErrIDNotFound for an absent/expired
// point. Does not change any vector or the TTL.
func (nc *NamedCollection) DeletePayloadKeys(id uint64, keys []string) error {
	_, err := nc.DeletePayloadKeysCAS(id, keys, CASCond{})
	return err
}

// DeletePayloadKeysCAS is DeletePayloadKeys with an optimistic-CAS precondition.
// Returns the resulting version; ErrVersionConflict (no mutation) on a mismatch.
func (nc *NamedCollection) DeletePayloadKeysCAS(id uint64, keys []string, cas CASCond) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.deletePayloadKeysLocked(id, keys, cas)
	}, id)
}

// DeletePayloadKeysCASAt is DeletePayloadKeysCAS judging the dead-point liveness gate
// against the leader apply stamp nowMs (dropping keys computes no deadline) (#4
// vector TTL determinism).
func (nc *NamedCollection) DeletePayloadKeysCASAt(id uint64, keys []string, cas CASCond, nowMs int64) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.deletePayloadKeysLockedAt(id, keys, cas, nowMs)
	}, id)
}

func (nc *NamedCollection) deletePayloadKeysLocked(id uint64, keys []string, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	return nc.deletePayloadKeysLockedAt(id, keys, cas, nc.nowMs())
}

func (nc *NamedCollection) deletePayloadKeysLockedAt(id uint64, keys []string, cas CASCond, now int64) (Metadata, map[string]int64, uint64, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if !nc.liveLockedAt(id, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(nc.version[id]); err != nil {
		return nil, nil, 0, err
	}
	newMeta := cloneMeta(nc.meta[id])
	ke := cloneKeyTTL(nc.keyTTL[id])
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
	nc.meta[id] = newMeta
	nc.payloadIdx.reindex(id, newMeta) // dropped keys may remove indexed fields
	if ke == nil {
		delete(nc.keyTTL, id)
	} else {
		nc.keyTTL[id] = ke
	}
	v := nc.version[id] + 1
	nc.version[id] = v
	nc.bumpData() // payload-value change: invalidate the order_by snapshot
	return newMeta, ke, v, nil
}

// ClearPayload removes ALL of id's shared payload (payload → nil). Pure map
// update, no reindex. WAL-logged (resulting nil payload, no per-key deadlines) on
// a WAL-mode collection. Returns ErrIDNotFound for an absent/expired point. Does
// not change any vector or the TTL.
func (nc *NamedCollection) ClearPayload(id uint64) error {
	_, err := nc.ClearPayloadCAS(id, CASCond{})
	return err
}

// ClearPayloadCAS is ClearPayload with an optimistic-CAS precondition. Returns
// the resulting version; ErrVersionConflict (no mutation) on a mismatch.
func (nc *NamedCollection) ClearPayloadCAS(id uint64, cas CASCond) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.clearPayloadLocked(id, cas)
	}, id)
}

// ClearPayloadCASAt is ClearPayloadCAS judging the dead-point liveness gate against
// the leader apply stamp nowMs (#4 vector TTL determinism).
func (nc *NamedCollection) ClearPayloadCASAt(id uint64, cas CASCond, nowMs int64) (uint64, error) {
	return nc.logPayloadOp(func() (Metadata, map[string]int64, uint64, error) {
		return nc.clearPayloadLockedAt(id, cas, nowMs)
	}, id)
}

func (nc *NamedCollection) clearPayloadLocked(id uint64, cas CASCond) (Metadata, map[string]int64, uint64, error) {
	return nc.clearPayloadLockedAt(id, cas, nc.nowMs())
}

func (nc *NamedCollection) clearPayloadLockedAt(id uint64, cas CASCond, now int64) (Metadata, map[string]int64, uint64, error) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if !nc.liveLockedAt(id, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(nc.version[id]); err != nil {
		return nil, nil, 0, err
	}
	nc.meta[id] = nil
	nc.payloadIdx.reindex(id, nil) // cleared payload carries no indexed fields
	delete(nc.keyTTL, id)          // clearing the payload clears all per-key deadlines
	v := nc.version[id] + 1
	nc.version[id] = v
	nc.bumpData() // payload-value change: invalidate the order_by snapshot
	return nil, nil, v, nil
}

// ScrollDocs lists live points matching filter (a zero filter matches all), up
// to limit, each carrying its SHARED payload (payload-only; no vectors). It
// iterates the AUTHORITATIVE shared id set (so a point counts even if it omits
// some named spaces) and predicate-evals the compiled filter against the shared
// payload; ttl-expired points are excluded. Distance/Score are 0 (no query).
// limit <= 0 means no cap. Iteration order is unspecified.
func (nc *NamedCollection) ScrollDocs(filter Filter, limit int) ([]Document, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err // fail loud on a malformed filter (matches the op handler's contract)
	}
	docs, _, _ := nc.scrollPage(filter, pred, nil, 0, 0, false, limit)
	return docs, nil
}

// ScrollDocsPage is the cursor-aware named scroll: up to limit live points (with
// shared payload) matching filter whose id is strictly greater than afterID when
// hasAfter (otherwise from the smallest id), id-ASCENDING. Returns docs, nextAfter
// (largest id returned), and hasMore (page filled limit AND a further id remains).
// The named-family analogue of Collection.ScrollDocsPage. See NamedCollection.scrollPage.
func (nc *NamedCollection) ScrollDocsPage(filter Filter, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, 0, false, err // fail loud on a malformed filter
	}
	docs, nextAfter, hasMore = nc.scrollPage(filter, pred, nil, afterID, 0, hasAfter, limit)
	return docs, nextAfter, hasMore, nil
}

// ScrollDocsPageOrder is the order_by-aware named cursor scroll: up to limit live
// points matching filter, ordered by the order_by field's (value, id) total order
// (see OrderBy / OrderLess). Points whose order field is missing/non-numeric are
// EXCLUDED. afterKey/afterID is the resume cursor's (value, id) position (the page
// returns rows strictly after it); on page 1 (hasAfter=false) order.StartFrom (when
// set) is the inclusive starting value bound. The order field travels in each
// Document's Metadata so the coordinator can read the last doc's order value for the
// v2 next-cursor. order == nil falls back to the id-ascending ScrollDocsPage path.
// The named-family analogue of Collection.ScrollDocsPageOrder.
func (nc *NamedCollection) ScrollDocsPageOrder(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool, err error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, 0, false, err // fail loud on a malformed filter
	}
	docs, nextAfter, hasMore = nc.scrollPage(filter, pred, order, afterID, afterKey, hasAfter, limit)
	return docs, nextAfter, hasMore, nil
}

// scrollPage is the named family's deterministic id-ASCENDING scroll primitive,
// the analogue of hnsw.scrollPage for the shared-payload named id set. The named
// family keeps its authoritative live set in nc.ids (NOT an hnsw arena), so the
// ordered walk sorts nc.ids ascending and applies the TTL + shared-payload filter
// gate the legacy scroll used. pred is the precompiled filter predicate (nil ⇒
// match all); the caller compiles it so a malformed filter fails loud at the edge.
// It backs both the no-cursor legacy ScrollDocs (afterID=0, hasAfter=false ⇒
// smallest-id `limit`) and the cursor fan-out path. Returns the collected
// docs, nextAfter (largest id collected), and hasMore (true iff stopped at `limit`
// with a further id remaining). limit <= 0 means no cap.
//
// The sort is per-call (named has no cached snapshot): the named id set
// is the small shared-payload map, and the named families are not on the dense
// hot scroll path. A cached snapshot for named is a documented follow-up if its
// scroll volume warrants it.
//
// When order != nil this swaps the id-ascending comparator for the (value, id) total
// order over the SAME live shared-payload view (ttl + per-key-TTL gated), EXCLUDES
// points whose order field is missing/non-numeric (Qdrant default), seeks STRICTLY
// PAST the (afterKey, afterID) cursor — or past order.StartFrom on page 1 — and walks
// limit. nil order ⇒ the existing id-ascending path VERBATIM (zero overhead). The
// named id set already sorts per-call, so order_by only swaps the comparator + adds
// the (value, id) seek.
func (nc *NamedCollection) scrollPage(filter Filter, pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
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
		nc.mu.RLock()
		if rows, ok := nc.filterFirstOrderRowsLocked(filter, pred, order); ok {
			docs, nextAfter, hasMore = nc.collectOrderedLocked(rows, pred, order, afterID, afterKey, hasAfter, limit, nc.nowMs())
			nc.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		// Warm path: a cached (field, direction) snapshot fresh at the current
		// dataVersion ⇒ walk under the read lock (rows immutable; a rebuild replaces
		// the pointer wholesale). Miss ⇒ relock for the cold rebuild (double-checked).
		if snap := nc.orderSnapWarmLocked(order); snap != nil {
			docs, nextAfter, hasMore = nc.collectOrderedLocked(snap.rows, pred, order, afterID, afterKey, hasAfter, limit, nc.nowMs())
			nc.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		nc.mu.RUnlock()
		nc.mu.Lock()
		defer nc.mu.Unlock()
		// Re-test filter-first under the write lock (the index may have changed in the
		// unlock/relock gap); build narrowed fresh, else the cached full snapshot.
		if rows, ok := nc.filterFirstOrderRowsLocked(filter, pred, order); ok {
			return nc.collectOrderedLocked(rows, pred, order, afterID, afterKey, hasAfter, limit, nc.nowMs())
		}
		snap := nc.orderSnapLocked(order)
		return nc.collectOrderedLocked(snap.rows, pred, order, afterID, afterKey, hasAfter, limit, nc.nowMs())
	}
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	now := nc.nowMs()
	// The full authoritative live id set, ascending. hasMore is ALWAYS computed
	// against this set so the boundary is byte-identical whether or not filter-first
	// narrowed the walked set (a filtered page can be followed by a trailing empty
	// page in BOTH paths).
	fullIDs := make([]uint64, 0, len(nc.ids))
	for id := range nc.ids {
		fullIDs = append(fullIDs, id)
	}
	sort.Slice(fullIDs, func(i, j int) bool { return fullIDs[i] < fullIDs[j] })
	// Filter-first narrowing: when a filter is present and the id-keyed payload index
	// narrows it to a selective candidate SUPERSET, walk that id-sorted superset
	// (TTL + per-key-TTL + predicate rechecked below) instead of the full set. The
	// recheck + the full-set hasMore make the page byte-identical to the full walk;
	// ok=false ⇒ the full nc.ids walk (non-accelerable / non-selective / no-filter).
	ids := fullIDs
	usedFF := false
	if cands, ok := nc.filterFirstScrollCandsLocked(filter, pred); ok {
		ids = cands
		usedFF = true
	}
	start := 0
	if hasAfter {
		start = sort.Search(len(ids), func(i int) bool { return ids[i] > afterID })
	}
	for i := start; i < len(ids); i++ {
		id := ids[i]
		if dl := nc.ttl[id]; dl != 0 && dl <= now {
			continue // ttl-expired
		}
		// Drop per-key-TTL-expired keys before BOTH the predicate eval and the
		// emitted payload, so a filter on an expired key never matches and the
		// scrolled payload omits it.
		payload := liveMetaMap(nc.meta[id], nc.keyTTL[id], now)
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
// Returns (nil, false) to fall back to the full nc.ids predicate-eval walk
// (no-filter / non-accelerable / non-selective). The candidate ids are a SUPERSET —
// the caller's TTL + per-key-TTL + predicate recheck drops over-cover, so the page is
// byte-identical to the full walk. Must hold nc.mu (read). The named-family analogue
// of hnsw.filterFirstScrollCands; id-keyed so candidates already returns ids.
func (nc *NamedCollection) filterFirstScrollCandsLocked(filter Filter, pred Predicate) ([]uint64, bool) {
	if pred == nil || nc.payloadIdx == nil {
		return nil, false // no filter / no index -> full walk unchanged
	}
	threshold := defaultFilterFirstThreshold
	cands, ok := nc.payloadIdx.candidates(filter, threshold)
	if !ok || len(cands) > threshold {
		return nil, false
	}
	// Intersect with the AUTHORITATIVE live set (nc.ids) so a stale index posting can
	// never emit a non-live point — exactly the ids the full nc.ids walk would
	// consider, just narrowed. The TTL + per-key-TTL + predicate recheck (in the
	// caller) drops the rest, so the page is byte-identical to the full walk.
	ids := make([]uint64, 0, len(cands))
	for _, id := range cands {
		if _, live := nc.ids[id]; live {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids, true
}

// orderSnapWarmLocked returns the cached named order snapshot for this (field,
// direction) IF it exists and is fresh at the current dataVersion; nil otherwise
// (caller falls to the cold rebuild). Must hold nc.mu (R or W). The returned
// *orderSnap's rows slice is immutable (a rebuild replaces the pointer), so a warm
// RLock reader is race-safe.
func (nc *NamedCollection) orderSnapWarmLocked(order *OrderBy) *orderSnap {
	if snap, ok := nc.orderSnaps[orderSnapCacheKey(order)]; ok && snap.ver == nc.dataVersion {
		return snap
	}
	return nil
}

// orderSnapLocked returns a fresh named order snapshot for (field, direction),
// rebuilding it if stale or absent. Double-checked: the warm path may have lost the
// version race and relocked, but another goroutine may have rebuilt in the gap, so
// re-test the version before rebuilding. Must hold nc.mu (WRITE). The snapshot is
// FILTER-INDEPENDENT — it caches EVERY live id that HAS the order field, sorted by
// (value, id); the per-query filter + TTL gate run later in collectOrderedLocked.
func (nc *NamedCollection) orderSnapLocked(order *OrderBy) *orderSnap {
	key := orderSnapCacheKey(order)
	if snap, ok := nc.orderSnaps[key]; ok && snap.ver == nc.dataVersion {
		return snap // rebuilt by another goroutine in the unlock/relock gap
	}
	rows := nc.buildOrderRowsLocked(order)
	nc.orderSeq++
	snap := &orderSnap{ver: nc.dataVersion, seq: nc.orderSeq, rows: rows}
	nc.orderSnaps[key] = snap
	if len(nc.orderSnaps) > orderCacheCap {
		evictOldestOrderSnap(nc.orderSnaps)
	}
	nc.orderRebuilds++
	return snap
}

// buildOrderRowsLocked collects every live id that HAS the order field into a
// (value, id)-sorted slice. TTL-expired-but-unswept ids are NOT excluded here (the
// named family ages TTL lazily without a mutation/bump — same as the dense id
// scrollSnap); the walk's TTL/per-key-TTL gate drops them. The per-query FILTER is
// deliberately NOT applied here so the snapshot is reusable across different
// filters. Missing / non-numeric order field ⇒ EXCLUDED (Qdrant default). The order
// value is read over the per-key-TTL-gated payload view (consistent with the walk).
// Must hold nc.mu (WRITE).
func (nc *NamedCollection) buildOrderRowsLocked(order *OrderBy) []OrderedID {
	now := nc.nowMs()
	rows := make([]OrderedID, 0, len(nc.ids))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for id := range nc.ids {
			payload := liveMetaMap(nc.meta[id], nc.keyTTL[id], now)
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
	for id := range nc.ids {
		payload := liveMetaMap(nc.meta[id], nc.keyTTL[id], now)
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
// payload-index candidate SUPERSET (∩ live nc.ids) for filter, or (nil, false) when
// filter-first order narrowing does not apply (no filter / no index / non-accelerable /
// non-selective). The candidate ids are a superset of the matching ids; the per-row
// field-presence EXCLUDE here + the per-row TTL/per-key-TTL + predicate recheck in
// collectOrderedLocked make the narrowed rows EXACTLY the field-present matches in the
// SAME value-order as the full snapshot, so the page (docs / nextAfter / cursor) is
// byte-identical to the predicate-eval order page. NOT cached (the orderSnaps key is
// filter-independent; a narrowed snapshot must never be stored there). Must hold nc.mu
// (R or W). The named analogue of hnsw.filterFirstOrderRowsLocked (id-keyed candidates).
func (nc *NamedCollection) filterFirstOrderRowsLocked(filter Filter, pred Predicate, order *OrderBy) ([]OrderedID, bool) {
	if pred == nil || nc.payloadIdx == nil {
		return nil, false
	}
	threshold := defaultFilterFirstThreshold
	cands, ok := nc.payloadIdx.candidates(filter, threshold)
	if !ok || len(cands) > threshold {
		return nil, false
	}
	now := nc.nowMs()
	rows := make([]OrderedID, 0, len(cands))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for _, id := range cands {
			if _, live := nc.ids[id]; !live {
				continue // stale index posting -> not live
			}
			payload := liveMetaMap(nc.meta[id], nc.keyTTL[id], now)
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
		if _, live := nc.ids[id]; !live {
			continue // stale index posting -> not live
		}
		payload := liveMetaMap(nc.meta[id], nc.keyTTL[id], now)
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

// collectOrderedLocked is the named family's order_by walk: it seeks the cached
// (value, id) sorted rows past the (afterKey, afterID) cursor — or past
// order.StartFrom on page 1 — then walks forward RE-READING live nc.meta/nc.keyTTL
// (the snapshot is filter-independent and TTL-lazy, so the TTL + per-key-TTL gate +
// the per-query FILTER run HERE over the live payload), materializing up to limit
// Documents. The emitted Metadata is the live per-key-TTL-gated payload. Must hold
// nc.mu (R or W). rows is immutable (never mutated in place).
func (nc *NamedCollection) collectOrderedLocked(rows []OrderedID, pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int, now int64) (docs []Document, nextAfter uint64, hasMore bool) {
	start := orderSeekStart(rows, order, afterID, afterKey, hasAfter)
	str := order.Kind == OrderString
	multi := isMultiKey(order)
	var keys []OrderBy
	if multi {
		keys = orderKeyList(order)
	}
	for i := start; i < len(rows); i++ {
		id := rows[i].ID
		if _, ok := nc.ids[id]; !ok {
			continue // deleted since the snapshot was built (benign; bump rebuilds next)
		}
		if dl := nc.ttl[id]; dl != 0 && dl <= now {
			continue // ttl-expired (lazy: still in the snapshot, dropped here)
		}
		// Re-read the live per-key-TTL-gated payload for BOTH the predicate eval and
		// the emitted payload (a filter on an expired key never matches; the scrolled
		// payload omits it).
		payload := liveMetaMap(nc.meta[id], nc.keyTTL[id], now)
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

// NumPoints returns the number of live points (shared id set).
func (nc *NamedCollection) NumPoints() int {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	return len(nc.ids)
}

// Named-collection snapshot format (mirrors MultiVectorIndex.snapshot/restore —
// a self-contained image for inclusion in a cluster FSM snapshot). v1 named
// collections rebuild from this blob / the Raft log; a WAL + instant-restart
// persist sidecar is a documented follow-up (see the named-vectors plan
// non-goals), so there is no on-disk sidecar here.
//
//	[magic "RNV1"][version:u8=1]
//	[nSpaces:u32]
//	  per space: [nameLen:u16][name][innerLen:u32][hnsw.Snapshot blob]
//	[shared meta count:u32]
//	  per point: [id:u64][entries:u32] then each [keyString][writeValue]
//	[ttl count:u32]   per point: [id:u64][deadline:i64]
//	[ids count:u32]   per point: [id:u64]
//	[keyTTL count:u32] (v2+) per point: [id:u64][entries:u32] then each [keyString][deadline:i64]
//	[version count:u32] (v3+) per point: [id:u64][version:u64]
//	[sparse spaces:u32] (v4 ONLY) per space: [nameLen:u16][name][count:u32] then
//	   per point [id:u64][nIdx:u32]{idx:u32...}[nVal:u32]{val:f32...}
//
// The version byte is 3 (namedSnapshotVersionDenseOnly) for a dense-only
// collection — NO trailing sparse block — so a dense-only snapshot is
// byte-identical to the pre-sparse format. A collection with at least one sparse
// space writes version 4 and the per-space sparse block; restore rebuilds each
// space's id-keyed inverted index from the stored vecs.
//
// The sub-index blobs carry NO arena metadata (the named sub-arenas are
// metadata-nil); the SHARED per-point payload is the outer meta block, so
// restore reconstructs the filtered-search/scroll source exactly. v2 added the
// per-key payload TTL block (ABSOLUTE unix-millis deadlines) at the tail; a v1
// snapshot has no block and restores keyTTL empty (no per-key TTL —
// backward-compatible). v3 appends the per-point CAS version block; a v1/v2
// snapshot has no version block and restores every live point's version to a sane
// default of 1 (so a restored old point has a non-zero CAS version).
var namedSnapshotMagic = []byte{'R', 'N', 'V', '1'}

// namedSnapshotVersion is the named-collection snapshot codec version. v1 = the
// original (sub-indexes + shared meta/ttl/ids). v2 appends the per-key payload TTL
// block (absolute unix-ms deadlines). v3 appends the per-point CAS version block.
// v1 snapshots load with keyTTL empty; v1/v2 snapshots default versions to 1.
const namedSnapshotVersion = 4

// namedSnapshotVersionDenseOnly is the version written for a collection with NO
// sparse spaces: the pre-sparse v3 codec, byte-identical to before this change. A
// collection WITH at least one sparse space writes namedSnapshotVersion (v4) and
// appends the per-space sparse block. This keeps a dense-only named collection's
// snapshot bytes unchanged (the version byte stays 3, no trailing sparse block).
const namedSnapshotVersionDenseOnly = 3

// namedSnapshotMinReadVersion is the oldest named snapshot Restore accepts. v1
// (no per-key TTL block) loads with keyTTL empty — backward compatible.
const namedSnapshotMinReadVersion = 1

// Snapshot writes a self-contained image (every named sub-index framed by its
// space name + the shared per-point meta/ttl/ids state) to w. Mirrors
// MultiVectorIndex.snapshot. Safe to call concurrently with reads.
func (nc *NamedCollection) Snapshot(w io.Writer) error {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	bw := bufio.NewWriter(w)
	if _, err := bw.Write(namedSnapshotMagic); err != nil {
		return err
	}
	// A dense-only collection writes v3 with NO trailing sparse block (byte-identical
	// to the pre-sparse format). A collection with at least one sparse space writes v4
	// and appends the per-space sparse block.
	hasSparse := len(nc.sparseSpaces) > 0
	snapVer := byte(namedSnapshotVersionDenseOnly)
	if hasSparse {
		snapVer = byte(namedSnapshotVersion)
	}
	if err := bw.WriteByte(snapVer); err != nil {
		return err
	}

	// Sub-indexes, framed by space name; each inner blob is length-prefixed so
	// restore can split it.
	if err := writeU32(bw, uint32(len(nc.indexes))); err != nil { //nolint:gosec
		return err
	}
	for sname, idx := range nc.indexes {
		var inner bytes.Buffer
		if err := idx.Snapshot(&inner); err != nil {
			return fmt.Errorf("vector: named snapshot space %q: %w", sname, err)
		}
		var hdr [2]byte
		binary.BigEndian.PutUint16(hdr[:], uint16(len(sname))) //nolint:gosec
		if _, err := bw.Write(hdr[:]); err != nil {
			return err
		}
		if _, err := bw.WriteString(sname); err != nil {
			return err
		}
		if err := writeBytes32(bw, inner.Bytes()); err != nil {
			return err
		}
	}

	// Shared per-point payload (the authoritative meta map). Uses the same
	// writeValue codec the hnsw arena metadata uses, so every Value kind round-
	// trips identically.
	if err := writeU32(bw, uint32(len(nc.meta))); err != nil { //nolint:gosec
		return err
	}
	for id, m := range nc.meta {
		if err := writeU64(bw, id); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(m))); err != nil { //nolint:gosec
			return err
		}
		for key, val := range m {
			if err := writeString(bw, key); err != nil {
				return err
			}
			if err := writeValue(bw, val); err != nil {
				return err
			}
		}
	}

	// Shared per-point ttl deadlines (absolute unix-millis; 0 = no expiry).
	if err := writeU32(bw, uint32(len(nc.ttl))); err != nil { //nolint:gosec
		return err
	}
	for id, dl := range nc.ttl {
		if err := writeU64(bw, id); err != nil {
			return err
		}
		if err := writeI64(bw, dl); err != nil {
			return err
		}
	}

	// Live id set (authoritative; a point counts even if it omits some spaces).
	if err := writeU32(bw, uint32(len(nc.ids))); err != nil { //nolint:gosec
		return err
	}
	for id := range nc.ids {
		if err := writeU64(bw, id); err != nil {
			return err
		}
	}

	// Per-key payload TTL (v2): id -> key -> ABSOLUTE unix-ms deadline. Only ids
	// WITH a non-empty deadline map are written (prefixed by a count); deadlines
	// are preserved verbatim so a pending key TTL is time-stable across restore. A
	// no-per-key-TTL collection pays just a zero count. v1 readers never reach this
	// block (the version gate stops them); v1 snapshots omit it entirely.
	withKeyTTL := 0
	for _, ke := range nc.keyTTL {
		if len(ke) > 0 {
			withKeyTTL++
		}
	}
	if err := writeU32(bw, uint32(withKeyTTL)); err != nil { //nolint:gosec
		return err
	}
	for id, ke := range nc.keyTTL {
		if len(ke) == 0 {
			continue
		}
		if err := writeU64(bw, id); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(ke))); err != nil { //nolint:gosec
			return err
		}
		for key, dl := range ke {
			if err := writeString(bw, key); err != nil {
				return err
			}
			if err := writeI64(bw, dl); err != nil {
				return err
			}
		}
	}

	// Per-point CAS version (v3): id -> monotonic version. Only ids WITH a non-zero
	// version are written (prefixed by a count); versions are preserved verbatim so
	// a restored point keeps its exact CAS version (NOT re-bumped). v1/v2 readers
	// never reach this block (the version gate stops them); v1/v2 snapshots omit it.
	withVersion := 0
	for _, v := range nc.version {
		if v != 0 {
			withVersion++
		}
	}
	if err := writeU32(bw, uint32(withVersion)); err != nil { //nolint:gosec
		return err
	}
	for id, v := range nc.version {
		if v == 0 {
			continue
		}
		if err := writeU64(bw, id); err != nil {
			return err
		}
		if err := writeU64(bw, v); err != nil {
			return err
		}
	}

	// Per-space SPARSE block (v4+). ONLY written when the collection has sparse
	// spaces (hasSparse → v4); a dense-only collection wrote v3 above and never
	// reaches a v4-only reader, so its bytes are unchanged. Each sparse space frames
	// its full vecs map: [nameLen:u16][name][count:u32] then per point [id:u64]
	// [sparsevec]. The id-keyed inverted index is NOT serialized — Restore rebuilds
	// it from vecs (mirrors the dense sparse index rebuilt from the arena).
	if hasSparse {
		var buf bytes.Buffer
		if err := writeU32(&buf, uint32(len(nc.sparseSpaces))); err != nil { //nolint:gosec
			return err
		}
		for sname, sp := range nc.sparseSpaces {
			var hdr [2]byte
			binary.BigEndian.PutUint16(hdr[:], uint16(len(sname))) //nolint:gosec
			buf.Write(hdr[:])
			buf.WriteString(sname)
			if err := writeU32(&buf, uint32(len(sp.vecs))); err != nil { //nolint:gosec
				return err
			}
			for id, sv := range sp.vecs {
				if err := writeU64(&buf, id); err != nil {
					return err
				}
				writeSparseVecFrame(&buf, sv)
			}
		}
		if _, err := bw.Write(buf.Bytes()); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// Restore rebuilds the collection from a blob written by Snapshot, replacing all
// current state. The collection MUST already be constructed with the matching
// NamedVectors config (NewNamedCollection) — the sub-index set is keyed by the
// snapshot's space names and each is restored into the pre-built sub-index.
// Mirrors MultiVectorIndex.restore. Round-trip identity: every sub-index vector
// + the shared payload/ttl/ids come back exactly.
func (nc *NamedCollection) Restore(r io.Reader) error {
	br := bufio.NewReader(r)
	magic := make([]byte, len(namedSnapshotMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return err
	}
	if string(magic) != string(namedSnapshotMagic) {
		return fmt.Errorf("vector: bad named-collection snapshot magic")
	}
	ver, err := br.ReadByte()
	if err != nil {
		return err
	}
	if ver < namedSnapshotMinReadVersion || ver > namedSnapshotVersion {
		return fmt.Errorf("vector: named-collection snapshot version %d unsupported", ver)
	}

	nc.mu.Lock()
	defer nc.mu.Unlock()

	nSpaces, err := readU32(br)
	if err != nil {
		return err
	}
	for i := uint32(0); i < nSpaces; i++ {
		var hdr [2]byte
		if _, err := io.ReadFull(br, hdr[:]); err != nil {
			return err
		}
		nameBuf := make([]byte, binary.BigEndian.Uint16(hdr[:]))
		if _, err := io.ReadFull(br, nameBuf); err != nil {
			return err
		}
		sname := string(nameBuf)
		inner, err := readBytes32(br)
		if err != nil {
			return err
		}
		idx, ok := nc.indexes[sname]
		if !ok {
			return fmt.Errorf("vector: named restore: unknown space %q (config mismatch)", sname)
		}
		if rerr := idx.Restore(bytes.NewReader(inner)); rerr != nil {
			return fmt.Errorf("vector: named restore space %q: %w", sname, rerr)
		}
	}

	// Shared per-point payload.
	metaCount, err := readU32(br)
	if err != nil {
		return err
	}
	meta := make(map[uint64]Metadata, metaCount)
	for i := uint32(0); i < metaCount; i++ {
		id, err := readU64(br)
		if err != nil {
			return err
		}
		entries, err := readU32(br)
		if err != nil {
			return err
		}
		m := make(Metadata, entries)
		for j := uint32(0); j < entries; j++ {
			key, err := readString(br)
			if err != nil {
				return err
			}
			val, err := readValue(br)
			if err != nil {
				return err
			}
			m[key] = val
		}
		meta[id] = m
	}

	ttlCount, err := readU32(br)
	if err != nil {
		return err
	}
	ttl := make(map[uint64]int64, ttlCount)
	for i := uint32(0); i < ttlCount; i++ {
		id, err := readU64(br)
		if err != nil {
			return err
		}
		dl, err := readI64(br)
		if err != nil {
			return err
		}
		ttl[id] = dl
	}

	idCount, err := readU32(br)
	if err != nil {
		return err
	}
	ids := make(map[uint64]struct{}, idCount)
	for i := uint32(0); i < idCount; i++ {
		id, err := readU64(br)
		if err != nil {
			return err
		}
		ids[id] = struct{}{}
	}

	// Per-key payload TTL (v2+). v1 snapshots have no block; keyTTL stays empty (no
	// per-key TTL — backward compatible). Deadlines are restored verbatim (absolute
	// unix-ms) so a pending key TTL survives restore time-stable.
	keyTTL := make(map[uint64]map[string]int64)
	if ver >= 2 {
		ktCount, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < ktCount; i++ {
			id, err := readU64(br)
			if err != nil {
				return err
			}
			entries, err := readU32(br)
			if err != nil {
				return err
			}
			ke := make(map[string]int64, entries)
			for j := uint32(0); j < entries; j++ {
				key, err := readString(br)
				if err != nil {
					return err
				}
				dl, err := readI64(br)
				if err != nil {
					return err
				}
				ke[key] = dl
			}
			if len(ke) > 0 {
				keyTTL[id] = ke
			}
		}
	}

	// Per-point CAS version (v3+). v1/v2 snapshots have no version block; default
	// every live point's version to 1 (a sane non-zero CAS version for an old
	// point). v3+ restores versions verbatim (the exact value the op produced).
	version := make(map[uint64]uint64)
	if ver >= 3 {
		vCount, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < vCount; i++ {
			id, err := readU64(br)
			if err != nil {
				return err
			}
			v, err := readU64(br)
			if err != nil {
				return err
			}
			if v != 0 {
				version[id] = v
			}
		}
	} else {
		for id := range ids {
			version[id] = 1
		}
	}

	// Per-space SPARSE block (v4+). v1/v2/v3 snapshots have no block (dense-only) —
	// the pre-built sparse spaces (matching the config) stay empty. v4 restores each
	// space's vecs map and rebuilds its id-keyed inverted index from those vecs (the
	// index is never serialized — mirrors the dense sparse index rebuilt on load).
	if ver >= 4 {
		nSparse, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < nSparse; i++ {
			var hdr [2]byte
			if _, err := io.ReadFull(br, hdr[:]); err != nil {
				return err
			}
			nameBuf := make([]byte, binary.BigEndian.Uint16(hdr[:]))
			if _, err := io.ReadFull(br, nameBuf); err != nil {
				return err
			}
			sname := string(nameBuf)
			sp, ok := nc.sparseSpaces[sname]
			if !ok {
				return fmt.Errorf("vector: named restore: unknown sparse space %q (config mismatch)", sname)
			}
			count, err := readU32(br)
			if err != nil {
				return err
			}
			vecs := make(map[uint64]*SparseVector, count)
			for j := uint32(0); j < count; j++ {
				pid, err := readU64(br)
				if err != nil {
					return err
				}
				sv, ok := readSparseVecFrame(br)
				if !ok {
					return fmt.Errorf("vector: named restore: torn sparse vector in space %q", sname)
				}
				vecs[pid] = sv
			}
			sp.vecs = vecs
			sp.idx.rebuild(vecs)
		}
	}

	nc.meta = meta
	nc.ttl = ttl
	nc.ids = ids
	nc.keyTTL = keyTTL
	nc.version = version
	// Rebuild the filter-first index from the restored payload (never serialized —
	// the rebuild-on-load path, mirroring dense snapshot restore). WAL replay on
	// top mutates incrementally via the reindex hooks; loadNamed also rebuilds once
	// more after replay completes as a belt-and-suspenders guarantee.
	if nc.payloadIdx == nil {
		nc.payloadIdx = newPayloadIndexID()
	}
	nc.payloadIdx.rebuild(nc.meta)
	// Wholesale state replacement: invalidate any cached order snapshots from a prior
	// life of this collection (orderSnaps may have been built before this Restore).
	nc.bumpData()
	return nil
}

// Close releases every sub-index and, on a WAL-mode collection, closes the WAL
// file (the durable records remain on disk for replay on the next open).
func (nc *NamedCollection) Close() error {
	// Stop the sweeper BEFORE taking nc.mu — the sweeper goroutine acquires nc.mu, so
	// stopping under the lock would deadlock. Idempotent + safe before startSweeper ran,
	// so Close on any teardown/error path joins the goroutine (no leak).
	nc.Stop()
	nc.mu.Lock()
	defer nc.mu.Unlock()
	var firstErr error
	for _, idx := range nc.indexes {
		if err := idx.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if nc.wal != nil {
		if err := nc.wal.close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
