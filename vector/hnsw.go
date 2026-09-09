// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/rostamlabs/rostam/vector/analysis"
)

// defaultMaxEfSearch caps dynamic ef widening for filtered search when
// Config.MaxEfSearch is 0.
const defaultMaxEfSearch = 1024

// defaultFilterFirstThreshold is the candidate-set ceiling under which the
// query planner brute-forces an equality-filtered search instead of traversing
// the graph (used when Config.FilterFirstThreshold is 0).
const defaultFilterFirstThreshold = 10_000

// filterFirstRelativeHardCap bounds the relative selectivity gate: even at
// FilterFirstRelativeBP=10000 (100% of liveCount) the effective materialization
// limit never exceeds this ceiling, so a non-selective filter on a billion-doc
// collection cannot brute-force unbounded memory.
const filterFirstRelativeHardCap = 1_000_000

// effectiveFilterFirstLimit is the SINGLE source of truth for the filter-first
// materialization limit across all four families (dense HNSW/IVF, named, MV). It
// folds the OPT-IN relative selectivity gate into the existing absolute threshold:
//
//	max(absThreshold, min(relativeBP*liveCount/10000, hardCap))
//
// When relativeBP == 0 (the DEFAULT) it returns absThreshold EXACTLY, so every
// gate site is byte/behaviour-identical to the pre-feature absolute behavior. When
// relativeBP > 0 the relative budget can raise the ceiling above absThreshold (a
// relatively-selective filter on a huge collection uses exact filter-first beyond
// the absolute cap), bounded by hardCap. The relativeBP*liveCount product is
// computed in int64 to avoid overflow: relativeBP up to 10000 times a multi-billion
// liveCount overflows a 32-bit-ish int; the int64 product is clamped to hardCap
// (<= MaxInt) BEFORE the narrowing conversion to int, so the result is always a
// valid non-negative int.
func effectiveFilterFirstLimit(absThreshold, relativeBP, liveCount int) int {
	if relativeBP <= 0 {
		return absThreshold // OFF (default) -> byte-identical to the absolute threshold
	}
	rel := int64(relativeBP) * int64(liveCount) / 10000
	if rel > filterFirstRelativeHardCap {
		rel = filterFirstRelativeHardCap
	}
	if int(rel) > absThreshold {
		return int(rel)
	}
	return absThreshold
}

// node is one entry in the HNSW graph. neighbors is one slice per level,
// up to and including the node's max level. Slot is redundant with the node's
// index in the parent nodes slice but kept here so a single *node carries
// enough info for stats / iteration paths.
type node struct {
	slot  uint32
	level int
	// upper holds neighbor lists for levels 1..level (upper[k] = level k+1);
	// nil for the common level-0-only node. Level-0 edges live in the index's
	// flat slab (hnsw.level0), not here. See graph.go.
	upper [][]uint32

	// unlinked marks a node that is PLACED but whose link phase has not run yet
	// — the Option B window in which it sits in h.nodes carrying its final level
	// and holding no edges at any level (see link_stripes.go).
	//
	// SUCH A NODE MUST BE INVISIBLE TO GRAPH TRAVERSAL. Before Option B, being in
	// h.nodes implied being linked, and every traversal — a query's, and every
	// linker's own descent — relied on it without ever saying so. A node with no
	// edges is a DEAD END: a descent that steps onto it finds nothing at the next
	// level down, so its frontier collapses to that one node and the level-0
	// search that follows returns a handful of candidates instead of ef. The new
	// node then links with an out-degree of one to three instead of M, to
	// whatever that dead end happened to touch. Queries recover on their next
	// attempt; a LINK does not — the starved edge set is written into the graph,
	// and those nodes go on to seed further starved neighbourhoods until a
	// pocket of them is reachable only from each other. That is the cluster
	// reshard failure: points present in the arena, byte-correct, returned by
	// scans, and permanently absent from search.
	//
	// It also gates the entry-point elections, which pick the highest-level
	// survivor by scanning h.nodes. Electing from inside the window is the same
	// bug at maximum blast radius: the entry point has no out-edges, so the
	// linker about to run for that very node reads itself back from entryState,
	// descends from itself, finds only itself, and becomes its own neighbour at
	// every level — orphaning the entire index at a stroke.
	//
	// The polarity is deliberate. A `linked` flag would have to be SET by every
	// construction site — restore, snapshot replay, both bulk builders, the
	// first-node path — and forgetting one silently erases nodes from the graph.
	// `unlinked` defaults to the safe value, so only the paths that actually
	// DEFER a link opt in, and each clears the flag at the end of linkNode.
	//
	// There are two such paths, and the second was found the hard way. The
	// incremental one is placeLockedAt handing off a linkTask. The other is
	// BuildConcurrent, which places EVERY node up front and links them later in
	// parallel — the same window, n nodes wide, and it originally left the flag
	// false throughout. The failure there is subtler than the edgeless-node case
	// above, because linkNode publishes a node's levels TOP-DOWN: after it
	// writes level lc's forward list and back-edges, the node is reachable at
	// level lc while its level lc-1..0 lists are still empty. A concurrent
	// worker's width-1 descent lands on it at an upper level, carries it down as
	// the sole level-0 frontier, and expands a node with no level-0 edges — so
	// the level-0 search returns that ONE node, the new node links to it alone,
	// and that single back-edge is later pruned away when the frontier node
	// writes its own 2M level-0 edges. In-degree zero, permanently. Setting the
	// flag in BuildConcurrent's placement loop is what closes it; see there for
	// the measurements and for why the seed slot must stay unflagged.
	//
	// Atomic because the traversal reads it under the READ lock while linkRead
	// clears it under the READ lock too — the elections' write-lock exclusion is
	// not enough once queries consult it. Only ever true→false, so a stale read
	// costs at most one skipped node on one traversal.
	unlinked atomic.Bool

	// staleLevel0 marks a node placed on a REUSED slot whose level-0 slab region
	// still holds the PREDECESSOR's edges, until its own forward write replaces
	// them. Those edges are deliberately left in place (see placeLockedAt): they
	// keep the slot traversable through the whole placement/link window, which is
	// what an upsert-heavy workload depends on.
	//
	// It is what placeLockedAt consults to decide whether the node has to be
	// HIDDEN during that window (node.unlinked): a slot carrying inherited edges
	// is a usable waypoint and must stay visible, an empty one is a dead end and
	// must not. Set under the write lock during placement, cleared by the level-0
	// forward write under that slot's link stripe, at the moment the inherited
	// edges stop being inherited.
	//
	// It no longer decides whether those edges are PRESERVED. linkNode now merges
	// them through the same append-and-prune path as a raced-in back-edge, so the
	// heuristic keeps the ones that are still good against the new vector and
	// drops the rest — see the merge in linkNode for the measurement that changed
	// this, and why discarding them wholesale orphaned points at low degree.
	staleLevel0 bool
}

// hnsw is an HNSW index implementing VectorIndex. One hnsw owns one arena
// (vector storage) and one graph (neighbor lists per level). Mutex semantics
// match the spec: RLock for reads, Lock for inserts/deletes/tombstone reclaim.
type hnsw struct {
	cfg   Config
	arena *arena
	quant quantizer // nil = full-precision float32 (no quantization)

	// pairExact/pairCode are the two build-time slot-to-slot distance closures,
	// allocated ONCE by initPairFns. buildPairFn used to build a fresh escaping
	// closure on every neighbor selection (36.8% of ingest allocations: every
	// back-edge prune and every diversity pass paid one), even though the pair
	// function depends only on h.quant/h.cfg.
	//
	// Two things are handled WITHOUT any invalidation step, because both
	// closures dereference h rather than capturing h.arena/h.quant: an arena
	// swap (snapshot restore installs a new *arena) and a train that flips
	// quantizedBuild() (buildPairFn re-evaluates that predicate on every call).
	//
	// ONE thing is NOT: pairExact captures the resolved distance kernel for
	// cfg.Metric/cfg.Dim by value, so ANY mutation of h.cfg MUST re-run
	// initPairFns. There is exactly one such site — readSnapshot's `h.cfg = cfg`
	// — and it does, right next to its mL recompute. Getting this wrong is
	// silent: the diversity heuristic would compare one metric's pf() against
	// another metric's candidate distances and quietly wreck recall on every
	// post-restore insert, with no error anywhere. See
	// TestRestoredIndexRebuildsPairFnForSnapshotMetric.
	//
	// Otherwise immutable ⇒ safe to read from the concurrent build's workers
	// (readSnapshot holds the write lock and no build is in flight).
	pairExact func(a, b uint32) float32
	pairCode  func(a, b uint32) float32

	rng *rand.Rand
	mL  float64 // 1/ln(M); precomputed for assignLevel (0 ⇒ every node level 0)

	// pruneAlpha is the RobustPrune α used by selectNeighbors: a candidate v is
	// dropped when α·dist(kept*, v) < dist(query, v). 1.0 is the original
	// Malkov-Yashunin heuristic (HNSW), so HNSW leaves it 0/1 and is
	// behavior-identical. Vamana's newVamana sets it to cfg.VamanaAlpha; its
	// two-pass build flips it between 1 (pass 1) and VamanaAlpha (pass 2). Read on
	// every selectNeighbors call; the passes are sequential (never overlapping) so
	// the plain field is race-free. Zero is treated as 1.0 (effFactorAlpha).
	pruneAlpha float32

	// vamana marks an IndexVamana index: single-layer (mL forced to 0 so every
	// node is level 0), R-bounded out-degree (m0 = VamanaR). Set by newVamana;
	// false for HNSW/IVF (byte/behavior-identical).
	vamana bool

	mu    sync.RWMutex
	nodes []*node // slot -> node (graph adjacency); nil entry = no/reclaimed node

	// Level-0 adjacency, stored flat instead of per-node to avoid millions of
	// tiny slice allocations (see graph.go). m0 = 2*M is the level-0 cap and the
	// per-slot stride; node `slot`'s edges are level0[slot*m0 : slot*m0+
	// level0Len[slot]]. Levels >= 1 live in node.upper.
	m0        int
	level0    []uint32
	level0Len []uint16

	// Optional mmap backing for the level-0 slab (Config.GraphMmapPath). When
	// set, level0 aliases graphRegion instead of the Go heap, so the graph's
	// largest array is off-heap and OS-reclaimable. level0Len and node.upper
	// stay on the heap (small). nil graphMmapF = heap-backed level0.
	graphMmapF  *os.File
	graphRegion []byte

	// graphRes, when non-nil, is the address-space reservation level0 is carved
	// out of — what makes slab growth O(1) with a STABLE base instead of an O(n)
	// copy/remap under the write lock (see reserve.go). File-backed when
	// graphMmapF is set, anonymous otherwise, so a heap-backed graph gets the
	// same treatment. nil = still on the legacy growth path.
	graphRes *slabReservation

	entryPoint uint32          // slot of the top-level entry node
	maxLevel   int             // top level in use; -1 when empty
	tombstoned map[uint32]bool // slots with logical deletion (Week 2: simple map; Week 5: bitset)

	// idSetVersion is bumped on every change to the LIVE id set — a new id added
	// (Insert / InsertIfAbsent / upsert-resurrection of a dead slot), an id
	// tombstoned (Delete), a TTL sweep that tombstones (sweepOnce), and a Reclaim
	// that frees tombstoned slots. It is NOT bumped on payload-only updates
	// (SetPayload/OverwritePayload/DeletePayloadKeys/ClearPayload), which leave the
	// id set unchanged, so a payload write never invalidates the scroll snapshot.
	// It is a plain uint64 guarded by h.mu: every id-set mutation already holds the
	// write lock, so the bump is a single increment with no extra synchronization —
	// the ONLY cost the scroll snapshot adds to the write path (no sort/index work).
	// scrollPage reads it under h.mu (R or W) to detect a stale cached snapshot.
	idSetVersion uint64

	// scrollSnap is a lazily-built, cached sorted-id snapshot for deterministic
	// id-ascending scroll (resume-after-id cursor seeks). ver records the
	// idSetVersion the ids slice was built at; when it differs from idSetVersion
	// the snapshot is stale and rebuilt on the next scroll. ids holds the LIVE
	// ids (idMap keys minus tombstoned) sorted ascending; TTL-expired ids are NOT
	// excluded here (they age without a mutation) — the forward walk's admits gate
	// filters them. Guarded by h.mu; rebuilt under the write lock.
	scrollSnap struct {
		ver uint64
		ids []uint64
	}

	// scrollRebuilds counts how many times scrollSnap was rebuilt — a test hook to
	// assert snapshot reuse (no rebuild without an id-set change). Guarded by h.mu.
	scrollRebuilds uint64

	// dataVersion is the order_by snapshot's invalidation counter. Unlike
	// idSetVersion (id-set changes only), it bumps on the UNION of id-set mutations
	// {Insert/Delete/reclaim/sweep/build/restore} AND payload mutations
	// {SetPayload/OverwritePayload/DeletePayloadKeys/ClearPayload/RestorePayload}: a
	// cached (field, direction) sorted snapshot is payload-VALUE-keyed, so a payload
	// write to the order field MUST invalidate it. It is deliberately SEPARATE from
	// idSetVersion so a payload write does NOT invalidate the id-scroll scrollSnap (no
	// regression). Guarded by h.mu; bumped via bumpData() under the write lock. Starts
	// at 1 (the zero-valued orderSnap.ver is reserved for "never built").
	dataVersion uint64

	// orderSnaps caches per-(field, direction) sorted snapshots for the order_by
	// scroll, version-stamped with dataVersion. Bounded at orderCacheCap with
	// oldest-built eviction. Guarded by h.mu (warm read under RLock of an immutable
	// rows slice; cold rebuild under Lock, double-checked). See order.go orderSnap.
	orderSnaps map[orderCacheKey]*orderSnap
	// orderSeq is a monotonic stamp assigned to each freshly built orderSnap so the
	// cap eviction can pick the oldest-built entry. Guarded by h.mu.
	orderSeq uint64
	// orderRebuilds counts order-snapshot rebuilds — a test hook mirroring
	// scrollRebuilds (assert warm reuse / correct invalidation). Guarded by h.mu.
	orderRebuilds uint64

	searchOps     atomic.Uint64
	insertOps     atomic.Uint64
	expiredCount  atomic.Uint64 // vectors filtered/swept due to TTL
	keysSwept     atomic.Uint64 // per-key payload TTL entries physically swept
	quotaRejects  atomic.Uint64 // inserts rejected by quota or rate limit
	filterRejects atomic.Uint64 // candidates rejected by a search predicate

	// filterGates counts filtered searches that ARMED the bitset admission gate
	// (filter_bitset.go) — one increment per query, not per candidate, so it is
	// free next to the traversal it reports on. It answers the operational
	// question the gate raises ("is the payload index actually accelerating my
	// filtered traversals, or silently declining every time?"), which is
	// otherwise invisible: a gate that never arms and a gate that always arms
	// produce byte-identical results.
	filterGates atomic.Uint64

	// complementGates is the SUBSET of filterGates armed from the rejection side
	// (buildComplementGate) rather than the accepted side. Kept apart because the
	// two answer different operational questions: filterGates says the index is
	// helping, complementGates says WHICH of the two cost regimes the workload is
	// in — and a high-pass-rate filter that stops arming after a schema change
	// (an optional field breaks the totality precondition) is otherwise a silent
	// throughput cliff with no counter that moved.
	complementGates atomic.Uint64

	// columnGates is the SUBSET of filterGates answered by the numeric column
	// sidecar (payload_column.go) rather than by a bitset. Worth its own counter
	// for the same reason complementGates is: the column path is the one with no
	// cost model, so the only way to see that a filter stopped taking it — a
	// changed op, a per-key TTL appearing on the collection, the column cap
	// reached — is a counter that stops moving.
	columnGates atomic.Uint64

	// columnDrops counts the times an insert reclaimed the column sidecar to stay
	// inside Config.MaxBytes. It is the signal that a collection is sized so
	// tightly that reads and writes are fighting over the same bytes: the writes
	// win (see insertLocked), so nothing breaks, but every drop is an
	// acceleration thrown away and a rebuild the next filtered query has to pay
	// for. A counter that climbs steadily means MaxBytes wants raising.
	columnDrops atomic.Uint64

	// now is the clock used for TTL deadlines. Defaults to time.Now().UnixMilli;
	// tests can override to control aging deterministically.
	now func() int64

	bucket    *tokenBucket // nil = no rate limit
	bytesUsed int64        // running byte estimate; guarded by mu

	searchLat latencyHistogram // per-Search wall time
	insertLat latencyHistogram // per-Insert wall time

	sparseIdx *sparseIndex // inverted index over sparse-vector dims; guarded by mu

	// bm25 is the full-text BM25 inverted index, allocated ONLY when full text is
	// enabled (Config.FullText != nil); nil otherwise so a non-full-text collection
	// is byte/behavior-identical. Guarded by the SAME mu as sparseIdx (write lock
	// for add/remove/rebuild, read lock for the scan). az is the resolved analyzer
	// the write/query/rebuild paths use; non-nil iff bm25 != nil.
	bm25 *bm25Index
	az   analysis.Analyzer

	payloadIdx *payloadIndex // equality index over metadata fields; guarded by mu

	// Link-phase concurrency. See link_stripes.go for the full contract; in
	// short, linkStripes is a permanent, lazily allocated array of per-slot
	// neighbor-list locks, and linkers counts the link phases currently in
	// flight so that a reader with none pays nothing.
	//
	// linkers sits alone on its cache line. It is read once per graph hop by
	// every concurrent query, so sharing a line with globalMu — which a linker
	// WRITES on every entry-point read and promotion — would invalidate that
	// line in every searching core for a value that changes twice per insert.
	_           [cacheLineSize]byte
	linkers     atomic.Int32
	_           [cacheLineSize]byte
	linkStripes []paddedMutex

	// globalMu serializes the rare entryPoint/maxLevel promotion performed by a
	// linker (which holds only the read lock) against the traversals that read
	// the pair. Write-lock holders own the pair outright and skip it.
	globalMu sync.Mutex

	// linkMu is the barrier between link phases and the two paths that
	// serialize the WHOLE graph — Snapshot and SavePersist. Both walk every
	// node's neighbor lists under the read lock, where they would otherwise be
	// concurrent with a linker and could capture a half-linked node (a point
	// present in the arena with an incomplete, possibly empty, edge set — an
	// orphan after restore). Linkers hold it shared, the serializers
	// exclusively, so a serialized graph is always a fully-linked one.
	linkMu sync.RWMutex
}

// SetNowFunc overrides the wall-clock source (unix millis) the index's non-apply
// expiry sites consult — the background sweeper, the client read/query filter, and
// the wall-clock branch of the write paths. nil restores the real clock. It is a
// TEST/advanced seam mirroring cache.Cache.SetNowFunc: production never calls it,
// so the default (time.Now().UnixMilli) is byte-identical. It does NOT affect the
// stamped apply path (InsertAt et al. take an explicit stamp). Takes the write lock.
func (h *hnsw) SetNowFunc(fn func() int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if fn == nil {
		h.now = func() int64 { return time.Now().UnixMilli() }
		return
	}
	h.now = fn
}

// newHNSW constructs an HNSW index. Returns ErrInvalid* if cfg is malformed.
func newHNSW(cfg Config) (*hnsw, error) {
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}
	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	a := newArena(cfg.Dim, 0)
	// A declared vector cap sizes the slab's address-space reservation (it is a
	// hint, never a limit — see slabReserveSize).
	a.maxVectorsHint = cfg.MaxVectors
	quant := newQuantizer(cfg.Quant, cfg.Dim, cfg.QuantPQM, cfg.SQBits, cfg.PRQLayers, cfg.PQNBits, cfg.Metric)
	if quant != nil {
		a.setQuant(quant)
		if cfg.QuantStorage == QuantMmap {
			if err := a.useMmap(cfg.MmapPath); err != nil {
				return nil, err
			}
		}
	}
	h := &hnsw{
		cfg:        cfg,
		arena:      a,
		quant:      quant,
		rng:        rand.New(rand.NewSource(seed)),
		mL:         1.0 / math.Log(float64(cfg.M)),
		nodes:      nil,
		m0:         2 * cfg.M,
		maxLevel:   -1,
		tombstoned: make(map[uint32]bool),
		now:        func() int64 { return time.Now().UnixMilli() },
		bucket:     newTokenBucket(cfg.MaxInsertsPerSecond),
		sparseIdx:  newSparseIndex(),
		payloadIdx: newPayloadIndex(),
		// Start at 1 so the zero-valued scrollSnap.ver (0) is reserved for
		// "no snapshot built yet": a bulk-built or restored index that never bumps
		// from 1 still forces the first scroll to rebuild (0 != 1). See scrollPage.
		idSetVersion: 1,
		// dataVersion starts at 1 for the same reason: the zero-valued orderSnap.ver
		// (0) means "never built", so a restored/bulk index forces the first rebuild.
		dataVersion: 1,
		orderSnaps:  make(map[orderCacheKey]*orderSnap),
	}
	h.initPairFns()
	if cfg.GraphMmapPath != "" {
		if err := h.useGraphMmap(cfg.GraphMmapPath); err != nil {
			_ = a.Close() // release the vector mmap opened above, if any
			return nil, err
		}
	}
	// Full-text BM25 lane: allocate the bm25Index + resolve the analyzer ONLY when
	// configured. Validate already accepted the analyzer name, so ByName succeeds.
	// nil FullText leaves h.bm25/h.az nil → byte/behavior-identical to before.
	if cfg.FullText != nil {
		az, err := analysis.ByName(cfg.FullText.Analyzer)
		if err != nil {
			_ = a.Close()
			return nil, err
		}
		// A resolved analyzer MUST implement CountingAnalyzer or the BM25 maintenance
		// path (analyzeCounts) silently produces an empty index — fail loud instead.
		if _, ok := az.(analysis.CountingAnalyzer); !ok {
			_ = a.Close()
			return nil, fmt.Errorf("full-text analyzer %q does not implement CountingAnalyzer", cfg.FullText.Analyzer)
		}
		h.az = az
		h.bm25 = newBM25Index(cfg.FullText.K1, cfg.FullText.B)
	}
	return h, nil
}

// assignLevel returns the level for a new node, drawn from a
// geometric distribution: level = floor(-ln(rand()) * mL). With M=16
// the mean is 1/ln(M) ≈ 0.36, so most nodes are at level 0 and the
// graph thins exponentially upward.
func (h *hnsw) assignLevel() int {
	r := h.rng.Float64()
	if r == 0 {
		return 0
	}
	return int(-math.Log(r) * h.mL)
}

// Stats returns a snapshot of runtime statistics.
func (h *hnsw) Stats() Stats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return Stats{
		Size:            h.arena.Size() - len(h.tombstoned),
		Tombstoned:      len(h.tombstoned),
		SearchOps:       h.searchOps.Load(),
		InsertOps:       h.insertOps.Load(),
		Expired:         h.expiredCount.Load(),
		QuotaRejects:    h.quotaRejects.Load(),
		FilterRejects:   h.filterRejects.Load(),
		FilterGates:     h.filterGates.Load(),
		ComplementGates: h.complementGates.Load(),
		ColumnGates:     h.columnGates.Load(),
		ColumnDrops:     h.columnDrops.Load(),
		SparseVectors:   h.countSparse(),
		SearchLatency:   h.searchLat.snapshot(),
		InsertLatency:   h.insertLat.snapshot(),
	}
}

// countSparse returns the number of live (non-tombstoned) slots that carry a
// sparse vector. Must be called with h.mu held. O(live) — fine for Stats,
// which is metering, not a hot path.
func (h *hnsw) countSparse() int {
	n := 0
	for _, slot := range h.arena.idMap {
		if h.tombstoned[slot] {
			continue
		}
		if h.arena.sparse[slot] != nil {
			n++
		}
	}
	return n
}

// scorer returns the distance from a fixed query to the vector at slot.
// searchLayer is agnostic to how the distance is computed: the build path and
// unquantized search use exact float32 distance, while quantized search uses
// the quantizer's approximate distance over codes.
type scorer func(slot uint32) float32

// metricDist returns the configured metric's distance function, dimension-aware
// (the wider AVX-512 kernels engage at high dim on capable CPUs — see
// pickDistDim). Called per scorer/selection setup, not per distance, so the
// dispatch is free relative to the work it guards. Reads h.cfg, so it is correct
// on every hnsw regardless of construction path (newHNSW, snapshot, persist).
func (h *hnsw) metricDist() distFunc {
	return pickDistDim(h.cfg.Metric, h.cfg.Dim)
}

// exactScorer returns a scorer computing the exact float32 distance between q
// and each candidate's stored vector under the configured metric. Used on the
// build path (graph quality) and by unquantized search. It reads arena.Vec, so
// it carries the raw-vector prefetcher.
func (h *hnsw) exactScorer(q []float32) layerScorer {
	dist := h.metricDist()
	return layerScorer{
		score: func(slot uint32) float32 { return dist(q, h.arena.Vec(slot)) },
		pf:    h.vecPrefetcher(),
		batch: h.batchExact(q),
	}
}

// batchExact returns the batched counterpart of exactScorer's per-slot scorer,
// or nil when no batched kernel applies — in which case searchLayerCore stays on
// the per-pair path, which is always correct and merely slower.
//
// It returns nil when:
//   - the raw vectors are not materialized (a quantized-only arena whose float
//     slab was dropped), where exactScorer itself would not be usable either;
//   - the arena's stride disagrees with cfg.Dim or with the query length. The
//     kernel derives every candidate address from a single (base, dim) pair, so
//     a mismatch there is not a slow path but a wrong one — better to decline;
//   - the metric has no batched kernel on this platform (pickNy returns nil).
//
// The slab header is captured once per traversal, matching what vecPrefetcher
// already does. That is safe because the arena can only grow or be swapped under
// h.mu, which every traversal holds at least read-shared for its whole duration;
// nyBoundsOK turns any violation of that invariant into a clear failure at the
// offending slot rather than a silent out-of-slab read.
func (h *hnsw) batchExact(q []float32) batchKernel {
	a := h.arena
	if !batchedExpand || a == nil || a.vecsDropped || a.dim <= 0 || len(a.vecs) == 0 {
		return batchKernel{}
	}
	if a.dim != h.cfg.Dim || len(q) != a.dim {
		return batchKernel{}
	}
	ny := pickNy(h.cfg.Metric, a.dim)
	if ny == nil {
		return batchKernel{}
	}
	return batchKernel{ny: ny, q: q, base: a.vecs, dim: a.dim, metric: h.cfg.Metric}
}

// trainAndEncodePQ trains the PQ codebooks on sample (the placed, normalized
// vectors), swaps the trained codec into the pqQuantizer, and encodes every live
// slot's code into the arena. Called by BuildConcurrent after placement and
// before the link phase. PLAIN (non-residual) global PQ: HNSW has no coarse
// centroids, so the sample vectors are passed to trainPQ directly. Reuses the
// parallel k-means (workers). After it returns h.quant.trained() is true, so the
// search path navigates on ADC and the snapshot can re-encode from the codebooks
// (the caller serializes them). Must be called under h.mu.
func (h *hnsw) trainAndEncodePQ(sample [][]float32, workers int) error {
	trained, err := h.trainPQOnly(sample, workers)
	if err != nil || !trained {
		return err
	}
	// Encode every placed slot now that the codebooks exist. The arena's codes
	// side-array was sized to n*CodeLen() by Reserve; write each code in place.
	// CONTRACT: this index-keyed encode is correct ONLY when sample[i] lives at
	// arena slot i — i.e. the DENSE BuildConcurrent path (PutAt packs slots
	// 0..n-1 with no holes). The INCREMENTAL auto-train path must NOT use this
	// loop (its sample is COMPACTED over holes, so sample[i] != slot i); it calls
	// trainPQOnly + encodeLiveSlotsLockedPQ, which keys the encode by real slot.
	cl := h.quant.CodeLen()
	buf := make([]byte, cl)
	for i := range sample {
		slot := uint32(i) //nolint:gosec // i < n, bounded by 2^32 vectors per arena
		h.quant.Encode(buf, sample[i])
		h.arena.SetCode(slot, buf)
	}
	return nil
}

// trainPQOnly trains the PQ codebooks on sample and swaps the trained codec into
// the pqQuantizer, but does NOT encode any slot. It is the codebook-train half of
// trainAndEncodePQ, factored out so the INCREMENTAL auto-train path can train on
// the (compacted) live sample yet encode SLOT-CORRECTLY afterwards (the dense
// BuildConcurrent path keeps the index-keyed encode in trainAndEncodePQ).
// Returns (true, nil) once the codec is trained, (false, nil) for a non-PQ index
// (nothing to train). Must hold h.mu. trained() flips true on success, so the
// search path then navigates on ADC and the snapshot can re-encode from codebooks.
func (h *hnsw) trainPQOnly(sample [][]float32, workers int) (bool, error) {
	pq, ok := h.quant.(*pqQuantizer)
	if !ok {
		return false, nil // not a PQ index; nothing to train
	}
	seed := h.cfg.Seed
	if seed == 0 {
		seed = 1
	}
	m := h.cfg.QuantPQM
	if m <= 0 {
		m = defaultPQM(h.cfg.Dim)
	}
	// Train on L2-internal codebooks (trainPQ minimizes sub-vector reconstruction
	// error regardless of metric); the collection metric drives the ADC LUT.
	// OPQ (cfg.OPQ): trainPQ builds R and trains the codebooks on the rotated
	// vectors Rx; encode/queryLUT rotate before the subspace split, reconstruct
	// un-rotates (Rᵀ). HNSW-PQ is PLAIN (non-residual) so R rotates the raw vec.
	// cfg.OPQ=false ⇒ rotation nil ⇒ byte-identical to plain PQ.
	codec, err := trainPQ(sample, m, h.cfg.Dim, seed, h.cfg.Metric, workers, h.cfg.OPQ, h.cfg.OPQIters, h.cfg.AnisotropicEta, pq.nbits)
	if err != nil {
		return false, err
	}
	pq.setCodec(codec)
	return true, nil
}

// encodeLiveSlotsLockedPQAt is encodeLiveSlotsLockedPQ judging slot liveness against
// the caller-supplied `now` (unix millis), so the auto-train re-encode covers the
// SAME live slot set as the training sample on every replica (#4 vector TTL
// determinism). Must hold h.mu.
func (h *hnsw) encodeLiveSlotsLockedPQAt(now uint64) {
	cl := h.quant.CodeLen()
	buf := make([]byte, cl)
	capacity := h.arena.Capacity()
	for s := 0; s < capacity; s++ {
		slot := uint32(s) //nolint:gosec // s < Capacity() < 2^32
		if !h.liveSlotLockedAt(slot, now) {
			continue
		}
		h.quant.Encode(buf, h.arena.Vec(slot))
		h.arena.SetCode(slot, buf)
	}
}

// pqUntrained reports whether the index uses a PQ quantizer whose codebooks have
// not yet been trained (codes are not meaningful). PQ-HNSW is training-free until
// BuildConcurrent runs trainPQ; an incremental Insert before that point keeps the
// floats and every distance path falls back to EXACT float32 (mirrors IVF's
// untrained=exact policy). Always false for sq8/bq1/none (training-free) and for
// a trained PQ index, so those paths are byte-identical to before. See
// searchScorer/buildScorer below and the train hook in BuildConcurrent.
func (h *hnsw) pqUntrained() bool {
	pq, ok := h.quant.(*pqQuantizer)
	return ok && !pq.trained()
}

// sqUntrained reports whether the index uses a TRAINED scalar quantizer
// (QuantSQ) whose per-dimension ranges have not yet been learned (codes are
// zero placeholders). Like PQ-HNSW, trained-SQ is training-free until the build
// sample is seen: an incremental Insert before that point keeps the floats and
// every distance path falls back to EXACT float32 (mirrors pqUntrained). Always
// false for the fixed-scale sq8/bq1/none (training-free) and for a trained SQ
// index, so those paths are byte-identical to before. See searchScorer/
// quantizedBuild and the train hooks below.
func (h *hnsw) sqUntrained() bool {
	sq, ok := h.quant.(*trainedSQ)
	return ok && !sq.trained()
}

// trainAndEncodeSQ learns the per-dimension SQ ranges on sample (the placed,
// normalized vectors), swaps them into the trainedSQ, and encodes every live
// slot's code into the arena. The SQ analogue of trainAndEncodePQ, called by
// BuildConcurrent after placement and before the link phase. Like the PQ hook,
// the index-keyed encode is correct ONLY on the DENSE BuildConcurrent path
// (sample[i] lives at arena slot i); the incremental auto-train path uses
// trainSQOnly + encodeLiveSlotsLockedPQ (slot-keyed). Must hold h.mu.
func (h *hnsw) trainAndEncodeSQ(sample [][]float32) error {
	if !h.trainSQOnly(sample) {
		return nil
	}
	cl := h.quant.CodeLen()
	buf := make([]byte, cl)
	for i := range sample {
		slot := uint32(i) //nolint:gosec // i < n, bounded by 2^32 vectors per arena
		h.quant.Encode(buf, sample[i])
		h.arena.SetCode(slot, buf)
	}
	return nil
}

// trainSQOnly learns the SQ ranges on sample and swaps them into the trainedSQ,
// but encodes no slot. The range-train half of trainAndEncodeSQ, factored out so
// the incremental auto-train path can train on the (compacted) live sample yet
// encode SLOT-CORRECTLY afterwards. Returns true once the quantizer is trained,
// false for a non-SQ index (nothing to train) or an empty sample. Must hold h.mu.
// trained() flips true on success so the search path then navigates on codes and
// the snapshot can re-encode from the learned ranges.
func (h *hnsw) trainSQOnly(sample [][]float32) bool {
	sq, ok := h.quant.(*trainedSQ)
	if !ok || len(sample) == 0 {
		return false
	}
	t := trainSQ(sample, h.cfg.Dim, h.cfg.SQBits, h.cfg.Metric)
	if !t.trained() {
		return false
	}
	sq.setRanges(t.min, t.max)
	return true
}

// shouldAutoTrainSQ reports whether an UNTRAINED incremental HNSW-SQ index has
// accumulated enough live vectors to deterministically auto-train its ranges. It
// mirrors shouldAutoTrainPQ EXACTLY: a PURE FUNCTION of applied state (the live
// count) and the replicated Config threshold, so it evaluates IDENTICALLY on
// every replica. Only ever true for a QuantSQ index whose ranges are unlearned —
// every non-SQ / already-trained index returns false, so those paths are
// byte-identical. Must hold h.mu (write).
func (h *hnsw) shouldAutoTrainSQ() bool {
	if h.cfg.Quant != QuantSQ || !h.sqUntrained() {
		return false
	}
	threshold := h.cfg.IVFTrainThreshold
	if threshold <= 0 {
		threshold = defaultIVFTrainThreshold
	}
	return h.liveCount() >= threshold
}

// autoTrainLockedSQ deterministically trains the incrementally-built HNSW-SQ
// index in place: it gathers the live vectors as a SLOT-ORDERED sample
// (deterministic — never Go-map order, via liveSampleLockedPQ) and learns the
// per-dimension ranges, then re-encodes every live slot SLOT-CORRECTLY (the
// sample is compacted over holes, so encode by real slot via
// encodeLiveSlotsLockedPQ). Unlike PQ there is no float-drop. Must hold h.mu
// (write) and be called only when shouldAutoTrainSQ() is true (train-once:
// afterwards sqUntrained() is false).
func (h *hnsw) autoTrainLockedSQ(now uint64) error {
	sample := h.liveSampleLockedPQAt(now)
	if len(sample) == 0 {
		return nil
	}
	if !h.trainSQOnly(sample) {
		return nil
	}
	h.encodeLiveSlotsLockedPQAt(now)
	return nil
}

// prqUntrained reports whether the index uses a PRQ quantizer whose layer codebooks
// have not yet been trained (codes are zero placeholders). Like PQ-HNSW, PRQ-HNSW is
// training-free until the build sample is seen: an incremental Insert before that
// point keeps the floats and every distance path falls back to EXACT float32
// (mirrors pqUntrained). Always false for the non-PRQ modes and for a trained PRQ
// index, so those paths are byte-identical. See searchScorer/quantizedBuild and the
// train hooks below.
func (h *hnsw) prqUntrained() bool {
	prq, ok := h.quant.(*prqQuantizer)
	return ok && !prq.trained()
}

// trainPRQOnly trains the L-layer PRQ codebooks on sample and swaps the trained
// codec into the prqQuantizer, but encodes no slot. The codebook-train half of
// trainAndEncodePRQ, factored out so the INCREMENTAL auto-train path can train on
// the (compacted) live sample yet encode SLOT-CORRECTLY afterwards. Returns
// (true, nil) once the codec is trained, (false, nil) for a non-PRQ index. Must
// hold h.mu. OPQ (cfg.OPQ) is applied ONCE before layer 0 (trainPRQ threads it into
// layer 0's trainPQ); cfg.OPQ=false ⇒ no rotation. trained() flips true on success.
func (h *hnsw) trainPRQOnly(sample [][]float32, workers int) (bool, error) {
	prq, ok := h.quant.(*prqQuantizer)
	if !ok {
		return false, nil // not a PRQ index; nothing to train
	}
	seed := h.cfg.Seed
	if seed == 0 {
		seed = 1
	}
	m := h.cfg.QuantPQM
	if m <= 0 {
		m = defaultPQM(h.cfg.Dim)
	}
	l := h.cfg.PRQLayers
	if l <= 0 {
		l = defaultPRQLayers
	}
	codec, err := trainPRQ(sample, m, h.cfg.Dim, l, seed, h.cfg.Metric, workers, h.cfg.OPQ, h.cfg.OPQIters)
	if err != nil {
		return false, err
	}
	prq.setCodec(codec)
	return true, nil
}

// trainAndEncodePRQ trains the PRQ codebooks on sample (the placed, normalized
// vectors), swaps the trained codec into the prqQuantizer, and encodes every live
// slot's code into the arena. The PRQ analogue of trainAndEncodePQ, called by
// BuildConcurrent after placement and before the link phase. The index-keyed encode
// is correct ONLY on the DENSE BuildConcurrent path (sample[i] lives at arena slot
// i); the incremental auto-train path uses trainPRQOnly + encodeLiveSlotsLockedPQ
// (slot-keyed). Must hold h.mu.
func (h *hnsw) trainAndEncodePRQ(sample [][]float32, workers int) error {
	trained, err := h.trainPRQOnly(sample, workers)
	if err != nil || !trained {
		return err
	}
	cl := h.quant.CodeLen()
	buf := make([]byte, cl)
	for i := range sample {
		slot := uint32(i) //nolint:gosec // i < n, bounded by 2^32 vectors per arena
		h.quant.Encode(buf, sample[i])
		h.arena.SetCode(slot, buf)
	}
	return nil
}

// shouldAutoTrainPRQ reports whether an UNTRAINED incremental HNSW-PRQ index has
// accumulated enough live vectors to deterministically auto-train its codebooks. It
// mirrors shouldAutoTrainPQ EXACTLY: a PURE FUNCTION of applied state (the live
// count) and the replicated Config threshold, so it evaluates IDENTICALLY on every
// replica. Only ever true for a QuantPRQ index whose codebooks are unlearned — every
// non-PRQ / already-trained index returns false, so those paths are byte-identical.
// Must hold h.mu (write).
func (h *hnsw) shouldAutoTrainPRQ() bool {
	if h.cfg.Quant != QuantPRQ || !h.prqUntrained() {
		return false
	}
	threshold := h.cfg.IVFTrainThreshold
	if threshold <= 0 {
		threshold = defaultIVFTrainThreshold
	}
	return h.liveCount() >= threshold
}

// autoTrainLockedPRQ deterministically trains the incrementally-built HNSW-PRQ index
// in place: it gathers the live vectors as a SLOT-ORDERED sample (deterministic —
// never Go-map order, via liveSampleLockedPQ) and trains the L-layer codebooks, then
// re-encodes every live slot SLOT-CORRECTLY (the sample is compacted over holes, so
// encode by real slot via encodeLiveSlotsLockedPQ). Must hold h.mu (write) and be
// called only when shouldAutoTrainPRQ() is true (train-once: afterwards
// prqUntrained() is false). Unlike PQ there is no float-drop for PRQ in v1.
func (h *hnsw) autoTrainLockedPRQ(now uint64) error {
	sample := h.liveSampleLockedPQAt(now)
	if len(sample) == 0 {
		return nil
	}
	trained, err := h.trainPRQOnly(sample, runtime.GOMAXPROCS(0))
	if err != nil {
		return err
	}
	if !trained {
		return nil
	}
	h.encodeLiveSlotsLockedPQAt(now)
	return nil
}

// shouldAutoTrainPQ reports whether an UNTRAINED incremental HNSW-PQ index has
// now accumulated enough live vectors to deterministically auto-train its
// codebooks. It mirrors ivf.shouldAutoTrain EXACTLY: a PURE FUNCTION of applied
// state (the live count = arena size) and the replicated Config threshold, so it
// evaluates IDENTICALLY on every replica (NO wall-clock, NO rand, NO Go-map
// iteration in the condition). Only ever true for a QuantPQ index whose codebooks
// have not yet trained — every non-PQ / already-trained index returns false, so
// those paths are byte-identical. Must hold h.mu (write). The threshold is
// cfg.IVFTrainThreshold (generalized to "incrementally-built quantized index"),
// defaulting to defaultIVFTrainThreshold when 0.
func (h *hnsw) shouldAutoTrainPQ() bool {
	if h.cfg.Quant != QuantPQ || !h.pqUntrained() {
		return false
	}
	threshold := h.cfg.IVFTrainThreshold
	if threshold <= 0 {
		threshold = defaultIVFTrainThreshold
	}
	return h.liveCount() >= threshold
}

// liveCount returns the number of live (present in the arena) vectors — the same
// population BuildConcurrent trains on (arena.Size() = len(idMap)). Mirrors
// ivf.liveCount. Must hold h.mu.
func (h *hnsw) liveCount() int {
	return h.arena.Size()
}

// liveSlotLocked reports whether slot currently holds a live vector: its id still
// maps to this exact slot in the arena (not reclaimed) and it is neither
// tombstoned nor TTL-expired. Mirrors ivf.liveSlotLocked. Must hold h.mu.
func (h *hnsw) liveSlotLocked(slot uint32) bool {
	return h.liveSlotLockedAt(slot, uint64(h.now()))
}

// liveSlotLockedAt is liveSlotLocked judging TTL expiry against the caller-supplied
// `now` (unix millis). Threaded from insertLockedAt's synchronous auto-train hook so
// the quantizer training SAMPLE (and its slot-correct re-encode) exclude the SAME
// expired slots on every replica — otherwise a stamped insert that triggers
// auto-train could train a divergent codebook from wall-clock-skewed liveness (#4
// vector TTL determinism). Must hold h.mu.
func (h *hnsw) liveSlotLockedAt(slot uint32, now uint64) bool {
	id := h.arena.ID(slot)
	if cur, ok := h.arena.idMap[id]; !ok || cur != slot {
		return false
	}
	return !h.tombstoned[slot] && !h.isExpiredAt(slot, now)
}

// liveSampleLockedPQAt is liveSampleLockedPQ judging slot liveness against the
// caller-supplied `now` (unix millis), so the auto-train sample excludes the SAME
// expired slots on every replica (#4 vector TTL determinism). Must hold h.mu.
func (h *hnsw) liveSampleLockedPQAt(now uint64) [][]float32 {
	capacity := h.arena.Capacity()
	sample := make([][]float32, 0, h.arena.Size())
	for s := 0; s < capacity; s++ {
		slot := uint32(s) //nolint:gosec // s < Capacity() < 2^32
		if !h.liveSlotLockedAt(slot, now) {
			continue
		}
		sample = append(sample, append([]float32(nil), h.arena.Vec(slot)...))
	}
	return sample
}

// autoTrainLockedPQ deterministically trains the incrementally-built HNSW-PQ
// index in place: it gathers the live vectors as a SLOT-ORDERED sample
// (deterministic — never Go-map order) and runs the EXACT same train+encode-all
// sequence BuildConcurrent uses (trainAndEncodePQ: trainPQ on cfg.Seed → encode
// every slot), then folds in the float-drop exactly as BuildConcurrent does
// (gated on cfg.PQDropVecs && a trained codec). Must hold h.mu (write) and be
// called only when shouldAutoTrainPQ() is true (i.e. !pqUntrained() afterwards →
// train-once). k-means is worker-count-invariant, so runtime.GOMAXPROCS(0)
// workers yields the identical result on every replica regardless of cores.
func (h *hnsw) autoTrainLockedPQ(now uint64) error {
	sample := h.liveSampleLockedPQAt(now)
	if len(sample) == 0 {
		return nil
	}
	// Train the codebooks on the compacted (slot-ordered, hole-skipping) live
	// sample — its order/content is deterministic and unaffected by holes, so the
	// codebooks are identical to BuildConcurrent's on the same live set. But do NOT
	// reuse trainAndEncodePQ's index-keyed encode: sample[i] is COMPACTED over
	// holes, so it does NOT live at slot i once any tombstoned/expired/reclaimed
	// slot sits below a live one. Encode by REAL arena slot instead (mirrors IVF's
	// rebuildListsLocked) so every slot holds ITS OWN vector's code.
	trained, err := h.trainPQOnly(sample, runtime.GOMAXPROCS(0))
	if err != nil {
		return err
	}
	if !trained {
		return nil // not a PQ index; nothing encoded
	}
	h.encodeLiveSlotsLockedPQAt(now)
	// Float-drop: mirror BuildConcurrent's guard. After train the codec is trained
	// (pqUntrained() false) and every slot encoded; release the resident floats so
	// only the M-byte codes stay in RAM. PQDropVecs=false ⇒ no-op (byte-identical).
	if h.cfg.PQDropVecs && !h.pqUntrained() {
		h.arena.dropVecs()
	}
	return nil
}

// vecsDropped reports whether the resident float32 vectors have been released
// (the PQDropVecs maximum-compression state: only the M-byte codes stay in RAM).
// It delegates to the arena's vecsDropped flag, flipped by arena.dropVecs() at
// the end of BuildConcurrent. Only ever true under a trained QuantPQ index (the
// drop is gated on cfg.PQDropVecs && a trained codec), so the float scorer / Vec
// reads are never reached on the dropped path. False for every non-PQDropVecs
// index ⇒ those paths are byte-identical to before. Read under h.mu.
func (h *hnsw) vecsDropped() bool { return h.arena.vecsDropped }

// vecFor returns slot's float vector for the EXACT (non-navigation) paths —
// Get / MMR / Recommend / Discover / reshard-scan / the rescore fallback. When
// the resident floats are present it returns arena.Vec directly (exact, aliases
// arena storage). After PQDropVecs drops them it reconstructs an APPROXIMATE
// vector from the slot's PQ code via the codec (un-rotates OPQ Rᵀ inside the
// codec). There is no coarse centroid in HNSW (unlike IVF.vecFor), so the
// reconstruct is the whole vector. The returned slice is freshly allocated when
// reconstructed and aliases arena storage otherwise; callers that retain it must
// clone. Must hold h.mu (read). Mirrors ivf.vecFor.
func (h *hnsw) vecFor(slot uint32) []float32 {
	// Under-lock backstop for the materialization paths (Get/GetProjected/GetInto,
	// discover context pairs, MMR diversity vecs): a failed mmap grow nils the
	// arena floats, so arena.Vec would slice a nil region and panic. Return nil
	// instead. The dense search scorer does NOT route through here (it reads the
	// arena base pointer directly), so this adds no atomic load to the hot loop;
	// that path is guarded once at the top of searchDenseLockedWith. See
	// arena.poisoned.
	if h.arena.poisoned.Load() {
		return nil
	}
	if !h.arena.vecsDropped {
		return h.arena.Vec(slot)
	}
	// Dropped: reconstruct from the code. vecsDropped is only ever set under a
	// trained pqQuantizer (BuildConcurrent gates the drop on it), so this type
	// assertion always holds on this path.
	if pq, ok := h.quant.(*pqQuantizer); ok {
		return pq.reconstruct(h.arena.Code(slot))
	}
	return nil // unreachable: vecsDropped implies a trained PQ codec
}

// quantizedBuild reports whether the build navigates/selects on quantizer codes
// (Config.QuantizedBuild) rather than exact float32 — 4x less memory traffic per
// distance, the dominant lever for high-dim build speed, at a small graph-quality
// cost the query-time rescore recovers.
func (h *hnsw) quantizedBuild() bool {
	// PQ-HNSW is excluded: it KEEPS the float vectors and builds the graph on
	// EXACT float32 distances for maximum graph quality (search then navigates on
	// ADC codes and exact-rescores the shortlist). Code-navigation build is the
	// sq8/bq1 lever (4x less memory traffic); PQ codes also exist only after the
	// BuildConcurrent train hook, so an exact-float build also sidesteps the
	// untrained window. sq8/bq1 behaviour is unchanged (they are not pqQuantizers).
	if _, isPQ := h.quant.(*pqQuantizer); isPQ {
		return false
	}
	// PRQ-HNSW is excluded for the same reasons as PQ: it KEEPS the float vectors,
	// builds the graph on EXACT float32 (the link phase reads arena.Vec), and its
	// codes exist only after the train hook — so an exact-float build sidesteps the
	// untrained window and maximizes graph quality (search then navigates on the
	// summed-LUT ADC and exact-rescores the shortlist).
	if _, isPRQ := h.quant.(*prqQuantizer); isPRQ {
		return false
	}
	// An UNTRAINED trained-SQ index has only zero placeholder codes (its ranges are
	// learned at the auto-train threshold), so navigate on EXACT float32 until then
	// — code-navigation would score garbage. Once trained, code-navigation engages
	// like sq8/bq1. The fixed-scale sq8/bq1 are training-free (sqUntrained false),
	// so their behaviour is unchanged.
	if h.sqUntrained() {
		return false
	}
	return h.cfg.QuantizedBuild && h.quant != nil
}

// buildSymNav makes QuantizedBuild navigation symmetric: the inserted node is
// encoded to its int8 code once and the graph descent scores code-vs-code via
// the VNNI/AVX2 symmetric kernel, instead of the asymmetric float-query-vs-code
// kernel. A CPU profile of the high-dim quantized build showed navigation
// (searchLayer) is ~38% of the build and was running on the slower AVX2
// asymmetric kernel while only neighbor selection used the fast symmetric one —
// this routes both halves through the symmetric kernel (Qdrant builds entirely
// on int8 too). A var (not const) so the build A/B can flip it for comparison.
var buildSymNav = true

// buildScorer returns the scorer used to link a new node into the graph: the
// quantizer's approximate distance over codes when QuantizedBuild is set, else
// exact float32 (the default, for maximum graph quality). Under QuantizedBuild
// with buildSymNav, navigation is symmetric code-vs-code (the inserted node is
// encoded once into s.qcode); otherwise it is the asymmetric float-query scorer.
func (h *hnsw) buildScorer(s *layerScratch, stored []float32) layerScorer {
	if h.quantizedBuild() {
		if buildSymNav {
			cl := h.quant.CodeLen()
			if cap(s.qcode) < cl {
				s.qcode = make([]byte, cl)
			} else {
				s.qcode = s.qcode[:cl]
			}
			h.quant.Encode(s.qcode, stored)
			qc := s.qcode
			q := h.quant
			a := h.arena
			return layerScorer{
				score: func(slot uint32) float32 { return q.CodeDistance(qc, a.Code(slot)) },
				pf:    h.codePrefetcher(),
			}
		}
		return h.searchScorer(stored)
	}
	// The default build path — and, note, the ONLY path a PQ or PRQ index ever
	// takes, since quantizedBuild() is false for both. It scores raw floats, so
	// exactScorer's vector prefetcher is what must travel with it.
	return h.exactScorer(stored)
}

// initPairFns allocates the two build-time pair-distance closures. Both read
// h.arena/h.quant through h on every call, so they stay correct across an arena
// swap or a quantizer train.
//
// It RESOLVES cfg.Metric/cfg.Dim to a kernel here and captures it, so this is
// NOT a construct-once-and-forget: it must run at construction AND again after
// any write to h.cfg. readSnapshot replaces h.cfg wholesale from the stream
// (the restored index can be a different metric AND dimension than the config
// it was constructed with — RestoreCollection deliberately constructs from a
// Cosine dim=1 placeholder), so it re-runs this alongside its mL recompute.
func (h *hnsw) initPairFns() {
	d := pickDistDim(h.cfg.Metric, h.cfg.Dim)
	h.pairExact = func(x, y uint32) float32 { return d(h.arena.Vec(x), h.arena.Vec(y)) }
	h.pairCode = func(x, y uint32) float32 {
		return h.quant.CodeDistance(h.arena.Code(x), h.arena.Code(y))
	}
}

// buildPairFn returns the slot-to-slot distance used for build-time neighbor
// selection: symmetric code distance under QuantizedBuild (reads only codes),
// else exact float32. Returns a PRE-BUILT closure (initPairFns) — the predicate
// is still re-evaluated per call, so a mid-build auto-train switches metrics at
// exactly the same insert it always did, but nothing allocates.
func (h *hnsw) buildPairFn() func(a, b uint32) float32 {
	if h.quantizedBuild() {
		return h.pairCode
	}
	return h.pairExact
}

// searchScorer returns the scorer used to navigate the graph for query q: the
// quantizer's approximate distance over codes when quantization is enabled,
// otherwise the exact float32 distance.
func (h *hnsw) searchScorer(q []float32) layerScorer {
	// No quantizer, or a PQ index not yet trained (codebooks absent ⇒ codes are
	// placeholders): navigate on EXACT float32. Once trainPQ runs (at build) the
	// PQ branch below takes over (ADC over codes). Note this is a DIFFERENT
	// predicate from quantizedBuild()'s — which is exactly why the prefetcher has
	// to be chosen here, inside each branch, rather than re-derived elsewhere.
	if h.quant == nil || h.pqUntrained() || h.sqUntrained() || h.prqUntrained() {
		return h.exactScorer(q)
	}
	qc := h.quant.PrepareQuery(q)
	return layerScorer{
		score: func(slot uint32) float32 { return h.quant.Distance(qc, h.arena.Code(slot)) },
		pf:    h.codePrefetcher(),
	}
}

// nodeAt returns the node at slot, or nil if the slot is empty or out of range.
// Replaces the old map's nil-for-missing lookup with a direct slice index (no
// hashing) — the graph is keyed by dense slot ids in [0, arena.Capacity()).
func (h *hnsw) nodeAt(slot uint32) *node {
	if int(slot) >= len(h.nodes) {
		return nil
	}
	return h.nodes[slot]
}

// prefetchDistance is how many neighbors ahead searchLayer prefetches: enough
// that an upcoming candidate's vector is in flight (a cache miss is ~100ns)
// while the intervening distance computations run.
const prefetchDistance = 4

// slotPrefetcher pulls the data score() will load for a slot toward the core —
// the quantized code when quantization is on, else the float32 vector. It hides
// the cache-miss latency that dominates graph traversal, where candidate
// vectors live at random arena slots.
//
// It exists as a hoisted (base, stride, lines) triple rather than a method on
// hnsw because the per-neighbor version cost more than it saved: it re-tested
// h.quant (an interface, loop-invariant for a whole traversal) on every
// neighbor, paid a full CALL for a single PREFETCHT0, and that one instruction
// covered 64 of a 512-byte vector — so the lookahead warmed the first line and
// the other seven still missed. Resolving the layout once per traversal and
// issuing the whole record in one call turns the chain into a bounds check plus
// a short PREFETCHT0 run.
type slotPrefetcher struct {
	base    unsafe.Pointer // start of the per-slot array (vectors or codes)
	stride  uintptr        // bytes per slot
	lines   int            // 64-byte lines covering one slot's record
	maxSlot uint32         // exclusive bound: slots at/after this are not backed
}

// codePrefetcher targets the arena's quantizer-code array — the right choice
// ONLY for a scorer that actually reads codes. A zero value (maxSlot == 0)
// disables prefetching, which is the correct result when codes are not
// materialized; every prefetch is a pure hint, so warming nothing is always
// safe.
func (h *hnsw) codePrefetcher() slotPrefetcher {
	a := h.arena
	if a.codeLen <= 0 || len(a.codes) == 0 {
		return slotPrefetcher{}
	}
	return slotPrefetcher{
		base:    unsafe.Pointer(&a.codes[0]),
		stride:  uintptr(a.codeLen),
		lines:   (a.codeLen + 63) / 64,
		maxSlot: uint32(len(a.codes) / a.codeLen),
	}
}

// vecPrefetcher targets the arena's raw float32 vectors — the right choice for
// any exact scorer. Zero value semantics match codePrefetcher.
func (h *hnsw) vecPrefetcher() slotPrefetcher {
	a := h.arena
	if a.dim <= 0 || len(a.vecs) == 0 {
		return slotPrefetcher{}
	}
	stride := a.dim * 4 // float32
	return slotPrefetcher{
		base:    unsafe.Pointer(&a.vecs[0]),
		stride:  uintptr(stride),
		lines:   (stride + 63) / 64,
		maxSlot: uint32(len(a.vecs) / a.dim),
	}
}

// layerScorer pairs a per-slot distance function with a prefetcher for the EXACT
// arena array that function reads.
//
// They are bundled because deriving them from separate predicates is precisely
// how this went wrong before: the prefetcher branched on `h.quant != nil` while
// the scorer branched on quantizedBuild() (unconditionally false for PQ and PRQ)
// or on the untrained checks. On a PQ index the build path therefore scored raw
// 3072-byte vectors while the prefetcher warmed a single line of PQ codes it
// never touched — buying nothing and evicting lines that were about to be read.
//
// Every constructor below returns both halves from the SAME branch, so a new
// scorer variant cannot compile without also stating what it reads.
type layerScorer struct {
	score scorer
	pf    slotPrefetcher

	// batch scores a whole block of candidate slots against the same query in
	// one call. Its zero value means "no batched kernel covers this scorer", and
	// the traversal stays on score. It is a third member of the same bundle for
	// the same reason the prefetcher is: choosing it apart from score is how the
	// two could disagree about which arena array — or which metric — they read.
	//
	// Only the exact float32 scorers set it. Every quantized scorer reads
	// arena.Code through a quantizer interface whose per-slot Distance has no
	// batched form, so those keep the per-pair path; so does any metric or
	// platform without a kernel. searchLayerCore branches on it ONCE per
	// traversal, not per candidate.
	batch batchKernel
}

// batchKernel computes the distance from a traversal's fixed query to every slot
// in a block, writing len(slots) results into out (which the caller sizes).
//
// It is a STRUCT rather than a closure on purpose. The obvious spelling —
// `batch func(slots []uint32, out []float32)` built by capturing the query, the
// slab and the metric — costs one heap allocation per search, because those
// captures escape into the closure. That showed up immediately as +79 B/op on
// BenchmarkHNSWSearchInto, a path whose whole point is to allocate nothing in
// steady state. Holding the same four values inline and reaching them through a
// method keeps the search path allocation-free and drops an indirect call.
type batchKernel struct {
	ny     nyFunc    // nil for "no batched kernel"; the raw L2 or dot kernel
	q      []float32 // the traversal's fixed query
	base   []float32 // the arena's flat vector slab
	dim    int       // slab stride, in float32 elements
	metric Metric
}

// ok reports whether this scorer has a batched kernel. Value receiver so it can
// be asked of a batchExact result directly; score keeps a pointer receiver to
// avoid copying the struct on every block.
func (b batchKernel) ok() bool { return b.ny != nil }

// score fills out[i] with the distance from the query to slots[i].
//
// The metric transform is applied over the finished block rather than inside the
// kernel, which is what lets Cosine and DotProduct share one dot kernel — and
// mirrors pickDist, where the same two transforms wrap the same per-pair kernel.
// The branch is per block, not per candidate.
func (b *batchKernel) score(slots []uint32, out []float32) {
	b.ny(b.q, b.base, b.dim, slots, out)
	switch b.metric {
	case Cosine:
		// Cosine over pre-normalized vectors is 1 - dot.
		for i, d := range out {
			out[i] = 1 - d
		}
	case DotProduct:
		for i, d := range out {
			out[i] = -d
		}
	}
}

// slot issues the prefetch for one slot's record. Out-of-range slots are
// dropped rather than clamped: a prefetch is advisory, so a skipped one costs
// only a miss later.
//
// The bound keeps the START of the walk inside the arena's backing array; the
// last line of a record whose length is not a multiple of 64 reaches up to 63
// bytes past its end, and for the final slot that is past the array. PREFETCHT0
// and PRFM are defined not to fault or trap on an inaccessible address, so this
// is safe FOR THESE TWO INSTRUCTIONS ONLY — anything added here that actually
// dereferences memory must re-derive its own bound rather than inherit this one.
func (p *slotPrefetcher) slot(s uint32) {
	if s >= p.maxSlot {
		return
	}
	prefetchRange(unsafe.Add(p.base, uintptr(s)*p.stride), p.lines)
}

// setNode places nd at slot, growing the node table and level-0 slab as needed
// so slot is addressable (append for a fresh slot, overwrite for a reused one).
// Returns an error only when an mmap-backed slab fails to remap on growth.
func (h *hnsw) setNode(slot uint32, nd *node) error {
	if err := h.ensureGraphSlot(slot); err != nil {
		return err
	}
	h.nodes[slot] = nd
	return nil
}

// neighborsAt returns node nd's level-lc neighbor slots. With no link phase in
// flight it returns the live slice directly — one atomic load and a branch,
// which is what the quiescent read path costs. While anything IS linking (a
// bulk build, or an incremental insert now that its link phase runs under the
// read lock) it copies the slice into the caller's per-goroutine scratch under
// the slot's link stripe, so the caller can iterate without holding the lock
// while a linker appends to or prunes that same list. The returned slice is
// valid only until the next neighborsAt call on the same scratch.
//
// See link_stripes.go for why observing a zero count is sufficient license to
// skip the lock entirely.
func (h *hnsw) neighborsAt(s *layerScratch, nd *node, lc int) []uint32 {
	if !h.linking() {
		return h.nbrsAt(nd, lc)
	}
	lk := h.stripe(nd.slot)
	lk.Lock()
	if lc <= nd.level {
		s.nbrScratch = append(s.nbrScratch[:0], h.nbrsAt(nd, lc)...)
	} else {
		s.nbrScratch = s.nbrScratch[:0]
	}
	lk.Unlock()
	return s.nbrScratch
}

// searchLayer performs greedy nearest-neighbor search at a single level,
// returning up to `ef` candidates sorted ascending by distance. entryPoints
// are the slots to start from (their distances are computed inside).
// Tombstoned slots are still TRAVERSED (their edges are followed) but are
// not returned in the result set — this matches the canonical algorithm
// while letting Delete remain lazy.
// searchLayer uses the caller-provided scratch (s) for all per-traversal
// buffers and writes its result into s.out, returning a slice that aliases it.
// The caller must consume the returned slice before the next searchLayer call
// on the same scratch (each call resets s.out). entryPoints may alias s.cur.
// now is the per-search wall-clock snapshot (unix millis) the caller reads once
// and threads in, so the admission gate does no per-candidate clock read.
func (h *hnsw) searchLayer(s *layerScratch, sc layerScorer, entryPoints []uint32, ef int, level int, pred Predicate, now uint64) []slotDist {
	// The admission closure is allocated ONCE per traversal call (not per
	// candidate): negligible against the graph walk it gates.
	return h.searchLayerCore(s, sc, entryPoints, ef, level, func(slot uint32) bool {
		return h.admitsScratch(s, slot, pred, now)
	})
}

// searchLayerCore is the ONE copy of the single-level greedy traversal shared by
// searchLayer and searchLayerWith. It is parameterized only by the per-candidate
// admission gate `admit` (whether a traversed slot is eligible for the result
// set); everything else — visited-set reset, entry-point seeding, the ef-based
// termination condition, neighbor prefetch, heap pop/push, and the result drain
// into s.out — is identical regardless of how admission is decided. `admit` is
// invoked exactly where the inline h.admits/h.admitsWith calls used to be, so
// stat counters (e.g. filterRejects, tallied inside the admit funcs) fire on
// the same slots in the same order as before. The closure is captured once by
// the caller, never reallocated inside the candidate loop.
//
// The admit closures the two callers pass tally filter rejections into
// s.filterRejects; this function owns the single flush of that tally to the
// shared atomic on return (see admitsScratch).
func (h *hnsw) searchLayerCore(s *layerScratch, sc layerScorer, entryPoints []uint32, ef int, level int, admit func(slot uint32) bool) []slotDist {
	s.reset(h.arena.Capacity())
	// Deferred so a panic recovered further up the stack (the admit closure runs
	// a user-supplied predicate) still publishes the tally instead of silently
	// dropping this traversal's rejections.
	defer func() {
		if s.filterRejects != 0 {
			h.filterRejects.Add(s.filterRejects)
			s.filterRejects = 0
		}
	}()
	// The prefetcher travels WITH the scorer (see layerScorer), so it always warms
	// the array score actually reads — no second predicate to drift out of sync.
	// batch is resolved here too, ONCE for the whole traversal: whether a layer
	// expands its neighbors one at a time or a block at a time is a property of
	// the scorer, not of any individual candidate.
	score, pf, batch := sc.score, sc.pf, sc.batch

	for _, ep := range entryPoints {
		if s.visited.seen(ep) {
			continue
		}
		s.visited.mark(ep)
		key := packKey(ep, score(ep))
		s.candidates.push(key)
		if admit(ep) {
			s.nearest.push(key)
		}
	}

	for s.candidates.len() > 0 {
		c := s.candidates.pop()
		// Termination: if the closest unexplored candidate is further than
		// the furthest "kept" nearest, no closer neighbor can be reached.
		// Compared on the DISTANCE bits only (not the whole packed key), so the
		// slot tiebreak cannot flip this test relative to the float compare.
		if s.nearest.len() >= ef && keyDist(c) > keyDist(s.nearest.peek()) {
			break
		}
		nd := h.nodeAt(keySlot(c))
		// nd.unlinked: a node inside its placement/link window has no edges at any
		// level. Expanding it is a DEAD END that collapses the frontier — harmless
		// for a query, permanent for a linker, whose starved candidate set gets
		// written into the graph as a one-to-three-edge node. See node.unlinked.
		if nd == nil || nd.unlinked.Load() || level > nd.level {
			continue
		}
		nbrs := h.neighborsAt(s, nd, level)
		if batch.ok() {
			h.expandBatched(s, &batch, nbrs, ef, admit)
			continue
		}
		for i := 0; i < len(nbrs); i++ {
			// Prefetch a few neighbors ahead so their (random-slot) vectors are
			// being fetched while the nearer candidates are scored. Gated on the
			// SAME visited test the lookahead slot will face when the loop
			// reaches it: between a third and a half of neighbors are already
			// visited and are never scored, so an ungated prefetch spent its
			// memory bandwidth pulling in lines nothing would read.
			if j := i + prefetchDistance; j < len(nbrs) {
				if ahead := nbrs[j]; !s.visited.seen(ahead) {
					pf.slot(ahead)
				}
			}
			m := nbrs[i]
			if s.visited.seen(m) {
				continue
			}
			s.visited.mark(m)
			key := packKey(m, score(m))
			if s.nearest.len() < ef || keyDist(key) < keyDist(s.nearest.peek()) {
				s.candidates.push(key)
				if admit(m) {
					if s.nearest.len() < ef {
						s.nearest.push(key)
					} else {
						// The bounded set is already at (or over) ef and we know
						// key is strictly closer than the root, so push-then-pop
						// would evict exactly that root: replaceTop is the same
						// net heap in one descent instead of two.
						s.nearest.replaceTop(key)
					}
				}
			}
		}
	}

	// Write results into the scratch's reused out buffer (ascending by dist).
	// The heap's backing array already holds every kept key, so instead of ef
	// heapsort pops (n log n with the heap's poor locality) we sort the packed
	// keys directly: their unsigned order IS ascending (dist, slot). Callers
	// consume only the first k, but a full sort of a ~ef-element uint64 slice is
	// still far cheaper than draining the heap.
	keys := []uint64(s.nearest)
	slices.Sort(keys)
	n := len(keys)
	if cap(s.out) < n {
		s.out = make([]slotDist, n)
	} else {
		s.out = s.out[:n]
	}
	for i, key := range keys {
		s.out[i] = keyUnpack(key)
	}
	return s.out
}

// The batched path takes a node's WHOLE unvisited neighbor list in one kernel
// call. That is only correct because the kernels prefetch internally (see
// nyFunc): an earlier revision batched the whole list but issued the prefetches
// from here, in one burst before the call, and measured 3-9% SLOWER than the
// per-pair path on SIFT-1M — a 32-neighbor list at dim 128 is 128 cache lines,
// far past the outstanding misses the core can track, thrown at the memory
// system with no work to overlap them with. Chunking the call to restore a
// bounded window fixed SIFT-1M but gave the amortization back, landing neutral
// on cache-resident corpora. Moving the warming inside the kernel loop
// decouples the two: one call per neighbor list, bounded window, fully
// overlapped.

// batchedExpand enables the batched exact-distance paths: HNSW's neighbor
// expansion (expandBatched) and IVF-Flat's cell scan (ivf.gatherFlatBatched).
// A var so the A/B benchmark can force the per-pair path for comparison on the
// SAME index — the two paths differ only in throughput, so an A/B that rebuilt
// the index between arms would be comparing two different concurrent-build
// graphs (or two different k-means partitions) and measuring mostly build
// nondeterminism.
//
// ONE knob for both because they share one mechanism: both consult pickNy
// through a batchExact that applies the same decline rules to the same arena
// type, and both hand a block of slots to the same batchKernel.
var batchedExpand = true

// expandBatched is searchLayerCore's neighbor-expansion step for a scorer that
// has a batched kernel. It filters nbrs down to the not-yet-visited slots,
// scores all of them in ONE kernel call, then runs the heap admissions in the
// original neighbor order. The kernel warms its own candidates as it goes, so
// this does no prefetching of its own — see the note above and nyFunc.
//
// WHAT THIS DOES NOT CHANGE. The admission sequence is identical to the
// per-pair path's, not merely equivalent. It is worth being precise about this,
// because batching a scoring loop is usually expected to make the
// "closer than the worst kept" gate read a stale bound — every distance is
// computed before any admission runs, so one might expect a candidate to be
// gated against a bound its own block-mates should already have tightened. That
// is NOT what happens here: the gate below reads
// s.nearest.peek() afresh on every iteration, and the admissions that tighten it
// happen in the same loop in the same order as before. Only the DISTANCE
// COMPUTATION is hoisted, and the batched kernels are bit-identical to the
// per-pair ones (see distance_ny.go), so each gate sees exactly the value and
// exactly the heap it saw before.
//
// So this is a pure throughput change with no behavioral delta — the same
// candidates are admitted, in the same order, with the same distances, and the
// same graph is traversed. TestBatchedTraversalDoesNotLoseRecall and
// TestBatchedTraversalDistancesAreExact pin that against the per-pair path
// directly, and on SIFT-1M the recall figures match the baseline to four
// decimals at every ef.
func (h *hnsw) expandBatched(s *layerScratch, batch *batchKernel, nbrs []uint32, ef int, admit func(slot uint32) bool) {
	// Size both scratch buffers to the WHOLE neighbor list up front. Letting
	// append grow batchSlots instead walks the usual doubling ladder (4, 8, 16,
	// 32...), which is several allocations per scratch on the way to a steady
	// state that a pooled buffer reaches anyway — and it showed up on the
	// zero-allocation search benchmark. One reservation bounded by m0 = 2M
	// converges after the first expansion and never allocates again.
	//
	// Both halves are tested, not just batchSlots. They are always assigned
	// together, so today cap(batchSlots) >= len(nbrs) implies len(batchDist) >=
	// len(nbrs) — but that is an INVISIBLE COUPLING between two fields any
	// future edit could break, and the way it would break is a bounds panic on
	// the search hot path (the batchDist reslice below). One extra compare per
	// expanded node is not worth relying on the invariant instead.
	if cap(s.batchSlots) < len(nbrs) || len(s.batchDist) < len(nbrs) {
		s.batchSlots = make([]uint32, 0, len(nbrs))
		s.batchDist = make([]float32, len(nbrs))
	}
	cand := s.batchSlots[:0]
	for _, m := range nbrs {
		if s.visited.seen(m) {
			continue
		}
		s.visited.mark(m)
		cand = append(cand, m)
	}
	s.batchSlots = cand
	if len(cand) == 0 {
		return
	}
	dists := s.batchDist[:len(cand)]
	batch.score(cand, dists)

	for i, m := range cand {
		key := packKey(m, dists[i])
		if s.nearest.len() < ef || keyDist(key) < keyDist(s.nearest.peek()) {
			s.candidates.push(key)
			if admit(m) {
				if s.nearest.len() < ef {
					s.nearest.push(key)
				} else {
					// Same fused push-then-pop as the per-pair path: the gate
					// just above proved key is strictly closer than the root, so
					// a push would evict exactly that root.
					s.nearest.replaceTop(key)
				}
			}
		}
	}
}

// searchLayerWith is searchLayer with an OPTIONAL external metadata provider
// (the named-vector hook). metaOf == nil delegates to searchLayer (byte-identical
// to today, no extra allocation on the single-vector hot path). When metaOf is
// set, the per-candidate admission gate evaluates the predicate against the
// EXTERNAL payload via metaOf(arena.ID(slot)) instead of the arena. Both paths
// run the same searchLayerCore traversal; only the admission closure differs.
func (h *hnsw) searchLayerWith(s *layerScratch, sc layerScorer, entryPoints []uint32, ef int, level int, pred Predicate, metaOf func(id uint64) Metadata, now uint64) []slotDist {
	if metaOf == nil {
		return h.searchLayer(s, sc, entryPoints, ef, level, pred, now)
	}
	// One closure per traversal call (not per candidate); see searchLayerCore.
	return h.searchLayerCore(s, sc, entryPoints, ef, level, func(slot uint32) bool {
		return h.admitsWithScratch(s, slot, pred, metaOf, now)
	})
}

// linkLayer is searchLayer for the LINK path: the same traversal and the same
// liveness gate, plus the two STRUCTURAL conditions a candidate has to meet
// before it can be somebody's neighbour. Both are already in structuralLayer's
// gate — the empty-candidate fallback got them right because it was written
// against a graph under churn; the ordinary path never needed them, because a
// FRESH insert cannot violate either one. Slot reuse is what made it need them.
// (structuralLayer additionally excludes placed-but-unlinked nodes, which this
// deliberately does not: an unlinked node is inside its own link window and is
// about to have edges, so it is a legitimate neighbour, and searchLayer admitted
// it here before.)
//
// (1) NOT THE NODE ITSELF.
//
// A NODE MUST NEVER BE ITS OWN NEIGHBOUR, and on the upsert path it otherwise
// is. A fresh insert cannot reach itself (it has no in-edges yet), so
// searchLayer's plain gate was sufficient for as long as that was the only way
// a node got linked. An upsert lands on the slot the deleted point held and
// KEEPS that slot's level-0 edges (see placeLockedAt), so every surviving
// in-edge into the slot still points at it — the traversal that links it walks
// straight back into it. It is live (the arena insert already happened) and it
// sits at distance EXACTLY ZERO from its own query vector, so it is admitted as
// the global best candidate and picked first by the diversity heuristic, which
// keeps the nearest candidate unconditionally.
//
// The damage is not one wasted edge, it is three:
//
//   - The forward list spends a slot on the self-loop, so the node links to
//     m-1 real neighbours.
//   - The back-edge loop then hands the "neighbour" (itself) the in-edge, so
//     the node is granted one fewer REAL in-edge than the algorithm intends.
//   - The self-loop is IMMORTAL. Every prune (reselect) sorts candidates by
//     distance to the node, and a self edge is at distance 0, so it sorts first
//     and the heuristic always keeps it. It can never be evicted, while real
//     in-edges around it are.
//
// Measured on a full upsert sweep (dim 8, n=192, EfConstruction=200): EVERY
// upserted node ended with exactly two level-0 self-loops (one forward pick,
// one back-edge) at every M from 2 to 16 — 368-384 of them per sweep against 0
// on fresh insert. At M=16 (m0=32) two dead slots are absorbed; at M=4 (m0=8)
// they are a quarter of the node's level-0 budget, and the in-edges squeezed
// out by them are what left points unreachable. See
// TestSearchabilityInvariantUpsertSweepDegrees.
//
// (2) CARRIES THIS LEVEL.
//
// An edge at level lc to a node that does not reach lc is DEAD IN BOTH
// DIRECTIONS: searchLayerCore will not expand it there, and linkNode's back-edge
// loop skips it (`lc > other.level`), so the connection is one-way and the level
// budget it spends is pure loss. In a graph that was only ever built by fresh
// inserts this cannot arise — every level-lc edge is written to a node that was
// linked at lc — so admission never had to check. An upsert REDRAWS the level on
// a reused slot, which breaks the induction: the slot's in-edges at levels above
// the new level survive, the traversal reaches the node through them, and
// without this condition the linker admitted it and wrote FRESH one-way edges to
// it. That is self-sustaining, and it compounded: after one 1000-point sweep at
// M=4, 58.3% of all upper-level edges were dead and the total upper-edge count
// had collapsed from 1392 to 854. With this gate: 18.8% and 1246 — the residual
// being edges inherited from before the upsert, which are stale but were legal
// when written.
//
// Both exclusions are from ADMISSION only, exactly as in structuralLayer: such a
// node is still TRAVERSED wherever it does carry the level (its inherited edges
// are followed), which is the whole point of keeping them. It just cannot be the
// answer.
func (h *hnsw) linkLayer(s *layerScratch, sc layerScorer, entryPoints []uint32, ef int, level int, now uint64, self uint32) []slotDist {
	return h.searchLayerCore(s, sc, entryPoints, ef, level, func(slot uint32) bool {
		if slot == self {
			return false
		}
		if nd := h.nodeAt(slot); nd == nil || nd.level < level {
			return false
		}
		return h.admitsScratch(s, slot, nil, now)
	})
}

// structuralLayer repeats searchLayer's traversal at one level with the LIVENESS
// half of the admission gate removed: a candidate qualifies if it is a real,
// already-linked node carrying this level, whether or not it is tombstoned or
// TTL-expired.
//
// It exists for ONE caller and one situation — linkNode's candidate set coming
// back EMPTY (see the fallback there). searchLayer admits only LIVE points, so
// when every node reachable from the entry point at this level is tombstoned or
// expired it returns nothing, and the node being linked is written into the graph
// with no edges in either direction. That is a permanent orphan, and if the node
// also draws a taller level it becomes an EDGELESS ENTRY POINT and takes the
// whole index with it.
//
// A tombstoned slot is still a graph vertex — searchLayerCore already TRAVERSES
// it, and Reclaim is what finally removes it and prunes the edges into it — so
// connecting to one is the canonical lazy-deletion behaviour, not a leak. What is
// still excluded is anything that is not a usable neighbour: an empty slot, a
// node inside its own placement/link window (edgeless, so linking to it is the
// same dead end by another route), a node that does not carry this level (the
// back-edge loop would skip it, leaving the connection one-way), and the node
// being linked itself (a self-loop is not a connection).
func (h *hnsw) structuralLayer(s *layerScratch, sc layerScorer, entryPoints []uint32, ef int, level int, self uint32) []slotDist {
	return h.searchLayerCore(s, sc, entryPoints, ef, level, func(slot uint32) bool {
		if slot == self {
			return false
		}
		nd := h.nodeAt(slot)
		return nd != nil && nd.level >= level && !nd.unlinked.Load()
	})
}

// effAlpha is the effective RobustPrune α: pruneAlpha, treating 0 (the
// HNSW/IVF zero value) as 1.0. So the HNSW path multiplies the diversity test by
// exactly 1.0 — bit-identical to the historical `pf(...) < c.dist` heuristic — and
// only an explicit α>1 (Vamana) changes behavior.
func (h *hnsw) effAlpha() float32 {
	if h.pruneAlpha <= 1 {
		return 1.0
	}
	return h.pruneAlpha
}

// maxM0 is the level-0 reverse-edge / re-prune cap and the slab stride. For HNSW
// m0 = 2*M, so this is the historical 2*M (byte-identical). For Vamana m0 =
// VamanaR, so the bottom-layer out-degree is bounded by R (the Vamana out-degree
// cap), not 2*M. Always equals the slab stride, so addBackEdge/pruneNeighborsTo
// never overflow the level-0 region.
func (h *hnsw) maxM0() int { return h.m0 }

// selectNeighbors implements algorithm 4 (heuristic neighbor selection)
// from Malkov & Yashunin: for each candidate (in ascending distance order),
// keep it only if it's closer to the query than to any already-kept neighbor.
// This biases the result toward diverse directions, which improves recall on
// the search side at the cost of slightly higher build time.
//
// candidates must already be sorted ascending by distance (searchLayer's
// output is). m is the target neighbor count.
func (h *hnsw) selectNeighbors(query []float32, candidates []slotDist, m int) []slotDist {
	return h.selectNeighborsInto(nil, query, candidates, m)
}

// selectNeighborsInto is selectNeighbors writing its result into dst (which must
// be empty; its capacity is reused). The back-edge prune passes a per-goroutine
// buffer so the O(inserts * M) reselects allocate nothing; dst == nil is the
// plain selectNeighbors behaviour. The returned slice aliases dst OR candidates
// (the len(candidates) <= m short-circuit), so it is only valid until the
// caller's next use of either.
func (h *hnsw) selectNeighborsInto(dst []slotDist, query []float32, candidates []slotDist, m int) []slotDist {
	if len(candidates) <= m {
		return candidates
	}
	// α-RobustPrune: drop candidate v when α·dist(kept*, v) < dist(query, v). α=1
	// (HNSW; effAlpha returns 1.0 when pruneAlpha is 0/1) is the original heuristic,
	// BIT-IDENTICAL to the historical `pf(...) < c.dist` test. A larger α (Vamana's
	// pass-2 VamanaAlpha) shrinks the kept-neighbor exclusion radius, so more
	// long-range edges survive — the Vamana diversity rule.
	alpha := h.effAlpha()
	pf := h.buildPairFn()
	kept := dst[:0]
	if cap(kept) < m {
		kept = make([]slotDist, 0, m)
	}
	for _, c := range candidates {
		if len(kept) >= m {
			break
		}
		good := true
		for _, k := range kept {
			if alpha*pf(c.slot, k.slot) < c.dist {
				good = false
				break
			}
		}
		if good {
			kept = append(kept, c)
		}
	}
	// If the heuristic over-pruned (rare but possible at small m), top up
	// with the next-closest candidates to guarantee exactly m neighbors.
	if len(kept) < m {
		for _, c := range candidates {
			if len(kept) >= m {
				break
			}
			already := false
			for _, k := range kept {
				if k.slot == c.slot {
					already = true
					break
				}
			}
			if !already {
				kept = append(kept, c)
			}
		}
	}
	return kept
}

// selectNeighborsExtended is selectNeighbors with the Malkov-Yashunin
// extendCandidates option (Config.ExtendCandidates). Before the diversity pick,
// the candidate pool is enriched with the level-lc neighbors of each base
// candidate — scored against the query and deduped — so the heuristic chooses
// from second-hop nodes too. This produces more directionally-diverse edges and
// a higher recall ceiling on clustered data, at build-time cost only (query and
// graph layout are unchanged). Reads neighbor lists via neighborsAt, so it is
// safe in both the serial Insert path and the concurrent build (where the list
// is copied under the owner's link lock). Must hold the write lock (serial) or
// run inside BuildConcurrent (linkLocks active). lc is the level being linked.
func (h *hnsw) selectNeighborsExtended(s *layerScratch, query []float32, candidates []slotDist, m, lc int, self uint32) []slotDist {
	if len(candidates) <= m {
		return candidates
	}
	// Direct per-candidate scoring, no graph traversal: only the distance function
	// is needed here, not the paired prefetcher.
	score := h.buildScorer(s, query).score
	s.extVisited.prepare(h.arena.Capacity())
	pool := s.extPool[:0]
	for _, c := range candidates {
		if !s.extVisited.seen(c.slot) {
			s.extVisited.mark(c.slot)
			pool = append(pool, c)
		}
	}
	// Second hop: pull each base candidate's neighbors at this level into the
	// pool. Candidates are ascending by distance, so the closest second-hops are
	// added first; an optional cap stops once the pool is large enough, bounding
	// the extra distance work on large collections. neighborsAt returns a slice
	// that may alias s.nbrScratch (concurrent build) or the live slab (serial) —
	// consume it before the next call.
	maxPool := h.cfg.ExtendCandidatesMax
extend:
	for _, c := range candidates {
		nd := h.nodeAt(c.slot)
		// Same dead-end exclusion as searchLayerCore (see node.unlinked).
		if nd == nil || nd.unlinked.Load() || lc > nd.level {
			continue
		}
		nbrs := h.neighborsAt(s, nd, lc)
		for _, nb := range nbrs {
			if maxPool > 0 && len(pool) >= maxPool {
				break extend
			}
			// The node being linked is not a candidate for its own neighbour list.
			// linkLayer keeps it out of `candidates`; a second hop can walk right
			// back into it through a surviving in-edge, where it would sort first
			// (distance 0) and be kept unconditionally. See linkLayer.
			if nb == self || s.extVisited.seen(nb) {
				continue
			}
			s.extVisited.mark(nb)
			pool = append(pool, slotDist{slot: nb, dist: score(nb)})
		}
	}
	s.extPool = pool // keep the (possibly grown) backing array for reuse
	sort.Slice(pool, func(i, j int) bool { return pool[i].dist < pool[j].dist })
	return h.selectNeighbors(query, pool, m)
}

// forwardM is the number of forward neighbors to select for a new node at level
// lc: 2*M (the level-0 cap) when Level0FullDegree is set and lc==0, else M. The
// extra bottom-layer forward links raise recall@k for large k. Reverse edges are
// always capped at 2*M (level 0) / M (above) regardless — see addBackEdge.
func (h *hnsw) forwardM(lc int) int {
	// Vamana: select up to R = m0 forward neighbors at the single layer (the full
	// out-degree budget; RobustPrune at α then thins it). HNSW is unaffected
	// (vamana == false).
	if lc == 0 && h.vamana {
		return h.m0
	}
	if lc == 0 && h.cfg.Level0FullDegree {
		return 2 * h.cfg.M
	}
	return h.cfg.M
}

// pickNeighbors selects up to m neighbors for query from candidates at level lc,
// dispatching to the extendCandidates heuristic when Config.ExtendCandidates is
// set, else the plain heuristic. Used by both build paths (serial Insert and
// BuildConcurrent) so they construct identical graphs.
//
// `self` is the slot being linked. linkLayer already keeps it out of
// `candidates`; it is threaded on for the extendCandidates path, whose SECOND
// HOP walks neighbour lists that can legitimately contain an in-edge back to it
// (see linkLayer for why a self-loop is not merely wasteful but permanent).
func (h *hnsw) pickNeighbors(s *layerScratch, query []float32, candidates []slotDist, m, lc int, self uint32) []slotDist {
	if h.cfg.ExtendCandidates {
		return h.selectNeighborsExtended(s, query, candidates, m, lc, self)
	}
	return h.selectNeighbors(query, candidates, m)
}

// Insert adds vec under id with optional TTL, metadata, and sparse vector.
// ttl=0 means no expiry; meta=nil means no metadata; sparse=nil means no
// sparse lane. Expired entries are filtered out of Search results and reaped
// by the background sweeper into the existing tombstone path. All optional
// data is stored on the same slot as the vector and survives until Delete
// (or slot reuse, which clears it).
func (h *hnsw) Insert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (uint64, map[string]uint64, error) {
	// stamped=false: snapshot the wall clock once under the lock (see insertBody).
	return h.insertBody(id, vec, ttl, meta, sparse, keyTTLMs, cas, false, 0)
}

// InsertAt is Insert with every TTL deadline computation AND liveness check judged
// against the EXPLICIT leader-stamped clock nowMs (unix millis) rather than the
// wall clock — the replicated-apply variant so each replica stamps byte-identical
// absolute point/per-key deadlines and agrees on CAS/reclaim liveness (#4 vector
// TTL determinism, mirroring cache.PutAt). Additive: Insert is byte-identical to
// before.
func (h *hnsw) InsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, map[string]uint64, error) {
	return h.insertBody(id, vec, ttl, meta, sparse, keyTTLMs, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

// insertBody is the shared implementation of Insert/InsertAt. When stamped it uses
// nowMs for EVERY clock-dependent decision in this op (CAS liveness, per-key and
// point deadline stamping, dead-slot reclaim, graph-traversal admission); when
// unstamped it snapshots the wall clock ONCE under the write lock and uses that for
// all of them — a single per-op clock read, consistent with the search path's
// one-snapshot rule and byte-identical to the historical multi-read path under a
// stable clock.
func (h *hnsw) insertBody(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (uint64, map[string]uint64, error) {
	if err := checkRecordValues(meta); err != nil {
		return 0, nil, err
	}
	// A failed mmap slab growth freed the backing region; reject rather than
	// write into it (see arena.poisoned / ErrIndexPoisoned).
	if h.arena.poisoned.Load() {
		return 0, nil, ErrIndexPoisoned
	}
	start := time.Now()
	defer func() { h.insertLat.observe(time.Since(start)) }()

	if len(vec) != h.cfg.Dim {
		return 0, nil, ErrDimMismatch
	}

	// Quota: rate limit. Takes a token from the bucket (nil = unlimited).
	// Checked before the write lock so a throttled caller doesn't block readers.
	if !h.bucket.Take() {
		h.quotaRejects.Add(1)
		return 0, nil, ErrCollectionRateLimited
	}

	// Cosine pre-normalization. The hot read path then uses dot product only.
	stored := vec
	if h.cfg.Metric == Cosine {
		stored = append([]float32(nil), vec...)
		normalize(stored)
	}

	// Placement under the write lock, linking under the read lock (see runLink).
	// The placement section is SCOPED TO A CLOSURE so the write lock is released
	// by a defer — the same shape, and for the same reason, as Collection's opMu
	// sections. The lock still drops before the link phase (that ordering is the
	// change queries feel; the closure changes HOW it drops, not when), but a
	// panic inside placement now unwinds through the Unlock instead of leaving
	// the write lock held forever. A bare explicit unlock made any recover()
	// upstream wedge the whole collection: every later insert, delete, snapshot
	// and search would block on a lock with no owner left to release it.
	//
	// The serializer barrier spans BOTH halves and is taken FIRST — see
	// link_stripes.go. Held from before placement until after linking, it is what
	// stops a Snapshot from serializing this point in the lock-free gap between
	// them, where it exists in the arena with an empty edge list and would
	// restore as an unreachable orphan.
	var task linkTask
	var keyExpires map[string]uint64
	h.linkMu.RLock()
	defer h.linkMu.RUnlock()
	version, err := func() (uint64, error) {
		h.mu.Lock()
		defer h.mu.Unlock()
		// One clock for the whole op: the leader stamp under replication (deterministic
		// across replicas), else a single wall-clock snapshot taken under the lock.
		now := nowMs
		if !stamped {
			now = uint64(h.now())
		}
		// CAS: read the current version (0 if absent) and compare under the lock,
		// BEFORE the upsert-over-dead-slot reclaim mutates anything — a mismatch must
		// leave the index untouched. The version then bumps inside placeLockedAt.
		if err := cas.check(h.currentVersionLockedAt(id, now)); err != nil {
			return 0, err
		}
		// Per-key payload TTL: compute the ABSOLUTE deadline map now+ttl against the
		// fresh point's payload (relative→absolute), mirroring set_payload. Empty/nil
		// stays the zero-overhead path. The resulting map is returned so the Collection
		// layer WAL-logs it verbatim (time-stable replay).
		keyExpires = computeInsertKeyExpires(int64(now), meta, keyTTLMs) //nolint:gosec // unix-millis fits int64
		return h.placeLockedAt(id, stored, ttl, meta, sparse, keyExpires, 0, now, &task)
	}()
	if err != nil {
		return 0, nil, err
	}
	if linkGapHook != nil {
		linkGapHook()
	}
	h.runLink(&task)
	return version, keyExpires, nil
}

// linkGapHook, when non-nil, runs at the exact point between an insert's
// placement section and its link phase — the interval in which the insert holds
// no lock at all but does hold the serializer barrier, and in which the new
// point is visible to a graph serializer with no edges.
//
// It is a TEST SEAM, nil in production, where it costs one perfectly-predicted
// branch on a path that takes hundreds of microseconds. It exists because the
// alternative is a test that re-implements this function's lock sequence by
// hand, and such a test proves nothing about the invariant that matters: it
// takes the barrier ITSELF, so it keeps passing after the barrier is deleted
// from here. Driving the real body is the only way that assertion is about this
// code. See TestSnapshotNeverSerializesAnUnlinkedPoint.
var linkGapHook func()

// RestoreInsert inserts id with an EXACT version (setVersion), set verbatim
// rather than bumped — the version-preserving primitive for WAL replay and
// reshard copy. No CAS, no rate limit (replay/copy is internal and idempotent).
func (h *hnsw) RestoreInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error {
	return h.restoreInsertBody(id, vec, ttl, meta, sparse, keyExpires, version, false, 0)
}

// RestoreInsertAt is RestoreInsert judging reclaim liveness and stamping the POINT
// ttl deadline against the EXPLICIT leader-stamped clock nowMs (the per-key
// keyExpires map is still installed VERBATIM). Used by the replicated
// version-preserving insert path (reshard/resplit backfill) so a stamped copy
// stamps a deterministic point deadline on every replica (#4 vector TTL
// determinism).
func (h *hnsw) RestoreInsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) error {
	return h.restoreInsertBody(id, vec, ttl, meta, sparse, keyExpires, version, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) restoreInsertBody(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, stamped bool, nowMs uint64) error {
	// REPLAY body: size only. Its callers are Collection.RestoreInsert[At] —
	// which already ran the full checkRecordValues on the caller's metadata —
	// and WAL replay, which must never refuse an acked record. See
	// checkRecordValuesSize.
	if err := checkRecordValuesSize(meta); err != nil {
		return err
	}
	if len(vec) != h.cfg.Dim {
		return ErrDimMismatch
	}
	stored := vec
	if h.cfg.Metric == Cosine {
		stored = append([]float32(nil), vec...)
		normalize(stored)
	}
	var task linkTask
	h.linkMu.RLock() // serializer barrier, spanning placement + link (see insertBody)
	defer h.linkMu.RUnlock()
	// Placement scoped to a closure so the write lock is released by a defer even
	// on a panic — see insertBody.
	err := func() error {
		h.mu.Lock()
		defer h.mu.Unlock()
		now := nowMs
		if !stamped {
			now = uint64(h.now())
		}
		// keyExpires is the ABSOLUTE per-key deadline map logged by the WAL — restored
		// VERBATIM (NOT recomputed now+ttl) so pending key deadlines survive a crash /
		// reshard copy time-stable, exactly like RestorePayload.
		_, err := h.placeLockedAt(id, stored, ttl, meta, sparse, keyExpires, version, now, &task)
		return err
	}()
	if err != nil {
		return err
	}
	h.runLink(&task)
	return nil
}

// currentVersionLockedAt is currentVersionLocked judging liveness against the
// caller-supplied `now` (unix millis) instead of a fresh wall-clock read, so a
// replicated apply evaluates the CAS liveness/version against the leader-stamped
// clock identically on every replica (#4 vector TTL determinism). Must hold h.mu.
func (h *hnsw) currentVersionLockedAt(id uint64, now uint64) uint64 {
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		return 0
	}
	return h.arena.Version(slot)
}

// placeLockedAt is the write-lock-held PLACEMENT half of Insert: the
// upsert-over-tombstone reclaim, the vector-count and byte-budget quota checks,
// the arena insert and all its per-slot bookkeeping (version, TTL, metadata,
// payload index, sparse, BM25), the level draw, and the node's entry in the
// graph. It does NOT link the node into the graph — it hands the caller a
// linkTask to run via runLink once the write lock is released. That split is
// the whole point of Option B: placement is O(1), linking is not, and only
// placement needs exclusivity.
//
// Caller preconditions: MUST hold h.mu (write); `stored` already
// cosine-normalized if the metric requires it; the rate-limit token already
// taken (h.bucket has its own mutex, so take it before acquiring h.mu). The
// count/byte quota is enforced HERE, not by the caller. Shared by Insert and
// InsertIfAbsent so the if-absent path runs the liveness check and the placement
// in ONE critical section (no re-lock via the public Insert).
//
// It places a fresh point and returns its resulting version. A new point's
// version is 1, UNLESS setVersion != 0, in which case the version is set
// VERBATIM (the WAL-replay / reshard-copy path, which must preserve the exact
// version rather than reset it to 1). HNSW has no in-place vector update — a live
// id collides (ErrDuplicateID) and Collection.Upsert is delete+insert — so every
// successful insert here is a fresh point (version 1 / setVersion).
//
// EVERY clock-dependent decision is judged against the caller-supplied `now`
// (unix millis): the dead-slot reclaim liveness check, the point-TTL absolute
// deadline (now+ttl), and — carried on the linkTask — the graph-traversal
// admission snapshot. Threading one `now` makes a replicated insert stamp
// byte-identical state, including graph edges, since expiry-gated admission
// during the build walk is judged on the same clock on every replica (#4 vector
// TTL determinism).
//
// `out` receives the link task. It is left untouched on every path that places
// nothing to link (an error, or the first node in an empty index), so a
// zero-valued linkTask is a correct "nothing to do".
func (h *hnsw) placeLockedAt(id uint64, stored []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, setVersion uint64, now uint64, out *linkTask) (uint64, error) {
	// PQDropVecs read-only: once the floats are dropped the index has no resident
	// vectors to navigate/link a new node on (the build/link path scores on exact
	// floats via buildScorer/selectNeighbors → arena.Vec). Rather than degrade the
	// graph with code-distance links, the post-drop index is read-mostly: reject
	// the insert loud so the caller rebuilds to add points. No float read, no
	// nil-deref. Only ever true under a trained QuantPQ index built with
	// PQDropVecs; every other index is unaffected (byte-identical).
	if h.vecsDropped() {
		return 0, ErrPQDropVecsReadOnly
	}
	// Upsert replace: if id already maps to a DEAD slot — tombstoned (a prior
	// Delete, as Collection.Upsert issues) or TTL-expired-but-not-yet-swept —
	// hard-remove that slot and free it for reuse so this insert resurrects the
	// id with the new vector. Un-tombstoning it keeps the live accounting balanced
	// (Size = arena.Size() - tombstoned). Reclaiming the expired case too lets
	// InsertIfAbsent treat an expired slot as absent without colliding (Exists
	// already reports it not-live). A LIVE id still collides — arena.Insert
	// returns ErrDuplicateID below.
	//
	// FREE AND REUSE MUST BE ATOMIC, which is why the decision is taken here but
	// the reclaim itself runs LOWER DOWN, past every check that can reject. The
	// reclaim is destructive in exactly the directions the admission gate reads:
	// it un-tombstones the slot, lets arena.Delete clear its expiry, and leaves
	// its stale id, vector and metadata in place while deliberately NOT pruning
	// the in-edges pointing at it. The ONLY thing that makes that safe is the
	// arena.Insert immediately after, which takes the slot straight back. An
	// early return in between would drop the write lock with a slot that is
	// un-tombstoned, un-expired and still reachable through those in-edges —
	// admits() would pass it and a search would emit the DELETED point, with its
	// old vector and old payload, as a live result. Both quota checks used to sit
	// in that gap, and a collection whose arena is over quota (a bulk build does
	// not consult the quota — see BuildConcurrentMeta) reaches them: every upsert
	// then resurrected the id it was trying to replace.
	//
	// Deciding the quota early does NOT tighten it. Upserting into a collection
	// that is exactly full is a legitimate replace and must still succeed, so each
	// check below is judged against the accounting the reclaim is ABOUT to produce
	// — one fewer vector, one insert's worth of bytes released — which is the same
	// arithmetic those checks saw when they ran after it.
	reclaimSlot, reclaiming := uint32(0), false
	if old, ok := h.arena.Slot(id); ok && (h.tombstoned[old] || h.isExpiredAt(old, now)) {
		reclaimSlot, reclaiming = old, true
	}

	// Quota: vector count.
	liveAfterReclaim := int64(h.arena.Size())
	if reclaiming {
		liveAfterReclaim-- // the reclaim releases this id's slot before the insert retakes it
	}
	if h.cfg.MaxVectors > 0 && liveAfterReclaim >= h.cfg.MaxVectors {
		h.quotaRejects.Add(1)
		return 0, ErrCollectionFull
	}

	// Quota: byte budget. Conservative estimate via estimateInsertBytes, PLUS the
	// numeric column sidecar (payload_column.go), which is the one allocation in
	// the index that bytesUsed does not track — it is made on the query path, so
	// it has no insert to be charged to. Counting it here is what keeps MaxBytes a
	// bound on the collection rather than a bound on its insert path.
	// (IVF shares payloadIndex but never columnises, so its counter is always zero
	// and its quota check is left alone.)
	//
	// COLUMNS YIELD TO WRITES. If the budget is only exceeded because columns are
	// holding memory, drop them and re-check: the sidecar is a pure read-side
	// CACHE — every byte in it is reconstructible from the payload index by the
	// next query that wants it — while an insert is durable data a caller cannot
	// reconstruct at all. Letting a cache permanently displace writes is the wrong
	// trade in any collection, and a badly wrong one at low dimensions, where a
	// full set of columns is twice the vector data (see maxNumColumns): a single
	// read-path build could otherwise push a collection into refusing EVERY
	// subsequent insert, forever, with nothing in the write path able to release
	// the memory.
	//
	// So ErrCollectionFull below now means what it is supposed to mean — the
	// collection's own durable data has filled its budget — and never "a query
	// cached something". Dropping is safe here for the same reason eviction is
	// safe in ensureColumn, and more so: insertLocked holds the WRITE lock, so no
	// reader is mid-traversal and no gate holds a column.
	insertBytes := estimateInsertBytes(h.cfg.Dim, h.cfg.M)
	bytesAfterReclaim := h.bytesUsed
	if reclaiming {
		bytesAfterReclaim -= insertBytes // released by the reclaim, exactly as it used to be
		if bytesAfterReclaim < 0 {
			bytesAfterReclaim = 0
		}
	}
	if h.cfg.MaxBytes > 0 && bytesAfterReclaim+h.payloadIdx.columnBytes()+insertBytes > h.cfg.MaxBytes {
		if h.payloadIdx.columnBytes() > 0 {
			h.payloadIdx.dropColumns()
			h.columnDrops.Add(1)
		}
		if bytesAfterReclaim+h.payloadIdx.columnBytes()+insertBytes > h.cfg.MaxBytes {
			h.quotaRejects.Add(1)
			return 0, ErrCollectionFull
		}
	}

	// PAST EVERY REJECTION. Free the dead slot now, and let the arena.Insert
	// directly below take it back: from here to that call there is no return, no
	// unlock and no allocation that can fail. arena.Insert itself cannot fail on
	// this path either — the dimension was validated by the caller, the id was
	// just removed from idMap, and a free-list slot never grows the arena.
	if reclaiming {
		old := reclaimSlot
		delete(h.tombstoned, old)
		if int(old) < len(h.nodes) {
			h.nodes[old] = nil // drop the old graph node; dangling in-edges are tolerated
		}
		h.payloadIdx.reindex(old, nil) // drop its payload keys
		if h.sparseIdx != nil {
			if sv := h.arena.Sparse(old); sv != nil {
				h.sparseIdx.remove(old, *sv) // drop its sparse postings
			}
		}
		if h.bm25 != nil {
			// Drop the old slot's BM25 postings + stats. Re-analyze the SAME stored
			// $content so the removal is the exact inverse of the prior add (the
			// analyzer is deterministic), keeping df/n/avgdl balanced.
			if content := contentOf(h.arena.Metadata(old)); content != "" {
				h.bm25.remove(old, h.analyzeCounts(content))
			}
		}
		h.bytesUsed = bytesAfterReclaim // the decrement the byte quota was judged on
		if h.entryPoint == old {
			h.electEntryPoint()
		}
		// arena.Delete clears the old slot's deadlines (point + per-key) through
		// the setters, so arena.deadlinePoints/deadlineKeys — the TTL sweep's
		// fast-path gate — drop this slot: it is wholesale removed, not swept.
		h.arena.Delete(id) // frees idMap[id] and returns `old` to the free list for reuse
	}

	slot, err := h.arena.Insert(id, stored)
	if err != nil {
		return 0, err
	}
	// Per-point version: a fresh insert is version 1 (the arena cleared the slot's
	// version to 0 on reuse / append), UNLESS setVersion is given (WAL replay /
	// reshard copy → restore the exact version verbatim, NOT 1).
	version := uint64(1)
	if setVersion != 0 {
		version = setVersion
	}
	h.arena.SetVersion(slot, version)
	// id-set membership grew (a brand-new id, or a resurrected dead slot whose id
	// the reclaim branch above removed from idMap before this Insert). A LIVE-id
	// collision returned ErrDuplicateID above and never reaches here, so this bump
	// fires ONLY when the live id set actually changes. Invalidates the scroll
	// snapshot; this single increment under h.mu is the only write-path cost.
	h.idSetVersion++
	h.bumpData() // id-set change also invalidates the order_by snapshot
	if ttl > 0 {
		deadline := now + uint64(ttl.Milliseconds())
		h.arena.SetExpires(slot, deadline)
	}
	if meta != nil {
		h.arena.SetMetadata(slot, meta)
	}
	// Per-key payload TTL: store the ABSOLUTE deadline map (key -> now+ttl, already
	// computed by the caller — relative→absolute for a live Insert, verbatim for a
	// WAL/reshard restore) pruned to the present payload keys. Empty/nil leaves the
	// slot's keyExpires cleared (arena.Insert already nilled it), so the no-key_ttl
	// path stays byte-identical / zero-overhead.
	if len(keyExpires) > 0 {
		h.arena.SetKeyExpires(slot, pruneKeyExpires(cloneKeyExpires(keyExpires), meta))
	}
	// Maintain the payload index. Called unconditionally so a reused slot's
	// stale entries are cleared even when the new insert has no metadata.
	h.payloadIdx.reindex(slot, meta)
	if sparse != nil && !sparse.IsZero() {
		h.arena.SetSparse(slot, sparse)
		h.sparseIdx.add(slot, *sparse)
	}
	if h.bm25 != nil {
		// Index this record's $content for BM25. The reserved field rides `meta`
		// (Collection.Upsert wraps it via withContent). Empty/missing content adds
		// no postings, so a contentless record never enters the BM25 corpus.
		if content := contentOf(meta); content != "" {
			h.bm25.add(slot, h.analyzeCounts(content))
		}
	}
	h.bytesUsed += insertBytes
	level := h.assignLevel()
	nd := &node{
		slot:  slot,
		level: level,
		upper: makeUpper(level),
	}
	if err := h.setNode(slot, nd); err != nil {
		return 0, err
	}
	// A REUSED slot still carries the previous occupant's level-0 edges, and it
	// KEEPS them until this node's own forward write replaces them. That is not a
	// leak: it keeps every other node's in-edges into this slot traversable for
	// the whole window, so the search this very insert is about to run — and any
	// concurrent query — can still step THROUGH it instead of dead-ending.
	//
	// How GOOD those inherited edges are depends on which reuse happened. The case
	// that motivates this is the upsert-over-dead-slot reclaim just above, where
	// the predecessor held the SAME id at this slot, so its neighbours are a
	// reasonable starting neighbourhood for the replacement. A slot off the
	// general free list (Reclaim) held a different id entirely, and even an
	// upsert's replacement vector may be nowhere near the old one. So the claim
	// here is only that the edges are traversable and get replaced promptly — NOT
	// that they are good neighbours. Being a waypoint is what matters; being the
	// right waypoint is what the forward write fixes moments later.
	//
	// Zeroing them here instead cost 63 partitioned-graph events and 7 failures
	// per 25 runs of the cluster reshard test, against 1 and 0 with this line
	// removed — measured by single-line ablation at the commit that introduced
	// it. Under churn the whole upserted band is in this window at one time or
	// another, and an id-ordered sweep of dead ends across a line-shaped corpus
	// cuts the graph in half.
	//
	// What the zeroing was FOR was linkNode's raced-edge merge, which would
	// otherwise "preserve" the predecessor's edges into the new node's list.
	// staleLevel0 says so directly instead — though the merge now ACCEPTS them
	// (subject to the heuristic) rather than declining them, because discarding a
	// slot's whole level-0 list is how an upsert sweep tore in-edges out of the
	// graph at low degree. See the merge in linkNode.
	nd.staleLevel0 = h.level0Len[slot] != 0
	h.insertOps.Add(1)

	// First node anchors the index. Nothing to link, and — matching the historical
	// path exactly — no auto-train evaluation either.
	//
	// The anchor is the ONE placement with no forward write to follow, so nothing
	// would ever replace inherited edges or clear the flag. Reclaim frees slots
	// without zeroing level0Len, and emptying an index re-elects maxLevel to -1,
	// so the very next insert lands here — on a free-list slot still carrying an
	// unrelated point's edge list, which it would then hold PERMANENTLY as the
	// entry point, and which Snapshot would serialize into a durable artifact.
	// The waypoint argument above does not apply: a node that never links has no
	// window to be a waypoint through.
	if h.maxLevel < 0 {
		h.level0Len[slot] = 0
		nd.staleLevel0 = false
		h.entryPoint = slot
		h.maxLevel = level
		return version, nil
	}

	// DETERMINISTIC AUTO-TRAIN. The three should*() predicates are pure functions
	// of APPLIED state — the live count and whether the codec has trained —
	// evaluated here, in the serial placement section, so "which insert trips the
	// threshold" is decided inside the one critical section that is strictly
	// ordered across replicas.
	t := linkTask{nd: nd, stored: stored, level: level, now: now}

	// Allocate the link stripes BEFORE either branch. Both link, and linkNode
	// locks a stripe unconditionally — for its own forward write and for every
	// back-edge — so the array has to exist on both paths, not just the one that
	// also raises the reader gate. Having this inside the ordinary branch alone
	// made correctness depend on arithmetic nobody was looking at: the training
	// branch fires at insert max(2, threshold) and the first insert returns
	// early, so a threshold of 1 or 2 reached linkNode with an empty slice and
	// panicked — while holding this write lock, on a path whose unlock a
	// recover() upstream would then never reach. It is idempotent and reduces to
	// one nil check once allocated.
	h.ensureLinkStripes()

	if h.shouldAutoTrainPQ() || h.shouldAutoTrainSQ() || h.shouldAutoTrainPRQ() {
		// TRAINING KEEPS THE OLD, FULLY EXCLUSIVE SHAPE — decide, link, and train
		// without ever dropping the write lock.
		//
		// Splitting the decision from the training is not safe here, and adding a
		// re-check at training time would not make it safe. The training SAMPLE is
		// liveSampleLockedPQAt, which excludes tombstoned slots; sweepOnce
		// tombstones on a WALL-CLOCK ticker and takes no opMu, so it can land
		// between a released and a re-acquired write lock and change the sample out
		// from under a decision already made. (It cannot flip the decision itself —
		// liveCount is len(idMap) and a sweep only tombstones — so a re-check would
		// have guarded the one thing that was never at risk while leaving the
		// sample, the thing replicas compare byte-for-byte, exposed.) Holding the
		// lock across all three steps is what actually removes the interleaving.
		//
		// The cost is one blocking insert per index lifetime: training already
		// runs k-means over thousands of vectors under this same lock, so the link
		// phase folded in beside it is noise. Every OTHER insert — all but the one
		// that trips the threshold, forever — takes the relocked path below.
		//
		// No reader GATE is raised for this link: h.linkers exists to tell readers
		// that a linker may be mutating neighbor lists, and no reader can be running
		// while this write lock is held, so leaving it at zero keeps the neighbor
		// READS here on their unsynchronized path. The neighbor WRITES still take
		// their stripe — linkNode always does, uncontended in this case — which is
		// why the stripe array must already exist above.
		s := getLayerScratch()
		h.linkNode(s, nd, stored, level, now)
		layerScratchPool.Put(s)
		// Order matters and is unchanged: the node is fully linked on EXACT floats
		// BEFORE training, which may re-encode every slot and, under PQDropVecs,
		// release the very floats the link phase reads. NEVER spawn a goroutine
		// here — the trained state must be committed before the next applied entry.
		// Trains ONCE: afterwards the *Untrained() predicates are false.
		if h.shouldAutoTrainPQ() {
			if err := h.autoTrainLockedPQ(now); err != nil {
				return 0, err
			}
		}
		if h.shouldAutoTrainSQ() {
			if err := h.autoTrainLockedSQ(now); err != nil {
				return 0, err
			}
		}
		if h.shouldAutoTrainPRQ() {
			if err := h.autoTrainLockedPRQ(now); err != nil {
				return 0, err
			}
		}
		// `out` stays zero: nothing is deferred, so runLink is a no-op.
		return version, nil
	}

	// Publish the link phase: the gate must be raised under THIS write lock,
	// before it is dropped, or a reader could acquire the read lock and take the
	// unsynchronized neighbor path against a live linker. (The stripes it steers
	// readers onto were allocated above, which is the ordering that matters:
	// gate visible ⇒ stripes exist.)
	h.linkers.Add(1)
	// Hide the node from traversal and from the entry-point elections until its
	// edges exist — but ONLY when it genuinely has none. A node on a reused slot
	// inherits the predecessor's edges and is a perfectly good waypoint; hiding
	// it is precisely the dead end this whole window has to avoid. Set under this
	// write lock, which the elections also hold, so they can never observe it
	// half-set.
	nd.unlinked.Store(!nd.staleLevel0)
	t.linking = true
	*out = t
	return version, nil
}

// linkTask carries an insert from its placement section (under the write lock)
// to its link phase (under the read lock). It exists because the two halves of
// an Insert now run under DIFFERENT locks: everything the link phase needs must
// be captured while the write lock is still held, since re-deriving it later
// would read state a concurrent writer may since have moved on.
//
// `linking` distinguishes "there is a link phase to run" from the zero value, so
// the paths that place without linking (the first node in an index, and every
// error return) hand back a task that runLink correctly does nothing with.
type linkTask struct {
	nd     *node
	stored []float32
	level  int
	now    uint64

	linking bool
}

// runLink executes the deferred half of an insert: the graph link phase, under
// the READ lock, concurrent with queries.
//
// The caller must hold h.mu NOT AT ALL and the serializer barrier (h.linkMu read
// side) THROUGHOUT — it took the barrier before placement and holds it until
// after this returns. The linker count placement raised is dropped by linkRead
// even if the traversal panics; leaving it raised would silently put every
// future query on the striped read path for the life of the index.
//
// There is no auto-training here. An insert that trips a training threshold
// never reaches this function: placement links and trains it inline, under the
// write lock it already holds, precisely so no wall-clock mutator can land
// between the decision and the sample it trains on. See placeLockedAt.
func (h *hnsw) runLink(t *linkTask) {
	if t.linking {
		h.linkRead(t)
	}
}

// linkRead runs one link phase under the read lock. Split out from runLink so
// the linker-count release and the lock release are unwound by defers on a
// scope that ends exactly where the read lock does — a panic inside the
// traversal (a user predicate is not reachable from here, but a corrupted graph
// would be) must not leave the gate permanently raised, which would silently
// cost every future query the striped read path.
//
// The CALLER holds the serializer barrier (h.linkMu read side) across both this
// and the placement that preceded it. It deliberately is NOT taken here: doing
// so would acquire it while already holding h.mu's read lock, inverting the
// global linkMu → h.mu order and closing a three-way deadlock — a Snapshot
// holding h.mu.R and waiting on linkMu.W, a TTL sweep or Reclaim queued on
// h.mu.W behind it, and this linker unable to get h.mu.R because Go's RWMutex
// stops admitting readers once a writer waits. See link_stripes.go.
func (h *hnsw) linkRead(t *linkTask) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	defer h.linkers.Add(-1)

	// The placed node must still BE the occupant of its slot. Between placement
	// dropping the write lock and this line, a writer could in principle have
	// tombstoned this id and then Reclaimed it, handing the slot to someone else;
	// linking then would write edges into another point's slab region. Production
	// cannot reach that (Collection.opMu serializes the delete behind this very
	// insert's return), but Reclaim is callable directly and the check is a single
	// slice index against a link phase costing hundreds of microseconds.
	if h.nodeAt(t.nd.slot) != t.nd {
		return
	}

	// Pooled scratch for this insert's searchLayer traversals. Checked out via
	// getLayerScratch so the bitset admission gate is disarmed: a linker runs
	// under the READ lock now, concurrently with queries, and could be handed the
	// very scratch a filtered search just returned to the pool. The link
	// traversal passes a nil predicate, so a stale gate could not actually change
	// its admissions — but that is an argument that depends on linkNode never
	// growing a predicate, and this costs two stores.
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	// Build navigates on exact float32 for graph quality by default; under
	// QuantizedBuild it navigates on the int8 codes (faster, slightly lower
	// quality — recovered by the query-time rescore). PQ-HNSW always navigates on
	// exact float32 (quantizedBuild() is false for PQ): an Insert before the
	// BuildConcurrent train hook has untrained codebooks (arena.Encode is a no-op,
	// so the code stays a zero placeholder), and once trained the float build still
	// gives the best graph; either way search ADC-navigates + exact-rescores.
	// linkNode owns that choice, along with the descent, the per-level neighbor
	// selection, the back-edges, and the entry-point promotion.
	// linkNode clears nd.unlinked itself, before its entry-point promotion — the
	// flag must not still be set on a node that has just become the entry point.
	// The identity check above returns before any of this, deliberately: a node
	// that lost its slot is unreachable garbage and must stay flagged.
	h.linkNode(s, t.nd, t.stored, t.level, t.now)
}

// Get retrieves the live record for id: a copy of its vector, metadata map, and
// sparse vector, plus the remaining TTL. ok is false when id is absent,
// tombstoned (deleted), or TTL-expired — the exact liveness gate the search path
// uses (mirror admits). The returned vec/sparse are owned by the caller, and so
// is the meta MAP — inserting or deleting keys in it never reaches the arena.
// The Values inside it are copied as structs, so a slice-backed one
// (strings/ints/floats/record) still points at the arena's own bytes: read it,
// never write through it. That is vtypes.Metadata's documented slice-ownership
// contract, identical for all four kinds; a caller that needs to own the bytes
// copies them, and every network client already gets a codec-written copy. For a cosine-metric index the stored
// (and therefore returned) vector is the NORMALIZED vector, not the original
// caller-supplied one — Insert normalizes on the way in. ttl is the remaining
// duration to the deadline (0 = no expiry).
func (h *hnsw) Get(id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	slot, present := h.arena.Slot(id)
	if !present || h.tombstoned[slot] || h.isExpired(slot) {
		return nil, nil, 0, nil, 0, false
	}
	version = h.arena.Version(slot)
	vec = append([]float32(nil), h.vecFor(slot)...) // COPY: vecFor aliases arena (exact) or allocates (reconstructed when PQDropVecs)
	if m := h.liveMeta(slot, uint64(h.now())); len(m) > 0 {
		out := make(Metadata, len(m))
		for k, v := range m {
			out[k] = v
		}
		meta = out
	}
	if exp := h.arena.ExpiresAt(slot); exp != 0 {
		if now := uint64(h.now()); exp > now { // not-yet-expired (isExpired already gated equal/past)
			ttl = time.Duration(exp-now) * time.Millisecond
		}
	}
	if sv := h.arena.Sparse(slot); sv != nil {
		sparse = sv.Clone() // clone: arena owns the pointer
	}
	return vec, meta, ttl, sparse, version, true
}

// GetProjected is Get with projection: withVec gates the dense-vector copy and
// withPayload gates the metadata + sparse clones, so a caller that will discard a
// projection never pays to copy it. This is the batch-read fast path
// (handleVectorGetBatch): a with_vector=false / with_payload=false get touches only
// the liveness gate + version + TTL and allocates NOTHING per point, versus Get
// which unconditionally deep-copies the vector, meta map, and sparse vector. All
// other semantics (liveness gate, the cosine-normalized-vector caveat, torn-read
// safety under RLock) match Get.
func (h *hnsw) GetProjected(id uint64, withVec, withPayload bool) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	slot, present := h.arena.Slot(id)
	if !present || h.tombstoned[slot] || h.isExpired(slot) {
		return nil, nil, 0, nil, 0, false
	}
	version = h.arena.Version(slot)
	if withVec {
		vec = append([]float32(nil), h.vecFor(slot)...) // COPY only when requested
	}
	if withPayload {
		if m := h.liveMeta(slot, uint64(h.now())); len(m) > 0 {
			out := make(Metadata, len(m))
			for k, v := range m {
				out[k] = v
			}
			meta = out
		}
	}
	if exp := h.arena.ExpiresAt(slot); exp != 0 {
		if now := uint64(h.now()); exp > now { // not-yet-expired (isExpired already gated equal/past)
			ttl = time.Duration(exp-now) * time.Millisecond
		}
	}
	if withPayload {
		if sv := h.arena.Sparse(slot); sv != nil {
			sparse = sv.Clone() // clone: arena owns the pointer
		}
	}
	return vec, meta, ttl, sparse, version, true
}

// GetInto is Get that appends the vector into the caller-owned scratch dst
// (passed as dst[:0]) instead of allocating a fresh []float32, so a hot-loop
// caller reusing one buffer pays zero allocations for the dense copy. The copy
// happens under h.mu.RLock exactly like Get, so it carries the same liveness gate
// and torn-read safety — only the destination of the vector copy changes. The
// returned vec aliases dst's backing when dst has the capacity; the cosine caveat
// (the returned vector is the NORMALIZED one) and the PQDropVecs caveat (vecFor
// reconstructs/allocates, which GetInto cannot elide) carry over from Get/vecFor.
func (h *hnsw) GetInto(dst []float32, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	slot, present := h.arena.Slot(id)
	if !present || h.tombstoned[slot] || h.isExpired(slot) {
		return nil, nil, 0, nil, 0, false
	}
	version = h.arena.Version(slot)
	vec = append(dst[:0], h.vecFor(slot)...) // COPY into dst: vecFor aliases arena (exact) or allocates (reconstructed when PQDropVecs)
	if m := h.liveMeta(slot, uint64(h.now())); len(m) > 0 {
		out := make(Metadata, len(m))
		for k, v := range m {
			out[k] = v
		}
		meta = out
	}
	if exp := h.arena.ExpiresAt(slot); exp != 0 {
		if now := uint64(h.now()); exp > now { // not-yet-expired (isExpired already gated equal/past)
			ttl = time.Duration(exp-now) * time.Millisecond
		}
	}
	if sv := h.arena.Sparse(slot); sv != nil {
		sparse = sv.Clone() // clone: arena owns the pointer
	}
	return vec, meta, ttl, sparse, version, true
}

// liveMeta returns the slot's metadata with any per-key-TTL-EXPIRED keys
// dropped, as the view every read path (Get / search results / scroll / the
// filter predicate) must see. When the slot has no per-key TTL (the common
// case — keyExpires[slot] is nil) it returns the arena's metadata view UNCHANGED
// (no clone, no allocation). Otherwise it clones the metadata and removes keys
// whose deadline is non-zero and <= now. A key is logically expired iff
// deadline != 0 && deadline <= now (mirrors point TTL's isExpired). Must be
// called with h.mu held (read or write).
//
// The returned map may ALIAS arena storage (the no-TTL fast path), so callers
// that mutate must clone first (cloneMeta) — the read paths here only read it.
func (h *hnsw) liveMeta(slot uint32, now uint64) Metadata {
	m := h.arena.Metadata(slot)
	ke := h.arena.KeyExpires(slot)
	if len(ke) == 0 {
		return m // fast path: no per-key TTL on this slot
	}
	// Determine whether any key is actually expired before allocating.
	var expiredAny bool
	for k, dl := range ke {
		if keyExpired(dl, now) {
			if _, present := m[k]; present {
				expiredAny = true
				break
			}
		}
	}
	if !expiredAny {
		return m // deadlines exist but none reached / none still present
	}
	out := make(Metadata, len(m))
	for k, v := range m {
		if keyExpired(ke[k], now) {
			continue // logically expired — drop
		}
		out[k] = v
	}
	return out
}

// keyExpired is the SINGLE SOURCE OF TRUTH for "is this per-key deadline
// expired": a deadline is expired iff it is non-zero and has been reached
// (mirrors point TTL's isExpired). Both the lazy read path (liveMeta) and the
// background sweep (sweepOnce) call this so they can never diverge — a key the
// sweep physically drops is exactly a key liveMeta would have hidden.
func keyExpired(deadline, now uint64) bool {
	return deadline != 0 && deadline <= now
}

// cloneMeta returns a MAP-level copy of m, or nil when m is empty: a fresh map
// holding the same Values, so adding, replacing or deleting a key in the copy
// never touches the arena's map. The Values themselves are copied as structs,
// which means a slice-backed Value (strings/ints/floats/record) still points at
// the same backing bytes — the payload mutators only ever replace whole Values,
// never write through one, and vtypes.Metadata documents that contract for
// callers too.
//
// Callers that read m from arena.Metadata(slot) MUST hold h.mu (read or write)
// while doing so; the copy then lets the four payload mutators build newMeta
// without aliasing the arena's MAP.
func cloneMeta(m Metadata) Metadata {
	if len(m) == 0 {
		return nil
	}
	out := make(Metadata, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// cloneKeyExpires returns a shallow copy of m (deadlines are scalars), or nil
// when m is empty — the clone-on-write step the payload mutators use so they
// never mutate the arena's live per-slot map in place.
func cloneKeyExpires(m map[string]uint64) map[string]uint64 {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]uint64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// pruneKeyExpires drops any deadline entry whose key is no longer present in
// meta (a deleted/overwritten-away key carries no deadline). Returns nil when
// nothing remains, so a slot with no live deadlines stores nil (the cheap path).
func pruneKeyExpires(ke map[string]uint64, meta Metadata) map[string]uint64 {
	if len(ke) == 0 {
		return nil
	}
	for k := range ke {
		if _, present := meta[k]; !present {
			delete(ke, k)
		}
	}
	if len(ke) == 0 {
		return nil
	}
	return ke
}

// liveMetaMap returns m with any per-key-TTL-EXPIRED keys dropped, the view every
// read path (Get / search results / scroll / the filter predicate) of the named +
// MV families must see. ke is the slot/id's per-key deadline map (key -> ABSOLUTE
// unix-millis; nil = no per-key TTL). When ke is empty m is returned UNCHANGED (no
// clone, no allocation — the cheap common case). Otherwise, only when at least one
// still-present key is actually expired, it clones m and drops keys whose deadline
// is non-zero and <= now. A key is logically expired iff deadline != 0 && <= now
// (mirrors point TTL / dense hnsw.liveMeta). The returned map may ALIAS the caller's
// map on the fast path, so callers that mutate must clone first; the named/MV read
// paths only read it. The named + MV analogue of hnsw.liveMeta (which reads the
// arena's u64 keyExpires; named/MV store i64 deadlines in plain maps).
func liveMetaMap(m Metadata, ke map[string]int64, now int64) Metadata {
	if len(ke) == 0 {
		return m // fast path: no per-key TTL
	}
	var expiredAny bool
	for k, dl := range ke {
		if dl != 0 && dl <= now {
			if _, present := m[k]; present {
				expiredAny = true
				break
			}
		}
	}
	if !expiredAny {
		return m // deadlines exist but none reached / none still present
	}
	out := make(Metadata, len(m))
	for k, v := range m {
		if dl := ke[k]; dl != 0 && dl <= now {
			continue // logically expired — drop
		}
		out[k] = v
	}
	return out
}

// cloneKeyTTL returns a shallow copy of m (deadlines are scalars), or nil when m
// is empty — the clone-on-write step the named/MV payload mutators use so they
// never mutate the live per-id deadline map in place. The named/MV (i64,
// snapshot-durable) analogue of cloneKeyExpires.
func cloneKeyTTL(m map[string]int64) map[string]int64 {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]int64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// pruneKeyTTL drops any deadline entry whose key is no longer present in meta (a
// deleted/overwritten-away key carries no deadline). Returns nil when nothing
// remains so the caller stores nil (the cheap path). The named/MV analogue of
// pruneKeyExpires.
func pruneKeyTTL(ke map[string]int64, meta Metadata) map[string]int64 {
	if len(ke) == 0 {
		return nil
	}
	for k := range ke {
		if _, present := meta[k]; !present {
			delete(ke, k)
		}
	}
	if len(ke) == 0 {
		return nil
	}
	return ke
}

// applyKeyTTLMs merges relative-ms per-key TTLs (keyTTLMs: key -> RELATIVE ms)
// into the absolute deadline map ke (cloned by the caller) against the merged
// payload: for each key present in merged, a ttlMs > 0 sets an ABSOLUTE deadline
// now+ttlMs; a ttlMs <= 0 clears any deadline (the key becomes permanent). Keys
// not in merged are skipped (a deadline only applies to a present key). Returns
// the (possibly newly allocated) deadline map. Shared by named + MV SetPayload.
func applyKeyTTLMs(ke map[string]int64, keyTTLMs map[string]int64, merged Metadata, now int64) map[string]int64 {
	if len(keyTTLMs) == 0 {
		return ke
	}
	for k, ttlMs := range keyTTLMs {
		if _, present := merged[k]; !present {
			continue
		}
		if ttlMs > 0 {
			if ke == nil {
				ke = make(map[string]int64, len(keyTTLMs))
			}
			ke[k] = now + ttlMs
		} else if ke != nil {
			delete(ke, k) // ttl <= 0 clears the deadline
		}
	}
	return ke
}

// computeInsertKeyExpires turns an insert-time per-key RELATIVE-ms map (key ->
// ttlMs) into the ABSOLUTE deadline map (key -> now+ttlMs) the arena stores,
// mirroring set_payload's per-key TTL: a deadline only applies to a key actually
// present in the fresh point's payload (meta), and a ttlMs <= 0 sets no deadline
// (the key is permanent). Returns nil when nothing applies, so an empty/absent map
// leaves the slot's keyExpires cleared (byte-identical zero-overhead path). The
// absolute deadlines are what the WAL logs, so replay restores them verbatim
// (time-stable, NOT recomputed). now is unix-millis.
func computeInsertKeyExpires(now int64, meta Metadata, keyTTLMs map[string]int64) map[string]uint64 {
	if len(keyTTLMs) == 0 {
		return nil
	}
	var ke map[string]uint64
	for k, ttlMs := range keyTTLMs {
		if _, present := meta[k]; !present || ttlMs <= 0 {
			continue // a deadline only applies to a present key; ttl<=0 = permanent
		}
		if ke == nil {
			ke = make(map[string]uint64, len(keyTTLMs))
		}
		ke[k] = uint64(now) + uint64(ttlMs)
	}
	return ke
}

// SetPayload MERGES patch into id's existing payload: patch keys overwrite or add,
// all other existing keys are retained. keyTTLMs carries an OPTIONAL per-key
// RELATIVE time-to-live in milliseconds (key -> ttlMs): for each entry with
// ttlMs > 0 the engine records an ABSOLUTE deadline now+ttlMs in the slot's
// keyExpires; ttlMs <= 0 clears any deadline on that key (the key becomes
// permanent again). Reads the current metadata, computes the merged map, sets it,
// recomputes the per-key deadlines, and reindexes the payload index — ALL in ONE
// write-lock critical section. Returns the RESULTING full payload AND the
// RESULTING absolute per-key deadline map (so the Collection layer WAL-logs
// exactly what was applied, time-stable) and ErrIDNotFound for a dead/absent
// point. Does not change the vector or point TTL.
func (h *hnsw) SetPayload(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return h.setPayloadBody(id, patch, keyTTLMs, cas, false, 0)
}

// SetPayloadAt is SetPayload whose per-key deadline computation (now+ttl) AND the
// dead-point liveness gate are judged against the EXPLICIT leader-stamped clock
// nowMs, so a replicated payload mutation stamps byte-identical per-key deadlines
// and agrees on the point's liveness on every replica (#4 vector TTL determinism).
func (h *hnsw) SetPayloadAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return h.setPayloadBody(id, patch, keyTTLMs, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) setPayloadBody(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	if err := checkRecordValues(patch); err != nil {
		return nil, nil, 0, err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(h.now())
	}
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(h.arena.Version(slot)); err != nil {
		return nil, nil, 0, err // CAS mismatch: no mutation, no bump
	}
	oldMeta := h.arena.Metadata(slot)
	merged := cloneMeta(oldMeta)
	if len(patch) > 0 {
		if merged == nil {
			merged = make(Metadata, len(patch))
		}
		for k, v := range patch {
			merged[k] = v
		}
	}
	ke := cloneKeyExpires(h.arena.KeyExpires(slot))
	if len(keyTTLMs) > 0 {
		for k, ttlMs := range keyTTLMs {
			if _, present := merged[k]; !present {
				continue // a deadline only applies to a key that is in the payload
			}
			if ttlMs > 0 {
				if ke == nil {
					ke = make(map[string]uint64, len(keyTTLMs))
				}
				ke[k] = now + uint64(ttlMs)
			} else if ke != nil {
				delete(ke, k) // ttl <= 0 clears the deadline (key becomes permanent)
			}
		}
	}
	ke = pruneKeyExpires(ke, merged)
	h.maintainBM25OnPayloadChange(slot, oldMeta, merged)
	h.arena.SetMetadata(slot, merged)
	h.arena.SetKeyExpires(slot, ke)
	h.payloadIdx.reindex(slot, merged)
	h.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return merged, ke, h.arena.BumpVersion(slot), nil
}

// OverwritePayload REPLACES id's entire payload with meta (a nil/empty meta
// clears it). keyTTLMs (RELATIVE ms, key -> ttlMs) REPLACES the per-key deadline
// set: deadlines from the new map are recomputed against now; any prior deadline
// is dropped. Read-compute-set-reindex run in one write-lock critical section.
// Returns the resulting payload (a deep copy of meta) AND the resulting absolute
// per-key deadline map, plus ErrIDNotFound for a dead/absent point. Does not
// change the vector or point TTL.
func (h *hnsw) OverwritePayload(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return h.overwritePayloadBody(id, meta, keyTTLMs, cas, false, 0)
}

// OverwritePayloadAt is OverwritePayload whose per-key deadline computation (now+ttl)
// AND the dead-point liveness gate are judged against the EXPLICIT leader-stamped
// clock nowMs (#4 vector TTL determinism).
func (h *hnsw) OverwritePayloadAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return h.overwritePayloadBody(id, meta, keyTTLMs, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) overwritePayloadBody(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	if err := checkRecordValues(meta); err != nil {
		return nil, nil, 0, err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(h.now())
	}
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(h.arena.Version(slot)); err != nil {
		return nil, nil, 0, err // CAS mismatch: no mutation, no bump
	}
	oldMeta := h.arena.Metadata(slot)
	newMeta := cloneMeta(meta)
	var ke map[string]uint64
	if len(keyTTLMs) > 0 {
		for k, ttlMs := range keyTTLMs {
			if _, present := newMeta[k]; !present || ttlMs <= 0 {
				continue
			}
			if ke == nil {
				ke = make(map[string]uint64, len(keyTTLMs))
			}
			ke[k] = now + uint64(ttlMs)
		}
	}
	h.maintainBM25OnPayloadChange(slot, oldMeta, newMeta)
	h.arena.SetMetadata(slot, newMeta)
	h.arena.SetKeyExpires(slot, ke)
	h.payloadIdx.reindex(slot, newMeta)
	h.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return newMeta, ke, h.arena.BumpVersion(slot), nil
}

// MutatePayloadRecord applies fn to the operate record stored under key in id's
// payload and stores what fn returns, all inside ONE write-lock critical
// section: read the current bytes, transform, bound, store/delete, prune the
// key's deadline, reindex, bump. It is the store-agnostic seam ops.handleVector-
// Operate drives with applyRecordBytes — vector never imports ops, so the
// transform arrives as a function.
//
// Returns the RESULTING payload and per-key deadlines (so Collection WAL-logs
// exactly what was applied), the resulting version, and changed. changed=false
// means the call was a deliberate no-op (a failed CHECK, or a delete of an
// absent key): nothing was stored, nothing was logged, the version is the
// CURRENT one and was not bumped.
func (h *hnsw) MutatePayloadRecord(id uint64, key string, fn RecordMutator, cas CASCond) (Metadata, map[string]uint64, uint64, bool, error) {
	return h.mutatePayloadRecordBody(id, key, fn, cas, false, 0)
}

// MutatePayloadRecordAt judges the dead-point liveness gate, the per-key
// deadline check and the stale-deadline drop against the EXPLICIT leader-stamped
// clock nowMs, so a replicated record mutation sees the same record and produces
// the same bytes on every replica (#4 vector TTL determinism, mirroring
// SetPayloadAt).
func (h *hnsw) MutatePayloadRecordAt(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, bool, error) {
	return h.mutatePayloadRecordBody(id, key, fn, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) mutatePayloadRecordBody(id uint64, key string, fn RecordMutator, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, bool, error) {
	if fn == nil || key == "" || key == contentField {
		return nil, nil, 0, false, ErrPayloadKeyNotRecord
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(h.now())
	}
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		return nil, nil, 0, false, ErrIDNotFound
	}
	if err := cas.check(h.arena.Version(slot)); err != nil {
		return nil, nil, 0, false, err // CAS mismatch: no mutation, no bump
	}
	oldMeta := h.arena.Metadata(slot)
	oldKE := h.arena.KeyExpires(slot)
	old, exists, err := currentRecordValue(oldMeta, key, keyExpired(oldKE[key], now))
	if err != nil {
		return nil, nil, 0, false, err
	}
	rec, act, err := fn(old, exists)
	if err != nil {
		return nil, nil, 0, false, err
	}
	switch act {
	case RecordUnchanged:
		return nil, nil, h.arena.Version(slot), false, nil
	case RecordDelete:
		if !exists {
			return nil, nil, h.arena.Version(slot), false, nil
		}
	case RecordStore:
		// The POST-mutation bound. checkRecordValues (vector/metadata.go) is the
		// same cap on the ingest side, but it only ever sees a caller's patch —
		// it cannot reach bytes the operate engine just produced, which is what
		// this site exists for. Keep the two in step.
		//
		// SIZE ONLY, DELIBERATELY: no record.Validate here. These bytes come from
		// applyRecordBytes, which phase 1 held to the tree oracle, so they are
		// well-formed by construction; re-decoding the whole record on every
		// counter increment would double the op's cost for a case that cannot
		// arise. The shape check belongs where bytes arrive from a CALLER, which
		// is checkRecordValues.
		if len(rec) > maxRecordValueBytes {
			return nil, nil, 0, false, fmt.Errorf("%w: payload key %q would hold a %d-byte record, the cap is %d bytes",
				ErrRecordTooLarge, key, len(rec), maxRecordValueBytes)
		}
	default:
		return nil, nil, 0, false, ErrRecordMutation
	}
	merged := cloneMeta(oldMeta)
	if act == RecordDelete {
		delete(merged, key)
		if len(merged) == 0 {
			// An emptied payload stores nil, exactly as deletePayloadKeysBody
			// normalizes it. Storing an empty non-nil map instead would make the
			// live state differ from the same state after a WAL replay (the log
			// encodes an empty payload as absent and restores nil), for no gain.
			merged = nil
		}
	} else {
		if merged == nil {
			merged = make(Metadata, 1)
		}
		merged[key] = Value{Kind: ValueRecord, Rec: rec} // ownership hand-off; see RecordMutator
	}
	ke := cloneKeyExpires(oldKE)
	// A logically expired key read as ABSENT above: its stale deadline must go,
	// or the record this call just created would be invisible the instant it is
	// written. pruneKeyExpires only drops deadlines for keys no longer present.
	if ke != nil && keyExpired(ke[key], now) {
		delete(ke, key)
	}
	ke = pruneKeyExpires(ke, merged)
	h.maintainBM25OnPayloadChange(slot, oldMeta, merged)
	h.arena.SetMetadata(slot, merged)
	h.arena.SetKeyExpires(slot, ke)
	h.payloadIdx.reindex(slot, merged)
	h.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return merged, ke, h.arena.BumpVersion(slot), true, nil
}

// DeletePayloadKeys removes the listed keys from id's payload (absent keys are a
// no-op) AND drops their per-key deadlines. Read-compute-set-reindex run in one
// write-lock critical section. Returns the resulting payload AND the resulting
// absolute per-key deadline map, plus ErrIDNotFound for a dead/absent point.
// Does not change the vector or point TTL.
func (h *hnsw) DeletePayloadKeys(id uint64, keys []string, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return h.deletePayloadKeysBody(id, keys, cas, false, 0)
}

// DeletePayloadKeysAt is DeletePayloadKeys whose dead-point liveness gate is judged
// against the EXPLICIT leader-stamped clock nowMs, so replicas agree on whether the
// point is live and therefore whether the mutation applies (#4 vector TTL
// determinism). Dropping keys computes no deadline, so nowMs only gates liveness.
func (h *hnsw) DeletePayloadKeysAt(id uint64, keys []string, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return h.deletePayloadKeysBody(id, keys, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) deletePayloadKeysBody(id uint64, keys []string, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(h.now())
	}
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(h.arena.Version(slot)); err != nil {
		return nil, nil, 0, err // CAS mismatch: no mutation, no bump
	}
	oldMeta := h.arena.Metadata(slot)
	newMeta := cloneMeta(oldMeta)
	ke := cloneKeyExpires(h.arena.KeyExpires(slot))
	for _, k := range keys {
		delete(newMeta, k)
		if ke != nil {
			delete(ke, k)
		}
	}
	if len(newMeta) == 0 {
		newMeta = nil
	}
	ke = pruneKeyExpires(ke, newMeta)
	h.maintainBM25OnPayloadChange(slot, oldMeta, newMeta)
	h.arena.SetMetadata(slot, newMeta)
	h.arena.SetKeyExpires(slot, ke)
	h.payloadIdx.reindex(slot, newMeta)
	h.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return newMeta, ke, h.arena.BumpVersion(slot), nil
}

// ClearPayload removes ALL of id's payload (payload → empty/nil) AND all per-key
// deadlines. Runs under one write-lock critical section. Returns the resulting
// (nil) payload + (nil) deadline map and ErrIDNotFound for a dead/absent point.
// Does not change the vector or point TTL.
func (h *hnsw) ClearPayload(id uint64, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return h.clearPayloadBody(id, cas, false, 0)
}

// ClearPayloadAt is ClearPayload whose dead-point liveness gate is judged against
// the EXPLICIT leader-stamped clock nowMs (#4 vector TTL determinism).
func (h *hnsw) ClearPayloadAt(id uint64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return h.clearPayloadBody(id, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) clearPayloadBody(id uint64, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(h.now())
	}
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(h.arena.Version(slot)); err != nil {
		return nil, nil, 0, err // CAS mismatch: no mutation, no bump
	}
	h.maintainBM25OnPayloadChange(slot, h.arena.Metadata(slot), nil)
	h.arena.SetMetadata(slot, nil)
	h.arena.SetKeyExpires(slot, nil)
	h.payloadIdx.reindex(slot, nil)
	h.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return nil, nil, h.arena.BumpVersion(slot), nil
}

// RestorePayload sets id's payload to meta AND its per-key deadlines to the given
// ABSOLUTE unix-millis map directly — NO now+ttl recomputation. It is the WAL
// replay primitive (the WAL logs the resulting payload + absolute deadlines, so
// replay must restore them verbatim or pending deadlines would drift forward by
// the time since the crash). Read-set-reindex under one write-lock critical
// section. Returns ErrIDNotFound for a dead/absent point. Both maps are stored by
// reference (caller hands off ownership); nil clears the respective state.
func (h *hnsw) RestorePayload(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) error {
	// REPLAY body: size only. Its only caller is the WAL replay of a set_payload
	// record (vector/collection.go replayRecord); a shape check here would refuse
	// to rebuild an acked write. See checkRecordValuesSize.
	if err := checkRecordValuesSize(meta); err != nil {
		return err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpired(slot) {
		return ErrIDNotFound
	}
	h.arena.SetMetadata(slot, meta)
	h.arena.SetKeyExpires(slot, pruneKeyExpires(cloneKeyExpires(keyExpires), meta))
	h.payloadIdx.reindex(slot, meta)
	// Restore the logged version VERBATIM (NOT a bump) so a payload mutation's
	// resulting version survives WAL replay time-stable. A 0 (an old WAL record
	// predating versions) leaves the version untouched — the insert-replay already
	// set a sane default (1).
	if version != 0 {
		h.arena.SetVersion(slot, version)
	}
	h.bumpData() // WAL-replayed payload-value change: invalidate the order_by snapshot
	return nil
}

// liveLocked reports whether id currently maps to a LIVE slot: present in the
// idMap, not tombstoned, and not TTL-expired — the exact admission predicate the
// search path uses (see admits/isExpired). Must hold h.mu (read or write).
func (h *hnsw) liveLocked(id uint64) bool {
	return h.liveLockedAt(id, uint64(h.now()))
}

// liveLockedAt is liveLocked judging TTL expiry against the caller-supplied `now`
// (unix millis) instead of a fresh wall-clock read, so the insert-if-absent
// liveness OUTCOME (insert vs no-op) is decided on the leader-stamped clock and is
// identical on every replica (#4 vector TTL determinism). Must hold h.mu.
func (h *hnsw) liveLockedAt(id uint64, now uint64) bool {
	slot, ok := h.arena.Slot(id)
	if !ok {
		return false
	}
	return !h.tombstoned[slot] && !h.isExpiredAt(slot, now)
}

// Exists reports whether id is currently LIVE in the index, using the same
// liveness definition as search admission: an id that is tombstoned (deleted) or
// TTL-expired is NOT live, and a never-inserted id is absent. O(1): an idMap
// lookup plus a tombstone/expiry check under the read lock — it never scans. This
// is the cheap liveness probe the online-copy resurrection guard uses to re-check
// the source generation after an insert-if-absent (Race B).
func (h *hnsw) Exists(id uint64) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.liveLocked(id)
}

// InsertIfAbsent inserts vec under id ONLY if id is not currently live, and
// reports whether it inserted. When id is already live it is a no-op (returns
// inserted=false) and the existing value is left untouched — it never clobbers a
// concurrent write. A tombstoned or TTL-expired slot counts as absent, so
// if-absent resurrects it with the new value (same reclaim path as Insert).
//
// ATOMICITY (Race A): the liveness check and the insert run inside ONE write-lock
// critical section — there is no check-then-act gap. Combined with Raft
// serialization (vector_insert_if_absent is a single OpReadWrite op on the
// partition's log), this guarantees that when if-absent races a concurrent
// upsert on the same id, the live value always wins:
//   - if-absent lands first → inserts stale, the upsert then replaces it ✓
//   - upsert lands first → if-absent sees the id live → no-op, upsert value kept ✓
//
// Args mirror Insert (ttl=0 no expiry, meta=nil none, sparse=nil none).
func (h *hnsw) InsertIfAbsent(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) (inserted bool, err error) {
	return h.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, nil, 0)
}

// InsertIfAbsentVersion is InsertIfAbsent that, on a real insert, sets the
// point's version VERBATIM to `version` (NOT 1). version==0 reproduces the
// InsertIfAbsent semantics (fresh insert → version 1). Used by the reshard copy
// pass to carry a copied point's per-point CAS version while still never
// clobbering a concurrent live dual-write (Race A). keyExpires is the ABSOLUTE
// per-key payload deadline map set VERBATIM (NOT recomputed) on a real insert so
// the online reshard copy keeps the point's original key deadlines time-stable;
// nil keyExpires (the common case) is byte-identical to today's behavior.
func (h *hnsw) InsertIfAbsentVersion(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) (inserted bool, err error) {
	return h.insertIfAbsentBody(id, vec, ttl, meta, sparse, keyExpires, version, false, 0)
}

// InsertIfAbsentVersionAt is InsertIfAbsentVersion whose liveness OUTCOME (insert
// vs no-op), dead-slot reclaim, and point-TTL deadline are all judged against the
// EXPLICIT leader-stamped clock nowMs — so skewed replicas agree on whether an
// expired id resurrects, and stamp identical deadlines (#4 vector TTL
// determinism).
func (h *hnsw) InsertIfAbsentVersionAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) (inserted bool, err error) {
	return h.insertIfAbsentBody(id, vec, ttl, meta, sparse, keyExpires, version, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) insertIfAbsentBody(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, stamped bool, nowMs uint64) (inserted bool, err error) {
	if err := checkRecordValues(meta); err != nil {
		return false, err
	}
	start := time.Now()
	defer func() { h.insertLat.observe(time.Since(start)) }()

	if len(vec) != h.cfg.Dim {
		return false, ErrDimMismatch
	}

	stored := vec
	if h.cfg.Metric == Cosine {
		stored = append([]float32(nil), vec...)
		normalize(stored)
	}

	var task linkTask
	var placed bool
	h.linkMu.RLock() // serializer barrier, spanning placement + link (see insertBody)
	defer h.linkMu.RUnlock()
	// Placement scoped to a closure so the write lock is released by a defer even
	// on a panic — see insertBody.
	err = func() error {
		h.mu.Lock()
		defer h.mu.Unlock()
		now := nowMs
		if !stamped {
			now = uint64(h.now())
		}

		// Single critical section: liveness check then PLACEMENT, no gap (Race A).
		// The gap Race A is about is between observing "absent" and claiming the id,
		// which placement completes; the link phase that follows outside the lock
		// cannot reintroduce it, because the id is already in the arena's idMap by
		// then and a competing if-absent would see it live.
		if h.liveLockedAt(id, now) {
			return nil // no-op: id already live, never clobber the live value
		}
		// Rate limit only the actual insert (the no-op path above must not consume
		// quota, or a copy's liveness-driven if-absent would drain the bucket). The
		// bucket has its own mutex, so taking it under h.mu is safe.
		if !h.bucket.Take() {
			h.quotaRejects.Add(1)
			return ErrCollectionRateLimited
		}
		// setVersion==version: 0 → fresh insert bumps to 1; non-zero → set verbatim.
		// keyExpires is the ABSOLUTE per-key deadline map set VERBATIM (NOT recomputed
		// now+ttl) — nil for a plain if-absent, the carried deadlines for an online
		// reshard copy. placeLockedAt sets it via arena.SetKeyExpires (mirror RestoreInsert).
		if _, err := h.placeLockedAt(id, stored, ttl, meta, sparse, keyExpires, version, now, &task); err != nil {
			return err
		}
		placed = true
		return nil
	}()
	if err != nil {
		return false, err
	}
	if !placed {
		return false, nil // the id was already live; nothing placed, nothing to link
	}
	h.runLink(&task)
	return true, nil
}

// Search returns the k nearest neighbors of query under the configured metric,
// sorted ascending by distance. k <= 0 returns an empty result. Allocates the
// returned slice; for a zero-allocation hot path, use SearchInto with a reused
// buffer.
func (h *hnsw) Search(query []float32, k int) ([]Result, error) {
	return h.SearchInto(nil, query, k, Filter{})
}

// SearchFiltered returns the k nearest neighbors whose metadata satisfies the
// filter. Convenience wrapper over SearchInto that allocates the result.
func (h *hnsw) SearchFiltered(query []float32, k int, filter Filter) ([]Result, error) {
	return h.SearchInto(nil, query, k, filter)
}

// SearchFilteredWith is SearchInto with an OPTIONAL external metadata provider
// (the named-vector hook). metaOf == nil is byte-identical to SearchInto (arena
// metadata + payload-index planner — today's single-vector behavior). When
// metaOf != nil the filter predicate is evaluated against the EXTERNAL per-point
// payload via metaOf(id): the sub-arena carries no metadata, so this is
// PREDICATE-EVAL ONLY — the payload-index/filter-first planner is bypassed
// (the empty index would otherwise narrow to nothing) and the search always
// traverses the graph, gating membership against the provider.
func (h *hnsw) SearchFilteredWith(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error) {
	return h.searchIntoWith(dst, query, k, filter, metaOf)
}

// SearchInto appends up to k nearest neighbors matching filter onto dst and
// returns the extended slice. Pass a reused dst (and the L2/DotProduct metric,
// which needs no query normalization) for a zero-allocation steady state — all
// per-query scratch is pooled and the result lands in the caller's buffer. A
// nil dst behaves like Search. When a predicate is active, ef is widened
// dynamically up to MaxEfSearch.
func (h *hnsw) SearchInto(dst []Result, query []float32, k int, filter Filter) ([]Result, error) {
	return h.searchIntoWith(dst, query, k, filter, nil)
}

// searchIntoWith is SearchInto with an OPTIONAL external metadata provider. The
// metaOf == nil path is byte-identical to the historical SearchInto body (arena
// metadata + the payload-index filter-first planner). When metaOf != nil the
// filter-first planner is SKIPPED entirely: a metadata-less sub-arena has no
// payload-index entries, so consulting it would narrow the candidate set to
// nothing and silently return wrong/empty results. The provider path therefore
// always traverses the graph (predicate-eval only), gating membership against
// the external payload via searchDenseLockedWith.
func (h *hnsw) searchIntoWith(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error) {
	// A failed mmap slab growth freed the backing region; reject rather than
	// read from it (see arena.poisoned / ErrIndexPoisoned).
	if h.arena.poisoned.Load() {
		return dst, ErrIndexPoisoned
	}
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if len(query) != h.cfg.Dim {
		return dst, ErrDimMismatch
	}
	if k <= 0 {
		return dst, nil
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return dst, err
	}

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	q := query
	if h.cfg.Metric == Cosine {
		s.qbuf = append(s.qbuf[:0], query...)
		normalize(s.qbuf)
		q = s.qbuf
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	// Re-check under h.mu (poison is published under the write lock): closes the
	// window between the pre-lock check and here, and covers the filter-first
	// branch below, whose exact rescore reads vecFor into a distance kernel.
	if h.arena.poisoned.Load() {
		return dst, ErrIndexPoisoned
	}

	// Query planner: when an indexed filter narrows to a materializable candidate
	// set, choose between exact brute force over those candidates (filter-first)
	// and filtered graph traversal by estimated cost (preferFilterFirst). Filter-
	// first is exact and wins for selective filters (which otherwise degrade HNSW
	// recall as the graph keeps rejecting candidates); graph wins for
	// non-selective filters where brute-forcing the large set is the costlier path.
	//
	// SKIPPED on the external-provider path (metaOf != nil): the sub-arena holds
	// no metadata, so its payload index is empty — filter-first would narrow to
	// nothing. The provider path is predicate-eval only (full graph traversal).
	if pred != nil && metaOf == nil {
		limit := h.effectiveFilterFirstLimit(h.arena.Size())
		// maxCand is the largest candidate count the planner would still
		// choose filter-first for (see filterFirstCrossover). Materializing
		// the candidate set only to discover preferFilterFirst rejects it is
		// pure waste — most of a filtered query's allocations, per profiling —
		// so candidatesCapped aborts the moment the set exceeds maxCand
		// instead of building the superset that would be thrown away.
		maxCand := h.filterFirstCrossover(k, limit)
		// ONE narrowing plan, TWO consumers. candidatesCapped used to own the
		// collection, but the traversal gate below needs the same posting sets, and
		// collecting twice would make a query pay twice for the sets the index has
		// to BUILD rather than look up (a Match intersection, a geo cell union).
		// Splitting collect from intersect keeps the planner's behavior identical —
		// same limit, same maxCand, same abort — while the gate reads the plan the
		// planner already paid for. It also means the gate can never materialize a
		// posting set the planner declined to: the limit that bounds one bounds both.
		//
		// The gate is disarmed on the way out so it cannot outlive the read lock it
		// was built under, nor the scratch's trip back to the pool. Registered
		// AFTER the pool Put above, so it runs BEFORE it — and hoisted here in m5,
		// because the column and complement oracles arm for filters the planner
		// produced no plan for at all, so the disarm can no longer live inside the
		// branch that has one.
		defer s.gate.disable()
		plan, planOK := h.payloadIdx.collectNarrowSets(filter, limit)
		if planOK {
			if cands, ok := intersectSlotSets(plan, maxCand); ok {
				return h.filterFirstKNN(dst, cands, q, k, pred), nil
			}
		}
		// The planner declined — either the candidate set was too big to
		// brute-force, or the filter produced no plan at all. Either way the query
		// is about to traverse the graph and pay an admission decision per
		// candidate, so what remains is choosing the cheapest oracle for that
		// decision. All three are optional: leaving the gate disabled is exactly
		// today's path and always correct, so every bail below is a throughput
		// decision.
		//
		// ORDER IS BY PER-QUERY PRICE, cheapest first.
		//
		//  1. COLUMNS (m5). A numeric range answered from a slot-indexed
		//     []float64: one array read per candidate and NOTHING per query, since
		//     the column is built once per field and reused forever. That makes it
		//     unconditionally better than either bitset when it applies, at every
		//     selectivity and every corpus size — which is why it is tried first
		//     and why it needs no cost model at all.
		//  2. THE POSITIVE BITSET (m4). Cheaper per candidate than a column (one
		//     bit, a far smaller footprint) but priced in accepted posting mass, so
		//     it is the right answer for a selective non-range filter.
		//  3. THE COMPLEMENT BITSET (m5). The same, priced in REJECTED mass, for
		//     the high-pass-rate filters (2) must decline.
		if !h.buildColumnGate(s, filter) {
			if planOK {
				h.buildAdmitGate(s, filter, plan, k)
			}
			if !s.gate.active() {
				h.buildComplementGate(s, filter, k)
			}
		}
	}
	return h.searchDenseLockedWith(s, q, k, pred, dst, metaOf), nil
}

// filterFirstCrossover returns the largest ncand in [0, limit] for which
// preferFilterFirst(ncand, k) is true. hnsw's preferFilterFirst is monotonically
// non-increasing in ncand (larger ncand -> larger selectivity -> smaller or
// equal efEff, while the ncand <= graphCost comparison's LHS only grows), so
// there is a single crossover point.
func (h *hnsw) filterFirstCrossover(k, limit int) int {
	return crossoverOf(func(ncand int) bool { return h.preferFilterFirst(ncand, k) }, limit)
}

// crossoverOf returns the largest ncand in [0, limit] for which prefer(ncand)
// holds, by binary search. It is shared by the hnsw and IVF planners, whose
// cost models differ but share the property this search requires: prefer is
// monotonically non-increasing in ncand and prefer(0) is unconditionally true.
// Because it evaluates the model itself rather than inverting it in closed
// form, the cap it returns is exact under the model's own arithmetic — a model
// change cannot silently desynchronize the cap from the decision.
func crossoverOf(prefer func(ncand int) bool, limit int) int {
	if limit <= 0 {
		return 0
	}
	if prefer(limit) {
		return limit // filter-first wins even at the limit -> no tighter cap
	}
	lo, hi := 0, limit // prefer(0) is unconditionally true
	for lo+1 < hi {
		mid := lo + (hi-lo)/2
		if prefer(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo
}

// preferFilterFirst is the cost-based planner decision: should an index-narrowed
// filtered search brute-force the ncand candidates (exact) rather than traverse
// the graph? It compares estimated distance-computation costs:
//
//   - filter-first ≈ ncand (one distance per candidate).
//   - graph ≈ efEff · 2M, where the filtered search widens ef to roughly k/s to
//     collect k admitted results (s = selectivity = ncand/N), clamped to the
//     [base, MaxEfSearch] range the search actually uses, and expands ~2M
//     neighbors per visited node.
//
// Because the graph term grows as the filter gets more selective (small s → wide
// ef), the model naturally prefers exact filter-first exactly where graph recall
// would suffer (the filtered-recall cliff), and switches to graph for
// non-selective filters where it is cheaper and recall is unaffected.
func (h *hnsw) preferFilterFirst(ncand, k int) bool {
	if ncand == 0 {
		return true // nothing to brute force: trivially exact and cheapest
	}
	n := h.arena.Size()
	if n <= 0 {
		return true
	}
	return float64(ncand) <= h.graphVisitEstimate(ncand, k)
}

// graphVisitEstimate is the graph term of the planner's cost model, factored out
// so it has exactly one definition: the estimated number of candidates a
// FILTERED graph traversal scores when the filter matches ncand of the n live
// points. The filtered search widens ef to roughly k/s (s = ncand/n) to collect
// k admitted results, clamped to the [base, MaxEfSearch] window the search
// actually uses, and expands ~2M neighbors per visited node — so efEff · 2M.
//
// preferFilterFirst weighs it against ncand distance computations;
// gateProfitable weighs the SAME number against the cost of building a bitset.
// Sharing the estimate is the point: the two decisions are consecutive branches
// on one query, and a gate that disagreed with the planner about the size of the
// traversal it is accelerating would be tuning against a fiction. Returns 0 for
// an empty index.
func (h *hnsw) graphVisitEstimate(ncand, k int) float64 {
	n := h.arena.Size()
	if n <= 0 {
		return 0
	}
	if k < 1 {
		k = 1
	}
	s := float64(ncand) / float64(n)
	if s > 1 {
		s = 1 // candidate set is a superset (tombstones) → treat as non-selective
	}

	base := h.cfg.EfSearch
	if base < 2*k {
		base = 2 * k
	}
	if h.quant != nil {
		rf := h.cfg.RescoreFactor
		if rf <= 0 {
			rf = defaultRescoreFactor
		}
		if rf*k > base {
			base = rf * k
		}
	}
	maxEf := h.cfg.MaxEfSearch
	if maxEf <= 0 {
		maxEf = defaultMaxEfSearch
	}
	efEff := float64(k) / s
	if efEff < float64(base) {
		efEff = float64(base)
	}
	if efEff > float64(maxEf) {
		efEff = float64(maxEf)
	}
	return efEff * float64(2*h.cfg.M)
}

// Dim returns the configured vector dimensionality.
func (h *hnsw) Dim() int { return h.cfg.Dim }

// vecsForIDs returns, for each live id present in this index, a COPY of its float
// vector. h.vecFor returns exact arena floats (which it ALIASES) when present and
// a freshly-allocated reconstruction when the floats were PQ-dropped; either way
// we copy so the caller never retains an arena view. Absent ids are omitted. Takes
// h.mu (read) itself (the caller holds the owning collection's lock, not this
// sub-index's). The id-keyed analogue of the per-space Get arena read.
func (h *hnsw) vecsForIDs(ids []uint64) map[uint64][]float32 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make(map[uint64][]float32, len(ids))
	for _, id := range ids {
		slot, ok := h.arena.Slot(id)
		if !ok {
			continue
		}
		out[id] = append([]float32(nil), h.vecFor(slot)...)
	}
	return out
}

// withVecAccess holds h.mu (read) for the duration of fn and passes a getter that
// returns each id's vector as an arena VIEW (no copy). The view is valid only
// while fn runs (the lock is released on return). The allocation-free analogue of
// vecsForIDs for the MaxSim hot path. (PQ-dropped floats: vecFor reconstructs into
// a fresh slice, same as vecsForIDs would.)
func (h *hnsw) withVecAccess(fn func(get func(id uint64) ([]float32, bool))) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	fn(func(id uint64) ([]float32, bool) {
		slot, ok := h.arena.Slot(id)
		if !ok {
			return nil, false
		}
		return h.vecFor(slot), true
	})
}

// filterFirstThreshold is the maximum candidate-set size for which the planner
// uses brute-force filter-first search instead of graph traversal.
func (h *hnsw) filterFirstThreshold() int {
	if h.cfg.FilterFirstThreshold > 0 {
		return h.cfg.FilterFirstThreshold
	}
	return defaultFilterFirstThreshold
}

// effectiveFilterFirstLimit folds the relative selectivity gate into the absolute
// threshold for the given live count (0 bp -> filterFirstThreshold exactly).
func (h *hnsw) effectiveFilterFirstLimit(liveCount int) int {
	return effectiveFilterFirstLimit(h.filterFirstThreshold(), h.cfg.FilterFirstRelativeBP, liveCount)
}

// filterFirstKNN computes exact top-k over an index-narrowed candidate slot set,
// re-applying the full predicate (and tombstone/TTL admission) so the result is
// correct even if the candidate set is a superset. Must hold the read lock.
func (h *hnsw) filterFirstKNN(dst []Result, cands []uint32, q []float32, k int, pred Predicate) []Result {
	dist := h.metricDist()
	now := uint64(h.now()) // one clock read for the whole candidate scan
	matched := make([]slotDist, 0, len(cands))
	for _, slot := range cands {
		if !h.admits(slot, pred, now) {
			continue
		}
		matched = append(matched, slotDist{slot: slot, dist: dist(q, h.vecFor(slot))})
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	for i := 0; i < len(matched) && i < k; i++ {
		// No allocation re-check: cands come from the payload index, which only
		// ever holds slots of inserted points, and admits() already rejected the
		// dead ones. Emitting the id verbatim is what makes user id 0 returnable.
		dst = append(dst, Result{ID: h.slotID(matched[i].slot), Distance: matched[i].dist})
	}
	return dst
}

// filterFirstByID computes exact top-k over an index-narrowed candidate ID set
// whose payload lives in an EXTERNAL map (the named/MV families: the sub-arena
// carries the VECTORS but no metadata). It is the id-keyed, external-metadata
// analogue of filterFirstKNN: for each candidate id it resolves the vector via
// the sub-arena, applies the tombstone/TTL liveness gate AND the external
// predicate RE-CHECK (pred evaluated against metaOf(id) — the live shared
// payload view, so TTL/per-key-TTL/over-cover are all handled), scores under the
// active metric, and returns the k closest.
//
// candidates() yields a SUPERSET, so the re-check is what makes the result
// correct: a stale/expired/over-covering id is simply dropped here. ids that the
// sub-arena does not carry (the point omitted this space) are skipped. Takes the
// read lock itself (the caller holds the owning collection's lock, not the
// sub-index's). q is normalized for the cosine metric exactly like SearchInto.
func (h *hnsw) filterFirstByID(dst []Result, cands []uint64, query []float32, k int, pred Predicate, metaOf func(id uint64) Metadata) []Result {
	if len(query) != h.cfg.Dim || k <= 0 {
		return dst
	}
	q := query
	if h.cfg.Metric == Cosine {
		qbuf := make([]float32, len(query))
		copy(qbuf, query)
		normalize(qbuf)
		q = qbuf
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	dist := h.metricDist()
	now := uint64(h.now()) // one clock read for the whole candidate scan
	matched := make([]slotDist, 0, len(cands))
	for _, id := range cands {
		slot, ok := h.arena.Slot(id)
		if !ok {
			continue // this point omitted this named space (no vector here)
		}
		if !h.admitsWith(slot, pred, metaOf, now) {
			continue // tombstoned/expired or fails the live-meta predicate re-check
		}
		matched = append(matched, slotDist{slot: slot, dist: dist(q, h.vecFor(slot))})
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	for i := 0; i < len(matched) && i < k; i++ {
		id := h.arena.ID(matched[i].slot)
		dst = append(dst, Result{ID: id, Distance: matched[i].dist})
	}
	return dst
}

// searchDenseLocked runs the dense KNN search under an already-held read lock,
// using the caller-provided scratch (s) for all per-query buffers and appending
// results onto dst. q must already be cosine-normalized if the metric requires
// it. pred is the compiled filter (nil = no filter). Shared by SearchInto and
// HybridSearch so they navigate the graph identically.
func (h *hnsw) searchDenseLocked(s *layerScratch, q []float32, k int, pred Predicate, dst []Result) []Result {
	return h.searchDenseLockedWith(s, q, k, pred, dst, nil)
}

// searchDenseLockedWith is searchDenseLocked with an OPTIONAL external metadata
// provider (the named-vector hook). metaOf == nil is byte-identical to today
// (the descent uses searchLayer with a nil predicate as before; the bottom-level
// expansion gates membership via the arena). When metaOf is set, only the
// bottom-level admission gate is rerouted to the external payload via
// searchLayerWith — navigation is unchanged. This is PREDICATE-EVAL ONLY; the
// caller (SearchFilteredWith) never routes the payload index on the provider
// path, so a metadata-less sub-arena is correctly searched by full traversal.
func (h *hnsw) searchDenseLockedWith(s *layerScratch, q []float32, k int, pred Predicate, dst []Result, metaOf func(id uint64) Metadata) []Result {
	// Under-lock backstop: a failed mmap slab growth nils the graph/vector regions,
	// so every dense read op (Search, MMR, group, discover, hybrid) funnels through
	// here before touching h.level0 / the arena floats. Return the caller's dst
	// unchanged rather than dereference the freed region. This runs under the
	// caller's h.mu RLock, so it also closes the TOCTOU against a concurrent grow
	// that poisons after a top-of-op check. Ops with an error return additionally
	// reject with ErrIndexPoisoned at their own top; the no-error ops (SearchText)
	// rely on this backstop for panic-safety. See arena.poisoned.
	if h.arena.poisoned.Load() {
		return dst
	}
	// The entry point and the top level are read as ONE pair (see entryState): a
	// linker running concurrently under the read lock can promote a taller node,
	// and descending from the old entry through the new top level would walk
	// levels that node does not have.
	ep, maxLevel := h.entryState()
	if maxLevel < 0 {
		return dst
	}

	// Navigation scorer: approximate (over codes) when quantized, else exact.
	score := h.searchScorer(q)

	// Snapshot the wall clock ONCE for the whole query: the current time is
	// loop-invariant under the search RLock, so the per-candidate admission gate
	// (TTL expiry + per-key-TTL metadata) reads it here instead of re-reading
	// time.Now thousands of times across the descent and bottom-level expansion.
	now := uint64(h.now())

	// Greedy descent: width 1 from the top level down to level 1, reusing s.cur
	// as the single-entry frontier. Navigation ignores the predicate.
	s.cur = append(s.cur[:0], ep)
	for lc := maxLevel; lc > 0; lc-- {
		near := h.searchLayer(s, score, s.cur, 1, lc, nil, now)
		if len(near) == 0 {
			continue
		}
		ep := near[0].slot
		s.cur = append(s.cur[:0], ep)
	}

	// Bottom-level expansion. Without a predicate, one pass at ef=max(EfSearch,k).
	// With a predicate, widen ef until we collect k matches or hit MaxEfSearch.
	ef := h.cfg.EfSearch
	if ef < k {
		ef = k
	}
	if pred != nil && ef < 2*k {
		ef = 2 * k
	}
	// Quantized search over-collects so the exact rescore stage has a wide
	// enough candidate pool to re-rank.
	if h.quant != nil {
		rf := h.cfg.RescoreFactor
		if rf <= 0 {
			rf = defaultRescoreFactor
		}
		if want := rf * k; ef < want {
			ef = want
		}
	}
	maxEf := h.cfg.MaxEfSearch
	if maxEf <= 0 {
		maxEf = defaultMaxEfSearch
	}

	base := len(dst)
	for {
		cand := h.searchLayerWith(s, score, s.cur, ef, 0, pred, metaOf, now)
		// Replace the approximate ordering with exact float32 distances before
		// top-k selection, so quantization never reaches the final ranking. SKIP
		// the exact rescore when the floats have been dropped (PQDropVecs): there
		// are no floats to rescore on, so the ADC ordering from searchLayerWith IS
		// the result — searchLayerWith already sorts ascending by ADC distance, and
		// we truncate to k below. The over-collect (ef = RescoreFactor*k) is kept so
		// the ADC ordering is stable across the top-k boundary even without rescore.
		// PQDropVecs=false (and every non-PQ quantized index) still rescores ⇒
		// byte/behaviour-identical to before.
		if h.quant != nil && !h.vecsDropped() {
			h.rescore(cand, q)
		}
		dst = dst[:base]
		for i := 0; i < len(cand) && i < k; i++ {
			// No allocation re-check. Not because a traversal cannot reach a freed
			// slot — it can, through a dangling IN-edge that outlived the node —
			// but because every path that frees a slot either prunes the graph or
			// refills the slot without ever unlocking: Reclaim compacts every
			// surviving neighbour list before it nils the nodes, and the same-id
			// upsert reuses the slot inside one critical section. So a slot a
			// reader reaches under the read lock is occupied, and its id is a real
			// user id. Emitting it verbatim is what makes user id 0 returnable.
			//
			// Caveat, tracked separately: an insert that frees a slot and then
			// takes a quota early-return leaves that slot free with its in-edges
			// intact, which breaks the invariant above. The old `id == 0` guard did
			// not cover that either (the stale id is usually non-zero), so this is
			// pre-existing and orthogonal to the id-0 repair.
			dst = append(dst, Result{ID: h.slotID(cand[i].slot), Distance: cand[i].dist})
		}
		if pred == nil || len(dst)-base >= k || ef >= maxEf {
			return dst
		}
		ef *= 2
		if ef > maxEf {
			ef = maxEf
		}
	}
}

// rescore recomputes exact float32 distances for the candidate set and sorts it
// ascending, replacing the quantizer's approximate ordering with the true
// metric ranking. Mutates cand in place (it aliases search scratch). No-op-safe
// for empty input. Must be called under the read lock.
func (h *hnsw) rescore(cand []slotDist, q []float32) {
	dist := h.metricDist()
	for i := range cand {
		cand[i].dist = dist(q, h.arena.Vec(cand[i].slot))
	}
	sort.Slice(cand, func(a, b int) bool { return cand[a].dist < cand[b].dist })
}

// HybridSearch combines a dense KNN lane and a sparse inverted-index lane,
// fusing the two rankings into the top-k. opts.Filter applies to both lanes.
// Degrades gracefully: an empty sparse query is pure dense search; a nil/empty
// dense query is pure sparse. Result.Score carries the fusion score (higher =
// more relevant); Result.Distance carries the dense metric distance (0 for
// sparse-only hits).
func (h *hnsw) HybridSearch(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	denseRes, sparseRes, err := h.buildLanes(dense, sparse, k, opts)
	if err != nil {
		return nil, err
	}

	// Single-lane degradation.
	// q == nil in the original iff len(dense) == 0 (q was only set when len(dense)>0).
	switch {
	case sparse.IsZero():
		if len(denseRes) > k {
			denseRes = denseRes[:k]
		}
		return denseRes, nil
	case len(dense) == 0:
		if len(sparseRes) > k {
			sparseRes = sparseRes[:k]
		}
		return sparseRes, nil
	}

	// Fuse both lanes.
	switch opts.Method {
	case FusionWeighted:
		alpha := opts.Alpha
		if alpha == 0 {
			alpha = 0.5
		}
		return fuseWeighted(denseRes, sparseRes, k, alpha), nil
	case FusionDBSF:
		alpha := opts.Alpha
		if alpha == 0 {
			alpha = 0.5
		}
		return fuseDBSF(denseRes, sparseRes, k, alpha), nil
	}
	return fuseRRF(denseRes, sparseRes, k, opts.RRFK), nil
}

// buildLanes builds the dense and sparse candidate lanes (unfused). Shared by
// HybridSearch (which then fuses) and HybridLanes (the fan-out primitive). It
// does NOT record search latency — each public entry point records its own.
func (h *hnsw) buildLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) (denseRes []Result, sparseRes []Result, err error) {
	if k <= 0 {
		return nil, nil, nil
	}
	if len(dense) > 0 && len(dense) != h.cfg.Dim {
		return nil, nil, ErrDimMismatch
	}
	if err := sparse.Validate(); err != nil {
		return nil, nil, err
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, nil, err
	}

	denseK := opts.DenseK
	if denseK <= 0 {
		denseK = k
		if denseK < 50 {
			denseK = 50
		}
	}
	sparseK := opts.SparseK
	if sparseK <= 0 {
		sparseK = k
		if sparseK < 50 {
			sparseK = 50
		}
	}

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	var q []float32
	if len(dense) > 0 {
		q = dense
		if h.cfg.Metric == Cosine {
			s.qbuf = append(s.qbuf[:0], dense...)
			normalize(s.qbuf)
			q = s.qbuf
		}
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so both
	// HybridSearch and HybridLanes surface the sentinel instead of a silent empty
	// result — see arena.poisoned.
	if h.arena.poisoned.Load() {
		return nil, nil, ErrIndexPoisoned
	}
	h.searchOps.Add(1)

	// Dense lane.
	if q != nil {
		denseRes = h.searchDenseLocked(s, q, denseK, pred, nil)
	}

	// Sparse lane: accumulate scores via the inverted index, gated by the same
	// admit rule (tombstone + TTL + filter) the dense lane uses.
	if !sparse.IsZero() {
		now := uint64(h.now()) // one clock read for the whole sparse admission scan
		admit := func(slot uint32) bool { return h.admits(slot, pred, now) }
		for _, ss := range h.sparseIdx.searchTopK(s, h.arena.Capacity(), sparse, sparseK, admit) {
			// No allocation re-check: searchTopK walks the inverted postings, which
			// only contain slots that were given a sparse vector at insert time, and
			// the admit closure already applied the liveness gate. Emitting the id
			// verbatim is what makes user id 0 returnable.
			sparseRes = append(sparseRes, Result{ID: h.slotID(ss.slot), Score: ss.score})
		}
	}

	return denseRes, sparseRes, nil
}

// HybridLanes builds the dense and sparse candidate lanes WITHOUT fusing them,
// so a cross-partition coordinator can union the lanes and re-fuse globally
// (exact hybrid fan-out). denseRes is sorted ascending by Distance; sparseRes is
// sorted descending by Score. DenseK/SparseK defaults match HybridSearch
// (max(k,50)). opts.Filter applies to both lanes, exactly as HybridSearch.
func (h *hnsw) HybridLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, []Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()
	return h.buildLanes(dense, sparse, k, opts)
}

// slotID looks up the user id for a slot via the arena's reverse map — O(1).
// Only called for live result slots (which passed admits), so the stored id
// is always the current one.
func (h *hnsw) slotID(slot uint32) uint64 {
	return h.arena.ID(slot)
}

// isExpiredAt reports whether the entry at slot has aged past its TTL as of the
// caller-supplied `now` (unix millis). The admission hot loop snapshots the clock
// once per search and passes it here, so a query reads the wall clock once rather
// than per candidate. Must be called with h.mu held (RLock is enough for reads).
func (h *hnsw) isExpiredAt(slot uint32, now uint64) bool {
	exp := h.arena.ExpiresAt(slot)
	return exp != 0 && exp <= now
}

// isExpired is isExpiredAt against a freshly read clock, for the non-hot-loop
// callers (single liveness checks in Insert/Delete/Get/sweep paths). Must be
// called with h.mu held (RLock is enough for reads; sweepOnce calls under write
// lock).
func (h *hnsw) isExpired(slot uint32) bool {
	return h.isExpiredAt(slot, uint64(h.now()))
}

// admitVerdict is the outcome of the admission gate. It separates the two
// rejection reasons because only ONE of them is counted (filterRejects tallies
// predicate rejections, not tombstones/TTL) and because the caller — not the
// gate — decides WHERE that tally lands: the shared atomic for one-off checks,
// a scratch-local counter for the search hot loop.
type admitVerdict uint8

const (
	admitOK       admitVerdict = iota // eligible for the result set
	admitDead                         // tombstoned or expired
	admitFiltered                     // live, but rejected by the search predicate
)

// admitVerdictOf is the admission gate itself: a slot may appear in a search
// result set only if it is not tombstoned, not expired, and (when pred is
// non-nil) its metadata satisfies the predicate. Graph navigation still
// TRAVERSES rejected slots — this only gates membership in the returned set.
// `now` is the per-search clock snapshot (unix millis) threaded from the search
// entry point, so the admission loop performs no per-candidate wall-clock read.
// Must be called under the read lock.
//
// g is the OPTIONAL per-query bitset gate (filter_bitset.go); nil or disabled
// means the historical path, byte for byte. When armed it answers the predicate
// question from the payload index instead of the metadata map — replacing the
// predicate outright when the index is EXACT for this filter, and pre-rejecting
// on a clear bit when it is only a superset.
//
// ORDER IS LOAD-BEARING. The liveness test stays FIRST. A tombstoned or expired
// slot must be charged to admitDead and never to filterRejects, and putting the
// (cheaper) bit test ahead of it would reclassify every dead slot whose bit
// happens to be clear — silently changing Stats().FilterRejects. The gate only
// ever displaces the predicate, never the liveness gate.
func (h *hnsw) admitVerdictOf(slot uint32, g *admitGate, pred Predicate, now uint64) admitVerdict {
	if h.tombstoned[slot] || h.isExpiredAt(slot, now) || h.notYetLinked(slot) {
		return admitDead
	}
	if pred == nil {
		return admitOK
	}
	if g != nil && g.active() {
		if g.columnar() {
			// The column oracle is exact by construction, so it answers both ways:
			// no bitset, no metadata read, one array index per term. A rejection here
			// is still a rejection BY THE FILTER and is tallied as one, exactly as a
			// clear bit is.
			if !g.testCols(slot) {
				return admitFiltered
			}
			return admitOK
		}
		if !g.test(slot) {
			// The index proved this slot cannot satisfy the filter. That is a
			// rejection BY THE FILTER, so it is tallied as one — a faster oracle
			// answering the same question must not change what the stat counts.
			return admitFiltered
		}
		if g.exact {
			return admitOK // index set == match set: no metadata read needed
		}
	}
	if !pred(h.liveMeta(slot, now)) {
		return admitFiltered
	}
	return admitOK
}

// admits is the admission gate for one-off checks, charging any predicate
// rejection straight to the shared counter. Gate-less by construction: the
// one-off callers (the sparse lane, single-slot liveness probes) hold no
// per-query gate.
func (h *hnsw) admits(slot uint32, pred Predicate, now uint64) bool {
	v := h.admitVerdictOf(slot, nil, pred, now)
	if v == admitFiltered {
		h.filterRejects.Add(1)
	}
	return v == admitOK
}

// admitsScratch is admits for the searchLayerCore hot loop: rejections
// accumulate in the caller's per-traversal scratch and are flushed to
// h.filterRejects ONCE when the traversal returns.
//
// The shared atomic increment was per REJECTED CANDIDATE, which on a selective
// filter is most of the traversal — every concurrent query hammering one
// cache line that no query ever reads. The observable counter is unchanged;
// only the moment it becomes visible moves, from mid-traversal to end-of-
// traversal.
//
// It is also the ONE place the per-query bitset gate is consulted, which is why
// the batched and per-pair traversal paths cannot diverge on it: both call the
// single admit closure searchLayer built, so both see the same s.gate and tally
// the same rejections (batched_traversal_test.go asserts that equality).
func (h *hnsw) admitsScratch(s *layerScratch, slot uint32, pred Predicate, now uint64) bool {
	v := h.admitVerdictOf(slot, &s.gate, pred, now)
	if v == admitFiltered {
		s.filterRejects++
	}
	return v == admitOK
}

// admitVerdictWith is admitVerdictOf with an OPTIONAL external metadata provider
// (the named-vector hook). When metaOf == nil it is byte-identical (reads
// h.arena.Metadata(slot)). When metaOf != nil the predicate is evaluated against
// metaOf(h.arena.ID(slot)) instead — the per-point payload lives in an EXTERNAL
// map (the sub-arena carries no metadata). The tombstone/TTL liveness gate is
// unchanged. Must be called under the read lock.
func (h *hnsw) admitVerdictWith(slot uint32, pred Predicate, metaOf func(id uint64) Metadata, now uint64) admitVerdict {
	if h.tombstoned[slot] || h.isExpiredAt(slot, now) || h.notYetLinked(slot) {
		return admitDead
	}
	if pred != nil {
		var meta Metadata
		if metaOf != nil {
			meta = metaOf(h.arena.ID(slot))
		} else {
			meta = h.liveMeta(slot, now)
		}
		if !pred(meta) {
			return admitFiltered
		}
	}
	return admitOK
}

// admitsWith is admitVerdictWith charging predicate rejections to the shared
// counter (the one-off form; see admits).
func (h *hnsw) admitsWith(slot uint32, pred Predicate, metaOf func(id uint64) Metadata, now uint64) bool {
	v := h.admitVerdictWith(slot, pred, metaOf, now)
	if v == admitFiltered {
		h.filterRejects.Add(1)
	}
	return v == admitOK
}

// admitsWithScratch is admitsWith for the searchLayerCore hot loop; see
// admitsScratch for why the counter is scratch-local.
func (h *hnsw) admitsWithScratch(s *layerScratch, slot uint32, pred Predicate, metaOf func(id uint64) Metadata, now uint64) bool {
	v := h.admitVerdictWith(slot, pred, metaOf, now)
	if v == admitFiltered {
		s.filterRejects++
	}
	return v == admitOK
}

// Delete tombstones the entry for id. The graph edges to/from the slot are
// left intact (the canonical HNSW deletion model) so search performance does
// not degrade; the tombstoned slot is filtered out of results. A future
// background reclaimer (Week 5) will rewire neighbors and reclaim the slot
// once the tombstone ratio crosses a threshold.
func (h *hnsw) Delete(id uint64, cas CASCond) (bool, error) {
	return h.deleteBody(id, cas, false, 0)
}

// DeleteAt is Delete whose dead-slot liveness gate is judged against the EXPLICIT
// leader-stamped clock nowMs, so replicas agree on whether the slot is already dead
// (already-expired ⇒ no-op, no new tombstone) versus live (tombstone it) and their
// tombstone sets stay identical (#4 vector TTL determinism).
func (h *hnsw) DeleteAt(id uint64, cas CASCond, nowMs int64) (bool, error) {
	return h.deleteBody(id, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (h *hnsw) deleteBody(id uint64, cas CASCond, stamped bool, nowMs uint64) (bool, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(h.now())
	}
	slot, ok := h.arena.Slot(id)
	if !ok {
		// Absent (current version 0): a CAS expecting 0 is satisfied but there is
		// nothing to remove (no-op); any other expectation is a conflict.
		if err := cas.check(0); err != nil {
			return false, err
		}
		return false, nil
	}
	if h.tombstoned[slot] || h.isExpiredAt(slot, now) {
		// Dead slot reads as absent (version 0) for CAS, mirroring currentVersionLocked.
		if err := cas.check(0); err != nil {
			return false, err
		}
		return false, nil
	}
	if err := cas.check(h.arena.Version(slot)); err != nil {
		return false, err // CAS mismatch: do not delete
	}
	// Clear the slot's deadlines (point + per-key) through the setters so
	// arena.deadlinePoints/deadlineKeys — the TTL sweep's fast-path gate — drop
	// this slot the moment it dies: a tombstoned slot never needs sweeping.
	h.arena.SetExpires(slot, 0)
	h.arena.SetKeyExpires(slot, nil)
	h.tombstoned[slot] = true
	// Live id set shrank (the id is now tombstoned → excluded from the snapshot's
	// live set). Invalidate the scroll snapshot. Single increment under h.mu.
	h.idSetVersion++
	h.bumpData() // id-set change also invalidates the order_by snapshot
	return true, nil
}

// electEntryPoint picks a new entry point after the current one is removed, or
// resets to an empty index (maxLevel = -1) if none remain. Must hold the write
// lock. O(nodes); called only when the current entry node is removed.
//
// For Vamana (single layer: every node is level 0) the "highest-level remaining
// node" scan below would pick the lowest-indexed live slot — an arbitrary
// peripheral node — and greedy search from a boundary node degrades recall. So
// for Vamana we instead recompute the MEDOID (the live point nearest the sample
// mean) via medoidSlot, exactly the entry point buildVamana elects, keeping
// search starting from a central hub. maxLevel stays 0 (or -1 when empty).
// medoidSlot only ever returns a live slot and holds no lock (the caller holds
// the write lock), so it is safe here.
func (h *hnsw) electEntryPoint() {
	if h.vamana {
		if medoid, ok := h.medoidSlot(); ok {
			h.entryPoint = medoid
			h.maxLevel = 0
		} else {
			// No live points remain: same as the empty case below.
			h.entryPoint = 0
			h.maxLevel = -1
		}
		return
	}
	h.entryPoint = 0
	h.maxLevel = -1
	for slot, nd := range h.nodes {
		// Skip a node still inside its placement/link window: it has no edges yet,
		// and electing it isolates the entry point and orphans the whole graph.
		// See node.unlinked.
		if nd == nil || nd.unlinked.Load() {
			continue
		}
		if nd.level > h.maxLevel {
			h.maxLevel = nd.level
			h.entryPoint = uint32(slot) //nolint:gosec // slot index < arena capacity (< 2^32)
		}
	}
}

// Close releases any resources held by the index. For a heap-backed index this
// is a no-op; an mmap-backed index (QuantMmap) syncs and unmaps its float32
// backing file. Idempotent.
func (h *hnsw) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	err := h.arena.Close()
	if gerr := h.closeGraphMmap(); gerr != nil && err == nil {
		err = gerr
	}
	return err
}

// Compile-time assertion that *hnsw satisfies VectorIndex.
var _ VectorIndex = (*hnsw)(nil)
