// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"io"
	"math"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// IVF-Flat index.
//
// An ivf partitions the vector space into nlist Voronoi cells whose centers are
// k-means CENTROIDS (the coarse quantizer, vector/kmeans.go). Each live slot is
// assigned to its nearest centroid; the per-centroid slot lists are the INVERTED
// LISTS. A query probes the nprobe nearest centroids, gathers their lists'
// candidate slots, applies the same admission gate as hnsw (tombstone + TTL +
// filter predicate), exact-reranks the survivors by metric distance, and returns
// the top-k. Until the index is TRAINED (BuildConcurrent runs k-means), search
// is an exact brute force over every live slot, so an ivf is always CORRECT —
// training only ACCELERATES it.
//
// DATA-PLANE PARITY. Everything that is not the graph/search — the arena, the
// payload index, tombstones, per-point and per-key TTL, the filter gate, scroll,
// sweep, reclaim, payload mutations, Get/Exists/Stats — must behave IDENTICALLY
// to hnsw, because a collection's correctness on those ops cannot depend on the
// index kind. The arena-backed methods below are therefore deliberate mirrors of
// the hnsw bodies (same liveness definition, same clone-on-write payload pattern,
// same scroll-snapshot versioning), reusing every package-level free helper they
// can (pickDistDim, keyExpired, cloneMeta/cloneKeyExpires/pruneKeyExpires,
// liveMetaMap, normalize, estimateInsertBytes, newPayloadIndex/newSparseIndex,
// the snapshot read/write primitives, GroupDocuments, fuseRRF/fuseWeighted).
//
// SHARING STRATEGY (documented decision): the FALLBACK from the plan — duplicate
// the arena-backed bodies here rather than extract a shared embedded pointStore.
// Extracting a pointStore would have to rewire ~30 methods across hnsw.go,
// rag.go, ttl.go, reclaim.go, mmr.go, recommend.go, discover.go, and group.go
// (all reach h.arena/h.payloadIdx/h.tombstoned/h.admits/h.liveMeta directly),
// a large, risky refactor of the 1756-line hnsw against the green dense-test
// gate. The plan says to prefer the fallback if extraction risks destabilizing
// hnsw; it does, so ivf is fully self-contained and touches ZERO hnsw code —
// hnsw cannot regress because nothing about it changed.

// ivf is an IVF-Flat index implementing VectorIndex. It owns its own arena and
// the same payload/TTL/tombstone machinery hnsw uses, plus the coarse-quantizer
// state (centroids + inverted lists). The mutex semantics match hnsw: RLock for
// reads/searches, Lock for writes/train/reclaim/sweep.
type ivf struct {
	cfg   Config
	arena *arena

	// IVF state. trained flips on the first BuildConcurrent (or an explicit
	// train); until then search brute-forces. centroids[c] is cell c's center;
	// lists[c] holds the slot ids assigned to cell c (may contain stale entries
	// for tombstoned/expired/reclaimed slots — the admission gate filters them on
	// scan). nlist is len(centroids) once trained; nprobe is the per-query cell
	// fan-out, clamped to [1, nlist].
	centroids [][]float32
	lists     [][]uint32
	nlist     int
	nprobe    int
	trained   bool

	// IVF-PQ state (cfg.IVFPQ). pq is the trained RESIDUAL product quantizer
	// (codebooks over {vec − centroid[cell]}); nil until trainLocked builds it
	// (or for a non-PQ IVF-Flat index, always nil). When set, the per-slot M-byte
	// residual code lives in the arena's codes side-array (written via SetCode),
	// and search scores by ADC instead of the exact arena.Vec scan. slotCell[slot]
	// records the coarse cell each slot's residual was encoded against, so an
	// approximate float vector can be reconstructed (centroid[cell] + pq.reconstruct
	// (code)) once the resident floats are dropped (PQ-only). pqDropped is true once
	// the resident floats have been released (PQ-only after train); IVFRerank keeps
	// them, so it stays false.
	pq        *pq
	slotCell  []uint32
	pqDropped bool

	// pq4 is the nibble-packed LUT16 fast-scan view of ix.pq, non-nil ONLY when the
	// residual codec is 4-bit (cfg.PQNBits==4, ix.pq.nbits==4). It shares ix.pq's
	// trained codebooks (a thin wrapper — see pq4); the encode path packs codes via
	// it ((m+1)/2 bytes/slot) and the gather path scores a probed list with the
	// in-register VPSHUFB fast-scan kernel (fastScanBlockInto). For 8-bit it stays
	// nil and the existing ix.pq.adc scalar path is UNCHANGED. Kept in sync with
	// ix.pq by refreshPQ4Locked at every ix.pq assignment (train + restore).
	pq4 *pq4

	// SOAR multi-assignment state (cfg.SOAR). When SOAR is on, every slot is filed
	// into a SECONDARY inverted list in addition to its primary one, chosen by the
	// orthogonality-amplified residual loss (see assignSOARLocked). cellOf2[slot] is
	// the secondary cell (parallel to slotCell, which holds the PRIMARY cell for
	// IVF-PQ); code2[slot] is the slot's residual PQ code against THAT secondary
	// centroid (IVF-PQ only — nil for IVF-Flat, where the slot id alone joins the
	// second list). soarTrained mirrors `trained` but is only true once the SOAR
	// secondary assignment has been built, so search/persist can branch on it
	// independently of the SOAR config flag (an index restored from a pre-SOAR
	// snapshot is trained but not soarTrained). cellOf2/code2 are sized to the arena
	// capacity at train and grown by the incremental insert path, exactly like
	// slotCell. All guarded by ix.mu. nil/false ⇒ the pre-SOAR single-assignment
	// path, byte-identical.
	cellOf2     []uint32
	code2       [][]byte
	soarTrained bool

	// Drift-retrain checkpoint state (cfg.IVFDriftRetrain). lastTrainCount is the
	// live count AT the most recent (re)train OR stage-2 DEFER — the high-water mark
	// the O(1) growth gate measures the next checkpoint against. lastTrainCost is the
	// mean nearest-centroid distance of the sample the CURRENT centroids were trained
	// on (set ONLY at a real (re)train, NOT on a defer — the centroids are unchanged
	// on a defer so the train-time reference stays valid). Both are a PURE function of
	// applied (Raft-replicated) state and are PERSISTED (writeIVFCore/readIVFCore),
	// so every replica evaluates the drift trigger at the identical applied insert and
	// an old snapshot lacking them restores 0 (drift simply won't fire until the next
	// train sets them). Guarded by ix.mu. Default-OFF (IVFDriftRetrain=false) never
	// reads them — the drift arm in insertLocked is never entered.
	lastTrainCount int
	lastTrainCost  float32

	mu         sync.RWMutex
	tombstoned map[uint32]bool

	// idSetVersion + scrollSnap mirror hnsw's deterministic-scroll machinery
	// exactly (bumped on every live-id-set change; the cached sorted snapshot is
	// rebuilt when stale). See hnsw for the full contract.
	idSetVersion uint64
	scrollSnap   struct {
		ver uint64
		ids []uint64
	}
	scrollRebuilds uint64

	// dataVersion + orderSnaps mirror hnsw's order_by snapshot machinery exactly:
	// dataVersion bumps on id-set AND payload mutations (separate from idSetVersion so
	// the id scrollSnap is unaffected by payload writes); orderSnaps is the bounded,
	// per-(field, direction) sorted-snapshot cache. See hnsw / order.go for the full
	// contract. Guarded by ix.mu.
	dataVersion   uint64
	orderSnaps    map[orderCacheKey]*orderSnap
	orderSeq      uint64
	orderRebuilds uint64

	searchOps     atomic.Uint64
	insertOps     atomic.Uint64
	expiredCount  atomic.Uint64
	keysSwept     atomic.Uint64
	quotaRejects  atomic.Uint64
	filterRejects atomic.Uint64

	now       func() int64
	bucket    *tokenBucket
	bytesUsed int64

	searchLat latencyHistogram
	insertLat latencyHistogram

	sparseIdx  *sparseIndex
	payloadIdx *payloadIndex
}

// defaultNprobe is the per-query cell fan-out used when Config does not specify
// one. A small fixed default keeps recall reasonable on the modest nlist values
// v1 trains.
const defaultNprobe = 8

// defaultIVFTrainThreshold is the live-vector count at which an INCREMENTALLY
// built IVF index auto-trains (when Config.IVFTrainThreshold == 0). It is the
// crossover where IVF's coarse pruning (and IVF-PQ's compression) starts paying
// off versus an exact brute-force scan:
//
//   - It is large enough that tiny collections and the bulk of the package's
//     unit tests (which insert tens to low-hundreds of vectors) stay UNTRAINED
//     and therefore exact-brute-force — so existing behaviour is unchanged and a
//     small IVF index is still bit-exact.
//   - It is small enough to engage on real data: at 2048 live vectors the default
//     heuristic nlist = max(1, 4*sqrt(N)) ≈ 180 cells averages ~11 points/cell,
//     a sane coarse quantizer, and the PQ codebooks have ample (2048) residual
//     training samples for 256 sub-centroids/subspace.
//
// As a create-time Config-derived constant it is identical across all replicas,
// so the auto-train TRIGGER fires at the same applied insert on every replica.
const defaultIVFTrainThreshold = 2048

// defaultIVFDriftGrowthFactor / defaultIVFDriftFactor are the resolved-when-zero
// values for cfg.IVFDriftGrowthFactor / cfg.IVFDriftFactor (drift-retrain). The
// growth factor (2.0) is the geometric live-count multiple between O(1) stage-1
// checkpoints; the drift factor (1.5) is how much the mean nearest-centroid
// distance must exceed its train-time reference before stage 2 triggers a retrain.
// Both are create-time Config-derived constants (identical across replicas), so a
// drift retrain fires at the same applied insert on every replica. Validate rejects
// an explicit value <= 1.0 at create; 0 resolves here.
const (
	defaultIVFDriftGrowthFactor = 2.0
	defaultIVFDriftFactor       = 1.5
)

// newIVF constructs an empty IVF-Flat index. Returns ErrInvalid* if cfg is
// malformed (same validation as hnsw). The index starts UNTRAINED: inserts land
// in the arena and search brute-forces until BuildConcurrent trains the coarse
// quantizer.
func newIVF(cfg Config) (*ivf, error) {
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}
	a := newArena(cfg.Dim, 0)
	// A declared vector cap sizes the slab's address-space reservation (a hint,
	// never a limit — see slabReserveSize).
	a.maxVectorsHint = cfg.MaxVectors
	if cfg.IVFPQ {
		// IVF-PQ: size the codes side-array to the residual-code length/slot WITHOUT a
		// quantizer (Insert must not auto-encode the raw vec — the IVF encodes the
		// residual itself and writes it via SetCode). The configured Quant is
		// ignored in PQ mode. The code length follows the codec width (m bytes for
		// 8-bit, (m+1)/2 for the 4-bit LUT16 codec) — see ivfPQCodeLen.
		a.enableCodes(ivfPQCodeLen(cfg))
	} else {
		quant := newQuantizer(cfg.Quant, cfg.Dim, cfg.QuantPQM, cfg.SQBits, cfg.PRQLayers, cfg.PQNBits, cfg.Metric)
		if quant != nil {
			a.setQuant(quant)
		}
	}
	// A Persistent IVF that RETAINS its float vectors backs them with the cfg.MmapPath
	// mmap file so SavePersist can externalize them into the instant-restart sidecar
	// (the floats stay in the file across restart, re-mapped zero-copy by
	// openPersistIVF). IVF-PQ-only (IVFPQ && !IVFRerank) DROPS the resident floats at
	// train (arena.dropVecs, heap-only), so it must stay heap — the sidecar then
	// carries the residual codes VERBATIM (no vecs file). IVF-Flat and IVFRerank keep
	// the floats and use the mmap. Non-persistent IVF (incl. cluster, which forces
	// Persistent=false) stays heap — the snapshot path is untouched.
	retainsFloats := !cfg.IVFPQ || cfg.IVFRerank
	if cfg.Persistent && cfg.MmapPath != "" && retainsFloats {
		if err := a.useMmap(cfg.MmapPath); err != nil {
			return nil, err
		}
	}
	nprobe := cfg.IVFNprobe
	if nprobe <= 0 {
		nprobe = defaultNprobe
	}
	return &ivf{
		cfg:          cfg,
		arena:        a,
		nprobe:       nprobe,
		tombstoned:   make(map[uint32]bool),
		now:          func() int64 { return time.Now().UnixMilli() },
		bucket:       newTokenBucket(cfg.MaxInsertsPerSecond),
		sparseIdx:    newSparseIndex(),
		payloadIdx:   newPayloadIndex(),
		idSetVersion: 1,
		dataVersion:  1,
		orderSnaps:   make(map[orderCacheKey]*orderSnap),
	}, nil
}

// openIVF reopens an IVF-Flat index. v1 IVF is snapshot-only (no mmap sidecar),
// so there is nothing to map: it returns a fresh empty index that the caller
// then Restores from a snapshot, mirroring how a non-persistent hnsw collection
// reopens via NewCollection+Restore. metaPath is accepted for signature symmetry
// with openIndex and ignored.
func openIVF(cfg Config, _ string) (*ivf, error) {
	return newIVF(cfg)
}

// ---------------------------------------------------------------------------
// liveness / metric helpers (mirror hnsw)
// ---------------------------------------------------------------------------

func (ix *ivf) metricDist() distFunc { return pickDistDim(ix.cfg.Metric, ix.cfg.Dim) }

// ivfPQM resolves the PQ sub-quantizer count for cfg: the configured IVFPQM, or
// defaultPQM(Dim) when 0. Validate has already ensured Dim % IVFPQM == 0.
func ivfPQM(cfg Config) int {
	if cfg.IVFPQM > 0 {
		return cfg.IVFPQM
	}
	return defaultPQM(cfg.Dim)
}

// ivfPQNBits resolves the residual-PQ code width for an IVF-PQ config: 4 (the
// LUT16 fast-scan codec, 16 sub-centroids/subspace, nibble-packed codes) when
// cfg.PQNBits==4, otherwise 8 (the byte-per-subspace default — 0 and 8 both map
// here, byte-identical). Validate has already rejected any other value.
func ivfPQNBits(cfg Config) int {
	if cfg.PQNBits == 4 {
		return 4
	}
	return 8
}

// ivfPQCodeLen resolves the per-slot residual-code length for an IVF-PQ config.
// The 8-bit codec stores one byte per subspace (m bytes); the 4-bit LUT16 codec
// packs two sub-codes per byte ((m+1)/2 bytes — see pq4CodeLen). This drives the
// arena codes side-array width (enableCodes) so the per-slot code path is the
// SAME for both widths — no persistence-format change, only the byte count.
func ivfPQCodeLen(cfg Config) int {
	m := ivfPQM(cfg)
	if ivfPQNBits(cfg) == 4 {
		return pq4CodeLen(m)
	}
	return m
}

// ivfRerankFactor is the ADC-shortlist over-collection multiple for IVFRerank:
// the exact rescore re-ranks rerankFactor*k candidates. Reuses the quantization
// default (3).
const ivfRerankFactor = defaultRescoreFactor

// pqActive reports whether the residual PQ codec is trained and in use (ADC
// search path). False for IVF-Flat and for an untrained IVF-PQ index (which
// brute-forces exact floats until BuildConcurrent trains the codebooks).
func (ix *ivf) pqActive() bool { return ix.pq != nil }

// refreshPQ4Locked re-syncs ix.pq4 with ix.pq: it sets ix.pq4 to a fast-scan
// wrapper of ix.pq when the codec is 4-bit (ix.pq.nbits==4), else nil. Called at
// every ix.pq assignment (train + snapshot/sidecar restore) so the encode/gather
// paths pick the nibble-packed 4-bit codec exactly when the codebooks are 16-wide.
// Caller holds ix.mu.
func (ix *ivf) refreshPQ4Locked() {
	if ix.pq != nil && ix.pq.nbits == 4 {
		ix.pq4 = &pq4{codec: ix.pq}
		return
	}
	ix.pq4 = nil
}

// pq4Active reports whether the gather path uses the 4-bit LUT16 codec (a per-cell
// uint8 lut16 + fastScanBlockInto over the slot list) instead of the 8-bit scalar
// adc. True whenever a trained 4-bit codec is present: the kernel runs the
// in-register VPSHUFB/TBL path when m is overflow-safe for the uint16 lanes, and
// fastScanBlockInto itself falls back to the per-code scalar adcScalar for a
// pathological m > 257 (so the nibble-packed codes are always decoded correctly —
// never misread by the 8-bit ix.pq.adc).
func (ix *ivf) pq4Active() bool {
	return ix.pq4 != nil
}

// vecFor returns slot's float vector for the exact paths (Get/rescore/reshard/
// MMR/recommend). When the resident floats are present it returns arena.Vec
// directly (exact). In PQ-only mode after the floats are dropped it reconstructs
// an APPROXIMATE vector from the residual code + the slot's coarse centroid —
// the documented PQ-only tradeoff (maximum compression; exact floats require
// IVFRerank). The returned slice is freshly allocated when reconstructed and
// aliases arena storage otherwise; callers that retain it must clone.
func (ix *ivf) vecFor(slot uint32) []float32 {
	// Under-lock backstop for the materialization paths (Get/GetProjected/GetInto,
	// discover context pairs): a failed mmap grow nils the arena floats, so
	// arena.Vec would slice a nil region and panic. Return nil instead. Callers
	// that dereference the result element-wise (meanOf, discover pairs) reject at
	// their own top under the lock; this keeps the Get* path panic-safe. See
	// arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil
	}
	if !ix.pqDropped {
		return ix.arena.Vec(slot)
	}
	// 4-bit codes are nibble-packed ((m+1)/2 bytes); the pq4 reconstruct unpacks
	// them. 8-bit uses the byte-per-subspace pq.reconstruct (UNCHANGED).
	out := ix.reconstructResidual(ix.arena.Code(slot))
	if int(slot) < len(ix.slotCell) {
		c := ix.centroids[ix.slotCell[slot]]
		for i := range out {
			out[i] += c[i]
		}
	}
	return out
}

// reconstructResidual reconstructs the residual float vector from a per-slot code,
// dispatching on the codec width: the nibble-packed 4-bit code via pq4.reconstruct
// (which unpacks two sub-codes per byte), or the byte-per-subspace 8-bit code via
// pq.reconstruct. Both un-rotate (Rᵀ) when OPQ is on. Caller holds ix.mu.
func (ix *ivf) reconstructResidual(code []byte) []float32 {
	if ix.pq4 != nil {
		return ix.pq4.reconstruct(code)
	}
	return ix.pq.reconstruct(code)
}

// isExpiredAt reports whether slot has aged past its point TTL as of the
// caller-supplied `now` (unix millis). The admission hot loops (gather*) snapshot
// the clock once per query and pass it here, so a query reads the wall clock once
// rather than per probed candidate. Must hold ix.mu.
func (ix *ivf) isExpiredAt(slot uint32, now uint64) bool {
	exp := ix.arena.ExpiresAt(slot)
	return exp != 0 && exp <= now
}

// isExpired is isExpiredAt against a freshly read clock, for the non-hot-loop
// callers (single liveness checks in Insert/Delete/Get/sweep paths). Must hold
// ix.mu.
func (ix *ivf) isExpired(slot uint32) bool {
	return ix.isExpiredAt(slot, uint64(ix.now()))
}

// SetNowFunc overrides the wall-clock source (unix millis) the index's non-apply
// expiry sites consult (sweeper + client read/query filter + the wall-clock branch
// of the write paths). nil restores the real clock. TEST/advanced seam mirroring
// cache.Cache.SetNowFunc; production never calls it (byte-identical default). Does
// NOT affect the stamped apply path (InsertAt takes an explicit stamp). Takes the
// write lock.
func (ix *ivf) SetNowFunc(fn func() int64) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if fn == nil {
		ix.now = func() int64 { return time.Now().UnixMilli() }
		return
	}
	ix.now = fn
}

// liveMeta returns the slot's metadata with per-key-TTL-expired keys dropped —
// the view every read path must see (mirror hnsw.liveMeta). Must hold ix.mu.
func (ix *ivf) liveMeta(slot uint32, now uint64) Metadata {
	m := ix.arena.Metadata(slot)
	ke := ix.arena.KeyExpires(slot)
	if len(ke) == 0 {
		return m
	}
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
		return m
	}
	out := make(Metadata, len(m))
	for k, v := range m {
		if keyExpired(ke[k], now) {
			continue
		}
		out[k] = v
	}
	return out
}

// admits is the search/scan membership gate: not tombstoned, not expired, and
// (when pred != nil) the live metadata satisfies the predicate. `now` is the
// per-query clock snapshot (unix millis) threaded from the gather entry point, so
// the probe loop performs no per-candidate wall-clock read. Must hold ix.mu.
func (ix *ivf) admits(slot uint32, pred Predicate, now uint64) bool {
	if ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return false
	}
	if pred != nil && !pred(ix.liveMeta(slot, now)) {
		ix.filterRejects.Add(1)
		return false
	}
	return true
}

func (ix *ivf) slotID(slot uint32) uint64 { return ix.arena.ID(slot) }

// liveLocked reports whether id maps to a live slot. Must hold ix.mu.
func (ix *ivf) liveLocked(id uint64) bool {
	return ix.liveLockedAt(id, uint64(ix.now()))
}

// liveLockedAt is liveLocked judging expiry against the caller-supplied `now`
// (unix millis), so the insert-if-absent outcome is decided on the leader-stamped
// clock identically on every replica (#4 vector TTL determinism). Must hold ix.mu.
func (ix *ivf) liveLockedAt(id uint64, now uint64) bool {
	slot, ok := ix.arena.Slot(id)
	if !ok {
		return false
	}
	return !ix.tombstoned[slot] && !ix.isExpiredAt(slot, now)
}

// ---------------------------------------------------------------------------
// writes
// ---------------------------------------------------------------------------

// Insert adds vec under id with optional TTL/metadata/sparse, mirroring
// hnsw.Insert. When the index is trained the new slot is incrementally assigned
// to its nearest centroid's inverted list; untrained, it just lives in the arena
// (the next search brute-forces, the next train sweeps it into a list).
func (ix *ivf) Insert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (uint64, map[string]uint64, error) {
	return ix.insertBody(id, vec, ttl, meta, sparse, keyTTLMs, cas, false, 0)
}

// InsertAt is Insert with every TTL deadline computation and liveness check judged
// against the EXPLICIT leader-stamped clock nowMs (unix millis), so a replicated
// apply stamps byte-identical state on every replica (#4 vector TTL determinism,
// mirroring hnsw.InsertAt / cache.PutAt). Insert is byte-identical to before.
func (ix *ivf) InsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (uint64, map[string]uint64, error) {
	return ix.insertBody(id, vec, ttl, meta, sparse, keyTTLMs, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) insertBody(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (uint64, map[string]uint64, error) {
	if err := checkRecordValues(meta); err != nil {
		return 0, nil, err
	}
	start := time.Now()
	defer func() { ix.insertLat.observe(time.Since(start)) }()

	if len(vec) != ix.cfg.Dim {
		return 0, nil, ErrDimMismatch
	}
	if !ix.bucket.Take() {
		ix.quotaRejects.Add(1)
		return 0, nil, ErrCollectionRateLimited
	}
	stored := vec
	if ix.cfg.Metric == Cosine {
		stored = append([]float32(nil), vec...)
		normalize(stored)
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	if err := cas.check(ix.currentVersionLockedAt(id, now)); err != nil {
		return 0, nil, err
	}
	keyExpires := computeInsertKeyExpires(int64(now), meta, keyTTLMs) //nolint:gosec // unix-millis fits int64
	version, err := ix.insertLockedAt(id, stored, ttl, meta, sparse, keyExpires, 0, now)
	if err != nil {
		return 0, nil, err
	}
	return version, keyExpires, nil
}

// RestoreInsert inserts id with an EXACT version verbatim (WAL replay / reshard
// copy). Mirrors hnsw.RestoreInsert. keyExpires is the ABSOLUTE per-key deadline
// map restored VERBATIM (NOT recomputed).
func (ix *ivf) RestoreInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error {
	return ix.restoreInsertBody(id, vec, ttl, meta, sparse, keyExpires, version, false, 0)
}

// RestoreInsertAt is RestoreInsert stamping the POINT ttl deadline and judging
// reclaim liveness against the EXPLICIT leader-stamped clock nowMs (keyExpires is
// still installed VERBATIM). Replicated version-preserving insert path (#4 vector
// TTL determinism).
func (ix *ivf) RestoreInsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) error {
	return ix.restoreInsertBody(id, vec, ttl, meta, sparse, keyExpires, version, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) restoreInsertBody(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, stamped bool, nowMs uint64) error {
	// REPLAY body: size only — the IVF twin of hnsw.restoreInsertBody. See
	// checkRecordValuesSize.
	if err := checkRecordValuesSize(meta); err != nil {
		return err
	}
	if len(vec) != ix.cfg.Dim {
		return ErrDimMismatch
	}
	stored := vec
	if ix.cfg.Metric == Cosine {
		stored = append([]float32(nil), vec...)
		normalize(stored)
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	_, err := ix.insertLockedAt(id, stored, ttl, meta, sparse, keyExpires, version, now)
	return err
}

// currentVersionLockedAt is currentVersionLocked judging liveness against the
// caller-supplied `now` (unix millis), so a replicated apply's CAS decision uses
// the leader-stamped clock identically on every replica (#4 vector TTL
// determinism). Must hold ix.mu.
func (ix *ivf) currentVersionLockedAt(id uint64, now uint64) uint64 {
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return 0
	}
	return ix.arena.Version(slot)
}

// insertLockedAt is insertLocked with the dead-slot reclaim liveness check and the
// point-TTL absolute deadline (now+ttl) judged against the caller-supplied `now`
// (unix millis), so a replicated insert stamps byte-identical state on every
// replica (#4 vector TTL determinism). Must hold ix.mu.
func (ix *ivf) insertLockedAt(id uint64, stored []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, setVersion uint64, now uint64) (uint64, error) {
	// Upsert replace: a dead (tombstoned or expired) slot for id is hard-removed
	// so this insert resurrects the id. The stale list entry for the freed slot is
	// tolerated — it is filtered by admits on the next scan, and slot reuse below
	// re-files the slot under its new centroid.
	//
	// FREE AND REUSE MUST BE ATOMIC, so the decision is taken here and the reclaim
	// itself runs past every check that can reject. "Filtered by admits" holds
	// only while the slot is dead, and the reclaim is what makes it stop being
	// dead: it un-tombstones the slot and arena.Delete clears its expiry, while
	// the stale list entry, id, vector and metadata all stay put. If anything
	// between the reclaim and arena.Insert returned, the next scan would find that
	// list entry, pass admits(), and emit the DELETED point with its old vector
	// and old payload. Both quota checks used to sit in that gap and a collection
	// whose arena is over quota (a bulk build does not consult the quota — see
	// BuildConcurrentMeta) reaches them. Mirrors hnsw.placeLockedAt, where the
	// same reordering is documented at length.
	//
	// Judging each check against the accounting the reclaim is ABOUT to produce
	// keeps the verdict identical to running it afterwards — in particular an
	// upsert into an exactly-full collection is a replace and still succeeds.
	reclaimSlot, reclaiming := uint32(0), false
	if old, ok := ix.arena.Slot(id); ok && (ix.tombstoned[old] || ix.isExpiredAt(old, now)) {
		reclaimSlot, reclaiming = old, true
	}

	liveAfterReclaim := int64(ix.arena.Size())
	if reclaiming {
		liveAfterReclaim-- // the reclaim releases this id's slot before the insert retakes it
	}
	if ix.cfg.MaxVectors > 0 && liveAfterReclaim >= ix.cfg.MaxVectors {
		ix.quotaRejects.Add(1)
		return 0, ErrCollectionFull
	}
	insertBytes := estimateInsertBytes(ix.cfg.Dim, ix.cfg.M)
	bytesAfterReclaim := ix.bytesUsed
	if reclaiming {
		bytesAfterReclaim -= insertBytes
		if bytesAfterReclaim < 0 {
			bytesAfterReclaim = 0
		}
	}
	if ix.cfg.MaxBytes > 0 && bytesAfterReclaim+insertBytes > ix.cfg.MaxBytes {
		ix.quotaRejects.Add(1)
		return 0, ErrCollectionFull
	}

	// PAST EVERY REJECTION: free the dead slot and let arena.Insert directly below
	// take it straight back. Nothing between the two may return.
	if reclaiming {
		old := reclaimSlot
		delete(ix.tombstoned, old)
		ix.payloadIdx.reindex(old, nil)
		if ix.sparseIdx != nil {
			if sv := ix.arena.Sparse(old); sv != nil {
				ix.sparseIdx.remove(old, *sv)
			}
		}
		ix.bytesUsed = bytesAfterReclaim // the decrement the byte quota was judged on
		ix.arena.Delete(id)
	}

	slot, err := ix.arena.Insert(id, stored)
	if err != nil {
		return 0, err
	}
	version := uint64(1)
	if setVersion != 0 {
		version = setVersion
	}
	ix.arena.SetVersion(slot, version)
	ix.idSetVersion++
	ix.bumpData() // id-set change also invalidates the order_by snapshot
	if ttl > 0 {
		deadline := now + uint64(ttl.Milliseconds())
		ix.arena.SetExpires(slot, deadline)
	}
	if meta != nil {
		ix.arena.SetMetadata(slot, meta)
	}
	// Per-key payload TTL: store the ABSOLUTE deadline map (computed by the caller),
	// pruned to present keys. Empty/nil keeps the slot cleared (zero-overhead).
	if len(keyExpires) > 0 {
		ix.arena.SetKeyExpires(slot, pruneKeyExpires(cloneKeyExpires(keyExpires), meta))
	}
	ix.payloadIdx.reindex(slot, meta)
	if sparse != nil && !sparse.IsZero() {
		ix.arena.SetSparse(slot, sparse)
		ix.sparseIdx.add(slot, *sparse)
	}
	ix.bytesUsed += insertBytes
	ix.insertOps.Add(1)

	// Incremental list assignment: only when trained. A reused slot may already
	// appear in some list from its prior life; assignToList appends it to its new
	// nearest cell, and the stale old entry is filtered on scan (admits). Lists
	// are rebuilt wholesale on the next train, so transient duplicates never
	// accumulate.
	if ix.trained {
		ix.assignToList(slot, stored)
		// DETERMINISTIC AUTO-RETRAIN-ON-DRIFT (opt-in, cfg.IVFDriftRetrain). After
		// assigning the new slot to a list, check whether the live distribution has
		// drifted far enough from the CURRENT centroids to warrant a fresh train. Like
		// the auto-train trigger above, driftRetrainCheckLocked is a PURE function of
		// applied state (the live count + the tombstone-only slot-ordered drift sample's
		// mean nearest-centroid distance) + the replicated Config, so every replica fires
		// (or defers) at the IDENTICAL applied insert over the IDENTICAL sample with the
		// IDENTICAL Seed ⇒ bit-identical centroids/codebooks. The whole arm is gated by
		// cfg.IVFDriftRetrain inside driftRetrainCheckLocked, so the DEFAULT (off) takes
		// no new branch and the existing trained-insert path stays byte-identical.
		// NOTE: driftRetrainCheckLocked mutates lastTrainCount on the DEFER path (see its
		// doc comment) — it requires the write lock, which insertLocked already holds.
		if ix.driftRetrainCheckLocked() {
			ix.autoTrainLocked()
		}
	} else if ix.shouldAutoTrain() {
		// DETERMINISTIC AUTO-TRAIN. The live count just crossed the threshold during
		// THIS applied insert, so train the coarse quantizer (and IVF-PQ residual
		// codebooks) NOW — synchronously, under ix.mu, before this mutator returns.
		// Because every replica applies the identical insert sequence in the identical
		// order and this trigger is a pure function of applied state + the replicated
		// Config threshold, every replica trains at the IDENTICAL insert over the
		// IDENTICAL (slot-ordered) sample with the IDENTICAL Seed ⇒ bit-identical
		// centroids and codebooks. NEVER spawn a goroutine here: the trained state must
		// be committed before the next applied entry. Trains ONCE — after this the index
		// is `trained`, so subsequent inserts assign to existing cells (no auto-retrain).
		ix.autoTrainLocked()
	}
	return version, nil
}

// shouldAutoTrain reports whether an UNTRAINED incremental IVF index has now
// accumulated enough live vectors to deterministically auto-train. It is a pure
// function of applied state (the live count) and the replicated threshold, so it
// evaluates identically on every replica. Must hold ix.mu; the caller guarantees
// !ix.trained.
func (ix *ivf) shouldAutoTrain() bool {
	threshold := ix.cfg.IVFTrainThreshold
	if threshold <= 0 {
		threshold = defaultIVFTrainThreshold
	}
	return ix.liveCount() >= threshold
}

// meanAssignCost is the deterministic drift metric: the mean distance of each
// sample vector to its nearest centroid, under the SAME metric kernel kmeans uses
// for assignment (ix.metricDist() == pickDistDim(metric, dim), which falls back to
// pickDist(metric) — the exact distFunc kmeans's nearestCentroidIdx applies). The
// sample is ALREADY slot-ordered (driftSampleLocked / the trainLocked sample), so
// the float sum runs in a FIXED, CANONICAL FORWARD ORDER (index 0 → len-1) and
// yields a bit-identical float32 for the same (sample, centroids) on every replica —
// NO Go-map iteration, NO wall-clock, NO per-replica input.
//
// ITERATION ORDER IS LOAD-BEARING FOR REPLICA DETERMINISM: reversing the loop or
// using a parallel/unordered reduction changes the floating-point result due to
// non-associativity of float32 addition. Do not change the loop order.
//
// Returns 0 for an empty sample or empty centroids (the caller's stage-1 gate
// guarantees both are non-empty when this drives a decision). Must hold ix.mu;
// the caller supplies a sample gathered under the lock.
func (ix *ivf) meanAssignCost(sample [][]float32, centroids [][]float32) float32 {
	if len(sample) == 0 || len(centroids) == 0 {
		return 0
	}
	dist := ix.metricDist()
	var sum float32
	// Forward iteration only — the order is load-bearing (see doc comment above).
	for i := 0; i < len(sample); i++ {
		v := sample[i]
		best := dist(v, centroids[0])
		for c := 1; c < len(centroids); c++ {
			if d := dist(v, centroids[c]); d < best {
				best = d
			}
		}
		sum += best
	}
	return sum / float32(len(sample))
}

// driftRetrainCheckLocked reports whether a TRAINED IVF index should retrain because
// the live distribution has DRIFTED away from the current centroids. It is a
// two-stage, PURE-function-of-applied-state trigger (no wall-clock, no background
// goroutine, no per-replica memory/load, no Go-map iteration):
//
//	stage 1 (O(1) growth checkpoint): cfg.IVFDriftRetrain on AND trained AND
//	  lastTrainCount>0 AND liveCount() >= lastTrainCount*growthFactor. This fires
//	  only at geometric live-count checkpoints, so the expensive stage 2 is rare.
//	stage 2 (drift metric, only when stage 1 passes): meanAssignCost over the
//	  TOMBSTONE-ONLY drift sample (driftSampleLocked — wall-clock TTL excluded for
//	  replica determinism) vs the CURRENT centroids. If it exceeds
//	  lastTrainCost*driftFactor the centroids no longer fit → return true (the
//	  caller retrains via autoTrainLocked → trainLocked, which RESETS
//	  lastTrainCount/lastTrainCost). Otherwise the data merely grew without drifting:
//	  DEFER — bump lastTrainCount to the current live count (push the next checkpoint
//	  geometrically) and LEAVE lastTrainCost (the centroids, and thus their train-time
//	  reference, are unchanged), then return false.
//
// MUTATES STATE on the DEFER path (bumps lastTrainCount) — requires the write lock
// (insertLocked already holds it). Default-OFF (cfg.IVFDriftRetrain=false) returns
// false on the first clause without touching any state — the insertLocked drift arm
// is never entered, so the trained path stays byte-identical.
//
// DETERMINISM CONTRACT: liveCount() = arena.Size() (not TTL-aware), and
// driftSampleLocked uses tombstone-only membership — both are pure functions of
// Raft-applied state (inserts + hard-deletes). Point-TTL expiry is wall-clock
// dependent and is explicitly excluded from this path; see driftSampleLocked.
func (ix *ivf) driftRetrainCheckLocked() bool {
	if !ix.cfg.IVFDriftRetrain || !ix.trained || ix.lastTrainCount <= 0 {
		return false
	}
	growthFactor := ix.cfg.IVFDriftGrowthFactor
	if growthFactor == 0 {
		growthFactor = defaultIVFDriftGrowthFactor
	}
	if ix.liveCount() < int(float64(ix.lastTrainCount)*growthFactor) {
		return false
	}
	driftFactor := ix.cfg.IVFDriftFactor
	if driftFactor == 0 {
		driftFactor = defaultIVFDriftFactor
	}
	// Use the tombstone-only sample (no wall-clock TTL) so the drift metric is a
	// pure function of applied state — identical on every replica at the same offset.
	cost := ix.meanAssignCost(ix.driftSampleLocked(), ix.centroids)
	if cost > ix.lastTrainCost*float32(driftFactor) {
		return true
	}
	// DEFER: grew but did not drift. Push the checkpoint forward; keep lastTrainCost
	// (centroids unchanged ⇒ its train-time reference is still valid).
	ix.lastTrainCount = ix.liveCount()
	return false
}

// liveCount returns the number of arena-resident vectors (arena.Size() = len(idMap)).
// A hard-tombstoned slot is removed from idMap on hard-delete/reclaim; soft-deleted
// (tombstoned[slot]=true) slots remain in idMap until reclaim. Wall-clock-expired
// points are NOT excluded — liveCount is TTL-blind, matching liveSampleLocked's
// tombstone-only membership so both the stage-1 count gate and the sample use the
// same population basis. Must hold ix.mu.
func (ix *ivf) liveCount() int {
	return ix.arena.Size()
}

// autoTrainLocked deterministically trains the incrementally-built index in place:
// it gathers the live vectors as a SLOT-ORDERED sample (deterministic — never Go-map
// iteration order) and runs the EXACT same train+encode sequence BuildConcurrent
// uses (trainLocked → kmeans coarse + trainResidualPQLocked if IVF-PQ +
// rebuildListsLocked/encode-all + dropVecs if PQ-only). Must hold ix.mu (write) and
// be called only when !ix.trained. Uses the same default worker count BuildConcurrent
// uses; k-means is worker-count-invariant (parallel-kmeans: disjoint writes +
// index-ordered serial reduce), so the result is identical regardless of workers.
func (ix *ivf) autoTrainLocked() {
	sample := ix.liveSampleLocked()
	if len(sample) == 0 {
		return
	}
	ix.trainLocked(sample, runtime.GOMAXPROCS(0))
}

// liveSampleLocked collects COPIES of every tombstone-free arena slot in ASCENDING
// SLOT ORDER — the deterministic order in which the vectors were inserted (arena
// slots are allocated sequentially; reuse only happens after a hard-delete/reclaim,
// which does not occur on the pre-train incremental path under test, and even then
// slot order remains a total deterministic order identical across replicas applying
// the same op sequence). This mirrors BuildConcurrent's index-ordered `train` slice,
// so kmeans/PQ see the same sample order on every replica.
//
// TOMBSTONE-ONLY MEMBERSHIP (intentionally TTL-blind): point-TTL expiry is
// wall-clock dependent — two replicas evaluating isExpired(slot) at slightly
// different physical times may disagree on whether a TTL'd-but-not-yet-reclaimed
// point is "live", producing different training samples and therefore different
// centroids (a replica-determinism violation). Hard tombstones ARE applied state
// (they are Raft-replicated Delete ops), so the tombstone map is bit-identical on
// every replica at the same applied offset. A wall-clock-expired-but-not-tombstoned
// point IS still applied state (the reclaim op has not been applied yet) — including
// it in the training sample is deterministic and transient.
//
// CONSISTENCY WITH liveCount(): liveCount() = arena.Size() is ALSO TTL-blind (it
// counts all arena slots regardless of TTL), so both the stage-1 count gate and the
// sample now use the same tombstone-only membership basis.
//
// The stored floats are already normalized for cosine (Insert normalizes before
// insertLocked). Must hold ix.mu.
func (ix *ivf) liveSampleLocked() [][]float32 {
	capacity := ix.arena.Capacity()
	sample := make([][]float32, 0, ix.arena.Size())
	for s := 0; s < capacity; s++ {
		slot := uint32(s) //nolint:gosec // s < Capacity() < 2^32
		if !ix.arenaSlotActiveLocked(slot) {
			continue
		}
		sample = append(sample, append([]float32(nil), ix.arena.Vec(slot)...))
	}
	return sample
}

// driftSampleLocked is an alias for liveSampleLocked — both the first-train path
// (autoTrainLocked) and the drift path (driftRetrainCheckLocked) now use the same
// tombstone-only sample. Kept as a named call site so call-site comments remain
// accurate; inlined by the compiler.
func (ix *ivf) driftSampleLocked() [][]float32 { return ix.liveSampleLocked() }

// assignToList appends slot to its nearest centroid's inverted list. Must hold
// ix.mu (write) and be called only when trained. In IVF-PQ mode it also encodes
// the slot's RESIDUAL (vec − centroid[cell]) into the arena codes side-array and
// records the cell so the float can later be reconstructed; PQ-only then drops
// the resident float for this slot is handled at train time (incremental inserts
// after a PQ-only train keep their float until... no — see encodeResidual).
func (ix *ivf) assignToList(slot uint32, vec []float32) {
	c := ix.nearestCentroid(vec)
	ix.lists[c] = append(ix.lists[c], slot)
	if ix.pqActive() {
		ix.encodeResidualLocked(slot, uint32(c), vec) //nolint:gosec // c < nlist < 2^32
	}
	// SOAR secondary assignment: file the slot into a complementary second list so a
	// query reaches it via either cell (recall rises at fixed nprobe). Only when the
	// SOAR multi-assignment has been built (soarTrained) — a non-SOAR index never
	// enters this branch, so its assignment is byte-identical.
	if ix.soarTrained {
		ix.assignSecondaryLocked(slot, c, vec)
	}
}

// encodeResidualLocked encodes vec's residual against centroid[cell] into slot's
// PQ code (arena codes side-array) and records slot→cell. Must hold ix.mu and
// have ix.pq trained. The residual is vec − centroid[cell]; the codec is
// centroid-agnostic (it just encodes the residual it is handed). Used by the
// build pass and incremental assignToList.
func (ix *ivf) encodeResidualLocked(slot, cell uint32, vec []float32) {
	cen := ix.centroids[cell]
	res := make([]float32, ix.cfg.Dim)
	for i := range res {
		res[i] = vec[i] - cen[i]
	}
	ix.arena.SetCode(slot, ix.encodeResidualCode(res))
	if int(slot) >= len(ix.slotCell) {
		// Grow with amortized doubling, not to exact slot+1: incremental inserts
		// add slots monotonically, so an exact regrow+copy on every new slot is
		// O(n) per insert → O(n²) total. Doubling capacity keeps it amortized O(1).
		// Re-slicing within existing capacity is safe because gap indices in
		// [len, slot) were never written (slots only ever appended), so they stay
		// zero-filled exactly as the old make()+copy left them.
		if int(slot) >= cap(ix.slotCell) {
			newCap := cap(ix.slotCell)
			if newCap == 0 {
				newCap = 1
			}
			for newCap <= int(slot) {
				newCap *= 2
			}
			grown := make([]uint32, int(slot)+1, newCap)
			copy(grown, ix.slotCell)
			ix.slotCell = grown
		} else {
			ix.slotCell = ix.slotCell[:int(slot)+1]
		}
	}
	ix.slotCell[slot] = cell
}

// codeForCell returns the residual PQ code to use when slot is reached via cell
// `c`. For a non-SOAR index (or a slot reached via its PRIMARY cell) this is the
// arena code, encoded against the primary centroid. For a SOAR slot reached via
// its SECONDARY cell it is code2[slot], encoded against the secondary centroid —
// so the ADC residual LUT (built for cell c) matches the code's reference cell.
// Falls back to the arena code when code2 is missing (defensive). Caller holds
// ix.mu and has ix.pq active.
func (ix *ivf) codeForCell(slot uint32, c int) []byte {
	if ix.soarTrained && int(slot) < len(ix.cellOf2) && int(ix.cellOf2[slot]) == c &&
		int(slot) < len(ix.code2) && ix.code2[slot] != nil &&
		(int(slot) >= len(ix.slotCell) || int(ix.slotCell[slot]) != c) {
		return ix.code2[slot]
	}
	return ix.arena.Code(slot)
}

// nearestCentroid returns the index of the centroid closest to vec under the
// configured metric. Caller ensures len(centroids) > 0.
func (ix *ivf) nearestCentroid(vec []float32) int {
	dist := ix.metricDist()
	best := 0
	bestD := dist(vec, ix.centroids[0])
	for c := 1; c < len(ix.centroids); c++ {
		if d := dist(vec, ix.centroids[c]); d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}

// defaultSOARLambda is the orthogonality-amplification weight λ used when
// cfg.SOARLambda is 0. ScaNN's reference value is ~1.5; a secondary residual
// parallel to the primary one is penalized λ× harder than its raw magnitude,
// so the chosen secondary cell tends to be COMPLEMENTARY to the primary.
const defaultSOARLambda = float32(1.5)

// soarCandidates is the number of nearest coarse cells (EXCLUDING the primary)
// considered as secondary-assignment candidates. ScaNN evaluates the SOAR loss
// over a small shortlist of the next-nearest cells; the orthogonality term then
// picks the most complementary among them. Kept small (cheap per insert) but > 1
// so the orthogonality choice is meaningful.
const soarCandidates = 8

// soarLambda resolves the configured λ (0 ⇒ the engine default).
func (ix *ivf) soarLambda() float32 {
	if ix.cfg.SOARLambda > 0 {
		return ix.cfg.SOARLambda
	}
	return defaultSOARLambda
}

// secondaryCellLocked picks the SOAR secondary cell for vec given its primary
// cell c1. It evaluates the SOAR loss over the next-nearest soarCandidates coarse
// cells (excluding c1) and returns the argmin:
//
//	loss(c) = ‖r_c‖² + λ·(r_c · r̂1)²   where r_c = vec − centroid(c),
//	r1 = vec − centroid(c1), r̂1 = r1/‖r1‖
//
// The λ·(r_c·r̂1)² term penalizes a secondary residual that is PARALLEL to the
// primary one, so the winner is the cell whose residual is most ORTHOGONAL to r1
// (a complementary encoding). The loss is computed in Euclidean residual space
// regardless of the collection metric (the residual geometry is L2-internal,
// exactly like the IVF-PQ residual codec). Returns c1 itself when no distinct
// candidate exists (nlist == 1 degenerate) — the caller treats c2 == c1 as a
// no-op second membership. Caller holds ix.mu and has centroids set.
func (ix *ivf) secondaryCellLocked(vec []float32, c1 int) int {
	if len(ix.centroids) <= 1 {
		return c1
	}
	dim := ix.cfg.Dim
	// Primary residual r1 and its unit direction r̂1.
	cen1 := ix.centroids[c1]
	r1 := make([]float32, dim)
	var n1sq float64
	for i := 0; i < dim; i++ {
		r1[i] = vec[i] - cen1[i]
		n1sq += float64(r1[i]) * float64(r1[i])
	}
	n1 := math.Sqrt(n1sq)
	// Shortlist the next-nearest cells to vec (excluding c1) by raw centroid L2.
	cands := ix.nearestCellsL2Excl(vec, c1, soarCandidates)
	lambda := float64(ix.soarLambda())
	bestC := c1
	bestLoss := math.MaxFloat64
	for _, c := range cands {
		cen := ix.centroids[c]
		var rcsq, dot float64
		for i := 0; i < dim; i++ {
			rc := float64(vec[i] - cen[i])
			rcsq += rc * rc
			if n1 > 0 {
				dot += rc * float64(r1[i])
			}
		}
		// proj = (r_c · r̂1) = (r_c · r1)/‖r1‖; the orthogonality penalty is λ·proj².
		var proj float64
		if n1 > 0 {
			proj = dot / n1
		}
		loss := rcsq + lambda*proj*proj
		if loss < bestLoss {
			bestLoss = loss
			bestC = c
		}
	}
	return bestC
}

// nearestCellsL2Excl returns up to `count` centroid indices nearest to vec under
// plain L2 (residual geometry), EXCLUDING `excl`, ascending by distance. Used to
// shortlist SOAR secondary candidates. Caller holds ix.mu.
func (ix *ivf) nearestCellsL2Excl(vec []float32, excl, count int) []int {
	type cd struct {
		c int
		d float32
	}
	scored := make([]cd, 0, len(ix.centroids))
	for c := range ix.centroids {
		if c == excl {
			continue
		}
		scored = append(scored, cd{c: c, d: l2Sq(vec, ix.centroids[c])})
	}
	sort.Slice(scored, func(a, b int) bool {
		if scored[a].d != scored[b].d {
			return scored[a].d < scored[b].d
		}
		return scored[a].c < scored[b].c // deterministic tie-break by cell index
	})
	if count > len(scored) {
		count = len(scored)
	}
	out := make([]int, count)
	for i := 0; i < count; i++ {
		out[i] = scored[i].c
	}
	return out
}

// l2Sq is the squared Euclidean distance, used for SOAR residual-geometry cell
// shortlisting independent of the collection metric.
func l2Sq(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// assignSecondaryLocked files slot into its SOAR secondary list and records
// cellOf2[slot] (+ code2[slot] for IVF-PQ). c2 == c1 (the degenerate nlist==1 or
// no-distinct-candidate case) is a no-op: a slot must never appear twice in the
// same list. Caller holds ix.mu, has SOAR active (soarTrained), and has already
// done the primary assignment (slotCell[slot]/primary list). vec is the stored
// (cosine-normalized) vector.
func (ix *ivf) assignSecondaryLocked(slot uint32, c1 int, vec []float32) {
	c2 := ix.secondaryCellLocked(vec, c1)
	ix.ensureSOARSlot(slot)
	if c2 == c1 {
		// No distinct secondary cell — record c1 so cellOf2 is well-defined, but do
		// NOT add a duplicate list membership (the slot is already in c1's list).
		ix.cellOf2[slot] = uint32(c1) //nolint:gosec // c1 < nlist < 2^32
		return
	}
	ix.cellOf2[slot] = uint32(c2) //nolint:gosec // c2 < nlist < 2^32
	ix.lists[c2] = append(ix.lists[c2], slot)
	if ix.pqActive() {
		// Encode the residual against the SECONDARY centroid (complementary code).
		cen := ix.centroids[c2]
		res := make([]float32, ix.cfg.Dim)
		for i := range res {
			res[i] = vec[i] - cen[i]
		}
		ix.code2[slot] = ix.encodeResidualCode(res)
	}
}

// encodeResidualCode encodes a residual into a per-slot PQ code using the active
// codec width: the 8-bit byte-per-subspace code (ix.pq.Encode → m bytes) by
// default, or the nibble-packed 4-bit code (ix.pq4.Encode → (m+1)/2 bytes) when
// the 4-bit LUT16 codec is active. Caller holds ix.mu and has ix.pq trained.
func (ix *ivf) encodeResidualCode(res []float32) []byte {
	if ix.pq4 != nil {
		return ix.pq4.Encode(res)
	}
	return ix.pq.Encode(res)
}

// ensureSOARSlot grows cellOf2 (+ code2 for IVF-PQ) so slot is addressable,
// mirroring encodeResidualLocked's amortized-doubling growth for slotCell. Caller
// holds ix.mu.
func (ix *ivf) ensureSOARSlot(slot uint32) {
	if int(slot) >= len(ix.cellOf2) {
		if int(slot) >= cap(ix.cellOf2) {
			newCap := cap(ix.cellOf2)
			if newCap == 0 {
				newCap = 1
			}
			for newCap <= int(slot) {
				newCap *= 2
			}
			grown := make([]uint32, int(slot)+1, newCap)
			copy(grown, ix.cellOf2)
			ix.cellOf2 = grown
		} else {
			ix.cellOf2 = ix.cellOf2[:int(slot)+1]
		}
	}
	if ix.pqActive() {
		if int(slot) >= len(ix.code2) {
			if int(slot) >= cap(ix.code2) {
				newCap := cap(ix.code2)
				if newCap == 0 {
					newCap = 1
				}
				for newCap <= int(slot) {
					newCap *= 2
				}
				grown := make([][]byte, int(slot)+1, newCap)
				copy(grown, ix.code2)
				ix.code2 = grown
			} else {
				ix.code2 = ix.code2[:int(slot)+1]
			}
		}
	}
}

// InsertIfAbsent inserts only if id is not currently live, atomically (mirror
// hnsw.InsertIfAbsent). Race A / Race B semantics are identical.
func (ix *ivf) InsertIfAbsent(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) (bool, error) {
	return ix.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, nil, 0)
}

// InsertIfAbsentVersion mirrors hnsw.InsertIfAbsentVersion: on a real insert it
// sets the version VERBATIM to `version` (0 → fresh insert version 1) and the
// ABSOLUTE per-key payload deadline map keyExpires VERBATIM (nil = none) so the
// online reshard copy keeps the doc's original key deadlines time-stable.
func (ix *ivf) InsertIfAbsentVersion(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) (bool, error) {
	return ix.insertIfAbsentBody(id, vec, ttl, meta, sparse, keyExpires, version, false, 0)
}

// InsertIfAbsentVersionAt is InsertIfAbsentVersion whose liveness OUTCOME, reclaim,
// and point-TTL deadline are judged against the EXPLICIT leader-stamped clock
// nowMs, so skewed replicas agree on resurrection and stamp identical deadlines
// (#4 vector TTL determinism).
func (ix *ivf) InsertIfAbsentVersionAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) (bool, error) {
	return ix.insertIfAbsentBody(id, vec, ttl, meta, sparse, keyExpires, version, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) insertIfAbsentBody(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, stamped bool, nowMs uint64) (bool, error) {
	if err := checkRecordValues(meta); err != nil {
		return false, err
	}
	start := time.Now()
	defer func() { ix.insertLat.observe(time.Since(start)) }()

	if len(vec) != ix.cfg.Dim {
		return false, ErrDimMismatch
	}
	stored := vec
	if ix.cfg.Metric == Cosine {
		stored = append([]float32(nil), vec...)
		normalize(stored)
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	if ix.liveLockedAt(id, now) {
		return false, nil
	}
	if !ix.bucket.Take() {
		ix.quotaRejects.Add(1)
		return false, ErrCollectionRateLimited
	}
	if _, err := ix.insertLockedAt(id, stored, ttl, meta, sparse, keyExpires, version, now); err != nil {
		return false, err
	}
	return true, nil
}

// Delete tombstones id. The list entry for its slot is left in place (the
// canonical IVF lazy-delete) and filtered out of results by admits on scan.
// cas is the optimistic-CAS precondition (CASCond{} = unconditional, as before);
// a mismatch returns ErrVersionConflict with no mutation.
func (ix *ivf) Delete(id uint64, cas CASCond) (bool, error) {
	return ix.deleteBody(id, cas, false, 0)
}

// DeleteAt is Delete judging the dead-slot liveness gate against the EXPLICIT
// leader-stamped clock nowMs, so replicas agree on already-dead vs live and their
// tombstone sets stay identical (#4 vector TTL determinism).
func (ix *ivf) DeleteAt(id uint64, cas CASCond, nowMs int64) (bool, error) {
	return ix.deleteBody(id, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) deleteBody(id uint64, cas CASCond, stamped bool, nowMs uint64) (bool, error) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	slot, ok := ix.arena.Slot(id)
	if !ok {
		if err := cas.check(0); err != nil {
			return false, err
		}
		return false, nil
	}
	if ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		if err := cas.check(0); err != nil {
			return false, err
		}
		return false, nil
	}
	if err := cas.check(ix.arena.Version(slot)); err != nil {
		return false, err
	}
	ix.tombstoned[slot] = true
	ix.idSetVersion++
	ix.bumpData() // id-set change also invalidates the order_by snapshot
	return true, nil
}

// BuildConcurrent bulk-loads an EMPTY index and TRAINS the coarse quantizer: it
// places every vector in the arena, runs k-means to nlist centroids, and files
// each slot into its nearest centroid's inverted list. This is the train-at-bulk-
// build trigger. workers parallelizes the dominant cost — the k-means assignment
// step (coarse quantizer + per-subspace IVF-PQ codebooks) — across goroutines
// while keeping the centroid update reduce in deterministic index order, so the
// centroids are bit-identical regardless of the worker count (workers<=1 ⇒
// serial). Constraints match hnsw: empty index, no TTL/sparse (a pure vector
// load); see BuildConcurrentMeta for the payload-bearing form.
func (ix *ivf) BuildConcurrent(ids []uint64, vecs [][]float32, workers int) error {
	return ix.BuildConcurrentMeta(ids, vecs, nil, workers)
}

// BuildConcurrentMeta is BuildConcurrent with an OPTIONAL per-point payload:
// metas is either nil/empty (byte-identical to BuildConcurrent) or exactly
// len(ids) long, with a nil entry per payload-less point. As in the HNSW build,
// payloads are applied by the SINGLE-THREADED placement loop, so the index
// becomes searchable and filterable at the same instant. IVF has no BM25 corpus,
// so a payload's postings are the arena's stored metadata and the payload index,
// and nothing else.
func (ix *ivf) BuildConcurrentMeta(ids []uint64, vecs [][]float32, metas []Metadata, workers int) error {
	if err := checkRecordValuesAll(metas); err != nil {
		return err
	}
	if len(metas) != 0 && len(metas) != len(ids) {
		return ErrBuildMetaLenMismatch
	}
	start := time.Now()
	defer func() { ix.insertLat.observe(time.Since(start)) }()

	if len(ids) != len(vecs) {
		return ErrBuildLenMismatch
	}
	for _, v := range vecs {
		if len(v) != ix.cfg.Dim {
			return ErrDimMismatch
		}
	}
	// Match the Collection.BuildConcurrent contract + hnsw: 0 ⇒ GOMAXPROCS, so the
	// default build path gets parallel (bit-identical) k-means, not serial.
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}

	ix.mu.Lock()
	defer ix.mu.Unlock()

	// The payload index joins "empty" now that the placement loop below writes
	// payload postings — same restatement as the HNSW/Vamana builders, and for the
	// same reason: a bulk build claims the whole slot space from zero, so a
	// pre-populated index would leave a previous occupant's postings pointing at
	// slots this build has just reused.
	if ix.arena.Size() != 0 || ix.trained || !ix.payloadIdx.isEmpty() {
		return ErrBuildNonEmpty
	}
	n := len(ids)
	if n == 0 {
		return nil
	}
	// Quota, on the same terms as the HNSW and Vamana bulk builds. IVF never
	// columnises, so its bound is the plainer of the two — exactly its own
	// insert-path check applied once to the whole load.
	if err := bulkQuotaErr(ix.cfg, n); err != nil {
		ix.quotaRejects.Add(1)
		return err
	}
	if err := ix.arena.Reserve(n); err != nil {
		return err
	}

	// Place every vector (normalizing for cosine, matching Insert) into a dense
	// slot 0..n-1.
	train := make([][]float32, n)
	for i := range ids {
		slot := uint32(i) //nolint:gosec // i < n, bounded by 2^32 vectors per arena
		v := vecs[i]
		if ix.cfg.Metric == Cosine {
			v = append([]float32(nil), v...)
			normalize(v)
		}
		ix.arena.PutAt(slot, ids[i], v)
		ix.arena.idMap[ids[i]] = slot
		train[i] = v
		// The index is empty on entry, so every slot is fresh and reindex has
		// nothing to drop — the one difference from the incremental path, which must
		// call reindex unconditionally because its slots can be reused.
		if len(metas) != 0 && len(metas[i]) > 0 { // length, not nilness — see hnsw.BuildConcurrentMeta
			ix.arena.SetMetadata(slot, metas[i])
			ix.payloadIdx.reindex(slot, metas[i])
		}
	}
	ix.bytesUsed += estimateInsertBytes(ix.cfg.Dim, ix.cfg.M) * int64(n)
	ix.insertOps.Add(uint64(n)) //nolint:gosec // n >= 0
	ix.idSetVersion++
	ix.bumpData() // bulk id-set change also invalidates the order_by snapshot

	ix.trainLocked(train, workers)
	return nil
}

// resolveNlist picks the number of inverted lists for liveN live vectors. When
// Config.IVFNlist > 0 it is honored (capped to liveN so k-means never has empty
// cells); otherwise it falls back to the default heuristic max(1, 4*sqrt(liveN)),
// capped so k-means has at least one point per cell on average.
func (ix *ivf) resolveNlist(liveN int) int {
	if liveN <= 0 {
		return 1
	}
	nl := ix.cfg.IVFNlist
	if nl <= 0 {
		nl = int(4 * math.Sqrt(float64(liveN)))
	}
	if nl < 1 {
		nl = 1
	}
	if nl > liveN {
		nl = liveN
	}
	return nl
}

// trainLocked runs k-means over `sample` (already normalized for cosine), sets
// the centroids, files every live arena slot into its nearest cell, and flips
// trained. workers parallelizes the k-means assignment step (deterministic;
// see kmeans). Must hold ix.mu (write). Used by BuildConcurrent.
func (ix *ivf) trainLocked(sample [][]float32, workers int) {
	nl := ix.resolveNlist(len(sample))
	seed := ix.cfg.Seed
	if seed == 0 {
		seed = 1 // deterministic default (kmeans treats 0 as a valid seed; pick a stable one)
	}
	ix.centroids = kmeans(sample, nl, seed, ix.cfg.Metric, workers)
	ix.nlist = len(ix.centroids)
	if ix.nprobe <= 0 {
		ix.nprobe = defaultNprobe
	}
	// IVF-PQ: train the RESIDUAL product quantizer on {sample[i] − centroid[cell]}
	// (the canonical IVFADC). The coarse assignment is by the same nearestCentroid
	// the lists use, so the residuals match what Encode will later subtract. Train
	// BEFORE rebuildListsLocked so the per-slot residual encode below sees ix.pq.
	if ix.cfg.IVFPQ && len(ix.centroids) > 0 {
		ix.trainResidualPQLocked(sample, workers)
	}
	// SOAR multi-assignment: flip soarTrained BEFORE rebuildListsLocked so the
	// list-fill loop also files each slot into its complementary secondary cell.
	// Reset the per-slot SOAR arrays so a retrain rebuilds them from scratch (a
	// stale cellOf2/code2 from the prior centroids must not survive). cfg.SOAR off
	// ⇒ soarTrained stays false and rebuildListsLocked is byte-identical.
	ix.soarTrained = ix.cfg.SOAR && len(ix.centroids) > 1
	ix.cellOf2 = nil
	ix.code2 = nil
	ix.rebuildListsLocked()
	ix.trained = true
	// Drift-retrain checkpoint (cfg.IVFDriftRetrain). Record the live count and the
	// mean nearest-centroid distance of the SAME slot-ordered sample we just trained
	// on against the FRESH centroids — the reference shouldDriftRetrain measures
	// future drift against. Set for BOTH first-train and retrain (this runs on every
	// train). Pure function of the applied sample + new centroids ⇒ identical on every
	// replica. Harmless when IVFDriftRetrain is off (the fields are simply unread).
	ix.lastTrainCount = ix.liveCount()
	ix.lastTrainCost = ix.meanAssignCost(sample, ix.centroids)
	// PQ-only (not IVFRerank): drop the resident floats — only the M-byte codes
	// stay in RAM. IVFRerank keeps them for the exact shortlist rescore.
	if ix.pqActive() && !ix.cfg.IVFRerank {
		ix.arena.dropVecs()
		ix.pqDropped = true
	}
}

// trainResidualPQLocked builds ix.pq from the residuals of sample against each
// vector's nearest coarse centroid. m is cfg.IVFPQM (or defaultPQM). Training is
// L2-internal (codebooks minimize reconstruction error); the collection metric
// only drives the ADC LUT (see pq.go). workers parallelizes each subspace's
// k-means assignment (deterministic). Must hold ix.mu and have centroids set.
func (ix *ivf) trainResidualPQLocked(sample [][]float32, workers int) {
	residuals := make([][]float32, len(sample))
	for i, v := range sample {
		c := ix.nearestCentroid(v)
		cen := ix.centroids[c]
		r := make([]float32, ix.cfg.Dim)
		for j := range r {
			r[j] = v[j] - cen[j]
		}
		residuals[i] = r
	}
	seed := ix.cfg.Seed
	if seed == 0 {
		seed = 1
	}
	// OPQ (cfg.OPQ): trainPQ builds R and trains the codebooks on the rotated
	// residuals R(x−centroid). encode/queryLUT rotate the residual before the
	// subspace split; reconstruct un-rotates (Rᵀ) BEFORE vecFor adds the centroid
	// back (see vecFor). cfg.OPQ=false ⇒ rotation nil ⇒ byte-identical to plain PQ.
	// PQNBits selects the residual code width: 8 (default, byte-per-subspace) or 4
	// (LUT16 — 16 sub-centroids/subspace, nibble-packed (m+1)/2-byte codes scored by
	// the in-register fast-scan kernel on the query path). The arena codes side-array
	// was sized to match (enableCodes(ivfPQCodeLen)) in newIVF.
	p, err := trainPQ(residuals, ivfPQM(ix.cfg), ix.cfg.Dim, seed, ix.cfg.Metric, workers, ix.cfg.OPQ, ix.cfg.OPQIters, ix.cfg.AnisotropicEta, ivfPQNBits(ix.cfg))
	if err != nil {
		// Validate guarantees dim % m == 0 and m > 0, so trainPQ cannot fail here;
		// guard defensively and leave PQ off (exact brute-force) if it ever does.
		return
	}
	ix.pq = p
	ix.refreshPQ4Locked()
}

// rebuildListsLocked re-files every LIVE slot into its nearest centroid's list,
// discarding any prior lists. Tombstoned/expired slots are skipped (they would
// only be filtered on scan anyway). Must hold ix.mu and have centroids set.
func (ix *ivf) rebuildListsLocked() {
	ix.lists = make([][]uint32, len(ix.centroids))
	// DETERMINISM: iterate arena slots in ASCENDING SLOT ORDER, never Go-map
	// (ix.arena.idMap) iteration order. List membership is a deterministic argmin, and
	// the ORDER WITHIN each list (and thus the serialized snapshot bytes) must be
	// identical across replicas — a slot-ordered fill makes the whole trained state,
	// including the inverted lists, bit-identical.
	//
	// TOMBSTONE-ONLY MEMBERSHIP (intentionally TTL-blind): this function runs inside
	// trainLocked (called by autoTrainLocked and the drift-fired retrain). Using
	// liveSlotLocked / isExpired here would make the inverted-list membership
	// wall-clock-dependent — a TTL'd-but-not-yet-tombstoned point that straddled the
	// apply windows of two replicas would land in the list on one replica but not the
	// other → divergent serialized lists → divergent snapshot bytes even when the
	// centroids are identical. We match the tombstone-only membership of
	// liveSampleLocked so the full trained state (centroids + lists) is pure applied
	// state and bit-identical across replicas. A stale slot is detected by the
	// arena.idMap round-trip (slot→id→idMap slot must equal slot) to exclude holes
	// left by hard-deletes/reclaim WITHOUT consulting the wall clock.
	capacity := ix.arena.Capacity()
	if ix.pqDropped {
		// PQ-only (floats dropped): re-file from the persisted slotCell (primary) and,
		// when SOAR is active, the persisted cellOf2 (secondary). The codes2 are NOT
		// re-encoded (no floats) — they already exist; this path only rebuilds the list
		// MEMBERSHIP (e.g. after reclaim frees slots). A secondary equal to the primary
		// was a no-op at assign time, so skip it here too (no duplicate membership).
		for s := 0; s < capacity; s++ {
			slot := uint32(s) //nolint:gosec // s < Capacity() < 2^32
			if !ix.arenaSlotActiveLocked(slot) {
				continue
			}
			if int(slot) < len(ix.slotCell) {
				c := ix.slotCell[slot]
				if int(c) < len(ix.lists) {
					ix.lists[c] = append(ix.lists[c], slot)
				}
				if ix.soarTrained && int(slot) < len(ix.cellOf2) {
					c2 := ix.cellOf2[slot]
					if c2 != c && int(c2) < len(ix.lists) {
						ix.lists[c2] = append(ix.lists[c2], slot)
					}
				}
			}
		}
		return
	}
	if ix.pqActive() {
		// IVF-PQ build: size slotCell to the arena capacity so every live slot can
		// record its cell. Floats are still resident here (dropVecs runs after).
		ix.slotCell = make([]uint32, ix.arena.Capacity())
	}
	if ix.soarTrained {
		// Size the SOAR secondary arrays to the arena capacity so every live slot has
		// a deterministic cellOf2 (+ code2 for IVF-PQ), mirroring the slotCell sizing.
		ix.cellOf2 = make([]uint32, ix.arena.Capacity())
		if ix.pqActive() {
			ix.code2 = make([][]byte, ix.arena.Capacity())
		}
	}
	for s := 0; s < capacity; s++ {
		slot := uint32(s) //nolint:gosec // s < Capacity() < 2^32
		if !ix.arenaSlotActiveLocked(slot) {
			continue
		}
		vec := ix.arena.Vec(slot)
		c := ix.nearestCentroid(vec)
		ix.lists[c] = append(ix.lists[c], slot)
		if ix.pqActive() {
			ix.encodeResidualLocked(slot, uint32(c), vec) //nolint:gosec // c < nlist < 2^32
		}
		if ix.soarTrained {
			ix.assignSecondaryLocked(slot, c, vec)
		}
	}
}

// arenaSlotActiveLocked reports whether slot is a non-stale, non-tombstoned arena
// slot, WITHOUT consulting the wall clock (no isExpired / ix.now()). Used by
// liveSampleLocked and rebuildListsLocked so the training sample and the inverted
// lists use tombstone-only membership — a pure function of Raft-applied state that
// is bit-identical across replicas regardless of wall-clock skew. Must hold ix.mu.
//
// A wall-clock-expired-but-not-tombstoned slot IS still applied state (the reclaim
// op has not been committed yet); including it in training is deterministic and safe.
func (ix *ivf) arenaSlotActiveLocked(slot uint32) bool {
	if ix.tombstoned[slot] {
		return false
	}
	id := ix.arena.ID(slot)
	cur, ok := ix.arena.idMap[id]
	return ok && cur == slot
}

// ---------------------------------------------------------------------------
// point reads (mirror hnsw)
// ---------------------------------------------------------------------------

// Exists reports whether id is currently live (mirror hnsw.Exists).
func (ix *ivf) Exists(id uint64) bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.liveLocked(id)
}

// Get returns a deep copy of id's record + remaining TTL (mirror hnsw.Get).
func (ix *ivf) Get(id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	slot, present := ix.arena.Slot(id)
	if !present || ix.tombstoned[slot] || ix.isExpired(slot) {
		return nil, nil, 0, nil, 0, false
	}
	version = ix.arena.Version(slot)
	vec = append([]float32(nil), ix.vecFor(slot)...)
	if m := ix.liveMeta(slot, uint64(ix.now())); len(m) > 0 {
		out := make(Metadata, len(m))
		for k, v := range m {
			out[k] = v
		}
		meta = out
	}
	if exp := ix.arena.ExpiresAt(slot); exp != 0 {
		if now := uint64(ix.now()); exp > now {
			ttl = time.Duration(exp-now) * time.Millisecond
		}
	}
	if sv := ix.arena.Sparse(slot); sv != nil {
		sparse = sv.Clone()
	}
	return vec, meta, ttl, sparse, version, true
}

// GetProjected is Get with projection: withVec gates the dense-vector copy and
// withPayload gates the metadata + sparse clones (mirror hnsw.GetProjected), so the
// batch-read fast path never copies a projection it will discard. A
// with_vector=false / with_payload=false get allocates nothing per point. All other
// semantics (liveness gate, torn-read safety under RLock, PQDropVecs reconstruction)
// match Get.
func (ix *ivf) GetProjected(id uint64, withVec, withPayload bool) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	slot, present := ix.arena.Slot(id)
	if !present || ix.tombstoned[slot] || ix.isExpired(slot) {
		return nil, nil, 0, nil, 0, false
	}
	version = ix.arena.Version(slot)
	if withVec {
		vec = append([]float32(nil), ix.vecFor(slot)...) // COPY only when requested
	}
	if withPayload {
		if m := ix.liveMeta(slot, uint64(ix.now())); len(m) > 0 {
			out := make(Metadata, len(m))
			for k, v := range m {
				out[k] = v
			}
			meta = out
		}
	}
	if exp := ix.arena.ExpiresAt(slot); exp != 0 {
		if now := uint64(ix.now()); exp > now {
			ttl = time.Duration(exp-now) * time.Millisecond
		}
	}
	if withPayload {
		if sv := ix.arena.Sparse(slot); sv != nil {
			sparse = sv.Clone()
		}
	}
	return vec, meta, ttl, sparse, version, true
}

// GetInto is Get that appends the vector into the caller-owned scratch dst
// (passed as dst[:0]) instead of allocating a fresh []float32 (mirror
// hnsw.GetInto). The copy happens under ix.mu.RLock exactly like Get, so it
// carries the same liveness gate and torn-read safety; the returned vec aliases
// dst's backing when dst has the capacity (zero-alloc on reuse), except where
// vecFor must reconstruct/allocate (PQDropVecs), which GetInto cannot elide.
func (ix *ivf) GetInto(dst []float32, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	slot, present := ix.arena.Slot(id)
	if !present || ix.tombstoned[slot] || ix.isExpired(slot) {
		return nil, nil, 0, nil, 0, false
	}
	version = ix.arena.Version(slot)
	vec = append(dst[:0], ix.vecFor(slot)...)
	if m := ix.liveMeta(slot, uint64(ix.now())); len(m) > 0 {
		out := make(Metadata, len(m))
		for k, v := range m {
			out[k] = v
		}
		meta = out
	}
	if exp := ix.arena.ExpiresAt(slot); exp != 0 {
		if now := uint64(ix.now()); exp > now {
			ttl = time.Duration(exp-now) * time.Millisecond
		}
	}
	if sv := ix.arena.Sparse(slot); sv != nil {
		sparse = sv.Clone()
	}
	return vec, meta, ttl, sparse, version, true
}

// ---------------------------------------------------------------------------
// search (IVF-specific candidate gathering; exact rerank)
// ---------------------------------------------------------------------------

// Search returns the k nearest neighbors (mirror hnsw.Search).
func (ix *ivf) Search(query []float32, k int) ([]Result, error) {
	return ix.SearchInto(nil, query, k, Filter{})
}

// SearchFiltered returns the k nearest neighbors satisfying filter.
func (ix *ivf) SearchFiltered(query []float32, k int, filter Filter) ([]Result, error) {
	return ix.SearchInto(nil, query, k, filter)
}

// SearchInto appends up to k nearest neighbors matching filter onto dst. The IVF
// analogue of hnsw.SearchInto: trained → probe nprobe cells + exact rerank;
// untrained → exact brute force over the whole arena.
func (ix *ivf) SearchInto(dst []Result, query []float32, k int, filter Filter) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if len(query) != ix.cfg.Dim {
		return dst, ErrDimMismatch
	}
	if k <= 0 {
		return dst, nil
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return dst, err
	}
	q := query
	if ix.cfg.Metric == Cosine {
		q = append([]float32(nil), query...)
		normalize(q)
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so the sentinel
	// surfaces instead of a silent empty result — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return dst, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)
	return ix.searchLocked(q, k, pred, dst), nil
}

// SearchFilteredWith is SearchInto with an OPTIONAL external metadata provider
// (the named/MV hook), the IVF analogue of hnsw.SearchFilteredWith. metaOf == nil
// is byte-identical to SearchInto (the predicate reads the sub-arena metadata via
// admits). When metaOf != nil the predicate is evaluated against the EXTERNAL
// per-point payload via metaOf(id): the named/MV sub-arena carries NO metadata,
// so admission re-checks the live shared payload via admitsWith. IVF has no
// payload-index filter-first planner (it always probes cells / brute-forces), so
// there is nothing to bypass — only the admit gate is rerouted. Superset-safety +
// predicate re-check hold exactly as in SearchFiltered: every probed candidate is
// re-admitted (tombstone/TTL liveness + predicate) before it is scored.
func (ix *ivf) SearchFilteredWith(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if len(query) != ix.cfg.Dim {
		return dst, ErrDimMismatch
	}
	if k <= 0 {
		return dst, nil
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return dst, err
	}
	q := query
	if ix.cfg.Metric == Cosine {
		q = append([]float32(nil), query...)
		normalize(q)
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so the sentinel
	// surfaces instead of a silent empty result — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return dst, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)

	matched := ix.gatherLockedWith(q, k, pred, metaOf)
	for i := 0; i < len(matched) && i < k; i++ {
		// No allocation re-check: matched comes from the inverted lists, which
		// only hold slots of assigned points, already past the liveness gate.
		// Emitting the id verbatim is what makes user id 0 returnable.
		dst = append(dst, Result{ID: ix.slotID(matched[i].slot), Distance: matched[i].dist})
	}
	return dst, nil
}

// gatherLockedWith is gatherLocked with an OPTIONAL external metadata provider:
// metaOf == nil is byte-identical to gatherLocked. When metaOf != nil every
// admission goes through admitsWith(slot, pred, metaOf) so the predicate is
// re-checked against the live shared payload (the sub-arena carries no metadata).
// Must hold ix.mu (read).
func (ix *ivf) gatherLockedWith(q []float32, k int, pred Predicate, metaOf metaProvider) []slotDist {
	// Under-lock backstop: a failed mmap grow nils the arena floats, so every dense
	// scan (this and gatherLocked below) would slice the freed region. Return an
	// empty candidate set instead. Runs under the caller's ix.mu RLock, closing the
	// TOCTOU against a concurrent grow that poisons after a top-of-op check. See
	// arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil
	}
	if metaOf == nil {
		return ix.gatherLocked(q, k, pred)
	}
	dist := ix.metricDist()
	now := uint64(ix.now()) // one clock read for the whole gather (loop-invariant under the RLock)
	if !ix.trained || len(ix.centroids) == 0 {
		// Untrained brute force with the external admit gate.
		matched := make([]slotDist, 0, ix.arena.Size())
		for _, slot := range ix.arena.idMap {
			if !ix.admitsWith(slot, pred, metaOf, now) {
				continue
			}
			matched = append(matched, slotDist{slot: slot, dist: dist(q, ix.vecFor(slot))})
		}
		sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
		return matched
	}

	nprobe := ix.nprobe
	if nprobe < 1 {
		nprobe = 1
	}
	if nprobe > len(ix.centroids) {
		nprobe = len(ix.centroids)
	}
	cells := ix.nearestCells(q, nprobe, dist)

	// IVF-PQ: ADC scan over residual codes, external admit gate. Pre-sized matched +
	// pooled visited set + per-query (not per-cell) residual/LUT buffers, mirroring
	// gatherADCLocked — see BENCHMARKS.md Wave 1 findings.
	if ix.pqActive() {
		total := 0
		for _, c := range cells {
			total += len(ix.lists[c])
		}
		matched := make([]slotDist, 0, total)
		seen := ivfGatherVisitedPool.Get().(*visitedSet)
		seen.prepare(ix.arena.Capacity())
		defer ivfGatherVisitedPool.Put(seen)
		res := make([]float32, ix.cfg.Dim)
		// FAST-SCAN (4-bit LUT16) vs scalar adc — see gatherADCLocked. Same per-cell
		// LUT build + per-cell flush; the admit gate here is the external-meta variant
		// (admitsWith metaOf), otherwise the gather is identical.
		fast := ix.pq4Active()
		var scorer *fastScanScorer
		if fast {
			scorer = ix.newFastScanScorer()
		}
		var lut []float32
		if !fast {
			lut = make([]float32, ix.pq.lutLen())
		}
		for _, c := range cells {
			cen := ix.centroids[c]
			for i := range res {
				res[i] = q[i] - cen[i]
			}
			var qlut *lut16
			if fast {
				qlut = ix.pq4.buildLUT16(res)
			} else {
				ix.pq.queryLUTInto(lut, res)
			}
			list := ix.lists[c]
			for i, slot := range list {
				if j := i + prefetchDistance; j < len(list) {
					prefetch(ix.arena.codePtr(list[j]))
				}
				if seen.seen(slot) {
					continue
				}
				seen.mark(slot)
				if !ix.admitsWith(slot, pred, metaOf, now) {
					continue
				}
				if fast {
					matched = scorer.add(qlut, slot, ix.codeForCell(slot, c), matched)
					continue
				}
				matched = append(matched, slotDist{slot: slot, dist: ix.pq.adc(lut, ix.codeForCell(slot, c))})
			}
			if fast {
				matched = scorer.flush(qlut, matched)
			}
		}
		sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
		if ix.cfg.IVFRerank && !ix.pqDropped {
			shortlist := k * ivfRerankFactor
			if shortlist < k {
				shortlist = k
			}
			if shortlist < len(matched) {
				matched = matched[:shortlist]
			}
			rdist := ix.metricDist()
			for i := range matched {
				matched[i].dist = rdist(q, ix.arena.Vec(matched[i].slot))
			}
			sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
		}
		return matched
	}

	// IVF-Flat: exact full-vector scan, external admit gate. Pre-sized matched +
	// pooled visited set, mirroring gatherLocked.
	total := 0
	for _, c := range cells {
		total += len(ix.lists[c])
	}
	// Same batched cell scan as gatherLocked, with the external-metadata admit
	// gate — see gatherFlatBatched.
	if batch := ix.batchExact(q); batch.ok() {
		return ix.gatherFlatBatched(&batch, cells, total, func(slot uint32) bool {
			return ix.admitsWith(slot, pred, metaOf, now)
		})
	}
	matched := make([]slotDist, 0, total)
	seen := ivfGatherVisitedPool.Get().(*visitedSet)
	seen.prepare(ix.arena.Capacity())
	defer ivfGatherVisitedPool.Put(seen)
	for _, c := range cells {
		for _, slot := range ix.lists[c] {
			if seen.seen(slot) {
				continue
			}
			seen.mark(slot)
			if !ix.admitsWith(slot, pred, metaOf, now) {
				continue
			}
			matched = append(matched, slotDist{slot: slot, dist: dist(q, ix.arena.Vec(slot))})
		}
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	_ = k
	return matched
}

// filterFirstByID computes exact top-k over an index-narrowed candidate ID set
// whose payload lives in an EXTERNAL map (the named/MV families: the sub-arena
// carries the VECTORS but no metadata). The IVF analogue of hnsw.filterFirstByID:
// resolve each candidate's slot, apply the tombstone/TTL liveness gate AND the
// external predicate RE-CHECK via admitsWith(metaOf), score with vecFor (exact
// floats present / reconstructed when PQ-dropped), return the k closest. The
// candidate set is a SUPERSET, so the re-check is what makes the result correct.
// Takes ix.mu (read) itself (the caller holds the owning collection's lock).
func (ix *ivf) filterFirstByID(dst []Result, cands []uint64, query []float32, k int, pred Predicate, metaOf func(id uint64) Metadata) []Result {
	if len(query) != ix.cfg.Dim || k <= 0 {
		return dst
	}
	q := query
	if ix.cfg.Metric == Cosine {
		q = append([]float32(nil), query...)
		normalize(q)
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	dist := ix.metricDist()
	now := uint64(ix.now()) // one clock read for the whole candidate scan
	matched := make([]slotDist, 0, len(cands))
	for _, id := range cands {
		slot, ok := ix.arena.Slot(id)
		if !ok {
			continue // this point omitted this named space (no vector here)
		}
		if !ix.admitsWith(slot, pred, metaOf, now) {
			continue // tombstoned/expired or fails the live-meta predicate re-check
		}
		matched = append(matched, slotDist{slot: slot, dist: dist(q, ix.vecFor(slot))})
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	for i := 0; i < len(matched) && i < k; i++ {
		id := ix.arena.ID(matched[i].slot)
		dst = append(dst, Result{ID: id, Distance: matched[i].dist})
	}
	return dst
}

// preferFilterFirst is the cost-based planner decision (the IVF analogue of
// hnsw.preferFilterFirst): should an index-narrowed filtered search brute-force
// the ncand candidates exactly rather than probe cells? IVF's per-query cost is
// roughly nprobe cells' worth of candidates; filter-first is ~ncand distances.
func (ix *ivf) preferFilterFirst(ncand, k int) bool {
	if ncand == 0 {
		return true // nothing to brute force: trivially exact and cheapest
	}
	n := ix.arena.Size()
	if n <= 0 {
		return true
	}
	if k < 1 {
		k = 1
	}
	// Estimate the candidates an IVF probe would scan: (nprobe / nlist) of the
	// arena, floored at k. Filter-first wins when the narrowed candidate set is no
	// larger than what the probe would scan.
	nlist := len(ix.centroids)
	if nlist <= 0 {
		// Untrained: a query brute-forces the whole arena, so filter-first over a
		// narrowed subset is always cheaper.
		return true
	}
	nprobe := ix.nprobe
	if nprobe < 1 {
		nprobe = 1
	}
	if nprobe > nlist {
		nprobe = nlist
	}
	probeCost := float64(n) * float64(nprobe) / float64(nlist)
	if probeCost < float64(k) {
		probeCost = float64(k)
	}
	return float64(ncand) <= probeCost
}

// filterFirstCrossover returns the largest ncand in [0, limit] for which
// preferFilterFirst(ncand, k) is true — the planner's own answer to "how big a
// candidate set could I still act on?", so a materialization that grows past it
// is provably wasted work and candidatesCapped can abandon it mid-build.
//
// IVF's cost model is monotone in ncand, but for a DIFFERENT reason than hnsw's,
// so this is verified here rather than inherited: probeCost depends only on
// (arena size, nprobe, nlist, k) — every term independent of ncand — and the
// decision is the single comparison ncand <= probeCost. That is non-increasing
// in ncand by inspection, and the ncand == 0 and untrained (nlist <= 0) early
// returns are both `true`, which is the weakest possible outcome to widen. So
// there is exactly one crossover and the shared binary search finds it.
func (ix *ivf) filterFirstCrossover(k, limit int) int {
	return crossoverOf(func(ncand int) bool { return ix.preferFilterFirst(ncand, k) }, limit)
}

// Dim returns the configured vector dimensionality.
func (ix *ivf) Dim() int { return ix.cfg.Dim }

// vecsForIDs returns, for each live id present in this index, a COPY of its float
// vector. vecFor returns exact arena floats (which it ALIASES) when present and a
// freshly-allocated reconstruction when the floats were PQ-dropped; either way we
// copy so the caller never retains an arena view. Absent ids are omitted. Takes
// ix.mu (read) itself.
func (ix *ivf) vecsForIDs(ids []uint64) map[uint64][]float32 {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := make(map[uint64][]float32, len(ids))
	for _, id := range ids {
		slot, ok := ix.arena.Slot(id)
		if !ok {
			continue
		}
		out[id] = append([]float32(nil), ix.vecFor(slot)...)
	}
	return out
}

// withVecAccess holds ix.mu (read) for the duration of fn and passes a getter that
// returns each id's vector as an arena VIEW (no copy), valid only while fn runs.
// The allocation-free analogue of vecsForIDs for the MaxSim hot path. (IVF-PQ with
// dropped floats: vecFor reconstructs into a fresh slice, same as vecsForIDs.)
func (ix *ivf) withVecAccess(fn func(get func(id uint64) ([]float32, bool))) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	fn(func(id uint64) ([]float32, bool) {
		slot, ok := ix.arena.Slot(id)
		if !ok {
			return nil, false
		}
		return ix.vecFor(slot), true
	})
}

// searchLocked is the shared KNN body (q already normalized; ix.mu held read).
// Returns the top-k as Results appended onto dst, exact-ranked by metric
// distance. Used directly by SearchInto and as the candidate gatherer for every
// exotic variant (MMR/Recommend/Discover/Groups), so they all WORK on IVF.
func (ix *ivf) searchLocked(q []float32, k int, pred Predicate, dst []Result) []Result {
	matched := ix.gatherLocked(q, k, pred)
	for i := 0; i < len(matched) && i < k; i++ {
		// No allocation re-check: matched comes from the inverted lists, which
		// only hold slots of assigned points, already past the liveness gate.
		// Emitting the id verbatim is what makes user id 0 returnable.
		dst = append(dst, Result{ID: ix.slotID(matched[i].slot), Distance: matched[i].dist})
	}
	return dst
}

// ivfGatherVisitedPool recycles the epoch-stamped visited set used to dedup
// candidate slots across probed cells, so steady-state gather does not allocate a
// per-query map. Each concurrent search borrows its own set under the read lock.
var ivfGatherVisitedPool = sync.Pool{New: func() any { return &visitedSet{} }}

// ivfBatchScratch is the IVF-Flat counterpart of layerScratch's batchSlots /
// batchDist pair: the admitted slots of ONE probed cell and the block of
// distances the batched kernel writes for them. Pooled rather than per-index
// because concurrent searches all run under ix.mu READ-shared, so a per-index
// buffer would be shared by racing scans.
//
// Both buffers are grown to the largest cell scanned so far, so a steady-state
// workload reaches its capacity within the first few queries and never allocates
// again — the same convergence expandBatched relies on.
type ivfBatchScratch struct {
	slots []uint32
	dist  []float32
}

// reserve sizes both buffers for a cell of n members. Both halves are tested
// for the same reason expandBatched tests both: they are always assigned
// together today, but relying on that invisible coupling would turn any future
// edit into a bounds panic on the search hot path.
func (s *ivfBatchScratch) reserve(n int) {
	if cap(s.slots) < n || len(s.dist) < n {
		s.slots = make([]uint32, 0, n)
		s.dist = make([]float32, n)
	}
}

var ivfBatchScratchPool = sync.Pool{New: func() any { return &ivfBatchScratch{} }}

// batchExact returns the batched "one query vs N arena slots" kernel for the
// IVF-Flat cell scan, or the zero batchKernel when none applies — in which case
// the gather stays on the per-pair path, which is always correct and merely
// slower. The decline rules are hnsw.batchExact's, for the same reasons, over
// the same arena type:
//
//   - the raw vectors are not materialized (vecsDropped — the IVF-PQ PQ-only
//     state). The flat scan reads arena.Vec, so it is not reachable there
//     anyway; declining keeps the guard local rather than inherited.
//   - the arena's stride disagrees with cfg.Dim or with the query length. The
//     kernel derives every candidate address from a single (base, dim) pair, so
//     a mismatch is not a slow path but a wrong one.
//   - the metric has no batched kernel on this platform (pickNy returns nil).
//
// The slab header is captured once per gather, matching what the per-pair path's
// arena.Vec would re-derive per candidate. That is safe because the arena can
// only grow or be swapped under ix.mu, which every gather holds read-shared for
// its whole duration; nyBoundsOK turns any violation into a clear failure at the
// offending slot rather than a silent out-of-slab read.
//
// The QUANTIZED gathers (gatherADCLocked and the pqActive branch of
// gatherLockedWith) never call this: their candidates are scored from
// arena.Code through the residual codec — scalar adc or the LUT16 fast-scan
// kernel — which has no batched float form and is left exactly as it was.
func (ix *ivf) batchExact(q []float32) batchKernel {
	a := ix.arena
	if !batchedExpand || a == nil || a.vecsDropped || a.dim <= 0 || len(a.vecs) == 0 {
		return batchKernel{}
	}
	if a.dim != ix.cfg.Dim || len(q) != a.dim {
		return batchKernel{}
	}
	ny := pickNy(ix.cfg.Metric, a.dim)
	if ny == nil {
		return batchKernel{}
	}
	return batchKernel{ny: ny, q: q, base: a.vecs, dim: a.dim, metric: ix.cfg.Metric}
}

// gatherFlatBatched is the batched IVF-Flat cell scan: per probed cell it
// filters the member list down to the deduped, admitted slots and scores ALL of
// them in ONE kernel call, then appends the {slot, dist} pairs in list order.
//
// The IVF cell scan is a better fit for the batched kernels than the graph
// traversal they were built for: there is no visited-set interleaving that
// matters (dedup is a pure filter here, not a traversal decision) and no
// heap-gating between members — every admitted candidate in every probed cell is
// scored and appended unconditionally, then sorted. So hoisting the distance
// computation out of the member loop cannot change WHICH candidates are scored,
// only when.
//
// It does no prefetching of its own: the kernels warm their own candidates
// nyPrefetchAhead ahead as they go (see nyFunc), which is what makes a
// whole-cell call safe — the caller-side burst the per-pair path's single-line
// prefetch approximates does not scale to a 1000-member cell.
//
// The batching is per CELL rather than per query so the scratch stays bounded by
// the largest list rather than by nprobe*listLen, and so matched is built in
// exactly the order the per-pair loop built it: cells in probe order, members in
// list order. Combined with the kernels' bit-identity that makes this gather's
// output byte-for-byte equal to the per-pair gather's, sort included.
//
// admit is the caller's per-candidate gate (ix.admits, or the external-metadata
// admitsWith), invoked on the same slots in the same order as the per-pair loop.
// Must hold ix.mu (read).
func (ix *ivf) gatherFlatBatched(batch *batchKernel, cells []int, total int, admit func(slot uint32) bool) []slotDist {
	matched := make([]slotDist, 0, total)
	seen := ivfGatherVisitedPool.Get().(*visitedSet)
	seen.prepare(ix.arena.Capacity())
	defer ivfGatherVisitedPool.Put(seen)
	sc := ivfBatchScratchPool.Get().(*ivfBatchScratch)
	defer ivfBatchScratchPool.Put(sc)
	for _, c := range cells {
		list := ix.lists[c]
		if len(list) == 0 {
			continue
		}
		sc.reserve(len(list))
		cand := sc.slots[:0]
		for _, slot := range list {
			if seen.seen(slot) {
				continue
			}
			seen.mark(slot)
			if !admit(slot) {
				continue
			}
			cand = append(cand, slot)
		}
		sc.slots = cand
		if len(cand) == 0 {
			continue
		}
		dists := sc.dist[:len(cand)]
		batch.score(cand, dists)
		for i, slot := range cand {
			matched = append(matched, slotDist{slot: slot, dist: dists[i]})
		}
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	return matched
}

// gatherLocked returns the admitted candidate slots scored by exact distance,
// sorted ascending. Trained: scan the nprobe nearest cells' lists. Untrained:
// brute-force the whole arena. k is a hint for cell fan-out width; the actual
// returned set is every admitted candidate in the probed cells (the caller
// top-k's it). Must hold ix.mu (read).
func (ix *ivf) gatherLocked(q []float32, k int, pred Predicate) []slotDist {
	// Under-lock backstop: a failed mmap grow nils the arena floats, so the cell
	// scan / brute force below (both read arena.Vec) would slice the freed region.
	// Return an empty candidate set instead; the error-returning ops reject with
	// ErrIndexPoisoned at their own top. See arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil
	}
	dist := ix.metricDist()
	if !ix.trained || len(ix.centroids) == 0 {
		return ix.bruteForceLocked(q, dist, pred)
	}

	// Probe the nprobe nearest centroids (clamped to [1, nlist]).
	nprobe := ix.nprobe
	if nprobe < 1 {
		nprobe = 1
	}
	if nprobe > len(ix.centroids) {
		nprobe = len(ix.centroids)
	}
	cells := ix.nearestCells(q, nprobe, dist)

	// IVF-PQ: ADC scan over residual codes (the compressed fast path). Per cell the
	// residual query LUT is built ONCE; each candidate costs M byte-indexed lookups.
	if ix.pqActive() {
		return ix.gatherADCLocked(q, k, pred, cells)
	}

	// IVF-Flat (non-PQ): exact full-vector scan. Gather + score + admit, deduping
	// slots (a reused slot can appear in two lists across an incremental insert
	// before the next retrain). matched is pre-sized to the total candidate count
	// and dedup uses a pooled epoch-stamped visited set, so steady-state gather
	// allocates only the matched slice — no per-query map and no append doubling.
	// See BENCHMARKS.md Wave 1 findings.
	now := uint64(ix.now()) // one clock read for the whole IVF-Flat gather (loop-invariant under the RLock)
	total := 0
	for _, c := range cells {
		total += len(ix.lists[c])
	}
	// Batched cell scan when a kernel covers this arena/metric: one call per
	// probed cell instead of one per member. Bit-identical to the loop below —
	// see gatherFlatBatched.
	if batch := ix.batchExact(q); batch.ok() {
		return ix.gatherFlatBatched(&batch, cells, total, func(slot uint32) bool {
			return ix.admits(slot, pred, now)
		})
	}
	matched := make([]slotDist, 0, total)
	seen := ivfGatherVisitedPool.Get().(*visitedSet)
	seen.prepare(ix.arena.Capacity())
	defer ivfGatherVisitedPool.Put(seen)
	for _, c := range cells {
		for _, slot := range ix.lists[c] {
			if seen.seen(slot) {
				continue
			}
			seen.mark(slot)
			if !ix.admits(slot, pred, now) {
				continue
			}
			matched = append(matched, slotDist{slot: slot, dist: dist(q, ix.arena.Vec(slot))})
		}
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	_ = k
	return matched
}

// gatherADCLocked is the IVF-PQ candidate gather: for each probed cell it builds
// the residual query LUT once (queryLUT(q − centroid[cell])) and scores every
// admitted candidate slot by adc(lut, code) — M byte-indexed lookups/candidate,
// no float reads, no inner-loop allocation. PQ-only: the ADC ranking IS the
// result. IVFRerank: the ADC top-(rerankFactor*k) shortlist is exact-rescored on
// the resident floats (mirror hnsw.rescore) for near-exact recall. Must hold
// ix.mu (read); cells are pre-clamped.
func (ix *ivf) gatherADCLocked(q []float32, k int, pred Predicate, cells []int) []slotDist {
	now := uint64(ix.now()) // one clock read for the whole gather (loop-invariant under the RLock)
	total := 0
	for _, c := range cells {
		total += len(ix.lists[c])
	}
	matched := make([]slotDist, 0, total)
	seen := ivfGatherVisitedPool.Get().(*visitedSet)
	seen.prepare(ix.arena.Capacity())
	defer ivfGatherVisitedPool.Put(seen)
	// Residual-query and LUT buffers reused across cells (both consumed
	// synchronously per cell): one alloc per query instead of one per probed cell.
	res := make([]float32, ix.cfg.Dim)
	// FAST-SCAN (4-bit LUT16): when the residual codec is 4-bit, score each probed
	// cell with the in-register VPSHUFB kernel (32 vectors/instruction) instead of
	// the per-slot scalar adc. A per-cell uint8 lut16 is built on the residual q −
	// centroid[c]; admitted slots' nibble-packed codes are batched (fastScanScorer)
	// and flushed per cell so the LUT matches each block's reference centroid. The
	// result is exact vs the scalar adcScalar reference (same uint8 LUT + sums) — the
	// admit/seen gating below is identical to the 8-bit path. 8-bit: float-LUT adc.
	fast := ix.pq4Active()
	var scorer *fastScanScorer
	if fast {
		scorer = ix.newFastScanScorer()
	}
	var lut []float32
	if !fast {
		lut = make([]float32, ix.pq.lutLen())
	}
	for _, c := range cells {
		cen := ix.centroids[c]
		for i := range res {
			res[i] = q[i] - cen[i]
		}
		var qlut *lut16
		if fast {
			qlut = ix.pq4.buildLUT16(res)
		} else {
			ix.pq.queryLUTInto(lut, res)
		}
		list := ix.lists[c]
		for i, slot := range list {
			// Prefetch a candidate's code prefetchDistance ahead so it is in
			// flight (a cache miss is ~100ns) while the intervening adc scans
			// run; codes live at random arena slots. No-op on non-amd64.
			if j := i + prefetchDistance; j < len(list) {
				prefetch(ix.arena.codePtr(list[j]))
			}
			if seen.seen(slot) {
				continue
			}
			seen.mark(slot)
			if !ix.admits(slot, pred, now) {
				continue
			}
			// adc against the code for THIS cell: the primary arena code, or the SOAR
			// secondary code2[slot] when the slot is reached via its secondary cell (so
			// the residual LUT built for cell c matches the code's reference centroid).
			if fast {
				matched = scorer.add(qlut, slot, ix.codeForCell(slot, c), matched)
				continue
			}
			matched = append(matched, slotDist{slot: slot, dist: ix.pq.adc(lut, ix.codeForCell(slot, c))})
		}
		if fast {
			// Flush the cell's partial block before the LUT changes for the next cell.
			matched = scorer.flush(qlut, matched)
		}
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })

	// IVFRerank: exact-rescore the ADC shortlist on the resident floats, then
	// re-sort. PQ-only (floats dropped) returns the ADC ranking directly.
	if ix.cfg.IVFRerank && !ix.pqDropped {
		shortlist := k * ivfRerankFactor
		if shortlist < k {
			shortlist = k
		}
		if shortlist < len(matched) {
			matched = matched[:shortlist]
		}
		dist := ix.metricDist()
		for i := range matched {
			matched[i].dist = dist(q, ix.arena.Vec(matched[i].slot))
		}
		sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	}
	return matched
}

// fastScanScorer batches a probed cell's admitted slots and scores them with the
// 4-bit LUT16 in-register fast-scan kernel (fastScanBlockInto): it gathers each
// admitted slot's nibble-packed code (codeForCell — primary or SOAR-secondary)
// contiguously into a pooled buffer, then runs the kernel over blocks of
// fastScanBlock (32) database vectors, dequantizing each block's integer sums to
// float ADC distances. The result is EXACT against the per-slot scalar adcScalar
// (same uint8 LUT, same per-subspace sums) — only the scoring is vectorized; the
// admit/tombstone/dedup gating in the caller is byte-for-byte unchanged. Buffers
// are reused across cells (one alloc set per query). Only used when ix.pq4Active().
type fastScanScorer struct {
	cl      int      // pq4CodeLen(m): packed bytes per slot
	codes   []byte   // [fastScanBlock*cl] contiguous packed codes for the pending block
	slots   []uint32 // [fastScanBlock] slot ids parallel to codes (block order)
	n       int      // pending slots in the current block (< fastScanBlock)
	scratch []byte   // [m*fastScanBlock] transpose scratch for the kernel
	acc     []uint16 // [fastScanBlock] integer accumulators
	dists   []float32
}

// newFastScanScorer sizes a scorer for the current 4-bit codec. Caller has
// ix.pq4Active().
func (ix *ivf) newFastScanScorer() *fastScanScorer {
	m := ix.pq.m
	cl := pq4CodeLen(m)
	return &fastScanScorer{
		cl:      cl,
		codes:   make([]byte, fastScanBlock*cl),
		slots:   make([]uint32, fastScanBlock),
		scratch: make([]byte, m*fastScanBlock),
		acc:     make([]uint16, fastScanBlock),
		dists:   make([]float32, fastScanBlock),
	}
}

// add buffers one admitted slot's packed code; when the block fills (fastScanBlock
// slots) it flushes via the kernel, appending the scored {slot,dist} to matched.
// code must be exactly cl (pq4CodeLen(m)) bytes — codeForCell guarantees this.
func (s *fastScanScorer) add(lut *lut16, slot uint32, code []byte, matched []slotDist) []slotDist {
	copy(s.codes[s.n*s.cl:], code)
	s.slots[s.n] = slot
	s.n++
	if s.n == fastScanBlock {
		matched = s.flush(lut, matched)
	}
	return matched
}

// flush scores the pending (partial or full) block and appends {slot,dist} to
// matched, resetting the block. A no-op when empty.
func (s *fastScanScorer) flush(lut *lut16, matched []slotDist) []slotDist {
	if s.n == 0 {
		return matched
	}
	lut.fastScanBlockInto(s.dists, s.codes[:s.n*s.cl], s.n, s.scratch, s.acc)
	for i := 0; i < s.n; i++ {
		matched = append(matched, slotDist{slot: s.slots[i], dist: s.dists[i]})
	}
	s.n = 0
	return matched
}

// bruteForceLocked scores every admitted live slot exactly (untrained search, or
// an empty-centroid degenerate). Must hold ix.mu (read).
func (ix *ivf) bruteForceLocked(q []float32, dist distFunc, pred Predicate) []slotDist {
	now := uint64(ix.now()) // one clock read for the whole brute-force scan
	matched := make([]slotDist, 0, ix.arena.Size())
	for _, slot := range ix.arena.idMap {
		if !ix.admits(slot, pred, now) {
			continue
		}
		matched = append(matched, slotDist{slot: slot, dist: dist(q, ix.arena.Vec(slot))})
	}
	sort.Slice(matched, func(a, b int) bool { return matched[a].dist < matched[b].dist })
	return matched
}

// nearestCells returns the indices of the nprobe centroids closest to q,
// ascending by distance. nprobe is pre-clamped to [1, len(centroids)].
func (ix *ivf) nearestCells(q []float32, nprobe int, dist distFunc) []int {
	scored := make([]slotDist, len(ix.centroids))
	for c := range ix.centroids {
		scored[c] = slotDist{slot: uint32(c), dist: dist(q, ix.centroids[c])} //nolint:gosec // c < nlist < 2^32
	}
	sort.Slice(scored, func(a, b int) bool { return scored[a].dist < scored[b].dist })
	out := make([]int, nprobe)
	for i := 0; i < nprobe; i++ {
		out[i] = int(scored[i].slot)
	}
	return out
}

// ---------------------------------------------------------------------------
// exotic search variants — all reuse searchLocked / gatherLocked as the
// candidate gatherer, so they WORK on IVF (unaccelerated but correct).
// ---------------------------------------------------------------------------

// HybridSearch fuses a dense IVF lane and a sparse inverted-index lane (mirror
// hnsw.HybridSearch, swapping graph traversal for the IVF candidate gather).
func (ix *ivf) HybridSearch(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	denseRes, sparseRes, err := ix.buildLanes(dense, sparse, k, opts)
	if err != nil {
		return nil, err
	}
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

// HybridLanes builds the dense + sparse lanes unfused (mirror hnsw.HybridLanes).
func (ix *ivf) HybridLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, []Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()
	return ix.buildLanes(dense, sparse, k, opts)
}

// buildLanes builds the dense (IVF) and sparse (inverted-index) candidate lanes.
// Mirrors hnsw.buildLanes; the dense lane uses searchLocked instead of the graph.
func (ix *ivf) buildLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) (denseRes []Result, sparseRes []Result, err error) {
	if k <= 0 {
		return nil, nil, nil
	}
	if len(dense) > 0 && len(dense) != ix.cfg.Dim {
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
		if ix.cfg.Metric == Cosine {
			q = append([]float32(nil), dense...)
			normalize(q)
		}
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so both
	// HybridSearch and HybridLanes surface the sentinel instead of a silent empty
	// dense lane — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil, nil, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)

	if q != nil {
		denseRes = ix.searchLocked(q, denseK, pred, nil)
	}
	if !sparse.IsZero() {
		now := uint64(ix.now()) // one clock read for the whole sparse admission scan
		admit := func(slot uint32) bool { return ix.admits(slot, pred, now) }
		for _, ss := range ix.sparseIdx.searchTopK(s, ix.arena.Capacity(), sparse, sparseK, admit) {
			// No allocation re-check: searchTopK walks the inverted postings, which
			// only contain slots given a sparse vector at insert time, and the admit
			// closure already applied the liveness gate. Emitting the id verbatim is
			// what makes user id 0 returnable.
			sparseRes = append(sparseRes, Result{ID: ix.slotID(ss.slot), Score: ss.score})
		}
	}
	return denseRes, sparseRes, nil
}

// SearchMMR over-collects FetchK IVF candidates then MMR-reranks (mirror
// hnsw.SearchMMR + mmrSelect).
func (ix *ivf) SearchMMR(query []float32, k int, opts MMROpts) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if len(query) != ix.cfg.Dim {
		return nil, ErrDimMismatch
	}
	if k <= 0 {
		return nil, nil
	}
	lambda := opts.Lambda
	switch {
	case lambda <= 0:
		lambda = 0.5
	case lambda > 1:
		lambda = 1
	}
	fetchK := opts.FetchK
	if fetchK < k {
		if fetchK = 4 * k; fetchK < 50 {
			fetchK = 50
		}
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	q := query
	if ix.cfg.Metric == Cosine {
		q = append([]float32(nil), query...)
		normalize(q)
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so the sentinel
	// surfaces instead of a silent empty result — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)

	cands := ix.searchLocked(q, fetchK, pred, nil)
	if len(cands) <= k {
		return cands, nil
	}
	return ix.mmrSelect(cands, k, float32(lambda)), nil
}

// mmrSelect mirrors hnsw.mmrSelect (reads candidate vectors from the arena under
// the held read lock).
func (ix *ivf) mmrSelect(cands []Result, k int, lambda float32) []Result {
	dist := ix.metricDist()
	n := len(cands)
	vecs := make([][]float32, n)
	for i := range cands {
		if slot, ok := ix.arena.Slot(cands[i].ID); ok {
			vecs[i] = ix.vecFor(slot)
		}
	}
	selected := make([]Result, 0, k)
	selectedVecs := make([][]float32, 0, k)
	used := make([]bool, n)
	selected = append(selected, cands[0])
	selectedVecs = append(selectedVecs, vecs[0])
	used[0] = true
	for len(selected) < k {
		best := -1
		var bestScore float32
		for i := 0; i < n; i++ {
			if used[i] || vecs[i] == nil {
				continue
			}
			relevance := -cands[i].Distance
			maxSim := -dist(vecs[i], selectedVecs[0])
			for _, sv := range selectedVecs[1:] {
				if sim := -dist(vecs[i], sv); sim > maxSim {
					maxSim = sim
				}
			}
			score := lambda*relevance - (1-lambda)*maxSim
			if best == -1 || score > bestScore {
				best, bestScore = i, score
			}
		}
		if best == -1 {
			break
		}
		selected = append(selected, cands[best])
		selectedVecs = append(selectedVecs, vecs[best])
		used[best] = true
	}
	return selected
}

// Recommend mirrors hnsw.Recommend (mean(pos)-mean(neg) target then SearchInto,
// excluding the example ids).
func (ix *ivf) Recommend(k int, opts RecommendOpts) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Positive) == 0 {
		return nil, ErrNoRecommendExamples
	}
	target := make([]float32, ix.cfg.Dim)
	ix.mu.RLock()
	// Reject on a poisoned index (failed mmap grow) before meanOf dereferences the
	// freed arena floats element-wise (vecFor's nil return would panic there) —
	// see arena.poisoned.
	if ix.arena.poisoned.Load() {
		ix.mu.RUnlock()
		return nil, ErrIndexPoisoned
	}
	nPos := ix.meanOf(target, opts.Positive)
	if nPos == 0 {
		ix.mu.RUnlock()
		return nil, ErrIDNotFound
	}
	if len(opts.Negative) > 0 {
		neg := make([]float32, ix.cfg.Dim)
		if ix.meanOf(neg, opts.Negative) > 0 {
			for i := range target {
				target[i] -= neg[i]
			}
		}
	}
	ix.mu.RUnlock()

	exclude := make(map[uint64]bool, len(opts.Positive)+len(opts.Negative))
	for _, id := range opts.Positive {
		exclude[id] = true
	}
	for _, id := range opts.Negative {
		exclude[id] = true
	}
	res, err := ix.SearchInto(nil, target, k+len(exclude), opts.Filter)
	if err != nil {
		return nil, err
	}
	out := res[:0]
	for _, r := range res {
		if exclude[r.ID] {
			continue
		}
		out = append(out, r)
		if len(out) == k {
			break
		}
	}
	return out, nil
}

// meanOf mirrors hnsw.meanOf. Must hold ix.mu.
func (ix *ivf) meanOf(dst []float32, ids []uint64) int {
	for i := range dst {
		dst[i] = 0
	}
	n := 0
	for _, id := range ids {
		slot, ok := ix.arena.Slot(id)
		if !ok {
			continue
		}
		v := ix.vecFor(slot)
		for i := range dst {
			dst[i] += v[i]
		}
		n++
	}
	if n > 0 {
		inv := 1.0 / float32(n)
		for i := range dst {
			dst[i] *= inv
		}
	}
	return n
}

// Discover mirrors hnsw.Discover (context-pair re-ranking over an IVF candidate
// pool).
func (ix *ivf) Discover(k int, opts DiscoverOpts) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Context) == 0 {
		return nil, ErrNoContextPairs
	}
	if opts.Target != nil && len(opts.Target) != ix.cfg.Dim {
		return nil, ErrDimMismatch
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	fetchK := discoverFetchK(opts.FetchK, k)

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) before the context-pair vectors
	// are materialized (vecFor returns nil, which discoverScoredLocked would
	// dereference) — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)

	pairs := make([]DiscoverPair, 0, len(opts.Context))
	for _, c := range opts.Context {
		ps, okp := ix.arena.Slot(c.Positive)
		ns, okn := ix.arena.Slot(c.Negative)
		if !okp || !okn {
			continue
		}
		pairs = append(pairs, DiscoverPair{Pos: ix.vecFor(ps), Neg: ix.vecFor(ns)})
	}
	if len(pairs) == 0 {
		return nil, ErrIDNotFound
	}

	return ix.discoverScoredLocked(k, fetchK, opts.Target, pairs, pred)
}

// DiscoverVecs is the RESOLVED-vectors discovery path (the Query API leaf form)
// for IVF — the IVF analogue of (*hnsw).DiscoverVecs. The target + context-pair
// example VECTORS are supplied directly (the coordinator resolved+embedded them),
// so no per-call id resolution happens; it runs the IDENTICAL seed + context-pair
// scorer as Discover (the equivalence oracle), score-descending.
func (ix *ivf) DiscoverVecs(k int, opts DiscoverVecsOpts) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Context) == 0 {
		return nil, ErrNoContextPairs
	}
	if opts.Target != nil && len(opts.Target) != ix.cfg.Dim {
		return nil, ErrDimMismatch
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	fetchK := discoverFetchK(opts.FetchK, k)

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so the sentinel
	// surfaces instead of a silent empty result — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)

	return ix.discoverScoredLocked(k, fetchK, opts.Target, opts.Context, pred)
}

// RecommendVecs is the BEST_SCORE recommend path (the Query API leaf form) for
// IVF — the IVF analogue of (*hnsw).RecommendVecs. The positive/negative example
// VECTORS are supplied directly (the coordinator resolved+embedded them), so no
// per-call id resolution happens; it runs the IDENTICAL positives-centroid seed +
// bestScore merge as the hnsw path (the equivalence oracle), score-descending.
func (ix *ivf) RecommendVecs(k int, opts RecommendVecsOpts) ([]Result, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Positive) == 0 {
		return nil, ErrNoRecommendExamples
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	fetchK := discoverFetchK(opts.FetchK, k)

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so the sentinel
	// surfaces instead of a silent empty result — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil, ErrIndexPoisoned
	}
	ix.searchOps.Add(1)

	return ix.recommendBestLocked(k, fetchK, opts.Positive, opts.Negative, pred)
}

// recommendBestLocked is the shared IVF BEST_SCORE core (mirror of the hnsw one):
// seed the candidate pool from the positives' centroid, search it distance-
// ascending, score each candidate by the bestScore merge, sort score-desc (pool
// distance tiebreak). Caller holds ix.mu (read) and has resolved the example
// VECTORS.
func (ix *ivf) recommendBestLocked(k, fetchK int, posVecs, negVecs [][]float32, pred Predicate) ([]Result, error) {
	dim := ix.cfg.Dim
	qbuf := make([]float32, dim)
	recommendBestSeed(qbuf, posVecs, ix.cfg.Metric)

	cands := ix.searchLocked(qbuf, fetchK, pred, nil)
	if len(cands) == 0 {
		return nil, nil
	}
	dist := ix.metricDist()
	for i := range cands {
		slot, ok := ix.arena.Slot(cands[i].ID)
		if !ok {
			continue
		}
		cv := ix.vecFor(slot)
		cands[i].Score = bestScore(cv, posVecs, negVecs, ix.cfg.Metric, dist)
	}
	sortRecommendBest(cands)
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands, nil
}

// discoverScoredLocked is the shared IVF discover core (mirror of the hnsw one):
// seed the candidate pool (target or mean of pair positives), search it
// distance-ascending, score each candidate by the context pairs, sort score-desc
// (pool distance tiebreak). Caller holds ix.mu (read) and has resolved the pair
// VECTORS — so Discover (id-form) and DiscoverVecs (resolved-vecs form) run the
// same body.
func (ix *ivf) discoverScoredLocked(k, fetchK int, target []float32, pairs []DiscoverPair, pred Predicate) ([]Result, error) {
	dim := ix.cfg.Dim
	qbuf := make([]float32, dim)
	posVecs := make([][]float32, len(pairs))
	for i, p := range pairs {
		posVecs[i] = p.Pos
	}
	discoverSeed(qbuf, target, posVecs, ix.cfg.Metric)

	cands := ix.searchLocked(qbuf, fetchK, pred, nil)
	if len(cands) == 0 {
		return nil, nil
	}
	dist := ix.metricDist()
	for i := range cands {
		slot, ok := ix.arena.Slot(cands[i].ID)
		if !ok {
			continue
		}
		cv := ix.vecFor(slot)
		cands[i].Score = discoverScore(cv, pairs, dist)
	}
	sortDiscover(cands)
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands, nil
}

// GroupCandidates returns the top-FetchK IVF candidate documents (mirror
// hnsw.GroupCandidates).
func (ix *ivf) GroupCandidates(query []float32, opts GroupOpts) ([]Document, error) {
	if len(query) != ix.cfg.Dim {
		return nil, ErrDimMismatch
	}
	if opts.GroupBy == "" {
		return nil, ErrEmptyGroupBy
	}
	fetchK := opts.FetchK
	if fetchK <= 0 {
		fetchK = 50
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	q := query
	if ix.cfg.Metric == Cosine {
		q = append([]float32(nil), query...)
		normalize(q)
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// Reject on a poisoned index (failed mmap grow) under the lock so the sentinel
	// surfaces instead of a silent empty result — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return nil, ErrIndexPoisoned
	}

	raw := ix.searchLocked(q, fetchK, pred, nil)
	docs := make([]Document, 0, len(raw))
	for _, r := range raw {
		if d, ok := ix.docForLocked(r); ok {
			docs = append(docs, d)
		}
	}
	return docs, nil
}

// SearchGroups runs a (filtered) IVF search and collapses by GroupBy (mirror
// hnsw.SearchGroups; the grouping itself reuses the shared GroupDocuments).
func (ix *ivf) SearchGroups(query []float32, k int, opts GroupOpts) ([]Group, error) {
	start := time.Now()
	defer func() { ix.searchLat.observe(time.Since(start)) }()

	if len(query) != ix.cfg.Dim {
		return nil, ErrDimMismatch
	}
	if opts.GroupBy == "" {
		return nil, ErrEmptyGroupBy
	}
	if k <= 0 {
		return nil, nil
	}
	groupSize := opts.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	if want := k * groupSize; opts.FetchK < want {
		if opts.FetchK = 4 * want; opts.FetchK < 50 {
			opts.FetchK = 50
		}
	}
	cands, err := ix.GroupCandidates(query, opts)
	if err != nil {
		return nil, err
	}
	ix.searchOps.Add(1)
	return GroupDocuments(cands, opts, k), nil
}

// ---------------------------------------------------------------------------
// doc enrichment / scan / scroll (mirror hnsw/rag.go)
// ---------------------------------------------------------------------------

// docForLocked builds a Document from arena state (mirror hnsw.docForLocked).
// Must hold ix.mu (read).
func (ix *ivf) docForLocked(r Result) (Document, bool) {
	slot, ok := ix.arena.idMap[r.ID]
	if !ok {
		return Document{}, false
	}
	d := Document{ID: r.ID, Distance: r.Distance, Score: r.Score}
	if meta := ix.liveMeta(slot, uint64(ix.now())); len(meta) > 0 {
		if cv, ok := meta[contentField]; ok && cv.Kind == ValueString {
			d.Content = cv.Str
		}
		out := make(Metadata, len(meta))
		for k, v := range meta {
			if k != contentField {
				out[k] = v
			}
		}
		if len(out) > 0 {
			d.Metadata = out
		}
	}
	return d, true
}

// fetchDocs enriches results with content + metadata (mirror hnsw.fetchDocs).
func (ix *ivf) fetchDocs(results []Result) []Document {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	docs := make([]Document, 0, len(results))
	for _, r := range results {
		if d, ok := ix.docForLocked(r); ok {
			docs = append(docs, d)
		}
	}
	return docs
}

// matchingIDs returns ids of live records satisfying pred (mirror
// hnsw.matchingIDs).
func (ix *ivf) matchingIDs(filter Filter, pred Predicate) ([]uint64, error) {
	return ix.matchingIDsAt(filter, pred, uint64(ix.now()))
}

// matchingIDsAt is matchingIDs judging admission against the caller-supplied `now`
// (unix millis), so a replicated delete-by-filter selects the SAME id set on every
// replica (#4 vector TTL determinism). Must NOT hold ix.mu.
func (ix *ivf) matchingIDsAt(filter Filter, pred Predicate, now uint64) ([]uint64, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	var ids []uint64
	limit := ix.effectiveFilterFirstLimit(ix.arena.Size())
	if cands, ok := ix.payloadIdx.candidates(filter, limit); ok && len(cands) <= limit {
		for _, slot := range cands {
			// The idMap fallback below emits every admitted id verbatim, so this
			// fast path must too — dropping id 0 here made the two lanes select
			// different id sets for the same filter (and silently under-deleted).
			if ix.admits(slot, pred, now) {
				ids = append(ids, ix.slotID(slot))
			}
		}
		return ids, nil
	}
	for id, slot := range ix.arena.idMap {
		if ix.admits(slot, pred, now) {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (ix *ivf) filterFirstThreshold() int {
	if ix.cfg.FilterFirstThreshold > 0 {
		return ix.cfg.FilterFirstThreshold
	}
	return defaultFilterFirstThreshold
}

// effectiveFilterFirstLimit folds the relative selectivity gate into the absolute
// threshold for the given live count (0 bp -> filterFirstThreshold exactly).
func (ix *ivf) effectiveFilterFirstLimit(liveCount int) int {
	return effectiveFilterFirstLimit(ix.filterFirstThreshold(), ix.cfg.FilterFirstRelativeBP, liveCount)
}

// scrollDocs enumerates live docs satisfying filter, id-ascending (mirror
// hnsw.scrollDocs).
func (ix *ivf) scrollDocs(filter Filter, limit int) ([]Document, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	docs, _, _ := ix.scrollPage(filter, pred, nil, nil, 0, 0, false, limit)
	return docs, nil
}

// scrollPage walks the live id set ascending, applying pred + the liveness gate,
// collecting up to limit docs strictly after afterID (mirror hnsw.scrollPage,
// including the warm/cold snapshot locking).
func (ix *ivf) scrollPage(filter Filter, pred Predicate, metaOf metaProvider, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	ix.mu.RLock()
	if order != nil {
		// Filter-first order narrowing (mirror hnsw.scrollPage): when a filter is
		// present and the payload index narrows it to a selective candidate SUPERSET,
		// build the value-sorted order rows over THOSE candidate slots (∩ live) FRESH
		// (never cached — the cache key is filter-independent) instead of the full
		// N-row snapshot, then collectOrderedLocked seeks + predicate-rechecks +
		// pages identically. The narrowed rows are a superset of the matches in the
		// SAME value-order, so the emitted docs / nextAfter / cursor are byte-identical
		// to the predicate-eval order page; hasMore is i+1<len(narrowedRows) (the
		// candidate superset), which can skip a trailing EMPTY page the full path would
		// emit — that page carries zero docs and is invisible on the wire (the leaf
		// discards hasMore; the coordinator derives next_cursor from len(docs)==limit).
		if rows, ok := ix.filterFirstOrderRowsLocked(filter, pred, metaOf, order); ok {
			docs, nextAfter, hasMore = ix.collectOrderedLocked(rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
			ix.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		if snap := ix.orderSnapWarmLocked(order); snap != nil {
			docs, nextAfter, hasMore = ix.collectOrderedLocked(snap.rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
			ix.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
	} else if ix.scrollSnap.ver == ix.idSetVersion {
		// Filter-first narrowing (id-ascending path only): walk the payload-index
		// candidate SUPERSET instead of the full snapshot when index-narrowable +
		// selective. hasMore computed against the FULL snapshot so the page is
		// byte-identical to the predicate-eval walk (mirror hnsw.scrollPage).
		if cands, ok := ix.filterFirstScrollCandsLocked(filter, pred, metaOf); ok {
			docs, nextAfter, hasMore = ix.walkScrollNarrowedLocked(cands, ix.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
			ix.mu.RUnlock()
			return docs, nextAfter, hasMore
		}
		docs, nextAfter, hasMore = ix.walkScrollLocked(ix.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
		ix.mu.RUnlock()
		return docs, nextAfter, hasMore
	}
	ix.mu.RUnlock()

	ix.mu.Lock()
	defer ix.mu.Unlock()
	if order != nil {
		// Filter-first order narrowing (cold path): build narrowed rows fresh; else the
		// cached full snapshot. See the warm-path comment above for the byte-identity +
		// hasMore reconciliation.
		if rows, ok := ix.filterFirstOrderRowsLocked(filter, pred, metaOf, order); ok {
			return ix.collectOrderedLocked(rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
		}
		snap := ix.orderSnapLocked(order)
		return ix.collectOrderedLocked(snap.rows, pred, metaOf, order, afterID, afterKey, hasAfter, limit)
	}
	if ix.scrollSnap.ver != ix.idSetVersion {
		ix.rebuildScrollSnapLocked()
	}
	if cands, ok := ix.filterFirstScrollCandsLocked(filter, pred, metaOf); ok {
		return ix.walkScrollNarrowedLocked(cands, ix.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
	}
	return ix.walkScrollLocked(ix.scrollSnap.ids, pred, metaOf, afterID, hasAfter, limit)
}

// orderSnapWarmLocked mirrors hnsw.orderSnapWarmLocked: the cached (field, direction)
// snapshot IF fresh at the current dataVersion, else nil. Must hold ix.mu (R or W).
func (ix *ivf) orderSnapWarmLocked(order *OrderBy) *orderSnap {
	if snap, ok := ix.orderSnaps[orderSnapCacheKey(order)]; ok && snap.ver == ix.dataVersion {
		return snap
	}
	return nil
}

// orderSnapLocked mirrors hnsw.orderSnapLocked: a fresh (field, direction) snapshot,
// rebuilt if stale/absent (double-checked), bounded by orderCacheCap with
// oldest-built eviction. FILTER-INDEPENDENT (the walk applies filter + TTL). Must
// hold ix.mu (WRITE).
func (ix *ivf) orderSnapLocked(order *OrderBy) *orderSnap {
	key := orderSnapCacheKey(order)
	if snap, ok := ix.orderSnaps[key]; ok && snap.ver == ix.dataVersion {
		return snap
	}
	rows := ix.buildOrderRowsLocked(order)
	ix.orderSeq++
	snap := &orderSnap{ver: ix.dataVersion, seq: ix.orderSeq, rows: rows}
	ix.orderSnaps[key] = snap
	if len(ix.orderSnaps) > orderCacheCap {
		evictOldestOrderSnap(ix.orderSnaps)
	}
	ix.orderRebuilds++
	return snap
}

// buildOrderRowsLocked mirrors hnsw.buildOrderRowsLocked: every LIVE (non-tombstoned)
// id that HAS the order field, sorted by (value, id). TTL-lazy + filter-independent
// (the walk's gate handles both). Must hold ix.mu (WRITE).
func (ix *ivf) buildOrderRowsLocked(order *OrderBy) []OrderedID {
	now := uint64(ix.now())
	rows := make([]OrderedID, 0, len(ix.arena.idMap))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for id, slot := range ix.arena.idMap {
			if ix.tombstoned[slot] {
				continue
			}
			meta := ix.liveMeta(slot, now)
			vals, ok := orderTupleKeys(meta, keys)
			if !ok {
				continue // EXCLUDE: some order key absent or wrong-type
			}
			rows = append(rows, OrderedID{ID: id, Keys: vals})
		}
		SortOrderedIDsTuple(rows, keys)
		return rows
	}
	str := order.Kind == OrderString
	for id, slot := range ix.arena.idMap {
		if ix.tombstoned[slot] {
			continue
		}
		meta := ix.liveMeta(slot, now)
		if str {
			sk, kok := OrderStringKey(meta, order.Key)
			if !kok {
				continue // EXCLUDE: order field absent or non-string
			}
			rows = append(rows, OrderedID{StrKey: sk, ID: id})
			continue
		}
		key, kok := OrderKey(meta, order.Key, order.IsDatetime)
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

// filterFirstOrderRowsLocked mirrors hnsw.filterFirstOrderRowsLocked: the value-sorted
// order rows built ONLY over the payload-index candidate SUPERSET (∩ live) for filter,
// or (nil, false) when filter-first order narrowing does not apply (no filter, provider
// path, non-accelerable, or non-selective). The candidate slots are a superset of the
// matching slots; the per-row field-presence EXCLUDE here + the per-row predicate recheck
// in collectOrderedLocked make the narrowed rows EXACTLY the field-present matches in the
// SAME value-order as the full snapshot, so the page is byte-identical (docs / nextAfter
// / cursor). NOT cached (the orderSnaps cache key is filter-independent; a narrowed
// snapshot must never be stored there or a different/no-filter scroll would read it).
// Must hold ix.mu (R or W).
func (ix *ivf) filterFirstOrderRowsLocked(filter Filter, pred Predicate, metaOf metaProvider, order *OrderBy) ([]OrderedID, bool) {
	if pred == nil || metaOf != nil {
		return nil, false // no filter / provider path -> full snapshot unchanged
	}
	threshold := ix.effectiveFilterFirstLimit(ix.arena.Size())
	slots, ok := ix.payloadIdx.candidates(filter, threshold)
	if !ok || len(slots) > threshold {
		return nil, false
	}
	now := uint64(ix.now())
	rows := make([]OrderedID, 0, len(slots))
	if isMultiKey(order) {
		keys := orderKeyList(order)
		for _, slot := range slots {
			if ix.tombstoned[slot] {
				continue
			}
			meta := ix.liveMeta(slot, now)
			vals, vok := orderTupleKeys(meta, keys)
			if !vok {
				continue // EXCLUDE: some order key absent or wrong-type
			}
			rows = append(rows, OrderedID{ID: ix.slotID(slot), Keys: vals})
		}
		SortOrderedIDsTuple(rows, keys)
		return rows, true
	}
	str := order.Kind == OrderString
	for _, slot := range slots {
		if ix.tombstoned[slot] {
			continue
		}
		meta := ix.liveMeta(slot, now)
		if str {
			sk, kok := OrderStringKey(meta, order.Key)
			if !kok {
				continue // EXCLUDE: order field absent or non-string
			}
			rows = append(rows, OrderedID{StrKey: sk, ID: ix.slotID(slot)})
			continue
		}
		key, kok := OrderKey(meta, order.Key, order.IsDatetime)
		if !kok {
			continue // EXCLUDE: order field absent or non-numeric
		}
		rows = append(rows, OrderedID{Key: key, ID: ix.slotID(slot)})
	}
	if str {
		SortOrderedIDsStr(rows, order.Desc)
	} else {
		SortOrderedIDs(rows, order.Desc)
	}
	return rows, true
}

// collectOrderedLocked mirrors hnsw.collectOrderedLocked: seek the cached sorted rows
// past the cursor / start_from, then walk applying the TTL/tombstone gate + per-query
// filter (the snapshot is filter-independent + TTL-lazy), materializing up to limit
// Documents. rows is immutable. Must hold ix.mu (R or W).
func (ix *ivf) collectOrderedLocked(rows []OrderedID, pred Predicate, metaOf metaProvider, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	now := uint64(ix.now())
	start := orderSeekStart(rows, order, afterID, afterKey, hasAfter)
	for i := start; i < len(rows); i++ {
		id := rows[i].ID
		slot, ok := ix.arena.idMap[id]
		if !ok {
			continue // reclaimed since the snapshot was built (benign)
		}
		if ix.tombstoned[slot] || ix.isExpired(slot) {
			continue
		}
		if pred != nil {
			var meta Metadata
			if metaOf != nil {
				meta = metaOf(ix.arena.ID(slot))
			} else {
				meta = ix.liveMeta(slot, now)
			}
			if !pred(meta) {
				ix.filterRejects.Add(1)
				continue
			}
		}
		d, dok := ix.docForLocked(Result{ID: id})
		if !dok {
			continue
		}
		docs = append(docs, d)
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			hasMore = i+1 < len(rows)
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// bumpData mirrors hnsw.bumpData: advance dataVersion to invalidate cached order
// snapshots. Called under ix.mu (WRITE) at every id-set + payload mutation.
func (ix *ivf) bumpData() { ix.dataVersion++ }

func (ix *ivf) rebuildScrollSnapLocked() {
	ids := make([]uint64, 0, len(ix.arena.idMap))
	for id, slot := range ix.arena.idMap {
		if ix.tombstoned[slot] {
			continue
		}
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	ix.scrollSnap.ids = ids
	ix.scrollSnap.ver = ix.idSetVersion
	ix.scrollRebuilds++
}

// admitsWith mirrors hnsw.admitsWith (predicate against an external payload
// provider when metaOf != nil; arena metadata otherwise). Must hold ix.mu.
func (ix *ivf) admitsWith(slot uint32, pred Predicate, metaOf metaProvider, now uint64) bool {
	if ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return false
	}
	if pred != nil {
		var meta Metadata
		if metaOf != nil {
			meta = metaOf(ix.arena.ID(slot))
		} else {
			meta = ix.liveMeta(slot, now)
		}
		if !pred(meta) {
			ix.filterRejects.Add(1)
			return false
		}
	}
	return true
}

func (ix *ivf) walkScrollLocked(ids []uint64, pred Predicate, metaOf metaProvider, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	now := uint64(ix.now()) // one clock read for the whole scroll walk
	start := 0
	if hasAfter {
		start = sort.Search(len(ids), func(i int) bool { return ids[i] > afterID })
	}
	for i := start; i < len(ids); i++ {
		id := ids[i]
		slot, ok := ix.arena.idMap[id]
		if !ok {
			continue
		}
		if !ix.admitsWith(slot, pred, metaOf, now) {
			continue
		}
		d, dok := ix.docForLocked(Result{ID: id})
		if !dok {
			continue
		}
		docs = append(docs, d)
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			hasMore = i+1 < len(ids)
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// filterFirstScrollCandsLocked is the IVF analogue of hnsw.filterFirstScrollCandsLocked:
// the id-ASCENDING candidate superset for filter, or (nil, false) when filter-first
// does not apply (no filter, provider path, non-accelerable, or non-selective). The
// narrowed walk's recheck drops over-cover. Must hold ix.mu (read).
func (ix *ivf) filterFirstScrollCandsLocked(filter Filter, pred Predicate, metaOf metaProvider) ([]uint64, bool) {
	if pred == nil || metaOf != nil {
		return nil, false
	}
	threshold := ix.effectiveFilterFirstLimit(ix.arena.Size())
	slots, ok := ix.payloadIdx.candidates(filter, threshold)
	if !ok || len(slots) > threshold {
		return nil, false
	}
	ids := make([]uint64, 0, len(slots))
	for _, slot := range slots {
		ids = append(ids, ix.arena.ID(slot))
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids, true
}

// walkScrollNarrowedLocked is the IVF analogue of hnsw.walkScrollNarrowedLocked: emit
// from the id-sorted candidate SUPERSET with the SAME admitsWith recheck, computing
// hasMore against the FULL snapshot (fullIDs) so the boundary matches walkScrollLocked
// EXACTLY (a filtered page can be followed by a trailing empty page in BOTH paths).
// Must hold ix.mu.
func (ix *ivf) walkScrollNarrowedLocked(cands, fullIDs []uint64, pred Predicate, metaOf metaProvider, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool) {
	now := uint64(ix.now()) // one clock read for the whole scroll walk
	start := 0
	if hasAfter {
		start = sort.Search(len(cands), func(i int) bool { return cands[i] > afterID })
	}
	for i := start; i < len(cands); i++ {
		id := cands[i]
		slot, ok := ix.arena.idMap[id]
		if !ok {
			continue
		}
		if !ix.admitsWith(slot, pred, metaOf, now) {
			continue
		}
		d, dok := ix.docForLocked(Result{ID: id})
		if !dok {
			continue
		}
		docs = append(docs, d)
		nextAfter = id
		if limit > 0 && len(docs) >= limit {
			hasMore = moreBeyond(fullIDs, nextAfter)
			return docs, nextAfter, hasMore
		}
	}
	return docs, nextAfter, false
}

// scanVectors enumerates every live record as a ScanRecord (mirror
// hnsw.scanVectors).
func (ix *ivf) scanVectors() []ScanRecord {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	now := uint64(ix.now())
	recs := make([]ScanRecord, 0, len(ix.arena.idMap))
	for id, slot := range ix.arena.idMap {
		if !ix.admits(slot, nil, now) {
			continue
		}
		rec := ScanRecord{
			ID:      id,
			Vec:     append([]float32(nil), ix.vecFor(slot)...),
			Version: ix.arena.Version(slot),
		}
		if exp := ix.arena.ExpiresAt(slot); exp > now {
			rec.TTL = time.Duration(exp-now) * time.Millisecond
		}
		if meta := ix.liveMeta(slot, now); len(meta) > 0 {
			out := make(Metadata, len(meta))
			for k, v := range meta {
				out[k] = v
			}
			rec.Metadata = out
		}
		if sv := ix.arena.Sparse(slot); sv != nil {
			rec.Sparse = sv.Clone()
		}
		recs = append(recs, rec)
	}
	return recs
}

// ---------------------------------------------------------------------------
// payload mutations (mirror hnsw)
// ---------------------------------------------------------------------------

func (ix *ivf) SetPayload(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return ix.setPayloadBody(id, patch, keyTTLMs, cas, false, 0)
}

// SetPayloadAt is SetPayload judging the per-key deadline computation and the
// dead-point liveness gate against the EXPLICIT leader-stamped clock nowMs (#4
// vector TTL determinism, mirroring hnsw.SetPayloadAt).
func (ix *ivf) SetPayloadAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return ix.setPayloadBody(id, patch, keyTTLMs, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) setPayloadBody(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	if err := checkRecordValues(patch); err != nil {
		return nil, nil, 0, err
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(ix.arena.Version(slot)); err != nil {
		return nil, nil, 0, err
	}
	merged := cloneMeta(ix.arena.Metadata(slot))
	if len(patch) > 0 {
		if merged == nil {
			merged = make(Metadata, len(patch))
		}
		for k, v := range patch {
			merged[k] = v
		}
	}
	ke := cloneKeyExpires(ix.arena.KeyExpires(slot))
	if len(keyTTLMs) > 0 {
		for k, ttlMs := range keyTTLMs {
			if _, present := merged[k]; !present {
				continue
			}
			if ttlMs > 0 {
				if ke == nil {
					ke = make(map[string]uint64, len(keyTTLMs))
				}
				ke[k] = now + uint64(ttlMs)
			} else if ke != nil {
				delete(ke, k)
			}
		}
	}
	ke = pruneKeyExpires(ke, merged)
	ix.arena.SetMetadata(slot, merged)
	ix.arena.SetKeyExpires(slot, ke)
	ix.payloadIdx.reindex(slot, merged)
	ix.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return merged, ke, ix.arena.BumpVersion(slot), nil
}

func (ix *ivf) OverwritePayload(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return ix.overwritePayloadBody(id, meta, keyTTLMs, cas, false, 0)
}

// OverwritePayloadAt is OverwritePayload judging the per-key deadline computation and
// the dead-point liveness gate against the EXPLICIT leader-stamped clock nowMs (#4
// vector TTL determinism).
func (ix *ivf) OverwritePayloadAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return ix.overwritePayloadBody(id, meta, keyTTLMs, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) overwritePayloadBody(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	if err := checkRecordValues(meta); err != nil {
		return nil, nil, 0, err
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(ix.arena.Version(slot)); err != nil {
		return nil, nil, 0, err
	}
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
	ix.arena.SetMetadata(slot, newMeta)
	ix.arena.SetKeyExpires(slot, ke)
	ix.payloadIdx.reindex(slot, newMeta)
	ix.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return newMeta, ke, ix.arena.BumpVersion(slot), nil
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
//
// THE KEY'S DEADLINE IS NOT CONSULTED HERE. Unstamped, `now` is this process's
// own wall clock, and letting it decide whether the record EXISTS would let two
// replicas store different bytes. So an unstamped mutation treats a
// deadline-passed record as PRESENT and passes its deadline through untouched;
// only the *At variant expires the key. See recordKeyPastDeadline
// (vector/record_mutate.go).
func (ix *ivf) MutatePayloadRecord(id uint64, key string, fn RecordMutator, cas CASCond) (Metadata, map[string]uint64, uint64, bool, error) {
	return ix.mutatePayloadRecordBody(id, key, fn, cas, false, 0)
}

// MutatePayloadRecordAt judges the dead-point liveness gate, the per-key
// deadline check and the stale-deadline drop against the EXPLICIT leader-stamped
// clock nowMs, so a replicated record mutation sees the same record and produces
// the same bytes on every replica (#4 vector TTL determinism, mirroring
// SetPayloadAt).
//
// This is the ONLY variant that may expire the payload key: its clock is the
// leader's stamp, identical on every replica. See recordKeyPastDeadline
// (vector/record_mutate.go) for why the unstamped twin must not.
func (ix *ivf) MutatePayloadRecordAt(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, bool, error) {
	return ix.mutatePayloadRecordBody(id, key, fn, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

// LOCKSTEP with vector/hnsw.go mutatePayloadRecordBody and the named/multi-vector
// mutatePayloadRecordLockedAt in vector/named.go and vector/multivector.go.
// These four bodies are ~110 near-identical lines each and are deliberately NOT
// factored into one: that matches the house style of their set_payload /
// overwrite_payload siblings, whose engine differences (slot- vs id-keyed
// indexes, uint64 vs int64 deadlines, BM25 on dense only) are real and are what
// a shared body would have to branch on.
//
// The cost is real too, and it has already been paid once: a single formatting
// difference in the size-bound message landed as a defect in all four at the
// same time, and the deadline rule below had to be corrected in all four at the
// same time. So both of those now live in SHARED helpers — recordTooLargeErr
// (vector/metadata.go) and recordKeyPastDeadline[Abs] (vector/record_mutate.go)
// — and any further change to the RULES, as opposed to the plumbing, belongs in
// a helper rather than in a fifth copy. A change here that is not mirrored in
// the other three is a bug, not a variation.
func (ix *ivf) mutatePayloadRecordBody(id uint64, key string, fn RecordMutator, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, bool, error) {
	if fn == nil || key == "" || key == contentField {
		return nil, nil, 0, false, ErrPayloadKeyNotRecord
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return nil, nil, 0, false, ErrIDNotFound
	}
	if err := cas.check(ix.arena.Version(slot)); err != nil {
		return nil, nil, 0, false, err // CAS mismatch: no mutation, no bump
	}
	oldMeta := ix.arena.Metadata(slot)
	oldKE := ix.arena.KeyExpires(slot)
	// Only a STAMPED apply may judge the key's deadline; an unstamped one treats
	// the record as PRESENT and leaves the deadline alone. recordKeyPastDeadline
	// (vector/record_mutate.go) carries the reasoning — the short version is that
	// a per-replica wall clock deciding whether a record EXISTS makes replicas
	// store different bytes, which is not the bounded-deadline-skew class
	// setPayloadBody lives with. LOCKSTEP: the same two lines exist in
	// vector/hnsw.go, vector/ivf.go, vector/named.go and vector/multivector.go.
	keyPastDeadline := recordKeyPastDeadline(stamped, oldKE[key], now)
	old, exists, err := currentRecordValue(oldMeta, key, keyPastDeadline)
	if err != nil {
		return nil, nil, 0, false, err
	}
	rec, act, err := fn(old, exists)
	if err != nil {
		return nil, nil, 0, false, err
	}
	switch act {
	case RecordUnchanged:
		return nil, nil, ix.arena.Version(slot), false, nil
	case RecordDelete:
		if !exists {
			return nil, nil, ix.arena.Version(slot), false, nil
		}
	case RecordStore:
		// The POST-mutation bound. checkRecordValues (vector/metadata.go) is the
		// same cap on the ingest side, but it only ever sees a caller's patch —
		// it cannot reach bytes the operate engine just produced, which is what
		// this site exists for. Keep the two in step.
		//
		// SIZE FIRST, THEN SHAPE — the same order and the same two gates
		// checkRecordValues applies to a caller's patch, so an oversize value is
		// refused without ever being decoded.
		//
		// The shape check is here because MutatePayloadRecordCAS is EXPORTED. The
		// mutator ops supplies returns applyRecordBytes' output, which phase 1
		// held to the tree oracle and which cannot be malformed — but a direct Go
		// caller of this API supplies its own mutator, and whatever bytes it
		// returns were being stored on the size cap alone. That poisons the
		// payload key for the whole collection: every path under it declines to
		// accelerate, and a later vector_operate on the same key fails with
		// wire.ErrOperateRecord. The cost is one decode of the bytes the engine
		// just produced, on a write that already decoded and re-encoded the
		// record.
		//
		// Both MESSAGES come from the one producer of their detailed form
		// (recordTooLargeErr / recordMalformedErr, vector/metadata.go): one
		// canonical shape for the transport matchers to anchor on across
		// replication, with the caller's key bounded.
		if len(rec) > maxRecordValueBytes {
			return nil, nil, 0, false, recordTooLargeErr(key, len(rec))
		}
		if verr := validateRecord(rec); verr != nil {
			return nil, nil, 0, false, recordMalformedErr(key, verr)
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
	// A key read as ABSENT above because its deadline had passed — STAMPED
	// applies only: its stale deadline must go, or the record this call just
	// created would be invisible the instant it is written. pruneKeyExpires only
	// drops deadlines for keys no longer present. On an unstamped apply
	// keyPastDeadline is false by construction, so an expired key's deadline is
	// passed through UNTOUCHED, exactly as setPayloadBody passes an expired key
	// through unchanged.
	if ke != nil && keyPastDeadline {
		delete(ke, key)
	}
	ke = pruneKeyExpires(ke, merged)
	ix.arena.SetMetadata(slot, merged)
	ix.arena.SetKeyExpires(slot, ke)
	ix.payloadIdx.reindex(slot, merged)
	ix.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return merged, ke, ix.arena.BumpVersion(slot), true, nil
}

func (ix *ivf) DeletePayloadKeys(id uint64, keys []string, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return ix.deletePayloadKeysBody(id, keys, cas, false, 0)
}

// DeletePayloadKeysAt is DeletePayloadKeys judging the dead-point liveness gate
// against the EXPLICIT leader-stamped clock nowMs (#4 vector TTL determinism).
func (ix *ivf) DeletePayloadKeysAt(id uint64, keys []string, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return ix.deletePayloadKeysBody(id, keys, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) deletePayloadKeysBody(id uint64, keys []string, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(ix.arena.Version(slot)); err != nil {
		return nil, nil, 0, err
	}
	newMeta := cloneMeta(ix.arena.Metadata(slot))
	ke := cloneKeyExpires(ix.arena.KeyExpires(slot))
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
	ix.arena.SetMetadata(slot, newMeta)
	ix.arena.SetKeyExpires(slot, ke)
	ix.payloadIdx.reindex(slot, newMeta)
	ix.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return newMeta, ke, ix.arena.BumpVersion(slot), nil
}

func (ix *ivf) ClearPayload(id uint64, cas CASCond) (Metadata, map[string]uint64, uint64, error) {
	return ix.clearPayloadBody(id, cas, false, 0)
}

// ClearPayloadAt is ClearPayload judging the dead-point liveness gate against the
// EXPLICIT leader-stamped clock nowMs (#4 vector TTL determinism).
func (ix *ivf) ClearPayloadAt(id uint64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error) {
	return ix.clearPayloadBody(id, cas, true, uint64(nowMs)) //nolint:gosec // stamped unix-millis is non-negative
}

func (ix *ivf) clearPayloadBody(id uint64, cas CASCond, stamped bool, nowMs uint64) (Metadata, map[string]uint64, uint64, error) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	now := nowMs
	if !stamped {
		now = uint64(ix.now())
	}
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpiredAt(slot, now) {
		return nil, nil, 0, ErrIDNotFound
	}
	if err := cas.check(ix.arena.Version(slot)); err != nil {
		return nil, nil, 0, err
	}
	ix.arena.SetMetadata(slot, nil)
	ix.arena.SetKeyExpires(slot, nil)
	ix.payloadIdx.reindex(slot, nil)
	ix.bumpData() // payload-value change: invalidate the order_by snapshot (NOT idSetVersion)
	return nil, nil, ix.arena.BumpVersion(slot), nil
}

func (ix *ivf) RestorePayload(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) error {
	// REPLAY body: size only — the IVF twin of hnsw.RestorePayload. See
	// checkRecordValuesSize.
	if err := checkRecordValuesSize(meta); err != nil {
		return err
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	slot, ok := ix.arena.Slot(id)
	if !ok || ix.tombstoned[slot] || ix.isExpired(slot) {
		return ErrIDNotFound
	}
	ix.arena.SetMetadata(slot, meta)
	ix.arena.SetKeyExpires(slot, pruneKeyExpires(cloneKeyExpires(keyExpires), meta))
	ix.payloadIdx.reindex(slot, meta)
	if version != 0 {
		ix.arena.SetVersion(slot, version)
	}
	ix.bumpData() // WAL-replayed payload-value change: invalidate the order_by snapshot
	return nil
}

// ---------------------------------------------------------------------------
// maintenance (mirror hnsw)
// ---------------------------------------------------------------------------

// sweepOnce tombstones point-TTL-expired slots and physically drops expired
// per-key payload entries (mirror hnsw.sweepOnce). Returns the number of POINTS
// tombstoned.
func (ix *ivf) sweepOnce() int {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	now := uint64(ix.now())
	swept := 0
	keysSwept := 0
	for _, slot := range ix.arena.idMap {
		exp := ix.arena.expires[slot]
		if exp != 0 && exp <= now {
			if ix.tombstoned[slot] {
				continue
			}
			ix.tombstoned[slot] = true
			ix.expiredCount.Add(1)
			swept++
			continue
		}
		if ix.tombstoned[slot] {
			continue
		}
		ke := ix.arena.KeyExpires(slot)
		if len(ke) == 0 {
			continue
		}
		meta := ix.arena.Metadata(slot)
		var expiredAny bool
		for k, dl := range ke {
			if keyExpired(dl, now) {
				if _, present := meta[k]; present {
					expiredAny = true
					break
				}
			}
		}
		if !expiredAny {
			continue
		}
		newMeta := cloneMeta(meta)
		ke = cloneKeyExpires(ke)
		for k, dl := range ke {
			if keyExpired(dl, now) {
				delete(newMeta, k)
				keysSwept++
			}
		}
		if len(newMeta) == 0 {
			newMeta = nil
		}
		ke = pruneKeyExpires(ke, newMeta)
		ix.arena.SetMetadata(slot, newMeta)
		ix.arena.SetKeyExpires(slot, ke)
		ix.payloadIdx.reindex(slot, newMeta)
	}
	if keysSwept > 0 {
		ix.keysSwept.Add(uint64(keysSwept))
	}
	if swept > 0 {
		ix.idSetVersion++
		ix.bumpData() // id-set change also invalidates the order_by snapshot
	}
	return swept
}

// Reclaim physically removes tombstoned slots from the arena and rebuilds the
// derived indexes + inverted lists (mirror hnsw.Reclaim, minus the graph-edge
// compaction, which IVF has no analogue for — its lists are rebuilt instead).
func (ix *ivf) Reclaim() int {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	if len(ix.tombstoned) == 0 {
		return 0
	}
	deletedSlots := make(map[uint32]bool, len(ix.tombstoned))
	for slot := range ix.tombstoned {
		deletedSlots[slot] = true
	}
	idsToDelete := make([]uint64, 0, len(deletedSlots))
	for id, slot := range ix.arena.idMap {
		if deletedSlots[slot] {
			idsToDelete = append(idsToDelete, id)
		}
	}
	for _, id := range idsToDelete {
		ix.arena.Delete(id)
	}
	count := len(deletedSlots)
	ix.tombstoned = make(map[uint32]bool)
	ix.idSetVersion++
	ix.bumpData() // id-set change also invalidates the order_by snapshot

	ix.bytesUsed -= estimateInsertBytes(ix.cfg.Dim, ix.cfg.M) * int64(count)
	if ix.bytesUsed < 0 {
		ix.bytesUsed = 0
	}
	if ix.sparseIdx != nil {
		ix.sparseIdx.rebuild(ix.arena, ix.tombstoned)
	}
	if ix.payloadIdx != nil {
		ix.payloadIdx.rebuild(ix.arena)
	}
	// Rebuild the inverted lists so reclaimed (now freed/reusable) slots no longer
	// linger in any cell. Cheap: a single nearest-centroid pass over the live set.
	if ix.trained {
		ix.rebuildListsLocked()
	}
	return count
}

// ---------------------------------------------------------------------------
// persistence (snapshot-only v1)
// ---------------------------------------------------------------------------

const (
	// ivfSnapshotMagic identifies an IVF-Flat snapshot, distinct from the hnsw
	// magic ("rstmhsw1") so the two formats can never be confused on restore.
	ivfSnapshotMagic = "rstmivf1"
	// ivfSnapshotVer 1 = original; v2 appends an arena.versions block after the
	// keyExpires block (per-point CAS version). v1 snapshots load with live points
	// defaulted to version 1 (a sane default for an existing point — see readArena).
	// v3 = IVF-PQ: the arena block carries a leading hasVecs byte (0 = floats
	// dropped, PQ-only) + an explicit codes block (verbatim, NOT re-encoded from
	// vecs — residual codes can't be re-derived without the centroid), and a PQ
	// trailer (codebooks + m + dropped + slotCell) after the lists block. Old
	// rstmivf1 v1/v2 snapshots (no PQ block) restore as IVF-Flat (pq == nil).
	// v4 appends the drift-retrain checkpoint (lastTrainCount + lastTrainCost) to the
	// IVF core block. v1/v2/v3 snapshots (no checkpoint) restore both as 0
	// (drift-retrain re-arms on the next train — back-compat).
	// Snapshot writes v4 ONLY when the drift feature is active (cfg.IVFDriftRetrain
	// or a non-zero factor); otherwise it writes v3 so the snapshot is byte-identical
	// to the pre-feature format and old readers can still open it (rolling-restart
	// safety). See ivfDriftActive.
	// v5 appends the SOAR secondary-assignment block (cellOf2 + per-slot code2) to the
	// IVF core block AFTER the (optional) drift checkpoint, so a SOAR IVF restores its
	// MULTI-ASSIGNMENT verbatim (both list memberships rebuilt). Written ONLY when the
	// SOAR feature is active (ix.soarTrained); v1..v4 restore cellOf2/code2 empty (a
	// non-SOAR index — byte-identical). See ivfSOARPersist.
	ivfSnapshotVer = uint32(5)
)

// ivfDriftActive reports whether the drift-retrain feature is active for cfg —
// i.e., at least one of the three drift knobs is set to a non-default value. Used
// to decide whether to write a v4 snapshot / v2 sidecar (with the drift checkpoint)
// or the older v3/v1 format (byte-identical to the pre-feature layout so an old
// binary can still read it). A false result means all three fields are at their
// zero/false default, so the drift path is never entered and no checkpoint state
// needs to be persisted.
func ivfDriftActive(cfg Config) bool {
	return cfg.IVFDriftRetrain || cfg.IVFDriftGrowthFactor != 0 || cfg.IVFDriftFactor != 0
}

// ivfSOARActive reports whether the index actually built a SOAR secondary
// assignment (soarTrained), so the snapshot/sidecar writer appends the SOAR block
// and selects the SOAR-aware format version. Keyed on the BUILT state (not just
// cfg.SOAR) so an untrained or nlist==1 SOAR collection — which has no secondary
// membership to persist — still writes the byte-identical pre-SOAR format. Caller
// holds the read lock.
func (ix *ivf) ivfSOARActive() bool { return ix.soarTrained }

// Snapshot serializes the ivf to w under the read lock. Format: the magic +
// version header, the config scalars, the full arena (vecs + expires + metadata
// + sparse + keyExpires + free + idMap), the tombstone set, and the IVF state
// (trained, nprobe, centroids, lists). The payload + sparse indexes are derived
// (rebuilt on restore), not serialized — same as hnsw.
//
// VERSION SELECTION: when the drift-retrain feature is active (any of the three
// drift knobs set), writes v4 (with the drift checkpoint). When the feature is OFF
// (all defaults), writes v3 — byte-identical to the pre-feature format so an old
// binary can open the snapshot during a rolling upgrade. This ensures true
// default-OFF byte-identical snapshots and preserves rolling-restart safety.
func (ix *ivf) Snapshot(w io.Writer) error {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	// A failed mmap slab growth freed the arena floats; refuse to serialize rather
	// than emit a silently truncated snapshot. Called from the raft FSM snapshot
	// and backup paths (goroutines with no panic recovery), so checked under ix.mu
	// (closes the TOCTOU with a concurrent grow) — see arena.poisoned.
	if ix.arena.poisoned.Load() {
		return ErrIndexPoisoned
	}
	bw := bufio.NewWriter(w)

	// Choose snapshot version: v5 (SOAR block) when SOAR built a secondary
	// assignment; v4 (drift checkpoint) when only the drift feature is active; v3
	// otherwise (pre-feature format, old-reader compatible). At v5 the drift
	// checkpoint is ALWAYS written too (so the on-disk layout is unambiguous —
	// readIVFCore reads the checkpoint at version>=4 and the SOAR block at
	// version>=5); the checkpoint bytes are benign when drift is off.
	driftOn := ivfDriftActive(ix.cfg)
	soarOn := ix.ivfSOARActive()
	snapVer := uint32(3)
	switch {
	case soarOn:
		snapVer = uint32(5)
	case driftOn:
		snapVer = uint32(4)
	}
	// At v5 the core block carries the drift checkpoint regardless of driftOn.
	writeDrift := driftOn || soarOn

	if _, err := bw.WriteString(ivfSnapshotMagic); err != nil {
		return err
	}
	if err := writeU32(bw, snapVer); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(ix.cfg.Dim)); err != nil {
		return err
	}
	if err := bw.WriteByte(byte(ix.cfg.Metric)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(ix.cfg.M)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(ix.cfg.EfConstruction)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(ix.cfg.EfSearch)); err != nil {
		return err
	}
	if err := writeI64(bw, ix.cfg.Seed); err != nil {
		return err
	}

	if err := ix.writeArena(bw, false); err != nil {
		return err
	}

	// Tombstones + IVF state (trained/nprobe/centroids/lists). v4+ appends the drift
	// checkpoint; v5 also appends the SOAR secondary-assignment block. Shared with the
	// mmap sidecar (ivf_persist.go) so the two formats stay in lockstep.
	if err := ix.writeIVFCore(bw, writeDrift, soarOn); err != nil {
		return err
	}

	// v3 PQ trailer. pqOn byte; when 1: m, the dropped byte, the residual
	// codebooks (m × len(codebook[s]) × dsub floats, each subspace length-prefixed
	// since small-n training can yield < 256 sub-centroids), and the slotCell array
	// (len + u32 per slot). pqOn 0 = IVF-Flat (no PQ); the trailer is a single byte.
	if err := ix.writePQTrailer(bw); err != nil {
		return err
	}
	return bw.Flush()
}

// writePQTrailer serializes the IVF-PQ codec state (codebooks + m + dropped +
// slotCell) after the lists block. A nil ix.pq writes a single 0 byte (IVF-Flat).
func (ix *ivf) writePQTrailer(bw *bufio.Writer) error {
	if ix.pq == nil {
		return bw.WriteByte(0)
	}
	if err := bw.WriteByte(1); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(ix.pq.m)); err != nil {
		return err
	}
	dropped := byte(0)
	if ix.pqDropped {
		dropped = 1
	}
	if err := bw.WriteByte(dropped); err != nil {
		return err
	}
	for s := 0; s < ix.pq.m; s++ {
		cb := ix.pq.codebooks[s]
		if err := writeU32(bw, uint32(len(cb))); err != nil {
			return err
		}
		for _, sub := range cb {
			for _, v := range sub {
				if err := writeF32(bw, v); err != nil {
					return err
				}
			}
		}
	}
	if err := writeU32(bw, uint32(len(ix.slotCell))); err != nil {
		return err
	}
	for _, c := range ix.slotCell {
		if err := writeU32(bw, c); err != nil {
			return err
		}
	}
	// OPQ rotation R (dim×dim floats), VERBATIM. Written ONLY when present so a
	// non-OPQ trailer is BYTE-IDENTICAL to before (no presence byte, no R). When
	// present: a 1 byte then dim×dim floats. The trailer is the last block in the
	// IVF snapshot, so readPQTrailer treats EOF here (old/no-R blobs) as "no R"
	// (rotation nil) — back-compat without a version bump.
	if ix.pq.rotation != nil {
		if err := bw.WriteByte(1); err != nil {
			return err
		}
		for _, v := range ix.pq.rotation {
			if err := writeF32(bw, v); err != nil {
				return err
			}
		}
	}
	return nil
}

// readPQTrailer reads the IVF-PQ codec state written by writePQTrailer. A leading
// 0 byte means IVF-Flat (no PQ): ix.pq stays nil. Otherwise it reconstructs the
// residual codec (m, dsub, metric, codebooks), the dropped flag, and slotCell.
func (ix *ivf) readPQTrailer(br *bufio.Reader, dim int) error {
	on, err := br.ReadByte()
	if err != nil {
		return err
	}
	if on == 0 {
		ix.pq = nil
		ix.pqDropped = false
		ix.slotCell = nil
		return nil
	}
	mVal, err := readU32(br)
	if err != nil {
		return err
	}
	m := int(mVal)
	if m <= 0 || dim%m != 0 {
		return ErrSnapshotFormat
	}
	dropped, err := br.ReadByte()
	if err != nil {
		return err
	}
	dsub := dim / m
	// nbits is NOT serialized in the trailer (the codebooks carry their own width via
	// len(codebooks[s])): it is re-derived from the restored cfg, which snapColCfg
	// carries across the cluster restore path (RestoreAll→NewCollection→toConfig). A
	// 4-bit codec MUST be reconstructed with nbits==4 so codeForCell/the gather pick
	// the nibble-packed (m+1)/2-byte codes the verbatim arena block restored.
	p := &pq{m: m, dsub: dsub, dim: dim, metric: ix.cfg.Metric, nbits: ivfPQNBits(ix.cfg), codebooks: make([][][]float32, m)}
	for s := 0; s < m; s++ {
		nc, err := readU32(br)
		if err != nil {
			return err
		}
		cb := make([][]float32, int(nc))
		for c := range cb {
			sub := make([]float32, dsub)
			for i := range sub {
				f, err := readF32(br)
				if err != nil {
					return err
				}
				sub[i] = f
			}
			cb[c] = sub
		}
		p.codebooks[s] = cb
	}
	ix.pq = p
	ix.refreshPQ4Locked()
	ix.pqDropped = dropped == 1
	if ix.pqDropped {
		ix.arena.vecsDropped = true
		ix.arena.nslots = ix.arena.Capacity()
		ix.arena.vecs = nil
		// This reconstructs dropVecs's end state field by field rather than calling
		// it (the slot count comes from the restored header, not from a vecs length
		// that is about to be cleared), so it must not skip dropVecs's other half:
		// handing the off-heap backing back to the OS.
		ix.arena.releaseVecsBacking()
	}
	scN, err := readU32(br)
	if err != nil {
		return err
	}
	ix.slotCell = make([]uint32, int(scN))
	for i := range ix.slotCell {
		c, err := readU32(br)
		if err != nil {
			return err
		}
		ix.slotCell[i] = c
	}
	// OPQ rotation R, written only when present (see writePQTrailer). The trailer
	// is the last block, so EOF here means an old / non-OPQ blob ⇒ rotation nil
	// (back-compat). A leading 1 byte ⇒ dim×dim floats restored VERBATIM so the
	// codec rotates/un-rotates bit-identically (codes re-encode identically).
	rb, err := br.ReadByte()
	if err == io.EOF {
		return nil
	}
	if err != nil {
		return err
	}
	if rb == 1 {
		R := make([]float32, dim*dim)
		for i := range R {
			f, ferr := readF32(br)
			if ferr != nil {
				return ferr
			}
			R[i] = f
		}
		p.rotation = R
	}
	return nil
}

// vecsMode marks how the arena float block is carried in a serialized arena. The
// snapshot path uses only INLINE (1) and DROPPED (0) — byte-identical to before.
// The mmap sidecar adds EXTERNAL (2): the floats live in the cfg.MmapPath mmap
// file, so the float block is OMITTED from the .meta stream and openPersistIVF
// re-maps it zero-copy. DROPPED and EXTERNAL both omit the float block but are
// distinct: DROPPED has no vecs file (PQ-only, codes are the truth) while EXTERNAL
// has a live vecs file that must be mapped on open.
const (
	vecsDroppedMode  byte = 0
	vecsInlineMode   byte = 1
	vecsExternalMode byte = 2
)

// writeArena serializes the arena (vecs + expires + metadata + sparse +
// keyExpires + free + idMap). Mirrors the hnsw snapshot's arena block so the
// per-slot blocks read back identically; reuses the shared write helpers.
//
// vecsExternal selects the mmap-sidecar mode: when true the float block is NOT
// inlined (the floats live in the cfg.MmapPath mmap file, re-mapped on open) and
// a vecsExternalMode marker is written instead of the inline/dropped marker. The
// SNAPSHOT path passes false, so its bytes are byte-identical to before (only
// vecsInlineMode / vecsDroppedMode are ever written).
func (ix *ivf) writeArena(bw *bufio.Writer, vecsExternal bool) error {
	a := ix.arena
	if err := writeU32(bw, uint32(a.Size())); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(a.Capacity())); err != nil {
		return err
	}
	// v3: hasVecs byte. 1 = the float block follows (IVF-Flat + IVFRerank).
	// 0 = floats were dropped (PQ-only); the block is omitted and the reader
	// relies on the verbatim codes block at the arena tail. 2 (sidecar only) =
	// floats live externally in the mmap file; the block is omitted but, unlike
	// dropped, the vecs file is mapped on open.
	hasVecs := vecsInlineMode
	switch {
	case a.vecsDropped:
		hasVecs = vecsDroppedMode
	case vecsExternal:
		hasVecs = vecsExternalMode
	}
	if err := bw.WriteByte(hasVecs); err != nil {
		return err
	}
	if hasVecs == vecsInlineMode {
		for _, v := range a.vecs {
			if err := writeF32(bw, v); err != nil {
				return err
			}
		}
	}
	for _, e := range a.expires {
		if err := writeU64(bw, e); err != nil {
			return err
		}
	}
	withMeta := 0
	for _, m := range a.metadata {
		if m != nil {
			withMeta++
		}
	}
	if err := writeU32(bw, uint32(withMeta)); err != nil {
		return err
	}
	for slot, m := range a.metadata {
		if m == nil {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(m))); err != nil {
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
	withSparse := 0
	for _, sv := range a.sparse {
		if sv != nil {
			withSparse++
		}
	}
	if err := writeU32(bw, uint32(withSparse)); err != nil {
		return err
	}
	for slot, sv := range a.sparse {
		if sv == nil {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(sv.Indices))); err != nil {
			return err
		}
		for i, dim := range sv.Indices {
			if err := writeU32(bw, dim); err != nil {
				return err
			}
			if err := writeF32(bw, sv.Values[i]); err != nil {
				return err
			}
		}
	}
	withKeyExp := 0
	for _, ke := range a.keyExpires {
		if len(ke) > 0 {
			withKeyExp++
		}
	}
	if err := writeU32(bw, uint32(withKeyExp)); err != nil {
		return err
	}
	for slot, ke := range a.keyExpires {
		if len(ke) == 0 {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(ke))); err != nil {
			return err
		}
		for key, dl := range ke {
			if err := writeString(bw, key); err != nil {
				return err
			}
			if err := writeU64(bw, dl); err != nil {
				return err
			}
		}
	}
	if err := writeU32(bw, uint32(len(a.free))); err != nil {
		return err
	}
	for _, s := range a.free {
		if err := writeU32(bw, s); err != nil {
			return err
		}
	}
	if err := writeU32(bw, uint32(len(a.idMap))); err != nil {
		return err
	}
	// DETERMINISM: serialize the idMap in ASCENDING ID ORDER, never Go-map iteration
	// order, so the snapshot BYTES are identical across replicas with the same logical
	// state (the map content is order-independent; restore reads it back into a map
	// regardless of order, so this is back-compatible with older blobs). Without this
	// the per-replica map iteration order would make otherwise-identical snapshots
	// byte-differ — a cluster would not be able to byte-compare snapshots.
	idsSorted := make([]uint64, 0, len(a.idMap))
	for id := range a.idMap {
		idsSorted = append(idsSorted, id)
	}
	sort.Slice(idsSorted, func(i, j int) bool { return idsSorted[i] < idsSorted[j] })
	for _, id := range idsSorted {
		if err := writeU64(bw, id); err != nil {
			return err
		}
		if err := writeU32(bw, a.idMap[id]); err != nil {
			return err
		}
	}
	// v2: arena.versions — present-only [slot u32, version u64] pairs (a no-version
	// index pays just a zero count). Appended LAST so a v1 reader stops cleanly and
	// a v2 reader picks it up after the idMap.
	withVer := 0
	for _, v := range a.versions {
		if v != 0 {
			withVer++
		}
	}
	if err := writeU32(bw, uint32(withVer)); err != nil {
		return err
	}
	for slot, v := range a.versions {
		if v == 0 {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU64(bw, v); err != nil {
			return err
		}
	}
	// v3: verbatim codes block. codeLen then len(codes) bytes. For IVF-PQ these
	// are the RESIDUAL codes (which cannot be re-derived from vecs on restore, so
	// they are serialized rather than re-encoded). codeLen 0 = no codes (IVF-Flat
	// without a quantizer); SQ8/BQ1 also serialize verbatim here (harmless — the
	// reader uses them directly instead of re-encoding).
	if err := writeU32(bw, uint32(a.codeLen)); err != nil {
		return err
	}
	if a.codeLen > 0 {
		if err := writeU32(bw, uint32(len(a.codes))); err != nil {
			return err
		}
		if _, err := bw.Write(a.codes); err != nil {
			return err
		}
	}
	return nil
}

// Restore reconstructs the ivf from a snapshot written by Snapshot, under the
// write lock. Rebuilds the payload + sparse indexes from the arena (derived
// state). Wraps malformed input with ErrSnapshotFormat.
func (ix *ivf) Restore(r io.Reader) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	br := bufio.NewReader(r)

	var magic [8]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return ErrSnapshotFormat
	}
	if string(magic[:]) != ivfSnapshotMagic {
		return errBadMagic
	}
	version, err := readU32(br)
	if err != nil {
		return ErrSnapshotFormat
	}
	if version < 1 || version > ivfSnapshotVer {
		return ErrSnapshotFormat
	}

	dim, err := readU32(br)
	if err != nil {
		return err
	}
	metric, err := br.ReadByte()
	if err != nil {
		return err
	}
	mVal, err := readU32(br)
	if err != nil {
		return err
	}
	efC, err := readU32(br)
	if err != nil {
		return err
	}
	efS, err := readU32(br)
	if err != nil {
		return err
	}
	seed, err := readI64(br)
	if err != nil {
		return err
	}
	cfg := Config{
		Dim: int(dim), Metric: Metric(metric), M: int(mVal),
		EfConstruction: int(efC), EfSearch: int(efS), Seed: seed,
	}
	cfg.Quant = ix.cfg.Quant
	cfg.RescoreFactor = ix.cfg.RescoreFactor
	// IndexType + IVF-PQ knobs are not in the snapshot scalars; they come from the
	// reopen config (newIVF set them, like Quant). The PQ trailer carries the codec
	// itself + the dropped state, so these only configure validation/sizing.
	cfg.IndexType = ix.cfg.IndexType
	cfg.IVFNlist = ix.cfg.IVFNlist
	cfg.IVFNprobe = ix.cfg.IVFNprobe
	cfg.IVFPQ = ix.cfg.IVFPQ
	cfg.IVFPQM = ix.cfg.IVFPQM
	cfg.IVFRerank = ix.cfg.IVFRerank
	// PQNBits is not in the snapshot scalars; carry it from the reopen config so the
	// residual codec is reconstructed at the SAME width (4-bit vs 8-bit). readPQTrailer
	// re-derives nbits from ix.cfg.PQNBits — without this carry a 4-bit codec would
	// restore as 8-bit and misread the nibble-packed (m+1)/2-byte codes. AnisotropicEta
	// is train-only (the trained codebooks are in the trailer) but carried too so a
	// post-restore retrain keeps the score-aware loss.
	cfg.PQNBits = ix.cfg.PQNBits
	cfg.AnisotropicEta = ix.cfg.AnisotropicEta
	// Drift-retrain knobs are not in the snapshot scalars; carry them from the reopen
	// config so drift stays armed after a Restore. Without this the three fields would
	// silently zero-out and drift would not fire post-restore.
	cfg.IVFDriftRetrain = ix.cfg.IVFDriftRetrain
	cfg.IVFDriftGrowthFactor = ix.cfg.IVFDriftGrowthFactor
	cfg.IVFDriftFactor = ix.cfg.IVFDriftFactor
	// SOAR knobs are not in the snapshot scalars; carry them from the reopen config so
	// incremental inserts AFTER the Restore keep performing the secondary assignment.
	// The restored cellOf2/code2 (read by readIVFCore) hold the EXISTING multi-
	// assignment; these flags govern only future inserts. Without them a post-restore
	// insert would single-assign and the index would silently drift toward non-SOAR.
	cfg.SOAR = ix.cfg.SOAR
	cfg.SOARLambda = ix.cfg.SOARLambda
	if err := ValidateConfig(cfg); err != nil {
		return ErrSnapshotFormat
	}
	ix.cfg = cfg

	if err := ix.readArena(br, int(dim), version, ""); err != nil {
		return err
	}

	// Tombstones + IVF state (trained/nprobe/centroids/lists). Shared with the mmap
	// sidecar reader (ivf_persist.go). The v4 drift-retrain checkpoint follows the
	// lists block (v1/v2/v3 omit it — restored as 0); the v5 SOAR block follows the
	// checkpoint (v1..v4 omit it — restored as a plain single-assignment IVF).
	if err := ix.readIVFCore(br, int(dim), version >= 4, version >= 5); err != nil {
		return err
	}

	// v3 PQ trailer (codebooks + m + dropped + slotCell). Absent in v1/v2 (which
	// restore as IVF-Flat, pq == nil). The codes themselves were read by readArena.
	if version >= 3 {
		if err := ix.readPQTrailer(br, int(dim)); err != nil {
			return err
		}
	}

	// Rebuild derived indexes from the restored arena.
	ix.payloadIdx = newPayloadIndex()
	ix.payloadIdx.rebuild(ix.arena)
	ix.sparseIdx = newSparseIndex()
	ix.sparseIdx.rebuild(ix.arena, ix.tombstoned)

	if ix.now == nil {
		ix.now = func() int64 { return time.Now().UnixMilli() }
	}
	ix.idSetVersion = 1
	ix.scrollSnap.ver = 0
	// Reset the order_by snapshot machinery: the live set + payloads were wholesale
	// replaced, so drop every cached snapshot and restart dataVersion at 1.
	ix.dataVersion = 1
	ix.orderSnaps = make(map[orderCacheKey]*orderSnap)
	ix.orderSeq = 0
	return nil
}

// readArena reads the arena block written by writeArena into a fresh arena.
//
// mmapPath is the cfg.MmapPath of the externalized vecs file, set ONLY by the
// mmap sidecar reader (openPersistIVF). When the arena marker is vecsExternalMode
// the float block was omitted from the stream and the floats are re-mapped from
// mmapPath zero-copy (loadMmapVecs). The snapshot reader passes "" — the stream
// only ever carries vecsInlineMode / vecsDroppedMode, byte-identical to before.
func (ix *ivf) readArena(br *bufio.Reader, dim int, snapVersion uint32, mmapPath string) error {
	size, err := readU32(br)
	if err != nil {
		return err
	}
	capacity, err := readU32(br)
	if err != nil {
		return err
	}
	a := newArena(dim, int(size))
	a.maxVectorsHint = ix.cfg.MaxVectors // as newIVF does: the cap sizes the slab reservation
	if ix.arena != nil && ix.arena.quant != nil {
		a.setQuant(ix.arena.quant)
	}
	// IVF-PQ: the live arena enabled a codes side-array WITHOUT a quantizer
	// (residual codes). Carry the codeLen over so the verbatim codes block below
	// loads into the right-sized array.
	if ix.arena != nil && ix.arena.quant == nil && ix.arena.codeLen > 0 {
		a.codeLen = ix.arena.codeLen
	}
	// v3: a hasVecs byte precedes the float block (0 = PQ-only, floats dropped;
	// 2 = sidecar floats live externally in the mmap file). v1/v2 always have floats.
	hasVecs := vecsInlineMode
	if snapVersion >= 3 {
		hasVecs, err = br.ReadByte()
		if err != nil {
			return err
		}
	}
	switch hasVecs {
	case vecsInlineMode:
		a.vecs = make([]float32, int(capacity)*dim)
		for i := range a.vecs {
			f, err := readF32(br)
			if err != nil {
				return err
			}
			a.vecs[i] = f
		}
	case vecsExternalMode:
		// Sidecar: the float block was omitted; re-map the vecs file zero-copy at
		// its first `capacity` vectors (loadMmapVecs validates the on-disk size).
		// Only the mmap sidecar reader sets mmapPath; a vecsExternalMode marker in a
		// plain snapshot (no mmapPath) is malformed.
		if mmapPath == "" {
			return ErrSnapshotFormat
		}
		if err := a.loadMmapVecs(mmapPath, int(capacity)); err != nil {
			return err
		}
	default:
		// Floats were dropped; track the slot count so Capacity/Insert keep working.
		a.vecsDropped = true
		a.nslots = int(capacity)
	}
	// v1/v2 re-encode codes from vecs (centroid-agnostic SQ8/BQ1). v3 serializes
	// the codes verbatim (residual PQ codes cannot be re-derived from vecs) and is
	// read from the arena tail (readCodes), so skip the re-encode here for v3.
	if snapVersion < 3 && a.quant != nil {
		a.codes = make([]byte, int(capacity)*a.codeLen)
		for slot := 0; slot < int(capacity); slot++ {
			a.quant.Encode(a.codes[slot*a.codeLen:(slot+1)*a.codeLen], a.Vec(uint32(slot))) //nolint:gosec
		}
	}
	a.expires = make([]uint64, int(capacity))
	for i := range a.expires {
		e, err := readU64(br)
		if err != nil {
			return err
		}
		a.expires[i] = e
	}
	a.metadata = make([]Metadata, int(capacity))
	withMeta, err := readU32(br)
	if err != nil {
		return err
	}
	for i := uint32(0); i < withMeta; i++ {
		slot, err := readU32(br)
		if err != nil {
			return err
		}
		nk, err := readU32(br)
		if err != nil {
			return err
		}
		m := make(Metadata, nk)
		for j := uint32(0); j < nk; j++ {
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
		a.metadata[slot] = m
	}
	a.sparse = make([]*SparseVector, int(capacity))
	withSparse, err := readU32(br)
	if err != nil {
		return err
	}
	for i := uint32(0); i < withSparse; i++ {
		slot, err := readU32(br)
		if err != nil {
			return err
		}
		nnz, err := readU32(br)
		if err != nil {
			return err
		}
		sv := &SparseVector{
			Indices: make([]uint32, nnz),
			Values:  make([]float32, nnz),
		}
		for j := uint32(0); j < nnz; j++ {
			d, err := readU32(br)
			if err != nil {
				return err
			}
			val, err := readF32(br)
			if err != nil {
				return err
			}
			sv.Indices[j] = d
			sv.Values[j] = val
		}
		a.sparse[slot] = sv
	}
	a.keyExpires = make([]map[string]uint64, int(capacity))
	withKeyExp, err := readU32(br)
	if err != nil {
		return err
	}
	for i := uint32(0); i < withKeyExp; i++ {
		slot, err := readU32(br)
		if err != nil {
			return err
		}
		nk, err := readU32(br)
		if err != nil {
			return err
		}
		ke := make(map[string]uint64, nk)
		for j := uint32(0); j < nk; j++ {
			key, err := readString(br)
			if err != nil {
				return err
			}
			dl, err := readU64(br)
			if err != nil {
				return err
			}
			ke[key] = dl
		}
		a.keyExpires[slot] = ke
	}
	freeCount, err := readU32(br)
	if err != nil {
		return err
	}
	a.free = make([]uint32, freeCount)
	for i := range a.free {
		s, err := readU32(br)
		if err != nil {
			return err
		}
		a.free[i] = s
	}
	idCount, err := readU32(br)
	if err != nil {
		return err
	}
	a.idMap = make(map[uint64]uint32, idCount)
	a.ids = make([]uint64, int(capacity))
	for i := uint32(0); i < idCount; i++ {
		id, err := readU64(br)
		if err != nil {
			return err
		}
		slot, err := readU32(br)
		if err != nil {
			return err
		}
		a.idMap[id] = slot
		a.ids[slot] = id
	}
	// v2: arena.versions block (present-only [slot, version]). v1 snapshots have no
	// block — every live point defaults to version 1 (a sane existing-point
	// version), so a restored old IVF index reads consistent versions.
	a.versions = make([]uint64, int(capacity))
	if snapVersion >= 2 {
		withVer, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < withVer; i++ {
			slot, err := readU32(br)
			if err != nil {
				return err
			}
			v, err := readU64(br)
			if err != nil {
				return err
			}
			if int(slot) < len(a.versions) {
				a.versions[slot] = v
			}
		}
	} else {
		for id := range a.idMap {
			a.versions[a.idMap[id]] = 1
		}
	}
	// v3: verbatim codes block (codeLen + len + bytes). Mirrors writeArena's tail.
	if snapVersion >= 3 {
		cl, err := readU32(br)
		if err != nil {
			return err
		}
		a.codeLen = int(cl)
		if cl > 0 {
			n, err := readU32(br)
			if err != nil {
				return err
			}
			a.codes = make([]byte, int(n))
			if _, err := io.ReadFull(br, a.codes); err != nil {
				return err
			}
		}
	}
	// Same ownership rule as hnsw's readSnapshot: the outgoing arena's slabs may be
	// off-heap reservations that the GC cannot reclaim and no finalizer releases,
	// so it must be Closed before the reference is dropped. Deliberately AFTER the
	// new arena is fully built — readArena reads ix.arena.quant, and when both are
	// mmap-backed by the same path the two mappings are MAP_SHARED views of one
	// page cache, so unmapping the old one cannot strand or stale the new one.
	if ix.arena != nil {
		_ = ix.arena.Close()
	}
	ix.arena = a
	// The Expires/KeyExpires blocks above wrote a.expires/a.keyExpires directly,
	// bypassing SetExpires/SetKeyExpires's incremental deadline-count maintenance
	// (arena.deadlinePoints/deadlineKeys — the dense TTL sweep's fast-path gate;
	// see ttl.go). Recompute once, in bulk, so a later gate consumer never
	// undercounts (the dangerous direction: an undercount can wrongly skip a
	// sweep and leave a TTL point permanently unswept).
	a.RecomputeDeadlineCounts()
	return nil
}

// SavePersist writes the IVF instant-restart mmap sidecar at metaPath. The float
// vectors stay in the cfg.MmapPath mmap file (msync'd here); the IVF structures
// (centroids, lists, slotCell, tombstones, codes, PQ trailer, arena metadata) are
// captured in the sidecar with the vecs EXTERNALIZED. Implemented in
// ivf_persist.go (mirrors hnsw.SavePersist / persist.go).
func (ix *ivf) SavePersist(metaPath string) error {
	return ix.savePersist(metaPath)
}

// ---------------------------------------------------------------------------
// observability / lifecycle (mirror hnsw)
// ---------------------------------------------------------------------------

// Stats returns a runtime snapshot (mirror hnsw.Stats).
func (ix *ivf) Stats() Stats {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return Stats{
		Size:          ix.arena.Size() - len(ix.tombstoned),
		Tombstoned:    len(ix.tombstoned),
		SearchOps:     ix.searchOps.Load(),
		InsertOps:     ix.insertOps.Load(),
		Expired:       ix.expiredCount.Load(),
		QuotaRejects:  ix.quotaRejects.Load(),
		FilterRejects: ix.filterRejects.Load(),
		SparseVectors: ix.countSparse(),
		SearchLatency: ix.searchLat.snapshot(),
		InsertLatency: ix.insertLat.snapshot(),
	}
}

// countSparse mirrors hnsw.countSparse. Must hold ix.mu.
func (ix *ivf) countSparse() int {
	n := 0
	for _, slot := range ix.arena.idMap {
		if ix.tombstoned[slot] {
			continue
		}
		if ix.arena.sparse[slot] != nil {
			n++
		}
	}
	return n
}

// Close releases the arena (mirror hnsw.Close; IVF is heap-backed in v1, so this
// is effectively a no-op beyond the arena's own Close).
func (ix *ivf) Close() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	return ix.arena.Close()
}

// Compile-time assertion that *ivf satisfies VectorIndex.
var _ VectorIndex = (*ivf)(nil)
