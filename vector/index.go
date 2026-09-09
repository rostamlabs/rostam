// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"io"
	"math"
	"time"

	"github.com/rostamlabs/rostam/vector/analysis"
)

// Metric, IndexType, and their Cosine/L2/DotProduct and IndexHNSW/IndexIVF/
// IndexVamana/IndexGPU constants now live in the engine-free vtypes leaf package
// and are re-exported via vtypes_aliases.go.

// Vamana defaults (applied when the corresponding Config field is 0).
const (
	defaultVamanaR     = 64  // max out-degree (level-0 slab stride)
	defaultVamanaL     = 100 // build/search beam width
	defaultVamanaAlpha = 1.2 // pass-2 RobustPrune α (pass 1 is always α=1)
)

// Config, NamedVectorParams, DefaultConfig, the QuantStore storage enum, and the
// config validation Err* sentinels now live in the engine-free vtypes leaf
// package (vtypes/config.go) and are re-exported via vtypes_aliases.go. Config's
// only engine-coupled method (Validate) and NamedVectorParams.toConfig became
// the free functions ValidateConfig and namedToConfig below, which stay in vector.

// FullTextConfig now lives in the engine-free vtypes leaf package and is
// re-exported via vtypes_aliases.go.

// NamedConfig is the collection-level config carrier for a named-vector
// collection: the per-space index params (Spaces) plus the collection-level
// single-node durability flags. It is the named analogue of the dense Config's
// {NamedVectors, WAL, WALNoSync} subset — the named family has no Persistent
// (mmap) mode, so WAL is its only single-node disk-durability option.
//
// WAL enables the named single-node WAL lifecycle (apply-then-log under opMu,
// open = restore-snapshot + replay-WAL-tail + open-WAL, Flush = snapshot +
// truncate). It is FORCED OFF on the cluster path (durability is Raft/SnapshotAll
// there, exactly like dense effectiveClusterConfig). Heap-only when WAL is false.
type NamedConfig struct {
	Spaces    map[string]NamedVectorParams `json:"spaces"`
	WAL       bool                         `json:"wal,omitempty"`
	WALNoSync bool                         `json:"wal_no_sync,omitempty"`
}

// namedToConfig builds the Config for this space's sub-HNSW. Named sub-indexes are
// always heap-backed/in-memory in v1 (no per-space WAL/Persistent/mmap — see the
// named-vectors plan non-goals); only the per-index params carry over. M /
// EfConstruction / EfSearch defaults match DefaultConfig so a zero-value
// NamedVectorParams (beyond Dim) yields the standard HNSW.
func namedToConfig(p NamedVectorParams) Config {
	c := DefaultConfig()
	c.Dim = p.Dim
	c.Metric = p.Metric
	if p.M > 0 {
		c.M = p.M
	}
	if p.EfConstruction > 0 {
		c.EfConstruction = p.EfConstruction
	}
	if p.EfSearch > 0 {
		c.EfSearch = p.EfSearch
	}
	c.Seed = p.Seed
	c.MaxEfSearch = p.MaxEfSearch
	c.Quant = p.Quant
	c.RescoreFactor = p.RescoreFactor
	c.FilterFirstRelativeBP = p.FilterFirstRelativeBP
	c.QuantizedBuild = p.QuantizedBuild
	// IVF / IVF-PQ knobs (ignored by newHNSW when IndexType == IndexHNSW). The
	// per-space Config is Validated by newIndex (newHNSW/newIVF both call
	// ValidateConfig(cfg)), so a bad IVF param fails loud at NewNamedCollection.
	c.IndexType = p.IndexType
	c.IVFNlist = p.IVFNlist
	c.IVFNprobe = p.IVFNprobe
	c.IVFPQ = p.IVFPQ
	c.IVFPQM = p.IVFPQM
	c.IVFRerank = p.IVFRerank
	c.OPQ = p.OPQ
	c.OPQIters = p.OPQIters
	c.IVFTrainThreshold = p.IVFTrainThreshold
	c.IVFDriftRetrain = p.IVFDriftRetrain
	c.IVFDriftGrowthFactor = p.IVFDriftGrowthFactor
	c.IVFDriftFactor = p.IVFDriftFactor
	// PQDropVecs (HNSW-PQ only) folds the float-drop into this space's incremental
	// auto-train. Validated QuantPQ-only by the ValidateConfig(namedToConfig(...)) in
	// validateNamedVectors / newIndex (ErrInvalidPQDropVecs otherwise).
	c.PQDropVecs = p.PQDropVecs
	return c
}

// ValidateConfig returns an error if the config is malformed. It is a free
// function (not a Config method) because it reaches engine code
// (analysis.ByName for the full-text analyzer, gpuSupported / defaultPQM /
// maxOPQIters for the GPU/PQ probes), so Config itself stays engine-free in
// the vtypes leaf package and can move there.
func ValidateConfig(c Config) error {
	if c.Dim <= 0 {
		return ErrInvalidDim
	}
	if c.Metric > DotProduct {
		return ErrInvalidMetric
	}
	if c.M <= 0 || c.M > 128 {
		return ErrInvalidM
	}
	if c.EfConstruction <= 0 {
		return ErrInvalidEf
	}
	if c.EfSearch <= 0 {
		return ErrInvalidEf
	}
	// QuantPQ, QuantSQ and QuantPRQ are allowed for the dense/HNSW family (navigate
	// on approximate codes, exact-rescore the over-collected shortlist on the kept
	// floats). Any quant mode above QuantPRQ is unknown. The allowance was extended
	// from QuantSQ to QuantPRQ (product-residual quantization); it admits ONLY the
	// known dense modes, no garbage. (The IVF index does NOT set cfg.Quant ==
	// QuantPQ/QuantSQ/QuantPRQ — it drives PQ via IVFPQ/enableCodes — so this
	// allowance is purely a new dense capability and leaves IVF's validation
	// untouched.)
	if c.Quant > QuantPRQ {
		return ErrInvalidQuant
	}
	// SQBits selects the trained scalar quantizer's bit-depth: 0 (⇒ 8), 4, 6, or 8.
	// 8-bit is one byte/dim (the Task-1 fast path); 4/6 pack levels LSB-first into a
	// bit stream (ceil(dim*bits/8) bytes). Anything else is rejected. Validated only
	// for QuantSQ; a non-SQ config ignores the field.
	if c.Quant == QuantSQ {
		// SQ on the graph engine (HNSW or Vamana): both share the trained-SQ auto-train
		// path (BuildConcurrent / buildVamana train + encode). An IVF index has no SQ
		// auto-train path, so IVF+QuantSQ would pass Validate but never train its
		// quantizer — reject it (only IVF is excluded here).
		if c.IndexType != IndexHNSW && c.IndexType != IndexVamana {
			return ErrInvalidQuant
		}
		switch c.SQBits {
		case 0, 4, 6, 8:
		default:
			return ErrInvalidQuant
		}
	}
	if c.Quant == QuantPQ {
		// PQ on the graph engine (HNSW or Vamana): both train + encode via
		// trainAndEncodePQ (BuildConcurrent / buildVamana) and navigate on ADC + exact
		// rescore — the disk-native Vamana path. An IVF index never carries Quant ==
		// QuantPQ (it uses IVFPQ), so only IVF is excluded here.
		if c.IndexType != IndexHNSW && c.IndexType != IndexVamana {
			return ErrInvalidQuant
		}
		// PQNBits selects the per-subspace code width: 0/8 (256 sub-centroids, the
		// 8-bit codec) or 4 (16 sub-centroids, the LUT16 fast-scan codec). Any other
		// value is rejected.
		switch c.PQNBits {
		case 0, 4, 8:
		default:
			return ErrInvalidPQNBits
		}
		// The configured M (QuantPQM, or defaultPQM(Dim) when 0) must divide Dim evenly.
		m := c.QuantPQM
		if m == 0 {
			m = defaultPQM(c.Dim)
		}
		if m <= 0 || c.Dim%m != 0 {
			return ErrInvalidQuantPQM
		}
	}
	if c.Quant == QuantPRQ {
		// PRQ on the graph engine (HNSW or Vamana): both train + encode via
		// trainAndEncodePRQ and navigate on the summed-LUT ADC + exact rescore. An IVF
		// index never carries Quant == QuantPRQ, so only IVF is excluded here.
		if c.IndexType != IndexHNSW && c.IndexType != IndexVamana {
			return ErrInvalidQuant
		}
		// PRQLayers must be >= 0 (0 ⇒ defaultPRQLayers). Negative is nonsensical.
		if c.PRQLayers < 0 {
			return ErrInvalidPRQLayers
		}
		// Each layer is an m-subquantizer PQ; m (QuantPQM, or defaultPQM(Dim) when 0)
		// must divide Dim evenly, exactly like QuantPQ.
		m := c.QuantPQM
		if m == 0 {
			m = defaultPQM(c.Dim)
		}
		if m <= 0 || c.Dim%m != 0 {
			return ErrInvalidQuantPQM
		}
	}
	if c.QuantPQM < 0 {
		return ErrInvalidQuantPQM
	}
	if c.PRQLayers < 0 {
		return ErrInvalidPRQLayers
	}
	if c.RescoreFactor < 0 {
		return ErrInvalidRescoreFactor
	}
	// Checked before QuantStorage so a Persistent config without a quantizer
	// reports the actionable cause (the store derives QuantStorage from it). IVF is
	// EXEMPT: its instant-restart sidecar mmaps the full-precision float vectors
	// directly (cfg.MmapPath), so an IVF-Flat (QuantNone) collection can be Persistent
	// — the mmap holds floats, not quantized codes. HNSW still requires a quantizer
	// (its mmap storage holds codes).
	if c.Persistent && c.Quant == QuantNone && c.IndexType != IndexIVF {
		return ErrInvalidPersistent
	}
	if c.QuantizedBuild && c.Quant == QuantNone {
		return ErrInvalidQuantizedBuild
	}
	if c.Partitions < 0 {
		return ErrInvalidPartitions
	}
	if c.WAL && c.Persistent {
		return ErrInvalidWAL
	}
	if c.QuantStorage > QuantMmap {
		return ErrInvalidQuantStorage
	}
	if c.QuantStorage == QuantMmap && (c.Quant == QuantNone || c.MmapPath == "") {
		return ErrInvalidQuantStorage
	}
	if c.GraphMmapPath != "" && c.GraphMmapPath == c.MmapPath {
		return ErrInvalidGraphMmap
	}
	if c.IndexType > IndexGPU {
		return ErrInvalidIndexType
	}
	// IndexGPU is only usable in a -tags cuda build. In the DEFAULT (pure-Go,
	// CGO_ENABLED=0) build gpuSupported is false, so fail LOUD here at config time
	// rather than silently falling back to HNSW.
	if c.IndexType == IndexGPU && !gpuSupported {
		return ErrGPUNotCompiled
	}
	if c.IVFNlist < 0 || c.IVFNprobe < 0 {
		return ErrInvalidIVFParams
	}
	// Vamana geometry: R (max out-degree), L (beam width), alpha (pass-2 prune α).
	// Defaults (R=64, L=100, alpha=1.2) apply for 0, so the constraints are checked
	// against the effective values: R>0, L>=R, alpha>=1. Negative R/L and alpha in
	// (0,1) are nonsensical (fail loud, mirroring the IVF/SQ param gates). Validated
	// for every config (cheap) but only meaningful for IndexVamana — a non-Vamana
	// config leaves the fields 0, which resolves to the defaults and passes.
	{
		r, l, alpha := c.VamanaR, c.VamanaL, c.VamanaAlpha
		if r == 0 {
			r = defaultVamanaR
		}
		if l == 0 {
			l = defaultVamanaL
		}
		if alpha == 0 {
			alpha = defaultVamanaAlpha
		}
		if r <= 0 || l < r || alpha < 1 {
			return ErrInvalidVamanaParams
		}
	}
	if c.IVFTrainThreshold < 0 {
		return ErrInvalidIVFTrainThreshold
	}
	// FilterFirstRelativeBP is basis points (0..10000); fail loud on out-of-range so
	// a nonsensical value never silently rides (mirrors the ivf_nlist negative reject).
	// 0 (default) is the OFF state (byte-identical absolute behavior).
	if c.FilterFirstRelativeBP < 0 || c.FilterFirstRelativeBP > 10000 {
		return ErrInvalidFilterFirstRelativeBP
	}
	// Drift-retrain factors must be > 1.0 (a checkpoint/threshold below 1 would
	// retrain on every insert or never). 0 is allowed — it resolves to the engine
	// default (2.0 growth / 1.5 drift) at use. Fail loud on an explicit <= 1.0,
	// mirroring the ivf_nlist negative reject. The factors are validated unconditionally
	// (even when IVFDriftRetrain is off) so a nonsensical value never silently rides.
	if c.IVFDriftGrowthFactor != 0 && c.IVFDriftGrowthFactor <= 1.0 {
		return ErrInvalidIVFDriftFactor
	}
	if c.IVFDriftFactor != 0 && c.IVFDriftFactor <= 1.0 {
		return ErrInvalidIVFDriftFactor
	}
	// IVF + Persistent is supported via the instant-restart mmap sidecar
	// (SavePersist/openPersistIVF): the float vectors live in the mmap'd vecs file
	// and the IVF structures (centroids, lists, slotCell, codes, PQ trailer) are
	// written to the .meta sidecar at Flush. Both vecs-present (IVF-Flat / IVFRerank)
	// and the float-dropped PQ-only state round-trip. No gate needed here; the
	// store wires cfg.MmapPath for a Persistent IVF (effectiveConfig) and the
	// cluster path still forces Persistent=false (snapshot/Raft durable). The MV
	// inner-IVF rejection (multivector_persist.go) is separate and stays.
	// IVF-PQ: residual product quantization is a mode of the IVF index. PQNBits
	// selects the residual code width — 0/8 (256 sub-centroids/subspace, the
	// byte-per-subspace default) or 4 (16 sub-centroids, the nibble-packed LUT16
	// fast-scan codec scored by the in-register kernel on the query path). IVFPQM
	// must divide Dim evenly; 0 is allowed and resolved to defaultPQM(Dim) at
	// construction.
	if (c.IVFPQ || c.IVFRerank) && c.IndexType != IndexIVF {
		return ErrInvalidIVFPQ
	}
	if c.IVFPQM < 0 {
		return ErrInvalidIVFPQM
	}
	if c.IVFPQ && c.IVFPQM > 0 && c.Dim%c.IVFPQM != 0 {
		return ErrInvalidIVFPQM
	}
	// PQNBits on IVF-PQ: only 0/4/8 are valid (mirrors the QuantPQ gate above). Any
	// other value is rejected fail-loud so a nonsensical width never silently rides.
	if c.IVFPQ {
		switch c.PQNBits {
		case 0, 4, 8:
		default:
			return ErrInvalidPQNBits
		}
	}
	// OPQ is a rotation that lives inside the PQ codec, so it is only meaningful
	// with a PQ-family mode active: HNSW-PQ (Quant == QuantPQ), HNSW-PRQ
	// (Quant == QuantPRQ — the rotation is applied once before layer 0), or IVF-PQ
	// (IVFPQ). Fail loud otherwise so an OPQ flag on a non-PQ index is not silently
	// ignored.
	if c.OPQ && c.Quant != QuantPQ && c.Quant != QuantPRQ && !c.IVFPQ {
		return ErrInvalidOPQ
	}
	// OPQIters drives full-OPQ iterative refinement (see Config.OPQIters). Fail
	// loud on an out-of-range value [0, maxOPQIters] so a nonsensical iteration
	// count never silently rides (mirrors the OPQ / IVFTrainThreshold range checks).
	// 0 (= 1, the v1 behavior) and 1 are byte-identical to before. The cap bounds
	// the one-time O(d²·sweeps·iters) training cost on large dim.
	if c.OPQIters < 0 || c.OPQIters > maxOPQIters {
		return ErrInvalidOPQIters
	}
	// AnisotropicEta is the ScaNN score-aware PQ weight (η ≥ 1; 0/1 = isotropic).
	// Reject a negative or NaN value fail-loud so a nonsensical knob never silently
	// rides (mirrors the OPQIters / IVFDriftFactor gates). 0 and 1 are the
	// byte-identical isotropic state; any η > 0 is accepted (η in (0,1) is treated as
	// isotropic by trainCodebooks, which only switches the anisotropic trainer for
	// η > 1). Validated unconditionally (cheap) — it only affects PQ training, so a
	// non-PQ collection leaves it 0 and passes.
	if c.AnisotropicEta < 0 || math.IsNaN(float64(c.AnisotropicEta)) {
		return ErrInvalidAnisotropicEta
	}
	// SOAR is an IVF-only multi-assignment mode (it adds a second inverted-list
	// membership per point). Reject it on a non-IVF index fail-loud so a SOAR flag
	// on HNSW/Vamana is not silently ignored. SOARLambda must be >= 0 and non-NaN
	// (0 resolves to the engine default 1.5 at assignment); validated
	// unconditionally so a nonsensical λ never silently rides even when SOAR is off.
	if c.SOAR && c.IndexType != IndexIVF {
		return ErrInvalidSOAR
	}
	if c.SOARLambda < 0 || math.IsNaN(float64(c.SOARLambda)) {
		return ErrInvalidSOARLambda
	}
	// PQDropVecs is the HNSW-PQ maximum-compression mode (drop resident floats →
	// ADC-only search). It is only meaningful when the dense HNSW PQ codec is
	// active (Quant == QuantPQ); fail loud so a PQDropVecs flag on a non-PQ index
	// (or on the separate IVFPQ family) is not silently ignored. Mirrors the OPQ
	// gate above.
	if c.PQDropVecs && c.Quant != QuantPQ {
		return ErrInvalidPQDropVecs
	}
	// FullText (BM25) is HNSW-only in v1 and must name a registered analyzer.
	// nil (default) is the byte-identical disabled state. The analyzer is resolved
	// at construction; validate the name here so a bad config fails loud at create.
	if c.FullText != nil {
		if c.IndexType != IndexHNSW {
			return ErrInvalidFullText
		}
		if _, err := analysis.ByName(c.FullText.Analyzer); err != nil {
			return ErrInvalidFullText
		}
	}
	return nil
}

// CASCond is the optimistic-concurrency precondition threaded into the dense
// mutators. It is the "expected version" half of a compare-and-set:
//
//   - Has == false (the zero value): NO precondition — an unconditional write.
//     The version STILL bumps on success. Every existing non-CAS caller passes
//     the zero value, so its behavior is unchanged (today's write + a bump).
//   - Has == true: the mutation applies ONLY when the point's current version
//     (0 if absent) equals Expected; otherwise it returns ErrVersionConflict
//     with no mutation and no bump. Expected == 0 means "expect the point to be
//     absent/new" — an insert-if-absent CAS (etcd/Qdrant-style).
//
// The check + bump are atomic under the engine write lock (the FSM-Apply
// serialization point), so the result is deterministic across Raft replicas.
type CASCond struct {
	Expected uint64 // the version the caller expects the point to currently have
	Has      bool   // whether to enforce the precondition at all
}

// check evaluates the precondition against the point's current version (0 if
// absent). It returns ErrVersionConflict when Has is set and current !=
// Expected; otherwise nil (no precondition, or a match). Callers MUST hold the
// engine write lock so the read-of-current and the subsequent apply+bump are one
// atomic step.
func (c CASCond) check(current uint64) error {
	if c.Has && current != c.Expected {
		return ErrVersionConflict
	}
	return nil
}

// Result (one search result) now lives in the engine-free vtypes leaf package and
// is re-exported from vtypes_aliases.go.

// Stats is a snapshot of an index's runtime statistics.
type Stats struct {
	Size            int
	Tombstoned      int
	SearchOps       uint64
	InsertOps       uint64
	AvgSearchDepth  float32
	Expired         uint64           // count of entries that aged out via TTL
	QuotaRejects    uint64           // cumulative inserts rejected by quota or rate limit
	FilterRejects   uint64           // cumulative candidates rejected by an active search filter
	FilterGates     uint64           // cumulative filtered searches that armed the payload-index bitset admission gate
	ComplementGates uint64           // the subset of FilterGates armed from the filter's REJECTION side (high-pass-rate ranges)
	ColumnGates     uint64           // the subset of FilterGates answered by the numeric column sidecar
	ColumnDrops     uint64           // times an insert reclaimed the column sidecar to stay inside MaxBytes
	SparseVectors   int              // live slots carrying a sparse vector
	SearchLatency   LatencyHistogram // per-search wall time, log-scale buckets
	InsertLatency   LatencyHistogram // per-insert wall time, log-scale buckets
}

// HybridOpts (the hybrid dense+sparse search config) now lives in the engine-free
// vtypes leaf package and is re-exported from vtypes_aliases.go.

// VectorIndex is the abstraction over any nearest-neighbor index. It is the full
// contract Collection requires of its backing index: every Collection method
// delegates to one of these. HNSW is the only implementation today; a future
// IVF-Flat implementation must satisfy this same interface. Several methods are
// unexported — that is fine because all implementations live in package vector.
type VectorIndex interface {
	// --- Writes ---
	// Insert upserts id. cas is the optimistic-CAS precondition (CASCond{} = no
	// precondition, an unconditional write that still bumps the version). On
	// success it returns the resulting version (1 for a fresh insert, current+1
	// for an in-place upsert); on a CAS mismatch it returns ErrVersionConflict
	// with no mutation.
	// keyTTLMs is an OPTIONAL per-key payload TTL map (key -> RELATIVE ms); the
	// engine computes the ABSOLUTE deadline now+ttl at insert (mirroring
	// set_payload) and stores it in the slot's keyExpires. Empty/nil = no per-key
	// TTL (byte-identical / zero-overhead).
	// It returns the resulting version AND the resulting ABSOLUTE per-key deadline
	// map (key -> now+ttl) the engine computed, so the Collection layer WAL-logs
	// exactly what was applied (time-stable). keyExpires is nil when no per-key TTL.
	Insert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (version uint64, keyExpires map[string]uint64, err error)
	// InsertAt is Insert with every TTL deadline computation AND liveness check
	// (CAS/reclaim) judged against the EXPLICIT leader-stamped clock nowMs (unix
	// millis) instead of the wall clock, so a replicated apply stamps byte-identical
	// absolute point/per-key deadlines and agrees on liveness on every replica (#4
	// vector TTL determinism, mirroring cache.PutAt). Insert is byte-identical to
	// before this method existed.
	InsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (version uint64, keyExpires map[string]uint64, err error)
	// RestoreInsert inserts id with an EXACT version set verbatim (NOT bumped) —
	// the version-preserving primitive used by WAL replay and reshard copy so a
	// restored/copied point keeps its original version. No CAS, no rate limit.
	// keyExpires is the ABSOLUTE per-key deadline map restored VERBATIM (NOT
	// recomputed now+ttl) so pending key deadlines survive a crash time-stable.
	RestoreInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error
	// RestoreInsertAt is RestoreInsert stamping the POINT ttl deadline and judging
	// reclaim liveness against the EXPLICIT leader-stamped clock nowMs (keyExpires is
	// still installed VERBATIM) — the replicated version-preserving insert path (#4
	// vector TTL determinism).
	RestoreInsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) error
	// InsertIfAbsent inserts only if id is not currently live (returns
	// inserted=false on a no-op), atomically (one critical section, no
	// check-then-act gap). The online-copy primitive that, with Raft
	// serialization, closes Race A (value clobber).
	InsertIfAbsent(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) (bool, error)
	// InsertIfAbsentVersion is InsertIfAbsent that, on a real insert, sets the
	// point's version VERBATIM to `version` (NOT 1) — the version-PRESERVING
	// online-copy primitive used by the reshard copy pass so a copied point keeps
	// its per-point CAS version while still never clobbering a concurrent live
	// dual-write. version==0 behaves exactly like InsertIfAbsent (version 1 on a
	// fresh insert). keyExpires is the ABSOLUTE per-key payload deadline map set
	// VERBATIM (NOT recomputed now+ttl) on a real insert so the online reshard copy
	// keeps the point's original key deadlines time-stable; version==0 && nil
	// keyExpires reproduces today's InsertIfAbsent behavior byte-identically.
	InsertIfAbsentVersion(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) (bool, error)
	// InsertIfAbsentVersionAt is InsertIfAbsentVersion whose liveness OUTCOME (insert
	// vs no-op), reclaim, and point-TTL deadline are judged against the EXPLICIT
	// leader-stamped clock nowMs, so skewed replicas agree on whether an expired id
	// resurrects and stamp identical deadlines (#4 vector TTL determinism).
	InsertIfAbsentVersionAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) (bool, error)
	// Delete removes id. cas is the optimistic-CAS precondition; on a mismatch it
	// returns ErrVersionConflict and removed=false with no mutation. A no-CAS
	// Delete (CASCond{}) behaves exactly as before (removed reports whether id was
	// live). The version is dropped with the point (a later reinsert starts at 1).
	Delete(id uint64, cas CASCond) (removed bool, err error)
	// DeleteAt is Delete judging the dead-slot liveness gate against the EXPLICIT
	// leader-stamped clock nowMs, so replicas agree on already-dead vs live and their
	// tombstone sets stay identical (#4 vector TTL determinism). Delete is
	// byte-identical to before.
	DeleteAt(id uint64, cas CASCond, nowMs int64) (removed bool, err error)
	// BuildConcurrent bulk-loads an empty index from ids/vecs using `workers`
	// goroutines (0 = GOMAXPROCS).
	BuildConcurrent(ids []uint64, vecs [][]float32, workers int) error
	// BuildConcurrentMeta is BuildConcurrent carrying an OPTIONAL per-point
	// payload: metas is nil/empty (identical to BuildConcurrent) or exactly
	// len(ids) long with a nil entry per payload-less point. It is on the
	// INTERFACE rather than behind a type assertion on purpose — an optional
	// interface would let a family that never implemented it silently drop the
	// payloads of a bulk load, which is the exact failure this path exists to
	// avoid.
	BuildConcurrentMeta(ids []uint64, vecs [][]float32, metas []Metadata, workers int) error

	// --- Liveness / point reads ---
	// Exists reports whether id is currently live, using the same liveness
	// definition as search admission (tombstoned + TTL-expired = not live). O(1)
	// idMap probe; the resurrection-guard liveness check (Race B).
	Exists(id uint64) bool
	// Get returns the live record plus the point's current version (>= 1 for a
	// live point; an absent/dead point returns ok=false and version 0).
	Get(id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool)
	// GetInto is Get that APPENDS the stored vector into the caller-owned scratch
	// buffer dst (passed as dst[:0]) instead of allocating a fresh []float32 each
	// call — the dense-read analogue of cache.GetInto. The returned vec aliases
	// dst's backing array when dst has the capacity, so a hot-loop caller reusing
	// one buffer incurs ZERO allocations for the vector copy. Everything else (meta,
	// ttl, sparse, version, ok) is identical to Get, including the same liveness
	// gate and torn-read safety — only the destination of the vector copy changes.
	// On a miss (absent/tombstoned/expired) it returns vec=nil like Get. NOTE: when
	// the storage path must RECONSTRUCT the vector (e.g. a PQDropVecs index where
	// vecFor allocates), that inner allocation is unavoidable and GetInto cannot
	// elide it — the zero-alloc guarantee holds only when vecFor aliases the arena.
	GetInto(dst []float32, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool)

	// --- Search ---
	Search(query []float32, k int) ([]Result, error)
	SearchFiltered(query []float32, k int, filter Filter) ([]Result, error)
	SearchInto(dst []Result, query []float32, k int, filter Filter) ([]Result, error)
	// SearchFilteredWith is SearchInto with an OPTIONAL external metadata provider
	// (the named/MV hook). metaOf == nil is byte-identical to SearchInto. When
	// metaOf != nil the filter predicate is evaluated against the EXTERNAL per-point
	// payload via metaOf(id) (the sub-arena carries no metadata) — predicate-eval
	// only, the payload-index/filter-first planner is bypassed.
	SearchFilteredWith(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error)
	// filterFirstByID computes exact top-k over an index-narrowed candidate ID set
	// whose payload lives in an EXTERNAL map (the named/MV families: the sub-arena
	// carries the VECTORS but no metadata). For each candidate id it resolves the
	// vector via the sub-arena, applies the tombstone/TTL liveness gate AND the
	// external predicate RE-CHECK (pred evaluated against metaOf(id)), scores under
	// the active metric, and returns the k closest. Takes the index's own read lock.
	filterFirstByID(dst []Result, cands []uint64, query []float32, k int, pred Predicate, metaOf func(id uint64) Metadata) []Result
	// preferFilterFirst is the cost-based planner decision: should an index-narrowed
	// filtered search brute-force the ncand candidates (exact) rather than traverse
	// the index? It compares estimated distance-computation costs.
	preferFilterFirst(ncand, k int) bool
	// filterFirstCrossover returns the largest ncand in [0, limit] for which
	// preferFilterFirst(ncand, k) is true — i.e. the biggest candidate set the
	// planner could still choose filter-first for. A caller that materializes a
	// candidate superset passes it as the early-abort ceiling, so a set that grows
	// past the cap is abandoned mid-build instead of being finished and discarded.
	// Each index computes it from ITS OWN cost model.
	filterFirstCrossover(k, limit int) int
	// filterFirstThreshold is the maximum candidate-set size for which the planner
	// uses brute-force filter-first search instead of index traversal.
	filterFirstThreshold() int
	// effectiveFilterFirstLimit folds the OPT-IN relative selectivity gate
	// (Config.FilterFirstRelativeBP) into filterFirstThreshold for the given live
	// count. When FilterFirstRelativeBP == 0 (default) it returns
	// filterFirstThreshold() EXACTLY (byte-identical); when > 0 it may raise the
	// limit up to a hard cap so a relatively-selective filter can use filter-first
	// beyond the absolute cap. The named/MV search gates call this on the inner
	// index with the inner arena's live count.
	effectiveFilterFirstLimit(liveCount int) int
	// Dim returns the configured vector dimensionality (cfg.Dim).
	Dim() int
	// vecsForIDs returns, for each live id present in this index, a COPY of its
	// float vector (exact arena floats when present, reconstructed from the PQ code
	// + coarse centroid when the floats were dropped). Absent/dead ids are omitted.
	// Takes the index's own read lock internally and never aliases arena storage.
	vecsForIDs(ids []uint64) map[uint64][]float32
	// withVecAccess holds the index's read lock for the duration of fn and passes a
	// getter that returns each live id's float vector as an arena VIEW (no per-id
	// copy/allocation — the MaxSim hot path resolves thousands of token vectors per
	// query). The view is valid ONLY for the duration of fn (the lock is released on
	// return). PQ-dropped floats are reconstructed into a fresh slice (rare). Absent
	// ids report ok=false. The allocation-free analogue of vecsForIDs.
	withVecAccess(fn func(get func(id uint64) ([]float32, bool)))
	HybridSearch(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, error)
	HybridLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, []Result, error)
	SearchMMR(query []float32, k int, opts MMROpts) ([]Result, error)
	Recommend(k int, opts RecommendOpts) ([]Result, error)
	Discover(k int, opts DiscoverOpts) ([]Result, error)
	// DiscoverVecs is the RESOLVED-vectors discovery path (the Query API leaf
	// form): the target + context-pair example VECTORS are supplied directly (the
	// coordinator resolved the ids and embedded them), so no per-call id resolution
	// runs. It executes the IDENTICAL seed + per-candidate context-pair scorer as
	// Discover (the equivalence oracle), score-descending.
	DiscoverVecs(k int, opts DiscoverVecsOpts) ([]Result, error)
	// RecommendVecs is the BEST_SCORE recommend path (the Query API leaf form): the
	// positive/negative example VECTORS are supplied directly (the coordinator
	// resolved the ids and embedded them). It seeds a candidate pool from the
	// positives' centroid, scores each candidate by the BEST_SCORE merge (bestScore —
	// max-sim-to-positive vs max-sim-to-negative), and sorts score-descending. The
	// recommend analogue of DiscoverVecs and the equivalence oracle for BEST_SCORE.
	RecommendVecs(k int, opts RecommendVecsOpts) ([]Result, error)
	SearchGroups(query []float32, k int, opts GroupOpts) ([]Group, error)
	GroupCandidates(query []float32, opts GroupOpts) ([]Document, error)

	// --- Payload mutations (return the resulting payload + per-key deadlines
	// for the WAL, mirroring how Collection logs them, PLUS the resulting version
	// after the bump). cas is the optimistic-CAS precondition (CASCond{} = no
	// precondition); a mismatch returns ErrVersionConflict with no mutation. ---
	SetPayload(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	OverwritePayload(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	DeletePayloadKeys(id uint64, keys []string, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	ClearPayload(id uint64, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	// The ...At variants judge the per-key deadline computation (Set/Overwrite) AND
	// the dead-point liveness gate against the EXPLICIT leader-stamped clock nowMs,
	// so a replicated payload mutation stamps byte-identical per-key deadlines and
	// agrees on liveness on every replica (#4 vector TTL determinism). The non-At
	// forms use the wall clock and are byte-identical to before.
	SetPayloadAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	OverwritePayloadAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	DeletePayloadKeysAt(id uint64, keys []string, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	ClearPayloadAt(id uint64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	// MutatePayloadRecord applies fn to the operate record stored under key in
	// id's payload and stores what fn returns, in ONE write-lock critical
	// section (read, transform, bound, store/delete, reindex, bump). It is the
	// store-agnostic seam the ops vector_operate handler drives: vector never
	// imports ops, so the transform arrives as a function.
	//
	// Returns the resulting payload, the resulting absolute per-key deadlines,
	// the resulting version, changed, and an error. changed=false means the
	// call was a deliberate no-op (a failed CHECK, or a delete of an absent
	// key): nothing was written and the returned version is the CURRENT,
	// UNBUMPED one, so a no-op costs neither a WAL record nor a version move a
	// CAS loop is watching.
	MutatePayloadRecord(id uint64, key string, fn RecordMutator, cas CASCond) (Metadata, map[string]uint64, uint64, bool, error)
	// MutatePayloadRecordAt judges the dead-point liveness gate, the per-key
	// deadline check and the stale-deadline drop against the EXPLICIT
	// leader-stamped clock nowMs, so a replicated record mutation reads the
	// same record and produces the same bytes on every replica (#4 vector TTL
	// determinism, mirroring SetPayloadAt).
	MutatePayloadRecordAt(id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, bool, error)
	// RestorePayload restores a logged payload + per-key deadlines AND the exact
	// version verbatim (NOT bumped) — the WAL-replay primitive.
	RestorePayload(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) error

	// --- Doc enrichment / scan / scroll (RAG + resplit + fan-out primitives) ---
	fetchDocs(results []Result) []Document
	matchingIDs(filter Filter, pred Predicate) ([]uint64, error)
	// matchingIDsAt is matchingIDs judging the tombstone/TTL admission gate against
	// the EXPLICIT leader-stamped clock nowMs, so a replicated delete-by-filter
	// selects the SAME id set on every replica (#4 vector TTL determinism).
	matchingIDsAt(filter Filter, pred Predicate, nowMs uint64) ([]uint64, error)
	scanVectors() []ScanRecord
	scrollDocs(filter Filter, limit int) ([]Document, error)
	// scrollPage walks the live id set and returns up to `limit` documents.
	//
	//   - order == nil: today's deterministic id-ASCENDING resume-after-id scroll
	//     (the cached scrollSnap path, byte-identical / zero-overhead). afterKey is
	//     ignored.
	//   - order != nil: order_by scroll — the admitted live set is sorted per call by
	//     the (value, id) total order for order.Key/Desc (see vector.OrderLess), ids
	//     whose order field is missing/non-numeric are EXCLUDED, and the page resumes
	//     STRICTLY PAST the cursor (afterKey, afterID) — or past order.StartFrom on
	//     page 1. Returned docs carry the order field in Metadata so the coordinator
	//     can read the last doc's order value for the next (v2) cursor.
	//
	// filter is the ORIGINAL (uncompiled) filter; pred is its compiled form. When a
	// filter is present (pred != nil) AND order == nil (the id-ascending path), the
	// implementation MAY consult the payload index for a candidate SUPERSET and walk
	// only those ids (filter-first narrowing) instead of the full id snapshot — the
	// SAME predicate recheck still runs per candidate, so the page (ids/order/cursor)
	// is identical to the predicate-eval walk. Falls back to the full-snapshot walk
	// when the filter is not index-narrowable or not selective. filter is unused on
	// the order_by path (it stays predicate-eval) and on the no-filter path.
	scrollPage(filter Filter, pred Predicate, metaOf metaProvider, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool)

	// --- Maintenance ---
	sweepOnce() int
	Reclaim() int

	// --- Persistence ---
	Snapshot(w io.Writer) error
	Restore(r io.Reader) error
	SavePersist(metaPath string) error

	// --- Observability / lifecycle ---
	Stats() Stats
	Close() error

	// SetNowFunc overrides the wall-clock source (unix millis) the index's non-apply
	// expiry sites consult (sweeper + read/query filter + the wall-clock branch of
	// the write paths). nil restores the real clock. TEST/advanced seam mirroring
	// cache.Cache.SetNowFunc; production never calls it, so the default is
	// byte-identical to time.Now. It does NOT affect the stamped apply path (InsertAt
	// et al. take an explicit stamp).
	SetNowFunc(fn func() int64)
}

// Error sentinels returned by VectorIndex implementations.
var (
	ErrInvalidIVFPersistent = errors.New("vector: IVF index is snapshot-only in v1 and cannot be combined with Persistent (mmap instant-restart); use snapshot/WAL durability instead")
	ErrPQDropVecsReadOnly   = errors.New("vector: index is read-only after PQDropVecs dropped the float vectors (rebuild to add points)")
	ErrFullTextDisabled     = errors.New("vector: full-text search requires a collection created with FullText enabled")
	ErrEmptyFilter          = errors.New("vector: DeleteByFilter requires a non-empty filter (refusing to delete the whole collection)")
	ErrEmptyGroupBy         = errors.New("vector: SearchGroups requires a non-empty GroupBy field")
	ErrEmptyDocument        = errors.New("vector: multi-vector Add requires at least one token vector")

	// Named-vector (NamedCollection) sentinels — fail-loud on a malformed config
	// or a request that names a space that does not exist.
	ErrEmptyNamedVectors  = errors.New("vector: named collection requires at least one named vector space")
	ErrEmptyVectorName    = errors.New("vector: named vector name must be non-empty")
	ErrReservedVectorName = errors.New("vector: named vector name must not contain reserved characters ('#', '@', '/')")
	ErrUnknownVectorName  = errors.New("vector: unknown named vector space")
	// ErrSpaceModalityMismatch is returned when a request supplies the wrong value
	// kind for a named space's modality: a sparse value for a DENSE space, or a
	// dense value for a SPARSE space. Fail loud (the wire fail-loud mapping follows).
	ErrSpaceModalityMismatch = errors.New("vector: value modality does not match named space (dense vs sparse)")
	ErrNoRecommendExamples   = errors.New("vector: recommend requires at least one positive example")
	// ErrRecommendBestScoreL2Negatives rejects a BEST_SCORE recommend that combines
	// the L2 (Euclidean) metric with negative examples. BEST_SCORE's piecewise merge
	// (max_pos > max_neg ? max_pos : -max_neg) is a SCORE-STYLE strategy: it assumes
	// sign-meaningful, higher-is-better similarities, which hold for the inner-product
	// metrics (Cosine: 1-dist; DotProduct: -dist). For L2 the similarity is -squared-L2,
	// which is ALWAYS <= 0, so the -max_neg sign-flip in the else branch produces a
	// POSITIVE score that outranks every (non-positive) positive-dominated candidate —
	// inverting the ranking (a candidate ON a positive ranks last; a candidate NEAR a
	// negative ranks first). Fail loud rather than return an inverted ranking. L2 +
	// BEST_SCORE with NO negatives is well-defined (score = nearest-positive similarity,
	// monotonic) and stays allowed; use Cosine/DotProduct for L2-shaped data with
	// negatives, or AVERAGE_VECTOR (which steers via mean(pos)-mean(neg) and is sound for L2).
	ErrRecommendBestScoreL2Negatives = errors.New("vector: best_score recommend with negative examples requires a score-style metric (cosine/dot-product), not L2")
	ErrNoContextPairs                = errors.New("vector: discover requires at least one context pair")
	ErrReserveNonEmpty               = errors.New("vector: Reserve requires an empty arena")
	ErrBuildNonEmpty                 = errors.New("vector: BuildConcurrent requires an empty index")
	ErrBuildLenMismatch              = errors.New("vector: BuildConcurrent ids and vecs length mismatch")
	ErrBuildMetaLenMismatch          = errors.New("vector: BuildConcurrentMeta payloads must be empty or one per id")
	ErrDuplicateID                   = errors.New("vector: id already present (delete first)")
	ErrIDNotFound                    = errors.New("vector: id not found")

	// ErrVersionConflict is returned by an optimistic-CAS mutation whose
	// CASCond.Expected does not match the point's current version (0 if absent).
	// On a conflict the mutation is NOT applied and the version is NOT bumped, so
	// it is safe to retry after re-reading the current version (Get returns it).
	ErrVersionConflict = errors.New("vector: version conflict (optimistic CAS expected_version mismatch)")
	ErrEmptyIndex      = errors.New("vector: index is empty")
	ErrSnapshotFormat  = errors.New("vector: snapshot has invalid format or version")
	ErrNotImplemented  = errors.New("vector: not implemented in this version")

	// ErrCollectionFull is returned when an insert would exceed the collection's
	// MaxVectors or MaxBytes quota.
	ErrCollectionFull = errors.New("vector: collection full (quota exceeded)")

	// ErrCollectionRateLimited is returned when MaxInsertsPerSecond is configured
	// and the token bucket is empty. Callers can retry after the bucket refills.
	ErrCollectionRateLimited = errors.New("vector: collection insert rate limited")

	// ErrIndexPoisoned is returned by every public op once a memory-mapped slab
	// growth (vecs or the level-0 graph) has failed. That failure unmaps the old
	// view before it can remap, so the backing region is gone and cannot be
	// recovered in place; the index is in a TERMINAL state and each op rejects
	// rather than dereference the freed mapping. It is not retryable — the
	// collection must be reopened/restored.
	ErrIndexPoisoned = errors.New("vector: index poisoned by a failed mmap growth")
)
