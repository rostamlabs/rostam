// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// writeConsistency carries the two OPTIONAL tunable-write-consistency knobs that
// every dense/MV/named WRITE body accepts. Embedded into each write-request
// struct so the fields are flat in the JSON body. Both are zero-valued by
// default — WriteConsistencyFactor=0 (no barrier) and Wait nil (treated as the
// default true) — so an old client body that omits them produces the exact same
// dispatch as before (no __wc__ envelope, byte-identical default path).
//
// Wait is a *bool (not bool) so the API contract "wait defaults to true" holds:
// an absent field decodes to nil ⇒ wait=true; an explicit `"wait":false` is the
// only way to skip the barrier. A plain bool would default to false and silently
// invert the contract.
type writeConsistency struct {
	WriteConsistencyFactor int   `json:"write_consistency_factor,omitempty"`
	Wait                   *bool `json:"wait,omitempty"`
}

// wait reports the effective wait flag (default true when the field is absent).
func (wc writeConsistency) wait() bool { return wc.Wait == nil || *wc.Wait }

// validate enforces write_consistency_factor >= 0 at the edge (negative ⇒ 400
// BEFORE dispatch). A value larger than the collection's RF is fine — the
// barrier clamps it to [1, RF]. Returns false (after writing the 400) on a
// negative factor.
func (wc writeConsistency) validate(w http.ResponseWriter) bool {
	if wc.WriteConsistencyFactor < 0 {
		writeError(w, http.StatusBadRequest, "write_consistency_factor must be non-negative")
		return false
	}
	return true
}

func (a *api) health(w http.ResponseWriter, r *http.Request) {
	// __ping__ needs no auth — it is a liveness probe.
	if _, err := a.disp.Call("__ping__", nil); err != nil {
		writeDispatchError(w, "__ping__", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ready is the READINESS probe (auth-exempt, like health): it calls the
// __ready__ op — always ready in single-node/Direct, a real hosted-shard leader
// check in cluster mode — and returns 200 {"status":"ready"} or 503
// {"status":"not ready", "detail": ...}. A load balancer routes on THIS, not
// /v1/health, so it stops sending writes to a node whose shard lost its leader.
func (a *api) ready(w http.ResponseWriter, r *http.Request) {
	if _, err := a.disp.Call(ops.ReadyOp, nil); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready", "detail": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// metrics serves the Prometheus text exposition for all dense collections. It
// dispatches the __metrics__ read op (authorized like any other read via
// a.call: open when no authenticator is configured, otherwise scope-gated), and
// writes the raw exposition body with the Prometheus content type.
func (a *api) metrics(w http.ResponseWriter, r *http.Request) {
	body, ok := a.call(w, r, ops.MetricsOp, nil)
	if !ok {
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// kvMetrics serves the node's KV cache stats as Prometheus text via the
// __kv_metrics__ read op. Separate from /metrics on purpose: that endpoint
// renders dense-collection stats and returns an empty body on a KV-only node,
// and folding the two together would change what existing vector scrapers see.
func (a *api) kvMetrics(w http.ResponseWriter, r *http.Request) {
	body, ok := a.call(w, r, ops.KVMetricsOp, nil)
	if !ok {
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// replication serves the per-hosted-shard replication view (mode, primary, ISR
// vs min-ISR, per-backup lag) as JSON. Like ready it is auth-exempt (an ops
// probe carries no token) and dispatches directly: the __repl_metrics__ op
// returns a ready-made JSON body — empty {"shards":[]} in single-node/Direct, a
// real per-shard view in cluster mode — which is written through verbatim. A
// dispatch error surfaces as the standard dispatch error mapping.
func (a *api) replication(w http.ResponseWriter, r *http.Request) {
	body, err := a.disp.Call(ops.ReplMetricsOp, nil)
	if err != nil {
		writeDispatchError(w, ops.ReplMetricsOp, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// collectionConfig is the friendly JSON form of vector.Config: metric and quant
// are readable strings rather than the integer enums the engine uses.
type collectionConfig struct {
	Dim            int    `json:"dim"`
	Metric         string `json:"metric"` // "cosine" (default) | "l2" | "dot"
	M              int    `json:"m"`      // graph degree (0 = engine default)
	EfConstruction int    `json:"ef_construction"`
	EfSearch       int    `json:"ef_search"`
	Seed           int64  `json:"seed"`
	Quant          string `json:"quant"` // "" / "none" | "sq8" | "bq1" | "pq" | "sq" | "prq"
	Persistent     bool   `json:"persistent"`
	RescoreFactor  int    `json:"rescore_factor"`
	// SQBits is the trained scalar quantizer's bit-depth for quant == "sq": 4, 6,
	// or 8 (0 = engine default 8). Ignored unless quant == "sq".
	SQBits int `json:"sq_bits"`
	// PRQLayers is the product-residual quantization layer count for quant == "prq"
	// (0 = engine default 2; code = prq_layers*quant_pq_m bytes). Ignored unless
	// quant == "prq".
	PRQLayers  int `json:"prq_layers"`
	Partitions int `json:"partitions"`
	// ExtendCandidates enables the extendCandidates build heuristic (richer,
	// more diverse graph → higher recall at a given ef; build-time cost only).
	// ExtendCandidatesMax bounds the enriched pool (0 = unbounded).
	ExtendCandidates    bool `json:"extend_candidates"`
	ExtendCandidatesMax int  `json:"extend_candidates_max"`
	// Level0FullDegree selects 2*M forward neighbors at level 0 (Qdrant's m0
	// convention) for higher recall@k at large k; build-time cost only.
	Level0FullDegree bool `json:"level0_full_degree"`
	// QuantizedBuild builds the graph on int8 codes (4x less memory traffic →
	// faster high-dim ingest), slightly lower recall (rescore recovers it).
	QuantizedBuild bool `json:"quantized_build"`
	// IndexType selects the backing index: "" / "hnsw" (default), "ivf", or
	// "vamana". IVFNlist / IVFNprobe parameterize an IVF-Flat index (0 = engine
	// default).
	IndexType string `json:"index_type"`
	IVFNlist  int    `json:"ivf_nlist"`
	IVFNprobe int    `json:"ivf_nprobe"`
	// VamanaR / VamanaL / VamanaAlpha parameterize a Vamana (DiskANN) index
	// (index_type == "vamana"): VamanaR is the max out-degree (0 = engine default
	// 64), VamanaL the build/search beam width (0 = default 100), VamanaAlpha the
	// pass-2 RobustPrune α (0 = default 1.2). Ignored unless index_type == "vamana".
	VamanaR     int     `json:"vamana_r"`
	VamanaL     int     `json:"vamana_l"`
	VamanaAlpha float32 `json:"vamana_alpha"`
	// IVF-PQ: ivf_pq enables residual product-quantization (compact codes + ADC);
	// ivf_pq_m is the sub-quantizer count (0 = engine default; dim must divide it);
	// ivf_rerank keeps full vectors for an exact rescore of the ADC shortlist.
	IVFPQ     bool `json:"ivf_pq"`
	IVFPQM    int  `json:"ivf_pq_m"`
	IVFRerank bool `json:"ivf_rerank"`
	// QuantPQM is the PQ sub-quantizer count for a PQ-HNSW index (quant == "pq");
	// 0 = engine default. dim must divide it. Ignored unless quant == "pq".
	QuantPQM int `json:"quant_pq_m"`
	// OPQ enables an OPQ orthogonal rotation R (d×d) inside the PQ codec for higher
	// recall at the same M/nbits. Requires a PQ mode (ivf_pq, or quant == "pq");
	// rejected with 400 otherwise. Default false.
	OPQ bool `json:"opq"`
	// OPQIters drives full-OPQ iterative Procrustes refinement (0 = 1 = the v1
	// single-random-rotation behavior, byte-identical; > 1 = that many refine
	// iterations). Ignored unless opq is set. Must be in [0, 20] (400 otherwise).
	OPQIters int `json:"opq_iters"`
	// PQDropVecs (HNSW-PQ only) drops the resident float vectors after the bulk
	// build for maximum compression (ADC-only search). Requires quant == "pq";
	// rejected with 400 otherwise. Default false.
	PQDropVecs bool `json:"pq_drop_vecs"`
	// IVFTrainThreshold is the live-vector count at which an incrementally-built
	// IVF index (index_type == "ivf") deterministically auto-trains its coarse
	// quantizer (and, for IVF-PQ, the residual codebooks). 0 = engine default.
	// Ignored unless index_type == "ivf".
	IVFTrainThreshold int `json:"ivf_train_threshold"`
	// IVFDriftRetrain opts an IVF index into deterministic auto-retrain-on-drift;
	// IVFDriftGrowthFactor (default 2.0) / IVFDriftFactor (default 1.5) are the
	// two-stage trigger thresholds (must be > 1.0; 0 = default). Ignored unless
	// index_type == "ivf".
	IVFDriftRetrain      bool    `json:"ivf_drift_retrain"`
	IVFDriftGrowthFactor float64 `json:"ivf_drift_growth_factor"`
	IVFDriftFactor       float64 `json:"ivf_drift_factor"`
	// FilterFirstRelativeBP is the opt-in relative selectivity gate (basis points of
	// the live collection size, 0..10000; 0 = off = byte-identical absolute behavior).
	FilterFirstRelativeBP int `json:"filter_first_relative_bp"`
	// FullText, when non-nil, enables the server-side BM25 full-text lane: each
	// upserted record's reserved $content is tokenized + indexed so search/text and
	// search/hybrid-text work. Omitted (nil) = no full-text lane (byte/behavior
	// identical to before). See vector.FullTextConfig (analyzer name + k1/b knobs).
	FullText *fullTextConfig `json:"full_text"`
	// AnisotropicEta is the ScaNN score-aware PQ quantization weight (η ≥ 1; 0/1 =
	// isotropic L2, byte-identical). > 1 weights parallel (MIPS-score) error more on
	// PQ codebook training. Ignored unless a PQ mode is selected.
	AnisotropicEta float32 `json:"anisotropic_eta"`
	// SOAR opts an IVF index (index_type == "ivf") into ScaNN-style multi-assignment
	// (each point joins a secondary cell whose residual is most orthogonal to the
	// primary), raising recall at fixed nprobe. Rejected on non-IVF (400). Default false.
	SOAR bool `json:"soar"`
	// SOARLambda is the orthogonality-amplification weight λ in the SOAR secondary
	// assignment loss (0 = engine default 1.5; must be >= 0). Ignored unless soar.
	SOARLambda float32 `json:"soar_lambda"`
	// PQNBits is the per-subspace PQ code width: 0/8 = 256 sub-centroids (8-bit, the
	// default), 4 = 16 sub-centroids (4-bit LUT16 fast-scan). Ignored unless a PQ
	// mode is selected.
	PQNBits int `json:"pq_nbits"`
}

// fullTextConfig is the friendly JSON form of vector.FullTextConfig.
type fullTextConfig struct {
	Analyzer string  `json:"analyzer"` // registered analyzer name; "" = "english"
	K1       float32 `json:"k1"`       // BM25 tf-saturation; 0 = 1.2
	B        float32 `json:"b"`        // BM25 length-norm; 0 = 0.75
}

func parseMetric(s string) (vector.Metric, bool) {
	switch s {
	case "", "cosine":
		return vector.Cosine, true
	case "l2", "euclidean":
		return vector.L2, true
	case "dot", "dotproduct", "ip":
		return vector.DotProduct, true
	}
	return 0, false
}

func parseQuant(s string) (vector.QuantMode, bool) {
	switch s {
	case "", "none":
		return vector.QuantNone, true
	case "sq8":
		return vector.QuantSQ8, true
	case "bq1":
		return vector.QuantBQ1, true
	case "pq":
		return vector.QuantPQ, true
	case "sq":
		return vector.QuantSQ, true
	case "prq":
		return vector.QuantPRQ, true
	}
	return 0, false
}

func (cc collectionConfig) toConfig() (vector.Config, string) {
	metric, ok := parseMetric(cc.Metric)
	if !ok {
		return vector.Config{}, "unknown metric " + strconv.Quote(cc.Metric)
	}
	quant, ok := parseQuant(cc.Quant)
	if !ok {
		return vector.Config{}, "unknown quant " + strconv.Quote(cc.Quant)
	}
	if cc.Partitions < 0 {
		return vector.Config{}, "partitions must be non-negative"
	}
	indexType, ok := parseIndexType(cc.IndexType)
	if !ok {
		return vector.Config{}, "unknown index_type " + strconv.Quote(cc.IndexType)
	}
	if cc.IVFNlist < 0 || cc.IVFNprobe < 0 {
		return vector.Config{}, "ivf_nlist and ivf_nprobe must be non-negative"
	}
	// Caught early (before the uint32 create-wire encoding silently wraps a
	// negative to a large positive) so a bad threshold fails loud, mirroring the
	// engine's Validate rule (IVFTrainThreshold >= 0).
	if cc.IVFTrainThreshold < 0 {
		return vector.Config{}, "ivf_train_threshold must be non-negative"
	}
	// Drift-retrain factors must be > 1.0 (0 = engine default), mirroring the engine
	// Validate rule. Fail loud before the create wire.
	if cc.IVFDriftGrowthFactor != 0 && cc.IVFDriftGrowthFactor <= 1.0 {
		return vector.Config{}, "ivf_drift_growth_factor must be > 1.0"
	}
	if cc.IVFDriftFactor != 0 && cc.IVFDriftFactor <= 1.0 {
		return vector.Config{}, "ivf_drift_factor must be > 1.0"
	}
	// FilterFirstRelativeBP is basis points (0..10000); fail loud before the create
	// wire, mirroring the engine's Validate rule.
	if cc.FilterFirstRelativeBP < 0 || cc.FilterFirstRelativeBP > 10000 {
		return vector.Config{}, "filter_first_relative_bp must be in [0, 10000]"
	}
	// Map the friendly full-text config onto vector.FullTextConfig (nil = disabled).
	// The engine Validate enforces HNSW-only + a registered analyzer name.
	var fullText *vector.FullTextConfig
	if cc.FullText != nil {
		fullText = &vector.FullTextConfig{
			Analyzer: cc.FullText.Analyzer,
			K1:       cc.FullText.K1,
			B:        cc.FullText.B,
		}
	}
	// Fill the standard HNSW defaults so a caller need only supply dim. The
	// engine now also defaults these (vector.applyHNSWDefaults), so this is
	// belt-and-suspenders rather than load-bearing — kept so the HTTP contract
	// does not depend on engine-internal behaviour.
	m, efc, efs := cc.M, cc.EfConstruction, cc.EfSearch
	if m <= 0 {
		m = 16
	}
	if efc <= 0 {
		efc = 200
	}
	if efs <= 0 {
		efs = 64
	}
	return vector.Config{
		Dim:                   cc.Dim,
		Metric:                metric,
		M:                     m,
		EfConstruction:        efc,
		EfSearch:              efs,
		Seed:                  cc.Seed,
		Quant:                 quant,
		SQBits:                cc.SQBits,
		PRQLayers:             cc.PRQLayers,
		Persistent:            cc.Persistent,
		RescoreFactor:         cc.RescoreFactor,
		ExtendCandidates:      cc.ExtendCandidates,
		ExtendCandidatesMax:   cc.ExtendCandidatesMax,
		Level0FullDegree:      cc.Level0FullDegree,
		QuantizedBuild:        cc.QuantizedBuild,
		Partitions:            cc.Partitions,
		IndexType:             indexType,
		VamanaR:               cc.VamanaR,
		VamanaL:               cc.VamanaL,
		VamanaAlpha:           cc.VamanaAlpha,
		IVFNlist:              cc.IVFNlist,
		IVFNprobe:             cc.IVFNprobe,
		IVFPQ:                 cc.IVFPQ,
		IVFPQM:                cc.IVFPQM,
		IVFRerank:             cc.IVFRerank,
		QuantPQM:              cc.QuantPQM,
		OPQ:                   cc.OPQ,
		OPQIters:              cc.OPQIters,
		PQDropVecs:            cc.PQDropVecs,
		IVFTrainThreshold:     cc.IVFTrainThreshold,
		IVFDriftRetrain:       cc.IVFDriftRetrain,
		IVFDriftGrowthFactor:  cc.IVFDriftGrowthFactor,
		IVFDriftFactor:        cc.IVFDriftFactor,
		FilterFirstRelativeBP: cc.FilterFirstRelativeBP,
		FullText:              fullText,
		AnisotropicEta:        cc.AnisotropicEta,
		SOAR:                  cc.SOAR,
		SOARLambda:            cc.SOARLambda,
		PQNBits:               cc.PQNBits,
	}, ""
}

func parseIndexType(s string) (vector.IndexType, bool) {
	switch s {
	case "", "hnsw":
		return vector.IndexHNSW, true
	case "ivf", "ivf-flat", "ivfflat":
		return vector.IndexIVF, true
	case "vamana", "diskann":
		return vector.IndexVamana, true
	}
	return 0, false
}

type createCollectionReq struct {
	Name   string           `json:"name"`
	Config collectionConfig `json:"config"`
}

func (a *api) createCollection(w http.ResponseWriter, r *http.Request) {
	var req createCollectionReq
	if !decodeBody(w, r, &req) {
		return
	}
	if strings.ContainsAny(req.Name, "#@") {
		writeError(w, http.StatusBadRequest,
			"vector: collection name "+strconv.Quote(req.Name)+" must not contain reserved characters '#' or '@'")
		return
	}
	if !validName(w, req.Name) {
		return
	}
	cfg, errMsg := req.Config.toConfig()
	if errMsg != "" {
		writeError(w, http.StatusBadRequest, errMsg)
		return
	}
	if _, ok := a.call(w, r, "vector_create_collection", ops.EncodeCreateCollectionArgs(req.Name, cfg)); !ok {
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"name": req.Name})
}

func (a *api) dropCollection(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if _, ok := a.call(w, r, "vector_drop_collection", ops.EncodeDropCollectionArgs(name)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"dropped": name})
}

// pointReq is the body for inserting or upserting a point. Upsert (the RAG
// write path, storing Content) is selected with "upsert": true; a plain insert
// rejects a duplicate id.
type pointReq struct {
	ID       uint64               `json:"id"`
	Vector   []float32            `json:"vector"`
	Content  string               `json:"content"`
	TTLMs    int64                `json:"ttl_ms"`
	Metadata vector.Metadata      `json:"metadata"`
	Sparse   *vector.SparseVector `json:"sparse"`
	Upsert   bool                 `json:"upsert"`
	// ExpectedVersion is the optimistic-CAS precondition: the per-point version the
	// caller expects the point to currently have. When non-nil the write applies
	// ONLY when it matches (0 = expect-absent/new); a mismatch is 409 Conflict. nil
	// (absent in the JSON) = an unconditional write (byte-identical to pre-feature).
	ExpectedVersion *uint64 `json:"expected_version"`
	// KeyTTLMs is an OPTIONAL per-key payload TTL map (payload key -> RELATIVE ms).
	// At insert/upsert the engine computes the absolute deadline now+ttl per key
	// (mirroring set_payload's key_ttl_ms) and lazily drops the key once it passes,
	// while the point lives on. Absent/empty = no per-key TTL (byte-identical wire).
	KeyTTLMs map[string]int64 `json:"key_ttl_ms"`
	writeConsistency
}

func (a *api) putPoint(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req pointReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !req.validate(w) {
		return
	}
	ttl := time.Duration(req.TTLMs) * time.Millisecond
	var sparse vector.SparseVector
	if req.Sparse != nil {
		sparse = *req.Sparse
	}
	exp, hasExp := uint64(0), false
	if req.ExpectedVersion != nil {
		exp, hasExp = *req.ExpectedVersion, true
	}
	if req.Upsert {
		args := ops.EncodeVectorUpsertArgsCASKeyTTL(name, req.ID, req.Vector, req.Content, ttl, req.Metadata, sparse, exp, hasExp, req.KeyTTLMs)
		if _, ok := a.callWrite(w, r, "vector_upsert", args, req.WriteConsistencyFactor, req.wait()); !ok {
			return
		}
	} else {
		meta := vector.WithContent(req.Metadata, req.Content)
		args := ops.EncodeVectorInsertArgsCASKeyTTL(name, req.ID, req.Vector, ttl, meta, sparse, exp, hasExp, req.KeyTTLMs)
		if _, ok := a.callWrite(w, r, "vector_insert", args, req.WriteConsistencyFactor, req.wait()); !ok {
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]uint64{"id": req.ID})
}

// pointsBatchReq is the body for a batch insert/upsert: many points in one HTTP
// request, so bulk ingestion pays a single round-trip instead of one per point.
type pointsBatchReq struct {
	Upsert bool       `json:"upsert"`
	Points []pointReq `json:"points"`
}

// batchPointResult is one per-point entry in the CAS-mode /points/batch response
// (emitted only when at least one point in the batch carries expected_version;
// see putPointsBatch). Status is "ok" for an applied write or "conflict" for an
// optimistic-CAS precondition miss (the per-id equivalent of the single-point
// 409 — a conflict does NOT fail the whole batch). Version is the point's
// post-write version, present only on an applied ("ok") write.
type batchPointResult struct {
	ID      uint64 `json:"id"`
	Version uint64 `json:"version,omitempty"`
	Status  string `json:"status"` // "ok" | "conflict"
}

// isVersionConflict reports whether a dispatch error is an optimistic-CAS
// precondition miss (the same condition statusForError maps to 409). It matches
// both the local sentinel and the string fallback for the clustered path where
// the sentinel is stringified across the Raft boundary — identical to the
// detection in statusForError so the per-id batch path and the single-point 409
// path agree on what "conflict" means.
func isVersionConflict(err error) bool {
	return errors.Is(err, vector.ErrVersionConflict) ||
		strings.Contains(err.Error(), "version conflict")
}

// putPointsBatch applies a batch of inserts/upserts within a single request. It
// authorizes once, then dispatches each point through the same op as putPoint —
// the round-trip (the dominant per-point network cost) is amortized across the
// whole batch.
//
// Two response shapes, selected by whether ANY point carries expected_version:
//
//	No-CAS body (no point has expected_version): byte-identical to the pre-CAS
//	contract — every point is an unconditional write (the plain insert/upsert
//	encoder), the loop stops at the first failing point and reports {"error",
//	"committed":<n>} with that point's status, and on full success returns
//	{"count":<len>}. A no-CAS body therefore behaves EXACTLY as before.
//
//	CAS body (at least one point carries expected_version): each point with
//	expected_version is a conditional write (EncodeVectorInsertArgsCAS /
//	EncodeVectorUpsertArgsCAS with its per-point CASCond, mirroring putPoint);
//	points without it stay unconditional. A per-point version conflict does NOT
//	fail the batch — it is reported as {"id":N,"status":"conflict"} (the per-id
//	409 equivalent) while other points still apply, and the overall response is
//	HTTP 200 with {"results":[{id,version,status}...]}. A TRUE error (unreachable
//	shard, bad collection, malformed args) still fails the whole batch loud
//	(same status mapping + {"error","committed"} as the no-CAS path) — only a
//	genuine ErrVersionConflict is downgraded to a per-id conflict, never a real
//	error. The post-write version on an applied point is read back with a
//	follow-up vector_get (the insert/upsert op returns no body), matching what a
//	CAS client would otherwise issue itself.
//
// NOTE: tunable write consistency (write_consistency_factor / wait) is NOT
// supported on the batch/bulk paths. Rather than silently degrade an accepted
// durability request, a batch carrying any per-point write_consistency_factor>0
// or wait=false is REJECTED with 400 up front (see the guard below). Full
// per-point support is deferred to a follow-up. Callers that need a
// write-consistency factor use the single-point putPoint route.
// A body sent as application/octet-stream is the dense binary re-encoding of the
// SAME request (see binary_bulk.go): it decodes into this very pointsBatchReq and
// then runs the identical apply path below, so there is exactly one set of batch
// semantics. Any other content type takes the byte-identical JSON path.
func (a *api) putPointsBatch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req pointsBatchReq
	if isBinaryBulk(r) {
		// The binary decoder authorizes itself, ahead of consuming any section the
		// client sized (see decodeBinPointsBatch).
		if !a.decodeBinPointsBatch(w, r, name, &req) {
			return
		}
	} else {
		if !decodeBodyBulk(w, r, &req) {
			return
		}
		// JSON authorizes AFTER decoding because the upsert flag — which selects
		// insert vs upsert scope — is a field of the body, so the correct op is not
		// knowable until the body is parsed. That is safe here in a way it would not
		// be on the binary path: a JSON body is bounded by MaxBytesReader at bytes
		// ACTUALLY read, with no declared count to amplify them.
		opName := "vector_insert"
		if req.Upsert {
			opName = "vector_upsert"
		}
		// Authorize ONCE for the whole batch against the collection it targets. Every
		// point in a batch shares the same collection (`name`, from the path), so a
		// representative args buffer carrying that collection is sufficient for the
		// per-collection scope check — the authorizer only reads the collection-name
		// field, not the per-point id/vector. We build it with the same encoder used
		// below so the wire layout (and thus CollectionNameFor) is identical.
		var authArgs []byte
		if req.Upsert {
			authArgs = ops.EncodeVectorUpsertArgs(name, 0, nil, "", 0, nil, vector.SparseVector{})
		} else {
			authArgs = ops.EncodeVectorInsertArgsExt(name, 0, nil, 0, nil, vector.SparseVector{})
		}
		if !a.authorize(w, r, opName, authArgs) {
			return
		}
	}
	opName := "vector_insert"
	if req.Upsert {
		opName = "vector_upsert"
	}

	// Tunable write consistency is NOT honored on the batch route: the __wc__
	// envelope + post-commit barrier are engaged only per single-point callWrite,
	// so a per-point write_consistency_factor>0 or wait=false would be SILENTLY
	// dropped here — the batch would ack at Raft majority only while the caller
	// believed the stronger factor held. Fail loud instead of silently degrading
	// (mirroring the codebase's fail-loud posture): reject the batch with 400 so
	// the weaker guarantee is never accepted unnoticed. Callers that need a
	// write-consistency factor use the single-point /points route (see putPoint).
	for i := range req.Points {
		if p := &req.Points[i]; p.WriteConsistencyFactor > 0 || !p.wait() {
			writeError(w, http.StatusBadRequest, "write_consistency_factor/wait is not supported on the batch route; use the single-point /points route")
			return
		}
	}

	// Detect whether ANY point opts into optimistic CAS. When none do, we take the
	// exact pre-CAS path below so the no-CAS body stays byte-identical.
	useCAS := false
	for i := range req.Points {
		if req.Points[i].ExpectedVersion != nil {
			useCAS = true
			break
		}
	}

	if !useCAS {
		for i := range req.Points {
			p := &req.Points[i]
			ttl := time.Duration(p.TTLMs) * time.Millisecond
			var sparse vector.SparseVector
			if p.Sparse != nil {
				sparse = *p.Sparse
			}
			var args []byte
			if req.Upsert {
				args = ops.EncodeVectorUpsertArgsCASKeyTTL(name, p.ID, p.Vector, p.Content, ttl, p.Metadata, sparse, 0, false, p.KeyTTLMs)
			} else {
				meta := vector.WithContent(p.Metadata, p.Content)
				args = ops.EncodeVectorInsertArgsKeyTTL(name, p.ID, p.Vector, ttl, meta, sparse, p.KeyTTLMs)
			}
			if _, err := a.disp.Call(opName, args); err != nil {
				// Route through clientError so a mid-batch 500 is redacted +
				// logged exactly like the single-point path (writeDispatchError),
				// while still reporting how many points committed. 4xx keeps its
				// descriptive message.
				status, msg := clientError(opName, err)
				writeJSON(w, status, map[string]any{"error": msg, "committed": i})
				return
			}
		}
		writeJSON(w, http.StatusOK, map[string]int{"count": len(req.Points)})
		return
	}

	// CAS mode: per-point conditional writes, per-id result array, partial conflicts.
	results := make([]batchPointResult, 0, len(req.Points))
	for i := range req.Points {
		p := &req.Points[i]
		ttl := time.Duration(p.TTLMs) * time.Millisecond
		var sparse vector.SparseVector
		if p.Sparse != nil {
			sparse = *p.Sparse
		}
		exp, hasExp := uint64(0), false
		if p.ExpectedVersion != nil {
			exp, hasExp = *p.ExpectedVersion, true
		}
		var args []byte
		if req.Upsert {
			args = ops.EncodeVectorUpsertArgsCASKeyTTL(name, p.ID, p.Vector, p.Content, ttl, p.Metadata, sparse, exp, hasExp, p.KeyTTLMs)
		} else {
			meta := vector.WithContent(p.Metadata, p.Content)
			args = ops.EncodeVectorInsertArgsCASKeyTTL(name, p.ID, p.Vector, ttl, meta, sparse, exp, hasExp, p.KeyTTLMs)
		}
		if _, err := a.disp.Call(opName, args); err != nil {
			if isVersionConflict(err) {
				// Per-id conflict: the point's current version did not match its
				// expected_version. Report it inline and keep applying the rest of
				// the batch — a conflict is NOT a batch failure.
				results = append(results, batchPointResult{ID: p.ID, Status: "conflict"})
				continue
			}
			// A genuine error (unreachable shard, bad collection, malformed args)
			// still fails the whole batch loud, mirroring the no-CAS path — routed
			// through clientError so a 500 is redacted + logged like the single-point
			// path while still reporting how many points committed.
			status, msg := clientError(opName, err)
			writeJSON(w, status, map[string]any{"error": msg, "committed": i})
			return
		}
		// Applied. The insert/upsert op returns no body, so read the committed
		// version back with a follow-up get (best-effort: a get error leaves the
		// version 0/omitted but the write itself succeeded).
		results = append(results, batchPointResult{ID: p.ID, Version: a.pointVersion(name, p.ID), Status: "ok"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

// pointVersion reads back a point's current version via vector_get, used to
// surface the post-write version in the CAS-mode batch response (the insert/
// upsert op returns no body). Returns 0 on any error or a vanished point — the
// write already succeeded; the version is informational.
func (a *api) pointVersion(name string, id uint64) uint64 {
	body, err := a.disp.Call("vector_get", ops.EncodeVectorGetArgs(name, id, 0))
	if err != nil {
		return 0
	}
	found, _, _, _, _, version, err := ops.DecodeVectorGetResultV(body)
	if err != nil || !found {
		return 0
	}
	return version
}

// putPointsBulk stages a batch of (id, vector) points for a concurrent bulk
// build (see buildBulk). Unlike /points/batch (which indexes each point inline,
// single-writer), bulk staging is cheap and parallel; the actual multi-core
// HNSW build happens on /points/bulk/build. Use for fast initial load of an
// empty collection.
//
// The staging op carries ids, vectors and per-point METADATA, and NOTHING ELSE.
// A point carrying content, a sparse vector, a TTL, a per-key payload TTL or a
// CAS precondition is REJECTED with 400 rather than staged with those fields
// quietly discarded — silently dropping a field is how a caller ends up querying
// data that was never stored, or trusting an expiry that will never happen.
//
// Metadata used to be refused here too, because the staging wire had nowhere to
// put it, and that refusal is what forced every filtered workload onto
// /points/batch's inline one-indexed-insert-per-point route — measured ~6x slower
// to searchable on 1M x 768d. It now rides a second staging op
// (vector_bulk_stage_payload) and is applied by the build's placement pass.
// Content and sparse vectors still have no bulk representation, so they are still
// refused rather than dropped.
//
// A body sent as application/octet-stream takes the dense binary framing instead
// (see binary_bulk.go) — same ops, same staging buffer, same response, but the
// vectors arrive as f32 rather than as base-10 text. Any other content type takes
// the JSON path below.
func (a *api) putPointsBulk(w http.ResponseWriter, r *http.Request) {
	if isBinaryBulk(r) {
		a.putPointsBulkBinary(w, r)
		return
	}
	name := r.PathValue("name")
	var req pointsBatchReq
	if !decodeBodyBulk(w, r, &req) {
		return
	}
	// Authorize BEFORE encoding, and before the per-point scan below.
	// `a.call(..., ops.EncodeBulkStageArgs(...))` reads as auth-then-work but is
	// not: Go evaluates the argument first, so the encoder — whose buffer is sized
	// from the request body — ran ahead of the auth check inside a.call. An
	// anonymous 289 KB POST allocated 1.61 GB that way and was then told 401. The
	// representative args below carry only the collection name, which is all the
	// authorizer reads.
	if !validWireName(w, name) {
		return
	}
	if req.Upsert {
		writeError(w, http.StatusBadRequest,
			"upsert is not supported on the bulk staging route; use /points/batch")
		return
	}
	// Which of the two staging ops this request dispatches is decided here, by a
	// scan over the ALREADY-DECODED body: it allocates nothing and sizes nothing,
	// so it is safe to run ahead of the authorization below. What must not run
	// ahead of it is the ENCODER, whose buffer is sized from the request body — an
	// anonymous 289 KB POST allocated 1.61 GB that way and was then told 401.
	op := "vector_bulk_stage"
	for i := range req.Points {
		p := &req.Points[i]
		// EVERY FIELD THE STAGING WIRE CANNOT CARRY IS REFUSED, NOT DROPPED.
		//
		// key_ttl_ms is the one this list was missing and the reason it is now
		// exhaustive. It only means anything ALONGSIDE a payload, and a payload used
		// to be refused outright on this route — so it was unreachable, and dropping
		// it silently cost nothing. Accepting metadata made it reachable for the
		// first time: a caller sending {"metadata":{...},"key_ttl_ms":{"pii":86400000}}
		// would have got a 200 and keys that never expire. That is exactly the
		// failure class the paragraph above says this route exists to prevent, so
		// the fix is to name it rather than to widen the silence.
		//
		// ttl_ms and expected_version were droppable here before this change too.
		// They are refused now rather than left alone because the answer is the same
		// for all three — the bulk build's precondition takes no TTL, and staging is
		// not a CAS path, so neither has anywhere to go — and because a route whose
		// contract is being rewritten is the right moment to stop dropping things
		// quietly. A caller relying on the old silence was already losing the field.
		switch {
		case p.Content != "" || p.Sparse != nil:
			writeError(w, http.StatusBadRequest,
				"bulk staging carries vectors and metadata only; use /points/batch for points with content or sparse vectors")
			return
		case len(p.KeyTTLMs) > 0 || p.TTLMs != 0 || p.ExpectedVersion != nil:
			writeError(w, http.StatusBadRequest,
				"bulk staging carries vectors and metadata only; ttl_ms, key_ttl_ms and expected_version "+
					"have no staged representation — use /points/batch for points that need them")
			return
		}
		if len(p.Metadata) > 0 {
			op = "vector_bulk_stage_payload"
		}
	}
	if !a.authorize(w, r, op, bulkStageAuthArgs(name)) {
		return
	}
	ids := make([]uint64, len(req.Points))
	vecs := make([][]float32, len(req.Points))
	for i := range req.Points {
		ids[i] = req.Points[i].ID
		vecs[i] = req.Points[i].Vector
	}
	// A batch with no metadata at all takes the vectors-only op, byte-identical to
	// what it has always dispatched: the payload column is not merely empty, it
	// does not exist, so neither the wire nor the staging buffer grows a per-point
	// slot for it.
	var args []byte
	var err error
	if op == "vector_bulk_stage" {
		args, err = ops.EncodeBulkStageArgs(name, ids, vecs)
	} else {
		metas := make([]vector.Metadata, len(req.Points))
		for i := range req.Points {
			metas[i] = req.Points[i].Metadata
		}
		args, err = ops.EncodeBulkStagePayloadArgs(name, ids, vecs, metas)
	}
	if err != nil {
		// A ragged batch (vectors of differing lengths). The wire carries one dim
		// for the whole batch, so this cannot be encoded — and silently encoding it
		// anyway is what stored points under fabricated ids.
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if _, err := a.disp.Call(op, args); err != nil {
		writeDispatchError(w, op, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"staged": len(req.Points)})
}

// bulkStageAuthArgs builds the representative vector_bulk_stage args used for the
// per-collection scope check: the collection name and an empty batch. It is
// built with the real encoder so the wire layout the authorizer inspects (and
// thus CollectionNameFor) is identical to the args that will be dispatched, and
// an empty batch cannot fail encoding.
func bulkStageAuthArgs(name string) []byte {
	args, err := ops.EncodeBulkStageArgs(name, nil, nil)
	if err != nil {
		// Unreachable: an empty batch is trivially uniform and within every bound.
		return nil
	}
	return args
}

type bulkBuildReq struct {
	Workers int `json:"workers"` // 0 = all cores (GOMAXPROCS)
}

// buildBulk builds everything staged for the collection into the index in one
// concurrent pass. The collection must be empty (bulk load is the initial-load
// path).
func (a *api) buildBulk(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	// The server sets WriteTimeout=120s to bound slow readers, and Go applies
	// that deadline from the START of the request — so it caps how long a
	// HANDLER may run, not just how slowly a client may read. Bulk build is the
	// one endpoint that legitimately runs for minutes: building HNSW over
	// 1M x 768d took ~288s here, and the client got
	// "RemoteDisconnected: Remote end closed connection without response"
	// at the 120s mark, every time, on any dataset big enough to matter. That
	// made the endpoint unusable at realistic scale (found by VectorDBBench,
	// which calls this as its `optimize` stage).
	//
	// Clear the deadline for THIS handler only. Every other route keeps the
	// slowloris bound. No fixed larger number is right either — build time
	// scales with corpus size — and the request is still bounded by the
	// client's own timeout and by context cancellation when it disconnects.
	if rc := http.NewResponseController(w); rc != nil {
		// Not all ResponseWriters support deadlines (e.g. some middleware
		// wrappers); ErrNotSupported is fine to ignore — the handler simply
		// keeps the server default in that case.
		_ = rc.SetWriteDeadline(time.Time{})
	}
	var req bulkBuildReq
	// Body is optional; ignore decode errors on an empty body.
	if r.ContentLength > 0 {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	if _, ok := a.call(w, r, "vector_bulk_build", ops.EncodeBulkBuildArgs(name, req.Workers)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "built"})
}

func (a *api) deletePoint(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid id: "+err.Error())
		return
	}
	wc, ok := queryWC(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	body, ok := a.callWrite(w, r, "vector_delete", ops.EncodeVectorDeleteArgsCAS(name, id, exp, hasExp), wc.WriteConsistencyFactor, wc.wait())
	if !ok {
		return
	}
	deleted := len(body) > 0 && body[0] == 1
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": deleted})
}

// queryWC reads the OPTIONAL write-consistency knobs from the query string for
// the id-targeted point routes whose JSON body is already fully spoken-for (the
// DELETE-by-id route has no body; the payload set/overwrite routes' body IS the
// raw payload/metadata; the clear route has no body). Carrying WCF in the body
// there would change those routes' body contract, so they take it as
// ?write_consistency_factor=N&wait=false instead. Both params are optional:
// absent ⇒ wcf=0, wait=true (the byte-identical default path). A non-integer
// factor, a non-bool wait, or a negative factor is a 400. Returns ok=false (after
// writing the error) on failure.
func queryWC(w http.ResponseWriter, r *http.Request) (writeConsistency, bool) {
	q := r.URL.Query()
	var wc writeConsistency
	if v := q.Get("write_consistency_factor"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			writeError(w, http.StatusBadRequest, "write_consistency_factor must be an integer")
			return writeConsistency{}, false
		}
		wc.WriteConsistencyFactor = n
	}
	if v := q.Get("wait"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			writeError(w, http.StatusBadRequest, "wait must be a boolean")
			return writeConsistency{}, false
		}
		wc.Wait = &b
	}
	if !wc.validate(w) {
		return writeConsistency{}, false
	}
	return wc, true
}

// queryExpectedVersion reads the OPTIONAL ?expected_version=N optimistic-CAS
// precondition from the query string for the id-targeted routes whose JSON body
// is already spoken-for (DELETE-by-id has no body; set/overwrite payload's body IS
// the raw payload; delete-keys' body is the key list; clear has no body). When
// present the write applies ONLY when the point's current version matches (0 =
// expect-absent); a mismatch is 409 Conflict. Absent ⇒ hasExpected=false (an
// unconditional write — byte-identical to the pre-feature wire). A non-integer or
// negative value is a 400. Returns ok=false (after writing the error) on failure.
func queryExpectedVersion(w http.ResponseWriter, r *http.Request) (expected uint64, hasExpected bool, ok bool) {
	v := r.URL.Query().Get("expected_version")
	if v == "" {
		return 0, false, true
	}
	n, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "expected_version must be a non-negative integer")
		return 0, false, false
	}
	return n, true, true
}

// parseReadConsistencyBounded reads the OPTIONAL ?read_consistency= / ?on_partition_unavailable=
// query params for the id-targeted GET reads (get / named-get / mv-get / get_config),
// which have no JSON body. Both default to 0 (AnyReplica / Partial) when absent —
// byte-identical to the pre-rc wire. A non-integer or out-of-range value is a 400
// (fail-loud, mirroring validConsistency). Also reads the OPTIONAL
// ?max_staleness= query param (uint64), in effect only when rc==3
// (bounded-staleness). A malformed value is a 400. Mirrors how rc/opa are parsed.
func parseReadConsistencyBounded(w http.ResponseWriter, r *http.Request) (rc, opa uint8, bound uint64, ok bool) {
	q := r.URL.Query()
	if v := q.Get("read_consistency"); v != "" {
		n, err := strconv.ParseUint(v, 10, 8)
		if err != nil {
			writeError(w, http.StatusBadRequest, "read_consistency must be 0 (any), 1 (leader), 2 (linearizable) or 3 (bounded-staleness)")
			return 0, 0, 0, false
		}
		rc = uint8(n)
	}
	if v := q.Get("on_partition_unavailable"); v != "" {
		n, err := strconv.ParseUint(v, 10, 8)
		if err != nil {
			writeError(w, http.StatusBadRequest, "on_partition_unavailable must be 0 or 1")
			return 0, 0, 0, false
		}
		opa = uint8(n)
	}
	if v := q.Get("max_staleness"); v != "" {
		n, err := strconv.ParseUint(v, 10, 64)
		if err != nil {
			writeError(w, http.StatusBadRequest, "max_staleness must be a non-negative integer (raft entries)")
			return 0, 0, 0, false
		}
		bound = n
	}
	if !validConsistency(w, rc, opa) {
		return 0, 0, 0, false
	}
	return rc, opa, bound, true
}

// parseGetFlags reads the with_vector/with_payload query params (both default
// true) into the ops get-flags byte. Any value other than "false"/"0" keeps a
// projection on, so a bare ?with_vector or a missing param both mean "include
// it" — only an explicit false disables it. Shared by all three get handlers.
func parseGetFlags(r *http.Request) uint8 {
	var f uint8
	if projectionOn(r.URL.Query().Get("with_vector")) {
		f |= ops.GetFlagWithVector
	}
	if projectionOn(r.URL.Query().Get("with_payload")) {
		f |= ops.GetFlagWithPayload
	}
	return f
}

// projectionOn reports whether a with_* query value (default empty) keeps its
// projection on. Only an explicit "false"/"0" turns it off.
func projectionOn(v string) bool {
	switch strings.ToLower(v) {
	case "false", "0":
		return false
	default:
		return true
	}
}

// pathID parses the {id} path value as a uint64, writing 400 and returning
// ok=false on a bad id. Shared by the get + payload-mutation point handlers.
func pathID(w http.ResponseWriter, r *http.Request) (uint64, bool) {
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid id: "+err.Error())
		return 0, false
	}
	return id, true
}

// deletePayloadKeysReq is the body for the delete-keys payload op.
type deletePayloadKeysReq struct {
	Keys []string `json:"keys"`
}

// getPoint retrieves a dense point by id → {found, vector, payload, ttl_ms}.
// with_vector/with_payload query params (default both true) gate the projections.
// A not-found point (found=0 flag, NOT an op error) is HTTP 404.
func (a *api) getPoint(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	rc, opa, bound, ok := parseReadConsistencyBounded(w, r)
	if !ok {
		return
	}
	body, ok := a.call(w, r, "vector_get", ops.EncodeVectorGetArgsOpts(name, id, parseGetFlags(r), rc, opa, bound))
	if !ok {
		return
	}
	found, vec, meta, ttl, _, version, err := ops.DecodeVectorGetResultV(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "point not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"found": found, "vector": vec, "payload": meta, "ttl_ms": ttl.Milliseconds(),
		"version": version,
	})
}

// batchGetReq is the body for the batch get-by-id route: an id list plus the
// two projection flags (default both true — only an explicit false turns one off).
type batchGetReq struct {
	IDs         []uint64 `json:"ids"`
	WithVector  *bool    `json:"with_vector"`
	WithPayload *bool    `json:"with_payload"`
}

// batchGetPoint is one present point in the batch-get response: its id + the same
// projected fields a single get carries (vector / payload / ttl_ms).
type batchGetPoint struct {
	ID      uint64          `json:"id"`
	Vector  []float32       `json:"vector"`
	Payload vector.Metadata `json:"payload"`
	TTLMs   int64           `json:"ttl_ms"`
	Version uint64          `json:"version"`
}

// getPointsBatch retrieves MANY dense points by id in ONE request (POST since the
// id list can be large). Body {ids:[...], with_vector, with_payload} → {points:
// [{id,vector,payload,ttl_ms}], missing:[...]}. The coordinator scatters the ids
// to their owning partitions and merges (via the vector_get_batch op). A partial
// miss is NORMAL — absent ids come back in "missing", NOT a 404 (unlike single
// get). Empty ids → empty points + missing. A malformed body is 400.
func (a *api) getPointsBatch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req batchGetReq
	if !decodeBodyBulk(w, r, &req) {
		return
	}
	var flags uint8
	if req.WithVector == nil || *req.WithVector {
		flags |= ops.GetFlagWithVector
	}
	if req.WithPayload == nil || *req.WithPayload {
		flags |= ops.GetFlagWithPayload
	}
	body, ok := a.call(w, r, "vector_get_batch", ops.EncodeVectorGetBatchArgs(name, req.IDs, flags))
	if !ok {
		return
	}
	rows, err := ops.DecodeVectorGetBatchResult(body)
	if err != nil {
		writeInternalError(w, "vector_get_batch decode", err)
		return
	}
	points := make([]batchGetPoint, 0, len(rows))
	missing := make([]uint64, 0)
	for i := range rows {
		row := &rows[i]
		if !row.Found {
			missing = append(missing, row.ID)
			continue
		}
		points = append(points, batchGetPoint{
			ID:      row.ID,
			Vector:  row.Vec,
			Payload: row.Meta,
			TTLMs:   int64(row.TTLMs), //nolint:gosec // TTL ms >= 0
			Version: row.Version,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"points": points, "missing": missing})
}

// applyPayload dispatches a payload-mutation op and renders the applied flag: an
// absent point (applied=0, NOT an op error) is HTTP 404; otherwise {"applied":true}.
// It reads the OPTIONAL write-consistency knobs from the query string (the body
// is reserved for the payload itself) and dispatches via callWrite, so a payload
// mutation engages the barrier exactly when ?write_consistency_factor>0 or
// ?wait=false; the default path is byte-identical.
func (a *api) applyPayload(w http.ResponseWriter, r *http.Request, opName string, args []byte) {
	wc, ok := queryWC(w, r)
	if !ok {
		return
	}
	body, ok := a.callWrite(w, r, opName, args, wc.WriteConsistencyFactor, wc.wait())
	if !ok {
		return
	}
	applied, err := ops.DecodePayloadResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	if !applied {
		writeError(w, http.StatusNotFound, "point not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"applied": applied})
}

// reservedKeyTTLField is the OPTIONAL per-key payload TTL control key in a dense
// set/overwrite-payload body. When present its value is a map[string]int64 of
// key -> RELATIVE ms; the remaining body keys are the payload itself. A body
// WITHOUT this key is the raw payload map exactly as before (no-key-ttl bodies
// unchanged). A payload key literally named key_ttl_ms is therefore reserved.
const reservedKeyTTLField = "key_ttl_ms"

// decodePayloadBodyWithTTL decodes a dense set/overwrite-payload body, peeling off
// the optional reservedKeyTTLField. The rest of the object is the payload. Returns
// ok=false (and writes the error) on malformed JSON / a bad key_ttl_ms value.
func decodePayloadBodyWithTTL(w http.ResponseWriter, r *http.Request) (meta vector.Metadata, keyTTLMs map[string]int64, ok bool) {
	var raw map[string]json.RawMessage
	if !decodeBody(w, r, &raw) {
		return nil, nil, false
	}
	if ttlRaw, has := raw[reservedKeyTTLField]; has {
		delete(raw, reservedKeyTTLField)
		km := make(map[string]int64)
		if err := json.Unmarshal(ttlRaw, &km); err != nil {
			writeError(w, http.StatusBadRequest, "invalid key_ttl_ms: "+err.Error())
			return nil, nil, false
		}
		if len(km) > 0 {
			keyTTLMs = km
		}
	}
	if len(raw) > 0 {
		m := make(vector.Metadata, len(raw))
		for k, v := range raw {
			var val vector.Value
			if err := json.Unmarshal(v, &val); err != nil {
				writeError(w, http.StatusBadRequest, "invalid payload value for "+k+": "+err.Error())
				return nil, nil, false
			}
			m[k] = val
		}
		meta = m
	}
	return meta, keyTTLMs, true
}

// setPayload merges the request-body payload into the point's existing payload.
// An optional key_ttl_ms object in the body sets per-key payload TTLs (relative
// ms); see decodePayloadBodyWithTTL.
func (a *api) setPayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	meta, keyTTLMs, ok := decodePayloadBodyWithTTL(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_set_payload", ops.EncodeSetPayloadArgsCAS(name, id, meta, keyTTLMs, exp, hasExp))
}

// overwritePayload replaces the point's entire payload with the request body. An
// optional key_ttl_ms object sets per-key TTLs on the new payload.
func (a *api) overwritePayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	meta, keyTTLMs, ok := decodePayloadBodyWithTTL(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_overwrite_payload", ops.EncodeSetPayloadArgsCAS(name, id, meta, keyTTLMs, exp, hasExp))
}

// deletePayloadKeys removes the listed keys from the point's payload.
func (a *api) deletePayloadKeys(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	var req deletePayloadKeysReq
	if !decodeBody(w, r, &req) {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_delete_payload_keys", ops.EncodeDeletePayloadKeysArgsCAS(name, id, req.Keys, exp, hasExp))
}

// clearPayload empties the point's payload.
func (a *api) clearPayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_clear_payload", ops.EncodeClearPayloadArgsCAS(name, id, exp, hasExp))
}

type searchReq struct {
	Query                  []float32     `json:"query"`
	K                      int           `json:"k"`
	Filter                 vector.Filter `json:"filter"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

// validConsistency rejects out-of-range consistency knobs at the edge: the
// enums only define 0 and 1, so any larger value is a 400 before dispatch.
// Returns false (after writing the error) when either value is out of range.
func validConsistency(w http.ResponseWriter, rc, opa uint8) bool {
	// read_consistency: 0=AnyReplica, 1=LeaderOnly, 2=Linearizable, 3=BoundedStaleness.
	// on_partition_unavailable: 0=Partial, 1=Fail.
	if rc > 3 {
		writeError(w, http.StatusBadRequest, "read_consistency must be 0 (any), 1 (leader), 2 (linearizable) or 3 (bounded-staleness)")
		return false
	}
	if opa > 1 {
		writeError(w, http.StatusBadRequest, "on_partition_unavailable must be 0 or 1")
		return false
	}
	return true
}

// maxCollectionNameLen caps a collection/alias name at the transport edge. The
// op wire codecs encode the collection-name length in a single byte
// (byte(len(collection))), so a name >=256 bytes would wrap modulo 256 and
// silently mis-decode or mis-route. canonicalName also prepends a "<tenant>/"
// prefix downstream, so the cap is set below 255 to leave headroom for the
// default "default/" (8-byte) prefix and keep the canonical name within the u8
// ceiling. Reject over-length names with a 400 before they reach a codec.
const maxCollectionNameLen = 247

// validName rejects an over-length collection/alias name at the edge so it can
// never silently truncate in the u8-prefixed wire codecs. Returns false (after
// writing the error) when the name exceeds maxCollectionNameLen.
func validName(w http.ResponseWriter, name string) bool {
	if len(name) > maxCollectionNameLen {
		writeError(w, http.StatusBadRequest, "collection name too long")
		return false
	}
	return true
}

// maxTopK is a defense-in-depth ceiling on a search/query result size requested
// at the transport edge. The engine already bounds its result allocations by the
// actual collection size, so this is not a memory-amplification fix; it simply
// rejects an obviously nonsensical k (<=0) or an absurdly large one before
// dispatch so the transport never forwards a degenerate request.
const maxTopK = 1 << 16

// validTopK rejects a non-positive or absurdly large k at the edge. Returns false
// (after writing the 400) when k is out of range.
func validTopK(w http.ResponseWriter, k int) bool {
	if k <= 0 || k > maxTopK {
		writeError(w, http.StatusBadRequest, "k must be between 1 and 65536")
		return false
	}
	return true
}

// validFilter validate-at-edge compiles a decoded filter: a bad RE2 regex or
// non-RFC3339 datetime literal (the rich-filter ops parse fine as JSON but fail
// to compile) is a 400 BEFORE dispatch. Returns false (after writing the error)
// on a compile failure — so a malformed filter never reaches the engine and, in
// particular, never triggers an over-broad delete_by_filter.
func validFilter(w http.ResponseWriter, f vector.Filter) bool {
	if _, err := vector.CompileFilter(f); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return false
	}
	return true
}

// missingJSON widens the missing-partition ids into ints so they marshal as
// plain JSON numbers. A nil/empty input yields an empty (non-null) array.
func missingJSON(missing []uint16) []int {
	out := make([]int, len(missing))
	for i, v := range missing {
		out[i] = int(v)
	}
	return out
}

func (a *api) search(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req searchReq
	if !decodeSearchBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	if !validTopK(w, req.K) {
		return
	}
	body, ok := a.call(w, r, "vector_search", ops.EncodeVectorSearchArgsOpts(name, req.K, req.Query, req.Filter, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
	if !ok {
		return
	}
	results, degraded, missing, err := ops.DecodeVectorSearchResultsDegraded(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "degraded": degraded, "missing": missingJSON(missing)})
}

func (a *api) searchDocs(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req searchReq
	if !decodeSearchBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	if !validTopK(w, req.K) {
		return
	}
	body, ok := a.call(w, r, "vector_search_docs", ops.EncodeVectorSearchArgsOpts(name, req.K, req.Query, req.Filter, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
	if !ok {
		return
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegradedRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "degraded": degraded, "missing": missingJSON(missing)})
}

type groupSearchReq struct {
	Query                  []float32     `json:"query"`
	K                      int           `json:"k"`
	GroupBy                string        `json:"group_by"`
	GroupSize              int           `json:"group_size"`
	FetchK                 int           `json:"fetch_k"`
	Filter                 vector.Filter `json:"filter"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

func (a *api) searchGroups(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req groupSearchReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	opts := vector.GroupOpts{GroupBy: req.GroupBy, GroupSize: req.GroupSize, FetchK: req.FetchK, Filter: req.Filter}
	if !validTopK(w, req.K) {
		return
	}
	body, ok := a.call(w, r, "vector_search_groups", ops.EncodeGroupSearchArgsOpts(name, req.K, req.Query, opts, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
	if !ok {
		return
	}
	groups, degraded, missing, err := ops.DecodeGroupsDegradedRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"groups": groups, "degraded": degraded, "missing": missingJSON(missing)})
}

type hybridReq struct {
	Dense   []float32            `json:"dense"`
	K       int                  `json:"k"`
	Sparse  *vector.SparseVector `json:"sparse"`
	Filter  vector.Filter        `json:"filter"`
	Method  string               `json:"method"` // "" / "rrf" | "weighted" | "dbsf"
	Alpha   float64              `json:"alpha"`
	RRFK    int                  `json:"rrf_k"`
	DenseK  int                  `json:"dense_k"`
	SparseK int                  `json:"sparse_k"`

	ReadConsistency        uint8  `json:"read_consistency"`
	OnPartitionUnavailable uint8  `json:"on_partition_unavailable"`
	MaxStaleness           uint64 `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

func (a *api) hybrid(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req hybridReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	var method vector.FusionMethod
	switch req.Method {
	case "", "rrf":
		method = vector.FusionRRF
	case "weighted":
		method = vector.FusionWeighted
	case "dbsf":
		method = vector.FusionDBSF
	default:
		writeError(w, http.StatusBadRequest, "unknown fusion method "+strconv.Quote(req.Method))
		return
	}
	var sparse vector.SparseVector
	if req.Sparse != nil {
		sparse = *req.Sparse
	}
	opts := vector.HybridOpts{
		Filter: req.Filter, Method: method, Alpha: req.Alpha,
		RRFK: req.RRFK, DenseK: req.DenseK, SparseK: req.SparseK,
	}
	if !validTopK(w, req.K) {
		return
	}
	body, ok := a.call(w, r, "vector_hybrid_search", ops.EncodeHybridSearchArgsOpts(name, req.Dense, req.K, sparse, opts, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
	if !ok {
		return
	}
	results, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "degraded": degraded, "missing": missingJSON(missing)})
}

// searchTextReq is the body for POST /points/search/text: a RAW query string the
// server tokenizes + BM25-scores (the client ships no tokens).
type searchTextReq struct {
	Text                   string        `json:"text"`
	K                      int           `json:"k"`
	Filter                 vector.Filter `json:"filter"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
	// GlobalIDF opts into the BM25 global-DF (dfs_query_then_fetch) two-phase text
	// search across partitions. Default false ⇒ the existing per-shard-local-IDF
	// fast path (byte-identical request when absent). Single-partition collections
	// ignore the flag (local corpus IS global).
	GlobalIDF bool `json:"global_idf"`
}

// searchText runs a BM25 full-text search (vector_search_text), returning the
// top-k Documents (content + metadata). Mirrors searchDocs but carries raw text.
func (a *api) searchText(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req searchTextReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	if !validTopK(w, req.K) {
		return
	}
	body, ok := a.call(w, r, "vector_search_text", ops.EncodeSearchTextArgsGlobal(name, req.Text, req.K, req.Filter, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness, req.GlobalIDF, nil))
	if !ok {
		return
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegradedRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "degraded": degraded, "missing": missingJSON(missing)})
}

// hybridTextReq is the body for POST /points/search/hybrid-text: a dense query
// vector plus RAW query text (the BM25 lane), fused server-side. Unlike hybridReq
// it carries no sparse query — the server analyzes the text into the text lane.
type hybridTextReq struct {
	Dense   []float32     `json:"vector"`
	Text    string        `json:"text"`
	K       int           `json:"k"`
	Filter  vector.Filter `json:"filter"`
	Method  string        `json:"method"` // "" / "rrf" | "weighted" | "dbsf"
	Alpha   float64       `json:"alpha"`
	RRFK    int           `json:"rrf_k"`
	DenseK  int           `json:"dense_k"`
	SparseK int           `json:"sparse_k"`

	ReadConsistency        uint8  `json:"read_consistency"`
	OnPartitionUnavailable uint8  `json:"on_partition_unavailable"`
	MaxStaleness           uint64 `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
	// GlobalIDF opts into the BM25 global-DF (dfs_query_then_fetch) two-phase text
	// lane across partitions. Default false ⇒ the existing per-shard-local-IDF fast
	// path (byte-identical request when absent). Affects only the BM25 text lane.
	GlobalIDF bool `json:"global_idf"`
}

// hybridText fuses a dense KNN lane with a BM25 full-text lane (vector_hybrid_text).
// Mirrors the hybrid handler; the text lane is raw query text analyzed server-side.
func (a *api) hybridText(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req hybridTextReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	var method vector.FusionMethod
	switch req.Method {
	case "", "rrf":
		method = vector.FusionRRF
	case "weighted":
		method = vector.FusionWeighted
	case "dbsf":
		method = vector.FusionDBSF
	default:
		writeError(w, http.StatusBadRequest, "unknown fusion method "+strconv.Quote(req.Method))
		return
	}
	opts := vector.HybridOpts{
		Filter: req.Filter, Method: method, Alpha: req.Alpha,
		RRFK: req.RRFK, DenseK: req.DenseK, SparseK: req.SparseK,
	}
	if !validTopK(w, req.K) {
		return
	}
	body, ok := a.call(w, r, "vector_hybrid_text", ops.EncodeHybridTextArgsGlobal(name, req.Dense, req.Text, req.K, opts, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness, req.GlobalIDF, nil))
	if !ok {
		return
	}
	results, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "degraded": degraded, "missing": missingJSON(missing)})
}

// queryLeafReq is one JSON query node in a /query body: exactly one of dense or
// sparse should be populated. k is the leaf's own top-k; lane_k is the per-lane
// candidate pool (0 = engine default). filter is the optional per-leaf metadata
// predicate (same JSON contract as the hybrid filter).
type queryLeafReq struct {
	Dense     []float32            `json:"dense"`
	Sparse    *vector.SparseVector `json:"sparse"`
	Recommend *recommendLeafReq    `json:"recommend"`
	Discover  *discoverLeafReq     `json:"discover"`
	K         int                  `json:"k"`
	LaneK     int                  `json:"lane_k"`
	Filter    vector.Filter        `json:"filter"`
}

// discoverLeafReq is the DISCOVER query node in a /query body (v1 dense family): a
// context of positive/negative example PAIRS that steer the ranking, plus an
// optional target anchor that seeds the candidate pool. Every example may be given
// either as a stored POINT-ID (the coordinator resolves it cluster-wide → vector
// and embeds it) or as a raw VECTOR (already embedded). The target is the optional
// anchor: TargetID (a stored point) OR Target (a raw vector); when neither is set
// the pool is seeded from the mean of the context positives. The leaf's k/filter
// ride the enclosing queryLeafReq (same as a dense leaf).
type discoverLeafReq struct {
	// A POINTER because "no anchor" and "the anchor is point id 0" are different
	// requests: a plain uint64 makes an omitted "target" and "target":0 decode to
	// the same value, which silently dropped the anchor and answered from the
	// context positives instead. nil = absent; non-nil = anchor, id 0 included.
	// The engine-side representation (QueryLeaf.DiscoverTargetID, a slice) was
	// already id-0-safe; only this DTO collapsed the two cases.
	TargetID *uint64           `json:"target"`     // optional anchor point-id
	Target   []float32         `json:"target_vec"` // optional anchor raw vector
	Context  []discoverPairReq `json:"context"`    // at least one pair required
}

// discoverPairReq is one DISCOVER context constraint: a positive/negative example
// each given as a stored point-id (Positive/Negative) OR a raw vector (PositiveVec/
// NegativeVec). A pair mixes neither form across its two sides in practice; the
// coordinator resolves the id form cluster-wide and the vector form is embedded as-is.
// Positive/Negative are POINTERS for the same reason TargetID is: a pair is
// dispatched to the id form or the vector form by inspection, and with plain
// uint64s an ABSENT id is indistinguishable from point id 0. A half-specified
// pair such as {"positive_vec":[...],"negative":5} failed the vector-form test
// and fell through to the id form, silently synthesizing id 0 as the positive
// example. While id 0 was unsearchable that produced an empty lane; now that id
// 0 is a first-class point it would anchor discover on a REAL user point. Such a
// pair is rejected outright by validate() instead.
type discoverPairReq struct {
	Positive    *uint64   `json:"positive"`
	Negative    *uint64   `json:"negative"`
	PositiveVec []float32 `json:"positive_vec"`
	NegativeVec []float32 `json:"negative_vec"`
}

// isVecForm / isIDForm are the two COMPLETE shapes a context pair may take. A
// pair that satisfies neither is malformed (half-specified, mixed, or empty) and
// must not be guessed at.
func (p discoverPairReq) isVecForm() bool {
	return len(p.PositiveVec) > 0 && len(p.NegativeVec) > 0 && p.Positive == nil && p.Negative == nil
}

func (p discoverPairReq) isIDForm() bool {
	return p.Positive != nil && p.Negative != nil && len(p.PositiveVec) == 0 && len(p.NegativeVec) == 0
}

// validate rejects a discover leaf whose context pairs are not each wholly in
// the id form or wholly in the vector form. Returns an error suitable for a 400.
func (d *discoverLeafReq) validate() error {
	for i, p := range d.Context {
		if !p.isVecForm() && !p.isIDForm() {
			return errors.New("discover context pair " + strconv.Itoa(i) +
				": give either both point-ids (positive, negative) or both vectors (positive_vec, negative_vec), not a mix")
		}
	}
	return nil
}

// validDiscover validate-at-edge checks a discover leaf's context pairs, writing
// a 400 and returning false on a malformed pair — so an ambiguous pair never
// reaches the engine as a silently-invented point id. Mirrors validFilter.
// A nil leaf (no discover payload) is trivially valid.
func validDiscover(w http.ResponseWriter, d *discoverLeafReq) bool {
	if d == nil {
		return true
	}
	if err := d.validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return false
	}
	return true
}

// recommendLeafReq is the RECOMMEND query node in a /query body: positive/negative
// EXAMPLE point-ids (v1 AVERAGE_VECTOR, dense family). The coordinator resolves the
// ids to stored vectors, derives mean(positive) - mean(negative), and rewrites this
// leaf into a dense leaf before the search. The leaf's k/lane_k/filter ride the
// enclosing queryLeafReq (same as a dense leaf).
type recommendLeafReq struct {
	Positive []uint64 `json:"positive"`
	Negative []uint64 `json:"negative"`
	// Strategy selects the recommend scorer: "average"/"average_vector"/"" (default) =
	// AVERAGE_VECTOR (derive mean(pos)-mean(neg) → dense); "best_score"/"best" =
	// BEST_SCORE (a custom per-candidate max-similarity scorer, score-descending).
	Strategy string `json:"strategy"`
}

// parseRecommendStrategy maps the JSON recommend strategy string to the engine
// enum: "best_score"/"best" → RecommendBestScore; anything else (including "",
// "average", "average_vector") → RecommendAverageVector (the default — byte-identical
// to the pre-strategy recommend). An int "1"/"0" string is also accepted.
func parseRecommendStrategy(s string) vector.RecommendStrategy {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "best_score", "best", "1":
		return vector.RecommendBestScore
	default:
		return vector.RecommendAverageVector
	}
}

// toLeaf converts the JSON leaf into an engine vector.QueryLeaf. A leaf with a
// recommend payload is LeafRecommend (the coordinator derives it to dense); a leaf
// with a sparse payload is LeafSparse; otherwise it is LeafDense (an empty dense
// leaf is the "no root" marker for FUSION). The per-leaf filter is validated by the
// caller via validFilter before encoding.
func (l queryLeafReq) toLeaf() vector.QueryLeaf {
	if l.Recommend != nil {
		// BEST_SCORE is a custom per-candidate scorer (score-descending, like Discover);
		// AVERAGE_VECTOR (the default) is a dense rewrite (distance-ascending).
		strategy := parseRecommendStrategy(l.Recommend.Strategy)
		return vector.QueryLeaf{
			Kind:      vector.LeafRecommend,
			Positive:  l.Recommend.Positive,
			Negative:  l.Recommend.Negative,
			Strategy:  strategy,
			ScoreDesc: strategy == vector.RecommendBestScore,
			K:         l.K, LaneK: l.LaneK, Filter: l.Filter,
		}
	}
	if l.Discover != nil {
		leaf := vector.QueryLeaf{
			Kind:      vector.LeafDiscover,
			ScoreDesc: true, // discover ranks score-descending (like MV MaxSim)
			K:         l.K, LaneK: l.LaneK, Filter: l.Filter,
		}
		// Target anchor: a point-id is resolved by the coordinator; a raw vector is
		// embedded as-is. When neither is set the pool seeds from the context positives.
		if l.Discover.TargetID != nil {
			leaf.DiscoverTargetID = []uint64{*l.Discover.TargetID}
		} else if len(l.Discover.Target) > 0 {
			leaf.DiscoverTarget = l.Discover.Target
		}
		// Each context pair is either the id-form (resolved by the coordinator) or the
		// already-embedded vector-form. A pair with both sides given as vectors goes to
		// DiscoverContext; otherwise the pair's ids go to DiscoverContextIDs.
		// Each pair is wholly the vector form or wholly the id form — validDiscover
		// rejected every other shape at the edge, so there is no half-specified
		// pair left to guess at here.
		for _, p := range l.Discover.Context {
			if p.isVecForm() {
				leaf.DiscoverContext = append(leaf.DiscoverContext, vector.DiscoverPair{Pos: p.PositiveVec, Neg: p.NegativeVec})
			} else {
				leaf.DiscoverContextIDs = append(leaf.DiscoverContextIDs, vector.ContextPair{Positive: *p.Positive, Negative: *p.Negative})
			}
		}
		return leaf
	}
	if l.Sparse != nil && !l.Sparse.IsZero() {
		return vector.QueryLeaf{
			Kind:   vector.LeafSparse,
			Sparse: *l.Sparse,
			K:      l.K, LaneK: l.LaneK, Filter: l.Filter,
		}
	}
	return vector.QueryLeaf{
		Kind:  vector.LeafDense,
		Dense: l.Dense,
		K:     l.K, LaneK: l.LaneK, Filter: l.Filter,
	}
}

// queryReq is the unified Query API HTTP body: a root leaf (RERANK) + N prefetch
// leaves, a combine mode ("fusion"|"rerank"), the fusion config, and the final
// top-k, plus the standard read-consistency knobs. Mirrors hybridReq's
// consistency fields.
type queryReq struct {
	Root     *queryLeafReq  `json:"root"`
	Prefetch []queryLeafReq `json:"prefetch"`
	Mode     string         `json:"mode"`   // "fusion" (default) | "rerank"
	Method   string         `json:"method"` // "" / "rrf" | "weighted" | "dbsf"
	Alpha    float64        `json:"alpha"`
	RRFK     int            `json:"rrf_k"`
	K        int            `json:"k"`

	// GroupBy / GroupSize make this a GROUPED query (Qdrant group_by): when group_by
	// is non-empty the response is {groups:[{key,hits:[]}]} (mirroring the standalone
	// /groups shape) instead of {results}. k is reinterpreted as the number of GROUPS.
	GroupBy   string `json:"group_by"`
	GroupSize int    `json:"group_size"`

	ReadConsistency        uint8  `json:"read_consistency"`
	OnPartitionUnavailable uint8  `json:"on_partition_unavailable"`
	MaxStaleness           uint64 `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

func (a *api) query(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req queryReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	var mode vector.QueryMode
	switch req.Mode {
	case "", "fusion":
		mode = vector.ModeFusion
	case "rerank":
		mode = vector.ModeRerank
	default:
		writeError(w, http.StatusBadRequest, "unknown query mode "+strconv.Quote(req.Mode))
		return
	}
	var method vector.FusionMethod
	switch req.Method {
	case "", "rrf":
		method = vector.FusionRRF
	case "weighted":
		method = vector.FusionWeighted
	case "dbsf":
		method = vector.FusionDBSF
	default:
		writeError(w, http.StatusBadRequest, "unknown fusion method "+strconv.Quote(req.Method))
		return
	}
	spec := vector.QuerySpec{
		Mode: mode, Method: method, Alpha: req.Alpha, RRFK: req.RRFK, K: req.K,
		GroupBy: req.GroupBy, GroupSize: req.GroupSize,
	}
	if req.Root != nil {
		if !validFilter(w, req.Root.Filter) || !validDiscover(w, req.Root.Discover) {
			return
		}
		spec.Root = req.Root.toLeaf()
	}
	for i := range req.Prefetch {
		if !validFilter(w, req.Prefetch[i].Filter) || !validDiscover(w, req.Prefetch[i].Discover) {
			return
		}
		spec.Prefetch = append(spec.Prefetch, vector.LeafSource(req.Prefetch[i].toLeaf()))
	}
	specBytes, err := ops.MarshalEngineQuerySpec(spec)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	body, ok := a.call(w, r, "vector_query", ops.EncodeQueryArgs(name, specBytes, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
	if !ok {
		return
	}
	// GROUPED query: the dispatcher returns the grouped result; surface the standalone
	// /groups response shape {groups:[{key,hits:[]}]}. The flat {results} path is
	// unchanged when group_by is empty.
	if req.GroupBy != "" {
		groups, degraded, missing, gerr := ops.DecodeGroupsDegradedRaw(body)
		if gerr != nil {
			writeError(w, http.StatusInternalServerError, gerr.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"groups": groups, "degraded": degraded, "missing": missingJSON(missing)})
		return
	}
	results, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "degraded": degraded, "missing": missingJSON(missing)})
}

type scrollReq struct {
	Filter                 vector.Filter `json:"filter"`
	Limit                  int           `json:"limit"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
	Cursor                 string        `json:"cursor,omitempty"`
	// OrderBy paginates by an arbitrary numeric/datetime payload field (Qdrant
	// order_by). Absent ⇒ id-ascending scroll. See orderByReq.
	OrderBy *orderByReq `json:"order_by,omitempty"`
}

// orderByReq is the JSON order_by object on a scroll request. start_from is a
// json.RawMessage so it accepts BOTH a numeric value (numeric field) AND an RFC3339
// STRING (datetime field, lowered to int-ms server-side). is_datetime marks the field
// as a datetime stored as unix-ms.
type orderByReq struct {
	Key        string          `json:"key"`
	Desc       bool            `json:"desc"`
	StartFrom  json.RawMessage `json:"start_from,omitempty"`
	IsDatetime bool            `json:"is_datetime"`
	// IsString selects the lexicographic (stringValue, id) order_by path. It is
	// mutually-exclusive with is_datetime and incompatible with start_from (both ⇒ a
	// 400 via vector.ErrBadOrderKind). Additive: an absent is_string ⇒ the unchanged
	// numeric/datetime path.
	IsString bool `json:"is_string"`
	// TailKeys carry the SECONDARY/tertiary sort keys of a MULTI-KEY order_by: this object
	// is the PRIMARY (key[0]); tail_keys[0] is the 2nd key, etc. EMPTY/absent ⇒ the
	// single-key path (BYTE/BEHAVIOUR-IDENTICAL: v2/v3 cursor, same results). A non-empty
	// tail_keys switches onto the composite (k1,…,kN,id) order + the v4 cursor. Each tail
	// key carries its own key/desc/is_datetime/is_string; start_from is primary-only and a
	// tail key's own tail_keys are ignored (the list is flat, one level deep).
	TailKeys []orderByReq `json:"tail_keys,omitempty"`
}

// toOrderBy validates the order_by JSON object into a *vector.OrderBy. nil receiver ⇒
// (nil, nil) (no order_by). An empty key, a malformed start_from JSON, or a bad
// RFC3339 datetime string ⇒ error (the caller maps to 400).
func (o *orderByReq) toOrderBy() (*vector.OrderBy, error) {
	if o == nil {
		return nil, nil
	}
	var startNum *float64
	var startDt *string
	if len(o.StartFrom) > 0 && string(o.StartFrom) != "null" {
		// Try numeric first; fall back to an RFC3339 string.
		var n float64
		if err := json.Unmarshal(o.StartFrom, &n); err == nil {
			startNum = &n
		} else {
			var s string
			if err := json.Unmarshal(o.StartFrom, &s); err != nil {
				return nil, vector.ErrBadOrderStart
			}
			startDt = &s
		}
	}
	ob, err := vector.ParseOrderBy(o.Key, o.Desc, o.IsDatetime, o.IsString, startNum, startDt)
	if err != nil {
		return nil, err
	}
	// MULTI-KEY: build the secondary/tertiary keys from tail_keys. Each tail key is a plain
	// order spec (key/desc/is_datetime/is_string); start_from is primary-only and ignored
	// on a tail key, and a tail key's own tail_keys are ignored (the list is flat).
	if len(o.TailKeys) > 0 {
		ob.Tail = make([]vector.OrderBy, 0, len(o.TailKeys))
		for i := range o.TailKeys {
			tk := o.TailKeys[i]
			tob, terr := vector.ParseOrderBy(tk.Key, tk.Desc, tk.IsDatetime, tk.IsString, nil, nil)
			if terr != nil {
				return nil, terr
			}
			ob.Tail = append(ob.Tail, vector.OrderBy{Key: tob.Key, Desc: tob.Desc, IsDatetime: tob.IsDatetime, Kind: tob.Kind})
		}
	}
	return ob, nil
}

func (a *api) scroll(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req scrollReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	// Validate-at-edge: a bad order_by (empty key / malformed start_from) is a 400.
	order, err := req.OrderBy.toOrderBy()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	// A malformed cursor — or a cursor whose version disagrees with order_by — is a
	// client error: reject with 400 BEFORE dispatch.
	afterID, hasAfter, scrollOrder, err := scrollCursorAndOrder(req.Cursor, order)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	body, ok := a.call(w, r, "vector_scroll", ops.EncodeScrollArgsOrderBounded(name, req.Filter, req.Limit, req.ReadConsistency, req.OnPartitionUnavailable, afterID, hasAfter, scrollOrder, req.MaxStaleness))
	if !ok {
		return
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResultRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "degraded": degraded, "missing": missingJSON(missing), "next_cursor": nextCursor})
}

// scrollCursorAndOrder decodes the scroll cursor TYPED and reconciles it with an
// optional order_by, producing the (afterID, hasAfter, *ops.ScrollOrder) the args
// codec needs — the HTTP mirror of the gRPC server's same-named helper. It enforces
// the cursor⇄order_by agreement at the edge (a v1 cursor with order_by, a v2 cursor
// without, or a direction/key change mid-pagination ⇒ error).
func scrollCursorAndOrder(cursor string, order *vector.OrderBy) (afterID uint64, hasAfter bool, scrollOrder *ops.ScrollOrder, err error) {
	dec, derr := ops.DecodeScrollCursorTyped(cursor)
	if derr != nil {
		return 0, false, nil, derr
	}
	if order == nil {
		if dec.Present && dec.Version != 1 {
			return 0, false, nil, ops.ErrCursorOrderMismatch
		}
		return dec.LastID, dec.Present, nil, nil
	}
	if len(order.Tail) > 0 {
		return scrollCursorAndOrderMulti(dec, order)
	}
	keyHash := vector.OrderKeyHash(order.Key)
	if order.Kind == vector.OrderString {
		if verr := ops.ValidateOrderCursorString(dec, order.Desc, keyHash); verr != nil {
			return 0, false, nil, verr
		}
		so := &ops.ScrollOrder{Key: order.Key, Desc: order.Desc, Kind: vector.OrderString}
		if dec.Present {
			afterID, hasAfter = dec.LastID, true
			so.ResumeStr, so.HasResumeStr = dec.StrValue, true
		}
		return afterID, hasAfter, so, nil
	}
	if verr := ops.ValidateOrderCursor(dec, order.Desc, keyHash); verr != nil {
		return 0, false, nil, verr
	}
	so := &ops.ScrollOrder{Key: order.Key, Desc: order.Desc, IsDatetime: order.IsDatetime, Kind: order.Kind, StartFrom: order.StartFrom, HasStart: order.HasStart}
	if dec.Present {
		afterID, hasAfter = dec.LastID, true
		so.ResumeKey, so.HasResume = dec.Value, true
	}
	return afterID, hasAfter, so, nil
}

// scrollCursorAndOrderMulti is the MULTI-KEY branch of scrollCursorAndOrder (the HTTP
// mirror of the gRPC server's same-named helper): it validates a v4 (k1,…,kN, id) tuple
// cursor against the request's primary direction + key-list hash + arity (a v1/v2/v3
// cursor, or a wrong-arity v4, is a loud mismatch) and threads the resume TUPLE onto
// ScrollOrder.ResumeKeys + the args afterID.
func scrollCursorAndOrderMulti(dec ops.DecodedScrollCursor, order *vector.OrderBy) (afterID uint64, hasAfter bool, scrollOrder *ops.ScrollOrder, err error) {
	keys := vector.OrderKeyList(order)
	keyHash := vector.OrderKeyListHash(keys)
	if verr := ops.ValidateOrderCursorTuple(dec, order.Desc, keyHash, len(keys)); verr != nil {
		return 0, false, nil, verr
	}
	so := &ops.ScrollOrder{Key: order.Key, Desc: order.Desc, IsDatetime: order.IsDatetime, Kind: order.Kind, Tail: ops.OrderByToScrollOrderTail(order)}
	if dec.Present {
		afterID, hasAfter = dec.LastID, true
		so.ResumeKeys = make([]ops.ScrollOrderVal, len(dec.Tuple))
		for i, kv := range dec.Tuple {
			so.ResumeKeys[i] = ops.ScrollOrderVal{Num: kv.Num, Str: kv.Str, Kind: vector.OrderKind(kv.Kind)}
		}
		so.HasResumeKeys = true
	}
	return afterID, hasAfter, so, nil
}

type deleteByFilterReq struct {
	Filter vector.Filter `json:"filter"`
	writeConsistency
}

func (a *api) deleteByFilter(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req deleteByFilterReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !req.validate(w) {
		return
	}
	// Validate-at-edge: a malformed filter must NEVER reach the engine — a bad
	// filter here would otherwise risk an over-broad delete. Compile first, 400
	// on error, and only then dispatch.
	if !validFilter(w, req.Filter) {
		return
	}
	// delete_by_filter is a fan-out write: the __wc__ handler barriers EACH touched
	// partition shard (first ErrWriteConsistency wins). The default path (no WCF) is
	// the plain op, unchanged.
	body, ok := a.callWrite(w, r, "vector_delete_by_filter", ops.EncodeDeleteByFilterArgs(name, req.Filter), req.WriteConsistencyFactor, req.wait())
	if !ok {
		return
	}
	n, err := ops.DecodeDeleteByFilterResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"deleted": n})
}

// resplitReq is the body for a resplit request: the target partition count for
// the collection's data. Shared by the dense and MV resplit handlers.
type resplitReq struct {
	NewPartitions int `json:"new_partitions"`
}

// resplit re-partitions a dense collection into new_partitions shards. Resplit
// is SYNCHRONOUS and OFFLINE: it rebuilds partition state in place, so writes
// must be quiesced for its duration and the caller/proxy must allow a long
// request timeout. On completion the orphaned old partitions remain until a
// resplit/cleanup pass drops them.
func (a *api) resplit(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req resplitReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.NewPartitions < 0 {
		writeError(w, http.StatusBadRequest, "new_partitions must be non-negative")
		return
	}
	if _, ok := a.call(w, r, "vector_resplit", ops.EncodeResplitArgs(name, req.NewPartitions)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "new_partitions": req.NewPartitions})
}

// resplitCleanup drops the orphaned old partitions left behind by a prior dense
// resplit, returning how many were dropped. Like resplit it is synchronous and
// offline (quiesce writes; allow a long request timeout).
func (a *api) resplitCleanup(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	body, ok := a.call(w, r, "vector_resplit_cleanup", ops.EncodeResplitCleanupArgs(name))
	if !ok {
		return
	}
	dropped, err := ops.DecodeResplitCleanupResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "dropped": dropped})
}

// reshard re-partitions a dense collection ONLINE into new_partitions shards.
// Unlike resplit, reads AND writes stay live for the duration: the orchestrator
// dual-writes to old+new gen during a streamed if-absent copy, then flips the
// catalog at cutover. The request is still SYNCHRONOUS — it blocks until cutover
// — so the caller/proxy must allow a long request timeout. Reuses resplitReq.
func (a *api) reshard(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req resplitReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.NewPartitions < 0 {
		writeError(w, http.StatusBadRequest, "new_partitions must be non-negative")
		return
	}
	if _, ok := a.call(w, r, "vector_reshard", ops.EncodeReshardArgs(name, req.NewPartitions)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "new_partitions": req.NewPartitions})
}

// reshardAbort aborts an in-flight dense reshard, restoring the old generation
// and dropping the new-gen partitions. Pre-cutover only — it errors (surfaced as
// a non-2xx) if the reshard has already flipped.
func (a *api) reshardAbort(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if _, ok := a.call(w, r, "vector_reshard_abort", ops.EncodeReshardAbortArgs(name)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name})
}
