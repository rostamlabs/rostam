// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// ErrVectorsNotAvailable is returned by vector op handlers when the
// dispatcher was constructed without a CollectionStore.
var ErrVectorsNotAvailable = errors.New("ops: vector store not available")

// vectorSearchScratch pools the per-search server-side buffers: the decoded
// query and the result slice. Reused across requests so the hot kNN path
// allocates only the (small) collection string and the response buffer.
// vector_search is read-only and runs concurrently, so the pool is keyed per
// goroutine via sync.Pool rather than hung off the (shared) TxContext.
type vectorSearchScratch struct {
	query   []float32
	results []vector.Result
}

var vectorSearchPool = sync.Pool{New: func() any { return &vectorSearchScratch{} }}

// vectorDenseBufPool recycles []float32 scratch backings for the dense vector on
// the single-point insert and get handlers, so a hot vector_insert/vector_get no
// longer allocates a throwaway []float32 per op. It pools a *[]float32 (not the
// slice itself) so Put never boxes/allocates. The buffer is only ever handed to a
// path that COPIES the floats out before the op returns (arena.Insert copies on
// insert; wire.EncodeVectorGetResultV serializes on get), so the recycled backing is
// never retained past the handler. The BULK staging path does NOT use this: its
// decoded vecs are retained in the collection's stage buffer until build.
var vectorDenseBufPool = sync.Pool{New: func() any { s := make([]float32, 0); return &s }}

// builtinHandlers maps every built-in op name to its server-side Handler.
// RegisterBuiltins walks wire.BuiltinOps (the canonical routing table also used
// by wire.RegisterRoutableBuiltins, the client's routing-only counterpart) and
// looks up each op's handler here, so the two registries can never drift apart
// on an op's name, kind, or routing key — only on whether a handler is bound.
var builtinHandlers = map[string]Handler{
	"get":    handleGet,
	"put":    handlePut,
	"del":    handleDel,
	"expire": handleExpire,
	"incr":   handleIncr,
	// Conditional-write KV ops (atomic under the shard write lock held for the
	// whole handler): set_nx = set-if-absent, cas = compare-and-swap, cad =
	// compare-and-delete.
	"set_nx": handleSetNX,
	"cas":    handleCAS,
	"cad":    handleCompareAndDel,
	// put_batch packs N puts into one Raft log entry (one fsync / round-trip /
	// apply for the whole batch). It routes by its FIRST key, so every key in a
	// batch must hash to the same shard — the cluster fan-out (Node.PutBatch)
	// guarantees that by grouping before it calls.
	"put_batch": handlePutBatch,
	// __ping__ is shardless.
	"__ping__": handlePing,
	// __ready__ is a shardless READINESS probe. The default handler here always
	// reports ready (correct for single-node / Direct, which is always its own
	// leader). In cluster mode cluster.Node intercepts __ready__ in its adminOps
	// BEFORE this registry lookup and runs a real per-hosted-shard leader check
	// (see cluster/admin_ops.go handleReady). Distinct from __ping__ (liveness):
	// readiness reflects whether this node can actually serve its shards.
	wire.ReadyOp: handleReady,
	// __metrics__ is shardless: it renders the local node's per-collection
	// Prometheus stats. follow-up: a clustered scrape would gather + concatenate
	// each shard's exposition; today it serves the node it is dispatched to.
	wire.MetricsOp: handleMetrics,
	// __repl_metrics__ is a shardless REPLICATION-observability op. The default
	// handler here reports no replicated shards (correct for single-node / Direct,
	// which replicates nothing). In cluster mode cluster.Node intercepts it in its
	// adminOps BEFORE this registry lookup and renders the real per-hosted-shard
	// ISR / lag view (see cluster/repl_metrics.go handleReplMetrics), mirroring how
	// __ready__ is overridden. Read-only, no args; result is a JSON body served
	// as-is by the HTTP /v1/replication handler.
	wire.ReplMetricsOp: handleReplMetrics,

	// Vector ops. Routing (kind/wire.KeyExtractor/wire.RouteLayout) lives in
	// wire.BuiltinOps; only the handler binding happens here.
	"vector_create_collection":         handleVectorCreateCollection,
	"vector_drop_collection":           handleVectorDropCollection,
	"vector_insert":                    handleVectorInsert,
	"vector_insert_if_absent":          handleVectorInsertIfAbsent,
	"vector_exists":                    handleVectorExists,
	"vector_delete":                    handleVectorDelete,
	"vector_get":                       handleVectorGet,
	"vector_get_batch":                 handleVectorGetBatch,
	"vector_set_payload":               handleVectorSetPayload,
	"vector_overwrite_payload":         handleVectorOverwritePayload,
	"vector_delete_payload_keys":       handleVectorDeletePayloadKeys,
	"vector_clear_payload":             handleVectorClearPayload,
	"vector_search":                    handleVectorSearch,
	"vector_hybrid_search":             handleVectorHybridSearch,
	"vector_hybrid_lanes":              handleVectorHybridLanes,
	"vector_search_text":               handleVectorSearchText,
	"vector_hybrid_text":               handleVectorHybridText,
	"vector_hybrid_text_lanes":         handleVectorHybridTextLanes,
	"vector_bm25_stats":                handleVectorBM25Stats,
	"vector_query":                     handleVectorQuery,
	"vector_upsert":                    handleVectorUpsert,
	"vector_bulk_stage":                handleVectorBulkStage,
	"vector_bulk_stage_payload":        handleVectorBulkStagePayload,
	"vector_bulk_build":                handleVectorBulkBuild,
	"vector_search_docs":               handleVectorSearchDocs,
	"vector_delete_by_filter":          handleVectorDeleteByFilter,
	"vector_search_groups":             handleVectorSearchGroups,
	"vector_group_candidates":          handleVectorGroupCandidates,
	"vector_scroll":                    handleVectorScroll,
	"vector_scan_vectors":              handleVectorScanVectors,
	"vector_get_config":                handleVectorGetConfig,
	"vector_mv_create_collection":      handleMVCreate,
	"vector_mv_drop_collection":        handleMVDrop,
	"vector_mv_add":                    handleMVAdd,
	"vector_mv_add_if_absent":          handleMVAddIfAbsent,
	"vector_mv_add_versioned":          handleMVAddVersioned,
	"vector_mv_add_batch":              handleMVAddBatch,
	"vector_mv_exists":                 handleMVExists,
	"vector_mv_search":                 handleMVSearch,
	"vector_mv_hybrid_search":          handleMVHybridSearch,
	"vector_mv_hybrid_lanes":           handleMVHybridLanes,
	"vector_mv_delete":                 handleMVDelete,
	"vector_mv_get":                    handleMVGet,
	"vector_mv_get_batch":              handleMVGetBatch,
	"vector_mv_set_payload":            handleMVSetPayload,
	"vector_mv_overwrite_payload":      handleMVOverwritePayload,
	"vector_mv_delete_payload_keys":    handleMVDeletePayloadKeys,
	"vector_mv_clear_payload":          handleMVClearPayload,
	"vector_mv_get_config":             handleMVGetConfig,
	"vector_mv_scan_vectors":           handleMVScanVectors,
	"vector_mv_scroll":                 handleMVScroll,
	"vector_mv_query":                  handleMVQuery,
	"vector_named_create_collection":   handleNamedCreate,
	"vector_named_drop_collection":     handleNamedDrop,
	"vector_named_insert":              handleNamedInsert,
	"vector_named_delete":              handleNamedDelete,
	"vector_named_get":                 handleNamedGet,
	"vector_named_get_batch":           handleNamedGetBatch,
	"vector_named_set_payload":         handleNamedSetPayload,
	"vector_named_overwrite_payload":   handleNamedOverwritePayload,
	"vector_named_delete_payload_keys": handleNamedDeletePayloadKeys,
	"vector_named_clear_payload":       handleNamedClearPayload,
	"vector_named_search":              handleNamedSearch,
	"vector_named_sparse_search":       handleNamedSparseSearch,
	"vector_named_hybrid_search":       handleNamedHybridSearch,
	"vector_named_hybrid_lanes":        handleNamedHybridLanes,
	"vector_named_search_docs":         handleNamedSearchDocs,
	"vector_named_scroll":              handleNamedScroll,
	"vector_named_get_config":          handleNamedGetConfig,
	"vector_named_query":               handleNamedQuery,
}

// RegisterBuiltins adds the standard set of ops to the registry:
//   - "get"    (read-only)   args: [keyLen u16][key]                    → value bytes (or ErrNotFound)
//   - "put"    (read-write)  args: [keyLen u16][key][valLen u32][val][ttlMs u64]
//   - "del"    (read-write)  args: [keyLen u16][key]                    → 1-byte 0/1
//   - "expire" (read-write)  args: [keyLen u16][key][ttlMs u64]
//   - "incr"   (read-write)  args: [keyLen u16][key][delta i64]         → new value as i64 BE
//   - "set_nx" (read-write)  args: [keyLen u16][key][valLen u32][val][ttlMs u64] → 1-byte 1=stored/0=present
//   - "cas"    (read-write)  args: [keyLen u16][key][valLen u32][val][hasExpected u8][expLen u32][expected][ttlMs u64] → 1-byte 1=stored/0=mismatch
//   - "cad"    (read-write)  args: [keyLen u16][key][expLen u32][expected] → 1-byte 1=deleted/0=mismatch|absent
//   - "__ping__" (read-only) args: (ignored)                             → empty
//
// Routing metadata (kind/wire.KeyExtractor/wire.RouteLayout) comes from wire.BuiltinOps,
// the SAME table wire.RegisterRoutableBuiltins walks for the client's
// routing-only registry, so the two can never disagree about how an op is
// named or routed — only the client's registry carries no handler.
func RegisterBuiltins(r *Registry) error {
	for _, o := range wire.BuiltinOps {
		fn, ok := builtinHandlers[o.Name]
		if !ok {
			return fmt.Errorf("ops: no handler registered for builtin op %q", o.Name)
		}
		if err := r.registerEntry(o.Name, o.Kind, fn, o.KE, o.Layout, o.CrossShard); err != nil {
			return err
		}
	}
	if len(builtinHandlers) != len(wire.BuiltinOps) {
		return fmt.Errorf("ops: builtinHandlers has %d entries but wire.BuiltinOps has %d; they must name the same set of ops", len(builtinHandlers), len(wire.BuiltinOps))
	}
	return nil
}

func handleGet(tx *TxContext, args []byte) ([]byte, error) {
	key, err := wire.DecodeKeyArgs(args)
	if err != nil {
		return nil, err
	}
	return tx.Get(key)
}

func handlePut(tx *TxContext, args []byte) ([]byte, error) {
	key, val, ttl, err := wire.DecodePutArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.Put(key, val, ttl)
}

func handleDel(tx *TxContext, args []byte) ([]byte, error) {
	key, err := wire.DecodeKeyArgs(args)
	if err != nil {
		return nil, err
	}
	existed, err := tx.Del(key)
	if err != nil {
		return nil, err
	}
	if existed {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

func handleExpire(tx *TxContext, args []byte) ([]byte, error) {
	key, ttl, err := wire.DecodeExpireArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.Expire(key, ttl)
}

func handleIncr(tx *TxContext, args []byte) ([]byte, error) {
	key, delta, err := wire.DecodeIncrArgs(args)
	if err != nil {
		return nil, err
	}
	var current int64
	v, err := tx.Get(key)
	switch {
	case err == cache.ErrNotFound:
		current = 0
	case err != nil:
		return nil, err
	case len(v) != 8:
		return nil, errors.New("ops: incr value is not 8 bytes")
	default:
		current = int64(binary.BigEndian.Uint64(v)) //nolint:gosec // safe: reinterpret stored i64 as u64 for binary read
	}
	next := current + delta
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(next)) //nolint:gosec // safe: store i64 as u64 for binary write
	if err := tx.Put(key, buf, 0); err != nil {
		return nil, err
	}
	return wire.EncodeIncrResult(next), nil
}

// handleSetNX sets key only if it is currently absent or expired. Result: 1=stored,
// 0=key already present. The whole Get→Put runs under the shard write lock, so the
// check and the store are atomic; tx.Get uses the leader-stamped clock on the
// replicated path, so every replica agrees on absent-vs-present.
func handleSetNX(tx *TxContext, args []byte) ([]byte, error) {
	key, val, ttl, err := wire.DecodePutArgs(args)
	if err != nil {
		return nil, err
	}
	switch _, err = tx.Get(key); {
	case err == cache.ErrNotFound:
		if err := tx.Put(key, val, ttl); err != nil {
			return nil, err
		}
		return []byte{1}, nil
	case err != nil:
		return nil, err
	default:
		return []byte{0}, nil
	}
}

// handleCAS sets key to val only if its current value equals expected (or, when
// hasExpected is false, only if the key is currently absent). Result: 1=stored,
// 0=mismatch. Atomic under the shard write lock; the liveness/compare read uses
// the leader-stamped clock on the replicated path.
func handleCAS(tx *TxContext, args []byte) ([]byte, error) {
	key, val, hasExpected, expected, ttl, err := wire.DecodeCASArgs(args)
	if err != nil {
		return nil, err
	}
	cur, err := tx.Get(key)
	switch {
	case err == cache.ErrNotFound:
		if hasExpected {
			return []byte{0}, nil
		}
	case err != nil:
		return nil, err
	default:
		if !hasExpected || !bytes.Equal(cur, expected) {
			return []byte{0}, nil
		}
	}
	if err := tx.Put(key, val, ttl); err != nil {
		return nil, err
	}
	return []byte{1}, nil
}

// handleCompareAndDel deletes key only if its current value equals expected — the
// safe-unlock primitive. Result: 1=deleted, 0=mismatch or absent. Atomic under the
// shard write lock; the compare read uses the leader-stamped clock on the
// replicated path.
func handleCompareAndDel(tx *TxContext, args []byte) ([]byte, error) {
	key, expected, err := wire.DecodeCADArgs(args)
	if err != nil {
		return nil, err
	}
	cur, err := tx.Get(key)
	if err == cache.ErrNotFound {
		return []byte{0}, nil
	}
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(cur, expected) {
		return []byte{0}, nil
	}
	existed, err := tx.Del(key)
	if err != nil {
		return nil, err
	}
	if existed {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// handlePing is a no-op heartbeat used by the client pool's stale-conn check.
// It does not touch the cache, must be cheap, and tolerates non-empty args.
func handlePing(_ *TxContext, _ []byte) ([]byte, error) {
	return nil, nil
}

// handleReady is the DEFAULT (single-node / Direct) readiness handler: always
// ready. Cluster mode overrides it with a real hosted-shard leader check.
func handleReady(_ *TxContext, _ []byte) ([]byte, error) {
	return nil, nil
}

// handleMetrics renders every dense collection's stats into the Prometheus text
// exposition format and returns it as the op result. It is read-only and takes
// no args. A node with no vector store (KV-only) returns ErrVectorsNotAvailable.
func handleMetrics(tx *TxContext, _ []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	var buf bytes.Buffer
	if err := tx.vectors.WritePrometheusAll(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// handleReplMetrics is the DEFAULT (single-node / Direct) replication-metrics
// handler: no shards are replicated, so it reports an empty shard list. Cluster
// mode overrides it with a real per-hosted-shard ISR/lag view. The empty JSON is
// a valid body the /v1/replication handler serves verbatim.
func handleReplMetrics(_ *TxContext, _ []byte) ([]byte, error) {
	return []byte(`{"shards":[]}`), nil
}

func handleVectorCreateCollection(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, cfg, err := wire.DecodeCreateCollectionArgs(args)
	if err != nil {
		return nil, err
	}
	if err := tx.vectors.CreateCollection(name, cfg); err != nil {
		return nil, err
	}
	return nil, nil
}

func handleVectorDropCollection(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := wire.DecodeDropCollectionArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.DropCollection(name)
}

func handleVectorInsert(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// Decode the dense vector into a pooled scratch backing instead of allocating a
	// fresh []float32 per insert. Both downstream paths (RestoreInsert and
	// InsertCASKeyTTL → hnsw.Insert → arena.Insert) COPY the vector into the arena
	// before this handler returns, so the scratch is never retained past the op and
	// is safe to recycle via the defer. (The BULK staging path retains its decoded
	// vecs and is deliberately left on the allocating path.)
	bufp := vectorDenseBufPool.Get().(*[]float32)
	defer vectorDenseBufPool.Put(bufp)
	name, id, vec, ttl, meta, sparse, version, expected, hasExpected, keyTTLMs, keyExpiresAbs, err := wire.DecodeVectorInsertArgsKeyExpiresInto((*bufp)[:0], args)
	if err != nil {
		return nil, err
	}
	*bufp = vec // retain the (possibly grown/reallocated) backing for the next op
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	if version != 0 {
		// Version-preserving reinsert (reshard/resplit backfill): restore the exact
		// per-point CAS version verbatim instead of bumping to 1. keyExpiresAbs is the
		// copied point's ABSOLUTE per-key payload deadline map (from the scan trailer),
		// applied VERBATIM by RestoreInsert (NOT recomputed now+ttl) so resharded
		// per-key TTLs survive time-stable; nil when the point has no per-key TTL.
		//
		// Under a replicated apply stamp the POINT ttl deadline is stamped against the
		// leader clock (RestoreInsertAt) so every replica records the identical absolute
		// point expiry; unstamped (single-node/Direct) keeps the wall-clock path
		// byte-identical (#4 vector TTL determinism).
		if tx.applyStamped {
			return nil, c.RestoreInsertAt(id, vec, ttl, meta, sparse, keyExpiresAbs, version, int64(tx.applyNowMs)) //nolint:gosec // stamped unix-millis fits int64
		}
		return nil, c.RestoreInsert(id, vec, ttl, meta, sparse, keyExpiresAbs, version)
	}
	// keyTTLMs (relative ms) → the engine computes the absolute deadline now+ttl at
	// insert and the WAL logs it (replay restores verbatim). Empty/nil = no per-key
	// TTL (zero-overhead).
	//
	// Under a replicated apply stamp EVERY deadline computation and liveness check
	// (point ttl, per-key ttl, CAS/reclaim) is judged against the leader-stamped
	// clock via InsertCASKeyTTLAt, so replicas at skewed wall clocks store
	// byte-identical committed state; unstamped keeps the wall-clock path
	// byte-identical (branch on applyStamped, NOT applyNowMs != 0 — see TxContext).
	if tx.applyStamped {
		_, err = c.InsertCASKeyTTLAt(id, vec, ttl, meta, sparse, keyTTLMs, vector.CASCond{Expected: expected, Has: hasExpected}, int64(tx.applyNowMs)) //nolint:gosec // stamped unix-millis fits int64
	} else {
		_, err = c.InsertCASKeyTTL(id, vec, ttl, meta, sparse, keyTTLMs, vector.CASCond{Expected: expected, Has: hasExpected})
	}
	return nil, err
}

// handleVectorInsertIfAbsent runs the atomic insert-if-absent engine op (reuses
// the insert-args wire shape). It is registered OpReadWrite: Raft serialization
// is the cross-op atomicity guarantee that closes Race A. Result: [inserted:u8].
func handleVectorInsertIfAbsent(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, vec, ttl, meta, sparse, version, _, _, _, keyExpiresAbs, err := wire.DecodeVectorInsertArgsKeyExpires(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// version!=0 → version-PRESERVING if-absent (the online reshard copy pass,
	// wire.EncodeVectorInsertArgsVersionedKeyExpires): carry the copied point's exact
	// per-point CAS version instead of resetting it to 1, while still never
	// clobbering a concurrent live dual-write (Race A). version==0 is the plain
	// if-absent. keyExpiresAbs is the copied point's ABSOLUTE per-key payload
	// deadline map (from the scan trailer), set VERBATIM on a real insert (NOT
	// recomputed) so resharded per-key TTLs survive time-stable; nil otherwise.
	// Under a replicated apply stamp the liveness OUTCOME (resurrect an expired id
	// vs no-op) and the point-TTL deadline are judged against the leader-stamped
	// clock, so skewed replicas agree on insert-vs-noop and stamp identical
	// deadlines; unstamped keeps the wall-clock path byte-identical (#4 vector TTL
	// determinism).
	var inserted bool
	if tx.applyStamped {
		inserted, err = c.InsertIfAbsentVersionAt(id, vec, ttl, meta, sparse, keyExpiresAbs, version, int64(tx.applyNowMs)) //nolint:gosec // stamped unix-millis fits int64
	} else {
		inserted, err = c.InsertIfAbsentVersion(id, vec, ttl, meta, sparse, keyExpiresAbs, version)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodeIfAbsentResult(inserted), nil
}

// handleVectorExists is the cheap dense liveness probe (OpReadOnly) the copy's
// resurrection guard uses (Race B). Result: [exists:u8].
func handleVectorExists(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, err := wire.DecodeExistsArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	return wire.EncodeExistsResult(c.Exists(id)), nil
}

// handleVectorUpsert reuses the insert-args wire shape; the caller embeds
// document content in the metadata ($content field), so Upsert is called with an
// empty content string and the content rides in meta.
func handleVectorUpsert(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, vec, ttl, meta, sparse, _, expected, hasExpected, keyTTLMs, err := wire.DecodeVectorInsertArgsKeyTTL(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	if ms, stamped := tx.applyStamp(); stamped {
		_, err = c.UpsertCASKeyTTLAt(id, vec, "", ttl, meta, sparse, keyTTLMs, cas, ms)
	} else {
		_, err = c.UpsertCASKeyTTL(id, vec, "", ttl, meta, sparse, keyTTLMs, cas)
	}
	return nil, err
}

// handleVectorBulkStage appends a batch of (id, vector) pairs to a collection's
// bulk-load staging buffer (nothing is indexed until vector_bulk_build).
func handleVectorBulkStage(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// NOTE: the decoded vecs are RETAINED in the collection's stage buffer
	// (StageBulk → c.stageVecs) until vector_bulk_build, so they are NOT poolable
	// the way the single-insert decode buffer is — the per-vector []float32 here is
	// kept live past this op and must stay independently owned.
	name, ids, vecs, err := wire.DecodeBulkStageArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.StageBulk(name, ids, vecs)
}

// handleVectorBulkStagePayload is handleVectorBulkStage for a batch that also
// carries a per-point payload — the shape a filtered workload needs, and the one
// the vectors-only staging op has no room for. The payloads are applied by the
// build's placement pass (see hnsw.BuildConcurrentMeta), so a filter case gets
// the multi-core build instead of one indexed insert per point.
func handleVectorBulkStagePayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// Like handleVectorBulkStage, the decoded vecs (and now the decoded metadata
	// maps) are RETAINED in the collection's stage buffer until vector_bulk_build,
	// so neither is poolable.
	name, ids, vecs, metas, err := wire.DecodeBulkStagePayloadArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.StageBulkPayloads(name, ids, vecs, metas)
}

// handleVectorBulkBuild builds a collection's staged vectors into the (empty)
// index in one concurrent pass — the multi-core initial-load path.
func handleVectorBulkBuild(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, workers, err := wire.DecodeBulkBuildArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.BuildStaged(name, workers)
}

func handleVectorSearchDocs(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, k, query, filter, err := wire.DecodeVectorSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	docs, err := c.SearchDocs(query, k, filter)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(docs), nil
}

func handleVectorSearchGroups(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, k, query, opts, err := wire.DecodeGroupSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	groups, err := c.SearchGroups(query, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeGroups(groups), nil
}

func handleVectorGroupCandidates(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, _, query, opts, err := wire.DecodeGroupSearchArgs(args) // k unused; coordinator groups
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	cands, err := c.GroupCandidates(query, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(cands), nil
}

func handleVectorScroll(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, filter, limit, _, _, afterID, hasAfter, order, err := wire.DecodeScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// Cursor-aware page (the partition fan-out passes the SAME global cursor to
	// every partition; the coordinator derives next_cursor from the merged docs, so
	// this handler returns only the per-partition docs). With no cursor
	// (hasAfter=false) and no order_by this is the deterministic id-ascending
	// smallest-id `limit` scroll.
	if order != nil {
		// order_by page: sort the live admitted set by the (value, id) order, EXCLUDE
		// missing-field points, seek past the (resumeKey, afterID) cursor / start_from.
		ob := wire.ScrollOrderToOrderBy(order)
		var afterKey float64
		if order.HasResume {
			afterKey = order.ResumeKey
		}
		docs, _, _, derr := c.ScrollDocsPageOrder(filter, ob, afterID, afterKey, hasAfter, limit)
		if derr != nil {
			return nil, derr
		}
		return wire.EncodeVectorDocs(docs), nil
	}
	docs, _, _, err := c.ScrollDocsPage(filter, afterID, hasAfter, limit)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(docs), nil
}

// handleVectorScanVectors enumerates every live record of a (physical
// partition) collection — the read primitive an offline resplit uses to
// re-insert each vector into a re-hashed generation. Read-only.
func handleVectorScanVectors(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := wire.DecodeScanVectorsArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	return wire.EncodeScanVectorsResult(c.ScanVectors()), nil
}

// handleVectorGetConfig returns a collection's Config so resplit can create the
// new-generation partitions with the same configuration. Read-only.
func handleVectorGetConfig(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := wire.DecodeGetConfigArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	return wire.EncodeGetConfigResult(c.Config()), nil
}

func handleVectorDeleteByFilter(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, filter, err := wire.DecodeDeleteByFilterArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	var n int
	if ms, stamped := tx.applyStamp(); stamped {
		n, err = c.DeleteByFilterAt(filter, ms)
	} else {
		n, err = c.DeleteByFilter(filter)
	}
	if err != nil {
		return nil, err
	}
	out := make([]byte, 4)
	binary.BigEndian.PutUint32(out, uint32(n)) //nolint:gosec // count >= 0
	return out, nil
}

func handleVectorHybridSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, k, sparse, opts, err := wire.DecodeHybridSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	results, err := c.HybridSearch(dense, sparse, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridResults(results), nil
}

func handleVectorHybridLanes(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, k, sparse, opts, err := wire.DecodeHybridSearchArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	denseRes, sparseRes, err := c.HybridLanes(dense, sparse, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridLanesResult(denseRes, sparseRes), nil
}

// handleVectorSearchText runs a BM25 full-text search: the raw query text is
// tokenized + scored server-side (the SDK ships no tokens). Returns Documents
// (content + metadata), like vector_search_docs. The collection must have been
// created with FullText (else the engine returns ErrFullTextDisabled).
func handleVectorSearchText(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, query, k, filter, _, _, _, _, g, err := wire.DecodeSearchTextArgsGlobal(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// Phase 1 of the global-DF (dfs) fan-out: when the coordinator supplied global
	// stats, score this shard's LOCAL postings with the injected global N/avgdl/df
	// so the returned scores are globally comparable. When absent, the EXISTING
	// per-shard-local path runs unchanged.
	if g != nil {
		docs, err := c.SearchTextGlobalDocs(query, k, filter, *g)
		if err != nil {
			return nil, err
		}
		return wire.EncodeVectorDocs(docs), nil
	}
	docs, err := c.SearchText(query, k, filter)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(docs), nil
}

// handleVectorBM25Stats is phase 0 of the global-DF (dfs_query_then_fetch) text
// fan-out: it returns this partition's CORPUS-WIDE BM25 stats (n, tokenTotal, and
// per-query-term df) for the query's analyzed terms. The coordinator sums these
// across all partitions into the global N/avgdl/df injected into phase 1. A
// collection without a BM25 lane contributes zero/empty (no error), so a mixed
// fleet sums cleanly.
func handleVectorBM25Stats(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, query, _, _, _, err := wire.DecodeBM25StatsArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	n, tokenTotal, df := c.CorpusStats(query)
	return wire.EncodeBM25StatsResult(n, tokenTotal, df), nil
}

// handleVectorHybridText fuses a dense KNN lane with a BM25 full-text lane. The
// raw query text is analyzed server-side. Returns fused hybrid results.
func handleVectorHybridText(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, query, k, opts, err := wire.DecodeHybridTextArgs(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	results, err := c.HybridText(dense, query, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridResults(results), nil
}

// handleVectorHybridTextLanes returns the UNFUSED dense + BM25-text candidate
// lanes for the partitioned hybrid-text fan-out (text_fanout.go), mirroring
// vector_hybrid_lanes. It shares the vector_hybrid_text wire (decoded with
// wire.DecodeHybridTextArgs).
func handleVectorHybridTextLanes(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, dense, query, k, opts, _, _, _, _, g, err := wire.DecodeHybridTextArgsGlobal(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// Phase 1 of the global-DF (dfs) hybrid-text fan-out: score the text lane with
	// the injected global stats when supplied; otherwise the EXISTING per-shard-local
	// lane builder runs unchanged. The dense lane is identical either way.
	var (
		denseRes, textRes []vector.Result
	)
	if g != nil {
		denseRes, textRes, err = c.HybridTextLanesGlobal(dense, query, k, opts, *g)
	} else {
		denseRes, textRes, err = c.HybridTextLanes(dense, query, k, opts)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridLanesResult(denseRes, textRes), nil
}

func handleVectorDelete(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, expected, hasExpected, err := wire.DecodeVectorDeleteArgsCAS(args)
	if err != nil {
		return nil, err
	}
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var removed bool
	if ms, stamped := tx.applyStamp(); stamped {
		removed, err = c.DeleteCASAt(id, cas, ms)
	} else {
		removed, err = c.DeleteCAS(id, cas)
	}
	if err != nil {
		return nil, err
	}
	if removed {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// handleVectorGet retrieves a dense point by id: vector + payload + remaining TTL
// + sparse, gated by the with_vector/with_payload flags. A missing point returns
// the found=0 FLAG (NEVER an op error), so a point-op fan-out treats "absent in
// this partition" as expected. Read-only.
func handleVectorGet(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, flags, err := wire.DecodeVectorGetArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&wire.GetFlagWithVector != 0
	withPayload := flags&wire.GetFlagWithPayload != 0
	// Read the dense vector into a pooled scratch backing instead of allocating a
	// fresh []float32 per get. wire.EncodeVectorGetResultV serializes the vector into the
	// response bytes (a full copy) before this handler returns, so the scratch is
	// never retained past the op and is recycled via the defer. (The BATCH get
	// handler retains each row's vec until the batch encode and stays unpooled.)
	bufp := vectorDenseBufPool.Get().(*[]float32)
	defer vectorDenseBufPool.Put(bufp)
	vec, meta, ttl, sparse, version, ok, err := tx.vectors.GetPointVersionInto((*bufp)[:0], name, id)
	if err != nil {
		return nil, err
	}
	if vec != nil {
		*bufp = vec // retain the grown backing; a miss returns nil, leave the buffer intact
	}
	return wire.EncodeVectorGetResultV(ok, vec, meta, ttl, sparse, withVec, withPayload, version), nil
}

// handleVectorGetBatch retrieves a subset of dense points by id in one op: for each
// requested id it runs the same GetPoint lookup as handleVectorGet and emits a row.
// A missing id is a Found=false row (NEVER an op error) so the coordinator can derive
// the global missing set from absent ids. Rows preserve the input id order (this is
// the per-partition handler — it returns rows for ITS id-subset in the order given).
// The with_vector/with_payload flags gate the vec and the meta+sparse projections,
// applied here at fetch time exactly as single get. Read-only.
func handleVectorGetBatch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, ids, flags, err := wire.DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&wire.GetFlagWithVector != 0
	withPayload := flags&wire.GetFlagWithPayload != 0
	// Acquire the collection ONCE for the whole batch and fetch each id through the
	// projection-aware getter, so a with_vector=false / with_payload=false batch pays
	// one Acquire/Release and copies NOTHING per point — versus the old per-id
	// GetPointVersion, which re-Acquired and deep-copied the dense vector + meta map +
	// sparse vector for every id regardless of the requested projection (a
	// vector_get_batch of 1000 dim-768 ids with with_vector=false copied ~3 MB of
	// float32 garbage per op). The callback owns each row's vec/meta/sparse and
	// retains them in the row slice until the single wire.EncodeVectorGetBatchResult below.
	rows := make([]wire.GetBatchRow, 0, len(ids))
	if err := tx.vectors.GetPointsProjected(name, ids, withVec, withPayload,
		func(id uint64, vec []float32, meta vector.Metadata, ttl time.Duration, sparse *vector.SparseVector, version uint64, ok bool) {
			row := wire.GetBatchRow{ID: id, Found: ok}
			if ok {
				row.Vec = vec                          // nil when !withVec (getter skipped the copy)
				row.Meta = meta                        // nil when !withPayload
				row.Sparse = sparse                    // nil when !withPayload
				row.TTLMs = uint64(ttl.Milliseconds()) //nolint:gosec // TTL >= 0
				row.Version = version
			}
			rows = append(rows, row)
		}); err != nil {
		return nil, err
	}
	return wire.EncodeVectorGetBatchResult(rows), nil
}

// handleVectorSetPayload merges the provided payload into the point's existing
// payload (reindexing + WAL-logging on a dense WAL-mode collection). A missing
// point returns applied=0 (the not-found FLAG, not an op error); a bad payload
// JSON is a hard decode error (fail-loud). Read-write.
func handleVectorSetPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, meta, keyTTLMs, expected, hasExpected, err := wire.DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.SetPayloadCASAt(name, id, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.SetPayloadCAS(name, id, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleVectorOverwritePayload replaces the point's entire payload. applied=0 for a
// missing point (not-found flag); bad JSON is a hard error. Read-write.
func handleVectorOverwritePayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, meta, keyTTLMs, expected, hasExpected, err := wire.DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.OverwritePayloadCASAt(name, id, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.OverwritePayloadCAS(name, id, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleVectorDeletePayloadKeys removes the listed keys from the point's payload.
// applied=0 for a missing point (not-found flag). Read-write.
func handleVectorDeletePayloadKeys(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, keys, expected, hasExpected, err := wire.DecodeDeletePayloadKeysArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.DeletePayloadKeysCASAt(name, id, keys, cas, ms)
	} else {
		applied, _, err = tx.vectors.DeletePayloadKeysCAS(name, id, keys, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleVectorClearPayload clears the point's payload. applied=0 for a missing
// point (not-found flag). Read-write.
func handleVectorClearPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, expected, hasExpected, err := wire.DecodeClearPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.ClearPayloadCASAt(name, id, cas, ms)
	} else {
		applied, _, err = tx.vectors.ClearPayloadCAS(name, id, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

func handleVectorSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	sc := vectorSearchPool.Get().(*vectorSearchScratch)
	defer vectorSearchPool.Put(sc)

	name, k, query, filter, err := wire.DecodeVectorSearchArgsInto(args, sc.query)
	if err != nil {
		return nil, err
	}
	sc.query = query // retain the (possibly regrown) buffer for reuse
	c, ok := tx.vectors.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	defer c.Release()
	// SearchInto writes into the pooled result slice (zero-alloc engine path);
	// wire.EncodeVectorSearchResults copies it into the response buffer before the
	// deferred Put recycles the scratch.
	results, err := c.SearchInto(sc.results[:0], query, k, filter)
	if err != nil {
		return nil, err
	}
	sc.results = results
	return wire.EncodeVectorSearchResults(results), nil
}
