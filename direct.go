// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
	"github.com/rostamlabs/rostam/wasm"
)

// DirectConfig configures a no-replication in-process Store. Use this
// when you need persistence (via mmap) but explicitly do NOT
// need Raft replication — single-node deployments. Writes bypass the
// Raft log entirely; durability comes from the mmap header (if
// Cache.Durable is set) and Cache.Close's final msync.
//
// All registered ops (read-only AND read-write) execute directly
// against the cache inside the shard's write lock. There is no
// applied-index, no log, no quorum.
type DirectConfig struct {
	// DataDir is the root directory for the cache's mmap file. Empty
	// means heap mode (no persistence).
	DataDir string

	// Ops is the caller's op registry. MUST include ops.RegisterBuiltins
	// plus any caller-registered ops. Required.
	Ops *ops.Registry

	// Cache configures the cache layer (mmap knobs).
	Cache CacheConfig

	// Authenticator, when non-nil, gates every request on all transports. It is
	// the unified RBAC authorizer (authz.Authenticator): it receives an
	// authz.AuthRequest{Token, Op, Args} (token from the protocol-v2 frame /
	// Bearer header / gRPC metadata — empty for v1 clients) and returns true to
	// allow. nil = no auth (legacy/open mode).
	//
	// Build it from a vector.KeyRegistry with granular per-collection scopes:
	//
	//	reg, _ := vector.OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	//	auth := authz.NewRBACAuthenticator(reg, opsReg, internalToken)
	Authenticator server.Authenticator
}

// NewDirect constructs an in-process Store backed by a single cache.Cache,
// with no Raft layer. Writes are ~30× faster than NewEmbedded because no
// log entry is created, no FSM dispatch happens, and no applied-index
// bookkeeping runs.
//
// Use NewEmbedded when you need replication (multi-node clusters). Use
// NewDirect when you have a single-node deployment and want the cache
// layer's raw write speed.
func NewDirect(cfg DirectConfig) (Store, error) {
	if cfg.Ops == nil {
		return nil, errors.New("rostam: DirectConfig.Ops is required")
	}

	cc := cache.DefaultConfig()
	if cfg.Cache.NumShardsPerNode > 0 {
		cc.NumShards = cfg.Cache.NumShardsPerNode
	}
	cc.DataDir = cfg.DataDir
	cc.Durable = cfg.Cache.Durable
	cc.Mlock = cfg.Cache.Mlock
	cc.DisableColdCompaction = cfg.Cache.DisableColdCompaction
	if cfg.Cache.MsyncIntervalMs > 0 {
		cc.MsyncIntervalMs = cfg.Cache.MsyncIntervalMs
	}
	// Derive the per-shard cap + page size from a TOTAL budget (after NumShards
	// is final — the geometry divides by it). Without this the per-shard cap
	// stayed at cache.DefaultConfig()'s 256 MiB and the real bound was
	// NumShards * 256 MiB: 64 GiB by default, 256 GiB at -shards 1024.
	if err := applyCacheBudget(&cc, cfg.Cache.MaxMemoryBytes, cc.NumShards); err != nil {
		return nil, err
	}

	c, err := cache.New(cc)
	if err != nil {
		return nil, fmt.Errorf("rostam: cache.New: %w", err)
	}
	vectorStore, err := vector.OpenCollectionStore(cfg.DataDir)
	if err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("rostam: open vector store: %w", err)
	}
	return &directStore{
		cache:    c,
		registry: cfg.Ops,
		vectors:  vectorStore,
		tx:       ops.NewTxContextWithVectors(c, vectorStore),
		opMu:     make([]sync.Mutex, c.NumShards()),
	}, nil
}

// directStore implements Store on top of a bare cache.Cache. It does
// not provide replication or log durability — only the persistence
// the mmap layer gives it.
type directStore struct {
	cache    *cache.Cache
	registry *ops.Registry
	vectors  *vector.CollectionStore
	tx       *ops.TxContext // reused across every Call — TxContext is stateless beyond *cache.Cache
	opMu     []sync.Mutex   // per-shard read-write op serialization (one per cache shard).
	//                         A routable RW op locks only its routing key's shard, so writes
	//                         to different shards run concurrently — matching Embedded's
	//                         independent per-shard Raft groups. Shardless ops lock all shards.
	wasmRT *wasm.Runtime // lazily created on first RegisterWASM
}

func (d *directStore) Get(_ context.Context, key []byte) ([]byte, error) {
	raw, err := d.cache.Get(key)
	if err != nil {
		if errors.Is(err, cache.ErrNotFound) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	return raw, nil
}

func (d *directStore) GetInto(_ context.Context, key, dst []byte) ([]byte, error) {
	raw, err := d.cache.GetInto(dst[:0], key)
	if err != nil {
		if errors.Is(err, cache.ErrNotFound) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	return raw, nil
}

// Put and Del write straight to the cache but MUST take the same per-shard opMu
// that Call's read-write path uses, so a raw Put/Del serializes against a
// multi-step RMW op (e.g. incr) on the same key. Without it, a Put could land
// between an incr's tx.Get and tx.Put and be silently overwritten (lost update),
// a divergence from Embedded — where put/del route through the same per-shard
// Raft FSM as incr and can never interleave. Get stays lock-free like read-only
// ops (the cache shard RWMutex gives each read its own atomicity).
func (d *directStore) Put(_ context.Context, key, value []byte, ttl time.Duration) error {
	mu := &d.opMu[d.cache.ShardIndex(key)]
	mu.Lock()
	defer mu.Unlock()
	return d.cache.Put(key, value, ttl)
}

func (d *directStore) Del(_ context.Context, key []byte) (bool, error) {
	mu := &d.opMu[d.cache.ShardIndex(key)]
	mu.Lock()
	defer mu.Unlock()
	return d.cache.Del(key)
}

// PutBatch writes many key/value pairs. A directStore is a single logical KV (no
// cross-shard Raft routing), so it simply chunks the entries to ops.MaxPutBatchSize
// and dispatches each chunk through the same registry Call path Put's read-write ops
// use ("put_batch" applies every entry within one op). Empty entries is a no-op.
func (d *directStore) PutBatch(_ context.Context, entries []ops.PutEntry) error {
	for len(entries) > 0 {
		chunk := entries
		if len(chunk) > ops.MaxPutBatchSize {
			chunk = entries[:ops.MaxPutBatchSize]
		}
		if _, err := d.Call(context.Background(), "put_batch", ops.EncodePutBatchArgs(chunk)); err != nil {
			return err
		}
		entries = entries[len(chunk):]
	}
	return nil
}

func (d *directStore) Call(_ context.Context, op string, args []byte) ([]byte, error) {
	handler, kind, ke, crossShard, ok := d.registry.LookupEntry(op)
	if !ok {
		return nil, fmt.Errorf("rostam: op %q not registered", op)
	}
	// Read-only ops run concurrently — the cache's per-shard RWMutex gives each
	// read its atomicity. A read-write op serializes so a multi-step RMW handler
	// (read-then-write) is atomic, matching Raft's single FSM-apply in Embedded.
	//
	// A SHARD-CONFINED routable op (KV builtins, vector ops) touches only its
	// routing key's shard, so it locks just that shard — writes to different
	// shards then run in parallel, exactly as Embedded's independent per-shard
	// Raft groups do. A SHARDLESS op (nil extractor / unparseable key) or a
	// CROSS-SHARD op (a WASM handler that may touch arbitrary keys) takes the
	// all-shards barrier for mutual exclusion against every other read-write op.
	if kind == ops.OpReadWrite {
		if ke != nil && !crossShard {
			if key, ok := ke(args); ok {
				mu := &d.opMu[d.cache.ShardIndex(key)]
				mu.Lock()
				defer mu.Unlock()
				return handler(d.tx, args)
			}
		}
		d.lockAllShards()
		defer d.unlockAllShards()
	}
	return handler(d.tx, args)
}

// lockAllShards acquires every per-shard op lock in ascending index order — a
// global write barrier for shardless or unparseable read-write ops (and for
// RegisterWASM). The fixed order is deadlock-free against the single-shard
// lockers, which only ever hold one lock and never wait on a second.
func (d *directStore) lockAllShards() {
	for i := range d.opMu {
		d.opMu[i].Lock()
	}
}

func (d *directStore) unlockAllShards() {
	for i := range d.opMu {
		d.opMu[i].Unlock()
	}
}

// IsLeader returns true unconditionally. Direct mode is single-node by
// definition, so the local process is always the "leader" of itself.
func (d *directStore) IsLeader(_ []byte) bool { return true }

// LeaderAddr returns "" because Direct mode does not expose a network
// surface. Callers wanting a network endpoint should use NewEmbedded
// (which wraps cluster.Node and serves over TCP).
func (d *directStore) LeaderAddr(_ []byte) string { return "" }

func (d *directStore) Close() error {
	var errs []error
	if d.wasmRT != nil {
		if err := d.wasmRT.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if d.vectors != nil {
		// A Direct store has no Raft log to replay from, so a Persistent
		// collection's mmap sidecar is the ONLY thing that carries its contents
		// across a restart — and nothing writes that sidecar except a Flush. Doing
		// it here makes a clean shutdown the flush point, which is what makes
		// Persistent mean anything at all for an embedded caller (before this, a
		// Persistent collection reopened empty). Non-persistent collections are
		// untouched: FlushPersistent skips them, so this costs a map scan for
		// stores that never asked for persistence.
		if err := d.vectors.FlushPersistent(); err != nil {
			errs = append(errs, err)
		}
		if err := d.vectors.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := d.cache.Close(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

// directStore is a no-Raft, single-node backend (RF==1, no replicas), so the
// write-consistency factor can never exceed majority — the barrier is a documented
// no-op here. The opts are accepted to satisfy the Store interface and ignored.
func (d *directStore) VectorInsert(_ context.Context, collection string, id uint64, vec []float32, _ ...WriteOpts) error {
	_, err := d.Call(context.Background(), "vector_insert", ops.EncodeVectorInsertArgs(collection, id, vec))
	return err
}

func (d *directStore) VectorSearch(_ context.Context, collection string, query []float32, k int) ([]VectorResult, error) {
	body, err := d.Call(context.Background(), "vector_search", ops.EncodeVectorSearchArgs(collection, k, query))
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorSearchResults(body)
}

// VectorSearchInto hits the HNSW engine's zero-alloc SearchInto path directly,
// bypassing the wire encode/decode the Call-based methods pay even in-process.
// With a reused dst this is the lowest-latency, allocation-free embedded search.
func (d *directStore) VectorSearchInto(_ context.Context, collection string, query []float32, k int, dst []VectorResult) ([]VectorResult, error) {
	if d.vectors == nil {
		return nil, ops.ErrVectorsNotAvailable
	}
	c, ok := d.vectors.Acquire(collection)
	if !ok {
		return nil, fmt.Errorf("rostam: unknown collection %q", collection)
	}
	defer c.Release()
	return c.SearchInto(dst, query, k, vector.Filter{})
}

func (d *directStore) VectorDelete(_ context.Context, collection string, id uint64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_delete", ops.EncodeVectorDeleteArgs(collection, id))
	if err != nil {
		return false, err
	}
	return len(body) == 1 && body[0] == 1, nil
}

func (d *directStore) VectorInsertIfAbsent(_ context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_insert_if_absent", ops.EncodeVectorInsertArgsExt(collection, id, vec, opts.TTL, opts.Metadata, opts.Sparse))
	if err != nil {
		return false, err
	}
	return ops.DecodeIfAbsentResult(body)
}

func (d *directStore) VectorExists(_ context.Context, collection string, id uint64) (bool, error) {
	body, err := d.Call(context.Background(), "vector_exists", ops.EncodeExistsArgs(collection, id))
	if err != nil {
		return false, err
	}
	return ops.DecodeExistsResult(body)
}

// CreateCollection creates a single-partition collection.
//
// Partitions > 1 is REFUSED rather than ignored. A Direct store has no partition
// catalog at all — VectorResplit/VectorReshard below say so explicitly — so the
// count could never take effect, and silently accepting it handed the caller a
// collection with a topology they did not ask for and no way to notice: the
// create returned 200, every write landed on one lock, and nothing said why.
// That is the same "silently dropping a field" failure the bulk-staging route
// already refuses (see putPointsBulk), and it is worst on a single node, which
// is what everyone tries first.
//
// Partitioning requires the clustered/embedded backend (NewEmbedded).
func (d *directStore) CreateCollection(_ context.Context, name string, cfg VectorConfig) error {
	if cfg.Partitions > 1 {
		return fmt.Errorf("rostam: Partitions=%d requires the clustered backend: a Direct store "+
			"has no partition catalog, so it can only create single-partition collections "+
			"(use NewEmbedded, or leave Partitions unset)", cfg.Partitions)
	}
	_, err := d.Call(context.Background(), "vector_create_collection", ops.EncodeCreateCollectionArgs(name, cfg))
	return err
}

func (d *directStore) VectorInsertExt(_ context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) error {
	_, err := d.Call(context.Background(), "vector_insert", ops.EncodeVectorInsertArgsKeyTTL(collection, id, vec, opts.TTL, opts.Metadata, opts.Sparse, opts.KeyTTLMs))
	return err
}

// VectorSearchExt returns a zero-value FanMeta: a Direct store has no partition
// catalog and never fans out, so a read is never degraded.
func (d *directStore) VectorSearchExt(_ context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_search", ops.EncodeVectorSearchArgsExt(collection, k, query, opts.Filter))
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeVectorSearchResults(body)
	return res, FanMeta{}, err
}

func (d *directStore) VectorHybridSearch(_ context.Context, collection string, dense []float32, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_hybrid_search", ops.EncodeHybridSearchArgs(collection, dense, k, opts.Sparse, toVectorHybridOpts(opts)))
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// mergeSingleShardQuery routes ONE shard's mode-tagged QueryResult through the
// SAME merge the embedded single-shard path uses, so a Direct store fuses/reranks
// identically to the single-node engine. RERANK merges the partition-local reranked
// top-k; FUSION folds the UNFUSED prefetch lanes. A spec with a nested MULTI-lane
// FUSION node (vector.SpecHasNestedFusion) ships the EXPANDED pre-order tree-lane
// list, which MUST be folded via treeFusionMergeFanOut — the flat fusionMergeFanOut
// consumes only len(spec.Prefetch) top-level lanes, mis-associating the nested
// sub-lanes and silently dropping trailing top-level leaves (finding 004). Mirrors
// the guarded single-shard branch in (*embedded).VectorMVQuery / VectorNamedQuery.
func mergeSingleShardQuery(qr vector.QueryResult, spec vector.QuerySpec) ([]VectorResult, error) {
	single := []vector.QueryResult{qr}
	switch spec.Mode {
	case vector.ModeRerank:
		return rerankMergeFanOut(single, spec.Root, queryK(spec)), nil
	default: // ModeFusion
		if vector.SpecHasNestedFusion(spec) {
			return treeFusionMergeFanOut(single, spec, queryK(spec))
		}
		return fusionMergeFanOut(single, spec, queryK(spec)), nil
	}
}

// VectorQuery runs the unified Query API on the single direct shard. A Direct
// store is non-clustered (one shard, no partition catalog), so the result is never
// degraded. The per-shard handler returns a MODE-TAGGED payload: RERANK fills Fused
// (the reranked top-k); FUSION fills Lanes (the UNFUSED prefetch lanes — the wire
// never carries the engine's locally-fused list). Route the one shard's result
// through the SAME merge the embedded single-shard path uses so FUSION actually
// fuses (reading qr.Fused directly would drop every FUSION result).
func (d *directStore) VectorQuery(_ context.Context, collection string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_query",
		ops.EncodeQueryArgs(collection, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, err := ops.DecodeQueryResult(body)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := mergeSingleShardQuery(qr, spec)
	if err != nil {
		return nil, FanMeta{}, err
	}
	return res, FanMeta{}, nil
}

func (d *directStore) VectorQueryGrouped(_ context.Context, collection string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorGroup, FanMeta, error) {
	// Single-node: handleVectorQuery returns the UNGROUPED flat result + per-id key map
	// (QueryGroupedFanOut). Run the SAME merge+group step the fan-out coordinator uses
	// over the one shard, so direct mode groups exactly like the single-node engine.
	body, err := d.Call(context.Background(), "vector_query",
		ops.EncodeQueryArgs(collection, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, keys, derr := ops.DecodeQueryResultGroupedFanOut(body)
	if derr != nil {
		return nil, FanMeta{}, derr
	}
	groupsK := queryK(spec)
	groupSize := spec.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	fetchK := vector.GroupFetchK(groupsK, groupSize, 0)
	groups := groupMergedQueryParts([]queryGroupPart{{qr: qr, keys: keys}}, spec, fetchK, groupsK, groupSize)
	return groups, FanMeta{}, nil
}

func (d *directStore) VectorUpsert(_ context.Context, collection string, id uint64, vec []float32, content string, opts VectorInsertOpts) error {
	_, err := d.Call(context.Background(), "vector_upsert", ops.EncodeVectorUpsertArgsCASKeyTTL(collection, id, vec, content, opts.TTL, opts.Metadata, opts.Sparse, 0, false, opts.KeyTTLMs))
	return err
}

func (d *directStore) VectorSearchDocs(_ context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_search_docs", ops.EncodeVectorSearchArgsExt(collection, k, query, opts.Filter))
	if err != nil {
		return nil, FanMeta{}, err
	}
	docs, err := ops.DecodeVectorDocs(body)
	return docs, FanMeta{}, err
}

func (d *directStore) VectorSearchText(_ context.Context, collection string, query string, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_search_text", ops.EncodeSearchTextArgs(collection, query, k, opts.Filter))
	if err != nil {
		return nil, FanMeta{}, err
	}
	docs, err := ops.DecodeVectorDocs(body)
	return docs, FanMeta{}, err
}

func (d *directStore) VectorHybridText(_ context.Context, collection string, dense []float32, query string, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_hybrid_text", ops.EncodeHybridTextArgs(collection, dense, query, k, toVectorHybridOpts(opts)))
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

func (d *directStore) VectorDeleteByFilter(_ context.Context, collection string, filter VectorFilter) (int, error) {
	body, err := d.Call(context.Background(), "vector_delete_by_filter", ops.EncodeDeleteByFilterArgs(collection, filter))
	if err != nil {
		return 0, err
	}
	return ops.DecodeDeleteByFilterResult(body)
}

func (d *directStore) VectorSearchGroups(_ context.Context, collection string, query []float32, k int, opts VectorGroupOpts) ([]VectorGroup, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_search_groups", ops.EncodeGroupSearchArgs(collection, k, query, opts))
	if err != nil {
		return nil, FanMeta{}, err
	}
	groups, err := ops.DecodeGroups(body)
	return groups, FanMeta{}, err
}

func (d *directStore) VectorMVCreateCollection(_ context.Context, name string, cfg MultiVectorConfig) error {
	_, err := d.Call(context.Background(), "vector_mv_create_collection", ops.EncodeMVCreateArgs(name, cfg))
	return err
}

func (d *directStore) VectorMVDropCollection(_ context.Context, name string) error {
	_, err := d.Call(context.Background(), "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(name, 0))
	return err
}

func (d *directStore) VectorMVAdd(_ context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata, opts ...WriteOpts) error {
	// keyTTL block rides AFTER the base block; the OPTIONAL doc-level sparse rides
	// LAST. Empty map + nil sparse = byte-identical to EncodeMVAddArgs (the prior wire
	// shape for this path).
	wo := firstWriteOpts(opts)
	_, err := d.Call(context.Background(), "vector_mv_add", ops.EncodeMVAddArgsCASKeyTTLSparse(name, docID, tokens, meta, 0, false, wo.KeyTTLMs, wo.Sparse))
	return err
}

func (d *directStore) VectorMVSearch(_ context.Context, name string, query [][]float32, k int, opts MultiSearchOpts) ([]MultiResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_mv_search", ops.EncodeMVSearchArgs(name, query, k, opts.CandidatesPerToken))
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := ops.DecodeMVResults(body)
	return res, FanMeta{}, err
}

func (d *directStore) VectorMVDelete(_ context.Context, name string, docID uint64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_delete", ops.EncodeMVDeleteArgs(name, docID))
	if err != nil {
		return false, err
	}
	return len(body) > 0 && body[0] == 1, nil
}

func (d *directStore) VectorMVAddIfAbsent(_ context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_add_if_absent", ops.EncodeMVAddArgs(name, docID, tokens, meta))
	if err != nil {
		return false, err
	}
	return ops.DecodeIfAbsentResult(body)
}

func (d *directStore) VectorMVExists(_ context.Context, name string, docID uint64) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_exists", ops.EncodeMVExistsArgs(name, docID))
	if err != nil {
		return false, err
	}
	return ops.DecodeExistsResult(body)
}

func (d *directStore) VectorNamedCreateCollection(_ context.Context, name string, cfg map[string]NamedVectorParams, partitions int) error {
	_, err := d.Call(context.Background(), "vector_named_create_collection", ops.EncodeNamedCreateArgs(name, cfg, partitions))
	return err
}

func (d *directStore) VectorNamedDropCollection(_ context.Context, name string) error {
	_, err := d.Call(context.Background(), "vector_named_drop_collection", ops.EncodeNamedNameArgs(name))
	return err
}

func (d *directStore) VectorNamedInsert(_ context.Context, name string, id uint64, vectors map[string][]float32, payload VectorMetadata, ttl time.Duration, opts ...WriteOpts) error {
	// keyTTL block rides AFTER the base block; empty map = byte-identical to
	// EncodeNamedInsertArgs (the prior wire shape for this path).
	_, err := d.Call(context.Background(), "vector_named_insert", ops.EncodeNamedInsertArgsKeyTTL(name, id, vectors, payload, ttl, firstWriteOpts(opts).KeyTTLMs))
	return err
}

// VectorNamedQuery runs the named-collection Query API on the single direct shard.
// A Direct store is non-clustered (one shard, no partition catalog), so the result
// is never degraded. The per-shard handler returns a MODE-TAGGED payload: RERANK
// fills Fused (the reranked top-k); FUSION fills Lanes (the UNFUSED prefetch lanes
// — the wire never carries the engine's locally-fused list). Route the one shard's
// result through the SAME merge the embedded single-shard path uses so FUSION
// actually fuses (reading qr.Fused directly would drop every FUSION result). The
// named analogue of VectorQuery.
func (d *directStore) VectorNamedQuery(_ context.Context, name string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_named_query",
		ops.EncodeQueryArgs(name, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, err := ops.DecodeQueryResult(body)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := mergeSingleShardQuery(qr, spec)
	if err != nil {
		return nil, FanMeta{}, err
	}
	return res, FanMeta{}, nil
}

func (d *directStore) VectorNamedSearch(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorResult, error) {
	return d.VectorNamedSearchExt(ctx, name, vectorName, query, k, NamedSearchOpts{Filter: filter})
}

// VectorNamedSearchExt: the single-node direct engine ignores read-consistency
// (no clustering / no peers), but threads opts.Filter through unchanged. The
// rc/opa trailer is byte-identical to the legacy encoder when both are 0.
func (d *directStore) VectorNamedSearchExt(_ context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorResult, error) {
	body, err := d.Call(context.Background(), "vector_named_search",
		ops.EncodeNamedSearchArgsOpts(name, vectorName, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorSearchResults(body)
}

func (d *directStore) VectorNamedSparseSearch(ctx context.Context, name, space string, query VectorSparse, k int, filter VectorFilter) ([]VectorResult, error) {
	return d.VectorNamedSparseSearchExt(ctx, name, space, query, k, NamedSearchOpts{Filter: filter})
}

func (d *directStore) VectorNamedSparseSearchExt(_ context.Context, name, space string, query VectorSparse, k int, opts NamedSearchOpts) ([]VectorResult, error) {
	body, err := d.Call(context.Background(), "vector_named_sparse_search",
		ops.EncodeNamedSparseSearchArgsOpts(name, space, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, err
	}
	// The sparse handler returns score-carrying hybrid results (id+distance+score).
	return ops.DecodeHybridResults(body)
}

// VectorNamedHybridSearch fuses a dense + a sparse named space. The single-node
// direct engine has no partitions, so it always runs the fused-search op directly
// (the engine collapses single-lane degradation); rc/opa ride the args but are
// inert here.
func (d *directStore) VectorNamedHybridSearch(ctx context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, error) {
	res, _, err := d.VectorNamedHybridSearchExt(ctx, name, denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
	return res, err
}

// VectorNamedHybridSearchExt returns a zero-value FanMeta: a Direct store has no
// partition catalog and never fans out, so a hybrid read is never degraded (the
// FanMeta channel exists only for interface symmetry with the clustered backend).
func (d *directStore) VectorNamedHybridSearchExt(_ context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_named_hybrid_search",
		ops.EncodeNamedHybridArgs(name, denseSpace, denseQ, sparseSpace, sparseQ, k, toNamedHybridVectorOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, err
	}
	// The fused handler returns score-carrying hybrid results (id+distance+score).
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// VectorMVHybridSearch fuses an MV collection's MaxSim lane + its doc-level sparse
// lane. The single-node direct engine has no partitions, so it always runs the
// fused-search op directly (the engine collapses single-lane degradation); rc/opa
// ride the args but are inert here.
func (d *directStore) VectorMVHybridSearch(ctx context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, error) {
	res, _, err := d.VectorMVHybridSearchExt(ctx, name, query, sparseQ, k, opts)
	return res, err
}

// VectorMVHybridSearchExt returns a zero-value FanMeta: a Direct store has no
// partition catalog and never fans out, so a hybrid read is never degraded (the
// FanMeta channel exists only for interface symmetry with the clustered backend).
func (d *directStore) VectorMVHybridSearchExt(_ context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_mv_hybrid_search",
		ops.EncodeMVHybridArgs(name, query, sparseQ, k, toMVHybridVectorOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, err
	}
	// The fused handler returns score-carrying hybrid results (id+distance+score).
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// VectorMVQuery runs the MV-collection Query API on the single direct shard. A
// Direct store is non-clustered (one shard, no partition catalog), so the result is
// never degraded. The per-shard handler returns a MODE-TAGGED payload: RERANK fills
// Fused (the reranked top-k); FUSION fills Lanes (the UNFUSED prefetch lanes — the
// wire never carries the engine's locally-fused list). Route the one shard's result
// through the SAME merge the embedded single-shard path uses so FUSION actually fuses
// (reading qr.Fused directly would drop every FUSION result). The MV analogue of
// VectorNamedQuery.
func (d *directStore) VectorMVQuery(_ context.Context, name string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	body, err := d.Call(context.Background(), "vector_mv_query",
		ops.EncodeQueryArgs(name, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, err := ops.DecodeQueryResult(body)
	if err != nil {
		return nil, FanMeta{}, err
	}
	res, err := mergeSingleShardQuery(qr, spec)
	if err != nil {
		return nil, FanMeta{}, err
	}
	return res, FanMeta{}, nil
}

func (d *directStore) VectorNamedSearchDocs(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorDocument, error) {
	return d.VectorNamedSearchDocsExt(ctx, name, vectorName, query, k, NamedSearchOpts{Filter: filter})
}

func (d *directStore) VectorNamedSearchDocsExt(_ context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorDocument, error) {
	body, err := d.Call(context.Background(), "vector_named_search_docs",
		ops.EncodeNamedSearchArgsOpts(name, vectorName, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorDocs(body)
}

func (d *directStore) VectorNamedDelete(_ context.Context, name string, id uint64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_named_delete", ops.EncodeNamedDeleteArgs(name, id))
	if err != nil {
		return false, err
	}
	return len(body) > 0 && body[0] == 1, nil
}

func (d *directStore) VectorNamedScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) ([]VectorDocument, string, error) {
	return d.VectorNamedScrollExt(ctx, name, filter, limit, cursor, NamedScrollOpts{})
}

func (d *directStore) VectorNamedScrollExt(_ context.Context, name string, filter VectorFilter, limit int, cursor string, opts NamedScrollOpts) ([]VectorDocument, string, error) {
	dec, err := ops.DecodeScrollCursorTyped(cursor)
	if err != nil {
		return nil, "", err
	}
	var order *ops.ScrollOrder
	var afterID uint64
	var hasAfter bool
	if opts.OrderBy != nil {
		ob := opts.OrderBy
		var verr error
		order, afterID, hasAfter, verr = buildScrollOrder(ob, dec)
		if verr != nil {
			return nil, "", verr
		}
	} else {
		if dec.Present && dec.Version != 1 {
			return nil, "", ops.ErrCursorOrderMismatch
		}
		afterID, hasAfter = dec.LastID, dec.Present
	}
	body, err := d.Call(context.Background(), "vector_named_scroll",
		ops.EncodeNamedScrollArgsOrderBounded(name, filter, limit, afterID, hasAfter, opts.ReadConsistency, opts.OnPartitionUnavailable, order, opts.MaxStaleness))
	if err != nil {
		return nil, "", err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, "", err
	}
	return docs, scrollNextCursorOrder(docs, limit, opts.OrderBy), nil
}

func (d *directStore) VectorMVScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) ([]VectorDocument, FanMeta, string, error) {
	return d.VectorMVScrollExt(ctx, name, filter, limit, cursor, MVScrollOpts{})
}

func (d *directStore) VectorMVScrollExt(_ context.Context, name string, filter VectorFilter, limit int, cursor string, opts MVScrollOpts) ([]VectorDocument, FanMeta, string, error) {
	dec, err := ops.DecodeScrollCursorTyped(cursor)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	var order *ops.ScrollOrder
	var afterID uint64
	var hasAfter bool
	if opts.OrderBy != nil {
		ob := opts.OrderBy
		var verr error
		order, afterID, hasAfter, verr = buildScrollOrder(ob, dec)
		if verr != nil {
			return nil, FanMeta{}, "", verr
		}
	} else {
		if dec.Present && dec.Version != 1 {
			return nil, FanMeta{}, "", ops.ErrCursorOrderMismatch
		}
		afterID, hasAfter = dec.LastID, dec.Present
	}
	// Direct stores have a single shard (no fan-out), so the cursor goes straight
	// to the handler; next_cursor follows the shared full-page rule.
	body, err := d.Call(context.Background(), "vector_mv_scroll",
		ops.EncodeMVScrollArgsOrderBounded(name, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, order, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	return docs, FanMeta{}, scrollNextCursorOrder(docs, limit, opts.OrderBy), nil
}

func (d *directStore) VectorNamedGetConfig(ctx context.Context, name string) (map[string]NamedVectorParams, error) {
	return d.VectorNamedGetConfigExt(ctx, name, ReadOpts{})
}

// VectorNamedGetConfigExt threads the rc opts into the get_config args; see
// VectorGetExt (the barrier is a no-op on the single-node directStore).
func (d *directStore) VectorNamedGetConfigExt(_ context.Context, name string, opts ReadOpts) (map[string]NamedVectorParams, error) {
	body, err := d.Call(context.Background(), "vector_named_get_config", ops.EncodeNamedNameArgsOpts(name, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, err
	}
	return ops.DecodeNamedConfigResult(body)
}

func (d *directStore) VectorGet(ctx context.Context, collection string, id uint64, withVector, withPayload bool) (bool, []float32, VectorMetadata, time.Duration, *VectorSparse, error) {
	return d.VectorGetExt(ctx, collection, id, withVector, withPayload, ReadOpts{})
}

// VectorGetExt threads the rc opts into the get args. directStore is single-node
// with NO Raft shard, so the readIndex barrier is a no-op at runtime; the rc byte
// rides the wire harmlessly (byte-identical to VectorGet when rc==0) so the public
// surface is symmetric with the clustered embedded backend.
func (d *directStore) VectorGetExt(_ context.Context, collection string, id uint64, withVector, withPayload bool, opts ReadOpts) (bool, []float32, VectorMetadata, time.Duration, *VectorSparse, error) {
	body, err := d.Call(context.Background(), "vector_get", ops.EncodeVectorGetArgsOpts(collection, id, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return false, nil, nil, 0, nil, err
	}
	return ops.DecodeVectorGetResult(body)
}

// VectorGetBatch runs one vector_get_batch op against the single in-process
// shard (directStore is single-node / unpartitioned, so there is no scatter)
// and splits the rows into points + missing. Duplicate ids are deduped up front.
func (d *directStore) VectorGetBatch(_ context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]BatchGetPoint, []uint64, error) {
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}
	body, err := d.Call(context.Background(), "vector_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, getFlags(withVector, withPayload)))
	if err != nil {
		return nil, nil, err
	}
	rows, err := ops.DecodeVectorGetBatchResult(body)
	if err != nil {
		return nil, nil, err
	}
	return splitBatchRows(rows)
}

func (d *directStore) VectorSetPayload(_ context.Context, collection string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_set_payload", ops.EncodeSetPayloadArgsOpts(collection, id, patch, keyTTLMs))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorOverwritePayload(_ context.Context, collection string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_overwrite_payload", ops.EncodeSetPayloadArgsOpts(collection, id, meta, keyTTLMs))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorDeletePayloadKeys(_ context.Context, collection string, id uint64, keys []string, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs(collection, id, keys))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorClearPayload(_ context.Context, collection string, id uint64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_clear_payload", ops.EncodeClearPayloadArgs(collection, id))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorNamedGet(ctx context.Context, name string, id uint64, withVector, withPayload bool) (bool, map[string][]float32, VectorMetadata, time.Duration, error) {
	return d.VectorNamedGetExt(ctx, name, id, withVector, withPayload, ReadOpts{})
}

// VectorNamedGetExt threads the rc opts into the get args; see VectorGetExt (the
// barrier is a no-op on the single-node directStore).
func (d *directStore) VectorNamedGetExt(_ context.Context, name string, id uint64, withVector, withPayload bool, opts ReadOpts) (bool, map[string][]float32, VectorMetadata, time.Duration, error) {
	body, err := d.Call(context.Background(), "vector_named_get", ops.EncodeVectorGetArgsOpts(name, id, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return false, nil, nil, 0, err
	}
	return ops.DecodeNamedGetResult(body)
}

// VectorNamedGetBatch runs one vector_named_get_batch op against the single
// in-process shard (directStore is single-node / unpartitioned, so there is no
// scatter) and splits the rows into points + missing. Duplicate ids are deduped
// up front. The named clone of VectorGetBatch.
func (d *directStore) VectorNamedGetBatch(_ context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]NamedBatchGetPoint, []uint64, error) {
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}
	body, err := d.Call(context.Background(), "vector_named_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, getFlags(withVector, withPayload)))
	if err != nil {
		return nil, nil, err
	}
	rows, err := ops.DecodeNamedGetBatchResult(body)
	if err != nil {
		return nil, nil, err
	}
	return splitNamedBatchRows(rows)
}

func (d *directStore) VectorNamedSetPayload(_ context.Context, name string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_named_set_payload", ops.EncodeSetPayloadArgsOpts(name, id, patch, keyTTLMs))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorNamedOverwritePayload(_ context.Context, name string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_named_overwrite_payload", ops.EncodeSetPayloadArgsOpts(name, id, meta, keyTTLMs))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorNamedDeletePayloadKeys(_ context.Context, name string, id uint64, keys []string, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_named_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs(name, id, keys))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorNamedClearPayload(_ context.Context, name string, id uint64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_named_clear_payload", ops.EncodeClearPayloadArgs(name, id))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorMVGet(ctx context.Context, name string, docID uint64, withVector, withPayload bool) (bool, [][]float32, VectorMetadata, error) {
	return d.VectorMVGetExt(ctx, name, docID, withVector, withPayload, ReadOpts{})
}

// VectorMVGetExt threads the rc opts into the get args; see VectorGetExt (the
// barrier is a no-op on the single-node directStore).
func (d *directStore) VectorMVGetExt(_ context.Context, name string, docID uint64, withVector, withPayload bool, opts ReadOpts) (bool, [][]float32, VectorMetadata, error) {
	body, err := d.Call(context.Background(), "vector_mv_get", ops.EncodeVectorGetArgsOpts(name, docID, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return false, nil, nil, err
	}
	return ops.DecodeMVGetResult(body)
}

// VectorMVGetBatch runs one vector_mv_get_batch op against the single in-process
// shard (directStore is single-node / unpartitioned, so there is no scatter) and
// splits the rows into points + missing. Duplicate ids are deduped up front. MV
// has NO ttl. The MV clone of VectorNamedGetBatch.
func (d *directStore) VectorMVGetBatch(_ context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]MVBatchGetPoint, []uint64, error) {
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}
	body, err := d.Call(context.Background(), "vector_mv_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, getFlags(withVector, withPayload)))
	if err != nil {
		return nil, nil, err
	}
	rows, err := ops.DecodeMVGetBatchResult(body)
	if err != nil {
		return nil, nil, err
	}
	return splitMVBatchRows(rows)
}

func (d *directStore) VectorMVSetPayload(_ context.Context, name string, docID uint64, patch VectorMetadata, keyTTLMs map[string]int64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_set_payload", ops.EncodeSetPayloadArgsOpts(name, docID, patch, keyTTLMs))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorMVOverwritePayload(_ context.Context, name string, docID uint64, meta VectorMetadata, keyTTLMs map[string]int64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_overwrite_payload", ops.EncodeSetPayloadArgsOpts(name, docID, meta, keyTTLMs))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorMVDeletePayloadKeys(_ context.Context, name string, docID uint64, keys []string, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs(name, docID, keys))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorMVClearPayload(_ context.Context, name string, docID uint64, _ ...WriteOpts) (bool, error) {
	body, err := d.Call(context.Background(), "vector_mv_clear_payload", ops.EncodeClearPayloadArgs(name, docID))
	if err != nil {
		return false, err
	}
	return ops.DecodePayloadResult(body)
}

func (d *directStore) VectorScroll(_ context.Context, collection string, filter VectorFilter, limit int, opts VectorScrollOpts) ([]VectorDocument, FanMeta, string, error) {
	// Decode the cursor TYPED so the order_by (v2) and id-scroll (v1) paths both
	// validate the cursor version against opts.OrderBy before dispatch.
	dec, err := ops.DecodeScrollCursorTyped(opts.Cursor)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	var order *ops.ScrollOrder
	var afterID uint64
	var hasAfter bool
	if opts.OrderBy != nil {
		ob := opts.OrderBy
		var verr error
		order, afterID, hasAfter, verr = buildScrollOrder(ob, dec)
		if verr != nil {
			return nil, FanMeta{}, "", verr
		}
	} else {
		if dec.Present && dec.Version != 1 {
			return nil, FanMeta{}, "", ops.ErrCursorOrderMismatch
		}
		afterID, hasAfter = dec.LastID, dec.Present
	}
	// Direct stores have a single shard (no fan-out), so the cursor goes straight
	// to the handler; next_cursor follows the shared full-page rule.
	body, err := d.Call(context.Background(), "vector_scroll",
		ops.EncodeScrollArgsOrderBounded(collection, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, order, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	return docs, FanMeta{}, scrollNextCursorOrder(docs, limit, opts.OrderBy), nil
}

// VectorResplit is unsupported on Direct stores: offline generational repartition
// is an embedded-orchestration primitive that depends on the partition catalog,
// which Direct mode (single in-process engine, no catalog) does not have.
func (d *directStore) VectorResplit(_ context.Context, _ string, _ int) error {
	return errors.New("rostam: VectorResplit not supported on a Direct store (no partition catalog)")
}

// VectorResplitCleanup is unsupported on Direct stores: orphan-partition cleanup is an
// embedded-orchestration primitive that probes/drops physical generations via the
// partition catalog, which Direct mode (single in-process engine, no catalog) lacks.
func (d *directStore) VectorResplitCleanup(_ context.Context, _ string) (int, error) {
	return 0, errors.New("rostam: VectorResplitCleanup not supported on a Direct store (no partition catalog)")
}

// VectorMVResplit is unsupported on Direct stores: offline generational repartition
// is an embedded-orchestration primitive that depends on the partition catalog,
// which Direct mode (single in-process engine, no catalog) does not have.
func (d *directStore) VectorMVResplit(_ context.Context, _ string, _ int) error {
	return errors.New("rostam: VectorMVResplit not supported on a Direct store (no partition catalog)")
}

// VectorMVResplitCleanup is unsupported on Direct stores: orphan-partition cleanup is
// an embedded-orchestration primitive that probes/drops physical generations via the
// partition catalog, which Direct mode (single in-process engine, no catalog) lacks.
func (d *directStore) VectorMVResplitCleanup(_ context.Context, _ string) (int, error) {
	return 0, errors.New("rostam: VectorMVResplitCleanup not supported on a Direct store (no partition catalog)")
}

// VectorReshard is unsupported on Direct stores: the online reshard orchestrator
// depends on the partition catalog (reshard-state + generation routing) and
// dual-write fan-out, neither of which Direct mode (single in-process engine, no
// catalog) has.
func (d *directStore) VectorReshard(_ context.Context, _ string, _ int) error {
	return errors.New("rostam: VectorReshard not supported on a Direct store (no partition catalog)")
}

// VectorReshardAbort is unsupported on Direct stores for the same reason as
// VectorReshard: there is no partition catalog to hold reshard state.
func (d *directStore) VectorReshardAbort(_ context.Context, _ string) error {
	return errors.New("rostam: VectorReshardAbort not supported on a Direct store (no partition catalog)")
}

// VectorMVReshard is unsupported on Direct stores for the same reason as
// VectorReshard: the online MV reshard orchestrator depends on the partition
// catalog and dual-write fan-out, neither of which Direct mode has.
func (d *directStore) VectorMVReshard(_ context.Context, _ string, _ int) error {
	return errors.New("rostam: VectorMVReshard not supported on a Direct store (no partition catalog)")
}

// VectorMVReshardAbort is unsupported on Direct stores for the same reason as
// VectorMVReshard: there is no partition catalog to hold reshard state.
func (d *directStore) VectorMVReshardAbort(_ context.Context, _ string) error {
	return errors.New("rostam: VectorMVReshardAbort not supported on a Direct store (no partition catalog)")
}

// CreateAlias is unsupported on Direct stores: alias management is a coordinator
// op backed by the meta-Raft alias catalog, which Direct mode (single in-process
// engine, no catalog) does not have.
func (d *directStore) CreateAlias(_ context.Context, _, _ string) error {
	return errors.New("rostam: CreateAlias not supported on a Direct store (no alias catalog)")
}

// DeleteAlias is unsupported on Direct stores for the same reason as CreateAlias.
func (d *directStore) DeleteAlias(_ context.Context, _ string) error {
	return errors.New("rostam: DeleteAlias not supported on a Direct store (no alias catalog)")
}

// AliasBatch is unsupported on Direct stores for the same reason as CreateAlias.
func (d *directStore) AliasBatch(_ context.Context, _ []AliasAction) error {
	return errors.New("rostam: AliasBatch not supported on a Direct store (no alias catalog)")
}

// ListAliases is unsupported on Direct stores for the same reason as CreateAlias.
func (d *directStore) ListAliases(_ context.Context, _ string) (map[string]string, error) {
	return nil, errors.New("rostam: ListAliases not supported on a Direct store (no alias catalog)")
}

// AsDispatcher returns a value that satisfies server.Dispatcher
// (`Call(name, args) → ([]byte, error)` and `LeaderAddr() string`) so a
// Direct-backed Store can sit behind the TCP server. Useful for the
// "single-host, multi-process" deployment where you want network access
// to a cache that doesn't need replication.
//
// Direct mode has no concept of leadership; LeaderAddr() returns "".
func (d *directStore) AsDispatcher() *directDispatcher {
	return &directDispatcher{store: d}
}

type directDispatcher struct {
	store *directStore
}

// Call routes a transport request into the Direct store. The unified Query ops are
// intercepted for the SAME reason the cluster fanoutDispatcher intercepts them
// (fanQuery): the per-shard vector_query handler returns a MODE-TAGGED payload whose
// FUSION variant carries the UNFUSED prefetch lanes (mode byte 0), never a flat
// top-k — but every network decoder (DecodeQueryResultDegraded / DecodeGroupsDegraded)
// accepts only the flat fused / grouped shape. Forwarding the raw handler result would
// make the DEFAULT (FUSION) and grouped /query fail to decode on every transport
// (HTTP 500, gRPC Internal, client decode error). So route through the fusion-aware
// directStore.Vector*Query methods and re-encode the flat/grouped degraded wire shape
// the decoders require — exactly as fanQuery does on the cluster path. Every other op
// (including RERANK queries, whose handler already returns the flat shape) passes
// straight through.
func (a *directDispatcher) Call(name string, args []byte) ([]byte, error) {
	switch name {
	case "vector_query":
		return a.fanQuery(args)
	case "vector_named_query":
		return a.flatQuery(args, a.store.VectorNamedQuery)
	case "vector_mv_query":
		return a.flatQuery(args, a.store.VectorMVQuery)
	default:
		return a.store.Call(context.Background(), name, args)
	}
}

// fanQuery mirrors fanoutDispatcher.fanQuery for the single Direct shard: a GROUPED
// spec routes through VectorQueryGrouped + EncodeGroupsDegraded; a flat spec routes
// through VectorQuery + EncodeQueryResultFusedDegraded. A Direct store never fans out,
// so FanMeta is always zero, but the degraded codec keeps the wire shape uniform with
// the cluster path (so the same RPC/HTTP/client decoders work against either backend).
func (a *directDispatcher) fanQuery(args []byte) ([]byte, error) {
	coll, specBytes, spec, rc, opa, bound, err := ops.DecodeQuerySpecArgs(args)
	if err != nil {
		return nil, err
	}
	opts := ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound}
	if spec.GroupBy != "" {
		groups, meta, gerr := a.store.VectorQueryGrouped(context.Background(), coll, specBytes, spec, opts)
		if gerr != nil {
			return nil, gerr
		}
		return ops.EncodeGroupsDegraded(groups, meta.Degraded, missingU16(meta.Missing)), nil
	}
	res, meta, err := a.store.VectorQuery(context.Background(), coll, specBytes, spec, opts)
	if err != nil {
		return nil, err
	}
	return ops.EncodeQueryResultFusedDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// flatQuery is the flat-only Query chokepoint for the named / MV families (which have
// no grouped variant, matching the cluster fanNamedQuery / fanMVQuery): route through
// the given fusion-aware method and re-encode the flat degraded wire shape.
func (a *directDispatcher) flatQuery(args []byte, run func(context.Context, string, []byte, vector.QuerySpec, ReadOpts) ([]VectorResult, FanMeta, error)) ([]byte, error) {
	coll, specBytes, spec, rc, opa, bound, err := ops.DecodeQuerySpecArgs(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := run(context.Background(), coll, specBytes, spec,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeQueryResultFusedDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (a *directDispatcher) LeaderAddr() string { return "" }

// DirectServer is a TCP server backed by a no-Raft Direct store. Use it
// when you want a Rostam cache reachable over the network without
// paying for replication — e.g. a per-host cache process that
// application code connects to via NewClient.
type DirectServer struct {
	store *directStore
	srv   *server.Server
}

// NewDirectServer constructs a Direct-backed cache and binds a TCP
// server to addr. Use "127.0.0.1:0" to get an OS-assigned port; read it
// back via Addr().
//
// Callers connect with NewClient pointing at the returned Addr().
// Close stops the listener, drains in-flight requests, and closes the
// cache (flushing the mmap when Durable is set).
func NewDirectServer(addr string, cfg DirectConfig) (*DirectServer, error) {
	s, err := NewDirect(cfg)
	if err != nil {
		return nil, err
	}
	store, ok := s.(*directStore)
	if !ok {
		_ = s.Close()
		return nil, errors.New("rostam: NewDirect returned unexpected type")
	}
	srv, err := server.New(server.Config{
		Addr:          addr,
		Dispatcher:    store.AsDispatcher(),
		Authenticator: cfg.Authenticator,
	})
	if err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("rostam: server.New: %w", err)
	}
	go func() { _ = srv.Serve() }()
	return &DirectServer{store: store, srv: srv}, nil
}

// Addr returns the bound TCP address (useful when addr was ":0").
func (s *DirectServer) Addr() string {
	if s.srv == nil {
		return ""
	}
	return s.srv.Addr().String()
}

// Close stops the TCP server and the underlying Direct store. Idempotent.
func (s *DirectServer) Close() error {
	var errs []error
	if s.srv != nil {
		if err := s.srv.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if s.store != nil {
		if err := s.store.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
