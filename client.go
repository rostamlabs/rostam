// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/wire"
	"github.com/rostamlabs/rostam/vector"
)

// ClientConfig configures a networked Rostam client.
type ClientConfig struct {
	// Servers is the initial bootstrap list of "host:port" entries.
	// Smart-client topology refresh discovers the rest. Required.
	Servers []string

	// Ops is the caller's op registry mirror. The client uses it for
	// KeyExtractor-based routing — without it, ops fall back to
	// round-robin. Required if any custom op needs per-key routing.
	Ops *ops.Registry

	// MaxConnsPerServer caps the per-server connection pool. Default 8.
	MaxConnsPerServer int

	// MaxNotLeaderHops caps retries on stale topology. Default 5.
	MaxNotLeaderHops int

	// TopologyRefreshInterval polls cluster topology. Default 5s.
	TopologyRefreshInterval time.Duration

	// AuthToken, when non-empty, is sent on every RPC via a protocol-v2 frame
	// prefix. The server's Authenticator hook validates it. Leave empty for
	// legacy (no-auth) deployments — that keeps the wire format on v1.
	AuthToken string

	// TLSConfig, when non-nil, dials every server over TLS instead of plaintext.
	// Build it via tlsutil.ClientTLS(caFile, certFile, keyFile, serverName): set
	// RootCAs to verify the server, and (for mTLS) a client cert/key — the cert CN
	// then becomes the principal when no AuthToken is set. nil ⇒ plaintext default.
	TLSConfig *tls.Config
}

// NewClient constructs a networked Rostam client and returns a Store
// backed by it.
func NewClient(cfg ClientConfig) (Store, error) {
	if len(cfg.Servers) == 0 {
		return nil, errors.New("rostam: ClientConfig.Servers is required")
	}
	if cfg.MaxConnsPerServer == 0 {
		cfg.MaxConnsPerServer = 8
	}
	if cfg.MaxNotLeaderHops == 0 {
		cfg.MaxNotLeaderHops = 5
	}
	if cfg.TopologyRefreshInterval == 0 {
		cfg.TopologyRefreshInterval = 5 * time.Second
	}
	// client.Config.Ops is a *wire.Registry (the client package cannot import
	// ops — that is what would drag cache/objstore/vector-analysis into the
	// networked client). cfg.Ops here is the caller's full server-side
	// *ops.Registry (routing + handlers); ExportRouting adapts it into the
	// routing-only registry the low-level client needs.
	var wireOps *wire.Registry
	if cfg.Ops != nil {
		wireOps = wire.NewRegistry()
		if err := cfg.Ops.ExportRouting(wireOps); err != nil {
			return nil, fmt.Errorf("rostam: export routing registry: %w", err)
		}
	}
	cc := client.Config{
		Servers:                 cfg.Servers,
		Ops:                     wireOps,
		MaxConnsPerServer:       int32(cfg.MaxConnsPerServer), //nolint:gosec // user-supplied, positive
		MaxNotLeaderHops:        cfg.MaxNotLeaderHops,
		TopologyRefreshInterval: cfg.TopologyRefreshInterval,
		AuthToken:               cfg.AuthToken,
		TLSConfig:               cfg.TLSConfig,
	}
	c, err := client.New(cc)
	if err != nil {
		return nil, fmt.Errorf("rostam: client.New: %w", err)
	}
	return &networkedStore{c: c}, nil
}

type networkedStore struct {
	c *client.Client
}

// kvArgsPool recycles the small request-args buffer shared by the KV ops
// (get/del key args, and put args for small values), so Get/GetInto/Del/Put
// allocate nothing on the request side in steady state. neither the inline
// doCall framing nor the pipelined encodeRequestFrame retains args past the
// call, so the buffer is safe to return afterward. Buffers grown past
// maxPooledKVArgs (a large Put value) are dropped rather than pooled, to bound
// retained memory.
const maxPooledKVArgs = 4 << 10

var kvArgsPool = sync.Pool{New: func() any { b := make([]byte, 0, 128); return &b }}

func putKVArgs(bp *[]byte) {
	if cap(*bp) <= maxPooledKVArgs {
		kvArgsPool.Put(bp)
	}
}

func (n *networkedStore) Get(ctx context.Context, key []byte) ([]byte, error) {
	bp := kvArgsPool.Get().(*[]byte)
	*bp = ops.AppendKeyArgs((*bp)[:0], key)
	raw, err := n.c.Call(ctx, "get", *bp)
	putKVArgs(bp)
	if err != nil {
		return nil, mapErr(err)
	}
	return raw, nil
}

// GetInto is the allocation-light Get: the value is copied into dst inside the
// zero-copy CallFunc callback (no defensive payload copy) and the request args
// are pooled, so a reused dst yields a zero-allocation read. Returns ErrNotFound
// (via mapErr) when the key is absent; the returned slice may alias dst.
func (n *networkedStore) GetInto(ctx context.Context, key, dst []byte) ([]byte, error) {
	bp := kvArgsPool.Get().(*[]byte)
	*bp = ops.AppendKeyArgs((*bp)[:0], key)
	var out []byte
	err := n.c.CallFunc(ctx, "get", *bp, func(payload []byte) error {
		out = append(dst[:0], payload...)
		return nil
	})
	putKVArgs(bp)
	if err != nil {
		return nil, mapErr(err)
	}
	return out, nil
}

func (n *networkedStore) Put(ctx context.Context, key, value []byte, ttl time.Duration) error {
	bp := kvArgsPool.Get().(*[]byte)
	*bp = ops.AppendPutArgs((*bp)[:0], key, value, ttl)
	_, err := n.c.Call(ctx, "put", *bp)
	putKVArgs(bp)
	return mapErr(err)
}

// PutBatch writes many key/value pairs, delegating to the client's shard-grouping
// bulk path so each shard takes one Raft log entry per chunk. ErrNotLeader
// propagates identically to Put.
func (n *networkedStore) PutBatch(ctx context.Context, entries []ops.PutEntry) error {
	return mapErr(n.c.PutBatch(ctx, entries))
}

func (n *networkedStore) Del(ctx context.Context, key []byte) (bool, error) {
	bp := kvArgsPool.Get().(*[]byte)
	*bp = ops.AppendKeyArgs((*bp)[:0], key)
	raw, err := n.c.Call(ctx, "del", *bp)
	putKVArgs(bp)
	if err != nil {
		return false, mapErr(err)
	}
	return len(raw) == 1 && raw[0] == 1, nil
}

func (n *networkedStore) Call(ctx context.Context, op string, args []byte) ([]byte, error) {
	raw, err := n.c.Call(ctx, op, args)
	if err != nil {
		return nil, mapErr(err)
	}
	return raw, nil
}

// IsLeader reports whether the topology cache has a known leader for
// key's shard. In Client mode the local process is never the leader;
// "true" here means "a write to key would route somewhere definite".
func (n *networkedStore) IsLeader(key []byte) bool {
	return n.leaderAddrFor(key) != ""
}

func (n *networkedStore) LeaderAddr(key []byte) string {
	return n.leaderAddrFor(key)
}

func (n *networkedStore) leaderAddrFor(key []byte) string {
	t := n.c.Topology()
	if t == nil || t.NumShards == 0 || len(t.Leaders) != t.NumShards {
		return ""
	}
	idx := xxhash.Sum64(key) % uint64(t.NumShards) //nolint:gosec // NumShards bounded
	return t.Leaders[idx]
}

func (n *networkedStore) Close() error {
	return n.c.Close()
}

// wcCall is the write-consistency wrapper for the networked write methods. When
// the supplied opts are INACTIVE (the default: WCF unset, wait=true — or no opts
// passed at all) it issues the plain op exactly as before — BYTE-IDENTICAL to the
// pre-feature client, so the default write path is unchanged and never builds an
// envelope. Only when opts.wcActive() does it wrap the op in the __wc__ envelope
// (ops.EncodeWCEnvelope) so the server's fanoutDispatcher intercepts it, dispatches
// the inner write through the normal routing/Raft path, and runs the post-commit
// barrier. The inner op name + args are byte-identical inside the envelope, so the
// server-side Raft log entry and every data-op decoder are untouched.
func (n *networkedStore) wcCall(ctx context.Context, opName string, innerArgs []byte, opts WriteOpts) ([]byte, error) {
	wireOp, wireArgs := wcWire(opName, innerArgs, opts)
	return n.c.Call(ctx, wireOp, wireArgs)
}

// wcWire is the pure write-consistency wire decision shared by every networked
// write method (extracted from wcCall so it is unit-testable without a live
// transport). When opts are INACTIVE it returns the plain op name + its
// byte-identical args (no envelope — the default path is unchanged). When active
// it returns (ops.WCEnvelopeOp, EncodeWCEnvelope(...)) wrapping the byte-identical
// inner op. The wait byte is 1 for the default/explicit-true and 0 for the
// explicit wait=false latency knob, matching the envelope handler's wait!=0 decode.
func wcWire(opName string, innerArgs []byte, opts WriteOpts) (string, []byte) {
	if !opts.wcActive() {
		return opName, innerArgs
	}
	wait := uint8(0)
	if opts.waitValue() {
		wait = 1
	}
	return ops.WCEnvelopeOp, ops.EncodeWCEnvelope(opts.WriteConsistencyFactor, wait, opName, innerArgs)
}

func (n *networkedStore) VectorInsert(ctx context.Context, collection string, id uint64, vec []float32, opts ...WriteOpts) error {
	_, err := n.wcCall(ctx, "vector_insert", ops.EncodeVectorInsertArgs(collection, id, vec), firstWriteOpts(opts))
	return mapErr(err)
}

func (n *networkedStore) VectorSearch(ctx context.Context, collection string, query []float32, k int) ([]VectorResult, error) {
	body, err := n.c.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs(collection, k, query))
	if err != nil {
		return nil, mapErr(err)
	}
	return ops.DecodeVectorSearchResults(body)
}

// vectorSearchArgsPool recycles the request-args buffer for VectorSearchInto so
// the hot search path allocates nothing on the client: the buffer is filled by
// AppendVectorSearchArgs, consumed by CallFunc (doCall writes+flushes it before
// returning), then returned. Initial cap fits a ~128-dim query without growing.
var vectorSearchArgsPool = sync.Pool{New: func() any { b := make([]byte, 0, 576); return &b }}

// VectorSearchInto sends the search over the zero-copy CallFunc path: the wire
// response is decoded straight into dst inside the callback (no defensive copy
// of the payload, and no result-slice allocation when dst is reused). The
// request-args buffer is pooled, so a reused dst yields a zero-allocation
// client round-trip.
func (n *networkedStore) VectorSearchInto(ctx context.Context, collection string, query []float32, k int, dst []VectorResult) ([]VectorResult, error) {
	bp := vectorSearchArgsPool.Get().(*[]byte)
	*bp = ops.AppendVectorSearchArgs((*bp)[:0], collection, k, query)
	var out []VectorResult
	err := n.c.CallFunc(ctx, "vector_search", *bp, func(payload []byte) error {
		var derr error
		out, derr = ops.DecodeVectorSearchResultsInto(payload, dst)
		return derr
	})
	vectorSearchArgsPool.Put(bp)
	if err != nil {
		return nil, mapErr(err)
	}
	return out, nil
}

func (n *networkedStore) VectorDelete(ctx context.Context, collection string, id uint64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_delete", ops.EncodeVectorDeleteArgs(collection, id), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return len(body) == 1 && body[0] == 1, nil
}

// VectorInsertIfAbsent and VectorExists are ordinary collection-routed ops over
// the wire (same dispatch as vector_insert / vector_delete) — no virtual-op
// plumbing needed. The server shard runs the atomic engine primitive; the call
// targets the LOGICAL collection (partition routing for resharded collections is
// the embedded coordinator's job, not the thin networked client's).
func (n *networkedStore) VectorInsertIfAbsent(ctx context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) (bool, error) {
	body, err := n.c.Call(ctx, "vector_insert_if_absent", ops.EncodeVectorInsertArgsExt(collection, id, vec, opts.TTL, opts.Metadata, opts.Sparse))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodeIfAbsentResult(body)
}

func (n *networkedStore) VectorExists(ctx context.Context, collection string, id uint64) (bool, error) {
	body, err := n.c.Call(ctx, "vector_exists", ops.EncodeExistsArgs(collection, id))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodeExistsResult(body)
}

func (n *networkedStore) CreateCollection(ctx context.Context, name string, cfg VectorConfig) error {
	if strings.ContainsAny(name, "#@") {
		return fmt.Errorf("vector: collection name %q must not contain reserved characters '#' or '@'", name)
	}
	_, err := n.c.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(name, cfg))
	return mapErr(err)
}

func (n *networkedStore) VectorInsertExt(ctx context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) error {
	_, err := n.wcCall(ctx, "vector_insert", ops.EncodeVectorInsertArgsKeyTTL(collection, id, vec, opts.TTL, opts.Metadata, opts.Sparse, opts.KeyTTLMs), opts.WriteOpts)
	return mapErr(err)
}

// missingInt converts the degraded wire trailer's partition indices ([]uint16)
// to the []int FanMeta.Missing carries. nil in → nil out (no partitions missing).
func missingInt(missing []uint16) []int {
	if len(missing) == 0 {
		return nil
	}
	out := make([]int, len(missing))
	for i, p := range missing {
		out[i] = int(p)
	}
	return out
}

func (n *networkedStore) VectorSearchExt(ctx context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_search",
		ops.EncodeVectorSearchArgsOpts(collection, k, query, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeVectorSearchResultsDegraded(body)
	return res, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

func (n *networkedStore) VectorHybridSearch(ctx context.Context, collection string, dense []float32, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_hybrid_search",
		ops.EncodeHybridSearchArgsOpts(collection, dense, k, opts.Sparse, toVectorHybridOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	return res, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

// VectorQuery runs the unified Query API through the networked transport. The
// server's fan-out dispatcher intercepts vector_query, fans to all partitions on
// a partitioned collection (queryFanOut), and re-encodes the merged top-k as a
// flat mode-tagged result WITH the degraded/missing trailer (P=1 returns the
// same flat shape from the per-shard handler); DecodeQueryResultDegraded reads
// the top-k plus the trailer so degraded/missing flow into FanMeta — matching
// VectorHybridSearch's DecodeHybridResultsDegraded. The dedicated gRPC
// VectorQuery RPC dispatches into this same op + result wire.
func (n *networkedStore) VectorQuery(ctx context.Context, collection string, specBytes []byte, _ vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_query",
		ops.EncodeQueryArgs(collection, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	out := make([]VectorResult, len(res))
	for i, r := range res {
		out[i] = VectorResult(r)
	}
	return out, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, nil
}

// VectorQueryGrouped is the GROUPED Query API client path: the same vector_query op
// (the spec carries group_by/group_size), but the dispatcher (fanQuery) groups the
// global ordered pool and returns the groups + degraded/missing trailer, decoded here
// via DecodeGroupsDegraded — mirroring VectorSearchGroups. The dedicated gRPC
// VectorQuery RPC dispatches into this same op + grouped result wire when group_by is set.
func (n *networkedStore) VectorQueryGrouped(ctx context.Context, collection string, specBytes []byte, _ vector.QuerySpec, opts ReadOpts) ([]VectorGroup, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_query",
		ops.EncodeQueryArgs(collection, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	groups, degraded, missing, err := ops.DecodeGroupsDegraded(body)
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	return groups, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, nil
}

func (n *networkedStore) VectorUpsert(ctx context.Context, collection string, id uint64, vec []float32, content string, opts VectorInsertOpts) error {
	_, err := n.wcCall(ctx, "vector_upsert", ops.EncodeVectorUpsertArgsCASKeyTTL(collection, id, vec, content, opts.TTL, opts.Metadata, opts.Sparse, 0, false, opts.KeyTTLMs), opts.WriteOpts)
	return mapErr(err)
}

func (n *networkedStore) VectorSearchDocs(ctx context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_search_docs",
		ops.EncodeVectorSearchArgsOpts(collection, k, query, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegraded(body)
	return docs, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

func (n *networkedStore) VectorSearchText(ctx context.Context, collection string, query string, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_search_text",
		ops.EncodeSearchTextArgsGlobal(collection, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness, opts.GlobalIDF, nil))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegraded(body)
	return docs, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

func (n *networkedStore) VectorHybridText(ctx context.Context, collection string, dense []float32, query string, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_hybrid_text",
		ops.EncodeHybridTextArgsGlobal(collection, dense, query, k, toVectorHybridOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness, opts.GlobalIDF, nil))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	return res, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

func (n *networkedStore) VectorDeleteByFilter(ctx context.Context, collection string, filter VectorFilter) (int, error) {
	body, err := n.c.Call(ctx, "vector_delete_by_filter", ops.EncodeDeleteByFilterArgs(collection, filter))
	if err != nil {
		return 0, mapErr(err)
	}
	return ops.DecodeDeleteByFilterResult(body)
}

func (n *networkedStore) VectorSearchGroups(ctx context.Context, collection string, query []float32, k int, opts VectorGroupOpts) ([]VectorGroup, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_search_groups",
		ops.EncodeGroupSearchArgsOpts(collection, k, query, opts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	groups, degraded, missing, err := ops.DecodeGroupsDegraded(body)
	return groups, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

func (n *networkedStore) VectorMVCreateCollection(ctx context.Context, name string, cfg MultiVectorConfig) error {
	if strings.ContainsAny(name, "#@") {
		return fmt.Errorf("vector: collection name %q must not contain reserved characters '#' or '@'", name)
	}
	_, err := n.c.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(name, cfg))
	return mapErr(err)
}

func (n *networkedStore) VectorMVDropCollection(ctx context.Context, name string) error {
	_, err := n.c.Call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(name, 0))
	return mapErr(err)
}

func (n *networkedStore) VectorMVAdd(ctx context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata, opts ...WriteOpts) error {
	wo := firstWriteOpts(opts)
	// keyTTL block rides AFTER the base block; the OPTIONAL doc-level sparse rides
	// LAST. Empty map + nil sparse = byte-identical to EncodeMVAddArgs (the prior wire
	// shape for this path).
	_, err := n.wcCall(ctx, "vector_mv_add", ops.EncodeMVAddArgsCASKeyTTLSparse(name, docID, tokens, meta, 0, false, wo.KeyTTLMs, wo.Sparse), wo)
	return mapErr(err)
}

func (n *networkedStore) VectorMVSearch(ctx context.Context, name string, query [][]float32, k int, opts MultiSearchOpts) ([]MultiResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_mv_search",
		ops.EncodeMVSearchArgsOpts(name, query, k, opts.CandidatesPerToken, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeMVResultsDegraded(body)
	return res, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

func (n *networkedStore) VectorMVDelete(ctx context.Context, name string, docID uint64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_mv_delete", ops.EncodeMVDeleteArgs(name, docID), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return len(body) > 0 && body[0] == 1, nil
}

func (n *networkedStore) VectorMVAddIfAbsent(ctx context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata) (bool, error) {
	body, err := n.c.Call(ctx, "vector_mv_add_if_absent", ops.EncodeMVAddArgs(name, docID, tokens, meta))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodeIfAbsentResult(body)
}

func (n *networkedStore) VectorMVExists(ctx context.Context, name string, docID uint64) (bool, error) {
	body, err := n.c.Call(ctx, "vector_mv_exists", ops.EncodeMVExistsArgs(name, docID))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodeExistsResult(body)
}

func (n *networkedStore) VectorNamedCreateCollection(ctx context.Context, name string, cfg map[string]NamedVectorParams, partitions int) error {
	if strings.ContainsAny(name, "#@") {
		return fmt.Errorf("vector: collection name %q must not contain reserved characters '#' or '@'", name)
	}
	_, err := n.c.Call(ctx, "vector_named_create_collection", ops.EncodeNamedCreateArgs(name, cfg, partitions))
	return mapErr(err)
}

func (n *networkedStore) VectorNamedDropCollection(ctx context.Context, name string) error {
	_, err := n.c.Call(ctx, "vector_named_drop_collection", ops.EncodeNamedNameArgs(name))
	return mapErr(err)
}

func (n *networkedStore) VectorNamedInsert(ctx context.Context, name string, id uint64, vectors map[string][]float32, payload VectorMetadata, ttl time.Duration, opts ...WriteOpts) error {
	wo := firstWriteOpts(opts)
	// keyTTL block rides AFTER the base block; empty map = byte-identical to
	// EncodeNamedInsertArgs (the prior wire shape for this path).
	_, err := n.wcCall(ctx, "vector_named_insert", ops.EncodeNamedInsertArgsKeyTTL(name, id, vectors, payload, ttl, wo.KeyTTLMs), wo)
	return mapErr(err)
}

func (n *networkedStore) VectorNamedSearch(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorResult, error) {
	return n.VectorNamedSearchExt(ctx, name, vectorName, query, k, NamedSearchOpts{Filter: filter})
}

// VectorNamedSearchExt sends the rc/opa trailer over the wire so the remote
// coordinator's fan-out dispatcher (fanNamedSearch) decodes it and arms the
// barriers. Byte-identical to the legacy request when rc==0 && opa==0.
func (n *networkedStore) VectorNamedSearchExt(ctx context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorResult, error) {
	body, err := n.c.Call(ctx, "vector_named_search",
		ops.EncodeNamedSearchArgsOpts(name, vectorName, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, mapErr(err)
	}
	return ops.DecodeVectorSearchResults(body)
}

func (n *networkedStore) VectorNamedSparseSearch(ctx context.Context, name, space string, query VectorSparse, k int, filter VectorFilter) ([]VectorResult, error) {
	return n.VectorNamedSparseSearchExt(ctx, name, space, query, k, NamedSearchOpts{Filter: filter})
}

// VectorNamedSparseSearchExt sends the sparse query + rc/opa trailer over the wire
// so the remote coordinator's fan-out dispatcher (fanNamedSparseSearch) decodes it
// and arms the barriers. Byte-identical to the legacy request when rc==0 && opa==0.
func (n *networkedStore) VectorNamedSparseSearchExt(ctx context.Context, name, space string, query VectorSparse, k int, opts NamedSearchOpts) ([]VectorResult, error) {
	body, err := n.c.Call(ctx, "vector_named_sparse_search",
		ops.EncodeNamedSparseSearchArgsOpts(name, space, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, mapErr(err)
	}
	// The sparse handler returns score-carrying hybrid results (id+distance+score).
	return ops.DecodeHybridResults(body)
}

// VectorNamedHybridSearch sends the dense + sparse queries, the fusion opts, and
// the rc/opa trailer over the wire so the remote coordinator's fan-out dispatcher
// (fanNamedHybridSearch) decodes them, fans the lanes leaf to each partition, and
// fuses once. Byte-identical to a single-partition request when rc==0 && opa==0.
func (n *networkedStore) VectorNamedHybridSearch(ctx context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, error) {
	res, _, err := n.VectorNamedHybridSearchExt(ctx, name, denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
	return res, err
}

// VectorNamedHybridSearchExt decodes the degraded/missing trailer the coordinator's
// fanNamedHybridSearch now appends (EncodeHybridResultsDegraded), so the networked
// client surfaces partition degradation via FanMeta exactly like VectorHybridSearch.
// DecodeHybridResultsDegraded is backward-compatible with a plain (unpartitioned)
// EncodeHybridResults body, so a single-partition target reports a zero FanMeta.
func (n *networkedStore) VectorNamedHybridSearchExt(ctx context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_named_hybrid_search",
		ops.EncodeNamedHybridArgs(name, denseSpace, denseQ, sparseSpace, sparseQ, k, toNamedHybridVectorOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	// The fused handler returns score-carrying hybrid results (id+distance+score).
	res, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	return res, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

// VectorNamedQuery runs the named-collection Query API through the networked
// transport. It rides the vector_named_query op exactly as the v1 VectorQuery
// client rides vector_query (the internal cluster transport is the op-call
// protocol, not the public gRPC service — the dedicated NamedVectorQuery RPC added
// is the EXTERNAL gRPC surface, which itself dispatches the same op).
// The remote coordinator's fan-out dispatcher (fanNamedQuery) fans to all
// partitions on a partitioned collection (namedQueryFanOut) and re-encodes the
// merged top-k as a flat mode-tagged result WITH the degraded/missing trailer (P=1
// returns the same flat shape from fanNamedQuery's local merge);
// DecodeQueryResultDegraded reads the top-k plus the trailer so degraded/missing
// flow into FanMeta — mirroring the v1 VectorQuery client. The engine spec is not
// needed on the client (the coordinator decodes it from specBytes), matching
// VectorQuery's signature.
func (n *networkedStore) VectorNamedQuery(ctx context.Context, name string, specBytes []byte, _ vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_named_query",
		ops.EncodeQueryArgs(name, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	out := make([]VectorResult, len(res))
	for i, r := range res {
		out[i] = VectorResult(r)
	}
	return out, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, nil
}

// VectorMVHybridSearch sends the MV token query matrix, the sparse query, the fusion
// opts, and the rc/opa trailer over the wire so the remote coordinator's fan-out
// dispatcher (fanMVHybridSearch) decodes them, fans the lanes leaf to each partition,
// and fuses once. Byte-identical to a single-partition request when rc==0 && opa==0.
func (n *networkedStore) VectorMVHybridSearch(ctx context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, error) {
	res, _, err := n.VectorMVHybridSearchExt(ctx, name, query, sparseQ, k, opts)
	return res, err
}

// VectorMVHybridSearchExt decodes the degraded/missing trailer the coordinator's
// fanMVHybridSearch now appends (EncodeHybridResultsDegraded), so the networked
// client surfaces partition degradation via FanMeta exactly like VectorHybridSearch.
// DecodeHybridResultsDegraded is backward-compatible with a plain (unpartitioned)
// EncodeHybridResults body, so a single-partition target reports a zero FanMeta.
func (n *networkedStore) VectorMVHybridSearchExt(ctx context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_mv_hybrid_search",
		ops.EncodeMVHybridArgs(name, query, sparseQ, k, toMVHybridVectorOpts(opts), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	// The fused handler returns score-carrying hybrid results (id+distance+score).
	res, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	return res, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, err
}

// VectorMVQuery runs the MV-collection Query API through the networked transport. It
// rides the vector_mv_query op exactly as the v1 VectorQuery client rides
// vector_query (the internal cluster transport is the op-call protocol, not the
// public gRPC service — the dedicated MVVectorQuery RPC is the
// EXTERNAL gRPC surface, which itself dispatches the same op). The remote
// coordinator's fan-out dispatcher (fanMVQuery) fans to all partitions on a
// partitioned collection (mvQueryFanOut) and re-encodes the merged top-k as a flat
// mode-tagged result WITH the degraded/missing trailer (P=1 returns the same flat
// shape from fanMVQuery's local merge); DecodeQueryResultDegraded reads the top-k
// plus the trailer so degraded/missing flow into FanMeta — mirroring the v1/v2 Query
// client. The engine spec is not needed on the client (the coordinator decodes it
// from specBytes), matching VectorNamedQuery's signature.
func (n *networkedStore) VectorMVQuery(ctx context.Context, name string, specBytes []byte, _ vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	body, err := n.c.Call(ctx, "vector_mv_query",
		ops.EncodeQueryArgs(name, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	res, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return nil, FanMeta{}, mapErr(err)
	}
	out := make([]VectorResult, len(res))
	for i, r := range res {
		out[i] = VectorResult(r)
	}
	return out, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, nil
}

func (n *networkedStore) VectorNamedSearchDocs(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorDocument, error) {
	return n.VectorNamedSearchDocsExt(ctx, name, vectorName, query, k, NamedSearchOpts{Filter: filter})
}

func (n *networkedStore) VectorNamedSearchDocsExt(ctx context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorDocument, error) {
	body, err := n.c.Call(ctx, "vector_named_search_docs",
		ops.EncodeNamedSearchArgsOpts(name, vectorName, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, mapErr(err)
	}
	return ops.DecodeVectorDocs(body)
}

func (n *networkedStore) VectorNamedDelete(ctx context.Context, name string, id uint64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_named_delete", ops.EncodeNamedDeleteArgs(name, id), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return len(body) > 0 && body[0] == 1, nil
}

func (n *networkedStore) VectorNamedScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) ([]VectorDocument, string, error) {
	return n.VectorNamedScrollExt(ctx, name, filter, limit, cursor, NamedScrollOpts{})
}

func (n *networkedStore) VectorNamedScrollExt(ctx context.Context, name string, filter VectorFilter, limit int, cursor string, opts NamedScrollOpts) ([]VectorDocument, string, error) {
	afterID, hasAfter, err := ops.DecodeScrollCursor(cursor)
	if err != nil {
		return nil, "", err
	}
	body, err := n.c.Call(ctx, "vector_named_scroll",
		ops.EncodeNamedScrollArgsOptsBounded(name, filter, limit, afterID, hasAfter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, "", mapErr(err)
	}
	docs, _, _, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		return nil, "", err
	}
	// next_cursor is server-authoritative (decoded from the result wire emitted by
	// fanNamedScroll). Backward-compatible with an OLD server (nextCursor=="").
	return docs, nextCursor, nil
}

func (n *networkedStore) VectorMVScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) ([]VectorDocument, FanMeta, string, error) {
	return n.VectorMVScrollExt(ctx, name, filter, limit, cursor, MVScrollOpts{})
}

func (n *networkedStore) VectorMVScrollExt(ctx context.Context, name string, filter VectorFilter, limit int, cursor string, opts MVScrollOpts) ([]VectorDocument, FanMeta, string, error) {
	afterID, hasAfter, err := ops.DecodeScrollCursor(cursor)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	body, err := n.c.Call(ctx, "vector_mv_scroll",
		ops.EncodeMVScrollArgsOptsBounded(name, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, "", mapErr(err)
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	// next_cursor is server-authoritative: the dispatcher (fanMVScroll) emits the
	// coordinator's globally-merged cursor on the result wire, so the client just
	// decodes it. DecodeScrollResult tolerates an OLD server's cursor-less body
	// (nextCursor==""), preserving backward-compat.
	return docs, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, nextCursor, nil
}

func (n *networkedStore) VectorNamedGetConfig(ctx context.Context, name string) (map[string]NamedVectorParams, error) {
	return n.VectorNamedGetConfigExt(ctx, name, ReadOpts{})
}

// VectorNamedGetConfigExt sends the named get_config op with the rc opts trailer;
// a Linearizable read arms the server-side meta-catalog + shard barriers. Wire-
// identical to VectorNamedGetConfig when rc==0.
func (n *networkedStore) VectorNamedGetConfigExt(ctx context.Context, name string, opts ReadOpts) (map[string]NamedVectorParams, error) {
	body, err := n.c.Call(ctx, "vector_named_get_config", ops.EncodeNamedNameArgsOpts(name, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return nil, mapErr(err)
	}
	return ops.DecodeNamedConfigResult(body)
}

// Get + payload-update are point-ops-by-id over the wire (same dispatch as
// vector_delete). The call targets the LOGICAL collection; partition routing for a
// partitioned collection is the server-side fan-out coordinator's job, not
// the thin networked client's. Not-found is a FLAG in the decoded result (found=0 /
// applied=0), never a wire error.
func (n *networkedStore) VectorGet(ctx context.Context, collection string, id uint64, withVector, withPayload bool) (bool, []float32, VectorMetadata, time.Duration, *VectorSparse, error) {
	return n.VectorGetExt(ctx, collection, id, withVector, withPayload, ReadOpts{})
}

// VectorGetExt sends the get op with the rc opts trailer; the server-side fan-out
// coordinator routes the single-id point-get to the owning partition's leader and
// arms the shard readIndex barrier for a Linearizable read. Byte-identical to
// VectorGet on the wire when rc==0.
func (n *networkedStore) VectorGetExt(ctx context.Context, collection string, id uint64, withVector, withPayload bool, opts ReadOpts) (bool, []float32, VectorMetadata, time.Duration, *VectorSparse, error) {
	body, err := n.c.Call(ctx, "vector_get", ops.EncodeVectorGetArgsOpts(collection, id, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return false, nil, nil, 0, nil, mapErr(err)
	}
	return ops.DecodeVectorGetResult(body)
}

// VectorGetBatch sends one vector_get_batch op carrying all (deduped) ids; the
// server-side fan-out dispatcher scatters the ids across partitions and returns
// the merged rows, which we split into points + missing. The coordinator owns
// the scatter (see fanGetBatch), so the client stays a single round-trip.
func (n *networkedStore) VectorGetBatch(ctx context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]BatchGetPoint, []uint64, error) {
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}
	body, err := n.c.Call(ctx, "vector_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, getFlags(withVector, withPayload)))
	if err != nil {
		return nil, nil, mapErr(err)
	}
	rows, err := ops.DecodeVectorGetBatchResult(body)
	if err != nil {
		return nil, nil, err
	}
	return splitBatchRows(rows)
}

func (n *networkedStore) VectorSetPayload(ctx context.Context, collection string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_set_payload", ops.EncodeSetPayloadArgsOpts(collection, id, patch, keyTTLMs), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorOverwritePayload(ctx context.Context, collection string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_overwrite_payload", ops.EncodeSetPayloadArgsOpts(collection, id, meta, keyTTLMs), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorDeletePayloadKeys(ctx context.Context, collection string, id uint64, keys []string, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs(collection, id, keys), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorClearPayload(ctx context.Context, collection string, id uint64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_clear_payload", ops.EncodeClearPayloadArgs(collection, id), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorNamedGet(ctx context.Context, name string, id uint64, withVector, withPayload bool) (bool, map[string][]float32, VectorMetadata, time.Duration, error) {
	return n.VectorNamedGetExt(ctx, name, id, withVector, withPayload, ReadOpts{})
}

// VectorNamedGetExt sends the named get op with the rc opts trailer; see
// VectorGetExt. Byte-identical to VectorNamedGet on the wire when rc==0.
func (n *networkedStore) VectorNamedGetExt(ctx context.Context, name string, id uint64, withVector, withPayload bool, opts ReadOpts) (bool, map[string][]float32, VectorMetadata, time.Duration, error) {
	body, err := n.c.Call(ctx, "vector_named_get", ops.EncodeVectorGetArgsOpts(name, id, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return false, nil, nil, 0, mapErr(err)
	}
	return ops.DecodeNamedGetResult(body)
}

// VectorNamedGetBatch sends one vector_named_get_batch op carrying all (deduped)
// ids; the server-side fan-out dispatcher scatters the ids across partitions and
// returns the merged rows, which we split into points + missing. The coordinator
// owns the scatter (see fanNamedGetBatch), so the client stays a single
// round-trip. The named clone of VectorGetBatch.
func (n *networkedStore) VectorNamedGetBatch(ctx context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]NamedBatchGetPoint, []uint64, error) {
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}
	body, err := n.c.Call(ctx, "vector_named_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, getFlags(withVector, withPayload)))
	if err != nil {
		return nil, nil, mapErr(err)
	}
	rows, err := ops.DecodeNamedGetBatchResult(body)
	if err != nil {
		return nil, nil, err
	}
	return splitNamedBatchRows(rows)
}

func (n *networkedStore) VectorNamedSetPayload(ctx context.Context, name string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_named_set_payload", ops.EncodeSetPayloadArgsOpts(name, id, patch, keyTTLMs), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorNamedOverwritePayload(ctx context.Context, name string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_named_overwrite_payload", ops.EncodeSetPayloadArgsOpts(name, id, meta, keyTTLMs), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorNamedDeletePayloadKeys(ctx context.Context, name string, id uint64, keys []string, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_named_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs(name, id, keys), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorNamedClearPayload(ctx context.Context, name string, id uint64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_named_clear_payload", ops.EncodeClearPayloadArgs(name, id), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorMVGet(ctx context.Context, name string, docID uint64, withVector, withPayload bool) (bool, [][]float32, VectorMetadata, error) {
	return n.VectorMVGetExt(ctx, name, docID, withVector, withPayload, ReadOpts{})
}

// VectorMVGetExt sends the MV get op with the rc opts trailer; see VectorGetExt.
// Byte-identical to VectorMVGet on the wire when rc==0.
func (n *networkedStore) VectorMVGetExt(ctx context.Context, name string, docID uint64, withVector, withPayload bool, opts ReadOpts) (bool, [][]float32, VectorMetadata, error) {
	body, err := n.c.Call(ctx, "vector_mv_get", ops.EncodeVectorGetArgsOpts(name, docID, getFlags(withVector, withPayload), opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness))
	if err != nil {
		return false, nil, nil, mapErr(err)
	}
	return ops.DecodeMVGetResult(body)
}

// VectorMVGetBatch sends one vector_mv_get_batch op carrying all (deduped) ids;
// the server-side fan-out dispatcher scatters the ids across partitions and
// returns the merged rows, which we split into points + missing. The coordinator
// owns the scatter (see fanMVGetBatch), so the client stays a single round-trip.
// MV has NO ttl. The MV clone of VectorNamedGetBatch.
func (n *networkedStore) VectorMVGetBatch(ctx context.Context, collection string, ids []uint64, withVector, withPayload bool) ([]MVBatchGetPoint, []uint64, error) {
	ids = dedupIDs(ids)
	if len(ids) == 0 {
		return nil, nil, nil
	}
	body, err := n.c.Call(ctx, "vector_mv_get_batch", ops.EncodeVectorGetBatchArgs(collection, ids, getFlags(withVector, withPayload)))
	if err != nil {
		return nil, nil, mapErr(err)
	}
	rows, err := ops.DecodeMVGetBatchResult(body)
	if err != nil {
		return nil, nil, err
	}
	return splitMVBatchRows(rows)
}

func (n *networkedStore) VectorMVSetPayload(ctx context.Context, name string, docID uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_mv_set_payload", ops.EncodeSetPayloadArgsOpts(name, docID, patch, keyTTLMs), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorMVOverwritePayload(ctx context.Context, name string, docID uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_mv_overwrite_payload", ops.EncodeSetPayloadArgsOpts(name, docID, meta, keyTTLMs), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorMVDeletePayloadKeys(ctx context.Context, name string, docID uint64, keys []string, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_mv_delete_payload_keys", ops.EncodeDeletePayloadKeysArgs(name, docID, keys), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorMVClearPayload(ctx context.Context, name string, docID uint64, opts ...WriteOpts) (bool, error) {
	body, err := n.wcCall(ctx, "vector_mv_clear_payload", ops.EncodeClearPayloadArgs(name, docID), firstWriteOpts(opts))
	if err != nil {
		return false, mapErr(err)
	}
	return ops.DecodePayloadResult(body)
}

func (n *networkedStore) VectorScroll(ctx context.Context, collection string, filter VectorFilter, limit int, opts VectorScrollOpts) ([]VectorDocument, FanMeta, string, error) {
	afterID, hasAfter, err := ops.DecodeScrollCursor(opts.Cursor)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	body, err := n.c.Call(ctx, "vector_scroll",
		ops.EncodeScrollArgsCursorBounded(collection, filter, limit, opts.ReadConsistency, opts.OnPartitionUnavailable, afterID, hasAfter, opts.MaxStaleness))
	if err != nil {
		return nil, FanMeta{}, "", mapErr(err)
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		return nil, FanMeta{}, "", err
	}
	// next_cursor is server-authoritative: the dispatcher (fanScroll) emits the
	// coordinator's globally-merged cursor on the result wire, so the client just
	// decodes it. DecodeScrollResult tolerates an OLD server's cursor-less body
	// (nextCursor==""), preserving backward-compat.
	return docs, FanMeta{Degraded: degraded, Missing: missingInt(missing)}, nextCursor, nil
}

// VectorResplit drives an offline generational resplit on the server. The op is
// intercepted by the server's fanout decorator and run as a coordinator op on the
// receiving node (it is never shard-routed). Resplit is SYNCHRONOUS and OFFLINE:
// the call blocks until every vector has been scanned and re-inserted into the new
// generation, and the caller MUST quiesce writes to the collection for the
// duration. ctx MUST carry a deadline long enough to cover the full scan +
// re-insert; too short a deadline aborts the resplit mid-flight. An interrupted
// resplit is resumable: retry VectorResplit, then VectorResplitCleanup.
func (n *networkedStore) VectorResplit(ctx context.Context, collection string, newP int) error {
	if newP <= 1 {
		return fmt.Errorf("rostam: VectorResplit: newP must be > 1, got %d", newP)
	}
	_, err := n.c.Call(ctx, "vector_resplit", ops.EncodeResplitArgs(collection, newP))
	return mapErr(err)
}

// VectorResplitCleanup drops orphaned old-generation partitions left behind by a
// completed (or interrupted) resplit and returns the number dropped. Run as a
// coordinator op on the receiving node. ctx should carry a generous deadline.
func (n *networkedStore) VectorResplitCleanup(ctx context.Context, collection string) (int, error) {
	body, err := n.c.Call(ctx, "vector_resplit_cleanup", ops.EncodeResplitCleanupArgs(collection))
	if err != nil {
		return 0, mapErr(err)
	}
	return ops.DecodeResplitCleanupResult(body)
}

// VectorMVResplit drives an offline generational resplit of a multi-vector
// collection on the server. Same semantics as VectorResplit: synchronous,
// offline (quiesce writes), and ctx MUST carry a long deadline. An interrupted
// resplit is resumable: retry, then VectorMVResplitCleanup.
func (n *networkedStore) VectorMVResplit(ctx context.Context, collection string, newP int) error {
	if newP <= 1 {
		return fmt.Errorf("rostam: VectorMVResplit: newP must be > 1, got %d", newP)
	}
	_, err := n.c.Call(ctx, "vector_mv_resplit", ops.EncodeResplitArgs(collection, newP))
	return mapErr(err)
}

// VectorReshard drives an ONLINE generational reshard on the server. Like
// VectorResplit the op is intercepted by the server's fanout decorator and run
// as a coordinator op on the receiving node (never shard-routed). Unlike
// resplit it is LIVE: reads AND writes stay up for the whole reshard (writes
// dual-write to old+new gen during the copy). The call is SYNCHRONOUS — it
// blocks until cutover — so ctx MUST carry a deadline long enough to cover the
// drain grace plus the full streamed copy; too short a deadline aborts the
// reshard mid-flight (resumable: re-invoke VectorReshard toward the same newP).
func (n *networkedStore) VectorReshard(ctx context.Context, collection string, newP int) error {
	if newP <= 1 {
		return fmt.Errorf("rostam: VectorReshard: newP must be > 1, got %d", newP)
	}
	_, err := n.c.Call(ctx, "vector_reshard", ops.EncodeReshardArgs(collection, newP))
	return mapErr(err)
}

// VectorReshardAbort aborts an in-flight online reshard, clearing the reshard
// state back to the old generation and dropping the new-gen partitions. Run as
// a coordinator op on the receiving node. Pre-cutover only — it errors if the
// reshard has already flipped (use a new reshard to go back).
func (n *networkedStore) VectorReshardAbort(ctx context.Context, collection string) error {
	_, err := n.c.Call(ctx, "vector_reshard_abort", ops.EncodeReshardAbortArgs(collection))
	return mapErr(err)
}

// VectorMVReshard drives an ONLINE generational reshard of a multi-vector
// collection on the server. Same semantics as VectorReshard: live (dual-write),
// synchronous, ctx MUST carry a long deadline; resumable by re-invocation.
func (n *networkedStore) VectorMVReshard(ctx context.Context, collection string, newP int) error {
	if newP <= 1 {
		return fmt.Errorf("rostam: VectorMVReshard: newP must be > 1, got %d", newP)
	}
	_, err := n.c.Call(ctx, "vector_mv_reshard", ops.EncodeReshardArgs(collection, newP))
	return mapErr(err)
}

// VectorMVReshardAbort aborts an in-flight online MV reshard; see
// VectorReshardAbort. Pre-cutover only. Run as a coordinator op on the node.
func (n *networkedStore) VectorMVReshardAbort(ctx context.Context, collection string) error {
	_, err := n.c.Call(ctx, "vector_mv_reshard_abort", ops.EncodeReshardAbortArgs(collection))
	return mapErr(err)
}

// VectorMVResplitCleanup drops orphaned old-generation MV partitions left behind
// by a completed (or interrupted) MV resplit and returns the number dropped. Run
// as a coordinator op on the receiving node. ctx should carry a generous deadline.
func (n *networkedStore) VectorMVResplitCleanup(ctx context.Context, collection string) (int, error) {
	body, err := n.c.Call(ctx, "vector_mv_resplit_cleanup", ops.EncodeResplitCleanupArgs(collection))
	if err != nil {
		return 0, mapErr(err)
	}
	return ops.DecodeResplitCleanupResult(body)
}

// CreateAlias creates (or upserts) an alias on the server. Alias management is a
// coordinator op intercepted by the server's fanout decorator (never
// shard-routed) — a thin one-action batch lowered to alias_batch.
func (n *networkedStore) CreateAlias(ctx context.Context, alias, collection string) error {
	_, err := n.c.Call(ctx, "alias_batch", ops.EncodeAliasCreateArgs(alias, collection))
	return mapErr(err)
}

// DeleteAlias removes an alias on the server (a one-action delete batch).
func (n *networkedStore) DeleteAlias(ctx context.Context, alias string) error {
	_, err := n.c.Call(ctx, "alias_batch", ops.EncodeAliasDeleteArgs(alias))
	return mapErr(err)
}

// AliasBatch atomically applies a batch of alias mutations on the server in ONE
// meta-Raft log entry (atomic swap). The whole batch is validated server-side
// before commit.
func (n *networkedStore) AliasBatch(ctx context.Context, actions []AliasAction) error {
	wire := make([]ops.AliasAction, len(actions))
	for i, a := range actions {
		wire[i] = ops.AliasAction{Alias: a.Alias, Canonical: a.Canonical, Delete: a.Delete}
	}
	_, err := n.c.Call(ctx, "alias_batch", ops.EncodeAliasBatchArgs(wire))
	return mapErr(err)
}

// ListAliases returns the alias→collection map from the server (optionally
// filtered by target collection). The list is a local read on the server.
func (n *networkedStore) ListAliases(ctx context.Context, collection string) (map[string]string, error) {
	body, err := n.c.Call(ctx, "alias_list", ops.EncodeAliasListArgs(collection))
	if err != nil {
		return nil, mapErr(err)
	}
	entries, err := ops.DecodeAliasListResult(body)
	if err != nil {
		return nil, err
	}
	out := make(map[string]string, len(entries))
	for _, e := range entries {
		out[e.Alias] = e.Collection
	}
	return out, nil
}

// KeysAdd registers a new API key on the server's live registry at RUNTIME (no
// restart) via the admin-gated __keys_add__ coordinator op. The caller must be
// admin-scoped (the server's authorize gate denies otherwise). The raw token is a
// SECRET consumed by the registry and never echoed back; a nil error means the key
// is live AND durably persisted. v1 is per-node-local: the key takes effect on the
// node that handled the call (cluster-wide propagation is a v2 follow-up).
func (n *networkedStore) KeysAdd(ctx context.Context, token, tenant string, scopes []string, certCN string) error {
	args := ops.EncodeKeysAddArgs(ops.KeysAddArgs{Token: token, Tenant: tenant, Scopes: scopes, CertCN: certCN})
	_, err := n.c.Call(ctx, ops.OpKeysAdd, args)
	return mapErr(err)
}

// KeysRevoke removes an API key by its raw token via the admin-gated
// __keys_revoke__ coordinator op. After a nil-error return the token no longer
// authenticates (and the removal is durably persisted). An unknown token surfaces
// as the underlying not-found error.
func (n *networkedStore) KeysRevoke(ctx context.Context, token string) error {
	_, err := n.c.Call(ctx, ops.OpKeysRevoke, ops.EncodeKeysRevokeArgs(token))
	return mapErr(err)
}

// KeysList returns the REDACTED registry snapshot via the admin-gated
// __keys_list__ coordinator op: each entry carries the token's fingerprint +
// tenant + scopes + cert_cn, NEVER the raw token (the op result codec has no token
// field, so the secret cannot cross the wire by construction).
func (n *networkedStore) KeysList(ctx context.Context) ([]vector.RedactedKey, error) {
	body, err := n.c.Call(ctx, ops.OpKeysList, ops.EncodeKeysListArgs())
	if err != nil {
		return nil, mapErr(err)
	}
	entries, err := ops.DecodeKeysListResult(body)
	if err != nil {
		return nil, err
	}
	out := make([]vector.RedactedKey, len(entries))
	for i, e := range entries {
		out[i] = vector.RedactedKey{
			Fingerprint: e.Fingerprint,
			Tenant:      e.Tenant,
			Scopes:      e.Scopes,
			CertCN:      e.CertCN,
		}
	}
	return out, nil
}
