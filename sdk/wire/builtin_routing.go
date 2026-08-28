// SPDX-License-Identifier: Apache-2.0

package wire

// BuiltinOp describes one built-in op's ROUTING metadata: everything a router
// needs to pick a destination shard, without a handler.
type BuiltinOp struct {
	Name       string
	Kind       OpKind
	KE         KeyExtractor
	Layout     RouteLayout
	CrossShard bool
}

// BuiltinOps is the canonical, ordered table of every built-in op's routing
// metadata — the five KV ops, put_batch, the four shardless admin ops, and every
// vector_*/vector_mv_*/vector_named_* op. RegisterRoutableBuiltins (this
// package, client-side routing-only) and ops.RegisterBuiltins (server, which
// additionally binds each op's real Handler) BOTH walk this SAME table, so the
// two registries can never drift apart on an op's name, kind, KeyExtractor, or
// RouteLayout.
//
// Vector ops are routed by collection name (VectorKeyColAt1/At2, via the
// routeAt1/routeAt2 layout pairs) so each collection lives on one shard's Raft
// group and collections distribute across shards. Args that lead with
// [flags][colLen] use At2; those that lead with [nameLen] use At1.
var BuiltinOps = []BuiltinOp{
	{"get", OpReadOnly, StdKeyExtractor, RouteLayoutNone, false},
	{"put", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	{"del", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	{"expire", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	{"incr", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	// Conditional-write KV ops. All three lead with [keyLen u16][key], so they
	// route by that key via StdKeyExtractor exactly like get/put/incr.
	{"set_nx", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	{"cas", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	{"cad", OpReadWrite, StdKeyExtractor, RouteLayoutNone, false},
	// put_batch packs N puts into one Raft log entry. It routes by its FIRST key,
	// so every key in a batch must hash to the same shard — the cluster fan-out
	// (Node.PutBatch) guarantees that by grouping before it calls.
	{"put_batch", OpReadWrite, putBatchKeyExtractor, RouteLayoutNone, false},
	// __ping__/__ready__/__metrics__/__repl_metrics__ are shardless (nil KeyExtractor).
	{"__ping__", OpReadOnly, nil, RouteLayoutNone, false},
	{ReadyOp, OpReadOnly, nil, RouteLayoutNone, false},
	{MetricsOp, OpReadOnly, nil, RouteLayoutNone, false},
	{ReplMetricsOp, OpReadOnly, nil, RouteLayoutNone, false},

	{"vector_create_collection", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_drop_collection", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_insert", OpReadWrite, routeAt2.ke, routeAt2.layout, false},
	{"vector_insert_if_absent", OpReadWrite, routeAt2.ke, routeAt2.layout, false},
	{"vector_exists", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_delete", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_get", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_get_batch", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_set_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_overwrite_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_delete_payload_keys", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_clear_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_search", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_hybrid_search", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_hybrid_lanes", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	// Full-text (BM25) ops. Both lead with [flags:u8][colLen:u8][col]... (the
	// At2 layout, IDENTICAL to vector_hybrid_search), so they route via
	// VectorKeyColAt2 — name at offset 1, behind the flags byte.
	// vector_hybrid_text_lanes is the per-partition fan-out leaf (shares the
	// vector_hybrid_text wire).
	{"vector_search_text", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_hybrid_text", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_hybrid_text_lanes", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	// vector_bm25_stats is phase 0 of the global-DF (dfs) text fan-out. Its args
	// lead with [colLen:u8][col]... (NO flags byte), so it routes via
	// VectorKeyColAt1 — name at offset 0, NOT At2 like the scoring ops.
	{"vector_bm25_stats", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	// vector_query is the unified Query API op. Its args lead with [colLen:u8]
	// [col] (the QuerySpec blob is opaque to routing), so it routes via
	// VectorKeyColAt1 like the rest of the At1 family.
	{"vector_query", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_upsert", OpReadWrite, routeAt2.ke, routeAt2.layout, false},
	{"vector_bulk_stage", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_bulk_stage_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_bulk_build", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_search_docs", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_delete_by_filter", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_search_groups", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_group_candidates", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_scroll", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_scan_vectors", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_get_config", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_create_collection", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_drop_collection", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_add", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_add_if_absent", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_add_versioned", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_add_batch", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_exists", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_search", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	// MV-hybrid ops use the At2 (flags-first) layout — [flags:u8][colLen:u8][col]...
	// (EncodeMVHybridArgs), IDENTICAL to the named/dense hybrid wire, so they route
	// via VectorKeyColAt2 (NOT At1 like the rest of the mv_* family). The lanes op
	// is the partition fan-out leaf.
	{"vector_mv_hybrid_search", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_mv_hybrid_lanes", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_mv_delete", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_get", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_get_batch", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_set_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_overwrite_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_delete_payload_keys", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_clear_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_get_config", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_scan_vectors", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_mv_scroll", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	// vector_mv_query is the multi-vector Query API op (MaxSim + doc-sparse
	// FUSION / RERANK). It shares the vector_query arg wire ([colLen:u8][col]
	// [specLen:u32][spec][optsTrailer]) and result codec, so it routes by
	// collection name at offset 0 (VectorKeyColAt1) like the rest of the At1 family.
	{"vector_mv_query", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_create_collection", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_drop_collection", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_insert", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_delete", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_get", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_get_batch", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_set_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_overwrite_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_delete_payload_keys", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_clear_payload", OpReadWrite, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_search", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_sparse_search", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_hybrid_search", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_named_hybrid_lanes", OpReadOnly, routeAt2.ke, routeAt2.layout, false},
	{"vector_named_search_docs", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_scroll", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	{"vector_named_get_config", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
	// vector_named_query is the named-collection Query API op (multi-space N-lane
	// FUSION / RERANK). It shares the vector_query arg wire ([colLen:u8][col]
	// [specLen:u32][spec][optsTrailer]) and result codec, so it routes by
	// collection name at offset 0 (VectorKeyColAt1) like the rest of the At1 family.
	{"vector_named_query", OpReadOnly, routeAt1.ke, routeAt1.layout, false},
}

// RegisterRoutableBuiltins registers every built-in op's ROUTING metadata (name,
// kind, KeyExtractor, RouteLayout, cross-shard) from BuiltinOps into reg — no
// handler. This is everything the Go client's shard-aware routing needs (see
// client.NewRouted): the client never invokes an op's handler locally, only its
// KeyExtractor, to pick a destination shard.
func RegisterRoutableBuiltins(reg *Registry) error {
	for _, o := range BuiltinOps {
		var err error
		switch {
		case o.Layout != RouteLayoutNone:
			err = reg.RegisterRoutableInto(o.Name, o.Kind, o.KE, o.Layout)
		case o.KE != nil:
			if o.CrossShard {
				err = reg.RegisterRoutableCrossShard(o.Name, o.Kind, o.KE)
			} else {
				err = reg.RegisterRoutable(o.Name, o.Kind, o.KE)
			}
		default:
			err = reg.Register(o.Name, o.Kind)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
