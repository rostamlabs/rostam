// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// innerDispatcher is the minimal surface the decorator wraps. Both *cluster.Node
// and *cluster.LeaderFollowingDispatcher satisfy it, and *fanoutDispatcher in
// turn satisfies server.Dispatcher / httpapi.Dispatcher / grpcapi.Dispatcher
// (all structurally identical to this).
type innerDispatcher interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}

// fanoutDispatcher makes the remote transports partition-aware. For known vector
// ops on a collection with Partitions>1 it routes through the embedded backend's
// existing fan-out coordinators and re-encodes the result in the op's wire
// format; for unpartitioned collections it passes the original bytes straight
// through, byte-identical to the non-decorated path.
type fanoutDispatcher struct {
	e     *embedded
	inner innerDispatcher
}

func newFanoutDispatcher(e *embedded, inner innerDispatcher) *fanoutDispatcher {
	return &fanoutDispatcher{e: e, inner: inner}
}

// LeaderAddr delegates to the wrapped dispatcher.
func (f *fanoutDispatcher) LeaderAddr() string { return f.inner.LeaderAddr() }

// missingU16 converts the FanMeta.Missing partition indices ([]int) to the
// []uint16 the degraded wire trailer carries. Partition indices are bounded by
// the partition count (≤ 65536), so the conversion is safe; out-of-range values
// are clamped to 65535 as a defensive guard.
func missingU16(missing []int) []uint16 {
	if len(missing) == 0 {
		return nil
	}
	out := make([]uint16, len(missing))
	for i, p := range missing {
		if p < 0 {
			p = 0
		} else if p > 65535 {
			p = 65535
		}
		out[i] = uint16(p) //nolint:gosec // bounded/clamped above
	}
	return out
}

// fanRoute is the ONE routing decision a dispatched vector op needs, resolved
// once in Call and handed to the fan handler. It replaces the prologue every
// handler used to repeat (cheap-peek the collection name, resolve the alias, read
// the partition catalog), computed once instead of two to three times per op.
//
// LANDMINE #1 lives here: alias resolution and the partition decision MUST come
// from the SAME name. When they did not, a partitioned collection reached through
// an alias looked unpartitioned to the dispatcher, passed through to the empty
// logical collection, and silently returned zero results. Resolving once makes
// that class of drift unrepresentable — there is no second resolution to disagree
// with this one — which is exactly why every fan handler takes this value rather
// than re-deriving any part of it.
//
// It deliberately carries NO partition count or generation. Those were here, set
// and never read, and that is a trap rather than a convenience: a pre-dispatch
// (P, gen) snapshot is a SECOND view of the catalog, taken before the coordinator
// re-reads it, and a handler that used the stale pair while the coordinator used a
// fresh one would resurrect exactly the resolve-vs-use divergence above (a reshard
// flips gen mid-dispatch). A handler that genuinely needs the breadth must read
// the catalog itself, at the moment it fans out — so this type answers only
// "which collection, and does it fan out at all".
type fanRoute struct {
	// coll is the CANONICAL (alias-resolved) collection name; valid only when ok.
	coll string
	// ok reports that this op is an alias-resolving data-plane op whose args
	// carried a well-formed collection name. When false the op has no route: the
	// handler passes the original bytes through.
	ok bool
	// partitioned is the fan-out gate: the collection has P>1. Implies ok.
	partitioned bool
}

// resolveRoute is the dispatcher's single collection-name chokepoint: it peeks the
// op's collection name, resolves any alias, rewrites the name in the args to the
// canonical target, and reads the partition catalog — ONCE per dispatched op, for
// both branches downstream:
//
//   - the fan-out branch (partitioned target): the returned route gates it and the
//     coll the embedded coordinator decodes from args is canonical;
//   - the pass-through branch (UNPARTITIONED target): inner.Call gets the canonical
//     name (the inner op handler has no alias knowledge, so without the rewrite an
//     unpartitioned-alias op would hit "unknown collection").
//
// Only DATA-PLANE ops resolve (dataPlaneAliasOps). Admin ops — create/drop,
// reshard/resplit, get_config, alias_* — are excluded so they keep real-name
// semantics: dropping the alias "prod" must never drop the underlying collection.
// They get a zero route and do their own (raw-name) catalog read.
//
// Reserved chars ('#' for physical partitions like "docs#2", '@' for higher
// generations like "docs@1#2") only ever appear in physical/internal names — a
// logical partitioned collection never contains them (guarded at create). Such a
// name neither resolves nor partitions, so a forwarded internal op naming a
// physical partition passes straight through and cannot double fan out.
//
// The partition catalog is read for EVERY resolvable data-plane op, including the
// eight that never consult the result (vector_exists, the *_if_absent pair,
// vector_mv_add_versioned — all of which pass through — and the vector_*query
// family, which always delegates). Skipping those would mean a SECOND
// classification of the op set, one bit per op saying "this one needs the
// partition decision", and getting that bit wrong for a future op is silent: the
// route reports not-partitioned, the handler passes through to the empty logical
// collection, and a partitioned collection returns zero results with no error.
// One catalog RLock is a poor trade against a failure mode with no symptom, so
// the decision is always computed and the op set is classified exactly once.
func (f *fanoutDispatcher) resolveRoute(op string, args []byte) (fanRoute, []byte) {
	if _, isDataPlane := dataPlaneAliasOps[op]; !isDataPlane {
		return fanRoute{}, args
	}
	peek, ok := ops.CollectionNameFor(op, args)
	if !ok {
		return fanRoute{}, args
	}
	if strings.ContainsAny(peek, "#@") {
		// Physical/internal name: never an alias, never partitioned.
		return fanRoute{coll: peek, ok: true}, args
	}
	r := fanRoute{coll: f.e.resolveAlias(peek), ok: true}
	if r.coll != peek {
		// The inner op handler has no alias knowledge, so the canonical target is
		// spliced into the args here. A rewrite that cannot be encoded leaves the
		// args alone: the route is still canonical, and the embedded coordinator
		// resolves the alias again on its own path.
		if rewritten, rok := ops.RewriteCollectionName(op, args, r.coll); rok {
			args = rewritten
		}
	}
	_, _, r.partitioned = f.partitionsOf(r.coll)
	return r, args
}

// partitionsOf reports the live partition count/generation of a CANONICAL
// (already alias-resolved) collection name, or (0,0,false) when it is
// unpartitioned (P<=1) or unknown.
func (f *fanoutDispatcher) partitionsOf(canon string) (int, uint32, bool) {
	p, gen, ok := f.e.catalog.PartitionsGen(canon)
	if !ok || p <= 1 {
		return 0, 0, false
	}
	return p, gen, true
}

// partitioned is the collection-NAME form of the decision resolveRoute makes from
// an op's args: the same '#'/'@' short-circuit, the same alias resolution, the same
// catalog read. The dispatch path does not use it — it resolves once, in
// resolveRoute — but the inttest fan-out tests assert the routing lookup directly
// (see inttest_support.go), so the two-step lookup stays expressed here in terms of
// the same pieces rather than being reimplemented in the test package.
func (f *fanoutDispatcher) partitioned(coll string) (int, uint32, bool) {
	if strings.ContainsAny(coll, "#@") {
		return 0, 0, false
	}
	return f.partitionsOf(f.e.resolveAlias(coll))
}

// dataPlaneAliasOps is the set of DATA-PLANE vector ops whose collection name is
// alias-resolved at the dispatcher chokepoint. It deliberately EXCLUDES admin ops
// — create/drop (collection lifecycle, real names + drop cascade), reshard/resplit
// (physical '#'/'@' names), get_config, and the alias_* coordinator ops — so those
// keep real-name semantics. The list covers every data-plane op that resolves in
// embedded.go / named_fanout.go. Some entries (vector_insert_if_absent,
// vector_exists, vector_mv_add_if_absent, vector_mv_exists) have no fan-out case
// in Call() — they are included so their collection name is rewritten on the
// inner.Call pass-through path; the fan-out vs pass-through distinction does not
// change which ops resolve.
var dataPlaneAliasOps = map[string]struct{}{
	"vector_search": {}, "vector_search_docs": {}, "vector_hybrid_search": {},
	"vector_search_text": {}, "vector_hybrid_text": {},
	"vector_query":         {},
	"vector_search_groups": {}, "vector_scroll": {}, "vector_delete_by_filter": {},
	"vector_insert": {}, "vector_upsert": {}, "vector_insert_if_absent": {},
	"vector_delete": {}, "vector_exists": {}, "vector_get": {}, "vector_get_batch": {},
	"vector_set_payload": {}, "vector_overwrite_payload": {},
	"vector_delete_payload_keys": {}, "vector_clear_payload": {}, "vector_operate": {},
	"vector_mv_add": {}, "vector_mv_delete": {}, "vector_mv_search": {},
	"vector_mv_hybrid_search": {},
	"vector_mv_query":         {},
	"vector_mv_add_versioned": {},
	"vector_mv_add_if_absent": {}, "vector_mv_exists": {}, "vector_mv_get": {},
	"vector_mv_get_batch": {}, "vector_mv_scroll": {},
	"vector_mv_set_payload": {}, "vector_mv_overwrite_payload": {},
	"vector_mv_delete_payload_keys": {}, "vector_mv_clear_payload": {}, "vector_mv_operate": {},
	"vector_named_insert": {}, "vector_named_delete": {}, "vector_named_search": {},
	"vector_named_sparse_search": {},
	"vector_named_hybrid_search": {},
	"vector_named_query":         {},
	"vector_named_search_docs":   {}, "vector_named_scroll": {}, "vector_named_get": {},
	"vector_named_get_batch":   {},
	"vector_named_set_payload": {}, "vector_named_overwrite_payload": {},
	"vector_named_delete_payload_keys": {}, "vector_named_clear_payload": {}, "vector_named_operate": {},
}

// Call routes a single op. Known read ops on a partitioned collection fan out
// through the embedded backend and are re-encoded with the op's single-shard
// wire encoder; everything else (and every op on an unpartitioned collection)
// passes through to the inner dispatcher unchanged.
func (f *fanoutDispatcher) Call(name string, args []byte) ([]byte, error) {
	// The op's whole routing decision, resolved ONCE here (alias rewrite included)
	// and handed to the fan handler, so the handler and the pass-through branch can
	// never disagree about which collection this is. See fanRoute / resolveRoute.
	r, args := f.resolveRoute(name, args)
	switch name {
	case "vector_search":
		return f.fanSearch(name, args, r)
	case "vector_search_docs":
		return f.fanDocs(name, args, r)
	case "vector_search_groups":
		return f.fanGroups(name, args, r)
	case "vector_hybrid_search":
		return f.fanHybrid(name, args, r)
	case "vector_search_text":
		return f.fanSearchText(name, args, r)
	case "vector_hybrid_text":
		return f.fanHybridText(name, args, r)
	case "vector_query":
		return f.fanQuery(name, args, r)
	case "vector_mv_hybrid_search":
		return f.fanMVHybridSearch(name, args, r)
	case "vector_mv_query":
		return f.fanMVQuery(name, args, r)
	case "vector_scroll":
		return f.fanScroll(name, args, r)
	case "vector_delete_by_filter":
		return f.fanDeleteByFilter(name, args, r)
	case "vector_insert":
		return f.fanInsert(name, args, r)
	case "vector_upsert":
		return f.fanUpsert(name, args, r)
	case "vector_delete":
		return f.fanDelete(name, args, r)
	case "vector_get":
		return f.fanGet(name, args, r)
	case "vector_get_batch":
		return f.fanGetBatch(name, args, r)
	case "vector_set_payload":
		return f.fanSetPayload(name, args, r)
	case "vector_overwrite_payload":
		return f.fanOverwritePayload(name, args, r)
	case "vector_delete_payload_keys":
		return f.fanDeletePayloadKeys(name, args, r)
	case "vector_clear_payload":
		return f.fanClearPayload(name, args, r)
	case "vector_operate":
		return f.fanOperate(name, args, r)
	case "vector_create_collection":
		return f.fanCreateCollection(name, args, r)
	case "vector_drop_collection":
		return f.fanDropCollection(name, args, r)
	case "vector_mv_create_collection":
		return f.fanMVCreate(name, args, r)
	case "vector_mv_add":
		return f.fanMVAdd(name, args, r)
	case "vector_mv_delete":
		return f.fanMVDelete(name, args, r)
	case "vector_mv_get":
		return f.fanMVGet(name, args, r)
	case "vector_mv_get_batch":
		return f.fanMVGetBatch(name, args, r)
	case "vector_mv_set_payload":
		return f.fanMVSetPayload(name, args, r)
	case "vector_mv_overwrite_payload":
		return f.fanMVOverwritePayload(name, args, r)
	case "vector_mv_delete_payload_keys":
		return f.fanMVDeletePayloadKeys(name, args, r)
	case "vector_mv_clear_payload":
		return f.fanMVClearPayload(name, args, r)
	case "vector_mv_operate":
		return f.fanMVOperate(name, args, r)
	case "vector_mv_search":
		return f.fanMVSearch(name, args, r)
	case "vector_mv_scroll":
		return f.fanMVScroll(name, args, r)
	case "vector_mv_drop_collection":
		return f.fanMVDrop(name, args, r)
	case "vector_named_create_collection":
		return f.fanNamedCreate(name, args, r)
	case "vector_named_insert":
		return f.fanNamedInsert(name, args, r)
	case "vector_named_delete":
		return f.fanNamedDelete(name, args, r)
	case "vector_named_get":
		return f.fanNamedGet(name, args, r)
	case "vector_named_get_batch":
		return f.fanNamedGetBatch(name, args, r)
	case "vector_named_set_payload":
		return f.fanNamedSetPayload(name, args, r)
	case "vector_named_overwrite_payload":
		return f.fanNamedOverwritePayload(name, args, r)
	case "vector_named_delete_payload_keys":
		return f.fanNamedDeletePayloadKeys(name, args, r)
	case "vector_named_clear_payload":
		return f.fanNamedClearPayload(name, args, r)
	case "vector_named_operate":
		return f.fanNamedOperate(name, args, r)
	case "vector_named_search":
		return f.fanNamedSearch(name, args, r)
	case "vector_named_sparse_search":
		return f.fanNamedSparseSearch(name, args, r)
	case "vector_named_hybrid_search":
		return f.fanNamedHybridSearch(name, args, r)
	case "vector_named_query":
		return f.fanNamedQuery(name, args, r)
	case "vector_named_search_docs":
		return f.fanNamedDocs(name, args, r)
	case "vector_named_scroll":
		return f.fanNamedScroll(name, args, r)
	case "vector_named_drop_collection":
		return f.fanNamedDrop(name, args, r)
	case "vector_named_get_config":
		return f.fanNamedGetConfig(name, args, r)
	case "vector_resplit":
		return f.handleResplit(args, f.e.VectorResplit)
	case "vector_mv_resplit":
		return f.handleResplit(args, f.e.VectorMVResplit)
	case "vector_resplit_cleanup":
		return f.handleResplitCleanup(args, f.e.VectorResplitCleanup)
	case "vector_mv_resplit_cleanup":
		return f.handleResplitCleanup(args, f.e.VectorMVResplitCleanup)
	case "vector_reshard":
		return f.handleReshard(args, f.e.VectorReshard)
	case "vector_mv_reshard":
		return f.handleReshard(args, f.e.VectorMVReshard)
	case "vector_reshard_abort":
		return f.handleReshardAbort(args, f.e.VectorReshardAbort)
	case "vector_mv_reshard_abort":
		return f.handleReshardAbort(args, f.e.VectorMVReshardAbort)
	case ops.WCEnvelopeOp:
		return f.handleWCEnvelope(args)
	case "alias_batch":
		return f.handleAliasBatch(args)
	case "alias_list":
		return f.handleAliasList(args)
	default:
		return f.inner.Call(name, args)
	}
}

// handleWCEnvelope intercepts the __wc__ write-consistency envelope (the
// networked/forwarded write path) exactly the way alias_batch/reshard coordinator
// ops are intercepted: it is NOT shard-routed and NOT in ops/builtin.go. The flow:
//
//  1. Decode the envelope (fail-loud on a malformed frame).
//  2. RECURSE through f.Call(inner, innerArgs) so the inner write gets the full
//     normal routing / partition fan-out / Raft majority-commit path, 100%
//     unchanged — the inner op name + args are byte-identical to a plain write.
//  3. On success, if the envelope is ACTIVE (wcf>0 or wait=false), run the
//     post-commit barrier on the target shard(s) the inner write landed on. A
//     decode error or an inner-write error short-circuits (no barrier).
//
// Target shard resolution (see wcBarrierInner): a single-point write barriers the
// ONE owning physical partition (PartitionOf(id,P)); delete_by_filter over a
// partitioned collection fans to every partition, so it barriers each touched
// shard; an unpartitioned collection barriers its single shard. Barrier errors
// (*cluster.ErrWriteConsistency) surface to the caller while the write stays
// durable at majority.
func (f *fanoutDispatcher) handleWCEnvelope(args []byte) ([]byte, error) {
	wcf, wait, inner, innerArgs, err := ops.DecodeWCEnvelope(args)
	if err != nil {
		return nil, err
	}
	res, err := f.Call(inner, innerArgs)
	if err != nil {
		return nil, err
	}
	// Active iff anything beyond the default (mirrors WriteOpts.wcActive): a
	// factor was requested OR wait was explicitly false. An inactive envelope
	// would never have been built by the client, but guard anyway so a
	// hand-rolled inactive envelope is a pure pass-through (no barrier).
	active := wcf > 0 || wait == 0
	if !active {
		return res, nil
	}
	if err := f.wcBarrierInner(inner, innerArgs, wcf, wait != 0); err != nil {
		return res, err
	}
	return res, nil
}

// wcBarrierInner resolves the physical shard(s) the inner write landed on and
// runs cluster.Node.BarrierForShard on each. It is READ-ONLY w.r.t. args. For a
// partitioned collection it resolves the alias and reads (P, gen) from the
// catalog — the SAME routing the inner write used — so the barrier targets the
// exact Raft group(s) that received the entry.
//
//   - delete_by_filter over a partitioned collection: the inner f.Call already
//     fanned the delete to every partition, so barrier each partition [0..P),
//     deduping by shard index. First *cluster.ErrWriteConsistency wins (deletes
//     are independently durable; earlier shards may have met the factor).
//   - single-point write (insert/upsert/delete/payload): extract the id and
//     barrier the one owning partition PartitionOf(id,P).
//   - unpartitioned collection (P<=1) or any op without a recoverable target:
//     barrier the single shard for the collection name.
func (f *fanoutDispatcher) wcBarrierInner(inner string, innerArgs []byte, wcf uint8, wait bool) error {
	coll, ok := ops.CollectionNameFor(inner, innerArgs)
	if !ok {
		// Not a collection-keyed write (should never reach here for a real write
		// op); nothing to barrier.
		return nil
	}
	coll = f.e.resolveAlias(coll)
	P, gen, partitioned := f.e.catalog.PartitionsGen(coll)

	barrier := func(physName string) error {
		return f.e.node.BarrierForShard(
			f.e.node.ShardIndexForName(physName), wcf, wait, cluster.WriteConsistencyTimeout)
	}

	// delete_by_filter: barrier every partition (dedup shard indices) of a
	// partitioned collection; the single shard otherwise.
	if inner == "vector_delete_by_filter" {
		if !partitioned || P <= 1 {
			return barrier(coll)
		}
		seen := make(map[int]struct{}, P)
		var firstErr error
		for p := 0; p < P; p++ {
			phys := string(ops.PartitionKeyGen(coll, gen, p))
			idx := f.e.node.ShardIndexForName(phys)
			if _, dup := seen[idx]; dup {
				continue
			}
			seen[idx] = struct{}{}
			if err := f.e.node.BarrierForShard(idx, wcf, wait, cluster.WriteConsistencyTimeout); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		return firstErr
	}

	// Single-point write: barrier the one owning physical partition. NOTE: this
	// uses the LIVE gen only. During an in-progress reshard the inner write
	// dual-writes both gens (embedded.dualTargets), and the embedded path
	// (applyDualWrite) barriers BOTH; this networked envelope path barriers only
	// the live gen. Both legs are durable at majority either way, so this is a
	// documented strictness gap, not a correctness bug —
	// the networked client does not drive reshards.
	if id, hasID := ops.PointIDFor(inner, innerArgs); hasID && partitioned && P > 1 {
		phys := string(ops.PartitionKeyGen(coll, gen, ops.PartitionOf(id, P)))
		return barrier(phys)
	}
	// Unpartitioned (or no recoverable id): the collection name is the physical
	// target.
	return barrier(coll)
}

// handleAliasBatch runs an atomic alias batch (create/delete) directly on the
// receiving node. Alias management is a coordinator "virtual op" — it mutates
// meta-Raft metadata (the alias catalog), NOT shard data — so like handleReshard
// it is intercepted here and is NEVER shard-routed. It validates the whole batch
// before commit (target exists, no shadow, no '#'/'@', target not an alias);
// validation/commit errors propagate as op errors. Returns an empty ack.
func (f *fanoutDispatcher) handleAliasBatch(args []byte) ([]byte, error) {
	wire, err := ops.DecodeAliasBatchArgs(args)
	if err != nil {
		return nil, err
	}
	actions := make([]AliasAction, len(wire))
	for i, a := range wire {
		actions[i] = AliasAction{Alias: a.Alias, Canonical: a.Canonical, Delete: a.Delete}
	}
	if err := f.e.AliasBatch(context.Background(), actions); err != nil {
		return nil, err
	}
	return nil, nil
}

// handleAliasList returns the alias→collection map (optionally filtered by target
// collection) as a coordinator op. The list is a local FSM read (no Raft), and
// like handleAliasBatch it is never shard-routed.
func (f *fanoutDispatcher) handleAliasList(args []byte) ([]byte, error) {
	coll, err := ops.DecodeAliasListArgs(args)
	if err != nil {
		return nil, err
	}
	m, err := f.e.ListAliases(context.Background(), coll)
	if err != nil {
		return nil, err
	}
	entries := make([]ops.AliasEntry, 0, len(m))
	for alias, target := range m {
		entries = append(entries, ops.AliasEntry{Alias: alias, Collection: target})
	}
	return ops.EncodeAliasListResult(entries), nil
}

// handleResplit runs an offline generational resplit (dense or MV, selected by
// the fn passed from the Call switch) directly on the receiving node. Resplit is
// a coordinator op — it drives the partition catalog and per-partition scan/
// re-insert itself — so it is intercepted here as a decorator "virtual op" and is
// never shard-routed. It validates partition state internally (erroring cleanly on
// a non-partitioned/physical name), so there is no partitioned()/cheap-peek gate.
func (f *fanoutDispatcher) handleResplit(args []byte, fn func(context.Context, string, int) error) ([]byte, error) {
	coll, newP, err := ops.DecodeResplitArgs(args)
	if err != nil {
		return nil, err
	}
	if err := fn(context.Background(), coll, newP); err != nil {
		return nil, err
	}
	return nil, nil
}

// handleResplitCleanup runs the orphan-partition cleanup for a resplit (dense or
// MV, selected by fn) on the receiving node and re-encodes the dropped count.
// Like handleResplit, it runs the coordinator op directly (never shard-routed).
func (f *fanoutDispatcher) handleResplitCleanup(args []byte, fn func(context.Context, string) (int, error)) ([]byte, error) {
	coll, err := ops.DecodeResplitCleanupArgs(args)
	if err != nil {
		return nil, err
	}
	n, err := fn(context.Background(), coll)
	if err != nil {
		return nil, err
	}
	return ops.EncodeResplitCleanupResult(n), nil
}

// handleReshard runs an ONLINE generational reshard (dense or MV, selected by
// the fn passed from the Call switch) directly on the receiving node. Like
// handleResplit it is a coordinator "virtual op" — the orchestrator drives the
// partition catalog, dual-write state, and the if-absent copy itself — so it is
// intercepted here as a decorator op and is never shard-routed. Unlike resplit,
// reads AND writes stay live for the duration; the call still blocks until
// cutover completes. It validates partition state internally (erroring cleanly
// on a non-partitioned/physical name), so there is no partitioned() gate. The
// orchestrator runs synchronously and can take a while (drain grace + full
// streamed copy), so it is driven with context.Background() — exactly like
// handleResplit — rather than inheriting a potentially short dispatch deadline.
func (f *fanoutDispatcher) handleReshard(args []byte, fn func(context.Context, string, int) error) ([]byte, error) {
	coll, newP, err := ops.DecodeReshardArgs(args)
	if err != nil {
		return nil, err
	}
	if err := fn(context.Background(), coll, newP); err != nil {
		return nil, err
	}
	return nil, nil
}

// handleReshardAbort aborts an in-flight online reshard (dense or MV, selected
// by fn) on the receiving node, clearing the reshard state back to the old
// generation and dropping the new-gen partitions. Pre-cutover only; the
// orchestrator errors if the reshard has already flipped. Like handleReshard it
// runs the coordinator op directly (never shard-routed) and returns an empty
// ack — errors propagate as op errors.
func (f *fanoutDispatcher) handleReshardAbort(args []byte, fn func(context.Context, string) error) ([]byte, error) {
	coll, err := ops.DecodeReshardAbortArgs(args)
	if err != nil {
		return nil, err
	}
	if err := fn(context.Background(), coll); err != nil {
		return nil, err
	}
	return nil, nil
}

func (f *fanoutDispatcher) fanSearch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, k, query, filter, rc, opa, bound, err := ops.DecodeVectorSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorSearchExt(context.Background(), coll, query, k,
		VectorSearchOpts{Filter: filter, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorSearchResultsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (f *fanoutDispatcher) fanDocs(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, k, query, filter, rc, opa, bound, err := ops.DecodeVectorSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorSearchDocs(context.Background(), coll, query, k,
		VectorSearchOpts{Filter: filter, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorDocsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (f *fanoutDispatcher) fanGroups(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, k, query, opts, rc, opa, bound, err := ops.DecodeGroupSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	opts.ReadConsistency = rc
	opts.OnPartitionUnavailable = opa
	opts.MaxStaleness = bound
	res, meta, err := f.e.VectorSearchGroups(context.Background(), coll, query, k, opts)
	if err != nil {
		return nil, err
	}
	return ops.EncodeGroupsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (f *fanoutDispatcher) fanHybrid(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, dense, k, sparse, hopts, rc, opa, bound, err := ops.DecodeHybridSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	// Inverse of toVectorHybridOpts (store.go): VectorSparse aliases
	// vector.SparseVector and FusionMethod aliases vector.FusionMethod, so the
	// decoded values map across directly.
	opts := VectorHybridOpts{
		Sparse:                 sparse,
		Filter:                 hopts.Filter,
		Method:                 hopts.Method,
		Alpha:                  hopts.Alpha,
		RRFK:                   hopts.RRFK,
		DenseK:                 hopts.DenseK,
		SparseK:                hopts.SparseK,
		ReadConsistency:        rc,
		OnPartitionUnavailable: opa,
		MaxStaleness:           bound,
	}
	res, meta, err := f.e.VectorHybridSearch(context.Background(), coll, dense, k, opts)
	if err != nil {
		return nil, err
	}
	return ops.EncodeHybridResultsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// fanSearchText is the coordinator side of the BM25 full-text search. It mirrors
// fanDocs (search_text returns Documents), fanning vector_search_text to every
// partition and merging by descending BM25 score (textDocsFanOut). Pass-through
// for an unpartitioned collection (the per-shard handler returns docs directly).
func (f *fanoutDispatcher) fanSearchText(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	// The inbound request carries the GlobalIDF intent flag; any phase-1 stats block
	// (the discarded value) is intentionally ignored — THIS coordinator computes the
	// global stats itself in phase 0 and is the source of truth, so a client-supplied
	// block is never trusted.
	coll, query, k, filter, rc, opa, bound, globalIDF, _, err := ops.DecodeSearchTextArgsGlobal(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorSearchText(context.Background(), coll, query, k,
		VectorSearchOpts{Filter: filter, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound, GlobalIDF: globalIDF})
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorDocsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// fanHybridText is the coordinator side of the dense + BM25-text hybrid. It
// mirrors fanHybrid, decoding the hybrid-text wire and delegating to
// embedded.VectorHybridText (which fans vector_hybrid_text_lanes and fuses once
// — hybridTextFanOut). Pass-through for an unpartitioned collection.
func (f *fanoutDispatcher) fanHybridText(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	// Inbound stats block (discarded) is intentionally ignored — the coordinator
	// computes global stats in phase 0; only the GlobalIDF intent flag is honored.
	coll, dense, query, k, hopts, rc, opa, bound, globalIDF, _, err := ops.DecodeHybridTextArgsGlobal(args)
	if err != nil {
		return nil, err
	}
	opts := VectorHybridOpts{
		Filter:                 hopts.Filter,
		Method:                 hopts.Method,
		Alpha:                  hopts.Alpha,
		RRFK:                   hopts.RRFK,
		DenseK:                 hopts.DenseK,
		SparseK:                hopts.SparseK,
		ReadConsistency:        rc,
		OnPartitionUnavailable: opa,
		MaxStaleness:           bound,
		GlobalIDF:              globalIDF,
	}
	res, meta, err := f.e.VectorHybridText(context.Background(), coll, dense, query, k, opts)
	if err != nil {
		return nil, err
	}
	return ops.EncodeHybridResultsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// fanQuery is the coordinator side of the unified Query API (vector_query). It
// decodes (collection, specBytes, engine spec, rc/opa/bound) and delegates to
// embedded.VectorQuery for BOTH the partitioned (P>1: fan vector_query to every
// partition + merge per mode via queryFanOut) AND the unpartitioned (P<=1: run
// on the read leader + merge the single shard's result through the SAME mode
// merge) paths, then RE-ENCODES the FLAT fused/reranked top-k + degraded/missing
// trailer so every reader (the dedicated VectorQuery RPC, HTTP /query, the
// networked client) decodes one uniform flat shape via DecodeQueryResultDegraded.
//
// Why NOT pass the unpartitioned case straight through to f.inner: the per-shard
// handler returns a mode-tagged payload whose FUSION variant carries the UNFUSED
// prefetch lanes (Lanes, not Fused) — never a flat top-k. embedded.VectorQuery
// runs the local fusion merge for the single shard so the wire is always flat;
// passing through would leak the lanes shape to the flat-only RPC/HTTP/client
// decoders. (callReadLeader dispatches via e.node, bypassing this dispatcher, so
// there is no recursion.)
//
// rc/opa/bound are threaded through to the embedded coordinator so a Linearizable
// query routes to each partition leader and arms the per-shard barrier; without
// this thread the rc would be silently dropped on the fan-out path.
func (f *fanoutDispatcher) fanQuery(name string, args []byte, _ fanRoute) ([]byte, error) {
	coll, specBytes, spec, rc, opa, bound, err := ops.DecodeQuerySpecArgs(args)
	if err != nil {
		return nil, err
	}
	// The shared fanRoute is unused here: BOTH the partitioned and the unpartitioned
	// case go through the embedded coordinator (see the doc above), so there is no
	// partition gate to consult.
	//
	// GROUPED query (spec.GroupBy != ""): the coordinator groups the global ordered pool
	// ONCE (VectorQueryGrouped, P>1==P1) and re-encodes the groups + degraded/missing
	// trailer so every reader (the VectorQuery RPC, HTTP /query, the networked client)
	// decodes one uniform grouped shape via DecodeGroupsDegraded — mirroring how the flat
	// path re-encodes a flat result. The flat (no group_by) path below is unchanged.
	if spec.GroupBy != "" {
		groups, meta, gerr := f.e.VectorQueryGrouped(context.Background(), coll, specBytes, spec,
			ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
		if gerr != nil {
			return nil, gerr
		}
		return ops.EncodeGroupsDegraded(groups, meta.Degraded, missingU16(meta.Missing)), nil
	}
	res, meta, err := f.e.VectorQuery(context.Background(), coll, specBytes, spec,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	// res is []VectorResult (== []vector.Result, an alias); encode as a flat
	// mode-tagged result with the degraded/missing trailer so the dedicated
	// VectorQuery RPC (and the networked client) surface FanMeta — mirroring
	// fanHybrid's EncodeHybridResultsDegraded.
	return ops.EncodeQueryResultFusedDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (f *fanoutDispatcher) fanScroll(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.ok {
		return f.inner.Call(name, args)
	}
	if !r.partitioned {
		// Unpartitioned: the plain per-shard handler returns docs only (it has no
		// global view to compute next_cursor). Wrap its docs with the dispatcher's
		// next_cursor so the client-facing result wire is uniform with the
		// partitioned path. next_cursor uses the shared full-page rule (== limit).
		body, err := f.inner.Call(name, args)
		if err != nil {
			return nil, err
		}
		_, _, limit, _, _, _, _, order, derr := ops.DecodeScrollArgsOrder(args)
		if derr != nil {
			return nil, derr
		}
		docs, err := ops.DecodeVectorDocs(body)
		if err != nil {
			return nil, err
		}
		return ops.EncodeScrollResult(docs, false, nil, scrollNextCursorOrder(docs, limit, scrollOrderByFromOps(order))), nil
	}
	coll, filter, limit, rc, opa, afterID, hasAfter, order, err := ops.DecodeScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	// Re-encode the decoded cursor into opts.Cursor so the coordinator's fan-out
	// merge sends the SAME global cursor to every partition. For order_by this is the
	// v2 (value, id) token; for id-scroll it is the v1 (id) token.
	var cursor string
	ob := scrollOrderByFromOps(order)
	if order != nil {
		cursor = reencodeScrollCursor(order, afterID)
	} else if hasAfter {
		cursor = ops.EncodeScrollCursor(afterID)
	}
	bound, _ := ops.ReadStalenessOf(name, args)
	res, meta, nextCursor, err := f.e.VectorScroll(context.Background(), coll, filter, limit,
		VectorScrollOpts{Cursor: cursor, ReadConsistency: rc, OnPartitionUnavailable: opa, OrderBy: ob, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	// next_cursor on the result wire is server-authoritative: the coordinator
	// computed the global page, so the client decodes this rather than re-deriving.
	return ops.EncodeScrollResult(res, meta.Degraded, missingU16(meta.Missing), nextCursor), nil
}

func (f *fanoutDispatcher) fanDeleteByFilter(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, filter, err := ops.DecodeDeleteByFilterArgs(args)
	if err != nil {
		return nil, err
	}
	n, err := f.e.VectorDeleteByFilter(context.Background(), coll, filter)
	if err != nil {
		return nil, err
	}
	// Match handleVectorDeleteByFilter (ops/builtin.go): 4-byte big-endian count.
	out := make([]byte, 4)
	binary.BigEndian.PutUint32(out, uint32(n)) //nolint:gosec // count >= 0
	return out, nil
}

func (f *fanoutDispatcher) fanInsert(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, vec, ttl, meta, sparse, _, err := ops.DecodeVectorInsertArgs(args)
	if err != nil {
		return nil, err
	}
	opts := VectorInsertOpts{TTL: ttl, Metadata: meta}
	if sparse != nil {
		opts.Sparse = *sparse // VectorSparse aliases vector.SparseVector
	}
	if err := f.e.VectorInsertExt(context.Background(), coll, id, vec, opts); err != nil {
		return nil, err
	}
	// handleVectorInsert returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanUpsert(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	// Upsert reuses the insert-args wire shape; content rides in metadata
	// ($content), so the single-shard handler (handleVectorUpsert) calls Upsert
	// with an empty content string and the content carried in meta — mirror that.
	coll, id, vec, ttl, meta, sparse, _, err := ops.DecodeVectorInsertArgs(args)
	if err != nil {
		return nil, err
	}
	opts := VectorInsertOpts{TTL: ttl, Metadata: meta}
	if sparse != nil {
		opts.Sparse = *sparse
	}
	if err := f.e.VectorUpsert(context.Background(), coll, id, vec, "", opts); err != nil {
		return nil, err
	}
	// handleVectorUpsert returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanDelete(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, err := ops.DecodeVectorDeleteArgs(args)
	if err != nil {
		return nil, err
	}
	ok, err := f.e.VectorDelete(context.Background(), coll, id)
	if err != nil {
		return nil, err
	}
	// Match handleVectorDelete (ops/builtin.go): []byte{1} deleted, []byte{0} not.
	if ok {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// getProjection splits the get-op flags byte into the with_vector / with_payload
// booleans, mirroring store.go getFlags / ops' getFlag* bits. Shared by the
// dense/named/MV get fan-out handlers.
func getProjection(flags uint8) (withVector, withPayload bool) {
	return flags&ops.GetFlagWithVector != 0, flags&ops.GetFlagWithPayload != 0
}

// fanGet routes a dense get-by-id to the ONE owning physical partition (a point
// lives in exactly one partition by id-hash). The not-found flag returned from
// that partition is genuine — this is route-by-id, NOT a scatter-union. Get is
// read-only (live gen only), so it does not dual-write during a reshard.
func (f *fanoutDispatcher) fanGet(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, flags, rc, opa, bound, err := ops.DecodeVectorGetArgsOpts(args)
	if err != nil {
		return nil, err
	}
	withVector, withPayload := getProjection(flags)
	// Thread rc through to the embedded coordinator so a Linearizable get on a
	// partitioned collection routes to the owning partition's LEADER and arms the
	// shard readIndex barrier — without this the rc would be silently dropped on
	// the fan-out path and the Linearizable get would serve stale.
	found, vec, meta, ttl, sparse, err := f.e.VectorGetExt(context.Background(), coll, id, withVector, withPayload,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorGetResult(found, vec, meta, ttl, sparse, withVector, withPayload), nil
}

// fanGetBatch is the coordinator side of a batch get. For an unpartitioned
// collection it passes through to the single shard (f.inner.Call). For a
// partitioned collection it decodes (collection, ids, flags), delegates to
// embedded.VectorGetBatch (which scatters the ids to their owning partitions,
// asking each ONLY for its subset, and merges), then RE-ENCODES the unified
// points + missing as a vector_get_batch result so the networked client decodes
// exactly what the per-partition handler shape produces. A partial miss is
// normal: absent ids come back as found=0 rows, never an error.
func (f *fanoutDispatcher) fanGetBatch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, ids, flags, err := ops.DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVector, withPayload := getProjection(flags)
	points, missing, err := f.e.VectorGetBatch(context.Background(), coll, ids, withVector, withPayload)
	if err != nil {
		return nil, err
	}
	rows := make([]ops.GetBatchRow, 0, len(points)+len(missing))
	for _, p := range points {
		rows = append(rows, ops.GetBatchRow{
			ID:     p.ID,
			Found:  true,
			Vec:    p.Vec,
			Meta:   p.Meta,
			TTLMs:  uint64(p.TTL.Milliseconds()), //nolint:gosec // TTL >= 0
			Sparse: p.Sparse,
		})
	}
	for _, id := range missing {
		rows = append(rows, ops.GetBatchRow{ID: id, Found: false})
	}
	return ops.EncodeVectorGetBatchResult(rows), nil
}

// fanOperate applies an operate op-list to a record inside a point's payload on
// the owning physical partition. It is the fan-out row for all three families:
// one wire shape (EncodeVectorOperateArgs), one decoder, and the op NAME picks
// which embedded method runs — so the three handlers below differ only in that
// call and cannot drift apart in their CAS handling.
//
// The CAS precondition is REBUILT into a WriteOpts and handed to the embedded
// method, which re-encodes it into the physical-partition args. fanSetPayload
// decodes with ops.DecodeSetPayloadArgsOpts, the NON-CAS decoder, so a
// partitioned set_payload silently loses the guard between the client and the
// partition. Nothing else in this file can restore it once it is dropped here,
// which is why the decode is the CAS-carrying one.
//
// It cannot recurse: resolveRoute short-circuits names containing '#'/'@' to
// partitioned=false, and the embedded method substitutes the physical partition
// name before its own Call.
func (f *fanoutDispatcher) fanOperate(name string, args []byte, r fanRoute) ([]byte, error) {
	return f.fanOperateVia(name, args, r, f.e.VectorOperate)
}

// fanNamedOperate is fanOperate for the named family. See fanOperate.
func (f *fanoutDispatcher) fanNamedOperate(name string, args []byte, r fanRoute) ([]byte, error) {
	return f.fanOperateVia(name, args, r, f.e.VectorNamedOperate)
}

// fanMVOperate is fanOperate for the multi-vector family. See fanOperate.
func (f *fanoutDispatcher) fanMVOperate(name string, args []byte, r fanRoute) ([]byte, error) {
	return f.fanOperateVia(name, args, r, f.e.VectorMVOperate)
}

// operateFn is the shape the three embedded operate methods share.
type operateFn func(ctx context.Context, coll string, id uint64, payloadKey string, a *wire.OperateArgs, opts ...WriteOpts) (bool, *wire.OperateResult, error)

// fanOperateVia is the shared body of the three operate fan handlers.
func (f *fanoutDispatcher) fanOperateVia(name string, args []byte, r fanRoute, call operateFn) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, pk, a, exp, hasExp, err := ops.DecodeVectorOperateArgs(args)
	if err != nil {
		return nil, err
	}
	var wo WriteOpts
	if hasExp {
		v := exp
		wo.ExpectedVersion = &v // WriteOpts carries the precondition as a *uint64
	}
	found, res, err := call(context.Background(), coll, id, pk, a, wo)
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorOperateResult(found, res)
}

// fanSetPayload merges a payload patch on the owning physical partition. Like
// VectorDelete it dual-writes during a reshard (the embedded method threads
// through dualTargets), so the new gen is not stale after cutover.
func (f *fanoutDispatcher) fanSetPayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, meta, keyTTLMs, err := ops.DecodeSetPayloadArgsOpts(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorSetPayload(context.Background(), coll, id, meta, keyTTLMs)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanOverwritePayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, meta, keyTTLMs, err := ops.DecodeSetPayloadArgsOpts(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorOverwritePayload(context.Background(), coll, id, meta, keyTTLMs)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanDeletePayloadKeys(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, keys, err := ops.DecodeDeletePayloadKeysArgs(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorDeletePayloadKeys(context.Background(), coll, id, keys)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanClearPayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, err := ops.DecodeClearPayloadArgs(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorClearPayload(context.Background(), coll, id)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

// guardAliasShadow rejects a create whose name collides with an existing alias
// (data-plane ops would otherwise be ambiguous: alias resolution vs the real
// collection). The PARTITIONED create path gets this guard inside
// embedded.CreateCollection / VectorMVCreateCollection /
// VectorNamedCreateCollection, but the SINGLE-PARTITION branch passes straight
// to f.inner.Call — which has no alias knowledge — so the guard must be applied
// here too, mirroring the embedded guard's message + sentinel. Physical-partition
// creates carry reserved '#'/'@' names (which can never be aliases) and are
// skipped so coordinator fan-out is unaffected.
func (f *fanoutDispatcher) guardAliasShadow(coll string) error {
	if strings.ContainsAny(coll, "#@") {
		return nil
	}
	if _, ok := f.e.catalog.ResolveAlias(coll); ok {
		return fmt.Errorf("vector: collection name %q is already an alias: %w", coll, ErrAliasShadowsCollection)
	}
	return nil
}

func (f *fanoutDispatcher) fanCreateCollection(name string, args []byte, _ fanRoute) ([]byte, error) {
	// Unlike the other handlers, this one can't cheap-peek the collection name to
	// decide routing: it gates on cfg.Partitions (not the name), which is only
	// available after a full decode — so it always decodes up front.
	coll, cfg, err := ops.DecodeCreateCollectionArgs(args)
	if err != nil {
		return nil, err
	}
	// Single-partition creates (cfg.Partitions<=1) take the passthrough path,
	// byte-identical to the non-decorated server: this covers both a user's plain
	// single-partition collection and the physical-partition creates a coordinator
	// emits during CreateCollection fan-out (those carry the reserved '#'/'@' in
	// their names and reset Partitions to 0, so re-routing them through
	// CreateCollection would trip its name guard).
	if cfg.Partitions <= 1 {
		if err := f.guardAliasShadow(coll); err != nil {
			return nil, err
		}
		return f.inner.Call(name, args)
	}
	// Partitioned logical creates route through the embedded backend so a remote
	// create gets the #/@ name guard, physical-partition creation, and catalog write.
	if err := f.e.CreateCollection(context.Background(), coll, cfg); err != nil {
		return nil, err
	}
	// handleVectorCreateCollection returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanDropCollection(name string, args []byte, _ fanRoute) ([]byte, error) {
	peek, ok := ops.CollectionNameFor(name, args)
	if !ok {
		return f.inner.Call(name, args)
	}
	// The shared fanRoute is deliberately IGNORED here (drop is not in
	// dataPlaneAliasOps, so it arrives zeroed anyway): a drop must act on the REAL
	// name, and the route's coll is alias-RESOLVED.
	//
	// Physical-partition drops (reserved '#'/'@' names) carry an explicit physical
	// name and must NOT trigger an alias cascade (they are internal, never a
	// user-facing collection). They also never resolve. partitioned() returns
	// false for them (the '#'/'@' short-circuit), so they take the pass-through
	// branch below without a cascade. A bare user drop of an unpartitioned
	// collection is the only pass-through that should cascade.
	physical := strings.ContainsAny(peek, "#@")
	// Drop is an ADMIN op — it must act on the REAL collection name, NOT resolve an
	// alias (dropping the alias "prod" must not drop the underlying "real"). The
	// partition decision therefore reads the catalog DIRECTLY for peek (bypassing
	// f.partitioned(), which resolves aliases) so an alias name is never mistaken
	// for the partitioned target. The cascade then removes aliases that targeted
	// the just-dropped real collection.
	if P, gen, ok := f.e.catalog.PartitionsGen(peek); ok && P > 1 {
		if err := f.e.dropCollectionFanout(context.Background(), peek, P, gen); err != nil {
			return nil, err
		}
		f.e.cleanupAliasesFor(context.Background(), peek)
		// handleVectorDropCollection returns a nil body.
		return nil, nil
	}
	body, err := f.inner.Call(name, args)
	if err != nil {
		return nil, err
	}
	if !physical {
		f.e.cleanupAliasesFor(context.Background(), peek)
	}
	return body, nil
}

func (f *fanoutDispatcher) fanMVCreate(name string, args []byte, _ fanRoute) ([]byte, error) {
	// Like fanCreateCollection, this can't cheap-peek the collection name to decide
	// routing: it gates on cfg.Partitions (not the name), which is only available
	// after a full decode — so it always decodes up front.
	coll, cfg, err := ops.DecodeMVCreateArgs(args)
	if err != nil {
		return nil, err
	}
	// Single-partition creates (cfg.Partitions<=1) take the passthrough path,
	// byte-identical to the non-decorated server: this covers both a user's plain
	// single-partition MV collection and the physical-partition creates the
	// coordinator emits during VectorMVCreateCollection fan-out (those carry the
	// reserved '#'/'@' in their names and reset Partitions to 0, so re-routing them
	// through VectorMVCreateCollection would trip its name guard).
	if cfg.Partitions <= 1 {
		if err := f.guardAliasShadow(coll); err != nil {
			return nil, err
		}
		return f.inner.Call(name, args)
	}
	// Partitioned logical creates route through the embedded backend so a remote
	// create gets the #/@ name guard, physical-partition creation, and catalog write.
	if err := f.e.VectorMVCreateCollection(context.Background(), coll, cfg); err != nil {
		return nil, err
	}
	// handleMVCreate returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanMVAdd(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, tokens, meta, err := ops.DecodeMVAddArgs(args)
	if err != nil {
		return nil, err
	}
	if err := f.e.VectorMVAdd(context.Background(), coll, docID, tokens, meta); err != nil {
		return nil, err
	}
	// handleMVAdd returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanMVDelete(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, err := ops.DecodeMVDeleteArgs(args)
	if err != nil {
		return nil, err
	}
	ok, err := f.e.VectorMVDelete(context.Background(), coll, docID)
	if err != nil {
		return nil, err
	}
	// Match handleMVDelete (ops/multivector.go): []byte{1} deleted, []byte{0} not.
	if ok {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// fanMVGet routes an MV get-by-id to the one owning physical partition. Like the
// dense get, the not-found flag is genuine (route-by-id) and the read is live-gen
// only.
func (f *fanoutDispatcher) fanMVGet(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, flags, rc, opa, bound, err := ops.DecodeVectorGetArgsOpts(args)
	if err != nil {
		return nil, err
	}
	withVector, withPayload := getProjection(flags)
	// Thread rc through (see fanGet): a Linearizable MV get must route to the
	// owning partition's leader + arm the barrier, not silently drop rc.
	found, tokens, meta, err := f.e.VectorMVGetExt(context.Background(), coll, docID, withVector, withPayload,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeMVGetResult(found, tokens, meta, withVector, withPayload), nil
}

// fanMVGetBatch is the coordinator side of an MV batch get. For an unpartitioned
// collection it passes through to the single shard. For a partitioned collection
// it decodes (collection, ids, flags), delegates to embedded.VectorMVGetBatch
// (which scatters the ids to their owning partitions, asking each ONLY for its
// subset, and merges), then RE-ENCODES the unified points + missing as a
// vector_mv_get_batch result so the networked client decodes exactly what the
// per-partition handler shape produces. A partial miss is normal: absent ids come
// back as found=0 rows, never an error. MV has NO ttl. The MV clone of
// fanNamedGetBatch.
func (f *fanoutDispatcher) fanMVGetBatch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, ids, flags, err := ops.DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVector, withPayload := getProjection(flags)
	points, missing, err := f.e.VectorMVGetBatch(context.Background(), coll, ids, withVector, withPayload)
	if err != nil {
		return nil, err
	}
	rows := make([]ops.MVGetBatchRow, 0, len(points)+len(missing))
	for _, p := range points {
		rows = append(rows, ops.MVGetBatchRow{
			ID:     p.ID,
			Found:  true,
			Tokens: p.Tokens,
			Meta:   p.Meta,
		})
	}
	for _, id := range missing {
		rows = append(rows, ops.MVGetBatchRow{ID: id, Found: false})
	}
	return ops.EncodeMVGetBatchResult(rows), nil
}

func (f *fanoutDispatcher) fanMVSetPayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, meta, keyTTLMs, err := ops.DecodeSetPayloadArgsOpts(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorMVSetPayload(context.Background(), coll, docID, meta, keyTTLMs)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanMVOverwritePayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, meta, keyTTLMs, err := ops.DecodeSetPayloadArgsOpts(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorMVOverwritePayload(context.Background(), coll, docID, meta, keyTTLMs)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanMVDeletePayloadKeys(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, keys, err := ops.DecodeDeletePayloadKeysArgs(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorMVDeletePayloadKeys(context.Background(), coll, docID, keys)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanMVClearPayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, docID, err := ops.DecodeClearPayloadArgs(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorMVClearPayload(context.Background(), coll, docID)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanMVSearch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, query, k, cand, rc, opa, bound, err := ops.DecodeMVSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorMVSearch(context.Background(), coll, query, k,
		MultiSearchOpts{CandidatesPerToken: cand, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeMVResultsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (f *fanoutDispatcher) fanMVScroll(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.ok {
		return f.inner.Call(name, args)
	}
	if !r.partitioned {
		// Unpartitioned: the plain per-shard handler returns docs only (no global
		// view to compute next_cursor). Wrap its docs with the dispatcher's
		// next_cursor so the client-facing result wire is uniform with the
		// partitioned path (MV scroll has no degraded/missing trailer of its own ⇒
		// degraded=false, missing=nil). next_cursor uses the shared full-page rule.
		body, err := f.inner.Call(name, args)
		if err != nil {
			return nil, err
		}
		_, _, limit, _, _, _, _, order, derr := ops.DecodeMVScrollArgsOrder(args)
		if derr != nil {
			return nil, derr
		}
		docs, err := ops.DecodeVectorDocs(body)
		if err != nil {
			return nil, err
		}
		return ops.EncodeScrollResult(docs, false, nil, scrollNextCursorOrder(docs, limit, scrollOrderByFromOps(order))), nil
	}
	coll, filter, limit, rc, opa, afterID, hasAfter, order, err := ops.DecodeMVScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	// Re-encode the decoded cursor into opts.Cursor so the coordinator's fan-out merge
	// sends the SAME global cursor to every partition. For order_by this is the v2
	// (value, id) token; for id-scroll it is the v1 (id) token.
	var cursor string
	ob := scrollOrderByFromOps(order)
	if order != nil {
		cursor = reencodeScrollCursor(order, afterID)
	} else if hasAfter {
		cursor = ops.EncodeScrollCursor(afterID)
	}
	bound, _ := ops.ReadStalenessOf(name, args)
	res, meta, nextCursor, err := f.e.VectorMVScrollExt(context.Background(), coll, filter, limit, cursor,
		MVScrollOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, OrderBy: ob, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	// Server-authoritative next_cursor: the coordinator computed the global page, so
	// the client decodes this rather than re-deriving.
	return ops.EncodeScrollResult(res, meta.Degraded, missingU16(meta.Missing), nextCursor), nil
}

func (f *fanoutDispatcher) fanMVDrop(name string, args []byte, _ fanRoute) ([]byte, error) {
	peek, ok := ops.CollectionNameFor(name, args)
	if !ok {
		return f.inner.Call(name, args)
	}
	// The shared fanRoute is IGNORED (see fanDropCollection): drop is a real-name op.
	// ADMIN op: the partition decision reads the catalog DIRECTLY for peek (NOT
	// f.partitioned, which resolves aliases) — drop acts on the real name, never an
	// alias target. The cascade then clears aliases that pointed at the dropped
	// collection. Physical-partition drops (reserved '#'/'@' names) are internal
	// and never cascade.
	physical := strings.ContainsAny(peek, "#@")
	if P, gen, ok := f.e.catalog.PartitionsGen(peek); ok && P > 1 {
		if err := f.e.mvDropCollectionFanout(context.Background(), peek, P, gen); err != nil {
			return nil, err
		}
		f.e.cleanupAliasesFor(context.Background(), peek)
		// handleMVDrop returns a nil body.
		return nil, nil
	}
	body, err := f.inner.Call(name, args)
	if err != nil {
		return nil, err
	}
	if !physical {
		f.e.cleanupAliasesFor(context.Background(), peek)
	}
	return body, nil
}

// fanNamedCreate intercepts a partitioned named-collection create. Like
// fanCreateCollection / fanMVCreate it can't cheap-peek to decide routing: it
// gates on the create payload's partitions count (not the name), only available
// after a full decode, so it always decodes up front.
func (f *fanoutDispatcher) fanNamedCreate(name string, args []byte, _ fanRoute) ([]byte, error) {
	coll, cfg, partitions, err := ops.DecodeNamedCreateArgs(args)
	if err != nil {
		return nil, err
	}
	// Single-partition creates (partitions<=1) take the passthrough path,
	// byte-identical to the non-decorated server: this covers both a user's plain
	// single-partition named collection and the physical-partition creates the
	// coordinator emits during VectorNamedCreateCollection fan-out (those carry the
	// reserved '#'/'@' in their names and have partitions reset to 0, so re-routing
	// them through VectorNamedCreateCollection would trip its name guard).
	if partitions <= 1 {
		if err := f.guardAliasShadow(coll); err != nil {
			return nil, err
		}
		return f.inner.Call(name, args)
	}
	// Partitioned logical creates route through the embedded backend so a remote
	// create gets the #/@ name guard, physical-partition creation, and catalog write.
	if err := f.e.VectorNamedCreateCollection(context.Background(), coll, cfg, partitions); err != nil {
		return nil, err
	}
	// handleNamedCreate returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanNamedInsert(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, vectors, sparseVectors, payload, ttl, _, _, _, err := ops.DecodeNamedInsertArgsSparseKeyTTL(args)
	if err != nil {
		return nil, err
	}
	if err := f.e.VectorNamedInsertSparse(context.Background(), coll, id, vectors, sparseVectors, payload, ttl); err != nil {
		return nil, err
	}
	// handleNamedInsert returns a nil body.
	return nil, nil
}

func (f *fanoutDispatcher) fanNamedDelete(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, err := ops.DecodeNamedDeleteArgs(args)
	if err != nil {
		return nil, err
	}
	ok, err := f.e.VectorNamedDelete(context.Background(), coll, id)
	if err != nil {
		return nil, err
	}
	// Match handleNamedDelete: []byte{1} deleted, []byte{0} not.
	if ok {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// fanNamedGet routes a named get-by-id to the one owning physical partition. The
// not-found flag is genuine (route-by-id); the read is live-gen only.
func (f *fanoutDispatcher) fanNamedGet(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, flags, rc, opa, bound, err := ops.DecodeVectorGetArgsOpts(args)
	if err != nil {
		return nil, err
	}
	withVector, withPayload := getProjection(flags)
	// Thread rc through (see fanGet): a Linearizable named get must route to the
	// owning partition's leader + arm the barrier, not silently drop rc.
	found, vectors, meta, ttl, err := f.e.VectorNamedGetExt(context.Background(), coll, id, withVector, withPayload,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeNamedGetResult(found, vectors, meta, ttl, withVector, withPayload), nil
}

// fanNamedGetConfig handles vector_named_get_config. The named config is logical
// (identical on every physical partition), so there is no fan-out. For rc==0
// (AnyReplica) it passes straight through to the inner shard — byte-identical to
// the legacy path. For a LeaderOnly/Linearizable read it routes through the
// embedded VectorNamedGetConfigExt so the meta-catalog read barrier
// (resolveCollectionForRead) arms FIRST — a Linearizable get_config sees a
// just-created / just-reconfigured collection — then the shard readIndex barrier
// arms on the leader, and the result is re-encoded for the wire.
func (f *fanoutDispatcher) fanNamedGetConfig(name string, args []byte, _ fanRoute) ([]byte, error) {
	coll, rc, opa, bound, err := ops.DecodeNamedNameArgsOpts(args)
	if err != nil {
		return nil, err
	}
	if rc < ops.ConsistencyLeaderOnly {
		return f.inner.Call(name, args) // legacy byte-identical AnyReplica path
	}
	cfg, err := f.e.VectorNamedGetConfigExt(context.Background(), coll,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeNamedConfigResult(cfg), nil
}

// fanNamedGetBatch is the coordinator side of a named batch get. For an
// unpartitioned collection it passes through to the single shard. For a
// partitioned collection it decodes (collection, ids, flags), delegates to
// embedded.VectorNamedGetBatch (which scatters the ids to their owning
// partitions, asking each ONLY for its subset, and merges), then RE-ENCODES the
// unified points + missing as a vector_named_get_batch result so the networked
// client decodes exactly what the per-partition handler shape produces. A partial
// miss is normal: absent ids come back as found=0 rows, never an error. The named
// clone of fanGetBatch.
func (f *fanoutDispatcher) fanNamedGetBatch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, ids, flags, err := ops.DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVector, withPayload := getProjection(flags)
	points, missing, err := f.e.VectorNamedGetBatch(context.Background(), coll, ids, withVector, withPayload)
	if err != nil {
		return nil, err
	}
	rows := make([]ops.NamedGetBatchRow, 0, len(points)+len(missing))
	for _, p := range points {
		rows = append(rows, ops.NamedGetBatchRow{
			ID:      p.ID,
			Found:   true,
			Vectors: p.Vectors,
			Meta:    p.Meta,
			TTLMs:   uint64(p.TTL.Milliseconds()), //nolint:gosec // TTL >= 0
		})
	}
	for _, id := range missing {
		rows = append(rows, ops.NamedGetBatchRow{ID: id, Found: false})
	}
	return ops.EncodeNamedGetBatchResult(rows), nil
}

func (f *fanoutDispatcher) fanNamedSetPayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, meta, keyTTLMs, err := ops.DecodeSetPayloadArgsOpts(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorNamedSetPayload(context.Background(), coll, id, meta, keyTTLMs)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanNamedOverwritePayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, meta, keyTTLMs, err := ops.DecodeSetPayloadArgsOpts(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorNamedOverwritePayload(context.Background(), coll, id, meta, keyTTLMs)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanNamedDeletePayloadKeys(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, keys, err := ops.DecodeDeletePayloadKeysArgs(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorNamedDeletePayloadKeys(context.Background(), coll, id, keys)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanNamedClearPayload(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, id, err := ops.DecodeClearPayloadArgs(args)
	if err != nil {
		return nil, err
	}
	applied, err := f.e.VectorNamedClearPayload(context.Background(), coll, id)
	if err != nil {
		return nil, err
	}
	return ops.EncodePayloadResult(applied), nil
}

func (f *fanoutDispatcher) fanNamedSearch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, vecName, query, k, filter, rc, opa, bound, err := ops.DecodeNamedSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	res, err := f.e.VectorNamedSearchExt(context.Background(), coll, vecName, query, k,
		NamedSearchOpts{Filter: filter, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorSearchResults(res), nil
}

func (f *fanoutDispatcher) fanNamedSparseSearch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, space, query, k, filter, rc, opa, bound, err := ops.DecodeNamedSparseSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	res, err := f.e.VectorNamedSparseSearchExt(context.Background(), coll, space, query, k,
		NamedSearchOpts{Filter: filter, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	// Sparse results carry a Score (the dot product); re-encode with the
	// score-carrying hybrid-results codec, matching the per-partition handler.
	return ops.EncodeHybridResults(res), nil
}

// fanNamedHybridSearch is the coordinator side of the cross-space named hybrid: it
// decodes the fused-search args and calls VectorNamedHybridSearch, which (for P>1)
// fans the UNFUSED-lanes leaf op (vector_named_hybrid_lanes) to every partition and
// fuses once globally. The fused result carries Score, so it re-encodes with the
// score-carrying hybrid-results codec, matching the per-partition fused handler.
func (f *fanoutDispatcher) fanNamedHybridSearch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, denseSpace, denseQ, sparseSpace, sparseQ, k, opts, rc, opa, bound, err := ops.DecodeNamedHybridArgs(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorNamedHybridSearchExt(context.Background(), coll, denseSpace, denseQ, sparseSpace, sparseQ, k,
		NamedHybridOpts{
			Filter: opts.Filter, Method: opts.Method, Alpha: opts.Alpha, RRFK: opts.RRFK,
			DenseK: opts.DenseK, SparseK: opts.SparseK, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound,
		})
	if err != nil {
		return nil, err
	}
	// Carry the degraded/missing trailer so the networked client surfaces FanMeta,
	// matching fanHybrid. Byte-identical to EncodeHybridResults when not degraded, so
	// readers that decode with the plain codec (httpapi/grpcapi) are unaffected.
	return ops.EncodeHybridResultsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// fanNamedQuery is the coordinator side of the named-collection Query API
// (vector_named_query). It mirrors fanQuery exactly for the named family: it
// decodes (collection, specBytes, engine spec, rc/opa/bound) and delegates to
// embedded.VectorNamedQuery for BOTH the partitioned (P>1: fan vector_named_query
// to every partition + merge per mode via namedQueryFanOut) AND the unpartitioned
// (P<=1: run on the read leader + merge the single shard's result through the SAME
// mode merge) paths, then RE-ENCODES the FLAT fused/reranked top-k + degraded/
// missing trailer so every reader (the future NamedVectorQuery RPC, the HTTP route,
// the networked client) decodes one uniform flat shape via DecodeQueryResultDegraded.
//
// Why NOT pass the unpartitioned case straight through: the per-shard handler
// returns a mode-tagged payload whose FUSION variant carries the UNFUSED prefetch
// lanes (Lanes, not Fused) — never a flat top-k. embedded.VectorNamedQuery runs the
// local fusion merge for the single shard so the wire is always flat (see fanQuery).
func (f *fanoutDispatcher) fanNamedQuery(name string, args []byte, _ fanRoute) ([]byte, error) {
	coll, specBytes, spec, rc, opa, bound, err := ops.DecodeQuerySpecArgs(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorNamedQuery(context.Background(), coll, specBytes, spec,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeQueryResultFusedDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// fanMVQuery is the coordinator side of the MV-collection Query API
// (vector_mv_query). It mirrors fanNamedQuery exactly for the MV family: it decodes
// (collection, specBytes, engine spec, rc/opa/bound) and delegates to
// embedded.VectorMVQuery for BOTH the partitioned (P>1: fan vector_mv_query to every
// partition + merge per mode via mvQueryFanOut) AND the unpartitioned (P<=1: run on
// the read leader + merge the single shard's result through the SAME orientation-aware
// mode merge) paths, then RE-ENCODES the FLAT fused/reranked top-k + degraded/missing
// trailer so every reader (the future MVVectorQuery RPC, the HTTP route, the
// networked client) decodes one uniform flat shape via DecodeQueryResultDegraded.
//
// Why NOT pass the unpartitioned case straight through: the per-shard handler returns
// a mode-tagged payload whose FUSION variant carries the UNFUSED prefetch lanes
// (Lanes, not Fused) — never a flat top-k. embedded.VectorMVQuery runs the local
// fusion merge for the single shard so the wire is always flat (see fanNamedQuery).
func (f *fanoutDispatcher) fanMVQuery(name string, args []byte, _ fanRoute) ([]byte, error) {
	coll, specBytes, spec, rc, opa, bound, err := ops.DecodeQuerySpecArgs(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorMVQuery(context.Background(), coll, specBytes, spec,
		ReadOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeQueryResultFusedDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

// fanMVHybridSearch is the coordinator side of the MV cross-modality hybrid: it
// decodes the fused-search args and calls VectorMVHybridSearch, which (for P>1) fans
// the UNFUSED-lanes leaf op (vector_mv_hybrid_lanes) to every partition and fuses
// once globally. The fused result carries Score, so it re-encodes with the
// score-carrying hybrid-results codec, matching the per-partition fused handler.
func (f *fanoutDispatcher) fanMVHybridSearch(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, query, sparseQ, k, opts, rc, opa, bound, err := ops.DecodeMVHybridArgs(args)
	if err != nil {
		return nil, err
	}
	res, meta, err := f.e.VectorMVHybridSearchExt(context.Background(), coll, query, sparseQ, k,
		MVHybridOpts{
			Filter: opts.Filter, Method: opts.Method, Alpha: opts.Alpha, RRFK: opts.RRFK,
			DenseK: opts.DenseK, SparseK: opts.SparseK, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound,
		})
	if err != nil {
		return nil, err
	}
	// Carry the degraded/missing trailer so the networked client surfaces FanMeta,
	// matching fanHybrid. Byte-identical to EncodeHybridResults when not degraded, so
	// readers that decode with the plain codec (httpapi/grpcapi) are unaffected.
	return ops.EncodeHybridResultsDegraded(res, meta.Degraded, missingU16(meta.Missing)), nil
}

func (f *fanoutDispatcher) fanNamedDocs(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.partitioned {
		return f.inner.Call(name, args)
	}
	coll, vecName, query, k, filter, rc, opa, bound, err := ops.DecodeNamedSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	docs, err := f.e.VectorNamedSearchDocsExt(context.Background(), coll, vecName, query, k,
		NamedSearchOpts{Filter: filter, ReadConsistency: rc, OnPartitionUnavailable: opa, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	return ops.EncodeVectorDocs(docs), nil
}

func (f *fanoutDispatcher) fanNamedScroll(name string, args []byte, r fanRoute) ([]byte, error) {
	if !r.ok {
		return f.inner.Call(name, args)
	}
	if !r.partitioned {
		// Unpartitioned: wrap the plain handler's docs with the dispatcher's
		// next_cursor (named has no degraded/missing trailer ⇒ degraded=false,
		// missing=nil), uniform with the partitioned path.
		body, err := f.inner.Call(name, args)
		if err != nil {
			return nil, err
		}
		_, _, limit, _, _, _, _, order, derr := ops.DecodeNamedScrollArgsOrder(args)
		if derr != nil {
			return nil, derr
		}
		docs, err := ops.DecodeVectorDocs(body)
		if err != nil {
			return nil, err
		}
		return ops.EncodeScrollResult(docs, false, nil, scrollNextCursorOrder(docs, limit, scrollOrderByFromOps(order))), nil
	}
	coll, filter, limit, afterID, hasAfter, rc, opa, order, err := ops.DecodeNamedScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	// Re-encode the decoded cursor into opts.Cursor so the named fan-out merge sends
	// the SAME global cursor to every partition. For order_by this is the v2 (value, id)
	// token; for id-scroll it is the v1 (id) token.
	var cursor string
	ob := scrollOrderByFromOps(order)
	if order != nil {
		cursor = reencodeScrollCursor(order, afterID)
	} else if hasAfter {
		cursor = ops.EncodeScrollCursor(afterID)
	}
	bound, _ := ops.ReadStalenessOf(name, args)
	docs, nextCursor, err := f.e.VectorNamedScrollExt(context.Background(), coll, filter, limit, cursor,
		NamedScrollOpts{ReadConsistency: rc, OnPartitionUnavailable: opa, OrderBy: ob, MaxStaleness: bound})
	if err != nil {
		return nil, err
	}
	// Server-authoritative next_cursor (named has no degraded/missing).
	return ops.EncodeScrollResult(docs, false, nil, nextCursor), nil
}

func (f *fanoutDispatcher) fanNamedDrop(name string, args []byte, _ fanRoute) ([]byte, error) {
	peek, ok := ops.CollectionNameFor(name, args)
	if !ok {
		return f.inner.Call(name, args)
	}
	// The shared fanRoute is IGNORED (see fanDropCollection): drop is a real-name op.
	// ADMIN op: see fanMVDrop. Real-name partition decision (no alias resolution) +
	// best-effort alias cascade; physical-partition drops never cascade.
	physical := strings.ContainsAny(peek, "#@")
	if P, gen, ok := f.e.catalog.PartitionsGen(peek); ok && P > 1 {
		if err := f.e.namedDropCollectionFanout(context.Background(), peek, P, gen); err != nil {
			return nil, err
		}
		f.e.cleanupAliasesFor(context.Background(), peek)
		// handleNamedDrop returns a nil body.
		return nil, nil
	}
	body, err := f.inner.Call(name, args)
	if err != nil {
		return nil, err
	}
	if !physical {
		f.e.cleanupAliasesFor(context.Background(), peek)
	}
	return body, nil
}
