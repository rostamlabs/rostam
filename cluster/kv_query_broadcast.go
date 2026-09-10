// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// kvQueryOpName is the client-facing KV record search. It is KEYLESS, so
// Node.shardIndexFor returns 0 for it and Call would answer the whole query
// from group 0's slice of the keyspace; Call intercepts it and fans out
// instead. See broadcastKVQuery.
const kvQueryOpName = "kv_query"

// opKVQueryShardName is the INTERNAL, node-local wrapper used to drive ONE
// group's kv_query leaf on a peer, the shape __flush_shard__ gives the flush
// broadcast. It dispatches off n.adminOps — before op-registry routing — so the
// receiving node answers for exactly the requested group and does NOT
// re-broadcast.
//
// Sending kv_query itself over the wire would re-enter Node.Call on the peer and
// fan out again over every group IT knows about: a query would multiply by the
// shard count at every hop and, in a cluster whose nodes host
// overlapping-but-unequal shard subsets, need not terminate.
//
// It is enumerated in authz.adminOps rather than left to actionFor's
// deny-by-default fallthrough, for the reason __flush_shard__ records: absence
// from the ops registry is a coincidence a refactor can remove, not a
// classification. `kv_query` ITSELF is an ordinary OpReadOnly and is authorised
// as a read; the wrapper is pinned at admin so it can never be handed to a
// read:* key as a way to address one specific shard group.
const opKVQueryShardName = "__kv_query_shard__"

// kvQueryGroupTimeout bounds ONE group's leg of the fan-out, mirroring
// flushBroadcastGroupTimeout: the legs run concurrently, so a query costs the
// slowest group's latency rather than their sum, and a group whose owner accepts
// the connection and then goes silent costs one timeout instead of hanging the
// client's read forever.
//
// A var, not a const, so a test can shrink it; nothing in production writes it.
var kvQueryGroupTimeout = 10 * time.Second

// kvQueryLegHook is a TEST SEAM, nil in production: when set it runs at the top
// of every group's leg, which is how the parallelism and per-group-timeout tests
// make one group slow or silent without a second cluster. It follows the
// precedent of shard.SetReadServedHook and MetaFSM's leaseRenewObserver.
var kvQueryLegHook func(group int)

// encodeShardScopedKVQuery is the wire form of opKVQueryShardName: the target
// shard index as 4 big-endian bytes, followed by the UNCHANGED leaf args.
//
// The leaf args travel verbatim, cursor and all. The leaf reads only its own
// group's continuation out of that cursor (ops.contFor), so there is nothing to
// rewrite per group — and rewriting would be worse than useless: the read
// consistency byte inside these bytes is what makes shard.Store.Call run
// VerifyLeader for a linearizable query, so the frame the peer's shard sees must
// be byte-identical to the one the client sent.
func encodeShardScopedKVQuery(shardIdx int, leafArgs []byte) []byte {
	out := make([]byte, 4, 4+len(leafArgs))
	binary.BigEndian.PutUint32(out, uint32(shardIdx)) //nolint:gosec // bounded by NumShards at every call site
	return append(out, leafArgs...)
}

// decodeShardScopedKVQuery reads what encodeShardScopedKVQuery wrote. The leaf
// args are returned as a SUB-SLICE of the frame and are only read from, never
// retained.
func decodeShardScopedKVQuery(args []byte) (int, []byte, error) {
	if len(args) < 4 {
		return 0, nil, fmt.Errorf("cluster: %s: want a 4-byte shard index and the leaf args, got %d bytes", opKVQueryShardName, len(args))
	}
	return int(binary.BigEndian.Uint32(args[:4])), args[4:], nil
}

// handleKVQueryShard is the node-local leaf for opKVQueryShardName. It answers
// for the ONE group named in the payload — never another group, never onward to
// another node — so the fan-out stays flat. ErrNoShardOwner when this node does
// not host the group lets the sender's forward() loop move on to the next owner.
func (n *Node) handleKVQueryShard(args []byte) ([]byte, error) {
	group, leafArgs, err := decodeShardScopedKVQuery(args)
	if err != nil {
		return nil, err
	}
	if group < 0 || group >= n.cfg.NumShards {
		return nil, fmt.Errorf("cluster: %s: shard %d out of range [0,%d)", opKVQueryShardName, group, n.cfg.NumShards)
	}
	s := n.getShard(group)
	if s == nil {
		return nil, ErrNoShardOwner
	}
	return n.callHostedShard(s, kvQueryOpName, leafArgs)
}

// broadcastKVQuery fans a read-only kv_query leaf to every shard group that can
// still contribute rows, IN PARALLEL, and merges the pages. It is the read twin
// of broadcastFlush: kv_query is keyless, so shardIndexFor would send it to
// group 0 alone and silently answer from a slice of the keyspace.
//
// PARTIAL FAILURE IS A HARD ERROR, and that is the one place it differs from
// flush beyond direction. An under-complete page is a WRONG ANSWER, not a
// partial effect: the caller cannot tell it from a complete one, there is no
// "retry completes it" story to tell, and a filtered read that silently omits
// matching rows is the failure this whole feature may not have. So one group's
// error fails the query, with the group named.
//
// EXHAUSTED GROUPS ARE NOT SENT. The leaf honours a cursor's After and IGNORES
// its More, so "this group is finished" is the coordinator's fact to keep: a
// group that mergeKVQuery dropped from the cursor is simply not in the next
// request, and a request that names groups at all is answered by exactly those.
// A first page (no cursor) asks every group.
func (n *Node) broadcastKVQuery(args []byte) ([]byte, error) {
	a, err := wire.DecodeKVQueryArgs(args)
	if err != nil {
		return nil, err
	}
	ask, err := kvQueryTargets(a.Cursor, n.cfg.NumShards)
	if err != nil {
		return nil, err
	}

	// Only the groups that can still contribute get a leg: a finished group is
	// skipped BEFORE a goroutine, a context and a timer are spent on it, which on
	// a long paging run is most of them.
	results, errs := n.forEachGroupIn(ask, kvQueryGroupTimeout, func(ctx context.Context, group int) ([]byte, error) {
		if kvQueryLegHook != nil {
			kvQueryLegHook(group)
		}
		return n.callKVQueryGroup(ctx, group, a.Consistency, args)
	})

	parts := make([]wire.KVQueryResult, n.cfg.NumShards)
	for g := range results {
		if !ask[g] {
			continue
		}
		if errs[g] != nil {
			return nil, fmt.Errorf("cluster: kv_query: shard group %d: %w", g, n.classifyKVQueryErr(a.Index, g, errs[g]))
		}
		// DecodeKVQueryResult re-checks the part against KVQueryMaxPageBytes, so
		// a peer cannot hand this node a page larger than the cap it is about to
		// merge under.
		part, derr := wire.DecodeKVQueryResult(results[g])
		if derr != nil {
			return nil, fmt.Errorf("cluster: kv_query: shard group %d: %w", g, derr)
		}
		parts[g] = part
	}

	merged := mergeKVQuery(parts, a.Cursor, int(a.Limit), wire.KVQueryMaxPageBytes)
	if err := checkKVQueryCursorFits(merged.Cursor); err != nil {
		return nil, err
	}
	return wire.EncodeKVQueryResult(merged)
}

// kvQueryTargets reports, per group, whether this page must ask it.
//
// An EMPTY cursor is a first page and asks everyone. A non-empty one asks
// exactly the groups it names, because that is what the previous page's merge
// decided: a group it dropped answered in full and has nothing left, and asking
// it again would cost a round trip per group per page for the life of the query
// (and, on a leader-pinned read, a leader resolution with it).
//
// A cursor naming a group this cluster does not have is REFUSED, permanently.
// It used to be silently ignored, which is the worst of the three options: a
// cursor naming only out-of-range groups asks nobody, so the caller gets an
// exhausted empty page and reads it as "no more rows" for a query that never
// ran. Silence is indistinguishable from an answer here, and the cursor is
// opaque to the caller, so an edited or cross-cluster one has exactly one honest
// reply — this cursor does not belong to this cluster, stop resending it.
//
// wire.ErrKVQueryArgs is the class the args codec already uses for a
// structurally decodable cursor that is semantically wrong (continuations not
// increasing by group, over the continuation cap), and every transport carries
// it as PERMANENT in ops.KVQueryErrorFamily. A retryable class would be a lie:
// no amount of waiting gives this cluster the group.
func kvQueryTargets(conts []wire.KVQueryCont, numShards int) ([]bool, error) {
	ask := make([]bool, numShards)
	if len(conts) == 0 {
		for g := range ask {
			ask[g] = true
		}
		return ask, nil
	}
	for _, c := range conts {
		if c.Group >= uint32(numShards) { //nolint:gosec // numShards is a small positive config value
			return nil, fmt.Errorf("%w: cursor names shard group %d, but this cluster has %d groups",
				wire.ErrKVQueryArgs, c.Group, numShards)
		}
		ask[c.Group] = true
	}
	return ask, nil
}

// callKVQueryGroup answers ONE group, routed by the query's read consistency.
//
//   - AnyReplica ("stale"): any replica will do, so a hosted copy answers
//     locally and anything else is forwarded to an owner.
//   - LeaderOnly and Linearizable: the read must be served by the group's
//     current Raft leader — serve locally when this node hosts AND leads it,
//     otherwise route to the leader. This is CallPhysical's routing (node.go),
//     including its re-route when the local serve discovers it has just lost
//     leadership.
//
// Linearizable gets its freshness barrier for free and NOT from here:
// wire.ReadConsistencyOf peeks the consistency byte out of the leaf args, so
// shard.Store.Call runs VerifyLeader before serving. This function only has to
// deliver the read to a leader.
//
// The LOCAL serve calls the leaf directly (kvQueryOpName on the hosted store);
// only a REMOTE hop carries the wrapper. The wrapper is a node-level admin op
// and is not in the shard's op registry, so sending it to a local store would be
// an unknown op, not a query.
func (n *Node) callKVQueryGroup(ctx context.Context, group int, consistency uint8, leafArgs []byte) ([]byte, error) {
	if consistency == wire.ConsistencyAnyReplica {
		if s := n.getShard(group); s != nil {
			return n.callHostedShard(s, kvQueryOpName, leafArgs)
		}
		return n.forwardTimeout(group, opKVQueryShardName, encodeShardScopedKVQuery(group, leafArgs), kvQueryGroupTimeout)
	}
	if s := n.getShard(group); s != nil && s.IsLeader() {
		res, err := n.callHostedShard(s, kvQueryOpName, leafArgs)
		if err == nil {
			return res, nil
		}
		// Leadership lost between the check and the serve (for a linearizable
		// read, discovered BY the barrier). Route to the real leader, whose
		// shard re-runs the barrier against fresh state. Any other error is the
		// group's answer and propagates.
		var nle *shard.NotLeaderError
		if !errors.As(err, &nle) {
			return nil, err
		}
	}
	return n.forwardToLeaderAs(ctx, group, kvQueryOpName, leafArgs, opKVQueryShardName, encodeShardScopedKVQuery(group, leafArgs))
}

// classifyKVQueryErr turns a group's PERMANENT "no such index" into the
// RETRYABLE ErrIndexBuilding when the name IS in this node's meta catalog — and,
// when this node has REJECTED that definition, into the permanent
// wire.ErrKVIndexDef instead, because an index that cannot be built here will
// never stop "building".
//
// The two facts live in different places on purpose. A shard group knows only
// its own installed definitions, so a name it has never seen is, to it,
// permanent; the meta catalog is the cluster-wide truth about what exists. In
// the window between the meta commit and the observer installing it on N nodes,
// a create-then-query would otherwise be a hard failure for a definition that
// certainly exists — the one shape a client cannot be expected to guess at.
//
// The check is by SENTINEL AND BY TEXT because a remote group's error crossed a
// process boundary and arrived as a string; the substring fallback is the same
// one httpapi uses for sentinels that cross the Raft/RPC boundary.
//
// Nothing else is reclassified. Every other leaf refusal is already the right
// kind of error at the client: ErrIndexBuilding / ErrIndexChanged /
// ErrKVQueryUnavailable say "retry", and ErrKVQueryFilter / ErrKVQueryScanRequired
// / ErrKVQueryScanBudget / ErrCandidateBudget / ErrKVIndexUnavailable are
// permanent facts about the query or the deployment, which a retry cannot
// change. The full table is kvQueryLeafErrors, at the foot of this file.
func (n *Node) classifyKVQueryErr(index string, group int, err error) error {
	if index == "" || n.meta == nil || !isKVQueryNoSuchIndex(err) {
		return err
	}
	if _, inCatalog := n.meta.FSM.KVIndexLookup(index); !inCatalog {
		return err
	}
	// IN THE CATALOG IS NOT THE SAME AS BUILDABLE. A definition this node's own
	// observer refused (kvindex.DefFrom cannot parse its path here) will never
	// install, so "still building" would be a promise nothing can keep and the
	// client would retry forever at one full fan-out per attempt. Since
	// validateKVIndexDef refuses such a definition at admission, the only way one
	// reaches the catalog is version skew — a newer node admitting a grammar this
	// build lacks — and the honest answer is the PERMANENT wire.ErrKVIndexDef,
	// which all three transports render as a client error. See kvIndexDefRejected
	// for what this does and does not cover.
	if n.kvIndexDefRejected(index) {
		return fmt.Errorf("%w: %q is in the meta catalog but this node cannot build it, so it will never install here; the definition needs a build that understands its payload path",
			wire.ErrKVIndexDef, index)
	}
	return fmt.Errorf("%w: %q is in the meta catalog but not yet installed on shard group %d (retry)",
		kvindex.ErrIndexBuilding, index, group)
}

// kvQueryRemoteErrPrefix is what a peer's reply looks like once it has crossed
// the wire: client.ServerError renders `server error on op %q: %s` around the
// message the peer's edge chose to return. It is stripped by an ANCHORED CUT,
// never searched for.
var kvQueryRemoteErrPrefix = fmt.Sprintf("client: server error on op %q: ", opKVQueryShardName)

// isKVQueryNoSuchIndex reports whether err is the leaf's "no such index"
// refusal — by sentinel when the group was served locally, and otherwise by the
// EXACT SHAPE of the message, since a remote group's error reaches this node as
// text with its type gone.
//
// IT IS DELIBERATELY NOT A SUBSTRING TEST, and that is a hole rather than a
// style point. This predicate decides whether a PERMANENT refusal is rewritten
// as RETRYABLE, and a client told "retryable" retries — so any string an
// attacker can get into an error message is, under a substring test, a way to
// make a permanent error retry forever, at one full cluster fan-out per
// attempt. Filter fields are client-chosen and are quoted verbatim into
// ops.ErrKVQueryFilter's message, so a field named `$kvindex: no such index`
// smuggles the sentinel text into a permanent error exactly. Reproduced against
// the substring version; TestKVQueryFilterErrorIsNotSmuggledAsRetryable pins it.
//
// Three things close it, in order: a message carrying a PERMANENT sentinel's
// text is never rewritten, whatever else it says; the known wrapper prefix is
// removed by anchored cut; and what remains must be the leaf's message in full
// — sentinel, a legal index name, the shard number, nothing else
// (ops.IsKVQueryNoSuchIndexMessage).
func isKVQueryNoSuchIndex(err error) bool {
	if errors.Is(err, kvindex.ErrNoSuchIndex) {
		return true
	}
	msg := err.Error()
	// The veto. Contains is safe HERE and only here, because it can only REFUSE
	// the rewrite: a false positive costs a retryable error being reported as
	// permanent, which is the safe direction.
	if strings.Contains(msg, ops.ErrKVQueryFilter.Error()) {
		return false
	}
	if rest, ok := strings.CutPrefix(msg, kvQueryRemoteErrPrefix); ok {
		msg = rest
	}
	return ops.IsKVQueryNoSuchIndexMessage(msg)
}

// checkKVQueryCursorFits refuses a composite cursor that would not fit an ARGS
// frame, which is where the client has to put it to ask for the next page.
//
// EncodeKVQueryResult bounds the whole page at KVQueryMaxPageBytes, but the
// cursor's own cap (KVQueryMaxCursorBytes) applies on the way BACK IN. A cursor
// over it would encode here and be rejected on the next request, which reads to
// the client as a query that dies at page two for no stated reason.
//
// THE RESIDUAL CEILING, and it is now a corner rather than a wall. The cursor
// cap is 4 MiB and the codec stores the keys' shared prefix once, so what each
// group actually spends is 7 bytes plus its key BEYOND that prefix — about
// (4 MiB - 3)/NumShards - 7 bytes each. At 128 groups that is ~32 KiB of
// distinguishing suffix per group, against a 64 KiB maximum key: reachable only
// by keys that are both enormous and share almost nothing after the index's
// KeyPrefix. It fails loud, with the per-group arithmetic in the message.
func checkKVQueryCursorFits(conts []wire.KVQueryCont) error {
	if len(conts) == 0 {
		return nil
	}
	// The EXACT encoded size, not an estimate: the codec factors the keys'
	// shared prefix out of the block, so an estimate that charged every group
	// for a whole key would refuse cursors that encode comfortably.
	if n := wire.KVQueryCursorBytes(conts); n > wire.KVQueryMaxCursorBytes {
		// Marked with ops.ErrKVQueryCursorCap so every transport classifies it as
		// the CLIENT error it is (400 / InvalidArgument) instead of redacting it
		// to an opaque 500 — the per-group arithmetic below is the whole value of
		// the message, and a caller that never sees it cannot act on it.
		return fmt.Errorf("%w: the continuation for %d shard groups needs %d bytes, over the %d-byte cursor cap; the room left per group is about %d bytes of key beyond the shared prefix",
			ops.ErrKVQueryCursorCap, len(conts), n, wire.KVQueryMaxCursorBytes, (wire.KVQueryMaxCursorBytes-3)/len(conts)-7)
	}
	return nil
}

// --- the merge ------------------------------------------------------------

// mergeKVQuery folds one page from every shard group into ONE page plus the
// composite cursor that resumes it. It is PURE — no Node, no clock, no I/O — so
// the paging invariants below are testable without a cluster.
//
// `in` is the cursor the CALLER sent. It is not decoration: a group whose rows
// were all cut by the limit must resume from exactly where it started, and the
// only record of that is `in`. Advancing such a group to its leaf's own
// continuation would step over rows this page never delivered, and no later page
// would go back for them.
//
// THE RULES ARE ORDERED, AND RULE 4 READS THE More RULE 3 WROTE:
//
//  1. Concatenate every part's rows — minus any row at or below that group's
//     incoming After, which the caller has already seen — and sort ascending by
//     key. Keys are globally unique across groups (a key lives in exactly one
//     group), so there is no dedup — TestMergeKVQueryKeysAreGloballyUnique
//     asserts that rather than assuming it. "Its rows" below means the rows that
//     survive this filter.
//  2. Take rows in order until `limit` rows or `maxBytes` of encoded frame,
//     whichever comes first. The first row is always emitted, so a page always
//     advances and paging always terminates.
//  3. THEN, per group: if any of its rows were dropped by that truncation, its
//     After becomes the last key of ITS OWN that the page emitted (or its
//     incoming After when it emitted none) and More becomes true. Otherwise it
//     keeps its own part's After and More.
//  4. THEN, reading the More rule 3 wrote, drop a group whose More is false and
//     which returned fewer rows than `limit`. An empty list means the result
//     carries no cursor and paging is over.
//
// Rule 4's row-count condition is deliberately conservative: a group that
// returned EXACTLY the limit is indistinguishable from one with more to give, so
// it is kept and costs one extra empty round trip. A lost row costs correctness.
//
// VALUE NIL-NESS IS PRESERVED. A nil Value means "omitted because it exceeded
// the page cap; fetch it with get", a zero-length one means "present and empty".
// Rows are carried through by value and never rebuilt, so the two stay distinct.
func mergeKVQuery(parts []wire.KVQueryResult, in []wire.KVQueryCont, limit, maxBytes int) wire.KVQueryResult {
	// limit is 1..KVQueryMaxLimit by the time it gets here: DecodeKVQueryArgs
	// refuses anything outside that range, so there is no clamp to make. A zero
	// would emit no row, leave every continuation where it was, and page forever
	// — which is why the decoder, not this function, is where it is caught.
	inAfter := make(map[uint32][]byte, len(in))
	for _, c := range in {
		inAfter[c.Group] = c.After
	}

	// Rows at or below the caller's cursor are DROPPED, not emitted: the caller
	// has already seen them, and emitting one would duplicate a row and drag the
	// continuation backwards. A correct leaf never returns one (ops.verifyPage
	// filters on the same bound); this is the coordinator refusing to trust a
	// peer's word about it.
	type sourced struct {
		row   wire.KVQueryRow
		group int
	}
	usable := make([]int, len(parts))
	all := make([]sourced, 0, kvQueryRowCount(parts))
	for g := range parts {
		after := inAfter[uint32(g)] //nolint:gosec // g indexes parts, which is NumShards long
		for _, row := range parts[g].Rows {
			if len(after) > 0 && bytes.Compare(row.Key, after) <= 0 {
				continue
			}
			usable[g]++
			all = append(all, sourced{row: row, group: g})
		}
	}
	sort.Slice(all, func(i, j int) bool {
		if c := bytes.Compare(all[i].row.Key, all[j].row.Key); c != 0 {
			return c < 0
		}
		return all[i].group < all[j].group // unreachable with unique keys; deterministic if it ever is
	})

	budget := maxBytes - kvQueryMergeOverhead(parts, inAfter)
	res := wire.KVQueryResult{}
	emitted := make([]int, len(parts))
	last := make([][]byte, len(parts))
	used := 0
	for _, s := range all {
		if len(res.Rows) >= limit {
			break
		}
		cost := kvQueryRowBytes(s.row)
		if len(res.Rows) > 0 && used+cost > budget {
			break
		}
		res.Rows = append(res.Rows, s.row)
		used += cost
		emitted[s.group]++
		last[s.group] = s.row.Key
	}

	var cursor []wire.KVQueryCont
	for g := range parts {
		group := uint32(g) //nolint:gosec // g indexes parts, which is NumShards long
		leaf, hasLeaf := kvQueryContFor(parts[g].Cursor, group)

		// Rule 3.
		var after []byte
		var more bool
		switch {
		case emitted[g] < usable[g]:
			more = true
			if emitted[g] > 0 {
				after = last[g]
			} else {
				after = inAfter[group]
			}
		case hasLeaf:
			after, more = leaf.After, leaf.More
		case emitted[g] > 0:
			after = last[g]
		default:
			after = inAfter[group]
		}
		// THE CONTINUATION NEVER MOVES BACKWARDS. Two of the three sources above
		// are this node's own arithmetic, but `leaf.After` is a PEER's word: a
		// buggy or hostile group that answered with a continuation below the
		// cursor it was given would make the next page re-request what it just
		// got, drop every row as already-seen, and rewrite the same cursor —
		// paging that never terminates and never advances. Clamping to the
		// incoming bound costs one comparison and makes that unreachable.
		if a := inAfter[group]; len(a) > 0 && bytes.Compare(after, a) < 0 {
			after = a
		}
		// Rule 4, reading the More rule 3 just wrote.
		if !more && usable[g] < limit {
			continue
		}
		cursor = append(cursor, wire.KVQueryCont{Group: group, After: after, More: more})
	}
	res.Cursor = cursor
	return res
}

// kvQueryContFor finds group's continuation in a cursor. It is a linear scan
// rather than the binary search the strictly-increasing order would allow,
// because it also reads a PART's cursor — a peer's word, not something this node
// decoded under that invariant — and at most one entry per group is ever there.
func kvQueryContFor(conts []wire.KVQueryCont, group uint32) (wire.KVQueryCont, bool) {
	for _, c := range conts {
		if c.Group == group {
			return c, true
		}
	}
	return wire.KVQueryCont{}, false
}

func kvQueryRowCount(parts []wire.KVQueryResult) int {
	n := 0
	for _, p := range parts {
		n += len(p.Rows)
	}
	return n
}

// kvQueryRowBytes is EXACTLY what one row costs inside EncodeKVQueryResult's
// frame: keyLen(2) + key + hasVal(1), plus valLen(4) + value when the value is
// present. Present is `!= nil`, not `len > 0` — a zero-length value is stored
// and encoded, and only a nil one is absent.
func kvQueryRowBytes(r wire.KVQueryRow) int {
	n := 2 + len(r.Key) + 1
	if r.Value != nil {
		n += 4 + len(r.Value)
	}
	return n
}

// kvQueryMergeOverhead upper-bounds everything the merged frame carries that is
// not row payload: the row count, the cursor count, and one continuation per
// group. Reserving it is what makes "the rows fit the budget" imply "the frame
// encodes".
//
// A group's continuation key is always one of three things rule 3 can choose —
// its incoming After, its part's After, or one of the keys in its part — so the
// longest of those bounds it exactly.
func kvQueryMergeOverhead(parts []wire.KVQueryResult, inAfter map[uint32][]byte) int {
	n := 4 + 2
	for g := range parts {
		group := uint32(g) //nolint:gosec // g indexes parts, which is NumShards long
		longest := len(inAfter[group])
		if c, ok := kvQueryContFor(parts[g].Cursor, group); ok && len(c.After) > longest {
			longest = len(c.After)
		}
		for _, row := range parts[g].Rows {
			if len(row.Key) > longest {
				longest = len(row.Key)
			}
		}
		n += 7 + longest
	}
	return n
}

// kvQueryLeafErrors names, in one place, every error a kv_query leaf can answer
// with and how a client should read it. It exists to be READ (and to keep the
// classification honest when a new refusal is added), not to be called.
//
//	RETRYABLE (the query is fine; this replica or this moment is not)
//	  kvindex.ErrIndexBuilding    the definition is installed here but still backfilling
//	  kvindex.ErrIndexChanged     the definition moved under the query (leaf retries once, then reports building)
//	  kvindex.ErrNoSuchIndex      ONLY as rewritten by classifyKVQueryErr: in the meta catalog, not yet installed on that group
//	  ops.ErrKVQueryUnavailable   the scan's walk was cut short (the shard is being removed) — never a short page
//	  transport / timeout / ErrNoShardOwner / NotLeader: the group could not be reached in time
//
//	PERMANENT (the query, or the deployment, must change)
//	  kvindex.ErrNoSuchIndex      the name is in no catalog this node knows
//	  kvindex.ErrCandidateBudget  the selector unions more posting keys than the node's budget allows
//	  ops.ErrKVQueryFilter        the filter is one the leaf will not compile
//	  ops.ErrKVQueryScanRequired  no leaf of the filter can drive the index, and scan:true was not set
//	  ops.ErrKVQueryScanBudget    one page's walk visited more keys than the node's scan budget
//	  ops.ErrKVIndexUnavailable   this dispatcher has no KV index at all
//	  wire.ErrKVQueryArgs / ErrKVQueryResult / ErrKVQueryArgsTruncated: a malformed frame
var kvQueryLeafErrors = []error{
	kvindex.ErrIndexBuilding, kvindex.ErrIndexChanged, kvindex.ErrNoSuchIndex, kvindex.ErrCandidateBudget,
	ops.ErrKVQueryUnavailable, ops.ErrKVQueryFilter, ops.ErrKVQueryScanRequired, ops.ErrKVQueryScanBudget,
	ops.ErrKVIndexUnavailable,
}
