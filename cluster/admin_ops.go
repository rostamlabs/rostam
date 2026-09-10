// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// Online-rebalancing slice 4: the network admin surface. The coordinator runs on
// the meta-Raft leader and drives shard membership changes on its peers; in a
// real multi-process cluster it reaches them through these shardless admin ops
// rather than in-process *Node handles. Node.Call dispatches them directly on
// the receiving node (bypassing key→shard routing), since an admin op targets
// the specific node it was sent to — it must not be forwarded to shard 0's
// owner like an ordinary shardless op.

// Admin op names. All are shardless and node-local (Call dispatches them before
// routing). The "__rb_*" ops are internal coordinator plumbing; "__rebalance__"
// is the operator-facing trigger.
const (
	opRBAddOwner      = "__rb_add_owner__"
	opRBRemoveOwner   = "__rb_remove_owner__"
	opRBAddVoter      = "__rb_add_voter__"
	opRBRemoveVoter   = "__rb_remove_voter__"
	opRBTransfer      = "__rb_transfer__"
	opRBSetPlacement  = "__rb_set_placement__"
	opRBStatus        = "__rb_status__"
	opRBPlacement     = "__rb_placement__"
	opRBRebalanceName = "__rebalance__"
	opSetCatalogName  = "__set_catalog__"
	opSetReshardName  = "__set_reshard__"
	opSetAliasesName  = "__set_aliases__"
	opCatalogGenName  = "__catalog_gen__"
	// opMetaReadIndexName is the internal readIndex op: a follower forwards it to
	// the meta-Raft leader to learn the leader's verified FSM command frontier (see
	// metaLeaderFrontier / metaReadBarrier in meta_readindex.go). Internal-only
	// (carries no RBAC scoping; the peerClient supplies the internal token), it
	// dispatches off n.adminOps like __catalog_gen__.
	opMetaReadIndexName = "__meta_readindex__"
	// opShardReadIndexName is the internal per-SHARD readIndex op: a follower forwards
	// it to a specific shard's Raft leader to learn that leader's committed frontier
	// (CommitIndex) for a ConsistencyBoundedStaleness read (see shardLeaderFrontier /
	// fetchShardLeaderFrontier in shard_readindex.go). Unlike __meta_readindex__ it
	// carries a ShardIdx and does NOT Barrier-drain (a cheap freshness ping, not a
	// catch-up). Internal-only (the peerClient supplies the internal token); it
	// dispatches off n.adminOps like __meta_readindex__.
	opShardReadIndexName = "__shard_readindex__"
	// opPBLeaseRenewName is the internal PB primary-liveness beacon op: a node that
	// is not the meta leader forwards its batched OpShardLeaseRenew to the leader
	// (mirroring __set_catalog__), so the beacon reaches meta consensus regardless
	// of which node hosts the primary. Internal-only (the peerClient supplies the
	// internal token); it dispatches off n.adminOps like __set_catalog__.
	opPBLeaseRenewName = "__pb_lease_renew__"
	// opPBSetISRName is the internal PB ISR-shrink op: a follower-hosted
	// primary forwards its epoch-guarded OpSetShardISR to the meta leader
	// (mirroring __pb_lease_renew__), so a shrink reaches meta consensus regardless
	// of which node hosts the primary. Internal-only (the peerClient supplies the
	// internal token); it dispatches off n.adminOps like __pb_lease_renew__.
	opPBSetISRName = "__pb_set_isr__"
)

// shardReadIndexVersion is the current __shard_readindex__ request wire version. The
// readindex is per-shard (ShardIdx) and forward-compatible (a future variant could
// carry extra hints without breaking old peers).
const shardReadIndexVersion uint8 = 1

// shardReadIndexReq is the gob-encoded argument payload for the __shard_readindex__
// admin op: a version byte, the target shard index, and the caller's remaining
// deadline budget in nanoseconds. Decoded defensively (old/empty payload gob-defaults
// all fields to 0). BudgetNanos bounds the leader-side context; the leader does no
// Barrier wait so it is informational/forward-compatible, mirroring metaReadIndexReq.
type shardReadIndexReq struct {
	Version     uint8
	ShardIdx    uint32
	BudgetNanos int64
}

// shardReadIndexReply is the reply to opShardReadIndexName: the shard leader's
// committed frontier (CommitIndex after VerifyLeader). OK=false when the receiving
// node was not the leader of the requested shard, so the CALLER fails closed (routes
// to the leader) rather than treating a zero frontier as authoritative.
type shardReadIndexReply struct {
	Frontier uint64
	OK       bool
}

// metaReadIndexVersion is the current __meta_readindex__ request wire version. The
// readindex is GLOBAL to the meta-Raft (it confirms the whole local meta-FSM is
// caught up to the leader's command frontier), so the request needs no collection —
// only a version byte for forward-compatibility (a future variant could carry a
// per-collection hint without breaking old peers).
const metaReadIndexVersion uint8 = 1

// metaReadIndexLeaderBudget is the leader-side default upper bound on the
// __meta_readindex__ Barrier when the caller did not supply a remaining budget (an
// old peer with BudgetNanos==0). It mirrors the other forwarded meta handlers' 5s
// internal deadline. When the caller DOES supply a budget (the read-path follower
// always does), the handler uses min(this, caller budget) so the leader never blocks
// past the caller's own deadline.
const metaReadIndexLeaderBudget = 5 * time.Second

// metaReadIndexReq is the gob-encoded argument payload for the __meta_readindex__
// admin op. It carries a version byte (the readindex is global to the meta-Raft; no
// collection is needed) and the caller's REMAINING deadline budget so the leader's
// Barrier honors the caller's bound instead of an unrelated hardcoded one (review
// nit N1). Decoded defensively: an empty/old payload gob-defaults Version to 0 and
// BudgetNanos to 0, which the handler tolerates (BudgetNanos<=0 ⇒ fall back to the
// internal default). The field is additive — old peers that never set it send 0 and
// the leader uses its own default bound, so the wire stays backward-compatible.
type metaReadIndexReq struct {
	Version uint8
	// BudgetNanos is the caller's remaining deadline budget in nanoseconds at the
	// moment it forwarded the op. The leader bounds its Barrier by
	// min(internal default, this budget) so it never blocks past the caller's
	// deadline (the follower would have already given up). 0 ⇒ unknown/old caller ⇒
	// the leader uses its internal default.
	BudgetNanos int64
}

// metaReadIndexReply is the reply to opMetaReadIndexName: the meta leader's VERIFIED
// FSM COMMAND FRONTIER (metaLeaderFrontier — VerifyLeader + self catch-up). It is a
// real command index every follower deterministically reaches (NOT the raw
// CommitIndex; see the no-op-entry landmine in meta_readindex.go). OK=false when the
// receiving node was not the meta leader (or could not drain) so the CALLER
// re-resolves the leader and retries within its deadline — never serving a stale or
// zero frontier.
type metaReadIndexReply struct {
	Frontier uint64
	OK       bool
}

// catalogGenReq is the gob-encoded argument payload for the __catalog_gen__
// admin op: the collection whose LOCAL catalog generation the receiving node
// should report. The caller passes the canonical name (ops.CanonicalName), the
// same key the meta-FSM catalog is stored under, so the lookup matches.
type catalogGenReq struct {
	Collection string
}

// catalogGenReply is the reply to opCatalogGenName: the receiving node's LOCAL
// CollectionPartitionsGen(collection) — a pure local meta-FSM read (no barrier,
// no leader confirmation). OK=false for an unknown/single-partition collection
// (P=Gen=0), mirroring CollectionPartitionsGen's contract. The all-nodes-applied
// cutover gate (waitAllNodesCatalogGen) polls this to learn what generation each
// node's catalog currently routes to.
type catalogGenReply struct {
	P   uint32
	Gen uint32
	OK  bool
}

// setCatalogReq is the gob-encoded argument payload for the __set_catalog__
// admin op: a durable meta-catalog write forwarded to the meta-Raft leader.
type setCatalogReq struct {
	Collection string
	Partitions uint32
	Generation uint32 // partition generation (0 until resplit; gob defaults old reqs to 0)
}

// pbLeaseRenewReq is the gob-encoded argument payload for the __pb_lease_renew__
// admin op: a PB primary-liveness beacon forwarded to the meta-Raft leader. Node
// is the beaconing node; Renews are the (shard,epoch) pairs it currently primaries.
type pbLeaseRenewReq struct {
	Node   string
	Renews []ShardEpochPair
}

// pbSetISRReq is the gob-encoded argument payload for the __pb_set_isr__ admin op
// (ISR shrink): a follower-hosted primary's epoch-guarded ISR update
// forwarded to the meta-Raft leader.
type pbSetISRReq struct {
	ShardID int
	Epoch   uint64
	ISR     []string
}

// setReshardReq is the gob-encoded argument payload for the __set_reshard__
// admin op: a durable online-reshard-state write forwarded to the meta-Raft
// leader. Old/unset fields gob-default to zero (Stable).
type setReshardReq struct {
	Collection string
	Status     uint8
	TargetP    int
	TargetGen  uint32
	SourceP    int    // source (old) partition count, pinned at reshard-begin
	SourceGen  uint32 // source (old) generation, pinned at reshard-begin
}

// setAliasReq is the gob-encoded argument payload for the __set_aliases__ admin
// op: a durable atomic alias-batch write forwarded to the meta-Raft leader.
type setAliasReq struct {
	Actions []AliasAction
}

// adminReq is the unified argument payload for the admin ops (only the fields an
// op needs are populated). It is gob-encoded.
type adminReq struct {
	ShardID   int
	NumShards int
	PeerID    string
	RaftAddr  string
	Owners    []string
	// Rebalance trigger only:
	Members []Peer
	RF      int
}

// adminStatus is the reply to opRBStatus: a shard's local Raft progress on the
// node that served the request.
type adminStatus struct {
	Hosted       bool
	IsLeader     bool
	LastIndex    uint64
	AppliedIndex uint64
}

// adminPlacement is the reply to opRBPlacement: the node's full local routing
// view.
type adminPlacement struct {
	Placement [][]string
}

// RebalanceResult is the reply to the __rebalance__ trigger: how many shard
// moves the plan contained and how they resolved.
type RebalanceResult struct {
	Moves  int
	Done   int
	Failed int
}

func gobEncode(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobDecode(b []byte, v any) error {
	if len(b) == 0 {
		// An empty payload decodes to the zero value (ops with no args).
		return nil
	}
	return gob.NewDecoder(bytes.NewReader(b)).Decode(v)
}

// registerAdminOps populates n.adminOps with the node-local handlers. Called in
// the multi-node constructor; single-node deployments have nothing to rebalance.
func (n *Node) registerAdminOps() {
	n.adminOps = map[string]func([]byte) ([]byte, error){
		opRBAddOwner:         n.handleAddOwner,
		opRBRemoveOwner:      n.handleRemoveOwner,
		opRBAddVoter:         n.handleAddVoter,
		opRBRemoveVoter:      n.handleRemoveVoter,
		opRBTransfer:         n.handleTransfer,
		opRBSetPlacement:     n.handleSetPlacement,
		opRBStatus:           n.handleStatus,
		opRBPlacement:        n.handlePlacement,
		opRBRebalanceName:    n.handleRebalance,
		opSetCatalogName:     n.handleSetCatalog,
		opPBLeaseRenewName:   n.handlePBLeaseRenew,
		opPBSetISRName:       n.handlePBSetISR,
		opSetReshardName:     n.handleSetReshard,
		opSetAliasesName:     n.handleSetAliases,
		opCatalogGenName:     n.handleCatalogGen,
		opMetaReadIndexName:  n.handleMetaReadIndex,
		opShardReadIndexName: n.handleShardReadIndex,
		ops.ReadyOp:          n.handleReady,
		ops.ReplMetricsOp:    n.handleReplMetrics,
		ops.KVMetricsOp:      n.handleKVMetrics,
		// Shard-scoped leg of the WASM-registration broadcast: proposes to the
		// ONE group named in its payload and never re-broadcasts. See
		// wasm_broadcast.go.
		opRegisterWASMShardName: n.handleRegisterWASMShard,
		// Shard-scoped leg of the KV flush broadcast: proposes a flush to the ONE
		// group named in its payload and never re-broadcasts. See flush_broadcast.go.
		opFlushShardName: n.handleFlushShard,
		// KV record search: the index catalog CRUD (set/list) and the per-group
		// readiness leaf the list handler gathers. Dispatched by exact name before
		// any routing, exactly like __set_catalog__; the ready leaf answers for the
		// ONE group in its payload and never re-broadcasts. See kv_index_admin.go.
		opKVIndexSetName:   n.handleSetKVIndex,
		opKVIndexListName:  n.handleListKVIndexes,
		opKVIndexReadyName: n.handleKVIndexReady,
		// WASM blob transport: how a node that lacks a module's bytes obtains
		// them. Both are node-local leaves — the put verifies, compiles and
		// stores; the get reads the content-addressed store and NOTHING ELSE (no
		// apply-path lock, by contract). See wasm_blob_transport.go.
		opWASMBlobPutName: n.handleWASMBlobPut,
		opWASMBlobGetName: n.handleWASMBlobGet,
	}
}

// handleReady is cluster mode's READINESS probe (overriding the registry
// default): the node is ready only if EVERY shard it hosts is serviceable. Two
// conditions make a hosted shard un-ready:
//
//   - No usable leader: the node is neither the leader nor knows the current
//     leader's address to forward to (quorum lost, election in flight, or — PB
//     mode — no valid primary lease, which surfaces as an empty LeaderAddr).
//     Writes to the shard would fail outright.
//   - PB under-replication (#3 linkage): the shard's in-sync set has fallen
//     below the configured min-ISR floor. The primary refuses to ack writes
//     below the floor (H3), so the shard cannot durably commit even though a
//     primary exists — an operator/load-balancer must treat it as degraded and
//     stop routing writes to it. Read from the MetaRaft-authoritative pbControl,
//     so every node hosting the shard reports the same (shard-wide) condition.
//     Raft mode has no equivalent per-shard quorum-loss signal on this surface,
//     so its behavior is unchanged (leader presence only).
//
// Returns nil when ready, ErrNotReady (with the offending shard ids) otherwise.
func (n *Node) handleReady(_ []byte) ([]byte, error) {
	var noLeader, underReplicated []int
	pb := n.cfg.ReplicationMode == ReplicationModePB && n.pbControl != nil
	for i := 0; i < n.cfg.NumShards; i++ {
		s := n.getShard(i)
		if s == nil {
			continue // not hosted here — not this node's readiness concern
		}
		if !s.IsLeader() && s.LeaderAddr() == "" {
			noLeader = append(noLeader, i)
		}
		if pb && len(n.pbControl.ISR(i)) < n.pbControl.MinISR(i) {
			underReplicated = append(underReplicated, i)
		}
	}
	if len(noLeader) > 0 && len(underReplicated) > 0 {
		return nil, fmt.Errorf("%w: hosted shards without a leader: %v; under-replicated (ISR < min-ISR): %v",
			ErrNotReady, noLeader, underReplicated)
	}
	if len(noLeader) > 0 {
		return nil, fmt.Errorf("%w: hosted shards without a leader: %v", ErrNotReady, noLeader)
	}
	if len(underReplicated) > 0 {
		return nil, fmt.Errorf("%w: under-replicated hosted shards (ISR < min-ISR): %v", ErrNotReady, underReplicated)
	}
	return nil, nil
}

// handleCatalogGen reports the receiving node's LOCAL catalog generation for a
// collection. It is a pure local meta-FSM read (CollectionPartitionsGen → a
// lock-guarded map lookup): NO write barrier, NO leader confirmation. The op
// answers "what generation does THIS node's catalog currently show", which is
// exactly the question the all-nodes-applied cutover gate asks each peer. Like
// __rb_status__ it is shardless/admin and carries no collection auth scoping —
// it dispatches directly off n.adminOps (Node.Call) over the internal-token
// peerClient, before any key-routing.
func (n *Node) handleCatalogGen(args []byte) ([]byte, error) {
	var req catalogGenReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __catalog_gen__ decode: %w", err)
	}
	p, gen, ok := n.CollectionPartitionsGen(req.Collection)
	return gobEncode(catalogGenReply{P: p, Gen: gen, OK: ok})
}

// handleMetaReadIndex serves the __meta_readindex__ op on the meta-Raft leader: it
// returns the leader's VERIFIED FSM command frontier (metaLeaderFrontier —
// VerifyLeader + self catch-up), the index every follower deterministically reaches.
// Like __catalog_gen__ it is shardless/admin and carries no collection auth scoping
// — it dispatches directly off n.adminOps (Node.Call) over the internal-token
// peerClient, before any key-routing. On the not-leader signal it returns OK=false
// (Frontier=0) so the CALLER re-resolves the leader and retries within its deadline;
// it does NOT silently return a stale/zero frontier as if it were authoritative. A
// genuine error (no meta / Barrier failure / timeout) is returned as an error so the
// caller's forward fails and is retried. The Barrier deadline is
// min(metaReadIndexLeaderBudget, the caller's remaining budget in req.BudgetNanos)
// so the leader honors the caller's bound (N1) and never blocks past it; the
// follower's own deadline still bounds the overall wait.
func (n *Node) handleMetaReadIndex(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	var req metaReadIndexReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __meta_readindex__ decode: %w", err)
	}
	// Bound the leader-side Barrier by min(internal default, caller's remaining
	// budget) so we never block past the caller's deadline (the follower would have
	// already given up). An old/unknown caller (BudgetNanos<=0) gets the internal
	// default. (N1: previously this hardcoded 5s and ignored the caller's budget.)
	budget := metaReadIndexLeaderBudget
	if req.BudgetNanos > 0 && time.Duration(req.BudgetNanos) < budget {
		budget = time.Duration(req.BudgetNanos)
	}
	deadline := time.Now().Add(budget)
	frontier, err := n.metaLeaderFrontier(deadline)
	if err != nil {
		if errors.Is(err, errMetaNotLeader) {
			// Not the leader (or leadership lost mid-barrier): tell the caller to
			// re-resolve + retry rather than treating a zero frontier as authoritative.
			return gobEncode(metaReadIndexReply{OK: false})
		}
		return nil, fmt.Errorf("cluster: __meta_readindex__: %w", err)
	}
	return gobEncode(metaReadIndexReply{Frontier: frontier, OK: true})
}

// handleShardReadIndex serves the __shard_readindex__ op on a shard leader: it
// returns that shard's committed frontier (shardLeaderFrontier — VerifyLeader +
// CommitIndex, NO Barrier wait). Like __meta_readindex__ it is shardless/admin and
// carries no collection auth scoping — it dispatches directly off n.adminOps before
// any key-routing. On the not-leader signal it returns OK=false (Frontier=0) so the
// CALLER fails closed (routes to the leader) rather than treating a stale/zero
// frontier as authoritative. A genuine error is returned so the caller's forward
// fails and the bounded read fails closed too.
func (n *Node) handleShardReadIndex(args []byte) ([]byte, error) {
	var req shardReadIndexReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __shard_readindex__ decode: %w", err)
	}
	budget := metaReadIndexLeaderBudget
	if req.BudgetNanos > 0 && time.Duration(req.BudgetNanos) < budget {
		budget = time.Duration(req.BudgetNanos)
	}
	deadline := time.Now().Add(budget)
	frontier, err := n.shardLeaderFrontier(int(req.ShardIdx), deadline)
	if err != nil {
		if errors.Is(err, errNotShardLeader) {
			return gobEncode(shardReadIndexReply{OK: false})
		}
		return nil, fmt.Errorf("cluster: __shard_readindex__: %w", err)
	}
	return gobEncode(shardReadIndexReply{Frontier: frontier, OK: true})
}

// handleSetCatalog applies a forwarded meta-catalog write. It runs on the
// receiving node, which the sender selected as the current meta-Raft leader, so
// it applies the entry locally via ApplySetCatalogEntry rather than re-entering
// the forwarding path — guaranteeing no forwarding loop.
func (n *Node) handleSetCatalog(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	var req setCatalogReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __set_catalog__ decode: %w", err)
	}
	return nil, n.meta.ApplySetCatalogEntry(req.Collection, req.Partitions, req.Generation, 5*time.Second)
}

// handlePBLeaseRenew applies a forwarded PB primary-liveness beacon. Like
// handleSetCatalog it runs on the meta-Raft leader (the sender selected it) and
// applies the entry locally via ApplyShardLeaseRenew, so there is no forwarding
// loop. The FSM apply mutates no replicated state — it only fires the leader-local
// liveness observer (epoch/primary-guarded).
func (n *Node) handlePBLeaseRenew(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	var req pbLeaseRenewReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __pb_lease_renew__ decode: %w", err)
	}
	return nil, n.meta.ApplyShardLeaseRenew(req.Node, req.Renews, 5*time.Second)
}

// handlePBSetISR applies a forwarded PB ISR shrink. Like
// handlePBLeaseRenew it runs on the meta-Raft leader (the sender selected it) and
// applies the entry locally via ApplySetShardISR, so there is no forwarding loop.
// The FSM apply is epoch-guarded, so a stale-epoch shrink (a fenced ex-primary) is
// a no-op.
func (n *Node) handlePBSetISR(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	var req pbSetISRReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __pb_set_isr__ decode: %w", err)
	}
	return nil, n.meta.ApplySetShardISR(req.ShardID, req.Epoch, req.ISR, 5*time.Second)
}

// handleSetReshard applies a forwarded online-reshard-state write. Like
// handleSetCatalog it runs on the meta-Raft leader (the sender selected it) and
// applies the entry locally, so there is no forwarding loop.
func (n *Node) handleSetReshard(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	var req setReshardReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __set_reshard__ decode: %w", err)
	}
	return nil, n.meta.ApplySetCatalogReshard(req.Collection, ReshardEntry{
		Status:    req.Status,
		TargetP:   req.TargetP,
		TargetGen: req.TargetGen,
		SourceP:   req.SourceP,
		SourceGen: req.SourceGen,
	}, 5*time.Second)
}

// handleSetAliases applies a forwarded atomic alias-batch write. Like
// handleSetReshard it runs on the meta-Raft leader (the sender selected it) and
// applies the entry locally via ApplySetAliasBatch, so there is no forwarding loop.
func (n *Node) handleSetAliases(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	var req setAliasReq
	if err := gobDecode(args, &req); err != nil {
		return nil, fmt.Errorf("cluster: __set_aliases__ decode: %w", err)
	}
	return nil, n.meta.ApplySetAliasBatch(req.Actions, 5*time.Second)
}

func decodeAdminReq(args []byte) (adminReq, error) {
	var req adminReq
	if err := gobDecode(args, &req); err != nil {
		return adminReq{}, fmt.Errorf("cluster: admin op decode: %w", err)
	}
	return req, nil
}

func (n *Node) handleAddOwner(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	return nil, n.AddShardOwner(req.ShardID)
}

func (n *Node) handleRemoveOwner(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	return nil, n.RemoveShardOwner(req.ShardID)
}

func (n *Node) handleAddVoter(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	return nil, n.ShardAddVoter(req.ShardID, req.PeerID, req.RaftAddr)
}

func (n *Node) handleRemoveVoter(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	return nil, n.ShardRemoveVoter(req.ShardID, req.PeerID)
}

func (n *Node) handleTransfer(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	return nil, n.ShardTransferLeadership(req.ShardID, req.PeerID, req.RaftAddr)
}

func (n *Node) handleSetPlacement(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	n.SetShardPlacement(req.ShardID, req.Owners)
	return nil, nil
}

func (n *Node) handleStatus(args []byte) ([]byte, error) {
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}
	return gobEncode(n.localStatus(req.ShardID))
}

func (n *Node) handlePlacement(_ []byte) ([]byte, error) {
	return gobEncode(adminPlacement{Placement: n.placementCopy()})
}

// localStatus reports a shard's Raft progress on this node (the basis for both
// the local shardAdmin and the opRBStatus reply).
func (n *Node) localStatus(shardID int) adminStatus {
	s := n.getShard(shardID)
	if s == nil {
		return adminStatus{}
	}
	return adminStatus{
		Hosted:       true,
		IsLeader:     s.IsLeader(),
		LastIndex:    s.LastIndex(),
		AppliedIndex: s.RaftAppliedIndex(),
	}
}
