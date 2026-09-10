// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// Online-rebalancing slice 1: the primitives to host a shard on a live node and
// to change a shard group's Raft membership. Higher slices (single-shard
// migration, the coordinator) orchestrate these.

// shardRaftID is the per-shard Raft server id for a node (matches the
// construction-time naming in toRaftServers' suffix scheme).
func shardRaftID(nodeID string, shardID int) string {
	return fmt.Sprintf("%s-shard-%04d", nodeID, shardID)
}

// buildShardConfig assembles the shard.Config for shardID. owners is the replica
// set used to derive the Raft peer list (nil/empty → join mode: the store starts
// idle and waits for the leader to AddVoter it). Used by both construction and
// AddShardOwner so the two paths stay identical.
func (n *Node) buildShardConfig(shardID int, owners []string, bootstrap bool) shard.Config {
	suffix := fmt.Sprintf("-shard-%04d", shardID)
	subCfg := n.cfg.ShardCfg
	subCfg.NodeID = n.cfg.NodeID + suffix
	subCfg.DataDir = filepath.Join(n.cfg.DataDir, fmt.Sprintf("shard-%04d", shardID))
	subCfg.Bootstrap = bootstrap
	subCfg.Cache.NumShards = 1
	if subCfg.DataDir != "" {
		subCfg.Cache.DataDir = filepath.Join(subCfg.DataDir, "cache")
	}
	subCfg.Ops = n.cfg.Ops
	// Tell the apply dispatcher which group it serves, so the node-wide
	// __register_wasm__ hook can attribute each apply to the log it came from
	// (see checkWASMRouteGate). This is set in BOTH replication modes; the PB
	// branch below no longer has to set it separately.
	subCfg.ShardIndex = shardID
	// Carry dynamic WASM registrations through snapshot install/restore. This is
	// what an AddShardOwner-joined replica depends on: it is built with an empty
	// peer list and catches up via InstallSnapshot, so it applies NONE of the
	// __register_wasm__ entries the snapshot replaced. The hooks are bound to
	// THIS group, because the snapshot has to record which registrations this
	// group's log carries and the restore has to attribute what it installs.
	subCfg.WASMSnapshot = func() []byte { return n.snapshotWASMState(shardID) }
	subCfg.WASMRestore = func(b []byte) error { return n.restoreWASMState(shardID, b) }
	// The classRetry block hooks: an apply that names a module version this
	// node does not hold parks, and these are what fetch it and make the park
	// visible. Neither blocks and neither returns holding wasmApplyMu — see
	// Node.onShardApplyRetry.
	subCfg.OnApplyRetry = n.onShardApplyRetry
	subCfg.OnApplyRetryCleared = n.onShardApplyRetryCleared
	// Route this shard's group over the fabric multiplexed transport when
	// selected, else the default mux StreamLayer. Exactly one of n.fabric/n.mux
	// is non-nil (see newMultiNode).
	if n.fabric != nil {
		subCfg.RaftTransport = n.fabric.For(uint32(shardID)) //nolint:gosec // shardID < NumShards <= 65536
	} else {
		subCfg.RaftStreamLayer = n.mux.For(uint32(shardID)) //nolint:gosec // shardID < NumShards <= 65536
	}
	subCfg.RaftPeers = toRaftServers(peersForOwners(n.cfg.Peers, owners), suffix)

	// Primary-backup mode: inject the per-shard PB wiring built in newMultiNode's
	// PB block. In raft mode this whole branch is skipped, so subCfg's PB fields
	// stay zero and construction is byte-identical to before. The PBRegister hook
	// does double duty: it registers this shard's engine as the inbound network
	// Receiver AND records it into n.pbEngines for the lease-keeper (and tests).
	if n.cfg.ReplicationMode == ReplicationModePB {
		subCfg.ReplicationMode = shard.ReplicationModePB
		// The PB engine identifies itself by the CLUSTER node ID, not the
		// raft-suffixed per-shard ID: the control plane (placement, ShardPrimary,
		// ISR) names members by cluster node ID, and the engine matches its own
		// nodeID against Primary()/ISR(). A suffixed ID would never match, so a
		// primary would self-fence (ErrNotPrimary) and would try to replicate to
		// its own cluster-ID as if it were a backup. PB shards use no per-shard Raft
		// group, so the suffixed ID is not needed here (DataDir stays per-shard).
		subCfg.NodeID = n.cfg.NodeID
		subCfg.PBControl = n.pbControl
		pbTr := newPBResolvingTransport(n.pbTransport.For(shardID), n.pbAddrOf)
		if n.cfg.pbTransportWrap != nil {
			// Test-only seam (nil in prod ⇒ identity): the shrink harness
			// splices a per-node drop injector here so replication to a chosen backup
			// fails deterministically, exercising the engine's failure counter.
			pbTr = n.cfg.pbTransportWrap(pbTr)
		}
		subCfg.PBTransport = pbTr
		subCfg.PBClock = n.pbNow
		if n.cfg.PBCommitPrimary {
			subCfg.PBCommitLevel = pbisr.CommitPrimary
		}
		subCfg.PBRegister = func(shard int, r pbisr.Receiver) {
			n.pbTransport.Register(shard, r)
			if eng, ok := r.(*pbisr.Engine); ok {
				n.pbEngines[shard] = eng
			}
		}
	}
	return subCfg
}

// getShard loads a shard slot pointer under the read lock. Returns nil when the
// node does not host the shard.
func (n *Node) getShard(idx int) *shard.Store {
	n.shardMu.RLock()
	defer n.shardMu.RUnlock()
	if idx < 0 || idx >= len(n.shards) {
		return nil
	}
	return n.shards[idx]
}

// snapshotShards returns a copy of the shard slice for safe iteration.
func (n *Node) snapshotShards() []*shard.Store {
	n.shardMu.RLock()
	defer n.shardMu.RUnlock()
	out := make([]*shard.Store, len(n.shards))
	copy(out, n.shards)
	return out
}

// AddShardOwner makes this node start hosting shardID by creating its store in
// join mode — idle until the shard's current leader adds it as a voter (which
// streams it the snapshot + log). Idempotent: a no-op if already hosting.
func (n *Node) AddShardOwner(shardID int) error {
	if shardID < 0 || shardID >= n.cfg.NumShards {
		return fmt.Errorf("cluster: shard %d out of range [0,%d)", shardID, n.cfg.NumShards)
	}
	n.shardMu.Lock()
	defer n.shardMu.Unlock()
	if n.shards[shardID] != nil {
		return nil
	}
	store, err := shard.New(n.buildShardConfig(shardID, nil, false))
	if err != nil {
		return fmt.Errorf("cluster: add shard %d owner: %w", shardID, err)
	}
	n.shards[shardID] = store
	// Re-arm the walk gate: a group that was removed earlier left it closed, and
	// the observer must be able to backfill the new store's empty index.
	n.reopenKVIndexWalks(shardID)
	// Then put THIS store's scan-mode kv_query walks behind that same gate, so a
	// scan in flight when the group is removed again is drained exactly like a
	// backfill rather than reading pages Close has unmapped. Order matters only in
	// that the gate must be open before a walk can register; installing the walker
	// starts none. See installKVQueryScanGate.
	n.installKVQueryScanGate(shardID, store)
	return nil
}

// RemoveShardOwner stops hosting shardID: it closes the store and deletes its
// on-disk directory. Idempotent. The caller must have already removed this node
// from the shard's Raft configuration (see ShardRemoveVoter) on the leader.
func (n *Node) RemoveShardOwner(shardID int) error {
	if shardID < 0 || shardID >= n.cfg.NumShards {
		return fmt.Errorf("cluster: shard %d out of range [0,%d)", shardID, n.cfg.NumShards)
	}
	n.shardMu.Lock()
	s := n.shards[shardID]
	n.shards[shardID] = nil
	// FOLD THE GROUP'S KV INDEX COUNTERS INTO THE NODE TOTALS BEFORE ITS INDEX
	// GOES AWAY. Stats().KVIndex.VerifyMisses and .ReconcileDrops are each the
	// node counter plus the sum over HOSTED groups (kvIndexStats), because the
	// query leaf runs in ops and the reconcile pass runs on the store's own
	// goroutine — both can only reach the Set they hold. Dropping a group
	// without folding would make those contracted uint64s DECREASE, which every
	// scraper reads as a process restart and a reset of every other counter
	// alongside them.
	//
	// Inside the lock, in the same critical section that takes the store out of
	// n.shards, so no observer can see the group counted by neither. A query
	// still in flight on this store can add a miss after the fold and lose it;
	// that is bounded by the in-flight queries and never makes the total go
	// backwards, which is the property being protected.
	if s != nil {
		if idx := s.KVIndex(); idx != nil {
			n.kvIndexVerifyMisses.Add(idx.VerifyMisses())
			n.kvIndexReconcileDrops.Add(idx.ReconcileDrops())
		}
	}
	n.shardMu.Unlock()
	// Drain any in-flight KV index backfill on this group BEFORE the store closes.
	// The walk aliases the store's live mmap, which Close unmaps; draining first
	// means the walk has returned before anything is unmapped, and the gate stays
	// closed so a pass holding a stale store pointer cannot start a new one. Done
	// unconditionally, s==nil included, so a concurrent removal of a group this
	// node had already dropped still shuts its gate.
	n.drainKVIndexWalks(shardID)
	if s == nil {
		return nil
	}
	err := s.Close()
	if n.cfg.DataDir != "" {
		_ = os.RemoveAll(filepath.Join(n.cfg.DataDir, fmt.Sprintf("shard-%04d", shardID)))
	}
	return err
}

// ownersFor returns a copy of shardIdx's owner set from this node's routing
// view, under the read lock (placement is mutated live by rebalancing).
func (n *Node) ownersFor(shardIdx int) []string {
	n.shardMu.RLock()
	defer n.shardMu.RUnlock()
	if shardIdx < 0 || shardIdx >= len(n.placement) {
		return nil
	}
	return append([]string(nil), n.placement[shardIdx]...)
}

// placementCopy returns a deep copy of the full placement table under the read
// lock, for inclusion in Topology.
func (n *Node) placementCopy() [][]string {
	n.shardMu.RLock()
	defer n.shardMu.RUnlock()
	out := make([][]string, len(n.placement))
	for i, owners := range n.placement {
		out[i] = append([]string(nil), owners...)
	}
	return out
}

// SetShardPlacement replaces this node's local routing view for shardID so Call
// forwards to the new owner set. The authoritative copy lives in meta-Raft
// State.Placement (committed via OpSetPlacement); this keeps the node's hot-path
// routing in sync as a migration advances.
func (n *Node) SetShardPlacement(shardID int, owners []string) {
	n.shardMu.Lock()
	defer n.shardMu.Unlock()
	if shardID >= 0 && shardID < len(n.placement) {
		n.placement[shardID] = append([]string(nil), owners...)
	}
}

// shardIsLeader reports whether this node currently leads shardID's Raft group.
func (n *Node) shardIsLeader(shardID int) bool {
	s := n.getShard(shardID)
	return s != nil && s.IsLeader()
}

// ShardTransferLeadership hands shardID's Raft leadership to toNodeID (at
// toRaftAddr). Must be called on the node that currently leads shardID.
func (n *Node) ShardTransferLeadership(shardID int, toNodeID, toRaftAddr string) error {
	s := n.getShard(shardID)
	if s == nil {
		return fmt.Errorf("cluster: not hosting shard %d", shardID)
	}
	return s.TransferLeadershipTo(shardRaftID(toNodeID, shardID), toRaftAddr)
}

// ShardAddVoter adds peer (a node id + its Raft transport addr) as a voter to
// shardID's Raft group. Must be called on the node that leads shardID; the peer
// must already have created its store via AddShardOwner.
func (n *Node) ShardAddVoter(shardID int, peerNodeID, peerRaftAddr string) error {
	s := n.getShard(shardID)
	if s == nil {
		return fmt.Errorf("cluster: not hosting shard %d", shardID)
	}
	return s.AddVoter(shardRaftID(peerNodeID, shardID), peerRaftAddr)
}

// ShardRemoveVoter removes peer from shardID's Raft group. Must be called on the
// node that leads shardID.
func (n *Node) ShardRemoveVoter(shardID int, peerNodeID string) error {
	s := n.getShard(shardID)
	if s == nil {
		return fmt.Errorf("cluster: not hosting shard %d", shardID)
	}
	return s.RemoveServer(shardRaftID(peerNodeID, shardID))
}
