// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/raft/logstore"
)

// metaGroupID is the mux group ID reserved for the meta-Raft transport.
// Chosen as the maximum uint32 so it cannot collide with shard IDs
// (which are 0..NumShards-1, bounded by NumShards <= 65536).
const metaGroupID uint32 = 0xFFFFFFFF

// MetaRaft owns the meta-Raft lifecycle for a single Rostam node.
// It is constructed directly against hashicorp/raft (bypassing the
// rostam/raft wrapper) because the meta-Raft is a small, isolated
// state machine that does not need the wrapper's machinery.
type MetaRaft struct {
	FSM       *MetaFSM
	Raft      *hraft.Raft
	transport hraft.Transport
	logStore  *logstore.WAL
	snapStore hraft.SnapshotStore
}

// startMetaRaft constructs and starts the meta-Raft group for this node.
// The caller supplies the meta group's transport (a per-group NetworkTransport
// over the mux StreamLayer, or the fabric per-group facade); the WAL log/stable
// store and a FileSnapshotStore are rooted at <cfg.DataDir>/meta/. On
// cfg.Bootstrap, it calls BootstrapCluster with cfg.Peers as the initial voter
// set; ErrCantBootstrap is tolerated (state already exists).
func startMetaRaft(cfg Config, transport hraft.Transport) (*MetaRaft, error) {
	dataDir := filepath.Join(cfg.DataDir, "meta")
	if err := os.MkdirAll(dataDir, 0o750); err != nil { //nolint:gosec // DataDir is caller-controlled
		return nil, fmt.Errorf("cluster: meta mkdir %s: %w", dataDir, err)
	}

	logStore, err := logstore.OpenWAL(filepath.Join(dataDir, "raftlog"), !cfg.ShardCfg.NoSync)
	if err != nil {
		return nil, fmt.Errorf("cluster: meta wal: %w", err)
	}

	snapStore, err := hraft.NewFileSnapshotStore(dataDir, 2, os.Stderr)
	if err != nil {
		_ = logStore.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, fmt.Errorf("cluster: meta snapshot store: %w", err)
	}

	raftCfg := hraft.DefaultConfig()
	raftCfg.LocalID = hraft.ServerID(cfg.NodeID + "-meta")
	if cfg.ShardCfg.RaftHeartbeatMs > 0 {
		raftCfg.HeartbeatTimeout = time.Duration(cfg.ShardCfg.RaftHeartbeatMs) * time.Millisecond
		raftCfg.LeaderLeaseTimeout = raftCfg.HeartbeatTimeout
	}
	if cfg.ShardCfg.RaftElectionMs > 0 {
		raftCfg.ElectionTimeout = time.Duration(cfg.ShardCfg.RaftElectionMs) * time.Millisecond
	}

	fsm := newMetaFSMForNode(cfg.NodeID)
	r, err := hraft.NewRaft(raftCfg, fsm, logStore, logStore, snapStore, transport)
	if err != nil {
		_ = logStore.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, fmt.Errorf("cluster: meta NewRaft: %w", err)
	}

	if cfg.Bootstrap {
		if len(cfg.Peers) == 0 {
			_ = r.Shutdown().Error() //nolint:errcheck,gosec // best-effort cleanup on error path
			_ = logStore.Close()     //nolint:errcheck,gosec // best-effort cleanup on error path
			return nil, errors.New("cluster: meta bootstrap requires non-empty Peers")
		}
		has, herr := hraft.HasExistingState(logStore, logStore, snapStore)
		if herr != nil {
			_ = r.Shutdown().Error() //nolint:errcheck,gosec // best-effort cleanup on error path
			_ = logStore.Close()     //nolint:errcheck,gosec // best-effort cleanup on error path
			return nil, fmt.Errorf("cluster: meta HasExistingState: %w", herr)
		}
		if !has {
			conf := hraft.Configuration{Servers: toRaftServers(cfg.Peers, "-meta")}
			if f := r.BootstrapCluster(conf); f.Error() != nil && !errors.Is(f.Error(), hraft.ErrCantBootstrap) {
				_ = r.Shutdown().Error() //nolint:errcheck,gosec // best-effort cleanup on error path
				_ = logStore.Close()     //nolint:errcheck,gosec // best-effort cleanup on error path
				return nil, fmt.Errorf("cluster: meta bootstrap: %w", f.Error())
			}
		}
	}

	return &MetaRaft{
		FSM:       fsm,
		Raft:      r,
		transport: transport,
		logStore:  logStore,
		snapStore: snapStore,
	}, nil
}

// ApplySetMembersIfLeader applies an OpSetMembers entry, but only if
// this node is the meta-Raft leader. It is idempotent: if the current
// FSM state already matches (peers + numShards), it returns nil without
// applying. Non-leader nodes return nil (no-op).
func (m *MetaRaft) ApplySetMembersIfLeader(peers []Peer, numShards, replicationFactor, minISR int, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return nil
	}
	// Sort before comparing so the order-sensitive peerSlicesEqual check
	// is deterministic regardless of call-site ordering.
	sortedPeers := append([]Peer(nil), peers...)
	sort.Slice(sortedPeers, func(i, j int) bool {
		return sortedPeers[i].NodeID < sortedPeers[j].NodeID
	})
	// Idempotency: skip if state already matches (INCLUDING the seeded ISR floor,
	// so a first bootstrap that must raise MinISR from 0 is not skipped).
	// ReplicationFactor is stored in State since it is not derivable from
	// Placement after an online rebalance (Placement diverges from the computed
	// layout); comparing st.ReplicationFactor avoids reverting rebalanced shards.
	// !st.ReplicationFactorSet means a snapshot predating this field: fall
	// through and let Apply record it without resetting Placement.
	st := m.FSM.State()
	if st.NumShards == numShards &&
		st.MinISR == minISR &&
		st.ReplicationFactorSet && st.ReplicationFactor == replicationFactor &&
		peerSlicesEqual(st.Members, sortedPeers) {
		return nil
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:                OpSetMembers,
		Members:           sortedPeers,
		NumShards:         numShards,
		ReplicationFactor: replicationFactor,
		MinISR:            minISR,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetMembers: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetMembers: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetMembers: %w", respErr)
		}
	}
	return nil
}

// ApplySetPlacement commits a single shard's owner set to meta-Raft. Returns
// hraft.ErrNotLeader on a follower so the caller can try another node; any
// other error is the apply/FSM failure. Online rebalancing calls this as each
// shard migration progresses so State.Placement advances shard-by-shard.
func (m *MetaRaft) ApplySetPlacement(shardID, numShards int, owners []string, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:        OpSetPlacement,
		ShardID:   shardID,
		NumShards: numShards,
		Owners:    owners,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetPlacement: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetPlacement: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetPlacement: %w", respErr)
		}
	}
	return nil
}

// ApplySetShardEpoch commits a shard leadership-epoch bump (epoch + primary) to
// meta-Raft — the primary-backup / ISR control plane (shard/pbisr/DESIGN.md,
// control plane). The FSM applies it MONOTONICALLY (a stale/equal epoch is a no-op) and
// RESETS the ISR to just {primary}. Returns hraft.ErrNotLeader on a follower so
// the caller can retry another node. Mirrors ApplySetPlacement's path.
func (m *MetaRaft) ApplySetShardEpoch(shardID int, epoch uint64, primary string, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:      OpSetShardEpoch,
		ShardID: shardID,
		Epoch:   epoch,
		Primary: primary,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetShardEpoch: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetShardEpoch: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetShardEpoch: %w", respErr)
		}
	}
	return nil
}

// ApplySetShardFormer designates node as the single owner that bootstraps
// shardID's Raft group (see State.ShardFormer). Returns whether THIS call
// installed the designation: false means one already existed, which is the
// normal outcome on a re-seed and the guard that keeps a rejoining node from
// forming a shard the cluster already formed.
//
// Leader-only, like every other Apply helper here — a follower gets
// hraft.ErrNotLeader and (false, err). The formation seeder therefore retries
// until it is the leader, and every node reads the RESULT from its replicated
// FSM rather than proposing for itself.
func (m *MetaRaft) ApplySetShardFormer(shardID int, node string, timeout time.Duration) (bool, error) {
	if m.Raft.State() != hraft.Leader {
		return false, hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:      OpSetShardFormer,
		ShardID: shardID,
		Node:    node,
	})
	if err != nil {
		return false, fmt.Errorf("cluster: meta encode SetShardFormer: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return false, fmt.Errorf("cluster: meta apply SetShardFormer: %w", err)
	}
	switch resp := f.Response().(type) {
	case error:
		return false, fmt.Errorf("cluster: meta FSM SetShardFormer: %w", resp)
	case bool:
		return resp, nil
	}
	return false, nil
}

// ApplySetShardSeed commits a shard's INITIAL primary-backup control state —
// (epoch, primary, full ISR) — as a SINGLE meta-Raft entry. It is the atomic form
// of "ApplySetShardEpoch then ApplySetShardISR" and exists because that two-entry
// sequence leaves an intermediate COMMITTED state whose ISR is the singleton
// {primary} (OpSetShardEpoch resets the ISR). A primary that reads that
// intermediate state from its LOCAL FSM acks writes against an ISR narrower than
// the committed one — on itself alone — and every such write is lost when it dies
// and a backup is promoted. One entry means that state never exists.
//
// Use for SEEDS only (bootstrap, DR restore). A FAILOVER promotion must keep using
// ApplySetShardEpoch: resetting the ISR to the new primary alone is the intended
// semantics there. Same monotonic epoch guard as ApplySetShardEpoch; a stale/equal
// epoch is a no-op. Returns hraft.ErrNotLeader on a follower.
func (m *MetaRaft) ApplySetShardSeed(shardID int, epoch uint64, primary string, isr []string, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:      OpSetShardEpoch,
		ShardID: shardID,
		Epoch:   epoch,
		Primary: primary,
		ISR:     isr,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetShardSeed: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetShardSeed: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetShardSeed: %w", respErr)
		}
	}
	return nil
}

// ApplySetShardISR commits a shard's in-sync replica set to meta-Raft — the
// primary-backup / ISR control plane (shard/pbisr/DESIGN.md). The FSM
// applies it only if epoch EXACTLY matches the shard's current epoch (a
// stale-epoch update is a no-op, defending against a fenced primary; H3/H6).
// Returns hraft.ErrNotLeader on a follower. Mirrors ApplySetPlacement's path.
func (m *MetaRaft) ApplySetShardISR(shardID int, epoch uint64, isr []string, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:      OpSetShardISR,
		ShardID: shardID,
		Epoch:   epoch,
		ISR:     isr,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetShardISR: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetShardISR: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetShardISR: %w", respErr)
		}
	}
	return nil
}

// ApplyShardLeaseRenew commits a primary-liveness beacon: one log entry carrying
// the beaconing node and the batch of (shard, epoch) pairs it currently primaries
// (automatic failover). The FSM applies it as a pure no-op that only fires
// the leader-local liveness observer (epoch/primary-guarded) — it mutates NO
// replicated state. COMMITTING the beacon IS the quorum-connection proof (a node
// partitioned from the meta quorum cannot land it), so there is no separate
// confirmMetaView gate on this path. Returns hraft.ErrNotLeader on a follower so
// the caller forwards to the leader (see Node.submitShardLeaseRenew). An empty
// batch is a no-op (nothing to renew). Mirrors ApplySetShardISR's path.
func (m *MetaRaft) ApplyShardLeaseRenew(node string, renews []ShardEpochPair, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	if len(renews) == 0 {
		return nil
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:         OpShardLeaseRenew,
		Node:       node,
		LeaseRenew: renews,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode ShardLeaseRenew: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply ShardLeaseRenew: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM ShardLeaseRenew: %w", respErr)
		}
	}
	return nil
}

// ApplySetCatalogEntry commits a collection's partition count to the meta-Raft
// catalog. partitions must be > 0. Returns hraft.ErrNotLeader if this node is
// not the meta leader (the caller forwards to the leader; see
// Node.SetCollectionPartitions).
func (m *MetaRaft) ApplySetCatalogEntry(collection string, partitions, generation uint32, timeout time.Duration) error {
	if partitions == 0 {
		return fmt.Errorf("cluster: SetCatalogEntry: partitions must be > 0")
	}
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:         OpSetCatalogEntry,
		Collection: collection,
		Partitions: partitions,
		Generation: generation,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetCatalogEntry: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetCatalogEntry: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetCatalogEntry: %w", respErr)
		}
	}
	return nil
}

// ApplySetCatalogReshard commits a collection's online-reshard state to the
// meta-Raft catalog. Returns hraft.ErrNotLeader if this node is not the meta
// leader (the caller forwards to the leader; see Node.SetCollectionReshard).
// Mirrors ApplySetCatalogEntry's locking/forwarding path.
func (m *MetaRaft) ApplySetCatalogReshard(collection string, e ReshardEntry, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:               OpSetCatalogReshard,
		Collection:       collection,
		ReshardStatus:    e.Status,
		ReshardTargetP:   e.TargetP,
		ReshardTargetGen: e.TargetGen,
		ReshardSourceP:   e.SourceP,
		ReshardSourceGen: e.SourceGen,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetCatalogReshard: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetCatalogReshard: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetCatalogReshard: %w", respErr)
		}
	}
	return nil
}

// ApplySetAliasBatch commits a batch of alias mutations to the meta-Raft catalog
// as ONE log entry (so the whole batch applies under a single FSM-lock region —
// atomic). Returns hraft.ErrNotLeader if this node is not the meta leader (the
// caller forwards to the leader; see Node.SetAliases). Mirrors
// ApplySetCatalogReshard's locking/forwarding path.
func (m *MetaRaft) ApplySetAliasBatch(actions []AliasAction, timeout time.Duration) error {
	if m.Raft.State() != hraft.Leader {
		return hraft.ErrNotLeader
	}
	entry, err := encodeLogEntry(LogEntry{
		Op:         OpSetAliasBatch,
		AliasBatch: actions,
	})
	if err != nil {
		return fmt.Errorf("cluster: meta encode SetAliasBatch: %w", err)
	}
	f := m.Raft.Apply(entry, timeout)
	if err := f.Error(); err != nil {
		return fmt.Errorf("cluster: meta apply SetAliasBatch: %w", err)
	}
	if resp := f.Response(); resp != nil {
		if respErr, ok := resp.(error); ok {
			return fmt.Errorf("cluster: meta FSM SetAliasBatch: %w", respErr)
		}
	}
	return nil
}

// Close shuts down the meta-Raft instance and releases its resources.
// All three sub-resources are closed unconditionally; errors are aggregated.
func (m *MetaRaft) Close() error {
	var errs []error
	if err := m.Raft.Shutdown().Error(); err != nil {
		errs = append(errs, fmt.Errorf("cluster: meta shutdown: %w", err))
	}
	if tr, ok := m.transport.(interface{ Close() error }); ok {
		if err := tr.Close(); err != nil {
			errs = append(errs, fmt.Errorf("cluster: meta transport close: %w", err))
		}
	}
	if err := m.logStore.Close(); err != nil {
		errs = append(errs, fmt.Errorf("cluster: meta log store close: %w", err))
	}
	return errors.Join(errs...)
}

// peerSlicesEqual compares two Peer slices by NodeID, RaftAddr, ServerAddr
// without regard to order-sensitivity beyond what's actually stored.
func peerSlicesEqual(a, b []Peer) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// toRaftServers converts a Peer slice to a hashicorp/raft Server slice,
// appending idSuffix to each NodeID. Used to derive distinct Raft
// server IDs for each shard group / meta from the shared Peer list.
func toRaftServers(peers []Peer, idSuffix string) []hraft.Server {
	out := make([]hraft.Server, 0, len(peers))
	for _, p := range peers {
		out = append(out, hraft.Server{
			ID:      hraft.ServerID(p.NodeID + idSuffix),
			Address: hraft.ServerAddress(p.RaftAddr),
		})
	}
	return out
}

// waitForAnyLeader blocks until r reports a non-empty leader address
// or the timeout elapses.
func waitForAnyLeader(r *hraft.Raft, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		addr, _ := r.LeaderWithID()
		if addr != "" {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	return fmt.Errorf("cluster: no leader within %s", timeout)
}
