// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/hashicorp/raft"
)

// MetaFSM is the meta-Raft state machine. Holds cluster-structural
// state and serializes Apply / Snapshot / Restore.
type MetaFSM struct {
	mu    sync.RWMutex
	state State
	// nodeID identifies which node owns this FSM (the cluster NodeID, e.g. "n2").
	// Carried solely so a test-only apply observer can target ONE node's
	// catalog-cutover apply (see metaApplyCatalogGate). Empty in unit-test FSMs
	// constructed via NewMetaFSM() with no id; production sets it from cfg.NodeID.
	nodeID string
	// appliedIndex is the index of the last COMMAND log entry this FSM has applied
	// — the FSM command frontier the linearizable meta-readIndex barrier polls.
	// hashicorp/raft calls Apply ONLY for LogCommand entries (never the election
	// no-op or config entries), so this is precisely the highest command index a
	// follower has materialized; advanced at the END of Apply via advanceApplied
	// (monotonic-max CAS, never regresses). Atomic because the barrier reads it
	// concurrently from the read path while Raft's single Apply goroutine writes it.
	// Persisted into / restored from the snapshot (see Snapshot/Restore + State.LastIndex)
	// so a snapshot-restored follower does NOT under-report 0 and wait forever for a
	// frontier <= the snapshot index. Mirrors shard/fsm.go's appliedIndex tracker.
	appliedIndex atomic.Uint64
	// leaseRenewObserver, when non-nil, is invoked from Apply for each (shard,epoch)
	// in an OpShardLeaseRenew liveness beacon whose CURRENT committed primary is
	// exactly the beaconing node at exactly that epoch (epoch/primary-guarded, like
	// OpSetShardISR). It is the leader-local liveness hook the Plan-4b failover
	// ticker consumes: the node's pbFailoverTracker stamps "last saw a renewal for
	// this shard/epoch" on its OWN monotonic clock. Two correctness properties:
	//   1. It mutates NO replicated state — the beacon is INERT in the FSM (no
	//      snapshot/State()/epoch change), so a node with no observer set (every node
	//      when PBAutoFailover is off, when no beacon op is ever logged anyway)
	//      applies the beacon as a pure no-op ⇒ byte-identical replicated state.
	//   2. It fires while the FSM write lock (m.mu) is held, so it is a LEAF callback:
	//      it may take ONLY the tracker's own mutex and MUST NOT call back into this
	//      MetaFSM (State/Apply/any m.mu-taking method) or it would self-deadlock.
	// Guarded by m.mu (set under the write lock, read under it in Apply) so it is
	// race-free even though the meta-Raft goroutine may already be applying.
	leaseRenewObserver func(shardID int, epoch uint64)
}

// SetLeaseRenewObserver installs (or clears with nil) the leader-local
// OpShardLeaseRenew liveness observer (leaseRenewObserver). Taken under the FSM
// write lock so it is race-free against a concurrent Apply. The installed fn is a
// LEAF callback (fires while m.mu is held) — it may take only the tracker's own
// mutex and MUST NOT re-enter the MetaFSM. Wired only when PBAutoFailover is on.
func (m *MetaFSM) SetLeaseRenewObserver(fn func(shardID int, epoch uint64)) {
	m.mu.Lock()
	m.leaseRenewObserver = fn
	m.mu.Unlock()
}

// AppliedIndex returns the index of the last COMMAND log entry this meta-FSM has
// fully applied (its command frontier). 0 before any command is applied. Read
// concurrently by the linearizable meta-readIndex barrier.
func (m *MetaFSM) AppliedIndex() uint64 { return m.appliedIndex.Load() }

// advanceApplied monotonically raises the applied command index to idx. Raft
// applies command entries in index order on a single goroutine, so this is
// effectively a store; the max-guard is belt-and-suspenders against an
// out-of-order or stale idx ever lowering it (mirrors shard/fsm.go:98-108).
func (m *MetaFSM) advanceApplied(idx uint64) {
	for {
		cur := m.appliedIndex.Load()
		if idx <= cur {
			return
		}
		if m.appliedIndex.CompareAndSwap(cur, idx) {
			return
		}
	}
}

// NewMetaFSM constructs an empty FSM with no node identity (unit tests). The
// production meta-Raft path uses newMetaFSMForNode so the per-node test apply
// gate can target a specific node.
func NewMetaFSM() *MetaFSM {
	return &MetaFSM{}
}

// newMetaFSMForNode constructs an empty FSM tagged with the owning node's ID so a
// test-only apply observer (metaApplyCatalogGate) can lag exactly one node's
// catalog-cutover apply. nil/zero gate in production ⇒ identical to NewMetaFSM.
func newMetaFSMForNode(nodeID string) *MetaFSM {
	return &MetaFSM{nodeID: nodeID}
}

// metaApplyCatalogGate, when non-nil, is invoked from MetaFSM.Apply for every
// OpSetCatalogEntry (a catalog partition/generation commit) BEFORE the entry is
// applied to local state, with the owning node's ID and the entry's (collection,
// partitions, generation). It exists SOLELY for tests to deterministically LAG a
// chosen node's apply of the reshard cutover (block in the hook ⇒ that node keeps
// reporting the old gen ⇒ it routes reads to the still-alive, still-dual-written
// old gen, exactly the lagging-follower window the linearizability proof needs).
// nil in production ⇒ zero overhead, behaviour byte-identical. It fires BEFORE the
// FSM write lock is taken, so a blocking gate stalls only THIS node's apply of the
// cutover entry while its prior catalog state stays readable (it keeps reporting
// the old gen) — the intended lag, not a hang. Set/reset under no concurrency.
var metaApplyCatalogGate func(nodeID, collection string, partitions, generation uint32)

// SetMetaApplyCatalogGate installs (or clears with nil) the test-only catalog-gen
// apply observer (metaApplyCatalogGate). Exported so the root-package integration
// tests (different package) can deterministically lag ONE node's cutover apply.
// Test-only; nil in production. Set/reset under no concurrency.
func SetMetaApplyCatalogGate(fn func(nodeID, collection string, partitions, generation uint32)) {
	metaApplyCatalogGate = fn
}

// State returns a deep copy of the current FSM state.
func (m *MetaFSM) State() State {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := m.state
	cp.Members = append([]Peer(nil), m.state.Members...)
	cp.Placement = make([][]string, len(m.state.Placement))
	for i, p := range m.state.Placement {
		cp.Placement[i] = append([]string(nil), p...)
	}
	if m.state.Catalog != nil {
		cp.Catalog = make(map[string]uint32, len(m.state.Catalog))
		for k, v := range m.state.Catalog {
			cp.Catalog[k] = v
		}
	}
	if m.state.CatalogGen != nil {
		cp.CatalogGen = make(map[string]uint32, len(m.state.CatalogGen))
		for k, v := range m.state.CatalogGen {
			cp.CatalogGen[k] = v
		}
	}
	if m.state.CatalogReshard != nil {
		cp.CatalogReshard = make(map[string]ReshardEntry, len(m.state.CatalogReshard))
		for k, v := range m.state.CatalogReshard {
			cp.CatalogReshard[k] = v
		}
	}
	if m.state.Aliases != nil {
		cp.Aliases = make(map[string]string, len(m.state.Aliases))
		for k, v := range m.state.Aliases {
			cp.Aliases[k] = v
		}
	}
	if m.state.ShardEpoch != nil {
		cp.ShardEpoch = make(map[int]uint64, len(m.state.ShardEpoch))
		for k, v := range m.state.ShardEpoch {
			cp.ShardEpoch[k] = v
		}
	}
	if m.state.ShardPrimary != nil {
		cp.ShardPrimary = make(map[int]string, len(m.state.ShardPrimary))
		for k, v := range m.state.ShardPrimary {
			cp.ShardPrimary[k] = v
		}
	}
	if m.state.ShardISR != nil {
		cp.ShardISR = make(map[int][]string, len(m.state.ShardISR))
		for k, v := range m.state.ShardISR {
			cp.ShardISR[k] = append([]string(nil), v...)
		}
	}
	if m.state.ShardFormer != nil {
		cp.ShardFormer = make(map[int]string, len(m.state.ShardFormer))
		for k, v := range m.state.ShardFormer {
			cp.ShardFormer[k] = v
		}
	}
	return cp
}

// CatalogLookup returns one collection's partition count from the catalog
// without copying the whole map. Hot-path read (used by Node.CollectionPartitions
// on every partitioned op); takes only a read lock.
func (m *MetaFSM) CatalogLookup(collection string) (uint32, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.state.Catalog[collection]
	return p, ok
}

// CatalogLookupGen returns one collection's partition count AND its generation
// from the catalog under a single read lock. gen is read from the parallel
// CatalogGen map and is nil-safe (a missing entry / nil map yields generation 0,
// the pre-resplit default). ok mirrors CatalogLookup: true iff the collection has
// a catalog entry (i.e. is partitioned).
func (m *MetaFSM) CatalogLookupGen(collection string) (uint32, uint32, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.state.Catalog[collection]
	gen := m.state.CatalogGen[collection] // nil map read => 0
	return p, gen, ok
}

// CatalogReshardLookup returns one collection's online-reshard state from the
// catalog under a single read lock. nil-safe: a missing entry / nil map yields
// the zero ReshardEntry (Stable). ok=true iff the collection is actively
// resharding (Status!=0); a Stable / absent collection reports (zero, false).
func (m *MetaFSM) CatalogReshardLookup(collection string) (ReshardEntry, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	e := m.state.CatalogReshard[collection] // nil map read => zero value (Stable)
	return e, e.Status != 0
}

// AliasLookup returns the canonical target collection an alias resolves to, from
// this node's local FSM Aliases map under a single read lock. nil-safe: a missing
// entry / nil map yields ("", false), i.e. the name is not an alias. Hot-path
// read used by ResolveAlias at every data-plane lookup.
func (m *MetaFSM) AliasLookup(alias string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	canonical, ok := m.state.Aliases[alias] // nil map read => ("", false)
	return canonical, ok
}

// AliasSnapshot returns a deep copy of the alias map (alias→target) under a read
// lock, for ListAliases. A nil/empty map yields an empty (non-nil) map.
func (m *MetaFSM) AliasSnapshot() map[string]string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]string, len(m.state.Aliases))
	for k, v := range m.state.Aliases {
		out[k] = v
	}
	return out
}

// ServerAddrForRaftAddr returns the client-facing ServerAddr of the member whose
// RaftAddr matches, or "" if none. It scans Members under the read lock WITHOUT
// deep-copying the whole meta state — unlike State(), which allocates ~11
// slices/maps. This is on the NotLeader-redirect hot path (Node.raftToServerAddr),
// which fires for the majority of writes when the client hits a non-leader, so
// the per-call State() deep-copy was a top allocation source.
func (m *MetaFSM) ServerAddrForRaftAddr(raftAddr string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, p := range m.state.Members {
		if p.RaftAddr == raftAddr {
			return p.ServerAddr
		}
	}
	return ""
}

// ShardEpoch returns one shard's current leadership epoch from the primary-backup
// / ISR control plane under a single read lock. nil-safe: a missing entry / nil
// map yields 0 (no leadership generation established yet). Part of the
// control plane (shard/pbisr/DESIGN.md); no data-plane consumer yet.
func (m *MetaFSM) ShardEpoch(shardID int) uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state.ShardEpoch[shardID] // nil map read => 0
}

// ShardPrimary returns one shard's current primary nodeID under a single read
// lock. nil-safe: a missing entry / nil map yields "" (no primary).
func (m *MetaFSM) ShardPrimary(shardID int) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state.ShardPrimary[shardID] // nil map read => ""
}

// ShardISR returns a COPY of one shard's in-sync replica set under a single read
// lock (a copy so a caller mutation cannot alias into FSM state). nil-safe: a
// missing entry / nil map yields nil (no ISR set).
func (m *MetaFSM) ShardISR(shardID int) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	isr := m.state.ShardISR[shardID] // nil map read => nil
	if isr == nil {
		return nil
	}
	return append([]string(nil), isr...)
}

// ShardFormer returns the nodeID designated to bootstrap one shard's Raft group
// under a single read lock. nil-safe: a missing entry / nil map yields "" (no
// former designated, so no node may form this shard).
func (m *MetaFSM) ShardFormer(shardID int) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state.ShardFormer[shardID] // nil map read => ""
}

// Apply runs a single log entry. hashicorp/raft contract.
func (m *MetaFSM) Apply(log *raft.Log) any {
	entry, err := decodeLogEntry(log.Data)
	if err != nil {
		return fmt.Errorf("meta-fsm: decode: %w", err)
	}
	// Advance the command frontier at the END of Apply (deferred so it also covers
	// any panic hraft recovers from in this goroutine, and fires for every command
	// entry — Apply is only ever called for LogCommand). Placed right after the
	// decode succeeds: a malformed entry returns above and does NOT move the frontier.
	// One atomic store on the meta Apply hot path (rare — catalog/membership mutations).
	defer m.advanceApplied(log.Index)
	// Test-only apply gate (nil in production). Fired for a catalog-gen commit
	// BEFORE the write lock is taken so the gate can BLOCK without holding the lock
	// — the node's prior catalog state stays readable (it keeps reporting the old
	// gen) while this node's apply of the cutover is deferred. That is precisely the
	// lagging-follower window: the node routes reads to the still-fresh old gen until
	// the test releases the gate. nil ⇒ no-op, no lock-ordering effect.
	if entry.Op == OpSetCatalogEntry && metaApplyCatalogGate != nil {
		metaApplyCatalogGate(m.nodeID, entry.Collection, entry.Partitions, entry.Generation)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	switch entry.Op {
	case OpSetMembers:
		// Sort by NodeID so State.Members is always ordered and the
		// order-sensitive peerSlicesEqual check in ApplySetMembersIfLeader
		// is deterministic regardless of call-site ordering.
		sorted := append([]Peer(nil), entry.Members...)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].NodeID < sorted[j].NodeID
		})
		// Backfill: if this is the first OpSetMembers to carry ReplicationFactor
		// (ReplicationFactorSet was false on a legacy snapshot) and members/shard
		// count are unchanged, only record the RF — do NOT recompute Placement,
		// which would clobber any per-shard rebalance work stored in the snapshot.
		backfill := !m.state.ReplicationFactorSet &&
			m.state.NumShards == entry.NumShards &&
			peerSlicesEqual(m.state.Members, sorted)
		m.state.Members = sorted
		m.state.NumShards = entry.NumShards
		m.state.ReplicationFactor = entry.ReplicationFactor
		m.state.ReplicationFactorSet = true
		if !backfill {
			// Placement distributes shards across members in replica sets of
			// ReplicationFactor nodes; rf 0 / >= len(members) = full replication.
			m.state.Placement = computePlacement(sorted, entry.NumShards, entry.ReplicationFactor)
		}
		// Seed the structural ISR floor (ISR hardening). Only RAISE it from a
		// real seed (>0); a stray/old OpSetMembers carrying MinISR==0 (raft mode, or
		// a pre-floor snapshot re-commit) must never clobber an established floor.
		if entry.MinISR > 0 {
			m.state.MinISR = entry.MinISR
		}
		return nil
	case OpSetPlacement:
		// Online rebalancing advances placement one shard at a time as each
		// migration commits, so routing follows reality incrementally.
		if entry.ShardID < 0 {
			return fmt.Errorf("meta-fsm: set placement negative shard %d", entry.ShardID)
		}
		// Self-heal the Placement table's size. The bootstrap OpSetMembers entry
		// can be lost to leadership churn (appended on a leader that steps down
		// before it commits, then overwritten), leaving Placement empty; a
		// per-shard commit must still work. entry.NumShards is the cluster's
		// shard count (the coordinator knows it); grow to at least that, and at
		// least past this shard.
		want := entry.NumShards
		if want < entry.ShardID+1 {
			want = entry.ShardID + 1
		}
		if m.state.NumShards < want {
			m.state.NumShards = want
		}
		for len(m.state.Placement) < want {
			m.state.Placement = append(m.state.Placement, nil)
		}
		m.state.Placement[entry.ShardID] = append([]string(nil), entry.Owners...)
		return nil
	case OpSetCatalogEntry:
		if m.state.Catalog == nil {
			m.state.Catalog = make(map[string]uint32)
		}
		m.state.Catalog[entry.Collection] = entry.Partitions
		if m.state.CatalogGen == nil {
			m.state.CatalogGen = make(map[string]uint32)
		}
		m.state.CatalogGen[entry.Collection] = entry.Generation
		return nil
	case OpSetCatalogReshard:
		if m.state.CatalogReshard == nil {
			m.state.CatalogReshard = make(map[string]ReshardEntry)
		}
		if entry.ReshardStatus == 0 {
			// Stable clears the entry so the map stays sparse and lookups of a
			// non-resharding collection are a clean miss.
			delete(m.state.CatalogReshard, entry.Collection)
		} else {
			m.state.CatalogReshard[entry.Collection] = ReshardEntry{
				Status:    entry.ReshardStatus,
				TargetP:   entry.ReshardTargetP,
				TargetGen: entry.ReshardTargetGen,
				SourceP:   entry.ReshardSourceP,
				SourceGen: entry.ReshardSourceGen,
			}
		}
		return nil
	case OpSetAliasBatch:
		// Atomic batch: all N alias mutations apply under the SINGLE mu.Lock()
		// already held here (one Raft log entry = one atomic apply). This is the
		// mechanism for the zero-downtime swap — a {delete prod, create prod→v2}
		// batch never exposes an intermediate state to a concurrent reader.
		if m.state.Aliases == nil {
			m.state.Aliases = make(map[string]string)
		}
		for _, a := range entry.AliasBatch {
			if a.Delete {
				delete(m.state.Aliases, a.Alias)
			} else {
				m.state.Aliases[a.Alias] = a.Canonical
			}
		}
		return nil
	case OpSetShardEpoch:
		// MONOTONIC: only a strictly-higher epoch takes effect. A stale or equal
		// epoch is a benign no-op (return nil, never an error) — replays and
		// out-of-order proposals must not regress the leadership generation. Maps
		// are lazily initialized.
		//
		// ISR: by default a new epoch RESETS the ISR to just {primary} — a fresh
		// leadership generation (a FAILOVER) starts with the promoted node as the
		// sole in-sync member until backups catch up and are re-added via
		// OpSetShardISR. That reset is deliberate and unconditional for a promotion.
		//
		// A SEED (bootstrap / DR restore) instead carries the full ISR in the SAME
		// entry (ApplySetShardSeed). This is what makes the seed ATOMIC: committing
		// (epoch, primary) and (ISR) as two sequential entries left an intermediate
		// COMMITTED state with a SINGLETON ISR, and a primary whose local FSM read
		// that intermediate state would ack a write against an ISR NARROWER than the
		// committed one — i.e. on itself alone — silently losing it on failover.
		// One entry, one apply, no intermediate state.
		//
		// The carried ISR is honored only when it is non-empty AND contains the
		// primary; anything else falls back to the {primary} reset, so a malformed
		// seed can never install an ISR that excludes the node named primary. NO
		// MinISR floor check here (unlike OpSetShardISR): the fallback is a
		// size-1 ISR, so rejecting a below-floor seed would install something
		// strictly NARROWER, not safer.
		if entry.Epoch <= m.state.ShardEpoch[entry.ShardID] {
			return nil
		}
		if m.state.ShardEpoch == nil {
			m.state.ShardEpoch = make(map[int]uint64)
		}
		if m.state.ShardPrimary == nil {
			m.state.ShardPrimary = make(map[int]string)
		}
		if m.state.ShardISR == nil {
			m.state.ShardISR = make(map[int][]string)
		}
		m.state.ShardEpoch[entry.ShardID] = entry.Epoch
		m.state.ShardPrimary[entry.ShardID] = entry.Primary
		m.state.ShardISR[entry.ShardID] = seedISRForEpoch(entry.Primary, entry.ISR)
		return nil
	case OpSetShardISR:
		// EPOCH-GUARDED: only apply an ISR update whose epoch EXACTLY matches the
		// shard's current epoch. A stale-epoch update is a no-op (return nil) — this
		// defends against a fenced/stale primary mutating the ISR set after a newer
		// epoch has been established (hazards H3/H6 in shard/pbisr/DESIGN.md). Store
		// a copy of the ISR slice so a later mutation of entry.ISR cannot alias into
		// FSM state. Map is lazily initialized (a matching epoch implies a prior
		// OpSetShardEpoch already ran, but stay defensive).
		if entry.Epoch != m.state.ShardEpoch[entry.ShardID] {
			return nil
		}
		// STRUCTURAL FLOOR (ISR hardening): the FSM itself refuses an ISR set
		// below the durability floor, so a buggy shrink/grow driver can NEVER commit
		// a below-floor ISR even if its own decidePBShrink floor is bypassed. An
		// empty ISR is ALWAYS rejected (a shard with no in-sync members is a lost
		// shard); a non-empty set below MetaState.MinISR is rejected when a floor is
		// seeded (MinISR>0). Reject = benign no-op (return nil), like the epoch guard
		// above — the offending op simply does not take effect. This does NOT touch
		// the election reset: OpSetShardEpoch (a DIFFERENT op) resets the ISR to
		// {primary} and is not floor-checked, so a failover to a lone survivor still
		// proceeds; grow then re-widens the ISR back above the floor.
		if len(entry.ISR) == 0 || (m.state.MinISR > 0 && len(entry.ISR) < m.state.MinISR) {
			return nil
		}
		if m.state.ShardISR == nil {
			m.state.ShardISR = make(map[int][]string)
		}
		m.state.ShardISR[entry.ShardID] = append([]string(nil), entry.ISR...)
		return nil
	case OpShardLeaseRenew:
		// Primary-liveness beacon (failover). This case MUTATES NO REPLICATED
		// STATE: it fires the leader-local observer only, so the op is INERT in the FSM
		// (no epoch/primary/ISR change, nothing in the snapshot) — a follower or a node
		// with no observer applies it as a pure no-op, and the replicated state stays
		// byte-identical to a cluster that never beacons.
		//
		// EPOCH/PRIMARY-GUARDED (like OpSetShardISR): a pair counts as liveness only if
		// its node is EXACTLY the shard's current committed primary at EXACTLY its
		// current epoch. This defends the failover decision against a fenced/stale
		// ex-primary's late beacon: after a promotion bumped the shard to (Q, E+1), a
		// straggling (P, E) beacon fails both guards and is ignored, so it can never
		// refresh liveness for a shard the cluster has already moved past.
		//
		// The observer runs while m.mu is held (this Apply's write lock). It is a LEAF
		// call: it takes only the tracker's own mutex and never re-enters MetaFSM — see
		// the leaseRenewObserver field comment for the lock-ordering contract.
		if m.leaseRenewObserver != nil {
			for _, pair := range entry.LeaseRenew {
				if entry.Node != "" &&
					entry.Node == m.state.ShardPrimary[pair.ShardID] &&
					pair.Epoch == m.state.ShardEpoch[pair.ShardID] {
					m.leaseRenewObserver(pair.ShardID, pair.Epoch)
				}
			}
		}
		return nil
	case OpSetShardFormer:
		// WRITE-ONCE designation of the single owner that bootstraps this shard's
		// Raft group (see State.ShardFormer). Returns true when THIS apply installed
		// the designation and false when one already existed, so the proposer can
		// tell "I am the former" from "someone else already formed this shard" —
		// decided by the meta log, not by a local race.
		//
		// Ignoring a repeat is the safety property, not an optimization: it is what
		// stops a fresh-disk node that rejoins an ESTABLISHED cluster from claiming
		// formation of a shard whose group already exists and creating a rival one.
		if entry.ShardID < 0 {
			return fmt.Errorf("meta-fsm: set shard former negative shard %d", entry.ShardID)
		}
		if entry.Node == "" {
			return fmt.Errorf("meta-fsm: set shard former empty node for shard %d", entry.ShardID)
		}
		if _, exists := m.state.ShardFormer[entry.ShardID]; exists {
			return false
		}
		if m.state.ShardFormer == nil {
			m.state.ShardFormer = make(map[int]string)
		}
		m.state.ShardFormer[entry.ShardID] = entry.Node
		return true
	default:
		return fmt.Errorf("meta-fsm: unknown op %d", entry.Op)
	}
}

// seedISRForEpoch resolves the ISR an OpSetShardEpoch apply installs: the entry's
// carried ISR when it is a well-formed seed (non-empty and containing primary),
// otherwise the {primary} promotion reset. Returns a fresh slice, so entry.ISR can
// never alias into FSM state. Pure.
func seedISRForEpoch(primary string, isr []string) []string {
	for _, m := range isr {
		if m == primary {
			return append([]string(nil), isr...)
		}
	}
	return []string{primary}
}

// Snapshot returns a snapshot of the current state.
func (m *MetaFSM) Snapshot() (raft.FSMSnapshot, error) {
	b, err := m.SnapshotBytes()
	if err != nil {
		return nil, err
	}
	return &metaSnapshot{data: b}, nil
}

// SnapshotBytes returns the encoded MetaFSM catalog state — the SAME
// serialization Snapshot() persists (state + the stamped command frontier) —
// as raw bytes, so cluster.Node.BackupMetaCatalog can ship it to object storage
// out-of-band (under the FSM read lock, no raft snapshot sink) and a restore can
// feed it back through MetaFSM.Restore / m.Raft.Restore. Safe to call from any
// goroutine.
func (m *MetaFSM) SnapshotBytes() ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	// Capture the command frontier INTO the snapshot blob (State.LastIndex). After
	// an install-snapshot raft sets its lastApplied to the snapshot index but our
	// atomic would otherwise read 0 — a snapshot-restored follower would then wait
	// forever for any frontier <= the snapshot index. Persisting it here closes that.
	// Snapshot the state under the read lock and stamp LastIndex onto the copy (not
	// m.state) so we never mutate live FSM state from a read path.
	snap := m.state
	snap.LastIndex = m.appliedIndex.Load()
	return encodeState(snap)
}

// Restore replaces state from a snapshot reader.
func (m *MetaFSM) Restore(r io.ReadCloser) error {
	defer func() { _ = r.Close() }()
	b, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	s, err := decodeState(b)
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.state = s
	m.mu.Unlock()
	// Restore the command frontier from the snapshot blob. An OLD-format snapshot
	// (written before LastIndex existed) gob-decodes it as 0 → the tracker starts at
	// 0 and catches up on the next applied command (acceptable: the node replays the
	// log after restore). advanceApplied keeps this monotonic — a stale/old snapshot
	// can never lower an already-higher frontier.
	m.advanceApplied(s.LastIndex)
	return nil
}

type metaSnapshot struct{ data []byte }

func (s *metaSnapshot) Persist(sink raft.SnapshotSink) error {
	if _, err := sink.Write(s.data); err != nil {
		_ = sink.Cancel()
		return err
	}
	return sink.Close()
}

func (s *metaSnapshot) Release() {}

var _ raft.FSM = (*MetaFSM)(nil)
