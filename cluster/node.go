// SPDX-License-Identifier: Apache-2.0

// Package cluster implements a single Rostam process hosting N
// independent Raft-replicated shards. Each shard is its own
// *shard.Store; cluster.Node routes Call by key via xxhash mod NumShards.
package cluster

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft/fabric"
	"github.com/rostamlabs/rostam/raft/mux"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/shard/pbisr"
	"github.com/rostamlabs/rostam/wasm"
)

// raftToServerAddr translates a Raft transport address to the
// corresponding client-facing server address by consulting the
// meta-Raft FSM's Members table. Returns "" when no mapping is
// available (e.g. unknown raft addr, or single-node mode where
// meta is nil). Used to fix up NotLeader hints before they reach
// the client, which speaks the Rostam wire protocol — not Raft.
func (n *Node) raftToServerAddr(raftAddr string) string {
	if raftAddr == "" {
		return ""
	}
	if n.meta != nil {
		// Targeted, allocation-free lookup — NOT State() (which deep-copies the
		// whole meta state); this is the NotLeader-redirect hot path.
		if sa := n.meta.FSM.ServerAddrForRaftAddr(raftAddr); sa != "" {
			return sa
		}
	}
	// Fall back to the static peer list. meta-Raft Members can be empty when the
	// bootstrap OpSetMembers commit was lost to early election churn (the same
	// fragility OpSetPlacement self-heals); cfg.Peers always carries the
	// RaftAddr→ServerAddr mapping, so NotLeader hints stay resolvable — which is
	// what the HTTP/gRPC leader-following redirect relies on.
	for _, p := range n.cfg.Peers {
		if p.RaftAddr == raftAddr {
			return p.ServerAddr
		}
	}
	return ""
}

// Node is a single Rostam process exposing N sharded sub-stores.
// All exported methods are safe for concurrent use.
type Node struct {
	cfg Config

	// shardMu guards the shards slice's elements, which become mutable at runtime
	// once online rebalancing can add/remove a shard on a live node. Readers load
	// a slot pointer under RLock; addShardOwner/removeShardOwner mutate under Lock.
	shardMu sync.RWMutex
	shards  []*shard.Store

	// placement[shard] = owner NodeIDs (replica set). A nil shards[i] means this
	// node does not host shard i; ops for it are forwarded to an owner.
	placement [][]string

	// peerClients are lazily-built forwarding clients keyed by a peer's
	// ServerAddr (each follows NotLeader to that shard group's leader, an owner).
	peerMu      sync.Mutex
	peerClients map[string]*client.Client

	// mux and meta are populated only in multi-node mode (cfg.Peers != nil).
	// Exactly one of mux/fabric is non-nil: mux is the default per-group
	// NetworkTransport StreamLayer; fabric is the multiplexed batching transport
	// selected by cfg.RaftTransport=="fabric". buildShardConfig routes each
	// shard's group through whichever is set.
	mux    *mux.StreamLayer
	fabric *fabric.Fabric
	meta   *MetaRaft

	// Primary-backup (ReplicationMode=="pb") wiring. All nil/zero in raft mode,
	// so the raft construction path is byte-identical. Populated by newMultiNode's
	// PB block and consumed by buildShardConfig's PB branch:
	//   pbTransport — this node's inter-node PB replication server/dialer.
	//   pbControl   — the MetaRaft-authoritative pbisr.Control (one per node).
	//   pbAddrOf    — node-ID → PBAddr map for pbResolvingTransport.
	//   pbNow       — the shared monotonic clock handed to BOTH every engine (via
	//                 shard.Config.PBClock → pbisr.WithClock) and the leaseKeeper,
	//                 so the lease expiry math lines up on one clock source.
	//   pbEngines   — owned shard index → its *pbisr.Engine, collected via the
	//                 PBRegister hook; handed to the leaseKeeper (also read by tests).
	//   leaseKeeper — the static local-read lease renewer for owned PB primaries.
	//   pbTracker   — leader-local PB primary-liveness memory (failover).
	//   pbBeacon    — this node's primary-liveness beacon goroutine (flag-gated).
	pbTransport *pbisr.NetTransport
	pbControl   *metaControl
	pbAddrOf    map[string]string
	pbNow       func() int64
	pbEngines   map[int]*pbisr.Engine
	leaseKeeper *leaseKeeper
	pbTracker   *pbFailoverTracker
	pbBeacon    *pbBeacon
	pbFailover  *pbFailover
	pbShrink    *pbShrinkDriver
	pbGrow      *pbGrowDriver
	// pbMetaContactStaleness is the effective follower quorum-contact freshness
	// bound used by confirmMetaView (config PBMetaContactStalenessMs, or the
	// metaContactStaleness default when 0). Stored per-node so the failover test can
	// shrink it alongside the other PB timings.
	pbMetaContactStaleness time.Duration

	// pbSeedStop signals the background PB control-plane seeder goroutine to exit
	// (closed in Close). nil in raft mode and on restart (Bootstrap=false).
	pbSeedStop chan struct{}
	// pbSeedWg tracks the seeder goroutine so Close() can wait for it to exit
	// before tearing down meta-Raft (which the goroutine uses).
	pbSeedWg sync.WaitGroup

	// kvIndexStop signals the KV index observer goroutine to exit; kvIndexWg
	// tracks it so Close waits for it before the shard stores it walks are torn
	// down. kvIndexStopOnce makes stopKVIndexObserver idempotent, so a test may
	// freeze the observer and Close may still run its own stop. nil in
	// single-node mode (no meta catalog to derive definitions from).
	kvIndexStop     chan struct{}
	kvIndexWg       sync.WaitGroup
	kvIndexStopOnce sync.Once
	// kvIndexApplyMu serialises one whole install-and-backfill pass. The observer
	// runs passes on its own goroutine and Close can race one; serialising them
	// keeps two walks off the same definition, which the Set's generation guard
	// survives but which would waste a full keyspace walk.
	kvIndexApplyMu sync.Mutex
	// kvIndexInstalled is the fingerprint (definitions + hosted groups) the last
	// COMPLETED pass installed. A pass whose fingerprint matches skips the
	// per-group Install entirely — the poll fires on every meta write, and PB
	// liveness beacons alone would otherwise rebuild every Set once a second for a
	// catalog that never changed. Guarded by kvIndexApplyMu, which a whole pass
	// holds.
	kvIndexInstalled string
	// kvIndexPasses and kvIndexInstalls count passes run and passes that actually
	// re-installed. They are diagnostics with no Stats field: what they exist for
	// is to make "the observer stopped" and "the pass skipped its install"
	// assertable, neither of which is visible in any other number.
	kvIndexPasses   atomic.Uint64
	kvIndexInstalls atomic.Uint64
	// kvIndexWalkGates registers in-flight backfill walks per shard group so a
	// shard removal can DRAIN them before closing the store the walk is reading
	// out of. See cluster/kv_index_walkgate.go for why a drain rather than a flag.
	kvIndexWalkMu    sync.Mutex
	kvIndexWalkGates map[int]*kvIndexWalkGate

	// KV index counters behind Stats().KVIndex. kvBackfills/kvBackfillKeys record
	// what the observer's walks have cost; kvIndexRejects COUNTS reject events and
	// kvIndexRejectedDefs GAUGES how many are broken right now (see
	// kvIndexDefsFromCatalog for why both).
	// kvIndexVerifyMisses and kvIndexReconcileDrops hold the counts that no
	// longer have a hosted group to be read from: RemoveShardOwner folds a
	// departing group's totals in here before dropping its index, so
	// Stats().KVIndex.VerifyMisses and .ReconcileDrops (each this plus the sum
	// over hosted groups) never decrease. Both are maintained on the per-shard
	// kvindex.Set — the query leaf runs in ops, and the reconcile pass runs on a
	// goroutine the store owns — so these node counters are only ever the
	// residue of groups that have left.
	kvBackfills           atomic.Uint64
	kvBackfillKeys        atomic.Uint64
	kvIndexRejects        atomic.Uint64
	kvIndexRejectedDefs   atomic.Uint64
	kvIndexVerifyMisses   atomic.Uint64
	kvIndexReconcileDrops atomic.Uint64

	// formationStop signals the shard-formation seeder and driver goroutines to
	// exit (closed in Close). See cluster/shard_formation.go: these form the Raft
	// groups whose owner set excludes the -bootstrap node, which nothing else does.
	formationStop chan struct{}
	// formationWg tracks the seeder and driver goroutines so Close() can wait for
	// them to exit before closing shards and meta-Raft.
	formationWg sync.WaitGroup

	// metaFrontier coalesces concurrent follower __meta_readindex__ forwards on THIS
	// node into one leader RTT, guaranteeing no read accepts a frontier captured
	// before it arrived (see meta_readindex_coalesce.go).
	metaFrontier metaFrontierCoalescer

	// shardFrontier coalesces concurrent follower __shard_readindex__ forwards on THIS
	// node into one leader RTT PER SHARD, so N concurrent ConsistencyBoundedStaleness
	// reads for the same shard share one leader frontier ping (see
	// shard_readindex_coalesce.go).
	shardFrontier shardFrontierCoalescer

	// adminOps are node-local rebalancing handlers (online-rebalancing slice 4),
	// populated in multi-node mode. Call dispatches a matching op directly on
	// this node, bypassing key→shard routing (an admin op targets the node it was
	// sent to, not a key's shard owner). nil in single-node mode.
	adminOps map[string]func([]byte) ([]byte, error)

	wasmRT *wasm.Runtime

	// wasmApplyMu serializes applyWASMRegistration across shard groups. A WASM
	// registration is replicated to EVERY shard Raft group, so each group's FSM
	// apply loop — an independent goroutine — invokes the hook for the same
	// module. They all write the same two files and mutate the same node-wide
	// runtime + ops registry, so they must not interleave.
	wasmApplyMu sync.Mutex

	// wasmState records which registration is installed under each op name; it is
	// the reference point for the order-independent (Epoch, fingerprint)
	// convergence rule and the source of the snapshot payload. Guarded by
	// wasmApplyMu.
	wasmState *wasmState

	// wasmRestorePending buffers snapshot WASM sections that arrived BEFORE the
	// node's WASM runtime existed. A shard's Raft node can install a snapshot
	// from within shard.New, and the shard stores are constructed before wasmRT
	// is created, so the restore hook has nowhere to install to yet. Dropping
	// those blobs would silently reintroduce the missing-module bug on exactly
	// the restart-from-snapshot path this whole mechanism exists for, so they are
	// held and drained once the runtime is up. Each entry keeps the shard group it
	// came from: that is the route-gate provenance, and it cannot be recovered
	// later. Guarded by wasmApplyMu.
	wasmRestorePending []pendingWASMRestore

	// wasmGate is the published, copy-on-write route-gate snapshot: op name → the
	// shard groups whose log is known to carry its registration. Call reads it
	// with one atomic load; nil until the first replicated WASM registration is
	// installed. See checkWASMRouteGate.
	wasmGate atomic.Pointer[wasmGateSnapshot]

	// wasmGateRefusals counts Calls checkWASMRouteGate has declined since process
	// start. The gate answers with a client-visible retryable error instead of
	// halting, so without a counter a permanently-wedged (op, group) pair is
	// invisible from the server side. Surfaced via Stats().WASMGate.
	wasmGateRefusals atomic.Uint64

	// wasmBlobPushAcks / wasmBlobPushSkips count the per-member legs of the
	// pre-registration blob push (pushWASMBlob) since process start: acked, and
	// skipped because the member could not be reached.
	//
	// A skip is deliberately NOT an error — an unreachable member must not block a
	// registration — which is exactly why it needs a counter. The reply payload
	// carries the same information, but only PER CALL and only to the caller that
	// made it; a member that has been skipped by every registration since it was
	// last restarted shows up as a trend nobody is looking at, in reports nobody
	// kept. Surfaced via Stats().WASMBlobPush.
	wasmBlobPushAcks  atomic.Uint64
	wasmBlobPushSkips atomic.Uint64

	// wasmBlobFloorFailures counts registrations REFUSED because the module could
	// not be delivered to a majority of cluster members (see pushWASMBlob's
	// durability floor). A non-zero value is not a client bug: it means the
	// cluster was too degraded to accept a registration safely, and it is the one
	// push counter that corresponds to a call the client saw fail.
	wasmBlobFloorFailures atomic.Uint64

	// wasmFetching deduplicates in-flight blob fetches PER FINGERPRINT, so N
	// groups blocked on one module issue ONE fetch. wasmFetchStop ends them at
	// Close; the counters make the loop observable.
	wasmFetchMu     sync.Mutex
	wasmFetching    map[[sha256.Size]byte]*wasmBlobFetch
	wasmFetchStop   chan struct{}
	wasmFetchStarts atomic.Uint64
	wasmFetchOKs    atomic.Uint64

	// wasmBlockLive is the live set of classRetry blocks, guarded by wasmBlockMu
	// and republished into wasmBlocks (copy-on-write) on every change. Stats reads
	// ONLY the published pointer, so "can I still see the block" never depends on
	// a lock a blocked apply path might be near. See wasm_blob_fetch.go.
	wasmBlockMu    sync.Mutex
	wasmBlockLive  map[blockKey]*blockRecord
	wasmBlocks     atomic.Pointer[[]WASMBlockedApply]
	wasmBlockTotal atomic.Uint64

	// wasmRetireUnrefSince is the retirement sweeper's clock: hex fingerprint →
	// the first sweep that observed the blob referenced by NOTHING on this node.
	// It is rebuilt from the directory listing on every pass (so it cannot grow
	// unboundedly) and it is PROCESS-LOCAL on purpose — a restart restarts every
	// blob's window, which delays retirement and never hastens it. Guarded by
	// wasmRetireMu; nil and untouched when retirement is off. See
	// wasm_blob_retire.go.
	wasmRetireMu         sync.Mutex
	wasmRetireUnrefSince map[string]time.Time

	// Retirement counters, surfaced via Stats().WASMBlobRetire. Sweeps separates
	// "off" from "on and finding nothing", Pending is how many blobs are
	// currently waiting out their window, and Retired is how many files have been
	// removed since process start.
	wasmRetireSweeps  atomic.Uint64
	wasmBlobsRetired  atomic.Uint64
	wasmRetirePending atomic.Int64

	closeOnce sync.Once
	closeErr  error
}

// New constructs a Node. When cfg.Peers is non-empty, the multi-node
// path runs: a shared mux StreamLayer is created, the meta-Raft group
// is started, and each shard is wired to a per-group sub-layer.
// Otherwise the single-node path runs (in-memory transports,
// per-shard bootstrap-as-single-node).
func New(cfg Config) (*Node, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.isMultiNode() {
		return newMultiNode(cfg)
	}
	return newSingleNode(cfg)
}

// isMultiNode reports whether cfg requests the multi-node path.
func (c Config) isMultiNode() bool { return len(c.Peers) > 0 }

// newSingleNode is the single-node path: per-shard self-bootstrap, no
// shared transport, no meta-Raft.
func newSingleNode(cfg Config) (*Node, error) {
	n := &Node{
		cfg:    cfg,
		shards: make([]*shard.Store, cfg.NumShards),
		// Built before the first shard.New because a shard's Raft node can install
		// a snapshot from inside that constructor, and the restore hook needs
		// somewhere to record what it installed.
		wasmState:     newWASMState(),
		wasmFetchStop: make(chan struct{}),
	}
	for i := range cfg.NumShards {
		subCfg := cfg.ShardCfg
		subCfg.NodeID = fmt.Sprintf("%s-shard-%04d", cfg.NodeID, i)
		subCfg.DataDir = filepath.Join(cfg.DataDir, fmt.Sprintf("shard-%04d", i))
		subCfg.Bootstrap = cfg.Bootstrap
		subCfg.Cache.NumShards = 1
		if subCfg.DataDir != "" {
			subCfg.Cache.DataDir = filepath.Join(subCfg.DataDir, "cache")
		}
		subCfg.Ops = cfg.Ops
		// Every WASM hook is bound to THIS group: the snapshot must say which
		// registrations this group's log carries, the restore must attribute what
		// it installs to this group, and the apply dispatcher must report this
		// group to the register hook. See checkWASMRouteGate.
		subCfg.ShardIndex = i
		grp := i
		subCfg.WASMSnapshot = func() []byte { return n.snapshotWASMState(grp) }
		subCfg.WASMRestore = func(b []byte) error { return n.restoreWASMState(grp, b) }
		// The classRetry block hooks: an apply that names a module version this
		// node does not hold parks, and these are what fetch it and make the park
		// visible. Neither blocks and neither returns holding wasmApplyMu — see
		// Node.onShardApplyRetry.
		subCfg.OnApplyRetry = n.onShardApplyRetry
		subCfg.OnApplyRetryCleared = n.onShardApplyRetryCleared

		store, err := shard.New(subCfg)
		if err != nil {
			// Rollback any sub-stores already created.
			for j := 0; j < i; j++ {
				_ = n.shards[j].Close() //nolint:errcheck,gosec // best-effort rollback
			}
			return nil, fmt.Errorf("cluster: shard %d: %w", i, err)
		}
		// Wire the ConsistencyBoundedStaleness follower-side leader-frontier hook
		// (coalesced per shard). Single-node (always leader) takes the leader path and
		// never invokes it. idx is captured per iteration.
		idx := i
		store.SetLeaderFrontierFn(func(dl time.Time) (uint64, error) {
			return n.shardFrontier.do(idx, dl, func(d time.Time) (uint64, error) {
				return n.fetchShardLeaderFrontier(idx, d)
			})
		})
		// A scan-mode kv_query walks this store's whole keyspace and aliases its
		// live mmap, so it goes behind the same per-group gate the index backfill
		// registers with — installed here, at the one moment the store exists and
		// nothing can be walking it yet. See installKVQueryScanGate.
		n.installKVQueryScanGate(i, store)
		n.shards[i] = store
	}

	// Register the __topology__ shardless op so smart clients can
	// discover a single-node cluster's layout the same way they discover
	// a multi-node one. The library shim relies on this for the
	// networked backend's IsLeader/LeaderAddr accessors.
	// Idempotent: callers may reuse a registry across restarts.
	if err := ops.RegisterTopology(cfg.Ops, n.Topology); err != nil && !errors.Is(err, ops.ErrDuplicateOp) {
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec // best-effort rollback
			}
		}
		return nil, fmt.Errorf("cluster: register __topology__: %w", err)
	}

	// WASM runtime + Raft hook.
	wasmRT, err := wasm.NewRuntime()
	if err != nil {
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec
			}
		}
		return nil, fmt.Errorf("cluster: new wasm runtime: %w", err)
	}
	n.setWASMRuntime(wasmRT)

	// Register the __register_wasm__ FSM hook. Idempotent across restarts.
	if err := ops.RegisterWASMRegisterOp(cfg.Ops, func(shardIdx int, r ops.WASMRegistration) error {
		// One registration is proposed to every shard group, so this hook runs
		// once per hosted group — concurrently, from each group's own FSM apply
		// goroutine. Serialize: applyWASMRegistration is idempotent but not
		// concurrency-safe against itself (shared files, runtime, registry).
		//
		// shardIdx is the group whose log carried THIS apply. Recording it is what
		// earns this node the right to propose invocations of the op into that
		// group's log, and nothing else earns it (see checkWASMRouteGate).
		//
		// THE PUBLISH IS UNCONDITIONAL, deliberately. applyWASMRegistration has a
		// path that mutates wasmState and THEN returns an error: a per-group
		// contract refusal (checkWASMGroupContract) is reported after the node-wide
		// install has already been recorded. Skipping the publish on a non-nil error
		// would leave the lock-free table the route gate and
		// wasm.Runtime.resolveModuleForInvoke read stale with respect to the state
		// that was actually committed. It happens to be benign today only via an
		// unstated chain: the published snapshot is built from `replicated` and
		// `groups` alone (wasmBindingSnapshot), a refusal restores the previous
		// `groups`, and `replicated` can only flip false→true for a name that had
		// no prior binding at all — which is exactly the case in which no refusal
		// can fire. Nothing checks that chain and nothing would re-derive it on the
		// next edit. Publishing is idempotent and purely derived from wasmState, so
		// deferring it removes the invariant instead of relying on it.
		n.wasmApplyMu.Lock()
		defer n.wasmApplyMu.Unlock()
		defer n.publishWASMGateLocked()
		return applyWASMRegistration(cfg.DataDir, n.wasmRT, cfg.Ops, n.wasmState, r, shardIdx, n.prefetchWASMBlob)
	}); err != nil && !errors.Is(err, ops.ErrDuplicateOp) {
		_ = n.wasmRT.Close()
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec
			}
		}
		return nil, fmt.Errorf("cluster: register __register_wasm__: %w", err)
	}

	if len(cfg.WASMModules) > 0 {
		// Iterate what loadWASMModules actually LOADED, not cfg: it skips any
		// config module whose name already has a REPLICATED sidecar on disk, and
		// registering that name from cfg here would install the config module's
		// Kind and key extractor and then make reloadWASMModulesFromDisk's
		// ErrDuplicateOp branch drop the replicated module entirely.
		loaded, err := loadWASMModules(cfg.DataDir, cfg.WASMModules, n.wasmRT, n.wasmState)
		if err != nil {
			_ = n.wasmRT.Close()
			for j := 0; j < cfg.NumShards; j++ {
				if n.shards[j] != nil {
					_ = n.shards[j].Close() //nolint:errcheck,gosec
				}
			}
			return nil, fmt.Errorf("cluster: load WASM modules: %w", err)
		}
		for _, lm := range loaded {
			c := lm.Registration
			ke := ops.WASMKeyExtractor()
			if err := wasm.RegisterModule(cfg.Ops, n.wasmRT, c.Name, lm.ModuleID, c.Kind, ke); err != nil && !errors.Is(err, ops.ErrDuplicateOp) {
				_ = n.wasmRT.Close()
				for j := 0; j < cfg.NumShards; j++ {
					if n.shards[j] != nil {
						_ = n.shards[j].Close() //nolint:errcheck,gosec
					}
				}
				return nil, fmt.Errorf("cluster: register module %q: %w", c.Name, err)
			}
		}
	}

	// Reload from disk (modules registered via Raft Apply in a previous
	// process) and install any snapshot-carried registrations that arrived
	// before the runtime existed. Config-loaded modules win on conflict.
	if err := n.finishWASMSetup(cfg.DataDir); err != nil {
		_ = n.wasmRT.Close()
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec
			}
		}
		return nil, fmt.Errorf("cluster: reload WASM modules: %w", err)
	}
	// Started LAST, after every module and binding this node knows about is
	// loaded: the sweeper's whole rule is "referenced by nothing", so it must not
	// run while the reference set is still being built. A no-op unless
	// cfg.WASMBlobRetention is set — see wasm_blob_retire.go.
	n.startWASMBlobRetirement()

	return n, nil
}

// awaitBringupWave blocks until every shard index in wave reports an elected
// leader (LeaderAddr != ""), or budget elapses — whichever comes first. It is a
// bringup-TIMING throttle only: it never touches election/quorum/log semantics,
// and on timeout it logs and PROCEEDS so a slow or not-yet-quorate group can never
// stall or fail startup (a group without quorum here almost always just has a peer
// that is not up yet, and elects once quorum forms). Called between successful
// shard creations in newMultiNode to stagger cold elections.
//
// Meta-leader gate: staggering only helps once a QUORUM OF NODES is up — which is
// exactly when the meta group has elected a leader. Until then no shard group can
// win a cold election either, so there is nothing to stagger and waiting the full
// budget would be pure dead time. This is the common case for the FIRST node to
// start (its peers are not up yet): it returns immediately and brings its groups up
// unthrottled, while the nodes that join AFTER quorum exists (meta has a leader) DO
// throttle, staggering the real cold-election storm as they enable it. The
// newMultiNode waitForAnyLeader(meta) call before the shard loop means that whenever
// node quorum exists, the meta leader is already visible here.
func (n *Node) awaitBringupWave(wave []int, budget time.Duration) {
	deadline := time.Now().Add(budget)
	for {
		if addr, _ := n.meta.Raft.LeaderWithID(); addr == "" {
			return // no node quorum yet — nothing can cold-elect; proceed unthrottled
		}
		ready := true
		for _, idx := range wave {
			if s := n.shards[idx]; s == nil || s.LeaderAddr() == "" {
				ready = false
				break
			}
		}
		if ready {
			return
		}
		if !time.Now().Before(deadline) {
			var pending []int
			for _, idx := range wave {
				if s := n.shards[idx]; s == nil || s.LeaderAddr() == "" {
					pending = append(pending, idx)
				}
			}
			slog.Warn("proceeding: shard(s) had no leader within bringup wave budget", "component", "cluster", "pending", pending, "budget", budget)
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
}

// newMultiNode constructs the multi-node Node: shared mux StreamLayer,
// meta-Raft group, and N data-shard Raft groups all sharing one TCP
// listener via per-group sub-layers.
func newMultiNode(cfg Config) (*Node, error) {
	groupIDs := make([]uint32, 0, cfg.NumShards+1)
	for i := 0; i < cfg.NumShards; i++ {
		groupIDs = append(groupIDs, uint32(i))
	}
	groupIDs = append(groupIDs, metaGroupID)

	// Resolve RaftAddr: auto-bind to 127.0.0.1:0 only when single-peer
	// self-config is in play and the caller left RaftAddr empty.
	raftAddr := cfg.RaftAddr
	if raftAddr == "" && len(cfg.Peers) == 1 && cfg.Peers[0].NodeID == cfg.NodeID {
		raftAddr = "127.0.0.1:0"
	}

	// Select the inter-node transport: the default per-group NetworkTransport
	// over the raft/mux shared listener, or the multiplexed batching transport
	// (raft/fabric). Exactly one of sl/fab is non-nil; localRaftAddr is the
	// resolved listener address and metaTransport is the meta group's facade.
	// closeTransport tears down whichever listener we bound on an error path.
	var (
		sl            *mux.StreamLayer
		fab           *fabric.Fabric
		localRaftAddr string
		metaTransport hraft.Transport
		//nolint:ineffassign // Defensive default, deliberately not removed: both
		// branches below do assign closeTransport, but each can also `return` on a
		// listen error BEFORE doing so. A nil func here would turn any future
		// early-return that calls it into a panic during cluster bring-up.
		closeTransport = func() {}
	)
	if cfg.RaftTransport == "fabric" {
		// Inter-node mTLS (opt-in): InterNodeServerTLS wraps this Raft listener
		// (mTLS + post-handshake CN allowlist), InterNodeTLS upgrades the outbound
		// peer dials. All nil/empty ⇒ plaintext, byte-identical to the historical path.
		f, err := fabric.New(raftAddr, groupIDs, cfg.InterNodeServerTLS, cfg.InterNodeTLS, cfg.NodeCNAllowlist)
		if err != nil {
			return nil, fmt.Errorf("cluster: fabric listen: %w", err)
		}
		fab = f
		localRaftAddr = f.Addr().String()
		metaTransport = f.For(metaGroupID)
		closeTransport = func() { _ = f.Close() }
	} else {
		// Same opt-in inter-node mTLS wiring as the fabric branch (see above).
		s, err := mux.New(raftAddr, groupIDs, cfg.InterNodeServerTLS, cfg.InterNodeTLS, cfg.NodeCNAllowlist)
		if err != nil {
			return nil, fmt.Errorf("cluster: mux listen: %w", err)
		}
		sl = s
		localRaftAddr = s.Addr().String()
		// Pull the meta group's per-group layer and, in tests only, let cfg wrap it
		// (metaStreamLayerWrap is nil in production ⇒ identity ⇒ this line is
		// byte-identical to constructing the transport directly over s.For(metaGroupID)).
		// The wrap isolates ONLY the meta group's transport — see Config.metaStreamLayerWrap.
		ml := s.For(metaGroupID)
		if cfg.metaStreamLayerWrap != nil {
			ml = cfg.metaStreamLayerWrap(ml)
		}
		metaTransport = hraft.NewNetworkTransport(ml, 3, 10*time.Second, os.Stderr)
		closeTransport = func() { _ = s.Close() }
	}

	// When we auto-bound a single-peer config, rewrite the peer's
	// RaftAddr to the resolved listener address so Raft can dial itself.
	peers := cfg.Peers
	if cfg.RaftAddr == "" && len(peers) == 1 {
		peers = []Peer{{
			NodeID:     cfg.NodeID,
			RaftAddr:   localRaftAddr,
			ServerAddr: peers[0].ServerAddr,
		}}
		// Rewrite the local cfg copy so the stored Node sees resolved
		// RaftAddrs (after auto-bind). The caller's Config is unaffected
		// because cfg here is a value-receiver copy.
		cfg.Peers = peers
	}

	meta, err := startMetaRaft(cfg, metaTransport)
	if err != nil {
		closeTransport() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, err
	}

	// In multi-node bootstrap a slow node may not see a leader within
	// the deadline; the eventual leader publishes state, and our local
	// ApplySetMembersIfLeader is a no-op when we are not the leader.
	// Swallowing the timeout is intentional and non-fatal here.
	_ = waitForAnyLeader(meta.Raft, 5*time.Second)
	// On Bootstrap=true, publish initial cluster state (best-effort: whoever is
	// the meta leader during this window commits it). On restart
	// (Bootstrap=false), state is already in the meta-Raft log; skip to avoid a
	// redundant entry per restart. If this commit is lost to bootstrap leadership
	// churn, per-shard OpSetPlacement self-heals the Placement table later (see
	// MetaFSM.Apply), so the coordinator does not depend on it landing here.
	if cfg.Bootstrap {
		// Seed the structural ISR floor only in PB mode — in raft mode
		// there is no OpSetShardISR to floor, so leave it 0 (disabled).
		minISRSeed := 0
		if cfg.ReplicationMode == ReplicationModePB {
			minISRSeed = cfg.MinISR
		}
		if err := meta.ApplySetMembersIfLeader(peers, cfg.NumShards, cfg.ReplicationFactor, minISRSeed, 5*time.Second); err != nil {
			_ = meta.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
			closeTransport() //nolint:errcheck,gosec // best-effort cleanup on error path
			return nil, fmt.Errorf("cluster: meta ApplySetMembers: %w", err)
		}
	}

	n := &Node{
		cfg:    cfg,
		shards: make([]*shard.Store, cfg.NumShards),
		mux:    sl,
		fabric: fab,
		meta:   meta,
		// See newSingleNode: built before any shard.New, because a snapshot can be
		// installed from inside that constructor.
		wasmState:     newWASMState(),
		wasmFetchStop: make(chan struct{}),
	}
	// Node-local rebalancing admin handlers (slice 4). Registered before the
	// server can dispatch to this node.
	n.registerAdminOps()

	// Placement assigns each data shard to a replica set; a node creates only
	// the shards it is placed on (true storage partitioning when
	// ReplicationFactor < len(peers); full replication otherwise). Computed
	// locally from the static peer list — identical to the meta-Raft FSM's
	// placement since both are deterministic over the same membership.
	n.placement = computePlacement(peers, cfg.NumShards, cfg.ReplicationFactor)

	// Primary-backup wiring (ReplicationMode=="pb"). Built AFTER n.placement (the
	// seed derives from it) and BEFORE the owned-shard loop (so buildShardConfig's
	// PB branch and shard.New's primary lookup can see the control-plane seed).
	// Raft mode skips this entirely — n's pb* fields stay nil and construction is
	// byte-identical.
	if cfg.ReplicationMode == ReplicationModePB {
		thisPBAddr := ""
		for _, p := range peers {
			if p.NodeID == cfg.NodeID {
				thisPBAddr = p.PBAddr
				break
			}
		}
		if thisPBAddr == "" {
			_ = meta.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
			closeTransport()
			return nil, fmt.Errorf("cluster: ReplicationMode=pb: no PBAddr for node %q in peers", cfg.NodeID)
		}
		// Same opt-in inter-node mTLS wiring as the Raft transports: wrap the PB
		// replication listener (mTLS + CN allowlist) and upgrade its outbound dials.
		pbTr, err := pbisr.NewNetTransport(thisPBAddr, cfg.InterNodeServerTLS, cfg.InterNodeTLS, cfg.NodeCNAllowlist)
		if err != nil {
			_ = meta.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
			closeTransport()
			return nil, fmt.Errorf("cluster: pb transport listen: %w", err)
		}
		n.pbTransport = pbTr
		// Extend the transport-teardown closure so EVERY subsequent construction
		// error path (shard loop, topology/wasm registration) also closes the PB
		// listener — not just the mux/fabric one.
		prevCloseTransport := closeTransport
		closeTransport = func() {
			prevCloseTransport()
			_ = pbTr.Close()
		}
		n.pbControl = newMetaControl(meta.FSM, cfg.MinISR)
		n.pbAddrOf = pbAddrMap(peers)
		n.pbEngines = make(map[int]*pbisr.Engine)
		// One shared monotonic clock for every engine (via shard.Config.PBClock →
		// pbisr.WithClock) AND the lease-keeper, anchored here at construction so
		// both judge lease expiry on the same time source.
		pbStart := time.Now()
		n.pbNow = func() int64 { return int64(time.Since(pbStart)) }

		// Seed the PB control plane (epoch 1: primary + ISR per shard, in ONE atomic
		// entry each — see ApplySetShardSeed). Unlike the
		// raft Members table (which OpSetPlacement self-heals), the PB seed has no
		// later self-heal, so a one-shot best-effort commit is not enough: during
		// bringup the node that ends up meta leader may have already passed its
		// window. Instead every bootstrapping node runs a bounded background seeder
		// that RETRIES until the control plane is seeded — bootstrapPBShardControl
		// is a no-op on a follower (fast ErrNotLeader) and idempotent on the leader
		// (OpSetShardEpoch is monotonic), so whichever node becomes leader commits
		// it once and all seeders then observe it and exit. It runs in the
		// background (not before the shard loop) because a synchronous retry would
		// stall this node's construction while its peers are not up yet; the
		// lease-keeper grants the self-lease once the seed lands (the expected
		// follower-timing path).
		if cfg.Bootstrap {
			n.pbSeedStop = make(chan struct{})
			seedStop := n.pbSeedStop
			seeds := pbShardControlSeeds(n.placement)
			n.pbSeedWg.Add(1)
			go func() {
				defer n.pbSeedWg.Done()
				n.seedPBControlPlane(meta, seeds, seedStop)
			}()
			// Ensure a later construction error path also signals the seeder to exit.
			prevCT := closeTransport
			closeTransport = func() {
				select {
				case <-seedStop:
				default:
					close(seedStop)
				}
				prevCT()
			}
		}
	}

	// Bringup election-concurrency throttle. shard.New starts each group's cold
	// election ASYNC, so creating every owned group back-to-back makes them all
	// cold-elect within ms of each other — a thundering herd that starves election
	// on a CPU-constrained host. Instead create owned shards in waves of at most k
	// and wait (bounded, proceed-on-timeout) for each wave to report an elected
	// leader before starting the next, staggering the cold elections. All nodes run
	// this loop in the same deterministic shard order (computePlacement), so the
	// per-node throttle self-synchronizes and no group ever waits on a later group
	// (no circular wait). When k >= owned-shard count the loop finishes before a
	// wave ever fills, so only the trailing barrier runs and — on a healthy box
	// whose groups already have leaders — it returns immediately: a no-op versus the
	// pre-throttle path.
	k := cfg.BringupElectionConcurrency
	if k <= 0 {
		k = runtime.GOMAXPROCS(0)
	}
	if k < 1 {
		k = 1
	}
	// Per-wave barrier budget: finite and scaled to the shard Raft election window
	// so it tracks the configured election timing. A not-yet-quorate group (e.g. a
	// peer not up yet during a sequential bringup, or a genuinely down peer) only
	// adds this bounded wait, never a hang — the barrier PROCEEDS on timeout.
	waveBudget := 10 * time.Second
	if ms := cfg.ShardCfg.RaftElectionMs; ms > 0 {
		waveBudget = time.Duration(ms) * time.Millisecond * 2
		if waveBudget > 10*time.Second {
			waveBudget = 10 * time.Second
		}
		if waveBudget < 2*time.Second {
			waveBudget = 2 * time.Second
		}
	}

	var wave []int // owned shard indices created since the last barrier
	for i := 0; i < cfg.NumShards; i++ {
		owners := n.placement[i]
		if !placementContains(owners, cfg.NodeID) {
			continue // not an owner of this shard — n.shards[i] stays nil
		}
		// The shard's Raft group spans only its owners (buildShardConfig reads
		// n.mux/n.cfg, both already set above).
		store, err := shard.New(n.buildShardConfig(i, owners, cfg.Bootstrap))
		if err != nil {
			for j := 0; j < i; j++ {
				if n.shards[j] != nil {
					_ = n.shards[j].Close() //nolint:errcheck,gosec // best-effort rollback
				}
			}
			_ = meta.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
			closeTransport() //nolint:errcheck,gosec // best-effort cleanup on error path
			return nil, fmt.Errorf("cluster: shard %d: %w", i, err)
		}
		// Wire the ConsistencyBoundedStaleness follower-side leader-frontier hook
		// (coalesced per shard). idx is captured per iteration; the closure forwards a
		// __shard_readindex__ to shard idx's leader when this node is a follower.
		idx := i
		store.SetLeaderFrontierFn(func(dl time.Time) (uint64, error) {
			return n.shardFrontier.do(idx, dl, func(d time.Time) (uint64, error) {
				return n.fetchShardLeaderFrontier(idx, d)
			})
		})
		// A scan-mode kv_query walks this store's whole keyspace and aliases its
		// live mmap, so it goes behind the same per-group gate the index backfill
		// registers with — installed here, at the one moment the store exists and
		// nothing can be walking it yet. See installKVQueryScanGate.
		n.installKVQueryScanGate(i, store)
		n.shards[i] = store

		// Throttle: once the wave is full, wait for it to elect before the next.
		// The barrier only spans successful creations — a shard.New error above
		// takes the rollback path before we ever reach here.
		wave = append(wave, i)
		if len(wave) >= k {
			n.awaitBringupWave(wave, waveBudget)
			wave = wave[:0]
		}
	}
	// Drain the trailing partial wave. On a healthy box (k >= owned count) this is
	// the only barrier, and it returns at once because those groups already elected.
	if len(wave) > 0 {
		n.awaitBringupWave(wave, waveBudget)
	}

	// Form the shard Raft groups the construction-time bootstrap could not: a node
	// hosts only the shards it OWNS, so with ReplicationFactor < len(peers) some
	// shards have an owner set that EXCLUDES the -bootstrap node and NOTHING ever
	// called BootstrapCluster for them — they stayed leaderless forever and every
	// write hashing there hung. The designation is made in the meta log and read
	// back from replicated state, so exactly one owner forms each group. See
	// cluster/shard_formation.go.
	//
	// Raft mode only: PB shards have no per-shard Raft group, and their control
	// plane is seeded by seedPBControlPlane above. Both goroutines are self-limiting
	// — they return as soon as every owned shard has a leader, which on an
	// RF == len(peers) cluster is the first pass, leaving that path as it was.
	// Gated on PARTIAL replication, which is the only shape that can produce an
	// unformed group: computePlacement gives full replication when
	// ReplicationFactor is 0 or >= len(peers), and then EVERY shard's owner set
	// contains the bootstrap node, so construction-time bootstrap already formed
	// them all. Skipping the goroutines there keeps this change a strict no-op for
	// every full-replication cluster rather than adding two background readers that
	// would have nothing to do.
	partialReplication := cfg.ReplicationFactor > 0 && cfg.ReplicationFactor < len(peers)
	if cfg.ReplicationMode != ReplicationModePB && partialReplication {
		n.formationStop = make(chan struct{})
		// placementCopy (a deep copy under n.shardMu), not n.placement: a migration
		// rewrites placement slots at runtime, so handing the live slice to a
		// goroutine would race. Formation only cares about the INITIAL placement
		// anyway — a shard added to a node later is created by AddShardOwner, which
		// joins an already-formed group rather than forming one.
		placement := n.placementCopy()
		n.formationWg.Add(2)
		go func() {
			defer n.formationWg.Done()
			n.seedShardFormers(meta, placement, n.formationStop)
		}()
		go func() {
			defer n.formationWg.Done()
			n.driveShardFormation(n.formationStop)
		}()
	}

	// Primary-backup mode: start the lease-keeper now that every owned shard's
	// engine has been collected (via the PBRegister hook into n.pbEngines). It
	// grants/renews the self-lease for each shard this node is primary of, which
	// is what lets Propose pass the OH1 lease fence — including the follower-timing
	// case where a shard was built before the seed commit replicated (no
	// construction-time retry; the keeper heals it once the FSM shows the primary).
	// It shares n.pbNow with the engines so the expiry math lines up on one clock.
	if n.pbTransport != nil {
		// Note: the barrier-gated leaseKeeper (quorum-confirmed renewal,
		// closing the OH1 double-primary window) is implemented and unit-tested,
		// but wiring n.MetaReadBarrier here as a SYNCHRONOUS fail-closed gate
		// regresses PB write liveness — it couples every lease renewal to meta
		// leader election + barrier-forwarding availability, which starves lease
		// acquisition during meta election churn (measured: election alone can
		// take ~5s under load, and the barrier forward is unavailable until then;
		// TestStaticPBClusterReplicatesAndReads times out). The live gate needs a
		// DECOUPLED-CONFIRMATION design (a background barrier that stamps a
		// "last quorum-confirmed" time; the tick renews only within a bounded
		// window past it, so transient meta blips don't starve leases but a
		// SUSTAINED partition still lapses them). Until that lands, keep the
		// pre-4a local-read renewal (barrier nil) so the static cluster works.
		// See shard/pbisr/DESIGN.md §4a robustness note.
		// Gate lease renewal on a quorum-connection check
		// (n.confirmMetaView). A partitioned node fails it and self-fences,
		// closing the OH1 double-primary window — without the read-index-forward
		// fragility that starves follower primaries (see confirmMetaView).
		// Resolve the four PB failover timings ONCE (config overrides, else the
		// package-const defaults). The honor rule (failoverTimeout > leaseTTL +
		// metaStaleness) was already asserted loud in cfg.Validate(); these are the
		// exact same values. metaStaleness is stored on the node so confirmMetaView
		// (the lease gate) uses the configured bound, letting the gate test shrink
		// every real (non-fake) clock together.
		pbLeaseTTLEff, pbMetaStaleEff, pbFailoverTimeoutEff, pbRenewIntervalEff := cfg.pbEffectiveTimings()
		n.pbMetaContactStaleness = pbMetaStaleEff

		// The second barrier (n.metaReadBarrier) runs ONCE PER (shard, epoch), not
		// per tick: it makes the local MetaFSM provably current before this node is
		// first licensed to ack at a new epoch, so the engine cannot build a write's
		// peer set from an ISR narrower than the committed one. Per-tick it would
		// reintroduce the follower-forward starvation confirmMetaView exists to
		// avoid; per epoch it is a handful of round-trips over a cluster's lifetime.
		n.leaseKeeper = newLeaseKeeper(meta.FSM, cfg.NodeID, n.pbEngines, pbLeaseTTLEff, pbLeaseRefresh, n.pbNow, n.confirmMetaView, n.metaReadBarrier, pbLeaseBarrierTimeout)
		n.leaseKeeper.start()
		// Fold keeper teardown into the transport-teardown closure so the remaining
		// construction error paths (topology/wasm registration) also stop it.
		lk := n.leaseKeeper
		prevCloseTransport := closeTransport
		closeTransport = func() {
			lk.stop()
			prevCloseTransport()
		}

		// Automatic failover (opt-in via PBAutoFailover; DEFAULT OFF). When
		// OFF, NEITHER the beacon NOR the ticker starts, so the meta-Raft log carries
		// zero OpShardLeaseRenew entries and zero promotion OpSetShardEpoch bumps —
		// the replicated state and its snapshots stay BYTE-IDENTICAL to the static
		// pre-Plan-4 cluster. When ON:
		//   - pbTracker: leader-local liveness memory on the shared n.pbNow clock.
		//   - the FSM observer stamps pbTracker for every valid beacon it applies.
		//   - pbBeacon: this node commits its current primary set every renewInterval.
		// (The always-on failover ticker is wired alongside this in a later stage.)
		if cfg.PBAutoFailover {
			n.pbTracker = newPBFailoverTracker(n.pbNow)
			// The observer is a LEAF callback fired under the FSM write lock — it takes
			// only the tracker's own mutex (see leaseRenewObserver's contract).
			meta.FSM.SetLeaseRenewObserver(n.pbTracker.observeRenew)

			n.pbBeacon = newPBBeacon(n, pbRenewIntervalEff, pbBeaconSubmitTimeout)
			n.pbBeacon.start()
			bc := n.pbBeacon
			prevCT := closeTransport
			closeTransport = func() {
				bc.stop()
				prevCT()
			}

			// The always-on failover ticker: it self-detects meta leadership each
			// tick (there is no leadership watcher) and, while leader, promotes an ISR
			// survivor for any primary silent past failoverTimeout. It shares n.pbNow
			// so its elapsed-gap math lines up with the observer's stamps. The
			// high-water resolver (the durable grow-abandon backstop) verifies each
			// candidate's applied high-water over the PB transport before promotion.
			n.pbFailover = newPBFailover(meta, meta.FSM, n.pbTracker, n.pbNow, int64(pbFailoverTimeoutEff), pbFailoverTickInterval, pbFailoverApplyTimeout, n.pbCandidateHighWater)
			n.pbFailover.start()
			fo := n.pbFailover
			prevCT2 := closeTransport
			closeTransport = func() {
				fo.stop()
				prevCT2()
			}

			// ISR SHRINK driver: on each owned primary it reads the engine's
			// per-peer replication-failure counter and, once a backup is dead past the
			// threshold, forwards an OpSetShardISR removing it (never below MinISR — the
			// FLOOR is enforced in decidePBShrink, NOT the FSM), then applies the
			// committed shrink to the engine (Engine.ShrinkISR) so the stalled pipeline
			// resumes. Only runs under PBAutoFailover, so a static cluster logs no
			// OpSetShardISR shrink and stays byte-identical.
			shrinkThreshold := pbShrinkFailureThreshold
			if cfg.PBShrinkThreshold > 0 {
				shrinkThreshold = cfg.PBShrinkThreshold
			}
			// One ISR-reconcile baseline SHARED by the shrink and grow drivers, so a
			// widen and a narrow at the same epoch never misread each other (see
			// pbISRReconcile).
			isrBaseline := newPBISRReconcile()
			n.pbShrink = newPBShrinkDriver(n, pbShrinkTickInterval, pbShrinkSubmitTimeout, shrinkThreshold, cfg.MinISR, isrBaseline)
			n.pbShrink.start()
			sd := n.pbShrink
			prevCT3 := closeTransport
			closeTransport = func() {
				sd.stop()
				prevCT3()
			}

			// ISR GROW driver: on each owned primary it catches a lagging
			// placement owner up to the frontier (the learner flip) and re-adds it to
			// the ISR, re-opening a minISR>=2 shard to writes after a failover reset and
			// un-doing a prior shrink once a member recovers. Shares the reconcile
			// baseline with the shrink driver. Only runs under PBAutoFailover.
			growTick := pbGrowTickInterval
			if cfg.PBGrowTickMs > 0 {
				growTick = time.Duration(cfg.PBGrowTickMs) * time.Millisecond
			}
			n.pbGrow = newPBGrowDriver(n, growTick, pbGrowSubmitTimeout, pbGrowCatchupTimeout, isrBaseline)
			n.pbGrow.start()
			gd := n.pbGrow
			prevCT4 := closeTransport
			closeTransport = func() {
				gd.stop()
				prevCT4()
			}
		}
	}

	// Register the __topology__ shardless op so smart clients can discover
	// the cluster layout. Placed after the shard loop so n.shards is fully
	// populated before any inbound call can reach Topology().
	if err := ops.RegisterTopology(cfg.Ops, n.Topology); err != nil {
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec // best-effort rollback
			}
		}
		_ = meta.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
		closeTransport() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, fmt.Errorf("cluster: register __topology__: %w", err)
	}

	// WASM runtime + Raft hook.
	wasmRT, err := wasm.NewRuntime()
	if err != nil {
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec
			}
		}
		_ = meta.Close() //nolint:errcheck,gosec
		closeTransport() //nolint:errcheck,gosec
		return nil, fmt.Errorf("cluster: new wasm runtime: %w", err)
	}
	n.setWASMRuntime(wasmRT)

	// Register the __register_wasm__ FSM hook. Idempotent across restarts.
	if err := ops.RegisterWASMRegisterOp(cfg.Ops, func(shardIdx int, r ops.WASMRegistration) error {
		// One registration is proposed to every shard group, so this hook runs
		// once per hosted group — concurrently, from each group's own FSM apply
		// goroutine. Serialize: applyWASMRegistration is idempotent but not
		// concurrency-safe against itself (shared files, runtime, registry).
		//
		// shardIdx is the group whose log carried THIS apply. Recording it is what
		// earns this node the right to propose invocations of the op into that
		// group's log, and nothing else earns it (see checkWASMRouteGate).
		//
		// THE PUBLISH IS UNCONDITIONAL, deliberately. applyWASMRegistration has a
		// path that mutates wasmState and THEN returns an error: a per-group
		// contract refusal (checkWASMGroupContract) is reported after the node-wide
		// install has already been recorded. Skipping the publish on a non-nil error
		// would leave the lock-free table the route gate and
		// wasm.Runtime.resolveModuleForInvoke read stale with respect to the state
		// that was actually committed. It happens to be benign today only via an
		// unstated chain: the published snapshot is built from `replicated` and
		// `groups` alone (wasmBindingSnapshot), a refusal restores the previous
		// `groups`, and `replicated` can only flip false→true for a name that had
		// no prior binding at all — which is exactly the case in which no refusal
		// can fire. Nothing checks that chain and nothing would re-derive it on the
		// next edit. Publishing is idempotent and purely derived from wasmState, so
		// deferring it removes the invariant instead of relying on it.
		n.wasmApplyMu.Lock()
		defer n.wasmApplyMu.Unlock()
		defer n.publishWASMGateLocked()
		return applyWASMRegistration(cfg.DataDir, n.wasmRT, cfg.Ops, n.wasmState, r, shardIdx, n.prefetchWASMBlob)
	}); err != nil && !errors.Is(err, ops.ErrDuplicateOp) {
		_ = n.wasmRT.Close()
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec
			}
		}
		_ = meta.Close() //nolint:errcheck,gosec
		closeTransport() //nolint:errcheck,gosec
		return nil, fmt.Errorf("cluster: register __register_wasm__: %w", err)
	}

	if len(cfg.WASMModules) > 0 {
		// Iterate what was actually LOADED — see the same loop in newSingleNode.
		loaded, err := loadWASMModules(cfg.DataDir, cfg.WASMModules, n.wasmRT, n.wasmState)
		if err != nil {
			_ = n.wasmRT.Close()
			for j := 0; j < cfg.NumShards; j++ {
				if n.shards[j] != nil {
					_ = n.shards[j].Close() //nolint:errcheck,gosec
				}
			}
			_ = meta.Close() //nolint:errcheck,gosec
			closeTransport() //nolint:errcheck,gosec
			return nil, fmt.Errorf("cluster: load WASM modules: %w", err)
		}
		for _, lm := range loaded {
			c := lm.Registration
			ke := ops.WASMKeyExtractor()
			if err := wasm.RegisterModule(cfg.Ops, n.wasmRT, c.Name, lm.ModuleID, c.Kind, ke); err != nil && !errors.Is(err, ops.ErrDuplicateOp) {
				_ = n.wasmRT.Close()
				for j := 0; j < cfg.NumShards; j++ {
					if n.shards[j] != nil {
						_ = n.shards[j].Close() //nolint:errcheck,gosec
					}
				}
				_ = meta.Close() //nolint:errcheck,gosec
				closeTransport() //nolint:errcheck,gosec
				return nil, fmt.Errorf("cluster: register module %q: %w", c.Name, err)
			}
		}
	}

	// Reload from disk (modules registered via Raft Apply in a previous
	// process) and install any snapshot-carried registrations that arrived
	// before the runtime existed. Config-loaded modules win on conflict.
	if err := n.finishWASMSetup(cfg.DataDir); err != nil {
		_ = n.wasmRT.Close()
		for j := 0; j < cfg.NumShards; j++ {
			if n.shards[j] != nil {
				_ = n.shards[j].Close() //nolint:errcheck,gosec
			}
		}
		_ = meta.Close() //nolint:errcheck,gosec
		closeTransport() //nolint:errcheck,gosec
		return nil, fmt.Errorf("cluster: reload WASM modules: %w", err)
	}
	// Started LAST, for the reason newSingleNode's copy states: the sweeper's rule
	// is "referenced by nothing", so it must not run while the reference set is
	// still being built. A no-op unless cfg.WASMBlobRetention is set.
	n.startWASMBlobRetirement()

	// Started LAST, after every construction failure path has been passed: the
	// observer installs into and walks the shard stores, so it must not be running
	// over stores a late error is about to close. See startKVIndexObserver.
	n.startKVIndexObserver()

	return n, nil
}

// Call dispatches an op to the shard that owns the routing key.
// Shardless ops (those registered via ops.Register, not RegisterRoutable)
// dispatch to shard 0 — with ONE exception: __register_wasm__ is broadcast to
// every shard group, because the op it registers may itself be routable and so
// be invoked from any group's log (see broadcastWASMRegistration).
//
// Invocations of a dynamically registered WASM op pass the ROUTE GATE before
// anything is proposed: this node will not put an invocation into a group it
// hosts until it knows that group's log already carries the op's registration.
// A group that fails the check answers with the client-visible, retryable
// ErrWASMOpNotInThisGroup instead. See checkWASMRouteGate.
//
// A registration that would UPDATE a live module in place is refused here with
// ErrWASMUpdateUnsupported; re-registering the identical module is allowed, since
// that is the idempotent retry. See checkWASMUpdateGate.
//
// A registration also PUSHES its module's bytes to every member this node can
// reach before the marker enters any log, and requires a compile verdict from
// each member that answers. A member that refuses fails the registration; a
// member that cannot be reached is named in the reply payload and fetches the
// bytes on demand later. See pushWASMBlob.
//
// In multi-node mode, any *shard.NotLeaderError returned by the
// underlying shard has its LeaderAddr rewritten from the Raft transport
// address to the corresponding client-facing server address (looked up
// via the meta-Raft Members table). Without this translation, the
// hint the client follows points to the mux/Raft listener — which
// closes any connection that doesn't start with a valid Raft group
// ID, surfacing as "connection reset by peer" on the client.
func (n *Node) Call(name string, args []byte) ([]byte, error) {
	// Rebalancing admin ops execute on the node that received them (the
	// coordinator targets a specific node by its server addr); they must not be
	// key-routed or forwarded to shard 0's owner. Dispatch them directly.
	if h, ok := n.adminOps[name]; ok {
		return h(args)
	}
	kind, ke, layout, ok := n.cfg.Ops.LookupRouting(name)
	if !ok {
		return nil, ErrUnknownOp
	}
	// A WASM registration is the one op that must land in EVERY shard group's
	// log, not just shard 0's. See broadcastWASMRegistration for why.
	if name == wasmRegisterOpName {
		// Run every propose-time refusal HERE, before anything is proposed: an
		// oversized or undecodable FRAME, an unsafe name, an out-of-range Kind, an
		// oversized module and an attempted in-place update must each fail while
		// nothing has entered any group's log. The same set runs in
		// handleRegisterWASMShard, which is the OTHER way into a group's log.
		// THE CLIENT-EDGE PAYLOAD AND THE REPLICATED MARKER ARE NOT THE SAME BYTES,
		// and this is the one place that distinction is made. args carries the
		// module (ops.EncodeWASMRegistrationRequest); what is broadcast into every
		// group's log is the THIN MARKER alone, which names the module by content
		// address. Broadcasting args instead would silently undo the whole stage —
		// the module would be back in NumShards Raft logs — so the marker is
		// re-encoded here rather than forwarded.
		r, module, err := n.checkWASMRegistrationRequest(args)
		if err != nil {
			return nil, err
		}
		// THE PUSH PHASE, before the marker reaches any log: store the module's
		// bytes locally and deliver them to every member this node can reach,
		// requiring a compile verdict from each one that answers, and requiring a
		// MAJORITY of members to hold them before anything is proposed. A member
		// that refuses fails the registration HERE, where the client sees it; a
		// cluster too degraded to reach the majority fails it too, because a marker
		// no majority backs could become permanently unfetchable. A member that is
		// merely unreachable below that floor is named in the reply and fetches on
		// demand. See pushWASMBlob.
		report, err := n.pushWASMBlob(r, module)
		if err != nil {
			return nil, err
		}
		if _, err := n.broadcastWASMRegistration(ops.EncodeWASMRegistration(r)); err != nil {
			return nil, err
		}
		// The skip report is the registration's reply payload (empty when every
		// member acked), so a partial push is visible to the caller rather than
		// only to Stats().WASMBlobPush.
		if report == "" {
			return nil, nil
		}
		return []byte(report), nil
	}
	// A flush is the other op that must land in EVERY shard group's log, not just
	// shard 0's: it is keyless (so shardIndexFor returns 0 for it) but wipes the
	// WHOLE keyspace, and each group owns an independent cache. See broadcastFlush.
	if name == flushOpName {
		return n.broadcastFlush()
	}
	// kv_query is the READ with the same shape: keyless (so shardIndexFor returns
	// 0 for it) but answering over the WHOLE keyspace, which lives in one
	// independent cache per shard group. Routed like an ordinary shardless op it
	// would answer from group 0's slice alone and present that as the complete
	// result. Unlike the two broadcasts above it lands in no log at all — it fans a
	// read-only leaf to every group that can still contribute rows and merges the
	// pages. See broadcastKVQuery.
	if name == kvQueryOpName {
		return n.broadcastKVQuery(args)
	}
	idx, err := n.shardIndexFor(ke, layout, args)
	if err != nil {
		return nil, err
	}
	// put_batch routes by its first key but applies ALL keys to that one shard.
	// It is reachable directly over the wire, so guard the same-shard invariant
	// here (where NumShards is known — the handler cannot see it): a batch whose
	// keys span shards would silently store the off-shard keys where they can
	// never be read back. Legitimate cross-shard bulk writes use Node.PutBatch,
	// which groups by shard before calling.
	if name == "put_batch" {
		if err := n.putBatchSameShard(args, idx); err != nil {
			return nil, err
		}
	}
	// A dynamically registered WASM op may only be proposed into a group whose log
	// already carries its registration. Checked BEFORE the propose, because the
	// whole point is that the entry must not enter that log. See
	// checkWASMRouteGate.
	if err := n.checkWASMRouteGate(name, kind, idx); err != nil {
		return nil, err
	}
	// If this node does not host the target shard (partitioned cluster), forward
	// to an owner. The forwarding client follows NotLeader to the owning group's
	// leader; owners host the shard so they never re-forward (no loops).
	s := n.getShard(idx)
	if s == nil {
		return n.forward(idx, name, args)
	}
	return n.callHostedShard(s, name, args)
}

// shardIndexFor picks the destination shard for one op invocation: 0 for a
// shardless op (nil extractors), otherwise shardOf(routing key). It prefers the
// allocation-free extractor — every BUILT-IN routable op has one — and falls back
// to the allocating KeyExtractor, which is all a dynamically registered (WASM) op
// has. Both spellings of a layout yield the identical key, so the chosen shard is
// the same either way.
//
// LIFETIME (the reason the fast path is safe): buf is this call's private stack
// scratch, and the key the extractor returns aliases either args or buf. Nothing
// here retains it — the key is consumed by shardOf (xxhash + modulo) and dropped
// before the function returns, so the scratch cannot outlive the routing decision.
// Returning the key, or storing it anywhere, would break that and is why this
// function returns only an int.
func (n *Node) shardIndexFor(ke ops.KeyExtractor, layout ops.RouteLayout, args []byte) (int, error) {
	if layout != ops.RouteLayoutNone {
		var buf [128]byte
		key := ops.RouteKeyInto(layout, args, buf[:0])
		if key == nil {
			return 0, ErrNoKeyExtractor
		}
		return shardOf(key, n.cfg.NumShards), nil
	}
	if ke != nil {
		key, hasKey := ke(args)
		if !hasKey {
			return 0, ErrNoKeyExtractor
		}
		return shardOf(key, n.cfg.NumShards), nil
	}
	return 0, nil // shardless ops dispatch to shard 0
}

// callHostedShard dispatches name/args to a shard store this node hosts and
// translates any *shard.NotLeaderError's leader identifier into the
// corresponding client-facing server address (see Call's doc for why the raw
// raft addr is unusable as a client hint).
func (n *Node) callHostedShard(s *shard.Store, name string, args []byte) ([]byte, error) {
	result, err := s.Call(name, args)
	if err != nil {
		var nle *shard.NotLeaderError
		if errors.As(err, &nle) && nle.LeaderAddr != "" {
			if srvAddr := n.resolveLeaderHint(nle.LeaderAddr); srvAddr != "" {
				// Return a fresh NotLeaderError carrying the server
				// addr so callers (server.mapResult → client) speak
				// the right protocol.
				return nil, &shard.NotLeaderError{LeaderAddr: srvAddr}
			}
			// No mapping found (e.g. raft addr from a node that has
			// not yet been published to meta-Raft state). Drop the
			// hint rather than misleading the client.
			return nil, &shard.NotLeaderError{LeaderAddr: ""}
		}
	}
	return result, err
}

// resolveLeaderHint turns a replicator's leader identifier into a client-facing
// server address. The identifier's FORM depends on the active replicator, and
// the cluster layer is the only place that can resolve both:
//
//   - Raft: raft.Node.LeaderAddr returns a RAFT TRANSPORT address
//     ("127.0.0.1:7402"), resolved via the meta-Raft Members table.
//   - PB: pbReplicator.LeaderAddr returns ctrl.Primary(shard), which is a NODE
//     ID ("n2") — pbisr.Control is deliberately address-agnostic (Primary and
//     ISR speak node ids), so the shard layer cannot translate it and must not
//     be given cluster addressing to do so.
//
// Before this existed, only the raft-addr mapping was tried, so a PB node id
// never resolved and the hint was DROPPED. The visible effect: a write sent to a
// node that hosts the target shard as a BACKUP came back as a bare
// "shard: not leader" with no address to follow. The native client survived it
// (it routes from its own __topology__ cache and falls back to round-robin), but
// an HTTP or gRPC caller — which has no such routing — simply failed, on 2-3 of
// 8 shards for any given endpoint at RF=2 across 3 nodes.
//
// The raft-addr mapping is tried FIRST so the Raft path stays byte-identical;
// the two key spaces are disjoint in practice (a node id is not a host:port).
func (n *Node) resolveLeaderHint(hint string) string {
	if srvAddr := n.raftToServerAddr(hint); srvAddr != "" {
		return srvAddr
	}
	return n.serverAddrFor(hint)
}

// MaxPutBatchSize caps how many entries ride in a single put_batch log entry.
// One put_batch holds the shard write-lock for its whole apply and forms one
// Raft log/fsync/replication payload, so an unbounded batch would stall every
// other op on the shard and bloat a single entry. PutBatch chunks larger groups.
// Aliases ops.MaxPutBatchSize (the canonical cap) so every batching path agrees.
const MaxPutBatchSize = ops.MaxPutBatchSize

// ErrPutBatchCrossShard is returned when a put_batch's keys do not all hash to
// the same shard. Bulk callers must use Node.PutBatch (which groups by shard).
var ErrPutBatchCrossShard = errors.New("cluster: put_batch keys span multiple shards; use PutBatch")

// putBatchSameShard rejects a put_batch whose keys do not all map to want.
func (n *Node) putBatchSameShard(args []byte, want int) error {
	entries, err := ops.DecodePutBatchArgs(args)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if shardOf(e.Key, n.cfg.NumShards) != want {
			return ErrPutBatchCrossShard
		}
	}
	return nil
}

// PutBatch applies many key/value puts, grouping them by shard so each shard
// receives put_batch ops — ONE Raft log entry (one fsync, one replicate-to-
// majority round-trip, one FSM apply) per chunk, instead of one per key. Entries
// are bucketed by shardOf(key) and each group is chunked to MaxPutBatchSize; each
// chunk is dispatched via Call, so it follows the same leader-routing /
// forwarding as a single put (a NotLeaderError propagates for the caller to
// redirect, exactly like Put). Chunks are sent sequentially; callers wanting
// cross-shard parallelism can shard the entry list themselves and call
// concurrently.
func (n *Node) PutBatch(entries []ops.PutEntry) error {
	if len(entries) == 0 {
		return nil
	}
	groups := make(map[int][]ops.PutEntry)
	for _, e := range entries {
		idx := shardOf(e.Key, n.cfg.NumShards)
		groups[idx] = append(groups[idx], e)
	}
	for _, g := range groups {
		for len(g) > 0 {
			chunk := g
			if len(chunk) > MaxPutBatchSize {
				chunk = g[:MaxPutBatchSize]
			}
			if _, err := n.Call("put_batch", ops.EncodePutBatchArgs(chunk)); err != nil {
				return err
			}
			g = g[len(chunk):]
		}
	}
	return nil
}

// CallPhysical routes op for a physical partition collection. The op's args
// already embed physCol as the collection name, so the normal key extractor
// routes it to the owning shard.
//
// leaderOnly selects the read-consistency path:
//   - false (AnyReplica): delegate to Call, which serves OpReadOnly from any
//     local replica (follower or leader) — the cheap, possibly-stale path.
//   - true (LeaderOnly OR Linearizable): the read must be served by the shard's
//     current Raft leader. If this node leads the shard, serve locally;
//     otherwise forward to the leader's server address (forwardToLeader). If no
//     leader can be resolved, return an error rather than serving a stale
//     follower copy.
//
// LeaderOnly is best-effort, NOT linearizable: the read bypasses Raft's
// readIndex barrier, so a just-demoted leader could briefly serve a read for
// which it is no longer authoritative. This is the documented semantic — it
// narrows the staleness window to "the current leader's applied state" without
// paying for a round-trip quorum confirmation per read.
//
// Linearizable shares this exact routing (it also pins to the leader), but the
// FRESHNESS barrier is enforced by the serving shard, NOT here: shard.Store.Call
// peeks the read_consistency byte from args and, when it is ConsistencyLinearizable,
// runs VerifyLeader + a commit-index catch-up before serving. So this method only
// has to deliver a Linearizable read to a leader. If the local-leader serve below
// discovers (via that barrier) that we have lost leadership since the IsLeader()
// check, s.Call returns a *shard.NotLeaderError; we then re-route to the real
// leader (forwardToLeader), whose shard re-runs the barrier against fresh state.
// A non-NotLeader error (e.g. a barrier timeout) propagates fail-loud.
func (n *Node) CallPhysical(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
	_ = physCol
	if !leaderOnly {
		return n.Call(op, args)
	}
	// Resolve the op's target shard the same way Call does.
	kind, ke, layout, ok := n.cfg.Ops.LookupRouting(op)
	if !ok {
		return nil, ErrUnknownOp
	}
	// LeaderOnly only needs special leader-pinning routing for read-only ops: an
	// OpReadOnly served on a follower would be a stale replica read. OpReadWrite
	// ops already route to the leader resiliently via Call → forward →
	// peer-client NotLeader-following (and that path tolerates the post-startup
	// election window), so leader-pin only reads.
	if kind != ops.OpReadOnly {
		return n.Call(op, args)
	}
	idx, err := n.shardIndexFor(ke, layout, args)
	if err != nil {
		return nil, err
	}
	// If we host the shard AND lead it, serve locally — no hop needed.
	if s := n.getShard(idx); s != nil && s.IsLeader() {
		res, err := s.Call(op, args)
		// Re-route on lost leadership. For a Linearizable read, shard.Store.Call
		// runs the readIndex barrier (VerifyLeader); if VerifyLeader discovers we
		// lost leadership between the IsLeader() check above and the barrier, it
		// returns a *shard.NotLeaderError. Rather than failing the read, forward to
		// the real leader, whose shard re-runs the barrier against fresh state. (For
		// LeaderOnly there is no barrier, so this never triggers — harmless.) Any
		// OTHER error — including a barrier timeout (ErrLinearizableTimeout) —
		// propagates fail-loud; FanOut's OnPartitionUnavailable governs upstream.
		if err != nil {
			var nle *shard.NotLeaderError
			if errors.As(err, &nle) {
				return n.forwardToLeader(idx, op, args)
			}
		}
		return res, err
	}
	return n.forwardToLeader(idx, op, args)
}

// forwardToLeader sends op to the current leader of shardIdx for a LeaderOnly
// read. It resolves the leader's client-facing server address using existing
// primitives — never a new wire protocol:
//   - Hosted shard: this node's Raft replica tracks the current leader, so
//     raftToServerAddr(getShard(idx).LeaderAddr()) yields the leader addr even
//     when this node is a follower.
//   - Non-hosted shard: ask each owner for its Topology via the __topology__
//     op; the owner's Leaders[idx] carries the leader's server addr (every
//     replica of the shard tracks it). The first non-empty answer wins.
//
// If the resolved leader is this node, serve locally. If no leader can be
// resolved, return ErrNoShardOwner-style failure (LeaderOnly cannot be
// satisfied — we must not silently fall back to a stale follower read).
//
// Leader resolution is retried with a bounded deadline so a LeaderOnly read
// issued during the post-startup election window waits the election out rather
// than failing immediately (matching the resilience of the OpReadWrite path,
// which reaches the leader via Call → forward → peer-client NotLeader-following).
// The peerClient already follows NotLeader internally, so most leader changes
// are handled by it; the retry specifically covers the "no leader elected or
// resolved yet" gap. The loop is bounded — on a leaderless shard it returns the
// last error after the deadline rather than spinning forever. No sleep happens
// once a leader is resolved and the call succeeds.
func (n *Node) forwardToLeader(shardIdx int, op string, args []byte) ([]byte, error) {
	return n.forwardToLeaderAs(context.Background(), shardIdx, op, args, op, args)
}

// forwardToLeaderAs is forwardToLeader with the LOCAL serve and the REMOTE hop
// named separately, and with the caller's context bounding the wait.
//
// The two op names are the same for every caller but one. A fan-out leg cannot
// send its own client-facing op name to a peer — the peer's Node.Call would fan
// out again — so it carries a shard-scoped WRAPPER instead (see
// opKVQueryShardName). That wrapper is a node-level admin op and is not in any
// shard's op registry, so the moment leader resolution lands back on THIS node
// the call has to switch to the leaf name and the leaf args. Sending the wrapper
// to a local shard.Store would be an unknown op, not a query — a "leader is us"
// race that would surface as a spurious failure for one page.
//
// ctx bounds the loop as well as each remote call: a leg that has already been
// abandoned by its caller's per-group timeout stops retrying instead of holding
// a peer connection for the rest of the internal deadline.
func (n *Node) forwardToLeaderAs(ctx context.Context, shardIdx int, localOp string, localArgs []byte, remoteOp string, remoteArgs []byte) ([]byte, error) {
	const (
		deadline = 3 * time.Second // comfortably above the raft election timeout
		backoff  = 25 * time.Millisecond
	)
	stopAt := time.Now().Add(deadline)
	if dl, ok := ctx.Deadline(); ok && dl.Before(stopAt) {
		stopAt = dl
	}
	var lastErr = ErrNoShardOwner
	for {
		leaderAddr := n.leaderServerAddr(shardIdx)
		if leaderAddr != "" {
			// If the leader is this node, serve locally (it must host & lead the shard).
			if leaderAddr == n.serverAddrFor(n.cfg.NodeID) {
				if s := n.getShard(shardIdx); s != nil {
					res, err := s.Call(localOp, localArgs)
					if err == nil {
						return res, nil
					}
					lastErr = err
				}
			} else {
				cl, err := n.peerClient(leaderAddr)
				if err != nil {
					lastErr = err
				} else {
					res, err := cl.Call(ctx, remoteOp, remoteArgs)
					if err == nil {
						return res, nil
					}
					lastErr = err
				}
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, lastErr
		}
		// Either the leader was unresolvable, or the call hit a transient error
		// (NotLeader / unreachable / no-owner) during the election window. Retry
		// until the deadline, then surface the last error.
		if time.Now().Add(backoff).After(stopAt) {
			return nil, lastErr
		}
		time.Sleep(backoff)
	}
}

// leaderServerAddr resolves the current leader's client-facing server address
// for shardIdx, or "" if unknown. Hosted shards read the local replica's
// tracked leader; non-hosted shards query owners' __topology__.
func (n *Node) leaderServerAddr(shardIdx int) string {
	if s := n.getShard(shardIdx); s != nil {
		if addr := n.raftToServerAddr(s.LeaderAddr()); addr != "" {
			return addr
		}
		return ""
	}
	// Not hosted: ask owners. Each replica's Topology reports Leaders[shardIdx]
	// (the leader's server addr) because every replica tracks the current leader.
	ctx := context.Background()
	for _, ownerID := range n.ownersFor(shardIdx) {
		if ownerID == n.cfg.NodeID {
			continue // we don't host it despite being listed — skip self
		}
		addr := n.serverAddrFor(ownerID)
		if addr == "" {
			continue
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			continue
		}
		raw, err := cl.Call(ctx, "__topology__", nil)
		if err != nil {
			continue
		}
		topo, err := ops.DecodeTopology(raw)
		if err != nil {
			continue
		}
		if shardIdx >= 0 && shardIdx < len(topo.Leaders) && topo.Leaders[shardIdx] != "" {
			return topo.Leaders[shardIdx]
		}
	}
	return ""
}

// forward sends an op for a shard this node does not host to one of the shard's
// owners. It tries owners in order; the per-owner client follows NotLeader to
// the owning group's leader (always an owner), so there is no re-forwarding.
func (n *Node) forward(shardIdx int, name string, args []byte) ([]byte, error) {
	return n.forwardTimeout(shardIdx, name, args, 0)
}

// forwardTimeout is forward with a per-CALL deadline (0 = none, forward's
// behavior). A bounded caller — one that forwards to many groups in sequence and
// must not be stalled indefinitely by a single unresponsive owner, such as the
// WASM registration broadcast — passes a non-zero timeout. The deadline covers
// the whole owner-rotation loop, which is what makes it a bound on the CALL
// rather than on one attempt.
func (n *Node) forwardTimeout(shardIdx int, name string, args []byte, timeout time.Duration) ([]byte, error) {
	return n.forwardTimeoutOpt(shardIdx, name, args, timeout, false)
}

// forwardTimeoutNoReplay is forwardTimeout for a NON-REPLAYABLE op. It differs in
// exactly one respect: when a per-owner cl.Call returns an AMBIGUOUS error (a
// post-transmission failure — the entry may already be committed, client.IsAmbiguous),
// it returns that error IMMEDIATELY instead of trying the next owner. Trying another
// owner after an ambiguous send is precisely a blind replay: it could apply the op a
// SECOND time. A PRE-transmission error (dial refused, NotLeader, no-leader-yet) is
// still safe to retry against the next owner, so those keep rotating.
//
// This exists for the flush per-group leg (cluster.proposeFlush): re-forwarding a
// committed-but-ambiguous __flush_shard__ to another owner would append a second
// flush and wipe any writes made in between — the double-apply the client-side
// nonReplayableOp guard already closes for the external "flush" op. The plain
// forwardTimeout is deliberately left untouched so WASM registration (which IS
// idempotent under replay) keeps its rotate-on-any-error behavior.
func (n *Node) forwardTimeoutNoReplay(shardIdx int, name string, args []byte, timeout time.Duration) ([]byte, error) {
	return n.forwardTimeoutOpt(shardIdx, name, args, timeout, true)
}

// forwardTimeoutOpt is the shared owner-rotation loop behind forwardTimeout and
// forwardTimeoutNoReplay. When noReplayOnAmbiguous is set, an ambiguous per-owner
// error short-circuits the loop (see forwardTimeoutNoReplay); otherwise every error
// rotates to the next owner (forwardTimeout's historical behavior).
func (n *Node) forwardTimeoutOpt(shardIdx int, name string, args []byte, timeout time.Duration, noReplayOnAmbiguous bool) ([]byte, error) {
	owners := n.ownersFor(shardIdx)
	if len(owners) == 0 {
		return nil, ErrNoShardOwner
	}
	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	var lastErr = ErrNoShardOwner
	for _, ownerID := range owners {
		if ownerID == n.cfg.NodeID {
			continue // we don't host it despite being listed — skip self
		}
		addr := n.serverAddrFor(ownerID)
		if addr == "" {
			continue
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			lastErr = err
			continue
		}
		res, err := cl.Call(ctx, name, args)
		if err == nil {
			return res, nil
		}
		// For a non-replayable op, an AMBIGUOUS (post-transmission) failure must NOT
		// fall through to the next owner: the entry may already be committed, so a
		// retry elsewhere would double-apply. Surface it now. Pre-transmission errors
		// are not ambiguous and stay safe to retry, so they keep rotating below.
		if noReplayOnAmbiguous && client.IsAmbiguous(err) {
			return nil, err
		}
		lastErr = err // try the next owner (transport error / no leader yet)
	}
	return nil, lastErr
}

// serverAddrFor resolves a node id to its client-facing server address from the
// static peer list.
func (n *Node) serverAddrFor(nodeID string) string {
	for _, p := range n.cfg.Peers {
		if p.NodeID == nodeID {
			return p.ServerAddr
		}
	}
	return ""
}

// peerCNVerifier builds the tls.Config.VerifyPeerCertificate callback that
// enforces the OPT-IN per-node identity allowlist on the inter-node CLIENT dial.
// It is attached ONLY when the allowlist is non-empty (see peerClient); with an
// empty allowlist no callback is set and the dial is byte-identical to the
// shared-CA path. The callback runs AFTER Go's standard chain verification (TLS
// invokes VerifyPeerCertificate only once the cert chains to the config's
// RootCAs), so verifiedChains[0][0] is the verified peer leaf. It is FAIL-CLOSED:
// a missing chain/leaf or a leaf whose Subject.CommonName is not in the allowlist
// returns an error, which aborts the handshake before any RPC is forwarded. The
// rejected CN is named in the error for operator diagnosis (a cert CN is not a
// secret); the allowlist contents are not logged.
func peerCNVerifier(allow map[string]bool) func([][]byte, [][]*x509.Certificate) error {
	return func(_ [][]byte, verifiedChains [][]*x509.Certificate) error {
		if len(verifiedChains) == 0 || len(verifiedChains[0]) == 0 {
			return errors.New("cluster: peer presented no verified certificate chain (per-node mTLS allowlist enabled)")
		}
		cn := verifiedChains[0][0].Subject.CommonName
		if !allow[cn] {
			return fmt.Errorf("cluster: peer cert CN %q not in node allowlist (per-node mTLS identity)", cn)
		}
		return nil
	}
}

// peerCNConnVerifier builds the tls.Config.VerifyConnection callback that
// re-enforces the per-node identity allowlist on EVERY inter-node connection,
// including resumed TLS sessions that skip VerifyPeerCertificate. cs is the
// completed connection's state: standard chain verification has already run
// (ServerName/RootCAs are still checked on resumption), so cs.PeerCertificates[0]
// is the verified peer leaf. It mirrors peerCNVerifier and is FAIL-CLOSED: an
// empty peer-cert set or a leaf whose CommonName is not in the allowlist aborts
// the connection before any RPC is forwarded. The rejected CN is named for
// operator diagnosis (a cert CN is not a secret); the allowlist is not logged.
func peerCNConnVerifier(allow map[string]bool) func(tls.ConnectionState) error {
	return func(cs tls.ConnectionState) error {
		if len(cs.PeerCertificates) == 0 {
			return errors.New("cluster: peer presented no certificate (per-node mTLS allowlist enabled)")
		}
		cn := cs.PeerCertificates[0].Subject.CommonName
		if !allow[cn] {
			return fmt.Errorf("cluster: peer cert CN %q not in node allowlist (per-node mTLS identity)", cn)
		}
		return nil
	}
}

// peerClient returns a lazily-created forwarding client targeting a single peer
// server address.
func (n *Node) peerClient(addr string) (*client.Client, error) {
	n.peerMu.Lock()
	defer n.peerMu.Unlock()
	if cl, ok := n.peerClients[addr]; ok {
		return cl, nil
	}
	// AuthToken carries the inter-node service credential (cfg.InternalToken) on
	// every forwarded op. peerClient is the SINGLE construction point for all
	// inter-node clients — request forwarding (forwardToLeader / forwardToOwner),
	// the shardAdmin remoteNode (rebalance_trigger.go, write_consistency.go), and
	// the rebalance coordinator all obtain their client here — so setting the
	// token once propagates it to every inter-node path. The destination node's
	// RBAC authorizer treats this token as the superuser service principal. Empty
	// token (open/nil-auth cluster) sends no token, preserving v1/no-auth behavior.
	//
	// When cfg.InterNodeTLS is set (client TLS enabled on the cluster), the peer's
	// client-facing port is TLS-wrapped, so the inter-node dial must ALSO be TLS or
	// the plaintext dial EOFs at the peer's TLS handshake. We dial over a CLONE of
	// InterNodeTLS with ServerName pinned to this peer's host so the peer's server
	// cert is verified against its SAN (never InsecureSkipVerify). AUTH is still the
	// internal token above; this TLS only provides the encrypted transport and
	// CA-verification of the peer. nil InterNodeTLS ⇒ TLSConfig stays nil ⇒ plaintext
	// dial, byte-identical to before (zero cost when client TLS is off).
	var tlsCfg *tls.Config
	if n.cfg.InterNodeTLS != nil {
		tlsCfg = n.cfg.InterNodeTLS.Clone()
		host, _, err := net.SplitHostPort(addr)
		if err != nil {
			host = addr
		}
		tlsCfg.ServerName = host
		// OPT-IN per-node identity: when an allowlist is configured, additionally
		// pin the dialed peer's verified leaf-cert CN to it (a CA-signed peer is no
		// longer enough — its CN must be an expected cluster member). The standard
		// CA/SAN verification runs FIRST (VerifyPeerCertificate is invoked only
		// after the chain verifies), so verifiedChains[0][0] is the verified leaf.
		// Empty allowlist => callback NOT attached => byte-identical to the
		// shared-CA path (this whole branch is skipped). Fail-loud: an empty chain
		// or an unlisted CN returns an error, failing the handshake (fail-closed).
		if len(n.cfg.NodeCNAllowlist) > 0 {
			tlsCfg.VerifyPeerCertificate = peerCNVerifier(n.cfg.NodeCNAllowlist)
			// VerifyPeerCertificate runs only during a FULL handshake; a resumed
			// TLS session (session ticket / client cache) skips it, which would let
			// a once-allowed peer bypass the CN allowlist on later resumed dials.
			// VerifyConnection runs on EVERY connection — full or resumed — so we
			// mirror the allowlist check here to keep the identity gate fail-closed
			// without disabling resumption.
			tlsCfg.VerifyConnection = peerCNConnVerifier(n.cfg.NodeCNAllowlist)
		}
	}
	cl, err := client.New(client.Config{
		Servers:          []string{addr},
		MaxNotLeaderHops: 3,
		AuthToken:        n.cfg.InternalToken,
		TLSConfig:        tlsCfg,
	})
	if err != nil {
		return nil, err
	}
	if n.peerClients == nil {
		n.peerClients = make(map[string]*client.Client)
	}
	n.peerClients[addr] = cl
	return cl, nil
}

// LeaderAddr returns the current leader's client-facing server address
// for shard 0, translating from the Raft transport address via the
// meta-Raft Members table in multi-node mode. In single-node mode
// (no meta-Raft), it returns shard 0's raft.LeaderAddr() directly —
// callers in that mode treat the cluster as one server anyway.
func (n *Node) LeaderAddr() string {
	s := n.getShard(0)
	if s == nil {
		return "" // shard 0 not hosted here (partitioned cluster); no local hint
	}
	raftAddr := s.LeaderAddr()
	if n.meta == nil {
		return raftAddr
	}
	return n.raftToServerAddr(raftAddr)
}

// Stats aggregates shard.Stats across all sub-stores. Allocates a
// fresh slice per call; intended for periodic metrics polling, not
// the per-request hot path.
func (n *Node) Stats() Stats {
	out := Stats{
		NumShards: n.cfg.NumShards,
		PerShard:  make([]shard.Stats, n.cfg.NumShards),
		WASMGate:  n.wasmGateStats(),
		WASMBlock: n.wasmBlockStats(),
		WASMBlobPush: WASMBlobPushStats{
			Acks:  n.wasmBlobPushAcks.Load(),
			Skips: n.wasmBlobPushSkips.Load(),
		},
		WASMBlobRetire: n.wasmBlobRetireStats(),
		KVIndex:        n.kvIndexStats(),
	}
	for i, s := range n.snapshotShards() {
		if s == nil {
			continue // shard not hosted on this node (partitioned cluster)
		}
		out.PerShard[i] = s.Stats()
	}
	return out
}

// Topology returns this node's view of the cluster routing state.
// Works in both single-node and multi-node mode. In single-node mode,
// Members is empty (there is no meta-Raft membership state to read);
// Leaders is still populated from the per-shard Raft groups so the
// smart-client and library shim can answer IsLeader/LeaderAddr
// queries against a single-node deployment.
//
// Lock-safe: n.shards is immutable post-construction; each Store's
// LeaderAddr() is internally synchronized; n.meta.FSM.State() (when
// non-nil) is RLock-protected.
func (n *Node) Topology() (ops.Topology, error) {
	var members []ops.TopologyMember
	if n.meta != nil {
		st := n.meta.FSM.State()
		members = make([]ops.TopologyMember, len(st.Members))
		for i, p := range st.Members {
			members[i] = ops.TopologyMember{NodeID: p.NodeID, ServerAddr: p.ServerAddr}
		}
	}
	leaders := make([]string, n.cfg.NumShards)
	for i, s := range n.snapshotShards() {
		if s == nil {
			continue // shard not hosted on this node (partitioned cluster)
		}
		raftAddr := s.LeaderAddr()
		if raftAddr == "" {
			continue
		}
		if n.meta == nil {
			// Single-node: the raft transport is in-memory, addresses
			// don't round-trip through raftToServerAddr. Use the raft
			// addr as-is; callers compare it to LeaderAddr() which uses
			// the same value in single-node mode.
			leaders[i] = raftAddr
			continue
		}
		leaders[i] = n.raftToServerAddr(raftAddr)
	}
	return ops.Topology{
		NumShards: n.cfg.NumShards,
		Members:   members,
		Leaders:   leaders,
		// Placement is fully known here (deterministic from membership), unlike
		// Leaders which this node only knows for the shards it hosts. It lets a
		// client route directly to an owner of any shard. Advanced
		// shard-by-shard by online rebalancing, so read under the lock.
		Placement: n.placementCopy(),
	}, nil
}

// IsLocalLeader returns true if this node is the Raft leader for the
// shard that owns key. Uses the same xxhash routing as Call.
func (n *Node) IsLocalLeader(key []byte) bool {
	if n.cfg.NumShards == 0 {
		return false
	}
	s := n.getShard(shardOf(key, n.cfg.NumShards))
	if s == nil {
		return false
	}
	return s.IsLeader()
}

// LeaderForKey returns the current leader's client-facing server address
// for the shard that owns key, or "" if unknown.
// In single-node mode (no meta-Raft), returns the shard's raw raft addr
// directly — the same behaviour as LeaderAddr() for shard 0.
func (n *Node) LeaderForKey(key []byte) string {
	if n.cfg.NumShards == 0 {
		return ""
	}
	s := n.getShard(shardOf(key, n.cfg.NumShards))
	if s == nil {
		return ""
	}
	raftAddr := s.LeaderAddr()
	if raftAddr == "" {
		return ""
	}
	if n.meta == nil {
		return raftAddr
	}
	return n.raftToServerAddr(raftAddr)
}

// Close shuts down every sub-store, then meta-Raft, then the mux
// listener (in that order). Idempotent; aggregates errors.
func (n *Node) Close() error {
	n.closeOnce.Do(func() {
		var errs []error
		// Stop the background PB seeder and the lease-keeper first so neither keeps
		// touching meta/engines while the shards and meta close (both nil in raft
		// mode). closeOnce guarantees the channel is closed exactly once.
		if n.pbSeedStop != nil {
			close(n.pbSeedStop)
			n.pbSeedWg.Wait()
		}
		// Stop the KV index observer before the shards close: it installs into and
		// walks their caches, and a walk aliases a store's live mmap. Draining the
		// walk gates first makes this bounded by a walk's abort check instead of by
		// a whole full-keyspace walk; stopKVIndexObserver then waits for the pass
		// itself, so no walk is in flight when the stores close below.
		n.drainAllKVIndexWalks()
		n.stopKVIndexObserver()
		// Stop shard formation before the shards close: the driver calls
		// BootstrapGroup on them.
		if n.formationStop != nil {
			close(n.formationStop)
			n.formationWg.Wait()
		}
		// End the blob fetch loops. They retry forever by design, so without this
		// a node with an unfetchable module would keep dialling after Close.
		if n.wasmFetchStop != nil {
			close(n.wasmFetchStop)
		}
		if n.leaseKeeper != nil {
			n.leaseKeeper.stop()
		}
		if n.pbBeacon != nil {
			n.pbBeacon.stop()
		}
		if n.pbFailover != nil {
			n.pbFailover.stop()
		}
		if n.pbShrink != nil {
			n.pbShrink.stop()
		}
		if n.pbGrow != nil {
			n.pbGrow.stop()
		}
		n.shardMu.Lock()
		for i, s := range n.shards {
			if s == nil {
				continue
			}
			if err := s.Close(); err != nil {
				errs = append(errs, fmt.Errorf("shard %d: %w", i, err))
			}
		}
		n.shardMu.Unlock()
		n.peerMu.Lock()
		for _, cl := range n.peerClients {
			_ = cl.Close()
		}
		n.peerClients = nil
		n.peerMu.Unlock()
		if n.wasmRT != nil {
			if err := n.wasmRT.Close(); err != nil {
				errs = append(errs, err)
			}
		}
		if n.meta != nil {
			if err := n.meta.Close(); err != nil {
				errs = append(errs, fmt.Errorf("meta: %w", err))
			}
		}
		if n.mux != nil {
			if err := n.mux.Close(); err != nil {
				errs = append(errs, fmt.Errorf("mux: %w", err))
			}
		}
		if n.fabric != nil {
			if err := n.fabric.Close(); err != nil {
				errs = append(errs, fmt.Errorf("fabric: %w", err))
			}
		}
		if n.pbTransport != nil {
			if err := n.pbTransport.Close(); err != nil {
				errs = append(errs, fmt.Errorf("pb transport: %w", err))
			}
		}
		if len(errs) > 0 {
			n.closeErr = errors.Join(errs...)
		}
	})
	return n.closeErr
}
