// SPDX-License-Identifier: Apache-2.0

// Package shard provides the Raft-based single-shard store for Rostam.
package shard

import (
	"errors"
	"fmt"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// Replication mode values for Config.ReplicationMode.
const (
	ReplicationModeRaft = "raft"
	ReplicationModePB   = "pb"
)

// Config governs a Store's behavior.
type Config struct {
	// NodeID is the unique identifier for this node in the Raft cluster.
	NodeID string

	// DataDir is the directory where Raft's BoltDB log and FileSnapshotStore live.
	DataDir string

	// Cache is the cache configuration. NumShards is forced to 1.
	Cache cache.Config

	// Ops is the user-supplied (plus built-in) op registry.
	Ops *ops.Registry

	// Bootstrap, when true and no existing Raft state is found, brings up
	// this node as a single-node cluster. Set true on the first node only.
	Bootstrap bool

	// SnapshotIntervalMs is how often Raft tries to take an FSM snapshot
	// (milliseconds). Default 5 min.
	SnapshotIntervalMs int

	// SnapshotThreshold is the minimum log entry count to trigger a snapshot.
	SnapshotThreshold uint64

	// KVIndexReconcileIntervalMs is how often this store reconciles ONE of its
	// KV index definitions against the live cache (milliseconds). Default 60 000;
	// 0 (or the zero value of a hand-built Config) DISABLES the pass, and a
	// negative value is a configuration error.
	//
	// BEWARE THE TWIN KNOB. rostam.DirectConfig has a field of the same name and
	// units whose zero means the OPPOSITE: there 0 keeps the 60 s default and a
	// NEGATIVE value disables. Each follows its own struct's established
	// convention — this one is reached through DefaultConfig, which fills the
	// default in, while DirectConfig is a bare literal and mirrors its own
	// Cache.TTLSweepIntervalMs. So porting a config across by copying the field
	// silently flips the pass on or off; set it deliberately on each side.
	//
	// Disabling it is safe and never changes an answer: every candidate is
	// re-read and re-checked against the live value, so a posting the pass would
	// have removed costs one wasted lookup and can never produce a wrong row.
	// What it bounds is MEMORY, against the few removal paths the cache's
	// onRemove hook cannot name a key for. See shard/kv_index_reconcile.go.
	KVIndexReconcileIntervalMs int

	// RaftHeartbeatMs and RaftElectionMs override hashicorp/raft defaults.
	RaftHeartbeatMs int
	RaftElectionMs  int

	// NoSync, when true, sets the BoltDB log store to non-fsync mode.
	// Trades durability for write latency. Recommended for cache workloads.
	NoSync bool

	// VolatileLog, when true, replaces the on-disk WAL with an in-memory log
	// (no write() syscall on the hot path); durability comes only from
	// replication. See raft.Config.VolatileLog for the fresh-rejoin safety
	// contract. Applies to data shards only (the meta group stays durable).
	VolatileLog bool

	// RaftLogLevel controls how loud the embedded hashicorp/raft is:
	// "TRACE"|"DEBUG"|"INFO"|"WARN"|"ERROR"|"OFF". Empty means INFO. Tests and
	// benchmarks that build clusters in a loop should set "ERROR" — `go test`
	// merges the test binary's stdout and stderr, so raft's per-election output
	// otherwise lands interleaved with benchmark results.
	RaftLogLevel string

	// RaftStreamLayer (optional) overrides the default Raft transport.
	// When set, hashicorp/raft uses this StreamLayer (wrapped in a
	// NetworkTransport); otherwise shard.Store falls back to the
	// in-memory transport. Set by cluster.Node for
	// the multi-node path to share a single TCP listener.
	RaftStreamLayer hraft.StreamLayer

	// RaftPeers (optional) overrides the default single-self bootstrap
	// voter list. When non-empty, BootstrapCluster is called with these
	// servers. Set by cluster.Node for the multi-node path.
	RaftPeers []hraft.Server

	// RaftTransport (optional) is a fully-built per-group Raft transport that
	// takes precedence over RaftStreamLayer. Set by cluster.Node when
	// RaftTransport=="fabric" to route this shard's group over the shared
	// multiplexed batching transport (raft/fabric) instead of the mux
	// StreamLayer.
	RaftTransport hraft.Transport

	// PersistentVectors makes this shard's vector collections mmap-backed
	// (off-heap) per node. Raft (log + FSM snapshot) remains the durability
	// authority — the mmap is a non-authoritative memory layout, wiped and
	// repopulated from Raft on restart/catch-up. See
	// vector.OpenCollectionStorePersistent.
	PersistentVectors bool

	// ReplicationMode selects the data-plane replication engine for this shard.
	// "" or "raft" → the default per-shard Raft group (unchanged). "pb" → the
	// primary-backup/ISR engine (shard/pbisr). Cluster-level: every shard in a
	// deployment uses the same mode. See shard/pbisr/DUAL-MODE-DESIGN.md.
	ReplicationMode string

	// PBControl / PBTransport are injected by cluster.Node when ReplicationMode
	// is "pb": the MetaRaft-authoritative control plane and the inter-node
	// replication transport. A single-node PB store (ISR={self}) needs no
	// transport, so PBTransport may be nil there. Ignored in Raft mode.
	PBControl   pbisr.Control
	PBTransport pbisr.Transport

	// WASMSnapshot / WASMRestore (optional; nil ok) carry DYNAMIC op
	// registrations across a snapshot install. They exist because the ops
	// registry a committed entry is dispatched through is MUTABLE at runtime:
	// cluster's __register_wasm__ loads a module and registers its op on every
	// replica that applies the entry. A replica that joins (or catches up) via
	// InstallSnapshot never applies those entries — the snapshot replaces them —
	// so without this it would be permanently missing every op registered before
	// the snapshot was taken, and would hit the ErrOpNotRegistered fail-closed
	// halt (see apply_class.go) on the first invocation. That is deterministic,
	// not a race: it happens to every AddShardOwner-joined replica and to every
	// replica brought up after log compaction.
	//
	// WASMSnapshot returns an opaque blob captured on the FSM goroutine (so it is
	// consistent with the cache/vector state at the same index); WASMRestore
	// installs a blob produced by it. The shard package does not interpret the
	// bytes — the encoding belongs to the cluster layer that owns the registry
	// and the WASM runtime.
	//
	// The cluster layer binds both to THIS store's group (see ShardIndex): the
	// blob records which registrations this group's log carries, which is what a
	// joining replica needs in order to know whether it may propose invocations
	// into the group it just joined. That is why they are per-store closures and
	// not one node-wide pair of functions.
	WASMSnapshot func() []byte
	WASMRestore  func([]byte) error

	// OnApplyRetry / OnApplyRetryCleared (optional; nil ok) observe a classRetry
	// BLOCK: a committed entry this replica cannot apply yet because the module
	// version it names is not resident here (thin registration markers name their
	// module rather than carrying it — see shard.classRetry).
	//
	// OnApplyRetry fires before EVERY wait, with a 1-based attempt count and how
	// long this entry has been blocked; OnApplyRetryCleared fires once when it
	// finally applies, carrying THE ERROR THAT CAUSED THE BLOCK rather than the
	// successful re-run's (nil) one — the observer keys its live record on the op
	// and group inside that error, so anything else would leave the record it
	// created un-retired forever. Together they are what makes an unbounded wait honest: the
	// design trades a halt for a client-visible retryable error and a silent wait,
	// and the wait's visibility is the whole of what makes that trade defensible.
	//
	// NEITHER MAY BLOCK, AND NEITHER MAY HOLD A LOCK AN APPLY PATH TAKES. They run
	// on the FSM apply goroutine. The cluster implementation kicks a
	// fire-and-forget fetch and returns; if it fetched inline, or returned still
	// holding cluster's wasmApplyMu, it would stall every other group's snapshot on
	// this node instead of only the group that is waiting.
	OnApplyRetry        func(err error, attempt int, blockedFor time.Duration)
	OnApplyRetryCleared func(err error, attempts int, blockedFor time.Duration)

	// ApplyRetryInterval is the BASE backoff between re-runs of a blocked entry;
	// it doubles up to a one-second cap. Zero means the default (50ms). It exists
	// so tests can drive a block deterministically without waiting on production
	// cadences, and so a deployment with an unusually slow blob transport can
	// stretch the schedule rather than spin on it.
	ApplyRetryInterval time.Duration

	// ShardIndex is the global shard index this Store represents. Used for PB
	// Control lookups (Primary/Epoch), pbisr.New, and newPBReplicator, and — in
	// BOTH replication modes — published to the op dispatcher as
	// ops.TxContext.ShardIndex.
	//
	// It is no longer ignored in Raft mode. Each Raft group is indeed scoped by
	// its own DataDir/transport, but a handler that mutates NODE-WIDE state
	// (__register_wasm__ installs into the node-wide ops registry) cannot tell
	// from that scoping which group's log it is applying, and the cluster layer
	// needs exactly that attribution. Defaults to 0, which is correct for a
	// single-group store; cluster.Node sets it per group.
	ShardIndex int

	// PBRegister (optional; nil ok) is invoked by shard.New's PB branch after
	// building the pbisr.Engine, as PBRegister(ShardIndex, eng), so a cluster
	// wiring layer can register this shard's engine as the network Receiver for
	// inbound replication (e.g. NetTransport.Register). Ignored in Raft mode.
	PBRegister func(shard int, r pbisr.Receiver)

	// PBClock (optional; nil ok) overrides the pbisr.Engine's monotonic clock
	// (via pbisr.WithClock). The cluster wiring passes ONE shared clock to every
	// shard's engine AND to its lease-keeper so the OH1 lease-expiry comparison
	// (engine now() vs granted expiry) is judged on a single time source. nil ⇒
	// the engine keeps its default process-start monotonic clock. Ignored in Raft
	// mode.
	PBClock func() int64

	// PBCommitLevel selects the PB durability contract (via pbisr.WithCommitLevel).
	// Zero value = pbisr.CommitFullISR = wait for the full ISR (default). Set to
	// pbisr.CommitPrimary for commit-on-primary / async replication (a durability
	// downgrade — see pbisr.CommitLevel). Ignored in Raft mode.
	PBCommitLevel pbisr.CommitLevel

	// PBFrontierStampEvery / PBFrontierStampInterval tune how often the PB applied
	// frontier is persisted into the cache header (the durable frontier). The
	// stamp is crash-ordered (page data msync'd before the header), which is what
	// makes the persisted watermark unable to over-report; it is AMORTISED rather
	// than made cheap, because weakening the ordering is what would make it able to.
	//
	//   - PBFrontierStampInterval: stamp at most this often, and (if anything was
	//     written) at least this often. It is the real bound on the under-report,
	//     and it is the knob to reach for. <= 0 selects 100ms.
	//   - PBFrontierStampEvery: an OPT-IN extra trigger that also stamps after this
	//     many applies. <= 0 (the default) disables it. It is off by default because
	//     an msync costs O(mapped region), not O(bytes changed), so a count trigger
	//     multiplies a large fixed cost — measured at +73% on the PB write path at
	//     1024 applies — while the ticker already bounds the lag at every write rate.
	//     See defaultPBFrontierStampEvery for the measurements.
	//
	// Neither knob can affect CORRECTNESS. A staler watermark only under-reports,
	// and pbisr's log matching turns an under-report into either a true-prefix
	// catch-up or a clean divergence reject. Ignored in Raft mode, and inert on a
	// heap-mode (no Cache.DataDir) cache, which has no header to stamp.
	PBFrontierStampEvery    int
	PBFrontierStampInterval time.Duration

	// EnableApplyStamp turns on the deterministic apply-time expiry stamp (#4 Phase
	// B / B1): when true, the leader/primary encodes each write as a STAMPED log
	// entry carrying its propose-time wall clock, and every replica evaluates and
	// re-stamps that write's expiry against the baked-in clock instead of its own —
	// making the replicated committed KV state LOGICALLY byte-identical under
	// per-node clock skew.
	//
	// "Logically byte-identical" = the committed key/value/absolute-expiry SET is
	// identical on every replica. The PHYSICAL serializeSnapshot bytes can still
	// differ, because Iterate wall-clock-filters logically-expired entries at
	// snapshot time; the determinism fingerprint pins a fixed clock precisely to
	// factor that out. SCOPE: this covers the KV cache expiry sites only. Vector
	// per-key TTL (the ...KeyTTL insert/upsert ops' keyTTLMs, still resolved to
	// now+ttl per replica) is a committed-state expiry site NOT yet on the stamped
	// clock — see the apply_class.go vector-audit TODO; it is out of scope for KV B1.
	//
	// Default FALSE for a SAFE TWO-PHASE ROLLOUT: a binary must be able to DECODE
	// stamped entries before any leader ENCODES them. Deploy the decode-capable
	// binary (this flag off) to every node first, THEN flip it on. Mixed-version
	// safety: a node that hits a version it cannot parse returns ErrLogEntryVersion,
	// which classifies classFatal and HALTS (fail-closed) rather than silently
	// skipping — this net covers a POST-decoder version bump / premature enable. The
	// PRE-decoder case (an OLD binary with no 0x00 support reads the marker as
	// opNameLen=0 → ErrOpNotRegistered → classAdvance → skip) is NOT caught by
	// classification and relies on the rollout ordering above. Single-node Direct
	// never encodes (no Raft/PB propose), so this is inert there. See
	// EncodeLogEntryStamped.
	EnableApplyStamp bool
}

// DefaultConfig builds a sensible Config rooted at dataDir, identifying as
// nodeID, using registry as the op source. Single-node bootstrap is OFF by
// default; callers must opt in.
func DefaultConfig(dataDir, nodeID string, registry *ops.Registry) Config {
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	return Config{
		NodeID:             nodeID,
		DataDir:            dataDir,
		Cache:              cc,
		Ops:                registry,
		Bootstrap:          false,
		SnapshotIntervalMs: 5 * 60 * 1000, // 5 min
		SnapshotThreshold:  10_000,

		KVIndexReconcileIntervalMs: defaultKVIndexReconcileIntervalMs,
		RaftHeartbeatMs:            1000,
		RaftElectionMs:             1000,
		NoSync:                     false,
	}
}

// Validate enforces invariants.
func (c Config) Validate() error {
	if c.NodeID == "" {
		return errors.New("shard.Config: NodeID is required")
	}
	if c.DataDir == "" {
		return errors.New("shard.Config: DataDir is required")
	}
	if c.Ops == nil {
		return errors.New("shard.Config: Ops registry is required")
	}
	if c.Cache.NumShards != 1 {
		return fmt.Errorf("shard.Config: Cache.NumShards=%d must be 1", c.Cache.NumShards)
	}
	if err := c.Cache.Validate(); err != nil {
		return fmt.Errorf("shard.Config: cache: %w", err)
	}
	if c.SnapshotIntervalMs < 0 {
		return errors.New("shard.Config: SnapshotIntervalMs must be >= 0")
	}
	if c.KVIndexReconcileIntervalMs < 0 {
		return errors.New("shard.Config: KVIndexReconcileIntervalMs must be >= 0 (0 disables the pass)")
	}
	if c.RaftHeartbeatMs < 0 {
		return errors.New("shard.Config: RaftHeartbeatMs must be >= 0")
	}
	if c.RaftElectionMs < 0 {
		return errors.New("shard.Config: RaftElectionMs must be >= 0")
	}
	switch c.ReplicationMode {
	case "", ReplicationModeRaft, ReplicationModePB:
	default:
		return fmt.Errorf("shard.Config: unknown ReplicationMode %q (want %q or %q)",
			c.ReplicationMode, ReplicationModeRaft, ReplicationModePB)
	}
	if c.ReplicationMode == ReplicationModePB && c.PBControl == nil {
		return errors.New("shard.Config: ReplicationMode=pb requires PBControl")
	}
	if c.ReplicationMode == ReplicationModePB && c.PersistentVectors {
		return errors.New("shard.Config: PersistentVectors is not supported with ReplicationMode=pb (-persistent-vectors " +
			"with -replication-mode=pb): persistent-vectors wipes on-disk vector data at boot and relies on Raft log " +
			"replay to repopulate it, but pb mode has no Raft log to replay from, so vector data would be lost with " +
			"no recovery path")
	}
	return nil
}
