// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft"
	"github.com/rostamlabs/rostam/shard/pbisr"
	"github.com/rostamlabs/rostam/vector"
)

// Stats is a snapshot combining cache + raft observability.
type Stats struct {
	Cache cache.Stats
	// Raft is hashicorp/raft's stats map; opaque per their docs.
	Raft map[string]string
}

// Store is the user-facing handle for a Raft-replicated cache.
type Store struct {
	cfg      Config
	cache    *cache.Cache
	registry *ops.Registry
	vectors  *vector.CollectionStore
	raft     replicator // the active data-plane engine (Raft by default; see replicator.go)
	fsm      *fsm
	tx       *ops.TxContext // reused across read-only Call paths; same caching rationale as fsm.tx

	// readindex coalesces concurrent Linearizable-read barriers on THIS Store into
	// one shared VerifyLeader+Barrier RTT, while guaranteeing no reader is served a
	// frontier captured before it arrived (see readindex_coalesce.go).
	readindex readindexCoalescer

	// leaderFrontierFn is the injected (by the cluster Node) follower-side hook a
	// ConsistencyBoundedStaleness read uses to learn the shard LEADER's committed
	// frontier via a coalesced RTT (__shard_readindex__). nil by default (single-node
	// / leader-only deployments never invoke it — the leader path uses its own
	// CommitIndex). When nil on a follower, the bounded-staleness path falls back to
	// the Linearizable barrier (VerifyLeader fails on a follower ⇒ NotLeaderError ⇒
	// caller routes to leader: a safe upgrade).
	leaderFrontierFn func(deadline time.Time) (uint64, error)

	// stop is closed by Close BEFORE the replication engine is shut down. It is
	// what lets a group parked in a classRetry block (an unbounded wait for module
	// bytes) abandon the wait: the FSM apply goroutine is the same goroutine raft
	// uses to observe its own shutdown, so without this Close would hang for as
	// long as the block lasts, which is forever by design.
	stop     chan struct{}
	stopOnce sync.Once

	// pbFrontier persists the PB engine's applied frontier into the cache
	// header. Non-nil only in PB mode. Close stops it AFTER replication has
	// shut down and BEFORE the cache is closed, which is the only window in which a
	// final, exact stamp is both possible (the mapping still exists) and complete
	// (no further apply can advance the frontier past it).
	pbFrontier *pbFrontierStamper
}

// SetLeaderFrontierFn installs the follower-side leader-frontier hook used by
// ConsistencyBoundedStaleness reads (see leaderFrontierFn). The cluster Node sets it
// per shard at construction, baking the shard index into the closure. nil clears it.
func (s *Store) SetLeaderFrontierFn(fn func(deadline time.Time) (uint64, error)) {
	s.leaderFrontierFn = fn
}

// defaultServerWriteTimeout is the FALLBACK write-deadline used only when the
// effective server.Config.WriteTimeout was not threaded into the cache config
// (Cache.ServerWriteTimeout == 0). It matches server.Config's own default
// WriteTimeout, so a library embedder that overrides neither gets the same 30s on
// both sides. It is duplicated here rather than imported to avoid a shard→server
// dependency. This is NO LONGER a silent mirror of a possibly-different operative
// value: when the transport sets a non-default WriteTimeout, that value is threaded
// in via Cache.ServerWriteTimeout and the derivation + fail-closed assertion below
// track it, so the drain fence cannot drift out of step with the real deadline.
const defaultServerWriteTimeout = 30 * time.Second

// replicatedCacheShard reports whether cfg is a cluster-replication shard whose
// cache must fail closed (reject) at capacity rather than silently evict (#4
// Phase B / B2). True for PB mode or any Raft shard wired with a cluster
// transport — cluster.Node's buildShardConfig sets exactly one of
// RaftStreamLayer/RaftTransport for every cluster shard, including
// resharding-joined ones, so this covers the online-join path that
// construction-time peer counts miss. A bare in-memory Store keeps evict.
func replicatedCacheShard(cfg Config) bool {
	return cfg.ReplicationMode == ReplicationModePB ||
		cfg.RaftStreamLayer != nil || cfg.RaftTransport != nil
}

// New constructs a Store. On Bootstrap, the node starts a fresh single-node
// cluster; otherwise it joins an existing one.
func New(cfg Config) (*Store, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	// #4 Phase B (B2): a replicated shard MUST NOT silently evict. Under the
	// default PolicyRingbufEvict each replica evicts DIFFERENT oldest keys based
	// on its own occupancy + wall-clock TTL — no error, so nothing catches the
	// resulting committed-state divergence. Force PolicyRejectWrites for any
	// cluster-replication shard so capacity pressure becomes a deterministic
	// cache.ErrFull that Phase A fails closed on (a loud halt, never silent
	// divergence).
	//
	// The signal is "does this shard participate in cluster replication": PB mode,
	// or a cluster-wired Raft transport. cluster.Node's buildShardConfig sets
	// exactly one of RaftStreamLayer/RaftTransport for EVERY cluster shard —
	// including resharding-joined shards — so this covers the online-join path
	// that construction-time peer/RF counts miss (RaftPeers is empty on a joiner).
	// A bare in-memory Store (no cluster transport) and single-node Direct (which
	// never reaches this constructor) keep evict. This overrides an
	// explicitly-configured RingbufEvict on a replicated shard by design —
	// evict-oldest is unsafe under replication.
	if replicatedCacheShard(cfg) && cfg.Cache.AtCapPolicy != cache.PolicyRejectWrites {
		cfg.Cache.AtCapPolicy = cache.PolicyRejectWrites
		slog.Warn("forcing cache AtCapPolicy=reject-writes for cross-replica determinism on replicated shard (evict would silently diverge)", "component", "shard", "shard", cfg.ShardIndex)
	}

	// #4 Phase B (B3a): mark the cache replicated so it suppresses WALL-CLOCK
	// physical removal of expired keys (background sweeper off; client-read lazy
	// drop suppressed; warm-restart rebuild keeps all entries). The same predicate
	// as B2 — a shard participates in cluster replication iff it is PB mode or has a
	// cluster-wired Raft transport. Combined with B1's stamped apply clock this
	// makes the committed key set LOGICALLY byte-identical (identical key/value/exp
	// set; physical snapshot bytes may still differ via the Iterate wall-clock
	// filter) across replicas ticking at different wall times; the apply path
	// (GetAt) still tombstones expired keys deterministically. A bare in-memory
	// Store and single-node Direct keep the sweeper + wall-clock lazy removal
	// (Replicated stays false).
	if replicatedCacheShard(cfg) {
		cfg.Cache.Replicated = true
		// Enable ONLINE relocating compaction with quarantine-then-reset recycle
		// (cache/compact_online.go) so a persistent replicated shard reclaims ghost
		// page bytes WHILE running instead of only at restart (cold compaction) — the
		// difference between eventually hitting ErrFull on dead versions and recovering
		// write capacity online. The cache-side master gate (isMmap && Replicated &&
		// reject-writes) means this is a no-op for a heap replicated shard (no DataDir)
		// and for single-node / ringbuf shards, which stay byte-for-byte unchanged.
		cfg.Cache.OnlineCompaction = true
		// AliasQuarantine is the drain fence a retired page must sit through before its
		// extent is reset and reused: it must exceed the maximum wall-clock lifetime of
		// a zero-copy read alias. The only holder of a raw alias across a blocking
		// network flush is the TCP response writer, bounded by the EFFECTIVE
		// server.Config.WriteTimeout (the INLINE non-pipelined path; the pipelined path
		// copies at dispatch, so its hold is ~0); every other consumer copies
		// synchronously. So the fence must be at least 2*WriteTimeout — conservative
		// headroom over the inline hold. We derive it from the effective WriteTimeout
		// threaded in via Cache.ServerWriteTimeout (single source of truth), NOT a
		// hardcoded mirror, so raising the server's WriteTimeout raises the fence in
		// lockstep instead of silently defeating it.
		effectiveWriteTimeout := cfg.Cache.ServerWriteTimeout
		if effectiveWriteTimeout <= 0 {
			// rostam.NewServer always threads WriteTimeout (its own default is the same
			// 30s), so this fallback is only reached by a library embedder building a
			// REPLICATED shard directly via NewEmbedded without setting WriteTimeout. If
			// that embedder runs a custom transport whose response write can block LONGER
			// than 30s, this fence under-sizes the real alias-hold and a recycle could tear
			// a read. We cannot see the embedder's transport, so warn loudly rather than
			// assume — the safe action is for them to set WriteTimeout to their transport's
			// write deadline (or set AliasQuarantine explicitly, which the fail-closed
			// check below then validates).
			slog.Warn("replicated shard has online compaction enabled but no server WriteTimeout was threaded; "+
				"the alias-drain fence falls back to 2*30s — if your transport can hold a zero-copy read response "+
				"longer than 30s, set Config.WriteTimeout (or an explicit Cache.AliasQuarantine) to your write deadline",
				"component", "shard", "shard", cfg.ShardIndex)
			effectiveWriteTimeout = defaultServerWriteTimeout
		}
		minAliasQuarantine := 2 * effectiveWriteTimeout
		if cfg.Cache.AliasQuarantine == 0 {
			cfg.Cache.AliasQuarantine = minAliasQuarantine
		}
		// FAIL-CLOSED corruption invariant. An operator may set AliasQuarantine
		// explicitly; if it is below 2*WriteTimeout the drain fence is shorter than the
		// worst-case inline alias hold, so an online page recycle could overwrite bytes
		// a response writer still aliases and ship a torn read. A comment cannot enforce
		// a corruption invariant — refuse to start rather than run an unsafe fence.
		if cfg.Cache.AliasQuarantine < minAliasQuarantine {
			return nil, fmt.Errorf("shard: unsafe online-compaction fence on replicated shard %d: "+
				"AliasQuarantine (%s) must be >= 2*server WriteTimeout (2*%s = %s); a shorter "+
				"quarantine can recycle a retired mmap page while a zero-copy response alias still "+
				"references its bytes (torn read)",
				cfg.ShardIndex, cfg.Cache.AliasQuarantine, effectiveWriteTimeout, minAliasQuarantine)
		}
	}

	c, err := cache.New(cfg.Cache)
	if err != nil {
		return nil, fmt.Errorf("shard: cache: %w", err)
	}

	vectorStore, err := vector.OpenCollectionStorePersistent(cfg.DataDir, cfg.PersistentVectors)
	if err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("shard: open vector store: %w", err)
	}

	fsmImpl := newFSM(c, cfg.Ops, cfg.Cache.Durable, vectorStore)
	fsmImpl.wasmSnapshot = cfg.WASMSnapshot
	fsmImpl.wasmRestore = cfg.WASMRestore
	fsmImpl.onApplyRetry = cfg.OnApplyRetry
	fsmImpl.onApplyRetryCleared = cfg.OnApplyRetryCleared
	fsmImpl.applyRetryWait = cfg.ApplyRetryInterval
	// The stop channel is built BEFORE raft.NewNode starts the FSM apply goroutine,
	// so the goroutine-start happens-before edge publishes it without a race.
	stop := make(chan struct{})
	fsmImpl.stop = stop
	// Tell the apply dispatcher which group it serves. Only handlers that mutate
	// NODE-WIDE state read it (today: __register_wasm__, whose cluster-side hook
	// has to attribute each apply to the group whose log carried it).
	fsmImpl.tx.SetShardIndex(cfg.ShardIndex)

	var rep replicator
	var pbFrontier *pbFrontierStamper // PB mode only; nil in Raft mode
	switch cfg.ReplicationMode {
	case ReplicationModePB:
		// Thread the cluster's shared monotonic clock into the engine (via
		// WithClock) so its lease fence and the lease-keeper's grants read the
		// SAME time source. A nil PBClock is ignored by WithClock (default clock).
		// WithDataRelease closes the encode-buffer loop: entries the engine's
		// catch-up ring evicts return to the shard's logEntryPool (see
		// EncodeLogEntryPooled's ownership contract).
		//
		// STAGE 4.2 — DURABLE FRONTIER. The cache warm-restarts its committed state
		// from the mmap pages (rebuildIndexFromPages), but in PB mode there is no log
		// and no snapshot to re-derive a POSITION from, so without the two calls below
		// a restarted node would present a genesis (0,0) frontier over a full FSM and
		// every catch-up decision downstream would be made on that lie.
		//   - PBFrontier() reads back the (seq, epoch) the pages are known to
		//     materialize; it is (0,0) — a no-op — on a fresh dir, in heap mode, and
		//     in Raft mode.
		//   - the stamper re-persists it as the engine applies, amortised, using the
		//     crash-ordered path so the persisted value can lag but never lead.
		//
		// STAGE 4.3 — SNAPSHOT TRANSFER. WithSnapshotStore gives the engine the
		// FSM serialize/install pair plus the durable POISON FENCE, which together
		// are the only exit from a ring-cold or diverged member. Wiring it also
		// SEEDS the engine's poisoned latch from the fence, so a node that crashed
		// mid-install comes back refusing to serve instead of presenting a
		// half-wiped FSM behind a meaningless watermark.
		//
		// The restored frontier is deliberately still read from the cache header:
		// the fence, not the frontier, is what says "do not trust this". A poisoned
		// node's watermark is simply never consulted (CatchupInfo answers not-OK and
		// zeroes it), so there is no second source of truth to keep in step.
		pbSeq, pbEpoch := c.PBFrontier()
		pbStamp := newPBFrontierStamper(c, cfg.PBFrontierStampEvery, cfg.PBFrontierStampInterval)
		pbSnap := newPBSnapshotStore(c, vectorStore, fsmImpl, pbStamp, newPBRestoreFence(cfg.DataDir))
		eng := pbisr.New(cfg.NodeID, cfg.ShardIndex, cfg.PBControl, cfg.PBTransport, newShardApplier(fsmImpl),
			pbisr.WithClock(cfg.PBClock), pbisr.WithDataRelease(ReleaseLogEntry),
			pbisr.WithCommitLevel(cfg.PBCommitLevel),
			pbisr.WithRestoredFrontier(pbSeq, pbEpoch),
			pbisr.WithFrontierSink(pbStamp.record),
			pbisr.WithSnapshotStore(pbSnap))
		pbFrontier = pbStamp
		// Single-node/self-lease bootstrap: without the Plan-2 control-plane lease
		// loop, grant the primary a long self-lease so Propose's OH1 fence passes.
		// MetaRaft-driven lease renewal replaces this.
		if cfg.PBControl.Primary(cfg.ShardIndex) == cfg.NodeID {
			eng.GrantLease(cfg.PBControl.Epoch(cfg.ShardIndex), int64(^uint64(0)>>1))
		}
		if cfg.PBRegister != nil {
			cfg.PBRegister(cfg.ShardIndex, eng)
		}
		rep = newPBReplicator(cfg.NodeID, cfg.ShardIndex, eng, cfg.PBControl)
	default:
		// Wire the FSM's fail-closed gate (production-readiness #4) to LIVE Raft group
		// membership — see fsm.isReplicated for why construction-time peers are unsafe
		// (online-resharding joiners are built with an empty peer list). Only the Raft
		// path consults isReplicated (fsm.Apply/ApplyBatch); the PB path uses
		// shardApplier.Apply, which classifies unconditionally, so it needs no wiring.
		//
		// The field is assigned BEFORE raft.NewNode, which starts the FSM apply
		// goroutine: a happens-before edge (goroutine start) makes the assignment
		// visible without a race. The node handle itself is published AFTER NewNode via
		// an atomic.Pointer — the closure reads it atomically, and while it is still nil
		// (the brief window before Store, during which the FSM may already apply the
		// bootstrap/config entries) the gate FAILS CLOSED (raftReplicatedFn treats
		// ok==false as replicated). No data ops are lost in that window; a config entry
		// is a non-command no-op in ApplyBatch.
		var raftNode atomic.Pointer[raft.Node]
		fsmImpl.isReplicated = raftReplicatedFn(func() (int, bool) {
			n := raftNode.Load()
			if n == nil {
				return 0, false // node not yet published — fail closed
			}
			return n.NumServers()
		})
		rn, err := raft.NewNode(raft.Config{
			NodeID:             cfg.NodeID,
			DataDir:            cfg.DataDir,
			Bootstrap:          cfg.Bootstrap,
			HeartbeatMs:        cfg.RaftHeartbeatMs,
			ElectionMs:         cfg.RaftElectionMs,
			SnapshotIntervalMs: cfg.SnapshotIntervalMs,
			SnapshotThreshold:  cfg.SnapshotThreshold,
			NoSync:             cfg.NoSync,
			VolatileLog:        cfg.VolatileLog,
			LogLevel:           cfg.RaftLogLevel,
			StreamLayer:        cfg.RaftStreamLayer,
			Transport:          cfg.RaftTransport,
			Peers:              cfg.RaftPeers,
		}, fsmImpl)
		if err != nil {
			_ = vectorStore.Close()
			_ = c.Close()
			return nil, fmt.Errorf("shard: raft: %w", err)
		}
		raftNode.Store(rn)
		rep = rn
	}

	storeTx := ops.NewTxContextWithVectors(c, vectorStore)
	storeTx.SetShardIndex(cfg.ShardIndex)
	return &Store{
		cfg:        cfg,
		cache:      c,
		registry:   cfg.Ops,
		vectors:    vectorStore,
		raft:       rep,
		fsm:        fsmImpl,
		tx:         storeTx,
		stop:       stop,
		pbFrontier: pbFrontier,
	}, nil
}

// raftReplicatedFn builds the FSM's isReplicated gate over a LIVE Raft group-size
// source (raft.Node.NumServers). It FAILS CLOSED: the gate reports replicated
// (halt-on-classFatal enabled) unless the group is positively observed to have
// exactly one server. Rationale:
//
//   - sizeFn returns (n, false): the configuration could not be read — do not risk
//     disabling the divergence guard on an unknown group; treat as replicated.
//   - n == 0: a fresh online-resharding joiner (built with an empty peer list via
//     cluster AddShardOwner) has not yet learned its configuration. It is destined
//     for an RF>1 group, so treat as replicated until membership is known.
//   - n == 1: a genuine single-node cluster — no peers to diverge from, so the gate
//     is OFF and Apply advances/surfaces classFatal errors normally (no needless
//     halt).
//   - n > 1: a real RF>1 group — gate ON.
func raftReplicatedFn(sizeFn func() (int, bool)) func() bool {
	return func() bool {
		n, ok := sizeFn()
		if !ok || n == 0 {
			return true // fail closed: cannot prove single-node
		}
		return n > 1
	}
}

// Close shuts down Raft and the cache. Aggregates errors from both
// via errors.Join so neither resource is leaked on a Raft shutdown failure.
func (s *Store) Close() error {
	var errs []error
	// Release any classRetry block FIRST. A blocked FSM apply goroutine is the
	// same goroutine hashicorp/raft would use to notice its shutdown, so shutting
	// raft down while a group is parked waiting for module bytes would wait for a
	// wait that has no end. sync.Once because Close is not documented as
	// single-shot and a second close(3) would panic.
	s.stopOnce.Do(func() { close(s.stop) })
	if err := s.raft.Shutdown(); err != nil {
		errs = append(errs, err)
	}
	// Replication is down, so the applied frontier can no longer move: stamp it
	// exactly, while the cache mapping is still live. A clean shutdown therefore
	// restores an EXACT frontier; only an unclean stop falls back to the amortised
	// (at most one interval stale) value.
	if s.pbFrontier != nil {
		s.pbFrontier.Close()
	}
	if s.vectors != nil {
		if err := s.vectors.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := s.cache.Close(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

// Get reads directly from the local cache (no Raft). Returns cache.ErrNotFound if absent.
func (s *Store) Get(key []byte) ([]byte, error) {
	return s.cache.Get(key)
}

// Put writes through Raft.
func (s *Store) Put(key, value []byte, ttl time.Duration) error {
	if !s.IsLeader() {
		return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	args := ops.EncodePutArgs(key, value, ttl)
	_, err := s.applyOp("put", args)
	return err
}

// Del writes through Raft. Returns (true, nil) if the entry existed and was
// deleted, (false, nil) if the key was not present, or (false, ErrNotLeader)
// if this node is not the leader.
func (s *Store) Del(key []byte) (bool, error) {
	if !s.IsLeader() {
		return false, &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	res, err := s.applyOp("del", ops.EncodeKeyArgs(key))
	if err != nil {
		return false, err
	}
	return len(res) == 1 && res[0] == 1, nil
}

// Expire updates the TTL of an existing key.
func (s *Store) Expire(key []byte, ttl time.Duration) error {
	if !s.IsLeader() {
		return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	_, err := s.applyOp("expire", ops.EncodeExpireArgs(key, ttl))
	return err
}

// readServedHook holds an optional test-only observer fired for each OpReadOnly
// serve with whether this replica is the current leader. Stored atomically so the
// (concurrent) serve path and test setup don't race. nil in prod → zero overhead;
// used to verify LeaderOnly vs AnyReplica routing.
var readServedHook atomic.Pointer[func(isLeader bool)]

// SetReadServedHook installs (or clears with nil) the OpReadOnly serve observer.
// Test-only.
func SetReadServedHook(fn func(isLeader bool)) {
	if fn == nil {
		readServedHook.Store(nil)
		return
	}
	readServedHook.Store(&fn)
}

// linearizableReadTimeout bounds the readIndex barrier's catch-up poll: after a
// confirmed-leadership VerifyLeader, how long to wait for the FSM-applied index to
// reach the committed index before failing loud with ErrLinearizableTimeout. On a
// healthy (single-node or lightly-loaded) leader the catch-up is satisfied in
// microseconds; this ceiling only bites under extreme apply backlog.
const linearizableReadTimeout = 5 * time.Second

// ErrLinearizableTimeout is returned by verifyLeaderAndCatchUp when the FSM did
// not catch up to the committed index within the deadline. It is fail-loud: a
// linearizable read NEVER silently downgrades to a stale serve.
var ErrLinearizableTimeout = errors.New("shard: linearizable read timed out waiting for FSM catch-up")

// verifyLeaderAndCatchUp is the readIndex barrier for a linearizable read. It
// confirms current leadership, then ensures the FSM has applied every command
// committed as of confirmed leadership before the caller serves from local state.
//
// Mechanism (correctness-critical, cheap on the hot path):
//  1. VerifyLeader — a cheap quorum heartbeat confirming this node is STILL the
//     leader. A node that lost leadership returns NotLeaderError and NEVER serves.
//  2. ci := CommitIndex (captured AFTER the verify, so it reflects everything
//     committed as of confirmed leadership).
//  3. FAST PATH: if the FSM has DIRECTLY applied a command at or beyond ci, then by
//     the single-goroutine in-order apply every command <= ci is already in local
//     state — serve immediately, no log write. (The common case under load.)
//  4. SLOW PATH: the FSM's command frontier is behind ci. This is NORMAL on an idle
//     leader (ci points at the election no-op / a config entry that NEVER reaches
//     fsm.Apply, so the command frontier can never reach ci) and also covers a
//     genuinely-lagging FSM. We CANNOT infer "drained" from a stable command
//     frontier — a command <= ci mid-Apply looks identical to a non-command gap, so
//     a "stability fixpoint" can serve before that command applies (a linearizability
//     violation). Instead use raft.Barrier: it commits a no-op (only a leader can,
//     so it re-confirms leadership) and resolves ONLY AFTER the FSM has applied
//     everything before it — every command <= ci, in order. Correct, never early.
//
// fsm.AppliedIndex() is the TRUE FSM-applied index advanced at the END of each
// fsm.Apply (correct in both heap and mmap modes, unlike cache.AppliedIndex() which
// is 0 in heap mode), NOT s.raft.AppliedIndex() (which only reflects what raft
// DISPATCHED to the FSM goroutine and LEADS the applied state).
//
// barrierEnteredHook holds an optional test-only observer fired ONCE each time
// the readIndex barrier is actually entered (i.e. a Linearizable read reached
// verifyLeaderAndCatchUp). Stored atomically like readServedHook; nil in prod →
// zero overhead. Used to PROVE the AnyReplica/LeaderOnly default paths never
// enter the barrier (zero added VerifyLeader cost).
var barrierEnteredHook atomic.Pointer[func()]

// SetBarrierEnteredHook installs (or clears with nil) the readIndex-barrier-entry
// observer. Test-only.
func SetBarrierEnteredHook(fn func()) {
	if fn == nil {
		barrierEnteredHook.Store(nil)
		return
	}
	barrierEnteredHook.Store(&fn)
}

// boundedStalenessServedHook holds an optional test-only observer fired ONCE each
// time a ConsistencyBoundedStaleness read resolves its freshness decision: with
// servedLocal=true when the replica is within bound and serves locally WITHOUT a
// barrier (the offload win), and servedLocal=false when it is out of bound /
// fail-closed and upgrades (taking the Linearizable barrier or returning
// NotLeaderError to route to the leader). Stored atomically like barrierEnteredHook;
// nil in prod ⇒ zero overhead.
var boundedStalenessServedHook atomic.Pointer[func(servedLocal bool)]

// SetBoundedStalenessServedHook installs (or clears with nil) the bounded-staleness
// freshness-decision observer. Test-only.
func SetBoundedStalenessServedHook(fn func(servedLocal bool)) {
	if fn == nil {
		boundedStalenessServedHook.Store(nil)
		return
	}
	boundedStalenessServedHook.Store(&fn)
}

// fireBoundedStalenessServed fires the bounded-staleness decision hook if installed.
// nil-safe and zero-cost when unset.
func fireBoundedStalenessServed(servedLocal bool) {
	if h := boundedStalenessServedHook.Load(); h != nil {
		(*h)(servedLocal)
	}
}

// verifyLeaderAndCatchUp coalesces concurrent Linearizable-read barriers via the
// per-Store readindex coalescer, then runs the actual VerifyLeader+Barrier body
// (verifyLeaderAndCatchUpBody) at most once per coalesced flight. The coalescer
// guarantees every caller receives a frontier captured at or after its own arrival
// (no stale pre-arrival serve); see readindex_coalesce.go for the proof.
func (s *Store) verifyLeaderAndCatchUp(deadline time.Time) error {
	return s.readindex.do(deadline, s.verifyLeaderAndCatchUpBody)
}

// verifyLeaderAndCatchUpBody is the un-coalesced readIndex barrier body: it runs
// once per coalesced flight (the leader of the batch invokes it AFTER the batch is
// closed, so every reader sharing this result arrived before the ci capture below).
// The barrierEnteredHook fires here — once per actual VerifyLeader RTT — so a test
// counting hook fires observes COALESCING (N reads ⇒ fewer than N fires), not one
// fire per caller.
func (s *Store) verifyLeaderAndCatchUpBody(deadline time.Time) error {
	if h := barrierEnteredHook.Load(); h != nil {
		(*h)()
	}
	if err := s.raft.VerifyLeader(); err != nil {
		if errors.Is(err, raft.ErrNotLeader) {
			return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
		}
		return err
	}
	ci := s.raft.CommitIndex()

	// Fast path (common under load, no log write): the FSM has DIRECTLY applied a
	// command at or beyond ci, so by in-order apply every command <= ci is in local
	// state. Safe to serve.
	if s.fsm.AppliedIndex() >= ci {
		return nil
	}

	// Slow path: the FSM command frontier is behind ci (idle no-op/config tail, or a
	// lagging FSM). Barrier commits a no-op and resolves only after the FSM has
	// applied everything before it — every command <= ci, in order. Correct, never
	// early. Its enqueue is bounded by the remaining deadline; if leadership is lost
	// the future resolves with not-leader (mapped) rather than hanging.
	timeout := time.Until(deadline)
	if timeout <= 0 {
		return ErrLinearizableTimeout
	}
	if err := s.raft.Barrier(timeout); err != nil {
		if errors.Is(err, raft.ErrNotLeader) {
			return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
		}
		return ErrLinearizableTimeout // FSM did not catch up in time — fail loud
	}
	return nil
}

// serveBoundedStaleness enforces a ConsistencyBoundedStaleness read's freshness
// bound before the caller serves from local state. The bound is the max raft entries
// the served replica may lag the leader's committed frontier; within bound ⇒ serve
// locally (the offload win, NO barrier), out of bound ⇒ upgrade (catch up on a
// leader, route to leader on a follower). It returns nil to proceed with the local
// serve, or an error (NotLeaderError ⇒ caller routes to the leader; or a barrier
// error) to abort.
//
//   - LEADER: its own CommitIndex IS the frontier (no RTT). lag = CommitIndex -
//     fsm.AppliedIndex() (guarded against underflow). lag<=bound ⇒ serve; else run
//     the Linearizable barrier to catch up, then serve.
//   - FOLLOWER: if no leaderFrontierFn is injected (single-shard / unwired), fall
//     back to verifyLeaderAndCatchUp — a follower's VerifyLeader yields
//     *NotLeaderError, routing the caller to the leader (a safe upgrade). Otherwise
//     fetch the leader's frontier via the (coalesced) RTT; on error fail CLOSED
//     (NotLeaderError ⇒ route to leader — covers the partitioned-follower hole). With
//     the frontier in hand, serve locally iff frontier <= applied+bound (lag within
//     bound, overflow-guarded); else return NotLeaderError (too stale ⇒ leader).
func (s *Store) serveBoundedStaleness(name string, args []byte) error {
	bound, _ := ops.ReadStalenessOf(name, args)
	deadline := time.Now().Add(linearizableReadTimeout)

	if s.IsLeader() {
		ci := s.CommitIndex()
		applied := s.fsm.AppliedIndex()
		var lag uint64
		if ci > applied {
			lag = ci - applied
		}
		if lag <= bound {
			fireBoundedStalenessServed(true)
			return nil
		}
		fireBoundedStalenessServed(false)
		return s.verifyLeaderAndCatchUp(deadline)
	}

	// Follower.
	if s.leaderFrontierFn == nil {
		fireBoundedStalenessServed(false)
		return s.verifyLeaderAndCatchUp(deadline)
	}
	frontier, err := s.leaderFrontierFn(deadline)
	if err != nil {
		// Fail closed: an unreachable / partitioned leader means we cannot prove
		// freshness ⇒ route to the leader rather than serve possibly-stale data.
		fireBoundedStalenessServed(false)
		return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	applied := s.fsm.AppliedIndex()
	// lag = frontier - applied; serve iff lag <= bound, i.e. frontier <= applied+bound.
	// Guard the applied+bound overflow: if it wraps, the bound is effectively
	// unbounded, so any frontier is within bound.
	withinBound := frontier <= applied || applied+bound < applied || frontier <= applied+bound
	if withinBound {
		fireBoundedStalenessServed(true)
		return nil
	}
	fireBoundedStalenessServed(false)
	return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
}

// Call dispatches a registered op by name. Read-only ops execute directly
// against the cache; read-write ops go through Raft.
func (s *Store) Call(name string, args []byte) ([]byte, error) {
	handler, kind, _, ok := s.registry.Lookup(name)
	if !ok {
		return nil, ErrOpNotRegistered
	}
	if kind == ops.OpReadOnly {
		// Cheap op-aware peek of the read_consistency byte. Only a Linearizable
		// read pays for the VerifyLeader round-trip; AnyReplica/LeaderOnly (the
		// default paths) cost just this peek — zero added consensus work.
		if rc, hasRC := ops.ReadConsistencyOf(name, args); hasRC {
			switch rc {
			case ops.ConsistencyLinearizable:
				if err := s.verifyLeaderAndCatchUp(time.Now().Add(linearizableReadTimeout)); err != nil {
					return nil, err
				}
			case ops.ConsistencyBoundedStaleness:
				if err := s.serveBoundedStaleness(name, args); err != nil {
					return nil, err
				}
			}
		}
		if h := readServedHook.Load(); h != nil {
			(*h)(s.IsLeader())
		}
		return handler(s.tx, args)
	}
	if !s.IsLeader() {
		return nil, &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	return s.applyOp(name, args)
}

// applyOp encodes a log entry and submits it via raft.Apply. Returns the
// handler's result and error.
func (s *Store) applyOp(name string, args []byte) ([]byte, error) {
	result, _, err := s.applyOpIndexed(name, args)
	return result, err
}

// applyOpIndexed is applyOp but also returns the committed Raft log index of the
// entry (via raft.ApplyIndexed). The write-consistency barrier uses this to know
// where in the per-shard log this write landed. applyOp delegates here and
// discards the index, so existing callers and behavior are byte-for-byte
// identical.
func (s *Store) applyOpIndexed(name string, args []byte) (result []byte, index uint64, err error) {
	// #4 Phase B / B1: the leader (Raft) / primary (PB) is the SINGLE encode site —
	// both replication modes reach ApplyIndexed here — so stamping once here stamps
	// every replicated write for either mode. When apply stamping is enabled, bake
	// this node's propose-time wall clock into the entry so every replica evaluates
	// the write's expiry against it; when disabled (the default first rollout
	// phase, and always for a bare/single-node store), emit a legacy entry that
	// decodes with stampMs=0 → wall-clock fallback, byte-identical to pre-B1.
	var entry []byte
	if s.cfg.EnableApplyStamp {
		// MONOTONIC CLAMP (#4 Phase B / B3b) — REQUIRED for sweeper-vs-write safety,
		// not an optimization. Take the max of this node's propose-time wall clock
		// and the cache's current logical clock (the largest apply-stamp already
		// folded in), so the stamp we bake NEVER regresses below any prior applied
		// stamp.
		//
		// Why it is required: the replicated sweeper physically removes a key K once
		// K.exp <= lastAppliedStampMs. A committed write W that lands later evaluates
		// K's liveness against W.stampMs. If a stamp could regress (W.stampMs <
		// lastAppliedStampMs) then W could judge K.exp > W.stampMs — i.e. still LIVE —
		// while the sweeper had already physically reclaimed it: the replicas that
		// swept and the write's view diverge. The clamp guarantees
		//     W.stampMs >= lastAppliedStampMs >= exp(any key the sweep removed),
		// so W ALSO sees every swept key as expired and can never resurrect one —
		// removal is provably safe. Across leader failover the invariant is preserved
		// for free: a new leader has applied every committed entry, so its own
		// lastAppliedStampMs already reflects the max before it stamps anything, and
		// the clamp carries that forward. The stamp still tracks real time whenever
		// the wall clock is ahead (the common case); the clamp only ever raises it.
		stampMs := uint64(time.Now().UnixMilli()) //nolint:gosec // positive Unix ms fits uint64
		if applied := s.cache.LastAppliedStampMs(); applied > stampMs {
			stampMs = applied
		}
		entry = EncodeLogEntryStampedPooled(name, args, stampMs)
	} else {
		entry = EncodeLogEntryPooled(name, args)
	}
	resp, idx, err := s.raft.ApplyIndexed(entry, 5*time.Second)
	if err != nil {
		if errors.Is(err, raft.ErrNotLeader) {
			return nil, 0, &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
		}
		return nil, 0, fmt.Errorf("raft apply: %w", err)
	}
	ar, ok := resp.(*ApplyResponse)
	if !ok {
		return nil, 0, fmt.Errorf("raft apply: unexpected response type %T", resp)
	}
	return ar.Result, idx, ar.Err
}

// AppliedIndex returns the minimum applied-index recorded across all cache
// shards. Useful for integration tests and for skipping the snapshot store.
func (s *Store) AppliedIndex() uint64 {
	return s.cache.AppliedIndex()
}

// IsLeader returns true if this node is the Raft leader.
func (s *Store) IsLeader() bool {
	return s.raft.IsLeader()
}

// CommitIndex returns the highest log index known committed (the shard leader's
// committed frontier). It is the freshness target a ConsistencyBoundedStaleness read
// compares against: on the leader, lag = CommitIndex - fsm.AppliedIndex(); off the
// leader, the leader's CommitIndex (fetched via __shard_readindex__) is the frontier
// a follower's applied index must be within `bound` of.
func (s *Store) CommitIndex() uint64 { return s.raft.CommitIndex() }

// VerifyLeader confirms this node is STILL the shard-Raft leader via a quorum
// heartbeat, mapping ErrNotLeader (a follower, or a leader that lost quorum) to a
// *NotLeaderError so the caller re-routes. It is the leader-side guard of the cheap
// __shard_readindex__ frontier ping (no Barrier wait — just verify + report
// CommitIndex), mirroring the barrier body's VerifyLeader handling.
func (s *Store) VerifyLeader() error {
	if err := s.raft.VerifyLeader(); err != nil {
		if errors.Is(err, raft.ErrNotLeader) {
			return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
		}
		return err
	}
	return nil
}

// LeaderAddr returns the current leader's transport address.
func (s *Store) LeaderAddr() string {
	return s.raft.LeaderAddr()
}

// BootstrapGroup forms this shard's Raft group after construction, with servers
// as the initial configuration. Used by the cluster-level shard formation driver
// once the control plane has designated this node as the shard's former.
//
// Raft mode only: it type-asserts the active replicator rather than widening the
// replicator interface, so the primary-backup engine (which has no per-shard Raft
// group to form) is untouched and returns a no-op nil.
func (s *Store) BootstrapGroup(servers []hraft.Server) error {
	rn, ok := s.raft.(*raft.Node)
	if !ok {
		return nil // PB mode: no per-shard Raft group exists to form
	}
	return rn.BootstrapGroup(servers)
}

// AddVoter adds a Raft voter (serverID at raftAddr) to this shard's group. Must
// be called on the leader; returns NotLeaderError otherwise. Used by online
// rebalancing to bring a new owner into a shard (Raft then streams it the
// snapshot + log). serverID is the joining node's per-shard Raft id.
func (s *Store) AddVoter(serverID, raftAddr string) error {
	if !s.raft.IsLeader() {
		return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	return s.raft.AddVoter(serverID, raftAddr, 0, 10*time.Second)
}

// RemoveServer removes a Raft voter from this shard's group. Must be called on
// the leader; returns NotLeaderError otherwise. Used by online rebalancing to
// drop an owner once its replacement has caught up.
func (s *Store) RemoveServer(serverID string) error {
	if !s.raft.IsLeader() {
		return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	return s.raft.RemoveServer(serverID, 0, 10*time.Second)
}

// TransferLeadershipTo hands this shard's Raft leadership to serverID (at
// raftAddr). Must be called on the leader. Online rebalancing uses this to move
// leadership off a node before removing it from the group, avoiding a
// leader-removes-itself transition.
func (s *Store) TransferLeadershipTo(serverID, raftAddr string) error {
	if !s.raft.IsLeader() {
		return &NotLeaderError{LeaderAddr: s.raft.LeaderAddr()}
	}
	return s.raft.LeadershipTransferToServer(serverID, raftAddr)
}

// LastIndex returns this shard's last Raft log index.
func (s *Store) LastIndex() uint64 { return s.raft.LastIndex() }

// RaftAppliedIndex returns the last Raft log index applied to this shard's FSM.
// Distinct from AppliedIndex (which reports the cache's min applied index): this
// reflects raw Raft progress and is op-type-agnostic, so it is the catch-up
// signal online rebalancing polls on a joining replica.
func (s *Store) RaftAppliedIndex() uint64 { return s.raft.AppliedIndex() }

// Stats returns combined cache + raft stats.
func (s *Store) Stats() Stats {
	return Stats{
		Cache: s.cache.Stats(),
		Raft:  s.raft.Stats(),
	}
}
