// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"sort"
	"sync"
	"time"

	hraft "github.com/hashicorp/raft"
)

// Automatic failover DRIVER (decision core).
//
// The meta leader observes each PB shard's primary liveness (a primary proves
// itself by committing periodic lease-renewal beacons; the leader tracks, on its
// OWN monotonic clock, when it last saw a renewal for each shard). When a
// primary goes silent past the failover timeout, the leader promotes a survivor
// from the current ISR — but ONLY from the ISR, because full-ISR commit
// guarantees every ISR member holds every acked write, so promotion loses no
// acknowledged data. A shard whose ISR has no survivor besides the dead primary
// stays UNAVAILABLE rather than risk data loss (consistency over availability).
//
// This file holds the PURE decision (decidePBPromotions), unit-tested in
// isolation. The plumbing — the renewal beacon op, the leader's last-seen
// tracking, and the failover ticker that commits OpSetShardEpoch — is layered on
// top and injects its observations here.

// pbShardLiveness is the meta leader's per-shard input to the failover decision.
type pbShardLiveness struct {
	shardID     int
	epoch       uint64
	primary     string   // the shard's current committed primary
	isr         []string // current in-sync set (includes primary); promotion targets come from here
	lastRenewNs int64    // leader-clock ns of the last observed renewal for (primary, epoch); 0 = never
}

// pbPromotion is a decided failover action: bump the shard to newEpoch with
// newPrimary. The caller commits it via OpSetShardEpoch (which also resets the
// ISR to {newPrimary} until backups re-catch-up).
type pbPromotion struct {
	shardID    int
	newEpoch   uint64
	newPrimary string
}

// decidePBPromotions returns the failovers the meta leader should commit. For
// each shard whose primary has been silent longer than failoverTimeout (leader
// monotonic ns), it promotes an ISR member OTHER than the presumed-dead primary
// — but ONLY one whose HIGH-WATER is verified: ISR membership is necessary but
// NOT sufficient (ISR grow can transiently leave a GAPPED member in the ISR;
// promoting it would lose acked writes). The candidate is chosen by the injected
// `highWater(shardID, candidate)` resolver:
//
//   - A candidate the resolver reports UNREACHABLE (ok=false) is NOT promotable —
//     we cannot verify it holds the committed tail, so we never promote it.
//   - Among reachable candidates, promote the one with the MAXIMUM applied
//     high-water (tie → lowest nodeID). Under full-ISR commit every legitimately
//     in-sync member holds every acked write, so the max-high-water reachable
//     member holds the committed frontier; a gapped member (its contiguous apply
//     stopped before its gap ⇒ a strictly LOWER high-water) is never the max while
//     any caught-up member survives — the single-failure model this protocol
//     tolerates. This makes a lossy election impossible regardless of how a member
//     got into the ISR gapped — the DURABLE backstop for the grow abandon race.
//
// A nil resolver is the LEGACY path (all candidates treated reachable at equal
// high-water ⇒ lowest-nodeID), used only by pure unit tests of the timeout/reset
// logic; the production failover ticker ALWAYS injects a real resolver.
//
// A shard with no reachable survivor gets NO promotion: no member provably holds
// the committed tail, so the shard stays unavailable (never promote a member that
// cannot be verified caught-up — the OH1/no-acked-loss guarantee).
//
// nowNs and failoverTimeout are on the leader's own monotonic clock, so the
// decision needs no cross-machine clock comparison: it measures the elapsed gap
// between renewals the leader itself observed. failoverTimeout MUST exceed the
// primary-lease TTL by a margin so the old primary has provably self-fenced
// (stopped acking) before its replacement is named (the promotion honor rule).
func decidePBPromotions(shards []pbShardLiveness, nowNs, failoverTimeout int64, highWater func(shardID int, candidate string) (uint64, bool)) []pbPromotion {
	var out []pbPromotion
	for _, s := range shards {
		if s.primary == "" {
			continue // no primary to fail over (unseeded shard)
		}
		if nowNs-s.lastRenewNs <= failoverTimeout {
			continue // primary renewed recently — presumed live
		}
		best := ""
		var bestHW uint64
		for _, m := range s.isr {
			if m == s.primary {
				continue // the presumed-dead primary is not a candidate
			}
			hw := uint64(0)
			reachable := true
			if highWater != nil {
				hw, reachable = highWater(s.shardID, m)
			}
			if !reachable {
				continue // unverifiable high-water ⇒ NOT promotable (the gate)
			}
			if best == "" || hw > bestHW || (hw == bestHW && m < best) {
				best, bestHW = m, hw
			}
		}
		if best == "" {
			continue // no reachable/verifiable ISR survivor — stay down
		}
		out = append(out, pbPromotion{
			shardID:    s.shardID,
			newEpoch:   s.epoch + 1,
			newPrimary: best,
		})
	}
	return out
}

// pbFailoverTracker is the meta leader's per-shard liveness MEMORY: the leader-
// local, monotonic-clock timestamp of the last beacon it observed for each shard's
// CURRENT (primary, epoch). It is Node-owned and NEVER replicated (not in the
// FSM): liveness is a leader-local judgement on the leader's own clock, so it
// needs no cross-node clock comparison and no consensus. Every node owns one; only
// the current meta leader's failover ticker consumes it. observeRenew is called
// from MetaFSM.Apply (the meta-Raft goroutine, via the leaseRenewObserver leaf
// callback); effectiveLastRenew/onBecomeLeader are called from the ticker
// goroutine — hence the mutex.
type pbFailoverTracker struct {
	now func() int64 // monotonic-ns clock (the shared n.pbNow)

	mu            sync.Mutex
	lastRenewNs   map[int]int64  // shardID -> leader-clock ns of the last observed beacon
	lastRenewEp   map[int]uint64 // shardID -> the epoch that beacon renewed (paired with lastRenewNs)
	leaderSinceNs int64          // leader-clock ns this node last became meta leader (the election floor)
}

// newPBFailoverTracker builds a tracker over the shared monotonic clock now (the
// same n.pbNow every engine and the lease-keeper use, so observeRenew stamps and
// the ticker's elapsed-gap math line up on one time source).
func newPBFailoverTracker(now func() int64) *pbFailoverTracker {
	return &pbFailoverTracker{
		now:         now,
		lastRenewNs: make(map[int]int64),
		lastRenewEp: make(map[int]uint64),
	}
}

// observeRenew records that a valid beacon for (shard, epoch) was just applied,
// stamping the leader's own clock. Called from MetaFSM.Apply as a LEAF callback
// (only this mutex is taken; it never re-enters the FSM).
func (t *pbFailoverTracker) observeRenew(shard int, epoch uint64) {
	n := t.now()
	t.mu.Lock()
	t.lastRenewNs[shard] = n
	t.lastRenewEp[shard] = epoch
	t.mu.Unlock()
}

// onBecomeLeader stamps the election floor: the leader-clock instant this node
// (re)acquired meta leadership. Called on the ticker's rising false->leader edge.
// This is RESET #1 (see effectiveLastRenew's election-floor branch).
func (t *pbFailoverTracker) onBecomeLeader() {
	n := t.now()
	t.mu.Lock()
	t.leaderSinceNs = n
	t.mu.Unlock()
}

// effectiveLastRenew returns the leader-clock ns the failover ticker should treat
// as shard's last proof-of-life for its CURRENT committed epoch curEpoch. It
// encodes BOTH correctness resets that keep automatic failover from churning:
//
//   - RESET #2, EPOCH-ADVANCE (the `curEpoch > lastRenewEp` branch): the committed
//     epoch is newer than any beacon we have stamped ⇒ a promotion (or bootstrap)
//     just advanced this shard and the NEW primary Q has not beaconed yet. Report
//     "fresh, right now" so the ticker does NOT immediately re-promote E+2 in the
//     window before Q's first beacon lands. WITHOUT it, every promotion cascades:
//     commit E+1, next tick still sees the stale (P,E) stamp as silent, promote
//     E+2, forever.
//
//   - RESET #1, ELECTION FLOOR (the `max(lastRenewNs, leaderSinceNs)` branch): a
//     node that just (re)won meta leadership must not treat pre-leadership silence
//     as failover evidence. A brand-new leader has no beacon history at all
//     (lastRenewNs == 0 ⇒ now-0 > timeout ⇒ it would mass-promote every shard the
//     instant it wins); a RE-elected leader may hold ANCIENT stamps from a prior
//     term (equally spurious). Flooring the effective last-renewal at leaderSinceNs
//     makes a fresh leader wait a full failoverTimeout of ACTUAL post-election
//     silence before promoting anything.
//
// The floor DELAYS a genuine promotion (by up to failoverTimeout past each
// leadership acquisition) but does not permanently mask it — PROVIDED meta
// leadership eventually stabilizes for at least one full failoverTimeout. A
// pathological cluster whose meta leadership FLAPS on a sub-failoverTimeout cadence
// would keep re-arming the floor and could starve promotion indefinitely; that is a
// degenerate meta-Raft-availability condition (the whole control plane is unusable
// then), out of scope here, not a masking bug in this reset.
//
// Both reads are on the leader's own monotonic clock — no cross-machine comparison.
func (t *pbFailoverTracker) effectiveLastRenew(shard int, curEpoch uint64) int64 {
	n := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	if curEpoch > t.lastRenewEp[shard] {
		return n // epoch-advance grace: a just-promoted/bootstrapped primary is presumed alive
	}
	last := t.lastRenewNs[shard]
	if t.leaderSinceNs > last {
		last = t.leaderSinceNs // election floor
	}
	return last
}

// observed reports the last beacon the tracker recorded for shard: the leader-clock
// ns it was stamped, the epoch it renewed, and whether any beacon has ever been
// observed for the shard. A locked read accessor (safe under -race), used by the
// failover wiring tests to assert a leader has seen its primaries' renewals.
func (t *pbFailoverTracker) observed(shard int) (int64, uint64, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	ns, ok := t.lastRenewNs[shard]
	return ns, t.lastRenewEp[shard], ok
}

// pbFailoverDecisions is the leader's per-tick failover decision: it reads the
// committed control plane from state, folds in each shard's leader-local
// effective-last-renewal (the two resets above), and returns the promotions
// decidePBPromotions decides. Shards are visited in ascending shardID so the
// result order is deterministic (map iteration is not). nowNs and failoverTimeout
// are on the leader's own monotonic clock. Pure w.r.t. state; reads t under its
// mutex.
func pbFailoverDecisions(state State, t *pbFailoverTracker, nowNs, failoverTimeout int64, highWater func(shardID int, candidate string) (uint64, bool)) []pbPromotion {
	ids := make([]int, 0, len(state.ShardPrimary))
	for shardID := range state.ShardPrimary {
		ids = append(ids, shardID)
	}
	sort.Ints(ids)
	shards := make([]pbShardLiveness, 0, len(ids))
	for _, shardID := range ids {
		epoch := state.ShardEpoch[shardID]
		shards = append(shards, pbShardLiveness{
			shardID:     shardID,
			epoch:       epoch,
			primary:     state.ShardPrimary[shardID],
			isr:         state.ShardISR[shardID],
			lastRenewNs: t.effectiveLastRenew(shardID, epoch),
		})
	}
	return decidePBPromotions(shards, nowNs, failoverTimeout, highWater)
}

// pbFailover is the ALWAYS-ON failover ticker (started only when PBAutoFailover is
// on — when off it never runs, so the meta log gets no promotion bumps and the
// static cluster stays byte-identical). It self-detects meta leadership every tick
// via meta.Raft.State() and tracks a goroutine-local rising edge; there is NO
// meta-leadership watcher (no NotifyCh/LeaderCh) in this codebase, so a
// continuously-running self-checking ticker is the correct shape. It commits a
// promotion only while it is the meta leader.
//
// HONOR RULE (the no-double-primary / no-acked-loss guarantee, enforced by
// construction — see Config.Validate's assertion
// failoverTimeout > pbLeaseTTL + metaContactStaleness + renewInterval + tick):
//
//	Consider a meta-network partition at time τ that isolates the old primary P
//	(the PB data net may still be up). P renews its engine lease only while
//	confirmMetaView passes — on a follower that means LastContact < metaContactStaleness.
//	So after τ, P's LAST successful renewal is ≤ τ + metaContactStaleness, and its
//	engine lease (validity pbLeaseTTL) therefore expires by
//	τ + metaContactStaleness + pbLeaseTTL, after which every Propose self-fences
//	(ErrLeaseExpired) — P acks nothing more (this fence is exercised by
//	TestOH1StalePrimarySelfFencesOnLeaseExpiry).
//
//	P commits no beacon after τ (it cannot reach the meta quorum). The ticker
//	promotes when now − lastRenewNs > failoverTimeout, where lastRenewNs is the
//	leader's LAST OBSERVED beacon for P. Crucially that beacon can already be a full
//	renewInterval old at τ (P beacons only every renewInterval), so the EARLIEST
//	possible promotion is τ + failoverTimeout − renewInterval — NOT τ + failoverTimeout.
//	Safety (promotion-time ≥ fence-time) therefore requires
//	failoverTimeout − renewInterval ≥ pbLeaseTTL + metaContactStaleness, i.e.
//	failoverTimeout ≥ pbLeaseTTL + metaContactStaleness + renewInterval — exactly the
//	construction assertion (which adds a one-tick + margin). Under it, Q is named
//	strictly AFTER P has provably self-fenced ⇒ never two live primaries. And because
//	every pre-τ acked write is on every ISR member (full-ISR commit) and Q is chosen
//	from that ISR, Q holds every acked write ⇒ no acked-loss, no double-ack.
//
//	COVERAGE: this partition property is backed by the construction assertion + the
//	self-fence unit test + the crash-stop no-acked-loss gate (TestPBFailoverNoAckedLoss)
//	AND the full network-partition e2e (TestPBFailoverPartitionNoDoublePrimary,
//	cluster/pb_partition_test.go: isolates one node's meta path while it keeps taking
//	writes, and asserts its last ack precedes the epoch bump). PBAutoFailover is now
//	default-on. Remaining hardening (not a correctness hole): a PB-mode linearizable
//	stale-primary-READ e2e mirroring shard/linearizable_partition_test.go.
//
//	DETECTION LATENCY: when the dead primary was ALSO the meta leader, the surviving
//	nodes must first elect a new meta leader, whose election-floor reset then requires
//	a FULL failoverTimeout of post-election silence before promoting. Worst-case
//	detection is therefore election_time + failoverTimeout, not bare failoverTimeout.
type pbFailover struct {
	meta            *MetaRaft
	fsm             *MetaFSM
	tracker         *pbFailoverTracker
	now             func() int64 // leader-clock monotonic ns (shared n.pbNow)
	failoverTimeout int64        // ns; the silent-primary promotion threshold
	interval        time.Duration
	applyTimeout    time.Duration
	// highWater verifies a promotion candidate's applied high-water over the PB
	// transport (the DURABLE grow-abandon backstop — see decidePBPromotions). ok is
	// false when the candidate is unreachable/unverifiable (⇒ not promotable). nil
	// only in unit tests that drive the pure decision directly.
	highWater func(shardID int, candidate string) (uint64, bool)

	mu      sync.Mutex
	started bool
	done    chan struct{}
	stopped chan struct{}
}

// newPBFailover constructs the failover ticker. now/failoverTimeout are on the
// leader's own monotonic clock (n.pbNow), so the decision needs no cross-node
// clock comparison.
func newPBFailover(meta *MetaRaft, fsm *MetaFSM, tracker *pbFailoverTracker, now func() int64, failoverTimeout int64, interval, applyTimeout time.Duration, highWater func(shardID int, candidate string) (uint64, bool)) *pbFailover {
	return &pbFailover{
		meta:            meta,
		fsm:             fsm,
		tracker:         tracker,
		now:             now,
		failoverTimeout: failoverTimeout,
		interval:        interval,
		applyTimeout:    applyTimeout,
		highWater:       highWater,
	}
}

// start spawns the ticker goroutine (no-op if already started). Mirrors
// leaseKeeper/pbBeacon lifecycle.
func (f *pbFailover) start() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.started {
		return
	}
	f.started = true
	f.done = make(chan struct{})
	f.stopped = make(chan struct{})
	go f.run()
}

// stop signals the ticker to exit and blocks until it has. Safe without a prior
// start and safe to call repeatedly.
func (f *pbFailover) stop() {
	f.mu.Lock()
	if !f.started {
		f.mu.Unlock()
		return
	}
	done := f.done
	stopped := f.stopped
	f.started = false
	f.mu.Unlock()

	select {
	case <-done:
	default:
		close(done)
	}
	<-stopped
}

func (f *pbFailover) run() {
	defer close(f.stopped)
	ticker := time.NewTicker(f.interval)
	defer ticker.Stop()
	// wasLeader is the goroutine-local rising-edge detector: there is no leadership
	// watcher, so we detect "just became leader" by comparing this tick's leadership
	// to the previous tick's.
	wasLeader := false
	for {
		select {
		case <-f.done:
			return
		case <-ticker.C:
			f.tick(&wasLeader)
		}
	}
}

// tick runs one failover evaluation. It is separated from run() so the leadership
// edge + reset logic is exercised deterministically; the pure decision it drives
// (pbFailoverDecisions + the two tracker resets) is unit-tested directly with an
// injectable clock.
func (f *pbFailover) tick(wasLeader *bool) {
	isLeader := f.meta.Raft.State() == hraft.Leader
	// RISING EDGE (false->leader): stamp the election floor (tracker reset #1) so a
	// freshly-(re)elected leader does NOT treat pre-leadership silence as failover
	// evidence and mass-promote the cluster the instant it wins the meta election.
	if isLeader && !*wasLeader {
		f.tracker.onBecomeLeader()
	}
	*wasLeader = isLeader
	if !isLeader {
		return // only the meta leader promotes
	}
	promos := pbFailoverDecisions(f.fsm.State(), f.tracker, f.now(), f.failoverTimeout, f.highWater)
	for _, p := range promos {
		// OpSetShardEpoch is monotonic + idempotent: a duplicate E+1 (two ticks fire
		// before the first commit replicates back into our local FSM) is a benign
		// no-op. After the commit lands, our FSM's ShardEpoch becomes E+1 while the
		// tracker's lastRenewEp stays E (the old primary's), so effectiveLastRenew's
		// epoch-advance grace (reset #2) reports the shard fresh until the NEW primary
		// Q beacons — preventing a re-promotion cascade (E+1, E+2, …). Best-effort:
		// an apply error (we just lost leadership, a transient) is retried next tick.
		_ = f.meta.ApplySetShardEpoch(p.shardID, p.newEpoch, p.newPrimary, f.applyTimeout) //nolint:errcheck,gosec // idempotent; retried next tick
	}
}
