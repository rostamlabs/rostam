// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// pbReadServeRecorder counts OpReadOnly serves (and how many were leader-serves)
// reported via shard.SetReadServedHook. Mutex-guarded for concurrent serves. It is
// the cluster-package twin of shard's unexported readServeRecorder.
type pbReadServeRecorder struct {
	mu           sync.Mutex
	leaderServes int
	totalServes  int
}

func (r *pbReadServeRecorder) hook(isLeader bool) {
	r.mu.Lock()
	r.totalServes++
	if isLeader {
		r.leaderServes++
	}
	r.mu.Unlock()
}

func (r *pbReadServeRecorder) reset() {
	r.mu.Lock()
	r.leaderServes, r.totalServes = 0, 0
	r.mu.Unlock()
}

func (r *pbReadServeRecorder) counts() (leader, total int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.leaderServes, r.totalServes
}

// TestPBLinearizableRejectsStalePrimary is the PB-mode analogue of the raft
// linearizability proof (shard.TestLinearizableRejectsStaleLeader): the LAST P6
// hardening lane called out in shard/pbisr/DESIGN.md — "a PB-mode linearizable
// stale-primary-read e2e" — before PB (`-replication-mode=pb`) was promoted out of
// experimental.
//
// WHAT IT PROVES. The PB primary LEASE self-fence protects linearizable READS, not
// just writes. On the SAME partitioned old primary, at (nearly) the same instant:
//   - a best-effort LeaderOnly read SERVES from local (stale-capable) state — it
//     only consults IsLeader() (ctrl.Primary == self, still cached-true because the
//     partitioned node never learns the new epoch), never the lease; and
//   - a Linearizable read REFUSES to serve stale — once the primary's lease lapses
//     (its leaseKeeper stops renewing because confirmMetaView reads the CUT meta
//     transport's LastContact), pbReplicator.VerifyLeader (primary && Engine.
//     LeaseValid) returns raft.ErrNotLeader, and shard.Store's readIndex barrier
//     (verifyLeaderAndCatchUp) maps that to a *NotLeaderError BEFORE the handler
//     ever reads local state.
//
// The LeaderOnly-serves vs Linearizable-rejects CONTRAST is the guarantee. The test
// would FAIL if the read path skipped VerifyLeader (a persistent stale serve).
//
// MECHANISM CHOICE (justified, mirrors the raft proof's rationale). The barrier is
// generic (shard/store.go Call → OpReadOnly → ops.ReadConsistencyOf → Linearizable →
// verifyLeaderAndCatchUp → s.raft.VerifyLeader). Only VECTOR read ops carry the
// read_consistency byte that arms it (the KV `get` op is absent from
// wire.ReadConsistencyOf's switch, so a KV read can never express Linearizable) —
// exactly why the raft proof drives vector_search too. So this test creates a small
// vector collection on the partitioned shard's Store and drives the barrier through
// vector_search, tagged ConsistencyLinearizable vs ConsistencyLeaderOnly. Calling
// the shard Store directly (getShard(sh).Call) targets the partitioned old primary's
// replicator (its *pbReplicator) on the real code path, bypassing only cluster-level
// collection routing — the same layering the raft SHARD proof uses.
//
// PARTITION PRIMITIVE. The meta-transport partition injector from
// pb_partition_test.go (newPartitionablePBTestCluster / partitionableStreamLayer):
// cutting exactly the primary's META path leaves its PB data path + client server up,
// so the lease lapses on the primary's own clock while the node is otherwise alive —
// precisely the stale-primary-read hazard.
func TestPBLinearizableRejectsStalePrimary(t *testing.T) {
	const numShards = 1
	const sh = 0

	// Same timings as the proven no-double-primary gate (honor-rule floor
	// = leaseTTL + staleness + renewInterval + failoverTick = 1000+500+300+500 =
	// 2300ms; failoverTimeout strictly exceeds it). These are the fast lease timings
	// that keep this test's runtime to a few seconds: the lease lapses ~ partition +
	// staleness + leaseTTL (≈1.5s) after the cut.
	tc := newPartitionablePBTestCluster(t, 3, numShards, 1, func(c *Config) {
		c.PBAutoFailover = true
		c.MinISR = 1
		c.PBLeaseTTLMs = 1000
		c.PBMetaContactStalenessMs = 500
		c.PBFailoverTimeoutMs = 3000
		c.PBRenewIntervalMs = 300
		c.ShardCfg.RaftHeartbeatMs = 200
	})

	findNodeIdx := func(nodeID string) int {
		for i, p := range tc.peers {
			if p.NodeID == nodeID {
				return i
			}
		}
		return -1
	}

	// A serve recorder over shard.SetReadServedHook: it fires once per OpReadOnly
	// serve with IsLeader(). With numShards=1 and this test as the only reader of
	// shard 0, its counts reflect exactly the reads we drive.
	rec := &pbReadServeRecorder{}
	shard.SetReadServedHook(rec.hook)
	t.Cleanup(func() { shard.SetReadServedHook(nil) })

	// docs collection: 8 points at {i,0,0}; query {8,0,0} is uniquely nearest id=8.
	colCfg := vector.Config{Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}
	query := []float32{8, 0, 0}
	// collection gets a FRESH name per setup attempt (below). A retry after a primary
	// change must not re-create an existing collection: vector_create_collection would
	// return ErrCollectionExists, and the PB Call path rebuilds errors so that sentinel
	// is not matchable by errors.Is — a unique name per attempt sidesteps both, so the
	// loop can never dead-loop into the 45s SKIP while a stable primary is actually
	// available. linArgs/leaderOnlyArgs are built once the winning name is known.
	var collection string

	// --- Establish a stable, leased shard-0 primary AND seed the collection on it.
	// Mirror the no-double-primary gate's SEQUENTIAL two-ack discovery (a probe put,
	// a 100ms gap, a second put with the primary identity unchanged) — light evidence
	// of a stable-enough lease. Then create the collection + upsert the points via the
	// primary's shard Store; each is a full-ISR-committed write, so on return every
	// backup already holds it (needed by the post-failover new-primary-serves check).
	// Re-run the whole setup if the primary changes under load. Fails SAFE: it can only
	// refuse to start (skip), never produce a false pass.
	probeArgs := ops.EncodePutArgs([]byte("probe"), []byte("v"), 0)
	var primaryIdx = -1
	var origPrimary string
	setupDeadline := time.Now().Add(45 * time.Second)
	attempt := 0
	for time.Now().Before(setupDeadline) && primaryIdx < 0 {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		idx := findNodeIdx(primary)
		if idx < 0 {
			t.Fatalf("primary %q not in peer list", primary)
		}
		// Two sequential acks ~100ms apart from the SAME FSM primary.
		if _, err := tc.nodes[idx].Call("put", probeArgs); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		time.Sleep(100 * time.Millisecond)
		if tc.nodes[0].meta.FSM.ShardPrimary(sh) != primary {
			continue
		}
		if _, err := tc.nodes[idx].Call("put", probeArgs); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		store := tc.nodes[idx].getShard(sh)
		if store == nil {
			t.Fatalf("primary node %d hosts no store for shard %d", idx, sh)
		}
		// Seed a FRESH collection + points (full-ISR writes) on this leased primary.
		// A fresh name per attempt means a retry after a primary change never hits
		// ErrCollectionExists (see the note where collection is declared).
		attempt++
		cand := "docs-" + strconv.Itoa(attempt)
		if _, err := store.Call("vector_create_collection", ops.EncodeCreateCollectionArgs(cand, colCfg)); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		seeded := true
		for i := 1; i <= 8; i++ {
			up := ops.EncodeVectorUpsertArgs(cand, uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, nil, vector.SparseVector{})
			if _, err := store.Call("vector_upsert", up); err != nil {
				seeded = false
				break
			}
		}
		if !seeded {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if tc.nodes[0].meta.FSM.ShardPrimary(sh) != primary {
			continue // primary changed during seeding; redo on the new one
		}
		collection = cand
		primaryIdx = idx
		origPrimary = primary
	}
	if primaryIdx < 0 {
		// PRECONDITION not met — NOT an invariant violation (same rationale as the
		// no-double-primary gate): on a CPU-oversubscribed host PB leases can thrash so
		// hard no primary stays leased long enough to seed. SKIP so a loaded host is
		// never a false red.
		t.Skip("SKIP: could not establish an alive, leased PB primary and seed the collection within 45s (host too loaded); stale-primary-read invariant not exercised this run")
	}
	// The winning attempt's collection name is now fixed; build the read args from it.
	linArgs := ops.EncodeVectorSearchArgsOpts(collection, 5, query, vector.Filter{}, ops.ConsistencyLinearizable, 0, 0)
	leaderOnlyArgs := ops.EncodeVectorSearchArgsOpts(collection, 5, query, vector.Filter{}, ops.ConsistencyLeaderOnly, 0, 0)
	oldPrimary := tc.nodes[primaryIdx].getShard(sh)
	eng := tc.nodes[primaryIdx].pbEngines[sh]
	if eng == nil {
		t.Fatalf("primary node %d has no PB engine for shard %d", primaryIdx, sh)
	}
	inj := tc.metaParts[origPrimary]
	if inj == nil {
		t.Fatalf("no meta partition injector for primary %q", origPrimary)
	}
	t.Logf("stable primary=%q idx=%d, collection seeded (full-ISR)", origPrimary, primaryIdx)

	// (pre) HEALTHY: a Linearizable read on the primary serves (lease valid → barrier
	// passes → handler returns the nearest point).
	if !eng.LeaseValid() {
		t.Skip("SKIP: primary lease not valid at healthy pre-check (host too loaded); cannot stage the contrast")
	}
	rec.reset()
	raw, err := oldPrimary.Call("vector_search", linArgs)
	if err != nil {
		t.Fatalf("(pre) healthy Linearizable read on primary errored: %v", err)
	}
	if hits, derr := ops.DecodeVectorSearchResults(raw); derr != nil || len(hits) == 0 || hits[0].ID != 8 {
		t.Fatalf("(pre) healthy Linearizable read = %+v err=%v; want nearest id=8", hits, derr)
	}
	if ls, _ := rec.counts(); ls == 0 {
		t.Fatal("(pre) healthy Linearizable read produced no leader serve")
	}

	// --- CUT the primary's META transport. PB data path + client server stay UP, so
	// the node is alive and still believes it is primary, but its leaseKeeper can no
	// longer confirm its meta view → it stops renewing → the lease lapses.
	partitionAt := time.Now()
	inj.partition()
	t.Logf("partitioned %q meta transport at %s", origPrimary, partitionAt)

	// (a) Within the lease window the partitioned node STILL reports IsLeader()==true
	// (ctrl.Primary == self, cached; it never learns the new epoch while cut). This is
	// the best-effort gap that makes a freshness check necessary.
	if !oldPrimary.IsLeader() {
		t.Fatal("partitioned old primary reports IsLeader()==false immediately — cannot stage " +
			"the lease-window best-effort gap (test-harness timing)")
	}

	// (b) A LeaderOnly read on the partitioned old primary SERVES from local state
	// (best-effort: IsLeader() cached-true ⇒ serve; no lease check, no barrier). The
	// serve hook confirms it served locally AS the (stale) primary. LeaderOnly never
	// runs VerifyLeader, so it cannot detect the partition. Poll briefly while the
	// cached leadership holds.
	servedLeaderOnly := false
	loDeadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(loDeadline) && oldPrimary.IsLeader() {
		rec.reset()
		if _, lerr := oldPrimary.Call("vector_search", leaderOnlyArgs); lerr != nil {
			t.Fatalf("(b) LeaderOnly read on partitioned old primary errored (%v); the best-effort "+
				"path is supposed to serve stale-capably — cannot stage the contrast", lerr)
		}
		if ls, total := rec.counts(); total > 0 && ls > 0 {
			servedLeaderOnly = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !servedLeaderOnly {
		t.Fatal("(b) LeaderOnly read never produced a local leader serve on the partitioned old " +
			"primary — the best-effort stale-capable path was not exercised")
	}

	// (c) THE rejection: a Linearizable read on the SAME partitioned node must NOT
	// serve stale. Once the lease lapses (leaseKeeper stops renewing because
	// confirmMetaView reads the cut meta transport), Engine.LeaseValid() goes false,
	// pbReplicator.VerifyLeader returns raft.ErrNotLeader, and the barrier maps it to a
	// *NotLeaderError BEFORE the handler reads local state. POLL until rejection within
	// a bound tied to the lease: ~ partition + staleness (500ms) + leaseTTL (1000ms),
	// plus renew slack — well under the 6s deadline below.
	var linErr error
	gotNotLeader := false
	rejectDeadline := time.Now().Add(6 * time.Second)
	for time.Now().Before(rejectDeadline) {
		rec.reset()
		_, linErr = oldPrimary.Call("vector_search", linArgs)
		if errors.Is(linErr, shard.ErrNotLeader) {
			gotNotLeader = true
			break // the required lease-lapse rejection — assertions below run on THIS attempt
		}
		// A nil (still serving via a not-yet-lapsed lease) or a transient
		// ErrLinearizableTimeout during the expiry window is NOT yet the proof: the
		// contract is that the lapsed lease maps to NotLeader. Keep polling for it.
		time.Sleep(50 * time.Millisecond)
	}
	sinceCut := time.Since(partitionAt)
	if !gotNotLeader {
		t.Fatalf("(c) CORRECTNESS HOLE: within %s a Linearizable read on the PARTITIONED old primary "+
			"never rejected with shard.ErrNotLeader (last result: %v) — the lease self-fence must gate the "+
			"read via VerifyLeader→NotLeader once the lease lapses, never serve stale (nil) or merely time out.", sinceCut, linErr)
	}
	if ls, _ := rec.counts(); ls != 0 {
		t.Fatalf("(c) the REJECTING Linearizable read produced %d local leader serve(s) despite "+
			"erroring — it must reject BEFORE serving, never read stale local state", ls)
	}
	// Corroborate at the engine: the lease is provably lapsed.
	if eng.LeaseValid() {
		t.Fatalf("(c) old primary Engine.LeaseValid()==true even though the Linearizable read rejected "+
			"(%v) — the rejection must coincide with a lapsed lease", linErr)
	}
	t.Logf("PROOF: at +%s the partitioned old primary — LeaderOnly served locally (stale-capable), "+
		"Linearizable REJECTED with: %v (lease lapsed)", sinceCut, linErr)

	// (c') DURABILITY of the rejection: a partitioned primary can never renew, so the
	// Linearizable read must keep rejecting. Sample a few more times.
	for i := range 5 {
		if _, err := oldPrimary.Call("vector_search", linArgs); err == nil {
			t.Fatalf("(c') Linearizable read on the still-partitioned old primary SERVED again on "+
				"probe %d — the rejection is not durable (lease must stay dead while cut)", i)
		}
		time.Sleep(50 * time.Millisecond)
	}

	// (4) RE-ROUTE: the surviving majority promotes a NEW primary that serves the
	// LATEST committed data for a Linearizable read (barrier passes on its fresh
	// lease). This confirms the Linearizable read is satisfiable elsewhere — it
	// refuses the STALE node, not the data. Find the new primary on a survivor's meta
	// FSM, then poll its Linearizable read until it serves (its lease is granted a beat
	// after promotion).
	survivorIdx := -1
	for i := range tc.nodes {
		if i != primaryIdx {
			survivorIdx = i
			break
		}
	}
	var newPrimary string
	promoteDeadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(promoteDeadline) {
		pr := tc.nodes[survivorIdx].meta.FSM.ShardPrimary(sh)
		if pr != "" && pr != origPrimary {
			newPrimary = pr
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if newPrimary == "" {
		t.Fatalf("no promotion within 20s (orig primary %q, survivor sees primary %q)",
			origPrimary, tc.nodes[survivorIdx].meta.FSM.ShardPrimary(sh))
	}
	newIdx := findNodeIdx(newPrimary)
	if newIdx < 0 || newIdx == primaryIdx {
		t.Fatalf("new primary %q resolves to bad node index %d (P was %d)", newPrimary, newIdx, primaryIdx)
	}
	newStore := tc.nodes[newIdx].getShard(sh)
	if newStore == nil {
		t.Fatalf("new primary node %d hosts no store for shard %d", newIdx, sh)
	}
	t.Logf("failover: %q -> %q at +%s", origPrimary, newPrimary, time.Since(partitionAt))

	var served bool
	serveDeadline := time.Now().Add(15 * time.Second)
	var lastErr error
	for time.Now().Before(serveDeadline) {
		rec.reset()
		raw, err := newStore.Call("vector_search", linArgs)
		if err != nil {
			lastErr = err
			time.Sleep(50 * time.Millisecond)
			continue
		}
		hits, derr := ops.DecodeVectorSearchResults(raw)
		if derr != nil {
			t.Fatalf("decode new-primary results: %v", derr)
		}
		if len(hits) == 0 || hits[0].ID != 8 {
			t.Fatalf("new-primary Linearizable results = %+v, want nearest id=8 (latest committed data)", hits)
		}
		if ls, _ := rec.counts(); ls == 0 {
			t.Fatal("new-primary Linearizable read produced no leader serve")
		}
		served = true
		break
	}
	if !served {
		t.Fatalf("new primary %q never served a Linearizable read within 15s (lastErr=%v)", newPrimary, lastErr)
	}
	t.Logf("PROOF: new primary %q serves the latest committed data (nearest id=8) for a Linearizable "+
		"read — the barrier refused the stale node, not the data", newPrimary)
}
