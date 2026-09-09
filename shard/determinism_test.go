// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// These tests prove #4 Phase B (B1 + B3a): a cluster-replicated shard's COMMITTED
// state is byte-identical across replicas whose wall clocks differ, and stays so
// under TTL traffic and snapshot install. The canonical per-replica fingerprint
// is serializeSnapshot bytes taken with a FIXED clock pinned (so the Iterate
// expiry filter is identical on every replica and cannot itself introduce a
// difference); it captures the stored (key,value,exp) set. The B1 case is written
// as a before/after: the SAME workload applied through UNSTAMPED (legacy /
// wall-clock) entries DIVERGES under skew, while the STAMPED path CONVERGES.

// detReplica is one simulated replica: an fsm over its own cache, plus a mutable
// injected wall clock (clock) the test skews and pins. cfg is retained so a
// PERSISTENT replica can be closed and reopened over the same DataDir (restart).
type detReplica struct {
	f     *fsm
	c     *cache.Cache
	clock *atomic.Uint64
	cfg   cache.Config
}

// newDetReplica builds a replica whose wall clock starts at clock0 (ms). When
// replicated is true the cache runs in #4 Phase B mode (sweeper off, wall-clock
// lazy removal suppressed); false is the single-node/Direct control (sweeper on).
func newDetReplica(t *testing.T, clock0 uint64, replicated bool, sweepMs int) *detReplica {
	t.Helper()
	clk := &atomic.Uint64{}
	clk.Store(clock0)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.Replicated = replicated
	cc.TTLSweepIntervalMs = sweepMs
	cc.NowFn = func() uint64 { return clk.Load() }
	c, err := cache.New(cc)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	return &detReplica{f: newFSM(c, reg, false, nil, nil), c: c, clock: clk, cfg: cc}
}

// newPersistentDetReplica is newDetReplica on the PRODUCTION persistent replicated
// profile: an mmap-backed DataDir, PolicyRejectWrites (what replication forces via
// B2), Replicated, and a page budget small enough that an ordinary workload crosses
// page boundaries. This is the configuration in which a warm restart — not a crash,
// not a snapshot install, just a graceful stop and start — re-derives the index from
// page bytes.
func newPersistentDetReplica(t *testing.T, dir string, clock0 uint64) *detReplica {
	t.Helper()
	clk := &atomic.Uint64{}
	clk.Store(clock0)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.Replicated = true
	cc.AtCapPolicy = cache.PolicyRejectWrites
	cc.TTLSweepIntervalMs = 0
	cc.PageSize = 1 << 20
	cc.MaxMemoryPerShard = 8 << 20 // 8 pages
	cc.DataDir = dir
	cc.NowFn = func() uint64 { return clk.Load() }
	r := &detReplica{clock: clk, cfg: cc}
	r.open(t)
	t.Cleanup(func() { _ = r.c.Close() })
	return r
}

// open (re)builds the replica's cache and fsm from r.cfg.
func (r *detReplica) open(t *testing.T) {
	t.Helper()
	c, err := cache.New(r.cfg)
	if err != nil {
		t.Fatalf("open cache: %v", err)
	}
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	r.c, r.f = c, newFSM(c, reg, false, nil, nil)
}

// restart closes the replica and reopens it over the SAME DataDir, replaying
// nothing. That is the whole point: a gracefully restarted node is fully caught up,
// so the fsm's warm-skip means no committed entry below the applied index is ever
// re-applied and no peer ever sends it an InstallSnapshot. Whatever the rebuild
// derives from the page bytes IS this node's committed state, permanently.
func (r *detReplica) restart(t *testing.T) {
	t.Helper()
	if err := r.c.Close(); err != nil {
		t.Fatalf("close for restart: %v", err)
	}
	r.open(t)
}

// apply runs one entry through the replica's fsm at the given log index and fails
// on any apply error.
func (r *detReplica) apply(t *testing.T, index uint64, entry []byte) {
	t.Helper()
	res := r.f.Apply(&hraft.Log{Index: index, Type: hraft.LogCommand, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok {
		t.Fatalf("Apply returned %T, want *ApplyResponse", res)
	}
	if resp.Err != nil {
		t.Fatalf("apply index %d: %v", index, resp.Err)
	}
}

// fingerprint pins a FIXED low clock (so nothing is filtered as expired) and
// returns the serializeSnapshot bytes of the stored (key,value,exp) set. The
// applied index is forced to 0 so the fingerprint reflects state only, never the
// trailer's index. The replica's own clock is restored afterward.
func (r *detReplica) fingerprint(t *testing.T) []byte {
	t.Helper()
	saved := r.clock.Load()
	r.clock.Store(1) // isExpired(exp,1) is false for every real exp → include all
	defer r.clock.Store(saved)
	data, err := serializeSnapshot(r.c, nil, 0, nil)
	if err != nil {
		t.Fatalf("serializeSnapshot: %v", err)
	}
	return data
}

// ttlWorkload is a fixed interleaving of incr/expire/put on TTL'd keys. Each op
// is encoded with the given per-op stamp via encode; passing EncodeLogEntry
// (stamp-less) reproduces today's wall-clock apply, EncodeLogEntryStamped(...,
// stamp) reproduces B1. It returns the entries in order.
func ttlWorkload(stamp uint64, stamped bool) [][]byte {
	enc := func(name string, args []byte) []byte {
		if stamped {
			return EncodeLogEntryStamped(name, args, stamp)
		}
		return EncodeLogEntry(name, args)
	}
	return [][]byte{
		enc("put", ops.EncodePutArgs([]byte("k1"), []byte("v1"), time.Second)),
		enc("put", ops.EncodePutArgs([]byte("k2"), []byte("v2"), 2*time.Second)),
		enc("incr", ops.EncodeIncrArgs([]byte("counter"), 5)),
		enc("expire", ops.EncodeExpireArgs([]byte("k1"), 5*time.Second)),
		enc("put", ops.EncodePutArgs([]byte("k3"), []byte("v3"), 0)), // no expiry
		enc("incr", ops.EncodeIncrArgs([]byte("counter"), 3)),
		enc("put", ops.EncodePutArgs([]byte("k2"), []byte("v2b"), 3*time.Second)), // overwrite w/ new ttl
	}
}

func applyWorkload(t *testing.T, r *detReplica, entries [][]byte) {
	t.Helper()
	for i, e := range entries {
		r.apply(t, uint64(i+1), e)
	}
}

// TestB1DeterministicApplyStamp is the core before/after: with per-replica wall
// clock skew, the UNSTAMPED (today's wall-clock) apply diverges the committed
// state, while the STAMPED (B1) apply keeps it byte-identical.
func TestB1DeterministicApplyStamp(t *testing.T) {
	// Three replicas at very different wall clocks (skew is seconds apart, which is
	// what makes a TTL'd write's absolute expiry differ on the wall-clock path).
	base := uint64(time.Now().UnixMilli())
	clocks := []uint64{base, base + 30_000, base - 20_000}

	// Leader stamp: one clock, baked into every stamped entry, identical on all
	// replicas regardless of their own wall clock.
	stamp := base + 7

	// --- STAMPED (B1): must converge. ---
	stampedFPs := make([][]byte, len(clocks))
	for i, ck := range clocks {
		r := newDetReplica(t, ck, true, 0)
		applyWorkload(t, r, ttlWorkload(stamp, true))
		stampedFPs[i] = r.fingerprint(t)
	}
	for i := 1; i < len(stampedFPs); i++ {
		if !bytes.Equal(stampedFPs[0], stampedFPs[i]) {
			t.Fatalf("B1 stamped: replica %d fingerprint differs from replica 0 — committed state diverged despite stamping", i)
		}
	}

	// --- UNSTAMPED (today's wall-clock apply): must diverge under skew. ---
	// This is the bug B1 fixes; the assertion documents that the skew is real and
	// the stamped result above is not a vacuous pass.
	unstampedFPs := make([][]byte, len(clocks))
	for i, ck := range clocks {
		r := newDetReplica(t, ck, true, 0)
		applyWorkload(t, r, ttlWorkload(stamp, false))
		unstampedFPs[i] = r.fingerprint(t)
	}
	allEqual := true
	for i := 1; i < len(unstampedFPs); i++ {
		if !bytes.Equal(unstampedFPs[0], unstampedFPs[i]) {
			allEqual = false
			break
		}
	}
	if allEqual {
		t.Fatal("B1 control: UNSTAMPED apply under wall-clock skew produced identical fingerprints — " +
			"the divergence repro is not exercising skew, so the stamped pass proves nothing")
	}
}

// TestB1StampZeroNoWallClockFallback closes the stamp==0 divergence footgun: a
// STAMPED entry whose leader clock is 0 must take the deterministic At-path (every
// replica uses 0), NOT fall back to each node's wall clock. Two replicated
// replicas at wildly different wall clocks apply the same stamped-0 put; their
// committed state must be identical, and the stored absolute expiry must equal the
// TTL (0 + ttl) — a small deterministic value, not ~now.
func TestB1StampZeroNoWallClockFallback(t *testing.T) {
	base := uint64(time.Now().UnixMilli())
	// Stamped entry with stampMs == 0 and a 1000ms TTL: PutAt(...,0) ⇒ exp = 1000.
	entry := EncodeLogEntryStamped("put", ops.EncodePutArgs([]byte("z"), []byte("v"), time.Second), 0)

	rA := newDetReplica(t, base+10_000, true, 0)
	rB := newDetReplica(t, base+900_000, true, 0)
	rA.apply(t, 1, entry)
	rB.apply(t, 1, entry)

	if fpA, fpB := rA.fingerprint(t), rB.fingerprint(t); !bytes.Equal(fpA, fpB) {
		t.Fatal("stamp-0 entry diverged across wall-clock-skewed replicas — it fell back to the wall clock")
	}
	// Prove the expiry is the deterministic 0+ttl=1000, independent of the ~1.7e12
	// wall clock: a read at a stamped clock of 500 (< 1000) hits, at 1500 (> 1000)
	// misses. If it had used the wall clock, exp would be ~base+1000 and BOTH reads
	// would hit (1500 << base+1000).
	if _, err := rA.c.GetAt([]byte("z"), 500); err != nil {
		t.Fatalf("stamp-0 read at t=500 (< exp): err=%v, want hit (exp must be 0+ttl=1000)", err)
	}
	if _, err := rA.c.GetAt([]byte("z"), 1500); err != cache.ErrNotFound {
		t.Fatalf("stamp-0 read at t=1500 (> exp): err=%v, want ErrNotFound (exp must be 0+ttl=1000, not wall-clock)", err)
	}
}

// TestB1LegacyEntryStampZero proves the stamp-disabled (legacy) path still
// functions: unstamped entries decode with stampMs=0 and apply correctly, so a
// single deployment with stamping off (the first rollout phase) is fully
// operational. State correctness (not cross-replica identity) is the point here.
func TestB1LegacyEntryStampZero(t *testing.T) {
	base := uint64(time.Now().UnixMilli())
	r := newDetReplica(t, base, false, 0)
	applyWorkload(t, r, ttlWorkload(0, false))

	if v, err := r.c.Get([]byte("k3")); err != nil || !bytes.Equal(v, []byte("v3")) {
		t.Fatalf("legacy apply k3 = %q, err=%v; want v3", v, err)
	}
	if v, err := r.c.Get([]byte("k2")); err != nil || !bytes.Equal(v, []byte("v2b")) {
		t.Fatalf("legacy apply k2 = %q, err=%v; want v2b", v, err)
	}
	got, err := r.c.Get([]byte("counter"))
	if err != nil {
		t.Fatalf("legacy apply counter: %v", err)
	}
	// counter = incr(+5) then incr(+3) = 8, stored as an 8-byte big-endian value.
	if v := beI64(got); v != 8 {
		t.Fatalf("legacy apply counter = %d, want 8", v)
	}
}

// TestB3aReplicatedNoWallClockRemoval proves that on a replicated shard a
// logically-expired key is FILTERED on a client read (miss) but NOT physically
// removed, so it survives in the committed state (fingerprint) — and two replicas
// at different wall clocks keep identical fingerprints. The single-node control
// (sweeper on, non-replicated) instead physically reclaims the expired key.
func TestB3aReplicatedNoWallClockRemoval(t *testing.T) {
	base := uint64(time.Now().UnixMilli())
	stamp := base

	// Two replicated replicas at different wall clocks, both past the key's expiry.
	// The key is written via a STAMPED apply (exp = stamp + 1s, identical on both).
	entry := EncodeLogEntryStamped("put", ops.EncodePutArgs([]byte("ttlkey"), []byte("v"), time.Second), stamp)

	rA := newDetReplica(t, base+10_000, true, 1000) // sweepMs>0, but replicated ⇒ sweeper OFF
	rB := newDetReplica(t, base+90_000, true, 1000)
	rA.apply(t, 1, entry)
	rB.apply(t, 1, entry)

	// Both wall clocks are well past exp (stamp+1000): a client Get sees a miss...
	if _, err := rA.c.Get([]byte("ttlkey")); err != cache.ErrNotFound {
		t.Fatalf("replicated client Get of expired key: err=%v, want ErrNotFound (filtered)", err)
	}
	// ...but the key is NOT physically removed (no lazy drop, no sweeper): the
	// fingerprint still contains it, and both replicas agree.
	fpA := rA.fingerprint(t)
	fpB := rB.fingerprint(t)
	if !bytes.Equal(fpA, fpB) {
		t.Fatal("B3a: replicated replicas at different wall clocks diverged — wall-clock physical removal leaked")
	}
	if rA.c.Stats().Expirations != 0 {
		t.Fatalf("B3a: replicated shard recorded %d expirations — physical removal was not suppressed", rA.c.Stats().Expirations)
	}

	// Single-node control: sweeper ON, non-replicated. The same key written via the
	// wall-clock client Put is physically reclaimed once the clock passes exp.
	ctrl := newDetReplica(t, base, false, 20)
	if err := ctrl.c.Put([]byte("ttlkey"), []byte("v"), time.Second); err != nil {
		t.Fatal(err)
	}
	ctrl.clock.Store(base + 10_000) // advance the control's wall clock past exp
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if ctrl.c.Stats().Expirations > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if ctrl.c.Stats().Expirations == 0 {
		t.Fatal("B3a control: single-node sweeper did not reclaim the expired key — control did not exercise sweeping")
	}
}

// TestB3aApplyPathTombstonesDeterministically proves the apply path (GetAt,
// stamped) STILL removes an expired key deterministically: an expire whose new
// TTL is already in the past for the stamped clock, or a committed read of an
// already-expired key, tombstones identically on every replica. Here a stamped
// incr on a key whose prior value has expired (relative to the stamp) treats it
// as absent (starts from 0) identically regardless of wall clock.
func TestB3aApplyPathTombstonesDeterministically(t *testing.T) {
	base := uint64(time.Now().UnixMilli())
	// put k with 1s ttl stamped at S; then incr k at stamp S+2000 (2s later) — the
	// stored value has expired relative to the incr's stamp, so incr must see a
	// miss (start at 0) on EVERY replica, not read the stale value.
	putEntry := EncodeLogEntryStamped("put", ops.EncodePutArgs([]byte("n"), i64(1), time.Second), base)
	// incr on an 8-byte key value; if the prior put is (wrongly) still visible incr
	// would read 1 and yield 6; if correctly expired it reads 0 and yields 5.
	incrEntry := EncodeLogEntryStamped("incr", ops.EncodeIncrArgs([]byte("n"), 5), base+2000)

	for _, ck := range []uint64{base, base + 40_000} {
		r := newDetReplica(t, ck, true, 0)
		r.apply(t, 1, putEntry)
		r.apply(t, 2, incrEntry)
		got, err := r.c.GetAt([]byte("n"), base+2000)
		if err != nil {
			t.Fatalf("clock %d: get n: %v", ck, err)
		}
		if v := beI64(got); v != 5 {
			t.Fatalf("clock %d: incr after stamped expiry = %d, want 5 (prior value must be expired vs the stamp on every replica)", ck, v)
		}
	}
}

// TestSnapshotInstallDeterministic proves the restore fix: installing the SAME
// snapshot on two followers at DIFFERENT injected wall times yields byte-identical
// post-install state, because PutAbs installs the recorded absolute expiry
// verbatim and the wall-clock "skip expired on restore" was removed.
func TestSnapshotInstallDeterministic(t *testing.T) {
	base := uint64(time.Now().UnixMilli())

	// Source state, captured at wall clock `base`: no-expiry, far-future, and a
	// "soon" entry whose absolute expiry (base+100s) is in the FUTURE at capture
	// (so serializeSnapshot's Iterate includes it) but in the PAST for a follower
	// that installs later. The old restore dropped any entry with expiry <= the
	// INSTALLER's wall clock, so a late follower would silently omit "soon" and
	// diverge; the fixed restore installs the absolute expiry verbatim (PutAbs) and
	// keeps it on every follower.
	src := newDetReplica(t, base, true, 0)
	if err := src.c.PutAbs([]byte("noexp"), []byte("a"), 0); err != nil {
		t.Fatal(err)
	}
	if err := src.c.PutAbs([]byte("future"), []byte("b"), base+1_000_000); err != nil {
		t.Fatal(err)
	}
	if err := src.c.PutAbs([]byte("soon"), []byte("c"), base+100_000); err != nil {
		t.Fatal(err)
	}
	snap, err := serializeSnapshot(src.c, nil, 42, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Two followers install the same bytes at wall clocks straddling "soon"'s
	// expiry: one BEFORE (base+5s), one well AFTER (base+500s).
	install := func(clock0 uint64) []byte {
		r := newDetReplica(t, clock0, true, 0)
		if _, rErr := restoreSnapshot(r.c, nil, nil, io.NopCloser(bytes.NewReader(snap))); rErr != nil {
			t.Fatalf("restore: %v", rErr)
		}
		return r.fingerprint(t)
	}
	fp1 := install(base + 5_000)
	fp2 := install(base + 500_000) // past "soon"'s expiry — old code would drop it here
	if !bytes.Equal(fp1, fp2) {
		t.Fatal("snapshot install: followers at different wall clocks produced different state — restore is not deterministic")
	}

	// The entry the late follower would have dropped survived with its absolute
	// expiry installed verbatim: install at the late clock, then pin a clock before
	// "soon"'s expiry and read it back.
	r := newDetReplica(t, base+500_000, true, 0)
	if _, rErr := restoreSnapshot(r.c, nil, nil, io.NopCloser(bytes.NewReader(snap))); rErr != nil {
		t.Fatal(rErr)
	}
	r.clock.Store(base + 50_000) // before the "soon" entry's expiry (base+100s)
	if v, gErr := r.c.Get([]byte("soon")); gErr != nil || !bytes.Equal(v, []byte("c")) {
		t.Fatalf("restored entry = %q err=%v; want it installed verbatim (value c), not dropped by the installer's wall clock", v, gErr)
	}
}

// canonicalState returns the replica's committed state as a SORTED, fully
// self-describing (key, expiry, value) listing.
//
// It is used instead of fingerprint() for the warm-restart comparison because
// fingerprint hands back raw serializeSnapshot bytes, whose ORDER is the shard's
// index-table slot order — a physical property of insertion history. A restarted
// replica rebuilds its index from page bytes and so lays the same keys out in a
// different slot order, which changes those bytes without changing a single
// answer the node would give. Physical layout is node-local and already differs
// freely between replicas (different restart histories, different snapshot
// installs); what must not differ is the logical key→(value,expiry) map. This
// canonicalizes exactly that and nothing more.
func (r *detReplica) canonicalState(t *testing.T) string {
	t.Helper()
	saved := r.clock.Load()
	r.clock.Store(1) // isExpired(exp,1) is false for every real exp → include all
	defer r.clock.Store(saved)
	var lines []string
	r.c.Iterate(func(k, v []byte, exp uint64) bool {
		lines = append(lines, fmt.Sprintf("%q exp=%d %q", k, exp, v))
		return true
	})
	sort.Strings(lines)
	return strings.Join(lines, "\n")
}

// warmRestartWorkload is a fixed, fully-stamped sequence of committed entries on
// the persistent replicated profile. It is deliberately ORDINARY — no crash, no
// torn write, no clock skew, no capacity pressure — and it exercises the two ways
// page order can disagree with the committed log:
//
//	OVERWRITE ACROSS A PAGE BOUNDARY. The first epoch of ~200 KiB values fills the
//	shard's initial page; the later ones spill into a different page. Overwriting
//	an early key then places its NEWER copy at a LOWER page index than its older
//	copy, and a rebuild that resolves by page order picks the older one (#12A).
//
//	DELETE. Del removes an index slot; if nothing about it is persisted, the
//	rebuild re-indexes the entry straight off the page and the key is back (#12B).
//
// Every entry carries the same leader stamp, so nothing here depends on either
// replica's wall clock — any divergence is physical-layout divergence, which is
// exactly the claim under test.
func warmRestartWorkload(stamp uint64) [][]byte {
	big := func(tag string, n int) []byte {
		v := bytes.Repeat([]byte("v"), 200<<10)
		copy(v, tag)
		return v[:n]
	}
	key := func(i int) []byte { return []byte(fmt.Sprintf("wr-key-%02d", i)) }
	put := func(k, v []byte) []byte {
		return EncodeLogEntryStamped("put", ops.EncodePutArgs(k, v, 0), stamp)
	}
	del := func(k []byte) []byte {
		return EncodeLogEntryStamped("del", ops.EncodeKeyArgs(k), stamp)
	}

	var entries [][]byte
	// Epoch 1: twelve large values — more than one page holds, so the run spans
	// the initial page and at least one other.
	for i := 0; i < 12; i++ {
		entries = append(entries, put(key(i), big(fmt.Sprintf("e1-%02d|", i), 200<<10)))
	}
	// Epoch 2: overwrite the keys written FIRST. Their newer copies land wherever
	// free tail room happens to be, which is not necessarily above their older ones.
	for i := 0; i < 4; i++ {
		entries = append(entries, put(key(i), big(fmt.Sprintf("e2-%02d|", i), 200<<10)))
	}
	// Epoch 3: deletes, including one of the keys just overwritten.
	entries = append(entries, del(key(1)), del(key(7)), del(key(11)))
	// Epoch 4: a small overwrite, which can fit a hole a large entry skipped.
	entries = append(entries, put(key(2), big("e4-02|", 4<<10)))
	return entries
}

// TestWarmRestartDoesNotDivergeReplicas is THE deliverable: proof that the
// warm-restart defects are a REPLICATED CORRECTNESS bug, not a single-node
// curiosity.
//
// Two replicas apply the identical committed, identically-stamped log to identical
// persistent shards. Then ONE of them is restarted — gracefully, the way an
// operator restarts a node to act on the page-occupancy alert whose documented
// remedy is exactly "restart this node". Nothing is replayed and nothing is
// re-installed: a cleanly restarted node is fully caught up, so the fsm's warm-skip
// re-applies no committed entry below the applied index and no peer ever sends it an
// InstallSnapshot. Its post-restart state IS its committed state, forever.
//
// If the fingerprints differ, two replicas that ACKed the same log now serve
// different answers to the same key, with nothing in the system able to notice or
// repair it. It must fail before the fix (a lost overwrite and/or a resurrected
// delete) and pass after.
func TestWarmRestartDoesNotDivergeReplicas(t *testing.T) {
	base := uint64(time.Now().UnixMilli())
	entries := warmRestartWorkload(base)

	rA := newPersistentDetReplica(t, t.TempDir(), base)
	rB := newPersistentDetReplica(t, t.TempDir(), base)
	applyWorkload(t, rA, entries)
	applyWorkload(t, rB, entries)

	before := rA.canonicalState(t)
	if peer := rB.canonicalState(t); peer != before {
		t.Fatal("the two replicas disagreed BEFORE any restart — the harness itself is " +
			"non-deterministic, so nothing below would prove anything")
	}

	// The only event between the two observations: A stops and starts.
	rA.restart(t)

	after := rA.canonicalState(t)
	if after == before {
		return // converged: the restart preserved committed state exactly.
	}
	// Diverged. Report WHICH keys, so the failure names the defect rather than
	// just showing two blobs.
	stateOf := func(r *detReplica) map[string]string {
		out := map[string]string{}
		saved := r.clock.Load()
		r.clock.Store(1)
		defer r.clock.Store(saved)
		r.c.Iterate(func(k, v []byte, _ uint64) bool {
			tag := v
			if len(tag) > 8 {
				tag = tag[:8]
			}
			out[string(k)] = string(tag)
			return true
		})
		return out
	}
	peer, restarted := stateOf(rB), stateOf(rA)
	for k, want := range peer {
		if got, ok := restarted[k]; !ok {
			t.Errorf("key %q: LOST by the restarted replica (peer has %q)", k, want)
		} else if got != want {
			t.Errorf("key %q: restarted replica resolved to %q, peer has %q — a committed "+
				"overwrite was silently reverted (#12A)", k, got, want)
		}
	}
	for k, got := range restarted {
		if _, ok := peer[k]; !ok {
			t.Errorf("key %q: RESURRECTED by the restart with value %q; the peer deleted it (#12B)", k, got)
		}
	}
	t.Fatal("SILENT CROSS-REPLICA DIVERGENCE: a graceful restart changed the restarted " +
		"replica's committed state. Both replicas ACKed the identical log; nothing in the " +
		"system can detect or repair the difference.")
}

// i64 encodes v as the 8-byte big-endian value the incr op expects.
func i64(v int64) []byte {
	b := make([]byte, 8)
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)
	return b
}

// beI64 decodes an 8-byte big-endian value.
func beI64(b []byte) int64 {
	if len(b) != 8 {
		return -1
	}
	var v int64
	for _, x := range b {
		v = v<<8 | int64(x)
	}
	return v
}
