// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"fmt"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"

	hraft "github.com/hashicorp/raft"
)

// These tests prove #4 vector TTL determinism (the vector analog of the KV B1/B3a
// work in determinism_test.go): a cluster-replicated shard's COMMITTED VECTOR state
// — the stored absolute point/per-key TTL deadlines and the insert-if-absent
// liveness outcome — is identical across replicas whose wall clocks differ, because
// the leader apply stamp drives every deadline computation and liveness check
// (InsertAt et al.) instead of each node's own clock. The B1 case is a before/after:
// the SAME workload applied through UNSTAMPED (wall-clock) entries DIVERGES under
// skew, while the STAMPED path CONVERGES.
//
// FINGERPRINT: the FSM snapshot cannot be the fingerprint here — the vector snapshot
// serializes per-slot metadata/keyExpires in Go map-iteration order (non-canonical),
// so two byte-identical states can serialize to different bytes, and the snapshot
// format is deliberately NOT changed by this work. Instead the fingerprint reads the
// committed state deterministically through Get at a FIXED set of pinned probe
// clocks that straddle the deadlines: at each probe it records, per id in sorted
// order, whether the point is live, its version, its remaining point TTL, and the
// set of live payload keys (per-key-TTL-filtered). Identical absolute deadlines
// (stamped) ⇒ identical membership at every probe on every replica; deadlines
// shifted by wall-clock skew (unstamped) ⇒ different membership at some probe.

// vecBase is a FIXED epoch-ms origin so the workload, the skewed replica clocks, and
// the probe clocks are all deterministic constants (no time.Now variability).
const vecBase = int64(1_700_000_000_000)

// fpProbes are the pinned clocks the fingerprint reads at, chosen to straddle the
// per-key TTLs (5/10/15/25 s) and point TTLs (20/30/45 s) the workload stamps at
// ~vecBase, so any wall-clock-skewed deadline shift changes membership at some probe.
var fpProbes = []int64{
	vecBase - 100_000, // everything live
	vecBase + 6_000,
	vecBase + 11_000,
	vecBase + 16_000,
	vecBase + 21_000,
	vecBase + 26_000,
	vecBase + 31_000,
	vecBase + 46_000,
	vecBase + 61_000,
}

// vecReplica is one simulated replica: an fsm over its own cache + replicated
// (sweeper-suppressed) vector store, plus a mutable injected wall clock (clock) the
// test skews and pins via the store-level SetNowFunc seam.
type vecReplica struct {
	f     *fsm
	c     *cache.Cache
	vs    *vector.CollectionStore
	clock *atomic.Int64
}

// vecCfg is the fixed collection config every replica builds. The FIXED Seed keeps
// the graph build deterministic across replicas from the same insert order.
func vecCfg() vector.Config {
	return vector.Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
}

// newVecReplica builds a replica whose injected vector wall clock starts at clock0
// (unix millis). replicated=true opens the store in persistent-cluster mode, which
// forces Config.SuppressSweep (background sweeper off, wall-clock lazy removal
// suppressed) — the #4 B3a analog. It creates the "docs" collection so a later
// stamped vector_insert has somewhere to land.
func newVecReplica(t *testing.T, clock0 int64, replicated bool) *vecReplica {
	t.Helper()
	clk := &atomic.Int64{}
	clk.Store(clock0)

	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	vs, err := vector.OpenCollectionStorePersistent(t.TempDir(), replicated)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vs.Close() })
	// Skew this replica's vector wall clock. Production never calls SetNowFunc, so
	// only the non-apply sites (sweeper/read filter/the wall-clock branch of writes)
	// see the skew; the stamped apply path takes the explicit leader stamp.
	vs.SetNowFunc(func() int64 { return clk.Load() })
	if err := vs.CreateCollection("docs", vecCfg()); err != nil {
		t.Fatal(err)
	}

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	return &vecReplica{f: newFSM(c, reg, false, vs, nil), c: c, vs: vs, clock: clk}
}

func (r *vecReplica) apply(t *testing.T, index uint64, entry []byte) {
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

// applyIfAbsent runs an insert-if-absent entry and returns whether it inserted.
func (r *vecReplica) applyIfAbsent(t *testing.T, index uint64, entry []byte) bool {
	t.Helper()
	res := r.f.Apply(&hraft.Log{Index: index, Type: hraft.LogCommand, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok {
		t.Fatalf("Apply returned %T, want *ApplyResponse", res)
	}
	if resp.Err != nil {
		t.Fatalf("apply index %d: %v", index, resp.Err)
	}
	inserted, err := ops.DecodeIfAbsentResult(resp.Result)
	if err != nil {
		t.Fatalf("decode if-absent result: %v", err)
	}
	return inserted
}

// fingerprint reads the committed vector TTL state deterministically: at each pinned
// probe clock it records, per id in sorted order, liveness + version + remaining
// point TTL + the sorted set of live payload keys. The replica's own clock is
// restored afterward. Two replicas with identical absolute deadlines produce
// identical bytes; a wall-clock-skewed deadline shift changes membership at a probe.
func (r *vecReplica) fingerprint(t *testing.T) []byte {
	t.Helper()
	saved := r.clock.Load()
	defer r.clock.Store(saved)
	col, ok := r.vs.Acquire("docs")
	if !ok {
		t.Fatal("acquire docs collection")
	}
	defer col.Release()

	var buf bytes.Buffer
	for _, p := range fpProbes {
		r.clock.Store(p)
		for id := uint64(1); id <= 8; id++ {
			_, meta, ttl, _, version, ok := col.Get(id)
			fmt.Fprintf(&buf, "p=%d id=%d ok=%t v=%d ttl=%d keys=[", p, id, ok, version, ttl.Milliseconds())
			keys := make([]string, 0, len(meta))
			for k := range meta {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				buf.WriteString(k)
				buf.WriteByte(',')
			}
			buf.WriteString("]\n")
		}
	}
	return buf.Bytes()
}

// vecInsertWorkload is a fixed set of vector_insert entries carrying BOTH a point
// TTL and a per-key payload TTL, encoded with the given per-op stamp: stamped=false
// (EncodeLogEntry) reproduces today's wall-clock apply, stamped=true
// (EncodeLogEntryStamped) reproduces the #4 fix. keys named in the per-key TTL map
// are present in meta so the deadline actually applies.
func vecInsertWorkload(stamp uint64, stamped bool) [][]byte {
	enc := func(name string, args []byte) []byte {
		if stamped {
			return EncodeLogEntryStamped(name, args, stamp)
		}
		return EncodeLogEntry(name, args)
	}
	mk := func(id uint64, x float32, pointTTL time.Duration, keyTTL map[string]int64) []byte {
		meta := vector.Metadata{"tag": vector.NewString("t"), "n": vector.NewInt(int64(id))}
		vec := []float32{x, -x, x + 1, x - 1}
		return enc("vector_insert", ops.EncodeVectorInsertArgsKeyTTL("docs", id, vec, pointTTL, meta, vector.SparseVector{}, keyTTL))
	}
	return [][]byte{
		mk(1, 1.0, 30*time.Second, map[string]int64{"tag": 10_000}),
		mk(2, 2.0, 0, map[string]int64{"n": 5_000}), // no point ttl, per-key only
		mk(3, 3.0, 45*time.Second, nil),             // point ttl only
		mk(4, 4.0, 0, nil),                          // no ttl at all
		mk(5, 5.0, 20*time.Second, map[string]int64{"tag": 15_000, "n": 25_000}),
	}
}

func applyVecWorkload(t *testing.T, r *vecReplica, entries [][]byte) {
	t.Helper()
	for i, e := range entries {
		r.apply(t, uint64(i+1), e)
	}
}

// TestVectorTTLDeterministicApplyStamp is the core before/after: with per-replica
// wall-clock skew, the UNSTAMPED (wall-clock) apply diverges the committed vector
// state (each replica stamps a different absolute point/per-key deadline), while the
// STAMPED apply keeps it identical.
func TestVectorTTLDeterministicApplyStamp(t *testing.T) {
	clocks := []int64{vecBase, vecBase + 30_000, vecBase - 20_000}
	stamp := uint64(vecBase + 7)

	// --- STAMPED: must converge. ---
	stampedFPs := make([][]byte, len(clocks))
	for i, ck := range clocks {
		r := newVecReplica(t, ck, true)
		applyVecWorkload(t, r, vecInsertWorkload(stamp, true))
		stampedFPs[i] = r.fingerprint(t)
	}
	for i := 1; i < len(stampedFPs); i++ {
		if !bytes.Equal(stampedFPs[0], stampedFPs[i]) {
			t.Fatalf("stamped: replica %d vector fingerprint differs from replica 0 — committed vector state diverged despite stamping", i)
		}
	}

	// --- UNSTAMPED (today's wall-clock apply): must diverge under skew. ---
	unstampedFPs := make([][]byte, len(clocks))
	for i, ck := range clocks {
		r := newVecReplica(t, ck, true)
		applyVecWorkload(t, r, vecInsertWorkload(stamp, false))
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
		t.Fatal("control: UNSTAMPED vector apply under wall-clock skew produced identical fingerprints — " +
			"the divergence repro is not exercising skew, so the stamped pass proves nothing")
	}
}

// TestVectorTTLStampedEqualsWallClockAligned is the byte-identity guarantee: the
// stamped At-path is a FAITHFUL mirror of the wall-clock path. When the injected
// wall clock is pinned to exactly the stamp value, an UNSTAMPED apply and a STAMPED
// apply of the same workload produce identical committed state — proving
// InsertAt(now=X) == Insert() with h.now()==X, so the additive At-path changes
// nothing for a single node.
func TestVectorTTLStampedEqualsWallClockAligned(t *testing.T) {
	stamp := uint64(vecBase)

	rStamped := newVecReplica(t, vecBase, true)
	applyVecWorkload(t, rStamped, vecInsertWorkload(stamp, true))

	// Same wall clock as the stamp, unstamped: the wall-clock branch reads exactly
	// vecBase for every deadline, matching the stamped path.
	rWall := newVecReplica(t, vecBase, true)
	applyVecWorkload(t, rWall, vecInsertWorkload(stamp, false))

	if !bytes.Equal(rStamped.fingerprint(t), rWall.fingerprint(t)) {
		t.Fatal("stamped apply at now==stamp differs from wall-clock apply at the same clock — the At-path is not a faithful mirror of the wall-clock path")
	}
}

// TestVectorInsertIfAbsentOutcomeDeterministic proves the insert-if-absent liveness
// OUTCOME is decided on the leader stamp, not each replica's wall clock: a point is
// inserted with a 1s point TTL stamped at S, then an insert-if-absent for the SAME
// id is applied stamped at S+2000 (2s later, so the prior point is expired relative
// to the stamp) on two replicas at wildly different wall clocks. Both must resurrect
// the id identically (inserted=true, identical fingerprint), because the expiry is
// judged against S+2000 on both — never each node's skewed clock.
func TestVectorInsertIfAbsentOutcomeDeterministic(t *testing.T) {
	stamp0 := uint64(vecBase)
	stamp1 := stamp0 + 2000

	vec := []float32{9, 8, 7, 6}
	vec2 := []float32{1, 2, 3, 4}
	meta := vector.Metadata{"g": vector.NewInt(1)}
	putEntry := EncodeLogEntryStamped("vector_insert",
		ops.EncodeVectorInsertArgsKeyTTL("docs", 42, vec, time.Second, meta, vector.SparseVector{}, nil), stamp0)
	ifAbsentEntry := EncodeLogEntryStamped("vector_insert_if_absent",
		ops.EncodeVectorInsertArgsKeyTTL("docs", 42, vec2, 0, meta, vector.SparseVector{}, nil), stamp1)

	fps := make([][]byte, 0, 2)
	for _, ck := range []int64{vecBase + 10_000, vecBase + 500_000} {
		r := newVecReplica(t, ck, true)
		r.apply(t, 1, putEntry)
		if inserted := r.applyIfAbsent(t, 2, ifAbsentEntry); !inserted {
			t.Fatalf("clock %d: insert-if-absent over a stamp-expired point returned inserted=false — the liveness outcome used the wall clock, not the stamp", ck)
		}
		fps = append(fps, r.fingerprint(t))
	}
	if !bytes.Equal(fps[0], fps[1]) {
		t.Fatal("insert-if-absent: replicas at different wall clocks diverged — the resurrection outcome/state was not judged on the leader stamp")
	}
}

// TestVectorSetPayloadTTLDeterministic proves the payload path: a stamped
// set_payload that adds a per-key payload TTL computes the SAME absolute per-key
// deadline on every replica (converges under skew), while the unstamped wall-clock
// path diverges. Points are first inserted (no TTL) via a stamped workload so the
// only clock-dependent state is the per-key deadline the set_payload stamps.
func TestVectorSetPayloadTTLDeterministic(t *testing.T) {
	clocks := []int64{vecBase, vecBase + 30_000, vecBase - 20_000}
	insertStamp := uint64(vecBase - 1_000) // insert well before the probes; no TTL
	setStamp := uint64(vecBase + 5)

	// Insert ids 1..3 with NO ttl (stamped, so the base state is identical), then a
	// set_payload on each that attaches a per-key TTL to the "tag" key.
	base := func(stamped bool) [][]byte {
		mk := func(id uint64) []byte {
			meta := vector.Metadata{"tag": vector.NewString("t")}
			e := ops.EncodeVectorInsertArgsKeyTTL("docs", id, []float32{float32(id), 0, 0, 0}, 0, meta, vector.SparseVector{}, nil)
			return EncodeLogEntryStamped("vector_insert", e, insertStamp)
		}
		set := func(id uint64, ttlMs int64) []byte {
			meta := vector.Metadata{"tag": vector.NewString("t")}
			e := ops.EncodeSetPayloadArgsOpts("docs", id, meta, map[string]int64{"tag": ttlMs})
			if stamped {
				return EncodeLogEntryStamped("vector_set_payload", e, setStamp)
			}
			return EncodeLogEntry("vector_set_payload", e)
		}
		return [][]byte{mk(1), mk(2), mk(3), set(1, 10_000), set(2, 20_000), set(3, 30_000)}
	}

	fp := func(stamped bool) [][]byte {
		out := make([][]byte, len(clocks))
		for i, ck := range clocks {
			r := newVecReplica(t, ck, true)
			for j, e := range base(stamped) {
				r.apply(t, uint64(j+1), e)
			}
			out[i] = r.fingerprint(t)
		}
		return out
	}

	stamped := fp(true)
	for i := 1; i < len(stamped); i++ {
		if !bytes.Equal(stamped[0], stamped[i]) {
			t.Fatalf("stamped set_payload: replica %d diverged — the per-key deadline was not stamped deterministically", i)
		}
	}
	unstamped := fp(false)
	allEqual := true
	for i := 1; i < len(unstamped); i++ {
		if !bytes.Equal(unstamped[0], unstamped[i]) {
			allEqual = false
			break
		}
	}
	if allEqual {
		t.Fatal("control: unstamped set_payload under skew produced identical fingerprints — the per-key-TTL divergence repro is not exercising skew")
	}
}

// TestNamedInsertTTLDeterministic proves the named-collection path: a
// stamped vector_named_insert carrying BOTH a point TTL and a per-key payload TTL
// stamps byte-identical deadlines on every replica (converges under skew), while the
// unstamped wall-clock path diverges. The fingerprint reads the named committed
// state through NamedGet at the fixed probe clocks (liveness + remaining point TTL +
// sorted live payload keys).
func TestNamedInsertTTLDeterministic(t *testing.T) {
	clocks := []int64{vecBase, vecBase + 30_000, vecBase - 20_000}
	stamp := uint64(vecBase + 5)

	namedFP := func(r *vecReplica) []byte {
		saved := r.clock.Load()
		defer r.clock.Store(saved)
		var buf bytes.Buffer
		for _, p := range fpProbes {
			r.clock.Store(p)
			for id := uint64(1); id <= 5; id++ {
				_, payload, ttl, ok, err := r.vs.NamedGet("ndocs", id)
				if err != nil {
					t.Fatalf("NamedGet: %v", err)
				}
				fmt.Fprintf(&buf, "p=%d id=%d ok=%t ttl=%d keys=[", p, id, ok, ttl.Milliseconds())
				keys := make([]string, 0, len(payload))
				for k := range payload {
					keys = append(keys, k)
				}
				sort.Strings(keys)
				for _, k := range keys {
					buf.WriteString(k)
					buf.WriteByte(',')
				}
				buf.WriteString("]\n")
			}
		}
		return buf.Bytes()
	}

	build := func(clock0 int64, stamped bool) []byte {
		r := newVecReplica(t, clock0, true)
		if err := r.vs.CreateNamed("ndocs", map[string]vector.NamedVectorParams{
			"d": {Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1},
		}); err != nil {
			t.Fatal(err)
		}
		// Re-apply the clock override so the just-created named collection inherits the
		// skewed clock (SetNowFunc propagates to currently-resident indexes).
		r.vs.SetNowFunc(func() int64 { return r.clock.Load() })
		enc := func(args []byte) []byte {
			if stamped {
				return EncodeLogEntryStamped("vector_named_insert", args, stamp)
			}
			return EncodeLogEntry("vector_named_insert", args)
		}
		idx := uint64(1)
		mk := func(id uint64, pointTTL time.Duration, keyTTL map[string]int64) {
			payload := vector.Metadata{"tag": vector.NewString("t"), "n": vector.NewInt(int64(id))}
			vecs := map[string][]float32{"d": {float32(id), -float32(id), 1, 0}}
			args := ops.EncodeNamedInsertArgsKeyTTL("ndocs", id, vecs, payload, pointTTL, keyTTL)
			r.apply(t, idx, enc(args))
			idx++
		}
		mk(1, 30*time.Second, map[string]int64{"tag": 10_000})
		mk(2, 0, map[string]int64{"n": 5_000})
		mk(3, 45*time.Second, nil)
		mk(4, 0, nil)
		mk(5, 20*time.Second, map[string]int64{"tag": 15_000})
		return namedFP(r)
	}

	stampedFPs := make([][]byte, len(clocks))
	for i, ck := range clocks {
		stampedFPs[i] = build(ck, true)
	}
	for i := 1; i < len(stampedFPs); i++ {
		if !bytes.Equal(stampedFPs[0], stampedFPs[i]) {
			t.Fatalf("stamped named insert: replica %d diverged — a point/per-key deadline was not stamped deterministically", i)
		}
	}
	unstampedFPs := make([][]byte, len(clocks))
	for i, ck := range clocks {
		unstampedFPs[i] = build(ck, false)
	}
	allEqual := true
	for i := 1; i < len(unstampedFPs); i++ {
		if !bytes.Equal(unstampedFPs[0], unstampedFPs[i]) {
			allEqual = false
			break
		}
	}
	if allEqual {
		t.Fatal("control: unstamped named insert under skew produced identical fingerprints — the divergence repro is not exercising skew")
	}
}

// TestVectorReplicatedNoWallClockSweep proves the sweeper gate (B3a
// analog): on a replicated (persistent-cluster) collection a point past its TTL is
// FILTERED on a client read (search miss) but NOT physically removed, so it survives
// identically across two replicas at different wall clocks and no expirations are
// recorded. The single-node control (sweeper on) instead reclaims the expired point.
func TestVectorReplicatedNoWallClockSweep(t *testing.T) {
	stamp := uint64(vecBase)
	vec := []float32{1, 0, 0, 0}
	meta := vector.Metadata{"d": vector.NewInt(1)}
	// exp = stamp + 1s, identical on both replicas.
	entry := EncodeLogEntryStamped("vector_insert",
		ops.EncodeVectorInsertArgsKeyTTL("docs", 7, vec, time.Second, meta, vector.SparseVector{}, nil), stamp)

	// Two replicated replicas, both wall clocks well past exp.
	rA := newVecReplica(t, vecBase+10_000, true)
	rB := newVecReplica(t, vecBase+90_000, true)
	rA.apply(t, 1, entry)
	rB.apply(t, 1, entry)

	// A client search at the skewed (past-exp) wall clock filters the point out...
	if docs, err := rA.vs.SearchDocs("docs", vec, 4, vector.Filter{}); err != nil {
		t.Fatal(err)
	} else if len(docs) != 0 {
		t.Fatalf("replicated search of expired point returned %d docs, want 0 (filtered)", len(docs))
	}
	// ...but the point is NOT physically removed: no expirations recorded, and both
	// replicas agree (fingerprint pins low so the point is visible at capture).
	if colA, ok := rA.vs.Acquire("docs"); ok {
		if exp := colA.Stats().Expired; exp != 0 {
			colA.Release()
			t.Fatalf("replicated collection recorded %d expirations — physical removal was not suppressed", exp)
		}
		colA.Release()
	}
	if !bytes.Equal(rA.fingerprint(t), rB.fingerprint(t)) {
		t.Fatal("replicated replicas at different wall clocks diverged — wall-clock physical removal leaked past the sweeper gate")
	}

	// Single-node control: sweeper ON (non-persistent), small interval. The same
	// point written via the wall-clock client path is physically reclaimed once the
	// clock passes exp.
	clk := &atomic.Int64{}
	clk.Store(vecBase)
	ctrl, err := vector.OpenCollectionStorePersistent(t.TempDir(), false)
	if err != nil {
		t.Fatal(err)
	}
	defer ctrl.Close()
	ctrl.SetNowFunc(func() int64 { return clk.Load() })
	cfg := vecCfg()
	cfg.SweepInterval = 20 * time.Millisecond
	if err := ctrl.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	if err := ctrl.Insert("docs", 7, vec, time.Second, meta, nil); err != nil {
		t.Fatal(err)
	}
	clk.Store(vecBase + 10_000) // advance the control's wall clock past exp
	col, _ := ctrl.Acquire("docs")
	defer col.Release()
	deadline := time.Now().Add(5 * time.Second)
	reclaimed := false
	for time.Now().Before(deadline) {
		if col.Stats().Expired > 0 {
			reclaimed = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !reclaimed {
		t.Fatal("control: single-node sweeper did not reclaim the expired point — control did not exercise sweeping")
	}
}
