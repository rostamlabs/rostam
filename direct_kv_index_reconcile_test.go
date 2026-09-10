// SPDX-License-Identifier: Apache-2.0

package rostam

// Lifecycle tests for the Direct store's KV index reconcile ticker. A Direct
// store builds no shard.Store and has no cluster Node, so if this ticker is not
// here the whole single-node embedded deployment never reconciles at all —
// which is exactly the deployment with nothing else watching the index grow.
//
// The tick's own semantics are pinned in ops and ops/kvindex; what is proved
// here is that a Direct store starts one, that it cleans, that the knob turns
// it off, and that Close stops and JOINS it before the cache is unmapped.

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

func directRecDef(t *testing.T) kvindex.Def {
	t.Helper()
	d, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        "by-rc",
		KeyPrefix:   []byte("u:"),
		PayloadPath: "rc",
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom: %v", err)
	}
	return d
}

func directRecRec(v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}, //nolint:gosec // fixture reinterprets a small i64
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

// newDirectForReconcile builds a Direct store with the reconcile knob set and
// returns the concrete type, so the test can reach the cache and the index the
// ticker works on. It does NOT register a Close cleanup: the lifecycle tests
// close it themselves.
func newDirectForReconcile(t *testing.T, intervalMs int) *directStore {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{
		Ops:                        reg,
		Cache:                      CacheConfig{NumShardsPerNode: 1},
		KVIndexReconcileIntervalMs: intervalMs,
	})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	d, ok := s.(*directStore)
	if !ok {
		t.Fatalf("NewDirect returned %T, want *directStore", s)
	}
	return d
}

// directDangle writes n keys, indexes them, then deletes them from the cache
// with the removal hook DETACHED — what a removal path that never reaches Drop
// looks like from the index's side, and the only residue this pass exists for.
func directDangle(t *testing.T, d *directStore, idx *kvindex.Set, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		k := []byte(fmt.Sprintf("u:%03d", i))
		val := directRecRec(7)
		if err := d.cache.Put(k, val, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		idx.Reindex(k, val)
	}
	d.cache.SetOnRemove(nil)
	for i := 0; i < n; i++ {
		if _, err := d.cache.Del([]byte(fmt.Sprintf("u:%03d", i))); err != nil {
			t.Fatalf("Del: %v", err)
		}
	}
	d.cache.SetOnRemove(idx.Drop)
}

// TestDirectReconcilerCleansOnItsOwn is the regression test for "a Direct
// deployment never reconciles": before this the ticker lived only on
// shard.Store, which NewDirect does not build.
func TestDirectReconcilerCleansOnItsOwn(t *testing.T) {
	d := newDirectForReconcile(t, 10)
	defer func() { _ = d.Close() }()

	idx := d.tx.KVIndex()
	if idx == nil {
		t.Fatal("a Direct store has no KV index")
	}
	idx.Install([]kvindex.Def{directRecDef(t)})
	directDangle(t, d, idx, 8)
	idx.MarkReady("by-rc")

	deadline := time.Now().Add(10 * time.Second)
	for {
		keys, _ := idx.Stats("by-rc")
		if keys == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the Direct ticker never cleaned the dangling postings: %d left", keys)
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := idx.ReconcileDrops(); got != 8 {
		t.Fatalf("ReconcileDrops() = %d, want 8", got)
	}
}

// TestDirectReconcilerLeavesLiveKeysAlone: many ticks against a keyspace with
// nothing dangling in it must not remove a single posting.
func TestDirectReconcilerLeavesLiveKeysAlone(t *testing.T) {
	d := newDirectForReconcile(t, 5)
	defer func() { _ = d.Close() }()

	idx := d.tx.KVIndex()
	idx.Install([]kvindex.Def{directRecDef(t)})
	for i := 0; i < 40; i++ {
		k := []byte(fmt.Sprintf("u:%03d", i))
		val := directRecRec(7)
		if err := d.cache.Put(k, val, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		idx.Reindex(k, val)
	}
	idx.MarkReady("by-rc")

	time.Sleep(100 * time.Millisecond) // many ticks
	if got := idx.ReconcileDrops(); got != 0 {
		t.Fatalf("the Direct reconciler dropped %d postings for LIVE keys", got)
	}
	if keys, _ := idx.Stats("by-rc"); keys != 40 {
		t.Fatalf("the index posts %d keys, want all 40", keys)
	}
}

// TestDirectReconcilerInterval covers the knob's three states, which follow the
// convention DirectConfig already uses for TTLSweepIntervalMs.
func TestDirectReconcilerInterval(t *testing.T) {
	if got := directReconcileInterval(0); got != defaultKVIndexReconcileInterval {
		t.Errorf("directReconcileInterval(0) = %v, want the %v default", got, defaultKVIndexReconcileInterval)
	}
	if got := directReconcileInterval(-1); got != 0 {
		t.Errorf("directReconcileInterval(-1) = %v, want 0 (disabled)", got)
	}
	if got := directReconcileInterval(250); got != 250*time.Millisecond {
		t.Errorf("directReconcileInterval(250) = %v, want 250ms", got)
	}
	// AN ABSURD SETTING MUST STAY ABSURD, not wrap into a different meaning.
	// time.Duration is int64 nanoseconds, so ms * time.Millisecond overflows
	// above about 9.2e9 — and the wrapped value can be NEGATIVE, which this
	// function's own contract reads as "disabled". A misconfigured knob would
	// then silently switch the reconciler off rather than run it slowly.
	//
	// Guarded on the platform int width: where int is 32 bits the largest
	// possible setting is about 2.1e9 ms, which still fits, so there is nothing
	// to clamp and nothing to assert.
	if maxReconcileIntervalMs < int64(math.MaxInt) {
		for _, ms := range []int64{maxReconcileIntervalMs + 1, math.MaxInt64} {
			if got := directReconcileInterval(int(ms)); got <= 0 {
				t.Errorf("directReconcileInterval(%d) = %v; an over-large interval must never read as disabled", ms, got)
			}
		}
		boundary := maxReconcileIntervalMs // via a variable: a constant conversion would not compile on a 32-bit int
		if got := directReconcileInterval(int(boundary)); got != time.Duration(maxReconcileIntervalMs)*time.Millisecond {
			t.Errorf("directReconcileInterval(%d) = %v, want the exact conversion at the clamp boundary", maxReconcileIntervalMs, got)
		}
	}

	d := newDirectForReconcile(t, -1)
	defer func() { _ = d.Close() }()
	if d.kvReconcileStop != nil {
		t.Fatal("a negative reconcile interval started the ticker anyway")
	}
	idx := d.tx.KVIndex()
	idx.Install([]kvindex.Def{directRecDef(t)})
	directDangle(t, d, idx, 4)
	idx.MarkReady("by-rc")
	time.Sleep(50 * time.Millisecond)
	if keys, _ := idx.Stats("by-rc"); keys != 4 {
		t.Fatalf("a disabled reconciler dropped postings: %d left, want 4", keys)
	}
}

// TestDirectReconcilerStopsOnClose: the ticker is store-owned, and Close stops
// AND JOINS it before d.cache.Close unmaps.
func TestDirectReconcilerStopsOnClose(t *testing.T) {
	base := directSettledGoroutines()
	for i := 0; i < 3; i++ {
		d := newDirectForReconcile(t, 5)
		idx := d.tx.KVIndex()
		idx.Install([]kvindex.Def{directRecDef(t)})
		idx.MarkReady("by-rc")
		time.Sleep(30 * time.Millisecond) // let it tick
		if err := d.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		// A second Close must not panic on a closed channel or hang on a
		// WaitGroup that is already done.
		_ = d.Close()
	}
	if now := directSettledGoroutines(); now > base+2 {
		buf := make([]byte, 1<<16)
		buf = buf[:runtime.Stack(buf, true)]
		t.Fatalf("goroutines %d -> %d after three closed Direct stores; the reconcile ticker leaks.\n%s", base, now, buf)
	}
}

// directSettledGoroutines waits for the goroutine count to stop moving, so a
// cache goroutine still winding down is not mistaken for a leak.
func directSettledGoroutines() int {
	prev := -1
	for i := 0; i < 100; i++ {
		runtime.GC()
		time.Sleep(20 * time.Millisecond)
		n := runtime.NumGoroutine()
		if n == prev {
			return n
		}
		prev = n
	}
	return runtime.NumGoroutine()
}

// TestDirectReconcileDefaultMatchesShard pins the two 60 s defaults together.
// They are separate constants in separate packages — shard's is unexported and
// a directStore has no shard.Config to read — so nothing but this stops one
// from drifting and leaving Embedded and Direct deployments reconciling at
// different cadences for no stated reason.
func TestDirectReconcileDefaultMatchesShard(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	shardDefault := time.Duration(shard.DefaultConfig(t.TempDir(), "n", reg).KVIndexReconcileIntervalMs) * time.Millisecond
	if shardDefault != defaultKVIndexReconcileInterval {
		t.Fatalf("Direct reconciles every %v but shard.DefaultConfig says %v; the two defaults have drifted",
			defaultKVIndexReconcileInterval, shardDefault)
	}
}

// TestDirectReconcileTickBlocksCloseUntilItReturns is the shard fence test's
// twin. Direct's reconciler is a SEPARATE COPY of the ticker, so the join in
// its Close needs its own proof: TestDirectReconcilerStopsOnClose above is the
// leak shape, which a signal-only stop also passes.
//
// The tick is parked INSIDE the probe — every sampled key is expired on a
// non-replicated cache, so cache.Get runs dropExpiredLocked and fires onRemove,
// where the hook blocks. Close must not return until it is released, or
// d.cache.Close unmaps pages that Get is reading.
func TestDirectReconcileTickBlocksCloseUntilItReturns(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{
		Ops: reg,
		Cache: CacheConfig{
			NumShardsPerNode:   1,
			TTLSweepIntervalMs: -1, // the probe's own Get must be what expires a key
		},
		KVIndexReconcileIntervalMs: 5,
	})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	d, ok := s.(*directStore)
	if !ok {
		t.Fatalf("NewDirect returned %T, want *directStore", s)
	}
	closed := false
	defer func() {
		if !closed {
			_ = d.Close()
		}
	}()

	idx := d.tx.KVIndex()
	idx.Install([]kvindex.Def{directRecDef(t)})
	for i := 0; i < 20; i++ {
		k := []byte(fmt.Sprintf("u:%03d", i))
		val := directRecRec(7)
		if err := d.cache.Put(k, val, 20*time.Millisecond); err != nil {
			t.Fatalf("Put: %v", err)
		}
		idx.Reindex(k, val)
	}
	idx.MarkReady("by-rc")

	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	d.cache.SetOnRemove(func(key []byte) {
		// The FIRST expiring probe only; later ones must not block, or the tick
		// would never drain after the release.
		once.Do(func() {
			close(entered)
			<-release
		})
		idx.Drop(key)
	})

	time.Sleep(30 * time.Millisecond) // let the keys expire
	select {
	case <-entered:
	case <-time.After(20 * time.Second):
		t.Fatal("no Direct reconcile tick reached the probe")
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- d.Close() }()

	select {
	case err := <-closeDone:
		close(release)
		t.Fatalf("Close returned (%v) while a reconcile tick was still inside the probe; "+
			"cache.Close would unmap pages that Get is reading", err)
	case <-time.After(300 * time.Millisecond):
		// Correct: Close is parked in stopKVIndexReconciler's join.
	}

	close(release)
	select {
	case err := <-closeDone:
		closed = true
		if err != nil {
			t.Fatalf("Close after the tick released: %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("Close did not return after the Direct reconcile tick was released")
	}
}
