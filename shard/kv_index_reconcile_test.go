// SPDX-License-Identifier: Apache-2.0

package shard

// Store-level tests for the bounded reconcile pass. What only a Store can show
// is the LIFECYCLE: that the ticker exists at all on an ordinary store (a
// single-node Direct one included, since Direct builds these), that it rotates
// over the definitions instead of starving all but one, and that Close stops it
// before the cache it probes is unmapped.
//
// The tick's own semantics — the three-phase split, the liveness rule, the
// expired-key deadlock — are pinned in ops and ops/kvindex, which is where the
// cache and the index meet without a Raft group in the way.

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// recStore builds a bootstrapped single-node Store with the reconcile ticker
// set to intervalMs (0 disables it).
func recStore(t *testing.T, intervalMs int) *Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	// t.TempDir() BEFORE the Close cleanup is registered, so cleanup LIFO runs
	// Close first and the directory is never removed under a live raft goroutine.
	dir := t.TempDir()
	cfg := DefaultConfig(dir, "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	cfg.KVIndexReconcileIntervalMs = intervalMs
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// recStoreDef builds a definition named name over the key prefix and the
// payload field of the same name.
func recStoreDef(t *testing.T, name, prefix, field string) kvindex.Def {
	t.Helper()
	d, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        name,
		KeyPrefix:   []byte(prefix),
		PayloadPath: field,
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom(%q): %v", name, err)
	}
	return d
}

func recStoreRec(field string, v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: field, Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}, //nolint:gosec // fixture reinterprets a small i64
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

// recStorePosted counts the keys name still posts.
func recStorePosted(s *Store, name string) int {
	keys, _ := s.kvIdx.Stats(name)
	return keys
}

// recStoreDangle writes n keys under prefix, indexes them, then deletes them
// from the cache WITH THE REMOVAL HOOK DETACHED — which is what a removal path
// that never reaches Drop looks like from the index's side, and the only thing
// the reconcile pass exists to clean up.
func recStoreDangle(t *testing.T, s *Store, prefix, field string, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		k := []byte(fmt.Sprintf("%s%03d", prefix, i))
		val := recStoreRec(field, 7)
		if err := s.cache.Put(k, val, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		s.kvIdx.Reindex(k, val)
	}
	s.cache.SetOnRemove(nil)
	for i := 0; i < n; i++ {
		if _, err := s.cache.Del([]byte(fmt.Sprintf("%s%03d", prefix, i))); err != nil {
			t.Fatalf("Del: %v", err)
		}
	}
	s.cache.SetOnRemove(s.kvIdx.Drop)
}

// TestReconcileRotatesDefinitions: one definition per tick, in rotation. A pass
// that always started at definition 0 would leave every later one uncleaned for
// as long as the first had work, which on a busy index is forever.
func TestReconcileRotatesDefinitions(t *testing.T) {
	s := recStore(t, 0) // no ticker: the test drives the ticks itself
	names := []string{"by-a", "by-b", "by-c"}
	defs := make([]kvindex.Def, 0, len(names))
	for i, n := range names {
		defs = append(defs, recStoreDef(t, n, fmt.Sprintf("%c:", 'a'+i), "rc"))
	}
	s.kvIdx.Install(defs)
	for i, n := range names {
		recStoreDangle(t, s, fmt.Sprintf("%c:", 'a'+i), "rc", 4)
		s.kvIdx.MarkReady(n)
		if got := recStorePosted(s, n); got != 4 {
			t.Fatalf("%s posts %d dangling keys, want 4", n, got)
		}
	}

	cleaned := 0
	for tick := 0; tick < len(names); tick++ {
		s.kvIndexReconcileTick(nil)
		cleaned = 0
		for _, n := range names {
			if recStorePosted(s, n) == 0 {
				cleaned++
			}
		}
		if cleaned != tick+1 {
			t.Fatalf("after %d tick(s) %d definitions are clean, want %d — the rotation is not advancing",
				tick+1, cleaned, tick+1)
		}
	}
}

// TestReconcileSkipsDefinitionsThatAreStillBuilding: a posting set mid-backfill
// is a proper subset being refilled, so there is nothing there to call dangling.
func TestReconcileSkipsDefinitionsThatAreStillBuilding(t *testing.T) {
	s := recStore(t, 0)
	s.kvIdx.Install([]kvindex.Def{recStoreDef(t, "by-rc", "u:", "rc")})
	recStoreDangle(t, s, "u:", "rc", 5)
	// NOT marked ready: this is what a definition whose backfill has not
	// finished looks like.
	s.kvIndexReconcileTick(nil)
	if got := recStorePosted(s, "by-rc"); got != 5 {
		t.Fatalf("a tick reconciled a building definition: %d keys left, want all 5", got)
	}
	s.kvIdx.MarkReady("by-rc")
	s.kvIndexReconcileTick(nil)
	if got := recStorePosted(s, "by-rc"); got != 0 {
		t.Fatalf("a tick on a ready definition left %d dangling postings, want 0", got)
	}
}

// TestReconcileTickerCleansOnItsOwn is the wiring test: a Store started with an
// interval reconciles without anyone calling the tick, which is what makes a
// single-node Direct store (which has no cluster observer) reconcile at all.
func TestReconcileTickerCleansOnItsOwn(t *testing.T) {
	s := recStore(t, 10)
	s.kvIdx.Install([]kvindex.Def{recStoreDef(t, "by-rc", "u:", "rc")})
	recStoreDangle(t, s, "u:", "rc", 8)
	s.kvIdx.MarkReady("by-rc")

	deadline := time.Now().Add(10 * time.Second)
	for recStorePosted(s, "by-rc") != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("the ticker never cleaned the dangling postings: %d left", recStorePosted(s, "by-rc"))
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := s.kvIdx.ReconcileDrops(); got != 8 {
		t.Fatalf("ReconcileDrops() = %d, want 8", got)
	}
}

// TestReconcileLeavesLiveKeysAlone: the ticker running against a keyspace with
// nothing dangling in it must not remove a single posting. This is the property
// a wrong liveness test would break, and it would break it silently — the rows
// would simply stop coming back.
func TestReconcileLeavesLiveKeysAlone(t *testing.T) {
	s := recStore(t, 5)
	d := recStoreDef(t, "by-rc", "u:", "rc")
	s.kvIdx.Install([]kvindex.Def{d})
	for i := 0; i < 40; i++ {
		k := []byte(fmt.Sprintf("u:%03d", i))
		val := recStoreRec("rc", 7)
		if err := s.cache.Put(k, val, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		s.kvIdx.Reindex(k, val)
	}
	s.kvIdx.MarkReady("by-rc")

	time.Sleep(100 * time.Millisecond) // many ticks
	if got := s.kvIdx.ReconcileDrops(); got != 0 {
		t.Fatalf("the reconciler dropped %d postings for LIVE keys", got)
	}
	keys, err := s.kvIdx.Candidates(kvindex.Selector{
		Def:    d,
		Op:     vtypes.FilterEq,
		Values: []vtypes.Value{vtypes.NewInt(7)},
	}, nil, 1<<20)
	if err != nil {
		t.Fatalf("Candidates: %v", err)
	}
	if len(keys) != 40 {
		t.Fatalf("the index offers %d keys, want all 40", len(keys))
	}
}

// TestReconcileStopsOnClose: the ticker is a store-owned goroutine, and Close
// stops AND JOINS it before the cache is unmapped. If it merely signalled, a
// tick already inside c.Get would keep probing a mapping cache.Close is about
// to tear down.
func TestReconcileStopsOnClose(t *testing.T) {
	base := settledGoroutines()
	for i := 0; i < 3; i++ {
		func() {
			reg := ops.NewRegistry()
			if err := ops.RegisterBuiltins(reg); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			cfg := DefaultConfig(dir, fmt.Sprintf("node%d", i), reg)
			cfg.Bootstrap = true
			cfg.RaftHeartbeatMs = 50
			cfg.RaftElectionMs = 100
			cfg.NoSync = true
			cfg.KVIndexReconcileIntervalMs = 5
			s, err := New(cfg)
			if err != nil {
				t.Fatalf("shard.New: %v", err)
			}
			s.kvIdx.Install([]kvindex.Def{recStoreDef(t, "by-rc", "u:", "rc")})
			s.kvIdx.MarkReady("by-rc")
			time.Sleep(30 * time.Millisecond) // let it tick
			if err := s.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			// A second Close must not panic on a closed channel or hang on a
			// WaitGroup that is already done.
			_ = s.Close()
		}()
	}
	if now := settledGoroutines(); now > base+2 {
		buf := make([]byte, 1<<16)
		buf = buf[:runtime.Stack(buf, true)]
		t.Fatalf("goroutines %d -> %d after three closed stores; the reconcile ticker leaks.\n%s", base, now, buf)
	}
}

// TestReconcileDisabledByZeroInterval: 0 means off, and off means no goroutine.
func TestReconcileDisabledByZeroInterval(t *testing.T) {
	s := recStore(t, 0)
	if s.kvReconcileStop != nil {
		t.Fatal("a zero reconcile interval started the ticker anyway")
	}
	s.kvIdx.Install([]kvindex.Def{recStoreDef(t, "by-rc", "u:", "rc")})
	recStoreDangle(t, s, "u:", "rc", 4)
	s.kvIdx.MarkReady("by-rc")
	time.Sleep(50 * time.Millisecond)
	if got := recStorePosted(s, "by-rc"); got != 4 {
		t.Fatalf("a disabled reconciler dropped postings: %d left, want 4", got)
	}
}

// TestReconcileDefaultIntervalIsOn: the default config runs the pass, so a
// deployment gets the memory bound without opting in.
func TestReconcileDefaultIntervalIsOn(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "n", reg)
	if cfg.KVIndexReconcileIntervalMs != defaultKVIndexReconcileIntervalMs {
		t.Fatalf("DefaultConfig sets KVIndexReconcileIntervalMs=%d, want %d",
			cfg.KVIndexReconcileIntervalMs, defaultKVIndexReconcileIntervalMs)
	}
	cfg.KVIndexReconcileIntervalMs = -1
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate accepted a negative reconcile interval")
	}
}

// settledGoroutines waits for the goroutine count to stop moving, so a raft or
// cache goroutine still winding down is not mistaken for a leak.
func settledGoroutines() int {
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
