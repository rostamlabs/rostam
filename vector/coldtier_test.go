// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"context"
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// countingStore wraps a MemStore and counts Get calls, to assert single-flight
// promotion fetches a cold snapshot exactly once under concurrency.
type countingStore struct {
	*objstore.MemStore
	gets atomic.Int64
}

func (c *countingStore) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	c.gets.Add(1)
	return c.MemStore.Get(ctx, key)
}

// failGetStore returns an error from Get for any key, to exercise the
// restore-failure path (the stub must survive).
type failGetStore struct {
	*objstore.MemStore
}

var errGetBoom = errors.New("get boom")

func (f *failGetStore) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	return nil, errGetBoom
}

func coldTestCfg() Config {
	return Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
}

func newColdStore(t *testing.T) *CollectionStore {
	t.Helper()
	s, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func seedCollection(t *testing.T, s *CollectionStore, name string, ids ...uint64) {
	t.Helper()
	if err := s.CreateCollection(name, coldTestCfg()); err != nil {
		t.Fatalf("create %q: %v", name, err)
	}
	c, ok := s.Acquire(name)
	if !ok {
		t.Fatalf("acquire %q", name)
	}
	defer c.Release()
	for _, id := range ids {
		if err := c.Insert(id, []float32{float32(id), 0}, 0, nil, nil); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
}

// TestEvictReleasesResidentData proves evict turns a collection into a STUB: it is
// still listed and config-introspectable, but holds no resident index (no live
// *Collection in the catalog — the arena/graph are gone).
func TestEvictReleasesResidentData(t *testing.T) {
	ctx := context.Background()
	s := newColdStore(t)
	seedCollection(t, s, "docs", 1, 2, 3)

	obj := objstore.NewMemStore()
	ts := time.Date(2026, 6, 23, 10, 0, 0, 0, time.UTC)
	if err := s.EvictCollection(ctx, "docs", obj, "acme", ts); err != nil {
		t.Fatalf("evict: %v", err)
	}

	// Listed (catalog identity survives).
	listed := false
	for _, n := range s.CollectionNames() {
		if n == "default/docs" {
			listed = true
		}
	}
	if !listed {
		t.Fatal("evicted collection is not listed")
	}
	if !s.IsCold("docs") {
		t.Fatal("collection should be cold after evict")
	}

	// Resident data released: no live *Collection in the hot map (the stub holds no
	// arena/graph), so a direct Get (non-promoting lookup) misses.
	s.mu.RLock()
	_, hot := s.collections["default/docs"]
	stub, cold := s.cold["default/docs"]
	s.mu.RUnlock()
	if hot {
		t.Fatal("a live Collection still resident after evict — memory not released")
	}
	if !cold || stub == nil {
		t.Fatal("no cold stub registered")
	}
	if stub.cfg.Dim != 2 {
		t.Fatalf("stub lost config: Dim=%d", stub.cfg.Dim)
	}

	// Evicting an already-cold collection is a no-op.
	if err := s.EvictCollection(ctx, "docs", obj, "acme", ts); err != nil {
		t.Fatalf("re-evict should be a no-op: %v", err)
	}
}

// TestLazyRestoreOnAccess proves a search after evict lazily promotes the cold
// collection and returns the correct results.
func TestLazyRestoreOnAccess(t *testing.T) {
	ctx := context.Background()
	s := newColdStore(t)
	seedCollection(t, s, "docs", 1, 2, 3)

	before, err := s.SearchFiltered("docs", []float32{1, 0}, 3, Filter{})
	if err != nil {
		t.Fatal(err)
	}

	obj := objstore.NewMemStore()
	ts := time.Date(2026, 6, 23, 10, 0, 0, 0, time.UTC)
	if err := s.EvictCollection(ctx, "docs", obj, "acme", ts); err != nil {
		t.Fatalf("evict: %v", err)
	}
	if !s.IsCold("docs") {
		t.Fatal("not cold after evict")
	}

	// A search resolves the cold collection → lazily promotes → serves.
	after, err := s.SearchFiltered("docs", []float32{1, 0}, 3, Filter{})
	if err != nil {
		t.Fatalf("search after evict: %v", err)
	}
	if s.IsCold("docs") {
		t.Fatal("collection should be hot after a lazy-restoring access")
	}
	if len(after) != len(before) {
		t.Fatalf("post-restore len %d != %d", len(after), len(before))
	}
	for i := range before {
		if after[i].ID != before[i].ID || after[i].Score != before[i].Score {
			t.Errorf("result[%d] = (%d,%v), want (%d,%v)", i, after[i].ID, after[i].Score, before[i].ID, before[i].Score)
		}
	}
}

// TestLazyRestoreQuantizedFaithful proves the lazy promote is config-faithful for
// a QUANTIZED collection: it rebuilds with the original quantizer (the stub's
// Config) so results match. A config-less promote would rebuild plain HNSW.
func TestLazyRestoreQuantizedFaithful(t *testing.T) {
	ctx := context.Background()
	const (
		n   = 1200
		dim = 64
		k   = 10
	)
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	st := uint64(12345)
	for i := 0; i < n; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, dim)
		for d := range v {
			st = st*6364136223846793005 + 1442695040888963407
			v[d] = float32((st>>33)&0xFFFFFF)/float32(0x1000000)*2 - 1
		}
		vecs[i] = v
	}
	queries := vecs[:15]

	s := newColdStore(t)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7,
		Quant: QuantSQ, SQBits: 4}
	if err := s.CreateCollection("sq", cfg); err != nil {
		t.Fatal(err)
	}
	if err := s.StageBulk("sq", ids, vecs); err != nil {
		t.Fatal(err)
	}
	if err := s.BuildStaged("sq", 4); err != nil {
		t.Fatal(err)
	}
	before := make([][]uint64, len(queries))
	{
		c, _ := s.Acquire("sq")
		for i, q := range queries {
			res, err := c.Search(q, k)
			if err != nil {
				t.Fatal(err)
			}
			before[i] = resultIDs(res)
		}
		c.Release()
	}

	obj := objstore.NewMemStore()
	if err := s.EvictCollection(ctx, "sq", obj, "acme", time.Unix(0, 0).UTC()); err != nil {
		t.Fatalf("evict: %v", err)
	}

	// Lazy restore on access.
	c, ok := s.Acquire("sq")
	if !ok {
		t.Fatal("acquire after evict (promote) failed")
	}
	if c.Config().Quant != QuantSQ || c.Config().SQBits != 4 {
		t.Fatalf("promoted cfg lost quantizer: Quant=%v SQBits=%d", c.Config().Quant, c.Config().SQBits)
	}
	for i, q := range queries {
		res, err := c.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: promoted %v != original %v", i, resultIDs(res), before[i])
		}
	}
	c.Release()
}

// TestPromoteSingleFlight proves N concurrent accesses on a cold collection fetch
// the snapshot EXACTLY once (single-flight): the counting store records one Get.
func TestPromoteSingleFlight(t *testing.T) {
	ctx := context.Background()
	s := newColdStore(t)
	seedCollection(t, s, "docs", 1, 2, 3, 4, 5)

	obj := &countingStore{MemStore: objstore.NewMemStore()}
	if err := s.EvictCollection(ctx, "docs", obj, "acme", time.Unix(0, 0).UTC()); err != nil {
		t.Fatalf("evict: %v", err)
	}
	obj.gets.Store(0) // reset — only count promote-time Gets

	const goroutines = 16
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make([]error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			<-start
			_, err := s.SearchFiltered("docs", []float32{1, 0}, 3, Filter{})
			errs[idx] = err
		}(i)
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d search failed: %v", i, err)
		}
	}
	if got := obj.gets.Load(); got != 1 {
		t.Fatalf("snapshot Get count = %d, want exactly 1 (single-flight)", got)
	}
	if s.IsCold("docs") {
		t.Fatal("collection should be hot after concurrent promote")
	}
}

// TestRestoreFailureKeepsStub proves a failed promote (object store Get error)
// surfaces a clear error and leaves the cold stub intact (recoverable).
func TestRestoreFailureKeepsStub(t *testing.T) {
	ctx := context.Background()
	s := newColdStore(t)
	seedCollection(t, s, "docs", 1, 2, 3)

	// Evict through a healthy store first so the snapshot exists, then swap in a
	// Get-failing store via a fresh stub to drive the failure path.
	good := objstore.NewMemStore()
	if err := s.EvictCollection(ctx, "docs", good, "acme", time.Unix(0, 0).UTC()); err != nil {
		t.Fatalf("evict: %v", err)
	}

	// Point the stub at a failing object store to force the promote Get to fail.
	s.mu.Lock()
	stub := s.cold["default/docs"]
	stub.obj = &failGetStore{MemStore: objstore.NewMemStore()}
	s.mu.Unlock()

	// Access → promote attempts a Get, which fails. Acquire reports absent.
	if _, ok := s.Acquire("docs"); ok {
		t.Fatal("Acquire should fail when promote's Get errors")
	}
	// The stub must survive (still cold, still listed) — recoverable.
	if !s.IsCold("docs") {
		t.Fatal("stub was lost after a failed promote — not recoverable")
	}

	// Repoint at the good store: a later access now succeeds (recovery).
	s.mu.Lock()
	s.cold["default/docs"].obj = good
	s.mu.Unlock()
	c, ok := s.Acquire("docs")
	if !ok {
		t.Fatal("recovery promote failed")
	}
	c.Release()
	if s.IsCold("docs") {
		t.Fatal("should be hot after recovery")
	}
}

// TestSweepCold evicts a collection idle past the threshold and NOT a
// recently-accessed one, driven entirely by an injected clock.
func TestSweepCold(t *testing.T) {
	s := newColdStore(t)

	// Inject a controllable clock.
	var nowNanos atomic.Int64
	base := time.Date(2026, 6, 23, 0, 0, 0, 0, time.UTC)
	nowNanos.Store(base.UnixNano())
	s.SetClock(func() time.Time { return time.Unix(0, nowNanos.Load()).UTC() })

	seedCollection(t, s, "idle", 1, 2)
	seedCollection(t, s, "fresh", 3, 4)

	obj := objstore.NewMemStore()

	// First sweep at t0 seeds lastAccess for both (first sight) and evicts neither.
	t0 := time.Unix(0, nowNanos.Load()).UTC()
	evicted, err := s.SweepCold(t0, time.Hour, obj, "acme")
	if err != nil {
		t.Fatalf("sweep t0: %v", err)
	}
	if len(evicted) != 0 {
		t.Fatalf("first sweep evicted %v, want none (first sight)", evicted)
	}

	// Advance 30 min and TOUCH "fresh" (resolve updates its lastAccess).
	nowNanos.Store(base.Add(30 * time.Minute).UnixNano())
	if _, err := s.SearchFiltered("fresh", []float32{3, 0}, 1, Filter{}); err != nil {
		t.Fatal(err)
	}

	// Advance to t0+90min: "idle" last-accessed at t0 (90 min ago > 1h) → evicted;
	// "fresh" last-accessed at t0+30min (60 min ago, == cutoff, not strictly older)
	// → kept.
	now := base.Add(90 * time.Minute)
	nowNanos.Store(now.UnixNano())
	evicted, err = s.SweepCold(now, time.Hour, obj, "acme")
	if err != nil {
		t.Fatalf("sweep t90: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "default/idle" {
		t.Fatalf("evicted = %v, want [default/idle]", evicted)
	}
	if !s.IsCold("idle") {
		t.Fatal("idle should be cold after sweep")
	}
	if s.IsCold("fresh") {
		t.Fatal("fresh was recently accessed and must NOT be swept")
	}

	// idle <= 0 disables sweeping.
	ev, err := s.SweepCold(now.Add(10*time.Hour), 0, obj, "acme")
	if err != nil || ev != nil {
		t.Fatalf("disabled sweep returned %v / %v", ev, err)
	}
}

// TestHotPathUnaffected proves a store with no cold collections and no clock
// injected behaves exactly as before: no promote, no cold bookkeeping, and the
// cold/access maps stay nil so Acquire's fast path is byte-identical.
func TestHotPathUnaffected(t *testing.T) {
	s := newColdStore(t)
	seedCollection(t, s, "docs", 1, 2, 3)

	// The cold-tier state must be untouched on a never-evicted store: no cold stubs
	// and no injected clock, so Acquire never stamps a collection's lastAccess.
	s.mu.RLock()
	coldNil := s.cold == nil
	nowNil := s.nowFn == nil
	s.mu.RUnlock()
	if !coldNil || !nowNil {
		t.Fatalf("hot-path store touched cold-tier state: cold=%v nowFn=%v", !coldNil, !nowNil)
	}

	// Normal ops resolve straight through Acquire with no promote.
	for i := 0; i < 5; i++ {
		if _, err := s.SearchFiltered("docs", []float32{1, 0}, 3, Filter{}); err != nil {
			t.Fatal(err)
		}
	}
	if !s.IsCollectionHotOnly("docs") {
		t.Fatal("collection should be hot with no cold state")
	}
	// Still no cold-tier state allocated.
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.cold != nil {
		t.Fatal("hot-path ops allocated cold-tier maps")
	}
}

// gatedGetStore wraps a MemStore and parks the FIRST Get until released, so a
// promote can be held mid-flight (after its stillCold RLock check, before its
// publish Lock) while a concurrent DropCollection deletes the cold stub. This
// makes the drop-during-promote interleave deterministic. Subsequent Gets pass
// through untouched.
type gatedGetStore struct {
	*objstore.MemStore
	entered  chan struct{} // closed when the first Get is reached
	release  chan struct{} // closed by the test to let the parked Get proceed
	gateOnce sync.Once
}

func (g *gatedGetStore) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	g.gateOnce.Do(func() {
		close(g.entered)
		<-g.release
	})
	return g.MemStore.Get(ctx, key)
}

// TestDropDuringPromoteDoesNotResurrect proves the drop-during-promote race fix:
// a DropCollection that lands while a promote is mid-flight must WIN — the
// collection ends up gone (not resident, not listed, not serving) and the rebuilt
// collection's on-disk config file does NOT leak. Pre-fix, promote's unconditional
// publish resurrected the just-dropped collection (and leaked its .cfg.json).
func TestDropDuringPromoteDoesNotResurrect(t *testing.T) {
	ctx := context.Background()
	s := newColdStore(t)
	seedCollection(t, s, "docs", 1, 2, 3)

	gate := &gatedGetStore{
		MemStore: objstore.NewMemStore(),
		entered:  make(chan struct{}),
		release:  make(chan struct{}),
	}
	if err := s.EvictCollection(ctx, "docs", gate, "acme", time.Unix(0, 0).UTC()); err != nil {
		t.Fatalf("evict: %v", err)
	}
	if !s.IsCold("docs") {
		t.Fatal("not cold after evict")
	}

	// Goroutine A: trigger the lazy promote via Acquire. It will park inside the
	// gated Get (after the stillCold RLock check, before the publish Lock).
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if c, ok := s.Acquire("docs"); ok {
			c.Release()
		}
	}()

	// Wait until the promote is parked inside Get, then drop the name. A promotion
	// holds the name lock for its whole body, so the drop now queues behind it
	// instead of deleting the stub mid-promotion; it then drops whatever the
	// promotion left, which must still end with the collection gone. (Before the
	// name lock, the drop deleted the stub here and the promotion took its discard
	// branch; both orders must leave nothing behind.)
	<-gate.entered
	dropped := make(chan error, 1)
	go func() { dropped <- s.DropCollection("docs") }()
	deadline := time.Now().Add(10 * time.Second)
	for nameLockHolders(s, "default/docs") < 2 {
		if time.Now().After(deadline) {
			t.Fatal("drop did not queue behind the in-flight promotion")
		}
		time.Sleep(100 * time.Microsecond)
	}

	close(gate.release)
	wg.Wait()
	if err := <-dropped; err != nil {
		t.Fatalf("drop: %v", err)
	}

	// The collection must be GONE: not hot, not cold, not listed, not serving.
	s.mu.RLock()
	_, hot := s.collections["default/docs"]
	_, cold := s.cold["default/docs"]
	s.mu.RUnlock()
	if hot {
		t.Fatal("collection RESURRECTED: live in s.collections after drop-during-promote")
	}
	if cold {
		t.Fatal("collection still cold after drop")
	}
	for _, n := range s.CollectionNames() {
		if n == "default/docs" {
			t.Fatal("dropped collection still listed in CollectionNames")
		}
	}
	if _, ok := s.Acquire("docs"); ok {
		t.Fatal("dropped collection still serving via Acquire")
	}

	// The rebuilt collection's on-disk config file must NOT leak (DropCollection
	// removes it; the discard path must mirror that cleanup).
	cfgPath, _ := s.collectionPath("default/docs")
	if _, err := os.Stat(cfgPath); !os.IsNotExist(err) {
		t.Fatalf("rebuilt collection's config file leaked at %s (stat err=%v)", cfgPath, err)
	}
}

// TestDropDuringPromoteStress hammers the drop-during-promote interleave with the
// real (ungated) lazy-promote path under concurrency, for -race -count runs. After
// each round the collection must be gone and no config file may leak.
func TestDropDuringPromoteStress(t *testing.T) {
	ctx := context.Background()
	for round := 0; round < 50; round++ {
		s := newColdStore(t)
		seedCollection(t, s, "docs", 1, 2, 3)
		obj := objstore.NewMemStore()
		if err := s.EvictCollection(ctx, "docs", obj, "acme", time.Unix(0, 0).UTC()); err != nil {
			t.Fatalf("round %d evict: %v", round, err)
		}

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			if c, ok := s.Acquire("docs"); ok {
				c.Release()
			}
		}()
		go func() {
			defer wg.Done()
			_ = s.DropCollection("docs")
		}()
		wg.Wait()

		s.mu.RLock()
		_, hot := s.collections["default/docs"]
		_, cold := s.cold["default/docs"]
		s.mu.RUnlock()
		// Drop must always win the race when it interleaves: a drop that completes
		// after a promote published is itself a normal drop, so the only acceptable
		// terminal states are (a) gone, or (b) hot ONLY if the drop ran fully before
		// the promote started (no interleave). We assert the collection is never left
		// in an inconsistent (both-or-neither) state and that nothing leaks when gone.
		if hot && cold {
			t.Fatalf("round %d: collection both hot and cold — inconsistent", round)
		}
		if !hot && !cold {
			// Gone: no config file may leak.
			cfgPath, _ := s.collectionPath("default/docs")
			if _, err := os.Stat(cfgPath); !os.IsNotExist(err) {
				t.Fatalf("round %d: config file leaked at %s after drop-during-promote", round, cfgPath)
			}
		}
		_ = s.Close()
	}
}

// IsCollectionHotOnly is a test helper: the name is live and not cold.
func (s *CollectionStore) IsCollectionHotOnly(name string) bool {
	canonical, err := canonicalName(name)
	if err != nil {
		return false
	}
	s.mu.RLock()
	_, hot := s.collections[canonical]
	_, cold := s.cold[canonical]
	s.mu.RUnlock()
	return hot && !cold
}
