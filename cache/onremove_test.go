// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// removeRecorder captures every key the onRemove hook fires with. It COPIES the
// key (string(key)) exactly as a real posting index must — the key aliases the
// page backing store and is only valid for the duration of the call.
type removeRecorder struct {
	mu   sync.Mutex
	keys []string
}

func (r *removeRecorder) hook(key []byte) {
	r.mu.Lock()
	r.keys = append(r.keys, string(key))
	r.mu.Unlock()
}

func (r *removeRecorder) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.keys))
	copy(out, r.keys)
	return out
}

func (r *removeRecorder) count(key string) int {
	n := 0
	for _, k := range r.snapshot() {
		if k == key {
			n++
		}
	}
	return n
}

func (r *removeRecorder) len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.keys)
}

// onRemoveTestConfig is a single-shard heap cache with the background sweeper
// OFF, so every test drives reclamation explicitly.
func onRemoveTestConfig() Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0
	return cfg
}

func newOnRemoveCache(t *testing.T, cfg Config) (*Cache, *removeRecorder) {
	t.Helper()
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	rec := &removeRecorder{}
	c.SetOnRemove(rec.hook)
	return c, rec
}

func TestOnRemoveFiresOnDelete(t *testing.T) {
	t.Run("heap", func(t *testing.T) {
		c, rec := newOnRemoveCache(t, onRemoveTestConfig())
		if err := c.Put([]byte("k"), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		ok, err := c.Del([]byte("k"))
		if err != nil || !ok {
			t.Fatalf("Del: ok=%v err=%v", ok, err)
		}
		if got := rec.snapshot(); len(got) != 1 || got[0] != "k" {
			t.Fatalf("onRemove keys = %v, want [k]", got)
		}
	})

	t.Run("mmap durable tombstone", func(t *testing.T) {
		// The temp dir must be allocated BEFORE the cache's Close cleanup is
		// registered: t.Cleanup is LIFO, so the dir is removed only after Close.
		dir := t.TempDir()
		cfg := onRemoveTestConfig()
		cfg.DataDir = dir
		cfg.PageSize = 1 << 20
		cfg.MaxMemoryPerShard = 4 << 20
		c, rec := newOnRemoveCache(t, cfg)
		if err := c.Put([]byte("k"), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		if ok, err := c.Del([]byte("k")); err != nil || !ok {
			t.Fatalf("Del: ok=%v err=%v", ok, err)
		}
		if got := rec.snapshot(); len(got) != 1 || got[0] != "k" {
			t.Fatalf("onRemove keys = %v, want [k]", got)
		}
	})

	t.Run("absent key fires nothing", func(t *testing.T) {
		c, rec := newOnRemoveCache(t, onRemoveTestConfig())
		if ok, _ := c.Del([]byte("nope")); ok {
			t.Fatal("Del of an absent key returned true")
		}
		if n := rec.len(); n != 0 {
			t.Fatalf("onRemove fired %d times for an absent key, want 0", n)
		}
	})
}

func TestOnRemoveFiresOnExpirySweep(t *testing.T) {
	var clock atomic.Uint64
	clock.Store(1_000_000)
	cfg := onRemoveTestConfig()
	cfg.NowFn = clock.Load
	c, rec := newOnRemoveCache(t, cfg)

	if err := c.Put([]byte("ttl"), []byte("v"), 50*time.Millisecond); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := c.Put([]byte("keep"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Not yet expired: a sweep must fire nothing.
	c.shards[0].sweepOnce()
	if n := rec.len(); n != 0 {
		t.Fatalf("onRemove fired %d times before expiry, want 0", n)
	}
	clock.Store(1_000_000 + 51)
	c.shards[0].sweepOnce()
	if got := rec.snapshot(); len(got) != 1 || got[0] != "ttl" {
		t.Fatalf("onRemove keys = %v, want [ttl]", got)
	}
}

// TestOnRemoveFiresOnExpiryLazyDrop covers dropExpiredLocked — the lazy removal a
// READ of an expired key performs on a non-replicated shard.
func TestOnRemoveFiresOnExpiryLazyDrop(t *testing.T) {
	var clock atomic.Uint64
	clock.Store(2_000_000)
	cfg := onRemoveTestConfig()
	cfg.NowFn = clock.Load
	c, rec := newOnRemoveCache(t, cfg)

	if err := c.Put([]byte("lazy"), []byte("v"), 10*time.Millisecond); err != nil {
		t.Fatalf("Put: %v", err)
	}
	clock.Store(2_000_000 + 11)
	if _, err := c.Get([]byte("lazy")); err != ErrNotFound {
		t.Fatalf("Get expired: %v, want ErrNotFound", err)
	}
	if got := rec.snapshot(); len(got) != 1 || got[0] != "lazy" {
		t.Fatalf("onRemove keys = %v, want [lazy]", got)
	}
}

// evictionConfig is a two-page ringbuf shard: small enough that a handful of
// large entries forces eviction.
func evictionConfig() Config {
	cfg := onRemoveTestConfig()
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 2 << 20
	cfg.AtCapPolicy = PolicyRingbufEvict
	return cfg
}

func TestOnRemoveFiresOnEviction(t *testing.T) {
	run := func(t *testing.T, cfg Config) {
		t.Helper()
		c, rec := newOnRemoveCache(t, cfg)
		s := c.shards[0]
		val := make([]byte, 64<<10)
		written := map[string]bool{}
		for i := 0; i < 200 && s.evictions.Load() == 0; i++ {
			k := fmt.Appendf(nil, "e%04d", i)
			if err := c.Put(k, val, 0); err != nil {
				t.Fatalf("Put %d: %v", i, err)
			}
			written[string(k)] = true
		}
		if s.evictions.Load() == 0 {
			t.Fatal("no eviction happened; test config is wrong")
		}
		got := rec.snapshot()
		if len(got) == 0 {
			t.Fatal("eviction removed entries but onRemove never fired")
		}
		for _, k := range got {
			if !written[k] {
				t.Fatalf("onRemove fired for %q which was never written", k)
			}
			if _, err := c.Get([]byte(k)); err == nil {
				t.Fatalf("onRemove fired for %q but it is still readable", k)
			}
		}
	}

	t.Run("heap retire", func(t *testing.T) {
		run(t, evictionConfig())
	})

	t.Run("mmap drain", func(t *testing.T) {
		dir := t.TempDir()
		cfg := evictionConfig()
		cfg.DataDir = dir
		run(t, cfg)
	})
}

// TestOnRemoveSkipsSupersededDuplicates is the cur == ref proof. A page walked by
// eviction contains DEAD DUPLICATES of keys whose index slot already points at a
// newer live copy on another page. Firing the hook per walked entry (instead of
// inside the `ok && cur == ref` guard) would drop the posting of a key that is
// still live — a missing row, the one failure verify-on-read cannot repair.
func TestOnRemoveSkipsSupersededDuplicates(t *testing.T) {
	c, rec := newOnRemoveCache(t, evictionConfig())
	s := c.shards[0]
	val := make([]byte, 64<<10)
	newVal := bytes.Repeat([]byte{0xAB}, 64<<10)

	// 1. K's FIRST physical copy lands on page 0.
	if err := c.Put([]byte("K"), val, 0); err != nil {
		t.Fatalf("Put K: %v", err)
	}
	// 2. Fill page 0 until a second page is allocated.
	filler := 0
	for {
		s.mu.RLock()
		npages := len(s.pages)
		s.mu.RUnlock()
		if npages >= 2 {
			break
		}
		if err := c.Put(fmt.Appendf(nil, "f%04d", filler), val, 0); err != nil {
			t.Fatalf("filler Put: %v", err)
		}
		filler++
		if filler > 1000 {
			t.Fatal("page 1 was never allocated")
		}
	}
	// 3. Overwrite K. The new copy lands on page 1; the page-0 copy is now a dead
	//    duplicate whose index slot points elsewhere.
	if err := c.Put([]byte("K"), newVal, 0); err != nil {
		t.Fatalf("overwrite K: %v", err)
	}
	if n := rec.count("K"); n != 0 {
		t.Fatalf("onRemove fired %d times for an overwrite, want 0", n)
	}
	// 4. Fill page 1 until the next write must evict page 0 (which still frames
	//    K's dead duplicate).
	for s.evictions.Load() == 0 {
		if err := c.Put(fmt.Appendf(nil, "f%04d", filler), val, 0); err != nil {
			t.Fatalf("filler Put: %v", err)
		}
		filler++
		if filler > 2000 {
			t.Fatal("eviction never happened")
		}
	}

	if n := rec.count("K"); n != 0 {
		t.Fatalf("onRemove fired %d times for K, whose live copy survived eviction; "+
			"the hook is not inside the cur == ref guard", n)
	}
	if rec.len() == 0 {
		t.Fatal("eviction drained a page but onRemove never fired for its dead keys")
	}
	got, err := c.Get([]byte("K"))
	if err != nil {
		t.Fatalf("K must still be live after its stale duplicate was evicted: %v", err)
	}
	if !bytes.Equal(got, newVal) {
		t.Fatal("K resolved to the stale copy")
	}
}

func TestOnRemoveNotFiredOnOverwrite(t *testing.T) {
	c, rec := newOnRemoveCache(t, onRemoveTestConfig())
	for i := range 5 {
		if err := c.Put([]byte("k"), fmt.Appendf(nil, "v%d", i), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	if n := rec.len(); n != 0 {
		t.Fatalf("onRemove fired %d times for overwrites, want 0", n)
	}
}

func TestOnRemoveNotFiredOnFlush(t *testing.T) {
	c, rec := newOnRemoveCache(t, onRemoveTestConfig())
	for i := range 50 {
		if err := c.Put(fmt.Appendf(nil, "k%02d", i), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if n := rec.len(); n != 0 {
		t.Fatalf("onRemove fired %d times for a flush, want 0 (the flush handler "+
			"calls Set.Reset instead)", n)
	}
	if _, err := c.Get([]byte("k00")); err != ErrNotFound {
		t.Fatalf("after Flush: %v, want ErrNotFound", err)
	}
}

// TestOnRemoveSkippedOnCorruptSlot pins the documented gap: a slot whose
// page.Read errors carries NO KEY, so the sweeper tombstones it without firing
// the hook. The reconcile pass is the backstop for that posting.
func TestOnRemoveSkippedOnCorruptSlot(t *testing.T) {
	cfg := onRemoveTestConfig()
	cfg.PageSize = 1 << 20 // small pages, so a 1 MiB value length overruns one
	cfg.MaxMemoryPerShard = 2 << 20
	c, rec := newOnRemoveCache(t, cfg)
	s := c.shards[0]
	key := []byte("corrupt")
	if err := c.Put(key, []byte("value"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Corrupt the entry's framed value length so decodeEntryFast reports a
	// truncated entry (a length larger than the page, but far below the int32
	// ceiling so the decode stays sane on 32-bit).
	s.mu.Lock()
	tab := s.tab.Load()
	_, ref, ok := tab.findSlot(hashKey(key))
	if !ok {
		s.mu.Unlock()
		t.Fatal("key not indexed")
	}
	entries := s.pages[ref.pageIdx()].entries()
	binary.LittleEndian.PutUint32(entries[int(ref.offset())+2:], 1<<20)
	s.mu.Unlock()

	s.sweepOnce()

	if n := rec.len(); n != 0 {
		t.Fatalf("onRemove fired %d times for a corrupt (keyless) slot, want 0", n)
	}
	if _, err := c.Get(key); err != ErrNotFound {
		t.Fatalf("corrupt slot still resolves: %v", err)
	}
}

// TestOnRemoveKeyIsCopyable asserts the key handed to the hook is a usable page
// alias for the duration of the call and survives being copied out, even after
// the page bytes behind it are reused.
func TestOnRemoveKeyIsCopyable(t *testing.T) {
	c, _ := newOnRemoveCache(t, evictionConfig())
	var copies []string
	var mu sync.Mutex
	c.SetOnRemove(func(key []byte) {
		mu.Lock()
		copies = append(copies, string(key)) // the copy every real index must make
		mu.Unlock()
	})
	s := c.shards[0]
	val := make([]byte, 64<<10)
	for i := 0; i < 200 && s.evictions.Load() == 0; i++ {
		if err := c.Put(fmt.Appendf(nil, "copy%04d", i), val, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	mu.Lock()
	first := append([]string(nil), copies...)
	mu.Unlock()
	if len(first) == 0 {
		t.Fatal("no eviction fired the hook")
	}
	// Churn the pages hard so the bytes the keys aliased are certainly reused.
	for i := range 200 {
		if err := c.Put(fmt.Appendf(nil, "churn%04d", i), val, 0); err != nil {
			t.Fatalf("churn Put: %v", err)
		}
	}
	for i, k := range first {
		if want := fmt.Sprintf("copy%04d", i); k != want {
			t.Fatalf("copied key %d = %q, want %q (the copy did not survive page reuse)", i, k, want)
		}
	}
}

func TestOnRemoveNilIsSafe(t *testing.T) {
	cfg := evictionConfig()
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	// No hook installed at all.
	val := make([]byte, 64<<10)
	for i := 0; i < 60; i++ {
		if err := c.Put(fmt.Appendf(nil, "n%04d", i), val, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	if _, err := c.Del([]byte("n0059")); err != nil {
		t.Fatalf("Del: %v", err)
	}
	c.shards[0].sweepOnce()
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// Install then clear: nothing fires after the clear.
	rec := &removeRecorder{}
	c.SetOnRemove(rec.hook)
	if err := c.Put([]byte("x"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := c.Del([]byte("x")); err != nil {
		t.Fatalf("Del: %v", err)
	}
	if rec.len() != 1 {
		t.Fatalf("hook fired %d times while installed, want 1", rec.len())
	}
	c.SetOnRemove(nil)
	if err := c.Put([]byte("y"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := c.Del([]byte("y")); err != nil {
		t.Fatalf("Del: %v", err)
	}
	if rec.len() != 1 {
		t.Fatalf("hook fired %d times after SetOnRemove(nil), want 1", rec.len())
	}
}

// TestOnRemoveNilCostsNothing pins the "nil hook is zero-cost" claim: the removal
// paths allocate nothing extra when no hook is installed, and installing a
// non-allocating hook adds no allocation either.
func TestOnRemoveNilCostsNothing(t *testing.T) {
	cfg := onRemoveTestConfig()
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	key := []byte("alloc")
	val := []byte("v")
	cycle := func() {
		_ = c.Put(key, val, 0)
		_, _ = c.Del(key)
	}
	noHook := testing.AllocsPerRun(200, cycle)
	if noHook != 0 {
		t.Fatalf("put+del with no hook allocates %v times, want 0", noHook)
	}
	var sink int
	c.SetOnRemove(func(k []byte) { sink += len(k) })
	withHook := testing.AllocsPerRun(200, cycle)
	if withHook != noHook {
		t.Fatalf("put+del allocates %v with a hook vs %v without", withHook, noHook)
	}
	if sink == 0 {
		t.Fatal("hook never ran")
	}
}

// TestOnRemoveEvictionRaceWithWriter runs the eviction path under a concurrent
// writer; meaningful under -race.
func TestOnRemoveEvictionRaceWithWriter(t *testing.T) {
	cfg := evictionConfig()
	cfg.NumShards = 4
	c, rec := newOnRemoveCache(t, cfg)
	val := make([]byte, 32<<10)
	n := 400
	if testing.Short() {
		n = 80
	}
	var wg sync.WaitGroup
	for w := range 4 {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range n {
				k := fmt.Appendf(nil, "w%d-%05d", w, i)
				if err := c.Put(k, val, 0); err != nil {
					t.Errorf("Put: %v", err)
					return
				}
				if i%7 == 0 {
					if _, err := c.Del(k); err != nil {
						t.Errorf("Del: %v", err)
						return
					}
				}
			}
		}(w)
	}
	wg.Wait()
	if rec.len() == 0 {
		t.Fatal("no removals recorded under concurrent eviction")
	}
}

// ---------------------------------------------------------------- IterateChunked

func TestIterateChunkedVisitsEverything(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 4
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const n = 5000
	for i := range n {
		if err := c.Put(fmt.Appendf(nil, "k%06d", i), fmt.Appendf(nil, "v%06d", i), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	want := map[string]string{}
	c.Iterate(func(k, v []byte, _ uint64) bool {
		want[string(k)] = string(v)
		return true
	})
	if len(want) != n {
		t.Fatalf("Iterate saw %d keys, want %d", len(want), n)
	}
	got := map[string]string{}
	c.IterateChunked(64, func(k, v []byte) bool {
		got[string(k)] = string(v)
		return true
	})
	if len(got) != len(want) {
		t.Fatalf("IterateChunked saw %d keys, Iterate saw %d", len(got), len(want))
	}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("IterateChunked[%q] = %q, want %q", k, got[k], v)
		}
	}
}

func TestIterateChunkedStopsWhenFnReturnsFalse(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 2
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	for i := range 500 {
		_ = c.Put(fmt.Appendf(nil, "k%04d", i), []byte("v"), 0)
	}
	count := 0
	c.IterateChunked(16, func(_, _ []byte) bool {
		count++
		return count < 10
	})
	if count != 10 {
		t.Fatalf("visited %d entries after stopping at 10", count)
	}
}

func TestIterateChunkedSkipsExpired(t *testing.T) {
	var clock atomic.Uint64
	clock.Store(5_000_000)
	cfg := onRemoveTestConfig()
	cfg.NowFn = clock.Load
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	_ = c.Put([]byte("live"), []byte("v"), 0)
	_ = c.Put([]byte("dead"), []byte("v"), 10*time.Millisecond)
	clock.Store(5_000_000 + 11)
	var seen []string
	c.IterateChunked(4, func(k, _ []byte) bool {
		seen = append(seen, string(k))
		return true
	})
	if len(seen) != 1 || seen[0] != "live" {
		t.Fatalf("IterateChunked saw %v, want [live]", seen)
	}
}

// TestIterateChunkedLetsWritesThrough is the reason IterateChunked exists: a plain
// Iterate holds one shard's RLock for that shard's ENTIRE walk, so every write to
// that shard stalls for the whole walk. The chunked walk releases the lock every
// `batch` slots, so a concurrent writer makes progress in EVERY 10 ms sample.
func TestIterateChunkedLetsWritesThrough(t *testing.T) {
	n := 100_000
	if testing.Short() {
		n = 10_000
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	for i := range n {
		if err := c.Put(fmt.Appendf(nil, "k%08d", i), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}

	// The writer OVERWRITES existing keys, so the index table never grows and the
	// walk is never restarted by a rehash — this test measures write latency, not
	// restart behaviour.
	var writes atomic.Int64
	stop := make(chan struct{})
	var writerWG sync.WaitGroup
	writerWG.Add(1)
	go func() {
		defer writerWG.Done()
		i := 0
		for {
			select {
			case <-stop:
				return
			default:
			}
			if err := c.Put(fmt.Appendf(nil, "k%08d", i%n), []byte("w"), 0); err != nil {
				return
			}
			writes.Add(1)
			i++
		}
	}()

	var sampleMu sync.Mutex
	var samples []int64
	sampleDone := make(chan struct{})
	samplerStopped := make(chan struct{})
	go func() {
		defer close(samplerStopped)
		prev := writes.Load()
		tick := time.NewTicker(10 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-sampleDone:
				return
			case <-tick.C:
				cur := writes.Load()
				sampleMu.Lock()
				samples = append(samples, cur-prev)
				sampleMu.Unlock()
				prev = cur
			}
		}
	}()

	const batch = 256
	visited := 0
	start := time.Now()
	c.IterateChunked(batch, func(_, _ []byte) bool {
		visited++
		if visited%batch == 0 {
			// Make the walk long enough to span many samples. The sleep happens
			// under the chunk's read lock, so it is also the worst-case stall a
			// writer can see — bounded by one chunk, never by the whole walk.
			time.Sleep(300 * time.Microsecond)
		}
		return true
	})
	elapsed := time.Since(start)
	close(sampleDone)
	<-samplerStopped
	close(stop)
	writerWG.Wait()

	if visited < n {
		t.Fatalf("walk visited %d of %d keys", visited, n)
	}
	sampleMu.Lock()
	got := append([]int64(nil), samples...)
	sampleMu.Unlock()
	if len(got) < 3 {
		t.Fatalf("walk finished in %v with only %d samples; too short to prove anything", elapsed, len(got))
	}
	for i, d := range got {
		if d <= 0 {
			t.Fatalf("writer made no progress in 10ms sample %d of %d (deltas %v) — "+
				"the walk is blocking writes", i, len(got), got)
		}
	}
}

// TestIterateChunkedSurvivesRehash forces a table swap mid-walk. The walk must
// restart that shard from slot 0 (safe: Reindex is idempotent) and still visit
// every pre-existing key.
func TestIterateChunkedSurvivesRehash(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const n = 2000
	for i := range n {
		if err := c.Put(fmt.Appendf(nil, "p%06d", i), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	s := c.shards[0]

	rehashOnce := sync.OnceFunc(func() {
		go func() {
			s.mu.Lock()
			s.tab.Store(s.tab.Load().rehashed())
			s.mu.Unlock()
		}()
	})

	seen := map[string]int{}
	visited := 0
	c.IterateChunked(32, func(k, _ []byte) bool {
		seen[string(k)]++
		visited++
		if visited == 32 {
			rehashOnce()
		}
		// Slow the walk so the rehash lands between chunks.
		if visited%32 == 0 {
			time.Sleep(200 * time.Microsecond)
		}
		return true
	})

	if s.chunkedRestarts.Load() == 0 {
		t.Fatal("the mid-walk rehash never forced a restart")
	}
	for i := range n {
		k := fmt.Sprintf("p%06d", i)
		if seen[k] == 0 {
			t.Fatalf("key %q was never visited across the restart", k)
		}
	}
}

// TestIterateChunkedTerminatesUnderConstantRehash drives the restart budget past
// its limit: after iterateChunkedMaxRestarts the walk falls back to one locked
// Iterate, so termination is guaranteed no matter how often the table churns.
func TestIterateChunkedTerminatesUnderConstantRehash(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const n = 1000
	for i := range n {
		if err := c.Put(fmt.Appendf(nil, "r%06d", i), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	s := c.shards[0]

	done := make(chan struct{})
	rehasherStopped := make(chan struct{})
	go func() {
		defer close(rehasherStopped)
		for {
			select {
			case <-done:
				return
			default:
			}
			s.mu.Lock()
			s.tab.Store(s.tab.Load().rehashed())
			s.mu.Unlock()
			time.Sleep(50 * time.Microsecond)
		}
	}()

	seen := map[string]bool{}
	visited := 0
	c.IterateChunked(8, func(k, _ []byte) bool {
		seen[string(k)] = true
		visited++
		if visited%8 == 0 {
			time.Sleep(100 * time.Microsecond)
		}
		return true
	})
	close(done)
	<-rehasherStopped

	if got := s.chunkedRestarts.Load(); got < iterateChunkedMaxRestarts {
		t.Fatalf("chunkedRestarts = %d, want >= %d (the fallback was never exercised)",
			got, iterateChunkedMaxRestarts)
	}
	for i := range n {
		k := fmt.Sprintf("r%06d", i)
		if !seen[k] {
			t.Fatalf("key %q was never visited; the locked fallback is not complete", k)
		}
	}
}

// TestIterateChunkedRaceWithWriter exercises the chunked walk against concurrent
// writes, deletes and evictions; meaningful under -race.
func TestIterateChunkedRaceWithWriter(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 2
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 4 << 20
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.TTLSweepIntervalMs = 0
	c, rec := newOnRemoveCache(t, cfg)

	n := 20_000
	if testing.Short() {
		n = 2_000
	}
	for i := range n {
		if err := c.Put(fmt.Appendf(nil, "k%08d", i), []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := range 2 {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			i := 0
			for {
				select {
				case <-stop:
					return
				default:
				}
				k := fmt.Appendf(nil, "k%08d", (i*7+w)%n)
				if err := c.Put(k, []byte("w"), 0); err != nil {
					return
				}
				if i%5 == 0 {
					if _, err := c.Del(k); err != nil {
						return
					}
				}
				i++
			}
		}(w)
	}

	for range 3 {
		c.IterateChunked(128, func(k, v []byte) bool {
			_ = append([]byte(nil), k...)
			_ = append([]byte(nil), v...)
			return true
		})
	}
	close(stop)
	wg.Wait()
	if rec.len() == 0 {
		t.Fatal("no removals recorded during the concurrent walk")
	}
}

func benchEvictCache(b *testing.B, hook func(key []byte)) *Cache {
	b.Helper()
	cfg := evictionConfig()
	c, err := New(cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	if hook != nil {
		c.SetOnRemove(hook)
	}
	return c
}

func BenchmarkEvictionNoHook(b *testing.B) {
	c := benchEvictCache(b, nil)
	defer func() { _ = c.Close() }()
	val := make([]byte, 64<<10)
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		_ = c.Put(fmt.Appendf(nil, "b%08d", i), val, 0)
	}
}

func BenchmarkEvictionWithHook(b *testing.B) {
	var sink int
	c := benchEvictCache(b, func(k []byte) { sink += len(k) })
	defer func() { _ = c.Close() }()
	val := make([]byte, 64<<10)
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		_ = c.Put(fmt.Appendf(nil, "b%08d", i), val, 0)
	}
	if sink == 0 {
		b.Fatal("hook never ran")
	}
}

func BenchmarkIterate(b *testing.B) {
	c := benchWalkCache(b)
	defer func() { _ = c.Close() }()
	b.ResetTimer()
	for b.Loop() {
		c.Iterate(func(_, _ []byte, _ uint64) bool { return true })
	}
}

func BenchmarkIterateChunked(b *testing.B) {
	c := benchWalkCache(b)
	defer func() { _ = c.Close() }()
	b.ResetTimer()
	for b.Loop() {
		c.IterateChunked(1024, func(_, _ []byte) bool { return true })
	}
}

func benchWalkCache(b *testing.B) *Cache {
	b.Helper()
	cfg := DefaultConfig()
	cfg.NumShards = 8
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	for i := range 100_000 {
		if err := c.Put(fmt.Appendf(nil, "k%08d", i), []byte("value"), 0); err != nil {
			b.Fatalf("Put: %v", err)
		}
	}
	return c
}
