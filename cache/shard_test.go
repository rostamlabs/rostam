// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"sync"
	"testing"
	"time"
)

func newTestShard(t *testing.T, cfg Config) *shard {
	t.Helper()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("invalid config: %v", err)
	}
	s, err := newShard(cfg, "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	return s
}

func TestShardPutGetRoundtrip(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	if err := s.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Get([]byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Get: %q want %q", got, "v")
	}
}

func TestShardGetMissing(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	if _, err := s.Get([]byte("missing")); err != ErrNotFound {
		t.Fatalf("Get missing: %v want ErrNotFound", err)
	}
}

func TestShardDel(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	_ = s.Put([]byte("k"), []byte("v"), 0)
	if ok, err := s.Del([]byte("k")); err != nil || !ok {
		t.Fatal("Del existing should return true")
	}
	if ok, err := s.Del([]byte("k")); err != nil || ok {
		t.Fatal("Del missing should return false")
	}
	if _, err := s.Get([]byte("k")); err != ErrNotFound {
		t.Fatal("after Del, Get must be ErrNotFound")
	}
}

func TestShardLazyPageAllocation(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	if got := s.numPages(); got != 0 {
		t.Fatalf("fresh shard: numPages=%d, want 0", got)
	}
	_ = s.Put([]byte("k"), []byte("v"), 0)
	if got := s.numPages(); got != 1 {
		t.Fatalf("after first Put: numPages=%d, want 1", got)
	}
}

func TestShardPageGrowthUnderLoad(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PageSize = 1 << 20          // 1 MiB
	cfg.MaxMemoryPerShard = 4 << 20 // 4 MiB → 4 pages max
	cfg.NumShards = 1
	s := newTestShard(t, cfg)

	// Write enough entries to span multiple pages.
	val := bytes.Repeat([]byte("X"), 4096)
	for i := range 300 {
		k := fmt.Appendf(nil, "k%05d", i)
		if err := s.Put(k, val, 0); err != nil {
			t.Fatalf("Put #%d: %v", i, err)
		}
	}
	if s.numPages() < 2 {
		t.Fatalf("expected multiple pages allocated, got %d", s.numPages())
	}
	if s.numPages() > cfg.MaxPagesPerShard() {
		t.Fatalf("page count %d exceeds cap %d", s.numPages(), cfg.MaxPagesPerShard())
	}
}

func TestShardAtCapEvicts(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // single page
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.NumShards = 1
	s := newTestShard(t, cfg)

	val := bytes.Repeat([]byte("X"), 4096)
	for i := range 500 {
		k := fmt.Appendf(nil, "k%05d", i)
		if err := s.Put(k, val, 0); err != nil {
			t.Fatalf("Put #%d: %v (must succeed under ringbuf evict)", i, err)
		}
	}
	// Oldest keys should be evicted; newest must still be present.
	if _, err := s.Get([]byte("k00000")); err != ErrNotFound {
		t.Errorf("k00000 should be evicted, got err=%v", err)
	}
	if _, err := s.Get([]byte("k00499")); err != nil {
		t.Errorf("k00499 should still be present, got err=%v", err)
	}
}

func TestShardAtCapRejectPolicy(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.NumShards = 1
	s := newTestShard(t, cfg)

	val := bytes.Repeat([]byte("X"), 4096)
	wrote := 0
	for i := range 500 {
		k := fmt.Appendf(nil, "k%05d", i)
		if err := s.Put(k, val, 0); err != nil {
			if err != ErrFull {
				t.Fatalf("Put #%d returned %v, want ErrFull", i, err)
			}
			break
		}
		wrote++
	}
	if wrote < 100 || wrote >= 500 {
		t.Fatalf("expected partial fill (100..499), got %d", wrote)
	}
}

func TestShardRingbufEvictionIsFIFOAcrossPages(t *testing.T) {
	// A 3-page ringbuf shard. Values are sized so each page holds exactly 5
	// entries, so the first 15 entries fill all three pages before any eviction.
	cfg := DefaultConfig()
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 3 << 20 // 3 pages
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.NumShards = 1
	s := newTestShard(t, cfg)

	const perPage = 5
	const pages = 3
	val := bytes.Repeat([]byte("X"), 200000) // entry ≈ 200027 B → 5 per 1 MiB page
	key := func(i int) []byte { return fmt.Appendf(nil, "k%08d", i) }

	// Fill exactly all three pages; no eviction should have happened yet.
	for i := range perPage * pages {
		if err := s.Put(key(i), val, 0); err != nil {
			t.Fatalf("Put #%d: %v", i, err)
		}
	}
	if got := s.numPages(); got != pages {
		t.Fatalf("after fill: numPages=%d, want %d", got, pages)
	}
	if ev := s.evictions.Load(); ev != 0 {
		t.Fatalf("no eviction expected while filling, got %d", ev)
	}

	// The next Put must trigger eviction of the OLDEST page (page 0, keys 0..4).
	if err := s.Put(key(perPage*pages), val, 0); err != nil {
		t.Fatalf("Put that should evict: %v", err)
	}
	if s.evictions.Load() == 0 {
		t.Fatal("expected eviction after all pages full")
	}

	// Drive well past a single full wrap.
	for i := perPage*pages + 1; i < 1000; i++ {
		if err := s.Put(key(i), val, 0); err != nil {
			t.Fatalf("Put #%d: %v", i, err)
		}
	}

	// FIFO across pages: entries from the first fill — including page 1 (keys
	// 5..9) and page 2 (keys 10..14) — must all be evicted, not just page 0's.
	// The bug (always draining the lowest-index page) freezes pages 1 and 2
	// forever, so keys 5..14 would survive.
	for _, i := range []int{0, 5, 9, 10, 14, 500} {
		if _, err := s.Get(key(i)); err != ErrNotFound {
			t.Errorf("key %d should be evicted (FIFO), got err=%v", i, err)
		}
	}
	// The most recent key is always present.
	if _, err := s.Get(key(999)); err != nil {
		t.Errorf("newest key 999 should be present, got err=%v", err)
	}
}

func TestHeapRejectPolicyScansPreallocatedPages(t *testing.T) {
	// Heap mode with InitialPagesPerShard > 1 and PolicyRejectWrites: writes must
	// fill every preallocated page before rejecting, not just the writeIdx page.
	cfg := DefaultConfig()
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 3 << 20 // 3 pages max
	cfg.InitialPagesPerShard = 3    // preallocate all three
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.NumShards = 1
	s := newTestShard(t, cfg)

	val := bytes.Repeat([]byte("X"), 4096) // ≈ 254 entries per 1 MiB page
	wrote := 0
	full := false
	for i := range 5000 {
		if err := s.Put(fmt.Appendf(nil, "k%08d", i), val, 0); err != nil {
			if err != ErrFull {
				t.Fatalf("Put #%d: %v, want ErrFull", i, err)
			}
			full = true
			break
		}
		wrote++
	}
	if !full {
		t.Fatal("expected ErrFull once all pages fill")
	}
	// One page holds ~254 entries; a correct all-pages scan fills all three
	// (~760). The single-page bug rejected after ~254.
	if wrote < 500 {
		t.Fatalf("only wrote %d entries; preallocated pages 0..1 were not used", wrote)
	}
}

func TestShardTTLLazyExpire(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	if err := s.Put([]byte("k"), []byte("v"), 10*time.Millisecond); err != nil {
		t.Fatalf("Put: %v", err)
	}
	time.Sleep(30 * time.Millisecond)
	if _, err := s.Get([]byte("k")); err != ErrNotFound {
		t.Fatalf("expired Get: %v, want ErrNotFound", err)
	}
}

func TestShardOverwrite(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	_ = s.Put([]byte("k"), []byte("v1"), 0)
	_ = s.Put([]byte("k"), []byte("v2"), 0)
	got, _ := s.Get([]byte("k"))
	if !bytes.Equal(got, []byte("v2")) {
		t.Errorf("overwrite: got %q, want v2", got)
	}
}

func TestShardConcurrentPutGet(t *testing.T) {
	s := newTestShard(t, DefaultConfig())
	const workers = 8
	const iters = 1000

	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range iters {
				k := fmt.Appendf(nil, "w%02d-k%05d", w, i)
				v := fmt.Appendf(nil, "v%05d", i)
				if err := s.Put(k, v, 0); err != nil {
					t.Errorf("Put: %v", err)
					return
				}
				got, err := s.Get(k)
				if err != nil {
					t.Errorf("Get: %v", err)
					return
				}
				if !bytes.Equal(got, v) {
					t.Errorf("got %q, want %q", got, v)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

func TestShardDelDoesNotAffectColliding(t *testing.T) {
	// Force a hash collision by manually inserting two entries at the same
	// index hash. We can't easily make xxhash collide on small inputs, so we
	// simulate the situation: Put a key, then directly verify that Del on a
	// DIFFERENT key that happens to live in the same map slot would NOT
	// remove the stored one.
	//
	// Since we can't synthesize a real xxhash collision cheaply, we test the
	// key-equality guard another way: Put key "A" with value "vA", then call
	// Del("B"). Del must return false and Get("A") must still succeed.
	s := newTestShard(t, DefaultConfig())
	if err := s.Put([]byte("A"), []byte("vA"), 0); err != nil {
		t.Fatalf("Put A: %v", err)
	}
	if ok, err := s.Del([]byte("B")); err != nil || ok {
		t.Fatal("Del(B) returned true; expected false (key not present)")
	}
	if got, err := s.Get([]byte("A")); err != nil || !bytes.Equal(got, []byte("vA")) {
		t.Fatalf("after Del(B): Get(A) = %q,%v; want vA,nil", got, err)
	}
}
