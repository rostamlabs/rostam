// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Tests for the mutable-region SUBSTRATE: the monotone mutable→sealed latch a heap
// page is born with, and the explicit FIFO of page pointers that decides which
// pages are still region members.
//
// Nothing in the cache CONSULTS the latch. These tests therefore have two jobs:
// prove the bookkeeping is what it claims to be (born mutable, sealed in FIFO
// order, never un-sealed, bounded memory), and prove that turning the sealing on
// changes NOTHING a configured cache does — including a cache with in-place
// same-size updates enabled, the one feature a mutable/sealed distinction would
// eventually gate.

// regionCfg is a single heap ringbuf shard with the sweeper off, holding at most
// `pages` pages — small enough that eviction (and therefore page retirement) is
// reachable.
func regionCfg(inPlace bool, pages int) Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 4 << 10
	cfg.MaxMemoryPerShard = pages * cfg.PageSize
	cfg.TTLSweepIntervalMs = 0
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.InPlaceSameSizeUpdate = inPlace
	return cfg
}

// newRegionShard returns a heap shard whose mutable region holds at most `bound`
// pages (0 = no bound, which is every configuration the cache ships with).
func newRegionShard(t *testing.T, bound int, inPlace bool) *shard {
	t.Helper()
	return newRegionShardPages(t, bound, inPlace, 16)
}

// newRegionShardPages is newRegionShard with room for `pages` pages, for the tests
// that allocate pages directly rather than driving writes through the shard.
func newRegionShardPages(t *testing.T, bound int, inPlace bool, pages int) *shard {
	t.Helper()
	s, err := newShard(regionCfg(inPlace, pages), "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	s.mu.Lock()
	s.mutableRegionPages = bound
	s.mu.Unlock()
	return s
}

// allocPages appends n fresh heap pages through the shard's own allocation path.
func allocPages(t *testing.T, s *shard, n int) []*page {
	t.Helper()
	out := make([]*page, 0, n)
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := 0; i < n; i++ {
		idx := s.allocHeapPageLocked()
		out = append(out, s.pages[idx])
	}
	return out
}

func regionMembers(s *shard) []*page {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]*page(nil), s.mutableRegion...)
}

// TestFreshHeapPageIsBornMutable is the base case: every heap page reaches its
// shard through freshHeapPageLocked, and every page that does is mutable.
func TestFreshHeapPageIsBornMutable(t *testing.T) {
	s := newRegionShard(t, 0, false)
	for _, p := range allocPages(t, s, 3) {
		if !p.Mutable() {
			t.Fatal("a freshly allocated heap page is not mutable")
		}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i, p := range s.pages {
		if !p.Mutable() {
			t.Fatalf("page %d is not mutable", i)
		}
	}
}

// TestRegionDisabledSealsNothing: with no bound there is no region, so no page is
// ever pushed out of one and nothing is ever sealed. This is the shipped shape.
func TestRegionDisabledSealsNothing(t *testing.T) {
	s := newRegionShardPages(t, 0, false, 64)
	pages := allocPages(t, s, 20)
	if got := s.mutablePagesSealed.Load(); got != 0 {
		t.Fatalf("sealed %d pages with the region disabled, want 0", got)
	}
	if members := regionMembers(s); len(members) != 0 {
		t.Fatalf("region holds %d pages with no bound, want 0 (nothing to track)", len(members))
	}
	for i, p := range pages {
		if !p.Mutable() {
			t.Fatalf("page %d sealed with the region disabled", i)
		}
	}
}

// TestRegionSealsOldestFirst: the FIFO decides membership by ARRIVAL ORDER, and the
// page it pushes out is the one that gets sealed.
func TestRegionSealsOldestFirst(t *testing.T) {
	s := newRegionShard(t, 2, false)
	sealedBefore := s.mutablePagesSealed.Load()
	pages := allocPages(t, s, 4)

	for i, p := range pages {
		wantMutable := i >= len(pages)-2 // the two newest are still members
		if p.Mutable() != wantMutable {
			t.Fatalf("page %d: Mutable()=%v, want %v", i, p.Mutable(), wantMutable)
		}
	}
	if got, want := s.mutablePagesSealed.Load()-sealedBefore, uint64(2); got != want {
		t.Fatalf("sealed %d pages, want %d", got, want)
	}
	members := regionMembers(s)
	if len(members) != 2 || members[0] != pages[2] || members[1] != pages[3] {
		t.Fatalf("region membership is not the two newest pages in arrival order: %v", members)
	}
}

// TestRegionMembershipIsBounded: the FIFO holds page POINTERS, so an unbounded one
// would keep every retired page's slab alive. It must never exceed its bound.
func TestRegionMembershipIsBounded(t *testing.T) {
	const bound = 3
	s := newRegionShardPages(t, bound, false, 64)
	for i := 0; i < 50; i++ {
		allocPages(t, s, 1)
		if n := len(regionMembers(s)); n > bound {
			t.Fatalf("region holds %d pages after %d births, bound is %d", n, i+1, bound)
		}
	}
	for _, p := range regionMembers(s) {
		if p == nil {
			t.Fatal("region holds a nil page")
		}
	}
}

// TestSealIsMonotone: sealing is a one-way latch. A second seal is a no-op that
// reports it did nothing (so it cannot double-count), and there is no operation
// anywhere that returns a sealed page to mutable.
func TestSealIsMonotone(t *testing.T) {
	s := newRegionShard(t, 1, false)
	p := allocPages(t, s, 1)[0]
	if !p.Mutable() {
		t.Fatal("fresh page is not mutable")
	}
	if !p.Seal() {
		t.Fatal("first Seal did not report the transition")
	}
	if p.Mutable() {
		t.Fatal("page is still mutable after Seal")
	}
	if p.Seal() {
		t.Fatal("second Seal reported a transition it did not perform")
	}
	if p.Mutable() {
		t.Fatal("page became mutable again")
	}
	// Reset is the page-reuse operation (cold compaction, corrupt-page recovery).
	// It must not resurrect mutability either.
	p.Reset()
	if p.Mutable() {
		t.Fatal("Reset returned a sealed page to mutable")
	}
}

// TestRetiredPageIsReplacedByAMutableOne is the chokepoint claim stated as behaviour:
// heap ringbuf retirement swaps in a fresh page object, and that object — like every
// other heap page — is born mutable and enters the region. The retired object keeps
// whatever state it had; it is frozen and nothing may write to it again.
func TestRetiredPageIsReplacedByAMutableOne(t *testing.T) {
	s := newRegionShard(t, 1, false)
	allocPages(t, s, 1)

	s.mu.Lock()
	idx := len(s.pages) - 1
	old := s.pages[idx]
	bornBefore := s.mutablePagesBorn.Load()
	s.retirePageLocked(idx)
	fresh := s.pages[idx]
	s.mu.Unlock()

	if fresh == old {
		t.Fatal("retirePageLocked did not replace the page object")
	}
	if !fresh.Mutable() {
		t.Fatal("the replacement page is not mutable")
	}
	if got := s.mutablePagesBorn.Load() - bornBefore; got != 1 {
		t.Fatalf("retirement produced %d region births, want 1", got)
	}
	if members := regionMembers(s); len(members) != 1 || members[0] != fresh {
		t.Fatalf("the replacement page is not the newest region member: %v", members)
	}
}

// TestExpiredPageRetirementEntersRegion covers the OTHER retirement path
// (tryRetireExpiredPageLocked), for the same reason: it must not be able to publish
// a page that skipped the bookkeeping.
func TestExpiredPageRetirementEntersRegion(t *testing.T) {
	cfg := regionCfg(false, 16)
	cfg.NowFn = func() uint64 { return 1_000 }
	s, err := newShard(cfg, "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	s.mu.Lock()
	s.mutableRegionPages = 1
	s.mu.Unlock()

	key := []byte("k")
	if err := s.putH(key, make([]byte, 64), 10*time.Millisecond, hashKey(key)); err != nil {
		t.Fatalf("put: %v", err)
	}
	s.mu.Lock()
	idx := s.writeIdx
	old := s.pages[idx]
	bornBefore := s.mutablePagesBorn.Load()
	s.tryRetireExpiredPageLocked(idx, 100_000) // well past the entry's expiry
	fresh := s.pages[idx]
	s.mu.Unlock()

	if fresh == old {
		t.Fatal("tryRetireExpiredPageLocked did not retire a page holding only expired entries")
	}
	if !fresh.Mutable() {
		t.Fatal("the replacement page is not mutable")
	}
	if got := s.mutablePagesBorn.Load() - bornBefore; got != 1 {
		t.Fatalf("expired-page retirement produced %d region births, want 1", got)
	}
}

// TestMmapPagesAreNeverRegionMembers: an mmap page is reused in place — drain keeps
// the SAME object and the same generation — so a mutable→sealed latch on it could go
// backwards. Mmap pages therefore never join the region and always read as not
// mutable, which is also the conservative answer (their bytes may never be
// overwritten where they lie).
func TestMmapPagesAreNeverRegionMembers(t *testing.T) {
	cfg := regionCfg(false, 3)
	cfg.PageSize = 1 << 20 // mmap shards refuse anything smaller
	cfg.MaxMemoryPerShard = 3 * cfg.PageSize
	cfg.DataDir = t.TempDir()
	cfg.DisableColdCompaction = true
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	s := c.shards[0]
	s.mu.Lock()
	s.mutableRegionPages = 1 // even asking for a region must not create one here
	s.mu.Unlock()

	val := make([]byte, 4<<10)
	for i := 0; i < 2_000; i++ {
		if err := c.Put([]byte(fmt.Sprintf("key-%d", i)), val, 0); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	if c.Stats().Evictions == 0 {
		t.Fatal("the mmap shard never evicted; the drain path was not exercised")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i, p := range s.pages {
		if p.Mutable() {
			t.Fatalf("mmap page %d reports itself mutable", i)
		}
	}
	if got := s.mutablePagesBorn.Load(); got != 0 {
		t.Fatalf("mmap shard recorded %d region births, want 0", got)
	}
	if got := s.mutablePagesSealed.Load(); got != 0 {
		t.Fatalf("mmap shard sealed %d pages, want 0", got)
	}
	if len(s.mutableRegion) != 0 {
		t.Fatalf("mmap shard has %d region members, want 0", len(s.mutableRegion))
	}
}

// TestSealingChangesNothing is the no-op evidence. The same workload runs on a shard
// sealing as aggressively as the substrate allows (a one-page region, so every page
// but the newest is sealed) and on one with the region disabled. Every counter the
// cache reports must match, in-place updates included: nothing reads the latch, so
// nothing may move.
func TestSealingChangesNothing(t *testing.T) {
	run := func(bound int, inPlace bool) (Stats, []byte, uint64) {
		s := newRegionShard(t, bound, inPlace)
		val := make([]byte, 96)
		for i := 0; i < 4_000; i++ {
			key := []byte(fmt.Sprintf("key-%d", i%64))
			for j := range val {
				val[j] = byte(i)
			}
			if err := s.putH(key, val, 0, hashKey(key)); err != nil {
				t.Fatalf("put %d: %v", i, err)
			}
		}
		var found []byte
		for i := 0; i < 64; i++ {
			key := []byte(fmt.Sprintf("key-%d", i))
			v, err := s.getH(key, hashKey(key))
			if err == nil {
				found = append(found, byte(i), v[0])
			}
		}
		return s.snapshot(), found, s.mutablePagesSealed.Load()
	}

	for _, inPlace := range []bool{false, true} {
		off, offKeys, offSealed := run(0, inPlace)
		on, onKeys, onSealed := run(1, inPlace)
		if inPlace && off.InPlaceUpdates == 0 {
			t.Fatal("the in-place case did not exercise a single in-place update")
		}
		if offSealed != 0 {
			t.Fatalf("the region-disabled run sealed %d pages", offSealed)
		}
		if onSealed == 0 {
			t.Fatal("the sealing run sealed nothing; the comparison would be vacuous")
		}
		if off != on {
			t.Fatalf("sealing changed the shard's behaviour (inPlace=%v):\n region off: %+v\n region on:  %+v", inPlace, off, on)
		}
		if string(offKeys) != string(onKeys) {
			t.Fatalf("sealing changed what the shard retained (inPlace=%v)", inPlace)
		}
	}
}

// TestFreshHeapPageLockedIsTheSoleChokepoint is the mechanical half of the claim the
// whole design rests on: a heap page that skipped freshHeapPageLocked would be
// published without the bookkeeping, and a page object handed a NEW latch anywhere
// else would be a sealed extent silently returned to mutable. Both are source-level
// facts, so assert them at the source: outside tests, newHeapPage is called once and
// a latch is minted once, and both calls are in freshHeapPageLocked.
func TestFreshHeapPageLockedIsTheSoleChokepoint(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	type site struct {
		file string
		line int
		text string
	}
	var heapPageCalls, latchMints []site
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, rerr := os.ReadFile(filepath.Clean(name))
		if rerr != nil {
			t.Fatalf("read %s: %v", name, rerr)
		}
		for i, line := range strings.Split(string(src), "\n") {
			code := line
			if idx := strings.Index(code, "//"); idx >= 0 {
				code = code[:idx] // ignore prose that merely NAMES the call
			}
			if strings.Contains(code, "newHeapPage(") && !strings.Contains(code, "func newHeapPage(") {
				heapPageCalls = append(heapPageCalls, site{name, i + 1, strings.TrimSpace(line)})
			}
			if strings.Contains(code, "pageseal.New(") {
				latchMints = append(latchMints, site{name, i + 1, strings.TrimSpace(line)})
			}
		}
	}
	if len(heapPageCalls) != 1 || heapPageCalls[0].file != "shard.go" {
		t.Fatalf("heap pages are allocated outside freshHeapPageLocked: %+v", heapPageCalls)
	}
	if len(latchMints) != 1 || latchMints[0].file != "shard.go" {
		t.Fatalf("mutable latches are minted outside freshHeapPageLocked: %+v", latchMints)
	}
	// …and both of those single sites are inside freshHeapPageLocked itself.
	src, err := os.ReadFile("shard.go")
	if err != nil {
		t.Fatalf("read shard.go: %v", err)
	}
	lines := strings.Split(string(src), "\n")
	start, end := -1, -1
	for i, line := range lines {
		if strings.HasPrefix(line, "func (s *shard) freshHeapPageLocked()") {
			start = i + 1
		} else if start >= 0 && strings.HasPrefix(line, "}") {
			end = i + 1
			break
		}
	}
	if start < 0 || end < 0 {
		t.Fatal("could not locate freshHeapPageLocked in shard.go")
	}
	for _, s := range append(append([]site{}, heapPageCalls...), latchMints...) {
		if s.line < start || s.line > end {
			t.Fatalf("%s:%d is outside freshHeapPageLocked (lines %d-%d): %s", s.file, s.line, start, end, s.text)
		}
	}
}
