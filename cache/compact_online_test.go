// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These tests cover ONLINE relocating compaction (cache/compact_online.go): the
// reader-safe primitive that, while the process runs, relocates the live entries out
// of a fragmented mmap page and retires the emptied source extent — WITHOUT
// disturbing the lock-free zero-copy read path. Stage 0 (reclaimable accounting +
// trigger) runs on every eligible shard; Stage 1 (the relocation action) is gated by
// Config.OnlineCompaction. Correctness and reader-safety dominate: a relocated shard
// must serve every live value unchanged throughout, survive a restart with the
// identical live set, and judge TTL expiry against the LOGICAL clock, never wall time.

// eligibleOnlineCache builds a single-shard mmap + replicated + reject-writes cache
// (the exact mode the online compactor operates on) with OnlineCompaction enabled and
// the ticker OFF so the test drives sweepOnce / compactRelocateOnce itself. clock0
// seeds the injected WALL clock; the apply path ignores it (that is the point) — the
// logical clock is advanced separately.
func eligibleOnlineCache(t *testing.T, clock0 uint64, pageSize, maxMem int) *Cache {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = pageSize
	cfg.MaxMemoryPerShard = maxMem
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.OnlineCompaction = true
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir()
	cfg.NowFn = func() uint64 { return clock0 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	if c.shards[0].region == nil {
		t.Fatal("expected mmap mode (region != nil) with DataDir set")
	}
	if !c.shards[0].onlineCompactionEligible() {
		t.Fatal("shard should be online-compaction eligible (mmap + replicated + reject-writes)")
	}
	return c
}

// eligibleOnlineCacheQ is eligibleOnlineCache with an explicit (short) AliasQuarantine
// so the recycle path — which only reclaims a retired page after its quarantine has
// elapsed — can be exercised in tens of milliseconds instead of the 60s default.
func eligibleOnlineCacheQ(t *testing.T, clock0 uint64, pageSize, maxMem int, quarantine time.Duration) *Cache {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = pageSize
	cfg.MaxMemoryPerShard = maxMem
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.OnlineCompaction = true
	cfg.AliasQuarantine = quarantine
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir()
	cfg.NowFn = func() uint64 { return clock0 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	if !c.shards[0].onlineCompactionEligible() {
		t.Fatal("shard should be online-compaction eligible")
	}
	return c
}

// TestOnlineReclaimableBytesAccounting proves the Stage-0 accounting: reclaimable =
// used - live counts superseded/dead bytes, and — crucially — judges TTL expiry
// against the LOGICAL clock, not the injected (high) wall clock.
func TestOnlineReclaimableBytesAccounting(t *testing.T) {
	const S, ttlMs = uint64(1_000_000), uint64(5_000)
	// Wall clock pinned FAR PAST every expiry we set: if the accounting ever consulted
	// wall time, the TTL key below would be wrongly counted reclaimable.
	c := eligibleOnlineCache(t, S+ttlMs+1_000_000, 1<<20, 8<<20)
	s := c.shards[0]
	val := bytes.Repeat([]byte("v"), 180_000)

	// Ten no-TTL keys, all live: reclaimable must be ~0 (used == live).
	for i := 0; i < 10; i++ {
		if err := c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, S); err != nil {
			t.Fatalf("put k%02d: %v", i, err)
		}
	}
	if got := s.reclaimableBytesNow(); got != 0 {
		t.Fatalf("all-live shard: reclaimable=%d, want 0", got)
	}

	// Overwrite half of them: each original copy becomes a dead duplicate. Reclaimable
	// must jump by ~5 entries' worth of bytes.
	perEntry := uint64(entrySize(len("k00"), len(val)))
	for i := 0; i < 5; i++ {
		if err := c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, S); err != nil {
			t.Fatalf("overwrite k%02d: %v", i, err)
		}
	}
	if got := s.reclaimableBytesNow(); got != 5*perEntry {
		t.Fatalf("after 5 overwrites: reclaimable=%d, want %d (5 dead duplicates)", got, 5*perEntry)
	}

	// A TTL key whose exp sits BETWEEN the (low) logical clock and the (high) wall clock.
	advanceLogicalClock(c, S) // logical clock = S, below exp = S+ttlMs
	ttlKey := []byte("ttl")
	if err := c.PutAt(ttlKey, val, time.Duration(ttlMs)*time.Millisecond, S); err != nil {
		t.Fatalf("put ttl key: %v", err)
	}
	// Logical clock (S) < exp (S+ttlMs): the key is LIVE, so it must NOT be reclaimable,
	// even though the wall clock is far past its expiry. This is the logical-vs-wall proof.
	if got := s.reclaimableBytesNow(); got != 5*perEntry {
		t.Fatalf("TTL key live at logical clock: reclaimable=%d, want %d (wall clock must NOT drop it)",
			got, 5*perEntry)
	}
	// Advance the logical clock past the TTL: now the key is logically expired and its
	// bytes become reclaimable.
	advanceLogicalClock(c, S+ttlMs+1)
	if got := s.reclaimableBytesNow(); got != 6*perEntry {
		t.Fatalf("after logical clock passes TTL: reclaimable=%d, want %d", got, 6*perEntry)
	}
}

// TestReclaimableStatsCacheRateLimits proves the Stats gauge is O(1) under frequent
// scraping: reclaimableBytesForStats serves a cached figure within reclaimableStatsTTL,
// so a fresh burst of dead bytes is NOT re-walked on every call (while the always-fresh
// reclaimableBytesNow does see it). This is the guard that keeps a metrics poll from
// turning each Stats() into an O(entries) walk under the read lock.
func TestReclaimableStatsCacheRateLimits(t *testing.T) {
	const S = uint64(1_000_000)
	c := eligibleOnlineCache(t, S+1_000_000, 1<<20, 8<<20)
	s := c.shards[0]
	val := bytes.Repeat([]byte("v"), 180_000)
	perEntry := uint64(entrySize(len("k00"), len(val)))

	for i := 0; i < 6; i++ {
		if err := c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, S); err != nil {
			t.Fatalf("put k%02d: %v", i, err)
		}
	}
	// First call computes fresh (cache empty) and caches: all-live ⇒ 0.
	if got := s.reclaimableBytesForStats(); got != 0 {
		t.Fatalf("first Stats call: reclaimable=%d, want 0", got)
	}
	// Create dead duplicates. The raw accounting sees them immediately...
	for i := 0; i < 3; i++ {
		if err := c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, S); err != nil {
			t.Fatalf("overwrite k%02d: %v", i, err)
		}
	}
	if got := s.reclaimableBytesNow(); got != 3*perEntry {
		t.Fatalf("raw accounting after overwrites: reclaimable=%d, want %d", got, 3*perEntry)
	}
	// ...but the Stats accessor serves the cache within the TTL instead of re-walking —
	// the O(1) property. Prove the within-TTL branch DETERMINISTICALLY (not by racing the
	// 2s TTL against however long this test takes under -race -count=3): stamp a distinct
	// sentinel value with a fresh timestamp, then assert the accessor returns THAT value
	// verbatim. A recompute would instead return 3*perEntry, so the sentinel can only be
	// observed via the cache-hit path.
	const sentinel = uint64(0xABCDEF)
	s.reclaimableCache.Store(&reclaimableSnapshot{bytes: sentinel, at: time.Now().UnixNano()})
	if got := s.reclaimableBytesForStats(); got != sentinel {
		t.Fatalf("Stats within TTL should serve the cached sentinel, got reclaimable=%d, want %d", got, sentinel)
	}
}

// TestOnlineRelocationPreservesLiveSet drives an actual relocation pass and asserts
// every live key survives with its exact value, that source pages are retired, and
// that the retirement is reader-visible only as a write-path capacity change (reads
// still resolve every key).
func TestOnlineRelocationPreservesLiveSet(t *testing.T) {
	const S = uint64(1_000_000)
	c := eligibleOnlineCache(t, 1, 1<<20, 16<<20) // 16 pages of churn room
	s := c.shards[0]

	const nKeys = 12
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("key%03d", i)) }
	origFor := func(i int) []byte { return bytes.Repeat([]byte{byte('A' + i)}, 180_000) }
	liveFor := func(i int) []byte { return bytes.Repeat([]byte{byte('a' + i)}, 180_000) } // distinct overwrite value

	// Write an original, then OVERWRITE every key with a DISTINCT value: each original
	// copy is now a dead duplicate on an early page (early pages holding only originals
	// become fully dead and retireable), while the fresh live copies pin later pages. A
	// distinct overwrite value means resolving a key to a stale (dead original) copy is
	// caught by the value check below, not masked by identical bytes.
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt(keyFor(i), origFor(i), 0, S); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt(keyFor(i), liveFor(i), 0, S); err != nil {
			t.Fatalf("overwrite %d: %v", i, err)
		}
	}

	advanceLogicalClock(c, S)
	// Drive relocation to completion (budget is 4 pages/call).
	for pass := 0; pass < 10; pass++ {
		s.compactRelocateOnce(S)
	}

	// Some source pages must have been retired (the all-dead early pages), proving the
	// evacuate-and-retire path ran.
	if got := c.Stats().OnlinePagesRetired; got == 0 {
		t.Fatal("relocation retired no pages — the evacuate/retire path did not run")
	}

	// Every key must still resolve to its exact LIVE (overwrite) value.
	for i := 0; i < nKeys; i++ {
		v, err := c.Get(keyFor(i))
		if err != nil {
			t.Fatalf("post-relocation Get(%d): %v", i, err)
		}
		if !bytes.Equal(v, liveFor(i)) {
			t.Fatalf("post-relocation Get(%d): value mismatch (resolved to a stale copy?)", i)
		}
	}

	// The dead duplicates the relocation left behind on retired pages are still
	// reclaimable bytes (Stage 1 does not free them), but the LIVE set is intact.
	if got := c.Stats().Rejects; got != 0 {
		t.Fatalf("unexpected rejects during relocation setup: %d", got)
	}
}

// TestOnlineRelocationSurvivesRestart is the determinism gate: after relocation, a
// warm restart (rebuildIndexFromPages, cold compaction DISABLED so the rebuild alone
// must reconstruct from the fragmented+relocated file) yields the IDENTICAL live
// key→value set. This proves the fresh writeSeq on a relocated copy makes it win the
// rebuild's max-seq contest over its stranded source original.
func TestOnlineRelocationSurvivesRestart(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	const S = uint64(1_000_000)
	const nKeys = 12
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("key%03d", i)) }
	origFor := func(i int) []byte { return bytes.Repeat([]byte{byte('A' + i)}, 180_000) }
	liveFor := func(i int) []byte { return bytes.Repeat([]byte{byte('a' + i)}, 180_000) } // distinct overwrite value

	dir := t.TempDir()
	newCfg := func() Config {
		cfg := DefaultConfig()
		cfg.NumShards = 1
		cfg.PageSize = 1 << 20
		cfg.MaxMemoryPerShard = 16 << 20
		cfg.AtCapPolicy = PolicyRejectWrites
		cfg.Replicated = true
		cfg.OnlineCompaction = true
		cfg.TTLSweepIntervalMs = 0
		cfg.DataDir = dir
		cfg.NowFn = func() uint64 { return 1 }
		return cfg
	}

	// Phase 1: fill, overwrite, relocate, then close (persisting the fragmented file).
	c1, err := New(newCfg())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < nKeys; i++ {
		if err := c1.PutAt(keyFor(i), origFor(i), 0, S); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	for i := 0; i < nKeys; i++ {
		if err := c1.PutAt(keyFor(i), liveFor(i), 0, S); err != nil {
			t.Fatalf("overwrite %d: %v", i, err)
		}
	}
	advanceLogicalClock(c1, S)
	for pass := 0; pass < 10; pass++ {
		c1.shards[0].compactRelocateOnce(S)
	}
	if c1.Stats().OnlinePagesRetired == 0 {
		t.Fatal("no pages retired before restart — relocation did not run")
	}
	if err := c1.Close(); err != nil {
		t.Fatalf("close c1: %v", err)
	}

	// Phase 2: reopen with cold compaction DISABLED, so rebuildIndexFromPages alone must
	// reconstruct the live set from the fragmented file (both stranded source copies AND
	// relocated copies are on the pages; the rebuild must pick the relocated ones).
	cfg2 := newCfg()
	cfg2.DisableColdCompaction = true
	c2, err := New(cfg2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c2.Close() })
	for i := 0; i < nKeys; i++ {
		v, gErr := c2.GetAt(keyFor(i), 1)
		if gErr != nil {
			t.Fatalf("post-restart GetAt(%d): %v", i, gErr)
		}
		if !bytes.Equal(v, liveFor(i)) {
			t.Fatalf("post-restart GetAt(%d): value mismatch — rebuild resolved a stale copy", i)
		}
	}
}

// TestOnlineRelocationExpiredDroppedByLogicalClock proves a relocation pass never
// relocates a logically-expired entry: the expired key is dropped (by the logical
// sweeper that runs first in sweepOnce), governed by the logical clock — NOT the
// (high) wall clock — while a non-expired key in the same shard survives and relocates.
func TestOnlineRelocationExpiredDroppedByLogicalClock(t *testing.T) {
	const S, ttlShort = uint64(2_000_000), uint64(1_000)
	// Wall clock pinned past the short TTL: if expiry ever used wall time, the short key
	// would be dropped immediately regardless of the logical clock.
	c := eligibleOnlineCache(t, S+ttlShort+1_000_000, 1<<20, 16<<20)
	s := c.shards[0]
	val := bytes.Repeat([]byte("z"), 180_000)

	shortKey, longKey := []byte("short"), []byte("long")
	if err := c.PutAt(shortKey, val, time.Duration(ttlShort)*time.Millisecond, S); err != nil {
		t.Fatalf("put short: %v", err)
	}
	if err := c.PutAt(longKey, val, 0, S); err != nil { // no TTL
		t.Fatalf("put long: %v", err)
	}
	// Overwrite the long key so there is a dead duplicate to make the shard fragmented.
	if err := c.PutAt(longKey, val, 0, S); err != nil {
		t.Fatalf("overwrite long: %v", err)
	}

	// Logical clock still BELOW the short TTL: a sweep must NOT drop the short key even
	// though the wall clock is far past it.
	advanceLogicalClock(c, S)
	s.sweepOnce()
	if _, err := c.GetAt(shortKey, 1); err != nil {
		t.Fatalf("short key dropped while logically live (wall clock must not govern): %v", err)
	}

	// Advance the logical clock past the short TTL and sweep: now it is logically
	// expired and reclaimed; the long key survives.
	advanceLogicalClock(c, S+ttlShort+1)
	s.sweepOnce()
	if _, err := c.GetAt(shortKey, 1); err != ErrNotFound {
		t.Fatalf("short key survived past its logical expiry: err=%v, want ErrNotFound", err)
	}
	v, err := c.GetAt(longKey, 1)
	if err != nil {
		t.Fatalf("long key reclaimed: %v", err)
	}
	if !bytes.Equal(v, val) {
		t.Fatal("long key value mismatch after sweep/relocation")
	}
}

// TestOnlineCompactionGateNoOpForOtherShards confirms the gate for shards the online
// compactor must NOT touch at all: a HEAP replicated shard and a SINGLE-NODE mmap
// shard are not eligible-mode, so they report zero reclaimable bytes and never
// relocate/retire — byte-for-byte unchanged behavior.
func TestOnlineCompactionGateNoOpForOtherShards(t *testing.T) {
	cases := []struct {
		name  string
		apply func(cfg *Config)
	}{
		{"heap-replicated", func(cfg *Config) { cfg.Replicated = true }}, // no DataDir ⇒ heap
		{"mmap-single-node", func(cfg *Config) { cfg.DataDir = t.TempDir() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.NumShards = 1
			cfg.PageSize = 1 << 20
			cfg.MaxMemoryPerShard = 8 << 20
			cfg.AtCapPolicy = PolicyRejectWrites
			cfg.TTLSweepIntervalMs = 0
			cfg.NowFn = func() uint64 { return 1 }
			tc.apply(&cfg)
			if cfg.DataDir != "" && runtime.GOOS != "linux" {
				t.Skip("mmap only on linux")
			}
			c, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = c.Close() })
			s := c.shards[0]
			if s.onlineCompactionEligible() {
				t.Fatal("shard must NOT be online-compaction eligible")
			}
			val := bytes.Repeat([]byte("v"), 180_000)
			for i := 0; i < 8; i++ {
				_ = c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, 1_000)
				_ = c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, 1_000) // dead duplicate
			}
			advanceLogicalClock(c, 1_000)
			s.sweepOnce()            // must not relocate/retire on a non-eligible shard
			s.maybeRelocateCompact() // direct call: also a no-op
			st := c.Stats()
			if st.ReclaimableBytes != 0 {
				t.Fatalf("non-eligible shard reported ReclaimableBytes=%d, want 0", st.ReclaimableBytes)
			}
			if st.OnlineRelocations != 0 || st.OnlinePagesRetired != 0 || st.OnlineBytesRelocated != 0 || st.OnlinePagesRecycled != 0 {
				t.Fatalf("non-eligible shard relocated/recycled: relocations=%d retired=%d bytes=%d recycled=%d, want all 0",
					st.OnlineRelocations, st.OnlinePagesRetired, st.OnlineBytesRelocated, st.OnlinePagesRecycled)
			}
		})
	}
}

// TestOnlineCompactionActionGatedWhenDisabled proves that with OnlineCompaction OFF an
// eligible-mode shard still runs STAGE 0 (reclaimable bytes are reported — pure
// observability) but takes NO relocation action even when the trigger fires: the
// sweeper's maybeRelocateCompact returns without relocating or retiring anything.
func TestOnlineCompactionActionGatedWhenDisabled(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 8 << 20
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.OnlineCompaction = false // eligible mode, action disabled
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir()
	cfg.NowFn = func() uint64 { return 1 }
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	s := c.shards[0]
	if !s.onlineCompactionEligible() {
		t.Fatal("shard should be eligible-mode (observability runs regardless of the action flag)")
	}
	val := bytes.Repeat([]byte("v"), 180_000)
	const nKeys = 8
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, 1_000); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt([]byte(fmt.Sprintf("k%02d", i)), val, 0, 1_000); err != nil { // dead duplicates
			t.Fatalf("overwrite %d: %v", i, err)
		}
	}
	advanceLogicalClock(c, 1_000)
	// Stage 0 observability IS live: the dead duplicates are reported as reclaimable.
	perEntry := uint64(entrySize(len("k00"), len(val)))
	if got := s.reclaimableBytesNow(); got != uint64(nKeys)*perEntry {
		t.Fatalf("reclaimable (observability) = %d, want %d", got, uint64(nKeys)*perEntry)
	}
	// But the ACTION is gated: driving the sweeper relocates/retires nothing.
	s.sweepOnce()
	st := c.Stats()
	if st.OnlineRelocations != 0 || st.OnlinePagesRetired != 0 {
		t.Fatalf("relocation ran with OnlineCompaction=false: relocations=%d retired=%d",
			st.OnlineRelocations, st.OnlinePagesRetired)
	}
	if st.ReclaimableBytes == 0 {
		t.Fatal("observability must stay on: ReclaimableBytes should be non-zero")
	}
}

// TestOnlineRelocationHeavyRace is the reader-safety gate. It runs lock-free
// reject-writes readers CONCURRENTLY with a writer creating fragmentation and a
// relocator continuously evacuating pages — the exact race the whole feature must be
// safe under. A relocation is mechanically an overwrite Put (append into fresh tail +
// atomic index repoint), so under `go test -race` any byte-level overwrite of
// reader-visible source bytes is flagged, and the per-read verify catches any torn or
// wrong value. A hit MUST carry the exact written value; a miss is never acceptable
// for a key that is always present (all keys are re-written every round with no TTL).
// It MUST pass `go test -race -count=3`.
func TestOnlineRelocationHeavyRace(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 64 << 20 // 64 pages: a long filling phase of continuous relocation
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = true
	cfg.OnlineCompaction = true
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir()
	cfg.NowFn = func() uint64 { return 1 } // wall clock low: reads observe live values
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	const nKeys = 32
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("hot%04d", i)) }
	valFor := func(i int) []byte { return bytes.Repeat([]byte{byte(i)}, 8<<10) } // 8 KiB, stable per key
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt(keyFor(i), valFor(i), 0, 1_000); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}

	var (
		stop atomic.Bool
		bad  atomic.Int64
		wg   sync.WaitGroup
	)

	// Lock-free readers: a hit must carry the exact stable value; a miss on an
	// always-present key is a failure (the value was lost during relocation).
	for r := 0; r < 6; r++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			var buf []byte
			for !stop.Load() {
				for i := 0; i < nKeys; i++ {
					// Every key is always present (seeded up front, rewritten each round, no
					// TTL, no delete), and relocation is miss-free by construction (the index
					// repoint is atomic, the old copy stays resolvable until then). So a miss —
					// a non-nil error — is a LOST key and a failure, exactly like a torn value;
					// count both, not just the value mismatch.
					if seed%2 == 0 {
						out, gErr := c.GetInto(buf[:0], keyFor(i))
						if gErr != nil || !bytes.Equal(out, valFor(i)) {
							bad.Add(1)
						}
					} else {
						v, gErr := c.Get(keyFor(i))
						if gErr != nil || !bytes.Equal(v, valFor(i)) {
							bad.Add(1)
						}
					}
				}
			}
		}(r)
	}

	// Relocator: hammer the relocation pass, racing the readers at the byte level.
	wg.Add(1)
	go func() {
		defer wg.Done()
		stamp := uint64(1_000)
		for !stop.Load() {
			c.shards[0].compactRelocateOnce(stamp)
		}
	}()

	// Writer: keep overwriting keys (creating fragmentation and driving relocation).
	// ErrFull is expected once the shard fills and is ignored — the keys keep their last
	// value, so readers still verify. The logical clock is advanced so the sweeper runs
	// too, but no TTL means nothing expires (keys stay present throughout).
	wg.Add(1)
	go func() {
		defer wg.Done()
		stamp := uint64(1_000)
		for iter := 0; iter < 4000 && !stop.Load(); iter++ {
			for i := 0; i < nKeys; i++ {
				_ = c.PutAt(keyFor(i), valFor(i), 0, stamp)
			}
			stamp += 1_000
			advanceLogicalClock(c, stamp)
			c.shards[0].sweepOnce()
		}
		stop.Store(true)
	}()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(90 * time.Second):
		stop.Store(true)
		<-done
	}
	if n := bad.Load(); n != 0 {
		t.Fatalf("%d malformed/torn reads during online relocation", n)
	}
	// Sanity: the relocation path was actually exercised.
	if c.Stats().OnlineRelocations == 0 {
		t.Fatal("no relocations occurred — the race window never opened")
	}
	// Final invariant: every key is still present with its exact value.
	for i := 0; i < nKeys; i++ {
		v, gErr := c.GetAt(keyFor(i), 1)
		if gErr != nil {
			t.Fatalf("final GetAt(%d): %v", i, gErr)
		}
		if !bytes.Equal(v, valFor(i)) {
			t.Fatalf("final GetAt(%d): value mismatch", i)
		}
	}
}
