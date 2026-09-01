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

// These tests cover the RECYCLE half of online relocating compaction
// (cache/compact_online.go): after relocation retires a fully-evacuated mmap page,
// its stranded extent is quarantined for AliasQuarantine and then RESET back into
// writable space (with a bumped generation), actually recovering write capacity. The
// gate remains mmap + replicated + reject-writes; heap / single-node / ringbuf shards
// are untouched. Reader-safety is non-negotiable: a recycled+reused page must never
// tear a concurrent lock-free reader.

// relocateToRetirement drives compactRelocateOnce until at least one page is retired
// (or a pass budget is exhausted). Returns once OnlinePagesRetired > 0.
func relocateToRetirement(t *testing.T, c *Cache, dropClock uint64) {
	t.Helper()
	for pass := 0; pass < 32; pass++ {
		c.shards[0].compactRelocateOnce(dropClock)
		if c.Stats().OnlinePagesRetired > 0 {
			return
		}
	}
	t.Fatalf("relocation retired no pages after 32 passes (retired=%d)", c.Stats().OnlinePagesRetired)
}

// fillUntilFull writes distinct full-value keys via stamped PutAt until the shard
// rejects with ErrFull, returning how many it wrote. keyPrefix keeps the keys from
// colliding with an earlier fill phase.
func fillUntilFull(t *testing.T, c *Cache, keyPrefix string, val []byte, stamp uint64) int {
	t.Helper()
	for i := 0; i < 100_000; i++ {
		err := c.PutAt([]byte(fmt.Sprintf("%s%06d", keyPrefix, i)), val, 0, stamp)
		if err == ErrFull {
			return i
		}
		if err != nil {
			t.Fatalf("fill PutAt #%d: %v", i, err)
		}
	}
	t.Fatal("shard never reached ErrFull — page budget too large for the test")
	return 0
}

// TestOnlineRecycleRecoversCapacity is the recovery gate. A shard is fragmented and
// relocated so pages retire, then filled to ErrFull. Before the alias quarantine
// elapses the retired extents MUST stay stranded (a premature reset could tear a
// reader), so a recycle pass reclaims nothing and the shard stays full. Once the
// quarantine elapses, recycling resets those extents and a full-size write that was
// ErrFull SUCCEEDS — capacity actually recovered.
func TestOnlineRecycleRecoversCapacity(t *testing.T) {
	const S = uint64(1_000_000)
	const quarantine = 80 * time.Millisecond
	c := eligibleOnlineCacheQ(t, 1, 1<<20, 8<<20, quarantine) // 8 pages
	s := c.shards[0]

	// Phase A — fragment then relocate so pages retire. Write originals, then overwrite
	// every key with a DISTINCT value: the early all-original pages become fully dead
	// and retire, and live copies consolidate onto later pages.
	const nKeys = 12
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("live%03d", i)) }
	origFor := func(i int) []byte { return bytes.Repeat([]byte{byte('A' + i)}, 180_000) }
	liveFor := func(i int) []byte { return bytes.Repeat([]byte{byte('a' + i)}, 180_000) }
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
	relocateToRetirement(t, c, S)
	retired := c.Stats().OnlinePagesRetired
	if retired == 0 {
		t.Fatal("no pages retired — cannot exercise recycle")
	}

	// Phase B — exhaust every remaining writable page. The retired extents are stranded
	// (firstPageWithRoomLocked skips them), so the shard is now genuinely full.
	bigVal := bytes.Repeat([]byte("z"), 180_000)
	fillUntilFull(t, c, "fill", bigVal, S)
	if err := c.PutAt([]byte("probe"), bigVal, 0, S); err != ErrFull {
		t.Fatalf("shard should be full before recycle: err=%v, want ErrFull", err)
	}

	// Phase C — quarantine NOT yet elapsed: a recycle pass must reclaim nothing (a reader
	// could still alias a retired page's bytes), so the shard stays full.
	s.compactRelocateOnce(S)
	if got := c.Stats().OnlinePagesRecycled; got != 0 {
		t.Fatalf("recycled %d pages BEFORE quarantine elapsed — premature reset risks a torn read", got)
	}
	if err := c.PutAt([]byte("probe"), bigVal, 0, S); err != ErrFull {
		t.Fatalf("write must still be ErrFull before quarantine elapses: err=%v", err)
	}

	// Phase D — quarantine elapsed: recycling resets the retired extents and the
	// previously-ErrFull write now succeeds.
	time.Sleep(2 * quarantine)
	s.compactRelocateOnce(S)
	if got := c.Stats().OnlinePagesRecycled; got == 0 {
		t.Fatal("recycled nothing after the quarantine elapsed — capacity was not recovered")
	}
	if err := c.PutAt([]byte("probe"), bigVal, 0, S); err != nil {
		t.Fatalf("post-recycle PutAt: err=%v, want success (recycle must have freed a page)", err)
	}
	// Every original live key must still resolve to its exact overwrite value.
	for i := 0; i < nKeys; i++ {
		v, err := c.Get(keyFor(i))
		if err != nil {
			t.Fatalf("post-recycle Get(%d): %v", i, err)
		}
		if !bytes.Equal(v, liveFor(i)) {
			t.Fatalf("post-recycle Get(%d): value mismatch (resolved a stale/recycled copy?)", i)
		}
	}
}

// TestOnlineRecycleSurvivesRestart is the recycle determinism gate: a shard that has
// relocated AND recycled (its retired extents reset and reused by fresh writes) must
// rebuild (rebuildIndexFromPages, cold compaction OFF) to the identical live set.
func TestOnlineRecycleSurvivesRestart(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	const S = uint64(1_000_000)
	const quarantine = 60 * time.Millisecond
	const nKeys = 12
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("key%03d", i)) }
	origFor := func(i int) []byte { return bytes.Repeat([]byte{byte('A' + i)}, 180_000) }
	liveFor := func(i int) []byte { return bytes.Repeat([]byte{byte('a' + i)}, 180_000) }
	extraFor := func(i int) []byte { return []byte(fmt.Sprintf("extra%03d", i)) }
	extraVal := func(i int) []byte { return bytes.Repeat([]byte{byte('m' + (i % 8))}, 120_000) }

	dir := t.TempDir()
	newCfg := func() Config {
		cfg := DefaultConfig()
		cfg.NumShards = 1
		cfg.PageSize = 1 << 20
		cfg.MaxMemoryPerShard = 8 << 20
		cfg.AtCapPolicy = PolicyRejectWrites
		cfg.Replicated = true
		cfg.OnlineCompaction = true
		cfg.AliasQuarantine = quarantine
		cfg.TTLSweepIntervalMs = 0
		cfg.DataDir = dir
		cfg.NowFn = func() uint64 { return 1 }
		return cfg
	}

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
	relocateToRetirement(t, c1, S)
	// Elapse the quarantine and recycle, then WRITE NEW keys that land on the recycled
	// extents (overwriting their old bytes) — this is what the rebuild must reconstruct.
	time.Sleep(2 * quarantine)
	c1.shards[0].compactRelocateOnce(S)
	if c1.Stats().OnlinePagesRecycled == 0 {
		t.Fatal("no pages recycled before restart — recycle path did not run")
	}
	nExtra := 0
	for i := 0; i < nKeys; i++ {
		if err := c1.PutAt(extraFor(i), extraVal(i), 0, S); err == nil {
			nExtra++
		} else if err != ErrFull {
			t.Fatalf("extra put %d: %v", i, err)
		}
	}
	if nExtra == 0 {
		t.Fatal("no extra keys landed on recycled space — recycle did not restore writable capacity")
	}
	if err := c1.Close(); err != nil {
		t.Fatalf("close c1: %v", err)
	}

	// Reopen with cold compaction DISABLED: rebuildIndexFromPages alone must reconstruct
	// the live set from the fragmented+relocated+recycled file.
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
			t.Fatalf("post-restart GetAt(key%d): %v", i, gErr)
		}
		if !bytes.Equal(v, liveFor(i)) {
			t.Fatalf("post-restart GetAt(key%d): value mismatch — rebuild resolved a stale copy", i)
		}
	}
	// The extra keys that were written onto recycled extents must survive too.
	for i := 0; i < nKeys; i++ {
		v, gErr := c2.GetAt(extraFor(i), 1)
		if gErr == ErrNotFound {
			continue // this extra key was ErrFull at write time — never stored.
		}
		if gErr != nil {
			t.Fatalf("post-restart GetAt(extra%d): %v", i, gErr)
		}
		if !bytes.Equal(v, extraVal(i)) {
			t.Fatalf("post-restart GetAt(extra%d): value mismatch on a recycled-extent write", i)
		}
	}
}

// TestOnlineRecycleHeavyRace is the recycle reader-safety gate. Lock-free readers run
// CONCURRENTLY with a writer creating fragmentation and a relocator that continuously
// relocates, retires, quarantines, RESETS, and lets fresh writes REUSE recycled pages
// — the exact race a byte-level overwrite of reader-visible bytes would lose. Under
// `go test -race` any overwrite of a still-aliased extent is flagged, and the per-read
// verify catches a torn/wrong value. A recycle (with the short quarantine) must have
// actually happened, and a write that would previously ErrFull must succeed after one.
// MUST pass `go test -race -count=3`.
func TestOnlineRecycleHeavyRace(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	const quarantine = 20 * time.Millisecond
	c := eligibleOnlineCacheQ(t, 1, 1<<20, 16<<20, quarantine) // 16 pages

	const nKeys = 48
	keyFor := func(i int) []byte { return []byte(fmt.Sprintf("hot%04d", i)) }
	valFor := func(i int) []byte { return bytes.Repeat([]byte{byte(i)}, 8<<10) } // 8 KiB, stable per key
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt(keyFor(i), valFor(i), 0, 1_000); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}

	var (
		stop    atomic.Bool
		bad     atomic.Int64
		reuseOK atomic.Int64 // writes that succeeded (fresh capacity was available)
		wg      sync.WaitGroup
	)

	// Lock-free readers: a hit must carry the exact stable value.
	for r := 0; r < 6; r++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			var buf []byte
			for !stop.Load() {
				for i := 0; i < nKeys; i++ {
					if seed%2 == 0 {
						out, gErr := c.GetInto(buf[:0], keyFor(i))
						if gErr == nil && !bytes.Equal(out, valFor(i)) {
							bad.Add(1)
						}
					} else {
						v, gErr := c.Get(keyFor(i))
						if gErr == nil && !bytes.Equal(v, valFor(i)) {
							bad.Add(1)
						}
					}
				}
			}
		}(r)
	}

	// Relocator: hammer relocation + recycle, racing readers at the byte level.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for !stop.Load() {
			c.shards[0].compactRelocateOnce(1_000)
		}
	}()

	// Writer: keep overwriting keys (fragmentation) and, whenever the shard is full,
	// rely on recycle to recover capacity so the write eventually succeeds again.
	wg.Add(1)
	go func() {
		defer wg.Done()
		stamp := uint64(1_000)
		for iter := 0; iter < 3000 && !stop.Load(); iter++ {
			for i := 0; i < nKeys; i++ {
				if err := c.PutAt(keyFor(i), valFor(i), 0, stamp); err == nil {
					reuseOK.Add(1)
				}
			}
			stamp += 1_000
			advanceLogicalClock(c, stamp)
			c.shards[0].sweepOnce()
			time.Sleep(time.Millisecond) // let the quarantine elapse so recycle can run
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
		t.Fatalf("%d malformed/torn reads during online recycle", n)
	}
	if got := c.Stats().OnlinePagesRecycled; got == 0 {
		t.Fatal("no pages recycled — the recycle race window never opened")
	}
	if reuseOK.Load() == 0 {
		t.Fatal("no writes ever succeeded — recycle never recovered capacity")
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
