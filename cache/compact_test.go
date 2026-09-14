// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// These tests cover COLD COMPACTION AT SHARD OPEN (cache/compact.go): the
// zero-read-cost reclamation of mmap page BYTES that closes the ghost-byte →
// ErrFull → Phase-A-halt cliff for persistent shards. The three things that can
// go wrong are covered separately:
//
//	1. correctness of the rewrite (every live key survives byte-identically);
//	2. DETERMINISM of the drop decision on a replicated shard (the crux: an entry
//	   must be judged against the persisted LOGICAL clock, never wall time, or a
//	   late-restarting replica silently diverges from a peer);
//	3. CRASH SAFETY of the file swap (a crash leaves the intact original or the
//	   complete compacted file — never a torn shard).

// compactTestCfg builds a single-shard mmap cache in reject-writes mode (what
// replication forces via B2) with a 4-page budget so it fills fast, the sweeper
// ticker OFF, and an injected wall clock.
func compactTestCfg(dir string, replicated bool, wall uint64) Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 4 << 20 // 4 mmap pages
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.Replicated = replicated
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = dir
	cfg.NowFn = func() uint64 { return wall }
	return cfg
}

func shardDirOf(dataDir string) string { return filepath.Join(dataDir, "shard-0000") }
func pagesPathOf(dataDir string) string {
	return filepath.Join(shardDirOf(dataDir), "pages.dat")
}

func openCompactCache(t *testing.T, cfg Config) *Cache {
	t.Helper()
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

// compactVal returns a distinct ~100 KiB value tagged with key index + round, so
// a survivor assertion proves the LATEST physical copy is the one that lived.
func compactVal(i, round int) []byte {
	b := bytes.Repeat([]byte("x"), 100<<10)
	copy(b, fmt.Sprintf("k%04d-r%04d|", i, round))
	return b
}

// shardEntry reads the value + absolute expiry of the entry the index currently
// resolves key to, i.e. the PHYSICAL truth, bypassing every expiry filter.
func shardEntry(s *shard, key []byte) (value []byte, exp uint64, ok bool) {
	_, ref, found := s.tab.Load().findSlot(hashKey(key))
	if !found {
		return nil, 0, false
	}
	k, v, e, err := s.pages[ref.pageIdx()].Read(ref.offset())
	if err != nil || !bytes.Equal(k, key) {
		return nil, 0, false
	}
	return append([]byte(nil), v...), e, true
}

// fillWithOverwrites writes distinct 100 KiB values round-robin over nKeys keys
// until the shard rejects with ErrFull, leaving every key but the newest copy as
// a superseded ghost. put is the write primitive (Put or a stamped PutAt).
func fillWithOverwrites(t *testing.T, nKeys int, key func(int) []byte, put func(k, v []byte) error) {
	t.Helper()
	for round := 0; round < 10_000; round++ {
		for i := 0; i < nKeys; i++ {
			err := put(key(i), compactVal(i, round))
			if errors.Is(err, ErrFull) {
				return
			}
			if err != nil {
				t.Fatalf("write key %d round %d: %v", i, round, err)
			}
		}
	}
	t.Fatal("shard never filled to ErrFull — page budget too large for this test")
}

// TestColdCompactionRoundTrip is the core round-trip proof: a shard carrying
// mostly ghost bytes (superseded overwrites + wall-clock-expired TTL entries)
// compacts at open. Every LIVE key survives with a byte-identical value and
// expiry, the file's allocated blocks shrink, occupancy falls below the alert
// band, and the capacity that was stuck at ErrFull is genuinely usable again.
func TestColdCompactionRoundTrip(t *testing.T) {
	dir := t.TempDir()
	const wall = uint64(5_000_000)
	cfg := compactTestCfg(dir, false, wall)

	liveKey := func(i int) []byte { return []byte(fmt.Sprintf("live%04d", i)) }
	ghostKey := func(i int) []byte { return []byte(fmt.Sprintf("ttl%04d", i)) }
	const nLive = 8

	c := openCompactCache(t, cfg)
	s := c.shards[0]

	// Two entries already past their absolute expiry at the injected wall clock:
	// ghost bytes that no read can ever return.
	for i := 0; i < 2; i++ {
		if err := c.PutAbs(ghostKey(i), compactVal(900+i, 0), wall-1_000); err != nil {
			t.Fatal(err)
		}
	}
	fillWithOverwrites(t, nLive, liveKey, func(k, v []byte) error { return c.Put(k, v, 0) })

	// Record the physical truth before the restart.
	type snap struct {
		val []byte
		exp uint64
	}
	want := make(map[string]snap, nLive)
	for i := 0; i < nLive; i++ {
		v, e, ok := shardEntry(s, liveKey(i))
		if !ok {
			t.Fatalf("live key %d missing before restart", i)
		}
		want[string(liveKey(i))] = snap{val: v, exp: e}
	}
	usedBefore, occBefore := s.bytesUsed(), s.occupancyRatio()
	if occBefore < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f < %.2f — the test must fill past the compaction mark",
			occBefore, mmapCompactMinOccupancy)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	blocksBefore := allocatedBlocks(t, pagesPathOf(dir))

	// Reopen: cold compaction runs inside newShard.
	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()
	s2 := c2.shards[0]

	for i := 0; i < nLive; i++ {
		w := want[string(liveKey(i))]
		got, exp, ok := shardEntry(s2, liveKey(i))
		if !ok {
			t.Fatalf("live key %d did not survive compaction", i)
		}
		if !bytes.Equal(got, w.val) {
			t.Fatalf("live key %d value changed across compaction (got %q… want %q…)",
				i, got[:16], w.val[:16])
		}
		if exp != w.exp {
			t.Fatalf("live key %d expiry changed across compaction: got %d want %d", i, exp, w.exp)
		}
		// And it is reachable through the normal read path.
		if v, err := c2.Get(liveKey(i)); err != nil || !bytes.Equal(v, w.val) {
			t.Fatalf("live key %d unreadable after compaction: err=%v", i, err)
		}
	}
	// The expired ghosts are gone, physically.
	for i := 0; i < 2; i++ {
		if _, _, ok := shardEntry(s2, ghostKey(i)); ok {
			t.Fatalf("wall-clock-expired ghost %d survived compaction on a single-node shard", i)
		}
	}

	usedAfter, occAfter := s2.bytesUsed(), s2.occupancyRatio()
	if usedAfter >= usedBefore {
		t.Fatalf("compaction reclaimed nothing: bytesUsed %d → %d", usedBefore, usedAfter)
	}
	if occAfter >= mmapOccupancyWarnLow {
		t.Fatalf("occupancy after compaction = %.3f, want below the alert re-arm mark %.2f",
			occAfter, mmapOccupancyWarnLow)
	}
	// The published file must be FULLY ALLOCATED — no hole above the pack frontier.
	// Compaction reclaims WRITE CAPACITY inside the file; it deliberately does NOT
	// return disk blocks. Leaving pages frontier+1..N-1 sparse would mean the first
	// runtime write up there fault-allocates, and on a full filesystem that fault is
	// SIGBUS — mid-serving process death, strictly worse than the startup crash loop
	// the staging reservation exists to prevent. Asserting "the file shrank" would
	// pin exactly that hazard, so assert the opposite. (blocksBefore is kept for the
	// diagnostic; st_blocks counts 512-byte units regardless of the FS block size.)
	fi, err := os.Stat(pagesPathOf(dir))
	if err != nil {
		t.Fatalf("stat compacted pages file: %v", err)
	}
	if blocksAfter := allocatedBlocks(t, pagesPathOf(dir)); blocksAfter*512 < fi.Size() {
		t.Fatalf("compacted pages file is SPARSE: %d allocated bytes < %d logical (was %d blocks); "+
			"a runtime write into the hole would SIGBUS on a full filesystem",
			blocksAfter*512, fi.Size(), blocksBefore)
	}
	// The decisive availability assertion: a full-size write that was ErrFull
	// before the restart now succeeds.
	if err := c2.Put([]byte("after-compaction"), compactVal(777, 0), 0); err != nil {
		t.Fatalf("post-compaction write: %v, want success (reclaimed capacity must be usable)", err)
	}
	// No staging file is left behind.
	if _, err := os.Stat(pagesPathOf(dir) + compactTmpSuffix); !os.IsNotExist(err) {
		t.Fatalf("compaction staging file still present after a successful swap (err=%v)", err)
	}
}

// TestColdCompactionReplicatedUsesPersistedLogicalClock is THE determinism test.
// A replicated shard is restarted with its wall clock far past every entry's
// expiry, but with a persisted logical clock that has NOT reached one of them.
// That entry MUST survive: dropping it would remove a key a peer (whose stamped
// reads judge liveness by the same logical clock) still considers live, and the
// next committed GetAt(stamp) would then disagree across replicas — silent
// divergence. The entries the logical clock HAS passed are reclaimed.
func TestColdCompactionReplicatedUsesPersistedLogicalClock(t *testing.T) {
	dir := t.TempDir()
	// Wall clock ~104 days past every expiry below: if compaction consulted it,
	// every entry in this test would be dropped.
	const wall = uint64(9_000_000_000)
	const S = uint64(1_000_000)
	const stamp = S + 5_000 // the logical clock we persist

	cfg := compactTestCfg(dir, true, wall)
	c := openCompactCache(t, cfg)

	survivor := []byte("survivor")
	survivorVal := compactVal(1, 1)
	// exp = S + 10_000_000 — far beyond the persisted logical clock, but far
	// BEHIND the wall clock.
	if err := c.PutAt(survivor, survivorVal, 10_000_000*time.Millisecond, S); err != nil {
		t.Fatal(err)
	}
	survivorExp := S + 10_000_000

	// Ghosts: short TTL (exp = S+1_000 <= stamp) plus round-robin overwrites, to
	// push occupancy past the compaction mark.
	ghostKey := func(i int) []byte { return []byte(fmt.Sprintf("ghost%04d", i)) }
	fillWithOverwrites(t, 6, ghostKey, func(k, v []byte) error {
		return c.PutAt(k, v, time.Second, S)
	})

	// A stamped apply at `stamp` advances the logical clock; SetAppliedIndex then
	// persists (appliedIndex, lastAppliedStampMs) into the header — exactly what
	// the FSM does on the apply path.
	if _, err := c.GetAt(survivor, stamp); err != nil {
		t.Fatalf("stamped read: %v", err)
	}
	if got := c.LastAppliedStampMs(); got != stamp {
		t.Fatalf("logical clock = %d, want %d", got, stamp)
	}
	c.SetAppliedIndex(4242, true)
	if occ := c.shards[0].occupancyRatio(); occ < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f < %.2f", occ, mmapCompactMinOccupancy)
	}
	usedBefore := c.shards[0].bytesUsed()
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	// Restart.
	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()
	s2 := c2.shards[0]

	// The persisted header round-trips both watermarks.
	if got := c2.LastAppliedStampMs(); got != stamp {
		t.Fatalf("restored logical clock = %d, want %d (the header must persist it)", got, stamp)
	}
	if got := c2.AppliedIndex(); got != 4242 {
		t.Fatalf("restored applied index = %d, want 4242", got)
	}

	// THE ASSERTION: exp > the persisted logical clock ⇒ survives, even though the
	// wall clock says it expired 104 days ago.
	val, exp, ok := shardEntry(s2, survivor)
	if !ok {
		t.Fatal("DIVERGENCE BUG: compaction dropped an entry the persisted logical clock " +
			"has not passed — a peer would still consider it live")
	}
	if !bytes.Equal(val, survivorVal) || exp != survivorExp {
		t.Fatalf("survivor corrupted by compaction: exp got %d want %d, value equal=%v",
			exp, survivorExp, bytes.Equal(val, survivorVal))
	}
	// Entries the logical clock HAS passed were reclaimed, so bytes came back.
	if used := s2.bytesUsed(); used >= usedBefore {
		t.Fatalf("replicated compaction reclaimed nothing: bytesUsed %d → %d", usedBefore, used)
	}
	for i := 0; i < 6; i++ {
		if _, _, present := shardEntry(s2, ghostKey(i)); present {
			t.Fatalf("ghost %d (exp=%d <= stamp=%d) survived compaction", i, S+1_000, stamp)
		}
	}
}

// TestColdCompactionSingleNodeDropsWallClockExpired is the contrast to the test
// above: the IDENTICAL entry (wall-clock expired, logical clock not yet past it)
// on a NON-replicated shard is dropped, because a single-node shard has no peer
// to diverge from and the wall clock is the only clock that means anything.
func TestColdCompactionSingleNodeDropsWallClockExpired(t *testing.T) {
	dir := t.TempDir()
	const wall = uint64(9_000_000_000)
	const S = uint64(1_000_000)

	cfg := compactTestCfg(dir, false, wall) // NOT replicated
	c := openCompactCache(t, cfg)

	victim := []byte("wall-expired")
	if err := c.PutAt(victim, compactVal(1, 1), 10_000_000*time.Millisecond, S); err != nil {
		t.Fatal(err)
	}
	ghostKey := func(i int) []byte { return []byte(fmt.Sprintf("ghost%04d", i)) }
	fillWithOverwrites(t, 6, ghostKey, func(k, v []byte) error { return c.Put(k, v, 0) })
	if occ := c.shards[0].occupancyRatio(); occ < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f < %.2f", occ, mmapCompactMinOccupancy)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()
	if _, _, ok := shardEntry(c2.shards[0], victim); ok {
		t.Fatal("a single-node shard must drop wall-clock-expired entries at open; it kept one")
	}
	// The still-live round-robin keys are untouched.
	for i := 0; i < 6; i++ {
		if _, _, ok := shardEntry(c2.shards[0], ghostKey(i)); !ok {
			t.Fatalf("non-expired key %d was dropped by single-node compaction", i)
		}
	}
}

// TestColdCompactionStaleTempFileRecovers is the crash-safety proof. A crash
// mid-compaction can only ever leave a staging file behind, because the atomic
// rename is the single step that publishes it. Simulate every such crash by
// planting a stale pages.dat.compact — truncated garbage (killed mid-write) and
// a complete-but-WRONG file (killed just before the rename) — and prove the
// original shard is what comes back, intact, with the stale file removed.
func TestColdCompactionStaleTempFileRecovers(t *testing.T) {
	for _, tc := range []struct {
		name  string
		stage func(t *testing.T, tmpPath string, size int64)
	}{
		{
			name: "truncated garbage",
			stage: func(t *testing.T, tmpPath string, _ int64) {
				// Killed partway through writing the staging file: a short file of
				// non-header bytes.
				if err := os.WriteFile(tmpPath, bytes.Repeat([]byte{0xAB}, 4096), 0o600); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "complete but unpublished",
			stage: func(t *testing.T, tmpPath string, size int64) {
				// Killed after the staging file was fully written and synced but
				// BEFORE the rename. It is a perfectly valid pages file — and it must
				// STILL lose to the original, which is the only published state.
				buf := make([]byte, size)
				writeHeader(buf, 1<<20, 4, 999_999)
				if err := os.WriteFile(tmpPath, buf, 0o600); err != nil {
					t.Fatal(err)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			const wall = uint64(5_000_000)
			cfg := compactTestCfg(dir, false, wall)

			liveKey := func(i int) []byte { return []byte(fmt.Sprintf("live%04d", i)) }
			c := openCompactCache(t, cfg)
			fillWithOverwrites(t, 8, liveKey, func(k, v []byte) error { return c.Put(k, v, 0) })
			want := make(map[string][]byte, 8)
			for i := 0; i < 8; i++ {
				v, _, ok := shardEntry(c.shards[0], liveKey(i))
				if !ok {
					t.Fatalf("key %d missing before the simulated crash", i)
				}
				want[string(liveKey(i))] = v
			}
			c.SetAppliedIndex(77, true)
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}

			size := int64(headerSize + 4*(1<<20))
			tmpPath := pagesPathOf(dir) + compactTmpSuffix
			tc.stage(t, tmpPath, size)

			c2 := openCompactCache(t, cfg)
			defer func() { _ = c2.Close() }()

			// The ORIGINAL data is what came back — no loss, no torn shard.
			for i := 0; i < 8; i++ {
				got, _, ok := shardEntry(c2.shards[0], liveKey(i))
				if !ok {
					t.Fatalf("key %d lost after recovering from a stale staging file", i)
				}
				if !bytes.Equal(got, want[string(liveKey(i))]) {
					t.Fatalf("key %d value changed after recovering from a stale staging file", i)
				}
			}
			if got := c2.AppliedIndex(); got != 77 {
				t.Fatalf("applied index = %d, want the original 77 (the staging file must not win)", got)
			}
			// The stale staging file is gone, and the shard is NOT left mapped to it.
			if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
				t.Fatalf("stale staging file survived the open (err=%v)", err)
			}
			// Nothing was rotated aside as corrupt.
			if bad, _ := filepath.Glob(pagesPathOf(dir) + ".bad-*"); len(bad) > 0 {
				t.Fatalf("the original pages file was rejected as corrupt: %v", bad)
			}
		})
	}
}

// downgradeHeaderToV3 rewrites a pages file's header as a version-3 header: the
// version field goes back to 3 and the core CRC is recomputed. The result is
// byte-for-byte a header the previous build wrote — while the PAGE bytes are
// still v4 entries, which is exactly the situation the version gate exists to
// catch (the two entry codecs frame differently and are mutually unintelligible).
func downgradeHeaderToV3(t *testing.T, path string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	hdr := make([]byte, headerSize)
	if _, err := f.ReadAt(hdr, 0); err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(hdr[8:12], 3)
	binary.LittleEndian.PutUint32(hdr[28:32], crc32.ChecksumIEEE(hdr[0:28]))
	if _, err := f.WriteAt(hdr, 0); err != nil {
		t.Fatal(err)
	}
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
}

// TestPreV4FileRotatesAside is the upgrade path for the v4 entry format. There is
// no deployed persistent state to migrate and the v3/v4 entry codecs are mutually
// unintelligible, so minReadableCacheVersion was raised to 4: a pre-v4 pages file
// takes the SAME tested path as a bad magic or a failed CRC — it is renamed
// .bad-<timestamp>, a fresh file is mapped in its place, and the shard starts
// empty. On a replicated node the committed state is then rebuilt from the log
// (replay, or an InstallSnapshot from a peer).
//
// The assertions that matter are that the open SUCCEEDS (a node must not refuse to
// start), that the stale file is preserved rather than deleted (an operator can
// still inspect it), and that not one byte of it is interpreted with the v4 reader.
func TestPreV4FileRotatesAside(t *testing.T) {
	dir := t.TempDir()
	const wall = uint64(5_000_000)
	cfg := compactTestCfg(dir, false, wall)

	c := openCompactCache(t, cfg)
	for i := 0; i < 4; i++ {
		if err := c.Put([]byte(fmt.Sprintf("k%04d", i)), compactVal(i, 0), 0); err != nil {
			t.Fatal(err)
		}
	}
	c.SetAppliedIndex(31, true)
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	origBlocks := allocatedBlocks(t, pagesPathOf(dir))

	downgradeHeaderToV3(t, pagesPathOf(dir))

	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()

	bad, _ := filepath.Glob(pagesPathOf(dir) + ".bad-*")
	if len(bad) != 1 {
		t.Fatalf("a pre-v4 file must be rotated aside exactly once; found %d .bad-* files: %v", len(bad), bad)
	}
	if got := allocatedBlocks(t, bad[0]); got != origBlocks {
		t.Fatalf("the rotated-aside file was modified: %d blocks, want %d", got, origBlocks)
	}
	// Started EMPTY: no key, no applied index, no logical clock carried over.
	for i := 0; i < 4; i++ {
		if _, err := c2.Get([]byte(fmt.Sprintf("k%04d", i))); err != ErrNotFound {
			t.Fatalf("key %d survived a rotate-aside: err=%v, want ErrNotFound — some part of "+
				"the old file was decoded with the v4 reader", i, err)
		}
	}
	if got := c2.AppliedIndex(); got != 0 {
		t.Fatalf("applied index after rotate-aside = %d, want 0 (the shard must replay from scratch)", got)
	}
	if used := c2.shards[0].bytesUsed(); used != 0 {
		t.Fatalf("bytesUsed after rotate-aside = %d, want 0", used)
	}
	// And it is a normal, writable shard afterwards.
	if err := c2.Put([]byte("fresh"), []byte("v"), 0); err != nil {
		t.Fatalf("write after rotate-aside: %v", err)
	}
	if v, err := c2.Get([]byte("fresh")); err != nil || !bytes.Equal(v, []byte("v")) {
		t.Fatalf("read back after rotate-aside = %q err=%v", v, err)
	}
}

// TestWarmRestartPreservesCommittedState covers the two ways an mmap warm
// restart used to change committed state on its own (#12A / #12B). Both lived in
// rebuildIndexFromPages and both reproduced with cold compaction nowhere in the
// picture — occupancy here stays far below mmapCompactMinOccupancy, so the
// compaction pass never runs and the plain rebuild is what is under test:
//
//	A. STALE-VALUE RESURRECTION. Writes do not fill pages in index order (the
//	   shard starts on the last page, and firstPageWithRoomLocked revisits lower
//	   pages), but the rebuild walk visited them 0, 1, … N-1 and let the LAST copy
//	   it saw win. A key overwritten across that boundary came back as its OLDER
//	   value — a committed write silently lost. Fixed by resolving on the entry's
//	   persisted write SEQUENCE.
//
//	B. DELETE RESURRECTION. Del only tombstoned an INDEX slot; nothing was
//	   recorded on the page, so the rebuild re-indexed the entry straight off the
//	   bytes and the deleted key came back. Fixed by appending a durable tombstone
//	   ENTRY on the persistent path.
//
// Cold compaction inherits whatever the rebuilt index resolves to and then makes
// it permanent by dropping the unreferenced copy, which is why these two are
// asserted directly rather than left to the compaction tests.
//
// The page-REVISIT variant of A — the one that does not depend on where writes
// start — is TestWarmRestartResolvesNewestAcrossPageRevisit.
func TestWarmRestartPreservesCommittedState(t *testing.T) {
	newCache := func(dir string) *Cache {
		cfg := compactTestCfg(dir, false, 5_000_000)
		return openCompactCache(t, cfg)
	}

	t.Run("A: overwrite survives the restart", func(t *testing.T) {
		dir := t.TempDir()
		c := newCache(dir)
		k := []byte("K")
		older, newer := compactVal(1, 0), compactVal(1, 99)
		if err := c.Put(k, older, 0); err != nil {
			t.Fatal(err)
		}
		// Fill the rest of the initial (last) page so the overwrite spills into
		// page 0, i.e. LOWER page index but LATER write.
		padPageDownTo(t, c, 0)
		if err := c.Put(k, newer, 0); err != nil {
			t.Fatal(err)
		}
		if got := keyPageIdx(t, c, k); got != 0 {
			t.Fatalf("the overwrite landed in page %d, want page 0 — the setup no longer "+
				"produces the page-order inversion it is meant to test", got)
		}
		if v, _ := c.Get(k); !bytes.Equal(v, newer) {
			t.Fatal("pre-restart read is already wrong; the setup is broken")
		}
		if occ := c.shards[0].occupancyRatio(); occ >= mmapCompactMinOccupancy {
			t.Fatalf("setup: occupancy %.3f would trigger compaction; this must test the plain rebuild", occ)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}

		c2 := newCache(dir)
		defer func() { _ = c2.Close() }()
		v, err := c2.Get(k)
		if err != nil {
			t.Fatalf("Get after restart: %v", err)
		}
		if bytes.Equal(v, older) {
			t.Fatal("warm restart resolved the key to its OLDER copy: a committed " +
				"overwrite was silently reverted (#12A)")
		}
		if !bytes.Equal(v, newer) {
			t.Fatalf("warm restart resolved the key to neither copy (%d bytes)", len(v))
		}
	})

	t.Run("B: deleted key stays deleted", func(t *testing.T) {
		dir := t.TempDir()
		c := newCache(dir)
		if err := c.Put([]byte("gone"), []byte("v"), 0); err != nil {
			t.Fatal(err)
		}
		if ok, err := c.Del([]byte("gone")); err != nil || !ok {
			t.Fatal("Del reported the key absent")
		}
		if _, err := c.Get([]byte("gone")); err != ErrNotFound {
			t.Fatalf("pre-restart Get after Del = %v, want ErrNotFound", err)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}

		c2 := newCache(dir)
		defer func() { _ = c2.Close() }()
		if _, err := c2.Get([]byte("gone")); err != ErrNotFound {
			t.Fatalf("Get after restart = %v, want ErrNotFound: the delete was not durable "+
				"and the key resurrected (#12B)", err)
		}
	})
}

// TestColdCompactionSkippedBelowThreshold: a lightly-filled shard must NOT pay
// for a rewrite it does not need — the file is left exactly as it was.
func TestColdCompactionSkippedBelowThreshold(t *testing.T) {
	dir := t.TempDir()
	cfg := compactTestCfg(dir, false, 5_000_000)
	c := openCompactCache(t, cfg)
	for i := 0; i < 4; i++ { // ~400 KiB of 4 MiB ⇒ occupancy ~0.1
		if err := c.Put([]byte(fmt.Sprintf("k%d", i)), compactVal(i, 0), 0); err != nil {
			t.Fatal(err)
		}
	}
	usedBefore := c.shards[0].bytesUsed()
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	blocksBefore := allocatedBlocks(t, pagesPathOf(dir))

	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()
	if used := c2.shards[0].bytesUsed(); used != usedBefore {
		t.Fatalf("a shard below the compaction mark was rewritten anyway: bytesUsed %d → %d", usedBefore, used)
	}
	if after := allocatedBlocks(t, pagesPathOf(dir)); after != blocksBefore {
		t.Fatalf("pages file was rewritten below the compaction mark: %d → %d blocks", blocksBefore, after)
	}
	for i := 0; i < 4; i++ {
		if _, err := c2.Get([]byte(fmt.Sprintf("k%d", i))); err != nil {
			t.Fatalf("key %d lost: %v", i, err)
		}
	}
}

// TestColdCompactionAllLiveKeepsEverything: a shard that is full of genuinely
// LIVE data has nothing reclaimable, so compaction must not run — and above all
// must not drop anything.
func TestColdCompactionAllLiveKeepsEverything(t *testing.T) {
	dir := t.TempDir()
	cfg := compactTestCfg(dir, false, 5_000_000)
	c := openCompactCache(t, cfg)

	var n int
	for i := 0; ; i++ {
		err := c.Put([]byte(fmt.Sprintf("uniq%04d", i)), compactVal(i, 0), 0)
		if errors.Is(err, ErrFull) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		n = i + 1
	}
	usedBefore := c.shards[0].bytesUsed()
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()
	if used := c2.shards[0].bytesUsed(); used != usedBefore {
		t.Fatalf("an all-live shard was rewritten: bytesUsed %d → %d", usedBefore, used)
	}
	for i := 0; i < n; i++ {
		if _, err := c2.Get([]byte(fmt.Sprintf("uniq%04d", i))); err != nil {
			t.Fatalf("live key %d lost on a shard with nothing to reclaim: %v", i, err)
		}
	}
}

// pageCopiesOf returns every PHYSICAL copy of key currently framed in the
// shard's pages, in page-then-offset order. This is the raw FILE truth,
// deliberately independent of what the index resolves to — the difference
// between "the read path answers with the newest value" and "the newest value
// still exists on disk at all".
func pageCopiesOf(s *shard, key []byte) [][]byte {
	var out [][]byte
	for _, p := range s.pages {
		tail := p.tail()
		entries := p.entries()
		for cursor := p.head(); cursor < tail; {
			k, v, _, err := decodeEntryFast(entries[cursor:tail])
			if err != nil {
				break
			}
			if bytes.Equal(k, key) {
				out = append(out, append([]byte(nil), v...))
			}
			cursor += entrySpanExact(len(k), len(v)) // EXACT: a walk over persisted mmap bytes
		}
	}
	return out
}

// fillPageAt writes padding until the shard's write cursor leaves page start,
// i.e. until that page is full. Returns the number of padding writes.
func fillPageAt(t *testing.T, c *Cache, tag string, start int) int {
	t.Helper()
	s := c.shards[0]
	for i := 0; i < 10_000; i++ {
		if err := c.Put([]byte(fmt.Sprintf("%s%04d", tag, i)), compactVal(7000+i, 0), 0); err != nil {
			t.Fatalf("padding write %d: %v", i, err)
		}
		if s.writeIdx != start {
			return i + 1
		}
	}
	t.Fatalf("the write cursor never left page %d", start)
	return 0
}

// TestColdCompactionResumesWritesAtPackFrontier pins the defect compaction USED
// TO ARM: permanent loss of a committed value on the second restart.
//
// A compaction packs the live set into pages 0..k and leaves k+1..N-1 EMPTY. If
// the shard then resumed writing at page N-1 (the default "write the last page
// first"), the first post-compaction writes would land in the HIGHEST page and
// everything after it — once N-1 filled — in LOWER pages, because
// firstPageWithRoomLocked always scans from 0. Write order would disagree with
// page order. rebuildIndexFromPages walks pages 0→N-1 and lets the LAST copy it
// sees win, so a key written into N-1 and then overwritten into a lower page
// resolves to its OLDER copy after a restart; entryIsLiveAtOpen then judges the
// NEWER copy not-index-current and physically DELETES it. A committed value is
// gone for good, and a peer that never restarted still serves it — silent
// cross-replica divergence.
//
// The fix is that a published compaction resumes writes at its PACK FRONTIER, so
// writes stay ascending and the rebuild picks the newest copy. This test drives
// exactly that sequence: compact → write K → fill K's page → overwrite K →
// restart, and demands the newer value both READS BACK and still EXISTS on disk
// after the restart's own compaction has had its say.
func TestColdCompactionResumesWritesAtPackFrontier(t *testing.T) {
	dir := t.TempDir()
	const wall = uint64(5_000_000)
	cfg := compactTestCfg(dir, true, wall) // replicated: the divergence case

	liveKey := func(i int) []byte { return []byte(fmt.Sprintf("live%04d", i)) }
	const nLive = 8

	// Round 1: build a shard that is mostly ghost bytes, so the next open compacts.
	c := openCompactCache(t, cfg)
	fillWithOverwrites(t, nLive, liveKey, func(k, v []byte) error { return c.Put(k, v, 0) })
	if occ := c.shards[0].occupancyRatio(); occ < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f < %.2f — the test must fill past the compaction mark",
			occ, mmapCompactMinOccupancy)
	}
	// The dry-run sizer must predict the pack EXACTLY: it is what bounds the
	// staging file's preallocation, and a pack that ran past that reservation
	// would be back to faulting blocks in on a possibly-full filesystem.
	wantPages := c.shards[0].packPagesNeeded(c.shards[0].compactDropClock())
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	// Round 2: compaction runs at open. Everything below happens on the packed
	// layout it leaves behind.
	c2 := openCompactCache(t, cfg)
	s2 := c2.shards[0]
	if n := s2.snapshot().Compactions; n != 1 {
		t.Fatalf("setup: compaction did not run at open (Compactions=%d, want 1)", n)
	}
	if got := s2.writeIdx + 1; got != wantPages {
		t.Fatalf("the pack filled %d pages but packPagesNeeded reserved %d — the staging "+
			"preallocation must cover exactly the pages the pack touches", got, wantPages)
	}

	victim := []byte("victim")
	older, newer := compactVal(4242, 1), compactVal(4242, 2)
	if err := c2.Put(victim, older, 0); err != nil {
		t.Fatalf("first write of the victim key: %v", err)
	}
	victimPage := s2.writeIdx
	fillPageAt(t, c2, "pf", victimPage) // fill the page the older copy landed in
	if err := c2.Put(victim, newer, 0); err != nil {
		t.Fatalf("overwrite of the victim key: %v", err)
	}
	overwritePage := s2.writeIdx
	// Churn the shard back up to the compaction mark so round 3 compacts too:
	// that is what turns a mis-resolved rebuild into a PHYSICAL deletion.
	fillWithOverwrites(t, 3, func(i int) []byte { return []byte(fmt.Sprintf("churn%02d", i)) },
		func(k, v []byte) error { return c2.Put(k, v, 0) })
	if err := c2.Close(); err != nil {
		t.Fatal(err)
	}

	// Round 3: restart. The rebuild must resolve the victim to its NEWER copy,
	// and this open's compaction must therefore keep those bytes.
	c3 := openCompactCache(t, cfg)
	defer func() { _ = c3.Close() }()
	s3 := c3.shards[0]

	got, err := c3.Get(victim)
	if err != nil {
		t.Fatalf("victim key lost across the restart: %v", err)
	}
	if !bytes.Equal(got, newer) {
		// Not fatal: the physical assertions below are the other half of the
		// failure (the resurrection is what makes compaction delete the newer
		// copy), and seeing both at once is what makes the defect legible.
		t.Errorf("victim resurrected an OLDER value after restart: got %q…, want %q… "+
			"(the rebuild resolved the wrong physical copy)", got[:16], newer[:16])
	}

	copies := pageCopiesOf(s3, victim)
	var haveNewer, haveOlder bool
	for _, v := range copies {
		if bytes.Equal(v, newer) {
			haveNewer = true
		}
		if bytes.Equal(v, older) {
			haveOlder = true
		}
	}
	if !haveNewer {
		t.Fatalf("the NEWER copy of the victim key is no longer on disk (%d physical copies remain) — "+
			"a committed value was permanently deleted", len(copies))
	}
	if n := s3.snapshot().Compactions; n != 1 {
		t.Fatalf("round 3 did not compact (Compactions=%d, want 1) — the physical-deletion half "+
			"of this test never ran", n)
	}
	if haveOlder {
		t.Fatal("the superseded OLDER copy survived a compaction that kept the newer one")
	}
	// The mechanism behind all of the above, stated directly.
	if overwritePage <= victimPage {
		t.Fatalf("the overwrite landed in page %d, at or BELOW the older copy's page %d — "+
			"post-compaction writes must continue in ascending page order",
			overwritePage, victimPage)
	}
}

// TestColdCompactionKillSwitch: DisableColdCompaction is the operational escape
// hatch for a default-on rewrite of the durable file. With it set, a shard that
// would otherwise compact at open must leave the pages file exactly as it found
// it — and still open, rebuild and serve every key.
func TestColdCompactionKillSwitch(t *testing.T) {
	dir := t.TempDir()
	cfg := compactTestCfg(dir, false, 5_000_000)

	key := func(i int) []byte { return []byte(fmt.Sprintf("live%04d", i)) }
	const nLive = 8

	c := openCompactCache(t, cfg)
	fillWithOverwrites(t, nLive, key, func(k, v []byte) error { return c.Put(k, v, 0) })
	usedBefore := c.shards[0].bytesUsed()
	if occ := c.shards[0].occupancyRatio(); occ < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f is below the compaction mark %.2f", occ, mmapCompactMinOccupancy)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	off := cfg
	off.DisableColdCompaction = true
	c2 := openCompactCache(t, off)
	if used := c2.shards[0].bytesUsed(); used != usedBefore {
		t.Fatalf("the pages file was rewritten with compaction disabled: bytesUsed %d → %d", usedBefore, used)
	}
	if st := c2.Stats(); st.Compactions != 0 {
		t.Fatalf("Compactions=%d with the kill switch on, want 0", st.Compactions)
	}
	for i := 0; i < nLive; i++ {
		if _, err := c2.Get(key(i)); err != nil {
			t.Fatalf("live key %d unreadable with compaction disabled: %v", i, err)
		}
	}
	if err := c2.Close(); err != nil {
		t.Fatal(err)
	}

	// Flipping it back off compacts the same untouched file, and the counters
	// report what it did.
	c3 := openCompactCache(t, cfg)
	defer func() { _ = c3.Close() }()
	st := c3.Stats()
	if st.Compactions != 1 {
		t.Fatalf("Compactions=%d after re-enabling, want 1", st.Compactions)
	}
	if st.CompactionsAborted != 0 {
		t.Fatalf("CompactionsAborted=%d, want 0", st.CompactionsAborted)
	}
	if st.CompactionBytesReclaimed == 0 || st.CompactionBytesReclaimed != usedBefore-st.BytesUsed {
		t.Fatalf("CompactionBytesReclaimed=%d, want %d (bytesUsed %d → %d)",
			st.CompactionBytesReclaimed, usedBefore-st.BytesUsed, usedBefore, st.BytesUsed)
	}
}
