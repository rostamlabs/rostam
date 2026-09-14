// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"fmt"
	"testing"
)

// Warm-restart resolution tests (#12A / #12B).
//
// An mmap shard rebuilds its index at open by walking pages 0→N-1 and letting the
// LAST copy of a key it sees win. That is only correct if page order equals WRITE
// order — and it does not. findOrMakePageLocked falls back to
// firstPageWithRoomLocked, which scans from index 0, so a write REVISITS any lower
// page that still has tail room. A large entry that skipped a hole leaves exactly
// such a hole behind, and the next entry small enough to fit it lands BELOW an
// earlier write.
//
// TestWarmRestartResolvesNewestAcrossPageRevisit builds that inversion from the
// revisit alone. It deliberately does NOT depend on newShard's "start at the last
// page" choice: it first drives the shard's open page down to page 0 whatever it
// started at, so the inversion it produces survives any change to writeIdx's
// initial value. That is the point — reshuffling the start page cannot make this
// test pass; only a persisted write-recency signal can.

// warmRestartCfg is a single-shard mmap reject-writes cache (what replication
// forces via B2) with an 8-page budget, so the three pages this test touches keep
// occupancy well below mmapCompactMinOccupancy and cold compaction never runs.
func warmRestartCfg(dir string) Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 8 << 20 // 8 mmap pages
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = dir
	cfg.NowFn = func() uint64 { return 5_000_000 }
	return cfg
}

// padValue returns a deterministic filler value of exactly n bytes.
func padValue(n int) []byte { return bytes.Repeat([]byte("p"), n) }

// padPageDownTo writes distinct padding keys into the shard's CURRENT open page
// until its contiguous tail room is exactly `leave` bytes, and returns that page's
// index. Every write is sized to stay inside the open page, so the page the shard
// is writing into never changes underneath the loop.
func padPageDownTo(t *testing.T, c *Cache, leave int) int {
	t.Helper()
	s := c.shards[0]
	idx := s.writeIdx
	for i := 0; ; i++ {
		free := s.pages[idx].FreeTail()
		if free <= leave {
			return idx
		}
		key := []byte(fmt.Sprintf("pad-%02d-%06d", idx, i))
		room := free - leave
		// OCCUPANCY throughout: this lands the page's FreeTail on an exact figure, so
		// it has to reason in the room a Put consumes.
		if room < entrySpan(len(key), 1, true) {
			t.Fatalf("cannot land exactly on %d free bytes: %d left, entry overhead %d",
				leave, room, entrySpan(len(key), 0, true))
		}
		vlen := room - entrySpan(len(key), 0, true)
		if vlen > 64<<10 {
			vlen = 64 << 10
		}
		if err := c.Put(key, padValue(vlen), 0); err != nil {
			t.Fatalf("padding write: %v", err)
		}
		if s.writeIdx != idx {
			t.Fatalf("padding write escaped page %d to %d", idx, s.writeIdx)
		}
	}
}

// openAtPageZero drives the shard until its open page is page 0, whatever page
// newShard chose to start on. Returns with page 0 selected and untouched apart
// from at most one tiny seed entry.
func openAtPageZero(t *testing.T, c *Cache) {
	t.Helper()
	s := c.shards[0]
	for i := 0; s.writeIdx != 0; i++ {
		if i > len(s.pages) {
			t.Fatalf("write cursor never reached page 0 (stuck at %d)", s.writeIdx)
		}
		padPageDownTo(t, c, 0)
		// The open page is now full, so the next write falls through to
		// firstPageWithRoomLocked, which scans from index 0.
		if err := c.Put([]byte(fmt.Sprintf("seed-%02d", i)), padValue(16), 0); err != nil {
			t.Fatalf("seed write: %v", err)
		}
	}
}

// keyPageIdx returns the page index the shard's index currently resolves key to.
func keyPageIdx(t *testing.T, c *Cache, key []byte) int {
	t.Helper()
	s := c.shards[0]
	_, ref, ok := s.tab.Load().findSlot(hashKey(key))
	if !ok {
		t.Fatalf("key %q is not indexed", key)
	}
	return int(ref.pageIdx())
}

// TestWarmRestartResolvesNewestAcrossPageRevisit is the revisit-inversion case: a
// key is first written into a HIGH page (because a hole in a lower page was too
// small for it) and later overwritten into that LOWER page (because it is now small
// enough to fit the hole). Write order and page order disagree with NO help from
// writeIdx's starting value, and the warm restart must still resolve the key to the
// value written LAST (#12A).
//
// This is the test that distinguishes a real fix from a writeIdx reshuffle. Where
// writes START is irrelevant here — the shard's cursor is driven down to page 0
// first — so nothing but a persisted write-recency signal can make it pass.
func TestWarmRestartResolvesNewestAcrossPageRevisit(t *testing.T) {
	const (
		hole     = 300 << 10 // tail room deliberately left in the lower page
		bigVal   = 400 << 10 // too big for the hole → skips to the next page
		smallVal = 8 << 10   // fits the hole → revisits the lower page
	)
	dir := t.TempDir()
	cfg := warmRestartCfg(dir)
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s := c.shards[0]

	openAtPageZero(t, c)

	// Leave a hole in page 0 that a big entry cannot use.
	lower := padPageDownTo(t, c, hole)
	if lower != 0 {
		t.Fatalf("expected to be filling page 0, got %d", lower)
	}

	k := []byte("revisit-key")
	older := append([]byte("OLDER|"), padValue(bigVal-6)...)
	newer := append([]byte("NEWER|"), padValue(smallVal-6)...)

	// (1) The big write does not fit page 0's hole, so it lands in page 1.
	if err := c.Put(k, older, 0); err != nil {
		t.Fatal(err)
	}
	upper := keyPageIdx(t, c, k)
	if upper <= lower {
		t.Fatalf("setup: the big write landed in page %d, want a page above %d", upper, lower)
	}

	// (2) Fill the page it landed in, so the next write must fall through to
	// firstPageWithRoomLocked — which scans from index 0 and finds the hole.
	if got := padPageDownTo(t, c, 0); got != upper {
		t.Fatalf("padding filled page %d, want the key's page %d", got, upper)
	}

	// (3) The small overwrite REVISITS the lower page: later in time, lower on disk.
	if err := c.Put(k, newer, 0); err != nil {
		t.Fatal(err)
	}
	if got := keyPageIdx(t, c, k); got != lower {
		t.Fatalf("the overwrite landed in page %d, want the revisited lower page %d — "+
			"the inversion this test needs was not constructed", got, lower)
	}
	if v, gErr := c.Get(k); gErr != nil || !bytes.Equal(v, newer) {
		t.Fatalf("pre-restart read is already wrong (err=%v); the setup is broken", gErr)
	}
	if occ := s.occupancyRatio(); occ >= mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f would trigger cold compaction; this must test the plain rebuild", occ)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c2.Close() }()

	v, err := c2.Get(k)
	if err != nil {
		t.Fatalf("Get after warm restart: %v", err)
	}
	if bytes.Equal(v, older) {
		t.Fatal("warm restart resolved the key to the OLDER copy: a committed overwrite " +
			"was silently lost because the newer copy sits at a LOWER page index (#12A)")
	}
	if !bytes.Equal(v, newer) {
		t.Fatalf("warm restart resolved the key to neither copy (%d bytes)", len(v))
	}
}

// liveMetaByKey returns the meta word of the entry the index CURRENTLY resolves
// each key to. Compaction must carry these through untouched.
func liveMetaByKey(s *shard) map[string]uint64 {
	out := map[string]uint64{}
	t := s.tab.Load()
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		ref := slabRef(t.refs[i].Load())
		p := s.pages[ref.pageIdx()]
		k, _, _, err := p.Read(ref.offset())
		if err != nil {
			continue
		}
		m, ok := p.MetaAt(ref.offset())
		if !ok {
			continue
		}
		out[string(k)] = m
	}
	return out
}

// physicalCopies counts every framed entry on every page whose key equals key —
// live, superseded, or tombstone. It is how a test asserts that bytes were
// genuinely reclaimed rather than merely unreferenced.
func physicalCopies(s *shard, key []byte) int {
	n := 0
	for _, p := range s.pages {
		entries := p.entries()
		tail := p.tail()
		for cursor := p.head(); cursor < tail; {
			k, v, _, err := decodeEntryFast(entries[cursor:tail])
			if err != nil {
				break
			}
			if bytes.Equal(k, key) {
				n++
			}
			cursor += entrySpanExact(len(k), len(v)) // EXACT: a walk over persisted mmap bytes
		}
	}
	return n
}

// TestWarmRestartDeleteSurvivesTwoRestarts proves the delete tombstone is itself
// DURABLE, not merely effective once. One restart could be explained by the
// tombstone happening to out-rank the entry in a single rebuild; a second restart
// re-reads the same pages with no in-memory state carried over at all, so the
// tombstone must still be on disk, still win the sequence contest, and still strip
// the slot. Occupancy is kept below the compaction mark throughout, so nothing is
// physically removed — the key is held down by the tombstone alone.
func TestWarmRestartDeleteSurvivesTwoRestarts(t *testing.T) {
	dir := t.TempDir()
	cfg := warmRestartCfg(dir)

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	gone, kept := []byte("gone"), []byte("kept")
	if err := c.Put(gone, []byte("value-that-must-not-return"), 0); err != nil {
		t.Fatal(err)
	}
	if err := c.Put(kept, []byte("value-that-must-survive"), 0); err != nil {
		t.Fatal(err)
	}
	ok, err := c.Del(gone)
	if err != nil {
		t.Fatalf("Del: %v", err)
	}
	if !ok {
		t.Fatal("Del reported the key absent")
	}
	if n := physicalCopies(c.shards[0], gone); n != 2 {
		t.Fatalf("physical copies of the deleted key = %d, want 2 (the value and its tombstone) — "+
			"the delete was not recorded on the page", n)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	for round := 1; round <= 2; round++ {
		c2, err := New(cfg)
		if err != nil {
			t.Fatalf("restart %d: %v", round, err)
		}
		if occ := c2.shards[0].occupancyRatio(); occ >= mmapCompactMinOccupancy {
			t.Fatalf("restart %d: occupancy %.3f triggered compaction; the tombstone must be "+
				"what keeps the key gone, not the reclamation", round, occ)
		}
		if _, err := c2.Get(gone); err != ErrNotFound {
			t.Fatalf("restart %d: deleted key came back (err=%v, want ErrNotFound) — the "+
				"tombstone did not survive to this rebuild (#12B)", round, err)
		}
		if v, err := c2.Get(kept); err != nil || !bytes.Equal(v, []byte("value-that-must-survive")) {
			t.Fatalf("restart %d: the untouched key was collateral damage: %q err=%v", round, v, err)
		}
		if err := c2.Close(); err != nil {
			t.Fatalf("restart %d close: %v", round, err)
		}
	}
}

// TestWarmRestartDeleteSurvivesOnRingbufShard is #12B for the DEFAULT capacity
// policy. DefaultConfig sets PolicyRingbufEvict and only shard/store.go's
// REPLICATED path overrides it to reject-writes, so a single-node persistent
// cache — a plain Store, or a directStore with a DataDir — runs a persistent
// RINGBUF shard. It rebuilds its index from page bytes at open exactly like the
// reject-writes profile does, so an index-only delete came straight back.
//
// Eviction is not a defence: it recycles pages eventually, and eventually is not a
// bound. Here the shard is nowhere near full, so nothing is ever evicted and the
// deleted key's bytes sit on the page indefinitely, waiting for the next restart
// to re-index them.
func TestWarmRestartDeleteSurvivesOnRingbufShard(t *testing.T) {
	dir := t.TempDir()
	cfg := warmRestartCfg(dir)
	cfg.AtCapPolicy = PolicyRingbufEvict // what DefaultConfig gives a non-replicated shard

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	gone, kept := []byte("gone"), []byte("kept")
	if err := c.Put(gone, []byte("value-that-must-not-return"), 0); err != nil {
		t.Fatal(err)
	}
	if err := c.Put(kept, []byte("value-that-must-survive"), 0); err != nil {
		t.Fatal(err)
	}
	ok, err := c.Del(gone)
	if err != nil {
		t.Fatalf("Del: %v", err)
	}
	if !ok {
		t.Fatal("Del reported the key absent")
	}
	if n := physicalCopies(c.shards[0], gone); n != 2 {
		t.Fatalf("physical copies of the deleted key = %d, want 2 (the value and its tombstone) — "+
			"a persistent ringbuf shard recorded the delete in the index only", n)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c2.Close() }()
	if occ := c2.shards[0].occupancyRatio(); occ >= mmapCompactMinOccupancy {
		t.Fatalf("occupancy %.3f triggered compaction; the tombstone must be what keeps "+
			"the key gone, not the reclamation", occ)
	}
	if _, err := c2.Get(gone); err != ErrNotFound {
		t.Fatalf("deleted key came back after restart (err=%v, want ErrNotFound): the delete "+
			"was never written to the page because the shard's policy is ringbuf-evict (#12B)", err)
	}
	if v, err := c2.Get(kept); err != nil || !bytes.Equal(v, []byte("value-that-must-survive")) {
		t.Fatalf("the untouched key was collateral damage: %q err=%v", v, err)
	}
}

// liveSlotCount counts the slots an index table actually holds as occupied, which
// must equal its bookkeeping counter t.live. rehashed() sizes the replacement
// table from t.live, so an understated counter can allocate a table too small for
// the entries copied into it — and upsert probing a table with no empty slot never
// terminates.
func liveSlotCount(t *indexTable) int {
	n := 0
	for i := range t.ctrl {
		if c := t.ctrl[i].Load(); c != ctrlEmpty && c != ctrlTombstone {
			n++
		}
	}
	return n
}

// TestRingbufTombstoneAppendThatEvictsKeepsIndexConsistent covers the hazard that
// extending durable tombstones to ringbuf shards introduces. delH used to capture
// its index slot BEFORE appending, justified by "a reject-writes shard cannot
// evict". A ringbuf shard can: the append falls through to evictUntilFitsLocked,
// which drops the index slot of every entry it drains — including, here, the very
// key being deleted. Acting on the pre-append slot afterwards then tombstones a
// slot eviction already tombstoned, double-counting live/tomb.
//
// The shard is given a SINGLE page so making room for the tombstone necessarily
// drains the key itself, which is the case that goes wrong.
func TestRingbufTombstoneAppendThatEvictsKeepsIndexConsistent(t *testing.T) {
	dir := t.TempDir()
	cfg := warmRestartCfg(dir)
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.MaxMemoryPerShard = cfg.PageSize // one page: any eviction drains everything

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s := c.shards[0]

	// A long key so its tombstone (entryHeaderSize + len(key)) is bigger than the
	// tail room deliberately left below.
	victim := bytes.Repeat([]byte("v"), 512)
	if err := c.Put(victim, []byte("x"), 0); err != nil {
		t.Fatal(err)
	}
	const leave = 300
	padPageDownTo(t, c, leave)
	// OCCUPANCY: the room the tombstone append will ask findOrMakePageLocked for.
	if need := entrySpan(len(victim), 0, true); need <= leave {
		t.Fatalf("setup: the tombstone needs %d bytes but %d are free — no eviction "+
			"would be forced", need, leave)
	}

	evictionsBefore := s.evictions.Load()
	ok, err := c.Del(victim)
	if err != nil {
		t.Fatalf("Del: %v", err)
	}
	if !ok {
		t.Fatal("Del reported the key absent even though it was live")
	}
	if s.evictions.Load() == evictionsBefore {
		t.Fatal("setup: appending the tombstone did not evict anything, so the stale-slot " +
			"hazard was never exercised")
	}
	if _, err := c.Get(victim); err != ErrNotFound {
		t.Fatalf("Get right after Del = %v, want ErrNotFound: the delete did not remove the "+
			"key from the LIVE index", err)
	}
	tab := s.tab.Load()
	if got, want := tab.live, liveSlotCount(tab); got != want {
		t.Fatalf("index bookkeeping corrupted: table.live=%d but %d slots are occupied — "+
			"the pre-append slot was tombstoned a second time after eviction had already "+
			"dropped it (rehashed() sizes the next table from this counter)", got, want)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c2.Close() }()
	if _, err := c2.Get(victim); err != ErrNotFound {
		t.Fatalf("deleted key came back after restart (err=%v, want ErrNotFound)", err)
	}
}

// TestWarmRestartDeleteThenRewriteKeepsTheRewrite is the case that rules out both
// tempting shortcuts in the rebuild. A key is written, DELETED, and written AGAIN.
// The tombstone must not win (it is older than the rewrite), and the pre-delete
// copy must not win either. Only sequence order gets this right: a rebuild that
// applied tombstones on sight would drop the rewrite, and one that subtracted a
// set of tombstoned keys at the end would drop it too.
func TestWarmRestartDeleteThenRewriteKeepsTheRewrite(t *testing.T) {
	dir := t.TempDir()
	cfg := warmRestartCfg(dir)

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	k := []byte("phoenix")
	if err := c.Put(k, []byte("first"), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Del(k); err != nil {
		t.Fatal(err)
	}
	if err := c.Put(k, []byte("second"), 0); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c2.Close() }()
	v, err := c2.Get(k)
	if err != nil {
		t.Fatalf("Get after restart = %v; the rewrite was swallowed by the tombstone that "+
			"PRECEDED it — the rebuild is not resolving by sequence", err)
	}
	if !bytes.Equal(v, []byte("second")) {
		t.Fatalf("Get after restart = %q, want \"second\"", v)
	}
}

// TestColdCompactionDropsTombstoneAndHistory: once a key is deleted, compaction
// must reclaim its ENTIRE byte history — every superseded copy AND the tombstone
// itself — in a single pass, and the key must stay gone across a further restart
// (i.e. dropping the tombstone did not un-delete anything).
func TestColdCompactionDropsTombstoneAndHistory(t *testing.T) {
	dir := t.TempDir()
	cfg := compactTestCfg(dir, false, 5_000_000)

	c := openCompactCache(t, cfg)
	s := c.shards[0]
	doomed := []byte("doomed")
	keep := func(i int) []byte { return []byte(fmt.Sprintf("keep%04d", i)) }

	// Give the doomed key a long history of superseded copies, then fill the shard
	// past the compaction mark with other keys.
	for round := 0; round < 6; round++ {
		if err := c.Put(doomed, compactVal(7000, round), 0); err != nil {
			t.Fatal(err)
		}
	}
	fillWithOverwrites(t, 6, keep, func(k, v []byte) error { return c.Put(k, v, 0) })

	if n := physicalCopies(s, doomed); n < 6 {
		t.Fatalf("setup: only %d physical copies of the doomed key, want >= 6", n)
	}
	ok, err := c.Del(doomed)
	if err != nil {
		t.Fatalf("Del: %v", err)
	}
	if !ok {
		t.Fatal("Del reported the doomed key absent")
	}
	usedBefore := s.bytesUsed()
	if occ := s.occupancyRatio(); occ < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f < %.2f — compaction would not run", occ, mmapCompactMinOccupancy)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2 := openCompactCache(t, cfg)
	s2 := c2.shards[0]
	if got := c2.Stats().Compactions; got != 1 {
		t.Fatalf("compactions = %d, want 1", got)
	}
	if _, err := c2.Get(doomed); err != ErrNotFound {
		t.Fatalf("deleted key survived compaction: err=%v, want ErrNotFound", err)
	}
	if n := physicalCopies(s2, doomed); n != 0 {
		t.Fatalf("%d physical copies of the deleted key survived compaction (tombstone and/or "+
			"history not reclaimed)", n)
	}
	if used := s2.bytesUsed(); used >= usedBefore {
		t.Fatalf("compaction reclaimed nothing: bytesUsed %d → %d", usedBefore, used)
	}
	for i := 0; i < 6; i++ {
		if _, err := c2.Get(keep(i)); err != nil {
			t.Fatalf("live key %d dropped by compaction: %v", i, err)
		}
	}
	if err := c2.Close(); err != nil {
		t.Fatal(err)
	}

	// A SECOND restart: with the tombstone reclaimed there is nothing left saying
	// "deleted", so this checks that nothing was left behind to resurrect either.
	c3 := openCompactCache(t, cfg)
	defer func() { _ = c3.Close() }()
	if _, err := c3.Get(doomed); err != ErrNotFound {
		t.Fatalf("deleted key returned after a second restart: err=%v, want ErrNotFound", err)
	}
}

// TestCompactionPreservesEntryMeta pins the invariant compaction's correctness
// rests on: it may DROP entries, but it must never ALTER a surviving one. If it
// re-stamped sequences (say, in walk order) the post-compaction rebuild would
// resolve keys against recency compaction invented rather than the order the
// writes actually happened in — reintroducing #12A through the back door.
func TestCompactionPreservesEntryMeta(t *testing.T) {
	dir := t.TempDir()
	cfg := compactTestCfg(dir, false, 5_000_000)

	c := openCompactCache(t, cfg)
	s := c.shards[0]
	key := func(i int) []byte { return []byte(fmt.Sprintf("meta%04d", i)) }
	fillWithOverwrites(t, 8, key, func(k, v []byte) error { return c.Put(k, v, 0) })

	before := liveMetaByKey(s)
	if len(before) < 8 {
		t.Fatalf("setup: only %d live keys, want >= 8", len(before))
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2 := openCompactCache(t, cfg)
	defer func() { _ = c2.Close() }()
	if got := c2.Stats().Compactions; got != 1 {
		t.Fatalf("compactions = %d, want 1 — nothing was rewritten, so this proves nothing", got)
	}
	after := liveMetaByKey(c2.shards[0])
	if len(after) != len(before) {
		t.Fatalf("live key count changed across compaction: %d → %d", len(before), len(after))
	}
	for k, want := range before {
		got, ok := after[k]
		if !ok {
			t.Errorf("key %q lost by compaction", k)
			continue
		}
		if got != want {
			t.Errorf("key %q: meta rewritten by compaction: %#x → %#x (seq %d → %d)",
				k, want, got, metaSeq(want), metaSeq(got))
		}
	}
}

// TestWarmRestartExpiredOverwriteDoesNotRevert closes the last way the rebuild
// could hand back a stale value.
//
// A NON-replicated shard drops TTL-expired entries at rebuild time (a replicated
// one must not — wall-clock drops would diverge replicas). If it simply SKIPS an
// expired entry, an older non-expired copy of the same key is still sitting on the
// page and gets indexed instead: the key does not go away, it REVERTS to a value
// that was overwritten. The overwrite's sequence has to suppress the older copy
// even though the overwrite itself is being dropped.
func TestWarmRestartExpiredOverwriteDoesNotRevert(t *testing.T) {
	const wall = uint64(5_000_000)
	dir := t.TempDir()
	cfg := warmRestartCfg(dir) // non-replicated, wall clock pinned at 5_000_000

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	k := []byte("reverting-key")
	if err := c.Put(k, []byte("older-no-ttl"), 0); err != nil { // never expires
		t.Fatal(err)
	}
	// Overwrite with an entry whose ABSOLUTE expiry is already in the past.
	if err := c.PutAbs(k, []byte("newer-expired"), wall-1_000); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Get(k); err != ErrNotFound {
		t.Fatalf("pre-restart Get = %v, want ErrNotFound (the newest copy is expired)", err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c2.Close() }()
	v, err := c2.Get(k)
	if err == nil {
		t.Fatalf("warm restart resurrected an overwritten value (%q): the expired NEWER copy "+
			"was skipped and the older copy took the slot", v)
	}
	if err != ErrNotFound {
		t.Fatalf("Get after restart = %v, want ErrNotFound", err)
	}
}
