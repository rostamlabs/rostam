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
	"strings"
	"testing"
)

// Ordered-bounds recovery: the WRITE-PATH guards for the torn-writeback fix.
//
// WHY THESE, AND NOT A FLIPPED IMAGE-SPLICE TEST. The confirmed skew repro
// (test/recovery-stale-frame-skew) splices a pathological image straight into
// pages.dat and reopens. rebuildIndexFromPages is byte-deterministic — it decodes
// identical bytes identically whether or not this fix is present — because the fix
// deliberately does NOT change recovery's decode. The defence is entirely in the
// WRITE/SYNC/REUSE path: the durable per-page header (head/tail) is no longer
// written by an append; it is a PROJECTION published only at a sync point AFTER the
// entries it names are flushed, and it is zeroed+flushed before any extent is
// reused. So a durable header can only ever LAG the durable entries, never lead
// them — and a header that never leads can never frame a previous life's bytes.
//
// The tests below assert that invariant on the write path, where it is observable
// and where "fixed" is distinguishable from "unfixed": on the unfixed code the
// append wrote the tail straight through to the mapped header, so several of these
// would fail.

// orderedBoundsCfg is a small single-shard mmap cache with cold compaction OFF (so
// recovery output is not reshaped by a compaction pass), no TTL sweeper, and
// Durable unset (so no background msyncLoop projects bounds behind the test's
// back). DataDir is filled in by the caller.
func orderedBoundsCfg(dir string) Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 4 << 20 // four pages
	cfg.TTLSweepIntervalMs = 0
	cfg.DisableColdCompaction = true
	cfg.DataDir = filepath.Join(dir, "cache")
	return cfg
}

// TestAppendDoesNotAdvanceDurableHeader is the core invariant, and it DISTINGUISHES
// the fix: an append advances only the runtime bounds. Until a sync point projects
// them, the durable per-page header stays at its seeded value — (0,0) on a fresh
// file. On the unfixed code setTail wrote the new tail straight into the mapped
// header, so the durable tail would already be non-zero here and this fails.
func TestAppendDoesNotAdvanceDurableHeader(t *testing.T) {
	c, err := New(orderedBoundsCfg(t.TempDir()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	for i := 0; i < 8; i++ {
		if err := c.Put([]byte(fmt.Sprintf("k%03d", i)), bytes.Repeat([]byte("v"), 128), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}

	sawRuntime := false
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i, p := range s.pages {
		rt := p.tail()
		dh, dt := p.durableBounds()
		if rt > 0 {
			sawRuntime = true
			if dh != 0 || dt != 0 {
				t.Fatalf("page %d: an append advanced the DURABLE header to (%d,%d) with runtime tail %d; "+
					"the fix requires the durable header to lag until a sync point projects it", i, dh, dt, rt)
			}
		}
	}
	if !sawRuntime {
		t.Fatal("no page received the writes; test set up nothing to observe")
	}
}

// TestSyncPointProjectsBoundsAndReopenRecovers checks the other half: a forced sync
// point (SetAppliedIndex) projects the runtime bounds into the durable header, and
// a reopen then recovers every key. This is the correctness round-trip that the
// lagging-header design must not break — if projection were missing, the reopen
// would recover nothing.
func TestSyncPointProjectsBoundsAndReopenRecovers(t *testing.T) {
	dir := t.TempDir()
	cfg := orderedBoundsCfg(dir)
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	const n = 12
	val := func(i int) []byte { return bytes.Repeat([]byte{byte('A' + i)}, 200) }
	for i := 0; i < n; i++ {
		if err := c.Put([]byte(fmt.Sprintf("key%03d", i)), val(i), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}

	// A forced applied-index update is a sync point: it flushes the data, projects
	// every page's bounds, then flushes the region.
	c.SetAppliedIndex(5, true)

	s := c.shards[0]
	s.mu.RLock()
	for i, p := range s.pages {
		dh, dt := p.durableBounds()
		if dh != p.head() || dt != p.tail() {
			s.mu.RUnlock()
			t.Fatalf("page %d durable header (%d,%d) != runtime (%d,%d) after a forced sync point",
				i, dh, dt, p.head(), p.tail())
		}
	}
	s.mu.RUnlock()

	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = c2.Close() }()
	for i := 0; i < n; i++ {
		got, err := c2.Get([]byte(fmt.Sprintf("key%03d", i)))
		if err != nil || !bytes.Equal(got, val(i)) {
			t.Fatalf("key %d not recovered after sync+reopen: err=%v", i, err)
		}
	}
	if idx := c2.AppliedIndex(); idx != 5 {
		t.Fatalf("applied index after reopen = %d, want 5", idx)
	}
}

// TestCleanCloseRecoversRecentWrites guards the Close sync point specifically: a
// clean shutdown with NO intervening SetAppliedIndex/SetPBFrontier must still make
// the writes durable. Close now projects every page's bounds around its final
// flush; without that the reopen would silently drop everything written since the
// last sync (the durable header would still lag).
func TestCleanCloseRecoversRecentWrites(t *testing.T) {
	dir := t.TempDir()
	cfg := orderedBoundsCfg(dir)
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	const n = 10
	for i := 0; i < n; i++ {
		if err := c.Put([]byte(fmt.Sprintf("c%03d", i)), []byte(fmt.Sprintf("val-%d", i)), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	if err := c.Close(); err != nil { // the ONLY durability event in this test
		t.Fatalf("Close: %v", err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = c2.Close() }()
	for i := 0; i < n; i++ {
		got, err := c2.Get([]byte(fmt.Sprintf("c%03d", i)))
		if err != nil || string(got) != fmt.Sprintf("val-%d", i) {
			t.Fatalf("clean-close write %d lost across reopen: %q err=%v", i, got, err)
		}
	}
}

// TestReusedExtentDurableHeaderStaysZeroUntilSync is the REVERSE-SKEW defence,
// observable on the write path and DISTINGUISHING. When a page is drained empty for
// reuse, its durable header is zeroed and flushed; subsequent appends into the
// reused extent advance only the runtime bounds, so the durable header stays (0,0)
// until a sync point re-projects it. A crash in that window therefore recovers an
// EMPTY page — never the old (larger) bound framing the reused bytes. On the
// unfixed code the drain's reset and the following appends both wrote the header
// through, so the durable tail would be non-zero here.
func TestReusedExtentDurableHeaderStaysZeroUntilSync(t *testing.T) {
	dir := t.TempDir()
	s, err := newShard(orderedBoundsCfg(dir), filepath.Join(dir, "s0"), nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Fill the active write page with a few entries.
	for i := 0; i < 4; i++ {
		if err := s.Put([]byte(fmt.Sprintf("old%02d", i)), bytes.Repeat([]byte("o"), 64), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	victim := -1
	s.mu.Lock()
	for i, p := range s.pages {
		if p.tail() > 0 {
			victim = i
			// Simulate a prior sync so the durable header is non-zero BEFORE the drain,
			// making the drain's zeroing observable rather than a no-op.
			p.projectBounds(p.head(), p.tail())
			break
		}
	}
	s.mu.Unlock()
	if victim < 0 {
		t.Fatal("no page received the writes")
	}
	if _, dt := s.pages[victim].durableBounds(); dt == 0 {
		t.Fatal("setup: durable tail still 0 after the simulated projection")
	}

	// Drain the page: every entry evicted, the page reset for reuse.
	s.mu.Lock()
	derr := s.drainPageLocked(victim)
	s.mu.Unlock()
	if derr != nil {
		t.Fatalf("drainPageLocked: %v", derr)
	}
	if dh, dt := s.pages[victim].durableBounds(); dh != 0 || dt != 0 {
		t.Fatalf("durable header after drain-to-reuse = (%d,%d), want (0,0) — the reused extent's old bound survived", dh, dt)
	}

	// Write NEW entries into the reused extent with NO sync point in between.
	s.mu.Lock()
	_, _, werr := s.pages[victim].Write([]byte("newlife"), bytes.Repeat([]byte("x"), 48), 0, makeMeta(1, false))
	rt := s.pages[victim].tail()
	s.mu.Unlock()
	if werr != nil {
		t.Fatalf("write into reused page: %v", werr)
	}
	if rt == 0 {
		t.Fatal("runtime tail did not advance after the reuse write")
	}
	if dh, dt := s.pages[victim].durableBounds(); dh != 0 || dt != 0 {
		t.Fatalf("durable header names reused-extent writes before any sync: (%d,%d) with runtime tail %d; "+
			"a crash here would recover the new bytes under a stale bound (reverse skew)", dh, dt, rt)
	}
}

// TestDurableHeaderNeverLeadsEntriesAfterCleanClose is the end-to-end necessary
// condition: drive the real write path (writes, overwrites, a delete), close
// cleanly, then read the ACTUAL durable bytes from pages.dat and confirm every
// page's persisted [head,tail) decodes fully with nothing dangling past tail. A
// header that led its entries would frame garbage or a stale record here. (This
// holds on the unfixed code too under a clean close — it is a regression guard for
// the projection logic, not a fixed-vs-unfixed discriminator.)
func TestDurableHeaderNeverLeadsEntriesAfterCleanClose(t *testing.T) {
	dir := t.TempDir()
	cfg := orderedBoundsCfg(dir)
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 20; i++ {
		if err := c.Put([]byte(fmt.Sprintf("k%03d", i)), bytes.Repeat([]byte("v"), 300), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	// Overwrite and delete some keys so pages carry dead duplicates + a tombstone.
	for i := 0; i < 5; i++ {
		if err := c.Put([]byte(fmt.Sprintf("k%03d", i)), bytes.Repeat([]byte("w"), 300), 0); err != nil {
			t.Fatalf("overwrite %d: %v", i, err)
		}
	}
	if _, err := c.Del([]byte("k007")); err != nil {
		t.Fatalf("Del: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(cfg.DataDir, "shard-0000", "pages.dat"))
	if err != nil {
		t.Fatalf("read pages.dat: %v", err)
	}
	fk, ok := readFramingKey(raw)
	if !ok {
		t.Fatal("pages.dat has no readable framing key")
	}
	for pg := 0; pg < cfg.MaxPagesPerShard(); pg++ {
		base := headerSize + pg*cfg.PageSize
		head := int(binary.LittleEndian.Uint32(raw[base : base+4]))
		tail := int(binary.LittleEndian.Uint32(raw[base+4 : base+8]))
		nonce := binary.LittleEndian.Uint64(raw[base+8 : base+16]) // per-page durable nonce
		entriesBase := base + pageHdrSize
		if head < 0 || tail < head || tail > cfg.PageSize-pageHdrSize {
			t.Fatalf("page %d durable bounds out of range: head=%d tail=%d", pg, head, tail)
		}
		for cursor := head; cursor < tail; {
			// Verify each framed entry's MAC at its durable (nonce, offset): if the
			// durable header ever led its entries, the bytes under [head,tail) would not
			// verify here.
			key, value, _, _, err := decodeEntry(raw[entriesBase+cursor:entriesBase+tail], fk, nonce, uint32(cursor))
			if err != nil {
				t.Fatalf("page %d: durable [head,tail)=[%d,%d) does not decode at offset %d: %v — "+
					"the durable header leads its entries", pg, head, tail, cursor, err)
			}
			cursor += entrySpanExact(len(key), len(value))
		}
	}
}

// TestWatermarkFlushIsStrictlyAfterBoundsFlush is the torn-final-msync regression
// guard for SetPBFrontier.
//
// WHY THIS SHAPE. An msync returns only once its pages are durable, so a crash can
// never revert bytes a COMPLETED msync wrote. The only torn writeback that can make
// the PB frontier LEAD the recoverable entries is therefore one where a SINGLE msync
// was flushing the frontier (in page 0) AND a page's projected bound (a different OS
// page) together, and the crash tore those two OS pages apart — persisting the
// frontier while the bound reverted. The fix removes that by flushing the projected
// bounds and the frontier in SEPARATE msync calls, frontier strictly last. This test
// asserts that separation directly through the msync seam: the flush that first
// makes the NEW frontier durable is strictly after the flush that first made the NEW
// bound durable. On the pre-fix arrangement (bounds + frontier in one full-region
// msync) both become durable in the SAME call and this fails. A byte-level replay
// cannot distinguish the two, since a crash tears only WITHIN one msync — hence the
// structural check.
func TestWatermarkFlushIsStrictlyAfterBoundsFlush(t *testing.T) {
	dir := t.TempDir()
	cfg := orderedBoundsCfg(dir)
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	for i := 0; i < 8; i++ {
		if err := c.Put([]byte(fmt.Sprintf("k%03d", i)), bytes.Repeat([]byte("v"), 256), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	s := c.shards[0]

	// A page carrying writes, and the runtime tail SetPBFrontier will project for it.
	targetPage, wantTail := -1, 0
	s.mu.RLock()
	for i, p := range s.pages {
		if p.tail() > 0 {
			targetPage, wantTail = i, p.tail()
			break
		}
	}
	s.mu.RUnlock()
	if targetPage < 0 {
		t.Fatal("no page received the writes")
	}

	const newSeq, newEpoch uint64 = 99, 7
	// tail is the second uint32 of the target page's 8-byte durable header.
	boundOff := headerSize + targetPage*cfg.PageSize + 4

	type flush struct{ frontierNew, boundNew bool }
	var flushes []flush
	prev := msyncTestHook
	msyncTestHook = func(f *os.File, region []byte) error {
		fr := false
		if len(region) >= hdrPBSeqOff+8 {
			fr = binary.LittleEndian.Uint64(region[hdrPBSeqOff:hdrPBSeqOff+8]) == newSeq
		}
		bn := false
		if len(region) >= boundOff+4 {
			bn = int(binary.LittleEndian.Uint32(region[boundOff:boundOff+4])) == wantTail
		}
		flushes = append(flushes, flush{frontierNew: fr, boundNew: bn})
		return msync(f, region) // preserve real durability
	}
	c.SetPBFrontier(newSeq, newEpoch)
	msyncTestHook = prev

	firstFrontier, firstBound := -1, -1
	for i, fl := range flushes {
		if firstFrontier < 0 && fl.frontierNew {
			firstFrontier = i
		}
		if firstBound < 0 && fl.boundNew {
			firstBound = i
		}
	}
	if firstBound < 0 {
		t.Fatalf("no flush carried the projected bound (want tail=%d at page %d); flushes=%+v", wantTail, targetPage, flushes)
	}
	if firstFrontier < 0 {
		t.Fatalf("no flush carried the new frontier seq=%d; flushes=%+v", newSeq, flushes)
	}
	if firstBound >= firstFrontier {
		t.Fatalf("PB frontier became durable at flush %d, not strictly after the projected bound (flush %d): "+
			"a crash tearing that combined flush could persist the watermark while a bound reverts, leaving the "+
			"frontier leading the recoverable entries; flushes=%+v", firstFrontier, firstBound, flushes)
	}
}

// TestFutureVersionFileIsRefusedNotRotated guards the H1 sentinel: a pages file
// written by a NEWER build must make the open FAIL, not be rotated aside and
// replaced by an empty shard — rotating a future format is silent data loss on a
// downgrade. Every other validation failure stays rotatable.
func TestFutureVersionFileIsRefusedNotRotated(t *testing.T) {
	dir := t.TempDir()
	cfg := orderedBoundsCfg(dir)
	sd := filepath.Join(dir, "s0")
	if err := os.MkdirAll(sd, 0o750); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	pagesPath := filepath.Join(sd, "pages.dat")

	maxPages := cfg.MaxPagesPerShard()
	size := headerSize + maxPages*cfg.PageSize
	region := make([]byte, size)
	writeHeader(region, uint32(cfg.PageSize), uint32(maxPages), 0)
	// Bump the on-disk version PAST what this build writes, and refresh the header
	// CRC so it is a well-formed header that fails ONLY the version gate.
	binary.LittleEndian.PutUint32(region[8:12], cacheVersion+1)
	binary.LittleEndian.PutUint32(region[28:32], crc32.ChecksumIEEE(region[0:28]))
	if err := os.WriteFile(pagesPath, region, 0o600); err != nil {
		t.Fatalf("write future-version pages.dat: %v", err)
	}

	_, err := newShard(cfg, sd, nil)
	if err == nil {
		t.Fatal("newShard opened a future-version pages file; want a refusal, not a silent rotate")
	}
	if !errors.Is(err, errFutureVersion) {
		t.Fatalf("newShard error = %v, want errFutureVersion", err)
	}

	// The file must be untouched — still the future-version bytes, not rotated.
	got, rerr := os.ReadFile(pagesPath)
	if rerr != nil {
		t.Fatalf("future-version pages.dat was removed/rotated: %v", rerr)
	}
	if v := binary.LittleEndian.Uint32(got[8:12]); v != cacheVersion+1 {
		t.Fatalf("pages.dat version = %d, want it preserved at %d", v, cacheVersion+1)
	}
	dirEntries, err := os.ReadDir(sd)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	for _, e := range dirEntries {
		if strings.Contains(e.Name(), ".bad-") {
			t.Fatalf("future-version file was rotated aside to %q; a downgrade must fail loudly instead", e.Name())
		}
	}
}
