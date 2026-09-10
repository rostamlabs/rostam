// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"
)

// writeVersion1BigEndianFile materializes a pages.dat in the released
// version-1 layout: a little-endian 64-byte header declaring version 1, and one
// ring-buffer entry in page 0 whose keyLen/valLen/expiry/CRC are big-endian
// (the pre-flip codec). The per-page head/tail offsets were always
// little-endian, so this file passes every header check except the version gate.
func writeVersion1BigEndianFile(t *testing.T, path string, pageSize, numPages int) {
	t.Helper()
	total := headerSize + numPages*pageSize
	buf := make([]byte, total)

	binary.LittleEndian.PutUint64(buf[0:8], cacheMagic)
	binary.LittleEndian.PutUint32(buf[8:12], 1) // version 1
	binary.LittleEndian.PutUint32(buf[12:16], uint32(pageSize))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(numPages))
	binary.LittleEndian.PutUint64(buf[20:28], 0) // appliedIndex
	binary.LittleEndian.PutUint32(buf[28:32], crc32.ChecksumIEEE(buf[0:28]))

	// One big-endian entry ("k" -> "v") at the start of page 0's entry region.
	key, val := []byte("k"), []byte("v")
	e := headerSize + pageHdrSize
	binary.BigEndian.PutUint16(buf[e:e+2], uint16(len(key)))
	binary.BigEndian.PutUint32(buf[e+2:e+6], uint32(len(val)))
	binary.BigEndian.PutUint64(buf[e+6:e+14], 0)
	copy(buf[e+entryHeaderSize:], key)
	copy(buf[e+entryHeaderSize+len(key):], val)
	crc := crc32.Checksum(buf[e:e+14], crcTable)
	crc = crc32.Update(crc, crcTable, buf[e+entryHeaderSize:e+entryHeaderSize+len(key)+len(val)])
	binary.BigEndian.PutUint32(buf[e+14:e+18], crc)

	// Page 0 head/tail (little-endian, unchanged across the codec flip).
	entrySz := entryHeaderSize + len(key) + len(val)
	binary.LittleEndian.PutUint32(buf[headerSize:headerSize+4], 0)
	binary.LittleEndian.PutUint32(buf[headerSize+4:headerSize+8], uint32(entrySz))

	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// writeCurrentFileWithEntries builds a pages.dat in the CURRENT on-disk format
// with the given entries laid into page 0 using the real encoder, and returns
// each entry's start offset within the page's entry region. Entries are stamped
// with ascending write sequences, matching what a live shard would have written.
// Callers may then corrupt a byte to simulate a torn write before reopening.
func writeCurrentFileWithEntries(t *testing.T, path string, pageSize, numPages int, entries [][2][]byte) []int {
	t.Helper()
	total := headerSize + numPages*pageSize
	buf := make([]byte, total)
	writeHeader(buf, uint32(pageSize), uint32(numPages), 0)

	base := headerSize + pageHdrSize
	cursor := 0
	offsets := make([]int, 0, len(entries))
	for i, kv := range entries {
		offsets = append(offsets, cursor)
		n, err := encodeEntry(buf[base+cursor:], kv[0], kv[1], 0, makeMeta(uint64(i+1), false))
		if err != nil {
			t.Fatalf("encode entry %d: %v", i, err)
		}
		cursor += n
	}
	binary.LittleEndian.PutUint32(buf[headerSize:headerSize+4], 0)
	binary.LittleEndian.PutUint32(buf[headerSize+4:headerSize+8], uint32(cursor))

	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return offsets
}

// TestRecoverRejectsVersion1File is the cross-version guard for the BE→LE entry
// codec flip: a durable file written by the released version-1 (big-endian)
// build must be rejected loudly at recovery, not decoded with the little-endian
// reader (which byte-swaps keyLen/valLen and silently drops every persisted
// key). With cacheVersion bumped to 2, validateHeader rejects it and newShard
// rotates the stale file aside, so Raft replay can rebuild state from index 0.
func TestRecoverRejectsVersion1File(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // 1 page

	dir := t.TempDir()
	writeVersion1BigEndianFile(t, filepath.Join(dir, "pages.dat"), cfg.PageSize, 1)

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Loud rejection: the version-1 file was rotated to a .bad-* sibling.
	bad, _ := filepath.Glob(filepath.Join(dir, "pages.dat.bad-*"))
	if len(bad) == 0 {
		t.Fatal("version-1 file was not rejected/rotated; version gate did not fire")
	}
	// The shard came up fresh from index 0 (repairable via replay), not with a
	// silently-empty index carrying the old applied-index watermark.
	if got := s.appliedIndex.Load(); got != 0 {
		t.Errorf("appliedIndex = %d after rejecting stale file, want 0", got)
	}
	if _, err := s.Get([]byte("k")); err != ErrNotFound {
		t.Errorf("Get on fresh shard = %v, want ErrNotFound", err)
	}
}

// TestRebuildCountsCorruptEntries covers the observability gap: a torn entry in
// a recovered page must bump CorruptionErrors and be logged, not swallowed by a
// bare break. Live entries indexed before the corruption stay reachable.
func TestRebuildCountsCorruptEntries(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // 1 page
	cfg.TTLSweepIntervalMs = 0      // no sweeper interference

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	offs := writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
		{[]byte("k1"), []byte("BBBB")},
		{[]byte("k2"), []byte("CCCC")},
	})

	// Corrupt a payload byte of entry 1 so its CRC fails during rebuild.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	corruptAt := int64(headerSize + pageHdrSize + offs[1] + entryHeaderSize)
	if _, err := f.WriteAt([]byte{0xFF}, corruptAt); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	if got := s.snapshot().CorruptionErrors; got == 0 {
		t.Error("CorruptionErrors = 0 after recovering a torn page; corruption was swallowed silently")
	}
	// Entry before the corruption is still reachable.
	if v, err := s.Get([]byte("k0")); err != nil || !bytes.Equal(v, []byte("AAAA")) {
		t.Errorf("Get(k0) = %q,%v; want AAAA,nil", v, err)
	}
	// The corrupt entry and everything after it in the page were dropped.
	if _, err := s.Get([]byte("k1")); err != ErrNotFound {
		t.Errorf("Get(k1) = %v, want ErrNotFound (corrupt)", err)
	}
	if _, err := s.Get([]byte("k2")); err != ErrNotFound {
		t.Errorf("Get(k2) = %v, want ErrNotFound (after corruption)", err)
	}
}

// TestRebuildRejectsCorruptPageTail covers the crash-loop hazard: an on-disk
// page tail that has been corrupted (bit-rot, or a crash mid-setTail) past the
// page's own capacity must not be trusted straight into entries[cursor:tail] —
// that slice expression panics before decodeEntry's CRC ever gets a chance to
// reject it, producing a permanent, unrecoverable crash loop. rebuildIndexFromPages
// must instead validate 0 <= head <= tail <= len(entries) and treat a violation
// like any other torn/corrupt page: reset it and bump the corruption counter.
func TestRebuildRejectsCorruptPageTail(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // 1 page
	cfg.TTLSweepIntervalMs = 0      // no sweeper interference

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
	})

	// Corrupt page 0's on-disk tail field to a huge garbage value, as a bit-flip
	// or a crash mid-setTail could produce.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	tailAt := int64(headerSize + 4)
	if _, err := f.WriteAt([]byte{0xF0, 0xFF, 0xFF, 0xFF}, tailAt); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v, want a clean error/reset instead of a panic", err)
	}
	defer func() { _ = s.Close() }()

	if got := s.snapshot().CorruptionErrors; got == 0 {
		t.Error("CorruptionErrors = 0 after recovering a page with a corrupt tail; corruption was swallowed silently")
	}
	if _, err := s.Get([]byte("k0")); err != ErrNotFound {
		t.Errorf("Get(k0) = %v, want ErrNotFound (page reset after corrupt tail)", err)
	}
}

// TestRebuildRejectsCorruptEqualHeadTail covers the case the equality shortcut
// missed: a page whose on-disk head AND tail are BOTH corrupted to the same
// out-of-range value (not just an out-of-range tail alone) still satisfies
// head==tail. Checking bounds only after that shortcut let such a page slip
// through untouched at recovery: no corruption counted, no reset, and
// FreeTail() = len(entries) - tail went negative, wedging every future Put
// against this page behind errPageFull forever. rebuildIndexFromPages must
// validate 0 <= head <= tail <= len(entries) BEFORE the head==tail shortcut,
// not after.
func TestRebuildRejectsCorruptEqualHeadTail(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // 1 page
	cfg.TTLSweepIntervalMs = 0      // no sweeper interference

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
	})

	// Corrupt page 0's on-disk head AND tail to the SAME huge garbage value, as
	// a bit-flip or a crash mid-setTail could produce. head lives at
	// region[0:4], tail at region[4:8] (see newMmapPage); page 0's region
	// starts right after the file header.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	headAt := int64(headerSize)
	tailAt := int64(headerSize + 4)
	garbage := []byte{0xF0, 0xFF, 0xFF, 0xFF}
	if _, err := f.WriteAt(garbage, headAt); err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt(garbage, tailAt); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v, want a clean error/reset instead of a panic", err)
	}
	defer func() { _ = s.Close() }()

	if got := s.snapshot().CorruptionErrors; got == 0 {
		t.Error("CorruptionErrors = 0 after recovering a page with corrupt equal head/tail; corruption was swallowed silently")
	}
	if free := s.pages[0].FreeTail(); free < 0 {
		t.Errorf("pages[0].FreeTail() = %d, want >= 0 (page must be reset, not left with an out-of-range tail)", free)
	}
	if err := s.Put([]byte("k1"), []byte("BBBB"), 0); err != nil {
		t.Errorf("Put after corrupt-equal-head/tail recovery: %v, want success (page must not be permanently wedged behind errPageFull)", err)
	}
	if _, err := s.Get([]byte("k0")); err != ErrNotFound {
		t.Errorf("Get(k0) = %v, want ErrNotFound (page reset after corrupt head/tail)", err)
	}
}

// TestTornEntrySurvivesEviction reproduces the wedged-shard scenario: a page
// recovered with a torn entry inside [head, tail) must not fail every
// eviction-triggering Put forever. rebuildIndexFromPages truncates the page to
// its validated prefix and EvictFront validates entry bounds, so the shard stays
// writable and the 0 <= head <= tail invariant holds.
func TestTornEntrySurvivesEviction(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // single page → forces ring eviction
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.TTLSweepIntervalMs = 0

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	offs := writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("a"), []byte("AAAA")},
		{[]byte("b"), []byte("BBBB")},
		{[]byte("c"), []byte("CCCC")},
	})

	// Tear entry 1's header (garbage valLen) inside the persisted [head, tail).
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	valLenAt := int64(headerSize + pageHdrSize + offs[1] + 2)
	if _, err := f.WriteAt([]byte{0xF0, 0xFF, 0xFF, 0xFF}, valLenAt); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Drive the shard hard enough to fill and evict the page many times over.
	val := bytes.Repeat([]byte("X"), 4096)
	for i := range 1000 {
		if err := s.Put(fmt.Appendf(nil, "k%08d", i), val, 0); err != nil {
			t.Fatalf("Put #%d failed (%v); torn entry wedged the shard", i, err)
		}
	}
	// Invariant preserved on the single page.
	s.mu.RLock()
	head, tail := s.pages[0].head(), s.pages[0].tail()
	s.mu.RUnlock()
	if head > tail {
		t.Errorf("invariant broken: head=%d > tail=%d", head, tail)
	}
}

// TestMmapRejectPolicyScansAllPages covers the mmap capacity bug: under
// PolicyRejectWrites an mmap shard preallocates every page but only checked the
// single writeIdx page, so it rejected after filling ~1/MaxPagesPerShard of its
// memory. The all-pages scan must let writes fill every page first.
func TestMmapRejectPolicyScansAllPages(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 2 << 20 // 2 pages, both preallocated in mmap mode
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.TTLSweepIntervalMs = 0

	s, err := newShard(cfg, t.TempDir(), nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	val := bytes.Repeat([]byte("X"), 4096) // ≈ 254 entries per page
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
		t.Fatal("expected ErrFull once both pages fill")
	}
	// Two pages hold ~508 entries; the single-page bug rejected after ~254.
	if wrote < 400 {
		t.Fatalf("only wrote %d entries; second mmap page was never used", wrote)
	}
}
