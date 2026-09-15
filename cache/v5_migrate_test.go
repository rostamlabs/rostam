// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"
)

// v4 fixture writers. These lay down a pages file in the RETIRED v4 on-disk format
// (64-byte header, 8-byte per-page header, 26-byte entry header with a trailing
// CRC32) so the v5 build's migrate-on-open path can be exercised against real v4
// bytes rather than a mock. They are deliberately independent of the production
// encoders, which now only speak v5.

type v4Entry struct {
	key, val []byte
	seq      uint64
	tomb     bool
}

func writeV4Header(region []byte, pageSize, numPages uint32, appliedIdx uint64) {
	binary.LittleEndian.PutUint64(region[0:8], cacheMagic)
	binary.LittleEndian.PutUint32(region[8:12], 4) // version 4
	binary.LittleEndian.PutUint32(region[12:16], pageSize)
	binary.LittleEndian.PutUint32(region[16:20], numPages)
	binary.LittleEndian.PutUint64(region[20:28], appliedIdx)
	binary.LittleEndian.PutUint32(region[28:32], crc32.ChecksumIEEE(region[0:28]))
	// v4 also carried the logical clock (32..43) and PB frontier (44..63) with their
	// own CRCs at the SAME offsets v5 uses, so the production setters apply here.
	setAppliedStamp(region, 0)
	setPBFrontier(region, 0, 0)
}

// encodeV4Entry writes a v4 CRC-framed entry (the codec cache/migrate.go's decoder
// reads). Returns the total byte length.
func encodeV4Entry(dst, key, val []byte, expiryMs, meta uint64) int {
	binary.LittleEndian.PutUint16(dst[0:2], uint16(len(key)))
	binary.LittleEndian.PutUint32(dst[2:6], uint32(len(val)))
	binary.LittleEndian.PutUint64(dst[6:14], expiryMs)
	binary.LittleEndian.PutUint64(dst[v4EntryMetaOff:v4EntryCRCOff], meta)
	copy(dst[v4EntryHeaderSize:v4EntryHeaderSize+len(key)], key)
	copy(dst[v4EntryHeaderSize+len(key):], val)
	total := v4EntryHeaderSize + len(key) + len(val)
	crc := crc32.Checksum(dst[0:v4EntryCRCOff], crc32.IEEETable)
	crc = crc32.Update(crc, crc32.IEEETable, dst[v4EntryHeaderSize:total])
	binary.LittleEndian.PutUint32(dst[v4EntryCRCOff:v4EntryHeaderSize], crc)
	return total
}

// writeV4File materializes a v4 pages.dat with the given entries laid into page 0.
func writeV4File(t *testing.T, path string, pageSize, numPages int, entries []v4Entry) {
	t.Helper()
	buf := make([]byte, v4HeaderSize+numPages*pageSize)
	writeV4Header(buf, uint32(pageSize), uint32(numPages), 0)
	base := v4HeaderSize + v4PageHdrSize
	cursor := 0
	for _, e := range entries {
		n := encodeV4Entry(buf[base+cursor:], e.key, e.val, 0, makeMeta(e.seq, e.tomb))
		cursor += n
	}
	binary.LittleEndian.PutUint32(buf[v4HeaderSize:v4HeaderSize+4], 0)                // page 0 head
	binary.LittleEndian.PutUint32(buf[v4HeaderSize+4:v4HeaderSize+8], uint32(cursor)) // page 0 tail
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write v4 file: %v", err)
	}
}

func fileVersion(t *testing.T, path string) uint32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return binary.LittleEndian.Uint32(raw[8:12])
}

// TestMigrateV4ToV5OnOpen proves the whole upgrade path: a v4 file with a superseded
// key and a delete tombstone opens under the v5 build, every LIVE key is recovered
// with its winning value, deleted keys stay deleted, the on-disk file is rewritten as
// v5, and a SECOND open reads it as a native v5 file. A build that could not read v4
// would have rotated the file aside and lost everything.
func TestMigrateV4ToV5OnOpen(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // 1 page
	cfg.TTLSweepIntervalMs = 0
	cfg.DisableColdCompaction = true

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeV4File(t, path, cfg.PageSize, 1, []v4Entry{
		{key: []byte("k0"), val: []byte("AAAA"), seq: 1},
		{key: []byte("k1"), val: []byte("old"), seq: 2},
		{key: []byte("k1"), val: []byte("new"), seq: 5}, // supersedes seq 2
		{key: []byte("gone"), val: []byte("x"), seq: 3},
		{key: []byte("gone"), val: nil, seq: 6, tomb: true}, // deletes "gone"
		{key: []byte("k2"), val: []byte("CCCC"), seq: 4},
	})

	if v := fileVersion(t, path); v != 4 {
		t.Fatalf("fixture version = %d, want 4", v)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard (migrate): %v", err)
	}
	check := func(s *shard) {
		if v, err := s.Get([]byte("k0")); err != nil || !bytes.Equal(v, []byte("AAAA")) {
			t.Errorf("Get(k0) = %q,%v; want AAAA", v, err)
		}
		if v, err := s.Get([]byte("k1")); err != nil || !bytes.Equal(v, []byte("new")) {
			t.Errorf("Get(k1) = %q,%v; want new (the higher-seq copy won)", v, err)
		}
		if v, err := s.Get([]byte("k2")); err != nil || !bytes.Equal(v, []byte("CCCC")) {
			t.Errorf("Get(k2) = %q,%v; want CCCC", v, err)
		}
		if _, err := s.Get([]byte("gone")); err != ErrNotFound {
			t.Errorf("Get(gone) = %v; want ErrNotFound (a v4 tombstone must stay deleted)", err)
		}
	}
	check(s)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if v := fileVersion(t, path); v != cacheVersion {
		t.Fatalf("file version after migration = %d, want %d", v, cacheVersion)
	}

	// Second open: now a native v5 file, no migration, same contents.
	s2, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard (reopen v5): %v", err)
	}
	defer func() { _ = s2.Close() }()
	check(s2)
}

// TestMigrateV4FailClosed: when the v5 rewrite cannot be staged, the open FAILS and
// the intact v4 file is left in place — never rotated aside, because rotating
// committed state is data loss. Staging is blocked by making the shard directory
// read-only, so CREATING the staging temp file fails while the existing pages.dat can
// still be opened and mapped.
func TestMigrateV4FailClosed(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeV4File(t, path, cfg.PageSize, 1, []v4Entry{{key: []byte("k0"), val: []byte("AAAA"), seq: 1}})

	// Make the shard directory read-only so CREATING the staging temp file fails
	// (EACCES) while the existing pages.dat can still be opened and mapped — a clean
	// simulation of "cannot stage the rewrite" (e.g. a full disk). Restored on cleanup
	// so the temp dir can be removed.
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o755) })

	if _, err := newShard(cfg, dir, nil); err == nil {
		t.Fatal("newShard succeeded despite an unwritable staging dir; migration must fail closed")
	}
	_ = os.Chmod(dir, 0o755) // restore so the assertions below can read the file
	// The v4 file must be untouched (still v4, no .bad sibling rotated aside).
	if v := fileVersion(t, path); v != 4 {
		t.Errorf("v4 file version = %d after a failed migration, want 4 (must be left intact)", v)
	}
	if bad, _ := filepath.Glob(path + ".bad-*"); len(bad) != 0 {
		t.Errorf("failed migration rotated the v4 file aside (%v); it must be left in place", bad)
	}
}

// TestDowngradeRefusesFutureVersion: a file whose header version is NEWER than this
// build writes (the shape an older build sees when handed a v5 file) is refused
// non-destructively with errFutureVersion, not rotated aside. This is the downgrade
// guard: a v5 file must never be silently discarded by a build that predates v5.
func TestDowngradeRefusesFutureVersion(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")

	// Write a real v5 file, then stamp a FUTURE version into its header (recomputing
	// the header CRC so only the version gate can reject it).
	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(raw[8:12], cacheVersion+1)
	binary.LittleEndian.PutUint32(raw[28:32], crc32.ChecksumIEEE(raw[0:28]))
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	_, err = newShard(cfg, dir, nil)
	if err == nil {
		t.Fatal("newShard opened a future-version file; it must refuse")
	}
	// Non-destructive refusal: no .bad rotation, file unchanged.
	if bad, _ := filepath.Glob(path + ".bad-*"); len(bad) != 0 {
		t.Errorf("future-version file was rotated aside (%v); refusal must be non-destructive", bad)
	}
	if v := fileVersion(t, path); v != cacheVersion+1 {
		t.Errorf("future-version file version = %d, want it left at %d", v, cacheVersion+1)
	}
}
