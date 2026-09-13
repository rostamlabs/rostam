// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"testing"
	"time"
)

// Relocating eviction on a SINGLE-NODE MMAP ringbuf shard. The pass is the same one
// the heap tests cover; these pin what is different about the in-place drain, and the
// recovery ordering that only exists once entries are on a file.
//
// NONE OF THESE DEPEND ON THE READER FIX. needsReadLockForGet() is true for mmap
// ringbuf, so reads hold the shard read lock while eviction holds the write lock and
// no reader can ever observe a half-done relocation: indexTable.rechaseSlot is
// unreachable from this path. Revert it and this file still passes — that is checked,
// not assumed.
//
// They also avoid arithmetic about WHICH page holds what. An mmap shard maps every
// page at construction and starts writing at the last one, so the fill order is not
// the heap's; instead each test drives the shard to its first eviction and only then
// seeds the records it cares about, which lands them on the frontier — the one page
// guaranteed to be evacuated before it is drained.

// relocMmapConfig is a single-shard mmap ringbuf cache over dir, with the sweeper off
// and cold compaction disabled so a reopen rebuilds from the bytes as they were left
// rather than from a rewritten file.
func relocMmapConfig(dir string, pages int, on bool) Config {
	cfg := relocConfig(pages, on)
	cfg.DataDir = dir
	cfg.DisableColdCompaction = true
	return cfg
}

// churnUntilFirstEviction fills an mmap shard to capacity through the hot key, so the
// records a test seeds afterwards land on the frontier. The first eviction of a shard
// drains its victim with nothing having evacuated it, which is the one page a record
// that must survive may not be seeded into.
func churnUntilFirstEviction(t *testing.T, c *Cache, hot []byte) {
	t.Helper()
	for n := 0; c.Stats().Evictions == 0; n++ {
		if n > 10_000 {
			t.Fatal("shard never reached its first eviction")
		}
		mustPut(t, c, hot, relocValue(n%256))
	}
}

// TestRelocatingEvictionMmapKeepsLiveRecordsOnADeadPage is the mmap counterpart of the
// heap mixed-page test: cold records sharing pages with a key rewritten constantly.
// The in-place drain takes the whole page — dead versions and live records together —
// unless relocation has moved the live ones out first.
func TestRelocatingEvictionMmapKeepsLiveRecordsOnADeadPage(t *testing.T) {
	const coldKeys = 2 // their bytes must fit one eviction's relocation budget
	run := func(on bool) (Stats, int) {
		c, err := New(relocMmapConfig(t.TempDir(), 3, on))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer func() { _ = c.Close() }()
		hot := relocKey(200)
		churnUntilFirstEviction(t, c, hot)
		for i := range coldKeys {
			mustPut(t, c, relocKey(100+i), relocValue(100+i))
		}
		for n := range 200 {
			mustPut(t, c, hot, relocValue(n%256))
		}
		survivors := 0
		for i := range coldKeys {
			if v, gerr := c.Get(relocKey(100 + i)); gerr == nil && relocUniform(v, 100+i) {
				survivors++
			}
		}
		return c.Stats(), survivors
	}
	offStats, offAlive := run(false)
	onStats, onAlive := run(true)

	if offAlive != 0 {
		t.Fatalf("flag off kept %d cold records; the in-place drain should have taken them all", offAlive)
	}
	if onAlive != coldKeys {
		t.Fatalf("flag on kept %d of %d cold records", onAlive, coldKeys)
	}
	if onStats.EvictionRelocations == 0 {
		t.Fatal("flag on relocated nothing")
	}
	if offStats.EvictionRelocations != 0 {
		t.Fatalf("flag off relocated %d records", offStats.EvictionRelocations)
	}
	if onStats.EvictionsLive >= offStats.EvictionsLive {
		t.Fatalf("EvictionsLive not reduced: off=%d on=%d", offStats.EvictionsLive, onStats.EvictionsLive)
	}
}

// TestRelocatingEvictionMmapPreservesTriple checks a move across the in-place drain
// carries the exact (key, value, exp) triple, expiry included: the relocated copy must
// keep the original absolute expiry rather than be refreshed by the copy.
func TestRelocatingEvictionMmapPreservesTriple(t *testing.T) {
	c, err := New(relocMmapConfig(t.TempDir(), 3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	hot := relocKey(200)
	churnUntilFirstEviction(t, c, hot)

	ttlKey := relocKey(100)
	ttlVal := relocValue(42)
	const ttl = 10 * time.Minute
	if err := c.Put(ttlKey, ttlVal, ttl); err != nil {
		t.Fatalf("Put ttl key: %v", err)
	}
	s := c.shards[0]
	_, wantExp, err := s.getWithExpiryH(ttlKey, hashKey(ttlKey))
	if err != nil {
		t.Fatalf("read back ttl key: %v", err)
	}
	if wantExp == 0 {
		t.Fatal("seeded key carries no expiry")
	}
	before := c.Stats().EvictionRelocations
	for n := range 200 {
		mustPut(t, c, hot, relocValue(n%256))
	}
	if c.Stats().EvictionRelocations == before {
		t.Fatal("nothing was relocated")
	}

	gotVal, gotExp, err := s.getWithExpiryH(ttlKey, hashKey(ttlKey))
	if err != nil {
		t.Fatalf("relocated ttl key: %v", err)
	}
	if !bytes.Equal(gotVal, ttlVal) {
		t.Fatal("relocated value differs from the original")
	}
	if gotExp != wantExp {
		t.Fatalf("relocated expiry = %d, want the original %d", gotExp, wantExp)
	}
}

// TestRelocatingEvictionMmapEntriesHeldWhileBytesFall is the shape of the win on mmap:
// across an eviction whose victim holds only dead versions, because the live record
// that was on it has already been carried forward, the index keeps every key it had
// while the page bytes behind them fall.
func TestRelocatingEvictionMmapEntriesHeldWhileBytesFall(t *testing.T) {
	c, err := New(relocMmapConfig(t.TempDir(), 3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	hot := relocKey(200)
	churnUntilFirstEviction(t, c, hot)
	cold := relocKey(100)
	mustPut(t, c, cold, relocValue(42))

	// The eviction that carries the cold record forward, then the one that drains the
	// page it came off.
	if _, after := putUntilEviction(t, c, hot, 0); after.EvictionRelocations == 0 {
		t.Fatal("the cold record was not carried forward")
	}
	beforeSt, afterSt := putUntilEviction(t, c, hot, 50)

	if afterSt.Entries != beforeSt.Entries {
		t.Fatalf("Entries %d -> %d; the eviction cost the index keys it should have kept",
			beforeSt.Entries, afterSt.Entries)
	}
	if afterSt.BytesUsed >= beforeSt.BytesUsed {
		t.Fatalf("BytesUsed %d -> %d; the dead page bytes were not reclaimed",
			beforeSt.BytesUsed, afterSt.BytesUsed)
	}
	if afterSt.EvictionsLive != beforeSt.EvictionsLive {
		t.Fatalf("EvictionsLive %d -> %d; a fully evacuated page lost live records",
			beforeSt.EvictionsLive, afterSt.EvictionsLive)
	}
	if v := mustGet(t, c, cold); !relocUniform(v, 42) {
		t.Fatal("cold record did not survive intact")
	}
}

// TestRelocatingEvictionMmapRecoveryResolvesRelocatedCopy is the recovery-ordering
// test. Relocation leaves the original framed on its source page until the NEXT
// eviction drains it, so between those two evictions the file holds TWO copies of one
// key. A restart in that window must resolve the key to the RELOCATED copy.
//
// Nothing about page order gives that: the relocated copy can sit at a lower page
// index than the original it replaced. What gives it is the write sequence —
// rebuildIndexFromPages runs every copy through a max-seq contest (indexedSeqAtLeast,
// "the sequence is the authority") — and relocation inherits a fresh, higher sequence
// by appending through the ordinary write path rather than copying the original's meta
// across. The assertions below are on the resolved copy's IDENTITY (page, offset,
// sequence), not just its value: both copies carry the same value, so a value check
// alone could not tell which one recovery picked.
func TestRelocatingEvictionMmapRecoveryResolvesRelocatedCopy(t *testing.T) {
	dir := t.TempDir()
	cold := relocKey(100)
	coldVal := relocValue(42)
	hot := relocKey(200)

	var origRef, relocRef slabRef
	var origSeq, relocSeq uint64
	func() {
		c, err := New(relocMmapConfig(dir, 3, true))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer func() { _ = c.Close() }()
		s := c.shards[0]
		h := hashKey(cold)

		churnUntilFirstEviction(t, c, hot)
		mustPut(t, c, cold, coldVal)
		_, ref0, ok := s.tab.Load().findSlot(h)
		if !ok {
			t.Fatal("seeded key is not in the index")
		}
		origRef = ref0

		// Stop at the exact Put that relocates the cold record: nothing else ever
		// repoints its slot, so a changed ref IS the relocation. Stopping here leaves
		// the original still framed — its page is drained one eviction later.
		for n := 0; ; n++ {
			if n > 10_000 {
				t.Fatal("the seeded record was never relocated")
			}
			mustPut(t, c, hot, relocValue(n%256))
			_, cur, ok := s.tab.Load().findSlot(h)
			if !ok {
				t.Fatal("seeded record was evicted instead of relocated")
			}
			if cur != ref0 {
				relocRef = cur
				break
			}
		}

		// Both copies are on the file, and the relocated one is strictly newer.
		k, v, _, derr := s.pages[origRef.pageIdx()].Read(origRef.offset())
		if derr != nil {
			t.Fatalf("original copy does not decode: %v", derr)
		}
		if !bytes.Equal(k, cold) || !bytes.Equal(v, coldVal) {
			t.Fatal("original copy is not the record under test; it was already overwritten")
		}
		meta, ok := s.pages[origRef.pageIdx()].MetaAt(origRef.offset())
		if !ok {
			t.Fatal("original copy has no readable meta")
		}
		origSeq = metaSeq(meta)
		meta, ok = s.pages[relocRef.pageIdx()].MetaAt(relocRef.offset())
		if !ok {
			t.Fatal("relocated copy has no readable meta")
		}
		relocSeq = metaSeq(meta)
		if relocSeq <= origSeq {
			t.Fatalf("relocated copy seq %d does not exceed the original's %d; recovery would resolve to the original",
				relocSeq, origSeq)
		}
		if metaIsTombstone(meta) {
			t.Fatal("relocated copy carries the tombstone flag")
		}
	}()

	// Reopen over the same bytes.
	c2, err := New(relocMmapConfig(dir, 3, true))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = c2.Close() }()
	s2 := c2.shards[0]

	v, gerr := c2.Get(cold)
	if gerr != nil {
		t.Fatalf("record did not survive the restart: %v", gerr)
	}
	if !relocUniform(v, 42) {
		t.Fatal("record came back with the wrong value")
	}
	_, ref, ok := s2.tab.Load().findSlot(hashKey(cold))
	if !ok {
		t.Fatal("record has no index slot after the restart")
	}
	if ref.pageIdx() != relocRef.pageIdx() || ref.offset() != relocRef.offset() {
		t.Fatalf("recovery resolved the key to page %d offset %d; want the RELOCATED copy at page %d offset %d (the original is at page %d offset %d)",
			ref.pageIdx(), ref.offset(), relocRef.pageIdx(), relocRef.offset(), origRef.pageIdx(), origRef.offset())
	}
	meta, ok := s2.pages[ref.pageIdx()].MetaAt(ref.offset())
	if !ok || metaSeq(meta) != relocSeq {
		t.Fatalf("resolved copy carries sequence %d, want the relocated copy's %d", metaSeq(meta), relocSeq)
	}
}

// BenchmarkRelocatingEvictionABMmap is the A/B harness over the mmap ringbuf path:
// same seed, same key population and same geometry as the heap rows, so the pairs are
// directly comparable. The absolute ns/op carries the cost of writing through a mapped
// file and is not comparable to the heap rows; the off-to-on delta within this pair is.
func BenchmarkRelocatingEvictionABMmap(b *testing.B) {
	relocABArms(b, b.TempDir())
}
