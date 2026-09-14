// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func newTestPage(t *testing.T, size int) *page {
	t.Helper()
	p := newHeapPage(size)
	if len(p.entries()) != size {
		t.Fatalf("page entries len = %d, want %d", len(p.entries()), size)
	}
	return p
}

func TestPageWriteThenRead(t *testing.T) {
	p := newTestPage(t, 1<<20)
	key := []byte("k1")
	val := []byte("v1")

	off, _, err := p.Write(key, val, 0, 0)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	gotKey, gotVal, exp, err := p.Read(off)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(gotKey, key) || !bytes.Equal(gotVal, val) {
		t.Fatalf("roundtrip mismatch: %q=%q expiry=%d", gotKey, gotVal, exp)
	}
}

func TestPageWriteReturnsErrPageFullWhenNoRoom(t *testing.T) {
	// page just big enough for one small entry.
	p := newTestPage(t, entryHeaderSize+8)
	if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
		t.Fatalf("first write should succeed: %v", err)
	}
	if _, _, err := p.Write([]byte("k2"), []byte("vv"), 0, 0); err != errPageFull {
		t.Fatalf("second write should return errPageFull, got %v", err)
	}
}

func TestPageEvictReclaimsSpaceFromHead(t *testing.T) {
	// Build a page where two entries fit, evict the first, and write a third.
	keyA, valA := []byte("a"), []byte("AAAA")
	keyB, valB := []byte("b"), []byte("BBBB")
	// Size for exactly 2 entries initially, then both evicted to make room
	size := 2 * (entryHeaderSize + len(keyA) + len(valA))
	p := newTestPage(t, size)

	offA, szA, err := p.Write(keyA, valA, 0, 0)
	if err != nil {
		t.Fatalf("write A: %v", err)
	}
	if _, _, err := p.Write(keyB, valB, 0, 0); err != nil {
		t.Fatalf("write B: %v", err)
	}
	// page is full; until eviction, no writes should succeed.
	if _, _, err := p.Write([]byte("c"), []byte("CCCC"), 0, 0); err != errPageFull {
		t.Fatalf("write C without evict: got %v, want errPageFull", err)
	}
	// Evict the front entry (entry A).
	evictedKey, evictedSize, err := p.EvictFront()
	if err != nil {
		t.Fatalf("EvictFront: %v", err)
	}
	if !bytes.Equal(evictedKey, keyA) {
		t.Errorf("evicted key = %q, want %q", evictedKey, keyA)
	}
	if evictedSize != szA || evictedSize == 0 {
		t.Errorf("evicted size = %d, want %d", evictedSize, szA)
	}
	if offA != 0 {
		t.Errorf("first entry expected at offset 0, got %d", offA)
	}
	// page still full (B occupies the tail). Evict B.
	if _, _, err := p.EvictFront(); err != nil {
		t.Fatalf("EvictFront (B): %v", err)
	}
	// Now page is empty and reset; write C should succeed.
	if _, _, err := p.Write([]byte("c"), []byte("CCCC"), 0, 0); err != nil {
		t.Fatalf("write C after evict: %v", err)
	}
}

func TestEvictFrontRejectsTornEntry(t *testing.T) {
	// A crash-torn or stale entry header can carry a garbage valLen. EvictFront
	// must validate the whole entry lies within [head, tail) before advancing
	// head — otherwise it walks head past tail, breaking the 0 <= head <= tail
	// invariant and wedging the page.
	p := newTestPage(t, 1<<20)
	if _, _, err := p.Write([]byte("k"), []byte("vvvv"), 0, 0); err != nil {
		t.Fatalf("Write: %v", err)
	}
	headBefore, tailBefore := p.head(), p.tail()

	// Corrupt valLen in place to a value that runs past tail.
	binary.LittleEndian.PutUint32(p.entries()[2:6], 0xFFFFFFF0)

	_, _, err := p.EvictFront()
	if err != errEntryTruncated {
		t.Fatalf("EvictFront on torn entry = %v, want errEntryTruncated", err)
	}
	if p.head() != headBefore {
		t.Errorf("head advanced to %d on torn entry (was %d); invariant broken", p.head(), headBefore)
	}
	if p.head() > p.tail() {
		t.Errorf("head=%d > tail=%d after torn EvictFront", p.head(), p.tail())
	}
	_ = tailBefore
}

func TestPageMmapWriteRead(t *testing.T) {
	region := make([]byte, 4096)
	p := newMmapPage(region)
	off, _, err := p.Write([]byte("k"), []byte("v"), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	key, val, _, err := p.Read(off)
	if err != nil {
		t.Fatal(err)
	}
	if string(key) != "k" || string(val) != "v" {
		t.Errorf("got %q=%q, want k=v", key, val)
	}
}

func TestPageMmapBoundsSurviveOnlyViaProjection(t *testing.T) {
	region := make([]byte, 4096)
	p1 := newMmapPage(region)
	off, sz, _ := p1.Write([]byte("k"), []byte("v"), 0, 0)

	// A bare re-wrap on the same region does NOT inherit the runtime bounds: after
	// the torn-writeback fix, an append advances only the runtime head/tail and never
	// the mapped header, so a fresh page object starts at (0,0). The entry BYTES are
	// in the region and still decode — it is the bounds that need a projection.
	p2 := newMmapPage(region)
	key, val, _, err := p2.Read(off)
	if err != nil {
		t.Fatal(err)
	}
	if string(key) != "k" || string(val) != "v" {
		t.Errorf("entry bytes did not survive re-wrap: got %q=%q, want k=v", key, val)
	}
	if p2.head() != 0 || p2.tail() != 0 {
		t.Errorf("bare re-wrap head=%d tail=%d, want (0,0): bounds must not be inferred without projection", p2.head(), p2.tail())
	}

	// Publish the bounds into the durable header, then a fresh wrap seeded from it
	// (exactly as attachMmapRegion seeds at open) recovers them.
	p1.projectBounds(p1.head(), p1.tail())
	p3 := newMmapPage(region)
	h, tl := p3.durableBounds()
	p3.heapHead, p3.heapTail = h, tl
	if p3.head() != 0 || p3.tail() != int(sz) {
		t.Errorf("seeded-from-durable head=%d tail=%d, want 0 and %d", p3.head(), p3.tail(), sz)
	}
}

func TestPageMmapWriteLeavesDurableHeaderToProjection(t *testing.T) {
	region := make([]byte, 4096)
	p := newMmapPage(region)
	if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
		t.Fatal(err)
	}
	// The append advanced the RUNTIME tail but must NOT have touched the durable
	// header (bytes 0..7): the header is a projection, written only by projectBounds.
	if p.tail() == 0 {
		t.Fatal("Write did not advance the runtime tail")
	}
	for i := 0; i < pageHdrSize; i++ {
		if region[i] != 0 {
			t.Fatalf("Write wrote durable header byte %d = %d; the header must be written only by projectBounds", i, region[i])
		}
	}
	// projectBounds is the sole writer of the header, and durableBounds reads it back.
	p.projectBounds(p.head(), p.tail())
	if dh, dt := p.durableBounds(); dh != p.head() || dt != p.tail() {
		t.Fatalf("durableBounds = (%d,%d), want (%d,%d) after projectBounds", dh, dt, p.head(), p.tail())
	}
	if region[4] == 0 && region[5] == 0 && region[6] == 0 && region[7] == 0 {
		t.Error("projectBounds did not write the tail into the mmap region")
	}
}

func TestPageHeapBackingUnchanged(t *testing.T) {
	// Heap-mode behavior must match heap-only mode exactly.
	p := newHeapPage(4096)
	off, _, _ := p.Write([]byte("k"), []byte("v"), 0, 0)
	key, val, _, _ := p.Read(off)
	if string(key) != "k" || string(val) != "v" {
		t.Errorf("heap mode roundtrip broken: %q=%q", key, val)
	}
}
