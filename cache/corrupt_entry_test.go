// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// What a single unreadable entry costs, on the two paths that meet one: the mmap
// in-place drain at runtime, and the warm-restart walk. See drainPageLocked and
// rebuildIndexFromPages for the reasoning these pin.

// corruptDrainShard is a single-page mmap ringbuf shard whose onRemove hook records
// every key it is handed. One page means the drain that evicts it is the only
// eviction, and the Put that follows writes back into the same page from offset 0.
func corruptDrainShard(t *testing.T) (*shard, func() map[string]int) {
	t.Helper()
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.TTLSweepIntervalMs = 0

	var mu sync.Mutex
	removed := map[string]int{}
	fn := func(key []byte) {
		mu.Lock()
		removed[string(key)]++
		mu.Unlock()
	}
	var hook atomic.Pointer[func([]byte)]
	hook.Store(&fn)

	s, err := newShard(cfg, t.TempDir(), &hook)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, func() map[string]int {
		mu.Lock()
		defer mu.Unlock()
		out := make(map[string]int, len(removed))
		for k, v := range removed {
			out[k] = v
		}
		return out
	}
}

// TestCorruptDrainLeavesNoSlotPointingIntoTheResetPage: when the drain meets an entry
// whose MAC does not verify mid-drain, v5 RESYNCS past it — evicting everything else
// on the page normally and dropping ONLY the damaged entry's slot, rather than
// resetting the whole page as v4 did. The damaged slot must still go (its generation
// is unchanged, so a slot left behind would pass the read path's generation gate and
// read whatever bytes later refill its offset — a second Iterate hit for an unrelated
// key). The torn record's onRemove must NOT fire (its key cannot be trusted), while
// every other displaced record fires exactly once, and only the ONE damaged record's
// bytes are counted as discarded.
func TestCorruptDrainLeavesNoSlotPointingIntoTheResetPage(t *testing.T) {
	s, removedKeys := corruptDrainShard(t)

	// Equal-size records, so the records written after the reset land on exactly the
	// offsets the drained ones occupied.
	val := bytes.Repeat([]byte("v"), 4096)
	span := entrySpanExact(len("k00000000"), len(val))
	perPage := (s.cfg.PageSize - pageHdrSize) / span
	const torn = 100
	for i := range perPage {
		if err := s.Put(fmt.Appendf(nil, "k%08d", i), val, 0); err != nil {
			t.Fatalf("Put #%d: %v", i, err)
		}
	}

	s.mu.Lock()
	p := s.pages[0]
	if p.tail() != perPage*span || p.FreeTail() >= span {
		s.mu.Unlock()
		t.Fatalf("setup: tail=%d free=%d, want a page full to within one record", p.tail(), p.FreeTail())
	}
	off := torn * span
	// A valLen no page can hold: decodeEntry refuses to frame it on every platform, so
	// its MAC never even gets checked — the drain resyncs to the next genuine frame.
	binary.LittleEndian.PutUint32(p.entries()[off+2:off+6], 0xFFFFFFF0)
	s.mu.Unlock()

	// The first record that does not fit drains the page, meeting the tear on the way.
	const refill = 150
	for i := range refill {
		if err := s.Put(fmt.Appendf(nil, "n%08d", i), val, 0); err != nil {
			t.Fatalf("refill Put #%d: %v", i, err)
		}
	}

	st := s.snapshot()
	if st.CorruptionErrors != 1 {
		t.Errorf("CorruptionErrors = %d, want 1", st.CorruptionErrors)
	}
	if want := uint64(span); st.CorruptionBytesDiscarded != want {
		t.Errorf("CorruptionBytesDiscarded = %d, want %d (only the one damaged record, not the whole tail)", st.CorruptionBytesDiscarded, want)
	}

	// No slot may address the page except the records written into it afterwards.
	s.mu.RLock()
	tab := s.tab.Load()
	for i := range tab.ctrl {
		c := tab.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		ref := slabRef(tab.refs[i].Load())
		key, _, _, err := s.pages[ref.pageIdx()].Read(ref.offset())
		if err != nil || hashKey(key) != tab.hashes[i] {
			t.Errorf("slot %d addresses page %d offset %d, which does not hold its key (read %q, err %v)",
				i, ref.pageIdx(), ref.offset(), key, err)
		}
	}
	s.mu.RUnlock()

	if st.Entries != refill {
		t.Errorf("Entries = %d, want %d (only the records written after the reset)", st.Entries, refill)
	}
	for i := range perPage {
		k := fmt.Appendf(nil, "k%08d", i)
		if v, err := s.Get(k); err != ErrNotFound {
			t.Errorf("Get(%s) = %d bytes, %v; want ErrNotFound (its page was drained)", k, len(v), err)
		}
	}
	seen := map[string]int{}
	s.iterate(func(key, _ []byte, _ uint64) bool {
		seen[string(key)]++
		return true
	})
	if len(seen) != refill {
		t.Errorf("Iterate yielded %d distinct keys, want %d", len(seen), refill)
	}
	for k, n := range seen {
		if n != 1 {
			t.Errorf("Iterate yielded %q %d times", k, n)
		}
	}

	// Every drained record was a removal, including those past the tear — except the
	// torn one itself, whose key cannot be read back and is never guessed at.
	removed := removedKeys()
	for i := range perPage {
		k := fmt.Sprintf("k%08d", i)
		switch {
		case i == torn && removed[k] != 0:
			t.Errorf("onRemove fired for the torn record %s, whose header cannot be trusted", k)
		case i != torn && removed[k] != 1:
			t.Errorf("onRemove fired %d times for %s, want 1", removed[k], k)
		}
	}
	// Every record on the page was displaced; all but the unreadable one were live.
	if st.Evictions != uint64(perPage) || st.EvictionsLive != uint64(perPage-1) {
		t.Errorf("Evictions=%d EvictionsLive=%d, want %d and %d", st.Evictions, st.EvictionsLive, perPage, perPage-1)
	}
}

// TestCorruptDrainResyncDropsEverySkippedSlot: when eviction's resync jumps over
// SEVERAL consecutive damaged frames, it must drop the index slot of EACH of them,
// not only the head one. A slot left addressing an offset inside the skipped gap is
// stranded below the advanced head — its bytes free to be overwritten on reuse — and a
// later Get resolves it through the MAC-less read path into whatever now lies there.
// Two consecutive tears (k1, k2) force the gap; after the drain no live slot may
// address the page, while the undamaged k0 and the resume target k3 evict normally.
func TestCorruptDrainResyncDropsEverySkippedSlot(t *testing.T) {
	s, removedKeys := corruptDrainShard(t)

	keys := [][]byte{[]byte("k0"), []byte("k1"), []byte("k2"), []byte("k3")}
	for _, k := range keys {
		if err := s.Put(k, []byte("val"), 0); err != nil {
			t.Fatalf("Put(%s): %v", k, err)
		}
	}

	s.mu.Lock()
	p := s.pages[0]
	gen := p.gen
	tab := s.tab.Load()
	// Corrupt k1 AND k2 (two consecutive live entries) by flipping a payload byte in
	// each, breaking their MACs so the drain's resync must skip both to reach k3.
	for _, k := range [][]byte{[]byte("k1"), []byte("k2")} {
		_, ref, ok := tab.findSlot(hashKey(k))
		if !ok {
			s.mu.Unlock()
			t.Fatalf("no index slot for %s", k)
		}
		p.entries()[int(ref.offset())+entryHeaderSize] ^= 0xFF
	}
	if err := s.drainPageLocked(0); err != nil {
		s.mu.Unlock()
		t.Fatalf("drainPageLocked: %v", err)
	}
	// After a full drain no live slot may still address this page generation. Pre-fix,
	// k2's slot (inside the skipped gap but not at its head) dangles here.
	dangling := 0
	tab = s.tab.Load()
	for i := range tab.ctrl {
		c := tab.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		if ref := slabRef(tab.refs[i].Load()); int(ref.pageIdx()) == 0 && ref.gen() == gen {
			dangling++
		}
	}
	s.mu.Unlock()
	if dangling != 0 {
		t.Errorf("%d index slot(s) still address the drained page; a multi-entry resync gap left them dangling", dangling)
	}

	rm := removedKeys()
	if rm["k0"] != 1 || rm["k3"] != 1 {
		t.Errorf("k0/k3 should evict normally (resync resumed at k3): removed=%v", rm)
	}
	if rm["k1"] != 0 || rm["k2"] != 0 {
		t.Errorf("damaged k1/k2 must not fire onRemove (keys unreadable): removed=%v", rm)
	}
}

// embeddedFrame is a complete, self-consistent v5 entry for key "forged" of exactly
// 64 bytes, carrying a write sequence far above anything the test writes. It is
// MAC'd under testFramingKey at offset 0 — a key the CLIENT chose, standing in for the
// per-file secret the client cannot know — so it is byte-for-byte a valid-looking
// frame, yet its MAC cannot verify at the position (and under the real per-file key
// and page nonce) it will occupy inside a carrier value.
func embeddedFrame(t *testing.T) []byte {
	t.Helper()
	key := []byte("forged")
	val := bytes.Repeat([]byte("P"), 64-entryHeaderSize-len(key))
	frame := make([]byte, entrySpanExact(len(key), len(val)))
	if _, err := encodeEntry(frame, key, val, 0, makeMeta(forgedSeq, false), testFramingKey, 0, 0); err != nil {
		t.Fatal(err)
	}
	if len(frame) != 64 {
		t.Fatalf("embedded frame is %d bytes, want 64", len(frame))
	}
	return frame
}

const forgedSeq = 1 << 50

// TestRebuildDoesNotAdoptAFrameStoredInsideAValue pins the FORGE resistance of the v5
// resync (issue #135 / the GHSA-m63m-rp87-w4rf class). A value is client bytes, and
// nothing stops a client storing bytes that form a complete, valid-looking entry — but
// the frame's MAC is keyed by a per-file secret the client cannot know and bound to the
// offset+nonce it sits at, so the resync forward scan NEVER adopts it: no key nobody
// wrote is served, and writeSeq is not lifted to the embedded frame's sequence.
//
// Two tears in the carrier cover the two resync strategies a naive implementation might
// use. A torn KEY BYTE leaves the header's lengths intact, so a forward scan for the
// next valid frame would reach the embedded one before the carrier's genuine successor.
// A torn LENGTH BIT (valLen 192 → 128) makes the carrier's own header claim an end
// exactly where the embedded frame starts, which is what a "skip the bad entry by its
// own length" walk steps to. Under v5 both instead skip the embedded frame (its MAC
// does not verify at that position) and resume at the carrier's GENUINE successor k2,
// which does verify — so k0 AND k2 survive and only the torn carrier is lost.
func TestRebuildDoesNotAdoptAFrameStoredInsideAValue(t *testing.T) {
	frame := embeddedFrame(t)
	carrierVal := append(bytes.Repeat([]byte("c"), 128), frame...) // valLen 192 = 0b11000000
	if len(carrierVal) != 192 {
		t.Fatalf("carrier value is %d bytes, want 192", len(carrierVal))
	}

	cases := []struct {
		name string
		tear func(entry []byte) // entry is the carrier's bytes on the page
	}{
		{"key byte", func(e []byte) { e[entryHeaderSize] ^= 0x01 }},
		{"valLen bit", func(e []byte) { binary.LittleEndian.PutUint32(e[2:6], 128) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.NumShards = 1
			cfg.PageSize = 1 << 20
			cfg.MaxMemoryPerShard = 1 << 20
			cfg.TTLSweepIntervalMs = 0
			cfg.DisableColdCompaction = true

			dir := t.TempDir()
			path := filepath.Join(dir, "pages.dat")
			offs := writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
				{[]byte("k0"), []byte("AAAA")},
				{[]byte("carrier"), carrierVal},
				{[]byte("k2"), []byte("CCCC")},
			})

			buf, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			base := headerSize + pageHdrSize
			tc.tear(buf[base+offs[1] : base+offs[2]])
			if err := os.WriteFile(path, buf, 0o644); err != nil {
				t.Fatal(err)
			}

			s, err := newShard(cfg, dir, nil)
			if err != nil {
				t.Fatalf("newShard: %v", err)
			}
			defer func() { _ = s.Close() }()

			if v, err := s.Get([]byte("forged")); err != ErrNotFound {
				t.Errorf("Get(forged) = %q, %v; a frame stored inside a value was indexed as an entry", v, err)
			}
			if s.writeSeq >= forgedSeq {
				t.Errorf("writeSeq = %d, lifted to the embedded frame's sequence", s.writeSeq)
			}
			if v, err := s.Get([]byte("k0")); err != nil || !bytes.Equal(v, []byte("AAAA")) {
				t.Errorf("Get(k0) = %q, %v; want AAAA (it precedes the tear)", v, err)
			}
			// The genuine entry AFTER the torn carrier is recovered by the resync, which
			// steps PAST the embedded forgery (its MAC fails at that offset) to k2's real
			// frame — the whole point of the position-bound MAC.
			if v, err := s.Get([]byte("k2")); err != nil || !bytes.Equal(v, []byte("CCCC")) {
				t.Errorf("Get(k2) = %q, %v; want CCCC (resync resumed at the genuine successor)", v, err)
			}
			st := s.snapshot()
			if st.CorruptionErrors != 1 {
				t.Errorf("CorruptionErrors = %d, want 1", st.CorruptionErrors)
			}
			// Only the torn carrier's bytes are discarded — from the tear to the start of
			// k2 — not the whole page tail a v4 build would have lost.
			if want := uint64(offs[2] - offs[1]); st.CorruptionBytesDiscarded != want {
				t.Errorf("CorruptionBytesDiscarded = %d, want %d (just the torn carrier)", st.CorruptionBytesDiscarded, want)
			}
		})
	}
}

// TestRebuildCountsTheWholePageWhenItsBoundsAreCorrupt: a page whose persisted
// head/tail are out of range is reset without walking it, and since those bounds are
// the only record of how much it held, the loss reported is the page's capacity.
func TestRebuildCountsTheWholePageWhenItsBoundsAreCorrupt(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{{[]byte("k0"), []byte("AAAA")}})
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte{0xF0, 0xFF, 0xFF, 0xFF}, int64(headerSize+4)); err != nil {
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
	if got, want := s.snapshot().CorruptionBytesDiscarded, uint64(cfg.PageSize-pageHdrSize); got != want {
		t.Errorf("CorruptionBytesDiscarded = %d, want %d (the page's whole capacity)", got, want)
	}
}

// TestRebuildSurvivesAValLenThatOverflowsInt32 is the warm-restart consequence of the
// decoder bound (entry_bounds_test.go): on a 32-bit build this valLen made
// decodeEntry slice with a negative bound, and the panic came back on every restart
// because the bytes that caused it were still on disk. It must be counted and the page
// truncated like any other unreadable entry.
func TestRebuildSurvivesAValLenThatOverflowsInt32(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	offs := writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
		{[]byte("b"), []byte("BBBB")},
	})
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	var lens [4]byte
	binary.LittleEndian.PutUint32(lens[:], 0x7FFFFFF0)
	if _, err := f.WriteAt(lens[:], int64(headerSize+pageHdrSize+offs[1]+2)); err != nil {
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
	if got := s.snapshot().CorruptionErrors; got != 1 {
		t.Errorf("CorruptionErrors = %d, want 1", got)
	}
	if v, err := s.Get([]byte("k0")); err != nil || !bytes.Equal(v, []byte("AAAA")) {
		t.Errorf("Get(k0) = %q, %v; want AAAA", v, err)
	}
	if _, err := s.Get([]byte("b")); err != ErrNotFound {
		t.Errorf("Get(b) = %v, want ErrNotFound", err)
	}
}

// TestCorruptDrainBoundsReportTheWholePage: a RUNNING mmap page whose persisted
// head/tail are corrupted — damage under a live shard, long after recovery checked
// them — must be reset by the drain with its slots dropped, and the loss reported as
// the page's capacity, exactly as recovery reports a page whose bounds it rejects.
// Those bounds were the only record of how much the page held, so no figure derived
// from them is honest: head > tail made tail-head negative, which wrapped to nearly
// 2^64 in the counter, and a tail past the page made the drain slice beyond it.
func TestCorruptDrainBoundsReportTheWholePage(t *testing.T) {
	cases := []struct {
		name       string
		head, tail func(capacity int) uint32
	}{
		// Both inside the page, with too little room past tail for the next record, so
		// the Put has to drain: EvictFront meets head > tail and refuses to frame.
		{"head past tail", func(c int) uint32 { return uint32(c - 60) }, func(c int) uint32 { return uint32(c - 100) }},
		{"tail past the page", func(int) uint32 { return 0 }, func(int) uint32 { return 0xFFFFFFF0 }},
		{"head and tail past the page", func(int) uint32 { return 0xFFFFFFF0 }, func(int) uint32 { return 0xFFFFFFF8 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, _ := corruptDrainShard(t)
			val := bytes.Repeat([]byte("v"), 4096)
			span := entrySpanExact(len("k00000000"), len(val))
			perPage := (s.cfg.PageSize - pageHdrSize) / span
			for i := range perPage {
				if err := s.Put(fmt.Appendf(nil, "k%08d", i), val, 0); err != nil {
					t.Fatalf("Put #%d: %v", i, err)
				}
			}

			s.mu.Lock()
			p := s.pages[0]
			capacity := len(p.entries())
			// Corrupt the RUNTIME bounds directly: after the torn-writeback fix the
			// mapped header no longer feeds head()/tail() (they are runtime fields
			// seeded from the durable header at open), so drainPageLocked's bounds
			// check reads these, not p.data[0:8]. int(uint32) reproduces the same
			// 32-bit widening the old header read had (a large value stays positive on
			// 64-bit, widens negative on 386).
			p.setHead(int(tc.head(capacity)))
			p.setTail(int(tc.tail(capacity)))
			s.mu.Unlock()

			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("draining a page with corrupt bounds panicked: %v", r)
					}
				}()
				if err := s.Put([]byte("n00000000"), val, 0); err != nil {
					t.Fatalf("Put after corrupting the bounds: %v", err)
				}
			}()

			st := s.snapshot()
			if st.CorruptionErrors != 1 {
				t.Errorf("CorruptionErrors = %d, want 1", st.CorruptionErrors)
			}
			if want := uint64(capacity); st.CorruptionBytesDiscarded != want {
				t.Errorf("CorruptionBytesDiscarded = %d, want %d (the page's capacity)", st.CorruptionBytesDiscarded, want)
			}
			if st.Entries != 1 {
				t.Errorf("Entries = %d, want 1 (only the record written after the reset)", st.Entries)
			}
			for i := range perPage {
				k := fmt.Appendf(nil, "k%08d", i)
				if _, err := s.Get(k); err != ErrNotFound {
					t.Errorf("Get(%s) = %v, want ErrNotFound", k, err)
					break
				}
			}
			s.mu.RLock()
			head, tail := s.pages[0].head(), s.pages[0].tail()
			s.mu.RUnlock()
			if head < 0 || head > tail || tail > capacity {
				t.Errorf("page bounds after the drain: head=%d tail=%d cap=%d", head, tail, capacity)
			}
		})
	}
}
