// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"sync/atomic"
)

// indexTable is a single-writer / multi-reader open-addressing hash table that
// replaces the per-shard map[uint64]slabRef. Writers mutate it under the shard's
// mu (serialized); readers take NO lock and rely on publication ordering plus a
// full-key bytes.Equal backstop.
//
// Each slot is described by three parallel arrays:
//
//	ctrl[i]   atomic control word: 0 = empty, 1 = tombstone, else a non-zero tag
//	          derived from the high bits of the key hash.
//	refs[i]   atomic slabRef (the packed page index + offset).
//	hashes[i] the full 64-bit key hash. Writer-only metadata (read only under mu);
//	          readers never touch it, so it needs no atomic access.
//
// The table is keyed by the full 64-bit hash h, exactly like the map it replaces:
// two keys that collide on h share one logical slot (last writer wins), and the
// bytes.Equal guard on the read path turns a stale/foreign entry into a miss.
type indexTable struct {
	ctrl   []atomic.Uint64
	refs   []atomic.Uint64
	hashes []uint64
	mask   uint64

	// writer-only bookkeeping (guarded by shard.mu)
	live int // occupied (tag) slots
	tomb int // tombstone slots
}

const (
	ctrlEmpty     uint64 = 0
	ctrlTombstone uint64 = 1
	// tagPresentBit forces every real tag above the empty/tombstone sentinels, so
	// a tag can never be mistaken for 0 (empty) or 1 (tombstone).
	tagPresentBit uint64 = 1 << 16
	// minIndexSlots is the smallest table (must be a power of two).
	minIndexSlots = 8
	// relocateProbeRetries bounds how many times one probe re-resolves a SINGLE
	// slot whose ref failed the page-generation gate (see rechaseSlot). A reader
	// takes no lock, so without a ceiling a writer relocating the same key over
	// and over could hold it on one slot indefinitely.
	relocateProbeRetries = 4
)

// tagFor derives the non-zero control tag for a hash. The tag prunes most
// mismatches before a page read; a false tag hit just costs one extra Read +
// bytes.Equal, so correctness never depends on tag width.
func tagFor(h uint64) uint64 { return (h >> 48) | tagPresentBit }

// nextPow2 returns the smallest power of two >= n, floored at minIndexSlots.
func nextPow2(n int) int {
	s := minIndexSlots
	for s < n {
		s <<= 1
	}
	return s
}

// newIndexTable allocates an empty table with room for at least `entries` live
// entries at a target load factor of 0.5 (so lookups stay short).
func newIndexTable(entries int) *indexTable {
	slots := nextPow2(2*entries + 1)
	return &indexTable{
		ctrl:   make([]atomic.Uint64, slots),
		refs:   make([]atomic.Uint64, slots),
		hashes: make([]uint64, slots),
		mask:   uint64(slots - 1), //nolint:gosec // slots is a positive power of two
	}
}

// get probes for key/h and returns the value slice (aliasing the page backing
// store), its expiry, and the slabRef. Lock-free: safe to call without mu.
// The returned value must be copied before use if the shard can overwrite pages
// in place (PolicyRingbufEvict) — see the shard read path.
func (t *indexTable) get(s *shard, key []byte, h uint64) (v []byte, exp uint64, ref slabRef, st lookupStatus) {
	tag := tagFor(h)
	for i := h & t.mask; ; i = (i + 1) & t.mask {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty {
			return nil, 0, 0, lkMiss // probe run ended: key absent
		}
		if c != tag {
			continue // tombstone or different tag
		}
		// Load the ref AFTER the control word: the writer stores ref before
		// publishing ctrl, so observing this tag guarantees a ref at least as new.
		r := slabRef(t.refs[i].Load())
		// Resolve the page OBJECT pointer atomically (never an index into a mutable
		// slice). Once loaded, p keeps that exact object GC-alive for the rest of
		// this read. pageSlots has fixed length MaxPagesPerShard and its header
		// never changes, so this single atomic load is race-free; the writer stores
		// the page into the slot before publishing the entry's ctrl, so a ref we
		// observed points at a populated slot.
		p := s.pageSlots[r.pageIdx()].Load()
		if p == nil || p.gen != r.gen() {
			// Slot unpopulated, or the page's generation does not match the ref. Do
			// NOT read bytes: a freshly-swapped page may be under active append by a
			// writer. A mismatch has TWO meanings now — the entry was evicted, or it
			// MOVED (relocating eviction copied it forward and repointed this very
			// slot) — and rechaseSlot tells them apart. A nil page answers "gone".
			p, r = t.rechaseSlot(s, i, r)
			if p == nil {
				continue
			}
		}
		k, val, e, err := p.Read(r.offset())
		if err != nil {
			return nil, 0, 0, lkCorrupt
		}
		if !bytes.Equal(k, key) {
			continue // false-tag or stale ref → keep probing
		}
		return val, e, r, lkHit
	}
}

// rechaseSlot re-resolves slot i after its ref failed the page-generation gate in
// get. Before relocating eviction a mismatch meant exactly one thing — the entry
// was evicted — and advancing to the next probe slot was the right answer.
// Relocation adds a second meaning: the entry MOVED, and this slot was repointed
// at the copy. Reloading the ref separates them. A CHANGED ref is a relocation, so
// the SAME slot is retried against the new page; an UNCHANGED ref is an eviction,
// so the caller advances exactly as it always did. Walking on past a relocated
// key's own slot would end the probe run somewhere else and report lkMiss — a
// wrong answer for a live key that was deliberately kept.
//
// The retry is bounded by relocateProbeRetries so no amount of churn can hold a
// lock-free reader on one slot. EXHAUSTING THE BOUND degrades to today's
// behaviour: it reports "gone" and the caller advances, which for a live key is a
// miss. The bound is therefore a ceiling nothing realistic reaches (a relocation
// needs a full eviction between repoints), not a tuning knob.
//
// Returns (nil, r) when the caller should advance to the next probe slot, or the
// resolved page and the ref that resolved against it.
func (t *indexTable) rechaseSlot(s *shard, i uint64, r slabRef) (*page, slabRef) {
	for range relocateProbeRetries {
		cur := slabRef(t.refs[i].Load())
		if cur == r {
			return nil, r // unchanged: the entry really is gone.
		}
		r = cur
		p := s.pageSlots[r.pageIdx()].Load()
		if p != nil && p.gen == r.gen() {
			return p, r
		}
	}
	return nil, r
}

// lookupStatus is the outcome of a table probe.
type lookupStatus uint8

const (
	lkMiss lookupStatus = iota
	lkCorrupt
	lkHit
)

// findSlot returns the slot index and current ref of the live entry for hash h,
// or ok=false if absent. Writer-side (call under mu).
func (t *indexTable) findSlot(h uint64) (slot uint64, ref slabRef, ok bool) {
	tag := tagFor(h)
	for i := h & t.mask; ; i = (i + 1) & t.mask {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty {
			return 0, 0, false
		}
		if c == tag && t.hashes[i] == h {
			return i, slabRef(t.refs[i].Load()), true
		}
	}
}

// upsert inserts or updates the entry for hash h to point at ref. Writer-side
// (call under mu). Publication order: ref is stored before ctrl so a lock-free
// reader that sees the tag also sees a valid ref.
func (t *indexTable) upsert(h uint64, ref slabRef) {
	tag := tagFor(h)
	firstFree := -1
	for i := h & t.mask; ; i = (i + 1) & t.mask {
		c := t.ctrl[i].Load()
		switch {
		case c == ctrlEmpty:
			slot := i
			if firstFree >= 0 {
				slot = uint64(firstFree) //nolint:gosec // firstFree is a valid slot index
				t.tomb--
			}
			t.hashes[slot] = h
			t.refs[slot].Store(uint64(ref)) // store value first
			t.ctrl[slot].Store(tag)         // then publish control
			t.live++
			return
		case c == ctrlTombstone:
			if firstFree < 0 {
				firstFree = int(i) //nolint:gosec // i <= mask fits an int
			}
		case c == tag && t.hashes[i] == h:
			// Existing key: repoint at the new physical copy. ctrl already carries
			// the tag, so a plain ref store republishes the value in place.
			t.refs[i].Store(uint64(ref))
			return
		}
	}
}

// tombstone marks a slot deleted. Writer-side (call under mu).
func (t *indexTable) tombstone(slot uint64) {
	t.ctrl[slot].Store(ctrlTombstone)
	t.live--
	t.tomb++
}

// overThreshold reports whether the fill (live + tombstones) has reached 3/4 of
// capacity, at which point the writer should rehash into a fresh table.
func (t *indexTable) overThreshold() bool {
	return (t.live+t.tomb)*4 >= len(t.ctrl)*3
}

// rehashed builds a fresh table sized for the current live set (load ≈ 0.5) and
// copies every live entry into it, dropping all tombstones. Writer-side.
func (t *indexTable) rehashed() *indexTable {
	nt := newIndexTable(t.live)
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		nt.upsert(t.hashes[i], slabRef(t.refs[i].Load()))
	}
	return nt
}
