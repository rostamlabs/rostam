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
	// The table this probe is running against. It is t until a rehash is discovered
	// mid-probe, at which point the probe restarts on the live table — see rechaseSlot.
	tab := t
probe:
	for {
		for i := h & tab.mask; ; i = (i + 1) & tab.mask {
			c := tab.ctrl[i].Load()
			if c == ctrlEmpty {
				return nil, 0, 0, lkMiss // probe run ended: key absent
			}
			if c != tag {
				continue // tombstone or different tag
			}
			// Load the ref AFTER the control word: the writer stores ref before
			// publishing ctrl, so observing this tag guarantees a ref at least as new.
			r := slabRef(tab.refs[i].Load())
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
				// MOVED (relocating eviction copied it forward and repointed its slot) —
				// and rechaseSlot tells them apart, on the table that can actually answer.
				var live *indexTable
				p, r, live = tab.rechaseSlot(s, i, r)
				if live != nil {
					tab = live
					continue probe // this table is frozen; start over on the live one.
				}
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
}

// rechaseSlot re-resolves slot i after its ref failed the page-generation gate in
// get. Before relocating eviction a mismatch meant exactly one thing — the entry
// was evicted — and advancing to the next probe slot was the right answer.
// Relocation adds a second meaning: the entry MOVED, and its slot was repointed at
// the copy. Walking on past a relocated key's own slot would end the probe run
// somewhere else and report lkMiss — a wrong answer for a live key the cache
// deliberately kept — so the slot has to be re-read before that conclusion is drawn.
//
// THE TABLE MUST BE RE-CHECKED FIRST, and a mismatch there is not recoverable in
// place. A rehash publishes a FRESH table (s.tab.Store(t.rehashed())) and the old one
// is never written again, so every repoint after that lands somewhere this table
// cannot see: re-reading a slot in it would find the ref unchanged and wrongly report
// the record gone. The only correct answer is to hand the live table back and have the
// probe start over on it, which is what a non-nil third return means. That restart is
// not a spin — each one requires a rehash to have actually happened, and a rehash
// costs a table's worth of inserts.
//
// WITHIN ONE TABLE, A SINGLE RE-READ IS AUTHORITATIVE — there is no retry budget here
// and none is needed. A slot always holds the LATEST ref for its key, so one re-read
// gets the end of the chain, never an intermediate hop: a record relocated twice is
// found at its second destination, not its first. That leaves exactly two outcomes,
// and both are final. An UNCHANGED ref means no relocation ever repointed this slot,
// so the record was evicted and the miss is correct (a tombstone store leaves refs
// alone, which is why the ref is the thing to compare). A CHANGED ref that STILL fails
// the generation gate means the latest copy is itself on a page that has been freed —
// and relocation repoints a record's slot BEFORE the page holding it is freed, which
// is the ordering the whole design rests on, so a latest ref pointing into a freed page
// is a record that was dropped rather than moved. The miss is correct there too.
//
// Returns (nil, r, nil) when the caller should advance to the next probe slot,
// (page, ref, nil) when the record was found, or (nil, r, live) when the caller must
// restart its probe against the returned table.
func (t *indexTable) rechaseSlot(s *shard, i uint64, r slabRef) (*page, slabRef, *indexTable) {
	if live := s.tab.Load(); live != t {
		return nil, r, live // frozen table: nothing it says about this slot is current.
	}
	cur := slabRef(t.refs[i].Load())
	if cur == r {
		return nil, r, nil // unchanged: the entry really is gone.
	}
	p := s.pageSlots[cur.pageIdx()].Load()
	if p == nil || p.gen != cur.gen() {
		return nil, cur, nil // latest copy is on a freed page: dropped, not moved.
	}
	return p, cur, nil
}

// getSeq is get under the per-entry seqlock (cache/seqlock.go), for a shard
// whose writers can rewrite a live entry's bytes where they lie. It differs from
// get in two ways, both forced by that:
//
//   - it COPIES the value into dst rather than returning an alias into the page,
//     because an alias cannot be validated — the bytes could change after the
//     caller has been handed them;
//   - it reports ok=false when the entry's version moved underneath it, meaning
//     the caller must retry (or take the read lock once its budget is spent).
//     ok=false is NOT a miss and must never be reported as one.
//
// TWO REASONS A READ RETRIES, AND THEY COMPOSE IN ONE ORDER ONLY. A generation
// mismatch means the record MOVED — relocating eviction copied it forward and
// repointed its slot — and rechaseSlot resolves that to the copy the index now
// names, inside this probe, exactly as get does. The version check means the
// record CHANGED WHERE IT LIES. The version must therefore be snapshotted from
// the page and offset rechase SETTLED ON, never from the ref the slot first
// advertised: a relocated record's old page is frozen and its counter never moves
// again, so validating against it would miss every later in-place update at the
// record's real home. Neither retry substitutes for the other — one re-resolves
// WHERE the record is, the other re-reads WHAT it says — and collapsing them
// would answer one question with the other's evidence.
//
// WHY THE RACY DECODE CANNOT MISBEHAVE. p.Read on bytes a writer may be
// rewriting looks alarming, but an in-place update is the ONLY thing that
// rewrites live bytes here, and it writes the SAME key at the SAME length — so
// the keyLen and valLen fields it stores are byte-for-byte what was already
// there. A torn read of an unchanged byte is that byte. The framing a reader
// decodes is therefore stable even mid-write, and the slice bounds derived from
// it cannot go out of range. Only the expiry, the meta word and the value can
// differ, and each is validated by the trailing version check before it reaches
// the caller. Eviction and relocation DO change framing, but both leave the old
// page frozen and are already handled by the generation gate above.
func (t *indexTable) getSeq(s *shard, dst, key []byte, h uint64) (out []byte, exp uint64, ref slabRef, st lookupStatus, ok bool) {
	tag := tagFor(h)
	tab := t
	// Restarts are bounded as well as retries: see seqlockMaxRestarts. Exhausting
	// this reports a retry, so the caller reaches the read lock rather than
	// circling here.
	for restarts := 0; ; restarts++ {
		if restarts > seqlockMaxRestarts {
			return dst, 0, 0, lkMiss, false
		}
		for i := h & tab.mask; ; i = (i + 1) & tab.mask {
			c := tab.ctrl[i].Load()
			if c == ctrlEmpty {
				return dst, 0, 0, lkMiss, true // probe run ended: key absent
			}
			if c != tag {
				continue
			}
			r := slabRef(tab.refs[i].Load())
			p := s.pageSlots[r.pageIdx()].Load()
			if p == nil || p.gen != r.gen() {
				// Evicted, or MOVED and the slot repointed. rechaseSlot tells them
				// apart on the table that can answer, and hands back the live table
				// when this one has been frozen by a rehash. See get.
				var live *indexTable
				p, r, live = tab.rechaseSlot(s, i, r)
				if live != nil {
					tab = live
					break // restart the probe on the live table
				}
				if p == nil {
					continue
				}
			}
			// The ref is final from here, so this is the counter that actually guards
			// the bytes about to be read.
			ver := p.versionAt(r.offset())
			if ver == nil {
				// No counter on this page, so nothing here can be validated. Report a
				// retry so the caller takes the read lock rather than trusting an
				// unguarded read.
				return dst, 0, 0, lkMiss, false
			}
			v1 := ver.Load()
			if v1&1 != 0 {
				return dst, 0, 0, lkMiss, false // a rewrite is in flight on this stripe
			}
			k, val, e, err := p.Read(r.offset())
			if err != nil {
				if ver.Load() != v1 {
					return dst, 0, 0, lkMiss, false
				}
				return dst, 0, 0, lkCorrupt, true
			}
			if !bytes.Equal(k, key) {
				// Either a genuinely different key on this hash, or bytes that moved
				// under the comparison. Only the version says which, and guessing wrong
				// in the second case would report a miss for a key that is present.
				if ver.Load() != v1 {
					return dst, 0, 0, lkMiss, false
				}
				continue
			}
			out = append(dst, val...)
			if ver.Load() != v1 {
				return dst, 0, 0, lkMiss, false // the value moved while it was copied
			}
			return out, e, r, lkHit, true
		}
	}
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
