// SPDX-License-Identifier: Apache-2.0

package cache

// Relocating eviction — keeping the live records that share a page with dead versions.
//
// ==========================================================================
// THE PROBLEM. Under PolicyRingbufEvict a heap shard at its page cap frees space
// POSITIONALLY: evictUntilFitsLocked picks the next non-empty page in rotation and
// drains it IN FULL, because tail room only reappears when a page empties. Pages are
// append-only, so overwriting a key leaves its previous copy framed exactly where it
// was and a page accumulates superseded versions of the keys written into it. Draining
// cannot pick and choose: a record that is STILL the live copy for its key goes out
// with the dead versions beside it, for a reason that has nothing to do with its own
// recency or TTL — only with which page it landed on. The busier the overwrite traffic
// on a page, the more of what it drops is already garbage, and the more arbitrary the
// loss of the one record that was not.
//
// ==========================================================================
// THE FIX. Before a page is drained, copy its live records forward and repoint their
// index slots, so the drain finds only dead versions to drop. Mechanically each move is
// the pair putAtExpLocked already performs — append into a page's fresh tail, then
// atomically repoint the slot — so it inherits that path's reader safety exactly: the
// bytes are written before the ref addressing them is published, and a lock-free reader
// resolves either the old copy (still framed, not yet drained) or the new one, never a
// tear. Nothing about the read path changes for it, with one exception described under
// THE READER below.
//
// ==========================================================================
// WHERE THE ROOM COMES FROM, AND WHY THE SOURCE IS THE *NEXT* VICTIM. Eviction runs
// only when NO page has `need` bytes of tail room — that is the condition that summoned
// it — so at the moment a victim is chosen there is nowhere to relocate anything TO. The
// only room in the shard is the room the eviction itself is about to create. So the pass
// runs immediately AFTER the victim is retired, filling the page just freed, and its
// source is the page the rotation cursor will pick NEXT. One eviction later that page is
// drained and by then holds no live copy to lose. In steady state every page is
// evacuated exactly one eviction before it is retired, and the shard stops paying for
// dead versions with live records.
//
// A freed page cannot rescue its OWN live records, and the reason is the read path
// rather than bookkeeping. pageSlots holds exactly ONE page object per index and a
// lock-free reader resolves every ref through it (indexTable.get); retiring publishes
// the fresh object into that slot, which is precisely what makes refs minted against the
// old object stop resolving. Relocating within one index would need both objects
// addressable at once: publish the fresh page first and every not-yet-moved record is
// unreachable until its repoint lands, repoint first and the new ref is unreachable
// until the publish lands. Either order opens a window where a live key resolves to
// nothing. Across two DIFFERENT page indices there is no such window — both objects stay
// published for the whole pass, so every ref, old or new, resolves at every instant.
//
// ==========================================================================
// ORDERING IS LOAD-BEARING. A relocated record is repointed at its new copy, so when the
// source page's own retire walk runs — the next eviction — that walk's `cur == ref` test
// fails for it. It is therefore NOT tombstoned, onRemove does NOT fire for it (which
// would drop a live key's postings from every derived index), and it is NOT counted in
// evictionsLive. The difference between a record that moved and a record that was lost
// is expressed entirely by the order of these two passes.
//
// Logically expired records are deliberately NOT moved: they are the retire walk's to
// tombstone, exactly as today, and carrying a corpse forward would spend budget to
// keep nothing.
//
// ==========================================================================
// THE READER. One read-path assumption does change, and indexTable.rechaseSlot is where
// it is repaired: a page-generation mismatch used to mean "this entry was evicted", so
// the probe advanced to the next slot. Relocation makes it also mean "this entry moved",
// and advancing past a live key's own slot would end the probe run elsewhere and report
// a miss. The reader now re-reads the ref once on a mismatch and retries the slot when
// it changed. See rechaseSlot for the bound.
//
// ==========================================================================
// BOUNDS. This runs on the WRITE PATH, under s.mu, inside a Put that is already paying
// for an eviction, and it copies bytes. Three bounds keep that cost a fraction of the
// eviction it rides on:
//
//   - `need` is reserved first. PolicyRingbufEvict guarantees a write succeeds;
//     relocation is a passenger on this one and must never consume the room it was
//     evicting for, nor fail it, nor stall it.
//   - at most PageSize/relocateMaxBytesPerEvictionDivisor bytes move per eviction, so
//     the rest of every freed page is genuinely reclaimed and the loop can never be
//     talked into handing back a page as full as the one it drained.
//   - it never allocates a page and NEVER recurses into eviction to make room. A record
//     that does not fit is left where it is and the retire walk drops it exactly as it
//     does today. Dropping is the degraded path, not an error.

// relocateMaxBytesPerEvictionDivisor caps what one eviction may spend relocating at
// PageSize/divisor bytes. It bounds the copy a single Put can be charged for, and it is
// what guarantees forward progress: at least (divisor-1)/divisor of every freed page is
// left reclaimed.
const relocateMaxBytesPerEvictionDivisor = 2

// relocateIntoFreedPageLocked copies the live records of the page the rotation cursor
// will drain NEXT into freedIdx — the page evictVictimLocked has just retired, and the
// only page with room at this point in an eviction. need is the triggering write's byte
// requirement, reserved before anything is relocated.
//
// Heap ringbuf only, and only under Config.RelocatingEviction. Must hold mu for writing.
func (s *shard) relocateIntoFreedPageLocked(freedIdx, need int) {
	dst := s.pages[freedIdx]
	// Room left AFTER the triggering write. The write that summoned this eviction is
	// served out of exactly this page, so its requirement comes off the top before any
	// budget exists at all.
	budget := dst.FreeTail() - need
	if maxBytes := s.cfg.PageSize / relocateMaxBytesPerEvictionDivisor; budget > maxBytes {
		budget = maxBytes
	}
	if budget <= 0 {
		return
	}
	// Skip the freed page itself: it is empty (so the scan would pass over it anyway)
	// and it must never be its own source — this keeps the search on the page that will
	// actually be drained next.
	src := s.nextNonEmptyPageLocked(freedIdx)
	if src < 0 {
		return
	}
	p := s.pages[src]
	entries := p.entries()
	tail := p.tail()
	t := s.tab.Load()
	// The eviction clock, the same wall clock retirePageLocked classifies against. It
	// decides only what is worth MOVING; what the shard retains is still decided by the
	// retire walk's cur == ref alone, so this cannot make two nodes diverge.
	now := s.now()
	spent := 0
	var moved, movedBytes uint64
	for cursor := p.head(); cursor < tail; {
		key, value, expiryMs, err := decodeEntryFast(entries[cursor:tail])
		if err != nil {
			// Heap pages carry no CRC and cannot be corrupted by anything outside the Go
			// runtime, so this is unreachable in practice; stop the walk defensively,
			// exactly as retirePageLocked does.
			break
		}
		size := entrySize(len(key), len(value))
		if spent+size > budget {
			break // budget spent; everything past here stays where it is.
		}
		ref := makeSlabRef(uint16(src), p.gen, uint32(cursor)) //nolint:gosec // src bounded by MaxPagesPerShard (≤65535); cursor < PageSize ≤ MaxInt32
		h := hashKey(key)
		_, cur, ok := t.findSlot(h)
		if !ok || cur != ref {
			// A superseded copy, a hash-collision loser, or an already-tombstoned slot:
			// no read path can reach it, so there is nothing here to keep.
			cursor += size
			continue
		}
		if isExpired(expiryMs, now) {
			// The retire walk's to tombstone, exactly as today.
			cursor += size
			continue
		}
		// Append into the freed page's fresh tail, then repoint the slot. key/value
		// alias the SOURCE page, a different slab from dst, so the copy can never
		// overlap itself. A fresh writeSeq keeps the moved copy strictly newer than the
		// original it leaves framed behind; the tombstone bit passes through verbatim
		// (it is always clear here — an index-current entry is not a delete record).
		newSeq := s.writeSeq + 1
		off, _, werr := dst.Write(key, value, expiryMs, makeMeta(newSeq, metaIsTombstone(entryMetaAt(entries[cursor:tail]))))
		if werr != nil {
			// Unreachable: the budget already proved the room. Stop rather than risk a
			// half-move.
			break
		}
		s.writeSeq = newSeq
		// upsert repoints the slot findSlot just matched, so the table's live/tombstone
		// counts are untouched and it can neither grow nor need a rehash here.
		t.upsert(h, makeSlabRef(uint16(freedIdx), dst.gen, off)) //nolint:gosec // freedIdx bounded by MaxPagesPerShard (≤65535)
		spent += size
		movedBytes += uint64(size) //nolint:gosec // entrySize is non-negative
		moved++
		cursor += size
	}
	if moved == 0 {
		return
	}
	// Bytes BEFORE the count: snapshot() loads them in the opposite order, so a scrape
	// landing between these two can only see bytes that run ahead of the count, never
	// the impossible pair of records relocated with nothing moved.
	s.evictRelocatedBytes.Add(movedBytes)
	s.evictRelocations.Add(moved)
}
