// SPDX-License-Identifier: Apache-2.0

package cache

// Relocating eviction — keeping the live records that share a page with dead versions.
//
// ==========================================================================
// SCOPE. Both PolicyRingbufEvict storage modes: HEAP shards, which free a page by
// retiring it (a fresh frozen object swapped into pageSlots), and SINGLE-NODE MMAP
// shards, which drain the fixed persisted region in place. No replicated shard
// reaches either path — replication forces reject-writes (shard/store.go) — and the
// mmap/replicated/reject-writes combination is the online compactor's, not this one
// (cache/compact_online.go). The two modes share the whole pass; what differs is
// collected under MMAP at the bottom.
//
// ==========================================================================
// THE PROBLEM. Under PolicyRingbufEvict a shard at its page cap frees space
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
//
// ==========================================================================
// MMAP. A single-node mmap ringbuf shard runs the identical pass; four things about
// it are worth stating because they are NOT the same argument as the heap case.
//
// THE DRAIN IS IN PLACE. drainPageLocked walks EvictFront, advancing head over each
// entry and tombstoning the slot when cur == ref, rather than swapping the page
// object. The freed page is therefore the SAME object, emptied (head == tail == 0),
// and relocation appends into its tail — over bytes the drain has just released. That
// is the same overwrite the ordinary Puts that follow an eviction already perform; it
// is not a new exposure.
//
// THE FREED PAGE IS STILL NOT ITS OWN SOURCE, BUT NOT FOR THE HEAP REASON. The
// publish-window argument above is a READER argument and it does NOT carry over here:
// nothing is swapped, the page object and its generation are unchanged, and readers
// are excluded by the lock regardless. What rules it out on mmap is the drain itself.
// By the time this page has room its records are already GONE from the index — the
// drain tombstoned every slot where cur == ref as it advanced head — so there is
// nothing left to move out of it. Rescuing them would mean copying them into a scratch
// buffer BEFORE the drain, teaching the drain to skip the ones earmarked for
// relocation (otherwise it tombstones them, fires onRemove for them and counts them in
// evictionsLive — the very ordering this pass exists to get right), and re-appending
// afterwards: an allocation on the write path and a second liveness rule, to cover the
// same pages this scheme already covers one eviction earlier. So both modes take the
// same shape, and on mmap that is a simplicity argument rather than a safety one.
//
// READERS ARE EXCLUDED, NOT RACED. needsReadLockForGet() is true for mmap ringbuf, so
// a reader holds the shard READ lock while eviction holds the WRITE lock: the whole
// pass — appends and repoints together — is atomic with respect to every reader. The
// stale-ref repair in indexTable.rechaseSlot is what makes the lock-FREE heap path
// safe; nothing on this path depends on it (it stays correct here, just unreachable,
// since a reader can never observe a half-done relocation to begin with).
//
// RECOVERY RESOLVES BY SEQUENCE, WHICH THE ORDINARY APPEND ALREADY GIVES US. A
// relocated copy leaves the original framed on its source page until that page is
// drained, so the file transiently holds two copies of one key. rebuildIndexFromPages
// resolves that by MAX SEQUENCE, not page order (see indexedSeqAtLeast), and
// relocation writes through the same writeSeq++ / makeMeta pair the write path uses,
// so the relocated copy is strictly newer and wins. The tombstone flag is written
// CLEAR rather than copied from the original: relocation only ever moves an
// INDEX-CURRENT entry, and a delete record's slot is already tombstoned, so cur == ref
// can never select one.
//
// CRASH CONSISTENCY IS AN ARGUMENT, NOT AN msync. Durability here is the periodic
// msync (MsyncIntervalMs), so neither the appended copy nor the victim's advanced head
// is synchronously durable and the OS may write them back in either order. We do NOT
// add a per-eviction sync — that would put a flush on the write path at eviction rate.
// The invariant that makes it unnecessary:
//
//	RELOCATION ONLY EVER MOVES AN ENTRY THAT AN EVICTION IS ALREADY GOING TO DELETE
//	OUTRIGHT — specifically the NEXT one, which drains the source page — and it does
//	not bring that deletion any closer. So its worst crash outcome is the outcome
//	today's policy produces unconditionally: it can fail to SAVE an entry, and can
//	never LOSE one today's code would have kept.
//
// The implementation honours it by construction: the source is the page the rotation
// cursor selects next (see nextNonEmptyPageLocked, where that alignment is spelled
// out), and only its index-current entries are touched — every one of them is doomed
// the moment that drain runs.
//
// THE DEFERRAL MAKES THIS EASIER TO SATISFY, NOT HARDER, which is worth stating
// because the entry is moved one eviction BEFORE the drain that would have deleted it.
// The interleaving to worry about — "the head advance is durable, the copy is lost" —
// cannot arise from the eviction that does the copying at all: that eviction never
// touches the source page's head, it only appends elsewhere. The original stays
// framed, exactly as durable as it was a moment earlier. The head advance that could
// strand a lost copy belongs to the NEXT eviction, which is precisely the one that was
// going to delete the entry regardless.
//
// Case by case, for an entry moved at eviction N out of the page eviction N+1 drains:
// the copy is durable and the entry survives; or the copy is lost and N+1's head
// advance is NOT durable, so the rebuild finds the original still framed and the entry
// survives — which is what today's code does from that same crash state too; or the
// copy is lost and N+1's head advance IS durable, so the entry is gone — which is what
// today's code does unconditionally at N+1. No case is worse than today, and the
// middle one is better.
//
// The reverse skew — appends durable, the drained page's reset head/tail not — leaves
// the rebuild framing stale bounds over partially overwritten bytes. That is the
// pre-existing exposure of any post-drain append, and the per-entry CRC is what
// catches it: the walk rejects the entry and truncates the page tail.

// relocateMaxBytesPerEvictionDivisor caps what one eviction may spend relocating at
// PageSize/divisor bytes. It bounds the copy a single Put can be charged for, and it is
// what guarantees forward progress: at least (divisor-1)/divisor of every freed page is
// left reclaimed.
const relocateMaxBytesPerEvictionDivisor = 2

// relocateIntoFreedPageLocked copies the live records of the page the rotation cursor
// will drain NEXT into freedIdx — the page evictVictimLocked has just freed (retired on
// heap, drained in place on mmap), and the only page with room at this point in an
// eviction. need is the triggering write's byte requirement, reserved before anything
// is relocated.
//
// Ringbuf only, and only under Config.RelocatingEviction. Must hold mu for writing.
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
	// The page the NEXT eviction will select, which is what makes every record moved
	// below one that was about to be dropped anyway. Skipping the freed page is what
	// keeps the two selections identical — see nextNonEmptyPageLocked for why that
	// equality is load-bearing and what breaks if the two callers ever drift.
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
		// Append into the freed page's fresh tail, then repoint the slot — the same
		// writeSeq++ / makeMeta / Write / upsert sequence putAtExpLocked performs.
		// key/value alias the SOURCE page, a different slab from dst, so the copy can
		// never overlap itself. Going through the ordinary append is what gives the
		// moved copy a strictly higher write sequence than the original it leaves
		// framed behind, which is what recovery resolves duplicates by. The tombstone
		// flag is written CLEAR, never carried over: only an index-current entry gets
		// here, and a delete record's slot is already tombstoned.
		newSeq := s.writeSeq + 1
		off, _, werr := dst.Write(key, value, expiryMs, makeMeta(newSeq, false))
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
