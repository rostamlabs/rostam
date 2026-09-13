// SPDX-License-Identifier: Apache-2.0

package cache

// The FREE-PAGE RESERVE — relocating eviction paid for by the sweeper, not by a write.
//
// ==========================================================================
// WHAT THIS LAYER IS FOR. cache/relocate_evict.go describes the pass itself: before a
// page is drained, copy its live records forward and repoint their index slots, so the
// drain finds only dead versions to drop. That pass runs INSIDE the eviction a write
// triggered, so the copying is charged to that write — measurably, in an arm where the
// shard sits at its page cap and every write evicts.
//
// This layer moves the copying to the per-shard sweeper and leaves the write path a
// FALLBACK. The sweeper keeps a small RESERVE of free room, so a write at capacity
// usually finds a page with room in firstPageWithRoomLocked and never reaches
// evictUntilFitsLocked at all. When the reserve is empty — a burst outran the sweeper,
// or the shard is too dense for the sweeper to get ahead of — the write evicts and
// relocates inline exactly as it does today. Nothing here can block a write, fail one,
// or weaken PolicyRingbufEvict's guarantee that a write always succeeds: the write path
// is not touched.
//
// ==========================================================================
// THE TRADE, STATED PLAINLY. This does MORE total copying than the synchronous design,
// not less. The synchronous pass moves a record at the last possible moment — one
// eviction before its page is drained — so a record superseded in the meantime is never
// copied at all. The sweeper moves records on a CLOCK instead, which means it copies
// records that would have died before their page came round, and it holds a page or two
// of capacity empty that the shard could otherwise have been storing data in. What it
// buys is that a steady-state write pays none of it. It is a latency-for-throughput
// trade with a capacity cost, not a free win.
//
// ==========================================================================
// EACH ROUND IS ONE EVICTION, RUN EARLY. A round takes the page the rotation cursor
// would drain next, moves its live records to safety, and — while the shard is short of
// room — retires it. That is evictUntilFitsLocked's body with the write taken out of
// it, and the victim comes from the IDENTICAL nextNonEmptyPageLocked(-1) call, so the
// two cannot select different pages: there is one selection rule between them. See
// nextNonEmptyPageLocked's doc for why that equality is load-bearing and what breaks if
// it is ever broken.
//
// The re-validation after every released lock enforces the same thing over time. If the
// rotation cursor moved while the lock was down — a write-path eviction ran — the page
// in hand is no longer the next drain target and the round ABANDONS it rather than
// finish evacuating a page nothing is about to drain.
//
// A RETIRE MUST EVACUATE ITS SUCCESSOR IN THE SAME BREATH, and leaving that out is a
// bug that costs live records. Retiring advances the rotation cursor onto the NEXT page,
// which nothing has evacuated; a write-path eviction slipping into that gap drains a page
// with live records still on it — a loss neither scheme takes on its own, since every
// synchronous eviction evacuates its successor on the way past. So the retire runs
// relocateIntoFreedPageLocked UNDER THE SAME LOCK HOLD, the same pass with the same
// budget evictVictimLocked would have run, leaving no gap for anything to slip into. Its
// bytes are charged to the sweeper, because no write paid for them.
//
// ==========================================================================
// THE LAST ROUND PREPARES RATHER THAN FREES. A tick runs one round MORE than the reserve
// needs. That extra round finds the reserve already met, evacuates the victim and leaves
// it in place — finishing whatever the retire's handoff budget did not reach. Whatever
// drains it next (a write-path eviction, or the next tick's first round, which then
// retires it for free) finds only dead versions. The chain the synchronous pass maintains
// through eviction pressure is maintained here through the clock, and the handoff between
// the two is seamless in both directions.
//
// ==========================================================================
// THE RESERVE IS COUNTED IN ROOM, NOT IN EMPTY PAGES, because room is what the write
// path consumes: findOrMakePageLocked wants one page with `need` contiguous tail bytes,
// and a page holding a little data still has most of a page of that. Counting empty
// pages instead would make a freed page stop counting the moment one record landed in
// it and would have the sweeper chasing a threshold rather than a quantity.
//
// ==========================================================================
// ONE INEQUALITY GOVERNS THE WHOLE PASS, and it is the thing to understand before
// changing anything here. For the page in hand, tail - page.relocatedOut is the room
// RETIRING it would actually hand back: the bytes framed on it, less everything either
// relocating pass has already spent carrying records off it. The pass works on a page
// only while that figure is at least PageSize/relocateReserveMinGainDivisor.
//
// Read forwards it is a PROGRESS guarantee. Every retire reclaims at least a quarter of a
// page more than it cost, so the reserve is reached in a bounded number of rounds and the
// pass then goes quiet until writes consume the room again. It is rate-matched to the
// write stream by construction rather than by a timer.
//
// Read backwards it is a SPEND CAP, and that is what rules out the one failure mode that
// would make this layer worse than no layer at all: a page most of whose records are LIVE
// being copied somewhere else in its entirety, so that the husk left behind looks
// completely dead and is retired for no room reclaimed — every tick, for every page,
// forever. Under the rule such a page is abandoned the moment carrying any more off it
// would leave less than a quarter page to reclaim, the rotation cursor stays put, and the
// pass does nothing further until the write path drains it. Which is exactly what the
// synchronous pass does with the part of a page its own per-eviction budget does not
// reach. No regression either way, and a hard ceiling on what this layer can ever copy.
//
// WHY page.relocatedOut HAS TO BE REMEMBERED rather than derived: to a walk of the page,
// an entry that is dead because a later write superseded it and an entry that is dead
// because relocation moved it look identical — and here they mean opposite things. The
// first is room to reclaim; the second is room already spent. Nothing else in the shard
// distinguishes them, so the pass keeps the figure itself.
//
// ==========================================================================
// THIS PASS NEVER DROPS A LIVE RECORD, which is the other place its contract is
// stricter. The synchronous pass runs inside an eviction that is going to happen
// regardless, so a record it cannot fit is simply left for the drain — losing it is the
// policy's decision, already made. Nothing has decided anything when the SWEEPER runs:
// no write asked for room. A background pass that dropped what it could not move would
// make a full shard shed live records while idle, which today's code never does.
//
// So a page is retired only once every live record is off it, and a round that cannot
// finish leaves the page exactly as it found it — partially evacuated at worst, which is
// not a loss and not even wasted, since the records already moved are live copies
// elsewhere. Eviction — the loss of live data — stays on the write path, where the
// policy puts it.
//
// ==========================================================================
// LOCK DISCIPLINE — CHUNKED, NOT PER PAGE. reclaimExpiredHeapPages takes the write lock
// once PER PAGE; a page here holds many entries and each one is a copy plus an index
// repoint, so a whole page under one hold would stall every writer on the shard for the
// duration. This goes finer: a chunk ends after relocateChunkEntries entries moved,
// PageSize/relocateChunkBytesDivisor bytes moved, or relocateChunkScanEntries entries
// EXAMINED — that last one because a page of already-dead entries, or one the pass has
// already emptied, moves nothing at all and would otherwise run to the end of the page
// under a single hold. One entry may overshoot the byte figure: an entry is never split.
//
// NOTHING SURVIVES A RELEASED LOCK WITHOUT BEING RE-CHECKED. Across a release a page can
// be retired, replaced, refilled or re-generationed, and an index slot can be repointed
// by a Put. So each chunk re-acquires and re-derives everything: the page object
// identity (a retire swaps in a fresh one), the rotation cursor, the index table pointer
// (a resize may have swapped it), and the page's entries slice, head and tail. The only
// things carried across are a byte OFFSET into a page that is append-only and whose
// identity was re-confirmed, and the running totals — and every entry at that offset is
// re-decoded and re-tested for index-currency from scratch before it is moved, so a view
// that went stale while the lock was down can cost efficiency and never correctness.
//
// ==========================================================================
// BOUNDED PER TICK. The sweeper runs on a ticker (TTLSweepIntervalMs, default 1s) and
// must not monopolise a core or the shard lock. A tick may copy at most PageSize bytes
// across at most relocateReserveFreePages+1 rounds. Hitting the byte budget stops the
// pass; the page it was working on keeps whatever was already moved off it and the next
// tick starts again, with that much less to do.
//
// ==========================================================================
// HEAP RINGBUF ONLY, and the reason is crash consistency rather than mechanism. The
// mmap argument in cache/relocate_evict.go rests on relocation never bringing a deletion
// CLOSER: it moves a record one eviction before the drain that was going to delete it,
// and that drain is still driven by write pressure. This pass performs the drain ITSELF,
// on a clock, for a page no write asked to be freed. On mmap the drain's head advance
// and the relocated copy are both non-durable until the periodic msync and the OS may
// write them back in either order, so a crash could lose a record that today's code —
// which would not have drained that page at all — would have kept. That is strictly
// worse than today, which the invariant forbids. Closing it needs an msync between the
// copies and the drain, and msync here is O(region): a 256 MiB walk per evacuated page.
// So mmap ringbuf shards keep the synchronous pass alone, unchanged.

const (
	// relocateReserveFreePages is how much free room the sweeper tries to keep in hand
	// on a shard at its page cap, counted in whole pages: one for the write stream to
	// be filling, one ready for the moment it fills. No small constant can absorb an
	// arbitrary burst — that is what the write-path fallback is for — so the value is
	// not critical in either direction: too low and more writes take the synchronous
	// path, too high and the shard holds more capacity empty. It is a constant rather
	// than a Config field because Config.RelocatingEviction already gates the whole
	// feature and a second knob on it would be one more thing to get wrong.
	relocateReserveFreePages = 2

	// relocateReserveShardDivisor caps the reserve at len(pages)/divisor pages, so the
	// pass can never hold a meaningful fraction of a small shard empty. A shard with
	// fewer than this many pages keeps no reserve and relies entirely on the write path.
	relocateReserveShardDivisor = 4

	// relocateChunkEntries bounds the entries MOVED under one lock acquisition. It is
	// the per-entry work — a copy plus hash, findSlot and upsert — which is not
	// proportional to bytes, so it needs its own bound alongside the byte one.
	relocateChunkEntries = 64

	// relocateChunkBytesDivisor bounds the bytes copied under one lock acquisition at
	// PageSize/divisor. One entry may overshoot it, since an entry is never split.
	relocateChunkBytesDivisor = 16

	// relocateReserveMinGainDivisor sets the ONE rule this pass runs on: it will touch a
	// page only while tail - page.relocatedOut — the room retiring it would actually hand
	// back, net of everything either relocating pass has already copied off it — is at
	// least PageSize/divisor. Read one way it is a progress guarantee (every retire
	// reclaims more room than it cost, so the reserve is reached in a bounded number of
	// rounds); read the other way it is a spend cap (a page stops being carried once it
	// can no longer repay the carrying). Both readings are the same inequality, and
	// reserveMoveVictim evaluates it once. See the top of this file for what it rules out.
	relocateReserveMinGainDivisor = 4

	// relocateChunkScanEntries bounds the entries EXAMINED under one lock acquisition,
	// which is what keeps a page of already-dead entries — or one the pass has already
	// emptied — from pinning the lock for a whole page even though it moves nothing.
	// Examining an entry is a decode plus one index probe, roughly what sweepIndex does
	// per slot, so it is bounded like sweepBatchSize rather than like a copy.
	relocateChunkScanEntries = 512
)

// topUpFreeReserve is the sweeper's half of relocating eviction. Called from
// sweepOnce's non-replicated branch. It takes and releases s.mu itself, many times;
// must NOT be called with it held.
//
// A no-op unless the shard is an opted-in heap ringbuf shard at its page cap. Bounded
// at PageSize bytes copied per call, and at one more round than the reserve target — so
// the last round is the one that finds the reserve already met and PREPARES instead.
func (s *shard) topUpFreeReserve() {
	if !s.reserveRelocationEligible() {
		return
	}
	budget := s.cfg.PageSize
	for range relocateReserveFreePages + 1 {
		spent, freed := s.reserveEvacuateVictim(budget)
		budget -= spent
		if !freed || budget <= 0 {
			return
		}
	}
}

// reserveRelocationEligible reports whether the background reserve applies to this
// shard at all. Heap ringbuf with the feature on: mmap is excluded for the
// crash-consistency reason at the top of this file, and replicated shards never evict
// (replication forces PolicyRejectWrites) — they reclaim through the online compactor.
func (s *shard) reserveRelocationEligible() bool {
	return s.cfg.RelocatingEviction &&
		s.cfg.AtCapPolicy == PolicyRingbufEvict &&
		!s.isMmap &&
		!s.cfg.Replicated
}

// reserveTargetBytesLocked is the free room this shard should hold, capped at a
// fraction of its size. Zero on a shard too small to hold a reserve. Must hold mu.
func (s *shard) reserveTargetBytesLocked() int {
	pages := relocateReserveFreePages
	if maxPages := len(s.pages) / relocateReserveShardDivisor; pages > maxPages {
		pages = maxPages
	}
	return pages * s.maxEntryBytes()
}

// freeRoomLocked is the shard's total contiguous tail room. It is the quantity the
// write path actually consumes, and counting ROOM rather than empty PAGES is what
// makes a freed page still count for the part of it a relocation has landed in. Must
// hold mu.
func (s *shard) freeRoomLocked() int {
	room := 0
	for _, p := range s.pages {
		if p.retired {
			continue // stranded by online compaction; never true on this path
		}
		room += p.FreeTail()
	}
	return room
}

// reserveDestinationLocked picks where a relocated record goes: the eligible page the
// rotation cursor reaches LAST, so the copy gets the longest possible run before its
// new page is itself a victim. That mirrors the synchronous pass, where the just-freed
// page sorts last from the cursor by construction.
//
// NON-EMPTY pages are preferred over empty ones — filling the reserve to build the
// reserve is the least useful place to put a record — but an empty page is taken rather
// than giving up, because giving up would leave the victim half-evacuated for no reason.
// What keeps that from churning the reserve in circles is the caller's rule on
// tail - relocatedOut, not this preference: see reserveMoveVictim.
//
// Returns -1 when nothing has room. Must hold mu for writing.
func (s *shard) reserveDestinationLocked(need, victim int) int {
	if idx := s.reserveScanDestinationLocked(need, victim, true); idx >= 0 {
		return idx
	}
	return s.reserveScanDestinationLocked(need, victim, false)
}

// reserveScanDestinationLocked scans rotation order backwards from the cursor —
// latest-drained page first — for a page with `need` bytes of tail room, optionally
// requiring it to be non-empty. Must hold mu.
func (s *shard) reserveScanDestinationLocked(need, victim int, nonEmptyOnly bool) int {
	n := len(s.pages)
	for off := n - 1; off >= 0; off-- {
		i := (s.nextVictim + off) % n
		if i == victim {
			continue
		}
		p := s.pages[i]
		if p.retired || (nonEmptyOnly && p.Empty()) {
			continue
		}
		if p.FreeTail() >= need {
			return i
		}
	}
	return -1
}

// reserveValidateVictimLocked re-confirms, after a released lock, that the page in hand
// is still the page the next eviction takes: the same object (a retire swaps in a
// fresh one) at the same index, and still the rotation cursor's choice. Must hold mu.
func (s *shard) reserveValidateVictimLocked(victim int, p *page) bool {
	return victim < len(s.pages) && s.pages[victim] == p && s.nextNonEmptyPageLocked(-1) == victim
}

// reserveEvacuateVictim runs one round against the page the next eviction would drain.
//
// It moves every live record off that page and — when the shard is short of room and
// the retire rule is satisfied — retires it, handing the room back. Returns the bytes
// it copied and whether it freed a page.
//
// freed=false is the signal to stop for this tick, and it covers both the failures
// (not at the page cap, no victim, nothing fits, the budget ran out, the cursor moved)
// and the SUCCESS that ends a tick: the victim was evacuated and left in place —
// PREPARED, so that whatever drains it next, a write-path eviction or the next tick's
// first round, finds only dead versions.
func (s *shard) reserveEvacuateVictim(budget int) (int, bool) {
	s.mu.Lock()
	if len(s.pages) < s.cfg.MaxPagesPerShard() {
		// Below the page cap the shard still grows a page when it needs one, so there
		// is no eviction pressure to get ahead of.
		s.mu.Unlock()
		return 0, false
	}
	target := s.reserveTargetBytesLocked()
	if target == 0 {
		s.mu.Unlock()
		return 0, false
	}
	// Decided once per round, at the top: retire while the shard is short of room,
	// otherwise this round only prepares.
	retire := s.freeRoomLocked() < target
	victim := s.nextNonEmptyPageLocked(-1)
	if victim < 0 {
		s.mu.Unlock()
		return 0, false
	}
	p := s.pages[victim]
	s.mu.Unlock()
	return s.reserveMoveVictim(victim, p, budget, retire)
}

// reserveMoveVictim copies every live record off the victim and, when retire is set, the
// page came out clean and the retire rule holds, retires it. Chunked, with everything
// re-derived after each release; see the lock-discipline note at the top of this file for
// what may and may not be carried across one.
//
// ONE INEQUALITY governs it, checked before each chunk and again before each record is
// moved: tail - page.relocatedOut, the room retiring this page would actually hand back,
// must be at least minGain. Falling below it stops the round where it stands, so reaching
// the end of the page with stop unset is itself the proof that the retire below repays
// what was spent getting there. The full argument — why it is simultaneously the progress
// guarantee and the anti-churn cap, and why relocatedOut has to be remembered rather than
// derived — is at the top of this file.
func (s *shard) reserveMoveVictim(victim int, p *page, budget int, retire bool) (int, bool) {
	chunkByteBudget := s.cfg.PageSize / relocateChunkBytesDivisor
	minGain := s.maxEntryBytes() / relocateReserveMinGainDivisor
	cursor, spent := -1, 0
	for {
		chunkEntries, chunkBytes, scanned, stop := 0, 0, 0, false
		s.mu.Lock()
		if !s.reserveValidateVictimLocked(victim, p) {
			s.mu.Unlock()
			return spent, false
		}
		// Re-derived under THIS hold: a Put may have appended to the page, and a resize
		// may have swapped the index table, since the last one.
		entries := p.entries()
		tail := p.tail()
		if cursor < 0 {
			cursor = p.head()
		}
		// THE RULE (see the top of this file), evaluated before any work: once the page
		// can no longer be retired profitably there is nothing to be gained by carrying
		// anything further off it. Costs nothing — both terms are page fields — which is
		// why it is the guard rather than a costing walk.
		if tail-p.relocatedOut < minGain {
			s.mu.Unlock()
			return spent, false
		}
		t := s.tab.Load()
		// The eviction clock, the same wall clock retirePageLocked classifies against.
		// It decides only what is worth MOVING; what the shard retains is still decided
		// by the retire walk's cur == ref alone.
		now := s.now()
		for cursor < tail && scanned < relocateChunkScanEntries {
			key, value, expiryMs, err := decodeEntryFast(entries[cursor:tail])
			if err != nil {
				stop = true // unreachable on a heap page: no CRC, nothing external can corrupt it.
				break
			}
			size := entrySize(len(key), len(value))
			ref := makeSlabRef(uint16(victim), p.gen, uint32(cursor)) //nolint:gosec // victim bounded by MaxPagesPerShard (≤65535); cursor < PageSize ≤ MaxInt32
			h := hashKey(key)
			_, cur, ok := t.findSlot(h)
			scanned++
			if !ok || cur != ref {
				// A superseded copy, a hash-collision loser, or an already-tombstoned
				// slot: no read path can reach it, so there is nothing here to keep.
				cursor += size
				continue
			}
			if isExpired(expiryMs, now) {
				// The retire walk's to tombstone, exactly as today; carrying a corpse
				// forward would spend budget to keep nothing.
				cursor += size
				continue
			}
			if spent+size > budget {
				stop = true // tick budget spent.
				break
			}
			dstIdx := s.reserveDestinationLocked(size, victim)
			if dstIdx < 0 {
				// Nowhere to put it, so the page cannot be fully evacuated — and this
				// pass never drops what it could not move. Leave it to the write path.
				stop = true
				break
			}
			dst := s.pages[dstIdx]
			// The same writeSeq++ / makeMeta / Write / upsert sequence putAtExpLocked
			// performs, so the bytes are written before the ref addressing them is
			// published and a lock-free reader resolves either copy, never a tear.
			// key/value alias the SOURCE page, a different slab from dst, so the copy
			// can never overlap itself.
			newSeq := s.writeSeq + 1
			off, _, werr := dst.Write(key, value, expiryMs, makeMeta(newSeq, false))
			if werr != nil {
				// Unreachable: reserveDestinationLocked already proved the room. Stop
				// rather than risk a half-move.
				stop = true
				break
			}
			s.writeSeq = newSeq
			// upsert repoints the slot findSlot just matched, so the table's
			// live/tombstone counts are untouched and it can neither grow nor rehash.
			t.upsert(h, makeSlabRef(uint16(dstIdx), dst.gen, off)) //nolint:gosec // dstIdx bounded by MaxPagesPerShard (≤65535)
			cursor += size
			spent += size
			p.relocatedOut += size
			chunkEntries++
			chunkBytes += size
			// Bytes BEFORE the count, matching the synchronous pass: snapshot() loads
			// them in the opposite order, so a scrape landing between the two can only
			// see bytes running ahead of the count, never records relocated with
			// nothing moved.
			s.reserveRelocatedBytes.Add(uint64(size)) //nolint:gosec // entrySize is non-negative
			s.reserveRelocations.Add(1)
			if tail-p.relocatedOut < minGain {
				stop = true // spent all this page could repay: it is the write path's now.
				break
			}
			if chunkEntries >= relocateChunkEntries || chunkBytes >= chunkByteBudget {
				break // chunk bound: drop the lock and come back for the rest.
			}
		}
		freed := false
		if !stop && cursor >= tail && retire {
			// tail - relocatedOut >= minGain holds by construction: the loop above stops
			// the moment it does not, so reaching the end of the page with stop unset is
			// itself the proof that retiring this page repays what was spent on it.
			// Every live record is out and the page gives back room the pass has not
			// already paid for (see THE RETIRE RULE above). Advance the cursor past the
			// victim and retire it in the order evictUntilFitsLocked uses, so this is
			// that eviction in every respect but the dropping: the retire walk finds no
			// index-current live entry, so it tombstones nothing that moved, fires
			// onRemove for nothing that moved, and counts nothing that moved in
			// evictionsLive.
			s.nextVictim = (victim + 1) % len(s.pages)
			s.retirePageLocked(victim)
			s.reservePagesFreed.Add(1)
			// UNDER THE SAME HOLD, and that is the whole point of it being here rather
			// than in the next round: retiring moves the rotation cursor onto a page
			// nothing has evacuated, and a write-path eviction slipping into the gap
			// would drain that page with live records still on it — a loss neither
			// scheme takes on its own. So this round closes the gap the way
			// evictVictimLocked does, with the very same pass and the very same budget,
			// and the round that follows then evacuates whatever that budget left.
			// Charged to the sweeper: no write paid for it.
			s.noteReserveRelocation(s.relocateIntoFreedPageLocked(victim, 0))
			freed = true
		}
		done := stop || cursor >= tail
		s.mu.Unlock()
		s.noteReserveChunk(chunkEntries, chunkBytes)
		if done {
			return spent, freed
		}
	}
}

// noteReserveChunk reports a finished chunk to the observer, if one is installed. It
// runs with s.mu RELEASED — the point of the hook is to let a test act in the window
// the chunking opens. Nil unless a test installs one; one atomic load per chunk.
func (s *shard) noteReserveChunk(entries, bytes int) {
	if h := s.reserveChunkHook.Load(); h != nil {
		(*h)(entries, bytes)
	}
}
