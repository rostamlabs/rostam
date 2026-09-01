// SPDX-License-Identifier: Apache-2.0

package cache

// Online relocating compaction — reclaiming fragmented mmap page space WHILE the
// process runs, without disturbing the lock-free zero-copy read path.
//
// ==========================================================================
// THE PROBLEM (recap). A replicated shard is forced to PolicyRejectWrites +
// mmap (shard/store.go). It is append-only: an overwrite writes a NEW copy and
// repoints the index; the OLD copy stays framed on its page as a dead duplicate,
// and a TTL-expired entry's bytes likewise linger after the logical sweeper
// tombstones its index slot. These "ghost" bytes are never reclaimed while the
// process runs — a file-backed page wraps a FIXED region and cannot be
// frozen-swapped the way a heap page is (reclaimExpiredHeapPages early-returns for
// mmap), and rewriting a page in place would trample bytes a lock-free
// reject-writes reader still aliases (those readers take NO lock, so holding s.mu
// would not exclude them). So ghost bytes climb until MaxPagesPerShard → ErrFull →
// the Phase-A fail-closed halt, recoverable only by a restart (cold compaction).
//
// ==========================================================================
// THE READER-SAFETY INVARIANT (never violated). Reject-writes/mmap Get is
// lock-free (needsReadLockForGet is false) and returns a []byte that ALIASES mmap
// page bytes; that alias can escape unboundedly. So we may NEVER overwrite, reset,
// munmap, MAP_FIXED-remap, or punch any page extent's bytes that were ever
// reader-visible. Relocation therefore only ever APPENDS to fresh destination
// bytes and repoints the index atomically; source pages stay immutable.
//
// ==========================================================================
// WHY THIS IS SAFE — reduction to an ordinary Put overwrite. Relocating one live
// entry is MECHANICALLY IDENTICAL to re-Putting the same (key, value, exp) into a
// different page:
//
//	dest.Write(key, value, exp, meta)          // append into never-read tail bytes
//	t.upsert(h, makeSlabRef(dest, gen, off))   // single atomic index repoint
//
// This is exactly what putAtExpLocked does. A concurrent lock-free reader loads the
// page object from pageSlots and the ref from the index, then checks p.gen ==
// ref.gen() before reading bytes (indextable.get), with a bytes.Equal backstop. It
// therefore resolves EITHER the OLD source copy (still mapped, still immutable) OR
// the NEW destination copy (published only AFTER its bytes were fully written, via
// the release/acquire pairing of the atomic ref store/load) — never a torn read.
// The destination's tail bytes were never addressed by any ref (refs only point at
// offsets < tail), so no reader can observe them mid-write. In short: relocation
// inherits, verbatim, the reader-safety an overwrite Put already has. No new
// epoch/RCU primitive is needed and the read path pays exactly nothing.
//
// ==========================================================================
// DETERMINISM. Physical layout is node-local and already differs freely between
// replicas (different restart/snapshot histories); writeSeq never enters committed
// state or snapshots. Relocation only moves bytes around physically and stamps the
// relocated copy with a FRESH node-local writeSeq — it changes nothing in the
// committed key/value/exp set, so replicas stay logically identical no matter how
// differently (or whether) they each relocate. TTL-expiry decisions use
// compactDropClock() — the LOGICAL clock (lastAppliedStampMs) on a replicated
// shard, never wall time — the same clock cold compaction and the logical sweeper
// use, so a relocation pass can never drop a key a peer still considers live.
//
// ==========================================================================
// SCOPE — RELOCATE THEN RECYCLE. This relocates live entries into EXISTING destination
// tail space (another not-yet-full page, or a genuinely fresh sparse page), marks
// fully-evacuated source pages RETIRED, and then — once the alias-drain QUARANTINE has
// elapsed since retirement — RECYCLES a retired page (compactRecycleRetiredLocked): its
// extent is reset (head/tail→0) with a bumped generation and handed back to the write
// path, so a shard that hit ErrFull on accumulated dead versions recovers write capacity
// WHILE running, not only at the next restart (cold compaction). Recycle is the exact
// frozen-swap the heap ringbuf retire path uses — a fresh page object over the SAME mmap
// extent, published atomically into pageSlots — the mmap being safe only because the
// quarantine has drained every escaped reader alias first (see compactRecycleRetiredLocked).
//
// OPT-IN FOR READER-SAFETY, not because recycle is unfinished. Config.OnlineCompaction
// gates the whole relocate+recycle ACTION and defaults OFF. The reason is the escaped
// alias: a reject-writes/mmap Get returns a []byte aliasing page bytes, and recycle
// overwrites those bytes after the quarantine. That is memory-safe ONLY when every read
// is released within the quarantine — i.e. all reads flow through the server transport,
// whose WriteTimeout bounds the alias (AliasQuarantine = 2*WriteTimeout). An in-process
// embedder that retains a raw Store.Get/Node.Call alias past the fence would see it
// overwritten, so the feature stays operator opt-in (rostam.ServerConfig.
// EnableOnlineCompaction) rather than forced on. The reclaimable-byte accounting and the
// trigger (Stage 0) run regardless of the flag — reclaimable is always visible in Stats.

import "time"

const (
	// compactRelocateMaxPagesPerTick bounds how many source pages a single sweeper
	// tick actively relocates, so one tick never pins the write lock for an unbounded
	// number of per-page holds. Each page is relocated under its OWN s.mu acquisition,
	// released before the next page — mirroring reclaimExpiredHeapPages' discipline —
	// so the apply path (which takes the same s.mu) is never blocked for an
	// O(all-pages) span.
	compactRelocateMaxPagesPerTick = 4

	// defaultAliasQuarantine is the fallback alias-drain window used when
	// Config.AliasQuarantine is unset (0). It must exceed the maximum wall-clock
	// lifetime of a zero-copy read alias (see Config.AliasQuarantine): the TCP
	// server bounds a response write to WriteTimeout (default 30s), so 60s =
	// 2*WriteTimeout is a conservative fence with comfortable headroom for the
	// CPU-only pipeline drain that follows a deadline abort. shard/store.go threads
	// in an explicit 2*WriteTimeout; this default only applies if it does not.
	defaultAliasQuarantine = 60 * time.Second
)

// aliasQuarantine is the alias-drain window this shard enforces before recycling a
// retired page: the configured Config.AliasQuarantine, or the conservative built-in
// default when unset.
func (s *shard) aliasQuarantine() time.Duration {
	if s.cfg.AliasQuarantine > 0 {
		return s.cfg.AliasQuarantine
	}
	return defaultAliasQuarantine
}

// onlineCompactionEligible reports whether this shard is one the online relocating
// compactor operates on: an mmap-backed, cluster-REPLICATED, REJECT-WRITES shard —
// the exact mode shard/store.go forces a cluster-replication shard into. Every
// other shard (heap, single-node, ringbuf) answers false and is left byte-for-byte
// unchanged. Mode is immutable after construction (isMmap, cfg.*), so this is
// lock-free.
func (s *shard) onlineCompactionEligible() bool {
	return s.isMmap && s.cfg.Replicated && s.cfg.AtCapPolicy == PolicyRejectWrites
}

// liveAndUsedBytes returns (liveBytes, usedBytes) for the shard, judged at
// dropClock. usedBytes is the sum of every page's framed bytes (tail-head) —
// identical to bytesUsed, the ghost bytes included. liveBytes counts only entries
// that are INDEX-CURRENT (cur == ref) and NOT expired at dropClock (via
// entryIsLiveAtOpen, the same predicate cold compaction uses). reclaimable =
// used - live is therefore exactly the superseded + deleted + logically-expired
// byte total a compaction could reclaim.
//
// Taken under the READ lock: it only reads page bytes and the index, so it runs
// concurrently with other readers and is excluded only during a write. O(entries);
// callers are the once-per-tick sweeper trigger and Stats, neither latency-critical.
func (s *shard) liveAndUsedBytes(dropClock uint64) (live, used uint64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	t := s.tab.Load()
	for pageIdx, p := range s.pages {
		head := p.head()
		tail := p.tail()
		used += uint64(tail - head) //nolint:gosec // tail >= head always
		entries := p.entries()
		for cursor := head; cursor < tail; {
			key, value, exp, err := decodeEntryFast(entries[cursor:tail])
			if err != nil {
				// A crash-torn tail on a file-backed page: stop this page's walk, exactly
				// like walkLiveAtOpen. The bytes past it are unreadable and count as neither
				// live nor (addressable) used.
				break
			}
			meta := entryMetaAt(entries[cursor:tail])
			size := entrySize(len(key), len(value))
			if s.entryIsLiveAtOpen(t, pageIdx, p.gen, cursor, key, exp, meta, dropClock) {
				live += uint64(size) //nolint:gosec // entrySize is non-negative
			}
			cursor += size
		}
	}
	return live, used
}

// pageLiveUsedLocked returns (liveBytes, usedBytes) for a SINGLE page idx, judged at
// dropClock, using the same liveness predicate as liveAndUsedBytes: used is the page's
// framed bytes (tail-head), live counts only index-current, not-expired-at-dropClock
// entries. reclaimable = used - live is the page's dead (superseded / deleted /
// logically-expired) share. O(entries-on-this-page). Must hold s.mu; used to decide
// whether a page is fragmented enough to be worth relocating (see tryRelocatePageLocked).
func (s *shard) pageLiveUsedLocked(idx int, t *indexTable, dropClock uint64) (live, used uint64) {
	p := s.pages[idx]
	head := p.head()
	tail := p.tail()
	used = uint64(tail - head) //nolint:gosec // tail >= head always
	entries := p.entries()
	for cursor := head; cursor < tail; {
		key, value, exp, err := decodeEntryFast(entries[cursor:tail])
		if err != nil {
			break // crash-torn tail: stop the walk (mirrors liveAndUsedBytes / walkLiveAtOpen).
		}
		meta := entryMetaAt(entries[cursor:tail])
		size := entrySize(len(key), len(value))
		if s.entryIsLiveAtOpen(t, idx, p.gen, cursor, key, exp, meta, dropClock) {
			live += uint64(size) //nolint:gosec // entrySize is non-negative
		}
		cursor += size
	}
	return live, used
}

// reclaimableBytesForStats is the Stats view of ghost-byte pressure: the page bytes
// that are not index-current-and-live at the shard's logical clock. Computed on
// demand for eligible shards only (the walk is skipped — and 0 returned — for every
// other shard, so Stats stays cheap for them and behaviorally unchanged).
func (s *shard) reclaimableBytesForStats() uint64 {
	if !s.onlineCompactionEligible() {
		return 0
	}
	live, used := s.liveAndUsedBytes(s.compactDropClock())
	if used <= live {
		return 0
	}
	return used - live
}

// maybeRelocateCompact is the sweeper hook. STAGE 0: it records the reclaimable-byte
// figure (already exposed via Stats) and evaluates the relocation TRIGGER. STAGE 1:
// when Config.OnlineCompaction is set and the trigger fires, it runs one budgeted,
// reader-safe relocation pass. Called from sweepOnce's replicated branch, AFTER
// sweepIndex has tombstoned logically-expired slots (so relocation never moves an
// expired entry) and reclaimExpiredHeapPages (a no-op for mmap).
//
// It runs even before apply-stamping advances the logical clock. dropClock is
// compactDropClock() — the logical clock (lastAppliedStampMs), 0 on a not-yet-stamped
// replicated shard. At 0, isExpired(exp, 0) is false for every entry, so NO key is
// dropped by TTL; but SUPERSEDED / index-dead versions (the actual dead-space problem)
// are still reclaimed and retired extents still recycled — so an opted-in shard is NOT
// inert until EnableApplyStamp lands. Once stamping is on, TTL-expired entries reclaim
// too. Using the logical clock (never wall time) is what keeps a replicated relocation
// pass from dropping a key a peer still considers live.
func (s *shard) maybeRelocateCompact() {
	if !s.onlineCompactionEligible() {
		return // gate: heap / single-node / ringbuf shards are entirely unaffected.
	}
	dropClock := s.compactDropClock() // logical clock; 0 (⇒ no TTL drops) until stamping is on.
	capacity := s.pageCapacityBytes()
	if capacity == 0 {
		return
	}
	live, used := s.liveAndUsedBytes(dropClock)
	reclaimable := uint64(0)
	if used > live {
		reclaimable = used - live
	}
	deadRatio := float64(reclaimable) / float64(capacity)
	occupancy := float64(used) / float64(capacity)
	// TRIGGER: relocate only when there is enough dead space to be worth the work
	// (dead-ratio ≥ min-reclaim) OR the shard is running hot (occupancy ≥ warn-high) —
	// the same two marks the restart-time cold compactor and the occupancy alert use.
	if deadRatio < mmapCompactMinReclaimRatio && occupancy < mmapOccupancyWarnHigh {
		return
	}
	if !s.cfg.OnlineCompaction {
		// Stage 0 only: the trigger fired and the reclaimable figure is live in Stats,
		// but the relocation ACTION is opt-in (see Config.OnlineCompaction — without a
		// retired-extent recycle stage, relocation alone only consumes fresh tail).
		return
	}
	s.compactRelocateOnce(dropClock)
}

// compactRelocateOnce runs one budgeted relocation pass over the shard's pages. Each
// page is evacuated under its OWN s.mu acquisition, released before the next — the
// exact lock discipline reclaimExpiredHeapPages uses so the apply path is never
// pinned across an O(all-pages) span. len(s.pages) and s.pages[idx] are re-read under
// the lock every iteration, so a Put that appended a page or wrote a fresh entry
// during a released window is always observed by that page's next acquisition;
// correctness holds across the release because each page's decision AND its
// relocation happen atomically within one hold. The per-tick budget bounds how many
// pages we actively relocate (relocation is heavier than a decode-only retire).
func (s *shard) compactRelocateOnce(dropClock uint64) {
	// First RECYCLE any retired pages whose alias-drain quarantine has fully elapsed,
	// returning their extents to writable space — so a freed page is available as a
	// relocation destination in this very pass, and (the whole point) so a shard that
	// hit ErrFull on dead versions regains write capacity. Recycling is O(pages) cheap
	// pointer swaps (no byte copy), so unlike relocation it runs under one lock hold.
	s.mu.Lock()
	s.compactRecycleRetiredLocked(time.Now(), s.aliasQuarantine())
	s.mu.Unlock()

	processed := 0
	for idx := 0; ; idx++ {
		s.mu.Lock()
		if idx >= len(s.pages) || processed >= compactRelocateMaxPagesPerTick {
			s.mu.Unlock()
			return
		}
		if s.tryRelocatePageLocked(idx, dropClock) {
			processed++
		}
		s.mu.Unlock()
	}
}

// compactRecycleRetiredLocked reclaims every retired page whose alias-drain
// quarantine (Config.AliasQuarantine) has elapsed since it was retired, returning
// its extent to writable space. Returns how many pages it recycled. Must hold s.mu
// for writing.
//
// Recycle is the RECOVERY half of online compaction: relocation (Stage 1) evacuates
// live entries and RETIRES the emptied source page, but a retired page's extent stays
// stranded — used bytes that are neither live nor writable — so relocation ALONE only
// consumes fresh tail. Recycling hands those extents back, so a write that would
// previously fail ErrFull from accumulated dead versions can succeed again.
//
// READER-SAFETY (the drain fence, made concrete). A recycled page's extent is reset
// (head/tail→0) and then overwritten by future writes. That is safe ONLY because,
// once AliasQuarantine has elapsed since retirement, no reader can still alias the
// page's OLD bytes: the sole consumer that holds a raw zero-copy alias across a
// blocking network flush is the INLINE (non-pipelined) TCP response writer, bounded by
// server.Config.WriteTimeout (AliasQuarantine is set to a conservative multiple of it,
// derived from the effective WriteTimeout so the fence tracks the real deadline). The
// pipelined TCP writer, which could otherwise queue an alias for many WriteTimeouts,
// now COPIES the payload at dispatch (server.copyPipelinePayload), so it holds no
// alias at all; every other consumer — HTTP handler, WASM op, snapshot/PB read,
// read-modify-write op — copies the value synchronously before any wait. A reader that
// resolved a slabRef into this page BEFORE retirement acquired that ref before
// retiredAt (after retirement the index no longer points here — relocation repointed
// every live slot), so its bounded read has long completed.
//
// THE QUARANTINE IS THE ONLY LINE OF DEFENCE FOR AN ESCAPED ALIAS. The gen bump does
// NOT back it up: newMmapPage(p.data) reuses the SAME mmap extent, so after Reset()
// and subsequent writes the OLD page object's bytes change too, and a []byte alias a
// reader ALREADY returned from indextable.get is never gen-rechecked — nothing revisits
// p.gen once the slice header has escaped. The gen gate protects ONLY a reader still
// INSIDE indextable.get that RE-RESOLVES pageSlots and compares p.gen == ref.gen()
// (a few-ns window): such a reader sees the fresh object's new gen, its stale ref
// misses, and it never dereferences recycled bytes. So the gen gate closes the
// resolve-time race; it gives an already-escaped alias nothing. The consequence is a
// hard rule: AliasQuarantine must NEVER be weakened on the belief the gen gate is a
// second fence for a lingering alias — it is not, and shard.New enforces the
// >= 2*WriteTimeout floor fail-closed for exactly this reason.
//
// The mechanism is the exact frozen-swap the heap ringbuf retire path uses
// (retirePageLocked / reclaimExpiredHeapPages): a FRESH page object over the SAME mmap
// extent, reset with a bumped gen, published atomically into pageSlots. A reader that
// resolves pages through pageSlots + the generation gate observes EITHER the old object
// (old gen — but no such in-flight resolver exists past the fence) OR the fresh object
// (new gen ⇒ its stale ref misses at resolve time); it never dereferences torn bytes.
// Mutating p.gen in place would instead race the lock-free reader's unsynchronized gen
// load — which is why a fresh object is swapped rather than the retired one edited.
func (s *shard) compactRecycleRetiredLocked(now time.Time, quarantine time.Duration) int {
	recycled := 0
	for idx := range s.pages {
		p := s.pages[idx]
		if !p.retired {
			continue
		}
		if now.Sub(p.retiredAt) < quarantine {
			continue // still within the alias-drain window; a reader may alias its bytes.
		}
		fresh := newMmapPage(p.data) // same mmap extent; retired=false, retiredAt zero.
		fresh.Reset()                // head/tail → 0 in the persisted header: full FreeTail restored.
		fresh.gen = s.nextGen()      // bump generation so any stale ref into the old content misses.
		s.pages[idx] = fresh
		s.pageSlots[idx].Store(fresh) // publish atomically for the lock-free read path.
		s.relocatePagesRecycled.Add(1)
		recycled++
	}
	return recycled
}

// tryRelocatePageLocked evacuates the live entries of page idx into other pages'
// fresh tail space and, if the page ends with NO index-current entry, marks it
// retired. Returns true if it did meaningful work (relocated an entry or retired the
// page) so the caller can budget. Must hold s.mu for writing; the whole decision and
// the relocations are one atomic unit under this single hold.
//
// The relocation of each entry is the reader-safe Put-equivalent described in the
// file header: append into a destination's never-read tail bytes, then repoint the
// index. Source bytes are never touched.
func (s *shard) tryRelocatePageLocked(idx int, dropClock uint64) bool {
	if idx == s.writeIdx {
		// Never relocate the active write page: fresh Puts land in its tail, and
		// retiring it would strand that tail (and break findOrMakePageLocked's fast
		// path, which writes into s.writeIdx without the retired check).
		return false
	}
	p := s.pages[idx]
	if p.retired || p.Empty() {
		return false
	}
	t := s.tab.Load()
	// SELECT only pages worth compacting. The shard-level trigger can fire on occupancy
	// alone (occupancy >= warn-high with reclaimable ~0), which would otherwise make this
	// pass relocate pages that are essentially ALL LIVE — pure copy/churn that consumes
	// fresh tail without retiring anything. Skip a page whose reclaimable (superseded /
	// deleted / logically-expired) share of its framed bytes is below the same
	// mmapCompactMinReclaimRatio the cold compactor uses, so a nearly-all-live page is
	// left alone and the budget is spent on genuinely fragmented pages. The pre-scan is
	// O(page) — far cheaper than the relocation writes it avoids on a skipped page.
	live, used := s.pageLiveUsedLocked(idx, t, dropClock)
	if used == 0 || float64(used-live)/float64(used) < mmapCompactMinReclaimRatio {
		return false
	}
	entries := p.entries()
	tail := p.tail()
	// pinned: an index-current entry could NOT be evacuated this pass — either it is
	// logically expired (the logical sweeper's slot to tombstone, not ours to move) or
	// no destination had room. A pinned page still has live index refs, so it MUST NOT
	// be retired. relocated/retire together decide the return value.
	pinned := false
	relocated := 0
	for cursor := p.head(); cursor < tail; {
		key, value, exp, err := decodeEntryFast(entries[cursor:tail])
		if err != nil {
			// A crash-torn tail on a file-backed page. Stop the walk (mirrors
			// walkLiveAtOpen) and pin the page: we never retire a page we could not fully
			// parse, so bytes past the tear are never treated as reclaimed.
			pinned = true
			break
		}
		size := entrySize(len(key), len(value))
		ref := makeSlabRef(uint16(idx), p.gen, uint32(cursor)) //nolint:gosec // idx ≤ MaxPagesPerShard; cursor < PageSize
		h := hashKey(key)
		_, cur, ok := t.findSlot(h)
		if !ok || cur != ref {
			// Superseded / index-dead duplicate (a later write repointed the slot, a
			// hash-collision loser, or a tombstoned slot): unreachable by every read path,
			// so it simply vanishes when this extent is later recycled. Nothing to move.
			cursor += size
			continue
		}
		// Index-current. An entry expired at the LOGICAL clock is the logical sweeper's
		// to tombstone (sweepIndex ran first this tick); we do not relocate it, and it
		// pins the page until that slot is gone. Using dropClock (not wall time) is what
		// keeps a replicated relocation pass from dropping a key a peer still holds.
		if isExpired(exp, dropClock) {
			pinned = true
			cursor += size
			continue
		}
		dest := s.findRelocDestLocked(size, idx)
		if dest < 0 {
			pinned = true // nowhere to evacuate to; the page stays live and un-retired.
			cursor += size
			continue
		}
		meta := entryMetaAt(entries[cursor:tail])
		// Append the live value into the destination's fresh tail and repoint the index —
		// the reader-safe Put-equivalent. A FRESH writeSeq keeps this copy strictly newer
		// than its stranded source original, so a warm-restart rebuild resolves the key to
		// the relocated copy (max-seq wins). The tombstone bit passes through verbatim,
		// though it is always clear here (an index-current entry is never a delete record).
		newSeq := s.writeSeq + 1
		off, _, werr := s.pages[dest].Write(key, value, exp, makeMeta(newSeq, metaIsTombstone(meta)))
		if werr != nil {
			// Unreachable: findRelocDestLocked already proved FreeTail >= size. Pin
			// defensively rather than risk a half-move that miscounts liveness.
			pinned = true
			cursor += size
			continue
		}
		s.writeSeq = newSeq
		t.upsert(h, makeSlabRef(uint16(dest), s.pages[dest].gen, off)) //nolint:gosec // dest ≤ MaxPagesPerShard
		s.relocations.Add(1)
		s.relocatedBytes.Add(uint64(size)) //nolint:gosec // entrySize is non-negative
		relocated++
		cursor += size
	}
	if pinned {
		return relocated > 0
	}
	// Every index-current entry was evacuated (or the page held only dead duplicates):
	// no index slot addresses this page any more. Mark it RETIRED so the write path
	// stops handing out its stale-framed tail. Retiring does NOT touch the bytes — they
	// stay mapped and immutable so any in-flight reader alias stays valid — and starts
	// the alias-drain quarantine; only compactRecycleRetiredLocked, after the quarantine
	// elapses, resets the extent in place and hands it back to the write path.
	p.retired = true
	p.retiredAt = time.Now() // start the alias-drain quarantine clock (WALL time; see page.retiredAt).
	s.relocatePagesGone.Add(1)
	return true
}

// findRelocDestLocked returns the index of a page (never the source, never a retired
// one) with at least `need` bytes of contiguous tail room, or -1 if none. A fresh /
// never-written page (FreeTail == full usable capacity) is a valid destination — its
// tail bytes were never reader-visible. Must hold s.mu.
func (s *shard) findRelocDestLocked(need, srcIdx int) int {
	for i := range s.pages {
		if i == srcIdx {
			continue // never relocate into the page we are evacuating.
		}
		if s.pages[i].retired {
			continue // stranded extent awaiting a recycle stage; not writable.
		}
		if s.pages[i].FreeTail() >= need {
			return i
		}
	}
	return -1
}
