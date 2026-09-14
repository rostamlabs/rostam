// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"log/slog"
	"os"
	"time"
)

// Cold compaction — reclaiming mmap page BYTES at shard open.
//
// THE PROBLEM. An mmap-backed shard reclaims expired/superseded INDEX SLOTS
// (the B3b logical sweeper) but never the page BYTES behind them: a file-backed
// page wraps a fixed region and cannot be frozen-swapped the way a heap page is
// (see reclaimExpiredHeapPages' early return), and rewriting it in place would
// trample bytes a lock-free reject-writes reader still aliases — those readers
// take no lock, so holding mu would not exclude them. Ghost bytes therefore
// climb monotonically under TTL churn until the shard hits MaxPagesPerShard →
// ErrFull → the Phase-A fail-closed halt.
//
// THE FIX. Do the rewrite at the ONE moment when the "concurrent reader" problem
// provably does not exist: during newShard, before the shard is published to any
// other goroutine. There is no reader to invalidate, so no epoch/RCU
// reader-reclamation primitive is needed and the read path pays exactly nothing.
// The cliff becomes RECOVERABLE BY RESTART instead of merely observable.
//
// The trade is that reclamation is no longer continuous: a long-lived process
// still accumulates ghost bytes between restarts, and the occupancy alert
// (checkMmapOccupancy) is what tells an operator a restart would now help.

const (
	// compactTmpSuffix names the staging file the compaction writes before its
	// atomic rename. It lives beside pages.dat in the same directory so the
	// rename is same-filesystem (hence atomic).
	compactTmpSuffix = ".compact"

	// mmapCompactMinOccupancy is the page-byte occupancy at or above which the
	// pages file is compacted at open. Deliberately the same mark at which the
	// runtime occupancy alert RE-ARMS (mmapOccupancyWarnLow): the alert warns an
	// operator that ghost bytes are climbing, and compacting from this mark makes
	// "restart the node" the concrete remedy that warning implies — a restart
	// lands the shard back below the alert band rather than just under it.
	mmapCompactMinOccupancy = mmapOccupancyWarnLow

	// mmapCompactMinReclaimRatio is the minimum share of total page capacity that
	// must actually be reclaimable before the file is rewritten. Without it a
	// shard legitimately full of LIVE data would rewrite its entire pages file on
	// every restart for no gain.
	mmapCompactMinReclaimRatio = 0.05
)

func compactTmpPath(pagesPath string) string { return pagesPath + compactTmpSuffix }

// discardCompactTemp removes a leftover compaction staging file.
//
// CRASH SAFETY. The staging file is published by exactly one step — the atomic
// rename over pages.dat — so its presence at open proves the previous
// compaction died BEFORE that rename, which in turn proves pages.dat is still
// the intact original. The partial staging file has no claim to any data and is
// discarded unconditionally. After a successful rename no staging path exists,
// so this can never delete a completed compaction.
func discardCompactTemp(pagesPath string) error {
	tmp := compactTmpPath(pagesPath)
	if err := os.Remove(tmp); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("cache: remove stale compaction temp %s: %w", tmp, err)
	}
	return nil
}

// compactDropClock returns the clock cold compaction judges TTL expiry against.
//
// ==========================================================================
// THE DETERMINISM ARGUMENT — why this clock, and why never time.Now on a
// replicated shard. This is the crux of the whole feature.
// ==========================================================================
//
// Compaction drops two disjoint classes of entry:
//
//	(a) NOT INDEX-CURRENT — a superseded older copy of a key (a later write
//	    repointed the slot elsewhere), a hash-collision loser, or a DELETED key.
//	    Deletes land here for free now that they are persisted as tombstone
//	    entries: the rebuild that just ran gave the key's slot to the tombstone on
//	    sequence and then stripped that slot, so nothing on the page addresses the
//	    tombstone OR any older copy of the key, and one compaction pass reclaims
//	    the key's entire byte history. (entryIsLiveAtOpen also rejects a tombstone
//	    explicitly, as defense in depth.) Before the v4 format this class excluded
//	    deletes entirely — Del recorded nothing on the page, so the rebuild
//	    resurrected the key and compaction dutifully preserved it (#12B).
//
//	    Dropping this class is unconditionally safe on ANY shard: an entry the
//	    index does not resolve to is unreachable by every read path on this node
//	    (Get, GetInto and the stamped apply-path GetAt all reach page bytes only
//	    through a slabRef the index handed them). Deleting bytes nothing can
//	    address cannot change any read's answer, so it cannot change this node's
//	    logical key set — and the logical key set is the only thing replicas must
//	    agree on. Physical layout is node-local and already differs freely
//	    between replicas (different restart histories, different snapshot
//	    installs).
//
//	(b) TTL-EXPIRED at this clock. THIS is where the wall clock is forbidden on a
//	    replicated shard. B1/B3a made every committed-state expiry decision read
//	    the LEADER-STAMPED clock precisely so replicas never disagree about what
//	    is live. If replica A (restarting late) physically dropped an entry that
//	    replica B still considers live under the stamped clock, a later committed
//	    write's GetAt(stamp) would return absent on A and present on B — silent
//	    divergence, the exact failure rebuildIndexFromPages already refuses to
//	    cause by never dropping on wall clock when Replicated.
//
// So a replicated shard judges (b) against lastAppliedStampMs — the logical
// clock, restored from the header at open (see cache/file.go). Safety, in full:
//
//  1. The restored value is never AHEAD of committed reality. lastAppliedStampMs
//     only ever advances by folding in the stamp of a committed, applied entry
//     (advanceAppliedStamp, called from getAtH/putAtH), and SetAppliedIndex
//     persists whatever value it held. So
//     persistedStamp <= max{stamp(e) : e a committed entry}.
//
//  2. Every FUTURE committed write judges those entries expired too. The
//     leader/primary clamps each new stamp to >= cache.LastAppliedStampMs()
//     (shard/store.go applyOpIndexed), and a leader has applied every committed
//     entry, so its own clock already equals max{stamp(e)} — which by (1)
//     dominates any replica's restored stamp. Hence every later committed write
//     carries stamp' >= restoredStamp >= exp of everything compaction dropped,
//     and GetAt(stamp') would have judged those entries expired anyway. Dropping
//     them early is invisible.
//
//     "A leader has applied every committed entry, so its clock equals
//     max{stamp(e)}" needs one more link, because the clock advances ONLY on the
//     stamped apply path (advanceAppliedStamp from getAtH/putAtH) — a snapshot
//     install runs through PutAbs, which deliberately does not advance it. A node
//     with a FRESH DataDir that acquires all its committed state from a snapshot
//     and has an EMPTY committed tail to replay would therefore hold clock 0; if
//     it were elected leader its clamp input would be 0 and it would stamp at
//     bare wall time, which may sit BELOW a peer's persisted stamp — and that
//     peer has already compacted entries away on the strength of this argument.
//     That gap is closed at the source: the snapshot CARRIES the logical clock
//     (shard/snapshot.go v4 trailer) and restoreSnapshot folds it into the
//     restoring node's clock (Cache.AdvanceAppliedStamp), so a snapshot-restored
//     node starts at >= the snapshotting node's clock rather than at 0. The one
//     residual case is a PRE-v4 snapshot (written by an older build), which
//     carries no clock and restores 0; the same exposure applies to the B3b
//     logical sweeper, which reclaims against the identical clock.
//
//  3. Replicas that restore DIFFERENT persisted stamps still converge. Nothing
//     requires two replicas to compact identically — (2) shows each replica's
//     drop set is a subset of "expired under any future committed read", so the
//     sets need not match, only be individually invisible. Each replica then
//     replays its committed log tail, which re-advances every clock to the same
//     value.
//
//  4. A header with no readable clock (a torn stamp write) restores 0, and
//     isExpired(e, 0) is false for every e — so such a file compacts
//     SUPERSEDED-ONLY. Under-restoring the clock costs reclamation, never
//     correctness.
//
// A NON-REPLICATED shard has no peer to diverge from, so it drops wall-clock
// expired entries freely — matching rebuildIndexFromPages, which already refuses
// to index them (they would be class (a) here regardless).
func (s *shard) compactDropClock() uint64 {
	if s.cfg.Replicated {
		return s.lastAppliedStampMs.Load()
	}
	return s.now()
}

// entryIsLiveAtOpen decides whether the physical entry at (pageIdx, gen, cursor)
// survives compaction. See compactDropClock for the safety argument behind both
// halves of the test.
//
// The identity check is the same `cur == ref` guard every other reclamation site
// uses (drainPageLocked, retirePageLocked, dropExpiredLocked): the index slot for
// this key's hash must point at exactly THIS physical copy. A slot pointing
// elsewhere means a later write superseded this copy; no slot at all means
// nothing can address it.
func (s *shard) entryIsLiveAtOpen(t *indexTable, pageIdx int, gen uint16, cursor int, key []byte, exp, meta, dropClock uint64) bool {
	if metaIsTombstone(meta) {
		// A delete record. Defense in depth: the rebuild that just ran already
		// stripped the slot a winning tombstone claimed, so a tombstone is class (a)
		// — index-dead — and would be dropped by the identity check below anyway.
		// Saying so explicitly means a tombstone can never survive a compaction even
		// if some future rebuild leaves its slot in place.
		return false
	}
	ref := makeSlabRef(uint16(pageIdx), gen, uint32(cursor)) //nolint:gosec // pageIdx < MaxPagesPerShard (≤65535); cursor < PageSize
	_, cur, ok := t.findSlot(hashKey(key))
	if !ok || cur != ref {
		return false // (a) superseded or index-dead — unreachable, always droppable
	}
	return !isExpired(exp, dropClock) // (b) expired at the mode-appropriate clock
}

// walkLiveAtOpen visits every LIVE framed entry, in page order then offset order
// — the SAME order rebuildIndexFromPages walks, so the visit sequence is exactly
// the sequence that decided which physical copy of each key the index resolves
// to. visit returns false to abort the walk. Construction-time only (no lock):
// the shard is not yet shared.
//
// A framing error stops that page's walk without touching the rest, mirroring
// rebuildIndexFromPages, which already truncated any torn tail before this runs
// — so in practice every byte in [head, tail) decodes.
// The visit callback receives the entry's META word so the rewrite can carry it
// through VERBATIM. That is an invariant, not a convenience: compaction must never
// alter a surviving entry's write sequence, or the next warm restart would resolve
// keys against sequences compaction invented rather than the ones the writes
// actually had.
func (s *shard) walkLiveAtOpen(dropClock uint64, visit func(key, value []byte, exp, meta uint64, size int) bool) {
	t := s.tab.Load()
	for pageIdx, p := range s.pages {
		tail := p.tail()
		entries := p.entries()
		for cursor := p.head(); cursor < tail; {
			key, value, exp, err := decodeEntryFast(entries[cursor:tail])
			if err != nil {
				break
			}
			meta := entryMetaAt(entries[cursor:tail])
			// EXACT, and NOT HEAP-REACHABLE. Cold compaction is the mmap pages file's
			// open-time rewrite (see the header of this file); a heap shard never gets
			// here. The figure is both the cursor advance over already-encoded bytes
			// and the size handed to the visit callback, which the callers sum into
			// the staging file's byte frontier. Both are encoder lengths on a
			// persisted page. (The frontier has to match what packLiveInto's
			// page.Write actually consumes; that is the encode today, and if a
			// persisted append ever reserved more than it encoded, the frontier would
			// have to follow the reservation — another reason mmap framing must not
			// acquire padding without this walk being revisited.)
			size := entrySpanExact(len(key), len(value))
			if s.entryIsLiveAtOpen(t, pageIdx, p.gen, cursor, key, exp, meta, dropClock) {
				if !visit(key, value, exp, meta, size) {
					return
				}
			}
			cursor += size
		}
	}
}

// liveBytesAtOpen sums the on-disk size of every entry that would survive.
func (s *shard) liveBytesAtOpen(dropClock uint64) uint64 {
	var live uint64
	s.walkLiveAtOpen(dropClock, func(_, _ []byte, _, _ uint64, size int) bool {
		live += uint64(size) //nolint:gosec // entrySpanExact is non-negative
		return true
	})
	return live
}

// packPagesNeeded reports how many destination pages packLiveInto will fill,
// WITHOUT writing anything. It mirrors packLiveInto's next-fit rule exactly
// (page.Write succeeds iff the entry fits the page's remaining tail), so the
// answer is the true frontier, not an estimate.
//
// Two callers need it before a single byte is written: the staging file's
// RESERVATION (only the pages we will actually touch may be preallocated — see
// mmapFileAlloc — or a full-disk write faults into an unrecoverable SIGBUS), and
// the up-front does-it-fit check.
func (s *shard) packPagesNeeded(dropClock uint64) int {
	usable := s.cfg.PageSize - pageHdrSize
	pages, used := 1, 0
	s.walkLiveAtOpen(dropClock, func(_, _ []byte, _, _ uint64, size int) bool {
		if used+size > usable {
			pages++
			used = 0
		}
		used += size
		return true
	})
	return pages
}

// packLiveInto writes every live entry densely into the destination region,
// filling page 0 first and moving on only when an entry does not fit the current
// page's tail. Entries are re-encoded through the normal page writer, so each
// one lands with a FRESH per-entry CRC and the destination pages carry correct
// persisted head/tail offsets — the compacted file is byte-for-byte the same
// format any other pages.dat is.
//
// Returns the PACK FRONTIER — the index of the page the last entry landed in,
// which is where the shard must resume writing (see compactAtOpen's writeIdx
// contract) — and false if the live set did not fit in MaxPagesPerShard pages,
// which makes the caller abandon the compaction and keep the original file.
// Repacking a strictly smaller set of same-ordered entries usually needs no more
// pages than the source used, but next-fit is not optimal and a pathological
// size mix can leave a bigger tail gap per page than the source did, so
// overflowing IS reachable in principle; failing that way is safe (abandon and
// keep the original), which is why the check is a plain guard rather than a
// proof obligation.
//
// Destination pages are materialized LAZILY, one at a time. That is deliberate:
// touching a page dirties it and forces the filesystem to allocate its first
// block, so eagerly Reset()-ing all MaxPagesPerShard pages would write into the
// sparse region beyond the reservation packPagesNeeded sized.
func (s *shard) packLiveInto(dst []byte, dropClock uint64) (int, bool) {
	maxPages := s.cfg.MaxPagesPerShard()
	dstPage := func(i int) *page {
		off := headerSize + i*s.cfg.PageSize
		p := newMmapPage(dst[off : off+s.cfg.PageSize])
		p.Reset() // the file is fresh (zero-filled); be explicit anyway
		return p
	}
	di, fits := 0, true
	cur := dstPage(0)
	s.walkLiveAtOpen(dropClock, func(key, value []byte, exp, meta uint64, _ int) bool {
		for {
			// meta passes through UNCHANGED — see walkLiveAtOpen.
			if _, _, err := cur.Write(key, value, exp, meta); err == nil {
				return true
			}
			di++
			if di >= maxPages {
				fits = false
				return false
			}
			cur = dstPage(di)
		}
	})
	return di, fits
}

// compactAtOpen rewrites the shard's pages file with only its live entries and
// maps the compacted result, when occupancy makes that worthwhile. Called from
// newShard AFTER rebuildIndexFromPages (the rebuilt index is what defines
// liveness) and BEFORE the shard is published, so no reader can observe the
// swap. A no-op for a shard below the occupancy/reclaim thresholds.
//
// CRASH-SAFETY SEQUENCE. Every step is ordered so that a crash at ANY point
// leaves either the intact original or the complete compacted file — never a
// torn shard:
//
//  1. write the live set into pages.dat.compact (a NEW file; pages.dat is not
//     touched, and is still the sole authority);
//  2. msync the staging region + fsync the staging file, so its bytes AND its
//     metadata are durable before anything points at it;
//  3. unmap the original (we hold no mapping of a file about to be replaced);
//  4. rename(pages.dat.compact → pages.dat) — atomic on POSIX: any observer, at
//     any instant, sees one complete file at that path;
//  5. fsync the directory so the rename itself is durable;
//  6. map pages.dat and re-validate it exactly like any other open.
//
// A crash before (4) leaves the original plus a stale staging file, which the
// NEXT open deletes (discardCompactTemp) before touching anything. A crash
// during (4) resolves atomically to old or new. A crash between (4) and (5) may
// lose the rename, which again yields the intact original. The failure modes
// are: everything before (3) abandons the compaction and keeps serving the
// original file; a failure at (3) or (6) is fatal to the open, because at that
// point the shard has no usable mapping and continuing would serve nothing.
//
// RETURNS the page index the shard must resume writing at — the PACK FRONTIER —
// or -1 when no compaction was published (disabled, below threshold, or
// abandoned), meaning the caller keeps its default. A compaction fills pages 0..k
// and leaves k+1..N-1 EMPTY, so the usual default of "start at the last page"
// would aim every post-compaction write at page N-1 and strand the freshly
// compacted region until firstPageWithRoomLocked wandered back down to it. This is
// a PACKING concern only. It used to be a correctness one — the rebuild resolved a
// key by page-walk order, so writes descending into lower pages resurrected older
// copies — but the v4 entry sequence (cache/ringbuf.go) removed that coupling:
// write order may disagree with page order freely, and the rebuild still resolves
// every key to the copy written LAST.
func (s *shard) compactAtOpen(dataDir, pagesPath string, size int64) (int, error) {
	if !s.isMmap || s.cfg.DisableColdCompaction {
		return -1, nil
	}
	capacity := s.pageCapacityBytes()
	if capacity == 0 {
		return -1, nil
	}
	used := s.bytesUsed()
	if float64(used)/float64(capacity) < mmapCompactMinOccupancy {
		return -1, nil
	}
	dropClock := s.compactDropClock()
	reclaimable := used - s.liveBytesAtOpen(dropClock) // live <= used by construction
	if float64(reclaimable) < mmapCompactMinReclaimRatio*float64(capacity) {
		return -1, nil
	}
	started := time.Now()

	// Check the live set fits before staging anything. (packPagesNeeded also tells
	// us the frontier, but see the reservation note below for why it is NOT the
	// reservation size.)
	pagesNeeded := s.packPagesNeeded(dropClock)
	if pagesNeeded > s.cfg.MaxPagesPerShard() {
		s.compactAborts.Add(1)
		slog.Error("cold compaction: live set does not fit; continuing uncompacted",
			"component", "cache", "path", pagesPath,
			"pages_needed", pagesNeeded, "max_pages", s.cfg.MaxPagesPerShard())
		return -1, nil
	}

	// Reserve the FULL file, not just the pages the pack touches. A truncate
	// allocates nothing, so without a reservation every store faults blocks in one
	// at a time, and on a full filesystem that fault arrives as SIGBUS — an
	// unrecoverable process kill. Reserving converts it into the ENOSPC handled
	// below.
	//
	// Reserving only [0, frontier] would close that at OPEN and REOPEN it at SERVE
	// time, which is strictly worse: the published file would carry a hole over
	// pages frontier+1..N-1 (compaction genuinely returns those blocks to the FS —
	// TestColdCompactionRoundTrip asserts the file shrinks), so the first runtime
	// write that lands up there faults, and on a full FS that is SIGBUS in the
	// middle of serving rather than a crash loop at startup with data intact. The
	// goal here is reclaiming WRITE CAPACITY inside the file, not returning disk
	// blocks — a mature pages.dat was already fully allocated anyway, so reserving
	// the whole thing costs nothing it did not already hold.
	tmpPath := compactTmpPath(pagesPath)
	tmpFile, tmpRegion, err := mmapFileAlloc(tmpPath, size, size, false)
	if err != nil {
		// Could not stage the rewrite (out of space, permissions). The original is
		// untouched and mapped: warn and serve uncompacted rather than refusing to
		// start. ENOSPC lands here BY DESIGN — see the reservation note above.
		s.compactAborts.Add(1)
		slog.Warn("cold compaction: cannot create staging file; continuing uncompacted",
			"component", "cache", "path", tmpPath, "alloc_bytes", size, "err", err)
		return -1, nil
	}
	//nolint:gosec // PageSize and MaxPagesPerShard are validated positive
	writeHeader(tmpRegion, uint32(s.cfg.PageSize), uint32(s.cfg.MaxPagesPerShard()), s.appliedIndex.Load())
	setAppliedStamp(tmpRegion, s.lastAppliedStampMs.Load())
	// Carry the PB applied frontier across the rewrite. Compaction preserves the
	// LIVE entry set exactly, so the compacted file materializes the same writes the
	// original did and the frontier that described the original still describes it.
	// Dropping it here would be safe (an under-report) but would silently reset every
	// durable PB node's watermark to genesis on the restart that compacts.
	setPBFrontier(tmpRegion, s.pbFrontierSeq.Load(), s.pbFrontierEpoch.Load())

	stageErr := error(nil)
	frontier, fits := s.packLiveInto(tmpRegion, dropClock)
	if !fits {
		// Unreachable given the packPagesNeeded gate above (same next-fit rule), so
		// getting here means those two disagree — the last line of defense before a
		// live entry would be dropped. Loud, and counted.
		stageErr = fmt.Errorf("live set does not fit in %d pages", s.cfg.MaxPagesPerShard())
	}
	if stageErr == nil {
		stageErr = msync(tmpFile, tmpRegion)
	}
	if stageErr == nil {
		stageErr = tmpFile.Sync() // data + metadata durable before anything points here
	}
	if cerr := munmapAndClose(tmpFile, tmpRegion); cerr != nil && stageErr == nil {
		stageErr = cerr
	}
	if stageErr != nil {
		_ = os.Remove(tmpPath)
		s.compactAborts.Add(1)
		slog.Error("cold compaction: staging failed; continuing uncompacted",
			"component", "cache", "path", tmpPath, "err", stageErr)
		return -1, nil
	}

	// Step 3: drop the original mapping. From here the shard has no backing store
	// until the remap below, and any failure is fatal to this open.
	oldFile, oldRegion := s.file, s.region
	s.file, s.region = nil, nil
	if cerr := munmapAndClose(oldFile, oldRegion); cerr != nil {
		_ = os.Remove(tmpPath)
		return -1, fmt.Errorf("cache: unmap before compaction swap: %w", cerr)
	}

	// Step 4: publish atomically. On failure the ORIGINAL is still intact on disk
	// (nothing has modified it), so drop the staging file and remap the original.
	compacted := true
	if rerr := os.Rename(tmpPath, pagesPath); rerr != nil {
		_ = os.Remove(tmpPath)
		s.compactAborts.Add(1)
		slog.Warn("cold compaction: rename failed; remapping the original file",
			"component", "cache", "path", pagesPath, "err", rerr)
		compacted = false
	} else {
		// Step 5: make the rename itself durable. Best-effort — losing it only
		// costs the compaction (the original reappears), never consistency.
		if derr := syncDir(dataDir); derr != nil {
			slog.Warn("cold compaction: directory fsync failed; the swap may not survive a crash",
				"component", "cache", "dir", dataDir, "err", derr)
		}
	}

	// Step 6: map whatever is now at pagesPath — the compacted file, or the
	// untouched original if the rename failed — and rebuild over it.
	if rerr := s.remapPagesFile(pagesPath, size); rerr != nil {
		return -1, rerr
	}
	if !compacted {
		return -1, nil
	}
	elapsed := time.Since(started)
	after := s.bytesUsed()
	s.compactions.Add(1)
	s.compactBytesReclaimed.Add(used - after)         // after <= used: the pack drops only entries
	s.compactNanos.Add(uint64(elapsed.Nanoseconds())) //nolint:gosec // a measured duration is non-negative
	slog.Info("cold compaction reclaimed mmap page bytes at open",
		"component", "cache", "path", pagesPath,
		"bytes_before", used, "bytes_after", after,
		"occupancy_after", s.occupancyRatio(), "drop_clock_ms", dropClock,
		"write_idx", frontier, "took_ms", elapsed.Milliseconds(),
		"replicated", s.cfg.Replicated)
	return frontier, nil
}

// remapPagesFile maps pagesPath as the shard's backing store, re-validating it
// exactly like any other open (a compacted file must pass the identical header
// checks — magic, version, page size, page count, CRC), then rebuilds the page
// objects and the index over it. Construction-time only.
func (s *shard) remapPagesFile(pagesPath string, size int64) error {
	file, region, err := mmapFile(pagesPath, size, s.cfg.Mlock)
	if err != nil {
		return fmt.Errorf("cache: remap %s after compaction: %w", pagesPath, err)
	}
	//nolint:gosec // PageSize and MaxPagesPerShard are validated positive
	appliedIdx, fresh, verr := validateHeader(region, uint32(s.cfg.PageSize), uint32(s.cfg.MaxPagesPerShard()))
	if verr != nil || fresh {
		_ = munmapAndClose(file, region)
		if verr == nil {
			verr = fmt.Errorf("header is all-zero")
		}
		return fmt.Errorf("cache: %s failed validation after compaction: %w", pagesPath, verr)
	}
	s.attachMmapRegion(file, region)
	s.appliedIndex.Store(appliedIdx)
	s.lastAppliedStampMs.Store(readAppliedStamp(region))
	pbSeq, pbEpoch := readPBFrontier(region)
	s.pbFrontierSeq.Store(pbSeq)
	s.pbFrontierEpoch.Store(pbEpoch)
	// The pre-compaction index refers to page objects and offsets that no longer
	// exist; rebuild it from the file now backing the shard.
	s.tab.Store(newIndexTable(0))
	s.rebuildIndexFromPages()
	return nil
}

// syncDir fsyncs a directory so a rename within it is durable.
func syncDir(dir string) error {
	// Nothing to force where the platform has no directory fsync (Windows): the
	// rename's metadata is the filesystem's own to commit there. Returning early
	// keeps every compaction from logging a durability warning about a step that
	// is not failing, only absent.
	if !dirSyncSupported {
		return nil
	}
	d, err := os.Open(dir) //nolint:gosec // G304: dir is the caller-configured DataDir
	if err != nil {
		return err
	}
	serr := d.Sync()
	if cerr := d.Close(); cerr != nil && serr == nil {
		serr = cerr
	}
	return serr
}
