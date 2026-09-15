// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cespare/xxhash/v2"
)

// ErrNotFound is returned when a key is absent or expired.
var ErrNotFound = errors.New("cache: not found")

// ErrFull is returned when a shard is at MaxPagesPerShard and AtCapPolicy is
// PolicyRejectWrites.
var ErrFull = errors.New("cache: shard at capacity")

// ErrCannotEvict is returned by PolicyRingbufEvict's evictUntilFitsLocked when
// there is nothing left to evict and still no room for the write. Like ErrFull
// it is NON-DETERMINISTIC across replicas — whether a given write hits it depends
// on this node's per-shard page occupancy, which need not match a peer's — so the
// replicated apply path must treat it as fatal (fail-closed) rather than advancing
// the applied index and silently diverging. Exported as a package-level sentinel
// so callers can errors.Is it (see shard/apply_class.go). The message is unchanged
// from the inline error it replaced.
var ErrCannotEvict = errors.New("cache: nothing left to evict and still no room")

// ErrFlushNotDurable marks a Cache.Flush() that could not make its per-shard
// durability watermark (the flushed.seq sidecar) durable — a create/write/fsync/
// rename/dir-fsync failure in writeFlushSidecar. It is NON-DETERMINISTIC across
// replicas: a local disk fault on one node need not occur on its peers. On the
// LEADER the sidecar is written BEFORE the index swap, so a failure returns with the
// keyspace intact; but a replicated apply must additionally guarantee that a FOLLOWER
// which hit this error does NOT advance its applied index past the flush, because
// advancing over a flush it could not durably apply would resurrect every pre-flush
// key on a later failover — silent divergence from peers whose sidecar landed. So
// shard/apply_class.go classifies it classFatal (fail-closed), exactly like ErrFull
// and ErrCannotEvict. Cache.Flush wraps every sidecar I/O failure so callers can
// errors.Is it through the apply path's multi-%w wrapping.
var ErrFlushNotDurable = errors.New("cache: flush watermark not durable")

// ErrReuseBarrier marks a write that could not proceed because the durable
// REUSE-ORDERING barrier for an mmap extent failed — either the crypto/rand nonce
// rotation or the msync of the reset (0,0)-bounds header (zeroDurableBoundsForReuseLocked).
// It is NON-DETERMINISTIC across replicas: a local disk/entropy fault on one node
// need not occur on its peers. The barrier is what makes a reused extent's cleared
// header durable BEFORE the write path refills it, so failing it leaves a pending
// durable-write obligation unmet; rather than write into a non-durable extent (which
// a later crash could reopen as the torn-writeback resurrection/forge window), the
// operation fails CLOSED and the extent stays unavailable. It surfaces from
// putAtExpLocked / delH when an eviction of the victim page hits the barrier, so the
// replicated apply path must treat it as fatal for the same reason ErrFull is (see
// shard/apply_class.go). A client may retry — a transient fault clears and the write
// then lands in a durably-reset extent.
var ErrReuseBarrier = errors.New("cache: could not make reused page header durable")

// shard is one independent slice of the cache: its own page list, lock,
// and index. Shards do not share state and are safe to use concurrently
// across goroutines.
type shard struct {
	cfg Config
	mu  sync.RWMutex
	// tab is the lock-free read index. Writers mutate the pointed-at table in
	// place under mu (atomic slot stores) and swap in a fresh table on resize via
	// tab.Store; readers snapshot tab.Load() once per Get and never take mu.
	tab   atomic.Pointer[indexTable]
	pages []*page
	// pageSlots is the read path's source of truth for resolving a page index to a
	// page OBJECT. It has fixed length MaxPagesPerShard — its slice header is set
	// once at construction and never mutated — so a lock-free reader indexes it and
	// atomically loads the slot with no snapshot and no lock. Writers keep it in
	// sync with `pages`: every mutation of a slot (initial alloc, heap growth, and
	// heap-mode ringbuf retire) does pageSlots[idx].Store(obj). Unallocated heap
	// slots are nil; a ref is only ever minted for a populated slot.
	pageSlots []atomic.Pointer[page]
	// genCounter mints page generations (see page.gen / slabRef.gen). Monotonic
	// per shard; accessed under mu (or during single-threaded construction).
	genCounter uint16

	// writeIdx is the page Put writes into first (the current "open" page). It
	// avoids re-scanning every page on each write: after eviction frees a page,
	// writes go straight back to it instead of falling through the at-cap path
	// and scanning all pages for room. Guarded by mu (write path only).
	writeIdx int

	// writeSeq is the LAST write sequence this shard stamped into an entry's meta
	// word (see cache/ringbuf.go); the next write uses writeSeq+1. Guarded by mu,
	// like every other write-path field.
	//
	// It is the persisted write-recency signal warm restart needs. Page order is
	// NOT write order — findOrMakePageLocked falls back to firstPageWithRoomLocked,
	// which scans from index 0 and so REVISITS any lower page that still has tail
	// room — so a rebuild that resolved a key by page-walk order could pick an
	// OLDER copy of an overwritten key (#12A). Resolving by max(seq) cannot.
	//
	// It is deliberately NOT persisted in the header. rebuildIndexFromPages
	// recovers it as max(seq) over every CRC-VALID entry on the pages, which is
	// crash-safe for free: a torn tail fails its CRC, is rejected, and therefore
	// never contributes a sequence the next write could collide with. A header
	// field would instead need its own durability story (it could be stale after a
	// crash, handing out sequences already on disk).
	//
	// It is node-local PHYSICAL state. Sequences never enter committed state:
	// Iterate — and therefore serializeSnapshot — emits key/value/expiry only.
	writeSeq uint64

	// nextVictim is the rotation cursor for PolicyRingbufEvict: eviction starts
	// its search for the oldest non-empty page here and advances past each
	// drained page, so victims cycle through all pages in FIFO order. Without it
	// eviction always drained the lowest-index page, which — once page 0 is
	// refilled with the newest writes — degenerates into an anti-FIFO,
	// single-page ring. Guarded by mu (write path only).
	nextVictim int

	// isMmap records the shard's storage MODE. It is set once in newShard, before
	// the shard is published to any other goroutine, and never mutated again — so
	// every path (including the lock-free read path) may test it without holding
	// mu. `region` is NOT a substitute: it is ordinary lock-guarded mutable state
	// (newShard reassigns it when cold compaction swaps the pages file), so
	// reading it off-lock to answer "is this shard mmap-backed?" was a data race
	// waiting to be scheduled. Mode never changes; the region behind it may.
	isMmap bool

	// sieve caches sieveVisited() (cache/sieve.go): whether this shard maintains the
	// SIEVE reference hint in its index control words. Set once in newShard, before
	// the shard is published, and never mutated — so the lock-free read path tests it
	// with no lock, and pays one bool load per hit for a feature that is off by
	// default.
	sieve bool

	// mmap-only state (nil/zero in heap mode). Guarded by mu.
	file   *os.File
	region []byte
	// framingKey is a COPY of the pages file's per-file 16-byte MAC secret (v5
	// header, cache/file.go). A copy, not a slice of region, so it survives a
	// compaction remap that unmaps the old region; every mmap page's framingKey
	// aliases this slice. nil in heap mode. Set in newShard / remapPagesFile before
	// attachMmapRegion, immutable for the mapping's life.
	framingKey   []byte
	appliedIndex atomic.Uint64

	// dataDir is the shard's on-disk directory (the parent of pages.dat), or "" in
	// heap mode. Set once in newShard, immutable thereafter, so flush can locate the
	// durability sidecar (dataDir/flushed.seq) without re-deriving it from s.file.
	dataDir string

	// flushedThroughSeq is the write-sequence FLOOR a prior Cache.Flush() recorded in
	// this shard's sidecar (dataDir/flushed.seq): every entry whose metaSeq(meta) <=
	// this value was logically wiped by that flush and MUST NOT be re-indexed by the
	// warm-restart rebuild, nor resurrected across a restart. Restored from the
	// sidecar at open BEFORE rebuildIndexFromPages runs; 0 when no sidecar exists
	// (byte-identical to pre-flush behaviour) or it fails its CRC. Written under mu on
	// flush; otherwise touched only at construction (single-threaded, pre-publish),
	// and never read off-lock at runtime, so a plain field suffices.
	flushedThroughSeq uint64

	// stats — atomic counters; no shard lock needed. `hits` is NOT stored: every
	// read bumps `gets` on entry and bumps `misses` on any non-returning path
	// (absent, collision, corrupt, expired-on-read), so Hits = Gets - Misses is
	// exact and the hot hit path pays a single atomic (gets) instead of two.
	gets        atomic.Uint64
	misses      atomic.Uint64
	puts        atomic.Uint64
	dels        atomic.Uint64
	expirations atomic.Uint64
	evictions   atomic.Uint64
	// evictionsLive counts only the evictions that displaced the entry the
	// index still pointed at - a record lost to CAPACITY rather than TTL.
	evictionsLive atomic.Uint64
	rejects       atomic.Uint64
	pagesAlloc    atomic.Uint64
	corrupt       atomic.Uint64
	// corruptBytes is the framed page bytes those corruption responses discarded
	// without reading them. See Stats.CorruptionBytesDiscarded.
	corruptBytes atomic.Uint64

	// inPlaceUpdates counts the writes that overwrote their key's stored copy where
	// it lay instead of appending a new one and stranding the old — the writes that
	// created no garbage (see Config.InPlaceSameSizeUpdate). It stays 0 on every
	// shard the feature is not enabled for, because those shards never run the
	// eligibility probe at all.
	//
	// With the feature on, inPlaceUpdates/puts is also the measured share of the
	// write stream that is pure same-size rewriting, which is the figure that says
	// whether the reader-side cost is buying anything on this workload.
	inPlaceUpdates atomic.Uint64

	// seqlockRetries and seqlockFallbacks measure the lock-free read protocol
	// (cache/seqlock.go): how many read attempts were thrown away because a
	// rewrite moved the bytes underneath them, and how many reads exhausted the
	// retry budget and took the read lock instead. Both stay 0 unless
	// Config.InPlaceSeqlockReads is on.
	//
	// They are the protocol's health check. Retries should be a small fraction of
	// Gets and fallbacks near zero; a fallback rate that is not near zero means
	// reads are being serialised after all, and the stripe mapping or the workload
	// — not the lock — is what to look at.
	seqlockRetries   atomic.Uint64
	seqlockFallbacks atomic.Uint64

	// indexRehashes / indexRehashNanos measure the index table's growth step: how
	// many times a write (or the warm-restart rebuild) replaced the table with a
	// freshly-sized one, and how long those rebuilds took in total. The rebuild
	// re-inserts every live entry and runs under s.mu, so it is the one write-path
	// step whose cost scales with the shard's entry count rather than the entry
	// being written; these two counters are what make its frequency and its cost
	// separable instead of a guess.
	indexRehashes    atomic.Uint64
	indexRehashNanos atomic.Uint64

	// cold-compaction counters (cache/compact.go). Written once, at open, before
	// the shard is published; atomic only so Stats can read them from any
	// goroutine afterwards. compactAborts covers every path that decided against
	// publishing a staged rewrite (no space to stage, a pack that did not fit, a
	// failed rename) — a non-zero value with compactions stuck at 0 is the signal
	// that a persistent shard's ghost bytes are NOT being reclaimed by restarts.
	compactions           atomic.Uint64
	compactAborts         atomic.Uint64
	compactBytesReclaimed atomic.Uint64
	compactNanos          atomic.Uint64

	// online relocating-compaction counters (cache/compact_online.go). Written under
	// s.mu on the relocation path; atomic so Stats can read them from any goroutine.
	// Zero on every shard that is not an mmap replicated reject-writes shard.
	relocations           atomic.Uint64 // live entries relocated out of fragmented pages
	relocatedBytes        atomic.Uint64 // their on-disk byte total
	relocatePagesGone     atomic.Uint64 // source pages fully evacuated and marked retired
	relocatePagesRecycled atomic.Uint64 // retired pages whose quarantine elapsed and were reset back into writable space

	// relocating-EVICTION counters (cache/relocate_evict.go) — a different mechanism
	// from the online compactor above, on the other storage mode: these count live
	// records a heap RINGBUF shard copied forward out of a page it was about to
	// drain. Written under s.mu on the eviction path; atomic so Stats can read them
	// from any goroutine. Zero unless Config.RelocatingEviction is set.
	//
	// WRITE THEM BYTES-FIRST, COUNT-SECOND. snapshot() loads them in the opposite
	// order, which is what keeps a concurrent scrape from ever reporting records
	// relocated with no bytes moved. See the note in snapshot().
	evictRelocations    atomic.Uint64 // live records copied forward instead of dropped
	evictRelocatedBytes atomic.Uint64 // their framed byte total

	// BACKGROUND (free-page reserve) relocation counters — the sweeper's half of the
	// same feature (cache/relocate_reserve.go), kept SEPARATE from the write-path pair
	// above so the split between the two is observable: a shard whose sweeper is
	// keeping up shows these climbing while evictRelocations stays flat. Same
	// bytes-first, count-second discipline, for the same reason.
	reserveRelocations    atomic.Uint64 // live records the reserve pass copied forward
	reserveRelocatedBytes atomic.Uint64 // their framed byte total
	reservePagesFreed     atomic.Uint64 // pages it fully evacuated and retired

	// reserveChunkHook observes each finished chunk of the reserve pass (entries and
	// bytes moved under one lock acquisition), called with s.mu released. Nil unless a
	// test installs one, to assert the chunk bound or to act inside the window
	// the chunking opens. Atomic because the sweeper goroutine reads it.
	reserveChunkHook atomic.Pointer[func(entries, bytes int)]

	// reclaimableCache holds the last-published ghost-byte snapshot (figure + the
	// wall-clock nanos it was taken), so Stats() stays O(1) under frequent scraping: the
	// O(entries) liveness walk (liveAndUsedBytes) runs at most once per reclaimableStatsTTL,
	// not once per Stats() call. The sweeper refreshes it for free on every pass when online
	// compaction is enabled; when disabled it is filled lazily by reclaimableBytesForStats.
	// It is a single atomic pointer (not two separate atomics) published CAS-if-newer, so
	// concurrent/out-of-order refreshers can neither tear the (figure, timestamp) pair nor
	// regress it to an older snapshot. Wall time here is observability-only (a gauge
	// freshness bound), never a logical clock.
	reclaimableCache atomic.Pointer[reclaimableSnapshot]

	// mmapHighWaterWarned is the rising-edge latch for the replicated-mmap
	// page-byte occupancy alert (#4 Option 3): a persistent replicated shard
	// reclaims expired INDEX SLOTS deterministically but cannot reclaim page
	// BYTES while it is RUNNING (see reclaimExpiredHeapPages' mmap early-return —
	// a file-backed page can't be frozen-swapped out from under a lock-free
	// reader), so under sustained TTL churn ghost bytes climb toward ErrFull. We
	// warn once when occupancy crosses the high-water and re-arm below the
	// low-water (hysteresis) so a chronically-full shard is visible BEFORE it
	// fail-closed halts. The warning's remedy is a restart: cold compaction at
	// open (cache/compact.go) rewrites the file live-only from this same band.
	mmapHighWaterWarned atomic.Bool

	// sweeper
	stopSweeper chan struct{}
	sweepWG     sync.WaitGroup

	// lastAppliedStampMs is the running MAX of every apply-stamp this shard has
	// observed on the STAMPED apply path (getAtH / putAtH — the sites that receive
	// an explicit leader/primary stamp; see advanceAppliedStamp). It is the LOGICAL
	// clock the replicated TTL sweeper reclaims against (#4 Phase B / B3b), NOT a
	// wall clock.
	//
	// Two properties make it safe to sweep against, and both come for free from the
	// replicated-log model:
	//
	//   1. DETERMINISTIC across replicas. Every replica applies the identical
	//      committed entries in the identical order, each carrying the identical
	//      leader-baked stamp, so max(stamps) is the SAME value on every replica at
	//      the same applied index. A sweep keyed off it therefore removes the exact
	//      same keys on every replica — no wall-clock divergence.
	//
	//   2. MONOTONIC non-decreasing. It only ever moves forward (max), and the
	//      leader clamps each new stamp to be >= this value (see shard/store.go
	//      applyOpIndexed), so it never regresses even across leader failover (a new
	//      leader has already applied every committed entry, so its own
	//      lastAppliedStampMs already reflects the max before it stamps anything).
	//      Monotonicity is what makes the sweeper-vs-later-write race safe: a write W
	//      applied after a sweep is stamped W.stampMs >= lastAppliedStampMs >= the
	//      exp of any key the sweep removed, so W also judges that key expired and
	//      never resurrects it.
	//
	// putAbsH (snapshot restore) does NOT advance it: PutAbs carries an ABSOLUTE
	// expiry, not a leader stamp, so there is no stamp to fold in. This is correct
	// for restore: the value is rebuilt deterministically from the committed log
	// tail applied AFTER restore (which re-advances it identically on every
	// replica), so the sweep stays cross-replica identical without the snapshot
	// having to carry the clock. Until the first stamped apply after a restore,
	// lastAppliedStampMs is 0 (or its pre-restore value on a warm cache) and the
	// replicated sweep is a no-op — no key is reclaimed, which is safe.
	lastAppliedStampMs atomic.Uint64

	// pbFrontierSeq / pbFrontierEpoch mirror the persisted PB applied frontier
	// (header bytes 44..63) for this shard: the (seq, epoch) identity of the newest
	// primary-backup write whose data was flushed BEFORE the header carrying this
	// pair. Restored at open (readPBFrontier) and rewritten by Cache.SetPBFrontier.
	// Zero in Raft mode and in heap mode — nothing stamps them there, and 0 is the
	// genesis frontier, which is the safe (under-reporting) answer.
	//
	// Held as two independent atomics but only ever WRITTEN as a pair under s.mu
	// (SetPBFrontier) and read as a pair by Cache.PBFrontier, which runs at open
	// before the cache is published to any writer. There is no concurrent-read seam
	// where a torn pair is observable.
	pbFrontierSeq   atomic.Uint64
	pbFrontierEpoch atomic.Uint64

	// nowFn overrides the wall-clock source for the non-apply expiry sites (see
	// Config.NowFn). Held as an atomic pointer so a test can swap the clock (e.g.
	// pin a fixed instant for a canonical fingerprint) without racing the
	// lock-free read path that consults it. nil ⇒ the real clock (nowMs).
	nowFn atomic.Pointer[func() uint64]

	// onRemove points at the OWNING CACHE's removal hook (Cache.onRemove), so a
	// SetOnRemove is one store seen by every shard. It is set in the STRUCT
	// LITERAL in newShard and never written again — the hook VALUE behind it is
	// what changes, atomically. nil for a shard built directly by newShard with
	// no owner (tests, benchmarks): fireOnRemove short-circuits on it.
	//
	// IT CANNOT BE ASSIGNED AFTER newShard RETURNS. newShard starts the sweeper
	// goroutine, and the sweeper reaches fireOnRemove on the very first tick of a
	// warm open that already holds expired entries — so a later `s.onRemove = …`
	// in Cache.New is a plain write racing a live reader. Passing it in removes
	// the window rather than narrowing it: by the time any goroutine exists, the
	// field is already final. See cache/onremove.go for the callback contract.
	onRemove *atomic.Pointer[func([]byte)]

	// chunkedRestarts counts how many times IterateChunked abandoned a shard pass
	// because a concurrent rehash swapped the index table. Diagnostics only —
	// nothing depends on it — but it is what proves the restart path (and its
	// bounded fallback) actually executes.
	chunkedRestarts atomic.Uint64
}

// newShard constructs a shard. dataDir="" selects heap mode (heap-only behavior).
// With a non-empty dataDir the shard is mmap-backed; the directory is created
// if it does not exist.
//
// onRemove is the owning cache's removal hook, or nil for an owner-less shard
// (tests, benchmarks). It is a PARAMETER because this function starts the
// sweeper: see the field's doc comment.
func newShard(cfg Config, dataDir string, onRemove *atomic.Pointer[func([]byte)]) (*shard, error) {
	s := &shard{
		cfg:         cfg,
		dataDir:     dataDir,
		stopSweeper: make(chan struct{}),
		onRemove:    onRemove,
	}
	if cfg.NowFn != nil {
		fn := cfg.NowFn
		s.nowFn.Store(&fn)
	}
	s.sieve = s.sieveVisited()
	s.tab.Store(newIndexTable(0))
	// Fixed-length read-path page table (see the pageSlots field). Sized to the
	// shard's hard page cap so every reachable page index is a valid slot.
	s.pageSlots = make([]atomic.Pointer[page], cfg.MaxPagesPerShard())

	if dataDir == "" {
		// Heap mode — heap-only behavior.
		for i := 0; i < cfg.InitialPagesPerShard; i++ {
			s.allocHeapPageLocked()
		}
		if len(s.pages) > 0 {
			s.writeIdx = len(s.pages) - 1 // match "write the last page first"
		}
		s.startSweeper()
		return s, nil
	}

	// Mmap mode.
	s.isMmap = true
	if err := os.MkdirAll(dataDir, 0o750); err != nil {
		return nil, fmt.Errorf("cache: mkdir %s: %w", dataDir, err)
	}
	pagesPath := filepath.Join(dataDir, "pages.dat")
	maxPages := cfg.MaxPagesPerShard()
	size := int64(headerSize + maxPages*cfg.PageSize)

	// CRASH RECOVERY, step 1 of the cold-compaction swap: a leftover temp file
	// means a previous compaction died BEFORE its atomic rename. The rename is
	// the only step that publishes the compacted file, so pages.dat is still the
	// intact original and the temp is a partial write with no claim to the data.
	// Discard it before mapping anything. (After the rename there is no temp
	// path left to find, so this can never delete a completed compaction.)
	if err := discardCompactTemp(pagesPath); err != nil {
		return nil, err
	}

	file, region, err := mmapFile(pagesPath, size, cfg.Mlock)
	if err != nil {
		return nil, err
	}
	appliedIdx, fresh, verr := validateHeader(region, uint32(cfg.PageSize), uint32(maxPages)) //nolint:gosec // PageSize and maxPages are validated positive
	if verr != nil && errors.Is(verr, errFutureVersion) {
		// A pages file written by a NEWER build. Do NOT rotate it aside — that would
		// silently destroy data a future format wrote deliberately. Refuse the open so
		// an operator downgrade is a loud, recoverable error instead of data loss.
		if cerr := munmapAndClose(file, region); cerr != nil {
			return nil, fmt.Errorf("cache: close future-version file: %w (validation: %w)", cerr, verr)
		}
		return nil, verr
	}
	if verr != nil {
		// Rotate the bad file and start fresh.
		if cerr := munmapAndClose(file, region); cerr != nil {
			return nil, fmt.Errorf("cache: close bad file: %w", cerr)
		}
		badPath := fmt.Sprintf("%s.bad-%s", pagesPath, time.Now().UTC().Format("2006-01-02T15-04-05.000"))
		if rerr := os.Rename(pagesPath, badPath); rerr != nil {
			return nil, fmt.Errorf("cache: rotate bad file: %w", rerr)
		}
		slog.Warn("rejecting pages file; renamed aside", "component", "cache", "path", pagesPath, "err", verr, "renamed_to", badPath)
		file, region, err = mmapFile(pagesPath, size, cfg.Mlock)
		if err != nil {
			return nil, err
		}
		fresh = true
		appliedIdx = 0
	}

	// The on-disk format version decides how the file is brought up. A fresh or
	// rotated-aside file is v5 by construction; an existing file's version is read from
	// the header (validateHeader already proved it in [minReadable, cacheVersion] and,
	// for a v5 file, that its framing key is readable). A v4 file validates here but
	// must NOT be framed with the v5 reader — it is migrated first.
	onDiskVersion := cacheVersion
	if !fresh {
		onDiskVersion = binary.LittleEndian.Uint32(region[8:12])
	}

	if fresh {
		// Lay down the v5 header — INCLUDING a fresh random framing key — BEFORE
		// attaching, so the pages built over the region pick the key up. A crypto/rand
		// failure aborts the open rather than publishing a file with a broken secret.
		if herr := writeHeader(region, uint32(cfg.PageSize), uint32(maxPages), 0); herr != nil { //nolint:gosec // PageSize and maxPages are validated positive
			_ = munmapAndClose(file, region)
			return nil, herr
		}
	}

	if !fresh && onDiskVersion < cacheVersion {
		// v4 → v5 MIGRATE-ON-OPEN, fail-closed. Reads the v4 file through the v4 decoder,
		// stages a v5 rewrite (fresh header + framing key + per-page nonces + MAC frames)
		// through the same crash-safe temp+rename+dir-fsync swap compaction uses, and
		// remaps. On success s.file/s.region/s.framingKey name the v5 file and the shard
		// falls through to the normal v5 rebuild below; on failure the intact v4 file is
		// left in place (never rotated aside) and the open fails loudly. Consumes the v4
		// mapping (file, region) internally.
		if merr := s.migrateV4ToV5(dataDir, pagesPath, file, region, size, appliedIdx); merr != nil {
			return nil, merr
		}
	} else {
		// Fresh or already-v5: adopt the framing key and attach. A copy, so the shard's
		// key survives a later compaction remap that unmaps this region.
		fk, ok := readFramingKey(region)
		if !ok {
			// Unreachable — validateHeader proved a v5 key readable and the fresh branch
			// just wrote one — but guard rather than frame every entry under a zero key.
			_ = munmapAndClose(file, region)
			return nil, fmt.Errorf("cache: framing key unreadable after open")
		}
		s.framingKey = append([]byte(nil), fk...)
		s.attachMmapRegion(file, region)
	}
	s.appliedIndex.Store(appliedIdx)
	// Restore the flush watermark BEFORE the index rebuild, so rebuildIndexFromPages
	// can skip every entry a prior Cache.Flush() logically wiped (seq <= floor). Read
	// unconditionally: on a fresh pages file it is normally absent (→ 0), but a
	// sidecar that outlived a rotated-aside pages file would otherwise let post-fresh
	// low-seq writes be wrongly skipped on the next restart — the writeSeq floor-lift
	// below closes that too.
	// A present-but-invalid sidecar FAILS the open (see readFlushSidecar): a corrupt
	// flush watermark cannot be silently downgraded to floor 0, which would re-index
	// pre-flush entries permanently and diverge this replica from peers with intact
	// sidecars. A genuinely absent sidecar returns (0, nil) and opens normally.
	flushedFloor, ferr := readFlushSidecar(dataDir)
	if ferr != nil {
		if s.file != nil {
			_ = munmapAndClose(s.file, s.region)
		}
		return nil, ferr
	}
	s.flushedThroughSeq = flushedFloor
	if !fresh {
		// Restore the persisted LOGICAL clock and PB frontier from the file now backing
		// the shard (s.region — which after a v4 migration is the fresh v5 file, whose
		// header migration carried these fields into). Read from s.region, never the
		// local `region`, which a migration has unmapped.
		//
		// The LOGICAL clock lets cold compaction judge TTL expiry deterministically on a
		// replicated shard (see cache/compact.go). The PB frontier is the ONLY thing that
		// lets a restarted PB node describe the FSM it just warm-restarted from: there is
		// no PB log or snapshot to re-derive it, so without it the engine would present
		// (0,0) — a genesis claim — over real data.
		s.lastAppliedStampMs.Store(readAppliedStamp(s.region))
		pbSeq, pbEpoch := readPBFrontier(s.region)
		s.pbFrontierSeq.Store(pbSeq)
		s.pbFrontierEpoch.Store(pbEpoch)
	}

	writeIdx := -1
	if !fresh {
		// A recovery-flush failure REFUSES THE OPEN: a corrected/truncated durable
		// bound that could not be made durable would let the next crash resurrect the
		// stale bytes this rebuild just excluded (see rebuildIndexFromPages).
		if rerr := s.rebuildIndexFromPages(); rerr != nil {
			_ = munmapAndClose(s.file, s.region)
			return nil, rerr
		}
		// Cold compaction: reclaim ghost page BYTES by rewriting the pages file
		// with only the live entries, then mapping the compacted result. Safe
		// precisely because it happens HERE — the shard has not been published to
		// any other goroutine, so there is no reader whose aliases could be
		// invalidated by swapping the mapped file. See compactAtOpen.
		wi, cerr := s.compactAtOpen(dataDir, pagesPath, size)
		if cerr != nil {
			_ = munmapAndClose(s.file, s.region)
			return nil, cerr
		}
		writeIdx = wi
	}
	if writeIdx < 0 {
		writeIdx = len(s.pages) - 1 // match "write the last page first"
	}
	// A published compaction refills pages 0..k and leaves the rest EMPTY, so it
	// hands back its pack frontier and writes RESUME there rather than at the last
	// page, which would strand the compacted region. Correctness no longer depends
	// on this: rebuildIndexFromPages resolves each key by WRITE SEQUENCE, so write
	// order is free to disagree with page order (it always did — see
	// firstPageWithRoomLocked). This is now purely about packing density.
	s.writeIdx = writeIdx

	// writeSeq-restore — the flush durability payoff, and a genuine hazard without it.
	// rebuildIndexFromPages recovers writeSeq as max(seq) over the SURVIVING entries;
	// when a prior Flush() emptied this shard there are no survivors and the recovered
	// max is ~0, while the flush floor may be large. Lift writeSeq to the floor so the
	// NEXT writes get seq > floor and can never be re-classified as flushed (seq <=
	// floor) — and therefore skipped — by a FUTURE restart's rebuild. Skipping this
	// step would silently lose every post-flush write on the second restart. It only
	// ever raises writeSeq (max is monotonic), so it is a no-op on a shard that was
	// never flushed.
	if s.flushedThroughSeq > s.writeSeq {
		s.writeSeq = s.flushedThroughSeq
	}

	s.startSweeper()
	return s, nil
}

// attachMmapRegion installs a freshly mapped pages file as the shard's backing
// store: it takes ownership of file/region and (re)builds the page objects and
// the read-path page table over it. It does NOT touch the index — the caller
// rebuilds or initializes that. Called only during construction (including the
// cold-compaction remap), before the shard is shared, so no lock is taken.
func (s *shard) attachMmapRegion(file *os.File, region []byte) {
	s.file = file
	s.region = region
	maxPages := s.cfg.MaxPagesPerShard()
	s.pages = make([]*page, maxPages)
	for i := 0; i < maxPages; i++ {
		offset := headerSize + i*s.cfg.PageSize
		// 3-index slice: cap the page at its own end so a bug in the append path can
		// never reach past this page into the next one's bytes through the region tail.
		p := newMmapPage(region[offset : offset+s.cfg.PageSize : offset+s.cfg.PageSize])
		p.gen = s.nextGen()
		// Every entry on this page is MAC'd under the per-file secret; the page needs it
		// to encode (append/relocate) and to verify (recovery/eviction resync).
		p.framingKey = s.framingKey
		// Seed the RUNTIME bounds from the DURABLE header. This is the one place the
		// mapped header is read back into runtime: recovery (rebuildIndexFromPages)
		// trusts the durable header bounds, so the runtime bounds head()/tail() serves
		// from here on must START from exactly what a prior process projected there.
		// After this, head()/tail() are plain runtime fields and the header is written
		// only by projectBounds at a flush point. A fresh file has a zeroed header, so
		// this seeds (0,0) — correct for an empty page.
		p.heapHead, p.heapTail = p.durableBounds()
		// Seed the RUNTIME nonce from the DURABLE per-page header alongside the bounds:
		// the entries already on this page were MAC'd under whatever nonce a prior
		// process last wrote there (0 on a fresh page's first life, or a rotated value
		// after a reuse), so verification here must key off exactly that. Rotated only
		// at the reuse point from here on (zeroDurableBoundsForReuseLocked).
		p.nonce = p.durableNonce()
		s.pages[i] = p
		s.pageSlots[i].Store(p) // mmap page objects are fixed; publish once
		regionNotePage(s, p)    // compiled out unless the measurement build tag is set
	}
}

// pageBoundSnap captures an mmap page's runtime bounds and generation at an
// instant, for a crash-ordered durable projection at an UNLOCKED sync point (see
// snapshotPageBounds / projectSnapshotBounds).
type pageBoundSnap struct {
	head, tail int
	gen        uint16
}

// snapshotPageBounds records every page's runtime head/tail and generation under
// s.mu. Taken BEFORE an unlocked data msync so the bounds later projected from it
// name only entries that msync flushed — never a write that lands DURING the msync,
// whose bytes are not yet on disk (a durable bound naming non-durable bytes is the
// forward skew this whole change removes). mmap shards only.
func (s *shard) snapshotPageBounds() []pageBoundSnap {
	s.mu.Lock()
	defer s.mu.Unlock()
	snaps := make([]pageBoundSnap, len(s.pages))
	for i, p := range s.pages {
		snaps[i] = pageBoundSnap{head: p.head(), tail: p.tail(), gen: p.gen}
	}
	return snaps
}

// projectSnapshotBounds writes each snapshot's bounds into its page's DURABLE
// header under s.mu. A page whose generation no longer matches its snapshot was
// RECYCLED during the unlocked window (compactRecycleRetiredLocked swapped in a
// fresh object over the same extent); its extent may now hold different bytes, so
// its durable bound is set to (0,0). Under-projecting a reused extent is the safe
// direction — recovery indexes nothing there until a later flush advances it —
// whereas projecting the snapshot's now-stale tail would name bytes the recycle
// freed. The caller must issue a full-region msync AFTER this to make the
// projection durable. mmap shards only.
//
// The generation is a uint16, so in principle a page recycled EXACTLY 65536 times
// within one unlocked window would alias its snapshot generation and be treated as
// unchanged. That needs 65536 alias-quarantine cycles between the snapshot and this
// call — astronomically unlikely on any real timescale — so it is documented, not
// guarded; widening gen would ripple through slabRef's bit packing for no practical
// gain.
func (s *shard) projectSnapshotBounds(snaps []pageBoundSnap) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, p := range s.pages {
		if i >= len(snaps) {
			break // mmap page count is fixed; defensive against a shorter snapshot
		}
		if p.gen == snaps[i].gen {
			p.projectBounds(snaps[i].head, snaps[i].tail)
		} else {
			p.projectBounds(0, 0)
		}
	}
}

// msyncPageHeaderLocked flushes just the OS page carrying mmap page idx's 8-byte
// durable header (head/tail). msync requires a PAGE-ALIGNED start address, and a
// page's header sits at headerSize+idx*PageSize — almost never OS-page-aligned — so
// the flush runs from the OS page boundary at or below that offset through the end
// of the header. That also flushes whatever entry bytes happen to share the header's
// OS page, which is harmless. A no-op on a heap shard (never mmap) and where the
// platform has no msync. Caller ensures exclusive access to the header bytes.
func (s *shard) msyncPageHeaderLocked(idx int) error {
	if !s.isMmap || s.region == nil {
		return nil
	}
	off := headerSize + idx*s.cfg.PageSize
	start := off &^ (os.Getpagesize() - 1)
	// Routed through syncRegion (the msyncTestHook seam) so a test can fault-inject a
	// reuse/recovery-barrier msync failure and assert the fail-closed behavior. The
	// only production cost is one nil-pointer load per header flush.
	return syncRegion(s.file, s.region[start:off+pageHdrSize])
}

// zeroDurableBoundsForReuseLocked rotates mmap page idx's per-page nonce, zeroes its
// DURABLE header bounds, and flushes that header to disk. It is the REUSE-ORDERING
// guard: whenever an mmap extent is reset to be handed back to the write path
// (drain-to-empty, corruption discard, online recycle), its durable bound must reach
// disk at (0,0) BEFORE the extent is republished, or the OS may write back the reused
// entry bytes while the header still holds the OLD (larger) bound — recovery would
// then frame a previous life's bytes (reverse skew / a forged frame).
//
// The NONCE ROTATION is the v5 second guard for that same window: a fresh random
// nonce means any old-life entry still lying in the extent can no longer verify (its
// MAC was computed under the previous nonce), so even a torn writeback that resurrects
// a stale bound cannot make those bytes pass as this life's entries. The runtime
// nonce is rotated too, so subsequent appends MAC under the new value. Both header
// writes ([8:16] nonce, [0:8] zeroed bounds) are made durable by the single msync
// below, strictly before the extent is republished; the nonce is otherwise stable for
// a page's life, so projectBounds at the per-write sync points (which touch only
// [0:8]) leave it intact and the ENTRIES→BOUNDS→WATERMARK ordering is unchanged.
//
// Mandatory even on a non-Durable mmap shard, whose pages the OS still flushes lazily.
// One small msync per page RESET, not per write. A no-op on a heap shard. Must hold
// s.mu.
//
// FAILS CLOSED (returns an error). The nonce rotation and the header msync are a
// SINGLE durable-write obligation, not a best-effort courtesy: dropping either one
// reopens the torn-writeback window this guard exists to close (a crash could then
// recover the extent's stale bytes under the old, larger bound / old nonce). So a
// failure of the crypto/rand nonce draw or of the msync is RETURNED, and every caller
// treats a non-nil return as "this extent is NOT safe to republish": the write path
// leaves it unavailable and fails the operation, keeping the pending durable write
// pending rather than committing it into a non-durable extent. The in-memory
// projection is still made (as before) so the runtime bounds/nonce are correct if the
// caller chooses to retry; what changes is only that a failed flush no longer lets the
// extent go back into service.
func (s *shard) zeroDurableBoundsForReuseLocked(idx int) error {
	if !s.isMmap || s.region == nil {
		return nil
	}
	p := s.pages[idx]
	// NONCE FIRST, and do NOT proceed on the old one. A crypto/rand failure leaves us
	// with no fresh nonce, and reusing the extent under the PREVIOUS nonce would let a
	// torn writeback that resurrects a stale bound present old-life bytes that still
	// verify — exactly the forge window the rotation closes. Return before touching the
	// bounds so the caller does not republish.
	n, err := randomNonce()
	if h := randomNonceTestHook; h != nil {
		n, err = h()
	}
	if err != nil {
		return fmt.Errorf("%w: rotate page nonce on reuse (page %d): %w", ErrReuseBarrier, idx, err)
	}
	p.nonce = n
	p.projectNonce(n)
	p.projectBounds(0, 0)
	// The single msync makes both header writes ([8:16] nonce, [0:8] zeroed bounds)
	// durable strictly BEFORE the extent is republished. If it fails, the durable
	// header still names the extent's former (larger) bound and old nonce, so a later
	// crash could frame its stale contents — return the error and leave the extent
	// unavailable rather than logging and continuing into a non-durable reuse.
	if err := s.msyncPageHeaderLocked(idx); err != nil {
		return fmt.Errorf("%w: msync reset page header (page %d): %w", ErrReuseBarrier, idx, err)
	}
	return nil
}

// flushRecoveredBoundsLocked projects mmap page idx's CURRENT runtime bounds into
// its durable header and flushes that header. It is the recovery-time counterpart
// to zeroDurableBoundsForReuseLocked: rebuildIndexFromPages corrects a page's bounds
// in runtime when it rejects a corrupt header or truncates a torn tail, and since
// setHead/setTail no longer write through to the durable header, that correction
// must be projected and flushed or the next open (or a reuse before the first sync
// point) would trust the stale on-disk bound again. Safe to flush immediately: the
// bytes the corrected bounds name are already on disk (they decoded from it), so
// there is no entries-before-bounds ordering to respect here. A no-op on a heap
// shard / where the platform has no msync.
//
// FAILS CLOSED (returns an error). If the msync fails, the corrected bound is only in
// runtime while the durable header still holds the stale/torn one — so the next open
// (or a reuse before the first sync point) would trust the bad bound again, and the
// resurrection/truncation window recovery just closed would reopen on the next crash.
// Returning the error lets shard construction REFUSE TO OPEN rather than serve a shard
// whose recovered bound is not durable. That is an intentional availability-for-
// integrity trade: a shard that cannot make its own recovery correction durable is
// one whose next restart could silently resurrect stale bytes, so a loud open failure
// (visible, retriable once the disk fault clears) is preferable to a silent one.
func (s *shard) flushRecoveredBoundsLocked(idx int) error {
	if !s.isMmap || s.region == nil {
		return nil
	}
	p := s.pages[idx]
	p.projectBounds(p.head(), p.tail())
	if err := s.msyncPageHeaderLocked(idx); err != nil {
		return fmt.Errorf("%w: msync recovered page header (page %d): %w", ErrReuseBarrier, idx, err)
	}
	return nil
}

// now returns the wall-clock time in ms for the non-apply expiry sites (client
// read filter, sweeper, warm-restart rebuild, Iterate), honoring an injected
// test clock. The apply path never calls this — it evaluates expiry against the
// explicit leader-stamped nowMs threaded through PutAt/GetAt, so its determinism
// is independent of any wall clock.
func (s *shard) now() uint64 {
	if fn := s.nowFn.Load(); fn != nil {
		return (*fn)()
	}
	return nowMs()
}

// advanceAppliedStamp folds stampMs into lastAppliedStampMs as a running MAX
// (#4 Phase B / B3b). Called from the STAMPED apply sites (getAtH / putAtH) with
// the explicit leader stamp. The CAS loop keeps it monotonic non-decreasing under
// concurrent applies; a stamp <= the current value is a no-op (including the
// stamp==0 legacy/first-epoch case, which leaves the logical clock at 0 so the
// replicated sweep stays a no-op). See the lastAppliedStampMs field doc for why
// the resulting value is deterministic and monotonic across replicas and failover.
func (s *shard) advanceAppliedStamp(stampMs uint64) {
	for {
		cur := s.lastAppliedStampMs.Load()
		if stampMs <= cur {
			return
		}
		if s.lastAppliedStampMs.CompareAndSwap(cur, stampMs) {
			return
		}
	}
}

// LastAppliedStampMs returns the shard's logical clock: the running max of the
// apply-stamps observed on the stamped apply path. Zero means no stamped apply has
// landed yet (apply-stamping disabled, or a freshly restored snapshot before its
// first stamped tail entry). The replicated sweeper reclaims iff exp <= this value.
func (s *shard) LastAppliedStampMs() uint64 { return s.lastAppliedStampMs.Load() }

// startSweeper starts the TTL sweeper goroutine (when TTLSweepIntervalMs > 0).
//
// The sweep CLOCK differs by mode (see sweepOnce):
//   - Non-replicated: the wall clock (s.now()) — the original single-node behavior.
//   - Replicated (#4 Phase B / B3b): the shard's LOGICAL clock (lastAppliedStampMs),
//     so every replica reclaims the SAME keys at the SAME logical point regardless
//     of wall-clock skew. This is what B3a (which turned the sweeper OFF under
//     replication) deferred; re-enabling it here — but driven by the deterministic
//     logical clock, never wall time — closes the B3a+B2 availability cliff where
//     expired ghost pages accumulated to MaxPagesPerShard and a committed write hit
//     cache.ErrFull → Phase A halt.
func (s *shard) startSweeper() {
	if s.cfg.TTLSweepIntervalMs > 0 {
		s.sweepWG.Add(1)
		go s.runSweeper()
	}
	// The free-page reserve keeps its own ticker on its own interval, started here so
	// both background passes have one launch site and one shutdown (Close closes
	// stopSweeper and waits on sweepWG, which covers both).
	s.startReserveSweeper()
}

func (s *shard) numPages() int {
	s.mu.RLock()
	n := len(s.pages)
	s.mu.RUnlock()
	return n
}

func hashKey(key []byte) uint64 { return xxhash.Sum64(key) }

// Get returns the value for key. Returns ErrNotFound if absent or expired.
//
// Under PolicyRingbufEvict the returned slice is a freshly-allocated copy the
// caller owns and may retain (and mutate) freely — so a zero-copy alias into the
// shared page store is never handed back on this policy. In mmap ringbuf mode
// eviction overwrites page bytes in place, so the copy is taken under the read
// lock; in heap ringbuf mode retired pages are frozen and reads are lock-free,
// but the copy is still made to honor the owned-slice contract. Under
// PolicyRejectWrites (no in-place overwrite ever) the returned slice aliases the
// page backing store for speed — callers must not retain it across subsequent
// writes to this shard; copy if needed.
func (s *shard) Get(key []byte) ([]byte, error) {
	return s.getH(key, hashKey(key))
}

// needsReadLockForGet reports whether the read path must take the shard read
// lock. The rule is one question: CAN A WRITER OVERWRITE LIVE PAGE BYTES ON THIS
// SHARD? Where it can, a lock-free read could both observe a torn value and race
// the writer at the byte level; where it cannot, the bytes behind a hit are
// immutable for the read's lifetime and the read needs no lock.
//
// A reject-writes shard never overwrites live bytes at all — at cap it rejects —
// so it always reads lock-free. Under PolicyRingbufEvict two things can rewrite
// live bytes:
//
//   - MMAP EVICTION. An mmap page object wraps a fixed region of the persisted
//     file and cannot be swapped for a fresh allocation, so eviction drains it in
//     place. Heap eviction does not: it RETIRES the page by swapping in a fresh
//     object, leaving the old one frozen, which is what lets heap ringbuf read
//     lock-free by default.
//   - SAME-SIZE UPDATES (Config.InPlaceSameSizeUpdate). A write overwrites the
//     entry already stored for its key — on a page that is live, not retired, and
//     under an index slot readers are actively resolving. That is precisely the
//     frozen-page invariant the lock-free heap path rests on, so enabling the
//     feature moves those shards onto the read-locked path — UNLESS they also
//     validate each read against a per-stripe version counter
//     (Config.InPlaceSeqlockReads), which detects the same hazard without
//     serialising every reader on one shared cache line. See cache/seqlock.go.
//
// mmap ringbuf is NOT released by the seqlock. Its reads hand back zero-copy
// aliases that outlive the read, and validating bytes DURING a read says nothing
// about overwriting them afterwards.
func (s *shard) needsReadLockForGet() bool {
	if s.cfg.AtCapPolicy != PolicyRingbufEvict {
		return false
	}
	if s.isMmap {
		return true
	}
	return s.cfg.InPlaceSameSizeUpdate && !s.seqlockReads()
}

// nextGen returns the next page generation. Called under mu or during
// single-threaded construction. Wraps after 65536 page allocations/retires per
// shard; a false generation match would require ~65536 retires of one slot to
// elapse inside a single reader's load→check window (nanoseconds), which is not
// physically reachable — see slabref.go and the ABA note in the v2 design.
func (s *shard) nextGen() uint16 {
	g := s.genCounter
	s.genCounter++
	return g
}

// getH is Get with a precomputed key hash, so callers that already had
// to hash for shard selection (Cache.Get) don't pay xxhash twice. This is the
// CLIENT read path: it evaluates expiry against the wall clock (s.now()) and,
// on a replicated shard, SUPPRESSES the physical index-slot removal so a
// logically-expired key is filtered (returned as a miss) without a
// nondeterministic mutation that would diverge the committed key set from a peer
// ticking at a different wall time (#4 Phase B / B3a). This suppression STAYS in
// place under B3b: physical reclamation on a replicated shard is solely the
// logical-clock sweeper's job (exp <= lastAppliedStampMs), never a wall-clock
// client read. Non-replicated shards keep the lazy drop-on-read reclamation exactly
// as before.
func (s *shard) getH(key []byte, h uint64) ([]byte, error) {
	v, _, err := s.getCore(key, h, s.now(), !s.cfg.Replicated)
	return v, err
}

// getWithExpiryH is getH that ALSO surfaces the entry's stored absolute expiry
// (ms since epoch; 0 = no expiry) — the read primitive the ttl / persist / incr_ex
// ops need. It shares getCore, so the wall-clock filtering, the copy-vs-alias
// return contract, and the ErrNotFound-on-absent-or-expired behaviour are all
// identical to getH; the only difference is the extra expiry return.
func (s *shard) getWithExpiryH(key []byte, h uint64) ([]byte, uint64, error) {
	return s.getCore(key, h, s.now(), !s.cfg.Replicated)
}

// getWithExpiryAtH is getAtH that ALSO surfaces the entry's stored absolute
// expiry — the apply-path counterpart of getWithExpiryH, judging liveness against
// the explicit leader-stamped nowMs. See getAtH for the stamp-fold rationale.
func (s *shard) getWithExpiryAtH(key []byte, h, nowMs uint64) ([]byte, uint64, error) {
	s.advanceAppliedStamp(nowMs)
	return s.getCore(key, h, nowMs, true)
}

// getAtH is the APPLY-path Get: expiry is evaluated against the explicit
// leader-stamped nowMs (not the wall clock), and physical removal is ALWAYS
// permitted because tombstoning an expired key here is a committed-state
// decision every replica makes identically from the same stamped clock (#4 Phase
// B / B1). Handlers reach it via TxContext.Get when an apply stamp is present.
func (s *shard) getAtH(key []byte, h, nowMs uint64) ([]byte, error) {
	// Fold the leader stamp into the shard's logical clock (#4 Phase B / B3b) BEFORE
	// serving the read. getAtH is reached ONLY from a committed, stamped apply (the
	// read-only Call path uses getH), so nowMs is identical and identically-ordered
	// across replicas — the max stays deterministic. Advancing here (not just on
	// putAtH) lets a stamped read alone drive reclamation forward on a read-mostly
	// workload.
	s.advanceAppliedStamp(nowMs)
	v, _, err := s.getCore(key, h, nowMs, true)
	return v, err
}

// getLockedCore is the read-locked probe: it takes the shard read lock for the
// probe AND the value copy, so the bytes cannot move underneath either.
//
// It serves the two configurations that can rewrite live page bytes and are not
// validating reads against a version counter: mmap ringbuf, whose eviction
// drains its fixed persisted region in place, and a heap ringbuf shard doing
// in-place same-size updates with the seqlock off. It is ALSO where a seqlock
// read lands when its retry budget is spent — taking the lock excludes the
// writer outright instead of racing it, which is what makes that read terminate.
//
// A lock-free read of these shards could both observe a torn value AND race the
// writer at the byte level; a seqlock addresses the first and not the second,
// which is why this path exists at all (see cache/seqlock.go).
func (s *shard) getLockedCore(key []byte, h, now uint64, allowPhysicalRemove bool) ([]byte, uint64, error) {
	s.mu.RLock()
	t := s.tab.Load()
	v, exp, ref, st := t.get(s, key, h)
	var vCopy []byte
	if st == lkHit {
		// Copy while the lock is held so a later overwrite can't tear it.
		vCopy = make([]byte, len(v))
		copy(vCopy, v)
	}
	s.mu.RUnlock()
	return s.resolveLookup(vCopy, key, h, exp, ref, st, now, allowPhysicalRemove)
}

// getIntoLockedCore is getLockedCore appending into the caller's buffer. On any
// non-hit it returns dst at its original length, matching getIntoCore.
func (s *shard) getIntoLockedCore(dst, key []byte, h, now uint64, allowPhysicalRemove bool) ([]byte, uint64, error) {
	s.mu.RLock()
	t := s.tab.Load()
	v, exp, ref, st := t.get(s, key, h)
	var out []byte
	if st == lkHit {
		out = append(dst, v...) // copy while the lock is held
	}
	s.mu.RUnlock()
	out, exp, err := s.resolveLookup(out, key, h, exp, ref, st, now, allowPhysicalRemove)
	if err != nil {
		return dst, 0, err
	}
	return out, exp, nil
}

// resolveLookup turns a completed probe into the caller's answer: it applies the
// corrupt/miss accounting and the expiry filter that every read path owes,
// whichever protocol produced the probe. v must already be an owned copy on the
// paths that require one — this runs after the lock (or the version check) has
// been released, so it must not touch page bytes.
func (s *shard) resolveLookup(v, key []byte, h, exp uint64, ref slabRef, st lookupStatus, now uint64, allowPhysicalRemove bool) ([]byte, uint64, error) {
	switch st {
	case lkCorrupt:
		s.corrupt.Add(1)
		s.misses.Add(1)
		return nil, 0, ErrNotFound
	case lkMiss:
		s.misses.Add(1)
		return nil, 0, ErrNotFound
	}
	if isExpired(exp, now) {
		if allowPhysicalRemove {
			s.dropExpiredLocked(key, h, ref, now)
		}
		s.misses.Add(1)
		return nil, 0, ErrNotFound
	}
	return v, exp, nil
}

// getCore is the shared Get implementation. now is the clock expiry is judged
// against; allowPhysicalRemove gates whether an expired-on-read entry's index
// slot is tombstoned (client path on a replicated shard passes false — filter
// only; apply path and all non-replicated reads pass true). The lock-free vs
// read-locked branch and the copy-vs-alias return contract are unchanged from
// the original getH — only the clock and the drop decision are parameterized.
//
// The read is lock-free except for mmap-backed ringbuf shards (see
// needsReadLockForGet): reject-writes never overwrites live bytes, and heap
// ringbuf retires pages by swapping in a fresh frozen object gated by the
// slabRef generation, so neither can race a writer. Only mmap ringbuf overwrites
// its fixed region in place and takes the read lock for the probe and value
// copy. reject-writes returns a zero-copy alias; ringbuf returns an owned copy.
func (s *shard) getCore(key []byte, h, now uint64, allowPhysicalRemove bool) ([]byte, uint64, error) {
	s.gets.Add(1)

	if s.seqlockReads() {
		if out, exp, ref, st, ok := s.getSeqRetry(nil, key, h); ok {
			return s.resolveLookup(out, key, h, exp, ref, st, now, allowPhysicalRemove)
		}
		// Retry budget spent. Fall through to the read-locked probe below, which
		// excludes the writer outright rather than racing it, so the read always
		// terminates. See seqlockMaxRetries.
		return s.getLockedCore(key, h, now, allowPhysicalRemove)
	}

	if s.needsReadLockForGet() {
		return s.getLockedCore(key, h, now, allowPhysicalRemove)
	}

	// Lock-free path: reject-writes (heap+mmap) and heap-mode ringbuf. For heap
	// ringbuf the generation gate in indexTable.get rejects a stale ref without
	// reading a swapped-in page's bytes, and retired pages are frozen, so the
	// bytes behind a hit are immutable for the read's lifetime — see doc.go.
	t := s.tab.Load()
	v, exp, ref, st := t.get(s, key, h)
	switch st {
	case lkCorrupt:
		s.corrupt.Add(1)
		s.misses.Add(1)
		return nil, 0, ErrNotFound
	case lkMiss:
		s.misses.Add(1)
		return nil, 0, ErrNotFound
	}
	if isExpired(exp, now) {
		if allowPhysicalRemove {
			s.dropExpiredLocked(key, h, ref, now)
		}
		s.misses.Add(1)
		return nil, 0, ErrNotFound
	}
	if s.cfg.AtCapPolicy == PolicyRingbufEvict {
		// Ringbuf callers own (and may mutate) the returned slice, and the frozen
		// page bytes must never be mutated, so hand back a copy rather than the
		// zero-copy alias the reject-writes contract permits. No lock: v aliases a
		// frozen/append-only page whose bytes at this range are immutable.
		vCopy := make([]byte, len(v))
		copy(vCopy, v)
		return vCopy, exp, nil
	}
	return v, exp, nil
}

// getIntoH is getH that appends the value into dst instead of returning a fresh
// allocation, so a hot-loop caller reusing one buffer pays zero allocations per
// hit. The value is always copied into dst. Under PolicyRingbufEvict the copy
// runs under the read lock (see getH) so a concurrent evict+Write cannot tear
// it; under PolicyRejectWrites it is lock-free. On miss/expiry dst is returned
// unchanged (its original length) with ErrNotFound.
func (s *shard) getIntoH(dst, key []byte, h uint64) ([]byte, error) {
	out, _, err := s.getIntoCore(dst, key, h, s.now(), !s.cfg.Replicated)
	return out, err
}

// getIntoWithExpiryH is getIntoH that ALSO surfaces the entry's stored absolute
// expiry, mirroring getWithExpiryH's relationship to getH.
func (s *shard) getIntoWithExpiryH(dst, key []byte, h uint64) ([]byte, uint64, error) {
	return s.getIntoCore(dst, key, h, s.now(), !s.cfg.Replicated)
}

// getIntoWithExpiryAtH is the APPLY-path counterpart: expiry is judged against
// the explicit leader-stamped nowMs and physical removal is always permitted,
// exactly as getWithExpiryAtH does for the allocating read. Keeping the clock
// explicit is what lets the apply path use the pooled read without reintroducing
// a wall-clock dependency into replicated state.
func (s *shard) getIntoWithExpiryAtH(dst, key []byte, h, nowMs uint64) ([]byte, uint64, error) {
	s.advanceAppliedStamp(nowMs)
	return s.getIntoCore(dst, key, h, nowMs, true)
}

// getIntoCore is getCore's append-into-dst twin: same probe, same clock and
// reclamation rules, but the value is copied into the caller's buffer instead
// of a fresh allocation. It is ALWAYS a copy, never an alias, so the owned-slice
// contract is identical to getCore's.
func (s *shard) getIntoCore(dst, key []byte, h, now uint64, allowPhysicalRemove bool) ([]byte, uint64, error) {
	s.gets.Add(1)

	if s.seqlockReads() {
		if out, exp, ref, st, ok := s.getSeqRetry(dst, key, h); ok {
			v, e, err := s.resolveLookup(out, key, h, exp, ref, st, now, allowPhysicalRemove)
			if err != nil {
				return dst, 0, err // miss/expiry leaves dst at its original length
			}
			return v, e, nil
		}
		return s.getIntoLockedCore(dst, key, h, now, allowPhysicalRemove)
	}

	if s.needsReadLockForGet() {
		return s.getIntoLockedCore(dst, key, h, now, allowPhysicalRemove)
	}

	// Lock-free path: reject-writes (heap+mmap) and heap-mode ringbuf. The value
	// is always copied into dst, so the owned-slice contract holds for both.
	t := s.tab.Load()
	v, exp, ref, st := t.get(s, key, h)
	switch st {
	case lkCorrupt:
		s.corrupt.Add(1)
		s.misses.Add(1)
		return dst, 0, ErrNotFound
	case lkMiss:
		s.misses.Add(1)
		return dst, 0, ErrNotFound
	}
	if isExpired(exp, now) {
		if allowPhysicalRemove {
			s.dropExpiredLocked(key, h, ref, now)
		}
		s.misses.Add(1)
		return dst, 0, ErrNotFound
	}
	return append(dst, v...), exp, nil
}

// dropExpiredLocked removes the entry for h that a reader just saw expire at
// `now`, unless it has been replaced in the meantime. Every read path reaches it
// AFTER releasing the lock (or the version check), so a writer can run in
// between, and the whole job of this function is to decide whether the entry it
// is about to tombstone is still the one the reader judged.
//
// cur == ref IS NO LONGER THAT DECISION ON ITS OWN. It was, for as long as every
// Put allocated a new copy: an overwrite always moved the entry, so a matching
// ref proved nothing had replaced it. A same-size in-place update
// (Config.InPlaceSameSizeUpdate) keeps the page, the offset AND the generation,
// so the ref is byte-identical after a rewrite and the comparison can no longer
// tell "untouched" from "replaced where it lay". A Put that refreshes an expired
// key's TTL would otherwise be undone by a read that saw the PREVIOUS value
// expire: the key vanishes though Put returned nil, and — worse, because nothing
// repairs it — onRemove fires for a key that is live, dropping its postings from
// every derived index.
//
// So the entry is RE-READ under the lock and only removed if it is STILL expired
// against the same `now` the reader judged it by. Re-reading is what makes this
// independent of how the replacement was written, so it also holds for any future
// write path that preserves a ref. The clock is the CALLER'S, deliberately: a
// reader that decided "expired" at its own clock reading must not have that
// decision re-litigated against a later one, or a sweeper-style drop could remove
// an entry the reader would have returned.
//
// `key` is the CALLER'S key. It is compared against the stored one rather than
// trusted, because the re-read can land on a different key's entry: the index is
// keyed by 64-bit hash, and a slot can be taken over between the probe and here.
func (s *shard) dropExpiredLocked(key []byte, h uint64, ref slabRef, now uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	t := s.tab.Load()
	slot, cur, ok := t.findSlot(h)
	if !ok || cur != ref {
		return // gone, or replaced by a copy somewhere else
	}
	idx := int(cur.pageIdx())
	if idx >= len(s.pages) {
		return
	}
	p := s.pages[idx]
	if p == nil || p.gen != cur.gen() {
		return // the page was retired under this ref
	}
	storedKey, _, storedExp, err := p.Read(cur.offset())
	if err != nil || !bytes.Equal(storedKey, key) {
		return // unreadable, or this slot is somebody else's now
	}
	if !isExpired(storedExp, now) {
		return // a writer refreshed it in place; it is live and must be kept
	}
	t.tombstone(slot)
	s.expirations.Add(1)
	s.fireOnRemove(key)
}

// Put inserts or replaces the entry for key with the given value and TTL.
// A TTL of zero means no expiry.
func (s *shard) Put(key, value []byte, ttl time.Duration) error {
	return s.putH(key, value, ttl, hashKey(key))
}

// putH is Put with a precomputed key hash. See getH. It stamps the absolute
// expiry from the WALL clock — the client/single-node write path (and the
// unstamped/legacy apply path). With no injected clock (the production default)
// it uses expiryAt(ttl) verbatim, byte-identical to pre-B1 behavior; an injected
// test clock (s.nowFn) routes through expiryAtFrom so a test can skew the
// wall-clock write path per replica to demonstrate the divergence B1 removes.
func (s *shard) putH(key, value []byte, ttl time.Duration, h uint64) error {
	var exp uint64
	if fn := s.nowFn.Load(); fn != nil {
		exp = expiryAtFrom(ttl, (*fn)())
	} else {
		exp = expiryAt(ttl)
	}
	return s.putAtExpLocked(key, value, exp, h)
}

// putAtH is the APPLY-path Put: the absolute expiry is computed as nowMs + ttl
// from the explicit leader-stamped clock, so every replica applying the same
// committed entry stores the SAME absolute expiry regardless of its wall clock
// (#4 Phase B / B1). Handlers reach it via TxContext.Put when an apply stamp is
// present.
func (s *shard) putAtH(key, value []byte, ttl time.Duration, nowMs uint64, h uint64) error {
	// Advance the shard's logical clock with the leader stamp (#4 Phase B / B3b).
	// See getAtH: this is a committed stamped apply, so nowMs is deterministic and
	// identically ordered on every replica, keeping the max identical.
	s.advanceAppliedStamp(nowMs)
	return s.putAtExpLocked(key, value, expiryAtFrom(ttl, nowMs), h)
}

// putAbsH inserts key with a pre-computed ABSOLUTE expiry (ms since epoch, 0 =
// none), bypassing any TTL→expiry conversion. Snapshot restore uses it to
// install the exact expiry the snapshot recorded, so two followers restoring the
// same snapshot at different wall times produce logically byte-identical state
// (identical key/value/exp set; #4 Phase B / B1). h is the precomputed key hash.
//
// It deliberately does NOT advance lastAppliedStampMs (#4 Phase B / B3b): the
// argument is an ABSOLUTE expiry, not a leader stamp, so there is no logical-clock
// value to fold in. Restore leaves the logical clock where it was (0 on a cold
// restore) and the committed log tail replayed after restore re-advances it
// identically on every replica, so the sweep stays deterministic without the
// snapshot carrying the clock. See the lastAppliedStampMs field doc.
func (s *shard) putAbsH(key, value []byte, expiryMs uint64, h uint64) error {
	return s.putAtExpLocked(key, value, expiryMs, h)
}

// inPlaceEligible reports whether this SHARD may overwrite a live entry where it
// lies. It is the per-shard half of the same-size update's guards; the per-write
// half is inPlaceTargetLocked. Both must pass, and neither subsumes the other.
//
// Immutable for the shard's life — the config and the storage mode are both
// fixed at construction — so this is three field loads and safe to call from
// anywhere, lock or no lock.
func (s *shard) inPlaceEligible() bool {
	// GUARD 1 — OPT-IN. Off by default. Enabling it costs every read on the shard
	// the read lock (see needsReadLockForGet), which is not a price to impose on a
	// workload that would not get the write-side benefit back.
	if !s.cfg.InPlaceSameSizeUpdate {
		return false
	}
	// GUARD 2 — RINGBUF ONLY. A PolicyRejectWrites shard's reads return a ZERO-COPY
	// ALIAS into the page bytes, and that alias outlives the read: it escapes to a
	// network response writer, which is why online compaction has to quarantine a
	// retired extent for twice the write deadline before recycling it. Overwriting
	// live bytes under such an alias would change a value a caller is still
	// holding. Ringbuf reads hand back an owned copy, so no alias survives.
	if s.cfg.AtCapPolicy != PolicyRingbufEvict {
		return false
	}
	// GUARD 3 — HEAP BACKING. An mmap page is the DURABLE copy, and the cost of a
	// torn write is NOT confined to the key being written.
	//
	// Compare the two write shapes for an ENTRY-DATA tear, i.e. one where the page's
	// persisted head/tail survives. An append writes only at the page TAIL, so
	// recovery discards the torn write and nothing beyond it. An in-place write
	// tears at an arbitrary offset INSIDE a live page, and rebuildIndexFromPages
	// answers a CRC failure by truncating the page at that offset and abandoning the
	// rest of it — it cannot simply skip the bad entry, because EvictFront frames
	// entries from raw bytes with no CRC, so the torn region has to be excluded from
	// every future eviction walk. The loss is therefore every entry AFTER the tear on
	// that page: unrelated keys, durable long before the write that tore.
	//
	// The page FRAMING is a second failure mode, and it belongs to the OTHER shape:
	// only append moves it. Write persists the tail after encoding, so a crash torn
	// mid-setTail can leave head/tail out of range and recovery resets the whole
	// page; WriteAt never calls setTail and only reads head/tail to bound itself, so
	// an in-place write cannot tear the framing at all.
	//
	// So each shape has its own rare whole-page failure. What is NOT symmetric is the
	// common case: an in-place tear takes the rest of its page with it, while an
	// append tear that leaves the framing intact costs only the write in flight — and
	// the previous version of the key is still framed and recoverable, which an
	// in-place write has already destroyed.
	//
	// It is also why the boundary cannot be redefined to make this safe — read
	// rebuildIndexFromPages before concluding otherwise. Heap pages are not
	// persisted, so there is nothing a torn write could cost that the process
	// dying has not already cost.
	return !s.isMmap
}

// inPlaceTarget is the entry a same-size update will overwrite, plus the index
// coordinates that located it. The coordinates travel together deliberately: a slot
// index is meaningless against any table but the one it was found in, and the ref is
// what distinguishes THAT record from whatever else may come to occupy the slot, so
// handing back the slot without both is handing back an identity that cannot be
// checked (see indexTable.setVisited).
type inPlaceTarget struct {
	p    *page
	off  uint32
	tab  *indexTable
	slot uint64
	ref  slabRef
}

// inPlaceTargetLocked resolves the entry a same-size update of (key, value)
// would overwrite: the page object and offset of the copy the index currently
// points at, when that copy's framing is byte-identical to what the new write
// would produce. ok=false means the write has no such target and must append.
//
// An entry is [entryHeaderSize][key][value] and the header is fixed-width, so
// for the SAME key and an identical value LENGTH the new framing occupies
// exactly the same bytes as the stored one. Everything the index holds — page,
// offset, generation — therefore stays correct across the overwrite, which is
// what makes the slot upsert, the page allocation and any eviction behind it
// unnecessary.
//
// Must be called with s.mu held for writing. It is the per-WRITE half of the
// guards (inPlaceEligible is the per-shard half). GUARD 4 — the index slot still
// points at this exact copy — is guards 4a..4d together: a ref that survives all
// four IS the current copy of this key, which is what makes the overwrite safe
// without touching the index. GUARD 5 is the size equality the whole scheme rests
// on.
//
//	4a. the key is indexed at all;
//	4b. its ref lands on an allocated page slot that is still the GENERATION the
//	    ref was minted against — heap ringbuf eviction retires a page by swapping
//	    in a fresh object, so a stale ref would otherwise resolve against a slab
//	    under active append;
//	4c. the framed entry decodes, and its key EQUALS the caller's — the index is
//	    keyed by 64-bit hash, so a collision resolves to a DIFFERENT key's entry
//	    and overwriting it would destroy an unrelated live record;
//	4d. the entry lies wholly inside the page's live band [head, tail). Outside it
//	    the bytes are either already evicted or unwritten free space, neither of
//	    which an index-current entry can occupy; refusing is cheaper than
//	    reasoning about how a violation could arise.
//
// The index coordinates come back with it (see inPlaceTarget), for the SIEVE
// reference hint and nothing else: an in-place rewrite is the one write that leaves
// no other trace of having happened, so the index slot is the only place it can
// record that the key is in use.
func (s *shard) inPlaceTargetLocked(key, value []byte, h uint64) (inPlaceTarget, bool) {
	t := s.tab.Load()
	slot, ref, found := t.findSlot(h)
	if !found {
		return inPlaceTarget{}, false // (4a) key absent
	}
	idx := int(ref.pageIdx())
	if idx >= len(s.pages) {
		return inPlaceTarget{}, false // (4b) ref outside the allocated page list
	}
	p := s.pages[idx]
	if p == nil || p.gen != ref.gen() {
		return inPlaceTarget{}, false // (4b) page retired and replaced since the ref was minted
	}
	off := ref.offset()
	storedKey, storedVal, storedExp, err := p.Read(off)
	if err != nil {
		return inPlaceTarget{}, false // (4c) unreadable framing
	}
	if !bytes.Equal(storedKey, key) {
		return inPlaceTarget{}, false // (4c) hash collision: a different key's entry
	}
	if len(storedVal) != len(value) {
		return inPlaceTarget{}, false // (5) different size: the framing would not line up
	}
	// OCCUPANCY: the band has to contain what the STORED entry takes up on the page.
	//
	// AND IT IS COMPUTED FROM THE NEW VALUE, NOT THE STORED ONE. That is correct
	// today for one reason only: guard 5 immediately above has just proved
	// len(storedVal) == len(value), so the two spans are the same number and it
	// does not matter which is measured. The check does NOT stand on its own.
	//
	// Anything that relaxes guard 5 — admitting a rewrite whose value length merely
	// FITS the stored entry rather than equalling it — invalidates this line and
	// must recompute the span from len(storedVal). Otherwise a shrinking rewrite
	// measures a band requirement smaller than the entry actually there and passes
	// a check the stored entry itself would fail.
	if int(off) < p.head() || int(off)+entrySpan(len(key), len(value), true) > p.tail() {
		return inPlaceTarget{}, false // (4d) not inside the page's live band
	}
	// GUARD 6 — the stored copy must still be LIVE. An expired copy is dead
	// weight, and rewriting it where it lies pins those bytes instead of letting
	// the append path leave them behind to be reclaimed. Appending also gives the
	// refreshed key a NEW ref, which is what a concurrent expired-on-read drop
	// compares against.
	//
	// That second effect is hygiene, NOT the guarantee. This shard has an
	// injectable clock and an apply path stamped by a leader, so a writer can
	// judge the stored copy live at the very moment a reader judges it expired,
	// and no test here can rule that skew out. dropExpiredLocked re-reads the
	// entry under the lock for exactly that reason; this guard only keeps the
	// common case from reaching it.
	//
	// The clock is read ONLY when the stored copy actually carries an expiry, so
	// the ordinary no-TTL write pays nothing for it.
	if storedExp != 0 && isExpired(storedExp, s.now()) {
		return inPlaceTarget{}, false // (6) the stored copy is already dead
	}
	return inPlaceTarget{p: p, off: off, tab: t, slot: slot, ref: ref}, true
}

// putAtExpLocked is the shared write body: it takes the already-resolved
// absolute expiry (exp) and performs the page write + index upsert. putH,
// putAtH, and putAbsH differ ONLY in how they derive exp — the storage path is
// identical.
func (s *shard) putAtExpLocked(key, value []byte, exp uint64, h uint64) error {
	s.puts.Add(1)

	s.mu.Lock()
	defer s.mu.Unlock()

	// SAME-SIZE UPDATE. When this shard is eligible and the new entry would be
	// framed byte-identically to the copy already stored for the key, overwrite
	// that copy where it lies. Nothing else changes: the index slot still names
	// the same page, offset and generation, so there is no upsert, no page
	// allocation, no eviction — and, the point of the exercise, no dead previous
	// version left behind. Any refusal falls through to the append path below,
	// which is unchanged.
	if s.inPlaceEligible() {
		if tgt, ok := s.inPlaceTargetLocked(key, value, h); ok {
			// Stamp the next sequence exactly as the append path does — one per
			// STORED write — but only commit it once the bytes are down, so a
			// refused overwrite leaves the shard's sequence untouched.
			seq := s.writeSeq + 1
			// The version bump brackets the payload stores so a lock-free reader can
			// tell that it saw them half-done. It is a no-op on a shard whose readers
			// take the read lock instead, which the write lock already excludes.
			werr := s.bumpVersionLocked(tgt.p, tgt.off, func() error {
				return tgt.p.WriteAt(tgt.off, key, value, exp, makeMeta(seq, false))
			})
			if werr == nil {
				s.writeSeq = seq
				s.inPlaceUpdates.Add(1)
				// AN UPDATE IS AN ACCESS (cache/sieve.go). It is also the ONLY signal
				// this write leaves: the record does not move, so its page age is
				// unchanged and the rotation reaches it on schedule however hot it is.
				// Marking here restores exactly the standing an APPENDING rewrite gets
				// for free by landing on the newest page — it does not grant anything
				// the append path never had.
				//
				// THE COST OF THIS LANDS ON THE WRITE LOCK, NOT ON READS, which is the
				// opposite of where it was expected. Marking on the READ path measures
				// free — it is one load of a word the probe has already fetched, on a
				// per-slot line rather than a shared one. This mark is inside s.mu, and
				// on a shard whose writes are serialised there that critical section is
				// the throughput limiter, so what is added here is amplified. Keep it
				// minimal: tab is the table the target was resolved through, so no
				// second s.tab load, and setVisited returns after one load once the
				// mark is already set.
				if s.sieve {
					tgt.tab.setVisited(tgt.slot, tagFor(h), tgt.ref)
				}
				regionNoteInPlace(s, tgt.p) // compiled out unless the measurement build tag is set
				// NO fireOnRemove: nothing was removed. The key is still live, at the
				// same address, and the hook reports REMOVALS — firing it here would
				// drop a live key's postings from every derived index.
				return nil
			}
		}
	}

	// Find a page with enough tail room; lazily allocate or evict as needed.
	// OCCUPANCY: this authorises the page.Write a few lines below — capacity/write
	// rule, see entrySpan.
	pageIdx, err := s.findOrMakePageLocked(entrySpan(len(key), len(value), true))
	if err != nil {
		return err
	}
	// Stamp the entry with the next write sequence. This is the entire runtime cost
	// of the warm-restart fix on the write path: one increment and one 8-byte store,
	// both under a lock we already hold.
	s.writeSeq++
	off, _, werr := s.pages[pageIdx].Write(key, value, exp, makeMeta(s.writeSeq, false))
	if werr != nil {
		return werr
	}
	t := s.tab.Load()
	ref := makeSlabRef(uint16(pageIdx), s.pages[pageIdx].gen, off) //nolint:gosec // pageIdx bounded by MaxPagesPerShard (≤65535)
	slot, inserted := t.upsert(h, ref)
	// AN UPDATE MARKS, A FIRST INSERTION DOES NOT (cache/sieve.go). The asymmetry is
	// the whole discrimination: a key written once and never touched again must arrive
	// unvisited, or a write-once stream marks everything it inserts and the hint stops
	// distinguishing anything. Marking BEFORE the rehash, so a hint earned by this
	// write is one rehashed() carries across rather than one it never sees.
	if s.sieve && !inserted {
		t.setVisited(slot, tagFor(h), ref)
	}
	s.rehashIfOverThresholdLocked(t)
	return nil
}

// Del removes the entry for key. Returns true if the entry was present. See delH
// for when it can return an error.
func (s *shard) Del(key []byte) (bool, error) {
	return s.delH(key, hashKey(key))
}

// needsDurableTombstone reports whether a delete on this shard must be RECORDED
// ON THE PAGE rather than only in the index. The rule is simply PERSISTENT ⇒
// DELETES ARE DURABLE: every mmap shard qualifies, whatever its capacity policy.
//
// HEAP shards (!isMmap) are the only exception, and only because there is nothing
// to be durable against: rebuildIndexFromPages runs in mmap mode alone, so an
// index-only delete is already permanent for them. Writing a tombstone would spend
// capacity recording something nothing will ever read back.
//
// Every other shard rebuilds its index from page bytes at open, so a deleted key
// whose entry is still framed on a page is re-indexed and COMES BACK (#12B) unless
// the removal is itself in those bytes. Under PolicyRingbufEvict no less than
// under PolicyRejectWrites: eviction does recycle pages, but "eventually" is not a
// bound — a key deleted from a rarely-touched page can outlive any number of
// restarts, and the resurrection is silent when it happens.
//
// The policy therefore does not enter into it. Resurrecting deleted data is a
// correctness violation, and no capacity policy amounts to consenting to one; a
// uniform invariant is also far easier to reason about than a policy-dependent
// one. Recording the delete may cost a ringbuf shard an eviction, which is the
// thing that shard is configured to do.
func (s *shard) needsDurableTombstone() bool {
	return s.isMmap
}

// delH is Del with a precomputed key hash. See getH.
//
// On a PERSISTENT shard (any capacity policy — see needsDurableTombstone) it
// APPENDS A TOMBSTONE ENTRY before tombstoning the index slot. Without it the
// delete lived only in memory: the warm-restart rebuild re-indexed the key
// straight off the page bytes and it came back (#12B) — and on a replicated shard
// a peer that had not restarted disagreed with it forever.
//
// An in-place flag on the existing entry was rejected: flipping a bit invalidates
// that entry's CRC, and a crash between the two stores leaves a bad CRC, which
// makes rebuildIndexFromPages TRUNCATE THE PAGE TAIL — losing every later entry in
// the page to record one delete. An appended record is crash-safe by construction:
// it is either fully written with a valid CRC or rejected as a torn tail.
//
// It can therefore return an error where it previously could not: ErrFull on a
// reject-writes shard with no room for the record, or ErrCannotEvict on a ringbuf
// shard with nothing left to evict. That is the honest answer, not a regression in
// disguise: a full reject-writes shard could never be freed by deleting anyway — Del has never returned bytes to the
// page store — so this turns "the delete silently succeeds and the next write
// halts" into "the delete halts". The replicated apply path classifies ErrFull as
// fatal, and halt → restart → cold compaction is the designed remedy, which now
// genuinely reclaims (the tombstone lets compaction drop the key's whole byte
// history).
func (s *shard) delH(key []byte, h uint64) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	t := s.tab.Load()
	slot, ref, ok := t.findSlot(h)
	if !ok {
		return false, nil
	}
	page := s.pages[ref.pageIdx()]
	storedKey, _, _, err := page.Read(ref.offset())
	if err != nil {
		// Corrupt entry: treat as missing. Cleaning it up is the sweeper's job.
		return false, nil
	}
	if !bytes.Equal(storedKey, key) {
		// Hash collision: the stored entry belongs to a different key. Do not delete.
		return false, nil
	}
	if s.needsDurableTombstone() {
		// A tombstone is a normal entry with an empty value and the flag set. It
		// carries the next write sequence, so it beats every earlier copy of the key
		// in the rebuild's max-seq contest — and loses to any LATER Put, which is what
		// makes delete-then-rewrite work.
		//
		// findOrMakePageLocked's "entry larger than PageSize" rejection is UNREACHABLE
		// from here, and is deliberately left as the plain deterministic error it is
		// (classifyApplyErr rates it classAdvance, which is correct — every replica
		// rejects an oversized entry identically). Getting this far required findSlot
		// plus bytes.Equal to prove the key is index-current, so the Put that indexed
		// it passed the very same guard with a STRICTLY LARGER entry (this key plus a
		// value), and PageSize cannot change under a live shard — validateHeader
		// rotates a file whose header disagrees aside instead of reopening it at a
		// different geometry.
		// OCCUPANCY: this authorises the tombstone page.Write below — capacity/write
		// rule, see entrySpan. The record is this key with an empty value.
		pageIdx, ferr := s.findOrMakePageLocked(entrySpan(len(key), 0, true))
		if ferr != nil {
			return false, ferr
		}
		s.writeSeq++
		if _, _, werr := s.pages[pageIdx].Write(key, nil, 0, makeMeta(s.writeSeq, true)); werr != nil {
			return false, werr
		}
		// RE-RESOLVE THE INDEX. `slot` was captured before the append, and on a
		// ringbuf shard the append can move the ground under it: findOrMakePageLocked
		// falls through to evictUntilFitsLocked, which drops the index slot of every
		// entry it drains — possibly this key's. Tombstoning a slot eviction already
		// tombstoned would double-count the table's live/tomb bookkeeping, and any
		// future rehash on this path would leave the captured slot pointing into a
		// table that is no longer live, so the delete would silently fail to remove
		// the key from the index and it would stay readable until the next restart.
		// Re-running findSlot on a freshly loaded table is the only form that holds
		// for every policy. (Reject-writes shards never evict, so for them this
		// re-resolves to exactly the slot that was captured.)
		t = s.tab.Load()
		slot, ref, ok = t.findSlot(h)
		if ok {
			// The re-resolved entry must still be OUR key: a drained page can leave a
			// slot pointing at bytes a later write has since reused.
			if k2, _, _, rerr := s.pages[ref.pageIdx()].Read(ref.offset()); rerr != nil || !bytes.Equal(k2, key) {
				ok = false
			}
		}
		if !ok {
			// The key is no longer index-current. Nothing is left to strip, the appended
			// record is harmless (the rebuild finds a delete for a key with no live
			// copy), and the entry DID exist when the caller asked — so report the
			// delete as having happened.
			//
			// NO onRemove CALL HERE, and only ONE of the three routes to this branch
			// has already notified:
			//   (a) eviction dropped this key's slot while making room for the tombstone
			//       record — that eviction fired the hook from inside its own
			//       cur == ref guard, so the key is already reported;
			//   (b) the re-read of the re-resolved entry FAILED — a keyless corrupt
			//       slot, which nothing anywhere reports (see sweepIndex);
			//   (c) the re-resolved slot holds a DIFFERENT key that took the hash over,
			//       so this key's slot is simply gone, unreported.
			// (b) and (c) therefore leave a STALE posting behind. That is within the
			// hook's contract: a stale posting costs one wasted candidate that
			// verify-on-read discards, and the reconcile pass clears it. The thing that
			// must never happen — dropping a LIVE key's posting — cannot happen here.
			s.dels.Add(1)
			return true, nil
		}
	}
	t.tombstone(slot)
	s.dels.Add(1)
	// The slot just removed is this key's — findSlot plus the bytes.Equal guard
	// above (and, on the durable path, the re-resolve) proved it — so a derived
	// index may drop the key's postings.
	s.fireOnRemove(key)
	return true, nil
}

// snapshot returns a Stats snapshot. Caller-side concurrency safe.
func (s *shard) snapshot() Stats {
	reclaimable := s.reclaimableBytesForStats()
	// Hits is derived: every read bumps gets on entry and bumps misses on any
	// non-returning path, so Hits = Gets - Misses. Load misses BEFORE gets so an
	// op in flight between the two loads can only leave gets >= misses (gets is
	// always bumped first), never underflowing the subtraction.
	misses := s.misses.Load()
	gets := s.gets.Load()
	// Same discipline for the relocation pair, which the write path publishes
	// bytes-first: load the COUNT first so a relocation in flight between the two
	// loads can only leave bytes >= what the count implies, never the impossible
	// pair (records relocated, zero bytes moved). Note the Evictions/EvictionsLive
	// pair below does NOT hold to this — it is read in the order the struct lists
	// it, against a write path that bumps evictionsLive first — so copy the
	// gets/misses shape here, not that one.
	relocations := s.evictRelocations.Load()
	relocatedBytes := s.evictRelocatedBytes.Load()
	reserveRelocations := s.reserveRelocations.Load()
	reserveRelocatedBytes := s.reserveRelocatedBytes.Load()
	live, tomb := s.entries()
	return Stats{
		Gets:             gets,
		Hits:             gets - misses,
		Misses:           misses,
		Puts:             s.puts.Load(),
		Dels:             s.dels.Load(),
		Expirations:      s.expirations.Load(),
		Evictions:        s.evictions.Load(),
		EvictionsLive:    s.evictionsLive.Load(),
		Rejects:          s.rejects.Load(),
		InPlaceUpdates:   s.inPlaceUpdates.Load(),
		SeqlockRetries:   s.seqlockRetries.Load(),
		SeqlockFallbacks: s.seqlockFallbacks.Load(),
		PagesAllocated:   s.pagesAlloc.Load(),
		BytesAllocated:   uint64(s.numPages()) * uint64(s.cfg.PageSize), //nolint:gosec // numPages and PageSize are always non-negative
		BytesUsed:        s.bytesUsed(),
		Entries:          live,
		Tombstones:       tomb,
		CorruptionErrors: s.corrupt.Load(),

		CorruptionBytesDiscarded: s.corruptBytes.Load(),

		Compactions:              s.compactions.Load(),
		CompactionsAborted:       s.compactAborts.Load(),
		CompactionBytesReclaimed: s.compactBytesReclaimed.Load(),
		CompactionDurationMs:     s.compactNanos.Load() / uint64(time.Millisecond),

		ReclaimableBytes:     reclaimable,
		OnlineRelocations:    s.relocations.Load(),
		OnlineBytesRelocated: s.relocatedBytes.Load(),
		OnlinePagesRetired:   s.relocatePagesGone.Load(),
		OnlinePagesRecycled:  s.relocatePagesRecycled.Load(),

		EvictionRelocations:    relocations,
		EvictionBytesRelocated: relocatedBytes,
		ReserveRelocations:     reserveRelocations,
		ReserveBytesRelocated:  reserveRelocatedBytes,
		ReservePagesFreed:      s.reservePagesFreed.Load(),
	}
}

// entries reports the index's occupied and tombstoned slot counts. Both are
// writer-only bookkeeping guarded by s.mu, so this takes the read lock the same
// way bytesUsed does; it is a pair of int reads, not a walk.
//
// The table pointer is swapped atomically on rehash, and a rehash is a writer,
// so holding the read lock is also what makes the Load stable for the duration.
func (s *shard) entries() (live, tomb uint64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	t := s.tab.Load()
	return uint64(t.live), uint64(t.tomb) //nolint:gosec // both are non-negative slot counts
}

func (s *shard) bytesUsed() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var total uint64
	for _, p := range s.pages {
		total += uint64(p.tail() - p.head()) //nolint:gosec // tail >= head always; difference is non-negative
	}
	return total
}

// Close stops the sweeper goroutine and, in mmap mode, syncs and unmaps the
// region. Safe to call once.
func (s *shard) Close() error {
	close(s.stopSweeper)
	s.sweepWG.Wait()
	if s.region != nil {
		// A clean shutdown is a SYNC POINT and must advance the durable bounds, or the
		// reopen loses every write since the last one. setHead/setTail no longer write
		// through to the mapped header (it is a projection now), so the header on disk
		// LAGS the runtime bounds until something projects it. Use the same
		// entries-before-bounds ordering as the other sync points: flush the entries,
		// project each page's current runtime bounds, then flush again so a bound never
		// reaches disk ahead of the bytes it names. The sweeper is stopped and the cache
		// is quiescing, but the lock keeps this correct if a caller races a stray write.
		_ = msync(s.file, s.region)
		s.mu.Lock()
		for _, p := range s.pages {
			p.projectBounds(p.head(), p.tail())
		}
		s.mu.Unlock()
		_ = msync(s.file, s.region) // best-effort final sync of entries + projected bounds
	}
	return munmapAndClose(s.file, s.region)
}

// --- internals ---

// resyncForward scans entries[start:tail] one byte at a time for the first offset c
// whose v5 frame both stays in bounds AND verifies its keyed MAC at (nonce, c) under
// framingKey. Returns (c, true) on the first such offset, or (tail, false) if none
// verifies before tail.
//
// This is the shared engine of the issue-#135 resync used by recovery
// (rebuildIndexFromPages) and eviction (drainPageLocked). Its safety rests entirely
// on decodeEntry verifying a POSITION-BOUND MAC: a frame verifies at c only if it was
// genuinely written at offset c on this page's current life, so the scan can never
// resume on a look-alike frame a client embedded in a value nor on a genuine frame
// copied/replayed to a different position. Byte-by-byte (not by any length field,
// which after a tear is untrustworthy) is what makes it land exactly on the next
// genuine frame boundary — the entry appended after the torn one.
//
// BOUNDED MAC WORK. A candidate offset costs a SipHash over its whole claimed frame
// (key+value) before it can be rejected, and a page's value bytes can encode a large
// in-bounds frame length at MANY offsets, so verifying every candidate would hash
// O(sum of candidate lengths) = O(page^2) bytes — a crafted-then-crashed (or unluckily
// corrupted) page could stall recovery/eviction for a very long time. That is an
// availability bug, so the total MAC input is capped by a per-call BYTE BUDGET of
// 2*len(entries): a header whose length does not fit is skipped for FREE (no hash,
// mirroring decodeEntry's own 32-bit-safe bounds), and only a candidate whose frame
// fits is charged its length and MAC-verified. The budget is a small multiple of the
// page so the common case (a genuine frame a few entries past a single tear) always
// fits, while a page that exceeds it is declared unrecoverable past the tear and
// truncated — the pre-v5 behaviour, and safe. Worst-case hashed input is
// budget + one page = O(PageSize). The MAC safety is unchanged: every candidate the
// budget does allow is still verified at (nonce, c).
func resyncForward(entries []byte, start, tail int, framingKey []byte, nonce uint64) (int, bool) {
	budget := 2 * len(entries)
	hashed := 0
	for c := start; c < tail; c++ {
		// Cheap peek: a candidate whose header does not fit, or whose claimed total runs
		// past tail, cannot be a frame at c — skip it without hashing. These are exactly
		// decodeEntry's bounds guards (including the 32-bit valLen-overflow checks), so a
		// candidate that survives here is one decodeEntry would actually MAC.
		if c+entryHeaderSize > tail {
			continue
		}
		keyLen := int(binary.LittleEndian.Uint16(entries[c : c+2]))
		valLen := int(binary.LittleEndian.Uint32(entries[c+2 : c+6]))
		if valLen < 0 || valLen > tail-c-entryHeaderSize-keyLen {
			continue
		}
		total := entryHeaderSize + keyLen + valLen
		// Charge the frame's length before hashing it. If the budget is spent, give up:
		// the caller truncates the remainder (pre-v5 behaviour) rather than hang.
		hashed += total
		if hashed > budget {
			return tail, false
		}
		if _, _, _, _, err := decodeEntry(entries[c:tail], framingKey, nonce, uint32(c)); err == nil { //nolint:gosec // c < PageSize
			return c, true
		}
	}
	return tail, false
}

// rebuildIndexFromPages rebuilds s.tab from the page bytes: it is what a WARM
// RESTART recovers this node's state from, so what it decides here IS the node's
// committed state (a gracefully restarted replica re-applies nothing below its
// applied index and is never sent an InstallSnapshot). Called only in mmap mode at
// construction time, before any concurrent traffic. No lock needed.
//
// RESOLUTION IS BY WRITE SEQUENCE, NOT BY PAGE ORDER. The walk still visits pages
// 0→N-1 and offsets front→back, but a copy is only allowed to claim a key's index
// slot if its seq is strictly higher than the seq of the copy already there. That
// is the fix for #12A: page order is not write order, because
// firstPageWithRoomLocked scans from index 0 and therefore REVISITS lower pages
// whenever a large entry skips a hole, so an overwrite can legitimately sit BELOW
// the copy it supersedes. Letting "last one the walk reaches" win resolved such a
// key to its OLDER value — a silently lost committed write, made permanent by cold
// compaction physically deleting the newer copy.
//
// EVERY well-framed entry ENTERS THE CONTEST, including ones that will not
// survive it — tombstones, and (on a non-replicated shard) TTL-expired entries.
// Removal happens in a single O(slots) pass at the END. Filtering earlier is
// wrong in both cases, and for the same underlying reason: an entry that is not
// going to survive still has to SUPPRESS every older copy of its key.
//
// For expired entries that is a stale-value bug in its own right. A key written
// with no TTL and later overwritten with an already-expired one must come back
// ABSENT, not reverted: skipping the expired newer copy leaves the older one on
// the page for the walk to index, resurrecting a value that was overwritten.
// (A REPLICATED shard drops nothing by wall clock — see below — so it is
// unaffected either way.)
//
// TOMBSTONES likewise participate as ordinary entries. Neither shortcut works:
//
//   - applying a tombstone ON SIGHT (drop the slot the moment one is decoded)
//     throws away the seq that tombstone carries, so a LOWER-seq copy of the same
//     key encountered later would re-index it — resurrecting exactly what the
//     tombstone was written to prevent;
//   - collecting tombstoned keys into a set and subtracting it at the end is
//     wrong in the other direction: a key may legitimately be deleted and then
//     WRITTEN AGAIN, and that later, higher-seq Put must keep the slot.
//
// Running everything through the same max-seq contest and then removing whichever
// slots a doomed entry actually WON is the only rule that gets every case right.
//
// It also recovers writeSeq as max(seq) over every CRC-valid entry (see the field
// doc): a torn tail fails its CRC and is rejected before it can contribute.
//
// FAILS CLOSED ON A RECOVERY-FLUSH FAILURE (returns an error). When it corrects a
// page's durable bound — rejecting a corrupt head/tail or truncating a torn tail — it
// must make that correction durable through flushRecoveredBoundsLocked, or the next
// open (or a reuse before the first sync point) would trust the stale on-disk bound
// again and reopen the resurrection/truncation window this rebuild just closed. If
// that flush fails, the correction is not durable, so the rebuild returns the error
// and shard construction REFUSES TO OPEN. This is an intentional availability-for-
// integrity trade: a shard that cannot make its own recovery correction durable is
// one whose next restart could silently resurrect stale bytes, and a loud open
// failure (retriable once the disk fault clears) is preferable to serving it.
func (s *shard) rebuildIndexFromPages() error {
	now := s.now()
	var maxSeq uint64
	for pageIdx, p := range s.pages {
		head := p.head()
		tail := p.tail()
		entries := p.entries()
		// Validate BEFORE the head==tail shortcut below: head/tail were seeded at open
		// from the raw durable header, and a corrupt page can carry them bit-rotted (or
		// crash-torn) to the SAME out-of-range value — e.g. both 0xFFFFFFF0. That still
		// satisfies head==tail, so checking bounds only in the else branch let such a
		// page slip through untouched: no corruption counted, no reset, and FreeTail()
		// = len(entries) - tail goes negative, wedging the page's writes behind
		// errPageFull forever. head is `int(uint32(...))`, so it is never negative in
		// practice; the `head < 0` check is kept only as defense in depth.
		if head < 0 || tail < head || tail > len(entries) {
			// Corrupt head/tail (e.g. bit-rot or a torn writeback): trusting these
			// values into entries[cursor:tail] below would either panic (tail beyond
			// the mmap region) or silently skip the whole page (head > tail, so the
			// walk below never runs). Treat it like the torn-entry path: drop the page
			// and count the loss instead of crash-looping or losing data with no signal.
			s.corruptBytes.Add(uint64(len(entries))) //nolint:gosec // a slice length is non-negative
			s.corrupt.Add(1)
			slog.Warn("corrupt page head/tail during recovery; resetting page",
				"component", "cache", "page", pageIdx, "head", head, "tail", tail, "cap", len(entries))
			p.Reset()
			// Persist the correction: setHead/setTail no longer write through to the
			// durable header, so a reset that stayed only in runtime would leave the
			// corrupt bound on disk — and the next open (or a reuse before the first
			// sync point) would trust it again. Project (0,0) and flush this page's
			// header now. The entries it named are being dropped, so nothing durable
			// depends on ordering here. A flush failure fails the open (see the
			// function doc): the correction would otherwise not survive the next crash.
			if ferr := s.flushRecoveredBoundsLocked(pageIdx); ferr != nil {
				return ferr
			}
			continue
		}
		if head == tail {
			continue
		}
		cursor := head
		for cursor < tail {
			key, value, _, meta, err := decodeEntry(entries[cursor:tail], p.framingKey, p.nonce, uint32(cursor)) //nolint:gosec // cursor < PageSize
			if err != nil {
				// A frame that does not verify at this offset (torn write, bit rot, or a
				// look-alike frame planted in some earlier value). RESYNC FORWARD to the
				// next offset whose frame verifies, and drop only the bytes in between.
				//
				// WHY A FORWARD RESYNC IS NOW SAFE (it was not under the v4 CRC — issue
				// #135). The integrity field is a KEYED, POSITION-BOUND MAC (cache/ringbuf.go):
				// SipHash over the page nonce, the candidate OFFSET, and the frame, under a
				// per-file secret. A CRC bounds a RANDOM false match but not a CHOSEN one, so
				// a value — which is attacker-chosen bytes — could carry a complete, correctly
				// CRC'd frame with a chosen sequence, and a v4 forward scan would adopt it:
				// a key nobody wrote, served with a value nobody stored, its sequence set high
				// enough to win the max-seq contest below and lift writeSeq. The MAC removes
				// that: a client cannot compute it (no secret), and even a byte-perfect copy of
				// a GENUINE frame fails to verify unless it sits at the exact offset and page
				// nonce it was written under. So a candidate that verifies here genuinely began
				// at that offset — the scan resumes on the page's own append log, never on a
				// forged or replayed frame. resyncForward starts one byte past the failure so it
				// cannot re-adopt the frame that just failed.
				resumeAt, found := resyncForward(entries, cursor+1, tail, p.framingKey, p.nonce)
				discarded := resumeAt - cursor
				s.corruptBytes.Add(uint64(discarded)) //nolint:gosec // resumeAt >= cursor, both in [head,tail]
				s.corrupt.Add(1)
				if found {
					// Only the damaged bytes [cursor, resumeAt) are dropped; the page keeps
					// framing [head, tail) and the walk resumes at the genuine next frame.
					// The durable bounds are unchanged — the torn bytes stay in the framed
					// range, harmless because the index never points at them and every later
					// walk (drain, cold compaction) resyncs past them the same way; a later
					// compaction is what physically removes them.
					slog.Warn("corrupt entry during recovery; resynchronised to the next valid frame",
						"component", "cache", "page", pageIdx, "offset", cursor, "resume_at", resumeAt, "discarded_bytes", discarded)
					cursor = resumeAt
					continue
				}
				// Nothing verified before tail: this was the last (or only) live frame on
				// the page, or the whole tail is damage. Truncate the page's persisted
				// framing to the validated prefix — the pre-v5 behaviour for the remainder —
				// so the torn region is excluded from every future eviction walk. Reset if
				// nothing valid preceded the corruption.
				slog.Warn("corrupt entry during recovery; truncating page tail",
					"component", "cache", "page", pageIdx, "offset", cursor, "tail", tail, "err", err, "truncate_to", cursor, "discarded_bytes", discarded)
				if cursor == head {
					p.Reset()
				} else {
					p.setTail(cursor)
				}
				// Persist the truncated bound to the durable header (setTail/Reset no
				// longer do — see the head/tail-corruption site above). The retained
				// prefix [head, cursor) is already durable (its bytes decoded from disk),
				// so reducing the durable tail to cursor names only on-disk bytes and,
				// crucially, drops the torn region from the durable framing so a reuse
				// before the first sync point cannot re-expose it. A flush failure fails
				// the open (see the function doc).
				if ferr := s.flushRecoveredBoundsLocked(pageIdx); ferr != nil {
					return ferr
				}
				break
			}
			// EXACT, and NOT HEAP-REACHABLE: rebuildIndexFromPages runs in mmap mode
			// only (newShard calls it on the persisted branch), so nothing that
			// changes how HEAP pages are framed reaches this walk. It steps over bytes
			// a previous process encoded, so the cursor advances by the encoder's
			// length. Should the append path ever reserve more than it encodes on a
			// PERSISTED page, this walk becomes the site that has to advance by that
			// reservation instead — but that is a change to mmap framing, not to slack
			// on heap pages, and it would be found here rather than assumed.
			esize := entrySpanExact(len(key), len(value))
			seq := metaSeq(meta)
			if seq <= s.flushedThroughSeq {
				// Wiped by a prior Cache.Flush(): this entry is below the durable flush
				// floor, so it is neither indexed NOR allowed to contribute to the
				// recovered max(seq) — letting it lift writeSeq back into the flushed
				// range would re-open the very hazard the writeSeq-restore in newShard
				// closes. Skip it entirely.
				cursor += esize
				continue
			}
			if seq > maxSeq {
				maxSeq = seq
			}
			h := hashKey(key)
			t := s.tab.Load()
			if s.indexedSeqAtLeast(t, h, seq) {
				// An already-indexed copy of this key was written at the same or a
				// later sequence. Page order said otherwise; the sequence is the
				// authority.
				cursor += esize
				continue
			}
			t.upsert(h, makeSlabRef(
				uint16(pageIdx), //nolint:gosec // pageIdx < MaxPagesPerShard
				p.gen,
				uint32(cursor), //nolint:gosec // cursor < PageSize
			))
			s.rehashIfOverThresholdLocked(t)
			cursor += esize
		}
	}
	s.stripDeadSlots(now)
	// Never regress: remapPagesFile rebuilds a second time over the compacted file,
	// and compaction preserves every survivor's meta verbatim, so the max is
	// preserved — but taking the max defensively costs nothing and makes the
	// sequence monotonic across any future caller.
	if maxSeq > s.writeSeq {
		s.writeSeq = maxSeq
	}
	return nil
}

// indexedSeqAtLeast reports whether the index already resolves hash h to a
// physical copy whose write sequence is >= seq. Recovery-time helper for
// rebuildIndexFromPages; reads the meta word of the currently-indexed copy
// straight off its page.
//
// An unreadable current ref answers false, so the candidate wins: the slot points
// at something we cannot judge, and a decodable entry is strictly better than an
// undecodable one.
func (s *shard) indexedSeqAtLeast(t *indexTable, h, seq uint64) bool {
	_, cur, ok := t.findSlot(h)
	if !ok {
		return false
	}
	meta, mok := s.pages[cur.pageIdx()].MetaAt(cur.offset())
	if !mok {
		return false
	}
	return metaSeq(meta) >= seq
}

// stripDeadSlots is the FINAL pass of the warm-restart rebuild: it removes every
// index slot whose winning physical copy should not be reachable — a delete
// record, or (non-replicated only) a TTL-expired entry. Running it after the whole
// max-seq contest, rather than filtering entries as they are decoded, is what lets
// a doomed entry still SUPPRESS older copies of its key, and what makes
// delete-then-rewrite work (the rewrite wins the contest on sequence, so its slot
// is never visited here). See rebuildIndexFromPages.
//
// The expiry half is deliberately NOT applied on a replicated shard: dropping by
// wall clock would remove different keys on replicas that restart at different
// instants, diverging the committed key set (#4 Phase B / B3a). There the absolute
// expiry stays intact on the page, logically-expired entries are filtered on read,
// and physical reclamation belongs to the logical-clock sweeper.
//
// O(index slots), construction-time only, no lock.
func (s *shard) stripDeadSlots(now uint64) {
	dropExpired := !s.cfg.Replicated
	t := s.tab.Load()
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		ref := slabRef(t.refs[i].Load())
		p := s.pages[ref.pageIdx()]
		// Both branches below remove a slot that IS the winning copy of its key (the
		// max-seq contest already ran), so each notifies the derived-index hook. In
		// practice the hook is always nil here: this runs inside newShard, before the
		// cache exists to install one. It fires anyway so the rule "every live-slot
		// removal notifies" holds without a caller-ordering caveat.
		if meta, ok := p.MetaAt(ref.offset()); ok && metaIsTombstone(meta) {
			t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index
			s.fireOnRemoveAt(p, ref.offset())
			continue
		}
		if dropExpired {
			if k, _, exp, err := p.Read(ref.offset()); err == nil && isExpired(exp, now) {
				t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index
				s.fireOnRemove(k)
			}
		}
	}
}

// findOrMakePageLocked returns the index of a page with at least `need` bytes
// of contiguous tail space. In heap mode, lazily allocates a new page if
// needed. In mmap mode, all pages exist from the start; falls through to
// AtCapPolicy when none have room.
//
// Must be called with s.mu held for writing.
func (s *shard) findOrMakePageLocked(need int) (int, error) {
	// Reject entries that can never fit a single (empty) page BEFORE any
	// eviction. A fresh page's usable capacity is PageSize - pageHdrSize in
	// mmap mode (heap pages have no header). maxValueLen (4 GiB-1) exceeds the
	// max PageSize (1 GiB), so an oversized value is reachable. Without this
	// up-front guard, the mmap path falls through to evictUntilFitsLocked,
	// which can never satisfy the request and drains the entire shard's live
	// data before failing. Applies to BOTH heap and mmap modes.
	if need > s.maxEntryBytes() {
		return 0, errors.New("cache: entry larger than PageSize")
	}
	// Fast path: the current open page still has contiguous tail room. A page whose
	// durable reuse barrier failed (page.reuseBarrierFailed) is excluded even here — it
	// shows full FreeTail but its cleared header is not on disk, so it must not be
	// written into; fall through to firstPageWithRoomLocked, which retry-clears or skips
	// it and leaves s.writeIdx pointing at a page that IS safe to reuse.
	if s.writeIdx < len(s.pages) && !s.pages[s.writeIdx].reuseBarrierFailed && s.pages[s.writeIdx].FreeTail() >= need {
		return s.writeIdx, nil
	}
	// Any other already-allocated page with tail room? writeIdx only tracks the
	// most recently used page, so this reaches pages it skips: in mmap mode all
	// MaxPagesPerShard pages exist from construction (writeIdx starts at the last
	// one), and in heap mode preallocated pages (InitialPagesPerShard > 1) may be
	// empty. Reusing them before growing/evicting/rejecting reclaims the full
	// provisioned capacity — previously only the writeIdx page was checked, so an
	// mmap shard under PolicyRejectWrites returned ErrFull after filling a single
	// page (~1/MaxPagesPerShard of its memory).
	if idx := s.firstPageWithRoomLocked(need); idx >= 0 {
		s.writeIdx = idx
		return idx, nil
	}
	if !s.isMmap && len(s.pages) < s.cfg.MaxPagesPerShard() {
		// Heap mode: lazily allocate another page.
		idx := s.allocHeapPageLocked()
		if s.pages[idx].FreeTail() < need {
			return 0, errors.New("cache: entry larger than PageSize")
		}
		s.writeIdx = idx
		return idx, nil
	}
	// Either at cap (heap mode), or mmap mode (always at cap). Apply policy.
	switch s.cfg.AtCapPolicy {
	case PolicyRingbufEvict:
		idx, err := s.evictUntilFitsLocked(need)
		if err != nil {
			return 0, err
		}
		s.writeIdx = idx
		return idx, nil
	case PolicyRejectWrites:
		s.rejects.Add(1)
		return 0, ErrFull
	default:
		return 0, errors.New("cache: unknown AtCapPolicy")
	}
}

// maxEntryBytes returns the largest entry (header + key + value) that can fit
// in a single empty page. Mmap-backed pages reserve pageHdrSize bytes at the
// front for the persisted head/tail; heap pages use the whole slab.
func (s *shard) maxEntryBytes() int {
	if s.isMmap {
		return s.cfg.PageSize - pageHdrSize
	}
	return s.cfg.PageSize
}

// freshHeapPageLocked allocates a heap page carrying the generation and, when
// the shard reads through the seqlock, the version counters its readers will
// validate against. EVERY heap page reaches its shard through here — the initial
// allocation and both retirement paths — so a page cannot be published to a
// reader missing the state that protocol needs. Must hold mu for writing (or run
// during construction, before the shard is shared).
func (s *shard) freshHeapPageLocked() *page {
	p := newHeapPage(s.cfg.PageSize)
	p.gen = s.nextGen()
	if s.seqlockReads() {
		p.enableVersions()
	}
	return p
}

// allocHeapPageLocked appends a fresh heap-backed page and returns its index.
// Must be called with s.mu held for writing (or from the constructor before
// the shard is shared).
func (s *shard) allocHeapPageLocked() int {
	p := s.freshHeapPageLocked()
	idx := len(s.pages)
	s.pages = append(s.pages, p)
	s.pageSlots[idx].Store(p) // publish for the lock-free read path
	regionNotePage(s, p)      // compiled out unless the measurement build tag is set
	s.pagesAlloc.Add(1)
	return idx
}

// firstPageWithRoomLocked returns the index of the first page with at least
// `need` bytes of contiguous tail room, or -1 if none. Must be called with s.mu
// held FOR WRITING (both callers, findOrMakePageLocked and evictUntilFitsLocked,
// do): it may retry-clear a poisoned page's durable reset (see
// clearReuseBarrierIfFlushableLocked), which mutates page state. Shared so the
// all-pages free-space scan lives in one place.
func (s *shard) firstPageWithRoomLocked(need int) int {
	for i := range s.pages {
		// Skip pages retired by online relocating compaction: their stale-framed bytes
		// are immutable and stranded (see page.retired). `retired` is always false on
		// heap / single-node / ringbuf shards, so this is a no-op there.
		if s.pages[i].retired {
			continue
		}
		if s.pages[i].FreeTail() < need {
			continue
		}
		// A page whose durable reuse barrier failed (page.reuseBarrierFailed) is empty in
		// runtime — so it passes the FreeTail check above — but its cleared header is not
		// on disk, so it must not be reused until the flush succeeds. Retry the flush
		// here, at the one spot that would hand the page out: on success the poison is
		// cleared and the page is used; on failure it is skipped and stays out of the
		// writable set. Only fires when a poisoned page is an actual candidate (a
		// failure-event state), never in the steady state.
		if s.pages[i].reuseBarrierFailed && !s.clearReuseBarrierIfFlushableLocked(i) {
			continue
		}
		return i
	}
	return -1
}

// clearReuseBarrierIfFlushableLocked retries the durable header flush for a page whose
// reuse barrier previously failed (page.reuseBarrierFailed). It retries the WHOLE
// barrier, not just the msync: the poison can have been set by a nonce-rotation
// failure, in which case zeroDurableBoundsForReuseLocked returned BEFORE projecting,
// so the mapped header still carries the extent's old nonce and old (larger) bounds.
// Re-msyncing that unchanged header would flush a stale reset and clearing the poison
// would then hand the write path an extent whose reset was never established — the
// exact resurrection/forge window this barrier closes. Retrying the full barrier
// (rotate nonce, project (0,0), msync) is idempotent and establishes the reset in
// both the nonce-failure and msync-failure cases. On success it clears the poison and
// returns true (the extent may be reused); on failure it leaves the poison set and
// returns false (kept out of the writable set until the fault clears). Must hold s.mu.
func (s *shard) clearReuseBarrierIfFlushableLocked(idx int) bool {
	if err := s.zeroDurableBoundsForReuseLocked(idx); err != nil {
		return false
	}
	s.pages[idx].reuseBarrierFailed = false
	return true
}

// evictUntilFitsLocked frees space until at least one page has `need` bytes
// of tail room. Removes evicted entries from the index.
// Must be called with s.mu held for writing.
//
// Tail room only reappears when a page is fully emptied: EvictFront advances the
// head (freeing FRONT space), but FreeTail is measured from the tail, which only
// shrinks back to zero on Reset (when the page empties). So once a victim is
// chosen, it is drained to completion in one inner loop — re-scanning every page
// after each single eviction (the old behavior) was O(entries × pages) wasted
// work, since no page gains tail room until its victim empties.
//
// Victims are chosen in rotation (nextVictim) rather than always taking the
// lowest-index non-empty page, so eviction cycles through every page in FIFO
// order instead of pinning to page 0 and evicting the newest writes.
func (s *shard) evictUntilFitsLocked(need int) (int, error) {
	for {
		// Check if any page already has room; return it as the new open page.
		if idx := s.firstPageWithRoomLocked(need); idx >= 0 {
			return idx, nil
		}
		// Pick the next non-empty page in rotation order and drain it fully; only
		// emptying it (Reset) restores tail room.
		victim := s.nextNonEmptyPageLocked(-1)
		if victim < 0 {
			return 0, ErrCannotEvict
		}
		// Advance the rotation cursor past this victim so the next eviction moves
		// on to the following page rather than re-selecting a just-refilled one.
		s.nextVictim = (victim + 1) % len(s.pages)
		if err := s.evictVictimLocked(victim, need); err != nil {
			return 0, err
		}
	}
}

// nextNonEmptyPageLocked returns the first non-empty page at or after the
// rotation cursor (nextVictim), wrapping, or -1 if every page is empty — the
// order eviction picks victims in. `skip` is one page index to pass over, or -1
// for none: relocating eviction uses it to keep the page it has just freshened
// out of the search. Must hold mu for writing.
//
// ITS TWO CALLERS MUST STAY ALIGNED, and the alignment is load-bearing rather than
// incidental. evictUntilFitsLocked calls it with skip = -1 to CHOOSE a victim and
// then advances nextVictim past that victim; relocating eviction calls it moments
// later with skip = the page just freed, to choose the page to EVACUATE. Both scan
// from the same cursor in the same order, and the freed page sorts LAST from that
// cursor (the scan starts just past it), so the page relocation evacuates is exactly
// the page the next eviction will select.
//
// That equality is the whole point of the pass. Break it — by advancing nextVictim
// somewhere other than immediately before evictVictimLocked, by giving one caller a
// different scan order, or by dropping the skip — and relocation copies records out
// of a page that is NOT about to be drained: pure write amplification, and the
// crash-consistency argument in cache/relocate_evict.go loses its premise along with
// it, because that argument rests on every moved record being one the next drain was
// going to delete anyway.
func (s *shard) nextNonEmptyPageLocked(skip int) int {
	n := len(s.pages)
	for off := 0; off < n; off++ {
		i := (s.nextVictim + off) % n
		if i == skip {
			continue
		}
		if !s.pages[i].Empty() {
			return i
		}
	}
	return -1
}

// evictVictimLocked frees page `victim` in full. In heap mode it retires the
// page — swapping in a fresh frozen object so lock-free readers stay safe (see
// retirePageLocked). In mmap mode it drains the fixed persisted region in place
// under the write lock (see drainPageLocked), because mmap page objects wrap the
// file and cannot be swapped for a fresh allocation. Must hold mu for writing.
//
// `need` is the byte requirement of the write that triggered this eviction. Both
// modes reserve it out of the freed page before relocating eviction may spend any
// of what is left (cache/relocate_evict.go).
func (s *shard) evictVictimLocked(victim, need int) error {
	if !s.isMmap {
		s.retirePageLocked(victim)
		// The freed page is the only page with room at this point in an eviction, so
		// this is where relocation gets to run at all. It evacuates the page the
		// rotation cursor will drain NEXT — not this one, whose live records the
		// retire walk above has already tombstoned, and which for reader-safety
		// reasons could never have rescued its own (see cache/relocate_evict.go).
		if s.cfg.RelocatingEviction {
			s.noteEvictRelocation(s.relocateIntoFreedPageLocked(victim, need))
		}
		return nil
	}
	if err := s.drainPageLocked(victim); err != nil {
		return err
	}
	// Same placement, same reason, on the in-place drain: the page the walk above
	// just emptied is the only one with room, and the records moved into it come off
	// the page the NEXT eviction drains — so they stop being index-current before
	// that drain reaches them. Readers are excluded here rather than raced
	// (needsReadLockForGet), and the crash-consistency argument for appending
	// without a sync is in cache/relocate_evict.go.
	if s.cfg.RelocatingEviction {
		s.noteEvictRelocation(s.relocateIntoFreedPageLocked(victim, need))
	}
	return nil
}

// retirePageLocked (heap ringbuf) frees page idx by dropping every index slot
// that still points into it and then REPLACING the page object with a fresh,
// empty one carrying a new generation. The retired object is never mutated again
// (frozen): a lock-free reader that already loaded its pointer reads immutable
// bytes; a reader that loads the fresh object sees gen(page) != gen(ref) and
// misses without reading the page a writer may be actively appending to. This is
// the invariant the naive in-place-Reset v2 violated. Must hold mu for writing.
func (s *shard) retirePageLocked(idx int) {
	old := s.pages[idx]
	// Walk the page's framed entries WITHOUT mutating it (no EvictFront/Reset),
	// dropping each index slot that still points at this physical copy. Mirrors
	// the cur == ref guard in drainPageLocked: a slot repointed by a newer Put
	// (cur != ref) belongs to live data elsewhere and must not be tombstoned.
	entries := old.entries()
	tail := old.tail()
	// Wall clock is correct here even under a replicated apply: these are
	// observability counters, not stored state. The tombstone decision below is
	// still cur == ref alone, so what the cache RETAINS stays deterministic
	// across replicas; only the counter may differ by a sweep's worth.
	now := s.now()
	for cursor := old.head(); cursor < tail; {
		key, value, expiryMs, err := decodeEntryFast(entries[cursor:tail])
		if err != nil {
			// Heap pages are written without a CRC and cannot be corrupted by
			// anything external, so this is unreachable in practice; stop the walk
			// defensively — the fresh page replaces the whole slab regardless.
			break
		}
		ref := makeSlabRef(uint16(idx), old.gen, uint32(cursor)) //nolint:gosec // idx bounded by MaxPagesPerShard (≤65535); cursor < PageSize ≤ MaxInt32
		h := hashKey(key)
		t := s.tab.Load()
		if slot, cur, ok := t.findSlot(h); ok && cur == ref {
			t.tombstone(slot)
			// Count it as a capacity loss only if it was BOTH index-current and
			// still live. cur == ref alone would also catch an entry that had
			// already expired but not yet been swept, which is TTL turnover
			// wearing a capacity costume - and precisely the case a
			// correctly-sized cache is full of.
			//
			// That turnover is counted as an EXPIRATION instead, which is where it
			// belongs and where a reader looking for it will go: Stats.Expirations is
			// what EvictionsLive is meant to be read against ("EvictionsLive > 0 with
			// Expirations low means the budget, not the TTL, is deciding how long
			// entries survive"), and that comparison is only true if the entries a
			// retire reaps ahead of the sweeper land in it. Whichever pass gets to an
			// expired entry first, the shard reports the same thing. It is additive:
			// the entry is still one of Evictions, which counts every framed entry a
			// retire displaces regardless of why.
			if isExpired(expiryMs, now) {
				s.expirations.Add(1)
			} else {
				s.evictionsLive.Add(1)
			}
			// INSIDE the cur == ref guard, never outside it. This walk visits every
			// framed entry on the page, DEAD DUPLICATES INCLUDED — copies a later Put
			// superseded, whose index slot points at a live copy on another page.
			// Notifying for one of those would drop a live key's postings.
			s.fireOnRemove(key)
		}
		s.evictions.Add(1)
		// OCCUPANCY. A HEAP page walk (this is the heap retire path — it ends in
		// freshHeapPageLocked), so the cursor has to step by the room each entry
		// takes up, not by the bytes its encoder wrote. Advance by anything smaller
		// and the walk lands mid-entry and mis-frames the rest of the page.
		cursor += entrySpan(len(key), len(value), true)
	}
	fresh := s.freshHeapPageLocked()
	s.pages[idx] = fresh
	s.pageSlots[idx].Store(fresh) // publish so readers resolve the new generation
	regionNotePage(s, fresh)      // compiled out unless the measurement build tag is set
}

// drainPageLocked evicts every live entry from page victim, dropping each from
// the index, until the page is empty. If the page's persisted head/tail are out of
// range, or EvictFront reports torn framing — an entry header whose lengths run past
// the page's live band — the page cannot be walked. The warm-restart walk would have
// rejected either, so these are bytes that changed under a running shard. The event is
// counted and logged, every index slot still addressing the page is dropped, and the
// page is Reset so the shard regains write availability instead of failing every future
// space-needing Put (see discardCorruptPageLocked).
// Must be called with s.mu held for writing.
//
// POISONED-DRAIN DEGRADATION (fail-closed, not data loss). The victim's entries are
// evicted from the index BEFORE the reuse barrier runs at the end, so if that barrier
// fails the eviction has already happened — the page is empty — and the extent is
// poisoned (reuseBarrierFailed) to keep it out of the writable set. Under a PERSISTENT
// barrier fault each failed Put therefore still spends one victim page's eviction
// before failing closed with ErrReuseBarrier. That is intended degradation (writes
// fail rather than reuse a non-durable extent), not lost committed data: eviction only
// drops cache entries the ringbuf policy was already free to discard.
//
// RESETTING WITHOUT SKIPPING LOSES NOTHING A DRAIN WOULD HAVE KEPT. This function
// empties the page whether or not it meets a tear, and relocating eviction takes its
// rescues off a page one eviction BEFORE the drain that empties it
// (cache/relocate_evict.go), so every record still on the page at this point was
// going out regardless. Resynchronising past the bad entry would change only which
// dead bytes get stepped over, and it would have to trust raw bytes to do it (see
// rebuildIndexFromPages for why that trust is not available). What the tear DOES
// cost is the walk, and the walk is what drops each record's index slot — so the
// slots are dropped from the index side instead.
//
// THE SLOTS ARE NOT OPTIONAL. An in-place Reset keeps the page object and its
// generation, so a slot left addressing the page still passes the read path's
// generation gate. Until the region is rewritten it serves the record this drain
// evicted; once the page refills, the bytes at its offset belong to a different
// record — which Iterate reports under that record's key a second time (it does not
// check a key against its slot), and which the slot, never matched by any later walk,
// keeps counted in Entries for the life of the process.
//
// This is the MMAP-ONLY eviction path (see evictVictimLocked): it mutates the
// page's fixed persisted region in place, which is safe because mmap ringbuf
// reads hold the read lock. Heap ringbuf never reaches here — it retires pages
// by frozen replacement (retirePageLocked) so its reads can stay lock-free.
func (s *shard) drainPageLocked(victim int) error {
	// See retirePageLocked: wall clock is fine for a counter, the retained set
	// stays decided by cur == ref alone.
	now := s.now()
	for !s.pages[victim].Empty() {
		// BOUNDS FIRST. head and tail are the runtime bounds, seeded at open from the
		// raw durable header (attachMmapRegion) and not re-checked since: recovery's
		// bounds check ran once at open, and nothing has validated them again. Every
		// read below trusts them — the expiry probe slices entries up to tail,
		// EvictFront slices from head — so a corrupted pair must be caught here, before
		// either runs. p.head() carries a value widened from a uint32 at open, so on
		// 32-bit a large seeded value is already negative, which the first test catches.
		if p := s.pages[victim]; p.head() < 0 || p.tail() < p.head() || p.tail() > len(p.entries()) {
			// Those bounds were the only record of how much the page held, so nothing
			// derived from them is a loss figure — tail-head is negative for head > tail.
			// The page's capacity is reported instead, as recovery does for a page whose
			// bounds it rejects.
			slog.Warn("corrupt page head/tail during eviction; resetting page",
				"component", "cache", "page", victim, "head", p.head(), "tail", p.tail(), "cap", len(p.entries()))
			// discardCorruptPageLocked resets the page and re-runs the durable
			// reuse barrier; if that barrier fails, the page is NOT safely reset and
			// the error propagates so this eviction (and the write that triggered it)
			// fails closed rather than treating a non-durable extent as available.
			return s.discardCorruptPageLocked(victim, len(p.entries()), now)
		}
		p := s.pages[victim]
		head, tail := p.head(), p.tail()
		off := uint32(head) //nolint:gosec // 0 <= head <= len(entries) is established above
		ent := p.entries()
		// VERIFY THE HEAD FRAME'S MAC before trusting its framing (this is the eviction
		// verify site, issue #135). A CRC-era drain framed the entry from raw bytes and
		// reset the WHOLE page on any framing error, losing every durable record after
		// the damage. With a keyed, position-bound MAC a damaged frame can be told from
		// a genuine one, so a tear now costs only the damaged entry: resync forward to
		// the next frame that verifies and keep evicting.
		evictedKey, evictedVal, headExpiryMs, _, derr := decodeEntry(ent[head:tail], p.framingKey, p.nonce, off)
		if derr != nil {
			// Damaged head entry. Find where the next genuine frame begins and drop only
			// the bytes in between; the frames after it are still evicted normally.
			resumeAt, _ := resyncForward(ent, head+1, tail, p.framingKey, p.nonce)
			discarded := resumeAt - head
			s.corruptBytes.Add(uint64(discarded)) //nolint:gosec // resumeAt >= head, both in [head,tail]
			s.corrupt.Add(1)
			slog.Warn("corrupt entry during eviction; resynchronised to the next valid frame",
				"component", "cache", "page", victim, "offset", off, "resume_at", resumeAt, "discarded_bytes", discarded, "err", derr)
			// Drop EVERY index slot addressing any offset in the whole skipped gap
			// [head, resumeAt), not just the head. When several consecutive frames fail,
			// resyncForward jumps over all of them, so a slot pointing at any offset inside
			// the gap would be left dangling — behind the advanced head, its bytes free to
			// be overwritten on reuse — and a later Get would resolve it through
			// decodeEntryFast (no MAC) into whatever now lies there. The frames did not
			// verify, so their keys cannot be read: the slots are found by physical ref
			// (O(index), acceptable for a corruption event) and no onRemove fires, since a
			// key read from damaged bytes would misnotify a derived index.
			s.dropDamagedSlotsLocked(victim, p.gen, off, uint32(resumeAt)) //nolint:gosec // head <= resumeAt <= tail <= PageSize
			if resumeAt >= tail {
				// Nothing valid remains; the page is now empty. Fall out of the loop so
				// the reuse-ordering flush below runs, exactly as a clean drain-to-empty.
				p.setHead(0)
				p.setTail(0)
			} else {
				p.setHead(resumeAt)
			}
			continue
		}
		// Advance head past the verified entry. entrySpanExact frames it from the lengths
		// the MAC just vouched for, so this matches what EvictFront would have computed.
		newHead := head + entrySpanExact(len(evictedKey), len(evictedVal))
		// Only drop the index slot if it still points at THIS physical copy.
		// A later Put may have overwritten the key with a newer entry in a
		// different page (leaving these bytes as a dead duplicate), or an
		// unrelated key may collide on this hash. Deleting unconditionally
		// would silently evict live data. Mirror the cur == ref guard in getH.
		ref := makeSlabRef(uint16(victim), p.gen, off) //nolint:gosec // victim bounded by MaxPagesPerShard (≤65535)
		h := hashKey(evictedKey)
		t := s.tab.Load()
		if slot, cur, ok := t.findSlot(h); ok && cur == ref {
			t.tombstone(slot)
			if !isExpired(headExpiryMs, now) {
				s.evictionsLive.Add(1)
			}
			// INSIDE the cur == ref guard, exactly as in retirePageLocked: this drain
			// walks every framed entry including dead duplicates whose slot points at a
			// newer live copy elsewhere. evictedKey still aliases the page (head only
			// moved past it; nothing has been written over it inside this loop), and the
			// hook contract requires the callback to copy before retaining.
			s.fireOnRemove(evictedKey)
		}
		s.evictions.Add(1)
		if newHead >= tail {
			p.setHead(0)
			p.setTail(0)
		} else {
			p.setHead(newHead)
		}
	}
	// REUSE ORDERING. The drain emptied this mmap extent (EvictFront's last step reset
	// the runtime head/tail to 0); the write path may now refill it from offset 0.
	// Zero the DURABLE header for the extent and flush it BEFORE that reuse, or the OS
	// could write back the new entry bytes while the header still held the old (larger)
	// bound — recovery would then frame the extent's former contents. The corrupt-page
	// exits above route through discardCorruptPageLocked, which zeroes the durable
	// header the same way after its Reset.
	//
	// FAIL CLOSED. If the barrier msync (or nonce rotation) fails, the extent's cleared
	// header is not on disk, so returning it to the write path would risk a crash
	// recovering its stale bytes. POISON the now-empty extent (reuseBarrierFailed) so the
	// write selectors keep it out of the writable set — the drain already reset its
	// runtime bounds, so without this it would show full FreeTail and a later Put would
	// reuse it before its header reset is durable — and propagate the error so the
	// triggering op fails closed. The poison self-heals via a later retry-flush.
	if err := s.zeroDurableBoundsForReuseLocked(victim); err != nil {
		s.pages[victim].reuseBarrierFailed = true
		return err
	}
	return nil
}

// dropDamagedSlotsLocked tombstones every index slot addressing an offset in the
// half-open range [lo, hi) of page victim's current generation — the gap the eviction
// drain skipped when it resynced past one or more frames it could not verify. It is
// the "drop only the damaged region" counterpart to discardCorruptPageLocked's
// whole-page sweep: a resynced tear costs only the skipped slots, not the page.
//
// THE WHOLE GAP, not just its first offset: resyncForward may jump over several
// consecutive failed frames, and a slot left pointing anywhere inside the gap would be
// stranded below the advanced head, its bytes free to be overwritten on reuse, and a
// later Get would resolve it through the MAC-less decodeEntryFast into whatever now
// lies there. It is O(index slots) but runs only on a corruption event, never in the
// steady state. No onRemove fires and no key is read: the frames did not verify, so
// their bytes cannot be trusted to name a key, and notifying a derived index under a
// key read from damage would drop some unrelated live key's postings (exactly the
// reasoning in discardCorruptPageLocked). Each dropped slot is counted as an eviction
// so its disappearance is observable. Must hold mu.
func (s *shard) dropDamagedSlotsLocked(victim int, gen uint16, lo, hi uint32) {
	t := s.tab.Load()
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		ref := slabRef(t.refs[i].Load())
		if int(ref.pageIdx()) != victim || ref.gen() != gen {
			continue
		}
		if o := ref.offset(); o < lo || o >= hi {
			continue
		}
		t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index
		s.evictions.Add(1)
	}
}

// discardCorruptPageLocked is drainPageLocked's response to a page it cannot walk: it
// records the loss, drops every index slot still addressing the page, and resets it.
// discarded must be non-negative — the caller has already decided what an honest
// figure is for its case. Must hold mu for writing.
//
// Returns the reuse-barrier error: the Reset clears only the RUNTIME bounds, and the
// durable header is not safely cleared until zeroDurableBoundsForReuseLocked flushes
// it. If that flush fails the page is NOT a safely-reset extent, so the caller must
// not treat it as available — the error propagates to fail the triggering op closed.
func (s *shard) discardCorruptPageLocked(victim, discarded int, now uint64) error {
	p := s.pages[victim]
	// Bytes before the incident: snapshot() loads the incident count first, so a
	// concurrent snapshot can see the bytes of an incident it does not yet count, but
	// never an incident without its bytes.
	s.corruptBytes.Add(uint64(discarded)) //nolint:gosec // callers pass a non-negative figure
	s.corrupt.Add(1)
	// O(index slots), under the write lock — acceptable only because this is a
	// corruption response, never the steady state. Every slot addressing this page
	// object is dropped, not just those at or past the tear: the page is about to be
	// empty, and an empty page holds no record.
	t := s.tab.Load()
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		ref := slabRef(t.refs[i].Load())
		if int(ref.pageIdx()) != victim || ref.gen() != p.gen {
			continue
		}
		t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index
		s.evictions.Add(1)
		// The slot says where its record starts, but the bytes there are the ones that
		// just proved untrustworthy on this page. A key is reported only when it decodes
		// AND hashes to the slot's own hash; anything else — the torn record itself,
		// most likely — notifies nothing, exactly as sweepIndex treats an unreadable
		// slot. Notifying under a key read from garbage would drop some unrelated live
		// key's postings from a derived index. Read bounds itself by the page, not by
		// the corrupt tail, so it is safe here whatever the header says.
		key, _, exp, rerr := p.Read(ref.offset())
		if rerr != nil || hashKey(key) != t.hashes[i] {
			continue
		}
		if !isExpired(exp, now) {
			s.evictionsLive.Add(1)
		}
		s.fireOnRemove(key)
	}
	p.Reset()
	// REUSE ORDERING (see zeroDurableBoundsForReuseLocked). Reset cleared only the
	// runtime bounds; the durable header still names the discarded extent's bytes.
	// Zero and flush it before the write path can refill this now-empty page; on a
	// barrier failure POISON the extent (reuseBarrierFailed) so the write selectors keep
	// it out of the writable set until a later retry-flush succeeds, and return the error
	// so the caller fails closed rather than reusing a non-durable extent.
	if err := s.zeroDurableBoundsForReuseLocked(victim); err != nil {
		s.pages[victim].reuseBarrierFailed = true
		return err
	}
	return nil
}

// runSweeper expires entries past their TTL on a fixed cadence.
func (s *shard) runSweeper() {
	defer s.sweepWG.Done()
	ticker := time.NewTicker(time.Duration(s.cfg.TTLSweepIntervalMs) * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-s.stopSweeper:
			return
		case <-ticker.C:
			s.sweepOnce()
		}
	}
}

// sweepBatchSize bounds how many index slots a single sweep lock acquisition
// scans, so a large shard's sweep can't pin the lock for an unbounded pass.
const sweepBatchSize = 4096

// sweepOnce runs one TTL reclamation pass. The CLOCK it judges expiry against is
// the ONLY thing that differs by mode:
//
//   - Non-replicated: the wall clock (s.now()). Unchanged single-node behavior —
//     index-slot tombstoning only (page bytes are reclaimed by ringbuf eviction, or
//     not at all under reject-writes; that is the pre-B3b status quo and Direct /
//     single-node output stays byte-identical).
//
//   - Replicated (#4 Phase B / B3b): the shard's LOGICAL clock,
//     lastAppliedStampMs — the running max of leader stamps, IDENTICAL on every
//     replica (see the field doc). It NEVER consults the wall clock, so no
//     wall-clock divergence can leak into the committed key set. If the logical
//     clock is still 0 (no stamped apply yet — e.g. apply-stamping disabled during
//     rollout phase 1, or a freshly restored snapshot) the pass is a NO-OP: nothing
//     is reclaimed, which is safe (the ghost-page cliff simply persists until
//     stamping is enabled). When it is non-zero the pass both tombstones expired
//     index slots AND retires whole expired heap pages, physically freeing capacity
//     so a committed write on a near-full replicated shard no longer hits ErrFull.
func (s *shard) sweepOnce() {
	if s.cfg.Replicated {
		stamp := s.lastAppliedStampMs.Load()
		if stamp == 0 {
			// No stamped apply has advanced the logical clock yet, so LOGICAL-clock TTL
			// reclamation (sweepIndex / reclaimExpiredHeapPages) is pointless — isExpired
			// (exp,0) is always false — and, if we ever changed that, non-deterministic. Do
			// none of it. But online relocation/recycle of SUPERSEDED (index-dead) versions
			// needs NO clock: it reclaims dead duplicates and recycles retired extents
			// (compactDropClock()==0 ⇒ zero TTL drops, full superseded reclaim), so an
			// opted-in shard must still run it here instead of being inert until stamping.
			s.maybeRelocateCompact()
			return
		}
		// Index-slot reclamation against the logical clock (removes expired keys
		// from the committed set identically on every replica) ...
		s.sweepIndex(stamp)
		// ... then physically reclaim whole expired heap pages so the shard regains
		// write capacity. reclaimExpiredHeapPages takes the write lock PER PAGE and
		// releases it between pages, so it never pins the apply path for an
		// O(all-pages) scan (see its doc).
		s.reclaimExpiredHeapPages(stamp)
		// mmap page-byte occupancy alert (#4 Option 3): the reclaim above is a
		// NO-OP for mmap (page bytes can't be freed while readers are running), so
		// ghost bytes accumulate until the next open compacts the file. Surface it
		// before ErrFull so the restart happens on purpose, not on the cliff.
		s.checkMmapOccupancy()
		// Online relocating compaction (cache/compact_online.go): Stage 0 records the
		// reclaimable-byte figure and evaluates the trigger; Stage 1 (when enabled)
		// relocates live entries out of fragmented pages WITHOUT disturbing the lock-free
		// read path. Gated to mmap replicated reject-writes shards, so it is a no-op for
		// heap / single-node / ringbuf shards. Runs AFTER sweepIndex so logically-expired
		// slots are already tombstoned and never relocated.
		s.maybeRelocateCompact()
		return
	}
	s.sweepIndex(s.now())
	// The free-page reserve is NOT run from here. It used to be, and the two jobs want
	// cadences an order of magnitude apart: this pass walks the whole index and wants to
	// run rarely, the reserve walks one page and earns nothing unless it runs often. One
	// ticker forced whichever ran on the other's setting. It has its own now — see
	// Config.RelocateReserveIntervalMs and runReserveSweeper.
}

// pageCapacityBytes is the total ENTRY capacity of the shard's pages: the bytes
// a write can actually consume, which on an mmap page excludes the pageHdrSize
// prefix holding the persisted head/tail. It is the denominator every occupancy
// figure uses (the runtime alert and the cold-compaction gate alike) so the two
// judge the same shard by the same measure; bytesUsed, the numerator, likewise
// counts only entry bytes (tail-head within entries()).
func (s *shard) pageCapacityBytes() uint64 {
	return uint64(s.numPages()) * uint64(s.maxEntryBytes()) //nolint:gosec // both non-negative
}

// occupancyRatio is live+ghost byte occupancy over the shard's total page entry
// capacity (bytesUsed / pageCapacityBytes), in [0,1]. Read-only; takes no lock
// beyond bytesUsed's RLock. For mmap all pages are mapped at construction so this
// tracks true fill; ghost (expired-but-not-physically-reclaimed) bytes count too.
func (s *shard) occupancyRatio() float64 {
	capacity := s.pageCapacityBytes()
	if capacity == 0 {
		return 0
	}
	return float64(s.bytesUsed()) / float64(capacity)
}

const (
	// mmapOccupancyWarnHigh / ...Low bracket the replicated-mmap page-byte
	// occupancy alert with hysteresis: warn once at High, re-arm below Low so a
	// shard hovering near the mark doesn't log every sweep tick.
	mmapOccupancyWarnHigh = 0.85
	mmapOccupancyWarnLow  = 0.75
)

// checkMmapOccupancy emits a throttled WARNING when a replicated mmap shard's
// page-byte occupancy crosses the high-water — the operator's advance signal
// that ghost bytes are climbing toward the ErrFull fail-closed halt (online
// page-byte reclamation is not possible on mmap; the logical sweeper only frees
// index slots). No-op for heap shards, which DO reclaim page bytes online.
// Occupancy itself is always scrapable via Stats (BytesUsed/BytesAllocated) —
// this is just the alert.
func (s *shard) checkMmapOccupancy() {
	if !s.isMmap {
		return
	}
	r := s.occupancyRatio()
	switch {
	case r >= mmapOccupancyWarnHigh && s.mmapHighWaterWarned.CompareAndSwap(false, true):
		slog.Warn("replicated mmap shard page-byte occupancy high — ghost bytes not reclaimable while running; approaching ErrFull",
			"component", "cache", "occupancy", r, "hint", "RESTART this node to reclaim: cold compaction at shard open rewrites the pages file live-only. Otherwise size persistent replicated shards with TTL-churn headroom")
	case r < mmapOccupancyWarnLow:
		s.mmapHighWaterWarned.Store(false) // re-arm once it recovers
	}
}

// sweepIndex walks the index table tombstoning entries expired at `now`. To avoid
// pinning the exclusive shard lock for an O(slots) scan (which would stall every
// Put/Del on a large shard each tick), it scans the slot array in fixed-size
// batches and releases the lock between them. Reads never take mu, so they run
// throughout. Lazy expiry on Get already guarantees correctness; this pass only
// reclaims index slots. If a Put resizes the table mid-sweep the scan stops and the
// next tick picks up the fresh table. `now` is the wall clock for a non-replicated
// shard and the logical clock (lastAppliedStampMs) for a replicated one — the
// batched-scan mechanics are identical either way.
func (s *shard) sweepIndex(now uint64) {
	t := s.tab.Load()
	n := len(t.ctrl)
	for start := 0; start < n; start += sweepBatchSize {
		end := start + sweepBatchSize
		if end > n {
			end = n
		}
		// Hold the write lock across the whole batch (including page.Read below):
		// this can briefly block PolicyRingbufEvict reads (which take RLock), but
		// the batch is bounded by sweepBatchSize and lazy expiry on Get is the
		// correctness backstop, so the stall is bounded and acceptable. Lock-free
		// PolicyRejectWrites reads are unaffected.
		s.mu.Lock()
		if s.tab.Load() != t {
			// A concurrent Put rehashed the table (which already drops tombstones
			// and stale slots); abandon this pass and let the next tick scan it.
			s.mu.Unlock()
			return
		}
		for i := start; i < end; i++ {
			c := t.ctrl[i].Load()
			if c == ctrlEmpty || c == ctrlTombstone {
				continue
			}
			ref := slabRef(t.refs[i].Load())
			key, _, exp, err := s.pages[ref.pageIdx()].Read(ref.offset())
			if err != nil || isExpired(exp, now) {
				t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index in [0,n)
				if err == nil {
					s.expirations.Add(1)
					// EXPIRY BRANCH ONLY. A slot dropped because its page.Read FAILED has
					// no key to report — the bytes are unreadable — so it notifies nothing
					// and the derived index's reconcile pass is the backstop for that one
					// posting. Notifying with a nil key would be worse than silence.
					s.fireOnRemove(key)
				}
			}
		}
		s.mu.Unlock()
	}
}

// reclaimExpiredHeapPages deterministically frees whole pages whose every live
// entry has expired at the logical stamp, restoring their FreeTail so a committed
// write on a near-full replicated shard does not hit ErrFull — the fix for the
// B3a+B2 availability cliff (#4 Phase B / B3b).
//
// It reclaims by frozen-page RETIREMENT — swapping in a fresh empty page object,
// exactly like the ringbuf eviction path (retirePageLocked) — so the lock-free
// reject-writes read path never observes a mutated or reused page: a reader that
// already loaded the old page pointer reads its immutable frozen bytes, and a
// reader that loads the fresh object sees a mismatched generation and misses. A
// single non-expired, index-current entry PINS the whole page (skipped); we never
// drop live data.
//
// LOCK DISCIPLINE — one write-lock acquisition PER PAGE, released between pages.
// Replicated writes apply through putAtExpLocked, which takes the same s.mu; a
// single hold spanning an O(all-pages) scan (each page decoded under it) would spike
// Raft/PB commit latency exactly under B3b's TTL-churn target workload. So, like
// sweepIndex's batching, each page's decode-decide-retire happens under its own hold
// and the lock is dropped before the next page. len(s.pages) and s.pages[idx] are
// re-read under the lock every iteration, so a Put that appended pages or wrote a
// fresh (possibly live) entry into a page during a released window is always
// observed by that page's next lock acquisition — correctness is preserved across
// the release because each page's decision AND its retirement are atomic within one
// hold (a page is never retired based on a view taken under an earlier, since-
// released lock).
//
// HEAP MODE ONLY. Mmap pages wrap the fixed file region and cannot be swapped for a
// fresh object; reclaiming them in place (drain + reuse) would overwrite bytes a
// concurrent lock-free reject-writes reader still aliases — reject-writes reads
// never take the shard lock, so holding mu would NOT exclude them. Mmap replicated
// shards therefore keep only the index-slot reclamation (sweepIndex) while RUNNING;
// their page bytes are reclaimed by COLD COMPACTION at the next shard open
// (cache/compact.go), which rewrites the file live-only at the one moment no reader
// exists. So on the persistent path the cliff is recoverable by restart — surfaced
// in advance by checkMmapOccupancy — rather than closed continuously.
//
// Determinism: `stamp` is the cross-replica-identical logical clock, and both the
// page contents and the index are identical across replicas (same committed log,
// same order), so the exact same set of pages is retired on every replica. Dropping
// the lock between pages does not affect this: a replica applies the same committed
// writes in the same order regardless of sweeper interleaving, and the sweep's
// removal decision depends only on (page bytes, index, stamp), all deterministic.
func (s *shard) reclaimExpiredHeapPages(stamp uint64) {
	if s.isMmap {
		return // mmap: see doc — cannot retire online; cold compaction at open reclaims.
	}
	// Index-addressed with a fresh length check each iteration: heap pages are only
	// ever appended or swapped (never removed), so idx stays valid, and a page
	// appended during a released window is picked up by the growing bound.
	for idx := 0; ; idx++ {
		s.mu.Lock()
		if idx >= len(s.pages) {
			s.mu.Unlock()
			return
		}
		s.tryRetireExpiredPageLocked(idx, stamp)
		s.mu.Unlock()
	}
}

// tryRetireExpiredPageLocked retires page idx IFF every index-current entry in it
// has expired at `stamp`. Single decode pass: it walks the page once, and the moment
// it finds a current (cur == ref — not a dead duplicate a later Put superseded) and
// NOT-expired entry it bails (the page is pinned by live data). Otherwise it drops
// every current-and-expired slot it collected and swaps in a fresh page. Counts
// reclaimed entries as EXPIRATIONS (TTL reclamation, not eviction); slots sweepIndex
// already tombstoned resolve to !ok and are not re-counted. Must hold mu for writing;
// the decision and the retirement are one atomic unit under this single hold.
func (s *shard) tryRetireExpiredPageLocked(idx int, stamp uint64) {
	p := s.pages[idx]
	if p.Empty() {
		return
	}
	t := s.tab.Load()
	entries := p.entries()
	tail := p.tail()
	// Current-and-expired slots to drop once we confirm the WHOLE page is dead. We
	// cannot tombstone as we go: a live entry found later must leave the page (and
	// all its slots) untouched. Each carries the entry OFFSET as well, so the
	// derived-index removal hook can be given the key when the drop actually
	// happens (the page bytes are still intact then — the fresh page is swapped in
	// only afterwards).
	type expiredSlot struct {
		slot uint64
		off  uint32
	}
	var expiredSlots []expiredSlot
	for cursor := p.head(); cursor < tail; {
		key, value, exp, err := decodeEntryFast(entries[cursor:tail])
		if err != nil {
			// Heap pages carry no CRC and cannot be externally corrupted, so this is
			// unreachable in practice; bail defensively — a page we cannot fully parse
			// is left pinned rather than retired, so we never drop data we could not
			// inspect.
			return
		}
		ref := makeSlabRef(uint16(idx), p.gen, uint32(cursor)) //nolint:gosec // idx ≤ MaxPagesPerShard; cursor < PageSize
		h := hashKey(key)
		if slot, cur, ok := t.findSlot(h); ok && cur == ref {
			if !isExpired(exp, stamp) {
				return // a live, index-current entry pins the whole page.
			}
			expiredSlots = append(expiredSlots, expiredSlot{slot: slot, off: uint32(cursor)}) //nolint:gosec // cursor < PageSize ≤ MaxInt32
		}
		// OCCUPANCY, exactly as in retirePageLocked: another HEAP page walk (it too
		// ends in freshHeapPageLocked), stepping over each entry's room on the page.
		cursor += entrySpan(len(key), len(value), true)
	}
	// No live current entry: drop every expired current slot and retire the page.
	for _, es := range expiredSlots {
		t.tombstone(es.slot)
		s.expirations.Add(1)
		// Collected inside the cur == ref guard above, so each of these IS the live
		// copy of its key — dead duplicates never reach the slice.
		s.fireOnRemoveAt(p, es.off)
	}
	fresh := s.freshHeapPageLocked()
	s.pages[idx] = fresh
	s.pageSlots[idx].Store(fresh) // publish so readers resolve the new generation
	regionNotePage(s, fresh)      // compiled out unless the measurement build tag is set
}

// rehashIfOverThresholdLocked replaces t with a freshly-sized table when its fill
// (live entries plus tombstones) has reached the resize threshold. Call under
// s.mu with t the table the caller just mutated.
//
// The rebuild re-inserts every live entry one by one, so its cost is proportional
// to the shard's entry count, not to the write that tripped it. That is worth
// counting separately from the write path it sits in: the pair of counters says
// how OFTEN the step runs and how much time it takes in total, which is what
// separates a step that is rare and expensive from one that is merely expensive.
//
// Time first, count second, as the other paired counters in this file do: a reader
// that sees the count has necessarily already seen the time it belongs to, so the
// pair can never report rehashes that took no time.
func (s *shard) rehashIfOverThresholdLocked(t *indexTable) {
	if !t.overThreshold() {
		return
	}
	t0 := time.Now()
	s.tab.Store(t.rehashed())
	s.indexRehashNanos.Add(uint64(time.Since(t0))) //nolint:gosec // a duration here is never negative
	s.indexRehashes.Add(1)
}
