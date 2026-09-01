// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
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

	// mmap-only state (nil/zero in heap mode). Guarded by mu.
	file         *os.File
	region       []byte
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
	rejects     atomic.Uint64
	pagesAlloc  atomic.Uint64
	corrupt     atomic.Uint64

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
}

// newShard constructs a shard. dataDir="" selects heap mode (heap-only behavior).
// With a non-empty dataDir the shard is mmap-backed; the directory is created
// if it does not exist.
func newShard(cfg Config, dataDir string) (*shard, error) {
	s := &shard{
		cfg:         cfg,
		dataDir:     dataDir,
		stopSweeper: make(chan struct{}),
	}
	if cfg.NowFn != nil {
		fn := cfg.NowFn
		s.nowFn.Store(&fn)
	}
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

	s.attachMmapRegion(file, region)
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
		// Restore the persisted LOGICAL clock (v3 header; 0 on a v2 file). This is
		// what lets cold compaction judge TTL expiry deterministically on a
		// replicated shard — see cache/compact.go for the full safety argument.
		s.lastAppliedStampMs.Store(readAppliedStamp(region))
		// Restore the persisted PB applied frontier. This is the ONLY
		// thing that lets a restarted PB node describe the FSM it just warm-restarted
		// from rebuildIndexFromPages: there is no PB log and no snapshot to re-derive
		// a frontier from, so without it the engine would present (0,0) — a genesis
		// claim — over real data. (0,0) here means "nothing was ever stamped", which
		// is exactly what a Raft-mode or heap shard yields.
		pbSeq, pbEpoch := readPBFrontier(region)
		s.pbFrontierSeq.Store(pbSeq)
		s.pbFrontierEpoch.Store(pbEpoch)
	}

	writeIdx := -1
	if fresh {
		writeHeader(region, uint32(cfg.PageSize), uint32(maxPages), 0) //nolint:gosec // PageSize and maxPages are validated positive
	} else {
		s.rebuildIndexFromPages()
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
		p := newMmapPage(region[offset : offset+s.cfg.PageSize])
		p.gen = s.nextGen()
		s.pages[i] = p
		s.pageSlots[i].Store(p) // mmap page objects are fixed; publish once
	}
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
// lock. Only mmap-backed ringbuf shards do: eviction overwrites their fixed
// persisted region in place, so a lock-free read could race the writer at the
// byte level. Heap ringbuf retires pages by swapping in a fresh frozen object
// (no in-place overwrite), and every reject-writes shard never overwrites live
// bytes at all, so both read lock-free.
func (s *shard) needsReadLockForGet() bool {
	return s.cfg.AtCapPolicy == PolicyRingbufEvict && s.isMmap
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

	if s.needsReadLockForGet() {
		// mmap ringbuf overwrites live page bytes in place on eviction, so a
		// lock-free read could both observe a torn value AND race the writer's
		// overwrite at the byte level (the race detector flags the latter
		// regardless of any seqlock, which only *detects* torn values after the
		// fact). Take the read lock for the probe and value copy: writers hold the
		// write lock while overwriting, so under RLock the bytes are stable.
		s.mu.RLock()
		t := s.tab.Load()
		v, exp, ref, st := t.get(s, key, h)
		var vCopy []byte
		if st == lkHit {
			// Copy while the lock is held so a later evict+Write can't tear it.
			vCopy = make([]byte, len(v))
			copy(vCopy, v)
		}
		s.mu.RUnlock()
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
				s.dropExpiredLocked(h, ref)
			}
			s.misses.Add(1)
			return nil, 0, ErrNotFound
		}
		return vCopy, exp, nil
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
			s.dropExpiredLocked(h, ref)
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
	s.gets.Add(1)

	if s.needsReadLockForGet() {
		s.mu.RLock()
		t := s.tab.Load()
		v, exp, ref, st := t.get(s, key, h)
		var out []byte
		if st == lkHit {
			out = append(dst, v...) // copy while the lock is held
		}
		s.mu.RUnlock()
		switch st {
		case lkCorrupt:
			s.corrupt.Add(1)
			s.misses.Add(1)
			return dst, ErrNotFound
		case lkMiss:
			s.misses.Add(1)
			return dst, ErrNotFound
		}
		if isExpired(exp, s.now()) {
			if !s.cfg.Replicated {
				s.dropExpiredLocked(h, ref)
			}
			s.misses.Add(1)
			return dst, ErrNotFound
		}
		return out, nil
	}

	// Lock-free path: reject-writes (heap+mmap) and heap-mode ringbuf. The value
	// is always copied into dst, so the owned-slice contract holds for both.
	t := s.tab.Load()
	v, exp, ref, st := t.get(s, key, h)
	switch st {
	case lkCorrupt:
		s.corrupt.Add(1)
		s.misses.Add(1)
		return dst, ErrNotFound
	case lkMiss:
		s.misses.Add(1)
		return dst, ErrNotFound
	}
	if isExpired(exp, s.now()) {
		if !s.cfg.Replicated {
			s.dropExpiredLocked(h, ref)
		}
		s.misses.Add(1)
		return dst, ErrNotFound
	}
	return append(dst, v...), nil
}

// dropExpiredLocked removes the entry for h if it still points at ref (i.e. the
// entry a reader just saw expire has not been overwritten by a newer Put).
// Mirrors the cur == ref guard the map path used.
func (s *shard) dropExpiredLocked(h uint64, ref slabRef) {
	s.mu.Lock()
	t := s.tab.Load()
	if slot, cur, ok := t.findSlot(h); ok && cur == ref {
		t.tombstone(slot)
		s.expirations.Add(1)
	}
	s.mu.Unlock()
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

// putAtExpLocked is the shared write body: it takes the already-resolved
// absolute expiry (exp) and performs the page write + index upsert. putH,
// putAtH, and putAbsH differ ONLY in how they derive exp — the storage path is
// identical.
func (s *shard) putAtExpLocked(key, value []byte, exp uint64, h uint64) error {
	s.puts.Add(1)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Find a page with enough tail room; lazily allocate or evict as needed.
	pageIdx, err := s.findOrMakePageLocked(entrySize(len(key), len(value)))
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
	t.upsert(h, makeSlabRef(uint16(pageIdx), s.pages[pageIdx].gen, off)) //nolint:gosec // pageIdx bounded by MaxPagesPerShard (≤65535)
	if t.overThreshold() {
		s.tab.Store(t.rehashed())
	}
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
		pageIdx, ferr := s.findOrMakePageLocked(entrySize(len(key), 0))
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
			// Eviction removed the key from the index while making room for the
			// tombstone. Nothing is left to strip, the appended record is harmless (the
			// rebuild finds a delete for a key with no live copy), and the entry DID
			// exist when the caller asked — so report the delete as having happened.
			s.dels.Add(1)
			return true, nil
		}
	}
	t.tombstone(slot)
	s.dels.Add(1)
	return true, nil
}

// snapshot returns a Stats snapshot. Caller-side concurrency safe.
func (s *shard) snapshot() Stats {
	// Hits is derived: every read bumps gets on entry and bumps misses on any
	// non-returning path, so Hits = Gets - Misses. Load misses BEFORE gets so an
	// op in flight between the two loads can only leave gets >= misses (gets is
	// always bumped first), never underflowing the subtraction.
	misses := s.misses.Load()
	gets := s.gets.Load()
	return Stats{
		Gets:             gets,
		Hits:             gets - misses,
		Misses:           misses,
		Puts:             s.puts.Load(),
		Dels:             s.dels.Load(),
		Expirations:      s.expirations.Load(),
		Evictions:        s.evictions.Load(),
		Rejects:          s.rejects.Load(),
		PagesAllocated:   s.pagesAlloc.Load(),
		BytesAllocated:   uint64(s.numPages()) * uint64(s.cfg.PageSize), //nolint:gosec // numPages and PageSize are always non-negative
		BytesUsed:        s.bytesUsed(),
		CorruptionErrors: s.corrupt.Load(),

		Compactions:              s.compactions.Load(),
		CompactionsAborted:       s.compactAborts.Load(),
		CompactionBytesReclaimed: s.compactBytesReclaimed.Load(),
		CompactionDurationMs:     s.compactNanos.Load() / uint64(time.Millisecond),

		ReclaimableBytes:     s.reclaimableBytesForStats(),
		OnlineRelocations:    s.relocations.Load(),
		OnlineBytesRelocated: s.relocatedBytes.Load(),
		OnlinePagesRetired:   s.relocatePagesGone.Load(),
		OnlinePagesRecycled:  s.relocatePagesRecycled.Load(),
	}
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
		_ = msync(s.file, s.region) // best-effort final sync
	}
	return munmapAndClose(s.file, s.region)
}

// --- internals ---

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
func (s *shard) rebuildIndexFromPages() {
	now := s.now()
	var maxSeq uint64
	for pageIdx, p := range s.pages {
		head := p.head()
		tail := p.tail()
		if head == tail {
			continue
		}
		entries := p.entries()
		cursor := head
		for cursor < tail {
			key, value, _, meta, err := decodeEntry(entries[cursor:tail])
			if err != nil {
				// Corrupt entry (e.g. a crash-torn write): the rest of this page
				// is unreadable. Record it so recovery-time loss is observable —
				// consistent with the runtime read paths (getH/getIntoH) that also
				// bump s.corrupt — rather than swallowing it silently.
				s.corrupt.Add(1)
				slog.Warn("corrupt entry during recovery; truncating page tail",
					"component", "cache", "page", pageIdx, "offset", cursor, "tail", tail, "err", err, "truncate_to", cursor)
				// Truncate the page's persisted framing to the validated prefix so
				// the torn region is excluded from every future eviction walk
				// (EvictFront frames entries from raw bytes without a CRC). Reset if
				// nothing valid preceded the corruption.
				if cursor == head {
					p.Reset()
				} else {
					p.setTail(cursor)
				}
				break
			}
			esize := entrySize(len(key), len(value))
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
			if t.overThreshold() {
				s.tab.Store(t.rehashed())
			}
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
		if meta, ok := p.MetaAt(ref.offset()); ok && metaIsTombstone(meta) {
			t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index
			continue
		}
		if dropExpired {
			if _, _, exp, err := p.Read(ref.offset()); err == nil && isExpired(exp, now) {
				t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index
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
	// Fast path: the current open page still has contiguous tail room.
	if s.writeIdx < len(s.pages) && s.pages[s.writeIdx].FreeTail() >= need {
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

// allocHeapPageLocked appends a fresh heap-backed page and returns its index.
// Must be called with s.mu held for writing (or from the constructor before
// the shard is shared).
func (s *shard) allocHeapPageLocked() int {
	p := newHeapPage(s.cfg.PageSize)
	p.gen = s.nextGen()
	idx := len(s.pages)
	s.pages = append(s.pages, p)
	s.pageSlots[idx].Store(p) // publish for the lock-free read path
	s.pagesAlloc.Add(1)
	return idx
}

// firstPageWithRoomLocked returns the index of the first page with at least
// `need` bytes of contiguous tail room, or -1 if none. Must be called with s.mu
// held (read or write). Shared by findOrMakePageLocked and evictUntilFitsLocked
// so the all-pages free-space scan lives in one place.
func (s *shard) firstPageWithRoomLocked(need int) int {
	for i := range s.pages {
		// Skip pages retired by online relocating compaction: their stale-framed bytes
		// are immutable and stranded (see page.retired). `retired` is always false on
		// heap / single-node / ringbuf shards, so this is a no-op there.
		if s.pages[i].retired {
			continue
		}
		if s.pages[i].FreeTail() >= need {
			return i
		}
	}
	return -1
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
		n := len(s.pages)
		victim := -1
		for off := 0; off < n; off++ {
			i := (s.nextVictim + off) % n
			if !s.pages[i].Empty() {
				victim = i
				break
			}
		}
		if victim < 0 {
			return 0, ErrCannotEvict
		}
		// Advance the rotation cursor past this victim so the next eviction moves
		// on to the following page rather than re-selecting a just-refilled one.
		s.nextVictim = (victim + 1) % n
		if err := s.evictVictimLocked(victim); err != nil {
			return 0, err
		}
	}
}

// evictVictimLocked frees page `victim` in full. In heap mode it retires the
// page — swapping in a fresh frozen object so lock-free readers stay safe (see
// retirePageLocked). In mmap mode it drains the fixed persisted region in place
// under the write lock (see drainPageLocked), because mmap page objects wrap the
// file and cannot be swapped for a fresh allocation. Must hold mu for writing.
func (s *shard) evictVictimLocked(victim int) error {
	if !s.isMmap {
		s.retirePageLocked(victim)
		return nil
	}
	return s.drainPageLocked(victim)
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
	for cursor := old.head(); cursor < tail; {
		key, value, _, err := decodeEntryFast(entries[cursor:tail])
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
		}
		s.evictions.Add(1)
		cursor += entrySize(len(key), len(value))
	}
	fresh := newHeapPage(s.cfg.PageSize)
	fresh.gen = s.nextGen()
	s.pages[idx] = fresh
	s.pageSlots[idx].Store(fresh) // publish so readers resolve the new generation
}

// drainPageLocked evicts every live entry from page victim, dropping each from
// the index, until the page is empty. If EvictFront reports torn framing
// (recovered from a crash-corrupted mmap region — a stale/garbage entry header
// that decodeEntry would have rejected), the page is treated as corrupt: the
// event is counted and logged and the page is Reset so the shard regains write
// availability instead of failing every future space-needing Put. Stale index
// slots left pointing into the reset region resolve to misses on Read and are
// reclaimed by the sweeper. Must be called with s.mu held for writing.
//
// This is the MMAP-ONLY eviction path (see evictVictimLocked): it mutates the
// page's fixed persisted region in place, which is safe because mmap ringbuf
// reads hold the read lock. Heap ringbuf never reaches here — it retires pages
// by frozen replacement (retirePageLocked) so its reads can stay lock-free.
func (s *shard) drainPageLocked(victim int) error {
	for !s.pages[victim].Empty() {
		// Capture the entry's offset BEFORE EvictFront advances head, so we
		// can reconstruct the slabRef of the copy being physically removed.
		off := uint32(s.pages[victim].head()) //nolint:gosec // head < PageSize which is validated ≤ MaxInt32
		evictedKey, _, err := s.pages[victim].EvictFront()
		if err != nil {
			s.corrupt.Add(1)
			slog.Warn("corrupt entry during eviction; resetting page", "component", "cache", "page", victim, "offset", off, "err", err)
			s.pages[victim].Reset()
			return nil
		}
		// Only drop the index slot if it still points at THIS physical copy.
		// A later Put may have overwritten the key with a newer entry in a
		// different page (leaving these bytes as a dead duplicate), or an
		// unrelated key may collide on this hash. Deleting unconditionally
		// would silently evict live data. Mirror the cur == ref guard in getH.
		ref := makeSlabRef(uint16(victim), s.pages[victim].gen, off) //nolint:gosec // victim bounded by MaxPagesPerShard (≤65535)
		h := hashKey(evictedKey)
		t := s.tab.Load()
		if slot, cur, ok := t.findSlot(h); ok && cur == ref {
			t.tombstone(slot)
		}
		s.evictions.Add(1)
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
			// No stamped apply has advanced the logical clock yet — reclaiming
			// against 0 would be both pointless (isExpired(exp,0) is always false)
			// and, if we ever changed that, non-deterministic. Do nothing.
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
		s.maybeRelocateCompact(stamp)
		return
	}
	s.sweepIndex(s.now())
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
			_, _, exp, err := s.pages[ref.pageIdx()].Read(ref.offset())
			if err != nil || isExpired(exp, now) {
				t.tombstone(uint64(i)) //nolint:gosec // i is a valid slot index in [0,n)
				if err == nil {
					s.expirations.Add(1)
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
	// all its slots) untouched.
	var expiredSlots []uint64
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
			expiredSlots = append(expiredSlots, slot)
		}
		cursor += entrySize(len(key), len(value))
	}
	// No live current entry: drop every expired current slot and retire the page.
	for _, slot := range expiredSlots {
		t.tombstone(slot)
		s.expirations.Add(1)
	}
	fresh := newHeapPage(s.cfg.PageSize)
	fresh.gen = s.nextGen()
	s.pages[idx] = fresh
	s.pageSlots[idx].Store(fresh) // publish so readers resolve the new generation
}
