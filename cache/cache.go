// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// Cache is a sharded in-memory KV store with lazy slab pool allocation
// and per-shard TTL. All operations are safe for concurrent use.
type Cache struct {
	cfg     Config
	shards  []*shard
	mask    uint64
	closed  atomic.Bool
	closeCh chan struct{}
	wg      sync.WaitGroup

	// onRemove is the derived-secondary-index removal hook installed by
	// SetOnRemove; nil by default. See cache/onremove.go for the full contract
	// (fires under the shard write lock, key aliases the page, hint channel only).
	// Every shard holds a POINTER to this one atomic, so installing or clearing the
	// hook is a single store observed by all of them, and a shard never needs a
	// back-pointer to the Cache.
	onRemove atomic.Pointer[func([]byte)]
}

// New constructs a Cache with the given configuration.
func New(cfg Config) (*Cache, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("cache: %w", err)
	}
	c := &Cache{
		cfg:     cfg,
		shards:  make([]*shard, cfg.NumShards),
		mask:    uint64(cfg.NumShards - 1), //nolint:gosec // NumShards is validated to be a positive power of two
		closeCh: make(chan struct{}),
	}
	for i := 0; i < cfg.NumShards; i++ {
		var sd string
		if cfg.DataDir != "" {
			sd = filepath.Join(cfg.DataDir, fmt.Sprintf("shard-%04d", i))
		}
		s, err := newShard(cfg, sd, &c.onRemove)
		if err != nil {
			// Roll back already-constructed shards.
			for j := 0; j < i; j++ {
				_ = c.shards[j].Close()
			}
			return nil, fmt.Errorf("cache: shard %d: %w", i, err)
		}
		c.shards[i] = s
	}
	if cfg.Durable && cfg.DataDir != "" {
		c.wg.Add(1)
		go c.msyncLoop()
	}
	return c, nil
}

// msyncLoop periodically flushes every mmap-backed shard's pages to disk.
// Runs only when Config.Durable is set; ticks every MsyncIntervalMs.
func (c *Cache) msyncLoop() {
	defer c.wg.Done()
	t := time.NewTicker(time.Duration(c.cfg.MsyncIntervalMs) * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			// A failing msync under -durable means pages did NOT reach disk while
			// the operator believes they did (silent durability loss exactly when
			// the disk is going bad). Surface it — coalesced to one line per tick
			// so a persistently-failing disk does not flood the log.
			failed, lastErr := 0, error(nil)
			for _, s := range c.shards {
				if s.isMmap {
					// Same crash-ordered projection as SetPBFrontier, minus a watermark:
					// snapshot the runtime bounds, flush the DATA, then project the
					// snapshot into the durable headers and flush a SECOND time. The
					// periodic durable flush must advance the durable per-page bounds so
					// a warm restart sees the data it wrote, but it must never project a
					// bound over an entry the data flush did not capture (forward skew),
					// which the snapshot-before-flush order prevents.
					snaps := s.snapshotPageBounds()
					if err := msync(s.file, s.region); err != nil {
						// Data not durable this tick: do NOT project the snapshot into
						// the durable headers. Projecting a bound whose entries the
						// failed flush did not capture is the forward-skew hazard; skip
						// it and leave the last durable bounds in place (a safe lag).
						failed++
						lastErr = err
						continue
					}
					s.projectSnapshotBounds(snaps)
					if err := msync(s.file, s.region); err != nil {
						failed++
						lastErr = err
					}
				}
			}
			if failed > 0 {
				slog.Warn("DURABILITY WARNING: msync failed this flush — pages may not be on disk", "component", "cache", "shards_failed", failed, "last_err", lastErr)
			}
		case <-c.closeCh:
			return
		}
	}
}

// NumShards returns the configured shard count.
func (c *Cache) NumShards() int { return c.cfg.NumShards }

// AtCapPolicy returns the configured at-capacity policy (evict vs reject). Used
// by the shard layer to verify replicated shards were forced to reject-writes.
func (c *Cache) AtCapPolicy() AtCapPolicy { return c.cfg.AtCapPolicy }

// LastAppliedStampMs returns the MAX per-shard logical clock across all shards —
// the largest apply-stamp any shard has folded in on the stamped apply path (#4
// Phase B / B3b). Zero means no stamped apply has landed on any shard yet.
//
// The replicated leader clamps each new apply-stamp to be >= this value (see
// shard/store.go applyOpIndexed) so stamps are monotonic non-decreasing. The GLOBAL
// max (not a per-shard value) is used deliberately: a single log entry is stamped
// once, before its target shard is known, so the clamp must dominate EVERY shard's
// logical clock to guarantee the new stamp is >= any exp the per-shard sweeper
// could have already reclaimed against — the invariant that makes sweeper-vs-write
// races safe. See the shard.lastAppliedStampMs field doc for the full argument.
func (c *Cache) LastAppliedStampMs() uint64 {
	var maxStamp uint64
	for _, s := range c.shards {
		if v := s.LastAppliedStampMs(); v > maxStamp {
			maxStamp = v
		}
	}
	return maxStamp
}

// AdvanceAppliedStamp folds stampMs into EVERY shard's logical clock as a
// running max (a no-op for shards already at or past it). It is the snapshot
// installer's entry point: PutAbs deliberately does not advance the clock (an
// absolute expiry carries no information about WHEN it was applied), so without
// this a node that acquired all of its committed state from a snapshot would
// hold clock 0 while its peers hold the leader's.
//
// That gap is not cosmetic. The logical clock is what the B3b sweeper reclaims
// against and what cold compaction judges TTL expiry against (cache/compact.go),
// and the safety argument for both is that any FUTURE committed write carries a
// stamp >= every replica's persisted clock — which holds because the leader
// clamps new stamps to >= its own clock. A leader at 0 would stamp at bare wall
// time and could fall BELOW a peer's persisted clock, retroactively invalidating
// reclamation that peer has already performed. Folding the snapshot's clock in
// keeps the clamp input correct on a snapshot-restored node.
//
// Applying it to every shard (rather than a per-shard value) matches the clamp,
// which reads the GLOBAL max: see LastAppliedStampMs.
func (c *Cache) AdvanceAppliedStamp(stampMs uint64) {
	if stampMs == 0 {
		return
	}
	for _, s := range c.shards {
		s.advanceAppliedStamp(stampMs)
	}
}

// ShardIndex returns the index in [0, NumShards) of the shard that owns key,
// using the same partitioning as shardForH. Callers that serialize work per
// shard (e.g. the Direct backend's per-shard op locks) use it to lock the exact
// shard a key's Get/Put/Del will touch, so independent keys don't contend.
func (c *Cache) ShardIndex(key []byte) int { return int(hashKey(key) & c.mask) }

// shardForH returns the precomputed hash and the shard that owns it, so
// the hot Get/Put/Del paths hash the key exactly once instead of twice
// (once for shard selection, once again inside the shard for its index
// lookup).
func (c *Cache) shardForH(key []byte) (uint64, *shard) {
	h := hashKey(key)
	return h, c.shards[h&c.mask]
}

// Get returns the value for key.
func (c *Cache) Get(key []byte) ([]byte, error) {
	h, s := c.shardForH(key)
	return s.getH(key, h)
}

// GetInto appends the value for key to dst and returns the extended slice. It is
// the allocation-free counterpart to Get: a caller in a hot loop passes its
// reused buffer (dst[:0]) and incurs zero allocations per hit, instead of the
// fresh []byte Get returns each call. The value is still COPIED out of the
// cache's backing store (just into dst rather than a new allocation), so the
// result is a safe, caller-owned copy — never an alias into the shared arena.
// On a miss or expiry it returns dst unchanged with ErrNotFound.
func (c *Cache) GetInto(dst, key []byte) ([]byte, error) {
	h, s := c.shardForH(key)
	return s.getIntoH(dst, key, h)
}

// GetWithExpiryInto is GetInto that ALSO surfaces the entry's stored absolute
// expiry (ms since epoch; 0 = no expiry) — GetWithExpiry's allocation-free
// counterpart. Like GetInto the value is COPIED into dst, never aliased.
func (c *Cache) GetWithExpiryInto(dst, key []byte) (val []byte, expiryMs uint64, err error) {
	h, s := c.shardForH(key)
	return s.getIntoWithExpiryH(dst, key, h)
}

// GetWithExpiryIntoAt is GetWithExpiryInto with expiry evaluated against the
// EXPLICIT clock nowMs rather than the wall clock — the apply-path counterpart,
// matching GetWithExpiryAt. Keeping the clock explicit is what lets a replicated
// apply use the pooled read without taking a wall-clock dependency.
func (c *Cache) GetWithExpiryIntoAt(dst, key []byte, nowMs uint64) (val []byte, expiryMs uint64, err error) {
	h, s := c.shardForH(key)
	return s.getIntoWithExpiryAtH(dst, key, h, nowMs)
}

// Put inserts or replaces the value for key with the given TTL.
// A TTL of zero means no expiry.
func (c *Cache) Put(key, value []byte, ttl time.Duration) error {
	h, s := c.shardForH(key)
	return s.putH(key, value, ttl, h)
}

// GetAt is Get with expiry evaluated against the EXPLICIT clock nowMs rather
// than the wall clock (#4 Phase B / B1). It is the read primitive the replicated
// apply path uses (via TxContext.Get under an apply stamp) so a committed-write's
// view of what is still live — and its deterministic tombstoning of what is not —
// is judged on the leader-stamped clock baked into the log entry, identically on
// every replica. Semantics otherwise match Get.
func (c *Cache) GetAt(key []byte, nowMs uint64) ([]byte, error) {
	h, s := c.shardForH(key)
	return s.getAtH(key, h, nowMs)
}

// PutAt is Put with the absolute expiry computed as nowMs + ttl from the
// EXPLICIT leader-stamped clock rather than the wall clock (#4 Phase B / B1), so
// every replica applying the same committed entry stores byte-identical absolute
// expiries. A TTL of zero means no expiry. It is the write primitive the
// replicated apply path uses (via TxContext.Put under an apply stamp).
func (c *Cache) PutAt(key, value []byte, ttl time.Duration, nowMs uint64) error {
	h, s := c.shardForH(key)
	return s.putAtH(key, value, ttl, nowMs, h)
}

// NowMs returns the cache's EFFECTIVE wall-clock now in milliseconds — the same
// clock the non-apply read path judges liveness with (an injected SetNowFunc
// override, else real wall time), NOT raw time.Now. The read-only ttl op turns a
// stored absolute expiry into a remaining-ms value against this so it can never
// report a key expired that the same-clock read would still accept. SetNowFunc
// sets one nowFn across every shard, so shard 0's clock is that shared clock.
func (c *Cache) NowMs() uint64 { return c.shards[0].now() }

// GetWithExpiry is Get that ALSO returns the entry's stored absolute expiry
// (ms since epoch; 0 = no expiry). It is the read primitive the ttl / persist /
// incr_ex ops need to inspect (and, for incr_ex, preserve) a key's deadline. It
// evaluates liveness against the wall clock exactly like Get and returns
// ErrNotFound for an absent or expired key.
//
// The returned val has EXACTLY Get's value-ownership contract (it shares Get's
// code path, only additionally surfacing the expiry): under PolicyRingbufEvict
// it is a freshly-allocated owned copy, but under PolicyRejectWrites it ALIASES
// the page backing store and must not be retained across subsequent writes to
// this shard — copy it if you need to. It does NOT unconditionally copy; see
// shard.Get for the full contract.
func (c *Cache) GetWithExpiry(key []byte) (val []byte, expiryMs uint64, err error) {
	h, s := c.shardForH(key)
	return s.getWithExpiryH(key, h)
}

// GetWithExpiryAt is GetWithExpiry with expiry evaluated against the EXPLICIT
// clock nowMs rather than the wall clock — the apply-path counterpart used by
// TxContext.GetWithExpiry under a leader apply stamp, mirroring GetAt. The
// returned val carries the same ownership contract as Get/GetAt (see
// GetWithExpiry): an owned copy under ringbuf, a page alias under reject-writes.
func (c *Cache) GetWithExpiryAt(key []byte, nowMs uint64) (val []byte, expiryMs uint64, err error) {
	h, s := c.shardForH(key)
	return s.getWithExpiryAtH(key, h, nowMs)
}

// PutAbs inserts key with a pre-computed ABSOLUTE expiry (ms since epoch; 0 =
// no expiry), bypassing any TTL→expiry conversion. Snapshot restore uses it to
// install the exact expiry recorded in the snapshot verbatim, so two followers
// restoring the same snapshot at different wall times produce logically
// byte-identical state (identical key/value/exp set; #4 Phase B / B1).
func (c *Cache) PutAbs(key, value []byte, expiryMs uint64) error {
	h, s := c.shardForH(key)
	return s.putAbsH(key, value, expiryMs, h)
}

// SetNowFunc overrides the wall-clock source consulted by the non-apply expiry
// sites (client read filter, warm-restart rebuild, Iterate; the sweeper is off
// under replication) across every shard. nil restores the real clock. It is a
// TEST/advanced seam — production never calls it, so the default path is
// byte-identical to time.Now. Callers must quiesce traffic before swapping the
// clock if they need a globally-consistent instant (e.g. a canonical fingerprint):
// the store is atomic but a mid-flight swap otherwise races the meaning of
// concurrent reads. It does NOT affect the apply path (PutAt/GetAt take an
// explicit stamp).
func (c *Cache) SetNowFunc(fn func() uint64) {
	for _, s := range c.shards {
		if fn == nil {
			s.nowFn.Store(nil)
			continue
		}
		f := fn
		s.nowFn.Store(&f)
	}
}

// Del removes the entry for key. Returns true if the entry was present.
//
// It can return ErrFull on a PERSISTENT shard: a delete there is recorded as a
// tombstone ENTRY on the page so it survives a warm restart, and appending that
// record needs room. See shard.delH for why the alternative (an in-place flag) is
// not crash-safe, and why failing here is better than a delete that silently does
// not persist. Every other shard mode returns a nil error.
func (c *Cache) Del(key []byte) (bool, error) {
	h, s := c.shardForH(key)
	return s.delH(key, h)
}

// Stats returns aggregated counters across all shards.
func (c *Cache) Stats() Stats {
	var agg Stats
	for _, s := range c.shards {
		agg.Add(s.snapshot())
	}
	return agg
}

// AppliedIndex returns the minimum applied-index across all shards.
// In mmap mode, this is read from each shard's header. In heap mode,
// returns 0.
func (c *Cache) AppliedIndex() uint64 {
	minIdx := ^uint64(0)
	for _, s := range c.shards {
		if !s.isMmap {
			return 0
		}
		if v := s.appliedIndex.Load(); v < minIdx {
			minIdx = v
		}
	}
	if minIdx == ^uint64(0) {
		return 0
	}
	return minIdx
}

// PBFrontier returns the persisted primary-backup applied frontier: the
// (seq, epoch) identity of the newest PB write these pages are known to
// materialize. It is restored from each shard's header at open, so this is the
// value a PB engine rebuilds its log identity from after a restart.
//
// It reports the MINIMUM across shards (and the pair is taken from the shard that
// holds that minimum — a seq from one shard and an epoch from another would name a
// write that never existed). Min, not max, for the same reason AppliedIndex uses
// min and for the reason that governs this whole field: a crash can interrupt
// SetPBFrontier's per-shard loop, leaving some shards stamped with the new pair
// and some with the old. Every shard's DATA covers at least the minimum, so the
// minimum is the strongest claim that is true of all of them. Any higher choice
// would over-report for the shards that never got the newer stamp.
//
// Returns (0, 0) — genesis — in heap mode or if any shard is non-mmap.
func (c *Cache) PBFrontier() (seq, epoch uint64) {
	minSeq := ^uint64(0)
	var minEpoch uint64
	for _, s := range c.shards {
		if !s.isMmap {
			return 0, 0
		}
		if v := s.pbFrontierSeq.Load(); v < minSeq {
			minSeq = v
			minEpoch = s.pbFrontierEpoch.Load()
		}
	}
	if minSeq == ^uint64(0) {
		return 0, 0
	}
	return minSeq, minEpoch
}

// msyncTestHook, when non-nil, is invoked in place of the real msync at the
// watermark sync points (SetPBFrontier / SetAppliedIndex). It is a TEST-ONLY seam —
// nil in production, so the only production cost is one nil-pointer load per flush —
// and it lets a test observe the ORDER and coverage of those paths' flushes, which
// is where the entries→bounds→watermark ordering is enforced. A hook must call the
// real msync itself if durability is wanted.
var msyncTestHook func(f *os.File, region []byte) error

// syncRegion is msync routed through the test seam. Only the watermark paths use it,
// because they are the only ones whose flush ORDERING a test needs to verify.
func syncRegion(f *os.File, region []byte) error {
	if h := msyncTestHook; h != nil {
		return h(f, region)
	}
	return msync(f, region)
}

// SetPBFrontier persists the primary-backup applied frontier (seq, epoch) into
// every shard's header, CRASH-ORDERED.
//
// THE ORDERING IS THE WHOLE POINT, and it is the same argument SetAppliedIndex
// makes for force=true: the page data region is msync'd BEFORE the header carrying
// the new frontier is written and msync'd. So at every instant a crash can observe,
// the persisted watermark names a write whose data is already on disk. The
// watermark can therefore lag reality (harmless: a restarted node is offered a
// delta from further back and log matching accepts it as a true prefix) but can
// never lead it (catastrophic: the node would claim a prefix it does not hold, and
// pbisr's log-matching check — which compares an incoming frame against THIS
// number — would certify a divergent append).
//
// There is deliberately NO force=false variant. PB applies one write at a time and
// exists for write throughput, so the msync is amortised by the CALLER stamping
// periodically (shard.pbFrontierStamper) rather than by making the individual write
// cheap. Amortising leaves the watermark behind by at most one interval, which is
// the safe direction; skipping the msync ordering would let the header reach disk
// ahead of the pages, which is the unsafe one.
//
// Callers must pass a frontier that is already fully materialized into the cache
// (every write up to and including it has returned from apply). Given that, the
// msync below flushes those writes' pages before the header names them.
//
// WHY THE DATA msync IS NOT UNDER s.mu (and SetAppliedIndex's is).
// A full-region msync is O(region size) — it walks a 256 MiB VMA by default — and
// holding the shard lock across it stalls EVERY writer for its duration, which
// measured as a >2x throughput loss on the PB write path. It does not need the
// lock: the ordering property is "every write <= seq is on disk before the header
// names it", and those writes returned from apply BEFORE the caller recorded seq,
// so their pages are already dirty when this msync starts and it flushes them.
// Writes that land DURING the msync only concern seqs ABOVE the one being stamped
// — they can make the flush do more work, never less, and their absence from disk
// cannot make this header over-report. The lock is still taken for the header
// mutation itself, which is what serializes it against SetAppliedIndex's header
// write (they touch disjoint bytes but not disjoint cache lines) and keeps the
// mirrored atomics consistent with the bytes.
//
// SetAppliedIndex keeps the lock across all of its msyncs (data, bounds, header).
// That is deliberately left alone: it is the Raft path, it already amortises per
// Apply BATCH, and the durable frontier must not change Raft behaviour.
func (c *Cache) SetPBFrontier(seq, epoch uint64) {
	for _, s := range c.shards {
		if !s.isMmap {
			continue
		}
		// Crash-ordered flush with the DATA msync deliberately UNLOCKED (throughput —
		// see above), which is why the bounds are SNAPSHOTTED before it rather than
		// read live after it: a write landing during the unlocked msync must not have
		// its (non-durable) tail projected as a durable bound.
		//
		// 1. Snapshot each page's runtime bounds + generation under s.mu.
		snaps := s.snapshotPageBounds()
		// 2. Flush the page DATA, unlocked. Every write <= seq returned from apply
		//    before the caller recorded seq, so its bytes are already dirty and this
		//    flushes them; a concurrent higher-seq write only adds work.
		if err := syncRegion(s.file, s.region); err != nil {
			// Data not durable: STOP this shard's protocol here. Projecting bounds
			// or stamping the frontier now would let a durable watermark name
			// entries that never reached disk (over-report — the catastrophic
			// direction). Leaving the old, lower frontier in place (in memory and on
			// disk) is the safe under-report; a caller stamps again next interval.
			slog.Warn("DURABILITY WARNING: page msync failed before pb frontier", "component", "cache", "pb_seq", seq, "pb_epoch", epoch, "err", err)
			continue
		}
		// The ordering below is ENTRIES → BOUNDS → WATERMARK, each made durable by its
		// OWN msync strictly after the previous. A single msync over bounds AND the
		// frontier would be wrong: msync gives no writeback ordering ACROSS OS pages,
		// so a crash mid-flush could persist the frontier (page 0) while a page K bound
		// (a different OS page) reverted to its pre-append value — recovery would then
		// frame fewer entries than the frontier claims, i.e. the watermark would LEAD
		// reality (the catastrophic direction this field exists to prevent, see the doc
		// above). Two flushes keep the frontier strictly last.
		//
		// 3. Project the SNAPSHOT bounds into each durable header ((0,0) for a page
		//    recycled during the unlocked window). The projected bounds name only
		//    entries the data msync above flushed.
		s.projectSnapshotBounds(snaps)
		// 4. FULL-REGION msync flushes the projected per-page bounds — scattered at
		//    headerSize+i*PageSize, so a header-only slice cannot reach them — BEFORE
		//    the frontier is stamped. No frontier is on disk yet, so an interrupted
		//    flush here just leaves the old (lower) frontier: the safe under-report.
		if err := syncRegion(s.file, s.region); err != nil {
			// Bounds not durable: same rule — do NOT stamp the frontier. The old
			// frontier stays; recovery will under-report, never over-report.
			slog.Warn("DURABILITY WARNING: bounds msync failed before pb frontier", "component", "cache", "pb_seq", seq, "pb_epoch", epoch, "err", err)
			continue
		}
		// 5. Stamp the frontier and flush ONLY the 128-byte file header, strictly after
		//    the bounds it depends on are durable. Now every write the frontier names
		//    is both on disk (step 2) and recoverable under a durable bound (step 4)
		//    before the frontier that names it reaches disk. The in-memory frontier is
		//    advanced only here, after the two data/bounds barriers succeeded, so it
		//    can never advertise a seq whose entries are not durable.
		s.mu.Lock()
		setPBFrontier(s.region, seq, epoch)
		s.pbFrontierSeq.Store(seq)
		s.pbFrontierEpoch.Store(epoch)
		err := syncRegion(s.file, s.region[:headerSize])
		s.mu.Unlock()
		if err != nil {
			// Data and bounds are durable; only the frontier header did not land. On
			// restart recovery sees the old frontier over durable data — under-report,
			// safe. The in-memory frontier is advanced (its data IS durable), so a
			// live report is correct; the next stamp re-persists it.
			slog.Warn("DURABILITY WARNING: header msync failed for pb frontier", "component", "cache", "pb_seq", seq, "pb_epoch", epoch, "err", err)
		}
	}
}

// SetAppliedIndex updates the applied-index in every shard's header.
//
// It ALSO persists each shard's logical clock (lastAppliedStampMs) into the
// same header write. The two are deliberately stamped together: the pair
// (appliedIndex, lastAppliedStampMs) is written from a state where every entry
// up to appliedIndex has already been applied, so the persisted stamp is always
// >= the max stamp of every entry the persisted index covers. That is the exact
// invariant cold compaction at open relies on (see cache/compact.go) — restoring
// the stamp can never over-state the shard's logical clock relative to the
// committed entries the shard is about to replay from.
//
// If force is true the write is made crash-consistent with three ordered flushes:
// the page DATA is msync'd, THEN the projected per-page BOUNDS, THEN the header
// carrying the new applied-index — each strictly after the previous. This
// guarantees the persisted watermark never advances ahead of the entries it
// commits NOR ahead of the durable bounds that make those entries recoverable, so
// a crash cannot leave the header claiming an index whose data was not flushed or
// whose bounds do not yet frame it (either of which would make fsm.Apply
// skip-replay entries that are not durably recoverable). The bounds get their own
// msync because they are scattered at every page header, not in region[:headerSize],
// so a header-only flush cannot reach them and a single combined flush gives no
// cross-OS-page ordering (see the body). If force is false, both the header and
// page data rely on opportunistic OS flushing (or the Durable msyncLoop) with no
// ordering guarantee.
func (c *Cache) SetAppliedIndex(idx uint64, force bool) {
	for _, s := range c.shards {
		if !s.isMmap {
			continue
		}
		s.mu.Lock()
		setAppliedStamp(s.region, s.lastAppliedStampMs.Load())
		if force {
			// Crash-ordered durable flush, ENTRIES → BOUNDS → WATERMARK, each made
			// durable by its OWN msync strictly after the previous. This whole path
			// holds s.mu, so no writer appends between the steps and the CURRENT runtime
			// bounds are exactly the bounds at data-flush time — no snapshot is needed.
			// The three flushes are NOT collapsible: msync gives no writeback ordering
			// across OS pages, so folding bounds and the applied-index into one flush
			// could persist the index (page 0) while a page K bound reverted, leaving
			// the watermark naming entries recovery cannot frame — fsm.Apply would then
			// skip-replay entries that are not durably recoverable.
			//
			// A failed barrier STOPS the protocol: we must not project bounds over,
			// or stamp/advance the index for, entries that are not durable. On any
			// failure below the old, lower applied-index stays (in memory and on
			// disk) — recovery under-reports, never over-reports.
			//
			// 1. Flush the page DATA so the bounds and watermark can't outrun the
			//    entries they name.
			dataOK := true
			if err := syncRegion(s.file, s.region); err != nil {
				slog.Warn("DURABILITY WARNING: page msync failed before applied-index", "component", "cache", "applied_index", idx, "err", err)
				dataOK = false
			}
			if dataOK {
				// 2. Project every mmap page's runtime bounds into its durable header
				//    (they are scattered at headerSize+i*PageSize) and flush them,
				//    BEFORE the applied index is stamped. No new index is on disk yet,
				//    so an interrupted flush here just leaves the old (lower) index.
				for _, p := range s.pages {
					p.projectBounds(p.head(), p.tail())
				}
				if err := syncRegion(s.file, s.region); err != nil {
					slog.Warn("DURABILITY WARNING: bounds msync failed before applied-index", "component", "cache", "applied_index", idx, "err", err)
				} else {
					// 3. Stamp the applied index and flush ONLY the 128-byte header,
					//    strictly after the bounds it depends on are durable, and
					//    advance the in-memory index only now — so it can never name
					//    entries a crash left unrecoverable.
					setAppliedIndex(s.region, idx)
					s.appliedIndex.Store(idx)
					if err := syncRegion(s.file, s.region[:headerSize]); err != nil {
						slog.Warn("DURABILITY WARNING: header msync failed for applied-index", "component", "cache", "applied_index", idx, "err", err)
					}
				}
			}
		} else {
			setAppliedIndex(s.region, idx)
			s.appliedIndex.Store(idx)
		}
		s.mu.Unlock()
	}
}

// Close stops background sweepers and, in mmap mode, flushes and unmaps all
// shard regions. Idempotent.
func (c *Cache) Close() error {
	if !c.closed.CompareAndSwap(false, true) {
		return nil
	}
	close(c.closeCh)
	c.wg.Wait()
	var firstErr error
	for _, s := range c.shards {
		if err := s.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
