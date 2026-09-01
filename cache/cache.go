// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"log/slog"
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
		s, err := newShard(cfg, sd)
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
		x := s.snapshot()
		agg.Gets += x.Gets
		agg.Hits += x.Hits
		agg.Misses += x.Misses
		agg.Puts += x.Puts
		agg.Dels += x.Dels
		agg.Expirations += x.Expirations
		agg.Evictions += x.Evictions
		agg.Rejects += x.Rejects
		agg.PagesAllocated += x.PagesAllocated
		agg.BytesAllocated += x.BytesAllocated
		agg.BytesUsed += x.BytesUsed
		agg.CorruptionErrors += x.CorruptionErrors
		agg.Compactions += x.Compactions
		agg.CompactionsAborted += x.CompactionsAborted
		agg.CompactionBytesReclaimed += x.CompactionBytesReclaimed
		agg.CompactionDurationMs += x.CompactionDurationMs
		agg.ReclaimableBytes += x.ReclaimableBytes
		agg.OnlineRelocations += x.OnlineRelocations
		agg.OnlineBytesRelocated += x.OnlineBytesRelocated
		agg.OnlinePagesRetired += x.OnlinePagesRetired
		agg.OnlinePagesRecycled += x.OnlinePagesRecycled
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
// SetAppliedIndex keeps the lock across both msyncs. That is deliberately left
// alone: it is the Raft path, it already amortises per Apply BATCH, and the durable frontier
// must not change Raft behaviour.
func (c *Cache) SetPBFrontier(seq, epoch uint64) {
	for _, s := range c.shards {
		if !s.isMmap {
			continue
		}
		// Data first, unlocked — see above.
		if err := msync(s.file, s.region); err != nil {
			slog.Warn("DURABILITY WARNING: page msync failed before pb frontier", "component", "cache", "pb_seq", seq, "pb_epoch", epoch, "err", err)
		}
		s.mu.Lock()
		setPBFrontier(s.region, seq, epoch)
		s.pbFrontierSeq.Store(seq)
		s.pbFrontierEpoch.Store(epoch)
		err := msync(s.file, s.region[:headerSize])
		s.mu.Unlock()
		if err != nil {
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
// If force is true the write is made crash-consistent: the page data region is
// msync'd to disk BEFORE the header (carrying the new applied-index) is written
// and msync'd. This ordering guarantees the persisted watermark never advances
// ahead of the page entries it logically commits, so a crash cannot leave the
// header claiming an index whose data was not yet flushed (which would make
// fsm.Apply skip-replay entries that were never durably stored). If force is
// false, both the header and page data rely on opportunistic OS flushing (or
// the Durable msyncLoop) with no ordering guarantee.
func (c *Cache) SetAppliedIndex(idx uint64, force bool) {
	for _, s := range c.shards {
		if !s.isMmap {
			continue
		}
		s.mu.Lock()
		setAppliedStamp(s.region, s.lastAppliedStampMs.Load())
		if force {
			// Flush page data first so the watermark can't outrun the entries
			// it commits, then write + flush the header. msync requires a
			// page-aligned start address, so we flush the full region (which
			// still carries the OLD applied-index here) before stamping the new
			// index into the header and flushing the header slice. The header is
			// updated only AFTER the page data is durable.
			// Durability watermark path: a swallowed msync here means the applied
			// index (and the data it certifies) may not be durable, yet we would
			// report it committed. Log both flushes' failures loudly.
			if err := msync(s.file, s.region); err != nil {
				slog.Warn("DURABILITY WARNING: page msync failed before applied-index", "component", "cache", "applied_index", idx, "err", err)
			}
			setAppliedIndex(s.region, idx)
			s.appliedIndex.Store(idx)
			if err := msync(s.file, s.region[:headerSize]); err != nil {
				slog.Warn("DURABILITY WARNING: header msync failed for applied-index", "component", "cache", "applied_index", idx, "err", err)
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
