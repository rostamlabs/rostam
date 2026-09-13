// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"time"
)

// maxIntervalMs is the largest millisecond figure that still converts to a
// time.Duration: beyond it, ms * time.Millisecond wraps. Every interval field in this
// Config is validated against it, because each one is multiplied out exactly that way
// when its ticker starts.
const maxIntervalMs = int64(math.MaxInt64) / int64(time.Millisecond)

// AtCapPolicy controls behavior when a shard is at MaxPages and all pages are full.
type AtCapPolicy uint8

const (
	// PolicyRingbufEvict overwrites the oldest entries in the oldest page.
	PolicyRingbufEvict AtCapPolicy = 0
	// PolicyRejectWrites returns ErrFull on write when the shard is at cap.
	PolicyRejectWrites AtCapPolicy = 1
)

// Config holds the cache configuration. Use DefaultConfig() and override fields.
type Config struct {
	// NumShards is the number of independent shards. Must be a power of two.
	NumShards int

	// PageSize is the byte size of each page slab. 1 MiB minimum, 1 GiB maximum.
	PageSize int

	// MaxMemoryPerShard caps total memory each shard may allocate.
	// Must be >= PageSize. The shard will allocate at most MaxMemoryPerShard/PageSize pages.
	MaxMemoryPerShard int

	// InitialPagesPerShard pre-allocates this many pages per shard on cache creation.
	// Zero means lazy allocation on first write.
	InitialPagesPerShard int

	// AtCapPolicy controls behavior when MaxPagesPerShard pages are full.
	AtCapPolicy AtCapPolicy

	// TTLSweepIntervalMs controls how often each shard's background sweeper runs.
	// Zero disables the sweeper (lazy expiration on read still works).
	TTLSweepIntervalMs int

	// DataDir is the directory holding per-shard pages.dat files. When
	// empty, the cache runs in heap-only mode (heap-only behavior; no
	// persistence). Requires a platform that can map files (Linux, Windows);
	// anywhere else Validate rejects DataDir != "".
	DataDir string

	// Mlock locks the mmap'd region into memory. Requires
	// RLIMIT_MEMLOCK >= total mmap size. Failure logs a warning and
	// continues without lock. Requires DataDir.
	Mlock bool

	// Durable runs msync on commit boundaries (cf. MsyncIntervalMs).
	// Default false (opportunistic OS flushing). Requires DataDir.
	Durable bool

	// MsyncIntervalMs is the maximum delay between msync flushes when
	// Durable is true. Default 100. Ignored when Durable is false.
	MsyncIntervalMs int

	// Replicated marks this cache as backing a cluster-replicated shard (#4 Phase
	// B). When set, WALL-CLOCK physical removal of logically-expired keys is
	// suppressed so the committed key set stays LOGICALLY byte-identical (identical
	// key/value/exp set; physical snapshot bytes may still differ via the Iterate
	// wall-clock filter) across replicas that tick at slightly different wall times:
	//
	//   - the WALL-CLOCK sweep is not used; instead a LOGICAL-CLOCK sweep runs
	//     (B3b, see below), reclaiming against lastAppliedStampMs — identical on
	//     every replica — never a node-local wall instant;
	//   - a client-read lazy expiry still FILTERS the key (returns a miss, correct
	//     staleness) but does NOT drop the index slot (no nondeterministic mutation);
	//     physical reclamation is solely the logical sweeper's job;
	//   - warm-restart rebuildIndexFromPages keeps ALL entries (absolute exp intact)
	//     rather than dropping by wall clock.
	//
	// The APPLY path (GetAt, evaluated against the leader-stamped clock) still
	// tombstones expired keys — that is a committed-state decision identical on
	// every replica.
	//
	// B3b — LOGICAL-CLOCK RECLAMATION. Each shard tracks lastAppliedStampMs, the
	// running MAX of leader apply-stamps (identical and monotonic on every replica).
	// A background sweep reclaims iff exp <= lastAppliedStampMs: it tombstones expired
	// index slots and, on HEAP shards, RETIRES whole expired pages, physically freeing
	// capacity. Without it, expired ghost pages (index slot AND page bytes) accumulated
	// to MaxPagesPerShard — and because a replicated shard is ALSO forced to
	// PolicyRejectWrites (B2) — the next committed write hit cache.ErrFull, Phase A
	// failed closed, and the shard entered a DETERMINISTIC crash-loop under sustained
	// TTL-heavy load. Reclamation is active ONLY once apply-stamping is enabled
	// (Config.EnableApplyStamp; rollout phase 2): with stamping off lastAppliedStampMs
	// stays 0 and the logical sweep is a no-op, so size replicated shards with TTL-
	// churn headroom until stamping is on.
	//
	// HEAP vs MMAP — the cliff is NOT closed to the same degree on both:
	//   - HEAP (pure in-memory) replicated shards: the cliff is CLOSED — pages are
	//     retired by frozen-swap (see cache/shard.go reclaimExpiredHeapPages), so page
	//     bytes are reclaimed deterministically and a would-be-ErrFull write succeeds.
	//   - MMAP (persistent, DataDir set) replicated shards: RECOVERABLE BY RESTART,
	//     not closed continuously. mmap pages wrap the fixed file region and cannot be
	//     swapped for a fresh object; reclaiming them in place would race the lock-free
	//     reject-writes reader. So while the process RUNS the logical sweep reclaims
	//     INDEX SLOTS deterministically and page BYTES not at all, and ghost bytes climb
	//     under TTL churn → ErrFull → Phase A halt. COLD COMPACTION AT SHARD OPEN
	//     (cache/compact.go) reclaims them: at newShard — the one moment no reader
	//     exists — a shard above the occupancy mark rewrites its pages file with only
	//     the live entries (superseded/index-dead always; TTL-expired judged against the
	//     PERSISTED LOGICAL CLOCK when replicated, never the wall clock) and atomically
	//     renames it into place. So a restart drains the ghosts; between restarts, size
	//     persistent replicated shards with TTL-churn headroom and watch the occupancy
	//     alert. NOTE: snapshotting does NOT reclaim the local file — serializeSnapshot
	//     compacts the snapshot BLOB (its Iterate filter skips expired/tombstoned), but
	//     an in-place restore writes back through Del + PutAbs, and on mmap Del only
	//     tombstones an index slot, so a restore APPENDS on top of the existing ghosts.
	//
	// Vector per-key TTL determinism is a separate committed-expiry site still open
	// (shard/apply_class.go vector-audit TODO).
	//
	// Single-node / Direct caches leave this false and keep the wall-clock sweeper +
	// wall-clock lazy removal — byte-identical to pre-Phase-B behavior.
	Replicated bool

	// DisableColdCompaction turns OFF the cold compaction an mmap shard performs
	// at open (cache/compact.go): the live-only rewrite of pages.dat that reclaims
	// ghost page BYTES a persistent shard cannot reclaim while it runs. Default
	// false = compaction ENABLED, which is the behavior every persistent shard
	// needs — without it ghost bytes climb monotonically under TTL churn until the
	// shard hits ErrFull and the Phase-A fail-closed halt, with no remedy short of
	// reformatting the DataDir.
	//
	// This exists as an OPERATIONAL ESCAPE HATCH, not a tuning knob: compaction
	// rewrites the durable file at every open, so if it ever misbehaves in the
	// field an operator needs a way to stop it that is not a binary rollback (and
	// a rollback is itself lossy — compaction upgrades the header to v3 and an
	// older build's exact-version gate rotates a v3 file aside). Setting it costs
	// only reclamation: the pages file is left exactly as it was found, and every
	// other recovery step (rebuild, stale-temp cleanup) is unchanged.
	//
	// Ignored for heap shards, which have no pages file and reclaim online.
	DisableColdCompaction bool

	// OnlineCompaction opts a REPLICATED MMAP REJECT-WRITES shard into ONLINE
	// relocating compaction with quarantine-then-reset recycle (cache/compact_online.go):
	// while the process runs, the TTL sweeper relocates the live entries out of
	// fragmented pages, RETIRES the emptied source extents, and — once the alias-drain
	// quarantine has elapsed — RECYCLES a retired extent (resets head/tail to 0 with a
	// bumped generation and hands it back to the write path), so a shard that hit ErrFull
	// on accumulated dead versions recovers write capacity WHILE running instead of only
	// at restart (cold compaction). Default false. Threaded from rostam.ServerConfig.
	// EnableOnlineCompaction.
	//
	// It is a no-op on every other shard (heap, single-node, ringbuf) — the online
	// compactor is gated on isMmap && Replicated && AtCapPolicy==PolicyRejectWrites,
	// the exact mode a cluster-replication shard is forced into (see shard/store.go).
	//
	// WHY OPT-IN, and why it is OFF by default. Recycle overwrites retired mmap page
	// bytes after AliasQuarantine. It is memory-safe ONLY when every read of the shard is
	// released within that window — i.e. all reads flow through the server transport,
	// whose WriteTimeout bounds the zero-copy response alias (AliasQuarantine =
	// 2*WriteTimeout, enforced fail-closed in shard.New). Do NOT enable it if any
	// in-process caller holds a Store.Get / Node.Call result (a raw cache alias) past
	// AliasQuarantine; those readers must copy the value out promptly. So the feature is
	// an explicit operator opt-in, not a default, even though recycle is fully
	// implemented. The reclaimable-bytes accounting and the trigger evaluation (Stage 0)
	// run regardless — reclaimable is always visible in Stats — this flag only gates the
	// relocate+recycle ACTION.
	OnlineCompaction bool

	// RelocatingEviction opts a RINGBUF shard into RELOCATING eviction
	// (cache/relocate_evict.go): before the rotation cursor drains a page, the live
	// records on it are COPIED FORWARD into the space the previous eviction freed and
	// their index slots repointed, so a record that is still the live copy for its key
	// is no longer dropped merely because it shares a page with superseded versions of
	// OTHER keys. Default false. Eviction under PolicyRingbufEvict is positional — it
	// picks the next non-empty page and drains it in full — so without this the page's
	// dead versions and its live records go together.
	//
	// It applies to BOTH ringbuf storage modes: heap shards, which free a page by
	// retiring it, and single-node mmap shards, which drain the fixed region in place.
	// On mmap a relocated copy is appended through the ordinary write path, so it
	// carries a higher write sequence than the original it leaves framed on the source
	// page and warm restart resolves the pair to the relocated copy. It is a no-op
	// under PolicyRejectWrites (nothing is ever evicted), which is what replication
	// forces — a replicated shard reclaims through the online compactor instead
	// (Config.OnlineCompaction).
	//
	// WHAT IT COSTS. Relocation runs on the WRITE PATH, inside the Put that triggered
	// the eviction, and copies entry bytes. It is bounded so that cost stays a
	// fraction of the eviction it rides on: it may spend only the room left in the
	// freed page AFTER the triggering write's own requirement, and at most
	// PageSize/relocateMaxBytesPerEvictionDivisor bytes per eviction. It never
	// allocates a page, never triggers another eviction, and never fails a write — a
	// record that does not fit the budget is simply left to be dropped as it is today.
	// Stats.EvictionRelocations / EvictionBytesRelocated report what it moved.
	//
	// WHO ACTUALLY PAYS IT. On a HEAP ringbuf shard with the sweeper running
	// (TTLSweepIntervalMs > 0) most of that copying moves OFF the write path: the
	// sweeper keeps a small reserve of free pages by evacuating and retiring rotation
	// victims ahead of time (cache/relocate_reserve.go), so a write at capacity finds
	// room in an already-free page and never evicts. The write-path pass above stays as
	// the fallback for when a burst outruns the sweeper or the shard is too dense to
	// evacuate. Stats.ReserveRelocations / ReserveBytesRelocated / ReservePagesFreed
	// report the background half, and reading them next to the Eviction* pair says how
	// much of the cost writes are still carrying. The reserve costs a page or two of
	// capacity held empty and MORE total copying than the write-path pass alone (it
	// moves records on a clock, so it copies some that would have been superseded
	// before their page came round); it buys a steady-state write paying none of it.
	// Mmap ringbuf shards keep the write-path pass alone — see the HEAP RINGBUF ONLY
	// note in cache/relocate_reserve.go for why.
	RelocatingEviction bool

	// RelocateReserveIntervalMs is how often each shard tops up its free-page reserve
	// (cache/relocate_reserve.go). Zero — the value a zero Config carries — runs no
	// reserve ticker at all, which leaves RelocatingEviction as exactly the write-path
	// pass and nothing else. DefaultConfig sets it; see below for the value and why.
	//
	// It is a CADENCE, not a second on/off switch: RelocatingEviction remains the gate,
	// and this field does nothing at all while that is false.
	//
	// WHY IT IS NOT TTLSweepIntervalMs, which is the ticker the reserve first rode. Those
	// are two different jobs with two different right answers. A tick of the TTL sweeper
	// runs sweepIndex, which takes the shard write lock for sweepBatchSize slots at a
	// time and decodes every live entry it passes; a tick of this one calls only
	// topUpFreeReserve, which walks at most the rotation victim and usually returns at
	// once. The reserve earns nothing at a cadence measured in seconds — it cannot get
	// ahead of the write stream — but running sweepIndex at the cadence the reserve wants
	// costs far more than the reserve saves. One interval forces one of the two onto the
	// other's setting, so they are separate and the TTL sweeper keeps its own.
	//
	// THE COST IS PER SHARD, which is the thing to hold on to when changing it. Every
	// shard runs its own ticker, so at a fixed interval the work the cache does per unit
	// time scales with NumShards while the write stream does not: an interval that is
	// free on a handful of shards is not the same setting on hundreds. Tick work is
	// bounded (see topUpFreeReserve) but it is not zero, and a shard below its page cap
	// still costs a lock acquisition and a length check per tick to discover it has
	// nothing to do.
	//
	// THE DEFAULT is the value the A/B measured as earning more than it costs
	// (cache/relocate_sharded_bench_test.go). Read that benchmark's doc before changing
	// it: the figure is the outcome of a measurement, not a round number, and the two
	// directions it is wrong in are not symmetric.
	RelocateReserveIntervalMs int

	// AliasQuarantine is how long online relocating compaction must let a RETIRED
	// mmap page sit — bytes mapped and immutable — before it may RECYCLE that page
	// (reset head/tail to 0 and hand its extent back to the write path). It is the
	// drain fence for the lock-free zero-copy read path: a reject-writes/mmap Get
	// returns a []byte aliasing the page bytes, and that alias escapes to a network
	// response writer. Recycling overwrites the extent, so it is safe only once no
	// reader can still hold an alias into the OLD content.
	//
	// The safe value is the maximum wall-clock lifetime of such an alias. The only
	// consumer that holds the RAW alias across a blocking network flush is the TCP
	// server's response writer, which arms a per-write deadline (server.Config.
	// WriteTimeout, default 30s) — every other consumer (HTTP handler, WASM op,
	// snapshot/replication read, read-modify-write op) copies the value synchronously
	// before any blocking wait. So the alias-hold bound is WriteTimeout plus a
	// CPU-only pipeline drain, and this should be set to a conservative multiple of
	// that (shard/store.go threads in 2*WriteTimeout). 0 ⇒ a conservative built-in
	// default (defaultAliasQuarantine). Only consulted on an OnlineCompaction-enabled
	// eligible shard; ignored everywhere else.
	AliasQuarantine time.Duration

	// ServerWriteTimeout is the EFFECTIVE server.Config.WriteTimeout threaded down
	// from the transport layer (rostam.ServerConfig → EmbeddedConfig → here). It is
	// the single source of truth for the alias-hold bound: shard.New derives
	// AliasQuarantine = 2*ServerWriteTimeout from it and FAILS CLOSED if an
	// explicitly-set AliasQuarantine is smaller, so the drain fence can never silently
	// fall below the real write deadline it must outlast (the hazard when the two were
	// independent constants). 0 ⇒ shard.New falls back to its built-in default
	// (defaultServerWriteTimeout), matching the server's own WriteTimeout default. Only
	// consulted when building a replicated (online-compaction-eligible) shard; ignored
	// everywhere else.
	ServerWriteTimeout time.Duration

	// InPlaceSameSizeUpdate lets a write OVERWRITE the entry already stored for its
	// key instead of appending a new copy after it, when the two are framed
	// identically — same key, same value LENGTH. Default false.
	//
	// WHAT IT BUYS. Every update appends today, so the previous copy stays framed
	// and dead in the pages. Under PolicyRingbufEvict those dead versions are what
	// carry a shard to capacity and start it evicting LIVE keys, so a workload that
	// rewrites the same keys at a roughly constant record size spends its budget on
	// garbage it never had to create. An entry is [header][key][value] with a
	// fixed-width header, so for the same key and an identical value length the new
	// bytes fit exactly where the old ones are — and because the index keeps
	// pointing at the same page, offset and generation, the write also skips the
	// slot upsert, the page allocation, and any eviction that would have followed.
	//
	// WHERE IT APPLIES. Only on a HEAP-backed PolicyRingbufEvict shard, and only
	// for a write whose key is index-current at a copy of exactly the new size;
	// every other write takes the append path unchanged. It is a no-op — not an
	// error — on a shard that does not qualify:
	//
	//   - PolicyRejectWrites shards hand out ZERO-COPY ALIASES into page bytes, and
	//     an alias may outlive the read that produced it (it escapes to a network
	//     response writer). Overwriting live bytes under it would change a value a
	//     caller is still holding.
	//   - MMAP shards are the DURABLE copy. Overwriting destroys the old version,
	//     and the loss is not limited to the key being written. For a tear in entry
	//     data, with the page's persisted framing intact: an append writes only at
	//     the page TAIL, while an in-place write tears mid-page, and recovery
	//     answers a torn entry by truncating the page there and abandoning the rest
	//     of it, so keys durable long before the torn write are lost with it. (A
	//     tear in the page framing resets the whole page for either shape.) Heap
	//     pages are not persisted, so the concern does not arise.
	//
	// WHAT IT ALSO COSTS: WRITE RECENCY, and this is a change in EVICTION
	// SEMANTICS, not only in locking. Eviction here reclaims whole pages in
	// rotation, so which records survive is decided by which PAGE they sit on. An
	// appending rewrite moves its key to the newest page, so a key touched often
	// drifts ahead of the rotation and outlives one touched rarely — recency the
	// ring buffer gets for free, without tracking anything. An in-place rewrite
	// keeps the key on whatever page it was first written to, so the rotation
	// reaches it on schedule however hot it is.
	//
	// It only bites on a shard whose live set EXCEEDS its budget, because that is
	// the only regime where in-place cannot simply stop evicting. There it is a
	// genuine trade rather than a win, and BenchmarkInPlaceWriteRecency exists to
	// put numbers on both halves: against the append path it retained about a
	// quarter more keys in total and evicted about a fifth less often, while
	// holding roughly ten percent less of the frequently-rewritten subset.
	//
	// Relocating eviction does NOT remedy it — it makes the recency loss somewhat
	// worse, not better, while raising the eviction rate. Carrying a record off a
	// drained page preserves the record, but it does not restore the ordering that
	// decides which page is drained next, which is the thing recency was.
	//
	// So: if a shard is sized for its working set, this is close to pure gain — it
	// is the garbage that drove eviction, and removing it removes the evictions.
	// If a shard is deliberately run over capacity as a ring buffer AND leans on
	// rewrite frequency to decide what stays, leave it off.
	//
	// WHAT IT COSTS. Heap ringbuf reads are lock-free today only because pages are
	// append-only and frozen: eviction retires a page by swapping in a fresh object,
	// never by rewriting live bytes, so the bytes behind a hit are immutable for the
	// read's lifetime. Writing in place breaks exactly that. So enabling this makes
	// heap ringbuf reads take the shard READ LOCK for the probe and value copy — the
	// same path mmap ringbuf has always used (see needsReadLockForGet). That is the
	// trade: writes stop creating garbage, reads stop being lock-free.
	InPlaceSameSizeUpdate bool

	// InPlaceSeqlockReads makes a shard with InPlaceSameSizeUpdate keep its reads
	// LOCK-FREE, validating each one against a per-stripe version counter instead
	// of taking the shard read lock. Default false. Ignored unless in-place
	// updates are enabled — it protects against exactly one hazard, a writer
	// rewriting an entry's bytes underneath a reader, which only in-place updates
	// create.
	//
	// It exists because the read lock is what in-place updates otherwise cost, and
	// the cost is large: an RWMutex read acquisition is a read-modify-write on one
	// cache line every reader shares, so read throughput stops scaling with cores.
	//
	// IT IS OFF BY DEFAULT AND SHOULD STAY OFF UNLESS THE READ:WRITE RATIO PAYS FOR
	// IT, for two reasons that are not tuning preferences:
	//
	//   - A seqlock's payload access is a DATA RACE by construction — validating
	//     after the fact is what it does instead of preventing the access — so the
	//     race detector reports it, and the protocol rests on how Go compiles its
	//     atomics rather than on a guarantee the memory model extends to racing
	//     plain accesses. cache/seqlock.go sets out the ordering argument and its
	//     limits in full. Accessing the payload atomically would close both gaps and
	//     costs more than the read lock it replaces (measured there).
	//   - It is only a win when reads outnumber writes by enough. The version bump
	//     is two atomic read-modify-writes on the WRITE path; below roughly six
	//     reads per write the read lock is the cheaper of the two.
	InPlaceSeqlockReads bool

	// NowFn overrides the WALL-CLOCK source for the non-apply expiry sites (client
	// read filter, sweeper, warm-restart rebuild, Iterate). nil ⇒ the real clock
	// (nowMs / time.Now) — the production default, byte-identical to pre-B1
	// behavior. It is a test seam for injecting per-replica clock skew and for
	// pinning a FIXED clock when computing a canonical cross-replica state
	// fingerprint. It does NOT affect the apply path, which always uses the
	// explicit leader-stamped nowMs passed to PutAt/GetAt.
	NowFn func() uint64
}

// DefaultConfig returns a config suitable for general use.
func DefaultConfig() Config {
	return Config{
		NumShards:            256,
		PageSize:             16 << 20,  // 16 MiB
		MaxMemoryPerShard:    256 << 20, // 256 MiB
		InitialPagesPerShard: 0,
		AtCapPolicy:          PolicyRingbufEvict,
		TTLSweepIntervalMs:   1000,
		MsyncIntervalMs:      100,
		// Inert unless RelocatingEviction is set, which is off by default.
		RelocateReserveIntervalMs: defaultRelocateReserveIntervalMs,
	}
}

// MaxPagesPerShard returns the page cap derived from MaxMemoryPerShard / PageSize.
func (c Config) MaxPagesPerShard() int {
	return c.MaxMemoryPerShard / c.PageSize
}

// Validate returns an error if the configuration is invalid.
func (c Config) Validate() error {
	if c.NumShards <= 0 {
		return errors.New("config: NumShards must be > 0")
	}
	if c.NumShards&(c.NumShards-1) != 0 {
		return fmt.Errorf("config: NumShards=%d must be a power of two", c.NumShards)
	}
	if c.PageSize < 1<<20 {
		return fmt.Errorf("config: PageSize=%d must be >= 1 MiB", c.PageSize)
	}
	if c.PageSize > 1<<30 {
		return fmt.Errorf("config: PageSize=%d must be <= 1 GiB", c.PageSize)
	}
	if c.MaxMemoryPerShard < c.PageSize {
		return fmt.Errorf("config: MaxMemoryPerShard=%d must be >= PageSize=%d",
			c.MaxMemoryPerShard, c.PageSize)
	}
	if c.InitialPagesPerShard < 0 {
		return errors.New("config: InitialPagesPerShard must be >= 0")
	}
	if c.InitialPagesPerShard > c.MaxPagesPerShard() {
		return fmt.Errorf("config: InitialPagesPerShard=%d exceeds MaxPagesPerShard=%d",
			c.InitialPagesPerShard, c.MaxPagesPerShard())
	}
	// slabRef packs the per-shard page index into 16 bits (see slabref.go), so
	// the page count must fit a uint16. Enforce the packing invariant here — the
	// single point every cache config passes through — rather than letting an
	// extreme-but-valid config (e.g. PageSize=1 MiB, MaxMemoryPerShard>64 GiB)
	// silently wrap makeSlabRef's pageIdx and misdirect reads.
	if c.MaxPagesPerShard() > maxPagesPerShardCap {
		return fmt.Errorf("config: MaxPagesPerShard=%d exceeds %d (MaxMemoryPerShard/PageSize must fit a uint16 page index)",
			c.MaxPagesPerShard(), maxPagesPerShardCap)
	}
	switch c.AtCapPolicy {
	case PolicyRingbufEvict, PolicyRejectWrites:
	default:
		return fmt.Errorf("config: AtCapPolicy=%d invalid", c.AtCapPolicy)
	}
	// Both intervals become time.Duration(ms) * time.Millisecond when their ticker
	// starts, which overflows silently past maxIntervalMs and hands time.NewTicker a
	// negative period — a panic on a background goroutine, at start-up, from a value
	// that passed validation. Reject it here instead.
	if c.TTLSweepIntervalMs < 0 {
		return errors.New("config: TTLSweepIntervalMs must be >= 0")
	}
	if int64(c.TTLSweepIntervalMs) > maxIntervalMs {
		return fmt.Errorf("config: TTLSweepIntervalMs=%d overflows a duration; must be <= %d",
			c.TTLSweepIntervalMs, maxIntervalMs)
	}
	if c.RelocateReserveIntervalMs < 0 {
		return errors.New("config: RelocateReserveIntervalMs must be >= 0")
	}
	if int64(c.RelocateReserveIntervalMs) > maxIntervalMs {
		return fmt.Errorf("config: RelocateReserveIntervalMs=%d overflows a duration; must be <= %d",
			c.RelocateReserveIntervalMs, maxIntervalMs)
	}
	if c.DataDir != "" && !mmapSupported {
		return fmt.Errorf("cache.Config: DataDir set but mmap not supported on %s; use DataDir=\"\"", runtime.GOOS)
	}
	if c.Durable && c.DataDir == "" {
		return errors.New("cache.Config: Durable requires DataDir")
	}
	if c.Mlock && c.DataDir == "" {
		return errors.New("cache.Config: Mlock requires DataDir")
	}
	if int64(c.MsyncIntervalMs) > maxIntervalMs {
		return fmt.Errorf("config: MsyncIntervalMs=%d overflows a duration; must be <= %d",
			c.MsyncIntervalMs, maxIntervalMs)
	}
	if c.MsyncIntervalMs < 1 {
		return errors.New("cache.Config: MsyncIntervalMs must be >= 1")
	}
	// The online-compaction durations are safety knobs. A NEGATIVE value is always a
	// typo — never a valid request — and must be rejected here rather than allowed to
	// silently fall back to a safety default (AliasQuarantine) or invert a deadline
	// (ServerWriteTimeout). 0 stays valid: it selects the built-in default downstream.
	if c.AliasQuarantine < 0 {
		return fmt.Errorf("cache.Config: AliasQuarantine=%s must be >= 0", c.AliasQuarantine)
	}
	if c.ServerWriteTimeout < 0 {
		return fmt.Errorf("cache.Config: ServerWriteTimeout=%s must be >= 0", c.ServerWriteTimeout)
	}
	return nil
}
