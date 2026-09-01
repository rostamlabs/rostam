// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"errors"
	"fmt"
	"runtime"
	"time"
)

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
	if c.TTLSweepIntervalMs < 0 {
		return errors.New("config: TTLSweepIntervalMs must be >= 0")
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
