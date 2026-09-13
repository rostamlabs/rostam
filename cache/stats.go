// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"io"
)

// Stats is a snapshot of cache counters.
// Most fields are cumulative counters accumulated since cache creation (Gets, Hits,
// Puts, Evictions, the Compaction* totals, …); sample and diff them for rates. The
// exceptions are point-in-time GAUGES that RISE and FALL and must NOT be diffed as
// counters (a decrease is not wraparound): PagesAllocated, BytesAllocated and BytesUsed
// report current capacity/occupancy, ReclaimableBytes reports current ghost-byte
// pressure (which compaction actively drives back down), and Entries/Tombstones report
// the index's current occupancy (both fall on delete, and Tombstones falls again when
// a rehash reclaims the slots).
type Stats struct {
	Gets        uint64
	Hits        uint64
	Misses      uint64
	Puts        uint64
	Dels        uint64
	Expirations uint64 // expired by sweeper or lazy-on-read
	Evictions   uint64 // entries displaced by ringbuf eviction, live or not
	// EvictionsLive counts only those evictions that displaced the entry the
	// index still pointed at: a record lost to CAPACITY rather than to its TTL.
	// Evictions also covers entries a newer Put had already superseded, so
	// EvictionsLive is the one to watch to decide whether a cache is sized for
	// its working set. EvictionsLive > 0 with Expirations low means the budget,
	// not the TTL, is deciding how long entries survive.
	EvictionsLive  uint64
	Rejects        uint64 // refused due to PolicyRejectWrites
	PagesAllocated uint64
	BytesAllocated uint64
	BytesUsed      uint64
	// Entries is the number of keys the index currently holds, and Tombstones
	// the deleted-but-not-yet-reclaimed slots beside them. Entries is the only
	// way to answer "how many keys fit in this budget": every other gauge here
	// is bytes. It cannot be derived from the counters either -- misses tracks
	// distinct keys only until the first eviction, after which an evicted key
	// that returns misses again.
	//
	// BytesUsed/Entries is occupancy per indexed key, NOT the size of a record.
	// BytesUsed counts every byte no live key owns: superseded copies, expired
	// entries not yet swept, and the ghost bytes behind deleted slots. So
	// overwriting, expiry and deletion each raise the ratio while the records
	// are unchanged. It is the right figure to divide a memory budget by, since
	// all of that occupies the budget, but a dashboard must not label it an
	// average record size.
	Entries          uint64
	Tombstones       uint64
	CorruptionErrors uint64 // CRC mismatches on read

	// Cold compaction at shard open (mmap only; cache/compact.go). These are the
	// operator's view of whether restarts are actually reclaiming the ghost page
	// bytes a persistent shard cannot reclaim while running:
	//   - Compactions: pages files rewritten live-only and swapped in;
	//   - CompactionsAborted: rewrites decided against or abandoned (no space to
	//     stage, pack overflow, failed rename) — the original file was kept;
	//   - CompactionBytesReclaimed: page bytes dropped by those rewrites;
	//   - CompactionDurationMs: total time spent in them (they run at open, so
	//     this is startup latency).
	Compactions              uint64
	CompactionsAborted       uint64
	CompactionBytesReclaimed uint64
	CompactionDurationMs     uint64

	// Online relocating compaction (mmap replicated reject-writes shards only;
	// cache/compact_online.go). The operator's live view of the ghost-byte pressure
	// a running persistent replicated shard is under, and of the online compactor's
	// activity when it is enabled (Config.OnlineCompaction):
	//   - ReclaimableBytes: page bytes that are NOT index-current-and-live at the
	//     shard's LOGICAL clock (superseded / deleted / logically-expired) — the space
	//     a compaction could reclaim. Computed on demand for eligible shards; 0 for
	//     every other shard. This is the Stage 0 signal and needs no compactor enabled.
	//     A point-in-time GAUGE (see the type comment): it falls when compaction
	//     reclaims, so unlike every other field it is not a monotonic counter.
	//   - OnlineRelocations: live entries relocated out of fragmented pages;
	//   - OnlineBytesRelocated: their on-disk byte total;
	//   - OnlinePagesRetired: source pages fully evacuated and marked retired (their
	//     extents stay mapped + immutable through the alias-drain quarantine);
	//   - OnlinePagesRecycled: retired pages whose quarantine elapsed and were reset
	//     (extent handed back to the write path) — the count of pages that actually
	//     recovered write capacity.
	ReclaimableBytes     uint64
	OnlineRelocations    uint64
	OnlineBytesRelocated uint64
	OnlinePagesRetired   uint64
	OnlinePagesRecycled  uint64

	// Relocating eviction (ringbuf shards with Config.RelocatingEviction, heap and
	// single-node mmap alike; cache/relocate_evict.go). A DIFFERENT mechanism from
	// the online compactor above, on the other at-cap policy:
	//   - EvictionRelocations: live records copied forward out of a page about to be
	//     drained, instead of being dropped with the dead versions sharing it. Read
	//     it next to EvictionsLive: relocation is what moves losses out of that
	//     counter, so the two together say how much of the eviction pressure on this
	//     shard is landing on records that were still live.
	//   - EvictionBytesRelocated: their framed byte total — the write-path copy the
	//     feature is charging for those saves.
	EvictionRelocations    uint64
	EvictionBytesRelocated uint64

	// The BACKGROUND half of the same feature (cache/relocate_reserve.go): the
	// per-shard sweeper keeping a small reserve of free pages so a write at capacity
	// finds room instead of evicting and relocating inline. Counted separately from
	// the write-path pair above precisely so the split is visible — on a shard whose
	// sweeper is keeping up these climb while EvictionRelocations stays flat, and the
	// ratio between them is how much of the relocation cost is still being charged to
	// writes.
	//   - ReserveRelocations: live records the reserve pass copied forward;
	//   - ReserveBytesRelocated: their framed byte total — background copying, not
	//     write-path copying;
	//   - ReservePagesFreed: pages it fully evacuated and retired, i.e. the free pages
	//     it actually produced. Zero while it is doing work but never finishing a page
	//     means the shard is too dense to evacuate and the write path is carrying it.
	ReserveRelocations    uint64
	ReserveBytesRelocated uint64
	ReservePagesFreed     uint64
}

// HitRate returns Hits / Gets, or 0 when Gets == 0.
func (s Stats) HitRate() float64 {
	if s.Gets == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Gets)
}

// Add accumulates o into s field by field. Cache.Stats uses it to fold its
// shards together, and the cluster node-local scrape uses it to fold the
// per-shard caches it hosts, so the field list lives in exactly one place.
func (s *Stats) Add(o Stats) {
	s.Gets += o.Gets
	s.Hits += o.Hits
	s.Misses += o.Misses
	s.Puts += o.Puts
	s.Dels += o.Dels
	s.Expirations += o.Expirations
	s.Evictions += o.Evictions
	s.EvictionsLive += o.EvictionsLive
	s.Rejects += o.Rejects
	s.PagesAllocated += o.PagesAllocated
	s.BytesAllocated += o.BytesAllocated
	s.BytesUsed += o.BytesUsed
	s.Entries += o.Entries
	s.Tombstones += o.Tombstones
	s.CorruptionErrors += o.CorruptionErrors
	s.Compactions += o.Compactions
	s.CompactionsAborted += o.CompactionsAborted
	s.CompactionBytesReclaimed += o.CompactionBytesReclaimed
	s.CompactionDurationMs += o.CompactionDurationMs
	s.ReclaimableBytes += o.ReclaimableBytes
	s.OnlineRelocations += o.OnlineRelocations
	s.OnlineBytesRelocated += o.OnlineBytesRelocated
	s.OnlinePagesRetired += o.OnlinePagesRetired
	s.OnlinePagesRecycled += o.OnlinePagesRecycled
	s.EvictionRelocations += o.EvictionRelocations
	s.EvictionBytesRelocated += o.EvictionBytesRelocated
	s.ReserveRelocations += o.ReserveRelocations
	s.ReserveBytesRelocated += o.ReserveBytesRelocated
	s.ReservePagesFreed += o.ReservePagesFreed
}

// WritePrometheus renders s in the Prometheus text exposition format. Counters
// are cumulative since node start; sample and diff them for rates.
//
// rostam_kv_evictions_live_total is the one worth alerting on: it counts
// records displaced because the cache was full rather than because their TTL
// elapsed. A cache sized for its working set holds that near zero and retires
// entries through rostam_kv_expirations_total instead.
func (s Stats) WritePrometheus(w io.Writer) error {
	counters := []struct {
		name string
		help string
		val  uint64
	}{
		{"rostam_kv_gets_total", "KV reads served", s.Gets},
		{"rostam_kv_hits_total", "KV reads that found a live entry", s.Hits},
		{"rostam_kv_misses_total", "KV reads that found nothing", s.Misses},
		{"rostam_kv_puts_total", "KV write ATTEMPTS - incremented before a capacity rejection can return ErrFull, so it includes writes that failed", s.Puts},
		{"rostam_kv_dels_total", "KV deletes applied", s.Dels},
		{"rostam_kv_expirations_total", "entries retired because their TTL elapsed", s.Expirations},
		{"rostam_kv_evictions_total", "entries displaced by ringbuf eviction, live or already superseded", s.Evictions},
		{"rostam_kv_evictions_live_total", "entries displaced by ringbuf eviction that were still the live record for their key - lost to capacity, not TTL", s.EvictionsLive},
		{"rostam_kv_eviction_relocations_total", "live records copied forward by relocating eviction instead of being dropped with the dead versions sharing their page - the saves that did NOT become evictions_live", s.EvictionRelocations},
		{"rostam_kv_eviction_bytes_relocated_total", "bytes copied by those relocations - the write-path cost of the saves", s.EvictionBytesRelocated},
		{"rostam_kv_reserve_relocations_total", "live records copied forward by the background free-page reserve instead of by a write - read against eviction_relocations to see how much of the copying the write path is still paying for", s.ReserveRelocations},
		{"rostam_kv_reserve_bytes_relocated_total", "bytes copied by the background reserve - background work, not write-path cost", s.ReserveBytesRelocated},
		{"rostam_kv_reserve_pages_freed_total", "pages the background reserve fully evacuated and retired - the free pages it produced for the write path", s.ReservePagesFreed},
		{"rostam_kv_rejects_total", "writes refused under PolicyRejectWrites", s.Rejects},
		{"rostam_kv_corruption_errors_total", "CRC mismatches seen on read", s.CorruptionErrors},
		{"rostam_kv_compactions_total", "page files rewritten live-only at shard open (mmap)", s.Compactions},
		{"rostam_kv_compactions_aborted_total", "compactions decided against or abandoned; the original file was kept", s.CompactionsAborted},
		{"rostam_kv_compaction_bytes_reclaimed_total", "page bytes dropped by those rewrites", s.CompactionBytesReclaimed},
		{"rostam_kv_compaction_duration_ms_total", "milliseconds spent compacting at open - this is startup latency", s.CompactionDurationMs},
		{"rostam_kv_online_relocations_total", "live entries relocated by online compaction", s.OnlineRelocations},
		{"rostam_kv_online_bytes_relocated_total", "bytes moved by those relocations", s.OnlineBytesRelocated},
		{"rostam_kv_online_pages_retired_total", "source pages fully evacuated and marked retired", s.OnlinePagesRetired},
		{"rostam_kv_online_pages_recycled_total", "retired pages whose quarantine elapsed and were reset into writable space", s.OnlinePagesRecycled},
	}
	gauges := []struct {
		name string
		help string
		val  uint64
	}{
		{"rostam_kv_pages_allocated", "cache pages currently allocated", s.PagesAllocated},
		{"rostam_kv_bytes_allocated", "bytes backing those pages", s.BytesAllocated},
		{"rostam_kv_bytes_used", "bytes occupied by entries in resident pages, INCLUDING superseded and expired entries not yet reclaimed - occupancy, not live data", s.BytesUsed},
		{"rostam_kv_entries", "keys currently held in the index - the denominator the eviction counters lack, and what to divide a memory budget by; bytes_used/entries is occupancy per key INCLUDING superseded, expired and deleted-but-unreclaimed bytes, not the size of one record", s.Entries},
		{"rostam_kv_tombstones", "index slots holding a deleted key that has not been reclaimed; a large share of entries means the index wants compacting", s.Tombstones},
		{"rostam_kv_reclaimable_bytes", "page bytes held by entries no longer reachable", s.ReclaimableBytes},
	}

	for _, c := range counters {
		if _, err := fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %d\n",
			c.name, c.help, c.name, c.name, c.val); err != nil {
			return err
		}
	}
	for _, g := range gauges {
		if _, err := fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %d\n",
			g.name, g.help, g.name, g.name, g.val); err != nil {
			return err
		}
	}
	return nil
}
