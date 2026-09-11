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
// report current capacity/occupancy, and ReclaimableBytes reports current ghost-byte
// pressure (which compaction actively drives back down).
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
	EvictionsLive    uint64
	Rejects          uint64 // refused due to PolicyRejectWrites
	PagesAllocated   uint64
	BytesAllocated   uint64
	BytesUsed        uint64
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
