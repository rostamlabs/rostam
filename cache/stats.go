// SPDX-License-Identifier: Apache-2.0

package cache

// Stats is a snapshot of cache counters.
// Most fields are cumulative counters accumulated since cache creation (Gets, Hits,
// Puts, Evictions, the Compaction* totals, …); sample and diff them for rates. The
// exceptions are point-in-time GAUGES that RISE and FALL and must NOT be diffed as
// counters (a decrease is not wraparound): PagesAllocated, BytesAllocated and BytesUsed
// report current capacity/occupancy, and ReclaimableBytes reports current ghost-byte
// pressure (which compaction actively drives back down).
type Stats struct {
	Gets             uint64
	Hits             uint64
	Misses           uint64
	Puts             uint64
	Dels             uint64
	Expirations      uint64 // expired by sweeper or lazy-on-read
	Evictions        uint64 // displaced by ringbuf eviction
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
