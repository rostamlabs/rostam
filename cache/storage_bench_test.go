// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkRingbufCodec measures the per-entry encode/decode kernels. encodeEntry
// includes the CRC32 (mmap/durable path); encodeEntryNoCRC is the heap-mode write
// path; decodeEntryFast is the hot Get/Del/sweep read path (no CRC).
func BenchmarkRingbufCodec(b *testing.B) {
	key := []byte("benchmark-key-0123456789")
	val := benchValue()
	dst := make([]byte, entrySize(len(key), len(val)))
	enc, _ := encodeEntry(dst, key, val, 0, makeMeta(1, false))
	_ = enc

	b.Run("encodeCRC", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_, _ = encodeEntry(dst, key, val, 0, makeMeta(1, false))
		}
	})
	b.Run("encodeNoCRC", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_, _ = encodeEntryNoCRC(dst, key, val, 0, makeMeta(1, false))
		}
	})
	b.Run("decodeFast", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_, _, _, _ = decodeEntryFast(dst)
		}
	})
	b.Run("decodeCRC", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_, _, _, _, _ = decodeEntry(dst)
		}
	})
}

// BenchmarkPageWrite measures raw slab appends into a heap page, resetting when
// full. This is the per-Put storage cost minus the shard lock + index update.
func BenchmarkPageWrite(b *testing.B) {
	key := []byte("benchmark-key-0123456789")
	val := benchValue()
	need := entrySize(len(key), len(val))
	p := newHeapPage(1 << 20)
	b.ReportAllocs()
	for b.Loop() {
		if p.FreeTail() < need {
			p.Reset()
		}
		_, _, _ = p.Write(key, val, 0, 0)
	}
}

// BenchmarkShardPutEvict measures Put throughput when the shard is at capacity
// and every write must evict to make room (PolicyRingbufEvict steady state). This
// exercises evictUntilFitsLocked + EvictFront, the slab free-list path. PageCount
// is varied because the eviction scan cost is sensitive to the number of pages.
func BenchmarkShardPutEvict(b *testing.B) {
	val := benchValue()
	for _, pages := range []int{4, 16, 64} {
		b.Run(fmt.Sprintf("pages=%d", pages), func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.NumShards = 1
			cfg.PageSize = 1 << 20 // 1 MiB
			cfg.MaxMemoryPerShard = pages << 20
			cfg.TTLSweepIntervalMs = 0 // no background sweeper interference
			s, err := newShard(cfg, "", nil)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = s.Close() }()
			// Fill to capacity so the timed loop is pure steady-state eviction.
			i := 0
			for s.pagesAlloc.Load() < uint64(pages) || s.evictions.Load() == 0 {
				k := fmt.Appendf(nil, "k%012d", i)
				_ = s.Put(k, val, 0)
				i++
				if i > 1<<24 { // safety valve
					break
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				k := fmt.Appendf(nil, "k%012d", i)
				_ = s.Put(k, val, 0)
				i++
			}
		})
	}
}

// BenchmarkShardSweep measures one TTL sweep pass over a populated index. The
// sweeper walks every index entry, reads its expiry, and deletes the expired
// ones. Here all entries are live (far-future TTL) so it measures the full walk
// cost without mutating the map mid-iteration.
func BenchmarkShardSweep(b *testing.B) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0
	s, err := newShard(cfg, "", nil)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	val := benchValue()
	for i := range 100_000 {
		k := fmt.Appendf(nil, "k%012d", i)
		_ = s.Put(k, val, time.Hour)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		s.sweepOnce()
	}
}
