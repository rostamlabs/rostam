// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"math/bits"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// BenchmarkRelocatingEvictionTail measures what a SINGLE write pays, which is the
// question every other row in this package's A/B tables cannot answer.
//
// WHY A MEAN IS THE WRONG SHAPE HERE. Eviction under PolicyRingbufEvict is positional and
// bursty: the write that finds no page with room drains a whole page under the shard write
// lock, and relocation copies up to PageSize/relocateMaxBytesPerEvictionDivisor bytes in
// the same hold, while the thousands of writes between two evictions touch none of it. A
// ns/op figure spreads that over every write and reports a cost no individual write ever
// pays. What the write path actually does is give almost every write the fast path and
// hand one write in a few thousand the bill for all of them.
//
// WHAT IS REPORTED, per arm: p50, p90, p99, p999 and max of per-write latency, and then
// the attribution — the fraction of writes that overlapped a page-freeing hold on their
// own shard, and the p999 of those writes against the p999 of the ones that did not.
// Those last three are what separate "the tail is large" from "the tail is THIS hold",
// which are different findings for anyone setting out to fix it.
//
// HOW A WRITE IS CLASSIFIED. Before timing, the harness resolves the key's shard exactly
// as Cache.Put does and reads that shard's eviction counter; after the write it reads it
// again. A write whose shard's counter moved is counted as HELD. That is deliberately
// broader than "this write performed the eviction": it also catches a write that merely
// BLOCKED on s.mu while some other holder freed a page, which is the same cost to the
// caller and, on the background arm, is precisely the question — whether the reserve moves
// the hold out of the write or merely moves it alongside. It biases against the background
// arm if anything, since that arm has a second party freeing pages, so a tail improvement
// measured here is not an artefact of the classification.
//
// HOW THE SAMPLES ARE COLLECTED. Each parallel worker owns a fixed-size log histogram and
// a running maximum, allocated BEFORE the timed loop and merged under a mutex after it, so
// the loop itself does no allocation and takes no shared lock. The cost inside the timed
// region is two time.Now calls; that overhead lands on every arm equally but it is a
// meaningful fraction of p50, so read p50 as an upper bound and the ratios as the result.
// The uninstrumented mean is in BenchmarkRelocatingEvictionABSharded.
//
// ==========================================================================
// WHAT IT FOUND, and the first finding revises the reason this benchmark was written.
//
// MOST OF THE TAIL IS NOT THE EVICTION HOLD. Writes whose own shard froze no page at all
// — the clean class, which by construction never waited on a retire — still reach roughly
// 0.6 to 0.9 ms at p999, in EVERY arm including the one with no relocation compiled into
// the path. Against a p50 under a microsecond that is about a thousandfold, and none of
// it is anything either relocation pass touches. The read baseline places the floor the
// machine imposes far below that (tens to a couple of hundred microseconds at p999), so
// this is not merely the box either: there is several hundred microseconds of write-path
// tail that is neither eviction nor scheduler. The leading suspect is the index rehash,
// which allocates a table and re-inserts every live entry under the write lock and grows
// with the entry count — that is a HYPOTHESIS, not a measurement, and nothing here has
// tested it.
//
// RELOCATION'S OWN CONTRIBUTION IS REAL AND IT SHOWS AT p99, which is where the write-path
// pass roughly doubles the no-relocation figure. That increment is the hold, and it is the
// part of the tail either scheme can do anything about.
//
// THE RESERVE HELPS AT p99 AND HURTS AT p999, which is the shape a mean cannot show and
// the reason this benchmark exists. It roughly HALVES how often a write meets a
// page-freeing hold at all, and it gives back most of the p99 the write-path pass added —
// a far larger improvement than the few per cent its mean shows. But the holds it does
// leave are LONGER, so p999 and max come out worse than the write-path pass's: the
// reserve's retire runs retirePageLocked and then relocateIntoFreedPageLocked in ONE
// uninterrupted lock acquisition, and a write that collides with it waits out the whole
// thing from outside rather than performing a shorter one itself.
//
// So the tail argument for the reserve is "fewer writes meet a hold, and p99 improves
// accordingly", not "the worst write gets better" — the worst write gets worse. Chunking
// that hold is the change both halves of that sentence point at; it is deliberately not
// made here, because it would also change the write path's own pass.
func BenchmarkRelocatingEvictionTail(b *testing.B) {
	for _, shards := range relocABShardCounts {
		b.Run(fmt.Sprintf("shards=%d", shards), func(b *testing.B) {
			relocTailArms(b, shards)
		})
	}
}

// relocTailArms runs the three arms worth distributions — no relocation, relocation on the
// write path, and relocation with the reserve on its own ticker at the shipped default —
// plus a READ baseline on the first of them.
//
// THE READ BASELINE IS THE CONTROL THAT MAKES THE TABLE READABLE, and leaving it out is
// how a tail table gets over-read. A Get takes only the read lock and can never evict,
// allocate a page or rehash, so whatever tail it still shows is the floor this harness and
// this machine impose on ANY operation — scheduler, preemption, the box — and none of it
// is anything either relocation pass could fix. A write's p999 means one thing against a
// read p999 of the same order and quite another against one a fifth the size. The swept
// controls are not repeated here: those price a ticker, and this benchmark is about what
// the write pays.
func relocTailArms(b *testing.B, shards int) {
	keyspace := relocABKeyspace * shards
	keys := make([][]byte, keyspace)
	for i := range keys {
		keys[i] = fmt.Appendf(nil, "ab-%08d", i)
	}
	val := make([]byte, relocABValueLen)
	for i := range val {
		val[i] = byte(i)
	}

	for _, arm := range relocABArmList() {
		switch arm.name {
		case "off", "sync", "background":
		default:
			continue
		}
		// Built and warmed outside b.Run for the same reason the sharded A/B does it; see
		// relocABShardedArms.
		cfg := DefaultConfig()
		cfg.NumShards = shards
		cfg.PageSize = 1 << 20
		cfg.MaxMemoryPerShard = relocABPages << 20
		cfg.InitialPagesPerShard = 0
		cfg.AtCapPolicy = PolicyRingbufEvict
		cfg.TTLSweepIntervalMs = arm.ttlMs
		cfg.RelocateReserveIntervalMs = arm.reserveMs
		cfg.RelocatingEviction = arm.on
		c, err := New(cfg)
		if err != nil {
			b.Fatal(err)
		}
		relocABShardedFill(c, keys, val, relocABWarmOps*shards)

		if arm.name == "off" {
			relocTailReadBaseline(b, c, keys, keyspace)
		}

		var (
			mu          sync.Mutex
			clean, held latHist
			workers     int64
		)
		b.Run("reloc="+arm.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				mu.Lock()
				workers++
				seed := relocABSeed + workers
				mu.Unlock()
				rng := rand.New(rand.NewSource(seed))                       //nolint:gosec // benchmark RNG
				z := rand.NewZipf(rng, relocABZipfS, 1, uint64(keyspace-1)) //nolint:gosec // keyspace > 1
				var myClean, myHeld latHist
				for pb.Next() {
					k := keys[z.Uint64()]
					_, s := c.shardForH(k)
					before := s.evictions.Load()
					t0 := time.Now()
					_ = c.Put(k, val, 0)
					ns := uint64(time.Since(t0)) //nolint:gosec // a duration here is never negative
					if s.evictions.Load() == before {
						myClean.add(ns)
					} else {
						myHeld.add(ns)
					}
				}
				mu.Lock()
				clean.merge(&myClean)
				held.merge(&myHeld)
				mu.Unlock()
			})
			b.StopTimer()

			var all latHist
			all.merge(&clean)
			all.merge(&held)
			b.ReportMetric(float64(all.quantile(0.50)), "p50_ns")
			b.ReportMetric(float64(all.quantile(0.90)), "p90_ns")
			b.ReportMetric(float64(all.quantile(0.99)), "p99_ns")
			b.ReportMetric(float64(all.quantile(0.999)), "p999_ns")
			b.ReportMetric(float64(all.max), "max_ns")
			frac := 0.0
			if all.n > 0 {
				frac = float64(held.n) / float64(all.n)
			}
			b.ReportMetric(frac, "held_frac")
			b.ReportMetric(float64(clean.quantile(0.999)), "clean_p999_ns")
			b.ReportMetric(float64(held.quantile(0.999)), "held_p999_ns")
		})
		_ = c.Close()
	}
}

// relocTailReadBaseline measures the same distribution over Gets on an identically warmed
// cache. It shares every column name with the write rows so the two can be read side by
// side; held_frac and the two attributed quantiles are omitted, since a read holds nothing
// and evicts nothing. See relocTailArms for why this row is not optional.
func relocTailReadBaseline(b *testing.B, c *Cache, keys [][]byte, keyspace int) {
	var (
		mu      sync.Mutex
		all     latHist
		workers int64
	)
	b.Run("reads", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			mu.Lock()
			workers++
			seed := relocABSeed + workers
			mu.Unlock()
			rng := rand.New(rand.NewSource(seed))                       //nolint:gosec // benchmark RNG
			z := rand.NewZipf(rng, relocABZipfS, 1, uint64(keyspace-1)) //nolint:gosec // keyspace > 1
			var mine latHist
			for pb.Next() {
				k := keys[z.Uint64()]
				t0 := time.Now()
				_, _ = c.Get(k)
				mine.add(uint64(time.Since(t0))) //nolint:gosec // a duration here is never negative
			}
			mu.Lock()
			all.merge(&mine)
			mu.Unlock()
		})
		b.StopTimer()
		b.ReportMetric(float64(all.quantile(0.50)), "p50_ns")
		b.ReportMetric(float64(all.quantile(0.90)), "p90_ns")
		b.ReportMetric(float64(all.quantile(0.99)), "p99_ns")
		b.ReportMetric(float64(all.quantile(0.999)), "p999_ns")
		b.ReportMetric(float64(all.max), "max_ns")
	})
}

// latHist is a fixed log histogram of nanosecond latencies: eight linear buckets below
// latSub, then latSub buckets per power of two, which is about 6% worst-case bucket width
// — far finer than the differences this benchmark is looking for and small enough that a
// worker can own one without allocating inside the timed loop.
const (
	latSub     = 8
	latBuckets = 256
)

type latHist struct {
	b   [latBuckets]uint64
	n   uint64
	max uint64
}

func (h *latHist) add(ns uint64) {
	h.b[latIndex(ns)]++
	h.n++
	if ns > h.max {
		h.max = ns
	}
}

func (h *latHist) merge(o *latHist) {
	for i := range o.b {
		h.b[i] += o.b[i]
	}
	h.n += o.n
	if o.max > h.max {
		h.max = o.max
	}
}

// quantile returns the lower bound of the bucket holding the q-th value, or 0 when the
// histogram is empty. Bucket lower bounds, not interpolated midpoints: the figure is then
// always a latency some write actually met or exceeded.
func (h *latHist) quantile(q float64) uint64 {
	if h.n == 0 {
		return 0
	}
	want := uint64(q * float64(h.n))
	var seen uint64
	for i := range h.b {
		seen += h.b[i]
		if seen >= want {
			return latValue(i)
		}
	}
	return h.max
}

// latIndex maps a nanosecond figure to its bucket: linear below latSub, then latSub
// buckets per octave with the leading bit implicit.
func latIndex(ns uint64) int {
	if ns < latSub {
		return int(ns)
	}
	octave := bits.Len64(ns) - 1
	idx := (octave-2)*latSub + int((ns>>(octave-3))&(latSub-1))
	if idx >= latBuckets {
		return latBuckets - 1
	}
	return idx
}

// latValue is latIndex's inverse onto the bucket's lower bound.
func latValue(idx int) uint64 {
	if idx < latSub {
		return uint64(idx)
	}
	octave := idx/latSub + 2
	return (latSub + uint64(idx%latSub)) << (octave - 3) //nolint:gosec // idx < latBuckets
}
