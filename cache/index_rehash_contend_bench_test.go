// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// BenchmarkWriteTailContention isolates the OTHER half of the write tail: how much of it
// is the write's own work, and how much is waiting for somebody else's.
//
// BenchmarkIndexRehashTail measures one writer alone and finds a p999 of a couple of
// microseconds — three orders of magnitude below the sub-millisecond p999 a parallel
// write benchmark reports for writes that evicted nothing. A per-write cost cannot differ
// by three orders of magnitude between one writer and several: the write does the same
// work either way. So the difference is not IN the write, and this benchmark is built to
// say what it is instead.
//
// THE TWO ARMS DIFFER IN EXACTLY ONE THING — whether the writers share a lock.
//
//	lock=shared     W writers on ONE shard. Every write takes the same s.mu.
//	lock=separate   W writers, each on its OWN single-shard cache. Same W goroutines,
//	                same per-write work, same allocation, same machine load — but no
//	                two writers ever contend for a lock.
//
// Both arms hold the same number of live entries PER SHARD, so the index table has the
// same size and a probe walks the same number of slots in either. Both run the same
// no-eviction turnover workload as BenchmarkIndexRehashTail, for the same reason: nothing
// measured here may be chargeable to capacity pressure.
//
// The separate arm is what makes the shared arm's number mean anything. W goroutines
// hammering a machine produce scheduler latency, GC assists and preemption whether or not
// they share a lock, and all of that lands on the tail. Only the DIFFERENCE between the
// arms is the lock.
//
// Run with a fixed op count, e.g. -benchtime=500000x: b.N is split across the writers, so
// the per-arm sample count is the op count and a p999 rests on a thousandth of it.
//
// ==========================================================================
// WHAT IT FOUND. Sharing a shard is the write tail, and nothing else here comes close.
//
// At one writer the two arms are the same experiment and they report the same distribution:
// a p999 of about two microseconds. Put a SECOND writer on the same shard and p999 jumps to
// tens of microseconds; at four it is well over a hundred; at eight it is a few hundred. The
// separate arm, running the same number of goroutines doing the same work on the same box,
// stays at about two microseconds throughout. The gap between the arms reaches two orders of
// magnitude, and it widens with every writer added.
//
// THE MEDIAN MOVES BY A SMALL FACTOR WHILE THE TAIL MOVES BY TWO ORDERS, which is what
// identifies the mechanism. p50 in the shared arm does rise with the writer count — it is
// not flat, and claiming it were would be overstating this — but it rises by something like
// two to five times where p999 rises by around two hundred. If the shared arm were simply
// doing more WORK per write (colder caches, a longer probe, a contended cache line), the two
// would move together, because every write would pay. A middle that creeps while the tail
// explodes is a queue: almost every write still goes straight through, and the few that
// arrive while the lock is held wait orders longer than the work itself takes.
//
// AND THE QUEUE IS FAR LONGER THAN THE WORK IN IT. With a service time of a few hundred
// nanoseconds, a writer that merely waited its turn behind seven others would wait a couple
// of microseconds. It waits hundreds. So the tail is not the queue's LENGTH — it is what
// happens when a waiter stops spinning and parks, and has to be scheduled back on. A mutex
// profile of the shared arm attributes essentially all of the recorded blocking delay to the
// shard lock, split between the put and the delete.
//
// THROUGHPUT SAYS THE SAME THING FROM THE OTHER SIDE. In the separate arm the loop's
// aggregate ns/op falls by roughly a factor of four from one writer to eight — writers
// scaling as writers should — while in the shared arm it RISES by a factor of three or more
// over the same range. Per-write mean (put_mean_ns) diverges even harder than p50 does,
// because the mean is what a heavy tail drags.
//
// The max column is the one thing the arms roughly SHARE: both reach milliseconds at every
// writer count, including at one writer where neither can be waiting for anything. That is the
// machine's own floor — the garbage collector and the scheduler — and nothing about shard
// geometry moves it. Read max as the box and p999 as the lock.
//
// THE COROLLARY, for anyone reading this before optimising a write path: the tail belongs to
// the LOCK, so the levers are how long it is held and how many writers share it. Shortening a
// hold helps every writer behind it. Making one write's own work cheaper does not, unless that
// work was inside the hold.
func BenchmarkWriteTailContention(b *testing.B) {
	for _, writers := range []int{1, 2, 4, 8} {
		for _, shared := range []bool{true, false} {
			name := "separate"
			if shared {
				name = "shared"
			}
			b.Run(fmt.Sprintf("writers=%d/lock=%s", writers, name), func(b *testing.B) {
				writeTailContentionArm(b, writers, shared)
			})
		}
	}
}

// writeTailContendLive is the live entry count held by EVERY shard the arm builds, in
// both arms, so the index table is the same size whichever arm is running and a probe
// walks the same number of slots.
const writeTailContendLive = 1 << 14

// writeTailContentionArm runs one (writers, shared) cell.
//
// BOTH ARMS BUILD AND WARM THE SAME NUMBER OF CACHES, which is what makes the comparison
// mean what it claims. An earlier version built W caches in the separate arm and ONE in
// the shared arm, so the separate arm carried W times the resident index and page
// footprint — the very cache-pressure and collector effects the arms are supposed to hold
// constant. Now W caches are built and warmed identically in both, and the ONLY difference
// is which cache the workers are pointed at: all of them at caches[0] when the lock is
// shared, worker i at caches[i] when it is not. Every worker executes a byte-identical key
// sequence in either arm, against a shard holding an identical live set; the W-1 caches
// that go idle in the shared arm stay resident exactly as their busy counterparts do in
// the other.
//
// Note which way the old asymmetry cut, because it is why the result survived it: the arm
// carrying the LARGER footprint was the FASTER one. The confound worked against the
// measured effect rather than manufacturing it.
func writeTailContentionArm(b *testing.B, writers int, shared bool) {
	opsEach := b.N / writers
	if opsEach < 1 {
		opsEach = 1
	}
	// Every worker owns a disjoint stripe, spaced a whole window apart so the stripes stay
	// disjoint for the life of the run. Each cache is warmed with EVERY stripe, so its live
	// set is the same whether the arm will drive one stripe of it or all of them.
	per := writeTailContendLive / writers
	stripe := uint64(per + opsEach) //nolint:gosec // both are non-negative benchmark sizes
	keyBase := func(i int) uint64 { return uint64(i) * stripe }

	// One capacity for every cache in both arms: enough for the warm set plus every write
	// the busiest cache could take. Pages are allocated lazily, so headroom an idle cache
	// never writes into costs nothing resident.
	capacity := writeTailContendLive + opsEach*writers
	caches := make([]*Cache, writers)
	shards := make([]*shard, writers)
	defer func() {
		for _, c := range caches {
			if c != nil {
				_ = c.Close()
			}
		}
	}()
	for i := range caches {
		caches[i], shards[i] = newIndexRehashShard(b, capacity)
	}

	val := make([]byte, indexRehashValueLen)
	var kbuf [indexRehashKeyLen]byte
	for ci := range caches {
		for w := range writers {
			for j := range per {
				indexRehashKey(&kbuf, keyBase(w)+uint64(j))
				if err := caches[ci].Put(kbuf[:], val, 0); err != nil {
					b.Fatalf("warm cache %d stripe %d key %d: %v", ci, w, j, err)
				}
			}
		}
	}

	// Baselines taken AFTER the fill. Warming rehashes on its own, and does so in every
	// cache, so a total that included it would report work no latency sample ever saw —
	// and would count W caches' worth of it here while the measured window touches one.
	// Only the delta over the measured window is reported.
	rehashesAtStart := make([]uint64, writers)
	for i, sh := range shards {
		rehashesAtStart[i] = sh.indexRehashes.Load()
	}

	var (
		mu  sync.Mutex
		all latHist
		wg  sync.WaitGroup
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := range writers {
		target := i
		if shared {
			target = 0
		}
		wg.Add(1)
		go func(c *Cache, base uint64) {
			defer wg.Done()
			var mine latHist
			var kb [indexRehashKeyLen]byte
			v := make([]byte, indexRehashValueLen)
			for j := range opsEach {
				indexRehashKey(&kb, base+uint64(per+j))
				t0 := time.Now()
				err := c.Put(kb[:], v, 0)
				mine.add(uint64(time.Since(t0))) //nolint:gosec // a duration here is never negative
				if err != nil {
					panic(fmt.Sprintf("put: %v", err))
				}
				// Retire the key written `per` steps ago, holding the live set flat so
				// every write leaves a tombstone. It is outside the MANUAL timing region
				// and contributes to no reported quantile — but it is still inside the
				// BENCHMARK timer, so Go's own ns/op covers the put and the delete
				// together. put_mean_ns is the mean of the timed operation; ns/op is the
				// loop's aggregate throughput. They are different figures and reading one
				// against the reported quantiles is a mistake.
				indexRehashKey(&kb, base+uint64(j))
				if _, err := c.Del(kb[:]); err != nil {
					panic(fmt.Sprintf("del: %v", err))
				}
			}
			mu.Lock()
			all.merge(&mine)
			mu.Unlock()
		}(caches[target], keyBase(i))
	}
	wg.Wait()
	b.StopTimer()

	var rehashes uint64
	for i, sh := range shards {
		if got := sh.evictions.Load(); got != 0 {
			b.Fatalf("cache %d evicted %d times; the arm is meant to run without capacity pressure", i, got)
		}
		if got := sh.rejects.Load(); got != 0 {
			b.Fatalf("cache %d refused %d writes; size the arm's memory for its op count", i, got)
		}
		rehashes += sh.indexRehashes.Load() - rehashesAtStart[i]
	}

	b.ReportMetric(float64(all.n), "samples")
	b.ReportMetric(all.mean(), "put_mean_ns")
	b.ReportMetric(emptyAsNaN(&all, all.quantile(0.50)), "p50_ns")
	b.ReportMetric(emptyAsNaN(&all, all.quantile(0.99)), "p99_ns")
	b.ReportMetric(emptyAsNaN(&all, all.quantile(0.999)), "p999_ns")
	b.ReportMetric(emptyAsNaN(&all, all.max), "max_ns")
	b.ReportMetric(float64(rehashes), "rehashes")
}
