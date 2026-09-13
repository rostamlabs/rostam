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
// THE MEDIAN BARELY MOVES while that happens, which is what identifies the mechanism. If the
// shared arm were doing more WORK per write — colder caches, a longer probe, a contended
// cache line — p50 would move with p999. It does not: p50 stays within a small factor across
// every writer count in both arms. A distribution whose middle is unchanged and whose tail
// grows a hundredfold is a queue, not a cost.
//
// AND THE QUEUE IS FAR LONGER THAN THE WORK IN IT. With a service time of a few hundred
// nanoseconds, a writer that merely waited its turn behind seven others would wait a couple
// of microseconds. It waits hundreds. So the tail is not the queue's LENGTH — it is what
// happens when a waiter stops spinning and parks, and has to be scheduled back on. A mutex
// profile of the shared arm attributes essentially all of the recorded blocking delay to the
// shard lock, split between the put and the delete.
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

// writeTailContendLive is the live entry count held by EACH shard in both arms, so the
// index table is the same size whichever arm is running and a probe costs the same.
const writeTailContendLive = 1 << 14

func writeTailContentionArm(b *testing.B, writers int, shared bool) {
	opsEach := b.N / writers
	if opsEach < 1 {
		opsEach = 1
	}

	// Per-worker state. In the shared arm every worker points at the same cache and owns
	// a disjoint slice of its keys; in the separate arm each worker gets its own.
	type worker struct {
		c        *Cache
		s        *shard
		keyBase  uint64 // first key id this worker owns
		liveEach int    // how many of the shard's live entries are this worker's
	}
	ws := make([]worker, writers)
	var caches []*Cache
	defer func() {
		for _, c := range caches {
			_ = c.Close()
		}
	}()

	if shared {
		c, s := newIndexRehashShard(b, writeTailContendLive+opsEach*writers)
		caches = append(caches, c)
		per := writeTailContendLive / writers
		for i := range ws {
			// Each worker's ids are spaced a whole window apart so the stripes stay
			// disjoint for the life of the run.
			ws[i] = worker{c: c, s: s, keyBase: uint64(i) * uint64(per+opsEach), liveEach: per}
		}
	} else {
		for i := range ws {
			c, s := newIndexRehashShard(b, writeTailContendLive+opsEach)
			caches = append(caches, c)
			ws[i] = worker{c: c, s: s, keyBase: 0, liveEach: writeTailContendLive}
		}
	}

	val := make([]byte, indexRehashValueLen)
	var kbuf [indexRehashKeyLen]byte
	for i := range ws {
		for j := range ws[i].liveEach {
			indexRehashKey(&kbuf, ws[i].keyBase+uint64(j))
			if err := ws[i].c.Put(kbuf[:], val, 0); err != nil {
				b.Fatalf("warm worker %d key %d: %v", i, j, err)
			}
		}
	}

	var (
		mu  sync.Mutex
		all latHist
		wg  sync.WaitGroup
	)
	b.ReportAllocs()
	b.ResetTimer()
	for i := range ws {
		wg.Add(1)
		go func(w worker) {
			defer wg.Done()
			var mine latHist
			var kb [indexRehashKeyLen]byte
			v := make([]byte, indexRehashValueLen)
			for j := range opsEach {
				indexRehashKey(&kb, w.keyBase+uint64(w.liveEach+j))
				t0 := time.Now()
				err := w.c.Put(kb[:], v, 0)
				mine.add(uint64(time.Since(t0))) //nolint:gosec // a duration here is never negative
				if err != nil {
					panic(fmt.Sprintf("put: %v", err))
				}
				indexRehashKey(&kb, w.keyBase+uint64(j))
				if _, err := w.c.Del(kb[:]); err != nil {
					panic(fmt.Sprintf("del: %v", err))
				}
			}
			mu.Lock()
			all.merge(&mine)
			mu.Unlock()
		}(ws[i])
	}
	wg.Wait()
	b.StopTimer()

	var rehashes uint64
	for i := range ws {
		if shared && i > 0 {
			break // one shard, already counted
		}
		if got := ws[i].s.evictions.Load(); got != 0 {
			b.Fatalf("shard evicted %d times; the arm is meant to run without capacity pressure", got)
		}
		if got := ws[i].s.rejects.Load(); got != 0 {
			b.Fatalf("shard refused %d writes; size the arm's memory for its op count", got)
		}
		rehashes += ws[i].s.indexRehashes.Load()
	}

	b.ReportMetric(float64(all.n), "samples")
	b.ReportMetric(float64(all.quantile(0.50)), "p50_ns")
	b.ReportMetric(float64(all.quantile(0.99)), "p99_ns")
	b.ReportMetric(float64(all.quantile(0.999)), "p999_ns")
	b.ReportMetric(float64(all.max), "max_ns")
	b.ReportMetric(float64(rehashes), "rehashes")
}
