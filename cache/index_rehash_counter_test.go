// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"testing"
	"time"
)

// TestIndexRehashCounterCountsEveryTableSwap guards the counter every attribution in
// BenchmarkIndexRehashTail and BenchmarkRelocatingEvictionTail rests on. Those benchmarks
// decide whether a write performed a rehash by watching s.indexRehashes move across it, and
// a counter that under-reported would silently move rehashing writes into the clean class
// and make the rehash look innocent — which is very nearly the conclusion those benchmarks
// draw, so the counter is exactly the thing that must not be taken on trust.
//
// The check is identity, not a threshold: a rehash is the ONE thing that replaces the table
// OBJECT on a shard doing nothing but puts and deletes, so counting how many times
// s.tab.Load() returns a different pointer gives an independent tally to hold the counter
// against. They must agree exactly, and both must be non-zero — a run that never reached the
// resize threshold would satisfy "they agree" while testing nothing, so that is failed
// explicitly.
func TestIndexRehashCounterCountsEveryTableSwap(t *testing.T) {
	const (
		live = 1 << 11
		ops  = 40_000
	)
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 64 << 20
	cfg.InitialPagesPerShard = 0
	// Reject-writes with room to spare, and no sweeper: eviction and the TTL sweep both
	// tombstone, and either would swap the table on a schedule this test does not control.
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	val := make([]byte, indexRehashValueLen)
	var kbuf [indexRehashKeyLen]byte
	put := func(n uint64) {
		t.Helper()
		indexRehashKey(&kbuf, n)
		if err := c.Put(kbuf[:], val, 0); err != nil {
			t.Fatalf("put %d: %v", n, err)
		}
	}
	del := func(n uint64) {
		t.Helper()
		indexRehashKey(&kbuf, n)
		if _, err := c.Del(kbuf[:]); err != nil {
			t.Fatalf("del %d: %v", n, err)
		}
	}

	for i := range live {
		put(uint64(i))
	}
	// Turnover at a FLAT live set is what keeps tripping the threshold: the live count
	// never grows, but every delete leaves a tombstone and the threshold counts those too.
	// Both tallies start HERE, after the fill. Building the live set rehashes several
	// times on its own, and a counter read from before it would be compared against swaps
	// that were never watched for.
	var swaps uint64
	rehashesAtStart := s.indexRehashes.Load()
	prev := s.tab.Load()
	countSwap := func() {
		if cur := s.tab.Load(); cur != prev {
			swaps++
			prev = cur
		}
	}
	for i := range ops {
		put(uint64(live + i))
		countSwap()
		del(uint64(i))
		countSwap()
	}

	if swaps == 0 {
		t.Fatal("the table was never replaced: this workload did not reach the resize threshold, so the counter was not tested")
	}
	if got := s.indexRehashes.Load() - rehashesAtStart; got != swaps {
		t.Fatalf("indexRehashes moved by %d, but the table object was replaced %d times", got, swaps)
	}
	// The COUNT above is exact on every platform and is the load-bearing assertion.
	// The nanos are asserted too, but only when this machine's MONOTONIC clock can
	// actually resolve the work: a rehash of a table this small takes a few
	// microseconds, and time.Since is quantised to the clock's tick, so on a host
	// whose tick is coarser than the whole run the counter legitimately reads zero.
	//
	// Gate on the MEASURED granularity rather than on GOOS. That keeps the
	// assertion live wherever it can hold, says in the failure message what the
	// clock actually did, and starts asserting by itself if a platform's clock
	// improves. (Windows is the case that forced this: Go's nanotime1 there reads
	// _INTERRUPT_TIME from the shared data page, which ticks at the system timer
	// interval rather than at the ~100 ns the precise wall-clock call would give.)
	if g := monotonicGranularity(); g <= time.Microsecond {
		if s.indexRehashNanos.Load() == 0 {
			t.Fatalf("indexRehashNanos = 0 after %d rehashes, on a clock with %v granularity", swaps, g)
		}
	} else {
		t.Logf("skipping the nanos assertion: monotonic clock granularity is %v, coarser than the work being timed", g)
	}
}

// monotonicGranularity returns the smallest non-zero interval time.Since can
// report on this machine, by spinning until the monotonic reading changes. It is
// the quantum below which timing a short operation yields zero however long the
// operation really took.
func monotonicGranularity() time.Duration {
	best := time.Duration(1 << 62)
	for range 5 {
		t0 := time.Now()
		var d time.Duration
		for d == 0 {
			d = time.Since(t0)
		}
		if d < best {
			best = d
		}
	}
	return best
}
