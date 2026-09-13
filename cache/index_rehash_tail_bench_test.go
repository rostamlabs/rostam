// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"fmt"
	"testing"
	"time"
)

// BenchmarkIndexRehashTail asks one question: how much of a write's LATENCY TAIL is the
// index table's growth step?
//
// The step is real and it is unbounded. When a table's fill (live entries plus
// tombstones) reaches three quarters of its slots, the writer holding s.mu allocates a
// fresh table and re-inserts every live entry into it, one upsert each, before releasing
// the lock. That is O(entries) work charged to whichever single write happened to trip
// the threshold, and it blocks every other writer on the shard for its duration. Nothing
// about that shape is in doubt. What is in doubt is whether it is big enough, OFTEN
// enough, to be what a tail quantile is made of — and frequency is the half that reading
// the code does not answer.
//
// WHY TOMBSTONES ARE THE INTERESTING CASE, and why a growth-only workload would be the
// wrong experiment. Under pure insertion the table doubles, so a run of N writes pays
// only about log2(N) rehashes — far too rare to reach any quantile anyone reports, and a
// measurement built that way would exonerate the step for a reason that does not
// generalise. But the threshold counts TOMBSTONES alongside live entries, and a deleted
// or evicted key leaves one behind. So a shard whose live set is FLAT but whose keys turn
// over keeps tripping the threshold forever, at a steady-state rate, which is the shape a
// cache under a real workload actually has. This benchmark drives exactly that: a
// constant live set with every write introducing a new key and retiring an old one.
//
// THE ARMS.
//
//	table=grow     the shipped sizing: the table is built from an empty shard and
//	               resizes whenever the threshold is met.
//	table=presized the control: the table is given enough slots up front that the
//	               threshold cannot be reached inside the measured window. The arm
//	               FAILS if a single rehash fires, so the control is checked rather
//	               than assumed. If the tail is the rehash, it must vanish here.
//
// Both arms are swept across live-set sizes, because that is the discriminator the
// hypothesis cannot dodge: the step costs O(entries), so if it is the tail then the
// tail's magnitude must grow with the live set. A tail that is flat in the live set is
// not this step.
//
// WHAT IS REPORTED, per arm: p50/p99/p999/max over every write, the number of samples
// behind them, and then the attribution — how many writes performed a rehash, what those
// writes cost, and what the p999 is of the writes that did NOT. The last pair is the
// result. A tail that is the rehash shows a no-rehash p999 far below the overall one; a
// tail that is something else shows the two the same.
//
// The classification is exact rather than statistical: the harness reads the shard's
// rehash counter either side of the write, so a write is attributed to a rehash only if
// one actually ran inside it. This is a single-writer arm, so unlike the eviction-hold
// classification in BenchmarkRelocatingEvictionTail there is no blocked-on-someone-else
// case to fold in.
//
// RUN IT WITH A FIXED OP COUNT, e.g. -benchtime=500000x. A p999 needs samples behind it,
// and the shard is deliberately never allowed to evict, so the op count also sets how
// much memory the arm reserves.
//
// ==========================================================================
// WHAT IT FOUND. The rehash is NOT the write tail. It is not close, and the shape of why
// is worth keeping, because the step looks like an excellent suspect right up until it is
// counted.
//
// THE STEP IS EXPENSIVE AND IT IS O(ENTRIES), exactly as reading the code suggests. Across
// the swept live sets its cost rises in step with them — roughly a hundred and fifty
// microseconds at four thousand entries, a couple of milliseconds at sixty-five thousand —
// and BenchmarkIndexRehashStep prices the same curve with no write path around it. A rehash
// is also the slowest write most runs see, and at the largest live set it is the slowest in
// every run — rehash_max_ns and max_ns the same sample. On that evidence alone the hypothesis
// looks confirmed.
//
// BUT ITS FREQUENCY FALLS JUST AS FAST, and the two cancel. The threshold is met once per
// table's worth of accumulated fill, so a shard holding L entries rehashes about once every
// few L writes: rehash_frac measures near 1e-4 at the smallest live set and near 1e-5 at the
// largest. The step cannot reach a quantile more common than its own rate, so there is no
// live set at which it is BOTH frequent enough to touch a p999 and large enough to matter
// there — make it frequent and it is small, make it large and it is rare. In every arm
// norehash_p999_ns equals p999_ns bucket for bucket: removing every rehashing write from the
// sample changes nothing.
//
// AND THE SWEEP ANSWERS IT WITHOUT ANY ATTRIBUTION AT ALL, which is the cleanest form of the
// result. p999 is FLAT across the live sets — the same couple of microseconds at four thousand
// entries as at sixty-five thousand — while the rehash it is supposed to be got sixteen times
// more expensive over that range. A tail that does not move when the only O(entries) step in
// the write path grows sixteenfold is not that step.
//
// THE PRE-SIZED CONTROL AGREES AND THEN GOES FURTHER. With the table sized so that no rehash
// can fire at all, p999 does not improve — and p50 gets roughly twice as SLOW, because a table
// with that much headroom no longer fits the caches a probe used to hit. So pre-sizing the
// index is not a latency fix waiting to be applied; measured here it is a regression on the
// common path that buys nothing at the tail.
//
// NOTE WHAT THE CONTROL HAD TO BE SIZED FROM, because it answers the obvious next idea —
// pre-size the table from the shard's memory budget, which bounds the live count. That does
// not work, and the arm demonstrates why: the control is sized from the OP COUNT, not the live
// set, and it has to be. Size it from the live set instead — even generously, several times
// over — and the arm fails with rehashes it was supposed to have prevented. The threshold
// counts tombstones, a delete leaves one, and tombstones accumulate until they fill whatever
// headroom the table was given. Headroom buys a number of operations proportional to the SLOT
// COUNT; it does not buy a steady state.
//
// WHAT THE TAIL ACTUALLY IS, measured in BenchmarkWriteTailContention: queueing on the shard
// write lock. Note the number to compare against — a single writer's p999 here is a couple of
// MICROSECONDS, two to three orders below what a parallel write benchmark reports for writes
// that evicted nothing. A per-write step cannot differ by that much between one writer and
// several; the wait for other writers can.
func BenchmarkIndexRehashTail(b *testing.B) {
	for _, live := range indexRehashLiveSets {
		for _, presized := range []bool{false, true} {
			name := "grow"
			if presized {
				name = "presized"
			}
			b.Run(fmt.Sprintf("live=%d/table=%s", live, name), func(b *testing.B) {
				indexRehashTailArm(b, live, presized)
			})
		}
	}
}

// indexRehashLiveSets are the live-set sizes the sweep runs at. They span two orders of
// magnitude because the point of the sweep is the SLOPE: a rehash costs O(entries), so
// its contribution has to move with these.
var indexRehashLiveSets = []int{1 << 12, 1 << 14, 1 << 16}

const (
	indexRehashKeyLen   = 16
	indexRehashValueLen = 8
)

func indexRehashTailArm(b *testing.B, live int, presized bool) {
	c, s := newIndexRehashShard(b, live+b.N)
	defer func() { _ = c.Close() }()

	if presized {
		// Sized for every key the arm will ever hold AND every tombstone it will ever
		// leave, so the threshold cannot be reached. Installed before the first write,
		// while the shard is still single-threaded and empty.
		s.tab.Store(newIndexTable(live + b.N))
	}

	val := make([]byte, indexRehashValueLen)
	var kbuf [indexRehashKeyLen]byte

	// Warm to the target live set. The grow arm pays its build-up rehashes here, outside
	// the measured window, so what the window sees is the steady-state turnover rate and
	// not the doubling sequence.
	for i := range live {
		indexRehashKey(&kbuf, uint64(i))
		if err := c.Put(kbuf[:], val, 0); err != nil {
			b.Fatalf("warm put %d: %v", i, err)
		}
	}

	rehashesAtStart := s.indexRehashes.Load()
	var all, rehashed, clean latHist

	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		indexRehashKey(&kbuf, uint64(live+i))
		before := s.indexRehashes.Load()
		t0 := time.Now()
		err := c.Put(kbuf[:], val, 0)
		ns := uint64(time.Since(t0)) //nolint:gosec // a duration here is never negative
		if err != nil {
			b.Fatalf("put %d: %v", live+i, err)
		}
		all.add(ns)
		if s.indexRehashes.Load() == before {
			clean.add(ns)
		} else {
			rehashed.add(ns)
		}
		// Retire the oldest key so the live set stays flat and each write leaves a
		// tombstone behind. Outside the timed region: the delete is the workload's
		// shape, not the thing being measured.
		indexRehashKey(&kbuf, uint64(i))
		if _, err := c.Del(kbuf[:]); err != nil {
			b.Fatalf("del %d: %v", i, err)
		}
	}
	b.StopTimer()

	// The premise of the whole arm: nothing here was evicted or refused, so no cost
	// below belongs to capacity pressure.
	if got := s.evictions.Load(); got != 0 {
		b.Fatalf("shard evicted %d times; the arm is meant to run without capacity pressure", got)
	}
	if got := s.rejects.Load(); got != 0 {
		b.Fatalf("shard refused %d writes; size the arm's memory for its op count", got)
	}
	rehashes := s.indexRehashes.Load() - rehashesAtStart
	if presized && rehashes != 0 {
		b.Fatalf("presized control rehashed %d times; the control is not controlling anything", rehashes)
	}

	b.ReportMetric(float64(all.n), "samples")
	b.ReportMetric(float64(all.quantile(0.50)), "p50_ns")
	b.ReportMetric(float64(all.quantile(0.99)), "p99_ns")
	b.ReportMetric(float64(all.quantile(0.999)), "p999_ns")
	b.ReportMetric(float64(all.max), "max_ns")
	b.ReportMetric(float64(rehashes), "rehashes")
	frac := 0.0
	if all.n > 0 {
		frac = float64(rehashed.n) / float64(all.n)
	}
	b.ReportMetric(frac, "rehash_frac")
	b.ReportMetric(float64(rehashed.quantile(0.50)), "rehash_p50_ns")
	b.ReportMetric(float64(rehashed.max), "rehash_max_ns")
	b.ReportMetric(float64(clean.quantile(0.999)), "norehash_p999_ns")
	b.ReportMetric(float64(clean.max), "norehash_max_ns")
}

// newIndexRehashShard builds a single-shard heap cache that can hold `entries` entries
// without ever evicting one. Reject-writes is the point: the arm must not be able to
// blame eviction for anything it measures, and a refused write fails the arm outright.
// The TTL sweeper is off for the same reason — it walks the index and can tombstone, so
// leaving it running would put a second, unattributed source of rehashes in the window.
func newIndexRehashShard(b *testing.B, entries int) (*Cache, *shard) {
	b.Helper()
	const pageSize = 4 << 20
	perEntry := entrySize(indexRehashKeyLen, indexRehashValueLen)
	// Round up to whole pages and add one, since an entry that does not fit a page's
	// tail room moves to the next page and strands the remainder.
	pages := (entries*perEntry)/pageSize + 2

	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = pageSize
	cfg.MaxMemoryPerShard = pages * pageSize
	cfg.InitialPagesPerShard = 0
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		b.Fatalf("new cache (%d entries, %d pages): %v", entries, pages, err)
	}
	return c, c.shards[0]
}

// indexRehashKey writes a fixed-width key for n into buf. Fixed width and no allocation:
// the timed loop must not be measuring a key formatter.
func indexRehashKey(buf *[indexRehashKeyLen]byte, n uint64) {
	copy(buf[:8], "rehash--")
	binary.BigEndian.PutUint64(buf[8:], n)
}

// BenchmarkIndexRehashStep prices the step itself, with no write path around it: build a
// table holding `live` entries, then time one rehashed() call on it.
//
// This is the calibration the tail benchmark's attribution is read against. The tail
// benchmark can say how OFTEN a rehash lands on a write; only this can say what a rehash
// of a given size COSTS, free of the lock, the page write and the timer overhead that
// surround it there. Together they bound the step's contribution to any quantile: a step
// that costs C and lands on a fraction f of writes cannot move a quantile below 1-f, and
// cannot move one above it by more than C.
func BenchmarkIndexRehashStep(b *testing.B) {
	for _, live := range indexRehashLiveSets {
		b.Run(fmt.Sprintf("live=%d", live), func(b *testing.B) {
			t := newIndexTable(live)
			for i := range live {
				t.upsert(indexRehashHash(uint64(i)), makeSlabRef(0, 1, uint32(i))) //nolint:gosec // i < live fits a uint32
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				_ = t.rehashed()
			}
			b.StopTimer()
			b.ReportMetric(float64(t.live), "live_entries")
		})
	}
}

// indexRehashHash is a cheap spread of n across the 64-bit hash space. The table's probe
// behaviour depends on the hashes being spread, not on them coming from any particular
// key bytes, so the step benchmark mints them directly rather than hashing keys.
func indexRehashHash(n uint64) uint64 {
	h := n * 0x9e3779b97f4a7c15
	h ^= h >> 29
	h *= 0xbf58476d1ce4e5b9
	h ^= h >> 32
	return h
}
