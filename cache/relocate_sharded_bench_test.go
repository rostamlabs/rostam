// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// BenchmarkRelocatingEvictionABSharded is BenchmarkRelocatingEvictionAB's experiment in
// the regime the background free-page reserve is FOR, which the single-shard harness
// cannot reach and says so.
//
// WHAT CHANGES, AND WHY ONLY THIS. One shard written by one goroutine flat out leaves its
// sweeper no idle lock time: every hold the sweeper takes comes straight out of the write
// stream, and the CPU it spends is CPU the writer wanted. So that harness cannot separate
// "the copying moved off the write path" from "a second goroutine is now fighting for the
// only lock there is", and its background row measures mostly the latter. Here the same
// write stream is spread over NumShards shards, each with its own lock and its own
// sweeper, and driven by as many writer goroutines as there are procs — so a sweeper
// competes with roughly 1/NumShards of the traffic rather than all of it.
//
// WHAT IS HELD CONSTANT, so the rows stay comparable with the single-shard table: the
// per-shard geometry (relocABPages 1 MiB pages, PolicyRingbufEvict, no TTLs), the value
// size, the Zipfian exponent, and the per-shard key population — the keyspace and the
// warm-up both scale with the shard count, so each shard sees the same number of distinct
// keys against the same number of pages as it does with one shard. The arms, the columns
// and the seed discipline are the single-shard harness's; only the shard count, the writer
// count and what scales with them differ.
//
// THE SWEEPER CADENCE IS PER SHARD, which matters more here than anywhere and is worth
// stating out loud: every shard runs its own ticker, so at a fixed cadence the background
// work the cache does per unit time scales with the SHARD COUNT while the write stream
// does not. A cadence that is free at one shard is not free at sixty-four. That is exactly
// what the swept control rows are for, and it is why every arm is run at both cadences.
//
// ==========================================================================
// WHAT THIS MEASURED, so the next person does not have to rediscover it. Every figure
// below is the `background` row — the reserve on its OWN ticker at its shipped default
// interval, with the TTL sweeper off — against `sync`, medians of five attempts.
//
//   - EIGHT shards: the reserve wins, modestly and repeatably. Around eight per cent
//     faster per write, at better retention rather than at retention's expense, with
//     roughly a third of the relocation moved off the write path.
//   - SIXTY-FOUR shards: per-write time is a WASH — the two medians land within a per
//     cent of each other, well inside the spread. Retention is not a wash: the reserve
//     holds five to seven points more of the key population at every attempt, and it
//     takes about HALF the write-path copying rather than a third, because each shard
//     sees a smaller share of the write stream and its ticker keeps up more easily.
//     Moving more of the copying did not make the writes faster, which says the
//     sweeper's copy is not cheaper per byte than the write path's: it pays for extra
//     lock acquisitions and the re-validation after each one.
//
// So the honest summary is a single-digit latency win at moderate shard counts, a wash at
// high ones, and a real retention gain at both. It is not a large result.
//
// THE FAST ROWS ARE THE COUNTER-EXAMPLE, and they are kept because they are why the
// reserve has its own interval at all. When it rode the TTL sweeper's ticker, the cadence
// it needed dragged sweepIndex along at the same rate: at sixty-four shards a
// millisecond tick cost over TWICE the no-ticker arm before relocation was even enabled
// (that is the swept-fast row, not the background one), and at eight shards the
// background arm came out over three times sync. At sixty-four shards background-fast was
// ABANDONED rather than measured — it had not finished one attempt after more than an
// hour of wall clock, which is a number of a kind: the arm is not slow, it is unusable.
//
// READING THE COLUMNS ACROSS ATTEMPTS. The cache is warmed once per arm and shared by the
// attempts, so it keeps evolving: hit_rate climbs monotonically down a run in BOTH arms.
// Compare like-for-like attempt indices, or medians across the same number of attempts —
// never the first attempt of one arm against the last of another.
func BenchmarkRelocatingEvictionABSharded(b *testing.B) {
	for _, shards := range relocABShardCounts {
		b.Run(fmt.Sprintf("shards=%d", shards), func(b *testing.B) {
			relocABShardedArms(b, shards)
		})
	}
}

// relocABShardCounts are the shard counts the sharded A/B runs at. Eight is enough to
// give every sweeper slack; sixty-four is where the per-shard sweepers themselves become
// the larger population, which is the number that says whether the design scales or
// merely works.
var relocABShardCounts = []int{8, 64}

// relocABShardedArms runs the four heap arms at one shard count. Same arm list, same
// meanings and same columns as relocABArms — see its doc for what each arm and each
// column is, including why the swept control is not optional.
func relocABShardedArms(b *testing.B, shards int) {
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
		// The cache is built and warmed OUTSIDE b.Run, and that is not a tidiness choice.
		// Go calls a benchmark body repeatedly with a growing b.N until it fills the time
		// budget, so a warm-up written inside the body runs once per attempt — and this
		// warm-up is relocABWarmOps PER SHARD, which at sixty-four shards is tens of
		// millions of writes. Hoisting it means it happens once per arm; the timed
		// attempts then share one cache that stays at its page cap throughout, which is
		// the state being measured anyway. `warm` is taken once and ops accumulates across
		// attempts, so the ratios the final report prints cover the whole arm.
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
		// Warm every shard to its page cap before any timing, fanned out so the fill is
		// not itself a single-writer experiment. Each filler's seed is its index, so the
		// warm key sequence is identical across arms.
		relocABShardedFill(c, keys, val, relocABWarmOps*shards)
		warm := c.Stats()
		var ops atomic.Int64
		var workers atomic.Int64

		b.Run("reloc="+arm.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				rng := rand.New(rand.NewSource(relocABSeed + workers.Add(1))) //nolint:gosec // benchmark RNG
				z := rand.NewZipf(rng, relocABZipfS, 1, uint64(keyspace-1))   //nolint:gosec // keyspace > 1
				n := int64(0)
				for pb.Next() {
					_ = c.Put(keys[z.Uint64()], val, 0)
					n++
				}
				ops.Add(n)
			})
			b.StopTimer()

			st := c.Stats()
			hits := 0
			for _, k := range keys {
				if _, gerr := c.Get(k); gerr == nil {
					hits++
				}
			}
			relocABReport(b, warm, st, ops.Load(), keyspace, hits)
		})
		_ = c.Close()
	}
}

// relocABShardedFill writes total ops across GOMAXPROCS goroutines, each drawing from the
// same Zipfian population with its own deterministically seeded generator. Writes go
// through Cache.Put, so the key hash picks the shard and the load spreads the way the
// timed loop's does.
func relocABShardedFill(c *Cache, keys [][]byte, val []byte, total int) {
	fillers := runtime.GOMAXPROCS(0)
	per := total / fillers
	var wg sync.WaitGroup
	for w := range fillers {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))                        //nolint:gosec // benchmark RNG
			z := rand.NewZipf(rng, relocABZipfS, 1, uint64(len(keys)-1)) //nolint:gosec // len(keys) > 1
			for range per {
				_ = c.Put(keys[z.Uint64()], val, 0)
			}
		}(relocABSeed + int64(w))
	}
	wg.Wait()
}

// relocABArm is one row of the A/B: a relocation setting and the two INDEPENDENT ticker
// intervals, which is what the arm table looked like once the reserve stopped riding the
// TTL sweeper's interval. Zero for either means that ticker does not run.
type relocABArm struct {
	name      string
	on        bool
	ttlMs     int
	reserveMs int
}

// relocABArmList is the arm table both harnesses run, so a sharded row and a single-shard
// row are the same configuration measured in two regimes. See relocABArms for what each
// arm is for and why the controls are not optional.
func relocABArmList() []relocABArm {
	return []relocABArm{
		{name: "off"},
		{name: "swept-fast", ttlMs: relocABSweepMs},
		{name: "swept-default", ttlMs: defaultRelocateReserveIntervalMs},
		{name: "sync", on: true},
		{name: "background-fast", on: true, reserveMs: relocABSweepMs},
		{name: "background", on: true, reserveMs: defaultRelocateReserveIntervalMs},
	}
}

// relocABReport emits the A/B columns. Shared by both harnesses so a sharded row and a
// single-shard row mean the same thing field for field; see relocABArms for what each
// column is.
func relocABReport(b *testing.B, warm, st Stats, ops int64, keyspace, hits int) {
	perOp := func(d uint64) float64 {
		if ops == 0 {
			return 0
		}
		return float64(d) / float64(ops)
	}
	bytesPerKey := 0.0
	if st.Entries > 0 {
		bytesPerKey = float64(st.BytesUsed) / float64(st.Entries)
	}
	b.ReportMetric(float64(hits)/float64(keyspace), "hit_rate")
	b.ReportMetric(float64(st.Entries), "entries")
	b.ReportMetric(bytesPerKey, "B/live_key")
	b.ReportMetric(perOp(st.Evictions-warm.Evictions), "evict/op")
	b.ReportMetric(perOp(st.EvictionsLive-warm.EvictionsLive), "evict_live/op")
	b.ReportMetric(perOp(st.EvictionRelocations-warm.EvictionRelocations), "reloc/op")
	b.ReportMetric(perOp(st.EvictionBytesRelocated-warm.EvictionBytesRelocated), "reloc_B/op")
	b.ReportMetric(perOp(st.ReserveRelocations-warm.ReserveRelocations), "bg_reloc/op")
	b.ReportMetric(float64(st.ReservePagesFreed-warm.ReservePagesFreed), "bg_pages")
}
