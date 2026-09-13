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
// WHAT THIS MEASURED, so the next person does not have to rediscover it. The reserve does
// take the copying off the write path, and what that is worth depends on the shard count
// and on the cadence, in that order:
//
//   - At the SLOW cadence and a moderate shard count the reserve wins, modestly. Eight
//     shards: background-slow is a few per cent faster per write than sync with its
//     distribution barely overlapping sync's, at equal or better retention, with about a
//     third of the relocation moved off the write path.
//   - At the SLOW cadence and a high shard count the two become indistinguishable. Sixty-
//     four shards: background-slow's median sits below sync's, but sync's spread is wide
//     enough to swallow the gap — and this is where the reserve moves the MOST work (about
//     two thirds), because each shard sees a smaller share of the write stream and its
//     sweeper keeps up easily. Moving more of the copying did not make the writes faster,
//     which says the sweeper's copy is not cheaper per byte than the write path's: it pays
//     for extra lock acquisitions and the re-validation after each one.
//   - At the FAST cadence the reserve loses badly, and the swept control shows most of that
//     is not the reserve at all: at sixty-four shards a sweeper ticking every millisecond
//     costs over twice the no-sweeper arm before relocation is enabled. At EIGHT shards
//     background-fast is over three times sync. At sixty-four it was ABANDONED rather than
//     measured: it had not finished a single attempt after more than an hour of wall clock,
//     which is a number of a kind — the arm is not slow, it is unusable — and the swept-fast
//     row at the same shard count already prices where that comes from.
//
// THE CADENCE THAT WINS IS NOT THE ONE ANYTHING RUNS AT. Both slow rows above use
// relocABSweepSlowMs; TTLSweepIntervalMs defaults to 1s, roughly twenty times slower, at
// which the reserve takes a fraction of a per cent of the copying and is effectively
// inert. The reserve has no cadence of its own — it rides the TTL sweeper's ticker — and
// that coupling, not the reserve's mechanism, is what bounds what this layer can do.
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

	for _, arm := range []struct {
		name    string
		on      bool
		sweepMs int
	}{
		{"off", false, 0},
		{"swept-fast", false, relocABSweepMs},
		{"swept-slow", false, relocABSweepSlowMs},
		{"sync", true, 0},
		{"background-fast", true, relocABSweepMs},
		{"background-slow", true, relocABSweepSlowMs},
	} {
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
		cfg.TTLSweepIntervalMs = arm.sweepMs
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
