// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"math/rand"
	"sync/atomic"
	"testing"
)

// Benchmarks for Config.InPlaceSameSizeUpdate. Every one of them is an A/B on
// the same shape with the same seed, flag off against flag on, because the
// feature is a TRADE and only a pair of numbers says what it costs: writes stop
// stranding a dead copy apiece, and in exchange reads on the shard give up being
// lock-free.

const (
	inPlaceKeySpan = 20_000   // working set
	inPlacePages   = 8        // 8 MiB per shard; benchValue() gives ~295 B records
	inPlaceKeyFmt  = "k%012d" // 13 bytes
)

// The three arms every benchmark here reports, which are the three states this
// work passes through: today's append path, in-place updates paying the read
// lock for them, and in-place updates validating reads against a version counter
// instead.
type inPlaceMode int

const (
	modeAppend  inPlaceMode = iota // no in-place updates; reads lock-free
	modeLocked                     // in-place updates; reads take the read lock
	modeSeqlock                    // in-place updates; reads lock-free via the seqlock
)

func (m inPlaceMode) String() string {
	switch m {
	case modeAppend:
		return "append"
	case modeLocked:
		return "locked"
	default:
		return "seqlock"
	}
}

var inPlaceModes = []inPlaceMode{modeAppend, modeLocked, modeSeqlock}

// inPlaceShard builds a single heap ringbuf shard in the requested mode, with
// nothing else differing between the arms.
func inPlaceShard(tb testing.TB, mode inPlaceMode) *shard {
	tb.Helper()
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = inPlacePages << 20
	cfg.TTLSweepIntervalMs = 0 // no background sweeper in the measurement
	cfg.InPlaceSameSizeUpdate = mode != modeAppend
	cfg.InPlaceSeqlockReads = mode == modeSeqlock
	s, err := newShard(cfg, "", nil)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = s.Close() })
	return s
}

func inPlaceKey(i int) []byte { return fmt.Appendf(nil, inPlaceKeyFmt, i%inPlaceKeySpan) }

// BenchmarkInPlaceOccupancy is the headline report: what a fixed number of
// same-size rewrites leaves behind, flag off against flag on, on a shard whose
// live set fits its budget with room to spare (20k keys x ~295 B in 8 MiB). It
// is not a throughput measurement — the op count is fixed so both arms do the
// same number of writes — and the numbers that matter are B/livekey (occupancy
// per key the index still holds), keys (how much of the working set survived),
// hit% (a read pass over the whole key span afterwards) and evict/put.
//
// The claim under test is that bytes-per-live-key collapses toward the record
// size, with no relocation or compaction involved at all: a rewrite that lands
// on its predecessor creates nothing to reclaim.
//
// Two access shapes, because they answer different halves of the question:
//
//	cyclic  the writes sweep the whole key span evenly. Every key is rewritten
//	        shortly after it is evicted, so the appending arm loses no hit rate —
//	        the damage is pure wasted work and occupancy, and that is what the
//	        numbers should show.
//	skewed  the writes concentrate on a HOT QUARTER of the span while the read
//	        pass covers all of it. The appending arm's garbage now evicts the cold
//	        keys, which nothing rewrites, so here the cost lands on the hit rate.
func BenchmarkInPlaceOccupancy(b *testing.B) {
	const (
		writes = 2_000_000
		hot    = inPlaceKeySpan / 4
	)
	shapes := []struct {
		name string
		span int // how many of the keys the write stream touches
	}{
		{"cyclic", inPlaceKeySpan},
		{"skewed", hot},
	}
	for _, sh := range shapes {
		for _, mode := range inPlaceModes {
			b.Run(fmt.Sprintf("%s/%s", sh.name, mode), func(b *testing.B) {
				for b.Loop() {
					b.StopTimer()
					s := inPlaceShard(b, mode)
					val := benchValue()
					// Seed the FULL key span, so the skewed shape has cold keys to lose.
					for i := range inPlaceKeySpan {
						_ = s.Put(inPlaceKey(i), val, 0)
					}
					b.StartTimer()
					for i := range writes {
						_ = s.Put(inPlaceKey(i%sh.span), val, 0)
					}
					b.StopTimer()

					// Read the whole key span once and count what is still there.
					hits := 0
					for i := range inPlaceKeySpan {
						if _, err := s.Get(inPlaceKey(i)); err == nil {
							hits++
						}
					}
					st := s.snapshot()
					b.ReportMetric(float64(st.BytesUsed)/float64(max(st.Entries, 1)), "B/livekey")
					b.ReportMetric(float64(st.Entries), "keys")
					b.ReportMetric(float64(hits)/float64(inPlaceKeySpan)*100, "hit%")
					b.ReportMetric(float64(st.Evictions)/float64(max(st.Puts, 1)), "evict/put")
					b.ReportMetric(float64(st.InPlaceUpdates)/float64(max(st.Puts, 1))*100, "%inplace")
					b.StartTimer()
				}
			})
		}
	}
}

// BenchmarkInPlaceWrite measures write throughput alone, at steady state on a
// shard already at capacity. The in-place path does strictly less work than the
// append path — no page search, no slot upsert, no eviction — so it should be
// the faster of the two.
func BenchmarkInPlaceWrite(b *testing.B) {
	for _, mode := range inPlaceModes {
		b.Run(mode.String(), func(b *testing.B) {
			s := inPlaceShard(b, mode)
			val := benchValue()
			// Warm until the shard is at capacity and turning over, so the timed
			// window is the steady state rather than the fill.
			for i := 0; i < 1<<22 && (s.evictions.Load() == 0 || i < inPlaceKeySpan*4); i++ {
				_ = s.Put(inPlaceKey(i), val, 0)
			}
			before := s.snapshot()
			b.ReportAllocs()
			b.ResetTimer()
			i := 0
			for b.Loop() {
				_ = s.Put(inPlaceKey(i), val, 0)
				i++
			}
			b.StopTimer()
			after := s.snapshot()
			puts := after.Puts - before.Puts
			b.ReportMetric(float64(after.InPlaceUpdates-before.InPlaceUpdates)/float64(max(puts, 1))*100, "%inplace")
		})
	}
}

// BenchmarkInPlaceRead prices the reader-side cost — the thing this feature asks
// the shard to pay. Every arm spreads readers over all available cores against a
// SINGLE shard, so the read lock is genuinely contended.
//
// Three shapes, narrow to broad:
//
//	readonly   no writers at all, both arms holding identical data. The only
//	           difference is the RLock/RUnlock pair on each get, so this is the
//	           floor: what the lock costs when nothing is competing for it.
//	lockcost   two reads per write, with value lengths JITTERED so a write is
//	           almost never a same-size rewrite and the in-place path stays
//	           essentially idle in both arms. Both arms therefore do the same work
//	           and reach the same occupancy, and the delta is the lock's cost under
//	           a real writer — the price, with nothing subsidising it. Watch the
//	           reported %inplace: if it is not near zero the isolation has failed
//	           and the comparison is not one.
//	samesize   two reads per write at a constant record size — the real workload.
//	           Here the flag-on arm also stops evicting, so the delta is the NET
//	           effect: the lock's cost minus what not thrashing the pages returns.
//
// Each parallel goroutine seeds its own RNG from a distinct counter. A shared
// seed makes every goroutine replay the same (key, length) sequence, which turns
// the jittered shape into a stream of same-size rewrites and silently destroys
// the isolation the lockcost arm exists for.
func BenchmarkInPlaceRead(b *testing.B) {
	shapes := []struct {
		name     string
		readOnly bool
		value    func(r *rand.Rand, base []byte) []byte
	}{
		{"readonly", true, func(_ *rand.Rand, base []byte) []byte { return base }},
		{"lockcost", false, func(r *rand.Rand, base []byte) []byte { return base[:len(base)-r.Intn(16)] }},
		{"samesize", false, func(_ *rand.Rand, base []byte) []byte { return base }},
	}
	for _, sh := range shapes {
		for _, mode := range inPlaceModes {
			b.Run(fmt.Sprintf("%s/%s", sh.name, mode), func(b *testing.B) {
				s := inPlaceShard(b, mode)
				base := benchValue()
				warm := rand.New(rand.NewSource(1)) //nolint:gosec // deterministic shape, not security
				if sh.readOnly {
					// Just the live set, well inside the budget: no eviction, no garbage,
					// so both arms hold byte-identical data and only the lock differs.
					for i := range inPlaceKeySpan {
						_ = s.Put(inPlaceKey(i), base, 0)
					}
				} else {
					for i := 0; i < 1<<22 && (s.evictions.Load() == 0 || i < inPlaceKeySpan*4); i++ {
						_ = s.Put(inPlaceKey(i), sh.value(warm, base), 0)
					}
				}
				before := s.snapshot()
				var seed atomic.Int64
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					r := rand.New(rand.NewSource(seed.Add(1))) //nolint:gosec // deterministic per-goroutine shape, not security
					n := 0
					for pb.Next() {
						if !sh.readOnly && n%3 == 2 { // two reads per write
							_ = s.Put(inPlaceKey(r.Intn(inPlaceKeySpan)), sh.value(r, base), 0)
						} else {
							_, _ = s.Get(inPlaceKey(r.Intn(inPlaceKeySpan)))
						}
						n++
					}
				})
				b.StopTimer()
				after := s.snapshot()
				gets := after.Gets - before.Gets
				puts := after.Puts - before.Puts
				b.ReportMetric(float64(after.Hits-before.Hits)/float64(max(gets, 1))*100, "hit%")
				b.ReportMetric(float64(after.InPlaceUpdates-before.InPlaceUpdates)/float64(max(puts, 1))*100, "%inplace")
				b.ReportMetric(float64(after.SeqlockRetries-before.SeqlockRetries)/float64(max(gets, 1))*100, "%retry")
				b.ReportMetric(float64(after.SeqlockFallbacks-before.SeqlockFallbacks)/float64(max(gets, 1))*100, "%fallback")
			})
		}
	}
}
