// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"math"
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

	// The lockcost arm's length jitter. Lengths run over
	// [inPlaceJitterLo, inPlaceJitterLo+inPlaceJitterSpan), centred on benchValSz
	// so the mean record size matches the fixed shapes'. The SPAN is what matters:
	// two successive writes of a key collide on a length with probability 1/span,
	// and every collision leaks a same-size rewrite into an arm whose entire
	// purpose is to have none.
	inPlaceJitterLo   = benchValSz / 2 // 128
	inPlaceJitterSpan = benchValSz + 1 // 257 lengths, 128..384, mean 256
)

// The arms every benchmark here reports. They are the states this work passes
// through, plus relocating eviction, which is the other way a ringbuf shard keeps
// live records it would otherwise drop and so the baseline in-place has to beat:
// today's append path, that path with relocating eviction, in-place updates
// paying the read lock for them, in-place updates validating reads against a
// version counter instead, and that last one with relocation as well.
//
// The last two arms cross the SIEVE reference hint (Config.SieveVisitedBit,
// cache/sieve.go) over the relocation axis, because the hint only changes what a
// RELOCATION rescues: with relocation off it does nothing but pay its read-path
// cost, which is exactly why that cell is worth measuring rather than assuming.
// Taking the four in-place cells together — {sieve off, sieve on} x {reloc off,
// reloc on} — is what says whether the hint recovers relocation's recency loss and
// what it charges the read path for it.
//
// HOW TO READ THE SIEVE ARMS IN BenchmarkInPlaceRead, which is where the cost side
// is priced. Setting the hint is a read-modify-write, which is the scaling hazard
// this whole line of work is about: the read lock is a read-modify-write on ONE line
// every reader shares, and that is what makes it unaffordable. Two things may make
// the hint behave differently, and neither is a reason to assume it does — the word
// is per-SLOT rather than one word for the shard, and a load-test-then-store means
// only the first read after a drain cleared a mark performs a store at all.
//
// READ THE PAIRS, NOT THE COLUMN, and prefer the pairs with a WRITER PRESENT. A
// read-only shard makes the hint look free twice over: nothing clears a mark, so
// every slot is marked once and every later read is a load and a compare, and there
// is no writer to contend with either. inplace+sieve against seqlock is therefore
// the floor, not the price. inplace+reloc+sieve against inplace+reloc is the honest
// pair: relocation clears marks continuously, so reads keep paying the store, and
// the lockcost and samesize shapes put a writer alongside them.
type inPlaceMode int

const (
	modeAppend            inPlaceMode = iota // no in-place updates; reads lock-free
	modeAppendReloc                          // append path + relocating eviction
	modeLocked                               // in-place updates; reads take the read lock
	modeSeqlock                              // in-place updates; reads lock-free via the seqlock
	modeSeqlockReloc                         // in-place updates + seqlock + relocating eviction
	modeSeqlockSieve                         // in-place + seqlock + the SIEVE hint, no relocation
	modeSeqlockRelocSieve                    // in-place + seqlock + relocation choosing by the hint
)

func (m inPlaceMode) String() string {
	switch m {
	case modeAppend:
		return "append"
	case modeAppendReloc:
		return "append+reloc"
	case modeSeqlockReloc:
		return "inplace+reloc"
	case modeSeqlockSieve:
		return "inplace+sieve"
	case modeSeqlockRelocSieve:
		return "inplace+reloc+sieve"
	case modeLocked:
		return "locked"
	default:
		return "seqlock"
	}
}

var inPlaceModes = []inPlaceMode{
	modeAppend, modeAppendReloc, modeLocked,
	modeSeqlock, modeSeqlockReloc, modeSeqlockSieve, modeSeqlockRelocSieve,
}

// inPlaceShard builds a single heap ringbuf shard in the requested mode, with
// nothing else differing between the arms.
func inPlaceShard(tb testing.TB, mode inPlaceMode) *shard {
	tb.Helper()
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = inPlacePages << 20
	cfg.TTLSweepIntervalMs = 0 // no background sweeper in the measurement
	cfg.InPlaceSameSizeUpdate = mode != modeAppend && mode != modeAppendReloc
	cfg.InPlaceSeqlockReads = mode != modeAppend && mode != modeAppendReloc && mode != modeLocked
	// The SIEVE hint (cache/sieve.go). It is crossed with relocation rather than
	// bundled into it: with relocation off nothing consumes the hint, so that arm
	// prices the READ-PATH cost of maintaining it with none of the retention it can
	// return — which is the honest way to see what it charges.
	cfg.SieveVisitedBit = mode == modeSeqlockSieve || mode == modeSeqlockRelocSieve
	// Relocating eviction is the other way a ringbuf shard keeps live records it
	// would otherwise drop, so it is the baseline in-place has to beat rather than
	// an unrelated question. Its background reserve ticker stays off: these
	// benchmarks drive every pass through Put, and a tick landing mid-measurement
	// would only add variance.
	cfg.RelocatingEviction = mode == modeAppendReloc || mode == modeSeqlockReloc || mode == modeSeqlockRelocSieve
	cfg.RelocateReserveIntervalMs = 0
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
					b.ReportMetric(float64(st.EvictionRelocations), "relocs")
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
//	           essentially idle in every arm. They therefore do the same work and
//	           reach the same occupancy, and the delta is the lock's cost under a
//	           real writer — the price, with nothing subsidising it. Watch the
//	           reported %inplace: if it is not near zero the isolation has failed
//	           and the comparison is not one.
//
//	           The jitter spans inPlaceJitterSpan lengths, not a handful. Two
//	           successive writes of one key collide on a length with probability
//	           1/span, and that collision rate IS the leaked in-place rate — a
//	           sixteen-length jitter leaks 6%, which fails the arm's own guard and
//	           quietly makes it price something other than the lock. The span is
//	           centred so the MEAN record size matches the fixed one the samesize
//	           arm writes, which keeps the two shapes comparable.
//	samesize   two reads per write at a constant record size — the real workload.
//	           Here the flag-on arm also stops evicting, so the delta is the NET
//	           effect: the lock's cost minus what not thrashing the pages returns.
//	decoupled  two reads per write at a constant record size, like samesize, but
//	           with the two key distributions PULLED APART: writes concentrate on
//	           a hot subset of the span while reads stay uniform over all of it.
//	           hot100 draws writes from the whole span, which is samesize again —
//	           the matched control is the degenerate end of the same sweep, so
//	           control and treatment sit on one axis.
//	churn      the decoupled shape with a share of writes inserting BRAND-NEW keys,
//	           which is the only thing that keeps the log rotating once in-place
//	           updates have stopped producing garbage. See inPlaceChurnPct: without
//	           it the page generations freeze and every residency figure below is
//	           an artifact of the order the shard was filled in.
//
// WHY THE DECOUPLED SHAPE EXISTS. Every other shape here draws reads and writes
// from the SAME distribution. That makes "the share of rewrites that hit a recent
// record" and "the share of reads that resolve into a recent record" the same
// underlying quantity — P(this key was written within the last R bytes of write
// traffic) — so they are identically equal by construction and no arrangement of
// them can show one being small while the other is large. Only a write
// distribution strictly more concentrated than the read distribution can separate
// them, and that separation is the only thing a region confined to the newest
// pages could ever be paid out of. The sweep parameter is how far apart they are
// pulled.
//
// The shape is not hypothetical. An operate call (ops/operate.go) performs
// exactly ONE Put of a complete replacement per call, and a record whose fields
// are fixed-width is overwritten cell-by-cell inside a buffer of unchanged
// length — so the replacement is byte-identical in size to what it replaces,
// which is precisely the same-size rewrite inPlaceTargetLocked looks for. Such a
// record is rewritten at constant size, repeatedly, on whatever narrow key set
// the application's counters live on, while being read on an independent
// schedule.
//
// WHAT THE RESIDENCY SWEEP SHOWED, medians of three repeats. Across every
// churning shape — writes confined to the whole key span, then to a half, a
// quarter, a tenth, a twentieth and a hundredth of it, reads uniform over all of
// it throughout — the read share and the write share move together exactly:
//
//	write share    fr1/fr8   fw1/fw8   fr4/fr8   fw4/fw8
//	1.00           0.069     0.070     0.468     0.468
//	0.50           0.073     0.073     0.466     0.467
//	0.25           0.072     0.073     0.463     0.461
//	0.10           0.068     0.068     0.472     0.474
//	0.05           0.070     0.071     0.479     0.478
//	0.01           0.108     0.110     0.501     0.504
//
// Shrinking the region hands back read protection and takes away rewrites in the
// same proportion, at every K and at every degree of decoupling, so the region is
// a DIAL rather than a win. The cause is that THE CACHE RE-MATCHES the two
// distributions this shape pulls apart: a key the write stream never touches is
// evicted, so what stays resident is what the writes touch, and a read that
// RESOLVES is therefore drawn from the same set the writes are. Equivalently,
// fr(K)/hit% equals fw(K)/%inplace to three digits at every point in the table.
//
// The raw shares DO differ, and only one thing separates them: the miss rate. At
// a hundredth of the span fr4 is 2.3% of gets against fw4 at 35.0% of puts — but
// hit% there is 4.9%. f_r is the residency ratio times the hit rate, so it drops
// below any interesting threshold only on a cache that is missing almost
// everything it is asked for.
//
// The NON-churning family is worse than a dial, and shows the mechanism. With no
// new bytes arriving the log stops rotating, and a hot record rewritten in place
// never returns to a newest page — in-place update is precisely the thing that
// stops refreshing a record's recency, which is what BenchmarkInPlaceWriteRecency
// measures from the other side. At a quarter of the span and below, every rewrite
// target sits on the OLDEST page: fw1, fw2 and fw4 are all 0 while fr4 is 64%. A
// region would there admit two thirds of the reads to the lock and not one single
// rewrite.
//
// Each parallel goroutine seeds its own RNG from a distinct counter. A shared
// seed makes every goroutine replay the same (key, length) sequence, which turns
// the jittered shape into a stream of same-size rewrites and silently destroys
// the isolation the lockcost arm exists for.
func BenchmarkInPlaceRead(b *testing.B) {
	fixed := func(_ *rand.Rand, base []byte) []byte { return base }
	shapes := []inPlaceReadShape{
		{name: "readonly", readOnly: true, value: fixed},
		{name: "lockcost", value: func(r *rand.Rand, base []byte) []byte {
			return base[:inPlaceJitterLo+r.Intn(inPlaceJitterSpan)]
		}},
		{name: "samesize", value: fixed},
	}
	for _, pct := range inPlaceHotPercents {
		hot := max(inPlaceKeySpan*pct/100, 1)
		shapes = append(shapes,
			inPlaceReadShape{name: fmt.Sprintf("decoupled_hot%03d", pct), hotKeys: hot, value: fixed},
			inPlaceReadShape{name: fmt.Sprintf("churn_hot%03d", pct), hotKeys: hot, churnPct: inPlaceChurnPct, value: fixed},
		)
	}
	for _, sh := range shapes {
		for _, mode := range inPlaceModes {
			b.Run(fmt.Sprintf("%s/%s", sh.name, mode), func(b *testing.B) {
				s := inPlaceShard(b, mode)
				// The jittered shape needs a base long enough to slice its whole range
				// out of; the fixed shapes keep the standard record size.
				base := benchValue()
				if !sh.readOnly && sh.name == "lockcost" {
					base = make([]byte, inPlaceJitterLo+inPlaceJitterSpan)
					for i := range base {
						base[i] = byte(i)
					}
				}
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
				// A decoupled or churning shape needs a SECOND warm phase under its own
				// write distribution. The seed above sweeps the whole span, so without this
				// the timed window would open on a shard whose page ages still reflect
				// uniform writes and would spend part of itself migrating to the steady
				// state the residency figures are meant to describe. Long enough to rotate
				// the ring several times over. It runs only for the new shapes, so every
				// pre-existing arm warms exactly as it did.
				var churn atomic.Int64
				if sh.hotKeys != 0 {
					for range inPlaceKeySpan * 8 {
						sh.put(s, warm, base, &churn)
					}
				}
				detach := regionFRAttach(s)
				defer detach()
				before := s.snapshot()
				frBefore := regionFRSnapshot()
				var seed atomic.Int64
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					r := rand.New(rand.NewSource(seed.Add(1))) //nolint:gosec // deterministic per-goroutine shape, not security
					n := 0
					for pb.Next() {
						if !sh.readOnly && n%3 == 2 { // two reads per write
							sh.put(s, r, base, &churn)
						} else {
							_, _ = s.Get(inPlaceKey(r.Intn(inPlaceKeySpan)))
						}
						n++
					}
				})
				b.StopTimer()
				after := s.snapshot()
				frAfter := regionFRSnapshot()
				gets := after.Gets - before.Gets
				puts := after.Puts - before.Puts
				b.ReportMetric(float64(after.Hits-before.Hits)/float64(max(gets, 1))*100, "hit%")
				b.ReportMetric(float64(after.InPlaceUpdates-before.InPlaceUpdates)/float64(max(puts, 1))*100, "%inplace")
				b.ReportMetric(float64(after.SeqlockRetries-before.SeqlockRetries)/float64(max(gets, 1))*100, "%retry")
				b.ReportMetric(float64(after.SeqlockFallbacks-before.SeqlockFallbacks)/float64(max(gets, 1))*100, "%fallback")
				reportRegionResidency(b, frBefore, frAfter, gets, puts)
			})
		}
	}
}

// BenchmarkInPlaceWriteRecency measures what an in-place rewrite gives up:
// WRITE RECENCY. An appending rewrite moves its key to the newest page, so a key
// touched often drifts ahead of the eviction rotation and survives. Rewriting a
// key where it lies leaves it on whatever page it was first written to, so the
// rotation reaches it on schedule however hot it is.
//
// That only bites when eviction is actually running, so this shard is
// deliberately sized SMALLER than its live set — the one regime where in-place
// cannot simply avoid evicting. Writes interleave a small hot set, rewritten over
// and over at a constant size, with a stream of cold keys each written ONCE; the
// number that matters is how much of the HOT set is still resident afterwards.
//
// The cold keys being distinct is load-bearing, not incidental. If they repeat,
// they are rewrites too, and the benchmark stops contrasting "rewritten often"
// against "written once" — which is the only contrast it exists to draw.
//
// Reported per arm: hot% (the hot set'"'"'s survival, the figure under test), the
// index'"'"'s total occupancy, evictions per write, and the share of writes that took
// the in-place path.
//
// THE FOUR IN-PLACE ARMS ARE A 2x2 and should be read as one. Relocating eviction
// made hot% WORSE here, not better — it rescues by position, and an in-place rewrite
// leaves no positional trace of itself, so the budget goes on whatever the walk meets
// first. The SIEVE hint (Config.SieveVisitedBit) is the answer to exactly that, and
// the cell that decides whether it works is inplace+reloc+sieve against inplace+reloc.
// The inplace+sieve cell is the control: nothing consumes the hint there, so it should
// land on inplace, and a difference would mean the hint is perturbing something it
// has no business touching.
func BenchmarkInPlaceWriteRecency(b *testing.B) {
	const (
		hotKeys   = 2_000
		writes    = 1_500_000
		hotEveryN = 4 // one hot rewrite per three cold inserts
	)
	hotKey := func(i int) []byte { return fmt.Appendf(nil, "h%011d", i%hotKeys) }
	// EVERY COLD KEY IS DISTINCT — no modulus here. Wrapping them turns the cold
	// stream into a rewrite stream, which is the opposite of the contrast this
	// benchmark is built on: the hot set is meant to be the only thing rewritten,
	// so that what survives measures recency and nothing else.
	coldKey := func(i int) []byte { return fmt.Appendf(nil, "c%011d", i) }

	for _, mode := range inPlaceModes {
		b.Run(mode.String(), func(b *testing.B) {
			for b.Loop() {
				b.StopTimer()
				s := inPlaceShard(b, mode)
				val := benchValue()
				b.StartTimer()
				cold := 0
				for i := range writes {
					if i%hotEveryN == 0 {
						_ = s.Put(hotKey(i/hotEveryN), val, 0)
					} else {
						_ = s.Put(coldKey(cold), val, 0)
						cold++
					}
				}
				b.StopTimer()

				hot := 0
				for i := range hotKeys {
					if _, err := s.Get(hotKey(i)); err == nil {
						hot++
					}
				}
				st := s.snapshot()
				b.ReportMetric(float64(hot)/float64(hotKeys)*100, "hot%")
				b.ReportMetric(float64(st.Entries), "keys")
				b.ReportMetric(float64(st.Evictions)/float64(max(st.Puts, 1)), "evict/put")
				b.ReportMetric(float64(st.InPlaceUpdates)/float64(max(st.Puts, 1))*100, "%inplace")
				b.StartTimer()
			}
		})
	}
}

// inPlaceHotPercents is the decoupled shape's sweep: what percentage of the key
// span the WRITE stream is confined to, while reads stay uniform over all of it.
// 100 is the matched-distribution control — the same shape as samesize — so the
// control is the degenerate end of the treatment rather than a separate
// experiment, and the whole sweep reads as one curve.
var inPlaceHotPercents = []int{100, 50, 25, 10, 5, 1}

// inPlaceChurnPct is the share of a churning shape's writes that insert a
// BRAND-NEW key instead of rewriting a hot one.
//
// It exists because without it the residency question has no subject. A shard
// whose live set fits its budget, rewriting in place, produces NO garbage — so it
// never evicts, never retires a page, and its page generations stop moving
// altogether. "The newest K pages" then names the pages the initial fill happened
// to end on, and every residency share it yields is an artifact of seeding order
// rather than of write recency. A mutable region is a claim about a log that TURNS
// OVER, and only a supply of new bytes makes one turn over. The churn keys are
// outside the read span, exactly like the cold stream in
// BenchmarkInPlaceWriteRecency, so they drive the rotation without changing what
// the reads are drawn from.
const inPlaceChurnPct = 25

// inPlaceChurnKey formats a churn key at the same 13 bytes as inPlaceKey, so a
// churning shape's records are the same size as everything else on the shard and
// page occupancy stays comparable across shapes.
func inPlaceChurnKey(n int64) []byte { return fmt.Appendf(nil, "n%012d", n) }

// inPlaceReadShape is one access shape of BenchmarkInPlaceRead: what the reads
// and the writes are each drawn from, and how big a record each write stores.
// Reads are ALWAYS uniform over the whole key span; only the write side varies.
type inPlaceReadShape struct {
	name     string
	readOnly bool
	// hotKeys confines writes to [0,hotKeys) of the span; 0 means the whole span,
	// which is the matched-distribution case.
	hotKeys int
	// churnPct is the share of writes that insert a fresh key instead, keeping the
	// log rotating. 0 for every shape that predates the region question.
	churnPct int
	value    func(r *rand.Rand, base []byte) []byte
}

// put performs one write of the shape. The churn draw is taken only when the
// shape has churn, so a shape without it consumes exactly the RNG sequence it
// consumed before this method existed.
func (sh inPlaceReadShape) put(s *shard, r *rand.Rand, base []byte, churn *atomic.Int64) {
	if sh.churnPct > 0 && r.Intn(100) < sh.churnPct {
		_ = s.Put(inPlaceChurnKey(churn.Add(1)), sh.value(r, base), 0)
		return
	}
	span := sh.hotKeys
	if span == 0 {
		span = inPlaceKeySpan
	}
	_ = s.Put(inPlaceKey(r.Intn(span)), sh.value(r, base), 0)
}

// inPlaceRegionKs is the region sizes, in pages, the residency shares are
// reported at. A shard here holds inPlacePages pages, so K = inPlacePages is the
// whole log and exists as the instrument's own check: at that K the read share
// must equal hit% and the write share must equal %inplace, because a region
// covering everything excludes nothing. If it does not, the classification is
// wrong and the smaller K are not worth reading.
var inPlaceRegionKs = []int{1, 2, 4, inPlacePages}

// reportRegionResidency reports, for each region size K, the share of GETS that
// resolved into the newest K pages and the share of PUTS whose same-size rewrite
// landed there. They are the two halves of the question a mutable region asks:
// %frK is what the region would make pay, %fwK is what it would pay out.
//
// Shares of gets and of puts rather than of hits and of in-place updates,
// because the decision is about the fraction of the offered LOAD that changes
// cost. hit% is reported alongside for anyone wanting the per-hit figure.
//
// Silent unless the measurement build tag is set — see cache/regionfr_off.go —
// since the counters are zero otherwise and reporting them would report a zero
// as a result.
func reportRegionResidency(b *testing.B, before, after regionFRCounts, gets, puts uint64) {
	b.Helper()
	if !regionFRInstrumented {
		return
	}
	var rcum, wcum uint64
	for k := 1; k <= regionFRMaxK; k++ {
		rcum += after.HitDist[k-1] - before.HitDist[k-1]
		wcum += after.PlaceDist[k-1] - before.PlaceDist[k-1]
		for _, want := range inPlaceRegionKs {
			if k != want {
				continue
			}
			b.ReportMetric(float64(rcum)/float64(max(gets, 1))*100, fmt.Sprintf("%%fr%d", k))
			b.ReportMetric(float64(wcum)/float64(max(puts, 1))*100, fmt.Sprintf("%%fw%d", k))
		}
	}
}

// inPlaceReadLockPercents is the sweep for BenchmarkReadLockFraction: the share
// of reads routed through the shard read lock.
var inPlaceReadLockPercents = []int{0, 5, 10, 20, 30, 50, 75, 100}

// benchRand is a xorshift64 generator for the per-read coin flip. It is here
// rather than math/rand because the flip is paid on EVERY read of every arm in
// the sweep, and a generator costing tens of nanoseconds would be a larger term
// than the thing being measured. Three shifts and an xor is small enough to be a
// constant the whole curve carries equally.
type benchRand uint64

func (x *benchRand) next() uint64 {
	v := uint64(*x)
	v ^= v << 13
	v ^= v >> 7
	v ^= v << 17
	*x = benchRand(v)
	return v
}

// BenchmarkReadLockFraction prices the shard read lock as a function of HOW MANY
// reads take it. It exists because the obvious way to price a design that locks
// only some reads — take the all-reads cost and scale it linearly — has no basis.
// An RLock is a read-modify-write on one cache line every reader shares (see
// cache/seqlock.go), so its cost is superlinear in the number of readers
// contending, and a cost that is superlinear going up falls away FASTER than
// linearly coming down. Where the curve actually crosses the lock-free arms is a
// measurement, not an extrapolation, and it is the number any "only a fraction of
// reads pay" design has to be held against.
//
// DELIBERATELY NOT A REGION. The lock is taken on a per-read COIN FLIP at a
// configured rate, independent of where the key lives, and in-place updates are
// OFF — so the locked and lock-free paths return identical answers and the only
// difference between them is the locking. This isolates the lock's cost from
// every question about which reads a real design would route through it.
//
// THE ENDPOINTS ARE THE INSTRUMENT'S OWN CHECK, and nothing in the middle is
// worth reading until they pass. frac000 must reproduce ref_append and frac100
// must reproduce ref_locked, because at those two rates the swept arm executes
// exactly the read path the corresponding mode's shard would select — plus one
// xorshift, which every point on the curve pays alike. The references are
// measured here, in the same run and through the same loop, rather than quoted
// from BenchmarkInPlaceRead, so the comparison carries no cross-run drift.
//
// The two shapes are readonly (the floor: no writer, so the lock is uncontended
// by anything but other readers) and lockcost (a real writer, with value lengths
// jittered so the in-place path stays idle even on the ref_locked shard — which
// is what makes that reference a pure read-path difference rather than a write
// path one; watch its %inplace).
//
// WHAT THE SWEEP SHOWED, as the share of the full-lock increment over the
// lock-free floor that a given locked share actually costs. Medians of nine
// repeats:
//
//	locked share   no writer   with a writer
//	0.05           0.06        0.59
//	0.10           0.08        0.59
//	0.20           0.15        0.90
//	0.30           0.25        0.74
//	0.50           0.45        1.14
//	0.75           0.69        1.05
//
// With no writer the cost is mildly SUBLINEAR — locking a fifth of reads costs
// about three quarters of what scaling the all-reads figure linearly would
// predict. Real, but small, and nowhere near enough on its own to move a
// crossover by much.
//
// With a writer it inverts, and far more violently: a twentieth of reads taking
// the lock already costs three fifths of what locking every read costs, and past
// a fifth the curve is flat inside its own noise (which is why that column is not
// monotone — the writer shape's arms span up to 1.9x across repeats, so only its
// ordering and its medians are claimed, not its individual points). A waiting
// writer blocks new readers outright, so every locked read pays a queueing
// penalty that being rare does not dilute.
//
// The crossover against the seqlock arm therefore sits at a locked share of
// roughly 0.02 to 0.05 in BOTH shapes, because the seqlock arm sits essentially
// on the lock-free floor and there is very little room beneath it to trade into.
//
// ENDPOINT REPRODUCTION. In the no-writer shape frac000 lands within about 2% of
// ref_append and frac100 within about 2% of ref_locked, inside both arms' own
// spread. In the writer shape frac000 matches ref_append and frac100 runs about
// 8% UNDER ref_locked — expected rather than a failure, because ref_locked's
// shard has in-place updates enabled and so runs the same-size eligibility probe
// on every write, work the swept shard does not do. That shape's endpoint check
// is therefore approximate, and the no-writer shape is the one the instrument
// rests on.
func BenchmarkReadLockFraction(b *testing.B) {
	shapes := []struct {
		name     string
		readOnly bool
		value    func(r *rand.Rand, base []byte) []byte
	}{
		{"readonly", true, func(_ *rand.Rand, base []byte) []byte { return base }},
		{"lockcost", false, func(r *rand.Rand, base []byte) []byte {
			return base[:inPlaceJitterLo+r.Intn(inPlaceJitterSpan)]
		}},
	}
	for _, sh := range shapes {
		for _, mode := range []inPlaceMode{modeAppend, modeLocked, modeSeqlock} {
			b.Run(fmt.Sprintf("%s/ref_%s", sh.name, mode), func(b *testing.B) {
				readLockArm(b, mode, -1, sh.readOnly, sh.value)
			})
		}
		for _, pct := range inPlaceReadLockPercents {
			b.Run(fmt.Sprintf("%s/frac%03d", sh.name, pct), func(b *testing.B) {
				// modeAppend: in-place OFF, so the shard's own read path is the lock-free
				// one and every locked read in this arm is there because the coin said so.
				readLockArm(b, modeAppend, pct, sh.readOnly, sh.value)
			})
		}
	}
}

// readLockArm runs one arm of BenchmarkReadLockFraction. lockPct < 0 is a
// REFERENCE arm: no coin, every read goes through s.Get and therefore through
// whatever path the shard's own configuration selects. Otherwise every read
// flips, and a win routes it through getLockedCore — which, with the gets counter
// bumped first, is exactly what s.Get does on a shard that needs the read lock.
func readLockArm(b *testing.B, mode inPlaceMode, lockPct int, readOnly bool, value func(r *rand.Rand, base []byte) []byte) {
	s := inPlaceShard(b, mode)
	base := benchValue()
	if !readOnly {
		base = make([]byte, inPlaceJitterLo+inPlaceJitterSpan)
		for i := range base {
			base[i] = byte(i)
		}
	}
	warm := rand.New(rand.NewSource(1)) //nolint:gosec // deterministic shape, not security
	if readOnly {
		for i := range inPlaceKeySpan {
			_ = s.Put(inPlaceKey(i), base, 0)
		}
	} else {
		for i := 0; i < 1<<22 && (s.evictions.Load() == 0 || i < inPlaceKeySpan*4); i++ {
			_ = s.Put(inPlaceKey(i), value(warm, base), 0)
		}
	}
	// The coin's threshold over the full uint64 range, so the endpoints are EXACT
	// rather than approached: 100 sits above every draw and 0 below every draw. The
	// 100 case is spelled out because computing it as a fraction of 2^64 overflows
	// to zero and silently turns the arm that must lock every read into the arm
	// that locks none — which is a broken instrument that still produces a
	// plausible-looking curve. A reference arm (lockPct < 0) leaves it at zero and
	// so runs the shard's own read path, whatever that shard's mode selects.
	var thresh uint64
	switch {
	case lockPct >= 100:
		thresh = math.MaxUint64
	case lockPct > 0:
		thresh = (uint64(lockPct) << 32) / 100 << 32 // pct/100 of 2^64, in two steps
	}
	before := s.snapshot()
	var seed atomic.Int64
	var lockedReads, totalReads atomic.Uint64
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		r := rand.New(rand.NewSource(seed.Add(1)))                 //nolint:gosec // deterministic per-goroutine shape, not security
		x := benchRand(uint64(seed.Load())*0x9e3779b97f4a7c15 + 1) //nolint:gosec // a nonzero xorshift seed, not security
		var locked, reads uint64
		n := 0
		for pb.Next() {
			if !readOnly && n%3 == 2 { // two reads per write, as in BenchmarkInPlaceRead
				_ = s.Put(inPlaceKey(r.Intn(inPlaceKeySpan)), value(r, base), 0)
				n++
				continue
			}
			key := inPlaceKey(r.Intn(inPlaceKeySpan))
			reads++
			// The coin is flipped on EVERY arm, reference arms included, even though a
			// reference's threshold can never be met. The flip is a constant the whole
			// curve carries, and a reference that skipped it would differ from frac000
			// by the instrument rather than by the thing being compared — which is the
			// one difference the endpoint check cannot tolerate, since that check is
			// the only evidence the middle of the curve means anything.
			if x.next() < thresh {
				// What getCore does on a shard that needs the read lock: count the get,
				// then probe and copy under it.
				s.gets.Add(1)
				_, _, _ = s.getLockedCore(key, hashKey(key), s.now(), !s.cfg.Replicated)
				locked++
			} else {
				_, _ = s.Get(key)
			}
			n++
		}
		lockedReads.Add(locked)
		totalReads.Add(reads)
	})
	b.StopTimer()
	after := s.snapshot()
	puts := after.Puts - before.Puts
	b.ReportMetric(float64(lockedReads.Load())/float64(max(totalReads.Load(), 1))*100, "%locked")
	b.ReportMetric(float64(after.InPlaceUpdates-before.InPlaceUpdates)/float64(max(puts, 1))*100, "%inplace")
}
