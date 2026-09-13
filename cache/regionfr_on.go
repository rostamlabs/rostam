// SPDX-License-Identifier: Apache-2.0

//go:build cacheregionfr

package cache

import "sync/atomic"

// Region residency: how much of the read stream, and how much of the rewrite
// stream, lands in the NEWEST K pages of a shard.
//
// A HybridLog-style mutable region would confine in-place updates to the newest
// pages and seal everything older, so only reads resolving INSIDE the region
// need protection and the rest stay lock-free. Two quantities decide whether
// that is a win or merely a dial:
//
//	f_w  the share of rewrites whose target still lies in the region. Already
//	     reported, as InPlaceUpdates/Puts — but only for a region covering the
//	     WHOLE log. PlaceDist resolves it against K.
//	f_r  the share of reads resolving into the region, which is what the design
//	     asks the shard to pay for. HitDist is that, against K.
//
// If the two track each other, "only a few reads pay" is exactly as true as "the
// in-place path almost never fires" and the region buys nothing. They can only
// come apart when the WRITE key distribution is more concentrated than the READ
// one — which is why both are histogrammed here rather than assumed.
//
// MEASURED, NOT SIMULATED. Nothing here changes behaviour: no write is refused,
// no read is rerouted. The histograms observe the page-age distribution of the
// shard as it already is. A real region would additionally push an out-of-region
// rewrite back onto the append path, which moves that record INTO the region, so
// both measured shares are lower bounds on what a built region would see. They
// are biased the same direction, which is what keeps their comparison honest.
//
// RECENCY IS PAGE GENERATION. s.nextGen() hands every page object a generation
// at construction, monotonically per shard, and heap ringbuf retirement replaces
// a page object rather than reusing it — so the newest generation is the page the
// shard is currently appending into, and newest-minus-gen is the page's age in
// retirements. The subtraction is modular on uint16 and stays correct across the
// counter's wrap, because the live pages of a shard always span far fewer than
// 65536 generations.
//
// THE PAGE MUST BE THE ONE THE PROBE SETTLED ON, never the one the index slot
// first advertised. rechaseSlot can hand back a page at a different index after a
// relocation, and can restart the probe on a fresh table after a rehash; the
// slot's first answer is then about a page that is frozen and cannot be the
// record's home. This is the same lesson indexTable.getSeq already records for
// the seqlock version word, for the same reason. Both call sites here are placed
// on the hit RETURN, after every re-resolution has finished, so the page passed in
// is the settled one by construction.

const regionFRInstrumented = true

const regionFRMaxK = 16

type regionFRCounts struct {
	// Hits is every read that resolved to a record, whatever its page age;
	// InPlace is every write that took the same-size overwrite path. They are the
	// denominators the distributions below are shares of.
	Hits    uint64
	InPlace uint64
	// HitDist[d] counts hits landing on a page d retirements behind the newest;
	// PlaceDist[d] the same for in-place rewrite targets. A record older than
	// regionFRMaxK pages is counted in the total and in neither array, so
	// sum(HitDist[:K])/Hits is the residency share for a region of K pages and
	// never over-counts.
	HitDist   [regionFRMaxK]uint64
	PlaceDist [regionFRMaxK]uint64
}

// regionProbe is the live measurement state for ONE shard. Benchmarks measure a
// single shard at a time (Config.NumShards = 1 throughout inplace_bench_test.go),
// so a single global slot is enough and costs a lock-free pointer load on the hit
// path instead of a per-shard map lookup.
type regionProbe struct {
	s      *shard
	newest atomic.Uint32 // generation of the most recently created page on s
	counts struct {
		hits      atomic.Uint64
		inPlace   atomic.Uint64
		hitDist   [regionFRMaxK]atomic.Uint64
		placeDist [regionFRMaxK]atomic.Uint64
	}
}

var regionActive atomic.Pointer[regionProbe]

// age returns how many page retirements behind the newest generation gen is, or
// -1 when that is further back than the histograms resolve.
func (pr *regionProbe) age(gen uint16) int {
	d := uint16(pr.newest.Load()) - gen //nolint:gosec // modular by intent: see the wrap note above
	if int(d) >= regionFRMaxK {
		return -1
	}
	return int(d)
}

// live returns the probe watching s, or nil.
func regionLive(s *shard) *regionProbe {
	pr := regionActive.Load()
	if pr == nil || pr.s != s {
		return nil
	}
	return pr
}

// regionNoteGen records a newly minted page generation as the shard's newest.
// Called from shard.nextGen, which holds mu (or runs during construction).
func regionNoteGen(s *shard, g uint16) {
	if pr := regionLive(s); pr != nil {
		pr.newest.Store(uint32(g))
	}
}

// regionNoteHit classifies a read that found its record. p must be the page the
// probe SETTLED on — see the note above.
func regionNoteHit(s *shard, p *page) {
	pr := regionLive(s)
	if pr == nil || p == nil {
		return
	}
	pr.counts.hits.Add(1)
	if d := pr.age(p.gen); d >= 0 {
		pr.counts.hitDist[d].Add(1)
	}
}

// regionNoteInPlace classifies a write that overwrote its record where it lay.
// Called under the shard write lock, after the bytes are down.
func regionNoteInPlace(s *shard, p *page) {
	pr := regionLive(s)
	if pr == nil || p == nil {
		return
	}
	pr.counts.inPlace.Add(1)
	if d := pr.age(p.gen); d >= 0 {
		pr.counts.placeDist[d].Add(1)
	}
}

// regionFRAttach points the probe at s and returns the function that detaches it.
// The counters are cumulative from here, so a caller measuring a window takes a
// regionFRSnapshot at each end and subtracts.
//
// It seeds newest from the shard's generation counter so a shard attached after
// its pages were built classifies correctly from the first hit.
func regionFRAttach(s *shard) func() {
	pr := &regionProbe{s: s}
	s.mu.Lock()
	pr.newest.Store(uint32(s.genCounter - 1)) // genCounter is the NEXT generation
	s.mu.Unlock()
	regionActive.Store(pr)
	return func() { regionActive.Store(nil) }
}

// regionFRSnapshot reads the attached probe's counters, or zeros if none.
func regionFRSnapshot() regionFRCounts {
	pr := regionActive.Load()
	if pr == nil {
		return regionFRCounts{}
	}
	out := regionFRCounts{Hits: pr.counts.hits.Load(), InPlace: pr.counts.inPlace.Load()}
	for i := range out.HitDist {
		out.HitDist[i] = pr.counts.hitDist[i].Load()
		out.PlaceDist[i] = pr.counts.placeDist[i].Load()
	}
	return out
}
