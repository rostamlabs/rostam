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
	// HitDist[d] counts hits landing on a page d retirements behind the newest;
	// PlaceDist[d] the same for in-place rewrite targets. A record older than
	// regionFRMaxK pages falls into neither array, so sum(HitDist[:K]) never
	// over-counts the residency of a region of K pages.
	//
	// There is deliberately no running TOTAL beside these. The denominators the
	// shares are taken over — gets and puts — are already counted by the shard
	// itself, so a total here would be a second contended atomic per hit and per
	// rewrite, paid inside the very window whose read/write interleaving is being
	// measured. sum(HitDist)/Gets and the shard's own hit% are each other's check
	// at K = the page count, which is what inPlaceRegionKs exists to report.
	HitDist   [regionFRMaxK]uint64
	PlaceDist [regionFRMaxK]uint64
}

// regionProbe is the live measurement state for ONE shard. Benchmarks measure a
// single shard at a time (Config.NumShards = 1 throughout inplace_bench_test.go),
// so a single global slot is enough and costs a lock-free pointer load on the hit
// path instead of a per-shard map lookup.
type regionProbe struct {
	s      *shard
	newest atomic.Uint32 // generation of the most recently PUBLISHED page on s
	counts struct {
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

// regionNotePage records a page's generation as the shard's newest. It must be
// called AFTER the page object is stored into pageSlots, never when the
// generation is minted.
//
// Minting runs first and publication can follow several statements later — the
// page is built, put into s.pages, and only then published — and a lock-free
// reader takes no part in that. A reader landing in the gap would measure every
// page it can actually resolve against a newest that no reader can reach yet,
// making each of them look one retirement older than it is and shifting some of
// them across a region boundary. The counters guarding the measurement's own
// correctness would show nothing: the shares stay plausible, they are just
// wrong. So the notification rides the publication store, on every path that
// makes one — first allocation, both heap retirement paths, mmap attach and the
// online-compaction recycle.
//
// Called with mu held (or during construction), like the store it follows.
func regionNotePage(s *shard, p *page) {
	if pr := regionLive(s); pr != nil && p != nil {
		pr.newest.Store(uint32(p.gen))
	}
}

// regionNoteHit classifies a read that found its record. p must be the page the
// probe SETTLED on — see the note above.
func regionNoteHit(s *shard, p *page) {
	pr := regionLive(s)
	if pr == nil || p == nil {
		return
	}
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
	if d := pr.age(p.gen); d >= 0 {
		pr.counts.placeDist[d].Add(1)
	}
}

// regionFRAttach points the probe at s and returns the function that detaches it.
// The counters are cumulative from here, so a caller measuring a window takes a
// regionFRSnapshot at each end and subtracts.
//
// It seeds newest from the newest page the shard has actually PUBLISHED, not
// from its generation counter, for the reason in regionNotePage: a shard caught
// mid-allocation has minted a generation no reader can resolve yet, and starting
// from it would mis-age every hit until the next publication.
func regionFRAttach(s *shard) func() {
	pr := &regionProbe{s: s}
	s.mu.Lock()
	var newest uint16
	first := true
	for i := range s.pageSlots {
		p := s.pageSlots[i].Load()
		if p == nil {
			continue
		}
		if first || p.gen-newest < 1<<15 { // modular "is ahead of", as in age
			newest, first = p.gen, false
		}
	}
	pr.newest.Store(uint32(newest))
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
	var out regionFRCounts
	for i := range out.HitDist {
		out.HitDist[i] = pr.counts.hitDist[i].Load()
		out.PlaceDist[i] = pr.counts.placeDist[i].Load()
	}
	return out
}
