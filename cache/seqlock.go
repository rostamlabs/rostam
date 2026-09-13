// SPDX-License-Identifier: Apache-2.0

package cache

import "sync/atomic"

// A seqlock over the entries a same-size update overwrites, so a heap ringbuf
// shard with Config.InPlaceSameSizeUpdate can keep reading WITHOUT the shard
// read lock. The lock is what that feature otherwise costs, and it is not cheap:
// an RWMutex read acquisition is a read-modify-write on one cache line shared by
// every reader, so it does not scale with cores.
//
// THE VERSION WORDS LIVE ON THE PAGE, NOT IN THE ENTRY HEADER, and not in the
// index. Two concrete reasons, both of which rule the other placements out
// rather than merely disfavouring them:
//
//   - THE ENTRY HEADER CANNOT HOLD ONE. Entries are packed back-to-back at
//     arbitrary byte offsets, so a word inside an entry has arbitrary alignment,
//     and Go's atomics require natural alignment — unaligned atomics fault on
//     arm64 and this module builds for 386/arm as well. Aligning entry starts
//     would not be enough either: the header's spare CRC slot sits at +22, which
//     is never 8-aligned however the entry is placed. Making it work means
//     padding the framing itself, which forks the heap entry format away from the
//     mmap one that recovery and cold compaction still have to read.
//   - THE INDEX CANNOT HOLD ONE EITHER, though its ctrl word has 47 spare bits
//     and the reader already loads it. A rehash publishes a NEW table, and a
//     reader that snapshotted the old one would keep checking a version word no
//     writer updates any more — a torn value nothing detects. The page object a
//     ref names is stable for exactly as long as the ref is valid, which is the
//     property a seqlock needs.
//
// So each heap page carries a small array of version counters and an entry maps
// to one by its offset. That makes the protection per-STRIPE rather than
// per-entry: a reader retries when any entry sharing its stripe is rewritten,
// not only its own. The cost of that is a retry, never a wrong answer, and with
// seqlockStripesPerPage stripes the collision odds are a fraction of a percent
// (see BenchmarkInPlaceRead's reported retry rate).
//
// # Ordering
//
// The writer holds s.mu, so writers are serialised against each other and this
// protocol only has to order ONE writer against concurrent lock-free readers.
//
//	writer:  ver.Add(1)   // even -> odd: "this stripe is being rewritten"
//	         <plain stores to the entry bytes>
//	         ver.Add(1)   // odd -> even: done, version advanced
//
//	reader:  v1 := ver.Load(); retry if odd
//	         <plain loads of the entry bytes, copied into a private buffer>
//	         v2 := ver.Load(); retry if v2 != v1
//
// Go's sync/atomic operations are sequentially consistent, so both of the
// writer's Adds are full barriers: the payload stores can neither be hoisted
// above the first nor sunk below the second. That is the half of the ordering
// the writer owes, and it is guaranteed by the language.
//
// THE READER'S HALF IS WEAKER, AND THE COMMENT SHOULD SAY SO RATHER THAN IMPLY
// OTHERWISE. Its first Load is an acquire, so the payload loads cannot be
// hoisted above it. Its trailing Load needs the opposite — the payload loads
// must not SINK BELOW it — and an acquire load does not by itself promise that.
// In practice the guarantee holds: on amd64 loads are not reordered with later
// loads at all, and on arm64 Go compiles an atomic load to LDAR, which Go's
// sequentially-consistent contract requires to be ordered against the accesses
// around it. But the payload accesses are PLAIN and racing, so the Go memory
// model extends them no happens-before edge of their own, and this protocol
// rests on how the atomics are compiled rather than on what the model
// guarantees. That is a real limitation, it is why this is opt-in and off by
// default, and it is the reason the alternative below is worth stating.
//
// # Why the payload is not accessed atomically
//
// Making the payload accesses atomic too would put the whole protocol inside the
// memory model AND silence the race detector, which flags a seqlock's payload
// race by construction (detecting a torn value after the fact is what a seqlock
// does; it does not stop the concurrent access TSan reports). It is not
// affordable. Word-wise atomic loads of a 256-byte payload measure ~23.5ns
// against ~3.9ns for the copy — tolerable — but word-wise atomic STORES measure
// ~181.5ns, on a write whose whole cost is ~168ns. The write side would more
// than double to buy the read side ~50ns, which inverts the trade this feature
// exists to make.
//
// The consequence is that a shard reading through the seqlock has a payload race
// the detector will report. Config.InPlaceSeqlockReads therefore defaults off,
// the read-locked path stays the default for in-place shards, and the tests that
// exercise the seqlock skip under -race.
const (
	// seqlockStripesPerPage is the number of version counters a heap page carries
	// when the seqlock is enabled. 64 words is 512 bytes against a page of at least
	// 1 MiB — under 0.05% — and spreads a page's entries thinly enough that two
	// readers of unrelated keys almost never share one.
	seqlockStripesPerPage = 64
	seqlockStripeMask     = seqlockStripesPerPage - 1

	// seqlockMaxRetries bounds how many times a read re-attempts before giving up
	// on the lock-free path. Exhaustion is not an error and not a miss: the caller
	// falls back to taking the shard READ LOCK for that one read, which excludes
	// the writer outright and therefore always terminates. A bound is what makes
	// that guarantee unconditional — without one, a reader unlucky enough to share
	// a stripe with a hot key could spin arbitrarily long.
	seqlockMaxRetries = 4
)

// enableVersions gives a heap page its seqlock version counters. Called on every
// heap page a seqlock-reading shard allocates — including the fresh page that
// replaces a retired one — before the page is published to any reader.
//
// A fresh page starts every counter at zero, which cannot be confused with the
// retired page's counters: a reader holding a ref into the old page also holds
// that old page OBJECT, and reads ITS versions. The generation gate has already
// turned the stale ref into a miss before any of this is consulted.
func (p *page) enableVersions() {
	p.vers = make([]atomic.Uint64, seqlockStripesPerPage)
}

// versionAt returns the version counter guarding the entry at offset, or nil if
// this page has none (every page on a shard that is not reading through the
// seqlock). A nil return means the caller must not use the lock-free path.
//
// Entries are mapped by offset shifted past the byte-level detail, so entries
// that are near neighbours land on different stripes and the mapping costs a
// shift and a mask.
func (p *page) versionAt(offset uint32) *atomic.Uint64 {
	if p.vers == nil {
		return nil
	}
	return &p.vers[(offset>>8)&seqlockStripeMask]
}

// seqlockReads reports whether this shard's reads run the seqlock protocol
// instead of taking the read lock. Immutable for the shard's life.
//
// It requires in-place updates to be enabled, because the seqlock protects
// exactly one thing: an entry whose bytes a writer can rewrite underneath a
// reader. With no in-place updates there is nothing to protect and the reads
// were already lock-free — turning this on alone would add a version check that
// guards against a hazard that cannot arise.
//
// It is heap+ringbuf only, for the same reasons in-place updates are (see
// inPlaceEligible), and mmap ringbuf is deliberately NOT included: its eviction
// rewrites live bytes too, but it is the DURABLE copy and its readers hold
// zero-copy aliases, neither of which a seqlock addresses.
func (s *shard) seqlockReads() bool {
	return s.cfg.InPlaceSeqlockReads && s.inPlaceEligible()
}

// bumpVersionLocked runs the writer half of the protocol around fn, which does
// the plain stores to the entry's bytes. Must hold s.mu for writing.
//
// When the shard is not reading through the seqlock there is no counter and fn
// runs bare — readers are taking the read lock in that configuration, and the
// writer already holds the write lock that excludes them.
func (s *shard) bumpVersionLocked(p *page, off uint32, fn func() error) error {
	ver := p.versionAt(off)
	if ver == nil {
		return fn()
	}
	ver.Add(1) // even -> odd: readers in this window must retry
	err := fn()
	ver.Add(1) // odd -> even: the new bytes are published
	return err
}

// getSeqRetry runs the lock-free seqlock probe, re-attempting while a concurrent
// rewrite keeps invalidating it. ok=false means the budget was spent and the
// caller must take the read lock for this read instead; it is NOT a miss, and
// reporting it as one would turn a present key into an absent one.
//
// The table is re-loaded on every attempt on purpose. A retry can follow a
// concurrent rehash, and the slot indices of the table that was snapshotted are
// meaningless against the one that replaced it.
//
// dst is appended into (nil for a fresh allocation), so the value is always an
// owned copy — a seqlock cannot hand back an alias, because an alias outlives
// the version check that would have validated it.
func (s *shard) getSeqRetry(dst, key []byte, h uint64) (out []byte, exp uint64, ref slabRef, st lookupStatus, ok bool) {
	for attempt := 0; attempt < seqlockMaxRetries; attempt++ {
		t := s.tab.Load()
		out, exp, ref, st, ok = t.getSeq(s, dst, key, h)
		if ok {
			if attempt > 0 {
				s.seqlockRetries.Add(uint64(attempt)) //nolint:gosec // attempt < seqlockMaxRetries
			}
			return out, exp, ref, st, true
		}
	}
	s.seqlockRetries.Add(seqlockMaxRetries)
	s.seqlockFallbacks.Add(1)
	return dst, 0, 0, lkMiss, false
}
