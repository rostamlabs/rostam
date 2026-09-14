// SPDX-License-Identifier: Apache-2.0

package cache

// A SIEVE-style visited bit, so relocating eviction rescues records by REFERENCE
// rather than by position.
//
// ==========================================================================
// THE PROBLEM IT ADDRESSES. Relocating eviction (cache/relocate_evict.go) drains a
// page by carrying its live records to a freed page instead of dropping them. It
// chooses what to rescue POSITIONALLY — everything index-current on the page the
// rotation cursor reached — with no reference to how recently anything was used, so
// it spends a bounded copying budget on whatever the walk meets first. Cold records
// are rescued at full copying cost and hot ones further along the page are left to
// be dropped.
//
// That is worst exactly where it matters most. Config.InPlaceSameSizeUpdate rewrites
// a record where it lies, so the record keeps its page and its offset and its page
// age only ever grows: in-place updates are precisely what STOPS refreshing a
// record's recency. Anything reasoning about PAGE AGE is therefore aimed away from
// the records in-place serves, which is why a mutable-region scheme was measured and
// rejected rather than tuned.
//
// A visited bit is the remaining instrument because it acts at drain time on the
// RECORD. The drain asks "was this record used?", not "how old is the page it sits
// on?", and an in-place rewrite can answer the first question where it cannot move
// the second.
//
// ==========================================================================
// WHERE THE BIT LIVES: THE INDEX CONTROL WORD (cache/indextable.go). ctrl has spare
// bits above the tag and the lock-free reader already loads it to compare that tag,
// so marking costs no extra fetch and the test at drain time costs no extra load
// either.
//
// cache/seqlock.go argues at length that the index is the WRONG place for state,
// because a rehash publishes a fresh table and a reader holding the old one keeps
// updating a word no writer consults. That argument is correct and it does NOT
// transfer here, and the distinction is worth stating because the seqlock's version
// of it is persuasive enough to be mistaken for a general rule:
//
//   - FOR A VERSION WORD IT IS FATAL. The version is the only thing standing between
//     a reader and a torn value. A counter the writer stopped advancing validates
//     anything, so a stale one means a corrupt value that nothing detects.
//   - FOR A HINT IT IS HARMLESS. A lost visited bit costs one missed retention — a
//     record the cache would have kept is dropped, which is the outcome the policy
//     produces for any unreferenced record anyway. It can never produce a wrong
//     answer, because nothing reads the value differently for it.
//
// THAT LICENCE IS NOT A LICENCE TO LOSE THEM IN BULK. indexTable.rehashed rebuilds
// the table by upserting every live entry and it CONTROLS that copy, so it carries
// the bit across; see the reasoning there. Without that, one rehash clears every
// hint in the shard simultaneously and the next drain sees an entire page of
// unvisited records — not "one missed retention" but a page of them. The residual
// this cannot address, and accepts: a bit set by a reader that snapshotted the table
// just before the swap lands on the frozen copy and is lost. That is one hint, for
// one record, in the window of one rehash.
//
// ==========================================================================
// WHAT SETS THE BIT: AN ACCESS, NEVER A FIRST INSERTION.
//
// A READ HIT MARKS. indexTable.get and getSeq set it once the key has been confirmed
// (and, on the seqlock path, once the read has been VALIDATED, so a torn read leaves
// no hint).
//
// AN UPDATE OF AN ALREADY-PRESENT KEY MARKS. shard.putAtExpLocked sets it on both
// write paths — the in-place overwrite and an appending rewrite. The justification is
// narrow and it is the reason this is not a departure from SIEVE: an APPENDING
// rewrite already refreshes a record's standing, by moving it to the newest page and
// so ahead of the rotation. In-place updates deliberately gave that up, and marking
// restores exactly what they took away rather than granting something the append path
// never had.
//
// AN INSERT DOES NOT MARK. This is SIEVE's one-hit-wonder resistance and it is the
// half that does the work here: a stream of keys written once and never touched again
// is the traffic that evicts everything else, and it must arrive unvisited or the bit
// discriminates nothing. indexTable.upsert publishes a clean control word for a new
// key, so this is the default and needs no code — only the discipline that nothing
// else marks. Relocation, compaction and the warm-restart rebuild all move bytes for
// reasons that are not accesses, and none of them mark.
//
// THE CONSEQUENCE FOR A WRITE-ONLY KEY, stated because it is the cost of the choice
// above. A key that is rewritten constantly and NEVER read stays marked, and so
// competes for retention with keys that are actually being served. Strict SIEVE would
// drop it. We keep it, because the cache cannot distinguish "written and about to be
// read" from "written and never read", and because in-place updates are opt-in
// precisely for rewrite-heavy shards — declining to retain what such a shard spends
// all its writes on would make the feature useless on its own workload. The retention
// is still bounded by activity: a key that stops being written and is not read loses
// its bit at the next drain that passes it and falls out on the one after.
//
// ==========================================================================
// WHAT THE DRAIN DOES WITH IT. relocateIntoFreedPageLocked rescues a live record only
// if its slot is marked, and CLEARS the mark as it rescues — SIEVE's hand clearing as
// it passes, in takeVisited, a single atomic test-and-clear. An unmarked live record
// is left where it is and the retire walk drops it on the next eviction, which is
// what a non-relocating eviction would have done to it anyway.
//
// A RESCUED RECORD STARTS ITS NEXT ROTATION UNVISITED. That is the property that makes
// this a policy rather than a ratchet: a record earns a rescue by being used, and has
// to be used again to earn another. A record rescued repeatedly is one referenced
// repeatedly, and it pays for each rescue with a fresh access. It also gets a full
// rotation of grace to earn the next one, because the copy lands on the page furthest
// from the cursor.
//
// THE DEGENERATE CASES, which is where a policy like this goes wrong:
//
//   - EVERYTHING ON THE PAGE IS MARKED. The pass behaves exactly as it does without
//     the bit — positional, budget-capped — which is the behaviour this is trying to
//     improve on, so the floor is the status quo and not something worse. It does not
//     stay there: the hand clears every bit it passes, so the NEXT rotation over those
//     records discriminates unless they were all used again.
//   - NOTHING ON THE PAGE IS MARKED. Nothing is copied, the eviction costs what a
//     non-relocating eviction costs, and the records dropped are the ones no reader or
//     writer touched. This is the case the feature exists for: it is where relocation
//     was paying full copying cost to rescue cold records.
//   - THE BUDGET IS EXHAUSTED. Unchanged. The budget bounds what one eviction may
//     copy, and the bit changes WHICH records that budget buys, never how much of it
//     there is. Marked records past the budget are dropped exactly as unmarked ones
//     are — and the hand has already cleared their mark by then, because it clears as
//     it passes rather than as it rescues. Nothing turns on that: the page they are on
//     is by construction the next eviction's victim, so they cease to exist before the
//     mark could be consulted again. The two orderings produce identical moves, since
//     a refused record consumes no budget either way.
//
// ==========================================================================
// SCOPE. The bit is maintained only on a PolicyRingbufEvict shard, and consumed only
// inside relocateIntoFreedPageLocked, which runs only under Config.RelocatingEviction
// and only from evictVictimLocked. Eviction itself is reachable from one place —
// findOrMakePageLocked's `case PolicyRingbufEvict` — so a PolicyRejectWrites shard
// never drains a page and nothing here can affect it. Replication forces
// reject-writes (shard/store.go), so no replicated shard is touched either, and the
// retained set therefore cannot diverge between peers.
//
// It is also NOT consulted by the background free-page reserve's own evacuation walk
// (cache/relocate_reserve.go). That pass has the opposite contract — it never drops
// what it could not move, and retires a page only once it is FULLY evacuated — so
// skipping unmarked records there would simply stop it ever freeing a page. The
// handoff it performs after retiring calls relocateIntoFreedPageLocked, which is the
// SIEVE-aware pass, and that is correct: the page that pass evacuates is the one the
// next write-path eviction will drain.

// sieveVisited reports whether this shard maintains the SIEVE reference hint.
// Immutable for the shard's life (config plus the capacity policy, both fixed at
// construction), which is why shard.sieve caches it: the read path tests it on every
// hit and a single bool load is what that should cost.
func (s *shard) sieveVisited() bool {
	// RelocatingEviction is part of the predicate because it is the ONLY consumer:
	// without it no drain reads the hint, so marking would be pure write traffic on
	// the read path for nothing. The config doc promises the bit is a no-op without
	// it, and this is where that promise is kept rather than merely described.
	return s.cfg.SieveVisitedBit && s.cfg.RelocatingEviction && s.cfg.AtCapPolicy == PolicyRingbufEvict
}

// setVisited marks slot as accessed. tag must be the tag the caller matched on, and
// ref the slabRef the caller SETTLED on — the one whose bytes it read and whose key
// it confirmed, after any rechase.
//
// THE TAG ALONE DOES NOT IDENTIFY THE RECORD, AND MARKING ON IT ALONE MARKS THE WRONG
// KEY. A lock-free reader confirms its key and only then marks; in between, a writer
// can delete that key and insert a different one, whose upsert reuses the slot's
// tombstone. A tag is 16 bits of hash, so an unrelated key shares one about once in
// 65,536 — and on a collision the late mark lands on the new occupant. That occupant
// is a freshly INSERTED key, which is precisely the record this policy must see as
// unvisited: the one-hit-wonder resistance the whole design rests on is that a
// write-once key arrives unmarked. So the failure is not a generic staleness, it is
// the discriminating half of SIEVE being defeated from the other direction.
//
// THE REF IS THE IDENTITY CHECK, AND IT IS FREE. It packs page index, generation and
// offset, so two different records cannot share one; and the caller already loaded it
// to find the bytes it read, so re-reading it here hits the same line the probe just
// touched. Checking it turns "16 bits of hash agree" into "this is the same physical
// record".
//
// THE RESIDUAL, STATED RATHER THAN PAPERED OVER. The check narrows the window to the
// few instructions between the ref load and the CAS; it does not close it. A wrong
// mark would now need the slot tombstoned, re-pointed AND re-published with a
// colliding tag inside that window. If it ever happened the cost is bounded to what
// relocation did before this hint existed — one live record carried forward
// positionally — for one rotation, because the hand clears the mark as it passes.
// That is the mildest failure available: the status quo ante, for one record.
//
// A MISMATCH IS A DROPPED HINT, WHICH IS ALWAYS SAFE. If relocation repoints the slot
// between the read and the mark, the ref no longer matches and the mark is skipped.
// Losing a hint costs one missed retention and can never produce a wrong answer, so
// the guard is free to be conservative.
//
// THE WRITE PATHS DO NOT NEED THIS GUARD BUT PASS IT ANYWAY. They hold s.mu, and every
// slot mutation happens under it, so no reuse can interleave with them. One function
// with one rule is worth more than a second entry point whose safety argument depends
// on its caller holding a lock.
//
// IT IS A COMPARE-AND-SWAP RATHER THAN AN OR, and the reason is a race a plain
// read-modify-write would lose. A lock-free reader can match a tag and then have the
// writer tombstone that slot underneath it; OR-ing the hint in afterwards would
// produce a control word that is neither a sentinel nor a tag — a slot upsert would
// never reuse and no probe would ever match, leaked until the next rehash. Requiring
// the word to be EXACTLY the unmarked tag makes the store land only on a slot that is
// still the one the caller matched, and turns every other outcome into a no-op. It
// also means the hint bit can only ever sit on a slot that carries a tag, which is
// what lets the ctrlEmpty/ctrlTombstone tests stay unmasked equality.
//
// A FAILED CAS IS NOT RETRIED. It means either that the slot changed — in which case
// there is nothing here to mark — or that the bit is already set, which is the state
// the caller wanted. Spinning would buy nothing and put a loop on the read path.
//
// THE STEADY-STATE READ PAYS NOTHING FOR THIS. A slot already marked fails the
// compare on its first attempt, so only the FIRST read after a drain cleared the bit
// performs a real store; and the line is per-SLOT, not one word every reader of the
// shard shares.
func (t *indexTable) setVisited(slot, tag uint64, ref slabRef) {
	if t.ctrl[slot].Load() != tag {
		return // already marked, or no longer the slot the caller matched
	}
	if slabRef(t.refs[slot].Load()) != ref {
		return // the slot no longer names the record the caller read
	}
	t.ctrl[slot].CompareAndSwap(tag, tag|ctrlVisited)
}

// takeVisited is SIEVE's hand: it reports whether slot was marked and clears the mark
// in the same atomic step. Writer-side (the drain holds mu), but atomic regardless,
// because a lock-free reader can be marking the same word concurrently.
//
// Test-and-clear as ONE operation is what keeps the two halves honest. Reading the
// bit and clearing it separately would let an access that lands between them be
// consumed by a rescue that had already decided the record was cold — the record is
// saved, which is harmless, but its fresh mark is destroyed, which costs it the NEXT
// rescue it had legitimately earned. A CAS either takes a mark that was there or
// finds none, and leaves a mark set after it as a mark.
func (t *indexTable) takeVisited(slot, tag uint64) bool {
	return t.ctrl[slot].CompareAndSwap(tag|ctrlVisited, tag)
}
