// SPDX-License-Identifier: Apache-2.0

package kvindex

// The bounded reconcile pass: the backstop for postings whose key left the
// cache by a path that never reached Drop.
//
// # Why it exists at all, and why correctness never depends on it
//
// Every candidate is re-read and re-checked against the live value before a row
// is emitted, so a posting for a key the cache no longer holds is a wasted
// lookup and NEVER a wrong row. The onRemove hook covers every removal path in
// this build. What it does not cover is the residue:
//
//   - a CORRUPT slot, which carries no key at all, so sweepIndex tombstones it
//     with nothing to name in the hook (cache.SetOnRemove says so);
//   - a torn page's in-place Reset, which abandons index slots WITHOUT bumping
//     the page generation, so those slots later decode as a FOREIGN key rather
//     than reaching the corrupt branch;
//   - a Rebuild whose walk straddles a flush, which posts keys the flush has
//     already removed (Set.Reset explains why a Rebuild, unlike a Backfill, is
//     not token-checked);
//   - any future cache path that forgets to fire the hook.
//
// So this bounds MEMORY, not correctness. It only ever drops.
//
// # Liveness is a KEY RE-READ, never an inference from slot state
//
// The residue above is exactly why. An abandoned slot can decode cleanly as
// some other key, so "the slot this posting pointed at still parses" proves
// nothing. The only sound test is to ask the cache for THIS key and see it come
// back: indexTable.get compares the stored key with bytes.Equal before
// reporting a hit, so a Get that returns a value has proved the live slot is
// this key's, and a Get that misses has proved it is not.
//
// # A batch is a TRUNCATED RANGE, and coverage is probabilistic
//
// The batch takes the first `budget` keys a range over the reverse map hands
// it and stops. Go starts every map range at a randomly chosen bucket, so
// consecutive ticks sample different keys and the whole reverse map is covered
// in expectation — the coupon-collector bound, about n·ln(n)/budget ticks to
// have touched every one of n keys at least once, with no guarantee for any
// particular key on any particular tick.
//
// THAT IS THE POINT OF THE DESIGN, not a concession. An ordered sweep with a
// per-definition cursor gives exact coverage but has to examine every key to
// find the next `budget` above the cursor, which is O(live keys) under the
// index lock every tick: 53 ms at 1 M keys and 208 ms at 4 M, measured — and
// because sync.RWMutex stops admitting readers as soon as a writer is waiting,
// the first Reindex to arrive mid-walk pins every subsequent Candidates behind
// the rest of that walk (202 ms, against 20 µs unloaded). Chunking it is not
// available either: a Go map range cannot survive an unlock, and restarting the
// range each chunk is O(n²/budget). An ordered cursor also STARVES under
// monotonically increasing keys — the cursor chases the appended tail and a low
// dangling key waits unboundedly — which the random sample does not.
//
// The pass is a memory bound that correctness never depends on, so trading
// exact coverage for a lock hold that does not scale with the index is the
// right way round. Measured (BenchmarkReconcileBatch, budget 10 000): 0.74 ms
// at 100 k keys and 1.31 ms at 1 M — 1.8x for 10x the keys, and that residue is
// cache locality on a bigger map, not a scan. The cost is the 10 000 marks and
// 10 000 key copies the budget itself buys, which is the same at every size.
//
// # Two phases, and the split is mandatory
//
// The probe takes cache locks, and Cache.Get of an expired key on a
// non-replicated shard runs dropExpiredLocked → the shard write lock →
// onRemove → Set.Drop → s.mu. An index method holding s.mu across that
// self-deadlocks single-threaded, with no concurrency required. So:
//
//	keys := idx.ReconcileBatch(name, budget) // s.mu held, then released
//	dead := probe(keys)                      // NO index lock held
//	n := idx.DropReconciled(name, dead)      // s.mu again
//
// ops.ReconcileKVIndex is the caller that does this; shard.Store and the
// single-node directStore each run it on a ticker. See the package doc for the
// lock order this obeys.
//
// # The re-add race, and how the drop is made exact
//
// A write can land in the middle phase, so a key that was probed as dead and
// then written AGAIN is live by the time DropReconciled runs. Dropping its
// posting would leave a live key unposted — a MISSING posting, which
// verify-on-read cannot repair and which loses a row.
//
// Checking the reverse map for the key is not enough, and neither is comparing
// the value it maps to: a delete followed by a re-add under the SAME value is
// indistinguishable from no write at all (ABA). The guard is therefore a
// SUSPECT MARK, not a snapshot comparison. ReconcileBatch marks the entries it
// hands out; posting.set and posting.drop clear the mark for any key they
// touch — set does it BEFORE its unchanged-value early return, because a
// rewrite that does not move the field is still proof the key is live — and
// DropReconciled removes only entries that are still marked. Anything that
// wrote the key in the window has therefore cleared its mark, whatever it
// wrote. Reset (a flush, a Rebuild, a Backfill) and an Install that REPLACES
// the posting drop every mark with it. An Install that keeps the posting —
// sameShape, so nothing about what it posts changed — keeps the marks too, and
// correctly: they still name the same entries in the same map.
//
// # Exactly ONE driver per Set
//
// The marks are a single map per posting, so a second concurrent tick over the
// same definition would replace the first one's marks and the first one's
// DropReconciled would then drop nothing. That is fail-SAFE (a missed drop
// costs a stale posting for one interval, never a row), but it is waste, and
// nothing in this package serialises it. Each Set is owned by exactly one
// store, and that store runs exactly one reconcile goroutine.

// ReconcileBudget is how many keys one tick examines. It bounds the batch, the
// marks it sets, and the probes the caller then runs, so a tick's cost does not
// grow with the index.
const ReconcileBudget = 10_000

// ReconcileBatch returns up to budget of name's posted keys and MARKS them for
// a drop the caller may confirm. The caller probes each key's liveness with NO
// index lock held and passes the dead ones to DropReconciled, which it must
// call to end the tick even when nothing died.
//
// The keys are whichever ones a range over the reverse map reaches first, which
// Go randomises per range, so successive ticks sample different keys and
// coverage is probabilistic rather than exact — see the file comment for why
// that is the right trade. The keys are fresh copies; nothing the caller holds
// afterwards points into the index.
//
// It takes the WRITE lock, because it marks as it collects. The hold is
// O(budget) map operations and key copies and nothing else — about 1 ms at the
// default budget, and essentially independent of how large the index is.
func (s *Set) ReconcileBatch(name string, budget int) [][]byte {
	if budget <= 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.posts[name]
	if p == nil {
		return nil
	}
	// Sized by what exists, not by what the caller asked for: budget arrives
	// from a config value and a size hint allocates eagerly.
	hint := budget
	if n := len(p.keys); n < hint {
		hint = n
	}
	// A fresh map, so an abandoned tick's marks cannot outlive it.
	p.suspects = make(map[string]struct{}, hint)
	out := make([][]byte, 0, hint)
	for k := range p.keys {
		p.suspects[k] = struct{}{}
		out = append(out, []byte(k))
		if len(out) == budget {
			break
		}
	}
	return out
}

// DropReconciled removes the postings for the keys the caller's probe found
// dead, and ends the tick. Only a key this Set marked in the matching
// ReconcileBatch — and that nothing has written since — is removed; see the
// re-add race above. It returns how many postings actually went.
//
// CALL IT EVEN WITH NO DEAD KEYS. Ending the tick is what clears the marks, and
// a mark left standing costs the write path one map delete per touched key
// until the next batch replaces it.
//
// It pairs with ONE ReconcileBatch on one goroutine; see the single-driver note
// above.
func (s *Set) DropReconciled(name string, dead [][]byte) int {
	s.mu.Lock()
	p := s.posts[name]
	if p == nil {
		s.mu.Unlock()
		return 0
	}
	n := 0
	for _, k := range dead {
		if _, marked := p.suspects[string(k)]; !marked {
			// Never offered by this tick's batch, or written since it was: either
			// way this posting is not the one the probe judged.
			continue
		}
		if p.drop(k) {
			n++
		}
	}
	p.suspects = nil
	s.mu.Unlock()
	if n > 0 {
		s.reconcileDrops.Add(uint64(n)) //nolint:gosec // n > 0 checked above
	}
	return n
}

// ReconcileDrops reports how many postings the reconcile pass has removed since
// this Set was created. Monotonic. The node sums it over its hosted groups into
// Stats().KVIndex.ReconcileDrops; a rising rate means keys are leaving the cache
// by a path that does not reach Drop.
func (s *Set) ReconcileDrops() uint64 { return s.reconcileDrops.Load() }
