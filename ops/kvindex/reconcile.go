// SPDX-License-Identifier: Apache-2.0

package kvindex

import "sort"

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
// # Three phases, and the split is mandatory
//
// The probe takes cache locks, and Cache.Get of an expired key on a
// non-replicated shard runs dropExpiredLocked → the shard write lock →
// onRemove → Set.Drop → s.mu. An index method holding s.mu across that
// self-deadlocks single-threaded, with no concurrency required. So:
//
//	keys, wrapped := idx.ReconcileBatch(name, budget) // s.mu held, then released
//	dead := probe(keys)                               // NO index lock held
//	n := idx.DropReconciled(name, dead)               // s.mu again
//
// ops.ReconcileKVIndex is the caller that does this; shard.Store runs it on a
// ticker. See the package doc for the lock order this obeys.
//
// # The re-add race, and how the drop is made exact
//
// A write can land in the middle phase, so a key that was deleted, probed as
// dead, and then written AGAIN is live by the time DropReconciled runs.
// Dropping its posting would leave a live key unposted — a MISSING posting,
// which verify-on-read cannot repair and which loses a row.
//
// Checking the reverse map for the key is not enough, and neither is comparing
// the value it maps to: a delete followed by a re-add under the SAME value is
// indistinguishable from no write at all (ABA). The guard is therefore a
// SUSPECT MARK, not a snapshot comparison. ReconcileBatch marks the entries it
// handed out; posting.set and posting.drop clear the mark for any key they
// touch — set does it BEFORE its unchanged-value early return, because a
// rewrite that does not move the field is still proof the key is live — and
// DropReconciled removes only entries that are still marked. Anything that
// wrote the key in the window has therefore cleared its mark, whatever it
// wrote. Reset and Install clear the whole set of marks with the posting.

// ReconcileBudget is how many keys one tick examines. It bounds both the copy
// ReconcileBatch makes and the probes the caller then runs, so a tick's cost
// does not grow with the index.
const ReconcileBudget = 10_000

// ReconcileBatch snapshots up to budget keys from name's reverse map and marks
// them for a drop the caller may confirm. The caller probes each key's liveness
// with NO index lock held and passes the dead ones to DropReconciled, which it
// must call to end the tick even when nothing died.
//
// Coverage rotates. Each definition carries a cursor; a batch takes the
// SMALLEST budget keys strictly above it and leaves the cursor at the largest
// one taken, so consecutive ticks sweep the whole reverse map in key order
// without repeating and without holding an iterator across a lock release.
// wrapped reports that this batch reached the end, and the cursor has restarted.
//
// The selection walk is O(keys) under the READ lock and O(budget) memory (a
// bounded max-heap keeps the smallest budget keys as it goes). It is a read
// lock rather than a write lock because that walk is the expensive half and
// Candidates must not queue behind it; the short marking pass that follows
// takes the write lock.
func (s *Set) ReconcileBatch(name string, budget int) (keys [][]byte, wrapped bool) {
	if budget <= 0 {
		return nil, false
	}
	p, gen, picked, found := s.reconcileSnapshot(name, budget)
	if p == nil {
		return nil, false
	}
	// found counts every key above the cursor, so found <= budget means the heap
	// holds ALL of them and the rotation has reached the end.
	wrapped = found <= budget
	sort.Strings(picked)
	next := ""
	if !wrapped && len(picked) > 0 {
		next = picked[len(picked)-1]
	}
	out, ok := s.markSuspects(name, p, gen, picked, next)
	if !ok {
		// The posting was replaced or emptied while the lock was down. The cursor
		// was not advanced and nothing was marked; the next tick starts over.
		return nil, false
	}
	return out, wrapped
}

// reconcileSnapshot does the locked selection half: it reads the cursor and
// keeps the smallest budget keys above it. It reports the posting and the
// generation it read them at (so the marking pass can tell whether they still
// belong to the same index) and how many keys were above the cursor in all.
func (s *Set) reconcileSnapshot(name string, budget int) (p *posting, gen uint64, picked []string, found int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p = s.posts[name]
	if p == nil {
		return nil, 0, nil, 0
	}
	// Size the heap by what actually exists, not by what the caller asked for:
	// budget reaches here from a config value and a size hint allocates eagerly.
	hint := budget
	if n := len(p.keys); n < hint {
		hint = n
	}
	h := smallestKeys{limit: budget, a: make([]string, 0, hint)}
	cur := p.recCursor
	for k := range p.keys {
		if cur != "" && k <= cur {
			continue
		}
		found++
		h.offer(k)
	}
	return p, p.gen, h.a, found
}

// markSuspects marks the selected keys and advances the cursor, under the write
// lock. It reports false when the posting it was told about is no longer the
// installed one (an Install replaced it) or has been emptied since (a Reset or
// a walk), in which case the keys describe an index that no longer exists.
//
// A key that has left the reverse map in the meantime is dropped from the
// batch: there is no posting left to reconcile, so probing it would be waste.
func (s *Set) markSuspects(name string, p *posting, gen uint64, picked []string, next string) ([][]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.posts[name] != p || p.gen != gen {
		return nil, false
	}
	p.recCursor = next
	// A fresh map, so an abandoned tick's marks cannot outlive it.
	p.suspects = make(map[string]struct{}, len(picked))
	out := make([][]byte, 0, len(picked))
	for _, k := range picked {
		if _, live := p.keys[k]; !live {
			continue
		}
		p.suspects[k] = struct{}{}
		out = append(out, []byte(k))
	}
	return out, true
}

// DropReconciled removes the postings for the keys the caller's probe found
// dead, and ends the tick. Only a key this Set marked in the matching
// ReconcileBatch — and that nothing has written since — is removed; see the
// re-add race above. It returns how many postings actually went.
//
// CALL IT EVEN WITH NO DEAD KEYS. Ending the tick is what clears the marks, and
// a mark left standing costs the write path one map delete per touched key
// until the next batch replaces it.
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

// smallestKeys keeps the smallest `limit` strings offered to it, as a max-heap
// so the largest kept one — the only candidate for eviction — is at a[0].
//
// A sort of everything above the cursor would be simpler and is the wrong
// shape: it allocates O(live keys) in a pass whose whole purpose is to bound
// memory. This is O(limit) memory and O(n log limit) time.
type smallestKeys struct {
	limit int
	a     []string
}

func (h *smallestKeys) offer(k string) {
	if len(h.a) < h.limit {
		h.a = append(h.a, k)
		h.up(len(h.a) - 1)
		return
	}
	if k >= h.a[0] {
		return // no smaller than the largest we already keep
	}
	h.a[0] = k
	h.down(0)
}

func (h *smallestKeys) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.a[parent] >= h.a[i] {
			return
		}
		h.a[parent], h.a[i] = h.a[i], h.a[parent]
		i = parent
	}
}

func (h *smallestKeys) down(i int) {
	n := len(h.a)
	for {
		l, largest := 2*i+1, i
		if l < n && h.a[l] > h.a[largest] {
			largest = l
		}
		if r := l + 1; r < n && h.a[r] > h.a[largest] {
			largest = r
		}
		if largest == i {
			return
		}
		h.a[i], h.a[largest] = h.a[largest], h.a[i]
		i = largest
	}
}
