// SPDX-License-Identifier: Apache-2.0

package cache

// onRemove and IterateChunked are the two seams a DERIVED SECONDARY INDEX needs
// from the cache. Neither changes what the cache stores, and neither is a
// durability unit: the index they feed is rebuilt by walking the cache at every
// restart, definition install, and snapshot restore.
//
// LOCK ORDER IS CACHE → INDEX, ONE WAY, ALWAYS. onRemove fires with a cache
// shard's write lock held, so the callback may take only its own lock and must
// never call back into the cache. Symmetrically, no index lock may be held
// across a cache call: Cache.Get of an expired key on a non-replicated shard
// runs dropExpiredLocked, which takes the shard write lock and fires the hook,
// so an index method holding its own lock across a cache read would self-
// deadlock single-threaded, with no concurrency required.

// SetOnRemove installs the callback the cache invokes whenever a LIVE index slot
// is removed: an explicit delete, a TTL expiry (lazy on read, or by the
// sweeper), a ringbuf eviction, an expired-page reclaim, and the warm-restart
// dead-slot strip. Passing nil clears it. It is safe to call at any time; the
// hook is read through an atomic pointer on every removal.
//
// THE CONTRACT, WHICH THE CALLBACK MUST HONOUR:
//
//   - It runs WITH THE CACHE SHARD'S WRITE LOCK HELD. It must not call back into
//     the cache, must not block, and must take only its own lock (see the lock
//     order above). The single exception is the warm-restart dead-slot strip,
//     which runs inside newShard with no lock at all because the shard is not yet
//     reachable — and where no hook can be installed either, since the Cache that
//     owns it does not exist until every shard is built. A callback that honours
//     the locked contract is therefore correct at every site.
//   - The key ALIASES THE PAGE BACKING STORE and is valid only for the duration
//     of the call. Copy before retaining — string(key) copies.
//   - It is a HINT CHANNEL, not a durability signal. A missed call costs a stale
//     posting, which a verify-on-read discards; it can never produce a wrong row.
//
// WHERE IT DELIBERATELY DOES NOT FIRE:
//
//   - AN OVERWRITE ITSELF. Repointing a slot is not a removal: a Put stores the
//     new physical copy and upserts, and the write path reindexes.
//
// BUT A Put CAN STILL FIRE THE HOOK FOR THE VERY KEY IT IS WRITING, so ORDER
// YOUR REINDEX AFTER THE Put RETURNS. On a ringbuf shard, Put →
// findOrMakePageLocked → evictUntilFitsLocked can choose as its victim the page
// that holds K's CURRENT copy. That copy matches `cur == ref`, so the eviction
// notifies onRemove(K) — from inside K's own Put, under the same lock, BEFORE
// the new copy is written. An index consumer that reindexed K first and then
// called Put would see the notification land after its own posting and drop it,
// leaving a live key with no posting: the one failure mode verify-on-read cannot
// repair. Reindex only once Put has returned, and the ordering is always safe.
// (TestOnRemoveEvictingPutFiresForItsOwnKey pins this.)
//   - A DEAD DUPLICATE. The eviction and page-reclaim walks visit every framed
//     entry on a page, including copies a later Put superseded. Those entries'
//     index slots already point at a newer live copy elsewhere, so the hook fires
//     only from inside the `ok && cur == ref` guard that tombstones the slot.
//     Firing per walked entry would drop a live key's posting — a missing row,
//     the one failure mode verify-on-read cannot repair.
//   - A FLUSH. Cache.Flush is an O(1) table swap with no per-entry walk; the
//     flush handler resets the whole index instead (an empty cache IS an exact
//     empty index).
//   - A CORRUPT SLOT. A slot whose page.Read fails carries NO KEY at all, so
//     sweepIndex tombstones it silently. The index's reconcile pass is the
//     backstop for that one posting.
func (c *Cache) SetOnRemove(fn func(key []byte)) {
	if fn == nil {
		c.onRemove.Store(nil)
		return
	}
	f := fn
	c.onRemove.Store(&f)
}

// fireOnRemove invokes the installed hook with key. Must be called with s.mu held
// for writing, from inside the branch that actually removed a live index slot.
//
// Zero cost when no hook is installed: one nil check on an immutable field plus
// one atomic load, no allocation, no call. s.onRemove is nil for a bare shard
// built outside Cache.New (tests, benchmarks).
func (s *shard) fireOnRemove(key []byte) {
	if s.onRemove == nil {
		return
	}
	if fn := s.onRemove.Load(); fn != nil {
		(*fn)(key)
	}
}

// fireOnRemoveAt is fireOnRemove for a removal site that holds a page reference
// rather than the key itself. The page read happens ONLY when a hook is
// installed, so a hookless shard pays nothing extra for it. An unreadable entry
// fires nothing (see the corrupt-slot note in SetOnRemove).
func (s *shard) fireOnRemoveAt(p *page, off uint32) {
	if s.onRemove == nil {
		return
	}
	fn := s.onRemove.Load()
	if fn == nil {
		return
	}
	if key, _, _, err := p.Read(off); err == nil {
		(*fn)(key)
	}
}

// iterateChunkedMaxRestarts bounds how many times a shard's chunked walk may be
// restarted by a concurrent rehash before it gives up and falls back to one
// locked Iterate. The fallback is what makes termination unconditional: a shard
// under a table swap on every chunk boundary would otherwise restart forever.
const iterateChunkedMaxRestarts = 8

// IterateChunked walks every live (non-expired) entry in the cache, releasing
// each shard's read lock every `batch` index slots so writers are not blocked for
// a whole shard's walk. Returning false from fn stops the whole iteration.
//
// WHY IT EXISTS. Iterate holds one shard's RLock for that shard's ENTIRE walk,
// and putAtExpLocked needs the same mutex for writing — so a plain Iterate stalls
// every write to the shard being walked until the walk finishes. The index
// backfill walks the whole cache, and a backfill must not stall the write path.
//
// If a shard's index table is SWAPPED between chunks (a concurrent rehash, which
// drops tombstones and renumbers every slot), the walk restarts that shard from
// slot 0 rather than reading the stale table. Restarting is safe because the
// consumer — a derived index rebuild — is idempotent: re-visiting a key just
// re-derives the same posting. After iterateChunkedMaxRestarts restarts the
// shard falls back to a single locked Iterate, so the walk always terminates.
//
// A non-positive batch means sweepBatchSize. The key and value slices alias the
// page backing store and are valid only for the duration of the fn call — and,
// because the lock IS released between chunks, they must not be retained across
// one. Copy if you need to keep them.
func (c *Cache) IterateChunked(batch int, fn func(key, value []byte) bool) {
	if batch <= 0 {
		batch = sweepBatchSize
	}
	for _, s := range c.shards {
		if !s.iterateChunked(batch, fn) {
			return
		}
	}
}

// iterateChunked walks one shard in chunks, restarting on a table swap and
// falling back to a locked walk once the restart budget is spent. Returns false
// if fn asked to stop the whole iteration.
func (s *shard) iterateChunked(batch int, fn func(key, value []byte) bool) bool {
	for restarts := 0; restarts < iterateChunkedMaxRestarts; restarts++ {
		stopped, rehashed := s.iterateChunkedPass(batch, fn)
		if stopped {
			return false
		}
		if !rehashed {
			return true
		}
		s.chunkedRestarts.Add(1)
	}
	// Restart budget spent: one locked pass, which cannot be restarted and so
	// always terminates.
	return s.iterate(func(key, value []byte, _ uint64) bool { return fn(key, value) })
}

// iterateChunkedPass makes one attempt at walking the shard. It reports whether
// fn stopped the iteration, and whether the pass was abandoned because the index
// table was swapped underneath it.
//
// The body matches shard.iterate's filters — same expiry test against the shard's
// wall clock, same tombstone guard — so the visited set matches Iterate's for a
// quiescent cache. Two things differ, both because the lock is released between
// chunks: one RLock per chunk instead of one for the whole shard (following
// sweepIndex's batching), and page resolution through the generation-gated
// pageSlots rather than s.pages (see the gate below).
func (s *shard) iterateChunkedPass(batch int, fn func(key, value []byte) bool) (stopped, rehashed bool) {
	now := s.now()
	t := s.tab.Load()
	n := len(t.ctrl)
	for start := 0; start < n; start += batch {
		end := start + batch
		if end > n {
			end = n
		}
		s.mu.RLock()
		if s.tab.Load() != t {
			// A concurrent Put (or compaction) rehashed the table: every slot index
			// we have left is meaningless against the new one. Abandon and restart.
			s.mu.RUnlock()
			return false, true
		}
		for i := start; i < end; i++ {
			c := t.ctrl[i].Load()
			if c == ctrlEmpty || c == ctrlTombstone {
				continue
			}
			ref := slabRef(t.refs[i].Load())
			// Resolve through pageSlots WITH THE GENERATION GATE, exactly as
			// indexTable.get does — not through s.pages, which Iterate can get away
			// with because it never lets go of the shard. This walk DOES let go
			// between chunks, so a page can be retired and replaced by a fresh object
			// mid-walk (ringbuf eviction, expired-page reclaim, online recycle). A
			// slot still pointing into the old content would then decode against the
			// NEW page's bytes and yield a bogus key/value pair. A generation mismatch
			// says that physical entry is gone: skip it.
			page := s.pageSlots[ref.pageIdx()].Load()
			if page == nil || page.gen != ref.gen() {
				continue
			}
			k, v, exp, err := page.Read(ref.offset())
			if err != nil {
				continue
			}
			if isExpired(exp, now) {
				continue
			}
			if meta, mok := page.MetaAt(ref.offset()); mok && metaIsTombstone(meta) {
				continue
			}
			if !fn(k, v) {
				s.mu.RUnlock()
				return true, false
			}
		}
		s.mu.RUnlock()
	}
	return false, false
}
