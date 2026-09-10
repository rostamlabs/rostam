// SPDX-License-Identifier: Apache-2.0

package cache

// Iterate visits every live (non-expired) entry in the cache. The fn callback
// receives the key, value, and absolute expiry timestamp in milliseconds (0 if
// no TTL). Returning false stops iteration early.
//
// The key and value slices alias into the page backing store and remain valid
// only for the duration of the fn call. Copy if you need to retain them.
//
// Iterate walks shards sequentially, taking each shard's read lock only while
// iterating that shard. Writes to other shards proceed without blocking; only
// the shard currently being iterated pauses writers. Expired entries are
// skipped (Iterate does NOT delete them — that's the sweeper's job).
//
// It returns ErrClosed when a shard is closed underneath the walk (see
// shard.Close). A caller that treats the walk's result as COMPLETE — a snapshot,
// a derived index — must check it: the entries visited before the close are a
// proper subset of the keyspace, and publishing them as the whole of it is the
// one answer neither consumer may give. An early stop requested by fn is not an
// error and returns nil.
func (c *Cache) Iterate(fn func(key, value []byte, expiryMs uint64) bool) error {
	for _, s := range c.shards {
		cont, err := s.iterate(fn)
		if err != nil {
			return err
		}
		if !cont {
			return nil
		}
	}
	return nil
}

// iterate walks one shard's index. Reports false when fn asked to stop the whole
// iteration, and ErrClosed when the shard was closed before the walk could take
// its lock. The expiry FILTER uses the shard's wall clock
// (s.now(), honoring an injected test clock) — so a canonical cross-replica
// fingerprint built from Iterate/serializeSnapshot must pin a FIXED clock via
// Cache.SetNowFunc first, otherwise two replicas with byte-identical stored state
// could filter differently and disagree (#4 Phase B).
func (s *shard) iterate(fn func(key, value []byte, expiryMs uint64) bool) (bool, error) {
	now := s.now()
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Checked under the lock Close takes to unmap: past this point the mapping
	// cannot go away while we hold the RLock. See shard.Close.
	if s.closed {
		return false, ErrClosed
	}
	t := s.tab.Load()
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		ref := slabRef(t.refs[i].Load())
		page := s.pages[ref.pageIdx()]
		k, v, exp, err := page.Read(ref.offset())
		if err != nil {
			continue
		}
		if isExpired(exp, now) {
			continue
		}
		// Defense in depth against a delete record leaking into a snapshot. A
		// tombstone is never index-CURRENT at runtime — delH tombstones the slot in
		// the same critical section that appends the record, and the warm-restart
		// rebuild strips any slot a tombstone won — so this is unreachable today.
		// It is checked anyway because Iterate feeds serializeSnapshot, and a
		// tombstone escaping into a snapshot would install a phantom empty-valued
		// key on every follower that restored it.
		if meta, mok := page.MetaAt(ref.offset()); mok && metaIsTombstone(meta) {
			continue
		}
		if !fn(k, v, exp) {
			return false, nil
		}
	}
	return true, nil
}
