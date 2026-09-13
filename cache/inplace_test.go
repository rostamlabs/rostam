// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Tests for Config.InPlaceSameSizeUpdate — a write that overwrites the copy
// already stored for its key rather than appending a new one after it.
//
// Every guard has a NEGATIVE CONTROL here: a case that trips exactly that guard
// and must therefore fall back to appending. A guard nothing can trip is a guard
// that is not being tested.

// inPlaceCfg is a single heap shard with the background sweeper off, small
// enough that eviction is reachable, with the feature in the requested state.
func inPlaceCfg(on bool) Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 2 << 20 // 2 pages
	cfg.TTLSweepIntervalMs = 0
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.InPlaceSameSizeUpdate = on
	return cfg
}

func newInPlaceShard(t *testing.T, on bool) *shard {
	t.Helper()
	s, err := newShard(inPlaceCfg(on), "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// refFor returns the slabRef the index currently holds for key, and whether the
// key is indexed at all. It is how these tests assert that a write did or did
// not MOVE the stored copy: an in-place update leaves the ref identical (same
// page, same offset, same generation), an append changes it.
func refFor(t *testing.T, s *shard, key []byte) (slabRef, bool) {
	t.Helper()
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ref, ok := s.tab.Load().findSlot(hashKey(key))
	return ref, ok
}

// TestInPlaceUpdatesValueAndExpiry is the base case: the new value and the new
// expiry are both actually stored and read back, and the entry did not move.
func TestInPlaceUpdatesValueAndExpiry(t *testing.T) {
	s := newInPlaceShard(t, true)
	key := []byte("hot")

	if err := s.Put(key, []byte("aaaaaaaa"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	first, ok := refFor(t, s, key)
	if !ok {
		t.Fatal("key not indexed after seed Put")
	}
	if _, _, err := s.getWithExpiryH(key, hashKey(key)); err != nil {
		t.Fatalf("seed Get: %v", err)
	}

	if err := s.Put(key, []byte("bbbbbbbb"), time.Hour); err != nil {
		t.Fatalf("rewrite Put: %v", err)
	}
	if n := s.inPlaceUpdates.Load(); n != 1 {
		t.Fatalf("inPlaceUpdates = %d, want 1 (the rewrite should have landed in place)", n)
	}
	got, exp, err := s.getWithExpiryH(key, hashKey(key))
	if err != nil {
		t.Fatalf("Get after rewrite: %v", err)
	}
	if !bytes.Equal(got, []byte("bbbbbbbb")) {
		t.Fatalf("value = %q, want %q", got, "bbbbbbbb")
	}
	if exp == 0 {
		t.Fatal("expiry = 0 after a rewrite that set a one-hour TTL")
	}
	after, _ := refFor(t, s, key)
	if after != first {
		t.Fatalf("slabRef moved on an in-place update: %#x -> %#x", uint64(first), uint64(after))
	}

	// And the TTL is genuinely the stored one, not a leftover: rewriting back to
	// no-expiry must clear it.
	if err := s.Put(key, []byte("cccccccc"), 0); err != nil {
		t.Fatalf("third Put: %v", err)
	}
	if _, exp, err = s.getWithExpiryH(key, hashKey(key)); err != nil {
		t.Fatalf("Get after third Put: %v", err)
	}
	if exp != 0 {
		t.Fatalf("expiry = %d after a rewrite with no TTL, want 0", exp)
	}
}

// TestInPlaceBytesUsedFlatAcrossRewrites is the headline property: many
// same-size rewrites of one key consume no additional page bytes, while the
// write counter climbs. Today every one of those writes would strand a dead copy.
func TestInPlaceBytesUsedFlatAcrossRewrites(t *testing.T) {
	const rewrites = 5_000
	for _, on := range []bool{false, true} {
		t.Run(fmt.Sprintf("inplace=%v", on), func(t *testing.T) {
			s := newInPlaceShard(t, on)
			key := []byte("hot")
			val := bytes.Repeat([]byte("x"), 64)
			if err := s.Put(key, val, 0); err != nil {
				t.Fatalf("seed Put: %v", err)
			}
			base := s.snapshot()

			for i := range rewrites {
				val[0] = byte(i)
				if err := s.Put(key, val, 0); err != nil {
					t.Fatalf("rewrite %d: %v", i, err)
				}
			}
			st := s.snapshot()
			if st.Puts-base.Puts != rewrites {
				t.Fatalf("Puts advanced by %d, want %d", st.Puts-base.Puts, rewrites)
			}
			grew := st.BytesUsed - base.BytesUsed
			if on {
				if grew != 0 {
					t.Fatalf("BytesUsed grew by %d over %d same-size rewrites, want 0", grew, rewrites)
				}
				if st.InPlaceUpdates != rewrites {
					t.Fatalf("InPlaceUpdates = %d, want %d", st.InPlaceUpdates, rewrites)
				}
			} else {
				// The control: today's append path pays for every rewrite.
				want := uint64(rewrites * entrySize(len(key), len(val)))
				if grew != want {
					t.Fatalf("append path grew BytesUsed by %d over %d rewrites, want %d", grew, rewrites, want)
				}
				if st.InPlaceUpdates != 0 {
					t.Fatalf("InPlaceUpdates = %d with the flag off, want 0", st.InPlaceUpdates)
				}
			}
		})
	}
}

// TestInPlaceTriggersNoEviction pins the consequence of the property above: a
// shard whose live set fits its budget never reaches capacity, so rewriting it
// forever evicts nothing. The same run with the flag off evicts continuously,
// which is what makes the assertion meaningful.
func TestInPlaceTriggersNoEviction(t *testing.T) {
	const (
		keys     = 200
		rewrites = 40
	)
	val := bytes.Repeat([]byte("v"), 512)
	keyFor := func(i int) []byte { return fmt.Appendf(nil, "k%05d", i) }

	run := func(on bool) Stats {
		s := newInPlaceShard(t, on)
		for r := range rewrites {
			for i := range keys {
				val[0] = byte(r)
				if err := s.Put(keyFor(i), val, 0); err != nil {
					t.Fatalf("Put: %v", err)
				}
			}
		}
		return s.snapshot()
	}

	on := run(true)
	if on.Evictions != 0 {
		t.Fatalf("in-place run evicted %d entries, want 0", on.Evictions)
	}
	if on.Entries != keys {
		t.Fatalf("in-place run retained %d keys, want %d", on.Entries, keys)
	}
	off := run(false)
	if off.Evictions == 0 {
		t.Fatal("control (flag off) evicted nothing, so the in-place assertion proves nothing")
	}
}

// TestInPlaceKeepsEntryIndexCurrent checks that the entry a same-size update
// rewrote is still the one the index resolves to, and that the index itself did
// not gain a slot or a tombstone. The ops layer reposts the KV-index entry after
// every write that goes through the cache's put body (TxContext.PutIndexed), so
// what the cache owes is that the key stays live and addressable at the same ref.
func TestInPlaceKeepsEntryIndexCurrent(t *testing.T) {
	s := newInPlaceShard(t, true)
	key := []byte("hot")
	if err := s.Put(key, []byte("0000"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	seed := s.snapshot()
	first, _ := refFor(t, s, key)

	for i := range 100 {
		if err := s.Put(key, fmt.Appendf(nil, "%04d", i), 0); err != nil {
			t.Fatalf("rewrite %d: %v", i, err)
		}
		ref, ok := refFor(t, s, key)
		if !ok {
			t.Fatalf("rewrite %d dropped the key from the index", i)
		}
		if ref != first {
			t.Fatalf("rewrite %d moved the entry: %#x -> %#x", i, uint64(first), uint64(ref))
		}
		got, err := s.Get(key)
		if err != nil {
			t.Fatalf("Get after rewrite %d: %v", i, err)
		}
		if want := fmt.Appendf(nil, "%04d", i); !bytes.Equal(got, want) {
			t.Fatalf("rewrite %d: value = %q, want %q", i, got, want)
		}
	}
	st := s.snapshot()
	if st.Entries != seed.Entries {
		t.Fatalf("Entries = %d after rewrites, want %d", st.Entries, seed.Entries)
	}
	if st.Tombstones != seed.Tombstones {
		t.Fatalf("Tombstones = %d after rewrites, want %d", st.Tombstones, seed.Tombstones)
	}
}

// TestInPlaceDoesNotFireOnRemove. An in-place update removes nothing — the key
// stays live at the same address — so the hook must stay silent. Firing it would
// drop a live key's postings from every derived index, the one failure mode
// verify-on-read cannot repair.
func TestInPlaceDoesNotFireOnRemove(t *testing.T) {
	cfg := inPlaceCfg(true)
	c, rec := newOnRemoveCache(t, cfg)
	key := []byte("hot")
	val := bytes.Repeat([]byte("v"), 128)
	if err := c.Put(key, val, 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	for i := range 2000 {
		val[0] = byte(i)
		if err := c.Put(key, val, 0); err != nil {
			t.Fatalf("rewrite %d: %v", i, err)
		}
	}
	if n := rec.len(); n != 0 {
		t.Fatalf("onRemove fired %d times for in-place updates, want 0: %v", n, rec.snapshot())
	}
	st := c.Stats()
	if st.InPlaceUpdates != 2000 {
		t.Fatalf("InPlaceUpdates = %d, want 2000 (the rewrites did not take the in-place path)", st.InPlaceUpdates)
	}
}

// TestInPlaceFlagOffIsTheAppendPath is the whole-feature negative control: with
// the flag off nothing about a rewrite changes. The stored copy MOVES on every
// write (the append path's signature), page bytes accumulate, the counter stays
// at zero, and — the part that matters for latency — reads stay lock-free.
func TestInPlaceFlagOffIsTheAppendPath(t *testing.T) {
	s := newInPlaceShard(t, false)
	if s.needsReadLockForGet() {
		t.Fatal("a heap ringbuf shard with the flag off must still read lock-free")
	}
	key := []byte("hot")
	val := bytes.Repeat([]byte("v"), 64)
	if err := s.Put(key, val, 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	prev, _ := refFor(t, s, key)
	for i := range 50 {
		val[0] = byte(i)
		if err := s.Put(key, val, 0); err != nil {
			t.Fatalf("rewrite %d: %v", i, err)
		}
		ref, ok := refFor(t, s, key)
		if !ok {
			t.Fatalf("rewrite %d dropped the key", i)
		}
		if ref == prev {
			t.Fatalf("rewrite %d did not move the entry; the append path must write a new copy", i)
		}
		prev = ref
	}
	if n := s.inPlaceUpdates.Load(); n != 0 {
		t.Fatalf("inPlaceUpdates = %d with the flag off, want 0", n)
	}
}

// TestInPlaceGuards is one negative control per guard: each case trips exactly
// one of them, and every one must fall back to appending — proved by the entry
// MOVING and the counter staying at zero.
func TestInPlaceGuards(t *testing.T) {
	t.Run("guard2/reject-writes", func(t *testing.T) {
		// Reject-writes shards hand out zero-copy aliases that outlive the read, so
		// their live bytes may never be overwritten — even with the flag on.
		cfg := inPlaceCfg(true)
		cfg.AtCapPolicy = PolicyRejectWrites
		s, err := newShard(cfg, "", nil)
		if err != nil {
			t.Fatalf("newShard: %v", err)
		}
		t.Cleanup(func() { _ = s.Close() })
		if s.inPlaceEligible() {
			t.Fatal("a reject-writes shard must not be eligible")
		}
		assertRewriteAppends(t, s)
	})

	t.Run("guard3/mmap", func(t *testing.T) {
		// An mmap page is the durable copy: an overwrite torn by a crash loses the
		// key, where an append leaves the previous version recoverable.
		cfg := inPlaceCfg(true)
		s, err := newShard(cfg, t.TempDir(), nil)
		if err != nil {
			t.Fatalf("newShard: %v", err)
		}
		t.Cleanup(func() { _ = s.Close() })
		if !s.isMmap {
			t.Fatal("shard did not come up mmap-backed; the control is not testing the guard")
		}
		if s.inPlaceEligible() {
			t.Fatal("an mmap shard must not be eligible")
		}
		assertRewriteAppends(t, s)
		// And the page refuses the write even if a caller reached it directly.
		if err := s.pages[0].WriteAt(0, []byte("k"), []byte("v"), 0, 0); err != errPageNotOverwritable {
			t.Fatalf("mmap page WriteAt error = %v, want %v", err, errPageNotOverwritable)
		}
	})

	t.Run("guard4a/key-absent", func(t *testing.T) {
		s := newInPlaceShard(t, true)
		if err := s.Put([]byte("fresh"), []byte("vvvv"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
		if n := s.inPlaceUpdates.Load(); n != 0 {
			t.Fatalf("inPlaceUpdates = %d for a first write of a key, want 0", n)
		}
	})

	t.Run("guard4b/stale-generation", func(t *testing.T) {
		// A ref minted before its page was retired must not resolve. Heap ringbuf
		// eviction swaps in a fresh page object rather than rewriting the old one,
		// so a stale ref names bytes that now belong to a different slab — one a
		// writer may be actively appending to. Reconstructed directly: keep the ref,
		// retire the page under it, then put the stale ref back in the index.
		s := newInPlaceShard(t, true)
		key := []byte("hot")
		if err := s.Put(key, []byte("aaaa"), 0); err != nil {
			t.Fatalf("seed Put: %v", err)
		}
		stale, _ := refFor(t, s, key)

		s.mu.Lock()
		s.retirePageLocked(int(stale.pageIdx()))
		s.tab.Load().upsert(hashKey(key), stale)
		_, _, _, _, ok := s.inPlaceTargetLocked(key, []byte("bbbb"), hashKey(key))
		gen := s.pages[stale.pageIdx()].gen
		s.mu.Unlock()

		if gen == stale.gen() {
			t.Fatal("retiring the page did not change its generation; the control is not testing the guard")
		}
		if ok {
			t.Fatal("a ref from a retired generation was accepted as an in-place target")
		}
	})

	t.Run("guard4c/hash-collision", func(t *testing.T) {
		// Two different keys can share one 64-bit hash, and so one index slot. What
		// that presents to the write path is a lookup by key B's hash that resolves
		// to key A's entry — reproduced here directly, because finding a real
		// 64-bit collision needs a birthday search of ~2^32 keys. Overwriting A's
		// bytes on B's behalf would destroy an unrelated live record.
		s := newInPlaceShard(t, true)
		victim, other := []byte("victim"), []byte("other!") // same length, different bytes
		if err := s.Put(victim, []byte("aaaa"), 0); err != nil {
			t.Fatalf("seed Put: %v", err)
		}
		s.mu.Lock()
		_, _, _, _, ok := s.inPlaceTargetLocked(other, []byte("bbbb"), hashKey(victim))
		s.mu.Unlock()
		if ok {
			t.Fatal("a colliding key was accepted as an in-place target for another key's entry")
		}
		// The same lookup for the key that really owns the entry does resolve, so
		// the refusal above came from the key comparison and not from some other
		// guard tripping first.
		s.mu.Lock()
		_, _, _, _, ok = s.inPlaceTargetLocked(victim, []byte("bbbb"), hashKey(victim))
		s.mu.Unlock()
		if !ok {
			t.Fatal("the owning key was refused too; the control is not isolating the key comparison")
		}
	})

	t.Run("guard4d/outside-live-band", func(t *testing.T) {
		// An offset below head names bytes eviction has already released.
		p := newHeapPage(1 << 20)
		off, _, err := p.Write([]byte("k"), []byte("vvvv"), 0, 0)
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
		if _, _, werr := p.Write([]byte("k2"), []byte("vvvv"), 0, 0); werr != nil {
			t.Fatalf("second Write: %v", werr)
		}
		p.setHead(p.tail()) // everything before tail is now released
		if err := p.WriteAt(off, []byte("k"), []byte("wwww"), 0, 0); err != errEntryTruncated {
			t.Fatalf("WriteAt below head: err = %v, want %v", err, errEntryTruncated)
		}
	})

	t.Run("guard5/different-length", func(t *testing.T) {
		s := newInPlaceShard(t, true)
		key := []byte("hot")
		if err := s.Put(key, []byte("aaaa"), 0); err != nil {
			t.Fatalf("seed Put: %v", err)
		}
		first, _ := refFor(t, s, key)
		if err := s.Put(key, []byte("aaaaa"), 0); err != nil { // one byte longer
			t.Fatalf("rewrite: %v", err)
		}
		if n := s.inPlaceUpdates.Load(); n != 0 {
			t.Fatalf("inPlaceUpdates = %d for a different-length rewrite, want 0", n)
		}
		ref, _ := refFor(t, s, key)
		if ref == first {
			t.Fatal("a different-length rewrite did not append")
		}
		got, err := s.Get(key)
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		if !bytes.Equal(got, []byte("aaaaa")) {
			t.Fatalf("value = %q, want %q", got, "aaaaa")
		}
	})
}

// assertRewriteAppends writes a key twice at the same size and requires the
// second write to have MOVED it — the signature of the append path — and the
// in-place counter to have stayed at zero.
func assertRewriteAppends(t *testing.T, s *shard) {
	t.Helper()
	key := []byte("hot")
	if err := s.Put(key, []byte("aaaa"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	first, ok := refFor(t, s, key)
	if !ok {
		t.Fatal("key not indexed after seed Put")
	}
	if err := s.Put(key, []byte("bbbb"), 0); err != nil {
		t.Fatalf("rewrite Put: %v", err)
	}
	if n := s.inPlaceUpdates.Load(); n != 0 {
		t.Fatalf("inPlaceUpdates = %d on an ineligible shard, want 0", n)
	}
	ref, _ := refFor(t, s, key)
	if ref == first {
		t.Fatal("the rewrite did not append; an ineligible shard must take the append path")
	}
	got, err := s.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("bbbb")) {
		t.Fatalf("value = %q, want %q", got, "bbbb")
	}
}

// TestInPlaceConcurrentReadersSeeNoTornValue is the reason the read-lock
// decision exists, so it must be ABLE TO FAIL: delete the
// InPlaceSameSizeUpdate term from needsReadLockForGet and this test reports torn
// values (and -race reports the underlying data race on the page bytes).
//
// The writer rewrites one key in place, forever, cycling the value through
// uniform byte runs — all 0x00, then all 0x01, and so on. Every value the readers
// see must therefore be uniform. A reader that catches the overwrite half-done
// sees a run of the old byte followed by a run of the new one, which is exactly
// what the uniformity check rejects. Nothing about the append path can produce
// that: it writes a whole new entry and only then republishes the ref.
func TestInPlaceConcurrentReadersSeeNoTornValue(t *testing.T) {
	const (
		valLen  = 4096 // long enough that a torn write is easy to catch mid-copy
		readers = 8
		writes  = 200_000
	)
	s := newInPlaceShard(t, true)
	if !s.needsReadLockForGet() {
		t.Fatal("in-place updates are enabled but the read path is not taking the read lock")
	}
	key := []byte("hot")
	val := make([]byte, valLen)
	if err := s.Put(key, val, 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}

	var (
		stop  atomic.Bool
		torn  atomic.Int64
		first atomic.Value // string
		wg    sync.WaitGroup
	)
	for r := range readers {
		wg.Add(1)
		useInto := r%2 == 0
		go func() {
			defer wg.Done()
			var buf []byte
			for !stop.Load() {
				var v []byte
				var err error
				if useInto {
					v, err = s.getIntoH(buf[:0], key, hashKey(key))
					// Keep the returned slice as the buffer. Without this buf stays nil
					// and buf[:0] hands getIntoH a zero-capacity destination every time,
					// so the "reuse a buffer" reader allocates on every read and the
					// path it is meant to exercise is never actually exercised.
					buf = v
				} else {
					v, err = s.Get(key)
				}
				if err != nil {
					continue
				}
				if len(v) != valLen {
					torn.Add(1)
					first.CompareAndSwap(nil, fmt.Sprintf("length %d, want %d", len(v), valLen))
					continue
				}
				want := v[0]
				for i, b := range v {
					if b != want {
						torn.Add(1)
						first.CompareAndSwap(nil, fmt.Sprintf("byte %d = %#x, want %#x (value is not uniform: torn)", i, b, want))
						break
					}
				}
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range writes {
			b := byte(i)
			for j := range val {
				val[j] = b
			}
			if err := s.Put(key, val, 0); err != nil {
				t.Errorf("Put: %v", err)
				break
			}
		}
		stop.Store(true)
	}()
	wg.Wait()

	if n := torn.Load(); n != 0 {
		t.Fatalf("%d torn/short values observed; first: %v", n, first.Load())
	}
	if got := s.inPlaceUpdates.Load(); got == 0 {
		t.Fatal("no write took the in-place path, so the test exercised nothing")
	}
}
