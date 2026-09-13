// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"sync/atomic"
	"testing"
	"time"
)

// The expired-on-read drop, against in-place same-size updates.
//
// A read that finds an entry expired releases the shard lock before it removes
// the index slot, so between the two a writer can replace that entry. The guard
// that has always made this safe is cur == ref: every Put used to write a NEW
// copy, so an overwrite always changed the ref and the drop became a no-op.
//
// In-place updates break exactly that. They keep the page, the offset and the
// generation, so the ref is identical after a rewrite and cur == ref can no
// longer tell "nobody touched this" from "somebody replaced it where it lay".
// A Put that refreshes an expired key's TTL therefore leaves a reader holding a
// ref that still matches, and the drop tombstones a key that is live.

// probeExpired runs the FIRST half of a read: it resolves the key exactly as the
// read path does and returns the ref and the clock reading the expiry decision
// rests on. The caller then plays a writer, and finally runs the second half
// (dropExpiredLocked) with what the probe captured.
func probeExpired(t *testing.T, s *shard, key []byte, now uint64) slabRef {
	t.Helper()
	_, exp, ref, st := s.tab.Load().get(s, key, hashKey(key))
	if st != lkHit {
		t.Fatalf("probe status = %v, want a hit", st)
	}
	if !isExpired(exp, now) {
		t.Fatal("the seeded entry is not expired at the probe clock; the test would prove nothing")
	}
	return ref
}

// TestDropExpiredDoesNotDiscardAnInPlaceRefresh reproduces the interleaving
// deterministically by running the read path's two halves around a writer,
// rather than by racing goroutines and hoping.
//
// IT NEEDS CLOCK SKEW, and that is the whole reason the revalidation exists
// rather than guard 6 alone. Guard 6 refuses to rewrite a copy the WRITER judges
// expired, so at a single clock reading the in-place path is unreachable here and
// the refresh would append, changing the ref and making cur == ref sufficient
// again. What guard 6 cannot rule out is the reader and the writer judging the
// same copy differently — this shard takes an injected clock and an apply path
// stamped by a leader, so that is a real ordering, not a contrived one. The
// writer below runs at an EARLIER reading than the reader: it sees a live copy
// and rewrites it in place, while the reader has already decided it was expired.
func TestDropExpiredDoesNotDiscardAnInPlaceRefresh(t *testing.T) {
	s := newInPlaceShard(t, true)
	clock := &fakeClock{}
	const (
		writerAt = 1_000_000 // the writer's reading: the seeded copy is still live
		readerAt = 1_002_000 // the reader's reading: it has expired
	)
	clock.set(writerAt)
	s.setNowForTest(clock.now)

	key := []byte("hot")
	if err := s.Put(key, bytes.Repeat([]byte("a"), 64), time.Second); err != nil {
		t.Fatalf("seed Put: %v", err)
	}

	// HALF ONE: the reader probes at its own, later reading and finds it expired.
	clock.set(readerAt)
	ref := probeExpired(t, s, key, readerAt)

	// BETWEEN THE HALVES: the writer refreshes the key at its earlier reading, so
	// guard 6 sees a live copy and the rewrite lands IN PLACE — same page, same
	// offset, same generation, so the reader's ref still matches it.
	clock.set(writerAt)
	fresh := bytes.Repeat([]byte("b"), 64)
	if err := s.Put(key, fresh, time.Hour); err != nil {
		t.Fatalf("refresh Put: %v", err)
	}
	if n := s.inPlaceUpdates.Load(); n != 1 {
		t.Fatalf("inPlaceUpdates = %d, want 1; the refresh appended, so the ref changed and the interleaving under test did not occur", n)
	}
	if after, _ := refFor(t, s, key); after != ref {
		t.Fatalf("the refresh moved the entry (%#x -> %#x); cur == ref would have caught it", uint64(ref), uint64(after))
	}

	// HALF TWO: the drop, with the ref AND the clock reading the probe captured.
	s.dropExpiredLocked(key, hashKey(key), ref, readerAt)

	// The key was refreshed by a Put that returned nil. It must still be there.
	clock.set(readerAt)
	got, err := s.Get(key)
	if err != nil {
		t.Fatalf("the refreshed key was dropped by a read that saw its PREVIOUS value expire: %v", err)
	}
	if !bytes.Equal(got, fresh) {
		t.Fatalf("value = %q, want %q", got, fresh)
	}
	if n := s.expirations.Load(); n != 0 {
		t.Fatalf("expirations = %d; a live key was counted as expired", n)
	}
}

// TestDropExpiredDoesNotFireOnRemoveForARefreshedKey is the derived-index half of
// the same bug, and the more damaging one: a dropped slot is recoverable by
// writing the key again, but a posting dropped for a key that stays live is the
// missing-row case verify-on-read cannot repair.
func TestDropExpiredDoesNotFireOnRemoveForARefreshedKey(t *testing.T) {
	c, rec := newOnRemoveCache(t, inPlaceCfg(true))
	clock := &fakeClock{}
	const (
		writerAt = 1_000_000
		readerAt = 1_002_000
	)
	clock.set(writerAt)
	c.SetNowFunc(clock.now)
	s := c.shards[0]

	key := []byte("hot")
	if err := c.Put(key, bytes.Repeat([]byte("a"), 64), time.Second); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	clock.set(readerAt)
	ref := probeExpired(t, s, key, readerAt)

	clock.set(writerAt)
	if err := c.Put(key, bytes.Repeat([]byte("b"), 64), time.Hour); err != nil {
		t.Fatalf("refresh Put: %v", err)
	}
	if n := s.inPlaceUpdates.Load(); n != 1 {
		t.Fatalf("inPlaceUpdates = %d, want 1; the interleaving under test did not occur", n)
	}
	s.dropExpiredLocked(key, hashKey(key), ref, readerAt)

	if n := rec.count("hot"); n != 0 {
		t.Fatalf("onRemove fired %d times for a key that is live; every derived index just dropped its postings", n)
	}
}

// TestInPlaceRefusesAnExpiredStoredCopy covers guard 6 on its own: when the
// writer's own clock says the stored copy is dead, the refresh must APPEND rather
// than rewrite it where it lies. Two reasons, and the first is the one that
// justifies the guard by itself — rewriting a dead copy in place pins those bytes
// instead of leaving them behind to be reclaimed. The second is that appending
// mints a new ref, which is what a concurrent expired-on-read drop compares
// against, so the common case never reaches the revalidation at all.
func TestInPlaceRefusesAnExpiredStoredCopy(t *testing.T) {
	s := newInPlaceShard(t, true)
	clock := &fakeClock{}
	clock.set(1_000_000)
	s.setNowForTest(clock.now)

	key := []byte("hot")
	if err := s.Put(key, bytes.Repeat([]byte("a"), 64), time.Second); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	before, _ := refFor(t, s, key)

	clock.set(1_002_000) // the stored copy is now dead at the writer's own clock
	if err := s.Put(key, bytes.Repeat([]byte("b"), 64), time.Hour); err != nil {
		t.Fatalf("refresh Put: %v", err)
	}
	if n := s.inPlaceUpdates.Load(); n != 0 {
		t.Fatalf("inPlaceUpdates = %d; an expired copy was rewritten where it lay", n)
	}
	after, ok := refFor(t, s, key)
	if !ok {
		t.Fatal("the key lost its index slot")
	}
	if after == before {
		t.Fatal("the refresh did not append; guard 6 did not fire")
	}
	got, err := s.Get(key)
	if err != nil {
		t.Fatalf("Get after the refresh: %v", err)
	}
	if !bytes.Equal(got, bytes.Repeat([]byte("b"), 64)) {
		t.Fatal("the refreshed value did not survive the append")
	}
}

// TestDropExpiredStillReclaimsAGenuinelyExpiredEntry is the control for both of
// the above. The revalidation must not turn the drop into a no-op: an entry that
// really is still expired when the lock is taken has to be reclaimed, or lazy
// expiry stops working and expired keys accumulate until the sweeper runs.
func TestDropExpiredStillReclaimsAGenuinelyExpiredEntry(t *testing.T) {
	s := newInPlaceShard(t, true)
	clock := &fakeClock{}
	clock.set(1_000_000)
	s.setNowForTest(clock.now)

	key := []byte("cold")
	if err := s.Put(key, bytes.Repeat([]byte("a"), 64), time.Second); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	clock.set(1_002_000)

	// The ordinary read path: it must report the miss AND reclaim the slot.
	if _, err := s.Get(key); err != ErrNotFound {
		t.Fatalf("Get on an expired key = %v, want %v", err, ErrNotFound)
	}
	if n := s.expirations.Load(); n != 1 {
		t.Fatalf("expirations = %d, want 1; the expired entry was not reclaimed", n)
	}
	if _, ok := refFor(t, s, key); ok {
		t.Fatal("the expired key still holds an index slot")
	}
}

// fakeClock is a settable millisecond clock for the expiry sites that consult
// Config.NowFn. Stored atomically because the read path may consult it from any
// goroutine.
type fakeClock struct{ ms atomic.Uint64 }

func (c *fakeClock) set(ms uint64) { c.ms.Store(ms) }
func (c *fakeClock) now() uint64   { return c.ms.Load() }

// setNowForTest points a bare shard (one built by newShard rather than through a
// Cache) at an injected clock. Cache.SetNowFunc is the exported route and sets
// every shard at once; this is its single-shard equivalent.
func (s *shard) setNowForTest(fn func() uint64) {
	f := fn
	s.nowFn.Store(&f)
}
