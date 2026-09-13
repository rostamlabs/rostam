// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

// Tests for Config.InPlaceSeqlockReads — reading a shard whose writers rewrite
// live entries WITHOUT taking the shard read lock, by validating every read
// against a per-stripe version counter.
//
// WHY SEVERAL OF THESE SKIP UNDER -race. A seqlock reads the payload while a
// writer may be writing it and throws the result away if the version moved; the
// concurrent access is the mechanism, not a defect in it. The race detector
// reports that access, correctly and by design, so the tests that actually
// exercise the protocol concurrently cannot run under -race. Everything that can
// be checked without a live writer — the guards, the counters, the fallback —
// runs in both modes. See cache/seqlock.go for why the race-free form (atomic
// word access on both sides) is not affordable.

func seqlockCfg() Config {
	cfg := inPlaceCfg(true)
	cfg.InPlaceSeqlockReads = true
	return cfg
}

func newSeqlockShard(t *testing.T) *shard {
	t.Helper()
	s, err := newShard(seqlockCfg(), "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// TestSeqlockReleasesTheReadLock is the whole point of the feature, stated as an
// assertion: in-place updates alone force the read-locked path, and adding the
// seqlock takes them back off it. mmap ringbuf stays on it either way — its
// reads hand out aliases that outlive the read, which no amount of validation
// during the read makes safe.
func TestSeqlockReleasesTheReadLock(t *testing.T) {
	lockedS, err := newShard(inPlaceCfg(true), "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = lockedS.Close() })
	if !lockedS.needsReadLockForGet() {
		t.Fatal("in-place updates without the seqlock must take the read lock")
	}
	if lockedS.seqlockReads() {
		t.Fatal("seqlockReads is on with InPlaceSeqlockReads unset")
	}

	seqS := newSeqlockShard(t)
	if seqS.needsReadLockForGet() {
		t.Fatal("the seqlock must release the read lock")
	}
	if !seqS.seqlockReads() {
		t.Fatal("seqlockReads is off with InPlaceSeqlockReads set")
	}

	// mmap ringbuf keeps the lock even with both flags set.
	mm, err := newShard(seqlockCfg(), t.TempDir(), nil)
	if err != nil {
		t.Fatalf("newShard(mmap): %v", err)
	}
	t.Cleanup(func() { _ = mm.Close() })
	if !mm.needsReadLockForGet() {
		t.Fatal("an mmap ringbuf shard must keep the read lock whatever the seqlock says")
	}
	if mm.seqlockReads() {
		t.Fatal("an mmap shard must not read through the seqlock")
	}
}

// TestSeqlockNeedsInPlaceUpdates: the flag alone does nothing. Without in-place
// updates no writer rewrites a live entry, so there is no hazard to validate
// against and the reads were already lock-free.
func TestSeqlockNeedsInPlaceUpdates(t *testing.T) {
	cfg := inPlaceCfg(false)
	cfg.InPlaceSeqlockReads = true
	s, err := newShard(cfg, "", nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	if s.seqlockReads() {
		t.Fatal("the seqlock engaged without in-place updates to protect against")
	}
	if s.needsReadLockForGet() {
		t.Fatal("a plain heap ringbuf shard must still read lock-free")
	}
}

// TestSeqlockVersionBracketsTheWrite pins the writer half of the protocol: the
// counter is EVEN at rest, ODD for exactly the window in which the bytes are
// being written, and advances by two per write. Odd-means-in-flight is what lets
// a reader reject a read it started mid-write without comparing anything.
func TestSeqlockVersionBracketsTheWrite(t *testing.T) {
	s := newSeqlockShard(t)
	key := []byte("hot")
	if err := s.Put(key, []byte("aaaa"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	ref, ok := refFor(t, s, key)
	if !ok {
		t.Fatal("key not indexed")
	}

	s.mu.Lock()
	p := s.pages[ref.pageIdx()]
	ver := p.versionAt(ref.offset())
	if ver == nil {
		s.mu.Unlock()
		t.Fatal("a seqlock shard's heap page has no version counters")
	}
	before := ver.Load()
	if before%2 != 0 {
		s.mu.Unlock()
		t.Fatalf("version %d is odd at rest; readers would retry forever", before)
	}
	var inside uint64
	err := s.bumpVersionLocked(p, ref.offset(), func() error {
		inside = ver.Load()
		return nil
	})
	after := ver.Load()
	s.mu.Unlock()

	if err != nil {
		t.Fatalf("bumpVersionLocked: %v", err)
	}
	if inside%2 == 0 {
		t.Fatalf("version %d is even during the write; a reader could not tell it was mid-flight", inside)
	}
	if after != before+2 {
		t.Fatalf("version went %d -> %d, want +2", before, after)
	}
}

// TestSeqlockPutAdvancesTheVersion checks the bracket is actually wired into the
// write path and not merely available to it.
func TestSeqlockPutAdvancesTheVersion(t *testing.T) {
	s := newSeqlockShard(t)
	key := []byte("hot")
	if err := s.Put(key, []byte("aaaa"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	ref, _ := refFor(t, s, key)
	s.mu.RLock()
	ver := s.pages[ref.pageIdx()].versionAt(ref.offset())
	before := ver.Load()
	s.mu.RUnlock()

	for i := range 10 {
		if err := s.Put(key, fmt.Appendf(nil, "%04d", i), 0); err != nil {
			t.Fatalf("rewrite %d: %v", i, err)
		}
	}
	s.mu.RLock()
	after := ver.Load()
	s.mu.RUnlock()
	if after != before+20 {
		t.Fatalf("version advanced by %d over 10 in-place rewrites, want 20", after-before)
	}
	if n := s.inPlaceUpdates.Load(); n != 10 {
		t.Fatalf("inPlaceUpdates = %d, want 10", n)
	}
}

// TestSeqlockFallsBackToTheReadLock exercises the exhaustion path without racing
// anything: a page stripped of its version counters cannot be validated, so
// getSeq refuses it and the read has to take the lock. The value must still come
// back correct — a fallback is a slower read, never a wrong or missing one.
func TestSeqlockFallsBackToTheReadLock(t *testing.T) {
	s := newSeqlockShard(t)
	key := []byte("hot")
	if err := s.Put(key, []byte("value123"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}
	ref, _ := refFor(t, s, key)

	s.mu.Lock()
	s.pages[ref.pageIdx()].vers = nil // no counters: nothing to validate against
	s.mu.Unlock()

	got, err := s.Get(key)
	if err != nil {
		t.Fatalf("Get through the fallback: %v", err)
	}
	if !bytes.Equal(got, []byte("value123")) {
		t.Fatalf("value = %q, want %q", got, "value123")
	}
	into, err := s.getIntoH(nil, key, hashKey(key))
	if err != nil {
		t.Fatalf("GetInto through the fallback: %v", err)
	}
	if !bytes.Equal(into, []byte("value123")) {
		t.Fatalf("GetInto value = %q, want %q", into, "value123")
	}
	st := s.snapshot()
	if st.SeqlockFallbacks != 2 {
		t.Fatalf("SeqlockFallbacks = %d, want 2", st.SeqlockFallbacks)
	}
	if st.SeqlockRetries != 2*seqlockMaxRetries {
		t.Fatalf("SeqlockRetries = %d, want %d", st.SeqlockRetries, 2*seqlockMaxRetries)
	}
	// The fallback resolved the key, so it must not have been counted as a miss.
	if st.Misses != 0 {
		t.Fatalf("Misses = %d after two successful fallback reads, want 0", st.Misses)
	}
}

// TestSeqlockMissIsNotAFallback: a genuinely absent key resolves on the
// lock-free path and reports a miss, without spending the retry budget. The two
// outcomes share a return value in getSeq and confusing them would either turn
// misses into lock acquisitions or, far worse, turn present keys into misses.
func TestSeqlockMissIsNotAFallback(t *testing.T) {
	s := newSeqlockShard(t)
	if err := s.Put([]byte("present"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := s.Get([]byte("absent")); err != ErrNotFound {
		t.Fatalf("Get(absent) = %v, want %v", err, ErrNotFound)
	}
	st := s.snapshot()
	if st.SeqlockFallbacks != 0 {
		t.Fatalf("SeqlockFallbacks = %d for a plain miss, want 0", st.SeqlockFallbacks)
	}
	if st.SeqlockRetries != 0 {
		t.Fatalf("SeqlockRetries = %d for a plain miss, want 0", st.SeqlockRetries)
	}
	if st.Misses != 1 {
		t.Fatalf("Misses = %d, want 1", st.Misses)
	}
}

// TestSeqlockReadsAreCorrectUnderRewrite is the seqlock's own torn-value control
// and the counterpart of TestInPlaceConcurrentReadersSeeNoTornValue: same shape,
// same uniform-byte encoding, but the readers take NO LOCK and rely entirely on
// the version check. Disable the version bump and this reports torn values.
//
// It skips under -race because the protocol's payload access is a deliberate
// race (see the file comment). That is a real gap in coverage and worth naming:
// under -race this protocol is unverified, which is part of why it is off by
// default.
func TestSeqlockReadsAreCorrectUnderRewrite(t *testing.T) {
	if raceEnabled {
		t.Skip("a seqlock races on the payload by construction; the detector reports it correctly")
	}
	const (
		valLen  = 4096
		readers = 8
		writes  = 200_000
	)
	s := newSeqlockShard(t)
	if s.needsReadLockForGet() {
		t.Fatal("the seqlock shard is taking the read lock; the test would prove nothing")
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
						first.CompareAndSwap(nil, fmt.Sprintf("byte %d = %#x, want %#x (torn)", i, b, want))
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
	st := s.snapshot()
	if st.InPlaceUpdates == 0 {
		t.Fatal("no write took the in-place path, so nothing was exercised")
	}
	if st.SeqlockRetries == 0 {
		t.Fatal("no read ever retried; the writer never collided with a reader and the protocol went untested")
	}
	t.Logf("gets=%d retries=%d fallbacks=%d inplace=%d",
		st.Gets, st.SeqlockRetries, st.SeqlockFallbacks, st.InPlaceUpdates)
}

// TestSeqlockManyKeysStayConsistent widens the previous test from one hot key to
// a working set, so the STRIPE mapping is exercised: distinct keys share a
// version counter whenever their offsets collide, and a reader of one must never
// be handed another's bytes or a stale answer.
func TestSeqlockManyKeysStayConsistent(t *testing.T) {
	if raceEnabled {
		t.Skip("a seqlock races on the payload by construction; the detector reports it correctly")
	}
	const (
		keys    = 256 // < 256+1 so byte(i) tags each key's value uniquely
		valLen  = 512
		readers = 8
		writes  = 200_000
	)
	s := newSeqlockShard(t)
	keyFor := func(i int) []byte { return fmt.Appendf(nil, "k%04d", i) }
	valFor := func(i int) []byte { return bytes.Repeat([]byte{byte(i)}, valLen) }
	for i := range keys {
		if err := s.Put(keyFor(i), valFor(i), 0); err != nil {
			t.Fatalf("seed Put %d: %v", i, err)
		}
	}

	var (
		stop  atomic.Bool
		bad   atomic.Int64
		first atomic.Value
		wg    sync.WaitGroup
	)
	for r := range readers {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			n := seed
			for !stop.Load() {
				i := n % keys
				n++
				v, err := s.Get(keyFor(i))
				if err != nil {
					continue
				}
				// Every value is a uniform run of byte(i): a torn read shows mixed
				// bytes, and another key's bytes show a uniform run of the wrong tag.
				if len(v) != valLen {
					bad.Add(1)
					first.CompareAndSwap(nil, fmt.Sprintf("key %d: length %d", i, len(v)))
					continue
				}
				for j, b := range v {
					if b != byte(i) {
						bad.Add(1)
						first.CompareAndSwap(nil, fmt.Sprintf("key %d byte %d = %#x, want %#x", i, j, b, byte(i)))
						break
					}
				}
			}
		}(r * 37)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for n := range writes {
			i := n % keys
			if err := s.Put(keyFor(i), valFor(i), 0); err != nil {
				t.Errorf("Put: %v", err)
				break
			}
		}
		stop.Store(true)
	}()
	wg.Wait()

	if n := bad.Load(); n != 0 {
		t.Fatalf("%d malformed values observed; first: %v", n, first.Load())
	}
	st := s.snapshot()
	if st.InPlaceUpdates == 0 {
		t.Fatal("no write took the in-place path")
	}
	t.Logf("gets=%d retries=%d fallbacks=%d inplace=%d/%d",
		st.Gets, st.SeqlockRetries, st.SeqlockFallbacks, st.InPlaceUpdates, st.Puts)
}
