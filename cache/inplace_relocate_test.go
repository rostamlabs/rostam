// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"testing"
)

// Where in-place same-size updates meet relocating eviction. The two touch the
// same records from opposite directions — relocation MOVES a record between pages
// and repoints its slot, an in-place update rewrites one WHERE IT LIES — so the
// combinations are worth pinning rather than reasoning about.

// TestInPlaceRelocationCombinations walks the whole in-place x relocation space
// and states what each corner does, because the read path's locking is decided by
// one predicate that both features have an opinion about.
//
// The rule that comes out of it: RELOCATION NEVER AFFECTS THE READ PROTOCOL. It
// appends into a freed page and repoints the slot, leaving the source page frozen
// — exactly the invariant the lock-free path already rests on — so it needs no
// lock and no version check. Only in-place updates rewrite live bytes, so only
// they decide between the three protocols.
func TestInPlaceRelocationCombinations(t *testing.T) {
	for _, tc := range []struct {
		inPlace, seqlock, relocate bool
		wantReadLock               bool // reads take the shard read lock
		wantSeqlock                bool // reads validate against a version counter
		wantEligible               bool // writes may overwrite a record in place
	}{
		{inPlace: false, relocate: false, wantReadLock: false, wantSeqlock: false, wantEligible: false},
		{inPlace: false, relocate: true, wantReadLock: false, wantSeqlock: false, wantEligible: false},
		{inPlace: true, relocate: false, wantReadLock: true, wantSeqlock: false, wantEligible: true},
		{inPlace: true, relocate: true, wantReadLock: true, wantSeqlock: false, wantEligible: true},
		{inPlace: true, seqlock: true, relocate: false, wantReadLock: false, wantSeqlock: true, wantEligible: true},
		{inPlace: true, seqlock: true, relocate: true, wantReadLock: false, wantSeqlock: true, wantEligible: true},
		// The seqlock flag alone protects against a hazard that cannot arise.
		{inPlace: false, seqlock: true, relocate: true, wantReadLock: false, wantSeqlock: false, wantEligible: false},
	} {
		name := fmt.Sprintf("inplace=%v/seqlock=%v/relocate=%v", tc.inPlace, tc.seqlock, tc.relocate)
		t.Run(name, func(t *testing.T) {
			cfg := relocConfig(3, tc.relocate)
			cfg.InPlaceSameSizeUpdate = tc.inPlace
			cfg.InPlaceSeqlockReads = tc.seqlock
			c, err := New(cfg)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer func() { _ = c.Close() }()
			s := c.shards[0]

			if got := s.needsReadLockForGet(); got != tc.wantReadLock {
				t.Errorf("needsReadLockForGet = %v, want %v", got, tc.wantReadLock)
			}
			if got := s.seqlockReads(); got != tc.wantSeqlock {
				t.Errorf("seqlockReads = %v, want %v", got, tc.wantSeqlock)
			}
			if got := s.inPlaceEligible(); got != tc.wantEligible {
				t.Errorf("inPlaceEligible = %v, want %v", got, tc.wantEligible)
			}
			// Exactly one read protocol, always: the read lock and the seqlock are
			// alternatives, never both and never neither-with-a-hazard-present.
			if s.needsReadLockForGet() && s.seqlockReads() {
				t.Error("both read protocols engaged at once")
			}
			if s.inPlaceEligible() && !s.needsReadLockForGet() && !s.seqlockReads() {
				t.Error("writes may rewrite live bytes but reads are unprotected")
			}

			// And the corner actually works: a key round-trips, a same-size rewrite
			// updates it, and a different-size rewrite does too.
			k := relocKey(1)
			mustPut(t, c, k, relocValue(1))
			mustPut(t, c, k, relocValue(2))
			if v := mustGet(t, c, k); !relocUniform(v, 2) {
				t.Fatal("same-size rewrite did not take effect")
			}
			if err := c.Put(k, relocValue(3)[:relocValueLen-1], 0); err != nil {
				t.Fatalf("short rewrite: %v", err)
			}
			v, err := c.Get(k)
			if err != nil {
				t.Fatalf("Get after short rewrite: %v", err)
			}
			if len(v) != relocValueLen-1 {
				t.Fatalf("value length %d after a short rewrite, want %d", len(v), relocValueLen-1)
			}
			st := c.Stats()
			if !tc.wantEligible && st.InPlaceUpdates != 0 {
				t.Fatalf("InPlaceUpdates = %d on an ineligible shard", st.InPlaceUpdates)
			}
			if tc.wantEligible && st.InPlaceUpdates == 0 {
				t.Fatal("no write took the in-place path on an eligible shard")
			}
		})
	}
}

// relocateOneRecord fills a 3-page shard with DISTINCT keys, triggers an
// eviction, and returns a key that relocation actually moved together with its
// ref before and after the move.
//
// The keys are all distinct on purpose. The relocation tests next door seed a
// page with repeated writes of one key so its older copies are dead weight — but
// repeated writes of one key at one size are exactly what an in-place shard stops
// creating, so that layout evaporates here. Discovering which record moved, rather
// than arranging for a known one to, is what makes this work under either setting.
func relocateOneRecord(t *testing.T, c *Cache) (key []byte, before, after slabRef) {
	t.Helper()
	s := c.shards[0]
	const seeded = 3 * relocPerPage
	for i := range seeded {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	if got := s.numPages(); got != 3 {
		t.Fatalf("expected the shard at its 3-page cap, got %d", got)
	}
	refs := make(map[int]slabRef, seeded)
	for i := range seeded {
		if r, ok := refFor(t, s, relocKey(i)); ok {
			refs[i] = r
		}
	}

	// The shard is full, so this evicts the rotation cursor's page and evacuates
	// the live records of the page behind it into the space that frees up.
	mustPut(t, c, relocKey(900), relocValue(9))
	if st := c.Stats(); st.EvictionRelocations == 0 {
		t.Fatal("nothing was relocated; the test would prove nothing")
	}
	for i := range seeded {
		was, had := refs[i]
		now, ok := refFor(t, s, relocKey(i))
		if had && ok && now != was {
			return relocKey(i), was, now
		}
	}
	t.Fatal("relocation was counted but no key changed ref")
	return nil, 0, 0
}

// TestInPlaceLandsOnTheRelocatedCopy is the ordering check where the two features
// actually meet. Relocation copies a live record forward and repoints its slot,
// leaving a STALE copy framed on the source page until that page is retired. An
// in-place update that followed the stale copy would write the new value where
// nothing reads it, and the key would keep serving the old one — a silent lost
// update, and the failure mode worth a test of its own.
//
// It cannot happen, and this pins why: inPlaceTargetLocked resolves through
// findSlot, so it sees the ref relocation just installed, and the generation gate
// rejects a ref into a retired page. Both run under the shard write lock, so
// neither can observe the other half-done.
func TestInPlaceLandsOnTheRelocatedCopy(t *testing.T) {
	cfg := relocConfig(3, true)
	cfg.InPlaceSameSizeUpdate = true
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	moved, beforeRef, afterRef := relocateOneRecord(t, c)
	if afterRef == beforeRef {
		t.Fatal("the record did not move")
	}

	// Rewrite it at the same size. It must land on the RELOCATED copy.
	inPlaceBefore := c.Stats().InPlaceUpdates
	mustPut(t, c, moved, relocValue(42))
	st := c.Stats()
	if st.InPlaceUpdates != inPlaceBefore+1 {
		t.Fatalf("InPlaceUpdates went %d -> %d; the rewrite did not take the in-place path",
			inPlaceBefore, st.InPlaceUpdates)
	}
	finalRef, _ := refFor(t, s, moved)
	if finalRef != afterRef {
		t.Fatalf("the in-place rewrite moved the entry: %#x -> %#x", uint64(afterRef), uint64(finalRef))
	}
	// The read follows the index, so this distinguishes a write that landed on the
	// relocated copy from one that landed on the stale copy behind it.
	if v := mustGet(t, c, moved); !relocUniform(v, 42) {
		t.Fatal("the key does not read back as the value just written: the update landed on the stale copy")
	}

	// Keep writing until the source page has been drained. Had the update gone to
	// the stale copy, the key would now revert or vanish.
	for i := range 2 * relocPerPage {
		mustPut(t, c, relocKey(700+i), relocValue(i))
	}
	if v, gerr := c.Get(moved); gerr != nil {
		t.Fatalf("relocated key lost after its source page drained: %v", gerr)
	} else if !relocUniform(v, 42) {
		t.Fatal("relocated key reverted after its source page drained")
	}
}

// TestSeqlockReadsSurviveRelocation runs the lock-free protocol across the move.
// A reader can resolve a slot, find the generation stale because the record was
// relocated, rechase to its new home, and only then validate the version — and
// the version it validates has to be the NEW page's, because the old page's
// counter is frozen and would never report a later in-place update.
func TestSeqlockReadsSurviveRelocation(t *testing.T) {
	cfg := relocConfig(3, true)
	cfg.InPlaceSameSizeUpdate = true
	cfg.InPlaceSeqlockReads = true
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]
	if !s.seqlockReads() {
		t.Fatal("the shard is not reading through the seqlock")
	}

	moved, beforeRef, afterRef := relocateOneRecord(t, c)
	if afterRef == beforeRef {
		t.Fatal("the record did not move")
	}

	// Read it through the seqlock at its new home.
	if _, gerr := c.Get(moved); gerr != nil {
		t.Fatalf("relocated record does not read back through the seqlock: %v", gerr)
	}
	// Rewrite in place at the new home and read again: the version that gets
	// validated must be the NEW page's, or this update would go unnoticed.
	mustPut(t, c, moved, relocValue(77))
	if v := mustGet(t, c, moved); !relocUniform(v, 77) {
		t.Fatal("in-place update after relocation is not visible through the seqlock")
	}
	st := c.Stats()
	if st.InPlaceUpdates == 0 {
		t.Fatal("the post-relocation rewrite did not take the in-place path")
	}
	// A record found by rechase must land on a page that HAS counters: every heap
	// page is built by freshHeapPageLocked, relocation destinations included, so a
	// rechased read can never be forced onto the lock by a missing counter.
	if st.SeqlockFallbacks != 0 {
		t.Fatalf("SeqlockFallbacks = %d with no concurrency; a relocation destination is missing its version counters", st.SeqlockFallbacks)
	}
}
