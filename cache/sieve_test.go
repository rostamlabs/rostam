// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"testing"
)

// Tests for the SIEVE reference hint (cache/sieve.go). They are grouped by the
// design decision each one pins: where the bit lives and what a rehash does with
// it, what sets it, what the drain does with it, and what is out of scope.

// sieveConfig is relocConfig with relocating eviction ON and the hint switchable,
// so every test below is an A/B on the hint alone.
func sieveConfig(pages int, sieve bool) Config {
	cfg := relocConfig(pages, true)
	cfg.SieveVisitedBit = sieve
	return cfg
}

// sieveMarked reports whether key's index slot currently carries the hint.
func sieveMarked(s *shard, key []byte) bool {
	h := hashKey(key)
	t := s.tab.Load()
	slot, _, ok := t.findSlot(h)
	if !ok {
		return false
	}
	return t.ctrl[slot].Load()&ctrlVisited != 0
}

// sieveClear drops key's hint, so a test can set up "not referenced since the last
// drain" without running one.
func sieveClear(s *shard, key []byte) {
	h := hashKey(key)
	t := s.tab.Load()
	if slot, _, ok := t.findSlot(h); ok {
		t.takeVisited(slot, tagFor(h))
	}
}

// keyPageOrFail returns the page index the shard's index CURRENTLY resolves key to.
// Which page a record sits on is what a drain decision moves, so this is the
// observation the relocation tests are built on — it says where the record went
// without a Get, which would itself mark the slot and destroy the state under test.
func keyPageOrFail(t *testing.T, s *shard, key []byte) uint16 {
	t.Helper()
	_, ref, ok := s.tab.Load().findSlot(hashKey(key))
	if !ok {
		t.Fatalf("key %q is not in the index", key)
	}
	return ref.pageIdx()
}

// ==========================================================================
// WHERE THE BIT LIVES.

// TestSieveHintDoesNotHideTheSlot is the masking guard. The hint sits ABOVE the tag
// in the same control word, so every tag comparison in cache/indextable.go has to
// mask it off; miss one and a marked key becomes invisible to that path — a read
// returns not-found for a key that is present, or an update inserts a second slot
// for a key that already has one. Each path is exercised on a MARKED slot.
func TestSieveHintDoesNotHideTheSlot(t *testing.T) {
	c, err := New(sieveConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	key := relocKey(1)
	mustPut(t, c, key, relocValue(7))
	// A read marks it (see below); this test only needs the bit set.
	mustGet(t, c, key)
	if !sieveMarked(s, key) {
		t.Fatal("precondition: the read should have marked the slot")
	}

	// findSlot — the writer-side probe.
	if _, _, ok := s.tab.Load().findSlot(hashKey(key)); !ok {
		t.Error("findSlot lost a marked slot")
	}
	// get — the lock-free reader probe.
	if v := mustGet(t, c, key); !relocUniform(v, 7) {
		t.Error("get returned the wrong bytes for a marked slot")
	}
	// upsert — an update must REPOINT the marked slot, not insert beside it.
	liveBefore := s.tab.Load().live
	mustPut(t, c, key, relocValue(8))
	if got := s.tab.Load().live; got != liveBefore {
		t.Errorf("upsert on a marked slot changed live count %d -> %d (it inserted a duplicate)", liveBefore, got)
	}
	if v := mustGet(t, c, key); !relocUniform(v, 8) {
		t.Error("the update did not take on a marked slot")
	}
}

// TestSieveHintSurvivesRehash pins the one thing rehashing MUST do. rehashed()
// rebuilds the table by upserting every live entry, and it controls that copy — so
// if it does not carry the hint across, every hint in the shard is cleared at once
// and a drain landing just after would see a whole page of records as unvisited and
// drop all of them. That is not the "one missed retention" the design accepts.
func TestSieveHintSurvivesRehash(t *testing.T) {
	const n = 12
	tab := newIndexTable(n)
	hashes := make([]uint64, n)
	for i := range n {
		hashes[i] = hashKey(fmt.Appendf(nil, "sieve-rehash-%03d", i))
		tab.upsert(hashes[i], makeSlabRef(0, 0, uint32(i*64))) //nolint:gosec // small fixed offsets
	}
	// Mark the even ones.
	for i := 0; i < n; i += 2 {
		slot, _, ok := tab.findSlot(hashes[i])
		if !ok {
			t.Fatalf("entry %d missing before the rehash", i)
		}
		tab.setVisited(slot, tagFor(hashes[i]))
	}

	nt := tab.rehashed()
	if nt.live != n {
		t.Fatalf("rehashed dropped entries: live=%d want %d", nt.live, n)
	}
	for i := range n {
		slot, _, ok := nt.findSlot(hashes[i])
		if !ok {
			t.Fatalf("entry %d missing after the rehash", i)
		}
		marked := nt.ctrl[slot].Load()&ctrlVisited != 0
		if want := i%2 == 0; marked != want {
			t.Errorf("entry %d: marked=%v after rehash, want %v", i, marked, want)
		}
	}
}

// ==========================================================================
// WHAT SETS THE BIT.

// TestSieveInsertDoesNotMark is SIEVE's one-hit-wonder resistance, and it is the
// half that does the work: a stream of keys written once and never touched again is
// the traffic that evicts everything else, so it has to arrive unvisited or the hint
// discriminates nothing.
func TestSieveInsertDoesNotMark(t *testing.T) {
	c, err := New(sieveConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	key := relocKey(2)
	mustPut(t, c, key, relocValue(1))
	if sieveMarked(s, key) {
		t.Error("a first insertion marked the slot; a write-once key must arrive unvisited")
	}
}

// TestSieveReadMarks is the plain access case.
func TestSieveReadMarks(t *testing.T) {
	c, err := New(sieveConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	key := relocKey(3)
	mustPut(t, c, key, relocValue(1))
	mustGet(t, c, key)
	if !sieveMarked(s, key) {
		t.Error("a read hit did not mark the slot")
	}
}

// TestSieveAppendingUpdateMarks covers the ordinary rewrite. The record moves to the
// newest page, so it is already refreshed positionally; marking keeps the rule "an
// access marks" uniform across both write paths.
func TestSieveAppendingUpdateMarks(t *testing.T) {
	c, err := New(sieveConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	key := relocKey(4)
	mustPut(t, c, key, relocValue(1))
	if sieveMarked(s, key) {
		t.Fatal("precondition: the insert should not have marked the slot")
	}
	mustPut(t, c, key, relocValue(2))
	if !sieveMarked(s, key) {
		t.Error("an update of an already-present key did not mark the slot")
	}
}

// TestSieveInPlaceUpdateMarks is the case the whole feature exists for. An in-place
// rewrite keeps the record's page AND its offset, so it leaves no positional trace of
// having happened — the index slot is the only place it can record that the key is in
// use, and if it does not, a rewrite-heavy shard looks identical to a dead one.
func TestSieveInPlaceUpdateMarks(t *testing.T) {
	cfg := sieveConfig(3, true)
	cfg.InPlaceSameSizeUpdate = true
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	key := relocKey(5)
	mustPut(t, c, key, relocValue(1))
	sieveClear(s, key)
	before := s.snapshot().InPlaceUpdates
	mustPut(t, c, key, relocValue(2)) // same key, same length ⇒ in place
	if got := s.snapshot().InPlaceUpdates; got != before+1 {
		t.Fatalf("precondition: expected one in-place update, InPlaceUpdates %d -> %d", before, got)
	}
	if !sieveMarked(s, key) {
		t.Error("an in-place update did not mark the slot")
	}
}

// ==========================================================================
// WHAT THE DRAIN DOES WITH IT.

// TestSieveDrainRescuesReferencedRecords is the feature itself, on a shard laid out
// so the answer is exact rather than statistical.
//
// Three 1 MiB pages hold five equal records each. The fourth page's worth of writes
// evicts page 0 and hands relocation the freed page, whose budget
// (PageSize/relocateMaxBytesPerEvictionDivisor) buys exactly TWO of these records —
// and page 1, the page the next eviction will drain, holds five LIVE ones. So the
// pass must choose two of five, and which two it chooses is the whole policy:
//
//	without the hint  positional — the first two the walk meets, B0 and B1, whether
//	                  or not anything ever used them.
//	with the hint     the referenced ones, B3 and B4, though they are last on the page.
//
// The check is on where the index resolves each key AFTERWARDS, never on a Get: a Get
// would mark the slot it read and destroy the state under test.
func TestSieveDrainRescuesReferencedRecords(t *testing.T) {
	for _, tc := range []struct {
		name   string
		sieve  bool
		moved  []int // the B-indices expected to have been carried to the freed page
		stayed []int // the B-indices expected to still be on the page about to drain
	}{
		{"off", false, []int{0, 1}, []int{2, 3, 4}},
		{"on", true, []int{3, 4}, []int{0, 1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := New(sieveConfig(3, tc.sieve))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer func() { _ = c.Close() }()
			s := c.shards[0]

			aKey := func(i int) []byte { return relocKey(i) }
			bKey := func(i int) []byte { return relocKey(100 + i) }
			cKey := func(i int) []byte { return relocKey(200 + i) }
			for i := range relocPerPage { // page 0
				mustPut(t, c, aKey(i), relocValue(i))
			}
			for i := range relocPerPage { // page 1 — the relocation source
				mustPut(t, c, bKey(i), relocValue(i))
			}
			for i := range relocPerPage { // page 2
				mustPut(t, c, cKey(i), relocValue(i))
			}
			if got := s.numPages(); got != 3 {
				t.Fatalf("expected the shard at its 3-page cap, got %d", got)
			}
			for i := range relocPerPage {
				if p := keyPageOrFail(t, s, bKey(i)); p != 1 {
					t.Fatalf("precondition: B%d is on page %d, want 1", i, p)
				}
			}

			// REFERENCE the LAST two records on the source page. They are the two the
			// positional walk reaches last, so nothing but the hint can rescue them.
			mustGet(t, c, bKey(3))
			mustGet(t, c, bKey(4))

			// One more record than the shard can hold: evicts page 0, then relocates out
			// of page 1 into the space that freed.
			mustPut(t, c, relocKey(300), relocValue(9))

			// Evictions counts dropped RECORDS: page 0's five, and only those. A second
			// drained page would mean the layout this test rests on has moved and the
			// placement assertions no longer say what they claim.
			if got := s.snapshot().Evictions; got != uint64(relocPerPage) {
				t.Fatalf("expected one page drained (%d records), got %d", relocPerPage, got)
			}
			for _, i := range tc.moved {
				if p := keyPageOrFail(t, s, bKey(i)); p != 0 {
					t.Errorf("B%d: on page %d, want 0 (it should have been rescued)", i, p)
				}
			}
			for _, i := range tc.stayed {
				if p := keyPageOrFail(t, s, bKey(i)); p != 1 {
					t.Errorf("B%d: on page %d, want 1 (it should have been left to drain)", i, p)
				}
			}
		})
	}
}

// TestSieveDrainClearsTheHintOnRescue is what keeps this a policy rather than a
// ratchet. A rescued record starts its next rotation unvisited, so it has to be
// referenced again to earn another rescue; without the clear, one access would pin a
// record in the shard permanently.
func TestSieveDrainClearsTheHintOnRescue(t *testing.T) {
	c, err := New(sieveConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	for i := range relocPerPage * 3 {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	// Reference one record on page 1, the page the next eviction's relocation reads.
	hot := relocKey(relocPerPage) // first key on page 1
	mustGet(t, c, hot)
	if !sieveMarked(s, hot) {
		t.Fatal("precondition: the read should have marked the slot")
	}

	mustPut(t, c, relocKey(999), relocValue(9)) // evict page 0, relocate out of page 1

	if p := keyPageOrFail(t, s, hot); p != 0 {
		t.Fatalf("precondition: the referenced record was not rescued (on page %d)", p)
	}
	if sieveMarked(s, hot) {
		t.Error("the hint survived the rescue; SIEVE's hand must clear as it passes")
	}
}

// TestSieveRetainsTheRewrittenSetUnderInPlaceUpdates is the end-to-end claim, in the
// shape the regression was measured in: a small set rewritten over and over at a
// constant size, interleaved with keys written ONCE, on a shard deliberately smaller
// than its live set. In-place updates are what make this hard — a rewrite keeps the
// record's page, so its page age only grows and positional relocation cannot tell it
// from the write-once stream beside it.
//
// The cold keys being distinct is load-bearing: if they repeated they would be
// rewrites too, and the contrast the test rests on would not exist.
func TestSieveRetainsTheRewrittenSetUnderInPlaceUpdates(t *testing.T) {
	const (
		hotKeys   = 2_000
		writes    = 200_000
		hotEveryN = 4
	)
	hotKey := func(i int) []byte { return fmt.Appendf(nil, "h%011d", i%hotKeys) }
	coldKey := func(i int) []byte { return fmt.Appendf(nil, "c%011d", i) }

	run := func(sieve bool) int {
		cfg := DefaultConfig()
		cfg.NumShards = 1
		cfg.PageSize = 1 << 20
		cfg.MaxMemoryPerShard = 4 << 20
		cfg.TTLSweepIntervalMs = 0
		cfg.RelocateReserveIntervalMs = 0
		cfg.AtCapPolicy = PolicyRingbufEvict
		cfg.RelocatingEviction = true
		cfg.InPlaceSameSizeUpdate = true
		cfg.SieveVisitedBit = sieve
		c, err := New(cfg)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer func() { _ = c.Close() }()
		val := make([]byte, 256)
		cold := 0
		for i := range writes {
			if i%hotEveryN == 0 {
				if err := c.Put(hotKey(i/hotEveryN), val, 0); err != nil {
					t.Fatalf("Put: %v", err)
				}
			} else {
				if err := c.Put(coldKey(cold), val, 0); err != nil {
					t.Fatalf("Put: %v", err)
				}
				cold++
			}
		}
		held := 0
		for i := range hotKeys {
			if _, err := c.Get(hotKey(i)); err == nil {
				held++
			}
		}
		return held
	}

	off, on := run(false), run(true)
	t.Logf("rewritten set retained: hint off %d/%d, hint on %d/%d", off, hotKeys, on, hotKeys)
	if on <= off {
		t.Errorf("the hint did not improve retention of the rewritten set: off=%d on=%d", off, on)
	}
}

// ==========================================================================
// SCOPE.

// TestSieveIsRingbufOnly confirms from the code what the design claims: the hint is
// maintained only where eviction can happen. PolicyRejectWrites never drains a page —
// findOrMakePageLocked rejects instead — and replication forces that policy, so no
// replicated shard can have its retained set perturbed by this, whatever the flag
// says.
func TestSieveIsRingbufOnly(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.TTLSweepIntervalMs = 0
	cfg.RelocateReserveIntervalMs = 0
	cfg.SieveVisitedBit = true
	cfg.AtCapPolicy = PolicyRejectWrites
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]
	if s.sieve {
		t.Fatal("a PolicyRejectWrites shard must not maintain the hint")
	}

	key := []byte("reject-writes-key")
	mustPut(t, c, key, []byte("v"))
	mustGet(t, c, key)
	if sieveMarked(s, key) {
		t.Error("a read marked a slot on a shard that never evicts")
	}
}

// TestSieveOffCostsNothing pins the other half of the scope: with the flag off, no
// path sets the bit, so an existing shard's control words are byte-for-byte what they
// were. A stray mark on a shard that never consults it would be harmless but would
// also mean the read path is paying for a feature nobody enabled.
func TestSieveOffCostsNothing(t *testing.T) {
	c, err := New(sieveConfig(3, false))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]
	if s.sieve {
		t.Fatal("the hint is enabled on a shard whose config has it off")
	}

	key := relocKey(6)
	mustPut(t, c, key, relocValue(1))
	mustGet(t, c, key)
	mustPut(t, c, key, relocValue(2))
	if sieveMarked(s, key) {
		t.Error("something marked a slot with the feature disabled")
	}
}
