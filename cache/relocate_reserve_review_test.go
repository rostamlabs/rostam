// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"testing"
	"time"
)

// reserveSizedValue is a value of exactly n bytes filled with tag, for the tests below
// that need records of DIFFERENT sizes on one page — which the uniform-entry layout the
// other reserve tests use cannot express.
func reserveSizedValue(n, tag int) []byte {
	v := make([]byte, n)
	for i := range v {
		v[i] = byte(tag)
	}
	return v
}

// reserveMixedPage builds a shard whose page 0 holds, in order, a LARGE live record, a
// SMALL live record and a second large live record, packed so that nothing written later
// can land back in it. It returns the shard and the three keys.
//
// The sizes matter: page 0 ends nearly full, so tail minus any one record still clears
// minGain and the budget is the only thing that can refuse a move. That is what makes the
// tests below able to tell the two refusals apart.
func reserveMixedPage(t *testing.T, c *Cache) (*shard, []byte, []byte, []byte) {
	t.Helper()
	const (
		bigLen   = 600 << 10
		smallLen = 1 << 10
		fillLen  = 400 << 10
		dstLen   = 100 << 10
	)
	bigKey, smallKey, fillKey := relocKey(1), relocKey(2), relocKey(3)
	mustPut(t, c, bigKey, reserveSizedValue(bigLen, 1))
	mustPut(t, c, smallKey, reserveSizedValue(smallLen, 2))
	mustPut(t, c, fillKey, reserveSizedValue(fillLen, 3))
	// A page 1 that is non-empty and has room: the only destination a relocation can use,
	// since reserveDestinationLocked declines empty pages.
	mustPut(t, c, relocKey(4), reserveSizedValue(dstLen, 4))

	s := c.shards[0]
	s.mu.RLock()
	pages, free0 := len(s.pages), s.pages[0].FreeTail()
	s.mu.RUnlock()
	if pages != 2 {
		t.Fatalf("seed left %d pages, want 2", pages)
	}
	// The destination write is the only one that follows, and it must not land back in
	// page 0 — firstPageWithRoomLocked takes the first page with room, not the newest.
	if free0 >= dstLen {
		t.Fatalf("page 0 has %d bytes free, so the destination write lands back in it", free0)
	}
	return s, bigKey, smallKey, fillKey
}

// TestReserveRelocationStepsOverAnOverBudgetRecord is the reserve's half of the correction
// dc4ee95 made to the synchronous pass: a live record too large for what is left of the
// tick must not end the round, because a SMALLER live record further along can still use
// the remainder. Ending the round there leaves records on a page the next eviction drains
// — the loss the whole feature exists to prevent — for no reason other than walk order.
//
// The two passes must not diverge on this, which is why the test mirrors
// TestRelocatingEvictionBudgetStepsOverOversizedRecords.
func TestReserveRelocationStepsOverAnOverBudgetRecord(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s, bigKey, smallKey, fillKey := reserveMixedPage(t, c)

	before := pageObjects(s)
	beforePages := indexPagesByHash(s)
	// A budget that clears the small record and not either large one.
	s.reserveMoveVictim(0, s.pages[0], 300<<10, true)

	st := c.Stats()
	if st.ReserveRelocations != 1 {
		t.Fatalf("ReserveRelocations = %d, want exactly the small record: the walk either "+
			"stopped at the oversized record in front of it or moved something it should not",
			st.ReserveRelocations)
	}
	if now, was := indexPagesByHash(s)[hashKey(smallKey)], beforePages[hashKey(smallKey)]; now == was {
		t.Fatal("the small record behind the oversized one was never relocated")
	}
	// And the page keeps everything, because a live record was left on it.
	if st.ReservePagesFreed != 0 {
		t.Fatal("the page was retired with a live record still framed on it")
	}
	if freed := changedPage(before, pageObjects(s)); freed >= 0 {
		t.Fatalf("page %d was replaced despite holding a stepped-over live record", freed)
	}
	for _, k := range [][]byte{bigKey, smallKey, fillKey} {
		if _, gerr := c.Get(k); gerr != nil {
			t.Fatalf("key %q lost: %v", k, gerr)
		}
	}
}

// TestReserveRelocationRemembersASkipAcrossChunks is the same guarantee as the test above
// — a page holding a stepped-over live record must not be retired — asserted where it
// actually broke: ACROSS A CHUNK BOUNDARY.
//
// The walk releases the lock and re-acquires it every relocateChunkScanEntries entries,
// and the flag that records "a live record was left behind" was declared inside that loop,
// so it reset at every re-acquisition. A skip in one chunk was forgotten by the next, and
// whichever chunk happened to reach the end of the page retired it with that record still
// indexed. The test above could not see it because its page is small enough that the skip
// and the end of the walk land in the SAME chunk, which is exactly why the guard looked
// right; this one puts more than relocateChunkScanEntries entries between them.
func TestReserveRelocationRemembersASkipAcrossChunks(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	// Page 0 leads with one large LIVE record, then carries enough small records to force
	// the walk across several chunks. Every small record is superseded afterwards, so the
	// walk past the skip finds nothing but dead entries and runs clean to the end of the
	// page — which is the state that used to trigger the retirement.
	const (
		bigLen   = 600 << 10
		smallLen = 400
	)
	bigKey := relocKey(1)
	mustPut(t, c, bigKey, reserveSizedValue(bigLen, 1))

	small := reserveSizedValue(smallLen, 2)
	fillers := 0
	for {
		s.mu.RLock()
		room, pages := s.pages[0].FreeTail(), len(s.pages)
		s.mu.RUnlock()
		if pages > 1 || room < entrySize(len(relocKey(0)), smallLen) {
			break
		}
		mustPut(t, c, relocKey(1000+fillers), small)
		fillers++
	}
	if fillers+1 <= relocateChunkScanEntries {
		t.Fatalf("page 0 holds %d entries, not enough to span a %d-entry chunk; the skip and "+
			"the end of the walk would fall in the same chunk and the test would prove nothing",
			fillers+1, relocateChunkScanEntries)
	}
	// Supersede every filler so the walk behind the skipped record is entirely dead.
	for i := range fillers {
		mustPut(t, c, relocKey(1000+i), small)
	}

	before := pageObjects(s)
	// A budget the large record cannot fit but that is far from exhausted, so the walk
	// steps over it and carries on rather than stopping.
	s.reserveMoveVictim(0, s.pages[0], 300<<10, true)

	st := c.Stats()
	if st.ReservePagesFreed != 0 {
		t.Fatalf("the page was retired across a chunk boundary with a live record still on "+
			"it: the skip recorded in an earlier chunk was forgotten by the chunk that "+
			"reached the end of the page (%d pages freed)", st.ReservePagesFreed)
	}
	if freed := changedPage(before, pageObjects(s)); freed >= 0 {
		t.Fatalf("page %d was replaced despite holding a stepped-over live record", freed)
	}
	if st.EvictionsLive != 0 {
		t.Fatalf("EvictionsLive = %d: the retire walk dropped the live record it should "+
			"never have reached", st.EvictionsLive)
	}
	if v, gerr := c.Get(bigKey); gerr != nil || !bytes.Equal(v, reserveSizedValue(bigLen, 1)) {
		t.Fatalf("the stepped-over live record was lost: %v", gerr)
	}
}

// TestReserveRelocationDeclinesARecordThatWouldEatTheGain pins the check that has to
// happen BEFORE the copy rather than after it. Once the bytes are written they are spent
// whether or not they bought anything, so a page whose live set nearly fills it could
// otherwise take a page of background capacity and free nothing at all — the exact churn
// the retire rule exists to rule out.
func TestReserveRelocationDeclinesARecordThatWouldEatTheGain(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	// One live record occupying nearly the whole page: moving it would leave far less
	// than minGain (PageSize/relocateReserveMinGainDivisor) to reclaim.
	hog := relocKey(1)
	mustPut(t, c, hog, reserveSizedValue(900<<10, 1))
	mustPut(t, c, relocKey(2), reserveSizedValue(200<<10, 2)) // page 1: a destination with room
	s := c.shards[0]

	s.mu.RLock()
	tail, minGain := s.pages[0].tail(), s.maxEntryBytes()/relocateReserveMinGainDivisor
	s.mu.RUnlock()
	if tail-entrySize(len(hog), 900<<10) >= minGain {
		t.Fatalf("the seed does not set up the case: tail %d minus the record still clears minGain %d",
			tail, minGain)
	}

	s.reserveMoveVictim(0, s.pages[0], s.cfg.PageSize, true)

	if st := c.Stats(); st.ReserveRelocations != 0 || st.ReserveBytesRelocated != 0 {
		t.Fatalf("copied %d records (%d bytes) off a page that would have freed nothing",
			st.ReserveRelocations, st.ReserveBytesRelocated)
	}
	if v, gerr := c.Get(hog); gerr != nil || !bytes.Equal(v, reserveSizedValue(900<<10, 1)) {
		t.Fatalf("the declined record did not survive untouched: %v", gerr)
	}
}

// TestReserveRelocationRetiresAnAllDeadPageOnAnExhaustedBudget pins the ordering of the
// no-fit exit. An all-dead page is the single most valuable thing this pass can find — it
// frees a whole page and moves nothing — and the budget has no bearing on reclaiming it,
// because nothing needs copying. Testing the budget BEFORE classifying the record put that
// page off to a later tick for a reason that could never apply to it, which is the same
// defect as charging the budget for dead bytes, one step earlier in the walk.
func TestReserveRelocationRetiresAnAllDeadPageOnAnExhaustedBudget(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	// Page 0 full of records that are all superseded by later copies on page 1.
	for i := range relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	for i := range relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(100+i))
	}
	s.mu.RLock()
	live0 := livePageBytesLocked(s, 0)
	s.mu.RUnlock()
	if live0 != 0 {
		t.Fatalf("page 0 still holds %d live bytes; the seed did not supersede everything", live0)
	}

	before := pageObjects(s)
	// A budget with no room even for an empty entry: nothing could be copied, and nothing
	// needs to be.
	s.reserveMoveVictim(0, s.pages[0], 0, true)

	st := c.Stats()
	if st.ReservePagesFreed != 1 {
		t.Fatalf("an all-dead page was not retired on an exhausted budget (%d freed); the "+
			"budget was consulted before the walk established there was nothing to copy",
			st.ReservePagesFreed)
	}
	if changedPage(before, pageObjects(s)) != 0 {
		t.Fatal("page 0 was not the page retired")
	}
	if st.ReserveRelocations != 0 {
		t.Fatalf("%d records were copied off a page with nothing live on it", st.ReserveRelocations)
	}
	for i := range relocPerPage {
		if _, gerr := c.Get(relocKey(i)); gerr != nil {
			t.Fatalf("key %d lost with the page: %v", i, gerr)
		}
	}
}

// TestReserveRelocationChargesTheHandoffToTheTick pins the per-tick byte bound as a
// property rather than a claim. The retirement runs the synchronous pass to evacuate the
// page the cursor lands on, and that pass used to arrive with a budget of its own — so a
// tick could spend its stated bound and then spend most of it again, and the bound the
// documentation advertised was not one the code kept.
func TestReserveRelocationChargesTheHandoffToTheTick(t *testing.T) {
	c, err := New(reserveConfig(8, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	const (
		coldKeys = 20
		hotKeys  = 40
	)
	reserveSeedChurnSet(t, c, coldKeys)
	for n := range 8 * reserveMediumPerPage {
		mustPut(t, c, relocKey(500+n%hotKeys), reserveMediumValue(n%256))
	}

	worst := uint64(0)
	for range 40 {
		before := c.Stats().ReserveBytesRelocated
		s.topUpFreeReserve()
		if moved := c.Stats().ReserveBytesRelocated - before; moved > worst {
			worst = moved
		}
		for n := range reserveMediumPerPage {
			mustPut(t, c, relocKey(500+n%hotKeys), reserveMediumValue(n%256))
		}
	}
	if worst == 0 {
		t.Fatal("no tick moved anything; the bound was never actually exercised")
	}
	if bound := uint64(s.cfg.PageSize); worst > bound { //nolint:gosec // PageSize is positive
		t.Fatalf("a single tick moved %d bytes against a stated per-tick bound of %d", worst, bound)
	}
}

// TestReserveRelocationDeclinesAShardTooSmallForAReserve pins that a shard whose page cap
// cannot support a reserve target is refused up front. reserveTargetBytesLocked floors to
// zero below the divisor, so such a shard would start a ticker, wake on schedule forever
// and turn round every time — a goroutine and a timer per shard to do nothing.
func TestReserveRelocationDeclinesAShardTooSmallForAReserve(t *testing.T) {
	for _, tc := range []struct {
		pages int
		want  bool
	}{
		{relocateReserveShardDivisor - 1, false},
		{relocateReserveShardDivisor, true},
	} {
		cfg := reserveConfig(tc.pages, true)
		c, err := New(cfg)
		if err != nil {
			t.Fatalf("New(%d pages): %v", tc.pages, err)
		}
		got := c.shards[0].reserveRelocationEligible()
		_ = c.Close()
		if got != tc.want {
			t.Fatalf("a %d-page shard reports eligible=%v, want %v", tc.pages, got, tc.want)
		}
	}
}

// TestRetireCountsExpiredEntriesAsExpirations pins where TTL turnover is reported when a
// retire reaches an expired entry before the sweeper does. Stats documents EvictionsLive
// as the figure to read AGAINST Expirations — "EvictionsLive > 0 with Expirations low
// means the budget, not the TTL, is deciding how long entries survive" — and that reading
// is only sound if the entries a retire reaps land in Expirations rather than vanishing
// from both. The reserve makes this reachable far more often than before, since it retires
// on a timer well ahead of the TTL sweeper's own cadence.
func TestRetireCountsExpiredEntriesAsExpirations(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const ttl = 20 * time.Millisecond
	for i := range relocPerPage {
		if perr := c.Put(relocKey(i), relocValue(i), ttl); perr != nil {
			t.Fatalf("Put: %v", perr)
		}
	}
	mustPut(t, c, relocKey(900), relocValue(9)) // page 1, so page 0 is closed
	s := c.shards[0]
	time.Sleep(3 * ttl)

	before := c.Stats()
	s.reserveMoveVictim(0, s.pages[0], s.cfg.PageSize, true)
	st := c.Stats()

	if st.ReservePagesFreed-before.ReservePagesFreed != 1 {
		t.Fatal("the all-expired page was not retired; the test measured nothing")
	}
	if got := st.Expirations - before.Expirations; got != relocPerPage {
		t.Fatalf("Expirations rose by %d, want the page's %d expired entries", got, relocPerPage)
	}
	if got := st.EvictionsLive - before.EvictionsLive; got != 0 {
		t.Fatalf("EvictionsLive rose by %d; expired entries are TTL turnover, not capacity loss", got)
	}
	if st.Evictions <= before.Evictions {
		t.Fatal("Evictions did not move; it counts every framed entry a retire displaces")
	}
}
