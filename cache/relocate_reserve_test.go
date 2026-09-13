// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// reserveConfig is relocConfig with both background tickers still OFF: the deterministic
// tests below drive topUpFreeReserve by hand, so a tick landing mid-assertion can never
// make them flaky. The one test that needs the REAL ticker sets its interval itself.
func reserveConfig(pages int, on bool) Config { return relocConfig(pages, on) }

// seedReserveShard fills a `pages`-page heap ringbuf shard to its page cap, leaving
// the LAST page holding a single entry. That shape matters: the reserve pass refuses
// to relocate into an EMPTY page (spending the reserve to build it is a wash), so
// without a partly-filled page there is nowhere for a relocated record to go. It is
// also the state a shard is in the moment it reaches capacity in the first place.
//
// Page 0 is seeded with `dead+1` copies of ONE key, so exactly one of its records is
// index-current and the rest are superseded — the page the pass exists to evacuate.
// The intermediate pages hold distinct live keys. Returns the hot key.
func seedReserveShard(t *testing.T, c *Cache, pages int) []byte {
	t.Helper()
	hot := relocKey(900)
	for i := range relocPerPage {
		mustPut(t, c, hot, relocValue(200+i))
	}
	for p := 1; p < pages-1; p++ {
		for i := range relocPerPage {
			mustPut(t, c, relocKey(p*100+i), relocValue(i))
		}
	}
	mustPut(t, c, relocKey(9000), relocValue(1))
	s := c.shards[0]
	s.mu.RLock()
	got := len(s.pages)
	s.mu.RUnlock()
	if got != pages {
		t.Fatalf("seed left %d pages, want the shard at its cap of %d", got, pages)
	}
	if st := c.Stats(); st.Evictions != 0 {
		t.Fatalf("seed evicted %d entries; it must only fill", st.Evictions)
	}
	return hot
}

// TestReserveRelocationKeepsLiveRecordOnADeadPage is the background layer in one page,
// and it is the same claim the synchronous pass makes with the write taken out of it:
// the sweeper frees a page whose records are nearly all superseded, the one record
// that was still live is carried forward rather than dropped, and no write was
// involved in any of it.
func TestReserveRelocationKeepsLiveRecordOnADeadPage(t *testing.T) {
	c, rec := newOnRemoveCache(t, reserveConfig(4, true))
	hot := seedReserveShard(t, c, 4)
	s := c.shards[0]
	want := relocValue(200 + relocPerPage - 1)

	before := pageObjects(s)
	s.topUpFreeReserve()

	st := c.Stats()
	if st.ReservePagesFreed != 1 {
		t.Fatalf("ReservePagesFreed = %d, want 1", st.ReservePagesFreed)
	}
	// At least the victim's one live record. It is not exactly one: the same call runs
	// one round MORE than the reserve needs, and that extra round PREPARES the next
	// victim — which is the point of it, and is asserted on its own in
	// TestReserveRelocationPreparesTheNextVictim.
	if st.ReserveRelocations < 1 {
		t.Fatal("ReserveRelocations = 0, so the page's live record was not carried forward")
	}
	if st.ReserveBytesRelocated != st.ReserveRelocations*relocEntrySize {
		t.Fatalf("ReserveBytesRelocated = %d for %d uniform %d-byte records",
			st.ReserveBytesRelocated, st.ReserveRelocations, relocEntrySize)
	}
	if st.EvictionRelocations != 0 {
		t.Fatalf("EvictionRelocations = %d: the write path relocated, but no write ran", st.EvictionRelocations)
	}
	if freed := changedPage(before, pageObjects(s)); freed != 0 {
		t.Fatalf("the pass retired page %d, want the rotation victim (page 0)", freed)
	}
	// The moved record is not a loss and must not be reported as one.
	if st.EvictionsLive != 0 {
		t.Fatalf("EvictionsLive = %d; a relocated record is not a capacity loss", st.EvictionsLive)
	}
	if n := rec.count(string(hot)); n != 0 {
		t.Fatalf("onRemove fired %d times for a relocated key; it must fire only for records actually removed", n)
	}
	if got := mustGet(t, c, hot); !bytes.Equal(got, want) {
		t.Fatal("relocated value differs from the original")
	}
	// The whole page's framed entries — the superseded copies included — are gone.
	if st.Evictions != relocPerPage {
		t.Fatalf("Evictions = %d, want the victim's %d framed entries", st.Evictions, relocPerPage)
	}
}

// TestReserveRelocationEvacuatesTheNextEvictionVictim is this layer's half of the
// alignment TestRelocatingEvictionEvacuatesThePageTheNextEvictionDrains pins for the
// synchronous pass. Evacuating any page OTHER than the one the next eviction takes is
// write amplification that saves nothing, and here the guarantee is stronger than
// agreement between two selections: the pass takes its victim from the SAME
// nextNonEmptyPageLocked(-1) call evictUntilFitsLocked uses, and retires that page
// itself. This asserts the page it frees is the page an eviction would have drained.
func TestReserveRelocationEvacuatesTheNextEvictionVictim(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	seedReserveShard(t, c, 4)
	s := c.shards[0]

	s.mu.Lock()
	wantVictim := s.nextNonEmptyPageLocked(-1)
	s.mu.Unlock()

	before := pageObjects(s)
	s.topUpFreeReserve()
	freed := changedPage(before, pageObjects(s))
	if freed != wantVictim {
		t.Fatalf("the pass freed page %d but an eviction would have drained page %d; "+
			"the reserve and the rotation cursor have drifted apart", freed, wantVictim)
	}
	s.mu.RLock()
	cursor := s.nextVictim
	s.mu.RUnlock()
	if cursor != (wantVictim+1)%4 {
		t.Fatalf("nextVictim = %d after freeing page %d, want %d: the pass must advance the "+
			"cursor exactly as evictUntilFitsLocked does", cursor, wantVictim, (wantVictim+1)%4)
	}
}

// livePageBytes is the framed bytes of page idx that are still index-current: the
// records a drain of that page would LOSE. Takes the read lock.
func livePageBytes(s *shard, idx int) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p := s.pages[idx]
	tab := s.tab.Load()
	entries, tail := p.entries(), p.tail()
	live := 0
	for cursor := p.head(); cursor < tail; {
		key, value, _, err := decodeEntryFast(entries[cursor:tail])
		if err != nil {
			return live
		}
		size := entrySize(len(key), len(value))
		ref := makeSlabRef(uint16(idx), p.gen, uint32(cursor)) //nolint:gosec // bounded by the page geometry
		if _, cur, ok := tab.findSlot(hashKey(key)); ok && cur == ref {
			live += size
		}
		cursor += size
	}
	return live
}

// TestReserveRelocationPreparesTheNextVictim pins the half of this layer that is easy to
// leave out and expensive to leave out. Retiring a page moves the rotation cursor onto
// the NEXT page; if nothing has evacuated that page, a write-path eviction arriving next
// drains it with live records still on it — a loss neither the synchronous pass nor the
// background one takes on its own, and one that surfaces only under concurrency, where it
// reads as a flaky missing key rather than as a bug here.
//
// The guarantee is an EQUIVALENCE, not emptiness: the retire runs the same
// relocateIntoFreedPageLocked with the same budget evictVictimLocked would have run, so
// the successor is carried at least as far as a write-path eviction would have carried
// it. Whether it ends up empty is the spend cap's business (a page more than half live is
// deliberately left half-carried for the write path), not this test's.
func TestReserveRelocationPreparesTheNextVictim(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	seedReserveShard(t, c, 4)
	s := c.shards[0]

	// The rotation cursor starts at 0, so page 0 is the victim and page 1 its successor.
	s.mu.RLock()
	victim := s.nextNonEmptyPageLocked(-1)
	s.mu.RUnlock()
	if victim != 0 {
		t.Fatalf("the seed left the rotation cursor on page %d, not 0", victim)
	}
	before := livePageBytes(s, 1)
	if before == 0 {
		t.Fatal("the successor holds nothing live; the test would pass for the wrong reason")
	}

	s.topUpFreeReserve()
	if c.Stats().ReservePagesFreed == 0 {
		t.Fatal("the pass freed no page, so there was never a successor to prepare")
	}
	moved := before - livePageBytes(s, 1)

	// What a write-path eviction of page 0 would have carried off page 1.
	want := relocPageSize / relocateMaxBytesPerEvictionDivisor
	if before < want {
		want = before
	}
	if moved < want {
		t.Fatalf("the retire carried %d of the successor's %d live bytes; a write-path eviction "+
			"would have carried %d, so the retire left its successor less prepared than an "+
			"ordinary eviction does", moved, before, want)
	}
}

// TestReserveRelocationPreservesTriple checks a background move carries the exact
// (key, value, exp) triple, including an entry whose TTL is live but near — a relocated
// record keeps its original absolute expiry and is never silently refreshed by the copy.
func TestReserveRelocationPreservesTriple(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	// Page 0: the TTL key, then superseded copies of another key.
	ttlKey := relocKey(100)
	ttlVal := relocValue(42)
	const ttl = 10 * time.Minute
	before := nowMs()
	if err := c.Put(ttlKey, ttlVal, ttl); err != nil {
		t.Fatalf("Put ttl key: %v", err)
	}
	after := nowMs()
	_, wantExp, err := s.getWithExpiryH(ttlKey, hashKey(ttlKey))
	if err != nil {
		t.Fatalf("read back ttl key: %v", err)
	}
	if wantExp < before+uint64(ttl.Milliseconds()) || wantExp > after+uint64(ttl.Milliseconds()) {
		t.Fatalf("seeded expiry %d outside the window the Put should have stamped", wantExp)
	}
	for i := range relocPerPage - 1 {
		mustPut(t, c, relocKey(200), relocValue(i))
	}
	// Pages 1 and 2 full, page 3 holding one entry, so the shard is at its cap with
	// somewhere for the relocated records to land.
	for p := 1; p < 3; p++ {
		for i := range relocPerPage {
			mustPut(t, c, relocKey(p*1000+i), relocValue(i))
		}
	}
	mustPut(t, c, relocKey(9000), relocValue(1))

	s.topUpFreeReserve()
	if n := c.Stats().ReserveRelocations; n == 0 {
		t.Fatal("nothing relocated")
	}
	gotVal, gotExp, err := s.getWithExpiryH(ttlKey, hashKey(ttlKey))
	if err != nil {
		t.Fatalf("relocated ttl key: %v", err)
	}
	if !bytes.Equal(gotVal, ttlVal) {
		t.Fatal("relocated value differs from the original")
	}
	if gotExp != wantExp {
		t.Fatalf("relocated expiry = %d, want the original %d", gotExp, wantExp)
	}
}

// TestReserveRelocationNeverDropsALiveRecord is the contract that is STRICTER here
// than on the write path. The synchronous pass rides an eviction that is happening
// anyway, so leaving a record behind is the policy's decision, already made. Nothing
// has decided anything when the sweeper runs. So when the victim's live set does not
// fit, the pass must leave the page alone rather than retire it and shed what it could
// not carry — a full shard must never lose records while idle.
func TestReserveRelocationNeverDropsALiveRecord(t *testing.T) {
	c, rec := newOnRemoveCache(t, reserveConfig(4, true))
	// Every page packed with DISTINCT live keys and no partially-filled frontier: there
	// is no room anywhere, so no record can be moved and no page may be retired.
	for i := range 4 * relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i%256))
	}
	s := c.shards[0]
	before := pageObjects(s)

	s.topUpFreeReserve()

	st := c.Stats()
	if st.ReservePagesFreed != 0 {
		t.Fatalf("ReservePagesFreed = %d: a page whose live set could not be moved was retired anyway", st.ReservePagesFreed)
	}
	if st.Evictions != 0 || st.EvictionsLive != 0 {
		t.Fatalf("the pass evicted (%d entries, %d live) with no write asking for room", st.Evictions, st.EvictionsLive)
	}
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("onRemove fired for %v; the background pass removes nothing", got)
	}
	if freed := changedPage(before, pageObjects(s)); freed >= 0 {
		t.Fatalf("page %d was replaced; no page should have been touched", freed)
	}
	for i := range 4 * relocPerPage {
		if _, err := c.Get(relocKey(i)); err != nil {
			t.Fatalf("key %d lost to the background pass: %v", i, err)
		}
	}
}

// TestReserveRelocationOffDoesNothing pins the flag-off path: the sweeper must do no
// extra work at all, so a shard that never opted in is byte-for-byte what it is today.
func TestReserveRelocationOffDoesNothing(t *testing.T) {
	c, err := New(reserveConfig(4, false))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	seedReserveShard(t, c, 4)
	s := c.shards[0]

	before := pageObjects(s)
	beforeStats := c.Stats()
	var chunks atomic.Int64
	hook := func(int, int) { chunks.Add(1) }
	s.reserveChunkHook.Store(&hook)
	s.topUpFreeReserve()
	s.sweepOnce()

	if n := chunks.Load(); n != 0 {
		t.Fatalf("the reserve pass ran %d chunks with RelocatingEviction off", n)
	}
	// And the ticker itself never starts on such a shard, so it costs not even a
	// goroutine: startReserveSweeper declines a shard the reserve could never apply to.
	if s.reserveRelocationEligible() {
		t.Fatal("a shard with RelocatingEviction off reports itself eligible for the reserve")
	}
	if freed := changedPage(before, pageObjects(s)); freed >= 0 {
		t.Fatalf("page %d was replaced with the feature off", freed)
	}
	st := c.Stats()
	if st.ReserveRelocations != 0 || st.ReserveBytesRelocated != 0 || st.ReservePagesFreed != 0 {
		t.Fatalf("reserve counters non-zero with the feature off: %+v", st)
	}
	if st.Evictions != beforeStats.Evictions || st.EvictionsLive != beforeStats.EvictionsLive {
		t.Fatal("the sweeper changed the eviction counters with the feature off")
	}
}

// reserveSmall is the geometry for the chunking tests: entries small enough that one
// page holds hundreds, so a single evacuation crosses the chunk bound many times.
const (
	reserveSmallValueLen = 1000
	reserveSmallEntry    = entryHeaderSize + relocKeyLen + reserveSmallValueLen
	reserveSmallPerPage  = relocPageSize / reserveSmallEntry
)

func reserveSmallValue(tag int) []byte {
	v := make([]byte, reserveSmallValueLen)
	for i := range v {
		v[i] = byte(tag)
	}
	return v
}

// reserveMedium is the geometry for the steady-state tests: a page holds a couple of
// hundred records, so a cold set of twenty is a small fraction of one — well inside the
// half-a-page rule, which a cold set of relocValue-sized records would blow straight
// through, leaving the pass declining every page it was meant to carry.
const (
	reserveMediumValueLen = 4000
	reserveMediumEntry    = entryHeaderSize + relocKeyLen + reserveMediumValueLen
	reserveMediumPerPage  = relocPageSize / reserveMediumEntry
)

func reserveMediumValue(tag int) []byte {
	v := make([]byte, reserveMediumValueLen)
	for i := range v {
		v[i] = byte(tag)
	}
	return v
}

// reserveSeedChurnSet fills page 0 with filler and then seeds coldKeys records that are
// written ONCE and must survive — the records relocation exists to carry. Page 0 is the
// one page drained with nothing having evacuated it (both passes run one page ahead of
// the rotation cursor), so nothing that must live may be seeded there.
func reserveSeedChurnSet(t *testing.T, c *Cache, coldKeys int) {
	t.Helper()
	for i := range reserveMediumPerPage {
		mustPut(t, c, relocKey(7000+i), reserveMediumValue(i%256))
	}
	for i := range coldKeys {
		mustPut(t, c, relocKey(i), reserveMediumValue(100+i))
	}
}

// seedReserveSmallShard fills a 4-page shard with reserveSmallEntry-sized records,
// leaving page 3 holding a single one — the same at-cap-with-a-frontier shape
// seedReserveShard builds, at a scale where one page is hundreds of entries.
//
// Page 0 is written as groups of THREE copies per key, so only a third of it is
// index-current. That is deliberate: the pass declines any page more than half live
// (the half-a-page rule), so a page 0 of distinct keys would be left alone and the
// chunking tests below would observe nothing at all.
func seedReserveSmallShard(t *testing.T, c *Cache) {
	t.Helper()
	n := 0
	for range reserveSmallPerPage {
		mustPut(t, c, relocKey(n/3), reserveSmallValue(n%256))
		n++
	}
	for p := 1; p < 3; p++ {
		for range reserveSmallPerPage {
			mustPut(t, c, relocKey(10_000+n), reserveSmallValue(n%256))
			n++
		}
	}
	mustPut(t, c, relocKey(90_000), reserveSmallValue(0))
	s := c.shards[0]
	s.mu.RLock()
	pages := len(s.pages)
	s.mu.RUnlock()
	if pages != 4 {
		t.Fatalf("seed left %d pages, want 4", pages)
	}
}

// TestReserveRelocationRespectsTheChunkBound is the lock-discipline assertion. A page
// here is hundreds of entries and each one is a copy plus an index repoint; holding
// the write lock for the whole page would stall every writer on the shard. The pass
// must therefore bound the work per ACQUISITION, not per page.
//
// The observer fires once per finished chunk, with s.mu released, so what it sees is
// exactly what one lock hold did.
func TestReserveRelocationRespectsTheChunkBound(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	seedReserveSmallShard(t, c)
	s := c.shards[0]

	var mu sync.Mutex
	var chunks [][2]int
	hook := func(entries, bytes int) {
		mu.Lock()
		chunks = append(chunks, [2]int{entries, bytes})
		mu.Unlock()
	}
	s.reserveChunkHook.Store(&hook)
	s.topUpFreeReserve()

	byteBudget := relocPageSize / relocateChunkBytesDivisor
	mu.Lock()
	defer mu.Unlock()
	if len(chunks) < 2 {
		t.Fatalf("the evacuation took %d chunk(s); it must not do a whole page under one lock hold", len(chunks))
	}
	moved := 0
	for i, ch := range chunks {
		if ch[0] > relocateChunkEntries {
			t.Fatalf("chunk %d moved %d entries under one lock hold, bound is %d", i, ch[0], relocateChunkEntries)
		}
		// An entry is never split, so the chunk that crosses the byte budget overshoots
		// it by at most the size of that one entry and then stops.
		if ch[1] > byteBudget+reserveSmallEntry {
			t.Fatalf("chunk %d moved %d bytes under one lock hold, bound is %d (+ at most one %d-byte entry)",
				i, ch[1], byteBudget, reserveSmallEntry)
		}
		moved += ch[0]
	}
	if moved <= relocateChunkEntries {
		t.Fatalf("only %d entries moved in total; the chunk bound was never actually crossed", moved)
	}
}

// TestReserveRelocationLetsAWriterThroughMidPass is the point of the chunking, stated
// as behaviour rather than as a bound: a writer must make progress while an evacuation
// is in flight. The observer blocks between two chunks — the pass is mid-page and
// holding nothing — and the write has to complete in that window.
func TestReserveRelocationLetsAWriterThroughMidPass(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	seedReserveSmallShard(t, c)
	s := c.shards[0]

	var once sync.Once
	wrote := make(chan error, 1)
	hook := func(int, int) {
		once.Do(func() {
			go func() { wrote <- c.Put(relocKey(77777), reserveSmallValue(9), 0) }()
			select {
			case err := <-wrote:
				wrote <- err // hand it back to the assertion below
			case <-time.After(20 * time.Second):
				wrote <- fmt.Errorf("writer did not complete while the pass was mid-page")
			}
		})
	}
	s.reserveChunkHook.Store(&hook)
	s.topUpFreeReserve()

	select {
	case err := <-wrote:
		if err != nil {
			t.Fatalf("concurrent write during a background pass: %v", err)
		}
	default:
		t.Fatal("the observer never fired; the pass did not chunk")
	}
	if v, err := c.Get(relocKey(77777)); err != nil || !bytes.Equal(v, reserveSmallValue(9)) {
		t.Fatalf("the write that went through mid-pass did not stick: %v", err)
	}
}

// TestReserveRelocationKeepsWritesOffTheSynchronousPass is the whole point of this
// layer: when the reserve keeps up, a steady-state write pays nothing for relocation.
// The sweeper is driven explicitly here rather than by its ticker, so "keeps up" is a
// fact of the test rather than a race with a clock — what is being asserted is that a
// keeping-up sweeper leaves the write path with NOTHING to relocate, not how fast a
// real one ticks.
func TestReserveRelocationKeepsWritesOffTheSynchronousPass(t *testing.T) {
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	// A COLD set written once — the records relocation exists to carry — under a HOT
	// set churned hard enough to keep every page filling with superseded versions. A
	// uniformly-churned keyspace would prove nothing here: its pages come round already
	// entirely dead, so the reserve would free pages without ever moving a record.
	const (
		coldKeys = 20
		hotKeys  = 40
	)
	// Reaching capacity for the FIRST time is not steady state: the shard fills by
	// GROWING pages, so the reserve has nothing to build from until an eviction has made
	// the first hole, and that eviction is the write path's. The baseline is therefore
	// taken at a fixed point past a full fill AND a full rotation of the ring, never at a
	// fraction of the run — a fraction of a -short run lands before capacity and folds
	// the fill straight into the measurement.
	const warmAt = 2 * 4 * reserveMediumPerPage
	ops := warmAt + 4*reserveMediumPerPage
	if testing.Short() {
		ops = warmAt + 2*reserveMediumPerPage
	}
	reserveSeedChurnSet(t, c, coldKeys)
	var warm Stats
	for n := range ops {
		mustPut(t, c, relocKey(500+n%hotKeys), reserveMediumValue(n%256))
		s.topUpFreeReserve()
		if n == warmAt {
			warm = c.Stats()
		}
	}

	st := c.Stats()
	if st.ReservePagesFreed-warm.ReservePagesFreed == 0 || st.ReserveRelocations-warm.ReserveRelocations == 0 {
		t.Fatalf("the reserve did no work (%d pages freed, %d records moved); the test proved nothing",
			st.ReservePagesFreed-warm.ReservePagesFreed, st.ReserveRelocations-warm.ReserveRelocations)
	}
	// The claim is about WHO pays, so it is asserted as a ratio. A write-path eviction
	// still happens from time to time — the reserve is a page of room, not an infinite
	// one — and #127's pass runs inside each of them, so the honest statement is that
	// the sweeper is carrying the overwhelming majority of the copying rather than that
	// the write path never does any.
	sync := st.EvictionRelocations - warm.EvictionRelocations
	bg := st.ReserveRelocations - warm.ReserveRelocations
	if sync*4 >= bg {
		t.Fatalf("the write path relocated %d records against the sweeper's %d; the copying has "+
			"not moved off the write path", sync, bg)
	}
	// Nothing live went out, from either path: the reserve pass only ever retires a page
	// it has already emptied of live records, and the write path's own evictions land on
	// pages the pass has already prepared. (Stats.Evictions counts the superseded copies
	// a retire drops and is bumped by the background pass's retire too, so it is not the
	// counter that says whether anything was LOST — this one is.)
	if n := st.EvictionsLive - warm.EvictionsLive; n != 0 {
		t.Fatalf("%d live records were lost while the sweeper was keeping up", n)
	}
	for i := range coldKeys {
		if _, err := c.Get(relocKey(i)); err != nil {
			t.Fatalf("cold key %d lost while the reserve was carrying the shard: %v", i, err)
		}
	}
}

// TestReserveRelocationFallsBackToTheWritePath is the other half of that claim. With
// the sweeper stopped, a burst outruns a reserve that is never topped up, and the
// write path must carry it exactly as it does today: the synchronous pass relocates
// and the background counters stay at zero.
func TestReserveRelocationFallsBackToTheWritePath(t *testing.T) {
	c, err := New(reserveConfig(4, true)) // reserveConfig leaves TTLSweepIntervalMs at 0
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const (
		coldKeys = 20
		hotKeys  = 40
	)
	// Enough writes to fill the shard, reach its cap, and then cycle the rotation
	// through every page several times over.
	ops := 12 * reserveMediumPerPage
	if testing.Short() {
		ops = 6 * reserveMediumPerPage
	}
	reserveSeedChurnSet(t, c, coldKeys)
	for n := range ops {
		mustPut(t, c, relocKey(500+n%hotKeys), reserveMediumValue(n%256))
	}

	st := c.Stats()
	if st.EvictionRelocations == 0 {
		t.Fatal("no write-path relocation with the sweeper stopped; the fallback never ran")
	}
	if st.ReserveRelocations != 0 || st.ReservePagesFreed != 0 {
		t.Fatalf("the background pass ran with no sweeper: %d records, %d pages",
			st.ReserveRelocations, st.ReservePagesFreed)
	}
}

// TestReserveRelocationRunsUnderItsOwnTicker is the wiring check the two tests above
// deliberately do not make: that Config.RelocateReserveIntervalMs actually starts a ticker
// that reaches topUpFreeReserve, WITHOUT the TTL sweeper running at all — which is the
// decoupling, stated as a test. It asserts only that the pass ran, never how much it kept
// up with, so there is nothing here for a slow machine to fail.
func TestReserveRelocationRunsUnderItsOwnTicker(t *testing.T) {
	cfg := reserveConfig(4, true)
	cfg.TTLSweepIntervalMs = 0 // the reserve must not need this one
	cfg.RelocateReserveIntervalMs = 5
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const hotKeys = 40
	reserveSeedChurnSet(t, c, 20)
	deadline := time.Now().Add(30 * time.Second)
	for n := 0; time.Now().Before(deadline); n++ {
		mustPut(t, c, relocKey(500+n%hotKeys), reserveMediumValue(n%256))
		if n%10 == 9 {
			time.Sleep(5 * time.Millisecond)
			if st := c.Stats(); st.ReservePagesFreed > 0 {
				return
			}
		}
	}
	t.Fatalf("the reserve ticker never freed a page in 30s: %d pages freed", c.Stats().ReservePagesFreed)
}

// TestReserveRelocationIntervalZeroRunsNoTicker pins the meaning of a zero interval: no
// reserve ticker at all, so an opted-in shard degrades to exactly the write-path pass —
// #127 and nothing else. It is the escape hatch for the cadence being wrong for a given
// shard count, and it has to work without also turning relocation off.
func TestReserveRelocationIntervalZeroRunsNoTicker(t *testing.T) {
	cfg := reserveConfig(4, true) // RelocatingEviction on, RelocateReserveIntervalMs 0
	cfg.TTLSweepIntervalMs = 5    // and the TTL sweeper ticking, which must not drive it
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	const hotKeys = 3
	reserveSeedChurnSet(t, c, 20)
	deadline := time.Now().Add(2 * time.Second)
	for n := 0; time.Now().Before(deadline); n++ {
		mustPut(t, c, relocKey(500+n%hotKeys), reserveMediumValue(n%256))
	}
	st := c.Stats()
	if st.ReservePagesFreed != 0 || st.ReserveRelocations != 0 {
		t.Fatalf("the reserve ran with no interval set: %d pages, %d records",
			st.ReservePagesFreed, st.ReserveRelocations)
	}
	if st.EvictionRelocations == 0 {
		t.Fatal("the write-path pass did not run either; the shard never reached its cap")
	}
}

// TestReserveIntervalValidation pins that the new field is validated like the interval it
// sits beside, so a negative is a config error rather than a ticker that never fires.
func TestReserveIntervalValidation(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.RelocateReserveIntervalMs != defaultRelocateReserveIntervalMs {
		t.Fatalf("DefaultConfig RelocateReserveIntervalMs = %d, want %d",
			cfg.RelocateReserveIntervalMs, defaultRelocateReserveIntervalMs)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("the default config does not validate: %v", err)
	}
	cfg.RelocateReserveIntervalMs = -1
	if err := cfg.Validate(); err == nil {
		t.Fatal("a negative reserve interval validated")
	}
}

// TestReserveRelocationConcurrentReadersNeverMiss runs the background pass against
// live readers and a writer. Relocation publishes the bytes before the ref that
// addresses them and retires a page only once every live record is off it, so a reader
// must resolve either the old copy or the new one at every instant — never a miss for a
// record the cache kept, and never a torn value. Run it under -race: this is the only
// test that has the background pass, the write path and lock-free readers all live at
// once.
func TestReserveRelocationConcurrentReadersNeverMiss(t *testing.T) {
	const (
		workingKeys = 6
		hotKeys     = 4
		valLen      = 40_000
	)
	writeOps := 40_000
	if testing.Short() {
		writeOps = 5_000
	}
	c, err := New(reserveConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]

	val := func(tag int) []byte {
		v := make([]byte, valLen)
		for i := range v {
			v[i] = byte(tag)
		}
		return v
	}
	ok := func(tag int, v []byte) bool {
		if len(v) != valLen {
			return false
		}
		for _, b := range v {
			if b != byte(tag) {
				return false
			}
		}
		return true
	}
	workKey := func(i int) []byte { return fmt.Appendf(nil, "work-%03d", i) }
	hotKey := func(i int) []byte { return fmt.Appendf(nil, "hot-%03d", i) }

	// Page 0 is drained with nothing having evacuated it — the rotation cursor starts
	// there and both passes always run one page ahead — so fill it with filler before
	// seeding anything that has to survive.
	fillerPerPage := relocPageSize/(entryHeaderSize+len(workKey(0))+valLen) + 1
	for i := range fillerPerPage {
		mustPut(t, c, fmt.Appendf(nil, "fill-%03d", i), val(0))
	}
	for i := range workingKeys {
		mustPut(t, c, workKey(i), val(100+i))
	}

	var (
		stop     atomic.Bool
		bad      atomic.Int64
		firstBad atomic.Value // string
		wg       sync.WaitGroup
	)
	record := func(desc string) {
		bad.Add(1)
		firstBad.CompareAndSwap(nil, desc)
	}
	for r := range 16 {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed)) //nolint:gosec // test RNG
			for !stop.Load() {
				i := rng.Intn(workingKeys)
				v, gerr := c.Get(workKey(i))
				switch {
				case gerr != nil:
					record(fmt.Sprintf("working key %d: %v on a record relocation keeps alive", i, gerr))
				case !ok(100+i, v):
					record(fmt.Sprintf("working key %d: malformed value %v", i, truncBytes(v)))
				}
			}
		}(int64(r) + 1)
	}
	// The sweeper's half, driven as hard as a ticker never would.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for !stop.Load() {
			s.topUpFreeReserve()
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for n := range writeOps {
			if err := c.Put(hotKey(n%hotKeys), val(n%hotKeys), 0); err != nil {
				record(fmt.Sprintf("Put: %v", err))
				break
			}
		}
		stop.Store(true)
	}()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(300 * time.Second):
		stop.Store(true)
		<-done
		t.Fatal("stress test did not finish within 300s")
	}

	if n := bad.Load(); n != 0 {
		t.Fatalf("%d bad reads of keys relocation keeps live; first: %v", n, firstBad.Load())
	}
	if st := c.Stats(); st.ReserveRelocations == 0 {
		t.Fatal("the background pass never relocated anything; the test proved nothing")
	}
}
