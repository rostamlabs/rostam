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

// Layout used by the deterministic tests below. PageSize is the 1 MiB minimum and
// every entry is exactly relocEntrySize bytes, so a page holds exactly
// relocPerPage of them and the leftover tail is SMALLER than one entry — which
// pins the fill order: firstPageWithRoomLocked can never place a write in an older
// page's remainder, so writes always land in the frontier and pages fill in index
// order. Every assertion about which page holds what depends on that.
const (
	relocPageSize  = 1 << 20
	relocValueLen  = 200_000
	relocKeyLen    = 6 // "k00000"
	relocEntrySize = entryHeaderSize + relocKeyLen + relocValueLen
	relocPerPage   = relocPageSize / relocEntrySize // 5
)

// relocConfig is a single-shard heap ringbuf cache of `pages` 1 MiB pages with the
// background sweeper OFF, so every test drives eviction explicitly through Put.
func relocConfig(pages int, on bool) Config {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = relocPageSize
	cfg.MaxMemoryPerShard = pages * relocPageSize
	cfg.InitialPagesPerShard = 0
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.TTLSweepIntervalMs = 0
	cfg.RelocatingEviction = on
	return cfg
}

func relocKey(i int) []byte { return fmt.Appendf(nil, "k%05d", i) }

// relocValue returns a relocValueLen-byte value entirely of byte(tag), so a torn
// or cross-key read is visible as a non-uniform run or the wrong tag.
func relocValue(tag int) []byte {
	v := make([]byte, relocValueLen)
	for i := range v {
		v[i] = byte(tag)
	}
	return v
}

func relocUniform(v []byte, tag int) bool {
	if len(v) != relocValueLen {
		return false
	}
	want := byte(tag)
	for _, b := range v {
		if b != want {
			return false
		}
	}
	return true
}

func mustPut(t *testing.T, c *Cache, key, val []byte) {
	t.Helper()
	if err := c.Put(key, val, 0); err != nil {
		t.Fatalf("Put %q: %v", key, err)
	}
}

func mustGet(t *testing.T, c *Cache, key []byte) []byte {
	t.Helper()
	v, err := c.Get(key)
	if err != nil {
		t.Fatalf("Get %q: %v", key, err)
	}
	return v
}

// TestRelocatingEvictionKeepsLiveRecordOnADeadPage is the feature in one page. The
// source page holds relocPerPage framed copies of ONE key, of which exactly one is
// index-current; draining it without relocation drops that live record along with
// its own superseded versions. With relocation the live copy is carried into the
// space the previous eviction freed and only the dead versions are dropped.
func TestRelocatingEvictionKeepsLiveRecordOnADeadPage(t *testing.T) {
	for _, tc := range []struct {
		name string
		on   bool
	}{{"off", false}, {"on", true}} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := New(relocConfig(3, tc.on))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer func() { _ = c.Close() }()

			// Page 0: relocPerPage distinct keys, all live.
			for i := range relocPerPage {
				mustPut(t, c, relocKey(i), relocValue(i))
			}
			// Page 1: relocPerPage copies of ONE key — only the last is index-current.
			hot := relocKey(100)
			for i := range relocPerPage {
				mustPut(t, c, hot, relocValue(200+i))
			}
			// Page 2: relocPerPage more distinct keys, all live.
			for i := range relocPerPage {
				mustPut(t, c, relocKey(300+i), relocValue(i))
			}
			s := c.shards[0]
			if got := s.numPages(); got != 3 {
				t.Fatalf("expected the shard to be at its 3-page cap, got %d", got)
			}

			// One more write: the shard is full, so this evicts page 0 (the rotation
			// cursor starts there) and, with the flag on, evacuates page 1 into it.
			mustPut(t, c, relocKey(400), relocValue(7))

			st := c.Stats()
			if tc.on {
				if st.EvictionRelocations != 1 {
					t.Fatalf("EvictionRelocations = %d, want 1 (only the live copy of the hot key)", st.EvictionRelocations)
				}
				if st.EvictionBytesRelocated != relocEntrySize {
					t.Fatalf("EvictionBytesRelocated = %d, want %d", st.EvictionBytesRelocated, relocEntrySize)
				}
			} else if st.EvictionRelocations != 0 {
				t.Fatalf("flag off: EvictionRelocations = %d, want 0", st.EvictionRelocations)
			}

			// The hot key reads back correctly either way at this point — page 1 has
			// not been drained yet.
			if v := mustGet(t, c, hot); !relocUniform(v, 200+relocPerPage-1) {
				t.Fatalf("hot key: value not the newest copy")
			}

			// Fill the rest of the freed page and write once more, draining page 1.
			for i := range relocPerPage - 1 {
				mustPut(t, c, relocKey(500+i), relocValue(i))
			}
			// Page 1 is drained by this write.
			mustPut(t, c, relocKey(600), relocValue(9))

			v, gerr := c.Get(hot)
			if tc.on {
				if gerr != nil {
					t.Fatalf("hot key was dropped with the dead versions it shared page 1 with: %v", gerr)
				}
				if !relocUniform(v, 200+relocPerPage-1) {
					t.Fatalf("hot key: relocated value is not the newest copy")
				}
			} else if gerr == nil {
				t.Fatal("flag off: expected the hot key to be dropped with its page")
			}
		})
	}
}

// TestRelocatingEvictionDoesNotCountLiveEvictionOrFireOnRemove pins the ordering
// that makes a relocated record a MOVE rather than a LOSS: by the time the source
// page's retire walk runs, the record is no longer index-current, so the walk must
// neither tombstone it, nor count it in EvictionsLive, nor notify onRemove — a
// notification would drop a live key's postings from every derived index.
func TestRelocatingEvictionDoesNotCountLiveEvictionOrFireOnRemove(t *testing.T) {
	c, rec := newOnRemoveCache(t, relocConfig(3, true))

	for i := range relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	hot := relocKey(100)
	for i := range relocPerPage {
		mustPut(t, c, hot, relocValue(200+i))
	}
	for i := range relocPerPage {
		mustPut(t, c, relocKey(300+i), relocValue(i))
	}
	// Evict page 0 (evacuating page 1 into it), then fill and evict page 1.
	mustPut(t, c, relocKey(400), relocValue(7))
	liveAfterFirst := c.Stats().EvictionsLive
	for i := range relocPerPage - 1 {
		mustPut(t, c, relocKey(500+i), relocValue(i))
	}
	mustPut(t, c, relocKey(600), relocValue(9))

	if n := rec.count(string(hot)); n != 0 {
		t.Fatalf("onRemove fired %d times for a relocated key; it must fire only for records actually removed", n)
	}
	if got := c.Stats().EvictionsLive - liveAfterFirst; got != 0 {
		t.Fatalf("draining the page a relocated record came from counted %d live evictions, want 0", got)
	}
	if _, err := c.Get(hot); err != nil {
		t.Fatalf("relocated key is gone: %v", err)
	}
}

// TestRelocatingEvictionPreservesTriple checks that a move carries the exact
// (key, value, exp) triple, including an entry whose TTL is near but not yet
// reached — a relocated record must keep its original absolute expiry, not be
// silently refreshed by the copy.
func TestRelocatingEvictionPreservesTriple(t *testing.T) {
	c, err := New(relocConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	for i := range relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	// Page 1 holds one key with a live TTL, then dead copies of another key.
	ttlKey := relocKey(100)
	ttlVal := relocValue(42)
	const ttl = 10 * time.Minute
	before := nowMs()
	if err := c.Put(ttlKey, ttlVal, ttl); err != nil {
		t.Fatalf("Put ttl key: %v", err)
	}
	after := nowMs()
	s := c.shards[0]
	_, wantExp, err := s.getWithExpiryH(ttlKey, hashKey(ttlKey))
	if err != nil {
		t.Fatalf("read back ttl key: %v", err)
	}
	if wantExp < before+uint64(ttl.Milliseconds()) || wantExp > after+uint64(ttl.Milliseconds()) {
		t.Fatalf("seeded expiry %d outside the window the Put should have stamped", wantExp)
	}
	for i := range relocPerPage - 1 {
		mustPut(t, c, relocKey(200+i), relocValue(i))
	}
	for i := range relocPerPage {
		mustPut(t, c, relocKey(300+i), relocValue(i))
	}
	// Evict page 0, evacuating page 1 (including the TTL key) into it.
	mustPut(t, c, relocKey(400), relocValue(7))
	if n := c.Stats().EvictionRelocations; n == 0 {
		t.Fatal("nothing relocated")
	}
	// Drain page 1 so only the relocated copy can answer.
	for i := range relocPerPage - 1 {
		mustPut(t, c, relocKey(500+i), relocValue(i))
	}
	mustPut(t, c, relocKey(600), relocValue(9))

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

// putUntilEviction writes successive churn values until ONE write raises the
// eviction count, and returns the stats either side of exactly that write. It is
// how the tests below land on an eviction boundary without depending on arithmetic
// about which write happens to fill which page.
func putUntilEviction(t *testing.T, c *Cache, key []byte, tag int) (before, after Stats) {
	t.Helper()
	for n := range 10_000 {
		before = c.Stats()
		mustPut(t, c, key, relocValue((tag+n)%256))
		after = c.Stats()
		if after.Evictions > before.Evictions {
			return before, after
		}
	}
	t.Fatal("no eviction after 10000 writes")
	return
}

// TestRelocatingEvictionEntriesHeldWhileBytesFall is the shape of the win: across
// an eviction whose victim holds only dead versions — because the live record that
// was on it has already been carried forward — the index keeps every key it had
// while the page bytes behind them fall.
func TestRelocatingEvictionEntriesHeldWhileBytesFall(t *testing.T) {
	c, err := New(relocConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	// Page 0 is filler: it is drained by the first eviction with nothing having
	// evacuated it, so nothing that must survive may be seeded there.
	for i := range relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	// One cold record, which relocation is expected to keep alive indefinitely.
	cold := relocKey(100)
	mustPut(t, c, cold, relocValue(42))

	hot := relocKey(200)
	// First eviction: drains the filler page and evacuates the cold record.
	if _, after := putUntilEviction(t, c, hot, 0); after.EvictionRelocations == 0 {
		t.Fatal("the cold record was not carried forward")
	}
	// Second eviction: its victim is the page the cold record came off, now dead.
	beforeSt, afterSt := putUntilEviction(t, c, hot, 50)

	if afterSt.Entries != beforeSt.Entries {
		t.Fatalf("Entries %d -> %d; the eviction cost the index keys it should have kept",
			beforeSt.Entries, afterSt.Entries)
	}
	if afterSt.BytesUsed >= beforeSt.BytesUsed {
		t.Fatalf("BytesUsed %d -> %d; the dead page bytes were not reclaimed",
			beforeSt.BytesUsed, afterSt.BytesUsed)
	}
	if afterSt.EvictionsLive != beforeSt.EvictionsLive {
		t.Fatalf("EvictionsLive %d -> %d; a fully evacuated page lost live records",
			beforeSt.EvictionsLive, afterSt.EvictionsLive)
	}
	if v := mustGet(t, c, cold); !relocUniform(v, 42) {
		t.Fatal("cold record did not survive intact")
	}
}

// TestRelocatingEvictionRespectsByteCap checks the two bounds that keep relocation
// a passenger on the write it rides: it may spend at most
// PageSize/relocateMaxBytesPerEvictionDivisor bytes, and the triggering write must
// still succeed. The source page here is ENTIRELY live, so the cap is the only
// thing that can stop the pass.
func TestRelocatingEvictionRespectsByteCap(t *testing.T) {
	c, err := New(relocConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	for i := range 3 * relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i%256))
	}
	// Full: this write evicts page 0 and evacuates page 1, which is all live.
	trigger := relocKey(999)
	mustPut(t, c, trigger, relocValue(11))

	st := c.Stats()
	maxBytes := uint64(relocPageSize / relocateMaxBytesPerEvictionDivisor)
	if st.EvictionBytesRelocated > maxBytes {
		t.Fatalf("relocated %d bytes in one eviction, cap is %d", st.EvictionBytesRelocated, maxBytes)
	}
	if st.EvictionRelocations == 0 {
		t.Fatal("nothing relocated from an all-live page")
	}
	if st.EvictionRelocations >= uint64(relocPerPage) {
		t.Fatalf("relocated %d of %d entries; the cap should have stopped the pass short",
			st.EvictionRelocations, relocPerPage)
	}
	// The write that summoned the eviction is never starved by it.
	if v := mustGet(t, c, trigger); !relocUniform(v, 11) {
		t.Fatal("the triggering write did not land intact")
	}
	// Its room came off the top: the frontier still had space for it after relocation.
	if got := c.shards[0].pages[0].FreeTail(); got < 0 {
		t.Fatalf("frontier overrun: FreeTail = %d", got)
	}
}

// TestRelocatingEvictionOffIsUnchanged pins that the default is inert. It drives a
// shard of known geometry past its capacity with distinct keys and asserts the
// EXACT positional-drain outcome: pages are drained whole in rotation order, so the
// survivors are precisely the keys still framed on the pages that have not been
// drained, every dropped key counts as a live eviction, and neither relocation
// counter moves. Any relocation work leaking into the default path changes one of
// those three numbers.
func TestRelocatingEvictionOffIsUnchanged(t *testing.T) {
	const (
		pages = 4
		slots = pages * relocPerPage
		keys  = slots + relocPerPage // exactly one page's worth of overflow
	)
	c, err := New(relocConfig(pages, false))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	for i := range keys {
		mustPut(t, c, relocKey(i), relocValue(i%256))
	}

	st := c.Stats()
	if st.EvictionRelocations != 0 || st.EvictionBytesRelocated != 0 {
		t.Fatalf("flag off relocated something: %d records / %d bytes",
			st.EvictionRelocations, st.EvictionBytesRelocated)
	}
	// One page's worth of overflow drains exactly one page, and every key on it was
	// still the live copy for its key.
	if st.EvictionsLive != uint64(relocPerPage) {
		t.Fatalf("EvictionsLive = %d, want %d (one whole page of live records)", st.EvictionsLive, relocPerPage)
	}
	for i := range keys {
		_, gerr := c.Get(relocKey(i))
		alive := gerr == nil
		// The first page written is the first drained; everything after it survives.
		want := i >= relocPerPage
		if alive != want {
			t.Fatalf("key %d alive=%v, want %v — the positional drain order changed", i, alive, want)
		}
	}
}

// TestRelocatingEvictionSavesLiveRecordsUnderOverwriteChurn is the motivating shape
// end to end: a handful of cold records sharing pages with a key that is rewritten
// constantly. Positional eviction drops the cold records along with the dead
// versions crowding them out; relocation carries them forward instead.
func TestRelocatingEvictionSavesLiveRecordsUnderOverwriteChurn(t *testing.T) {
	const coldKeys = 2 // live bytes must fit one eviction's relocation budget
	run := func(on bool) (Stats, int) {
		c, err := New(relocConfig(3, on))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer func() { _ = c.Close() }()
		// Filler page first: the first eviction drains its victim unevacuated.
		for i := range relocPerPage {
			mustPut(t, c, relocKey(i), relocValue(i%256))
		}
		for i := range coldKeys {
			mustPut(t, c, relocKey(100+i), relocValue(100+i))
		}
		hot := relocKey(200)
		for n := range 200 {
			mustPut(t, c, hot, relocValue(n%256))
		}
		survivors := 0
		for i := range coldKeys {
			if v, gerr := c.Get(relocKey(100 + i)); gerr == nil && relocUniform(v, 100+i) {
				survivors++
			}
		}
		return c.Stats(), survivors
	}
	offStats, offAlive := run(false)
	onStats, onAlive := run(true)

	if offAlive != 0 {
		t.Fatalf("flag off kept %d cold records; the positional drain should have taken them all", offAlive)
	}
	if onAlive != coldKeys {
		t.Fatalf("flag on kept %d of %d cold records", onAlive, coldKeys)
	}
	if onStats.EvictionsLive >= offStats.EvictionsLive {
		t.Fatalf("EvictionsLive not reduced: off=%d on=%d", offStats.EvictionsLive, onStats.EvictionsLive)
	}
	if onStats.EvictionRelocations == 0 {
		t.Fatal("flag on relocated nothing")
	}
}

// TestRelocatedKeyResolvesFromAStaleRef is the deterministic control for the
// reader fix. It reproduces the ONE state a lock-free reader can be caught in —
// holding a ref it loaded before the key was relocated, resolving the page only
// after the source page was retired — and asserts the probe still finds the record.
//
// The stress test below drives the same hazard through real concurrency, but the
// window is two atomic loads wide and the two writer events that open it are a
// page of writes apart, so it is not a control: a passing stress run does not mean
// the branch is doing anything. This does. Replace rechaseSlot's body with
// `return nil, r` — exactly the unconditional `continue` the reader did before —
// and this test fails, because that is the moment a live record reads back as
// ErrNotFound.
func TestRelocatedKeyResolvesFromAStaleRef(t *testing.T) {
	c, err := New(relocConfig(3, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	// Filler page, then the record whose ref the reader will be holding.
	for i := range relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i))
	}
	cold := relocKey(100)
	mustPut(t, c, cold, relocValue(42))

	s := c.shards[0]
	h := hashKey(cold)
	// The reader's state at the instant it is descheduled: it has loaded the ref for
	// this key and not yet resolved the page behind it.
	_, staleRef, ok := s.tab.Load().findSlot(h)
	if !ok {
		t.Fatal("seeded key is not in the index")
	}

	// Churn until the record has been relocated AND the page its stale ref points
	// into has been retired — the two writer events the reader is caught between.
	hot := relocKey(200)
	for n := 0; ; n++ {
		if n > 10_000 {
			t.Fatal("the seeded record was never relocated and retired out from under its ref")
		}
		mustPut(t, c, hot, relocValue(n%256))
		p := s.pageSlots[staleRef.pageIdx()].Load()
		if p != nil && p.gen == staleRef.gen() {
			continue // the stale ref still resolves; the hazard has not opened yet.
		}
		if c.Stats().EvictionRelocations > 0 {
			break
		}
	}

	// Precondition: the ref the reader holds no longer passes the generation gate,
	// so the pre-fix reader would have advanced to the next probe slot here.
	if p := s.pageSlots[staleRef.pageIdx()].Load(); p != nil && p.gen == staleRef.gen() {
		t.Fatal("stale ref still resolves; the test did not reach the state it is about")
	}
	// And the record is genuinely still live, so advancing would be a wrong answer.
	if v := mustGet(t, c, cold); !relocUniform(v, 42) {
		t.Fatal("the record under test is not live")
	}

	tab := s.tab.Load()
	slot, _, ok := tab.findSlot(h)
	if !ok {
		t.Fatal("relocated key lost its index slot")
	}
	p, cur := tab.rechaseSlot(s, slot, staleRef)
	if p == nil {
		t.Fatal("a relocated live record resolved as gone: the probe would walk past its own slot and report ErrNotFound")
	}
	k, v, _, rerr := p.Read(cur.offset())
	if rerr != nil {
		t.Fatalf("rechased ref does not decode: %v", rerr)
	}
	if !bytes.Equal(k, cold) {
		t.Fatalf("rechased ref resolved to key %q, want %q", k, cold)
	}
	if !relocUniform(v, 42) {
		t.Fatal("rechased ref resolved to the wrong value")
	}
}

// TestRelocatingEvictionConcurrentReadersNeverMiss is the test the reader fix in
// indexTable.rechaseSlot exists for. A pool of readers hammers a working set that
// relocation keeps alive indefinitely, while a writer churns a hot set hard enough
// to evict continuously. Every read of a working-set key must hit, with the exact
// value: a relocated key whose slot the reader walks PAST (because the page
// generation moved under it) comes back as ErrNotFound, which is a wrong answer for
// a record the cache deliberately kept.
//
// It is a GUARD, not the control — see TestRelocatedKeyResolvesFromAStaleRef for
// that. What it does cover that the control cannot: the relocation path running
// concurrently with real readers (run it under -race), and the values those readers
// get back being whole.
func TestRelocatingEvictionConcurrentReadersNeverMiss(t *testing.T) {
	const (
		workingKeys = 6
		hotKeys     = 4
	)
	writeOps := 60_000
	if testing.Short() {
		writeOps = 8_000
	}

	cfg := relocConfig(4, true)
	// Smaller entries than the deterministic tests use: the whole live set (working
	// + hot) must fit one eviction's relocation budget, or a working-set key would
	// be legitimately dropped and the assertion below would be wrong rather than
	// failing. It stays a uniform size so writes still always land in the frontier.
	const valLen = 40_000
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

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

	// Fill the FIRST page with filler before seeding the working set. Page 0 is the
	// one page drained with nothing having evacuated it (the rotation cursor starts
	// there and pre-evacuation always runs one eviction ahead), so a working-set key
	// seeded into it would be legitimately lost and the assertion below would be
	// wrong rather than failing.
	fillerPerPage := relocPageSize/(entryHeaderSize+len(workKey(0))+valLen) + 1
	for i := range fillerPerPage {
		mustPut(t, c, fmt.Appendf(nil, "fill-%03d", i), val(0))
	}
	for i := range workingKeys {
		mustPut(t, c, workKey(i), val(100+i))
	}
	for i := range workingKeys {
		if v := mustGet(t, c, workKey(i)); !ok(100+i, v) {
			t.Fatalf("seeded working key %d did not read back", i)
		}
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
	for r := range 32 {
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
	if st := c.Stats(); st.EvictionRelocations == 0 {
		t.Fatal("no relocation happened; the test proved nothing")
	}
}

// BenchmarkRelocatingEvictionAB is the A/B harness: two shards of identical
// geometry driven by the SAME seed over the SAME key population, one with
// relocating eviction off and one with it on.
//
// The workload is the shape the feature exists for — a Zipfian overwrite stream, so
// a few keys are rewritten constantly (piling dead versions onto every page) while
// the tail of the population is written rarely and is exactly what a positional
// drain throws away. ns/op is the honest cost side: the timed loop is Put only, so
// any write-path regression from the relocation copy shows up there directly.
//
// Reported per arm:
//
//	hit_rate       fraction of the key population still resident at the end (one
//	               untimed Get per key) — what the cache is actually holding
//	entries        index keys held at the end
//	B/live_key     BytesUsed / Entries: page occupancy per key held, dead versions
//	               included, so it rises when a shard is carrying garbage
//	evict/op       page entries dropped per write
//	evict_live/op  of those, the ones that were still the live copy for their key —
//	               the loss relocation is meant to remove
//	reloc/op       records copied forward per write
//	reloc_B/op     bytes those copies moved per write — the price of the saves
func BenchmarkRelocatingEvictionAB(b *testing.B) {
	relocABArms(b, "")
}

// relocABArms runs the off/on pair over one storage mode: dataDir "" is heap, a
// directory is mmap. Both arms of a pair get the same seed and the same key
// population, so the only difference between the two rows is the flag.
func relocABArms(b *testing.B, dataDir string) {
	keys := make([][]byte, relocABKeyspace)
	for i := range keys {
		keys[i] = fmt.Appendf(nil, "ab-%08d", i)
	}
	val := make([]byte, relocABValueLen)
	for i := range val {
		val[i] = byte(i)
	}

	for _, arm := range []struct {
		name string
		on   bool
	}{{"off", false}, {"on", true}} {
		b.Run("reloc="+arm.name, func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.NumShards = 1
			cfg.PageSize = 1 << 20
			cfg.MaxMemoryPerShard = relocABPages << 20
			cfg.InitialPagesPerShard = 0
			cfg.AtCapPolicy = PolicyRingbufEvict
			cfg.TTLSweepIntervalMs = 0
			cfg.RelocatingEviction = arm.on
			dir := dataDir
			if dir != "" {
				dir = b.TempDir() // a fresh pages file per arm
				cfg.DataDir = dir
			}
			s, err := newShard(cfg, dir, nil)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = s.Close() }()

			rng := rand.New(rand.NewSource(relocABSeed)) //nolint:gosec // benchmark RNG
			z := rand.NewZipf(rng, relocABZipfS, 1, relocABKeyspace-1)
			for range relocABWarmOps {
				_ = s.Put(keys[z.Uint64()], val, 0)
			}
			warm := s.snapshot()

			ops := 0
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				_ = s.Put(keys[z.Uint64()], val, 0)
				ops++
			}
			b.StopTimer()

			st := s.snapshot()
			hits := 0
			for _, k := range keys {
				if _, gerr := s.Get(k); gerr == nil {
					hits++
				}
			}
			perOp := func(d uint64) float64 {
				if ops == 0 {
					return 0
				}
				return float64(d) / float64(ops)
			}
			bytesPerKey := 0.0
			if st.Entries > 0 {
				bytesPerKey = float64(st.BytesUsed) / float64(st.Entries)
			}
			b.ReportMetric(float64(hits)/float64(relocABKeyspace), "hit_rate")
			b.ReportMetric(float64(st.Entries), "entries")
			b.ReportMetric(bytesPerKey, "B/live_key")
			b.ReportMetric(perOp(st.Evictions-warm.Evictions), "evict/op")
			b.ReportMetric(perOp(st.EvictionsLive-warm.EvictionsLive), "evict_live/op")
			b.ReportMetric(perOp(st.EvictionRelocations-warm.EvictionRelocations), "reloc/op")
			b.ReportMetric(perOp(st.EvictionBytesRelocated-warm.EvictionBytesRelocated), "reloc_B/op")
		})
	}
}

// Shared A/B geometry, so the heap and mmap rows are the same experiment run over
// two storage modes rather than two experiments.
const (
	relocABKeyspace = 12_000
	relocABValueLen = 400
	relocABPages    = 8
	relocABZipfS    = 1.3
	relocABSeed     = 0x5EED
	relocABWarmOps  = 400_000
)

// pageObjects snapshots the shard's page object identities. A heap eviction RETIRES
// its victim — swapping in a fresh object — so comparing two snapshots names exactly
// which page an eviction drained, without the test having to predict it.
func pageObjects(s *shard) []*page {
	out := make([]*page, len(s.pages))
	copy(out, s.pages)
	return out
}

// changedPage returns the single index whose page object was replaced between two
// snapshots, or -1 if none was.
func changedPage(before, after []*page) int {
	for i := range before {
		if before[i] != after[i] {
			return i
		}
	}
	return -1
}

// indexPagesByHash snapshots which PAGE each live index slot currently resolves to,
// keyed by the slot's full key hash so the snapshot survives a rehash. Comparing two
// of these is how the test below observes which page relocation actually read from,
// rather than recomputing the selection it was supposed to make.
func indexPagesByHash(s *shard) map[uint64]uint16 {
	t := s.tab.Load()
	out := make(map[uint64]uint16, t.live)
	for i := range t.ctrl {
		c := t.ctrl[i].Load()
		if c == ctrlEmpty || c == ctrlTombstone {
			continue
		}
		out[t.hashes[i]] = slabRef(t.refs[i].Load()).pageIdx()
	}
	return out
}

// TestRelocatingEvictionEvacuatesThePageTheNextEvictionDrains pins the alignment the
// whole pass rests on: the page relocation evacuates at one eviction must be the page
// the NEXT eviction drains. Both selections come from nextNonEmptyPageLocked over the
// same cursor — eviction with skip = -1, relocation with skip = the page just freed,
// which sorts last from that cursor — so today they agree by construction.
//
// If they ever stop agreeing, nothing fails loudly: relocation quietly copies records
// out of a page that is not about to be drained, which is write amplification that
// saves nothing, and the crash-consistency argument in cache/relocate_evict.go loses
// its premise (every moved record being one the next drain was going to delete
// anyway). This test is the alarm for that, so it observes both halves rather than
// deriving either: the source by watching which page the relocated slots were
// repointed AWAY from, and the drained page by page-object identity.
func TestRelocatingEvictionEvacuatesThePageTheNextEvictionDrains(t *testing.T) {
	c, err := New(relocConfig(4, true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	s := c.shards[0]
	hot := relocKey(900)
	hotHash := hashKey(hot)

	// Distinct live records on every page, so each eviction has something to move.
	for i := range 4 * relocPerPage {
		mustPut(t, c, relocKey(i), relocValue(i%256))
	}
	putUntilEviction(t, c, hot, 0)

	checked := 0
	for round := range 4 {
		// Eviction N. Watch which page the records that landed on the freed page were
		// repointed away from: that page IS relocation's source.
		beforeObjs, beforePages := pageObjects(s), indexPagesByHash(s)
		putUntilEviction(t, c, hot, 10*round)
		freed := changedPage(beforeObjs, pageObjects(s))
		if freed < 0 {
			t.Fatalf("round %d: no page was retired by an eviction", round)
		}
		sources := map[uint16]int{}
		for h, now := range indexPagesByHash(s) {
			if int(now) != freed || h == hotHash {
				continue // not moved here, or the triggering write itself
			}
			if was, ok := beforePages[h]; ok && int(was) != freed {
				sources[was]++
			}
		}
		if len(sources) == 0 {
			continue // nothing was relocated this round; it proves nothing either way
		}
		if len(sources) != 1 {
			t.Fatalf("round %d: relocation drew from %d pages (%v); it must evacuate exactly one",
				round, len(sources), sources)
		}
		var evacuated int
		for p := range sources {
			evacuated = int(p)
		}

		// Eviction N+1 must drain exactly that page.
		beforeObjs = pageObjects(s)
		putUntilEviction(t, c, hot, 10*round+5)
		drained := changedPage(beforeObjs, pageObjects(s))
		if drained != evacuated {
			t.Fatalf("round %d: relocation evacuated page %d but the next eviction drained page %d; "+
				"the victim cursor and the relocation source have drifted apart",
				round, evacuated, drained)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no round observed a relocation; the alignment was never actually checked")
	}
}
