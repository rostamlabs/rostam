// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"testing"
	"time"
)

// A walk that finishes a page while the shard still has room must record where it
// finished, so the next round can tell the page apart from one it has never seen.
func TestReserveMarksAPageItFinishedPreparing(t *testing.T) {
	c, _ := newOnRemoveCache(t, reserveConfig(4, true))
	seedReserveShard(t, c, 4)
	s := c.shards[0]

	s.mu.RLock()
	victim := s.nextVictim
	p := s.pages[victim]
	tail := p.tail()
	s.mu.RUnlock()
	if p.preparedTail != 0 {
		t.Fatalf("preparedTail = %d before any walk, want 0", p.preparedTail)
	}

	// retire=false is the case under test: the pass prepares but does not retire.
	spent, freed := s.reserveMoveVictim(victim, p, s.cfg.PageSize, false)
	if freed {
		t.Fatalf("reserveMoveVictim retired a page with retire=false (spent %d)", spent)
	}

	s.mu.RLock()
	got := p.preparedTail
	s.mu.RUnlock()
	if got != tail || got == 0 {
		t.Errorf("preparedTail = %d after a completed walk, want the page's tail %d", got, tail)
	}
}

// The walk must NOT claim a page is prepared when it stopped early, or the page is
// never finished: the records it did not reach would be skipped from then on.
func TestReserveDoesNotMarkAPageItStoppedShortOn(t *testing.T) {
	c, _ := newOnRemoveCache(t, reserveConfig(4, true))
	seedReserveShard(t, c, 4)
	s := c.shards[0]

	s.mu.RLock()
	victim := s.nextVictim
	p := s.pages[victim]
	s.mu.RUnlock()

	// A budget of one header cannot move anything, so the walk stops short.
	if _, freed := s.reserveMoveVictim(victim, p, entryHeaderSize, false); freed {
		t.Fatal("a one-header budget retired a page")
	}

	s.mu.RLock()
	got := p.preparedTail
	s.mu.RUnlock()
	if got != 0 {
		t.Errorf("preparedTail = %d after a walk that stopped short, want 0", got)
	}
}

// The skip predicate is the whole behaviour change, so it is tested on its own: the
// fixtures above cannot hold a shard in the "has room" state, which is exactly the
// state the bug lived in.
func TestReserveSkipPrepared(t *testing.T) {
	newPrepared := func(t *testing.T) *page {
		t.Helper()
		p := newHeapPage(4096)
		if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
			t.Fatalf("Write: %v", err)
		}
		p.preparedTail = p.tail()
		return p
	}

	t.Run("skips a prepared page when there is room", func(t *testing.T) {
		if !reserveSkipPrepared(false, newPrepared(t)) {
			t.Error("walked a page already prepared; that is the repeated full-page walk this fixes")
		}
	})

	t.Run("walks it anyway when the shard is short of room", func(t *testing.T) {
		if reserveSkipPrepared(true, newPrepared(t)) {
			t.Error("skipped a page the shard needs retired")
		}
	})

	t.Run("walks a page that has grown since", func(t *testing.T) {
		p := newPrepared(t)
		if _, _, err := p.Write([]byte("grown"), []byte("v"), 0, 0); err != nil {
			t.Fatalf("Write: %v", err)
		}
		if reserveSkipPrepared(false, p) {
			t.Error("skipped a page whose tail moved; its new entries were never examined")
		}
	})

	t.Run("walks a page never prepared", func(t *testing.T) {
		p := newHeapPage(4096)
		if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
			t.Fatalf("Write: %v", err)
		}
		if reserveSkipPrepared(false, p) {
			t.Error("skipped a page the reserve has never walked")
		}
	})
}

// Reusing a page must clear the mark with everything else it clears, or the fresh page
// inherits a conclusion drawn about the one it replaced.
func TestPageResetClearsPreparedTail(t *testing.T) {
	p := newHeapPage(4096)
	if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
		t.Fatalf("Write: %v", err)
	}
	p.preparedTail = p.tail()
	p.relocatedOut = 7

	p.Reset()

	if p.preparedTail != 0 {
		t.Errorf("preparedTail = %d after Reset, want 0", p.preparedTail)
	}
	if p.relocatedOut != 0 {
		t.Errorf("relocatedOut = %d after Reset, want 0", p.relocatedOut)
	}
}

// Expiry is the one liveness test the walk cannot settle for good: it is read off a
// clock, not off the page. A page holding an entry that is still index-current but has
// expired still has a tombstone owing on it, so marking it prepared would leave that
// entry sitting there for as long as nothing appends to the page.
func TestReserveDoesNotMarkAPageHoldingAnExpiredEntry(t *testing.T) {
	c, _ := newOnRemoveCache(t, reserveConfig(4, true))
	clock := &fakeClock{}
	clock.set(1_000_000)
	c.SetNowFunc(clock.now)

	// seedReserveShard's shape, with the last of page 0's superseded copies replaced by
	// a key that will have expired by the time the walk reaches it. Page 0 still holds a
	// live record of its own, so the walk has something to carry off and nothing else
	// makes it stop short.
	hot := relocKey(900)
	for i := range relocPerPage - 1 {
		mustPut(t, c, hot, relocValue(200+i))
	}
	if err := c.Put(relocKey(901), relocValue(1), time.Second); err != nil {
		t.Fatalf("Put the perishable key: %v", err)
	}
	for p := 1; p < 3; p++ {
		for i := range relocPerPage {
			mustPut(t, c, relocKey(p*100+i), relocValue(i))
		}
	}
	mustPut(t, c, relocKey(9000), relocValue(1))

	s := c.shards[0]
	s.mu.RLock()
	pages, victim := len(s.pages), s.nextVictim
	p := s.pages[victim]
	s.mu.RUnlock()
	if pages != 4 {
		t.Fatalf("seed left %d pages, want the shard at its cap of 4", pages)
	}

	clock.set(1_002_000) // past the perishable key's TTL

	if _, freed := s.reserveMoveVictim(victim, p, s.cfg.PageSize, false); freed {
		t.Fatal("reserveMoveVictim retired a page with retire=false")
	}

	s.mu.RLock()
	got := p.preparedTail
	s.mu.RUnlock()
	if got != 0 {
		t.Errorf("preparedTail = %d after a walk that stepped over an expired entry, want 0", got)
	}
}
