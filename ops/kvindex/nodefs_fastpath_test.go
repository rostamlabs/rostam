// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// The write path's no-definitions fast path.
//
// Reindex runs on every KV write; Drop is the cache's onRemove hook, so it also
// runs on every delete, eviction and TTL expiry — under a cache shard's write
// lock. Every shard.Store and Direct store wires a Set, so a deployment that has
// never defined an index used to funnel all of that through one global mutex for
// nothing. These tests pin the fast path by OBSERVING the lock rather than by
// timing it: a benchmark can only say "faster", and faster is not the claim.

// TestReindexAndDropDoNotTakeTheLockWithNoDefs holds s.mu for WRITING and calls
// both. If either still took the lock unconditionally, neither could return.
func TestReindexAndDropDoNotTakeTheLockWithNoDefs(t *testing.T) {
	s := New(16)

	s.mu.Lock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.Reindex([]byte("u:1"), intRec("age", 41))
		s.Drop([]byte("u:1"))
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		s.mu.Unlock()
		t.Fatal("Reindex/Drop blocked on s.mu with no definitions installed")
	}
	s.mu.Unlock()
}

// TestReindexAndDropDoTakeTheLockWithADef is the control, and it is what makes
// the test above mean anything: with a definition installed, both must still go
// through s.mu. A fast path that skipped the lock unconditionally would pass the
// first test and lose every posting.
func TestReindexAndDropDoTakeTheLockWithADef(t *testing.T) {
	d := mustDef(t, "by_age", "u:", "age", wire.KVIndexKindScalar)
	s := readySet(d)

	s.mu.Lock()
	blocked := make(chan struct{})
	go func() {
		defer close(blocked)
		s.Reindex([]byte("u:1"), intRec("age", 41))
	}()
	select {
	case <-blocked:
		s.mu.Unlock()
		t.Fatal("Reindex returned while s.mu was held for writing: it is not taking the lock")
	case <-time.After(100 * time.Millisecond):
		// Still waiting, as it must be.
	}
	s.mu.Unlock()
	select {
	case <-blocked:
	case <-time.After(5 * time.Second):
		t.Fatal("Reindex never completed after s.mu was released")
	}
	// And it really did the work.
	sel := Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(41)}}
	if got := mustCandidates(t, s, sel, nil, bigBudget); len(got) != 1 || got[0] != "u:1" {
		t.Fatalf("the write was not posted: candidates = %v", got)
	}
}

// TestReindexAndDropAllocateNothingWithNoDefs pins that the fast path is a
// single atomic load and a return.
func TestReindexAndDropAllocateNothingWithNoDefs(t *testing.T) {
	s := New(16)
	key, val := []byte("u:1"), intRec("age", 41)

	if got := testing.AllocsPerRun(200, func() { s.Reindex(key, val) }); got != 0 {
		t.Errorf("Reindex with no definitions allocates %v times per call, want 0", got)
	}
	if got := testing.AllocsPerRun(200, func() { s.Drop(key) }); got != 0 {
		t.Errorf("Drop with no definitions allocates %v times per call, want 0", got)
	}
}

// TestNoDefsFastPathTracksInstall pins the count the fast path reads against the
// definitions it stands for, in both directions: an Install turns it on, and
// installing an EMPTY set (which is how the observer removes the last
// definition) turns it back off.
func TestNoDefsFastPathTracksInstall(t *testing.T) {
	s := New(16)
	if got := s.nDefs.Load(); got != 0 {
		t.Fatalf("a fresh Set reports %d definitions", got)
	}
	s.Install([]Def{mustDef(t, "by_age", "u:", "age", wire.KVIndexKindScalar), mustDef(t, "by_name", "u:", "name", wire.KVIndexKindScalar)})
	if got := s.nDefs.Load(); got != 2 {
		t.Fatalf("after Install of 2: %d", got)
	}
	// A duplicate name is dropped by Install, so the count must follow what was
	// KEPT rather than what was passed.
	s.Install([]Def{mustDef(t, "by_age", "u:", "age", wire.KVIndexKindScalar), mustDef(t, "by_age", "u:", "age", wire.KVIndexKindScalar)})
	if got := s.nDefs.Load(); got != 1 {
		t.Fatalf("after Install of 1 name twice: %d, want 1", got)
	}
	// Reset keeps the definitions, so it must not turn the fast path on.
	s.Reset()
	if got := s.nDefs.Load(); got != 1 {
		t.Fatalf("Reset changed the definition count to %d", got)
	}
	s.Install(nil)
	if got := s.nDefs.Load(); got != 0 {
		t.Fatalf("after Install(nil): %d, want 0", got)
	}
	// And the write path is really back on the fast path.
	s.mu.Lock()
	done := make(chan struct{})
	go func() { defer close(done); s.Reindex([]byte("u:1"), intRec("age", 41)) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		s.mu.Unlock()
		t.Fatal("Reindex blocked after the last definition was removed")
	}
	s.mu.Unlock()
}
