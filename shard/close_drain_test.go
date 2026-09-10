// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// newStoreWithSlowRead builds a single-node store whose registry carries one
// extra OpReadOnly op that parks until the test lets it go — a stand-in for the
// real slow read (a scan-mode kv_query over a large keyspace) without needing
// one.
func newStoreWithSlowRead(t *testing.T, entered chan<- struct{}, release <-chan struct{}) *Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	if err := reg.Register("__slow_read__", ops.OpReadOnly, func(tx *ops.TxContext, _ []byte) ([]byte, error) {
		// Touch the cache the way a real read does, so this is not merely a
		// sleeping goroutine: the bytes a handler holds are page-backed, which
		// is the whole reason Close must not unmap under it.
		_, _ = tx.Get([]byte("k")) //nolint:errcheck // the key may not exist; the read is the point
		close(entered)
		<-release
		return []byte("done"), nil
	}); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	waitLeader(t, s)
	return s
}

// Close must not unmap the cache under a Call that is still running.
//
// THE SHAPE THIS PROTECTS. A cluster read is fanned out to every shard group and
// each leg carries its own timeout; a leg that times out is ABANDONED by the
// coordinator and keeps running (cluster.forEachGroup says so in as many words).
// If RemoveShardOwner or Node.Close lands in that window, the abandoned handler
// is holding bytes that point into a mapping cache.Close is about to tear down —
// which is a segfault, not a stale read. It was reproduced exactly that way.
func TestStoreCloseDrainsInFlightCalls(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	s := newStoreWithSlowRead(t, entered, release)

	done := make(chan error, 1)
	go func() {
		_, err := s.Call("__slow_read__", nil)
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the slow read never started")
	}

	closed := make(chan error, 1)
	go func() { closed <- s.Close() }()

	// Close MUST still be waiting: returning here is the bug, because the unmap
	// happens inside it.
	select {
	case err := <-closed:
		t.Fatalf("Close returned (err=%v) while a Call was still inside the store", err)
	case <-time.After(300 * time.Millisecond):
	}

	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the in-flight Call failed: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the in-flight Call never returned")
	}
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not return after the Call drained")
	}

	// And a Call arriving after Close is REFUSED rather than allowed to walk into
	// a closed cache — without that, the drain would be a window, not a fence.
	if _, err := s.Call("get", ops.EncodeKeyArgs([]byte("k"))); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("a Call after Close returned %v, want ErrStoreClosed", err)
	}
}

// The bound is what keeps a stuck handler from making a store unclosable. Close
// waits, gives up, and proceeds.
func TestStoreCloseIsBoundedWhenACallIsStuck(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	s := newStoreWithSlowRead(t, entered, release)

	var stuck atomic.Bool
	returned := make(chan struct{})
	go func() {
		defer close(returned)
		_, _ = s.Call("__slow_read__", nil) //nolint:errcheck // the call is deliberately abandoned
		stuck.Store(true)
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the slow read never started")
	}

	// Shrink the bound for the test by draining with a short one first: the real
	// Close then finds the door already shut and nothing new can enter.
	if s.drainCalls(150 * time.Millisecond) {
		t.Fatal("drainCalls reported the store quiet while a Call was parked in a handler")
	}
	if stuck.Load() {
		t.Fatal("fixture: the parked Call returned on its own")
	}
	// New calls are refused from the moment the drain begins, bound or no bound.
	if _, err := s.Call("get", ops.EncodeKeyArgs([]byte("k"))); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("a Call during the drain returned %v, want ErrStoreClosed", err)
	}

	// Torn down IN THE TEST BODY, not in a cleanup, and in this exact order: the
	// parked call is released, waited for, and only then is the store closed —
	// leaving the store open would hand t.TempDir's own cleanup a directory to
	// delete out from under a live raft goroutine, which panics inside
	// hashicorp/raft and reads as an unrelated flake.
	close(release)
	<-returned
	if err := s.Close(); err != nil {
		t.Fatalf("Close after the drain: %v", err)
	}
}

// A second Close (and the ordinary no-calls-in-flight case) must not block or
// double-close the idle channel.
func TestStoreCloseDrainIsIdempotent(t *testing.T) {
	s := newSingleNodeStore(t)
	if !s.drainCalls(time.Second) {
		t.Fatal("drainCalls reported busy on an idle store")
	}
	if !s.drainCalls(time.Second) {
		t.Fatal("a second drain on an idle store reported busy")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
