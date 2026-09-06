// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// Tests for two durability fixes:
//   BUG A — acked deletes must be durable: a Delete whose WAL append/fsync fails
//           must return an ERROR (not a false ok), so a client is never told a
//           point is gone when it can resurrect after a crash.
//   BUG B — fsyncgate: on the FIRST f.Sync() failure the WAL is POISONED and
//           fails closed (ErrWALPoisoned) until reopen, so no later writer is
//           acked on bytes a retried-and-falsely-successful Sync left un-durable.
//
// Both use the existing beforeSync seam to force a deterministic Sync() error by
// closing the underlying fd right before the leader's f.Sync() (Sync on a closed
// *os.File returns an error) — no production-only test seam is added.

// TestDeleteCASPropagatesWALSyncError is the BUG A proof at the production call
// path: a WAL-mode DeleteCAS whose durability fsync fails returns (false, err)
// instead of the pre-fix (true, nil) false-ack.
func TestDeleteCASPropagatesWALSyncError(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)
	if _, err := c.InsertCASKeyTTL(1, vec, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Force the DELETE's fsync to fail: close the WAL fd right before its Sync.
	var once sync.Once
	c.wal.beforeSync = func() { once.Do(func() { _ = c.wal.f.Close() }) }

	removed, derr := c.DeleteCAS(1, CASCond{})
	if derr == nil {
		t.Fatal("DeleteCAS returned nil error despite a failed WAL fsync — false ack; the point can resurrect after a crash")
	}
	if removed {
		t.Fatalf("DeleteCAS returned removed=true on a durability failure; want removed=false")
	}
	// The failed fsync must also have poisoned the WAL (BUG B), so a follow-up
	// write is fail-closed rather than a fresh false success.
	if !c.wal.poisoned.Load() {
		t.Fatal("WAL not poisoned after the delete's fsync failed")
	}
	// A no-op delete of an absent id on the NOW-POISONED WAL stages nothing
	// (commitWaitStaged(0) is a no-op), so it must still succeed as (false, nil):
	// a no-op touches no durability and must not be turned into ErrWALPoisoned.
	noop, nerr := c.DeleteCAS(4242, CASCond{})
	if nerr != nil {
		t.Fatalf("no-op DeleteCAS on a poisoned WAL returned %v, want nil (no-op touches no durability)", nerr)
	}
	if noop {
		t.Fatal("no-op DeleteCAS reported removed=true for an absent id")
	}
}

// TestDeleteCASNoOpStillSucceeds guards the seq==0 no-op case: deleting an absent
// id stages nothing (commitWaitStaged(0) is a legit no-op), so it must stay a
// (false, nil) success — the fix must not turn a not-removed delete into an error.
func TestDeleteCASNoOpStillSucceeds(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	removed, derr := c.DeleteCAS(999, CASCond{}) // never inserted
	if derr != nil {
		t.Fatalf("no-op DeleteCAS returned error %v, want nil", derr)
	}
	if removed {
		t.Fatal("no-op DeleteCAS reported removed=true for an absent id")
	}
}

// TestWALPoisonedAfterSyncFailure is the BUG B proof on the raw wal: after a
// forced Sync() failure (1) commitWait returns the error AND the wal is poisoned;
// (2) a subsequent append/commit is fail-closed with ErrWALPoisoned; (3) a fresh
// reopen clears the poison and works again.
func TestWALPoisonedAfterSyncFailure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "p.wal")
	w, err := openWAL(path, false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}

	var once sync.Once
	w.beforeSync = func() { once.Do(func() { _ = w.f.Close() }) }

	// (1) the fsync fails: appendFramed surfaces the error and poisons the wal.
	if err := w.appendFramed([]byte{byte(walDelete), 1, 2, 3}); err == nil {
		t.Fatal("appendFramed returned nil despite a failed fsync (false ack)")
	}
	if !w.poisoned.Load() {
		t.Fatal("wal not poisoned after fsync failure")
	}

	// (2) fail-closed: further appends and commit-waits return ErrWALPoisoned.
	if _, aerr := w.appendDeleteStaged(7); !errors.Is(aerr, ErrWALPoisoned) {
		t.Fatalf("appendDeleteStaged on poisoned wal = %v, want ErrWALPoisoned", aerr)
	}
	if _, aerr := w.appendInsertStaged(8, []float32{1, 2}, 0, nil, nil, nil, 0); !errors.Is(aerr, ErrWALPoisoned) {
		t.Fatalf("appendInsertStaged on poisoned wal = %v, want ErrWALPoisoned", aerr)
	}
	if cerr := w.commitWaitStaged(1); !errors.Is(cerr, ErrWALPoisoned) {
		t.Fatalf("commitWaitStaged on poisoned wal = %v, want ErrWALPoisoned", cerr)
	}
	_ = w.close()

	// (3) a fresh reopen clears the poison and works again.
	w2, err := openWAL(path, false)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = w2.close() }()
	if w2.poisoned.Load() {
		t.Fatal("reopened wal is still poisoned — poison must clear on reopen")
	}
	if _, aerr := w2.appendDeleteStaged(9); aerr != nil {
		t.Fatalf("append after reopen failed: %v", aerr)
	}
	if cerr := w2.commitWaitStaged(w2.writeSeq); cerr != nil {
		t.Fatalf("commit after reopen failed: %v", cerr)
	}
}

// TestWALPoisonWakesParkedFollower proves the fail-closed path for a CONCURRENT
// follower: when the in-flight leader's Sync fails and poisons the wal, a writer
// parked in commitWait wakes and returns ErrWALPoisoned rather than becoming the
// next leader and retrying the (fsyncgate-unsafe) Sync.
func TestWALPoisonWakesParkedFollower(t *testing.T) {
	dir := t.TempDir()
	w, err := openWAL(filepath.Join(dir, "pf.wal"), false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	defer func() { _ = w.close() }()

	var once sync.Once
	parked := make(chan struct{})
	gate := make(chan struct{})
	w.beforeSync = func() {
		once.Do(func() {
			close(parked) // a leader has captured its target and is about to Sync
			<-gate        // hold until the follower has queued behind us
			_ = w.f.Close()
		})
	}

	leaderErr := make(chan error, 1)
	go func() { leaderErr <- w.appendFramed([]byte{byte(walDelete), 1}) }()
	<-parked // leader is parked in beforeSync (syncing == true)

	followerErr := make(chan error, 1)
	go func() { followerErr <- w.appendFramed([]byte{byte(walDelete), 2}) }()

	// Wait until the follower has written its bytes (writeSeq == 2) so it is
	// queued behind the in-flight leader before we let the leader's Sync fail. A
	// deadline expiry means the follower never parked — the path under test was
	// NOT exercised, so this must FAIL (not a silent break / false pass).
	ready := false
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		w.syncMu.Lock()
		ws := w.writeSeq
		w.syncMu.Unlock()
		if ws == 2 {
			ready = true
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !ready {
		close(gate)   // release the parked leader so it can finish its in-flight Sync
		<-leaderErr   // drain it BEFORE t.Fatal so the deferred w.close() can't race the leader's f.Sync()
		<-followerErr // and the follower, for the same reason
		t.Fatal("follower never wrote its record (writeSeq did not reach 2) — parked-follower path not exercised")
	}
	close(gate) // let the leader's Sync run (and fail)

	if lerr := <-leaderErr; lerr == nil {
		t.Fatal("leader appendFramed returned nil despite a failed fsync")
	}
	if ferr := <-followerErr; !errors.Is(ferr, ErrWALPoisoned) {
		t.Fatalf("parked follower = %v, want ErrWALPoisoned (must not retry the failed Sync)", ferr)
	}
	if !w.poisoned.Load() {
		t.Fatal("wal not poisoned after the leader's fsync failed")
	}
}

// TestWALTruncateSyncFailurePoisons proves truncate() (Flush rotation) also fails
// closed: if its own f.Sync() fails, the wal is poisoned.
func TestWALTruncateSyncFailurePoisons(t *testing.T) {
	dir := t.TempDir()
	w, err := openWAL(filepath.Join(dir, "tr.wal"), false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	// Close the fd so truncate's Truncate/Seek/Sync fails.
	_ = w.f.Close()
	if err := w.truncate(); err == nil {
		t.Fatal("truncate returned nil on a closed fd; expected an error")
	}
	if !w.poisoned.Load() {
		t.Fatal("truncate did not poison the wal after its own sync/IO failure")
	}
}
