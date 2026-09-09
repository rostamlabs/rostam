// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft"
)

// blockingOp registers an op whose handler returns ops.ErrWASMModuleNotResident
// until release() is called, then applies normally by incrementing applied.
//
// It stands in for a WASM invocation whose module has not been fetched yet
// WITHOUT dragging the wasm runtime into this package: what shard's apply
// contract owes the rest of the system is stated entirely in terms of the
// sentinel and its class, so the sentinel is the right seam to test at. The
// cluster-level gates (a group blocked on a real unresolvable fingerprint while
// its neighbours keep applying and snapshotting) live in cluster/.
type blockingOp struct {
	blocked atomic.Bool
	applied atomic.Int64
}

func (b *blockingOp) register(t *testing.T, reg *ops.Registry, name string) {
	t.Helper()
	b.blocked.Store(true)
	err := reg.Register(name, ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		if b.blocked.Load() {
			return nil, &ops.WASMNotResidentError{Op: name, Group: 7}
		}
		b.applied.Add(1)
		return nil, nil
	})
	if err != nil {
		t.Fatalf("register %s: %v", name, err)
	}
}

func (b *blockingOp) release() { b.blocked.Store(false) }

// newRetryFSM builds an fsm wired for a fast, deterministic block: a 1ms base
// backoff and a stop channel the test can close.
func newRetryFSM(t *testing.T) (*fsm, chan struct{}) {
	t.Helper()
	f, _ := newTestFSM(t)
	// REPLICATED, because that is the only configuration in which the three
	// classes actually differ: fsm.replicated() gates the classFatal halt, so a
	// bare single-node fsm would advance past a fatal error and a falsification
	// that mis-classifies classRetry as classFatal would look like a mere advance.
	f.isReplicated = func() bool { return true }
	f.applyRetryWait = time.Millisecond
	stop := make(chan struct{})
	f.stop = stop
	t.Cleanup(func() {
		select {
		case <-stop:
		default:
			close(stop)
		}
	})
	return f, stop
}

// TestClassifyRetryIsNeitherAdvanceNorFatal is the classification gate. It is
// separate from the behavioural tests below because the two failure modes it
// rules out are opposite and both catastrophic:
//
//   - classified classAdvance, the entry is SKIPPED while every peer that holds
//     the module executes it — silent permanent divergence;
//   - classified classFatal, defaultFatalApply os.Exit(1)s the process, the
//     committed entry replays into the same missing blob on restart, and the
//     node crash-loops. A halt is meant to be bounded and recoverable; this one
//     would be neither.
func TestClassifyRetryIsNeitherAdvanceNorFatal(t *testing.T) {
	err := &ops.WASMNotResidentError{Op: "udf", Group: 3}
	if got := classifyApplyErr(err); got != classRetry {
		t.Fatalf("classifyApplyErr(not-resident) = %v, want classRetry", got)
	}
	// Through the multi-%w wraps the apply path adds on the way out.
	wrapped := fmt.Errorf("shard: fatal non-deterministic apply at index %d: %w", 5, err)
	if got := classifyApplyErr(wrapped); got != classRetry {
		t.Fatalf("classifyApplyErr(wrapped not-resident) = %v, want classRetry", got)
	}
	if !errors.Is(err, ops.ErrWASMModuleNotResident) {
		t.Fatal("WASMNotResidentError must satisfy errors.Is against its sentinel")
	}
	// The neighbours it must not be confused with keep their classes.
	if got := classifyApplyErr(ops.ErrWASMNoGroupBinding); got != classFatal {
		t.Fatalf("ErrWASMNoGroupBinding = %v, want classFatal (a real divergence signal)", got)
	}
	if got := classifyApplyErr(ErrOpNotRegistered); got != classFatal {
		t.Fatalf("ErrOpNotRegistered = %v, want classFatal", got)
	}
	if got := classifyApplyErr(errors.New("some handler business error")); got != classAdvance {
		t.Fatalf("unknown error = %v, want classAdvance (the inclusion list's default)", got)
	}
}

// TestApplyBlockedMutatesNothingAndAdvancesNothing is the core invariant: an
// apply that cannot resolve its module mutates nothing, advances nothing, halts
// nothing, and re-runs the same entry until it can.
//
// FALSIFICATION (performed, message recorded in the commit): making
// classifyApplyErr return classAdvance for the sentinel makes this fail at the
// "applied index advanced while blocked" check, because the entry is skipped and
// the index moves. Making it classFatal fires onFatalApply, which the test
// records and rejects.
func TestApplyBlockedMutatesNothingAndAdvancesNothing(t *testing.T) {
	f, _ := newRetryFSM(t)
	var blocked blockingOp
	blocked.register(t, f.registry, "blockme")

	var halted atomic.Bool
	f.onFatalApply = func(error) { halted.Store(true) }

	var retries atomic.Int64
	f.onApplyRetry = func(error, int, time.Duration) { retries.Add(1) }
	cleared := make(chan struct{})
	var clearOnce sync.Once
	f.onApplyRetryCleared = func(error, int, time.Duration) { clearOnce.Do(func() { close(cleared) }) }

	done := make(chan *ApplyResponse, 1)
	go func() {
		done <- f.Apply(&hraft.Log{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("blockme", nil)}).(*ApplyResponse)
	}()

	// Give the block time to be entered and retried several times. Short and
	// explicit: a test that waits on a block must never wait without a bound.
	deadline := time.Now().Add(3 * time.Second)
	for retries.Load() < 3 {
		// Naming the WRONG outcome precisely is the point of this branch: if the
		// apply has already returned, the entry was not retried at all, and which
		// of the two catastrophes happened is exactly what the caller needs told.
		select {
		case <-done:
			switch {
			case halted.Load():
				t.Fatal("the apply HALTED instead of retrying: a classRetry condition is group-local and self-healing, and the committed entry replays into it on restart — the node would crash-loop")
			case f.AppliedIndex() != 0:
				t.Fatalf("the apply ADVANCED the index to %d instead of retrying: peers that hold the module execute this entry, so skipping it is silent permanent divergence", f.AppliedIndex())
			default:
				t.Fatal("the apply returned without retrying and without advancing")
			}
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("the FSM did not retry a classRetry apply within 3s")
		}
		time.Sleep(2 * time.Millisecond)
	}

	if got := f.AppliedIndex(); got != 0 {
		t.Fatalf("applied index advanced to %d while blocked; a blocked apply must record nothing", got)
	}
	if blocked.applied.Load() != 0 {
		t.Fatal("the handler ran its mutating path while blocked")
	}
	if halted.Load() {
		t.Fatal("a classRetry apply must NOT halt: the condition is group-local and self-healing, while a halt is process-global and replays into itself on restart")
	}

	// The blob arrives (the __wasm_blob_put__ escape hatch, or a fetch landing).
	blocked.release()
	select {
	case resp := <-done:
		if resp.Err != nil {
			t.Fatalf("apply after release: %v", resp.Err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the blocked apply did not complete within 5s of the module becoming available")
	}
	select {
	case <-cleared:
	case <-time.After(time.Second):
		t.Fatal("OnApplyRetryCleared did not fire; a block that is never reported as cleared leaves a permanent phantom in Stats")
	}
	if got := blocked.applied.Load(); got != 1 {
		t.Fatalf("handler applied %d times, want exactly 1: a retry re-runs an entry that never ran, it does not re-apply one", got)
	}
	if got := f.AppliedIndex(); got != 5 {
		t.Fatalf("applied index = %d after the block cleared, want 5", got)
	}
}

// TestApplyBatchRecordsThePrefixBeforeWaiting is the non-obvious requirement and
// the one with real corruption behind it.
//
// Batch [incr@5, INV@6-blocked]. While blocked, AppliedIndex() must already read
// 5. Without that, an operator restarting a node that has "stopped making
// progress" — which is the single most likely response to a block — replays the
// WHOLE batch, and incr@5 applies a SECOND time. It is the hazard fsm.go already
// documents for the panic path, except that this path is a normal, deliberately
// long-lived state rather than an unexpected bug.
//
// FALSIFICATION (performed, message recorded in the commit): deleting the
// beforeFirstWait callback body makes this fail at "applied index while blocked
// = 0, want 5".
func TestApplyBatchRecordsThePrefixBeforeWaiting(t *testing.T) {
	f, _ := newRetryFSM(t)
	var blocked blockingOp
	blocked.register(t, f.registry, "blockme")

	// A NON-IDEMPOTENT prefix op — the whole point. incr is the codebase's own
	// example of an op a double-apply corrupts.
	var incrs atomic.Int64
	err := f.registry.Register("countme", ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		incrs.Add(1)
		return nil, nil
	})
	if err != nil {
		t.Fatalf("register countme: %v", err)
	}

	f.onFatalApply = func(e error) { t.Errorf("unexpected halt: %v", e) }
	var retries atomic.Int64
	f.onApplyRetry = func(error, int, time.Duration) { retries.Add(1) }

	logs := []*hraft.Log{
		{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("countme", nil)},
		{Index: 6, Type: hraft.LogCommand, Data: EncodeLogEntry("blockme", nil)},
	}
	done := make(chan []any, 1)
	go func() { done <- f.ApplyBatch(logs) }()

	deadline := time.Now().Add(3 * time.Second)
	for retries.Load() < 3 {
		if time.Now().After(deadline) {
			t.Fatal("the FSM did not retry the blocked batch entry within 3s")
		}
		time.Sleep(2 * time.Millisecond)
	}

	if got := f.AppliedIndex(); got != 5 {
		t.Fatalf("applied index while blocked = %d, want 5: the successfully-applied PREFIX must be recorded DURABLY BEFORE the wait, or a restart mid-block replays incr@5 and double-counts it", got)
	}
	if got := incrs.Load(); got != 1 {
		t.Fatalf("prefix op ran %d times while blocked, want 1", got)
	}

	blocked.release()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the blocked batch did not complete within 5s of release")
	}
	if got := incrs.Load(); got != 1 {
		t.Fatalf("prefix op ran %d times in total, want EXACTLY 1: the retry must re-run only the blocked entry, never the prefix", got)
	}
	if got := f.AppliedIndex(); got != 6 {
		t.Fatalf("applied index = %d after release, want 6", got)
	}
}

// TestApplyBlockAbandonedOnStopDoesNotAdvance covers the one exit from an
// otherwise-unbounded wait.
//
// It is not a nicety. hashicorp/raft runs Apply, ApplyBatch, Snapshot and
// Restore on ONE goroutine, so a group parked in a retry is a group that cannot
// observe raft's shutdown channel — without this exit, Store.Close would wait for
// a wait that has no end. Abandoning is safe for exactly the reason the halt path
// is safe: the applied index is NOT advanced, so the committed entry replays on
// the next start.
func TestApplyBlockAbandonedOnStopDoesNotAdvance(t *testing.T) {
	f, stop := newRetryFSM(t)
	var blocked blockingOp
	blocked.register(t, f.registry, "blockme")
	f.onFatalApply = func(e error) { t.Errorf("unexpected halt: %v", e) }
	var retries atomic.Int64
	f.onApplyRetry = func(error, int, time.Duration) { retries.Add(1) }

	done := make(chan *ApplyResponse, 1)
	go func() {
		done <- f.Apply(&hraft.Log{Index: 9, Type: hraft.LogCommand, Data: EncodeLogEntry("blockme", nil)}).(*ApplyResponse)
	}()
	deadline := time.Now().Add(3 * time.Second)
	for retries.Load() < 2 {
		if time.Now().After(deadline) {
			t.Fatal("the FSM did not retry within 3s")
		}
		time.Sleep(2 * time.Millisecond)
	}
	close(stop)
	select {
	case resp := <-done:
		if !errors.Is(resp.Err, ops.ErrWASMModuleNotResident) {
			t.Fatalf("abandoned apply returned %v, want the not-resident sentinel", resp.Err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("closing the Store's stop channel did not release the block within 3s; Close would hang")
	}
	if got := f.AppliedIndex(); got != 0 {
		t.Fatalf("applied index = %d after abandoning a still-blocked entry, want 0: nothing ran, so nothing may be recorded", got)
	}
}

// newDurableRetryFSM is newRetryFSM over a DURABLE (mmap) cache.
//
// It exists because the heap-mode helper cannot see half of what the apply
// contract promises: cache.AppliedIndex() is 0 by construction in heap mode, so
// every assertion about the PERSISTED watermark — which is the thing a restart
// actually reads — reduces to a tautology there.
func newDurableRetryFSM(t *testing.T) (*fsm, chan struct{}) {
	t.Helper()
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.DataDir = t.TempDir() // durable header live
	c, err := cache.New(cc)
	if err != nil {
		t.Skipf("durable cache unavailable on this platform: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	f := newFSM(c, reg, true /* durable */, nil, nil)
	f.isReplicated = func() bool { return true }
	f.applyRetryWait = time.Millisecond
	stop := make(chan struct{})
	f.stop = stop
	t.Cleanup(func() {
		select {
		case <-stop:
		default:
			close(stop)
		}
	})
	return f, stop
}

// TestApplyBatchRecordsThePrefixDurablyBeforeWaiting is the DURABLE half of
// TestApplyBatchRecordsThePrefixBeforeWaiting, and it is not a duplicate of it.
//
// The heap-mode test asserts f.AppliedIndex(), which the barrier reads and which
// beforeFirstWait's advanceApplied sets. But the hazard the callback's own message
// names — an operator restarting a blocked node, incr@5 applying a SECOND time —
// is defeated by the OTHER line, cache.SetAppliedIndex, which writes the mmap
// header a warm restart consults. In heap mode cache.AppliedIndex() is 0 by
// construction, so deleting that line leaves the heap test green while the
// double-apply goes live. This is the assertion that fails.
//
// FALSIFICATION (performed, message recorded in the commit): deleting the
// `f.cache.SetAppliedIndex(appliedPrefix, f.durable)` line from beforeFirstWait
// makes this fail at "durable applied header while blocked = 0, want 5".
func TestApplyBatchRecordsThePrefixDurablyBeforeWaiting(t *testing.T) {
	f, _ := newDurableRetryFSM(t)
	var blocked blockingOp
	blocked.register(t, f.registry, "blockme")

	var incrs atomic.Int64
	err := f.registry.Register("countme", ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		incrs.Add(1)
		return nil, nil
	})
	if err != nil {
		t.Fatalf("register countme: %v", err)
	}
	f.onFatalApply = func(e error) { t.Errorf("unexpected halt: %v", e) }
	var retries atomic.Int64
	f.onApplyRetry = func(error, int, time.Duration) { retries.Add(1) }

	if got := f.cache.AppliedIndex(); got != 0 {
		t.Fatalf("precondition: durable applied header = %d, want 0", got)
	}
	logs := []*hraft.Log{
		{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("countme", nil)},
		{Index: 6, Type: hraft.LogCommand, Data: EncodeLogEntry("blockme", nil)},
	}
	done := make(chan []any, 1)
	go func() { done <- f.ApplyBatch(logs) }()

	deadline := time.Now().Add(3 * time.Second)
	for retries.Load() < 3 {
		if time.Now().After(deadline) {
			t.Fatal("the FSM did not retry the blocked batch entry within 3s")
		}
		time.Sleep(2 * time.Millisecond)
	}

	if got := f.cache.AppliedIndex(); got != 5 {
		t.Fatalf("durable applied header while blocked = %d, want 5: the PERSISTED watermark is what a warm restart reads, "+
			"and an operator restarting a blocked node replays the whole batch without it — incr@5 applies a second time", got)
	}
	// The blocked entry must be EXCLUDED from that watermark, or the restart the
	// header exists to guide would skip an entry that never ran.
	if got := f.cache.AppliedIndex(); got >= 6 {
		t.Fatalf("durable applied header = %d, want strictly below the blocked index 6", got)
	}

	blocked.release()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the blocked batch did not complete within 5s of release")
	}
	if got := incrs.Load(); got != 1 {
		t.Fatalf("prefix op ran %d times in total, want EXACTLY 1", got)
	}
	if got := f.cache.AppliedIndex(); got != 6 {
		t.Fatalf("durable applied header = %d after release, want 6", got)
	}
}

// TestAbandonedApplyPoisonsTheSnapshotterAndEveryLaterApply is about RAFT'S
// bookkeeping, not ours, and that is the only reason it is hard to see.
//
// Our accounting after an abandon is correct: the prefix is recorded, the blocked
// entry is not. hashicorp/raft's is not. runFSM (v1.7.3) computes lastBatchIndex
// over EVERY log in the batch BEFORE calling ApplyBatch and assigns
// lastIndex = lastBatchIndex unconditionally once it returns — it has no way to be
// told a prefix applied, because until the classRetry abandon no path could return
// from ApplyBatch in a live process without having applied the entry. Two things
// then act on raft's wrong belief:
//
//   - snapshot(): stamps the request with lastIndex, which snapshot.go hands to
//     snapshots.Create. The snapshot would hold state as of the PREFIX under a
//     label covering the tail, and a restart sets lastApplied from that label and
//     never redelivers the missing entries;
//   - runFSM's own loop: Store.Close closes stop BEFORE raft.Shutdown closes
//     shutdownCh, so when the abandon unwinds the only ready case is the 128-deep
//     fsmMutateCh full of batches committed while the group was parked. Applying
//     those writes a durable watermark ABOVE the hole.
//
// FALSIFICATION (performed, messages recorded in the commit): removing the
// Snapshot guard fails at "Snapshot succeeded after an abandoned apply"; removing
// the ApplyBatch guard fails at "durable applied header = 8 after a post-abandon
// batch".
func TestAbandonedApplyPoisonsTheSnapshotterAndEveryLaterApply(t *testing.T) {
	f, stop := newDurableRetryFSM(t)
	var blocked blockingOp
	blocked.register(t, f.registry, "blockme")
	f.onFatalApply = func(e error) { t.Errorf("unexpected halt: %v", e) }
	var retries atomic.Int64
	f.onApplyRetry = func(error, int, time.Duration) { retries.Add(1) }

	// The snapshotter WORKS before an abandon — without this the assertion below
	// would pass on an fsm that could never snapshot at all.
	if _, err := f.Snapshot(); err != nil {
		t.Fatalf("precondition: Snapshot before any abandon: %v", err)
	}

	// [put@5, blocked@6, put@7]. Raft will record 7 as applied whatever we do here.
	logs := []*hraft.Log{
		{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("a"), []byte("1"), 0))},
		{Index: 6, Type: hraft.LogCommand, Data: EncodeLogEntry("blockme", nil)},
		{Index: 7, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("b"), []byte("2"), 0))},
	}
	done := make(chan []any, 1)
	go func() { done <- f.ApplyBatch(logs) }()

	deadline := time.Now().Add(3 * time.Second)
	for retries.Load() < 2 {
		if time.Now().After(deadline) {
			t.Fatal("the FSM did not retry the blocked batch entry within 3s")
		}
		time.Sleep(2 * time.Millisecond)
	}
	close(stop)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("closing stop did not release the block within 3s")
	}

	if got := f.cache.AppliedIndex(); got != 5 {
		t.Fatalf("durable applied header after the abandon = %d, want 5 (the applied prefix)", got)
	}

	if _, err := f.Snapshot(); err == nil {
		t.Fatal("Snapshot succeeded after an abandoned apply: raft stamps the snapshot request with the LAST index of the " +
			"batch it handed us, so this snapshot would be written to disk labelled with entries 6 and 7 that never ran — " +
			"and on restart raft sets lastApplied from that label and NEVER redelivers them")
	} else if !strings.Contains(err.Error(), "refusing to snapshot") {
		t.Fatalf("the refusal must say what it is refusing and why, since the reason is entirely non-local; got: %v", err)
	}

	// Nothing may apply afterwards either — see the runFSM-loop arm above.
	next := []*hraft.Log{
		{Index: 8, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("c"), []byte("3"), 0))},
	}
	resps := f.ApplyBatch(next)
	if len(resps) != len(next) {
		t.Fatalf("ApplyBatch returned %d responses for %d logs; raft panics on a mismatch", len(resps), len(next))
	}
	if got := f.cache.AppliedIndex(); got != 5 {
		t.Fatalf("durable applied header = %d after a post-abandon batch, want 5: entries 6 and 7 never ran, so a watermark "+
			"above them makes warm restart skip them permanently", got)
	}
	if _, err := f.cache.Get([]byte("c")); err == nil {
		t.Fatal("a post-abandon batch mutated state: this FSM's state must stop where its applied prefix stops")
	}

	// And the single-entry path, which abandons the same way.
	resp := f.Apply(&hraft.Log{Index: 9, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("d"), []byte("4"), 0))})
	if _, ok := resp.(*ApplyResponse); !ok {
		t.Fatalf("Apply returned %T, want *ApplyResponse", resp)
	}
	if got := f.cache.AppliedIndex(); got != 5 {
		t.Fatalf("durable applied header = %d after a post-abandon single Apply, want 5", got)
	}
	if _, err := f.cache.Get([]byte("d")); err == nil {
		t.Fatal("a post-abandon single Apply mutated state")
	}
}

// TestRaftsSnapshotterSurfacesTheAbandonRefusal closes the seam the unit test
// above cannot: it asserts through hashicorp/raft's OWN snapshotter that our
// error is propagated rather than swallowed, and that no snapshot file is written.
//
// The unit test proves fsm.Snapshot refuses. That is worth nothing if raft logs
// the error and creates the snapshot anyway — and the whole defect is about raft's
// behaviour, so asserting only our half would be asserting the half that was never
// in doubt. takeSnapshot's error return happens BEFORE snapshots.Create, and the
// directory count is what pins that.
func TestRaftsSnapshotterSurfacesTheAbandonRefusal(t *testing.T) {
	s := newSingleNodeStore(t)
	rn, ok := s.raft.(*raft.Node)
	if !ok {
		t.Skipf("this store's replicator is %T, not the Raft engine", s.raft)
	}
	if err := s.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	snapDir := filepath.Join(s.cfg.DataDir, "snapshots")

	// Baseline: a forced snapshot succeeds and writes a file. Without this the
	// assertion below could pass on a store that never snapshots.
	if err := rn.Snapshot(); err != nil {
		t.Fatalf("baseline forced snapshot: %v", err)
	}
	before := countRaftSnapshots(t, snapDir)
	if before == 0 {
		t.Fatalf("the baseline snapshot wrote nothing under %s, so this test proves nothing", snapDir)
	}

	s.fsm.abandoned.Store(true)

	err := rn.Snapshot()
	if err == nil {
		t.Fatal("raft's snapshotter completed after an abandoned apply: it stamps the snapshot with ITS last index, " +
			"which covers entries this FSM never applied, and a restart would set lastApplied past them")
	}
	if !strings.Contains(err.Error(), "refusing to snapshot") {
		t.Fatalf("raft did not surface the FSM's refusal; got: %v", err)
	}
	if after := countRaftSnapshots(t, snapDir); after != before {
		t.Fatalf("snapshot count went %d -> %d: the refusal must happen BEFORE snapshots.Create, or the mislabelled "+
			"snapshot is on disk regardless of what was returned", before, after)
	}
}

// countRaftSnapshots counts completed snapshots in a FileSnapshotStore directory.
// In-progress ones carry a .tmp suffix and are not snapshots yet.
func countRaftSnapshots(t *testing.T, dir string) int {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0
		}
		t.Fatalf("read snapshot dir %s: %v", dir, err)
	}
	n := 0
	for _, e := range ents {
		if e.IsDir() && !strings.HasSuffix(e.Name(), ".tmp") {
			n++
		}
	}
	return n
}

// TestPBApplierRetriesRatherThanNACKing pins the PB surface, which is distinct
// from the Raft one.
//
// A NACK (errPBInfra) wedges the shard ANYWAY under full-ISR commit, but does
// so from inside the engine's receiveLocked — i.e. while holding the engine lock
// — converting a self-resolving condition into one that blocks every other engine
// operation on the way. Retrying wedges no more than the NACK would and releases
// the moment the blob lands.
func TestPBApplierRetriesRatherThanNACKing(t *testing.T) {
	f, _ := newRetryFSM(t)
	var blocked blockingOp
	blocked.register(t, f.registry, "blockme")
	var retries atomic.Int64
	f.onApplyRetry = func(error, int, time.Duration) { retries.Add(1) }

	a := newShardApplier(f)
	done := make(chan error, 1)
	go func() {
		_, err := a.Apply(EncodeLogEntry("blockme", nil))
		done <- err
	}()
	deadline := time.Now().Add(3 * time.Second)
	for retries.Load() < 3 {
		select {
		case err := <-done:
			t.Fatalf("PB apply returned %v instead of retrying; a classRetry must not become errPBInfra", err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("the PB applier did not retry within 3s")
		}
		time.Sleep(2 * time.Millisecond)
	}
	blocked.release()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("PB apply after release: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the PB applier did not complete within 5s of release")
	}
	if got := blocked.applied.Load(); got != 1 {
		t.Fatalf("handler applied %d times, want exactly 1", got)
	}
}
