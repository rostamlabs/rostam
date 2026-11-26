// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync/atomic"
	"testing"
	"time"
)

// WHY Flush STILL SERIALIZES THE SNAPSHOT UNDER opMu.
//
// Flush (CollectionStore.Flush, FlushNamed, FlushMVWAL) holds the collection's
// opMu across {full snapshot serialization + fsync + WAL truncate}, so every
// writer stalls for time proportional to index size. The obvious fix is to move
// the serialization OUT of opMu:
//
//	(a) under opMu: capture the WAL write position
//	(b) release opMu
//	(c) serialize the snapshot under the index's READ lock — "writes proceed"
//	(d) re-take opMu and truncate the WAL up to the captured position
//
// Step (c) is the whole point of the exercise, and it does not work. The
// snapshot writers hold the index mutex in READ mode for the ENTIRE
// serialization, streaming straight to the destination file:
//
//	hnsw.Snapshot            vector/snapshot.go:427-431   h.mu.RLock  + writeSnapshot(w)
//	NamedCollection.Snapshot vector/named.go:1991-1993    nc.mu.RLock + the whole body
//	MultiVectorIndex.snapshot vector/multivector_persist.go:45-47  m.mu.RLock + inner Snapshot
//
// and EVERY mutator takes the exclusive WRITE lock on that SAME mutex
// (hnsw.insertLockedAt via h.mu.Lock at vector/hnsw.go:1525, and ~10 more sites
// per family). A Go sync.RWMutex blocks writers for as long as any reader holds
// it. So moving the serialization out of opMu does not let writes proceed — it
// only changes WHICH lock they block on, for exactly the same duration. The
// restructure trades a correct, simple invariant ("the checkpoint subsumes the
// truncated log", guaranteed by holding opMu throughout) for new crash-safe
// WAL prefix-truncation machinery, and buys zero throughput.
//
// This test pins step (c)'s premise directly, with no restructure required: it
// runs Snapshot on its own — opMu is NOT involved — and shows a concurrent
// writer cannot make progress until the serialization finishes. If a future
// change makes snapshot serialization non-blocking for writers (a copy-on-write
// / fork-style snapshot, or serializing from an immutable frozen view), THIS
// test flips to failing and the Flush restructure becomes worth revisiting.
//
// The assertion direction is deliberately conservative: an unrelated slowdown
// can only make the writer look MORE blocked, i.e. it can flake toward passing,
// never toward a false failure. The decisive claim — "the writer completed while
// serialization was in flight" — would be a real refutation.

// blockingWriter is an io.Writer that parks the FIRST Write until release is
// closed, signalling entered when it parks. It makes "the snapshot is mid
// serialization" a deterministic, wall-clock-free state instead of a sleep.
type blockingWriter struct {
	entered chan struct{}
	release chan struct{}
	once    atomic.Bool
}

func newBlockingWriter() *blockingWriter {
	return &blockingWriter{entered: make(chan struct{}), release: make(chan struct{})}
}

func (b *blockingWriter) Write(p []byte) (int, error) {
	if b.once.CompareAndSwap(false, true) {
		close(b.entered)
		<-b.release
	}
	return len(p), nil
}

// TestSnapshotSerializationBlocksWriters is the evidence behind the decision to
// leave Flush holding opMu across the snapshot: serialization holds the index
// read lock throughout, so a concurrent writer is blocked whether or not opMu is
// involved.
func TestSnapshotSerializationBlocksWriters(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	for i := 1; i <= 64; i++ {
		vec := make([]float32, 16)
		vec[i%16] = float32(i)
		if err := c.Insert(uint64(i), vec, 0, Metadata{"seed": NewInt(int64(i))}, nil); err != nil {
			t.Fatalf("seed Insert(%d): %v", i, err)
		}
	}

	bw := newBlockingWriter()
	snapDone := make(chan error, 1)
	go func() { snapDone <- c.Snapshot(bw) }()

	// Deterministic rendezvous: the snapshot is now parked mid-serialization,
	// holding h.mu.RLock. NOTE: opMu is not held by anyone here — this is exactly
	// step (c) of the proposed restructure, in isolation.
	<-bw.entered

	insertDone := make(chan error, 1)
	go func() {
		vec := make([]float32, 16)
		vec[3] = 42
		insertDone <- c.Insert(999, vec, 0, nil, nil)
	}()

	// The writer must NOT get through while serialization is in flight. This wait
	// is a bounded observation window, not a timing assertion: completing here is
	// the only outcome that would refute the claim.
	select {
	case err := <-insertDone:
		t.Fatalf("a write COMPLETED (err=%v) while the snapshot held the index read lock — "+
			"snapshot serialization no longer blocks writers, so moving Flush's snapshot "+
			"out of opMu would now actually help; revisit the Flush restructure", err)
	case <-time.After(250 * time.Millisecond):
		// Blocked, as expected: the read lock held across serialization excludes it.
	}

	// Releasing the serialization must unblock the writer — proving it was the
	// snapshot's read lock holding it, not some unrelated stall.
	close(bw.release)
	if err := <-snapDone; err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	select {
	case err := <-insertDone:
		if err != nil {
			t.Fatalf("Insert after the snapshot released: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Insert never completed after the snapshot finished — the writer was stuck on something else")
	}
}

// TestSnapshotSerializationBlocksWritersMV is the multi-vector analogue. MV is
// the worst case of the three: MultiVectorIndex.snapshot serializes the ENTIRE
// inner HNSW into a memory buffer (vector/multivector_persist.go:49) while
// holding m.mu.RLock, so the exclusion window covers the full inner-index
// serialization too.
func TestSnapshotSerializationBlocksWritersMV(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV index missing")
	}
	for i := 1; i <= 32; i++ {
		tokens := [][]float32{{float32(i), 0, 0, 0}}
		if _, err := idx.AddCASKeyTTLSparse(uint64(i), tokens, Metadata{"seed": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("seed add %d: %v", i, err)
		}
	}

	bw := newBlockingWriter()
	snapDone := make(chan error, 1)
	go func() { snapDone <- idx.snapshot(bw) }()
	<-bw.entered

	addDone := make(chan error, 1)
	go func() {
		_, err := idx.AddCASKeyTTLSparse(999, [][]float32{{7, 7, 7, 7}}, nil, nil, nil, CASCond{})
		addDone <- err
	}()

	select {
	case err := <-addDone:
		t.Fatalf("an MV add COMPLETED (err=%v) while the snapshot held m.mu.RLock — "+
			"revisit the FlushMVWAL restructure", err)
	case <-time.After(250 * time.Millisecond):
	}

	close(bw.release)
	if err := <-snapDone; err != nil {
		t.Fatalf("MV snapshot: %v", err)
	}
	select {
	case err := <-addDone:
		if err != nil {
			t.Fatalf("MV add after the snapshot released: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("MV add never completed after the snapshot finished")
	}
}

// TestReplayInsertOverLiveIDIsANoOp pins that the index primitive RestoreInsert
// still COLLIDES with a live id (ErrDuplicateID, state unchanged). Dense WAL
// replay no longer uses it: it applies an insert record with restoreReplace,
// which replaces a live point (see Collection.replayWAL and
// TestReplaceReplayOverCheckpoint). What this guards is everyone ELSE that calls
// RestoreInsert — the reshard/resplit copy and Collection.RestoreInsert, which
// rely on the collision never clobbering a live value — and, for older binaries,
// what their replay does with a replace record: it collides, the error is
// discarded, and the old version stays.
//
// It does not make the Flush restructure safe: replay is only known correct over
// a snapshot taken at the start or the end of the log (see the note on
// CollectionStore.Flush), never one cut mid-log by writes landing during
// serialization.
func TestReplayInsertOverLiveIDIsANoOp(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	original := make([]float32, 16)
	original[0] = 1
	if err := c.RestoreInsert(1, original, 0, Metadata{"gen": NewInt(1)}, nil, nil, 5); err != nil {
		t.Fatalf("seed RestoreInsert: %v", err)
	}

	// A "newer" record for the SAME live id — what replay would see if a write
	// landed during snapshot serialization and its record survived truncation.
	newer := make([]float32, 16)
	newer[0] = 2
	err := c.idx.RestoreInsert(1, newer, 0, Metadata{"gen": NewInt(2)}, nil, nil, 6)
	if err == nil {
		t.Fatal("RestoreInsert over a LIVE id unexpectedly succeeded — the replay hazard " +
			"documented here has changed; re-derive the Flush restructure's safety argument")
	}

	// And the state is UNCHANGED — replay discards this error, so the newer record
	// would have been silently dropped.
	_, meta, _, _, version, ok := c.Get(1)
	if !ok {
		t.Fatal("point 1 vanished")
	}
	if got := meta["gen"]; !got.Equal(NewInt(1)) {
		t.Fatalf("payload = %v, want the ORIGINAL gen=1 (the collision must not mutate state)", meta)
	}
	if version != 5 {
		t.Fatalf("version = %d, want the ORIGINAL 5", version)
	}
}
