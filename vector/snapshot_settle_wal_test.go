// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"sync"
	"testing"
	"time"
)

// A snapshot of a log-backed collection (a checkpoint, a backup, a cold-tier
// eviction) must not capture a write whose record is not on disk. The window is
// a group-commit fsync, which runs after the writer has released opMu: the write
// is applied, its bytes are in the log, and its sync then fails, so the writer
// is told it failed. A snapshot taken in that window used to hold the refused
// write; restoring it would replace the acknowledged state.

// blockSyncThenFail parks w's next fsync until release is closed, then makes it
// fail. inSync is closed once that fsync has started.
func blockSyncThenFail(w *wal) (inSync, release chan struct{}) {
	inSync, release = make(chan struct{}), make(chan struct{})
	var once sync.Once
	w.beforeSync = func() {
		once.Do(func() {
			close(inSync)
			<-release
			_ = w.f.Close()
		})
	}
	return inSync, release
}

// snapshotWaitsForFailedSync starts write, waits until its fsync is in flight,
// then takes a snapshot. The snapshot must not return while that sync is
// pending, and must refuse with ErrWALPoisoned once it fails.
func snapshotWaitsForFailedSync(t *testing.T, w *wal, write func() error, snapshot func(*bytes.Buffer) error) {
	t.Helper()
	inSync, release := blockSyncThenFail(w)
	wrote := make(chan error, 1)
	go func() { wrote <- write() }()
	<-inSync

	var buf bytes.Buffer
	snapped := make(chan error, 1)
	go func() { snapped <- snapshot(&buf) }()
	select {
	case err := <-snapped:
		close(release)
		<-wrote
		t.Fatalf("snapshot returned (err=%v, %d bytes) while a write's sync was still in flight", err, buf.Len())
	case <-time.After(100 * time.Millisecond):
	}
	close(release)
	if err := <-wrote; err == nil {
		t.Fatal("a write whose fsync failed was acknowledged")
	}
	if err := <-snapped; !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("snapshot after the pending sync failed = %v, want ErrWALPoisoned", err)
	}
	if buf.Len() != 0 {
		t.Fatalf("a refused snapshot wrote %d bytes", buf.Len())
	}
}

func TestSnapshotWaitsForPendingWALSyncDense(t *testing.T) {
	cs := openReplaceStore(t, t.TempDir(), durablePersistentCfg())
	t.Cleanup(func() { _ = cs.Close() })
	c := acquire(t, cs)
	defer c.Release()
	snapshotWaitsForFailedSync(t, c.wal,
		func() error { return c.Upsert(1, durableVec(9), "refused", 0, nil, nil) },
		func(b *bytes.Buffer) error { return c.Snapshot(b) })
}

func TestSnapshotWaitsForPendingWALSyncNamed(t *testing.T) {
	nc := seedB2Named(t)
	snapshotWaitsForFailedSync(t, nc.wal,
		func() error {
			_, err := nc.InsertCAS(1, map[string][]float32{"title": {0, 1, 0, 0}}, Metadata{"k": NewInt(7)}, 0, CASCond{})
			return err
		},
		func(b *bytes.Buffer) error { return nc.Snapshot(b) })
}

func TestSnapshotWaitsForPendingWALSyncMultiVector(t *testing.T) {
	m := seedB2MV(t)
	snapshotWaitsForFailedSync(t, m.wal,
		func() error {
			_, err := m.AddCAS(1, [][]float32{{0, 1, 0, 0}}, Metadata{"k": NewInt(7)}, CASCond{})
			return err
		},
		func(b *bytes.Buffer) error { return m.snapshot(b) })
}

// TestSnapshotRefusesPoisonedWALMultiVector: the multi-vector snapshot refuses a
// poisoned log and writes nothing.
func TestSnapshotRefusesPoisonedWALMultiVector(t *testing.T) {
	m := seedB2MV(t)
	m.wal.poisoned.Store(true)
	var buf bytes.Buffer
	if err := m.snapshot(&buf); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("snapshot of a poisoned multi-vector log = %v, want ErrWALPoisoned", err)
	}
	if buf.Len() != 0 {
		t.Fatalf("a refused snapshot wrote %d bytes", buf.Len())
	}
}

// TestSnapshotOfHealthyWALSucceeds: settling a healthy log syncs and lets the
// snapshot through.
func TestSnapshotOfHealthyWALSucceeds(t *testing.T) {
	cs := openReplaceStore(t, t.TempDir(), durablePersistentCfg())
	t.Cleanup(func() { _ = cs.Close() })
	c := acquire(t, cs)
	defer c.Release()
	var buf bytes.Buffer
	if err := c.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	if buf.Len() == 0 {
		t.Fatal("empty snapshot")
	}
}
