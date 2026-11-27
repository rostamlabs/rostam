// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// TestDurableBulkBuildSurvivesACrash: a bulk build writes no log record, so on a
// listed collection it is made durable by a checkpoint before BuildStaged
// returns. A copy of the store taken right after the call, before any Close,
// stands for a crash; reopening it must hold every built point.
func TestDurableBulkBuildSurvivesACrash(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "bulk")
	t.Cleanup(func() { _ = cs.Close() })
	if err := cs.CreateCollection("bulk", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	ids := []uint64{1, 2, 3}
	vecs := [][]float32{durableVec(1), durableVec(2), durableVec(3)}
	if err := cs.StageBulkPayloads("bulk", ids, vecs, []Metadata{{"k": NewInt(1)}, nil, {"k": NewInt(3)}}); err != nil {
		t.Fatal(err)
	}
	if err := cs.BuildStaged("bulk", 1); err != nil {
		t.Fatal(err)
	}

	crashed := filepath.Join(t.TempDir(), "copy")
	copyTree(t, dir, crashed)
	re := openDurable(t, crashed, "bulk")
	defer func() { _ = re.Close() }()
	c, ok := re.Acquire("bulk")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	defer c.Release()
	for _, id := range ids {
		if !c.Exists(id) {
			t.Fatalf("bulk-built point %d lost by a crash after BuildStaged returned", id)
		}
	}
	if _, m, _, _, _, _ := c.Get(3); !reflect.DeepEqual(m["k"], NewInt(3)) {
		t.Fatalf("payload of point 3 after reopen = %v, want k=3", m)
	}
}

func stageThree(t *testing.T, cs *CollectionStore, col string) {
	t.Helper()
	ids := []uint64{1, 2, 3}
	vecs := [][]float32{durableVec(1), durableVec(2), durableVec(3)}
	if err := cs.StageBulk(col, ids, vecs); err != nil {
		t.Fatal(err)
	}
}

func setBuildStagedHook(t *testing.T, fn func()) {
	t.Helper()
	buildStagedHook = fn
	t.Cleanup(func() { buildStagedHook = nil })
}

func openBulkStore(t *testing.T, dir string) *CollectionStore {
	t.Helper()
	cs := openDurable(t, dir, "bulk")
	if err := cs.CreateCollection("bulk", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	return cs
}

func payloadK(t *testing.T, cs *CollectionStore, id uint64) Metadata {
	t.Helper()
	c, ok := cs.Acquire("bulk")
	if !ok {
		t.Fatal("collection missing")
	}
	defer c.Release()
	_, m, _, _, _, _ := c.Get(id)
	return m
}

// TestDurableBulkBuildConcurrentWriteSurvivesACrash: a write that arrives while
// a listed collection's bulk build is running must not be acknowledged on top
// of a build no log or snapshot holds yet. It waits for the checkpoint, and is
// then logged after it, so a crash at any point after its acknowledgement
// keeps it.
func TestDurableBulkBuildConcurrentWriteSurvivesACrash(t *testing.T) {
	dir := t.TempDir()
	cs := openBulkStore(t, dir)
	t.Cleanup(func() { _ = cs.Close() })
	stageThree(t, cs, "bulk")

	wrote := make(chan error, 1)
	var earlyCopy string
	setBuildStagedHook(t, func() {
		go func() {
			_, err := cs.SetPayload("bulk", 1, Metadata{"k": NewInt(7)}, nil)
			wrote <- err
		}()
		select {
		case err := <-wrote:
			// Acknowledged before the checkpoint: a crash now must keep it.
			if err == nil {
				earlyCopy = filepath.Join(t.TempDir(), "early")
				copyTree(t, dir, earlyCopy)
			}
			wrote <- err
		case <-time.After(200 * time.Millisecond):
		}
	})
	if err := cs.BuildStaged("bulk", 1); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-wrote:
		if err != nil {
			t.Fatalf("write after the build: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("the write queued during the build never returned")
	}
	if earlyCopy != "" {
		re := openDurable(t, earlyCopy, "bulk")
		got := payloadK(t, re, 1)
		_ = re.Close()
		if !reflect.DeepEqual(got["k"], NewInt(7)) {
			t.Fatalf("a write acknowledged during the build was lost by a crash: payload %v", got)
		}
	}
	crashed := filepath.Join(t.TempDir(), "copy")
	copyTree(t, dir, crashed)
	re := openDurable(t, crashed, "bulk")
	defer func() { _ = re.Close() }()
	if got := payloadK(t, re, 1); !reflect.DeepEqual(got["k"], NewInt(7)) {
		t.Fatalf("acknowledged write lost by a crash: payload %v", got)
	}
}

// TestDurableBulkBuildFailedCheckpointPoisons: when the checkpoint after a
// build fails, the build is in memory and in no file. The log is poisoned, so
// nothing is acknowledged on top of it, and a reopen has the pre-build state.
func TestDurableBulkBuildFailedCheckpointPoisons(t *testing.T) {
	dir := t.TempDir()
	cs := openBulkStore(t, dir)
	t.Cleanup(func() { _ = cs.Close() })
	stageThree(t, cs, "bulk")
	// A directory where the snapshot's temp file goes makes the write fail.
	if err := os.Mkdir(filepath.Join(dir, "vectors", DefaultTenant, "bulk.snap.tmp"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := cs.BuildStaged("bulk", 1); err == nil {
		t.Fatal("BuildStaged succeeded although its checkpoint could not be written")
	}
	if _, err := cs.SetPayload("bulk", 1, Metadata{"k": NewInt(7)}, nil); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("a write after a failed build checkpoint = %v, want ErrWALPoisoned", err)
	}
	if err := cs.Upsert("bulk", 9, durableVec(9), "x", 0, nil, nil); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("an insert after a failed build checkpoint = %v, want ErrWALPoisoned", err)
	}
	crashed := filepath.Join(t.TempDir(), "copy")
	copyTree(t, dir, crashed)
	if err := os.Remove(filepath.Join(crashed, "vectors", DefaultTenant, "bulk.snap.tmp")); err != nil {
		t.Fatal(err)
	}
	re := openDurable(t, crashed, "bulk")
	defer func() { _ = re.Close() }()
	if n := len(pointsOf(t, re, "bulk")); n != 0 {
		t.Fatalf("reopen holds %d points, want the pre-build state (0)", n)
	}
}

// TestDurableBulkBuildNothingStagedSkipsCheckpoint: with nothing staged the
// build is a no-op, and so is the checkpoint: one that could not be written
// is not even tried, and the log stays healthy.
func TestDurableBulkBuildNothingStagedSkipsCheckpoint(t *testing.T) {
	dir := t.TempDir()
	cs := openBulkStore(t, dir)
	t.Cleanup(func() { _ = cs.Close() })
	if err := os.Mkdir(filepath.Join(dir, "vectors", DefaultTenant, "bulk.snap.tmp"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := cs.BuildStaged("bulk", 1); err != nil {
		t.Fatalf("BuildStaged with nothing staged = %v, want nil", err)
	}
	if err := cs.Upsert("bulk", 9, durableVec(9), "x", 0, nil, nil); err != nil {
		t.Fatalf("a write after an empty build = %v, want nil", err)
	}
}

// TestDurableBulkBuildNoDeadlock: Flush, drop, restore over the collection and
// eviction, started while a listed collection's build holds it, all finish.
func TestDurableBulkBuildNoDeadlock(t *testing.T) {
	ops := []struct {
		name string
		op   func(cs *CollectionStore) error
	}{
		{"Flush", func(cs *CollectionStore) error { return cs.Flush("bulk") }},
		{"DropCollection", func(cs *CollectionStore) error { return cs.DropCollection("bulk") }},
		{"RestoreCollectionWithConfig", func(cs *CollectionStore) error {
			src, err := NewCollection("src", durablePersistentCfg())
			if err != nil {
				return err
			}
			defer func() { _ = src.Close() }()
			var buf bytes.Buffer
			if err := src.Snapshot(&buf); err != nil {
				return err
			}
			return cs.RestoreCollectionWithConfig("bulk", durablePersistentCfg(), &buf)
		}},
		{"EvictCollection", func(cs *CollectionStore) error {
			return cs.EvictCollection(context.Background(), "bulk", objstore.NewMemStore(), "t", time.Unix(1, 0))
		}},
	}
	for _, tc := range ops {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cs := openBulkStore(t, dir)
			t.Cleanup(func() { _ = cs.Close() })
			stageThree(t, cs, "bulk")
			done := make(chan error, 1)
			setBuildStagedHook(t, func() {
				go func() { done <- tc.op(cs) }()
				time.Sleep(50 * time.Millisecond)
			})
			built := make(chan error, 1)
			go func() { built <- cs.BuildStaged("bulk", 1) }()
			for _, ch := range []chan error{built, done} {
				select {
				case <-ch:
				case <-time.After(30 * time.Second):
					t.Fatalf("%s and a durable bulk build deadlocked", tc.name)
				}
			}
		})
	}
}

// TestDurableBulkBuildRejectedOnPopulatedStaysWritable: a second bulk load on
// a populated listed collection is refused before anything is placed
// (ErrBuildNonEmpty). Nothing changed, so the log stays healthy: the next write
// and a checkpoint succeed and survive a reopen.
func TestDurableBulkBuildRejectedOnPopulatedStaysWritable(t *testing.T) {
	dir := t.TempDir()
	cs := openBulkStore(t, dir)
	if err := cs.Upsert("bulk", 1, durableVec(1), "first", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	stageThree(t, cs, "bulk")
	if err := cs.BuildStaged("bulk", 1); !errors.Is(err, ErrBuildNonEmpty) {
		t.Fatalf("BuildStaged on a populated collection = %v, want ErrBuildNonEmpty", err)
	}
	if err := cs.Upsert("bulk", 2, durableVec(2), "second", 0, nil, nil); err != nil {
		t.Fatalf("a write after a rejected build = %v, want nil", err)
	}
	if err := cs.CheckpointDurable(); err != nil {
		t.Fatalf("a checkpoint after a rejected build = %v, want nil", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	re := openDurable(t, dir, "bulk")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "bulk"), map[uint64]string{1: "first", 2: "second"})
}

// TestDurableBulkBuildQuotaRejectionStaysWritable: a quota refusal comes before
// placement too, on an empty collection, and leaves it writable.
func TestDurableBulkBuildQuotaRejectionStaysWritable(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "bulk")
	t.Cleanup(func() { _ = cs.Close() })
	cfg := durablePersistentCfg()
	cfg.MaxVectors = 2
	if err := cs.CreateCollection("bulk", cfg); err != nil {
		t.Fatal(err)
	}
	stageThree(t, cs, "bulk")
	if err := cs.BuildStaged("bulk", 1); !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("BuildStaged over quota = %v, want ErrCollectionFull", err)
	}
	if err := cs.Upsert("bulk", 9, durableVec(9), "x", 0, nil, nil); err != nil {
		t.Fatalf("a write after a quota-rejected build = %v, want nil", err)
	}
}

// setBuildStagedFn replaces the build BuildStaged runs. Tests only.
func setBuildStagedFn(t *testing.T, fn func(c *Collection, workers int) (bool, error)) {
	t.Helper()
	prev := buildStagedFn
	buildStagedFn = fn
	t.Cleanup(func() { buildStagedFn = prev })
}

// TestDurableBulkBuildFailureAfterPlacementPoisons: a build on an empty listed
// collection that places points and then fails, or panics, leaves memory
// ahead of every file; later writes are refused.
func TestDurableBulkBuildFailureAfterPlacementPoisons(t *testing.T) {
	for _, panics := range []bool{false, true} {
		name := "error"
		if panics {
			name = "panic"
		}
		t.Run(name, func(t *testing.T) {
			const injectedPanic = "injected: build panics after placement"
			dir := t.TempDir()
			cs := openBulkStore(t, dir)
			t.Cleanup(func() { _ = cs.Close() })
			stageThree(t, cs, "bulk")
			setBuildStagedFn(t, func(c *Collection, workers int) (bool, error) {
				if _, err := c.buildStaged(workers); err != nil {
					return true, err
				}
				if panics {
					panic(injectedPanic)
				}
				return true, errors.New("injected: build fails after placement")
			})
			func() {
				// Only the injected panic, and only in the panic case: any other
				// recovered value fails the test.
				defer func() {
					var want any
					if panics {
						want = injectedPanic
					}
					if r := recover(); r != want {
						t.Fatalf("recovered %v, want %v", r, want)
					}
				}()
				if err := cs.BuildStaged("bulk", 1); err == nil {
					t.Fatal("BuildStaged succeeded although the build failed")
				}
			}()
			if err := cs.Upsert("bulk", 9, durableVec(9), "x", 0, nil, nil); !errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("a write after a build that failed after placement = %v, want ErrWALPoisoned", err)
			}
		})
	}
}
