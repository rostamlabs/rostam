// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

const durableTestCollection = "mcp_memory"

// openDurableDirect opens a Direct store on dir with durableTestCollection
// listed as durable. As with openDirectOn, the caller sequences the Close.
func openDurableDirect(t *testing.T, dir string) Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{
		DataDir:            dir,
		Ops:                reg,
		Cache:              CacheConfig{NumShardsPerNode: 1, MaxMemoryBytes: 64 << 20},
		DurableCollections: []string{durableTestCollection},
	})
	if err != nil {
		t.Fatalf("NewDirect(%q): %v", dir, err)
	}
	return s
}

// createAndUpsert creates durableTestCollection asking for the Persistent mode,
// which the durable list overrides, and writes one point to it.
func createAndUpsert(t *testing.T, s Store) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, durableTestCollection, persistentTestConfig()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	vec := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	if err := s.VectorUpsert(ctx, durableTestCollection, 42, vec, "hello", VectorInsertOpts{}); err != nil {
		t.Fatalf("VectorUpsert: %v", err)
	}
}

func assertHasPoint(t *testing.T, s Store) {
	t.Helper()
	docs, _, _, err := s.VectorScroll(context.Background(), durableTestCollection, vector.Filter{}, 10, VectorScrollOpts{})
	if err != nil {
		t.Fatalf("VectorScroll: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != 42 || docs[0].Content != "hello" {
		t.Fatalf("scroll = %+v, want the one point 42 with content %q", docs, "hello")
	}
}

// copyDataDir copies src into a fresh temp dir, as the files stand right now.
func copyDataDir(t *testing.T, src string) string {
	t.Helper()
	dst := t.TempDir()
	err := filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o750)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, b, 0o600)
	})
	if err != nil {
		t.Fatalf("copy %s: %v", src, err)
	}
	return dst
}

// TestDirectDurableCollectionSurvivesWithoutClose: an acknowledged write to a
// listed collection is on disk before any clean shutdown, so a copy of the data
// directory taken while the store is still open holds it.
func TestDirectDurableCollectionSurvivesWithoutClose(t *testing.T) {
	dir := t.TempDir()
	s := openDurableDirect(t, dir)
	t.Cleanup(func() { _ = s.Close() })
	createAndUpsert(t, s)

	copied := openDurableDirect(t, copyDataDir(t, dir))
	t.Cleanup(func() { _ = copied.Close() })
	assertHasPoint(t, copied)
}

// TestDirectCloseCheckpointsDurable: a clean Close snapshots a listed
// collection and empties its log, and the point is still there on reopen.
func TestDirectCloseCheckpointsDurable(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "vectors", vector.DefaultTenant, durableTestCollection+".wal")

	s := openDurableDirect(t, dir)
	createAndUpsert(t, s)
	before, err := os.Stat(walPath)
	if err != nil {
		t.Fatalf("stat log before Close: %v", err)
	}
	if before.Size() == 0 {
		t.Fatal("log is empty before Close: the write was not logged")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	after, err := os.Stat(walPath)
	if err != nil {
		t.Fatalf("stat log after Close: %v", err)
	}
	if after.Size() != 0 {
		t.Fatalf("log size after Close = %d, want 0", after.Size())
	}

	s2 := openDurableDirect(t, dir)
	t.Cleanup(func() { _ = s2.Close() })
	assertHasPoint(t, s2)
}
