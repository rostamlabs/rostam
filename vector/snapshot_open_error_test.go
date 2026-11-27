// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A snapshot that exists but cannot be opened is not "no snapshot": the open
// fails and no file is removed or rewritten, so a later open, once the file can
// be read, still holds every point.

var errSnapshotUnreadable = errors.New("injected: snapshot cannot be opened")

// failSnapshotOpen makes every open of a path ending in ext fail.
func failSnapshotOpen(t *testing.T, ext string) {
	t.Helper()
	prev := openSnapshotFile
	openSnapshotFile = func(name string) (*os.File, error) {
		if strings.HasSuffix(name, ext) {
			return nil, errSnapshotUnreadable
		}
		return prev(name)
	}
	t.Cleanup(func() { openSnapshotFile = prev })
}

// dirFiles reads every regular file under dir, keyed by relative path.
func dirFiles(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		b, rerr := os.ReadFile(p) //nolint:gosec // test temp dir
		if rerr != nil {
			return rerr
		}
		rel, _ := filepath.Rel(dir, p)
		out[rel] = b
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func assertSameFiles(t *testing.T, got, want map[string][]byte) {
	t.Helper()
	for k, w := range want {
		g, ok := got[k]
		if !ok {
			t.Fatalf("file %s was removed", k)
		}
		if !bytes.Equal(g, w) {
			t.Fatalf("file %s was rewritten", k)
		}
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			t.Fatalf("file %s was created", k)
		}
	}
}

func TestUnopenableSnapshotFailsTheOpen(t *testing.T) {
	cases := []struct {
		name string
		ext  string
		// seed creates the collection with points, checkpoints it and closes.
		seed func(t *testing.T, dir string)
		// count reports how many points the reopened store holds.
		count func(t *testing.T, cs *CollectionStore) int
	}{
		{"dense with a log", ".snap", func(t *testing.T, dir string) {
			cs := openDurable(t, dir)
			if err := cs.CreateCollection(tailCol, tailDenseCfg()); err != nil {
				t.Fatal(err)
			}
			for id := uint64(1); id <= 3; id++ {
				if err := cs.Upsert(tailCol, id, tailVec(id), "x", 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
			if err := cs.Flush(tailCol); err != nil {
				t.Fatal(err)
			}
			if err := cs.Close(); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, cs *CollectionStore) int { return len(pointsOf(t, cs, tailCol)) }},
		{"dense without a log", ".snap", func(t *testing.T, dir string) {
			cs := openDurable(t, dir)
			cfg := tailDenseCfg()
			cfg.WAL = false
			if err := cs.CreateCollection(tailCol, cfg); err != nil {
				t.Fatal(err)
			}
			for id := uint64(1); id <= 3; id++ {
				if err := cs.Upsert(tailCol, id, tailVec(id), "x", 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
			if err := cs.Flush(tailCol); err != nil {
				t.Fatal(err)
			}
			if err := cs.Close(); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, cs *CollectionStore) int { return len(pointsOf(t, cs, tailCol)) }},
		{"named", ".nsnap", func(t *testing.T, dir string) {
			cs := openDurable(t, dir)
			if err := cs.CreateCollection(tailCol, namedWALConfig()); err != nil {
				t.Fatal(err)
			}
			nc, _ := cs.GetNamed(tailCol)
			for id := uint64(1); id <= 3; id++ {
				if _, err := nc.InsertCAS(id, map[string][]float32{"title": {float32(id), 1, 0, 0}}, nil, 0, CASCond{}); err != nil {
					t.Fatal(err)
				}
			}
			if err := cs.FlushNamed(tailCol); err != nil {
				t.Fatal(err)
			}
			if err := cs.Close(); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, cs *CollectionStore) int {
			nc, ok := cs.GetNamed(tailCol)
			if !ok {
				t.Fatal("named collection missing")
			}
			return nc.NumPoints()
		}},
		{"multi-vector", ".mvwsnap", func(t *testing.T, dir string) {
			cs := openDurable(t, dir)
			if err := cs.CreateMultiVector(tailCol, mvWALConfig()); err != nil {
				t.Fatal(err)
			}
			idx, _ := cs.GetMultiVector(tailCol)
			for id := uint64(1); id <= 3; id++ {
				if _, err := idx.AddCAS(id, [][]float32{{float32(id), 1, 0, 0}}, nil, CASCond{}); err != nil {
					t.Fatal(err)
				}
			}
			if err := cs.FlushMVWAL(tailCol); err != nil {
				t.Fatal(err)
			}
			if err := cs.Close(); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, cs *CollectionStore) int {
			idx, ok := cs.GetMultiVector(tailCol)
			if !ok {
				t.Fatal("multi-vector collection missing")
			}
			return idx.NumDocs()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.seed(t, dir)
			if m, _ := filepath.Glob(filepath.Join(dir, "vectors", DefaultTenant, "*"+tc.ext)); len(m) != 1 {
				t.Fatalf("seed left no %s snapshot: %v", tc.ext, m)
			}
			before := dirFiles(t, dir)

			failSnapshotOpen(t, tc.ext)
			cs, err := OpenCollectionStoreWith(dir, StoreOptions{})
			if err == nil {
				_ = cs.Close()
				t.Fatal("the store opened with a snapshot it could not read")
			}
			if !errors.Is(err, errSnapshotUnreadable) {
				t.Fatalf("open = %v, want the snapshot's open error", err)
			}
			assertSameFiles(t, dirFiles(t, dir), before)

			openSnapshotFile = os.Open
			re := openDurable(t, dir)
			defer func() { _ = re.Close() }()
			if n := tc.count(t, re); n != 3 {
				t.Fatalf("reopen holds %d points, want 3", n)
			}
		})
	}
}

// TestListedLogManifestWithoutSnapshotBesideSidecarFailsTheOpen: a listed
// collection whose manifest names the log form, whose snapshot is missing and
// whose earlier mapped form's sidecar is still there cannot be loaded safely:
// the upgrade writes the snapshot before the manifest, so the snapshot is gone
// for a reason the store cannot know, and loading would remove the sidecar.
func TestListedLogManifestWithoutSnapshotBesideSidecarFailsTheOpen(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, replaceCol)
	if err := cs.CreateCollection(replaceCol, durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(replaceCol, 1, durableVec(1), "old", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Flush(replaceCol); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	if !manifestOf(t, dir, replaceCol).WAL {
		t.Fatal("the listed collection's manifest does not name the log form")
	}
	if err := os.Remove(snapPathOf(dir, replaceCol)); err != nil {
		t.Fatal(err)
	}
	metaPath := filepath.Join(dir, "vectors", DefaultTenant, replaceCol+".meta")
	if err := os.WriteFile(metaPath, []byte("sidecar of the earlier mapped form"), 0o600); err != nil {
		t.Fatal(err)
	}
	before := dirFiles(t, dir)

	re, err := OpenCollectionStoreWith(dir, StoreOptions{DurableCollections: []string{replaceCol}})
	if err == nil {
		_ = re.Close()
		t.Fatal("the store opened a listed log collection with no snapshot beside a sidecar")
	}
	assertSameFiles(t, dirFiles(t, dir), before)
}

// TestDanglingSnapshotSymlinkFailsTheOpen: a snapshot entry that is a symlink to
// a missing file is not "no snapshot". The open follows it and gets not-exist,
// but the entry is there, so the store must fail the open rather than load the
// collection without its checkpointed points.
func TestDanglingSnapshotSymlinkFailsTheOpen(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	if err := cs.CreateCollection(tailCol, tailDenseCfg()); err != nil {
		t.Fatal(err)
	}
	for id := uint64(1); id <= 3; id++ {
		if err := cs.Upsert(tailCol, id, tailVec(id), "x", 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := cs.Flush(tailCol); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	snap := filepath.Join(dir, "vectors", DefaultTenant, tailCol+".snap")
	moved := filepath.Join(dir, "elsewhere.snap")
	if err := os.Rename(snap, moved); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "missing.snap"), snap); err != nil {
		t.Skipf("symlinks not available: %v", err)
	}

	cs, err := OpenCollectionStoreWith(dir, StoreOptions{})
	if err == nil {
		n := len(pointsOf(t, cs, tailCol))
		_ = cs.Close()
		t.Fatalf("the store opened through a dangling snapshot symlink, holding %d points", n)
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("open = %v, want the snapshot's not-exist error", err)
	}

	if err := os.Remove(snap); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(moved, snap); err != nil {
		t.Fatal(err)
	}
	re := openDurable(t, dir)
	defer func() { _ = re.Close() }()
	if n := len(pointsOf(t, re, tailCol)); n != 3 {
		t.Fatalf("reopen holds %d points, want 3", n)
	}
}

// TestMissingSnapshotIsNoSnapshot: an absent entry is still "no snapshot".
func TestMissingSnapshotIsNoSnapshot(t *testing.T) {
	f, err := openSnapshot(filepath.Join(t.TempDir(), "absent.snap"))
	if f != nil || err != nil {
		t.Fatalf("openSnapshot(absent) = (%v, %v), want (nil, nil)", f, err)
	}
}
