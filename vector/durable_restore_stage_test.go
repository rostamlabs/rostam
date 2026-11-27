// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// TestDurableRestoreFailedSnapshotKeepsOldCollection: a durable restore writes
// the restored snapshot before it removes the collection it replaces. It used
// to remove the old collection first, so a snapshot that could not be written
// left neither collection.
func TestDurableRestoreFailedSnapshotKeepsOldCollection(t *testing.T) {
	src := openDurable(t, t.TempDir())
	if err := src.CreateCollection("docs", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := src.Upsert("docs", 7, durableVec(7), "restored", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	c, _ := src.Get("docs")
	var snap bytes.Buffer
	if err := c.Snapshot(&snap); err != nil {
		t.Fatal(err)
	}
	_ = src.Close()

	dir := t.TempDir()
	ds := openDurable(t, dir, "docs")
	if err := ds.CreateCollection("docs", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := ds.Upsert("docs", 1, durableVec(1), "old", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	// A directory where the snapshot's temporary file goes makes writing it fail
	// (under either staging name).
	snapPath := filepath.Join(dir, "vectors", DefaultTenant, "docs.snap")
	for _, p := range []string{snapPath + ".tmp", snapPath + restoreStageSuffix + ".tmp"} {
		if err := os.MkdirAll(p, 0o750); err != nil {
			t.Fatal(err)
		}
	}
	if err := ds.RestoreCollectionWithConfig("docs", durablePersistentCfg(), bytes.NewReader(snap.Bytes())); err == nil {
		t.Fatal("a restore whose snapshot could not be written succeeded")
	}
	assertPoints(t, pointsOf(t, ds, "docs"), map[uint64]string{1: "old"})
	if err := ds.Close(); err != nil {
		t.Fatal(err)
	}
	re := openDurable(t, dir, "docs")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "docs"), map[uint64]string{1: "old"})
}

// TestDurableRestoreLeftoversRemoved: a stop in the middle of a durable restore
// can leave its staged snapshot (and the staging file's own temporary) beside
// the collection. The next open removes them, and so does a drop.
func TestDurableRestoreLeftoversRemoved(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "docs")
	if err := cs.CreateCollection("docs", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()
	snapPath := filepath.Join(dir, "vectors", DefaultTenant, "docs.snap")
	plant := func() []string {
		t.Helper()
		ps := []string{snapPath + restoreStageSuffix, snapPath + restoreStageSuffix + ".tmp"}
		for _, p := range ps {
			if err := os.WriteFile(p, []byte("leftover"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		return ps
	}
	left := plant()
	cs = openDurable(t, dir, "docs")
	defer func() { _ = cs.Close() }()
	for _, p := range left {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("%s left after open (stat err %v)", p, err)
		}
	}
	left = plant()
	if err := cs.DropCollection("docs"); err != nil {
		t.Fatal(err)
	}
	for _, p := range left {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("%s left after drop (stat err %v)", p, err)
		}
	}
}

// durableSnapshotOf returns a snapshot of a collection holding point 7.
func durableSnapshotOf(t *testing.T) []byte {
	t.Helper()
	src := openDurable(t, t.TempDir())
	defer func() { _ = src.Close() }()
	if err := src.CreateCollection("docs", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := src.Upsert("docs", 7, durableVec(7), "restored", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	c, _ := src.Get("docs")
	var snap bytes.Buffer
	if err := c.Snapshot(&snap); err != nil {
		t.Fatal(err)
	}
	return snap.Bytes()
}

// TestDurableRestoreOverExistingCollection: a durable restore over a live
// collection replaces it, and the restored points survive a reopen.
func TestDurableRestoreOverExistingCollection(t *testing.T) {
	snap := durableSnapshotOf(t)
	for _, cold := range []bool{false, true} {
		name := "hot"
		if cold {
			name = "cold"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			ds := openDurable(t, dir, "docs")
			if err := ds.CreateCollection("docs", durablePersistentCfg()); err != nil {
				t.Fatal(err)
			}
			if err := ds.Upsert("docs", 1, durableVec(1), "old", 0, nil, nil); err != nil {
				t.Fatal(err)
			}
			if cold {
				if err := ds.EvictCollection(context.Background(), "docs", objstore.NewMemStore(), "t", time.Unix(1, 0)); err != nil {
					t.Fatal(err)
				}
				if !ds.IsCold("docs") {
					t.Fatal("not evicted")
				}
			}
			if err := ds.RestoreCollectionWithConfig("docs", durablePersistentCfg(), bytes.NewReader(snap)); err != nil {
				t.Fatalf("restore over an existing collection: %v", err)
			}
			assertPoints(t, pointsOf(t, ds, "docs"), map[uint64]string{7: "restored"})
			// Acknowledged once the restore returns: on disk without a Close.
			cp := t.TempDir()
			copyTree(t, dir, cp)
			_ = ds.Close()
			re := openDurable(t, cp, "docs")
			defer func() { _ = re.Close() }()
			assertPoints(t, pointsOf(t, re, "docs"), map[uint64]string{7: "restored"})
		})
	}
}
