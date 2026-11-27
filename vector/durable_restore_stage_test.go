// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
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
