// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"path/filepath"
	"testing"
)

// moveToFlatLayout moves col's manifest and snapshot from vectors/<tenant>/ to
// the legacy flat layout, vectors/<col>.json and vectors/<col>.snap, which an
// open still loads. The log stays in the tenant directory, where it always is.
func moveToFlatLayout(t *testing.T, dir, col string) {
	t.Helper()
	vd := filepath.Join(dir, "vectors")
	for _, ext := range []string{".json", ".snap"} {
		from := filepath.Join(vd, DefaultTenant, col+ext)
		if _, err := os.Stat(from); os.IsNotExist(err) {
			continue
		}
		if err := os.Rename(from, filepath.Join(vd, col+ext)); err != nil {
			t.Fatal(err)
		}
	}
}

// TestCheckpointDurableFlatLayoutKeepsWrites: a durable collection still in the
// flat layout loads its snapshot from vectors/<col>.snap. A checkpoint used to
// write the tenant path instead and then empty the log, so the next open read
// the old flat snapshot with an empty log and the checkpointed writes were gone.
func TestCheckpointDurableFlatLayoutKeepsWrites(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "docs")
	if err := cs.CreateCollection("docs", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert("docs", 1, durableVec(1), "before", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.CheckpointDurable(); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	moveToFlatLayout(t, dir, "docs")

	cs = openDurable(t, dir, "docs")
	if err := cs.Upsert("docs", 2, durableVec(2), "after", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.CheckpointDurable(); err != nil {
		t.Fatal(err)
	}
	// Reopen what CheckpointDurable left, before anything else writes: a copy
	// taken now, while the store is still open (Close writes nothing either).
	checkpointed := filepath.Join(t.TempDir(), "copy")
	copyTree(t, dir, checkpointed)
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	re := openDurable(t, checkpointed, "docs")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "docs"), map[uint64]string{1: "before", 2: "after"})
}
