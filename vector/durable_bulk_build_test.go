// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"path/filepath"
	"reflect"
	"testing"
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
