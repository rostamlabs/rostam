// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops/kvindex"
)

// The cluster's whole share of the bounded reconcile pass is the COUNTER. The
// ticker itself is store-owned (shard/kv_index_reconcile.go says why), so what
// has to hold here is that Stats().KVIndex.ReconcileDrops sums the hosted
// groups' counts and never goes backwards when a group leaves.

// forceReconcileDrops makes idx drop n of its own postings through the real
// two-phase API, without a cache: the keys are posted and never written, so
// every one of them is dangling by construction. It returns how many went.
//
// The liveness probe is deliberately not involved — that is pinned in ops,
// where a cache is reachable. What is being produced here is a count.
func forceReconcileDrops(t *testing.T, idx *kvindex.Set, name string, keys [][]byte, rec []byte) int {
	t.Helper()
	for _, k := range keys {
		idx.Reindex(k, rec)
	}
	batch, _ := idx.ReconcileBatch(name, kvindex.ReconcileBudget)
	if len(batch) == 0 {
		t.Fatalf("ReconcileBatch(%q) returned nothing to reconcile", name)
	}
	// Only the phantom keys are declared dead; anything else the batch picked up
	// is left alone.
	want := make(map[string]bool, len(keys))
	for _, k := range keys {
		want[string(k)] = true
	}
	dead := make([][]byte, 0, len(keys))
	for _, k := range batch {
		if want[string(k)] {
			dead = append(dead, k)
		}
	}
	return idx.DropReconciled(name, dead)
}

func TestKVIndexReconcileDropsSurviveShardRemoval(t *testing.T) {
	// Same contract, and the same defect, as VerifyMisses: the count is kept on
	// the per-shard Set (the pass runs on the store's own goroutine and cannot
	// reach this Node), so the node total is a SUM over hosted groups. Dropping a
	// group without folding its count in makes a contracted monotonic uint64
	// DECREASE, which a scraper reads as a process restart.
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]

	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	rec := kvIndexRecord(t, 1)

	hosted, total := 0, 0
	for group, s := range n.snapshotShards() {
		if s == nil {
			continue
		}
		idx := s.KVIndex()
		if idx == nil {
			continue
		}
		waitForKVIndex(t, 20*time.Second, "the observer to install and backfill by_age", func() bool {
			return idx.IsReady("by_age")
		})
		hosted++
		keys := [][]byte{
			[]byte("u:phantom-a"),
			[]byte("u:phantom-b"),
			[]byte("u:phantom-c"),
		}
		got := forceReconcileDrops(t, idx, "by_age", keys, rec)
		if got != len(keys) {
			t.Fatalf("group %d: reconcile dropped %d phantom postings, want %d", group, got, len(keys))
		}
		total += got
	}
	if hosted < 2 {
		t.Fatalf("fixture: %d hosted groups with an index, want 2", hosted)
	}

	before := n.Stats().KVIndex.ReconcileDrops
	if want := uint64(total); before != want { //nolint:gosec // total is a small positive test count
		t.Fatalf("ReconcileDrops = %d before removal, want %d", before, want)
	}

	if err := n.RemoveShardOwner(0); err != nil {
		t.Fatalf("RemoveShardOwner: %v", err)
	}

	if after := n.Stats().KVIndex.ReconcileDrops; after < before {
		t.Fatalf("ReconcileDrops went BACKWARDS across a shard removal: %d -> %d", before, after)
	} else if after != before {
		t.Fatalf("ReconcileDrops = %d after removal, want %d unchanged", after, before)
	}
}
