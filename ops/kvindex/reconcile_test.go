// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"fmt"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// Tests for the two halves of the bounded reconcile pass. What they pin is the
// SPLIT: ReconcileBatch snapshots and DropReconciled drops, with the liveness
// probe in between run by a caller that holds no lock here. The whole-tick
// tests (the probe, the expired-key deadlock, the ticker) live in ops and
// shard, which are the packages that can reach a cache.

// postKeys posts n keys "u:%03d" under value v and returns them sorted.
func postKeys(s *Set, n int, v int64) []string {
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		k := fmt.Sprintf("u:%03d", i)
		s.Reindex([]byte(k), intRec("rc", v))
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func strs(keys [][]byte) []string {
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, string(k))
	}
	return out
}

// TestReconcileBatchIsBudgeted: a batch never returns more than budget keys,
// and successive batches cover the whole reverse map without repeating.
func TestReconcileBatchIsBudgeted(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	want := postKeys(s, 100, 7)

	var got []string
	for tick := 0; tick < 4; tick++ {
		keys, wrapped := s.ReconcileBatch("by-rc", 30)
		if len(keys) > 30 {
			t.Fatalf("tick %d returned %d keys, budget was 30", tick, len(keys))
		}
		got = append(got, strs(keys)...)
		s.DropReconciled("by-rc", nil) // end the tick, drop nothing
		if wrapped {
			if tick != 3 {
				t.Fatalf("wrapped on tick %d; 100 keys at budget 30 needs 4 ticks", tick)
			}
			break
		}
	}
	sort.Strings(got)
	if len(got) != len(want) {
		t.Fatalf("four ticks covered %d keys, want the whole %d-key reverse map", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("coverage differs at %d: got %q want %q", i, got[i], want[i])
		}
	}
}

// TestReconcileBatchWrapsAndRestarts: once the cursor reaches the end it
// reports wrapped and the next batch starts from the beginning again.
func TestReconcileBatchWrapsAndRestarts(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	postKeys(s, 10, 1)

	first, wrapped := s.ReconcileBatch("by-rc", 4)
	s.DropReconciled("by-rc", nil)
	if wrapped {
		t.Fatal("a 4-key batch over 10 keys must not report wrapped")
	}
	// Drain to the end.
	for i := 0; i < 5; i++ {
		_, wrapped = s.ReconcileBatch("by-rc", 4)
		s.DropReconciled("by-rc", nil)
		if wrapped {
			break
		}
	}
	if !wrapped {
		t.Fatal("the rotation never reached the end of a 10-key index")
	}
	again, _ := s.ReconcileBatch("by-rc", 4)
	s.DropReconciled("by-rc", nil)
	if got, want := strs(again), strs(first); len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("after wrapping the batch is %v, want the first batch %v again", got, want)
	}
}

// TestReconcileBatchUnknownIndex: an index that is not installed has nothing to
// reconcile and says so rather than panicking.
func TestReconcileBatchUnknownIndex(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	if keys, wrapped := s.ReconcileBatch("nope", 10); keys != nil || wrapped {
		t.Fatalf("ReconcileBatch on an uninstalled index = (%v, %v), want (nil, false)", keys, wrapped)
	}
	if n := s.DropReconciled("nope", [][]byte{[]byte("u:000")}); n != 0 {
		t.Fatalf("DropReconciled on an uninstalled index dropped %d", n)
	}
}

// TestDropReconciledRechecksReverseMap is the re-add race, and it is the one
// that decides whether this pass can lose a row.
//
// The probe runs with NO index lock held (it must — see the package doc), so a
// write can land between the snapshot and the drop. If the drop trusted the
// snapshot, a key that was probed as dead and then written would lose its
// posting while being live: a MISSING posting, which verify-on-read cannot
// repair.
//
// Four shapes, because each defeats a different cheaper guard:
//
//   - deleted and re-added under a DIFFERENT value — a "still in the reverse
//     map?" check alone would drop it;
//   - deleted and re-added under the SAME value — a "still mapped to the value
//     I snapshotted?" check would drop it too (ABA);
//   - REWRITTEN IN PLACE with a different value, never leaving the index — the
//     probe can miss a key that is still posted (a replicated shard filters an
//     expired key without removing its slot), so this is not hypothetical;
//   - rewritten in place with the SAME value, which takes posting.set's
//     unchanged-value early return and so must clear the mark BEFORE it.
func TestDropReconciledRechecksReverseMap(t *testing.T) {
	for _, tc := range []struct {
		name   string
		delete bool
		reAdd  int64
		want   int // postings the reconcile pass may drop
	}{
		{"deleted then re-added under a different value", true, 9, 2},
		{"deleted then re-added under the same value (ABA)", true, 7, 2},
		{"rewritten in place under a different value", false, 9, 2},
		{"rewritten in place under the same value", false, 7, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
			postKeys(s, 3, 7)

			keys, _ := s.ReconcileBatch("by-rc", 10)
			if len(keys) != 3 {
				t.Fatalf("batch returned %d keys, want 3", len(keys))
			}
			// The window: the write lands while no index lock is held.
			if tc.delete {
				s.Drop([]byte("u:001"))
			}
			s.Reindex([]byte("u:001"), intRec("rc", tc.reAdd))

			// The probe (run before the write) said every key was dead.
			if n := s.DropReconciled("by-rc", keys); n != tc.want {
				t.Fatalf("DropReconciled dropped %d, want %d (the written key must survive)", n, tc.want)
			}
			if got, _ := s.Stats("by-rc"); got != 1 {
				t.Fatalf("index holds %d keys, want 1 (the written u:001)", got)
			}
			d, _ := s.Lookup("by-rc")
			cands, err := s.Candidates(Selector{
				Def:    d,
				Op:     vtypes.FilterEq,
				Values: []vtypes.Value{vtypes.NewInt(tc.reAdd)},
			}, nil, 100)
			if err != nil {
				t.Fatalf("Candidates: %v", err)
			}
			if got := strs(cands); len(got) != 1 || got[0] != "u:001" {
				t.Fatalf("the surviving key is not queryable: candidates = %v, want [u:001]", got)
			}
		})
	}
}

// TestDropReconciledIgnoresKeysItNeverSnapshotted: a caller cannot hand
// DropReconciled a key the batch did not offer it and have it removed. Without
// this the drop is an unguarded delete-by-key on a live index.
func TestDropReconciledIgnoresKeysItNeverSnapshotted(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	postKeys(s, 5, 7)

	// A batch that covers only the first two keys.
	keys, _ := s.ReconcileBatch("by-rc", 2)
	if len(keys) != 2 {
		t.Fatalf("batch returned %d keys, want 2", len(keys))
	}
	if n := s.DropReconciled("by-rc", [][]byte{[]byte("u:004")}); n != 0 {
		t.Fatalf("DropReconciled removed %d keys outside its own batch, want 0", n)
	}
	if got, _ := s.Stats("by-rc"); got != 5 {
		t.Fatalf("index holds %d keys, want all 5", got)
	}
}

// TestReconcileDropsAreCounted: the Set counts what the pass removed, which is
// what the node's Stats().KVIndex.ReconcileDrops is summed from.
func TestReconcileDropsAreCounted(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	postKeys(s, 4, 7)
	if got := s.ReconcileDrops(); got != 0 {
		t.Fatalf("a fresh Set reports %d reconcile drops", got)
	}
	keys, _ := s.ReconcileBatch("by-rc", 10)
	if n := s.DropReconciled("by-rc", keys); n != 4 {
		t.Fatalf("dropped %d, want 4", n)
	}
	if got := s.ReconcileDrops(); got != 4 {
		t.Fatalf("ReconcileDrops() = %d, want 4", got)
	}
}

// TestReconcileStateClearedByReset: a flush empties the postings, so a tick
// that straddles it must drop nothing — the keys it snapshotted describe a
// keyspace that no longer exists.
func TestReconcileStateClearedByReset(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	postKeys(s, 4, 7)
	keys, _ := s.ReconcileBatch("by-rc", 10)

	s.Reset()
	postKeys(s, 4, 7) // the keyspace refilled after the flush

	if n := s.DropReconciled("by-rc", keys); n != 0 {
		t.Fatalf("a tick straddling a Reset dropped %d postings, want 0", n)
	}
	if got, _ := s.Stats("by-rc"); got != 4 {
		t.Fatalf("index holds %d keys, want 4", got)
	}
}

// TestReconcileStateClearedByInstall: a redefinition replaces the posting, so
// the in-flight tick's keys belong to an index that is no longer installed.
func TestReconcileStateClearedByInstall(t *testing.T) {
	d := mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	postKeys(s, 4, 7)
	keys, _ := s.ReconcileBatch("by-rc", 10)

	s.Install([]Def{mustDef(t, "by-rc", "u:", "other", wire.KVIndexKindScalar)}) // a different path: fresh posting
	postKeys(s, 4, 7)                                                            // posts nothing (no "other" field)
	s.Reindex([]byte("u:000"), intRec("other", 1))

	if n := s.DropReconciled("by-rc", keys); n != 0 {
		t.Fatalf("a tick straddling an Install dropped %d postings, want 0", n)
	}
	if got, _ := s.Stats("by-rc"); got != 1 {
		t.Fatalf("index holds %d keys, want the 1 posted under the new definition", got)
	}
}
