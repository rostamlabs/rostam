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
// SPLIT: ReconcileBatch samples and marks, DropReconciled drops, and the
// liveness probe in between is run by a caller that holds no lock here. The
// whole-tick tests (the probe, the expired-key deadlock, the ticker) live in
// ops, shard and the root package, which are the ones that can reach a cache.

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

// TestReconcileBatchIsBudgeted: a batch is exactly min(budget, live keys) —
// never more, so a tick's cost is bounded, and never fewer when the whole index
// fits, so a small index is covered in one tick.
func TestReconcileBatchIsBudgeted(t *testing.T) {
	for _, tc := range []struct{ posted, budget, want int }{
		{100, 30, 30},
		{100, 100, 100},
		{100, 250, 100}, // budget above the index size is capped by the index
		{0, 30, 0},
	} {
		t.Run(fmt.Sprintf("posted=%d/budget=%d", tc.posted, tc.budget), func(t *testing.T) {
			s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
			postKeys(s, tc.posted, 7)
			keys := s.ReconcileBatch("by-rc", tc.budget)
			if len(keys) != tc.want {
				t.Fatalf("batch returned %d keys, want %d", len(keys), tc.want)
			}
			// No duplicates within one batch: a range visits each entry once.
			seen := make(map[string]bool, len(keys))
			for _, k := range strs(keys) {
				if seen[k] {
					t.Fatalf("batch returned %q twice", k)
				}
				seen[k] = true
			}
			s.DropReconciled("by-rc", nil)
		})
	}
}

// TestReconcileBatchNonPositiveBudget: a caller that asks for nothing gets
// nothing, and no marks are left behind for a later drop to honour.
func TestReconcileBatchNonPositiveBudget(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	all := postKeys(s, 5, 7)
	for _, budget := range []int{0, -1} {
		if keys := s.ReconcileBatch("by-rc", budget); keys != nil {
			t.Fatalf("ReconcileBatch(budget=%d) = %v, want nil", budget, strs(keys))
		}
	}
	dead := make([][]byte, 0, len(all))
	for _, k := range all {
		dead = append(dead, []byte(k))
	}
	if n := s.DropReconciled("by-rc", dead); n != 0 {
		t.Fatalf("DropReconciled after a zero-budget batch dropped %d, want 0", n)
	}
}

// TestReconcileBatchCoversEveryKeyEventually is the property that replaced the
// ordered rotation.
//
// A batch is a TRUNCATED RANGE over the reverse map, and Go starts every range
// at a randomly chosen bucket, so coverage is probabilistic: the expected
// number of ticks to touch all n keys is the coupon-collector bound
// n·ln(n)/budget, with no guarantee for any one key on any one tick. That is
// the trade the pass makes for a lock hold of microseconds instead of a walk of
// every live key. The bound below is ~40x the expectation, so a failure means
// the sample is not moving at all — a fixed starting point, say — rather than
// an unlucky run.
func TestReconcileBatchCoversEveryKeyEventually(t *testing.T) {
	const (
		posted   = 100
		budget   = 10
		maxTicks = 2000 // expectation is ~46
	)
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	want := postKeys(s, posted, 7)

	seen := make(map[string]bool, posted)
	ticks := 0
	for ; ticks < maxTicks && len(seen) < posted; ticks++ {
		for _, k := range strs(s.ReconcileBatch("by-rc", budget)) {
			seen[k] = true
		}
		s.DropReconciled("by-rc", nil) // end the tick; drop nothing
	}
	if len(seen) != posted {
		missing := make([]string, 0, posted-len(seen))
		for _, k := range want {
			if !seen[k] {
				missing = append(missing, k)
			}
		}
		t.Fatalf("after %d ticks %d/%d keys had been sampled; never sampled: %v",
			ticks, len(seen), posted, missing[:min(len(missing), 8)])
	}
	t.Logf("covered %d keys in %d ticks at budget %d (coupon-collector expectation ~46)", posted, ticks, budget)
}

// TestReconcileBatchUnknownIndex: an index that is not installed has nothing to
// reconcile and says so rather than panicking.
func TestReconcileBatchUnknownIndex(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	if keys := s.ReconcileBatch("nope", 10); keys != nil {
		t.Fatalf("ReconcileBatch on an uninstalled index = %v, want nil", strs(keys))
	}
	if n := s.DropReconciled("nope", [][]byte{[]byte("u:000")}); n != 0 {
		t.Fatalf("DropReconciled on an uninstalled index dropped %d", n)
	}
}

// TestDropReconciledRechecksReverseMap is the re-add race, and it is the one
// that decides whether this pass can lose a row.
//
// The probe runs with NO index lock held (it must — see the package doc), so a
// write can land between the batch and the drop. If the drop trusted the batch,
// a key that was probed as dead and then written would lose its posting while
// being live: a MISSING posting, which verify-on-read cannot repair.
//
// Four shapes, because each defeats a different cheaper guard:
//
//   - deleted and re-added under a DIFFERENT value — a "still in the reverse
//     map?" check alone would drop it;
//   - deleted and re-added under the SAME value — a "still mapped to the value
//     I sampled?" check would drop it too (ABA);
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
	}{
		{"deleted then re-added under a different value", true, 9},
		{"deleted then re-added under the same value (ABA)", true, 7},
		{"rewritten in place under a different value", false, 9},
		{"rewritten in place under the same value", false, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
			postKeys(s, 3, 7)

			// Budget above the index size, so the batch is the whole of it and the
			// written key is certainly in it.
			keys := s.ReconcileBatch("by-rc", 10)
			if len(keys) != 3 {
				t.Fatalf("batch returned %d keys, want 3", len(keys))
			}
			// The window: the write lands while no index lock is held.
			if tc.delete {
				s.Drop([]byte("u:001"))
			}
			s.Reindex([]byte("u:001"), intRec("rc", tc.reAdd))

			// The probe (run before the write) said every key was dead.
			if n := s.DropReconciled("by-rc", keys); n != 2 {
				t.Fatalf("DropReconciled dropped %d, want 2 (the written key must survive)", n)
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

// TestDropReconciledIgnoresKeysItNeverSampled: a caller cannot hand
// DropReconciled a key the batch did not offer it and have it removed. Without
// this the drop is an unguarded delete-by-key on a live index.
func TestDropReconciledIgnoresKeysItNeverSampled(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	all := postKeys(s, 5, 7)

	// One key sampled, four not. Which one is random, so the unsampled key is
	// chosen by difference rather than assumed.
	keys := s.ReconcileBatch("by-rc", 1)
	if len(keys) != 1 {
		t.Fatalf("batch returned %d keys, want 1", len(keys))
	}
	sampled := string(keys[0])
	var unsampled string
	for _, k := range all {
		if k != sampled {
			unsampled = k
			break
		}
	}
	if n := s.DropReconciled("by-rc", [][]byte{[]byte(unsampled)}); n != 0 {
		t.Fatalf("DropReconciled removed %q, which its batch never offered", unsampled)
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
	keys := s.ReconcileBatch("by-rc", 10)
	if n := s.DropReconciled("by-rc", keys); n != 4 {
		t.Fatalf("dropped %d, want 4", n)
	}
	if got := s.ReconcileDrops(); got != 4 {
		t.Fatalf("ReconcileDrops() = %d, want 4", got)
	}
}

// TestReconcileStateClearedByReset: a flush empties the postings, so a tick
// that straddles it must drop nothing — the keys it sampled describe a keyspace
// that no longer exists.
func TestReconcileStateClearedByReset(t *testing.T) {
	s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
	postKeys(s, 4, 7)
	keys := s.ReconcileBatch("by-rc", 10)

	s.Reset()
	postKeys(s, 4, 7) // the keyspace refilled after the flush

	if n := s.DropReconciled("by-rc", keys); n != 0 {
		t.Fatalf("a tick straddling a Reset dropped %d postings, want 0", n)
	}
	if got, _ := s.Stats("by-rc"); got != 4 {
		t.Fatalf("index holds %d keys, want 4", got)
	}
}

// TestReconcileStateAcrossInstall pins BOTH halves of what an Install does to
// an in-flight tick, because Install is two different events wearing one name.
//
//   - A definition whose path or prefix MOVED gets a fresh posting, so the
//     tick's marks (and the keys behind them) describe an index that answers a
//     different question: it must drop nothing.
//   - A definition that did not move (sameShape) KEEPS its posting, postings
//     and readiness included. The marks are still in that same map and still
//     name the same entries, so honouring them is correct — dropping them
//     instead would silently make an Install of an unchanged catalog cancel
//     every reconcile tick that happened to overlap it, and the observer
//     re-installs on every meta write.
func TestReconcileStateAcrossInstall(t *testing.T) {
	t.Run("a replaced posting loses the marks", func(t *testing.T) {
		s := readySet(mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar))
		postKeys(s, 4, 7)
		keys := s.ReconcileBatch("by-rc", 10)

		s.Install([]Def{mustDef(t, "by-rc", "u:", "other", wire.KVIndexKindScalar)})
		postKeys(s, 4, 7) // posts nothing: no "other" field
		s.Reindex([]byte("u:000"), intRec("other", 1))

		if n := s.DropReconciled("by-rc", keys); n != 0 {
			t.Fatalf("a tick straddling a REPLACING Install dropped %d postings, want 0", n)
		}
		if got, _ := s.Stats("by-rc"); got != 1 {
			t.Fatalf("index holds %d keys, want the 1 posted under the new definition", got)
		}
	})

	t.Run("a same-shape Install keeps the marks", func(t *testing.T) {
		d := mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)
		s := readySet(d)
		postKeys(s, 4, 7)
		keys := s.ReconcileBatch("by-rc", 10)

		// The same definition again — what the observer re-installs on any meta
		// write. sameShape, so the posting (and its marks) are kept.
		s.Install([]Def{mustDef(t, "by-rc", "u:", "rc", wire.KVIndexKindScalar)})

		if n := s.DropReconciled("by-rc", keys); n != 4 {
			t.Fatalf("a tick straddling a SAME-SHAPE Install dropped %d postings, want 4", n)
		}
		if got, _ := s.Stats("by-rc"); got != 0 {
			t.Fatalf("index holds %d keys, want 0", got)
		}
	})
}

// BenchmarkReconcileBatch measures the LOCK HOLD a tick costs, which is the
// number the truncated-range design exists to keep flat. The ordered-cursor
// design it replaced was O(live keys) per tick — 53 ms at 1 M keys, 208 ms at
// 4 M — and because sync.RWMutex stops admitting readers once a writer waits,
// one Reindex arriving mid-walk pinned every later Candidates behind the rest
// of it. This one is bounded by the BUDGET instead: 0.74 ms at 100 k keys and
// 1.31 ms at 1 M, so 10x the index costs 1.8x the hold rather than 10x.
func BenchmarkReconcileBatch(b *testing.B) {
	for _, n := range []int{100_000, 1_000_000} {
		b.Run(fmt.Sprintf("keys=%d", n), func(b *testing.B) {
			s := New(1024)
			d, err := DefFrom(wire.KVIndexDef{
				Name: "by-rc", KeyPrefix: []byte("u:"), PayloadPath: "rc",
				Kind: wire.KVIndexKindScalar, Enabled: true,
			}, 1)
			if err != nil {
				b.Fatal(err)
			}
			s.Install([]Def{d})
			s.MarkReady("by-rc")
			rec := intRec("rc", 7)
			for i := 0; i < n; i++ {
				s.Reindex([]byte(fmt.Sprintf("u:%09d", i)), rec)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				keys := s.ReconcileBatch("by-rc", ReconcileBudget)
				if len(keys) != ReconcileBudget {
					b.Fatalf("batch returned %d keys, want %d", len(keys), ReconcileBudget)
				}
				s.DropReconciled("by-rc", nil)
			}
		})
	}
}
