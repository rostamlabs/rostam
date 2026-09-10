// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// The whole-tick tests: ReconcileKVIndex is the one caller that runs the three
// phases against a real cache, so this is where the lock discipline and the
// liveness rule are actually provable. The leaf-side split (batching, the
// re-add race, the rotation) is pinned in ops/kvindex.

const recIndexName = "by-rc"

// recRec encodes the one-field record the fixture index posts on.
func recRec(v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}, //nolint:gosec // fixture reinterprets a small i64
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

func recDef(t *testing.T) kvindex.Def {
	t.Helper()
	d, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        recIndexName,
		KeyPrefix:   []byte("u:"),
		PayloadPath: "rc",
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom: %v", err)
	}
	return d
}

// recCache builds a NON-REPLICATED single-shard heap cache with the background
// TTL sweeper off, wired to a fresh index Set through the same seam a Store
// uses. Non-replicated is load-bearing for the deadlock test: only there does
// Cache.Get of an expired key physically remove it and fire onRemove.
func recCache(t *testing.T) (*cache.Cache, *kvindex.Set) {
	t.Helper()
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.TTLSweepIntervalMs = 0 // the probe's own Get must be what expires a key
	c, err := cache.New(cc)
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	idx := NewKVIndexFor(c)
	idx.Install([]kvindex.Def{recDef(t)})
	idx.MarkReady(recIndexName)
	return c, idx
}

// recPut writes a key and reindexes it, in the order the write path uses
// (reindex only after the Put returns — see cache.SetOnRemove).
func recPut(t *testing.T, c *cache.Cache, idx *kvindex.Set, key string, v int64, ttl time.Duration) {
	t.Helper()
	val := recRec(v)
	if err := c.Put([]byte(key), val, ttl); err != nil {
		t.Fatalf("Put(%s): %v", key, err)
	}
	idx.Reindex([]byte(key), val)
}

// recCandidates lists the keys the index still offers for rc == v.
func recCandidates(t *testing.T, idx *kvindex.Set, v int64) []string {
	t.Helper()
	d, ok := idx.Lookup(recIndexName)
	if !ok {
		t.Fatalf("index %q is not installed", recIndexName)
	}
	keys, err := idx.Candidates(kvindex.Selector{
		Def:    d,
		Op:     vtypes.FilterEq,
		Values: []vtypes.Value{vtypes.NewInt(v)},
	}, nil, false, 1<<20)
	if err != nil {
		t.Fatalf("Candidates: %v", err)
	}
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, string(k))
	}
	sort.Strings(out)
	return out
}

// TestReconcileDropsDangling: with the removal hook detached, deleting keys
// leaves their postings behind — exactly the residue this pass exists for —
// and a tick removes precisely those postings and no others.
func TestReconcileDropsDangling(t *testing.T) {
	c, idx := recCache(t)
	for i := 0; i < 100; i++ {
		recPut(t, c, idx, fmt.Sprintf("u:%03d", i), 7, 0)
	}
	// Detach the hook: this is what a removal path that never reaches Drop looks
	// like from the index's side (a corrupt slot, a torn page's abandoned slots).
	c.SetOnRemove(nil)

	dead := make(map[string]bool, 20)
	for i := 0; i < 20; i++ {
		k := fmt.Sprintf("u:%03d", i*5)
		if _, err := c.Del([]byte(k)); err != nil {
			t.Fatalf("Del(%s): %v", k, err)
		}
		dead[k] = true
	}
	if got := len(recCandidates(t, idx, 7)); got != 100 {
		t.Fatalf("before reconciling the index offers %d keys, want all 100 (the dangling postings)", got)
	}

	// Budget above the index size, so one tick samples the whole reverse map and
	// the assertion is exact. (Coverage BELOW the budget is probabilistic and is
	// pinned separately, in ops/kvindex.)
	total := ReconcileKVIndex(idx, c, recIndexName, kvindex.ReconcileBudget, nil)
	if total != 20 {
		t.Fatalf("the tick dropped %d postings, want exactly the 20 deleted keys", total)
	}
	got := recCandidates(t, idx, 7)
	if len(got) != 80 {
		t.Fatalf("the index offers %d keys, want 80", len(got))
	}
	for _, k := range got {
		if dead[k] {
			t.Fatalf("deleted key %q is still a candidate", k)
		}
	}
}

// TestReconcileOnExpiredKeysDoesNotDeadlock is the lock-discipline test.
//
// Every probed key is expired on a NON-REPLICATED shard, so each Get runs
// dropExpiredLocked → the shard write lock → onRemove → kvindex.Set.Drop →
// Set.mu. A pass that held the index lock across the probe would deadlock here
// single-threaded, with no concurrency at all. The tick simply has to finish.
func TestReconcileOnExpiredKeysDoesNotDeadlock(t *testing.T) {
	c, idx := recCache(t)
	for i := 0; i < 50; i++ {
		recPut(t, c, idx, fmt.Sprintf("u:%03d", i), 7, time.Millisecond)
	}
	// The hook stays ATTACHED: the point is that the probe re-enters the index
	// through it while the pass is running.
	time.Sleep(20 * time.Millisecond)

	done := make(chan int, 1)
	go func() {
		done <- ReconcileKVIndex(idx, c, recIndexName, kvindex.ReconcileBudget, nil)
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("the reconcile tick did not finish: it is holding the index lock across the liveness probe")
	}
	if got := recCandidates(t, idx, 7); len(got) != 0 {
		t.Fatalf("the index still offers %d expired keys, want 0", len(got))
	}
}

// TestReconcilePlainGetOnExpiredKeyDoesNotDeadlock pins the same cycle from the
// other side, through the plain Cache.Get the probe uses, so the property does
// not depend on which read primitive the pass happens to call.
func TestReconcilePlainGetOnExpiredKeyDoesNotDeadlock(t *testing.T) {
	c, idx := recCache(t)
	recPut(t, c, idx, "u:000", 7, time.Millisecond)
	time.Sleep(20 * time.Millisecond)

	done := make(chan struct{})
	go func() {
		_, _ = c.Get([]byte("u:000"))
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("Cache.Get of an expired key deadlocked against the index")
	}
	if got := recCandidates(t, idx, 7); len(got) != 0 {
		t.Fatalf("the expiring Get left %d postings, want 0", len(got))
	}
}

// TestReconcileDropsAPostingWhoseKeyRereadsAsAnother is the Task 2 ruling.
//
// A torn page's in-place Reset abandons index slots WITHOUT bumping the page
// generation, so those slots decode cleanly as some OTHER key rather than as
// corrupt. Liveness therefore cannot be inferred from slot state; it has to be
// a re-read of THIS key. Here the deleted key's slot is taken over by a
// different one: the dangling posting must go, and the new key's must not.
func TestReconcileDropsAPostingWhoseKeyRereadsAsAnother(t *testing.T) {
	c, idx := recCache(t)
	recPut(t, c, idx, "u:aaa", 7, 0)
	c.SetOnRemove(nil)
	if _, err := c.Del([]byte("u:aaa")); err != nil {
		t.Fatalf("Del: %v", err)
	}
	c.SetOnRemove(idx.Drop)
	recPut(t, c, idx, "u:bbb", 7, 0)

	if got := recCandidates(t, idx, 7); len(got) != 2 {
		t.Fatalf("before reconciling the index offers %v, want both keys", got)
	}
	n := ReconcileKVIndex(idx, c, recIndexName, kvindex.ReconcileBudget, nil)
	if n != 1 {
		t.Fatalf("the tick dropped %d postings, want 1", n)
	}
	if got := recCandidates(t, idx, 7); len(got) != 1 || got[0] != "u:bbb" {
		t.Fatalf("the index offers %v, want only the live key [u:bbb]", got)
	}
}

// TestReconcileTickStopsPromptly: a tick asked to stop drops nothing and
// returns, so Close is bounded by the abort check rather than by the batch.
func TestReconcileTickStopsPromptly(t *testing.T) {
	c, idx := recCache(t)
	for i := 0; i < 100; i++ {
		recPut(t, c, idx, fmt.Sprintf("u:%03d", i), 7, 0)
	}
	c.SetOnRemove(nil)
	for i := 0; i < 100; i++ {
		if _, err := c.Del([]byte(fmt.Sprintf("u:%03d", i))); err != nil {
			t.Fatalf("Del: %v", err)
		}
	}
	stop := make(chan struct{})
	close(stop)
	if n := ReconcileKVIndex(idx, c, recIndexName, kvindex.ReconcileBudget, stop); n != 0 {
		t.Fatalf("a stopped tick dropped %d postings, want 0", n)
	}
	if got := len(recCandidates(t, idx, 7)); got != 100 {
		t.Fatalf("a stopped tick dropped postings: %d keys left, want all 100", got)
	}
}

// TestReconcileNilArgs: the tick is a no-op on a dispatcher built without an
// index, rather than a panic on a nil Set.
func TestReconcileNilArgs(t *testing.T) {
	c, idx := recCache(t)
	if n := ReconcileKVIndex(nil, c, recIndexName, 10, nil); n != 0 {
		t.Fatalf("nil index dropped %d, want 0", n)
	}
	if n := ReconcileKVIndex(idx, nil, recIndexName, 10, nil); n != 0 {
		t.Fatalf("nil cache dropped %d, want 0", n)
	}
}
