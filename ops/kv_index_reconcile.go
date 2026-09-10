// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
)

// kvReconcileStopCheckEvery is how often a tick looks at its stop signal, in
// probed keys. A non-blocking channel receive is cheap but not free and the
// loop is otherwise one map lookup per key, so checking on a power-of-two
// stride keeps it off the probe path while still bounding how long a Close
// waits to a few hundred lookups. Same shape and same reason as the index
// walk's kvIndexAbortCheckEvery.
const kvReconcileStopCheckEvery = 256

// ReconcileKVIndex runs ONE bounded reconcile tick for one index definition:
// sample a batch of posted keys, probe each one's liveness against the live
// cache, and drop the postings whose key is gone. It reports how many postings
// went.
//
// The batch is a truncated range over the reverse map, so which keys a tick
// examines is randomised and coverage across ticks is probabilistic. See
// ops/kvindex/reconcile.go for why exact coverage was traded away.
//
// THIS IS THE FUNCTION THE THREE-PHASE SPLIT EXISTS FOR, and the shape of the
// loop below is the whole of it: NO INDEX LOCK IS HELD ACROSS c.Get. Cache.Get
// of an expired key on a non-replicated shard runs dropExpiredLocked, which
// takes the cache shard's write lock and fires onRemove → kvindex.Set.Drop →
// Set.mu; a pass holding Set.mu across the probe self-deadlocks
// single-threaded, no concurrency required. See ops/kvindex/reconcile.go and
// cache/onremove.go, which state the same lock order from their own side.
//
// LIVENESS IS THE KEY RE-READ ITSELF. indexTable.get compares the stored key
// with bytes.Equal before reporting a hit, so a Get that returns proves the
// live slot holds THIS key and a Get that misses proves it does not. That is
// the only sound test: a torn page's abandoned slots decode cleanly as some
// OTHER key, so nothing about slot state can stand in for it.
//
// It judges liveness on the WALL CLOCK, deliberately. The pass only ever drops
// postings, and a posting is a hint that costs one wasted lookup, so judging a
// borderline key generously or strictly costs at most one stale posting for one
// more interval — never a row.
//
// The tick abandons quietly when stop is closed: it ends the batch (which
// clears the marks it set) and drops NOTHING, because a half-probed batch is
// not evidence about the keys it never reached.
func ReconcileKVIndex(idx *kvindex.Set, c *cache.Cache, name string, budget int, stop <-chan struct{}) (dropped int) {
	if idx == nil || c == nil {
		return 0
	}
	keys := idx.ReconcileBatch(name, budget)
	if len(keys) == 0 {
		// Still end the tick: ReconcileBatch may have marked nothing, but calling
		// DropReconciled is what guarantees no marks are left standing.
		idx.DropReconciled(name, nil)
		return 0
	}

	var dead [][]byte
	for i, k := range keys {
		if i%kvReconcileStopCheckEvery == 0 && kvReconcileStopped(stop) {
			idx.DropReconciled(name, nil)
			return 0
		}
		// The value is discarded — only the hit/miss matters. It is not retained
		// past this call either, which is what makes the zero-copy alias a
		// reject-writes cache hands back safe here.
		if _, err := c.Get(k); errors.Is(err, cache.ErrNotFound) {
			dead = append(dead, k)
		}
	}
	return idx.DropReconciled(name, dead)
}

// kvReconcileStopped reports whether the tick has been asked to abandon. A nil
// channel never fires, which is what a caller with no lifecycle to respect
// (a test, a one-shot pass) passes.
func kvReconcileStopped(stop <-chan struct{}) bool {
	if stop == nil {
		return false
	}
	select {
	case <-stop:
		return true
	default:
		return false
	}
}
