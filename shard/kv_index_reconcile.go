// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"log/slog"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
)

// defaultKVIndexReconcileIntervalMs is how often a store reconciles one of its
// index definitions. A minute is deliberately slow: the pass is a MEMORY bound
// on a residue that only a corrupt slot, an abandoned torn-page slot, or a
// rebuild straddling a flush can produce, and every dangling posting it has not
// got to yet costs one wasted lookup on a query that happens to select it.
const defaultKVIndexReconcileIntervalMs = 60_000

// WHY THE TICKER LIVES ON THE STORE AND NOT ON THE CLUSTER NODE.
//
// Three things decide it, and they all point the same way.
//
//  1. The probe needs the CACHE. Liveness is a re-read of the key through the
//     same cache the postings describe, and that is a per-store object. A
//     node-level goroutine would be reaching back into each store to find one.
//
//  2. A SINGLE-NODE Direct store has no cluster Node at all. rostam.NewEmbedded
//     builds shard.Stores directly, so a pass hosted in cluster would never run
//     for the whole single-node deployment — the one where nothing else is
//     going to notice the index growing either.
//
//  3. CLOSE IS THE FENCE, and the store already owns it. The probe touches the
//     live mmap and cache.Close unmaps it, which is the same hazard the index
//     WALK has (cluster.kvIndexWalkGate) and the same one in-flight Calls have
//     (Store.drainCalls). Hosting the ticker here lets Close stop and JOIN it
//     directly, before anything is unmapped — strictly tighter than the
//     per-group gate, which drains before Store.Close and would still leave a
//     node-hosted ticker to be stopped separately. RemoveShardOwner calls
//     Store.Close, so a group leaving the node stops its reconciler by the same
//     path, and no walk-gate registration is needed for this pass.
//
// The cluster layer keeps only the counter: Stats().KVIndex.ReconcileDrops sums
// kvindex.Set.ReconcileDrops over the hosted groups, exactly as it already does
// for VerifyMisses, and RemoveShardOwner folds a departing group's count into
// the node total so the contracted uint64 never decreases.

// startKVIndexReconciler starts this store's reconcile ticker, unless the
// interval disables it. Called from New, before the store is returned, so the
// goroutine-start happens-before edge publishes the fields it reads.
func (s *Store) startKVIndexReconciler() {
	if s.cfg.KVIndexReconcileIntervalMs <= 0 || s.kvIdx == nil {
		return
	}
	interval := time.Duration(s.cfg.KVIndexReconcileIntervalMs) * time.Millisecond
	s.kvReconcileStop = make(chan struct{})
	stop := s.kvReconcileStop
	s.kvReconcileWg.Add(1)
	go func() {
		defer s.kvReconcileWg.Done()
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				s.kvIndexReconcileTick(stop)
			}
		}
	}()
}

// stopKVIndexReconciler signals the ticker and WAITS for it, which is the whole
// point: past this call no goroutine is inside the cache on this store's behalf,
// so Close may unmap it. Idempotent, and safe on a store whose ticker never
// started.
//
// The wait is bounded by the tick's own abort check (ops.ReconcileKVIndex looks
// at the stop signal every few hundred probes) plus one batch selection, not by
// the size of the keyspace.
func (s *Store) stopKVIndexReconciler() {
	s.kvReconcileOnce.Do(func() {
		if s.kvReconcileStop != nil {
			close(s.kvReconcileStop)
		}
	})
	s.kvReconcileWg.Wait()
}

// kvIndexReconcileTick reconciles ONE definition, taking the next one in
// rotation each time.
//
// One per tick, not all of them: a tick's cost is then bounded by one budget
// however many definitions a deployment has, and the rotation is what stops the
// first definition's work from starving the rest. A definition still BUILDING
// is skipped — its posting set is a proper subset being refilled by a walk, so
// nothing in it can be called dangling yet.
func (s *Store) kvIndexReconcileTick(stop <-chan struct{}) {
	if s.kvIdx == nil {
		return
	}
	defs := s.kvIdx.Defs()
	if len(defs) == 0 {
		return
	}
	// Modulo a length that can change between ticks: a definition added or
	// removed shifts the rotation, which costs at most one repeated or skipped
	// definition for one interval on a pass that only ever drops.
	i := int((s.kvReconcileNext.Add(1) - 1) % uint64(len(defs))) //nolint:gosec // bounded by len(defs)
	d := defs[i]
	if !s.kvIdx.IsReady(d.Name) {
		return
	}
	dropped, _ := ops.ReconcileKVIndex(s.kvIdx, s.cache, d.Name, kvindex.ReconcileBudget, stop)
	if dropped > 0 {
		slog.Info("kv index reconcile dropped dangling postings",
			"component", "shard", "shard", s.cfg.ShardIndex, "index", d.Name, "dropped", dropped)
	}
}
