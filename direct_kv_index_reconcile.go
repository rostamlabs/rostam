// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"log/slog"
	"math"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
)

// The bounded KV index reconcile pass for a Direct store.
//
// A directStore is the one deployment shape that reaches neither of the other
// two hosts for this pass: it builds no shard.Store, so shard's per-store ticker
// never exists for it, and it has no cluster.Node either. It holds a bare
// cache.Cache and one kvindex.Set — exactly what ops.ReconcileKVIndex takes —
// so it runs its own copy of the same tick, on the same default interval, with
// the same Close fence.
//
// THE FENCE IS THE REASON THIS IS NOT JUST A GOROUTINE. The probe reads the
// cache and directStore.Close unmaps it, so Close must stop AND JOIN the
// ticker before it reaches d.cache.Close(). See shard/kv_index_reconcile.go,
// which makes the identical argument one layer up.

// directReconcileInterval turns the config knob into a duration, following the
// convention DirectConfig already uses for TTLSweepIntervalMs: 0 means the
// default, negative disables, positive is the interval in milliseconds.
//
// THE MULTIPLICATION IS CLAMPED. time.Duration is int64 NANOSECONDS, so a
// setting above maxReconcileIntervalMs — about 9.2e12 ms, or roughly 292 years,
// so only an absurd one — overflows it. The wrap is not a large interval but an
// arbitrary one, including negative values, which this function's own contract
// reads as "disabled": a misconfigured knob would silently turn the reconciler
// OFF rather than set it slowly. Clamping keeps an absurd setting absurd instead
// of letting it change the meaning of the field.
func directReconcileInterval(ms int) time.Duration {
	switch {
	case ms < 0:
		return 0 // disabled
	case ms == 0:
		return defaultKVIndexReconcileInterval
	case int64(ms) > maxReconcileIntervalMs:
		return time.Duration(math.MaxInt64)
	default:
		return time.Duration(ms) * time.Millisecond
	}
}

// maxReconcileIntervalMs is the largest millisecond count that still fits a
// time.Duration. Shared spelling with shard.Store's ticker so both hosts of this
// pass clamp identically.
const maxReconcileIntervalMs = int64(math.MaxInt64) / int64(time.Millisecond)

// defaultKVIndexReconcileInterval matches shard.Config's default, so a Direct
// deployment and an Embedded one reconcile on the same cadence. It is spelled
// out here rather than imported because shard's copy is unexported and a
// directStore has no shard.Config to read.
const defaultKVIndexReconcileInterval = 60 * time.Second

// startKVIndexReconciler starts the ticker unless the interval disables it.
// Called from NewDirect on the fully built store, so the goroutine-start
// happens-before edge publishes every field it reads.
func (d *directStore) startKVIndexReconciler(interval time.Duration) {
	idx := d.tx.KVIndex()
	if interval <= 0 || idx == nil {
		return
	}
	d.kvReconcileStop = make(chan struct{})
	stop := d.kvReconcileStop
	d.kvReconcileWg.Add(1)
	go func() {
		defer d.kvReconcileWg.Done()
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				d.kvIndexReconcileTick(stop)
			}
		}
	}()
}

// stopKVIndexReconciler signals the ticker and WAITS for it, so past this call
// no goroutine is inside the cache and Close may unmap it. Idempotent, and safe
// on a store whose ticker never started.
func (d *directStore) stopKVIndexReconciler() {
	d.kvReconcileOnce.Do(func() {
		if d.kvReconcileStop != nil {
			close(d.kvReconcileStop)
		}
	})
	d.kvReconcileWg.Wait()
}

// kvIndexReconcileTick reconciles ONE definition, taking the next in rotation,
// and skips one that is still building. Same shape as the shard ticker's.
func (d *directStore) kvIndexReconcileTick(stop <-chan struct{}) {
	idx := d.tx.KVIndex()
	if idx == nil {
		return
	}
	defs := idx.Defs()
	if len(defs) == 0 {
		return
	}
	i := int((d.kvReconcileNext.Add(1) - 1) % uint64(len(defs))) //nolint:gosec // bounded by len(defs)
	def := defs[i]
	if !idx.IsReady(def.Name) {
		return
	}
	dropped := ops.ReconcileKVIndex(idx, d.cache, def.Name, kvindex.ReconcileBudget, stop)
	if dropped > 0 {
		slog.Info("kv index reconcile dropped dangling postings",
			"component", "direct", "index", def.Name, "dropped", dropped)
	}
}
