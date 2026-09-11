// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// TxContext is the per-call handle passed to op handlers. It wraps a cache
// reference and exposes Get/Put/Del/Expire convenience methods so handlers
// don't need to import cache directly.
//
// Within a single FSM.Apply call, the underlying cache shard lock is held
// for the duration of the handler — so multiple TxContext operations against
// the same shard execute atomically from external observers' perspective.
type TxContext struct {
	c       *cache.Cache
	vectors *vector.CollectionStore

	// applyStamped and applyNowMs carry the leader-stamped apply clock for the
	// CURRENT replicated apply, threaded in by fsm.applyEntryData from the log
	// entry and cleared after the handler returns (#4 Phase B / B1).
	//
	// applyStamped — NOT applyNowMs != 0 — is the AUTHORITATIVE signal to use the
	// cache's At-variants (GetAt/PutAt), so the choice of clock is driven by the
	// ENTRY FORMAT, never by the stamp's value. A stamped entry whose leader clock
	// is legitimately 0 must still take the At-path (every replica deterministically
	// uses 0 — identical, no divergence); keying off applyNowMs != 0 would route
	// that entry back to each node's wall clock and silently diverge them. When
	// applyStamped is false — the Direct/single-node path, the read-only Call path,
	// and legacy (unstamped) entries — the wrappers use the wall-clock Get/Put, so
	// those paths are byte-for-byte unchanged.
	applyStamped bool
	applyNowMs   uint64

	// shardIdx is the index of the shard GROUP whose dispatcher owns this
	// TxContext. It is set once when the dispatcher is built (one TxContext per
	// shard.Store / per FSM) and never varies per entry, unlike applyStamped.
	//
	// It exists because a handler that mutates NODE-WIDE state needs to know
	// which group's log the entry it is applying came from. The one such handler
	// is __register_wasm__: the ops registry it installs into is node-wide, but
	// the safety argument for invoking a dynamically registered op is per-GROUP
	// ("group g's log carries the registration"), so the cluster layer has to
	// attribute each apply to its group. See cluster.checkWASMRouteGate.
	shardIdx int
}

// NoShardIndex is what ShardIndex reports when there is no dispatcher behind the
// TxContext (a nil receiver). Handlers that attribute node-wide state changes to
// a shard group must treat it as "no group provenance" and record nothing —
// never as group 0, which would be a false attribution.
//
// Defined in the leaf (ops/wire) because wire's WASMNotResidentError carries a
// Group that is NoShardIndex for a node-wide (no group provenance) resolution;
// aliased here so every existing ops.NoShardIndex call site is unaffected.
const NoShardIndex = wire.NoShardIndex

// SetApplyStamp sets the leader-stamped apply clock and stamped-ness for the next
// handler run (stamped=false, nowMs=0 to clear). Only fsm.applyEntryData calls it,
// immediately before dispatching a replicated entry's handler; it is not part of
// the op-handler API.
func (tx *TxContext) SetApplyStamp(nowMs uint64, stamped bool) {
	tx.applyNowMs = nowMs
	tx.applyStamped = stamped
}

// applyStamp returns the leader apply stamp (unix millis) and whether the current
// apply is stamped. Vector handlers use it to route a replicated write through the
// deterministic ...At engine variant (which judges every TTL deadline computation
// and liveness check against the stamp) or, when unstamped, the wall-clock path.
// The bool — not stampMs != 0 — is authoritative (a stamped 0 is still
// deterministic), mirroring TxContext.Get/Put.
func (tx *TxContext) applyStamp() (int64, bool) {
	return int64(tx.applyNowMs), tx.applyStamped //nolint:gosec // unix-millis fits int64
}

// SetShardIndex records the shard group this TxContext's dispatcher serves. It
// is called once at dispatcher construction (shard.New), not per entry.
func (tx *TxContext) SetShardIndex(idx int) { tx.shardIdx = idx }

// ShardIndex reports the shard group this TxContext's dispatcher serves, or
// NoShardIndex when there is no dispatcher (nil receiver — reachable only from
// tests that invoke a handler directly).
func (tx *TxContext) ShardIndex() int {
	if tx == nil {
		return NoShardIndex
	}
	return tx.shardIdx
}

// NewTxContext constructs a TxContext bound to a cache.
func NewTxContext(c *cache.Cache) *TxContext {
	return &TxContext{c: c}
}

// NewTxContextWithVectors constructs a TxContext that has both KV cache
// and vector CollectionStore reachable. Vector op handlers require this
// variant; legacy KV-only call sites use NewTxContext.
func NewTxContextWithVectors(c *cache.Cache, v *vector.CollectionStore) *TxContext {
	return &TxContext{c: c, vectors: v}
}

// Vectors returns the CollectionStore (nil if the dispatcher wasn't built
// with one).
func (tx *TxContext) Vectors() *vector.CollectionStore { return tx.vectors }

// Cache returns the underlying cache. Callers should prefer the wrapper
// methods; this escape hatch exists for handlers that need iteration or
// stats during their op.
func (tx *TxContext) Cache() *cache.Cache {
	return tx.c
}

// Get returns the value for key. Returns cache.ErrNotFound if absent or
// expired. The returned slice aliases into the page backing store; copy if
// you need to retain it past the handler's return.
//
// Under a replicated apply STAMP (applyStamped) expiry is judged against the
// leader-stamped clock via GetAt, so every replica agrees on what is live and
// tombstones the same expired keys; otherwise it uses the wall-clock Get. The
// branch is on applyStamped, not applyNowMs != 0, so a stamp of 0 still uses the
// deterministic At-path (see the field doc).
func (tx *TxContext) Get(key []byte) ([]byte, error) {
	if tx.applyStamped {
		return tx.c.GetAt(key, tx.applyNowMs)
	}
	return tx.c.Get(key)
}

// Put inserts or replaces the entry for key. Under a replicated apply stamp the
// absolute expiry is computed as stamp + ttl via PutAt (identical across
// replicas); otherwise the wall-clock Put is used. Branches on applyStamped so a
// stamp of 0 is still deterministic (see the field doc).
func (tx *TxContext) Put(key, value []byte, ttl time.Duration) error {
	if tx.applyStamped {
		return tx.c.PutAt(key, value, ttl, tx.applyNowMs)
	}
	return tx.c.Put(key, value, ttl)
}

// GetWithExpiry is Get that ALSO returns the entry's stored absolute expiry
// (ms since epoch; 0 = no expiry). Handlers use it to inspect a key's deadline
// (ttl / persist) or preserve it across a rewrite (incr_ex). Like Get, under a
// replicated apply stamp liveness is judged against the leader-stamped clock via
// GetWithExpiryAt so every replica agrees on what is live; otherwise it uses the
// wall-clock GetWithExpiry. Branches on applyStamped, not applyNowMs != 0, so a
// stamp of 0 is still deterministic (see the field doc). The returned value
// slice aliases the page backing store; copy if you need to retain it.
// GetWithExpiryInto is GetWithExpiry appending into dst, so a handler that
// reuses one buffer pays no allocation per hit. It honours the apply stamp
// exactly as GetWithExpiry does.
func (tx *TxContext) GetWithExpiryInto(dst, key []byte) (val []byte, expiryMs uint64, err error) {
	if tx.applyStamped {
		return tx.c.GetWithExpiryIntoAt(dst, key, tx.applyNowMs)
	}
	return tx.c.GetWithExpiryInto(dst, key)
}

func (tx *TxContext) GetWithExpiry(key []byte) (val []byte, expiryMs uint64, err error) {
	if tx.applyStamped {
		return tx.c.GetWithExpiryAt(key, tx.applyNowMs)
	}
	return tx.c.GetWithExpiry(key)
}

// PutAbs inserts or replaces the entry for key with a pre-computed ABSOLUTE
// expiry (ms since epoch; 0 = no expiry), bypassing any TTL→now+ttl conversion.
// It is the primitive for REPLACING a value while preserving an existing
// deadline: the caller reads the stored expiry with GetWithExpiry and writes it
// back verbatim (incr_ex on an existing key), or clears it with 0 (persist).
//
// It is deterministic across replicas because the absolute expiry is supplied by
// the caller and is identical on every apply — the committed state the read
// returned is the same everywhere — so unlike Put it needs no apply-stamp branch.
func (tx *TxContext) PutAbs(key, value []byte, expiryMs uint64) error {
	return tx.c.PutAbs(key, value, expiryMs)
}

// Del removes the entry for key. Returns true if the entry existed. Del carries
// no TTL, so it is clock-independent and needs no At-variant.
//
// The error is cache.ErrFull when a persistent shard has no room to append the
// delete's durable tombstone record. On the replicated apply path that is
// classified as a NON-DETERMINISTIC (fatal) error — occupancy is per-node state —
// so it halts rather than advancing the applied index over a delete that did not
// persist. See cache/shard.go delH.
func (tx *TxContext) Del(key []byte) (bool, error) {
	return tx.c.Del(key)
}

// Expire updates the TTL of an existing entry. Returns cache.ErrNotFound if
// the key is absent. Equivalent to Get + Put with the same value but a new
// TTL — so under a replicated apply stamp both halves use the stamped clock
// (GetAt to test liveness, PutAt to stamp the new absolute expiry), keeping the
// re-stamped expiry identical on every replica.
func (tx *TxContext) Expire(key []byte, ttl time.Duration) error {
	v, err := tx.Get(key)
	if err != nil {
		return err
	}
	// Copy the value because Get aliases into the page; cache.Put may
	// overwrite that page region during the Put call itself.
	buf := make([]byte, len(v))
	copy(buf, v)
	return tx.Put(key, buf, ttl)
}
