// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
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

	// kvIdx is the KV record index this dispatcher maintains, or nil when the
	// dispatcher was built without one (NewTxContext / NewTxContextWithVectors:
	// the legacy KV-only call sites and most tests).
	//
	// ############ ONE Set PER cache.Cache, SHARED BY EVERY TxContext ##########
	//
	// A cache has exactly one index, created next to it (ops.NewKVIndexFor) and
	// handed to every TxContext built over that cache — the FSM's, which
	// maintains it on the write path, AND the store's read-only one, which
	// answers kv_query from it. Two Sets over one cache is not merely wasteful,
	// it is silently wrong: reads would be served from a Set no write ever
	// reached, so every query would answer empty. See shard.New / NewDirect.
	//
	// It is never a durability unit: nothing here is snapshotted, logged or
	// replicated. Reindex is a pure function of (key, value, definitions) with
	// no clock in it, so maintaining it inside a replicated apply cannot
	// diverge replicas, and a resolve failure means "no posting" and nothing
	// else — see reindexKV.
	kvIdx *kvindex.Set

	// walker is the full-keyspace walk a scan-mode kv_query uses, or nil for
	// the plain, ungated walk of this dispatcher's own cache. See SetWalker.
	walker kvindex.Walker

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

// NewTxContextWithIndex constructs a TxContext that also maintains a KV record
// index. It is the dispatcher constructor for a store that has one: shard.New
// builds a single Set next to the cache and passes the SAME pointer here for
// the FSM's TxContext and for the store's read-only one.
func NewTxContextWithIndex(c *cache.Cache, v *vector.CollectionStore, idx *kvindex.Set) *TxContext {
	return &TxContext{c: c, vectors: v, kvIdx: idx}
}

// SetWalker installs the full-keyspace walk a scan-mode kv_query on this
// dispatcher uses. It is called once at wiring time, not per call.
//
// WHY THIS IS A SEAM AND NOT JUST CacheWalker. A walk aliases a LIVE mmap, so a
// walk still running when its shard is removed reads unmapped memory. The index
// observer already closes that hole for BACKFILL walks by registering each one
// and draining it before Store.Close (cluster.Node.beginKVIndexWalk). A scan
// page walks the same keyspace for as long, and CacheWalker can never fail, so
// without this seam a scan has exactly the exposure the drain was built to
// remove — and ops.ErrKVQueryUnavailable would be unreachable.
//
// The store/cluster layer installs a GATED walker here: one that registers the
// walk with the same gate and returns kvindex.ErrWalkAborted when the shard is
// going away. The leaf surfaces any walk error as the retryable
// ErrKVQueryUnavailable, so the abort reaches the client as "retry", never as a
// short page.
//
// Unset, the dispatcher walks its own cache directly, which is right for Direct
// and for every embedder that has no shard lifecycle to race.
func (tx *TxContext) SetWalker(w kvindex.Walker) { tx.walker = w }

// Walker returns the walk installed by SetWalker, or the plain walk of this
// dispatcher's cache.
func (tx *TxContext) Walker() kvindex.Walker {
	if tx.walker != nil {
		return tx.walker
	}
	return CacheWalker(tx.c)
}

// KVIndex returns the KV record index, or nil when the dispatcher was built
// without one. Callers must handle nil: the KV builtins work identically with
// and without an index, and every embedder that predates it has none.
func (tx *TxContext) KVIndex() *kvindex.Set { return tx.kvIdx }

// kvIndexResolverCache is how many decoded record schemas one index's resolver
// keeps. A shard sees few distinct schemas and reuses them on every write, so
// the cache turns a schema-mode resolve into offset arithmetic.
const kvIndexResolverCache = 1024

// kvIndexWalkBatch is how many index slots a rebuild's chunked walk holds a
// cache shard's read lock for before releasing it. It matches the cache's own
// sweep batch: large enough that the per-chunk bookkeeping disappears, small
// enough that a writer never waits for a whole shard's walk.
const kvIndexWalkBatch = 4096

// NewKVIndexFor creates THE record index for c and installs its Drop as c's
// onRemove hook.
//
// Call it exactly once per cache.Cache, next to cache.New, and share the
// returned pointer with every TxContext built over that cache. Both halves
// matter: a second Set would be maintained by only one of the two dispatchers
// (see TxContext.kvIdx), and a second SetOnRemove would REPLACE the first —
// cache.SetOnRemove stores one hook, so the earlier index would stop hearing
// about removals and would keep postings for keys that no longer exist.
//
// The hook runs under a cache shard's write lock, which is why Drop is the
// hook and not a closure that reads the cache: lock order is cache → index,
// one way, always.
func NewKVIndexFor(c *cache.Cache) *kvindex.Set {
	idx := kvindex.New(kvIndexResolverCache)
	c.SetOnRemove(idx.Drop)
	return idx
}

// CacheWalker returns c's chunked full-keyspace walk in the shape
// kvindex.Rebuild and kvindex.Backfill take.
//
// It always completes, so it always reports nil. The error in the Walker
// signature is for a walker that can be ABORTED — the cluster observer wraps
// this one so a shard being removed can stop the walk and wait for it (see
// cluster.Node.beginKVIndexWalk), and an aborted walk must not be mistaken for a
// finished one.
func CacheWalker(c *cache.Cache) kvindex.Walker {
	return func(fn func(key, value []byte) bool) error {
		c.IterateChunked(kvIndexWalkBatch, fn)
		return nil
	}
}

// RebuildKVIndex refills idx from c's live entries: the warm-start and
// post-Restore rebuild point, where the cache has content the index has never
// seen (the index is not snapshotted, not logged, and not replicated — it is
// always re-derived by walking the cache).
//
// With NO definition installed it walks NOTHING. That is not just an
// optimisation: a rebuild over an empty definition set has no result to
// publish, so paying for a full-keyspace walk at every start — which is the
// common case, since definitions arrive later from the meta log — would be
// pure cost. Installing a definition later starts its own backfill.
//
// idx or c being nil is a no-op, so a store built without an index can call it
// unconditionally.
//
// It returns the walk's error, in which case NOTHING was marked ready — the
// index stays building and queries get a retryable refusal rather than a
// silently short answer. CacheWalker never aborts, so this is nil today; the
// path exists for the cluster observer's abortable wrapper.
func RebuildKVIndex(idx *kvindex.Set, c *cache.Cache) error {
	if idx == nil || c == nil || len(idx.Defs()) == 0 {
		return nil
	}
	return idx.Rebuild(CacheWalker(c))
}

// reindexKV is the single seam every KV write handler calls, with the bytes it
// just stored, AFTER the store returns.
//
// AFTER, never before: an evicting Cache.Put can fire onRemove for the very key
// being written (its old copy sits on the page the write retires), so a posting
// made before the Put would be dropped by that eviction and the fresh value
// would be unfindable. See cache.SetOnRemove.
//
// It is a no-op when the dispatcher was built without an index, never returns
// an error, and never touches the cache — an index failure must not change what
// the apply stored. That is what lets definitions be installed node-locally:
// the index is derived state, so a node whose index disagrees answers fewer
// candidates, never a different keyspace.
func (tx *TxContext) reindexKV(key, value []byte) {
	if tx.kvIdx == nil {
		return
	}
	tx.kvIdx.Reindex(key, value)
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
func (tx *TxContext) GetWithExpiry(key []byte) (val []byte, expiryMs uint64, err error) {
	if tx.applyStamped {
		return tx.c.GetWithExpiryAt(key, tx.applyNowMs)
	}
	return tx.c.GetWithExpiry(key)
}

// GetWithExpiryInto is GetWithExpiry appending into dst, so a handler that
// reuses one buffer pays no allocation per hit. It honours the apply stamp
// exactly as GetWithExpiry does.
//
// Unlike GetWithExpiry the result does NOT alias the page backing store: the
// value is copied into dst, so it is a caller-owned slice that may be mutated
// and retained.
func (tx *TxContext) GetWithExpiryInto(dst, key []byte) (val []byte, expiryMs uint64, err error) {
	if tx.applyStamped {
		return tx.c.GetWithExpiryIntoAt(dst, key, tx.applyNowMs)
	}
	return tx.c.GetWithExpiryInto(dst, key)
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
//
// IT REINDEXES, even though it stores the SAME BYTES, and that is not
// redundant — it is the case the obvious rule gets wrong. Expire goes through
// the cache's PUT body, so it can evict the page framing this key's own current
// copy and fire onRemove(key) → Set.Drop from inside the write; every posting
// for the key is dropped mid-write while the key itself stays live. Nothing
// about an unchanged value looks like a change, so without the re-post expire
// would silently turn a findable key into an unfindable one — the missing-row
// class. Hence the rule: EVERY path that writes through the cache's put body
// reindexes after it returns, whether or not the bytes moved. The re-post is
// idempotent in (key, value), so it costs a resolve and changes nothing else.
//
// This is also the seam for expire, caex and the WASM cache_expire host
// function, all of which route here and so need no reindex of their own.
func (tx *TxContext) Expire(key []byte, ttl time.Duration) error {
	v, err := tx.Get(key)
	if err != nil {
		return err
	}
	// Copy the value because Get aliases into the page; cache.Put may
	// overwrite that page region during the Put call itself.
	buf := make([]byte, len(v))
	copy(buf, v)
	if err := tx.Put(key, buf, ttl); err != nil {
		return err
	}
	tx.reindexKV(key, buf)
	return nil
}

// PutIndexed is Put followed by the index maintenance every KV write owes. It
// is the exported seam for writers OUTSIDE this package that store bytes under
// a key; the WASM cache_put host function is the one such writer today.
//
// Prefer it to Put plus a hand-written reindex, because the ordering is a
// correctness requirement rather than a style choice: an evicting Put can fire
// onRemove for the very key being written, so a posting made first is dropped
// by the write that follows. Callers inside this package use the unexported
// seam directly.
func (tx *TxContext) PutIndexed(key, value []byte, ttl time.Duration) error {
	if err := tx.Put(key, value, ttl); err != nil {
		return err
	}
	tx.reindexKV(key, value)
	return nil
}
