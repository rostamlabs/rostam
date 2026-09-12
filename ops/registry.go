// SPDX-License-Identifier: Apache-2.0

// Package ops provides a registry for user-supplied stored procedures
// (ops) that run inside a shard's FSM Apply path. Read-only ops bypass
// the Raft log; write ops are encoded into log entries and applied on
// every replica deterministically.
package ops

import (
	"errors"
	"fmt"
	"sync"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// OpKind, KeyExtractor, KeyExtractorInto, and RouteLayout are aliases onto the
// leaf (ops/wire) definitions. The leaf owns these as plain value/function types
// with no dependency on TxContext or cache, so the client's routing-only
// wire.Registry and the server's handler-carrying ops.Registry (below) can share
// one definition of "what an op's routing key looks like" without either
// package needing to import the other's Registry type.
type (
	OpKind           = wire.OpKind
	KeyExtractor     = wire.KeyExtractor
	KeyExtractorInto = wire.KeyExtractorInto
	RouteLayout      = wire.RouteLayout
)

const (
	// OpReadOnly ops execute directly against the cache; they must not mutate state.
	OpReadOnly = wire.OpReadOnly
	// OpReadWrite ops are serialised as Raft log entries and applied by the FSM.
	OpReadWrite = wire.OpReadWrite

	// RouteLayoutNone: route via KeyExtractor (or not at all, if that is nil too).
	RouteLayoutNone = wire.RouteLayoutNone
	// RouteLayoutColAt1: [colLen:u8][col]... — the collection name at offset 0.
	RouteLayoutColAt1 = wire.RouteLayoutColAt1
	// RouteLayoutColAt2: [flags:u8][colLen:u8][col]... — the name at offset 1.
	RouteLayoutColAt2 = wire.RouteLayoutColAt2
)

// Handler is the function signature for a registered op. It stays in ops
// (rather than moving to the leaf with OpKind/KeyExtractor/RouteLayout) because
// TxContext wraps *cache.Cache — a server-only engine dependency the leaf must
// not import.
type Handler func(tx *TxContext, args []byte) ([]byte, error)

// AppendHandler is the allocation-free twin of Handler: instead of returning a
// freshly allocated payload it APPENDS the payload to dst and returns the
// extended slice, so a transport holding a reusable buffer can serve a call
// without allocating a reply at all.
//
// It is OPTIONAL and per-op. An op without one is served by its Handler exactly
// as before, so this costs nothing for the 90-odd ops that do not have it and
// changes no existing behaviour.
//
// OWNERSHIP, AND WHY THIS IS NOT THE DEFAULT. The returned bytes alias the
// CALLER's buffer and stay valid only until that caller reuses it. That is safe
// for the TCP server, which copies the payload into its response frame before
// reading the next request on the connection. It is NOT safe for Store.Call,
// which is public API handing bytes to in-process application code that may
// retain them indefinitely — the same hazard that keeps online compaction
// opt-in (see cache/compact_online.go). So Store.Call keeps using Handler, and
// only a transport that can prove the lifetime asks for the append form.
type AppendHandler func(tx *TxContext, args, dst []byte) ([]byte, error)

// ErrDuplicateOp is returned when registering a name that already exists.
var ErrDuplicateOp = errors.New("ops: op name already registered")

// maxOpNameLen mirrors the wire-format limit (uint8-length prefix).
const maxOpNameLen = 255

// entry holds one registered handler.
type entry struct {
	kind OpKind
	fn   Handler
	// appendFn is the optional allocation-free twin of fn (see AppendHandler).
	// nil for every op that has not opted in, which is nearly all of them.
	appendFn AppendHandler
	ke       KeyExtractor // nil = shardless
	// layout is the allocation-free routing twin of ke, set ONLY for built-in
	// routable ops (registered from builtin.go via the shared wire table). RouteKeyInto
	// on this layout extracts the exact same key as ke — same offsets, same
	// canonicalization — and RouteLayoutNone simply means "route through ke", so a
	// dynamically registered (WASM) op keeps the allocating path.
	layout RouteLayout
	// crossShard marks a read-write op whose handler may touch keys beyond its
	// routing key's shard (e.g. a WASM op, whose host functions can read/write
	// arbitrary keys). The routing key (ke) still picks the destination shard in
	// the cluster, but a single-node serializer (the Direct backend) must NOT
	// rely on a per-shard lock for such an op — it must take a global barrier so
	// the multi-key handler stays atomic against every other read-write op.
	// Ops that touch only their routing key (KV builtins, vector ops) leave this
	// false and benefit from per-shard parallelism.
	crossShard bool
}

// Registry is a thread-safe map of op name to handler.
type Registry struct {
	mu sync.RWMutex
	m  map[string]entry
}

// NewRegistry constructs an empty registry.
func NewRegistry() *Registry {
	return &Registry{m: make(map[string]entry, 16)}
}

// Register adds a shardless op (no routing key). Suitable for cluster-
// level ops like __ping__. For ops that should route to a specific
// shard, use RegisterRoutable.
func (r *Registry) Register(name string, kind OpKind, fn Handler) error {
	return r.registerEntry(name, kind, fn, nil, RouteLayoutNone, false)
}

// RegisterRoutable adds an op with an explicit KeyExtractor. The
// cluster layer uses the extractor to pick the destination shard.
// Returns an error if ke is nil — for shardless ops, use Register.
//
// The handler is expected to touch ONLY its routing key's shard (true for the
// KV builtins and vector ops). A single-node serializer may then guard it with
// just that shard's lock. For a routable op whose handler may touch arbitrary
// keys across shards (e.g. a WASM op), use RegisterRoutableCrossShard.
func (r *Registry) RegisterRoutable(name string, kind OpKind, fn Handler, ke KeyExtractor) error {
	if ke == nil {
		return errors.New("ops: RegisterRoutable requires non-nil KeyExtractor; use Register for shardless ops")
	}
	return r.registerEntry(name, kind, fn, ke, RouteLayoutNone, false)
}

// RegisterRoutableCrossShard is like RegisterRoutable but marks the op as
// cross-shard: its routing key still selects the destination shard in the
// cluster, but the handler may read/write keys on OTHER shards too. A single-
// node serializer (the Direct backend) must take a global barrier for such an
// op rather than a per-shard lock, so the multi-key handler stays atomic
// against every other read-write op. WASM ops register through here.
func (r *Registry) RegisterRoutableCrossShard(name string, kind OpKind, fn Handler, ke KeyExtractor) error {
	if ke == nil {
		return errors.New("ops: RegisterRoutableCrossShard requires non-nil KeyExtractor; use Register for shardless ops")
	}
	return r.registerEntry(name, kind, fn, ke, RouteLayoutNone, true)
}

// Replace installs name unconditionally, overwriting any existing entry instead
// of returning ErrDuplicateOp. It exists for exactly one caller: replacing a
// DYNAMICALLY registered WASM op when a strictly newer registration wins the
// (Epoch, fingerprint) comparison (see WASMRegistration.Epoch).
//
// Why plain Register is not enough there: a replacement may change Kind or the
// KeyExtractor, and those live in the registry entry, not in the wasm.Runtime.
// Leaving the loser's entry in place would make two replicas that received the
// same two registrations in different orders route or classify the SAME op
// differently — the divergence the epoch rule exists to prevent.
//
// Do NOT use this to redefine a built-in op: the duplicate check on Register is
// what keeps two subsystems from silently claiming the same name.
func (r *Registry) Replace(name string, kind OpKind, fn Handler, ke KeyExtractor, crossShard bool) error {
	if err := validateEntry(name, kind, fn); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[name] = entry{kind: kind, fn: fn, ke: ke, crossShard: crossShard}
	return nil
}

// CrossShard reports whether the named op was registered as cross-shard (its
// handler may touch keys beyond its routing shard). False for unknown ops.
func (r *Registry) CrossShard(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.m[name].crossShard
}

func (r *Registry) registerEntry(name string, kind OpKind, fn Handler, ke KeyExtractor, layout RouteLayout, crossShard bool) error {
	if err := validateEntry(name, kind, fn); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.m[name]; exists {
		return ErrDuplicateOp
	}
	r.m[name] = entry{kind: kind, fn: fn, ke: ke, layout: layout, crossShard: crossShard}
	return nil
}

// validateEntry holds the name/handler/kind rules shared by registerEntry and
// Replace, so an unconditional install cannot bypass them.
func validateEntry(name string, kind OpKind, fn Handler) error {
	if name == "" {
		return errors.New("ops: empty op name")
	}
	if len(name) > maxOpNameLen {
		return fmt.Errorf("ops: op name length %d exceeds max %d", len(name), maxOpNameLen)
	}
	// Protocol-v2 wire compatibility: an op name of length 2 is ambiguous with
	// the v2 frame's [version=0x02][...] prefix. Reject by convention; existing
	// ops (get/put/del/expire/incr/__ping__/vector_*) all clear this bar.
	if len(name) == 2 {
		return fmt.Errorf("ops: op name %q has length 2 which collides with protocol v2 version byte; use 3+ chars", name)
	}
	if fn == nil {
		return errors.New("ops: nil handler")
	}
	if kind != OpReadOnly && kind != OpReadWrite {
		return fmt.Errorf("ops: invalid OpKind %d", kind)
	}
	return nil
}

// Lookup returns the handler, its kind, its KeyExtractor (nil for
// shardless ops), and whether the op was found. Callers that also need the
// cross-shard flag (to pick a single-node lock strategy) should use LookupEntry,
// which reads it from the SAME entry in one registry acquisition.
func (r *Registry) Lookup(name string) (Handler, OpKind, KeyExtractor, bool) {
	fn, kind, ke, _, ok := r.LookupEntry(name)
	return fn, kind, ke, ok
}

// LookupRouting returns everything the cluster router needs for one op — its kind,
// its KeyExtractor, and its RouteLayout (RouteLayoutNone unless the op is a
// built-in) — from ONE registry entry under ONE RLock. A router should prefer
// RouteKeyInto(layout, ...) and fall back to ke when the layout is None; both yield
// the same key.
// SetAppendVariant attaches an allocation-free twin to an already-registered
// op (see AppendHandler). It is separate from Register so the 90-odd ops that
// do not want one keep their existing registration untouched.
func (r *Registry) SetAppendVariant(name string, fn AppendHandler) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.m[name]
	if !ok {
		return fmt.Errorf("ops: op %q not registered", name)
	}
	e.appendFn = fn
	r.m[name] = e
	return nil
}

// LookupAppend returns the op's AppendHandler, if it has one. A false result
// means "serve this op through Lookup", not "no such op".
func (r *Registry) LookupAppend(name string) (AppendHandler, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok || e.appendFn == nil {
		return nil, false
	}
	return e.appendFn, true
}

func (r *Registry) LookupRouting(name string) (kind OpKind, ke KeyExtractor, layout RouteLayout, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return 0, nil, RouteLayoutNone, false
	}
	return e.kind, e.ke, e.layout, true
}

// LookupEntry is Lookup plus the crossShard flag (whether the op's handler may
// touch keys beyond its routing shard — see RegisterRoutableCrossShard), read
// from the SAME single registry entry under ONE RLock. A routing caller (e.g.
// the Direct backend's per-shard write serializer) uses it to decide its lock
// strategy without a redundant second CrossShard lock + map lookup.
func (r *Registry) LookupEntry(name string) (fn Handler, kind OpKind, ke KeyExtractor, crossShard, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return nil, 0, nil, false, false
	}
	return e.fn, e.kind, e.ke, e.crossShard, true
}

// ExportRouting copies every registered op's ROUTING metadata (kind,
// KeyExtractor, RouteLayout, cross-shard) into dst, a client-safe routing-only
// registry — no handler crosses over, since a Handler binds *TxContext (which
// wraps *cache.Cache) and the leaf must not import cache.
//
// This is how a caller holding a full server-side Registry (e.g. one built by
// RegisterBuiltins plus any custom or WASM ops) adapts it into the
// wire.Registry a networked client needs for shard-aware routing: see
// rostam.NewClient, which calls this to translate ClientConfig.Ops into the
// low-level client.Config.Ops.
func (r *Registry) ExportRouting(dst *wire.Registry) error {
	r.mu.RLock()
	entries := make(map[string]entry, len(r.m))
	for name, e := range r.m {
		entries[name] = e
	}
	r.mu.RUnlock()
	for name, e := range entries {
		var err error
		switch {
		case e.layout != RouteLayoutNone:
			// The allocation-free layout path has no cross-shard variant on the leaf
			// (no builtin is both layout-routed and cross-shard). Fail loud rather
			// than silently drop the flag if that ever changes.
			if e.crossShard {
				return fmt.Errorf("ops: op %q is both layout-routed and cross-shard; wire.Registry has no cross-shard layout path", name)
			}
			err = dst.RegisterRoutableInto(name, e.kind, e.ke, e.layout)
		case e.ke != nil:
			if e.crossShard {
				err = dst.RegisterRoutableCrossShard(name, e.kind, e.ke)
			} else {
				err = dst.RegisterRoutable(name, e.kind, e.ke)
			}
		default:
			err = dst.Register(name, e.kind)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
