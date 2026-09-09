// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"strconv"
	"strings"
)

// Routing key extractors for vector ops. A vector op's routing key is its
// collection name (canonicalized so a bare name and its "default/"-qualified
// form hash to the same shard), so each collection lives on one shard's Raft
// group and collections distribute across shards instead of all landing on
// shard 0. Direct/single-shard dispatch ignores the extractor; only
// cluster.Node.Call uses it.

// vectorRouteKey canonicalizes a collection name for routing, matching
// vector.canonicalName's bare -> "default/<name>" rule. An empty name yields a
// nil key (caller treats as "no key").
func vectorRouteKey(name string) []byte {
	if name == "" {
		return nil
	}
	if strings.IndexByte(name, '/') < 0 {
		return []byte("default/" + name)
	}
	return []byte(name)
}

// routeExtractor pairs a wire layout's allocating KeyExtractor with its
// RouteLayout tag, so the built-in op table names ONE value per layout and the two
// spellings of a layout can never drift apart there.
type routeExtractor struct {
	ke     KeyExtractor
	layout RouteLayout
}

// routeAt1 / routeAt2 are the two vector-op routing layouts: [colLen:u8][col]...
// and [flags:u8][colLen:u8][col]... respectively.
var (
	routeAt1 = routeExtractor{ke: VectorKeyColAt1, layout: RouteLayoutColAt1}
	routeAt2 = routeExtractor{ke: VectorKeyColAt2, layout: RouteLayoutColAt2}
)

// RouteKeyInto extracts an op's routing key from args WITHOUT allocating: the key
// is either a window into args or "default/<name>" appended to the caller-owned
// scratch. It is the allocation-free twin of the layout's KeyExtractor and returns
// the byte-identical key; nil means "no key" (malformed args, or a layout of
// RouteLayoutNone).
//
// The returned key ALIASES args OR scratch, so — exactly as KeyExtractor already
// documents — the caller must not retain it past the routing decision, and must not
// reuse scratch while it is live. Callers pass a stack array (`var buf [128]byte;
// RouteKeyInto(layout, args, buf[:0])`); this call is DIRECT, which is what lets
// the compiler keep that array on the stack.
func RouteKeyInto(layout RouteLayout, args, scratch []byte) []byte {
	switch layout {
	case RouteLayoutColAt1:
		return vectorKeyColAt1Into(args, scratch)
	case RouteLayoutColAt2:
		return vectorKeyColAt2Into(args, scratch)
	default:
		return nil
	}
}

// CanonicalName returns the canonical routing form of a collection name
// (bare "docs" -> "default/docs"); an already-qualified name is unchanged.
// Used so the meta-Raft catalog keys match the in-process catalog's keying.
func CanonicalName(collection string) string {
	return string(vectorRouteKey(collection))
}

// VectorKeyColAt1 extracts the collection name from args laid out as
// [nameLen:u8][name]... (create/drop/delete/delete_by_filter/scroll/
// search_groups and all vector_mv_* ops).
func VectorKeyColAt1(args []byte) ([]byte, bool) {
	if len(args) < 1 {
		return nil, false
	}
	n := int(args[0])
	if n == 0 || len(args) < 1+n {
		return nil, false
	}
	key := vectorRouteKey(string(args[1 : 1+n]))
	return key, key != nil
}

// VectorKeyColAt2 extracts the collection name from args laid out as
// [flags:u8][colLen:u8][col]... (insert/upsert/search/search_docs/hybrid).
func VectorKeyColAt2(args []byte) ([]byte, bool) {
	if len(args) < 2 {
		return nil, false
	}
	n := int(args[1])
	if n == 0 || len(args) < 2+n {
		return nil, false
	}
	key := vectorRouteKey(string(args[2 : 2+n]))
	return key, key != nil
}

// Compile-time proof that both Into extractors have the KeyExtractorInto shape.
// They are called as plain functions (through RouteKeyInto, a direct call) rather
// than stored as values of that type — see KeyExtractorInto for why an indirect
// call would put the caller's scratch on the heap.
var _, _ KeyExtractorInto = vectorKeyColAt1Into, vectorKeyColAt2Into

// vectorKeyColAt1Into / vectorKeyColAt2Into are the allocation-free forms of
// VectorKeyColAt1 / VectorKeyColAt2 (see KeyExtractorInto): same offsets, same
// canonicalization, but the key is either a window INTO args (an already-qualified
// name needs no rewriting) or "default/" + name appended to the caller's scratch.
// A nil return means "no key", exactly like the (nil, false) of the allocating
// pair.
func vectorKeyColAt1Into(args, scratch []byte) []byte {
	if len(args) < 1 {
		return nil
	}
	n := int(args[0])
	if n == 0 || len(args) < 1+n {
		return nil
	}
	return appendRouteKey(scratch, args[1:1+n])
}

func vectorKeyColAt2Into(args, scratch []byte) []byte {
	if len(args) < 2 {
		return nil
	}
	n := int(args[1])
	if n == 0 || len(args) < 2+n {
		return nil
	}
	return appendRouteKey(scratch, args[2:2+n])
}

// appendRouteKey is vectorRouteKey without the allocation: it applies the SAME
// bare -> "default/<name>" canonicalization, but an already-qualified name is
// returned as a window into the caller's args (nothing to rewrite) and a bare one
// is built in the caller's scratch. The result therefore aliases args OR scratch,
// which is exactly the lifetime KeyExtractor already documents: the key is not
// retained past the routing decision.
func appendRouteKey(scratch, name []byte) []byte {
	if len(name) == 0 {
		return nil
	}
	if bytes.IndexByte(name, '/') >= 0 {
		return name
	}
	out := append(scratch[:0], "default/"...)
	return append(out, name...)
}

// CollectionNameFor cheaply extracts the canonical collection name from a vector
// op's args without fully decoding the payload (no query-slice allocation). It
// reuses the same offset logic the routing layer uses (VectorKeyColAt1/At2), so
// the wire-layout knowledge lives in exactly one place. The returned name is
// canonicalized (bare "docs" -> "default/docs"), matching CanonicalName.
//
// ok is false for ops that are not collection-keyed vector ops, or when args are
// too short to contain a name. Callers that only need a partitioning decision can
// treat (_, false) as "pass through unchanged".
func CollectionNameFor(op string, args []byte) (name string, ok bool) {
	var ke KeyExtractor
	switch op {
	// At2 layout: [flags:u8][colLen:u8][col]...
	case "vector_search", "vector_search_docs", "vector_hybrid_search",
		"vector_insert", "vector_upsert", "vector_insert_if_absent",
		// The named-hybrid ops share the dense-hybrid wire ([flags:u8][colLen:u8]
		// [col]...): the collection name sits at offset 1, behind the flags byte.
		"vector_named_hybrid_search", "vector_named_hybrid_lanes",
		// The MV-hybrid ops use the SAME At2 flags-first wire (EncodeMVHybridArgs):
		// [flags:u8][colLen:u8][col]... — name at offset 1, NOT offset 0 like the rest
		// of the mv_* family. This MUST match the encoder's layout or P>1 routing
		// silently degrades to a single partition (the named-hybrid bug class).
		"vector_mv_hybrid_search", "vector_mv_hybrid_lanes",
		// Full-text ops: EncodeSearchTextArgs / EncodeHybridTextArgs lead with
		// [flags:u8][colLen:u8][col]... — the collection sits at offset 1 (behind the
		// flags byte), so they route via VectorKeyColAt2 like vector_hybrid_search.
		// vector_hybrid_text_lanes shares the vector_hybrid_text wire (the fan-out leaf).
		"vector_search_text", "vector_hybrid_text", "vector_hybrid_text_lanes":
		ke = VectorKeyColAt2
	// At1 layout: [colLen:u8][col]...
	case "vector_search_groups", "vector_scroll", "vector_delete_by_filter",
		"vector_delete", "vector_drop_collection", "vector_exists",
		"vector_get", "vector_get_batch", "vector_set_payload", "vector_overwrite_payload",
		"vector_delete_payload_keys", "vector_clear_payload",
		"vector_mv_add", "vector_mv_add_versioned", "vector_mv_delete", "vector_mv_search",
		"vector_mv_drop_collection", "vector_mv_add_if_absent", "vector_mv_exists",
		"vector_mv_get", "vector_mv_get_batch",
		"vector_mv_set_payload", "vector_mv_overwrite_payload",
		"vector_mv_delete_payload_keys", "vector_mv_clear_payload", "vector_mv_scroll",
		"vector_named_create_collection", "vector_named_drop_collection",
		"vector_named_insert", "vector_named_delete", "vector_named_search",
		"vector_named_sparse_search",
		"vector_named_search_docs", "vector_named_scroll", "vector_named_get_config",
		"vector_named_get", "vector_named_get_batch",
		"vector_named_set_payload", "vector_named_overwrite_payload",
		"vector_named_delete_payload_keys", "vector_named_clear_payload",
		"vector_bulk_stage", "vector_bulk_stage_payload", "vector_bulk_build",
		// vector_bm25_stats (phase 0 of the global-DF text fan-out) leads with
		// [colLen:u8][col] — no flags byte — so the collection sits at offset 0 (At1).
		"vector_bm25_stats",
		// vector_query (unified Query API) leads with [colLen:u8][col] — the
		// QuerySpec blob is opaque to routing, so the collection sits at offset 0.
		// vector_named_query / vector_mv_query share the exact same arg wire (At1).
		"vector_query", "vector_named_query", "vector_mv_query",
		// The operate family leads with [colLen:u8][col][id:u64] — the same At1
		// layout as the set_payload rows above. All three families share one arg
		// wire (EncodeVectorOperateArgs); the op NAME selects the family.
		"vector_operate", "vector_named_operate", "vector_mv_operate":
		ke = VectorKeyColAt1
	default:
		return "", false
	}
	key, found := ke(args)
	if !found {
		return "", false
	}
	return string(key), true
}

// CollectionNameOffset returns the byte offset of the [len:u8][name] collection
// field within a vector op's args (0 for the At1 layout, 1 for At2). ok is false
// for ops that are not collection-keyed vector ops. It mirrors the op→layout
// switch in CollectionNameFor so the wire-layout knowledge stays in one place.
func CollectionNameOffset(op string) (off int, ok bool) {
	switch op {
	case "vector_search", "vector_search_docs", "vector_hybrid_search",
		"vector_insert", "vector_upsert", "vector_insert_if_absent",
		// Named-hybrid ops: [flags:u8][colLen:u8][col]... — name at offset 1.
		"vector_named_hybrid_search", "vector_named_hybrid_lanes",
		// MV-hybrid ops: same At2 flags-first wire (EncodeMVHybridArgs) — name at offset 1.
		"vector_mv_hybrid_search", "vector_mv_hybrid_lanes",
		// Full-text ops: [flags:u8][colLen:u8][col]... — name at offset 1.
		"vector_search_text", "vector_hybrid_text", "vector_hybrid_text_lanes":
		return 1, true
	case "vector_search_groups", "vector_scroll", "vector_delete_by_filter",
		"vector_delete", "vector_drop_collection", "vector_exists",
		"vector_get", "vector_get_batch", "vector_set_payload", "vector_overwrite_payload",
		"vector_delete_payload_keys", "vector_clear_payload",
		"vector_mv_add", "vector_mv_add_versioned", "vector_mv_delete", "vector_mv_search",
		"vector_mv_drop_collection", "vector_mv_add_if_absent", "vector_mv_exists",
		"vector_mv_get", "vector_mv_get_batch",
		"vector_mv_set_payload", "vector_mv_overwrite_payload",
		"vector_mv_delete_payload_keys", "vector_mv_clear_payload", "vector_mv_scroll",
		"vector_named_create_collection", "vector_named_drop_collection",
		"vector_named_insert", "vector_named_delete", "vector_named_search",
		"vector_named_sparse_search",
		"vector_named_search_docs", "vector_named_scroll", "vector_named_get_config",
		"vector_named_get", "vector_named_get_batch",
		"vector_named_set_payload", "vector_named_overwrite_payload",
		"vector_named_delete_payload_keys", "vector_named_clear_payload",
		"vector_bulk_stage", "vector_bulk_stage_payload", "vector_bulk_build",
		// vector_bm25_stats: [colLen:u8][col]... — collection at offset 0 (At1).
		"vector_bm25_stats",
		// vector_query / vector_named_query / vector_mv_query: [colLen:u8][col]... — collection at offset 0.
		"vector_query", "vector_named_query", "vector_mv_query",
		// The operate family: [colLen:u8][col][id:u64]... — collection at offset 0,
		// the same At1 layout as the set_payload rows above.
		"vector_operate", "vector_named_operate", "vector_mv_operate":
		return 0, true
	default:
		return 0, false
	}
}

// RewriteCollectionName returns a NEW args buffer with the op's collection-name
// field replaced by newName (the canonical target an alias resolved to). The name
// is length-prefixed in both wire layouts, so the field is spliced out and newName
// (length-prefixed) spliced in; the rest of the payload (id/query/filter/...) is
// byte-preserved. Used by the fan-out dispatcher to make an UNPARTITIONED alias
// resolve on the pass-through path (the inner op handler has no alias knowledge,
// so the alias name must be rewritten to the canonical before dispatch).
//
// ok is false when the op is not collection-keyed or the args are malformed (the
// caller then passes the original args through unchanged). newName must be <=255
// bytes (the length prefix is a u8 — the same bound CreateCollection enforces);
// an over-long name yields ok=false.
func RewriteCollectionName(op string, args []byte, newName string) (out []byte, ok bool) {
	off, ok := CollectionNameOffset(op)
	if !ok {
		return nil, false
	}
	if len(args) < off+1 {
		return nil, false
	}
	oldLen := int(args[off])
	if oldLen == 0 || len(args) < off+1+oldLen {
		return nil, false
	}
	if len(newName) == 0 || len(newName) > 255 {
		return nil, false
	}
	out = make([]byte, 0, off+1+len(newName)+(len(args)-off-1-oldLen))
	out = append(out, args[:off]...)
	out = append(out, byte(len(newName)))
	out = append(out, newName...)
	out = append(out, args[off+1+oldLen:]...)
	return out, true
}

// singlePointWriteOps is the set of data-plane write ops that mutate exactly ONE
// point by id (dense, MV, and named families: insert/upsert, delete-by-id, and
// every payload mutation). In every one of these op wire layouts the point id is
// a big-endian u64 immediately following the length-prefixed collection-name
// field — At2 ([flags:u8][colLen:u8][col][id:u64]) for insert/upsert, At1
// ([colLen:u8][col][id:u64]) for everything else — so PointIDFor reuses
// CollectionNameOffset to skip past the name and read the next 8 bytes.
//
// delete_by_filter is deliberately ABSENT: it has no single id (it fans the
// delete across a predicate), so PointIDFor returns ok=false for it and the
// __wc__ handler barriers every partition of the collection instead.
var singlePointWriteOps = map[string]struct{}{
	// At2 layout (offset 1): [flags:u8][colLen:u8][col][id:u64]...
	"vector_insert": {}, "vector_upsert": {},
	// At1 layout (offset 0): [colLen:u8][col][id:u64]...
	"vector_delete":                    {},
	"vector_set_payload":               {},
	"vector_operate":                   {},
	"vector_overwrite_payload":         {},
	"vector_delete_payload_keys":       {},
	"vector_clear_payload":             {},
	"vector_mv_add":                    {},
	"vector_mv_add_versioned":          {},
	"vector_mv_delete":                 {},
	"vector_mv_set_payload":            {},
	"vector_mv_operate":                {},
	"vector_mv_overwrite_payload":      {},
	"vector_mv_delete_payload_keys":    {},
	"vector_mv_clear_payload":          {},
	"vector_named_insert":              {},
	"vector_named_delete":              {},
	"vector_named_set_payload":         {},
	"vector_named_operate":             {},
	"vector_named_overwrite_payload":   {},
	"vector_named_delete_payload_keys": {},
	"vector_named_clear_payload":       {},
}

// PointIDFor cheaply extracts the target point id (a big-endian u64) from a
// single-point write op's args WITHOUT a full decode, mirroring the cheap-peek
// style of CollectionNameFor. It is READ-ONLY (never mutates args) and is used by
// the __wc__ envelope handler to resolve which physical partition a single-point
// write landed on (PartitionOf(id, P) → PartitionKeyGen) so the post-commit
// barrier targets the correct shard.
//
// The id sits at a fixed offset past the length-prefixed collection name, which
// is identical across all single-point write ops: CollectionNameOffset (0 for the
// At1 family, 1 for the At2 insert/upsert family) gives the start of the
// [len:u8][name] field, the id follows immediately.
//
// ok is false for ops that are NOT single-point writes (e.g. delete_by_filter,
// searches, admin ops) and for any args too short to contain the id (fail-safe:
// the caller treats (_, false) as "no single shard to barrier").
func PointIDFor(op string, args []byte) (id uint64, ok bool) {
	if _, isWrite := singlePointWriteOps[op]; !isWrite {
		return 0, false
	}
	off, ok := CollectionNameOffset(op)
	if !ok {
		return 0, false
	}
	// [..off..][colLen:u8][col][id:u64]
	if len(args) < off+1 {
		return 0, false
	}
	colLen := int(args[off])
	idStart := off + 1 + colLen
	if len(args) < idStart+8 {
		return 0, false
	}
	return binary.BigEndian.Uint64(args[idStart : idStart+8]), true
}

// PartitionKey is the physical route key (and physical collection name on the
// shard) for partition p of a collection: canonical name + "#" + p. The cluster
// shard hasher maps this string to one of NumShards shards.
func PartitionKey(collection string, p int) []byte {
	base := vectorRouteKey(collection) // canonical, e.g. "default/docs"
	if base == nil {
		return nil
	}
	out := make([]byte, 0, len(base)+1+3)
	out = append(out, base...)
	out = append(out, '#')
	out = append(out, strconv.Itoa(p)...)
	return out
}

// PartitionKeyGen is the physical route key for partition p of a collection at
// generation g. g==0 yields "canonical#p" (byte-identical to PartitionKey — the
// backward-compatible legacy form); g>=1 yields "canonical@g#p" so a new
// generation's partitions never collide with the live generation during resplit.
func PartitionKeyGen(collection string, g uint32, p int) []byte {
	base := vectorRouteKey(collection)
	if base == nil {
		return nil
	}
	out := make([]byte, 0, len(base)+1+10+1+3)
	out = append(out, base...)
	if g != 0 {
		out = append(out, '@')
		out = append(out, strconv.FormatUint(uint64(g), 10)...)
	}
	out = append(out, '#')
	out = append(out, strconv.Itoa(p)...)
	return out
}

// PartitionGenOf is the inverse of the generation encoding in PartitionKeyGen: it
// extracts the generation g from a PHYSICAL partition name. A name without an '@'
// (the legacy "canonical#p" form) is generation 0; "canonical@g#p" is generation
// g. It parses the digits between the LAST '@' and the trailing '#'. A malformed
// name (no trailing '#', non-numeric or overflowing gen, '#' before the '@')
// yields 0 — the conservative default that treats an unrecognized name as the
// legacy (gen-0) form rather than guessing a higher generation.
//
// Used by the dual-write path to decide whether the secondary (target) leg points
// at the RETIRING old generation (gen lower than the live leg's) — a not-found on
// that leg, after the reshard has dropped the old gen, is tolerable; a not-found
// on a higher (being-built) generation is not.
func PartitionGenOf(physName string) uint32 {
	at := strings.LastIndexByte(physName, '@')
	if at < 0 {
		return 0
	}
	hash := strings.LastIndexByte(physName, '#')
	// The '#' must come after the '@' for the form "...@g#p"; otherwise malformed.
	if hash <= at+1 {
		return 0
	}
	digits := physName[at+1 : hash]
	g, err := strconv.ParseUint(digits, 10, 32)
	if err != nil {
		return 0
	}
	return uint32(g)
}

// PartitionOf returns the partition index for id given P partitions (P<=1 -> 0).
// Uses splitmix64 so it is deterministic across processes/restarts (Go's maphash
// is randomly seeded per process and would NOT be stable).
func PartitionOf(id uint64, P int) int {
	if P <= 1 {
		return 0
	}
	z := id + 0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z = z ^ (z >> 31)
	return int(z % uint64(P))
}
