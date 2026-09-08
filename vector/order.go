// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"hash/fnv"
	"math"
	"sort"
	"strings"
	"time"
)

// OrderKind (and its OrderNumeric / OrderDatetime / OrderString constants), the
// OrderBy struct, and the OrderVal tuple element now live in the engine-free
// vtypes leaf package and are re-exported from vtypes_aliases.go.

// ErrEmptyOrderKey is returned by ParseOrderBy when an order_by is requested with an
// empty key. Transports map it to a 400 / InvalidArgument.
var ErrEmptyOrderKey = errors.New("vector: order_by requires a non-empty key")

// ErrBadOrderStart is returned by ParseOrderBy when a datetime start_from string is
// not valid RFC3339. Transports map it to a 400 / InvalidArgument.
var ErrBadOrderStart = errors.New("vector: order_by start_from is not a valid RFC3339 datetime")

// ErrBadOrderKind is returned by ParseOrderBy when the wire requests an impossible
// order-kind combination — is_datetime AND is_string both set (a field is one kind),
// or a string order with a start_from bound (start_from is a numeric/datetime-only
// bound; the string mode has no value-start). Transports map it to a 400 /
// InvalidArgument (fail-loud on a bad combination).
var ErrBadOrderKind = errors.New("vector: order_by kind is ambiguous (is_datetime and is_string both set, or start_from with a string order)")

// ParseOrderBy builds a validated *OrderBy from the wire fields shared by the gRPC
// and HTTP scroll surfaces, so both transports lower/validate order_by identically:
//
//   - key must be non-empty (ErrEmptyOrderKey).
//   - startNumeric (a numeric value bound) and startDatetime (an RFC3339 string bound)
//     are mutually-exclusive optionals; a datetime string is lowered to int-ms via
//     DatetimeMillisFloat (ErrBadOrderStart on an unparseable string). At most one
//     may be set; if both are set the numeric one wins (callers should send one).
//   - isDatetime is the float64-path datetime field-type hint (extraction is numeric).
//   - isString selects the OrderString lexicographic path; it is mutually-exclusive
//     with isDatetime (both ⇒ ErrBadOrderKind) and incompatible with a start_from
//     bound (the string mode has no value-start, so a start_from ⇒ ErrBadOrderKind).
//
// The resulting OrderBy.Kind is OrderString when isString, OrderDatetime when
// isDatetime, else OrderNumeric (the zero value). Numeric/datetime results are
// byte/behaviour-identical to the pre-string signature (Kind is informational for the
// float64 path; the engine only branches on OrderString).
//
// Returns (nil, nil) when key is empty AND no order is intended is NOT this function's
// job — callers decide whether order_by is present; if they call this, key is required.
func ParseOrderBy(key string, desc, isDatetime, isString bool, startNumeric *float64, startDatetime *string) (*OrderBy, error) {
	if key == "" {
		return nil, ErrEmptyOrderKey
	}
	if isString && isDatetime {
		return nil, ErrBadOrderKind
	}
	if isString && (startNumeric != nil || (startDatetime != nil && *startDatetime != "")) {
		return nil, ErrBadOrderKind
	}
	o := &OrderBy{Key: key, Desc: desc, IsDatetime: isDatetime}
	switch {
	case isString:
		o.Kind = OrderString
	case isDatetime:
		o.Kind = OrderDatetime
	default:
		o.Kind = OrderNumeric
	}
	switch {
	case startNumeric != nil:
		o.StartFrom = *startNumeric
		o.HasStart = true
	case startDatetime != nil && *startDatetime != "":
		ms, ok := DatetimeMillisFloat(*startDatetime)
		if !ok {
			return nil, ErrBadOrderStart
		}
		o.StartFrom = ms
		o.HasStart = true
	}
	return o, nil
}

// orderKeyList returns the full ordered sort-key list for an OrderBy: the primary
// (the OrderBy itself, copied WITHOUT its Tail to avoid recursion) followed by Tail.
// The result's element [0] is the primary key, [1] the secondary, etc. Used by the
// tuple snapshot/comparator/cursor paths so they iterate one flat slice. A single-key
// order (empty Tail) yields a 1-element list whose single element drives the byte-
// identical fast path; the multi-key path uses every element.
func orderKeyList(order *OrderBy) []OrderBy {
	keys := make([]OrderBy, 0, 1+len(order.Tail))
	head := *order
	head.Tail = nil
	keys = append(keys, head)
	keys = append(keys, order.Tail...)
	return keys
}

// OrderKeyList is the exported form of orderKeyList: the full ordered sort-key list
// (primary head + Tail) for an OrderBy, each element a single-key OrderBy carrying its
// own Key/Desc/Kind/IsDatetime. Used by the coordinator fan-out (the P>1 tuple merge and
// the v4 cursor bridge) so it iterates the SAME flat key list the engine snapshot does.
func OrderKeyList(order *OrderBy) []OrderBy { return orderKeyList(order) }

// isMultiKey reports whether the order carries secondary/tertiary keys (a non-empty
// Tail). The single-key (false) path stays byte/behaviour-identical (single Key/StrKey
// fields, v2/v3 cursor); the multi-key (true) path uses the Keys tuple + v4 cursor.
func isMultiKey(order *OrderBy) bool { return order != nil && len(order.Tail) > 0 }

// OrderKey extracts the float64 ordering key for the order_by field from a point's
// metadata. It reuses lookupPath (exact/dotted key resolution) + numericValue
// (int+float → float64) so int and float fields share one total order.
//
// A datetime field is stored as a ValueInt holding unix-ms, so it flows through
// numericValue identically (int-ms → float64); the isDatetime hint exists for the
// caller's semantics (and future native datetime handling) and does not change the
// extraction today.
//
// Returns (0, false) when the field is absent or non-numeric. The caller MUST then
// EXCLUDE the point from the order_by scroll (Qdrant default), NOT sort it last.
// A NaN float is treated as non-numeric (excluded): NaN violates the strict-weak
// ordering sort.Slice requires and would never satisfy OrderAfter, corrupting a
// page. NaN can't arrive over JSON (rejected on both transports); this guards the
// in-process embedded API which can construct Value{Kind:ValueFloat, Flt:NaN}.
func OrderKey(meta Metadata, key string, isDatetime bool) (float64, bool) {
	_ = isDatetime // datetime is stored as int-ms; numericValue handles it as an int.
	v, ok := lookupPath(meta, key)
	if !ok {
		return 0, false
	}
	k, ok := numericValue(v)
	if !ok || math.IsNaN(k) {
		return 0, false
	}
	return k, true
}

// stringValue extracts a string from a scalar string Value (ValueString only).
// Returns ("", false) for any non-string kind. The string-order counterpart of
// numericValue: a ValueStrings (multi-value) field is NOT a scalar string key and
// is treated as not-ok (excluded), matching numericValue's scalar-only contract.
func stringValue(v Value) (string, bool) {
	if v.Kind == ValueString {
		return v.Str, true
	}
	// ValueRecord considered: falls to the ("", false) below, correctly
	// declining — a record is not a scalar string ordering key.
	return "", false
}

// OrderStringKey extracts the STRING ordering key for an OrderString field from a
// point's metadata. It is the string analogue of OrderKey: it resolves the (dotted)
// key via lookupPath then reads a scalar string via stringValue.
//
// Returns ("", false) when the field is absent or non-string. The caller MUST then
// EXCLUDE the point from the order_by scroll — the SAME missing-value policy as the
// numeric OrderKey (missing/non-string points are dropped from the snapshot, never
// sorted to an end).
func OrderStringKey(meta Metadata, key string) (string, bool) {
	v, ok := lookupPath(meta, key)
	if !ok {
		return "", false
	}
	return stringValue(v)
}

// OrderLessStr is the (stringValue, id) TOTAL ORDER comparator for OrderString — the
// exact string mirror of OrderLess:
//
//   - ASC:  a < b  iff  aKey < bKey  ||  (aKey == bKey && aID < bID)
//   - DESC: a < b  iff  aKey > bKey  ||  (aKey == bKey && aID < bID)
//
// DESC reverses the order on the KEY only; the id tiebreak stays ASCENDING in BOTH
// directions (same rationale as OrderLess: a globally-unique ascending id tiebreak
// gives a fully deterministic total order and a simple cursor lower-bound seek).
// strings.Compare gives the byte-lexicographic order.
func OrderLessStr(aKey string, aID uint64, bKey string, bID uint64, desc bool) bool {
	if c := strings.Compare(aKey, bKey); c != 0 {
		if desc {
			return c > 0
		}
		return c < 0
	}
	return aID < bID
}

// OrderAfterStr reports whether (key, id) falls STRICTLY AFTER the cursor position
// (curKey, curID) in the (stringValue, id) total order for the direction — the v3
// string resume predicate, the exact mirror of OrderAfter:
//
//	OrderAfterStr(curKey, curID, key, id, desc) == OrderLessStr(curKey, curID, key, id, desc)
func OrderAfterStr(curKey string, curID uint64, key string, id uint64, desc bool) bool {
	return OrderLessStr(curKey, curID, key, id, desc)
}

// OrderLess is the (value, id) TOTAL ORDER comparator used to sort a scroll page
// deterministically.
//
//   - ASC:  a < b  iff  aKey < bKey  ||  (aKey == bKey && aID < bID)
//   - DESC: a < b  iff  aKey > bKey  ||  (aKey == bKey && aID < bID)
//
// DESC reverses the order on the KEY only — the id tiebreak stays ASCENDING in
// BOTH directions. This is a deliberate choice: ids are globally unique, so an
// ascending id tiebreak makes every page a fully deterministic total order
// (no two distinct points ever compare equal) AND keeps the fan-out merge correct
// (disjoint per-partition id sets resolve value-ties identically everywhere). A
// desc tiebreak would NOT change determinism but would complicate the cursor's
// lower-bound seek; ascending is simpler and sufficient.
func OrderLess(aKey float64, aID uint64, bKey float64, bID uint64, desc bool) bool {
	if aKey != bKey {
		if desc {
			return aKey > bKey
		}
		return aKey < bKey
	}
	return aID < bID
}

// OrderAfter reports whether the point (key, id) falls STRICTLY AFTER the cursor
// position (curKey, curID) in the (value, id) total order for the given direction.
// It is the lower-bound resume predicate used to seek the first row of
// the next page (the row immediately after the last row of the previous page):
//
//	OrderAfter(curKey, curID, key, id, desc) == OrderLess(curKey, curID, key, id, desc)
//
// i.e. "the cursor is strictly less than (key,id)". Exposed as a named predicate so
// the seek site reads clearly and the asymmetry (cursor row itself is EXCLUDED) is
// explicit. For a sort.Search lower bound over a slice already sorted by OrderLess,
// search for the first index i where OrderAfter(curKey, curID, s[i].key, s[i].id, desc)
// is true.
func OrderAfter(curKey float64, curID uint64, key float64, id uint64, desc bool) bool {
	return OrderLess(curKey, curID, key, id, desc)
}

// OrderedID pairs an ordering key with its point id for sorting a scroll page.
// Key holds the float64 key for OrderNumeric/OrderDatetime; StrKey holds the string
// key for OrderString (Key is unused / zero then). One row type carries either key
// by the order kind so the snapshot + cursor seek work uniformly across both modes.
//
// Keys is the MULTI-KEY tuple (one OrderVal per sort key, index 0 the primary). It is
// EMPTY for a single-key order — that path uses the Key/StrKey fast-path fields and
// stays byte/behaviour-identical. The multi-key snapshot fills Keys and sorts by
// OrderLessTuple; Key/StrKey are left zero then (the tuple is the source of truth).
//
// Keys is a slice, so OrderedID is NOT comparable with == when Keys is non-nil. The
// single-key path leaves Keys nil, keeping struct-equality valid for the existing
// single-key tests (which only ever build single-key rows).
type OrderedID struct {
	Key    float64
	ID     uint64
	StrKey string
	Keys   []OrderVal
}

// orderValLess reports whether OrderVal a sorts before b under the per-key kind+desc.
// It is the single-key comparator factored to one OrderVal so the tuple comparator can
// reuse the exact numeric/string total order at each tuple position. NOTE: this is the
// KEY-ONLY compare (no id tiebreak); the tuple comparator applies the id tiebreak once,
// after all keys tie. a and b are assumed to share the same Kind (the snapshot builds a
// fixed per-position kind), so a.Kind selects the branch.
func orderValLess(a, b OrderVal, desc bool) (less, equal bool) {
	if a.Kind == OrderString {
		c := strings.Compare(a.Str, b.Str)
		if c == 0 {
			return false, true
		}
		if desc {
			return c > 0, false
		}
		return c < 0, false
	}
	if a.Num == b.Num {
		return false, true
	}
	if desc {
		return a.Num > b.Num, false
	}
	return a.Num < b.Num, false
}

// OrderLessTuple is the (k1, k2, …, kN, id) TUPLE-LEXICOGRAPHIC total order comparator
// for a MULTI-KEY order: compare a.Keys[0] vs b.Keys[0] under keys[0].Desc; on a tie,
// a.Keys[1] vs b.Keys[1] under keys[1].Desc; …; on all keys equal, the id tiebreak
// (aID < bID, ASCENDING in every direction — same rationale as OrderLess: a globally-
// unique ascending id tiebreak gives a deterministic total order and a simple cursor
// lower-bound seek). keys carries the per-position Kind+Desc (a.Keys[i].Kind must match
// keys[i].Kind — the snapshot builds them in lockstep). A 1-element keys list reduces
// EXACTLY to OrderLess/OrderLessStr (the byte-identical single-key order), so the
// single-key path could delegate here; the engine keeps the dedicated single-key fast
// path for zero churn and this is the N>1 general path.
func OrderLessTuple(a, b OrderedID, keys []OrderBy) bool {
	for i := range keys {
		less, equal := orderValLess(a.Keys[i], b.Keys[i], keys[i].Desc)
		if !equal {
			return less
		}
	}
	return a.ID < b.ID
}

// OrderAfterTuple reports whether the row (key tuple, id) falls STRICTLY AFTER the
// cursor position cur in the tuple-lexicographic total order for keys — the v4 multi-
// key resume predicate, the tuple analogue of OrderAfter:
//
//	OrderAfterTuple(cur, row, keys) == OrderLessTuple(cur, row, keys)
//
// i.e. "the cursor is strictly less than the row". For a sort.Search lower bound over a
// slice already sorted by OrderLessTuple, search for the first index i where
// OrderAfterTuple(cur, rows[i], keys) is true (the first row of the next page; the
// cursor row itself is EXCLUDED).
func OrderAfterTuple(cur, row OrderedID, keys []OrderBy) bool {
	return OrderLessTuple(cur, row, keys)
}

// SortOrderedIDsTuple sorts rows in place by the tuple-lexicographic (k1,…,kN, id)
// total order (OrderLessTuple) for the per-key kinds+directions in keys — the multi-key
// analogue of SortOrderedIDs, sharing OrderLessTuple with the v4 cursor seek.
func SortOrderedIDsTuple(rows []OrderedID, keys []OrderBy) {
	sort.Slice(rows, func(i, j int) bool {
		return OrderLessTuple(rows[i], rows[j], keys)
	})
}

// orderTupleKeys extracts the per-key OrderVal tuple for a point's metadata under the
// ordered key list keys. It is the multi-key analogue of OrderKey/OrderStringKey: each
// key resolves by its Kind (string ⇒ OrderStringKey, else OrderKey). It returns
// (nil, false) — EXCLUDE the point — if ANY key is missing/wrong-type (the same
// missing-value policy as single-key, applied per key: a row must HAVE every order
// field to be sortable by the full tuple). The returned slice has one OrderVal per key,
// each stamped with that key's Kind so OrderLessTuple dispatches correctly.
func orderTupleKeys(meta Metadata, keys []OrderBy) ([]OrderVal, bool) {
	vals := make([]OrderVal, len(keys))
	for i := range keys {
		if keys[i].Kind == OrderString {
			sk, ok := OrderStringKey(meta, keys[i].Key)
			if !ok {
				return nil, false
			}
			vals[i] = OrderVal{Str: sk, Kind: OrderString}
			continue
		}
		k, ok := OrderKey(meta, keys[i].Key, keys[i].IsDatetime)
		if !ok {
			return nil, false
		}
		vals[i] = OrderVal{Num: k, Kind: keys[i].Kind}
	}
	return vals, true
}

// SortOrderedIDs sorts rows in place by the (value, id) total order (OrderLess)
// for the given direction. A small convenience so callers use one helper rather
// than re-spelling the comparator at every scroll site.
func SortOrderedIDs(rows []OrderedID, desc bool) {
	// Reuse OrderLess so the sort and the cursor seek share one comparator.
	sort.Slice(rows, func(i, j int) bool {
		return OrderLess(rows[i].Key, rows[i].ID, rows[j].Key, rows[j].ID, desc)
	})
}

// SortOrderedIDsStr sorts rows in place by the (stringValue, id) total order
// (OrderLessStr) for the given direction — the OrderString analogue of
// SortOrderedIDs, sharing OrderLessStr with the v3 cursor seek.
func SortOrderedIDsStr(rows []OrderedID, desc bool) {
	sort.Slice(rows, func(i, j int) bool {
		return OrderLessStr(rows[i].StrKey, rows[i].ID, rows[j].StrKey, rows[j].ID, desc)
	})
}

// orderSeekStart computes the index of the first row a scroll page should collect
// from a snapshot already sorted by the order's (value, id) total order. It is the
// SHARED, kind-aware seek used by every family's collectOrderedLocked so the numeric,
// datetime and string seeks stay in one place and the comparator matches the sort:
//
//   - hasAfter (a resume cursor): the first row STRICTLY AFTER (afterKey/ResumeStr,
//     afterID) in the direction — OrderAfter for the float kinds, OrderAfterStr for
//     OrderString (the resume STRING rides order.ResumeStr; afterID is the shared id).
//   - else order.HasStart (page-1 start_from): the first row at/after the inclusive
//     StartFrom value bound (numeric/datetime only; string has no start_from in this
//     mode, so a string order with no cursor starts at index 0).
//   - else: index 0 (page 1, no bound).
//
// The numeric/datetime branch is byte-identical to the per-family seek it replaced.
//
// MULTI-KEY (order.Tail non-empty): the seek uses the tuple total order. With a v4
// resume cursor (hasAfter): the first row STRICTLY AFTER the (ResumeKeys, afterID)
// tuple via OrderAfterTuple. Without a cursor: index 0 (multi-key has no start_from —
// it is primary-only and the multi-key request does not carry one). The single-key
// branches below are UNTOUCHED (byte-identical).
func orderSeekStart(rows []OrderedID, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool) int {
	if isMultiKey(order) {
		if !hasAfter || !order.HasResumeKeys {
			return 0
		}
		keys := orderKeyList(order)
		cur := OrderedID{ID: afterID, Keys: order.ResumeKeys}
		return sort.Search(len(rows), func(i int) bool {
			return OrderAfterTuple(cur, rows[i], keys)
		})
	}
	if order.Kind == OrderString {
		if hasAfter {
			return sort.Search(len(rows), func(i int) bool {
				return OrderAfterStr(order.ResumeStr, afterID, rows[i].StrKey, rows[i].ID, order.Desc)
			})
		}
		return 0
	}
	switch {
	case hasAfter:
		return sort.Search(len(rows), func(i int) bool {
			return OrderAfter(afterKey, afterID, rows[i].Key, rows[i].ID, order.Desc)
		})
	case order.HasStart:
		return sort.Search(len(rows), func(i int) bool {
			if order.Desc {
				return rows[i].Key <= order.StartFrom
			}
			return rows[i].Key >= order.StartFrom
		})
	}
	return 0
}

// orderCacheKey identifies a cached order_by sorted snapshot by its field and
// direction. Two scrolls over the same (field, direction) share one snapshot;
// asc and desc are distinct entries (the sort order differs).
//
// MULTI-KEY: a multi-key snapshot is sorted by the FULL tuple, so a single-key cache
// key would collide distinct tuples (or share a snapshot sorted by a different tuple).
// orderSnapCacheKey derives the field component from the full key list (field+kind+desc
// per key) for a multi-key order so each distinct tuple gets its own cached snapshot;
// a single-key order keeps the byte-identical {field, desc} key (zero churn). desc here
// is the PRIMARY direction (informational for multi-key; the per-key desc is encoded in
// the field string).
type orderCacheKey struct {
	field string
	desc  bool
}

// orderSnapCacheKey builds the snapshot cache key for an order. Single-key:
// {order.Key, order.Desc} — byte-identical to the pre-multi-key key. Multi-key: the
// field component encodes every key's (kind, desc, field) so distinct tuples never
// share a snapshot, and the snapshot's tuple sort always matches the request.
func orderSnapCacheKey(order *OrderBy) orderCacheKey {
	if !isMultiKey(order) {
		return orderCacheKey{order.Key, order.Desc}
	}
	var b strings.Builder
	for i, k := range orderKeyList(order) {
		if i > 0 {
			b.WriteByte('\x1f') // unit separator: keys can't contain it meaningfully
		}
		b.WriteByte(byte(k.Kind))
		if k.Desc {
			b.WriteByte('d')
		} else {
			b.WriteByte('a')
		}
		b.WriteString(k.Key)
	}
	return orderCacheKey{field: b.String(), desc: order.Desc}
}

// orderSnap is a cached, fully-sorted (value, id) snapshot of every LIVE id that
// HAS the order field, stamped with the dataVersion it was built at. It mirrors the
// id-scroll scrollSnap but is field-specific (so it must invalidate on payload
// writes, hence the separate dataVersion counter — idSetVersion ignores payloads).
//
// The rows slice is IMMUTABLE once stamped: a rebuild REPLACES the *orderSnap in the
// map wholesale (never mutates rows in place), so a concurrent RLock warm reader that
// captured the old pointer keeps reading a stable, self-consistent slice. `seq` is a
// monotonic stamp used purely for oldest-entry eviction (lower seq == built earlier).
type orderSnap struct {
	ver  uint64
	seq  uint64
	rows []OrderedID
}

// orderCacheCap bounds the per-index order-snapshot map. order_by scroll typically
// uses 1-2 (field, direction) combos; the cap prevents unbounded growth across many
// distinct order fields (the one growth mode the single scrollSnap never had). On
// overflow the oldest-built entry is evicted.
const orderCacheCap = 4

// evictOldestOrderSnap drops the lowest-seq (oldest-built) entry from snaps so it
// stays within orderCacheCap. Called under the index write lock before inserting a
// fresh snapshot. Eviction only removes the map entry; the evicted *orderSnap's rows
// slice is never mutated, so any warm reader still holding the old pointer is safe.
func evictOldestOrderSnap(snaps map[orderCacheKey]*orderSnap) {
	var oldestKey orderCacheKey
	var oldestSeq uint64
	first := true
	for k, s := range snaps {
		if first || s.seq < oldestSeq {
			oldestKey, oldestSeq, first = k, s.seq, false
		}
	}
	if !first {
		delete(snaps, oldestKey)
	}
}

// DatetimeMillisFloat lowers an RFC3339 datetime STRING to unix-MILLISECONDS as a
// float64 — the order_by datetime start_from counterpart of the filter's
// datetimeBound (filter.go), so a datetime start_from and a datetime filter agree on
// the int-ms convention. ok=false for an unparseable string. The transports use this
// to accept an RFC3339 string start_from for a datetime order field (a numeric ms
// start_from is used as-is).
func DatetimeMillisFloat(s string) (float64, bool) {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return 0, false
	}
	return float64(t.UnixMilli()), true
}

// OrderKeyHash returns a cheap 16-bit hash of the order_by key, used in the v2
// cursor so mid-pagination order_by-key changes are DETECTED (the resume cursor's
// keyHash must match the request's order_by key) and rejected loudly rather than
// silently mis-paging. It is NOT cryptographic — collision-resistance is not
// required, only that a different key string very likely yields a different hash.
func OrderKeyHash(key string) uint16 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	s := h.Sum32()
	return uint16(s ^ (s >> 16))
}

// OrderKeyListHash returns a 16-bit hash over a MULTI-KEY order's full ordered key
// list — the v4 cursor's mid-pagination guard analogue of OrderKeyHash. It hashes each
// key's field name (with a unit separator between keys so {"ab","c"} and {"a","bc"}
// differ) so a changed key set, a reordered key list, or a different arity yields a
// different hash and a resume across an order_by change is rejected loudly. Direction
// and kind are NOT hashed (they are validated separately by ValidateOrderCursorTuple's
// desc + arity checks); the hash is purely the key-identity guard, symmetric with the
// single-key OrderKeyHash (which likewise hashes only the key string). A 1-element list
// is NOT used here (single-key emits v2/v3 via OrderKeyHash); this is the N>1 path.
func OrderKeyListHash(keys []OrderBy) uint16 {
	h := fnv.New32a()
	for i := range keys {
		if i > 0 {
			_, _ = h.Write([]byte{'\x1f'}) // unit separator between keys
		}
		_, _ = h.Write([]byte(keys[i].Key))
	}
	s := h.Sum32()
	return uint16(s ^ (s >> 16))
}
