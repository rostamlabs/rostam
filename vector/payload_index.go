// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/rostamlabs/rostam/sdk/record"
)

// scalarKey is a comparable map key for an equality-indexable scalar Value
// (string, int, float, bool). Only the field matching kind is meaningful.
type scalarKey struct {
	kind ValueKind
	str  string
	i    int64
	f    float64
	b    bool
}

// scalarKeyOf returns the index key for a scalar Value, or ok=false for kinds
// that are not equality-indexable (none, slices) — and for the one VALUE that is
// not indexable either: NaN.
//
// NaN IS NOT A MAP KEY. A scalarKey carrying NaN hashes fine but never compares
// equal to itself, so `vals[key]` after `vals[key] = set` misses: every insert
// of a NaN-valued field appended a NEW, permanently unreachable posting entry,
// and drop's delete could never remove it. That was an unbounded leak whose
// entries no query could ever return — and, once the sorted key cache picked
// them up, a NaN with no position under `<` sitting in a list the range path
// BINARY-SEARCHES, which can mis-place the split point for the ordinary keys
// around it.
//
// Declining here is both the fix and the semantics: under the IEEE rule
// (orderingHoldsFloat) a NaN field satisfies no range predicate, and under `==`
// it satisfies no equality predicate either, so a NaN value has no query that
// can match it — "not indexed" and "matches nothing" say the same thing, which
// is exactly the equivalence every posting set's exactness proof needs. A NaN
// BOUND is declined for the same reason and with the same consequence: the
// filter simply stops narrowing and the (rejecting) predicate answers.
//
// TWIN: ops/kvindex/def.go carries the same scalarKey/scalarKeyOf pair for the
// KV record posting index (that package is engine-free by design and cannot
// import this one). Keep the two in lockstep — if they ever disagree about
// which values are indexable, the KV postings stop being a superset of the
// predicate they narrow, which is the one thing that index may never be.
func scalarKeyOf(v Value) (scalarKey, bool) {
	switch v.Kind {
	case ValueString:
		return scalarKey{kind: ValueString, str: v.Str}, true
	case ValueInt:
		return scalarKey{kind: ValueInt, i: v.Int}, true
	case ValueFloat:
		if v.Flt != v.Flt { // NaN
			return scalarKey{}, false
		}
		return scalarKey{kind: ValueFloat, f: v.Flt}, true
	case ValueBool:
		return scalarKey{kind: ValueBool, b: v.Bool}, true
	case ValueRecord:
		// A record is a byte blob, not a scalar: it has no equality/ordering key
		// here (its top-level fields get their own synthetic scalar entries via
		// the record resolver, not this index). Made explicit rather than left to
		// the default below, matching this file's own NaN-decline discipline.
		return scalarKey{}, false
	default:
		return scalarKey{}, false
	}
}

// fieldKey names a (field, scalar value) pair in the payload index.
type fieldKey struct {
	field string
	key   scalarKey
}

// geoSlotCell names a (field, grid cell) entry a slot occupies in the geo
// spatial index, so reindex/drop can remove the slot's old cell on reuse.
type geoSlotCell struct {
	field string
	cell  geoCell
}

// fieldToken names a (field, token) entry a slot occupies in the inverted token
// index, so reindex/drop can remove the slot's old tokens on reuse — the token
// analogue of fieldKey/geoSlotCell.
type fieldToken struct {
	field string
	token string
}

// payloadIndex is an inverted index over equality-valued metadata fields:
// field -> scalar value -> set of slots. It accelerates selective equality
// filters via the filter-first search path.
//
// It is an ACCELERATION ONLY. candidates() returns a superset of the matching
// live slots and the search path re-applies the full predicate (and tombstone/
// TTL admission), so correctness never depends on the index being exact — it
// may legitimately contain tombstoned slots between reclaims. Guarded by the
// owning hnsw's mutex.
type payloadIndex struct {
	fields   map[string]map[scalarKey]map[uint32]struct{}
	slotKeys map[uint32][]fieldKey // per-slot keys, for O(keys) reindex on slot reuse

	// geo is a per-field geohash-prefix spatial index: field -> grid cell ->
	// slot set. A ValueGeo metadata field's point is bucketed by its fixed-
	// precision geohash cell (geohashPrecision ≈ 4.9 km, see geohash.go).
	// Geo values do NOT participate in the scalar eq/range structures above
	// (scalarKeyOf declines them); they live ONLY here. geoSlotCells records,
	// per slot, the (field, cell) entries to drop on reindex/reuse — the geo
	// analogue of slotKeys.
	geo          map[string]map[geoCell]map[uint32]struct{}
	geoSlotCells map[uint32][]geoSlotCell

	// tokens is a per-field inverted token index: field -> token -> slot set.
	// A string (ValueString) or strings (ValueStrings) metadata field is
	// tokenized with the SAME tokenize() the FilterMatch predicate uses, so a
	// `match` clause can narrow to a candidate superset (then the predicate
	// re-checks). Postings are PER-DOCUMENT: a token present in ANY element of a
	// ValueStrings field posts the slot once; the predicate's per-element
	// tokensContainAll re-narrows. tokenSlotKeys records, per slot, the
	// (field, token) entries to drop on reindex/reuse — the token analogue of
	// geoSlotCells. Like the rest of the index this is ACCELERATION ONLY (a
	// superset); it never carries the contentField.
	tokens        map[string]map[string]map[uint32]struct{}
	tokenSlotKeys map[uint32][]fieldToken

	// contains is a per-field inverted ELEMENT index: field -> array element value
	// -> slot set. An array (ValueStrings/Ints/Floats) metadata field posts the
	// slot under EACH DISTINCT element it holds, keyed by the SAME scalarKey the
	// eq index + compileContains use (so a `contains X` clause narrows to
	// contains[field][scalarKeyOf(X)] — exactly the docs whose array contains X,
	// then the predicate re-checks). The element kind carried in the scalarKey
	// mirrors the predicate's array-kind check (a string want only hits a
	// ValueStrings element). containsSlotKeys records, per slot, the (field,
	// element key) entries to drop on reindex/reuse — the contains analogue of
	// tokenSlotKeys. ACCELERATION ONLY (a no-false-negative superset); never
	// carries the contentField. A doc with a huge array posts many element keys
	// (memory) — acceptable v1, mirrors the token index.
	contains         map[string]map[scalarKey]map[uint32]struct{}
	containsSlotKeys map[uint32][]fieldKey

	// sorted caches per-field distinct keys in sorted order so range queries
	// binary-search the bounds (O(log D + matches)) instead of scanning every
	// distinct value (O(D)). Built lazily on the first range query after a
	// structural change; invalidated only when a distinct key is added/removed
	// (set-membership changes are reflected live, since orderingSet fetches the
	// posting sets from `fields`). sortedMu serializes concurrent readers' lazy
	// builds (candidates runs under the owner's read lock, which excludes
	// writers but not other readers).
	sorted   map[string]*sortedKeys
	sortedMu sync.Mutex

	// posts counts, per field, how many SLOTS carry a numeric (int/float) resp.
	// string scalar key for it. Maintained incrementally by reindex/drop under the
	// owner's write lock — never derived by iteration, because the whole point is
	// to answer "is this field total?" in O(1) on a query path.
	//
	// WHY IT EXISTS: the COMPLEMENT gate (filter_bitset.go) marks the slots a
	// range predicate REJECTS and admits everything else, which is only sound if
	// the index knows about every live slot — a slot with no value for the field
	// is rejected by the predicate but appears in no posting set, so it would be
	// silently admitted. `posts[f].num >= arena.Size()` is exactly the proof that
	// no such slot exists; see fieldTotalNumeric for the argument in full.
	//
	// One key per (field, slot) — reindex walks `meta` once per field — so these
	// are slot counts, not entry counts, and a field's numeric and string counts
	// are disjoint.
	posts map[string]fieldPosts

	// badRecords counts, per PAYLOAD KEY, the live slots holding a ValueRecord
	// that IndexEntries could not enumerate, and badSlotKeys records which keys
	// each slot contributes so reindex removes its contribution exactly once.
	// A non-zero count makes every path under that key un-narrowable — see
	// recordPoison for why that is the only sound answer. Maintained by reindex
	// under the owner's write lock, and rebuilt by rebuild for free because
	// rebuild reindexes every slot.
	badRecords  recordPoison
	badSlotKeys map[uint32][]string

	// cols is the numeric column sidecar: field -> slot-indexed []float64 with
	// NaN for "no numeric value". Built lazily by the query path (ensureColumn,
	// under colsMu) and maintained by reindex (updateColumns, under the owner's
	// write lock). See payload_column.go.
	cols   map[string]*numColumn
	colsMu sync.Mutex
	// colClock stamps column use for the LRU eviction the cap needs; guarded by
	// colsMu. colBytes is the sidecar's total footprint, kept atomic because it is
	// READ on the insert path (under the write lock, to charge columns against
	// Config.MaxBytes) and WRITTEN on the query path (under colsMu) — two
	// different locks, so neither can guard it.
	colClock uint64
	colBytes atomic.Int64
}

// fieldPosts is one field's per-kind slot counts. bool and geo values are
// counted by neither: no ordering op can be answered from them, so no
// totality claim is ever made about them.
type fieldPosts struct {
	num int // slots carrying a ValueInt or ValueFloat key
	str int // slots carrying a ValueString key
}

// addPost applies delta to the counter for (field, kind), dropping the entry
// when it returns to zero so a field that goes away leaves nothing behind.
// Must hold the owner's write lock.
func (p *payloadIndex) addPost(field string, kind ValueKind, delta int) {
	c, ok := p.posts[field]
	switch kind {
	case ValueInt, ValueFloat:
		c.num += delta
	case ValueString:
		c.str += delta
	default: // ValueRecord (and bool) considered: never reaches here — scalarKeyOf declines records, so no scalarKey carries kind == ValueRecord.
		return // bool: counted by neither, see fieldPosts
	}
	if c.num == 0 && c.str == 0 {
		if ok {
			delete(p.posts, field)
		}
		return
	}
	p.posts[field] = c
}

// fieldTotalNumeric reports whether EVERY live slot carries a numeric value for
// field, given liveCount = arena.Size(). It is the precondition the complement
// gate's exactness rests on, and the proof has three links:
//
//  1. Every slot with a posting is in arena.idMap. Every path that removes an id
//     from idMap drops the slot's postings FIRST (insertLocked's dead-slot
//     reclaim calls reindex(old, nil) immediately before arena.Delete; Restore
//     and Reclaim go through rebuild, which refills from idMap). A tombstoned or
//     TTL-expired slot is still IN idMap and still indexed, which is fine — it is
//     counted on both sides. TestPayloadIndexHasNoSlotsOutsideTheIdMap pins this.
//  2. posts[field].num counts SLOTS, not entries: reindex iterates `meta` once
//     per field, so a slot contributes at most one key per field.
//  3. arena.Size() is len(idMap), i.e. an UPPER bound on the live slots.
//
// Together: posts >= Size means the numeric postings cover all of idMap, hence
// all live slots. The comparison is >= rather than == purely defensively; (1)
// and (2) make > impossible.
func (p *payloadIndex) fieldTotalNumeric(field string, liveCount int) bool {
	if admitGateSkipTotality {
		return true // mutation seam; see admitGateSkipTotality
	}
	return liveCount > 0 && p.posts[field].num >= liveCount
}

// fieldTotalString is fieldTotalNumeric for string-keyed ordering (lexicographic
// ranges). Same argument, other counter.
func (p *payloadIndex) fieldTotalString(field string, liveCount int) bool {
	if admitGateSkipTotality {
		return true // mutation seam; see admitGateSkipTotality
	}
	return liveCount > 0 && p.posts[field].str >= liveCount
}

// numEntry is a numeric distinct value with its index key, sorted by val.
type numEntry struct {
	val float64
	key scalarKey
}

// sortedKeys is a field's distinct keys in sorted order: numeric keys (int/float
// unified as float64) and string keys, each list ascending. dirty marks it stale
// after a key add/remove.
type sortedKeys struct {
	dirty bool
	num   []numEntry
	str   []scalarKey
}

func newPayloadIndex() *payloadIndex {
	return &payloadIndex{
		fields:           make(map[string]map[scalarKey]map[uint32]struct{}),
		slotKeys:         make(map[uint32][]fieldKey),
		sorted:           make(map[string]*sortedKeys),
		geo:              make(map[string]map[geoCell]map[uint32]struct{}),
		geoSlotCells:     make(map[uint32][]geoSlotCell),
		tokens:           make(map[string]map[string]map[uint32]struct{}),
		tokenSlotKeys:    make(map[uint32][]fieldToken),
		contains:         make(map[string]map[scalarKey]map[uint32]struct{}),
		containsSlotKeys: make(map[uint32][]fieldKey),
		posts:            make(map[string]fieldPosts),
		badRecords:       make(recordPoison),
		badSlotKeys:      make(map[uint32][]string),
	}
}

// markDirty invalidates field's sorted cache (called when a distinct key is
// added or removed). A no-op if the field has never been range-queried.
func (p *payloadIndex) markDirty(field string) {
	if sc := p.sorted[field]; sc != nil {
		sc.dirty = true
	}
}

// ensureSorted returns field's sorted-key cache, rebuilding it from the live
// posting map if stale. Caller holds the owner's read lock (so `fields` is
// stable); sortedMu serializes concurrent rebuilds.
func (p *payloadIndex) ensureSorted(field string) *sortedKeys {
	p.sortedMu.Lock()
	defer p.sortedMu.Unlock()
	sc := p.sorted[field]
	if sc == nil {
		sc = &sortedKeys{dirty: true}
		p.sorted[field] = sc
	}
	if !sc.dirty {
		return sc
	}
	sc.num = sc.num[:0]
	sc.str = sc.str[:0]
	for key := range p.fields[field] {
		// ValueRecord considered: key.kind can never be ValueRecord — scalarKeyOf
		// declines records before a scalarKey is ever minted, so no case is needed.
		switch key.kind {
		case ValueInt:
			sc.num = append(sc.num, numEntry{float64(key.i), key})
		case ValueFloat:
			// DEFENCE IN DEPTH. scalarKeyOf already refuses to mint a NaN key, so
			// this branch cannot fire today — but the whole range path's correctness
			// rests on sc.num being TOTALLY ordered under `<`, and a single NaN
			// breaks that for its neighbours as well as itself (sort.Slice with an
			// inconsistent comparator does not merely misplace the odd element, and
			// sort.Search's probe is false AT the NaN, so the split point can land
			// early). One comparison here is a cheap price for never having to
			// re-derive that argument if a future key source forgets.
			if key.f != key.f {
				continue
			}
			sc.num = append(sc.num, numEntry{key.f, key})
		case ValueString:
			sc.str = append(sc.str, key)
		}
	}
	sort.Slice(sc.num, func(i, j int) bool { return sc.num[i].val < sc.num[j].val })
	sort.Slice(sc.str, func(i, j int) bool { return sc.str[i].str < sc.str[j].str })
	sc.dirty = false
	return sc
}

// reindex removes any existing entries for slot, then indexes the scalar fields
// of meta (meta may be nil/empty, which just removes). Calling it on every
// insert keeps the index correct across slot reuse. Must hold the owner's lock.
func (p *payloadIndex) reindex(slot uint32, meta Metadata) {
	// Columns first, and before the empty-meta early return below: a payload that
	// was CLEARED must reset the slot's column entries to NaN, which is exactly
	// the case that return would skip.
	p.updateColumns(slot, meta)
	if old, ok := p.slotKeys[slot]; ok {
		for _, fk := range old {
			p.drop(fk, slot)
		}
		delete(p.slotKeys, slot)
	}
	if old, ok := p.geoSlotCells[slot]; ok {
		for _, gc := range old {
			p.dropGeo(gc, slot)
		}
		delete(p.geoSlotCells, slot)
	}
	if old, ok := p.tokenSlotKeys[slot]; ok {
		for _, ft := range old {
			p.dropToken(ft, slot)
		}
		delete(p.tokenSlotKeys, slot)
	}
	if old, ok := p.containsSlotKeys[slot]; ok {
		for _, fk := range old {
			p.dropContains(fk, slot)
		}
		delete(p.containsSlotKeys, slot)
	}
	// The malformed-record markers are reverse entries like any other: dropped
	// FIRST, so a slot whose record was repaired (or whose payload was cleared,
	// which is the early return below) stops poisoning its payload key.
	if old, ok := p.badSlotKeys[slot]; ok {
		for _, key := range old {
			p.dropBadRecord(key)
		}
		delete(p.badSlotKeys, slot)
	}
	if len(meta) == 0 {
		return
	}
	var keys []fieldKey
	var geoCells []geoSlotCell
	var slotTokens []fieldToken
	var slotContains []fieldKey
	var recKeys []fieldKey // scratch, reused across this slot's record fields
	var badKeys []string   // payload keys whose record this slot could not index
	for field, v := range meta {
		if field == contentField {
			continue // document content is not a filterable field; never index it
		}
		// Geo values bucket into the spatial index ONLY (they decline the scalar
		// index via scalarKeyOf=false); maintained incrementally per slot here.
		if v.Kind == ValueGeo {
			cell := cellOf(v.Lat, v.Lon)
			cells := p.geo[field]
			if cells == nil {
				cells = make(map[geoCell]map[uint32]struct{})
				p.geo[field] = cells
			}
			set := cells[cell]
			if set == nil {
				set = make(map[uint32]struct{})
				cells[cell] = set
			}
			set[slot] = struct{}{}
			geoCells = append(geoCells, geoSlotCell{field, cell})
			continue
		}
		// Record values expand into SYNTHETIC eq entries — one per top-level
		// scalar field and per table row count, named "<field>/<recordField>"
		// — and into nothing else. They take no other branch below anyway
		// (scalarKeyOf declines a record, it is not a string or an array, so
		// neither the token nor the contains index would see it), so the
		// `continue` here changes nothing for the other sub-indexes; it just
		// says so once.
		if v.Kind == ValueRecord {
			var whole bool
			recKeys, whole = appendRecordIndexKeys(recKeys[:0], meta, field, v.Rec)
			if !whole {
				// Enumeration failed, so this record's fields are NOT in the
				// index — while Resolve will still answer for whichever of them
				// survived the damage. Poison the payload key rather than post
				// what little we have: see recordPoison.
				p.addBadRecord(field)
				badKeys = append(badKeys, field)
				continue
			}
			for _, rk := range recKeys {
				if p.postScalar(rk.field, rk.key, slot) {
					keys = append(keys, rk)
				}
			}
			continue
		}
		// Token index: a string/strings field is tokenized (same tokenize() as
		// the FilterMatch predicate) and each DISTINCT token posts the slot once
		// (per-document). The eq index below still indexes a ValueString as one
		// whole-string key; the two are independent. ValueStrings only lives here
		// (scalarKeyOf declines slices). ValueRecord considered: no case, so it
		// correctly declines tokenization like every other non-string kind.
		switch v.Kind {
		case ValueString:
			slotTokens = p.addTokens(field, tokenize(v.Str), slot, slotTokens)
		case ValueStrings:
			var toks []string
			for _, s := range v.Strs {
				toks = append(toks, tokenize(s)...)
			}
			slotTokens = p.addTokens(field, toks, slot, slotTokens)
		}
		// Contains index: an array field (strings/ints/floats) posts the slot under
		// each DISTINCT element key (same scalarKey as eq/compileContains). Arrays
		// decline the scalar eq index below (scalarKeyOf=false), so they live here
		// (+ the token index for string arrays). Mirrors the token discipline.
		if ek := elementKeysOf(v); len(ek) > 0 {
			slotContains = p.addContains(field, ek, slot, slotContains)
		}
		key, ok := scalarKeyOf(v)
		if !ok {
			continue
		}
		if p.postScalar(field, key, slot) {
			keys = append(keys, fieldKey{field, key})
		}
	}
	if keys != nil {
		p.slotKeys[slot] = keys
	}
	if geoCells != nil {
		p.geoSlotCells[slot] = geoCells
	}
	if slotTokens != nil {
		p.tokenSlotKeys[slot] = slotTokens
	}
	if slotContains != nil {
		p.containsSlotKeys[slot] = slotContains
	}
	if badKeys != nil {
		p.badSlotKeys[slot] = badKeys
	}
}

// addBadRecord records that one more live slot holds an unindexable record
// under payloadKey. Must hold the owner's write lock.
func (p *payloadIndex) addBadRecord(payloadKey string) {
	p.badRecords[payloadKey]++
}

// dropBadRecord removes one slot's contribution to payloadKey's unindexable
// count, deleting the entry at zero so the key leaves nothing behind and
// poisoned() is a plain map probe. Must hold the owner's write lock.
func (p *payloadIndex) dropBadRecord(payloadKey string) {
	if n := p.badRecords[payloadKey]; n > 1 {
		p.badRecords[payloadKey] = n - 1
	} else {
		delete(p.badRecords, payloadKey)
	}
}

// postScalar posts slot under (field, key) in the eq index, maintaining the
// distinct-key bookkeeping (sorted-cache invalidation and the per-kind slot
// counter). It reports whether the entry was NEW for this slot, i.e. whether
// the caller must record a reverse entry for it.
//
// THE DUPLICATE CHECK IS THE ONE-KEY-PER-(SLOT, FIELD) INVARIANT, which two
// separate proofs in this file rest on: posts[field] counts SLOTS (see
// fieldTotalNumeric), and two distinct keys of one field have DISJOINT posting
// sets (see unionRange's key-count pre-check). A literal field cannot violate
// it — reindex walks `meta` once per field — but a record CAN: two entries of
// one record can carry the same synthetic name (a table "b" and a scalar
// literally named "b#count" both spell "b#count"). Both resolve through
// lookupPath to the SAME value, so the second post is a pure duplicate; it is
// dropped here rather than double-counted. reindex has already dropped this
// slot's previous keys by the time anything is posted, so a slot found in the
// set can only have been put there by this same call.
func (p *payloadIndex) postScalar(field string, key scalarKey, slot uint32) bool {
	vals := p.fields[field]
	if vals == nil {
		vals = make(map[scalarKey]map[uint32]struct{})
		p.fields[field] = vals
	}
	set := vals[key]
	if set == nil {
		set = make(map[uint32]struct{})
		vals[key] = set
		p.markDirty(field) // new distinct key -> sorted cache stale
	}
	if _, dup := set[slot]; dup {
		return false
	}
	set[slot] = struct{}{}
	p.addPost(field, key.kind, 1)
	return true
}

// appendRecordIndexKeys appends the synthetic eq entries a ValueRecord payload
// key contributes to the index: for each top-level field and table row count
// record.Resolver.IndexEntries enumerates, the pair (payloadKey + "/" + entry
// name, that field's scalar key). Shared by both reindex implementations (the
// slot-keyed dense index and the id-keyed named/MV mirror) so the two can
// never index different things.
//
// THE VALUE COMES FROM lookupPath, NOT FROM THE ENTRY. IndexEntries is used
// only to ENUMERATE the names; what is posted under each name is whatever
// lookupPath resolves for it — which is, verbatim, the function the compiled
// predicate reads the field through. That makes the exactness invariant
// ("the posting set is exactly the slots whose lookupPath value matches")
// true by construction rather than by an argument about two independent
// walks agreeing, and it settles every awkward corner for free:
//
//   - Exact-key precedence. A literal payload key spelled "session/rc"
//     shadows the record's rc for lookupPath, so it must shadow it here too.
//     The literal is indexed by reindex's own loop under the SAME field name,
//     so the synthetic entry is skipped outright rather than posted twice.
//   - Ambiguous spellings. A record field named "b#count" and a table named
//     "b" produce the same synthetic name; a schema carrying two fields of the
//     same name (possible from hostile bytes — DecodeSchema does not Validate)
//     produces the same name twice. lookupPath answers ONE value for the name
//     either way, and that one value is what gets posted.
//   - Malformed records. IndexEntries errors and whole == false; the caller
//     poisons the payload key, because a PARTIALLY damaged record still
//     resolves its intact fields (see recordPoison — this is the fix-round-1
//     CRITICAL, and the reason this function reports the failure rather than
//     swallowing it).
//   - NaN. IndexEntries skips a NaN float and scalarKeyOf declines one, so it
//     is unindexed on both counts; the comparators reject NaN, so "unindexed"
//     and "matches nothing" say the same thing (see scalarKeyOf).
//
// The cost is one record resolution per enumerated entry, on the write path.
// For the records this is built for (a handful of fields over a cached
// schema) that is a few map probes; it scales with (entries × record size)
// for a pathologically wide record.
func appendRecordIndexKeys(acc []fieldKey, meta Metadata, payloadKey string, rec []byte) (out []fieldKey, whole bool) {
	entries, err := recordResolver.IndexEntries(rec)
	if err != nil {
		// The record could not be enumerated IN FULL. That is not the same as
		// "it holds nothing": IndexEntries fails if ANY field is damaged, while
		// Resolve reads only the bytes on its own path and keeps answering for
		// the intact ones. The caller must fail the whole payload key closed —
		// see recordPoison — because there is no partial posting set that could
		// be trusted.
		return acc, false
	}
	for i := range entries {
		name := payloadKey + "/" + entries[i].Field
		if !recordPathIndexedShape(name) {
			continue // a shape the planner will never trust; posting it would be dead weight
		}
		if _, literal := meta[name]; literal {
			continue // an exact payload key of this name wins, and is indexed as itself
		}
		v, ok := lookupPath(meta, name)
		if !ok {
			continue
		}
		key, ok := scalarKeyOf(v)
		if !ok {
			continue
		}
		acc = append(acc, fieldKey{name, key})
	}
	return acc, true
}

// addContains posts slot under each (already DISTINCT) element key in keys within
// field's contains index, appending the new (field, key) reverse entries to acc
// (so reindex can drop them on slot reuse). elementKeysOf dedups within the
// value, so each (field, key) reverse entry is distinct — keeping dropContains'
// per-entry delete exact. The contains analogue of addTokens.
func (p *payloadIndex) addContains(field string, keys []scalarKey, slot uint32, acc []fieldKey) []fieldKey {
	byKey := p.contains[field]
	if byKey == nil {
		byKey = make(map[scalarKey]map[uint32]struct{})
		p.contains[field] = byKey
	}
	for _, k := range keys {
		set := byKey[k]
		if set == nil {
			set = make(map[uint32]struct{})
			byKey[k] = set
		}
		set[slot] = struct{}{}
		acc = append(acc, fieldKey{field, k})
	}
	return acc
}

// dropContains removes slot from the contains posting list for fk, cleaning up
// empty maps. The contains analogue of drop/dropToken.
func (p *payloadIndex) dropContains(fk fieldKey, slot uint32) {
	byKey := p.contains[fk.field]
	if byKey == nil {
		return
	}
	if set := byKey[fk.key]; set != nil {
		delete(set, slot)
		if len(set) == 0 {
			delete(byKey, fk.key)
		}
	}
	if len(byKey) == 0 {
		delete(p.contains, fk.field)
	}
}

// addTokens posts slot under each DISTINCT token of toks within field's token
// index, appending the new (field, token) reverse entries to acc (so reindex can
// drop them on slot reuse). Duplicate tokens (repeated across a ValueStrings, or
// recurring in one string) post once — both the posting set (already a set) and
// the reverse list stay distinct, keeping dropToken's per-entry delete exact.
func (p *payloadIndex) addTokens(field string, toks []string, slot uint32, acc []fieldToken) []fieldToken {
	if len(toks) == 0 {
		return acc
	}
	byTok := p.tokens[field]
	if byTok == nil {
		byTok = make(map[string]map[uint32]struct{})
		p.tokens[field] = byTok
	}
	for _, tok := range toks {
		set := byTok[tok]
		if set == nil {
			set = make(map[uint32]struct{})
			byTok[tok] = set
		}
		if _, dup := set[slot]; dup {
			continue // already posted this slot under this token -> distinct
		}
		set[slot] = struct{}{}
		acc = append(acc, fieldToken{field, tok})
	}
	return acc
}

// dropToken removes slot from the token posting list for (field, token),
// cleaning up empty maps. The token analogue of drop/dropGeo.
func (p *payloadIndex) dropToken(ft fieldToken, slot uint32) {
	byTok := p.tokens[ft.field]
	if byTok == nil {
		return
	}
	if set := byTok[ft.token]; set != nil {
		delete(set, slot)
		if len(set) == 0 {
			delete(byTok, ft.token)
		}
	}
	if len(byTok) == 0 {
		delete(p.tokens, ft.field)
	}
}

// dropGeo removes slot from the geo posting list for (field, cell), cleaning up
// empty maps. The geo analogue of drop.
func (p *payloadIndex) dropGeo(gc geoSlotCell, slot uint32) {
	cells := p.geo[gc.field]
	if cells == nil {
		return
	}
	if set := cells[gc.cell]; set != nil {
		delete(set, slot)
		if len(set) == 0 {
			delete(cells, gc.cell)
		}
	}
	if len(cells) == 0 {
		delete(p.geo, gc.field)
	}
}

// drop removes slot from the posting list for fk, cleaning up empty maps.
func (p *payloadIndex) drop(fk fieldKey, slot uint32) {
	vals := p.fields[fk.field]
	if vals == nil {
		return
	}
	if set := vals[fk.key]; set != nil {
		// Probe before deleting: `delete` on an absent slot is a silent no-op, and
		// decrementing on one would drift the posting counter UPWARD relative to
		// reality — the one direction that can make fieldTotalNumeric claim a
		// coverage the index does not have.
		if _, present := set[slot]; present {
			delete(set, slot)
			p.addPost(fk.field, fk.key.kind, -1)
		}
		if len(set) == 0 {
			delete(vals, fk.key)
			p.markDirty(fk.field) // distinct key removed -> sorted cache stale
		}
	}
	if len(vals) == 0 {
		delete(p.fields, fk.field)
	}
}

// rebuild discards and reconstructs the index from EVERY slot the arena's idMap
// holds — tombstoned included, see the loop below. Used after Reclaim and
// Restore. Must hold the lock.
func (p *payloadIndex) rebuild(a *arena) {
	p.fields = make(map[string]map[scalarKey]map[uint32]struct{})
	p.slotKeys = make(map[uint32][]fieldKey)
	p.sorted = make(map[string]*sortedKeys) // drop stale sorted caches
	p.geo = make(map[string]map[geoCell]map[uint32]struct{})
	p.geoSlotCells = make(map[uint32][]geoSlotCell)
	p.tokens = make(map[string]map[string]map[uint32]struct{})
	p.tokenSlotKeys = make(map[uint32][]fieldToken)
	p.contains = make(map[string]map[scalarKey]map[uint32]struct{})
	p.containsSlotKeys = make(map[uint32][]fieldKey)
	p.posts = make(map[string]fieldPosts) // reindex below refills it
	// Reset the malformed-record counters too, or a key poisoned before the
	// rebuild would stay poisoned by slots that no longer exist. reindex below
	// refills them from the metadata that actually survived.
	p.badRecords = make(recordPoison)
	p.badSlotKeys = make(map[uint32][]string)
	p.dropColumns() // drop the column sidecar; the query path rebuilds it lazily
	// EVERY SLOT IN idMap, TOMBSTONED OR NOT. This used to skip tombstoned slots,
	// which looked like a tidy saving and was actually a divergence: the
	// INCREMENTAL path (Delete tombstones without touching the index) leaves them
	// indexed, so a rebuilt index covered strictly less than an equivalent
	// never-restarted one. Nothing depended on the narrower set — every consumer
	// re-checks liveness before admitting a candidate, which is what makes a stale
	// posting harmless in the first place — but one thing depended on the two
	// agreeing: fieldTotalNumeric compares the per-field posting count against
	// arena.Size(), which counts idMap and therefore INCLUDES tombstones. Under
	// the old rebuild the count could never reach it, so a single tombstone at
	// snapshot time silently disabled the complement gate for every field, on
	// every restore, permanently and with no counter to notice by.
	//
	// Covering idMap exactly makes "the payload index covers arena.idMap" true in
	// both states, which is the invariant fieldTotalNumeric's proof is written
	// against and TestPayloadIndexHasNoSlotsOutsideTheIdMap pins from the other
	// side. Reclaim, the other rebuild caller, has already emptied the tombstone
	// set and removed those ids from idMap, so nothing changes for it.
	for _, slot := range a.idMap {
		p.reindex(slot, a.Metadata(slot))
	}
}

// isEmpty reports whether the index holds nothing at all — no postings, no
// derived state. Used by BuildConcurrent to assert the precondition its bulk
// placement loop relies on (it writes slots without reindexing them, which is
// only sound on an index with nothing to keep in sync). A nil receiver is empty,
// so the sub-index families that never build one need no special case.
func (p *payloadIndex) isEmpty() bool {
	if p == nil {
		return true
	}
	return len(p.fields) == 0 && len(p.slotKeys) == 0 && len(p.geo) == 0 &&
		len(p.tokens) == 0 && len(p.contains) == 0 && len(p.posts) == 0 && len(p.cols) == 0 &&
		len(p.badRecords) == 0 && len(p.badSlotKeys) == 0
}

// emptySlotSet is a shared read-only sentinel returned when a field has no
// postings; callers only ever read it (intersection -> empty result).
var emptySlotSet = map[uint32]struct{}{}

// indexNarrowable reports whether `field` can be narrowed by the payload index
// at all.
//
// THE EMPTY-SET INFERENCE, AND THE FIELDS THAT BREAK IT. Every narrowing
// path answers a missing field with the empty sentinel, and that is normally
// sound: a field no document carries as an indexable value is a field no
// document can MATCH on, so "no postings" and "no matches" are the same
// statement. The inference depends on the index having TRIED to index the
// field. Two field shapes never get that try:
//
// contentField. reindex skips $content by design (a document body is not a
// filter key, and equality-indexing megabytes of text would be absurd), so its
// posting maps are permanently empty — while the compiled predicate reads
// $content perfectly well, because content is stored as an ordinary entry in
// the slot's metadata map and lookupPath finds it like any other key. Only the
// RETURN path strips it (fetchDocs / docForLocked). So for $content alone, "no
// postings" means "never indexed", not "no matches", and treating the empty
// set as a superset silently drops every matching row.
//
// A record path (a "payloadKey/path" field naming a location inside a
// ValueRecord payload key, per sdk/record) can break the SAME inference the
// SAME way, and used to break it for every shape: reindex posted only LITERAL
// scalar keys, so p.fields["session/rc"] was always nil even though lookupPath
// resolves "session/rc" through the record perfectly well at
// predicate-evaluation time.
//
// reindex now expands a record's top-level fields and table row counts into
// synthetic postings named "<payloadKey>/<field>" (appendRecordIndexKeys), so
// SOME record paths have postings and some do not — and the ops differ too,
// because a record scalar gets an eq key and nothing else. THREE things decide
// it, in this order:
//
//  1. the payload key must not be POISONED (recordPoison): a live slot holding
//     a record IndexEntries could not enumerate makes every path under that key
//     unprovable, because the damaged record still resolves its intact fields;
//  2. the SHAPE must be one reindex posts (recordShapeIndexed);
//  3. the OP's posting set must be exact for that shape (recordOpIndexed).
//
// filterIndexExact and columnExpressible route through this same function, so
// no consumer can disagree with the writer about what was indexed. `op` is the
// leaf's op (already lowered to the plain ordering spelling where a caller
// lowers FilterDt*, which is harmless: both spellings are narrowable).
//
// A KNOWN, ACCEPTED PHASE-1 COST, AND IT IS NOT UNIFORM ACROSS SHAPES. A
// LITERAL payload key that merely LOOKS like a record path loses acceleration
// here even though reindex builds its tokens, contains entries and geo cells
// exactly as for any other literal key. HOW MUCH it loses depends on what its
// tail parses as, because the shape guard runs before the op guard:
//
//   - ONE INDEXED SEGMENT ("a/b", "a/b#count"): recordShapeIndexed passes, so
//     the op guard decides — eq / in / range still accelerate, and only
//     match / contains / geo are declined.
//   - ANY OTHER PARSEABLE PATH — a longer path ("metrics/q1/42", naming a row
//     or a cell) or a positional segment ("a/#0") — fails recordShapeIndexed,
//     so EVERY op is declined, eq and range included. These shapes have no
//     postings at all, synthetic or otherwise, so there is nothing to be exact
//     about.
//   - A TAIL THAT IS NOT A VALID PATH ("metrics/2024/q1": "2024" is a legal
//     field but "q1" is not a legal row key) never reaches either guard — the
//     ParsePath failure above returns true and the key narrows like any other
//     literal key, at full acceleration.
//
// The decision is per FIELD NAME, and a field name cannot tell the two apart:
// whether "a/b" is a literal key or a path into a record under "a" is a
// per-POINT fact, and one collection may hold both. Declining costs a graph
// fallback (and some dead index weight); guessing the other way would cost
// correctness on the synthetic side. Making this exact would need a per-field
// record of how each name was posted, which is Phase 2.
//
// Declining is always safe: an un-narrowed conjunct is simply re-checked by
// the predicate, and a filter that narrows on nothing else falls back to graph
// traversal, which evaluates $content and every record path correctly.
func indexNarrowable(field string, op FilterOp, poison recordPoison) bool {
	if field == contentField {
		return false
	}
	// ONE parse decides everything below: whether field is a record path at
	// all, which payload key it hangs off, and which shape it names.
	payloadKey, path, ok := record.SplitField(field)
	if !ok {
		return true // no '/': an ordinary literal key, indexed as it always was
	}
	// A record hanging off $content is invisible to the index for the SAME
	// reason $content itself is: reindex skips the contentField entry before it
	// reaches the ValueRecord branch, so no synthetic posting is ever written
	// for any path under it. Without this, "$content/rc" splits, parses, and
	// satisfies both recordShapeIndexed and recordOpIndexed, so the planner
	// grades a permanently EMPTY posting set as exact and filter-first drops
	// every matching row — the bare-$content hole one level deeper. Declining
	// here (rather than teaching reindex to expand records under $content)
	// keeps "$content is never a filter key" whole and keeps ONE record
	// expansion site.
	if payloadKey == contentField {
		return false
	}
	p, err := record.ParsePath(path)
	if err != nil {
		// A literal key that merely CONTAINS a '/' but is not a valid path
		// ("a/b/rc" with a non-numeric row segment). lookupPath resolves it by
		// exact match only, and reindex posts it by exact match only, so the
		// two agree and it narrows like any other literal key.
		return true
	}
	if poison.poisoned(payloadKey) {
		return false
	}
	return recordShapeIndexed(p) && recordOpIndexed(op)
}

// candidates returns (slots, true) when filter can be narrowed by the payload
// index — the slots are a SUPERSET of the filter's matching live slots, so the
// caller must still apply the full predicate. Returns (nil, false) when the
// filter is not index-narrowable (e.g. Or/Not/Ne/Contains, or a non-selective
// range), signalling the caller to fall back to graph search. `limit` caps the
// work the range path will do before giving up. Must hold the owner's read lock.
func (p *payloadIndex) candidates(f Filter, limit int) ([]uint32, bool) {
	return p.candidatesCapped(f, limit, math.MaxInt)
}

// candidatesCapped is candidates with an additional early-abort ceiling: the
// final merged candidate set is abandoned (ok=false) the moment its size would
// exceed maxCand, without finishing the materialization. maxCand must be <=
// limit for the abort to ever trigger tighter than limit already does;
// candidates calls this with maxCand = math.MaxInt, so it never aborts early
// and is byte-identical to the uncapped behavior. Used by the filter-first
// planner (searchIntoWith), which already knows the largest candidate count it
// could ever act on and so has no use for a superset that blows past it.
func (p *payloadIndex) candidatesCapped(f Filter, limit, maxCand int) ([]uint32, bool) {
	sets, ok := p.collectNarrowSets(f, limit)
	if !ok {
		return nil, false
	}
	return intersectSlotSets(sets, maxCand)
}

// narrowClass grades one posting set against the predicate that will re-check
// it. The planner does not care — it re-checks everything — but the traversal
// admission gate does: a gate that PRE-REJECTS on a missing bit is only sound if
// the set is a proven superset, and a gate that SKIPS the predicate entirely is
// only sound if the set is exact.
type narrowClass uint8

const (
	// narrowExact: the set equals the predicate's live match set. See
	// filterIndexExact for the per-op derivation (Eq / Contains / In).
	narrowExact narrowClass = iota
	// narrowSuper: the set is a PROVEN superset — it may contain slots the
	// predicate rejects, never the reverse. Geo (a cell cover of a bounding box)
	// and Match (per-document token postings vs. per-element matching).
	//
	// The proof rests on the field having been INDEXED. It does not hold for
	// contentField, whose posting maps are empty because reindex refuses to fill
	// them, not because nothing matches — an empty set graded narrowSuper there
	// would be a superset of nothing. That case never reaches this grading:
	// indexNarrowable declines $content at every posting-set lookup, so no
	// $content set is ever built to be graded. Keep it that way — grading is the
	// wrong layer to fix it at, because the planner consumes these sets too and
	// would still have been handed the empty one.
	narrowSuper
	// narrowUnproven: the set narrows in the common case but its superset
	// property is NOT proven, so it may only be used where a wrong answer is
	// re-checked away — i.e. by the planner, never by the gate.
	//
	// NO OP PRODUCES THIS GRADE TODAY, and the history is worth keeping because
	// it is the reason the grade exists. The ordering family (Gt/Gte/Lt/Lte and
	// the FilterDt* lowering) lived here: orderingSet binary-searches a sorted key
	// cache, while compareFloat classified NaN as EQUAL to every bound, so the
	// predicate ACCEPTED a NaN-valued field for Gte/Lte that the binary search
	// walked straight past. The index answered the NARROWER question — the one
	// direction a pre-filtering gate cannot survive.
	//
	// m5 closed that at the root rather than working around it: NaN is now
	// unordered for range ops (orderingHoldsFloat) and unindexable as a key
	// (scalarKeyOf), so predicate and index agree exactly and the ordering family
	// is narrowExact. The grade stays because the SURVEY that consumes it
	// (buildAdmitGate) is the safety net for the next op family whose posting set
	// is merely plausible — dropping an unproven set only widens the gate, which
	// is always safe, and having the escape hatch already wired is what makes it
	// cheap to add such an op later.
	narrowUnproven
)

// narrowSet is one leaf's posting set with its grade.
type narrowSet struct {
	set   map[uint32]struct{}
	class narrowClass
}

// tagSets grades a helper's raw output uniformly. Every collect* helper covers
// exactly one op family, so one class per helper is the whole mapping.
func tagSets(sets []map[uint32]struct{}, class narrowClass) []narrowSet {
	out := make([]narrowSet, 0, len(sets))
	for _, s := range sets {
		out = append(out, narrowSet{set: s, class: class})
	}
	return out
}

// collectNarrowSets is the narrowing PLAN: the posting sets whose intersection is
// the candidate superset, WITHOUT materializing that intersection. It is the
// single place the per-op narrowing precedence lives; candidatesCapped is just
// this plus intersectSlotSets, and the traversal-time admission gate
// (hnsw.buildAdmitGate) is this plus a bitset fill — so the planner and the gate
// can never disagree about which conjuncts the index answered.
//
// ok=false means the filter is not index-narrowable at all (Or/Not, a bare
// non-narrowable leaf, an overflowed selectivity cap) and the caller must fall
// back to unassisted graph traversal. ok=true with a set list means: the
// intersection of the returned sets is a superset of the filter's matching live
// slots and the predicate re-check is still the authority. Each set additionally
// carries its narrowClass, which is what a caller weaker than "re-check
// everything" (the admission gate) must consult before trusting it.
//
// EXACTLY ONE SET IS APPENDED PER COVERED LEAF. That invariant is what lets a
// caller decide EXACTNESS by counting: len(sets) == filterLeafCount(f) means no
// conjunct went un-narrowed, so (for ops whose posting set is exact rather than
// merely a superset — see filterIndexExact) the intersection IS the match set.
// Every collect* helper below already appends one set per leaf it covers; do not
// break that by folding two leaves into one set.
//
// A missing field / absent posting key short-circuits with the shared empty
// sentinel appended, which makes the intersection empty — the same answer, and
// the same skipped work, as the early `return []uint32{}, true` this replaced.
func (p *payloadIndex) collectNarrowSets(f Filter, limit int) ([]narrowSet, bool) {
	// Geo narrowing: a top-level geo op (or an And containing one) reduces its
	// query region to a bbox, enumerates the covering geohash cells, and unions
	// their posting lists into a superset the predicate re-checks. Tried first so
	// And(geo, eq/range) narrows on the geo conjunct. Bails (ok=false) on a region
	// too large to cover within the cell cap → graph traversal stays correct.
	if geoSets, ok := p.collectGeoSets(f, limit); ok && len(geoSets) > 0 {
		sets := tagSets(geoSets, narrowSuper)
		// Intersect the geo sets with any equality sets in the same And so the most
		// selective conjunct(s) drive the candidate set; the predicate re-checks
		// everything else. A pure geo filter just returns the geo union.
		if eqTerms, ok := collectEqTerms(f, p.badRecords); ok {
			for _, t := range eqTerms {
				vals := p.fields[t.field]
				if vals == nil {
					return append(sets, narrowSet{emptySlotSet, narrowExact}), true
				}
				set := vals[t.key]
				if set == nil {
					return append(sets, narrowSet{emptySlotSet, narrowExact}), true
				}
				sets = append(sets, narrowSet{set, narrowExact})
			}
		}
		// Fold any Match conjunct sets in too, so And(geo, match) narrows on both.
		if matchSets, ok := p.collectMatchSets(f, limit); ok && len(matchSets) > 0 {
			sets = append(sets, tagSets(matchSets, narrowSuper)...)
		}
		// Fold any Contains conjunct sets in too, so And(geo, contains) narrows on both.
		if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
			sets = append(sets, tagSets(containsSets, narrowExact)...)
		}
		return sets, true
	}

	// Equality narrowing. Also covers And(eq, range): the eq terms narrow and
	// the predicate re-checks the range conjuncts, so range terms need no index
	// here. Unchanged fast path.
	if terms, ok := collectEqTerms(f, p.badRecords); ok {
		sets := make([]narrowSet, 0, len(terms))
		for _, t := range terms {
			vals := p.fields[t.field]
			if vals == nil {
				// no slot carries this field -> empty match
				return append(sets, narrowSet{emptySlotSet, narrowExact}), true
			}
			set := vals[t.key]
			if set == nil {
				return append(sets, narrowSet{emptySlotSet, narrowExact}), true
			}
			sets = append(sets, narrowSet{set, narrowExact})
		}
		// Fold any Match conjunct sets in too, so And(eq, match) narrows on both.
		if matchSets, ok := p.collectMatchSets(f, limit); ok && len(matchSets) > 0 {
			sets = append(sets, tagSets(matchSets, narrowSuper)...)
		}
		// Fold any Contains conjunct sets in too, so And(eq, contains) narrows on both.
		if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
			sets = append(sets, tagSets(containsSets, narrowExact)...)
		}
		return sets, true
	}

	// Match narrowing: a top-level Match leaf, or an And whose only narrowable
	// conjuncts are Match terms (no eq/geo to lean on above). Each Match term's
	// query tokens are intersected within its field; the predicate re-checks.
	if matchSets, ok := p.collectMatchSets(f, limit); ok && len(matchSets) > 0 {
		sets := tagSets(matchSets, narrowSuper)
		// Fold any range/In conjunct sets in so And(match, range) narrows on both.
		if rangeSets, ok := p.collectRangeSets(f, limit); ok {
			sets = append(sets, rangeSets...)
		}
		// Fold any Contains conjunct sets in so And(match, contains) narrows on both.
		if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
			sets = append(sets, tagSets(containsSets, narrowExact)...)
		}
		return sets, true
	}

	// Contains narrowing: a top-level Contains leaf, or an And whose only
	// narrowable conjuncts are Contains terms (no eq/geo/match to lean on above).
	// Each Contains term's element posting list is intersected within the And; the
	// predicate re-checks. Folds any range/In conjunct sets in too.
	if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
		sets := tagSets(containsSets, narrowExact)
		if rangeSets, ok := p.collectRangeSets(f, limit); ok {
			sets = append(sets, rangeSets...)
		}
		return sets, true
	}

	// Range-only / In-only narrowing: no equality conjunct to lean on, so build
	// candidate sets directly from the (reused) posting lists.
	sets, ok := p.collectRangeSets(f, limit)
	if !ok || len(sets) == 0 {
		return nil, false
	}
	return sets, true
}

// intersectSlotSets returns (the intersection of the given posting sets, true)
// as a slice, scanning the smallest set and probing the rest — UNLESS the
// intersection would exceed maxCand, in which case it abandons the scan as soon
// as that's known and returns (nil, false) instead of finishing the
// materialization. A single-set intersection is just that set, so its size is
// known from the map length alone: decide against maxCand before copying
// anything.
// The planner re-checks every candidate with the full predicate, so it ignores
// narrowClass entirely and intersects the whole plan — including the sets the
// gate refuses to touch.
func intersectSlotSets(sets []narrowSet, maxCand int) ([]uint32, bool) {
	if len(sets) == 0 {
		return []uint32{}, true
	}
	if len(sets) == 1 {
		if len(sets[0].set) > maxCand {
			return nil, false
		}
		out := make([]uint32, 0, len(sets[0].set))
		for slot := range sets[0].set {
			out = append(out, slot)
		}
		return out, true
	}
	smallest := 0
	for i := 1; i < len(sets); i++ {
		if len(sets[i].set) < len(sets[smallest].set) {
			smallest = i
		}
	}
	out := make([]uint32, 0, len(sets[smallest].set))
	for slot := range sets[smallest].set {
		inAll := true
		for i, s := range sets {
			if i == smallest {
				continue
			}
			if _, ok := s.set[slot]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			out = append(out, slot)
			if len(out) > maxCand {
				return nil, false
			}
		}
	}
	return out, true
}

// collectEqTerms extracts equality terms that constrain the whole filter: a
// single Eq, or the Eq conjuncts of a top-level And (recursing into nested
// Ands). Non-Eq conjuncts (ranges, etc.) are ignored — they are additional
// constraints the caller re-checks via the predicate, so the collected terms'
// intersection remains a superset of the matching set. Returns ok=false when
// the filter is not an And/Eq narrowing shape (Or, Not, range-only, ...).
func collectEqTerms(f Filter, poison recordPoison) ([]fieldKey, bool) {
	switch f.Op {
	case FilterEq:
		if !indexNarrowable(f.Field, FilterEq, poison) {
			return nil, false
		}
		if key, ok := scalarKeyOf(f.Value); ok {
			return []fieldKey{{f.Field, key}}, true
		}
		return nil, false
	case FilterAnd:
		var acc []fieldKey
		for _, c := range f.And {
			switch c.Op {
			case FilterEq:
				if !indexNarrowable(c.Field, FilterEq, poison) {
					continue // $content is never indexed; the predicate re-checks it
				}
				if key, ok := scalarKeyOf(c.Value); ok {
					acc = append(acc, fieldKey{c.Field, key})
				}
			case FilterAnd:
				if t, ok := collectEqTerms(c, poison); ok {
					acc = append(acc, t...)
				}
			}
		}
		return acc, len(acc) > 0
	default:
		return nil, false
	}
}

// collectGeoSets gathers the geohash-index slot sets for the geo leaves of a
// filter: a single geo op, or the geo conjuncts of a top-level And (recursing
// into nested Ands). Each geo leaf reduces its region to a bbox, enumerates the
// covering cells, and unions their posting lists into ONE superset set. Non-geo
// conjuncts are ignored here (the eq path in candidates and the predicate
// re-check them). ok=false means either nothing was geo, OR a geo leaf
// overflowed the cover-cell cap / hit a pole/antimeridian (bail to graph) — in
// that case we decline entirely rather than return an under-covered set, because
// a geo leaf whose set is dropped would leave the predicate's geo constraint
// un-narrowed yet still applied, which is correct, BUT dropping it could let a
// non-selective region masquerade as selective. Declining keeps the overflow
// bail honest (huge radius → graph traversal).
func (p *payloadIndex) collectGeoSets(f Filter, limit int) ([]map[uint32]struct{}, bool) {
	switch f.Op {
	case FilterGeoRadius, FilterGeoBox, FilterGeoPolygon:
		set, ok := p.geoSet(f.Field, f.Op, f.Geo, limit)
		if !ok {
			return nil, false
		}
		return []map[uint32]struct{}{set}, true
	case FilterAnd:
		var acc []map[uint32]struct{}
		found := false
		for i := range f.And {
			switch f.And[i].Op {
			case FilterGeoRadius, FilterGeoBox, FilterGeoPolygon:
				set, ok := p.geoSet(f.And[i].Field, f.And[i].Op, f.And[i].Geo, limit)
				if !ok {
					return nil, false // a geo conjunct overflowed/bailed → graph
				}
				acc = append(acc, set)
				found = true
			case FilterAnd:
				// A nested And contributes its geo narrowing if it has any; a
				// non-geo nested And (or an overflowed one) returns ok=false and is
				// simply skipped here. Skipping is correctness-safe: the outer set
				// stays a superset (the predicate re-checks the nested constraints),
				// and we do NOT bail the whole geo path — ok=false is ambiguous
				// between "no geo here" and "overflowed", and over-bailing would
				// drop a sibling geo leaf's valid narrowing.
				if sets, ok := p.collectGeoSets(f.And[i], limit); ok && len(sets) > 0 {
					acc = append(acc, sets...)
					found = true
				}
			}
		}
		return acc, found
	default:
		return nil, false
	}
}

// collectMatchSets gathers the token-index slot sets for the Match leaves of a
// filter: a single FilterMatch op, or the Match conjuncts of a top-level And
// (recursing into nested Ands). Each Match leaf intersects its query tokens'
// postings within its field into ONE superset set (match-ALL ⇒ intersect). The
// returned sets are each a SUPERSET of that Match's predicate matches (per-
// document postings ⊇ per-element matches; tokenize is identical to the
// predicate's), so intersecting them with the caller's other conjunct sets and
// re-checking the predicate stays correct.
//
// Bail/skip semantics mirror collectGeoSets:
//   - A bare Match LEAF that is not narrowable (empty query tokens, or the
//     selectivity cap overflowed) returns ok=false — candidates then declines
//     the match path and lets the predicate scan handle it.
//   - A Match CONJUNCT of an And that is not narrowable is simply SKIPPED (not a
//     whole-And bail): the outer set stays a superset because the predicate
//     re-checks the skipped constraint, and a sibling conjunct may still narrow.
//     Skipping an empty-token Match is correct: an empty match matches every
//     present field value, so it adds no constraint.
func (p *payloadIndex) collectMatchSets(f Filter, limit int) ([]map[uint32]struct{}, bool) {
	switch f.Op {
	case FilterMatch:
		set, ok := p.matchSet(f.Field, f.Value, limit)
		if !ok {
			return nil, false
		}
		return []map[uint32]struct{}{set}, true
	case FilterAnd:
		var acc []map[uint32]struct{}
		found := false
		for i := range f.And {
			switch f.And[i].Op {
			case FilterMatch:
				set, ok := p.matchSet(f.And[i].Field, f.And[i].Value, limit)
				if !ok {
					continue // non-narrowable match conjunct -> predicate re-checks
				}
				acc = append(acc, set)
				found = true
			case FilterAnd:
				if sets, ok := p.collectMatchSets(f.And[i], limit); ok && len(sets) > 0 {
					acc = append(acc, sets...)
					found = true
				}
			}
		}
		return acc, found
	default:
		return nil, false
	}
}

// containsSet builds the candidate superset for one Contains op via the contains
// element index: look up contains[field][scalarKeyOf(want)] — the slots whose
// array holds an element == want. This is EXACTLY the predicate match set
// (no-false-negative: elementKeysOf posts every element compileContains would
// match, under the SAME scalarKey scalarKeyOf(want) produces, with the same kind
// carried — a string want only hits a ValueStrings element). The predicate
// re-check (the contract) drops any over-cover.
//
// ok=false (not narrowable) when: want is not a scalar (scalarKeyOf declines it —
// compileContains also rejects a non-scalar want at compile time); or the posting
// set exceeds `limit` (selectivity cap: bail to graph rather than materialize a
// near-full-corpus set — never a truncated set). A field with no contains
// postings, or a missing element key, yields the empty sentinel set (ok=true): no
// doc's array contains want, so the exact superset is empty.
func (p *payloadIndex) containsSet(field string, want Value, limit int) (map[uint32]struct{}, bool) {
	if !indexNarrowable(field, FilterContains, p.badRecords) {
		return nil, false
	}
	key, ok := scalarKeyOf(want)
	if !ok {
		return nil, false // non-scalar want -> compileContains rejects it; not narrowable
	}
	byKey := p.contains[field]
	if byKey == nil {
		return emptySlotSet, true // no slot carries an array for this field -> empty
	}
	set := byKey[key]
	if set == nil {
		return emptySlotSet, true // no array contains want -> empty match
	}
	if len(set) > limit {
		return nil, false // selectivity cap: posting list too large -> graph fallback
	}
	return set, true
}

// collectContainsSets gathers the contains-index slot sets for the Contains
// leaves of a filter: a single FilterContains op, or the Contains conjuncts of a
// top-level And (recursing into nested Ands). Each returned set is a SUPERSET of
// that Contains leaf's predicate matches. Bail/skip semantics mirror
// collectMatchSets: a bare Contains LEAF that is not narrowable (non-scalar want,
// or the selectivity cap overflowed) returns ok=false; a non-narrowable Contains
// CONJUNCT of an And is simply SKIPPED (the predicate re-checks it, and a sibling
// conjunct may still narrow).
func (p *payloadIndex) collectContainsSets(f Filter, limit int) ([]map[uint32]struct{}, bool) {
	switch f.Op {
	case FilterContains:
		set, ok := p.containsSet(f.Field, f.Value, limit)
		if !ok {
			return nil, false
		}
		return []map[uint32]struct{}{set}, true
	case FilterAnd:
		var acc []map[uint32]struct{}
		found := false
		for i := range f.And {
			switch f.And[i].Op {
			case FilterContains:
				set, ok := p.containsSet(f.And[i].Field, f.And[i].Value, limit)
				if !ok {
					continue // non-narrowable contains conjunct -> predicate re-checks
				}
				acc = append(acc, set)
				found = true
			case FilterAnd:
				if sets, ok := p.collectContainsSets(f.And[i], limit); ok && len(sets) > 0 {
					acc = append(acc, sets...)
					found = true
				}
			}
		}
		return acc, found
	default:
		return nil, false
	}
}

// matchSet builds the candidate superset for one Match op via the token index:
// tokenize the query (SAME tokenize() as compileMatch), then INTERSECT the
// posting lists of every query token within `field` (match-ALL). The intersect
// scans the smallest token's postings and probes the rest, capped at `limit`.
//
// ok=false (not narrowable) when:
//   - the query has zero tokens (an empty match matches any present value -> the
//     whole field, never a narrowing) — matches compileMatch's "empty query
//     trivially true";
//   - the field is non-string (want.Kind != ValueString — compileMatch rejects
//     it at compile time; declining here is a safety net);
//   - the intersected set would exceed `limit` (selectivity cap: a stopword
//     present in most docs would materialize a near-full-corpus set, so bail to
//     graph instead — never return a truncated set).
//
// A missing query token yields an empty set (ok=true): no doc has all tokens, so
// the exact superset is empty. The returned set is a fresh map the caller owns.
func (p *payloadIndex) matchSet(field string, want Value, limit int) (map[uint32]struct{}, bool) {
	if !indexNarrowable(field, FilterMatch, p.badRecords) {
		return nil, false
	}
	if want.Kind != ValueString {
		return nil, false
	}
	queryTokens := tokenize(want.Str)
	if len(queryTokens) == 0 {
		return nil, false // empty query -> matches any present value, not narrowable
	}
	byTok := p.tokens[field]
	if byTok == nil {
		return emptySlotSet, true // no slot carries this field -> empty match
	}
	// Gather the posting sets for the distinct query tokens; a missing token's
	// absent set makes the intersection empty (no doc has all tokens).
	sets := make([]map[uint32]struct{}, 0, len(queryTokens))
	for _, tok := range queryTokens {
		set := byTok[tok]
		if set == nil {
			return emptySlotSet, true // a query token is in no doc -> empty intersect
		}
		sets = append(sets, set)
	}
	// Intersect smallest-first (like intersectSlotSets), capped at limit so a
	// non-selective intersection bails to graph rather than materializing.
	smallest := 0
	for i := 1; i < len(sets); i++ {
		if len(sets[i]) < len(sets[smallest]) {
			smallest = i
		}
	}
	if len(sets[smallest]) > limit {
		return nil, false // even the smallest token set already exceeds limit -> bail
	}
	out := make(map[uint32]struct{})
	for slot := range sets[smallest] {
		inAll := true
		for i, s := range sets {
			if i == smallest {
				continue
			}
			if _, ok := s[slot]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			out[slot] = struct{}{}
			if len(out) > limit {
				return nil, false // selectivity cap: superset too large -> graph fallback
			}
		}
	}
	return out, true
}

// geoSet builds the candidate superset for one geo op via the geohash index:
// reduce the region to a bbox (radius → circle bbox over-estimate; box → itself;
// polygon → vertices' bbox), enumerate the covering cells, and union their
// posting lists. Returns ok=false on a malformed/nil condition (the compiler is
// the authoritative rejecter — here we just decline to narrow), on a region that
// reaches a pole / would cross the antimeridian (bail to graph), or when the
// covering-cell count or union size exceeds the cap (the overflow bail). The
// returned set is ALWAYS a superset of the true predicate match set (the
// SUPERSET INVARIANT): the bbox fully contains the region and the cover fully
// contains the bbox.
func (p *payloadIndex) geoSet(field string, op FilterOp, g *GeoCondition, limit int) (map[uint32]struct{}, bool) {
	if !indexNarrowable(field, op, p.badRecords) {
		return nil, false
	}
	if g == nil {
		return nil, false
	}
	var bbox geoBBox
	switch op {
	case FilterGeoRadius:
		b, ok := radiusBBox(g.CenterLat, g.CenterLon, g.RadiusM)
		if !ok {
			return nil, false // pole/antimeridian/degenerate → graph fallback
		}
		bbox = b
	case FilterGeoBox:
		// An antimeridian-crossing box (minLon > maxLon) is rejected at compile
		// time; declining here is just a safety net.
		if g.MinLat > g.MaxLat || g.MinLon > g.MaxLon {
			return nil, false
		}
		bbox = geoBBox{minLat: g.MinLat, minLon: g.MinLon, maxLat: g.MaxLat, maxLon: g.MaxLon}
	case FilterGeoPolygon:
		if len(g.Polygon) < 6 || len(g.Polygon)%2 != 0 {
			return nil, false
		}
		bbox = polygonBBox(g.Polygon)
	default:
		return nil, false
	}

	cells, ok := coverCells(bbox, nil)
	if !ok {
		return nil, false // too many covering cells → graph fallback
	}
	cellMap := p.geo[field]
	if cellMap == nil {
		return emptySlotSet, true // no slot carries this geo field → empty match
	}
	out := make(map[uint32]struct{})
	for _, c := range cells {
		for slot := range cellMap[c] {
			out[slot] = struct{}{}
			if len(out) > limit {
				return nil, false // candidate union too large → graph fallback
			}
		}
	}
	return out, true
}

// collectRangeSets gathers the index-narrowable leaf sets of a range/In filter:
// a single ordering/In/Eq leaf, or the conjuncts of a top-level And (recursing
// into nested Ands). Leaves that are not narrowable or whose union overflows
// `limit` are skipped — the predicate re-checks them, which only widens the
// superset. ok=false means nothing was narrowable.
//
// Every leaf this collector covers is narrowExact. The ordering leaves were
// narrowUnproven until m5 closed the NaN disagreement between orderingHoldsFloat
// and orderingSet (see narrowClass); with predicate and index answering the same
// question, an ordering posting set IS the live match set — see filterIndexExact
// for the case analysis (mixed-kind fields, absent fields, non-numeric values).
func (p *payloadIndex) collectRangeSets(f Filter, limit int) ([]narrowSet, bool) {
	switch f.Op {
	case FilterEq:
		if set, ok := p.eqSet(f.Field, f.Value); ok {
			return []narrowSet{{set, narrowExact}}, true
		}
		return nil, false
	case FilterGt, FilterGte, FilterLt, FilterLte:
		if set, ok := p.orderingSet(f.Field, f.Op, f.Value, limit); ok {
			return []narrowSet{{set, narrowExact}}, true
		}
		return nil, false
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		// Datetime range narrows EXACTLY like the numeric ordering ops: the field
		// is stored as int64 unix-ms, so we lower the RFC3339 bound to int64-ms via
		// the SAME shared helper the compiler uses (datetimeBound) and feed it to
		// orderingSet as a numeric Value under the plain ordering op. The compiler
		// is the authoritative rejecter of an invalid RFC3339 filter; here an
		// unparseable bound just declines to narrow (ok=false → graph fallback),
		// never a panic. Because the bound is parsed identically, the index set is
		// the same set the predicate re-checks — the LOWERING introduces no error
		// of its own, so it inherits orderingSet's grade, which m5 raised to
		// narrowExact.
		ms, ok := datetimeBound(f.Value)
		if !ok {
			return nil, false
		}
		if set, ok := p.orderingSet(f.Field, dtToOrdering(f.Op), NewInt(ms), limit); ok {
			return []narrowSet{{set, narrowExact}}, true
		}
		return nil, false
	case FilterIn:
		if set, ok := p.inSet(f.Field, f.Value, limit); ok {
			return []narrowSet{{set, narrowExact}}, true
		}
		return nil, false
	case FilterAnd:
		var acc []narrowSet
		for i := range f.And {
			if sets, ok := p.collectRangeSets(f.And[i], limit); ok {
				acc = append(acc, sets...)
			}
		}
		return acc, len(acc) > 0
	default:
		return nil, false
	}
}

// eqSet returns the posting set for field == v (the shared stored map, not a
// copy). ok=false when v is not equality-indexable.
func (p *payloadIndex) eqSet(field string, v Value) (map[uint32]struct{}, bool) {
	if !indexNarrowable(field, FilterEq, p.badRecords) {
		return nil, false
	}
	key, ok := scalarKeyOf(v)
	if !ok {
		return nil, false
	}
	vals := p.fields[field]
	if vals == nil {
		return emptySlotSet, true
	}
	if set := vals[key]; set != nil {
		return set, true
	}
	return emptySlotSet, true
}

// orderingSet builds the union of posting lists whose key satisfies `op` vs
// `want`. Numeric keys (int and float) compare as float64 so they interoperate,
// matching the predicate; string keys compare lexicographically. ok=false when
// want's kind can't drive the comparison, or when the union exceeds `limit`
// (signalling the caller to drop this leaf and let the predicate handle it).
func (p *payloadIndex) orderingSet(field string, op FilterOp, want Value, limit int) (map[uint32]struct{}, bool) {
	// The historical single-budget form: capping distinct keys at the same number
	// as slots is exactly the bound the pre-check derives anyway (disjoint,
	// non-empty posting sets ⇒ keys <= slots), so this is behaviour-preserving.
	return p.orderingSetCapped(field, op, want, limit, limit)
}

// orderingSetCapped is orderingSet with the two costs budgeted SEPARATELY: the
// number of SLOTS in the union (massLimit) and the number of DISTINCT KEYS
// walked to produce it (keyLimit).
//
// Splitting them exists for the complement gate, whose two build costs scale
// with different things. Reaching a key is a scalarKey hash probe into `vals`
// plus a fresh map iterator over its posting set — measured ~96ns, over 10x the
// ~8ns of handling one more posting from a set already in hand. A range over a
// UNIQUE-valued field (one posting per key) therefore costs an order of
// magnitude more than the same number of postings under a handful of enum keys,
// and a single budget cannot tell those two apart. The gate declines the first
// shape and arms on the second; see gateRangeKeyDNS.
func (p *payloadIndex) orderingSetCapped(field string, op FilterOp, want Value, massLimit, keyLimit int) (map[uint32]struct{}, bool) {
	if !indexNarrowable(field, op, p.badRecords) {
		return nil, false
	}
	vals := p.fields[field]
	if vals == nil {
		return emptySlotSet, true // no slot carries this field -> empty match
	}
	sc := p.ensureSorted(field)
	// Starts as the shared read-only sentinel so an EMPTY range (lo == hi) costs
	// no allocation at all; unionRange replaces it with a real map before it
	// writes anything, so the sentinel is never mutated.
	out := emptySlotSet

	// unionRange adds the posting sets of the in-range keys, bailing (ok=false)
	// if the union exceeds limit so a non-selective range falls back to graph.
	//
	// THE KEY-COUNT PRE-CHECK IS THE WHOLE COST OF A NON-SELECTIVE RANGE. Within
	// one field, reindex posts each slot under EXACTLY ONE key (it iterates
	// `meta` once per field), so the posting sets of two distinct keys are
	// DISJOINT — and drop deletes a key the moment its set empties, so every key
	// still in `vals` has at least one slot. The union of the keys in [lo, hi) is
	// therefore at least hi-lo elements, and hi-lo > limit proves the overflow in
	// O(1) from indices the binary search already produced.
	//
	// Skipping the proof cost a measured HALF of every non-selective filtered
	// query. The VDBBench shape — `id >= N` passing 99% of a corpus with one
	// distinct id per point — walked ~`limit` keys, and each step was a scalarKey
	// hash lookup, a fresh map ITERATOR over a one-element set, and a map insert
	// into a growing map: ~190ns per posting, ~1.9ms per query, all of it to
	// build a set the planner was always going to reject as too large. A CPU
	// profile put unionRange at 49% of the filtered query against 10% for the
	// per-candidate admission the m4 gate was built to attack.
	//
	// The check only fires when the range is wide in DISTINCT VALUES. A wide
	// range over a low-cardinality field (few keys, huge posting lists) still
	// materializes and still bails incrementally below — that shape has the
	// posting mass but not the per-key overhead, so it is the cheap one anyway.
	unionRange := func(lo, hi int, keyAt func(i int) scalarKey) bool {
		if hi-lo > keyLimit || hi-lo > massLimit {
			return false
		}
		if hi == lo {
			return true // empty range: keep the sentinel, allocate nothing
		}
		out = make(map[uint32]struct{})
		for i := lo; i < hi; i++ {
			for slot := range vals[keyAt(i)] {
				out[slot] = struct{}{}
				if len(out) > massLimit {
					return false
				}
			}
		}
		return true
	}

	if wf, ok := numericValue(want); ok {
		if wf != wf {
			// A NaN bound orders against nothing (orderingHoldsFloat), so the
			// predicate rejects every row. Declining to narrow is the conservative
			// answer — the predicate still gives the right (empty) result — and it
			// keeps this function from having to defend a binary search against a
			// probe value that is false on both sides.
			return nil, false
		}
		lo, hi := numRange(sc.num, op, wf)
		if !unionRange(lo, hi, func(i int) scalarKey { return sc.num[i].key }) {
			return nil, false
		}
		return out, true
	}
	if want.Kind == ValueString {
		lo, hi := strRange(sc.str, op, want.Str)
		if !unionRange(lo, hi, func(i int) scalarKey { return sc.str[i] }) {
			return nil, false
		}
		return out, true
	}
	return nil, false
}

// negateOrdering maps an ordering op to the op that holds exactly when it does
// not — over the ORDERED reals (or ordered strings). It is a total complement
// only because NaN is excluded from both sides: orderingHoldsFloat rejects a NaN
// operand for op AND for negateOrdering(op), and scalarKeyOf keeps NaN out of
// the index entirely, so "satisfies op" and "satisfies ¬op" partition exactly
// the slots that carry a comparable value. Before m5 this function could not
// have existed: NaN satisfied Gte and Lte simultaneously, so Gte's complement
// was not Lt.
func negateOrdering(op FilterOp) (FilterOp, bool) {
	switch op {
	case FilterGt:
		return FilterLte, true
	case FilterGte:
		return FilterLt, true
	case FilterLt:
		return FilterGte, true
	case FilterLte:
		return FilterGt, true
	}
	return 0, false
}

// collectComplementSets is collectNarrowSets' mirror image: instead of the
// posting sets whose INTERSECTION contains every matching slot, it returns the
// posting sets whose UNION is exactly the set of live slots the filter REJECTS.
// The admission gate then marks those slots and admits everything else — which
// is the only way to accelerate a filter that passes nearly everything, because
// there the accept side is the expensive one to enumerate and the reject side is
// cheap.
//
// THE SAFETY ARGUMENT, in the direction that matters. Write R(f) for the live
// slots the compiled predicate rejects and U for the union this function
// returns.
//
//   - U ⊆ R(f) is UNCONDITIONAL. Each returned set is the postings of keys
//     satisfying ¬op, so every slot in it genuinely fails its conjunct, and
//     failing one conjunct of an And is failing the And. A CLEARED bit is
//     therefore always a proven reject — the same standing that a clear bit has
//     in the ordinary (superset) gate, and it needs no preconditions at all.
//   - R(f) ⊆ U is the CONDITIONAL half, and it is what the gate's EXACT
//     admission rests on: with it, a slot that is not marked is provably not
//     rejected, so the predicate need never run. It fails for exactly one reason
//     — a slot with NO comparable value for the field (absent, wrong kind, NaN)
//     is rejected by the predicate but lives in no posting set of either sign.
//     fieldTotalNumeric / fieldTotalString is precisely the statement that no
//     such slot exists, and every leaf must pass it.
//
// SO THE BAIL DIRECTION IS ASYMMETRIC, and getting it backwards is the bug this
// design has to avoid: marking FEWER slots rejected costs only a predicate call,
// while marking MORE loses a matching point. Every failure here — a
// non-ordering leaf, a non-total field, a budget overflow — returns ok=false and
// no gate at all, never a partial union. In particular an And is ALL-OR-NOTHING:
// a partial union would still be a sound pre-reject filter, but it would break
// R(f) ⊆ U, and a complement gate that cannot admit without the predicate saves
// nothing at the pass rates it exists for.
//
// THE TWO BUDGETS HAVE DIFFERENT SCOPES, and the caller sets them accordingly
// (buildComplementGate states the same split from its side):
//
//   - massLimit is a WHOLE-QUERY budget. appendComplementSets threads a running
//     posting total across every leaf and compares it against this number, so an
//     And already pays for all of its conjuncts out of one allowance. A caller
//     that pre-divides it by the leaf count charges that division twice.
//   - keyLimit is PER LEAF. It is the O(1) `hi-lo > keyLimit` proof over one
//     leaf's own key range, answerable from two binary-search indices and with no
//     running total to compare against, so dividing it is what bounds the sum
//     across leaves.
//
// Both are hard: a leaf over budget returns ok=false, never a truncated set.
//
// NOTHING IS UNIONED HERE, and that is a deliberate 6x. The obvious
// implementation reuses orderingSet and returns one merged reject set per leaf —
// but merging means a map INSERT per posting (measured 494 dns,
// BenchmarkAdmitGateUnits/unionposting) only for buildComplement to iterate the
// result and clear a bit (another 76). The gate does not need a set: clearing a
// bit is idempotent, so a slot appearing under two keys or two conjuncts can
// simply be cleared twice. Handing back the index's OWN per-key posting maps
// skips the intermediate map entirely and takes the per-posting cost down to the
// bit write, which is what gatePostingDNS already prices. The maps are the live
// stored ones, read under the same read lock the positive gate reads its plan
// under, and never written.
func (p *payloadIndex) collectComplementSets(f Filter, liveCount, massLimit, keyLimit int) ([]map[uint32]struct{}, bool) {
	acc, _, ok := p.appendComplementSets(f, liveCount, massLimit, keyLimit, nil, 0)
	return acc, ok
}

// appendComplementSets is collectComplementSets threading the running posting
// MASS through the recursion, so an And's massLimit is spent ACROSS its
// conjuncts rather than granted to each of them. massLimit is therefore a
// whole-query budget and the caller must NOT pre-divide it by the leaf count —
// doing so charges the same division twice. keyLimit is the opposite: it is
// re-checked per leaf against that leaf's own key range (orderingPostings), so
// the caller divides it. See buildComplementGate, where both are set.
func (p *payloadIndex) appendComplementSets(f Filter, liveCount, massLimit, keyLimit int, acc []map[uint32]struct{}, mass int) ([]map[uint32]struct{}, int, bool) {
	switch f.Op {
	case FilterGt, FilterGte, FilterLt, FilterLte:
		return p.complementLeaf(f.Field, f.Op, f.Value, liveCount, massLimit, keyLimit, acc, mass)
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		// Lowered through the SAME datetimeBound the compiler uses, exactly as
		// collectRangeSets does, so the datetime spelling of a range and the plain
		// spelling complement identically.
		ms, ok := datetimeBound(f.Value)
		if !ok {
			return acc, mass, false
		}
		return p.complementLeaf(f.Field, dtToOrdering(f.Op), NewInt(ms), liveCount, massLimit, keyLimit, acc, mass)
	case FilterAnd:
		if len(f.And) == 0 {
			return acc, mass, false // matches everything; there is nothing to reject
		}
		for i := range f.And {
			var ok bool
			acc, mass, ok = p.appendComplementSets(f.And[i], liveCount, massLimit, keyLimit, acc, mass)
			if !ok {
				return acc, mass, false // all-or-nothing: see the safety argument above
			}
		}
		return acc, mass, true
	default:
		// Or/Not/Eq/In/Match/geo/... An Or's complement is an INTERSECTION of
		// complements, which a union-of-marks cannot express; Eq's complement is
		// "every key but one", which is the expensive side by construction. Both
		// are follow-ups, not oversights.
		return acc, mass, false
	}
}

// complementLeaf appends one ordering leaf's REJECT posting sets — one per
// in-range key of the NEGATED comparison — or reports ok=false if the leaf
// cannot be complemented exactly. See collectComplementSets for why every guard
// here is a hard bail.
func (p *payloadIndex) complementLeaf(field string, op FilterOp, want Value, liveCount, massLimit, keyLimit int, acc []map[uint32]struct{}, mass int) ([]map[uint32]struct{}, int, bool) {
	if !indexNarrowable(field, op, p.badRecords) {
		return acc, mass, false
	}
	neg, ok := negateOrdering(op)
	if !ok {
		return acc, mass, false
	}
	// The totality check is per KIND, because the comparison the predicate runs
	// is per kind: a numeric bound reads the field through numericValue and
	// rejects every string-valued slot, so "every live slot has a numeric value"
	// is the claim that has to hold — not "every live slot has a value".
	if wf, isNum := numericValue(want); isNum {
		if wf != wf {
			return acc, mass, false // NaN bound: rejects everything, complement is everything
		}
		if !p.fieldTotalNumeric(field, liveCount) {
			return acc, mass, false
		}
	} else if want.Kind == ValueString {
		if !p.fieldTotalString(field, liveCount) {
			return acc, mass, false
		}
	} else {
		return acc, mass, false // a want that drives neither comparison rejects everything
	}
	return p.orderingPostings(field, neg, want, massLimit, keyLimit, acc, mass)
}

// orderingPostings appends the index's OWN posting sets for the keys satisfying
// op vs want — one map per in-range key, unmerged — enforcing both budgets and
// returning the running posting mass. ok=false means over budget or not
// comparable, and it is a HARD bail: acc may hold partial results and the caller
// must discard them (a partial rejection union is unsafe to admit from).
//
// It shares numRange/strRange with orderingSetCapped, so "the keys in range" can
// never mean two different things on the two paths.
func (p *payloadIndex) orderingPostings(field string, op FilterOp, want Value, massLimit, keyLimit int, acc []map[uint32]struct{}, mass int) ([]map[uint32]struct{}, int, bool) {
	vals := p.fields[field]
	if vals == nil {
		return acc, mass, true // no slot carries this field: nothing to reject
	}
	sc := p.ensureSorted(field)
	appendRange := func(lo, hi int, keyAt func(i int) scalarKey) bool {
		if hi-lo > keyLimit {
			return false // the O(1) key-count proof, same as orderingSetCapped's
		}
		for i := lo; i < hi; i++ {
			set := vals[keyAt(i)]
			if len(set) == 0 {
				continue
			}
			mass += len(set)
			if mass > massLimit {
				return false
			}
			acc = append(acc, set)
		}
		return true
	}
	if wf, ok := numericValue(want); ok {
		if wf != wf {
			return acc, mass, false
		}
		lo, hi := numRange(sc.num, op, wf)
		if !appendRange(lo, hi, func(i int) scalarKey { return sc.num[i].key }) {
			return acc, mass, false
		}
		return acc, mass, true
	}
	if want.Kind == ValueString {
		lo, hi := strRange(sc.str, op, want.Str)
		if !appendRange(lo, hi, func(i int) scalarKey { return sc.str[i] }) {
			return acc, mass, false
		}
		return acc, mass, true
	}
	return acc, mass, false
}

// numRange returns the [lo, hi) index range of a sorted numeric key list whose
// values satisfy `op` vs `want`, via binary search. lb is the first index with
// val >= want; ub the first with val > want.
func numRange(s []numEntry, op FilterOp, want float64) (lo, hi int) {
	lb := sort.Search(len(s), func(i int) bool { return s[i].val >= want })
	ub := sort.Search(len(s), func(i int) bool { return s[i].val > want })
	switch op {
	case FilterGt:
		return ub, len(s)
	case FilterGte:
		return lb, len(s)
	case FilterLt:
		return 0, lb
	case FilterLte:
		return 0, ub
	}
	return 0, 0
}

// strRange is numRange for a sorted string key list (lexicographic).
func strRange(s []scalarKey, op FilterOp, want string) (lo, hi int) {
	lb := sort.Search(len(s), func(i int) bool { return s[i].str >= want })
	ub := sort.Search(len(s), func(i int) bool { return s[i].str > want })
	switch op {
	case FilterGt:
		return ub, len(s)
	case FilterGte:
		return lb, len(s)
	case FilterLt:
		return 0, lb
	case FilterLte:
		return 0, ub
	}
	return 0, 0
}

// inSet builds the union of posting lists for each member of an In value's
// array (strings/ints/floats). ok=false for a non-array value or when the union
// exceeds `limit`.
func (p *payloadIndex) inSet(field string, want Value, limit int) (map[uint32]struct{}, bool) {
	if !indexNarrowable(field, FilterIn, p.badRecords) {
		return nil, false
	}
	vals := p.fields[field]
	if vals == nil {
		return emptySlotSet, true
	}
	out := make(map[uint32]struct{})
	add := func(set map[uint32]struct{}) bool {
		for slot := range set {
			out[slot] = struct{}{}
			if len(out) > limit {
				return false
			}
		}
		return true
	}
	switch want.Kind {
	case ValueStrings:
		for _, s := range want.Strs {
			if set := vals[scalarKey{kind: ValueString, str: s}]; set != nil {
				if !add(set) {
					return nil, false
				}
			}
		}
		return out, true
	case ValueInts:
		for _, n := range want.Ints {
			if set := vals[scalarKey{kind: ValueInt, i: n}]; set != nil {
				if !add(set) {
					return nil, false
				}
			}
		}
		return out, true
	case ValueFloats:
		for _, x := range want.Flts {
			if set := vals[scalarKey{kind: ValueFloat, f: x}]; set != nil {
				if !add(set) {
					return nil, false
				}
			}
		}
		return out, true
	default:
		// ValueRecord considered: falls here, correctly declining — a record is
		// not an array, so it can never be a FilterIn value.
		return nil, false
	}
}
