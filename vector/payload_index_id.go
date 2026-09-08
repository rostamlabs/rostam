// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"sort"
	"sync"
)

// payloadIndexID is the id-keyed mirror of the dense slot-keyed payloadIndex.
// It is an inverted index over equality-valued metadata fields:
// field -> scalar value -> set of point/doc ids. It accelerates selective
// equality/range/datetime/In/geo filters for the named and multi-vector
// collections (which key payloads by uint64 id, not by dense arena slot) via
// the filter-first search path.
//
// It is the EXACT structural analog of payloadIndex with the posting-set element
// type changed from arena slot (uint32) to point/doc id (uint64): named/MV have
// no slots and no slot reuse, so idKeys/geoIDCells are the reverse lists keyed by
// id. All value-agnostic helpers (scalarKeyOf, numRange/strRange, the geohash
// cell derivation, datetimeBound, the sortedKeys cache types) are REUSED from the
// dense implementation unchanged.
//
// Like the dense index it is an ACCELERATION ONLY. candidates() returns a
// SUPERSET of the matching ids and the search path re-applies the full predicate
// (and tombstone/TTL admission), so correctness never depends on the index being
// exact — over-cover is harmless (the re-check rejects it); the superset must
// never be MISSING a true match. Rebuilt-on-load (never serialized). Guarded by
// the owning collection's mutex.
type payloadIndexID struct {
	fields map[string]map[scalarKey]map[uint64]struct{}
	idKeys map[uint64][]fieldKey // per-id keys, for O(keys) reindex on mutation/delete

	// geo is a per-field geohash-prefix spatial index: field -> grid cell -> id
	// set. A ValueGeo field's point is bucketed by its fixed-precision geohash
	// cell (see geohash.go). Geo values do NOT participate in the scalar eq/range
	// structures above (scalarKeyOf declines them); they live ONLY here.
	// geoIDCells records, per id, the (field, cell) entries to drop on
	// reindex/delete — the geo analogue of idKeys.
	geo        map[string]map[geoCell]map[uint64]struct{}
	geoIDCells map[uint64][]geoSlotCell

	// tokens is a per-field inverted token index: field -> token -> id set. A
	// string (ValueString) or strings (ValueStrings) field is tokenized with the
	// SAME tokenize() the FilterMatch predicate uses, so a `match` clause can
	// narrow to a candidate superset (then the predicate re-checks). Postings are
	// PER-DOCUMENT: a token present in ANY element of a ValueStrings field posts
	// the id once; the predicate's per-element tokensContainAll re-narrows.
	// tokenIDKeys records, per id, the (field, token) entries to drop on
	// reindex/delete — the token analogue of geoIDCells. Like the rest of the
	// index this is ACCELERATION ONLY (a superset); it never carries the
	// contentField. The id-keyed mirror of payloadIndex.tokens/tokenSlotKeys.
	tokens      map[string]map[string]map[uint64]struct{}
	tokenIDKeys map[uint64][]fieldToken

	// contains is a per-field inverted ELEMENT index: field -> array element value
	// -> id set. An array (ValueStrings/Ints/Floats) field posts the id under EACH
	// DISTINCT element it holds, keyed by the SAME scalarKey the eq index +
	// compileContains use (so a `contains X` clause narrows to
	// contains[field][scalarKeyOf(X)] — exactly the ids whose array contains X,
	// then the predicate re-checks). containsIDKeys records, per id, the (field,
	// element key) entries to drop on reindex/delete — the contains analogue of
	// tokenIDKeys. ACCELERATION ONLY (a no-false-negative superset); never carries
	// the contentField. The id-keyed mirror of payloadIndex.contains/
	// containsSlotKeys.
	contains       map[string]map[scalarKey]map[uint64]struct{}
	containsIDKeys map[uint64][]fieldKey

	// sorted caches per-field distinct keys in sorted order so range queries
	// binary-search the bounds. Built lazily on the first range query after a
	// structural change; invalidated only when a distinct key is added/removed.
	// REUSES the dense sortedKeys/numEntry types verbatim. sortedMu serializes
	// concurrent readers' lazy builds (candidates runs under the owner's read
	// lock, which excludes writers but not other readers).
	sorted   map[string]*sortedKeys
	sortedMu sync.Mutex
}

func newPayloadIndexID() *payloadIndexID {
	return &payloadIndexID{
		fields:         make(map[string]map[scalarKey]map[uint64]struct{}),
		idKeys:         make(map[uint64][]fieldKey),
		sorted:         make(map[string]*sortedKeys),
		geo:            make(map[string]map[geoCell]map[uint64]struct{}),
		geoIDCells:     make(map[uint64][]geoSlotCell),
		tokens:         make(map[string]map[string]map[uint64]struct{}),
		tokenIDKeys:    make(map[uint64][]fieldToken),
		contains:       make(map[string]map[scalarKey]map[uint64]struct{}),
		containsIDKeys: make(map[uint64][]fieldKey),
	}
}

// markDirty invalidates field's sorted cache (called when a distinct key is
// added or removed). A no-op if the field has never been range-queried.
func (p *payloadIndexID) markDirty(field string) {
	if sc := p.sorted[field]; sc != nil {
		sc.dirty = true
	}
}

// ensureSorted returns field's sorted-key cache, rebuilding it from the live
// posting map if stale. Caller holds the owner's read lock (so `fields` is
// stable); sortedMu serializes concurrent rebuilds. Mirrors the dense
// ensureSorted exactly (it touches only `fields` keys, not posting-set element
// type, so the logic is value-agnostic — duplicated id-side only because the
// receiver type differs).
func (p *payloadIndexID) ensureSorted(field string) *sortedKeys {
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

// reindex removes any existing entries for id, then indexes the scalar+geo
// fields of meta (meta may be nil/empty, which is a pure removal). Calling it on
// every payload mutation keeps the index correct. Must hold the owner's lock.
// Mirrors dense reindex; skips contentField.
func (p *payloadIndexID) reindex(id uint64, meta Metadata) {
	if old, ok := p.idKeys[id]; ok {
		for _, fk := range old {
			p.drop(fk, id)
		}
		delete(p.idKeys, id)
	}
	if old, ok := p.geoIDCells[id]; ok {
		for _, gc := range old {
			p.dropGeo(gc, id)
		}
		delete(p.geoIDCells, id)
	}
	if old, ok := p.tokenIDKeys[id]; ok {
		for _, ft := range old {
			p.dropToken(ft, id)
		}
		delete(p.tokenIDKeys, id)
	}
	if old, ok := p.containsIDKeys[id]; ok {
		for _, fk := range old {
			p.dropContains(fk, id)
		}
		delete(p.containsIDKeys, id)
	}
	if len(meta) == 0 {
		return
	}
	var keys []fieldKey
	var geoCells []geoSlotCell
	var idTokens []fieldToken
	var idContains []fieldKey
	for field, v := range meta {
		if field == contentField {
			continue // document content is not a filterable field; never index it
		}
		// Geo values bucket into the spatial index ONLY (they decline the scalar
		// index via scalarKeyOf=false); maintained incrementally per id here.
		if v.Kind == ValueGeo {
			cell := cellOf(v.Lat, v.Lon)
			cells := p.geo[field]
			if cells == nil {
				cells = make(map[geoCell]map[uint64]struct{})
				p.geo[field] = cells
			}
			set := cells[cell]
			if set == nil {
				set = make(map[uint64]struct{})
				cells[cell] = set
			}
			set[id] = struct{}{}
			geoCells = append(geoCells, geoSlotCell{field, cell})
			continue
		}
		// Token index: a string/strings field is tokenized (same tokenize() as the
		// FilterMatch predicate) and each DISTINCT token posts the id once (per-
		// document). The eq index below still indexes a ValueString as one whole-
		// string key; the two are independent. ValueStrings only lives here
		// (scalarKeyOf declines slices). ValueRecord considered: no case, so it
		// correctly declines tokenization like every other non-string kind.
		switch v.Kind {
		case ValueString:
			idTokens = p.addTokens(field, tokenize(v.Str), id, idTokens)
		case ValueStrings:
			var toks []string
			for _, s := range v.Strs {
				toks = append(toks, tokenize(s)...)
			}
			idTokens = p.addTokens(field, toks, id, idTokens)
		}
		// Contains index: an array field (strings/ints/floats) posts the id under
		// each DISTINCT element key (same scalarKey as eq/compileContains). Arrays
		// decline the scalar eq index below; they live here. Mirrors token discipline.
		if ek := elementKeysOf(v); len(ek) > 0 {
			idContains = p.addContains(field, ek, id, idContains)
		}
		key, ok := scalarKeyOf(v)
		if !ok {
			continue
		}
		vals := p.fields[field]
		if vals == nil {
			vals = make(map[scalarKey]map[uint64]struct{})
			p.fields[field] = vals
		}
		set := vals[key]
		if set == nil {
			set = make(map[uint64]struct{})
			vals[key] = set
			p.markDirty(field) // new distinct key -> sorted cache stale
		}
		set[id] = struct{}{}
		keys = append(keys, fieldKey{field, key})
	}
	if keys != nil {
		p.idKeys[id] = keys
	}
	if geoCells != nil {
		p.geoIDCells[id] = geoCells
	}
	if idTokens != nil {
		p.tokenIDKeys[id] = idTokens
	}
	if idContains != nil {
		p.containsIDKeys[id] = idContains
	}
}

// addContains posts id under each (already DISTINCT) element key in keys within
// field's contains index, appending the new (field, key) reverse entries to acc
// (so reindex can drop them on mutation/delete). elementKeysOf dedups within the
// value, so each reverse entry is distinct — keeping dropContains' per-entry
// delete exact. The id-keyed mirror of payloadIndex.addContains.
func (p *payloadIndexID) addContains(field string, keys []scalarKey, id uint64, acc []fieldKey) []fieldKey {
	byKey := p.contains[field]
	if byKey == nil {
		byKey = make(map[scalarKey]map[uint64]struct{})
		p.contains[field] = byKey
	}
	for _, k := range keys {
		set := byKey[k]
		if set == nil {
			set = make(map[uint64]struct{})
			byKey[k] = set
		}
		set[id] = struct{}{}
		acc = append(acc, fieldKey{field, k})
	}
	return acc
}

// dropContains removes id from the contains posting list for fk, cleaning up
// empty maps. The contains analogue of drop/dropToken.
func (p *payloadIndexID) dropContains(fk fieldKey, id uint64) {
	byKey := p.contains[fk.field]
	if byKey == nil {
		return
	}
	if set := byKey[fk.key]; set != nil {
		delete(set, id)
		if len(set) == 0 {
			delete(byKey, fk.key)
		}
	}
	if len(byKey) == 0 {
		delete(p.contains, fk.field)
	}
}

// addTokens posts id under each DISTINCT token of toks within field's token
// index, appending the new (field, token) reverse entries to acc (so reindex can
// drop them on mutation/delete). Duplicate tokens (repeated across a
// ValueStrings, or recurring in one string) post once — both the posting set
// (already a set) and the reverse list stay distinct, keeping dropToken's per-
// entry delete exact. The id-keyed mirror of payloadIndex.addTokens.
func (p *payloadIndexID) addTokens(field string, toks []string, id uint64, acc []fieldToken) []fieldToken {
	if len(toks) == 0 {
		return acc
	}
	byTok := p.tokens[field]
	if byTok == nil {
		byTok = make(map[string]map[uint64]struct{})
		p.tokens[field] = byTok
	}
	for _, tok := range toks {
		set := byTok[tok]
		if set == nil {
			set = make(map[uint64]struct{})
			byTok[tok] = set
		}
		if _, dup := set[id]; dup {
			continue // already posted this id under this token -> distinct
		}
		set[id] = struct{}{}
		acc = append(acc, fieldToken{field, tok})
	}
	return acc
}

// dropToken removes id from the token posting list for (field, token), cleaning
// up empty maps. The token analogue of drop/dropGeo.
func (p *payloadIndexID) dropToken(ft fieldToken, id uint64) {
	byTok := p.tokens[ft.field]
	if byTok == nil {
		return
	}
	if set := byTok[ft.token]; set != nil {
		delete(set, id)
		if len(set) == 0 {
			delete(byTok, ft.token)
		}
	}
	if len(byTok) == 0 {
		delete(p.tokens, ft.field)
	}
}

// dropGeo removes id from the geo posting list for (field, cell), cleaning up
// empty maps. The geo analogue of drop.
func (p *payloadIndexID) dropGeo(gc geoSlotCell, id uint64) {
	cells := p.geo[gc.field]
	if cells == nil {
		return
	}
	if set := cells[gc.cell]; set != nil {
		delete(set, id)
		if len(set) == 0 {
			delete(cells, gc.cell)
		}
	}
	if len(cells) == 0 {
		delete(p.geo, gc.field)
	}
}

// drop removes id from the posting list for fk, cleaning up empty maps.
func (p *payloadIndexID) drop(fk fieldKey, id uint64) {
	vals := p.fields[fk.field]
	if vals == nil {
		return
	}
	if set := vals[fk.key]; set != nil {
		delete(set, id)
		if len(set) == 0 {
			delete(vals, fk.key)
			p.markDirty(fk.field) // distinct key removed -> sorted cache stale
		}
	}
	if len(vals) == 0 {
		delete(p.fields, fk.field)
	}
}

// rebuild discards and reconstructs the index from the given payload map (the
// collection's live id -> Metadata). Used after load/Restore and WAL replay —
// the rebuild-on-load path (the index is never serialized). Must hold the lock.
// Mirrors dense rebuild but iterates a map instead of an arena.
func (p *payloadIndexID) rebuild(meta map[uint64]Metadata) {
	p.fields = make(map[string]map[scalarKey]map[uint64]struct{})
	p.idKeys = make(map[uint64][]fieldKey)
	p.sorted = make(map[string]*sortedKeys) // drop stale sorted caches
	p.geo = make(map[string]map[geoCell]map[uint64]struct{})
	p.geoIDCells = make(map[uint64][]geoSlotCell)
	p.tokens = make(map[string]map[string]map[uint64]struct{})
	p.tokenIDKeys = make(map[uint64][]fieldToken)
	p.contains = make(map[string]map[scalarKey]map[uint64]struct{})
	p.containsIDKeys = make(map[uint64][]fieldKey)
	for id, m := range meta {
		p.reindex(id, m)
	}
}

// emptyIDSet is a shared read-only sentinel returned when a field has no
// postings; callers only ever read it (intersection -> empty result).
var emptyIDSet = map[uint64]struct{}{}

// candidates returns (ids, true) when filter can be narrowed by the payload
// index — the ids are a SUPERSET of the filter's matching ids, so the caller
// must still apply the full predicate. Returns (nil, false) when the filter is
// not index-narrowable (Or/Not/Ne/Contains/Regex/IsEmpty/IsNull, or a
// non-selective range/match), signalling the caller to fall back to graph search.
// `limit` caps the work the range path will do before giving up. Must hold the
// owner's read lock. Mirrors dense candidates exactly.
func (p *payloadIndexID) candidates(f Filter, limit int) ([]uint64, bool) {
	return p.candidatesCapped(f, limit, math.MaxInt)
}

// candidatesCapped is candidates with an additional early-abort ceiling: the
// final merged candidate set is abandoned (ok=false) the moment its size would
// exceed maxCand, without finishing the materialization. maxCand must be <=
// limit for the abort to ever trigger tighter than limit already does;
// candidates calls this with maxCand = math.MaxInt, so it never aborts early and
// is byte-identical to the uncapped behavior. Used by the named/MV filter-first
// planners, which already know the largest candidate count they could act on
// (filterFirstCrossover) and so have no use for a superset that blows past it.
// Mirrors dense candidatesCapped exactly.
func (p *payloadIndexID) candidatesCapped(f Filter, limit, maxCand int) ([]uint64, bool) {
	// Geo narrowing first so And(geo, eq/range) narrows on the geo conjunct.
	if sets, ok := p.collectGeoSets(f, limit); ok && len(sets) > 0 {
		// Intersect the geo sets with any equality sets in the same And so the most
		// selective conjunct(s) drive the candidate set; the predicate re-checks
		// everything else. A pure geo filter just returns the geo union.
		if eqTerms, ok := collectEqTerms(f); ok {
			for _, t := range eqTerms {
				vals := p.fields[t.field]
				if vals == nil {
					return []uint64{}, true
				}
				set := vals[t.key]
				if set == nil {
					return []uint64{}, true
				}
				sets = append(sets, set)
			}
		}
		// Fold any Match conjunct sets in too, so And(geo, match) narrows on both.
		if matchSets, ok := p.collectMatchSets(f, limit); ok && len(matchSets) > 0 {
			sets = append(sets, matchSets...)
		}
		// Fold any Contains conjunct sets in too, so And(geo, contains) narrows on both.
		if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
			sets = append(sets, containsSets...)
		}
		return intersectIDSets(sets, maxCand)
	}

	// Equality narrowing. Also covers And(eq, range): the eq terms narrow and the
	// predicate re-checks the range conjuncts.
	if terms, ok := collectEqTerms(f); ok {
		sets := make([]map[uint64]struct{}, 0, len(terms))
		for _, t := range terms {
			vals := p.fields[t.field]
			if vals == nil {
				return []uint64{}, true // no id carries this field -> empty match
			}
			set := vals[t.key]
			if set == nil {
				return []uint64{}, true
			}
			sets = append(sets, set)
		}
		// Fold any Match conjunct sets in too, so And(eq, match) narrows on both.
		if matchSets, ok := p.collectMatchSets(f, limit); ok && len(matchSets) > 0 {
			sets = append(sets, matchSets...)
		}
		// Fold any Contains conjunct sets in too, so And(eq, contains) narrows on both.
		if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
			sets = append(sets, containsSets...)
		}
		return intersectIDSets(sets, maxCand)
	}

	// Match narrowing: a top-level Match leaf, or an And whose only narrowable
	// conjuncts are Match terms (no eq/geo to lean on above). Each Match term's
	// query tokens are intersected within its field; the predicate re-checks.
	if matchSets, ok := p.collectMatchSets(f, limit); ok && len(matchSets) > 0 {
		// Fold any range/In conjunct sets in so And(match, range) narrows on both.
		if rangeSets, ok := p.collectRangeSets(f, limit); ok {
			matchSets = append(matchSets, rangeSets...)
		}
		// Fold any Contains conjunct sets in so And(match, contains) narrows on both.
		if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
			matchSets = append(matchSets, containsSets...)
		}
		return intersectIDSets(matchSets, maxCand)
	}

	// Contains narrowing: a top-level Contains leaf, or an And whose only
	// narrowable conjuncts are Contains terms (no eq/geo/match to lean on above).
	// Each Contains term's element posting list is intersected within the And; the
	// predicate re-checks. Folds any range/In conjunct sets in too.
	if containsSets, ok := p.collectContainsSets(f, limit); ok && len(containsSets) > 0 {
		if rangeSets, ok := p.collectRangeSets(f, limit); ok {
			containsSets = append(containsSets, rangeSets...)
		}
		return intersectIDSets(containsSets, maxCand)
	}

	// Range-only / In-only narrowing: no equality conjunct to lean on, so build
	// candidate sets directly from the posting lists.
	return p.rangeCandidates(f, limit, maxCand)
}

// intersectIDSets returns (the intersection of the given posting sets, true) as
// a slice, scanning the smallest set and probing the rest — UNLESS the
// intersection would exceed maxCand, in which case it abandons the scan as soon
// as that's known and returns (nil, false) instead of finishing the
// materialization. A single-set intersection is just that set, so its size is
// known from the map length alone: decide against maxCand before copying
// anything. Mirrors intersectSlotSets.
func intersectIDSets(sets []map[uint64]struct{}, maxCand int) ([]uint64, bool) {
	if len(sets) == 0 {
		return []uint64{}, true
	}
	if len(sets) == 1 {
		if len(sets[0]) > maxCand {
			return nil, false
		}
		out := make([]uint64, 0, len(sets[0]))
		for id := range sets[0] {
			out = append(out, id)
		}
		return out, true
	}
	smallest := 0
	for i := 1; i < len(sets); i++ {
		if len(sets[i]) < len(sets[smallest]) {
			smallest = i
		}
	}
	out := make([]uint64, 0, len(sets[smallest]))
	for id := range sets[smallest] {
		inAll := true
		for i, s := range sets {
			if i == smallest {
				continue
			}
			if _, ok := s[id]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			out = append(out, id)
			if len(out) > maxCand {
				return nil, false
			}
		}
	}
	return out, true
}

// collectGeoSets gathers the geohash-index id sets for the geo leaves of a
// filter: a single geo op, or the geo conjuncts of a top-level And (recursing
// into nested Ands). ok=false means either nothing was geo, OR a geo leaf
// overflowed the cover-cell cap / hit a pole/antimeridian (bail to graph).
// Mirrors dense collectGeoSets.
func (p *payloadIndexID) collectGeoSets(f Filter, limit int) ([]map[uint64]struct{}, bool) {
	switch f.Op {
	case FilterGeoRadius, FilterGeoBox, FilterGeoPolygon:
		set, ok := p.geoSet(f.Field, f.Op, f.Geo, limit)
		if !ok {
			return nil, false
		}
		return []map[uint64]struct{}{set}, true
	case FilterAnd:
		var acc []map[uint64]struct{}
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
				// A nested And contributes its geo narrowing if it has any; a non-geo
				// or overflowed nested And returns ok=false and is simply skipped
				// (correctness-safe: the outer set stays a superset).
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

// collectMatchSets gathers the token-index id sets for the Match leaves of a
// filter: a single FilterMatch op, or the Match conjuncts of a top-level And
// (recursing into nested Ands). Each Match leaf intersects its query tokens'
// postings within its field into ONE superset set (match-ALL ⇒ intersect). The
// returned sets are each a SUPERSET of that Match's predicate matches (per-
// document postings ⊇ per-element matches; tokenize is identical to the
// predicate's). Bail/skip semantics mirror collectGeoSets: a bare Match LEAF
// that is not narrowable (empty query tokens, or the selectivity cap overflowed)
// returns ok=false; a non-narrowable Match CONJUNCT of an And is simply SKIPPED
// (the predicate re-checks it, and an empty-token match adds no constraint). The
// id-keyed mirror of payloadIndex.collectMatchSets.
func (p *payloadIndexID) collectMatchSets(f Filter, limit int) ([]map[uint64]struct{}, bool) {
	switch f.Op {
	case FilterMatch:
		set, ok := p.matchSet(f.Field, f.Value, limit)
		if !ok {
			return nil, false
		}
		return []map[uint64]struct{}{set}, true
	case FilterAnd:
		var acc []map[uint64]struct{}
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
// element index: look up contains[field][scalarKeyOf(want)] — the ids whose array
// holds an element == want. EXACTLY the predicate match set (no-false-negative;
// elementKeysOf posts every element compileContains matches under the SAME
// scalarKey). ok=false when want is not a scalar (compileContains rejects it) or
// the posting set exceeds `limit` (selectivity cap). A field/element with no
// postings yields the empty sentinel (ok=true). The id-keyed mirror of
// payloadIndex.containsSet.
func (p *payloadIndexID) containsSet(field string, want Value, limit int) (map[uint64]struct{}, bool) {
	if !indexNarrowable(field) {
		return nil, false
	}
	key, ok := scalarKeyOf(want)
	if !ok {
		return nil, false
	}
	byKey := p.contains[field]
	if byKey == nil {
		return emptyIDSet, true
	}
	set := byKey[key]
	if set == nil {
		return emptyIDSet, true
	}
	if len(set) > limit {
		return nil, false
	}
	return set, true
}

// collectContainsSets gathers the contains-index id sets for the Contains leaves
// of a filter: a single FilterContains op, or the Contains conjuncts of a top-
// level And (recursing into nested Ands). Each set is a SUPERSET of that leaf's
// predicate matches. Bail/skip semantics mirror collectMatchSets: a bare Contains
// LEAF that is not narrowable returns ok=false; a non-narrowable Contains CONJUNCT
// of an And is SKIPPED. The id-keyed mirror of payloadIndex.collectContainsSets.
func (p *payloadIndexID) collectContainsSets(f Filter, limit int) ([]map[uint64]struct{}, bool) {
	switch f.Op {
	case FilterContains:
		set, ok := p.containsSet(f.Field, f.Value, limit)
		if !ok {
			return nil, false
		}
		return []map[uint64]struct{}{set}, true
	case FilterAnd:
		var acc []map[uint64]struct{}
		found := false
		for i := range f.And {
			switch f.And[i].Op {
			case FilterContains:
				set, ok := p.containsSet(f.And[i].Field, f.And[i].Value, limit)
				if !ok {
					continue
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
// posting lists of every query token within `field` (match-ALL), scanning the
// smallest token's postings and probing the rest, capped at `limit`.
//
// ok=false (not narrowable) when: the query has zero tokens (empty match -> the
// whole field, never a narrowing); the field is non-string (compileMatch rejects
// it; declining here is a safety net); or the intersected set would exceed
// `limit` (selectivity cap: bail to graph instead of materializing a near-full-
// corpus set — never a truncated set). A missing query token yields an empty set
// (ok=true): no id has all tokens. The id-keyed mirror of payloadIndex.matchSet.
func (p *payloadIndexID) matchSet(field string, want Value, limit int) (map[uint64]struct{}, bool) {
	if !indexNarrowable(field) {
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
		return emptyIDSet, true // no id carries this field -> empty match
	}
	sets := make([]map[uint64]struct{}, 0, len(queryTokens))
	for _, tok := range queryTokens {
		set := byTok[tok]
		if set == nil {
			return emptyIDSet, true // a query token is in no doc -> empty intersect
		}
		sets = append(sets, set)
	}
	smallest := 0
	for i := 1; i < len(sets); i++ {
		if len(sets[i]) < len(sets[smallest]) {
			smallest = i
		}
	}
	if len(sets[smallest]) > limit {
		return nil, false // even the smallest token set already exceeds limit -> bail
	}
	out := make(map[uint64]struct{})
	for id := range sets[smallest] {
		inAll := true
		for i, s := range sets {
			if i == smallest {
				continue
			}
			if _, ok := s[id]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			out[id] = struct{}{}
			if len(out) > limit {
				return nil, false // selectivity cap: superset too large -> graph fallback
			}
		}
	}
	return out, true
}

// geoSet builds the candidate superset for one geo op via the geohash index:
// reduce the region to a bbox, enumerate the covering cells, and union their
// posting lists. ok=false on a malformed/nil condition, a pole/antimeridian
// region, or when the covering-cell count / union size exceeds the cap (the
// overflow bail). The returned set is ALWAYS a superset of the true predicate
// match set. Mirrors dense geoSet (reuses radiusBBox/polygonBBox/coverCells).
func (p *payloadIndexID) geoSet(field string, op FilterOp, g *GeoCondition, limit int) (map[uint64]struct{}, bool) {
	if !indexNarrowable(field) {
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
		return emptyIDSet, true // no id carries this geo field → empty match
	}
	out := make(map[uint64]struct{})
	for _, c := range cells {
		for id := range cellMap[c] {
			out[id] = struct{}{}
			if len(out) > limit {
				return nil, false // candidate union too large → graph fallback
			}
		}
	}
	return out, true
}

// rangeCandidates narrows a range-only / In-only filter using the posting lists.
// Mirrors dense rangeCandidates.
func (p *payloadIndexID) rangeCandidates(f Filter, limit, maxCand int) ([]uint64, bool) {
	sets, ok := p.collectRangeSets(f, limit)
	if !ok || len(sets) == 0 {
		return nil, false
	}
	return intersectIDSets(sets, maxCand)
}

// collectRangeSets gathers the index-narrowable leaf sets of a range/In filter.
// Leaves that are not narrowable or whose union overflows `limit` are skipped —
// the predicate re-checks them, which only widens the superset. ok=false means
// nothing was narrowable. Mirrors dense collectRangeSets (reuses datetimeBound /
// dtToOrdering for the datetime ops).
func (p *payloadIndexID) collectRangeSets(f Filter, limit int) ([]map[uint64]struct{}, bool) {
	switch f.Op {
	case FilterEq:
		if set, ok := p.eqSet(f.Field, f.Value); ok {
			return []map[uint64]struct{}{set}, true
		}
		return nil, false
	case FilterGt, FilterGte, FilterLt, FilterLte:
		if set, ok := p.orderingSet(f.Field, f.Op, f.Value, limit); ok {
			return []map[uint64]struct{}{set}, true
		}
		return nil, false
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		// Datetime range narrows EXACTLY like the numeric ordering ops: the field is
		// stored as int64 unix-ms, so we lower the RFC3339 bound to int64-ms via the
		// SAME shared helper the compiler uses (datetimeBound) and feed it to
		// orderingSet as a numeric Value under the plain ordering op.
		ms, ok := datetimeBound(f.Value)
		if !ok {
			return nil, false
		}
		if set, ok := p.orderingSet(f.Field, dtToOrdering(f.Op), NewInt(ms), limit); ok {
			return []map[uint64]struct{}{set}, true
		}
		return nil, false
	case FilterIn:
		if set, ok := p.inSet(f.Field, f.Value, limit); ok {
			return []map[uint64]struct{}{set}, true
		}
		return nil, false
	case FilterAnd:
		var acc []map[uint64]struct{}
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
// copy). ok=false when v is not equality-indexable. Mirrors dense eqSet.
func (p *payloadIndexID) eqSet(field string, v Value) (map[uint64]struct{}, bool) {
	if !indexNarrowable(field) {
		return nil, false
	}
	key, ok := scalarKeyOf(v)
	if !ok {
		return nil, false
	}
	vals := p.fields[field]
	if vals == nil {
		return emptyIDSet, true
	}
	if set := vals[key]; set != nil {
		return set, true
	}
	return emptyIDSet, true
}

// orderingSet builds the union of posting lists whose key satisfies `op` vs
// `want`. Numeric keys (int and float) compare as float64; string keys
// lexicographically. ok=false when want's kind can't drive the comparison, or
// when the union exceeds `limit`. Mirrors dense orderingSet (reuses
// numericValue/numRange/strRange).
func (p *payloadIndexID) orderingSet(field string, op FilterOp, want Value, limit int) (map[uint64]struct{}, bool) {
	if !indexNarrowable(field) {
		return nil, false
	}
	vals := p.fields[field]
	if vals == nil {
		return emptyIDSet, true // no id carries this field -> empty match
	}
	sc := p.ensureSorted(field)
	out := make(map[uint64]struct{})

	unionRange := func(lo, hi int, keyAt func(i int) scalarKey) bool {
		for i := lo; i < hi; i++ {
			for id := range vals[keyAt(i)] {
				out[id] = struct{}{}
				if len(out) > limit {
					return false
				}
			}
		}
		return true
	}

	if wf, ok := numericValue(want); ok {
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

// inSet builds the union of posting lists for each member of an In value's array
// (strings/ints/floats). ok=false for a non-array value or when the union
// exceeds `limit`. Mirrors dense inSet.
func (p *payloadIndexID) inSet(field string, want Value, limit int) (map[uint64]struct{}, bool) {
	if !indexNarrowable(field) {
		return nil, false
	}
	vals := p.fields[field]
	if vals == nil {
		return emptyIDSet, true
	}
	out := make(map[uint64]struct{})
	add := func(set map[uint64]struct{}) bool {
		for id := range set {
			out[id] = struct{}{}
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
