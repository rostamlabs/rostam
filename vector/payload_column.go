// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// The numeric column sidecar.
//
// WHAT IT IS. A per-field, slot-indexed []float64 holding each slot's numeric
// value for that field — NaN where the slot has no numeric value at all. It
// exists so a range predicate can be answered from ONE array index instead of
// arena.Metadata(slot) -> a string-hashed map probe -> a 96-byte Value copy ->
// the compiled closure, which is what every traversed candidate of a filtered
// query pays today.
//
// WHY A COLUMN AND NOT ANOTHER BITSET. The admission gate (filter_bitset.go)
// answers the same question faster still — one bit — but it has to be BUILT per
// query, and its build cost is proportional to posting mass on whichever side is
// cheaper. That is a good trade for a selective filter and a losing one for a
// filter that passes 99% of a million points: neither side is small. A column is
// built ONCE per field and then costs nothing per query, at ANY selectivity and
// any corpus size, which is exactly the regime the gates cannot reach. The two
// are complementary, and the search path prefers the column precisely because
// its per-query price is zero.
//
// WHY NaN IS THE ABSENT SENTINEL, AND WHY THAT IS NOT A HACK. The m5 range
// semantics (orderingHoldsFloat) make a NaN operand UNORDERED, so gt/gte/lt/lte
// are all false for it. The compiled predicate rejects a slot whose field is
// absent, non-numeric, or NaN — three different reasons, one outcome. Mapping
// all three to NaN in the column makes the column comparison produce that
// outcome by IEEE's own rule, with no presence bitmap and no branch: the column
// is EXACT by construction rather than by a case analysis. Piece one of m5 is
// what makes piece three a two-line evaluator.
//
// EXACTNESS PRECONDITIONS. Only one thing the column cannot see: a PER-KEY TTL,
// which hides a key from liveMeta while the arena still stores it (and while
// this column still carries its value). The search path refuses the column
// whenever the arena reports any per-key deadline at all — the same veto, for
// the same reason, that the gate's exactness check applies.
//
// MAINTENANCE. Every payload mutation in the system funnels through
// payloadIndex.reindex — insert, SetPayload, OverwritePayload,
// DeletePayloadKeys, ClearPayload, the per-key TTL sweep, and the dead-slot
// reclaim — so updateColumns hangs off that single choke point and cannot be
// bypassed by a path that forgot to call it. Tombstoned slots are deliberately
// NOT cleared: they keep a stale value that no query can observe, because
// admitVerdictOf's liveness test runs strictly before any admission oracle.

// maxNumColumns bounds how many fields may carry a column at once, and with it
// the sidecar's worst-case memory: 8 bytes per slot per column, so at most 64
// bytes per slot.
//
// WHETHER THAT IS CHEAP DEPENDS ENTIRELY ON THE DIMENSION, and the honest way to
// state it is as a ratio against the vector itself (4·dim bytes per slot):
// negligible at 768d (64 B against 3 KiB, ~2%), a QUARTER of the vector data at
// 64d (64 B against 256), and TWICE the vector data at 8d — which is not
// hypothetical, it is the dimension several tests in this package use.
// Low-dimensional collections are the case to watch, which is why the cap is not
// the only control: columns are charged against Config.MaxBytes (columnBytes),
// are EVICTED rather than held forever (evictColdestColumn), and are dropped
// outright when an insert would otherwise be refused for want of their bytes
// (insertLocked) — a read-side cache never outranks a durable write.
//
// The cap exists so the worst case is BOUNDED and statable, not because eight is
// special.
const maxNumColumns = 8

// columnBytesPerSlot is one column's per-slot cost: a float64. Named so the
// quota arithmetic reads as accounting rather than as a magic 8.
const columnBytesPerSlot = 8

// numColumn is one field's column plus the bookkeeping that lets the cap be
// adaptive instead of first-come-first-served.
type numColumn struct {
	// vals is the column: vals[slot] is the slot's numeric value for the field,
	// NaN when it has none. Mutated in place by updateColumns under the owner's
	// WRITE lock; read by queries under the read lock.
	vals []float64
	// lastUse is a clock stamp bumped every time a query resolves this column.
	// Eviction takes the smallest, which is plain LRU — cheap to maintain (one
	// store under colsMu, which the lookup already holds) and cheap to consult (a
	// linear scan over at most maxNumColumns entries).
	lastUse uint64
}

// columnTerm is one range comparison compiled against a column: "the slot's
// value for this field, related to bound by op". vals is the column itself,
// captured so evaluation needs no map lookup.
type columnTerm struct {
	vals  []float64
	bound float64
	op    FilterOp
}

// holds evaluates the term at slot. A slot past the column's end has never been
// reindexed, hence carries no payload, hence is rejected — the same answer the
// predicate gives for an absent field, reached without a bounds panic.
//
// The comparisons are Go's own float64 operators, and that is the m5 NaN
// decision showing up as an ABSENCE of code: IEEE-754 says every ordered
// comparison against NaN is false, Go implements IEEE, and orderingHoldsFloat
// was written to say exactly that — so `v >= t.bound` already rejects the
// absent/non-numeric/NaN sentinel with no presence test, no branch, and no
// possibility of the two evaluators drifting apart. Spelled out here rather than
// delegated so the hot loop is four instructions; equivalence to
// orderingHoldsFloat is pinned by TestColumnHoldsMatchesOrderingHoldsFloat.
func (t *columnTerm) holds(slot uint32) bool {
	if int(slot) >= len(t.vals) {
		return false
	}
	v := t.vals[slot]
	switch t.op {
	case FilterGt:
		return v > t.bound
	case FilterGte:
		return v >= t.bound
	case FilterLt:
		return v < t.bound
	case FilterLte:
		return v <= t.bound
	}
	return false
}

// numericKeyValue returns a scalar key's numeric value, or ok=false for a key
// that is not numeric. It is the inverse of the reindex-side scalarKeyOf for the
// two kinds a column can hold.
func numericKeyValue(k scalarKey) (float64, bool) {
	switch k.kind {
	case ValueInt:
		return float64(k.i), true
	case ValueFloat:
		return k.f, true
	default:
		// ValueRecord considered: k.kind can never be ValueRecord — scalarKeyOf
		// declines records before a scalarKey is ever minted, so this correctly
		// declines by construction, not just by accident.
		return 0, false
	}
}

// growColumn returns col extended so that index slot is addressable, NaN-filling
// the new tail (NaN = "no numeric value", see the file header). Growth is
// amortized by append's own doubling.
func growColumn(col []float64, slot uint32) []float64 {
	need := int(slot) + 1
	if len(col) >= need {
		return col
	}
	for len(col) < need {
		col = append(col, math.NaN())
	}
	return col
}

// columnBytes is the total memory the sidecar currently holds. Read on the
// INSERT path (which holds the write lock) to charge columns against
// Config.MaxBytes, and on the query path to decide whether another column can be
// afforded — two different locks, hence the atomic rather than colsMu.
func (p *payloadIndex) columnBytes() int64 { return p.colBytes.Load() }

// ensureColumn returns field's column, building it if absent. Called on the
// QUERY path under the owner's read lock (which excludes writers but not other
// readers), so colsMu serializes concurrent builds exactly as sortedMu does for
// the sorted key cache.
//
// The build reads the field's posting map, which is the only place the values
// live in indexed form: one pass over every posting, writing each slot's value.
// That is O(live postings) — the same order as the sorted-key cache's first
// build, and paid once per field rather than once per query.
//
// budget is the most this call may ALLOCATE (see buildColumnGate, which derives
// it from Config.MaxBytes); a negative budget means unlimited. An existing
// column is always returned regardless of budget: it is already paid for, and
// refusing to use memory that is already resident would be a pure loss.
//
// ok=false means the field will not get a column — no postings, no numeric
// values, or no room — and the caller must fall back to the metadata predicate.
func (p *payloadIndex) ensureColumn(field string, capacity int, budget int64) ([]float64, bool) {
	p.colsMu.Lock()
	defer p.colsMu.Unlock()
	p.colClock++
	if c, ok := p.cols[field]; ok {
		// A column shorter than capacity is not stale: every slot that was ever
		// reindexed is covered (updateColumns grows it), and a slot beyond the end
		// has no payload, which columnTerm.holds already answers correctly. Growing
		// here would only pre-pay for slots that do not exist yet.
		c.lastUse = p.colClock
		return c.vals, true
	}
	vals := p.fields[field]
	if vals == nil {
		return nil, false // no postings: nothing to build from, and nothing to match
	}
	// THE BUDGET IS TESTED AGAINST THE POST-EVICTION FOOTPRINT, which is why the
	// victim is chosen (but not yet dropped) first.
	//
	// The order used to be the other way round, and it quietly restored the very
	// policy eviction exists to prevent. `budget` already has the resident columns
	// subtracted out of it, so a byte-capped collection at the cap has a budget
	// near zero and refused the ninth field — even though swapping it for the
	// coldest of the eight is memory-NEUTRAL, or a saving when the victim is
	// larger. Every collection with MaxBytes set was back to first-come-forever,
	// with the cap's LRU reachable only when MaxBytes was unset.
	//
	// Charging the swap rather than the allocation is the honest accounting: what
	// the collection is about to hold is `want` MINUS whatever the eviction gives
	// back. peekColdestColumn does not mutate, so a budget refusal below still
	// leaves the cache exactly as it was.
	want := int64(capacity) * columnBytesPerSlot
	atCap := len(p.cols) >= maxNumColumns
	var reclaimable int64
	if atCap {
		bytes, ok := p.peekColdestColumn()
		if !ok {
			return nil, false // at the cap with nothing to evict: impossible, but not assumed
		}
		reclaimable = bytes
	}
	if budget >= 0 && want-reclaimable > budget {
		return nil, false // even after the swap this would cross the byte quota
	}
	col := make([]float64, capacity)
	for i := range col {
		col[i] = math.NaN()
	}
	built := false
	for key, set := range vals {
		v, ok := numericKeyValue(key)
		if !ok {
			continue // string/bool keys: not part of a numeric column
		}
		built = true
		for slot := range set {
			if int(slot) < len(col) {
				col[slot] = v
			}
		}
	}
	if !built {
		return nil, false // the field exists but carries no numeric values
	}
	// EVICTION HAPPENS HERE, AFTER the build is known to have produced something.
	// Doing it before the scan would throw away a live column to make room for a
	// field that then turns out to carry no numeric values at all — paying a
	// rebuild for nothing.
	//
	// AT THE CAP, EVICT THE COLDEST rather than refuse. First-come-first-served is
	// the wrong policy for a cache whose entries are created by whatever query
	// happened to run first: eight ad-hoc range queries during a backfill would
	// otherwise own all eight slots for the life of the process while the
	// steady-state workload got none. LRU makes the cap adapt to what is actually
	// being queried, and the cost of being wrong is a rebuild, not a wrong answer.
	//
	// Evicting a column an in-flight query is READING is safe: that query captured
	// the slice and Go keeps the array alive for it. It cannot go stale underneath
	// it either, because the only writer to a column is updateColumns under the
	// WRITE lock, which cannot run until every reader has released the read lock.
	if atCap && !p.evictColdestColumn() {
		return nil, false
	}
	if p.cols == nil {
		p.cols = make(map[string]*numColumn, maxNumColumns)
	}
	p.cols[field] = &numColumn{vals: col, lastUse: p.colClock}
	p.colBytes.Add(int64(cap(col)) * columnBytesPerSlot)
	return col, true
}

// peekColdestColumn returns the bytes the next eviction would reclaim, without
// evicting. It picks the same victim evictColdestColumn would — nothing can
// change between the two calls, since both run under colsMu — so the budget
// arithmetic above and the eviction below cannot disagree about what is being
// swapped. Caller holds colsMu.
func (p *payloadIndex) peekColdestColumn() (int64, bool) {
	field, ok := p.coldestColumn()
	if !ok {
		return 0, false
	}
	return int64(cap(p.cols[field].vals)) * columnBytesPerSlot, true
}

// coldestColumn names the least recently resolved column. Caller holds colsMu.
func (p *payloadIndex) coldestColumn() (string, bool) {
	var victim string
	var oldest uint64
	found := false
	for field, c := range p.cols {
		if !found || c.lastUse < oldest {
			victim, oldest, found = field, c.lastUse, true
		}
	}
	return victim, found
}

// evictColdestColumn drops the least recently resolved column and reports
// whether it dropped one. Caller holds colsMu.
func (p *payloadIndex) evictColdestColumn() bool {
	victim, ok := p.coldestColumn()
	if !ok {
		return false
	}
	p.colBytes.Add(-int64(cap(p.cols[victim].vals)) * columnBytesPerSlot)
	delete(p.cols, victim)
	return true
}

// updateColumns writes slot's value into EVERY existing column — including the
// columns of fields meta does not carry, which must be reset to NaN or a
// rewritten payload would keep answering with the value it used to have. Called
// from reindex under the owner's WRITE lock, which excludes every reader, so it
// needs no colsMu.
//
// It is a no-op (one length check) for the overwhelmingly common case of an
// index with no columns at all, which is every index until something
// range-queries it.
func (p *payloadIndex) updateColumns(slot uint32, meta Metadata) {
	if len(p.cols) == 0 {
		return
	}
	for field, c := range p.cols {
		before := cap(c.vals)
		c.vals = growColumn(c.vals, slot)
		if grown := cap(c.vals) - before; grown != 0 {
			// Growth is a real allocation and has to be charged, or the quota drifts
			// low over the life of an append-heavy collection.
			p.colBytes.Add(int64(grown) * columnBytesPerSlot)
		}
		v := math.NaN()
		if got, ok := lookupPath(meta, field); ok {
			if f, isNum := numericValue(got); isNum {
				v = f
			}
		}
		c.vals[slot] = v
	}
}

// dropColumns releases the whole sidecar and its byte accounting. Used by
// rebuild, which reconstructs the index from scratch and would otherwise leave
// columns sized and populated for the OLD slot layout.
func (p *payloadIndex) dropColumns() {
	p.cols = nil
	p.colBytes.Store(0)
}

// collectColumnTerms compiles f into column terms, or reports ok=false if any
// part of it cannot be answered from columns alone.
//
// IT IS ALL-OR-NOTHING, and that is the whole safety story. Each term is EXACT
// — column value vs bound under the same orderingHoldsFloat the predicate uses,
// with absent/non-numeric/NaN all mapped to NaN and therefore all rejected — so
// an And of terms is exact too, and admission needs no metadata read. A filter
// with one leaf the columns cannot express (an Eq, an Or, a geo condition) gets
// no column path at all rather than a partial one, because a partial answer
// would have to consult the predicate anyway and the predicate is the cost being
// removed.
//
// Ops covered: the ordering family, plus the FilterDt* family lowered through
// the SAME datetimeBound the compiler uses, so a datetime range and its numeric
// spelling compile to identical terms.
//
// budget caps what this call may ALLOCATE in new columns (negative = unlimited);
// buildColumnGate derives it from Config.MaxBytes. It is deliberately a single
// whole-filter budget rather than a per-leaf one: an And of two ranges over two
// uncolumnised fields must not be able to allocate twice what the quota allows,
// so each successful build subtracts from what remains.
func (p *payloadIndex) collectColumnTerms(f Filter, capacity int, budget int64, acc []columnTerm) ([]columnTerm, bool) {
	// Shape check BEFORE any column is resolved. Without it, And(range, eq) would
	// build (and then maintain forever) a column for the range leaf before the eq
	// leaf declined the whole filter — a permanent per-slot cost bought for a
	// query that cannot use it.
	if !columnExpressible(f) {
		return acc, false
	}
	acc, _, ok := p.appendColumnTerms(f, capacity, budget, acc)
	return acc, ok
}

// columnExpressible reports whether f's SHAPE can be answered from columns,
// touching nothing but the filter itself. It must stay in exact agreement with
// appendColumnTerms' op coverage; the two are adjacent for that reason.
func columnExpressible(f Filter) bool {
	switch f.Op {
	case FilterGt, FilterGte, FilterLt, FilterLte:
		bound, ok := numericValue(f.Value)
		return ok && bound == bound && indexNarrowable(f.Field, f.Op)
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		_, ok := datetimeBound(f.Value)
		return ok && indexNarrowable(f.Field, f.Op)
	case FilterAnd:
		if len(f.And) == 0 {
			return false
		}
		for i := range f.And {
			if !columnExpressible(f.And[i]) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

// appendColumnTerms is collectColumnTerms threading the remaining allocation
// budget through the recursion; it returns what is left of it.
func (p *payloadIndex) appendColumnTerms(f Filter, capacity int, budget int64, acc []columnTerm) ([]columnTerm, int64, bool) {
	switch f.Op {
	case FilterGt, FilterGte, FilterLt, FilterLte:
		bound, ok := numericValue(f.Value)
		if !ok || bound != bound {
			// A non-numeric want takes the predicate's STRING path (or fails), and a
			// NaN bound rejects everything; neither is a numeric column comparison.
			return acc, budget, false
		}
		return p.appendColumnTerm(acc, f.Field, f.Op, bound, capacity, budget)
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		ms, ok := datetimeBound(f.Value)
		if !ok {
			return acc, budget, false
		}
		return p.appendColumnTerm(acc, f.Field, dtToOrdering(f.Op), float64(ms), capacity, budget)
	case FilterAnd:
		if len(f.And) == 0 {
			return acc, budget, false // an empty And matches everything; nothing to compare
		}
		for i := range f.And {
			var ok bool
			acc, budget, ok = p.appendColumnTerms(f.And[i], capacity, budget, acc)
			if !ok {
				return acc, budget, false
			}
		}
		return acc, budget, true
	default:
		return acc, budget, false
	}
}

// appendColumnTerm resolves one leaf's column and appends its term, returning
// the budget less whatever the resolution had to allocate.
func (p *payloadIndex) appendColumnTerm(acc []columnTerm, field string, op FilterOp, bound float64, capacity int, budget int64) ([]columnTerm, int64, bool) {
	if !indexNarrowable(field, op) {
		// $content is readable by the predicate but never indexed, so its posting
		// map is empty for a reason that has nothing to do with what matches — the
		// same asymmetry that makes it un-narrowable makes it un-columnisable.
		return acc, budget, false
	}
	before := p.columnBytes()
	col, ok := p.ensureColumn(field, capacity, budget)
	if !ok {
		return acc, budget, false
	}
	if budget >= 0 {
		if spent := p.columnBytes() - before; spent > 0 {
			budget -= spent
			if budget < 0 {
				budget = 0
			}
		}
	}
	return append(acc, columnTerm{vals: col, bound: bound, op: op}), budget, true
}
