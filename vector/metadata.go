// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"

	"github.com/rostamlabs/rostam/sdk/record"
)

// The Value tagged union, its ValueKind enumeration, the Metadata map, the
// Value constructors, and the ValueKind text (un)marshaling now live in the
// engine-free vtypes leaf package and are re-exported via vtypes_aliases.go.
// The two helpers below stay here because they are used only by the filter
// compiler in this package, not by the wire codec or client.

// recordResolver is the package-level record.Resolver shared by every filter
// evaluation and (from Task 6) indexing pass. 1024 rather than a smaller
// number: on clear-when-full the cache falls back to a full schema decode,
// and a single collection can carry many schema versions over its lifetime
// (every operate schema change adds one), so a small bound would thrash.
// Resolver is safe for concurrent use (its own doc comment), so one instance
// is shared across every query goroutine.
var recordResolver = record.NewResolver(1024)

// lookupPath resolves a (possibly dotted, possibly record-path) field name
// against metadata. It always tries the EXACT key first, so a flattened
// dotted key (e.g. the flat entry {"address.city": ...}) matches and
// non-dotted keys behave exactly as a raw map lookup — fully
// backward-compatible. A payload key literally named "a/b" therefore always
// wins over treating "a/b" as "payload key a, record path b".
//
// Only when there is no exact match does it try record.SplitField: if field
// splits into payloadKey/path and m[payloadKey] holds a ValueRecord, the path
// is parsed and resolved against that record's bytes via the shared
// recordResolver, and a Scalar/Count result is converted to a Value.
//
// Every other outcome — no '/' in field, no exact match and payloadKey holds
// something other than a ValueRecord, a path that fails to parse, or a
// Resolve that returns ErrPath/ErrRecord or a Table/RowPresent/Absent result
// (none of which is a scalar a filter can compare) — reports (Value{},
// false). This is deliberate even for a malformed path: the filter is
// compiled ONCE for the whole predicate tree, but the record named by
// payloadKey varies per point, so a path that is well-formed syntax but
// cannot apply to THIS point's record must degrade to "field absent" rather
// than fail the whole search. compileRowPresence is the one place a record
// path is inspected for something other than a scalar value (row presence).
func lookupPath(m Metadata, field string) (Value, bool) {
	if m == nil {
		return Value{}, false
	}
	// Exact key wins (covers flattened dotted keys, record-path-shaped keys,
	// and all plain keys).
	if v, ok := m[field]; ok {
		return v, true
	}
	payloadKey, path, ok := record.SplitField(field)
	if !ok {
		return Value{}, false
	}
	rv, ok := m[payloadKey]
	if !ok || rv.Kind != ValueRecord {
		return Value{}, false
	}
	p, err := record.ParsePath(path)
	if err != nil {
		return Value{}, false
	}
	res, err := recordResolver.Resolve(rv.Rec, p)
	if err != nil {
		return Value{}, false
	}
	return record.ResultValue(res)
}

// ErrRecordTooLarge is returned by every metadata-accepting mutation entry —
// insert, insert-if-absent, upsert, set-payload, overwrite, bulk stage/build,
// dense/IVF/named/multi-vector alike — when the payload carries a ValueRecord
// longer than maxRecordValueBytes.
//
// WHY THE CAP IS ENFORCED AT INGEST AND NOT WHERE IT IS DISCOVERED.
// maxRecordValueBytes is the SNAPSHOT/WAL codec's cap (writeValue), so a point
// whose record exceeds it can be held in memory but can never be written down:
// the WAL append would have to fail mid-record and every snapshot of the
// collection would fail for as long as the point stayed live. Accepting the
// write and failing the durability is the one outcome that leaves the in-memory
// state and the durable state permanently unable to agree, so the cap is
// checked HERE, before any state change, where refusing costs the caller one
// error and nothing else. It is a caller mistake, classified 400 on both
// transports like ErrDimMismatch.
var ErrRecordTooLarge = errors.New("vector: record payload value exceeds the storage cap")

// checkRecordValues rejects a payload carrying an oversize ValueRecord. Every
// mutation entry calls it BEFORE it touches any state or stages a WAL write;
// the inner helpers those entries share deliberately do NOT repeat it, so each
// op pays exactly one pass over its own payload.
func checkRecordValues(m Metadata) error {
	for k, v := range m {
		if v.Kind == ValueRecord && len(v.Rec) > maxRecordValueBytes {
			return fmt.Errorf("%w: payload key %q holds a %d-byte record, the cap is %d bytes",
				ErrRecordTooLarge, k, len(v.Rec), maxRecordValueBytes)
		}
	}
	return nil
}

// checkRecordValuesAll is checkRecordValues over a bulk batch, naming the
// offending row so a rejected bulk stage/build says which point to fix.
func checkRecordValuesAll(metas []Metadata) error {
	for i, m := range metas {
		if err := checkRecordValues(m); err != nil {
			return fmt.Errorf("payload %d: %w", i, err)
		}
	}
	return nil
}

// recordPoison is the READ side of the payload index's per-payload-key
// malformed-record counters (payloadIndex.badRecords / payloadIndexID.
// badRecords): it answers "does some live slot hold a record under this
// payload key that the index could not enumerate?".
//
// WHY THE GUARD NEEDS IT, AND WHY NOTHING CHEAPER WORKS. IndexEntries is
// ALL-OR-NOTHING: it walks every field of a record and returns an error if ANY
// of them is damaged. Resolve is not — it reads only the bytes on its own path
// (sdk/record/resolve.go says so), so a record with a torn tail still answers
// for a leading field: in schema mode schemaFieldOffset reaches a FIXED field
// without walking the variable-length tail at all, and in dynamic mode the
// walk stops at the named match before it reaches the damage. A partially
// damaged record therefore indexes NOTHING while lookupPath keeps resolving
// it, and every empty-set inference over a path under that payload key becomes
// a lie: "no posting" would mean "never indexed", not "no match", so an
// accelerated query drops the row — silently, and with an exact gate, without
// even a predicate re-check to catch it.
//
// Posting the entries IndexEntries produced before it failed is not an
// alternative (it produces none on error), and posting a partial list would
// leave the identical hole for every field past the damage. So the index fails
// CLOSED for the whole payload key while any live slot under it is
// unindexable. It is a STATE, not a latch: repairing or dropping the last bad
// record restores acceleration, because reindex maintains the count from both
// sides.
//
// A nil recordPoison poisons nothing — the right answer for an index that
// holds no records at all, and for a caller that has no index in hand.
type recordPoison map[string]int

func (rp recordPoison) poisoned(payloadKey string) bool { return rp[payloadKey] > 0 }

// recordShapeIndexed reports whether a PARSED record path names a location the
// payload index posts a synthetic entry for. It is the single definition of
// "indexed shape", consulted from BOTH sides — appendRecordIndexKeys asks it
// what to write (through recordPathIndexedShape, which parses first) and
// indexNarrowable asks it what the planner may read — so the two cannot drift
// into the one disagreement that matters: a shape the planner trusts but
// nothing ever posted, whose empty posting map reads as "nothing matches"
// instead of "never indexed".
//
// The indexed shapes are exactly the ones record.Resolver.IndexEntries
// enumerates: a single top-level field segment named BY NAME ("session/rc"),
// optionally with the "#count" suffix naming a table's row count
// ("session/b#count").
//
// A POSITION ("session/#0") is deliberately NOT an indexed shape, and it is
// the one place this set is narrower than the Task 6 addendum's list. Resolve
// answers a positional segment for EVERY schema-mode record, but IndexEntries
// names a names-carrying schema's fields BY NAME and only a names-LESS
// schema's by position (schemaFieldLabel). So for the common record —
// StoreNames = true — "session/#0" resolves to a value while no posting for it
// exists, which is exactly the under-report that turns into dropped rows. The
// index cannot tell from the field string which spelling a given point's
// record will produce (one collection can hold both), so the positional
// spelling stays on the predicate.
//
// Everything longer (a row "session/b/42", a cell "session/b/42/hi") is not
// indexed either: this phase never expands table contents.
func recordShapeIndexed(p record.Path) bool {
	if len(p.Segs) != 1 {
		return false
	}
	seg := p.Segs[0]
	if seg.ByPos {
		return false
	}
	return seg.Kind == record.SegField || seg.Kind == record.SegCount
}

// recordOpIndexed reports whether op's posting set over an indexed record
// shape is EXACT — the second half of the narrowing decision, after the shape.
//
// A record's scalars are posted as whole-value equality keys and NOTHING else:
// no tokens, no contains-element entries, no geo cells. So `contains`, `match`
// and `regex`, and the shape-only predicates `is_empty`, `is_null`,
// `row_exists` and `row_absent`, have no postings of their own and must stay on
// the predicate. (`ne` is narrowed by nothing, for any field anywhere, so it
// needs no case: it falls into the default with the rest.)
func recordOpIndexed(op FilterOp) bool {
	switch op {
	case FilterEq, FilterIn,
		FilterGt, FilterGte, FilterLt, FilterLte,
		FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		return true
	default:
		return false
	}
}

// recordPathIndexedShape is recordShapeIndexed for an unparsed field name: it
// splits the payload key off and parses the tail, reporting false for anything
// that is not a record path at all. Used by the WRITE side
// (appendRecordIndexKeys) to decide what to post; the read side parses once
// inside indexNarrowable instead of calling this.
func recordPathIndexedShape(field string) bool {
	_, path, ok := record.SplitField(field)
	if !ok {
		return false
	}
	p, err := record.ParsePath(path)
	if err != nil {
		return false
	}
	return recordShapeIndexed(p)
}

// numericValue extracts a float64 from a scalar numeric Value (int or float).
// Returns (0, false) for non-numeric kinds. Used by the filter compiler's
// ordering predicates to compare across int/float.
func numericValue(v Value) (float64, bool) {
	switch v.Kind {
	case ValueInt:
		return float64(v.Int), true
	case ValueFloat:
		return v.Flt, true
	default:
		// ValueRecord considered: falls here, correctly declining — a record is
		// not a scalar number.
		return 0, false
	}
}
