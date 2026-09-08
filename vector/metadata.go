// SPDX-License-Identifier: Apache-2.0

package vector

import "github.com/rostamlabs/rostam/sdk/record"

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

// isRecordPath reports whether field has the "payloadKey/path" shape a
// record path filter uses: record.SplitField finds a '/' AND the tail
// parses as a record.Path. A field that does not have that shape (no '/',
// or a malformed tail) is a literal key and isRecordPath reports false for
// it — unchanged from today's behaviour.
//
// This is intentionally syntactic only: it does not (and cannot) know
// whether some point's metadata happens to carry field as an EXACT literal
// key (lookupPath's exact-key precedence would still honor that at
// evaluation time). Its sole caller, indexNarrowable, uses it to route: a
// field that is not a record path is a plain literal key the index handles
// as it always has, and a field that IS one goes on to recordPathIndexed,
// which decides per shape and per op whether postings exist to prove
// anything from. Declining is always sound (the predicate re-checks), so
// both a path-shaped-but-possibly-literal field and a genuinely literal
// path-shaped key are safe either way: the literal case indexes and resolves
// through the SAME field name on both sides.
func isRecordPath(field string) bool {
	_, path, ok := record.SplitField(field)
	if !ok {
		return false
	}
	_, err := record.ParsePath(path)
	return err == nil
}

// recordPathIndexedShape reports whether field names a record location the
// payload index POSTS a synthetic entry for. It is the single definition of
// "indexed shape", used on BOTH sides — reindex asks it what to write, and
// recordPathIndexed asks it what the planner may read — so the two can never
// drift into the one disagreement that matters: a shape the planner trusts
// but nothing ever posted, whose empty posting map would read as "nothing
// matches" instead of "never indexed".
//
// The indexed shapes are exactly the ones record.Resolver.IndexEntries
// enumerates: a single top-level field segment named BY NAME ("session/rc"),
// optionally with the "#count" suffix naming a table's row count
// ("session/b#count").
//
// A POSITION ("session/#0") is deliberately NOT an indexed shape, and it is
// the one place this set is narrower than the Task 6 addendum's list. Resolve
// answers a positional segment for EVERY schema-mode record, but IndexEntries
// names a names-carrying schema's fields by NAME and only a names-LESS
// schema's by position (schemaFieldLabel). So for the common record —
// StoreNames = true — "session/#0" resolves to a value while no posting for it
// exists, which is exactly the under-report that turns into dropped rows. The
// index cannot tell from the field string which spelling a given point's
// record will produce, so the positional spelling stays on the predicate.
//
// Everything longer (a row "session/b/42", a cell "session/b/42/hi") is not
// indexed either: this phase never expands table contents.
func recordPathIndexedShape(field string) bool {
	_, path, ok := record.SplitField(field)
	if !ok {
		return false
	}
	p, err := record.ParsePath(path)
	if err != nil || len(p.Segs) != 1 {
		return false
	}
	seg := p.Segs[0]
	if seg.ByPos {
		return false
	}
	return seg.Kind == record.SegField || seg.Kind == record.SegCount
}

// recordPathIndexed reports whether the payload index can prove anything about
// a RECORD PATH field under op — i.e. whether the postings for field are
// EXACTLY the slots whose lookupPath(field) value satisfies op.
//
// It has two halves, and both are necessary:
//
// SHAPE: recordPathIndexedShape above — only the shapes reindex actually
// posts. For any other shape the posting map is empty because nothing ever
// wrote to it, not because nothing matches, which is the same asymmetry that
// makes contentField un-narrowable.
//
// OP: only the ops whose posting set is exact for an indexed shape — equality,
// In, the ordering family, and the FilterDt* spellings that lower to it. A
// record's scalars are posted as whole-value eq keys ONLY: no tokens, no
// contains-element entries, no geo cells. So contains / match / regex, and the
// shape-only predicates is_empty / is_null / row_exists / row_absent, have no
// postings of their own and must stay on the predicate. (ne is narrowed by
// nothing, for any field anywhere, so it needs no case of its own — it simply
// falls into the default with the rest.)
//
// Callers pass a field that may or may not be a record path at all; a field
// that is not one is not this function's business, which is why
// indexNarrowable asks isRecordPath first.
func recordPathIndexed(field string, op FilterOp) bool {
	if !recordPathIndexedShape(field) {
		return false
	}
	switch op {
	case FilterEq, FilterIn,
		FilterGt, FilterGte, FilterLt, FilterLte,
		FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		return true
	default:
		// FilterNe, FilterContains, FilterMatch, FilterRegex, FilterIsEmpty,
		// FilterIsNull, FilterRowExists, FilterRowAbsent, the geo family.
		return false
	}
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
