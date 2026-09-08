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
