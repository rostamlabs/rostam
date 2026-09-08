// SPDX-License-Identifier: Apache-2.0

package vector

// The Value tagged union, its ValueKind enumeration, the Metadata map, the
// Value constructors, and the ValueKind text (un)marshaling now live in the
// engine-free vtypes leaf package and are re-exported via vtypes_aliases.go.
// The two helpers below stay here because they are used only by the filter
// compiler in this package, not by the wire codec or client.

// lookupPath resolves a (possibly dotted) field name against metadata. It
// always tries the EXACT key first, so a flattened dotted key (e.g. the flat
// entry {"address.city": ...}) matches and non-dotted keys behave exactly as a
// raw map lookup — fully backward-compatible. The dotted-path traversal of
// nested ValueObject maps is a documented follow-up: today Metadata is flat
// (no ValueObject kind), so the practical effect is exact-key lookup. Keeping
// the indirection here lets the future nested traversal slot in without
// touching every leaf op.
func lookupPath(m Metadata, field string) (Value, bool) {
	if m == nil {
		return Value{}, false
	}
	// Exact key wins (covers flattened dotted keys and all plain keys).
	if v, ok := m[field]; ok {
		return v, true
	}
	// Future: when ValueObject exists, split on '.' and traverse nested maps
	// here. No such kind today, so a non-exact dotted key simply misses.
	return Value{}, false
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
