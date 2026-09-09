// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"bytes"
	"fmt"

	"github.com/rostamlabs/rostam/sdk/record"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// Def is a KV index definition this node can actually post on: a validated
// wire.KVIndexDef with its payload path already PARSED, so the write path
// spends no work re-parsing a constant.
type Def struct {
	// Name identifies the index (a kv_query names it; the meta log keys on it).
	Name string
	// Prefix scopes the index to keys sharing these bytes. Empty covers the
	// whole keyspace. Always a private copy of the definition's KeyPrefix.
	Prefix []byte
	// PathText is the path as written, kept for equality checks (Install
	// compares it) and for diagnostics.
	PathText string
	// Path is PathText parsed. A single segment: a top-level field, or that
	// field's "#count".
	Path record.Path
	// Kind is wire.KVIndexKindScalar or wire.KVIndexKindCount, agreeing with
	// Path by construction.
	Kind uint8
	// MetaIndex is the meta-log index of the definition this Def was built
	// from, so a node can tell a re-issued definition from a stale one.
	MetaIndex uint64
}

// DefFrom validates d and parses its payload path.
//
// THIS IS WHERE A PATH IS REALLY VALIDATED. wire.KVIndexDef.Validate is a
// SHAPE check only — sdk/record imports sdk/wire, so sdk/wire cannot import
// record to parse the path, and it settles for "no '/', a #count suffix that
// agrees with Kind, and the length caps". Everything the grammar says beyond
// that (the field-name charset, canonical "#N" positions, the position cap,
// a bare "#count" with no field in front of it) is caught here and nowhere
// else. A definition the meta FSM accepted but this build cannot parse is
// therefore rejected at install time, counted as a reject, and simply not
// installed — never silently installed as an index that posts nothing.
func DefFrom(d wire.KVIndexDef, metaIndex uint64) (Def, error) {
	if err := d.Validate(); err != nil {
		return Def{}, err
	}
	p, err := record.ParsePath(d.PayloadPath)
	if err != nil {
		return Def{}, fmt.Errorf("%w: index %q: %w", wire.ErrKVIndexDef, d.Name, err)
	}
	if len(p.Segs) != 1 {
		return Def{}, fmt.Errorf("%w: index %q: path %q has %d segments; an index path is one top-level field or its #count",
			wire.ErrKVIndexDef, d.Name, d.PayloadPath, len(p.Segs))
	}
	isCount := p.Segs[0].Kind == record.SegCount
	if !isCount && p.Segs[0].Kind != record.SegField {
		return Def{}, fmt.Errorf("%w: index %q: path %q is not a top-level field", wire.ErrKVIndexDef, d.Name, d.PayloadPath)
	}
	// Kind is carried and validated, not dispatched on: record.Resolve already
	// reports Scalar vs Count, so a disagreement is caught once, here, instead
	// of mis-indexing every write thereafter. Validate checked the SPELLING of
	// the suffix; this checks the PARSE, which is the authority.
	switch d.Kind {
	case wire.KVIndexKindScalar:
		if isCount {
			return Def{}, fmt.Errorf("%w: index %q: path %q is a #count path but Kind is scalar", wire.ErrKVIndexDef, d.Name, d.PayloadPath)
		}
	case wire.KVIndexKindCount:
		if !isCount {
			return Def{}, fmt.Errorf("%w: index %q: path %q is not a #count path but Kind is count", wire.ErrKVIndexDef, d.Name, d.PayloadPath)
		}
	default:
		return Def{}, fmt.Errorf("%w: index %q: unknown kind %d", wire.ErrKVIndexDef, d.Name, d.Kind)
	}
	return Def{
		Name: d.Name,
		// COPIED, not aliased: KeyPrefix arrives from a decoder's buffer, and a
		// definition that could be re-scoped by whoever still holds that buffer
		// would silently change which keys an index answers for.
		Prefix:    append([]byte(nil), d.KeyPrefix...),
		PathText:  d.PayloadPath,
		Path:      p,
		Kind:      d.Kind,
		MetaIndex: metaIndex,
	}, nil
}

// sameShape reports whether two definitions post the SAME thing about the
// same keys — everything Install must compare to decide whether an existing
// posting set is still valid. MetaIndex is deliberately excluded: a
// re-issued, byte-identical definition at a newer meta index describes the
// same postings, and throwing them away would cost a full backfill for
// nothing.
func sameShape(a, b Def) bool {
	return a.Name == b.Name &&
		a.PathText == b.PathText &&
		a.Kind == b.Kind &&
		bytes.Equal(a.Prefix, b.Prefix)
}

// scalarKey is a comparable map key for an equality-indexable scalar Value.
// Only the field matching kind is meaningful.
//
// TWIN of vector/payload_index.go's scalarKey/scalarKeyOf: the two index the
// same vtypes.Value space under the same rules and MUST be kept in lockstep
// (the phase-1 minUvarintLen precedent — a comment on each side naming the
// other). The duplication is deliberate: this package is engine-free, and
// pulling in vector to share five fields would drag the whole engine into a
// package the write path calls on every Put.
type scalarKey struct {
	kind vtypes.ValueKind
	str  string
	i    int64
	f    float64
	b    bool
}

// scalarKeyOf returns the posting key for a scalar Value, or ok=false for the
// kinds that are not equality-indexable (none, lists, geo, record) — and for
// the one VALUE that is not indexable either: NaN.
//
// NaN IS NOT A MAP KEY. A scalarKey carrying NaN hashes fine but never
// compares equal to itself, so vals[key] after vals[key] = set misses: every
// write of a NaN-valued field would append a NEW, permanently unreachable
// posting entry that the reverse map could never remove. Declining is also
// the right SEMANTICS, which is what keeps supersetness true: under IEEE
// (orderingHoldsFloat) a NaN field satisfies no range predicate, and under
// vtypes.Value.Equal it satisfies no equality predicate either, so a NaN
// value has no query that could match it. "Not indexed" and "matches nothing"
// say the same thing. A NaN BOUND is declined by the same rule on the query
// side, with the same consequence: no candidates, and a predicate that would
// have rejected every one of them anyway.
func scalarKeyOf(v vtypes.Value) (scalarKey, bool) {
	switch v.Kind {
	case vtypes.ValueString:
		return scalarKey{kind: vtypes.ValueString, str: v.Str}, true
	case vtypes.ValueInt:
		return scalarKey{kind: vtypes.ValueInt, i: v.Int}, true
	case vtypes.ValueFloat:
		if v.Flt != v.Flt { // NaN
			return scalarKey{}, false
		}
		return scalarKey{kind: vtypes.ValueFloat, f: v.Flt}, true
	case vtypes.ValueBool:
		return scalarKey{kind: vtypes.ValueBool, b: v.Bool}, true
	default:
		// ValueNone, the three list kinds, ValueGeo and ValueRecord: none has an
		// equality key here. A record's own fields get their postings through
		// their own definitions, not through the blob.
		return scalarKey{}, false
	}
}

// numericKey is numericValue (vector/metadata.go) for a posting key: the two
// numeric kinds interoperate as float64 and nothing else is a number. Twin —
// keep in lockstep with vector.numericValue.
func numericKey(k scalarKey) (float64, bool) {
	switch k.kind {
	case vtypes.ValueInt:
		return float64(k.i), true
	case vtypes.ValueFloat:
		return k.f, true
	default:
		return 0, false
	}
}

// numericBound is numericValue for the query side: the same two kinds.
func numericBound(v vtypes.Value) (float64, bool) {
	switch v.Kind {
	case vtypes.ValueInt:
		return float64(v.Int), true
	case vtypes.ValueFloat:
		return v.Flt, true
	default:
		return 0, false
	}
}

// orderingHoldsFloat decides a NUMERIC range comparison, and it is a TWIN of
// vector/filter.go's function of the same name — keep the two in lockstep. A
// NaN operand makes the pair UNORDERED, so gt/gte/lt/lte are ALL false; that
// is IEEE-754's rule and what Go's own < and > do. If these two ever disagree
// the index stops being a superset of the predicate, which is the one failure
// mode this package cannot tolerate.
func orderingHoldsFloat(op vtypes.FilterOp, a, b float64) bool {
	// a != a is the allocation-free, math-import-free NaN test.
	if a != a || b != b {
		return false
	}
	switch {
	case a < b:
		return orderingHolds(op, -1)
	case a > b:
		return orderingHolds(op, 1)
	default:
		return orderingHolds(op, 0)
	}
}

// orderingHolds maps a -1/0/+1 comparison result to the op's truth value.
// Twin of vector/filter.go's orderingHolds.
func orderingHolds(op vtypes.FilterOp, cmp int) bool {
	switch op {
	case vtypes.FilterGt:
		return cmp > 0
	case vtypes.FilterGte:
		return cmp >= 0
	case vtypes.FilterLt:
		return cmp < 0
	case vtypes.FilterLte:
		return cmp <= 0
	default:
		return false
	}
}
