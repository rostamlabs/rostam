// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"strings"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/record"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// KVRecordAlias is the reserved metadata key a kv_query's live value is
// presented under while the filter runs. It joins "$content" as a reserved
// name: no user field may be called this, and no filter may address it.
//
// WHY AN ALIAS AT ALL. A KV filter's fields are BARE record paths ("rc",
// "b#count", "t/3/2") because a KV value IS one record — there is no payload
// map to key into. The filter evaluator, on the other hand, only knows how to
// resolve "payloadKey/path" against a metadata map (vector.fieldLookup). So the
// leaf synthesizes the smallest map that makes the evaluator work — exactly one
// entry, this alias holding the live value as a ValueRecord — and rewrites every
// filter leaf's field from "p" to "$rec/p" to address it. Zero new evaluator
// code: record.SplitField splits at the first '/', so the alias is the payload
// key and the rest is the path the phase-1 resolver already understands.
const KVRecordAlias = "$rec"

// ErrKVQueryFilter marks a kv_query filter as one this leaf will not compile:
// a leaf addressing a reserved "$" name (the record alias itself, or any other
// name the evaluator gives special meaning), a field record.ParsePath rejects,
// or a tree over the node/depth budget.
//
// A MALFORMED PATH IS AN ERROR, NOT A FALSE LEAF. vector.newFieldLookup
// degrades an unparseable path to an exact-key lookup, which against this
// one-entry map can only miss — so without this check "rc/" or "a/b/c/d" would
// compile to a predicate that is silently false for every row, and the query
// would answer "no matches" to a question it never asked. The user gets an
// error instead.
var ErrKVQueryFilter = errors.New("ops: kv_query: invalid filter")

// BuildKVPredicate compiles a KV record filter into a predicate over the
// synthesized metadata kvMeta describes: it clones the tree once, rewriting
// every LEAF's Field from "p" to KVRecordAlias + "/" + p, and hands the clone
// to vector.CompileFilter.
//
// The clone matters: SelectorFor reads the SAME filter the caller passed and
// matches its leaves against the definition's bare path, so rewriting in place
// would leave every field prefixed and no index could ever drive a query
// again. Nothing reachable from the input is mutated — And/Or get fresh
// slices and Not a fresh pointer.
//
// A zero filter compiles to a nil predicate, which callers read as "match all"
// and skip evaluating (vector.CompileFilter's own contract).
func BuildKVPredicate(f vtypes.Filter) (vector.Predicate, error) {
	if f.IsZero() {
		return nil, nil
	}
	// Re-checked here rather than trusted: DecodeKVQueryArgs checks it for
	// every filter that arrived on the wire, but this function is exported and
	// the coordinator can reach it with a tree that never went through the args
	// decoder. The rewrite below recurses, so the depth cap is what keeps a
	// hostile tree off the goroutine stack.
	if err := wire.CheckFilterBudget(f, wire.KVQueryMaxFilterNodes, wire.KVQueryMaxFilterDepth); err != nil {
		return nil, err
	}
	rewritten, err := rewriteKVFilter(f)
	if err != nil {
		return nil, err
	}
	return vector.CompileFilter(rewritten)
}

// rewriteKVFilter returns a copy of f with every leaf field aliased. It
// recurses over composites only; every other op is a leaf, including the ones
// this build has not heard of (vector.CompileFilter rejects those by op, which
// is its job, not this function's).
func rewriteKVFilter(f vtypes.Filter) (vtypes.Filter, error) {
	switch f.Op {
	case vtypes.FilterAnd:
		out := f
		out.And = make([]vtypes.Filter, len(f.And))
		for i, c := range f.And {
			rc, err := rewriteKVFilter(c)
			if err != nil {
				return vtypes.Filter{}, err
			}
			out.And[i] = rc
		}
		return out, nil
	case vtypes.FilterOr:
		out := f
		out.Or = make([]vtypes.Filter, len(f.Or))
		for i, c := range f.Or {
			rc, err := rewriteKVFilter(c)
			if err != nil {
				return vtypes.Filter{}, err
			}
			out.Or[i] = rc
		}
		return out, nil
	case vtypes.FilterNot:
		if f.Not == nil {
			return vtypes.Filter{}, fmt.Errorf("%w: op 'not' requires a 'not' child", ErrKVQueryFilter)
		}
		inner, err := rewriteKVFilter(*f.Not)
		if err != nil {
			return vtypes.Filter{}, err
		}
		out := f
		out.Not = &inner
		return out, nil
	default:
		field, err := aliasKVField(f.Field)
		if err != nil {
			return vtypes.Filter{}, err
		}
		out := f
		out.Field = field
		return out, nil
	}
}

// aliasKVField validates one leaf's bare record path and returns it aliased.
func aliasKVField(field string) (string, error) {
	if field == "" {
		return "", fmt.Errorf("%w: a leaf requires a field", ErrKVQueryFilter)
	}
	// Leading '$' is reserved WHOLESALE, not just for this alias. The metadata
	// map the predicate runs against holds the live record under KVRecordAlias,
	// so a field of exactly "$rec" would compare against the whole blob; and
	// "$content" already means something else to the evaluator. Refusing the
	// whole namespace means a future reserved name cannot silently change what
	// an existing query matches.
	if strings.HasPrefix(field, "$") {
		return "", fmt.Errorf("%w: field %q addresses the reserved %q namespace", ErrKVQueryFilter, field, "$")
	}
	if _, err := record.ParsePath(field); err != nil {
		return "", fmt.Errorf("%w: field %q: %v", ErrKVQueryFilter, field, err)
	}
	return KVRecordAlias + "/" + field, nil
}

// SelectorFor returns the leaf of f that d's postings can drive, if there is
// one: the filter root when it is a positive leaf on d's path, else the first
// such child of a TOP-LEVEL `and`. `eq` wins over a range when both are
// present, because an equality lookup is one map probe while a range walks
// every distinct value.
//
// IT NEVER DESCENDS INTO `or` OR `not`, and never into a nested `and`. This is
// the whole supersetness argument (constraints.md): a candidate set narrows
// soundly only when EVERY record satisfying the filter satisfies the selector
// too. A conjunct qualifies — an `and` is false unless all of its children are
// true. A disjunct does not: a record can satisfy an `or` through its other
// branch, have no posting under this leaf's value, and be lost. A negation is
// the same failure the other way round. Both are left to the predicate, which
// re-checks them on the live value and can only ever remove rows.
//
// The bounds it accepts are scalars and nothing else. A record path resolves to
// a scalar or to nothing at all (record.ResultValue), and only a scalar has a
// posting — so a scalar bound is the only one with a superset proof behind it.
// Declining the rest costs an ErrKVQueryScanRequired refusal, never a row.
func SelectorFor(f vtypes.Filter, d kvindex.Def) (kvindex.Selector, bool) {
	if sel, ok := kvLeafSelector(f, d); ok {
		return sel, true
	}
	if f.Op != vtypes.FilterAnd {
		return kvindex.Selector{}, false
	}
	var best kvindex.Selector
	found := false
	for _, c := range f.And {
		sel, ok := kvLeafSelector(c, d)
		if !ok {
			continue
		}
		if !found {
			best, found = sel, true
			continue
		}
		// First match wins, except that an eq displaces a non-eq: one probe
		// beats a walk over every distinct value in the index.
		if best.Op != vtypes.FilterEq && sel.Op == vtypes.FilterEq {
			best = sel
		}
	}
	return best, found
}

// kvLeafSelector turns one leaf into a selector for d, or declines.
//
// The field must equal d.PathText EXACTLY — a definition on "b#count" indexes
// the row count, a leaf on "b" addresses the field itself, and the two are
// different questions with different postings.
func kvLeafSelector(f vtypes.Filter, d kvindex.Def) (kvindex.Selector, bool) {
	if f.Field == "" || f.Field != d.PathText {
		return kvindex.Selector{}, false
	}
	switch f.Op {
	case vtypes.FilterEq, vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte:
		if !kvSelectableBound(f.Value) {
			return kvindex.Selector{}, false
		}
		return kvindex.Selector{Def: d, Op: f.Op, Values: []vtypes.Value{f.Value}}, true
	case vtypes.FilterIn:
		vals := expandKVInValues(f.Value)
		if len(vals) == 0 {
			return kvindex.Selector{}, false
		}
		return kvindex.Selector{Def: d, Op: vtypes.FilterIn, Values: vals}, true
	default:
		// ne, contains, is_empty, is_null, match, regex, the dt_* and geo_*
		// families, row_exists/row_absent: none is a positive scalar equality
		// or ordering on this path, so none has a posting set that is a
		// superset of its matches.
		return kvindex.Selector{}, false
	}
}

// kvSelectableBound reports whether v is a bound the postings can be compared
// against: the four scalar kinds kvindex.scalarKeyOf posts under. ValueNone,
// the list kinds, a geo point and a record blob are all declined.
func kvSelectableBound(v vtypes.Value) bool {
	switch v.Kind {
	case vtypes.ValueString, vtypes.ValueInt, vtypes.ValueFloat, vtypes.ValueBool:
		return true
	default:
		return false
	}
}

// expandKVInValues flattens an `in` leaf's list value into the N SCALAR values
// kvindex.Selector carries — its posting lookup keys on one scalar at a time,
// so the list has to be taken apart here rather than in the index.
//
// The result is bounded by the filter's own byte cap (wire.KVQueryMaxFilterBytes,
// 64 KiB), so the allocation is bounded by a frame the decoder already capped.
func expandKVInValues(v vtypes.Value) []vtypes.Value {
	switch v.Kind {
	case vtypes.ValueStrings:
		out := make([]vtypes.Value, 0, len(v.Strs))
		for _, s := range v.Strs {
			out = append(out, vtypes.NewString(s))
		}
		return out
	case vtypes.ValueInts:
		out := make([]vtypes.Value, 0, len(v.Ints))
		for _, n := range v.Ints {
			out = append(out, vtypes.NewInt(n))
		}
		return out
	case vtypes.ValueFloats:
		out := make([]vtypes.Value, 0, len(v.Flts))
		for _, x := range v.Flts {
			out = append(out, vtypes.NewFloat(x))
		}
		return out
	default:
		// vector.compileIn refuses a non-list `in` outright, so the predicate
		// side is an error, not a false leaf — nothing is lost by declining
		// the selector here as well.
		return nil
	}
}
