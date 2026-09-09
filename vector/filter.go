// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/rostamlabs/rostam/sdk/record"
)

// Predicate is a compiled filter: given a vector's metadata (nil if the
// vector has none), it reports whether the vector matches.
type Predicate func(m Metadata) bool

// CompileFilter turns a filter into an executable Predicate. It lives in vector
// (not vtypes) because Predicate is engine machinery; the leaf stays
// behavior-free. It walks the filter once and returns a Predicate closure. A
// zero/empty filter compiles to a nil Predicate — callers treat nil as "match
// all" and skip evaluation entirely (the search hot path relies on this).
func CompileFilter(f Filter) (Predicate, error) {
	if f.IsZero() {
		return nil, nil
	}
	return compileNode(f)
}

func compileNode(f Filter) (Predicate, error) {
	switch f.Op {
	case FilterAnd:
		preds, err := compileChildren(f.And)
		if err != nil {
			return nil, err
		}
		return func(m Metadata) bool {
			for _, p := range preds {
				if !p(m) {
					return false
				}
			}
			return true
		}, nil
	case FilterOr:
		preds, err := compileChildren(f.Or)
		if err != nil {
			return nil, err
		}
		return func(m Metadata) bool {
			for _, p := range preds {
				if p(m) {
					return true
				}
			}
			return false
		}, nil
	case FilterNot:
		if f.Not == nil {
			return nil, fmt.Errorf("vector: filter op 'not' requires a 'not' child")
		}
		inner, err := compileNode(*f.Not)
		if err != nil {
			return nil, err
		}
		return func(m Metadata) bool { return !inner(m) }, nil
	case FilterEq, FilterNe, FilterGt, FilterGte, FilterLt, FilterLte, FilterIn, FilterContains,
		FilterMatch, FilterRegex, FilterIsEmpty, FilterIsNull,
		FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte,
		FilterGeoRadius, FilterGeoBox, FilterGeoPolygon,
		FilterRowExists, FilterRowAbsent:
		return compileLeaf(f)
	default:
		return nil, fmt.Errorf("vector: unknown filter op %d", f.Op)
	}
}

func compileChildren(children []Filter) ([]Predicate, error) {
	preds := make([]Predicate, 0, len(children))
	for i := range children {
		p, err := compileNode(children[i])
		if err != nil {
			return nil, err
		}
		preds = append(preds, p)
	}
	return preds, nil
}

func compileLeaf(f Filter) (Predicate, error) {
	if f.Field == "" {
		return nil, fmt.Errorf("vector: leaf filter op %q requires a field", mustOpName(f.Op))
	}
	field := f.Field
	want := f.Value
	// The record path in field (when it has one) is a compile-time constant:
	// newFieldLookup parses it ONCE here so the per-point work in the closures
	// below is a map lookup plus at most one Resolve. See fieldLookup.
	look := newFieldLookup(field)

	switch f.Op {
	case FilterEq:
		return func(m Metadata) bool {
			got, ok := look.get(m)
			return ok && got.Equal(want)
		}, nil
	case FilterNe:
		// Strict-exists: a missing field does NOT satisfy 'ne'.
		return func(m Metadata) bool {
			got, ok := look.get(m)
			return ok && !got.Equal(want)
		}, nil
	case FilterGt, FilterGte, FilterLt, FilterLte:
		return compileOrdering(field, f.Op, want)
	case FilterIn:
		return compileIn(field, want)
	case FilterContains:
		return compileContains(field, want)
	case FilterMatch:
		return compileMatch(field, want)
	case FilterRegex:
		return compileRegex(field, want)
	case FilterIsEmpty:
		return compileIsEmpty(field)
	case FilterIsNull:
		return compileIsNull(field)
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		return compileDatetime(field, f.Op, want)
	case FilterGeoRadius, FilterGeoBox, FilterGeoPolygon:
		return compileGeo(field, f.Op, f.Geo)
	case FilterRowExists, FilterRowAbsent:
		return compileRowPresence(f)
	default:
		return nil, fmt.Errorf("vector: compileLeaf called with non-leaf op %d", f.Op)
	}
}

// compileOrdering builds a gt/gte/lt/lte predicate. Numeric fields compare as
// float64 (int and float interoperate); string fields compare lexicographically.
// A kind mismatch or missing field evaluates to false.
func compileOrdering(field string, op FilterOp, want Value) (Predicate, error) {
	look := newFieldLookup(field)
	return func(m Metadata) bool {
		got, ok := look.get(m)
		if !ok {
			return false
		}
		// Numeric path.
		if gv, gok := numericValue(got); gok {
			wv, wok := numericValue(want)
			if !wok {
				return false
			}
			return orderingHoldsFloat(op, gv, wv)
		}
		// String path.
		if got.Kind == ValueString && want.Kind == ValueString {
			return orderingHolds(op, compareString(got.Str, want.Str))
		}
		return false
	}, nil
}

// orderingHolds maps a -1/0/+1 comparison result to the op's truth value.
func orderingHolds(op FilterOp, cmp int) bool {
	switch op {
	case FilterGt:
		return cmp > 0
	case FilterGte:
		return cmp >= 0
	case FilterLt:
		return cmp < 0
	case FilterLte:
		return cmp <= 0
	}
	return false
}

// orderingHoldsFloat is orderingHolds for a NUMERIC comparison, and it is the
// single place Rostam's NaN range semantics live.
//
// THE SEMANTICS: a NaN operand makes the pair UNORDERED, so gt / gte / lt / lte
// are ALL false. A NaN-valued field satisfies no range predicate, and a NaN
// bound is satisfied by nothing. This is IEEE-754's own rule — it is exactly
// what Go's `<` and `>` do on float64, what Rust's PartialOrd returns (None),
// and what Milvus and Qdrant answer, because they too lower a range predicate to
// their host language's float comparison. It is NOT what PostgreSQL or Lucene
// do: both impose a TOTAL order that sorts NaN above +Inf so a btree/doc-values
// index can be built over it. Rostam is in the first camp deliberately (see
// below), and this comment is the record of that choice.
//
// WHY IT CHANGED, AND WHY IT IS A BUG FIX RATHER THAN A PREFERENCE. compareFloat
// classifies "neither < nor >" as EQUAL, which is right for every pair of real
// numbers and wrong for NaN: it made `score >= 3` and `score <= 3` BOTH accept a
// NaN-valued field. The payload index disagreed — orderingSet binary-searches a
// key list sorted by `<`, under which NaN has no position, so the NaN slot was
// never in the posting set. Predicate and index therefore answered different
// questions, and the index answered the NARROWER one: filter-first could already
// drop a NaN-valued point from a >= result, and the m4 admission gate had to
// refuse the whole ordering family (narrowUnproven) to avoid turning that rare
// wrong row into a wrong row on every filtered traversal.
//
// Adopting IEEE's answer removes the disagreement AT THE ROOT rather than
// papering over it: the predicate now rejects NaN, the index already did, and
// the ordering family becomes provably EXACT — which is what lets the gate use
// it at all. The alternative (index NaN keys explicitly and keep the predicate
// accepting them for >= and <=) would have had to invent an ordering for a value
// that has none, and would still leave `x >= 3 AND x <= 2` matching a NaN row.
//
// AFFECTED BEHAVIOUR, stated plainly for the changelog: a point whose numeric
// field is NaN previously matched `field >= b` and `field <= b` for EVERY bound
// b, and now matches no range predicate at all. Nothing else moves — Eq/Ne/In
// already compared NaN with `==` (never equal to itself) and are untouched.
func orderingHoldsFloat(op FilterOp, a, b float64) bool {
	// a != a is the allocation-free, math-import-free NaN test; the compiler
	// lowers it to the same UCOMISD parity check math.IsNaN compiles to.
	if a != a || b != b {
		return false
	}
	return orderingHolds(op, compareFloat(a, b))
}

func compareFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func compareString(a, b string) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

// compileIn builds an 'in' predicate: the field's scalar value must be a
// member of the want array (strings/ints/floats). Type checked at compile.
func compileIn(field string, want Value) (Predicate, error) {
	look := newFieldLookup(field)
	switch want.Kind {
	case ValueStrings:
		set := want.Strs
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueString {
				return false
			}
			for _, s := range set {
				if s == got.Str {
					return true
				}
			}
			return false
		}, nil
	case ValueInts:
		set := want.Ints
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueInt {
				return false
			}
			for _, n := range set {
				if n == got.Int {
					return true
				}
			}
			return false
		}, nil
	case ValueFloats:
		set := want.Flts
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueFloat {
				return false
			}
			for _, x := range set {
				if x == got.Flt {
					return true
				}
			}
			return false
		}, nil
	default:
		return nil, fmt.Errorf("vector: filter op 'in' requires an array value (strings/ints/floats), got kind %d", want.Kind)
	}
}

// compileContains builds a 'contains' predicate: the field must be an array
// (strings/ints/floats) containing the want scalar. Type checked at compile.
func compileContains(field string, want Value) (Predicate, error) {
	look := newFieldLookup(field)
	switch want.Kind {
	case ValueString:
		needle := want.Str
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueStrings {
				return false
			}
			for _, s := range got.Strs {
				if s == needle {
					return true
				}
			}
			return false
		}, nil
	case ValueInt:
		needle := want.Int
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueInts {
				return false
			}
			for _, n := range got.Ints {
				if n == needle {
					return true
				}
			}
			return false
		}, nil
	case ValueFloat:
		needle := want.Flt
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueFloats {
				return false
			}
			for _, x := range got.Flts {
				if x == needle {
					return true
				}
			}
			return false
		}, nil
	default:
		return nil, fmt.Errorf("vector: filter op 'contains' requires a scalar value (string/int/float), got kind %d", want.Kind)
	}
}

// elementKeysOf returns the contains-index posting keys for a metadata Value:
// one scalarKey per DISTINCT element of an array value (ValueStrings →
// scalarKey{ValueString,str}, ValueInts → scalarKey{ValueInt,i}, ValueFloats →
// scalarKey{ValueFloat,f}); a non-array value contributes none.
//
// This is the SOUNDNESS linchpin of the contains inverted index: the keys here
// are EXACTLY the values compileContains scans for membership over (the predicate
// does `s == needle` / `n == needle` / `x == needle` element-by-element, and
// scalarKeyOf(want) builds the lookup key the SAME way). Because the scalarKey
// carries the element's kind, a string `want` only collides with a ValueStrings
// element key — mirroring the predicate's `got.Kind != ValueStrings` kind-check.
// So contains[field][scalarKeyOf(want)] is exactly {docs whose array contains an
// element == want} = a no-false-negative superset of the predicate matches.
// Dedup within the value keeps the posting + reverse list distinct (so the
// per-entry drop on reindex/delete stays exact), mirroring addTokens' discipline.
func elementKeysOf(v Value) []scalarKey {
	switch v.Kind {
	case ValueStrings:
		if len(v.Strs) == 0 {
			return nil
		}
		seen := make(map[string]struct{}, len(v.Strs))
		out := make([]scalarKey, 0, len(v.Strs))
		for _, s := range v.Strs {
			if _, dup := seen[s]; dup {
				continue
			}
			seen[s] = struct{}{}
			out = append(out, scalarKey{kind: ValueString, str: s})
		}
		return out
	case ValueInts:
		if len(v.Ints) == 0 {
			return nil
		}
		seen := make(map[int64]struct{}, len(v.Ints))
		out := make([]scalarKey, 0, len(v.Ints))
		for _, n := range v.Ints {
			if _, dup := seen[n]; dup {
				continue
			}
			seen[n] = struct{}{}
			out = append(out, scalarKey{kind: ValueInt, i: n})
		}
		return out
	case ValueFloats:
		if len(v.Flts) == 0 {
			return nil
		}
		seen := make(map[float64]struct{}, len(v.Flts))
		out := make([]scalarKey, 0, len(v.Flts))
		for _, x := range v.Flts {
			if _, dup := seen[x]; dup {
				continue
			}
			seen[x] = struct{}{}
			out = append(out, scalarKey{kind: ValueFloat, f: x})
		}
		return out
	default:
		return nil
	}
}

// compileMatch builds a full-text-lite predicate: the field's token set must
// be a superset of the query's token set (case-insensitive, whole-token).
// Tokenizing the query happens ONCE at compile time; per-candidate eval only
// tokenizes the field value (which is unavoidable without an index). An empty
// query matches any present string/strings field. Missing/wrong-kind → false.
func compileMatch(field string, want Value) (Predicate, error) {
	if want.Kind != ValueString {
		return nil, fmt.Errorf("vector: filter op 'match' requires a string value, got kind %d", want.Kind)
	}
	queryTokens := tokenize(want.Str)
	look := newFieldLookup(field)
	return func(m Metadata) bool {
		got, ok := look.get(m)
		if !ok {
			return false
		}
		switch got.Kind {
		case ValueString:
			return fieldContainsAllTokens(got.Str, queryTokens)
		case ValueStrings:
			for _, s := range got.Strs {
				if fieldContainsAllTokens(s, queryTokens) {
					return true
				}
			}
			return false
		default:
			return false
		}
	}, nil
}

// fieldContainsAllTokens reports whether every token in queryTokens (already
// lowercased by tokenize) appears as a token of field. It is the allocation-free
// equivalent of tokensContainAll(tokenize(field), queryTokens): it walks field's
// word runs in place (no []string) and case-folds each on the fly (no per-token
// ToLower), so a filtered search no longer allocates per candidate. Byte-identical
// to the old path — same word boundaries (eachFieldToken mirrors tokenize's
// FieldsFunc split) and same lowercase comparison (foldEqualLower applies the
// per-rune unicode.ToLower that strings.ToLower applies).
func fieldContainsAllTokens(field string, queryTokens []string) bool {
	if len(queryTokens) == 0 {
		return true // empty query is trivially satisfied (matches tokensContainAll)
	}
	if len(queryTokens) > 64 {
		// Rare: fall back to the allocating path for the bitmask's 64-token limit.
		return tokensContainAll(tokenize(field), queryTokens)
	}
	want := ^uint64(0) >> (64 - len(queryTokens)) // one bit per query token
	var matched uint64
	done := false
	eachFieldToken(field, func(word string) bool {
		for i, qt := range queryTokens {
			if matched&(1<<uint(i)) != 0 {
				continue
			}
			if foldEqualLower(word, qt) {
				matched |= 1 << uint(i)
				if matched == want {
					done = true
					return true // all query tokens matched — stop the walk
				}
			}
		}
		return false
	})
	return done || matched == want
}

// eachFieldToken invokes fn for each word run of s (a run of Unicode letters/
// digits), stopping early if fn returns true. The split is byte-identical to
// strings.FieldsFunc(s, sep) with sep = "not letter and not digit" — i.e. tokenize
// without the []string allocation. The yielded word is a substring of s (no copy).
func eachFieldToken(s string, fn func(word string) bool) {
	start := -1
	for i, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 {
			if fn(s[start:i]) {
				return
			}
			start = -1
		}
	}
	if start >= 0 {
		fn(s[start:])
	}
}

// foldEqualLower reports whether word, lowercased per-rune via unicode.ToLower,
// equals lower (which tokenize already lowercased the same way). Equivalent to
// strings.ToLower(word) == lower but allocation-free.
func foldEqualLower(word, lower string) bool {
	for _, lr := range lower {
		if word == "" {
			return false
		}
		wr, size := utf8.DecodeRuneInString(word)
		if unicode.ToLower(wr) != lr {
			return false
		}
		word = word[size:]
	}
	return word == ""
}

// tokensContainAll reports whether every token in need appears in have. An
// empty need is trivially satisfied (matches any present field value).
func tokensContainAll(have, need []string) bool {
	for _, n := range need {
		found := false
		for _, h := range have {
			if h == n {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// tokenize splits s into lowercased word tokens on runs of Unicode
// non-alphanumeric characters. Used by FilterMatch (full-text-lite). The
// returned slice is freshly allocated; callers own it.
func tokenize(s string) []string {
	if s == "" {
		return nil
	}
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	for i, f := range fields {
		fields[i] = strings.ToLower(f)
	}
	return fields
}

// compileRegex builds an RE2 regex predicate. The pattern is compiled ONCE
// here (an invalid pattern surfaces as the Compile() error, never a
// per-candidate panic); the compiled *regexp.Regexp is captured in the
// closure. Matches a string field; for a strings field, ANY element matching
// satisfies. Missing/wrong-kind → false.
func compileRegex(field string, want Value) (Predicate, error) {
	if want.Kind != ValueString {
		return nil, fmt.Errorf("vector: filter op 'regex' requires a string value, got kind %d", want.Kind)
	}
	re, err := regexp.Compile(want.Str)
	if err != nil {
		return nil, fmt.Errorf("vector: filter op 'regex' has invalid pattern %q: %w", want.Str, err)
	}
	look := newFieldLookup(field)
	return func(m Metadata) bool {
		got, ok := look.get(m)
		if !ok {
			return false
		}
		switch got.Kind {
		case ValueString:
			return re.MatchString(got.Str)
		case ValueStrings:
			for _, s := range got.Strs {
				if re.MatchString(s) {
					return true
				}
			}
			return false
		default:
			return false
		}
	}, nil
}

// compileIsEmpty builds an is_empty predicate: true iff the field is absent,
// present with ValueNone, an empty string, or an empty array. It returns an
// error for signature uniformity with the other leaf compilers; it never fails.
func compileIsEmpty(field string) (Predicate, error) {
	look := newFieldLookup(field)
	return func(m Metadata) bool {
		got, ok := look.get(m)
		if !ok {
			return true
		}
		switch got.Kind {
		case ValueNone:
			return true
		case ValueString:
			return got.Str == ""
		case ValueStrings:
			return len(got.Strs) == 0
		case ValueInts:
			return len(got.Ints) == 0
		case ValueFloats:
			return len(got.Flts) == 0
		case ValueGeo:
			// A geo point is never "empty": it always carries a lat/lon pair (a
			// zero lat/lon is the valid point 0,0 off the African coast, not an
			// absence). Made explicit instead of relying on the default below so a
			// future kind can't silently inherit the wrong answer here.
			return false
		case ValueRecord:
			// A record's own emptiness is its byte length, mirroring the
			// ValueString case above — an explicit case rather than falling to
			// the default (which would wrongly answer "never empty" the way a
			// geo point never is).
			return len(got.Rec) == 0
		default:
			return false
		}
	}, nil
}

// compileIsNull builds an is_null predicate: true iff the field is PRESENT and
// its kind is ValueNone (an explicit null). An absent field is NOT null —
// that's the is_empty/is_null distinction. It returns an error for signature
// uniformity with the other leaf compilers; it never fails.
//
// ValueRecord considered: this checks presence + ValueNone only (no per-kind
// switch), so a present record already answers correctly — not null — with no
// change needed here.
func compileIsNull(field string) (Predicate, error) {
	look := newFieldLookup(field)
	return func(m Metadata) bool {
		got, ok := look.get(m)
		return ok && got.Kind == ValueNone
	}, nil
}

// compileDatetime builds a datetime comparison predicate. The RFC3339 literal
// is parsed ONCE here into an int64 unix-millisecond bound (invalid → the
// Compile() error, never a per-candidate panic); the bound is captured in the
// closure. The field is read via numericValue (datetime stored as int64
// unix-ms by convention). Missing/non-numeric field → false.
func compileDatetime(field string, op FilterOp, want Value) (Predicate, error) {
	look := newFieldLookup(field)
	ms, ok := datetimeBound(want)
	if !ok {
		if want.Kind != ValueString {
			return nil, fmt.Errorf("vector: filter op %q requires an RFC3339 string value, got kind %d", mustOpName(op), want.Kind)
		}
		return nil, fmt.Errorf("vector: filter op %q has invalid RFC3339 value %q", mustOpName(op), want.Str)
	}
	bound := float64(ms)
	return func(m Metadata) bool {
		got, ok := look.get(m)
		if !ok {
			return false
		}
		gv, gok := numericValue(got)
		if !gok {
			return false
		}
		// Same NaN rule as the plain ordering ops (orderingHoldsFloat): the bound
		// is an int64-derived float and so never NaN, but the FIELD can be — a
		// float64 payload on a datetime-typed field — and it must not compare
		// equal to every instant.
		return orderingHoldsFloat(dtToOrdering(op), gv, bound)
	}, nil
}

// datetimeBound parses a datetime op's RFC3339 string Value into an int64 unix
// MILLISECOND bound. It is the SINGLE source of truth for the datetime→int64-ms
// lowering: both the filter compiler (compileDatetime, the authoritative
// rejection of invalid filters) and the payload index's range path
// (collectRangeSets) call it, so the compiled predicate and the index narrowing
// can never disagree on the bound. ok=false for a non-string Value or an
// unparseable RFC3339 string — the compiler turns that into the Compile() error;
// the index simply declines to narrow (a wider candidate set is always safe).
func datetimeBound(v Value) (int64, bool) {
	if v.Kind != ValueString {
		return 0, false
	}
	t, err := time.Parse(time.RFC3339, v.Str)
	if err != nil {
		return 0, false
	}
	return t.UnixMilli(), true
}

// dtToOrdering maps a datetime op to its plain ordering counterpart so the
// shared orderingHolds logic applies.
func dtToOrdering(op FilterOp) FilterOp {
	switch op {
	case FilterDtGt:
		return FilterGt
	case FilterDtGte:
		return FilterGte
	case FilterDtLt:
		return FilterLt
	case FilterDtLte:
		return FilterLte
	}
	return op
}

// compileGeo builds a geo predicate (radius / bounding-box / polygon). ALL
// validation happens ONCE here — a nil Geo, out-of-range coordinate,
// non-positive radius, inverted box, or a polygon with bad arity surfaces as
// the Compile() error, NEVER a per-candidate panic. The returned closure is
// allocation-free: it captures the parsed condition (and, for polygon, the
// slice itself) and only reads the field via lookupPath, requiring a ValueGeo
// (missing field or non-geo kind -> false).
func compileGeo(field string, op FilterOp, g *GeoCondition) (Predicate, error) {
	look := newFieldLookup(field)
	if g == nil {
		return nil, fmt.Errorf("vector: filter op %q requires a 'geo' condition", mustOpName(op))
	}
	switch op {
	case FilterGeoRadius:
		if err := validateLatLon(g.CenterLat, g.CenterLon); err != nil {
			return nil, fmt.Errorf("vector: filter op %q center: %w", mustOpName(op), err)
		}
		// !(>0) rejects NaN (all NaN comparisons are false); IsInf rejects +Inf.
		// A NaN radius would otherwise compile to a silent never-match and +Inf to
		// a silent match-all.
		if !(g.RadiusM > 0) || math.IsInf(g.RadiusM, 1) {
			return nil, fmt.Errorf("vector: filter op %q requires a finite radius_m > 0, got %v", mustOpName(op), g.RadiusM)
		}
		centerLat, centerLon, radius := g.CenterLat, g.CenterLon, g.RadiusM
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueGeo {
				return false
			}
			return haversineMeters(centerLat, centerLon, got.Lat, got.Lon) <= radius
		}, nil
	case FilterGeoBox:
		if err := validateLatLon(g.MinLat, g.MinLon); err != nil {
			return nil, fmt.Errorf("vector: filter op %q min corner: %w", mustOpName(op), err)
		}
		if err := validateLatLon(g.MaxLat, g.MaxLon); err != nil {
			return nil, fmt.Errorf("vector: filter op %q max corner: %w", mustOpName(op), err)
		}
		// Antimeridian-crossing boxes are a documented non-goal: require an
		// ordered SW->NE box (min <= max on both axes).
		if g.MinLat > g.MaxLat {
			return nil, fmt.Errorf("vector: filter op %q requires min_lat <= max_lat (got %v > %v)", mustOpName(op), g.MinLat, g.MaxLat)
		}
		if g.MinLon > g.MaxLon {
			return nil, fmt.Errorf("vector: filter op %q requires min_lon <= max_lon, antimeridian crossing unsupported (got %v > %v)", mustOpName(op), g.MinLon, g.MaxLon)
		}
		minLat, minLon, maxLat, maxLon := g.MinLat, g.MinLon, g.MaxLat, g.MaxLon
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueGeo {
				return false
			}
			return pointInBox(got.Lat, got.Lon, minLat, minLon, maxLat, maxLon)
		}, nil
	case FilterGeoPolygon:
		poly := g.Polygon
		if len(poly)%2 != 0 {
			return nil, fmt.Errorf("vector: filter op %q polygon must be a flat lat,lon slice (even length), got len %d", mustOpName(op), len(poly))
		}
		if len(poly) < 6 {
			return nil, fmt.Errorf("vector: filter op %q polygon needs >= 3 vertices (len >= 6), got len %d", mustOpName(op), len(poly))
		}
		for i := 0; i < len(poly); i += 2 {
			if err := validateLatLon(poly[i], poly[i+1]); err != nil {
				return nil, fmt.Errorf("vector: filter op %q polygon vertex %d: %w", mustOpName(op), i/2, err)
			}
		}
		return func(m Metadata) bool {
			got, ok := look.get(m)
			if !ok || got.Kind != ValueGeo {
				return false
			}
			return pointInPolygon(got.Lat, got.Lon, poly)
		}, nil
	default:
		return nil, fmt.Errorf("vector: compileGeo called with non-geo op %d", op)
	}
}

// validateLatLon checks a WGS84 coordinate is in range (lat [-90,90], lon
// [-180,180]). Called only at compile time.
func validateLatLon(lat, lon float64) error {
	if lat < -90 || lat > 90 {
		return fmt.Errorf("latitude %v out of range [-90,90]", lat)
	}
	if lon < -180 || lon > 180 {
		return fmt.Errorf("longitude %v out of range [-180,180]", lon)
	}
	return nil
}

// compileRowPresence builds the FilterRowExists / FilterRowAbsent predicate.
// f.Field must have the "payloadKey/path" shape (record.SplitField) with a
// path naming exactly a table row (a field segment then a row segment, e.g.
// "b/42") — that shape is validated ONCE here, at compile time, because it
// depends only on the filter's field STRING, never on any point's data; the
// parsed record.Path is cached in the closure so per-point work is Resolve
// only, per the Task 5 parsing-cost ruling (contrast with lookupPath, which
// re-parses per call because it is shared by every scalar leaf op and has no
// dedicated per-op compile step).
//
// At each point, row_exists is true iff Resolve reports Kind == RowPresent.
// Every "cannot apply" outcome — a missing payload key, a payload key that
// is not a ValueRecord, or a Resolve error (e.g. the record's schema puts a
// scalar where this filter expects a table) — already falls out of the
// Kind != RowPresent check, so row_exists needs no separate branch for it.
//
// The per-point closure tries the EXACT payload key m[f.Field] first, like
// every other leaf op (lookupPath, fieldLookup.get): a literal payload key
// that itself contains '/' wins over splitting the field. Both ops answer
// false for such a point — the field named a value, not a table. Only the
// SHAPE check above is compile-time; whether a literal key exists varies per
// point and cannot be hoisted.
//
// row_absent is DELIBERATELY NOT simply "not row_exists": it reports true
// only when Resolve AFFIRMATIVELY finds the row missing from a REAL table on
// this point (Kind == Absent) — never for a point that cannot even be asked
// the question. Concretely, row_absent answers false (not true) when the
// payload key is missing, holds something other than a ValueRecord, holds a
// record whose shape has NO SUCH FIELD at all (an unknown field name, or an
// out-of-range position), or when Resolve errors (e.g. this point's record
// has a scalar, not a table, at that field). The rationale: "the row is
// absent" presupposes there was a table to search; a point with no such
// record, or no such table, is not in a position to assert anything about a
// row inside one. This is the one asymmetry in an otherwise-mirrored pair of
// ops, and it is why row_absent cannot be compiled as `!rowExists(m)`.
func compileRowPresence(f Filter) (Predicate, error) {
	payloadKey, pathStr, ok := record.SplitField(f.Field)
	if !ok {
		return nil, fmt.Errorf("vector: filter op %q requires a payloadKey/path field, got %q", mustOpName(f.Op), f.Field)
	}
	p, err := record.ParsePath(pathStr)
	if err != nil {
		return nil, fmt.Errorf("vector: filter op %q has an invalid record path %q: %w", mustOpName(f.Op), pathStr, err)
	}
	if len(p.Segs) != 2 || p.Segs[1].Kind != record.SegRow {
		return nil, fmt.Errorf("vector: filter op %q requires a path naming exactly a table row (field/row), got %q", mustOpName(f.Op), pathStr)
	}
	exists := f.Op == FilterRowExists
	field := f.Field
	// The FIELD-ONLY prefix of the same path, cached beside it for row_absent's
	// "is there a table here at all?" question — see the closure below. Sliced
	// with a full slice expression so an append through either Path can never
	// reach the other's backing array.
	fieldPath := record.Path{Segs: p.Segs[:1:1]}
	return func(m Metadata) bool {
		if m == nil {
			return false
		}
		// EXACT PAYLOAD KEY FIRST, exactly as lookupPath and fieldLookup.get
		// resolve a field string: a literal payload key wins over splitting at
		// the first '/'. Whether the literal key exists is a per-POINT fact, so
		// this cannot be hoisted to the compile-time validation above; only the
		// path SHAPE can be.
		//
		// Both ops answer false here, and for the same reason: the field named a
		// literal payload value, not a table, so there was never a table to
		// search. row_exists is false because no row was found; row_absent is
		// false by the same asymmetry documented above — a point that cannot be
		// asked the question is not in a position to assert anything about a row.
		// Without this, a point holding both {"session": <record>} and a literal
		// "session/b/42" key answered row_exists true while lookupPath on the
		// same field returned the literal value, so two seams disagreed about one
		// field string — the divergence the exact-key-first rule exists to
		// prevent.
		if _, exact := m[field]; exact {
			return false
		}
		rv, ok := m[payloadKey]
		if !ok || rv.Kind != ValueRecord {
			return false
		}
		res, rerr := recordResolver.Resolve(rv.Rec, p)
		if rerr != nil {
			return false
		}
		if exists {
			return res.Kind == record.RowPresent
		}
		if res.Kind != record.Absent {
			return false
		}
		// Absent is ambiguous, and only for row_absent. resolveSchema answers
		// Absent for a field NAME that is not in this record's schema (and for
		// an out-of-range positional segment) BEFORE it ever looks for a row, so
		// "session/zzz/42" on a record with no zzz field reached here as Absent
		// and read as "the row is missing from the table". That contradicts the
		// contract above, and it disagreed with the neighbouring case: a SCALAR
		// at the field errors with ErrPath and correctly answers false, though
		// both are "no table at that field".
		//
		// Resolve the FIELD-ONLY prefix to disambiguate: Kind == Table means the
		// field really is a table, so the earlier Absent was a genuinely missing
		// row. This is a second Resolve, and there is no cheaper "is this field a
		// table" question in the resolver — but it runs ONLY on the row_absent
		// branch and ONLY for points that already resolved Absent, and it is the
		// cheapest kind of Resolve there is: in schema mode a field with no
		// further segment answers Table off the cached schema without reading the
		// table's bytes at all.
		fres, ferr := recordResolver.Resolve(rv.Rec, fieldPath)
		if ferr != nil {
			return false
		}
		return fres.Kind == record.Table
	}, nil
}

func mustOpName(op FilterOp) string {
	if b, err := op.MarshalText(); err == nil {
		return string(b)
	}
	return fmt.Sprintf("op(%d)", op)
}
