// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/rostamlabs/rostam/sdk/record"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// This file holds the ORACLE the posting index is held to: a deliberately
// naive second implementation that resolves EVERY key in the keyspace with
// record.Resolve and scans the whole thing for each selector, plus the
// generators that drive both. set.go answers the same question out of a
// map[scalarKey]keySet and never touches a key it did not post.
//
// The property under test is not "the two agree exactly" — it cannot be, and
// must not be. Postings are HINTS: Candidates is required to return a
// SUPERSET of the true matches within the definition's prefix (never a proper
// subset, which would lose rows) and a SUBSET of the live keys (never a key
// that does not exist). The re-read on the live value is what turns the
// superset back into an exact answer, one shard-level lookup at a time.

// su64 reinterprets a signed value as the uint64 bit pattern wire.Cell.U
// carries for the signed integer types (phase-1 test convention).
func su64(i int64) uint64 { return uint64(i) }

// --- the keyspace ---------------------------------------------------------

// oraclePrefixes are the three key prefixes a generated keyspace spreads its
// keys over. They are distinct in their FIRST byte so a prefix test is not
// accidentally satisfied by a neighbour, and none is a prefix of another.
var oraclePrefixes = []string{"u:", "o:", "z:"}

// keyspace is one generated store snapshot: the live keys (sorted, unique)
// and the raw value bytes each holds.
type keyspace struct {
	keys []string
	vals map[string][]byte
}

// live reports whether k is a key of this keyspace.
func (ks keyspace) live(k string) bool {
	_, ok := ks.vals[k]
	return ok
}

// valueShape enumerates what a generated key's value can be. Every one of
// these reaches Reindex in production: a record written by the schema engine,
// one written by the dynamic engine, a plain KV blob that was never a record
// at all, an empty value, and a record some bit rot (or an attacker) has
// mangled.
const (
	shapeSchema = iota
	shapeDynamic
	shapeNonRecord
	shapeEmpty
	shapeHostile
	shapeCount
)

// randomKeyspace builds a 300-key keyspace spread over the three prefixes,
// with values drawn from all five shapes.
func randomKeyspace(rng *rand.Rand) keyspace {
	ks := keyspace{vals: make(map[string][]byte, 300)}
	for i := 0; i < 300; i++ {
		k := fmt.Sprintf("%s%03d", oraclePrefixes[i%len(oraclePrefixes)], i)
		ks.vals[k] = randomValue(rng)
		ks.keys = append(ks.keys, k)
	}
	sort.Strings(ks.keys)
	return ks
}

func randomValue(rng *rand.Rand) []byte {
	switch rng.Intn(shapeCount) {
	case shapeSchema:
		return schemaRecord(rng)
	case shapeDynamic:
		return dynamicRecord(rng)
	case shapeNonRecord:
		// Never starts with mode byte 1 or 2, so it is not a record even by
		// accident.
		return []byte("plain-" + strings.Repeat("x", rng.Intn(8)))
	case shapeEmpty:
		if rng.Intn(2) == 0 {
			return nil
		}
		return []byte{}
	default:
		return hostileRecord(rng)
	}
}

// --- record builders ------------------------------------------------------
//
// Every generated record draws its indexable fields from the same small pool
// so eq and range selectors actually HIT: "rc" is a small integer, "sc" a
// short string, "fl" a float (NaN and an integer-valued float included), and
// "tb" a table whose row count is what a "#count" definition posts.

var oracleStrings = []string{"alpha", "beta", "gamma", "delta", ""}

func oracleInt(rng *rand.Rand) int64 { return int64(1 + rng.Intn(20)) }

func oracleFloat(rng *rand.Rand) float64 {
	switch rng.Intn(6) {
	case 0:
		return math.NaN()
	case 1:
		return math.Inf(1)
	case 2:
		// An integer-VALUED float: the cross-kind range case (a float 3.0 and
		// an int 3 are different scalar keys but the same number).
		return float64(1 + rng.Intn(6))
	default:
		return rng.NormFloat64() * 4
	}
}

// schemaRecord builds a schema-mode record. Half the schemas omit "sc" (so a
// definition on it resolves Absent), and one in four stores no field names at
// all (so a by-name path is an ERROR, not an Absent — a third outcome
// Reindex must swallow).
func schemaRecord(rng *rand.Rand) []byte {
	s := &wire.Schema{Version: 1, StoreNames: rng.Intn(4) != 0}
	s.Fields = append(s.Fields, wire.FieldDef{Name: "rc", Type: wire.OperateTypeI64})
	withSC := rng.Intn(2) == 0
	if withSC {
		s.Fields = append(s.Fields, wire.FieldDef{Name: "sc", Type: wire.OperateTypeBytes})
	}
	s.Fields = append(s.Fields, wire.FieldDef{Name: "fl", Type: wire.OperateTypeF64})
	s.Fields = append(s.Fields, wire.FieldDef{
		Name: "tb",
		Type: wire.OperateTypeTable,
		Table: &wire.TableDef{
			KeyType: wire.OperateTypeU32,
			Cols:    []wire.ColumnDef{{Name: "q", Type: wire.OperateTypeU16}},
		},
	})
	if err := s.Validate(); err != nil {
		panic("oracle: generated schema invalid: " + err.Error())
	}

	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s}
	rec.Fields = append(rec.Fields, wire.Field{
		Name: "rc",
		Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(oracleInt(rng))},
	})
	if withSC {
		rec.Fields = append(rec.Fields, wire.Field{
			Name: "sc",
			Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte(oracleStrings[rng.Intn(len(oracleStrings))])},
		})
	}
	rec.Fields = append(rec.Fields, wire.Field{
		Name: "fl",
		Cell: wire.Cell{Type: wire.OperateTypeF64, F: oracleFloat(rng)},
	})
	rec.Fields = append(rec.Fields, wire.Field{
		Name:  "tb",
		Cell:  wire.Cell{Type: wire.OperateTypeTable},
		Table: oracleTable(rng, 4),
	})
	enc := rec.Encode()
	if enc == nil {
		panic("oracle: generated schema record failed to encode")
	}
	return enc
}

// dynamicRecord builds a dynamic-mode record over a random subset of the same
// field pool, so a definition's path is sometimes present and sometimes not.
func dynamicRecord(rng *rand.Rand) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic}
	if rng.Intn(4) != 0 {
		rec.Fields = append(rec.Fields, wire.Field{
			Name: "rc",
			Cell: wire.Cell{Type: wire.OperateTypeIVarint, U: su64(oracleInt(rng))},
		})
	}
	if rng.Intn(2) == 0 {
		rec.Fields = append(rec.Fields, wire.Field{
			Name: "sc",
			Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte(oracleStrings[rng.Intn(len(oracleStrings))])},
		})
	}
	if rng.Intn(2) == 0 {
		rec.Fields = append(rec.Fields, wire.Field{
			Name: "fl",
			Cell: wire.Cell{Type: wire.OperateTypeF64, F: oracleFloat(rng)},
		})
	}
	if rng.Intn(2) == 0 {
		rec.Fields = append(rec.Fields, wire.Field{
			Name:  "tb",
			Cell:  wire.Cell{Type: wire.OperateTypeTable},
			Table: oracleTable(rng, 4),
		})
	}
	if len(rec.Fields) == 0 {
		rec.Fields = append(rec.Fields, wire.Field{
			Name: "other",
			Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7},
		})
	}
	enc := rec.Encode()
	if enc == nil {
		panic("oracle: generated dynamic record failed to encode")
	}
	return enc
}

func oracleTable(rng *rand.Rand, maxRows int) *wire.Table {
	t := &wire.Table{}
	n := rng.Intn(maxRows + 1)
	for i := 0; i < n; i++ {
		key := make([]byte, 4)
		key[0] = byte(i)
		t.Rows = append(t.Rows, wire.Row{
			Key:  key,
			Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU16, U: uint64(rng.Intn(1000))}}},
		})
	}
	return t
}

// hostileRecord takes a well-formed record and damages it: a truncation, a
// flipped byte, or a mode byte no engine writes. None of these may make
// Reindex panic, and none may leave a posting behind.
func hostileRecord(rng *rand.Rand) []byte {
	b := schemaRecord(rng)
	if rng.Intn(2) == 0 {
		b = dynamicRecord(rng)
	}
	out := append([]byte(nil), b...)
	switch rng.Intn(3) {
	case 0:
		if len(out) > 1 {
			out = out[:1+rng.Intn(len(out)-1)]
		}
	case 1:
		out[rng.Intn(len(out))] ^= byte(1 << uint(rng.Intn(8)))
	default:
		out[0] = byte(3 + rng.Intn(200))
	}
	return out
}

// --- the definitions ------------------------------------------------------

// oracleDefs are the definitions every random keyspace is indexed under: two
// prefix-scoped scalar definitions on the same prefix, one on another, and a
// "#count" definition over the WHOLE keyspace (an empty prefix), so the
// prefix-scoping invariant is exercised in both directions.
func oracleDefs() []Def {
	specs := []wire.KVIndexDef{
		{Name: "by-rc", KeyPrefix: []byte("u:"), PayloadPath: "rc", Kind: wire.KVIndexKindScalar, Enabled: true},
		{Name: "by-sc", KeyPrefix: []byte("u:"), PayloadPath: "sc", Kind: wire.KVIndexKindScalar, Enabled: true},
		{Name: "by-fl", KeyPrefix: []byte("o:"), PayloadPath: "fl", Kind: wire.KVIndexKindScalar, Enabled: true},
		{Name: "by-cnt", KeyPrefix: nil, PayloadPath: "tb#count", Kind: wire.KVIndexKindCount, Enabled: true},
	}
	defs := make([]Def, 0, len(specs))
	for i, s := range specs {
		d, err := DefFrom(s, uint64(i+1))
		if err != nil {
			panic("oracle: definition " + s.Name + " rejected: " + err.Error())
		}
		defs = append(defs, d)
	}
	return defs
}

// --- the brute-force index ------------------------------------------------

// bruteIndex is def name -> key -> the value that key's record resolves to at
// the definition's path. A key with no resolvable value simply has no entry.
type bruteIndex map[string]map[string]vtypes.Value

// buildBruteIndex resolves every key of ks under every def, the slow way:
// one record.Resolve per (def, key) pair, with a resolver that caches
// nothing. Nothing about the posting structure is consulted.
func buildBruteIndex(defs []Def, ks keyspace) bruteIndex {
	res := record.NewResolver(0)
	bi := make(bruteIndex, len(defs))
	for _, d := range defs {
		m := make(map[string]vtypes.Value)
		for _, k := range ks.keys {
			if !bytes.HasPrefix([]byte(k), d.Prefix) {
				continue
			}
			r, err := res.Resolve(ks.vals[k], d.Path)
			if err != nil {
				continue
			}
			v, ok := record.ResultValue(r)
			if !ok {
				continue
			}
			m[k] = v
		}
		bi[d.Name] = m
	}
	return bi
}

// bruteCandidates scans every key the definition covers and keeps the ones
// whose resolved value satisfies the selector, above the cursor. This is the
// TRUE answer the posting index must not lose a key from.
func bruteCandidates(bi bruteIndex, sel Selector, after []byte) [][]byte {
	var out [][]byte
	for k, v := range bi[sel.Def.Name] {
		if len(after) > 0 && k <= string(after) {
			continue
		}
		if bruteMatch(sel.Op, v, sel.Values) {
			out = append(out, []byte(k))
		}
	}
	sort.Slice(out, func(i, j int) bool { return bytes.Compare(out[i], out[j]) < 0 })
	return out
}

// bruteMatch is the filter semantics written out longhand, mirroring what
// vector.compileLeaf does for the six positive leaf ops: eq/in are KIND-STRICT
// (vtypes.Value.Equal compares Kind first), and the ordering family is
// cross-kind numeric with IEEE NaN handling, or lexicographic between two
// strings, and false for anything else.
func bruteMatch(op vtypes.FilterOp, got vtypes.Value, wants []vtypes.Value) bool {
	switch op {
	case vtypes.FilterEq, vtypes.FilterIn:
		for _, w := range wants {
			if got.Equal(w) {
				return true
			}
		}
		return false
	case vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte:
		if len(wants) != 1 {
			return false
		}
		w := wants[0]
		if gf, ok := bruteNumeric(got); ok {
			wf, ok := bruteNumeric(w)
			if !ok {
				return false
			}
			if math.IsNaN(gf) || math.IsNaN(wf) {
				return false
			}
			return bruteOrder(op, cmpFloat(gf, wf))
		}
		if got.Kind == vtypes.ValueString && w.Kind == vtypes.ValueString {
			return bruteOrder(op, strings.Compare(got.Str, w.Str))
		}
		return false
	default:
		return false
	}
}

func bruteNumeric(v vtypes.Value) (float64, bool) {
	switch v.Kind {
	case vtypes.ValueInt:
		return float64(v.Int), true
	case vtypes.ValueFloat:
		return v.Flt, true
	default:
		return 0, false
	}
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func bruteOrder(op vtypes.FilterOp, cmp int) bool {
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

// --- the selector generator -----------------------------------------------

var selectorOps = []vtypes.FilterOp{
	vtypes.FilterEq, vtypes.FilterIn,
	vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte,
}

// randomSelector draws a definition, an op and its values. One draw in four
// deliberately mismatches the kind the definition's values actually have (a
// string bound on an integer field, a numeric bound on a string field), which
// is the case where the true answer is EMPTY and the index must not invent
// candidates that the predicate would then have to reject.
func randomSelector(rng *rand.Rand, defs []Def) Selector {
	d := defs[rng.Intn(len(defs))]
	op := selectorOps[rng.Intn(len(selectorOps))]
	n := 1
	if op == vtypes.FilterIn {
		n = 1 + rng.Intn(4)
	}
	vals := make([]vtypes.Value, 0, n)
	for i := 0; i < n; i++ {
		vals = append(vals, randomBound(rng, d))
	}
	return Selector{Def: d, Op: op, Values: vals}
}

func randomBound(rng *rand.Rand, d Def) vtypes.Value {
	if rng.Intn(4) == 0 {
		// Deliberate kind mismatch, NaN and bool included.
		switch rng.Intn(4) {
		case 0:
			return vtypes.NewString(oracleStrings[rng.Intn(len(oracleStrings))])
		case 1:
			return vtypes.NewFloat(math.NaN())
		case 2:
			return vtypes.NewBool(rng.Intn(2) == 0)
		default:
			return vtypes.NewInt(oracleInt(rng))
		}
	}
	switch d.PathText {
	case "sc":
		return vtypes.NewString(oracleStrings[rng.Intn(len(oracleStrings))])
	case "fl":
		return vtypes.NewFloat(oracleFloat(rng))
	default: // "rc", "tb#count"
		return vtypes.NewInt(oracleInt(rng))
	}
}

// walkOf turns a keyspace into the `walk` callback shape Rebuild/Backfill
// take (cache.IterateChunked bound to a store). Keys are visited in sorted
// order so a test that stops the walk stops at a predictable place; the real
// cache walk has no order, which is exactly why nothing here may depend on it.
func walkOf(ks keyspace) Walker {
	return func(fn func(key, value []byte) bool) error {
		for _, k := range ks.keys {
			if !fn([]byte(k), ks.vals[k]) {
				return nil
			}
		}
		return nil
	}
}
