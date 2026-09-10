// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// FuzzKVQueryFilter drives the two halves of a kv_query's filter path with
// arbitrary bytes: the JSON a caller sends, and the record bytes a candidate's
// live value turns out to hold.
//
// WHY BOTH INPUTS. These are the two attacker-controlled surfaces the leaf
// puts together, and each is hostile in its own way. The filter JSON reaches
// BuildKVPredicate from the REST body and the args frame alike, having passed
// only a byte cap; the record bytes reach the compiled predicate straight out
// of the cache, where a value written by any other path — including one that
// is not a record at all — can be sitting under a key the postings named. A
// panic in either one is a crash on a client-driven read.
//
// The properties:
//
//   - BuildKVPredicate never panics on any filter within the byte cap, and
//     never mutates the tree it was handed. The no-mutation half is not
//     tidiness: SelectorFor reads the SAME filter afterwards and matches its
//     leaves against the definition's bare path, so a rewrite in place would
//     leave every field prefixed with the alias and no index could drive a
//     query again.
//   - A compiled predicate never panics on arbitrary record bytes, and answers
//     the same way twice for the same input (the metadata map is REUSED across
//     candidates inside one page, so a predicate that retained it would show up
//     as an unstable answer here).
func FuzzKVQueryFilter(f *testing.F) {
	// The doc page's example filters (docs/kv/querying-records.md), which are
	// the shapes a reader is most likely to send: an indexable leaf, a range,
	// a top-level `and` mixing an indexed leaf with a live one, a #count path,
	// and the negation/absence family that never narrows.
	for _, s := range []string{
		`{"op":"eq","field":"rc","value":{"kind":"int","int":7}}`,
		`{"op":"gt","field":"rc","value":{"kind":"int","int":5}}`,
		`{"op":"in","field":"tag","value":{"kind":"strs","strs":["de","fr"]}}`,
		`{"op":"and","and":[
			{"op":"gt","field":"rc","value":{"kind":"int","int":5}},
			{"op":"gte","field":"b#count","value":{"kind":"int","int":1}}
		]}`,
		`{"op":"eq","field":"b/42/hi","value":{"kind":"int","int":500}}`,
		`{"op":"not","not":{"op":"eq","field":"rc","value":{"kind":"int","int":7}}}`,
		`{"op":"or","or":[
			{"op":"eq","field":"rc","value":{"kind":"int","int":1}},
			{"op":"is_null","field":"tag"}
		]}`,
		`{"op":"row_exists","field":"b/42"}`,
		`{"op":"regex","field":"tag","value":{"kind":"str","str":"^d"}}`,
		// The rejections, seeded so the fuzzer starts adjacent to them: the
		// reserved namespace, an unparseable path, and an empty field.
		`{"op":"eq","field":"$rec","value":{"kind":"int","int":1}}`,
		`{"op":"eq","field":"rc/","value":{"kind":"int","int":1}}`,
		`{"op":"eq","field":"","value":{"kind":"int","int":1}}`,
		`{}`,
		`{"op":"and"}`,
	} {
		f.Add([]byte(s), fuzzKVFilterRecord())
	}
	// Record bytes that are NOT a record, against a filter that is: the shape a
	// posting for a key whose value was overwritten by a plain put produces.
	f.Add([]byte(`{"op":"eq","field":"rc","value":{"kind":"int","int":7}}`), []byte("not a record"))
	f.Add([]byte(`{"op":"eq","field":"rc","value":{"kind":"int","int":7}}`), []byte(nil))
	f.Add([]byte(`{"op":"eq","field":"rc","value":{"kind":"int","int":7}}`), []byte{0xff})
	f.Add([]byte(""), []byte(""))

	f.Fuzz(func(t *testing.T, filterJSON, rec []byte) {
		// The cap the args decoder and the REST body enforce before this
		// function is ever reached. Fuzzing above it would be testing a path
		// production cannot take.
		if len(filterJSON) > wire.KVQueryMaxFilterBytes {
			return
		}
		var filt vtypes.Filter
		if err := json.Unmarshal(filterJSON, &filt); err != nil {
			return
		}
		before := deepCopyKVFilter(filt)

		pred, err := BuildKVPredicate(filt)
		if !reflect.DeepEqual(filt, before) {
			t.Fatalf("BuildKVPredicate mutated its input filter:\n got %+v\nwant %+v", filt, before)
		}
		if err != nil {
			if pred != nil {
				t.Fatal("BuildKVPredicate returned both a predicate and an error")
			}
			return
		}
		if pred == nil {
			return // the zero filter compiles to "match all"; nothing to evaluate
		}

		// Exactly the map the leaf builds: one entry, the reserved alias, the
		// live value presented as a record.
		meta := vector.Metadata{KVRecordAlias: vtypes.Value{Kind: vtypes.ValueRecord, Rec: rec}}
		got := pred(meta)
		if again := pred(meta); again != got {
			t.Fatalf("predicate answered %v then %v for the same record", got, again)
		}
		// The same map object reused for a second candidate, which is what
		// verifyPage does for every key in a page.
		meta[KVRecordAlias] = vtypes.Value{Kind: vtypes.ValueRecord, Rec: fuzzKVFilterRecord()}
		_ = pred(meta)
		// And a candidate whose value is absent altogether.
		meta[KVRecordAlias] = vtypes.Value{}
		_ = pred(meta)
	})
}

// fuzzKVFilterRecord is a small dynamic-mode record carrying one of each
// scalar kind a record path can resolve to, plus a two-row table — enough for
// every seed filter above to have something real to resolve against.
func fuzzKVFilterRecord() []byte {
	tbl := &wire.Table{Rows: []wire.Row{
		{Key: []byte{42, 0, 0, 0, 0, 0, 0, 0}, Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}}}},
		{Key: []byte{99, 0, 0, 0, 0, 0, 0, 0}, Cols: []wire.Col{{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}}}},
	}}
	r := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "b", Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: tbl},
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: 7}},
		{Name: "tag", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
	}}
	b := r.Encode()
	if b == nil {
		panic("fuzz fixture record failed to encode")
	}
	return b
}

// deepCopyKVFilter copies f deeply, so the mutation check above compares
// against a snapshot that shares no slice or pointer with the value handed to
// BuildKVPredicate. A shallow copy would miss exactly the mutation that
// matters — a rewrite through And/Or/Not, which is where the aliasing rewrite
// recurses.
func deepCopyKVFilter(f vtypes.Filter) vtypes.Filter {
	out := f
	if f.And != nil {
		out.And = make([]vtypes.Filter, len(f.And))
		for i, c := range f.And {
			out.And[i] = deepCopyKVFilter(c)
		}
	}
	if f.Or != nil {
		out.Or = make([]vtypes.Filter, len(f.Or))
		for i, c := range f.Or {
			out.Or[i] = deepCopyKVFilter(c)
		}
	}
	if f.Not != nil {
		inner := deepCopyKVFilter(*f.Not)
		out.Not = &inner
	}
	if f.Geo != nil {
		g := *f.Geo
		if f.Geo.Polygon != nil {
			g.Polygon = append([]float64(nil), f.Geo.Polygon...)
		}
		out.Geo = &g
	}
	if f.Value.Strs != nil {
		out.Value.Strs = append([]string(nil), f.Value.Strs...)
	}
	if f.Value.Ints != nil {
		out.Value.Ints = append([]int64(nil), f.Value.Ints...)
	}
	if f.Value.Flts != nil {
		out.Value.Flts = append([]float64(nil), f.Value.Flts...)
	}
	if f.Value.Rec != nil {
		out.Value.Rec = append([]byte(nil), f.Value.Rec...)
	}
	return out
}
