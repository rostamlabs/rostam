// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// FuzzDecodeKVQueryArgs extends the DecodeVectorOperateArgs-style identity to
// the kv_query args frame: DecodeKVQueryArgs must never panic on any input,
// must never mutate the bytes it was handed, and any successfully decoded
// call must re-encode to bytes that decode back to an identical value.
func FuzzDecodeKVQueryArgs(f *testing.F) {
	seed := func(a KVQueryArgs) {
		b, err := EncodeKVQueryArgs(a)
		if err == nil {
			f.Add(b)
		}
	}
	seed(KVQueryArgs{
		Index:       "by_rc",
		Filter:      sampleKVQueryFilter(),
		Limit:       500,
		Return:      KVQueryReturnRecords,
		Consistency: ConsistencyLeaderOnly,
		Cursor: []KVQueryCont{
			{Group: 1, After: []byte("k1"), More: true},
			{Group: 5, After: []byte("k5"), More: false},
		},
	})
	seed(KVQueryArgs{Limit: 10, Return: KVQueryReturnKeys, Consistency: ConsistencyAnyReplica, Scan: true})
	seed(KVQueryArgs{Index: "x", Limit: 1, Return: KVQueryReturnValues, Consistency: ConsistencyLinearizable})
	// The VALUE-ONLY filter: non-empty only in a Value field that Kind does not
	// name, which is what a JSON body setting "int" without "kind" decodes to.
	// It is the shape the encoder's presence test used to DROP
	// (TestKVQueryArgsValueOnlyFilterSurvivesRoundtrip states the rule); seeded
	// so the mutator starts adjacent to it instead of rediscovering it.
	seed(KVQueryArgs{
		Index:  "by_rc",
		Filter: vtypes.Filter{Value: vtypes.Value{Int: 42}},
		Limit:  10, Return: KVQueryReturnKeys, Consistency: ConsistencyLeaderOnly,
	})
	// A composite tree, so the mutator has the recursive JSON shapes to work
	// from and not only flat leaves — including a `not`, whose pointer child is
	// the one part of the tree that can be nil.
	seed(KVQueryArgs{
		Index: "by_rc",
		Filter: vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
			{Op: vtypes.FilterGt, Field: "rc", Value: vtypes.NewInt(5)},
			{Op: vtypes.FilterNot, Not: &vtypes.Filter{Op: vtypes.FilterEq, Field: "tag", Value: vtypes.NewString("de")}},
			{Op: vtypes.FilterIn, Field: "b#count", Value: vtypes.Value{Kind: vtypes.ValueInts, Ints: []int64{1, 2}}},
		}},
		Limit: 1000, Return: KVQueryReturnValues, Consistency: ConsistencyLinearizable,
	})
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{1})

	// A hostile filterLen (0xFFFFFFFF) on an otherwise plausible 32-byte
	// frame — the same seed TestKVQueryArgsRejects builds by hand, given to
	// the fuzzer as a starting point for mutation.
	hostile := make([]byte, 32)
	hostile[0] = kvQueryFlagFilter
	hostile[1] = 2
	hostile[2] = 'i'
	hostile[3] = 'x'
	binary.BigEndian.PutUint16(hostile[4:], 10)
	hostile[6] = KVQueryReturnKeys
	hostile[7] = ConsistencyAnyReplica
	binary.BigEndian.PutUint32(hostile[8:], 0xFFFFFFFF)
	f.Add(hostile)

	f.Fuzz(func(t *testing.T, b []byte) {
		before := append([]byte(nil), b...)
		a, err := DecodeKVQueryArgs(b)
		if !bytes.Equal(b, before) {
			t.Fatal("DecodeKVQueryArgs mutated its input")
		}
		if err != nil {
			return
		}
		b2, err := EncodeKVQueryArgs(a)
		if err != nil {
			t.Fatal("decoded args do not re-encode:", err)
		}
		a2, err := DecodeKVQueryArgs(b2)
		if err != nil {
			t.Fatal("re-encoded frame failed to decode:", err)
		}
		if !reflect.DeepEqual(a, a2) {
			t.Fatal("not stable")
		}
	})
}

// FuzzDecodeKVQueryCont covers the cursor sub-frame decodeKVQueryCursor reads
// and appendKVQueryCursor writes — shared by both the kv_query args cursor
// and the kv_query result continuation, so one fuzz target covers both call
// sites. Same identity property as FuzzDecodeKVQueryArgs: no panic, no input
// mutation, and a successful decode re-encodes/re-decodes to itself.
func FuzzDecodeKVQueryCont(f *testing.F) {
	seed := func(conts []KVQueryCont) {
		f.Add(appendKVQueryCursor(nil, conts))
	}
	seed(nil)
	seed([]KVQueryCont{{Group: 1, After: []byte("k1"), More: true}})
	seed([]KVQueryCont{
		{Group: 1, After: []byte("k1"), More: true},
		{Group: 5, After: nil, More: false},
	})
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{0, 1})

	f.Fuzz(func(t *testing.T, b []byte) {
		before := append([]byte(nil), b...)
		conts, n, err := decodeKVQueryCursor(b)
		if !bytes.Equal(b, before) {
			t.Fatal("decodeKVQueryCursor mutated its input")
		}
		if err != nil {
			return
		}
		if n > len(b) {
			t.Fatal("n exceeds input length")
		}
		b2 := appendKVQueryCursor(nil, conts)
		conts2, n2, err := decodeKVQueryCursor(b2)
		if err != nil {
			t.Fatal("re-encoded cursor failed to decode:", err)
		}
		if n2 != len(b2) {
			t.Fatal("re-encoded cursor did not fully consume its own bytes")
		}
		if !reflect.DeepEqual(conts, conts2) {
			t.Fatal("not stable")
		}
	})
}

// FuzzDecodeKVQueryCursor covers the EXPORTED standalone cursor codec — the one
// the REST surface hands the outside world as an opaque base64 blob, and the
// only member of this family a caller can hand back byte for byte.
//
// It is a separate target from FuzzDecodeKVQueryCont even though both cover the
// same block, because the exported pair adds two rules the internal one does
// not have and which are exactly where a hostile cursor would aim: the length
// is checked against KVQueryMaxCursorBytes BEFORE any decode, and the block must
// be consumed EXACTLY, so a cursor that decodes from a prefix of its input and
// leaves bytes unexamined is refused rather than silently accepted.
//
// The properties: never panic, never mutate the input, an empty input is an
// empty cursor rather than an error, and any accepted cursor re-encodes to bytes
// that decode back to an identical value.
func FuzzDecodeKVQueryCursor(f *testing.F) {
	seed := func(conts []KVQueryCont) {
		if b, err := EncodeKVQueryCursor(conts); err == nil {
			f.Add(b)
		}
	}
	seed(nil)
	seed([]KVQueryCont{{Group: 0, After: nil, More: false}})
	seed([]KVQueryCont{{Group: 1, After: []byte("u:0000001"), More: true}})
	seed([]KVQueryCont{
		{Group: 1, After: []byte("u:0000001"), More: true},
		{Group: 2, After: []byte("u:0000002"), More: true},
		{Group: 7, After: nil, More: false},
	})
	// A shared-prefix block, which is the shape the LCP factoring produces.
	seed([]KVQueryCont{
		{Group: 0, After: []byte("prefix/aaaaaaaa"), More: true},
		{Group: 1, After: []byte("prefix/bbbbbbbb"), More: true},
	})
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{0, 1})
	f.Add([]byte{0xff, 0xff, 0xff})

	f.Fuzz(func(t *testing.T, b []byte) {
		before := append([]byte(nil), b...)
		conts, err := DecodeKVQueryCursor(b)
		if !bytes.Equal(b, before) {
			t.Fatal("DecodeKVQueryCursor mutated its input")
		}
		if err != nil {
			if len(b) == 0 {
				t.Fatal("an empty cursor must decode to an empty continuation, not an error")
			}
			return
		}
		if len(b) == 0 && conts != nil {
			t.Fatal("an empty cursor must decode to a nil continuation")
		}
		b2, err := EncodeKVQueryCursor(conts)
		if err != nil {
			t.Fatal("an accepted cursor did not re-encode:", err)
		}
		conts2, err := DecodeKVQueryCursor(b2)
		if err != nil {
			t.Fatal("re-encoded cursor failed to decode:", err)
		}
		if !reflect.DeepEqual(conts, conts2) {
			t.Fatal("not stable")
		}
	})
}
