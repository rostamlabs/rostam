// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"
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
