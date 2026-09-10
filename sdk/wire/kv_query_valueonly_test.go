// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// TestKVQueryArgsValueOnlyFilterSurvivesRoundtrip pins the bug
// FuzzDecodeKVQueryArgs found and EncodeKVQueryArgs now guards against: a
// filter that is non-empty ONLY in its Value's non-Kind fields used to be
// classified as absent and silently dropped on encode.
//
// THE SHAPE, and why it is reachable rather than exotic. vtypes.Filter.IsZero
// delegates the Value half to vtypes.Value.IsZero, which inspects Kind and
// NOTHING else — so a Value carrying Int, Str, Strs or Rec with Kind left at
// its zero (ValueNone) reads as "no value". A caller does not have to
// construct that by hand: json.Unmarshal fills each field independently and
// gates none of them on "kind", so a JSON filter body whose "value" object
// sets "int" without a matching "kind" decodes to exactly this Filter. Under
// the old IsZero() test the encoder wrote no filter block at all, the leaf
// evaluated the query as match-all, and the caller got every key under the
// index's prefix back instead of an error or a narrowed page.
//
// The fix is a full structural comparison against the literal zero Filter,
// which has no such blind spot. This test is the explicit statement of it;
// until now the property was covered only by the fuzz corpus, which is not
// where a future refactor of the presence test would look.
func TestKVQueryArgsValueOnlyFilterSurvivesRoundtrip(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value vtypes.Value
	}{
		// Kind is deliberately LEFT ZERO in every case: that is the whole
		// point. A Value with a real Kind was never at risk.
		{"int only", vtypes.Value{Int: 42}},
		{"string only", vtypes.Value{Str: "de"}},
		{"float only", vtypes.Value{Flt: 1.5}},
		{"bool only", vtypes.Value{Bool: true}},
		{"list only", vtypes.Value{Ints: []int64{1, 2, 3}}},
		{"record only", vtypes.Value{Rec: []byte{0x01, 0x02}}},
		{"geo coords only", vtypes.Value{Lat: 1, Lon: 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			filt := vtypes.Filter{Value: tc.value}
			// The precondition that makes this a regression test rather than a
			// tautology: the helper the encoder USED to consult still calls
			// this filter empty. If that ever stops being true the bug class is
			// gone, and this test should be re-read rather than deleted.
			if !filt.IsZero() {
				t.Skip("Filter.IsZero no longer treats a value-only filter as empty; the guarded case has changed")
			}

			a := KVQueryArgs{
				Index:       "by_rc",
				Filter:      filt,
				Limit:       10,
				Return:      KVQueryReturnKeys,
				Consistency: ConsistencyLeaderOnly,
			}
			b, err := EncodeKVQueryArgs(a)
			if err != nil {
				t.Fatalf("EncodeKVQueryArgs: %v", err)
			}
			got, err := DecodeKVQueryArgs(b)
			if err != nil {
				t.Fatalf("DecodeKVQueryArgs: %v", err)
			}
			if got.Filter.IsZero() && !reflect.DeepEqual(got.Filter, filt) {
				t.Fatalf("the filter was dropped on encode: got %+v, want %+v", got.Filter, filt)
			}
			if !reflect.DeepEqual(got.Filter, filt) {
				t.Fatalf("filter did not survive the round trip:\n got %+v\nwant %+v", got.Filter, filt)
			}
			if !reflect.DeepEqual(got, a) {
				t.Fatalf("args did not survive the round trip:\n got %+v\nwant %+v", got, a)
			}
		})
	}
}
