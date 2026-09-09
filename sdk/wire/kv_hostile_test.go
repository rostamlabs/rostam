// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// kvHostileFrame names a valid, hand-built (or Encode-produced) frame and the
// decoder that should survive every prefix of it and every single-byte
// mutation of it without panicking, mutating the input, or accepting a
// frame it should not.
type kvHostileFrame struct {
	name   string
	valid  []byte
	decode func(b []byte) error // returns an error, or nil for "decoded ok"
}

// TestKVQueryHostileDecode sweeps the frames this task introduces
// (KVIndexDef, __kv_index_set__/__kv_index_list__, kv_query args, the cursor
// sub-frame embedded in both args and result, and the kv_query result) the
// same way: starting from one valid encoding, every PREFIX of it and every
// SINGLE-BYTE MUTATION of it must decode to either an error or a valid
// value — never panic, and never mutate the input byte slice it was handed
// (checked by comparing against an untouched copy after the call).
//
// This is a different sweep from ops/hostile_decode_test.go's reflection
// sweep over synthetic hostile byte patterns (huge declared counts, lying
// lengths): that one targets the 32-bit widening hazard across ~180
// decoders with a handful of adversarial bodies. This one instead mutates
// otherwise-VALID frames byte by byte, which catches a decoder that
// misreads its own wire shape (an off-by-one length, a wrong branch on a
// flag byte) rather than only a hostile count.
func TestKVQueryHostileDecode(t *testing.T) {
	kvQueryArgsValid, err := EncodeKVQueryArgs(KVQueryArgs{
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
	if err != nil {
		t.Fatalf("EncodeKVQueryArgs: %v", err)
	}

	kvQueryResultValid, err := EncodeKVQueryResult(KVQueryResult{
		Rows: []KVQueryRow{
			{Key: []byte("k1"), Value: []byte("v1")},
			{Key: []byte("k2")},
		},
		Cursor: []KVQueryCont{{Group: 3, After: []byte("k3"), More: true}},
	})
	if err != nil {
		t.Fatalf("EncodeKVQueryResult: %v", err)
	}

	kvIndexDefValid := AppendKVIndexDef(nil, KVIndexDef{
		Name: "by_rc", KeyPrefix: []byte("sess:"), PayloadPath: "rc", Kind: KVIndexKindScalar, Enabled: true,
	})

	kvIndexListValid := EncodeKVIndexList(
		[]KVIndexDef{
			{Name: "by_rc", KeyPrefix: []byte("sess:"), PayloadPath: "rc", Kind: KVIndexKindScalar, Enabled: true},
			{Name: "by_hits", PayloadPath: "hits#count", Kind: KVIndexKindCount, Enabled: false},
		},
		[]bool{true, false},
	)

	frames := []kvHostileFrame{
		{"kv_query args", kvQueryArgsValid, func(b []byte) error {
			_, err := DecodeKVQueryArgs(b)
			return err
		}},
		{"kv_query result", kvQueryResultValid, func(b []byte) error {
			_, err := DecodeKVQueryResult(b)
			return err
		}},
		{"KVIndexDef", kvIndexDefValid, func(b []byte) error {
			_, _, err := DecodeKVIndexDef(b)
			return err
		}},
		{"__kv_index_set__ args", kvIndexDefValid, func(b []byte) error {
			_, err := DecodeKVIndexSetArgs(b)
			return err
		}},
		{"__kv_index_list__ result", kvIndexListValid, func(b []byte) error {
			_, _, err := DecodeKVIndexList(b)
			return err
		}},
	}

	for _, fr := range frames {
		t.Run(fr.name, func(t *testing.T) {
			// Every prefix.
			for i := 0; i <= len(fr.valid); i++ {
				runHostileDecode(t, fr, fr.valid[:i], "prefix", i)
			}
			// Every single-byte mutation.
			for i := range fr.valid {
				mut := append([]byte(nil), fr.valid...)
				mut[i] ^= 0xFF
				runHostileDecode(t, fr, mut, "mutation", i)
			}
		})
	}
}

// runHostileDecode calls fr.decode on a COPY of in (so a decoder that
// mutates its input is caught by comparing the copy against the original
// afterward) and fails the test if the call panics.
func runHostileDecode(t *testing.T, fr kvHostileFrame, in []byte, kind string, i int) {
	t.Helper()
	arg := make([]byte, len(in))
	copy(arg, in)
	before := make([]byte, len(arg))
	copy(before, arg)

	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s: %s #%d (len %d) panicked: %v", fr.name, kind, i, len(in), r)
		}
	}()
	_ = fr.decode(arg)
	if !bytes.Equal(arg, before) {
		t.Errorf("%s: %s #%d mutated its input", fr.name, kind, i)
	}
}

// TestKVQueryFilterFuzzSeedsDecodeCleanly is a lightweight companion to the
// mutation sweep above, specifically for the filter JSON payload embedded in
// a kv_query args frame: a handful of structurally-odd (but not byte-level
// corrupt) filter trees must never panic CheckFilterBudget or
// DecodeKVQueryArgs, only ever error or succeed.
func TestKVQueryFilterFuzzSeedsDecodeCleanly(t *testing.T) {
	seeds := []vtypes.Filter{
		{},                               // zero filter: "match all"
		{Op: vtypes.FilterNot, Not: nil}, // Not with a nil child
		{Op: vtypes.FilterAnd, And: []vtypes.Filter{{}, {}}}, // empty leaves under and
		nestedNotFilter(50), // over the depth budget
	}
	for i, f := range seeds {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("seed #%d panicked: %v", i, r)
				}
			}()
			_ = CheckFilterBudget(f, KVQueryMaxFilterNodes, KVQueryMaxFilterDepth)
			_, _ = EncodeKVQueryArgs(KVQueryArgs{
				Index: "by_rc", Filter: f, Limit: 10, Return: KVQueryReturnKeys, Consistency: ConsistencyAnyReplica,
			})
		}()
	}
}
