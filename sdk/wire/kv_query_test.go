// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"reflect"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

func sampleKVQueryFilter() vtypes.Filter {
	return vtypes.Filter{Op: vtypes.FilterGt, Field: "rc", Value: vtypes.NewInt(100)}
}

func TestKVQueryArgsRoundtrip(t *testing.T) {
	a := KVQueryArgs{
		Index:       "by_rc",
		Filter:      sampleKVQueryFilter(),
		Limit:       500,
		Return:      KVQueryReturnRecords,
		Consistency: ConsistencyLeaderOnly,
		Scan:        false,
		Cursor: []KVQueryCont{
			{Group: 1, After: []byte("k1"), More: true},
			{Group: 5, After: []byte("k5"), More: false},
		},
	}
	b, err := EncodeKVQueryArgs(a)
	if err != nil {
		t.Fatalf("EncodeKVQueryArgs: %v", err)
	}
	got, err := DecodeKVQueryArgs(b)
	if err != nil {
		t.Fatalf("DecodeKVQueryArgs: %v", err)
	}
	if !reflect.DeepEqual(got, a) {
		t.Fatalf("got %+v, want %+v", got, a)
	}

	// A zero cursor and an absent filter (a scan query) round-trip too.
	a2 := KVQueryArgs{Limit: 10, Return: KVQueryReturnKeys, Consistency: ConsistencyAnyReplica, Scan: true}
	b2, err := EncodeKVQueryArgs(a2)
	if err != nil {
		t.Fatalf("EncodeKVQueryArgs (scan): %v", err)
	}
	got2, err := DecodeKVQueryArgs(b2)
	if err != nil {
		t.Fatalf("DecodeKVQueryArgs (scan): %v", err)
	}
	if !reflect.DeepEqual(got2, a2) {
		t.Fatalf("got %+v, want %+v", got2, a2)
	}
}

func TestKVQueryArgsRejects(t *testing.T) {
	base := func() KVQueryArgs {
		return KVQueryArgs{Index: "by_rc", Limit: 10, Return: KVQueryReturnKeys, Consistency: ConsistencyAnyReplica}
	}
	cases := []struct {
		name string
		mut  func(a *KVQueryArgs)
	}{
		{"limit zero", func(a *KVQueryArgs) { a.Limit = 0 }},
		{"limit over cap", func(a *KVQueryArgs) { a.Limit = KVQueryMaxLimit + 1 }},
		{"return out of range", func(a *KVQueryArgs) { a.Return = 3 }},
		{"bounded staleness consistency", func(a *KVQueryArgs) { a.Consistency = ConsistencyBoundedStaleness }},
		{"no index, no scan", func(a *KVQueryArgs) { a.Index, a.Scan = "", false }},
		{"cursor groups not increasing", func(a *KVQueryArgs) {
			a.Cursor = []KVQueryCont{{Group: 2}, {Group: 1}}
		}},
		{"cursor over max conts", func(a *KVQueryArgs) {
			conts := make([]KVQueryCont, KVQueryMaxCursorConts+1)
			for i := range conts {
				conts[i] = KVQueryCont{Group: uint32(i) + 1} //nolint:gosec // test data, small bounded loop
			}
			a.Cursor = conts
		}},
		{"filter blob over max bytes", func(a *KVQueryArgs) {
			a.Filter = vtypes.Filter{Op: vtypes.FilterEq, Field: "f", Value: vtypes.NewString(strings.Repeat("x", KVQueryMaxFilterBytes))}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := base()
			tc.mut(&a)
			if _, err := EncodeKVQueryArgs(a); err == nil {
				t.Fatalf("%+v: expected Encode error", a)
			}
		})
	}

	// A filterLen of 0xFFFFFFFF on an otherwise plausible 32-byte frame must
	// be rejected immediately — the declared length is compared against the
	// cap in unsigned space BEFORE any int conversion or allocation, so it
	// never has a chance to size a giant make()/slice.
	t.Run("hostile filterLen no large alloc", func(t *testing.T) {
		frame := make([]byte, 32)
		frame[0] = kvQueryFlagFilter // flags: filter present
		frame[1] = 2                 // indexLen
		frame[2] = 'i'
		frame[3] = 'x'
		binary.BigEndian.PutUint16(frame[4:], 10) // limit
		frame[6] = KVQueryReturnKeys
		frame[7] = ConsistencyAnyReplica
		binary.BigEndian.PutUint32(frame[8:], 0xFFFFFFFF) // filterLen: hostile
		if _, err := DecodeKVQueryArgs(frame); err == nil {
			t.Fatal("hostile filterLen accepted")
		}
		// The bound is generous (small formatting/string-conversion allocs are
		// fine and expected) — the invariant under test is that decoding does
		// NOT allocate anything sized by the hostile 0xFFFFFFFF length itself
		// (which would be a multi-gigabyte make()/slice, not a handful of
		// small allocations).
		n := testing.AllocsPerRun(50, func() {
			_, _ = DecodeKVQueryArgs(frame)
		})
		if n > 10 {
			t.Fatalf("hostile filterLen allocated %v times; the length must be rejected before any allocation it sizes", n)
		}
	})
}

func TestKVQueryResultRoundtrip(t *testing.T) {
	r := KVQueryResult{
		Rows: []KVQueryRow{
			{Key: []byte("k1"), Value: []byte("v1")},
			{Key: []byte("k2")},                  // keys-only row: nil Value
			{Key: []byte("k3"), Value: []byte{}}, // a legitimate empty (non-nil) value
		},
		Cursor: []KVQueryCont{{Group: 2, After: []byte("k2"), More: true}},
	}
	b, err := EncodeKVQueryResult(r)
	if err != nil {
		t.Fatalf("EncodeKVQueryResult: %v", err)
	}
	got, err := DecodeKVQueryResult(b)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult: %v", err)
	}
	if !reflect.DeepEqual(got, r) {
		t.Fatalf("got %+v, want %+v", got, r)
	}

	// An empty result (no rows, no cursor) round-trips too.
	empty := KVQueryResult{}
	be, err := EncodeKVQueryResult(empty)
	if err != nil {
		t.Fatalf("EncodeKVQueryResult (empty): %v", err)
	}
	gotEmpty, err := DecodeKVQueryResult(be)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult (empty): %v", err)
	}
	if len(gotEmpty.Rows) != 0 || len(gotEmpty.Cursor) != 0 {
		t.Fatalf("got %+v, want empty", gotEmpty)
	}
}

func TestKVQueryResultRejectsOversizePage(t *testing.T) {
	// A single oversized row's value alone pushes EncodeKVQueryResult over
	// KVQueryMaxPageBytes.
	big := KVQueryResult{Rows: []KVQueryRow{{Key: []byte("k"), Value: make([]byte, KVQueryMaxPageBytes+1)}}}
	if _, err := EncodeKVQueryResult(big); err == nil {
		t.Fatal("oversize page encoded")
	}

	// A frame over the cap must fail to decode outright, before any of its
	// (possibly malformed) content is even inspected.
	oversize := make([]byte, KVQueryMaxPageBytes+1)
	if _, err := DecodeKVQueryResult(oversize); err == nil {
		t.Fatal("oversize page decoded")
	}
}

// nestedNotFilter builds a chain of depth nested FilterNot wrappers around a
// single leaf, so CheckFilterBudget's deepest walked node sits at depth
// `depth` (depth==1 is just the bare leaf).
func nestedNotFilter(depth int) vtypes.Filter {
	f := vtypes.Filter{Op: vtypes.FilterEq, Field: "f", Value: vtypes.NewInt(1)}
	for i := 1; i < depth; i++ {
		inner := f
		f = vtypes.Filter{Op: vtypes.FilterNot, Not: &inner}
	}
	return f
}

func TestCheckFilterBudget(t *testing.T) {
	children := make([]vtypes.Filter, 257)
	for i := range children {
		children[i] = vtypes.Filter{Op: vtypes.FilterEq, Field: "f", Value: vtypes.NewInt(int64(i))}
	}
	big := vtypes.Filter{Op: vtypes.FilterAnd, And: children}
	if err := CheckFilterBudget(big, KVQueryMaxFilterNodes, KVQueryMaxFilterDepth); err == nil {
		t.Fatal("257-node and chain accepted")
	}

	if err := CheckFilterBudget(nestedNotFilter(32), KVQueryMaxFilterNodes, KVQueryMaxFilterDepth); err != nil {
		t.Fatalf("32-deep not nest rejected: %v", err)
	}
	if err := CheckFilterBudget(nestedNotFilter(33), KVQueryMaxFilterNodes, KVQueryMaxFilterDepth); err == nil {
		t.Fatal("33-deep not nest accepted")
	}
}

func TestReadConsistencyOfKVQuery(t *testing.T) {
	a := KVQueryArgs{Index: "by_rc", Limit: 10, Return: KVQueryReturnKeys, Consistency: ConsistencyLinearizable}
	b, err := EncodeKVQueryArgs(a)
	if err != nil {
		t.Fatal(err)
	}
	rc, ok := ReadConsistencyOf("kv_query", b)
	if !ok || rc != ConsistencyLinearizable {
		t.Fatalf("rc=%d ok=%v, want %d true", rc, ok, ConsistencyLinearizable)
	}

	if _, ok := ReadConsistencyOf("kv_query", []byte{0xFF}); ok {
		t.Fatal("garbage args reported ok")
	}
}

func TestKVQueryRoutingRowIsKeyless(t *testing.T) {
	var row *BuiltinOp
	for i := range BuiltinOps {
		if BuiltinOps[i].Name == "kv_query" {
			row = &BuiltinOps[i]
			break
		}
	}
	if row == nil {
		t.Fatal("kv_query has no BuiltinOps row")
	}
	if row.Kind != OpReadOnly {
		t.Fatalf("kv_query Kind = %v, want OpReadOnly", row.Kind)
	}
	if row.KE != nil {
		t.Fatal("kv_query KeyExtractor is not nil (kv_query must be keyless)")
	}

	for _, name := range []string{"__kv_index_set__", "__kv_index_list__"} {
		for _, o := range BuiltinOps {
			if o.Name == name {
				t.Fatalf("%s must not have a BuiltinOps row — it is an admin op, like __set_catalog__", name)
			}
		}
	}
}
