// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// The coordinator must not wedge the query on a row too large for the merged
// frame, and it must not strip a value from a row that would have fitted.
//
// The first row of a page is always emitted, so that paging advances. But the
// merged frame reserves one continuation per shard GROUP where the leaf that
// sized the row reserved one, so with enough groups carrying long keys the
// merged page is the tighter of the two and a row the leaf sized to fit can push
// it past the cap. EncodeKVQueryResult would then fail the whole query with NO
// continuation, every retry would rebuild the same page, and every key sorting
// after that row would be unreachable — the wedge the leaf's own oversize branch
// exists to prevent.
func TestEncodeMergedKVQueryOversizeFirstRowIsKeyOnly(t *testing.T) {
	// One row whose value alone dwarfs the page cap: the merge emits it because a
	// page must always advance, and the frame then cannot encode.
	merged := wire.KVQueryResult{
		Rows: []wire.KVQueryRow{{Key: []byte("k"), Value: make([]byte, wire.KVQueryMaxPageBytes+1)}},
	}
	if _, err := wire.EncodeKVQueryResult(merged); err == nil {
		t.Fatal("fixture: the oversize page encodes, so this test proves nothing")
	}

	before := ops.KVQueryOversizeRows()
	body, err := encodeMergedKVQuery(merged)
	if err != nil {
		t.Fatalf("the coordinator failed the whole query instead of omitting the value: %v", err)
	}
	got, err := wire.DecodeKVQueryResult(body)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult: %v", err)
	}
	if len(got.Rows) != 1 || string(got.Rows[0].Key) != "k" {
		t.Fatalf("the page came back as %+v, want the single key %q", got.Rows, "k")
	}
	if got.Rows[0].Value != nil {
		t.Fatalf("the oversize row kept a %d-byte value; it must come back key-only", len(got.Rows[0].Value))
	}
	if n := ops.KVQueryOversizeRows(); n != before+1 {
		t.Fatalf("the omitted value was not counted: KVQueryOversizeRows went %d -> %d", before, n)
	}
}

// The other half, and the reason the decision is made on the ACTUAL encoded size
// rather than on mergeKVQuery's budget arithmetic: kvQueryMergeOverhead is a
// deliberate upper bound — the longest key every group OFFERED, whether or not
// that group ends up in the cursor — so deciding from it would strip values off
// small rows that encode perfectly well, contradicting what a nil Value promises
// a client on this wire. A page that fits must pass through untouched.
func TestEncodeMergedKVQueryKeepsValuesThatFit(t *testing.T) {
	merged := wire.KVQueryResult{
		Rows:   []wire.KVQueryRow{{Key: []byte("k"), Value: []byte("small")}},
		Cursor: []wire.KVQueryCont{{Group: 0, After: []byte("k"), More: true}},
	}
	before := ops.KVQueryOversizeRows()
	body, err := encodeMergedKVQuery(merged)
	if err != nil {
		t.Fatalf("encodeMergedKVQuery: %v", err)
	}
	got, err := wire.DecodeKVQueryResult(body)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult: %v", err)
	}
	if len(got.Rows) != 1 || string(got.Rows[0].Value) != "small" {
		t.Fatalf("a row that fits came back as %+v; its value must survive", got.Rows)
	}
	if n := ops.KVQueryOversizeRows(); n != before {
		t.Fatalf("a page that fits bumped the oversize counter: %d -> %d", before, n)
	}
}

// TestMergeKVQueryResumptionIsPresenceNotLength pins the coordinator's half of
// the empty-key rule. The empty key is a storable KV key, so a group's incoming
// continuation can legitimately BE the empty key — meaning "you have already
// had that row". Reading len(After)==0 as "no cursor" lets the row through
// again on every page, and the query never advances.
func TestMergeKVQueryResumptionIsPresenceNotLength(t *testing.T) {
	parts := make([]wire.KVQueryResult, 1)
	parts[0] = wire.KVQueryResult{Rows: kvMergeRows("", "a")}

	// Resuming AT the empty key: it has been returned, so only "a" is left.
	in := []wire.KVQueryCont{{Group: 0, After: []byte{}, More: true}}
	merged := mergeKVQuery(parts, in, 10, wire.KVQueryMaxPageBytes)
	got := kvMergeKeys(merged)
	if len(got) != 1 || got[0] != "a" {
		t.Fatalf("a merge resumed at the empty key returned %q, want only %q", got, "a")
	}

	// No cursor at all: the same empty key is a first-page row and must appear.
	merged = mergeKVQuery(parts, nil, 10, wire.KVQueryMaxPageBytes)
	got = kvMergeKeys(merged)
	if len(got) != 2 || got[0] != "" {
		t.Fatalf("an unresumed merge returned %q, want both keys starting with the empty one", got)
	}
}

// A merged page that fills its byte budget exactly must still ENCODE. The budget
// is maxBytes minus kvQueryMergeOverhead, so that reservation has to cover every
// byte of the frame the rows do not pay for — and the cursor block's header is
// three bytes, not two: appendKVQueryCursor writes the continuation count AND a
// prefixLen byte before the shared prefix.
//
// One unreserved byte is enough to break a real query. A keys-only page cannot be
// rescued by encodeMergedKVQuery's retry (there is no value on the first row to
// drop), so the merge would fail the whole request with no continuation, every
// retry would rebuild the same page, and every key after it would be unreachable.
//
// The fixture SEARCHES for a key length whose rows divide the budget exactly,
// asking the real kvQueryMergeOverhead for the reservation rather than restating
// it, so the test lands on the boundary whatever that reservation becomes.
func TestMergeKVQueryFillingTheBudgetStillEncodes(t *testing.T) {
	const groups = 3

	keyOf := func(g, i, keyLen int) []byte {
		k := make([]byte, keyLen)
		for j := range k {
			k[j] = 'a'
		}
		k[0] = byte('0' + g) // distinct first byte: the groups share no prefix
		k[keyLen-2] = byte(i >> 8)
		k[keyLen-1] = byte(i)
		return k
	}
	// One row per group is enough to measure the reservation: it depends only on
	// how long the keys are, not on how many there are.
	probe := func(keyLen int) ([]wire.KVQueryResult, map[uint32][]byte) {
		parts := make([]wire.KVQueryResult, groups)
		inAfter := make(map[uint32][]byte, groups)
		for g := 0; g < groups; g++ {
			k := keyOf(g, 0, keyLen)
			parts[g] = wire.KVQueryResult{
				Rows:   []wire.KVQueryRow{{Key: k}},
				Cursor: []wire.KVQueryCont{{Group: uint32(g), After: k, More: true}}, //nolint:gosec // g < groups
			}
			inAfter[uint32(g)] = k //nolint:gosec // g < groups
		}
		return parts, inAfter
	}

	keyLen, rows := 0, 0
	for n := 8000; n < 16000 && keyLen == 0; n++ {
		parts, inAfter := probe(n)
		budget := wire.KVQueryMaxPageBytes - kvQueryMergeOverhead(parts, inAfter)
		rowCost := 2 + n + 1 // kvQueryRowBytes for a keys-only row
		if budget > 0 && budget%rowCost == 0 && budget/rowCost <= wire.KVQueryMaxLimit {
			keyLen, rows = n, budget/rowCost
		}
	}
	if keyLen == 0 {
		t.Skip("no key length in the search range divides the budget exactly")
	}

	// Every group keeps rows in reserve so none is exhausted: all three stay in
	// the merged cursor, which is the shape the reservation is sized for.
	per := rows/groups + 2
	parts := make([]wire.KVQueryResult, groups)
	in := make([]wire.KVQueryCont, 0, groups)
	for g := 0; g < groups; g++ {
		rs := make([]wire.KVQueryRow, 0, per)
		for i := 0; i < per; i++ {
			rs = append(rs, wire.KVQueryRow{Key: keyOf(g, i, keyLen)})
		}
		parts[g] = wire.KVQueryResult{
			Rows:   rs,
			Cursor: []wire.KVQueryCont{{Group: uint32(g), After: rs[len(rs)-1].Key, More: true}}, //nolint:gosec // g < groups
		}
		in = append(in, wire.KVQueryCont{Group: uint32(g), After: nil, More: true}) //nolint:gosec // g < groups
	}

	merged := mergeKVQuery(parts, in, wire.KVQueryMaxLimit, wire.KVQueryMaxPageBytes)
	if len(merged.Rows) != rows {
		t.Fatalf("fixture: the merge emitted %d rows, want the %d that fill the budget exactly", len(merged.Rows), rows)
	}
	body, err := wire.EncodeKVQueryResult(merged)
	if err != nil {
		t.Fatalf("a page the merge sized to fit does not encode: %v", err)
	}
	if len(body) > wire.KVQueryMaxPageBytes {
		t.Fatalf("the merged frame is %d bytes, over the %d cap", len(body), wire.KVQueryMaxPageBytes)
	}

	// And the reservation must be an upper bound on what the frame actually spends
	// outside the rows, which is the invariant the constant exists to hold.
	rowBytes := 0
	for _, r := range merged.Rows {
		rowBytes += kvQueryRowBytes(r)
	}
	inAfter := map[uint32][]byte{}
	for _, c := range in {
		inAfter[c.Group] = c.After
	}
	if spent, reserved := len(body)-rowBytes, kvQueryMergeOverhead(parts, inAfter); spent > reserved {
		t.Fatalf("the frame spent %d bytes outside its rows but only %d were reserved", spent, reserved)
	}
}
