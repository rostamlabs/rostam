// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestMergeKVQueryOversizeFirstRowIsKeyOnly pins the coordinator against the
// wedge the leaf already refuses to fall into.
//
// The first row of a page is always emitted, so that paging advances. But the
// merged frame reserves one continuation per shard GROUP where the leaf that
// sized the row reserved one, so with enough groups carrying long keys the
// merge budget is the smaller of the two and a row the leaf sized to fit can
// push the merged frame past the cap. EncodeKVQueryResult would then fail the
// whole query with NO continuation, and every retry would rebuild the same
// page — so every key sorting after it becomes unreachable.
//
// The coordinator must therefore do exactly what the leaf does: emit the row
// WITHOUT its value (nil already means "omitted, fetch it with get" on this
// wire) and count it where the leaf counts its own.
func TestMergeKVQueryOversizeFirstRowIsKeyOnly(t *testing.T) {
	parts := make([]wire.KVQueryResult, 1)
	// One row whose value alone dwarfs the budget passed to the merge.
	parts[0] = wire.KVQueryResult{
		Rows: []wire.KVQueryRow{{Key: []byte("k"), Value: make([]byte, 4096)}},
	}

	before := ops.KVQueryOversizeRows()
	// A maxBytes far below the row cost: whatever the per-group overhead works
	// out to, this row cannot fit with its value.
	merged := mergeKVQuery(parts, nil, 10, 512)
	if len(merged.Rows) != 1 {
		t.Fatalf("merge emitted %d rows, want the single oversize row (a page must always advance)", len(merged.Rows))
	}
	if merged.Rows[0].Value != nil {
		t.Fatalf("the oversize row kept its %d-byte value; it must be emitted key-only so the frame encodes",
			len(merged.Rows[0].Value))
	}
	if string(merged.Rows[0].Key) != "k" {
		t.Fatalf("merge emitted key %q, want %q", merged.Rows[0].Key, "k")
	}
	if got := ops.KVQueryOversizeRows(); got != before+1 {
		t.Fatalf("the omitted value was not counted: KVQueryOversizeRows went %d -> %d", before, got)
	}
	// And the whole point: the merged page encodes.
	if _, err := wire.EncodeKVQueryResult(merged); err != nil {
		t.Fatalf("the merged page does not encode: %v", err)
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
