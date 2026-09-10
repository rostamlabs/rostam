// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// TestKVIndexOpNamesMatchTheClient pins ONE NAME PER OP end to end.
//
// The cluster constants are unexported and the client is a separate module, so
// neither side can reference the other's spelling: the client repeats the
// literals and this test is the only place the two are compared. Without it a
// rename on either side compiles, ships, and turns every catalog call into
// "op not registered" at runtime.
func TestKVIndexOpNamesMatchTheClient(t *testing.T) {
	for _, tc := range []struct {
		what           string
		cluster, typed string
	}{
		{"index set", opKVIndexSetName, client.OpKVIndexSet},
		{"index list", opKVIndexListName, client.OpKVIndexList},
		{"kv query", kvQueryOpName, client.OpKVQuery},
	} {
		if tc.cluster != tc.typed {
			t.Errorf("%s: cluster says %q, the client says %q", tc.what, tc.cluster, tc.typed)
		}
	}
}

// TestKVQueryClientErrorTextsMatchTheSentinels pins the message prefixes
// client.mapKVQueryErr anchors on against the sentinels that actually produce
// them.
//
// The client re-types these refusals from TEXT — it cannot import ops or
// kvindex, so errors.Is is unavailable across the transport — and the anchor is
// the sentinel's Error() string. A reworded sentinel would silently demote every
// retryable "still building" to an unclassified error the caller stops retrying,
// which is exactly the create-then-query case. Comparing the PREFIXES here (not
// the whole message: the leaf appends the index name and shard group) makes that
// rewording a failing test instead.
func TestKVQueryClientErrorTextsMatchTheSentinels(t *testing.T) {
	// Each server-side error, rendered as the client will see it, must map to
	// the client sentinel this table names.
	for _, tc := range []struct {
		msg  string
		want error
	}{
		{kvindex.ErrNoSuchIndex.Error(), client.ErrKVIndexNotFound},
		{kvindex.ErrIndexBuilding.Error(), client.ErrKVIndexBuilding},
		{kvindex.ErrIndexChanged.Error(), client.ErrKVIndexBuilding},
		{kvindex.ErrCandidateBudget.Error(), client.ErrKVQueryFilter},
		{ops.ErrKVQueryFilter.Error(), client.ErrKVQueryFilter},
		{ops.ErrKVQueryScanRequired.Error(), client.ErrKVQueryFilter},
		{ops.ErrKVQueryScanBudget.Error(), client.ErrKVQueryFilter},
		{ops.ErrKVIndexUnavailable.Error(), client.ErrKVQueryFilter},
		{ops.ErrKVQueryUnavailable.Error(), client.ErrKVQueryUnavailable},
		{shard.ErrStoreClosed.Error(), client.ErrKVQueryUnavailable},
	} {
		if got := client.ClassifyKVQueryMessage(tc.msg); got != tc.want {
			t.Errorf("%q classified as %v, want %v", tc.msg, got, tc.want)
		}
	}
}

// TestKVIndexDefaultLimitFitsTheWireCap pins the client's default page size
// inside the codec's ceiling, so the zero-value call the docs advertise can
// never be the one request EncodeKVQueryArgs refuses.
func TestKVIndexDefaultLimitFitsTheWireCap(t *testing.T) {
	if client.KVQueryDefaultLimit == 0 || client.KVQueryDefaultLimit > wire.KVQueryMaxLimit {
		t.Fatalf("default limit %d is outside 1..%d", client.KVQueryDefaultLimit, wire.KVQueryMaxLimit)
	}
}
