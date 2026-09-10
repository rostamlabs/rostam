// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// TestStoreClosedIsClientFacing pins shard.ErrStoreClosed onto the unredacted
// side of this edge.
//
// It is a REFUSAL, not a fault: the store is draining for close, the op never
// ran, and in a cluster the group's other replicas can serve it. Redacted to
// "internal error" the caller has nothing left to act on — and the case that
// matters is the __kv_query_shard__ leg, where a PEER's edge is this classifier:
// a coordinator draining one replica mid-fan-out could not tell the refusal from
// a real fault, so a page that merely needed another owner became a hard error
// on every multi-node cluster.
func TestStoreClosedIsClientFacing(t *testing.T) {
	if !clientFacingErr(shard.ErrStoreClosed) {
		t.Fatal("shard.ErrStoreClosed must cross the wire unredacted")
	}
	wrapped := fmt.Errorf("cluster: kv_query: shard group 3: %w", shard.ErrStoreClosed)
	if !clientFacingErr(wrapped) {
		t.Fatalf("a wrapped %v must cross the wire unredacted", shard.ErrStoreClosed)
	}
	if clientFacingErr(errors.New("open /var/lib/rostam/shard-7: no such file")) {
		t.Fatal("an unrelated fault must stay redacted")
	}
}

// TestStoreClosedTextIsTheSharedSpelling pins shard.ErrStoreClosed's message to
// ops.StoreClosedMsg.
//
// httpapi and grpcapi cannot import shard and recognise this refusal by MESSAGE
// (ops.IsStoreClosedMessage). The sentinel is declared from the shared constant
// so the two can never drift; this asserts that, so a future edit that spells
// the text inline is a failing test rather than two transports silently
// reclassifying the refusal as an internal fault.
func TestStoreClosedTextIsTheSharedSpelling(t *testing.T) {
	if shard.ErrStoreClosed.Error() != ops.StoreClosedMsg {
		t.Fatalf("shard.ErrStoreClosed = %q, want ops.StoreClosedMsg %q", shard.ErrStoreClosed, ops.StoreClosedMsg)
	}
	if !ops.IsStoreClosedMessage(shard.ErrStoreClosed.Error()) {
		t.Fatal("the matcher the other two transports use does not recognise the sentinel it was written for")
	}
}

// TestKVQueryCursorCapIsClientFacing pins the one kv_query refusal Task 7 left
// untyped. Its message is the per-group cursor arithmetic the caller acts on;
// redacted it says nothing at all.
func TestKVQueryCursorCapIsClientFacing(t *testing.T) {
	if !clientFacingErr(ops.ErrKVQueryCursorCap) {
		t.Fatal("ops.ErrKVQueryCursorCap must cross the wire unredacted")
	}
}

// TestKVQueryFamilyIsClientFacing restates the whole family at this edge, so the
// three transports' tables are compared against one list rather than drifting
// apart one sentinel at a time.
func TestKVQueryFamilyIsClientFacing(t *testing.T) {
	for _, err := range []error{
		ops.ErrKVQueryFilter,
		ops.ErrKVQueryScanRequired,
		ops.ErrKVQueryScanBudget,
		ops.ErrKVIndexUnavailable,
		ops.ErrKVQueryUnavailable,
		ops.ErrKVQueryCursorCap,
		kvindex.ErrNoSuchIndex,
		kvindex.ErrIndexBuilding,
		kvindex.ErrIndexChanged,
		kvindex.ErrCandidateBudget,
		wire.ErrKVQueryArgs,
		wire.ErrKVQueryResult,
		wire.ErrKVQueryArgsTruncated,
		shard.ErrStoreClosed,
	} {
		if !clientFacingErr(err) {
			t.Errorf("%v must cross the wire unredacted", err)
		}
	}
}
