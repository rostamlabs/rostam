// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/ops"
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

// TestKVQueryFamilyIsClientFacing walks the ONE canonical family list.
//
// This edge has only two answers — unredacted or redacted — so every class in
// the table lands in the same bucket here: all of them must cross the wire
// intact. That is not a weaker assertion than the other two transports'. It is
// the PRECONDITION for theirs: server.clientFacingErr runs at the peer's edge on
// the __kv_query_shard__ leg, and anything it redacts reaches the coordinator as
// "internal error" with nothing left for httpapi or grpcapi to classify.
//
// Sharing the list is the point. The Task 7 review found this classifier ahead
// of httpapi's by three sentinels while both files' comments claimed parity;
// adding a refusal to ops.KVQueryErrorFamily now reaches all three tests at
// once.
func TestKVQueryFamilyIsClientFacing(t *testing.T) {
	for _, spec := range ops.KVQueryErrorFamily() {
		t.Run(spec.Name, func(t *testing.T) {
			if !clientFacingErr(spec.Err) {
				t.Fatalf("%s (%s) must cross the wire unredacted", spec.Name, spec.Class)
			}
			wrapped := fmt.Errorf("cluster: kv_query: shard group 3: %w", spec.Err)
			if !clientFacingErr(wrapped) {
				t.Fatalf("wrapped %s (%s) must cross the wire unredacted", spec.Name, spec.Class)
			}
		})
	}
}

// TestKVQueryFamilyCoversTheRealStoreClosedSentinel closes the one gap the
// shared table cannot close by itself.
//
// ops cannot import shard, so the family carries shard.ErrStoreClosed as its
// MESSAGE form. This package can import shard, so it is the only place that can
// check the stand-in against the value production actually raises — by identity
// here, and through the matcher the other two transports use.
func TestKVQueryFamilyCoversTheRealStoreClosedSentinel(t *testing.T) {
	if !clientFacingErr(shard.ErrStoreClosed) {
		t.Fatal("the real shard.ErrStoreClosed must cross the wire unredacted")
	}
	if shard.ErrStoreClosed.Error() != ops.StoreClosedMsg {
		t.Fatalf("shard.ErrStoreClosed = %q, want ops.StoreClosedMsg %q", shard.ErrStoreClosed, ops.StoreClosedMsg)
	}
	if !ops.IsStoreClosedMessage(shard.ErrStoreClosed.Error()) {
		t.Fatal("the matcher the other two transports use does not recognise the sentinel it was written for")
	}
	// The family's stand-in must be indistinguishable, at this edge, from the
	// real thing — otherwise the other two transports are testing a fiction.
	var standIn error
	for _, spec := range ops.KVQueryErrorFamily() {
		if spec.Err.Error() == ops.StoreClosedMsg {
			standIn = spec.Err
		}
	}
	if standIn == nil {
		t.Fatal("ops.KVQueryErrorFamily no longer carries a store-closed entry")
	}
	if clientFacingErr(standIn) != clientFacingErr(shard.ErrStoreClosed) {
		t.Fatal("the family's store-closed stand-in classifies differently from the real sentinel")
	}
}

// TestKVQueryFamilyDoesNotLeakUnrelatedFaults is the control: widening the
// family must not have widened what escapes redaction.
func TestKVQueryFamilyDoesNotLeakUnrelatedFaults(t *testing.T) {
	for _, err := range []error{
		errors.New("open /var/lib/rostam/shard-7: no such file"),
		errors.New(ops.StoreClosedMsg + " while writing /var/lib/rostam/wal-3"),
		fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", ops.ErrKVQueryFilter),
	} {
		if clientFacingErr(err) {
			t.Errorf("%v must stay redacted", err)
		}
	}
}
