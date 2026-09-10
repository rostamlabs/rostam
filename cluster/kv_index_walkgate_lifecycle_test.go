// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/shard"
)

// TestKVIndexWalkGateBornClosedAfterNodeDrain pins the shutdown latch.
//
// Gates are created lazily, so before the latch a group that had never been
// walked had no gate at all — and drainAllKVIndexWalks, which snapshots the map
// under kvIndexWalkMu, therefore had nothing to shut for it. A first scan
// arriving in the window between that snapshot and Node.Close's unmap would
// create a fresh OPEN gate, register, and walk pages that were about to stop
// existing. The latch makes every gate created from the drain onwards born
// closed, so the snapshot does not have to be complete to be safe.
func TestKVIndexWalkGateBornClosedAfterNodeDrain(t *testing.T) {
	n := &Node{}

	// One group has a gate before the drain; group 7 deliberately does not, which
	// is the case the snapshot used to miss.
	if _, _, ok := n.beginKVIndexWalk(1); !ok {
		t.Fatal("a walk on a fresh node must be admitted")
	}
	n.kvIndexWalkGateFor(1).wg.Done()

	n.drainAllKVIndexWalks()

	if _, _, ok := n.beginKVIndexWalk(1); ok {
		t.Fatal("a walk on an already-gated group was admitted after the node drain")
	}
	if _, _, ok := n.beginKVIndexWalk(7); ok {
		t.Fatal("a walk on a group whose gate did not exist at drain time was admitted; the gate must be born closed")
	}
	// And the born-closed gate's stop channel is already signalled, so a walk
	// that somehow held one would abort at its first check rather than block.
	select {
	case <-n.kvIndexWalkGateFor(7).stop:
	default:
		t.Fatal("a gate created after the node drain has an open stop channel")
	}
}

// TestKVIndexWalkGateShutAndReArmAreOrdered pins the primitive the shard-swap
// ordering rests on: whichever of shut/re-arm happens LAST decides the gate, so
// running both inside the same n.shardMu section as the store swap makes the
// gate always agree with the shard slot.
//
// The bug this replaces: RemoveShardOwner shut the gate after releasing
// shardMu, so a concurrent AddShardOwner could install the replacement store and
// re-arm the gate first, and the trailing shut would close it again — a live
// store behind a permanently closed gate, every scan on it answering
// ErrWalkAborted and the observer unable to backfill.
func TestKVIndexWalkGateShutAndReArmAreOrdered(t *testing.T) {
	n := &Node{}

	// Remove-then-add (the order the shardMu section now guarantees when the add
	// lands second): the group ends hosted, so the gate must end open.
	n.closeKVIndexWalkGate(3)
	n.waitKVIndexWalks(3)
	n.reopenKVIndexWalks(3)
	stop, done, ok := n.beginKVIndexWalk(3)
	if !ok {
		t.Fatal("after a re-arm the replacement store's gate must admit walks")
	}
	select {
	case <-stop:
		t.Fatal("a re-armed gate handed out an already-closed stop channel")
	default:
	}
	done()

	// Add-then-remove: the group ends unhosted, so the gate must end closed.
	n.closeKVIndexWalkGate(3)
	if _, _, ok := n.beginKVIndexWalk(3); ok {
		t.Fatal("after a removal the gate must refuse walks")
	}
}

// TestKVIndexWalkRegistrationIsBoundToTheStore.
//
// The gate is per GROUP; the hazard is per STORE, and that gap is a real window.
// A removal shuts the gate inside the shardMu section that nils the slot, then
// waits, then closes the store — but a concurrent AddShardOwner re-arms the gate
// for the REPLACEMENT store, correctly, since that store must be walkable. A
// pass still holding the OLD store pointer could then register against the
// re-armed gate AFTER the removal's wait had already returned, and be walking the
// old store when Close unmapped it.
//
// n.shards is the generation that closes it: the removal nils the slot before it
// shuts the gate, both under shardMu, so any walk registering afterwards reads a
// slot that no longer names its store and is refused.
func TestKVIndexWalkRegistrationIsBoundToTheStore(t *testing.T) {
	c := newTestCluster(t, 1, 2)
	n := c.nodes[0]

	live := n.getShard(0)
	if live == nil {
		t.Fatal("fixture: the node does not host shard group 0")
	}

	// The store this node actually hosts is admitted.
	stop, done, ok := n.beginKVIndexWalkFor(0, live)
	if !ok {
		t.Fatal("a walk on the hosted store was refused")
	}
	if stop == nil {
		t.Fatal("an admitted walk got no stop channel")
	}
	done()

	// A DIFFERENT store pointer for the same group — what a pass holds after a
	// remove and re-add — is refused even though the gate is wide open.
	stale := &shard.Store{}
	if _, _, ok := n.beginKVIndexWalkFor(0, stale); ok {
		t.Fatal("a walk registered against a store this node no longer hosts for the group; a re-armed gate must not admit a stale store pointer")
	}

	// And the refusal must not leak a registration, or a later removal would wait
	// on a walk that never started. A drain that returns proves the count is zero.
	n.drainKVIndexWalks(0)
	if _, _, ok := n.beginKVIndexWalkFor(0, live); ok {
		t.Fatal("the gate admitted a walk after being drained")
	}
}
