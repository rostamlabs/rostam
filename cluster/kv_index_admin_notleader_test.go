// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// TestSetKVIndexOnFollowerCarriesALeaderHint.
//
// __kv_index_set__ is unusual among the forwarded meta writes: it has a
// client-facing entry point. The native client's CreateKVIndex and DropKVIndex
// send this op straight to whichever node they are connected to, so a follower
// is an ordinary destination for it, and CreateKVIndex's doc promises the write
// "can be issued against any node".
//
// The meta layer refuses with hraft.ErrNotLeader, which is NOT
// shard.ErrNotLeader — so server.mapResult never answered StatusNotLeader for
// it and the client's own not-leader hop loop could not fire: the caller saw a
// flat failure with no way to learn where to go instead. This pins the
// translation. It does NOT forward: one hop, decided by the caller, is what the
// handler's doc argues for.
func TestSetKVIndexOnFollowerCarriesALeaderHint(t *testing.T) {
	c := newTestCluster(t, 3, 4)

	leader := metaLeaderNode(t, c)
	var follower *Node
	for _, n := range c.nodes {
		if n != leader && n.meta != nil {
			follower = n
			break
		}
	}
	if follower == nil {
		t.Fatal("no non-leader meta node found")
	}

	d := kvIndexDefFixture("hintcheck", "u:")
	_, err := follower.Call(opKVIndexSetName, wire.EncodeKVIndexSetArgs(d))
	if err == nil {
		t.Fatal("a follower applied an index definition; only the meta leader may")
	}
	if !errors.Is(err, shard.ErrNotLeader) {
		t.Fatalf("a follower answered %v; it must satisfy shard.ErrNotLeader so the transport maps it to StatusNotLeader and the client hops", err)
	}
	var nle *shard.NotLeaderError
	if !errors.As(err, &nle) {
		t.Fatalf("the refusal carries no NotLeaderError to read a hint out of: %v", err)
	}
	if nle.LeaderAddr == "" {
		t.Fatal("the refusal names no leader address; the client has nothing to hop to")
	}
	if nle.LeaderAddr == follower.serverAddrFor(follower.cfg.NodeID) {
		t.Fatalf("the refusal points the client back at the follower it just failed on (%s)", nle.LeaderAddr)
	}
	if want := leader.serverAddrFor(leader.cfg.NodeID); nle.LeaderAddr != want {
		t.Fatalf("the hint names %s, want the meta leader's client-facing address %s", nle.LeaderAddr, want)
	}
	if _, present := follower.meta.FSM.KVIndexes()["hintcheck"]; present {
		t.Fatal("a refused write still reached the catalog")
	}
}
