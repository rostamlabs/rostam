// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// --- fixtures -------------------------------------------------------------

// kvQueryAll matches every record seeded by seedKVIndexRecords (their
// kvIndexField holds i%5), and is a POSITIVE leaf on the indexed path, so it can
// drive the index rather than forcing a scan.
func kvQueryAll() vtypes.Filter {
	return vtypes.Filter{Op: vtypes.FilterGte, Field: kvIndexField, Value: vtypes.NewInt(0)}
}

// kvQueryCall runs ONE page through the node's coordinator, exactly as a client
// would: encode, Call("kv_query"), decode.
func kvQueryCall(t *testing.T, n *Node, a wire.KVQueryArgs) (wire.KVQueryResult, error) {
	t.Helper()
	args, err := wire.EncodeKVQueryArgs(a)
	if err != nil {
		t.Fatalf("EncodeKVQueryArgs(%+v): %v", a, err)
	}
	raw, err := n.Call(kvQueryOpName, args)
	if err != nil {
		return wire.KVQueryResult{}, err
	}
	res, derr := wire.DecodeKVQueryResult(raw)
	if derr != nil {
		t.Fatalf("DecodeKVQueryResult: %v", derr)
	}
	return res, nil
}

// kvQueryPageAll threads the composite cursor to exhaustion and returns every
// key the query delivered, in the order the pages delivered them.
func kvQueryPageAll(t *testing.T, n *Node, a wire.KVQueryArgs, maxPages int) []string {
	t.Helper()
	var got []string
	for page := 0; ; page++ {
		if page > maxPages {
			t.Fatalf("paging did not terminate after %d pages (%d keys so far)", page, len(got))
		}
		res, err := kvQueryCall(t, n, a)
		if err != nil {
			t.Fatalf("page %d: %v", page, err)
		}
		for _, row := range res.Rows {
			got = append(got, string(row.Key))
		}
		if len(res.Cursor) == 0 {
			return got
		}
		a.Cursor = res.Cursor
	}
}

// seedKVQueryRecordsVia writes the same records seedKVIndexRecords does, but
// THROUGH THE CLUSTER CLIENT.
//
// Node.PutBatch dispatches through Node.Call, which surfaces a NotLeaderError
// for a group this node hosts as a FOLLOWER and expects the caller to follow the
// hint. That is fine on a single node and on RF=1 (where the owner always
// leads), and it is exactly what fails on RF=2 — which is the placement every
// forwarding test here needs. The client follows the hint, so the seed lands
// wherever the leader is.
func seedKVQueryRecordsVia(t *testing.T, tc *testCluster, prefix string, from, to int) {
	t.Helper()
	ctx := context.Background()
	for i := from; i < to; i++ {
		k := fmt.Appendf(nil, "%s%07d", prefix, i)
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, kvIndexRecord(t, int64(i%5)), 0)); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}
}

// kvQuerySeededKeys is what seedKVIndexRecords(prefix, from, to) wrote.
func kvQuerySeededKeys(prefix string, from, to int) []string {
	out := make([]string, 0, to-from)
	for i := from; i < to; i++ {
		out = append(out, fmt.Sprintf("%s%07d", prefix, i))
	}
	return out
}

// waitKVIndexReadyEverywhere blocks until EVERY node reports the definition
// ready on every group it hosts.
//
// Readiness is answered by ONE replica per group, but a stale (AnyReplica) query
// can be served by ANY of them — so a test that waited only for the aggregate
// bit could still race a replica that is mid-backfill and get the retryable
// ErrIndexBuilding. Waiting for every node closes that window for good.
func waitKVIndexReadyEverywhere(t *testing.T, tc *testCluster, defs int) {
	t.Helper()
	waitForKVIndex(t, 40*time.Second, "every node to report the index ready on every group it hosts", func() bool {
		for _, n := range tc.nodes {
			if n == nil {
				continue
			}
			if n.Stats().KVIndex.Ready != defs {
				return false
			}
		}
		return true
	})
}

// --- the fan-out ----------------------------------------------------------

// A kv_query answers over the WHOLE keyspace, which lives in one independent
// cache per shard group. Without the fan-out it would answer from group 0 alone
// and present that as complete.
func TestKVQueryFanOutThreeShards(t *testing.T) {
	const (
		shards = 6
		keys   = 120
	)
	tc := newTestCluster(t, 3, shards, 2) // RF=2: no node hosts every group, so legs are forwarded
	n := tc.nodes[0]

	seedKVQueryRecordsVia(t, tc, "u:", 0, keys)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	want := kvQuerySeededKeys("u:", 0, keys)
	for _, tcase := range []struct {
		name string
		rc   uint8
	}{
		{"stale", wire.ConsistencyAnyReplica},
		{"leader", wire.ConsistencyLeaderOnly},
		{"linearizable", wire.ConsistencyLinearizable},
	} {
		t.Run(tcase.name, func(t *testing.T) {
			// One page big enough for everything...
			whole, err := kvQueryCall(t, n, wire.KVQueryArgs{
				Index: "by_age", Filter: kvQueryAll(), Limit: keys, Consistency: tcase.rc,
			})
			if err != nil {
				t.Fatalf("single-page query: %v", err)
			}
			gotWhole := make([]string, 0, len(whole.Rows))
			for _, row := range whole.Rows {
				gotWhole = append(gotWhole, string(row.Key))
			}
			if fmt.Sprint(gotWhole) != fmt.Sprint(want) {
				t.Fatalf("single page returned %d keys, want %d\n got=%v\nwant=%v", len(gotWhole), len(want), gotWhole, want)
			}

			// ...and the same answer PAGED, seven rows at a time, which is where
			// the composite cursor has to be right.
			paged := kvQueryPageAll(t, n, wire.KVQueryArgs{
				Index: "by_age", Filter: kvQueryAll(), Limit: 7, Consistency: tcase.rc,
			}, 200)
			if fmt.Sprint(paged) != fmt.Sprint(want) {
				t.Fatalf("paged query returned %d keys, want %d\n got=%v\nwant=%v", len(paged), len(want), paged, want)
			}
		})
	}

	// And every node answers identically: the coordinator is whichever node the
	// client reached, not a special one.
	for i, other := range tc.nodes {
		if other == nil {
			continue
		}
		got := kvQueryPageAll(t, other, wire.KVQueryArgs{
			Index: "by_age", Filter: kvQueryAll(), Limit: 33, Consistency: wire.ConsistencyLeaderOnly,
		}, 100)
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Fatalf("node %d returned %d keys, want %d", i, len(got), len(want))
		}
	}
}

// The legs run CONCURRENTLY: a query costs the slowest group's latency, not the
// sum over groups. Sequentially this would be 6 x the delay.
func TestKVQueryFanOutIsParallel(t *testing.T) {
	const (
		shards = 6
		delay  = 200 * time.Millisecond
	)
	tc := newTestCluster(t, 1, shards)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 20)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	var legs atomic.Int32
	kvQueryLegHook = func(int) {
		legs.Add(1)
		time.Sleep(delay)
	}
	t.Cleanup(func() { kvQueryLegHook = nil })

	began := time.Now()
	if _, err := kvQueryCall(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 100}); err != nil {
		t.Fatalf("query: %v", err)
	}
	elapsed := time.Since(began)

	if got := legs.Load(); got != shards {
		t.Fatalf("%d legs ran, want one per shard group (%d)", got, shards)
	}
	if elapsed >= 2*delay {
		t.Fatalf("a %d-group fan-out with a %s delay per leg took %s; the legs are running in sequence", shards, delay, elapsed)
	}
}

// One hung group must not hang the query: each leg carries its own timeout, and
// the failure names the group.
func TestKVQueryGroupTimeout(t *testing.T) {
	tc := newTestCluster(t, 1, 4)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 20)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	prev := kvQueryGroupTimeout
	kvQueryGroupTimeout = 300 * time.Millisecond
	release := make(chan struct{})
	resumed := make(chan struct{})
	kvQueryLegHook = func(group int) {
		if group == 1 {
			<-release // never answers until the test lets it go
			close(resumed)
		}
	}
	// THE ABANDONED LEG MUST FINISH BEFORE THE HARNESS CLOSES THE NODE. A leg the
	// per-group timeout gave up on is still running — that is the documented
	// shape of forEachGroup — and here it is parked inside a store whose cache
	// tc.Close is about to unmap. Releasing it and waiting is what keeps this
	// test from segfaulting on the way out; the settle covers the microseconds
	// between the hook returning and the local read completing. (In production
	// the node stays up and the leg simply finishes; a read in flight when
	// Node.Close runs is the pre-existing Get-during-Close follow-up recorded in
	// Task 5, which this test does not widen.)
	t.Cleanup(func() {
		kvQueryGroupTimeout = prev
		close(release)
		<-resumed
		time.Sleep(500 * time.Millisecond)
		kvQueryLegHook = nil
	})

	began := time.Now()
	_, err := kvQueryCall(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 100})
	elapsed := time.Since(began)
	if err == nil {
		t.Fatal("a query with a silent group returned a page; a short page is a wrong answer")
	}
	if !strings.Contains(err.Error(), "shard group 1") {
		t.Fatalf("error does not name the hung group: %v", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error is not a deadline: %v", err)
	}
	if elapsed > 10*kvQueryGroupTimeout {
		t.Fatalf("the query took %s, far past the %s per-group bound", elapsed, kvQueryGroupTimeout)
	}
}

// A group that cannot be reached fails the QUERY. A read has no "retry completes
// it" story: a page missing one group's rows is indistinguishable from a
// complete one, so it must never be returned.
func TestKVQueryPartialFailureIsAnError(t *testing.T) {
	const shards = 6
	tc := newTestCluster(t, 3, shards, 1) // RF=1: every group has exactly one owner
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 60)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	full := kvQueryPageAll(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 100}, 20)
	if len(full) != 60 {
		t.Fatalf("the healthy query returned %d keys, want 60", len(full))
	}

	// Take the sole owner of some groups away.
	var lost []int
	for g := 0; g < shards; g++ {
		if tc.nodes[1].getShard(g) != nil {
			lost = append(lost, g)
		}
	}
	if len(lost) == 0 {
		t.Fatal("node 1 hosts no shard group; the test would prove nothing")
	}
	_ = tc.servers[1].Close()
	_ = tc.nodes[1].Close()
	tc.servers[1], tc.nodes[1] = nil, nil

	prev := kvQueryGroupTimeout
	kvQueryGroupTimeout = 2 * time.Second // the owner is gone; do not wait the full bound
	t.Cleanup(func() { kvQueryGroupTimeout = prev })

	_, err := kvQueryCall(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 100})
	if err == nil {
		t.Fatalf("a query missing the owner of groups %v returned a page instead of an error", lost)
	}
	if !strings.Contains(err.Error(), "shard group") {
		t.Fatalf("the failure does not name a shard group: %v", err)
	}
}

// --- error classification -------------------------------------------------

// Between the meta commit and the observer installing the definition, a group
// answers "no such index" — permanently, as far as IT can tell. The coordinator
// knows the name is in the catalog and rewrites it as retryable.
func TestKVQueryUninstalledIndexIsRetryable(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n) // nothing will install the definition

	seedKVIndexRecords(t, n, "u:", 0, 20)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	if _, ok := n.meta.FSM.KVIndexLookup("by_age"); !ok {
		t.Fatal("the definition is not in this node's meta catalog")
	}
	if idx := n.getShard(0).KVIndex(); idx != nil {
		if _, installed := idx.Lookup("by_age"); installed {
			t.Fatal("the observer installed the definition despite being frozen")
		}
	}

	_, err := kvQueryCall(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 10})
	if err == nil {
		t.Fatal("a query naming an uninstalled index returned a page")
	}
	if !errors.Is(err, kvindex.ErrIndexBuilding) {
		t.Fatalf("want the RETRYABLE ErrIndexBuilding for a definition the catalog holds, got %v", err)
	}
	if errors.Is(err, kvindex.ErrNoSuchIndex) {
		t.Fatalf("the permanent ErrNoSuchIndex escaped for a definition the catalog holds: %v", err)
	}

	// A name in NO catalog stays permanent: there is nothing to wait for.
	_, err = kvQueryCall(t, n, wire.KVQueryArgs{Index: "never_defined", Filter: kvQueryAll(), Limit: 10})
	if !errors.Is(err, kvindex.ErrNoSuchIndex) {
		t.Fatalf("want the permanent ErrNoSuchIndex for an unknown name, got %v", err)
	}
}

// A REMOTE group's error crossed a process boundary and arrived as text, so the
// sentinel is gone and only the message is left. The rewrite must still fire, or
// create-then-query is a hard failure on every multi-node cluster.
func TestKVQueryClassifiesStringifiedNoSuchIndex(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}

	// What a peer's reply looks like once the type has been lost.
	remote := errors.New(`cluster: __kv_query_shard__: kvindex: no such index: "by_age" on shard group 1`)
	got := n.classifyKVQueryErr("by_age", 1, remote)
	if !errors.Is(got, kvindex.ErrIndexBuilding) {
		t.Fatalf("a stringified no-such-index was not reclassified: %v", got)
	}
	// A name the catalog does not hold is left exactly as it came.
	if got := n.classifyKVQueryErr("other", 1, remote); !errors.Is(got, remote) {
		t.Fatalf("an unknown name's error was rewritten: %v", got)
	}
	// And every other leaf refusal passes through untouched.
	for _, leaf := range kvQueryLeafErrors {
		if errors.Is(leaf, kvindex.ErrNoSuchIndex) {
			continue
		}
		wrapped := fmt.Errorf("shard group 1: %w", leaf)
		if got := n.classifyKVQueryErr("by_age", 1, wrapped); !errors.Is(got, leaf) {
			t.Fatalf("%v was reclassified into %v", leaf, got)
		}
	}
}

// --- the shard-scoped wrapper --------------------------------------------

// The wrapper is a LEAF: it answers for the one group in its payload out of that
// group's store and never re-enters the fan-out. Sending kv_query itself between
// nodes would multiply the query by the shard count at every hop.
func TestKVQueryWrapperDoesNotRebroadcast(t *testing.T) {
	const shards = 4
	tc := newTestCluster(t, 1, shards)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 60)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	leafArgs, err := wire.EncodeKVQueryArgs(wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 1000})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	const group = 1
	raw, err := n.Call(opKVQueryShardName, encodeShardScopedKVQuery(group, leafArgs))
	if err != nil {
		t.Fatalf("%s: %v", opKVQueryShardName, err)
	}
	res, err := wire.DecodeKVQueryResult(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(res.Rows) == 0 {
		t.Fatal("the wrapper returned no rows at all")
	}
	for _, row := range res.Rows {
		if g := shardOf(row.Key, shards); g != group {
			t.Fatalf("the wrapper returned key %q from group %d; it fanned out instead of answering for group %d", row.Key, g, group)
		}
	}
	// Which is a strict subset of what the coordinator returns.
	whole := kvQueryPageAll(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 1000}, 10)
	if len(whole) != 60 {
		t.Fatalf("the fan-out returned %d keys, want 60", len(whole))
	}
	if len(res.Rows) >= len(whole) {
		t.Fatalf("one group returned %d of the cluster's %d keys", len(res.Rows), len(whole))
	}

	// A frame naming a group outside the shard count is refused, and one naming a
	// group this node does not host answers ErrNoShardOwner so the sender rotates
	// to the next owner rather than treating it as an empty group.
	if _, err := n.Call(opKVQueryShardName, encodeShardScopedKVQuery(shards, leafArgs)); err == nil {
		t.Fatal("an out-of-range shard index was accepted")
	}
	if _, err := n.Call(opKVQueryShardName, []byte{1, 2}); err == nil {
		t.Fatal("a truncated wrapper frame was accepted")
	}
	if err := n.RemoveShardOwner(group); err != nil {
		t.Fatalf("RemoveShardOwner: %v", err)
	}
	if _, err := n.Call(opKVQueryShardName, encodeShardScopedKVQuery(group, leafArgs)); !errors.Is(err, ErrNoShardOwner) {
		t.Fatalf("a wrapper for an unhosted group answered %v, want ErrNoShardOwner", err)
	}
}

func TestKVQueryWrapperFrameRoundTrip(t *testing.T) {
	leaf := []byte{9, 8, 7}
	group, got, err := decodeShardScopedKVQuery(encodeShardScopedKVQuery(3, leaf))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if group != 3 || string(got) != string(leaf) {
		t.Fatalf("round trip = group %d args %v, want group 3 args %v", group, got, leaf)
	}
	if _, _, err := decodeShardScopedKVQuery([]byte{0, 0, 0}); err == nil {
		t.Fatal("a 3-byte frame was accepted")
	}
	// A wrapper with NO leaf args behind the index is structurally fine here and
	// fails in the codec, not with a panic.
	if _, args, err := decodeShardScopedKVQuery(encodeShardScopedKVQuery(0, nil)); err != nil || len(args) != 0 {
		t.Fatalf("empty leaf args = %v, %v", args, err)
	}
}

// --- linearizable ---------------------------------------------------------

// Linearizable does not get its barrier from the coordinator: the consistency
// byte rides inside the leaf args, so the SERVING shard runs VerifyLeader before
// it answers. The coordinator's only job is to deliver the read to a leader.
func TestKVQueryLinearizableRunsBarrier(t *testing.T) {
	const shards = 4
	tc := newTestCluster(t, 3, shards, 2)
	n := tc.nodes[0]

	seedKVQueryRecordsVia(t, tc, "u:", 0, 40)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	// THE BARRIER COUNT, NOT THE SERVE RECORDER, IS THE ASSERTION HERE. The
	// serve recorder is global to the process and cannot tell one OpReadOnly from
	// another, and this fan-out issues plenty of others: resolving the leader of a
	// group this node does not host asks each owner for __topology__, which the
	// owner serves off ITS shard 0 — as a follower, most of the time. Counting
	// follower serves would therefore fail on unrelated traffic. Every barrier, by
	// contrast, can only be entered by a kv_query leg here, and passing one is
	// exactly what "served by the leader" means: VerifyLeader is the check.
	var barriers atomic.Int32
	shard.SetBarrierEnteredHook(func() { barriers.Add(1) })
	defer shard.SetBarrierEnteredHook(nil)

	// A control first: the SAME query at the default (stale) consistency must
	// enter no barrier at all, so the count below is about this query's
	// consistency and not about kv_query in general.
	if _, err := kvQueryCall(t, n, wire.KVQueryArgs{Index: "by_age", Filter: kvQueryAll(), Limit: 100}); err != nil {
		t.Fatalf("stale control query: %v", err)
	}
	if got := barriers.Load(); got != 0 {
		t.Fatalf("a stale kv_query entered %d readIndex barriers, want 0", got)
	}

	res, err := kvQueryCall(t, n, wire.KVQueryArgs{
		Index: "by_age", Filter: kvQueryAll(), Limit: 100, Consistency: wire.ConsistencyLinearizable,
	})
	if err != nil {
		t.Fatalf("linearizable query: %v", err)
	}
	if len(res.Rows) != 40 {
		t.Fatalf("linearizable query returned %d rows, want 40", len(res.Rows))
	}
	if got := barriers.Load(); got < shards {
		t.Fatalf("%d readIndex barriers for a %d-group linearizable kv_query: some leg was served without VerifyLeader", got, shards)
	}
}

// --- the scan walk gate ---------------------------------------------------

// A scan-mode page walks the whole keyspace and aliases the store's live mmap.
// It must go through the SAME per-group gate a backfill does, so a removal
// drains it instead of unmapping pages underneath it — and the refusal must
// reach the client as retryable, never as a short page.
func TestKVQueryScanIsGatedByTheWalkGate(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)
	seedKVIndexRecords(t, n, "u:", 0, 200)

	scan := wire.KVQueryArgs{Filter: kvQueryAll(), Limit: 100, Scan: true}

	// Control: with both gates open the scan answers.
	res, err := kvQueryCall(t, n, scan)
	if err != nil {
		t.Fatalf("scan with open gates: %v", err)
	}
	if len(res.Rows) == 0 {
		t.Fatal("the control scan matched nothing; the gate assertion below would prove nothing")
	}

	// Shut ONE group's gate — what RemoveShardOwner does before Store.Close.
	n.drainKVIndexWalks(0)
	_, err = kvQueryCall(t, n, scan)
	if err == nil {
		t.Fatal("a scan over a group whose gate is shut returned a page; a partial walk must never be presented as complete")
	}
	if !errors.Is(err, ops.ErrKVQueryUnavailable) {
		t.Fatalf("want the retryable ops.ErrKVQueryUnavailable, got %v", err)
	}
	if !strings.Contains(err.Error(), "shard group 0") {
		t.Fatalf("the failure does not name the drained group: %v", err)
	}
}

// The same guarantee with a REAL removal racing a real scan: it must neither
// take the process down nor hand back a partial page as a complete one.
func TestKVQueryScanDrainsBeforeShardRemoval(t *testing.T) {
	seeded := 100_000
	if testing.Short() {
		seeded = 20_000
	}
	const attempts = 4
	unavailable := 0
	for attempt := 0; attempt < attempts; attempt++ {
		func() {
			tc := newTestCluster(t, 1, 1)
			n := tc.nodes[0]
			freezeKVIndexObserver(t, n)
			seedKVIndexRecords(t, n, "u:", 0, seeded)

			type outcome struct {
				rows int
				err  error
			}
			done := make(chan outcome, 1)
			go func() {
				res, err := kvQueryCall(t, n, wire.KVQueryArgs{Filter: kvQueryAll(), Limit: 1000, Scan: true})
				done <- outcome{rows: len(res.Rows), err: err}
			}()

			time.Sleep(2 * time.Millisecond) // let the walk get going
			if err := n.RemoveShardOwner(0); err != nil {
				t.Fatalf("attempt %d: RemoveShardOwner: %v", attempt, err)
			}
			select {
			case o := <-done:
				switch {
				case o.err == nil:
					// The scan finished before the removal reached it. Fine —
					// the drain waited for it, which is the whole point.
				case errors.Is(o.err, ops.ErrKVQueryUnavailable):
					unavailable++
				default:
					t.Fatalf("attempt %d: scan failed with %v, want either success or ErrKVQueryUnavailable", attempt, o.err)
				}
			case <-time.After(30 * time.Second):
				t.Fatalf("attempt %d: the scan never returned", attempt)
			}
			tc.Close()
		}()
	}
	t.Logf("%d of %d attempts raced the removal and were refused as retryable", unavailable, attempts)
}

// --- projections ----------------------------------------------------------

// nil Value means "no value here"; a non-nil one means the value is present.
// The merge re-encodes rows it did not produce, so the distinction has to
// survive the coordinator, not just the leaf.
func TestKVQueryValuePresenceThroughTheCoordinator(t *testing.T) {
	tc := newTestCluster(t, 1, 4)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 40)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	keysOnly, err := kvQueryCall(t, n, wire.KVQueryArgs{
		Index: "by_age", Filter: kvQueryAll(), Limit: 100, Return: wire.KVQueryReturnKeys,
	})
	if err != nil {
		t.Fatalf("keys query: %v", err)
	}
	if len(keysOnly.Rows) != 40 {
		t.Fatalf("keys query returned %d rows, want 40", len(keysOnly.Rows))
	}
	for _, row := range keysOnly.Rows {
		if row.Value != nil {
			t.Fatalf("a keys-only row carried a value for %q: %q", row.Key, row.Value)
		}
	}

	withValues, err := kvQueryCall(t, n, wire.KVQueryArgs{
		Index: "by_age", Filter: kvQueryAll(), Limit: 100, Return: wire.KVQueryReturnRecords,
	})
	if err != nil {
		t.Fatalf("records query: %v", err)
	}
	if len(withValues.Rows) != 40 {
		t.Fatalf("records query returned %d rows, want 40", len(withValues.Rows))
	}
	for _, row := range withValues.Rows {
		if row.Value == nil {
			t.Fatalf("a records row came back with no value for %q", row.Key)
		}
		if _, derr := wire.DecodeRecord(row.Value); derr != nil {
			t.Fatalf("a records row is not a record for %q: %v", row.Key, derr)
		}
	}
}

// --- cursor hardening -----------------------------------------------------

// The cursor is attacker-controlled bytes on the way back in. A group it names
// that this cluster does not have is ignored, not a panic and not a query that
// answers from nowhere.
func TestKVQueryCursorNamingAnUnknownGroupIsIgnored(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 20)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitKVIndexReadyEverywhere(t, tc, 1)

	res, err := kvQueryCall(t, n, wire.KVQueryArgs{
		Index: "by_age", Filter: kvQueryAll(), Limit: 100,
		Cursor: []wire.KVQueryCont{{Group: 0, More: true}, {Group: 4_000_000_000, More: true}},
	})
	if err != nil {
		t.Fatalf("a cursor naming an unknown group failed the query: %v", err)
	}
	// Only group 0 was asked, so only its keys come back.
	for _, row := range res.Rows {
		if g := shardOf(row.Key, 2); g != 0 {
			t.Fatalf("key %q from group %d answered a cursor that named only group 0", row.Key, g)
		}
	}
}

// kvQueryTargets is what "stop sending exhausted groups" means in code: a first
// page asks everyone, and a later page asks exactly the groups its cursor names.
func TestKVQueryTargetsFollowTheCursor(t *testing.T) {
	all := kvQueryTargets(nil, 4)
	for g, ask := range all {
		if !ask {
			t.Fatalf("a first page did not ask group %d", g)
		}
	}
	some := kvQueryTargets([]wire.KVQueryCont{{Group: 1}, {Group: 3}}, 4)
	want := []bool{false, true, false, true}
	for g := range want {
		if some[g] != want[g] {
			t.Fatalf("targets = %v, want %v — an exhausted group must never be sent again", some, want)
		}
	}
	// Out of range is ignored rather than panicking.
	if got := kvQueryTargets([]wire.KVQueryCont{{Group: 99}}, 4); len(got) != 4 || got[0] || got[1] || got[2] || got[3] {
		t.Fatalf("targets for an out-of-range cursor = %v, want none asked", got)
	}
}

// A composite cursor that would not fit an ARGS frame fails loud here rather
// than at the client's next request, where it would read as a query that dies at
// page two for no stated reason.
func TestKVQueryCursorOverTheArgsCapIsRefused(t *testing.T) {
	if err := checkKVQueryCursorFits(nil); err != nil {
		t.Fatalf("an empty cursor was refused: %v", err)
	}
	small := []wire.KVQueryCont{{Group: 0, After: []byte("k"), More: true}}
	if err := checkKVQueryCursorFits(small); err != nil {
		t.Fatalf("a one-entry cursor was refused: %v", err)
	}
	big := make([]wire.KVQueryCont, 0, 64)
	for g := 0; g < 64; g++ {
		big = append(big, wire.KVQueryCont{Group: uint32(g), After: make([]byte, 2048), More: true}) //nolint:gosec // small loop bound
	}
	err := checkKVQueryCursorFits(big)
	if err == nil {
		t.Fatalf("a %d-byte cursor was accepted over the %d-byte args cap", 64*(7+2048)+2, wire.KVQueryMaxCursorBytes)
	}
	if !strings.Contains(err.Error(), "cursor cap") {
		t.Fatalf("the refusal does not explain itself: %v", err)
	}
}
