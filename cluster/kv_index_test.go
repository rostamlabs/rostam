// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// --- fixtures -------------------------------------------------------------

// kvIndexField is the record field every fixture in this file indexes.
const kvIndexField = "age"

// kvIndexDefFixture builds a valid scalar definition over kvIndexField.
func kvIndexDefFixture(name, prefix string) wire.KVIndexDef {
	return wire.KVIndexDef{
		Name:        name,
		KeyPrefix:   []byte(prefix),
		PayloadPath: kvIndexField,
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}
}

// kvIndexRecord encodes a one-field dynamic-mode record holding v at
// kvIndexField. Dynamic mode keeps the fixture self-describing: no schema has
// to travel with the value.
func kvIndexRecord(t *testing.T, v int64) []byte {
	t.Helper()
	rec := &wire.Record{
		Mode:   wire.OperateModeDynamic,
		Fields: []wire.Field{{Name: kvIndexField, Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}},
	}
	b := rec.Encode()
	if b == nil {
		t.Fatal("record fixture failed to encode")
	}
	return b
}

// applyKVIndexEntry commits one definition straight into a bare MetaFSM at the
// given log index, returning whatever Apply returned.
func applyKVIndexEntry(t *testing.T, f *MetaFSM, d wire.KVIndexDef, index uint64) any {
	t.Helper()
	data, err := encodeLogEntry(LogEntry{Op: OpSetKVIndex, KVIndex: d})
	if err != nil {
		t.Fatalf("encodeLogEntry: %v", err)
	}
	return f.Apply(&raft.Log{Data: data, Index: index})
}

// waitForKVIndex polls cond until it holds or d elapses.
func waitForKVIndex(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", d, what)
}

// seedKVIndexRecords writes keys prefix+%07d for i in [from,to) as records
// carrying i%5 at kvIndexField. Node.PutBatch groups by shard before encoding,
// so one call covers a multi-group cluster in one Raft entry per group.
func seedKVIndexRecords(t *testing.T, n *Node, prefix string, from, to int) {
	t.Helper()
	const chunk = 1000
	entries := make([]ops.PutEntry, 0, chunk)
	flush := func() {
		if len(entries) == 0 {
			return
		}
		if err := n.PutBatch(entries); err != nil {
			t.Fatalf("PutBatch: %v", err)
		}
		entries = entries[:0]
	}
	for i := from; i < to; i++ {
		entries = append(entries, ops.PutEntry{
			Key: fmt.Appendf(nil, "%s%07d", prefix, i),
			Val: kvIndexRecord(t, int64(i%5)),
		})
		if len(entries) == chunk {
			flush()
		}
	}
	flush()
}

// freezeKVIndexObserver stops the polling goroutine and then clears the stop
// channel, so a test can drive passes BY HAND at moments it chooses.
//
// Clearing it is not incidental: a pass checks the stop signal between shards
// and between definitions, so a hand-driven pass on a merely-stopped node
// returns having done nothing — which would make several tests below pass for
// entirely the wrong reason.
func freezeKVIndexObserver(t *testing.T, n *Node) {
	t.Helper()
	n.stopKVIndexObserver()
	n.kvIndexStop = nil
}

// --- meta FSM -------------------------------------------------------------

func TestMetaApplySetKVIndex(t *testing.T) {
	f := NewMetaFSM()

	d := kvIndexDefFixture("by_age", "u:")
	if got := applyKVIndexEntry(t, f, d, 7); got != nil {
		t.Fatalf("Apply(enable): %v", got)
	}
	cat := f.KVIndexes()
	e, ok := cat["by_age"]
	if !ok {
		t.Fatalf("definition not in the catalog: %+v", cat)
	}
	if e.MetaIndex != 7 {
		t.Errorf("MetaIndex = %d, want 7", e.MetaIndex)
	}
	if !bytes.Equal(e.Def.KeyPrefix, []byte("u:")) || e.Def.PayloadPath != kvIndexField {
		t.Errorf("stored def = %+v", e.Def)
	}
	// The accessor must hand out a private copy: mutating it cannot reach FSM state.
	cat["by_age"] = KVIndexEntry{}
	delete(cat, "by_age")
	if _, still := f.KVIndexes()["by_age"]; !still {
		t.Fatal("KVIndexes() aliases FSM state")
	}
	// So must State(), which Task 7's coordinator reads.
	st := f.State()
	if _, inState := st.KVIndexes["by_age"]; !inState {
		t.Fatal("State().KVIndexes is missing the definition")
	}
	st.KVIndexes["by_age"].Def.KeyPrefix[0] = 'X'
	if got := f.KVIndexes()["by_age"].Def.KeyPrefix; !bytes.Equal(got, []byte("u:")) {
		t.Fatalf("State() aliases the FSM's KeyPrefix: now %q", got)
	}

	// Disable drops the entry: the map stays sparse.
	off := d
	off.Enabled = false
	if got := applyKVIndexEntry(t, f, off, 8); got != nil {
		t.Fatalf("Apply(disable): %v", got)
	}
	if len(f.KVIndexes()) != 0 {
		t.Fatalf("disable did not drop the entry: %+v", f.KVIndexes())
	}

	// The 65th distinct definition is refused at the cap.
	for i := 0; i < wire.KVIndexMaxDefs; i++ {
		if got := applyKVIndexEntry(t, f, kvIndexDefFixture(fmt.Sprintf("idx%02d", i), "u:"), uint64(100+i)); got != nil {
			t.Fatalf("Apply(%d): %v", i, got)
		}
	}
	if n := len(f.KVIndexes()); n != wire.KVIndexMaxDefs {
		t.Fatalf("catalog has %d definitions, want %d", n, wire.KVIndexMaxDefs)
	}
	over := applyKVIndexEntry(t, f, kvIndexDefFixture("one_too_many", "u:"), 200)
	if _, isErr := over.(error); !isErr {
		t.Fatalf("the 65th definition was accepted: %v", over)
	}
	if n := len(f.KVIndexes()); n != wire.KVIndexMaxDefs {
		t.Fatalf("a refused definition mutated the catalog: %d entries", n)
	}
	// An existing name may still be UPDATED at the cap (it takes no new slot).
	if got := applyKVIndexEntry(t, f, kvIndexDefFixture("idx00", "v:"), 201); got != nil {
		t.Fatalf("update at the cap: %v", got)
	}
	if got := f.KVIndexes()["idx00"].Def.KeyPrefix; !bytes.Equal(got, []byte("v:")) {
		t.Fatalf("update at the cap did not take: prefix %q", got)
	}

	// An invalid definition is refused and mutates nothing.
	bad := kvIndexDefFixture("bad name", "u:")
	if got := applyKVIndexEntry(t, f, bad, 300); got == nil {
		t.Fatal("an invalid definition was accepted")
	}
	if _, present := f.KVIndexes()["bad name"]; present {
		t.Fatal("an invalid definition entered the catalog")
	}
	if n := len(f.KVIndexes()); n != wire.KVIndexMaxDefs {
		t.Fatalf("a refused definition mutated the catalog: %d entries", n)
	}
}

func TestMetaSnapshotRestoreKeepsKVIndexes(t *testing.T) {
	src := NewMetaFSM()
	if got := applyKVIndexEntry(t, src, kvIndexDefFixture("by_age", "u:"), 11); got != nil {
		t.Fatalf("Apply: %v", got)
	}
	if got := applyKVIndexEntry(t, src, kvIndexDefFixture("by_rows", ""), 12); got != nil {
		t.Fatalf("Apply: %v", got)
	}

	snap, err := src.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := snap.Persist(noopSink{w: &buf}); err != nil {
		t.Fatal(err)
	}
	dst := NewMetaFSM()
	if err := dst.Restore(io.NopCloser(&buf)); err != nil {
		t.Fatal(err)
	}

	got := dst.KVIndexes()
	if len(got) != 2 {
		t.Fatalf("restored catalog has %d definitions, want 2: %+v", len(got), got)
	}
	if e := got["by_age"]; e.MetaIndex != 11 || !bytes.Equal(e.Def.KeyPrefix, []byte("u:")) {
		t.Errorf("by_age restored as %+v", e)
	}
	if !equalState(src.State(), dst.State()) {
		t.Error("state mismatch after restore")
	}
}

// --- the admin ops --------------------------------------------------------

func TestSetKVIndexForwardsToMetaLeader(t *testing.T) {
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

	d := kvIndexDefFixture("by_age", "u:")
	// The public entry point at a FOLLOWER: it forwards __kv_index_set__ to the
	// meta leader, whose handler proposes locally.
	if err := follower.SetKVIndex(d, 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex at a follower: %v", err)
	}
	// Read-your-writes: the call returned only after THIS node's own FSM applied
	// it, so no polling is allowed here — the assertion is immediate.
	if _, ok := follower.meta.FSM.KVIndexes()["by_age"]; !ok {
		t.Fatal("the follower returned before its own FSM reflected the write")
	}
	// And the leader committed it.
	waitForKVIndex(t, 5*time.Second, "the meta leader to apply the definition", func() bool {
		_, ok := leader.meta.FSM.KVIndexes()["by_age"]
		return ok
	})

	// A disable through the same path drops it everywhere.
	off := d
	off.Enabled = false
	if err := follower.SetKVIndex(off, 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex disable: %v", err)
	}
	if _, ok := follower.meta.FSM.KVIndexes()["by_age"]; ok {
		t.Fatal("the follower returned before its own FSM reflected the disable")
	}

	// A malformed definition is refused at the edge, not committed.
	if _, err := leader.Call(opKVIndexSetName, []byte{0xff}); err == nil {
		t.Fatal("a malformed __kv_index_set__ frame was accepted")
	}
}

// The forwarded op must be a LEAF on the node that receives it: it proposes
// locally or it fails. A handler that re-entered Node.SetKVIndex would forward
// again, and under a leadership flap that chain can loop between nodes, each hop
// holding a goroutine on a 5s call. A mis-addressed forward has to fail fast
// instead, so the caller re-resolves the leader.
func TestSetKVIndexHandlerDoesNotForward(t *testing.T) {
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

	d := kvIndexDefFixture("leafcheck", "u:")
	_, err := follower.Call(opKVIndexSetName, wire.EncodeKVIndexSetArgs(d))
	if err == nil {
		t.Fatal("the follower's handler committed the write — it forwarded instead of proposing locally")
	}
	if !errors.Is(err, raft.ErrNotLeader) {
		t.Fatalf("follower handler error = %v, want a wrap of raft.ErrNotLeader", err)
	}
	if _, present := follower.meta.FSM.KVIndexes()["leafcheck"]; present {
		t.Fatal("a refused forward still reached the catalog")
	}

	// The same op AT the leader proposes and commits.
	if _, err := leader.Call(opKVIndexSetName, wire.EncodeKVIndexSetArgs(d)); err != nil {
		t.Fatalf("%s at the meta leader: %v", opKVIndexSetName, err)
	}
	if _, ok := leader.meta.FSM.KVIndexLookup("leafcheck"); !ok {
		t.Fatal("the leader's handler returned before its own FSM applied the entry")
	}
}

// --- the observer ---------------------------------------------------------

func TestKVIndexObserverInstallsAndBackfills(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]

	// 60 pre-written record keys under the prefix, plus one outside it.
	seedKVIndexRecords(t, n, "u:", 0, 60)
	if _, err := n.Call("put", ops.EncodePutArgs([]byte("x:0000000"), kvIndexRecord(t, 1), 0)); err != nil {
		t.Fatalf("put out-of-prefix key: %v", err)
	}

	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}

	s := n.getShard(0)
	if s == nil {
		t.Fatal("node does not host shard 0")
	}
	idx := s.KVIndex()
	waitForKVIndex(t, 20*time.Second, "the observer to install and backfill by_age", func() bool {
		return idx.IsReady("by_age")
	})

	keys, distinct := idx.Stats("by_age")
	if keys != 60 {
		t.Errorf("posted keys = %d, want 60 (the out-of-prefix key must not be posted)", keys)
	}
	if distinct != 5 {
		t.Errorf("distinct values = %d, want 5", distinct)
	}
	d, ok := idx.Lookup("by_age")
	if !ok {
		t.Fatal("by_age is not installed")
	}
	for v := 0; v < 5; v++ {
		sel := kvindex.Selector{Def: d, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(int64(v))}}
		cands, err := idx.Candidates(sel, nil, 1<<20)
		if err != nil {
			t.Fatalf("Candidates(%d): %v", v, err)
		}
		if len(cands) != 12 {
			t.Errorf("Candidates(%d) returned %d keys, want 12", v, len(cands))
		}
	}
}

func TestKVIndexEnableThenBackfillLosesNoWrite(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	// Freeze the poller so activation happens exactly where this test says it does.
	freezeKVIndexObserver(t, n)

	const (
		seeded    = 20_000
		writers   = 8
		minWrites = 200
	)
	seedKVIndexRecords(t, n, "u:", 0, seeded)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	// Encoded up front: a t.Fatal from a writer goroutine is illegal.
	recs := make([][]byte, 5)
	for v := range recs {
		recs[v] = kvIndexRecord(t, int64(v))
	}

	// The writers run for the WHOLE walk (they stop only once it has finished and
	// at least minWrites have landed), so writes are in flight before the install,
	// during the walk, and after it publishes. A key written before the install is
	// in the cache the walk reads; a key written after it is posted by the write
	// path. Neither may be missing.
	done := make(chan struct{})
	var written atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; ; i++ {
				key := fmt.Appendf(nil, "u:live%02d_%07d", w, i)
				if _, err := n.Call("put", ops.EncodePutArgs(key, recs[i%len(recs)], 0)); err != nil {
					t.Errorf("writer %d put %d: %v", w, i, err)
					return
				}
				written.Add(1)
				select {
				case <-done:
					if written.Load() >= minWrites {
						return
					}
				default:
				}
			}
		}(w)
	}
	go func() {
		defer close(done)
		n.applyKVIndexDefs()
	}()
	wg.Wait()

	live := int(written.Load())
	if live < minWrites {
		t.Fatalf("only %d concurrent writes landed, want at least %d", live, minWrites)
	}
	idx := n.getShard(0).KVIndex()
	if !idx.IsReady("by_age") {
		t.Fatal("by_age is not ready after a completed backfill")
	}
	keys, _ := idx.Stats("by_age")
	if keys != seeded+live {
		t.Fatalf("posted keys = %d, want %d (%d seeded + %d written during the walk) — a write was lost between the install and the walk",
			keys, seeded+live, seeded, live)
	}
}

func TestKVIndexBackfillDoesNotBlockWrites(t *testing.T) {
	seeded := 100_000
	if testing.Short() {
		seeded = 20_000
	}
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)

	seedKVIndexRecords(t, n, "u:", 0, seeded)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		n.applyKVIndexDefs()
	}()

	// Probe with real writes while the walk runs. IterateChunked releases each
	// cache shard's read lock every few thousand slots, so a writer waits at most
	// one chunk — not the length of a whole shard's walk.
	var worst time.Duration
	samples := 0
	for {
		key := fmt.Appendf(nil, "u:probe%07d", samples)
		began := time.Now()
		if _, err := n.Call("put", ops.EncodePutArgs(key, kvIndexRecord(t, 3), 0)); err != nil {
			t.Fatalf("probe put %d: %v", samples, err)
		}
		if el := time.Since(began); el > worst {
			worst = el
		}
		samples++
		select {
		case <-done:
			goto finished
		default:
		}
	}
finished:
	st := n.Stats().KVIndex
	if st.BackfillKeys < uint64(seeded) {
		t.Fatalf("the backfill visited %d keys, want at least %d — it did not walk the seeded keyspace", st.BackfillKeys, seeded)
	}
	t.Logf("backfill over %d keys: %d probe writes, worst %s", st.BackfillKeys, samples, worst)
	if worst > 250*time.Millisecond {
		t.Fatalf("a write issued during the backfill took %s (cap 250ms) — the walk is holding readers out", worst)
	}
}

// A flush is kvindex.Set.Reset's job and Reset's alone: it clears every posting
// AND marks every definition ready, because an empty posting set is exact over
// an empty keyspace. The observer must therefore see nothing to do — a
// re-install or a re-backfill after every flush would be pure waste, and on a
// large keyspace a recurring one.
func TestKVIndexFlushDoesNotRebackfill(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 40)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	idx := n.getShard(0).KVIndex()
	waitForKVIndex(t, 20*time.Second, "by_age to become ready", func() bool { return idx.IsReady("by_age") })
	freezeKVIndexObserver(t, n)

	if _, err := n.Call("flush", nil); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if !idx.IsReady("by_age") {
		t.Fatal("a flush left the definition not ready — an empty index over an empty keyspace IS exact")
	}
	if keys, _ := idx.Stats("by_age"); keys != 0 {
		t.Fatalf("posted keys after a flush = %d, want 0", keys)
	}

	before := n.Stats().KVIndex.Backfills
	n.applyKVIndexDefs() // exactly what the next observer tick would do
	if after := n.Stats().KVIndex.Backfills; after != before {
		t.Fatalf("the observer re-backfilled after a flush (%d -> %d walks)", before, after)
	}
	if keys, _ := idx.Stats("by_age"); keys != 0 {
		t.Fatalf("posted keys after the observer tick = %d, want 0", keys)
	}
}

func TestKVIndexObserverStopsOnClose(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	if n.kvIndexStop == nil {
		t.Fatal("the observer was never started")
	}
	// The observer is alive to begin with: a meta write moves the applied index
	// and the next tick runs a pass. Without this the assertion below would pass
	// on an observer that never ran at all.
	if err := n.SetKVIndex(kvIndexDefFixture("alive", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitForKVIndex(t, 20*time.Second, "the observer to run a pass", func() bool {
		return n.kvIndexPasses.Load() > 0
	})

	// stopKVIndexObserver must return (the goroutine must exit); a hang here is
	// the failure, surfaced by the test timeout.
	stopped := make(chan struct{})
	go func() {
		n.stopKVIndexObserver()
		close(stopped)
	}()
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("stopKVIndexObserver did not return within 5s")
	}
	// Idempotent: Close() calls it again.
	n.stopKVIndexObserver()

	// The goroutine is really gone, not merely un-waited-for: a meta write moves
	// the applied index, and a live ticker would run a pass within one interval.
	// This is a leak assertion with no goroutine-count tolerance to tune — a
	// tolerance wide enough for raft's own churn is wide enough to hide this one.
	before := n.kvIndexPasses.Load()
	if err := n.SetKVIndex(kvIndexDefFixture("after_stop", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex after stop: %v", err)
	}
	time.Sleep(3 * kvIndexObserveInterval)
	if after := n.kvIndexPasses.Load(); after != before {
		t.Fatalf("the observer ran %d passes after it was stopped — the goroutine is still alive", after-before)
	}

	tc.Close()
}

// The poll fires on EVERY meta write, and in PB mode the liveness beacons alone
// move the applied index every interval. A pass whose definitions and hosted
// groups have not changed must not rebuild every group's Set.
func TestKVIndexPassSkipsUnchangedInstall(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)
	seedKVIndexRecords(t, n, "u:", 0, 500)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}

	n.applyKVIndexDefs()
	first := n.kvIndexInstalls.Load()
	if first == 0 {
		t.Fatal("the first pass installed nothing")
	}
	// Several more passes over an unchanged catalog: no further installs.
	for i := 0; i < 3; i++ {
		n.applyKVIndexDefs()
	}
	if got := n.kvIndexInstalls.Load(); got != first {
		t.Fatalf("installs = %d after 3 no-op passes, want %d — an unchanged catalog is rebuilding every Set", got, first)
	}
	if got := n.kvIndexPasses.Load(); got < 4 {
		t.Fatalf("passes = %d, want at least 4 — the passes themselves must still run", got)
	}

	// A NEW definition is a change, so the install runs again.
	if err := n.SetKVIndex(kvIndexDefFixture("by_age2", "v:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	n.applyKVIndexDefs()
	if got := n.kvIndexInstalls.Load(); got != first+1 {
		t.Fatalf("installs = %d after a new definition, want %d", got, first+1)
	}
}

// A group removed and re-added at the SAME index is a different store with a
// brand-new, empty index Set. Keyed on the index alone, the two passes look
// identical, the install is skipped, and that Set never receives the definitions
// — after which Backfill has no posting to fill and no-ops forever. The index
// then stays not-ready on that group for the life of the process, which is
// fail-closed but entirely silent.
//
// It also covers the walk gate's re-arm: a re-added group can only be
// backfilled if AddShardOwner reopened the gate RemoveShardOwner closed.
func TestKVIndexPassInstallsIntoAReAddedShard(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)
	seedKVIndexRecords(t, n, "u:", 0, 200)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}

	n.applyKVIndexDefs()
	if st := n.Stats().KVIndex; st.Ready != 1 {
		t.Fatalf("Ready = %d before the swap, want 1", st.Ready)
	}

	// Take group 1 away and give it straight back at the same index.
	if err := n.RemoveShardOwner(1); err != nil {
		t.Fatalf("RemoveShardOwner: %v", err)
	}
	if err := n.AddShardOwner(1); err != nil {
		t.Fatalf("AddShardOwner: %v", err)
	}
	fresh := n.getShard(1)
	if fresh == nil {
		t.Fatal("the group was not re-added")
	}
	if len(fresh.KVIndex().Defs()) != 0 {
		t.Fatal("precondition: a re-added group must start with an empty index Set")
	}

	// The next pass has to notice. Ready counts names ready on EVERY hosted
	// group, so it can only return to 1 if the fresh Set was installed into and
	// then backfilled.
	n.applyKVIndexDefs()
	if got := len(fresh.KVIndex().Defs()); got != 1 {
		t.Fatalf("the re-added group has %d definitions installed, want 1 — the pass skipped its install", got)
	}
	if st := n.Stats().KVIndex; st.Ready != 1 {
		t.Fatalf("Ready = %d after the swap, want 1 — the re-added group never became usable", st.Ready)
	}
	if !fresh.KVIndex().IsReady("by_age") {
		t.Fatal("the re-added group's index never became ready")
	}
}

// Two reject numbers answering two questions: Rejects is a monotonic event
// counter (a rate can be taken from it), RejectedDefs a gauge that reports what
// is broken NOW and returns to zero when the offending definition is removed.
func TestKVIndexRejectsCounterAndGauge(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)

	// "#count" with no field in front of it passes wire.KVIndexDef.Validate and
	// fails record.ParsePath in kvindex.DefFrom.
	bad := wire.KVIndexDef{Name: "headless", PayloadPath: "#count", Kind: wire.KVIndexKindCount, Enabled: true}
	if err := n.SetKVIndex(bad, 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex(unparsable): %v", err)
	}
	n.applyKVIndexDefs()
	st := n.Stats().KVIndex
	if st.RejectedDefs != 1 {
		t.Fatalf("RejectedDefs = %d, want 1", st.RejectedDefs)
	}
	if st.Rejects == 0 {
		t.Fatal("Rejects = 0, want a reject event counted")
	}
	// Repeated passes: the gauge holds, the counter climbs.
	for i := 0; i < 3; i++ {
		n.applyKVIndexDefs()
	}
	st2 := n.Stats().KVIndex
	if st2.RejectedDefs != 1 {
		t.Fatalf("RejectedDefs = %d after 3 more passes, want 1 — a gauge reports state, not events", st2.RejectedDefs)
	}
	if st2.Rejects <= st.Rejects {
		t.Fatalf("Rejects = %d after 3 more passes, want > %d — the counter must be monotonic per event", st2.Rejects, st.Rejects)
	}

	// Removing the definition clears the GAUGE and leaves the counter alone.
	off := bad
	off.Enabled = false
	if err := n.SetKVIndex(off, 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex(disable): %v", err)
	}
	n.applyKVIndexDefs()
	st3 := n.Stats().KVIndex
	if st3.RejectedDefs != 0 {
		t.Fatalf("RejectedDefs = %d after the bad definition was dropped, want 0", st3.RejectedDefs)
	}
	if st3.Rejects != st2.Rejects {
		t.Fatalf("Rejects moved from %d to %d with nothing left to reject — it must never go backwards or drift", st2.Rejects, st3.Rejects)
	}
}

func TestKVIndexStatsPopulated(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 40)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitForKVIndex(t, 20*time.Second, "by_age to become ready", func() bool {
		return n.Stats().KVIndex.Ready == 1
	})

	st := n.Stats().KVIndex
	if st.Definitions != 1 {
		t.Errorf("Definitions = %d, want 1", st.Definitions)
	}
	if st.Backfills == 0 {
		t.Error("Backfills = 0, want at least one completed backfill")
	}
	if st.BackfillKeys < 40 {
		t.Errorf("BackfillKeys = %d, want at least 40", st.BackfillKeys)
	}
	if st.Rejects != 0 {
		t.Errorf("Rejects = %d, want 0", st.Rejects)
	}

	// A definition the meta FSM accepts but this build cannot parse: "#count"
	// with no field in front of it passes wire.KVIndexDef.Validate (no '/', the
	// suffix agrees with Kind) and fails record.ParsePath in kvindex.DefFrom.
	unparsable := wire.KVIndexDef{
		Name:        "headless_count",
		PayloadPath: "#count",
		Kind:        wire.KVIndexKindCount,
		Enabled:     true,
	}
	if err := n.SetKVIndex(unparsable, 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex(unparsable): %v", err)
	}
	waitForKVIndex(t, 20*time.Second, "the observer to reject the unparsable definition", func() bool {
		return n.Stats().KVIndex.Rejects > 0
	})
	if got := n.Stats().KVIndex.Definitions; got != 1 {
		t.Errorf("Definitions = %d after a rejected definition, want 1 (it must be skipped, not installed)", got)
	}
}

// --- __kv_index_list__ ----------------------------------------------------

func TestKVIndexListReportsReadyPerGroup(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]

	seedKVIndexRecords(t, n, "u:", 0, 40)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitForKVIndex(t, 20*time.Second, "both groups to report ready", func() bool {
		_, ready, err := n.ListKVIndexes(5 * time.Second)
		return err == nil && len(ready) == 1 && ready[0]
	})

	// Over the wire, through the admin op, with the shared codec.
	raw, err := n.Call(opKVIndexListName, nil)
	if err != nil {
		t.Fatalf("%s: %v", opKVIndexListName, err)
	}
	defs, ready, err := wire.DecodeKVIndexList(raw)
	if err != nil {
		t.Fatalf("DecodeKVIndexList: %v", err)
	}
	if len(defs) != 1 || defs[0].Name != "by_age" || !ready[0] {
		t.Fatalf("list = %+v ready=%v", defs, ready)
	}

	// Put ONE group back into "building" and the aggregate bit must go false.
	n.stopKVIndexObserver()
	building := n.getShard(1)
	if building == nil {
		t.Fatal("node does not host shard 1")
	}
	d, ok := building.KVIndex().Lookup("by_age")
	if !ok {
		t.Fatal("by_age is not installed on shard 1")
	}
	building.KVIndex().Install(nil)
	building.KVIndex().Install([]kvindex.Def{d})
	if building.KVIndex().IsReady("by_age") {
		t.Fatal("a freshly installed definition reports ready")
	}
	_, ready, err = n.ListKVIndexes(5 * time.Second)
	if err != nil {
		t.Fatalf("ListKVIndexes: %v", err)
	}
	if len(ready) != 1 || ready[0] {
		t.Fatalf("ready = %v, want [false]: one group is still building", ready)
	}
}

func TestKVIndexListUnreachableGroupIsNotReady(t *testing.T) {
	names := []string{"a", "b"}
	readyOnAll := encodeKVIndexReadyReply([]bool{true, true})

	// Every group answers ready.
	if got := aggregateKVIndexReady(names, [][]byte{readyOnAll, readyOnAll}, []error{nil, nil}); !got[0] || !got[1] {
		t.Fatalf("all-ready aggregate = %v, want [true true]", got)
	}
	// One group is unreachable: every bit goes FALSE, and the slice keeps its
	// length — a missing group must never shorten (or drop from) the answer.
	got := aggregateKVIndexReady(names, [][]byte{readyOnAll, nil}, []error{nil, errors.New("dial tcp: refused")})
	if len(got) != len(names) {
		t.Fatalf("aggregate len = %d, want %d", len(got), len(names))
	}
	if got[0] || got[1] {
		t.Fatalf("unreachable-group aggregate = %v, want [false false]", got)
	}
	// A group whose reply cannot be decoded is treated exactly like an
	// unreachable one, never as "ready".
	got = aggregateKVIndexReady(names, [][]byte{readyOnAll, {0xff}}, []error{nil, nil})
	if got[0] || got[1] {
		t.Fatalf("undecodable-reply aggregate = %v, want [false false]", got)
	}
	// A reply that is ready for one name and not the other narrows per name.
	half := encodeKVIndexReadyReply([]bool{true, false})
	got = aggregateKVIndexReady(names, [][]byte{readyOnAll, half}, []error{nil, nil})
	if !got[0] || got[1] {
		t.Fatalf("per-name aggregate = %v, want [true false]", got)
	}
}

func TestKVIndexReadyRequestRoundTrip(t *testing.T) {
	names := []string{"by_age", "by_rows"}
	g, got, err := decodeKVIndexReadyReq(encodeKVIndexReadyReq(3, names))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if g != 3 {
		t.Errorf("group = %d, want 3", g)
	}
	if len(got) != 2 || got[0] != "by_age" || got[1] != "by_rows" {
		t.Errorf("names = %v", got)
	}
	// Truncation is a decode error, never a short name list.
	frame := encodeKVIndexReadyReq(3, names)
	for i := 0; i < len(frame); i++ {
		if _, _, err := decodeKVIndexReadyReq(frame[:i]); err == nil {
			t.Errorf("a %d-byte prefix of the frame decoded cleanly", i)
		}
	}
	if _, _, err := decodeKVIndexReadyReq(append(frame, 0)); err == nil {
		t.Error("trailing bytes decoded cleanly")
	}
}

// --- forEachGroup ---------------------------------------------------------

// forEachGroupNode builds the smallest Node forEachGroup needs: it reads only
// the configured group count.
func forEachGroupNode(groups int) *Node { return &Node{cfg: Config{NumShards: groups}} }

func TestForEachGroupResultsAreInGroupOrder(t *testing.T) {
	n := forEachGroupNode(8)
	results, errs := n.forEachGroup(5*time.Second, func(_ context.Context, g int) ([]byte, error) {
		// Finish in reverse order, so a helper that appended as legs completed
		// would scramble the mapping.
		time.Sleep(time.Duration(8-g) * 5 * time.Millisecond)
		if g == 3 {
			return nil, fmt.Errorf("group %d refused", g)
		}
		return fmt.Appendf(nil, "g%d", g), nil
	})
	if len(results) != 8 || len(errs) != 8 {
		t.Fatalf("results/errs len = %d/%d, want 8/8", len(results), len(errs))
	}
	for g := 0; g < 8; g++ {
		if g == 3 {
			if errs[g] == nil {
				t.Errorf("group 3: want an error")
			}
			if results[g] != nil {
				t.Errorf("group 3: result = %q, want nil", results[g])
			}
			continue
		}
		if errs[g] != nil {
			t.Errorf("group %d: %v", g, errs[g])
		}
		if want := fmt.Sprintf("g%d", g); string(results[g]) != want {
			t.Errorf("results[%d] = %q, want %q", g, results[g], want)
		}
	}
}

func TestForEachGroupRunsInParallel(t *testing.T) {
	const (
		groups = 8
		leg    = 150 * time.Millisecond
	)
	n := forEachGroupNode(groups)
	began := time.Now()
	_, errs := n.forEachGroup(5*time.Second, func(_ context.Context, _ int) ([]byte, error) {
		time.Sleep(leg)
		return nil, nil
	})
	elapsed := time.Since(began)
	for g, err := range errs {
		if err != nil {
			t.Fatalf("group %d: %v", g, err)
		}
	}
	// Sequential would be groups*leg; parallel is one leg plus scheduling slack.
	if elapsed > 3*leg {
		t.Fatalf("%d groups of %s took %s — the legs are not running concurrently", groups, leg, elapsed)
	}
}

func TestForEachGroupPerGroupTimeout(t *testing.T) {
	const timeout = 200 * time.Millisecond
	n := forEachGroupNode(4)
	release := make(chan struct{})
	defer close(release)

	began := time.Now()
	results, errs := n.forEachGroup(timeout, func(ctx context.Context, g int) ([]byte, error) {
		if g == 1 {
			// Deliberately ignores ctx: the bound must hold even against a leg
			// that never looks at its deadline.
			<-release
			return []byte("late"), nil
		}
		_ = ctx
		return fmt.Appendf(nil, "g%d", g), nil
	})
	elapsed := time.Since(began)

	if errs[1] == nil {
		t.Fatal("the hung group did not time out")
	}
	if !errors.Is(errs[1], context.DeadlineExceeded) {
		t.Errorf("hung group error = %v, want a DeadlineExceeded wrap", errs[1])
	}
	if results[1] != nil {
		t.Errorf("hung group result = %q, want nil", results[1])
	}
	for _, g := range []int{0, 2, 3} {
		if errs[g] != nil {
			t.Errorf("group %d: %v", g, errs[g])
		}
		if want := fmt.Sprintf("g%d", g); string(results[g]) != want {
			t.Errorf("results[%d] = %q, want %q", g, results[g], want)
		}
	}
	if elapsed > 5*timeout {
		t.Fatalf("one hung group stalled the fan-out for %s (per-group timeout %s)", elapsed, timeout)
	}
}

// A backfill walk aliases the store's LIVE MMAP, which Store.Close unmaps. The
// guarantee is a DRAIN, in the shape the cache already uses for its own
// goroutines: RemoveShardOwner signals the walks registered on that group and
// WAITS for them before it closes the store, so the store is never closed while
// a walk is inside it.
//
// This half is deterministic — it pins the ordering with no walk timing in it.
func TestRemoveShardOwnerDrainsIndexWalks(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)

	stop, done, ok := n.beginKVIndexWalk(0)
	if !ok {
		t.Fatal("a walk could not register on a hosted group")
	}

	removed := make(chan error, 1)
	go func() { removed <- n.RemoveShardOwner(0) }()

	// It must NOT return while the walk is still registered: returning here is
	// exactly the bug — Store.Close would unmap pages the walk is reading.
	select {
	case err := <-removed:
		t.Fatalf("RemoveShardOwner returned (err=%v) while a walk was in flight", err)
	case <-time.After(200 * time.Millisecond):
	}
	// And the walk has been told to stop, so the wait is bounded.
	select {
	case <-stop:
	default:
		t.Fatal("the in-flight walk was never signalled — RemoveShardOwner would wait forever")
	}

	done()
	select {
	case err := <-removed:
		if err != nil {
			t.Fatalf("RemoveShardOwner: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("RemoveShardOwner did not return after the walk stopped")
	}

	// No NEW walk may start on the removed group: a pass holding a stale store
	// pointer must not begin reading a store that has just been closed.
	if _, _, ok := n.beginKVIndexWalk(0); ok {
		t.Fatal("a walk registered on a group that has been removed")
	}
	// A group this node still hosts is unaffected.
	if _, d, ok := n.beginKVIndexWalk(1); !ok {
		t.Fatal("removing group 0 shut group 1's gate")
	} else {
		d()
	}
}

// The same guarantee end to end, with a REAL backfill racing a real
// RemoveShardOwner: it must neither take the process down nor publish what it
// managed to read. Run under -race with repeats.
func TestKVIndexBackfillDrainsBeforeShardRemoval(t *testing.T) {
	seeded := 100_000
	if testing.Short() {
		seeded = 20_000
	}
	aborts := 0
	const attempts = 4
	for attempt := 0; attempt < attempts; attempt++ {
		func() {
			tc := newTestCluster(t, 1, 1)
			n := tc.nodes[0]
			freezeKVIndexObserver(t, n)

			seedKVIndexRecords(t, n, "u:", 0, seeded)
			if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
				t.Fatalf("attempt %d: SetKVIndex: %v", attempt, err)
			}
			s := n.getShard(0)
			idx := s.KVIndex()
			idx.Install(n.kvIndexDefsFromCatalog())

			finished := make(chan bool, 1)
			go func() { finished <- n.backfillKVIndex(0, s, idx, "by_age") }()

			// Let the walk get going, then pull the shard out from under it.
			time.Sleep(2 * time.Millisecond)
			if err := n.RemoveShardOwner(0); err != nil {
				t.Fatalf("attempt %d: RemoveShardOwner: %v", attempt, err)
			}

			// RemoveShardOwner waited for it, so it has already returned.
			select {
			case completed := <-finished:
				if !completed {
					aborts++
					if idx.IsReady("by_age") {
						t.Fatalf("attempt %d: an aborted backfill was published as ready", attempt)
					}
				}
			case <-time.After(time.Second):
				t.Fatalf("attempt %d: RemoveShardOwner returned while the backfill was still running", attempt)
			}
			tc.Close()
		}()
	}
	// At least one attempt must actually have raced, or the test proved nothing
	// about the drain. The walk over `seeded` keys takes tens of milliseconds
	// against a 2ms head start, so this is not a close call.
	if aborts == 0 {
		t.Fatalf("no attempt out of %d aborted its walk — the removal never overlapped a live backfill", attempts)
	}
}

// Close waits on the in-flight pass, so the pass has to look at the stop signal.
// Without that check a node hosting several groups over a large keyspace takes
// the SUM of every remaining backfill to shut down.
func TestKVIndexPassAbortsOnStop(t *testing.T) {
	tc := newTestCluster(t, 1, 4)
	n := tc.nodes[0]
	n.stopKVIndexObserver() // this test drives the passes itself
	seedKVIndexRecords(t, n, "u:", 0, 2000)

	// Control: with the observer already stopped, a pass must do nothing at all.
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	began := time.Now()
	n.applyKVIndexDefs()
	if got := n.Stats().KVIndex.Backfills; got != 0 {
		t.Fatalf("a pass ran %d backfills after the observer was stopped, want 0", got)
	}
	if elapsed := time.Since(began); elapsed > 2*time.Second {
		t.Fatalf("an aborted pass took %s", elapsed)
	}

	// And the same pass on a node that has NOT been stopped does the work, so
	// the assertion above is about the stop signal and not about the pass being
	// broken.
	tc2 := newTestCluster(t, 1, 4)
	n2 := tc2.nodes[0]
	seedKVIndexRecords(t, n2, "u:", 0, 2000)
	if err := n2.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitForKVIndex(t, 20*time.Second, "every hosted group to finish its backfill", func() bool {
		return n2.Stats().KVIndex.Ready == 1
	})
	if got := n2.Stats().KVIndex.Backfills; got == 0 {
		t.Fatal("a live observer ran no backfills at all")
	}
}

// Node construction must not pay for a full-keyspace walk. The definitions go in
// synchronously (so the very first apply reindexes through them); the walk
// happens on the observer goroutine afterwards.
func TestKVIndexBackfillIsOffTheConstructionPath(t *testing.T) {
	seeded := 100_000
	if testing.Short() {
		seeded = 20_000
	}
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	seedKVIndexRecords(t, n, "u:", 0, seeded)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	waitForKVIndex(t, 30*time.Second, "the first node to finish its backfill", func() bool {
		return n.Stats().KVIndex.Ready == 1
	})
	dir := tc.dataDirs[0]
	tc.Close()

	// Restart on the SAME data dir: the catalog and the keyspace are both already
	// there, so this is the restart the split is for.
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID:    tc.peers[0].NodeID,
		DataDir:   dir,
		NumShards: 1,
		Bootstrap: true,
		RaftAddr:  tc.peers[0].RaftAddr,
		Peers:     tc.peers,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg, Bootstrap: true,
			RaftHeartbeatMs: 200, RaftElectionMs: 1000, NoSync: true,
		},
		Ops: reg,
	}
	began := time.Now()
	n2, err := New(cfg)
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	construct := time.Since(began)
	defer func() { _ = n2.Close() }()

	// Raft bootstrap dominates this number, so it is a smoke bound, not the
	// assertion — TestKVIndexInstallWithoutBackfill is what pins the split.
	if construct > 20*time.Second {
		t.Fatalf("node construction took %s", construct)
	}
	// The definitions are back (the install is synchronous, so this needs at most
	// one poll interval for the meta FSM to finish replaying its own log).
	waitForKVIndex(t, 30*time.Second, "the restarted node to install its definitions", func() bool {
		return n2.Stats().KVIndex.Definitions == 1
	})
	// And the walk lands on the observer goroutine.
	waitForKVIndex(t, 30*time.Second, "the restarted node to backfill", func() bool {
		return n2.Stats().KVIndex.Ready == 1
	})
	if got := n2.Stats().KVIndex.BackfillKeys; got < uint64(seeded) {
		t.Fatalf("the restarted node walked %d keys, want at least %d", got, seeded)
	}
}

// The deterministic half of the split: the synchronous part of node start
// installs definitions and walks NOTHING. Calling the two halves by hand removes
// every timing question — construction cannot be slow because of a walk it does
// not perform.
func TestKVIndexInstallWithoutBackfill(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)
	seedKVIndexRecords(t, n, "u:", 0, 2000)
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex: %v", err)
	}
	n.installKVIndexDefs()

	st := n.Stats().KVIndex
	if st.Definitions != 1 {
		t.Fatalf("Definitions after the install-only pass = %d, want 1", st.Definitions)
	}
	if st.BackfillKeys != 0 || st.Backfills != 0 {
		t.Fatalf("the install-only pass walked %d keys in %d backfills, want 0/0", st.BackfillKeys, st.Backfills)
	}
	if st.Ready != 0 {
		t.Fatalf("Ready = %d after an install with no walk, want 0 — an unwalked index must not be usable", st.Ready)
	}

	// The full pass then does the walk the install skipped.
	n.applyKVIndexDefs()
	st = n.Stats().KVIndex
	if st.Ready != 1 {
		t.Fatalf("Ready = %d after the full pass, want 1", st.Ready)
	}
	if st.BackfillKeys < 2000 {
		t.Fatalf("the full pass walked %d keys, want at least 2000", st.BackfillKeys)
	}
}
