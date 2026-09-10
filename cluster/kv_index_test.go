// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
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
	// Issue the ADMIN OP (not the Go method) so the op name is exercised end to end.
	if _, err := follower.Call(opKVIndexSetName, wire.EncodeKVIndexSetArgs(d)); err != nil {
		t.Fatalf("%s at a follower: %v", opKVIndexSetName, err)
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
	if _, err := follower.Call(opKVIndexSetName, wire.EncodeKVIndexSetArgs(off)); err != nil {
		t.Fatalf("%s disable: %v", opKVIndexSetName, err)
	}
	if _, ok := follower.meta.FSM.KVIndexes()["by_age"]; ok {
		t.Fatal("the follower returned before its own FSM reflected the disable")
	}

	// A malformed definition is refused at the edge, not committed.
	if _, err := follower.Call(opKVIndexSetName, []byte{0xff}); err == nil {
		t.Fatal("a malformed __kv_index_set__ frame was accepted")
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
	n.stopKVIndexObserver()

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
	n.stopKVIndexObserver()

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
	n.stopKVIndexObserver()

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
	baseline := runtime.NumGoroutine()

	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	if n.kvIndexStop == nil {
		t.Fatal("the observer was never started")
	}

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

	tc.Close()
	waitForKVIndex(t, 10*time.Second, "goroutines to return to baseline", func() bool {
		return runtime.NumGoroutine() <= baseline+8
	})
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
