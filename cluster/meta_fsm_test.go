// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"encoding/gob"
	"io"
	"sync"
	"testing"

	"github.com/hashicorp/raft"
)

// gobEncodeForTest gob-encodes v into w, mirroring how encodeState writes a
// snapshot blob. Used to synthesize an OLD-format (pre-LastIndex) snapshot.
func gobEncodeForTest(w io.Writer, v any) error {
	return gob.NewEncoder(w).Encode(v)
}

func TestMetaFSMApplySetMembers(t *testing.T) {
	f := NewMetaFSM()
	entry := LogEntry{
		Op:        OpSetMembers,
		Members:   []Peer{{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"}},
		NumShards: 4,
	}
	data, err := encodeLogEntry(entry)
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Apply(&raft.Log{Data: data}); got != nil {
		t.Fatalf("Apply: %v", got)
	}
	st := f.State()
	if len(st.Members) != 1 || st.Members[0].NodeID != "n1" {
		t.Errorf("Members = %+v", st.Members)
	}
	if st.NumShards != 4 {
		t.Errorf("NumShards = %d, want 4", st.NumShards)
	}
	if len(st.Placement) != 4 {
		t.Errorf("Placement len = %d, want 4", len(st.Placement))
	}
	for i, p := range st.Placement {
		if len(p) != 1 || p[0] != "n1" {
			t.Errorf("Placement[%d] = %v", i, p)
		}
	}
}

func TestMetaFSMSnapshotRestore(t *testing.T) {
	f1 := NewMetaFSM()
	data, _ := encodeLogEntry(LogEntry{
		Op:        OpSetMembers,
		Members:   []Peer{{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"}},
		NumShards: 4,
	})
	_ = f1.Apply(&raft.Log{Data: data})

	snap, err := f1.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := snap.Persist(noopSink{w: &buf}); err != nil {
		t.Fatal(err)
	}

	f2 := NewMetaFSM()
	if err := f2.Restore(io.NopCloser(&buf)); err != nil {
		t.Fatal(err)
	}
	if !equalState(f1.State(), f2.State()) {
		t.Errorf("state mismatch after restore")
	}
}

func TestMetaFSMApplyUnknownOp(t *testing.T) {
	f := NewMetaFSM()
	bad := []byte{0xFF}
	// Encode the bad-op via the path: 1 byte op + minimal gob body for empty struct.
	good, _ := encodeLogEntry(LogEntry{Op: 0xFF})
	if got := f.Apply(&raft.Log{Data: good}); got == nil {
		t.Error("expected error on unknown op")
	}
	// Also: malformed data (1 byte, no gob payload).
	if got := f.Apply(&raft.Log{Data: bad}); got == nil {
		t.Error("expected decode error on malformed entry")
	}
}

func TestMetaFSMStateIsDeepCopy(t *testing.T) {
	f := NewMetaFSM()
	data, _ := encodeLogEntry(LogEntry{
		Op:        OpSetMembers,
		Members:   []Peer{{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"}},
		NumShards: 2,
	})
	_ = f.Apply(&raft.Log{Data: data})

	st1 := f.State()
	// Mutate the returned copy.
	if len(st1.Members) > 0 {
		st1.Members[0].NodeID = "MUTATED"
	}
	if len(st1.Placement) > 0 && len(st1.Placement[0]) > 0 {
		st1.Placement[0][0] = "MUTATED"
	}
	st2 := f.State()
	if st2.Members[0].NodeID != "n1" {
		t.Errorf("Members not deep-copied: %s", st2.Members[0].NodeID)
	}
	if st2.Placement[0][0] != "n1" {
		t.Errorf("Placement not deep-copied: %s", st2.Placement[0][0])
	}
}

func TestApplySetMembersRFChangeIsNotIdempotent(t *testing.T) {
	f := NewMetaFSM()
	peers := []Peer{
		{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"},
		{NodeID: "n2", RaftAddr: "b:1", ServerAddr: "b:2"},
		{NodeID: "n3", RaftAddr: "c:1", ServerAddr: "c:2"},
	}
	apply := func(rf int) {
		data, _ := encodeLogEntry(LogEntry{Op: OpSetMembers, Members: peers, NumShards: 2, ReplicationFactor: rf})
		if resp := f.Apply(&raft.Log{Data: data}); resp != nil {
			t.Fatalf("Apply(RF=%d): %v", rf, resp)
		}
	}
	// Bootstrap with RF=3 (full replication on 3 nodes).
	apply(3)
	st := f.State()
	if len(st.Placement[0]) != 3 {
		t.Fatalf("after RF=3: Placement[0] len = %d, want 3", len(st.Placement[0]))
	}
	// Change only RF to 2. Placement must reflect the new RF.
	apply(2)
	st = f.State()
	if len(st.Placement[0]) != 2 {
		t.Fatalf("after RF=2: Placement[0] len = %d, want 2 (RF change was dropped)", len(st.Placement[0]))
	}
	if st.ReplicationFactor != 2 || !st.ReplicationFactorSet {
		t.Fatalf("ReplicationFactor not recorded: got %d set=%v", st.ReplicationFactor, st.ReplicationFactorSet)
	}
}

func TestMetaFSMApplySortsMembersByNodeID(t *testing.T) {
	f := NewMetaFSM()
	// Pass members in reverse alphabetical order.
	entry := LogEntry{
		Op: OpSetMembers,
		Members: []Peer{
			{NodeID: "n3", RaftAddr: "c:1", ServerAddr: "c:2"},
			{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"},
			{NodeID: "n2", RaftAddr: "b:1", ServerAddr: "b:2"},
		},
		NumShards: 2,
	}
	data, err := encodeLogEntry(entry)
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Apply(&raft.Log{Data: data}); got != nil {
		t.Fatalf("Apply: %v", got)
	}
	st := f.State()
	for i, want := range []string{"n1", "n2", "n3"} {
		if st.Members[i].NodeID != want {
			t.Errorf("Members[%d] = %q, want %q", i, st.Members[i].NodeID, want)
		}
	}
}

func TestMetaFSMSnapshotRestoreCatalog(t *testing.T) {
	src := NewMetaFSM()
	for _, e := range []struct {
		c string
		p uint32
	}{{"default/docs", 8}, {"default/img", 3}} {
		b, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: e.c, Partitions: e.p})
		if resp := src.Apply(&raft.Log{Data: b}); resp != nil {
			t.Fatalf("apply: %v", resp)
		}
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
	st := dst.State()
	if st.Catalog["default/docs"] != 8 || st.Catalog["default/img"] != 3 {
		t.Fatalf("restored catalog = %v, want docs=8 img=3", st.Catalog)
	}
}

func TestMetaFSMApplySetCatalogEntry(t *testing.T) {
	m := NewMetaFSM()
	apply := func(c string, p uint32) {
		e, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: c, Partitions: p})
		if resp := m.Apply(&raft.Log{Data: e}); resp != nil {
			t.Fatalf("apply(%s,%d) returned %v", c, p, resp)
		}
	}
	apply("default/docs", 8)
	apply("default/img", 3)
	if st := m.State(); st.Catalog["default/docs"] != 8 || st.Catalog["default/img"] != 3 {
		t.Fatalf("catalog = %v", st.Catalog)
	}
	apply("default/docs", 16) // resplit overwrites
	if st := m.State(); st.Catalog["default/docs"] != 16 {
		t.Fatalf("after resplit docs = %d, want 16", st.Catalog["default/docs"])
	}
}

func TestMetaFSMStateCatalogIsDeepCopy(t *testing.T) {
	m := NewMetaFSM()
	e, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "default/docs", Partitions: 8})
	if resp := m.Apply(&raft.Log{Data: e}); resp != nil {
		t.Fatalf("apply returned %v", resp)
	}
	st := m.State()
	st.Catalog["default/docs"] = 999 // mutate the returned copy
	if again := m.State(); again.Catalog["default/docs"] != 8 {
		t.Fatalf("deep-copy violated: FSM catalog mutated to %d", again.Catalog["default/docs"])
	}
}

func TestMetaFSMCatalogGeneration(t *testing.T) {
	m := NewMetaFSM()
	e, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8, Generation: 3})
	if resp := m.Apply(&raft.Log{Data: e}); resp != nil {
		t.Fatalf("apply returned %v", resp)
	}
	p, gen, ok := m.CatalogLookupGen("docs")
	if !ok || p != 8 || gen != 3 {
		t.Fatalf("CatalogLookupGen = (%d,%d,%v), want (8,3,true)", p, gen, ok)
	}
	// State() must deep-copy CatalogGen.
	st := m.State()
	st.CatalogGen["docs"] = 999
	if _, gen2, _ := m.CatalogLookupGen("docs"); gen2 != 3 {
		t.Fatalf("CatalogGen not deep-copied: FSM gen mutated to %d", gen2)
	}
}

// TestMetaFSMApplySetCatalogReshard mirrors TestMetaFSMApplySetCatalogEntry for
// the reshard path: applying an OpSetCatalogReshard (Status!=0) records the entry,
// and applying a Stable clear (Status=0) deletes it so lookups report a clean miss.
func TestMetaFSMApplySetCatalogReshard(t *testing.T) {
	m := NewMetaFSM()
	begin, _ := encodeLogEntry(LogEntry{
		Op:               OpSetCatalogReshard,
		Collection:       "docs",
		ReshardStatus:    1,
		ReshardTargetP:   4,
		ReshardTargetGen: 1,
		ReshardSourceP:   2,
		ReshardSourceGen: 0,
	})
	if resp := m.Apply(&raft.Log{Data: begin}); resp != nil {
		t.Fatalf("apply begin returned %v", resp)
	}
	e, ok := m.CatalogReshardLookup("docs")
	if !ok || e.Status != 1 || e.TargetP != 4 || e.TargetGen != 1 || e.SourceP != 2 || e.SourceGen != 0 {
		t.Fatalf("CatalogReshardLookup = (%+v,%v), want (Status=1,TargetP=4,TargetGen=1,SourceP=2,SourceGen=0,true)", e, ok)
	}
	// Stable clear (Status=0) deletes the entry.
	clear, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogReshard, Collection: "docs", ReshardStatus: 0})
	if resp := m.Apply(&raft.Log{Data: clear}); resp != nil {
		t.Fatalf("apply clear returned %v", resp)
	}
	if e, ok := m.CatalogReshardLookup("docs"); ok {
		t.Fatalf("after Stable clear lookup = (%+v,%v), want (zero,false)", e, ok)
	}
	// The entry must actually be removed from the underlying map (kept sparse).
	if _, present := m.State().CatalogReshard["docs"]; present {
		t.Fatal("Stable clear left a stale CatalogReshard entry")
	}
}

// TestMetaFSMCatalogReshardDeepCopy mirrors TestMetaFSMStateCatalogIsDeepCopy:
// mutating the map returned by State() must not affect the FSM's internal map.
func TestMetaFSMCatalogReshardDeepCopy(t *testing.T) {
	m := NewMetaFSM()
	e, _ := encodeLogEntry(LogEntry{
		Op:               OpSetCatalogReshard,
		Collection:       "docs",
		ReshardStatus:    1,
		ReshardTargetP:   4,
		ReshardTargetGen: 1,
	})
	if resp := m.Apply(&raft.Log{Data: e}); resp != nil {
		t.Fatalf("apply returned %v", resp)
	}
	st := m.State()
	st.CatalogReshard["docs"] = ReshardEntry{Status: 9, TargetP: 99, TargetGen: 99} // mutate returned copy
	if again, ok := m.CatalogReshardLookup("docs"); !ok || again.Status != 1 || again.TargetP != 4 || again.TargetGen != 1 {
		t.Fatalf("CatalogReshard not deep-copied: FSM entry mutated to %+v (ok=%v)", again, ok)
	}
}

// TestMetaFSMSnapshotRestoreCatalogReshard mirrors TestMetaFSMSnapshotRestoreCatalog
// for the reshard path. This guards crash recovery: an ACTIVE reshard entry must
// survive snapshot+restore intact. Also: an old-style state (nil CatalogReshard)
// must restore cleanly as all-Stable.
func TestMetaFSMSnapshotRestoreCatalogReshard(t *testing.T) {
	src := NewMetaFSM()
	// A partitioned collection that is actively resharding.
	cat, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 2, Generation: 0})
	if resp := src.Apply(&raft.Log{Data: cat}); resp != nil {
		t.Fatalf("apply catalog: %v", resp)
	}
	rsh, _ := encodeLogEntry(LogEntry{
		Op:               OpSetCatalogReshard,
		Collection:       "docs",
		ReshardStatus:    1,
		ReshardTargetP:   4,
		ReshardTargetGen: 1,
		ReshardSourceP:   2,
		ReshardSourceGen: 0,
	})
	if resp := src.Apply(&raft.Log{Data: rsh}); resp != nil {
		t.Fatalf("apply reshard: %v", resp)
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
	e, ok := dst.CatalogReshardLookup("docs")
	if !ok || e.Status != 1 || e.TargetP != 4 || e.TargetGen != 1 || e.SourceP != 2 || e.SourceGen != 0 {
		t.Fatalf("restored reshard = (%+v,%v), want (Status=1,TargetP=4,TargetGen=1,SourceP=2,SourceGen=0,true)", e, ok)
	}
	// Full state must match across the snapshot/restore boundary (relies on the
	// updated equalState comparing CatalogReshard).
	if !equalState(src.State(), dst.State()) {
		t.Fatalf("state mismatch after restore: src=%+v dst=%+v", src.State(), dst.State())
	}

	// Old-style state with nil CatalogReshard restores cleanly as all-Stable.
	old := NewMetaFSM()
	oldCat, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
	if resp := old.Apply(&raft.Log{Data: oldCat}); resp != nil {
		t.Fatalf("apply old catalog: %v", resp)
	}
	osnap, err := old.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	var obuf bytes.Buffer
	if err := osnap.Persist(noopSink{w: &obuf}); err != nil {
		t.Fatal(err)
	}
	odst := NewMetaFSM()
	if err := odst.Restore(io.NopCloser(&obuf)); err != nil {
		t.Fatal(err)
	}
	if odst.State().CatalogReshard != nil {
		t.Fatalf("old-format restored CatalogReshard = %v, want nil (all-Stable)", odst.State().CatalogReshard)
	}
	if e, ok := odst.CatalogReshardLookup("docs"); ok {
		t.Fatalf("old-format reshard lookup = (%+v,%v), want (zero,false)", e, ok)
	}
}

// TestMetaFSMAppliedIndexAdvances proves the command frontier starts at 0 and
// advances to log.Index on EVERY command Apply (across op types).
func TestMetaFSMAppliedIndexAdvances(t *testing.T) {
	m := NewMetaFSM()
	if got := m.AppliedIndex(); got != 0 {
		t.Fatalf("fresh AppliedIndex = %d, want 0", got)
	}
	apply := func(idx uint64, e LogEntry) {
		b, _ := encodeLogEntry(e)
		if resp := m.Apply(&raft.Log{Index: idx, Data: b}); resp != nil {
			t.Fatalf("apply idx=%d returned %v", idx, resp)
		}
	}
	apply(5, LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
	if got := m.AppliedIndex(); got != 5 {
		t.Fatalf("AppliedIndex after idx=5 = %d, want 5", got)
	}
	apply(9, LogEntry{Op: OpSetAliasBatch, AliasBatch: []AliasAction{{Alias: "a", Canonical: "docs"}}})
	if got := m.AppliedIndex(); got != 9 {
		t.Fatalf("AppliedIndex after idx=9 = %d, want 9", got)
	}
}

// TestMetaFSMAppliedIndexMonotonic proves a lower index applied after a higher
// one does NOT regress the frontier. Also: a malformed entry (decode failure)
// must NOT advance the frontier (the defer sits after a successful decode).
func TestMetaFSMAppliedIndexMonotonic(t *testing.T) {
	m := NewMetaFSM()
	hi, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
	if resp := m.Apply(&raft.Log{Index: 100, Data: hi}); resp != nil {
		t.Fatalf("apply idx=100 returned %v", resp)
	}
	if got := m.AppliedIndex(); got != 100 {
		t.Fatalf("AppliedIndex = %d, want 100", got)
	}
	// A lower index must not regress.
	lo, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "img", Partitions: 3})
	if resp := m.Apply(&raft.Log{Index: 42, Data: lo}); resp != nil {
		t.Fatalf("apply idx=42 returned %v", resp)
	}
	if got := m.AppliedIndex(); got != 100 {
		t.Fatalf("AppliedIndex regressed to %d after lower idx, want 100", got)
	}
	// A malformed entry at a higher index must NOT advance the frontier.
	if resp := m.Apply(&raft.Log{Index: 200, Data: []byte{0xFF}}); resp == nil {
		t.Fatal("expected decode error on malformed entry")
	}
	if got := m.AppliedIndex(); got != 100 {
		t.Fatalf("AppliedIndex advanced on a decode failure to %d, want 100", got)
	}
}

// TestMetaFSMAppliedIndexConcurrent runs concurrent applies under -race and
// asserts the frontier ends at the highest index and never regresses. Raft
// serializes Apply in production; this exercises advanceApplied's CAS loop +
// the atomic load in AppliedIndex for the concurrent barrier reader.
func TestMetaFSMAppliedIndexConcurrent(t *testing.T) {
	m := NewMetaFSM()
	const n = 200
	var wg sync.WaitGroup
	for i := 1; i <= n; i++ {
		wg.Add(1)
		go func(idx uint64) {
			defer wg.Done()
			b, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
			_ = m.Apply(&raft.Log{Index: idx, Data: b})
		}(uint64(i))
	}
	// Concurrent readers (the barrier's access pattern).
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _ = m.AppliedIndex() }()
	}
	wg.Wait()
	if got := m.AppliedIndex(); got != n {
		t.Fatalf("AppliedIndex = %d after concurrent applies, want %d", got, n)
	}
}

// TestMetaFSMAppliedIndexSurvivesSnapshotRestore is the CRITICAL test: the
// command frontier must survive a Snapshot→Persist→Restore round-trip via the
// REAL paths, so a snapshot-restored follower reports the snapshot's index (NOT
// 0) and does not wait forever for a frontier <= that index.
func TestMetaFSMAppliedIndexSurvivesSnapshotRestore(t *testing.T) {
	src := NewMetaFSM()
	b, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
	if resp := src.Apply(&raft.Log{Index: 777, Data: b}); resp != nil {
		t.Fatalf("apply: %v", resp)
	}
	if got := src.AppliedIndex(); got != 777 {
		t.Fatalf("src AppliedIndex = %d, want 777", got)
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
	if dst.AppliedIndex() != 0 {
		t.Fatalf("fresh dst AppliedIndex = %d, want 0", dst.AppliedIndex())
	}
	if err := dst.Restore(io.NopCloser(&buf)); err != nil {
		t.Fatal(err)
	}
	if got := dst.AppliedIndex(); got != 777 {
		t.Fatalf("restored AppliedIndex = %d, want 777 (would wait forever at 0)", got)
	}
	// Restore must be monotonic: a snapshot at a LOWER index must not regress an
	// FSM that has already applied past it.
	ahead := NewMetaFSM()
	a, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "img", Partitions: 3})
	if resp := ahead.Apply(&raft.Log{Index: 1000, Data: a}); resp != nil {
		t.Fatalf("apply: %v", resp)
	}
	// buf was drained by Restore; re-snapshot src to get a fresh idx=777 blob.
	snap2, _ := src.Snapshot()
	var blob777 bytes.Buffer
	if err := snap2.Persist(noopSink{w: &blob777}); err != nil {
		t.Fatal(err)
	}
	if err := ahead.Restore(io.NopCloser(&blob777)); err != nil {
		t.Fatal(err)
	}
	if got := ahead.AppliedIndex(); got != 1000 {
		t.Fatalf("AppliedIndex regressed to %d on a lower-index snapshot restore, want 1000", got)
	}
}

// TestMetaFSMRestoreOldFormatSnapshotNoIndex proves an OLD-format snapshot (a gob
// State written BEFORE the LastIndex field existed) restores WITHOUT panic and
// reports frontier 0 (the node replays the log afterward). We synthesize the old
// format by gob-encoding a struct that has the same field names MINUS LastIndex,
// which is byte-identical to what the pre-upgrade encodeState produced.
func TestMetaFSMRestoreOldFormatSnapshotNoIndex(t *testing.T) {
	// oldState mirrors State without LastIndex — exactly the pre-upgrade gob layout.
	type oldState struct {
		NumShards      int
		Members        []Peer
		Placement      [][]string
		Catalog        map[string]uint32
		CatalogGen     map[string]uint32
		CatalogReshard map[string]ReshardEntry
		Aliases        map[string]string
	}
	var buf bytes.Buffer
	if err := gobEncodeForTest(&buf, oldState{
		NumShards: 4,
		Catalog:   map[string]uint32{"docs": 8},
	}); err != nil {
		t.Fatal(err)
	}

	m := NewMetaFSM()
	if err := m.Restore(io.NopCloser(&buf)); err != nil {
		t.Fatalf("restore of old-format snapshot failed: %v", err)
	}
	if got := m.AppliedIndex(); got != 0 {
		t.Fatalf("old-format restored AppliedIndex = %d, want 0", got)
	}
	// The catalog content must still restore correctly (proves field-name match).
	if p, ok := m.CatalogLookup("docs"); !ok || p != 8 {
		t.Fatalf("old-format restored catalog docs = (%d,%v), want (8,true)", p, ok)
	}
}

type noopSink struct{ w io.Writer }

func (n noopSink) Write(p []byte) (int, error) { return n.w.Write(p) }
func (n noopSink) Close() error                { return nil }
func (n noopSink) ID() string                  { return "test" }
func (n noopSink) Cancel() error               { return nil }

func equalState(a, b State) bool {
	if a.NumShards != b.NumShards || len(a.Members) != len(b.Members) || len(a.Placement) != len(b.Placement) {
		return false
	}
	if a.ReplicationFactor != b.ReplicationFactor || a.ReplicationFactorSet != b.ReplicationFactorSet {
		return false
	}
	for i := range a.Members {
		if a.Members[i] != b.Members[i] {
			return false
		}
	}
	for i := range a.Placement {
		if len(a.Placement[i]) != len(b.Placement[i]) {
			return false
		}
		for j := range a.Placement[i] {
			if a.Placement[i][j] != b.Placement[i][j] {
				return false
			}
		}
	}
	if len(a.Catalog) != len(b.Catalog) {
		return false
	}
	for k, v := range a.Catalog {
		if b.Catalog[k] != v {
			return false
		}
	}
	if len(a.CatalogGen) != len(b.CatalogGen) {
		return false
	}
	for k, v := range a.CatalogGen {
		if b.CatalogGen[k] != v {
			return false
		}
	}
	// CatalogReshard must be compared too: without this a snapshot/restore test
	// that loses reshard state would silently pass.
	if len(a.CatalogReshard) != len(b.CatalogReshard) {
		return false
	}
	for k, v := range a.CatalogReshard {
		if b.CatalogReshard[k] != v {
			return false
		}
	}
	// Aliases must be compared too: without this a snapshot/restore test that
	// loses alias state would silently pass.
	if len(a.Aliases) != len(b.Aliases) {
		return false
	}
	for k, v := range a.Aliases {
		if b.Aliases[k] != v {
			return false
		}
	}
	// ShardEpoch/ShardPrimary/ShardISR (primary-backup/ISR control plane) must be
	// compared too so a re-applied identical epoch/ISR is detected and a
	// snapshot/restore that loses shard-replication state cannot silently pass.
	if len(a.ShardEpoch) != len(b.ShardEpoch) {
		return false
	}
	for k, v := range a.ShardEpoch {
		if b.ShardEpoch[k] != v {
			return false
		}
	}
	if len(a.ShardPrimary) != len(b.ShardPrimary) {
		return false
	}
	for k, v := range a.ShardPrimary {
		if b.ShardPrimary[k] != v {
			return false
		}
	}
	if len(a.ShardISR) != len(b.ShardISR) {
		return false
	}
	for k, v := range a.ShardISR {
		bv := b.ShardISR[k]
		if len(v) != len(bv) {
			return false
		}
		for i := range v {
			if v[i] != bv[i] {
				return false
			}
		}
	}
	return true
}

// TestFSMApplyAliasBatchAtomic proves the atomic-batch mechanism: an
// OpSetAliasBatch carrying N actions applies all N under the single Apply lock,
// so a swap ({delete a},{create a→z}) leaves the alias at the NEW target, never
// undefined. Also asserts State() deep-copies Aliases.
func TestFSMApplyAliasBatchAtomic(t *testing.T) {
	m := NewMetaFSM()
	applyBatch := func(actions []AliasAction) {
		e, _ := encodeLogEntry(LogEntry{Op: OpSetAliasBatch, AliasBatch: actions})
		if resp := m.Apply(&raft.Log{Data: e}); resp != nil {
			t.Fatalf("apply batch %+v returned %v", actions, resp)
		}
	}
	// Create two aliases in one atomic batch.
	applyBatch([]AliasAction{
		{Alias: "a", Canonical: "x"},
		{Alias: "b", Canonical: "y"},
	})
	if c, ok := m.AliasLookup("a"); !ok || c != "x" {
		t.Fatalf("AliasLookup(a) = (%q,%v), want (x,true)", c, ok)
	}
	if c, ok := m.AliasLookup("b"); !ok || c != "y" {
		t.Fatalf("AliasLookup(b) = (%q,%v), want (y,true)", c, ok)
	}
	// Atomic swap: delete a then re-create a→z in ONE batch. The final state is
	// a→z; no intermediate (a absent) is ever observable to another applier.
	applyBatch([]AliasAction{
		{Alias: "a", Delete: true},
		{Alias: "a", Canonical: "z"},
	})
	if c, ok := m.AliasLookup("a"); !ok || c != "z" {
		t.Fatalf("after swap AliasLookup(a) = (%q,%v), want (z,true)", c, ok)
	}
	// b is untouched by the swap batch.
	if c, ok := m.AliasLookup("b"); !ok || c != "y" {
		t.Fatalf("AliasLookup(b) after swap = (%q,%v), want (y,true)", c, ok)
	}
	// A delete removes the alias.
	applyBatch([]AliasAction{{Alias: "b", Delete: true}})
	if c, ok := m.AliasLookup("b"); ok {
		t.Fatalf("after delete AliasLookup(b) = (%q,%v), want miss", c, ok)
	}
	// State().Aliases must be a deep copy: mutating it must not affect the FSM.
	st := m.State()
	st.Aliases["a"] = "MUTATED"
	if c, ok := m.AliasLookup("a"); !ok || c != "z" {
		t.Fatalf("Aliases not deep-copied: FSM mutated to (%q,%v)", c, ok)
	}
}

// TestFSMSnapshotRestoreWithAliases mirrors TestMetaFSMSnapshotRestoreCatalogReshard:
// an FSM with aliases must survive snapshot+restore, and an OLD-format state (no
// Aliases field) must restore cleanly (nil → no aliases).
func TestFSMSnapshotRestoreWithAliases(t *testing.T) {
	src := NewMetaFSM()
	b, _ := encodeLogEntry(LogEntry{Op: OpSetAliasBatch, AliasBatch: []AliasAction{
		{Alias: "prod", Canonical: "default/coll_v2"},
		{Alias: "stage", Canonical: "default/coll_v1"},
	}})
	if resp := src.Apply(&raft.Log{Data: b}); resp != nil {
		t.Fatalf("apply: %v", resp)
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
	if c, ok := dst.AliasLookup("prod"); !ok || c != "default/coll_v2" {
		t.Fatalf("restored AliasLookup(prod) = (%q,%v), want (default/coll_v2,true)", c, ok)
	}
	if c, ok := dst.AliasLookup("stage"); !ok || c != "default/coll_v1" {
		t.Fatalf("restored AliasLookup(stage) = (%q,%v), want (default/coll_v1,true)", c, ok)
	}
	if !equalState(src.State(), dst.State()) {
		t.Fatalf("state mismatch after restore: src=%+v dst=%+v", src.State(), dst.State())
	}

	// OLD-format state with nil Aliases restores cleanly (no aliases).
	old := NewMetaFSM()
	oldCat, _ := encodeLogEntry(LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
	if resp := old.Apply(&raft.Log{Data: oldCat}); resp != nil {
		t.Fatalf("apply old catalog: %v", resp)
	}
	osnap, err := old.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	var obuf bytes.Buffer
	if err := osnap.Persist(noopSink{w: &obuf}); err != nil {
		t.Fatal(err)
	}
	odst := NewMetaFSM()
	if err := odst.Restore(io.NopCloser(&obuf)); err != nil {
		t.Fatal(err)
	}
	if odst.State().Aliases != nil {
		t.Fatalf("old-format restored Aliases = %v, want nil", odst.State().Aliases)
	}
	if c, ok := odst.AliasLookup("prod"); ok {
		t.Fatalf("old-format AliasLookup = (%q,%v), want miss", c, ok)
	}
}
