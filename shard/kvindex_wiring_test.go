// SPDX-License-Identifier: Apache-2.0

package shard

// Store-level wiring tests for the KV record index. What is proved here is what
// only a whole Store can show: that the index the WRITE path maintains is the
// same object the READ path is served from, and that a snapshot restore leaves
// the index describing the keyspace that actually came back.

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

const kvWiringIndexName = "by-rc"

// kvWiringRec encodes a one-field dynamic-mode record {rc: v}.
func kvWiringRec(v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}, //nolint:gosec // fixture reinterprets a small i64
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

func kvWiringDef(t *testing.T) kvindex.Def {
	t.Helper()
	d, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        kvWiringIndexName,
		KeyPrefix:   []byte("u:"),
		PayloadPath: "rc",
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom: %v", err)
	}
	return d
}

// kvWiringCandidates asks the index for the keys posted under rc == v.
func kvWiringCandidates(t *testing.T, idx *kvindex.Set, v int64) []string {
	t.Helper()
	d, ok := idx.Lookup(kvWiringIndexName)
	if !ok {
		t.Fatalf("index %q is not installed", kvWiringIndexName)
	}
	keys, err := idx.Candidates(kvindex.Selector{
		Def:    d,
		Op:     vtypes.FilterEq,
		Values: []vtypes.Value{vtypes.NewInt(v)},
	}, nil, false, 1<<20)
	if err != nil {
		t.Fatalf("Candidates(rc == %d): %v", v, err)
	}
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, string(k))
	}
	return out
}

// TestSingleSetPerStore is the regression test for the two-Set defect, and it
// is worth being precise about what that defect would look like, because it is
// invisible to the type system: a Store that built one index for its FSM and a
// second for its read-only dispatcher would compile, apply every write
// correctly, and answer every query EMPTY — writes would fill one Set while
// reads consulted the other.
//
// So the assertion is deliberately end-to-end: the write goes through the
// replicated apply path (Store.Call on a read-write op → Raft → fsm.tx) and the
// read comes back through the read-only path (Store.Call on a read-only op →
// handler(s.tx, args)), with a probe op standing in for kv_query, which lands
// later in the phase.
func TestSingleSetPerStore(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	// The probe: a READ-ONLY op that answers from whatever index the dispatcher
	// it is handed carries — the same seam kv_query will use.
	if err := reg.Register("__kv_probe__", ops.OpReadOnly, func(tx *ops.TxContext, args []byte) ([]byte, error) {
		idx := tx.KVIndex()
		if idx == nil {
			return nil, fmt.Errorf("the read-only dispatcher has no KV index")
		}
		d, ok := idx.Lookup(kvWiringIndexName)
		if !ok {
			return nil, fmt.Errorf("the read-only dispatcher's index has no definition %q", kvWiringIndexName)
		}
		keys, err := idx.Candidates(kvindex.Selector{
			Def:    d,
			Op:     vtypes.FilterEq,
			Values: []vtypes.Value{vtypes.NewInt(int64(args[0]))},
		}, nil, false, 1<<20)
		if err != nil {
			return nil, err
		}
		return bytes.Join(keys, []byte(",")), nil
	}); err != nil {
		t.Fatal(err)
	}

	// t.TempDir() is called BEFORE the Close cleanup is registered, so cleanup
	// LIFO runs Close first and the directory is never removed under a live raft
	// goroutine.
	cfg := DefaultConfig(t.TempDir(), "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	waitLeader(t, s)

	if s.KVIndex() == nil {
		t.Fatal("a Store built by New must have a KV index")
	}
	if s.KVIndex() != s.fsm.tx.KVIndex() {
		t.Fatal("the FSM's index is not the Store's: writes would be indexed into a Set no read ever consults")
	}
	if s.KVIndex() != s.tx.KVIndex() {
		t.Fatal("the read-only dispatcher's index is not the Store's")
	}

	s.KVIndex().Install([]kvindex.Def{kvWiringDef(t)})
	s.KVIndex().MarkReady(kvWiringIndexName)

	// WRITE through the replicated apply path.
	if _, err := s.Call("put", ops.EncodePutArgs([]byte("u:a"), kvWiringRec(5), 0)); err != nil {
		t.Fatalf("put through the apply path: %v", err)
	}
	// READ through the read-only path.
	got, err := s.Call("__kv_probe__", []byte{5})
	if err != nil {
		t.Fatalf("probe through the read-only path: %v", err)
	}
	if string(got) != "u:a" {
		t.Fatalf("the read-only path found %q under rc == 5, want \"u:a\" — "+
			"the write path and the read path are not sharing one index", got)
	}

	// And a delete, whose posting is dropped by the cache's onRemove hook, is
	// visible on the read path too.
	if _, err := s.Call("del", ops.EncodeKeyArgs([]byte("u:a"))); err != nil {
		t.Fatalf("del through the apply path: %v", err)
	}
	if got, err = s.Call("__kv_probe__", []byte{5}); err != nil || len(got) != 0 {
		t.Fatalf("after del the read-only path still finds %q (err %v)", got, err)
	}
}

// TestRestoreRebuildsIndex — the index is in no snapshot, no WAL and no log, so
// a restore leaves it describing the keyspace that was just replaced. Restore
// must rebuild it IN PLACE (restoreSnapshot refills the existing cache rather
// than installing a new one, so the Set to refill is the one already wired to
// that cache's onRemove hook).
func TestRestoreRebuildsIndex(t *testing.T) {
	// --- a source shard with 50 record keys, snapshotted -------------------
	srcDir := t.TempDir()
	srcCache, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = srcCache.Close() })
	srcVectors, err := vector.OpenCollectionStore(srcDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = srcVectors.Close() })

	want := make([]string, 0, 50)
	for i := 0; i < 50; i++ {
		k := fmt.Appendf(nil, "u:%02d", i)
		if err := srcCache.Put(k, kvWiringRec(7), 0); err != nil {
			t.Fatalf("seed put %d: %v", i, err)
		}
		want = append(want, string(k))
	}
	// One key outside the definition's prefix: it must survive the restore and
	// still not be posted.
	if err := srcCache.Put([]byte("zz:x"), kvWiringRec(7), 0); err != nil {
		t.Fatal(err)
	}
	data, err := serializeSnapshot(srcCache, srcVectors, 42, nil)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}

	// --- a destination FSM with its own cache and THE index of that cache --
	dstDir := t.TempDir()
	dstCache, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dstCache.Close() })
	dstVectors, err := vector.OpenCollectionStore(dstDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dstVectors.Close() })

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	idx := ops.NewKVIndexFor(dstCache)
	f := newFSM(dstCache, reg, false, dstVectors, idx)

	// Stale content the restore replaces, and a definition installed over it —
	// the state a node that had been serving queries would be in.
	if err := dstCache.Put([]byte("u:stale"), kvWiringRec(7), 0); err != nil {
		t.Fatal(err)
	}
	idx.Install([]kvindex.Def{kvWiringDef(t)})
	idx.Rebuild(ops.CacheWalker(dstCache))
	if got := kvWiringCandidates(t, idx, 7); len(got) != 1 || got[0] != "u:stale" {
		t.Fatalf("pre-restore postings = %v, want [u:stale]", got)
	}

	if err := f.Restore(io.NopCloser(bytes.NewReader(data))); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	if !idx.IsReady(kvWiringIndexName) {
		t.Fatal("the definition must be ready again once Restore's rebuild has walked the restored keyspace")
	}
	got := kvWiringCandidates(t, idx, 7)
	if len(got) != len(want) {
		t.Fatalf("post-restore postings = %d keys, want %d (%v)", len(got), len(want), got)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("post-restore postings = %v, want %v", got, want)
		}
	}
	// The pre-restore key is gone from the keyspace, so it must be gone from the
	// index; the out-of-prefix key is in the keyspace but not this definition's.
	for _, k := range got {
		if k == "u:stale" || k == "zz:x" {
			t.Fatalf("post-restore postings still contain %q", k)
		}
	}
	if _, err := dstCache.Get([]byte("zz:x")); err != nil {
		t.Fatalf("the out-of-prefix key did not survive the restore: %v", err)
	}
}

// TestStoreCacheWalkerVisitsEveryKey — CacheWalker is what the activation
// observer hands kvindex.Backfill, so it has to see every live entry across
// every cache shard.
func TestStoreCacheWalkerVisitsEveryKey(t *testing.T) {
	s := newSingleNodeStore(t)
	want := map[string]bool{}
	for i := 0; i < 64; i++ {
		k := fmt.Appendf(nil, "u:%02d", i)
		if err := s.Put(k, kvWiringRec(int64(i)), 0); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
		want[string(k)] = true
	}
	seen := map[string]bool{}
	s.CacheWalker()(func(key, _ []byte) bool {
		seen[string(key)] = true
		return true
	})
	for k := range want {
		if !seen[k] {
			t.Fatalf("CacheWalker never visited %q", k)
		}
	}
}

// --- the other two restore paths -----------------------------------------
//
// restoreSnapshot has three callers, and all three replace the whole keyspace
// the index describes: fsm.Restore (raft), pbSnapshotStore.InstallFSM (a PB
// catch-up transfer) and Store.RestoreSnapshot's PB branch (disaster recovery).
// A caller that refills the cache without rebuilding leaves the index holding
// postings for keys that are gone and NONE for the keys that arrived — the
// second half being the missing-row class. The raft path is covered by
// TestRestoreRebuildsIndex above; these two cover the rest.

// installIndexOn installs the fixture definition on a store's index and marks
// it ready, then returns the index.
func installIndexOn(t *testing.T, s *Store) *kvindex.Set {
	t.Helper()
	idx := s.KVIndex()
	if idx == nil {
		t.Fatal("the store has no KV index")
	}
	idx.Install([]kvindex.Def{kvWiringDef(t)})
	idx.MarkReady(kvWiringIndexName)
	return idx
}

// seedIndexedKeys writes n record keys under the definition's prefix.
func seedIndexedKeys(t *testing.T, s *Store, n int, rc int64) []string {
	t.Helper()
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		k := fmt.Appendf(nil, "u:%02d", i)
		if err := s.Put(k, kvWiringRec(rc), 0); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
		out = append(out, string(k))
	}
	return out
}

func TestPBSnapshotInstallRebuildsKVIndex(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	src, _ := pbFrontierStore(t, srcDir, 0)
	defer func() { _ = src.Close() }()
	dst, _ := pbFrontierStore(t, dstDir, 0)
	defer func() { _ = dst.Close() }()

	want := seedIndexedKeys(t, src, 20, 7)

	// The target is diverged AND indexed: its postings describe keys the source
	// has never heard of.
	dstIdx := installIndexOn(t, dst)
	seedIndexedKeys(t, dst, 3, 9)
	if got := kvWiringCandidates(t, dstIdx, 9); len(got) != 3 {
		t.Fatalf("pre-install postings = %v, want 3 ghosts", got)
	}

	blob, err := pbSnapStoreFor(src).SnapshotFSM(20)
	if err != nil {
		t.Fatalf("SnapshotFSM: %v", err)
	}
	target := pbSnapStoreFor(dst)
	if err := target.BeginInstall(20, 1); err != nil {
		t.Fatalf("BeginInstall: %v", err)
	}
	if err := target.InstallFSM(blob); err != nil {
		t.Fatalf("InstallFSM: %v", err)
	}
	if err := target.CommitInstall(20, 1); err != nil {
		t.Fatalf("CommitInstall: %v", err)
	}

	if got := kvWiringCandidates(t, dstIdx, 9); len(got) != 0 {
		t.Fatalf("after the install, the pre-install postings survive: %v", got)
	}
	got := kvWiringCandidates(t, dstIdx, 7)
	if len(got) != len(want) {
		t.Fatalf("after the install, rc == 7 finds %d keys, want %d — InstallFSM must "+
			"rebuild the index over the keyspace it just installed", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("post-install postings = %v, want %v", got, want)
		}
	}
}

func TestPBRestoreSnapshotRebuildsKVIndex(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	src, _ := pbFrontierStore(t, srcDir, 0)
	defer func() { _ = src.Close() }()
	dst, _ := pbFrontierStore(t, dstDir, 0)
	defer func() { _ = dst.Close() }()

	want := seedIndexedKeys(t, src, 20, 7)
	dstIdx := installIndexOn(t, dst)
	seedIndexedKeys(t, dst, 3, 9)

	blob, appliedIndex, err := src.BackupSnapshot(context.Background())
	if err != nil {
		t.Fatalf("BackupSnapshot: %v", err)
	}
	if err := dst.RestoreSnapshot(context.Background(), blob, appliedIndex); err != nil {
		t.Fatalf("RestoreSnapshot: %v", err)
	}

	if got := kvWiringCandidates(t, dstIdx, 9); len(got) != 0 {
		t.Fatalf("after the DR restore, the pre-restore postings survive: %v", got)
	}
	if got := kvWiringCandidates(t, dstIdx, 7); len(got) != len(want) {
		t.Fatalf("after the DR restore, rc == 7 finds %d keys, want %d — the PB branch of "+
			"RestoreSnapshot must rebuild the index", len(got), len(want))
	}
}
