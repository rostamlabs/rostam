// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/vector"
)

// testConfig is the plain-HNSW config used across the backup tests. The dense
// snapshot stream carries Dim/Metric/M/Ef*/Seed, so RestoreCollection
// reconstructs this geometry from the snapshot alone.
func testConfig() vector.Config {
	return vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
}

func newStore(t *testing.T) *vector.CollectionStore {
	t.Helper()
	store, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func mustCreate(t *testing.T, store *vector.CollectionStore, name string, ids ...uint64) {
	t.Helper()
	if err := store.CreateCollection(name, testConfig()); err != nil {
		t.Fatalf("create %q: %v", name, err)
	}
	c, ok := store.Acquire(name)
	if !ok {
		t.Fatalf("acquire %q after create", name)
	}
	defer c.Release()
	for _, id := range ids {
		// Deterministic distinct vectors: id along x.
		if err := c.Insert(id, []float32{float32(id), 0}, 0, nil, nil); err != nil {
			t.Fatalf("insert %d into %q: %v", id, name, err)
		}
	}
}

// TestBackupRestoreRoundTrip is the headline test: snapshot a populated
// collection to an in-memory object store, drop it, restore from the backup, and
// assert a search returns the SAME ids/scores as before the drop.
func TestBackupRestoreRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "docs", 1, 2, 3)

	c, _ := store.Acquire("docs")
	before, err := c.Search([]float32{1, 0}, 3)
	c.Release()
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != 3 {
		t.Fatalf("pre-backup search returned %d, want 3", len(before))
	}

	obj := objstore.NewMemStore()
	ts := time.Date(2026, 6, 23, 10, 0, 0, 0, time.UTC)
	results, err := Backup(ctx, store, obj, BackupOpts{Tenant: "acme", Timestamp: ts})
	if err != nil {
		t.Fatalf("backup: %v", err)
	}
	if len(results) != 1 || results[0].Err != nil {
		t.Fatalf("backup results = %+v, want 1 ok", results)
	}
	if results[0].Size <= 0 {
		t.Fatalf("backup size = %d, want > 0", results[0].Size)
	}

	if err := store.DropCollection("docs"); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Get("docs"); ok {
		t.Fatal("collection should be gone after drop")
	}

	// RestoreLatest = LatestKey + Restore (create-or-replace from the reader).
	if err := RestoreLatest(ctx, store, obj, "acme", "default/docs"); err != nil {
		t.Fatalf("restore latest: %v", err)
	}

	c2, ok := store.Get("docs")
	if !ok {
		t.Fatal("restored collection missing")
	}
	after, err := c2.Search([]float32{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) {
		t.Fatalf("post-restore search returned %d, want %d", len(after), len(before))
	}
	for i := range before {
		if after[i].ID != before[i].ID {
			t.Errorf("result[%d] id = %d, want %d", i, after[i].ID, before[i].ID)
		}
		if after[i].Score != before[i].Score {
			t.Errorf("result[%d] score = %v, want %v", i, after[i].Score, before[i].Score)
		}
	}
}

// TestRestoreByExplicitKey covers the Restore(key) path directly (not via
// RestoreLatest), using the key reported by Backup.
func TestRestoreByExplicitKey(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "docs", 1, 2)

	obj := objstore.NewMemStore()
	ts := time.Date(2026, 6, 23, 10, 0, 0, 0, time.UTC)
	results, err := Backup(ctx, store, obj, BackupOpts{Tenant: "acme", Timestamp: ts})
	if err != nil {
		t.Fatal(err)
	}
	key := results[0].Key

	if err := store.DropCollection("docs"); err != nil {
		t.Fatal(err)
	}
	if err := Restore(ctx, store, obj, "acme", "default/docs", key); err != nil {
		t.Fatalf("restore: %v", err)
	}
	c, ok := store.Get("docs")
	if !ok {
		t.Fatal("restored collection missing")
	}
	res, err := c.Search([]float32{1, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("search returned %d, want 2", len(res))
	}
}

// TestRetentionKeepsNewestN backs up the same collection at N+2 distinct
// timestamps with Retention=N and asserts exactly N objects remain — the N
// newest — and the older ones were deleted (the newest is NEVER pruned).
func TestRetentionKeepsNewestN(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "docs", 1)

	obj := objstore.NewMemStore()
	const keep = 2
	stamps := []time.Time{
		time.Date(2026, 6, 23, 1, 0, 0, 0, time.UTC),
		time.Date(2026, 6, 23, 2, 0, 0, 0, time.UTC),
		time.Date(2026, 6, 23, 3, 0, 0, 0, time.UTC),
		time.Date(2026, 6, 23, 4, 0, 0, 0, time.UTC), // N+2 = 4 runs
	}
	var wantKeys []string
	for _, ts := range stamps {
		results, err := Backup(ctx, store, obj, BackupOpts{Tenant: "acme", Timestamp: ts, Retention: keep})
		if err != nil {
			t.Fatalf("backup @%s: %v", ts, err)
		}
		wantKeys = append(wantKeys, results[0].Key)
	}

	prefix := collectionKeyPrefix("acme", "default/docs")
	infos, err := obj.List(ctx, prefix)
	if err != nil {
		t.Fatal(err)
	}
	// Each backup writes TWO objects (a <ts>.snap and its <ts>.cfg.json sibling),
	// so retention=keep leaves keep snapshots == 2*keep objects. Count snapshots.
	var snapInfos []objstore.ObjectInfo
	for _, in := range infos {
		if strings.HasSuffix(in.Key, ".snap") {
			snapInfos = append(snapInfos, in)
		}
	}
	if len(snapInfos) != keep {
		var got []string
		for _, in := range infos {
			got = append(got, in.Key)
		}
		t.Fatalf("retained %d snapshots, want %d (all objects: %v)", len(snapInfos), keep, got)
	}
	// Every retained snapshot must keep its sibling config object too.
	present := map[string]bool{}
	for _, in := range infos {
		present[in.Key] = true
	}
	for _, in := range snapInfos {
		if !present[cfgKeyFor(in.Key)] {
			t.Errorf("retained snapshot %q lost its sibling config %q", in.Key, cfgKeyFor(in.Key))
		}
	}
	// The kept keys must be the two newest (last two stamps).
	kept := map[string]bool{}
	for _, in := range snapInfos {
		kept[in.Key] = true
	}
	newest := wantKeys[len(wantKeys)-keep:]
	for _, k := range newest {
		if !kept[k] {
			t.Errorf("newest key %q was pruned (kept=%v)", k, kept)
		}
	}
	// The oldest two must be gone.
	for _, k := range wantKeys[:len(wantKeys)-keep] {
		if kept[k] {
			t.Errorf("stale key %q should have been pruned", k)
		}
	}
}

// failingPutStore wraps a MemStore and fails Put for any key containing failOn,
// to exercise per-collection backup isolation.
type failingPutStore struct {
	*objstore.MemStore
	failOn string
}

var errPutBoom = errors.New("put boom")

func (f *failingPutStore) Put(ctx context.Context, key string, r io.Reader, size int64) error {
	if strings.Contains(key, f.failOn) {
		return errPutBoom
	}
	return f.MemStore.Put(ctx, key, r, size)
}

// TestBackupResilience asserts that one collection's Put failure does not abort
// the others: the healthy collection still lands and the failure is reported in
// its own BackupResult.
func TestBackupResilience(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "good", 1, 2)
	mustCreate(t, store, "bad", 3, 4)

	// The key for canonical "default/bad" is acme/default%2Fbad/<ts>.snap, so the
	// substring "bad" matches it and never matches the "good" collection's key.
	obj := &failingPutStore{MemStore: objstore.NewMemStore(), failOn: "bad"}

	results, err := Backup(ctx, store, obj, BackupOpts{Tenant: "acme", Timestamp: time.Unix(0, 0).UTC()})
	if err == nil {
		t.Fatal("expected a joined error from the failed collection")
	}
	if !errors.Is(err, errPutBoom) {
		t.Errorf("joined error = %v, want it to wrap errPutBoom", err)
	}

	var good, bad *BackupResult
	for i := range results {
		switch results[i].Collection {
		case "default/good":
			good = &results[i]
		case "default/bad":
			bad = &results[i]
		}
	}
	if good == nil || bad == nil {
		t.Fatalf("missing results: %+v", results)
	}
	if good.Err != nil {
		t.Errorf("good collection failed: %v", good.Err)
	}
	if good.Key == "" || good.Size <= 0 {
		t.Errorf("good result not populated: %+v", good)
	}
	if bad.Err == nil {
		t.Error("bad collection should report an error")
	}
	// The good snapshot must actually be in the store; the bad one must not.
	if _, gerr := obj.Get(ctx, good.Key); gerr != nil {
		t.Errorf("good snapshot not stored: %v", gerr)
	}
}

// TestBackupRefusesExistingSnapshot pins the write-once rule: a second run at
// a timestamp whose snapshot already exists must fail with ErrSnapshotExists
// and leave BOTH existing objects (snapshot and sibling config) byte-identical,
// even though the collection's config changed in between. An orphan config with
// no snapshot (a run interrupted between the two Puts) must NOT block a re-run.
func TestBackupRefusesExistingSnapshot(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "c", 1, 2)
	obj := objstore.NewMemStore()
	ts := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	opts := BackupOpts{Tenant: "acme", Timestamp: ts}

	first, err := Backup(ctx, store, obj, opts)
	if err != nil {
		t.Fatalf("first backup: %v", err)
	}
	key := first[0].Key
	snapBefore := mustGet(t, obj, key)
	cfgBefore := mustGet(t, obj, cfgKeyFor(key))

	// Change what a re-run would write, so a silent overwrite would be visible.
	if err := store.DropCollection("c"); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateCollection("c", vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.Cosine}); err != nil {
		t.Fatal(err)
	}

	second, err := Backup(ctx, store, obj, opts)
	if err == nil {
		t.Fatal("re-run at an existing timestamp must fail")
	}
	if !errors.Is(err, ErrSnapshotExists) || !errors.Is(second[0].Err, ErrSnapshotExists) {
		t.Fatalf("err = %v / %v, want ErrSnapshotExists", err, second[0].Err)
	}
	if got := mustGet(t, obj, key); got != snapBefore {
		t.Error("existing snapshot was modified by the refused re-run")
	}
	if got := mustGet(t, obj, cfgKeyFor(key)); got != cfgBefore {
		t.Error("existing sibling config was modified by the refused re-run")
	}

	// Orphan config only (no snapshot) must not block: simulate a run that died
	// between the config Put and the snapshot Put, then re-run at that timestamp.
	if err := obj.Delete(ctx, key); err != nil {
		t.Fatal(err)
	}
	third, err := Backup(ctx, store, obj, opts)
	if err != nil {
		t.Fatalf("re-run over an orphan config must succeed: %v", err)
	}
	if third[0].Key != key {
		t.Fatalf("key = %q, want %q", third[0].Key, key)
	}
	if got := mustGet(t, obj, cfgKeyFor(key)); got == cfgBefore {
		t.Error("orphan config was not replaced by the re-run")
	}
}

// mustGet reads the whole object at key, failing the test on any error.
func mustGet(t *testing.T, obj objstore.ObjectStore, key string) string {
	t.Helper()
	rc, err := obj.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("get %q: %v", key, err)
	}
	defer rc.Close()
	b, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("read %q: %v", key, err)
	}
	return string(b)
}

// TestLatestKeyPicksNewest checks LatestKey returns the most recent timestamp's
// key and ErrNotFound when nothing exists.
func TestLatestKeyPicksNewest(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "docs", 1)

	obj := objstore.NewMemStore()
	if _, err := LatestKey(ctx, obj, "acme", "default/docs"); !errors.Is(err, objstore.ErrNotFound) {
		t.Fatalf("LatestKey on empty store = %v, want ErrNotFound", err)
	}

	stamps := []time.Time{
		time.Date(2026, 6, 23, 5, 0, 0, 0, time.UTC),
		time.Date(2026, 6, 23, 9, 0, 0, 0, time.UTC), // newest
		time.Date(2026, 6, 23, 7, 0, 0, 0, time.UTC),
	}
	var newestKey string
	for _, ts := range stamps {
		results, err := Backup(ctx, store, obj, BackupOpts{Tenant: "acme", Timestamp: ts})
		if err != nil {
			t.Fatal(err)
		}
		if ts.Hour() == 9 {
			newestKey = results[0].Key
		}
	}
	got, err := LatestKey(ctx, obj, "acme", "default/docs")
	if err != nil {
		t.Fatal(err)
	}
	if got != newestKey {
		t.Errorf("LatestKey = %q, want %q", got, newestKey)
	}
}

// TestDeterministicKey asserts the same Timestamp yields the same object key.
func TestDeterministicKey(t *testing.T) {
	ts := time.Date(2026, 6, 23, 12, 30, 45, 0, time.UTC)
	k1 := snapshotKey("acme", "default/docs", ts)
	k2 := snapshotKey("acme", "default/docs", ts)
	if k1 != k2 {
		t.Fatalf("keys differ for same timestamp: %q vs %q", k1, k2)
	}
	// And the key embeds the escaped collection name + RFC3339 stamp + .snap.
	if !strings.HasSuffix(k1, ".snap") || !strings.Contains(k1, "2026-06-23T12:30:45Z") {
		t.Errorf("unexpected key shape: %q", k1)
	}
}
