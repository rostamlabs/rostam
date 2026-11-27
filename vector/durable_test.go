// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

const durableTestDim = 8

// durablePersistentCfg is what a caller that asks for the instant-restart mode
// sends: Persistent, SQ8, full text, no write-ahead log.
func durablePersistentCfg() Config {
	return Config{
		Dim: durableTestDim, Metric: Cosine, M: 16, EfConstruction: 64, EfSearch: 32, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true,
		FullText: &FullTextConfig{},
	}
}

func durableVec(seed int) []float32 {
	v := make([]float32, durableTestDim)
	for i := range v {
		v[i] = float32((seed*7+i*3)%11) + 1
	}
	normalize(v)
	return v
}

func openDurable(t *testing.T, dir string, names ...string) *CollectionStore {
	t.Helper()
	cs, err := OpenCollectionStoreWith(dir, StoreOptions{DurableCollections: names})
	if err != nil {
		t.Fatalf("open %s: %v", dir, err)
	}
	return cs
}

func manifestOf(t *testing.T, dir, col string) Config {
	t.Helper()
	cfg, err := readConfig(filepath.Join(dir, "vectors", DefaultTenant, col+".json"))
	if err != nil {
		t.Fatalf("read manifest of %s: %v", col, err)
	}
	return cfg
}

func walPathOf(dir, col string) string {
	return filepath.Join(dir, "vectors", DefaultTenant, col+".wal")
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return st.Size()
}

// pointsOf returns id -> content for every live point of the collection.
func pointsOf(t *testing.T, cs *CollectionStore, col string) map[uint64]string {
	t.Helper()
	docs, err := cs.ScrollDocs(col, Filter{}, 1000)
	if err != nil {
		t.Fatalf("scroll %s: %v", col, err)
	}
	out := make(map[uint64]string, len(docs))
	for _, d := range docs {
		out[d.ID] = d.Content
	}
	return out
}

func assertPoints(t *testing.T, got, want map[uint64]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("points = %v, want %v", got, want)
	}
	for id, c := range want {
		if g, ok := got[id]; !ok || g != c {
			t.Fatalf("points = %v, want %v", got, want)
		}
	}
}

func assertDurableManifest(t *testing.T, cfg Config) {
	t.Helper()
	if !cfg.WAL || cfg.Persistent || cfg.WALNoSync {
		t.Fatalf("durable manifest: WAL=%v Persistent=%v WALNoSync=%v, want WAL=true Persistent=false WALNoSync=false",
			cfg.WAL, cfg.Persistent, cfg.WALNoSync)
	}
}

// writeHistory upserts 1 and 2, overwrites 1 with new content and deletes 2.
// Every step is acknowledged before the next one starts.
func writeHistory(t *testing.T, cs *CollectionStore, col string) {
	t.Helper()
	if err := cs.Upsert(col, 1, durableVec(1), "first", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(col, 2, durableVec(2), "second", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(col, 1, durableVec(3), "first, rewritten", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := cs.DeleteCAS(col, 2, CASCond{}); err != nil {
		t.Fatal(err)
	}
}

var historyFinal = map[uint64]string{1: "first, rewritten"}

func copyTree(t *testing.T, src, dst string) {
	t.Helper()
	err := filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, 0o750)
		}
		in, err := os.Open(path) //nolint:gosec // test temp dir
		if err != nil {
			return err
		}
		defer func() { _ = in.Close() }()
		out, err := os.Create(target) //nolint:gosec // test temp dir
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, in); err != nil {
			_ = out.Close()
			return err
		}
		return out.Close()
	})
	if err != nil {
		t.Fatalf("copy %s -> %s: %v", src, dst, err)
	}
}

func TestDurableCreateForcesWAL(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "mcp_memory")
	defer func() { _ = cs.Close() }()
	for _, col := range []string{"mcp_memory", "other"} {
		if err := cs.CreateCollection(col, durablePersistentCfg()); err != nil {
			t.Fatalf("create %s: %v", col, err)
		}
	}
	assertDurableManifest(t, manifestOf(t, dir, "mcp_memory"))
	if other := manifestOf(t, dir, "other"); !other.Persistent || other.WAL {
		t.Fatalf("unlisted manifest: Persistent=%v WAL=%v, want Persistent=true WAL=false", other.Persistent, other.WAL)
	}
	if _, err := os.Stat(walPathOf(dir, "mcp_memory")); err != nil {
		t.Fatalf("listed collection has no log: %v", err)
	}
}

func TestDurableNamesAreCanonicalised(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "default/mcp_memory")
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	assertDurableManifest(t, manifestOf(t, dir, "mcp_memory"))

	if _, err := OpenCollectionStoreWith(t.TempDir(), StoreOptions{DurableCollections: []string{"a/b/c"}}); err == nil {
		t.Fatal("an invalid durable name opened the store")
	}
}

func TestDurableRestoreForcesWAL(t *testing.T) {
	src := t.TempDir()
	cs := openDurable(t, src)
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert("mcp_memory", 1, durableVec(1), "kept", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	c, _ := cs.Get("mcp_memory")
	var snap bytes.Buffer
	if err := c.Snapshot(&snap); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	dir := t.TempDir()
	ds := openDurable(t, dir, "mcp_memory")
	if err := ds.RestoreCollectionWithConfig("mcp_memory", durablePersistentCfg(), bytes.NewReader(snap.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	assertDurableManifest(t, manifestOf(t, dir, "mcp_memory"))

	// The restored points are acknowledged once the restore returns, so they
	// must be on disk without a Close: open a copy taken while ds is live.
	cp := t.TempDir()
	copyTree(t, dir, cp)
	_ = ds.Close()
	re := openDurable(t, cp, "mcp_memory")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "mcp_memory"), map[uint64]string{1: "kept"})
}

func TestDurableWritesSurviveWithoutClose(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "mcp_memory")
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	writeHistory(t, cs, "mcp_memory")

	cp := t.TempDir()
	copyTree(t, dir, cp)
	re := openDurable(t, cp, "mcp_memory")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "mcp_memory"), historyFinal)
}

func TestCheckpointDurableTruncatesOnlyListed(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "mcp_memory")
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	walOnly := durablePersistentCfg()
	walOnly.Persistent, walOnly.WAL = false, true
	if err := cs.CreateCollection("unlisted", walOnly); err != nil {
		t.Fatal(err)
	}
	writeHistory(t, cs, "mcp_memory")
	writeHistory(t, cs, "unlisted")
	unlistedBefore := fileSize(t, walPathOf(dir, "unlisted"))
	if unlistedBefore == 0 {
		t.Fatal("unlisted log is empty before the checkpoint")
	}

	if err := cs.CheckpointDurable(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if got := fileSize(t, walPathOf(dir, "mcp_memory")); got != 0 {
		t.Fatalf("listed log size after checkpoint = %d, want 0", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "vectors", DefaultTenant, "mcp_memory.snap")); err != nil {
		t.Fatalf("listed snapshot missing: %v", err)
	}
	if got := fileSize(t, walPathOf(dir, "unlisted")); got != unlistedBefore {
		t.Fatalf("unlisted log size = %d, want %d (untouched)", got, unlistedBefore)
	}
	_ = cs.Close()

	re := openDurable(t, dir, "mcp_memory")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "mcp_memory"), historyFinal)
	assertPoints(t, pointsOf(t, re, "unlisted"), historyFinal)
}

func TestCheckpointDurableEphemeralStoreWritesNothing(t *testing.T) {
	cs, err := OpenCollectionStoreWith("", StoreOptions{DurableCollections: []string{"mcp_memory"}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	writeHistory(t, cs, "mcp_memory")
	if err := cs.CheckpointDurable(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if _, err := os.Stat(filepath.Join(cs.dir, "vectors", DefaultTenant, "mcp_memory.snap")); !os.IsNotExist(err) {
		t.Fatalf("heap-mode checkpoint wrote a snapshot (stat err %v)", err)
	}
}

func TestDurableLargeLogCheckpointedAtOpen(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "mcp_memory")
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	writeHistory(t, cs, "mcp_memory")
	_ = cs.Close()
	if fileSize(t, walPathOf(dir, "mcp_memory")) == 0 {
		t.Fatal("log is empty before reopen")
	}

	old := durableCheckpointBytes
	durableCheckpointBytes = 1
	t.Cleanup(func() { durableCheckpointBytes = old })

	re := openDurable(t, dir, "mcp_memory")
	if got := fileSize(t, walPathOf(dir, "mcp_memory")); got != 0 {
		t.Fatalf("log size after open = %d, want 0 (checkpointed)", got)
	}
	assertPoints(t, pointsOf(t, re, "mcp_memory"), historyFinal)
	_ = re.Close()

	again := openDurable(t, dir, "mcp_memory")
	defer func() { _ = again.Close() }()
	assertPoints(t, pointsOf(t, again, "mcp_memory"), historyFinal)
}

func TestCheckpointCrashBeforeTruncateReplaysCorrectly(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "mcp_memory")
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	writeHistory(t, cs, "mcp_memory")
	saved, err := os.ReadFile(walPathOf(dir, "mcp_memory"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CheckpointDurable(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	_ = cs.Close()
	// A stop between the snapshot's rename and the log's truncate leaves the new
	// snapshot next to the whole old log.
	if err := os.WriteFile(walPathOf(dir, "mcp_memory"), saved, 0o600); err != nil {
		t.Fatal(err)
	}

	re := openDurable(t, dir, "mcp_memory")
	defer func() { _ = re.Close() }()
	assertPoints(t, pointsOf(t, re, "mcp_memory"), historyFinal)
}

func TestClusterStoreIgnoresDurable(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStoreWith(dir, StoreOptions{PersistentCluster: true, DurableCollections: []string{"mcp_memory"}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if cs.isDurable(DefaultTenant + "/mcp_memory") {
		t.Fatal("a cluster store reports a durable collection")
	}
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(walPathOf(dir, "mcp_memory")); !os.IsNotExist(err) {
		t.Fatalf("cluster store wrote a log (stat err %v)", err)
	}
	if err := cs.CheckpointDurable(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	entries, _ := os.ReadDir(filepath.Join(dir, "vectors", DefaultTenant))
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".wal" || filepath.Ext(e.Name()) == ".snap" {
			t.Fatalf("cluster store wrote %s", e.Name())
		}
	}
}

// TestCheckpointDurableUnderConcurrentWrites: every write acknowledged while
// checkpoints run is in the store after a reopen that copies the files while
// the store is still open.
func TestCheckpointDurableUnderConcurrentWrites(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir, "mcp_memory")
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("mcp_memory", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	const writers, each = 4, 40
	var wg sync.WaitGroup
	errc := make(chan error, writers)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				id := uint64(w*each + i + 1)
				if err := cs.Upsert("mcp_memory", id, durableVec(int(id)), "c", 0, nil, nil); err != nil {
					errc <- err
					return
				}
			}
		}(w)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	for stop := false; !stop; {
		select {
		case <-done:
			stop = true
		default:
		}
		if err := cs.CheckpointDurable(); err != nil {
			t.Fatalf("checkpoint: %v", err)
		}
	}
	close(errc)
	for err := range errc {
		t.Fatal(err)
	}
	cp := t.TempDir()
	copyTree(t, dir, cp)
	re := openDurable(t, cp, "mcp_memory")
	defer func() { _ = re.Close() }()
	if got := len(pointsOf(t, re, "mcp_memory")); got != writers*each {
		t.Fatalf("points after reopen = %d, want %d", got, writers*each)
	}
}
