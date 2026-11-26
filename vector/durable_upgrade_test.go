// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	upgradeCol = "mem"
	upgradeDim = 32
)

// legacyPoint is what a point of the legacy store held when it was flushed.
type legacyPoint struct {
	vec     []float32
	meta    Metadata
	version uint64
}

func legacyCfg() Config {
	return Config{
		Dim: upgradeDim, Metric: Cosine, M: 16, EfConstruction: 64, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true,
		FullText: &FullTextConfig{},
	}
}

// uniqueWord spells id in consonants, so every point's content carries one word
// no other point has, and no stemmer folds two of them together.
func uniqueWord(id uint64) string {
	const letters = "bcdfghjklmnpqrtvwxz"
	var b strings.Builder
	b.WriteString("zq")
	for {
		b.WriteByte(letters[id%uint64(len(letters))])
		id /= uint64(len(letters))
		if id == 0 {
			break
		}
	}
	return b.String()
}

// writeLegacyStore writes, with an unlisted store, the store today's code leaves:
// a Persistent SQ8 full-text collection holding 200 upserts with content and
// metadata, 5 of them deleted, flushed and cleanly closed. It returns the points
// the store held at the flush.
func writeLegacyStore(t *testing.T, dir string) map[uint64]legacyPoint {
	t.Helper()
	cs := openDurable(t, dir)
	if err := cs.CreateCollection(upgradeCol, legacyCfg()); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic test data
	for i := 0; i < 200; i++ {
		id := uint64(i % 180) // the last 20 overwrite earlier ids
		vec := make([]float32, upgradeDim)
		for d := range vec {
			vec[d] = rng.Float32()*2 - 1
		}
		content := fmt.Sprintf("memory %d about %s", i, uniqueWord(id))
		meta := Metadata{"n": NewInt(int64(i)), "kind": NewString(fmt.Sprintf("k%d", i%4))}
		if err := cs.Upsert(upgradeCol, id, vec, content, 0, meta, nil); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []uint64{3, 17, 50, 101, 179} {
		if _, err := cs.DeleteCAS(upgradeCol, id, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	want := make(map[uint64]legacyPoint)
	for id := uint64(0); id < 180; id++ {
		vec, meta, _, _, version, ok, err := cs.GetPointVersion(upgradeCol, id)
		if err != nil {
			t.Fatal(err)
		}
		if ok {
			want[id] = legacyPoint{vec: vec, meta: meta, version: version}
		}
	}
	if len(want) != 175 {
		t.Fatalf("legacy store holds %d points, want 175", len(want))
	}
	if err := cs.FlushPersistent(); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	return want
}

func colFile(dir, ext string) string {
	return filepath.Join(dir, "vectors", DefaultTenant, upgradeCol+ext)
}

func assertGone(t *testing.T, paths ...string) {
	t.Helper()
	for _, p := range paths {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("%s still exists (stat err %v)", p, err)
		}
	}
}

func mappedFiles(dir string) []string {
	return []string{colFile(dir, ".vecs"), colFile(dir, ".graph"), colFile(dir, ".meta")}
}

// assertLegacyPoints checks every wanted point is present with the same metadata
// (content included) and version and, when checkVecs, the same vector, and that
// the collection holds nothing else.
func assertLegacyPoints(t *testing.T, cs *CollectionStore, want map[uint64]legacyPoint, checkVecs bool) {
	t.Helper()
	for id, w := range want {
		vec, meta, _, _, version, ok, err := cs.GetPointVersion(upgradeCol, id)
		if err != nil || !ok {
			t.Fatalf("point %d: ok=%v err=%v", id, ok, err)
		}
		if !reflect.DeepEqual(meta, w.meta) {
			t.Fatalf("point %d metadata = %v, want %v", id, meta, w.meta)
		}
		if version != w.version {
			t.Fatalf("point %d version = %d, want %d", id, version, w.version)
		}
		if checkVecs {
			for d := range w.vec {
				if math.Abs(float64(vec[d]-w.vec[d])) > 1e-6 {
					t.Fatalf("point %d vector[%d] = %v, want %v", id, d, vec[d], w.vec[d])
				}
			}
		}
	}
	if n := len(pointsOf(t, cs, upgradeCol)); n != len(want) {
		t.Fatalf("collection holds %d points, want %d", n, len(want))
	}
}

// assertSearchable checks full-text and vector search over the upgraded points.
func assertSearchable(t *testing.T, cs *CollectionStore, want map[uint64]legacyPoint) {
	t.Helper()
	hits := 0
	for id, w := range want {
		docs, err := cs.SearchText(upgradeCol, uniqueWord(id), 1, Filter{})
		if err != nil {
			t.Fatal(err)
		}
		if len(docs) != 1 || docs[0].ID != id {
			t.Fatalf("SearchText(%q) = %v, want id %d", uniqueWord(id), docs, id)
		}
		res, err := cs.SearchFiltered(upgradeCol, w.vec, 1, Filter{})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 1 && res[0].ID == id {
			hits++
		}
	}
	if float64(hits) < 0.99*float64(len(want)) {
		t.Fatalf("vector search found the point itself first for %d of %d", hits, len(want))
	}
}

func captureLogs(t *testing.T) *syncBuffer {
	t.Helper()
	buf := &syncBuffer{}
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(buf, nil)))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return buf
}

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func setUpgradeHook(t *testing.T, fn func(step string) error) {
	t.Helper()
	upgradeStepHook = fn
	t.Cleanup(func() { upgradeStepHook = nil })
}

func fileHash(t *testing.T, path string) [32]byte {
	t.Helper()
	data, err := os.ReadFile(path) //nolint:gosec // test temp dir
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return sha256.Sum256(data)
}

func TestUpgradeKeepsEveryPoint(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	// A log left from an earlier life must not be replayed over the upgrade.
	if err := os.WriteFile(colFile(dir, ".wal"), []byte("stale log bytes"), 0o600); err != nil {
		t.Fatal(err)
	}

	cs := openDurable(t, dir, upgradeCol)
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	assertLegacyPoints(t, cs, want, true)
	assertSearchable(t, cs, want)
	assertGone(t, mappedFiles(dir)...)
	if got := fileSize(t, colFile(dir, ".wal")); got != 0 {
		t.Fatalf("log after upgrade is %d bytes, want a fresh empty log", got)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	assertLegacyPoints(t, cs, want, true)
	assertSearchable(t, cs, want)
}

func TestUpgradeAfterUnflushedWrites(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	cs := openDurable(t, dir)
	for i := uint64(0); i < 50; i++ {
		id := 1000 + i
		if err := cs.Upsert(upgradeCol, id, durableVecDim(int(id)), "late "+uniqueWord(id), 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := cs.Close(); err != nil { // no flush: the sidecar still describes the earlier state
		t.Fatal(err)
	}

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	assertLegacyPoints(t, cs, want, false)
	for i := uint64(0); i < 50; i++ {
		if _, _, _, _, ok, err := cs.GetPoint(upgradeCol, 1000+i); err != nil || ok {
			t.Fatalf("unflushed point %d: ok=%v err=%v, want absent", 1000+i, ok, err)
		}
	}
}

func durableVecDim(seed int) []float32 {
	v := make([]float32, upgradeDim)
	for i := range v {
		v[i] = float32((seed*7+i*3)%11) + 1
	}
	return v
}

var upgradeSteps = []string{"loaded", "snapshot", "log-removed", "manifest"}

func TestUpgradeCrashAtEachStep(t *testing.T) {
	for _, step := range upgradeSteps {
		t.Run(step, func(t *testing.T) {
			dir := t.TempDir()
			want := writeLegacyStore(t, dir)
			crashed := filepath.Join(t.TempDir(), "crashed")
			seen := false
			setUpgradeHook(t, func(s string) error {
				if s == step {
					copyTree(t, dir, crashed)
					seen = true
				}
				return nil
			})
			cs := openDurable(t, dir, upgradeCol)
			_ = cs.Close()
			if !seen {
				t.Fatalf("step %q never ran", step)
			}
			upgradeStepHook = nil

			cs = openDurable(t, crashed, upgradeCol)
			defer func() { _ = cs.Close() }()
			assertDurableManifest(t, manifestOf(t, crashed, upgradeCol))
			assertLegacyPoints(t, cs, want, true)
			assertGone(t, mappedFiles(crashed)...)
		})
	}
}

func TestUpgradeFailureKeepsOldFormAndWarns(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	logs := captureLogs(t)
	blocker := colFile(dir, ".snap.tmp")
	if err := os.Mkdir(blocker, 0o750); err != nil {
		t.Fatal(err)
	}

	cs := openDurable(t, dir, upgradeCol)
	if cfg := manifestOf(t, dir, upgradeCol); !cfg.Persistent || cfg.WAL {
		t.Fatalf("manifest after a failed upgrade: Persistent=%v WAL=%v, want the old form", cfg.Persistent, cfg.WAL)
	}
	assertLegacyPoints(t, cs, want, true)
	if !strings.Contains(logs.String(), "not crash-safe") || !strings.Contains(logs.String(), upgradeCol) {
		t.Fatalf("log = %q, want a not crash-safe warning naming the collection", logs.String())
	}
	// The old form still takes writes and flushes them on a clean shutdown.
	if err := cs.Upsert(upgradeCol, 500, durableVecDim(500), "after failure "+uniqueWord(500), 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.FlushPersistent(); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	if err := os.Remove(blocker); err != nil {
		t.Fatal(err)
	}
	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	if _, meta, _, _, ok, err := cs.GetPoint(upgradeCol, 500); err != nil || !ok || meta[contentField].Str != "after failure "+uniqueWord(500) {
		t.Fatalf("point written after the failed upgrade: ok=%v err=%v meta=%v", ok, err, meta)
	}
	delete(want, 500)
	for id, w := range want {
		_, meta, _, _, ok, err := cs.GetPoint(upgradeCol, id)
		if err != nil || !ok || !reflect.DeepEqual(meta, w.meta) {
			t.Fatalf("point %d after the retried upgrade: ok=%v err=%v", id, ok, err)
		}
	}
}

// A failure injected after any step before the commit point leaves the old
// files byte-identical and the old form working; one after the manifest step is
// past the commit point and the collection loads converted.
func TestUpgradeStepFailure(t *testing.T) {
	for _, step := range upgradeSteps {
		t.Run(step, func(t *testing.T) {
			dir := t.TempDir()
			want := writeLegacyStore(t, dir)
			before := map[string][32]byte{}
			for _, p := range append(mappedFiles(dir), colFile(dir, ".json")) {
				before[p] = fileHash(t, p)
			}
			logs := captureLogs(t)
			setUpgradeHook(t, func(s string) error {
				if s == step {
					return errors.New("injected")
				}
				return nil
			})
			cs := openDurable(t, dir, upgradeCol)
			defer func() { _ = cs.Close() }()

			if step == "manifest" {
				assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
				assertLegacyPoints(t, cs, want, true)
				assertGone(t, mappedFiles(dir)...)
				return
			}
			if cfg := manifestOf(t, dir, upgradeCol); !cfg.Persistent || cfg.WAL {
				t.Fatalf("manifest: Persistent=%v WAL=%v, want the old form", cfg.Persistent, cfg.WAL)
			}
			for p, h := range before {
				if fileHash(t, p) != h {
					t.Fatalf("%s changed by a failed upgrade", p)
				}
			}
			if !strings.Contains(logs.String(), "not crash-safe") || !strings.Contains(logs.String(), "injected") {
				t.Fatalf("log = %q, want a not crash-safe warning with the error", logs.String())
			}
			assertGone(t, colFile(dir, ".snap"))
			assertLegacyPoints(t, cs, want, true)
			assertSearchable(t, cs, want)
		})
	}
}

func TestUpgradeNeverFlushedCollection(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	if err := cs.CreateCollection(upgradeCol, legacyCfg()); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(upgradeCol, 1, durableVecDim(1), "never flushed", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	if got := pointsOf(t, cs, upgradeCol); len(got) != 0 {
		t.Fatalf("points = %v, want none", got)
	}
	assertGone(t, mappedFiles(dir)...)
	if err := cs.Upsert(upgradeCol, 2, durableVecDim(2), "after", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
}

func TestLeftoverMappedFilesRemoved(t *testing.T) {
	dir := t.TempDir()
	writeLegacyStore(t, dir)
	cs := openDurable(t, dir, upgradeCol)
	_ = cs.Close()
	for _, p := range mappedFiles(dir) {
		if err := os.WriteFile(p, []byte("left over"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertGone(t, mappedFiles(dir)...)
}

// An unlisted WAL collection keeps any such files: only listed ones are cleaned.
func TestLeftoverMappedFilesKeptWhenUnlisted(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	cfg := legacyCfg()
	cfg.Persistent, cfg.WAL = false, true
	if err := cs.CreateCollection(upgradeCol, cfg); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()
	meta := colFile(dir, ".meta")
	if err := os.WriteFile(meta, []byte("left over"), 0o600); err != nil {
		t.Fatal(err)
	}
	cs = openDurable(t, dir)
	defer func() { _ = cs.Close() }()
	if _, err := os.Stat(meta); err != nil {
		t.Fatalf("unlisted collection's file removed: %v", err)
	}
}

func TestDurableClearsWALNoSync(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	cfg := legacyCfg()
	cfg.Persistent, cfg.WAL, cfg.WALNoSync = false, true, true
	if err := cs.CreateCollection(upgradeCol, cfg); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(upgradeCol, 1, durableVecDim(1), "one", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs = openDurable(t, dir, upgradeCol)
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	assertWALSyncs(t, cs)
	assertPoints(t, pointsOf(t, cs, upgradeCol), map[uint64]string{1: "one"})
	_ = cs.Close()
}

func TestDurableClearsWALNoSyncEvenIfManifestRewriteFails(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	cfg := legacyCfg()
	cfg.Persistent, cfg.WAL, cfg.WALNoSync = false, true, true
	if err := cs.CreateCollection(upgradeCol, cfg); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()
	logs := captureLogs(t)
	if err := os.Mkdir(colFile(dir, ".json.tmp"), 0o750); err != nil {
		t.Fatal(err)
	}

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	if !manifestOf(t, dir, upgradeCol).WALNoSync {
		t.Fatal("manifest rewritten through a blocked temp file")
	}
	assertWALSyncs(t, cs)
	if !strings.Contains(logs.String(), upgradeCol) {
		t.Fatalf("log = %q, want a warning naming the collection", logs.String())
	}
}

func assertWALSyncs(t *testing.T, cs *CollectionStore) {
	t.Helper()
	c, ok := cs.Acquire(upgradeCol)
	if !ok {
		t.Fatal("collection missing")
	}
	defer c.Release()
	if c.wal == nil || c.wal.noSync || c.cfg.WALNoSync {
		t.Fatalf("log of a listed collection does not sync (wal=%v)", c.wal != nil)
	}
}

// Only an HNSW collection is converted; another index type stays as it was
// and warns.
func TestUpgradeSkipsOtherIndexTypes(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	cfg := Config{
		Dim: upgradeDim, Metric: Cosine, Seed: 7, IndexType: IndexVamana,
		VamanaR: 16, VamanaL: 32, VamanaAlpha: 1.2, Quant: QuantSQ8, Persistent: true,
	}
	if err := cs.CreateCollection(upgradeCol, cfg); err != nil {
		t.Fatal(err)
	}
	if err := cs.Insert(upgradeCol, 1, durableVecDim(1), 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.FlushPersistent(); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()
	logs := captureLogs(t)

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	if got := manifestOf(t, dir, upgradeCol); !got.Persistent || got.WAL {
		t.Fatalf("manifest: Persistent=%v WAL=%v, want unchanged", got.Persistent, got.WAL)
	}
	if !strings.Contains(logs.String(), "not crash-safe") {
		t.Fatalf("log = %q, want a not crash-safe warning", logs.String())
	}
	if _, _, _, _, ok, err := cs.GetPoint(upgradeCol, 1); err != nil || !ok {
		t.Fatalf("point 1: ok=%v err=%v", ok, err)
	}
}

// An unlisted Persistent collection is never touched.
func TestUnlistedPersistentNotUpgraded(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	cs := openDurable(t, dir, "other")
	defer func() { _ = cs.Close() }()
	if cfg := manifestOf(t, dir, upgradeCol); !cfg.Persistent || cfg.WAL {
		t.Fatalf("unlisted manifest changed: Persistent=%v WAL=%v", cfg.Persistent, cfg.WAL)
	}
	for _, p := range mappedFiles(dir) {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("unlisted collection lost %s: %v", p, err)
		}
	}
	assertLegacyPoints(t, cs, want, true)
}

// The upgrade carries a point's TTL, sparse vector and per-key deadlines, and
// drops a point that has expired.
func TestUpgradeCarriesTTLSparseAndKeyDeadlines(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	if err := cs.CreateCollection(upgradeCol, legacyCfg()); err != nil {
		t.Fatal(err)
	}
	c, _ := cs.Acquire(upgradeCol)
	sparse := &SparseVector{Indices: []uint32{3, 9}, Values: []float32{0.5, 1.5}}
	meta := Metadata{"keep": NewString("k"), "brief": NewString("b")}
	if err := c.InsertKeyTTL(1, durableVecDim(1), time.Hour, meta, sparse, map[string]int64{"brief": 3_600_000}); err != nil {
		t.Fatal(err)
	}
	if err := c.Insert(2, durableVecDim(2), time.Millisecond, nil, nil); err != nil {
		t.Fatal(err)
	}
	h := c.idx.(*hnsw)
	slot, _ := h.arena.Slot(1)
	wantKeyExp := maps.Clone(h.arena.KeyExpires(slot))
	c.Release()
	time.Sleep(20 * time.Millisecond)
	if err := cs.FlushPersistent(); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	_, gotMeta, ttl, gotSparse, ok, err := cs.GetPoint(upgradeCol, 1)
	if err != nil || !ok {
		t.Fatalf("point 1: ok=%v err=%v", ok, err)
	}
	if ttl <= 59*time.Minute || ttl > time.Hour {
		t.Fatalf("point 1 ttl = %v, want just under an hour", ttl)
	}
	if !reflect.DeepEqual(gotMeta, meta) || !reflect.DeepEqual(gotSparse, sparse) {
		t.Fatalf("point 1 meta=%v sparse=%v, want %v %v", gotMeta, gotSparse, meta, sparse)
	}
	c, _ = cs.Acquire(upgradeCol)
	defer c.Release()
	h = c.idx.(*hnsw)
	slot, _ = h.arena.Slot(1)
	if got := h.arena.KeyExpires(slot); !reflect.DeepEqual(got, wantKeyExp) || len(got) != 1 {
		t.Fatalf("per-key deadlines = %v, want %v", got, wantKeyExp)
	}
	if _, _, _, _, ok, _ := cs.GetPoint(upgradeCol, 2); ok {
		t.Fatal("expired point 2 carried over")
	}
}

// Stored data that does not hold together fails the copy with an error rather
// than a panic, so the upgrade fails and the store still opens.
func TestCopyLivePointsRefusesInconsistentData(t *testing.T) {
	build := func(t *testing.T) (*Collection, *hnsw) {
		t.Helper()
		cfg := legacyCfg()
		cfg.Persistent = false
		c, err := NewCollection(upgradeCol, cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Close() })
		for id := uint64(1); id <= 3; id++ {
			if err := c.Insert(id, durableVecDim(int(id)), 0, Metadata{"n": NewInt(int64(id))}, nil); err != nil {
				t.Fatal(err)
			}
		}
		return c, c.idx.(*hnsw)
	}
	dst := func(t *testing.T) *Collection {
		t.Helper()
		cfg := legacyCfg()
		cfg.Persistent = false
		c, err := NewCollection(upgradeCol, cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Close() })
		return c
	}

	t.Run("slot beyond the stored data", func(t *testing.T) {
		_, h := build(t)
		h.arena.idMap[99] = 1 << 20
		if err := copyLivePoints(h, dst(t)); err == nil {
			t.Fatal("copy succeeded over a point naming a missing slot")
		}
	})
	t.Run("panic while reading", func(t *testing.T) {
		_, h := build(t)
		h.arena.metadata = h.arena.metadata[:1] // slots 1 and 2 now read out of range
		err := copyLivePoints(h, dst(t))
		if err == nil || !strings.Contains(err.Error(), "reading the stored points") {
			t.Fatalf("err = %v, want the panic returned as an error", err)
		}
	})
}

// A failure to load the new form after the commit point puts the old manifest
// back: the open succeeds in the old form, warns, and the next open retries.
func TestUpgradeCommittedButUnloadableRollsBack(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	logs := captureLogs(t)
	setUpgradeHook(t, func(s string) error {
		if s == "manifest" {
			return os.WriteFile(colFile(dir, ".snap"), []byte("not a snapshot"), 0o600)
		}
		return nil
	})
	cs := openDurable(t, dir, upgradeCol)
	if cfg := manifestOf(t, dir, upgradeCol); !cfg.Persistent || cfg.WAL {
		t.Fatalf("manifest: Persistent=%v WAL=%v, want the old form put back", cfg.Persistent, cfg.WAL)
	}
	assertLegacyPoints(t, cs, want, true)
	assertSearchable(t, cs, want)
	assertGone(t, colFile(dir, ".snap"))
	if !strings.Contains(logs.String(), "not crash-safe") {
		t.Fatalf("log = %q, want a not crash-safe warning", logs.String())
	}
	if err := cs.FlushPersistent(); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	upgradeStepHook = nil
	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	assertLegacyPoints(t, cs, want, true)
}

// If putting the old manifest back fails too, the store still opens in the old
// form and the warning names that case.
func TestUpgradeRollbackFailureStillOpens(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	logs := captureLogs(t)
	setUpgradeHook(t, func(s string) error {
		if s == "manifest" {
			if err := os.Mkdir(colFile(dir, ".json.tmp"), 0o750); err != nil {
				return err
			}
			return os.WriteFile(colFile(dir, ".snap"), []byte("not a snapshot"), 0o600)
		}
		return nil
	})
	cs := openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertLegacyPoints(t, cs, want, true)
	if !strings.Contains(logs.String(), "could not be put back") {
		t.Fatalf("log = %q, want the rollback failure named", logs.String())
	}
}

// The new form is proven to load before the commit point: a snapshot that does
// not restore fails the upgrade the ordinary way.
func TestUpgradeUnloadableSnapshotFailsBeforeCommit(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	logs := captureLogs(t)
	setUpgradeHook(t, func(s string) error {
		if s == "snapshot" {
			return os.WriteFile(colFile(dir, ".snap"), []byte("not a snapshot"), 0o600)
		}
		return nil
	})
	cs := openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	if cfg := manifestOf(t, dir, upgradeCol); !cfg.Persistent || cfg.WAL {
		t.Fatalf("manifest: Persistent=%v WAL=%v, want the old form", cfg.Persistent, cfg.WAL)
	}
	assertGone(t, colFile(dir, ".snap"))
	assertLegacyPoints(t, cs, want, true)
	if !strings.Contains(logs.String(), "not crash-safe") {
		t.Fatalf("log = %q, want a not crash-safe warning", logs.String())
	}
}

// writeConfig reported an error and the manifest cannot be read back: the old
// manifest is written again, so the state is certain, and the old form stays.
func TestUpgradeUnreadableManifestPutBack(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	logs := captureLogs(t)
	setUpgradeHook(t, func(s string) error {
		if s == "manifest" {
			if err := os.WriteFile(colFile(dir, ".json"), []byte("{torn"), 0o600); err != nil {
				return err
			}
			return errors.New("injected")
		}
		return nil
	})
	cs := openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	if cfg := manifestOf(t, dir, upgradeCol); !cfg.Persistent || cfg.WAL {
		t.Fatalf("manifest: Persistent=%v WAL=%v, want the old form written back", cfg.Persistent, cfg.WAL)
	}
	assertGone(t, colFile(dir, ".snap"))
	assertLegacyPoints(t, cs, want, true)
	if !strings.Contains(logs.String(), "not crash-safe") {
		t.Fatalf("log = %q, want a not crash-safe warning", logs.String())
	}
}

// writeConfig reported an error but its rename landed. The directory sync that
// makes the rename durable failed, so the old files are not removed until it
// succeeds: a later open still has them.
func TestUpgradeUnsyncedManifestKeepsOldFiles(t *testing.T) {
	dir := t.TempDir()
	want := writeLegacyStore(t, dir)
	prev := syncDirf
	t.Cleanup(func() { syncDirf = prev })
	setUpgradeHook(t, func(s string) error {
		if s == "manifest" {
			syncDirf = func(string) error { return errors.New("injected fsync") }
			return errors.New("injected fsync")
		}
		return nil
	})
	cs := openDurable(t, dir, upgradeCol)
	assertDurableManifest(t, manifestOf(t, dir, upgradeCol))
	assertLegacyPoints(t, cs, want, true)
	for _, p := range mappedFiles(dir) {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s removed although the manifest rename was not made durable: %v", p, err)
		}
	}
	_ = cs.Close()

	syncDirf = prev
	upgradeStepHook = nil
	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertLegacyPoints(t, cs, want, true)
	assertGone(t, mappedFiles(dir)...)
}

// A Persistent manifest with no sidecar beside a snapshot is not a state the
// upgrade produces, but if it is ever met the snapshot is kept aside, never
// overwritten by the empty collection the old form opens as.
func TestUpgradeKeepsOrphanSnapshot(t *testing.T) {
	dir := t.TempDir()
	writeLegacyStore(t, dir)
	cs := openDurable(t, dir, upgradeCol)
	_ = cs.Close()
	snapHash := fileHash(t, colFile(dir, ".snap"))
	if err := writeConfig(colFile(dir, ".json"), legacyCfg()); err != nil {
		t.Fatal(err)
	}
	logs := captureLogs(t)

	cs = openDurable(t, dir, upgradeCol)
	defer func() { _ = cs.Close() }()
	kept, _ := filepath.Glob(colFile(dir, ".snap.kept-*"))
	if len(kept) != 1 || fileHash(t, kept[0]) != snapHash {
		t.Fatalf("kept snapshots = %v, want one copy of the original", kept)
	}
	if !strings.Contains(logs.String(), kept[0]) {
		t.Fatalf("log = %q, want the kept snapshot named", logs.String())
	}
}

// An empty snapshot beside a never-flushed collection, as an upgrade
// interrupted after its snapshot step leaves, is not kept aside.
func TestUpgradeEmptyOrphanSnapshotNotKept(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	if err := cs.CreateCollection(upgradeCol, legacyCfg()); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()
	crashed := filepath.Join(t.TempDir(), "crashed")
	setUpgradeHook(t, func(s string) error {
		if s == "snapshot" {
			copyTree(t, dir, crashed)
		}
		return nil
	})
	cs = openDurable(t, dir, upgradeCol)
	_ = cs.Close()
	upgradeStepHook = nil

	cs = openDurable(t, crashed, upgradeCol)
	defer func() { _ = cs.Close() }()
	assertDurableManifest(t, manifestOf(t, crashed, upgradeCol))
	if kept, _ := filepath.Glob(colFile(crashed, ".snap.kept-*")); len(kept) != 0 {
		t.Fatalf("empty snapshot kept aside: %v", kept)
	}
}

// A log manifest left by an interrupted upgrade, with the old sidecar still
// beside it, whose snapshot does not load, fails the open rather than falling
// back to the old files: nothing proves the log form took no writes since, so
// the old files may be older than the collection. Nothing is removed.
func TestInterruptedUpgradeWithUnloadableSnapshotFailsOpen(t *testing.T) {
	dir := t.TempDir()
	writeLegacyStore(t, dir)
	crashed := filepath.Join(t.TempDir(), "crashed")
	setUpgradeHook(t, func(s string) error {
		if s == "manifest" {
			copyTree(t, dir, crashed)
		}
		return nil
	})
	cs := openDurable(t, dir, upgradeCol)
	_ = cs.Close()
	upgradeStepHook = nil
	if err := os.WriteFile(colFile(crashed, ".snap"), []byte("not a snapshot"), 0o600); err != nil {
		t.Fatal(err)
	}
	if cs, err := OpenCollectionStoreWith(crashed, StoreOptions{DurableCollections: []string{upgradeCol}}); err == nil {
		_ = cs.Close()
		t.Fatal("open succeeded over an unloadable snapshot")
	}
	for _, p := range mappedFiles(crashed) {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s removed by a failed open: %v", p, err)
		}
	}
}
