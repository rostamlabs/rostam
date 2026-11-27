// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Tests added in response to an independent review of the staged-commit-wait
// perf work (Fix 2/3): a real opMu-leak-on-panic blocker (B1), a real
// staged-delete-never-waited-on-insert-error blocker (B2), a gap in the
// group-commit effectiveness coverage (named/MV, not just dense Collection),
// and a gap in the deadline-counter persistence coverage (openPersist/Restore
// round-trip).

// panicOnceIndex wraps a VectorIndex and panics on the FIRST call to its
// `target` method ("Insert", "Delete", "RestoreInsert", or "replaceInsert", the
// atomic replace an Upsert over a live id uses), then delegates normally afterward
// (armed is single-shot). Used to force a real panic inside the staged-commit
// closures (Collection.InsertCASKeyTTL / UpsertCASKeyTTL) so a B1 regression —
// opMu leaking across a panicking op — would deadlock the very next write.
type panicOnceIndex struct {
	VectorIndex
	target string
	armed  atomic.Bool
}

func newPanicOnceIndex(inner VectorIndex, target string) *panicOnceIndex {
	p := &panicOnceIndex{VectorIndex: inner, target: target}
	p.armed.Store(true)
	return p
}

func (p *panicOnceIndex) Insert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (uint64, map[string]uint64, error) {
	if p.target == "Insert" && p.armed.CompareAndSwap(true, false) {
		panic("panicOnceIndex: forced panic (B1 opMu-leak regression test)")
	}
	return p.VectorIndex.Insert(id, vec, ttl, meta, sparse, keyTTLMs, cas)
}

func (p *panicOnceIndex) Delete(id uint64, cas CASCond) (bool, error) {
	if p.target == "Delete" && p.armed.CompareAndSwap(true, false) {
		panic("panicOnceIndex: forced panic (B1 opMu-leak regression test)")
	}
	return p.VectorIndex.Delete(id, cas)
}

func (p *panicOnceIndex) replaceInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, stamped bool, nowMs uint64) (uint64, map[string]uint64, error) {
	if p.target == "replaceInsert" && p.armed.CompareAndSwap(true, false) {
		panic("panicOnceIndex: forced panic (B1 opMu-leak regression test)")
	}
	return p.VectorIndex.replaceInsert(id, vec, ttl, meta, sparse, keyTTLMs, cas, stamped, nowMs)
}

func (p *panicOnceIndex) RestoreInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error {
	if p.target == "RestoreInsert" && p.armed.CompareAndSwap(true, false) {
		panic("panicOnceIndex: forced panic (B1 opMu-leak regression test)")
	}
	return p.VectorIndex.RestoreInsert(id, vec, ttl, meta, sparse, keyExpires, version)
}

// mustNotDeadlock runs op in a goroutine and fails the test if it doesn't
// return within the deadline — the direct proof that opMu was NOT left locked
// by a prior panicking op.
func mustNotDeadlock(t *testing.T, deadline time.Duration, op func() error) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- op() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("op after the recovered panic: %v", err)
		}
	case <-time.After(deadline):
		t.Fatal("B1 regression: opMu leaked across a panicking op — the next write deadlocked")
	}
}

// TestOpMuReleasedOnPanicInsert is the B1 regression test for the plain-insert
// staged closure (Collection.InsertCASKeyTTL, collection.go:206): a panic
// inside idx.Insert while opMu is held must still release opMu (via the
// closure's deferred Unlock) — server/handlers.go recovers per-request
// panics, so an unrecovered lock would silently deadlock every later write on
// this collection.
func TestOpMuReleasedOnPanicInsert(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	c.idx = newPanicOnceIndex(c.idx, "Insert") // same-package test-only swap

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected a panic from the wrapped index, got none")
			}
		}()
		_, _ = c.InsertCASKeyTTL(1, vec, 0, nil, nil, nil, CASCond{})
		t.Fatal("unreachable: InsertCASKeyTTL should have panicked")
	}()

	mustNotDeadlock(t, 5*time.Second, func() error {
		_, err := c.InsertCASKeyTTL(2, vec, 0, nil, nil, nil, CASCond{})
		return err
	})
}

// TestOpMuReleasedOnPanicUpsert is the B1 regression test for the logged
// upsert's staged closure (Collection.upsertLogged): a panic inside the
// index's atomic replace while opMu is held must still release opMu.
func TestOpMuReleasedOnPanicUpsert(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)
	if err := c.Insert(1, vec, 0, nil, nil); err != nil {
		t.Fatalf("initial Insert: %v", err)
	}
	c.idx = newPanicOnceIndex(c.idx, "replaceInsert")

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected a panic from the wrapped index, got none")
			}
		}()
		_, _ = c.UpsertCASKeyTTL(1, vec, "", 0, nil, nil, nil, CASCond{})
		t.Fatal("unreachable: UpsertCASKeyTTL should have panicked")
	}()

	mustNotDeadlock(t, 5*time.Second, func() error {
		_, err := c.UpsertCASKeyTTL(1, vec, "", 0, nil, nil, nil, CASCond{})
		return err
	})
}

// TestUpsertDimMismatchKeepsOldPoint (wall-clock path, Collection.UpsertCASKeyTTL):
// a replace refused with an ordinary bad request (ErrDimMismatch) surfaces the
// error and changes nothing — the old point stays live in memory and nothing
// is logged, so memory and log agree. (It used to delete the point and stage a
// delete record before the insert failed.)
func TestUpsertDimMismatchKeepsOldPoint(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)
	if err := c.Insert(1, vec, 0, nil, nil); err != nil {
		t.Fatalf("initial Insert: %v", err)
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	wrongDim := make([]float32, 4) // configured Dim is 16 (walCfg)
	_, err = c.UpsertCASKeyTTL(1, wrongDim, "", 0, nil, nil, nil, CASCond{})
	if !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("Upsert with wrong dim: err = %v, want ErrDimMismatch", err)
	}

	if got := syncs.Load(); got != 0 {
		t.Fatalf("a refused replace synced the log %d times; it must log nothing", got)
	}
	if !c.Exists(1) {
		t.Fatal("a refused replace removed the point it was replacing")
	}
}

// TestUpsertAtDimMismatchKeepsOldPoint is TestUpsertDimMismatchKeepsOldPoint for
// the leader-apply-stamped variant (UpsertCASKeyTTLAt).
func TestUpsertAtDimMismatchKeepsOldPoint(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)
	const nowMs int64 = 1_700_000_000_000
	if _, err := c.InsertCASKeyTTLAt(1, vec, 0, nil, nil, nil, CASCond{}, nowMs); err != nil {
		t.Fatalf("initial InsertCASKeyTTLAt: %v", err)
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	wrongDim := make([]float32, 4)
	_, err = c.UpsertCASKeyTTLAt(1, wrongDim, "", 0, nil, nil, nil, CASCond{}, nowMs+1)
	if !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("UpsertCASKeyTTLAt with wrong dim: err = %v, want ErrDimMismatch", err)
	}
	if got := syncs.Load(); got != 0 {
		t.Fatalf("a refused replace synced the log %d times; it must log nothing", got)
	}
	if !c.Exists(1) {
		t.Fatal("a refused replace removed the point it was replacing")
	}
}

// TestNamedGroupCommitEffective is TestCollectionGroupCommitEffective
// (wal_staged_commit_test.go) for NamedCollection.InsertCASKeyTTL: concurrent
// inserts must overlap in commitWaitStaged (opMu held only across
// apply+WRITE), coalescing many inserts into far fewer fsyncs.
func TestNamedGroupCommitEffective(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, ok := cs.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing")
	}

	var syncs atomic.Int64
	nc.wal.onSync = func() { syncs.Add(1) }

	const n = 200
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(id uint64) {
			defer wg.Done()
			vectors := map[string][]float32{
				"title": {float32(id), 0, 0, 0},
				"image": {0, float32(id), 0},
			}
			if _, err := nc.InsertCASKeyTTL(id, vectors, nil, nil, 0, nil, CASCond{}); err != nil {
				t.Errorf("InsertCASKeyTTL(%d): %v", id, err)
			}
		}(uint64(i + 1))
	}
	wg.Wait()

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for NamedCollection: %d fsyncs for %d concurrent inserts (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatal("no syncs recorded — durability hook broken")
	}
	t.Logf("NamedCollection group-commit: %d concurrent inserts coalesced into %d fsync(s)", n, got)
}

// TestMVGroupCommitEffective is TestCollectionGroupCommitEffective
// (wal_staged_commit_test.go) for MultiVectorIndex.AddCASKeyTTLSparse:
// concurrent adds must overlap in commitWaitStaged the same way.
func TestMVGroupCommitEffective(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing")
	}

	var syncs atomic.Int64
	idx.wal.onSync = func() { syncs.Add(1) }

	const n = 200
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(id uint64) {
			defer wg.Done()
			tokens := [][]float32{{float32(id), 0, 0, 0}}
			if _, err := idx.AddCASKeyTTLSparse(id, tokens, nil, nil, nil, CASCond{}); err != nil {
				t.Errorf("AddCASKeyTTLSparse(%d): %v", id, err)
			}
		}(uint64(i + 1))
	}
	wg.Wait()

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for MultiVectorIndex: %d fsyncs for %d concurrent adds (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatal("no syncs recorded — durability hook broken")
	}
	t.Logf("MultiVectorIndex group-commit: %d concurrent adds coalesced into %d fsync(s)", n, got)
}

// TestDeadlineCountsSurviveOpenPersistRoundtrip proves the Fix-4 counters
// (arena.deadlinePoints/deadlineKeys, the TTL sweep's fast-path gate) survive
// an mmap-instant-restart round-trip: openPersist writes arena.expires
// directly (bypassing SetExpires's incremental maintenance), so it must call
// RecomputeDeadlineCounts or the reopened index would silently never sweep a
// pending TTL point (an undercount — the dangerous direction).
func TestDeadlineCountsSurviveOpenPersistRoundtrip(t *testing.T) {
	const dim = 8
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		RescoreFactor: 2,
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	mkVec := func(seed float32) []float32 {
		v := make([]float32, dim)
		for i := range v {
			v[i] = seed + float32(i)
		}
		normalize(v)
		return v
	}
	if _, _, err := h.Insert(1, mkVec(1), 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1 (permanent): %v", err)
	}
	if _, _, err := h.Insert(2, mkVec(2), 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2 (TTL): %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots before persist = %d, want 1", got)
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()
	h2.now = func() int64 { return fakeNow }

	if got := h2.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots after openPersist = %d, want 1 (RecomputeDeadlineCounts missing/broken)", got)
	}

	fakeNow += 100
	if n := h2.sweepOnce(); n != 1 {
		t.Fatalf("sweepOnce after reopen = %d, want 1 (the TTL point should expire)", n)
	}
	if got := h2.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after the post-reopen sweep = %d, want 0", got)
	}
}

// TestDeadlineCountsSurviveSnapshotRestore is
// TestDeadlineCountsSurviveOpenPersistRoundtrip for the heap Snapshot/Restore
// path (hnsw.Restore, snapshot.go) instead of the mmap sidecar.
func TestDeadlineCountsSurviveSnapshotRestore(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1 (permanent): %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2 (TTL): %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots before snapshot = %d, want 1", got)
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	h2, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW (restore target): %v", err)
	}
	h2.now = func() int64 { return fakeNow }
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	if got := h2.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots after Restore = %d, want 1 (RecomputeDeadlineCounts missing/broken)", got)
	}

	fakeNow += 100
	if n := h2.sweepOnce(); n != 1 {
		t.Fatalf("sweepOnce after restore = %d, want 1 (the TTL point should expire)", n)
	}
	if got := h2.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after the post-restore sweep = %d, want 0", got)
	}
}
