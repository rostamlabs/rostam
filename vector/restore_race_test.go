// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// These tests force the interleavings a create-or-replace restore has with the
// other operations that change what a name refers to. Each barrier is a real
// blocking point on the path under test — an in-flight user pinning a collection
// so its drop cannot finish, a snapshot reader or object-store fetch that has not
// returned — rather than a flag set by hand.

// gateReader blocks its first Read until release is closed.
type gateReader struct {
	r       io.Reader
	once    sync.Once
	entered chan struct{}
	release chan struct{}
}

func newGateReader(b []byte) *gateReader {
	return &gateReader{r: bytes.NewReader(b), entered: make(chan struct{}), release: make(chan struct{})}
}

func (g *gateReader) Read(p []byte) (int, error) {
	g.once.Do(func() { close(g.entered); <-g.release })
	return g.r.Read(p)
}

// gateStore blocks its first Get until release is closed.
type gateStore struct {
	objstore.ObjectStore
	once    sync.Once
	entered chan struct{}
	release chan struct{}
}

func newGateStore() *gateStore {
	return &gateStore{ObjectStore: objstore.NewMemStore(), entered: make(chan struct{}), release: make(chan struct{})}
}

func (g *gateStore) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	g.once.Do(func() { close(g.entered); <-g.release })
	return g.ObjectStore.Get(ctx, key)
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(100 * time.Microsecond)
	}
}

// nameLockHolders reports how many goroutines hold or wait for canonical's name
// lock.
func nameLockHolders(s *CollectionStore, canonical string) int {
	s.nameLocksMu.Lock()
	defer s.nameLocksMu.Unlock()
	if l := s.nameLocks[canonical]; l != nil {
		return l.refs
	}
	return 0
}

func closed(ch <-chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

// restoreAsync runs a restore and closes done when it returns.
func restoreAsync(s *CollectionStore, cfg Config, r io.Reader) (done chan struct{}, err *error) {
	done = make(chan struct{})
	err = new(error)
	go func() {
		*err = s.RestoreCollectionWithConfig("docs", cfg, r)
		close(done)
	}()
	return done, err
}

// requireReplacementDurable checks the restored WAL collection's marker and log
// are on disk, then proves it end to end: a write after the restore survives a
// reopen, along with the restored points, and the original's do not.
func requireReplacementDurable(t *testing.T, s *CollectionStore, dir string, restored, original []uint64) {
	t.Helper()
	base := filepath.Join(dir, "vectors", DefaultTenant, "docs")
	for _, ext := range []string{".json", ".wal"} {
		if _, err := os.Stat(base + ext); err != nil {
			t.Errorf("restored collection's %s is missing: %v", ext, err)
		}
	}
	requirePoints(t, s, "docs", restored, original)
	insertIDs(t, s, "docs", []uint64{5000})
	if err := s.Flush("docs"); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = s2.Close() }()
	requirePoints(t, s2, "docs", append(append([]uint64(nil), restored...), 5000), original)
}

// TestRestoreWaitsForInFlightDropCleanup: a drop removes the collection from the
// catalog, then waits for in-flight users before deleting its files by path. A
// restore that published in that gap used to find nothing to drop, write its
// marker and log — and then have the earlier drop's cleanup delete both. The
// restore returned success, served the data, and lost it on restart.
func TestRestoreWaitsForInFlightDropCleanup(t *testing.T) {
	cfg := restoreReplaceModes[1].cfg // WAL: a marker and a log to lose
	restored := restoreIDs(1, 40)
	snap := sourceSnapshot(t, cfg, restored)
	dir := t.TempDir()
	s, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	original := restoreIDs(1000, 50)
	insertIDs(t, s, "docs", original)

	pin, _ := s.Acquire("docs")
	dropped := make(chan struct{})
	go func() {
		_ = s.DropCollection("docs")
		close(dropped)
	}()
	waitUntil(t, "the drop to remove the catalog entry", func() bool { _, ok := s.Get("docs"); return !ok })

	done, rerr := restoreAsync(s, cfg, bytes.NewReader(snap))
	waitUntil(t, "the restore to finish or queue behind the drop", func() bool {
		return closed(done) || nameLockHolders(s, "default/docs") >= 2
	})
	pin.Release() // the drop's cleanup runs now
	<-dropped
	<-done
	if *rerr != nil {
		t.Fatalf("restore: %v", *rerr)
	}
	requireReplacementDurable(t, s, dir, restored, original)
}

// TestCreateWaitsForInFlightDropCleanup is the same race without a restore: a
// create of a name whose drop is still draining wrote its marker and log, and the
// drop's cleanup then deleted both.
func TestCreateWaitsForInFlightDropCleanup(t *testing.T) {
	cfg := restoreReplaceModes[1].cfg
	dir := t.TempDir()
	s, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	insertIDs(t, s, "docs", restoreIDs(1000, 5))

	pin, _ := s.Acquire("docs")
	dropped := make(chan struct{})
	go func() {
		_ = s.DropCollection("docs")
		close(dropped)
	}()
	waitUntil(t, "the drop to remove the catalog entry", func() bool { _, ok := s.Get("docs"); return !ok })

	created := make(chan error, 1)
	go func() { created <- s.CreateCollection("docs", cfg) }()
	waitUntil(t, "the create to finish or queue behind the drop", func() bool {
		return len(created) == 1 || nameLockHolders(s, "default/docs") >= 2
	})
	pin.Release()
	<-dropped
	if err := <-created; err != nil {
		t.Fatalf("create: %v", err)
	}
	insertIDs(t, s, "docs", []uint64{7})
	base := filepath.Join(dir, "vectors", DefaultTenant, "docs")
	for _, ext := range []string{".json", ".wal"} {
		if _, err := os.Stat(base + ext); err != nil {
			t.Errorf("created collection's %s is missing: %v", ext, err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = s2.Close() }()
	requirePoints(t, s2, "docs", []uint64{7}, []uint64{1000})
}

// TestRestoreWaitsForInFlightPromotion: resolving a cold collection promotes it,
// and a promotion that finds its stub already dropped discards what it built by
// deleting the name's files by path. A restore that published while a promotion
// was fetching its snapshot used to have its marker and log deleted that way.
func TestRestoreWaitsForInFlightPromotion(t *testing.T) {
	cfg := restoreReplaceModes[1].cfg
	restored := restoreIDs(1, 40)
	snap := sourceSnapshot(t, cfg, restored)
	dir := t.TempDir()
	s, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	original := restoreIDs(1000, 50)
	insertIDs(t, s, "docs", original)
	obj := newGateStore()
	if err := s.EvictCollection(context.Background(), "docs", obj, "t", time.Unix(1, 0)); err != nil {
		t.Fatal(err)
	}

	promoted := make(chan struct{})
	go func() {
		if c, ok := s.Acquire("docs"); ok {
			c.Release()
		}
		close(promoted)
	}()
	<-obj.entered // the promotion is fetching the cold snapshot

	done, rerr := restoreAsync(s, cfg, bytes.NewReader(snap))
	waitUntil(t, "the restore to finish or queue behind the promotion", func() bool {
		return closed(done) || nameLockHolders(s, "default/docs") >= 2
	})
	close(obj.release)
	<-promoted
	<-done
	if *rerr != nil {
		t.Fatalf("restore: %v", *rerr)
	}
	if s.IsCold("docs") {
		t.Fatal("name is still cold after the restore")
	}
	requireReplacementDurable(t, s, dir, restored, original)
}

// TestRestoreClusterPromotionCannotShareStagedGeneration: on a persistent-cluster
// store a collection's mmap files are named by the store generation, and a
// promotion builds on the current one — which is the generation a restore has
// just advanced to for its staged copy. A promotion during staging used to open
// the staged copy's own files, and dropping that promoted collection at publish
// deleted them out from under the restored index.
func TestRestoreClusterPromotionCannotShareStagedGeneration(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 32, Seed: 1, Quant: QuantSQ8}
	restored := restoreIDs(1, 40)
	snap := sourceSnapshot(t, cfg, restored)
	dir := t.TempDir()
	s, err := OpenCollectionStorePersistent(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	original := restoreIDs(1000, 50)
	insertIDs(t, s, "docs", original)
	if err := s.EvictCollection(context.Background(), "docs", objstore.NewMemStore(), "t", time.Unix(1, 0)); err != nil {
		t.Fatal(err)
	}

	gr := newGateReader(snap)
	done, rerr := restoreAsync(s, cfg, gr)
	<-gr.entered // staged on a fresh generation, reading the snapshot

	promoted := make(chan struct{})
	go func() {
		if c, ok := s.Acquire("docs"); ok {
			c.Release()
		}
		close(promoted)
	}()
	waitUntil(t, "the promotion to finish or queue behind the restore", func() bool {
		return closed(promoted) || nameLockHolders(s, "default/docs") >= 2
	})
	close(gr.release)
	<-done
	<-promoted
	if *rerr != nil {
		t.Fatalf("restore: %v", *rerr)
	}
	requirePoints(t, s, "docs", restored, original)
	c, ok := s.Acquire("docs")
	if !ok {
		t.Fatal("restored collection missing")
	}
	files := []string{c.cfg.GraphMmapPath, c.cfg.MmapPath}
	c.Release()
	for _, p := range files {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("restored collection's mmap file %s is gone: %v", filepath.Base(p), err)
		}
	}
}

// TestRestoreAllCannotLandInsidePublication: a whole-store RestoreAll swaps every
// catalog map at once. Landing between a restore's drop and its registration, it
// made the restore fail — after the original was already gone — with "registered
// by another operation". The swap must fall wholly before or after publication.
func TestRestoreAllCannotLandInsidePublication(t *testing.T) {
	cfg := restoreReplaceModes[0].cfg
	restored := restoreIDs(1, 40)
	snap := sourceSnapshot(t, cfg, restored)
	other := newCollectionStore(t)
	if err := other.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	insertIDs(t, other, "docs", []uint64{7000})
	var whole bytes.Buffer
	if err := other.SnapshotAll(&whole); err != nil {
		t.Fatal(err)
	}

	s := newCollectionStore(t)
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	insertIDs(t, s, "docs", restoreIDs(1000, 5))

	pin, _ := s.Acquire("docs") // holds the publication open inside its drop
	done, rerr := restoreAsync(s, cfg, bytes.NewReader(snap))
	waitUntil(t, "publication to drop the original", func() bool { _, ok := s.Get("docs"); return !ok })

	allDone := make(chan struct{})
	var allErr error
	go func() {
		allErr = s.RestoreAll(bytes.NewReader(whole.Bytes()))
		close(allDone)
	}()
	waitUntil(t, "RestoreAll to swap or wait for the publication", func() bool {
		if _, _, _, _, ok, _ := s.GetPoint("docs", 7000); ok {
			return true // swapped inside the window
		}
		if s.catalogMu.TryRLock() {
			s.catalogMu.RUnlock()
			return false
		}
		return true // a writer is waiting behind the publication
	})
	pin.Release()
	<-done
	<-allDone
	if *rerr != nil {
		t.Fatalf("restore: %v", *rerr)
	}
	if allErr != nil {
		t.Fatalf("RestoreAll: %v", allErr)
	}
	// The swap came after the publication, so the whole-store state is the result.
	requirePoints(t, s, "docs", []uint64{7000}, []uint64{1, 1000})
}

// TestEvictDoesNotCommitOverReplacement: an eviction snapshots the collection,
// releases it, then commits by replacing the name's catalog entry with a cold
// stub. It used to commit over whatever was registered by then, so a restore
// published in that gap was replaced by a stub of the OLD data.
func TestEvictDoesNotCommitOverReplacement(t *testing.T) {
	cfg := restoreReplaceModes[0].cfg
	restored := restoreIDs(1, 40)
	snap := sourceSnapshot(t, cfg, restored)
	s := newCollectionStore(t)
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	original := restoreIDs(1000, 50)
	insertIDs(t, s, "docs", original)

	entered, release := make(chan struct{}), make(chan struct{})
	evictCommitHook = func() { close(entered); <-release }
	t.Cleanup(func() { evictCommitHook = nil })

	evicted := make(chan error, 1)
	go func() {
		evicted <- s.EvictCollection(context.Background(), "docs", objstore.NewMemStore(), "t", time.Unix(1, 0))
	}()
	<-entered // snapshot taken and released, not yet committed
	if err := s.RestoreCollectionWithConfig("docs", cfg, bytes.NewReader(snap)); err != nil {
		t.Fatalf("restore: %v", err)
	}
	close(release)
	if err := <-evicted; err != nil {
		t.Fatalf("evict: %v", err)
	}
	if s.IsCold("docs") {
		t.Fatal("the eviction replaced the restored collection with a stub of the old one")
	}
	requirePoints(t, s, "docs", restored, original)
}

// TestRestorePublishesCurrentClock: SetNowFunc updates only catalogued
// collections, so an override installed while a restore is staging must be
// applied to the staged collection when it is registered.
func TestRestorePublishesCurrentClock(t *testing.T) {
	cfg := restoreReplaceModes[0].cfg
	snap := sourceSnapshot(t, cfg, restoreIDs(1, 40))
	s := newCollectionStore(t)

	gr := newGateReader(snap)
	done, rerr := restoreAsync(s, cfg, gr)
	<-gr.entered
	s.SetNowFunc(func() int64 { return 42 })
	close(gr.release)
	<-done
	if *rerr != nil {
		t.Fatalf("restore: %v", *rerr)
	}
	c, ok := s.Acquire("docs")
	if !ok {
		t.Fatal("restored collection missing")
	}
	defer c.Release()
	if got := c.idx.(*hnsw).now(); got != 42 {
		t.Errorf("restored collection's clock = %d, want the store override installed during staging (42)", got)
	}
}

// TestRestoreOverColdCollectionCompletes: a restore holds the name lock for its
// whole body, and resolving a cold collection promotes it under that same lock,
// which is not reentrant. If anything on the restore path resolved the cold
// original, the restore would wait on itself forever. It must complete, and an
// operation that resolves the cold name meanwhile must wait for it and then see
// the restored collection.
func TestRestoreOverColdCollectionCompletes(t *testing.T) {
	for _, mode := range restoreReplaceModes {
		t.Run(mode.name, func(t *testing.T) {
			restored := restoreIDs(1, 40)
			snap := sourceSnapshot(t, mode.cfg, restored)
			dir := t.TempDir()
			s, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = s.Close() }()
			if err := s.CreateCollection("docs", mode.cfg); err != nil {
				t.Fatal(err)
			}
			original := restoreIDs(1000, 50)
			insertIDs(t, s, "docs", original)
			if err := s.EvictCollection(context.Background(), "docs", objstore.NewMemStore(), "t", time.Unix(1, 0)); err != nil {
				t.Fatal(err)
			}

			gr := newGateReader(snap)
			done, rerr := restoreAsync(s, mode.cfg, gr)
			<-gr.entered

			sawRestored := make(chan bool, 1)
			go func() {
				_, _, _, _, ok, _ := s.GetPoint("docs", 1)
				sawRestored <- ok
			}()
			waitUntil(t, "the cold read to queue behind the restore", func() bool {
				return nameLockHolders(s, "default/docs") >= 2
			})
			close(gr.release)

			select {
			case <-done:
			case <-time.After(30 * time.Second):
				t.Fatal("restore over a cold collection did not complete: it is waiting on its own name lock")
			}
			if *rerr != nil {
				t.Fatalf("restore: %v", *rerr)
			}
			if !<-sawRestored {
				t.Error("a read that waited on the restore did not see the restored collection")
			}
			if s.IsCold("docs") {
				t.Error("name is still cold after the restore")
			}
			requirePoints(t, s, "docs", restored, original)
		})
	}
}

// TestNameLocksDoNotAccumulate: the name-lock map holds an entry only while a
// name is locked or awaited, so churning through many distinct names — and
// contending on one — leaves it empty.
func TestNameLocksDoNotAccumulate(t *testing.T) {
	cfg := restoreReplaceModes[0].cfg
	snap := sourceSnapshot(t, cfg, restoreIDs(1, 5))
	s := newCollectionStore(t)

	for i := 0; i < 300; i++ {
		name := fmt.Sprintf("dense-%d", i)
		if err := s.CreateCollection(name, cfg); err != nil {
			t.Fatal(err)
		}
		if err := s.DropCollection(name); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 50; i++ {
		named := fmt.Sprintf("named-%d", i)
		if err := s.CreateNamed(named, map[string]NamedVectorParams{"a": {Dim: 4}}); err != nil {
			t.Fatal(err)
		}
		if err := s.DropNamed(named); err != nil {
			t.Fatal(err)
		}
		multi := fmt.Sprintf("multi-%d", i)
		if err := s.CreateMultiVector(multi, MultiVectorConfig{Dim: 4}); err != nil {
			t.Fatal(err)
		}
		if err := s.DropMultiVector(multi); err != nil {
			t.Fatal(err)
		}
		restoredName := fmt.Sprintf("restored-%d", i)
		if err := s.RestoreCollectionWithConfig(restoredName, cfg, bytes.NewReader(snap)); err != nil {
			t.Fatal(err)
		}
		// A failed restore releases its lock too.
		if err := s.RestoreCollectionWithConfig(restoredName, cfg, bytes.NewReader(snap[:len(snap)/2])); err == nil {
			t.Fatal("truncated restore succeeded")
		}
		cold := fmt.Sprintf("cold-%d", i)
		if err := s.CreateCollection(cold, cfg); err != nil {
			t.Fatal(err)
		}
		insertIDs(t, s, cold, []uint64{1})
		if err := s.EvictCollection(context.Background(), cold, objstore.NewMemStore(), "t", time.Unix(1, 0)); err != nil {
			t.Fatal(err)
		}
		requirePoints(t, s, cold, []uint64{1}, nil) // promotes
	}

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				_ = s.CreateCollection("contended", cfg)
				_ = s.DropCollection("contended")
			}
		}()
	}
	wg.Wait()

	s.nameLocksMu.Lock()
	n := len(s.nameLocks)
	var some []string
	for name := range s.nameLocks {
		if len(some) == 5 {
			break
		}
		some = append(some, name)
	}
	s.nameLocksMu.Unlock()
	if n != 0 {
		t.Fatalf("name-lock map holds %d entries after every operation returned (e.g. %s)", n, strings.Join(some, ", "))
	}
}
