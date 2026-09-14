// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// collectionFiles lists the files under dir's default-tenant vectors directory
// that belong to the collection named col: every store-managed file is named
// "<col>.<suffix>", generation-suffixed cluster files included.
func collectionFiles(t *testing.T, dir, col string) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(dir, "vectors", DefaultTenant))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), col+".") {
			out = append(out, e.Name())
		}
	}
	return out
}

// openDropStore opens the store in the mode under test.
func openDropStore(t *testing.T, dir string, cluster bool) *CollectionStore {
	t.Helper()
	s, err := OpenCollectionStorePersistent(dir, cluster)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return s
}

// TestDropRemovesEveryFamilysFiles creates a collection of each family in each
// storage mode, writes to it, checkpoints where the mode has checkpoints, drops
// it, and requires that no file of the collection is left and that a reopened
// store does not know the name. Named-vector collections are dropped both through
// DropNamed and through the family-agnostic DropCollection, which is the entry
// point the generic drop op and the HTTP and gRPC drop routes use.
func TestDropRemovesEveryFamilysFiles(t *testing.T) {
	const col = "docs"
	dense := func(cfg Config) func(*testing.T, *CollectionStore) {
		return func(t *testing.T, s *CollectionStore) {
			if err := s.CreateCollection(col, cfg); err != nil {
				t.Fatal(err)
			}
			if err := s.Insert(col, 1, []float32{1, 0, 0, 0}, 0, nil, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	multi := func(cfg MultiVectorConfig) func(*testing.T, *CollectionStore) {
		return func(t *testing.T, s *CollectionStore) {
			if err := s.CreateMultiVector(col, cfg); err != nil {
				t.Fatal(err)
			}
			if err := s.MultiAdd(col, 1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	named := func(wal bool) func(*testing.T, *CollectionStore) {
		return func(t *testing.T, s *CollectionStore) {
			if err := s.CreateNamedConfig(col, NamedConfig{Spaces: namedTestConfig(), WAL: wal}); err != nil {
				t.Fatal(err)
			}
			if err := s.NamedInsert(col, 1, map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}, nil, 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	denseCfg := Config{Dim: 4, M: 8, EfConstruction: 32, EfSearch: 32, Seed: 1}
	walDense := denseCfg
	walDense.WAL = true
	persistentDense := denseCfg
	persistentDense.Persistent = true
	persistentDense.Quant = QuantSQ8
	persistentDense.RescoreFactor = 2

	dropDense := func(s *CollectionStore) error { return s.DropCollection(col) }
	dropMulti := func(s *CollectionStore) error { return s.DropMultiVector(col) }
	dropNamed := func(s *CollectionStore) error { return s.DropNamed(col) }

	cases := []struct {
		name    string
		cluster bool
		create  func(*testing.T, *CollectionStore)
		flush   func(*CollectionStore) error // nil: the mode has no checkpoint
		drop    func(*CollectionStore) error
		// writesFiles: the mode has files on disk before the drop. Heap-only
		// multi-vector and named collections, and named collections in a
		// persistent-cluster store, write none.
		writesFiles bool
	}{
		{"dense/heap", false, dense(denseCfg), func(s *CollectionStore) error { return s.Flush(col) }, dropDense, true},
		{"dense/wal", false, dense(walDense), func(s *CollectionStore) error { return s.Flush(col) }, dropDense, true},
		{"dense/persistent", false, dense(persistentDense), func(s *CollectionStore) error { return s.Flush(col) }, dropDense, true},
		{"dense/cluster", true, dense(denseCfg), nil, dropDense, true},
		{"multi/heap", false, multi(MultiVectorConfig{Dim: 4, Seed: 1}), nil, dropMulti, false},
		{"multi/wal", false, multi(MultiVectorConfig{Dim: 4, Seed: 1, WAL: true}), func(s *CollectionStore) error { return s.FlushMVWAL(col) }, dropMulti, true},
		{"multi/persistent", false, multi(MultiVectorConfig{Dim: 4, Seed: 1, Persistent: true}), func(s *CollectionStore) error { return s.FlushMultiVector(col) }, dropMulti, true},
		{"multi/cluster", true, multi(MultiVectorConfig{Dim: 4, Seed: 1}), nil, dropMulti, true},
		{"multi/heap/DropCollection", false, multi(MultiVectorConfig{Dim: 4, Seed: 1}), nil, dropDense, false},
		{"multi/wal/DropCollection", false, multi(MultiVectorConfig{Dim: 4, Seed: 1, WAL: true}), func(s *CollectionStore) error { return s.FlushMVWAL(col) }, dropDense, true},
		{"multi/wal-unflushed/DropCollection", false, multi(MultiVectorConfig{Dim: 4, Seed: 1, WAL: true}), nil, dropDense, true},
		{"multi/persistent/DropCollection", false, multi(MultiVectorConfig{Dim: 4, Seed: 1, Persistent: true}), func(s *CollectionStore) error { return s.FlushMultiVector(col) }, dropDense, true},
		{"multi/cluster/DropCollection", true, multi(MultiVectorConfig{Dim: 4, Seed: 1}), nil, dropDense, true},
		{"named/heap/DropNamed", false, named(false), nil, dropNamed, false},
		{"named/heap/DropCollection", false, named(false), nil, dropDense, false},
		{"named/wal/DropNamed", false, named(true), func(s *CollectionStore) error { return s.FlushNamed(col) }, dropNamed, true},
		{"named/wal/DropCollection", false, named(true), func(s *CollectionStore) error { return s.FlushNamed(col) }, dropDense, true},
		{"named/wal-unflushed/DropCollection", false, named(true), nil, dropDense, true},
		{"named/cluster/DropNamed", true, named(true), nil, dropNamed, false},
		{"named/cluster/DropCollection", true, named(true), nil, dropDense, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			s := openDropStore(t, dir, tc.cluster)
			tc.create(t, s)
			if tc.flush != nil {
				if err := tc.flush(s); err != nil {
					t.Fatalf("flush: %v", err)
				}
			}
			// Guard against a vacuous pass: a mode that persists must have written
			// something for the drop to clean up.
			if before := collectionFiles(t, dir, col); (len(before) > 0) != tc.writesFiles {
				t.Fatalf("files before drop = %v, want files: %v", before, tc.writesFiles)
			}
			if err := tc.drop(s); err != nil {
				t.Fatalf("drop: %v", err)
			}
			// In-memory no-op guard: the drop must remove the name from the live
			// store's maps, not only its files. A heap-only mode writes nothing, so
			// the file and reopen checks below pass vacuously; only this catches a
			// drop that reported success while leaving the index in memory.
			if _, ok := s.Get(col); ok {
				t.Error("dropped name still live as a dense collection")
			}
			if _, ok := s.GetMultiVector(col); ok {
				t.Error("dropped name still live as a multi-vector collection")
			}
			if _, ok := s.GetNamed(col); ok {
				t.Error("dropped name still live as a named-vector collection")
			}
			if left := collectionFiles(t, dir, col); len(left) != 0 {
				t.Errorf("files left after drop: %v", left)
			}
			if err := s.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}

			s2 := openDropStore(t, dir, tc.cluster)
			defer func() { _ = s2.Close() }()
			if _, ok := s2.Get(col); ok {
				t.Error("dropped name reloaded as a dense collection")
			}
			if _, ok := s2.GetMultiVector(col); ok {
				t.Error("dropped name reloaded as a multi-vector collection")
			}
			if _, ok := s2.GetNamed(col); ok {
				t.Error("dropped name reloaded as a named-vector collection")
			}
		})
	}
}

// TestDropNamedThenReuseAsDenseLoadsOneFamily drops a named-vector collection
// with DropCollection, creates a dense collection under the same name, and
// reopens the store: the name must load as the dense collection only. A named
// collection whose files outlived the drop would load beside it, putting one name
// in two families.
func TestDropNamedThenReuseAsDenseLoadsOneFamily(t *testing.T) {
	const col = "docs"
	for _, mode := range []struct {
		name    string
		cluster bool
	}{{"single-node", false}, {"cluster", true}} {
		t.Run(mode.name, func(t *testing.T) {
			dir := t.TempDir()
			s := openDropStore(t, dir, mode.cluster)
			if err := s.CreateNamedConfig(col, NamedConfig{Spaces: namedTestConfig(), WAL: true}); err != nil {
				t.Fatal(err)
			}
			if err := s.NamedInsert(col, 1, map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0); err != nil {
				t.Fatal(err)
			}
			if err := s.DropCollection(col); err != nil {
				t.Fatalf("drop: %v", err)
			}
			if err := s.CreateCollection(col, Config{Dim: 4, M: 8, EfConstruction: 32, EfSearch: 32, Seed: 1}); err != nil {
				t.Fatalf("create dense over the dropped name: %v", err)
			}
			if err := s.Insert(col, 7, []float32{1, 0, 0, 0}, 0, nil, nil); err != nil {
				t.Fatal(err)
			}
			if err := s.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}

			s2 := openDropStore(t, dir, mode.cluster)
			defer func() { _ = s2.Close() }()
			if _, ok := s2.Get(col); !ok {
				t.Error("the dense collection did not reload")
			}
			if _, ok := s2.GetNamed(col); ok {
				t.Error("the dropped named-vector collection reloaded beside the dense one")
			}
		})
	}
}
