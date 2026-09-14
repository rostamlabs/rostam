// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// restoreReplaceModes are the single-node storage modes a create-or-replace
// restore has to get right. They differ in exactly what lives on disk under the
// collection's name — nothing but the config marker (heap), a write-ahead log
// (WAL), or the mmap'd vector and graph files plus an instant-restart sidecar
// (Persistent) — which is what a staged restore must not collide with.
var restoreReplaceModes = []struct {
	name string
	cfg  Config
}{
	{"heap", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 32, Seed: 1}},
	{"wal", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 32, Seed: 1, WAL: true, WALNoSync: true}},
	{"persistent", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 32, Seed: 1, Quant: QuantSQ8, Persistent: true}},
}

func restoreVec(id uint64) []float32 {
	return []float32{float32(id), float32(id % 7), 1, 0}
}

func insertIDs(t *testing.T, s *CollectionStore, name string, ids []uint64) {
	t.Helper()
	for _, id := range ids {
		if err := s.Insert(name, id, restoreVec(id), 0, nil, nil); err != nil {
			t.Fatalf("insert %d into %q: %v", id, name, err)
		}
	}
}

func restoreIDs(from, n uint64) []uint64 {
	out := make([]uint64, n)
	for i := range out {
		out[i] = from + uint64(i)
	}
	return out
}

// requirePoints asserts every id in present resolves in the named collection and
// every id in absent does not.
func requirePoints(t *testing.T, s *CollectionStore, name string, present, absent []uint64) {
	t.Helper()
	for _, id := range present {
		_, _, _, _, ok, err := s.GetPoint(name, id)
		if err != nil || !ok {
			t.Fatalf("point %d in %q: ok=%v err=%v, want present", id, name, ok, err)
		}
	}
	for _, id := range absent {
		_, _, _, _, ok, err := s.GetPoint(name, id)
		if err != nil || ok {
			t.Fatalf("point %d in %q: ok=%v err=%v, want absent", id, name, ok, err)
		}
	}
}

// sourceSnapshot builds a collection holding ids in a throwaway store and returns
// its snapshot bytes — the stream a backup would hand to a restore.
func sourceSnapshot(t *testing.T, cfg Config, ids []uint64) []byte {
	t.Helper()
	src := newCollectionStore(t)
	if err := src.CreateCollection("src", cfg); err != nil {
		t.Fatalf("create source: %v", err)
	}
	insertIDs(t, src, "src", ids)
	c, ok := src.Acquire("src")
	if !ok {
		t.Fatal("source collection missing")
	}
	defer c.Release()
	var buf bytes.Buffer
	if err := c.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot source: %v", err)
	}
	return buf.Bytes()
}

// tenantFiles lists the entries of a tenant directory.
func tenantFiles(t *testing.T, dir, tenant string) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(dir, "vectors", tenant))
	if err != nil {
		t.Fatalf("read tenant dir: %v", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name()+"/")
			continue
		}
		names = append(names, e.Name())
	}
	return names
}

// requireOnlyOwnFiles fails if the tenant directory holds anything that is not
// one of the collection's own files — a directory, or a file named for anything
// but the collection.
func requireOnlyOwnFiles(t *testing.T, dir, tenant, col string) {
	t.Helper()
	for _, name := range tenantFiles(t, dir, tenant) {
		if strings.HasSuffix(name, "/") || !strings.HasPrefix(name, col+".") {
			t.Errorf("unexpected entry %q in tenant dir after restore", name)
		}
	}
}

// TestRestoreFailureKeepsOriginal is the regression test for a create-or-replace
// restore that destroyed its target when the snapshot turned out to be bad. The
// old sequence dropped the existing collection (removing its files) and created
// an empty one BEFORE it read a byte of the snapshot, so a truncated stream left
// an empty collection where the original had been.
//
// A failed restore must leave the original serving exactly as it was, keep its
// on-disk state loadable (proven by reopening the store), and leave nothing of
// the attempt behind.
func TestRestoreFailureKeepsOriginal(t *testing.T) {
	for _, mode := range restoreReplaceModes {
		t.Run(mode.name, func(t *testing.T) {
			snap := sourceSnapshot(t, mode.cfg, restoreIDs(1, 40))
			bad := map[string][]byte{
				// Cut mid-stream: the header parses, the body runs out.
				"truncated": snap[:len(snap)/2],
				// Not a snapshot from the first byte.
				"bad-magic": append([]byte("not a snapshot"), snap[16:]...),
			}
			for badName, badSnap := range bad {
				t.Run(badName, func(t *testing.T) {
					dir := t.TempDir()
					s, err := OpenCollectionStore(dir)
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = s.Close() }()
					if err := s.CreateCollection("docs", mode.cfg); err != nil {
						t.Fatalf("create original: %v", err)
					}
					original := restoreIDs(1000, 50)
					insertIDs(t, s, "docs", original)
					filesBefore := strings.Join(tenantFiles(t, dir, DefaultTenant), ",")

					err = s.RestoreCollectionWithConfig("docs", mode.cfg, bytes.NewReader(badSnap))
					if err == nil {
						t.Fatal("restore from a bad snapshot succeeded, want an error")
					}

					if _, ok := s.Get("docs"); !ok {
						t.Fatalf("original collection is gone after a failed restore (err: %v)", err)
					}
					requirePoints(t, s, "docs", original, []uint64{1, 2, 3})
					if files := strings.Join(tenantFiles(t, dir, DefaultTenant), ","); files != filesBefore {
						t.Errorf("tenant dir after a failed restore = [%s], want unchanged [%s]", files, filesBefore)
					}
					if names := s.CollectionNames(); len(names) != 1 || names[0] != "default/docs" {
						t.Fatalf("catalog after failed restore = %v, want only default/docs", names)
					}
					// Still writable, and the write lands on the original.
					insertIDs(t, s, "docs", []uint64{2000})

					// The original's files must be intact: flush, reopen, and find it.
					if err := s.Flush("docs"); err != nil {
						t.Fatalf("flush original after failed restore: %v", err)
					}
					if err := s.Close(); err != nil {
						t.Fatal(err)
					}
					s2, err := OpenCollectionStore(dir)
					if err != nil {
						t.Fatalf("reopen after failed restore: %v", err)
					}
					defer func() { _ = s2.Close() }()
					requirePoints(t, s2, "docs", append(original, 2000), []uint64{1, 2, 3})
				})
			}
		})
	}
}

// TestRestoreReplacesCollection pins the success path: the restored snapshot
// fully replaces the existing collection (old points gone, snapshot points
// present), the replacement is writable, its durability machinery is attached
// under the collection's own name (it flushes and reopens from its own files, and
// a WAL collection logs writes made after the checkpoint), and nothing else is
// left on disk.
func TestRestoreReplacesCollection(t *testing.T) {
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
				t.Fatalf("create original: %v", err)
			}
			original := restoreIDs(1000, 50)
			insertIDs(t, s, "docs", original)

			if err := s.RestoreCollectionWithConfig("docs", mode.cfg, bytes.NewReader(snap)); err != nil {
				t.Fatalf("restore: %v", err)
			}
			requirePoints(t, s, "docs", restored, original)
			requireOnlyOwnFiles(t, dir, DefaultTenant, "docs")
			insertIDs(t, s, "docs", []uint64{4000})
			requirePoints(t, s, "docs", []uint64{4000}, nil)
			if mode.cfg.Persistent {
				// Stop here for Persistent: Collection.Restore replaces the index's
				// mmap-backed vector arena with a heap one, so a restored Persistent
				// collection cannot Flush (ErrPersistUnsupported), staged publish or
				// not. That is a separate defect of Restore itself.
				return
			}

			if err := s.Flush("docs"); err != nil {
				t.Fatalf("flush restored collection: %v", err)
			}
			// Written after the checkpoint: durable across a restart only through
			// the WAL, so it proves the WAL is attached to the replacement.
			insertIDs(t, s, "docs", []uint64{5000})
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}

			s2, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatalf("reopen after restore: %v", err)
			}
			defer func() { _ = s2.Close() }()
			present := append(append([]uint64(nil), restored...), 4000)
			absent := append([]uint64(nil), original...)
			if mode.cfg.WAL {
				present = append(present, 5000)
			} else {
				// No log: a write after the last checkpoint does not survive a
				// restart. Asserting it absent is what makes the WAL branch's
				// "present" a statement about the WAL rather than about reopening.
				absent = append(absent, 5000)
			}
			requirePoints(t, s2, "docs", present, absent)
			requireOnlyOwnFiles(t, dir, DefaultTenant, "docs")
		})
	}
}

// TestRestoreOntoAbsentName covers create-or-replace with nothing to replace.
func TestRestoreOntoAbsentName(t *testing.T) {
	for _, mode := range restoreReplaceModes {
		t.Run(mode.name, func(t *testing.T) {
			restored := restoreIDs(1, 20)
			snap := sourceSnapshot(t, mode.cfg, restored)
			dir := t.TempDir()
			s, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = s.Close() }()

			if err := s.RestoreCollectionWithConfig("acme/docs", mode.cfg, bytes.NewReader(snap[:len(snap)/2])); err == nil {
				t.Fatal("restore from a truncated snapshot succeeded, want an error")
			}
			if _, ok := s.Get("acme/docs"); ok {
				t.Fatal("a failed restore onto an absent name registered a collection")
			}
			if err := s.RestoreCollectionWithConfig("acme/docs", mode.cfg, bytes.NewReader(snap)); err != nil {
				t.Fatalf("restore: %v", err)
			}
			requirePoints(t, s, "acme/docs", restored, nil)
			requireOnlyOwnFiles(t, dir, "acme", "docs")
		})
	}
}

// TestRestoreRejectedConfigKeepsOriginal covers a failure that is the config's
// fault rather than the stream's: a config the index refuses to build. It used
// to surface from the create that ran after the existing collection was dropped.
func TestRestoreRejectedConfigKeepsOriginal(t *testing.T) {
	cfg := restoreReplaceModes[0].cfg
	snap := sourceSnapshot(t, cfg, restoreIDs(1, 40))

	s := newCollectionStore(t)
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	original := restoreIDs(1000, 50)
	insertIDs(t, s, "docs", original)
	bad := cfg
	bad.Dim = -1
	if err := s.RestoreCollectionWithConfig("docs", bad, bytes.NewReader(snap)); err == nil {
		t.Fatal("restore with an invalid config succeeded, want an error")
	}
	requirePoints(t, s, "docs", original, []uint64{1, 2, 3})
}

// TestRestoreRefusesOtherFamilies: a dense snapshot replaces a dense collection
// or fills an absent name, and nothing else. Over a named-vector collection the
// old sequence dropped it from memory but left its marker, snapshot and WAL on
// disk, then registered a dense collection too — so after a restart the name
// loaded in BOTH families. The refusal must leave the other family's collection
// intact across a reopen, and must not have written a dense marker beside it.
func TestRestoreRefusesOtherFamilies(t *testing.T) {
	cfg := restoreReplaceModes[0].cfg
	snap := sourceSnapshot(t, cfg, restoreIDs(1, 40))

	for _, tc := range []struct {
		name   string
		create func(s *CollectionStore) error
		exists func(s *CollectionStore) bool
	}{
		{"named-vector", func(s *CollectionStore) error {
			return s.CreateNamedConfig("docs", NamedConfig{Spaces: map[string]NamedVectorParams{"a": {Dim: 4}}, WAL: true, WALNoSync: true})
		}, func(s *CollectionStore) bool { _, ok := s.GetNamed("docs"); return ok }},
		{"multi-vector", func(s *CollectionStore) error {
			return s.CreateMultiVector("docs", MultiVectorConfig{Dim: 4, WAL: true, WALNoSync: true})
		}, func(s *CollectionStore) bool { _, ok := s.GetMultiVector("docs"); return ok }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			s, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatal(err)
			}
			if err := tc.create(s); err != nil {
				t.Fatal(err)
			}
			err = s.RestoreCollectionWithConfig("docs", cfg, bytes.NewReader(snap))
			if !errors.Is(err, ErrCollectionExists) {
				t.Fatalf("dense restore over a %s collection = %v, want ErrCollectionExists", tc.name, err)
			}
			if !tc.exists(s) {
				t.Fatalf("the %s collection is gone after a refused restore", tc.name)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}

			s2, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatalf("reopen: %v", err)
			}
			defer func() { _ = s2.Close() }()
			if !tc.exists(s2) {
				t.Errorf("the %s collection did not survive a reopen", tc.name)
			}
			if _, dense := s2.Get("docs"); dense {
				t.Errorf("after reopen the name also loads as a dense collection")
			}
		})
	}
}

// TestRestoreClusterStoreStagesOnFreshGeneration covers a persistent-cluster
// store, whose collections are always mmap-backed on generation-suffixed files.
// The restore stages on a new generation, so a failure removes only that
// generation's files and a success leaves only the new generation's.
func TestRestoreClusterStoreStagesOnFreshGeneration(t *testing.T) {
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
	before := strings.Join(tenantFiles(t, dir, DefaultTenant), ",")

	if err := s.RestoreCollectionWithConfig("docs", cfg, bytes.NewReader(snap[:len(snap)/2])); err == nil {
		t.Fatal("restore from a truncated snapshot succeeded, want an error")
	}
	requirePoints(t, s, "docs", original, []uint64{1, 2, 3})
	if after := strings.Join(tenantFiles(t, dir, DefaultTenant), ","); after != before {
		t.Fatalf("tenant dir after a failed restore = [%s], want unchanged [%s]", after, before)
	}

	if err := s.RestoreCollectionWithConfig("docs", cfg, bytes.NewReader(snap)); err != nil {
		t.Fatalf("restore: %v", err)
	}
	requirePoints(t, s, "docs", restored, original)
	insertIDs(t, s, "docs", []uint64{4000})

	c, ok := s.Acquire("docs")
	if !ok {
		t.Fatal("restored collection missing")
	}
	genPrefix := strings.TrimSuffix(filepath.Base(c.cfg.GraphMmapPath), "graph") // "docs.gN."
	c.Release()
	if strings.Contains(before, genPrefix) {
		t.Fatalf("restored collection is on a pre-restore generation (%s in [%s])", genPrefix, before)
	}
	for _, name := range tenantFiles(t, dir, DefaultTenant) {
		if name != "docs.json" && !strings.HasPrefix(name, genPrefix) {
			t.Errorf("entry %q left after restore; want only docs.json and %s* files", name, genPrefix)
		}
	}
}
