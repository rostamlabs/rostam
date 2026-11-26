// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A replacing write (an upsert over a live id) on a log-backed collection must
// never lose both versions: whatever interrupts it, a reopen holds the old
// version or the new one, and the new one whenever it was acknowledged.

const replaceCol = "mcp_memory"

func snapPathOf(dir, col string) string {
	return filepath.Join(dir, "vectors", DefaultTenant, col+".snap")
}

// openReplaceStore opens a store whose replaceCol is durable and holds 1:"old"
// and 2:"other", both acknowledged.
func openReplaceStore(t *testing.T, dir string, cfg Config) *CollectionStore {
	t.Helper()
	cs := openDurable(t, dir, replaceCol)
	if err := cs.CreateCollection(replaceCol, cfg); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(replaceCol, 1, durableVec(1), "old", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert(replaceCol, 2, durableVec(2), "other", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	return cs
}

func acquire(t *testing.T, cs *CollectionStore) *Collection {
	t.Helper()
	c, ok := cs.Acquire(replaceCol)
	if !ok {
		t.Fatal("collection missing")
	}
	return c
}

func reopenPoints(t *testing.T, dir string) map[uint64]string {
	t.Helper()
	re := openDurable(t, dir, replaceCol)
	defer func() { _ = re.Close() }()
	return pointsOf(t, re, replaceCol)
}

// TestReplaceLogCutAtEveryByte cuts the log at every byte boundary inside the
// bytes one replace appended and reopens: each reopen holds the old version,
// except the uncut log, which holds the new one.
func TestReplaceLogCutAtEveryByte(t *testing.T) {
	// Every cut but the last is a torn tail, and each one is logged.
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(func() { slog.SetDefault(prev) })
	for _, stamped := range []bool{false, true} {
		name := "wall clock"
		if stamped {
			name = "stamped"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			cs := openReplaceStore(t, dir, durablePersistentCfg())
			walPath := walPathOf(dir, replaceCol)
			before := fileSize(t, walPath)
			c := acquire(t, cs)
			var err error
			if stamped {
				_, err = c.UpsertCASKeyTTLAt(1, durableVec(3), "new", 0, nil, nil, nil, CASCond{}, time.Now().UnixMilli())
			} else {
				_, err = c.UpsertCASKeyTTL(1, durableVec(3), "new", 0, nil, nil, nil, CASCond{})
			}
			if err != nil {
				t.Fatal(err)
			}
			saved, err := os.ReadFile(walPath) //nolint:gosec // test temp dir
			if err != nil {
				t.Fatal(err)
			}
			c.Release()
			_ = cs.Close()
			after := int64(len(saved))
			if after <= before {
				t.Fatalf("the replace appended nothing (log %d -> %d bytes)", before, after)
			}

			for cut := before; cut <= after; cut++ {
				sub := t.TempDir()
				copyTree(t, dir, sub)
				if err := os.WriteFile(walPathOf(sub, replaceCol), saved[:cut], 0o600); err != nil {
					t.Fatal(err)
				}
				want := map[uint64]string{1: "old", 2: "other"}
				if cut == after {
					want[1] = "new"
				}
				got := reopenPoints(t, sub)
				if len(got) != len(want) || got[1] != want[1] || got[2] != want[2] {
					t.Fatalf("log cut at %d of [%d, %d]: points = %v, want %v", cut, before, after, got, want)
				}
			}
		})
	}
}

// shortWriteHook lands only keep bytes of the next frame, then fails every
// write as a full disk would.
func shortWriteHook(keep int) func(f *os.File, p []byte) (int, error) {
	left := keep
	return func(f *os.File, p []byte) (int, error) {
		n := min(len(p), left)
		left -= n
		m, err := f.Write(p[:n])
		if err != nil {
			return m, err
		}
		if m < len(p) {
			return m, errors.New("injected: no space left on device")
		}
		return m, nil
	}
}

type fileState struct{ snap, wal []byte }

func readFiles(t *testing.T, dir string) fileState {
	t.Helper()
	var st fileState
	var err error
	if st.snap, err = os.ReadFile(snapPathOf(dir, replaceCol)); err != nil { //nolint:gosec // test temp dir
		t.Fatal(err)
	}
	if st.wal, err = os.ReadFile(walPathOf(dir, replaceCol)); err != nil { //nolint:gosec // test temp dir
		t.Fatal(err)
	}
	return st
}

// TestPoisonedLogIsNeverCheckpointed poisons the log (a short write inside a
// replace, or a failed fsync), then refuses more writes, then runs every
// checkpoint path. Each must fail, leave the snapshot and the log byte for byte
// as they were, and a reopen must hold exactly the acknowledged state.
func TestPoisonedLogIsNeverCheckpointed(t *testing.T) {
	cases := []struct {
		name   string
		poison func(t *testing.T, c *Collection)
	}{
		{"short write mid header", func(t *testing.T, c *Collection) {
			c.wal.writeHook = shortWriteHook(3)
			if _, err := c.UpsertCAS(1, durableVec(3), "new", 0, nil, nil, CASCond{}); err == nil {
				t.Fatal("a replace whose write failed was acknowledged")
			}
		}},
		{"short write mid payload", func(t *testing.T, c *Collection) {
			c.wal.writeHook = shortWriteHook(20)
			if _, err := c.UpsertCAS(1, durableVec(3), "new", 0, nil, nil, CASCond{}); err == nil {
				t.Fatal("a replace whose write failed was acknowledged")
			}
		}},
		{"short write in a stamped replace", func(t *testing.T, c *Collection) {
			c.wal.writeHook = shortWriteHook(20)
			if _, err := c.UpsertCASKeyTTLAt(1, durableVec(3), "new", 0, nil, nil, nil, CASCond{}, time.Now().UnixMilli()); err == nil {
				t.Fatal("a replace whose write failed was acknowledged")
			}
		}},
		{"failed fsync", func(t *testing.T, c *Collection) {
			// What commitWait does when f.Sync fails.
			c.wal.poisoned.Store(true)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cs := openReplaceStore(t, dir, durablePersistentCfg())
			if err := cs.CheckpointDurable(); err != nil {
				t.Fatal(err)
			}
			// Acknowledged after the checkpoint, so they live only in the log.
			if err := cs.Upsert(replaceCol, 3, durableVec(4), "third", 0, nil, nil); err != nil {
				t.Fatal(err)
			}
			if err := cs.Upsert(replaceCol, 2, durableVec(5), "other, rewritten", 0, nil, nil); err != nil {
				t.Fatal(err)
			}
			acked := map[uint64]string{1: "old", 2: "other, rewritten", 3: "third"}

			c := acquire(t, cs)
			tc.poison(t, c)
			c.wal.writeHook = nil
			// Refused once the log is poisoned: a delete, an overwrite and a fresh insert.
			if _, err := c.DeleteCAS(2, CASCond{}); err == nil {
				t.Fatal("a delete on a poisoned log was acknowledged")
			}
			if _, err := c.UpsertCAS(3, durableVec(6), "third, rewritten", 0, nil, nil, CASCond{}); err == nil {
				t.Fatal("an overwrite on a poisoned log was acknowledged")
			}
			if err := c.Upsert(4, durableVec(7), "fourth", 0, nil, nil); err == nil {
				t.Fatal("an insert on a poisoned log was acknowledged")
			}
			c.Release()

			files := readFiles(t, dir)
			if err := cs.CheckpointDurable(); !errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("CheckpointDurable on a poisoned log = %v, want ErrWALPoisoned", err)
			}
			if err := cs.Flush(replaceCol); !errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("Flush on a poisoned log = %v, want ErrWALPoisoned", err)
			}
			if got := readFiles(t, dir); !bytes.Equal(got.snap, files.snap) || !bytes.Equal(got.wal, files.wal) {
				t.Fatal("a checkpoint of a poisoned log changed the snapshot or the log")
			}
			_ = cs.Close()
			if got := readFiles(t, dir); !bytes.Equal(got.snap, files.snap) || !bytes.Equal(got.wal, files.wal) {
				t.Fatal("closing a store with a poisoned log changed the snapshot or the log")
			}
			assertPoints(t, reopenPoints(t, dir), acked)
		})
	}
}

// TestFailedReplaceKeepsOldVersion makes a replace fail in each way it can and
// then stops cleanly (a checkpoint) and reopens: the old version is there.
func TestFailedReplaceKeepsOldVersion(t *testing.T) {
	now := func() int64 { return time.Now().UnixMilli() }
	cases := []struct {
		name    string
		cfg     func() Config
		replace func(c *Collection) error
	}{
		{"wrong dimension", durablePersistentCfg, func(c *Collection) error {
			_, err := c.UpsertCAS(1, make([]float32, durableTestDim+1), "new", 0, nil, nil, CASCond{})
			return err
		}},
		{"wrong dimension, stamped", durablePersistentCfg, func(c *Collection) error {
			_, err := c.UpsertCASKeyTTLAt(1, make([]float32, durableTestDim+1), "new", 0, nil, nil, nil, CASCond{}, now())
			return err
		}},
		{"version conflict", durablePersistentCfg, func(c *Collection) error {
			_, err := c.UpsertCAS(1, durableVec(3), "new", 0, nil, nil, CASCond{Has: true, Expected: 99})
			return err
		}},
		{"version conflict, stamped", durablePersistentCfg, func(c *Collection) error {
			_, err := c.UpsertCASKeyTTLAt(1, durableVec(3), "new", 0, nil, nil, nil, CASCond{Has: true, Expected: 99}, now())
			return err
		}},
		{"malformed record", durablePersistentCfg, func(c *Collection) error {
			meta := Metadata{"r": Value{Kind: ValueRecord, Rec: []byte{0xDE, 0xAD}}}
			_, err := c.UpsertCAS(1, durableVec(3), "new", 0, meta, nil, CASCond{})
			return err
		}},
		{"rate limited", func() Config {
			cfg := durablePersistentCfg()
			cfg.MaxInsertsPerSecond = 1000
			return cfg
		}, func(c *Collection) error {
			drainBucket(c)
			_, err := c.UpsertCAS(1, durableVec(3), "new", 0, nil, nil, CASCond{})
			return err
		}},
		{"rate limited, stamped", func() Config {
			cfg := durablePersistentCfg()
			cfg.MaxInsertsPerSecond = 1000
			return cfg
		}, func(c *Collection) error {
			drainBucket(c)
			_, err := c.UpsertCASKeyTTLAt(1, durableVec(3), "new", 0, nil, nil, nil, CASCond{}, now())
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cs := openReplaceStore(t, dir, tc.cfg())
			c := acquire(t, cs)
			if err := tc.replace(c); err == nil {
				t.Fatal("the replace was expected to fail")
			}
			if _, meta, _, _, _, ok := c.Get(1); !ok || contentOf(meta) != "old" {
				t.Fatalf("after a failed replace the live point 1 is %v (ok=%v), want the old version", meta, ok)
			}
			c.Release()
			if err := cs.CheckpointDurable(); err != nil {
				t.Fatal(err)
			}
			_ = cs.Close()
			assertPoints(t, reopenPoints(t, dir), map[uint64]string{1: "old", 2: "other"})
		})
	}
}

// drainBucket empties the collection's insert rate bucket and stops it refilling.
func drainBucket(c *Collection) {
	b := c.idx.(*hnsw).bucket
	b.mu.Lock()
	defer b.mu.Unlock()
	frozen := b.now()
	b.now = func() time.Time { return frozen }
	b.lastRefill = frozen
	b.tokens = 0
}

// TestReplaceReplayOverCheckpoint replays a whole log of replaces and deletes of
// one id over a snapshot that already holds the final state (a checkpoint whose
// truncate never ran), and over the snapshot from before the log.
func TestReplaceReplayOverCheckpoint(t *testing.T) {
	histories := []struct {
		name  string
		write func(t *testing.T, c *Collection)
		want  map[uint64]string
	}{
		{"ends in a replace", func(t *testing.T, c *Collection) {
			for _, s := range []string{"a", "b"} {
				if err := c.Upsert(1, durableVec(len(s)+3), s, 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := c.DeleteCAS(1, CASCond{}); err != nil {
				t.Fatal(err)
			}
			for i, s := range []string{"c", "d", "e"} {
				if err := c.Upsert(1, durableVec(i+9), s, 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
		}, map[uint64]string{1: "e", 2: "other"}},
		{"ends in a delete", func(t *testing.T, c *Collection) {
			for i, s := range []string{"a", "b", "c"} {
				if err := c.Upsert(1, durableVec(i+3), s, 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := c.DeleteCAS(1, CASCond{}); err != nil {
				t.Fatal(err)
			}
		}, map[uint64]string{2: "other"}},
		{"CAS and stamped replaces", func(t *testing.T, c *Collection) {
			if _, err := c.UpsertCAS(1, durableVec(3), "a", 0, nil, nil, CASCond{Has: true, Expected: 1}); err != nil {
				t.Fatal(err)
			}
			if _, err := c.UpsertCASKeyTTLAt(1, durableVec(4), "b", 0, nil, nil, nil, CASCond{}, time.Now().UnixMilli()); err != nil {
				t.Fatal(err)
			}
		}, map[uint64]string{1: "b", 2: "other"}},
	}
	for _, h := range histories {
		t.Run(h.name, func(t *testing.T) {
			dir := t.TempDir()
			cs := openReplaceStore(t, dir, durablePersistentCfg())
			if err := cs.CheckpointDurable(); err != nil {
				t.Fatal(err)
			}
			preSnap, err := os.ReadFile(snapPathOf(dir, replaceCol)) //nolint:gosec // test temp dir
			if err != nil {
				t.Fatal(err)
			}
			c := acquire(t, cs)
			h.write(t, c)
			c.Release()
			log, err := os.ReadFile(walPathOf(dir, replaceCol)) //nolint:gosec // test temp dir
			if err != nil {
				t.Fatal(err)
			}
			if err := cs.CheckpointDurable(); err != nil {
				t.Fatal(err)
			}
			_ = cs.Close()

			// The checkpoint's snapshot with the whole log beside it.
			if err := os.WriteFile(walPathOf(dir, replaceCol), log, 0o600); err != nil {
				t.Fatal(err)
			}
			assertPoints(t, reopenPoints(t, dir), h.want)

			// The snapshot from before the log, with the log.
			if err := os.WriteFile(snapPathOf(dir, replaceCol), preSnap, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(walPathOf(dir, replaceCol), log, 0o600); err != nil {
				t.Fatal(err)
			}
			assertPoints(t, reopenPoints(t, dir), h.want)
		})
	}
}

// TestNamedAndMVReplaceLogCutAtEveryByte: the named and multi-vector families
// already log a replace as one record. Cut that record at every byte and the
// reopened id holds the old payload or, uncut, the new one — never nothing.
func TestNamedAndMVReplaceLogCutAtEveryByte(t *testing.T) {
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(func() { slog.SetDefault(prev) })

	newMeta := Metadata{"v": NewInt(2)}
	families := []struct {
		name    string
		fixture func(t *testing.T) (string, string)
		replace func(t *testing.T, cs *CollectionStore)
		payload func(t *testing.T, cs *CollectionStore) (Metadata, bool)
	}{
		{"named", namedFixture, func(t *testing.T, cs *CollectionStore) {
			nc, _ := cs.GetNamed(tailCol)
			if err := nc.Insert(1, map[string][]float32{"title": tailVec(9)}, newMeta, 0); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, cs *CollectionStore) (Metadata, bool) {
			nc, _ := cs.GetNamed(tailCol)
			_, meta, _, _, ok := nc.Get(1)
			return meta, ok
		}},
		{"multi-vector", mvFixture, func(t *testing.T, cs *CollectionStore) {
			idx, _ := cs.GetMultiVector(tailCol)
			if err := idx.Add(1, [][]float32{tailVec(9)}, newMeta); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, cs *CollectionStore) (Metadata, bool) {
			idx, _ := cs.GetMultiVector(tailCol)
			_, meta, _, ok := idx.Get(1)
			return meta, ok
		}},
	}
	for _, fam := range families {
		t.Run(fam.name, func(t *testing.T) {
			dir, walPath := fam.fixture(t)
			before := fileSize(t, walPath)
			cs, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatal(err)
			}
			fam.replace(t, cs)
			if err := cs.Close(); err != nil {
				t.Fatal(err)
			}
			saved, err := os.ReadFile(walPath) //nolint:gosec // test temp dir
			if err != nil {
				t.Fatal(err)
			}
			after := int64(len(saved))
			if after <= before {
				t.Fatalf("the replace appended nothing (log %d -> %d bytes)", before, after)
			}
			for cut := before; cut <= after; cut++ {
				sub := t.TempDir()
				copyTree(t, dir, sub)
				if err := os.WriteFile(filepath.Join(sub, "vectors", "default", filepath.Base(walPath)), saved[:cut], 0o600); err != nil {
					t.Fatal(err)
				}
				re, err := OpenCollectionStore(sub)
				if err != nil {
					t.Fatalf("log cut at %d: open: %v", cut, err)
				}
				meta, ok := fam.payload(t, re)
				_ = re.Close()
				gotNew := ok && meta["v"].Equal(NewInt(2))
				switch {
				case !ok:
					t.Fatalf("log cut at %d of [%d, %d]: id 1 is gone", cut, before, after)
				case cut == after && !gotNew:
					t.Fatalf("uncut log: id 1 payload = %v, want the new one", meta)
				case cut < after && gotNew:
					t.Fatalf("log cut at %d: id 1 holds the new payload from a torn record", cut)
				}
			}
		})
	}
}
