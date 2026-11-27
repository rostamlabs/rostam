// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// An eviction snapshots the collection, uploads the snapshot, then replaces the
// catalog entry with a cold stub. A write acknowledged after the snapshot used
// to land on the collection being retired and be missing from the cold object,
// so the next access, which promotes from that object, no longer had it.

type evictMode struct {
	name    string
	cfg     Config
	durable bool
}

// evictModes is every restoreReplaceModes mode, plus a listed durable one.
func evictModes() []evictMode {
	var modes []evictMode
	for _, m := range restoreReplaceModes {
		modes = append(modes, evictMode{m.name, m.cfg, false})
	}
	return append(modes, evictMode{"durable", restoreReplaceModes[0].cfg, true})
}

func openEvictStore(t *testing.T, durable bool) *CollectionStore {
	t.Helper()
	if durable {
		cs := openDurable(t, t.TempDir(), "docs")
		t.Cleanup(func() { _ = cs.Close() })
		return cs
	}
	return newCollectionStore(t)
}

// TestEvictKeepsWriteMadeDuringUpload: a write that completes between the
// snapshot and the commit is not lost.
func TestEvictKeepsWriteMadeDuringUpload(t *testing.T) {
	for _, m := range evictModes() {
		t.Run(m.name, func(t *testing.T) {
			s := openEvictStore(t, m.durable)
			if err := s.CreateCollection("docs", m.cfg); err != nil {
				t.Fatal(err)
			}
			insertIDs(t, s, "docs", restoreIDs(1, 20))
			evictCommitHook = func() { insertIDs(t, s, "docs", []uint64{999}) }
			t.Cleanup(func() { evictCommitHook = nil })

			err := s.EvictCollection(context.Background(), "docs", objstore.NewMemStore(), "t", time.Unix(1, 0))
			evictCommitHook = nil
			if err != nil && !errors.Is(err, ErrEvictBusy) {
				t.Fatalf("evict: %v", err)
			}
			requirePoints(t, s, "docs", append(restoreIDs(1, 20), 999), nil)
		})
	}
}

// TestEvictKeepsWriteFromEarlierHolder: a caller that acquired the collection
// before the eviction and writes after its snapshot is not lost either.
func TestEvictKeepsWriteFromEarlierHolder(t *testing.T) {
	for _, m := range evictModes() {
		t.Run(m.name, func(t *testing.T) {
			s := openEvictStore(t, m.durable)
			if err := s.CreateCollection("docs", m.cfg); err != nil {
				t.Fatal(err)
			}
			insertIDs(t, s, "docs", restoreIDs(1, 20))
			held, ok := s.Acquire("docs")
			if !ok {
				t.Fatal("collection missing")
			}
			entered := make(chan struct{})
			evictCommitHook = func() { close(entered) }
			t.Cleanup(func() { evictCommitHook = nil })
			evicted := make(chan error, 1)
			go func() {
				evicted <- s.EvictCollection(context.Background(), "docs", objstore.NewMemStore(), "t", time.Unix(1, 0))
			}()
			select {
			case <-entered:
			case err := <-evicted:
				evicted <- err
			}
			if err := held.Insert(999, restoreVec(999), 0, nil, nil); err != nil {
				t.Fatalf("write through the earlier handle: %v", err)
			}
			held.Release()
			if err := <-evicted; err != nil && !errors.Is(err, ErrEvictBusy) {
				t.Fatalf("evict: %v", err)
			}
			requirePoints(t, s, "docs", append(restoreIDs(1, 20), 999), nil)
		})
	}
}

// TestSweepColdSkipsBusyCollection: the idle sweep leaves a collection that is
// in use hot, without an error, so the next sweep can try again.
func TestSweepColdSkipsBusyCollection(t *testing.T) {
	s := newCollectionStore(t)
	if err := s.CreateCollection("docs", restoreReplaceModes[0].cfg); err != nil {
		t.Fatal(err)
	}
	insertIDs(t, s, "docs", restoreIDs(1, 5))
	held, ok := s.Acquire("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	held.lastAccess.Store(time.Unix(1, 0).UnixNano())
	evicted, err := s.SweepCold(time.Unix(1000, 0), time.Second, objstore.NewMemStore(), "t")
	held.Release()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(evicted) != 0 || s.IsCold("docs") {
		t.Fatalf("a collection in use was evicted: %v", evicted)
	}
}

