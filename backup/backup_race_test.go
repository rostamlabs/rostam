// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// arrivalBarrier holds the first two callers until both have arrived, then lets
// every later caller straight through. It is how the race test forces two runs
// to be inside the store at the same step instead of hoping the scheduler lines
// them up. The wait is bounded so a run that never reaches the store fails the
// assertions instead of hanging the test.
type arrivalBarrier struct {
	mu    sync.Mutex
	n     int
	ready chan struct{}
}

func newArrivalBarrier() *arrivalBarrier {
	return &arrivalBarrier{ready: make(chan struct{})}
}

func (b *arrivalBarrier) arrive() {
	b.mu.Lock()
	b.n++
	if b.n == 2 {
		close(b.ready)
	}
	held := b.n <= 2
	b.mu.Unlock()
	if held {
		select {
		case <-b.ready:
		case <-time.After(5 * time.Second):
		}
	}
}

// barrierStore parks each run's FIRST store call — whatever backupOne does
// first, a probe or a write — at a shared barrier, so both runs have taken that
// step before either takes the next one. Every call is gated, so the barrier
// lands on the first one regardless of which method backupOne starts with.
type barrierStore struct {
	objstore.ObjectStore
	b *arrivalBarrier
}

func (s *barrierStore) List(ctx context.Context, prefix string) ([]objstore.ObjectInfo, error) {
	s.b.arrive()
	return s.ObjectStore.List(ctx, prefix)
}

func (s *barrierStore) Put(ctx context.Context, key string, r io.Reader, size int64) error {
	s.b.arrive()
	return s.ObjectStore.Put(ctx, key, r, size)
}

func (s *barrierStore) PutIfAbsent(ctx context.Context, key string, r io.Reader, size int64) error {
	s.b.arrive()
	return s.ObjectStore.PutIfAbsent(ctx, key, r, size)
}

// TestConcurrentBackupsAtOneTimestampPublishOnce pins the write-once rule under
// concurrency: two backup runs at the same timestamp — an admin backup landing
// in the ticker's second in one process, or two processes backing up the same
// prefix — must not both publish. Exactly one run succeeds; the other reports
// ErrSnapshotExists, and the published snapshot/config pair is the winner's.
//
// The "gated" variant forces both runs past their first store call before
// either continues, which is exactly the interleaving a check-then-put misses.
// The "free" variant runs them with no gate, many times.
func TestConcurrentBackupsAtOneTimestampPublishOnce(t *testing.T) {
	backends := []struct {
		name string
		// open returns the two stores the two runs use. They share one backing
		// namespace; for the filesystem that means two independent store values on
		// one root, the way two processes would see it.
		open func(t *testing.T) (objstore.ObjectStore, objstore.ObjectStore)
	}{
		{"mem", func(t *testing.T) (objstore.ObjectStore, objstore.ObjectStore) {
			m := objstore.NewMemStore()
			return m, m
		}},
		{"fs", func(t *testing.T) (objstore.ObjectStore, objstore.ObjectStore) {
			root := t.TempDir()
			a, err := NewFSObjectStore(root)
			if err != nil {
				t.Fatal(err)
			}
			b, err := NewFSObjectStore(root)
			if err != nil {
				t.Fatal(err)
			}
			return a, b
		}},
	}
	for _, be := range backends {
		t.Run(be.name+"/gated", func(t *testing.T) {
			for i := 0; i < 10; i++ {
				a, b := be.open(t)
				bar := newArrivalBarrier()
				runConcurrentBackups(t, &barrierStore{a, bar}, &barrierStore{b, bar})
			}
		})
		t.Run(be.name+"/free", func(t *testing.T) {
			for i := 0; i < 30; i++ {
				a, b := be.open(t)
				runConcurrentBackups(t, a, b)
			}
		})
	}
}

func runConcurrentBackups(t *testing.T, objA, objB objstore.ObjectStore) {
	t.Helper()
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "c", 1, 2, 3)
	opts := BackupOpts{Tenant: "acme", Timestamp: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)}

	var wg sync.WaitGroup
	results := make([][]BackupResult, 2)
	for i, obj := range []objstore.ObjectStore{objA, objB} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], _ = Backup(ctx, store, obj, opts)
		}()
	}
	wg.Wait()

	var winners, refused int
	var won BackupResult
	for i, rs := range results {
		if len(rs) != 1 {
			t.Fatalf("run %d: %d results, want 1", i, len(rs))
		}
		switch r := rs[0]; {
		case r.Err == nil:
			winners++
			won = r
		case errors.Is(r.Err, ErrSnapshotExists):
			refused++
		default:
			t.Fatalf("run %d: unexpected error %v", i, r.Err)
		}
	}
	if winners != 1 || refused != 1 {
		t.Fatalf("winners = %d, refused = %d; want exactly one of each (both runs published)", winners, refused)
	}
	infos, err := objA.List(ctx, won.Key)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 1 || infos[0].Key != won.Key || infos[0].Size != won.Size {
		t.Fatalf("published snapshot = %+v, want the winner's %q (%d bytes)", infos, won.Key, won.Size)
	}
	if got := mustGet(t, objA, cfgKeyFor(won.Key)); got == "" {
		t.Fatal("winner's sibling config is missing or empty")
	}
}
