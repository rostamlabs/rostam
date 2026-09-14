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
)

// cancelOnRead cancels its context on the first Read, then yields its data and
// EOF: a PutIfAbsent whose caller gives up while the body is being consumed.
type cancelOnRead struct {
	cancel context.CancelFunc
	data   []byte
	done   bool
}

func (c *cancelOnRead) Read(p []byte) (int, error) {
	if c.done {
		return 0, io.EOF
	}
	c.cancel()
	c.done = true
	return copy(p, c.data), nil
}

// TestPutIfAbsentCancelledMidBodyPublishesNothing pins that a PutIfAbsent whose
// context is cancelled while its body is being read returns the cancellation
// promptly and publishes nothing, on every backend. A backend that checks the
// context only on entry would buffer the body and then publish it anyway.
func TestPutIfAbsentCancelledMidBodyPublishesNothing(t *testing.T) {
	backends := map[string]func(t *testing.T) objstore.ObjectStore{
		"mem": func(t *testing.T) objstore.ObjectStore { return objstore.NewMemStore() },
		"fs": func(t *testing.T) objstore.ObjectStore {
			s, err := NewFSObjectStore(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			return s
		},
		"s3": func(t *testing.T) objstore.ObjectStore {
			_, s := newFakeS3(t, "honor")
			return s
		},
	}
	for name, open := range backends {
		t.Run(name, func(t *testing.T) {
			obj := open(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			body := []byte("cancelled before publish")
			errc := make(chan error, 1)
			go func() {
				errc <- obj.PutIfAbsent(ctx, "t/c/k", &cancelOnRead{cancel: cancel, data: body}, int64(len(body)))
			}()
			select {
			case err := <-errc:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("PutIfAbsent = %v, want context.Canceled", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("PutIfAbsent did not return after its context was cancelled")
			}
			if _, err := obj.Get(context.Background(), "t/c/k"); !errors.Is(err, objstore.ErrNotFound) {
				t.Fatalf("a cancelled PutIfAbsent published its object (Get = %v)", err)
			}
		})
	}
}

// stuckDeleteStore never completes a Delete until the Delete's context ends.
type stuckDeleteStore struct {
	*objstore.MemStore
}

func (s *stuckDeleteStore) Delete(ctx context.Context, key string) error {
	<-ctx.Done()
	return ctx.Err()
}

// TestBackupClaimReleaseIsBoundedAndReported pins the two properties of
// releasing a timestamp claim. The release must not hang a backup on a store
// that stalls: it runs detached from the caller's cancellation, so without its
// own deadline nothing would ever end it. And a release that fails must be
// reported: here the run found a snapshot with no config of its own at its key,
// so the config it claimed would otherwise stay silently paired with a snapshot
// it does not describe.
func TestBackupClaimReleaseIsBoundedAndReported(t *testing.T) {
	old := cleanupTimeout
	cleanupTimeout = 50 * time.Millisecond
	t.Cleanup(func() { cleanupTimeout = old })

	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "c", 1, 2)
	obj := &stuckDeleteStore{objstore.NewMemStore()}
	opts := BackupOpts{Tenant: "acme", Timestamp: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)}
	key := snapshotKey(opts.Tenant, "default/c", opts.Timestamp)
	if err := obj.Put(ctx, key, strings.NewReader("config-less"), 11); err != nil {
		t.Fatal(err)
	}

	errc := make(chan error, 1)
	go func() {
		_, err := Backup(ctx, store, obj, opts)
		errc <- err
	}()
	select {
	case err := <-errc:
		if !errors.Is(err, ErrSnapshotExists) {
			t.Fatalf("Backup = %v, want ErrSnapshotExists", err)
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Backup = %v; the failed release of the claimed config was not reported", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Backup hung in the release of its claim")
	}
}
