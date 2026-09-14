// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam/objstore"
)

// fakeS3 is a minimal S3 endpoint for PutIfAbsent tests: object PUT/GET/DELETE
// over path-style URLs, with the create-if-absent check made under one lock so
// it is atomic the way the real service's is. mode selects how it treats an
// If-None-Match: * PUT:
//
//   - "honor": 412 when the key exists (what AWS S3 and current compatibles do);
//   - "ignore": overwrite as if the header were absent (the dangerous
//     compatible, which accepts the request and silently replaces the object);
//   - "unimplemented": 501 NotImplemented for any conditional PUT.
type fakeS3 struct {
	mode string

	mu      sync.Mutex
	objects map[string][]byte
	// condPuts counts conditional PUTs and unsignedCond counts those whose
	// signature did not cover If-None-Match.
	condPuts, unsignedCond int
	// conns counts TCP connections the server accepted. A client that leaves a
	// response body unclosed cannot reuse its connection, so it shows up here.
	conns atomic.Int64
}

// deniedKey is refused with 403 on PUT, whatever the mode.
const deniedKey = "t/c/denied"

func newFakeS3(t *testing.T, mode string) (*fakeS3, *objstore.S3Store) {
	t.Helper()
	f := &fakeS3{mode: mode, objects: map[string][]byte{}}
	srv := httptest.NewUnstartedServer(f)
	srv.Config.ConnState = func(_ net.Conn, st http.ConnState) {
		if st == http.StateNew {
			f.conns.Add(1)
		}
	}
	srv.Start()
	t.Cleanup(srv.Close)
	s, err := objstore.NewS3Store(objstore.Config{
		Endpoint:   srv.URL,
		Region:     "us-east-1",
		Bucket:     "bkt",
		Creds:      objstore.Credentials{AccessKeyID: "AKID", SecretAccessKey: "secret"},
		PathStyle:  true,
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return f, s
}

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/bkt/")
	switch r.Method {
	case http.MethodPut:
		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if key == deniedKey {
			w.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(w, "<Error><Code>AccessDenied</Code><Message>denied</Message></Error>")
			return
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		if r.Header.Get("If-None-Match") == "*" {
			f.condPuts++
			if !strings.Contains(r.Header.Get("Authorization"), "if-none-match") {
				f.unsignedCond++
			}
			switch f.mode {
			case "unimplemented":
				w.WriteHeader(http.StatusNotImplemented)
				_, _ = io.WriteString(w, "<Error><Code>NotImplemented</Code><Message>no</Message></Error>")
				return
			case "honor":
				if _, ok := f.objects[key]; ok {
					w.WriteHeader(http.StatusPreconditionFailed)
					_, _ = io.WriteString(w, "<Error><Code>PreconditionFailed</Code><Message>exists</Message></Error>")
					return
				}
			}
		}
		f.objects[key] = body
		w.WriteHeader(http.StatusOK)
		// A body on success too, which the client has no use for: a client that
		// neither reads nor closes it cannot put the connection back in the pool.
		_, _ = io.WriteString(w, "<PutObjectResult/>")
	case http.MethodGet:
		f.mu.Lock()
		b, ok := f.objects[key]
		f.mu.Unlock()
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write(b)
	case http.MethodDelete:
		f.mu.Lock()
		delete(f.objects, key)
		f.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (f *fakeS3) keys() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for k := range f.objects {
		out = append(out, k)
	}
	sortStrings(out)
	return out
}

// TestPutIfAbsentPerBackend runs the same write-once contract against every
// ObjectStore implementation. A backend whose PutIfAbsent silently overwrites —
// or that is merely check-then-put — fails the "present" or the "concurrent"
// case.
func TestPutIfAbsentPerBackend(t *testing.T) {
	backends := []struct {
		name string
		open func(t *testing.T) objstore.ObjectStore
		// after, if set, checks backend-specific leftovers once the cases ran.
		after func(t *testing.T, obj objstore.ObjectStore)
	}{
		{name: "mem", open: func(t *testing.T) objstore.ObjectStore { return objstore.NewMemStore() }},
		{
			name: "fs",
			open: func(t *testing.T) objstore.ObjectStore {
				s, err := NewFSObjectStore(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				return s
			},
			after: func(t *testing.T, obj objstore.ObjectStore) {
				dir := filepath.Join(obj.(*FSObjectStore).root, "t", "c")
				if tmps := listTempFiles(t, dir); len(tmps) != 0 {
					t.Fatalf("staging files left behind: %v", tmps)
				}
			},
		},
		{
			name: "s3",
			open: func(t *testing.T) objstore.ObjectStore {
				_, s := newFakeS3(t, "honor")
				return s
			},
		},
	}
	for _, be := range backends {
		t.Run(be.name, func(t *testing.T) {
			ctx := context.Background()
			obj := be.open(t)

			t.Run("absent", func(t *testing.T) {
				if err := putIfAbsentString(ctx, obj, "t/c/a", "first"); err != nil {
					t.Fatalf("PutIfAbsent on an absent key: %v", err)
				}
				if got := mustGet(t, obj, "t/c/a"); got != "first" {
					t.Fatalf("stored %q, want %q", got, "first")
				}
			})
			t.Run("present", func(t *testing.T) {
				if err := putIfAbsentString(ctx, obj, "t/c/a", "second"); !errors.Is(err, objstore.ErrExists) {
					t.Fatalf("PutIfAbsent on a present key = %v, want ErrExists", err)
				}
				if got := mustGet(t, obj, "t/c/a"); got != "first" {
					t.Fatalf("existing object became %q; PutIfAbsent overwrote it", got)
				}
				if err := obj.Put(ctx, "t/c/b", strings.NewReader("by put"), 6); err != nil {
					t.Fatal(err)
				}
				if err := putIfAbsentString(ctx, obj, "t/c/b", "later"); !errors.Is(err, objstore.ErrExists) {
					t.Fatalf("PutIfAbsent over a Put object = %v, want ErrExists", err)
				}
				if got := mustGet(t, obj, "t/c/b"); got != "by put" {
					t.Fatalf("existing object became %q; PutIfAbsent overwrote it", got)
				}
			})
			t.Run("concurrent", func(t *testing.T) {
				const rounds, writers = 40, 12
				for round := 0; round < rounds; round++ {
					key := fmt.Sprintf("t/c/race-%d", round)
					payloads := make([][]byte, writers)
					errs := make([]error, writers)
					start := make(chan struct{})
					var wg sync.WaitGroup
					for w := range writers {
						// Distinct, non-trivial payloads, so a torn or interleaved
						// write would not happen to equal any single writer's bytes.
						payloads[w] = bytes.Repeat([]byte(fmt.Sprintf("writer-%02d|", w)), 2048)
						wg.Add(1)
						go func() {
							defer wg.Done()
							<-start
							errs[w] = obj.PutIfAbsent(ctx, key, bytes.NewReader(payloads[w]), int64(len(payloads[w])))
						}()
					}
					close(start)
					wg.Wait()

					winner := -1
					for w, err := range errs {
						switch {
						case err == nil:
							if winner >= 0 {
								t.Fatalf("round %d: writers %d and %d both succeeded", round, winner, w)
							}
							winner = w
						case errors.Is(err, objstore.ErrExists):
						default:
							t.Fatalf("round %d: writer %d: unexpected error %v", round, w, err)
						}
					}
					if winner < 0 {
						t.Fatalf("round %d: no writer succeeded", round)
					}
					if got := mustGet(t, obj, key); got != string(payloads[winner]) {
						t.Fatalf("round %d: stored object is not writer %d's payload intact (%d bytes)", round, winner, len(got))
					}
				}
			})
			if be.after != nil {
				be.after(t, obj)
			}
		})
	}
}

func putIfAbsentString(ctx context.Context, obj objstore.ObjectStore, key, s string) error {
	return obj.PutIfAbsent(ctx, key, strings.NewReader(s), int64(len(s)))
}

// TestS3PutIfAbsentSignsTheCondition pins that the conditional PUT carries
// If-None-Match: * and that the signature covers it, so it cannot be stripped
// in transit without failing verification.
func TestS3PutIfAbsentSignsTheCondition(t *testing.T) {
	f, s := newFakeS3(t, "honor")
	if err := putIfAbsentString(context.Background(), s, "t/c/k", "v"); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.condPuts == 0 {
		t.Fatal("no PUT carried If-None-Match: *")
	}
	if f.unsignedCond != 0 {
		t.Fatalf("%d conditional PUTs did not sign If-None-Match", f.unsignedCond)
	}
}

// TestS3PutIfAbsentRefusesUnconditionalBackend pins the refusal on an
// S3-compatible store that cannot guarantee a conditional create. A store that
// ignores If-None-Match would otherwise answer 200 and replace the object, so
// PutIfAbsent must detect that before writing the real key, report
// ErrConditionalWriteUnsupported, leave an existing object untouched, write
// nothing under an absent key, and not leave its probe behind.
func TestS3PutIfAbsentRefusesUnconditionalBackend(t *testing.T) {
	for _, mode := range []string{"ignore", "unimplemented"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			f, s := newFakeS3(t, mode)
			if err := s.Put(ctx, "t/c/have", strings.NewReader("old"), 3); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 2; i++ { // the refusal is sticky, not a one-off
				if err := putIfAbsentString(ctx, s, "t/c/have", "new"); !errors.Is(err, objstore.ErrConditionalWriteUnsupported) {
					t.Fatalf("PutIfAbsent over an existing key = %v, want ErrConditionalWriteUnsupported", err)
				}
				if err := putIfAbsentString(ctx, s, "t/c/absent", "new"); !errors.Is(err, objstore.ErrConditionalWriteUnsupported) {
					t.Fatalf("PutIfAbsent on an absent key = %v, want ErrConditionalWriteUnsupported", err)
				}
			}
			if got := mustGet(t, s, "t/c/have"); got != "old" {
				t.Fatalf("existing object became %q", got)
			}
			if got := f.keys(); len(got) != 1 || got[0] != "t/c/have" {
				t.Fatalf("objects after refusal = %v, want only the pre-existing key", got)
			}
		})
	}
}

// TestS3PutsReuseTheirConnection pins that every PUT path — success, the 412
// that becomes ErrExists, a plain error status, and the one-time support probe
// with its delete — closes the response body it was handed. An unclosed body
// keeps its connection out of the pool, so a sequence of calls would open a new
// connection each time instead of reusing one.
func TestS3PutsReuseTheirConnection(t *testing.T) {
	ctx := context.Background()
	f, s := newFakeS3(t, "honor")
	for i := 0; i < 8; i++ {
		if err := putIfAbsentString(ctx, s, fmt.Sprintf("t/c/new-%d", i), "v"); err != nil {
			t.Fatal(err)
		}
		if err := putIfAbsentString(ctx, s, "t/c/new-0", "v"); !errors.Is(err, objstore.ErrExists) {
			t.Fatalf("PutIfAbsent over an existing key = %v, want ErrExists", err)
		}
		if err := s.Put(ctx, "t/c/new-0", strings.NewReader("w"), 1); err != nil {
			t.Fatal(err)
		}
		// Any other error status, on both PUT paths.
		if err := s.Put(ctx, deniedKey, strings.NewReader("w"), 1); err == nil {
			t.Fatal("Put of the denied key unexpectedly succeeded")
		}
		if err := putIfAbsentString(ctx, s, deniedKey, "w"); err == nil || errors.Is(err, objstore.ErrExists) {
			t.Fatalf("PutIfAbsent of the denied key = %v, want a plain error", err)
		}
	}
	if n := f.conns.Load(); n != 1 {
		t.Fatalf("the client opened %d connections for sequential PUTs, want 1: a response body was left unclosed", n)
	}
}
