// SPDX-License-Identifier: Apache-2.0

package objstore

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// probeServer is a conditional-write S3 endpoint whose support probe can be
// made to stall: stallProbePut parks every conditional PUT of a probe object,
// stallProbeDelete parks every DELETE of one. A parked request waits until
// release is closed or the client gives up on it.
type probeServer struct {
	stallProbePut, stallProbeDelete bool
	entered                         chan struct{} // closed when the first parked request arrives
	release                         chan struct{}

	enterOnce sync.Once
	mu        sync.Mutex
	objects   map[string]bool
}

func newProbeServer(t *testing.T) (*probeServer, *S3Store) {
	t.Helper()
	p := &probeServer{entered: make(chan struct{}), release: make(chan struct{}), objects: map[string]bool{}}
	srv := httptest.NewServer(p)
	t.Cleanup(func() {
		select {
		case <-p.release:
		default:
			close(p.release)
		}
		srv.Close()
	})
	return p, newTestStore(t, srv, true, testCreds())
}

func (p *probeServer) park(r *http.Request) {
	p.enterOnce.Do(func() { close(p.entered) })
	select {
	case <-p.release:
	case <-r.Context().Done():
	}
}

func (p *probeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/mybucket/")
	probe := strings.Contains(key, condProbePrefix)
	switch r.Method {
	case http.MethodPut:
		_, _ = io.Copy(io.Discard, r.Body)
		if probe && p.stallProbePut {
			p.park(r)
		}
		p.mu.Lock()
		defer p.mu.Unlock()
		if r.Header.Get("If-None-Match") == "*" && p.objects[key] {
			w.WriteHeader(http.StatusPreconditionFailed)
			return
		}
		p.objects[key] = true
	case http.MethodDelete:
		if probe && p.stallProbeDelete {
			p.park(r)
			if r.Context().Err() != nil {
				return
			}
		}
		p.mu.Lock()
		delete(p.objects, key)
		p.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (p *probeServer) keys() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []string
	for k := range p.objects {
		out = append(out, k)
	}
	return out
}

// TestS3PutIfAbsentCancelledWhileAnotherCallerProbes pins that waiting for
// another caller's support probe honours the waiter's context. One caller's
// probe stalls on the service; a second caller whose context is then cancelled
// must return promptly with the cancellation and write nothing, rather than
// queue behind the stalled probe for as long as it takes.
func TestS3PutIfAbsentCancelledWhileAnotherCallerProbes(t *testing.T) {
	p, s := newProbeServer(t)
	p.stallProbePut = true

	firstDone := make(chan struct{})
	go func() {
		defer close(firstDone)
		_ = s.PutIfAbsent(context.Background(), "t/c/first", strings.NewReader("a"), 1)
	}()
	// Let the stalled probe finish before the test returns, so it does not run on
	// into the next test.
	defer func() {
		close(p.release)
		<-firstDone
	}()
	select {
	case <-p.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the first caller's probe never reached the service")
	}

	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- s.PutIfAbsent(ctx, "t/c/second", strings.NewReader("b"), 1) }()
	time.Sleep(20 * time.Millisecond) // let it reach the wait
	cancel()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("PutIfAbsent = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a cancelled PutIfAbsent stayed blocked behind another caller's stalled probe")
	}
	for _, k := range p.keys() {
		if k == "t/c/second" {
			t.Fatal("the cancelled caller's object was written")
		}
	}
}

// TestS3ProbeCleanupIsBoundedAndReported pins the probe's own cleanup: the
// DELETE of the probe object runs under a deadline of its own, so a stalled
// service cannot hold PutIfAbsent past it, and a probe object that could not be
// removed is reported by name rather than left behind in silence. The verdict
// the probe reached still stands, so the next call writes without probing again.
func TestS3ProbeCleanupIsBoundedAndReported(t *testing.T) {
	old := cleanupTimeout
	cleanupTimeout = 50 * time.Millisecond
	t.Cleanup(func() { cleanupTimeout = old })

	p, s := newProbeServer(t)
	p.stallProbeDelete = true

	errc := make(chan error, 1)
	go func() { errc <- s.PutIfAbsent(context.Background(), "t/c/k", strings.NewReader("v"), 1) }()
	select {
	case err := <-errc:
		if err == nil || !strings.Contains(err.Error(), condProbePrefix) || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("PutIfAbsent = %v, want an error naming the probe object left behind", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("PutIfAbsent hung in the probe's cleanup")
	}

	if err := s.PutIfAbsent(context.Background(), "t/c/k", strings.NewReader("v"), 1); err != nil {
		t.Fatalf("PutIfAbsent after a decided probe = %v, want success", err)
	}
	var probes int
	for _, k := range p.keys() {
		if strings.Contains(k, condProbePrefix) {
			probes++
		}
	}
	if probes != 1 {
		t.Fatalf("%d probe objects on the service, want exactly the one reported (no second probe)", probes)
	}
}
