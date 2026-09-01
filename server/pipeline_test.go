// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// echoDispatcher answers every op with its args and records peak concurrency.
// gate, when non-nil, blocks each Call until released so a test can prove
// pipelined requests dispatch concurrently.
type echoDispatcher struct {
	gate    chan struct{}
	cur     atomic.Int64
	peak    atomic.Int64
	nilHint string
}

func (d *echoDispatcher) Call(op string, args []byte) ([]byte, error) {
	cur := d.cur.Add(1)
	for {
		p := d.peak.Load()
		if cur <= p || d.peak.CompareAndSwap(p, cur) {
			break
		}
	}
	if d.gate != nil {
		<-d.gate
	}
	d.cur.Add(-1)
	out := append([]byte("echo:"), args...)
	return out, nil
}

func (d *echoDispatcher) LeaderAddr() string { return d.nilHint }

// pipeSend writes one v1 request frame for op with args.
func pipeSend(t *testing.T, w *bufio.Writer, op string, args []byte) {
	t.Helper()
	body := EncodeRequest(op, args)
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
	if _, err := w.Write(hdr[:]); err != nil {
		t.Fatalf("write hdr: %v", err)
	}
	if _, err := w.Write(body); err != nil {
		t.Fatalf("write body: %v", err)
	}
}

// pipeRecv reads one response frame and returns (status, payload).
func pipeRecv(t *testing.T, r *bufio.Reader) (uint8, []byte) {
	t.Helper()
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		t.Fatalf("read resp len: %v", err)
	}
	n := binary.BigEndian.Uint32(hdr[:])
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		t.Fatalf("read resp body: %v", err)
	}
	if len(body) < 5 {
		t.Fatalf("short response body: %d", len(body))
	}
	plen := binary.BigEndian.Uint32(body[1:5])
	return body[0], body[5 : 5+plen]
}

func startPipelineTestServer(t *testing.T, d Dispatcher) (addr string, closeFn func()) {
	t.Helper()
	srv, err := New(Config{Addr: "127.0.0.1:0", Dispatcher: d})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	go func() { _ = srv.Serve() }()
	return srv.Addr().String(), func() { _ = srv.Close() }
}

// startPipelineTestServerCompaction starts a server with EnableOnlineCompaction on,
// so the pipelined path COPIES each payload out of its cache alias and enforces the
// per-connection byte budget — the opted-in configuration the recycle-integrity and
// byte-budget tests exercise. WriteTimeout is set short so a wedged writer aborts the
// connection quickly instead of pinning the test for the 30s default.
func startPipelineTestServerCompaction(t *testing.T, d Dispatcher, writeTimeout time.Duration) (addr string, closeFn func()) {
	t.Helper()
	srv, err := New(Config{
		Addr:                   "127.0.0.1:0",
		Dispatcher:             d,
		EnableOnlineCompaction: true,
		WriteTimeout:           writeTimeout,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	go func() { _ = srv.Serve() }()
	return srv.Addr().String(), func() { _ = srv.Close() }
}

// TestPipelinedRequestsAnswerInOrder sends a burst of requests on ONE conn
// without reading responses, then asserts every response arrives in request
// order with the right payload.
func TestPipelinedRequestsAnswerInOrder(t *testing.T) {
	d := &echoDispatcher{}
	addr, stop := startPipelineTestServer(t, d)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer c.Close()
	w := bufio.NewWriter(c)
	r := bufio.NewReader(c)

	const burst = 200 // > connPipelineWindow: exercises window backpressure too
	done := make(chan error, 1)
	go func() {
		for i := 0; i < burst; i++ {
			pipeSend(t, w, "noop", []byte(fmt.Sprintf("req-%03d", i)))
		}
		done <- w.Flush()
	}()
	for i := 0; i < burst; i++ {
		status, payload := pipeRecv(t, r)
		if status != StatusOK {
			t.Fatalf("resp %d: status %d", i, status)
		}
		want := fmt.Sprintf("echo:req-%03d", i)
		if string(payload) != want {
			t.Fatalf("resp %d: got %q want %q — responses out of request order", i, payload, want)
		}
	}
	if err := <-done; err != nil {
		t.Fatalf("send flush: %v", err)
	}
}

// TestPipelinedRequestsDispatchConcurrently proves the point of the feature:
// with the dispatcher gated, a pipelining client gets >1 request in flight on
// one connection (the old serial loop's peak was exactly 1).
func TestPipelinedRequestsDispatchConcurrently(t *testing.T) {
	d := &echoDispatcher{gate: make(chan struct{})}
	addr, stop := startPipelineTestServer(t, d)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer c.Close()
	w := bufio.NewWriter(c)
	r := bufio.NewReader(c)

	const burst = 16
	for i := 0; i < burst; i++ {
		pipeSend(t, w, "noop", []byte{byte(i)})
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for d.peak.Load() < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := d.peak.Load(); got < 2 {
		t.Fatalf("peak in-flight dispatches = %d, want >= 2 (pipelining never overlapped)", got)
	}
	close(d.gate)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < burst; i++ {
			status, payload := pipeRecv(t, r)
			if status != StatusOK || len(payload) != 6 || payload[5] != byte(i) {
				t.Errorf("resp %d: status=%d payload=%v", i, status, payload)
				return
			}
		}
	}()
	wg.Wait()
}

// TestNonPipeliningClientUnchanged pins the inline fast path: strict
// request-response traffic still round-trips correctly (and, per the
// implementation, without ever entering the pipelined machinery).
func TestNonPipeliningClientUnchanged(t *testing.T) {
	d := &echoDispatcher{}
	addr, stop := startPipelineTestServer(t, d)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer c.Close()
	w := bufio.NewWriter(c)
	r := bufio.NewReader(c)
	for i := 0; i < 20; i++ {
		pipeSend(t, w, "noop", []byte{byte(i)})
		if err := w.Flush(); err != nil {
			t.Fatalf("flush: %v", err)
		}
		status, payload := pipeRecv(t, r)
		if status != StatusOK || payload[5] != byte(i) {
			t.Fatalf("resp %d: status=%d payload=%v", i, status, payload)
		}
	}
	if got := d.peak.Load(); got != 1 {
		t.Fatalf("peak concurrency = %d, want exactly 1 for a strict request-response client", got)
	}
}
