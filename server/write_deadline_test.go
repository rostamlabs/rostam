// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"io"
	"net"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// startServer builds a server over a fresh single-node store with the given config
// tweaks applied, serving on a random port. Used by the write-deadline tests, which
// need a short WriteTimeout the shared newTestServer helper does not expose.
func startServer(t *testing.T, tweak func(*Config)) (*Server, string) {
	t.Helper()
	store := newTestStore(t)
	cfg := Config{
		Addr: "127.0.0.1:0", Dispatcher: store,
		MaxConns: 100, IdleTimeout: 30 * time.Second,
	}
	if tweak != nil {
		tweak(&cfg)
	}
	srv, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })
	return srv, srv.Addr().String()
}

// sendGetNoRead dials addr, sends a single get request frame for key, and returns
// the connection WITHOUT reading the response — the caller drives a slow/absent
// reader so the server's send path backs up and blocks in Flush.
func sendGetNoRead(t *testing.T, addr string, key []byte) net.Conn {
	t.Helper()
	c, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	w := bufio.NewWriter(c)
	if err := writeFrame(w, EncodeRequest("get", ops.EncodeKeyArgs(key))); err != nil {
		_ = c.Close()
		t.Fatalf("write get frame: %v", err)
	}
	if err := w.Flush(); err != nil {
		_ = c.Close()
		t.Fatalf("flush get frame: %v", err)
	}
	return c
}

// TestServerWriteDeadlineClosesStalledReader proves the response write deadline
// bounds the alias hold: a client that sends a get for a large value and then stops
// reading has its connection aborted within ~WriteTimeout (the server cannot be left
// blocked in Flush indefinitely — which is what pins a zero-copy mmap read alias).
func TestServerWriteDeadlineClosesStalledReader(t *testing.T) {
	const writeTimeout = 200 * time.Millisecond
	srv, addr := startServer(t, func(c *Config) { c.WriteTimeout = writeTimeout })
	_ = srv

	// A value far larger than any socket buffer, so the server MUST block in Flush
	// against a non-reading client (it cannot deliver ~15 MiB into a few MiB of
	// kernel buffers). Only the bytes buffered BEFORE the deadline can ever reach the
	// client, so a cut-off response is a fraction of valLen — the discriminator below.
	const valLen = 15 << 20
	val := make([]byte, valLen)
	for i := range val {
		val[i] = byte(i)
	}
	key := []byte("big")
	if status, _ := rawCall(t, addr, "put", ops.EncodePutArgs(key, val, 0)); status != StatusOK {
		t.Fatalf("put status = %d, want OK", status)
	}

	c := sendGetNoRead(t, addr, key)
	defer func() { _ = c.Close() }()

	// Stall as a non-reading client past the write deadline: the server fills its
	// buffers, blocks in Flush, the deadline fires (~writeTimeout) and it closes the
	// conn. Sleeping >> writeTimeout but << IdleTimeout ensures the closure is the
	// WRITE deadline, not the idle timeout.
	time.Sleep(3 * writeTimeout)

	// Now drain to EOF. The server was cut off mid-write, so only the bytes it
	// buffered before the deadline (a few MiB, bounded by the socket buffers — well
	// under valLen) are deliverable, then the conn is closed (EOF/reset). If the
	// deadline had NOT fired the server would still be live and, as we drain, would
	// resume and deliver the FULL valLen before blocking again — so a near-full read
	// (or a drain that never reaches EOF within the deadline) is the failure signal.
	_ = c.SetReadDeadline(time.Now().Add(8 * time.Second))
	start := time.Now()
	n, _ := io.Copy(io.Discard, c)
	elapsed := time.Since(start)
	if n >= valLen/2 {
		t.Fatalf("read %d bytes (>= half of %d): server was NOT cut off — write deadline did not fire", n, valLen)
	}
	if elapsed >= 8*time.Second {
		t.Fatalf("drain took %v (hit the client read deadline): conn was not closed by the server", elapsed)
	}
}

// TestServerCloseCompletesWithStalledWriter is the cleanup-hang gate. A stalled
// reader leaves the connection's writer goroutine blocked in Flush; before the write
// deadline this wedged the per-conn cleanup (writerWG.Wait) and therefore Server.Close
// forever. With the deadline the write aborts, the writer drains and returns, so
// Close completes. We assert Close returns well within a bound.
func TestServerCloseCompletesWithStalledWriter(t *testing.T) {
	const writeTimeout = 200 * time.Millisecond
	srv, addr := startServer(t, func(c *Config) { c.WriteTimeout = writeTimeout })

	const valLen = 8 << 20
	val := make([]byte, valLen)
	key := []byte("big")
	if status, _ := rawCall(t, addr, "put", ops.EncodePutArgs(key, val, 0)); status != StatusOK {
		t.Fatalf("put status = %d, want OK", status)
	}

	c := sendGetNoRead(t, addr, key)
	defer func() { _ = c.Close() }()

	// Give the server a moment to pick up the request and block in Flush against the
	// non-reading client, so Close races a genuinely wedged writer.
	time.Sleep(2 * writeTimeout)

	done := make(chan error, 1)
	go func() { done <- srv.Close() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Close returned error: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Server.Close hung with a stalled writer — write deadline did not unblock cleanup")
	}
}
