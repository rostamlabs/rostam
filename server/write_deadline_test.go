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

// dialStalled dials addr and pins the client's TCP receive buffer to 64 KiB, so that a
// non-reading client cannot let the kernel absorb a multi-MiB response: the server's
// send path fills and it genuinely blocks in Flush. Without pinning SO_RCVBUF,
// receive-window autotuning can grow the client buffer past even a 15 MiB response on
// some hosts, and the write would never stall (coderabbit). 64 KiB is small enough that
// the deliverable-before-deadline bytes (this buffer plus the server's send buffer) stay
// well under the tests' valLen/2 discriminator, yet above the tiny-SO_RCVBUF floor where
// window scaling misbehaves and the deadline stops firing.
func dialStalled(t *testing.T, addr string) net.Conn {
	t.Helper()
	c, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	if tc, ok := c.(*net.TCPConn); ok {
		_ = tc.SetReadBuffer(64 << 10)
	}
	return c
}

// sendGetNoRead dials addr, sends a single get request frame for key, and returns
// the connection WITHOUT reading the response — the caller drives a slow/absent
// reader so the server's send path backs up and blocks in Flush.
func sendGetNoRead(t *testing.T, addr string, key []byte) net.Conn {
	t.Helper()
	c := dialStalled(t, addr)
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

// sendGetsNoRead writes `count` get frames for the same key in ONE buffered flush and
// returns the still-open, non-reading conn. Sending two or more frames together forces
// the server onto the PIPELINED path DETERMINISTICALLY: the first frame's read leaves
// the later frame(s) buffered (r.Buffered() > 0), and once the first large response
// stalls the writer, every following frame sees outstanding > 0. A single small get is
// only *usually* pipelined — whether the tiny body is buffered alongside its header
// depends on TCP segmentation — so a test that must exercise the ordered-writer goroutine
// (its writerWG cleanup) sends a pair rather than relying on that timing.
func sendGetsNoRead(t *testing.T, addr string, key []byte, count int) net.Conn {
	t.Helper()
	c := dialStalled(t, addr)
	w := bufio.NewWriter(c)
	for i := 0; i < count; i++ {
		if err := writeFrame(w, EncodeRequest("get", ops.EncodeKeyArgs(key))); err != nil {
			_ = c.Close()
			t.Fatalf("write get frame %d: %v", i, err)
		}
	}
	if err := w.Flush(); err != nil {
		_ = c.Close()
		t.Fatalf("flush get frames: %v", err)
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

// TestServerCloseCompletesWithStalledWriter is the cleanup-hang gate. A stalled reader
// leaves the connection's writer goroutine blocked in Flush; before the write deadline
// this wedged the per-conn cleanup (writerWG.Wait) forever. This test deliberately does
// NOT let Server.Close's force-close be what unblocks the writer — otherwise it would pass
// even with the deadline code deleted (cubic P2). Instead it first proves, WITHOUT calling
// Close, that the WRITE DEADLINE alone unblocks the stalled (pipelined) writer, THEN checks
// Close returns promptly with the writer already drained.
func TestServerCloseCompletesWithStalledWriter(t *testing.T) {
	const writeTimeout = 200 * time.Millisecond
	srv, addr := startServer(t, func(c *Config) { c.WriteTimeout = writeTimeout })

	// Each response is > any socket buffer, so the server MUST block in Flush against a
	// non-reading client. Sending TWO gets in one write forces the PIPELINED path
	// deterministically — the ordered writer goroutine whose writerWG.Wait cleanup is the
	// historic hang site — rather than depending on whether a single small request's body
	// happens to be buffered with its header (cubic P2).
	const valLen = 15 << 20
	val := make([]byte, valLen)
	key := []byte("big")
	if status, _ := rawCall(t, addr, "put", ops.EncodePutArgs(key, val, 0)); status != StatusOK {
		t.Fatalf("put status = %d, want OK", status)
	}

	c := sendGetsNoRead(t, addr, key, 2)
	defer func() { _ = c.Close() }()

	// Stall as a NON-reading client past the write deadline: the server fills its buffers,
	// blocks in Flush, the deadline fires (~writeTimeout) and the writer aborts and closes
	// the conn — all WITHOUT us calling Server.Close. Sleeping >> writeTimeout but <<
	// IdleTimeout ensures the closure is the WRITE deadline, not the idle timeout. (Draining
	// without this stall would turn the client into a reader and the write would never
	// block, so the deadline would never be the thing under test.)
	time.Sleep(3 * writeTimeout)

	// Now drain to EOF. Because the writer was cut off by the deadline, only the pre-deadline
	// bytes (bounded by the socket buffers, well under valLen) are deliverable, then EOF. If
	// the deadline were absent the writer would still be wedged and, as we drain, resume and
	// deliver the FULL valLen before blocking again — so a near-full read (or a drain that
	// never reaches EOF) is the failure signal. This makes the test actually gate the
	// deadline rather than rely on Close's force-close. Reaching EOF also means the per-conn
	// writerWG.Wait cleanup has already completed.
	_ = c.SetReadDeadline(time.Now().Add(8 * time.Second))
	start := time.Now()
	n, _ := io.Copy(io.Discard, c)
	if elapsed := time.Since(start); elapsed >= 8*time.Second {
		t.Fatalf("drain took %v (hit the client read deadline): stalled writer was NOT unblocked by the deadline", elapsed)
	}
	if n >= valLen/2 {
		t.Fatalf("read %d bytes (>= half of %d): writer was NOT cut off by the write deadline", n, valLen)
	}

	// With the writer already unblocked by the deadline, Server.Close must return promptly.
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
