// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"errors"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- fakes -------------------------------------------------------------------

type nopDispatcher struct{}

func (nopDispatcher) Call(string, []byte) ([]byte, error) { return nil, nil }
func (nopDispatcher) LeaderAddr() string                  { return "" }

// gateConn is a net.Conn whose Read blocks until the conn is closed. A handler
// that gets one parks in Read, so "did Read happen" is an exact, edge-triggered
// answer to "did a handler goroutine start on this connection".
type gateConn struct {
	readCalled chan struct{}
	closed     chan struct{}
	closeOnce  sync.Once
	readOnce   sync.Once
}

func newGateConn() *gateConn {
	return &gateConn{readCalled: make(chan struct{}), closed: make(chan struct{})}
}

func (c *gateConn) Read([]byte) (int, error) {
	c.readOnce.Do(func() { close(c.readCalled) })
	<-c.closed
	return 0, errors.New("gateConn closed")
}
func (c *gateConn) Write(b []byte) (int, error) { return len(b), nil }
func (c *gateConn) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return nil
}
func (c *gateConn) LocalAddr() net.Addr              { return fakeAddr{} }
func (c *gateConn) RemoteAddr() net.Addr             { return fakeAddr{} }
func (c *gateConn) SetDeadline(time.Time) error      { return nil }
func (c *gateConn) SetReadDeadline(time.Time) error  { return nil }
func (c *gateConn) SetWriteDeadline(time.Time) error { return nil }

// wasHandled reports whether a handler goroutine reached Read within d.
func (c *gateConn) wasHandled(d time.Duration) bool {
	select {
	case <-c.readCalled:
		return true
	case <-time.After(d):
		return false
	}
}

type fakeAddr struct{}

func (fakeAddr) Network() string { return "fake" }
func (fakeAddr) String() string  { return "fake" }

// gateListener hands connections to Accept only when the test says so, which is
// what makes the shutdown interleaving below exact rather than sampled.
//
// Close deliberately does NOT unblock Accept. That models the one ordering a
// real listener also permits and that this regression is about: the kernel has
// already handed a connection to an Accept that is mid-flight when Close lands,
// so Accept still yields a connection after the listener was closed. The test
// calls unblock explicitly when it wants Accept to start failing.
type gateListener struct {
	gate   chan net.Conn
	stop   chan struct{}
	once   sync.Once
	closed atomic.Bool
}

func newGateListener() *gateListener {
	return &gateListener{gate: make(chan net.Conn), stop: make(chan struct{})}
}

func (l *gateListener) Accept() (net.Conn, error) {
	select {
	case c := <-l.gate:
		return c, nil
	case <-l.stop:
		return nil, errors.New("gateListener closed")
	}
}
func (l *gateListener) Close() error {
	l.closed.Store(true)
	return nil
}
func (l *gateListener) unblock()       { l.once.Do(func() { close(l.stop) }) }
func (l *gateListener) Addr() net.Addr { return fakeAddr{} }

// newGatedServer builds a Server around a listener the test drives directly. It
// bypasses New (which binds a real socket) so the accept/shutdown ordering can
// be staged step by step instead of sampled under load.
func newGatedServer(t *testing.T) (*Server, *gateListener) {
	t.Helper()
	cfg := Config{Addr: "fake", Dispatcher: nopDispatcher{}, MaxConns: 16}
	cfg.applyDefaults()
	ln := newGateListener()
	return &Server{
		cfg:     cfg,
		ln:      ln,
		connSem: make(chan struct{}, cfg.MaxConns),
		conns:   make(map[net.Conn]struct{}),
		closeCh: make(chan struct{}),
	}, ln
}

// --- the regression ----------------------------------------------------------

// TestCloseRefusesConnectionAcceptedDuringShutdown pins the accept/shutdown
// fence deterministically.
//
// Close's contract is that it "stops accepting new connections, force-closes any
// in-flight connections, and waits for their handler goroutines to return". A
// connection whose Accept lands after Close has walked s.conns can satisfy none
// of it: Close will never force-close that conn (it was not in the map when
// Close looked), and when the handler is counted in s.wg only AFTER the conn is
// published, Close's Wait can observe a zero counter and return while the
// handler is only just starting.
//
// The staging here is that worst case made exact: Close runs to completion with
// the accept loop parked inside Accept, and only then does a connection arrive.
// The accept path must drop it without starting a handler. With the admission
// step outside the shutdown lock this failed on every run — it is not sampling a
// window.
func TestCloseRefusesConnectionAcceptedDuringShutdown(t *testing.T) {
	srv, ln := newGatedServer(t)

	var served sync.WaitGroup
	served.Add(1)
	go func() { defer served.Done(); _ = srv.Serve() }()

	// Serve is parked in Accept with nothing admitted, so Close walks an empty
	// conns map and Waits on a zero counter: it returns without blocking.
	done := make(chan struct{})
	go func() { defer close(done); _ = srv.Close() }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not return with no connections admitted")
	}
	if !ln.closed.Load() {
		t.Fatal("Close returned without closing the listener")
	}

	// Only now does a connection arrive. Close has already returned, so by
	// contract no handler may run for it.
	c := newGateConn()
	select {
	case ln.gate <- c:
	case <-time.After(10 * time.Second):
		t.Fatal("accept loop never took the post-shutdown connection")
	}

	if c.wasHandled(2 * time.Second) {
		t.Fatal("a handler goroutine started for a connection accepted after Close returned: " +
			"Close cannot wait for it, and its wg.Add races Close's wg.Wait")
	}
	select {
	case <-c.closed:
	case <-time.After(2 * time.Second):
		t.Fatal("the refused connection was not closed")
	}

	ln.unblock()  // a broken accept loop would otherwise park in Accept forever
	served.Wait() // Serve must return too, not spin
	srv.mu.Lock()
	n := len(srv.conns)
	srv.mu.Unlock()
	if n != 0 {
		t.Fatalf("conns map not empty after shutdown: %d", n)
	}
	if len(srv.connSem) != 0 {
		t.Fatalf("connSem leaked %d slot(s) on the refused connection", len(srv.connSem))
	}
}

// TestCloseWaitsForAdmittedHandler is the other half of the fence: a connection
// admitted BEFORE Close must still be force-closed and waited for. It guards
// against "fixing" the race by refusing too much, or by dropping the wg.Add.
func TestCloseWaitsForAdmittedHandler(t *testing.T) {
	srv, ln := newGatedServer(t)

	var served sync.WaitGroup
	served.Add(1)
	go func() { defer served.Done(); _ = srv.Serve() }()

	c := newGateConn()
	ln.gate <- c
	if !c.wasHandled(10 * time.Second) {
		t.Fatal("handler never started for a connection accepted before Close")
	}

	done := make(chan struct{})
	go func() { defer close(done); _ = srv.Close() }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Close never returned; it must force-close the admitted handler's conn")
	}

	srv.mu.Lock()
	n := len(srv.conns)
	srv.mu.Unlock()
	if n != 0 {
		t.Fatalf("Close returned with %d handler(s) still tracked", n)
	}
	ln.unblock()
	served.Wait()
}

// teardownDispatcher models the real lifecycle: the process closes the Server
// and then tears the dispatcher down. torndown is deliberately UNSYNCHRONIZED —
// that is the point. Handlers only READ it (read/read between handlers is not a
// race); the churn test WRITES it once, after Close has returned. Close promises
// every handler goroutine has returned by then, so that write must be exclusive.
type teardownDispatcher struct {
	calls    atomic.Int64
	torndown int
}

func (d *teardownDispatcher) Call(string, []byte) ([]byte, error) {
	d.calls.Add(1)
	if d.torndown != 0 {
		return []byte{1}, nil
	}
	return nil, nil
}
func (d *teardownDispatcher) LeaderAddr() string { return "" }

// TestCloseUnderAcceptChurn is the unstaged counterpart: real sockets, real
// accepts, Close landing at an arbitrary point in the churn. It is a stress
// probe rather than the gate — the deterministic tests above are the gate — but
// it is the shape that first exposed this, and it fails three ways when the
// admission step is not under the shutdown lock: the runtime's own "WaitGroup is
// reused before previous Wait has returned" panic inside Close, a data race on
// the dispatcher being torn down, or Close simply returning with handlers still
// tracked.
//
// Handlers are kept short-lived on purpose. The dangerous interleaving needs the
// wg counter at or near zero when Close reaches Wait, which is exactly when a
// straggler's uncounted Add is fatal; long-lived connections hide it by keeping
// the counter high.
func TestCloseUnderAcceptChurn(t *testing.T) {
	const (
		iterations = 400
		dialers    = 8
	)
	violations := 0
	for i := 0; i < iterations; i++ {
		disp := &teardownDispatcher{}
		srv, err := New(Config{
			Addr: "127.0.0.1:0", Dispatcher: disp,
			MaxConns: 512, IdleTimeout: 200 * time.Millisecond,
		})
		if err != nil {
			t.Fatal(err)
		}
		var served sync.WaitGroup
		served.Add(1)
		go func() { defer served.Done(); _ = srv.Serve() }()
		addr := srv.Addr().String()

		var stop atomic.Bool
		var churn sync.WaitGroup
		for d := 0; d < dialers; d++ {
			churn.Add(1)
			go func() {
				defer churn.Done()
				body := EncodeRequest("ping", nil)
				for !stop.Load() {
					c, err := net.Dial("tcp", addr)
					if err != nil {
						return
					}
					w := bufio.NewWriter(c)
					if writeFrame(w, body) == nil {
						_ = w.Flush()
						_, _ = readFrame(bufio.NewReader(c))
					}
					_ = c.Close()
				}
			}()
		}
		time.Sleep(time.Millisecond) // let the churn reach steady state

		_ = srv.Close()
		// Close has returned, so by contract no handler is running and this
		// write is exclusive. Under -race it is the tripwire.
		disp.torndown = 1

		srv.mu.Lock()
		n := len(srv.conns)
		srv.mu.Unlock()
		if n != 0 {
			violations++
			t.Errorf("iteration %d: Close returned with %d handler(s) still tracked", i, n)
		}

		stop.Store(true)
		churn.Wait()
		served.Wait()
	}
	if violations > 0 {
		t.Fatalf("%d/%d iterations: Close returned before its handlers finished", violations, iterations)
	}
}
