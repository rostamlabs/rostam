// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"encoding/binary"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/panjf2000/gnet/v2"

	"github.com/rostamlabs/rostam/rlog"
)

// EpollServer is an experimental epoll-based (event-loop) TCP transport, an
// alternative to the goroutine-per-connection model in Server. A small pool of
// event-loop goroutines (NumEventLoop) services many connections via non-blocking
// I/O, so a lightly-loaded server does not pay the Go-scheduler park/unpark churn
// that goroutine-per-connection incurs at low concurrency (see the netkv
// GOMAXPROCS analysis).
//
// It speaks the SAME wire protocol as Server — request {len u32}{body}, response
// {bodyLen u32}{status u8}{payloadLen u32}{payload} — and reuses the same,
// I/O-agnostic dispatch().
//
// PLAINTEXT ONLY, by design. Go's crypto/tls is a blocking API built around a
// net.Conn; running it inside an event loop would require a non-blocking TLS
// state machine (a memory-BIO feed), which is fragile and largely defeats the
// point (a per-connection tls.Conn re-introduces a blocking goroutine). So the
// event loop serves cleartext, and clientCN is always "" (no cert identity).
// Encrypted / mTLS-authenticated traffic keeps using the goroutine-per-connection
// Server (or terminates TLS at a proxy in front of the plaintext epoll port).
// NewServer wires this: -epoll is used only when TLSConfig is nil, otherwise it
// falls back to Server. This is intended for a trusted network / behind a
// terminator, not as a drop-in TLS replacement for Server.
type EpollServer struct {
	gnet.BuiltinEventEngine
	disp        Dispatcher
	auth        Authenticator
	alog        *rlog.AccessLog // OPT-IN access log; nil/disabled = zero-cost dispatch
	numLoops    int
	idleTimeout time.Duration // close conns idle this long; <= 0 disables the janitor
	addr        string        // protocol addr, set in Run
	eng         gnet.Engine   // captured in OnBoot; used by Stop
	booted      chan struct{} // closed once OnBoot has run (eng is valid)
	conns       sync.Map      // gnet.Conn -> *epollConn, for the idle sweep (OnTick)
	// canAppend caches whether disp implements AppendDispatcher, resolved once at
	// construction rather than per frame. When false the loop allocates no reply
	// buffer at all and every dispatch is byte-identical to before.
	canAppend bool
}

// epollConn is per-connection state. lastActiveNanos is updated on every OnTraffic
// (event-loop goroutine) and read by the idle sweep (ticker goroutine), so it is
// an atomic. respBuf is a reused response scratch buffer — safe because gnet's
// Conn.Write does not retain the caller's slice (it writes synchronously or copies
// the remainder into its outbound buffer). Only the connection's own event loop
// touches respBuf, so it needs no synchronization.
type epollConn struct {
	lastActiveNanos atomic.Int64
	respBuf         []byte
	// payloadBuf is the reply buffer handed to dispatchInto, reused across every
	// frame on this connection. encode() COPIES the payload into respBuf before
	// the loop reads the next request, so overwriting it on the next iteration is
	// safe; see AppendDispatcher.
	payloadBuf []byte
}

// encode writes the response frame into the connection's reused buffer and returns
// it (valid until the next encode on this connection). Responses larger than the
// retain cap are one-off allocations so a single huge reply is not pinned per conn.
func (m *epollConn) encode(status uint8, payload []byte) []byte {
	need := 9 + len(payload)
	if need > connBufRetainCap {
		return encodeResponse(status, payload)
	}
	if cap(m.respBuf) < need {
		m.respBuf = make([]byte, need)
	}
	b := m.respBuf[:need]
	binary.BigEndian.PutUint32(b[0:4], uint32(1+4+len(payload))) //nolint:gosec // bounded by MaxFrameSize on write
	b[4] = status
	binary.BigEndian.PutUint32(b[5:9], uint32(len(payload))) //nolint:gosec // bounded above
	copy(b[9:], payload)
	return b
}

// NewEpollServer builds an epoll transport bound to the given Dispatcher/Authenticator.
// numLoops <= 0 lets gnet default to GOMAXPROCS event loops. idleTimeout <= 0
// disables the idle-connection sweep (accept indefinitely); pass a positive value
// (e.g. 5 min) for slow-loris protection.
func NewEpollServer(disp Dispatcher, auth Authenticator, alog *rlog.AccessLog, numLoops int, idleTimeout time.Duration) *EpollServer {
	_, canAppend := disp.(AppendDispatcher)
	return &EpollServer{disp: disp, auth: auth, alog: alog, numLoops: numLoops, idleTimeout: idleTimeout,
		booted: make(chan struct{}), canAppend: canAppend}
}

// Run binds addr ("host:port") and blocks serving until the engine stops.
func (s *EpollServer) Run(addr string) error {
	s.addr = "tcp://" + addr
	opts := []gnet.Option{gnet.WithMulticore(true), gnet.WithReuseAddr(true)}
	if s.numLoops > 0 {
		opts = append(opts, gnet.WithNumEventLoop(s.numLoops))
	}
	if s.idleTimeout > 0 {
		opts = append(opts, gnet.WithTicker(true)) // enables OnTick (the idle sweep)
	}
	return gnet.Run(s, s.addr, opts...)
}

// Start runs the engine in the background and returns once it has bound (success)
// or failed to bind. Unlike Run (which blocks until Stop), a bind/setup error is
// surfaced synchronously to the caller instead of being lost in a detached
// goroutine — so NewServer can fail loudly on a port conflict, matching Server.
func (s *EpollServer) Start(addr string) error {
	errCh := make(chan error, 1)
	go func() { errCh <- s.Run(addr) }()
	select {
	case <-s.booted: // OnBoot fired ⇒ the listener is bound and serving
		return nil
	case err := <-errCh: // Run returned before booting ⇒ bind/setup failure
		if err == nil {
			err = errors.New("epoll engine stopped before boot")
		}
		return err
	}
}

// Addr returns the address this transport serves, without the "tcp://" scheme
// gnet requires, or "" before Run/Start has been called.
//
// It reports the address the engine was ASKED to bind, not one resolved from the
// listener: gnet's Engine exposes no accessor for the bound address, so a caller
// that passed port 0 gets ":0" back rather than the port the kernel chose. Every
// production path names a real port; the caveat matters only to a test that binds
// ephemerally and then expects to dial what this returns.
func (s *EpollServer) Addr() string {
	return strings.TrimPrefix(s.addr, "tcp://")
}

// Stop gracefully shuts the event engine down (drains, closes listeners); the
// blocked Run returns. If the engine never booted, it waits for ctx to cancel
// rather than blocking forever.
func (s *EpollServer) Stop(ctx context.Context) error {
	select {
	case <-s.booted:
		return s.eng.Stop(ctx)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *EpollServer) OnBoot(eng gnet.Engine) gnet.Action {
	s.eng = eng
	close(s.booted)
	slog.Info("epoll transport up", "transport", "tcp")
	return gnet.None
}

// OnOpen allocates per-connection state (the reused response buffer) and, when the
// idle sweep is on, registers the connection for it.
func (s *EpollServer) OnOpen(c gnet.Conn) ([]byte, gnet.Action) {
	m := &epollConn{}
	if s.idleTimeout > 0 {
		m.lastActiveNanos.Store(time.Now().UnixNano())
		s.conns.Store(c, m)
	}
	c.SetContext(m) // set on the conn's own loop before any OnTraffic — safe
	return nil, gnet.None
}

// OnClose deregisters the connection.
func (s *EpollServer) OnClose(c gnet.Conn, _ error) gnet.Action {
	if s.idleTimeout > 0 {
		s.conns.Delete(c)
	}
	return gnet.None
}

// OnTick sweeps the connection registry and closes any connection idle longer
// than idleTimeout (slow-loris / dead-peer protection). Runs on gnet's ticker
// goroutine; Conn.Close and the atomic lastActive read are concurrency-safe.
func (s *EpollServer) OnTick() (time.Duration, gnet.Action) {
	interval := s.idleTimeout / 2
	if interval < time.Second {
		interval = time.Second
	}
	cutoff := time.Now().Add(-s.idleTimeout).UnixNano()
	s.conns.Range(func(k, v any) bool {
		if v.(*epollConn).lastActiveNanos.Load() < cutoff {
			_ = k.(gnet.Conn).Close() // async close on the conn's loop; OnClose deregisters
		}
		return true
	})
	return interval, gnet.None
}

// OnTraffic drains every COMPLETE frame currently buffered for c, dispatches each,
// and queues its response. Partial frames stay in gnet's inbound buffer until the
// rest arrives (OnTraffic fires again). Returning gnet.None keeps the connection;
// gnet flushes queued writes after this returns.
func (s *EpollServer) OnTraffic(c gnet.Conn) gnet.Action {
	m, _ := c.Context().(*epollConn) // set in OnOpen; holds the reused response buffer
	if m != nil && s.idleTimeout > 0 {
		m.lastActiveNanos.Store(time.Now().UnixNano()) // same-loop write; idle sweep reads atomically
	}
	for {
		if c.InboundBuffered() < 4 {
			return gnet.None // not even a length prefix yet
		}
		hdr, _ := c.Peek(4)
		n := int(binary.BigEndian.Uint32(hdr))
		if n <= 0 || n > MaxFrameSize {
			slog.Warn("epoll invalid frame length", "transport", "tcp", "len", n)
			return gnet.Close
		}
		if c.InboundBuffered() < 4+n {
			return gnet.None // body not fully arrived; wait for more
		}
		// Consume the whole frame. buf aliases gnet's inbound buffer and is only
		// valid until the next read on c — we finish with it (copy payload into the
		// response) before looping.
		buf, _ := c.Next(4 + n)
		// The append buffer exists ONLY when the dispatcher can use it. Allocating
		// it otherwise would put a 512-byte buffer on every connection to serve a
		// path that never reads it, and would change the fallback this PR
		// documents as untouched. s.canAppend is resolved once, at construction.
		var dst []byte
		if m != nil && s.canAppend {
			if m.payloadBuf == nil {
				// Allocated on the first frame rather than in OnOpen so a connection
				// that never sends one costs nothing. Slicing a nil buffer to [:0]
				// yields a nil slice, which dispatchInto reads as "no append buffer" -
				// so without this the first frame on every connection would silently
				// take the allocating path.
				m.payloadBuf = make([]byte, 0, 512)
			}
			dst = m.payloadBuf[:0]
		}
		status, payload, appended := dispatchInto(s.disp, buf[4:4+n], s.auth, "", s.alog, dst)
		// Keep the buffer the append path returned, whether or not it still
		// aliases dst. A payload that had to GROW past dst is a NEW array, and it
		// is exactly the one worth keeping: adopting only aliasing payloads left a
		// connection whose records outgrow its buffer reallocating on every single
		// request, forever. `appended` excludes the freshly built error and
		// not-leader frames that aliasing used to screen out.
		if m != nil && appended && cap(payload) > cap(m.payloadBuf) && cap(payload) <= connBufRetainCap {
			m.payloadBuf = payload // keep the larger array; bounded like respBuf
		}
		status, payload = clampResponse(status, payload) // match Server.writeResponse's MaxFrameSize bound
		var resp []byte
		if m != nil {
			resp = m.encode(status, payload) // reuses the per-conn buffer (gnet.Write doesn't retain it)
		} else {
			resp = encodeResponse(status, payload)
		}
		if _, err := c.Write(resp); err != nil {
			return gnet.Close
		}
	}
}

// clampResponse enforces the same MaxFrameSize bound Server.writeResponse applies
// (conn.go): an oversized dispatch result becomes a StatusError frame, so both
// transports fail an over-limit reply identically instead of the epoll path
// emitting a frame the client would reject on decode. With this, the encoders'
// uint32 length casts are provably in range.
func clampResponse(status uint8, payload []byte) (uint8, []byte) {
	if 1+4+len(payload) > MaxFrameSize {
		return StatusError, EncodeErrorPayload("response exceeds MaxFrameSize")
	}
	return status, payload
}

// encodeResponse builds one response frame: {bodyLen u32}{status u8}{payloadLen u32}{payload}.
// Mirrors writeResponse but returns bytes (gnet writes buffers, not a bufio.Writer).
// Callers pass MaxFrameSize-bounded payloads (see clampResponse).
func encodeResponse(status uint8, payload []byte) []byte {
	bodyLen := 1 + 4 + len(payload)
	resp := make([]byte, 9+len(payload))
	binary.BigEndian.PutUint32(resp[0:4], uint32(bodyLen)) //nolint:gosec // bounded by MaxFrameSize on write
	resp[4] = status
	binary.BigEndian.PutUint32(resp[5:9], uint32(len(payload))) //nolint:gosec // bounded above
	copy(resp[9:], payload)
	return resp
}
