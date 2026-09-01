// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rostamlabs/rostam/rlog"
)

// Config describes the TCP server.
type Config struct {
	// Addr is the bind address, e.g. "0.0.0.0:7001" or "127.0.0.1:0" (pick free port).
	Addr string

	// Dispatcher is the dispatch target. Required.
	Dispatcher Dispatcher

	// MaxConns bounds concurrent client connections. Default 10000.
	MaxConns int

	// IdleTimeout closes connections that send nothing for this duration. Default 5 min.
	IdleTimeout time.Duration

	// WriteTimeout bounds how long a single response write+flush may block on a
	// stalled client before the connection is aborted. Default 30s.
	//
	// This is a CORRECTNESS bound, not just slow-loris hygiene: a response payload
	// on a reject-writes/mmap shard is a ZERO-COPY []byte that ALIASES live mmap page
	// bytes (handleGet returns the cache's alias verbatim). That alias is referenced
	// by the write path across w.Flush(). With no write deadline a client that stops
	// reading blocks the writer in Flush indefinitely, so the alias is held for an
	// UNBOUNDED time — and the online page-recycle (cache/compact_online.go) can only
	// be safe if every reader alias is released within a bounded wall-clock window.
	// Arming this deadline before every response write guarantees the write either
	// completes or aborts (and closes the conn) within WriteTimeout, which is the
	// drain fence the cache's AliasQuarantine is derived from. 30s is generous enough
	// for a legitimately slow client to drain a max-size (16 MiB) frame over a slow
	// link while still bounding a wedged one to a small multiple of a heartbeat.
	WriteTimeout time.Duration

	// KeepAlivePeriod sets TCP keepalive interval. Default 30s.
	KeepAlivePeriod time.Duration

	// Authenticator gates every RPC. It receives an authz.AuthRequest built from
	// the protocol-v2 frame (token from the frame prefix — empty for v1 clients —
	// plus the op name and args) and returns true to allow, false to fail with
	// StatusUnauthorized. A nil Authenticator accepts every request (legacy/no-auth
	// mode).
	Authenticator Authenticator

	// TLSConfig, when non-nil, makes the server wrap its listener with
	// tls.NewListener so every accepted connection is a *tls.Conn — the v1/v2
	// framing is unchanged, just carried over TLS. When the config requires a
	// verified client cert (mTLS), the verified leaf's CommonName is extracted from
	// the completed handshake (tls.Conn.ConnectionState().VerifiedChains) and
	// threaded into authz.AuthRequest.ClientCN, so a cert-only client authorizes by
	// its CN's scopes. nil ⇒ plaintext (unchanged).
	TLSConfig *tls.Config

	// AccessLog, when non-nil (enabled), emits one structured access line per
	// dispatched request (request-id, op, status, latency, redacted principal,
	// bytes). nil/disabled ⇒ the dispatch hot path is byte-identical to the
	// pre-access-log server (no id generation, no timing).
	AccessLog *rlog.AccessLog
}

func (c *Config) applyDefaults() {
	if c.MaxConns <= 0 {
		c.MaxConns = 10_000
	}
	if c.IdleTimeout == 0 {
		c.IdleTimeout = 5 * time.Minute
	}
	if c.WriteTimeout == 0 {
		c.WriteTimeout = 30 * time.Second
	}
	if c.KeepAlivePeriod == 0 {
		c.KeepAlivePeriod = 30 * time.Second
	}
}

func (c Config) validate() error {
	if c.Addr == "" {
		return errors.New("server: ServerConfig.Addr is required")
	}
	if c.Dispatcher == nil {
		return errors.New("server: ServerConfig.Dispatcher is required")
	}
	return nil
}

// Server hosts a TCP listener that dispatches frames into a Dispatcher.
type Server struct {
	cfg     Config
	ln      net.Listener
	connSem chan struct{}
	wg      sync.WaitGroup

	mu    sync.Mutex
	conns map[net.Conn]struct{} // tracks active conns so Close can force-close them

	closeOnce sync.Once
	closeCh   chan struct{}
}

// New constructs a Server, opening its listener immediately so port-binding
// errors surface here. The caller must call Serve to start accepting.
func New(cfg Config) (*Server, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cfg.applyDefaults()
	ln, err := net.Listen("tcp", cfg.Addr)
	if err != nil {
		return nil, fmt.Errorf("server: listen %s: %w", cfg.Addr, err)
	}
	// Wrap in a TLS listener when configured: every Accept then yields a
	// *tls.Conn. The bound port is read from the underlying net.Listener before
	// wrapping is irrelevant (tls.NewListener forwards Addr()), so Addr() still
	// reports the real bound address (important for ":0" test binds).
	if cfg.TLSConfig != nil {
		ln = tls.NewListener(ln, cfg.TLSConfig)
	}
	return &Server{
		cfg:     cfg,
		ln:      ln,
		connSem: make(chan struct{}, cfg.MaxConns),
		conns:   make(map[net.Conn]struct{}),
		closeCh: make(chan struct{}),
	}, nil
}

func (s *Server) trackConn(c net.Conn) {
	s.mu.Lock()
	s.conns[c] = struct{}{}
	s.mu.Unlock()
}

func (s *Server) untrackConn(c net.Conn) {
	s.mu.Lock()
	delete(s.conns, c)
	s.mu.Unlock()
}

// Addr returns the bound listener address (useful for tests that bind :0).
func (s *Server) Addr() net.Addr { return s.ln.Addr() }

// Serve accepts connections in a loop until Close is called.
func (s *Server) Serve() error {
	for {
		conn, err := s.ln.Accept()
		if err != nil {
			select {
			case <-s.closeCh:
				return nil
			default:
			}
			return err
		}
		select {
		case s.connSem <- struct{}{}:
		default:
			_ = conn.Close()
			continue
		}
		s.trackConn(conn)
		s.wg.Add(1)
		go func(c net.Conn) {
			defer s.wg.Done()
			defer func() { <-s.connSem }()
			defer s.untrackConn(c)
			s.handleConn(c)
		}(conn)
	}
}

// Close stops accepting new connections, force-closes any in-flight
// connections, and waits for their handler goroutines to return.
// Safe to call once. The force-close is necessary because a chatty
// client (e.g., one polling a refresh op) can keep handleConn alive
// indefinitely by resetting the read deadline on each request.
func (s *Server) Close() error {
	var closeErr error
	s.closeOnce.Do(func() {
		close(s.closeCh)
		closeErr = s.ln.Close()
		s.mu.Lock()
		for c := range s.conns {
			_ = c.Close()
		}
		s.mu.Unlock()
		s.wg.Wait()
	})
	return closeErr
}

func (s *Server) handleConn(c net.Conn) {
	defer func() { _ = c.Close() }()
	if tc, ok := c.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
		_ = tc.SetKeepAlive(true)
		_ = tc.SetKeepAlivePeriod(s.cfg.KeepAlivePeriod)
	}
	// Extract the VERIFIED mTLS client-cert CN once per connection. When TLS is
	// enabled the accepted conn is a *tls.Conn; forcing the handshake here lets us
	// read ConnectionState().VerifiedChains — which crypto/tls populates ONLY after
	// a successfully verified chain (RequireAndVerifyClientCert /
	// VerifyClientCertIfGiven-with-cert). We deliberately use VerifiedChains, NOT
	// PeerCertificates: PeerCertificates carries the raw presented cert even when
	// unverified, so using it as a principal would let a spoofed/self-signed cert
	// claim any CN. clientCN stays "" for plaintext conns, for a client that
	// presented no cert, or for an unverified cert → the authorizer then falls back
	// to token-or-deny. A handshake failure (e.g. RequireAndVerifyClientCert with no
	// client cert) errors here and the conn is dropped before any frame is read.
	var clientCN string
	if tc, ok := c.(*tls.Conn); ok {
		_ = tc.SetReadDeadline(time.Now().Add(s.cfg.IdleTimeout))
		if err := tc.Handshake(); err != nil {
			return // mTLS handshake rejected (e.g. missing/non-CA client cert) — drop.
		}
		_ = tc.SetReadDeadline(time.Time{})
		if st := tc.ConnectionState(); len(st.VerifiedChains) > 0 && len(st.VerifiedChains[0]) > 0 {
			clientCN = st.VerifiedChains[0][0].Subject.CommonName
		}
	}
	r := bufio.NewReaderSize(c, connBufSize)
	w := bufio.NewWriterSize(c, connBufSize)
	// reqHdr / respHdr / reqBuf are hoisted out of the loop so their
	// single heap allocations (forced by the bufio escape on Write/Read)
	// are paid once per connection lifetime, not once per request. This
	// also lets us skip the sync.Pool round-trip that readFrame uses for
	// short-lived RPC traffic — at 20+ concurrent goroutines the pool
	// churned harder than it amortized.
	var (
		reqHdr  [4]byte
		respHdr [9]byte
		reqBuf  = make([]byte, 0, 256)
	)

	// Per-connection PIPELINING (see docs in this file's header comment block):
	// a client MAY send its next request before reading the previous response.
	// Requests then dispatch CONCURRENTLY (their replication round trips
	// overlap) while responses are emitted strictly in REQUEST ORDER by one
	// writer goroutine fed a FIFO of per-request result channels. A client that
	// never pipelines keeps the exact inline path below (outstanding == 0 on
	// every iteration): same zero-copy dispatch, same syscalls, byte-identical
	// behavior. NOTE the semantics pipelining buys AND costs: requests in the
	// same window execute concurrently, so two pipelined writes to the SAME key
	// may apply in either order — a client that needs read-your-write or
	// write-after-write ordering on a key must not pipeline those requests on
	// one connection (exactly the contract of Kafka's max.in.flight > 1).
	var (
		outstanding atomic.Int64                                       // submitted, not yet written by the writer
		pendingCh   = make(chan chan connPipeResp, connPipelineWindow) // FIFO; its capacity IS the window
		writerErr   atomic.Bool                                        // writer hit a write error; conn is dying
		writerWG    sync.WaitGroup
	)
	writerWG.Add(1)
	go func() {
		defer writerWG.Done()
		var hdr [9]byte
		for rc := range pendingCh {
			resp := <-rc
			if !writerErr.Load() {
				// Arm a write deadline BEFORE every write+flush. resp.payload may be a
				// zero-copy alias into live mmap page bytes (handleGet on a reject-writes
				// shard), and this goroutine references it across writeResponse/Flush. A
				// client that stops reading would otherwise block Flush indefinitely, so
				// the alias hold would be unbounded — the exact hazard the cache recycle
				// must exclude. On a deadline the write/flush errors, we set writerErr +
				// Close, and the loop below drains the rest of pendingCh WITHOUT writing
				// (so writerWG.Wait() in the defer always completes — this also fixes the
				// latent cleanup hang where a wedged Flush stalled Close). Re-armed per
				// response: writes are far rarer than the read-deadline hot path, so the
				// read-deadline's skip optimization is unnecessary here.
				_ = c.SetWriteDeadline(time.Now().Add(s.cfg.WriteTimeout))
				if werr := writeResponse(w, &hdr, resp.status, resp.payload); werr != nil {
					writerErr.Store(true)
					_ = c.Close() // unblock the reader; conn is done
				} else if len(pendingCh) == 0 {
					// Flush only when no further response is imminent: a burst of
					// pipelined responses coalesces into one flush (and one syscall).
					if ferr := w.Flush(); ferr != nil {
						writerErr.Store(true)
						_ = c.Close()
					}
				}
			}
			if resp.reqBuf != nil {
				putConnReqBuf(resp.reqBuf)
			}
			// Return the owned payload copy (Fix: pipelined alias recycle). Done
			// unconditionally — including the writerErr drain path that skips the write
			// above — so no buffer leaks when the connection is dying.
			if resp.respBuf != nil {
				putConnRespBuf(resp.respBuf)
			}
			outstanding.Add(-1)
		}
	}()
	// One dispatch pool PER CONNECTION. Sharding it this way means the idle
	// channel is touched only by this connection's reader goroutine and the
	// workers it spawned, so submissions never contend across connections — the
	// failure mode a single server-wide pool showed on short GET requests.
	dpool := newDispatchPool()
	defer func() {
		close(pendingCh)
		writerWG.Wait()
		// Retire after the writer has drained, so nothing is still running.
		dpool.close()
	}()

	// lastDeadline is the wall-clock time the read deadline was last armed.
	// SetReadDeadline mutates a runtime timer on every call (~1.25% of server
	// CPU under GET-heavy load) even though IdleTimeout only needs SOME
	// deadline set in the future, not a fresh one per request. Re-arm only
	// once more than half the idle window has elapsed since the last arm, but
	// arm 1.5x IdleTimeout out (not 1x): a deadline armed at lastDeadline+1.5x
	// only gets used when the NEXT arm is skipped, and skipping only happens
	// when the gap since lastDeadline is <= IdleTimeout/2 — so the deadline
	// fires no sooner than lastDeadline+1.5x - IdleTimeout/2 = lastDeadline+1x
	// after the true last read, and no later than lastDeadline+1.5x. A
	// deadline remains set on the conn at all times (slow-loris protection is
	// never dropped, only refreshed less often); effective idle tolerance
	// ranges within [1x, 1.5x] IdleTimeout — never SHORTER than the original
	// per-request-refresh guarantee, only occasionally longer.
	var lastDeadline time.Time
	for {
		now := time.Now()
		if lastDeadline.IsZero() || now.Sub(lastDeadline) > s.cfg.IdleTimeout/2 {
			_ = c.SetReadDeadline(now.Add(s.cfg.IdleTimeout + s.cfg.IdleTimeout/2))
			lastDeadline = now
		}
		if _, err := io.ReadFull(r, reqHdr[:]); err != nil {
			var netErr net.Error
			isTimeout := errors.As(err, &netErr) && netErr.Timeout()
			if !errors.Is(err, io.EOF) && !errors.Is(err, net.ErrClosed) && !isTimeout {
				slog.Error("read frame length", "transport", "tcp", "err", err)
			}
			return
		}
		n := binary.BigEndian.Uint32(reqHdr[:])
		if n == 0 || n > MaxFrameSize {
			slog.Warn("invalid frame length", "transport", "tcp", "len", n)
			return
		}

		// Pipelined path: the client has MORE request bytes already buffered (it
		// sent ahead without waiting), or responses are still in flight — so this
		// request must go through the ordered writer. Otherwise take the inline
		// path: outstanding == 0 proves the writer goroutine is parked on its
		// channel recv (it decrements only AFTER its final write), so the reader
		// may safely use the shared bufio.Writer itself.
		if outstanding.Load() > 0 || r.Buffered() > 0 {
			reqBp := getConnReqBuf(int(n))
			req := *reqBp
			if _, err := io.ReadFull(r, req); err != nil {
				putConnReqBuf(reqBp)
				slog.Error("read frame body", "transport", "tcp", "err", err)
				return
			}
			rc := make(chan connPipeResp, 1)
			outstanding.Add(1)
			pendingCh <- rc // window-full blocks here = natural backpressure
			// Runs on a POOLED goroutine rather than a fresh one: the closure
			// below is identical, but a reused goroutine's stack is already
			// grown, which removes the per-request runtime.copystack this path
			// used to pay (11.89% of server CPU under replicated write load).
			// The pool never blocks the submitter, so in-flight concurrency is
			// unchanged — see dispatchpool.go.
			dpool.run(func() {
				status, payload := dispatch(s.cfg.Dispatcher, req, s.cfg.Authenticator, clientCN, s.cfg.AccessLog)
				// Copy the payload out of any zero-copy cache alias BEFORE enqueueing it
				// on the ordered writer: a queued pipelined response can be held across a
				// slow flush for far longer than the cache's AliasQuarantine, so handing
				// the writer the raw mmap alias would let an online page recycle overwrite
				// bytes it still references (torn read). See copyPipelinePayload.
				owned, respBp := copyPipelinePayload(payload)
				rc <- connPipeResp{status: status, payload: owned, reqBuf: reqBp, respBuf: respBp}
			})
			continue
		}

		if cap(reqBuf) < int(n) {
			reqBuf = make([]byte, n)
		} else {
			reqBuf = reqBuf[:n]
		}
		if _, err := io.ReadFull(r, reqBuf); err != nil {
			slog.Error("read frame body", "transport", "tcp", "err", err)
			return
		}
		status, payload := dispatch(s.cfg.Dispatcher, reqBuf, s.cfg.Authenticator, clientCN, s.cfg.AccessLog)
		// Bound the alias hold on the inline (non-pipelined) path too: payload may
		// alias live mmap page bytes and is referenced across writeResponse/Flush, so
		// a stalled reader must not pin it past WriteTimeout. On the deadline the
		// write/flush errors and handleConn returns (its defer closes the conn),
		// releasing the alias — same bound the pipelined writer goroutine enforces.
		_ = c.SetWriteDeadline(time.Now().Add(s.cfg.WriteTimeout))
		if werr := writeResponse(w, &respHdr, status, payload); werr != nil {
			slog.Error("write response", "transport", "tcp", "err", werr)
			return
		}
		if ferr := w.Flush(); ferr != nil {
			return
		}
		// Release a one-off large frame's backing array rather than pinning it for
		// the connection's lifetime. Without this, a single MaxFrameSize (16 MiB)
		// request keeps 16 MiB resident per idle connection; at MaxConns that is a
		// multi-GiB slow-loris-style amplification. The common small-frame case
		// stays allocation-free (cap under the threshold is reused as before).
		if cap(reqBuf) > connBufRetainCap {
			reqBuf = make([]byte, 0, 4096)
		}
	}
}

// connPipelineWindow caps how many pipelined requests one connection may have
// in flight; pendingCh's capacity enforces it (a full window blocks the reader
// — backpressure, not an error).
const connPipelineWindow = 64

// connPipeResp is one pipelined request's completed response, handed to the
// connection's ordered writer. reqBuf is the pooled request copy (its *[]byte
// wrapper, so putConnReqBuf can hand the SAME wrapper back to the pool instead
// of boxing a fresh one) to recycle once the response has been written
// (dispatch results may alias the buffer).
//
// respBuf is the pooled OWNED copy of the payload (see copyPipelinePayload): the
// pipelined path copies the dispatch result out of any zero-copy cache alias at
// dispatch time, so the ordered writer never references live mmap page bytes
// across a QUEUED network flush (the alias-recycle hazard — see the copy site and
// cache/compact_online.go). payload points into *respBuf; nil respBuf means an
// empty payload that needed no buffer. Recycled after the write, next to reqBuf.
type connPipeResp struct {
	status  uint8
	payload []byte
	reqBuf  *[]byte
	respBuf *[]byte
}

// connReqPool recycles pipelined request copies (the shared reqBuf cannot be
// used concurrently, so the pipelined path takes an owned buffer per request).
//
// Get/Put always hand around the SAME *[]byte wrapper obtained from the pool:
// getConnReqBuf mutates the pointee in place and returns the pointer;
// putConnReqBuf takes that same pointer back. Previously putConnReqBuf took a
// []byte and did `connReqPool.Put(&b)` — the address of a fresh local — which
// heap-allocated a brand new *[]byte wrapper on every single return trip
// (12.8% of all server allocations under load). Threading the pointer through
// the request's lifecycle (connPipeResp.reqBuf) removes that allocation
// entirely: the wrapper is boxed once per pool item, not once per request.
var connReqPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 512)
		return &b
	},
}

func getConnReqBuf(n int) *[]byte {
	bp := connReqPool.Get().(*[]byte)
	if cap(*bp) >= n {
		*bp = (*bp)[:n]
	} else {
		*bp = make([]byte, n)
	}
	return bp
}

func putConnReqBuf(bp *[]byte) {
	if cap(*bp) > connBufRetainCap {
		return
	}
	*bp = (*bp)[:0]
	connReqPool.Put(bp)
}

// connRespPool recycles the OWNED payload copies the pipelined path makes (see
// copyPipelinePayload). It mirrors connReqPool exactly — same *[]byte wrapper
// discipline (getConnRespBuf mutates the pointee and returns the pointer;
// putConnRespBuf takes that same pointer back, boxing the wrapper once per pool
// item rather than once per response) and the same connBufRetainCap release
// ceiling so a one-off large value is not pinned for the connection's lifetime.
var connRespPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 512)
		return &b
	},
}

func getConnRespBuf(n int) *[]byte {
	bp := connRespPool.Get().(*[]byte)
	if cap(*bp) >= n {
		*bp = (*bp)[:n]
	} else {
		*bp = make([]byte, n)
	}
	return bp
}

func putConnRespBuf(bp *[]byte) {
	if cap(*bp) > connBufRetainCap {
		return
	}
	*bp = (*bp)[:0]
	connRespPool.Put(bp)
}

// copyPipelinePayload copies a dispatch payload out of any zero-copy cache alias
// into an owned, pooled buffer, returning the owned copy and the pooled wrapper to
// recycle after the write (an empty payload needs neither: nil, nil).
//
// This is THE fix for the pipelined alias-recycle hazard. On a reject-writes/mmap
// shard handleGet returns a []byte that ALIASES live mmap page bytes; every other
// consumer (HTTP, WASM, mget, getdel/getset, snapshot, and the INLINE non-pipelined
// TCP path) copies or consumes that alias synchronously within one WriteTimeout, so
// the online compactor's AliasQuarantine (>= 2*WriteTimeout) safely fences a page
// recycle. The PIPELINED path is the exception: it QUEUES the response on pendingCh
// (window depth connPipelineWindow), so the ordered writer could hold the raw alias
// for up to (queued responses)*WriteTimeout before flushing it over a slow link —
// far past the quarantine — and the compactor could then overwrite (recycle) the
// backing extent while the writer still aliases it, shipping a TORN read to the
// client. Copying here, at dispatch time, makes the writer reference only owned
// bytes, so the pipelined alias-hold bound collapses to ~0 (copied before enqueue),
// independent of pipeline depth and WriteTimeout. The inline path stays zero-copy —
// it is bounded to a single WriteTimeout and drains before the next dispatch.
//
// TODO(perf): scope the pipelined copy to eligible shards — a payload that does not
// alias online-compaction-eligible (reject-writes/mmap) cache memory cannot be
// recycled and need not be copied. Threading that signal from dispatch is invasive,
// so this conservatively copies every non-empty pipelined payload (correctness first).
func copyPipelinePayload(payload []byte) ([]byte, *[]byte) {
	if len(payload) == 0 {
		return nil, nil
	}
	bp := getConnRespBuf(len(payload))
	copy(*bp, payload)
	return *bp, bp
}
