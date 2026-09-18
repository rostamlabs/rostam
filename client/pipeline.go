// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// Pipelining (opt-in via Config.PipelineDepth) lets ONE connection carry many
// requests in flight instead of one-at-a-time. It removes the
// "throughput = connections / per-op latency" ceiling of the request-response
// pool: with W ops outstanding per connection, per-op latency no longer caps
// per-connection throughput, so a latency-bound replicated-write workload
// stops being connection-limited. The server answers in REQUEST ORDER (see
// server per-connection pipelining), so responses are correlated by FIFO
// position — no request IDs on the wire.
//
// SEMANTICS the caller must accept: requests on one pipelined connection may be
// IN FLIGHT concurrently, so two pipelined writes to the SAME key can apply in
// either order. A caller needing per-key ordering must not pipeline those writes
// (identical to Kafka's max.in.flight > 1). The default (PipelineDepth 0) keeps
// the strict request-response pool — byte-identical to before.

// pipeResp is one completed response handed to a waiting caller.
type pipeResp struct {
	status  uint8
	payload []byte
	err     error
}

// pipeWaiter is one in-flight request's result slot (buffered 1 so the reader
// never blocks delivering, even if the caller abandoned on ctx cancellation).
type pipeWaiter struct{ done chan pipeResp }

// pipeConn is one pipelined connection to a server. Writes serialize under mu
// (which also enforces FIFO enqueue order); one reader goroutine matches
// responses to waiters in that same FIFO order. On any I/O error the connection
// is marked dead and every pending waiter fails; the owner dials a fresh one.
type pipeConn struct {
	authToken string
	callT     time.Duration

	mu sync.Mutex
	// queued counts callers blocked on mu with a frame to add. A caller holding
	// the lock reads it to decide whether to pay for the flush itself or leave
	// its bytes for the next writer to carry out with theirs.
	queued atomic.Int32
	w      *bufio.Writer
	tcp    net.Conn
	fifo   chan *pipeWaiter // cap == pipeline depth; holds waiters in wire order
	// slots is the window itself: one token per in-flight request, taken before
	// the write lock and handed back when the reader matches a response. It is
	// what fifo's capacity used to do, split out so the WAIT for room happens
	// where a caller can still be cancelled.
	slots  chan struct{}
	dead   atomic.Bool
	closed chan struct{}
}

func dialPipeConn(ctx context.Context, addr, authToken string, depth int, callT time.Duration, dialer *net.Dialer, tlsCfg *tls.Config) (*pipeConn, error) {
	raw, err := dialer.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, err
	}
	if t, ok := raw.(*net.TCPConn); ok {
		_ = t.SetNoDelay(true) // set on the raw conn before any TLS wrap
	}
	tcp := raw
	if tlsCfg != nil {
		tconn := tls.Client(raw, tlsCfg)
		if err := tconn.HandshakeContext(ctx); err != nil {
			_ = raw.Close()
			return nil, err
		}
		tcp = tconn
	}
	pc := &pipeConn{
		authToken: authToken,
		callT:     callT,
		w:         bufio.NewWriterSize(tcp, 64<<10),
		tcp:       tcp,
		fifo:      make(chan *pipeWaiter, depth),
		slots:     make(chan struct{}, depth),
		closed:    make(chan struct{}),
	}
	go pc.readLoop(tcp)
	return pc, nil
}

// lockWrite takes the write lock, counting this caller as queued while it waits
// so whoever holds the lock can see that another frame is about to follow.
func (pc *pipeConn) lockWrite() {
	pc.queued.Add(1)
	pc.mu.Lock()
	pc.queued.Add(-1)
}

// reserve takes one of the connection's in-flight slots, waiting when the
// window is full. Nothing is held while it waits, so a caller can still honour
// its ctx, and the wait ends at the call's own deadline: queueing for a slot and
// waiting for the answer are two phases of ONE call, so the budget is shared
// rather than granted twice.
func (pc *pipeConn) reserve(ctx context.Context, budget time.Time) error {
	select {
	case pc.slots <- struct{}{}:
		return nil
	default:
	}
	t := time.NewTimer(time.Until(budget))
	defer t.Stop()
	select {
	case pc.slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return context.DeadlineExceeded
	case <-pc.closed:
		return errPipeDead
	}
}

// release hands a slot back. Every reserve is paired with exactly one release:
// the reader releases when it matches a response to its waiter, and call
// releases on the one path that reserves without enqueueing.
func (pc *pipeConn) release() { <-pc.slots }

// call sends op+args and blocks for its ordered response, ctx (or the
// per-call timeout) cancellation, or the connection dying. On ctx cancel the
// waiter is deliberately LEFT in the fifo: the reader still delivers this
// response to its (now-abandoned, buffered) slot, preserving FIFO alignment for
// every later caller — the response is simply discarded.
func (pc *pipeConn) call(ctx context.Context, op string, args []byte) (uint8, []byte, error) {
	if pc.dead.Load() {
		return 0, nil, errPipeDead
	}
	frame, err := encodeRequestFrame(pc.authToken, op, args)
	if err != nil {
		return 0, nil, err
	}
	w := &pipeWaiter{done: make(chan pipeResp, 1)}

	// ONE deadline for the whole call. Queueing for a window slot and waiting for
	// the answer are phases of the same call, so giving each its own callT would
	// let a call run to twice what CallTimeout documents as its cap.
	budget := time.Now().Add(pc.callT)

	// Take a window slot BEFORE the write lock. Waiting here is selectable, so
	// the caller's ctx and its own call timeout stay live; waiting for one while
	// HOLDING the write lock would make every caller behind it uncancellable,
	// because sync.Mutex has no ctx-aware acquire and their deadlines cannot fire
	// while a stalled peer keeps the holder parked.
	if rerr := pc.reserve(ctx, budget); rerr != nil {
		return 0, nil, rerr
	}

	pc.lockWrite()
	if pc.dead.Load() {
		pc.mu.Unlock()
		pc.release()
		return 0, nil, errPipeDead
	}
	// Enqueue the waiter and write the frame under the SAME lock so wire order
	// and fifo order are identical: the server replies in wire order and a
	// response is matched to its waiter by fifo POSITION, there being no request
	// id on the wire. A caller that dropped the lock between the two could be
	// overtaken by one that enqueued later but wrote first, and be handed
	// someone else's answer. The send cannot block -- the reservation above is
	// exactly the guarantee that fifo has room.
	pc.fifo <- w
	_, werr := pc.w.Write(frame)
	// Flush only when nobody is queued behind us. A caller already blocked on mu
	// is about to write into this same buffer, and its flush carries our bytes
	// out with its own -- so a burst of concurrent calls costs ONE write syscall
	// instead of one each. Every burst ends with a caller that sees zero queued,
	// so nothing is ever left unflushed, and the wait is bounded by the mutex
	// hand-off rather than by a timer.
	if werr == nil && pc.queued.Load() == 0 {
		werr = pc.w.Flush()
	}
	pc.mu.Unlock()
	if werr != nil {
		pc.fail(werr)
		return 0, nil, werr
	}

	deadline := time.NewTimer(time.Until(budget)) // what is LEFT of the budget
	defer deadline.Stop()
	select {
	case r := <-w.done:
		return r.status, r.payload, r.err
	case <-ctx.Done():
		return 0, nil, ctx.Err()
	case <-deadline.C:
		return 0, nil, context.DeadlineExceeded
	case <-pc.closed:
		return 0, nil, errPipeDead
	}
}

func (pc *pipeConn) readLoop(tcp net.Conn) {
	r := bufio.NewReaderSize(tcp, 64<<10)
	var hdr [4]byte
	for {
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			pc.fail(err)
			return
		}
		n := binary.BigEndian.Uint32(hdr[:])
		if n == 0 || n > MaxFrameSize {
			pc.fail(ErrFrameTooLarge)
			return
		}
		body := make([]byte, n) // owned by the waiter — cannot reuse across pipelined responses
		if _, err := io.ReadFull(r, body); err != nil {
			pc.fail(err)
			return
		}
		status, payload, derr := decodeResponse(body)
		w := <-pc.fifo // FIFO: this response belongs to the oldest outstanding request
		pc.release()   // the window has room again
		w.done <- pipeResp{status: status, payload: payload, err: derr}
	}
}

// fail marks the conn dead once and drains every pending waiter with err, so no
// caller blocks forever on a broken connection.
func (pc *pipeConn) fail(err error) {
	if !pc.dead.CompareAndSwap(false, true) {
		return
	}
	close(pc.closed)
	_ = pc.tcp.Close()
	for {
		select {
		case w := <-pc.fifo:
			w.done <- pipeResp{err: err}
		default:
			return
		}
	}
}

func (pc *pipeConn) close() { pc.fail(errPipeDead) }

// pipeSet is a small round-robin set of pipelined connections to one server.
// Concurrent Calls spread across them; each carries up to PipelineDepth in
// flight. A dead conn is redialed lazily on the next pick.
type pipeSet struct {
	addr      string
	authToken string
	depth     int
	callT     time.Duration
	dialer    *net.Dialer
	tlsCfg    *tls.Config

	mu    sync.Mutex
	conns []*pipeConn
	rr    uint64
}

func (s *pipeSet) pick(ctx context.Context) (*pipeConn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.conns) == 0 {
		return nil, errPipeDead // never populated; caller falls back
	}
	i := int(s.rr % uint64(len(s.conns)))
	s.rr++
	pc := s.conns[i]
	if pc == nil || pc.dead.Load() {
		fresh, err := dialPipeConn(ctx, s.addr, s.authToken, s.depth, s.callT, s.dialer, s.tlsCfg)
		if err != nil {
			return nil, err
		}
		s.conns[i] = fresh
		return fresh, nil
	}
	return pc, nil
}

func (s *pipeSet) closeAll() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, pc := range s.conns {
		if pc != nil {
			pc.close()
		}
	}
	s.conns = nil
}
