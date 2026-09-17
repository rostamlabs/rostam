// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"
)

// pipeEchoServer answers each framed request with the args it carried, so a
// pipelined caller can check it got ITS answer and not a neighbour's. It reads
// into one buffer and records how many frames arrived per read, which is the
// coalescing this file is about.
type pipeEchoServer struct {
	ln net.Listener

	mu       sync.Mutex
	reads    int
	frames   int
	maxBatch int
}

func startPipeEchoServer(t *testing.T) *pipeEchoServer {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := &pipeEchoServer{ln: ln}
	go s.accept()
	t.Cleanup(func() { _ = ln.Close() })
	return s
}

func (s *pipeEchoServer) accept() {
	for {
		c, err := s.ln.Accept()
		if err != nil {
			return
		}
		go s.serve(c)
	}
}

// serve reads raw, so it can count frames per read rather than per request the
// way a bufio.Reader would hide.
func (s *pipeEchoServer) serve(c net.Conn) {
	defer func() { _ = c.Close() }()
	buf := make([]byte, 64<<10)
	var pending []byte
	for {
		n, err := c.Read(buf)
		if n > 0 {
			pending = append(pending, buf[:n]...)
			batch := 0
			var out []byte
			for {
				if len(pending) < 4 {
					break
				}
				fl := int(binary.BigEndian.Uint32(pending[:4]))
				if len(pending) < 4+fl {
					break
				}
				out = appendEchoReply(out, pending[4:4+fl])
				pending = pending[4+fl:]
				batch++
			}
			if batch > 0 {
				s.mu.Lock()
				s.reads++
				s.frames += batch
				if batch > s.maxBatch {
					s.maxBatch = batch
				}
				s.mu.Unlock()
				if _, werr := c.Write(out); werr != nil {
					return
				}
			}
		}
		if err != nil {
			return
		}
	}
}

// appendEchoReply parses one request body and appends a StatusOK reply carrying
// its args. Wire in: [0x02 tokenLen token]? [opLen op] [argsLen u32] [args].
func appendEchoReply(dst, body []byte) []byte {
	p := 0
	if len(body) > 0 && body[0] == 0x02 {
		p = 2 + int(body[1])
	}
	p += 1 + int(body[p]) // opLen + op
	args := body[p+4:]

	var hdr [9]byte
	binary.BigEndian.PutUint32(hdr[0:4], uint32(1+4+len(args)))
	hdr[4] = 0 // StatusOK
	binary.BigEndian.PutUint32(hdr[5:9], uint32(len(args)))
	return append(append(dst, hdr[:]...), args...)
}

func (s *pipeEchoServer) stats() (reads, frames, maxBatch int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reads, s.frames, s.maxBatch
}

func dialTestPipe(t *testing.T, addr string, depth int) *pipeConn {
	t.Helper()
	pc, err := dialPipeConn(context.Background(), addr, "", depth, 5*time.Second,
		&net.Dialer{Timeout: 2 * time.Second}, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(pc.close)
	return pc
}

// The flush is now conditional, so the case with nothing to coalesce WITH is the
// one that can hang: a lone caller must still put its frame on the wire rather
// than wait for a second caller that never comes.
func TestPipelineLoneCallStillFlushes(t *testing.T) {
	s := startPipeEchoServer(t)
	pc := dialTestPipe(t, s.ln.Addr().String(), 8)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	status, payload, err := pc.call(ctx, "echo", []byte("alone"))
	if err != nil {
		t.Fatalf("call: %v", err) // a stranded frame shows up here as a timeout
	}
	if status != 0 || !bytes.Equal(payload, []byte("alone")) {
		t.Fatalf("status=%d payload=%q", status, payload)
	}
}

// Responses are correlated by FIFO position, not by request id, so coalescing
// concurrent writers must not disturb the order frames reach the wire in. Each
// caller sends a distinct payload and must get that same payload back.
func TestPipelineConcurrentCallsKeepTheirOwnAnswers(t *testing.T) {
	s := startPipeEchoServer(t)
	pc := dialTestPipe(t, s.ln.Addr().String(), 32)

	const callers = 64
	var wg sync.WaitGroup
	errs := make(chan error, callers)
	start := make(chan struct{})
	for i := range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start // release together, so the writers actually contend
			want := fmt.Appendf(nil, "payload-%04d", i)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			status, got, err := pc.call(ctx, "echo", want)
			switch {
			case err != nil:
				errs <- fmt.Errorf("caller %d: %w", i, err)
			case status != 0:
				errs <- fmt.Errorf("caller %d: status %d", i, status)
			case !bytes.Equal(got, want):
				errs <- fmt.Errorf("caller %d: got %q want %q", i, got, want)
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	// Reported, not asserted: how much coalescing happens depends on the
	// scheduler, and a machine that runs the callers strictly one after another
	// would legitimately see none. The live A/B is what measures the saving.
	reads, frames, maxBatch := s.stats()
	t.Logf("server saw %d frames in %d reads (max %d frames in one read)", frames, reads, maxBatch)
	if frames != callers {
		t.Errorf("server saw %d frames, want %d", frames, callers)
	}
}

// CallFunc is the zero-copy path the latency-sensitive callers use, and it had
// its own connection-acquire that never consulted PipelineDepth. So a caller
// could set the knob, see 64 pooled connections, and measure no change --
// pipelining was simply off for it. Nothing else pins this.
func TestCallFuncHonoursPipelineDepth(t *testing.T) {
	s := startPipeEchoServer(t)
	c, err := New(Config{
		Servers:           []string{s.ln.Addr().String()},
		PipelineDepth:     8,
		PipelineConns:     1,
		MaxConnsPerServer: 16,
		CallTimeout:       5 * time.Second,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	want := []byte("through-the-pipe")
	var got []byte
	if err := c.CallFunc(ctx, "echo", want, func(payload []byte) error {
		got = append(got[:0], payload...) // payload is only ours for the callback
		return nil
	}); err != nil {
		t.Fatalf("CallFunc: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}

	c.pipeMu.Lock()
	sets := len(c.pipeSets)
	c.pipeMu.Unlock()
	if sets == 0 {
		t.Error("CallFunc opened no pipelined set: it took the pooled path despite PipelineDepth > 0")
	}
}
