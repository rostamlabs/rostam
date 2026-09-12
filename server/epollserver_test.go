// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"testing"
	"time"
)

// echoDisp is a minimal Dispatcher: it returns the args prefixed with "ok:", so a
// test client can verify the response corresponds to the request it sent.
type echoDisp struct{}

func (echoDisp) Call(_ string, args []byte) ([]byte, error) {
	return append([]byte("ok:"), args...), nil
}
func (echoDisp) LeaderAddr() string { return "" }

// startEpoll boots an EpollServer on a free loopback port and returns its addr
// plus a stop func. It waits until the listener accepts connections.
func startEpoll(t *testing.T, idle time.Duration) (addr string, stop func()) {
	t.Helper()
	return startEpollWith(t, echoDisp{}, idle)
}

// startEpollWith is startEpoll with a caller-supplied dispatcher, so a test can
// pick whether the loop takes the append path (canAppend) or the fallback.
func startEpollWith(t *testing.T, disp Dispatcher, idle time.Duration) (addr string, stop func()) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0") // grab a free port, then hand it to gnet
	if err != nil {
		t.Fatal(err)
	}
	addr = l.Addr().String()
	_ = l.Close()

	es := NewEpollServer(disp, nil, nil, 2, idle)
	if err := es.Start(addr); err != nil { // returns once bound (or on bind failure)
		t.Fatalf("epoll start: %v", err)
	}
	return addr, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = es.Stop(ctx)
	}
}

// writeTestFrame frames body as {len u32}{body} and writes it, optionally splitting
// the write into two pieces with a gap to exercise partial-frame buffering.
func writeTestFrame(t *testing.T, c net.Conn, body []byte, split bool) {
	t.Helper()
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
	frame := append(hdr[:], body...)
	if split && len(frame) > 6 {
		if _, err := c.Write(frame[:5]); err != nil { // header + 1 body byte
			t.Error(err)
			return
		}
		time.Sleep(time.Millisecond) // force a separate OnTraffic with a partial frame
		if _, err := c.Write(frame[5:]); err != nil {
			t.Error(err)
		}
		return
	}
	if _, err := c.Write(frame); err != nil {
		t.Error(err)
	}
}

// readResp reads one response frame and returns its status + payload.
func readResp(t *testing.T, r io.Reader) (uint8, []byte) {
	t.Helper()
	var h [4]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		t.Fatalf("read resp header: %v", err)
	}
	body := make([]byte, binary.BigEndian.Uint32(h[:]))
	if _, err := io.ReadFull(r, body); err != nil {
		t.Fatalf("read resp body: %v", err)
	}
	plen := binary.BigEndian.Uint32(body[1:5])
	return body[0], body[5 : 5+plen]
}

// TestEpollConcurrentFrames hammers the epoll transport with many concurrent
// connections issuing framed requests — some written in split pieces (partial
// frames), some pipelined two-at-a-time — and verifies every response matches the
// request that produced it. Run under -race to catch data races in the OnTraffic
// frame parser and gnet buffer handling.
func TestEpollConcurrentFrames(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	const conns, perConn = 24, 150
	var wg sync.WaitGroup
	for w := 0; w < conns; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			c, err := net.Dial("tcp", addr)
			if err != nil {
				t.Errorf("dial: %v", err)
				return
			}
			defer func() { _ = c.Close() }()
			for i := 0; i < perConn; i++ {
				arg := []byte{byte(w), byte(i), byte(i >> 8)}
				body := EncodeRequest("echo", arg)
				writeTestFrame(t, c, body, i%3 == 0) // split every 3rd frame
				status, payload := readResp(t, c)
				if status != StatusOK {
					t.Errorf("w%d i%d: status=%d", w, i, status)
					return
				}
				want := append([]byte("ok:"), arg...)
				if string(payload) != string(want) {
					t.Errorf("w%d i%d: payload=%q want %q", w, i, payload, want)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

// TestEpollPipelined verifies multiple frames written in a single syscall (the
// classic pipelining case) are each parsed and answered in order.
func TestEpollPipelined(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Three frames in one Write.
	var batch []byte
	for i := 0; i < 3; i++ {
		body := EncodeRequest("echo", []byte{byte(i)})
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
		batch = append(append(batch, hdr[:]...), body...)
	}
	if _, err := c.Write(batch); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		status, payload := readResp(t, c)
		if status != StatusOK || string(payload) != string(append([]byte("ok:"), byte(i))) {
			t.Fatalf("frame %d: status=%d payload=%q", i, status, payload)
		}
	}
}

// TestEpollIdleTimeout verifies the idle sweep closes a connection that goes
// silent longer than idleTimeout (slow-loris protection), and does NOT close an
// active one.
func TestEpollIdleTimeout(t *testing.T) {
	addr, stop := startEpoll(t, 300*time.Millisecond)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// One live request confirms the connection works and resets last-active.
	writeTestFrame(t, c, EncodeRequest("echo", []byte{1}), false)
	if s, _ := readResp(t, c); s != StatusOK {
		t.Fatalf("live request status=%d", s)
	}

	// Now go idle. The sweep (interval = idleTimeout/2) should close us well
	// within 3s. A successful Read of 0 bytes + io.EOF means the server closed us.
	_ = c.SetReadDeadline(time.Now().Add(3 * time.Second))
	if _, err := c.Read(make([]byte, 1)); err == nil {
		t.Fatal("expected the idle connection to be closed, but Read returned data")
	} else if ne, ok := err.(net.Error); ok && ne.Timeout() {
		t.Fatalf("idle connection was NOT closed within 3s (got a read timeout): %v", err)
	}
}

// appendEchoDisp is echoDisp that ALSO implements AppendDispatcher, so
// NewEpollServer sets canAppend and the loop actually exercises the reply
// buffer. echoDisp does not, which is why the plain epoll tests above never
// reach that branch at all.
//
// Two ops, because the reply buffer's whole contract turns on the difference:
//   - "echo" has an append variant. It writes into dst and reports appended, so
//     the payload came out of the connection's buffer (or, once it outgrew that,
//     out of the fresh array append gave it) and the transport may keep it.
//   - "foreign" has none. It is served normally and hands back memory the
//     dispatcher owns, standing in for the zero-copy page a read-only shard
//     returns from tx.Get. appended is false and the transport must NOT keep it.
type appendEchoDisp struct {
	page []byte // "the store's own memory" — returned as is, never written after init
}

func newAppendEchoDisp() *appendEchoDisp {
	d := &appendEchoDisp{page: make([]byte, 1024)}
	for i := range d.page {
		d.page[i] = 0xAA
	}
	return d
}

func (*appendEchoDisp) Call(name string, args []byte) ([]byte, error) {
	return append([]byte("ok:"), args...), nil
}
func (*appendEchoDisp) LeaderAddr() string { return "" }

func (d *appendEchoDisp) CallAppend(name string, args, dst []byte) ([]byte, bool, error) {
	if name == "foreign" {
		// No append handler: the reply is the dispatcher's own memory, returned
		// untouched, and appended is false.
		return d.page[:900], false, nil
	}
	return append(append(dst, "ok:"...), args...), true, nil
}

// Compile-time proof that appendEchoDisp really does trip canAppend. A method
// set is structural, so a signature change would otherwise leave these tests
// silently running the allocating fallback and covering nothing — which is
// exactly how the signature change in this branch built cleanly while every
// implementation had stopped satisfying the interface.
var _ AppendDispatcher = (*appendEchoDisp)(nil)

// TestEpollAppendPipelinedStraddlesBuffer pipelines frames on ONE connection
// whose replies alternate around the 512-byte initial reply buffer — small (fits,
// comes back in the buffer), large (outgrows it, so append returns a new array
// the connection keeps), small, larger still — and checks every response byte for
// byte. That is the invariant the buffer reuse rests on and which otherwise lives
// only in comments: the loop copies each payload into its response frame before
// dispatching the next request, so replacing the buffer between frames is safe.
func TestEpollAppendPipelinedStraddlesBuffer(t *testing.T) {
	addr, stop := startEpollWith(t, newAppendEchoDisp(), 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Reply length is len("ok:") + len(arg); the connection starts with a 512-byte
	// buffer, so 8 fits and 900 / 2000 do not.
	sizes := []int{8, 900, 16, 2000, 8, 900}
	args := make([][]byte, len(sizes))
	var batch []byte
	for i, n := range sizes {
		arg := make([]byte, n)
		for j := range arg {
			arg[j] = byte(i*7 + j) // a distinct pattern per frame
		}
		args[i] = arg
		body := EncodeRequest("echo", arg)
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
		batch = append(append(batch, hdr[:]...), body...)
	}
	if _, err := c.Write(batch); err != nil { // all frames in one syscall
		t.Fatal(err)
	}
	for i, arg := range args {
		status, payload := readResp(t, c)
		if status != StatusOK {
			t.Fatalf("frame %d (arg %d bytes): status=%d", i, len(arg), status)
		}
		want := append([]byte("ok:"), arg...)
		if !bytes.Equal(payload, want) {
			t.Fatalf("frame %d (arg %d bytes): payload len=%d want %d", i, len(arg), len(payload), len(want))
		}
	}
}

// TestEpollDoesNotAdoptUnappendedReply is the safety property the appended flag
// exists to enforce, and nothing else pins it.
//
// A reply from an op with no append variant is memory the STORE owns — here the
// dispatcher's page, standing in for the zero-copy page a PolicyRejectWrites
// shard returns from tx.Get. It is bigger than the connection's 512-byte buffer,
// so a transport that keyed the buffer swap on size alone would adopt it; the
// very next append would then write through into the store's own memory.
//
// So: ask for that reply, then ask for one big enough to be appended into
// whatever buffer the connection is now holding, and check the page is still
// untouched. Checked by mutation — keying the swap on `callErr == nil` instead of
// `appended`, which is the bug this branch fixed, fails it.
func TestEpollDoesNotAdoptUnappendedReply(t *testing.T) {
	disp := newAppendEchoDisp()
	addr, stop := startEpollWith(t, disp, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// 1. The foreign reply: 900 bytes of the dispatcher's own page, appended=false.
	writeTestFrame(t, c, EncodeRequest("foreign", nil), false)
	status, payload := readResp(t, c)
	if status != StatusOK {
		t.Fatalf("foreign: status=%d", status)
	}
	if len(payload) != 900 || !bytes.Equal(payload, bytes.Repeat([]byte{0xAA}, 900)) {
		t.Fatalf("foreign: got %d bytes, want 900 of 0xAA", len(payload))
	}

	// 2. A reply the append path DOES build, long enough to overwrite the page had
	// the connection adopted it.
	arg := bytes.Repeat([]byte{0x5A}, 700)
	writeTestFrame(t, c, EncodeRequest("echo", arg), false)
	status, payload = readResp(t, c)
	if status != StatusOK {
		t.Fatalf("echo: status=%d", status)
	}
	if want := append([]byte("ok:"), arg...); !bytes.Equal(payload, want) {
		t.Fatalf("echo: payload len=%d want %d", len(payload), len(want))
	}

	// 3. The page must be exactly as the dispatcher left it.
	for i, b := range disp.page {
		if b != 0xAA {
			t.Fatalf("the connection adopted a reply it was not handed: store memory "+
				"overwritten at byte %d (0x%02X, want 0xAA)", i, b)
		}
	}
}
