// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"
)

// frameBatch frames each arg as a request for the echo dispatcher and
// concatenates them, so the whole slice can go out in one client write.
func frameBatch(args [][]byte) []byte {
	var batch []byte
	for _, arg := range args {
		body := EncodeRequest("echo", arg)
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
		batch = append(append(batch, hdr[:]...), body...)
	}
	return batch
}

// The point of accumulating responses is that a pipelined batch leaves in ONE
// write, so count the writes. Counting client READS cannot prove it: TCP may
// split one write across several reads or coalesce several writes into one, so
// that assertion can fail on this implementation and pass on the old one.
func TestEpollBatchesPipelinedResponsesIntoOneWrite(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	args := [][]byte{{1}, {2}, {3}, {4}}
	before := epollWrites.Load()
	if _, err := c.Write(frameBatch(args)); err != nil {
		t.Fatal(err)
	}
	for i := range args {
		status, payload := readResp(t, c)
		if status != StatusOK || !bytes.Equal(payload, []byte{'o', 'k', ':', byte(i + 1)}) {
			t.Fatalf("frame %d: status=%d payload=%q", i, status, payload)
		}
	}

	// The four frames go out in one client write, so the server drains them in
	// one pass. Anything above one write means a reply left on its own.
	if n := epollWrites.Load() - before; n != 1 {
		t.Errorf("%d server writes for a batch of %d frames, want 1", n, len(args))
	}
}

// A trailing PARTIAL frame stops the drain, and the completed frames before it
// must still be written. Holding them until the rest of that frame arrives is
// the deadlock this restructure could have introduced: the client is waiting on
// the replies it has already earned, and the server is waiting on the client.
func TestEpollFlushesCompletedFramesWhenTheLastIsPartial(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	batch := frameBatch([][]byte{{1}, {2}})
	tail := frameBatch([][]byte{{3}})
	if _, err := c.Write(append(batch, tail[:3]...)); err != nil { // 2 whole frames + 3 bytes
		t.Fatal(err)
	}

	if err := c.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 2; i++ {
		status, payload := readResp(t, c)
		if status != StatusOK || !bytes.Equal(payload, []byte{'o', 'k', ':', byte(i)}) {
			t.Fatalf("frame %d: status=%d payload=%q", i, status, payload)
		}
	}

	// And the held-back frame completes normally once its rest arrives.
	if _, err := c.Write(tail[3:]); err != nil {
		t.Fatal(err)
	}
	if status, payload := readResp(t, c); status != StatusOK || !bytes.Equal(payload, []byte("ok:\x03")) {
		t.Fatalf("completed frame: status=%d payload=%q", status, payload)
	}
}

// Past epollWriteBatchBytes the drain flushes mid-loop and keeps accumulating.
// Every reply must still arrive, once, in request order.
func TestEpollBatchPastTheWriteThreshold(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Replies total ~4x the threshold, so the loop crosses it several times.
	const each = 2000
	args := make([][]byte, 4*epollWriteBatchBytes/each)
	for i := range args {
		arg := make([]byte, each)
		for j := range arg {
			arg[j] = byte(i*13 + j)
		}
		args[i] = arg
	}
	go func() { _, _ = c.Write(frameBatch(args)) }() // may exceed the socket buffer

	if err := c.SetReadDeadline(time.Now().Add(15 * time.Second)); err != nil {
		t.Fatal(err)
	}
	br := io.Reader(c)
	for i, arg := range args {
		status, payload := readResp(t, br)
		if status != StatusOK {
			t.Fatalf("frame %d: status=%d", i, status)
		}
		if want := append([]byte("ok:"), arg...); !bytes.Equal(payload, want) {
			t.Fatalf("frame %d: payload mismatch (len %d, want %d)", i, len(payload), len(want))
		}
	}
}

// appendResponse is the framing every epoll reply now goes through, so its
// output has to match what a reader expects byte for byte, including when it
// appends onto a buffer that already holds frames.
func TestAppendResponseFraming(t *testing.T) {
	var out []byte
	payloads := [][]byte{[]byte("first"), nil, bytes.Repeat([]byte("x"), 300)}
	statuses := []uint8{StatusOK, StatusError, StatusOK}
	for i, p := range payloads {
		out = appendResponse(out, statuses[i], p)
	}

	r := bytes.NewReader(out)
	for i, p := range payloads {
		status, payload := readResp(t, r)
		if status != statuses[i] {
			t.Errorf("frame %d: status=%d want %d", i, status, statuses[i])
		}
		if !bytes.Equal(payload, p) {
			t.Errorf("frame %d: payload=%q want %q", i, payload, p)
		}
	}
	if r.Len() != 0 {
		t.Errorf("%d trailing bytes after three frames", r.Len())
	}
}

// The invalid-length exit changed from an immediate close to a break-and-flush,
// and the flush-on-every-exit claim is only as good as its least-tested exit.
// Replies earned before the bad frame must still arrive.
func TestEpollFlushesRepliesEarnedBeforeAnInvalidFrame(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Two good frames, then a length the server must reject and close on.
	batch := frameBatch([][]byte{{1}, {2}})
	var bad [4]byte
	binary.BigEndian.PutUint32(bad[:], uint32(MaxFrameSize+1))
	if _, err := c.Write(append(batch, bad[:]...)); err != nil {
		t.Fatal(err)
	}

	if err := c.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 2; i++ {
		status, payload := readResp(t, c)
		if status != StatusOK || !bytes.Equal(payload, []byte{'o', 'k', ':', byte(i)}) {
			t.Fatalf("reply %d: status=%d payload=%q", i, status, payload)
		}
	}

	// And the connection is then closed, rather than left open on a bad frame.
	var scratch [1]byte
	if _, err := c.Read(scratch[:]); err == nil {
		t.Error("connection still readable after an invalid frame length")
	}
}
