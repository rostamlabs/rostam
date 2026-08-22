// SPDX-License-Identifier: Apache-2.0
package rostam

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"testing"
)

// cannedValueServer answers every request with the SAME StatusOK response
// carrying `value` as the payload, reusing its read buffer so the server side
// allocates nothing per request. This isolates the CLIENT-side allocation count
// of GetInto over a real TCP round-trip.
func cannedValueServer(tb testing.TB, value []byte) (addr string, stop func()) {
	tb.Helper()
	// body: [status:1=OK][payloadLen:u32][value]; frame: [frameLen:u32][body].
	body := make([]byte, 1+4+len(value))
	body[0] = 0
	binary.BigEndian.PutUint32(body[1:5], uint32(len(value)))
	copy(body[5:], value)
	frame := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(frame[0:4], uint32(len(body)))
	copy(frame[4:], body)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatal(err)
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				hdr := make([]byte, 4)
				var reqBody []byte
				for {
					if _, err := io.ReadFull(c, hdr); err != nil {
						return
					}
					n := binary.BigEndian.Uint32(hdr)
					if cap(reqBody) < int(n) {
						reqBody = make([]byte, n)
					}
					if _, err := io.ReadFull(c, reqBody[:n]); err != nil {
						return
					}
					if _, err := c.Write(frame); err != nil {
						return
					}
				}
			}(conn)
		}
	}()
	return ln.Addr().String(), func() { _ = ln.Close() }
}

// TestGetIntoZeroAllocClientSide guards the KV read promise: with a reused dst,
// a GetInto round-trip allocates nothing on the client — pooled request args, no
// defensive payload copy, value copied straight into dst. It broke before
// AppendKeyArgs + the kv args pool + the GetInto CallFunc path landed.
func TestGetIntoZeroAllocClientSide(t *testing.T) {
	value := []byte("a-cached-value-of-modest-size-0123456789")
	addr, stop := cannedValueServer(t, value)
	defer stop()

	cli, err := NewClient(ClientConfig{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cli.Close() }()

	ctx := context.Background()
	key := []byte("some-key")
	dst := make([]byte, 0, len(value))

	// Warm up: first call dials the conn and sizes pool/read/args buffers.
	dst, err = cli.GetInto(ctx, key, dst[:0])
	if err != nil {
		t.Fatal(err)
	}
	if string(dst) != string(value) {
		t.Fatalf("got %q, want %q", dst, value)
	}

	allocs := testing.AllocsPerRun(2000, func() {
		var e error
		dst, e = cli.GetInto(ctx, key, dst[:0])
		if e != nil {
			t.Fatal(e)
		}
	})
	if allocs != 0 {
		t.Fatalf("GetInto: got %v allocs/op, want 0", allocs)
	}
}
