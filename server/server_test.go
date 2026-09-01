// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// Compile-time guard: *shard.Store satisfies the Dispatcher interface.
var _ Dispatcher = (*shard.Store)(nil)

func newTestStore(t *testing.T) *shard.Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	store, err := shard.New(shard.Config{
		NodeID: "node1", DataDir: t.TempDir(),
		Cache: cc, Ops: reg,
		Bootstrap:       true,
		RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if store.IsLeader() {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !store.IsLeader() {
		t.Fatal("store never became leader")
	}
	return store
}

func newTestServer(t *testing.T) (*Server, *shard.Store) {
	t.Helper()
	store := newTestStore(t)
	srv, err := New(Config{
		Addr: "127.0.0.1:0", Dispatcher: store,
		MaxConns: 100, IdleTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })
	return srv, store
}

// rawCall opens a fresh TCP connection, sends one request frame, reads
// one response frame, closes. Used only in tests.
func rawCall(t *testing.T, addr string, op string, args []byte) (uint8, []byte) {
	t.Helper()
	c, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = c.Close() }()
	w := bufio.NewWriter(c)
	body := EncodeRequest(op, args)
	if err := writeFrame(w, body); err != nil {
		t.Fatalf("write frame: %v", err)
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	r := bufio.NewReader(c)
	respBody, err := readFrame(r)
	if err != nil {
		t.Fatalf("read frame: %v", err)
	}
	status, payload, err := DecodeResponse(*respBody)
	if err != nil {
		t.Fatalf("decode response: %v", err)
	}
	cp := make([]byte, len(payload))
	copy(cp, payload)
	putFrameBuf(respBody)
	return status, cp
}

func TestServerPutGet(t *testing.T) {
	srv, _ := newTestServer(t)
	addr := srv.Addr().String()

	status, _ := rawCall(t, addr, "put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))
	if status != StatusOK {
		t.Fatalf("put status = %d, want OK", status)
	}
	status, payload := rawCall(t, addr, "get", ops.EncodeKeyArgs([]byte("k")))
	if status != StatusOK {
		t.Fatalf("get status = %d, want OK", status)
	}
	if !bytes.Equal(payload, []byte("v")) {
		t.Fatalf("get payload = %q, want v", payload)
	}
}

func TestServerGetMissingReturnsNotFound(t *testing.T) {
	srv, _ := newTestServer(t)
	addr := srv.Addr().String()
	status, _ := rawCall(t, addr, "get", ops.EncodeKeyArgs([]byte("absent")))
	if status != StatusNotFound {
		t.Fatalf("missing get status = %d, want NotFound (1)", status)
	}
}

func TestServerUnknownOpReturnsError(t *testing.T) {
	srv, _ := newTestServer(t)
	addr := srv.Addr().String()
	status, payload := rawCall(t, addr, "no_such_op", nil)
	if status != StatusError {
		t.Fatalf("unknown op status = %d, want Error", status)
	}
	msg, err := DecodeErrorPayload(payload)
	if err != nil {
		t.Fatalf("decode err payload: %v", err)
	}
	if msg == "" {
		t.Fatal("error payload empty")
	}
}

func TestServerPingRoundtrip(t *testing.T) {
	srv, _ := newTestServer(t)
	addr := srv.Addr().String()
	status, payload := rawCall(t, addr, "__ping__", nil)
	if status != StatusOK {
		t.Fatalf("ping status = %d, want OK", status)
	}
	if len(payload) != 0 {
		t.Fatalf("ping payload = %v, want empty", payload)
	}
}

func TestServerConcurrentClients(t *testing.T) {
	srv, _ := newTestServer(t)
	addr := srv.Addr().String()
	const clients = 32
	const iters = 50
	var wg sync.WaitGroup
	for i := range clients {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := range iters {
				k := []byte{byte(id % 256), byte(j % 256)} //nolint:gosec // id and j are bounded by clients/iters constants
				putArgs := ops.EncodePutArgs(k, []byte{1}, 0)
				if st, _ := rawCall(t, addr, "put", putArgs); st != StatusOK {
					t.Errorf("put id=%d j=%d status=%d", id, j, st)
					return
				}
			}
		}(i)
	}
	wg.Wait()
}

func TestServerCloseStopsAccept(t *testing.T) {
	srv, _ := newTestServer(t)
	addr := srv.Addr().String()
	if err := srv.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, err := net.DialTimeout("tcp", addr, 200*time.Millisecond)
	if err == nil {
		t.Fatal("Dial after Close succeeded; expected refusal")
	}
}

// keep ctx + errors imports live
var _ = context.Background
var _ = errors.New
