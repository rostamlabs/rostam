// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

func startTestStack(t *testing.T) (string, func()) {
	t.Helper()
	// The embedded shard store dispatches ops locally, so it needs the
	// full handler-carrying ops.Registry (not the client's routing-only
	// wire.Registry, which the client-construction tests use instead).
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

	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: store})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()

	return srv.Addr().String(), func() {
		_ = srv.Close()
		_ = store.Close()
	}
}

func TestClientPutGet(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put: %v", err)
	}
	res, err := c.Call(ctx, "get", wire.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("get result = %q, want v", res)
	}
}

// Named to avoid colliding with the root Store-API test of the same intent
// (client_test.go's TestClientGetMissingReturnsErrNotFound); this one exercises
// the typed client's Call path directly.
func TestClientCallGetMissingReturnsErrNotFound(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	_, err := c.Call(context.Background(), "get", wire.EncodeKeyArgs([]byte("absent")))
	if !errors.Is(err, client.ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestClientUnknownOpReturnsRemoteError(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	_, err := c.Call(context.Background(), "no_such_op", nil)
	var rErr *client.RemoteError
	if !errors.As(err, &rErr) {
		t.Fatalf("err type = %T (%v), want *RemoteError", err, err)
	}
	if rErr.Op != "no_such_op" || rErr.Msg == "" {
		t.Fatalf("RemoteError = %+v", rErr)
	}
}

func TestClientIncrAccumulates(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()
	for range 3 {
		_, err := c.Call(ctx, "incr", wire.EncodeIncrArgs([]byte("counter"), 2))
		if err != nil {
			t.Fatal(err)
		}
	}
	res, _ := c.Call(ctx, "incr", wire.EncodeIncrArgs([]byte("counter"), -1))
	v, _ := wire.DecodeIncrResult(res)
	if v != 5 {
		t.Fatalf("incr accumulated = %d, want 5", v)
	}
}

func TestClientConditionalWrites(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	// SetNX: first wins, second (key present) is refused.
	stored, err := c.SetNX(ctx, []byte("k"), []byte("v1"), 0)
	if err != nil {
		t.Fatalf("SetNX: %v", err)
	}
	if !stored {
		t.Fatal("first SetNX = false, want true")
	}
	stored, err = c.SetNX(ctx, []byte("k"), []byte("v2"), 0)
	if err != nil {
		t.Fatalf("SetNX 2: %v", err)
	}
	if stored {
		t.Fatal("second SetNX on present key = true, want false")
	}

	// CAS: correct expected succeeds, wrong expected fails.
	swapped, err := c.CAS(ctx, []byte("k"), []byte("v2"), []byte("v1"), 0)
	if err != nil {
		t.Fatalf("CAS: %v", err)
	}
	if !swapped {
		t.Fatal("CAS with correct expected = false, want true")
	}
	swapped, err = c.CAS(ctx, []byte("k"), []byte("v3"), []byte("WRONG"), 0)
	if err != nil {
		t.Fatalf("CAS 2: %v", err)
	}
	if swapped {
		t.Fatal("CAS with wrong expected = true, want false")
	}

	// CompareAndDelete: wrong token no-ops, right token deletes.
	deleted, err := c.CompareAndDelete(ctx, []byte("k"), []byte("WRONG"))
	if err != nil {
		t.Fatalf("CompareAndDelete wrong: %v", err)
	}
	if deleted {
		t.Fatal("CompareAndDelete with wrong token = true, want false")
	}
	deleted, err = c.CompareAndDelete(ctx, []byte("k"), []byte("v2"))
	if err != nil {
		t.Fatalf("CompareAndDelete right: %v", err)
	}
	if !deleted {
		t.Fatal("CompareAndDelete with right token = false, want true")
	}
	if _, err := c.Call(ctx, "get", wire.EncodeKeyArgs([]byte("k"))); !errors.Is(err, client.ErrNotFound) {
		t.Fatalf("post-delete get: err = %v, want ErrNotFound", err)
	}
}

func TestClientSetNXExactlyOneWinner(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}, MaxConnsPerServer: 8})
	defer func() { _ = c.Close() }()

	const goroutines = 24
	var wins atomic.Int64
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			ok, err := c.SetNX(context.Background(), []byte("race-key"), []byte("mine"), 0)
			if err != nil {
				t.Errorf("SetNX: %v", err)
				return
			}
			if ok {
				wins.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()
	if got := wins.Load(); got != 1 {
		t.Fatalf("SetNX winners = %d, want exactly 1", got)
	}
}

func TestClientConcurrentCalls(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}, MaxConnsPerServer: 4})
	defer func() { _ = c.Close() }()

	var wg sync.WaitGroup
	const goroutines = 16
	const iters = 50
	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := context.Background()
			for i := range iters {
				k := []byte{byte(id), byte(i)} //nolint:gosec // id and i are bounded by goroutines/iters constants (16, 50)
				if _, err := c.Call(ctx, "put", wire.EncodePutArgs(k, []byte{1}, 0)); err != nil {
					t.Errorf("put id=%d i=%d: %v", id, i, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}

func TestClientCloseIdempotent(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	if err := c.Close(); err != nil {
		t.Fatalf("Close 1: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close 2: %v", err)
	}
}

// TestNewRoutedKeyedRoundTrip exercises the routed client's Call path against a
// real node stack: a keyed put must round-trip. The routing-registry wiring
// itself is asserted white-box in the client module
// (client.TestNewRoutedWiresRoutingRegistry); this engine-backed half lives here
// because it needs the embedded stack the engine-free client module cannot import.
func TestNewRoutedKeyedRoundTrip(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()

	c, err := client.NewRouted(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put via routed client: %v", err)
	}
}
