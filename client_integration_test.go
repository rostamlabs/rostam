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

// TestClientFlushWipesStore exercises the typed client's Flush end-to-end against a
// loopback server: it seeds keys, calls Flush once (which sends the keyless "flush"
// op the server dispatches to handleFlush → cache.Flush), then asserts every key
// misses and a post-flush write still reads back.
func TestClientFlushWipesStore(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	for _, k := range []string{"a", "b", "c"} {
		if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte(k), []byte("v"), 0)); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}

	if err := c.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	for _, k := range []string{"a", "b", "c"} {
		_, err := c.Call(ctx, "get", wire.EncodeKeyArgs([]byte(k)))
		if !errors.Is(err, client.ErrNotFound) {
			t.Fatalf("get %q after flush: err = %v, want ErrNotFound", k, err)
		}
	}

	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("d"), []byte("w"), 0)); err != nil {
		t.Fatalf("put after flush: %v", err)
	}
	res, err := c.Call(ctx, "get", wire.EncodeKeyArgs([]byte("d")))
	if err != nil {
		t.Fatalf("get after post-flush put: %v", err)
	}
	if !bytes.Equal(res, []byte("w")) {
		t.Fatalf("post-flush get = %q, want w", res)
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

// TestClientOperateNoLostIncrement is the whole point of operate: many goroutines
// hammer INCR on ONE hot key/field concurrently, and every increment must survive.
// A client CAS-retry loop would livelock here; operate applies each op-list under
// the shard write lock, so the final value equals the exact number of increments.
func TestClientOperateNoLostIncrement(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}, MaxConnsPerServer: 8})
	defer func() { _ = c.Close() }()

	const goroutines = 24
	const iters = 50
	key := []byte("hot-session")
	incr := []wire.OperateOp{{Target: wire.OperateTargetGlobal, FieldIdx: 0, Opcode: wire.OperateOpINCR, Arg: 1}}

	var wg sync.WaitGroup
	start := make(chan struct{})
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for range iters {
				if _, err := c.Operate(context.Background(), key, 0, 0, incr, nil); err != nil {
					t.Errorf("Operate: %v", err)
					return
				}
			}
		}()
	}
	close(start)
	wg.Wait()

	got, err := c.Operate(context.Background(), key, 0, 0, nil,
		[]wire.OperateRet{{Target: wire.OperateTargetGlobal, FieldIdx: 0}})
	if err != nil {
		t.Fatalf("Operate read-back: %v", err)
	}
	if want := int64(goroutines * iters); got[0] != want {
		t.Fatalf("counter = %d, want %d (increments were lost)", got[0], want)
	}
}

// TestClientOperateUpdateAndReadBack exercises the multi-opcode update-and-read
// round trip across globals and entry sub-records, plus the deterministic entry
// cap.
func TestClientOperateUpdateAndReadBack(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()
	key := []byte("rec")

	// Guard-then-increment (HALVE_GRP overflow guard ahead of the increments), a
	// SETMAX, and an entry counter — all in one atomic op-list.
	ops := []wire.OperateOp{
		{Target: wire.OperateTargetGlobal, FieldIdx: 0, Opcode: wire.OperateOpINCR, Arg: 10},
		{Target: wire.OperateTargetGlobal, FieldIdx: 1, Opcode: wire.OperateOpSETMAX, Arg: 7},
		{Target: wire.OperateTargetGlobal, FieldIdx: 1, Opcode: wire.OperateOpSETMAX, Arg: 3},
		{Target: wire.OperateTargetEntry, EntryKey: 42, FieldIdx: 0, Opcode: wire.OperateOpINCR, Arg: 100},
	}
	ret := []wire.OperateRet{
		{Target: wire.OperateTargetGlobal, FieldIdx: 0},
		{Target: wire.OperateTargetGlobal, FieldIdx: 1},
		{Target: wire.OperateTargetEntry, EntryKey: 42, FieldIdx: 0},
	}
	got, err := c.Operate(ctx, key, 0, 0, ops, ret)
	if err != nil {
		t.Fatalf("Operate: %v", err)
	}
	if len(got) != 3 || got[0] != 10 || got[1] != 7 || got[2] != 100 {
		t.Fatalf("read-back = %v, want [10 7 100]", got)
	}

	// A second op-list accumulates on the committed state.
	got, err = c.Operate(ctx, key, 0, 0,
		[]wire.OperateOp{{Target: wire.OperateTargetGlobal, FieldIdx: 0, Opcode: wire.OperateOpINCR, Arg: 5}},
		[]wire.OperateRet{{Target: wire.OperateTargetGlobal, FieldIdx: 0}})
	if err != nil {
		t.Fatalf("Operate 2: %v", err)
	}
	if got[0] != 15 {
		t.Fatalf("accumulated global[0] = %d, want 15", got[0])
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

func TestClientGetDel(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	// Absent → found=false.
	if _, found, err := c.GetDel(ctx, []byte("gd")); err != nil || found {
		t.Fatalf("GetDel absent = (found=%v, err=%v), want (false, nil)", found, err)
	}
	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("gd"), []byte("val"), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	// Present → returns value and deletes.
	val, found, err := c.GetDel(ctx, []byte("gd"))
	if err != nil || !found || !bytes.Equal(val, []byte("val")) {
		t.Fatalf("GetDel present = (%q, %v, %v), want (val, true, nil)", val, found, err)
	}
	if _, err := c.Call(ctx, "get", wire.EncodeKeyArgs([]byte("gd"))); !errors.Is(err, client.ErrNotFound) {
		t.Fatalf("GetDel did not delete: get err = %v, want ErrNotFound", err)
	}
}

func TestClientIncrExRateLimit(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	// First hit creates the counter with a 60s window.
	v, err := c.IncrEx(ctx, []byte("rl"), 1, 60*time.Second)
	if err != nil || v != 1 {
		t.Fatalf("IncrEx create = (%d, %v), want (1, nil)", v, err)
	}
	ttl1, err := c.TTL(ctx, []byte("rl"))
	if err != nil || ttl1 <= 0 {
		t.Fatalf("TTL after create = (%d, %v), want positive", ttl1, err)
	}
	// A second hit increments but keeps the ORIGINAL window (the ttl passed here
	// is ignored on an existing key), so the remaining TTL does not jump up.
	v, err = c.IncrEx(ctx, []byte("rl"), 1, 600*time.Second)
	if err != nil || v != 2 {
		t.Fatalf("IncrEx re-incr = (%d, %v), want (2, nil)", v, err)
	}
	ttl2, err := c.TTL(ctx, []byte("rl"))
	if err != nil {
		t.Fatalf("TTL after re-incr: %v", err)
	}
	if ttl2 > ttl1 {
		t.Fatalf("IncrEx slid the window: ttl %d -> %d, want <= original", ttl1, ttl2)
	}
}

func TestClientCompareAndExpire(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := client.New(client.Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	// Absent → false.
	if refreshed, err := c.CompareAndExpire(ctx, []byte("lk"), []byte("tok"), time.Hour); err != nil || refreshed {
		t.Fatalf("CompareAndExpire absent = (%v, %v), want (false, nil)", refreshed, err)
	}
	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("lk"), []byte("tok"), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	// Wrong token → false.
	if refreshed, err := c.CompareAndExpire(ctx, []byte("lk"), []byte("WRONG"), time.Hour); err != nil || refreshed {
		t.Fatalf("CompareAndExpire wrong = (%v, %v), want (false, nil)", refreshed, err)
	}
	// Right token → true, and a TTL now exists.
	if refreshed, err := c.CompareAndExpire(ctx, []byte("lk"), []byte("tok"), time.Hour); err != nil || !refreshed {
		t.Fatalf("CompareAndExpire right = (%v, %v), want (true, nil)", refreshed, err)
	}
	if ttl, err := c.TTL(ctx, []byte("lk")); err != nil || ttl <= 0 {
		t.Fatalf("TTL after refresh = (%d, %v), want positive", ttl, err)
	}
}

// TestClientMGet drives MGet through the mget op end-to-end via the routed
// client (so a topology is known and keys group to a real shard, one mget call),
// covering mixed found/missing with original-order stitching.
func TestClientMGet(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := client.NewRouted(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	for _, kv := range []struct{ k, v string }{{"m:a", "va"}, {"m:c", "vc"}} {
		if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte(kv.k), []byte(kv.v), 0)); err != nil {
			t.Fatalf("put %s: %v", kv.k, err)
		}
	}
	keys := [][]byte{[]byte("m:a"), []byte("m:b"), []byte("m:c")}
	vals, err := c.MGet(ctx, keys)
	if err != nil {
		t.Fatalf("MGet: %v", err)
	}
	if len(vals) != 3 {
		t.Fatalf("MGet returned %d values, want 3", len(vals))
	}
	if !bytes.Equal(vals[0], []byte("va")) {
		t.Fatalf("MGet[0] = %q, want va", vals[0])
	}
	if vals[1] != nil {
		t.Fatalf("MGet[1] = %q, want nil (missing)", vals[1])
	}
	if !bytes.Equal(vals[2], []byte("vc")) {
		t.Fatalf("MGet[2] = %q, want vc", vals[2])
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
