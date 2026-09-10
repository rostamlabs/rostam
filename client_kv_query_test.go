// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

// startKVQueryStack is startTestStack with the store handed back, so a test can
// install a KV index definition on it.
//
// A shard store has a KV record index but NO WAY TO DEFINE ONE: the catalog ops
// live in cluster (the meta log is what makes a definition cluster-wide), and
// the observer that installs definitions into a store's Set runs there too. So
// the definition is installed directly through Store.KVIndex(), which is exactly
// what the observer does — this exercises the CLIENT and the transport against a
// real dispatcher, which is the part cluster's own tests do not cover.
func startKVQueryStack(t *testing.T) (string, *shard.Store, func()) {
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
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && !store.IsLeader() {
		time.Sleep(20 * time.Millisecond)
	}
	if !store.IsLeader() {
		_ = store.Close()
		t.Fatal("store never became leader")
	}
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: store})
	if err != nil {
		_ = store.Close()
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	return srv.Addr().String(), store, func() {
		_ = srv.Close()
		_ = store.Close()
	}
}

// kvQueryUserRecord is the dynamic-mode record each seeded key holds.
func kvQueryUserRecord(t *testing.T, age int64) []byte {
	t.Helper()
	rec := &wire.Record{
		Mode:   wire.OperateModeDynamic,
		Fields: []wire.Field{{Name: "age", Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(age)}}},
	}
	return rec.Encode()
}

// TestClientKVQueryEndToEnd runs the typed client's KVQuery over the real binary
// transport against a real store: seed records, define and backfill an index,
// then page a filtered query to exhaustion and read the decoded records back.
//
// This is the only place the client's KVQuery meets an actual kv_query handler.
// The client's own tests answer from a fake server (that module cannot import
// the engine at all — TestClientIsEngineFree locks it), so without this the
// args the client encodes and the result it decodes were never checked against
// the code that produces and consumes them.
func TestClientKVQueryEndToEnd(t *testing.T) {
	addr, store, stop := startKVQueryStack(t)
	defer stop()

	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	const n = 7
	for i := 0; i < n; i++ {
		key := fmt.Appendf(nil, "u:%03d", i)
		if _, perr := c.Call(ctx, "put", wire.EncodePutArgs(key, kvQueryUserRecord(t, int64(i)), 0)); perr != nil {
			t.Fatalf("put %s: %v", key, perr)
		}
	}
	// Define the index over the seeded keys and backfill it, the two steps the
	// cluster observer performs when a definition reaches a node.
	def, derr := kvindex.DefFrom(wire.KVIndexDef{
		Name: "by_age", KeyPrefix: []byte("u:"), PayloadPath: "age",
		Kind: wire.KVIndexKindScalar, Enabled: true,
	}, 1)
	if derr != nil {
		t.Fatal(derr)
	}
	idx := store.KVIndex()
	idx.Install([]kvindex.Def{def})
	if berr := idx.Backfill("by_age", store.CacheWalker()); berr != nil {
		t.Fatalf("backfill: %v", berr)
	}

	args := wire.KVQueryArgs{
		Index:  "by_age",
		Filter: vtypes.Filter{Op: vtypes.FilterGte, Field: "age", Value: vtypes.NewInt(0)},
		Limit:  3,
		Return: wire.KVQueryReturnRecords,
	}
	seen := map[string]int64{}
	for page := 0; page < 10; page++ {
		p, qerr := c.KVQuery(ctx, args)
		if qerr != nil {
			t.Fatalf("page %d: %v", page, qerr)
		}
		if len(p.Records) != len(p.Rows) {
			t.Fatalf("page %d: %d records for %d rows", page, len(p.Records), len(p.Rows))
		}
		for i, row := range p.Rows {
			rec := p.Records[i]
			if rec == nil || len(rec.Fields) != 1 {
				t.Fatalf("row %q decoded to %+v, want one field", row.Key, rec)
			}
			seen[string(row.Key)] = int64(rec.Fields[0].Cell.U)
		}
		if len(p.Cursor) == 0 {
			break
		}
		args.Cursor = p.Cursor
	}
	if len(seen) != n {
		t.Fatalf("paged %d keys, want %d: %v", len(seen), n, seen)
	}
	for i := 0; i < n; i++ {
		key := fmt.Sprintf("u:%03d", i)
		if seen[key] != int64(i) {
			t.Fatalf("%s decoded age %d, want %d", key, seen[key], i)
		}
	}
}

// TestClientKVQueryEndToEndUnknownIndex pins that the client's typed error
// survives the real transport: the leaf's refusal crosses the binary edge
// unredacted (server.clientFacingErr) and the client re-types it, so a caller
// can errors.Is it instead of reading the message.
func TestClientKVQueryEndToEndUnknownIndex(t *testing.T) {
	addr, _, stop := startKVQueryStack(t)
	defer stop()

	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	_, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{
		Index:  "nope",
		Filter: vtypes.Filter{Op: vtypes.FilterGte, Field: "age", Value: vtypes.NewInt(0)},
	})
	if !errors.Is(qerr, client.ErrKVIndexNotFound) {
		t.Fatalf("KVQuery against an undefined index = %v, want client.ErrKVIndexNotFound", qerr)
	}
}

// TestClientKVQueryEndToEndBadFilterIsPermanent is the other half: a filter the
// leaf will not compile must reach the caller as the PERMANENT sentinel, so a
// retry loop written against ErrKVIndexBuilding does not spin on it.
func TestClientKVQueryEndToEndBadFilterIsPermanent(t *testing.T) {
	addr, _, stop := startKVQueryStack(t)
	defer stop()

	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// The reserved "$" namespace is the record alias the leaf synthesizes; a
	// filter addressing it is refused outright.
	_, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{
		Index:  "by_age",
		Scan:   true,
		Filter: vtypes.Filter{Op: vtypes.FilterEq, Field: ops.KVRecordAlias, Value: vtypes.NewInt(1)},
	})
	if !errors.Is(qerr, client.ErrKVQueryFilter) {
		t.Fatalf("KVQuery with a reserved field = %v, want client.ErrKVQueryFilter", qerr)
	}
}
