//go:build cgo && linux

// SPDX-License-Identifier: Apache-2.0

package wasm

// A guest-driven write is a KV write like any other, so it owes the record
// index the same maintenance a builtin handler does. The cache_put host
// function is the only guest path that stores bytes under a key; cache_del
// removes a live slot (the cache's onRemove hook unposts it) and cache_expire
// routes through ops.TxContext.Expire, which re-posts.
//
// Without this, a stored procedure that wrote a record left the posting
// describing whatever the key held before, and a kv_query for the new value
// missed a live key — the missing-row class.

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestWASMHostPutMaintainsKVIndex invokes testdata/put.wasm, which writes its
// args (minus the 2-byte routing prefix) as BOTH key and value. Handing it a
// record blob therefore stores that record under itself, which is all this test
// needs: the value the guest stored has to be findable through the index the
// host maintains.
func TestWASMHostPutMaintainsKVIndex(t *testing.T) {
	rt, id := loadHostErrModule(t, "testdata/put.wasm", "put")

	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = c.Close() }()

	idx := ops.NewKVIndexFor(c)
	// An empty prefix covers every key, which is what lets the fixture use the
	// record blob itself as the key (put.wasm writes one buffer as both).
	def, err := kvindex.DefFrom(wire.KVIndexDef{
		Name: "by-rc", KeyPrefix: nil, PayloadPath: "rc",
		Kind: wire.KVIndexKindScalar, Enabled: true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom: %v", err)
	}
	idx.Install([]kvindex.Def{def})
	idx.MarkReady("by-rc")

	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: 5}},
	}}
	blob := rec.Encode()
	if blob == nil {
		t.Fatal("fixture record failed to encode")
	}

	tx := ops.NewTxContextWithIndex(c, nil, idx)
	if _, err := rt.Invoke(id, tx, stdArgs(blob)); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if _, err := c.Get(blob); err != nil {
		t.Fatalf("the guest's write did not land: %v", err)
	}

	keys, err := idx.Candidates(kvindex.Selector{
		Def: def, Op: vtypes.FilterEq, Values: []vtypes.Value{vtypes.NewInt(5)},
	}, nil, 1<<20)
	if err != nil {
		t.Fatalf("Candidates: %v", err)
	}
	if len(keys) != 1 || string(keys[0]) != string(blob) {
		t.Fatalf("after a guest cache_put, rc == 5 finds %d candidates, want the key the guest wrote — "+
			"the host function must store through ops.TxContext.PutIndexed, not a bare Put", len(keys))
	}
}

// TestWASMHostPutWithoutIndexStillWorks — the host path must be unchanged for
// every embedder that has no record index at all.
func TestWASMHostPutWithoutIndexStillWorks(t *testing.T) {
	rt, id := loadHostErrModule(t, "testdata/put.wasm", "put")

	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = c.Close() }()

	tx := ops.NewTxContext(c)
	if tx.KVIndex() != nil {
		t.Fatal("NewTxContext must build a dispatcher with no index")
	}
	key := []byte("plain-guest-key")
	if _, err := rt.Invoke(id, tx, stdArgs(key)); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if got, err := c.Get(key); err != nil || string(got) != string(key) {
		t.Fatalf("guest write with no index: Get = (%q, %v)", got, err)
	}
}
