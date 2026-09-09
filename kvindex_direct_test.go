// SPDX-License-Identifier: Apache-2.0

package rostam

// The embedded single-process store gets the same one-index-per-cache wiring as
// a replicated shard, plus one thing a shard does not have: raw Put and Del
// methods that bypass the op registry entirely. Those have to maintain the
// index themselves, or a value written through them would leave the posting
// describing the PREVIOUS value and a query for the new one would miss the key.

import (
	"context"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

const directIndexName = "by-rc"

func directRec(v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}, //nolint:gosec // fixture reinterprets a small i64
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

func directPosted(t *testing.T, idx *kvindex.Set, v int64) []string {
	t.Helper()
	d, ok := idx.Lookup(directIndexName)
	if !ok {
		t.Fatalf("index %q is not installed", directIndexName)
	}
	keys, err := idx.Candidates(kvindex.Selector{
		Def:    d,
		Op:     vtypes.FilterEq,
		Values: []vtypes.Value{vtypes.NewInt(v)},
	}, nil, 1<<20)
	if err != nil {
		t.Fatalf("Candidates(rc == %d): %v", v, err)
	}
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, string(k))
	}
	return out
}

func newIndexedDirect(t *testing.T) (*directStore, *kvindex.Set) {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	s, err := NewDirect(DirectConfig{Ops: reg, DataDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	d, ok := s.(*directStore)
	if !ok {
		t.Fatalf("NewDirect returned %T, want *directStore", s)
	}
	idx := d.tx.KVIndex()
	if idx == nil {
		t.Fatal("a directStore must be built with a KV index")
	}
	def, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        directIndexName,
		KeyPrefix:   []byte("u:"),
		PayloadPath: "rc",
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom: %v", err)
	}
	idx.Install([]kvindex.Def{def})
	idx.MarkReady(directIndexName)
	return d, idx
}

// TestDirectKVIndexFollowsTheOpPath — a write dispatched through the registry
// maintains the index, and a delete unposts through the cache's onRemove hook.
func TestDirectKVIndexFollowsTheOpPath(t *testing.T) {
	d, idx := newIndexedDirect(t)
	ctx := context.Background()

	if _, err := d.Call(ctx, "put", ops.EncodePutArgs([]byte("u:a"), directRec(5), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	if got := directPosted(t, idx, 5); len(got) != 1 || got[0] != "u:a" {
		t.Fatalf("after put, rc == 5 finds %v, want [u:a]", got)
	}

	if _, err := d.Call(ctx, "del", ops.EncodeKeyArgs([]byte("u:a"))); err != nil {
		t.Fatalf("del: %v", err)
	}
	if got := directPosted(t, idx, 5); len(got) != 0 {
		t.Fatalf("after del, rc == 5 still finds %v", got)
	}
}

// TestDirectRawPutAndDelMaintainKVIndex covers the two methods that never reach
// an op handler. The raw Put is the one that could silently lose rows: it
// changes the stored bytes, so a stale posting under the OLD value would make
// the key unfindable under the new one.
func TestDirectRawPutAndDelMaintainKVIndex(t *testing.T) {
	d, idx := newIndexedDirect(t)
	ctx := context.Background()
	key := []byte("u:raw")

	if err := d.Put(ctx, key, directRec(1), 0); err != nil {
		t.Fatalf("raw Put: %v", err)
	}
	if got := directPosted(t, idx, 1); len(got) != 1 || got[0] != "u:raw" {
		t.Fatalf("after a raw Put, rc == 1 finds %v, want [u:raw]", got)
	}

	if err := d.Put(ctx, key, directRec(2), 0); err != nil {
		t.Fatalf("raw overwriting Put: %v", err)
	}
	if got := directPosted(t, idx, 1); len(got) != 0 {
		t.Fatalf("a raw overwrite left the OLD posting: rc == 1 still finds %v", got)
	}
	if got := directPosted(t, idx, 2); len(got) != 1 || got[0] != "u:raw" {
		t.Fatalf("after a raw overwrite, rc == 2 finds %v, want [u:raw]", got)
	}

	deleted, err := d.Del(ctx, key)
	if err != nil || !deleted {
		t.Fatalf("raw Del = (%v, %v), want (true, nil)", deleted, err)
	}
	if got := directPosted(t, idx, 2); len(got) != 0 {
		t.Fatalf("after a raw Del, rc == 2 still finds %v", got)
	}
}

// TestDirectPutBatchReindexesEveryKey — PutBatch chunks through the put_batch
// op, so every key in it must be posted, not just the routing key.
func TestDirectPutBatchReindexesEveryKey(t *testing.T) {
	d, idx := newIndexedDirect(t)

	entries := make([]ops.PutEntry, 0, 8)
	want := make([]string, 0, 8)
	for i := 0; i < 8; i++ {
		k := fmt.Appendf(nil, "u:%d", i)
		entries = append(entries, ops.PutEntry{Key: k, Val: directRec(3)})
		want = append(want, string(k))
	}
	if err := d.PutBatch(context.Background(), entries); err != nil {
		t.Fatalf("PutBatch: %v", err)
	}
	got := directPosted(t, idx, 3)
	if len(got) != len(want) {
		t.Fatalf("after PutBatch, rc == 3 finds %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("after PutBatch, rc == 3 finds %v, want %v", got, want)
		}
	}
}
