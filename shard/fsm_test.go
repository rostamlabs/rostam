// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

func newTestFSM(t *testing.T) (*fsm, *cache.Cache) {
	t.Helper()
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	c, err := cache.New(cc)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	return newFSM(c, reg, false, nil, nil), c
}

func TestFSMApplyPut(t *testing.T) {
	f, c := newTestFSM(t)
	entry := EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))
	res := f.Apply(&hraft.Log{Index: 1, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok {
		t.Fatalf("Apply returned %T, want *ApplyResponse", res)
	}
	if resp.Err != nil {
		t.Fatalf("Apply err: %v", resp.Err)
	}
	got, err := c.Get([]byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Get = %q, want v", got)
	}
}

func TestFSMApplyDel(t *testing.T) {
	f, c := newTestFSM(t)
	_ = c.Put([]byte("k"), []byte("v"), 0)
	entry := EncodeLogEntry("del", ops.EncodeKeyArgs([]byte("k")))
	res := f.Apply(&hraft.Log{Index: 1, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok {
		t.Fatalf("Apply returned %T, want *ApplyResponse", res)
	}
	if resp.Err != nil {
		t.Fatalf("Apply err: %v", resp.Err)
	}
	if len(resp.Result) != 1 || resp.Result[0] != 1 {
		t.Fatalf("del result = %v, want [1]", resp.Result)
	}
	if _, err := c.Get([]byte("k")); err != cache.ErrNotFound {
		t.Fatal("post-del Get must be ErrNotFound")
	}
}

func TestFSMApplyUnknownOp(t *testing.T) {
	f, _ := newTestFSM(t)
	entry := EncodeLogEntry("nonexistent", nil)
	res := f.Apply(&hraft.Log{Index: 1, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok || resp.Err != ErrOpNotRegistered {
		t.Fatalf("Apply unknown op: %v, want ErrOpNotRegistered", resp)
	}
}

func TestFSMApplyMalformedEntry(t *testing.T) {
	f, _ := newTestFSM(t)
	res := f.Apply(&hraft.Log{Index: 1, Data: []byte{0xff, 0xff}})
	resp, ok := res.(*ApplyResponse)
	if !ok || resp.Err == nil {
		t.Fatalf("malformed entry: expected error, got %v", resp)
	}
}

func TestFSMApplyReadOnlyOpRejected(t *testing.T) {
	f, _ := newTestFSM(t)
	entry := EncodeLogEntry("get", ops.EncodeKeyArgs([]byte("k")))
	res := f.Apply(&hraft.Log{Index: 1, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok || resp.Err == nil {
		t.Fatalf("read-only op via Apply: expected error, got %v", resp)
	}
}

func TestApplyEntryData(t *testing.T) {
	f, c := newTestFSM(t)
	entry := EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))
	resp := f.applyEntryData(entry)
	if resp.Err != nil {
		t.Fatalf("applyEntryData put: %v", resp.Err)
	}
	if got, err := c.Get([]byte("k")); err != nil || !bytes.Equal(got, []byte("v")) {
		t.Fatalf("get after applyEntryData: got=%q err=%v", got, err)
	}
}

func TestFSMApplyBatch(t *testing.T) {
	f, c := newTestFSM(t)
	// A batch of puts, a LogConfiguration entry mixed in (Raft includes those in
	// the batch though plain Apply never sees them), and a del.
	logs := []*hraft.Log{
		{Index: 1, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("a"), []byte("1"), 0))},
		{Index: 2, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("b"), []byte("2"), 0))},
		{Index: 3, Type: hraft.LogConfiguration, Data: []byte("cfg-not-an-op")},
		{Index: 4, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("c"), []byte("3"), 0))},
	}
	resps := f.ApplyBatch(logs)
	if len(resps) != len(logs) {
		t.Fatalf("ApplyBatch returned %d responses, want %d (one per log)", len(resps), len(logs))
	}
	for i, r := range resps {
		resp, ok := r.(*ApplyResponse)
		if !ok {
			t.Fatalf("resp[%d] type %T, want *ApplyResponse", i, r)
		}
		if resp.Err != nil {
			t.Fatalf("resp[%d] err: %v (the config entry must NOT be run as an op)", i, resp.Err)
		}
	}
	for _, kv := range []struct{ k, v string }{{"a", "1"}, {"b", "2"}, {"c", "3"}} {
		got, err := c.Get([]byte(kv.k))
		if err != nil || !bytes.Equal(got, []byte(kv.v)) {
			t.Fatalf("Get %q = %q, %v; want %q", kv.k, got, err, kv.v)
		}
	}
	// Applied index advanced ONCE to the batch's max index.
	if idx := f.AppliedIndex(); idx != 4 {
		t.Fatalf("AppliedIndex = %d, want 4 (batch max)", idx)
	}
}

// TestFSMApplyBatchMatchesSequential asserts batch-apply is state-equivalent to
// applying the same entries one-by-one via Apply.
func TestFSMApplyBatchMatchesSequential(t *testing.T) {
	build := func() (*fsm, *cache.Cache) { return newTestFSM(t) }
	entries := func(base uint64) []*hraft.Log {
		return []*hraft.Log{
			{Index: base + 1, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("x"), []byte("10"), 0))},
			{Index: base + 2, Type: hraft.LogCommand, Data: EncodeLogEntry("incr", ops.EncodeIncrArgs([]byte("n"), 5))},
			{Index: base + 3, Type: hraft.LogCommand, Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("x"), []byte("20"), 0))},
		}
	}
	fb, cb := build()
	fb.ApplyBatch(entries(0))
	fs, cs := build()
	for _, l := range entries(0) {
		fs.Apply(l)
	}
	for _, k := range []string{"x", "n"} {
		gb, _ := cb.Get([]byte(k))
		gs, _ := cs.Get([]byte(k))
		if !bytes.Equal(gb, gs) {
			t.Fatalf("key %q: batch=%q sequential=%q — must match", k, gb, gs)
		}
	}
	if fb.AppliedIndex() != fs.AppliedIndex() {
		t.Fatalf("appliedIndex batch=%d sequential=%d", fb.AppliedIndex(), fs.AppliedIndex())
	}
}

// TestFSMApplyBatchNoWatermarkRegression guards the durable applied-index
// watermark against a warm-restart replay batch whose maxIdx is below the
// persisted watermark. Unlike the monotonic advanceApplied, cache.SetAppliedIndex
// stores unconditionally, so an unguarded batch write would regress the header
// (114 < 500) and defeat warm-restart skip. Fails without the maxIdx>cacheApplied
// guard in ApplyBatch.
func TestFSMApplyBatchNoWatermarkRegression(t *testing.T) {
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.DataDir = t.TempDir() // durable mmap region: SetAppliedIndex/AppliedIndex are live
	c, err := cache.New(cc)
	if err != nil {
		t.Skipf("durable cache unavailable on this platform: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	f := newFSM(c, reg, true /* durable */, nil, nil)

	// Seed the persisted watermark high, as after normal operation.
	c.SetAppliedIndex(500, true)
	if c.AppliedIndex() != 500 {
		t.Fatalf("seed: AppliedIndex=%d, want 500", c.AppliedIndex())
	}

	// Warm-restart replay: raft re-delivers already-applied entries in a batch
	// whose maxIdx (114) is far below the persisted watermark (500). Every entry
	// is warm-skipped; the durable watermark must NOT regress.
	var logs []*hraft.Log
	for i := uint64(51); i <= 114; i++ {
		logs = append(logs, &hraft.Log{Index: i, Type: hraft.LogCommand,
			Data: EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))})
	}
	f.ApplyBatch(logs)
	if got := c.AppliedIndex(); got != 500 {
		t.Fatalf("watermark regressed to %d after replay batch (maxIdx=114); want 500", got)
	}
}

// TestFSMApplyPutBatchSingleEntry proves the put_batch win: N puts applied by
// ONE Raft log entry (AppliedIndex advances by exactly 1, not N).
func TestFSMApplyPutBatchSingleEntry(t *testing.T) {
	f, c := newTestFSM(t)
	entries := []ops.PutEntry{
		{Key: []byte("a"), Val: []byte("1")},
		{Key: []byte("b"), Val: []byte("2")},
		{Key: []byte("c"), Val: []byte("3")},
	}
	entry := EncodeLogEntry("put_batch", ops.EncodePutBatchArgs(entries))
	res := f.Apply(&hraft.Log{Index: 1, Type: hraft.LogCommand, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok || resp.Err != nil {
		t.Fatalf("apply put_batch: %v", resp)
	}
	for _, e := range entries {
		got, err := c.Get(e.Key)
		if err != nil || !bytes.Equal(got, e.Val) {
			t.Fatalf("get %q = %q,%v; want %q", e.Key, got, err, e.Val)
		}
	}
	if idx := f.AppliedIndex(); idx != 1 {
		t.Fatalf("AppliedIndex = %d, want 1 (one log entry for %d puts)", idx, len(entries))
	}
}

func TestFSMSnapshotRestoreRoundtrip(t *testing.T) {
	f, c := newTestFSM(t)
	for i := byte(0); i < 50; i++ {
		_ = c.Put([]byte{i}, []byte{i, i + 1}, 0)
	}
	snap, err := f.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	sink := newMemSink()
	if err := snap.Persist(sink); err != nil {
		t.Fatalf("Persist: %v", err)
	}
	snap.Release()

	f2, c2 := newTestFSM(t)
	if err := f2.Restore(sink.reader()); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	for i := byte(0); i < 50; i++ {
		got, err := c2.Get([]byte{i})
		if err != nil {
			t.Fatalf("Get %d: %v", i, err)
		}
		if !bytes.Equal(got, []byte{i, i + 1}) {
			t.Fatalf("Get %d: %v, want %v", i, got, []byte{i, i + 1})
		}
	}
	count := 0
	c2.Iterate(func(_, _ []byte, _ uint64) bool {
		count++
		return true
	})
	if count != 50 {
		t.Fatalf("restored entry count = %d, want 50", count)
	}
	_ = time.Now()
}

// memSink is an in-memory raft.SnapshotSink for testing.
type memSink struct {
	buf bytes.Buffer
	id  string
}

func newMemSink() *memSink                     { return &memSink{id: "test"} }
func (m *memSink) Write(p []byte) (int, error) { return m.buf.Write(p) }
func (m *memSink) Close() error                { return nil }
func (m *memSink) ID() string                  { return m.id }
func (m *memSink) Cancel() error               { return nil }
func (m *memSink) reader() *memSinkReader      { return &memSinkReader{r: bytes.NewReader(m.buf.Bytes())} }

type memSinkReader struct{ r *bytes.Reader }

func (r *memSinkReader) Read(p []byte) (int, error) { return r.r.Read(p) }
func (r *memSinkReader) Close() error               { return nil }
