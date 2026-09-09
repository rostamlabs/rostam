// SPDX-License-Identifier: Apache-2.0

package ops

// Wiring tests for the KV record index: which handlers maintain it, which
// deliberately do not, in what ORDER relative to the store, and what it costs
// a store that has no index at all.
//
// The index's own behaviour (what resolves to a posting, how candidates are
// selected, readiness, budgets) belongs to ops/kvindex and is covered there.
// What is proved here is only the seam: every write handler that changes the
// STORED BYTES calls reindexKV with those bytes after the store returns, every
// removal is covered by the cache's onRemove hook, and nothing else touches
// the index.

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- fixtures -------------------------------------------------------------

const wiringIndexName = "by-rc"

// wiringRec encodes a one-field dynamic-mode record {rc: v}. Dynamic mode keeps
// the fixtures readable: the field name travels in the record, so no schema has
// to be carried alongside it.
func wiringRec(v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(v)}},
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture record failed to encode")
	}
	return b
}

// padRec is wiringRec with a fixed 32 KiB padding field, so that every entry in
// the eviction fixture costs the shard the same number of bytes whatever its rc
// is. Uniform entry sizes are what make "the next write must evict" a property
// of the fill level alone.
func padRec(v int64) []byte {
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(v)}},
		{Name: "pad", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: make([]byte, 32<<10)}},
	}}
	b := rec.Encode()
	if b == nil {
		panic("fixture padded record failed to encode")
	}
	return b
}

// wiringDef is the definition every test in this file indexes by: the "rc"
// field of every record under the "u:" key prefix.
func wiringDef(t *testing.T, path string) kvindex.Def {
	t.Helper()
	d, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        wiringIndexName,
		KeyPrefix:   []byte("u:"),
		PayloadPath: path,
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		t.Fatalf("DefFrom(%q): %v", path, err)
	}
	return d
}

// newIndexedTx builds a one-shard cache with THE index wired to it exactly as a
// store does (ops.NewKVIndexFor: one Set, its Drop registered as the cache's
// onRemove hook) and a TxContext over both. The definition is installed and
// marked ready, which is the state a completed backfill leaves behind.
func newIndexedTx(t *testing.T, path string) (*TxContext, *cache.Cache, *kvindex.Set) {
	t.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	idx := NewKVIndexFor(c)
	idx.Install([]kvindex.Def{wiringDef(t, path)})
	idx.MarkReady(wiringIndexName)
	return NewTxContextWithIndex(c, nil, idx), c, idx
}

// posted returns the keys posted under rc == v, sorted, as strings.
func posted(t *testing.T, idx *kvindex.Set, v int64) []string {
	t.Helper()
	d, ok := idx.Lookup(wiringIndexName)
	if !ok {
		t.Fatalf("index %q is not installed", wiringIndexName)
	}
	keys, err := kvindexCandidates(idx, d, v)
	if err != nil {
		t.Fatalf("Candidates(rc == %d): %v", v, err)
	}
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, string(k))
	}
	return out
}

func kvindexCandidates(idx *kvindex.Set, d kvindex.Def, v int64) ([][]byte, error) {
	return idx.Candidates(kvindex.Selector{
		Def:    d,
		Op:     vtypes.FilterEq,
		Values: []vtypes.Value{vtypes.NewInt(v)},
	}, nil, 1<<20)
}

func wantPosted(t *testing.T, idx *kvindex.Set, v int64, want ...string) {
	t.Helper()
	got := posted(t, idx, v)
	if len(got) != len(want) {
		t.Fatalf("keys posted under rc == %d: got %v, want %v", v, got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("keys posted under rc == %d: got %v, want %v", v, got, want)
		}
	}
}

// --- the handlers that reindex -------------------------------------------

func TestPutReindexes(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")

	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), wiringRec(5), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	wantPosted(t, idx, 5, "u:a")

	// An overwrite MOVES the posting: the old value must not still find it.
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), wiringRec(9), 0)); err != nil {
		t.Fatalf("put overwrite: %v", err)
	}
	wantPosted(t, idx, 5)
	wantPosted(t, idx, 9, "u:a")

	// A key outside the definition's prefix is not this index's business.
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("other:a"), wiringRec(9), 0)); err != nil {
		t.Fatalf("put out of prefix: %v", err)
	}
	wantPosted(t, idx, 9, "u:a")
}

// TestPutBatchReindexesEveryKey pins that the batch handler posts per ENTRY,
// not once for the batch: three keys under the prefix and one outside it must
// leave exactly three postings.
func TestPutBatchReindexesEveryKey(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")

	entries := []PutEntry{
		{Key: []byte("u:a"), Val: wiringRec(7)},
		{Key: []byte("u:b"), Val: wiringRec(7)},
		{Key: []byte("u:c"), Val: wiringRec(7)},
		{Key: []byte("zz:d"), Val: wiringRec(7)},
	}
	out, err := handlePutBatch(tx, EncodePutBatchArgs(entries))
	if err != nil {
		t.Fatalf("put_batch: %v", err)
	}
	if n, err := wire.DecodePutBatchResult(out); err != nil || n != len(entries) {
		t.Fatalf("put_batch result = (%d, %v), want (%d, nil)", n, err, len(entries))
	}
	wantPosted(t, idx, 7, "u:a", "u:b", "u:c")

	keys, distinct := idx.Stats(wiringIndexName)
	if keys != 3 || distinct != 1 {
		t.Fatalf("Stats = (%d keys, %d distinct), want (3, 1) — the out-of-prefix key must not be posted", keys, distinct)
	}
}

// TestOperateReindexes drives the record through operate itself: an in-place
// increment of rc from 5 to 6 must leave the key findable under 6 and not
// under 5.
func TestOperateReindexes(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	key := []byte("u:sess")

	callOperate(t, tx, 0, &wire.OperateArgs{Key: key, Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeI64, Path: namePath("rc"), A: 5}}})
	wantPosted(t, idx, 5, "u:sess")

	callOperate(t, tx, 0, &wire.OperateArgs{Key: key, Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: wire.OperateTypeI64, Path: namePath("rc"), A: 1}}})
	wantPosted(t, idx, 5)
	wantPosted(t, idx, 6, "u:sess")
}

// TestOperateFailedCheckDoesNotReindex — a failed CHECK stores nothing, so the
// posting must still describe the record as it stands.
func TestOperateFailedCheckDoesNotReindex(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	key := []byte("u:sess")

	callOperate(t, tx, 0, &wire.OperateArgs{Key: key, Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeI64, Path: namePath("rc"), A: 5}}})

	res := callOperate(t, tx, 0, &wire.OperateArgs{Key: key, Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpCHECK, Type: wire.OperateTypeI64, Aux: wire.OperateCmpEQ, Path: namePath("rc"), A: 999},
			{Opcode: wire.OperateOpSET, Type: wire.OperateTypeI64, Path: namePath("rc"), A: 42},
		}})
	if res.Status != wire.OperateStatusCheckFailed {
		t.Fatalf("operate status = %d, want check-failed", res.Status)
	}
	wantPosted(t, idx, 5, "u:sess")
	wantPosted(t, idx, 42)
}

func TestGetSetReindexes(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")

	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), wiringRec(1), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	out, err := handleGetSet(tx, wire.EncodePutArgs([]byte("u:a"), wiringRec(2), 0))
	if err != nil {
		t.Fatalf("getset: %v", err)
	}
	old, found, err := wire.DecodeGetDelResult(out)
	if err != nil || !found || !bytes.Equal(old, wiringRec(1)) {
		t.Fatalf("getset returned (%q, %v, %v), want the old record", old, found, err)
	}
	wantPosted(t, idx, 1)
	wantPosted(t, idx, 2, "u:a")
}

// TestIncrAndIncrExReindex — both counters store new bytes, so both must call
// the seam. A counter is 8 raw bytes and never resolves to a record, so what
// the seam does with it is a DROP — and the drop is exactly what makes the call
// observable. The test plants a stale posting for the key first (straight into
// the index, so the cache never hears about it) and then asserts the counter
// write removed it; without the reindexKV call it would survive.
func TestIncrAndIncrExReindex(t *testing.T) {
	tx, c, idx := newIndexedTx(t, "rc")

	counter := make([]byte, 8)
	for _, tc := range []struct {
		name string
		key  []byte
		call func(key []byte) error
	}{
		{"incr", []byte("u:n"), func(key []byte) error {
			_, err := handleIncr(tx, wire.EncodeIncrArgs(key, 1))
			return err
		}},
		{"incr_ex", []byte("u:m"), func(key []byte) error {
			_, err := handleIncrEx(tx, wire.EncodeIncrExArgs(key, 1, time.Minute))
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := c.Put(tc.key, counter, 0); err != nil {
				t.Fatalf("seed counter: %v", err)
			}
			idx.Reindex(tc.key, wiringRec(3)) // the stale posting
			wantPosted(t, idx, 3, string(tc.key))

			if err := tc.call(tc.key); err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if got := posted(t, idx, 3); len(got) != 0 {
				t.Fatalf("%s left the stale posting %v in place; it must reindex the bytes it stored",
					tc.name, got)
			}
			if _, err := handleDel(tx, wire.EncodeKeyArgs(tc.key)); err != nil {
				t.Fatalf("cleanup del: %v", err)
			}
		})
	}
}

// TestSetNXFailureDoesNotReindex and its CAS twin: the refused branch stores
// nothing, so the posting must still describe what is actually stored.
func TestSetNXFailureDoesNotReindex(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	key := []byte("u:a")

	out, err := handleSetNX(tx, wire.EncodePutArgs(key, wiringRec(1), 0))
	if err != nil || len(out) != 1 || out[0] != 1 {
		t.Fatalf("set_nx on an absent key = (%v, %v), want stored", out, err)
	}
	wantPosted(t, idx, 1, "u:a")

	out, err = handleSetNX(tx, wire.EncodePutArgs(key, wiringRec(2), 0))
	if err != nil || len(out) != 1 || out[0] != 0 {
		t.Fatalf("set_nx on a present key = (%v, %v), want refused", out, err)
	}
	wantPosted(t, idx, 1, "u:a")
	wantPosted(t, idx, 2)
}

func TestCASReindexesOnlyOnSuccess(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	key := []byte("u:a")

	if _, err := handlePut(tx, wire.EncodePutArgs(key, wiringRec(1), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}

	// Mismatch: nothing stored, nothing reposted.
	out, err := handleCAS(tx, wire.EncodeCASArgs(key, wiringRec(2), true, wiringRec(99), 0))
	if err != nil || len(out) != 1 || out[0] != 0 {
		t.Fatalf("cas mismatch = (%v, %v), want refused", out, err)
	}
	wantPosted(t, idx, 1, "u:a")
	wantPosted(t, idx, 2)

	// Match: stored, so reposted.
	out, err = handleCAS(tx, wire.EncodeCASArgs(key, wiringRec(2), true, wiringRec(1), 0))
	if err != nil || len(out) != 1 || out[0] != 1 {
		t.Fatalf("cas match = (%v, %v), want stored", out, err)
	}
	wantPosted(t, idx, 1)
	wantPosted(t, idx, 2, "u:a")
}

// --- the removals, all covered by the cache's onRemove hook ---------------

func TestDelDropsPostingViaHook(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), wiringRec(5), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	if _, err := handleDel(tx, wire.EncodeKeyArgs([]byte("u:a"))); err != nil {
		t.Fatalf("del: %v", err)
	}
	wantPosted(t, idx, 5)
	if keys, distinct := idx.Stats(wiringIndexName); keys != 0 || distinct != 0 {
		t.Fatalf("Stats after del = (%d, %d), want (0, 0)", keys, distinct)
	}
}

func TestGetDelDropsPosting(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), wiringRec(5), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	if _, err := handleGetDel(tx, wire.EncodeKeyArgs([]byte("u:a"))); err != nil {
		t.Fatalf("getdel: %v", err)
	}
	wantPosted(t, idx, 5)
}

func TestCompareAndDelDropsPosting(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	rec := wiringRec(5)
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), rec, 0)); err != nil {
		t.Fatalf("put: %v", err)
	}

	// A mismatch deletes nothing, so the posting stays.
	out, err := handleCompareAndDel(tx, wire.EncodeCADArgs([]byte("u:a"), wiringRec(6)))
	if err != nil || len(out) != 1 || out[0] != 0 {
		t.Fatalf("cad mismatch = (%v, %v), want refused", out, err)
	}
	wantPosted(t, idx, 5, "u:a")

	out, err = handleCompareAndDel(tx, wire.EncodeCADArgs([]byte("u:a"), rec))
	if err != nil || len(out) != 1 || out[0] != 1 {
		t.Fatalf("cad match = (%v, %v), want deleted", out, err)
	}
	wantPosted(t, idx, 5)
}

// TestOperateDeleteDropsPostingViaHook — operate's delete branch reindexes
// nothing on purpose; tx.Del removes the live slot and the hook does the work.
func TestOperateDeleteDropsPostingViaHook(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	key := []byte("u:sess")

	callOperate(t, tx, 0, &wire.OperateArgs{Key: key, Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeI64, Path: namePath("rc"), A: 5}}})
	wantPosted(t, idx, 5, "u:sess")

	// Deleting the only field deletes the record, and with it the key.
	callOperate(t, tx, 0, &wire.OperateArgs{Key: key, Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpDEL, Path: namePath("rc")}}})
	wantPosted(t, idx, 5)
	if keys, _ := idx.Stats(wiringIndexName); keys != 0 {
		t.Fatalf("operate's delete branch left %d posted keys", keys)
	}
}

// --- the handlers that deliberately reindex NOTHING -----------------------

// TestExpireAndPersistDoNotDisturbPostings: expire, caex and persist all
// rewrite the SAME bytes with a different deadline, so the posting is already
// correct and must be left exactly as it is — the key stays findable
// throughout.
func TestExpireAndPersistDoNotDisturbPostings(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	key := []byte("u:a")
	rec := wiringRec(5)
	if _, err := handlePut(tx, wire.EncodePutArgs(key, rec, 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	before, distinctBefore := idx.Stats(wiringIndexName)

	if _, err := handleExpire(tx, wire.EncodeExpireArgs(key, time.Hour)); err != nil {
		t.Fatalf("expire: %v", err)
	}
	wantPosted(t, idx, 5, "u:a")

	out, err := handleCAEX(tx, wire.EncodeCAEXArgs(key, rec, 2*time.Hour))
	if err != nil || len(out) != 1 || out[0] != 1 {
		t.Fatalf("caex = (%v, %v), want refreshed", out, err)
	}
	wantPosted(t, idx, 5, "u:a")

	out, err = handlePersist(tx, wire.EncodeKeyArgs(key))
	if err != nil || len(out) != 1 || out[0] != 1 {
		t.Fatalf("persist = (%v, %v), want cleared", out, err)
	}
	wantPosted(t, idx, 5, "u:a")

	if after, distinctAfter := idx.Stats(wiringIndexName); after != before || distinctAfter != distinctBefore {
		t.Fatalf("TTL-only ops changed the postings: (%d, %d) → (%d, %d)",
			before, distinctBefore, after, distinctAfter)
	}

	// ###################### AND THEY MUST NOT LOSE THEM #####################
	//
	// "Leaves the posting alone" is only half the requirement, and the easy
	// half: everything above runs on a DefaultConfig cache, which never evicts,
	// so it would pass even if these handlers did nothing at all about the
	// index. The half that bites is that expire, caex and persist all reach the
	// cache's PUT body — expire and caex through TxContext.Expire (get, copy,
	// put the same bytes with a new deadline), persist through PutAbs — and a
	// put on a ringbuf shard can evict the page framing the key's own current
	// copy, firing onRemove for the key mid-write. The bytes being UNCHANGED is
	// exactly why that is dangerous: nothing about the value looks like a
	// change, so nothing re-posts, and the key is left LIVE WITH NO POSTING.
	//
	// So the rule is not "reindex when the bytes change", it is: every path
	// that writes through the cache's put body reindexes after it returns.
	t.Run("under self-eviction", func(t *testing.T) {
		seedWithTTL := func(t *testing.T, tx *TxContext) {
			if _, err := handlePut(tx, wire.EncodePutArgs(evictionKey, padRec(1), time.Hour)); err != nil {
				t.Fatalf("seed put: %v", err)
			}
		}
		// The stored bytes never change, so the key must still be findable
		// under the value it was posted with before the write.
		stillPosted := func(op string) func(*testing.T, *cache.Cache, *kvindex.Set) {
			return func(t *testing.T, c *cache.Cache, idx *kvindex.Set) {
				live := true
				if v, err := c.Get(evictionKey); err != nil || !bytes.Equal(v, padRec(1)) {
					live = false
				}
				got := posted(t, idx, 1)
				if len(got) != 1 || got[0] != string(evictionKey) {
					t.Fatalf("after an evicting %s, rc == 1 finds %v, want [%s] — key live: %v. "+
						"%s rewrites the key through the cache's put body, so it must reindex "+
						"after that write returns even though the bytes are unchanged",
						op, got, evictionKey, live, op)
				}
			}
		}

		t.Run("expire", func(t *testing.T) {
			runUnderSelfEviction(t, seedWithTTL, func(t *testing.T, tx *TxContext) {
				if _, err := handleExpire(tx, wire.EncodeExpireArgs(evictionKey, 2*time.Hour)); err != nil {
					t.Fatalf("expire: %v", err)
				}
			}, stillPosted("expire"))
		})

		t.Run("caex", func(t *testing.T) {
			runUnderSelfEviction(t, seedWithTTL, func(t *testing.T, tx *TxContext) {
				out, err := handleCAEX(tx, wire.EncodeCAEXArgs(evictionKey, padRec(1), 2*time.Hour))
				if err != nil || len(out) != 1 || out[0] != 1 {
					t.Fatalf("caex = (%v, %v), want refreshed", out, err)
				}
			}, stillPosted("caex"))
		})

		t.Run("persist", func(t *testing.T) {
			runUnderSelfEviction(t, seedWithTTL, func(t *testing.T, tx *TxContext) {
				out, err := handlePersist(tx, wire.EncodeKeyArgs(evictionKey))
				if err != nil || len(out) != 1 || out[0] != 1 {
					t.Fatalf("persist = (%v, %v), want cleared", out, err)
				}
			}, stillPosted("persist"))
		})
	})
}

// TestFlushResetsIndexButKeepsReady — Flush is O(shards) and fires no per-key
// onRemove, so Set.Reset is the flush handler's complete index action: every
// posting gone, every definition STILL ready (an empty posting set over an
// empty keyspace is exact), and a query answers zero rows rather than
// "building".
func TestFlushResetsIndexButKeepsReady(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	for i := 0; i < 5; i++ {
		k := fmt.Appendf(nil, "u:%d", i)
		if _, err := handlePut(tx, wire.EncodePutArgs(k, wiringRec(5), 0)); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	if keys, _ := idx.Stats(wiringIndexName); keys != 5 {
		t.Fatalf("Stats before flush = %d keys, want 5", keys)
	}

	if _, err := handleFlush(tx, nil); err != nil {
		t.Fatalf("flush: %v", err)
	}

	if keys, distinct := idx.Stats(wiringIndexName); keys != 0 || distinct != 0 {
		t.Fatalf("Stats after flush = (%d, %d), want (0, 0)", keys, distinct)
	}
	if !idx.IsReady(wiringIndexName) {
		t.Fatal("flush left the index NOT ready; an empty index over an empty keyspace is exact, " +
			"and nothing re-installs or re-backfills after a flush")
	}
	// A query over the flushed index is an empty answer, not an error.
	if got := posted(t, idx, 5); len(got) != 0 {
		t.Fatalf("post-flush query returned %v, want no rows", got)
	}
}

// TestFlushWithoutIndexIsUnchanged — the flush handler must work identically on
// a dispatcher that has no index at all.
func TestFlushWithoutIndexIsUnchanged(t *testing.T) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	tx := NewTxContext(c)
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), []byte("v"), 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	if _, err := handleFlush(tx, nil); err != nil {
		t.Fatalf("flush without an index: %v", err)
	}
	if _, err := c.Get([]byte("u:a")); err != cache.ErrNotFound {
		t.Fatal("flush did not empty the cache")
	}
}

// --- ordering, failure isolation and cost --------------------------------

// evictionKey is the key every eviction fixture in this file writes. It is the
// same length as the fillers' keys so that "is there room for one more" is one
// question rather than a size-dependent one.
var evictionKey = []byte("u:KKKK")

// runUnderSelfEviction is the fixture behind every ordering test here, and it
// exists because the defect it hunts is invisible on an ordinary cache: a
// DefaultConfig cache never evicts, so a test written against one passes
// whatever the handler does.
//
// On a two-page ringbuf shard a write can evict the page that still frames the
// key's CURRENT copy, and the cache fires onRemove for THAT KEY from inside the
// write. Every posting for the key is dropped mid-write. Any handler that does
// not (re)post AFTER the write returns therefore leaves a LIVE key with NO
// posting — the missing-row class, the one failure this index may never have.
//
// The harness sweeps fill levels until `write` is itself the evicting write,
// since that is the only arrangement in which the hook fires for the key being
// written. seed lays down the key's first copy and any TTL the case needs;
// check runs once, at that fill level, with the eviction confirmed to have
// happened. It fails loudly rather than passing vacuously if no fill level
// produces the case.
func runUnderSelfEviction(
	t *testing.T,
	seed func(t *testing.T, tx *TxContext),
	write func(t *testing.T, tx *TxContext),
	check func(t *testing.T, c *cache.Cache, idx *kvindex.Set),
) {
	t.Helper()
	filler := padRec(0)

	for fill := 0; fill < 400; fill++ {
		cfg := cache.DefaultConfig()
		cfg.NumShards = 1
		cfg.PageSize = 1 << 20
		cfg.MaxMemoryPerShard = 2 << 20
		cfg.AtCapPolicy = cache.PolicyRingbufEvict
		c, err := cache.New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		idx := NewKVIndexFor(c)
		// Wrap the hook so the fixture can see it fire for the key being
		// written. The index's Drop is still what runs — this only observes.
		var droppedKey int
		c.SetOnRemove(func(k []byte) {
			if bytes.Equal(k, evictionKey) {
				droppedKey++
			}
			idx.Drop(k)
		})
		idx.Install([]kvindex.Def{wiringDef(t, "rc")})
		idx.MarkReady(wiringIndexName)
		tx := NewTxContextWithIndex(c, nil, idx)

		// The key's only copy lands on the first page.
		seed(t, tx)
		for i := 0; i < fill; i++ {
			if err := c.Put(fmt.Appendf(nil, "f:%04d", i), filler, 0); err != nil {
				t.Fatalf("filler put: %v", err)
			}
		}
		if droppedKey != 0 {
			// A filler already evicted the key's page: this fill level is past
			// the interesting one, and every higher one will be too.
			_ = c.Close()
			break
		}

		// THE WRITE UNDER TEST. At exactly one fill level it is the write that
		// must evict, and the page it drains is the one holding the key.
		write(t, tx)
		if droppedKey == 0 {
			_ = c.Close()
			continue // not the evicting fill level yet
		}
		check(t, c, idx)
		_ = c.Close()
		return
	}
	t.Fatal("no fill level made the write under test the evicting one; the fixture needs adjusting")
}

// TestReindexHappensAfterPut is the regression test for the ORDER a handler
// must use. A handler that posted BEFORE its Put would have the fresh posting
// dropped by the eviction that follows, and the new value would be unfindable.
func TestReindexHappensAfterPut(t *testing.T) {
	runUnderSelfEviction(t,
		func(t *testing.T, tx *TxContext) {
			if _, err := handlePut(tx, wire.EncodePutArgs(evictionKey, padRec(1), 0)); err != nil {
				t.Fatalf("seed put: %v", err)
			}
		},
		func(t *testing.T, tx *TxContext) {
			if _, err := handlePut(tx, wire.EncodePutArgs(evictionKey, padRec(2), 0)); err != nil {
				t.Fatalf("overwriting put: %v", err)
			}
		},
		func(t *testing.T, c *cache.Cache, idx *kvindex.Set) {
			got := posted(t, idx, 2)
			if len(got) != 1 || got[0] != string(evictionKey) {
				t.Fatalf("after an evicting Put of the key itself, rc == 2 finds %v, want [%s] — "+
					"the handler must call reindexKV AFTER Put returns, never before", got, evictionKey)
			}
			if v, err := c.Get(evictionKey); err != nil || !bytes.Equal(v, padRec(2)) {
				t.Fatalf("the key itself did not survive its evicting Put: (%q, %v)", v, err)
			}
		})
}

// TestIndexFailureNeverFailsApply — a definition whose path never resolves is
// "no posting", never an error the apply must handle. Every write still
// succeeds and stores byte-identical bytes.
func TestIndexFailureNeverFailsApply(t *testing.T) {
	// "nope" is a field no fixture record has, so every resolve misses.
	tx, c, idx := newIndexedTx(t, "nope")

	rec := wiringRec(5)
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:a"), rec, 0)); err != nil {
		t.Fatalf("put: %v", err)
	}
	// Not even a record: the resolver fails outright, which is still not an error.
	if _, err := handlePut(tx, wire.EncodePutArgs([]byte("u:b"), []byte("not-a-record"), 0)); err != nil {
		t.Fatalf("put of a non-record: %v", err)
	}
	if _, err := handlePutBatch(tx, EncodePutBatchArgs([]PutEntry{
		{Key: []byte("u:c"), Val: rec},
	})); err != nil {
		t.Fatalf("put_batch: %v", err)
	}
	if _, err := handleGetSet(tx, wire.EncodePutArgs([]byte("u:a"), rec, 0)); err != nil {
		t.Fatalf("getset: %v", err)
	}
	callOperate(t, tx, 0, &wire.OperateArgs{Key: []byte("u:d"), Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeI64, Path: namePath("rc"), A: 5}}})

	for _, tc := range []struct{ key, want []byte }{
		{[]byte("u:a"), rec},
		{[]byte("u:b"), []byte("not-a-record")},
		{[]byte("u:c"), rec},
	} {
		got, err := c.Get(tc.key)
		if err != nil {
			t.Fatalf("Get %q: %v", tc.key, err)
		}
		if !bytes.Equal(got, tc.want) {
			t.Fatalf("Get %q = %q, want %q — an index failure changed what was stored", tc.key, got, tc.want)
		}
	}
	if keys, distinct := idx.Stats(wiringIndexName); keys != 0 || distinct != 0 {
		t.Fatalf("an unresolvable path posted something: (%d keys, %d distinct)", keys, distinct)
	}
}

// baselineNoIndexPutAllocs is what one handlePut allocated on a dispatcher with
// NO index, measured on this branch's parent (ae094d2) before any of this
// task's wiring existed, with testing.AllocsPerRun(2000) over a 1-shard cache
// and a 17-byte value. Wiring the index in must not move it.
const baselineNoIndexPutAllocs = 0

// TestNoIndexInstalledIsZeroCost — a store that never defines an index pays a
// nil check per write and nothing else.
func TestNoIndexInstalledIsZeroCost(t *testing.T) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	tx := NewTxContext(c)
	if tx.KVIndex() != nil {
		t.Fatal("NewTxContext must build a dispatcher with no index")
	}
	args := wire.EncodePutArgs([]byte("u:1"), []byte("hello-world-value"), 0)
	got := testing.AllocsPerRun(2000, func() {
		if _, err := handlePut(tx, args); err != nil {
			t.Fatalf("put: %v", err)
		}
	})
	if got != baselineNoIndexPutAllocs {
		t.Fatalf("handlePut with no index allocates %v/op, baseline is %d/op", got, baselineNoIndexPutAllocs)
	}
}

// TestRebuildKVIndexNoDefsWalksNothing — the warm-start / post-restore rebuild
// point is reached on every start, almost always with no definition installed.
// It must not pay for a full-keyspace walk then. Rebuild itself allocates (a
// per-walk token map and a closure), so zero allocations is evidence that its
// body never ran.
func TestRebuildKVIndexNoDefsWalksNothing(t *testing.T) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	idx := NewKVIndexFor(c)
	for i := 0; i < 200; i++ {
		if err := c.Put(fmt.Appendf(nil, "u:%d", i), wiringRec(int64(i)), 0); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	if got := testing.AllocsPerRun(50, func() { RebuildKVIndex(idx, c) }); got != 0 {
		t.Fatalf("RebuildKVIndex with no definitions allocates %v/op, want 0 (it must not walk)", got)
	}
}

// TestRebuildKVIndexRefillsFromCache is the warm-start / post-restore
// behaviour: a cache that already has content and an index that has never seen
// it end up agreeing, and the definition is published ready only after the
// walk.
func TestRebuildKVIndexRefillsFromCache(t *testing.T) {
	cfg := cache.DefaultConfig()
	cfg.NumShards = 4
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	idx := NewKVIndexFor(c)

	// Content written straight to the cache — the index hears nothing about it,
	// exactly as a warm restart or a snapshot restore leaves things.
	want := make([]string, 0, 50)
	for i := 0; i < 50; i++ {
		k := fmt.Appendf(nil, "u:%02d", i)
		if err := c.Put(k, wiringRec(7), 0); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
		want = append(want, string(k))
	}
	if err := c.Put([]byte("zz:outside"), wiringRec(7), 0); err != nil {
		t.Fatalf("put out of prefix: %v", err)
	}

	idx.Install([]kvindex.Def{wiringDef(t, "rc")})
	if idx.IsReady(wiringIndexName) {
		t.Fatal("a freshly installed definition must not be ready before its walk")
	}
	RebuildKVIndex(idx, c)
	if !idx.IsReady(wiringIndexName) {
		t.Fatal("the definition must be ready once the rebuild's walk has finished")
	}
	wantPosted(t, idx, 7, want...)
}

// --- cost ----------------------------------------------------------------

// BenchmarkPutHotPath measures one handlePut three ways: with no index at all
// (what every store that never defines one pays — a nil check), with an index
// whose definition covers the key, and with an index whose definitions do not
// (the key is outside every prefix, so the write only pays the prefix test).
func BenchmarkPutHotPath(b *testing.B) {
	newCache := func(b *testing.B) *cache.Cache {
		b.Helper()
		cfg := cache.DefaultConfig()
		cfg.NumShards = 1
		c, err := cache.New(cfg)
		if err != nil {
			b.Fatal(err)
		}
		b.Cleanup(func() { _ = c.Close() })
		return c
	}
	def := func(b *testing.B) kvindex.Def {
		b.Helper()
		d, err := kvindex.DefFrom(wire.KVIndexDef{
			Name: wiringIndexName, KeyPrefix: []byte("u:"), PayloadPath: "rc",
			Kind: wire.KVIndexKindScalar, Enabled: true,
		}, 1)
		if err != nil {
			b.Fatal(err)
		}
		return d
	}

	run := func(b *testing.B, tx *TxContext, key []byte) {
		args := wire.EncodePutArgs(key, wiringRec(5), 0)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := handlePut(tx, args); err != nil {
				b.Fatal(err)
			}
		}
	}

	b.Run("no-index", func(b *testing.B) {
		run(b, NewTxContext(newCache(b)), []byte("u:a"))
	})
	b.Run("indexed-key", func(b *testing.B) {
		c := newCache(b)
		idx := NewKVIndexFor(c)
		idx.Install([]kvindex.Def{def(b)})
		idx.MarkReady(wiringIndexName)
		run(b, NewTxContextWithIndex(c, nil, idx), []byte("u:a"))
	})
	b.Run("out-of-prefix-key", func(b *testing.B) {
		c := newCache(b)
		idx := NewKVIndexFor(c)
		idx.Install([]kvindex.Def{def(b)})
		idx.MarkReady(wiringIndexName)
		run(b, NewTxContextWithIndex(c, nil, idx), []byte("zz:a"))
	})
}
