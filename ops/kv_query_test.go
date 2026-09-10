// SPDX-License-Identifier: Apache-2.0

package ops

// Tests for the per-shard kv_query leaf: candidate selection from the index (or
// a budgeted scan), verify-on-read against the live value, and a bounded,
// key-ordered page with a continuation that pages the whole answer exactly once.
//
// The index's own behaviour belongs to ops/kvindex and is covered there. What
// is proved here is the LEAF: that the page is a correct, complete, duplicate-
// free enumeration of the answer, that every budget refuses rather than
// truncating, and that a stale posting costs a lookup rather than a wrong row.

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- harness --------------------------------------------------------------

// kvBigPad is the size of one padding cell in kvBigRec. A record BYTES cell is
// capped at wire.OperateMaxBytesLen (64 KiB), so a big fixture record is built
// from SEVERAL pad fields rather than one large one.
const kvBigPad = 60 << 10

// kvBigRec is kvRec with padFields padding cells, for the page-byte-budget
// fixture. Dynamic mode stores fields in ascending name order, which
// "pad0".."pad9" < "rc" already satisfies.
func kvBigRec(rc int64, padFields int) []byte {
	fields := make([]wire.Field, 0, padFields+1)
	for i := 0; i < padFields; i++ {
		fields = append(fields, wire.Field{
			Name: fmt.Sprintf("pad%d", i),
			Cell: wire.Cell{Type: wire.OperateTypeBytes, B: make([]byte, kvBigPad)},
		})
	}
	fields = append(fields, wire.Field{Name: "rc", Cell: wire.Cell{Type: wire.OperateTypeI64, U: su64(rc)}})
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: fields}
	b := rec.Encode()
	if b == nil {
		panic("fixture padded record failed to encode")
	}
	return b
}

// seedKV stores a record under key and posts it, through the same seam every
// write handler uses (store first, then reindex).
func seedKV(t *testing.T, tx *TxContext, key string, val []byte) {
	t.Helper()
	if err := tx.PutIndexed([]byte(key), val, 0); err != nil {
		t.Fatalf("PutIndexed(%q): %v", key, err)
	}
}

// withKVQueryBudget installs a budget for one test and restores the previous
// one afterwards. Budgets are node config, so this is package state: no test
// that calls it may run in parallel.
func withKVQueryBudget(t *testing.T, b KVQueryBudget) {
	t.Helper()
	prev := kvQueryBudget()
	SetKVQueryBudget(b)
	t.Cleanup(func() { SetKVQueryBudget(prev) })
}

func kvQueryArgsFor(t *testing.T, a wire.KVQueryArgs) []byte {
	t.Helper()
	if a.Limit == 0 {
		a.Limit = 100
	}
	b, err := wire.EncodeKVQueryArgs(a)
	if err != nil {
		t.Fatalf("EncodeKVQueryArgs(%+v): %v", a, err)
	}
	return b
}

func runKVQuery(t *testing.T, tx *TxContext, a wire.KVQueryArgs) wire.KVQueryResult {
	t.Helper()
	out, err := handleKVQuery(tx, kvQueryArgsFor(t, a))
	if err != nil {
		t.Fatalf("handleKVQuery(%+v): %v", a, err)
	}
	res, err := wire.DecodeKVQueryResult(out)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult: %v", err)
	}
	return res
}

func runKVQueryErr(t *testing.T, tx *TxContext, a wire.KVQueryArgs) error {
	t.Helper()
	out, err := handleKVQuery(tx, kvQueryArgsFor(t, a))
	if err == nil {
		t.Fatalf("handleKVQuery(%+v): want an error, got a %d-byte page", a, len(out))
	}
	if out != nil {
		t.Fatalf("handleKVQuery returned BOTH an error and a %d-byte page; a refusal is never a partial answer", len(out))
	}
	return err
}

// pageAll threads the continuation cursor until a page comes back without one,
// returning every key in order of arrival and the number of pages it took.
func pageAll(t *testing.T, tx *TxContext, a wire.KVQueryArgs, maxPages int) ([]string, int) {
	t.Helper()
	var keys []string
	pages := 0
	for {
		res := runKVQuery(t, tx, a)
		pages++
		for _, r := range res.Rows {
			keys = append(keys, string(r.Key))
		}
		if len(res.Cursor) == 0 {
			return keys, pages
		}
		if pages >= maxPages {
			t.Fatalf("paging did not terminate: still going after %d pages (%d keys)", pages, len(keys))
		}
		a.Cursor = res.Cursor
	}
}

func eqFilter(field string, v vtypes.Value) vtypes.Filter {
	return vtypes.Filter{Op: vtypes.FilterEq, Field: field, Value: v}
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func assertAscendingUnique(t *testing.T, keys []string) {
	t.Helper()
	for i := 1; i < len(keys); i++ {
		if keys[i] <= keys[i-1] {
			t.Fatalf("keys are not strictly ascending at %d: %q then %q", i, keys[i-1], keys[i])
		}
	}
}

// --- the basics -----------------------------------------------------------

func TestKVQueryReturnsMatchingKeys(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:a", kvRec(7, "gold"))
	seedKV(t, tx, "u:b", kvRec(9, "gold"))
	seedKV(t, tx, "u:c", kvRec(7, "silver"))

	res := runKVQuery(t, tx, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
	})
	got := make([]string, 0, len(res.Rows))
	for _, r := range res.Rows {
		got = append(got, string(r.Key))
		if r.Value != nil {
			t.Fatalf("return: keys must carry no value, got %d bytes for %q", len(r.Value), r.Key)
		}
	}
	if strings.Join(got, ",") != "u:a,u:c" {
		t.Fatalf("eq rc == 7: got %v, want [u:a u:c]", got)
	}
	if len(res.Cursor) != 0 {
		t.Fatalf("a complete page must carry no continuation, got %+v", res.Cursor)
	}

	// An unindexed conjunct narrows further, on the live value.
	res = runKVQuery(t, tx, wire.KVQueryArgs{
		Index: wiringIndexName,
		Filter: vtypes.Filter{Op: vtypes.FilterAnd, And: []vtypes.Filter{
			eqFilter("rc", vtypes.NewInt(7)),
			eqFilter("tier", vtypes.NewString("silver")),
		}},
	})
	if len(res.Rows) != 1 || string(res.Rows[0].Key) != "u:c" {
		t.Fatalf("rc == 7 and tier == silver: got %d rows %+v, want [u:c]", len(res.Rows), res.Rows)
	}
}

func TestKVQueryPagesInKeyOrder(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	want := make([]string, 0, 50)
	for i := 0; i < 50; i++ {
		k := fmt.Sprintf("u:%03d", i)
		seedKV(t, tx, k, kvRec(7, "gold"))
		want = append(want, k)
	}
	// A decoy that must never appear: same prefix, different indexed value.
	seedKV(t, tx, "u:999", kvRec(8, "gold"))

	a := wire.KVQueryArgs{Index: wiringIndexName, Filter: eqFilter("rc", vtypes.NewInt(7)), Limit: 20}

	// Page sizes are exactly 20 / 20 / 10, and the third page closes the query.
	var sizes []int
	var got []string
	for {
		res := runKVQuery(t, tx, a)
		sizes = append(sizes, len(res.Rows))
		for _, r := range res.Rows {
			got = append(got, string(r.Key))
		}
		if len(res.Cursor) == 0 {
			break
		}
		if len(sizes) > 10 {
			t.Fatalf("paging did not terminate: sizes %v", sizes)
		}
		a.Cursor = res.Cursor
	}
	if fmt.Sprint(sizes) != "[20 20 10]" {
		t.Fatalf("page sizes = %v, want [20 20 10]", sizes)
	}
	assertAscendingUnique(t, got)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("paged union != brute force\n got %v\nwant %v", got, want)
	}
}

func TestKVQueryContinuationIsExclusive(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	for _, k := range []string{"u:a", "u:b", "u:c"} {
		seedKV(t, tx, k, kvRec(7, "gold"))
	}
	a := wire.KVQueryArgs{Index: wiringIndexName, Filter: eqFilter("rc", vtypes.NewInt(7)), Limit: 1}
	res := runKVQuery(t, tx, a)
	if len(res.Cursor) != 1 {
		t.Fatalf("a truncated page must carry exactly one continuation, got %+v", res.Cursor)
	}
	c := res.Cursor[0]
	if c.Group != 0 || !c.More || string(c.After) != "u:a" {
		t.Fatalf("continuation = %+v, want {group 0, more, after u:a}", c)
	}
	a.Cursor = res.Cursor
	res = runKVQuery(t, tx, a)
	if len(res.Rows) != 1 || string(res.Rows[0].Key) != "u:b" {
		t.Fatalf("after u:a: got %+v, want [u:b] — the cursor is EXCLUSIVE", res.Rows)
	}
}

// --- verify-on-read -------------------------------------------------------

func TestKVQueryVerifiesOnRead(t *testing.T) {
	tx, _, idx := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:real", kvRec(7, "gold"))
	// A posting for a key the cache has never held: exactly the shape a stale
	// posting has after a key is gone but its posting is not (a torn page, an
	// abandoned walk, a reconcile that has not run yet).
	idx.Reindex([]byte("u:ghost"), kvRec(7, "gold"))

	before := idx.VerifyMisses()
	res := runKVQuery(t, tx, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
	})
	if len(res.Rows) != 1 || string(res.Rows[0].Key) != "u:real" {
		t.Fatalf("a stale posting must not become a row: got %+v", res.Rows)
	}
	if got := idx.VerifyMisses() - before; got != 1 {
		t.Fatalf("VerifyMisses advanced by %d, want 1", got)
	}
}

func TestKVQueryVerifiesTheLiveValue(t *testing.T) {
	tx, c, idx := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:a", kvRec(7, "gold"))
	// Overwrite the VALUE without reindexing, so the posting says rc == 7 while
	// the live record says rc == 99. The predicate re-runs on the live value,
	// so the row is dropped — a stale posting costs a lookup, never a wrong row.
	if err := c.Put([]byte("u:a"), kvRec(99, "gold"), 0); err != nil {
		t.Fatal(err)
	}
	// The posting must still SAY rc == 7, or this test would pass for the wrong
	// reason (no candidate at all, rather than a candidate the predicate drops).
	if stale := posted(t, idx, 7); len(stale) != 1 || stale[0] != "u:a" {
		t.Fatalf("fixture: want a stale posting for u:a under rc == 7, got %v", stale)
	}
	res := runKVQuery(t, tx, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
	})
	if len(res.Rows) != 0 {
		t.Fatalf("a stale posting whose live value no longer matches must yield no row, got %+v", res.Rows)
	}
}

func TestKVQueryOnExpiredKeyDoesNotDeadlock(t *testing.T) {
	// THE LOCK-ORDER REGRESSION TEST. On a non-replicated shard, Cache.Get of an
	// expired key calls dropExpiredLocked under the shard WRITE lock, which fires
	// onRemove -> kvindex.Set.Drop -> Set.mu. If the query held any Set lock
	// across its tx.Get, this self-deadlocks single-threaded, no concurrency
	// required. It must return cleanly.
	tx, _, _ := newIndexedTx(t, "rc")
	for i := 0; i < 16; i++ {
		k := []byte(fmt.Sprintf("u:%02d", i))
		if err := tx.PutIndexed(k, kvRec(7, "gold"), 40*time.Millisecond); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(150 * time.Millisecond) // every candidate is now expired

	done := make(chan wire.KVQueryResult, 1)
	go func() {
		done <- runKVQuery(t, tx, wire.KVQueryArgs{
			Index:  wiringIndexName,
			Filter: eqFilter("rc", vtypes.NewInt(7)),
		})
	}()
	select {
	case res := <-done:
		if len(res.Rows) != 0 {
			t.Fatalf("expired keys must not be rows, got %+v", res.Rows)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("kv_query deadlocked verifying expired candidates (cache lock -> index lock inversion)")
	}
}

// --- return modes ---------------------------------------------------------

func TestKVQueryReturnModes(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	good := kvRec(7, "gold")
	junk := []byte{0xFF, 0xFE, 0xFD, 0xFC}
	if _, err := wire.DecodeRecord(junk); err == nil {
		t.Fatal("fixture: the junk value must NOT decode as a record")
	}
	seedKV(t, tx, "u:a", good)
	seedKV(t, tx, "u:junk", junk)

	// A scan with no filter matches everything, which is what puts the
	// non-record value in front of every return mode.
	base := wire.KVQueryArgs{Scan: true}

	keysOnly := runKVQuery(t, tx, base)
	if len(keysOnly.Rows) != 2 {
		t.Fatalf("return: keys — got %d rows, want 2", len(keysOnly.Rows))
	}
	for _, r := range keysOnly.Rows {
		if r.Value != nil {
			t.Fatalf("return: keys must carry no value, got %d bytes for %q", len(r.Value), r.Key)
		}
	}

	vals := base
	vals.Return = wire.KVQueryReturnValues
	vres := runKVQuery(t, tx, vals)
	if len(vres.Rows) != 2 {
		t.Fatalf("return: values — got %d rows, want 2", len(vres.Rows))
	}
	byKey := map[string][]byte{}
	for _, r := range vres.Rows {
		byKey[string(r.Key)] = r.Value
	}
	if !bytes.Equal(byKey["u:a"], good) || !bytes.Equal(byKey["u:junk"], junk) {
		t.Fatalf("return: values must be the raw stored bytes, got %v", byKey)
	}

	recs := base
	recs.Return = wire.KVQueryReturnRecords
	rres := runKVQuery(t, tx, recs)
	if len(rres.Rows) != 1 || string(rres.Rows[0].Key) != "u:a" {
		t.Fatalf("return: records must SKIP the value DecodeRecord rejects, got %+v", rres.Rows)
	}
	if _, err := wire.DecodeRecord(rres.Rows[0].Value); err != nil {
		t.Fatalf("return: records emitted a value DecodeRecord rejects: %v", err)
	}
}

// --- budgets --------------------------------------------------------------

func TestKVQueryUnindexedNeedsScan(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:a", kvRec(7, "gold"))

	// The filter names an index but no leaf of it can drive that index, and the
	// caller did not consent to a scan.
	unindexed := wire.KVQueryArgs{Index: wiringIndexName, Filter: eqFilter("tier", vtypes.NewString("gold"))}
	if err := runKVQueryErr(t, tx, unindexed); !errors.Is(err, ErrKVQueryScanRequired) {
		t.Fatalf("want ErrKVQueryScanRequired, got %v", err)
	}
	// A negation over the indexed path is not a selector either.
	notLeaf := wire.KVQueryArgs{Index: wiringIndexName, Filter: vtypes.Filter{
		Op: vtypes.FilterNot, Not: ptrFilter(eqFilter("rc", vtypes.NewInt(7))),
	}}
	if err := runKVQueryErr(t, tx, notLeaf); !errors.Is(err, ErrKVQueryScanRequired) {
		t.Fatalf("not-wrapped: want ErrKVQueryScanRequired, got %v", err)
	}

	// The same query WITH consent scans and answers.
	unindexed.Scan = true
	res := runKVQuery(t, tx, unindexed)
	if len(res.Rows) != 1 || string(res.Rows[0].Key) != "u:a" {
		t.Fatalf("scan: true must answer the same question, got %+v", res.Rows)
	}
}

func TestKVQueryCandidateBudget(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	for i := 0; i < 40; i++ {
		seedKV(t, tx, fmt.Sprintf("u:%03d", i), kvRec(7, "gold"))
	}
	withKVQueryBudget(t, KVQueryBudget{Candidates: 8, Scan: 1 << 20, ScanChunk: 1000})

	err := runKVQueryErr(t, tx, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
	})
	if !errors.Is(err, kvindex.ErrCandidateBudget) {
		t.Fatalf("want ErrCandidateBudget, got %v", err)
	}
}

func TestKVQueryScanBudget(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	for i := 0; i < 40; i++ {
		seedKV(t, tx, fmt.Sprintf("u:%03d", i), kvRec(7, "gold"))
	}
	withKVQueryBudget(t, KVQueryBudget{Candidates: 1 << 20, Scan: 10, ScanChunk: 1000})

	err := runKVQueryErr(t, tx, wire.KVQueryArgs{Scan: true, Filter: eqFilter("rc", vtypes.NewInt(7))})
	if !errors.Is(err, ErrKVQueryScanBudget) {
		t.Fatalf("want ErrKVQueryScanBudget, got %v", err)
	}
}

func TestKVQueryPageByteBudget(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	const (
		rows      = 100
		padFields = 4 // ~240 KiB per record, so ~34 rows fill the 8 MiB page cap
	)
	want := make([]string, 0, rows)
	for i := 0; i < rows; i++ {
		k := fmt.Sprintf("u:%03d", i)
		seedKV(t, tx, k, kvBigRec(7, padFields))
		want = append(want, k)
	}

	a := wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  rows,
		Return: wire.KVQueryReturnValues,
	}
	res := runKVQuery(t, tx, a)
	if len(res.Rows) == 0 || len(res.Rows) >= rows {
		t.Fatalf("the byte budget must truncate the page below the row limit: got %d of %d rows", len(res.Rows), rows)
	}
	total := 0
	for _, r := range res.Rows {
		total += len(r.Key) + len(r.Value)
	}
	if total > wire.KVQueryMaxPageBytes {
		t.Fatalf("page carries %d bytes, over the %d cap", total, wire.KVQueryMaxPageBytes)
	}
	if len(res.Cursor) != 1 || !res.Cursor[0].More {
		t.Fatalf("a byte-truncated page must set More, got %+v", res.Cursor)
	}

	// And the rest pages out with no gap and no duplicate.
	got, pages := pageAll(t, tx, a, 200)
	assertAscendingUnique(t, got)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("byte-budget paging lost or repeated rows: %d keys in %d pages", len(got), pages)
	}
}

// --- scan paging ----------------------------------------------------------

func TestKVQueryScanPagesToCompletion(t *testing.T) {
	if testing.Short() {
		t.Skip("25 000 keys x 25 full walks")
	}
	tx, _, _ := newIndexedTx(t, "rc")
	const n = 25000
	want := make([]string, 0, 64)
	for i := 0; i < n; i++ {
		k := fmt.Sprintf("u:%05d", i)
		rc := int64(1)
		if i%500 == 0 {
			rc = 7
			want = append(want, k)
		}
		seedKV(t, tx, k, kvRec(rc, "gold"))
	}
	// ScanChunk 1000 over 25 000 keys: the chunk heap overflows on every page,
	// so paging terminates only if the continuation advances by the whole
	// chunk. The filter is selective enough that the row limit is never the
	// binding constraint, which is what makes that the observable behaviour.
	withKVQueryBudget(t, KVQueryBudget{Candidates: 1 << 20, Scan: 1 << 20, ScanChunk: 1000})

	got, pages := pageAll(t, tx, wire.KVQueryArgs{
		Scan:   true,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  50,
	}, 200)
	assertAscendingUnique(t, got)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("scan paging: got %d keys in %d pages, want %d", len(got), pages, len(want))
	}
	// About ceil(n / ScanChunk) pages: fewer would mean the chunk heap is not
	// bounding the page at all, more would mean the continuation is advancing
	// by matched rows instead of by the whole chunk.
	if pages < 20 || pages > 40 {
		t.Fatalf("scan paging took %d pages for %d keys at chunk 1000, want about %d", pages, n, n/1000)
	}
}

func TestKVQueryScanWalkErrorIsRetryable(t *testing.T) {
	// A walk that fails part-way (the cache was closed under it — a shard being
	// removed) must surface as a RETRYABLE refusal and never as a short page:
	// the keys it did not reach are indistinguishable from keys that do not
	// match.
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:a", kvRec(7, "gold"))

	closed := errors.New("cache: closed")
	walk := kvindex.Walker(func(fn func(key, value []byte) bool) error {
		fn([]byte("u:a"), kvRec(7, "gold")) // some progress, then the failure
		return closed
	})
	out, err := scanPage(tx, walk, nil, nil, wire.KVQueryArgs{Scan: true, Limit: 10}, 0, kvQueryBudget())
	if out != nil {
		t.Fatalf("a failed walk must yield no page, got %d bytes", len(out))
	}
	if !errors.Is(err, closed) {
		t.Fatalf("the walk error must be wrapped, got %v", err)
	}
	if !errors.Is(err, ErrKVQueryUnavailable) {
		t.Fatalf("a failed walk must be classified retryable (ErrKVQueryUnavailable), got %v", err)
	}
}

func TestKVQueryScanWalkAbortedIsRetryable(t *testing.T) {
	// kvindex.ErrWalkAborted is the concrete walk error this branch's fence
	// produces: the cluster observer's wrapper returns it when a shard is being
	// removed under the walk. It must reach the caller as the same retryable
	// refusal any other cut-short walk does — never as a short page, because the
	// keys the walk did not reach are indistinguishable from keys that did not
	// match.
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:a", kvRec(7, "gold"))

	walk := kvindex.Walker(func(fn func(key, value []byte) bool) error {
		fn([]byte("u:a"), kvRec(7, "gold"))
		return kvindex.ErrWalkAborted
	})
	out, err := scanPage(tx, walk, nil, nil, wire.KVQueryArgs{Scan: true, Limit: 10}, 0, kvQueryBudget())
	if out != nil {
		t.Fatalf("an aborted walk must yield no page, got %d bytes", len(out))
	}
	if !errors.Is(err, kvindex.ErrWalkAborted) {
		t.Fatalf("the abort must stay inspectable through the wrap, got %v", err)
	}
	if !errors.Is(err, ErrKVQueryUnavailable) {
		t.Fatalf("an aborted walk must be classified retryable, got %v", err)
	}
}

// --- the index gate -------------------------------------------------------

func TestKVQueryRouteGate(t *testing.T) {
	t.Run("no index on the dispatcher", func(t *testing.T) {
		cfg := cache.DefaultConfig()
		cfg.NumShards = 1
		c, err := cache.New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Close() })
		tx := NewTxContext(c)
		if err := runKVQueryErr(t, tx, wire.KVQueryArgs{Scan: true}); !errors.Is(err, ErrKVIndexUnavailable) {
			t.Fatalf("want ErrKVIndexUnavailable, got %v", err)
		}
	})

	t.Run("unknown index", func(t *testing.T) {
		tx, _, _ := newIndexedTx(t, "rc")
		err := runKVQueryErr(t, tx, wire.KVQueryArgs{Index: "nope", Filter: eqFilter("rc", vtypes.NewInt(7))})
		if !errors.Is(err, kvindex.ErrNoSuchIndex) {
			t.Fatalf("want ErrNoSuchIndex, got %v", err)
		}
		if !strings.Contains(err.Error(), "shard group 0") {
			t.Fatalf("the refusal must name the group it applies to, got %q", err)
		}
	})

	t.Run("installed but still building", func(t *testing.T) {
		// Readiness is answered by ONE replica per group, so a stale read that
		// lands on a replica still backfilling must refuse rather than answer
		// from a proper subset of the postings.
		cfg := cache.DefaultConfig()
		cfg.NumShards = 1
		c, err := cache.New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Close() })
		idx := NewKVIndexFor(c)
		idx.Install([]kvindex.Def{wiringDef(t, "rc")}) // installed, NOT marked ready
		tx := NewTxContextWithIndex(c, nil, idx)

		err = runKVQueryErr(t, tx, wire.KVQueryArgs{
			Index:  wiringIndexName,
			Filter: eqFilter("rc", vtypes.NewInt(7)),
		})
		if !errors.Is(err, kvindex.ErrIndexBuilding) {
			t.Fatalf("want ErrIndexBuilding, got %v", err)
		}
		if !strings.Contains(err.Error(), "shard group 0") {
			t.Fatalf("the refusal must name the group it applies to, got %q", err)
		}
		// A building index must not be answered from a scan behind the caller's
		// back either, even when the caller allowed one.
		err = runKVQueryErr(t, tx, wire.KVQueryArgs{
			Index:  wiringIndexName,
			Filter: eqFilter("rc", vtypes.NewInt(7)),
			Scan:   true,
		})
		if !errors.Is(err, kvindex.ErrIndexBuilding) {
			t.Fatalf("scan: true must not silently replace a building index, got %v", err)
		}
	})
}

func TestKVQueryNeverLeaksIndexChanged(t *testing.T) {
	// kvindex.ErrIndexChanged means the definition moved under the query, so the
	// postings answer a different question. The leaf re-reads the definition and
	// retries once; if it still cannot get a coherent read it returns the
	// RETRYABLE refusal. What it must never do is leak ErrIndexChanged to the
	// caller or fall through to a silent scan.
	tx, _, idx := newIndexedTx(t, "rc")
	for i := 0; i < 40; i++ {
		seedKV(t, tx, fmt.Sprintf("u:%03d", i), kvRec(7, "gold"))
	}
	shapes := []kvindex.Def{wiringDef(t, "rc"), wiringDef(t, "tier")}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			idx.Install([]kvindex.Def{shapes[i%2]})
			idx.MarkReady(wiringIndexName)
		}
	}()

	args := kvQueryArgsFor(t, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  10,
	})
	for i := 0; i < 300; i++ {
		out, err := handleKVQuery(tx, args)
		switch {
		case err == nil:
			if _, derr := wire.DecodeKVQueryResult(out); derr != nil {
				t.Errorf("undecodable page: %v", derr)
			}
		case errors.Is(err, kvindex.ErrIndexChanged):
			t.Errorf("ErrIndexChanged leaked to the caller: %v", err)
		case errors.Is(err, kvindex.ErrIndexBuilding),
			errors.Is(err, kvindex.ErrNoSuchIndex),
			errors.Is(err, ErrKVQueryScanRequired):
			// Every one of these is a legitimate answer while the definition
			// set is being replaced under the query.
		default:
			t.Errorf("unexpected error: %v", err)
		}
		if t.Failed() {
			break
		}
	}
	close(stop)
	wg.Wait()
}

// --- scope ----------------------------------------------------------------

func TestKVQueryPrefixIsNotAFilter(t *testing.T) {
	// An INDEXED kv_query answers "the keys under this definition's KeyPrefix
	// that match the filter", never "the keys that match the filter". A key
	// outside the prefix is outside the answer; only scan: true sees it.
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:in", kvRec(7, "gold"))
	seedKV(t, tx, "x:out", kvRec(7, "gold"))

	res := runKVQuery(t, tx, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
	})
	if len(res.Rows) != 1 || string(res.Rows[0].Key) != "u:in" {
		t.Fatalf("the indexed path must not see outside its prefix, got %+v", res.Rows)
	}

	scan := runKVQuery(t, tx, wire.KVQueryArgs{Scan: true, Filter: eqFilter("rc", vtypes.NewInt(7))})
	got := make([]string, 0, len(scan.Rows))
	for _, r := range scan.Rows {
		got = append(got, string(r.Key))
	}
	if strings.Join(sortedCopy(got), ",") != "u:in,x:out" {
		t.Fatalf("a scan is not scoped by any prefix: got %v, want [u:in x:out]", got)
	}
}

// --- the oracle -----------------------------------------------------------

func TestKVQueryEqualsBruteForce(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	rng := rand.New(rand.NewSource(20260910))

	tiers := []string{"gold", "silver", "bronze"}
	type entry struct {
		key string
		val []byte
	}
	entries := make([]entry, 0, 300)
	for i := 0; i < 300; i++ {
		// A quarter of the keys sit OUTSIDE the definition's prefix, so the
		// oracle has to be restricted to the prefix or the comparison fails.
		prefix := "u:"
		if i%4 == 3 {
			prefix = "x:"
		}
		k := fmt.Sprintf("%s%04d", prefix, i)
		v := kvRec(int64(rng.Intn(12)), tiers[rng.Intn(len(tiers))])
		seedKV(t, tx, k, v)
		entries = append(entries, entry{k, v})
	}

	for n := 0; n < 40; n++ {
		// One indexed leaf (the driver) plus 0-2 unindexed conjuncts.
		var driver vtypes.Filter
		switch rng.Intn(3) {
		case 0:
			driver = eqFilter("rc", vtypes.NewInt(int64(rng.Intn(12))))
		case 1:
			driver = vtypes.Filter{Op: vtypes.FilterGte, Field: "rc", Value: vtypes.NewInt(int64(rng.Intn(12)))}
		default:
			driver = vtypes.Filter{Op: vtypes.FilterIn, Field: "rc", Value: vtypes.NewInts([]int64{
				int64(rng.Intn(12)), int64(rng.Intn(12)),
			})}
		}
		conjuncts := []vtypes.Filter{driver}
		for e := rng.Intn(3); e > 0; e-- {
			switch rng.Intn(2) {
			case 0:
				conjuncts = append(conjuncts, eqFilter("tier", vtypes.NewString(tiers[rng.Intn(len(tiers))])))
			default:
				conjuncts = append(conjuncts, vtypes.Filter{
					Op:  vtypes.FilterNot,
					Not: ptrFilter(eqFilter("tier", vtypes.NewString(tiers[rng.Intn(len(tiers))]))),
				})
			}
		}
		f := driver
		if len(conjuncts) > 1 {
			f = vtypes.Filter{Op: vtypes.FilterAnd, And: conjuncts}
		}

		// The oracle: the compiled predicate over every seeded value, RESTRICTED
		// TO THE DEFINITION'S PREFIX (which is what an indexed query answers).
		pred, err := BuildKVPredicate(f)
		if err != nil {
			t.Fatalf("filter %d: BuildKVPredicate: %v", n, err)
		}
		var want []string
		for _, e := range entries {
			if !strings.HasPrefix(e.key, "u:") {
				continue
			}
			if pred == nil || pred(kvMeta(e.val)) {
				want = append(want, e.key)
			}
		}
		sort.Strings(want)

		got, _ := pageAll(t, tx, wire.KVQueryArgs{
			Index:  wiringIndexName,
			Filter: f,
			Limit:  7, // small, so every case pages
		}, 200)
		assertAscendingUnique(t, got)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Fatalf("filter %d %+v:\n got %d keys %v\nwant %d keys %v", n, f, len(got), got, len(want), want)
		}
	}
}

// --- hostile input --------------------------------------------------------

func TestKVQueryHostileArgs(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "u:a", kvRec(7, "gold"))

	valid := kvQueryArgsFor(t, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  10,
	})

	// Truncations: every prefix of a valid frame must be refused, never panic.
	for n := 0; n < len(valid); n++ {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("handleKVQuery panicked on a %d-byte prefix: %v", n, r)
				}
			}()
			if _, err := handleKVQuery(tx, valid[:n]); err == nil {
				t.Fatalf("a %d-byte prefix of a %d-byte frame was accepted", n, len(valid))
			}
		}()
	}

	// Byte sweep: flip each byte to a few hostile values. The frame may become
	// valid again (a different limit, say), so an answer is allowed — a panic
	// is not.
	for i := 0; i < len(valid); i++ {
		for _, b := range []byte{0x00, 0x01, 0x7F, 0x80, 0xFF} {
			frame := append([]byte(nil), valid...)
			frame[i] = b
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("handleKVQuery panicked with byte %d set to %#x: %v", i, b, r)
					}
				}()
				_, _ = handleKVQuery(tx, frame)
			}()
		}
	}

	// A field that record.ParsePath refuses is a query ERROR, not a silently
	// false predicate that answers "no matches" to a question nobody asked.
	err := runKVQueryErr(t, tx, wire.KVQueryArgs{Scan: true, Filter: eqFilter("a/b/c/d", vtypes.NewInt(1))})
	if !errors.Is(err, ErrKVQueryFilter) {
		t.Fatalf("want ErrKVQueryFilter for an unparseable path, got %v", err)
	}
	err = runKVQueryErr(t, tx, wire.KVQueryArgs{Scan: true, Filter: eqFilter(KVRecordAlias, vtypes.NewInt(1))})
	if !errors.Is(err, ErrKVQueryFilter) {
		t.Fatalf("want ErrKVQueryFilter for the reserved alias, got %v", err)
	}
}

func TestKVQueryBudgetSanitised(t *testing.T) {
	// Budgets are node config. A zero or negative one is a misconfiguration,
	// not an instruction to refuse every query or to loop forever on a
	// zero-sized scan chunk.
	prev := kvQueryBudget()
	t.Cleanup(func() { SetKVQueryBudget(prev) })
	SetKVQueryBudget(KVQueryBudget{Candidates: 0, Scan: -1, ScanChunk: 0})
	got := kvQueryBudget()
	if got.Candidates <= 0 || got.Scan <= 0 || got.ScanChunk <= 0 {
		t.Fatalf("SetKVQueryBudget must fall back to the defaults for non-positive values, got %+v", got)
	}
}

// --- the scan chunk heap --------------------------------------------------

// heapContents drains h and returns what it held, ascending.
func heapContents(h *kvKeyHeap) []string {
	out := make([]string, 0, h.Len())
	for _, k := range h.ascending() {
		out = append(out, string(k))
	}
	return out
}

// assertSmallestPrefix pins the ONE invariant scanPage's soundness rests on:
// what the chunk retained is the smallest len(got) keys of everything offered.
// scanPage's continuation is the largest key retained, so a retained set that
// is NOT a prefix of the sorted offer set leaves a hole below the continuation
// — and a key in that hole is never visited again on any later page.
func assertSmallestPrefix(t *testing.T, got, offered []string) {
	t.Helper()
	if len(got) == 0 {
		t.Fatal("the chunk kept nothing; a page that retains no key never advances")
	}
	want := sortedCopy(offered)[:len(got)]
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("the chunk is not the smallest %d keys offered\n got %v\nwant %v", len(got), got, want)
	}
}

func TestKVQueryScanHeapKeepsTheSmallestKeys(t *testing.T) {
	h := &kvKeyHeap{max: 8, maxBytes: 1 << 20}
	rng := rand.New(rand.NewSource(7))
	offered := make([]string, 0, 200)
	for i := 0; i < 200; i++ {
		k := fmt.Sprintf("k%04d", rng.Intn(100000))
		if slices.Contains(offered, k) {
			continue
		}
		offered = append(offered, k)
		h.offer([]byte(k))
	}
	if h.Len() != 8 {
		t.Fatalf("the count bound must hold: %d keys, want 8", h.Len())
	}
	assertSmallestPrefix(t, heapContents(h), offered)
}

func TestKVQueryScanHeapBoundsItsBytes(t *testing.T) {
	// THE ORDER THAT USED TO DEFEAT THE CAP: short high-sorting keys first, so
	// the heap fills by COUNT while staying tiny in bytes, then long low-sorting
	// keys, every one of which DISPLACES a short key. The old cap guarded only
	// the growth branch, so displacement grew the heap without limit.
	const cap = 8 << 10
	h := &kvKeyHeap{max: 64, maxBytes: cap}

	offered := make([]string, 0, 84)
	for i := 0; i < 64; i++ {
		k := fmt.Sprintf("z%03d", i)
		offered = append(offered, k)
		h.offer([]byte(k))
	}
	if h.bytes > cap {
		t.Fatalf("short keys alone already exceeded the cap: %d > %d", h.bytes, cap)
	}
	for i := 0; i < 20; i++ {
		k := fmt.Sprintf("a%03d%s", i, strings.Repeat("p", 1024))
		offered = append(offered, k)
		h.offer([]byte(k))
	}

	if h.bytes > cap {
		t.Fatalf("displacement escaped the byte cap: %d bytes held, cap %d", h.bytes, cap)
	}
	// And it stayed SOUND while doing it: the low-sorting long keys are exactly
	// the ones that belong in the chunk, so they must all be there.
	assertSmallestPrefix(t, heapContents(h), offered)
}

func TestKVQueryScanHeapKeepsOneOversizeKey(t *testing.T) {
	// A single key larger than the whole cap is kept anyway: a chunk of nothing
	// yields a continuation that never advances, which is an infinite paging
	// loop. One key is bounded by the encoder's 64 KiB key cap.
	h := &kvKeyHeap{max: 16, maxBytes: 64}
	h.offer([]byte(strings.Repeat("x", 4096)))
	if h.Len() != 1 {
		t.Fatalf("an oversize lone key must be kept, got %d keys", h.Len())
	}
}
