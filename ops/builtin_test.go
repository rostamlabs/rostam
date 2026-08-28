// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

func newTestSetup(t *testing.T) (*Registry, *TxContext) {
	t.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatal(err)
	}
	return r, NewTxContext(c)
}

func TestBuiltinPutGetRoundtrip(t *testing.T) {
	r, tx := newTestSetup(t)

	putH, kind, _, ok := r.Lookup("put")
	if !ok {
		t.Fatal("put not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("put kind = %v, want OpReadWrite", kind)
	}
	args := EncodePutArgs([]byte("k"), []byte("v"), 0)
	if _, err := putH(tx, args); err != nil {
		t.Fatalf("put handler: %v", err)
	}

	getH, kind, _, ok := r.Lookup("get")
	if !ok {
		t.Fatal("get not registered")
	}
	if kind != OpReadOnly {
		t.Fatalf("get kind = %v, want OpReadOnly", kind)
	}
	res, err := getH(tx, EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("get handler: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("get result = %q, want v", res)
	}
}

func TestBuiltinGetMissingReturnsErr(t *testing.T) {
	r, tx := newTestSetup(t)
	getH, _, _, _ := r.Lookup("get")
	_, err := getH(tx, EncodeKeyArgs([]byte("missing")))
	if err != cache.ErrNotFound {
		t.Fatalf("get missing: err = %v, want ErrNotFound", err)
	}
}

func TestBuiltinDel(t *testing.T) {
	r, tx := newTestSetup(t)
	putH, _, _, _ := r.Lookup("put")
	_, _ = putH(tx, EncodePutArgs([]byte("k"), []byte("v"), 0))

	delH, kind, _, ok := r.Lookup("del")
	if !ok {
		t.Fatal("del not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("del kind = %v, want OpReadWrite", kind)
	}
	res, err := delH(tx, EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("del: %v", err)
	}
	if len(res) != 1 || res[0] != 1 {
		t.Fatalf("del result = %v, want [1] (true)", res)
	}
	// Second del returns 0
	res, _ = delH(tx, EncodeKeyArgs([]byte("k")))
	if len(res) != 1 || res[0] != 0 {
		t.Fatalf("del missing result = %v, want [0]", res)
	}
}

func TestBuiltinExpire(t *testing.T) {
	r, tx := newTestSetup(t)
	putH, _, _, _ := r.Lookup("put")
	_, _ = putH(tx, EncodePutArgs([]byte("k"), []byte("v"), 0))

	expH, kind, _, ok := r.Lookup("expire")
	if !ok {
		t.Fatal("expire not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("expire kind = %v, want OpReadWrite", kind)
	}
	if _, err := expH(tx, EncodeExpireArgs([]byte("k"), 10*time.Millisecond)); err != nil {
		t.Fatalf("expire: %v", err)
	}
	time.Sleep(30 * time.Millisecond)
	getH, _, _, _ := r.Lookup("get")
	if _, err := getH(tx, EncodeKeyArgs([]byte("k"))); err != cache.ErrNotFound {
		t.Fatalf("post-expire get: err = %v, want ErrNotFound", err)
	}
}

func TestBuiltinIncrCreatesKey(t *testing.T) {
	r, tx := newTestSetup(t)
	incrH, kind, _, ok := r.Lookup("incr")
	if !ok {
		t.Fatal("incr not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("incr kind = %v, want OpReadWrite", kind)
	}
	res, err := incrH(tx, EncodeIncrArgs([]byte("counter"), 5))
	if err != nil {
		t.Fatalf("incr: %v", err)
	}
	got, err := DecodeIncrResult(res)
	if err != nil {
		t.Fatal(err)
	}
	if got != 5 {
		t.Fatalf("incr result = %d, want 5", got)
	}
}

func TestBuiltinIncrAccumulates(t *testing.T) {
	r, tx := newTestSetup(t)
	incrH, _, _, _ := r.Lookup("incr")
	for range 3 {
		_, err := incrH(tx, EncodeIncrArgs([]byte("counter"), 2))
		if err != nil {
			t.Fatal(err)
		}
	}
	res, _ := incrH(tx, EncodeIncrArgs([]byte("counter"), -1))
	v, _ := DecodeIncrResult(res)
	if v != 5 { // 2 + 2 + 2 - 1
		t.Fatalf("incr accumulated = %d, want 5", v)
	}
}

func TestBuiltinSetNX(t *testing.T) {
	r, tx := newTestSetup(t)
	h, kind, _, ok := r.Lookup("set_nx")
	if !ok {
		t.Fatal("set_nx not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("set_nx kind = %v, want OpReadWrite", kind)
	}
	getH, _, _, _ := r.Lookup("get")

	// Stores when absent.
	res, err := h(tx, EncodeSetNXArgs([]byte("k"), []byte("v1"), 0))
	if err != nil {
		t.Fatalf("set_nx: %v", err)
	}
	if stored, _ := DecodeCASResult(res); !stored {
		t.Fatal("set_nx on absent key = false, want true (stored)")
	}
	got, _ := getH(tx, EncodeKeyArgs([]byte("k")))
	if !bytes.Equal(got, []byte("v1")) {
		t.Fatalf("value = %q, want v1", got)
	}

	// No-op when present: returns false and does NOT overwrite.
	res, _ = h(tx, EncodeSetNXArgs([]byte("k"), []byte("v2"), 0))
	if stored, _ := DecodeCASResult(res); stored {
		t.Fatal("set_nx on present key = true, want false")
	}
	got, _ = getH(tx, EncodeKeyArgs([]byte("k")))
	if !bytes.Equal(got, []byte("v1")) {
		t.Fatalf("value overwritten to %q, want v1", got)
	}
}

func TestBuiltinSetNXReacquiresAfterExpiry(t *testing.T) {
	r, tx := newTestSetup(t)
	h, _, _, _ := r.Lookup("set_nx")

	res, _ := h(tx, EncodeSetNXArgs([]byte("lock"), []byte("a"), 10*time.Millisecond))
	if stored, _ := DecodeCASResult(res); !stored {
		t.Fatal("first set_nx not stored")
	}
	// While live, a second set_nx is refused.
	res, _ = h(tx, EncodeSetNXArgs([]byte("lock"), []byte("b"), 10*time.Millisecond))
	if stored, _ := DecodeCASResult(res); stored {
		t.Fatal("set_nx acquired a still-live key")
	}
	// After expiry, set_nx re-acquires.
	time.Sleep(30 * time.Millisecond)
	res, _ = h(tx, EncodeSetNXArgs([]byte("lock"), []byte("c"), 0))
	if stored, _ := DecodeCASResult(res); !stored {
		t.Fatal("set_nx did not re-acquire after expiry")
	}
	getH, _, _, _ := r.Lookup("get")
	got, _ := getH(tx, EncodeKeyArgs([]byte("lock")))
	if !bytes.Equal(got, []byte("c")) {
		t.Fatalf("value = %q, want c", got)
	}
}

func TestBuiltinCAS(t *testing.T) {
	r, tx := newTestSetup(t)
	h, kind, _, ok := r.Lookup("cas")
	if !ok {
		t.Fatal("cas not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("cas kind = %v, want OpReadWrite", kind)
	}
	putH, _, _, _ := r.Lookup("put")
	getH, _, _, _ := r.Lookup("get")

	_, _ = putH(tx, EncodePutArgs([]byte("k"), []byte("v1"), 0))

	// Match writes.
	res, err := h(tx, EncodeCASArgs([]byte("k"), []byte("v2"), true, []byte("v1"), 0))
	if err != nil {
		t.Fatalf("cas: %v", err)
	}
	if ok, _ := DecodeCASResult(res); !ok {
		t.Fatal("cas match = false, want true")
	}
	got, _ := getH(tx, EncodeKeyArgs([]byte("k")))
	if !bytes.Equal(got, []byte("v2")) {
		t.Fatalf("value = %q, want v2", got)
	}

	// Mismatch no-ops.
	res, _ = h(tx, EncodeCASArgs([]byte("k"), []byte("v3"), true, []byte("WRONG"), 0))
	if ok, _ := DecodeCASResult(res); ok {
		t.Fatal("cas mismatch = true, want false")
	}
	got, _ = getH(tx, EncodeKeyArgs([]byte("k")))
	if !bytes.Equal(got, []byte("v2")) {
		t.Fatalf("value changed on mismatch to %q, want v2", got)
	}
}

func TestBuiltinCASExpectAbsent(t *testing.T) {
	r, tx := newTestSetup(t)
	h, _, _, _ := r.Lookup("cas")
	getH, _, _, _ := r.Lookup("get")

	// Expect-absent (hasExpected=false) succeeds on an absent key.
	res, _ := h(tx, EncodeCASArgs([]byte("k"), []byte("v1"), false, nil, 0))
	if ok, _ := DecodeCASResult(res); !ok {
		t.Fatal("cas expect-absent on absent = false, want true")
	}
	got, _ := getH(tx, EncodeKeyArgs([]byte("k")))
	if !bytes.Equal(got, []byte("v1")) {
		t.Fatalf("value = %q, want v1", got)
	}

	// Expect-absent fails on a present key.
	res, _ = h(tx, EncodeCASArgs([]byte("k"), []byte("v2"), false, nil, 0))
	if ok, _ := DecodeCASResult(res); ok {
		t.Fatal("cas expect-absent on present = true, want false")
	}

	// hasExpected=true fails on an absent key.
	res, _ = h(tx, EncodeCASArgs([]byte("absent"), []byte("v"), true, []byte("anything"), 0))
	if ok, _ := DecodeCASResult(res); ok {
		t.Fatal("cas expect-value on absent = true, want false")
	}
}

func TestBuiltinCompareAndDel(t *testing.T) {
	r, tx := newTestSetup(t)
	h, kind, _, ok := r.Lookup("cad")
	if !ok {
		t.Fatal("cad not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("cad kind = %v, want OpReadWrite", kind)
	}
	putH, _, _, _ := r.Lookup("put")
	getH, _, _, _ := r.Lookup("get")

	_, _ = putH(tx, EncodePutArgs([]byte("k"), []byte("tok"), 0))

	// Mismatch no-ops (key stays).
	res, _ := h(tx, EncodeCADArgs([]byte("k"), []byte("WRONG")))
	if ok, _ := DecodeCASResult(res); ok {
		t.Fatal("cad mismatch = true, want false")
	}
	if _, err := getH(tx, EncodeKeyArgs([]byte("k"))); err != nil {
		t.Fatalf("key deleted on mismatch: %v", err)
	}

	// Match deletes.
	res, _ = h(tx, EncodeCADArgs([]byte("k"), []byte("tok")))
	if ok, _ := DecodeCASResult(res); !ok {
		t.Fatal("cad match = false, want true")
	}
	if _, err := getH(tx, EncodeKeyArgs([]byte("k"))); err != cache.ErrNotFound {
		t.Fatalf("post-cad get: err = %v, want ErrNotFound", err)
	}

	// No-op on absent.
	res, _ = h(tx, EncodeCADArgs([]byte("absent"), []byte("x")))
	if ok, _ := DecodeCASResult(res); ok {
		t.Fatal("cad on absent = true, want false")
	}
}

func TestCASCADCodecs(t *testing.T) {
	// Short-args rejection (never a panic) for the new decoders.
	if _, _, _, _, _, err := DecodeCASArgs([]byte{0}); err != ErrShortArgs {
		t.Fatalf("DecodeCASArgs short: err = %v, want ErrShortArgs", err)
	}
	if _, _, err := DecodeCADArgs([]byte{0}); err != ErrShortArgs {
		t.Fatalf("DecodeCADArgs short: err = %v, want ErrShortArgs", err)
	}
	// A value-length claim with no body behind it is rejected, not read past.
	if _, _, _, _, _, err := DecodeCASArgs([]byte{0, 0, 0, 0, 0, 5}); err != ErrShortArgs {
		t.Fatalf("DecodeCASArgs truncated val: err = %v, want ErrShortArgs", err)
	}
	if _, _, err := DecodeCADArgs([]byte{0, 0, 0, 0, 0, 5}); err != ErrShortArgs {
		t.Fatalf("DecodeCADArgs truncated expected: err = %v, want ErrShortArgs", err)
	}
	// A near-MaxInt32 expLen behind a valid zero preamble must be rejected, not
	// read past. On 32-bit this is where an additive `elen+8` bounds check would
	// overflow negative, slip the guard, and panic the slice — so the check must
	// never add the untrusted length. Frame: klen=0, vlen=0, hasExpected=0,
	// expLen=0x7FFFFFFF, nothing behind it.
	casOverflow := []byte{0, 0 /*klen*/, 0, 0, 0, 0 /*vlen*/, 0 /*hasExpected*/, 0x7F, 0xFF, 0xFF, 0xFF /*expLen*/}
	if _, _, _, _, _, err := DecodeCASArgs(casOverflow); err != ErrShortArgs {
		t.Fatalf("DecodeCASArgs huge expLen: err = %v, want ErrShortArgs", err)
	}
	// DecodeCASResult shape.
	if _, err := DecodeCASResult([]byte{}); err != ErrShortArgs {
		t.Fatalf("DecodeCASResult empty: err = %v, want ErrShortArgs", err)
	}
	if ok, _ := DecodeCASResult([]byte{1}); !ok {
		t.Fatal("DecodeCASResult([1]) = false, want true")
	}
	if ok, _ := DecodeCASResult([]byte{0}); ok {
		t.Fatal("DecodeCASResult([0]) = true, want false")
	}

	// CAS roundtrip (expect-value).
	k, v, has, exp, ttl, err := DecodeCASArgs(EncodeCASArgs([]byte("k"), []byte("v"), true, []byte("e"), 2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(k, []byte("k")) || !bytes.Equal(v, []byte("v")) || !has || !bytes.Equal(exp, []byte("e")) || ttl != 2*time.Second {
		t.Fatalf("CAS roundtrip: k=%q v=%q has=%v exp=%q ttl=%v", k, v, has, exp, ttl)
	}
	// Expect-absent roundtrip drops the expected bytes.
	_, _, has, exp, _, err = DecodeCASArgs(EncodeCASArgs([]byte("k"), []byte("v"), false, []byte("ignored"), 0))
	if err != nil {
		t.Fatal(err)
	}
	if has || exp != nil {
		t.Fatalf("expect-absent roundtrip: has=%v exp=%q, want false/nil", has, exp)
	}
	// CAD roundtrip.
	dk, de, err := DecodeCADArgs(EncodeCADArgs([]byte("key"), []byte("expected")))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dk, []byte("key")) || !bytes.Equal(de, []byte("expected")) {
		t.Fatalf("CAD roundtrip: k=%q e=%q", dk, de)
	}
}

func TestArgsCodecRoundtrip(t *testing.T) {
	// EncodeKeyArgs ↔ DecodeKeyArgs
	k := []byte("user:42")
	args := EncodeKeyArgs(k)
	dk, err := DecodeKeyArgs(args)
	if err != nil {
		t.Fatalf("DecodeKeyArgs: %v", err)
	}
	if !bytes.Equal(dk, k) {
		t.Fatalf("DecodeKeyArgs: %q, want %q", dk, k)
	}

	// EncodePutArgs ↔ DecodePutArgs
	args = EncodePutArgs(k, []byte("v"), 5*time.Second)
	dk2, dv, dttl, err := DecodePutArgs(args)
	if err != nil {
		t.Fatalf("DecodePutArgs: %v", err)
	}
	if !bytes.Equal(dk2, k) || !bytes.Equal(dv, []byte("v")) {
		t.Fatalf("DecodePutArgs: %q,%q", dk2, dv)
	}
	if dttl != 5*time.Second {
		t.Fatalf("DecodePutArgs ttl = %v, want 5s", dttl)
	}
}

func TestBuiltinPing(t *testing.T) {
	r, tx := newTestSetup(t)

	pingH, kind, _, ok := r.Lookup("__ping__")
	if !ok {
		t.Fatal("__ping__ not registered")
	}
	if kind != OpReadOnly {
		t.Fatalf("__ping__ kind = %v, want OpReadOnly", kind)
	}
	res, err := pingH(tx, nil)
	if err != nil {
		t.Fatalf("ping: %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("ping result = %v, want empty", res)
	}
	// Tolerate non-empty args (handler must not validate to keep the cheap path cheap).
	res, err = pingH(tx, []byte("anything"))
	if err != nil || len(res) != 0 {
		t.Fatalf("ping with args: res=%v err=%v", res, err)
	}
}

func TestStdKeyExtractor(t *testing.T) {
	args := EncodePutArgs([]byte("user:42"), []byte("v"), 0)
	key, ok := StdKeyExtractor(args)
	if !ok {
		t.Fatal("StdKeyExtractor: no key")
	}
	if string(key) != "user:42" {
		t.Fatalf("key = %q, want user:42", key)
	}

	// Short args.
	if _, ok := StdKeyExtractor([]byte{0}); ok {
		t.Error("1-byte args: extractor returned key")
	}
	if _, ok := StdKeyExtractor([]byte{0, 5, 'a', 'b'}); ok {
		t.Error("truncated key: extractor returned key")
	}
}

func TestBuiltinsRegisterWithExtractor(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"get", "put", "del", "expire", "incr"} {
		_, _, ke, ok := r.Lookup(name)
		if !ok {
			t.Errorf("%q not registered", name)
			continue
		}
		if ke == nil {
			t.Errorf("%q registered without extractor", name)
		}
	}
	_, _, ke, ok := r.Lookup("__ping__")
	if !ok {
		t.Error("__ping__ not registered")
	}
	if ke != nil {
		t.Error("__ping__ should be shardless (nil extractor)")
	}
}

func TestVectorOpsViaDispatch(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	// Create
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// Insert
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 2, []float32{2, 0})); err != nil {
		t.Fatal(err)
	}
	// Search
	body, err := handleVectorSearch(tx, EncodeVectorSearchArgs("docs", 1, []float32{1, 0}))
	if err != nil {
		t.Fatal(err)
	}
	results, err := DecodeVectorSearchResults(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != 1 {
		t.Errorf("search returned %+v, want id 1", results)
	}
}

// TestVectorCreateCollectionPersistentViaDispatch proves the networked
// create-collection path honors Config.Persistent: a collection created through
// the op handler is persistent server-side, and after Flush + store reopen it
// instant-restarts with identical search results.
func TestVectorCreateCollectionPersistentViaDispatch(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.Cosine,
		Quant: vector.QuantSQ8, RescoreFactor: 2, Persistent: true,
	}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	coll, ok := vstore.Get("docs")
	if !ok || !coll.Config().Persistent {
		t.Fatalf("collection not persistent server-side (ok=%v)", ok)
	}

	for i := 1; i <= 8; i++ {
		v := []float32{float32(i), 1, float32(i % 3), 0}
		if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", uint64(i), v)); err != nil {
			t.Fatal(err)
		}
	}
	q := []float32{2, 1, 1, 0}
	beforeBody, err := handleVectorSearch(tx, EncodeVectorSearchArgs("docs", 3, q))
	if err != nil {
		t.Fatal(err)
	}
	before, _ := DecodeVectorSearchResults(beforeBody)

	if err := vstore.Flush("docs"); err != nil {
		t.Fatalf("flush persistent collection: %v", err)
	}
	if err := vstore.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen the store — instant restart of the persistent collection.
	vstore2, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	defer vstore2.Close()
	tx2 := NewTxContextWithVectors(c, vstore2)
	afterBody, err := handleVectorSearch(tx2, EncodeVectorSearchArgs("docs", 3, q))
	if err != nil {
		t.Fatal(err)
	}
	after, _ := DecodeVectorSearchResults(afterBody)

	if len(before) == 0 || len(before) != len(after) {
		t.Fatalf("result count changed across restart: %d -> %d", len(before), len(after))
	}
	for i := range before {
		if before[i].ID != after[i].ID {
			t.Errorf("result %d id changed across restart: %d -> %d", i, before[i].ID, after[i].ID)
		}
	}
}

// TestHandleVectorHybridLanes verifies that handleVectorHybridLanes returns
// the same dense and sparse lanes as Collection.HybridLanes directly.
func TestHandleVectorHybridLanes(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	// Create a collection with dim=4 so we can exercise both dense and sparse.
	cfg := vector.Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}

	// Insert docs 1-5 near dense origin with weak shared sparse term.
	for i := uint64(1); i <= 5; i++ {
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		sv := vector.SparseVector{Indices: []uint32{1}, Values: []float32{0.1}}
		args := EncodeVectorUpsertArgs("docs", i, v, "", 0, nil, sv)
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}
	// Insert doc 100 far in dense space but strong sparse term 42.
	sv100 := vector.SparseVector{Indices: []uint32{42}, Values: []float32{10.0}}
	args100 := EncodeVectorUpsertArgs("docs", 100, []float32{9, 9, 9, 9}, "", 0, nil, sv100)
	if _, err := handleVectorUpsert(tx, args100); err != nil {
		t.Fatalf("upsert 100: %v", err)
	}

	denseQuery := []float32{0, 0, 0, 0}
	sparseQuery := vector.SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	opts := vector.HybridOpts{DenseK: 10, SparseK: 10}
	encArgs := EncodeHybridSearchArgs("docs", denseQuery, 5, sparseQuery, opts)

	body, err := handleVectorHybridLanes(tx, encArgs)
	if err != nil {
		t.Fatal(err)
	}
	gotD, gotS, err := DecodeHybridLanesResult(body)
	if err != nil {
		t.Fatal(err)
	}

	// Compare against Collection.HybridLanes directly.
	coll, ok := tx.vectors.Acquire("docs")
	if !ok {
		t.Fatal("acquire docs")
	}
	defer coll.Release()
	wantD, wantS, err := coll.HybridLanes(denseQuery, sparseQuery, 5, opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(gotD) != len(wantD) || len(gotS) != len(wantS) {
		t.Fatalf("lanes lengths got (%d,%d) want (%d,%d)", len(gotD), len(gotS), len(wantD), len(wantS))
	}
	for i := range wantD {
		if gotD[i].ID != wantD[i].ID || gotD[i].Distance != wantD[i].Distance {
			t.Fatalf("dense lane mismatch at %d: got %+v want %+v", i, gotD[i], wantD[i])
		}
	}
	for i := range wantS {
		if gotS[i].ID != wantS[i].ID || gotS[i].Score != wantS[i].Score {
			t.Fatalf("sparse lane mismatch at %d: got %+v want %+v", i, gotS[i], wantS[i])
		}
	}
}
