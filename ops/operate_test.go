// SPDX-License-Identifier: Apache-2.0

package ops

// Handler-level tests for "operate" (design doc §3.5): registration, TTL
// modes, all-or-nothing behavior against a real TxContext/cache, and
// determinism across replicas. The op's own apply semantics are covered
// exhaustively by operate_semantics_test.go (against applyRecordBytes
// directly and, here, against the handler through handlerApply) and by the
// schema-engine equivalence property in operate_schema_engine_test.go; this
// file does not re-derive that coverage.

import (
	"bytes"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// callOperate encodes a, runs it through handleOperate under the given apply
// stamp, and decodes the result frame. stampMs == 0 means unstamped, matching
// applyStamp's contract elsewhere in this package.
func callOperate(t *testing.T, tx *TxContext, stampMs int64, a *wire.OperateArgs) *wire.OperateResult {
	t.Helper()
	tx.SetApplyStamp(uint64(stampMs), stampMs != 0) //nolint:gosec // test stamps are small positive millis
	defer tx.SetApplyStamp(0, false)
	b, err := wire.EncodeOperateArgs(a)
	if err != nil {
		t.Fatalf("EncodeOperateArgs: %v", err)
	}
	out, err := handleOperate(tx, b)
	if err != nil {
		t.Fatalf("handleOperate: %v", err)
	}
	res, err := wire.DecodeOperateResult(out)
	if err != nil {
		t.Fatalf("DecodeOperateResult: %v", err)
	}
	return res
}

// readAt reads key's value and expiry judged against the EXPLICIT clock
// stampMs, exactly as the replicated apply path would. Every TTL-sensitive
// assertion in this file goes through readAt rather than the wall-clock
// Get/GetWithExpiry: this file's stamps are small, arbitrary millis (matching
// the operate semantics suite's convention), which the real wall clock would
// judge as already expired.
func readAt(t *testing.T, tx *TxContext, stampMs int64, key []byte) ([]byte, uint64) {
	t.Helper()
	tx.SetApplyStamp(uint64(stampMs), true) //nolint:gosec // test stamps are small positive millis
	defer tx.SetApplyStamp(0, false)
	v, expiryMs, err := tx.GetWithExpiry(key)
	if err != nil {
		t.Fatalf("GetWithExpiry at stamp %d: %v", stampMs, err)
	}
	return v, expiryMs
}

// addField0 is the simplest possible schema-mode call: bump sessionSchema's
// first counter field by 1. Several tests below only need SOME call that
// creates and then mutates a schema record; they share this rather than
// each inventing their own.
func addField0(key []byte) *wire.OperateArgs {
	return &wire.OperateArgs{Key: key, Create: wire.OperateCreateSchema, Schema: sessionSchema().Encode(),
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}}}
}

func TestOperatePersistsAcrossCalls(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("persist-key")
	a := addField0(key)

	for i := 0; i < 3; i++ {
		res := callOperate(t, tx, 0, a)
		if res.Status != wire.OperateStatusOK {
			t.Fatalf("call %d: status = %v, want OperateStatusOK", i, res.Status)
		}
	}

	v, err := tx.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	rec, err := wire.DecodeRecord(v)
	if err != nil {
		t.Fatalf("DecodeRecord: %v", err)
	}
	if got := rec.Fields[0].Cell.U; got != 3 {
		t.Fatalf("field 0 = %d, want 3 (three ADD calls persisted)", got)
	}
}

// handlerApplyTx is the TxContext handlerApply drives. It is package-level
// (mirroring skipDynamic's pattern) because the applier function type it
// implements carries no state of its own: TestOperateHandlerSemantics sets
// it before calling runSemantics(t, handlerApply) and clears it via
// t.Cleanup.
var handlerApplyTx *TxContext

// handlerApply adapts handleOperate to the semantics suite's applier type
// (operate_semantics_test.go), the handler-level counterpart of Task 8's
// bytesApply: it seeds handlerApplyTx's store with rec's encoded bytes (or
// clears the key when rec is nil), stamps the apply clock exactly as a
// replicated apply would, drives the call through the real handler, and
// decodes whatever ends up stored back into a tree. Running the whole suite
// through it proves handleOperate's TTL/key-lookup wrapper does not change
// the semantics applyRecordBytes already establishes.
func handlerApply(rec *wire.Record, a *wire.OperateArgs, stampMs int64) (*wire.Record, *wire.OperateResult, error) {
	tx := handlerApplyTx
	if _, err := tx.Del(a.Key); err != nil {
		return nil, nil, err
	}
	if rec != nil {
		if err := tx.Put(a.Key, rec.Encode(), 0); err != nil {
			return nil, nil, err
		}
	}

	tx.SetApplyStamp(uint64(stampMs), stampMs != 0) //nolint:gosec // suite stamps are small positive millis
	argBytes, err := wire.EncodeOperateArgs(a)
	if err != nil {
		tx.SetApplyStamp(0, false)
		return nil, nil, err
	}
	out, err := handleOperate(tx, argBytes)
	tx.SetApplyStamp(0, false)
	if err != nil {
		if errors.Is(err, cache.ErrNotFound) {
			// handleOperate maps errOperateAbsent (create=NONE against an
			// absent key) to the store's own not-found error at the handler
			// boundary (see its doc comment; TestOperateCreateNoneOnAbsent
			// pins that mapping directly). The semantics suite compares
			// against the pre-mapped sentinel, so translate back here. This
			// is safe because it is the ONLY way handleOperate itself can
			// return cache.ErrNotFound: a genuine absent-on-read never
			// reaches the caller as an error — GetWithExpiry's ErrNotFound
			// is caught and turned into the `absent` bool before any of
			// this handler's own error returns.
			return nil, nil, errOperateAbsent
		}
		return nil, nil, err
	}
	res, err := wire.DecodeOperateResult(out)
	if err != nil {
		return nil, nil, err
	}

	v, _, err := tx.GetWithExpiry(a.Key)
	switch {
	case errors.Is(err, cache.ErrNotFound):
		return nil, res, nil
	case err != nil:
		return nil, nil, err
	}
	got, derr := wire.DecodeRecord(v)
	if derr != nil {
		return nil, nil, fmt.Errorf("handler produced undecodable bytes %x: %w", v, derr)
	}
	return got, res, nil
}

// TestOperateHandlerSemantics runs the whole operate semantics suite twice:
// once against applyRecordBytes directly (bytesApply, Task 8's adapter) and
// once through the real handler (handlerApply, above) on a live TxContext.
// Both modes run in both lanes: every rule in the suite must hold through the
// handler exactly as it does against the apply core.
func TestOperateHandlerSemantics(t *testing.T) {
	t.Run("bytes", func(t *testing.T) {
		runSemantics(t, bytesApply)
	})

	t.Run("handler", func(t *testing.T) {
		_, tx := newTestSetup(t)
		handlerApplyTx = tx
		t.Cleanup(func() { handlerApplyTx = nil })
		runSemantics(t, handlerApply)
	})
}

func TestOperateTTLModes(t *testing.T) {
	t.Run("keep", func(t *testing.T) {
		_, tx := newTestSetup(t)
		key := []byte("keep-key")
		a := addField0(key)
		a.TTLMode = wire.OperateTTLKeep

		callOperate(t, tx, 1_000, a)
		if _, expiryMs := readAt(t, tx, 1_000, key); expiryMs != 0 {
			t.Fatalf("KEEP on create set an expiry: %d, want 0 (none)", expiryMs)
		}

		// Give the record an expiry out of band (as if an earlier SET call
		// had put one there), then confirm KEEP leaves it alone.
		v, _, err := tx.GetWithExpiry(key)
		if err != nil {
			t.Fatalf("GetWithExpiry: %v", err)
		}
		if err := tx.PutAbs(key, v, 5_000); err != nil {
			t.Fatalf("PutAbs: %v", err)
		}

		callOperate(t, tx, 1_000, a)
		if _, expiryMs := readAt(t, tx, 1_000, key); expiryMs != 5_000 {
			t.Fatalf("KEEP on an existing key changed the expiry: got %d, want 5000", expiryMs)
		}
	})

	t.Run("set", func(t *testing.T) {
		_, tx := newTestSetup(t)
		key := []byte("set-key")
		a := addField0(key)
		a.TTLMode = wire.OperateTTLSet
		a.TTL = 10 * time.Second

		callOperate(t, tx, 1_000_000, a)
		if _, expiryMs := readAt(t, tx, 1_000_000, key); expiryMs != 1_010_000 {
			t.Fatalf("SET on create expiry = %d, want 1010000 (stamp+ttl)", expiryMs)
		}

		callOperate(t, tx, 1_005_000, a)
		if _, expiryMs := readAt(t, tx, 1_005_000, key); expiryMs != 1_015_000 {
			t.Fatalf("SET on an existing key did not refresh: expiry = %d, want 1015000", expiryMs)
		}

		// SET with TTL == 0 means "refresh to no expiry" (design doc §3.5):
		// it must clear the deadline on an existing key, not leave the old
		// one in place or misinterpret zero as "unspecified".
		clear := addField0(key)
		clear.TTLMode = wire.OperateTTLSet
		clear.TTL = 0
		callOperate(t, tx, 1_010_000, clear)
		if _, expiryMs := readAt(t, tx, 1_010_000, key); expiryMs != 0 {
			t.Fatalf("SET with TTL=0 did not clear the expiry: got %d, want 0 (none)", expiryMs)
		}
	})

	t.Run("create_only", func(t *testing.T) {
		_, tx := newTestSetup(t)
		key := []byte("create-only-key")
		a := addField0(key)
		a.TTLMode = wire.OperateTTLCreateOnly
		a.TTL = 10 * time.Second

		callOperate(t, tx, 1_000_000, a)
		if _, expiryMs := readAt(t, tx, 1_000_000, key); expiryMs != 1_010_000 {
			t.Fatalf("CREATE_ONLY on create expiry = %d, want 1010000 (stamp+ttl)", expiryMs)
		}

		// A later stamp must not move the deadline once the key exists.
		callOperate(t, tx, 1_005_000, a)
		if _, expiryMs := readAt(t, tx, 1_005_000, key); expiryMs != 1_010_000 {
			t.Fatalf("CREATE_ONLY changed the expiry on an existing key: got %d, want unchanged 1010000", expiryMs)
		}
	})
}

// TestOperateDeleteOnEmptyDynamic pins DEL of a dynamic record's last field
// (design doc §2.5) through the real handler: the record itself disappears,
// and the key is gone from the store rather than holding an empty record.
func TestOperateDeleteOnEmptyDynamic(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("dyn-key")
	set := &wire.OperateArgs{Key: key, Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU8, Path: namePath("a"), A: 1}}}
	callOperate(t, tx, 0, set)

	del := &wire.OperateArgs{Key: key, Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpDEL, Path: namePath("a")}}}
	callOperate(t, tx, 0, del)

	if _, err := tx.Get(key); !errors.Is(err, cache.ErrNotFound) {
		t.Fatalf("Get after deleting the last dynamic field: err = %v, want cache.ErrNotFound", err)
	}
}

func TestOperateAllOrNothing(t *testing.T) {
	_, tx := newTestSetup(t)
	s := sessionSchema()
	key := []byte("aon-key")
	callOperate(t, tx, 1_000, addField0(key))

	// Give the record an expiry too, so a would-be TTL change would also be
	// visible.
	v, _, err := tx.GetWithExpiry(key)
	if err != nil {
		t.Fatalf("GetWithExpiry: %v", err)
	}
	if err := tx.PutAbs(key, v, 50_000); err != nil {
		t.Fatalf("PutAbs: %v", err)
	}
	before, beforeExpiry := readAt(t, tx, 1_000, key)

	t.Run("bad opcode", func(t *testing.T) {
		bad := &wire.OperateArgs{Key: key, Create: wire.OperateCreateSchema, Schema: s.Encode(),
			Ops: []wire.OperateOp{
				{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1},
				{Opcode: 250, Path: fieldPath(0)}, // not a defined opcode
			}}
		b, err := wire.EncodeOperateArgs(bad)
		if err != nil {
			t.Fatalf("EncodeOperateArgs: %v", err)
		}
		tx.SetApplyStamp(1_000, true)
		_, err = handleOperate(tx, b)
		tx.SetApplyStamp(0, false)
		if err == nil {
			t.Fatal("bad opcode after a successful op: want an error, got nil")
		}
		got, gotExpiry := readAt(t, tx, 1_000, key)
		if !bytes.Equal(got, before) || gotExpiry != beforeExpiry {
			t.Fatalf("record changed after a failing op list: bytes %x (want %x), expiry %d (want %d)",
				got, before, gotExpiry, beforeExpiry)
		}
	})

	t.Run("check false", func(t *testing.T) {
		chk := &wire.OperateArgs{Key: key, Create: wire.OperateCreateSchema, Schema: s.Encode(),
			Ops: []wire.OperateOp{
				{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1},
				{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpEQ, Path: fieldPath(0), A: 999},
			}}
		res := callOperate(t, tx, 1_000, chk)
		if res.Status != wire.OperateStatusCheckFailed {
			t.Fatalf("status = %v, want OperateStatusCheckFailed", res.Status)
		}
		got, gotExpiry := readAt(t, tx, 1_000, key)
		if !bytes.Equal(got, before) || gotExpiry != beforeExpiry {
			t.Fatalf("record changed after CHECK_FAILED: bytes %x (want %x), expiry %d (want %d)",
				got, before, gotExpiry, beforeExpiry)
		}
	})
}

func TestOperateCreateNoneOnAbsent(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("absent-key")
	a := &wire.OperateArgs{Key: key, Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU8, Path: fieldPath(0), A: 1}}}
	b, err := wire.EncodeOperateArgs(a)
	if err != nil {
		t.Fatalf("EncodeOperateArgs: %v", err)
	}
	if _, err := handleOperate(tx, b); !errors.Is(err, cache.ErrNotFound) {
		t.Fatalf("err = %v, want cache.ErrNotFound", err)
	}
	if _, err := tx.Get(key); !errors.Is(err, cache.ErrNotFound) {
		t.Fatalf("Get after a create=NONE call against an absent key: err = %v, want cache.ErrNotFound", err)
	}
}

func TestOperateRejectsPlainPutValue(t *testing.T) {
	_, tx := newTestSetup(t)
	key := []byte("plain-key")
	plain := []byte{0, 1, 2}
	if err := tx.Put(key, plain, 0); err != nil {
		t.Fatalf("Put: %v", err)
	}

	a := &wire.OperateArgs{Key: key, Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU8, Path: fieldPath(0), A: 1}}}
	b, err := wire.EncodeOperateArgs(a)
	if err != nil {
		t.Fatalf("EncodeOperateArgs: %v", err)
	}
	if _, err := handleOperate(tx, b); !errors.Is(err, wire.ErrOperateRecord) {
		t.Fatalf("err = %v, want wire.ErrOperateRecord", err)
	}
	got, err := tx.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, plain) {
		t.Fatalf("value changed: got %x, want %x", got, plain)
	}
}

func TestOperateRecordTooLargeRejected(t *testing.T) {
	old := maxOperateRecordBytes
	maxOperateRecordBytes = 64
	t.Cleanup(func() { maxOperateRecordBytes = old })

	_, tx := newTestSetup(t)
	s := floatSchema()
	key := []byte("too-big-key")
	a := &wire.OperateArgs{Key: key, Create: wire.OperateCreateSchema, Schema: s.Encode(),
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes, Path: fieldPath(2), Bytes: make([]byte, 100)}}}
	b, err := wire.EncodeOperateArgs(a)
	if err != nil {
		t.Fatalf("EncodeOperateArgs: %v", err)
	}
	if _, err := handleOperate(tx, b); !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("err = %v, want wire.ErrOperateCap", err)
	}
	if _, err := tx.Get(key); !errors.Is(err, cache.ErrNotFound) {
		t.Fatalf("Get after a cap-rejected create: err = %v, want cache.ErrNotFound (key stays absent)", err)
	}
}

// TestOperateReplicaReapplyMatches drives two independent TxContexts (each
// its own cache) through the same STAMP-and-eviction call sequence with
// identical stamps, and requires byte-identical stored records and
// byte-identical expiries after every call — the property that makes
// operate safe to replay on every Raft follower.
func TestOperateReplicaReapplyMatches(t *testing.T) {
	_, tx1 := newTestSetup(t)
	_, tx2 := newTestSetup(t)

	s := sessionSchema()
	s.Fields[3].Table.Cap = 3
	key := []byte("replica-key")
	mk := func(k uint64) *wire.OperateArgs {
		a := schemaArgs(s,
			op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1),
			op(wire.OperateOpSTAMP, opFromSchema, colPath(3, keyU64(k), 2), 0).withAux(wire.OperateStampS))
		a.Key = key
		a.TTLMode = wire.OperateTTLSet
		a.TTL = 60 * time.Second
		return a
	}

	steps := []struct {
		k  uint64
		st int64
	}{{5, 1_000_000}, {9, 1_000_000}, {1, 1_001_000}, {7, 1_002_000}}
	for _, step := range steps {
		r1 := callOperate(t, tx1, step.st, mk(step.k))
		r2 := callOperate(t, tx2, step.st, mk(step.k))
		if r1.Status != r2.Status {
			t.Fatalf("k=%d: status diverged: %v vs %v", step.k, r1.Status, r2.Status)
		}
		v1, e1 := readAt(t, tx1, step.st, key)
		v2, e2 := readAt(t, tx2, step.st, key)
		if !bytes.Equal(v1, v2) {
			t.Fatalf("k=%d: stored bytes diverged:\n  %x\nvs\n  %x", step.k, v1, v2)
		}
		if e1 != e2 {
			t.Fatalf("k=%d: expiry diverged: %d vs %d", step.k, e1, e2)
		}
	}
}

// TestOperateUnstampedIsDeterministic mirrors TestOperateReplicaReapplyMatches
// for the unstamped path (stampMs == 0, LRU-by-stamp degrades to the
// oracle-defined tie-break): two replicas fed the same call sequence with no
// stamp at all must still land on identical bytes.
func TestOperateUnstampedIsDeterministic(t *testing.T) {
	_, tx1 := newTestSetup(t)
	_, tx2 := newTestSetup(t)

	s := sessionSchema()
	s.Fields[3].Table.Cap = 2
	s.Fields[3].Table.Policy = wire.OperatePolicyMinKey
	key := []byte("unstamped-key")
	mk := func(k uint64) *wire.OperateArgs {
		a := schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1))
		a.Key = key
		return a
	}

	for _, k := range []uint64{4, 256, 5} {
		callOperate(t, tx1, 0, mk(k))
		callOperate(t, tx2, 0, mk(k))
	}
	v1, err := tx1.Get(key)
	if err != nil {
		t.Fatalf("Get tx1: %v", err)
	}
	v2, err := tx2.Get(key)
	if err != nil {
		t.Fatalf("Get tx2: %v", err)
	}
	if !bytes.Equal(v1, v2) {
		t.Fatalf("unstamped apply diverged:\n  %x\nvs\n  %x", v1, v2)
	}
}

func TestOperateRegistered(t *testing.T) {
	r, _ := newTestSetup(t)
	h, kind, _, ok := r.Lookup("operate")
	if !ok {
		t.Fatal("\"operate\" not registered")
	}
	if h == nil {
		t.Fatal("\"operate\" has a nil handler")
	}
	if kind != OpReadWrite {
		t.Fatalf("\"operate\" kind = %v, want OpReadWrite", kind)
	}

	found := false
	for _, o := range wire.BuiltinOps {
		if o.Name == "operate" {
			found = true
			if o.Kind != OpReadWrite {
				t.Fatalf("wire.BuiltinOps[\"operate\"].Kind = %v, want OpReadWrite", o.Kind)
			}
			break
		}
	}
	if !found {
		t.Fatal("wire.BuiltinOps does not contain \"operate\"")
	}
}

// TestOperateOversizedStoredValueNotCopied covers the §2.7 ceiling on the
// way IN. The value under an operate key is whatever the store holds — a
// plain `put` can leave anything there — so it can be far larger than a
// valid operate record. copyRecord must reject it BEFORE it copies: without
// the check, one call would allocate an arbitrary multiple of the record cap
// just to discover the bytes are not a record.
func TestOperateOversizedStoredValueNotCopied(t *testing.T) {
	old := maxOperateRecordBytes
	maxOperateRecordBytes = 256
	t.Cleanup(func() { maxOperateRecordBytes = old })

	// A well-formed dynamic record one byte over the lowered cap: it is only
	// the SIZE that must reject it, not its contents.
	rec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{
		{Name: "a", Cell: wire.Cell{Type: wire.OperateTypeBytes, B: make([]byte, 300)}}}}
	stored := rec.Encode()
	if len(stored) <= maxOperateRecordBytes {
		t.Fatalf("fixture is %d bytes, not over the %d-byte cap", len(stored), maxOperateRecordBytes)
	}
	if _, derr := wire.DecodeRecord(stored); derr != nil {
		t.Fatalf("fixture is not a well-formed record: %v", derr)
	}

	a := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("n"), A: 1}}}
	out, deleted, res, err := applyRecordBytes(stored, a, 0)
	if !errors.Is(err, wire.ErrOperateRecord) {
		t.Fatalf("err = %v, want wire.ErrOperateRecord", err)
	}
	if out != nil || deleted || res != nil {
		t.Fatalf("the rejected call returned out=%x deleted=%v res=%+v", out, deleted, res)
	}

	// And it costs no copy: the whole point of checking before copyRecord's
	// make. The pooled engines are already warm after the call above, so the
	// only allocation this could make is the copy itself.
	if n := testing.AllocsPerRun(50, func() { _, _, _, _ = applyRecordBytes(stored, a, 0) }); n != 0 {
		t.Fatalf("the rejected call allocated %v times; the size check must precede the copy", n)
	}

	// At exactly the cap the same record opens: the rejection above is the
	// ceiling, not the record. A call that changes nothing is used here, so
	// the §2.7 check on the FINISHED record cannot be what passes or fails.
	maxOperateRecordBytes = len(stored)
	noop := &wire.OperateArgs{Create: wire.OperateCreateDynamic}
	if _, _, _, err := applyRecordBytes(stored, noop, 0); err != nil {
		t.Fatalf("a stored record exactly at the cap must open: %v", err)
	}
	// With headroom for the new field, the mutating call goes through too.
	maxOperateRecordBytes = len(stored) + 64
	if _, _, _, err := applyRecordBytes(stored, a, 0); err != nil {
		t.Fatalf("a stored record under the cap must apply: %v", err)
	}
}
