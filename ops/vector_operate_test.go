// SPDX-License-Identifier: Apache-2.0

package ops

// Handler-level tests for vector_operate / vector_named_operate /
// vector_mv_operate: the operate op applied to a record held inside a POINT's
// payload rather than under a KV key.
//
// The op's apply semantics are not re-derived here — applyRecordBytes is the
// same pure function handleOperate drives, and operate_semantics_test.go covers
// it exhaustively. What these tests pin is everything the vector wrapper adds:
// the not-found FLAG for a missing point, the create=NONE error, the non-record
// payload key error, the three-outcome mutator mapping (store / delete /
// unchanged), CAS, and that the leader stamp is the only clock.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// vecOperateHandler is the shape the three family handlers share, so one table
// can drive dense, named and multi-vector through the same case.
type vecOperateHandler func(*TxContext, []byte) ([]byte, error)

// callVecOperate encodes a vector_operate call, runs it through h under the
// given apply stamp, and decodes the reply frame. stampMs == 0 means unstamped
// (the single-node path), matching applyStamp's contract everywhere else in this
// package.
func callVecOperate(t *testing.T, tx *TxContext, h vecOperateHandler, stampMs int64,
	col string, id uint64, pk string, a *wire.OperateArgs, expected uint64, hasExpected bool,
) (bool, *wire.OperateResult, error) {
	t.Helper()
	args, err := EncodeVectorOperateArgs(col, id, pk, a, expected, hasExpected)
	if err != nil {
		t.Fatalf("EncodeVectorOperateArgs: %v", err)
	}
	tx.SetApplyStamp(uint64(stampMs), stampMs != 0) //nolint:gosec // test stamps are small positive millis
	defer tx.SetApplyStamp(0, false)
	body, herr := h(tx, args)
	if herr != nil {
		return false, nil, herr
	}
	found, res, _, derr := DecodeVectorOperateResult(body)
	if derr != nil {
		t.Fatalf("DecodeVectorOperateResult: %v", derr)
	}
	return found, res, nil
}

// mustVecOperate is callVecOperate for the calls that must not error.
func mustVecOperate(t *testing.T, tx *TxContext, h vecOperateHandler, stampMs int64,
	col string, id uint64, pk string, a *wire.OperateArgs,
) (bool, *wire.OperateResult) {
	t.Helper()
	found, res, err := callVecOperate(t, tx, h, stampMs, col, id, pk, a, 0, false)
	if err != nil {
		t.Fatalf("handler: %v", err)
	}
	return found, res
}

// setDyn is the simplest create-and-write call: SET a by-name U64 field on a
// dynamic-mode record, creating the record if the payload key is empty.
func setDyn(field string, v int64) *wire.OperateArgs {
	return &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpSET, Type: wire.OperateTypeU64, Path: namePath(field), A: v}}}
}

// addDynRet bumps a by-name U64 field and returns its value.
func addDynRet(field string, delta int64) *wire.OperateArgs {
	return &wire.OperateArgs{Create: wire.OperateCreateDynamic,
		Ops:  []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath(field), A: delta}},
		Rets: []wire.OperateRet{{Mode: wire.OperateRetValue, Path: namePath(field)}}}
}

// vecSessionSchema mirrors vector/record_filter_test.go's canonical session
// fixture (the constraints name it the one record shape these phases test
// against). It is rebuilt here because package vector's test helpers are not
// importable from ops — the SHAPE is what must match, not the code:
//
//	#0 rc U8 #1 bc U8 #2 hist U32 #3 bal I32 #4 tag BYTES
//	#5 b TABLE(key U64; hi U32, lo U32)
func vecSessionSchema(t *testing.T) *wire.Schema {
	t.Helper()
	s := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "rc", Type: wire.OperateTypeU8},
		{Name: "bc", Type: wire.OperateTypeU8},
		{Name: "hist", Type: wire.OperateTypeU32},
		{Name: "bal", Type: wire.OperateTypeI32},
		{Name: "tag", Type: wire.OperateTypeBytes},
		{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{
			KeyType: wire.OperateTypeU64,
			Cols: []wire.ColumnDef{
				{Name: "hi", Type: wire.OperateTypeU32},
				{Name: "lo", Type: wire.OperateTypeU32},
			},
		}},
	}}
	if err := s.Validate(); err != nil {
		t.Fatalf("session schema invalid: %v", err)
	}
	return s
}

// vecSessionRecordBytes is vecSessionSchema's record with rc = 7 and two rows in
// table b, matching vector/record_filter_test.go's sessionRecordBytes.
func vecSessionRecordBytes(t *testing.T) []byte {
	t.Helper()
	leKey := func(v uint64) []byte { return binary.LittleEndian.AppendUint64(nil, v) }
	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: vecSessionSchema(t), Fields: []wire.Field{
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 7}},
		{Cell: wire.Cell{Type: wire.OperateTypeU8, U: 3}},
		{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1234}},
		{Cell: wire.Cell{Type: wire.OperateTypeI32, U: su64(-5)}},
		{Cell: wire.Cell{Type: wire.OperateTypeBytes, B: []byte("de")}},
		{Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{Rows: []wire.Row{
			{Key: leKey(42), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 500}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 1}},
			}},
			{Key: leKey(99), Cols: []wire.Col{
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 7}},
				{Cell: wire.Cell{Type: wire.OperateTypeU32, U: 2}},
			}},
		}}},
	}}
	enc := rec.Encode()
	if enc == nil {
		t.Fatal("session record failed to encode")
	}
	return enc
}

// newDenseOperateTx creates a dense collection "docs" holding point 1 with the
// given payload, and returns the TxContext over it.
func newDenseOperateTx(t *testing.T, meta vector.Metadata) *TxContext {
	t.Helper()
	tx := newVecTx(t)
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgsExt(
		"docs", 1, []float32{1, 0}, 0, meta, vector.SparseVector{})); err != nil {
		t.Fatalf("insert: %v", err)
	}
	return tx
}

// densePayload reads point 1's payload and version back through the real get op.
func densePayload(t *testing.T, tx *TxContext, id uint64) (vector.Metadata, uint64) {
	t.Helper()
	body, err := handleVectorGet(tx, EncodeVectorGetArgs("docs", id, GetFlagsBoth))
	if err != nil {
		t.Fatalf("vector_get: %v", err)
	}
	found, _, meta, _, _, version, err := DecodeVectorGetResultV(body)
	if err != nil {
		t.Fatalf("DecodeVectorGetResultV: %v", err)
	}
	if !found {
		t.Fatalf("point %d not found", id)
	}
	return meta, version
}

// cellU decodes a returned tagged cell and reports its unsigned value.
func cellU(t *testing.T, b []byte) uint64 {
	t.Helper()
	c, _, err := wire.DecodeTaggedCell(b)
	if err != nil {
		t.Fatalf("DecodeTaggedCell(%x): %v", b, err)
	}
	return c.U
}

func TestVectorOperateCreatesAndIncrements(t *testing.T) {
	tx := newDenseOperateTx(t, vector.Metadata{"lang": vector.NewString("en")})

	found, res := mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", setDyn("rc", 1))
	if !found {
		t.Fatal("call 1: found = false, want true (the point exists)")
	}
	if res == nil || res.Status != wire.OperateStatusOK {
		t.Fatalf("call 1: res = %+v, want OperateStatusOK", res)
	}

	found, res = mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", addDynRet("rc", 1))
	if !found || res == nil || res.Status != wire.OperateStatusOK {
		t.Fatalf("call 2: found=%v res=%+v", found, res)
	}
	if len(res.Values) != 1 {
		t.Fatalf("call 2: %d return values, want 1", len(res.Values))
	}
	if got := cellU(t, res.Values[0]); got != 2 {
		t.Fatalf("rc = %d, want 2 (created at 1, then incremented)", got)
	}

	meta, _ := densePayload(t, tx, 1)
	v, ok := meta["session"]
	if !ok {
		t.Fatalf("payload = %+v, want a session key", meta)
	}
	if v.Kind != vector.ValueRecord {
		t.Fatalf("session kind = %v, want ValueRecord", v.Kind)
	}
	if meta["lang"].Str != "en" {
		t.Fatalf("payload = %+v, want the pre-existing lang key untouched", meta)
	}
}

func TestVectorOperateAgainstSchemaRecord(t *testing.T) {
	tx := newDenseOperateTx(t, vector.Metadata{"session": vector.NewRecord(vecSessionRecordBytes(t))})
	s := vecSessionSchema(t)

	a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(),
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1}},
		Rets: []wire.OperateRet{
			{Mode: wire.OperateRetValue, Path: fieldPath(0)},
			{Mode: wire.OperateRetCount, Path: fieldPath(5)},
		}}
	found, res := mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", a)
	if !found || res == nil || res.Status != wire.OperateStatusOK {
		t.Fatalf("found=%v res=%+v", found, res)
	}
	if got := cellU(t, res.Values[0]); got != 8 {
		t.Fatalf("rc = %d, want 8 (7 + 1)", got)
	}
	if got := cellU(t, res.Values[1]); got != 2 {
		t.Fatalf("COUNT b = %d, want 2 (the table's rows survive a scalar ADD)", got)
	}
}

func TestVectorOperateMissingPointIsAFlag(t *testing.T) {
	tx := newDenseOperateTx(t, nil)

	found, res, err := callVecOperate(t, tx, handleVectorOperate, 0, "docs", 999, "session", setDyn("rc", 1), 0, false)
	if err != nil {
		t.Fatalf("missing point: err = %v, want nil (a not-found FLAG, not an op error)", err)
	}
	if found {
		t.Fatal("missing point: found = true, want false")
	}
	if res != nil {
		t.Fatalf("missing point: res = %+v, want nil (the mutator never ran)", res)
	}
}

func TestVectorOperateCreateNoneAgainstAbsentRecord(t *testing.T) {
	tx := newDenseOperateTx(t, vector.Metadata{"lang": vector.NewString("en")})

	a := &wire.OperateArgs{Create: wire.OperateCreateNone, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("rc"), A: 1}}}
	_, _, err := callVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", a, 0, false)
	if !errors.Is(err, ErrVectorRecordAbsent) {
		t.Fatalf("err = %v, want ErrVectorRecordAbsent", err)
	}

	meta, _ := densePayload(t, tx, 1)
	if len(meta) != 1 || meta["lang"].Str != "en" {
		t.Fatalf("payload = %+v, want the point's other keys untouched", meta)
	}
}

func TestVectorOperateNonRecordKeyErrors(t *testing.T) {
	tx := newDenseOperateTx(t, vector.Metadata{"country": vector.NewString("de")})

	_, _, err := callVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "country", setDyn("rc", 1), 0, false)
	if !errors.Is(err, vector.ErrPayloadKeyNotRecord) {
		t.Fatalf("err = %v, want vector.ErrPayloadKeyNotRecord", err)
	}
	meta, _ := densePayload(t, tx, 1)
	if meta["country"].Str != "de" {
		t.Fatalf("payload = %+v, want the string value preserved, not replaced by a record", meta)
	}
}

func TestVectorOperateCheckFailedIsANoOp(t *testing.T) {
	tx := newDenseOperateTx(t, nil)
	mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", setDyn("rc", 1))
	before, versionBefore := densePayload(t, tx, 1)
	recBefore := append([]byte(nil), before["session"].Rec...)

	// Op 0 succeeds, op 1's CHECK fails: the whole list aborts and NOTHING is
	// written, so op 0's increment must not survive either.
	a := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("rc"), A: 1},
		{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpEQ, Path: namePath("rc"), A: 99},
	}}
	found, res := mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", a)
	if !found {
		t.Fatal("found = false, want true (the point exists; only the CHECK failed)")
	}
	if res == nil || res.Status != wire.OperateStatusCheckFailed {
		t.Fatalf("res = %+v, want OperateStatusCheckFailed", res)
	}
	if res.FailedOp != 1 {
		t.Fatalf("FailedOp = %d, want 1 (the CHECK's index in the op list)", res.FailedOp)
	}

	after, versionAfter := densePayload(t, tx, 1)
	if !bytes.Equal(recBefore, after["session"].Rec) {
		t.Fatalf("record changed on a failed CHECK: %x -> %x", recBefore, after["session"].Rec)
	}
	if versionAfter != versionBefore {
		t.Fatalf("version = %d, want %d (a failed CHECK bumps nothing)", versionAfter, versionBefore)
	}
}

func TestVectorOperateDeletesTheRecord(t *testing.T) {
	tx := newDenseOperateTx(t, vector.Metadata{"session": vector.NewRecord(vecSessionRecordBytes(t))})

	// The record path is indexed before the delete: rc = 7 is findable.
	hits := searchRecordPath(t, tx, 7)
	if len(hits) != 1 || hits[0] != 1 {
		t.Fatalf("filter session/rc == 7 before the delete = %v, want [1]", hits)
	}

	a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: vecSessionSchema(t).Encode(),
		Ops: []wire.OperateOp{{Opcode: wire.OperateOpDEL, Path: recPath()}}}
	found, res := mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", a)
	if !found || res == nil || res.Status != wire.OperateStatusOK {
		t.Fatalf("found=%v res=%+v", found, res)
	}

	meta, _ := densePayload(t, tx, 1)
	if _, ok := meta["session"]; ok {
		t.Fatalf("payload = %+v, want the session key gone", meta)
	}
	if hits := searchRecordPath(t, tx, 7); len(hits) != 0 {
		t.Fatalf("filter session/rc == 7 after the delete = %v, want no hits (the posting is gone)", hits)
	}
}

// searchRecordPath runs a filtered search over "docs" for session/rc == rc and
// returns the matching ids — the record posting the payload index maintains.
func searchRecordPath(t *testing.T, tx *TxContext, rc int64) []uint64 {
	t.Helper()
	body, err := handleVectorSearch(tx, EncodeVectorSearchArgsExt("docs", 10, []float32{1, 0},
		vector.Filter{Op: vector.FilterEq, Field: "session/rc", Value: vector.NewInt(rc)}))
	if err != nil {
		t.Fatalf("vector_search: %v", err)
	}
	rs, derr := DecodeVectorSearchResults(body)
	if derr != nil {
		t.Fatalf("DecodeVectorSearchResults: %v", derr)
	}
	out := make([]uint64, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.ID)
	}
	return out
}

func TestVectorOperateCAS(t *testing.T) {
	tx := newDenseOperateTx(t, nil)
	mustVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", setDyn("rc", 1))
	before, version := densePayload(t, tx, 1)
	recBefore := append([]byte(nil), before["session"].Rec...)

	_, _, err := callVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", addDynRet("rc", 1), version+999, true)
	if !errors.Is(err, vector.ErrVersionConflict) {
		t.Fatalf("stale CAS: err = %v, want vector.ErrVersionConflict", err)
	}
	after, versionAfter := densePayload(t, tx, 1)
	if !bytes.Equal(recBefore, after["session"].Rec) || versionAfter != version {
		t.Fatal("a rejected CAS mutated the point")
	}

	found, res, err := callVecOperate(t, tx, handleVectorOperate, 0, "docs", 1, "session", addDynRet("rc", 1), version, true)
	if err != nil {
		t.Fatalf("matching CAS: %v", err)
	}
	if !found || res == nil || res.Status != wire.OperateStatusOK {
		t.Fatalf("matching CAS: found=%v res=%+v", found, res)
	}
	if got := cellU(t, res.Values[0]); got != 2 {
		t.Fatalf("rc = %d, want 2", got)
	}
}

// TestVectorOperateIsDeterministicUnderTheStamp proves the handler threads the
// leader apply stamp into applyRecordBytes and nowhere consults a wall clock:
// two INDEPENDENT collections, in independent stores, running the same
// STAMP-carrying op-list under the same stamp must store byte-identical record
// bytes — and, run under two DIFFERENT stamps, must not. The second half is what
// stops the test passing by the stamp being ignored altogether.
//
// (The leader-vs-follower proof is Task 6's replica test; this is the handler's
// half of it.)
func TestVectorOperateIsDeterministicUnderTheStamp(t *testing.T) {
	stampOp := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpSTAMP, Type: wire.OperateTypeU64, Aux: wire.OperateStampMs, Path: namePath("t")},
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("rc"), A: 1},
	}}
	run := func(t *testing.T, stampMs int64) []byte {
		t.Helper()
		tx := newDenseOperateTx(t, nil)
		if found, res := mustVecOperate(t, tx, handleVectorOperate, stampMs, "docs", 1, "session", stampOp); !found ||
			res.Status != wire.OperateStatusOK {
			t.Fatalf("stamp %d: found=%v res=%+v", stampMs, found, res)
		}
		meta, _ := densePayload(t, tx, 1)
		return append([]byte(nil), meta["session"].Rec...)
	}

	const stampA = int64(1_700_000_000_000)
	const stampB = int64(1_700_000_777_000)

	a1 := run(t, stampA)
	a2 := run(t, stampA)
	if !bytes.Equal(a1, a2) {
		t.Fatalf("two collections under the SAME stamp stored different bytes:\n%x\n%x", a1, a2)
	}
	b := run(t, stampB)
	if bytes.Equal(a1, b) {
		t.Fatal("two DIFFERENT stamps stored identical bytes — the stamp is not reaching applyRecordBytes")
	}
}

// TestVectorOperateNamedAndMV runs the create/increment, failed-CHECK and
// missing-point cases against the named and multi-vector families, whose
// handlers differ from the dense one only in the engine entry point.
func TestVectorOperateNamedAndMV(t *testing.T) {
	for _, fam := range []struct {
		name    string
		col     string
		handler vecOperateHandler
		setup   func(t *testing.T, tx *TxContext)
		payload func(t *testing.T, tx *TxContext, id uint64) vector.Metadata
	}{
		{
			name: "named", col: "named", handler: handleNamedVectorOperate,
			setup: func(t *testing.T, tx *TxContext) {
				t.Helper()
				cfg := map[string]vector.NamedVectorParams{"title": {Dim: 2, Metric: vector.Cosine}}
				if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("named", cfg, 0)); err != nil {
					t.Fatalf("named create: %v", err)
				}
				if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("named", 1,
					map[string][]float32{"title": {1, 0}}, nil, 0)); err != nil {
					t.Fatalf("named insert: %v", err)
				}
			},
			payload: func(t *testing.T, tx *TxContext, id uint64) vector.Metadata {
				t.Helper()
				body, err := handleNamedGet(tx, EncodeVectorGetArgs("named", id, GetFlagsBoth))
				if err != nil {
					t.Fatalf("named get: %v", err)
				}
				found, _, meta, _, _, derr := DecodeNamedGetResultV(body)
				if derr != nil || !found {
					t.Fatalf("named get: found=%v err=%v", found, derr)
				}
				return meta
			},
		},
		{
			name: "mv", col: "mv", handler: handleMVVectorOperate,
			setup: func(t *testing.T, tx *TxContext) {
				t.Helper()
				if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", vector.MultiVectorConfig{
					Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1})); err != nil {
					t.Fatalf("mv create: %v", err)
				}
				if _, err := handleMVAdd(tx, EncodeMVAddArgs("mv", 1, [][]float32{{1, 0}}, nil)); err != nil {
					t.Fatalf("mv add: %v", err)
				}
			},
			payload: func(t *testing.T, tx *TxContext, id uint64) vector.Metadata {
				t.Helper()
				body, err := handleMVGet(tx, EncodeVectorGetArgs("mv", id, GetFlagsBoth))
				if err != nil {
					t.Fatalf("mv get: %v", err)
				}
				found, _, meta, _, derr := DecodeMVGetResultV(body)
				if derr != nil || !found {
					t.Fatalf("mv get: found=%v err=%v", found, derr)
				}
				return meta
			},
		},
	} {
		t.Run(fam.name, func(t *testing.T) {
			tx := newVecTx(t)
			fam.setup(t, tx)

			// create + increment
			if found, res := mustVecOperate(t, tx, fam.handler, 0, fam.col, 1, "session", setDyn("rc", 1)); !found ||
				res.Status != wire.OperateStatusOK {
				t.Fatalf("create: found=%v res=%+v", found, res)
			}
			found, res := mustVecOperate(t, tx, fam.handler, 0, fam.col, 1, "session", addDynRet("rc", 1))
			if !found || res.Status != wire.OperateStatusOK {
				t.Fatalf("increment: found=%v res=%+v", found, res)
			}
			if got := cellU(t, res.Values[0]); got != 2 {
				t.Fatalf("rc = %d, want 2", got)
			}
			meta := fam.payload(t, tx, 1)
			if meta["session"].Kind != vector.ValueRecord {
				t.Fatalf("payload = %+v, want a ValueRecord under session", meta)
			}
			recBefore := append([]byte(nil), meta["session"].Rec...)

			// failed CHECK: no-op
			check := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
				{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("rc"), A: 1},
				{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpEQ, Path: namePath("rc"), A: 99},
			}}
			found, res = mustVecOperate(t, tx, fam.handler, 0, fam.col, 1, "session", check)
			if !found || res.Status != wire.OperateStatusCheckFailed || res.FailedOp != 1 {
				t.Fatalf("check: found=%v res=%+v", found, res)
			}
			if after := fam.payload(t, tx, 1); !bytes.Equal(recBefore, after["session"].Rec) {
				t.Fatal("a failed CHECK changed the stored record")
			}

			// missing point: the not-found flag, not an error
			found, res, err := callVecOperate(t, tx, fam.handler, 0, fam.col, 999, "session", setDyn("rc", 1), 0, false)
			if err != nil {
				t.Fatalf("missing point: err = %v, want nil", err)
			}
			if found || res != nil {
				t.Fatalf("missing point: found=%v res=%+v, want false/nil", found, res)
			}
		})
	}
}

// TestVectorOperateRegistered pins the three ops into the builtin registry. It
// also covers ops/builtin.go's handler-count equality gate: a row added to
// wire.BuiltinOps without its handler (or the reverse) fails RegisterBuiltins.
func TestVectorOperateRegistered(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	a := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("rc"), A: 1}}}
	args, err := EncodeVectorOperateArgs("docs", 7, "session", a, 0, false)
	if err != nil {
		t.Fatalf("EncodeVectorOperateArgs: %v", err)
	}
	for _, op := range []string{"vector_operate", "vector_named_operate", "vector_mv_operate"} {
		_, kind, ke, ok := r.Lookup(op)
		if !ok {
			t.Errorf("%s not registered", op)
			continue
		}
		if kind != OpReadWrite {
			t.Errorf("%s kind = %v, want OpReadWrite", op, kind)
		}
		if ke == nil {
			t.Fatalf("%s has a nil key extractor (it would route to shard 0)", op)
		}
		key, found := ke(args)
		if !found || string(key) != "default/docs" {
			t.Errorf("%s key extractor = (%q,%v), want (default/docs,true)", op, key, found)
		}
	}
}

// TestVectorOperateHandlerDefendsInnerKeyAndTTL pins the handler's own guard.
// The codec already rejects both on the wire (sdk/wire's
// TestVectorOperateArgsRejectsInnerKey / RejectsTTLMode), so this is defence in
// depth: the target is named exactly once, by (collection, id, payloadKey), and
// a point's TTL is the point's. Neither may become a silently-ignored second
// source of truth if a future codec revision, or an in-process caller, hands the
// handler an OperateArgs the decoder never saw.
func TestVectorOperateHandlerDefendsInnerKeyAndTTL(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    *wire.OperateArgs
	}{
		{"nil args", nil},
		{"inner key", &wire.OperateArgs{Key: []byte("k"), Create: wire.OperateCreateDynamic}},
		{"ttl mode set", &wire.OperateArgs{Create: wire.OperateCreateDynamic, TTLMode: wire.OperateTTLSet, TTL: time.Minute}},
		{"ttl without a mode", &wire.OperateArgs{Create: wire.OperateCreateDynamic, TTL: time.Minute}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := checkVectorOperateArgs(tc.a); !errors.Is(err, wire.ErrOperateArgs) {
				t.Fatalf("err = %v, want wire.ErrOperateArgs", err)
			}
		})
	}
	ok := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: namePath("rc"), A: 1}}}
	if err := checkVectorOperateArgs(ok); err != nil {
		t.Fatalf("a well-formed call was rejected: %v", err)
	}
}

// callVecOperateV is callVecOperate keeping the version the reply frame now
// carries, which callVecOperate drops so its many callers stay unchanged.
func callVecOperateV(t *testing.T, tx *TxContext, h vecOperateHandler, stampMs int64,
	col string, id uint64, pk string, a *wire.OperateArgs,
) (bool, *wire.OperateResult, uint64) {
	t.Helper()
	args, err := EncodeVectorOperateArgs(col, id, pk, a, 0, false)
	if err != nil {
		t.Fatalf("EncodeVectorOperateArgs: %v", err)
	}
	tx.SetApplyStamp(uint64(stampMs), stampMs != 0) //nolint:gosec // test stamps are small positive millis
	defer tx.SetApplyStamp(0, false)
	body, herr := h(tx, args)
	if herr != nil {
		t.Fatalf("handler: %v", herr)
	}
	found, res, version, derr := DecodeVectorOperateResult(body)
	if derr != nil {
		t.Fatalf("DecodeVectorOperateResult: %v", derr)
	}
	return found, res, version
}

// TestVectorOperateReturnsTheAppliedVersion pins the handler's version
// threading, which is what lets a CAS loop retry from the reply frame instead of
// re-reading the point. Three properties, all of them checkable only here
// because only the handler sees both the engine's version and the wire frame:
// an applied call reports the point's NEW version, a failed CHECK reports the
// CURRENT unbumped one (so the retry has something to fence on), and a
// not-found point reports 0.
//
// Each version is cross-checked against what a real get op reports, so a
// plausible-looking but wrong number fails.
func TestVectorOperateReturnsTheAppliedVersion(t *testing.T) {
	tx := newDenseOperateTx(t, nil)

	found, _, v1 := callVecOperateV(t, tx, handleVectorOperate, 0, "docs", 1, "session", setDyn("rc", 1))
	if !found {
		t.Fatal("found = false, want true")
	}
	_, getV1 := densePayload(t, tx, 1)
	if v1 == 0 || v1 != getV1 {
		t.Fatalf("version = %d, want the point's version %d (non-zero)", v1, getV1)
	}

	_, _, v2 := callVecOperateV(t, tx, handleVectorOperate, 0, "docs", 1, "session", setDyn("rc", 2))
	_, getV2 := densePayload(t, tx, 1)
	if v2 <= v1 || v2 != getV2 {
		t.Fatalf("version = %d after a second apply, want > %d and == the point's %d", v2, v1, getV2)
	}

	// A failed CHECK writes nothing and bumps nothing, and must still report the
	// version a caller would retry against.
	checkFails := &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: []wire.OperateOp{
		{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpEQ, Path: namePath("rc"), A: 99},
	}}
	found, res, vFail := callVecOperateV(t, tx, handleVectorOperate, 0, "docs", 1, "session", checkFails)
	if !found || res == nil || res.Status != wire.OperateStatusCheckFailed {
		t.Fatalf("found=%v res=%+v, want a CheckFailed result", found, res)
	}
	if vFail != v2 {
		t.Fatalf("version = %d on a failed CHECK, want the current unbumped %d", vFail, v2)
	}

	// An absent point is the not-found flag, and carries no version.
	found, _, vMissing := callVecOperateV(t, tx, handleVectorOperate, 0, "docs", 9999, "session", setDyn("rc", 1))
	if found {
		t.Fatal("found = true for an absent point")
	}
	if vMissing != 0 {
		t.Fatalf("version = %d for an absent point, want 0", vMissing)
	}
}
