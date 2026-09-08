// SPDX-License-Identifier: Apache-2.0

package ops

// The operate v2 semantics suite.
//
// runSemantics is the executable form of the design doc's rules (§2.4 paths
// and typing, §2.5 determinism and deletion, §2.8 schema lifecycle, §2.9
// dynamic mode, §3.1–§3.5 the op vocabulary, returns, and call parameters).
// It is written once, against the `applier` function type, and run by every
// implementation of those semantics: the tree oracle here (TestOracleSemantics)
// and, in later tasks, each byte-level engine. An engine that passes this
// suite is, by definition, semantically equivalent to the oracle.
//
// Skipping dynamic mode. A sub-test whose name starts with "dynamic:" needs
// dynamic-mode records. An engine that implements only schema mode sets the
// package-level `skipDynamic` to true before calling runSemantics (and back
// to false afterwards); `sub` then skips exactly those sub-tests. Every other
// sub-test is schema mode (or mode-independent) and must pass everywhere.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// applier is one implementation of the operate apply semantics: rec == nil
// means the key is absent, and a nil returned record means the key is deleted
// (design doc §2.5). stampMs is the transaction timestamp STAMP writes; 0
// means unstamped.
type applier func(rec *wire.Record, a *wire.OperateArgs, stampMs int64) (*wire.Record, *wire.OperateResult, error)

// skipDynamic makes sub tests whose name starts with "dynamic:" skip. Set by
// an engine test whose engine implements schema mode only.
//
// It is package-level state, as is maxOperateRecordBytes, which one sub-test
// lowers and restores. No test in this file may call t.Parallel.
var skipDynamic bool

// sub runs one semantics sub-test, honouring skipDynamic.
func sub(t *testing.T, name string, fn func(t *testing.T)) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		if skipDynamic && strings.HasPrefix(name, "dynamic:") {
			t.Skip("engine under test implements schema mode only")
		}
		fn(t)
	})
}

// opFromSchema is the "type comes from the schema (or the stored field)" type
// byte, spelled short because nearly every op carries it (design doc §2.4).
const opFromSchema = wire.OperateTypeFromSchema

func TestOracleSemantics(t *testing.T) { runSemantics(t, applyTree) }

// TestOracleRoundTrips runs the same suite through an applier that encodes
// every record the oracle returns and decodes it again before handing it back.
// It proves two things at once: the oracle only ever builds canonical,
// decodable trees (design doc §2.5), and the suite really is written against
// the applier type, not against applyTree — which is how the byte engines will
// consume it.
func TestOracleRoundTrips(t *testing.T) {
	runSemantics(t, func(rec *wire.Record, a *wire.OperateArgs, stampMs int64) (*wire.Record, *wire.OperateResult, error) {
		out, res, err := applyTree(rec, a, stampMs)
		if err != nil || out == nil {
			return out, res, err
		}
		b := out.Encode()
		got, derr := wire.DecodeRecord(b)
		if derr != nil {
			return nil, nil, fmt.Errorf("oracle produced undecodable bytes %x: %w", b, derr)
		}
		if !bytes.Equal(got.Encode(), b) {
			return nil, nil, fmt.Errorf("record does not round-trip: %x", b)
		}
		return got, res, nil
	})
}

// TestSemanticsSkipDynamic checks the mechanism a schema-only engine uses to
// opt out of the dynamic-mode sub-tests.
func TestSemanticsSkipDynamic(t *testing.T) {
	skipDynamic = true
	defer func() { skipDynamic = false }()
	sub(t, "dynamic: sentinel", func(t *testing.T) { t.Fatal("a dynamic sub-test ran with skipDynamic set") })
	ran := false
	sub(t, "schema sentinel", func(t *testing.T) { ran = true })
	if !ran {
		t.Fatal("skipDynamic skipped a non-dynamic sub-test")
	}
}

func runSemantics(t *testing.T, apply applier) { //nolint:maintidx // one sub-test per spec rule, deliberately flat
	sub(t, "session flow", func(t *testing.T) {
		s := sessionSchema()
		a := &wire.OperateArgs{Key: []byte("s"), Create: wire.OperateCreateSchema, Schema: s.Encode(), Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpIF, Aux: wire.OperateCmpGE, Path: fieldPath(0), A: 255, B: 2},
			{Opcode: wire.OperateOpSHR, Type: opFromSchema, Path: fieldPath(0), A: 1},
			{Opcode: wire.OperateOpSHR, Type: opFromSchema, Path: fieldPath(1), A: 1},
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: fieldPath(0), A: 1},
			{Opcode: wire.OperateOpADD, Type: opFromSchema, Path: colPath(3, keyU64(77), 0), A: 1},
			{Opcode: wire.OperateOpMAX, Type: opFromSchema, Path: colPath(3, keyU64(77), 1), A: 500},
			{Opcode: wire.OperateOpSTAMP, Type: opFromSchema, Aux: wire.OperateStampS, Path: colPath(3, keyU64(77), 2)},
		}, Rets: []wire.OperateRet{
			{Mode: wire.OperateRetValue, Path: fieldPath(0)},
			{Mode: wire.OperateRetValue, Path: rowPath(3, keyU64(77))},
			{Mode: wire.OperateRetCount, Path: fieldPath(3)},
		}}
		var rec *wire.Record
		var res *wire.OperateResult
		var err error
		for i := 0; i < 300; i++ {
			rec, res, err = apply(rec, a, 1_700_000_000_000)
			if err != nil {
				t.Fatal(i, err)
			}
		}
		if res.Status != wire.OperateStatusOK {
			t.Fatal(res.Status)
		}
		rc, _, _ := wire.DecodeTaggedCell(res.Values[0])
		// 300 increments with a halve-at-255 guard: 255 → 127 → … never
		// exceeds 255 and is > 127.
		if rc.U > 255 || rc.U <= 127 {
			t.Fatalf("rc=%d", rc.U)
		}
		row := rec.Fields[3].Table.Rows[0]
		if row.Cols[0].Cell.U != 300 || row.Cols[1].Cell.U != 500 || row.Cols[2].Cell.U != 1_700_000_000 {
			t.Fatalf("%+v", row)
		}
		cnt, _, _ := wire.DecodeTaggedCell(res.Values[2])
		if cnt.U != 1 {
			t.Fatal(cnt.U)
		}
	})

	sub(t, "eviction MIN_COL with ties by key", func(t *testing.T) {
		s := sessionSchema()
		s.Fields[3].Table.Cap = 3
		mk := func(k uint64) *wire.OperateArgs {
			return schemaArgs(s,
				op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1),
				op(wire.OperateOpSTAMP, opFromSchema, colPath(3, keyU64(k), 2), 0).withAux(wire.OperateStampS))
		}
		var rec *wire.Record
		for _, step := range []struct {
			k  uint64
			st int64
		}{{5, 1000}, {9, 1000}, {1, 2000}, {7, 3000}} {
			var err error
			rec, _, err = apply(rec, mk(step.k), step.st*1000)
			if err != nil {
				t.Fatal(step.k, err)
			}
		}
		keys := rowKeys(rec.Fields[3].Table)
		if !reflect.DeepEqual(keys, []uint64{1, 7, 9}) {
			t.Fatalf("victim should be key 5 (oldest stamp, smallest key): %v", keys)
		}
	})

	sub(t, "eviction MIN_KEY orders numeric keys as integers", func(t *testing.T) {
		// LE key bytes of 256 are {0,1,0,...}, of 4 are {4,0,0,...}: bytewise
		// 256 < 4, as integers 4 < 256. The victim and the stored row order
		// both prove the comparison is integer, not bytewise (design doc §2.3).
		s := sessionSchema()
		s.Fields[3].Table.Cap = 2
		s.Fields[3].Table.Policy = wire.OperatePolicyMinKey
		var rec *wire.Record
		for _, k := range []uint64{4, 256, 5} {
			var err error
			rec, _, err = apply(rec, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1)), 0)
			if err != nil {
				t.Fatal(k, err)
			}
		}
		if keys := rowKeys(rec.Fields[3].Table); !reflect.DeepEqual(keys, []uint64{5, 256}) {
			t.Fatalf("want [5 256] (evict integer-smallest 4, sort as integers), got %v", keys)
		}
	})

	sub(t, "PolicyNone with a cap evicts by MIN_KEY", func(t *testing.T) {
		// Ruling: cap 0 is unbounded, but a cap > 0 with PolicyNone passes
		// Schema.Validate, so the applier must still pick a deterministic
		// victim; it evicts the smallest key.
		s := sessionSchema()
		s.Fields[3].Table.Cap = 2
		s.Fields[3].Table.Policy = wire.OperatePolicyNone
		var rec *wire.Record
		for _, k := range []uint64{4, 256, 5} {
			var err error
			rec, _, err = apply(rec, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1)), 0)
			if err != nil {
				t.Fatal(k, err)
			}
		}
		if keys := rowKeys(rec.Fields[3].Table); !reflect.DeepEqual(keys, []uint64{5, 256}) {
			t.Fatalf("want [5 256], got %v", keys)
		}
	})

	sub(t, "dynamic: CHECK aborts atomically and returns the old record", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeI32, namePath("stock"), 3)), 0)
		if err != nil {
			t.Fatal(err)
		}
		before := rec.Encode()
		rec2, res, err := apply(rec, withRet(dynArgs(
			op(wire.OperateOpCHECK, 0, namePath("stock"), 5).withAux(wire.OperateCmpGE),
			op(wire.OperateOpADD, wire.OperateTypeI32, namePath("stock"), -5),
			op(wire.OperateOpADD, wire.OperateTypeI32, namePath("reserved"), 5)), namePath("stock")), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed || res.FailedOp != 0 {
			t.Fatal(res, err)
		}
		if !bytes.Equal(rec2.Encode(), before) || !bytes.Equal(rec.Encode(), before) {
			t.Fatal("record changed on CHECK failure")
		}
		v, _, _ := wire.DecodeTaggedCell(res.Values[0])
		if int64(v.U) != 3 {
			t.Fatal("return not evaluated against the unchanged record")
		}
	})

	sub(t, "CHECK aborts atomically in schema mode", func(t *testing.T) {
		s := sessionSchema()
		rec, _, err := apply(nil, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 3)), 0)
		if err != nil {
			t.Fatal(err)
		}
		before := rec.Encode()
		rec2, res, err := apply(rec, withRet(schemaArgs(s,
			op(wire.OperateOpCHECK, opFromSchema, fieldPath(0), 5).withAux(wire.OperateCmpGE),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 10)), fieldPath(0)), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed || res.FailedOp != 0 {
			t.Fatal(res, err)
		}
		if !bytes.Equal(rec2.Encode(), before) {
			t.Fatal("record changed on CHECK failure")
		}
		v, _, _ := wire.DecodeTaggedCell(res.Values[0])
		if v.U != 3 {
			t.Fatalf("rc=%d, want the pre-call 3", v.U)
		}
		// A CHECK that fails after an op has already written must roll that
		// write back too: the list is all-or-nothing (design doc §3.3).
		rec3, res, err := apply(rec, withRet(schemaArgs(s,
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 10),
			op(wire.OperateOpCHECK, opFromSchema, fieldPath(0), 100).withAux(wire.OperateCmpGE),
			op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(1), 0), 1)), fieldPath(0)), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed || res.FailedOp != 1 {
			t.Fatal(res, err)
		}
		if !bytes.Equal(rec3.Encode(), before) {
			t.Fatal("a write before the failing CHECK was not rolled back")
		}
		v, _, _ = wire.DecodeTaggedCell(res.Values[0])
		if v.U != 3 {
			t.Fatalf("rc=%d, want the pre-call 3", v.U)
		}
	})

	sub(t, "CHECK failure on an absent record leaves it absent", func(t *testing.T) {
		s := sessionSchema()
		rec, res, err := apply(nil, withRet(schemaArgs(s,
			op(wire.OperateOpCHECK, opFromSchema, fieldPath(0), 5).withAux(wire.OperateCmpGE),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), fieldPath(0)), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed {
			t.Fatal(res, err)
		}
		if rec != nil {
			t.Fatal("record created despite CHECK failure")
		}
		if !bytes.Equal(res.Values[0], []byte{wire.OperateTypeUnset}) {
			t.Fatalf("want UNSET for a return against an absent record, got %x", res.Values[0])
		}
	})

	sub(t, "IF skips n ops", func(t *testing.T) {
		// The fixed-window rate limiter of design doc §4: 150 calls, each
		// guarded by IF(rc LT 100, skip 1), leave rc pinned at 100.
		s := sessionSchema()
		a := schemaArgs(s,
			op(wire.OperateOpIF, opFromSchema, fieldPath(0), 100).withAux(wire.OperateCmpLT).withB(1),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1))
		var rec *wire.Record
		for i := 0; i < 150; i++ {
			var err error
			rec, _, err = apply(rec, a, 0)
			if err != nil {
				t.Fatal(i, err)
			}
		}
		if rec.Fields[0].Cell.U != 100 {
			t.Fatalf("rc=%d, want 100", rec.Fields[0].Cell.U)
		}
	})

	sub(t, "IF skip count out of range is an error", func(t *testing.T) {
		s := sessionSchema()
		for _, n := range []int64{-1, 5} {
			_, _, err := apply(nil, schemaArgs(s,
				op(wire.OperateOpIF, opFromSchema, fieldPath(0), 100).withAux(wire.OperateCmpGE).withB(n),
				op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
			if !errors.Is(err, errOperateIfRange) {
				t.Fatalf("n=%d: err=%v, want errOperateIfRange", n, err)
			}
		}
		// The bound is eager: an out-of-range n is an error even when the
		// condition holds and nothing would be skipped.
		if _, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpIF, opFromSchema, fieldPath(0), 0).withAux(wire.OperateCmpEQ).withB(5),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0); !errors.Is(err, errOperateIfRange) {
			t.Fatalf("condition true, n out of range: %v", err)
		}
		// n exactly equal to the remaining op count is legal.
		rec, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpIF, opFromSchema, fieldPath(0), 100).withAux(wire.OperateCmpGE).withB(1),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[0].Cell.U != 0 {
			t.Fatal("guarded op ran")
		}
	})

	sub(t, "dynamic: vivify, stored type wins, domain mismatch errors", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath("x"), 200)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.Fields) != 1 || rec.Fields[0].Name != "x" ||
			rec.Fields[0].Cell.Type != wire.OperateTypeU8 || rec.Fields[0].Cell.U != 200 {
			t.Fatalf("%+v", rec.Fields)
		}
		if _, _, err := apply(rec, dynArgs(op(wire.OperateOpADD, wire.OperateTypeF64, namePath("x"), f64(1.5))), 0); !errors.Is(err, wire.ErrOperateType) {
			t.Fatalf("float op on an int field: %v", err)
		}
		got, _, err := apply(rec, dynArgs(op(wire.OperateOpADD, wire.OperateTypeU32, namePath("x"), 100)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Fields[0].Cell.Type != wire.OperateTypeU8 || got.Fields[0].Cell.U != 255 {
			t.Fatalf("stored U8 must win and saturate: %+v", got.Fields[0].Cell)
		}
		if _, _, err := apply(rec, dynArgs(op(wire.OperateOpADD, opFromSchema, namePath("fresh"), 1)), 0); !errors.Is(err, wire.ErrOperateType) {
			t.Fatalf("0xFF on a missing dynamic target: %v", err)
		}
		// SET replaces the scalar including its type (design doc §2.9).
		got, _, err = apply(rec, dynArgs(op(wire.OperateOpSET, wire.OperateTypeBytes, namePath("x"), 0).withBytes([]byte("hi"))), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Fields[0].Cell.Type != wire.OperateTypeBytes || string(got.Fields[0].Cell.B) != "hi" {
			t.Fatalf("SET did not replace the type: %+v", got.Fields[0].Cell)
		}
		// A signed field stores the sign-extended bit pattern.
		got, _, err = apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeI32, namePath("d"), -5)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Fields[0].Cell.U != su64(-5) {
			t.Fatalf("%+v", got.Fields[0].Cell)
		}
		// A FIXED field created in dynamic mode takes N = len(Bytes).
		got, _, err = apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeFixed, namePath("cc"), 0).withBytes([]byte("DE"))), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Fields[0].Cell.N != 2 || string(got.Fields[0].Cell.B) != "DE" {
			t.Fatalf("%+v", got.Fields[0].Cell)
		}
	})

	sub(t, "dynamic: a scalar and a table never swap without DEL", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("t", keyU64(1), "c"), 4)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := apply(rec, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath("t"), 1)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("scalar SET over a table: %v", err)
		}
		scalar, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath("t"), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := apply(scalar, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("t", keyU64(1), "c"), 4)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("row path into a scalar field: %v", err)
		}
		// DEL first, then the other kind is free to appear.
		swapped, _, err := apply(scalar, dynArgs(
			op(wire.OperateOpDEL, opFromSchema, namePath("t"), 0),
			op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("t", keyU64(1), "c"), 4)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if swapped.Fields[0].Cell.Type != wire.OperateTypeTable || len(swapped.Fields[0].Table.Rows) != 1 {
			t.Fatalf("%+v", swapped.Fields[0])
		}
	})

	sub(t, "schema mode rejects bad paths, types, versions and modes", func(t *testing.T) {
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(1), 0), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		v2 := sessionSchema()
		v2.Version = 2
		cases := []struct {
			name string
			rec  *wire.Record
			args *wire.OperateArgs
			want error
		}{
			{"position outside the schema", base,
				schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(9), 1)), wire.ErrOperatePath},
			{"name segment without stored names", base,
				schemaArgs(s, op(wire.OperateOpADD, opFromSchema, namePath("rc"), 1)), wire.ErrOperatePath},
			{"column outside the schema", base,
				schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(1), 7), 1)), wire.ErrOperatePath},
			{"row path into a non-table field", base,
				schemaArgs(s, op(wire.OperateOpADD, opFromSchema, rowPath(0, keyU64(1)), 1)), wire.ErrOperatePath},
			{"scalar op on a table field", base,
				schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(3), 1)), wire.ErrOperatePath},
			{"row key of the wrong width", base,
				schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, []byte{1, 2, 3, 4}, 0), 1)), wire.ErrOperatePath},
			{"type byte disagreeing with the schema", base,
				schemaArgs(s, op(wire.OperateOpADD, wire.OperateTypeU16, fieldPath(0), 1)), wire.ErrOperateType},
			{"schema version mismatch", base,
				schemaArgs(v2, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), wire.ErrOperateSchemaVersion},
			{"create mode disagreeing with the record", base,
				dynArgs(op(wire.OperateOpADD, wire.OperateTypeU8, namePath("rc"), 1)), wire.ErrOperateMode},
			{"create NONE against an absent record", nil,
				noneArgs(op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), errOperateAbsent},
			{"CONFIG in schema mode", base,
				schemaArgs(s, op(wire.OperateOpCONFIG, opFromSchema, fieldPath(3), 4).withAux(wire.OperatePolicyMinKey)), wire.ErrOperateOpcode},
			{"unknown opcode", base,
				schemaArgs(s, op(99, opFromSchema, fieldPath(0), 1)), wire.ErrOperateOpcode},
			{"unknown comparator", base,
				schemaArgs(s, op(wire.OperateOpIF, opFromSchema, fieldPath(0), 1).withAux(99).withB(0)), wire.ErrOperateCmp},
			{"create SCHEMA without a blob", base,
				&wire.OperateArgs{Create: wire.OperateCreateSchema}, wire.ErrOperateSchema},
		}
		for _, tc := range cases {
			_, _, err := apply(tc.rec, tc.args, 0)
			if !errors.Is(err, tc.want) {
				t.Errorf("%s: err=%v, want %v", tc.name, err, tc.want)
			}
		}
		// create NONE against an existing schema-mode record is fine. The
		// dynamic-mode half of that rule is exercised by the freeze sub-test,
		// which migrates a dynamic record with noneArgs.
		if _, _, err := apply(base, noneArgs(op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0); err != nil {
			t.Fatalf("create NONE on an existing record: %v", err)
		}
	})

	sub(t, "schema mode accepts name segments when the schema stores names", func(t *testing.T) {
		s := sessionSchema()
		s.StoreNames = true
		rec, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpADD, opFromSchema, namePath("rc"), 2),
			op(wire.OperateOpADD, opFromSchema, nameColPath("b", keyU64(9), "c"), 3)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[0].Cell.U != 2 || rec.Fields[3].Table.Rows[0].Cols[0].Cell.U != 3 {
			t.Fatalf("%+v", rec.Fields)
		}
		if _, _, err := apply(rec, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, namePath("nope"), 1)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("unknown name: %v", err)
		}
	})

	sub(t, "schema fields are always present and read as zero", func(t *testing.T) {
		// A schema-mode record cannot distinguish "written 0" from "never
		// written" — a fixed-width field is the same bytes either way — so
		// every declared field is present, and an all-zero record is a valid
		// state (design doc §2.5), while arithmetic still sees zero.
		s := sessionSchema()
		rec, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpIF, opFromSchema, fieldPath(0), 1).withAux(wire.OperateCmpLT).withB(1),
			op(wire.OperateOpADD, opFromSchema, fieldPath(2), 7)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[2].Cell.U != 7 {
			t.Fatalf("absent-is-zero guard did not fire: %+v", rec.Fields[2].Cell)
		}
		_, res, err := apply(rec, schemaArgs(s, op(wire.OperateOpCHECK, opFromSchema, fieldPath(1), 0).withAux(wire.OperateCmpExists)), 0)
		if err != nil || res.Status != wire.OperateStatusOK {
			t.Fatalf("EXISTS on a declared field: %v %+v", err, res)
		}
		_, res, err = apply(rec, schemaArgs(s, op(wire.OperateOpCHECK, opFromSchema, fieldPath(1), 0).withAux(wire.OperateCmpAbsent)), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed {
			t.Fatalf("ABSENT on a declared field: %v %+v", err, res)
		}
	})

	sub(t, "reads and comparisons never vivify", func(t *testing.T) {
		s := sessionSchema()
		rec, res, err := apply(nil, withCount(schemaArgs(s,
			op(wire.OperateOpIF, opFromSchema, rowPath(3, keyU64(4)), 0).withAux(wire.OperateCmpExists).withB(0),
			op(wire.OperateOpCHECK, opFromSchema, rowPath(3, keyU64(4)), 0).withAux(wire.OperateCmpAbsent)),
			fieldPath(3)), 0)
		if err != nil || res.Status != wire.OperateStatusOK {
			t.Fatal(err, res)
		}
		if len(rec.Fields[3].Table.Rows) != 0 {
			t.Fatal("a comparison vivified a row")
		}
		cnt, _, _ := wire.DecodeTaggedCell(res.Values[0])
		if cnt.U != 0 {
			t.Fatalf("count=%d", cnt.U)
		}
	})

	sub(t, "dynamic: absent is zero; EXISTS and ABSENT", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(
			op(wire.OperateOpIF, 0, namePath("x"), 1).withAux(wire.OperateCmpLT).withB(1),
			op(wire.OperateOpSET, wire.OperateTypeU8, namePath("x"), 7)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec.Fields) != 1 || rec.Fields[0].Cell.U != 7 {
			t.Fatalf("absent read as zero should have let the guarded SET run: %+v", rec.Fields)
		}
		_, res, err := apply(rec, dynArgs(
			op(wire.OperateOpCHECK, 0, namePath("z"), 0).withAux(wire.OperateCmpExists),
			op(wire.OperateOpSET, wire.OperateTypeU8, namePath("z"), 1)), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed || res.FailedOp != 0 {
			t.Fatalf("%v %+v", err, res)
		}
		got, _, err := apply(rec, dynArgs(op(wire.OperateOpIF, 0, namePath("q"), 1).withAux(wire.OperateCmpLT).withB(0)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(got.Fields) != 1 {
			t.Fatal("an IF vivified a dynamic field")
		}
	})

	sub(t, "DEL semantics in schema mode", func(t *testing.T) {
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 5),
			op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(1), 0), 1),
			op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(2), 0), 2)), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := apply(base, schemaArgs(s, op(wire.OperateOpDEL, opFromSchema, fieldPath(0), 0)), 0)
		if err != nil || got == nil {
			t.Fatal(err, got)
		}
		if got.Fields[0].Cell.U != 0 {
			t.Fatalf("DEL of a schema field must zero it: %+v", got.Fields[0].Cell)
		}
		got, _, err = apply(base, schemaArgs(s, op(wire.OperateOpDEL, opFromSchema, colPath(3, keyU64(1), 0), 0)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Fields[3].Table.Rows[0].Cols[0].Cell.U != 0 || len(got.Fields[3].Table.Rows) != 2 {
			t.Fatalf("DEL of a column must zero it and keep the row: %+v", got.Fields[3].Table.Rows)
		}
		got, _, err = apply(base, schemaArgs(s, op(wire.OperateOpDEL, opFromSchema, rowPath(3, keyU64(1)), 0)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if keys := rowKeys(got.Fields[3].Table); !reflect.DeepEqual(keys, []uint64{2}) {
			t.Fatalf("DEL of a row: %v", keys)
		}
		got, _, err = apply(base, schemaArgs(s, op(wire.OperateOpDEL, opFromSchema, rowPath(3, keyU64(99)), 0)), 0)
		if err != nil {
			t.Fatalf("DEL of an absent row must be a no-op: %v", err)
		}
		if len(got.Fields[3].Table.Rows) != 2 {
			t.Fatal("DEL of an absent row changed the table")
		}
		got, _, err = apply(base, schemaArgs(s, op(wire.OperateOpDEL, opFromSchema, fieldPath(3), 0)), 0)
		if err != nil || got == nil {
			t.Fatal(err, got)
		}
		if len(got.Fields[3].Table.Rows) != 0 {
			t.Fatal("DEL of a table field must empty it")
		}
		// A schema-mode record is never deleted implicitly: an all-zero
		// record survives (design doc §2.5).
		got, _, err = apply(base, schemaArgs(s,
			op(wire.OperateOpDEL, opFromSchema, fieldPath(0), 0),
			op(wire.OperateOpDEL, opFromSchema, fieldPath(3), 0)), 0)
		if err != nil || got == nil {
			t.Fatal("an emptied schema record must survive", err)
		}
		got, _, err = apply(base, schemaArgs(s, op(wire.OperateOpDEL, opFromSchema, recPath(), 0)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != nil {
			t.Fatal("DEL () must delete the record")
		}
	})

	sub(t, "dynamic: DEL of the last field deletes the record", func(t *testing.T) {
		base, _, err := apply(nil, dynArgs(
			op(wire.OperateOpSET, wire.OperateTypeU8, namePath("a"), 1),
			op(wire.OperateOpSET, wire.OperateTypeU8, namePath("b"), 2)), 0)
		if err != nil {
			t.Fatal(err)
		}
		one, _, err := apply(base, dynArgs(op(wire.OperateOpDEL, opFromSchema, namePath("a"), 0)), 0)
		if err != nil || one == nil || len(one.Fields) != 1 {
			t.Fatalf("%v %+v", err, one)
		}
		gone, _, err := apply(one, dynArgs(op(wire.OperateOpDEL, opFromSchema, namePath("b"), 0)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if gone != nil {
			t.Fatal("deleting the last dynamic field must delete the record")
		}
		// A dynamic column is removed, not zeroed, and the row survives.
		tbl, _, err := apply(nil, dynArgs(
			op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("t", keyU64(1), "c"), 4),
			op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("t", keyU64(1), "d"), 5)), 0)
		if err != nil {
			t.Fatal(err)
		}
		tbl, _, err = apply(tbl, dynArgs(op(wire.OperateOpDEL, opFromSchema, nameColPath("t", keyU64(1), "c"), 0)), 0)
		if err != nil {
			t.Fatal(err)
		}
		row := tbl.Fields[0].Table.Rows[0]
		if len(row.Cols) != 1 || row.Cols[0].Name != "d" {
			t.Fatalf("%+v", row)
		}
		// Ruling: a row is a keyed entity, so it survives losing its last
		// column and still counts as present.
		bare, res, err := apply(tbl, withCount(dynArgs(
			op(wire.OperateOpDEL, opFromSchema, nameColPath("t", keyU64(1), "d"), 0)),
			nameRowPath("t", keyU64(1))), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(bare.Fields[0].Table.Rows) != 1 || len(bare.Fields[0].Table.Rows[0].Cols) != 0 {
			t.Fatalf("the row should survive its last column: %+v", bare.Fields[0].Table.Rows)
		}
		if c, _, _ := wire.DecodeTaggedCell(res.Values[0]); c.U != 1 {
			t.Fatalf("COUNT of a column-less row %d, want 1", c.U)
		}
	})

	sub(t, "dynamic: TRIM and CONFIG", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(
			op(wire.OperateOpCONFIG, opFromSchema, namePath("ev"), 2).withAux(wire.OperatePolicyMinKey)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[0].Table == nil || rec.Fields[0].Table.Cap != 2 || rec.Fields[0].Table.Policy != wire.OperatePolicyMinKey {
			t.Fatalf("CONFIG must create the table with the eviction triple: %+v", rec.Fields[0])
		}
		for k := uint64(1); k <= 3; k++ {
			rec, _, err = apply(rec, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("ev", keyU64(k), "v"), int64(k))), 0)
			if err != nil {
				t.Fatal(k, err)
			}
		}
		if keys := rowKeys(rec.Fields[0].Table); !reflect.DeepEqual(keys, []uint64{2, 3}) {
			t.Fatalf("cap 2 MIN_KEY: %v", keys)
		}
		rec, _, err = apply(rec, dynArgs(op(wire.OperateOpTRIM, opFromSchema, namePath("ev"), 1).withAux(wire.OperatePolicyMaxKey)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if keys := rowKeys(rec.Fields[0].Table); !reflect.DeepEqual(keys, []uint64{2}) {
			t.Fatalf("TRIM keep 1 MAX_KEY: %v", keys)
		}
		if rec.Fields[0].Table.Policy != wire.OperatePolicyMinKey {
			t.Fatal("TRIM must not change the stored policy")
		}
		// CONFIG is a table op: its type byte is meaningless and ignored.
		rec, _, err = apply(rec, dynArgs(
			op(wire.OperateOpCONFIG, wire.OperateTypeF32, namePath("ev"), 5).withAux(wire.OperatePolicyMaxKey)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[0].Table.Cap != 5 || rec.Fields[0].Table.Policy != wire.OperatePolicyMaxKey {
			t.Fatalf("%+v", rec.Fields[0].Table)
		}
	})

	sub(t, "dynamic: MIN_COL treats a missing column as zero", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(
			op(wire.OperateOpCONFIG, opFromSchema, namePath("ev"), 2).
				withAux(wire.OperatePolicyMinCol).withBytes([]byte("s")),
			op(wire.OperateOpSET, wire.OperateTypeU32, nameColPath("ev", keyU64(1), "s"), 5),
			op(wire.OperateOpSET, wire.OperateTypeU32, nameColPath("ev", keyU64(2), "v"), 9)), 0)
		if err != nil {
			t.Fatal(err)
		}
		rec, _, err = apply(rec, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU32, nameColPath("ev", keyU64(3), "s"), 7)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if keys := rowKeys(rec.Fields[0].Table); !reflect.DeepEqual(keys, []uint64{1, 3}) {
			t.Fatalf("row 2 (no 's' column, so 0) should be the victim: %v", keys)
		}
	})

	sub(t, "TRIM in schema mode keeps the stored policy", func(t *testing.T) {
		s := sessionSchema()
		var rec *wire.Record
		var err error
		for k := uint64(1); k <= 3; k++ {
			rec, _, err = apply(rec, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1)), 0)
			if err != nil {
				t.Fatal(k, err)
			}
		}
		rec, _, err = apply(rec, schemaArgs(s, op(wire.OperateOpTRIM, opFromSchema, fieldPath(3), 1).withAux(wire.OperatePolicyMinKey)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if keys := rowKeys(rec.Fields[3].Table); !reflect.DeepEqual(keys, []uint64{3}) {
			t.Fatalf("%v", keys)
		}
		if rec.Fields[3].Table.Policy != wire.OperatePolicyMinCol {
			t.Fatal("TRIM changed the stored policy")
		}
	})

	sub(t, "caps are enforced with the record unchanged", func(t *testing.T) {
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(1), 0), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		before := base.Encode()
		big := int64(wire.OperateMaxRows) + 1
		cases := []struct {
			name string
			args *wire.OperateArgs
		}{
			{"TRIM keep above maxRows", schemaArgs(s, op(wire.OperateOpTRIM, opFromSchema, fieldPath(3), big).withAux(wire.OperatePolicyMinKey))},
			{"TRIM keep negative", schemaArgs(s, op(wire.OperateOpTRIM, opFromSchema, fieldPath(3), -1).withAux(wire.OperatePolicyMinKey))},
		}
		for _, tc := range cases {
			got, _, err := apply(base, tc.args, 0)
			if !errors.Is(err, wire.ErrOperateCap) {
				t.Errorf("%s: err=%v, want ErrOperateCap", tc.name, err)
			}
			if got != nil && !bytes.Equal(got.Encode(), before) {
				t.Errorf("%s: record changed", tc.name)
			}
		}
		if !bytes.Equal(base.Encode(), before) {
			t.Fatal("input record mutated by a failing call")
		}
	})

	sub(t, "dynamic: names and keys are capped at 255 bytes", func(t *testing.T) {
		long := strings.Repeat("n", 256)
		if _, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath(long), 1)), 0); !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("long name: %v", err)
		}
		if _, _, err := apply(nil, dynArgs(
			op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("t", bytes.Repeat([]byte{7}, 256), "c"), 1)), 0); !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("long key: %v", err)
		}
		if _, _, err := apply(nil, dynArgs(
			op(wire.OperateOpSET, wire.OperateTypeBytes, namePath("b"), 0).withBytes(bytes.Repeat([]byte{1}, wire.OperateMaxBytesLen+1))), 0); !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("long BYTES: %v", err)
		}
		if _, _, err := apply(nil, dynArgs(
			op(wire.OperateOpCONFIG, opFromSchema, namePath("ev"), int64(wire.OperateMaxRows)+1).withAux(wire.OperatePolicyMinKey)), 0); !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("CONFIG cap above maxRows: %v", err)
		}
		if _, _, err := apply(nil, dynArgs(
			op(wire.OperateOpCONFIG, opFromSchema, namePath("ev"), 2).withAux(wire.OperatePolicyMinCol)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("CONFIG MIN_COL without a column name: %v", err)
		}
	})

	sub(t, "MIGRATE is append-only and must be the first op", func(t *testing.T) {
		s1 := sessionSchema()
		var rec *wire.Record
		var err error
		for i, k := range []uint64{1, 2, 3} {
			rec, _, err = apply(rec, schemaArgs(s1,
				op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), int64(i+1)),
				op(wire.OperateOpSTAMP, opFromSchema, colPath(3, keyU64(k), 2), 0).withAux(wire.OperateStampS)),
				int64(i+1)*1000*1000)
			if err != nil {
				t.Fatal(k, err)
			}
		}
		s2 := sessionSchema()
		s2.Version = 2
		s2.Fields[3].Table.Cols = append(s2.Fields[3].Table.Cols, wire.ColumnDef{Name: "n", Type: wire.OperateTypeU8})
		s2.Fields[3].Table.Cap = 2
		s2.Fields = append(s2.Fields, wire.FieldDef{Name: "extra", Type: wire.OperateTypeI16})
		migrate := op(wire.OperateOpMIGRATE, opFromSchema, recPath(), 1).withBytes(s2.Encode())

		got, _, err := apply(rec, noneArgs(migrate), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Schema.Version != 2 || len(got.Fields) != 5 {
			t.Fatalf("%+v", got.Schema)
		}
		if keys := rowKeys(got.Fields[3].Table); !reflect.DeepEqual(keys, []uint64{2, 3}) {
			t.Fatalf("lowered cap must trim by the new policy (MIN_COL t): %v", keys)
		}
		row := got.Fields[3].Table.Rows[0]
		if len(row.Cols) != 4 || row.Cols[0].Cell.U != 2 || row.Cols[3].Cell.U != 0 {
			t.Fatalf("append-only rebuild: %+v", row)
		}
		if got.Fields[4].Cell.U != 0 || got.Fields[4].Cell.Type != wire.OperateTypeI16 {
			t.Fatalf("appended field: %+v", got.Fields[4].Cell)
		}
		// The migrated record now demands the new version.
		if _, _, err := apply(got, schemaArgs(s1, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0); !errors.Is(err, wire.ErrOperateSchemaVersion) {
			t.Fatalf("stale version accepted: %v", err)
		}
		bad := []struct {
			name string
			args *wire.OperateArgs
			want error
		}{
			{"wrong from-version", noneArgs(op(wire.OperateOpMIGRATE, opFromSchema, recPath(), 5).withBytes(s2.Encode())), wire.ErrOperateSchemaVersion},
			{"not the first op", noneArgs(op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1), migrate), wire.ErrOperateOpcode},
			{"path is not the record", noneArgs(op(wire.OperateOpMIGRATE, opFromSchema, fieldPath(0), 1).withBytes(s2.Encode())), wire.ErrOperatePath},
			{"not an append-only extension", noneArgs(op(wire.OperateOpMIGRATE, opFromSchema, recPath(), 1).withBytes(widenedSchema().Encode())), wire.ErrOperateSchema},
			{"malformed schema blob", noneArgs(op(wire.OperateOpMIGRATE, opFromSchema, recPath(), 1).withBytes([]byte{9, 9})), wire.ErrShortArgs},
		}
		for _, tc := range bad {
			if _, _, err := apply(rec, tc.args, 0); !errors.Is(err, tc.want) {
				t.Errorf("%s: err=%v, want %v", tc.name, err, tc.want)
			}
		}
	})

	sub(t, "dynamic: MIGRATE freezes a dynamic record into a schema", func(t *testing.T) {
		base, _, err := apply(nil, dynArgs(
			op(wire.OperateOpSET, wire.OperateTypeU32, namePath("hits"), 7),
			op(wire.OperateOpSET, wire.OperateTypeBytes, namePath("name"), 0).withBytes([]byte("ab")),
			op(wire.OperateOpSET, wire.OperateTypeU16, nameColPath("b", keyU64(9), "c"), 3)), 0)
		if err != nil {
			t.Fatal(err)
		}
		fs := frozenSchema()
		migrate := op(wire.OperateOpMIGRATE, opFromSchema, recPath(), wire.OperateMigrateFromDynamic).withBytes(fs.Encode())
		got, _, err := apply(base, noneArgs(migrate), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got.Mode != wire.OperateModeSchema || got.Schema.Version != 3 {
			t.Fatalf("%+v", got.Schema)
		}
		if got.Fields[0].Cell.U != 7 || string(got.Fields[1].Cell.B) != "ab" {
			t.Fatalf("values not preserved: %+v", got.Fields)
		}
		row := got.Fields[2].Table.Rows[0]
		if !bytes.Equal(row.Key, keyU64(9)) || row.Cols[0].Cell.U != 3 {
			t.Fatalf("rows not packed: %+v", row)
		}
		// An extra dynamic field is an error unless DROP_EXTRA is set.
		withExtra, _, err := apply(base, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath("zz"), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := apply(withExtra, noneArgs(migrate), 0); !errors.Is(err, wire.ErrOperateSchema) {
			t.Fatalf("extra field accepted: %v", err)
		}
		dropped, _, err := apply(withExtra, noneArgs(migrate.withAux(wire.OperateMigrateDropExtra)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(dropped.Fields) != 3 || dropped.Fields[0].Cell.U != 7 {
			t.Fatalf("%+v", dropped.Fields)
		}
		// A domain mismatch between the dynamic value and the schema is an error.
		wrong, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeBytes, namePath("hits"), 0).withBytes([]byte("x"))), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := apply(wrong, noneArgs(migrate.withAux(wire.OperateMigrateDropExtra)), 0); !errors.Is(err, wire.ErrOperateType) {
			t.Fatalf("domain mismatch accepted: %v", err)
		}
		// A dynamic row key that is not exactly the schema's key width cannot
		// be packed fixed-width.
		narrow, _, err := apply(nil, dynArgs(
			op(wire.OperateOpSET, wire.OperateTypeU16, nameColPath("b", []byte{1, 2, 3, 4}, "c"), 3)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := apply(narrow, noneArgs(migrate.withAux(wire.OperateMigrateDropExtra)), 0); !errors.Is(err, wire.ErrOperateSchema) {
			t.Fatalf("row key of the wrong width accepted: %v", err)
		}
		// Freezing needs names in the target schema.
		nameless := frozenSchema()
		nameless.StoreNames = false
		anon := op(wire.OperateOpMIGRATE, opFromSchema, recPath(), wire.OperateMigrateFromDynamic).withBytes(nameless.Encode())
		if _, _, err := apply(base, noneArgs(anon), 0); !errors.Is(err, wire.ErrOperateSchema) {
			t.Fatalf("nameless target schema accepted: %v", err)
		}
	})

	sub(t, "VALUE and COUNT of record, table, row, scalar and absent", func(t *testing.T) {
		s := sessionSchema()
		k := keyU64(77)
		rec, res, err := apply(nil, &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(),
			Ops: []wire.OperateOp{
				wire.OperateOp(op(wire.OperateOpADD, opFromSchema, fieldPath(0), 9)),
				wire.OperateOp(op(wire.OperateOpADD, opFromSchema, colPath(3, k, 0), 1)),
				wire.OperateOp(op(wire.OperateOpMAX, opFromSchema, colPath(3, k, 1), 500)),
				wire.OperateOp(op(wire.OperateOpSTAMP, opFromSchema, colPath(3, k, 2), 0).withAux(wire.OperateStampS)),
			},
			Rets: []wire.OperateRet{
				{Mode: wire.OperateRetValue, Path: recPath()},
				{Mode: wire.OperateRetValue, Path: fieldPath(3)},
				{Mode: wire.OperateRetValue, Path: rowPath(3, k)},
				{Mode: wire.OperateRetValue, Path: fieldPath(0)},
				{Mode: wire.OperateRetValue, Path: colPath(3, keyU64(999), 0)},
				{Mode: wire.OperateRetValue, Path: rowPath(3, keyU64(999))},
				{Mode: wire.OperateRetCount, Path: recPath()},
				{Mode: wire.OperateRetCount, Path: fieldPath(3)},
				{Mode: wire.OperateRetCount, Path: rowPath(3, k)},
				{Mode: wire.OperateRetCount, Path: rowPath(3, keyU64(999))},
				{Mode: wire.OperateRetCount, Path: fieldPath(0)},
			}}, 1_700_000_000_000)
		if err != nil || res.Status != wire.OperateStatusOK {
			t.Fatal(err, res)
		}
		if !bytes.Equal(res.Values[0], rec.Encode()) {
			t.Fatal("record VALUE must be the full stored encoding")
		}

		cCell := wire.Cell{Type: wire.OperateTypeU16, U: 1}
		hiCell := wire.Cell{Type: wire.OperateTypeI64, U: 500}
		tCell := wire.Cell{Type: wire.OperateTypeU32, U: 1_700_000_000}

		wantTable := []byte{wire.OperateModeSchema}
		wantTable = binary.AppendUvarint(wantTable, 1)
		wantTable = append(wantTable, k...)
		for _, c := range []wire.Cell{cCell, hiCell, tCell} {
			wantTable = wire.AppendCellData(wantTable, c)
		}
		if !bytes.Equal(res.Values[1], wantTable) {
			t.Fatalf("table VALUE\n got %x\nwant %x", res.Values[1], wantTable)
		}

		wantRow := []byte{wire.OperateModeSchema}
		wantRow = binary.AppendUvarint(wantRow, 3)
		for _, c := range []wire.Cell{cCell, hiCell, tCell} {
			wantRow = wire.AppendTaggedCell(wantRow, c)
		}
		if !bytes.Equal(res.Values[2], wantRow) {
			t.Fatalf("row VALUE\n got %x\nwant %x", res.Values[2], wantRow)
		}

		if !bytes.Equal(res.Values[3], wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU8, U: 9})) {
			t.Fatalf("scalar VALUE %x", res.Values[3])
		}
		for _, i := range []int{4, 5} {
			if !bytes.Equal(res.Values[i], []byte{wire.OperateTypeUnset}) {
				t.Fatalf("absent VALUE %d: %x", i, res.Values[i])
			}
		}
		for i, want := range map[int]uint64{6: 4, 7: 1, 8: 1, 9: 0, 10: 1} {
			c, _, derr := wire.DecodeTaggedCell(res.Values[i])
			if derr != nil || c.Type != wire.OperateTypeU64 || c.U != want {
				t.Fatalf("COUNT %d: %+v %v, want %d", i, c, derr, want)
			}
		}
	})

	sub(t, "dynamic: VALUE of a table and a row carry names", func(t *testing.T) {
		rec, res, err := apply(nil, &wire.OperateArgs{Create: wire.OperateCreateDynamic,
			Ops: []wire.OperateOp{
				wire.OperateOp(op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("ev", keyU64(1), "v"), 4)),
				wire.OperateOp(op(wire.OperateOpSET, wire.OperateTypeU8, nameColPath("ev", keyU64(1), "a"), 5)),
			},
			Rets: []wire.OperateRet{
				{Mode: wire.OperateRetValue, Path: namePath("ev")},
				{Mode: wire.OperateRetValue, Path: nameRowPath("ev", keyU64(1))},
				{Mode: wire.OperateRetValue, Path: namePath("nope")},
				{Mode: wire.OperateRetCount, Path: recPath()},
				{Mode: wire.OperateRetCount, Path: namePath("nope")},
				{Mode: wire.OperateRetCount, Path: namePath("ev")},
			}}, 0)
		if err != nil {
			t.Fatal(err)
		}
		// The record holds exactly one field, "ev", so the bytes its encoder
		// emits for that field are everything after
		// [mode][nFields=1][nlen=2]["ev"][type=TABLE].
		enc := rec.Encode()
		wantTable := append([]byte{wire.OperateModeDynamic}, enc[1+1+1+2+1:]...)
		if !bytes.Equal(res.Values[0], wantTable) {
			t.Fatalf("table VALUE\n got %x\nwant %x", res.Values[0], wantTable)
		}
		wantRow := []byte{wire.OperateModeDynamic}
		wantRow = binary.AppendUvarint(wantRow, 2)
		for _, c := range []struct {
			name string
			cell wire.Cell
		}{{"a", wire.Cell{Type: wire.OperateTypeU8, U: 5}}, {"v", wire.Cell{Type: wire.OperateTypeU8, U: 4}}} {
			wantRow = append(wantRow, byte(len(c.name)))
			wantRow = append(wantRow, c.name...)
			wantRow = wire.AppendTaggedCell(wantRow, c.cell)
		}
		if !bytes.Equal(res.Values[1], wantRow) {
			t.Fatalf("row VALUE\n got %x\nwant %x", res.Values[1], wantRow)
		}
		if !bytes.Equal(res.Values[2], []byte{wire.OperateTypeUnset}) {
			t.Fatalf("absent field VALUE %x", res.Values[2])
		}
		c, _, _ := wire.DecodeTaggedCell(res.Values[3])
		if c.U != 1 {
			t.Fatalf("record COUNT %d, want 1 dynamic field", c.U)
		}
		if c, _, _ = wire.DecodeTaggedCell(res.Values[4]); c.U != 0 {
			t.Fatalf("COUNT of an absent field %d, want 0", c.U)
		}
		if c, _, _ = wire.DecodeTaggedCell(res.Values[5]); c.U != 1 {
			t.Fatalf("COUNT of a table %d, want 1 row", c.U)
		}
	})

	sub(t, "unstamped STAMP writes 0", func(t *testing.T) {
		s := sessionSchema()
		rec, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpSTAMP, opFromSchema, colPath(3, keyU64(1), 2), 0).withAux(wire.OperateStampS),
			op(wire.OperateOpSTAMP, opFromSchema, fieldPath(2), 0).withAux(wire.OperateStampMs)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[3].Table.Rows[0].Cols[2].Cell.U != 0 || rec.Fields[2].Cell.U != 0 {
			t.Fatalf("unstamped STAMP must write 0: %+v", rec.Fields)
		}
	})

	sub(t, "determinism: op order never leaks into the stored bytes", func(t *testing.T) {
		s := sessionSchema()
		mk := func(keys []uint64) *wire.OperateArgs {
			ops := make([]oper, 0, len(keys))
			for _, k := range keys {
				ops = append(ops, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1))
			}
			return schemaArgs(s, ops...)
		}
		a, _, err := apply(nil, mk([]uint64{9, 1, 5}), 0)
		if err != nil {
			t.Fatal(err)
		}
		b, _, err := apply(nil, mk([]uint64{5, 9, 1}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a.Encode(), b.Encode()) {
			t.Fatal("insert order changed the stored bytes")
		}
	})

	sub(t, "dynamic: determinism across field, column and row order", func(t *testing.T) {
		mk := func(names []string) *wire.OperateArgs {
			ops := make([]oper, 0, len(names))
			for i, n := range names {
				ops = append(ops, op(wire.OperateOpSET, wire.OperateTypeU8, namePath(n), int64(i)))
			}
			return dynArgs(ops...)
		}
		a, _, err := apply(nil, mk([]string{"z", "a", "m"}), 0)
		if err != nil {
			t.Fatal(err)
		}
		b, _, err := apply(nil, mk([]string{"a", "m", "z"}), 0)
		if err != nil {
			t.Fatal(err)
		}
		// The values differ (they are the op index), so compare the field
		// order, not the bytes.
		var an, bn []string
		for i := range a.Fields {
			an = append(an, a.Fields[i].Name)
			bn = append(bn, b.Fields[i].Name)
		}
		if !reflect.DeepEqual(an, []string{"a", "m", "z"}) || !reflect.DeepEqual(bn, an) {
			t.Fatalf("dynamic fields must be sorted by name: %v %v", an, bn)
		}
	})

	sub(t, "record EXISTS and ABSENT distinguish a created record", func(t *testing.T) {
		// Ruling: the record path's presence is whether the record existed
		// before this call, which is the only reading under which §2.4's
		// "() is valid for CHECK EXISTS/ABSENT" says anything — a call that
		// reaches the op list always has a record by then.
		s := sessionSchema()
		mk := func(cmp uint8) *wire.OperateArgs {
			return schemaArgs(s, op(wire.OperateOpCHECK, opFromSchema, recPath(), 0).withAux(cmp))
		}
		rec, res, err := apply(nil, mk(wire.OperateCmpAbsent), 0)
		if err != nil || res.Status != wire.OperateStatusOK {
			t.Fatalf("ABSENT on a record this call created: %v %+v", err, res)
		}
		if rec == nil {
			t.Fatal("record not created")
		}
		if _, res, err = apply(nil, mk(wire.OperateCmpExists), 0); err != nil || res.Status != wire.OperateStatusCheckFailed {
			t.Fatalf("EXISTS on a record this call created: %v %+v", err, res)
		}
		if _, res, err = apply(rec, mk(wire.OperateCmpExists), 0); err != nil || res.Status != wire.OperateStatusOK {
			t.Fatalf("EXISTS on a stored record: %v %+v", err, res)
		}
		if _, res, err = apply(rec, mk(wire.OperateCmpAbsent), 0); err != nil || res.Status != wire.OperateStatusCheckFailed {
			t.Fatalf("ABSENT on a stored record: %v %+v", err, res)
		}
	})

	sub(t, "dynamic: a position segment is rejected", func(t *testing.T) {
		rec, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath("x"), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		// Positions are unstable in dynamic mode: fields are kept sorted by
		// name, so only names address them (design doc §2.4).
		if _, _, err := apply(rec, dynArgs(op(wire.OperateOpADD, wire.OperateTypeU8, fieldPath(0), 1)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("field position: %v", err)
		}
		if _, _, err := apply(rec, dynArgs(op(wire.OperateOpADD, wire.OperateTypeU8, colPath(0, keyU64(1), 0), 1)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("column position: %v", err)
		}
	})

	sub(t, "schema mode floats and the variable-length tail", func(t *testing.T) {
		s := floatSchema()
		rec, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpSET, opFromSchema, fieldPath(0), f64(1.1)),
			op(wire.OperateOpADD, opFromSchema, fieldPath(1), f64(2.5)),
			op(wire.OperateOpADD, opFromSchema, fieldPath(1), f64(2.5)),
			op(wire.OperateOpSET, opFromSchema, fieldPath(2), 0).withBytes([]byte("hello")),
			op(wire.OperateOpSET, opFromSchema, fieldPath(3), 0).withBytes([]byte("DE")),
			op(wire.OperateOpADD, opFromSchema, fieldPath(4), 300)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[0].Cell.F != float64(float32(1.1)) {
			t.Fatalf("F32 must be stored rounded: %v", rec.Fields[0].Cell.F)
		}
		if rec.Fields[1].Cell.F != 5 || string(rec.Fields[2].Cell.B) != "hello" ||
			string(rec.Fields[3].Cell.B) != "DE" || rec.Fields[4].Cell.U != 300 {
			t.Fatalf("%+v", rec.Fields)
		}
		rec, _, err = apply(rec, schemaArgs(s, op(wire.OperateOpMAX, opFromSchema, fieldPath(2), 0).withBytes([]byte("zz"))), 0)
		if err != nil {
			t.Fatal(err)
		}
		if string(rec.Fields[2].Cell.B) != "zz" {
			t.Fatalf("bytewise MAX: %q", rec.Fields[2].Cell.B)
		}
		// An op that cannot apply to the field's type is ErrOperateType
		// (design doc §2.4), and a FIXED operand must be exactly N bytes.
		for _, tc := range []struct {
			name string
			args *wire.OperateArgs
		}{
			{"shift on a float", schemaArgs(s, op(wire.OperateOpSHL, opFromSchema, fieldPath(0), 1))},
			{"add on bytes", schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(2), 1))},
			{"shift on a varint", schemaArgs(s, op(wire.OperateOpSHR, opFromSchema, fieldPath(4), 1))},
			{"FIXED operand of the wrong width", schemaArgs(s, op(wire.OperateOpSET, opFromSchema, fieldPath(3), 0).withBytes([]byte("DEU")))},
		} {
			if _, _, err := apply(rec, tc.args, 0); !errors.Is(err, wire.ErrOperateType) {
				t.Errorf("%s: err=%v, want ErrOperateType", tc.name, err)
			}
		}
	})

	sub(t, "every scalar opcode reaches the arithmetic", func(t *testing.T) {
		s := sessionSchema()
		rec, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpSET, opFromSchema, fieldPath(0), 3),
			op(wire.OperateOpMUL, opFromSchema, fieldPath(0), 4), // 12
			op(wire.OperateOpMIN, opFromSchema, fieldPath(0), 5), // 5
			op(wire.OperateOpSET, opFromSchema, fieldPath(2), 0b1010),
			op(wire.OperateOpSHL, opFromSchema, fieldPath(2), 2), // 0b101000
			op(wire.OperateOpOR, opFromSchema, fieldPath(2), 1),  // 0b101001
			op(wire.OperateOpAND, opFromSchema, fieldPath(2), 0b111100),
			op(wire.OperateOpXOR, opFromSchema, fieldPath(2), 0b000100)), 0)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Fields[0].Cell.U != 5 {
			t.Fatalf("MUL/MIN: %d", rec.Fields[0].Cell.U)
		}
		// 0b1010 <<2 = 0b101000, |1 = 0b101001, &0b111100 = 0b101000, ^0b100 = 0b101100.
		if rec.Fields[2].Cell.U != 0b101100 {
			t.Fatalf("SHL/OR/AND/XOR: %b", rec.Fields[2].Cell.U)
		}
	})

	sub(t, "a table compares by its row count", func(t *testing.T) {
		s := sessionSchema()
		var rec *wire.Record
		var err error
		for k := uint64(1); k <= 2; k++ {
			rec, _, err = apply(rec, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(k), 0), 1)), 0)
			if err != nil {
				t.Fatal(k, err)
			}
		}
		// "if the table has >= 2 rows" is a plain IF (design doc §3.3).
		got, _, err := apply(rec, schemaArgs(s,
			op(wire.OperateOpIF, opFromSchema, fieldPath(3), 2).withAux(wire.OperateCmpGE).withB(1),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
		if err != nil || got.Fields[0].Cell.U != 1 {
			t.Fatalf("%v %+v", err, got.Fields[0].Cell)
		}
		_, res, err := apply(rec, schemaArgs(s, op(wire.OperateOpCHECK, opFromSchema, fieldPath(3), 3).withAux(wire.OperateCmpEQ)), 0)
		if err != nil || res.Status != wire.OperateStatusCheckFailed {
			t.Fatalf("%v %+v", err, res)
		}
	})

	sub(t, "maxOps and maxRet are enforced", func(t *testing.T) {
		s := sessionSchema()
		many := make([]oper, wire.OperateMaxOps+1)
		for i := range many {
			many[i] = op(wire.OperateOpADD, opFromSchema, fieldPath(0), 0)
		}
		if _, _, err := apply(nil, schemaArgs(s, many...), 0); !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("maxOps: %v", err)
		}
		a := schemaArgs(s)
		for i := 0; i <= wire.OperateMaxRet; i++ {
			a.Rets = append(a.Rets, wire.OperateRet{Mode: wire.OperateRetCount, Path: fieldPath(0)})
		}
		if _, _, err := apply(nil, a, 0); !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("maxRet: %v", err)
		}
	})

	sub(t, "DEL () is terminal and later ops do not resurrect the record", func(t *testing.T) {
		// Ruling: DEL () ends the op list. An engine that instead cleared the
		// record and let a later write refill it would return a live record
		// here.
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 5)), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, res, err := apply(base, withCount(withRet(schemaArgs(s,
			op(wire.OperateOpDEL, opFromSchema, recPath(), 0),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), fieldPath(0)), recPath()), 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != nil {
			t.Fatalf("an op after DEL () resurrected the record: %+v", got.Fields)
		}
		if !bytes.Equal(res.Values[0], []byte{wire.OperateTypeUnset}) {
			t.Fatalf("VALUE after DEL (): %x, want UNSET", res.Values[0])
		}
		if c, _, _ := wire.DecodeTaggedCell(res.Values[1]); c.U != 0 {
			t.Fatalf("COUNT after DEL (): %d, want 0", c.U)
		}
	})

	sub(t, "create NONE ignores a schema blob", func(t *testing.T) {
		// Ruling: §3.5 says a create = NONE call carries no blob, so one that
		// arrives anyway is ignored — not checked against the stored version.
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		other := sessionSchema()
		other.Version = 9
		other.StoreNames = true
		a := noneArgs(op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1))
		a.Schema = other.Encode()
		got, _, err := apply(base, a, 0)
		if err != nil {
			t.Fatalf("create NONE must ignore the blob: %v", err)
		}
		if got.Fields[0].Cell.U != 2 || got.Schema.Version != 1 {
			t.Fatalf("%d v%d", got.Fields[0].Cell.U, got.Schema.Version)
		}
	})

	sub(t, "create SCHEMA compares the version, not the blob bytes", func(t *testing.T) {
		// Ruling: §2.8 matches versions. The stored schema stays
		// authoritative, so a call may carry a differently encoded blob of the
		// same version without changing the record.
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		named := sessionSchema()
		named.StoreNames = true // same version, different bytes
		if bytes.Equal(named.Encode(), s.Encode()) {
			t.Fatal("the two blobs must differ for this test to mean anything")
		}
		got, _, err := apply(base, schemaArgs(named, op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
		if err != nil {
			t.Fatalf("same version, different blob: %v", err)
		}
		if got.Fields[0].Cell.U != 2 {
			t.Fatalf("rc=%d", got.Fields[0].Cell.U)
		}
		// The stored schema won: it still does not store names, so a name
		// segment is still rejected.
		if _, _, err := apply(got, schemaArgs(named, op(wire.OperateOpADD, opFromSchema, namePath("rc"), 1)), 0); !errors.Is(err, wire.ErrOperatePath) {
			t.Fatalf("the call's blob replaced the stored schema: %v", err)
		}
	})

	sub(t, "control and table ops ignore their type byte", func(t *testing.T) {
		// Ruling: IF, CHECK, DEL, TRIM, CONFIG and MIGRATE neither read nor
		// write a typed value, so a type byte that disagrees with the schema
		// is not an error for them. (CONFIG's half is in the dynamic
		// TRIM/CONFIG sub-test; MIGRATE's is in the MIGRATE sub-test, which
		// passes 0xFF against a table-bearing schema.)
		s := sessionSchema()
		base, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 5),
			op(wire.OperateOpADD, opFromSchema, colPath(3, keyU64(1), 0), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := apply(base, schemaArgs(s,
			op(wire.OperateOpIF, wire.OperateTypeF64, fieldPath(0), 1).withAux(wire.OperateCmpGE).withB(1),
			op(wire.OperateOpADD, opFromSchema, fieldPath(0), 1)), 0)
		if err != nil || got.Fields[0].Cell.U != 6 {
			t.Fatalf("IF: %v %+v", err, got.Fields[0].Cell)
		}
		_, res, err := apply(base, schemaArgs(s,
			op(wire.OperateOpCHECK, wire.OperateTypeBytes, fieldPath(0), 1).withAux(wire.OperateCmpGE)), 0)
		if err != nil || res.Status != wire.OperateStatusOK {
			t.Fatalf("CHECK: %v %+v", err, res)
		}
		got, _, err = apply(base, schemaArgs(s, op(wire.OperateOpDEL, wire.OperateTypeF32, fieldPath(0), 0)), 0)
		if err != nil || got.Fields[0].Cell.U != 0 {
			t.Fatalf("DEL: %v %+v", err, got.Fields[0].Cell)
		}
		got, _, err = apply(base, schemaArgs(s,
			op(wire.OperateOpTRIM, wire.OperateTypeBytes, fieldPath(3), 0).withAux(wire.OperatePolicyMinKey)), 0)
		if err != nil || len(got.Fields[3].Table.Rows) != 0 {
			t.Fatalf("TRIM: %v %+v", err, got.Fields[3].Table)
		}
	})

	sub(t, "dynamic: an absent target reads as the zero of the op's type byte", func(t *testing.T) {
		// Ruling: the op's type byte picks the domain an absent dynamic target
		// compares in. Here the same comparison is true as bytes (empty equals
		// empty) and false as an integer (0 != 5).
		mk := func(typ uint8) *wire.OperateArgs {
			return dynArgs(
				op(wire.OperateOpSET, wire.OperateTypeU8, namePath("base"), 1),
				op(wire.OperateOpIF, typ, namePath("x"), 5).withAux(wire.OperateCmpEQ).withB(1),
				op(wire.OperateOpSET, wire.OperateTypeU8, namePath("hit"), 1))
		}
		asBytes, _, err := apply(nil, mk(wire.OperateTypeBytes), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(asBytes.Fields) != 2 {
			t.Fatalf("empty BYTES should equal the empty operand: %+v", asBytes.Fields)
		}
		asInt, _, err := apply(nil, mk(wire.OperateTypeU8), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(asInt.Fields) != 1 {
			t.Fatalf("integer zero should not equal 5: %+v", asInt.Fields)
		}
	})

	sub(t, "dynamic: MIN_COL orders mixed column types by domain", func(t *testing.T) {
		// Ruling: in dynamic mode two rows may store different types under one
		// column name. They order by domain rank (int < float < bytes) so the
		// victim stays a pure function of the stored bytes.
		mk := func(a, b oper) []uint64 {
			rec, _, err := apply(nil, dynArgs(
				op(wire.OperateOpCONFIG, opFromSchema, namePath("ev"), 2).
					withAux(wire.OperatePolicyMinCol).withBytes([]byte("s")),
				a, b), 0)
			if err != nil {
				t.Fatal(err)
			}
			rec, _, err = apply(rec, dynArgs(
				op(wire.OperateOpSET, wire.OperateTypeU32, nameColPath("ev", keyU64(3), "s"), 1)), 0)
			if err != nil {
				t.Fatal(err)
			}
			return rowKeys(rec.Fields[0].Table)
		}
		// Row 1 holds an int, row 2 a float: the int row goes even though 999
		// is numerically the larger value.
		keys := mk(
			op(wire.OperateOpSET, wire.OperateTypeU32, nameColPath("ev", keyU64(1), "s"), 999),
			op(wire.OperateOpSET, wire.OperateTypeF64, nameColPath("ev", keyU64(2), "s"), f64(1)))
		if !reflect.DeepEqual(keys, []uint64{2, 3}) {
			t.Fatalf("int must sort below float: %v", keys)
		}
		// Row 1 holds bytes, row 2 an int: the int row goes.
		keys = mk(
			op(wire.OperateOpSET, wire.OperateTypeBytes, nameColPath("ev", keyU64(1), "s"), 0).withBytes([]byte("a")),
			op(wire.OperateOpSET, wire.OperateTypeU32, nameColPath("ev", keyU64(2), "s"), 5))
		if !reflect.DeepEqual(keys, []uint64{1, 3}) {
			t.Fatalf("int must sort below bytes: %v", keys)
		}
	})

	sub(t, "dynamic: TRIM on an absent table field is a no-op", func(t *testing.T) {
		// Ruling: TRIM is a shrink, not a write, so it does not vivify.
		rec, _, err := apply(nil, dynArgs(op(wire.OperateOpSET, wire.OperateTypeU8, namePath("a"), 1)), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := apply(rec, dynArgs(
			op(wire.OperateOpTRIM, opFromSchema, namePath("ev"), 1).withAux(wire.OperatePolicyMinKey)), 0)
		if err != nil {
			t.Fatalf("TRIM of an absent table field: %v", err)
		}
		if len(got.Fields) != 1 || got.Fields[0].Name != "a" {
			t.Fatalf("TRIM vivified a table: %+v", got.Fields)
		}
	})

	sub(t, "an over-large record is a cap error with the record unchanged", func(t *testing.T) {
		// §2.7's maxRecordBytes backstop. The bound is lowered rather than
		// building 16 MiB of record; it is package-level state, so no test in
		// this file may run in parallel.
		restore := maxOperateRecordBytes
		maxOperateRecordBytes = 512
		t.Cleanup(func() { maxOperateRecordBytes = restore })

		s := floatSchema()
		small, _, err := apply(nil, schemaArgs(s,
			op(wire.OperateOpSET, opFromSchema, fieldPath(2), 0).withBytes(bytes.Repeat([]byte{7}, 64))), 0)
		if err != nil || small == nil {
			t.Fatalf("a record under the bound must still apply: %v", err)
		}
		before := small.Encode()
		got, _, err := apply(small, schemaArgs(s,
			op(wire.OperateOpSET, opFromSchema, fieldPath(2), 0).withBytes(bytes.Repeat([]byte{7}, 1024))), 0)
		if !errors.Is(err, wire.ErrOperateCap) {
			t.Fatalf("err=%v, want ErrOperateCap", err)
		}
		if got != nil && !bytes.Equal(got.Encode(), before) {
			t.Fatal("record changed by a call that hit the record-size cap")
		}
	})

	sub(t, "TTL modes are reported", func(t *testing.T) {
		// The applier signature carries no expiry: TTL/ttlMode live in the
		// handler around it (design doc §3.5) and are covered by the handler
		// task's tests, not by the semantics oracle.
		t.Skip("ttlMode is a handler concern; covered by the operate handler tests")
	})
}
