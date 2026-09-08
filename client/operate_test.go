// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// sessionSchema mirrors sdk/wire/operate_schema_test.go's helper of the same
// name (design doc §4's v1-session-record-in-v2 worked example), qualified
// with wire. since this is a different package.
func sessionSchema() *wire.Schema {
	return &wire.Schema{Version: 1, Fields: []wire.FieldDef{
		{Name: "rc", Type: wire.OperateTypeU8}, {Name: "bc", Type: wire.OperateTypeU8}, {Name: "hist", Type: wire.OperateTypeU32},
		{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{KeyType: wire.OperateTypeU64, Cols: []wire.ColumnDef{
			{Name: "c", Type: wire.OperateTypeU16}, {Name: "hi", Type: wire.OperateTypeI64}, {Name: "t", Type: wire.OperateTypeU32}},
			Cap: 1024, Policy: wire.OperatePolicyMinCol, ByCol: 2}},
	}}
}

// TestOperateBuilderSchemaMode builds design doc §4's session-record worked
// example through the builder and checks the resolved wire.OperateArgs
// against a hand-written one: positions resolved from names, Type ==
// OperateTypeFromSchema on every untyped scalar op, and a bare zero Type on
// the control op (IF never declares one, design doc §2.4).
func TestOperateBuilderSchemaMode(t *testing.T) {
	s := sessionSchema()
	key := []byte("session:42")
	id := uint64(7)
	rowKey := KeyU64(id)
	const bid = int64(99)
	const won = int64(1)

	got, err := NewOperate(key).WithSchema(s).
		TTL(90*time.Second, wire.OperateTTLCreateOnly).
		If(F("rc"), wire.OperateCmpGE, 255, 2).
		Shr(F("rc"), 1).
		Shr(F("bc"), 1).
		Add(F("rc"), 1).
		Add(Col("b", rowKey, "c"), 1).
		Max(Col("b", rowKey, "hi"), bid).
		Stamp(Col("b", rowKey, "t"), wire.OperateStampS).
		Shl(F("hist"), 1).
		Or(F("hist"), won).
		Return(F("rc")).
		Return(Row("b", rowKey)).
		Args()
	if err != nil {
		t.Fatalf("Args: %v", err)
	}

	want := &wire.OperateArgs{
		Key:     key,
		TTL:     90 * time.Second,
		TTLMode: wire.OperateTTLCreateOnly,
		Create:  wire.OperateCreateSchema,
		Schema:  s.Encode(),
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpIF, Aux: wire.OperateCmpGE,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 0}}, A: 255, B: 2},
			{Opcode: wire.OperateOpSHR, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 0}}, A: 1},
			{Opcode: wire.OperateOpSHR, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 1}}, A: 1},
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 0}}, A: 1},
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathCol, Field: wire.OperateSeg{Pos: 3}, Key: rowKey, Col: wire.OperateSeg{Pos: 0}}, A: 1},
			{Opcode: wire.OperateOpMAX, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathCol, Field: wire.OperateSeg{Pos: 3}, Key: rowKey, Col: wire.OperateSeg{Pos: 1}}, A: bid},
			{Opcode: wire.OperateOpSTAMP, Type: wire.OperateTypeFromSchema, Aux: wire.OperateStampS,
				Path: wire.OperatePath{Kind: wire.OperatePathCol, Field: wire.OperateSeg{Pos: 3}, Key: rowKey, Col: wire.OperateSeg{Pos: 2}}},
			{Opcode: wire.OperateOpSHL, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 2}}, A: 1},
			{Opcode: wire.OperateOpOR, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 2}}, A: won},
		},
		Rets: []wire.OperateRet{
			{Mode: wire.OperateRetValue, Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: 0}}},
			{Mode: wire.OperateRetValue, Path: wire.OperatePath{Kind: wire.OperatePathRow, Field: wire.OperateSeg{Pos: 3}, Key: rowKey}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("\n got %+v\nwant %+v", got, want)
	}
}

// TestOperateBuilderSchemaErrors checks the three resolution failures Args
// must report against a schema: an unknown field name, a row key of the
// wrong width for the table's key type, and a column path targeting a
// non-table field.
func TestOperateBuilderSchemaErrors(t *testing.T) {
	s := sessionSchema()

	if _, err := NewOperate([]byte("k")).WithSchema(s).Return(F("nope")).Args(); err == nil {
		t.Fatal("expected error for unknown field name")
	}
	if _, err := NewOperate([]byte("k")).WithSchema(s).Return(Row("b", []byte{1, 2, 3})).Args(); err == nil {
		t.Fatal("expected error for wrong key width")
	}
	if _, err := NewOperate([]byte("k")).WithSchema(s).Return(Col("rc", []byte{0}, "x")).Args(); err == nil {
		t.Fatal("expected error for a column path on a non-table field")
	}
}

// TestOperateBuilderConfigTrimColPolicyGuard checks that Args rejects a
// Config/Trim call whose policy is a *_COL policy (MIN_COL/MAX_COL) but
// whose byCol is empty, both in dynamic mode (where the wire itself would
// otherwise accept byColLen 0 and silently mean column 0) and in schema
// mode (where an empty byCol would otherwise resolve nothing at all).
func TestOperateBuilderConfigTrimColPolicyGuard(t *testing.T) {
	if _, err := NewOperate([]byte("k")).Dynamic().
		Config(F("t"), 1024, wire.OperatePolicyMinCol, "").Args(); err == nil {
		t.Fatal("expected error for Config with MinCol policy and empty byCol (dynamic mode)")
	}
	if _, err := NewOperate([]byte("k")).Dynamic().
		Trim(F("t"), 10, wire.OperatePolicyMaxCol, "").Args(); err == nil {
		t.Fatal("expected error for Trim with MaxCol policy and empty byCol (dynamic mode)")
	}

	s := sessionSchema()
	if _, err := NewOperate([]byte("k")).WithSchema(s).
		Config(F("b"), 1024, wire.OperatePolicyMinCol, "").Args(); err == nil {
		t.Fatal("expected error for Config with MinCol policy and empty byCol (schema mode)")
	}

	// A non-*_COL policy still needs no byCol.
	if _, err := NewOperate([]byte("k")).Dynamic().
		Config(F("t"), 1024, wire.OperatePolicyNone, "").Args(); err != nil {
		t.Fatalf("Config with OperatePolicyNone and empty byCol: unexpected error: %v", err)
	}
}

// TestOperateBuilderDynamicMode checks that without a schema, Field/Col
// resolve to name segments, typed variants (AddT/SetBytesT) carry the given
// type, and an untyped variant (Add) still emits OperateTypeFromSchema —
// valid only against an existing target, per design doc §2.9.
func TestOperateBuilderDynamicMode(t *testing.T) {
	got, err := NewOperate([]byte("k")).Dynamic().
		AddT(F("hits"), wire.OperateTypeI64, 5).
		SetBytesT(F("name"), wire.OperateTypeBytes, []byte("v")).
		Add(F("hits"), 1).
		Args()
	if err != nil {
		t.Fatalf("Args: %v", err)
	}
	want := &wire.OperateArgs{
		Key:    []byte("k"),
		Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeI64,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: "hits"}}, A: 5},
			{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: "name"}}, Bytes: []byte("v")},
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeFromSchema,
				Path: wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: "hits"}}, A: 1},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("\n got %+v\nwant %+v", got, want)
	}
}

// TestOperateEndToEnd runs a builder-built call through a fake server: the
// server decodes the request frame's op name and args, asserts the args
// DeepEqual what the builder produced, and replies with an OperateResult;
// Client.Operate must decode that result, and DecodeOperateValue must read
// the single returned scalar back out of it.
func TestOperateEndToEnd(t *testing.T) {
	builderArgs, err := NewOperate([]byte("k")).Dynamic().
		AddT(F("hits"), wire.OperateTypeI64, 1).
		Return(F("hits")).
		Args()
	if err != nil {
		t.Fatalf("Args: %v", err)
	}

	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		opName, argsBytes := decodeOpFrame(t, body)
		if opName != "operate" {
			t.Fatalf("op = %q, want operate", opName)
		}
		gotArgs, derr := wire.DecodeOperateArgs(argsBytes)
		if derr != nil {
			t.Fatalf("DecodeOperateArgs: %v", derr)
		}
		if !reflect.DeepEqual(gotArgs, builderArgs) {
			t.Fatalf("args on the wire:\n got %+v\nwant %+v", gotArgs, builderArgs)
		}
		payload, eerr := wire.EncodeOperateResult(&wire.OperateResult{
			Status: wire.OperateStatusOK,
			Values: [][]byte{wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU8, U: 9})},
		})
		if eerr != nil {
			t.Fatalf("EncodeOperateResult: %v", eerr)
		}
		return StatusOK, payload
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	res, err := c.Operate(context.Background(), builderArgs)
	if err != nil {
		t.Fatalf("Operate: %v", err)
	}
	if res.Status != wire.OperateStatusOK || len(res.Values) != 1 {
		t.Fatalf("unexpected result: %+v", res)
	}
	cell, err := DecodeOperateValue(res.Values[0])
	if err != nil {
		t.Fatalf("DecodeOperateValue: %v", err)
	}
	if cell.Type != wire.OperateTypeU8 || cell.U != 9 {
		t.Fatalf("cell = %+v, want U8(9)", cell)
	}
}

// TestDecodeOperateValueAbsent checks the absent-value convention
// DecodeOperateValue documents: a tagged UNSET byte decodes to a Cell whose
// Type is OperateTypeUnset, not an error.
func TestDecodeOperateValueAbsent(t *testing.T) {
	c, err := DecodeOperateValue([]byte{wire.OperateTypeUnset})
	if err != nil {
		t.Fatalf("DecodeOperateValue: %v", err)
	}
	if c.Type != wire.OperateTypeUnset {
		t.Fatalf("cell = %+v, want Unset", c)
	}
}

// TestOperateNonReplayable checks that operate is registered as
// non-replayable: it mutates counters non-idempotently, so a blind replay
// after an ambiguous post-commit failure would apply every increment twice.
func TestOperateNonReplayable(t *testing.T) {
	if !nonReplayableOp("operate") {
		t.Fatal("nonReplayableOp(\"operate\") = false, want true")
	}
}

// TestOperateEncodeErrorNoNetworkCall checks that an OperateArgs which fails
// wire.EncodeOperateArgs (here, too many ops) is rejected by Client.Operate
// before any network call is attempted: the server address is a closed port
// that a dial would fail loudly on, but since pools are created lazily
// (New's MinConnsPerServer defaults to 0) the encode error must come back
// without ever touching it.
func TestOperateEncodeErrorNoNetworkCall(t *testing.T) {
	c, err := New(Config{Servers: []string{"127.0.0.1:1"}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	args := &wire.OperateArgs{Key: []byte("k"), Ops: make([]wire.OperateOp, wire.OperateMaxOps+1)}
	_, err = c.Operate(context.Background(), args)
	if err == nil {
		t.Fatal("expected an encode error")
	}
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("err = %v, want wire.ErrOperateCap", err)
	}
}

// TestOperateBuilderMigrateValidatesSchema covers Migrate's schema check.
// WithSchema validates the schema it attaches, so Migrate — which ships a
// second schema blob the server will store — must do the same, or an invalid
// target schema only surfaces after a round trip.
func TestOperateBuilderMigrateValidatesSchema(t *testing.T) {
	bad := &wire.Schema{Version: 2, Fields: []wire.FieldDef{
		{Name: "f", Type: wire.OperateTypeFixed, N: 0}, // FIXED(0) is not a width
	}}
	if err := bad.Validate(); err == nil {
		t.Fatal("the fixture schema must be invalid for this test to mean anything")
	}
	_, err := NewOperate([]byte("k")).Dynamic().
		Migrate(wire.OperateMigrateFromDynamic, bad, false).
		Args()
	if err == nil {
		t.Fatal("Migrate accepted a schema that fails Validate")
	}

	good := &wire.Schema{Version: 2, Fields: []wire.FieldDef{{Name: "f", Type: wire.OperateTypeU64}}}
	if _, err := NewOperate([]byte("k")).Dynamic().
		Migrate(wire.OperateMigrateFromDynamic, good, false).Args(); err != nil {
		t.Fatalf("a valid migration schema must build: %v", err)
	}
}

// TestOperateBuilderConfigIsDynamicOnly covers the two mode rules the
// builder can check for itself: CONFIG against a schema-mode record is
// wire.ErrOperateOpcode on the server (design doc §3.2 — a schema-mode
// table's eviction triple comes from its schema), so the call can only ever
// fail; TRIM is valid in both modes and must keep building.
func TestOperateBuilderConfigIsDynamicOnly(t *testing.T) {
	s := &wire.Schema{Version: 1, Fields: []wire.FieldDef{
		{Name: "ev", Type: wire.OperateTypeTable, Table: &wire.TableDef{
			KeyType: wire.OperateTypeU64,
			Cols:    []wire.ColumnDef{{Name: "hits", Type: wire.OperateTypeU32}},
		}},
	}}

	if _, err := NewOperate([]byte("k")).WithSchema(s).
		Config(F("ev"), 10, wire.OperatePolicyMinKey, "").Args(); err == nil {
		t.Fatal("Config built a call against a schema-mode record")
	}
	// The same op is fine without a schema.
	if _, err := NewOperate([]byte("k")).Dynamic().
		Config(F("ev"), 10, wire.OperatePolicyMinKey, "").Args(); err != nil {
		t.Fatalf("Config in dynamic mode: %v", err)
	}

	// TRIM builds in both modes.
	if _, err := NewOperate([]byte("k")).WithSchema(s).
		Trim(F("ev"), 5, wire.OperatePolicyMinCol, "hits").Args(); err != nil {
		t.Fatalf("Trim in schema mode: %v", err)
	}
	if _, err := NewOperate([]byte("k")).Dynamic().
		Trim(F("ev"), 5, wire.OperatePolicyMinCol, "hits").Args(); err != nil {
		t.Fatalf("Trim in dynamic mode: %v", err)
	}
}
