// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"strings"
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
// whose byCol is empty, in dynamic mode (where the wire itself would
// otherwise accept byColLen 0 and silently mean column 0) and in schema mode
// (where an empty byCol would otherwise resolve nothing at all).
//
// The schema-mode case uses Trim, not Config. Config in schema mode is
// refused by the mode guard before the byCol rule is ever consulted, so a
// Config sub-case here would pass for the wrong reason and keep passing if
// the byCol rule were deleted; Trim is valid in both modes, so it reaches the
// rule under test.
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
	_, err := NewOperate([]byte("k")).WithSchema(s).
		Trim(F("b"), 10, wire.OperatePolicyMinCol, "").Args()
	if err == nil {
		t.Fatal("expected error for Trim with MinCol policy and empty byCol (schema mode)")
	}
	if !strings.Contains(err.Error(), "column name") {
		t.Fatalf("err = %v, want the *_COL byCol rule, not some earlier guard", err)
	}

	// The same Trim with a real column name builds, so the case above is
	// failing on the empty byCol and not on the path or the mode.
	if _, err := NewOperate([]byte("k")).WithSchema(s).
		Trim(F("b"), 10, wire.OperatePolicyMinCol, "c").Args(); err != nil {
		t.Fatalf("Trim with MinCol and a real byCol (schema mode): %v", err)
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

// TestOperateBuilderModesAreExclusive covers the create-mode setters in both
// orders. Before this guard, a builder that had been through both shipped a
// DYNAMIC create byte alongside a schema blob and schema-position paths: the
// wire decoder accepts that frame, so nothing rejected it until the dynamic
// engine failed every path with wire.ErrOperatePath, a round trip later.
func TestOperateBuilderModesAreExclusive(t *testing.T) {
	s := &wire.Schema{Version: 1, Fields: []wire.FieldDef{{Name: "hits", Type: wire.OperateTypeU64}}}

	got, err := NewOperate([]byte("k")).WithSchema(s).Dynamic().
		Add(F("hits"), 1).Args()
	if err == nil {
		t.Fatalf("WithSchema().Dynamic() built a call: %+v", got)
	}
	if !strings.Contains(err.Error(), "schema mode") {
		t.Fatalf("err = %v, want it to name the mode already set", err)
	}

	got, err = NewOperate([]byte("k")).Dynamic().WithSchema(s).
		AddT(F("hits"), wire.OperateTypeU64, 1).Args()
	if err == nil {
		t.Fatalf("Dynamic().WithSchema() built a call: %+v", got)
	}
	if !strings.Contains(err.Error(), "dynamic mode") {
		t.Fatalf("err = %v, want it to name the mode already set", err)
	}

	// Neither guard fires when only one of the two is used, nor when the same
	// one is used twice.
	if _, err := NewOperate([]byte("k")).WithSchema(s).WithSchema(s).
		Add(F("hits"), 1).Args(); err != nil {
		t.Fatalf("WithSchema twice: %v", err)
	}
	if _, err := NewOperate([]byte("k")).Dynamic().Dynamic().
		AddT(F("hits"), wire.OperateTypeU64, 1).Args(); err != nil {
		t.Fatalf("Dynamic twice: %v", err)
	}
}

// TestOperateBuilderColumnNeedsRowKey covers a Path carrying a column but no
// row key. A column belongs to a row (design doc §2.4), and the wire form
// for such a path is OperatePathField, which DROPS the column and aims the
// op at the field itself — in dynamic mode a SET meant for one column would
// overwrite a scalar field of the same name.
func TestOperateBuilderColumnNeedsRowKey(t *testing.T) {
	orphan := Path{Field: "b", Col: "c", HasCol: true}

	got, err := NewOperate([]byte("k")).Dynamic().SetT(orphan, wire.OperateTypeU64, 1).Args()
	if err == nil {
		t.Fatalf("a column path without a key built a call: %+v", got)
	}
	if !strings.Contains(err.Error(), `"c"`) || !strings.Contains(err.Error(), `"b"`) {
		t.Fatalf("err = %v, want it to name the column and the field", err)
	}

	// A return spec goes through the same resolver, so it is caught too.
	if _, err := NewOperate([]byte("k")).Dynamic().Return(orphan).Args(); err == nil {
		t.Fatal("a column ret spec without a key built a call")
	}

	// The well-formed version of the same path still builds, as a Col path.
	args, err := NewOperate([]byte("k")).Dynamic().
		SetT(Col("b", []byte("r"), "c"), wire.OperateTypeU64, 1).Args()
	if err != nil {
		t.Fatalf("Col(field, key, col): %v", err)
	}
	if args.Ops[0].Path.Kind != wire.OperatePathCol || args.Ops[0].Path.Col.Name != "c" {
		t.Fatalf("path = %+v", args.Ops[0].Path)
	}
}

// TestOperateNilArgs covers Client.Operate against a nil args pointer, which
// is the shape a caller lands on by ignoring OperateBuilder.Args's error.
// EncodeOperateArgs dereferences it, so this used to panic inside the client.
func TestOperateNilArgs(t *testing.T) {
	c, err := New(Config{Servers: []string{"127.0.0.1:1"}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	res, err := c.Operate(context.Background(), nil)
	if !errors.Is(err, wire.ErrOperateArgs) {
		t.Fatalf("err = %v, want wire.ErrOperateArgs", err)
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}
}

// TestDecodeOperateValueTrailingBytes covers the "one tagged cell, exactly"
// rule. A value that decodes and leaves bytes over is not the value it claims
// to be — a table's or a row's encoding read as a scalar, or a corrupt
// frame — and returning its first cell would hand the caller a prefix of
// something else.
func TestDecodeOperateValueTrailingBytes(t *testing.T) {
	one := wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU16, U: 7})
	if c, err := DecodeOperateValue(one); err != nil || c.U != 7 {
		t.Fatalf("a single tagged cell must decode: %+v %v", c, err)
	}
	two := wire.AppendTaggedCell(append([]byte(nil), one...), wire.Cell{Type: wire.OperateTypeU8, U: 1})
	if _, err := DecodeOperateValue(two); !errors.Is(err, wire.ErrOperateArgs) {
		t.Fatalf("two cells: err = %v, want wire.ErrOperateArgs", err)
	}
	if _, err := DecodeOperateValue(append(append([]byte(nil), one...), 0)); err == nil {
		t.Fatal("a single trailing byte was accepted")
	}
	// The absent-value convention is one byte and still decodes.
	if c, err := DecodeOperateValue([]byte{wire.OperateTypeUnset}); err != nil || c.Type != wire.OperateTypeUnset {
		t.Fatalf("UNSET: %+v %v", c, err)
	}
}

// TestOperateBuilderMigrateMustBeFirst covers MIGRATE's position. applyOps
// rejects a MIGRATE at any index but 0 with wire.ErrOperateOpcode (design doc
// §2.8), so building one after another op can only ever produce a request the
// server refuses.
func TestOperateBuilderMigrateMustBeFirst(t *testing.T) {
	s := &wire.Schema{Version: 2, Fields: []wire.FieldDef{{Name: "rc", Type: wire.OperateTypeU64}}}

	_, err := NewOperate([]byte("k")).Dynamic().
		AddT(F("rc"), wire.OperateTypeU64, 1).
		Migrate(wire.OperateMigrateFromDynamic, s, false).
		Args()
	if err == nil {
		t.Fatal("Migrate after another op built a call")
	}
	if !strings.Contains(err.Error(), "first op") {
		t.Fatalf("err = %v, want it to name the position rule", err)
	}

	// Two MIGRATEs is the same rule: the second is not the first op.
	if _, err := NewOperate([]byte("k")).Dynamic().
		Migrate(wire.OperateMigrateFromDynamic, s, false).
		Migrate(wire.OperateMigrateFromDynamic, s, false).Args(); err == nil {
		t.Fatal("a second Migrate built a call")
	}

	// Migrate first is fine, and stays op 0.
	args, err := NewOperate([]byte("k")).Dynamic().
		Migrate(wire.OperateMigrateFromDynamic, s, false).
		Add(F("rc"), 1).Args()
	if err != nil {
		t.Fatalf("Migrate first: %v", err)
	}
	if args.Ops[0].Opcode != wire.OperateOpMIGRATE {
		t.Fatalf("op 0 = %d, want MIGRATE", args.Ops[0].Opcode)
	}
}

// TestOperateBuilderMigrateInstallsResolutionSchema covers what a MIGRATE
// does to name resolution. It is the first op, so every other op and every
// return spec runs against the record as the migration leaves it, and their
// Field/Col names must resolve against the TARGET schema — by position, with
// Type OperateTypeFromSchema — not as dynamic name segments.
func TestOperateBuilderMigrateInstallsResolutionSchema(t *testing.T) {
	s := sessionSchema()
	args, err := NewOperate([]byte("k")).Dynamic().
		Migrate(wire.OperateMigrateFromDynamic, s, false).
		Add(F("rc"), 1).
		Count(F("b")).
		Args()
	if err != nil {
		t.Fatalf("Args: %v", err)
	}
	if args.Create != wire.OperateCreateDynamic {
		t.Fatalf("create = %d, want CreateDynamic; a freeze does not change the create mode", args.Create)
	}
	// The call's own schema blob stays empty: the server checks the blob
	// against the version the record is stored at when it OPENS it, and this
	// record is dynamic. The target schema rides on the MIGRATE op instead.
	if len(args.Schema) != 0 {
		t.Fatalf("args.Schema = %x, want empty for a dynamic-create call", args.Schema)
	}
	add := args.Ops[1]
	if add.Path.Field.ByName || add.Path.Field.Pos != 0 {
		t.Fatalf("Add path = %+v, want position 0 (rc) resolved against the target schema", add.Path.Field)
	}
	if add.Type != wire.OperateTypeFromSchema {
		t.Fatalf("Add type = %d, want OperateTypeFromSchema", add.Type)
	}
	if ret := args.Rets[0]; ret.Path.Field.ByName || ret.Path.Field.Pos != 3 {
		t.Fatalf("ret path = %+v, want position 3 (b)", ret.Path.Field)
	}

	// A schema-mode evolution keeps shipping the version the record is stored
	// at while resolving later names against the target.
	v1 := sessionSchema()
	v2 := sessionSchema()
	v2.Version = 2
	v2.Fields = append(v2.Fields, wire.FieldDef{Name: "extra", Type: wire.OperateTypeU64})
	args, err = NewOperate([]byte("k")).WithSchema(v1).
		Migrate(1, v2, false).
		Add(F("extra"), 1).
		Args()
	if err != nil {
		t.Fatalf("schema-mode evolution: %v", err)
	}
	if !bytes.Equal(args.Schema, v1.Encode()) {
		t.Fatal("the call's schema blob must stay the version the record is stored at")
	}
	if got := args.Ops[1].Path.Field.Pos; got != 4 {
		t.Fatalf("Add path pos = %d, want 4 (the field only v2 declares)", got)
	}
}

// TestVectorOperateNonReplayable checks that the three vector operate ops are
// registered as non-replayable alongside the KV "operate". They are the SAME
// non-idempotent op-list applied to a record held in a point's payload, so a
// blind replay after an ambiguous post-commit failure would apply every ADD
// twice — the exact hole the KV entry exists to close.
func TestVectorOperateNonReplayable(t *testing.T) {
	for _, op := range []string{"vector_operate", "vector_named_operate", "vector_mv_operate"} {
		if !nonReplayableOp(op) {
			t.Errorf("nonReplayableOp(%q) = false, want true", op)
		}
	}
	// A read-only vector op must NOT be caught by the same switch.
	if nonReplayableOp("vector_get") {
		t.Error("nonReplayableOp(\"vector_get\") = true, want false")
	}
}
