// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// Operate runs one atomic multi-field update against a's key and returns
// its result (design doc for operate v2, §3-§3.5): a generic list of
// scalar/table/control ops applied to a record, evaluated in order, with
// return specs read back in the same round trip.
//
// An error from wire.EncodeOperateArgs (a's op/ret list, key, or schema
// blob exceeds a wire cap, or carries a structurally invalid field) is
// returned unchanged, without a network round trip — build a's OperateArgs
// with OperateBuilder to avoid ever hitting one by hand. A nil a is
// wire.ErrOperateArgs rather than a panic: it is the shape a caller lands on
// by ignoring OperateBuilder.Args's error, which is exactly when a panic is
// least welcome.
func (c *Client) Operate(ctx context.Context, a *wire.OperateArgs) (*wire.OperateResult, error) {
	if a == nil {
		return nil, wire.ErrOperateArgs
	}
	args, err := wire.EncodeOperateArgs(a)
	if err != nil {
		return nil, err
	}
	res, err := c.Call(ctx, "operate", args)
	if err != nil {
		return nil, err
	}
	return wire.DecodeOperateResult(res)
}

// DecodeOperateValue decodes one scalar value from an OperateResult.Values
// entry (design doc §3.4's tagged VALUE encoding: [type u8][n iff
// FIXED][data]). An absent node ([]byte{wire.OperateTypeUnset}) decodes to
// wire.Cell{Type: wire.OperateTypeUnset} rather than an error, so a caller
// can read res.Values[i] uniformly without special-casing "not present"
// against a decode failure.
//
// The tagged cell must consume b exactly. A value that decodes and leaves
// bytes over is not the value it claims to be — a table's or a row's
// encoding read as a scalar, or a corrupt frame — and returning its first
// cell would silently hand the caller a prefix of something else.
func DecodeOperateValue(b []byte) (wire.Cell, error) {
	c, n, err := wire.DecodeTaggedCell(b)
	if err != nil {
		return wire.Cell{}, err
	}
	if n != len(b) {
		return wire.Cell{}, fmt.Errorf("client: operate: %d bytes left after the tagged cell: %w", len(b)-n, wire.ErrOperateArgs)
	}
	return c, nil
}

// Path addresses one node inside an operate record (design doc §2.4): the
// record itself (RecordPath), a record field (F), a whole row of a table
// field (Row), or one column of that row (Col). Field and Col are always
// names here; OperateBuilder.Args resolves them to schema positions when a
// schema is attached (WithSchema) and otherwise carries them as name
// segments on the wire (Dynamic).
type Path struct {
	Field  string
	Col    string
	Key    []byte
	HasKey bool
	HasCol bool
}

// F addresses the record field named field.
func F(field string) Path { return Path{Field: field} }

// Row addresses the row keyed by key in the table field named field.
func Row(field string, key []byte) Path {
	return Path{Field: field, Key: key, HasKey: true}
}

// Col addresses the column named col of the row keyed by key in the table
// field named field.
func Col(field string, key []byte, col string) Path {
	return Path{Field: field, Key: key, HasKey: true, Col: col, HasCol: true}
}

// RecordPath addresses the record itself — the only valid path for
// CHECK EXISTS/ABSENT, DEL, and a whole-record return (design doc §2.4).
func RecordPath() Path { return Path{} }

// KeyU64 encodes k as the little-endian 8-byte row key a schema-mode table
// with a U64 key type expects (design doc §2.3: numeric keys are stored
// little-endian and compared as integers).
func KeyU64(k uint64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, k)
	return b
}

// KeyU32 encodes k as the little-endian 4-byte row key a schema-mode table
// with a U32 key type expects.
func KeyU32(k uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, k)
	return b
}

// KeyU16 encodes k as the little-endian 2-byte row key a schema-mode table
// with a U16 key type expects.
func KeyU16(k uint16) []byte {
	b := make([]byte, 2)
	binary.LittleEndian.PutUint16(b, k)
	return b
}

// KeyU8 encodes k as the 1-byte row key a schema-mode table with a U8 key
// type expects.
func KeyU8(k uint8) []byte { return []byte{k} }

// floatBits is a float operand's wire encoding (design doc §2.4): always
// the float64 bit pattern, F32 fields included — storage rounds.
func floatBits(v float64) int64 {
	return int64(math.Float64bits(v)) //nolint:gosec // reinterpret float bits as int64, per the wire convention
}

// builderOp is one not-yet-resolved op: its Path (a name, not yet a
// position) and, for CONFIG/TRIM, its byCol name are resolved against the
// builder's schema (if any) in OperateBuilder.Args, once the whole call is
// known.
type builderOp struct {
	opcode, typ, aux uint8
	path             Path
	a, b             int64
	bytes            []byte
	byColName        string
}

// builderRet is one not-yet-resolved return spec.
type builderRet struct {
	mode uint8
	path Path
}

// OperateBuilder builds a wire.OperateArgs for one Client.Operate call
// (design doc §3-§4): a fluent alternative to constructing
// wire.OperateOp/wire.OperateRet by hand. With a schema attached
// (WithSchema), every Field/Col name is resolved to its schema position and
// an untyped op's Type is wire.OperateTypeFromSchema; without one
// (Dynamic), names ride the wire as name segments and a typed op's Type is
// whatever the caller passed.
//
// Every method except Args returns the builder so calls chain (see the
// package example in design doc §4). The first resolution error
// encountered — an unknown field name, a wrong-width row key, a column path
// into a non-table field, a typed op whose type disagrees with the schema —
// is recorded and returned by Args, so no intermediate call needs its own
// error check.
// schema and createSchema are deliberately two fields. createSchema is the
// blob the CALL carries, which the server checks against the version the
// record is stored at when it opens it — so it must stay the schema the
// record has NOW, and only WithSchema sets it. schema is what Field/Col
// names resolve against, which a MIGRATE moves forward: MIGRATE is the first
// op, so every other op and every return spec runs against the record as the
// migration leaves it, and must be resolved against the target schema.
type OperateBuilder struct {
	key          []byte
	schema       *wire.Schema
	createSchema *wire.Schema
	create       uint8
	ttl          time.Duration
	ttlMode      uint8
	ops          []builderOp
	rets         []builderRet
	err          error
}

// NewOperate starts building an operate call against key. The call defaults
// to wire.OperateCreateNone (valid only against an existing record) until
// WithSchema or Dynamic is called.
func NewOperate(key []byte) *OperateBuilder {
	return &OperateBuilder{key: key}
}

// WithSchema attaches s to the builder: the call creates an absent record
// in schema mode with s's encoded blob (design doc §2.8), and every
// subsequent Field/Col name is resolved against s. s is validated once,
// here; a nil s or a validation failure is recorded and returned by Args
// rather than panicking.
//
// It is exclusive with Dynamic: the two set the same `create` byte to
// opposite values, and a builder that has been through both would ship a
// dynamic call carrying a schema blob and schema-POSITION paths. The wire
// decoder accepts that frame, so nothing catches it until the dynamic
// engine rejects every path with wire.ErrOperatePath — calling the second
// one is recorded as an error here instead.
func (ob *OperateBuilder) WithSchema(s *wire.Schema) *OperateBuilder {
	if ob.create == wire.OperateCreateDynamic {
		if ob.err == nil {
			ob.err = fmt.Errorf("client: operate: builder is already in dynamic mode; WithSchema and Dynamic are exclusive")
		}
		return ob
	}
	if ob.err == nil {
		if s == nil {
			ob.err = fmt.Errorf("client: operate: WithSchema requires a non-nil schema")
			return ob
		}
		if err := s.Validate(); err != nil {
			ob.err = err
			return ob
		}
	}
	ob.schema, ob.createSchema = s, s
	ob.create = wire.OperateCreateSchema
	return ob
}

// Dynamic marks the call as creating an absent record in dynamic mode
// (design doc §2.9): no schema, every field and column named and typed by
// the ops that first touch it. It is exclusive with WithSchema, for the
// reason given there.
func (ob *OperateBuilder) Dynamic() *OperateBuilder {
	if ob.create == wire.OperateCreateSchema {
		if ob.err == nil {
			ob.err = fmt.Errorf("client: operate: builder is already in schema mode; Dynamic and WithSchema are exclusive")
		}
		return ob
	}
	ob.create = wire.OperateCreateDynamic
	return ob
}

// TTL sets the call's TTL and ttlMode (design doc §3.5: KEEP, SET, or
// CREATE_ONLY).
func (ob *OperateBuilder) TTL(d time.Duration, mode uint8) *OperateBuilder {
	ob.ttl = d
	ob.ttlMode = mode
	return ob
}

func (ob *OperateBuilder) addOp(op builderOp) *OperateBuilder {
	ob.ops = append(ob.ops, op)
	return ob
}

// Set writes v to the int/float-domain scalar at path, typed from the
// schema (design doc §3.1). Use SetT in dynamic mode to create a new
// target, or SetFloat/SetBytes for a float or bytes/fixed operand.
func (ob *OperateBuilder) Set(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSET, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// SetFloat writes v to the float scalar at path, typed from the schema.
func (ob *OperateBuilder) SetFloat(path Path, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSET, typ: wire.OperateTypeFromSchema, path: path, a: floatBits(v)})
}

// SetBytes writes v to the bytes/fixed scalar at path, typed from the
// schema.
func (ob *OperateBuilder) SetBytes(path Path, v []byte) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSET, typ: wire.OperateTypeFromSchema, path: path, bytes: v})
}

// Add adds v (saturating) to the int scalar at path, typed from the schema.
func (ob *OperateBuilder) Add(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpADD, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// AddFloat adds v (IEEE) to the float scalar at path, typed from the
// schema.
func (ob *OperateBuilder) AddFloat(path Path, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpADD, typ: wire.OperateTypeFromSchema, path: path, a: floatBits(v)})
}

// Mul multiplies (saturating) the int scalar at path by v, typed from the
// schema.
func (ob *OperateBuilder) Mul(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMUL, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// MulFloat multiplies (IEEE) the float scalar at path by v, typed from the
// schema.
func (ob *OperateBuilder) MulFloat(path Path, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMUL, typ: wire.OperateTypeFromSchema, path: path, a: floatBits(v)})
}

// Min sets the int scalar at path to min(current, v), typed from the
// schema.
func (ob *OperateBuilder) Min(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMIN, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// Max sets the int scalar at path to max(current, v), typed from the
// schema.
func (ob *OperateBuilder) Max(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMAX, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// MinFloat sets the float scalar at path to min(current, v), typed from the
// schema.
func (ob *OperateBuilder) MinFloat(path Path, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMIN, typ: wire.OperateTypeFromSchema, path: path, a: floatBits(v)})
}

// MaxFloat sets the float scalar at path to max(current, v), typed from the
// schema.
func (ob *OperateBuilder) MaxFloat(path Path, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMAX, typ: wire.OperateTypeFromSchema, path: path, a: floatBits(v)})
}

// And bitwise-ANDs the fixed-width int scalar at path with v (masked to the
// field width), typed from the schema.
func (ob *OperateBuilder) And(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpAND, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// Or bitwise-ORs the fixed-width int scalar at path with v, typed from the
// schema.
func (ob *OperateBuilder) Or(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpOR, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// Xor bitwise-XORs the fixed-width int scalar at path with v, typed from
// the schema.
func (ob *OperateBuilder) Xor(path Path, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpXOR, typ: wire.OperateTypeFromSchema, path: path, a: v})
}

// Shl shifts the fixed-width int scalar at path left by n (0..64, masked to
// the field width), typed from the schema.
func (ob *OperateBuilder) Shl(path Path, n int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSHL, typ: wire.OperateTypeFromSchema, path: path, a: n})
}

// Shr shifts the fixed-width int scalar at path right by n (0..64;
// arithmetic for signed fields, logical for unsigned), typed from the
// schema.
func (ob *OperateBuilder) Shr(path Path, n int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSHR, typ: wire.OperateTypeFromSchema, path: path, a: n})
}

// Stamp writes tx.applyStamp() (in unit's resolution — wire.OperateStampMs
// or wire.OperateStampS) to the int scalar at path, typed from the schema.
func (ob *OperateBuilder) Stamp(path Path, unit uint8) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSTAMP, typ: wire.OperateTypeFromSchema, aux: unit, path: path})
}

// Del removes path: a record field becomes UNSET, a row is removed, a table
// field is emptied, and the record path deletes the whole record (design
// doc §2.5/§3.1).
func (ob *OperateBuilder) Del(path Path) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpDEL, typ: wire.OperateTypeFromSchema, path: path})
}

// SetT writes v to the int/float-domain scalar at path, creating it with
// type typ in dynamic mode (or checking typ against the schema's type in
// schema mode).
func (ob *OperateBuilder) SetT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSET, typ: typ, path: path, a: v})
}

// SetBytesT writes v (BYTES or FIXED) to the scalar at path, creating it
// with type typ in dynamic mode (or checking typ against the schema's type
// in schema mode).
func (ob *OperateBuilder) SetBytesT(path Path, typ uint8, v []byte) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSET, typ: typ, path: path, bytes: v})
}

// AddT adds v (saturating) to the scalar at path, creating it with type typ
// in dynamic mode (or checking typ against the schema's type in schema
// mode).
func (ob *OperateBuilder) AddT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpADD, typ: typ, path: path, a: v})
}

// MulT multiplies (saturating) the scalar at path by v, creating it with
// type typ in dynamic mode (or checking typ against the schema's type in
// schema mode).
func (ob *OperateBuilder) MulT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMUL, typ: typ, path: path, a: v})
}

// MinT sets the scalar at path to min(current, v), creating it with type
// typ in dynamic mode (or checking typ against the schema's type in schema
// mode).
func (ob *OperateBuilder) MinT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMIN, typ: typ, path: path, a: v})
}

// MaxT sets the scalar at path to max(current, v), creating it with type
// typ in dynamic mode (or checking typ against the schema's type in schema
// mode).
func (ob *OperateBuilder) MaxT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMAX, typ: typ, path: path, a: v})
}

// AndT bitwise-ANDs the scalar at path with v, creating it with type typ in
// dynamic mode (or checking typ against the schema's type in schema mode).
func (ob *OperateBuilder) AndT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpAND, typ: typ, path: path, a: v})
}

// OrT bitwise-ORs the scalar at path with v, creating it with type typ in
// dynamic mode (or checking typ against the schema's type in schema mode).
func (ob *OperateBuilder) OrT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpOR, typ: typ, path: path, a: v})
}

// XorT bitwise-XORs the scalar at path with v, creating it with type typ in
// dynamic mode (or checking typ against the schema's type in schema mode).
func (ob *OperateBuilder) XorT(path Path, typ uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpXOR, typ: typ, path: path, a: v})
}

// StampT writes tx.applyStamp() (in unit's resolution) to the scalar at
// path, creating it with type typ in dynamic mode (or checking typ against
// the schema's type in schema mode).
func (ob *OperateBuilder) StampT(path Path, typ uint8, unit uint8) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSTAMP, typ: typ, aux: unit, path: path})
}

// SetFloatT writes v to the float scalar at path, creating it with type typ
// (wire.OperateTypeF32 or F64) in dynamic mode (or checking typ against the
// schema's type in schema mode).
func (ob *OperateBuilder) SetFloatT(path Path, typ uint8, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpSET, typ: typ, path: path, a: floatBits(v)})
}

// AddFloatT adds v (IEEE) to the float scalar at path, creating it with
// type typ in dynamic mode (or checking typ against the schema's type in
// schema mode).
func (ob *OperateBuilder) AddFloatT(path Path, typ uint8, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpADD, typ: typ, path: path, a: floatBits(v)})
}

// MulFloatT multiplies (IEEE) the float scalar at path by v, creating it
// with type typ in dynamic mode (or checking typ against the schema's type
// in schema mode).
func (ob *OperateBuilder) MulFloatT(path Path, typ uint8, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMUL, typ: typ, path: path, a: floatBits(v)})
}

// MinFloatT sets the float scalar at path to min(current, v), creating it
// with type typ in dynamic mode (or checking typ against the schema's type
// in schema mode).
func (ob *OperateBuilder) MinFloatT(path Path, typ uint8, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMIN, typ: typ, path: path, a: floatBits(v)})
}

// MaxFloatT sets the float scalar at path to max(current, v), creating it
// with type typ in dynamic mode (or checking typ against the schema's type
// in schema mode).
func (ob *OperateBuilder) MaxFloatT(path Path, typ uint8, v float64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpMAX, typ: typ, path: path, a: floatBits(v)})
}

// Config sets the eviction triple (capacity, policy, and, for a *_COL
// policy, byCol) on the dynamic-mode table field at path, creating an empty
// table if absent (design doc §3.2: dynamic mode only — in schema mode
// eviction is part of the schema and changes via Migrate). byCol may be ""
// when policy does not need one.
//
// Because it is dynamic-mode only, Args returns an error rather than a call
// when a schema is attached (WithSchema): the server would answer
// wire.ErrOperateOpcode for it.
func (ob *OperateBuilder) Config(path Path, capacity uint32, policy uint8, byCol string) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpCONFIG, aux: policy, path: path, a: int64(capacity), byColName: byCol})
}

// Trim shrinks the table field at path to at most keep rows by policy (and,
// for a *_COL policy, byCol), without changing the table's stored eviction
// config (design doc §3.2). Unlike Config it is valid in BOTH modes: it is a
// one-off shrink, not a change to the table's configuration, so it does not
// collide with a schema-mode table's schema-declared eviction triple. byCol
// may be "" when policy does not need one.
func (ob *OperateBuilder) Trim(path Path, keep uint32, policy uint8, byCol string) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpTRIM, aux: policy, path: path, a: int64(keep), byColName: byCol})
}

// Migrate moves the record from its stored version (from, or
// wire.OperateMigrateFromDynamic when the record is dynamic-mode) to s's
// version in this atomic call, an append-only evolution or a dynamic-mode
// freeze (design doc §2.8/§2.9). dropExtra sets
// wire.OperateMigrateDropExtra, discarding dynamic fields s does not declare
// rather than rejecting the call. s is validated here, as WithSchema
// validates its own; a nil s or a validation failure is recorded and
// returned by Args rather than panicking.
//
// It must be the FIRST op: applyOps rejects a MIGRATE at any other index
// with wire.ErrOperateOpcode (design doc §2.8), so calling it after another
// op can only ever build a request the server refuses. That is recorded as a
// builder error here instead.
//
// It also moves the builder's resolution schema to s, without touching the
// call's `create` byte or the schema blob the call carries. Every op after
// the MIGRATE — which is all of them — and every return spec runs against
// the record as the migration leaves it, so their Field/Col names resolve
// against the TARGET schema. The call still declares the version the record
// is stored at, which is what the server checks when it opens it.
func (ob *OperateBuilder) Migrate(from int64, s *wire.Schema, dropExtra bool) *OperateBuilder {
	if s == nil {
		if ob.err == nil {
			ob.err = fmt.Errorf("client: operate: Migrate requires a non-nil schema")
		}
		return ob
	}
	if err := s.Validate(); err != nil {
		if ob.err == nil {
			ob.err = err
		}
		return ob
	}
	if len(ob.ops) > 0 {
		if ob.err == nil {
			ob.err = fmt.Errorf("client: operate: Migrate must be the first op, but %d op(s) were added before it", len(ob.ops))
		}
		return ob
	}
	var aux uint8
	if dropExtra {
		aux = wire.OperateMigrateDropExtra
	}
	ob.schema = s
	return ob.addOp(builderOp{opcode: wire.OperateOpMIGRATE, aux: aux, path: RecordPath(), a: from, bytes: s.Encode()})
}

// If skips the next n ops when node(path) cmp v is false (design doc §3.3);
// n is bounded by the remaining op list. Use IfBytes to compare against a
// bytes/fixed operand.
func (ob *OperateBuilder) If(path Path, cmp uint8, v int64, n int) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpIF, aux: cmp, path: path, a: v, b: int64(n)})
}

// IfBytes skips the next n ops when node(path) cmp v is false, comparing
// bytewise.
func (ob *OperateBuilder) IfBytes(path Path, cmp uint8, v []byte, n int) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpIF, aux: cmp, path: path, bytes: v, b: int64(n)})
}

// Check aborts the whole call (record unchanged, result status
// OperateStatusCheckFailed) when node(path) cmp v is false — the
// server-side compare-and-swap primitive on any field (design doc §3.3).
// Use CheckBytes to compare against a bytes/fixed operand.
func (ob *OperateBuilder) Check(path Path, cmp uint8, v int64) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpCHECK, aux: cmp, path: path, a: v})
}

// CheckBytes aborts the whole call when node(path) cmp v is false,
// comparing bytewise.
func (ob *OperateBuilder) CheckBytes(path Path, cmp uint8, v []byte) *OperateBuilder {
	return ob.addOp(builderOp{opcode: wire.OperateOpCHECK, aux: cmp, path: path, bytes: v})
}

// Return adds a return spec that reads back path's full, self-describing
// value after all ops apply (design doc §3.4).
func (ob *OperateBuilder) Return(path Path) *OperateBuilder {
	ob.rets = append(ob.rets, builderRet{mode: wire.OperateRetValue, path: path})
	return ob
}

// Count adds a return spec that reads back path's count after all ops
// apply: row count for a table, field count for the record, 1/0 for a
// present/absent scalar or row (design doc §3.4).
func (ob *OperateBuilder) Count(path Path) *OperateBuilder {
	ob.rets = append(ob.rets, builderRet{mode: wire.OperateRetCount, path: path})
	return ob
}

// Args resolves every op and return spec's Path against the builder's
// schema (if any) and returns the finished wire.OperateArgs. It never
// panics on a malformed builder: WithSchema's validation failure, or any
// resolution error hit along the way (an unknown field name, a wrong-width
// row key, a column path into a non-table field, a typed op whose type
// disagrees with the schema), is returned here instead.
func (ob *OperateBuilder) Args() (*wire.OperateArgs, error) {
	if ob.err != nil {
		return nil, ob.err
	}
	args := &wire.OperateArgs{
		Key:     ob.key,
		TTL:     ob.ttl,
		TTLMode: ob.ttlMode,
		Create:  ob.create,
	}
	if ob.createSchema != nil {
		args.Schema = ob.createSchema.Encode()
	}
	for _, op := range ob.ops {
		if op.opcode == wire.OperateOpCONFIG && ob.schema != nil {
			// The server answers wire.ErrOperateOpcode for a CONFIG against a
			// schema-mode record (design doc §3.2: a schema-mode table's
			// eviction triple comes from its schema and changes only via
			// MIGRATE), so this call can only ever fail — say so here rather
			// than after a round trip. A schema installed by a MIGRATE counts:
			// the record is in schema mode from the freeze onwards, and the
			// MIGRATE is the first op, so every CONFIG in the list is after it.
			return nil, fmt.Errorf("client: operate: Config is dynamic-mode only; a schema-mode table's eviction triple changes via Migrate")
		}
		wop, err := ob.resolveOp(op)
		if err != nil {
			return nil, err
		}
		args.Ops = append(args.Ops, wop)
	}
	for _, ret := range ob.rets {
		p, err := ob.resolvePath(ret.path)
		if err != nil {
			return nil, err
		}
		args.Rets = append(args.Rets, wire.OperateRet{Mode: ret.mode, Path: p})
	}
	return args, nil
}

// isScalarOpcode reports whether opcode is one of the scalar ops (design
// doc §3.1) whose Type names a field's domain and so must, in schema mode,
// agree with the schema's own type for a typed op. MIGRATE/CONFIG/TRIM
// don't carry a field type in this sense, and IF/CHECK never declare one
// (design doc §2.4).
func isScalarOpcode(opcode uint8) bool {
	switch opcode {
	case wire.OperateOpSET, wire.OperateOpADD, wire.OperateOpMUL, wire.OperateOpMIN, wire.OperateOpMAX,
		wire.OperateOpAND, wire.OperateOpOR, wire.OperateOpXOR, wire.OperateOpSHL, wire.OperateOpSHR, wire.OperateOpSTAMP:
		return true
	default:
		return false
	}
}

// resolveOp resolves one builderOp's Path (and, for CONFIG/TRIM, byColName)
// against ob's schema, and, for a typed scalar op in schema mode, checks
// typ against the schema's own type at that path.
func (ob *OperateBuilder) resolveOp(op builderOp) (wire.OperateOp, error) {
	path, err := ob.resolvePath(op.path)
	if err != nil {
		return wire.OperateOp{}, err
	}
	if ob.schema != nil && isScalarOpcode(op.opcode) && op.typ != wire.OperateTypeFromSchema {
		actual, aerr := ob.actualType(path)
		if aerr != nil {
			return wire.OperateOp{}, aerr
		}
		if actual != op.typ {
			return wire.OperateOp{}, fmt.Errorf("client: operate: op type %d does not match schema type %d for field %q", op.typ, actual, op.path.Field)
		}
	}
	wop := wire.OperateOp{Opcode: op.opcode, Type: op.typ, Aux: op.aux, Path: path, A: op.a, B: op.b, Bytes: op.bytes}
	if op.opcode == wire.OperateOpCONFIG || op.opcode == wire.OperateOpTRIM {
		if (op.aux == wire.OperatePolicyMinCol || op.aux == wire.OperatePolicyMaxCol) && op.byColName == "" {
			return wire.OperateOp{}, fmt.Errorf("client: operate: *_COL policy needs a column name")
		}
		wop.Bytes = []byte(op.byColName)
		if op.byColName != "" && ob.schema != nil {
			pos, cerr := ob.tableColPos(path, op.byColName)
			if cerr != nil {
				return wire.OperateOp{}, cerr
			}
			wop.B = int64(pos)
		}
	}
	return wop, nil
}

// resolvePath turns a client Path into a wire.OperatePath: an empty Path
// (no field, key, or column) is the record path; otherwise Field is
// resolved (by schema position, or as a name segment without a schema),
// and, if Key is set, checked against the field's table (schema mode: the
// field must be a table, and the key must be exactly the table's key
// width) before Col (if set) is resolved against that table's columns.
//
// A column belongs to a row, so a Path carrying Col without Key addresses
// nothing (design doc §2.4). It is rejected rather than narrowed: the wire
// form for such a path is OperatePathField, which silently DROPS the column
// and aims the op at the field itself — in dynamic mode that means a SET
// meant for one column of one row would overwrite a scalar field of the
// same name.
func (ob *OperateBuilder) resolvePath(p Path) (wire.OperatePath, error) {
	if p.HasCol && !p.HasKey {
		return wire.OperatePath{}, fmt.Errorf("client: operate: column %q of field %q needs a row key; use Col(field, key, col)", p.Col, p.Field)
	}
	if p.Field == "" && !p.HasKey && !p.HasCol {
		return wire.OperatePath{Kind: wire.OperatePathRecord}, nil
	}
	fseg, tdef, err := ob.resolveField(p.Field)
	if err != nil {
		return wire.OperatePath{}, err
	}
	if !p.HasKey {
		return wire.OperatePath{Kind: wire.OperatePathField, Field: fseg}, nil
	}
	if ob.schema != nil {
		if tdef == nil {
			return wire.OperatePath{}, fmt.Errorf("client: operate: field %q is not a table", p.Field)
		}
		width := wire.CellWidth(tdef.KeyType, tdef.KeyN)
		if len(p.Key) != width {
			return wire.OperatePath{}, fmt.Errorf("client: operate: key for field %q must be %d bytes, got %d", p.Field, width, len(p.Key))
		}
	}
	if !p.HasCol {
		return wire.OperatePath{Kind: wire.OperatePathRow, Field: fseg, Key: p.Key}, nil
	}
	cseg, err := ob.resolveCol(tdef, p.Col)
	if err != nil {
		return wire.OperatePath{}, err
	}
	return wire.OperatePath{Kind: wire.OperatePathCol, Field: fseg, Key: p.Key, Col: cseg}, nil
}

// resolveField resolves name to a wire.OperateSeg: a schema position (and
// that field's TableDef, nil unless it is a table field) when ob.schema is
// set, or a name segment otherwise.
func (ob *OperateBuilder) resolveField(name string) (wire.OperateSeg, *wire.TableDef, error) {
	if ob.schema == nil {
		return wire.OperateSeg{ByName: true, Name: name}, nil, nil
	}
	pos, ok := ob.schema.FieldPos(name)
	if !ok {
		return wire.OperateSeg{}, nil, fmt.Errorf("client: operate: unknown field %q", name)
	}
	return wire.OperateSeg{Pos: uint32(pos)}, ob.schema.Fields[pos].Table, nil //nolint:gosec // FieldPos returns a valid slice index
}

// resolveCol resolves name to a wire.OperateSeg within tdef's columns: a
// position when ob.schema is set (tdef is guaranteed non-nil by
// resolvePath's caller in that case), or a name segment otherwise.
func (ob *OperateBuilder) resolveCol(tdef *wire.TableDef, name string) (wire.OperateSeg, error) {
	if ob.schema == nil {
		return wire.OperateSeg{ByName: true, Name: name}, nil
	}
	for i, c := range tdef.Cols {
		if c.Name == name {
			return wire.OperateSeg{Pos: uint32(i)}, nil //nolint:gosec // i indexes tdef.Cols, bounded by OperateMaxCols
		}
	}
	return wire.OperateSeg{}, fmt.Errorf("client: operate: unknown column %q", name)
}

// tableColPos resolves colName to a column position within the table field
// path addresses (used by Config/Trim's byCol operand). ob.schema is
// non-nil whenever this is called.
func (ob *OperateBuilder) tableColPos(path wire.OperatePath, colName string) (int, error) {
	fi := int(path.Field.Pos)
	if fi < 0 || fi >= len(ob.schema.Fields) {
		return 0, fmt.Errorf("client: operate: field position %d out of range", fi)
	}
	tdef := ob.schema.Fields[fi].Table
	if tdef == nil {
		return 0, fmt.Errorf("client: operate: field at position %d is not a table", fi)
	}
	seg, err := ob.resolveCol(tdef, colName)
	if err != nil {
		return 0, err
	}
	return int(seg.Pos), nil
}

// actualType returns the schema's own type for the field or column p
// addresses. ob.schema is non-nil whenever this is called.
func (ob *OperateBuilder) actualType(p wire.OperatePath) (uint8, error) {
	fi := int(p.Field.Pos)
	if fi < 0 || fi >= len(ob.schema.Fields) {
		return 0, fmt.Errorf("client: operate: field position %d out of range", fi)
	}
	switch p.Kind {
	case wire.OperatePathField:
		return ob.schema.Fields[fi].Type, nil
	case wire.OperatePathCol:
		tdef := ob.schema.Fields[fi].Table
		if tdef == nil {
			return 0, fmt.Errorf("client: operate: field at position %d is not a table", fi)
		}
		ci := int(p.Col.Pos)
		if ci < 0 || ci >= len(tdef.Cols) {
			return 0, fmt.Errorf("client: operate: column position %d out of range", ci)
		}
		return tdef.Cols[ci].Type, nil
	default:
		return 0, fmt.Errorf("client: operate: path kind %d has no scalar type", p.Kind)
	}
}
