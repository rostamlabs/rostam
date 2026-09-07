// SPDX-License-Identifier: Apache-2.0

package wire

import "errors"

// Operate scalar type tags (design doc §2.1). A stored cell carries no type of
// its own — the type lives once, in the schema (schema mode) or beside the
// name (dynamic mode) — these tags are the wire vocabulary shared by both.
//
//	0–7  U8 U16 U32 U64 I8 I16 I32 I64   fixed width 1/2/4/8, little-endian,
//	                                     arithmetic saturates
//	8–9  F32 F64                        IEEE, canonical NaN
//	10–11 UVARINT IVARINT               LEB128 (IVARINT zigzagged); record
//	                                     fields only, never a table column
//	12   BYTES                          [len uvarint][data], ≤ OperateMaxBytesLen
//	13   FIXED(n)                       n raw bytes (1–255); allowed as a column
//	                                     and as a row key
//	14   TABLE                          record fields only; not a cell itself
//	15   UNSET                          0 bytes; a declared-but-never-written slot
const (
	OperateTypeU8 uint8 = iota
	OperateTypeU16
	OperateTypeU32
	OperateTypeU64
	OperateTypeI8
	OperateTypeI16
	OperateTypeI32
	OperateTypeI64
	OperateTypeF32
	OperateTypeF64
	OperateTypeUVarint
	OperateTypeIVarint
	OperateTypeBytes
	OperateTypeFixed
	OperateTypeTable
	OperateTypeUnset

	// OperateTypeCount is one past the last valid type tag; any tag >= this is
	// unknown and rejected by the decoders.
	OperateTypeCount = 16

	// OperateTypeFromSchema marks an op's type byte as "whatever the schema (or,
	// in dynamic mode, the existing field) already says" rather than a type to
	// create with. It is never a value stored on the wire, only an op operand.
	OperateTypeFromSchema uint8 = 0xFF
)

// Caps (design doc §2.7): hard, DoS-defensive ceilings enforced before any
// allocation, all-or-nothing. Hitting one is an error with the record left
// unchanged — never an implicit eviction.
const (
	OperateMaxFields      = 65535   // a schema declaring more fields is rejected
	OperateMaxCols        = 65535   // a table declaring more columns is rejected
	OperateMaxSchemaBytes = 4096    // bounds schema parsing and the offset-cache entry
	OperateMaxRows        = 1 << 20 // per table; a hostile cap above it is clamped
	OperateMaxRowWidth    = 4096    // bounds FIXED(n) columns × count, in bytes
	OperateMaxKeyLen      = 255     // u8; FIXED(n) keys (schema mode) or any key (dynamic mode)
	OperateMaxNameLen     = 255     // u8; dynamic-mode field/column names
	OperateMaxBytesLen    = 65535   // u16 on the wire, per BYTES field
	OperateMaxNameBytes   = 4096    // per record; names are stored once
	OperateMaxOps         = 4096    // per call; bounds CPU per apply
	OperateMaxRet         = 4096    // per call; bounds CPU per apply
)

// Errors returned by the operate wire codec and, downstream, its apply engine.
var (
	// ErrOperateType marks an op whose type byte does not match the domain
	// (int/float/bytes) of the field or column it targets, or an op that
	// cannot apply to that domain at all (e.g. a shift on a float).
	ErrOperateType = errors.New("wire: operate type not applicable")
	// ErrOperatePath marks a path that does not resolve against the record: an
	// out-of-range position, an unknown name, a row path into a non-table
	// field, or a scalar op on a table field.
	ErrOperatePath = errors.New("wire: operate path does not fit the record")
	// ErrOperateOpcode marks an unknown opcode, or a table opcode used where a
	// scalar opcode is expected (and vice versa).
	ErrOperateOpcode = errors.New("wire: operate unknown opcode")
	// ErrOperateCmp marks an unknown comparator in a CHECK/IF op.
	ErrOperateCmp = errors.New("wire: operate unknown comparator")
	// ErrOperateSchema marks a schema blob that fails to decode or validate.
	ErrOperateSchema = errors.New("wire: operate schema invalid")
	// ErrOperateSchemaVersion marks a call whose schema version does not match
	// the version already stored on the record.
	ErrOperateSchemaVersion = errors.New("wire: operate schema version mismatch")
	// ErrOperateMode marks a call whose Create mode does not match the mode
	// (schema vs. dynamic) byte already stored on the record.
	ErrOperateMode = errors.New("wire: operate record mode mismatch")
	// ErrOperateCap marks a call that would exceed one of the caps above.
	ErrOperateCap = errors.New("wire: operate cap exceeded")
	// ErrOperateRecord marks a stored record whose bytes are malformed
	// (hostile or corrupt) rather than a call whose input is invalid.
	ErrOperateRecord = errors.New("wire: operate stored record malformed")
)
