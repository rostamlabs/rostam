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

// Operate opcodes (design doc §3.1–§3.3): the byte carried in an OperateOp's
// Opcode field. Scalar ops (§3.1) apply to a record field or row column;
// table ops (§3.2) apply to the table field itself; control ops (§3.3) gate
// or abort a run of the other ops. Groups are numbered with gaps so a later
// revision can add an op to a group without renumbering the others.
const (
	OperateOpSET   uint8 = 1  // write a (int/float) or bytes (bytes/fixed)
	OperateOpDEL   uint8 = 2  // field -> UNSET; row -> removed; table -> emptied; record -> deleted
	OperateOpADD   uint8 = 3  // x += a, saturating (int) / IEEE (float)
	OperateOpMUL   uint8 = 4  // x *= a, saturating (int) / IEEE (float)
	OperateOpMIN   uint8 = 5  // x = min(x, v); int, float, bytes/fixed (bytewise)
	OperateOpMAX   uint8 = 6  // x = max(x, v); int, float, bytes/fixed (bytewise)
	OperateOpAND   uint8 = 7  // bitwise, masked to the field width; fixed-width int only
	OperateOpOR    uint8 = 8  // bitwise, masked to the field width; fixed-width int only
	OperateOpXOR   uint8 = 9  // bitwise, masked to the field width; fixed-width int only
	OperateOpSHL   uint8 = 10 // shift 0..64, masked to the field width; fixed-width int only
	OperateOpSHR   uint8 = 11 // shift 0..64; arithmetic for signed, logical for unsigned
	OperateOpSTAMP uint8 = 12 // x = tx.applyStamp() in aux's unit (OperateStamp*), saturating; int only

	OperateOpMIGRATE uint8 = 20 // must be the first op: append-only schema evolution or dynamic freeze
	OperateOpCONFIG  uint8 = 21 // dynamic mode only: set the table's eviction triple at path
	OperateOpTRIM    uint8 = 22 // one-off shrink of the table at path to a.Keep rows

	OperateOpIF    uint8 = 30 // if node(path) cmp v is false, skip the next b ops
	OperateOpCHECK uint8 = 31 // if node(path) cmp v is false, abort the whole list (CHECK_FAILED)
)

// OperateCmp* are the comparators usable by IF, CHECK, and CompareCell /
// CompareCount (design doc §3.3): a scalar compares in its domain (an absent
// numeric cell compares as zero; bytes compare bytewise); a table compares
// by its row count. EXISTS/ABSENT ignore the operand and test presence.
const (
	OperateCmpEQ uint8 = iota
	OperateCmpNE
	OperateCmpLT
	OperateCmpLE
	OperateCmpGT
	OperateCmpGE
	OperateCmpExists
	OperateCmpAbsent
)

// OperatePolicy* selects which row CONFIG/TRIM (or a full table) evicts:
// the row with the smallest/largest row key, or the smallest/largest value
// in the table's configured eviction column (design doc §3.2, "cap, policy,
// byCol").
const (
	OperatePolicyNone uint8 = iota
	OperatePolicyMinKey
	OperatePolicyMaxKey
	OperatePolicyMinCol
	OperatePolicyMaxCol
)

// OperateMode* is the record's stored mode byte (design doc §2.2/§2.9): the
// first byte of every stored operate record, distinguishing schema mode
// from self-describing dynamic mode.
const (
	OperateModeSchema  uint8 = 1
	OperateModeDynamic uint8 = 2
)

// OperateCreate* is a call's `create` parameter (design doc §3.5): whether
// (and how) an absent record may be created by this call.
const (
	OperateCreateNone uint8 = iota
	OperateCreateSchema
	OperateCreateDynamic
)

// OperateTTL* is a call's `ttlMode` parameter (design doc §3.5): how the
// call's ttlMs affects the record's expiry.
const (
	OperateTTLKeep uint8 = iota
	OperateTTLSet
	OperateTTLCreateOnly
)

// OperateStatus* is the result frame's status byte (design doc §3.4).
const (
	OperateStatusOK uint8 = iota
	OperateStatusCheckFailed
)

// OperateRet* selects a return spec's mode (design doc §3.4): the full,
// self-describing value, or just a count.
const (
	OperateRetValue uint8 = iota
	OperateRetCount
)

// OperateStamp* is STAMP's aux unit (design doc §3.1): the transaction
// timestamp is written in milliseconds or seconds, saturating to the
// field's stored type.
const (
	OperateStampMs uint8 = iota
	OperateStampS
)

// OperatePath* is a path's Kind byte (design doc §2.4): which of the
// record, a field, a table row, or a table column the path resolves to.
const (
	OperatePathRecord uint8 = iota
	OperatePathField
	OperatePathRow
	OperatePathCol
)

// OperateMigrateFromDynamic marks a MIGRATE op's `a` (from-version) operand
// as "the existing record is dynamic mode", distinguishing it from every
// non-negative schema version (design doc §2.8/§2.9).
const OperateMigrateFromDynamic int64 = -1

// OperateMigrateDropExtra is a bit in a MIGRATE op's aux byte: when set,
// fields/columns the new schema drops are discarded rather than rejected.
const OperateMigrateDropExtra uint8 = 1

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
