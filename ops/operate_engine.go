// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"errors"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// maxOperateRecordBytes is the design doc §2.7 backstop on a stored operate
// record: after a call's ops apply, the encoded record must still fit, or the
// call fails with wire.ErrOperateCap and the record is left unchanged (a cap
// hit is never an implicit eviction). It is a var rather than a const so a
// test can lower it and exercise the bound without building 16 MiB of record.
var maxOperateRecordBytes = 16 << 20

var (
	// errOperateAbsent is the "create = NONE and the key does not exist"
	// error of design doc §3.5. The handler maps it to the store's
	// not-found error.
	errOperateAbsent = errors.New("ops: operate record absent")
	// errOperateIfRange marks an IF whose skip count is negative or reaches
	// past the end of the op list (design doc §3.3, "n bounded by the
	// remaining list"). The bound is checked whenever the IF executes, not
	// only when the branch is taken, so a malformed op list is rejected the
	// same way regardless of the data it runs against.
	errOperateIfRange = errors.New("ops: operate IF skip count out of range")
)

// ref addresses one node inside a record for one op: the record itself, a
// scalar field or column, a table field, or one row of a table. It is the
// engine's own bookkeeping — field, row, col, off, typ, and n are filled in
// (and read back) however a given engine needs, and applyOps never looks at
// them. applyOps reads exactly two of ref's members:
//
//   - kind: one of the refKind* constants below, so evalCond (and nothing
//     else in the loop) can tell a count-shaped target (the record, a table
//     field, or a table row — design doc §3.3, "a table compares by its row
//     count") from a scalar one without otherwise interpreting the ref.
//   - present: whether the path resolved to something that already existed
//     before this call (design doc §2.4/§2.5). For OperatePathRecord this is
//     "did the record exist before the call", not "does it exist now" (the
//     oracle's ruling: CHECK((), ABSENT) means "this call created it").
//
// resolve is the only place a ref is produced; every other engine method
// takes one back exactly as resolve returned it.
type ref struct {
	kind    uint8
	field   int
	row     int
	col     int
	off     int
	typ, n  uint8
	present bool
}

// refKind* are the values an engine's resolve stores in ref.kind (see ref's
// doc comment). They are this file's own vocabulary — not part of the wire
// protocol — shared with every engine that implements engine below.
const (
	// refKindScalar is a record field or table column holding one Cell:
	// compared with wire.CompareCell, applied with wire.ApplyScalar.
	refKindScalar uint8 = iota
	// refKindRecord is the record itself (o.Path.Kind == wire.OperatePathRecord):
	// compared by field count via count().
	refKindRecord
	// refKindTable is a table field: compared by row count via count().
	refKindTable
	// refKindRow is one row of a table: compared by presence, which count()
	// reports as 1 (present) or 0 (absent) — the oracle's ruling that a row
	// has no value of its own, so EXISTS/ABSENT are the only comparators
	// with a useful meaning on it.
	refKindRow
)

// engine is what applyOps and evalRets need from a stored operate record to
// run one call's op and return specs (design doc §3). Two implementations
// exist: a schema-mode engine and a dynamic-mode engine, each patching its
// own stored byte layout in place; applyOps is the one place that knows op
// dispatch semantics, so neither engine needs to duplicate it.
//
// Every method's error is returned verbatim by applyOps/evalRets — engines
// return the wire.ErrOperate* sentinels, never a wrapped or engine-private
// error, so callers (and tests comparing against the oracle) can compare by
// errors.Is.
type engine interface {
	// resolve looks up path and returns a ref to it. create=false never
	// vivifies anything (used by DEL, TRIM, IF/CHECK, and evalRets); create
	// =true vivifies what a write needs — rows in both modes, fields/columns
	// in dynamic mode (schema-mode fields always exist, design doc §2.5).
	// typ/n are the type to create an absent DYNAMIC-mode scalar target
	// with (schema mode ignores them, since the schema is authoritative);
	// applyOps computes n as len(o.Bytes) for a FIXED target it is about to
	// create (see fixedN) and 0 otherwise, matching the wire encoding of a
	// freshly created FIXED cell.
	resolve(p wire.OperatePath, create bool, typ, n uint8) (ref, error)
	// get returns r's current scalar value. If !r.present this is the
	// type's zero cell (design doc §2.4, "absent reads as zero") — callers
	// pass r.present alongside it to whatever needs to distinguish the two.
	get(r ref) (wire.Cell, error)
	// set writes c to r's scalar target, vivifying it if resolve did not
	// already (create was true).
	set(r ref, c wire.Cell) error
	// del implements DEL for r's kind (design doc §3.1/§2.5): the record ->
	// deleted; a field -> UNSET (schema mode) or removed (dynamic mode); a
	// table -> emptied; a row/col -> removed. It never vivifies: deleting
	// something r.present == false reports is a no-op, not an error.
	del(r ref) error
	// count reports r's count for a compare (evalCond) or a RetCount
	// (evalRets; design doc §3.3/§3.4): field count for refKindRecord, row
	// count for refKindTable, and presence as 1 (present) or 0 (absent) for
	// refKindRow and refKindScalar alike (the oracle's ruling: a plain
	// scalar's "count" is only ever meaningful as exists/absent, the same
	// convention as a row). evalCond never calls it for refKindScalar — it
	// uses CompareCell there instead — but evalRets' RetCount does not
	// special-case any kind, so an engine must still answer for a scalar
	// ref.
	count(r ref) (uint64, error)
	// trim shrinks the table field at r to at most keep rows by policy (and,
	// for a *_COL policy, byCol in schema mode or byColName in dynamic
	// mode), leaving the table's stored eviction config untouched (design
	// doc §3.2). It is a shrink, not a write: an absent table (r.present ==
	// false) is a no-op.
	trim(r ref, keep uint32, policy uint8, byCol uint16, byColName string) error
	// config sets the table field at r's eviction triple (cap, policy,
	// byColName), creating an empty table if absent, and evicts down to the
	// new cap (design doc §3.2). Dynamic mode only.
	config(r ref, capN uint32, policy uint8, byColName string) error
	// migrate implements MIGRATE (design doc §2.8/§2.9): append-only schema
	// evolution (from >= 0) or a dynamic-mode freeze (from ==
	// wire.OperateMigrateFromDynamic). applyOps guarantees it is only ever
	// called for ops[0].
	migrate(from int64, flags uint8, schemaBlob []byte) error
	// value returns r's tagged encoding for a RetValue (design doc §3.4):
	// AppendTaggedCell's output for a scalar, the table's own encoding for
	// refKindTable, and (design doc §2.4) the row's encoding for refKindRow.
	// It is never called when r.present is false — evalRets substitutes
	// []byte{wire.OperateTypeUnset} itself in that case.
	value(r ref) ([]byte, error)
	// bytes returns the record's current encoded form.
	bytes() []byte
	// empty reports whether the record should be deleted as a side effect
	// of the ops that ran (design doc §2.5: a dynamic-mode record that lost
	// its last field). applyOps does not call this itself — it is read by
	// the caller once applyOps returns OK.
	empty() bool
}

// checkRowCap bounds a CONFIG/TRIM cap operand to [0, wire.OperateMaxRows]
// (design doc §2.7): negative or oversized is wire.ErrOperateCap, checked
// before resolve so a hostile cap never even vivifies the table field.
func checkRowCap(a int64) error {
	if a < 0 || a > int64(wire.OperateMaxRows) {
		return wire.ErrOperateCap
	}
	return nil
}

// trimByCol narrows a TRIM's eviction-column operand to the uint16 the engines
// address columns with, without letting a hostile value alias a real column: an
// int64 outside [0, OperateMaxCols) becomes OperateMaxCols, which is one past
// the highest addressable position (a schema's ByCol must be < len(Cols), and
// len(Cols) <= OperateMaxCols), so a schema-mode engine rejects it with
// wire.ErrOperatePath exactly as the oracle rejects the original out-of-range
// value. Dynamic mode addresses the eviction column by name and ignores this
// entirely. Without the clamp, o.B = 65536 (or -65536) would silently trim by
// column 0.
func trimByCol(b int64) uint16 {
	if b < 0 || b >= int64(wire.OperateMaxCols) {
		return wire.OperateMaxCols
	}
	return uint16(b) //nolint:gosec // bounded to [0, OperateMaxCols) above
}

// fixedN computes the n (width) applyOps passes to resolve when a scalar op
// might create a FIXED-typed dynamic-mode target: a FIXED cell's width is
// the length of its operand bytes (design doc §2.1, "FIXED(n): n raw bytes
// (1-255)"), never a declared width, since there is no schema to declare
// one. Every other type creates with n=0. Mirrors the oracle's
// treeCreateType FIXED branch; an out-of-[1,255] length is
// wire.ErrOperateType, the same error a real too-wide/empty FIXED value
// gets from wire.ApplyScalar/AppendTaggedCell elsewhere.
func fixedN(o *wire.OperateOp) (uint8, error) {
	if o.Type != wire.OperateTypeFixed {
		return 0, nil
	}
	if len(o.Bytes) < 1 || len(o.Bytes) > wire.OperateMaxKeyLen {
		return 0, wire.ErrOperateType
	}
	return uint8(len(o.Bytes)), nil //nolint:gosec // bounded to [1,255] above
}

// evalCond evaluates one IF/CHECK op's comparator against the node its path
// addresses (design doc §3.3). It never vivifies (resolve's create=false).
// The comparator travels in o.Aux (control ops otherwise ignore the type
// byte — o.Type is never consulted here, matching resolve's typ=0/n=0).
func evalCond(e engine, o *wire.OperateOp) (bool, error) {
	r, err := e.resolve(o.Path, false, 0, 0)
	if err != nil {
		return false, err
	}
	op := wire.Operand{A: o.A, Bytes: o.Bytes}
	if r.kind == refKindScalar {
		c, err := e.get(r)
		if err != nil {
			return false, err
		}
		return wire.CompareCell(o.Aux, c, r.present, op)
	}
	// refKindRecord, refKindTable, refKindRow: all three compare by count
	// (design doc §3.3) — a row's count() is defined as 1/0 by presence.
	n, err := e.count(r)
	if err != nil {
		return false, err
	}
	return wire.CompareCount(o.Aux, n, r.present, op)
}

// applyOps runs a call's op list against e in order (design doc §3.1-§3.3),
// returning the result status, the failing op's index on
// wire.OperateStatusCheckFailed, or a non-nil error. On any error the loop
// stops immediately — it never continues to a later op, so nothing beyond
// what e's own methods already touched is applied; the caller is expected to
// hold e's changes in a copy that is discarded whole on error (the engine
// owns atomicity of that buffer, applyOps only owns not making it worse by
// running further ops after a failure).
func applyOps(e engine, ops []wire.OperateOp, stampMs int64) (uint8, uint16, error) {
	for i := 0; i < len(ops); i++ {
		o := &ops[i]
		switch o.Opcode {
		case wire.OperateOpMIGRATE:
			// MIGRATE must be the first op (design doc §2.8); the caller
			// enforces it can only ever be Ops[0] by construction, but a
			// hostile/malformed op list might still carry one later.
			if i != 0 {
				return 0, 0, wire.ErrOperateOpcode
			}
			// migrate() takes no path (it always addresses the whole
			// record), so this is the only place that can reject a
			// MIGRATE aimed at a field/row/col (oracle ruling).
			if o.Path.Kind != wire.OperatePathRecord {
				return 0, 0, wire.ErrOperatePath
			}
			if err := e.migrate(o.A, o.Aux, o.Bytes); err != nil {
				return 0, 0, err
			}

		case wire.OperateOpIF, wire.OperateOpCHECK:
			// IF's skip count is validated whenever the op runs, not only
			// when the branch not-taken path would use it: a malformed op
			// list is malformed regardless of what the record holds (design
			// doc §3.3, oracle ruling).
			if o.Opcode == wire.OperateOpIF && (o.B < 0 || o.B > int64(len(ops)-1-i)) {
				return 0, 0, errOperateIfRange
			}
			ok, err := evalCond(e, o)
			if err != nil {
				return 0, 0, err
			}
			if !ok {
				if o.Opcode == wire.OperateOpCHECK {
					return wire.OperateStatusCheckFailed, uint16(i), nil //nolint:gosec // i bounded by wire.OperateMaxOps
				}
				i += int(o.B)
			}

		case wire.OperateOpDEL:
			r, err := e.resolve(o.Path, false, 0, 0)
			if err != nil {
				return 0, 0, err
			}
			if err := e.del(r); err != nil {
				return 0, 0, err
			}
			// DEL () is terminal (oracle ruling): the record is gone, so
			// the list ends here rather than running further ops against
			// nothing (design doc §3.1).
			if o.Path.Kind == wire.OperatePathRecord {
				return wire.OperateStatusOK, 0, nil
			}

		case wire.OperateOpCONFIG:
			// config() takes no path (only the ref it resolves to), so this
			// is the only place that can reject CONFIG at a row/col/record
			// path — a table's eviction config lives on the field itself
			// (oracle ruling).
			if o.Path.Kind != wire.OperatePathField {
				return 0, 0, wire.ErrOperatePath
			}
			if err := checkRowCap(o.A); err != nil {
				return 0, 0, err
			}
			r, err := e.resolve(o.Path, true, wire.OperateTypeTable, 0)
			if err != nil {
				return 0, 0, err
			}
			if err := e.config(r, uint32(o.A), o.Aux, string(o.Bytes)); err != nil { //nolint:gosec // o.A bounded above
				return 0, 0, err
			}

		case wire.OperateOpTRIM:
			// trim() takes no path either, and unlike CONFIG it does not
			// even create — a row/col/record path is rejected here before
			// resolve gets a chance to no-op it as merely absent.
			if o.Path.Kind != wire.OperatePathField {
				return 0, 0, wire.ErrOperatePath
			}
			if err := checkRowCap(o.A); err != nil {
				return 0, 0, err
			}
			// The policy byte is validated before resolve, not inside the
			// engine: an unknown policy is a malformed op whatever the
			// record holds (the oracle checks it in the same place).
			if o.Aux > wire.OperatePolicyMaxCol {
				return 0, 0, wire.ErrOperateSchema
			}
			r, err := e.resolve(o.Path, false, 0, 0)
			if err != nil {
				return 0, 0, err
			}
			if err := e.trim(r, uint32(o.A), o.Aux, trimByCol(o.B), string(o.Bytes)); err != nil { //nolint:gosec // o.A bounded by checkRowCap above
				return 0, 0, err
			}

		default: // the scalar ops (SET/ADD/MUL/MIN/MAX/AND/OR/XOR/SHL/SHR/STAMP);
			// wire.ApplyScalar itself rejects anything else with
			// wire.ErrOperateOpcode.
			n, err := fixedN(o)
			if err != nil {
				return 0, 0, err
			}
			r, err := e.resolve(o.Path, true, o.Type, n)
			if err != nil {
				return 0, 0, err
			}
			// resolve reports what it found, not whether it fits a scalar
			// op: a path can resolve straight to the record, a table, or a
			// row (e.g. a stray Col-shaped op against a plain field). Only
			// refKindScalar is a valid scalar-op target (oracle ruling).
			if r.kind != refKindScalar {
				return 0, 0, wire.ErrOperatePath
			}
			cur, err := e.get(r)
			if err != nil {
				return 0, 0, err
			}
			next, err := wire.ApplyScalar(o.Opcode, cur, r.present,
				wire.Operand{A: o.A, Aux: o.Aux, Bytes: o.Bytes, StampMs: stampMs})
			if err != nil {
				return 0, 0, err
			}
			if err := e.set(r, next); err != nil {
				return 0, 0, err
			}
		}
	}
	return wire.OperateStatusOK, 0, nil
}

// evalRets evaluates every return spec after applyOps returns (or, on
// wire.OperateStatusCheckFailed, against the pre-call record — the caller
// picks which engine instance to pass; evalRets itself is agnostic to that,
// design doc §3.4). It never vivifies (resolve's create=false).
//
// Absent-path mechanism: a ret whose path resolves with ref.present == false
// never calls value()/count() at all — evalRets substitutes
// []byte{wire.OperateTypeUnset} for RetValue and a tagged zero for RetCount
// itself. This is the "absent reads as zero/unset" rule (design doc §2.4)
// applied uniformly across scalar, row, table, and record paths, and it is
// the only absent-path mechanism an engine needs to support: value() and
// count() are never asked to report absence themselves.
func evalRets(e engine, rets []wire.OperateRet) ([][]byte, error) {
	if len(rets) == 0 {
		return nil, nil
	}
	out := make([][]byte, 0, len(rets))
	for i := range rets {
		r := &rets[i]
		if r.Mode != wire.OperateRetValue && r.Mode != wire.OperateRetCount {
			return nil, wire.ErrOperateArgs
		}
		nd, err := e.resolve(r.Path, false, 0, 0)
		if err != nil {
			return nil, err
		}
		if !nd.present {
			if r.Mode == wire.OperateRetCount {
				out = append(out, appendCountValue(0))
			} else {
				out = append(out, []byte{wire.OperateTypeUnset})
			}
			continue
		}
		if r.Mode == wire.OperateRetCount {
			n, err := e.count(nd)
			if err != nil {
				return nil, err
			}
			out = append(out, appendCountValue(n))
			continue
		}
		v, err := e.value(nd)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// appendCountValue encodes a RetCount result as a tagged U64 (design doc
// §3.4).
func appendCountValue(n uint64) []byte {
	return wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU64, U: n})
}

// --- shared byte helpers ---------------------------------------------------
//
// Both byte-level engines patch a record by replacing one span of it with
// another, so the two primitives live here rather than in either engine.

// splice replaces buf[off:off+oldLen] with repl and returns the resulting
// buffer, reusing buf's storage whenever its capacity can absorb the growth
// (the initial record copy reserves slack precisely so a row insert usually
// can). The caller is responsible for charging the growth against the
// record-size cap first: splice itself allocates whatever it is asked to.
func splice(buf []byte, off, oldLen int, repl []byte) []byte {
	delta := len(repl) - oldLen
	switch {
	case delta == 0:
		copy(buf[off:], repl)
		return buf
	case delta > 0 && cap(buf)-len(buf) < delta:
		out := make([]byte, len(buf)+delta, len(buf)+delta+64)
		copy(out, buf[:off])
		copy(out[off:], repl)
		copy(out[off+len(repl):], buf[off+oldLen:])
		return out
	case delta > 0:
		old := len(buf)
		buf = buf[:old+delta]
		// copy is a memmove: the tail may overlap its destination.
		copy(buf[off+len(repl):], buf[off+oldLen:old])
		copy(buf[off:], repl)
		return buf
	default:
		copy(buf[off:], repl)
		copy(buf[off+len(repl):], buf[off+oldLen:])
		return buf[:len(buf)+delta]
	}
}

// spliceUvarint replaces buf[off:off+oldLen] with the canonical uvarint
// encoding of v, returning the buffer and the new encoding's length. It is
// how a row count or a length prefix is rewritten in place when the value's
// encoded width may change.
func spliceUvarint(buf []byte, off, oldLen int, v uint64) ([]byte, int) {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(tmp[:], v)
	return splice(buf, off, oldLen, tmp[:n]), n
}
