// SPDX-License-Identifier: Apache-2.0

package ops

// The operate v2 tree oracle.
//
// applyTree is the deliberately simple decode → apply → encode implementation
// the design doc keeps as the reference semantics (§2.6, decision 10): it
// copies the record tree, mutates it, re-sorts, and re-encodes. It is never
// used by the handler — the shipped engines patch stored bytes in place — but
// it is the executable definition of what those engines must do, and
// operate_semantics_test.go runs the same suite against both.
//
// Everything here is test-only. errOperateAbsent moves to the apply engine
// when that lands; the other helpers stay in the test files.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- shared test helpers ---------------------------------------------------

// sessionSchema is the design doc's §4 worked example, copied from
// sdk/wire/operate_schema_test.go (package ops cannot see that helper):
// three scalar counters and a bidder table with an LRU on its stamp column.
func sessionSchema() *wire.Schema {
	return &wire.Schema{Version: 1, Fields: []wire.FieldDef{
		{Name: "rc", Type: wire.OperateTypeU8}, {Name: "bc", Type: wire.OperateTypeU8}, {Name: "hist", Type: wire.OperateTypeU32},
		{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{KeyType: wire.OperateTypeU64, Cols: []wire.ColumnDef{
			{Name: "c", Type: wire.OperateTypeU16}, {Name: "hi", Type: wire.OperateTypeI64}, {Name: "t", Type: wire.OperateTypeU32}},
			Cap: 1024, Policy: wire.OperatePolicyMinCol, ByCol: 2}},
	}}
}

// frozenSchema is the target of a dynamic-mode freeze (design doc §2.9): it
// stores names, because a freeze matches dynamic fields and columns by name.
func frozenSchema() *wire.Schema {
	return &wire.Schema{Version: 3, StoreNames: true, Fields: []wire.FieldDef{
		{Name: "hits", Type: wire.OperateTypeU32},
		{Name: "name", Type: wire.OperateTypeBytes},
		{Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{KeyType: wire.OperateTypeU64,
			Cols: []wire.ColumnDef{{Name: "c", Type: wire.OperateTypeU16}}}},
	}}
}

// floatSchema exercises the parts of §2.2 sessionSchema does not: the float
// types and the whole variable-length tail (BYTES, FIXED, UVARINT), which sit
// after the fixed-width fields and are the ones a reader has to walk.
func floatSchema() *wire.Schema {
	return &wire.Schema{Version: 7, Fields: []wire.FieldDef{
		{Name: "f32", Type: wire.OperateTypeF32},
		{Name: "f64", Type: wire.OperateTypeF64},
		{Name: "blob", Type: wire.OperateTypeBytes},
		{Name: "cc", Type: wire.OperateTypeFixed, N: 2},
		{Name: "v", Type: wire.OperateTypeUVarint},
	}}
}

// widenedSchema is a version bump that is not an append-only extension: it
// widens an existing field, which Schema.Extends rejects (design doc §2.8).
func widenedSchema() *wire.Schema {
	s := sessionSchema()
	s.Version = 2
	s.Fields[0].Type = wire.OperateTypeU16
	return s
}

// su64 reinterprets a signed value as the bits Cell.U stores for it. Go
// rejects uint64(int64(-5)) as a constant expression, so tests need the
// conversion to happen in a function.
func su64(i int64) uint64 { return uint64(i) }

// f64 is a float operand as it travels on the wire: the IEEE bit pattern of a
// float64, for F32 fields too (design doc §3).
func f64(x float64) int64 { return int64(math.Float64bits(x)) }

// keyU64 encodes a U64 row key the way a row stores it: little-endian, 8
// bytes (design doc §2.3).
func keyU64(k uint64) []byte {
	return binary.LittleEndian.AppendUint64(nil, k)
}

func recPath() wire.OperatePath { return wire.OperatePath{Kind: wire.OperatePathRecord} }

func fieldPath(pos uint32) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{Pos: pos}}
}

func rowPath(pos uint32, key []byte) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathRow, Field: wire.OperateSeg{Pos: pos}, Key: key}
}

func colPath(pos uint32, key []byte, col uint32) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathCol, Field: wire.OperateSeg{Pos: pos}, Key: key,
		Col: wire.OperateSeg{Pos: col}}
}

func namePath(name string) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: name}}
}

func nameRowPath(field string, key []byte) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathRow, Field: wire.OperateSeg{ByName: true, Name: field}, Key: key}
}

func nameColPath(field string, key []byte, col string) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathCol, Field: wire.OperateSeg{ByName: true, Name: field}, Key: key,
		Col: wire.OperateSeg{ByName: true, Name: col}}
}

// rowKeys returns a table's row keys, in stored order, decoded as
// little-endian u64s — the shape every test in the suite uses for keys.
func rowKeys(t *wire.Table) []uint64 {
	out := make([]uint64, 0, len(t.Rows))
	for _, r := range t.Rows {
		var v uint64
		for i := len(r.Key) - 1; i >= 0; i-- {
			v = v<<8 | uint64(r.Key[i])
		}
		out = append(out, v)
	}
	return out
}

// oper is a wire.OperateOp under construction: op() fills the fields every op
// carries and the with* methods add the optional operands, so an op list in a
// test reads like the design doc's worked examples (§4).
type oper wire.OperateOp

func op(opcode, typ uint8, p wire.OperatePath, a int64) oper {
	return oper{Opcode: opcode, Type: typ, Path: p, A: a}
}

func (o oper) withAux(aux uint8) oper  { o.Aux = aux; return o }
func (o oper) withB(b int64) oper      { o.B = b; return o }
func (o oper) withBytes(b []byte) oper { o.Bytes = b; return o }
func toOps(ops []oper) []wire.OperateOp {
	out := make([]wire.OperateOp, len(ops))
	for i := range ops {
		out[i] = wire.OperateOp(ops[i])
	}
	return out
}

// schemaArgs, dynArgs and noneArgs build a call with each of the three
// `create` modes (design doc §3.5).
func schemaArgs(s *wire.Schema, ops ...oper) *wire.OperateArgs {
	return &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode(), Ops: toOps(ops)}
}

func dynArgs(ops ...oper) *wire.OperateArgs {
	return &wire.OperateArgs{Create: wire.OperateCreateDynamic, Ops: toOps(ops)}
}

func noneArgs(ops ...oper) *wire.OperateArgs {
	return &wire.OperateArgs{Create: wire.OperateCreateNone, Ops: toOps(ops)}
}

// withRet and withCount append VALUE / COUNT return specs (design doc §3.4).
func withRet(a *wire.OperateArgs, paths ...wire.OperatePath) *wire.OperateArgs {
	for _, p := range paths {
		a.Rets = append(a.Rets, wire.OperateRet{Mode: wire.OperateRetValue, Path: p})
	}
	return a
}

func withCount(a *wire.OperateArgs, paths ...wire.OperatePath) *wire.OperateArgs {
	for _, p := range paths {
		a.Rets = append(a.Rets, wire.OperateRet{Mode: wire.OperateRetCount, Path: p})
	}
	return a
}

// --- the oracle ------------------------------------------------------------

// applyTree applies a's op list to rec and returns the new record, the result
// frame, and an error. rec == nil means the key is absent; a nil returned
// record means the key is deleted (design doc §2.5). rec is never mutated: the
// oracle works on a deep copy, so a CHECK failure — and any error — returns
// the caller's original tree untouched (design doc §2.7: a cap hit is an
// error with the record unchanged, never a partial write).
//
// Rulings this implementation makes where the design doc is silent are marked
// "Ruling:" in the comments below.
func applyTree(rec *wire.Record, a *wire.OperateArgs, stampMs int64) (*wire.Record, *wire.OperateResult, error) {
	if len(a.Ops) > wire.OperateMaxOps || len(a.Rets) > wire.OperateMaxRet {
		return rec, nil, wire.ErrOperateCap
	}
	cur, err := treeOpen(rec, a)
	if err != nil {
		return rec, nil, err
	}
	st := &treeState{rec: cur, existed: rec != nil, stampMs: stampMs}

	status, failedOp, err := st.run(a.Ops)
	if err != nil {
		return rec, nil, err
	}
	if status == wire.OperateStatusCheckFailed {
		// The whole list aborts with the record unchanged, and the return
		// specs are evaluated against that unchanged record (design doc §3.3).
		vals, verr := treeRets(rec, rec != nil, a.Rets)
		if verr != nil {
			return rec, nil, verr
		}
		return rec, &wire.OperateResult{Status: status, FailedOp: failedOp, Values: vals}, nil
	}

	out := st.rec
	if st.deleted || (out.Mode == wire.OperateModeDynamic && len(out.Fields) == 0) {
		out = nil
	}
	if out != nil {
		treeNormalize(out)
		// §2.7's backstop, checked on the result rather than op by op: a call
		// that would store an over-large record fails with the record
		// unchanged.
		if len(out.Encode()) > maxOperateRecordBytes {
			return rec, nil, wire.ErrOperateCap
		}
	}
	vals, verr := treeRets(out, true, a.Rets)
	if verr != nil {
		return rec, nil, verr
	}
	return out, &wire.OperateResult{Status: wire.OperateStatusOK, Values: vals}, nil
}

// treeState is one in-flight apply: the working copy of the record, whether
// that record existed before the call (record-path EXISTS/ABSENT), the
// transaction stamp, and whether DEL () deleted it.
type treeState struct {
	rec     *wire.Record
	existed bool
	stampMs int64
	deleted bool
}

// treeOpen resolves the call's `create` parameter against the stored record
// (design doc §3.5) and returns the working copy to apply ops to.
func treeOpen(rec *wire.Record, a *wire.OperateArgs) (*wire.Record, error) {
	cur := treeClone(rec)
	switch a.Create {
	case wire.OperateCreateNone:
		// Ruling: with create = NONE the call carries no schema blob, so a
		// blob that arrives anyway is ignored rather than checked — there is
		// nothing in §3.5 for it to be checked against.
		if cur == nil {
			return nil, errOperateAbsent
		}
	case wire.OperateCreateSchema:
		if len(a.Schema) == 0 {
			return nil, wire.ErrOperateSchema
		}
		s, n, err := wire.DecodeSchema(a.Schema)
		if err != nil {
			return nil, err
		}
		if n != len(a.Schema) {
			return nil, wire.ErrOperateSchema
		}
		if cur == nil {
			cur = treeNewSchemaRecord(s)
			break
		}
		if cur.Mode != wire.OperateModeSchema {
			return nil, wire.ErrOperateMode
		}
		// Ruling: §2.8 compares versions, not blobs. The stored schema stays
		// authoritative for the record; the call's blob only has to agree on
		// the version.
		if cur.Schema.Version != s.Version {
			return nil, wire.ErrOperateSchemaVersion
		}
	case wire.OperateCreateDynamic:
		if cur == nil {
			cur = &wire.Record{Mode: wire.OperateModeDynamic}
			break
		}
		if cur.Mode != wire.OperateModeDynamic {
			return nil, wire.ErrOperateMode
		}
	default:
		return nil, wire.ErrOperateArgs
	}
	treeNormalize(cur)
	return cur, nil
}

// run walks the op list (design doc §3.1-§3.3), returning the result status
// and, on CHECK_FAILED, the failing op's index.
func (st *treeState) run(ops []wire.OperateOp) (uint8, uint16, error) {
	for i := 0; i < len(ops); i++ {
		o := ops[i]
		if o.Opcode == wire.OperateOpMIGRATE {
			if i != 0 {
				return 0, 0, wire.ErrOperateOpcode
			}
			if err := st.migrate(o); err != nil {
				return 0, 0, err
			}
			continue
		}
		switch o.Opcode {
		case wire.OperateOpIF:
			// The skip count is validated whenever the IF runs, not only when
			// the branch is taken: a malformed op list is malformed whatever
			// the record happens to hold (design doc §3.3, "n bounded by the
			// remaining list").
			if o.B < 0 || o.B > int64(len(ops)-i-1) {
				return 0, 0, errOperateIfRange
			}
			ok, err := st.compare(o)
			if err != nil {
				return 0, 0, err
			}
			if !ok {
				i += int(o.B)
			}
		case wire.OperateOpCHECK:
			ok, err := st.compare(o)
			if err != nil {
				return 0, 0, err
			}
			if !ok {
				return wire.OperateStatusCheckFailed, uint16(i), nil
			}
		case wire.OperateOpDEL:
			if err := st.del(o); err != nil {
				return 0, 0, err
			}
			// Ruling: DEL () is terminal. §3.1 says the record is deleted;
			// rather than invent semantics for ops that would run against a
			// deleted record, the list ends here and the return specs see an
			// absent record.
			if st.deleted {
				return wire.OperateStatusOK, 0, nil
			}
		case wire.OperateOpCONFIG:
			if err := st.config(o); err != nil {
				return 0, 0, err
			}
		case wire.OperateOpTRIM:
			if err := st.trim(o); err != nil {
				return 0, 0, err
			}
		default:
			if err := st.scalar(o); err != nil {
				return 0, 0, err
			}
		}
	}
	return wire.OperateStatusOK, 0, nil
}

// scalar applies one §3.1 op to the scalar the path addresses, vivifying
// what the write needs on the way (design doc §2.4, "writes vivify").
func (st *treeState) scalar(o wire.OperateOp) error {
	nd, err := st.resolve(o.Path, o, true)
	if err != nil {
		return err
	}
	if nd.kind == wire.OperatePathRecord || nd.kind == wire.OperatePathRow || nd.isTable {
		return wire.ErrOperatePath
	}
	cell := treeCellAt(st.rec, nd)
	out, err := wire.ApplyScalar(o.Opcode, cell, nd.present,
		wire.Operand{A: o.A, Aux: o.Aux, Bytes: o.Bytes, StampMs: st.stampMs})
	if err != nil {
		return err
	}
	treeSetCell(st.rec, nd, out)
	return nil
}

// compare evaluates an IF/CHECK comparator against the node the path
// addresses (design doc §3.3). It never vivifies.
func (st *treeState) compare(o wire.OperateOp) (bool, error) {
	nd, err := st.resolve(o.Path, o, false)
	if err != nil {
		return false, err
	}
	opnd := wire.Operand{A: o.A, Aux: o.Aux, Bytes: o.Bytes, StampMs: st.stampMs}
	switch {
	case nd.kind == wire.OperatePathRecord:
		return wire.CompareCount(o.Aux, uint64(st.fieldCount()), nd.present, opnd)
	case nd.isTable:
		var n uint64
		if nd.tbl != nil {
			n = uint64(len(nd.tbl.Rows))
		}
		return wire.CompareCount(o.Aux, n, nd.present, opnd)
	case nd.kind == wire.OperatePathRow:
		// Ruling: a row has no value of its own, so it compares as a count of
		// 1 (present) or 0 (absent) — which makes EXISTS/ABSENT the only
		// comparators with a useful meaning on it.
		var n uint64
		if nd.present {
			n = 1
		}
		return wire.CompareCount(o.Aux, n, nd.present, opnd)
	default:
		return wire.CompareCell(o.Aux, treeCellAt(st.rec, nd), nd.present, opnd)
	}
}

// del implements DEL for every path kind (design doc §3.1/§2.5). It never
// vivifies: deleting something absent is a no-op, not an error.
func (st *treeState) del(o wire.OperateOp) error {
	nd, err := st.resolve(o.Path, o, false)
	if err != nil {
		return err
	}
	switch nd.kind {
	case wire.OperatePathRecord:
		st.deleted = true
	case wire.OperatePathField:
		if nd.fi < 0 {
			return nil
		}
		if st.rec.Mode == wire.OperateModeDynamic {
			st.rec.Fields = append(st.rec.Fields[:nd.fi], st.rec.Fields[nd.fi+1:]...)
			return nil
		}
		if nd.isTable {
			nd.tbl.Rows = nil
			return nil
		}
		st.rec.Fields[nd.fi].Cell = wire.ZeroCell(nd.typ, nd.n)
	case wire.OperatePathRow:
		if !nd.present || nd.ri < 0 {
			return nil
		}
		nd.tbl.Rows = append(nd.tbl.Rows[:nd.ri], nd.tbl.Rows[nd.ri+1:]...)
	case wire.OperatePathCol:
		if nd.ri < 0 || nd.ci < 0 || !nd.present {
			return nil
		}
		row := &nd.tbl.Rows[nd.ri]
		if st.rec.Mode == wire.OperateModeDynamic {
			// Ruling: a dynamic row survives losing its last column; a row is
			// a keyed entity, and an empty column list encodes fine.
			row.Cols = append(row.Cols[:nd.ci], row.Cols[nd.ci+1:]...)
			return nil
		}
		row.Cols[nd.ci].Cell = wire.ZeroCell(nd.typ, nd.n)
	}
	return nil
}

// config implements CONFIG (design doc §3.2): dynamic mode only, it sets the
// eviction triple on the table at path, creating an empty table if absent,
// and evicts down to the new cap.
func (st *treeState) config(o wire.OperateOp) error {
	if st.rec.Mode != wire.OperateModeDynamic {
		return wire.ErrOperateOpcode
	}
	if o.Path.Kind != wire.OperatePathField {
		return wire.ErrOperatePath
	}
	if !o.Path.Field.ByName {
		return wire.ErrOperatePath
	}
	if len(o.Path.Field.Name) > wire.OperateMaxNameLen || len(o.Bytes) > wire.OperateMaxNameLen {
		return wire.ErrOperateCap
	}
	if o.A < 0 || o.A > int64(wire.OperateMaxRows) {
		return wire.ErrOperateCap
	}
	// Ruling: an unknown policy byte is a schema-shaped error, the same one
	// Schema.Validate returns for a table declaring it.
	if o.Aux > wire.OperatePolicyMaxCol {
		return wire.ErrOperateSchema
	}
	byColName := string(o.Bytes)
	if (o.Aux == wire.OperatePolicyMinCol || o.Aux == wire.OperatePolicyMaxCol) && byColName == "" {
		// Ruling: mirrors Schema.Validate's "ByCol must name a real column"
		// rule for a *_COL policy.
		return wire.ErrOperatePath
	}

	fi := treeFindField(st.rec.Fields, o.Path.Field.Name)
	if fi < 0 {
		if len(st.rec.Fields)+1 > wire.OperateMaxFields {
			return wire.ErrOperateCap
		}
		st.rec.Fields = append(st.rec.Fields, wire.Field{Name: o.Path.Field.Name,
			Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{}})
		treeSortFields(st.rec)
		fi = treeFindField(st.rec.Fields, o.Path.Field.Name)
	} else if st.rec.Fields[fi].Cell.Type != wire.OperateTypeTable {
		return wire.ErrOperatePath
	}
	tbl := st.rec.Fields[fi].Table
	tbl.Cap = uint32(o.A)
	tbl.Policy = o.Aux
	tbl.ByColName = byColName
	for tbl.Cap > 0 && uint32(len(tbl.Rows)) > tbl.Cap {
		treeEvictOne(tbl, tbl.Policy, int(tbl.ByCol), tbl.ByColName, true)
	}
	return nil
}

// trim implements TRIM (design doc §3.2): a one-off shrink of the table at
// path to `keep` rows by the op's policy, leaving the stored policy alone.
func (st *treeState) trim(o wire.OperateOp) error {
	if o.Path.Kind != wire.OperatePathField {
		return wire.ErrOperatePath
	}
	if o.A < 0 || o.A > int64(wire.OperateMaxRows) {
		return wire.ErrOperateCap
	}
	if o.Aux > wire.OperatePolicyMaxCol {
		return wire.ErrOperateSchema
	}
	nd, err := st.resolve(o.Path, o, false)
	if err != nil {
		return err
	}
	if nd.fi < 0 {
		// Ruling: TRIM is a shrink, not a write, so an absent dynamic table
		// field is a no-op rather than a vivification.
		return nil
	}
	if !nd.isTable || nd.tbl == nil {
		return wire.ErrOperatePath
	}

	dynamic := st.rec.Mode == wire.OperateModeDynamic
	byCol := int(o.B)
	byColName := string(o.Bytes)
	if o.Aux == wire.OperatePolicyMinCol || o.Aux == wire.OperatePolicyMaxCol {
		if dynamic {
			if byColName == "" {
				return wire.ErrOperatePath
			}
		} else if byCol < 0 || byCol >= len(st.rec.Schema.Fields[nd.fi].Table.Cols) {
			return wire.ErrOperatePath
		}
	}
	for len(nd.tbl.Rows) > int(o.A) {
		treeEvictOne(nd.tbl, o.Aux, byCol, byColName, dynamic)
	}
	return nil
}

// migrate implements MIGRATE (design doc §2.8/§2.9). It must be Ops[0]; the
// caller enforces that.
func (st *treeState) migrate(o wire.OperateOp) error {
	if o.Path.Kind != wire.OperatePathRecord {
		return wire.ErrOperatePath
	}
	ns, n, err := wire.DecodeSchema(o.Bytes)
	if err != nil {
		return err
	}
	if n != len(o.Bytes) {
		return wire.ErrOperateSchema
	}
	if o.A == wire.OperateMigrateFromDynamic {
		return st.freeze(ns, o.Aux)
	}
	if st.rec.Mode != wire.OperateModeSchema {
		return wire.ErrOperateMode
	}
	if o.A < 0 || o.A > math.MaxUint16 || uint16(o.A) != st.rec.Schema.Version {
		return wire.ErrOperateSchemaVersion
	}
	if err := st.rec.Schema.Extends(ns); err != nil {
		return err
	}

	old := st.rec
	fields := make([]wire.Field, len(ns.Fields))
	for i := range ns.Fields {
		fd := &ns.Fields[i]
		if fd.Type != wire.OperateTypeTable {
			c := wire.ZeroCell(fd.Type, fd.N)
			if i < len(old.Fields) {
				c = treeSchemaCell(old.Fields[i].Cell, fd.Type, fd.N)
			}
			fields[i] = wire.Field{Name: fd.Name, Cell: c}
			continue
		}
		tbl := &wire.Table{Cap: fd.Table.Cap, Policy: fd.Table.Policy, ByCol: fd.Table.ByCol}
		if i < len(old.Fields) && old.Fields[i].Table != nil {
			for _, r := range old.Fields[i].Table.Rows {
				row := wire.Row{Key: append([]byte(nil), r.Key...), Cols: make([]wire.Col, len(fd.Table.Cols))}
				for c := range fd.Table.Cols {
					cd := &fd.Table.Cols[c]
					if c < len(r.Cols) {
						row.Cols[c] = wire.Col{Cell: treeSchemaCell(r.Cols[c].Cell, cd.Type, cd.N)}
						continue
					}
					row.Cols[c] = wire.Col{Cell: wire.ZeroCell(cd.Type, cd.N)}
				}
				tbl.Rows = append(tbl.Rows, row)
			}
		}
		treeSortRows(tbl, fd.Table.KeyType, false)
		for tbl.Cap > 0 && uint32(len(tbl.Rows)) > tbl.Cap {
			treeEvictOne(tbl, tbl.Policy, int(tbl.ByCol), "", false)
		}
		fields[i] = wire.Field{Name: fd.Name, Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: tbl}
	}
	st.rec = &wire.Record{Mode: wire.OperateModeSchema, Schema: ns, Fields: fields}
	treeNormalize(st.rec)
	return nil
}

// freeze converts a dynamic record into schema mode (design doc §2.9): each
// schema field and column is filled from the dynamic value of the same name,
// widths saturate, rows are packed, and a dynamic field or column the schema
// does not have is an error unless DROP_EXTRA is set.
func (st *treeState) freeze(ns *wire.Schema, flags uint8) error {
	if st.rec.Mode != wire.OperateModeDynamic {
		return wire.ErrOperateMode
	}
	if !ns.StoreNames {
		return wire.ErrOperateSchema
	}
	dropExtra := flags&wire.OperateMigrateDropExtra != 0

	used := make(map[string]bool, len(ns.Fields))
	fields := make([]wire.Field, len(ns.Fields))
	for i := range ns.Fields {
		fd := &ns.Fields[i]
		if fd.Name == "" {
			return wire.ErrOperateSchema
		}
		used[fd.Name] = true
		si := treeFindField(st.rec.Fields, fd.Name)

		if fd.Type != wire.OperateTypeTable {
			var src wire.Cell
			if si >= 0 {
				if st.rec.Fields[si].Cell.Type == wire.OperateTypeTable {
					return wire.ErrOperateType
				}
				src = st.rec.Fields[si].Cell
			}
			cell, err := treeConvertCell(src, si >= 0, fd.Type, fd.N)
			if err != nil {
				return err
			}
			fields[i] = wire.Field{Name: fd.Name, Cell: cell}
			continue
		}

		tbl := &wire.Table{Cap: fd.Table.Cap, Policy: fd.Table.Policy, ByCol: fd.Table.ByCol}
		if si >= 0 {
			src := &st.rec.Fields[si]
			if src.Cell.Type != wire.OperateTypeTable || src.Table == nil {
				return wire.ErrOperateType
			}
			keyWidth := wire.CellWidth(fd.Table.KeyType, fd.Table.KeyN)
			for _, r := range src.Table.Rows {
				// Ruling: a schema-mode row key is exactly the table's key
				// width, so a dynamic key of any other length cannot be
				// packed and the freeze is rejected.
				if len(r.Key) != keyWidth {
					return wire.ErrOperateSchema
				}
				row := wire.Row{Key: append([]byte(nil), r.Key...), Cols: make([]wire.Col, len(fd.Table.Cols))}
				matched := 0
				for c := range fd.Table.Cols {
					cd := &fd.Table.Cols[c]
					if cd.Name == "" {
						return wire.ErrOperateSchema
					}
					ci := treeFindCol(r.Cols, cd.Name)
					var src wire.Cell
					if ci >= 0 {
						src = r.Cols[ci].Cell
						matched++
					}
					cell, err := treeConvertCell(src, ci >= 0, cd.Type, cd.N)
					if err != nil {
						return err
					}
					row.Cols[c] = wire.Col{Cell: cell}
				}
				if matched != len(r.Cols) && !dropExtra {
					return wire.ErrOperateSchema
				}
				tbl.Rows = append(tbl.Rows, row)
			}
		}
		treeSortRows(tbl, fd.Table.KeyType, false)
		for tbl.Cap > 0 && uint32(len(tbl.Rows)) > tbl.Cap {
			treeEvictOne(tbl, tbl.Policy, int(tbl.ByCol), "", false)
		}
		fields[i] = wire.Field{Name: fd.Name, Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: tbl}
	}

	if !dropExtra {
		for i := range st.rec.Fields {
			if !used[st.rec.Fields[i].Name] {
				return wire.ErrOperateSchema
			}
		}
	}
	st.rec = &wire.Record{Mode: wire.OperateModeSchema, Schema: ns, Fields: fields}
	treeNormalize(st.rec)
	return nil
}

func (st *treeState) fieldCount() int {
	if st.rec.Mode == wire.OperateModeSchema {
		return len(st.rec.Schema.Fields)
	}
	return len(st.rec.Fields)
}

// --- path resolution -------------------------------------------------------

// treeNode is a resolved path: which field, row and column it lands on, the
// declared type of the scalar there, and whether that scalar existed before
// this op (the "absent is zero" flag, design doc §2.4).
type treeNode struct {
	kind    uint8
	fi      int
	tbl     *wire.Table
	ri      int
	ci      int
	typ     uint8
	n       uint8
	present bool
	isTable bool
	retype  bool // dynamic SET replacing the stored type
}

func (st *treeState) resolve(p wire.OperatePath, o wire.OperateOp, create bool) (*treeNode, error) {
	nd := &treeNode{kind: p.Kind, fi: -1, ri: -1, ci: -1}
	if p.Kind > wire.OperatePathCol {
		return nil, wire.ErrOperatePath
	}
	if p.Kind == wire.OperatePathRecord {
		// Ruling: the record's presence is whether it existed before this
		// call, so CHECK((), ABSENT) means "this call created it". Presence
		// "now" would make both comparators constants, since a call that
		// reaches the op list always has a record.
		nd.present = st.existed
		return nd, nil
	}
	if st.rec.Mode == wire.OperateModeSchema {
		return st.resolveSchema(nd, p, o, create)
	}
	return st.resolveDynamic(nd, p, o, create)
}

func (st *treeState) resolveSchema(nd *treeNode, p wire.OperatePath, o wire.OperateOp, create bool) (*treeNode, error) {
	s := st.rec.Schema
	if p.Field.ByName {
		// Names address a schema record only when it stores them (§2.4).
		if !s.StoreNames {
			return nil, wire.ErrOperatePath
		}
		pos, ok := s.FieldPos(p.Field.Name)
		if !ok {
			return nil, wire.ErrOperatePath
		}
		nd.fi = pos
	} else {
		if p.Field.Pos >= uint32(len(s.Fields)) {
			return nil, wire.ErrOperatePath
		}
		nd.fi = int(p.Field.Pos)
	}
	fd := &s.Fields[nd.fi]

	if p.Kind == wire.OperatePathField {
		nd.present = true // every declared schema field exists (§2.5)
		if fd.Type == wire.OperateTypeTable {
			nd.isTable = true
			nd.tbl = st.rec.Fields[nd.fi].Table
			return nd, nil
		}
		nd.typ, nd.n = fd.Type, fd.N
		if err := treeCheckSchemaType(o, fd.Type); err != nil {
			return nil, err
		}
		return nd, nil
	}

	if fd.Type != wire.OperateTypeTable {
		return nil, wire.ErrOperatePath // a row path into a non-table field
	}
	td := fd.Table
	nd.tbl = st.rec.Fields[nd.fi].Table
	if len(p.Key) != wire.CellWidth(td.KeyType, td.KeyN) {
		return nil, wire.ErrOperatePath
	}

	if p.Kind == wire.OperatePathCol {
		if p.Col.ByName {
			if !s.StoreNames {
				return nil, wire.ErrOperatePath
			}
			nd.ci = -1
			for j := range td.Cols {
				if p.Col.Name != "" && td.Cols[j].Name == p.Col.Name {
					nd.ci = j
					break
				}
			}
			if nd.ci < 0 {
				return nil, wire.ErrOperatePath
			}
		} else {
			if p.Col.Pos >= uint32(len(td.Cols)) {
				return nil, wire.ErrOperatePath
			}
			nd.ci = int(p.Col.Pos)
		}
		nd.typ, nd.n = td.Cols[nd.ci].Type, td.Cols[nd.ci].N
		if err := treeCheckSchemaType(o, td.Cols[nd.ci].Type); err != nil {
			return nil, err
		}
	}

	ri, found := treeFindRow(nd.tbl.Rows, p.Key, td.KeyType, false)
	nd.present = found
	if !found {
		if !create {
			return nd, nil
		}
		if err := treeInsertSchemaRow(nd.tbl, td, p.Key); err != nil {
			return nil, err
		}
		ri, _ = treeFindRow(nd.tbl.Rows, p.Key, td.KeyType, false)
	}
	nd.ri = ri
	return nd, nil
}

func (st *treeState) resolveDynamic(nd *treeNode, p wire.OperatePath, o wire.OperateOp, create bool) (*treeNode, error) {
	// Positions are unstable in dynamic mode (fields are kept sorted by
	// name), so only names address it (§2.4).
	if !p.Field.ByName {
		return nil, wire.ErrOperatePath
	}
	if len(p.Field.Name) > wire.OperateMaxNameLen {
		return nil, wire.ErrOperateCap
	}
	if p.Kind != wire.OperatePathField {
		if len(p.Key) == 0 {
			return nil, wire.ErrOperatePath
		}
		if len(p.Key) > wire.OperateMaxKeyLen {
			return nil, wire.ErrOperateCap
		}
	}
	if p.Kind == wire.OperatePathCol {
		if !p.Col.ByName {
			return nil, wire.ErrOperatePath
		}
		if len(p.Col.Name) > wire.OperateMaxNameLen {
			return nil, wire.ErrOperateCap
		}
	}
	fi := treeFindField(st.rec.Fields, p.Field.Name)

	if p.Kind == wire.OperatePathField {
		if fi < 0 {
			if !create {
				nd.typ, nd.n = treeZeroTypeFor(o)
				return nd, nil
			}
			typ, n, err := treeCreateType(o)
			if err != nil {
				return nil, err
			}
			if len(st.rec.Fields)+1 > wire.OperateMaxFields {
				return nil, wire.ErrOperateCap
			}
			st.rec.Fields = append(st.rec.Fields, wire.Field{Name: p.Field.Name, Cell: wire.ZeroCell(typ, n)})
			treeSortFields(st.rec)
			nd.fi = treeFindField(st.rec.Fields, p.Field.Name)
			nd.typ, nd.n = typ, n
			return nd, nil
		}
		nd.fi = fi
		nd.present = true
		f := &st.rec.Fields[fi]
		if f.Cell.Type == wire.OperateTypeTable {
			nd.isTable = true
			nd.tbl = f.Table
			return nd, nil
		}
		nd.typ, nd.n = f.Cell.Type, f.Cell.N
		if err := treeCheckDynType(o, f.Cell.Type); err != nil {
			return nil, err
		}
		if typ, n, ok := treeRetype(o); ok {
			nd.typ, nd.n, nd.retype = typ, n, true
		}
		return nd, nil
	}

	// Row and column paths: the field must be (or become) a table.
	if fi < 0 {
		if !create {
			nd.typ, nd.n = treeZeroTypeFor(o)
			return nd, nil
		}
		if len(st.rec.Fields)+1 > wire.OperateMaxFields {
			return nil, wire.ErrOperateCap
		}
		st.rec.Fields = append(st.rec.Fields, wire.Field{Name: p.Field.Name,
			Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: &wire.Table{}})
		treeSortFields(st.rec)
		fi = treeFindField(st.rec.Fields, p.Field.Name)
	} else if st.rec.Fields[fi].Cell.Type != wire.OperateTypeTable {
		return nil, wire.ErrOperatePath // a row path into a scalar field
	}
	nd.fi = fi
	nd.tbl = st.rec.Fields[fi].Table

	ri, found := treeFindRow(nd.tbl.Rows, p.Key, 0, true)
	if !found {
		if !create {
			nd.typ, nd.n = treeZeroTypeFor(o)
			return nd, nil
		}
		if nd.tbl.Cap > 0 && uint32(len(nd.tbl.Rows)) >= nd.tbl.Cap {
			treeEvictOne(nd.tbl, nd.tbl.Policy, int(nd.tbl.ByCol), nd.tbl.ByColName, true)
		}
		if len(nd.tbl.Rows)+1 > wire.OperateMaxRows {
			return nil, wire.ErrOperateCap
		}
		nd.tbl.Rows = append(nd.tbl.Rows, wire.Row{Key: append([]byte(nil), p.Key...)})
		treeSortRows(nd.tbl, 0, true)
		ri, _ = treeFindRow(nd.tbl.Rows, p.Key, 0, true)
	}
	nd.ri = ri
	if p.Kind == wire.OperatePathRow {
		nd.present = found
		return nd, nil
	}

	row := &nd.tbl.Rows[ri]
	ci := treeFindCol(row.Cols, p.Col.Name)
	if ci < 0 {
		if !create {
			nd.typ, nd.n = treeZeroTypeFor(o)
			return nd, nil
		}
		typ, n, err := treeCreateType(o)
		if err != nil {
			return nil, err
		}
		if len(row.Cols)+1 > wire.OperateMaxCols {
			return nil, wire.ErrOperateCap
		}
		row.Cols = append(row.Cols, wire.Col{Name: p.Col.Name, Cell: wire.ZeroCell(typ, n)})
		treeSortCols(row)
		nd.ci = treeFindCol(row.Cols, p.Col.Name)
		nd.typ, nd.n = typ, n
		return nd, nil
	}
	nd.ci = ci
	nd.present = true
	c := row.Cols[ci].Cell
	nd.typ, nd.n = c.Type, c.N
	if err := treeCheckDynType(o, c.Type); err != nil {
		return nil, err
	}
	if typ, n, ok := treeRetype(o); ok {
		nd.typ, nd.n, nd.retype = typ, n, true
	}
	return nd, nil
}

// treeInsertSchemaRow vivifies a row, evicting first if the table is full
// (design doc §2.3/§2.4).
func treeInsertSchemaRow(tbl *wire.Table, td *wire.TableDef, key []byte) error {
	if td.Cap > 0 && uint32(len(tbl.Rows)) >= td.Cap {
		treeEvictOne(tbl, td.Policy, int(td.ByCol), "", false)
	}
	if len(tbl.Rows)+1 > wire.OperateMaxRows {
		return wire.ErrOperateCap
	}
	row := wire.Row{Key: append([]byte(nil), key...), Cols: make([]wire.Col, len(td.Cols))}
	for c := range td.Cols {
		row.Cols[c] = wire.Col{Cell: wire.ZeroCell(td.Cols[c].Type, td.Cols[c].N)}
	}
	tbl.Rows = append(tbl.Rows, row)
	treeSortRows(tbl, td.KeyType, false)
	return nil
}

// --- type rules (design doc §2.4) ------------------------------------------

// treeIsControl reports whether an opcode's type byte is meaningless: the
// control, table and structural ops do not read or write a typed value, so
// they neither check nor create a type.
func treeIsControl(opcode uint8) bool {
	switch opcode {
	case wire.OperateOpIF, wire.OperateOpCHECK, wire.OperateOpDEL,
		wire.OperateOpTRIM, wire.OperateOpCONFIG, wire.OperateOpMIGRATE:
		return true
	default:
		return false
	}
}

// treeValidScalarType reports whether t names a type a value can have.
func treeValidScalarType(t uint8) bool {
	return t < wire.OperateTypeCount && t != wire.OperateTypeTable && t != wire.OperateTypeUnset
}

// treeDomain groups types the way §2.4's "the domain must match" rule does.
func treeDomain(t uint8) int {
	switch {
	case wire.TypeIsFloat(t):
		return 1
	case wire.TypeIsInt(t):
		return 0
	case wire.TypeIsBytes(t):
		return 2
	default:
		return -1
	}
}

// treeCheckSchemaType enforces §2.4's schema-mode rule: an op's type byte is
// either 0xFF ("from schema") or exactly the schema's type.
func treeCheckSchemaType(o wire.OperateOp, want uint8) error {
	if treeIsControl(o.Opcode) || o.Type == wire.OperateTypeFromSchema || o.Type == want {
		return nil
	}
	return wire.ErrOperateType
}

// treeCheckDynType enforces §2.4's dynamic-mode rule on an existing target:
// the op's domain must match the stored one, except for SET, which replaces
// the scalar including its type (§2.9).
func treeCheckDynType(o wire.OperateOp, stored uint8) error {
	if treeIsControl(o.Opcode) || o.Type == wire.OperateTypeFromSchema {
		return nil
	}
	if !treeValidScalarType(o.Type) {
		return wire.ErrOperateType
	}
	if o.Opcode == wire.OperateOpSET {
		return nil
	}
	if treeDomain(o.Type) != treeDomain(stored) {
		return wire.ErrOperateType
	}
	return nil
}

// treeCreateType is the type a missing dynamic field or column is created
// with: the op's own type byte, never "from schema" (there is no schema).
// A FIXED target takes its width from the operand bytes.
func treeCreateType(o wire.OperateOp) (uint8, uint8, error) {
	if o.Type == wire.OperateTypeFromSchema || !treeValidScalarType(o.Type) {
		return 0, 0, wire.ErrOperateType
	}
	if o.Type == wire.OperateTypeFixed {
		if len(o.Bytes) < 1 || len(o.Bytes) > wire.OperateMaxKeyLen {
			return 0, 0, wire.ErrOperateType
		}
		return o.Type, uint8(len(o.Bytes)), nil
	}
	return o.Type, 0, nil
}

// treeRetype reports the new type of a dynamic SET that names one.
func treeRetype(o wire.OperateOp) (uint8, uint8, bool) {
	if o.Opcode != wire.OperateOpSET || o.Type == wire.OperateTypeFromSchema {
		return 0, 0, false
	}
	typ, n, err := treeCreateType(o)
	if err != nil {
		return 0, 0, false
	}
	return typ, n, true
}

// treeZeroTypeFor is the type an absent dynamic target reads as. Ruling: the
// op's type byte picks the zero's domain when it names a type (so a bytes
// comparison against an absent field compares bytewise), and an integer zero
// is the fallback.
func treeZeroTypeFor(o wire.OperateOp) (uint8, uint8) {
	if treeValidScalarType(o.Type) {
		if o.Type != wire.OperateTypeFixed {
			return o.Type, 0
		}
		if len(o.Bytes) >= 1 && len(o.Bytes) <= wire.OperateMaxKeyLen {
			return o.Type, uint8(len(o.Bytes))
		}
	}
	return wire.OperateTypeU8, 0
}

// treeConvertCell re-types src as (typ, n) for a freeze (design doc §2.9):
// the domains must match and widths saturate, which is exactly a SET of the
// source value into a zero cell of the target type.
func treeConvertCell(src wire.Cell, present bool, typ, n uint8) (wire.Cell, error) {
	dst := wire.ZeroCell(typ, n)
	if !present {
		return dst, nil
	}
	if treeDomain(src.Type) != treeDomain(typ) || treeDomain(typ) < 0 {
		return wire.Cell{}, wire.ErrOperateType
	}
	var opnd wire.Operand
	switch treeDomain(typ) {
	case 0:
		opnd.A = treeIntOperand(src)
	case 1:
		opnd.A = int64(math.Float64bits(src.F))
	default:
		opnd.Bytes = src.B
	}
	return wire.ApplyScalar(wire.OperateOpSET, dst, false, opnd)
}

// treeIntOperand renders an integer cell as the int64 operand a SET carries,
// saturating an unsigned value that does not fit.
func treeIntOperand(c wire.Cell) int64 {
	if wire.TypeIsUnsigned(c.Type) && c.U > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(c.U)
}

// --- cells -----------------------------------------------------------------

// treeCellAt reads the scalar a resolved node addresses, or the zero of its
// declared type when it is absent (design doc §2.4, "absent is zero").
func treeCellAt(rec *wire.Record, nd *treeNode) wire.Cell {
	if nd.retype {
		return wire.ZeroCell(nd.typ, nd.n)
	}
	switch nd.kind {
	case wire.OperatePathField:
		if nd.fi < 0 {
			return wire.ZeroCell(nd.typ, nd.n)
		}
		if rec.Mode == wire.OperateModeSchema {
			return treeSchemaCell(rec.Fields[nd.fi].Cell, nd.typ, nd.n)
		}
		return rec.Fields[nd.fi].Cell
	case wire.OperatePathCol:
		if nd.fi < 0 || nd.ri < 0 || nd.ci < 0 {
			return wire.ZeroCell(nd.typ, nd.n)
		}
		c := rec.Fields[nd.fi].Table.Rows[nd.ri].Cols[nd.ci].Cell
		if rec.Mode == wire.OperateModeSchema {
			return treeSchemaCell(c, nd.typ, nd.n)
		}
		return c
	default:
		return wire.ZeroCell(nd.typ, nd.n)
	}
}

func treeSetCell(rec *wire.Record, nd *treeNode, c wire.Cell) {
	switch nd.kind {
	case wire.OperatePathField:
		rec.Fields[nd.fi].Cell = c
	case wire.OperatePathCol:
		rec.Fields[nd.fi].Table.Rows[nd.ri].Cols[nd.ci].Cell = c
	}
}

// treeSchemaCell re-types a tree cell at the schema's declared type: in
// schema mode the schema, never the cell's own Type, says how a value is
// stored, and a Go zero-value cell is the declared type's zero (mirrors
// wire's schemaCellOrZero, whose reasoning applies identically here).
func treeSchemaCell(c wire.Cell, typ, n uint8) wire.Cell {
	if c.Type == wire.OperateTypeUnset || treeIsZeroCell(c) {
		return wire.ZeroCell(typ, n)
	}
	return wire.Cell{Type: typ, N: n, U: c.U, F: c.F, B: c.B}
}

func treeIsZeroCell(c wire.Cell) bool {
	return c.Type == 0 && c.N == 0 && c.U == 0 && c.F == 0 && len(c.B) == 0
}

// --- rows, ordering and eviction -------------------------------------------

// treeCompareKey orders two row keys: bytewise in dynamic mode and for FIXED
// keys, as little-endian integers for the numeric key types (§2.3).
func treeCompareKey(a, b []byte, keyType uint8, dynamic bool) int {
	if dynamic || keyType == wire.OperateTypeFixed {
		return bytes.Compare(a, b)
	}
	var av, bv uint64
	for i := len(a) - 1; i >= 0; i-- {
		av = av<<8 | uint64(a[i])
	}
	for i := len(b) - 1; i >= 0; i-- {
		bv = bv<<8 | uint64(b[i])
	}
	switch {
	case av < bv:
		return -1
	case av > bv:
		return 1
	default:
		return 0
	}
}

func treeFindRow(rows []wire.Row, key []byte, keyType uint8, dynamic bool) (int, bool) {
	i := sort.Search(len(rows), func(i int) bool {
		return treeCompareKey(rows[i].Key, key, keyType, dynamic) >= 0
	})
	if i < len(rows) && treeCompareKey(rows[i].Key, key, keyType, dynamic) == 0 {
		return i, true
	}
	return i, false
}

func treeFindField(fields []wire.Field, name string) int {
	for i := range fields {
		if fields[i].Name == name {
			return i
		}
	}
	return -1
}

func treeFindCol(cols []wire.Col, name string) int {
	for i := range cols {
		if cols[i].Name == name {
			return i
		}
	}
	return -1
}

func treeSortRows(tbl *wire.Table, keyType uint8, dynamic bool) {
	sort.SliceStable(tbl.Rows, func(i, j int) bool {
		return treeCompareKey(tbl.Rows[i].Key, tbl.Rows[j].Key, keyType, dynamic) < 0
	})
}

func treeSortFields(rec *wire.Record) {
	sort.SliceStable(rec.Fields, func(i, j int) bool { return rec.Fields[i].Name < rec.Fields[j].Name })
}

func treeSortCols(row *wire.Row) {
	sort.SliceStable(row.Cols, func(i, j int) bool { return row.Cols[i].Name < row.Cols[j].Name })
}

// treeEvictOne removes the victim the policy names (design doc §2.3). Rows
// are kept sorted by key, so MIN_KEY/MAX_KEY are the ends of the slice.
// Ruling: PolicyNone with a cap above 0 passes Schema.Validate, so it still
// needs a deterministic victim — it evicts by MIN_KEY.
func treeEvictOne(tbl *wire.Table, policy uint8, byCol int, byColName string, dynamic bool) {
	if len(tbl.Rows) == 0 {
		return
	}
	victim := 0
	switch policy {
	case wire.OperatePolicyMaxKey:
		victim = len(tbl.Rows) - 1
	case wire.OperatePolicyMinCol, wire.OperatePolicyMaxCol:
		victim = treeColVictim(tbl, policy == wire.OperatePolicyMaxCol, byCol, byColName, dynamic)
	default:
		victim = 0
	}
	tbl.Rows = append(tbl.Rows[:victim], tbl.Rows[victim+1:]...)
}

// treeColVictim picks the row with the smallest (or largest) value in the
// eviction column; ties go to the smallest key, which is the first row in
// stored order.
func treeColVictim(tbl *wire.Table, wantMax bool, byCol int, byColName string, dynamic bool) int {
	best := 0
	bestCell := treeEvictCell(tbl.Rows[0], byCol, byColName, dynamic)
	for i := 1; i < len(tbl.Rows); i++ {
		c := treeEvictCell(tbl.Rows[i], byCol, byColName, dynamic)
		ord := treeCompareCells(c, bestCell)
		if (wantMax && ord > 0) || (!wantMax && ord < 0) {
			best, bestCell = i, c
		}
	}
	return best
}

// treeEvictCell reads a row's eviction column; a dynamic row that does not
// carry the column reads as zero (design doc §2.4, "absent is zero").
func treeEvictCell(row wire.Row, byCol int, byColName string, dynamic bool) wire.Cell {
	if dynamic {
		if ci := treeFindCol(row.Cols, byColName); ci >= 0 {
			return row.Cols[ci].Cell
		}
		return wire.Cell{Type: wire.OperateTypeU8}
	}
	if byCol >= 0 && byCol < len(row.Cols) {
		return row.Cols[byCol].Cell
	}
	return wire.Cell{Type: wire.OperateTypeU8}
}

// treeCompareCells orders two cells deterministically. Within a domain the
// comparison is the natural one; across domains (only reachable in dynamic
// mode, where two rows may store different types under one column name) the
// domain rank orders them, so the victim is still a pure function of the
// stored bytes.
func treeCompareCells(a, b wire.Cell) int {
	da, db := treeDomain(a.Type), treeDomain(b.Type)
	if da != db {
		switch {
		case da < db:
			return -1
		default:
			return 1
		}
	}
	switch da {
	case 1:
		switch {
		case a.F < b.F:
			return -1
		case a.F > b.F:
			return 1
		default:
			return 0
		}
	case 2:
		return bytes.Compare(a.B, b.B)
	case 0:
		return treeCompareInts(a, b)
	default:
		return 0
	}
}

func treeCompareInts(a, b wire.Cell) int {
	au, bu := wire.TypeIsUnsigned(a.Type), wire.TypeIsUnsigned(b.Type)
	switch {
	case au && bu:
		switch {
		case a.U < b.U:
			return -1
		case a.U > b.U:
			return 1
		default:
			return 0
		}
	case !au && !bu:
		x, y := int64(a.U), int64(b.U)
		switch {
		case x < y:
			return -1
		case x > y:
			return 1
		default:
			return 0
		}
	case au: // a unsigned, b signed
		return -treeCompareMixed(b, a)
	default: // a signed, b unsigned
		return treeCompareMixed(a, b)
	}
}

// treeCompareMixed compares a signed cell against an unsigned one exactly:
// a negative signed value is below every unsigned value, otherwise both fit
// in uint64.
func treeCompareMixed(signed, unsigned wire.Cell) int {
	s := int64(signed.U)
	if s < 0 {
		return -1
	}
	switch {
	case uint64(s) < unsigned.U:
		return -1
	case uint64(s) > unsigned.U:
		return 1
	default:
		return 0
	}
}

// --- record construction, copying and normalization ------------------------

func treeNewSchemaRecord(s *wire.Schema) *wire.Record {
	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s, Fields: make([]wire.Field, len(s.Fields))}
	treeNormalize(rec)
	return rec
}

func treeClone(r *wire.Record) *wire.Record {
	if r == nil {
		return nil
	}
	out := &wire.Record{Mode: r.Mode, Schema: r.Schema, Fields: make([]wire.Field, len(r.Fields))}
	for i := range r.Fields {
		f := &r.Fields[i]
		nf := wire.Field{Name: f.Name, Cell: treeCloneCell(f.Cell)}
		if f.Table != nil {
			t := &wire.Table{Cap: f.Table.Cap, Policy: f.Table.Policy, ByCol: f.Table.ByCol,
				ByColName: f.Table.ByColName, Rows: make([]wire.Row, len(f.Table.Rows))}
			for j := range f.Table.Rows {
				src := &f.Table.Rows[j]
				row := wire.Row{Key: append([]byte(nil), src.Key...), Cols: make([]wire.Col, len(src.Cols))}
				for k := range src.Cols {
					row.Cols[k] = wire.Col{Name: src.Cols[k].Name, Cell: treeCloneCell(src.Cols[k].Cell)}
				}
				t.Rows[j] = row
			}
			nf.Table = t
		}
		out.Fields[i] = nf
	}
	return out
}

func treeCloneCell(c wire.Cell) wire.Cell {
	if c.B != nil {
		c.B = append([]byte(nil), c.B...)
	}
	return c
}

// treeNormalize puts a record into the canonical shape the encoder expects:
// schema-mode fields and columns fully materialized at their declared types
// with the schema's eviction triple, dynamic fields and columns sorted by
// name, rows sorted by key (design doc §2.5).
func treeNormalize(rec *wire.Record) {
	if rec == nil {
		return
	}
	if rec.Mode != wire.OperateModeSchema {
		treeSortFields(rec)
		for i := range rec.Fields {
			if rec.Fields[i].Cell.Type == wire.OperateTypeTable {
				if rec.Fields[i].Table == nil {
					rec.Fields[i].Table = &wire.Table{}
				}
				treeSortRows(rec.Fields[i].Table, 0, true)
				for j := range rec.Fields[i].Table.Rows {
					treeSortCols(&rec.Fields[i].Table.Rows[j])
				}
			}
		}
		return
	}

	s := rec.Schema
	if len(rec.Fields) < len(s.Fields) {
		grown := make([]wire.Field, len(s.Fields))
		copy(grown, rec.Fields)
		rec.Fields = grown
	}
	rec.Fields = rec.Fields[:len(s.Fields)]
	for i := range s.Fields {
		fd := &s.Fields[i]
		f := &rec.Fields[i]
		f.Name = fd.Name
		if fd.Type != wire.OperateTypeTable {
			f.Table = nil
			f.Cell = treeSchemaCell(f.Cell, fd.Type, fd.N)
			continue
		}
		if f.Table == nil {
			f.Table = &wire.Table{}
		}
		f.Cell = wire.Cell{Type: wire.OperateTypeTable}
		// The schema, not the stored tree, owns the eviction triple.
		f.Table.Cap, f.Table.Policy, f.Table.ByCol = fd.Table.Cap, fd.Table.Policy, fd.Table.ByCol
		f.Table.ByColName = ""
		for r := range f.Table.Rows {
			row := &f.Table.Rows[r]
			if len(row.Cols) < len(fd.Table.Cols) {
				grown := make([]wire.Col, len(fd.Table.Cols))
				copy(grown, row.Cols)
				row.Cols = grown
			}
			row.Cols = row.Cols[:len(fd.Table.Cols)]
			for c := range fd.Table.Cols {
				row.Cols[c].Name = ""
				row.Cols[c].Cell = treeSchemaCell(row.Cols[c].Cell, fd.Table.Cols[c].Type, fd.Table.Cols[c].N)
			}
		}
		treeSortRows(f.Table, fd.Table.KeyType, false)
	}
}

// --- return specs (design doc §3.4) ----------------------------------------

func treeRets(rec *wire.Record, existed bool, rets []wire.OperateRet) ([][]byte, error) {
	if len(rets) == 0 {
		return nil, nil
	}
	vals := make([][]byte, 0, len(rets))
	for _, r := range rets {
		if r.Mode > wire.OperateRetCount {
			return nil, wire.ErrOperateArgs
		}
		// Ruling: against an absent record every return is absent, without
		// resolving the path — there is no record for it to resolve against.
		if rec == nil {
			if r.Mode == wire.OperateRetCount {
				vals = append(vals, treeCountValue(0))
			} else {
				vals = append(vals, []byte{wire.OperateTypeUnset})
			}
			continue
		}
		st := &treeState{rec: rec, existed: existed}
		nd, err := st.resolve(r.Path, wire.OperateOp{Opcode: wire.OperateOpIF, Type: wire.OperateTypeFromSchema}, false)
		if err != nil {
			return nil, err
		}
		if r.Mode == wire.OperateRetCount {
			vals = append(vals, treeCountValue(treeCountOf(st, nd)))
			continue
		}
		vals = append(vals, treeValueOf(rec, nd))
	}
	return vals, nil
}

func treeCountValue(n uint64) []byte {
	return wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU64, U: n})
}

func treeCountOf(st *treeState, nd *treeNode) uint64 {
	switch {
	case nd.kind == wire.OperatePathRecord:
		return uint64(st.fieldCount())
	case nd.isTable:
		if nd.tbl == nil {
			return 0
		}
		return uint64(len(nd.tbl.Rows))
	default:
		if nd.present {
			return 1
		}
		return 0
	}
}

func treeValueOf(rec *wire.Record, nd *treeNode) []byte {
	switch {
	case nd.kind == wire.OperatePathRecord:
		return rec.Encode()
	case nd.isTable:
		return treeTableValue(rec, nd.fi)
	case nd.kind == wire.OperatePathRow:
		if !nd.present || nd.ri < 0 {
			return []byte{wire.OperateTypeUnset}
		}
		return treeRowValue(rec, nd)
	default:
		if !nd.present {
			return []byte{wire.OperateTypeUnset}
		}
		return wire.AppendTaggedCell(nil, treeCellAt(rec, nd))
	}
}

// treeTableValue is the table's stored encoding with the record's mode byte
// in front (§3.4). Ruling: no schema blob rides along — the caller holds the
// schema, and the mode byte is all a reader needs to know which of the two
// table layouts follows. The bytes
// are exactly what the record encoder emits for that field: they are taken
// from a one-field record built around it, rather than re-implemented.
func treeTableValue(rec *wire.Record, fi int) []byte {
	if rec.Mode == wire.OperateModeSchema {
		sub := &wire.Schema{Version: rec.Schema.Version, StoreNames: rec.Schema.StoreNames,
			Fields: []wire.FieldDef{rec.Schema.Fields[fi]}}
		subRec := &wire.Record{Mode: wire.OperateModeSchema, Schema: sub, Fields: []wire.Field{rec.Fields[fi]}}
		enc := subRec.Encode()
		return append([]byte{wire.OperateModeSchema}, enc[1+len(sub.Encode()):]...)
	}
	name := rec.Fields[fi].Name
	subRec := &wire.Record{Mode: wire.OperateModeDynamic, Fields: []wire.Field{{
		Name: name, Cell: wire.Cell{Type: wire.OperateTypeTable}, Table: rec.Fields[fi].Table}}}
	enc := subRec.Encode()
	// [mode][nFields uvarint = 1][nlen u8][name][type u8] then the table.
	return append([]byte{wire.OperateModeDynamic}, enc[1+1+1+len(name)+1:]...)
}

// treeRowValue is [mode][nCols]{tagged} in column order (schema mode) or
// [mode][nCols]{[nlen][name] tagged} (dynamic mode), per §3.4.
func treeRowValue(rec *wire.Record, nd *treeNode) []byte {
	row := rec.Fields[nd.fi].Table.Rows[nd.ri]
	out := []byte{rec.Mode}
	if rec.Mode == wire.OperateModeSchema {
		cols := rec.Schema.Fields[nd.fi].Table.Cols
		out = binary.AppendUvarint(out, uint64(len(cols)))
		for c := range cols {
			cell := wire.ZeroCell(cols[c].Type, cols[c].N)
			if c < len(row.Cols) {
				cell = treeSchemaCell(row.Cols[c].Cell, cols[c].Type, cols[c].N)
			}
			out = wire.AppendTaggedCell(out, cell)
		}
		return out
	}
	out = binary.AppendUvarint(out, uint64(len(row.Cols)))
	for _, c := range row.Cols {
		out = append(out, byte(len(c.Name)))
		out = append(out, c.Name...)
		out = wire.AppendTaggedCell(out, c.Cell)
	}
	return out
}

// --- random schemas and calls ----------------------------------------------
//
// The generators the equivalence property tests drive both appliers with.
// They live here, next to the oracle, because every byte-level engine reuses
// them.

// randomScalarTypes are the types a record field may have; randomColTypes are
// the subset a table column may have (columns are fixed-width only, §2.3).
var randomScalarTypes = []uint8{
	wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64,
	wire.OperateTypeI8, wire.OperateTypeI16, wire.OperateTypeI32, wire.OperateTypeI64,
	wire.OperateTypeF32, wire.OperateTypeF64, wire.OperateTypeUVarint, wire.OperateTypeIVarint,
	wire.OperateTypeBytes, wire.OperateTypeFixed,
}

var randomColTypes = []uint8{
	wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64,
	wire.OperateTypeI8, wire.OperateTypeI16, wire.OperateTypeI32, wire.OperateTypeI64,
	wire.OperateTypeF32, wire.OperateTypeF64, wire.OperateTypeFixed,
}

var randomKeyTypes = []uint8{
	wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64, wire.OperateTypeFixed,
}

// randomSchema builds a small, always-valid schema: 1-6 fields with up to two
// tables, small caps so eviction fires often, and every eviction policy. Names
// are always assigned (so name paths are generatable) but only addressable
// when StoreNames is set.
func randomSchema(rng *rand.Rand) *wire.Schema {
	for {
		s := &wire.Schema{Version: 1, StoreNames: rng.Intn(2) == 0}
		nFields := 1 + rng.Intn(6)
		tables := 0
		for i := 0; i < nFields; i++ {
			f := wire.FieldDef{Name: fmt.Sprintf("f%d", i)}
			if tables < 2 && rng.Intn(3) == 0 {
				tables++
				f.Type = wire.OperateTypeTable
				f.Table = randomTableDef(rng)
			} else {
				f.Type = randomScalarTypes[rng.Intn(len(randomScalarTypes))]
				if f.Type == wire.OperateTypeFixed {
					f.N = uint8(1 + rng.Intn(4))
				}
			}
			s.Fields = append(s.Fields, f)
		}
		if s.Validate() == nil {
			return s
		}
	}
}

func randomTableDef(rng *rand.Rand) *wire.TableDef {
	td := &wire.TableDef{KeyType: randomKeyTypes[rng.Intn(len(randomKeyTypes))]}
	if td.KeyType == wire.OperateTypeFixed {
		td.KeyN = uint8(1 + rng.Intn(3))
	}
	for c := 0; c < 1+rng.Intn(3); c++ {
		cd := wire.ColumnDef{Name: fmt.Sprintf("c%d", c), Type: randomColTypes[rng.Intn(len(randomColTypes))]}
		if cd.Type == wire.OperateTypeFixed {
			cd.N = uint8(1 + rng.Intn(3))
		}
		td.Cols = append(td.Cols, cd)
	}
	td.Cap = uint32(rng.Intn(5))
	td.Policy = uint8(rng.Intn(int(wire.OperatePolicyMaxCol) + 1))
	if td.Policy == wire.OperatePolicyMinCol || td.Policy == wire.OperatePolicyMaxCol {
		td.ByCol = uint16(rng.Intn(len(td.Cols)))
	}
	return td
}

// randomArgs builds one call against s: 1-8 ops drawn from the whole
// vocabulary (including the control, table and structural ops and deliberately
// invalid ones, so the error paths are compared too) plus up to three return
// specs of both modes.
func randomArgs(rng *rand.Rand, s *wire.Schema) *wire.OperateArgs {
	a := &wire.OperateArgs{Create: wire.OperateCreateSchema, Schema: s.Encode()}
	if rng.Intn(24) == 0 {
		// A blob with something after it is not a schema blob, on a fresh
		// record or an existing one.
		a.Schema = append(a.Schema, byte(rng.Intn(256))) //nolint:gosec // deterministic test input
	}
	if rng.Intn(16) == 0 {
		a.Create = wire.OperateCreateNone
		a.Schema = nil
	}
	// A migration travels alone: it rewrites the schema the rest of the call
	// would have been written against.
	if rng.Intn(24) == 0 {
		a.Create = wire.OperateCreateNone
		a.Schema = nil
		a.Ops = []wire.OperateOp{{Opcode: wire.OperateOpMIGRATE, Type: wire.OperateTypeFromSchema,
			Path: recPath(), A: int64(s.Version), Bytes: randomExtension(rng, s).Encode()}}
		return a
	}

	nOps := 1 + rng.Intn(8)
	for i := 0; i < nOps; i++ {
		a.Ops = append(a.Ops, randomOp(rng, s))
	}
	for i := range a.Ops {
		if a.Ops[i].Opcode != wire.OperateOpIF {
			continue
		}
		if rng.Intn(16) == 0 {
			a.Ops[i].B = int64(len(a.Ops)) // out of range on purpose
			continue
		}
		a.Ops[i].B = int64(rng.Intn(len(a.Ops) - i))
	}
	for i := 0; i < rng.Intn(4); i++ {
		mode := uint8(wire.OperateRetValue)
		if rng.Intn(2) == 0 {
			mode = wire.OperateRetCount
		}
		a.Rets = append(a.Rets, wire.OperateRet{Mode: mode, Path: randomPath(rng, s)})
	}
	return a
}

// randomOpcodes is the whole vocabulary, plus one value that is not an opcode
// at all so the unknown-opcode path is compared too. randomOp draws from it
// directly only occasionally: a uniformly random opcode against a uniformly
// random target is almost always a type error, which would make the property
// compare error paths and little else.
var randomOpcodes = []uint8{
	wire.OperateOpSET, wire.OperateOpDEL, wire.OperateOpADD, wire.OperateOpMUL,
	wire.OperateOpMIN, wire.OperateOpMAX, wire.OperateOpAND, wire.OperateOpOR,
	wire.OperateOpXOR, wire.OperateOpSHL, wire.OperateOpSHR, wire.OperateOpSTAMP,
	wire.OperateOpCONFIG, wire.OperateOpTRIM, wire.OperateOpIF, wire.OperateOpCHECK, 99,
}

// opcodes that apply to each kind of target (design doc §3.1/§3.2).
var (
	intOpcodes = []uint8{wire.OperateOpSET, wire.OperateOpADD, wire.OperateOpMUL, wire.OperateOpMIN,
		wire.OperateOpMAX, wire.OperateOpAND, wire.OperateOpOR, wire.OperateOpXOR,
		wire.OperateOpSHL, wire.OperateOpSHR, wire.OperateOpSTAMP, wire.OperateOpDEL,
		wire.OperateOpIF, wire.OperateOpCHECK}
	varintOpcodes = []uint8{wire.OperateOpSET, wire.OperateOpADD, wire.OperateOpMUL, wire.OperateOpMIN,
		wire.OperateOpMAX, wire.OperateOpSTAMP, wire.OperateOpDEL, wire.OperateOpIF, wire.OperateOpCHECK}
	floatOpcodes = []uint8{wire.OperateOpSET, wire.OperateOpADD, wire.OperateOpMUL, wire.OperateOpMIN,
		wire.OperateOpMAX, wire.OperateOpDEL, wire.OperateOpIF, wire.OperateOpCHECK}
	bytesOpcodes = []uint8{wire.OperateOpSET, wire.OperateOpMIN, wire.OperateOpMAX,
		wire.OperateOpDEL, wire.OperateOpIF, wire.OperateOpCHECK}
	tableOpcodes = []uint8{wire.OperateOpDEL, wire.OperateOpTRIM, wire.OperateOpIF, wire.OperateOpCHECK}
	nodeOpcodes  = []uint8{wire.OperateOpDEL, wire.OperateOpIF, wire.OperateOpCHECK}
)

func randomOp(rng *rand.Rand, s *wire.Schema) wire.OperateOp {
	p := randomPath(rng, s)
	typ, n, ok := targetType(s, p)
	o := wire.OperateOp{Type: wire.OperateTypeFromSchema, Path: p}
	if !ok || rng.Intn(8) == 0 {
		o.Opcode = randomOpcodes[rng.Intn(len(randomOpcodes))]
	} else {
		pool := opcodePool(typ)
		o.Opcode = pool[rng.Intn(len(pool))]
	}
	switch o.Opcode {
	case wire.OperateOpIF, wire.OperateOpCHECK:
		o.Aux = uint8(rng.Intn(int(wire.OperateCmpAbsent) + 2)) // one past the last comparator
	case wire.OperateOpTRIM, wire.OperateOpCONFIG:
		o.A = int64(rng.Intn(5))
		if rng.Intn(16) == 0 {
			o.A = -1
		}
		o.Aux = uint8(rng.Intn(int(wire.OperatePolicyMaxCol) + 2))
		o.B = int64(rng.Intn(4))
		if rng.Intn(8) == 0 {
			// Out of range, including the two values that would alias column
			// 0 and column 2 if the operand were merely truncated to uint16.
			o.B = []int64{-1, -65536, 1 << 16, 1<<16 + 2, int64(wire.OperateMaxCols),
				math.MaxInt64}[rng.Intn(6)]
		}
	case wire.OperateOpSTAMP:
		o.Aux = uint8(rng.Intn(2))
	}
	o.A = randomOperandA(rng, o, typ)
	o.Bytes = randomOperandBytes(rng, typ, n)
	if rng.Intn(8) == 0 { // a type byte that may or may not match the schema
		if ok && rng.Intn(2) == 0 {
			o.Type = typ
		} else {
			o.Type = randomScalarTypes[rng.Intn(len(randomScalarTypes))]
		}
	}
	return o
}

// opcodePool is the set of opcodes that can apply to a target of type typ.
// OperateTypeTable stands for a table field and OperateTypeUnset for a row or
// the record itself — neither has a value of its own.
func opcodePool(typ uint8) []uint8 {
	switch {
	case typ == wire.OperateTypeTable:
		return tableOpcodes
	case typ == wire.OperateTypeUnset:
		return nodeOpcodes
	case wire.TypeIsFloat(typ):
		return floatOpcodes
	case wire.TypeIsFixedInt(typ):
		return intOpcodes
	case wire.TypeIsInt(typ):
		return varintOpcodes
	default:
		return bytesOpcodes
	}
}

// targetType reports the declared type (and FIXED width) of what p addresses,
// or ok == false when the path does not resolve at all — in which case the op
// is an error whatever it says, and randomOp draws its opcode uniformly.
func targetType(s *wire.Schema, p wire.OperatePath) (uint8, uint8, bool) {
	if p.Kind == wire.OperatePathRecord {
		return wire.OperateTypeUnset, 0, true
	}
	pos := -1
	if p.Field.ByName {
		if s.StoreNames {
			if i, found := s.FieldPos(p.Field.Name); found {
				pos = i
			}
		}
	} else if int(p.Field.Pos) < len(s.Fields) {
		pos = int(p.Field.Pos)
	}
	if pos < 0 {
		return 0, 0, false
	}
	fd := &s.Fields[pos]
	if p.Kind == wire.OperatePathField {
		return fd.Type, fd.N, true
	}
	if fd.Type != wire.OperateTypeTable || len(p.Key) != wire.CellWidth(fd.Table.KeyType, fd.Table.KeyN) {
		return 0, 0, false
	}
	if p.Kind == wire.OperatePathRow {
		return wire.OperateTypeUnset, 0, true
	}
	col := -1
	if p.Col.ByName {
		for j := range fd.Table.Cols {
			if p.Col.Name != "" && fd.Table.Cols[j].Name == p.Col.Name {
				col = j
			}
		}
	} else if int(p.Col.Pos) < len(fd.Table.Cols) {
		col = int(p.Col.Pos)
	}
	if col < 0 {
		return 0, 0, false
	}
	return fd.Table.Cols[col].Type, fd.Table.Cols[col].N, true
}

func randomOperandA(rng *rand.Rand, o wire.OperateOp, typ uint8) int64 {
	switch o.Opcode {
	case wire.OperateOpTRIM, wire.OperateOpCONFIG, wire.OperateOpSTAMP:
		return o.A
	}
	if wire.TypeIsFloat(typ) && rng.Intn(4) != 0 {
		// A float operand always travels as the bit pattern of a float64.
		return f64(float64(rng.Intn(200)-100) / 4)
	}
	switch rng.Intn(6) {
	case 0:
		return -1
	case 1:
		return 0
	case 2:
		return rng.Int63n(1 << 20)
	case 3:
		return rng.Int63n(1 << 40)
	case 4:
		return math.MaxInt64
	default:
		return int64(rng.Intn(8))
	}
}

// randomOperandBytes sizes a bytes operand for the target: exactly N for a
// FIXED one (any other width is ErrOperateType), otherwise short and drawn
// from a tiny alphabet so MIN/MAX comparisons actually change the value.
func randomOperandBytes(rng *rand.Rand, typ, n uint8) []byte {
	size := 1 + rng.Intn(4)
	if typ == wire.OperateTypeFixed && rng.Intn(8) != 0 {
		size = int(n)
	}
	b := make([]byte, size)
	for i := range b {
		b[i] = byte(rng.Intn(4))
	}
	return b
}

// randomPath builds a path against s: usually a valid one (a field, or a row
// or column of a table field), sometimes an out-of-range position, an unknown
// name, or a key of the wrong width.
func randomPath(rng *rand.Rand, s *wire.Schema) wire.OperatePath {
	if rng.Intn(12) == 0 {
		return recPath()
	}
	pos := rng.Intn(len(s.Fields))
	if rng.Intn(16) == 0 {
		return fieldPath(uint32(len(s.Fields) + rng.Intn(3)))
	}
	fd := &s.Fields[pos]
	byName := s.StoreNames && rng.Intn(3) == 0

	if fd.Type != wire.OperateTypeTable || rng.Intn(6) == 0 {
		if byName {
			return namePath(fd.Name)
		}
		return fieldPath(uint32(pos))
	}

	key := randomRowKey(rng, fd.Table)
	if rng.Intn(2) == 0 { // a row path
		if byName {
			return nameRowPath(fd.Name, key)
		}
		return rowPath(uint32(pos), key)
	}
	col := rng.Intn(len(fd.Table.Cols))
	if rng.Intn(16) == 0 {
		col = len(fd.Table.Cols) + rng.Intn(2)
	}
	if byName && col < len(fd.Table.Cols) {
		return nameColPath(fd.Name, key, fd.Table.Cols[col].Name)
	}
	return colPath(uint32(pos), key, uint32(col))
}

// randomRowKey draws from a small pool so rows collide often, occasionally
// returning a key of the wrong width.
func randomRowKey(rng *rand.Rand, td *wire.TableDef) []byte {
	w := wire.CellWidth(td.KeyType, td.KeyN)
	if rng.Intn(16) == 0 {
		w++
	}
	key := make([]byte, w)
	if w > 0 {
		key[0] = byte(rng.Intn(6))
	}
	return key
}

// randomExtension builds an append-only evolution of s (design doc §2.8):
// usually a legal one, sometimes not, so MIGRATE's rejection path is compared
// as well.
func randomExtension(rng *rand.Rand, s *wire.Schema) *wire.Schema {
	ns := &wire.Schema{Version: s.Version + 1, StoreNames: s.StoreNames}
	for i := range s.Fields {
		f := s.Fields[i]
		if f.Table != nil {
			td := *f.Table
			td.Cols = append([]wire.ColumnDef(nil), f.Table.Cols...)
			if rng.Intn(2) == 0 {
				td.Cols = append(td.Cols, wire.ColumnDef{
					Name: fmt.Sprintf("c%d", len(td.Cols)), Type: wire.OperateTypeU16})
			}
			td.Cap = uint32(rng.Intn(4))
			td.Policy = uint8(rng.Intn(int(wire.OperatePolicyMaxCol) + 1))
			td.ByCol = 0
			if td.Policy == wire.OperatePolicyMinCol || td.Policy == wire.OperatePolicyMaxCol {
				td.ByCol = uint16(rng.Intn(len(td.Cols)))
			}
			f.Table = &td
		}
		ns.Fields = append(ns.Fields, f)
	}
	if rng.Intn(2) == 0 {
		ns.Fields = append(ns.Fields, wire.FieldDef{
			Name: fmt.Sprintf("f%d", len(ns.Fields)), Type: randomScalarTypes[rng.Intn(len(randomScalarTypes))], N: 2})
	}
	if rng.Intn(8) == 0 && len(ns.Fields) > 0 { // not append-only: widen a field
		ns.Fields[0].Type = wire.OperateTypeU64
		ns.Fields[0].Table = nil
	}
	if ns.Validate() != nil {
		return s
	}
	return ns
}

// migratedSchema reports the schema a call migrates the record to, or nil if
// it is not a migration. The property test uses it to keep speaking the
// record's current version after a successful MIGRATE.
func migratedSchema(a *wire.OperateArgs) *wire.Schema {
	if len(a.Ops) != 1 || a.Ops[0].Opcode != wire.OperateOpMIGRATE {
		return nil
	}
	s, n, err := wire.DecodeSchema(a.Ops[0].Bytes)
	if err != nil || n != len(a.Ops[0].Bytes) {
		return nil
	}
	return s
}
