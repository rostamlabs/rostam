// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"sort"
)

// Record is the decoded tree form of a stored operate record (design doc
// §2.2/§2.9), in either mode. It is the reference representation: the
// server's byte-level apply engines (which patch stored bytes directly, per
// §2.6) must produce exactly the bytes DecodeRecord/Encode agree on, and
// clients decode a `get` result with this type.
//
// In schema mode, Fields[i] corresponds to Schema.Fields[i] by position; a
// TABLE field's Table is filled from the schema's TableDef (Cap/Policy/ByCol
// are informational copies — Encode ignores them and rebuilds them from
// Schema). In dynamic mode, Schema is nil and Fields is logically a set kept
// sorted by Name.
type Record struct {
	Mode   uint8
	Schema *Schema
	Fields []Field
}

// Field is one field of a Record tree. Cell.Type == OperateTypeTable implies
// Table != nil (and, in schema mode, means the schema at this position
// declares a TABLE field; in dynamic mode, Name carries the field's name and
// Cell.Type is otherwise the field's own stored type).
type Field struct {
	Name  string
	Cell  Cell
	Table *Table
}

// Table is a table field's rows plus its eviction triple (design doc §2.3).
// Rows are kept sorted by Key: numeric key types (U8..U64) compare as
// integers (the bytes are little-endian), OperateTypeFixed keys compare
// bytewise. In schema mode ByColName is unused (ByCol, a schema-relative
// column position, carries the information, and Cap/Policy/ByCol are
// informational — see Record's doc comment); in dynamic mode ByCol is
// unused (ByColName, a column name, carries the information).
type Table struct {
	Cap       uint32
	Policy    uint8
	ByCol     uint16
	ByColName string
	Rows      []Row
}

// Row is one row of a Table. In schema mode, Cols is addressed by position
// (Cols[i] is the schema's i'th column) and Name is unused; in dynamic mode
// Cols is kept sorted by Name.
type Row struct {
	Key  []byte
	Cols []Col
}

// Col is one column value of a Row. Name is used only in dynamic mode.
type Col struct {
	Name string
	Cell Cell
}

// Encode serializes r in the stored wire format (design doc §2.2 for schema
// mode, §2.9 for dynamic mode). It is deterministic: rows are sorted by key
// and, in dynamic mode, fields and columns are sorted by name, regardless of
// the order they appear in r — so op order and Go map/slice iteration never
// leak into stored bytes (design doc §2.5).
//
// Encode assumes r is a well-formed tree matching its Schema (schema mode):
// a wrong column count, a key of the wrong width, or an invalid Mode may
// produce garbage bytes or panic. DecodeRecord, not Encode, is the hardened
// side of this codec — it is the one exposed to untrusted (stored) bytes.
func (r *Record) Encode() []byte {
	switch r.Mode {
	case OperateModeSchema:
		return r.encodeSchema()
	case OperateModeDynamic:
		return r.encodeDynamic()
	default:
		return nil
	}
}

func (r *Record) encodeSchema() []byte {
	s := r.Schema
	layout, err := s.Layout()
	if err != nil {
		return nil
	}
	b := make([]byte, 0, 64)
	b = append(b, OperateModeSchema)
	b = append(b, s.Encode()...)

	for i := range s.Fields {
		if layout.FixedOff[i] < 0 {
			continue
		}
		f := &s.Fields[i]
		var tree Cell
		if i < len(r.Fields) {
			tree = r.Fields[i].Cell
		}
		b = AppendCellData(b, schemaCellOrZero(tree, i < len(r.Fields), f.Type, f.N))
	}

	for _, i := range layout.VarTail {
		f := &s.Fields[i]
		var tree Field
		present := i < len(r.Fields)
		if present {
			tree = r.Fields[i]
		}
		if f.Type == OperateTypeTable {
			b = appendSchemaTable(b, f.Table, tree.Table)
			continue
		}
		b = AppendCellData(b, schemaCellOrZero(tree.Cell, present, f.Type, f.N))
	}
	return b
}

// schemaCellOrZero returns tree's payload (U/F/B) re-typed at the schema's
// declared type and width (typ, n): in schema mode the SCHEMA, never the
// tree cell's own Type, determines how AppendCellData interprets a field or
// column (design doc §2.2 — "no tag stored"). tree.Type is otherwise
// ignored, because OperateTypeU8 is also Go's zero value for uint8 — a
// freshly zero-valued Cell{} (e.g. from a tree whose Fields slice is shorter
// than the schema) is bit-for-bit indistinguishable from an explicit
// Cell{Type: OperateTypeU8}, so branching on tree.Type alone would encode
// every never-touched non-U8 field as a 1-byte U8 zero instead of its real
// width, corrupting every fixed-offset field after it. A field that is
// absent, explicitly OperateTypeUnset, or the Go zero value in every other
// regard encodes as ZeroCell(typ, n) — for OperateTypeFixed this is n zero
// bytes, not zero bytes, which matters just as much.
func schemaCellOrZero(tree Cell, present bool, typ, n uint8) Cell {
	if !present || tree.Type == OperateTypeUnset || isZeroCell(tree) {
		return ZeroCell(typ, n)
	}
	return Cell{Type: typ, N: n, U: tree.U, F: tree.F, B: tree.B}
}

// isZeroCell reports whether c is the Go zero value of Cell in every field.
func isZeroCell(c Cell) bool {
	return c.Type == 0 && c.N == 0 && c.U == 0 && c.F == 0 && len(c.B) == 0
}

// appendSchemaTable appends a schema-mode table's value area (design doc
// §2.3): [nRows uvarint]{key col₀ col₁ …}*, rows sorted by key per tdef's key
// type. tbl may be nil (an unset TABLE field encodes as zero rows).
func appendSchemaTable(b []byte, tdef *TableDef, tbl *Table) []byte {
	var rows []Row
	if tbl != nil {
		rows = append([]Row(nil), tbl.Rows...)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		return compareKey(rows[i].Key, rows[j].Key, tdef.KeyType) < 0
	})
	b = binary.AppendUvarint(b, uint64(len(rows))) //nolint:gosec // bounded by OperateMaxRows on a well-formed tree
	for _, row := range rows {
		b = append(b, row.Key...)
		for c := range tdef.Cols {
			cd := &tdef.Cols[c]
			var tree Cell
			present := c < len(row.Cols)
			if present {
				tree = row.Cols[c].Cell
			}
			b = AppendCellData(b, schemaCellOrZero(tree, present, cd.Type, cd.N))
		}
	}
	return b
}

func (r *Record) encodeDynamic() []byte {
	fields := append([]Field(nil), r.Fields...)
	sort.SliceStable(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })

	b := []byte{OperateModeDynamic}
	b = binary.AppendUvarint(b, uint64(len(fields))) //nolint:gosec // bounded by OperateMaxFields on a well-formed tree
	for _, f := range fields {
		b = append(b, byte(len(f.Name))) //nolint:gosec // bounded by OperateMaxNameLen on a well-formed tree
		b = append(b, f.Name...)
		if f.Cell.Type == OperateTypeTable {
			b = append(b, OperateTypeTable)
			b = appendDynamicTable(b, f.Table)
			continue
		}
		b = append(b, f.Cell.Type)
		if f.Cell.Type == OperateTypeFixed {
			b = append(b, f.Cell.N)
		}
		b = AppendCellData(b, f.Cell)
	}
	return b
}

// appendDynamicTable appends a dynamic-mode table's value (design doc §2.9):
// [byteLen uvarint][cap uvarint][policy u8][byColLen u8][byColName]
// [nRows uvarint]{rows}*, byteLen covering everything after itself to the
// end of the table. t may be nil (an unset TABLE field encodes as an empty,
// zero-eviction table).
func appendDynamicTable(dst []byte, t *Table) []byte {
	var capV uint32
	var policy uint8
	var byColName string
	var rows []Row
	if t != nil {
		capV, policy, byColName = t.Cap, t.Policy, t.ByColName
		rows = append([]Row(nil), t.Rows...)
	}
	sort.SliceStable(rows, func(i, j int) bool { return bytes.Compare(rows[i].Key, rows[j].Key) < 0 })

	inner := make([]byte, 0, 32)
	inner = binary.AppendUvarint(inner, uint64(capV))
	inner = append(inner, policy)
	inner = append(inner, byte(len(byColName))) //nolint:gosec // bounded by OperateMaxNameLen on a well-formed tree
	inner = append(inner, byColName...)
	inner = binary.AppendUvarint(inner, uint64(len(rows))) //nolint:gosec // bounded by OperateMaxRows on a well-formed tree
	for _, row := range rows {
		rowBytes := appendDynamicRow(nil, row)
		inner = binary.AppendUvarint(inner, uint64(len(rowBytes))) //nolint:gosec // bounded by construction
		inner = append(inner, rowBytes...)
	}

	dst = binary.AppendUvarint(dst, uint64(len(inner))) //nolint:gosec // bounded by construction
	return append(dst, inner...)
}

// appendDynamicRow appends one dynamic-mode row's content (everything after
// its rowLen prefix): [klen u8][key][nCols uvarint]{cols}*, columns sorted
// by name. Encode assumes a well-formed tree (design doc §2.4: a dynamic
// row key is 1-255 bytes); a row with an empty Key encodes a klen of 0,
// which DecodeRecord — the hardened side of this codec — will then reject,
// so such a tree cannot round-trip. Encode does not itself guard against it.
func appendDynamicRow(dst []byte, row Row) []byte {
	dst = append(dst, byte(len(row.Key))) //nolint:gosec // bounded by OperateMaxKeyLen on a well-formed tree
	dst = append(dst, row.Key...)

	cols := append([]Col(nil), row.Cols...)
	sort.SliceStable(cols, func(i, j int) bool { return cols[i].Name < cols[j].Name })
	dst = binary.AppendUvarint(dst, uint64(len(cols))) //nolint:gosec // bounded by OperateMaxCols on a well-formed tree
	for _, c := range cols {
		dst = append(dst, byte(len(c.Name))) //nolint:gosec // bounded by OperateMaxNameLen on a well-formed tree
		dst = append(dst, c.Name...)
		dst = append(dst, c.Cell.Type)
		if c.Cell.Type == OperateTypeFixed {
			dst = append(dst, c.Cell.N)
		}
		dst = AppendCellData(dst, c.Cell)
	}
	return dst
}

// compareKey compares two table row keys the way the design mandates
// (§2.3): OperateTypeFixed keys bytewise, every other legal key type
// (U8..U64) as a little-endian-encoded unsigned integer.
func compareKey(a, b []byte, keyType uint8) int {
	if keyType == OperateTypeFixed {
		return bytes.Compare(a, b)
	}
	return compareLE(a, b)
}

// compareLE compares two little-endian byte strings (at most 8 bytes, per
// CellWidth's fixed-width int types) as unsigned integers.
func compareLE(a, b []byte) int {
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

// decodeCellDataCanonical wraps DecodeCellData with a re-encode check for
// the value shapes that can be spelled non-canonically:
//
//   - UVARINT/IVARINT and BYTES's length prefix: DecodeCellData (Task 1)
//     accepts any length-legal LEB128 encoding, not necessarily the minimal
//     one AppendCellData would produce (e.g. a two-byte over-long encoding
//     of the value 1, [0x81, 0x00]) — fine for the op-args wire format, but
//     a record byte stream carrying such an encoding would decode
//     successfully yet not re-encode identically.
//   - F32 and F64: AppendCellData canonicalizes every NaN to the quiet NaN
//     of design doc §2.5 (and narrows an F32 through float32), so exactly
//     one NaN bit pattern per float type re-encodes to itself. A stored NaN
//     spelled any other way — a signaling NaN, or a quiet NaN with a
//     non-zero payload, neither of which a correctly canonicalizing writer
//     produces, though a hostile or corrupt stream could contain one —
//     decodes fine yet re-encodes to different bytes. Non-NaN floats are
//     unaffected: an F64 keeps its bits verbatim and an F32 survives the
//     float64 widen/narrow round trip exactly.
//
// Rejecting a non-canonical encoding here, at every such read, keeps
// DecodeRecord(b).Encode() == b for every b it accepts (the identity
// FuzzDecodeRecord checks), without changing Task 1's cell codec. Every
// other type is fixed-width raw bytes with exactly one possible encoding, so
// the re-encode-and-compare would always trivially pass — skipped for those.
func decodeCellDataCanonical(t uint8, n uint8, b []byte) (Cell, int, error) {
	c, m, err := DecodeCellData(t, n, b)
	if err != nil {
		return Cell{}, 0, err
	}
	switch t {
	case OperateTypeUVarint, OperateTypeIVarint, OperateTypeBytes, OperateTypeF32, OperateTypeF64:
		if !bytes.Equal(AppendCellData(nil, c), b[:m]) {
			return Cell{}, 0, ErrOperateRecord
		}
	}
	return c, m, nil
}

// fitsRemaining reports whether a decoded byte-length n (as opposed to an
// element count — CountFitsIn is for those) fits within remaining bytes,
// without ever narrowing n to int: n can be an arbitrary attacker-controlled
// uvarint up to 2^64-1, and converting it to int first (as CountFitsIn does
// for element counts, after bounding them against a small cap) would wrap on
// a 32-bit int and defeat the check.
func fitsRemaining(n uint64, remaining int) bool {
	if remaining < 0 {
		return false
	}
	return n <= uint64(remaining) //nolint:gosec // remaining >= 0 checked above
}

// DecodeRecord reads one Encode-formatted record from the front of b,
// returning the decoded tree. b must be consumed exactly (trailing bytes are
// an error), so a successfully decoded record always satisfies
// bytes.Equal(r.Encode(), b) — the identity FuzzDecodeRecord checks.
//
// Every count (field, row, column) is bounded against the remaining input
// with CountFitsIn (or, for byte-length prefixes that are not element
// counts, fitsRemaining) before it is trusted to size an allocation; rows
// must arrive strictly ascending by key and, in dynamic mode, fields and
// columns strictly ascending by name — the invariant a follower's Encode
// would itself produce, checked here on the untrusted read side. A mode byte
// of 0 or greater than 2 is ErrOperateRecord. No input drives DecodeRecord to
// panic.
func DecodeRecord(b []byte) (*Record, error) {
	if len(b) < 1 {
		return nil, ErrOperateRecord
	}
	switch b[0] {
	case OperateModeSchema:
		return decodeSchemaRecord(b[1:])
	case OperateModeDynamic:
		return decodeDynamicRecord(b[1:])
	default:
		return nil, ErrOperateRecord
	}
}

func decodeSchemaRecord(b []byte) (*Record, error) {
	schema, n, err := DecodeSchema(b)
	if err != nil {
		return nil, err
	}
	off := n
	layout, err := schema.Layout()
	if err != nil {
		return nil, err
	}

	fields := make([]Field, len(schema.Fields))
	for i := range schema.Fields {
		if layout.FixedOff[i] < 0 {
			continue
		}
		f := &schema.Fields[i]
		cell, m, cerr := decodeCellDataCanonical(f.Type, f.N, b[off:])
		if cerr != nil {
			return nil, cerr
		}
		off += m
		fields[i].Cell = cell
	}

	budget := 0
	for _, i := range layout.VarTail {
		f := &schema.Fields[i]
		if f.Type == OperateTypeTable {
			tbl, m, terr := decodeSchemaTable(f.Table, layout.Tables[i], b[off:], &budget)
			if terr != nil {
				return nil, terr
			}
			off += m
			fields[i].Cell = Cell{Type: OperateTypeTable}
			fields[i].Table = tbl
			continue
		}
		cell, m, cerr := decodeCellDataCanonical(f.Type, f.N, b[off:])
		if cerr != nil {
			return nil, cerr
		}
		off += m
		fields[i].Cell = cell
	}

	if off != len(b) {
		return nil, ErrOperateRecord
	}
	return &Record{Mode: OperateModeSchema, Schema: schema, Fields: fields}, nil
}

// decodeSchemaTable reads a schema-mode table value ([nRows uvarint]{key
// col₀ col₁ …}* nRows, design doc §2.3) from the front of b, returning the
// decoded table and bytes consumed. budget accumulates every row decoded
// anywhere in the record (across every TABLE field) against OperateMaxRows,
// bounding total allocation even when several tables each stay under the
// per-table cap.
func decodeSchemaTable(tdef *TableDef, tl *TableLayout, b []byte, budget *int) (*Table, int, error) {
	nRows, m, err := decodeCanonicalUvarint(b)
	if err != nil {
		return nil, 0, err
	}
	off := m
	if nRows > OperateMaxRows {
		return nil, 0, ErrOperateRecord
	}
	if !CountFitsIn(int(nRows), len(b)-off, tl.RowWidth) {
		return nil, 0, ErrOperateRecord
	}
	*budget += int(nRows)
	if *budget > OperateMaxRows {
		return nil, 0, ErrOperateRecord
	}

	rows := make([]Row, nRows)
	var prevKey []byte
	for r := range rows {
		if len(b)-off < tl.KeyWidth {
			return nil, 0, ErrShortArgs
		}
		key := append([]byte(nil), b[off:off+tl.KeyWidth]...)
		off += tl.KeyWidth
		if r > 0 && compareKey(prevKey, key, tdef.KeyType) >= 0 {
			return nil, 0, ErrOperateRecord
		}
		prevKey = key

		cols := make([]Col, len(tdef.Cols))
		for c := range tdef.Cols {
			cd := &tdef.Cols[c]
			cell, m2, cerr := decodeCellDataCanonical(cd.Type, cd.N, b[off:])
			if cerr != nil {
				return nil, 0, cerr
			}
			off += m2
			cols[c] = Col{Cell: cell}
		}
		rows[r] = Row{Key: key, Cols: cols}
	}
	return &Table{Cap: tdef.Cap, Policy: tdef.Policy, ByCol: tdef.ByCol, Rows: rows}, off, nil
}

func decodeDynamicRecord(b []byte) (*Record, error) {
	nFields, m, err := decodeCanonicalUvarint(b)
	if err != nil {
		return nil, err
	}
	off := m
	if nFields > OperateMaxFields {
		return nil, ErrOperateRecord
	}
	if !CountFitsIn(int(nFields), len(b)-off, 3) {
		return nil, ErrOperateRecord
	}
	// nFields is already bounded by OperateMaxFields (65535) above, well
	// under OperateMaxRows (1<<20): no separate cap check is reachable here.
	budget := int(nFields)

	fields := make([]Field, nFields)
	var prevName string
	for i := range fields {
		if len(b)-off < 1 {
			return nil, ErrShortArgs
		}
		nlen := int(b[off])
		off++
		if nlen == 0 || nlen > OperateMaxNameLen {
			return nil, ErrOperateRecord
		}
		if len(b)-off < nlen {
			return nil, ErrShortArgs
		}
		name := string(b[off : off+nlen])
		off += nlen
		if i > 0 && name <= prevName {
			return nil, ErrOperateRecord
		}
		prevName = name

		if len(b)-off < 1 {
			return nil, ErrShortArgs
		}
		typ := b[off]
		off++

		if typ == OperateTypeTable {
			tbl, m2, terr := decodeDynamicTable(b[off:], &budget)
			if terr != nil {
				return nil, terr
			}
			off += m2
			fields[i] = Field{Name: name, Cell: Cell{Type: OperateTypeTable}, Table: tbl}
			continue
		}

		var n uint8
		if typ == OperateTypeFixed {
			if len(b)-off < 1 {
				return nil, ErrShortArgs
			}
			n = b[off]
			off++
			// FIXED's declared width is 1-255 (design doc §2.1); a stored 0
			// is a malformed record, not a zero-width value.
			if n == 0 {
				return nil, ErrOperateRecord
			}
		}
		cell, m2, cerr := decodeCellDataCanonical(typ, n, b[off:])
		if cerr != nil {
			return nil, cerr
		}
		off += m2
		fields[i] = Field{Name: name, Cell: cell}
	}

	if off != len(b) {
		return nil, ErrOperateRecord
	}
	return &Record{Mode: OperateModeDynamic, Fields: fields}, nil
}

// decodeDynamicTable reads a dynamic-mode table value ([byteLen uvarint]
// [cap uvarint][policy u8][byColLen u8][byColName][nRows uvarint]{rows}*,
// design doc §2.9) from the front of b. byteLen must fit the remaining
// bytes and the parse of the fields it covers must consume it exactly, and
// a MIN_COL/MAX_COL policy must name the column it evicts by.
// budget accumulates every field, row, and column decoded anywhere in the
// record against OperateMaxRows.
func decodeDynamicTable(b []byte, budget *int) (*Table, int, error) {
	byteLen, m, err := decodeCanonicalUvarint(b)
	if err != nil {
		return nil, 0, err
	}
	off := m
	if !fitsRemaining(byteLen, len(b)-off) {
		return nil, 0, ErrOperateRecord
	}
	inner := b[off : off+int(byteLen)]
	off += int(byteLen)

	ioff := 0
	capV, m2, err := decodeCanonicalUvarint(inner[ioff:])
	if err != nil {
		return nil, 0, err
	}
	ioff += m2
	if capV > OperateMaxRows {
		return nil, 0, ErrOperateRecord
	}

	if len(inner)-ioff < 1 {
		return nil, 0, ErrShortArgs
	}
	policy := inner[ioff]
	ioff++
	if policy > OperatePolicyMaxCol {
		return nil, 0, ErrOperateRecord
	}

	if len(inner)-ioff < 1 {
		return nil, 0, ErrShortArgs
	}
	byColLen := int(inner[ioff])
	ioff++
	if byColLen > OperateMaxNameLen {
		return nil, 0, ErrOperateRecord
	}
	if len(inner)-ioff < byColLen {
		return nil, 0, ErrShortArgs
	}
	byColName := string(inner[ioff : ioff+byColLen])
	ioff += byColLen
	// A MIN_COL/MAX_COL policy evicts by the value in a named column, so it
	// is meaningless without the name (design doc §3.2). Nothing that writes
	// a table produces this pairing — CONFIG rejects it with
	// wire.ErrOperatePath before it stores anything — so a record carrying
	// it is malformed, and accepting it would leave the eviction victim
	// undefined.
	if (policy == OperatePolicyMinCol || policy == OperatePolicyMaxCol) && byColLen == 0 {
		return nil, 0, ErrOperateRecord
	}

	nRows, m3, err := decodeCanonicalUvarint(inner[ioff:])
	if err != nil {
		return nil, 0, err
	}
	ioff += m3
	if nRows > OperateMaxRows {
		return nil, 0, ErrOperateRecord
	}
	if !CountFitsIn(int(nRows), len(inner)-ioff, 3) {
		return nil, 0, ErrOperateRecord
	}
	*budget += int(nRows)
	if *budget > OperateMaxRows {
		return nil, 0, ErrOperateRecord
	}

	rows := make([]Row, nRows)
	var prevKey []byte
	for r := range rows {
		rowLen, m4, rerr := decodeCanonicalUvarint(inner[ioff:])
		if rerr != nil {
			return nil, 0, rerr
		}
		ioff += m4
		if !fitsRemaining(rowLen, len(inner)-ioff) {
			return nil, 0, ErrOperateRecord
		}
		rowBuf := inner[ioff : ioff+int(rowLen)]
		ioff += int(rowLen)

		row, consumed, rowErr := decodeDynamicRow(rowBuf, budget)
		if rowErr != nil {
			return nil, 0, rowErr
		}
		if consumed != len(rowBuf) {
			return nil, 0, ErrOperateRecord
		}
		if r > 0 && bytes.Compare(prevKey, row.Key) >= 0 {
			return nil, 0, ErrOperateRecord
		}
		prevKey = row.Key
		rows[r] = row
	}

	if ioff != len(inner) {
		return nil, 0, ErrOperateRecord
	}
	return &Table{Cap: uint32(capV), Policy: policy, ByColName: byColName, Rows: rows}, off, nil //nolint:gosec // capV bounded by OperateMaxRows above
}

// decodeDynamicRow reads one dynamic-mode row's content (everything after
// its rowLen prefix, which the caller has already sliced exactly): [klen
// u8][key][nCols uvarint]{[nlen u8][name][type u8][n u8 iff FIXED][data]}*,
// columns strictly ascending by name.
func decodeDynamicRow(b []byte, budget *int) (Row, int, error) {
	if len(b) < 1 {
		return Row{}, 0, ErrShortArgs
	}
	klen := int(b[0])
	off := 1
	if klen == 0 || klen > OperateMaxKeyLen {
		return Row{}, 0, ErrOperateRecord
	}
	if len(b)-off < klen {
		return Row{}, 0, ErrShortArgs
	}
	key := append([]byte(nil), b[off:off+klen]...)
	off += klen

	nCols, m, err := decodeCanonicalUvarint(b[off:])
	if err != nil {
		return Row{}, 0, err
	}
	off += m
	if nCols > OperateMaxCols {
		return Row{}, 0, ErrOperateRecord
	}
	if !CountFitsIn(int(nCols), len(b)-off, 3) {
		return Row{}, 0, ErrOperateRecord
	}
	*budget += int(nCols)
	if *budget > OperateMaxRows {
		return Row{}, 0, ErrOperateRecord
	}

	cols := make([]Col, nCols)
	var prevName string
	for c := range cols {
		if len(b)-off < 1 {
			return Row{}, 0, ErrShortArgs
		}
		nlen := int(b[off])
		off++
		if nlen == 0 || nlen > OperateMaxNameLen {
			return Row{}, 0, ErrOperateRecord
		}
		if len(b)-off < nlen {
			return Row{}, 0, ErrShortArgs
		}
		name := string(b[off : off+nlen])
		off += nlen
		if c > 0 && name <= prevName {
			return Row{}, 0, ErrOperateRecord
		}
		prevName = name

		if len(b)-off < 1 {
			return Row{}, 0, ErrShortArgs
		}
		typ := b[off]
		off++
		var n uint8
		if typ == OperateTypeFixed {
			if len(b)-off < 1 {
				return Row{}, 0, ErrShortArgs
			}
			n = b[off]
			off++
			// Same rule as a dynamic field header: FIXED(0) is not a width.
			if n == 0 {
				return Row{}, 0, ErrOperateRecord
			}
		}
		cell, m2, cerr := decodeCellDataCanonical(typ, n, b[off:])
		if cerr != nil {
			return Row{}, 0, cerr
		}
		off += m2
		cols[c] = Col{Name: name, Cell: cell}
	}
	return Row{Key: key, Cols: cols}, off, nil
}
