// SPDX-License-Identifier: Apache-2.0

package ops

// The dynamic-mode byte engine (design doc §2.9): it navigates and patches a
// self-describing record's bytes directly, never building a tree. A field is
// a binary search over the name-sorted field index, a row is a walk with
// O(1) rowLen skips that stops early (rows are key-sorted), and a column is
// a walk within the row. Every edit that changes a width is a splice, and
// each enclosing length prefix (nCols, rowLen, nRows, byteLen, nFields) is
// rewritten from the inside out, so a prefix that changes width never
// invalidates an offset the same edit still has to use.
//
// The tree oracle (operate_oracle_test.go) is the definition of what this
// file must do; TestDynamicEngineMatchesOracle compares them byte for byte,
// and TestCrossModeEquivalence checks it against the schema engine.
//
// Stored bytes are untrusted — a plain `put` can store anything under the
// key — so newDynamicEngine validates the whole frame with a bounds-checked
// walk before any offset here is trusted, and every method below is written
// to be unable to panic on input that passed that walk.

import (
	"bytes"
	"encoding/binary"
	"math"
	"sort"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- index -----------------------------------------------------------------

// inlineDynFields is the number of field-index entries that ride inside the
// engine struct. Dynamic records in practice hold a handful of fields; a
// wider one falls back to a heap slice, which a pooled engine then keeps.
const inlineDynFields = 8

// dynField is one stored record field: the offsets of its parts within the
// engine's buffer, in stored (name-sorted) order.
//
//	[off: nlen u8][nameOff: name][typOff: type u8][n u8 iff FIXED][valOff: value][end
type dynField struct {
	off     int
	nameOff int
	nameLen int
	typOff  int
	valOff  int
	end     int
	typ     uint8
	n       uint8
}

// dynTable is one table field's header, parsed from its value area:
//
//	[lenOff: byteLen][capOff: cap][policyOff: policy][byColOff: byColLen][byColName]
//	[nRowsOff: nRows][rowsOff: rows…][end
type dynTable struct {
	lenOff, lenLen     int
	byteLen            int
	capOff, capLen     int
	policyOff          int
	byColOff, byColLen int // byColOff is the byColLen byte; byColLen the name's length
	nRowsOff, nRowsLen int
	rowsOff            int
	end                int
	nRows              int
	capV               uint32
	policy             uint8
}

// dynRow is one row of a dynamic table:
//
//	[off: rowLen][bodyOff/keyLenOff: klen][keyOff: key][nColsOff: nCols][colsOff: cols…][end
type dynRow struct {
	off                int
	lenLen             int
	bodyOff            int
	bodyLen            int
	keyOff, keyLen     int
	nColsOff, nColsLen int
	colsOff            int
	end                int
	nCols              int
}

// dynCol is one column of a dynamic row:
//
//	[off: nlen u8][nameOff: name][typOff: type u8][n u8 iff FIXED][valOff: data][end
type dynCol struct {
	off     int
	nameOff int
	nameLen int
	typOff  int
	valOff  int
	end     int
	typ     uint8
	n       uint8
}

// --- construction and structural validation --------------------------------

// dynamicEngine implements engine over a dynamic-mode record's stored bytes.
// buf is the engine's own copy, which it patches in place and splices as
// needed; fields is the name-sorted field index, rebuilt after every splice.
//
// inner is non-nil once MIGRATE has frozen the record into schema mode
// (design doc §2.9): the bytes are no longer dynamic, so every method
// delegates to a schema engine over the frozen buffer for the rest of the
// call — later ops and the return specs must see the frozen record.
type dynamicEngine struct {
	buf     []byte
	cache   *schemaCache
	fields  []dynField
	nOff    int // offset of the nFields uvarint (always 1)
	nLen    int
	existed bool
	deleted bool
	inner   *schemaEngine

	fieldsArr [inlineDynFields]dynField
	// enc builds one entry (a tagged cell, a fresh field, a row) before it is
	// spliced in. It is kept between calls so the splice paths do not
	// allocate on a warm engine.
	enc []byte
	// scratch encodes one fixed-width cell's data for the in-place path,
	// which is the only path the §2.6 allocation budget applies to.
	scratch [256]byte
}

// newDynamicEngine builds an engine over buf, which becomes the engine's
// private buffer (it is patched in place). buf is untrusted: the whole frame
// — mode byte, field entries, and every table's rows and columns — is
// validated here by a bounds-checked walk that must land exactly on the end
// of the buffer, and anything that does not fit is wire.ErrOperateRecord.
func newDynamicEngine(buf []byte) (*dynamicEngine, error) {
	e := &dynamicEngine{}
	if err := e.reset(buf, operateSchemas); err != nil {
		return nil, err
	}
	return e, nil
}

// reset re-points an engine at buf, so a pooled engine can be reused without
// re-allocating its field index. It performs the same validation as
// newDynamicEngine. cache is used only by a freeze, which needs the target
// schema's layout.
func (e *dynamicEngine) reset(buf []byte, cache *schemaCache) error {
	e.clear()
	e.cache = cache
	if len(buf) < 1 || buf[0] != wire.OperateModeDynamic {
		return wire.ErrOperateRecord
	}
	e.buf = buf
	if cap(e.fields) == 0 {
		e.fields = e.fieldsArr[:0]
	}
	return e.walk(true)
}

// clear drops the engine's references so a pooled engine does not pin a
// record buffer between calls. The field index keeps its capacity: it holds
// no pointers, so a large record's index is reused rather than re-allocated.
func (e *dynamicEngine) clear() {
	e.buf, e.cache, e.inner = nil, nil, nil
	e.fields = e.fields[:0]
	e.enc = e.enc[:0]
	e.nOff, e.nLen, e.existed, e.deleted = 0, 0, false, false
}

// setExisted records whether the record existed before this call, which is
// what a record-path EXISTS/ABSENT compares (design doc §2.4). A frozen
// record answers from its inner engine, so the flag travels there too.
func (e *dynamicEngine) setExisted(v bool) {
	e.existed = v
	if e.inner != nil {
		e.inner.existed = v
	}
}

// walk rebuilds the field index from the buffer, checking that the walk lands
// exactly on the end of it. With deep set it also validates every table's
// rows and columns — the untrusted-input pass newDynamicEngine/reset make
// once; a re-walk after a splice only needs the field level, since the bytes
// it is re-reading are ones this engine just wrote.
func (e *dynamicEngine) walk(deep bool) error {
	buf := e.buf
	nFields, m, err := readUvarint(buf[1:])
	if err != nil {
		return wire.ErrOperateRecord
	}
	if nFields > wire.OperateMaxFields {
		return wire.ErrOperateRecord
	}
	e.nOff, e.nLen = 1, m
	off := 1 + m
	if !wire.CountFitsIn(int(nFields), len(buf)-off, 3) { //nolint:gosec // bounded by OperateMaxFields above
		return wire.ErrOperateRecord
	}
	// Every field, row and column decoded anywhere in the record is charged
	// against one budget, exactly as wire.DecodeRecord does, so the two agree
	// on which records exist at all.
	budget := int(nFields) //nolint:gosec // bounded by OperateMaxFields above

	e.fields = e.fields[:0]
	if int(nFields) > cap(e.fields) { //nolint:gosec // bounded above
		e.fields = make([]dynField, 0, nFields)
	}
	prevOff, prevLen := 0, -1
	for i := 0; i < int(nFields); i++ { //nolint:gosec // bounded above
		var f dynField
		f.off = off
		if len(buf)-off < 1 {
			return wire.ErrOperateRecord
		}
		f.nameLen = int(buf[off])
		off++
		if len(buf)-off < f.nameLen {
			return wire.ErrOperateRecord
		}
		f.nameOff = off
		off += f.nameLen
		if prevLen >= 0 && bytes.Compare(buf[prevOff:prevOff+prevLen], buf[f.nameOff:off]) >= 0 {
			// Strictly ascending by name, the invariant the encoder produces
			// and the record decoder enforces.
			return wire.ErrOperateRecord
		}
		prevOff, prevLen = f.nameOff, f.nameLen

		if len(buf)-off < 1 {
			return wire.ErrOperateRecord
		}
		f.typOff = off
		f.typ = buf[off]
		off++

		if f.typ == wire.OperateTypeTable {
			f.valOff = off
			ln, terr := e.tableSpan(off, deep, &budget)
			if terr != nil {
				return terr
			}
			off += ln
		} else {
			if f.typ == wire.OperateTypeFixed {
				if len(buf)-off < 1 {
					return wire.ErrOperateRecord
				}
				f.n = buf[off]
				off++
			}
			f.valOff = off
			ln, cerr := cellDataLen(f.typ, f.n, buf[off:])
			if cerr != nil {
				return cerr
			}
			off += ln
		}
		f.end = off
		e.fields = append(e.fields, f)
	}
	if off != len(buf) {
		return wire.ErrOperateRecord
	}
	return nil
}

// tableSpan returns the stored length of the table value at off ([byteLen]
// plus the bytes it covers), validating the rows and columns inside it when
// deep is set.
func (e *dynamicEngine) tableSpan(off int, deep bool, budget *int) (int, error) {
	byteLen, m, err := readUvarint(e.buf[off:])
	if err != nil {
		return 0, wire.ErrOperateRecord
	}
	if !lenFits(byteLen, len(e.buf)-off-m) {
		return 0, wire.ErrOperateRecord
	}
	total := m + int(byteLen) //nolint:gosec // bounded by lenFits above
	if deep {
		if err := e.validateTable(off, off+m, int(byteLen), budget); err != nil { //nolint:gosec // bounded above
			return 0, err
		}
	}
	return total, nil
}

// validateTable walks a table's inner bytes (everything byteLen covers),
// checking the header, every row's exact extent, and that the rows arrive
// strictly ascending by key.
func (e *dynamicEngine) validateTable(lenOff, innerOff, innerLen int, budget *int) error {
	t, err := e.parseTableAt(lenOff, innerOff, innerLen)
	if err != nil {
		return err
	}
	if !wire.CountFitsIn(t.nRows, t.end-t.rowsOff, 3) {
		return wire.ErrOperateRecord
	}
	*budget += t.nRows
	if *budget > wire.OperateMaxRows {
		return wire.ErrOperateRecord
	}

	off := t.rowsOff
	prevOff, prevLen := 0, -1
	for i := 0; i < t.nRows; i++ {
		r, rerr := e.rowAt(off, t.end)
		if rerr != nil {
			return rerr
		}
		if err := e.validateRow(r, budget); err != nil {
			return err
		}
		if prevLen >= 0 && bytes.Compare(e.buf[prevOff:prevOff+prevLen], e.buf[r.keyOff:r.keyOff+r.keyLen]) >= 0 {
			return wire.ErrOperateRecord
		}
		prevOff, prevLen = r.keyOff, r.keyLen
		off = r.end
	}
	if off != t.end {
		return wire.ErrOperateRecord
	}
	return nil
}

// validateRow walks one row's columns, checking each cell's extent and that
// the columns arrive strictly ascending by name.
func (e *dynamicEngine) validateRow(r dynRow, budget *int) error {
	if !wire.CountFitsIn(r.nCols, r.end-r.colsOff, 3) {
		return wire.ErrOperateRecord
	}
	*budget += r.nCols
	if *budget > wire.OperateMaxRows {
		return wire.ErrOperateRecord
	}
	off := r.colsOff
	prevOff, prevLen := 0, -1
	for i := 0; i < r.nCols; i++ {
		c, err := e.colAt(off, r.end)
		if err != nil {
			return err
		}
		if prevLen >= 0 && bytes.Compare(e.buf[prevOff:prevOff+prevLen], e.buf[c.nameOff:c.nameOff+c.nameLen]) >= 0 {
			return wire.ErrOperateRecord
		}
		prevOff, prevLen = c.nameOff, c.nameLen
		off = c.end
	}
	if off != r.end {
		return wire.ErrOperateRecord
	}
	return nil
}

// lenFits reports whether a decoded byte length fits within remaining bytes
// without narrowing it to int first: a stored length is an arbitrary uvarint
// up to 2^64-1, and converting before comparing would wrap on a 32-bit int.
func lenFits(n uint64, remaining int) bool {
	if remaining < 0 {
		return false
	}
	return n <= uint64(remaining) //nolint:gosec // remaining >= 0 checked above
}

// cellDataLen returns the stored length of one cell's data of type typ (and,
// for FIXED, width n) at the front of b, rejecting anything the record codec
// would reject: a truncated value, a non-canonical varint or BYTES length, an
// over-long BYTES payload, an F32 bit pattern that does not survive the
// codec's float64 round trip, and any type byte that is not cell data at all
// (TABLE, or a tag past the last known type). It never allocates, which is
// what lets the structural walk stay off the heap.
func cellDataLen(typ, n uint8, b []byte) (int, error) {
	switch typ {
	case wire.OperateTypeUnset:
		return 0, nil
	case wire.OperateTypeU8, wire.OperateTypeI8,
		wire.OperateTypeU16, wire.OperateTypeI16,
		wire.OperateTypeU32, wire.OperateTypeI32,
		wire.OperateTypeU64, wire.OperateTypeI64, wire.OperateTypeF64:
		w := wire.CellWidth(typ, n)
		if len(b) < w {
			return 0, wire.ErrOperateRecord
		}
		return w, nil
	case wire.OperateTypeF32:
		if len(b) < 4 {
			return 0, wire.ErrOperateRecord
		}
		// The codec stores an F32 through a float64 (Cell.F), so a bit
		// pattern that does not come back identically — a signaling NaN on
		// hardware that quiets it — decodes fine yet re-encodes differently.
		// wire.DecodeRecord rejects those; so must this walk, or the two
		// would disagree on which records exist.
		v := binary.LittleEndian.Uint32(b[:4])
		if math.Float32bits(float32(float64(math.Float32frombits(v)))) != v {
			return 0, wire.ErrOperateRecord
		}
		return 4, nil
	case wire.OperateTypeUVarint, wire.OperateTypeIVarint:
		_, m, err := readUvarint(b)
		if err != nil {
			return 0, wire.ErrOperateRecord
		}
		return m, nil
	case wire.OperateTypeBytes:
		ln, m, err := readUvarint(b)
		if err != nil {
			return 0, wire.ErrOperateRecord
		}
		if ln > wire.OperateMaxBytesLen {
			return 0, wire.ErrOperateRecord
		}
		if !wire.CountFitsIn(int(ln), len(b)-m, 1) { //nolint:gosec // bounded above
			return 0, wire.ErrOperateRecord
		}
		return m + int(ln), nil //nolint:gosec // bounded above
	case wire.OperateTypeFixed:
		if !wire.CountFitsIn(int(n), len(b), 1) {
			return 0, wire.ErrOperateRecord
		}
		return int(n), nil
	default: // TABLE (never a cell) and every unknown tag
		return 0, wire.ErrOperateRecord
	}
}

// --- parsing helpers -------------------------------------------------------

// table parses the header of the table field at index fi.
func (e *dynamicEngine) table(fi int) (dynTable, error) {
	if fi < 0 || fi >= len(e.fields) || e.fields[fi].typ != wire.OperateTypeTable {
		return dynTable{}, wire.ErrOperatePath
	}
	f := &e.fields[fi]
	byteLen, m, err := readUvarint(e.buf[f.valOff:])
	if err != nil {
		return dynTable{}, wire.ErrOperateRecord
	}
	if !lenFits(byteLen, len(e.buf)-f.valOff-m) {
		return dynTable{}, wire.ErrOperateRecord
	}
	return e.parseTableAt(f.valOff, f.valOff+m, int(byteLen)) //nolint:gosec // bounded by lenFits above
}

// parseTableAt reads a table's header. lenOff is the byteLen prefix, innerOff
// the first byte it covers, and innerLen how many.
func (e *dynamicEngine) parseTableAt(lenOff, innerOff, innerLen int) (dynTable, error) {
	buf := e.buf
	t := dynTable{lenOff: lenOff, lenLen: innerOff - lenOff, byteLen: innerLen, end: innerOff + innerLen}
	if lenOff < 0 || innerOff < lenOff || t.end > len(buf) {
		return dynTable{}, wire.ErrOperateRecord
	}
	inner := buf[innerOff:t.end]

	capV, m, err := readUvarint(inner)
	if err != nil {
		return dynTable{}, wire.ErrOperateRecord
	}
	if capV > wire.OperateMaxRows {
		return dynTable{}, wire.ErrOperateRecord
	}
	t.capOff, t.capLen, t.capV = innerOff, m, uint32(capV) //nolint:gosec // bounded by OperateMaxRows above
	off := innerOff + m

	if t.end-off < 1 {
		return dynTable{}, wire.ErrOperateRecord
	}
	t.policyOff, t.policy = off, buf[off]
	off++
	if t.policy > wire.OperatePolicyMaxCol {
		return dynTable{}, wire.ErrOperateRecord
	}

	if t.end-off < 1 {
		return dynTable{}, wire.ErrOperateRecord
	}
	t.byColOff = off
	t.byColLen = int(buf[off])
	off++
	if t.end-off < t.byColLen {
		return dynTable{}, wire.ErrOperateRecord
	}
	off += t.byColLen

	nRows, m2, err := readUvarint(buf[off:t.end])
	if err != nil {
		return dynTable{}, wire.ErrOperateRecord
	}
	if nRows > wire.OperateMaxRows {
		return dynTable{}, wire.ErrOperateRecord
	}
	t.nRowsOff, t.nRowsLen, t.nRows = off, m2, int(nRows) //nolint:gosec // bounded by OperateMaxRows above
	t.rowsOff = off + m2
	return t, nil
}

// byColName is the table's eviction column name, as stored.
func (e *dynamicEngine) byColName(t dynTable) []byte {
	return e.buf[t.byColOff+1 : t.byColOff+1+t.byColLen]
}

// rowAt parses the row whose rowLen prefix starts at off; end bounds the
// table it lives in.
func (e *dynamicEngine) rowAt(off, end int) (dynRow, error) {
	buf := e.buf
	if off < 0 || off > end || end > len(buf) {
		return dynRow{}, wire.ErrOperateRecord
	}
	bodyLen, m, err := readUvarint(buf[off:end])
	if err != nil {
		return dynRow{}, wire.ErrOperateRecord
	}
	if !lenFits(bodyLen, end-off-m) {
		return dynRow{}, wire.ErrOperateRecord
	}
	r := dynRow{off: off, lenLen: m, bodyOff: off + m, bodyLen: int(bodyLen)} //nolint:gosec // bounded by lenFits above
	r.end = r.bodyOff + r.bodyLen

	if r.end-r.bodyOff < 1 {
		return dynRow{}, wire.ErrOperateRecord
	}
	r.keyLen = int(buf[r.bodyOff])
	r.keyOff = r.bodyOff + 1
	// A row key is 1-255 bytes (design doc §2.4): a zero-length key would
	// make two distinct rows compare equal and is rejected by the codec.
	if r.keyLen < 1 || r.end-r.keyOff < r.keyLen {
		return dynRow{}, wire.ErrOperateRecord
	}
	off2 := r.keyOff + r.keyLen

	nCols, m2, err := readUvarint(buf[off2:r.end])
	if err != nil {
		return dynRow{}, wire.ErrOperateRecord
	}
	if nCols > wire.OperateMaxCols {
		return dynRow{}, wire.ErrOperateRecord
	}
	r.nColsOff, r.nColsLen, r.nCols = off2, m2, int(nCols) //nolint:gosec // bounded by OperateMaxCols above
	r.colsOff = off2 + m2
	return r, nil
}

// colAt parses the column entry starting at off; end bounds the row.
func (e *dynamicEngine) colAt(off, end int) (dynCol, error) {
	buf := e.buf
	c := dynCol{off: off}
	if off < 0 || end > len(buf) || end-off < 1 {
		return dynCol{}, wire.ErrOperateRecord
	}
	c.nameLen = int(buf[off])
	c.nameOff = off + 1
	if end-c.nameOff < c.nameLen {
		return dynCol{}, wire.ErrOperateRecord
	}
	off = c.nameOff + c.nameLen
	if end-off < 1 {
		return dynCol{}, wire.ErrOperateRecord
	}
	c.typOff, c.typ = off, buf[off]
	off++
	if c.typ == wire.OperateTypeFixed {
		if end-off < 1 {
			return dynCol{}, wire.ErrOperateRecord
		}
		c.n = buf[off]
		off++
	}
	c.valOff = off
	ln, err := cellDataLen(c.typ, c.n, buf[off:end])
	if err != nil {
		return dynCol{}, err
	}
	c.end = off + ln
	return c, nil
}

// --- lookup ----------------------------------------------------------------

// compareBytesString orders stored bytes against a path segment's name
// without converting either: a []byte(name) conversion in the hot lookup
// would allocate once per resolve.
func compareBytesString(b []byte, s string) int {
	n := len(b)
	if len(s) < n {
		n = len(s)
	}
	for i := 0; i < n; i++ {
		switch {
		case b[i] < s[i]:
			return -1
		case b[i] > s[i]:
			return 1
		}
	}
	switch {
	case len(b) < len(s):
		return -1
	case len(b) > len(s):
		return 1
	default:
		return 0
	}
}

// findField binary-searches the name-sorted field index, returning the index
// and whether it matched; when it did not, the index is where a field with
// that name would be inserted.
func (e *dynamicEngine) findField(name string) (int, bool) {
	lo, hi := 0, len(e.fields)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		f := &e.fields[mid]
		if compareBytesString(e.buf[f.nameOff:f.nameOff+f.nameLen], name) < 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(e.fields) {
		f := &e.fields[lo]
		if compareBytesString(e.buf[f.nameOff:f.nameOff+f.nameLen], name) == 0 {
			return lo, true
		}
	}
	return lo, false
}

// findRow walks the table's rows for key, stopping as soon as it passes where
// the key would be (rows are sorted by key, compared bytewise in dynamic
// mode). On a miss the returned row's off is the insertion offset.
func (e *dynamicEngine) findRow(t dynTable, key []byte) (dynRow, int, bool, error) {
	off := t.rowsOff
	for i := 0; i < t.nRows; i++ {
		r, err := e.rowAt(off, t.end)
		if err != nil {
			return dynRow{}, 0, false, err
		}
		switch bytes.Compare(e.buf[r.keyOff:r.keyOff+r.keyLen], key) {
		case 0:
			return r, i, true, nil
		case 1:
			return dynRow{off: off}, i, false, nil
		}
		off = r.end
	}
	return dynRow{off: off}, t.nRows, false, nil
}

// rowByIndex walks to row ri of the table field at fi.
func (e *dynamicEngine) rowByIndex(fi, ri int) (dynTable, dynRow, error) {
	t, err := e.table(fi)
	if err != nil {
		return dynTable{}, dynRow{}, err
	}
	if ri < 0 || ri >= t.nRows {
		return dynTable{}, dynRow{}, wire.ErrOperatePath
	}
	off := t.rowsOff
	for i := 0; ; i++ {
		r, rerr := e.rowAt(off, t.end)
		if rerr != nil {
			return dynTable{}, dynRow{}, rerr
		}
		if i == ri {
			return t, r, nil
		}
		off = r.end
	}
}

// findCol walks the row's columns for name (they are sorted by name), and on
// a miss reports the insertion offset in the returned column's off.
func (e *dynamicEngine) findCol(r dynRow, name string) (dynCol, int, bool, error) {
	off := r.colsOff
	for i := 0; i < r.nCols; i++ {
		c, err := e.colAt(off, r.end)
		if err != nil {
			return dynCol{}, 0, false, err
		}
		switch compareBytesString(e.buf[c.nameOff:c.nameOff+c.nameLen], name) {
		case 0:
			return c, i, true, nil
		case 1:
			return dynCol{off: off}, i, false, nil
		}
		off = c.end
	}
	return dynCol{off: off}, r.nCols, false, nil
}

// colByIndex walks to column ci of a row.
func (e *dynamicEngine) colByIndex(r dynRow, ci int) (dynCol, error) {
	if ci < 0 || ci >= r.nCols {
		return dynCol{}, wire.ErrOperatePath
	}
	off := r.colsOff
	for i := 0; ; i++ {
		c, err := e.colAt(off, r.end)
		if err != nil {
			return dynCol{}, err
		}
		if i == ci {
			return c, nil
		}
		off = c.end
	}
}

// --- type rules (design doc §2.4/§2.9) -------------------------------------

// validDynScalarType reports whether t names a type a dynamic value can have.
func validDynScalarType(t uint8) bool {
	return t < wire.OperateTypeCount && t != wire.OperateTypeTable && t != wire.OperateTypeUnset
}

// dynDomain groups types the way §2.4's "the domain must match" rule does,
// and ranks them (int < float < bytes) for the cross-type ordering an
// eviction column needs when two rows store different types under one name.
func dynDomain(t uint8) int {
	switch {
	case wire.TypeIsInt(t):
		return 0
	case wire.TypeIsFloat(t):
		return 1
	case wire.TypeIsBytes(t):
		return 2
	default:
		return -1
	}
}

// createType is the type a missing dynamic field or column is created with
// (the oracle's treeCreateType): the op's own type byte, never "from schema"
// — there is no schema to take one from. n arrives from applyOps' fixedN.
func createType(typ, n uint8) (uint8, uint8, error) {
	if typ == wire.OperateTypeFromSchema || !validDynScalarType(typ) {
		return 0, 0, wire.ErrOperateType
	}
	if typ == wire.OperateTypeFixed && n < 1 {
		return 0, 0, wire.ErrOperateType
	}
	return typ, n, nil
}

// zeroType sanitizes the type an absent target reads as. applyOps hands it
// through resolve already sanitized (zeroTypeFor), so this only guards the
// callers that pass a bare 0.
func zeroType(typ, n uint8) (uint8, uint8) {
	if !validDynScalarType(typ) {
		return wire.OperateTypeU8, 0
	}
	if typ == wire.OperateTypeFixed && n < 1 {
		return wire.OperateTypeU8, 0
	}
	return typ, n
}

// dynTargetType is §2.4/§2.9's rule for an EXISTING dynamic scalar: a control op
// (create=false) never checks anything, 0xFF takes the stored type, SET
// replaces the type outright, and every other op must agree with the stored
// type's domain. It returns the type the op's value is computed and stored
// in (the oracle's treeCheckDynType + treeRetype).
func dynTargetType(create bool, opcode, typ, n, storedTyp, storedN uint8) (uint8, uint8, error) {
	if !create || typ == wire.OperateTypeFromSchema {
		return storedTyp, storedN, nil
	}
	if !validDynScalarType(typ) {
		return 0, 0, wire.ErrOperateType
	}
	if opcode == wire.OperateOpSET {
		// SET replaces the scalar including its type (design doc §2.9), so it
		// names a type exactly as it would on a missing target — including a
		// FIXED operand that cannot form a cell, which is wire.ErrOperateType
		// either way. There is no stored type left for such a SET to fall
		// back on, and §2.4 leaves no room for reinterpreting the op against
		// one.
		return createType(typ, n)
	}
	if dynDomain(typ) != dynDomain(storedTyp) {
		return 0, 0, wire.ErrOperateType
	}
	return storedTyp, storedN, nil
}

// --- resolve ---------------------------------------------------------------

// resolve implements engine.resolve for dynamic mode: fields are addressed by
// name only (positions are unstable, since fields are kept sorted by name),
// and create vivifies a field, a row, and a column as the path needs.
func (e *dynamicEngine) resolve(p wire.OperatePath, create bool, opcode, typ, n uint8) (ref, error) {
	if e.inner != nil {
		return e.inner.resolve(p, create, opcode, typ, n)
	}
	if p.Kind > wire.OperatePathCol {
		return ref{}, wire.ErrOperatePath
	}
	if opcode == wire.OperateOpCONFIG && p.Kind != wire.OperatePathField {
		// A table's eviction config lives on the field itself (design doc
		// §3.2), so a record, row or column path is malformed — checked
		// before the record path's early return below, since the oracle
		// rejects a CONFIG at () too.
		return ref{}, wire.ErrOperatePath
	}
	if p.Kind == wire.OperatePathRecord {
		// Presence of the record is whether it existed before this call
		// (oracle ruling): CHECK((), ABSENT) means "this call created it".
		return ref{kind: refKindRecord, field: -1, row: -1, col: -1, off: -1, present: e.existed}, nil
	}
	if !p.Field.ByName {
		return ref{}, wire.ErrOperatePath
	}
	if len(p.Field.Name) > wire.OperateMaxNameLen {
		return ref{}, wire.ErrOperateCap
	}
	if p.Kind != wire.OperatePathField {
		if len(p.Key) == 0 {
			return ref{}, wire.ErrOperatePath
		}
		if len(p.Key) > wire.OperateMaxKeyLen {
			return ref{}, wire.ErrOperateCap
		}
	}
	if p.Kind == wire.OperatePathCol {
		if !p.Col.ByName {
			return ref{}, wire.ErrOperatePath
		}
		if len(p.Col.Name) > wire.OperateMaxNameLen {
			return ref{}, wire.ErrOperateCap
		}
	}

	fi, found := e.findField(p.Field.Name)
	if p.Kind == wire.OperatePathField {
		return e.resolveField(p, create, opcode, typ, n, fi, found)
	}
	return e.resolveRowCol(p, create, opcode, typ, n, fi, found)
}

// resolveField resolves a field path, vivifying the field when create is set:
// an empty table for CONFIG (whose target is a table's eviction config), the
// op's own type's zero for anything else.
func (e *dynamicEngine) resolveField(p wire.OperatePath, create bool, opcode, typ, n uint8, fi int, found bool) (ref, error) {
	if !found {
		if !create {
			zt, zn := zeroType(typ, n)
			return ref{kind: refKindScalar, field: -1, row: -1, col: -1, off: -1, typ: zt, n: zn}, nil
		}
		if opcode == wire.OperateOpCONFIG {
			if err := e.insertTableField(fi, p.Field.Name); err != nil {
				return ref{}, err
			}
			idx, _ := e.findField(p.Field.Name)
			return ref{kind: refKindTable, field: idx, row: -1, col: -1, off: -1}, nil
		}
		ctyp, cn, err := createType(typ, n)
		if err != nil {
			return ref{}, err
		}
		if err := e.insertScalarField(fi, p.Field.Name, ctyp, cn); err != nil {
			return ref{}, err
		}
		idx, _ := e.findField(p.Field.Name)
		return ref{kind: refKindScalar, field: idx, row: -1, col: -1, off: e.fields[idx].typOff,
			typ: ctyp, n: cn}, nil
	}

	f := &e.fields[fi]
	if f.typ == wire.OperateTypeTable {
		return ref{kind: refKindTable, field: fi, row: -1, col: -1, off: -1, present: true}, nil
	}
	if opcode == wire.OperateOpCONFIG {
		// A scalar field cannot take a table's eviction config — it would
		// have to become a table, which only DEL then a row write can do
		// (design doc §2.9) — but that is config's rejection to make, after
		// the operand checks, which is the order the oracle rejects it in.
		// The op's TABLE type byte is not a scalar type and is not checked
		// against the stored one here for the same reason.
		return ref{kind: refKindScalar, field: fi, row: -1, col: -1, off: f.typOff,
			typ: f.typ, n: f.n, present: true}, nil
	}
	rtyp, rn, err := dynTargetType(create, opcode, typ, n, f.typ, f.n)
	if err != nil {
		return ref{}, err
	}
	return ref{kind: refKindScalar, field: fi, row: -1, col: -1, off: f.typOff,
		typ: rtyp, n: rn, present: true}, nil
}

// resolveRowCol resolves a row or column path, vivifying the table field, the
// row (evicting first if the table is at its cap) and the column as create
// requires.
func (e *dynamicEngine) resolveRowCol(p wire.OperatePath, create bool, opcode, typ, n uint8, fi int, found bool) (ref, error) { //nolint:gocognit // one vivification step per path segment, mirroring the oracle
	kind := refKindRow
	if p.Kind == wire.OperatePathCol {
		kind = refKindScalar
	}
	if !found {
		if !create {
			zt, zn := zeroType(typ, n)
			return ref{kind: kind, field: -1, row: -1, col: -1, off: -1, typ: zt, n: zn}, nil
		}
		if err := e.insertTableField(fi, p.Field.Name); err != nil {
			return ref{}, err
		}
		fi, _ = e.findField(p.Field.Name)
	} else if e.fields[fi].typ != wire.OperateTypeTable {
		return ref{}, wire.ErrOperatePath // a row path into a scalar field
	}

	t, err := e.table(fi)
	if err != nil {
		return ref{}, err
	}
	row, ri, rowFound, err := e.findRow(t, p.Key)
	if err != nil {
		return ref{}, err
	}
	if !rowFound {
		if !create {
			zt, zn := zeroType(typ, n)
			return ref{kind: kind, field: fi, row: -1, col: -1, off: -1, typ: zt, n: zn}, nil
		}
		if t.capV > 0 && uint32(t.nRows) >= t.capV { //nolint:gosec // nRows bounded by OperateMaxRows
			if err := e.evictOne(fi, t.policy, e.byColName(t)); err != nil {
				return ref{}, err
			}
			if t, err = e.table(fi); err != nil {
				return ref{}, err
			}
			if row, ri, rowFound, err = e.findRow(t, p.Key); err != nil {
				return ref{}, err
			}
		}
		if !rowFound {
			if t.nRows+1 > wire.OperateMaxRows {
				return ref{}, wire.ErrOperateCap
			}
			if err := e.insertRow(fi, t, row.off, p.Key); err != nil {
				return ref{}, err
			}
			if t, err = e.table(fi); err != nil {
				return ref{}, err
			}
			if row, ri, _, err = e.findRow(t, p.Key); err != nil {
				return ref{}, err
			}
		}
	}
	if p.Kind == wire.OperatePathRow {
		return ref{kind: refKindRow, field: fi, row: ri, col: -1, off: -1, present: rowFound}, nil
	}

	col, ci, colFound, err := e.findCol(row, p.Col.Name)
	if err != nil {
		return ref{}, err
	}
	if !colFound {
		if !create {
			zt, zn := zeroType(typ, n)
			return ref{kind: refKindScalar, field: fi, row: ri, col: -1, off: -1, typ: zt, n: zn}, nil
		}
		ctyp, cn, cerr := createType(typ, n)
		if cerr != nil {
			return ref{}, cerr
		}
		if row.nCols+1 > wire.OperateMaxCols {
			return ref{}, wire.ErrOperateCap
		}
		if err := e.insertCol(fi, ri, col.off, p.Col.Name, ctyp, cn); err != nil {
			return ref{}, err
		}
		_, row, err = e.rowByIndex(fi, ri)
		if err != nil {
			return ref{}, err
		}
		if col, ci, _, err = e.findCol(row, p.Col.Name); err != nil {
			return ref{}, err
		}
		return ref{kind: refKindScalar, field: fi, row: ri, col: ci, off: col.typOff, typ: ctyp, n: cn}, nil
	}
	rtyp, rn, err := dynTargetType(create, opcode, typ, n, col.typ, col.n)
	if err != nil {
		return ref{}, err
	}
	return ref{kind: refKindScalar, field: fi, row: ri, col: ci, off: col.typOff,
		typ: rtyp, n: rn, present: true}, nil
}

// --- get and set -----------------------------------------------------------

// get implements engine.get: the scalar at r, or its type's zero when the
// target is absent — or when a SET is about to replace the stored type, since
// the new type's zero is what the operand applies to (design doc §2.9).
func (e *dynamicEngine) get(r ref) (wire.Cell, error) {
	if e.inner != nil {
		return e.inner.get(r)
	}
	if r.kind != refKindScalar {
		return wire.Cell{}, wire.ErrOperatePath
	}
	if r.off < 0 {
		return wire.ZeroCell(r.typ, r.n), nil
	}
	storedTyp, storedN, dataOff, err := e.taggedAt(r.off)
	if err != nil {
		return wire.Cell{}, err
	}
	if storedTyp != r.typ || storedN != r.n {
		return wire.ZeroCell(r.typ, r.n), nil
	}
	c, _, err := wire.DecodeCellData(r.typ, r.n, e.buf[dataOff:])
	if err != nil {
		return wire.Cell{}, err
	}
	return c, nil
}

// taggedAt reads the [type][n iff FIXED] header a field or column stores in
// front of its data, returning the type, the width, and the data's offset.
func (e *dynamicEngine) taggedAt(off int) (uint8, uint8, int, error) {
	if off < 0 || off >= len(e.buf) {
		return 0, 0, 0, wire.ErrOperateRecord
	}
	typ := e.buf[off]
	off++
	var n uint8
	if typ == wire.OperateTypeFixed {
		if off >= len(e.buf) {
			return 0, 0, 0, wire.ErrOperateRecord
		}
		n = e.buf[off]
		off++
	}
	return typ, n, off, nil
}

// taggedEnd is one past the last byte of the [type][n?][data] entry at off.
func (e *dynamicEngine) taggedEnd(off int) (int, error) {
	typ, n, dataOff, err := e.taggedAt(off)
	if err != nil {
		return 0, err
	}
	ln, err := cellDataLen(typ, n, e.buf[dataOff:])
	if err != nil {
		return 0, err
	}
	return dataOff + ln, nil
}

// set implements engine.set: a cell whose stored width does not change is
// overwritten in place — the whole point of the byte engine — and anything
// else (a new type, a varint that grew, a BYTES value) is spliced, with every
// enclosing length prefix rewritten.
func (e *dynamicEngine) set(r ref, c wire.Cell) error {
	if e.inner != nil {
		return e.inner.set(r, c)
	}
	if r.kind != refKindScalar || r.off < 0 {
		return wire.ErrOperatePath
	}
	storedTyp, storedN, dataOff, err := e.taggedAt(r.off)
	if err != nil {
		return err
	}
	if c.Type == wire.OperateTypeFixed && len(c.B) != int(c.N) {
		return wire.ErrOperateType
	}
	if c.Type == wire.OperateTypeBytes && len(c.B) > wire.OperateMaxBytesLen {
		return wire.ErrOperateCap
	}
	if storedTyp == c.Type && storedN == c.N {
		if w := wire.CellWidth(c.Type, c.N); w >= 0 {
			if len(e.buf)-dataOff < w {
				return wire.ErrOperateRecord
			}
			copy(e.buf[dataOff:dataOff+w], wire.AppendCellData(e.scratch[:0], c))
			return nil
		}
	}
	end, err := e.taggedEnd(r.off)
	if err != nil {
		return err
	}
	e.enc = wire.AppendTaggedCell(e.enc[:0], c)
	return e.spliceValue(r, r.off, end-r.off, e.enc)
}

// spliceValue replaces the entry at [off, off+oldLen) with repl and rewrites
// the enclosing lengths: nothing for a record field (its bytes are the
// record's own), rowLen and the table's byteLen for a column.
func (e *dynamicEngine) spliceValue(r ref, off, oldLen int, repl []byte) error {
	delta := len(repl) - oldLen
	if r.row < 0 {
		if err := e.charge(delta); err != nil {
			return err
		}
		e.buf = splice(e.buf, off, oldLen, repl)
		return e.reindex()
	}
	// The row's and the table's length prefixes can widen along with the value
	// they cover, so their worst case is charged before the first splice too.
	if err := e.charge(delta + 2*binary.MaxVarintLen64); err != nil {
		return err
	}
	t, row, err := e.rowByIndex(r.field, r.row)
	if err != nil {
		return err
	}
	e.buf = splice(e.buf, off, oldLen, repl)
	if err := e.patchRowLen(t, row, delta); err != nil {
		return err
	}
	return e.reindex()
}

// patchRowLen rewrites a row's length prefix and then the table's byteLen,
// from the inside out: the row's prefix sits after the table's, so rewriting
// it (and changing its width) cannot move the table's own offset. bodyDelta
// is how much the row's body grew or shrank.
func (e *dynamicEngine) patchRowLen(t dynTable, row dynRow, bodyDelta int) error {
	newBody := row.bodyLen + bodyDelta
	if newBody < 0 {
		return wire.ErrOperateRecord
	}
	buf, m := spliceUvarint(e.buf, row.off, row.lenLen, uint64(newBody)) //nolint:gosec // checked non-negative above
	e.buf = buf
	return e.patchTableLen(t, bodyDelta+m-row.lenLen)
}

// patchTableLen rewrites just the table's byteLen after an edit inside it.
func (e *dynamicEngine) patchTableLen(t dynTable, delta int) error {
	newByteLen := t.byteLen + delta
	if newByteLen < 0 {
		return wire.ErrOperateRecord
	}
	e.buf, _ = spliceUvarint(e.buf, t.lenOff, t.lenLen, uint64(newByteLen)) //nolint:gosec // checked non-negative above
	return nil
}

// charge rejects a growth that would take the record past the §2.7
// record-size cap before the splice allocates for it.
func (e *dynamicEngine) charge(delta int) error {
	if delta > 0 && len(e.buf)+delta > maxOperateRecordBytes {
		return wire.ErrOperateCap
	}
	return nil
}

// reindex re-walks the field level after a splice.
func (e *dynamicEngine) reindex() error { return e.walk(false) }

// --- splice paths: fields, rows, columns -----------------------------------

// insertScalarField splices a fresh field, holding its type's zero, in at its
// sorted position idx.
func (e *dynamicEngine) insertScalarField(idx int, name string, typ, n uint8) error {
	e.enc = appendDynName(e.enc[:0], name)
	e.enc = wire.AppendTaggedCell(e.enc, wire.ZeroCell(typ, n))
	return e.insertField(idx, e.enc)
}

// insertTableField splices an empty table field — no rows and no eviction
// config — in at its sorted position idx.
func (e *dynamicEngine) insertTableField(idx int, name string) error {
	e.enc = appendDynName(e.enc[:0], name)
	e.enc = append(e.enc, wire.OperateTypeTable)
	// [cap 0][policy 0][byColLen 0][nRows 0], and the byteLen that covers it.
	e.enc = binary.AppendUvarint(e.enc, 4)
	e.enc = append(e.enc, 0, 0, 0, 0)
	return e.insertField(idx, e.enc)
}

// insertField splices one encoded field entry in at sorted position idx and
// rewrites nFields.
func (e *dynamicEngine) insertField(idx int, entry []byte) error {
	if len(e.fields)+1 > wire.OperateMaxFields {
		return wire.ErrOperateCap
	}
	at := len(e.buf)
	if idx < len(e.fields) {
		at = e.fields[idx].off
	}
	if err := e.charge(len(entry) + binary.MaxVarintLen64); err != nil {
		return err
	}
	e.buf = splice(e.buf, at, 0, entry)
	e.buf, _ = spliceUvarint(e.buf, e.nOff, e.nLen, uint64(len(e.fields)+1))
	return e.reindex()
}

// removeField splices field fi out entirely and rewrites nFields. A dynamic
// field is removed, not zeroed: there is no schema declaring it (design doc
// §2.5), which is also why losing the last one deletes the record.
func (e *dynamicEngine) removeField(fi int) error {
	if fi < 0 || fi >= len(e.fields) {
		return nil
	}
	f := e.fields[fi]
	e.buf = splice(e.buf, f.off, f.end-f.off, nil)
	e.buf, _ = spliceUvarint(e.buf, e.nOff, e.nLen, uint64(len(e.fields)-1))
	return e.reindex()
}

// insertRow splices an empty row (key, no columns) in at its sorted offset
// and rewrites nRows and the table's byteLen.
func (e *dynamicEngine) insertRow(fi int, t dynTable, at int, key []byte) error {
	body := append(e.enc[:0], byte(len(key))) //nolint:gosec // key length bounded by OperateMaxKeyLen in resolve
	body = append(body, key...)
	body = append(body, 0) // nCols = 0
	e.enc = body
	var hdr [binary.MaxVarintLen64]byte
	m := binary.PutUvarint(hdr[:], uint64(len(body)))
	// The row's bytes plus the worst case of the two prefixes that cover it
	// widening: the table's nRows and its byteLen.
	if err := e.charge(m + len(body) + 2*binary.MaxVarintLen64); err != nil {
		return err
	}
	e.buf = splice(e.buf, at, 0, hdr[:m])
	e.buf = splice(e.buf, at+m, 0, body)

	rowBytes := m + len(body)
	buf, m2 := spliceUvarint(e.buf, t.nRowsOff, t.nRowsLen, uint64(t.nRows+1)) //nolint:gosec // nRows >= 0
	e.buf = buf
	if err := e.patchTableLen(t, rowBytes+m2-t.nRowsLen); err != nil {
		return err
	}
	return e.reindex()
}

// removeRow splices row ri out of the table field at fi and rewrites nRows
// and byteLen.
func (e *dynamicEngine) removeRow(fi, ri int) error {
	t, row, err := e.rowByIndex(fi, ri)
	if err != nil {
		return err
	}
	removed := row.end - row.off
	e.buf = splice(e.buf, row.off, removed, nil)
	buf, m := spliceUvarint(e.buf, t.nRowsOff, t.nRowsLen, uint64(t.nRows-1)) //nolint:gosec // ri < nRows, so nRows >= 1
	e.buf = buf
	if err := e.patchTableLen(t, m-t.nRowsLen-removed); err != nil {
		return err
	}
	return e.reindex()
}

// insertCol splices a fresh column, holding its type's zero, into row ri at
// its sorted offset, and rewrites nCols, rowLen and byteLen.
func (e *dynamicEngine) insertCol(fi, ri, at int, name string, typ, n uint8) error {
	e.enc = appendDynName(e.enc[:0], name)
	e.enc = wire.AppendTaggedCell(e.enc, wire.ZeroCell(typ, n))
	entry := e.enc
	// The column's bytes plus the worst case of the three prefixes that cover
	// it widening: the row's nCols and rowLen, and the table's byteLen.
	if err := e.charge(len(entry) + 3*binary.MaxVarintLen64); err != nil {
		return err
	}
	t, row, err := e.rowByIndex(fi, ri)
	if err != nil {
		return err
	}
	e.buf = splice(e.buf, at, 0, entry)
	buf, m := spliceUvarint(e.buf, row.nColsOff, row.nColsLen, uint64(row.nCols+1)) //nolint:gosec // nCols >= 0
	e.buf = buf
	if err := e.patchRowLen(t, row, len(entry)+m-row.nColsLen); err != nil {
		return err
	}
	return e.reindex()
}

// removeCol splices column ci out of row ri and rewrites nCols, rowLen and
// byteLen. Ruling (the oracle's): the row survives losing its last column — a
// row is a keyed entity, and an empty column list encodes fine.
func (e *dynamicEngine) removeCol(fi, ri, ci int) error {
	t, row, err := e.rowByIndex(fi, ri)
	if err != nil {
		return err
	}
	c, err := e.colByIndex(row, ci)
	if err != nil {
		return err
	}
	removed := c.end - c.off
	e.buf = splice(e.buf, c.off, removed, nil)
	buf, m := spliceUvarint(e.buf, row.nColsOff, row.nColsLen, uint64(row.nCols-1)) //nolint:gosec // ci < nCols, so nCols >= 1
	e.buf = buf
	if err := e.patchRowLen(t, row, m-row.nColsLen-removed); err != nil {
		return err
	}
	return e.reindex()
}

// appendDynName appends a stored [nlen u8][name] pair. The caller has already
// bounded the name at OperateMaxNameLen.
func appendDynName(dst []byte, name string) []byte {
	dst = append(dst, byte(len(name))) //nolint:gosec // bounded by OperateMaxNameLen in resolve
	return append(dst, name...)
}

// --- del, count ------------------------------------------------------------

// del implements engine.del (design doc §3.1/§2.5): the record is marked
// deleted, and a dynamic field, row or column is removed outright rather than
// zeroed. Deleting something absent is a no-op.
func (e *dynamicEngine) del(r ref) error {
	if e.inner != nil {
		return e.inner.del(r)
	}
	switch r.kind {
	case refKindRecord:
		e.deleted = true
		return nil
	case refKindTable:
		// Ruling (the oracle's): in dynamic mode DEL of a field removes the
		// field whatever it holds — unlike schema mode, where a table field
		// is declared by the schema and can only be emptied.
		return e.removeField(r.field)
	case refKindRow:
		if !r.present {
			return nil
		}
		return e.removeRow(r.field, r.row)
	default: // refKindScalar: a field or a column
		if !r.present {
			return nil
		}
		if r.row < 0 {
			return e.removeField(r.field)
		}
		return e.removeCol(r.field, r.row, r.col)
	}
}

// count implements engine.count (design doc §3.3/§3.4): fields for the
// record, rows for a table, and presence as 1/0 for a row or a scalar.
func (e *dynamicEngine) count(r ref) (uint64, error) {
	if e.inner != nil {
		return e.inner.count(r)
	}
	switch r.kind {
	case refKindRecord:
		return uint64(len(e.fields)), nil //nolint:gosec // bounded by OperateMaxFields
	case refKindTable:
		t, err := e.table(r.field)
		if err != nil {
			return 0, err
		}
		return uint64(t.nRows), nil //nolint:gosec // nRows >= 0
	default:
		if r.present {
			return 1, nil
		}
		return 0, nil
	}
}

// --- eviction, TRIM and CONFIG ---------------------------------------------

// evictOne removes the victim the policy names (design doc §2.3): the
// smallest or largest key (the ends of the sorted table), or the smallest or
// largest value in the eviction column, ties going to the smallest key.
//
// Ruling (mirrors the oracle): a cap above 0 with PolicyNone still needs a
// deterministic victim — it evicts by MIN_KEY, as does any unknown policy.
func (e *dynamicEngine) evictOne(fi int, policy uint8, byColName []byte) error {
	t, err := e.table(fi)
	if err != nil {
		return err
	}
	if t.nRows == 0 {
		return nil
	}
	victim := 0
	switch policy {
	case wire.OperatePolicyMaxKey:
		victim = t.nRows - 1
	case wire.OperatePolicyMinCol, wire.OperatePolicyMaxCol:
		if victim, err = e.colVictim(t, policy == wire.OperatePolicyMaxCol, byColName); err != nil {
			return err
		}
	default:
		victim = 0
	}
	return e.removeRow(fi, victim)
}

// colVictim scans the table once for the row with the smallest (or largest)
// value in the eviction column; a row that does not carry the column reads as
// zero (design doc §2.4), and ties go to the first row in stored order, which
// is the smallest key.
func (e *dynamicEngine) colVictim(t dynTable, wantMax bool, byColName []byte) (int, error) {
	best := 0
	var bestCell wire.Cell
	off := t.rowsOff
	for i := 0; i < t.nRows; i++ {
		row, err := e.rowAt(off, t.end)
		if err != nil {
			return 0, err
		}
		c, err := e.evictCell(row, byColName)
		if err != nil {
			return 0, err
		}
		if i == 0 {
			bestCell = c
		} else if ord := compareDynCells(c, bestCell); (wantMax && ord > 0) || (!wantMax && ord < 0) {
			best, bestCell = i, c
		}
		off = row.end
	}
	return best, nil
}

// evictCell reads a row's eviction column, or the integer zero when the row
// does not carry it.
func (e *dynamicEngine) evictCell(row dynRow, byColName []byte) (wire.Cell, error) {
	off := row.colsOff
	for i := 0; i < row.nCols; i++ {
		c, err := e.colAt(off, row.end)
		if err != nil {
			return wire.Cell{}, err
		}
		switch bytes.Compare(e.buf[c.nameOff:c.nameOff+c.nameLen], byColName) {
		case 0:
			cell, _, derr := wire.DecodeCellData(c.typ, c.n, e.buf[c.valOff:c.end])
			if derr != nil {
				return wire.Cell{}, derr
			}
			return cell, nil
		case 1:
			return wire.Cell{Type: wire.OperateTypeU8}, nil // sorted by name: past it
		}
		off = c.end
	}
	return wire.Cell{Type: wire.OperateTypeU8}, nil
}

// compareDynCells orders two cells deterministically (the oracle's
// treeCompareCells). Within a domain the comparison is the natural one;
// across domains — reachable in dynamic mode, where two rows may store
// different types under one column name — the domain rank (int < float <
// bytes) orders them, so the victim stays a pure function of the bytes.
func compareDynCells(a, b wire.Cell) int {
	da, db := dynDomain(a.Type), dynDomain(b.Type)
	if da != db {
		if da < db {
			return -1
		}
		return 1
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
		return compareDynInts(a, b)
	default:
		return 0
	}
}

// compareDynInts orders two integer cells exactly, including a signed cell
// against an unsigned one (a negative value is below every unsigned value).
func compareDynInts(a, b wire.Cell) int {
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
		x, y := int64(a.U), int64(b.U) //nolint:gosec // reinterpret the stored signed bits
		switch {
		case x < y:
			return -1
		case x > y:
			return 1
		default:
			return 0
		}
	case au:
		return -compareDynMixed(b, a)
	default:
		return compareDynMixed(a, b)
	}
}

func compareDynMixed(signed, unsigned wire.Cell) int {
	s := int64(signed.U) //nolint:gosec // reinterpret the stored signed bits
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

// trim implements engine.trim (design doc §3.2): a one-off shrink to keep
// rows by the op's own policy, leaving the table's stored eviction config
// alone. It is a shrink, not a write, so an absent field is a no-op.
func (e *dynamicEngine) trim(r ref, keep uint32, policy uint8, byCol uint16, byColName string) error {
	_ = byCol // dynamic mode addresses the eviction column by name
	if e.inner != nil {
		return e.inner.trim(r, keep, policy, byCol, byColName)
	}
	if policy > wire.OperatePolicyMaxCol {
		return wire.ErrOperateSchema
	}
	if r.kind != refKindTable {
		if r.kind == refKindScalar && !r.present {
			return nil // an absent dynamic table field: nothing to shrink
		}
		return wire.ErrOperatePath
	}
	if policy == wire.OperatePolicyMinCol || policy == wire.OperatePolicyMaxCol {
		if byColName == "" {
			// Mirrors Schema.Validate's "ByCol must name a real column" rule
			// for a *_COL policy (oracle ruling).
			return wire.ErrOperatePath
		}
	}
	name := []byte(byColName)
	for {
		t, err := e.table(r.field)
		if err != nil {
			return err
		}
		if uint64(t.nRows) <= uint64(keep) { //nolint:gosec // nRows >= 0
			return nil
		}
		if err := e.evictOne(r.field, policy, name); err != nil {
			return err
		}
	}
}

// config implements engine.config (design doc §3.2): it writes the table's
// eviction triple into the stored header and evicts down to the new cap.
func (e *dynamicEngine) config(r ref, capN uint32, policy uint8, byColName string) error {
	if e.inner != nil {
		return e.inner.config(r, capN, policy, byColName)
	}
	// applyOps has already bounded byColName's length and rejected an unknown
	// policy byte, in that order (see its CONFIG branch); what is left is the
	// pair of rules the loop cannot state, both of them wire.ErrOperatePath.
	if (policy == wire.OperatePolicyMinCol || policy == wire.OperatePolicyMaxCol) && byColName == "" {
		return wire.ErrOperatePath
	}
	if r.kind != refKindTable {
		return wire.ErrOperatePath
	}
	t, err := e.table(r.field)
	if err != nil {
		return err
	}
	hdr := binary.AppendUvarint(e.enc[:0], uint64(capN))
	hdr = append(hdr, policy)
	hdr = appendDynName(hdr, byColName)
	e.enc = hdr
	oldLen := t.nRowsOff - t.capOff
	if err := e.charge(len(hdr) - oldLen); err != nil {
		return err
	}
	e.buf = splice(e.buf, t.capOff, oldLen, hdr)
	if err := e.patchTableLen(t, len(hdr)-oldLen); err != nil {
		return err
	}
	if err := e.reindex(); err != nil {
		return err
	}

	name := []byte(byColName)
	for capN > 0 {
		t, err := e.table(r.field)
		if err != nil {
			return err
		}
		if uint32(t.nRows) <= capN { //nolint:gosec // nRows bounded by OperateMaxRows
			break
		}
		if err := e.evictOne(r.field, policy, name); err != nil {
			return err
		}
	}
	return nil
}

// --- migrate (the freeze) --------------------------------------------------

// migrate implements engine.migrate for dynamic mode (design doc §2.9): the
// only migration a dynamic record accepts is a freeze into schema mode. The
// schema blob is decoded (and its layout resolved) first, so a malformed blob
// is reported as such whatever the from-version says — the oracle's order.
func (e *dynamicEngine) migrate(from int64, flags uint8, schemaBlob []byte) error {
	if e.inner != nil {
		return e.inner.migrate(from, flags, schemaBlob)
	}
	ent, err := e.cache.lookupCall(schemaBlob)
	if err != nil {
		return err
	}
	if from != wire.OperateMigrateFromDynamic {
		// An append-only evolution (§2.8) needs a schema-mode record to
		// extend; this one has no schema to extend from.
		return wire.ErrOperateMode
	}
	if !ent.schema.StoreNames {
		// A freeze matches dynamic fields and columns by name, so the target
		// schema has to carry them.
		return wire.ErrOperateSchema
	}
	return e.freeze(ent, flags&wire.OperateMigrateDropExtra != 0)
}

// freeze rewrites the record in schema mode: every schema field and column is
// filled from the dynamic value of the same name (domains must match, widths
// saturate), rows are packed fixed-width in the schema's key order and
// evicted down to the schema's cap, and a dynamic field or column the schema
// does not have is an error unless dropExtra is set.
func (e *dynamicEngine) freeze(ent *schemaEntry, dropExtra bool) error {
	s, layout := ent.schema, ent.layout
	cells := make([]wire.Cell, len(s.Fields))
	tables := make([][]byte, len(s.Fields))
	rowCounts := make([]int, len(s.Fields))
	matched := 0

	for i := range s.Fields {
		fd := &s.Fields[i]
		if fd.Name == "" {
			return wire.ErrOperateSchema
		}
		fi, found := e.findField(fd.Name)
		if found {
			matched++
		}

		if fd.Type != wire.OperateTypeTable {
			var src wire.Cell
			if found {
				f := &e.fields[fi]
				if f.typ == wire.OperateTypeTable {
					return wire.ErrOperateType
				}
				c, _, err := wire.DecodeCellData(f.typ, f.n, e.buf[f.valOff:f.end])
				if err != nil {
					return err
				}
				src = c
			}
			c, err := convertCell(src, found, fd.Type, fd.N)
			if err != nil {
				return err
			}
			cells[i] = c
			continue
		}

		packed, rows, err := e.freezeTable(fi, found, fd, layout.Tables[i], dropExtra)
		if err != nil {
			return err
		}
		cells[i] = wire.Cell{Type: wire.OperateTypeTable}
		tables[i], rowCounts[i] = packed, rows
	}

	if !dropExtra && matched != len(e.fields) {
		return wire.ErrOperateSchema
	}
	return e.installFrozen(ent, cells, tables, rowCounts)
}

// freezeTable packs one dynamic table into the schema's fixed-width row
// layout, sorted by the schema's key comparison and evicted down to its cap.
func (e *dynamicEngine) freezeTable(fi int, found bool, fd *wire.FieldDef, tl *wire.TableLayout, dropExtra bool) ([]byte, int, error) {
	td := fd.Table
	if !found {
		return nil, 0, nil
	}
	f := &e.fields[fi]
	if f.typ != wire.OperateTypeTable {
		return nil, 0, wire.ErrOperateType
	}
	t, err := e.table(fi)
	if err != nil {
		return nil, 0, err
	}
	// rows x the NEW row width can reach 2^20 x 4096, which wraps a 32-bit
	// int: the projection is charged in uint64 before anything is allocated.
	if tableBytes(t.nRows, tl.RowWidth) > uint64(maxOperateRecordBytes) { //nolint:gosec // maxOperateRecordBytes > 0
		return nil, 0, wire.ErrOperateCap
	}

	packed := make([]byte, 0, t.nRows*tl.RowWidth)
	off := t.rowsOff
	for i := 0; i < t.nRows; i++ {
		row, rerr := e.rowAt(off, t.end)
		if rerr != nil {
			return nil, 0, rerr
		}
		// Ruling (the oracle's): a schema-mode row key is exactly the table's
		// key width, so a dynamic key of any other length cannot be packed.
		if row.keyLen != tl.KeyWidth {
			return nil, 0, wire.ErrOperateSchema
		}
		packed = append(packed, e.buf[row.keyOff:row.keyOff+row.keyLen]...)
		hits := 0
		for c := range td.Cols {
			cd := &td.Cols[c]
			if cd.Name == "" {
				return nil, 0, wire.ErrOperateSchema
			}
			col, _, colFound, cerr := e.findCol(row, cd.Name)
			if cerr != nil {
				return nil, 0, cerr
			}
			var src wire.Cell
			if colFound {
				hits++
				cell, _, derr := wire.DecodeCellData(col.typ, col.n, e.buf[col.valOff:col.end])
				if derr != nil {
					return nil, 0, derr
				}
				src = cell
			}
			cell, cerr2 := convertCell(src, colFound, cd.Type, cd.N)
			if cerr2 != nil {
				return nil, 0, cerr2
			}
			packed = wire.AppendCellData(packed, cell)
		}
		if hits != row.nCols && !dropExtra {
			return nil, 0, wire.ErrOperateSchema
		}
		off = row.end
	}

	rows := t.nRows
	packed = sortPackedRows(packed, rows, tl, td.KeyType)
	for td.Cap > 0 && uint32(rows) > td.Cap { //nolint:gosec // rows bounded by OperateMaxRows
		packed, rows = evictPackedRow(packed, rows, tl, td)
	}
	return packed, rows, nil
}

// sortPackedRows reorders fixed-width packed rows into the schema's key
// order: numeric keys compare as little-endian integers, FIXED keys
// bytewise. The rows arrive in the dynamic record's bytewise key order, which
// already is the schema's order for a FIXED key, so only a numeric key can
// need the reorder — and even then only when the two orders differ.
func sortPackedRows(packed []byte, rows int, tl *wire.TableLayout, keyType uint8) []byte {
	if rows < 2 || keyType == wire.OperateTypeFixed {
		return packed
	}
	w := tl.RowWidth
	key := func(i int) []byte { return packed[i*w : i*w+tl.KeyWidth] }
	idx := make([]int, rows)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		return compareRowKey(key(idx[a]), key(idx[b]), keyType) < 0
	})
	moved := false
	for i := range idx {
		if idx[i] != i {
			moved = true
			break
		}
	}
	if !moved {
		return packed
	}
	out := make([]byte, rows*w)
	for i, src := range idx {
		copy(out[i*w:(i+1)*w], packed[src*w:(src+1)*w])
	}
	return out
}

// evictPackedRow removes the victim the schema's policy names from packed
// rows, returning the shortened buffer and row count.
func evictPackedRow(packed []byte, rows int, tl *wire.TableLayout, td *wire.TableDef) ([]byte, int) {
	if rows == 0 {
		return packed, 0
	}
	w := tl.RowWidth
	victim := 0
	switch td.Policy {
	case wire.OperatePolicyMaxKey:
		victim = rows - 1
	case wire.OperatePolicyMinCol, wire.OperatePolicyMaxCol:
		byCol := int(td.ByCol)
		if byCol >= 0 && byCol < len(td.Cols) {
			typ, n := td.Cols[byCol].Type, td.Cols[byCol].N
			width := wire.CellWidth(typ, n)
			at := tl.ColOff[byCol]
			wantMax := td.Policy == wire.OperatePolicyMaxCol
			bestAt := at
			for i := 1; i < rows; i++ {
				cur := i*w + at
				ord := compareCellBytes(packed[cur:cur+width], packed[bestAt:bestAt+width], typ)
				if (wantMax && ord > 0) || (!wantMax && ord < 0) {
					victim, bestAt = i, cur
				}
			}
		}
	default:
		victim = 0
	}
	copy(packed[victim*w:], packed[(victim+1)*w:rows*w])
	return packed[:(rows-1)*w], rows - 1
}

// convertCell re-types src as (typ, n) for a freeze (design doc §2.9): the
// domains must match and widths saturate, which is exactly a SET of the
// source value into a zero cell of the target type.
func convertCell(src wire.Cell, present bool, typ, n uint8) (wire.Cell, error) {
	dst := wire.ZeroCell(typ, n)
	if !present {
		return dst, nil
	}
	if dynDomain(src.Type) != dynDomain(typ) || dynDomain(typ) < 0 {
		return wire.Cell{}, wire.ErrOperateType
	}
	var op wire.Operand
	switch dynDomain(typ) {
	case 0:
		op.A = intOperand(src)
	case 1:
		op.A = int64(math.Float64bits(src.F)) //nolint:gosec // a float operand travels as its bit pattern
	default:
		op.Bytes = src.B
	}
	return wire.ApplyScalar(wire.OperateOpSET, dst, false, op)
}

// intOperand renders an integer cell as the int64 operand a SET carries,
// saturating an unsigned value that does not fit.
func intOperand(c wire.Cell) int64 {
	if wire.TypeIsUnsigned(c.Type) && c.U > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(c.U) //nolint:gosec // reinterpret the stored bits
}

// installFrozen writes the schema-mode record the freeze produced and hands
// the engine over to a schema engine on it: every later op in this call, and
// every return spec, must see the frozen record.
func (e *dynamicEngine) installFrozen(ent *schemaEntry, cells []wire.Cell, tables [][]byte, rowCounts []int) error {
	s, layout := ent.schema, ent.layout
	size := uint64(1) + uint64(len(ent.blob)) + uint64(layout.FixedLen)
	for _, i := range layout.VarTail {
		if s.Fields[i].Type == wire.OperateTypeTable {
			size += uint64(uvarintLen(uint64(rowCounts[i]))) + uint64(len(tables[i])) //nolint:gosec // rowCounts >= 0
			continue
		}
		size += uint64(len(wire.AppendCellData(nil, cells[i])))
	}
	if size > uint64(maxOperateRecordBytes) { //nolint:gosec // maxOperateRecordBytes > 0
		return wire.ErrOperateCap
	}

	out := make([]byte, 0, size+64)
	out = append(out, wire.OperateModeSchema)
	out = append(out, ent.blob...)
	for i := range s.Fields {
		if layout.FixedOff[i] < 0 {
			continue
		}
		out = wire.AppendCellData(out, cells[i])
	}
	for _, i := range layout.VarTail {
		if s.Fields[i].Type == wire.OperateTypeTable {
			out = binary.AppendUvarint(out, uint64(rowCounts[i])) //nolint:gosec // rowCounts >= 0
			out = append(out, tables[i]...)
			continue
		}
		out = wire.AppendCellData(out, cells[i])
	}

	inner, err := newSchemaEngine(out, e.cache)
	if err != nil {
		return err
	}
	inner.existed = e.existed
	inner.deleted = e.deleted
	e.inner = inner
	e.buf = out
	e.fields = e.fields[:0]
	return nil
}

// --- value, bytes, empty ---------------------------------------------------

// value implements engine.value (design doc §3.4): the whole record, the
// table's stored value behind the mode byte, a row's named columns as tagged
// cells, or one tagged scalar.
func (e *dynamicEngine) value(r ref) ([]byte, error) {
	if e.inner != nil {
		return e.inner.value(r)
	}
	switch r.kind {
	case refKindRecord:
		return append([]byte(nil), e.buf...), nil
	case refKindTable:
		// Ruling (the oracle's): no names ride along for the table itself —
		// the mode byte says which of the two table layouts follows, and the
		// bytes are exactly the ones the record encoder emits for the field.
		if r.field < 0 || r.field >= len(e.fields) {
			return nil, wire.ErrOperatePath
		}
		f := &e.fields[r.field]
		out := make([]byte, 0, 1+f.end-f.valOff)
		out = append(out, wire.OperateModeDynamic)
		return append(out, e.buf[f.valOff:f.end]...), nil
	case refKindRow:
		_, row, err := e.rowByIndex(r.field, r.row)
		if err != nil {
			return nil, err
		}
		// [mode][nCols]{[nlen][name] tagged}: a stored column entry is
		// already [nlen][name][type][n?][data], so the row's own bytes are
		// the answer.
		out := make([]byte, 0, 1+row.end-row.nColsOff)
		out = append(out, wire.OperateModeDynamic)
		return append(out, e.buf[row.nColsOff:row.end]...), nil
	default:
		end, err := e.taggedEnd(r.off)
		if err != nil {
			return nil, err
		}
		return append([]byte(nil), e.buf[r.off:end]...), nil
	}
}

// bytes implements engine.bytes: the record's current encoded form, which is
// the engine's working buffer itself.
func (e *dynamicEngine) bytes() []byte {
	if e.inner != nil {
		return e.inner.bytes()
	}
	return e.buf
}

// empty implements engine.empty: a dynamic record that lost its last field is
// deleted (design doc §2.5) — there is no schema declaring an empty shell, so
// an empty dynamic record has no meaning. A frozen record answers as the
// schema record it now is.
func (e *dynamicEngine) empty() bool {
	if e.inner != nil {
		return e.inner.empty()
	}
	return e.deleted || len(e.fields) == 0
}
