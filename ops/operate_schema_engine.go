// SPDX-License-Identifier: Apache-2.0

package ops

// The schema-mode byte engine (design doc §2.6): it navigates and patches a
// stored operate record's bytes directly, never building a tree. A
// fixed-width field's offset is a schema constant, a row is a binary search
// over fixed-width rows, and a column is rowStart + colOffset; only the
// variable-length tail costs a short walk, which reindexTail does once per
// call and again after any splice that moves it.
//
// The tree oracle (operate_oracle_test.go) is the definition of what this
// file must do; TestSchemaEngineMatchesOracle compares them byte for byte.
//
// Stored bytes are untrusted — a plain `put` can store anything under the
// key — so newSchemaEngine validates the whole frame with a bounds-checked
// walk before any offset here is trusted, and every method below is written
// to be unable to panic on input that passed that walk.

import (
	"bytes"
	"encoding/binary"
	"math"
	"sync"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- schema cache ----------------------------------------------------------

// schemaEntry is one decoded schema: the schema itself, its byte layout, its
// canonical blob (what a record stores in front of its values), and the
// reverse of layout.VarTail — field position to tail slot, -1 for a
// fixed-width field. Every member is immutable once published, so entries are
// shared across calls and goroutines without copying.
type schemaEntry struct {
	schema   *wire.Schema
	layout   *wire.Layout
	blob     []byte
	tailSlot []int
}

// schemaCache is the design doc §2.8 offset cache: a bounded map from schema
// blob bytes to the decoded schema and its layout, so a call whose record
// carries a known schema resolves every path by arithmetic with no parsing.
// It is a pure function of the bytes — it can never affect what a call
// stores, only how fast it gets there.
//
// Eviction is deliberately crude: at max entries the whole map is dropped and
// rebuilt. A schema blob is at most maxSchemaBytes, the working set of one
// deployment is a handful of shapes, and "clear when full" needs no ordering
// state (and so cannot leak Go map iteration order into anything).
type schemaCache struct {
	mu   sync.Mutex
	max  int
	ents map[string]*schemaEntry
}

// operateSchemas is the process-wide schema cache.
var operateSchemas = newSchemaCache(256)

func newSchemaCache(max int) *schemaCache {
	if max < 1 {
		max = 1
	}
	return &schemaCache{max: max, ents: make(map[string]*schemaEntry, max)}
}

// get decodes a call's schema blob, returning the schema and its layout. It is
// the contract's entry point and, like every call-argument path, requires the
// blob to be exactly one schema.
func (c *schemaCache) get(b []byte) (*wire.Schema, *wire.Layout, error) {
	ent, err := c.lookupCall(b)
	if err != nil {
		return nil, nil, err
	}
	return ent.schema, ent.layout, nil
}

// lookupCall is the call-argument form of lookup: the blob a call carries must
// be exactly one schema, with nothing after it (the oracle's treeOpen rejects a
// trailing byte the same way). A stored record's blob, by contrast, is a prefix
// of the record and uses lookup directly.
func (c *schemaCache) lookupCall(b []byte) (*schemaEntry, error) {
	ent, n, err := c.lookup(b)
	if err != nil {
		return nil, err
	}
	if n != len(b) {
		return nil, wire.ErrOperateSchema
	}
	return ent, nil
}

// lookup returns the cached entry for the schema blob at the front of b and
// the blob's encoded length.
//
// The cache is keyed by the exact blob bytes, so a hit needs the blob's
// length before the map lookup: schemaBlobLen walks the framing without
// decoding. A wrong length from that walk cannot produce a wrong answer —
// b[:n] only matches a key if it is byte-identical to an already-decoded,
// already-validated blob, which fixes n — and on a miss DecodeSchema is
// authoritative for both the schema and the length.
func (c *schemaCache) lookup(b []byte) (*schemaEntry, int, error) {
	if n, ok := schemaBlobLen(b); ok {
		c.mu.Lock()
		ent := c.ents[string(b[:n])] // no allocation: Go elides the key copy
		c.mu.Unlock()
		if ent != nil {
			return ent, n, nil
		}
	}

	s, n, err := wire.DecodeSchema(b)
	if err != nil {
		return nil, 0, err
	}
	layout, err := s.Layout()
	if err != nil {
		return nil, 0, err
	}
	ent := &schemaEntry{schema: s, layout: layout, blob: s.Encode(), tailSlot: tailSlots(s, layout)}

	c.mu.Lock()
	if len(c.ents) >= c.max {
		c.ents = make(map[string]*schemaEntry, c.max)
	}
	c.ents[string(ent.blob)] = ent
	c.mu.Unlock()
	return ent, n, nil
}

// tailSlots inverts layout.VarTail: field position to its index in the
// variable-length tail, or -1 for a fixed-width field.
func tailSlots(s *wire.Schema, l *wire.Layout) []int {
	slots := make([]int, len(s.Fields))
	for i := range slots {
		slots[i] = -1
	}
	for slot, field := range l.VarTail {
		slots[field] = slot
	}
	return slots
}

// schemaBlobLen returns the encoded length of the schema blob at the front of
// b, mirroring Encode's framing (design doc §2.2) without decoding anything.
// It exists only so the cache can be keyed by the blob's exact bytes without
// paying for a decode on every call; ok == false means "cannot tell", which
// is never an error by itself — the caller falls through to DecodeSchema,
// which is the authority on both validity and length.
func schemaBlobLen(b []byte) (int, bool) {
	off := 3 // version u16 + flags u8
	if len(b) < off+1 {
		return 0, false
	}
	// byteAt consumes one byte, and uvarintAt one uvarint, each reporting
	// failure rather than reading past the end: b is untrusted (it is the
	// front of a stored record), so every step is bounds-checked even though
	// a wrong answer here would only ever cost a cache miss.
	byteAt := func() (uint8, bool) {
		if len(b)-off < 1 {
			return 0, false
		}
		v := b[off]
		off++
		return v, true
	}
	uvarintAt := func() (uint64, bool) {
		if off > len(b) {
			return 0, false
		}
		v, m, ok := scanUvarint(b[off:])
		if !ok {
			return 0, false
		}
		off += m
		return v, true
	}

	nFields, ok := uvarintAt()
	if !ok || nFields > wire.OperateMaxFields {
		return 0, false
	}
	for i := uint64(0); i < nFields; i++ {
		typ, ok := byteAt()
		if !ok {
			return 0, false
		}
		if typ == wire.OperateTypeFixed {
			if _, ok := byteAt(); !ok {
				return 0, false
			}
		}
		if typ != wire.OperateTypeTable {
			continue
		}
		keyType, ok := byteAt()
		if !ok {
			return 0, false
		}
		if keyType == wire.OperateTypeFixed {
			if _, ok := byteAt(); !ok {
				return 0, false
			}
		}
		nCols, ok := uvarintAt()
		if !ok || nCols > wire.OperateMaxCols {
			return 0, false
		}
		for j := uint64(0); j < nCols; j++ {
			ctyp, ok := byteAt()
			if !ok {
				return 0, false
			}
			if ctyp == wire.OperateTypeFixed {
				if _, ok := byteAt(); !ok {
					return 0, false
				}
			}
		}
		if _, ok := uvarintAt(); !ok { // cap
			return 0, false
		}
		if _, ok := byteAt(); !ok { // policy
			return 0, false
		}
		if _, ok := uvarintAt(); !ok { // byCol
			return 0, false
		}
	}
	nameBytes, ok := uvarintAt()
	if !ok || nameBytes > wire.OperateMaxNameBytes {
		return 0, false
	}
	if !wire.CountFitsIn(int(nameBytes), len(b)-off, 1) { //nolint:gosec // bounded above
		return 0, false
	}
	return off + int(nameBytes), true //nolint:gosec // bounded above
}

// scanUvarint reads one uvarint, reporting failure rather than an error: it
// is only ever used by schemaBlobLen, whose caller treats "cannot tell" as a
// cache miss.
func scanUvarint(b []byte) (uint64, int, bool) {
	v, m := binary.Uvarint(b)
	if m <= 0 {
		return 0, 0, false
	}
	return v, m, true
}

// --- construction and structural validation --------------------------------

// tailEntry is one variable-length field's position in the stored tail: the
// offset of its value area and, for a table, its row count (-1 otherwise).
type tailEntry struct {
	off  int
	rows int
}

// inlineTail is the number of variable-length fields whose tail index rides
// inside the engine struct. Schemas in practice have one or two (the session
// shape has exactly one table); a wider one falls back to a heap slice.
const inlineTail = 4

// schemaEngine implements engine over a schema-mode record's stored bytes.
// buf is the engine's own copy, which it patches in place and splices as
// needed; every other member is derived from it.
type schemaEngine struct {
	buf     []byte
	cache   *schemaCache
	ent     *schemaEntry
	schema  *wire.Schema
	layout  *wire.Layout
	hdrLen  int // 1 (mode byte) + len(schema blob)
	tail    []tailEntry
	existed bool
	deleted bool

	tailArr [inlineTail]tailEntry
	// scratch encodes one cell's data without allocating. 256 bytes covers
	// every fixed-width type and the widest FIXED(n); BYTES is spliced from
	// its operand directly and never uses it.
	scratch [256]byte
}

// newSchemaEngine builds an engine over buf, which becomes the engine's
// private buffer (it is patched in place). buf is untrusted: the whole frame
// is validated here — mode byte, schema blob, the fixed-width area, and every
// tail field's length walked to exactly the end of the buffer — and anything
// that does not fit is wire.ErrOperateRecord.
func newSchemaEngine(buf []byte, cache *schemaCache) (*schemaEngine, error) {
	e := &schemaEngine{}
	if err := e.reset(buf, cache); err != nil {
		return nil, err
	}
	return e, nil
}

// reset re-points an engine at buf, so a pooled engine can be reused without
// re-allocating its tail index. It performs the same validation as
// newSchemaEngine.
func (e *schemaEngine) reset(buf []byte, cache *schemaCache) error {
	e.clear()
	e.cache = cache
	if len(buf) < 1 || buf[0] != wire.OperateModeSchema {
		return wire.ErrOperateRecord
	}
	ent, n, err := cache.lookup(buf[1:])
	if err != nil {
		// A stored record whose own schema will not decode is a malformed
		// record, not a malformed call.
		return wire.ErrOperateRecord
	}
	e.buf, e.ent, e.schema, e.layout = buf, ent, ent.schema, ent.layout
	e.hdrLen = 1 + n
	if len(e.layout.VarTail) <= inlineTail {
		e.tail = e.tailArr[:0]
	} else {
		e.tail = make([]tailEntry, 0, len(e.layout.VarTail))
	}
	return e.reindexTail()
}

// clear drops the engine's references so a pooled engine does not pin a
// record buffer between calls.
func (e *schemaEngine) clear() {
	e.buf, e.cache, e.ent, e.schema, e.layout = nil, nil, nil, nil, nil
	e.hdrLen, e.tail, e.existed, e.deleted = 0, nil, false, false
}

// reindexTail walks the variable-length tail, recording each field's offset
// and each table's row count, and checks that the walk lands exactly on the
// end of the buffer. It is both the structural validator (on an untrusted
// buffer) and the index rebuild after any splice that moved the tail; it is
// O(number of variable-length fields), since a table is skipped by
// arithmetic rather than read.
func (e *schemaEngine) reindexTail() error {
	off := e.hdrLen + e.layout.FixedLen
	if off < 0 || off > len(e.buf) {
		return wire.ErrOperateRecord
	}
	e.tail = e.tail[:0]
	for _, i := range e.layout.VarTail {
		ent := tailEntry{off: off, rows: -1}
		switch e.schema.Fields[i].Type {
		case wire.OperateTypeTable:
			n, m, err := readUvarint(e.buf[off:])
			if err != nil {
				return err
			}
			if n > wire.OperateMaxRows {
				return wire.ErrOperateRecord
			}
			tl := e.layout.Tables[i]
			if !wire.CountFitsIn(int(n), len(e.buf)-off-m, tl.RowWidth) { //nolint:gosec // bounded above
				return wire.ErrOperateRecord
			}
			ent.rows = int(n) //nolint:gosec // bounded by OperateMaxRows above
			off += m + ent.rows*tl.RowWidth
		case wire.OperateTypeBytes:
			ln, m, err := readUvarint(e.buf[off:])
			if err != nil {
				return err
			}
			if ln > wire.OperateMaxBytesLen {
				return wire.ErrOperateRecord
			}
			if !wire.CountFitsIn(int(ln), len(e.buf)-off-m, 1) { //nolint:gosec // bounded above
				return wire.ErrOperateRecord
			}
			off += m + int(ln) //nolint:gosec // bounded above
		default: // UVARINT, IVARINT
			_, m, err := readUvarint(e.buf[off:])
			if err != nil {
				return err
			}
			off += m
		}
		e.tail = append(e.tail, ent)
	}
	if off != len(e.buf) {
		return wire.ErrOperateRecord
	}
	return nil
}

// readUvarint reads one canonically encoded uvarint. A non-minimal encoding
// is rejected: the record codec never produces one, and accepting it would
// break the byte-for-byte identity with the oracle's output.
func readUvarint(b []byte) (uint64, int, error) {
	v, m := binary.Uvarint(b)
	if m <= 0 {
		return 0, 0, wire.ErrOperateRecord
	}
	if m != uvarintLen(v) {
		return 0, 0, wire.ErrOperateRecord
	}
	return v, m, nil
}

// uvarintLen is the number of bytes binary.AppendUvarint uses for v.
func uvarintLen(v uint64) int {
	n := 1
	for v >= 0x80 {
		v >>= 7
		n++
	}
	return n
}

// newSchemaRecord returns the stored bytes of a fresh, empty record for the
// schema blob: [mode][blob][zeroed fixed-width fields][each tail field's
// zero]. Every tail field's zero is a single 0x00 byte — a BYTES length of
// zero, a varint zero, or a table with zero rows.
func newSchemaRecord(schemaBlob []byte, cache *schemaCache) ([]byte, error) {
	if len(schemaBlob) == 0 {
		return nil, wire.ErrOperateSchema
	}
	ent, err := cache.lookupCall(schemaBlob)
	if err != nil {
		return nil, err
	}
	size := 1 + len(ent.blob) + ent.layout.FixedLen + len(ent.layout.VarTail)
	if size > maxOperateRecordBytes {
		return nil, wire.ErrOperateCap
	}
	buf := make([]byte, size, size+64)
	buf[0] = wire.OperateModeSchema
	copy(buf[1:], ent.blob)
	// The fixed area and every tail field's zero byte are already zero.
	return buf, nil
}

// --- resolve ---------------------------------------------------------------

// resolve implements engine.resolve for schema mode: a field is a schema
// position (or a name, when the schema stores names), a row is a binary
// search, and a column is a constant offset within that row. Schema-mode
// fields and columns always exist (design doc §2.5), so create only ever
// vivifies a row.
func (e *schemaEngine) resolve(p wire.OperatePath, create bool, opcode, typ, n uint8) (ref, error) {
	_ = n // schema mode takes widths from the schema, never from the op
	if opcode == wire.OperateOpCONFIG {
		// A schema-mode table's eviction triple comes from its schema and
		// cannot be set per record (design doc §3.2), so CONFIG is not an op
		// this record has at all — reported here, before the path is even
		// looked at, because that is where the oracle reports it (its config
		// checks the record's mode first). config() answers the same thing
		// for a caller that gets past this.
		return ref{}, wire.ErrOperateOpcode
	}
	// The opcode otherwise only matters where a stored type can be replaced,
	// which is dynamic mode's SET (design doc §2.9). A schema-mode field's
	// type is the schema's, whatever the op says, so checkType below is the
	// whole rule here.
	if p.Kind > wire.OperatePathCol {
		return ref{}, wire.ErrOperatePath
	}
	if p.Kind == wire.OperatePathRecord {
		// Presence of the record is whether it existed before this call
		// (oracle ruling): CHECK((), ABSENT) means "this call created it".
		return ref{kind: refKindRecord, field: -1, row: -1, col: -1, present: e.existed}, nil
	}

	pos, err := e.fieldPos(p.Field)
	if err != nil {
		return ref{}, err
	}
	fd := &e.schema.Fields[pos]

	if p.Kind == wire.OperatePathField {
		if fd.Type == wire.OperateTypeTable {
			return ref{kind: refKindTable, field: pos, row: -1, col: -1, present: true}, nil
		}
		if err := e.checkType(create, typ, fd.Type); err != nil {
			return ref{}, err
		}
		return ref{kind: refKindScalar, field: pos, row: -1, col: -1,
			off: e.fieldOff(pos), typ: fd.Type, n: fd.N, present: true}, nil
	}

	if fd.Type != wire.OperateTypeTable {
		return ref{}, wire.ErrOperatePath // a row path into a non-table field
	}
	td := fd.Table
	tl := e.layout.Tables[pos]
	if len(p.Key) != tl.KeyWidth {
		return ref{}, wire.ErrOperatePath
	}

	col := -1
	if p.Kind == wire.OperatePathCol {
		col, err = e.colPos(td, p.Col)
		if err != nil {
			return ref{}, err
		}
		if err := e.checkType(create, typ, td.Cols[col].Type); err != nil {
			return ref{}, err
		}
	}

	idx, found := e.findRow(pos, p.Key)
	if !found && create {
		// The oracle vivifies the row for any create=true resolve, including
		// a row-kind path (whose op applyOps then rejects); mirrored here so
		// the two agree on which error comes out.
		if idx, err = e.insertRow(pos, idx, p.Key); err != nil {
			return ref{}, err
		}
	}
	if p.Kind == wire.OperatePathRow {
		return ref{kind: refKindRow, field: pos, row: idx, col: -1, present: found}, nil
	}
	r := ref{kind: refKindScalar, field: pos, row: idx, col: col,
		typ: td.Cols[col].Type, n: td.Cols[col].N, present: found, off: -1}
	if found || create {
		r.off = e.rowStart(pos, idx) + tl.ColOff[col]
	}
	return r, nil
}

// fieldPos resolves a path's field segment to a schema position. A name
// addresses a schema record only when the schema stores names (design doc
// §2.4).
func (e *schemaEngine) fieldPos(seg wire.OperateSeg) (int, error) {
	if seg.ByName {
		if !e.schema.StoreNames {
			return 0, wire.ErrOperatePath
		}
		pos, ok := e.schema.FieldPos(seg.Name)
		if !ok {
			return 0, wire.ErrOperatePath
		}
		return pos, nil
	}
	if seg.Pos >= uint32(len(e.schema.Fields)) { //nolint:gosec // len is bounded by OperateMaxFields
		return 0, wire.ErrOperatePath
	}
	return int(seg.Pos), nil
}

// colPos resolves a path's column segment to a column position.
func (e *schemaEngine) colPos(td *wire.TableDef, seg wire.OperateSeg) (int, error) {
	if seg.ByName {
		if !e.schema.StoreNames || seg.Name == "" {
			return 0, wire.ErrOperatePath
		}
		for j := range td.Cols {
			if td.Cols[j].Name == seg.Name {
				return j, nil
			}
		}
		return 0, wire.ErrOperatePath
	}
	if seg.Pos >= uint32(len(td.Cols)) { //nolint:gosec // len is bounded by OperateMaxCols
		return 0, wire.ErrOperatePath
	}
	return int(seg.Pos), nil
}

// checkType enforces design doc §2.4's schema-mode rule: an op's type byte is
// either 0xFF ("from the schema") or exactly the schema's type.
//
// Ruling: applyOps passes typ=0 for every op whose type byte is meaningless
// (IF, CHECK, DEL, TRIM, and every return spec) and the op's own type byte
// only for the scalar ops, which are exactly the ops that pass create=true —
// so create is the signal that typ is a real type byte to check. The one
// op this mis-reads is CONFIG, which also resolves with create=true (typ =
// TABLE): aimed at a scalar field it fails here with ErrOperateType where the
// oracle reports ErrOperateOpcode. Both reject the call with the record
// unchanged, and CONFIG is never valid in schema mode at all.
func (e *schemaEngine) checkType(create bool, typ, want uint8) error {
	if !create || typ == wire.OperateTypeFromSchema || typ == want {
		return nil
	}
	return wire.ErrOperateType
}

// fieldOff is the byte offset of field pos's value: a schema constant for a
// fixed-width field, the walked tail offset for a variable-length one.
func (e *schemaEngine) fieldOff(pos int) int {
	if off := e.layout.FixedOff[pos]; off >= 0 {
		return e.hdrLen + off
	}
	return e.tail[e.ent.tailSlot[pos]].off
}

// rows is the row count of the table field at pos.
func (e *schemaEngine) rows(pos int) int { return e.tail[e.ent.tailSlot[pos]].rows }

// rowsOff is the offset of the table field's row count uvarint.
func (e *schemaEngine) rowsOff(pos int) int { return e.tail[e.ent.tailSlot[pos]].off }

// rowStart is the offset of row idx of the table field at pos. idx*RowWidth
// cannot overflow: reindexTail admitted the table only after CountFitsIn
// proved rows*RowWidth fits the buffer, and idx <= rows. Every other
// count-times-width product in this file (reindexTail, removeRows, colVictim,
// value) is bounded the same way, by the buffer that already holds the rows;
// rebuiltSize is the one that projects a *new* width and so does its
// arithmetic in uint64.
func (e *schemaEngine) rowStart(pos, idx int) int {
	ent := &e.tail[e.ent.tailSlot[pos]]
	return ent.off + uvarintLen(uint64(ent.rows)) + idx*e.layout.Tables[pos].RowWidth //nolint:gosec // rows >= 0
}

// findRow binary-searches the table field at pos for key, returning the row
// index and whether it matched; when it did not, the index is where the row
// would be inserted to keep the table sorted. Numeric keys compare as
// little-endian integers and FIXED keys bytewise (design doc §2.3).
func (e *schemaEngine) findRow(pos int, key []byte) (int, bool) {
	tl := e.layout.Tables[pos]
	keyType := e.schema.Fields[pos].Table.KeyType
	start := e.rowStart(pos, 0)
	lo, hi := 0, e.rows(pos)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		at := start + mid*tl.RowWidth
		if compareRowKey(e.buf[at:at+tl.KeyWidth], key, keyType) < 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < e.rows(pos) {
		at := start + lo*tl.RowWidth
		if compareRowKey(e.buf[at:at+tl.KeyWidth], key, keyType) == 0 {
			return lo, true
		}
	}
	return lo, false
}

// compareRowKey orders two row keys of the same width: bytewise for a FIXED
// key, as little-endian unsigned integers for the numeric key types.
func compareRowKey(a, b []byte, keyType uint8) int {
	if keyType == wire.OperateTypeFixed {
		return bytes.Compare(a, b)
	}
	av, bv := leUint(a), leUint(b)
	switch {
	case av < bv:
		return -1
	case av > bv:
		return 1
	default:
		return 0
	}
}

// leUint reads up to 8 little-endian bytes as an unsigned integer.
func leUint(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// --- in-place patching -----------------------------------------------------

// get implements engine.get: the scalar at r, or its type's zero when r
// addresses a column of a row that does not exist (design doc §2.4).
func (e *schemaEngine) get(r ref) (wire.Cell, error) {
	if r.kind != refKindScalar {
		return wire.Cell{}, wire.ErrOperatePath
	}
	if r.off < 0 {
		return wire.ZeroCell(r.typ, r.n), nil
	}
	c, _, err := wire.DecodeCellData(r.typ, r.n, e.buf[r.off:])
	if err != nil {
		return wire.Cell{}, err
	}
	return c, nil
}

// set implements engine.set: a fixed-width cell is overwritten in place (the
// width cannot change, the schema owns it), and a BYTES or varint field is
// spliced when its encoded width does.
func (e *schemaEngine) set(r ref, c wire.Cell) error {
	if r.kind != refKindScalar || r.off < 0 {
		return wire.ErrOperatePath
	}
	if w := wire.CellWidth(r.typ, r.n); w >= 0 {
		if r.typ == wire.OperateTypeFixed && len(c.B) != int(r.n) {
			return wire.ErrOperateType
		}
		copy(e.buf[r.off:r.off+w], e.encodeCell(c))
		return nil
	}
	if r.typ == wire.OperateTypeBytes {
		return e.setBytes(r.off, c.B)
	}
	// UVARINT / IVARINT: at most 10 bytes, encoded through the same
	// canonical writer the record codec uses.
	_, m, err := readUvarint(e.buf[r.off:])
	if err != nil {
		return err
	}
	return e.spliceTail(r.off, m, e.encodeCell(c))
}

// encodeCell renders one cell's data into the engine's scratch space. It
// never allocates for a fixed-width type, a FIXED(n), or a varint — the only
// callers — because scratch is wider than any of them.
func (e *schemaEngine) encodeCell(c wire.Cell) []byte {
	return wire.AppendCellData(e.scratch[:0], c)
}

// setBytes replaces the BYTES field stored at off with b, splicing the length
// prefix and the payload separately so no scratch buffer has to hold both.
func (e *schemaEngine) setBytes(off int, b []byte) error {
	ln, m, err := readUvarint(e.buf[off:])
	if err != nil {
		return err
	}
	if len(b) > wire.OperateMaxBytesLen {
		return wire.ErrOperateCap
	}
	delta := (uvarintLen(uint64(len(b))) + len(b)) - (m + int(ln)) //nolint:gosec // ln bounded by reindexTail
	if err := e.charge(delta); err != nil {
		return err
	}
	buf, m2 := spliceUvarint(e.buf, off, m, uint64(len(b)))
	e.buf = splice(buf, off+m2, int(ln), b) //nolint:gosec // ln bounded by reindexTail
	return e.reindexTail()
}

// spliceTail replaces buf[off:off+oldLen] with repl and re-walks the tail.
func (e *schemaEngine) spliceTail(off, oldLen int, repl []byte) error {
	if err := e.charge(len(repl) - oldLen); err != nil {
		return err
	}
	e.buf = splice(e.buf, off, oldLen, repl)
	return e.reindexTail()
}

// charge rejects a growth that would take the record past the §2.7
// record-size cap before the splice allocates for it. The oracle checks the
// same bound once, on the finished record; charging every splice is stricter
// only for an op list that grows past the cap and shrinks back under it
// within one call.
func (e *schemaEngine) charge(delta int) error {
	if delta > 0 && len(e.buf)+delta > maxOperateRecordBytes {
		return wire.ErrOperateCap
	}
	return nil
}

// --- splice paths: rows ----------------------------------------------------

// insertRow vivifies the row for key at sorted position idx, evicting first
// if the table is at its cap (design doc §2.3/§2.4). It returns the index the
// new row ended up at, which the eviction may have moved.
func (e *schemaEngine) insertRow(pos, idx int, key []byte) (int, error) {
	td := e.schema.Fields[pos].Table
	tl := e.layout.Tables[pos]
	if td.Cap > 0 && uint32(e.rows(pos)) >= td.Cap { //nolint:gosec // rows bounded by OperateMaxRows
		if err := e.evictOne(pos, td.Policy, int(td.ByCol)); err != nil {
			return 0, err
		}
		var found bool
		if idx, found = e.findRow(pos, key); found {
			// The evicted row cannot be the one being inserted (it is not
			// there yet), so this is unreachable; treated as "already
			// present" rather than inserting a duplicate.
			return idx, nil
		}
	}
	rows := e.rows(pos)
	if rows+1 > wire.OperateMaxRows {
		return 0, wire.ErrOperateCap
	}
	grow := tl.RowWidth + uvarintLen(uint64(rows+1)) - uvarintLen(uint64(rows)) //nolint:gosec // rows >= 0
	if err := e.charge(grow); err != nil {
		return 0, err
	}

	at := e.rowStart(pos, idx)
	e.buf = insertGap(e.buf, at, tl.RowWidth)
	copy(e.buf[at:at+tl.KeyWidth], key)
	off := e.rowsOff(pos)
	e.buf, _ = spliceUvarint(e.buf, off, uvarintLen(uint64(rows)), uint64(rows+1)) //nolint:gosec // rows >= 0
	if err := e.reindexTail(); err != nil {
		return 0, err
	}
	return idx, nil
}

// removeRows splices count rows out of the table field at pos, starting at
// row index idx, and rewrites its row count.
func (e *schemaEngine) removeRows(pos, idx, count int) error {
	if count <= 0 {
		return nil
	}
	rows := e.rows(pos)
	tl := e.layout.Tables[pos]
	at := e.rowStart(pos, idx)
	e.buf = splice(e.buf, at, count*tl.RowWidth, nil)
	off := e.rowsOff(pos)
	e.buf, _ = spliceUvarint(e.buf, off, uvarintLen(uint64(rows)), uint64(rows-count)) //nolint:gosec // 0 <= count <= rows
	return e.reindexTail()
}

// insertGap opens n zeroed bytes at off, growing buf in one append when its
// capacity cannot absorb the gap.
func insertGap(buf []byte, off, n int) []byte {
	if n <= 0 {
		return buf
	}
	old := len(buf)
	if cap(buf)-old >= n {
		buf = buf[:old+n]
	} else {
		grown := make([]byte, old+n, old+n+64)
		copy(grown, buf[:off])
		copy(grown[off+n:], buf[off:old])
		for i := off; i < off+n; i++ {
			grown[i] = 0
		}
		return grown
	}
	copy(buf[off+n:], buf[off:old])
	for i := off; i < off+n; i++ {
		buf[i] = 0
	}
	return buf
}

// --- del, count ------------------------------------------------------------

// del implements engine.del (design doc §3.1/§2.5): the record is marked
// deleted, a schema field or column is zeroed (it cannot be removed — the
// schema declares it, so it is always present), a table is emptied, and a row
// is spliced out. Deleting something absent is a no-op.
func (e *schemaEngine) del(r ref) error {
	switch r.kind {
	case refKindRecord:
		e.deleted = true
		return nil
	case refKindTable:
		return e.removeRows(r.field, 0, e.rows(r.field))
	case refKindRow:
		if !r.present {
			return nil
		}
		return e.removeRows(r.field, r.row, 1)
	default: // refKindScalar: a field or a column
		if !r.present || r.off < 0 {
			return nil
		}
		return e.set(r, wire.ZeroCell(r.typ, r.n))
	}
}

// count implements engine.count (design doc §3.3/§3.4): fields for the
// record, rows for a table, and presence as 1/0 for a row or a scalar.
func (e *schemaEngine) count(r ref) (uint64, error) {
	switch r.kind {
	case refKindRecord:
		return uint64(len(e.schema.Fields)), nil
	case refKindTable:
		return uint64(e.rows(r.field)), nil //nolint:gosec // rows >= 0
	default:
		if r.present {
			return 1, nil
		}
		return 0, nil
	}
}

// --- eviction and TRIM -----------------------------------------------------

// evictOne removes the victim the policy names (design doc §2.3): the
// smallest or largest key (the ends of the sorted table), or the smallest or
// largest value in the eviction column, ties going to the smallest key.
//
// Ruling (mirrors the oracle): a cap above 0 with PolicyNone passes
// Schema.Validate, so it still needs a deterministic victim — it evicts by
// MIN_KEY, as does any policy byte outside the known set.
func (e *schemaEngine) evictOne(pos int, policy uint8, byCol int) error {
	rows := e.rows(pos)
	if rows == 0 {
		return nil
	}
	victim := 0
	switch policy {
	case wire.OperatePolicyMaxKey:
		victim = rows - 1
	case wire.OperatePolicyMinCol, wire.OperatePolicyMaxCol:
		victim = e.colVictim(pos, policy == wire.OperatePolicyMaxCol, byCol)
	default:
		victim = 0
	}
	return e.removeRows(pos, victim, 1)
}

// colVictim scans the table once for the row with the smallest (or largest)
// value in the eviction column; ties go to the first row in stored order,
// which is the smallest key.
func (e *schemaEngine) colVictim(pos int, wantMax bool, byCol int) int {
	td := e.schema.Fields[pos].Table
	tl := e.layout.Tables[pos]
	if byCol < 0 || byCol >= len(td.Cols) {
		// Unreachable: a *_COL schema policy is validated against the column
		// count, and TRIM checks its own byCol. Kept as a total function.
		return 0
	}
	typ, n := td.Cols[byCol].Type, td.Cols[byCol].N
	width := wire.CellWidth(typ, n)
	start := e.rowStart(pos, 0) + tl.ColOff[byCol]
	best := 0
	bestAt := start
	for i := 1; i < e.rows(pos); i++ {
		at := start + i*tl.RowWidth
		ord := compareCellBytes(e.buf[at:at+width], e.buf[bestAt:bestAt+width], typ)
		if (wantMax && ord > 0) || (!wantMax && ord < 0) {
			best, bestAt = i, at
		}
	}
	return best
}

// compareCellBytes orders two stored cells of the same type without decoding
// them into a Cell. Floats compare as float64s (a NaN compares equal to
// everything, so it never wins a tie-break), signed integers sign-extended,
// unsigned ones zero-extended, and FIXED bytewise — the same order the
// oracle's treeCompareCells produces for two cells of one type.
func compareCellBytes(a, b []byte, typ uint8) int {
	switch {
	case wire.TypeIsFloat(typ):
		af, bf := cellFloat(a, typ), cellFloat(b, typ)
		switch {
		case af < bf:
			return -1
		case af > bf:
			return 1
		default:
			return 0
		}
	case wire.TypeIsUnsigned(typ):
		av, bv := leUint(a), leUint(b)
		switch {
		case av < bv:
			return -1
		case av > bv:
			return 1
		default:
			return 0
		}
	case wire.TypeIsInt(typ):
		av, bv := signExtendLE(a), signExtendLE(b)
		switch {
		case av < bv:
			return -1
		case av > bv:
			return 1
		default:
			return 0
		}
	default: // FIXED
		return bytes.Compare(a, b)
	}
}

func cellFloat(b []byte, typ uint8) float64 {
	if typ == wire.OperateTypeF32 {
		return float64(math.Float32frombits(uint32(leUint(b)))) //nolint:gosec // 4 bytes by construction
	}
	return math.Float64frombits(leUint(b))
}

// signExtendLE reads up to 8 little-endian bytes as a signed integer of that
// width.
func signExtendLE(b []byte) int64 {
	v := leUint(b)
	bits := uint(len(b)) * 8
	if bits >= 64 {
		return int64(v) //nolint:gosec // reinterpret the stored bits
	}
	shift := 64 - bits
	return int64(v<<shift) >> shift //nolint:gosec // sign-extend
}

// trim implements engine.trim (design doc §3.2): a one-off shrink to keep
// rows by the op's own policy, leaving the table's stored eviction config
// alone. It evicts one row at a time (O(n·k)); k is the number of rows over
// the target, which is small in every intended use.
func (e *schemaEngine) trim(r ref, keep uint32, policy uint8, byCol uint16, byColName string) error {
	_ = byColName // schema mode addresses the eviction column by position
	if policy > wire.OperatePolicyMaxCol {
		// An unknown policy byte is a schema-shaped error, the same one
		// Schema.Validate returns for a table declaring it (oracle ruling).
		return wire.ErrOperateSchema
	}
	if r.kind != refKindTable {
		return wire.ErrOperatePath
	}
	td := e.schema.Fields[r.field].Table
	if policy == wire.OperatePolicyMinCol || policy == wire.OperatePolicyMaxCol {
		if int(byCol) >= len(td.Cols) {
			return wire.ErrOperatePath
		}
	}
	for e.rows(r.field) > int(keep) {
		if err := e.evictOne(r.field, policy, int(byCol)); err != nil {
			return err
		}
	}
	return nil
}

// config implements engine.config: a schema-mode table's eviction triple
// comes from its schema and cannot be set per record (design doc §3.2).
func (e *schemaEngine) config(_ ref, _ uint32, _ uint8, _ string) error {
	return wire.ErrOperateOpcode
}

// --- migrate ---------------------------------------------------------------

// migrate implements engine.migrate for schema mode (design doc §2.8):
// append-only schema evolution in one deterministic rewrite. Existing fields
// and columns keep their bytes, appended ones are zero-filled, and a table
// whose cap the new schema lowered is evicted down to it by the new policy.
func (e *schemaEngine) migrate(from int64, flags uint8, schemaBlob []byte) error {
	_ = flags // DROP_EXTRA only means anything to a dynamic-mode freeze
	ent, err := e.cache.lookupCall(schemaBlob)
	if err != nil {
		return err
	}
	if from == wire.OperateMigrateFromDynamic {
		// A freeze targets a dynamic record; this one is already frozen.
		return wire.ErrOperateMode
	}
	if from < 0 || from > math.MaxUint16 || uint16(from) != e.schema.Version { //nolint:gosec // range-checked
		return wire.ErrOperateSchemaVersion
	}
	if err := e.schema.Extends(ent.schema); err != nil {
		return err
	}
	return e.rebuild(ent)
}

// rebuild rewrites the record under a new (append-only extended) schema:
// fixed-width fields keep their bytes at their new offsets, each table's rows
// are copied with the appended columns zero-filled, and appended fields start
// at their zero.
func (e *schemaEngine) rebuild(ent *schemaEntry) error {
	old, oldLayout := e.schema, e.layout
	size, err := e.rebuiltSize(ent)
	if err != nil {
		return err
	}
	out := make([]byte, 0, size+64)
	out = append(out, wire.OperateModeSchema)
	out = append(out, ent.blob...)

	for i := range ent.schema.Fields {
		off := ent.layout.FixedOff[i]
		if off < 0 {
			continue
		}
		w := wire.CellWidth(ent.schema.Fields[i].Type, ent.schema.Fields[i].N)
		if i < len(old.Fields) {
			at := e.hdrLen + oldLayout.FixedOff[i]
			out = append(out, e.buf[at:at+w]...)
			continue
		}
		out = appendZeros(out, w)
	}

	for _, i := range ent.layout.VarTail {
		fd := &ent.schema.Fields[i]
		if i >= len(old.Fields) { // an appended field: its zero is one byte
			out = append(out, 0)
			continue
		}
		if fd.Type != wire.OperateTypeTable {
			at := e.fieldOff(i)
			end := at + e.varFieldLen(i)
			out = append(out, e.buf[at:end]...)
			continue
		}
		out = e.appendMigratedTable(out, i, ent)
	}

	e.buf = out
	e.ent, e.schema, e.layout = ent, ent.schema, ent.layout
	e.hdrLen = 1 + len(ent.blob)
	if len(ent.layout.VarTail) <= inlineTail {
		e.tail = e.tailArr[:0]
	} else {
		e.tail = make([]tailEntry, 0, len(ent.layout.VarTail))
	}
	if err := e.reindexTail(); err != nil {
		return err
	}
	// A lowered cap trims during the migration, by the new policy.
	for _, i := range ent.layout.VarTail {
		td := ent.schema.Fields[i].Table
		if td == nil || td.Cap == 0 {
			continue
		}
		for uint32(e.rows(i)) > td.Cap { //nolint:gosec // rows bounded by OperateMaxRows
			if err := e.evictOne(i, td.Policy, int(td.ByCol)); err != nil {
				return err
			}
		}
	}
	return nil
}

// rebuiltSize is the exact byte length the rebuilt record will have, charged
// against the record-size cap before anything is allocated for it.
func (e *schemaEngine) rebuiltSize(ent *schemaEntry) (int, error) {
	// Accumulated in uint64, never int: the migration can widen every row, so
	// rows x newRowWidth reaches 2^20 x 4096 = 2^32 — which wraps to a small
	// positive on a 32-bit int and would slip past the cap check, leaving
	// rebuild to append unbounded.
	size := uint64(1) + uint64(len(ent.blob)) + uint64(ent.layout.FixedLen)
	for _, i := range ent.layout.VarTail {
		switch {
		case i >= len(e.schema.Fields):
			size++
		case ent.schema.Fields[i].Type != wire.OperateTypeTable:
			size += uint64(e.varFieldLen(i)) //nolint:gosec // a stored length, bounded by the buffer
		default:
			size += tableBytes(e.rows(i), ent.layout.Tables[i].RowWidth)
		}
		if size > uint64(maxOperateRecordBytes) { //nolint:gosec // maxOperateRecordBytes > 0
			return 0, wire.ErrOperateCap
		}
	}
	// Unconditionally, not only per tail field: a target schema may have no
	// variable-length fields at all (every field fixed-width is legal), and
	// then the loop above never runs — while the header and the fixed area it
	// already counted can exceed the cap on their own, since OperateMaxFields
	// FIXED(255) fields are ~16.7 MB.
	if size > uint64(maxOperateRecordBytes) { //nolint:gosec // maxOperateRecordBytes > 0
		return 0, wire.ErrOperateCap
	}
	return int(size), nil //nolint:gosec // bounded by maxOperateRecordBytes, itself an int
}

// tableBytes is the stored size of a table of rows rows at rowWidth bytes each,
// computed in uint64 so the product cannot wrap: rows is bounded by
// OperateMaxRows (2^20) and rowWidth by OperateMaxRowWidth (4096), whose
// product does not fit a 32-bit int.
func tableBytes(rows, rowWidth int) uint64 {
	if rows < 0 || rowWidth < 0 {
		return 0
	}
	return uint64(uvarintLen(uint64(rows))) + uint64(rows)*uint64(rowWidth)
}

// varFieldLen is the stored length of variable-length field i under the
// engine's current schema.
func (e *schemaEngine) varFieldLen(i int) int {
	off := e.fieldOff(i)
	slot := e.ent.tailSlot[i]
	if slot+1 < len(e.tail) {
		return e.tail[slot+1].off - off
	}
	return len(e.buf) - off
}

// appendMigratedTable copies table field i's rows into out under the new
// schema, zero-filling every appended column.
func (e *schemaEngine) appendMigratedTable(out []byte, i int, ent *schemaEntry) []byte {
	rows := e.rows(i)
	oldWidth := e.layout.Tables[i].RowWidth
	newWidth := ent.layout.Tables[i].RowWidth
	out = binary.AppendUvarint(out, uint64(rows)) //nolint:gosec // rows >= 0
	at := e.rowStart(i, 0)
	for r := 0; r < rows; r++ {
		out = append(out, e.buf[at:at+oldWidth]...)
		out = appendZeros(out, newWidth-oldWidth)
		at += oldWidth
	}
	return out
}

func appendZeros(dst []byte, n int) []byte {
	for i := 0; i < n; i++ {
		dst = append(dst, 0)
	}
	return dst
}

// --- value, bytes, empty ---------------------------------------------------

// value implements engine.value (design doc §3.4): the whole record, a
// table's stored encoding behind the mode byte, a row's columns as tagged
// cells, or one tagged scalar.
func (e *schemaEngine) value(r ref) ([]byte, error) {
	switch r.kind {
	case refKindRecord:
		return append([]byte(nil), e.buf...), nil
	case refKindTable:
		// Ruling (the oracle's): no schema blob rides along — the caller
		// holds the schema, and the mode byte says which table layout
		// follows.
		off := e.rowsOff(r.field)
		end := off + uvarintLen(uint64(e.rows(r.field))) + e.rows(r.field)*e.layout.Tables[r.field].RowWidth //nolint:gosec // rows >= 0
		out := make([]byte, 0, 1+end-off)
		out = append(out, wire.OperateModeSchema)
		return append(out, e.buf[off:end]...), nil
	case refKindRow:
		return e.rowValue(r)
	default:
		c, err := e.get(r)
		if err != nil {
			return nil, err
		}
		return wire.AppendTaggedCell(nil, c), nil
	}
}

// rowValue is [mode][nCols uvarint]{tagged cell}* in column order.
func (e *schemaEngine) rowValue(r ref) ([]byte, error) {
	td := e.schema.Fields[r.field].Table
	tl := e.layout.Tables[r.field]
	start := e.rowStart(r.field, r.row)
	out := make([]byte, 0, 2+tl.RowWidth+len(td.Cols))
	out = append(out, wire.OperateModeSchema)
	out = binary.AppendUvarint(out, uint64(len(td.Cols)))
	for c := range td.Cols {
		cell, _, err := wire.DecodeCellData(td.Cols[c].Type, td.Cols[c].N, e.buf[start+tl.ColOff[c]:])
		if err != nil {
			return nil, err
		}
		out = wire.AppendTaggedCell(out, cell)
	}
	return out, nil
}

// bytes implements engine.bytes: the record's current encoded form, which is
// the engine's working buffer itself.
func (e *schemaEngine) bytes() []byte { return e.buf }

// empty implements engine.empty. A schema-mode record is never implicitly
// deleted — every declared field exists and an all-zero record is a valid
// state (design doc §2.5) — so only DEL () empties it.
func (e *schemaEngine) empty() bool { return e.deleted }
