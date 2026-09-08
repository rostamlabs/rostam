// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"sync"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// Kind classifies what a Resolve found at the end of a path.
type Kind uint8

const (
	// Absent means the path is well-formed for this record's shape but
	// names nothing in it: an unknown field, a row key with no row, a
	// column the table does not have. It is never an error — a filter over
	// many records expects most of them not to have the field.
	Absent Kind = iota
	// Scalar means the path resolved to a value; Result.Cell holds it,
	// typed as the schema (or, in dynamic mode, the stored tag) declares.
	Scalar
	// Count means the path ended in "#count" on a table field;
	// Result.Count holds the table's row count.
	Count
	// RowPresent means the path named a table row that exists. The row's
	// columns are not read (use a third segment for one).
	RowPresent
	// Table means the path named a table field with no row segment after
	// it. Result carries no value: a table is not a scalar, and its row
	// count is what "#count" is for.
	Table
)

// Result is what Resolve found. Cell is meaningful only for Scalar and
// Count only for Count; both are the zero value otherwise.
type Result struct {
	Kind  Kind
	Cell  wire.Cell
	Count uint64
}

// schemaEntry is one cached decoded schema: the schema itself, its
// precomputed byte layout, and the length of the blob it was decoded from
// (which is where a record's values begin).
type schemaEntry struct {
	schema  *wire.Schema
	layout  *wire.Layout
	blobLen int
}

// Resolver resolves paths against stored operate records (design doc
// §2.2/§2.9) by walking the stored bytes, without decoding the record into
// a tree. It caches decoded schemas so a schema-mode read of a fixed field,
// or of a column of an existing row, costs offset arithmetic and one binary
// search and allocates nothing.
//
// # Semantics
//
// Resolve is a pure function of the record bytes and the parsed Path. The
// schema cache is keyed by the schema blob's exact bytes, so a hit and a
// miss cannot produce different answers; nothing here consults the clock or
// any other ambient state. A Resolver is safe for concurrent use — the
// vector search path shares one across query goroutines.
//
// # Row keys
//
// A row-key segment names bytes, and how it does so depends on the mode:
//
//   - Schema mode: the table's key type and width come from the schema. A
//     decimal segment denotes the key type's width in little-endian bytes
//     (the encoding client.KeyU8/U16/U32/U64 produce); a value too large
//     for that width resolves to Absent. A quoted segment denotes its
//     unescaped bytes, which must be exactly the key width, else Absent.
//   - Dynamic mode: keys are opaque byte strings with no declared width. A
//     decimal segment denotes the 8-byte little-endian encoding of the
//     number (what client.KeyU64 produces, the convention dynamic writers
//     follow); a quoted segment denotes its unescaped bytes verbatim.
//
// # Errors
//
// Resolve returns an error only for record bytes it cannot read (wrapping
// ErrRecord) or a path that cannot apply to this record's shape at all
// (wrapping ErrPath): a row segment on a scalar field, "#count" on a
// non-table, a name where the schema stores no names, a position in
// dynamic mode. Anything merely missing is Result{Kind: Absent} with a nil
// error.
//
// # Hardening
//
// Stored record bytes are attacker-influenced (a client can put anything
// under a payload key), so every offset is bounds-checked before use, every
// count is bounded before it is trusted, only canonical uvarints are
// accepted, and BYTES/FIXED payloads are copied rather than aliased into
// the caller's buffer. Resolve reads only the bytes its path touches, so it
// is not a whole-record validator: it rejects everything wire.DecodeRecord
// would reject along that path, and does not notice damage elsewhere. Two
// deliberate consequences: a schema-mode table whose rows are not in the
// ascending key order DecodeRecord enforces may resolve a present row to
// Absent (binary search cannot see the disorder, and it never returns a row
// whose key does not match), and a field with no further segment resolves
// to Table without reading the table's bytes at all.
type Resolver struct {
	mu    sync.RWMutex
	cache map[string]*schemaEntry
	limit int
}

// NewResolver returns a Resolver whose schema cache holds at most
// cacheSize entries (cleared wholesale when full — schemas are few and
// long-lived, so an eviction policy would cost more than it saves). A
// cacheSize below 1 disables caching: every call decodes the schema afresh,
// which is slower but answers identically.
func NewResolver(cacheSize int) *Resolver {
	r := &Resolver{limit: cacheSize}
	if cacheSize > 0 {
		r.cache = make(map[string]*schemaEntry, cacheSize)
	}
	return r
}

// Resolve resolves p against the stored record rec.
func (r *Resolver) Resolve(rec []byte, p Path) (Result, error) {
	if err := checkPathShape(p); err != nil {
		return Result{}, err
	}
	if len(rec) < 1 {
		return Result{}, fmt.Errorf("%w: empty record", ErrRecord)
	}
	switch rec[0] {
	case wire.OperateModeSchema:
		return r.resolveSchema(rec, p)
	case wire.OperateModeDynamic:
		return resolveDynamic(rec, p)
	default:
		return Result{}, fmt.Errorf("%w: unknown mode byte %d", ErrRecord, rec[0])
	}
}

// checkPathShape rejects a Path that ParsePath would never build. Resolve
// is exported and Path is a plain struct, so a caller can hand over a
// hand-assembled one; every later stage assumes the segment kinds line up.
func checkPathShape(p Path) error {
	if len(p.Segs) == 0 || len(p.Segs) > 3 {
		return fmt.Errorf("%w: path has %d segments", ErrPath, len(p.Segs))
	}
	if k := p.Segs[0].Kind; k != SegField && k != SegCount {
		return fmt.Errorf("%w: first segment is not a field", ErrPath)
	}
	if p.Segs[0].Kind == SegCount && len(p.Segs) != 1 {
		return fmt.Errorf("%w: #count must be the only segment", ErrPath)
	}
	if len(p.Segs) >= 2 && p.Segs[1].Kind != SegRow {
		return fmt.Errorf("%w: second segment is not a row key", ErrPath)
	}
	if len(p.Segs) == 3 && p.Segs[2].Kind != SegCol {
		return fmt.Errorf("%w: third segment is not a column", ErrPath)
	}
	return nil
}

// --- schema mode ----------------------------------------------------------

func (r *Resolver) resolveSchema(rec []byte, p Path) (Result, error) {
	e, err := r.schemaFor(rec)
	if err != nil {
		return Result{}, err
	}
	s, l := e.schema, e.layout
	base := 1 + e.blobLen
	if base > len(rec) {
		return Result{}, fmt.Errorf("%w: record ends inside its schema blob", ErrRecord)
	}

	seg := p.Segs[0]
	var pos int
	if seg.ByPos {
		if uint64(seg.Pos) >= uint64(len(s.Fields)) {
			return Result{Kind: Absent}, nil
		}
		pos = int(seg.Pos)
	} else {
		if !s.StoreNames {
			return Result{}, fmt.Errorf("%w: schema stores no names, use a position", ErrPath)
		}
		i, ok := s.FieldPos(seg.Name)
		if !ok {
			return Result{Kind: Absent}, nil
		}
		pos = i
	}

	fdef := &s.Fields[pos]
	isTable := fdef.Type == wire.OperateTypeTable

	if seg.Kind == SegCount {
		if !isTable {
			return Result{}, fmt.Errorf("%w: #count on a non-table field", ErrPath)
		}
		off, err := schemaFieldOffset(rec, base, s, l, pos)
		if err != nil {
			return Result{}, err
		}
		nRows, _, err := schemaTableHeader(rec, off, l.Tables[pos])
		if err != nil {
			return Result{}, err
		}
		return Result{Kind: Count, Count: nRows}, nil
	}

	if len(p.Segs) == 1 {
		if isTable {
			return Result{Kind: Table}, nil
		}
		off, err := schemaFieldOffset(rec, base, s, l, pos)
		if err != nil {
			return Result{}, err
		}
		cell, err := readCellAt(rec, off, fdef.Type, fdef.N)
		if err != nil {
			return Result{}, err
		}
		return Result{Kind: Scalar, Cell: cell}, nil
	}

	if !isTable {
		return Result{}, fmt.Errorf("%w: row segment on a non-table field", ErrPath)
	}
	tl := l.Tables[pos]
	if tl == nil || fdef.Table == nil {
		return Result{}, fmt.Errorf("%w: table field has no layout", ErrRecord)
	}
	off, err := schemaFieldOffset(rec, base, s, l, pos)
	if err != nil {
		return Result{}, err
	}
	nRows, rowsOff, err := schemaTableHeader(rec, off, tl)
	if err != nil {
		return Result{}, err
	}
	idx, found := findSchemaRow(rec, rowsOff, nRows, tl, fdef.Table.KeyType, p.Segs[1])
	if !found {
		return Result{Kind: Absent}, nil
	}
	if len(p.Segs) == 2 {
		return Result{Kind: RowPresent}, nil
	}

	cseg := p.Segs[2]
	cols := fdef.Table.Cols
	var ci int
	if cseg.ByPos {
		if uint64(cseg.Pos) >= uint64(len(cols)) {
			return Result{Kind: Absent}, nil
		}
		ci = int(cseg.Pos)
	} else {
		if !s.StoreNames {
			return Result{}, fmt.Errorf("%w: schema stores no names, use a position", ErrPath)
		}
		found := -1
		for i := range cols {
			if cols[i].Name == cseg.Name {
				found = i
				break
			}
		}
		if found < 0 {
			return Result{Kind: Absent}, nil
		}
		ci = found
	}
	// idx < nRows and nRows*RowWidth was bounded against the remaining
	// bytes by schemaTableHeader, so this product cannot overflow an int.
	cellOff := rowsOff + idx*tl.RowWidth + tl.ColOff[ci]
	cell, err := readCellAt(rec, cellOff, cols[ci].Type, cols[ci].N)
	if err != nil {
		return Result{}, err
	}
	return Result{Kind: Scalar, Cell: cell}, nil
}

// schemaFieldOffset returns the offset of field target's stored value. A
// fixed-width field sits at a constant offset; a variable-length one has to
// be walked to, skipping every tail field in front of it.
func schemaFieldOffset(rec []byte, base int, s *wire.Schema, l *wire.Layout, target int) (int, error) {
	if l.FixedOff[target] >= 0 {
		off := base + l.FixedOff[target]
		if off > len(rec) {
			return 0, fmt.Errorf("%w: record ends before field %d", ErrRecord, target)
		}
		return off, nil
	}
	off := base + l.FixedLen
	if off > len(rec) {
		return 0, fmt.Errorf("%w: record ends inside its fixed fields", ErrRecord)
	}
	for _, i := range l.VarTail {
		if i == target {
			return off, nil
		}
		f := &s.Fields[i]
		if f.Type == wire.OperateTypeTable {
			nRows, rowsOff, err := schemaTableHeader(rec, off, l.Tables[i])
			if err != nil {
				return 0, err
			}
			off = rowsOff + int(nRows)*l.Tables[i].RowWidth
			continue
		}
		n, err := cellDataLen(rec, off, f.Type, f.N)
		if err != nil {
			return 0, err
		}
		off += n
	}
	// Unreachable for a Layout built from this schema: every field is
	// either fixed or in the tail.
	return 0, fmt.Errorf("%w: field %d is in neither layout half", ErrRecord, target)
}

// walkSchemaFields visits every top-level field of a schema-mode record, in
// schema position order, exactly once, reporting each field's byte offset
// (and, for a table field, its row count — visit needs it anyway to know
// where the table ends, so the walk hands it over rather than making the
// caller re-read the header). It is schemaFieldOffset's iteration form:
// IndexEntries needs an offset for every field, and calling
// schemaFieldOffset once per field would re-walk the variable-length tail
// from scratch each time, costing O(n²) instead of O(n).
func walkSchemaFields(rec []byte, base int, s *wire.Schema, l *wire.Layout, visit func(pos, off int, isTable bool, nRows uint64) error) error {
	off := base + l.FixedLen
	if off > len(rec) {
		return fmt.Errorf("%w: record ends inside its fixed fields", ErrRecord)
	}
	tailIdx := 0
	for pos := range s.Fields {
		if l.FixedOff[pos] >= 0 {
			foff := base + l.FixedOff[pos]
			if foff > len(rec) {
				return fmt.Errorf("%w: record ends before field %d", ErrRecord, pos)
			}
			if err := visit(pos, foff, false, 0); err != nil {
				return err
			}
			continue
		}
		if tailIdx >= len(l.VarTail) || l.VarTail[tailIdx] != pos {
			// Unreachable for a Layout built from this schema (VarTail lists
			// exactly the non-fixed positions, in order).
			return fmt.Errorf("%w: field %d is in neither layout half", ErrRecord, pos)
		}
		tailIdx++
		foff := off
		f := &s.Fields[pos]
		isTable := f.Type == wire.OperateTypeTable
		var nRows uint64
		if isTable {
			var rowsOff int
			var err error
			nRows, rowsOff, err = schemaTableHeader(rec, off, l.Tables[pos])
			if err != nil {
				return err
			}
			off = rowsOff + int(nRows)*l.Tables[pos].RowWidth
		} else {
			n, err := cellDataLen(rec, off, f.Type, f.N)
			if err != nil {
				return err
			}
			off += n
		}
		if err := visit(pos, foff, isTable, nRows); err != nil {
			return err
		}
	}
	return nil
}

// schemaTableHeader reads a schema-mode table's row count from off and
// returns it with the offset of the first row, having bounded the rows
// against the bytes actually left (design doc §2.3).
func schemaTableHeader(rec []byte, off int, tl *wire.TableLayout) (uint64, int, error) {
	if tl == nil || tl.RowWidth < 1 {
		return 0, 0, fmt.Errorf("%w: table has no row width", ErrRecord)
	}
	nRows, m, err := readCanonicalUvarint(rec, off)
	if err != nil {
		return 0, 0, err
	}
	if nRows > wire.OperateMaxRows {
		return 0, 0, fmt.Errorf("%w: table declares %d rows", ErrRecord, nRows)
	}
	rowsOff := off + m
	if !wire.CountFitsIn(int(nRows), len(rec)-rowsOff, tl.RowWidth) {
		return 0, 0, fmt.Errorf("%w: table declares %d rows but the record is too short", ErrRecord, nRows)
	}
	return nRows, rowsOff, nil
}

// findSchemaRow binary-searches a schema-mode table's fixed-width rows for
// the key seg names, returning its index. Rows are stored strictly
// ascending by key (DecodeRecord enforces it), numeric keys ordered as
// little-endian integers and FIXED keys bytewise.
func findSchemaRow(rec []byte, rowsOff int, nRows uint64, tl *wire.TableLayout, keyType uint8, seg Segment) (int, bool) {
	kw := tl.KeyWidth
	var target []byte
	var targetVal uint64
	if seg.KeyQuoted {
		if len(seg.Key) != kw {
			// No row of this table can have a key of that length.
			return 0, false
		}
		target = seg.Key
	} else {
		v, err := strconv.ParseUint(seg.KeyText, 10, 64)
		if err != nil {
			return 0, false
		}
		if kw < 8 && v >= uint64(1)<<(8*uint(kw)) {
			return 0, false
		}
		targetVal = v
	}

	lo, hi := 0, int(nRows)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		start := rowsOff + mid*tl.RowWidth
		k := rec[start : start+kw]
		var cmp int
		if target != nil {
			cmp = compareStoredKeyBytes(k, target, keyType)
		} else {
			cmp = compareStoredKeyValue(k, targetVal, keyType)
		}
		if cmp < 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= int(nRows) {
		return 0, false
	}
	start := rowsOff + lo*tl.RowWidth
	k := rec[start : start+kw]
	if target != nil {
		if compareStoredKeyBytes(k, target, keyType) != 0 {
			return 0, false
		}
	} else if compareStoredKeyValue(k, targetVal, keyType) != 0 {
		return 0, false
	}
	return lo, true
}

// compareStoredKeyBytes compares a stored key against target under the
// table's key ordering: bytewise for FIXED, little-endian integer for every
// other legal key type. Both are the table's key width.
func compareStoredKeyBytes(k, target []byte, keyType uint8) int {
	if keyType == wire.OperateTypeFixed {
		return bytes.Compare(k, target)
	}
	return compareLEBytes(k, target)
}

// compareStoredKeyValue compares a stored key against the key a decimal
// segment denotes, without materialising that key: for FIXED the target is
// its little-endian bytes compared bytewise, otherwise it is the integer
// itself.
func compareStoredKeyValue(k []byte, v uint64, keyType uint8) int {
	if keyType == wire.OperateTypeFixed {
		for i := 0; i < len(k); i++ {
			t := leByte(v, i)
			if k[i] != t {
				if k[i] < t {
					return -1
				}
				return 1
			}
		}
		return 0
	}
	kv := leValueOf(k)
	switch {
	case kv < v:
		return -1
	case kv > v:
		return 1
	default:
		return 0
	}
}

// leByte returns byte i of v's little-endian encoding, zero past its width.
func leByte(v uint64, i int) byte {
	if i >= 8 {
		return 0
	}
	return byte(v >> (8 * uint(i)))
}

// leValueOf decodes at most 8 little-endian bytes as an unsigned integer.
func leValueOf(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// compareLEBytes compares two equal-width little-endian byte strings as
// unsigned integers.
func compareLEBytes(a, b []byte) int {
	av, bv := leValueOf(a), leValueOf(b)
	switch {
	case av < bv:
		return -1
	case av > bv:
		return 1
	default:
		return 0
	}
}

// --- schema cache ---------------------------------------------------------

// schemaFor returns the decoded schema for rec, from the cache when the
// blob is one already seen.
func (r *Resolver) schemaFor(rec []byte) (*schemaEntry, error) {
	blob := rec[1:]
	if n, ok := schemaBlobLen(blob); ok {
		// A cached key is a complete, valid schema blob. The stored format
		// is parsed strictly front to back, so if those exact bytes are a
		// prefix of this record's blob they ARE this record's blob — the
		// scanner only proposes where to cut, the byte equality decides.
		if e := r.lookup(blob[:n]); e != nil {
			return e, nil
		}
	}
	s, n, err := wire.DecodeSchema(blob)
	if err != nil {
		return nil, fmt.Errorf("%w: schema blob: %v", ErrRecord, err)
	}
	l, err := s.Layout()
	if err != nil {
		return nil, fmt.Errorf("%w: schema layout: %v", ErrRecord, err)
	}
	e := &schemaEntry{schema: s, layout: l, blobLen: n}
	r.store(blob[:n], e)
	return e, nil
}

func (r *Resolver) lookup(blob []byte) *schemaEntry {
	if r.cache == nil {
		return nil
	}
	r.mu.RLock()
	// Written as a direct index of a string conversion so the compiler
	// looks the key up without allocating it.
	e := r.cache[string(blob)]
	r.mu.RUnlock()
	return e
}

func (r *Resolver) store(blob []byte, e *schemaEntry) {
	if r.cache == nil {
		return
	}
	r.mu.Lock()
	if len(r.cache) >= r.limit {
		clear(r.cache)
	}
	r.cache[string(blob)] = e
	r.mu.Unlock()
}

// schemaBlobLen reports how many bytes the schema blob at the front of b
// occupies, walking its declarations without decoding them (design doc
// §2.2/§2.3). It is a cache-lookup optimisation only: it never decides
// whether a blob is valid, and a false result — or a wrong one — costs a
// cache miss and nothing else, because the cached entry is keyed by the
// blob's exact bytes and wire.DecodeSchema is the authority on the miss
// path.
func schemaBlobLen(b []byte) (int, bool) {
	if len(b) < 4 {
		return 0, false
	}
	off := 3 // version u16 + flags u8
	nFields, m, ok := scanUvarint(b, off)
	if !ok || nFields > wire.OperateMaxFields {
		return 0, false
	}
	off += m
	for i := uint64(0); i < nFields; i++ {
		if off >= len(b) {
			return 0, false
		}
		typ := b[off]
		off++
		if typ == wire.OperateTypeFixed {
			if off >= len(b) {
				return 0, false
			}
			off++
		}
		if typ != wire.OperateTypeTable {
			continue
		}
		if off >= len(b) {
			return 0, false
		}
		keyType := b[off]
		off++
		if keyType == wire.OperateTypeFixed {
			if off >= len(b) {
				return 0, false
			}
			off++
		}
		nCols, m2, ok := scanUvarint(b, off)
		if !ok || nCols > wire.OperateMaxCols {
			return 0, false
		}
		off += m2
		for c := uint64(0); c < nCols; c++ {
			if off >= len(b) {
				return 0, false
			}
			ct := b[off]
			off++
			if ct == wire.OperateTypeFixed {
				if off >= len(b) {
					return 0, false
				}
				off++
			}
		}
		if _, m3, ok := scanUvarint(b, off); ok { // cap
			off += m3
		} else {
			return 0, false
		}
		if off >= len(b) {
			return 0, false
		}
		off++                                     // policy
		if _, m4, ok := scanUvarint(b, off); ok { // byCol
			off += m4
		} else {
			return 0, false
		}
	}
	nameBytes, m5, ok := scanUvarint(b, off)
	if !ok {
		return 0, false
	}
	off += m5
	if nameBytes > uint64(len(b)-off) {
		return 0, false
	}
	return off + int(nameBytes), true
}

// scanUvarint reads one uvarint at off without the canonicality check
// readCanonicalUvarint applies: schemaBlobLen only needs the length, and a
// non-canonical blob is rejected by DecodeSchema on the cache-miss path.
func scanUvarint(b []byte, off int) (uint64, int, bool) {
	if off < 0 || off > len(b) {
		return 0, 0, false
	}
	v, m := binary.Uvarint(b[off:])
	if m <= 0 {
		return 0, 0, false
	}
	return v, m, true
}

// --- dynamic mode ---------------------------------------------------------

func resolveDynamic(rec []byte, p Path) (Result, error) {
	seg := p.Segs[0]
	if seg.ByPos {
		return Result{}, fmt.Errorf("%w: dynamic records address fields by name, not position", ErrPath)
	}

	nFields, m, err := readCanonicalUvarint(rec, 1)
	if err != nil {
		return Result{}, err
	}
	off := 1 + m
	if nFields > wire.OperateMaxFields {
		return Result{}, fmt.Errorf("%w: record declares %d fields", ErrRecord, nFields)
	}
	if !wire.CountFitsIn(int(nFields), len(rec)-off, 3) {
		return Result{}, fmt.Errorf("%w: record declares %d fields but is too short", ErrRecord, nFields)
	}

	var prevName []byte
	for i := uint64(0); i < nFields; i++ {
		name, next, err := readName(rec, off)
		if err != nil {
			return Result{}, err
		}
		off = next
		// Fields are stored strictly ascending by name; checking it as we
		// walk is what lets the walk stop early, and is the same invariant
		// DecodeRecord enforces.
		if prevName != nil && bytes.Compare(prevName, name) >= 0 {
			return Result{}, fmt.Errorf("%w: fields out of order", ErrRecord)
		}
		prevName = name

		cmp := compareBytesString(name, seg.Name)
		if cmp > 0 {
			return Result{Kind: Absent}, nil
		}

		typ, n, valOff, err := readTypeTag(rec, off)
		if err != nil {
			return Result{}, err
		}
		if cmp == 0 {
			return resolveDynamicField(rec, valOff, typ, n, p)
		}
		off, err = skipDynamicValue(rec, valOff, typ, n)
		if err != nil {
			return Result{}, err
		}
	}
	return Result{Kind: Absent}, nil
}

// resolveDynamicField finishes a dynamic-mode resolution once the named
// field has been found at valOff with the stored type (typ, n).
func resolveDynamicField(rec []byte, valOff int, typ, n uint8, p Path) (Result, error) {
	isTable := typ == wire.OperateTypeTable

	if p.Segs[0].Kind == SegCount {
		if !isTable {
			return Result{}, fmt.Errorf("%w: #count on a non-table field", ErrPath)
		}
		inner, _, err := dynamicTableInner(rec, valOff)
		if err != nil {
			return Result{}, err
		}
		nRows, _, err := dynamicTableRows(inner)
		if err != nil {
			return Result{}, err
		}
		return Result{Kind: Count, Count: nRows}, nil
	}
	if len(p.Segs) == 1 {
		if isTable {
			return Result{Kind: Table}, nil
		}
		cell, err := readCellAt(rec, valOff, typ, n)
		if err != nil {
			return Result{}, err
		}
		return Result{Kind: Scalar, Cell: cell}, nil
	}
	if !isTable {
		return Result{}, fmt.Errorf("%w: row segment on a non-table field", ErrPath)
	}

	inner, _, err := dynamicTableInner(rec, valOff)
	if err != nil {
		return Result{}, err
	}
	nRows, rowsOff, err := dynamicTableRows(inner)
	if err != nil {
		return Result{}, err
	}

	seg := p.Segs[1]
	var keyBuf [8]byte
	target := seg.Key
	if !seg.KeyQuoted {
		v, perr := strconv.ParseUint(seg.KeyText, 10, 64)
		if perr != nil {
			return Result{Kind: Absent}, nil
		}
		binary.LittleEndian.PutUint64(keyBuf[:], v)
		target = keyBuf[:]
	}

	off := rowsOff
	var prevKey []byte
	for i := uint64(0); i < nRows; i++ {
		rowLen, m, err := readCanonicalUvarint(inner, off)
		if err != nil {
			return Result{}, err
		}
		off += m
		if !fitsRemaining(rowLen, len(inner)-off) {
			return Result{}, fmt.Errorf("%w: row runs past its table", ErrRecord)
		}
		row := inner[off : off+int(rowLen)]
		off += int(rowLen)

		if len(row) < 1 {
			return Result{}, fmt.Errorf("%w: empty row", ErrRecord)
		}
		klen := int(row[0])
		if klen == 0 || klen > wire.OperateMaxKeyLen {
			return Result{}, fmt.Errorf("%w: row key length %d", ErrRecord, klen)
		}
		if len(row)-1 < klen {
			return Result{}, fmt.Errorf("%w: row ends inside its key", ErrRecord)
		}
		key := row[1 : 1+klen]
		// Rows are stored strictly ascending by key bytes, the invariant
		// DecodeRecord enforces and the one the early stop below rests on.
		if prevKey != nil && bytes.Compare(prevKey, key) >= 0 {
			return Result{}, fmt.Errorf("%w: rows out of order", ErrRecord)
		}
		prevKey = key

		switch bytes.Compare(key, target) {
		case 1:
			return Result{Kind: Absent}, nil
		case 0:
			return resolveDynamicRow(row[1+klen:], p)
		}
	}
	return Result{Kind: Absent}, nil
}

// resolveDynamicRow finishes a dynamic-mode resolution inside the row whose
// key matched; b is the row's content after its key.
func resolveDynamicRow(b []byte, p Path) (Result, error) {
	if len(p.Segs) == 2 {
		return Result{Kind: RowPresent}, nil
	}
	cseg := p.Segs[2]
	if cseg.ByPos {
		return Result{}, fmt.Errorf("%w: dynamic rows address columns by name, not position", ErrPath)
	}

	nCols, m, err := readCanonicalUvarint(b, 0)
	if err != nil {
		return Result{}, err
	}
	off := m
	if nCols > wire.OperateMaxCols {
		return Result{}, fmt.Errorf("%w: row declares %d columns", ErrRecord, nCols)
	}
	if !wire.CountFitsIn(int(nCols), len(b)-off, 3) {
		return Result{}, fmt.Errorf("%w: row declares %d columns but is too short", ErrRecord, nCols)
	}

	var prevName []byte
	for i := uint64(0); i < nCols; i++ {
		name, next, err := readName(b, off)
		if err != nil {
			return Result{}, err
		}
		off = next
		if prevName != nil && bytes.Compare(prevName, name) >= 0 {
			return Result{}, fmt.Errorf("%w: columns out of order", ErrRecord)
		}
		prevName = name

		cmp := compareBytesString(name, cseg.Name)
		if cmp > 0 {
			return Result{Kind: Absent}, nil
		}
		typ, n, valOff, err := readTypeTag(b, off)
		if err != nil {
			return Result{}, err
		}
		if typ == wire.OperateTypeTable {
			// A table is never a column value (design doc §2.9).
			return Result{}, fmt.Errorf("%w: column holds a table", ErrRecord)
		}
		if cmp == 0 {
			cell, cerr := readCellAt(b, valOff, typ, n)
			if cerr != nil {
				return Result{}, cerr
			}
			return Result{Kind: Scalar, Cell: cell}, nil
		}
		ln, lerr := cellDataLen(b, valOff, typ, n)
		if lerr != nil {
			return Result{}, lerr
		}
		off = valOff + ln
	}
	return Result{Kind: Absent}, nil
}

// dynamicTableInner slices a dynamic-mode table's byteLen-delimited body
// (design doc §2.9) and reports where the field's value ends.
func dynamicTableInner(rec []byte, off int) ([]byte, int, error) {
	byteLen, m, err := readCanonicalUvarint(rec, off)
	if err != nil {
		return nil, 0, err
	}
	start := off + m
	if !fitsRemaining(byteLen, len(rec)-start) {
		return nil, 0, fmt.Errorf("%w: table runs past the record", ErrRecord)
	}
	end := start + int(byteLen)
	return rec[start:end], end, nil
}

// dynamicTableRows reads a dynamic-mode table body's eviction triple and
// row count, returning the count and the offset of the first row within
// inner.
func dynamicTableRows(inner []byte) (uint64, int, error) {
	capV, m, err := readCanonicalUvarint(inner, 0)
	if err != nil {
		return 0, 0, err
	}
	off := m
	if capV > wire.OperateMaxRows {
		return 0, 0, fmt.Errorf("%w: table cap %d", ErrRecord, capV)
	}
	if len(inner)-off < 1 {
		return 0, 0, fmt.Errorf("%w: table ends before its policy", ErrRecord)
	}
	policy := inner[off]
	off++
	if policy > wire.OperatePolicyMaxCol {
		return 0, 0, fmt.Errorf("%w: unknown eviction policy %d", ErrRecord, policy)
	}
	if len(inner)-off < 1 {
		return 0, 0, fmt.Errorf("%w: table ends before its eviction column", ErrRecord)
	}
	byColLen := int(inner[off])
	off++
	if len(inner)-off < byColLen {
		return 0, 0, fmt.Errorf("%w: table ends inside its eviction column name", ErrRecord)
	}
	off += byColLen
	// A *_COL policy evicts by a named column, so it is malformed without
	// the name — the same pairing DecodeRecord refuses.
	if (policy == wire.OperatePolicyMinCol || policy == wire.OperatePolicyMaxCol) && byColLen == 0 {
		return 0, 0, fmt.Errorf("%w: column eviction policy with no column", ErrRecord)
	}

	nRows, m2, err := readCanonicalUvarint(inner, off)
	if err != nil {
		return 0, 0, err
	}
	off += m2
	if nRows > wire.OperateMaxRows {
		return 0, 0, fmt.Errorf("%w: table declares %d rows", ErrRecord, nRows)
	}
	if !wire.CountFitsIn(int(nRows), len(inner)-off, 3) {
		return 0, 0, fmt.Errorf("%w: table declares %d rows but is too short", ErrRecord, nRows)
	}
	return nRows, off, nil
}

// skipDynamicValue returns the offset just past the value of a dynamic
// field of the given stored type.
func skipDynamicValue(rec []byte, off int, typ, n uint8) (int, error) {
	if typ == wire.OperateTypeTable {
		_, end, err := dynamicTableInner(rec, off)
		return end, err
	}
	ln, err := cellDataLen(rec, off, typ, n)
	if err != nil {
		return 0, err
	}
	return off + ln, nil
}

// readName reads a [len u8][name] entry, returning the name's bytes (a
// sub-slice of b, never copied) and the offset just past it.
func readName(b []byte, off int) ([]byte, int, error) {
	if off < 0 || off >= len(b) {
		return nil, 0, fmt.Errorf("%w: truncated name", ErrRecord)
	}
	n := int(b[off])
	off++
	if n == 0 || n > wire.OperateMaxNameLen {
		return nil, 0, fmt.Errorf("%w: name length %d", ErrRecord, n)
	}
	if len(b)-off < n {
		return nil, 0, fmt.Errorf("%w: truncated name", ErrRecord)
	}
	return b[off : off+n], off + n, nil
}

// readTypeTag reads a [type u8][n u8 iff FIXED] tag, returning the type,
// the FIXED width, and the offset of the value that follows.
func readTypeTag(b []byte, off int) (typ, n uint8, valOff int, err error) {
	if off < 0 || off >= len(b) {
		return 0, 0, 0, fmt.Errorf("%w: truncated type tag", ErrRecord)
	}
	typ = b[off]
	off++
	if typ == wire.OperateTypeFixed {
		if off >= len(b) {
			return 0, 0, 0, fmt.Errorf("%w: truncated FIXED width", ErrRecord)
		}
		n = b[off]
		off++
		// FIXED's declared width is 1-255; a stored 0 is malformed, not a
		// zero-width value.
		if n == 0 {
			return 0, 0, 0, fmt.Errorf("%w: FIXED(0) is not a width", ErrRecord)
		}
	}
	return typ, n, off, nil
}

// --- shared byte readers --------------------------------------------------

// readCellAt decodes one cell of type (typ, n) at off. It rejects the
// non-canonical spellings wire.DecodeRecord rejects — an over-long varint,
// and any NaN but the canonical quiet one — so the resolver accepts exactly
// what the tree decoder accepts on the bytes it touches.
func readCellAt(rec []byte, off int, typ, n uint8) (wire.Cell, error) {
	if off < 0 || off > len(rec) {
		return wire.Cell{}, fmt.Errorf("%w: cell offset out of range", ErrRecord)
	}
	b := rec[off:]
	switch typ {
	case wire.OperateTypeUVarint, wire.OperateTypeIVarint, wire.OperateTypeBytes:
		if _, _, err := readCanonicalUvarint(b, 0); err != nil {
			return wire.Cell{}, err
		}
	case wire.OperateTypeF32:
		if len(b) < 4 {
			return wire.Cell{}, fmt.Errorf("%w: truncated F32", ErrRecord)
		}
		bits := binary.LittleEndian.Uint32(b[:4])
		if math.IsNaN(float64(math.Float32frombits(bits))) && bits != canonicalNaN32 {
			return wire.Cell{}, fmt.Errorf("%w: non-canonical F32 NaN", ErrRecord)
		}
	case wire.OperateTypeF64:
		if len(b) < 8 {
			return wire.Cell{}, fmt.Errorf("%w: truncated F64", ErrRecord)
		}
		bits := binary.LittleEndian.Uint64(b[:8])
		if math.IsNaN(math.Float64frombits(bits)) && bits != canonicalNaN64 {
			return wire.Cell{}, fmt.Errorf("%w: non-canonical F64 NaN", ErrRecord)
		}
	}
	cell, _, err := wire.DecodeCellData(typ, n, b)
	if err != nil {
		return wire.Cell{}, fmt.Errorf("%w: cell: %v", ErrRecord, err)
	}
	return cell, nil
}

// canonicalNaN32 and canonicalNaN64 are the only NaN bit patterns
// wire.AppendCellData writes (design doc §2.5).
const (
	canonicalNaN32 uint32 = 0x7FC00000
	canonicalNaN64 uint64 = 0x7FF8000000000000
)

// cellDataLen returns the stored byte length of a cell of type (typ, n) at
// off, without decoding its value — what a skip needs.
func cellDataLen(rec []byte, off int, typ, n uint8) (int, error) {
	if off < 0 || off > len(rec) {
		return 0, fmt.Errorf("%w: cell offset out of range", ErrRecord)
	}
	switch typ {
	case wire.OperateTypeUnset:
		return 0, nil
	case wire.OperateTypeUVarint, wire.OperateTypeIVarint:
		_, m, err := readCanonicalUvarint(rec, off)
		return m, err
	case wire.OperateTypeBytes:
		ln, m, err := readCanonicalUvarint(rec, off)
		if err != nil {
			return 0, err
		}
		if ln > wire.OperateMaxBytesLen {
			return 0, fmt.Errorf("%w: BYTES declares %d bytes", ErrRecord, ln)
		}
		if !wire.CountFitsIn(int(ln), len(rec)-(off+m), 1) {
			return 0, fmt.Errorf("%w: BYTES runs past the record", ErrRecord)
		}
		return m + int(ln), nil
	case wire.OperateTypeTable:
		return 0, fmt.Errorf("%w: a table is not cell data", ErrRecord)
	default:
		w := wire.CellWidth(typ, n)
		if w < 1 {
			// -1 is an unknown type tag; 0 can only come from FIXED(0),
			// which is not a legal width.
			return 0, fmt.Errorf("%w: type %d has no stored width", ErrRecord, typ)
		}
		if len(rec)-off < w {
			return 0, fmt.Errorf("%w: record ends inside a %d-byte cell", ErrRecord, w)
		}
		return w, nil
	}
}

// readCanonicalUvarint reads one uvarint at off, rejecting a truncated
// encoding and a non-minimal (zero-padded) one — the same rule
// sdk/wire/operate_record.go applies, and the one that keeps a stored
// record's bytes the only spelling of its value.
func readCanonicalUvarint(b []byte, off int) (uint64, int, error) {
	if off < 0 || off > len(b) {
		return 0, 0, fmt.Errorf("%w: varint offset out of range", ErrRecord)
	}
	v, m := binary.Uvarint(b[off:])
	if m <= 0 { // 0 = truncated, <0 = more than 64 bits
		return 0, 0, fmt.Errorf("%w: truncated or oversized varint", ErrRecord)
	}
	if m != minUvarintLen(v) {
		return 0, 0, fmt.Errorf("%w: non-canonical varint", ErrRecord)
	}
	return v, m, nil
}

// minUvarintLen returns the number of bytes binary.AppendUvarint uses for
// v: the canonical LEB128 length.
func minUvarintLen(v uint64) int {
	n := 1
	for v >= 0x80 {
		v >>= 7
		n++
	}
	return n
}

// fitsRemaining reports whether a decoded byte length fits in remaining
// bytes, without narrowing it to int first: a length prefix is an arbitrary
// attacker-controlled uvarint, and converting before comparing would wrap
// on a 32-bit int.
func fitsRemaining(n uint64, remaining int) bool {
	if remaining < 0 {
		return false
	}
	return n <= uint64(remaining)
}

// compareBytesString compares a byte slice against a string the way
// bytes.Compare would, without the allocation a conversion would cost.
func compareBytesString(b []byte, s string) int {
	n := len(b)
	if len(s) < n {
		n = len(s)
	}
	for i := 0; i < n; i++ {
		if b[i] != s[i] {
			if b[i] < s[i] {
				return -1
			}
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
