// SPDX-License-Identifier: Apache-2.0

package wire

import "encoding/binary"

// ColumnDef is one fixed-width column of a table field (design doc §2.3):
// its type and, for OperateTypeFixed, declared width. Name is optional and
// used only client-side (§2.2) unless the schema's StoreNames is set.
type ColumnDef struct {
	Name string
	Type uint8
	N    uint8 // OperateTypeFixed width only; ignored otherwise
}

// TableDef is a table field's header (design doc §2.3): its row key type,
// its columns (fixed-width only), and its optional eviction policy. Cap == 0
// means unbounded; Policy/ByCol select the eviction victim when the table is
// full (OperatePolicy*).
type TableDef struct {
	KeyType uint8
	KeyN    uint8 // OperateTypeFixed key width only; ignored otherwise
	Cols    []ColumnDef
	Cap     uint32
	Policy  uint8
	ByCol   uint16
}

// FieldDef is one field of a record's schema (design doc §2.2): its type
// and, for OperateTypeFixed, declared width, or, for OperateTypeTable, the
// table's header. Name is optional client-side metadata, stored on the wire
// only when the owning Schema's StoreNames is set.
type FieldDef struct {
	Name  string
	Type  uint8
	N     uint8 // OperateTypeFixed width only; ignored otherwise
	Table *TableDef
}

// Schema is a record's field layout (design doc §2.2/§2.8): a client-chosen
// Version and an ordered list of fields. It is defined by the client,
// versioned, and immutable per record — a record's schema is stored inline
// in front of its values so the record is self-describing. StoreNames
// controls whether field/column names ride the wire (costing bytes) or live
// only in the client's in-memory Schema object.
type Schema struct {
	Version    uint16
	StoreNames bool
	Fields     []FieldDef
}

// TableLayout is the precomputed byte layout of one table field (design doc
// §2.3): the key's width, the width of one full row (key + columns), and
// each column's byte offset within a row.
type TableLayout struct {
	KeyWidth int
	RowWidth int
	ColOff   []int
}

// Layout is a schema's precomputed byte layout (design doc §2.2): the
// fixed-width fields are packed first, in schema order, at constant offsets;
// the variable-length fields (BYTES, UVARINT, IVARINT, TABLE) follow in the
// "tail" and must be read to skip. FixedOff[i] is the byte offset of field i
// among the fixed-width fields, or -1 if field i is variable-length.
// Tables[i] is non-nil exactly when field i is an OperateTypeTable field.
type Layout struct {
	FixedOff []int
	FixedLen int
	VarTail  []int
	Tables   []*TableLayout
}

// schemaFlagStoreNames is bit 0 of a schema blob's flags byte: whether the
// names section at the end of the blob is present (design doc §2.2).
const schemaFlagStoreNames uint8 = 1 << 0

// minUvarintLen returns the number of bytes binary.AppendUvarint would use to
// encode v: the canonical (minimal) LEB128 length.
//
// TWIN: sdk/record/resolve.go carries an unexported copy so the record leaf can
// enforce the same canonical-uvarint rule without importing it. Keep both in
// lockstep.
func minUvarintLen(v uint64) int {
	n := 1
	for v >= 0x80 {
		v >>= 7
		n++
	}
	return n
}

// decodeCanonicalUvarint reads one uvarint from the front of b, rejecting a
// truncated encoding and also a non-canonical (zero-padded) one: LEB128
// allows the same value to be spelled with extra all-continuation-bit-set,
// zero-value bytes (e.g. 0x80 0x00 for 0), which would decode fine but never
// be produced by Encode — accepting it would break the
// Encode(DecodeSchema(b)) == b identity hardened decoding requires.
func decodeCanonicalUvarint(b []byte) (uint64, int, error) {
	v, m := binary.Uvarint(b)
	if m <= 0 {
		return 0, 0, ErrShortArgs
	}
	if m != minUvarintLen(v) {
		return 0, 0, ErrOperateSchema
	}
	return v, m, nil
}

// Encode writes s in the wire format (design doc §2.2/§2.3):
//
//	[version u16 LE][flags u8][nFields uvarint]
//	  per field: [type u8][n u8 iff FIXED]
//	             iff TABLE: [keyType u8][keyN u8 iff FIXED][nCols uvarint]
//	                        {[type u8][n u8 iff FIXED]}* [cap uvarint][policy u8][byCol uvarint]
//	[nameBytes uvarint]
//	  iff StoreNames: per field [len u8][name], then per table's columns [len u8][name], in field order
//
// nameBytes is the byte length of the names section that follows it (0 when
// StoreNames is false, or trivially when there are no fields). Encode
// assumes s is well-formed (Validate has been, or will be, checked
// separately); it does not itself return an error.
func (s *Schema) Encode() []byte {
	b := make([]byte, 0, 32)
	b = binary.LittleEndian.AppendUint16(b, s.Version)
	var flags uint8
	if s.StoreNames {
		flags |= schemaFlagStoreNames
	}
	b = append(b, flags)
	b = binary.AppendUvarint(b, uint64(len(s.Fields))) //nolint:gosec // bounded by OperateMaxFields on validated schemas

	for i := range s.Fields {
		f := &s.Fields[i]
		b = append(b, f.Type)
		if f.Type == OperateTypeFixed {
			b = append(b, f.N)
		}
		if f.Type == OperateTypeTable && f.Table != nil {
			t := f.Table
			b = append(b, t.KeyType)
			if t.KeyType == OperateTypeFixed {
				b = append(b, t.KeyN)
			}
			b = binary.AppendUvarint(b, uint64(len(t.Cols))) //nolint:gosec // bounded by OperateMaxCols on validated schemas
			for _, c := range t.Cols {
				b = append(b, c.Type)
				if c.Type == OperateTypeFixed {
					b = append(b, c.N)
				}
			}
			b = binary.AppendUvarint(b, uint64(t.Cap))
			b = append(b, t.Policy)
			b = binary.AppendUvarint(b, uint64(t.ByCol))
		}
	}

	if !s.StoreNames {
		return binary.AppendUvarint(b, 0)
	}

	var names []byte
	for i := range s.Fields {
		names = appendName(names, s.Fields[i].Name)
	}
	for i := range s.Fields {
		f := &s.Fields[i]
		if f.Type == OperateTypeTable && f.Table != nil {
			for _, c := range f.Table.Cols {
				names = appendName(names, c.Name)
			}
		}
	}
	b = binary.AppendUvarint(b, uint64(len(names))) //nolint:gosec // bounded by OperateMaxNameBytes on validated schemas
	return append(b, names...)
}

// appendName appends one [len u8][name] entry. Callers are responsible for
// keeping name within OperateMaxNameLen (Validate enforces this).
func appendName(dst []byte, name string) []byte {
	dst = append(dst, byte(len(name))) //nolint:gosec // bounded by OperateMaxNameLen on validated schemas
	return append(dst, name...)
}

// DecodeSchema reads one Encode-formatted schema blob from the front of b,
// returning the decoded schema and the number of bytes consumed. Every
// count is bounded with CountFitsIn against the remaining input, against the
// relevant cap, and against OperateMaxSchemaBytes (the blob a schema must
// fit in) before it is trusted to size an allocation, and every read is
// truncation-checked, so DecodeSchema never panics or over-reads on hostile
// input. A successfully decoded schema always passes Validate — DecodeSchema
// runs it before returning.
func DecodeSchema(b []byte) (*Schema, int, error) {
	if len(b) < 4 {
		return nil, 0, ErrShortArgs
	}
	s := &Schema{Version: binary.LittleEndian.Uint16(b[0:2])}
	flags := b[2]
	if flags&^schemaFlagStoreNames != 0 {
		// Reserved flag bits set: either a future format this decoder does
		// not understand, or hostile input. Either way, silently accepting
		// and then re-encoding would drop those bits, breaking the
		// Encode(DecodeSchema(b)) == b identity — reject instead.
		return nil, 0, ErrOperateSchema
	}
	s.StoreNames = flags&schemaFlagStoreNames != 0
	off := 3

	nFields, m, err := decodeCanonicalUvarint(b[off:])
	if err != nil {
		return nil, 0, err
	}
	off += m
	// budget is what is left of OperateMaxSchemaBytes, and it is spent by
	// every declaration in the blob rather than reset per declaration.
	//
	// CountFitsIn bounds a count against the bytes actually left, but b is
	// the whole STORED record — up to maxRecordBytes, 16 MiB — not just the
	// schema blob, and the 4 KiB blob cap is only checked by Validate, after
	// the whole thing has been decoded. A per-declaration bound is therefore
	// not a bound at all in aggregate: every one of thousands of table fields
	// can declare thousands of columns, each passing its own check, while the
	// sum sizes hundreds of megabytes of ColumnDef before anything says no.
	//
	// One running budget closes that. Every field costs at least its type
	// byte, every column at least its own, and the names section costs its
	// declared length, so a schema whose declarations sum past the blob cap
	// cannot possibly encode within it — and the budget is spent BEFORE each
	// make, so an over-declaration is rejected rather than allocated for. It
	// never rejects a legal schema: len(Encode()) >= nFields + Σ nCols +
	// nameBytes, and Validate already requires len(Encode()) <= the cap.
	budget := uint64(OperateMaxSchemaBytes)
	if nFields > OperateMaxFields || nFields > budget {
		return nil, 0, ErrOperateSchema
	}
	budget -= nFields
	if !CountFitsIn(int(nFields), len(b)-off, 1) {
		return nil, 0, ErrShortArgs
	}

	fields := make([]FieldDef, nFields)
	for i := range fields {
		if len(b)-off < 1 {
			return nil, 0, ErrShortArgs
		}
		typ := b[off]
		off++
		fields[i].Type = typ
		if typ == OperateTypeFixed {
			if len(b)-off < 1 {
				return nil, 0, ErrShortArgs
			}
			fields[i].N = b[off]
			off++
		}
		if typ != OperateTypeTable {
			continue
		}

		td := &TableDef{}
		if len(b)-off < 1 {
			return nil, 0, ErrShortArgs
		}
		td.KeyType = b[off]
		off++
		if td.KeyType == OperateTypeFixed {
			if len(b)-off < 1 {
				return nil, 0, ErrShortArgs
			}
			td.KeyN = b[off]
			off++
		}

		nCols, m2, err := decodeCanonicalUvarint(b[off:])
		if err != nil {
			return nil, 0, err
		}
		off += m2
		// Charged against the same running budget the fields were, so the
		// columns of every table in the blob are bounded in aggregate and
		// not merely one table at a time.
		if nCols > OperateMaxCols || nCols > budget {
			return nil, 0, ErrOperateSchema
		}
		budget -= nCols
		if !CountFitsIn(int(nCols), len(b)-off, 1) {
			return nil, 0, ErrShortArgs
		}

		cols := make([]ColumnDef, nCols)
		for j := range cols {
			if len(b)-off < 1 {
				return nil, 0, ErrShortArgs
			}
			ctyp := b[off]
			off++
			cols[j].Type = ctyp
			if ctyp == OperateTypeFixed {
				if len(b)-off < 1 {
					return nil, 0, ErrShortArgs
				}
				cols[j].N = b[off]
				off++
			}
		}
		td.Cols = cols

		capV, m3, err := decodeCanonicalUvarint(b[off:])
		if err != nil {
			return nil, 0, err
		}
		off += m3
		if capV > OperateMaxRows {
			return nil, 0, ErrOperateSchema
		}
		td.Cap = uint32(capV) //nolint:gosec // bounded by OperateMaxRows above

		if len(b)-off < 1 {
			return nil, 0, ErrShortArgs
		}
		td.Policy = b[off]
		off++

		byColV, m4, err := decodeCanonicalUvarint(b[off:])
		if err != nil {
			return nil, 0, err
		}
		off += m4
		if byColV > OperateMaxCols {
			return nil, 0, ErrOperateSchema
		}
		td.ByCol = uint16(byColV) //nolint:gosec // bounded by OperateMaxCols above

		fields[i].Table = td
	}
	s.Fields = fields

	nameBytes, m5, err := decodeCanonicalUvarint(b[off:])
	if err != nil {
		return nil, 0, err
	}
	off += m5
	// The last charge against the budget: the names section is the rest of
	// what a blob spends its 4 KiB on.
	if nameBytes > OperateMaxNameBytes || nameBytes > budget {
		return nil, 0, ErrOperateSchema
	}
	if !CountFitsIn(int(nameBytes), len(b)-off, 1) {
		return nil, 0, ErrShortArgs
	}

	if !s.StoreNames {
		if nameBytes != 0 {
			return nil, 0, ErrOperateSchema
		}
	} else {
		end := off + int(nameBytes)
		buf := b[off:end]
		pos := 0
		readName := func() (string, error) {
			if len(buf)-pos < 1 {
				return "", ErrShortArgs
			}
			ln := int(buf[pos])
			pos++
			if ln > OperateMaxNameLen {
				return "", ErrOperateSchema
			}
			if len(buf)-pos < ln {
				return "", ErrShortArgs
			}
			name := string(buf[pos : pos+ln])
			pos += ln
			return name, nil
		}
		for i := range s.Fields {
			name, err := readName()
			if err != nil {
				return nil, 0, err
			}
			s.Fields[i].Name = name
		}
		for i := range s.Fields {
			f := &s.Fields[i]
			if f.Type == OperateTypeTable && f.Table != nil {
				for j := range f.Table.Cols {
					name, err := readName()
					if err != nil {
						return nil, 0, err
					}
					f.Table.Cols[j].Name = name
				}
			}
		}
		if pos != len(buf) {
			return nil, 0, ErrOperateSchema
		}
		off = end
	}

	if err := s.Validate(); err != nil {
		return nil, 0, err
	}
	return s, off, nil
}

// validOperateKeyType reports whether t is a legal table row-key type
// (design doc §2.3): one of the unsigned fixed-width integer types or
// OperateTypeFixed. Signed integers, floats, and every variable-length type
// are not legal keys.
func validOperateKeyType(t uint8) bool {
	switch t {
	case OperateTypeU8, OperateTypeU16, OperateTypeU32, OperateTypeU64, OperateTypeFixed:
		return true
	default:
		return false
	}
}

// Validate checks s against every structural rule the design imposes
// (§2.2/§2.3/§2.7/§2.8): known, non-UNSET types; FIXED widths >= 1; table
// columns fixed-width only; a legal key type; row width, field/column
// counts, cap, and total encoded size within the hard caps (§2.7); *_COL
// eviction policies pointing at a real column; and unique names among
// fields, and among one table's own columns, when names are present. A
// schema that fails Validate must never be stored or acted on.
func (s *Schema) Validate() error {
	if len(s.Fields) > OperateMaxFields {
		return ErrOperateSchema
	}

	fieldNames := make(map[string]struct{}, len(s.Fields))
	for i := range s.Fields {
		f := &s.Fields[i]
		if f.Type >= OperateTypeCount || f.Type == OperateTypeUnset {
			return ErrOperateSchema
		}
		if f.Type == OperateTypeFixed && f.N < 1 {
			return ErrOperateSchema
		}
		if f.Name != "" {
			if len(f.Name) > OperateMaxNameLen {
				return ErrOperateSchema
			}
			if _, dup := fieldNames[f.Name]; dup {
				return ErrOperateSchema
			}
			fieldNames[f.Name] = struct{}{}
		}

		if f.Type != OperateTypeTable {
			continue
		}
		if f.Table == nil {
			return ErrOperateSchema
		}
		t := f.Table
		if !validOperateKeyType(t.KeyType) {
			return ErrOperateSchema
		}
		if t.KeyType == OperateTypeFixed && t.KeyN < 1 {
			return ErrOperateSchema
		}
		if len(t.Cols) > OperateMaxCols {
			return ErrOperateSchema
		}
		if t.Cap > OperateMaxRows {
			return ErrOperateSchema
		}
		if t.Policy > OperatePolicyMaxCol {
			return ErrOperateSchema
		}
		if (t.Policy == OperatePolicyMinCol || t.Policy == OperatePolicyMaxCol) && int(t.ByCol) >= len(t.Cols) {
			return ErrOperateSchema
		}

		rowWidth := CellWidth(t.KeyType, t.KeyN)
		if rowWidth < 1 {
			return ErrOperateSchema
		}
		colNames := make(map[string]struct{}, len(t.Cols))
		for _, c := range t.Cols {
			if c.Type >= OperateTypeCount {
				return ErrOperateSchema
			}
			w := CellWidth(c.Type, c.N)
			if w < 1 {
				return ErrOperateSchema
			}
			rowWidth += w
			if c.Name != "" {
				if len(c.Name) > OperateMaxNameLen {
					return ErrOperateSchema
				}
				if _, dup := colNames[c.Name]; dup {
					return ErrOperateSchema
				}
				colNames[c.Name] = struct{}{}
			}
		}
		if rowWidth > OperateMaxRowWidth {
			return ErrOperateSchema
		}
	}

	if len(s.Encode()) > OperateMaxSchemaBytes {
		return ErrOperateSchema
	}
	return nil
}

// FieldPos returns the schema position of the field named name, and whether
// it was found. It is a client-side convenience (design doc §2.2): the wire
// never carries names, only positions.
func (s *Schema) FieldPos(name string) (int, bool) {
	if name == "" {
		return 0, false
	}
	for i := range s.Fields {
		if s.Fields[i].Name == name {
			return i, true
		}
	}
	return 0, false
}

// Layout computes s's byte layout (design doc §2.2/§2.3): fixed-width
// fields packed first in schema order, then the variable-length tail. It
// returns an error rather than panicking if s contains a structural
// impossibility (a table field with a nil Table, or a key/column whose
// width cannot be determined) — callers normally call this only on an
// already-Validate'd schema, for which it cannot fail.
func (s *Schema) Layout() (*Layout, error) {
	l := &Layout{
		FixedOff: make([]int, len(s.Fields)),
		Tables:   make([]*TableLayout, len(s.Fields)),
	}
	off := 0
	for i := range s.Fields {
		f := &s.Fields[i]
		if f.Type == OperateTypeFixed && f.N < 1 {
			return nil, ErrOperateSchema
		}
		w := CellWidth(f.Type, f.N)
		if w < 0 {
			l.FixedOff[i] = -1
			l.VarTail = append(l.VarTail, i)
			if f.Type == OperateTypeTable {
				if f.Table == nil {
					return nil, ErrOperateSchema
				}
				t := f.Table
				kw := CellWidth(t.KeyType, t.KeyN)
				if kw < 1 {
					return nil, ErrOperateSchema
				}
				colOff := make([]int, len(t.Cols))
				rowWidth := kw
				for c := range t.Cols {
					cw := CellWidth(t.Cols[c].Type, t.Cols[c].N)
					if cw < 1 {
						return nil, ErrOperateSchema
					}
					colOff[c] = rowWidth
					rowWidth += cw
				}
				l.Tables[i] = &TableLayout{KeyWidth: kw, RowWidth: rowWidth, ColOff: colOff}
			}
			continue
		}
		l.FixedOff[i] = off
		off += w
	}
	l.FixedLen = off
	return l, nil
}

// Extends reports whether newer is a valid append-only evolution of s
// (design doc §2.8): a strictly greater version; every existing field kept
// at its position with an unchanged type and, for OperateTypeFixed, an
// unchanged width — N is compared only where it means something, since it is
// ignored (and not even encoded) for every other type (a table's key
// type/width unchanged and its existing columns kept as an unchanged-type
// prefix, while
// Cap/Policy/ByCol are free to change); new fields and columns may be
// appended; a name already stored may not change (though a field/column that
// had no name may gain one). Anything else is ErrOperateSchema.
func (s *Schema) Extends(newer *Schema) error {
	if newer.Version <= s.Version {
		return ErrOperateSchema
	}
	if len(newer.Fields) < len(s.Fields) {
		return ErrOperateSchema
	}
	for i := range s.Fields {
		of := &s.Fields[i]
		nf := &newer.Fields[i]
		// N is the FIXED(n) width and is ignored for every other type
		// (FieldDef.N's own doc says so), so comparing it unconditionally
		// would refuse an otherwise identical schema over a byte nothing
		// reads — Encode does not even write it.
		if nf.Type != of.Type || (of.Type == OperateTypeFixed && nf.N != of.N) {
			return ErrOperateSchema
		}
		if of.Name != "" && nf.Name != of.Name {
			return ErrOperateSchema
		}
		if of.Type != OperateTypeTable {
			continue
		}
		if of.Table == nil || nf.Table == nil {
			return ErrOperateSchema
		}
		ot, nt := of.Table, nf.Table
		// KeyN, like N, is the FIXED key's width and ignored otherwise.
		if ot.KeyType != nt.KeyType || (ot.KeyType == OperateTypeFixed && ot.KeyN != nt.KeyN) {
			return ErrOperateSchema
		}
		if len(nt.Cols) < len(ot.Cols) {
			return ErrOperateSchema
		}
		for c := range ot.Cols {
			oc, nc := &ot.Cols[c], &nt.Cols[c]
			if nc.Type != oc.Type || (oc.Type == OperateTypeFixed && nc.N != oc.N) {
				return ErrOperateSchema
			}
			if oc.Name != "" && nc.Name != oc.Name {
				return ErrOperateSchema
			}
		}
		// Cap, Policy, and ByCol may change freely.
	}
	return nil
}
