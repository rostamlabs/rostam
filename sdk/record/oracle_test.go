// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// This file holds the ORACLE the byte-level resolver is held to: a second,
// deliberately naive implementation of the same semantics over the decoded
// record tree (wire.DecodeRecord), plus the random record and path
// generators TestResolveMatchesOracle drives both with. The oracle walks
// slices and maps; resolve.go walks stored bytes with offset arithmetic.
// They must agree on every well-formed record.

// resolveTree resolves p against the decoded record tree rec by the same
// rules resolve.go implements over stored bytes. It is the reference
// implementation, written for obviousness rather than speed.
func resolveTree(rec *wire.Record, p Path) (Result, error) {
	if len(p.Segs) == 0 || len(p.Segs) > 3 {
		return Result{}, fmt.Errorf("%w: path has %d segments", ErrPath, len(p.Segs))
	}
	switch rec.Mode {
	case wire.OperateModeSchema:
		return resolveTreeSchema(rec, p)
	case wire.OperateModeDynamic:
		return resolveTreeDynamic(rec, p)
	default:
		return Result{}, fmt.Errorf("%w: unknown mode byte %d", ErrRecord, rec.Mode)
	}
}

func resolveTreeSchema(rec *wire.Record, p Path) (Result, error) {
	s := rec.Schema
	seg := p.Segs[0]

	var pos int
	if seg.ByPos {
		if int(seg.Pos) >= len(s.Fields) {
			return Result{Kind: Absent}, nil
		}
		pos = int(seg.Pos)
	} else {
		if !s.StoreNames {
			return Result{}, fmt.Errorf("%w: schema stores no names", ErrPath)
		}
		i, ok := s.FieldPos(seg.Name)
		if !ok {
			return Result{Kind: Absent}, nil
		}
		pos = i
	}

	fdef := &s.Fields[pos]
	fld := &rec.Fields[pos]
	isTable := fdef.Type == wire.OperateTypeTable

	if seg.Kind == SegCount {
		if !isTable {
			return Result{}, fmt.Errorf("%w: #count on a non-table field", ErrPath)
		}
		return Result{Kind: Count, Count: uint64(len(fld.Table.Rows))}, nil
	}
	if len(p.Segs) == 1 {
		if isTable {
			return Result{Kind: Table}, nil
		}
		return Result{Kind: Scalar, Cell: fld.Cell}, nil
	}
	if !isTable {
		return Result{}, fmt.Errorf("%w: row segment on a non-table field", ErrPath)
	}

	td := fdef.Table
	kw := wire.CellWidth(td.KeyType, td.KeyN)
	if kw < 1 {
		return Result{}, fmt.Errorf("%w: table key has no width", ErrRecord)
	}
	key, ok := treeTargetKey(p.Segs[1], kw)
	if !ok {
		return Result{Kind: Absent}, nil
	}
	var row *wire.Row
	for i := range fld.Table.Rows {
		if bytes.Equal(fld.Table.Rows[i].Key, key) {
			row = &fld.Table.Rows[i]
			break
		}
	}
	if row == nil {
		return Result{Kind: Absent}, nil
	}
	if len(p.Segs) == 2 {
		return Result{Kind: RowPresent}, nil
	}

	cseg := p.Segs[2]
	var ci int
	if cseg.ByPos {
		if int(cseg.Pos) >= len(td.Cols) {
			return Result{Kind: Absent}, nil
		}
		ci = int(cseg.Pos)
	} else {
		if !s.StoreNames {
			return Result{}, fmt.Errorf("%w: schema stores no names", ErrPath)
		}
		found := -1
		for i := range td.Cols {
			if td.Cols[i].Name == cseg.Name {
				found = i
				break
			}
		}
		if found < 0 {
			return Result{Kind: Absent}, nil
		}
		ci = found
	}
	if ci >= len(row.Cols) {
		return Result{Kind: Absent}, nil
	}
	return Result{Kind: Scalar, Cell: row.Cols[ci].Cell}, nil
}

func resolveTreeDynamic(rec *wire.Record, p Path) (Result, error) {
	seg := p.Segs[0]
	if seg.ByPos {
		return Result{}, fmt.Errorf("%w: dynamic mode has no field positions", ErrPath)
	}
	idx := -1
	for i := range rec.Fields {
		if rec.Fields[i].Name == seg.Name {
			idx = i
			break
		}
	}
	if idx < 0 {
		return Result{Kind: Absent}, nil
	}
	fld := &rec.Fields[idx]
	isTable := fld.Cell.Type == wire.OperateTypeTable

	if seg.Kind == SegCount {
		if !isTable {
			return Result{}, fmt.Errorf("%w: #count on a non-table field", ErrPath)
		}
		return Result{Kind: Count, Count: uint64(len(fld.Table.Rows))}, nil
	}
	if len(p.Segs) == 1 {
		if isTable {
			return Result{Kind: Table}, nil
		}
		return Result{Kind: Scalar, Cell: fld.Cell}, nil
	}
	if !isTable {
		return Result{}, fmt.Errorf("%w: row segment on a non-table field", ErrPath)
	}

	key, ok := treeDynamicKey(p.Segs[1])
	if !ok {
		return Result{Kind: Absent}, nil
	}
	var row *wire.Row
	for i := range fld.Table.Rows {
		if bytes.Equal(fld.Table.Rows[i].Key, key) {
			row = &fld.Table.Rows[i]
			break
		}
	}
	if row == nil {
		return Result{Kind: Absent}, nil
	}
	if len(p.Segs) == 2 {
		return Result{Kind: RowPresent}, nil
	}

	cseg := p.Segs[2]
	if cseg.ByPos {
		return Result{}, fmt.Errorf("%w: dynamic mode has no column positions", ErrPath)
	}
	for i := range row.Cols {
		if row.Cols[i].Name == cseg.Name {
			return Result{Kind: Scalar, Cell: row.Cols[i].Cell}, nil
		}
	}
	return Result{Kind: Absent}, nil
}

// treeTargetKey builds the schema-mode row key a row segment denotes at key
// width kw, or reports that no row of that width can match.
func treeTargetKey(seg Segment, kw int) ([]byte, bool) {
	if seg.KeyQuoted {
		if len(seg.Key) != kw {
			return nil, false
		}
		return seg.Key, true
	}
	v, err := strconv.ParseUint(seg.KeyText, 10, 64)
	if err != nil {
		return nil, false
	}
	return leKeyBytes(v, kw)
}

// treeDynamicKey builds the dynamic-mode row key a row segment denotes: the
// raw bytes of a quoted segment, or the 8-byte little-endian encoding of a
// decimal one.
func treeDynamicKey(seg Segment) ([]byte, bool) {
	if seg.KeyQuoted {
		return seg.Key, true
	}
	v, err := strconv.ParseUint(seg.KeyText, 10, 64)
	if err != nil {
		return nil, false
	}
	return leKeyBytes(v, 8)
}

// leKeyBytes encodes v as width little-endian bytes, reporting false when v
// does not fit in that width.
func leKeyBytes(v uint64, width int) ([]byte, bool) {
	if width < 1 {
		return nil, false
	}
	if width < 8 && v >= uint64(1)<<(8*uint(width)) {
		return nil, false
	}
	b := make([]byte, width)
	for i := 0; i < width && i < 8; i++ {
		b[i] = byte(v >> (8 * uint(i)))
	}
	return b, true
}

// leValue decodes at most 8 little-endian bytes as an unsigned integer.
func leValue(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// --- generators -----------------------------------------------------------

var (
	// schemaNamePool supplies unique, path-legal field names; a schema never
	// has more fields than this pool has entries.
	schemaNamePool = []string{"aa", "bb", "cc", "dd", "ee", "ff"}
	// colNamePool supplies unique, path-legal column names.
	colNamePool = []string{"c0", "c1", "c2", "c3"}
	// dynNamePool supplies dynamic-mode field names.
	dynNamePool = []string{"a", "bb", "ccc", "dd", "ee", "ff"}
	// keyAlphabet is the byte alphabet FIXED and dynamic row keys are drawn
	// from: path-safe (no '/', which ParsePath splits on before unquoting)
	// but including the two bytes a quoted segment must escape.
	keyAlphabet = []byte(`abcXY019"\`)

	// scalarFieldTypes are the record-field types a generated schema draws
	// from (TABLE is added separately; UNSET is not legal in a schema).
	scalarFieldTypes = []uint8{
		wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64,
		wire.OperateTypeI8, wire.OperateTypeI16, wire.OperateTypeI32, wire.OperateTypeI64,
		wire.OperateTypeF32, wire.OperateTypeF64,
		wire.OperateTypeUVarint, wire.OperateTypeIVarint,
		wire.OperateTypeBytes, wire.OperateTypeFixed,
	}
	// fixedWidthTypes are the types legal as a table column (fixed width only).
	fixedWidthTypes = []uint8{
		wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64,
		wire.OperateTypeI8, wire.OperateTypeI16, wire.OperateTypeI32, wire.OperateTypeI64,
		wire.OperateTypeF32, wire.OperateTypeF64, wire.OperateTypeFixed,
	}
	// keyTypes are the legal table row-key types.
	keyTypes = []uint8{
		wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32,
		wire.OperateTypeU64, wire.OperateTypeFixed,
	}
	// dynFieldTypes are the dynamic-mode scalar field types (UNSET included:
	// a dynamic field may legally carry no bytes at all).
	dynFieldTypes = []uint8{
		wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32, wire.OperateTypeU64,
		wire.OperateTypeI8, wire.OperateTypeI16, wire.OperateTypeI32, wire.OperateTypeI64,
		wire.OperateTypeF32, wire.OperateTypeF64,
		wire.OperateTypeUVarint, wire.OperateTypeIVarint,
		wire.OperateTypeBytes, wire.OperateTypeFixed, wire.OperateTypeUnset,
	}
)

// randomSchemaRecord builds a random schema-mode record, encodes it, and
// returns the stored bytes alongside the tree DecodeRecord reads back from
// them. The DECODED tree — not the tree that was built — is the oracle's
// input, so the oracle and the resolver see exactly the same record.
func randomSchemaRecord(rng *rand.Rand) ([]byte, *wire.Record, error) {
	s := &wire.Schema{
		Version:    uint16(rng.Intn(4)),
		StoreNames: rng.Intn(2) == 0,
	}
	n := 2 + rng.Intn(5) // 2..6
	tables := 0
	for i := 0; i < n; i++ {
		fd := wire.FieldDef{Name: schemaNamePool[i]}
		typ := scalarFieldTypes[rng.Intn(len(scalarFieldTypes))]
		if tables < 2 && rng.Intn(3) == 0 {
			typ = wire.OperateTypeTable
		}
		fd.Type = typ
		switch typ {
		case wire.OperateTypeFixed:
			fd.N = uint8(1 + rng.Intn(4))
		case wire.OperateTypeTable:
			tables++
			fd.Table = randomTableDef(rng)
		}
		s.Fields = append(s.Fields, fd)
	}
	if err := s.Validate(); err != nil {
		return nil, nil, fmt.Errorf("generated schema invalid: %w", err)
	}

	rec := &wire.Record{Mode: wire.OperateModeSchema, Schema: s}
	for i := range s.Fields {
		f := &s.Fields[i]
		if f.Type == wire.OperateTypeTable {
			rec.Fields = append(rec.Fields, wire.Field{
				Name:  f.Name,
				Cell:  wire.Cell{Type: wire.OperateTypeTable},
				Table: randomSchemaTable(rng, f.Table),
			})
			continue
		}
		rec.Fields = append(rec.Fields, wire.Field{Name: f.Name, Cell: randomCell(rng, f.Type, f.N)})
	}

	enc := rec.Encode()
	if enc == nil {
		return nil, nil, errors.New("generated record failed to encode")
	}
	dec, err := wire.DecodeRecord(enc)
	if err != nil {
		return nil, nil, fmt.Errorf("generated record failed to decode: %w", err)
	}
	return enc, dec, nil
}

func randomTableDef(rng *rand.Rand) *wire.TableDef {
	td := &wire.TableDef{KeyType: keyTypes[rng.Intn(len(keyTypes))]}
	if td.KeyType == wire.OperateTypeFixed {
		td.KeyN = uint8(1 + rng.Intn(4))
	}
	nCols := 1 + rng.Intn(3)
	for c := 0; c < nCols; c++ {
		cd := wire.ColumnDef{Name: colNamePool[c], Type: fixedWidthTypes[rng.Intn(len(fixedWidthTypes))]}
		if cd.Type == wire.OperateTypeFixed {
			cd.N = uint8(1 + rng.Intn(3))
		}
		td.Cols = append(td.Cols, cd)
	}
	return td
}

func randomSchemaTable(rng *rand.Rand, td *wire.TableDef) *wire.Table {
	kw := wire.CellWidth(td.KeyType, td.KeyN)
	nRows := rng.Intn(6)
	t := &wire.Table{}
	seen := make(map[string]bool, nRows)
	for attempts := 0; len(t.Rows) < nRows && attempts < 100; attempts++ {
		var key []byte
		if td.KeyType == wire.OperateTypeFixed {
			key = randomKeyBytes(rng, kw)
		} else {
			v := uint64(rng.Intn(60))
			key, _ = leKeyBytes(v, kw)
		}
		if seen[string(key)] {
			continue
		}
		seen[string(key)] = true
		row := wire.Row{Key: key}
		for c := range td.Cols {
			row.Cols = append(row.Cols, wire.Col{Cell: randomCell(rng, td.Cols[c].Type, td.Cols[c].N)})
		}
		t.Rows = append(t.Rows, row)
	}
	return t
}

// randomKeyBytes draws n bytes from keyAlphabet.
func randomKeyBytes(rng *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = keyAlphabet[rng.Intn(len(keyAlphabet))]
	}
	return b
}

func randomCell(rng *rand.Rand, typ, n uint8) wire.Cell {
	c := wire.Cell{Type: typ, N: n}
	switch typ {
	case wire.OperateTypeU8:
		c.U = uint64(rng.Intn(256))
	case wire.OperateTypeU16:
		c.U = uint64(rng.Intn(65536))
	case wire.OperateTypeU32:
		c.U = uint64(rng.Uint32())
	case wire.OperateTypeU64, wire.OperateTypeUVarint:
		c.U = rng.Uint64() >> uint(rng.Intn(64))
	case wire.OperateTypeI8:
		c.U = su64(int64(int8(rng.Intn(256))))
	case wire.OperateTypeI16:
		c.U = su64(int64(int16(rng.Intn(65536))))
	case wire.OperateTypeI32:
		c.U = su64(int64(int32(rng.Uint32())))
	case wire.OperateTypeI64, wire.OperateTypeIVarint:
		c.U = su64(int64(rng.Uint64()) >> uint(rng.Intn(64)))
	case wire.OperateTypeF32:
		c.F = float64(float32(rng.NormFloat64()))
	case wire.OperateTypeF64:
		c.F = rng.NormFloat64()
	case wire.OperateTypeBytes:
		c.B = randomKeyBytes(rng, rng.Intn(6))
	case wire.OperateTypeFixed:
		c.B = randomKeyBytes(rng, int(n))
	}
	return c
}

// randomDynamicRecord builds a random dynamic-mode record, encodes it, and
// returns the stored bytes alongside the decoded tree.
func randomDynamicRecord(rng *rand.Rand) ([]byte, *wire.Record, error) {
	rec := &wire.Record{Mode: wire.OperateModeDynamic}
	n := 1 + rng.Intn(5)
	used := make(map[string]bool, n)
	for i := 0; i < n; i++ {
		name := dynNamePool[rng.Intn(len(dynNamePool))]
		if used[name] {
			continue
		}
		used[name] = true
		if rng.Intn(3) == 0 {
			rec.Fields = append(rec.Fields, wire.Field{
				Name:  name,
				Cell:  wire.Cell{Type: wire.OperateTypeTable},
				Table: randomDynamicTable(rng),
			})
			continue
		}
		typ := dynFieldTypes[rng.Intn(len(dynFieldTypes))]
		var w uint8
		if typ == wire.OperateTypeFixed {
			w = uint8(1 + rng.Intn(4))
		}
		rec.Fields = append(rec.Fields, wire.Field{Name: name, Cell: randomCell(rng, typ, w)})
	}
	if len(rec.Fields) == 0 {
		rec.Fields = append(rec.Fields, wire.Field{Name: "a", Cell: randomCell(rng, wire.OperateTypeU8, 0)})
	}

	enc := rec.Encode()
	if enc == nil {
		return nil, nil, errors.New("generated record failed to encode")
	}
	dec, err := wire.DecodeRecord(enc)
	if err != nil {
		return nil, nil, fmt.Errorf("generated record failed to decode: %w", err)
	}
	return enc, dec, nil
}

func randomDynamicTable(rng *rand.Rand) *wire.Table {
	t := &wire.Table{Cap: uint32(rng.Intn(8))}
	if rng.Intn(4) == 0 {
		t.Policy = wire.OperatePolicyMinCol
		t.ByColName = colNamePool[0]
	} else {
		t.Policy = uint8(rng.Intn(3)) // NONE, MIN_KEY, MAX_KEY
	}
	nRows := rng.Intn(5)
	seen := make(map[string]bool, nRows)
	for attempts := 0; len(t.Rows) < nRows && attempts < 100; attempts++ {
		var key []byte
		if rng.Intn(2) == 0 {
			key, _ = leKeyBytes(uint64(rng.Intn(60)), 8)
		} else {
			key = randomKeyBytes(rng, 1+rng.Intn(4))
		}
		if seen[string(key)] {
			continue
		}
		seen[string(key)] = true

		row := wire.Row{Key: key}
		nCols := rng.Intn(4)
		usedCols := make(map[string]bool, nCols)
		for c := 0; c < nCols; c++ {
			name := colNamePool[rng.Intn(len(colNamePool))]
			if usedCols[name] {
				continue
			}
			usedCols[name] = true
			typ := dynFieldTypes[rng.Intn(len(dynFieldTypes))]
			var w uint8
			if typ == wire.OperateTypeFixed {
				w = uint8(1 + rng.Intn(3))
			}
			row.Cols = append(row.Cols, wire.Col{Name: name, Cell: randomCell(rng, typ, w)})
		}
		t.Rows = append(t.Rows, row)
	}
	return t
}

// randomPath draws a path string against rec: a mix of paths that resolve,
// paths that are near misses (an absent name, an absent key, an
// out-of-range position), and paths of the wrong shape for the field they
// name (a row segment on a scalar, "#count" on a scalar). The caller runs
// it through ParsePath; a string the grammar rejects is skipped.
func randomPath(rng *rand.Rand, rec *wire.Record) string {
	if rec.Mode == wire.OperateModeSchema {
		return randomSchemaPath(rng, rec)
	}
	return randomDynamicPath(rng, rec)
}

func randomSchemaPath(rng *rand.Rand, rec *wire.Record) string {
	s := rec.Schema
	i := rng.Intn(len(s.Fields) + 1) // the last draw is out of range

	var head string
	switch {
	case rng.Intn(2) == 0:
		head = "#" + strconv.Itoa(i)
	case rng.Intn(8) == 0:
		head = "zz" // a name no generated schema uses
	case i < len(s.Fields) && s.Fields[i].Name != "":
		head = s.Fields[i].Name
	default:
		head = schemaNamePool[rng.Intn(len(schemaNamePool))]
	}

	switch rng.Intn(6) {
	case 0:
		return head + countSuffix
	case 1, 2:
		return head
	}

	var td *wire.TableDef
	var tbl *wire.Table
	if i < len(s.Fields) && s.Fields[i].Type == wire.OperateTypeTable {
		td = s.Fields[i].Table
		tbl = rec.Fields[i].Table
	}
	path := head + "/" + randomKeyText(rng, td, tbl)
	if rng.Intn(2) == 0 {
		return path
	}
	return path + "/" + randomColText(rng, td)
}

// randomKeyText renders a row-key segment: a real key of the table (when
// there is one) half the time, a near miss otherwise.
func randomKeyText(rng *rand.Rand, td *wire.TableDef, tbl *wire.Table) string {
	if td != nil && tbl != nil && len(tbl.Rows) > 0 && rng.Intn(2) == 0 {
		key := tbl.Rows[rng.Intn(len(tbl.Rows))].Key
		if td.KeyType == wire.OperateTypeFixed {
			return quoteKeyText(key)
		}
		return strconv.FormatUint(leValue(key), 10)
	}
	if rng.Intn(2) == 0 {
		return strconv.FormatUint(uint64(rng.Intn(70)), 10)
	}
	return quoteKeyText(randomKeyBytes(rng, 1+rng.Intn(4)))
}

func randomColText(rng *rand.Rand, td *wire.TableDef) string {
	if td != nil && len(td.Cols) > 0 && rng.Intn(2) == 0 {
		c := rng.Intn(len(td.Cols))
		if td.Cols[c].Name != "" && rng.Intn(2) == 0 {
			return td.Cols[c].Name
		}
		return "#" + strconv.Itoa(c)
	}
	if rng.Intn(2) == 0 {
		return "#" + strconv.Itoa(rng.Intn(5))
	}
	return colNamePool[rng.Intn(len(colNamePool))]
}

func randomDynamicPath(rng *rand.Rand, rec *wire.Record) string {
	var head string
	i := rng.Intn(len(rec.Fields) + 1)
	switch {
	case rng.Intn(8) == 0:
		head = "#" + strconv.Itoa(i) // positions never apply in dynamic mode
	case i < len(rec.Fields):
		head = rec.Fields[i].Name
	default:
		head = dynNamePool[rng.Intn(len(dynNamePool))]
	}

	switch rng.Intn(6) {
	case 0:
		return head + countSuffix
	case 1, 2:
		return head
	}

	var tbl *wire.Table
	if i < len(rec.Fields) {
		tbl = rec.Fields[i].Table
	}
	path := head + "/" + randomDynamicKeyText(rng, tbl)
	if rng.Intn(2) == 0 {
		return path
	}
	if rng.Intn(8) == 0 {
		return path + "/#0" // a column position never applies in dynamic mode
	}
	return path + "/" + colNamePool[rng.Intn(len(colNamePool))]
}

func randomDynamicKeyText(rng *rand.Rand, tbl *wire.Table) string {
	if tbl != nil && len(tbl.Rows) > 0 && rng.Intn(2) == 0 {
		key := tbl.Rows[rng.Intn(len(tbl.Rows))].Key
		if len(key) == 8 {
			return strconv.FormatUint(leValue(key), 10)
		}
		return quoteKeyText(key)
	}
	if rng.Intn(2) == 0 {
		return strconv.FormatUint(uint64(rng.Intn(70)), 10)
	}
	return quoteKeyText(randomKeyBytes(rng, 1+rng.Intn(4)))
}

// quoteKeyText renders b as a quoted row-key segment, escaping the two
// bytes the grammar escapes.
func quoteKeyText(b []byte) string {
	var sb strings.Builder
	sb.WriteByte('"')
	for _, c := range b {
		if c == '"' || c == '\\' {
			sb.WriteByte('\\')
		}
		sb.WriteByte(c)
	}
	sb.WriteByte('"')
	return sb.String()
}

// TestOracleGeneratorsAreWellFormed checks the generators themselves: every
// record they emit must round-trip through DecodeRecord (which is what
// makes the oracle's tree the authority on the stored bytes), and a decent
// share of the paths they draw must parse.
func TestOracleGeneratorsAreWellFormed(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	parsed, total := 0, 0
	for i := 0; i < 200; i++ {
		var enc []byte
		var dec *wire.Record
		var err error
		if i%2 == 0 {
			enc, dec, err = randomSchemaRecord(rng)
		} else {
			enc, dec, err = randomDynamicRecord(rng)
		}
		if err != nil {
			t.Fatalf("generator: %v", err)
		}
		if !bytes.Equal(dec.Encode(), enc) {
			t.Fatalf("record %d does not re-encode identically", i)
		}
		for j := 0; j < 10; j++ {
			total++
			if _, perr := ParsePath(randomPath(rng, dec)); perr == nil {
				parsed++
			}
		}
	}
	if parsed*2 < total {
		t.Fatalf("only %d of %d generated paths parse; the generator is drifting from the grammar", parsed, total)
	}
}

// TestLEKeyBytesRoundTrip checks the two helpers the oracle builds and
// reads row keys with agree, and that the 8-byte form is exactly what
// client.KeyU64 writes — the encoding a decimal row key denotes.
func TestLEKeyBytesRoundTrip(t *testing.T) {
	for _, v := range []uint64{0, 1, 255, 256, 65535, 1 << 32, ^uint64(0)} {
		for _, w := range []int{1, 2, 4, 8} {
			b, ok := leKeyBytes(v, w)
			if !ok {
				continue
			}
			if got := leValue(b); got != v {
				t.Fatalf("leKeyBytes(%d, %d) round-tripped to %d", v, w, got)
			}
			if w == 8 {
				var want [8]byte
				binary.LittleEndian.PutUint64(want[:], v)
				if !bytes.Equal(b, want[:]) {
					t.Fatalf("leKeyBytes(%d, 8) = % x, want % x", v, b, want)
				}
			}
		}
	}
}
