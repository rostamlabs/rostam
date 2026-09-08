// SPDX-License-Identifier: Apache-2.0

package record

import (
	"bytes"
	"fmt"
	"math"
	"strconv"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// CellValue maps a decoded wire.Cell to the payload-shaped vtypes.Value the
// filter compilers compare against, per the value mapping ruling:
//
//   - the eight fixed-width int types and IVARINT already carry the value's
//     int64 bit pattern in Cell.U (design doc §2.1: sign-extended for a
//     signed type), so they map straight to ValueInt.
//   - U64 and UVARINT can legitimately hold a magnitude above what an int64
//     can represent; when Cell.U exceeds math.MaxInt64 the value maps to
//     ValueFloat(float64(u)) instead — a lossy but total conversion, since
//     there is no wider signed payload kind.
//   - F32/F64 map to ValueFloat (F32 is already widened to float64 in Cell.F
//     by DecodeCellData).
//   - BYTES/FIXED map to ValueString, the raw bytes reinterpreted as a Go
//     string (a copy, since a string conversion of a []byte always copies).
//   - UNSET, TABLE, and any tag this package does not know map to
//     (Value{}, false): neither carries a value a filter can compare.
func CellValue(c wire.Cell) (vtypes.Value, bool) {
	switch c.Type {
	case wire.OperateTypeU64, wire.OperateTypeUVarint:
		if c.U > math.MaxInt64 {
			return vtypes.NewFloat(float64(c.U)), true
		}
		return vtypes.NewInt(int64(c.U)), true //nolint:gosec // bounded above
	case wire.OperateTypeU8, wire.OperateTypeU16, wire.OperateTypeU32,
		wire.OperateTypeI8, wire.OperateTypeI16, wire.OperateTypeI32, wire.OperateTypeI64,
		wire.OperateTypeIVarint:
		// U8/U16/U32 never exceed math.MaxInt64; the signed types and
		// IVARINT already hold their int64 bit pattern (see doc comment
		// above), so a plain reinterpret is exact in every case.
		return vtypes.NewInt(int64(c.U)), true //nolint:gosec // reinterpret stored bits
	case wire.OperateTypeF32, wire.OperateTypeF64:
		return vtypes.NewFloat(c.F), true
	case wire.OperateTypeBytes, wire.OperateTypeFixed:
		return vtypes.NewString(string(c.B)), true
	default: // OperateTypeUnset, OperateTypeTable, and any unknown tag
		return vtypes.Value{}, false
	}
}

// ResultValue maps a Resolve result to the payload-shaped vtypes.Value a
// filter compares against: Scalar goes through CellValue; Count (a table's
// row count) maps to ValueInt; Absent, RowPresent, and Table carry no value
// a filter can compare, so they report ok == false.
func ResultValue(res Result) (vtypes.Value, bool) {
	switch res.Kind {
	case Scalar:
		return CellValue(res.Cell)
	case Count:
		// res.Count is a table row count, bounded by wire.OperateMaxRows
		// (1<<20) — the callers that produce a Count (schemaTableHeader,
		// dynamicTableRows) both check this before returning it — so the
		// cast to int64 never loses precision.
		return vtypes.NewInt(int64(res.Count)), true //nolint:gosec // bounded by OperateMaxRows
	default: // Absent, RowPresent, Table
		return vtypes.Value{}, false
	}
}

// Entry is one synthetic index field IndexEntries extracts from a record:
// Field is the path part after the payload key ("rc", "#2", "b#count" — the
// same spelling a filter's "payloadKey/path" field string would use to name
// it), and Value is what that field maps to.
type Entry struct {
	Field string
	Value vtypes.Value
}

// IndexEntries extracts every synthetic index field a collection
// auto-indexes for rec, per the indexing ruling: one Entry per top-level
// scalar field (skipping one whose CellValue declines, and skipping a NaN
// float), plus one "<field>#count" Entry per top-level table field holding
// its row count. Table columns are never indexed. Entries are emitted in
// schema field order (schema mode) or ascending stored name order (dynamic
// mode) — both a pure function of rec's bytes, so the result is
// deterministic.
//
// IndexEntries never panics on hostile bytes: any error wraps ErrRecord.
func (r *Resolver) IndexEntries(rec []byte) ([]Entry, error) {
	if len(rec) < 1 {
		return nil, fmt.Errorf("%w: empty record", ErrRecord)
	}
	switch rec[0] {
	case wire.OperateModeSchema:
		return r.indexEntriesSchema(rec)
	case wire.OperateModeDynamic:
		return indexEntriesDynamic(rec)
	default:
		return nil, fmt.Errorf("%w: unknown mode byte %d", ErrRecord, rec[0])
	}
}

// indexEntriesSchema implements IndexEntries for a schema-mode record.
func (r *Resolver) indexEntriesSchema(rec []byte) ([]Entry, error) {
	e, err := r.schemaFor(rec)
	if err != nil {
		return nil, err
	}
	s, l := e.schema, e.layout
	base := 1 + e.blobLen
	if base > len(rec) {
		return nil, fmt.Errorf("%w: record ends inside its schema blob", ErrRecord)
	}

	// Every field contributes at most one Entry, so len(s.Fields) — already
	// bounded by wire.OperateMaxFields when this schema was decoded — is a
	// safe, tight capacity.
	entries := make([]Entry, 0, len(s.Fields))
	err = walkSchemaFields(rec, base, s, l, func(pos, off int, isTable bool, nRows uint64) error {
		name := schemaFieldLabel(s.StoreNames, s.Fields[pos].Name, pos)
		if isTable {
			entries = append(entries, Entry{Field: name + countSuffix, Value: vtypes.NewInt(int64(nRows))}) //nolint:gosec // bounded by OperateMaxRows
			return nil
		}
		fdef := &s.Fields[pos]
		cell, cerr := readCellAt(rec, off, fdef.Type, fdef.N)
		if cerr != nil {
			return cerr
		}
		if v, ok := CellValue(cell); ok && !isNaNValue(v) {
			entries = append(entries, Entry{Field: name, Value: v})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return entries, nil
}

// indexEntriesDynamic implements IndexEntries for a dynamic-mode record,
// walking its fields the same way resolveDynamic does but visiting every
// one instead of stopping at a named match.
func indexEntriesDynamic(rec []byte) ([]Entry, error) {
	nFields, m, err := readCanonicalUvarint(rec, 1)
	if err != nil {
		return nil, err
	}
	off := 1 + m
	if nFields > wire.OperateMaxFields {
		return nil, fmt.Errorf("%w: record declares %d fields", ErrRecord, nFields)
	}
	if !wire.CountFitsIn(int(nFields), len(rec)-off, 3) {
		return nil, fmt.Errorf("%w: record declares %d fields but is too short", ErrRecord, nFields)
	}

	// nFields is already bounded by wire.OperateMaxFields above, and each
	// field contributes at most one Entry.
	entries := make([]Entry, 0, nFields)
	var prevName []byte
	for i := uint64(0); i < nFields; i++ {
		name, next, nerr := readName(rec, off)
		if nerr != nil {
			return nil, nerr
		}
		off = next
		if prevName != nil && bytes.Compare(prevName, name) >= 0 {
			return nil, fmt.Errorf("%w: fields out of order", ErrRecord)
		}
		prevName = name

		typ, n, valOff, terr := readTypeTag(rec, off)
		if terr != nil {
			return nil, terr
		}

		if typ == wire.OperateTypeTable {
			inner, end, ierr := dynamicTableInner(rec, valOff)
			if ierr != nil {
				return nil, ierr
			}
			nRows, _, rerr := dynamicTableRows(inner)
			if rerr != nil {
				return nil, rerr
			}
			entries = append(entries, Entry{Field: string(name) + countSuffix, Value: vtypes.NewInt(int64(nRows))}) //nolint:gosec // bounded by OperateMaxRows
			off = end
			continue
		}

		cell, cerr := readCellAt(rec, valOff, typ, n)
		if cerr != nil {
			return nil, cerr
		}
		nextOff, serr := skipDynamicValue(rec, valOff, typ, n)
		if serr != nil {
			return nil, serr
		}
		off = nextOff

		if v, ok := CellValue(cell); ok && !isNaNValue(v) {
			entries = append(entries, Entry{Field: string(name), Value: v})
		}
	}
	return entries, nil
}

// schemaFieldLabel renders a schema-mode field's index-entry name: the
// field name when the schema stores names, else its position as "#N" — the
// same spelling ParsePath accepts back as a field segment.
func schemaFieldLabel(storeNames bool, name string, pos int) string {
	if storeNames {
		return name
	}
	return "#" + strconv.Itoa(pos)
}

// isNaNValue reports whether v is a NaN float — the one value CellValue can
// produce that IndexEntries declines to index (the indexing ruling: "the
// index declines them anyway; emit nothing").
func isNaNValue(v vtypes.Value) bool {
	return v.Kind == vtypes.ValueFloat && math.IsNaN(v.Flt)
}
