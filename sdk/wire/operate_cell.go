// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"math"
)

// Cell is one decoded scalar value for the operate op family (design doc
// §2.1). Its meaning depends on Type:
//
//   - the eight fixed-width int types (U8..I64) and the two varint types
//     (UVARINT, IVARINT): U holds the value's bits reinterpreted as uint64 —
//     zero-extended for an unsigned type, sign-extended (as an int64 bit
//     pattern) for a signed type. This is the same convention v1 used so a
//     signed and unsigned field share one arithmetic path.
//   - F32/F64: F holds the value as a float64 (F32 stored rounded through
//     float32, see CanonFloat).
//   - BYTES: B holds the opaque payload.
//   - FIXED: N is the declared width (1–255) and B holds exactly N bytes.
//   - TABLE, UNSET: neither is stored cell data (TABLE is never a cell at
//     all; UNSET carries no bytes).
type Cell struct {
	Type uint8
	N    uint8 // FIXED width only; ignored otherwise
	U    uint64
	F    float64
	B    []byte
}

// CellWidth returns the fixed on-the-wire width in bytes of a cell of the
// given type, or 0 for UNSET (no data), or -1 when the type has no fixed
// width (BYTES, UVARINT, IVARINT are variable-length; TABLE is not a cell).
// n is the declared width for OperateTypeFixed and is ignored for every
// other type.
func CellWidth(t uint8, n uint8) int {
	switch t {
	case OperateTypeU8, OperateTypeI8:
		return 1
	case OperateTypeU16, OperateTypeI16:
		return 2
	case OperateTypeU32, OperateTypeI32, OperateTypeF32:
		return 4
	case OperateTypeU64, OperateTypeI64, OperateTypeF64:
		return 8
	case OperateTypeFixed:
		return int(n)
	case OperateTypeUnset:
		return 0
	default:
		return -1
	}
}

// ZeroCell returns the zero value of a cell of the given type: the value an
// absent or UNSET path reads as (design doc §2.4, "absent is zero"). For
// OperateTypeFixed it allocates n zero bytes so callers can compare/append
// without a nil special case.
func ZeroCell(t uint8, n uint8) Cell {
	c := Cell{Type: t, N: n}
	if t == OperateTypeFixed {
		c.B = make([]byte, n)
	}
	return c
}

// TypeIsInt reports whether t is one of the eight fixed-width integer types
// or one of the two varint types — every type ApplyScalar's integer path
// handles.
func TypeIsInt(t uint8) bool {
	switch t {
	case OperateTypeU8, OperateTypeU16, OperateTypeU32, OperateTypeU64,
		OperateTypeI8, OperateTypeI16, OperateTypeI32, OperateTypeI64,
		OperateTypeUVarint, OperateTypeIVarint:
		return true
	default:
		return false
	}
}

// TypeIsFixedInt reports whether t is one of the eight fixed-width integer
// types (U8..I64) — the subset that bitwise/shift ops apply to (varints have
// no fixed bit width to mask or shift within).
func TypeIsFixedInt(t uint8) bool {
	switch t {
	case OperateTypeU8, OperateTypeU16, OperateTypeU32, OperateTypeU64,
		OperateTypeI8, OperateTypeI16, OperateTypeI32, OperateTypeI64:
		return true
	default:
		return false
	}
}

// TypeIsUnsigned reports whether t is an unsigned fixed-width int type or the
// unsigned varint type — the types whose stored bits in Cell.U are
// zero-extended rather than sign-extended.
func TypeIsUnsigned(t uint8) bool {
	switch t {
	case OperateTypeU8, OperateTypeU16, OperateTypeU32, OperateTypeU64, OperateTypeUVarint:
		return true
	default:
		return false
	}
}

// TypeIsFloat reports whether t is F32 or F64 — the types ApplyScalar's
// float path handles, stored in Cell.F.
func TypeIsFloat(t uint8) bool {
	return t == OperateTypeF32 || t == OperateTypeF64
}

// TypeIsBytes reports whether t is BYTES or FIXED — the types ApplyScalar's
// bytewise path handles, stored in Cell.B.
func TypeIsBytes(t uint8) bool {
	return t == OperateTypeBytes || t == OperateTypeFixed
}

// AppendCellData appends c's data-only encoding (no type tag, no FIXED
// width) to dst: little-endian for the fixed-width int and float types,
// binary.AppendUvarint for UVARINT, zigzagged binary.AppendUvarint for
// IVARINT, [len uvarint][data] for BYTES, and the raw N bytes for FIXED.
// UNSET appends nothing. It is the caller's responsibility not to call this
// with OperateTypeTable, which is never cell data.
func AppendCellData(dst []byte, c Cell) []byte {
	switch c.Type {
	case OperateTypeU8, OperateTypeI8:
		return append(dst, byte(c.U))
	case OperateTypeU16, OperateTypeI16:
		return binary.LittleEndian.AppendUint16(dst, uint16(c.U)) //nolint:gosec // low 16 bits by design
	case OperateTypeU32, OperateTypeI32:
		return binary.LittleEndian.AppendUint32(dst, uint32(c.U)) //nolint:gosec // low 32 bits by design
	case OperateTypeU64, OperateTypeI64:
		return binary.LittleEndian.AppendUint64(dst, c.U)
	case OperateTypeF32:
		return binary.LittleEndian.AppendUint32(dst, math.Float32bits(float32(c.F)))
	case OperateTypeF64:
		return binary.LittleEndian.AppendUint64(dst, math.Float64bits(c.F))
	case OperateTypeUVarint:
		return binary.AppendUvarint(dst, c.U)
	case OperateTypeIVarint:
		return binary.AppendUvarint(dst, zigzag(int64(c.U))) //nolint:gosec // reinterpret stored signed bits
	case OperateTypeBytes:
		dst = binary.AppendUvarint(dst, uint64(len(c.B))) //nolint:gosec // bounded by OperateMaxBytesLen on encode paths
		return append(dst, c.B...)
	case OperateTypeFixed:
		return append(dst, c.B...)
	default: // OperateTypeUnset, OperateTypeTable: no data
		return dst
	}
}

// DecodeCellData reads one value of type t (and, for OperateTypeFixed,
// declared width n) from the front of b, returning the decoded cell and the
// number of bytes consumed. Every branch is truncation-checked; BYTES length
// is additionally bounded by OperateMaxBytesLen before it is trusted to size
// an allocation. t == OperateTypeTable is rejected: a table is never cell
// data.
func DecodeCellData(t uint8, n uint8, b []byte) (Cell, int, error) {
	c := Cell{Type: t, N: n}
	switch t {
	case OperateTypeUnset:
		return c, 0, nil
	case OperateTypeU8:
		if len(b) < 1 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(b[0])
		return c, 1, nil
	case OperateTypeI8:
		if len(b) < 1 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(int64(int8(b[0]))) //nolint:gosec // sign-extend
		return c, 1, nil
	case OperateTypeU16:
		if len(b) < 2 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(binary.LittleEndian.Uint16(b[:2]))
		return c, 2, nil
	case OperateTypeI16:
		if len(b) < 2 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(int64(int16(binary.LittleEndian.Uint16(b[:2])))) //nolint:gosec // sign-extend
		return c, 2, nil
	case OperateTypeU32:
		if len(b) < 4 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(binary.LittleEndian.Uint32(b[:4]))
		return c, 4, nil
	case OperateTypeI32:
		if len(b) < 4 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(int64(int32(binary.LittleEndian.Uint32(b[:4])))) //nolint:gosec // sign-extend
		return c, 4, nil
	case OperateTypeU64, OperateTypeI64:
		if len(b) < 8 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = binary.LittleEndian.Uint64(b[:8])
		return c, 8, nil
	case OperateTypeF32:
		if len(b) < 4 {
			return Cell{}, 0, ErrShortArgs
		}
		c.F = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[:4])))
		return c, 4, nil
	case OperateTypeF64:
		if len(b) < 8 {
			return Cell{}, 0, ErrShortArgs
		}
		c.F = math.Float64frombits(binary.LittleEndian.Uint64(b[:8]))
		return c, 8, nil
	case OperateTypeUVarint:
		v, m := binary.Uvarint(b)
		if m <= 0 { // 0 = truncated, <0 = overflow (> 10 bytes / > 64 bits)
			return Cell{}, 0, ErrShortArgs
		}
		c.U = v
		return c, m, nil
	case OperateTypeIVarint:
		v, m := binary.Uvarint(b)
		if m <= 0 {
			return Cell{}, 0, ErrShortArgs
		}
		c.U = uint64(unzigzag(v)) //nolint:gosec // store signed value bits
		return c, m, nil
	case OperateTypeBytes:
		ln, m := binary.Uvarint(b)
		if m <= 0 {
			return Cell{}, 0, ErrShortArgs
		}
		if ln > OperateMaxBytesLen {
			return Cell{}, 0, ErrOperateCap
		}
		if !CountFitsIn(int(ln), len(b)-m, 1) {
			return Cell{}, 0, ErrShortArgs
		}
		data := make([]byte, ln)
		copy(data, b[m:m+int(ln)])
		c.B = data
		return c, m + int(ln), nil
	case OperateTypeFixed:
		if !CountFitsIn(int(n), len(b), 1) {
			return Cell{}, 0, ErrShortArgs
		}
		data := make([]byte, n)
		copy(data, b[:n])
		c.B = data
		return c, int(n), nil
	default: // includes OperateTypeTable
		return Cell{}, 0, ErrOperateType
	}
}

// AppendTaggedCell appends a self-describing cell: [type][n iff FIXED][data].
// It is the wire form used wherever a value must carry its own type (a
// dynamic-mode field, an operate return value).
func AppendTaggedCell(dst []byte, c Cell) []byte {
	dst = append(dst, c.Type)
	if c.Type == OperateTypeFixed {
		dst = append(dst, c.N)
	}
	return AppendCellData(dst, c)
}

// DecodeTaggedCell reads one AppendTaggedCell-encoded value from the front of
// b, returning the decoded cell and the number of bytes consumed. A type tag
// >= OperateTypeCount is unknown and rejected; OperateTypeTable is rejected
// too, since a table is never a cell in its own right.
func DecodeTaggedCell(b []byte) (Cell, int, error) {
	if len(b) < 1 {
		return Cell{}, 0, ErrShortArgs
	}
	t := b[0]
	if t >= OperateTypeCount || t == OperateTypeTable {
		return Cell{}, 0, ErrOperateType
	}
	off := 1
	var n uint8
	if t == OperateTypeFixed {
		if len(b) < 2 {
			return Cell{}, 0, ErrShortArgs
		}
		n = b[1]
		off = 2
	}
	c, m, err := DecodeCellData(t, n, b[off:])
	if err != nil {
		return Cell{}, 0, err
	}
	return c, off + m, nil
}

// CanonFloat canonicalizes f for storage as type t (design doc §2.5): every
// NaN becomes the canonical quiet NaN (float64 bits
// 0x7FF8000000000000, which narrows to the canonical float32 quiet NaN
// 0x7FC00000), and OperateTypeF32 values are rounded through float32 so the
// stored bytes are a pure function of the value regardless of the caller's
// native width.
func CanonFloat(t uint8, f float64) float64 {
	if math.IsNaN(f) {
		f = math.Float64frombits(0x7FF8000000000000)
	}
	if t == OperateTypeF32 {
		return float64(float32(f))
	}
	return f
}

// zigzag and unzigzag map a signed int64 to/from an unsigned "zigzag" value
// so small magnitudes (positive or negative) encode as small LEB128 varints:
// standard zigzag encoding.
func zigzag(i int64) uint64   { return uint64((i << 1) ^ (i >> 63)) } //nolint:gosec // standard zigzag
func unzigzag(u uint64) int64 { return int64(u>>1) ^ -int64(u&1) }    //nolint:gosec // standard zigzag
