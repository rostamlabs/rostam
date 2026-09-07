// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"math"
	"math/bits"
)

// Operand carries a scalar op's operand fields (design doc §3, "Every op is
// (opcode, type, aux, path, a i64, b i64, bytes)"): A holds an int operand or
// the IEEE bit pattern of a float operand, per the field's stored type; Aux
// selects STAMP's unit (OperateStamp*); Bytes carries a BYTES/FIXED operand;
// StampMs is the transaction timestamp STAMP writes (already resolved by the
// caller, e.g. from the applying transaction), independent of A/Bytes.
type Operand struct {
	A       int64
	Aux     uint8
	Bytes   []byte
	StampMs int64
}

// ApplyScalar applies opcode to c using op, returning the new cell (design
// doc §3.1). present reflects whether the path resolved to an existing value
// before this call — the "absent reads as zero" rule (§2.4) means c is
// already ZeroCell(c.Type, c.N) in that case, so present itself does not
// change the arithmetic here; it exists for callers/comparators that need it
// (see CompareCell). An opcode not valid for c.Type's domain, or unknown
// outright, returns ErrOperateType / ErrOperateOpcode and a zero Cell.
func ApplyScalar(opcode uint8, c Cell, present bool, op Operand) (Cell, error) { //nolint:revive // present kept for API symmetry with CompareCell; unused here by design, see doc comment
	switch {
	case TypeIsFloat(c.Type):
		return applyFloat(opcode, c, op)
	case TypeIsInt(c.Type):
		return applyInt(opcode, c, op)
	case TypeIsBytes(c.Type):
		return applyBytes(opcode, c, op)
	default:
		return Cell{}, ErrOperateType
	}
}

// applyInt implements the integer scalar ops (design doc §3.1) for the eight
// fixed-width int types and the two varint types. ADD/MUL saturate at the
// stored type's range; MIN/MAX clamp the operand into that range first, then
// compare; AND/OR/XOR/SHL/SHR require a fixed-width type (varints have no
// fixed bit width to mask or shift within) and operate on the field's raw
// bits, masked to its width and sign-extended back for a signed type; STAMP
// writes the transaction timestamp in its aux unit, saturating.
func applyInt(opcode uint8, c Cell, op Operand) (Cell, error) {
	unsigned := TypeIsUnsigned(c.Type)
	switch opcode {
	case OperateOpSET:
		c.U = storeIntOperand(c.Type, unsigned, op.A)
		return c, nil

	case OperateOpADD:
		if unsigned {
			c.U = addSatUnsigned(c.U, op.A, typeUMax(c.Type))
		} else {
			lo, hi := typeSignedRange(c.Type)
			cur := signExtend(c.U, typeBits(c.Type))
			c.U = uint64(addSatSigned(cur, op.A, lo, hi)) //nolint:gosec // reinterpret store convention
		}
		return c, nil

	case OperateOpMUL:
		if unsigned {
			c.U = mulSatUnsigned(c.U, op.A, typeUMax(c.Type))
		} else {
			lo, hi := typeSignedRange(c.Type)
			cur := signExtend(c.U, typeBits(c.Type))
			c.U = uint64(mulSatSigned(cur, op.A, lo, hi)) //nolint:gosec // reinterpret store convention
		}
		return c, nil

	case OperateOpMIN, OperateOpMAX:
		wantMin := opcode == OperateOpMIN
		if unsigned {
			v := clampUnsigned(op.A, typeUMax(c.Type))
			if wantMin == (v < c.U) {
				c.U = v
			}
		} else {
			lo, hi := typeSignedRange(c.Type)
			v := clampSigned(op.A, lo, hi)
			cur := signExtend(c.U, typeBits(c.Type))
			if wantMin == (v < cur) {
				c.U = uint64(v) //nolint:gosec // reinterpret store convention
			}
		}
		return c, nil

	case OperateOpAND, OperateOpOR, OperateOpXOR, OperateOpSHL, OperateOpSHR:
		return applyBitOp(opcode, c, unsigned, op.A)

	case OperateOpSTAMP:
		ms := op.StampMs
		if op.Aux == OperateStampS {
			ms /= 1000
		}
		c.U = storeIntOperand(c.Type, unsigned, ms)
		return c, nil

	default:
		return Cell{}, ErrOperateOpcode
	}
}

// applyBitOp implements AND/OR/XOR/SHL/SHR (design doc §3.1): all five
// require a fixed-width int type — varints have no fixed bit width to mask
// or shift within. The field's current bits are taken low-width-bits-only
// (c.U & mask; upper bits are just this convention's sign/zero extension),
// combined or shifted within that width, then written back zero-extended
// (unsigned) or sign-extended (signed) per Cell.U's storage convention.
func applyBitOp(opcode uint8, c Cell, unsigned bool, a int64) (Cell, error) {
	if !TypeIsFixedInt(c.Type) {
		return Cell{}, ErrOperateType
	}
	w := typeBits(c.Type)
	mask := widthMask(w)
	cur := c.U & mask

	var raw uint64
	switch opcode {
	case OperateOpAND:
		raw = cur & (uint64(a) & mask) //nolint:gosec // bit pattern reinterpret, per field convention
	case OperateOpOR:
		raw = cur | (uint64(a) & mask) //nolint:gosec // bit pattern reinterpret
	case OperateOpXOR:
		raw = cur ^ (uint64(a) & mask) //nolint:gosec // bit pattern reinterpret
	case OperateOpSHL:
		if a < 0 || a > 64 {
			return Cell{}, ErrOperateType
		}
		if uint64(a) >= uint64(w) { //nolint:gosec // a checked >= 0 above
			raw = 0
		} else {
			raw = (cur << uint(a)) & mask //nolint:gosec // a in [0,64)
		}
	case OperateOpSHR:
		if a < 0 || a > 64 {
			return Cell{}, ErrOperateType
		}
		switch {
		case unsigned:
			if uint64(a) >= uint64(w) { //nolint:gosec // a checked >= 0 above
				raw = 0
			} else {
				raw = cur >> uint(a) //nolint:gosec // a in [0,64)
			}
		default:
			signedCur := signExtend(c.U, w)
			if uint64(a) >= uint64(w) { //nolint:gosec // a checked >= 0 above
				if signedCur < 0 {
					raw = mask // full shift-out of a negative value is all-ones (-1) within the width
				} else {
					raw = 0
				}
			} else {
				raw = uint64(signedCur>>uint(a)) & mask //nolint:gosec // arithmetic shift, then re-mask to width
			}
		}
	}

	if unsigned {
		c.U = raw
	} else {
		c.U = uint64(signExtend(raw, w)) //nolint:gosec // reinterpret store convention
	}
	return c, nil
}

// storeIntOperand clamps/truncates operand v into c's declared type and
// returns it in Cell.U's storage convention (zero-extended for unsigned,
// sign-extended for signed), for SET and STAMP.
func storeIntOperand(t uint8, unsigned bool, v int64) uint64 {
	if unsigned {
		return clampUnsigned(v, typeUMax(t))
	}
	lo, hi := typeSignedRange(t)
	return uint64(clampSigned(v, lo, hi)) //nolint:gosec // reinterpret store convention
}

// applyFloat implements the float scalar ops (design doc §3.1) for F32/F64.
// The operand is the IEEE bit pattern of a float64 (or, for F32, the low 32
// bits of a float32 pattern), per §3 ("a carries ... the IEEE bit pattern of
// a float operand"). Every result is canonicalized via CanonFloat so a NaN
// is always the canonical quiet NaN and F32 storage round-trips exactly.
// MIN/MAX use a plain IEEE comparison, so a NaN operand never replaces the
// stored value (NaN compares false against everything).
func applyFloat(opcode uint8, c Cell, op Operand) (Cell, error) {
	switch opcode {
	case OperateOpSET:
		c.F = CanonFloat(c.Type, floatOperand(c.Type, op.A))
	case OperateOpADD:
		c.F = CanonFloat(c.Type, c.F+floatOperand(c.Type, op.A))
	case OperateOpMUL:
		c.F = CanonFloat(c.Type, c.F*floatOperand(c.Type, op.A))
	case OperateOpMIN:
		if v := floatOperand(c.Type, op.A); v < c.F {
			c.F = CanonFloat(c.Type, v)
		}
	case OperateOpMAX:
		if v := floatOperand(c.Type, op.A); v > c.F {
			c.F = CanonFloat(c.Type, v)
		}
	default:
		return Cell{}, ErrOperateType
	}
	return c, nil
}

// floatOperand reinterprets a as the IEEE-754 bit pattern of a float64
// operand (design doc §3: "a carries ... the IEEE bit pattern of a float
// operand"). This is uniform across F32 and F64 fields — the operand always
// travels as a double; a field's own stored width (4 bytes for F32) is a
// property of Cell's wire encoding, not of how its operands are carried. The
// result is rounded back down through CanonFloat by the caller when t is
// OperateTypeF32.
func floatOperand(_ uint8, a int64) float64 {
	return math.Float64frombits(uint64(a)) //nolint:gosec // 64-bit IEEE pattern
}

// applyBytes implements the BYTES/FIXED scalar ops: SET, MIN, and MAX,
// compared bytewise via bytes.Compare (design doc §3.1). A FIXED cell's
// operand length must equal c.N; a BYTES operand is bounded by
// OperateMaxBytesLen (design doc §2.7). The result never aliases op.Bytes.
func applyBytes(opcode uint8, c Cell, op Operand) (Cell, error) {
	if c.Type == OperateTypeFixed {
		if len(op.Bytes) != int(c.N) {
			return Cell{}, ErrOperateType
		}
	} else if len(op.Bytes) > OperateMaxBytesLen {
		return Cell{}, ErrOperateCap
	}

	switch opcode {
	case OperateOpSET:
		c.B = cloneBytes(op.Bytes)
	case OperateOpMIN:
		if bytes.Compare(op.Bytes, c.B) < 0 {
			c.B = cloneBytes(op.Bytes)
		}
	case OperateOpMAX:
		if bytes.Compare(op.Bytes, c.B) > 0 {
			c.B = cloneBytes(op.Bytes)
		}
	default:
		return Cell{}, ErrOperateType
	}
	return c, nil
}

// cloneBytes returns an independent copy of b so a returned Cell never
// aliases a caller-owned operand slice.
func cloneBytes(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

// CompareCell evaluates cmp against c (design doc §3.3): EXISTS/ABSENT test
// present alone; every other comparator compares in c's domain — numeric
// (an absent cell reads as zero, per §2.4) or bytewise for BYTES/FIXED. For
// an unsigned cell, a negative operand is below every unsigned value (it
// cannot be represented in that domain). An unknown comparator (or one
// applied to a domain it does not support) returns ErrOperateCmp.
func CompareCell(cmp uint8, c Cell, present bool, op Operand) (bool, error) {
	switch cmp {
	case OperateCmpExists:
		return present, nil
	case OperateCmpAbsent:
		return !present, nil
	}

	var order int
	switch {
	case TypeIsFloat(c.Type):
		v := floatOperand(c.Type, op.A)
		order = floatCompare(c.F, v)
	case TypeIsInt(c.Type):
		if TypeIsUnsigned(c.Type) {
			if op.A < 0 {
				order = 1 // every unsigned value is above a negative operand
			} else {
				order = uint64Compare(c.U, uint64(op.A)) //nolint:gosec // op.A >= 0 checked above
			}
		} else {
			order = int64Compare(signExtend(c.U, typeBits(c.Type)), op.A)
		}
	case TypeIsBytes(c.Type):
		order = bytes.Compare(c.B, op.Bytes)
	default:
		return false, ErrOperateCmp
	}
	return compareOrder(cmp, order)
}

// CompareCount evaluates cmp against a table's row count n (design doc
// §3.3, "a table compares by its row count"): an unsigned comparison of n
// against op.A, where a negative op.A is below every count. EXISTS/ABSENT
// test present alone, same as CompareCell.
func CompareCount(cmp uint8, n uint64, present bool, op Operand) (bool, error) {
	switch cmp {
	case OperateCmpExists:
		return present, nil
	case OperateCmpAbsent:
		return !present, nil
	}
	var order int
	if op.A < 0 {
		order = 1 // every count is above a negative operand
	} else {
		order = uint64Compare(n, uint64(op.A)) //nolint:gosec // op.A >= 0 checked above
	}
	return compareOrder(cmp, order)
}

// compareOrder maps a three-way order (-1/0/1) to the EQ/NE/LT/LE/GT/GE
// result for cmp, rejecting any comparator not in that set (EXISTS/ABSENT
// are handled by the caller before order is ever computed).
func compareOrder(cmp uint8, order int) (bool, error) {
	switch cmp {
	case OperateCmpEQ:
		return order == 0, nil
	case OperateCmpNE:
		return order != 0, nil
	case OperateCmpLT:
		return order < 0, nil
	case OperateCmpLE:
		return order <= 0, nil
	case OperateCmpGT:
		return order > 0, nil
	case OperateCmpGE:
		return order >= 0, nil
	default:
		return false, ErrOperateCmp
	}
}

func floatCompare(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func uint64Compare(a, b uint64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func int64Compare(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

// --- v1 helpers (ports of ops/operate.go, unchanged behavior) --------------

// typeUMax returns the maximum value of unsigned type t (U8/U16/U32 exactly;
// U64 and UVARINT both saturate at the u64 boundary).
func typeUMax(t uint8) uint64 {
	switch t {
	case OperateTypeU8:
		return math.MaxUint8
	case OperateTypeU16:
		return math.MaxUint16
	case OperateTypeU32:
		return math.MaxUint32
	default: // U64, UVARINT
		return math.MaxUint64
	}
}

// typeSignedRange returns the [lo, hi] range of signed type t (I8/I16/I32
// exactly; I64 and IVARINT both saturate at the i64 boundary).
func typeSignedRange(t uint8) (lo, hi int64) {
	switch t {
	case OperateTypeI8:
		return math.MinInt8, math.MaxInt8
	case OperateTypeI16:
		return math.MinInt16, math.MaxInt16
	case OperateTypeI32:
		return math.MinInt32, math.MaxInt32
	default: // I64, IVARINT
		return math.MinInt64, math.MaxInt64
	}
}

// typeBits returns the fixed bit width of a fixed-width int type (U8..I64).
// It is only meaningful for TypeIsFixedInt(t); varints have no fixed width.
func typeBits(t uint8) int {
	switch t {
	case OperateTypeU8, OperateTypeI8:
		return 8
	case OperateTypeU16, OperateTypeI16:
		return 16
	case OperateTypeU32, OperateTypeI32:
		return 32
	default:
		return 64
	}
}

// widthMask returns a mask with the low bits bits set (all 64 bits for
// bits >= 64).
func widthMask(bits int) uint64 {
	if bits >= 64 {
		return math.MaxUint64
	}
	return (uint64(1) << uint(bits)) - 1
}

// signExtend reinterprets the low bits bits of v as a signed integer of that
// width, sign-extended to int64.
func signExtend(v uint64, bits int) int64 {
	if bits >= 64 {
		return int64(v) //nolint:gosec // full width
	}
	shift := uint(64 - bits)
	return int64(v<<shift) >> shift //nolint:gosec // arithmetic shift sign-extends
}

// clampUnsigned clamps a into [0, max]: negative saturates to 0.
func clampUnsigned(a int64, max uint64) uint64 {
	if a < 0 {
		return 0
	}
	ua := uint64(a) //nolint:gosec // a >= 0 checked above
	if ua > max {
		return max
	}
	return ua
}

// clampSigned clamps a into [lo, hi].
func clampSigned(a, lo, hi int64) int64 {
	switch {
	case a < lo:
		return lo
	case a > hi:
		return hi
	default:
		return a
	}
}

// addSatUnsigned adds a signed delta to an unsigned value, saturating at
// [0, max] (never wrapping). uint64(-delta) yields the correct magnitude
// even for MinInt64 (two's-complement negation wraps back to the same bit
// pattern, which reinterpreted as uint64 is exactly |MinInt64|).
func addSatUnsigned(cur uint64, delta int64, max uint64) uint64 {
	if delta >= 0 {
		d := uint64(delta) //nolint:gosec // delta >= 0 checked above
		if d > max-cur {
			return max
		}
		return cur + d
	}
	d := uint64(-delta) //nolint:gosec // magnitude; correct even at MinInt64
	if d > cur {
		return 0
	}
	return cur - d
}

// addSatSigned adds delta to cur, saturating at [lo, hi] and detecting the
// int64 wrap first (so I64 saturates at the int64 boundary instead of
// wrapping).
func addSatSigned(cur, delta, lo, hi int64) int64 {
	sum := cur + delta
	switch {
	case delta > 0 && sum < cur:
		return hi // overflowed positive
	case delta < 0 && sum > cur:
		return lo // overflowed negative
	case sum < lo:
		return lo
	case sum > hi:
		return hi
	default:
		return sum
	}
}

// absU64 returns the magnitude of x as a uint64, correct even for
// math.MinInt64 (whose negation overflows int64 but wraps, via two's
// complement, back to a bit pattern that reinterpreted as uint64 is exactly
// the true magnitude, 1<<63).
func absU64(x int64) uint64 {
	if x < 0 {
		return uint64(-x) //nolint:gosec // two's-complement magnitude, correct even at MinInt64
	}
	return uint64(x)
}

// mulSatUnsigned multiplies cur by a signed delta, saturating at [0, max].
// A negative a has no meaning for an unsigned value, so it saturates to 0
// (matching addSatUnsigned's floor).
func mulSatUnsigned(cur uint64, a int64, max uint64) uint64 {
	if a < 0 {
		return 0
	}
	hi, lo := bits.Mul64(cur, uint64(a)) //nolint:gosec // a >= 0 checked above
	if hi != 0 || lo > max {
		return max
	}
	return lo
}

// mulSatSigned multiplies cur by a, saturating at [lo, hi]. It multiplies
// magnitudes with bits.Mul64 (a 64x64->128 multiply, so no product can
// silently wrap) and reattaches the sign, checking the magnitude against
// whichever bound (|lo| or hi) the result's sign would saturate toward.
func mulSatSigned(cur, a, lo, hi int64) int64 {
	if cur == 0 || a == 0 {
		return 0
	}
	neg := (cur < 0) != (a < 0)
	hiBits, loBits := bits.Mul64(absU64(cur), absU64(a))
	if neg {
		magLo := absU64(lo)
		if hiBits != 0 || loBits > magLo {
			return lo
		}
		return -int64(loBits) //nolint:gosec // magnitude bounded by |lo|; two's complement handles MinInt64 exactly
	}
	magHi := uint64(hi) //nolint:gosec // hi is non-negative for every signed range here
	if hiBits != 0 || loBits > magHi {
		return hi
	}
	return int64(loBits) //nolint:gosec // magnitude bounded by hi
}
