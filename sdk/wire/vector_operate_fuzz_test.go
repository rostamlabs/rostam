// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
)

// FuzzDecodeVectorOperateArgs extends the FuzzDecodeOperateArgs identity to
// the outer vector_operate frame: DecodeVectorOperateArgs must never panic on
// any input, and any successfully decoded call must re-encode to bytes that
// decode back to an identical call.
func FuzzDecodeVectorOperateArgs(f *testing.F) {
	seed := func(collection string, id uint64, payloadKey string, a *OperateArgs, ev uint64, has bool) {
		b, err := EncodeVectorOperateArgs(collection, id, payloadKey, a, ev, has)
		if err == nil {
			f.Add(b)
		}
	}
	seed("docs", 42, "body", vecOpArgs(), 0, false)
	seed("docs", 42, "body", vecOpArgs(), 7, true)
	seed("c", 0, "k", &OperateArgs{Create: OperateCreateSchema, Schema: sessionSchema().Encode()}, 0, false)
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{1})
	// A frame whose declared inner blob is LONGER than the operate args behind
	// it: every outer field decodes, the outer trailing-bytes check passes
	// because the slack sits INSIDE the declared length, and only the inner
	// decoder's exact-consumption rule catches it.
	if b := overDeclaredInnerBlob(); b != nil {
		f.Add(b)
	}

	f.Fuzz(func(t *testing.T, b []byte) {
		collection, id, payloadKey, a, ev, has, err := DecodeVectorOperateArgs(b)
		if err != nil {
			return
		}
		b2, err := EncodeVectorOperateArgs(collection, id, payloadKey, a, ev, has)
		if err != nil {
			t.Fatal("decoded args do not re-encode:", err)
		}
		collection2, id2, payloadKey2, a2, ev2, has2, err := DecodeVectorOperateArgs(b2)
		if err != nil {
			t.Fatal("re-encoded frame failed to decode:", err)
		}
		if collection2 != collection || id2 != id || payloadKey2 != payloadKey || ev2 != ev || has2 != has || !reflect.DeepEqual(a, a2) {
			t.Fatal("not stable")
		}
	})
}

// FuzzDecodeVectorOperateResult is FuzzDecodeVectorOperateArgs for the RESULT
// frame: DecodeVectorOperateResult must never panic on any input, and any frame
// it accepts must re-encode to bytes that decode back to the same (found,
// result, version) triple.
//
// The version is in the identity deliberately. It is an APPEND-ONLY trailing
// field, which is the one place a frame can silently lose information: a decoder
// that ignored a short remainder, or an encoder that dropped the field, would
// still round-trip the result and pass a laxer check.
func FuzzDecodeVectorOperateResult(f *testing.F) {
	seed := func(found bool, r *OperateResult, version uint64) {
		if b, err := EncodeVectorOperateResult(found, r, version); err == nil {
			f.Add(b)
		}
	}
	rOK := &OperateResult{Status: OperateStatusOK, Values: [][]byte{AppendTaggedCell(nil, Cell{Type: OperateTypeU8, U: 9}), {OperateTypeUnset}}}
	seed(false, nil, 0)
	seed(true, rOK, 0)
	seed(true, rOK, 42)
	seed(true, rOK, ^uint64(0))
	seed(true, &OperateResult{Status: OperateStatusCheckFailed, FailedOp: 3}, 7)
	// A pre-version frame: the same bytes with the trailing 8 stripped.
	if b, err := EncodeVectorOperateResult(true, rOK, 42); err == nil {
		f.Add(b[:len(b)-8])
	}
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{1})
	f.Add([]byte{2})

	f.Fuzz(func(t *testing.T, b []byte) {
		found, r, version, err := DecodeVectorOperateResult(b)
		if err != nil {
			return
		}
		b2, err := EncodeVectorOperateResult(found, r, version)
		if err != nil {
			t.Fatal("decoded result does not re-encode:", err)
		}
		found2, r2, version2, err := DecodeVectorOperateResult(b2)
		if err != nil {
			t.Fatal("re-encoded frame failed to decode:", err)
		}
		if found2 != found || version2 != version || !reflect.DeepEqual(r, r2) {
			t.Fatal("not stable")
		}
	})
}

// overDeclaredInnerBlob builds a vector_operate frame whose [argsLen u32]
// declares four bytes more than the inner operate blob actually occupies, with
// four zero bytes of slack sitting behind it. Every outer field is well formed
// and the OUTER trailing-bytes check passes, because the slack is inside the
// declared length — the frame is only refused because the inner decoder must
// consume its blob exactly.
func overDeclaredInnerBlob() []byte {
	b, err := EncodeVectorOperateArgs("docs", 42, "body", vecOpArgs(), 0, false)
	if err != nil {
		return nil
	}
	// Walk the outer header to the [argsLen u32] the encoder wrote:
	// [colLen u8][col][id u64][pkLen u16][payloadKey][casPresent u8].
	off := 1 + int(b[0]) + 8
	off += 2 + int(binary.BigEndian.Uint16(b[off:]))
	off++ // casPresent == 0 for this fixture
	inner := binary.BigEndian.Uint32(b[off:])
	out := append([]byte(nil), b...)
	binary.BigEndian.PutUint32(out[off:], inner+4)
	return append(out, 0, 0, 0, 0)
}

// TestVectorOperateArgsRejectsSlackInsideTheDeclaredBlob is the regression for
// the frame overDeclaredInnerBlob builds: bytes hidden inside an over-declared
// inner length used to be accepted, because the inner decoder stopped where the
// call ended and never checked that it had reached the end of its slice.
func TestVectorOperateArgsRejectsSlackInsideTheDeclaredBlob(t *testing.T) {
	good, err := EncodeVectorOperateArgs("docs", 42, "body", vecOpArgs(), 0, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(good); err != nil {
		t.Fatalf("the honest frame must decode: %v", err)
	}
	bad := overDeclaredInnerBlob()
	if bad == nil {
		t.Fatal("fixture failed to build")
	}
	if len(bad) != len(good)+4 {
		t.Fatalf("fixture is %d bytes, want %d (the honest frame plus four bytes of slack)", len(bad), len(good)+4)
	}
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(bad); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("frame with slack inside the declared inner blob = %v, want ErrOperateArgs", err)
	}
}

// TestVectorOperateResultRejectsSlackInsideTheDeclaredBlob is the same rule on
// the RESULT frame, whose inner blob is bounded by its own declared [resLen u32].
func TestVectorOperateResultRejectsSlackInsideTheDeclaredBlob(t *testing.T) {
	good, err := EncodeVectorOperateResult(true, &OperateResult{Status: OperateStatusOK}, 9)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, _, _, err := DecodeVectorOperateResult(good); err != nil {
		t.Fatalf("the honest frame must decode: %v", err)
	}
	// [found u8][resLen u32][inner][version u64]: grow resLen by four and slide
	// four zero bytes in behind the inner blob, leaving the version where the
	// decoder expects it.
	resLen := binary.BigEndian.Uint32(good[1:])
	bad := make([]byte, 0, len(good)+4)
	bad = append(bad, good[0])
	bad = binary.BigEndian.AppendUint32(bad, resLen+4)
	bad = append(bad, good[5:5+resLen]...)
	bad = append(bad, 0, 0, 0, 0)
	bad = append(bad, good[5+resLen:]...)
	if _, _, _, err := DecodeVectorOperateResult(bad); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("result frame with slack inside the declared inner blob = %v, want ErrOperateArgs", err)
	}
}

// TestOperateArgsRejectsTrailingBytes is the same rule at the KV entry point,
// where the blob is the whole request body rather than a declared sub-slice.
func TestOperateArgsRejectsTrailingBytes(t *testing.T) {
	good, err := EncodeOperateArgs(vecOpArgs())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, err := DecodeOperateArgs(good); err != nil {
		t.Fatalf("the honest frame must decode: %v", err)
	}
	if _, err := DecodeOperateArgs(append(append([]byte(nil), good...), 0)); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("operate args with a trailing byte = %v, want ErrOperateArgs", err)
	}
	goodRes, err := EncodeOperateResult(&OperateResult{Status: OperateStatusOK})
	if err != nil {
		t.Fatalf("encode result: %v", err)
	}
	if _, err := DecodeOperateResult(append(append([]byte(nil), goodRes...), 0)); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("operate result with a trailing byte = %v, want ErrOperateArgs", err)
	}
}
