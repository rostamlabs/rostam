// SPDX-License-Identifier: Apache-2.0

package wire

import (
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
