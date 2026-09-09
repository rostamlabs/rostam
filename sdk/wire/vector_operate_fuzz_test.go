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
