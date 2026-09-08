// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"reflect"
	"strings"
	"testing"
	"time"
)

func sampleArgs() *OperateArgs {
	return &OperateArgs{Key: []byte("session:42"), TTL: 90 * time.Second, TTLMode: OperateTTLCreateOnly, Create: OperateCreateSchema,
		Schema: sessionSchema().Encode(),
		Ops: []OperateOp{
			{Opcode: OperateOpIF, Aux: OperateCmpGE, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 0}}, A: 255, B: 2},
			{Opcode: OperateOpSHR, Type: OperateTypeFromSchema, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 0}}, A: 1},
			{Opcode: OperateOpADD, Type: OperateTypeFromSchema, Path: OperatePath{Kind: OperatePathCol, Field: OperateSeg{Pos: 3}, Key: []byte{1, 0, 0, 0, 0, 0, 0, 0}, Col: OperateSeg{Pos: 0}}, A: 1},
			{Opcode: OperateOpSET, Type: OperateTypeBytes, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{ByName: true, Name: "name"}}, Bytes: []byte("v")},
		},
		Rets: []OperateRet{{Mode: OperateRetValue, Path: OperatePath{Kind: OperatePathRecord}}, {Mode: OperateRetCount, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 3}}}}}
}

func TestOperateArgsRoundtrip(t *testing.T) {
	a := sampleArgs()
	b, err := EncodeOperateArgs(a)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeOperateArgs(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, a) {
		t.Fatalf("\n got %+v\nwant %+v", got, a)
	}
}

func TestOperateArgsLimits(t *testing.T) {
	a := sampleArgs()
	a.Ops = make([]OperateOp, OperateMaxOps+1)
	if _, err := EncodeOperateArgs(a); err == nil {
		t.Fatal("too many ops encoded")
	}
	a = sampleArgs()
	a.Ops[0].Path.Field = OperateSeg{ByName: true, Name: strings.Repeat("x", 256)}
	if _, err := EncodeOperateArgs(a); err == nil {
		t.Fatal("long name encoded")
	}
	a = sampleArgs()
	a.Ops[0].Bytes = make([]byte, 65536)
	if _, err := EncodeOperateArgs(a); err == nil {
		t.Fatal("long bytes encoded")
	}
}

func TestDecodeOperateArgsTruncation(t *testing.T) {
	b, _ := EncodeOperateArgs(sampleArgs())
	for i := 0; i < len(b); i++ {
		if _, err := DecodeOperateArgs(b[:i]); err == nil {
			t.Fatalf("prefix %d accepted", i)
		}
	}
}

func TestDecodeOperateArgsHostileCounts(t *testing.T) {
	b, _ := EncodeOperateArgs(&OperateArgs{Key: []byte("k"), Create: OperateCreateDynamic})
	// nOps sits after key(3)+ttl(8)+ttlMode(1)+create(1)+schemaLen(2)
	off := 2 + 1 + 8 + 1 + 1 + 2
	binary.BigEndian.PutUint16(b[off:], 0xFFFF)
	if _, err := DecodeOperateArgs(b); err == nil {
		t.Fatal("hostile nOps accepted")
	}
	b, _ = EncodeOperateArgs(&OperateArgs{Key: []byte("k"), Create: OperateCreateDynamic})
	b[2+1+8+1] = 5 // the create byte (after keyLen, key, ttlMs, ttlMode)
	if _, err := DecodeOperateArgs(b); err == nil {
		t.Fatal("bad create accepted")
	}
}

func TestOperateResultRoundtrip(t *testing.T) {
	r := &OperateResult{Status: OperateStatusCheckFailed, FailedOp: 3, Values: [][]byte{AppendTaggedCell(nil, Cell{Type: OperateTypeU8, U: 9}), {OperateTypeUnset}}}
	got, err := DecodeOperateResult(EncodeOperateResult(r))
	if err != nil || !reflect.DeepEqual(got, r) {
		t.Fatalf("%+v %v", got, err)
	}
	ok := &OperateResult{Status: OperateStatusOK, Values: [][]byte{}}
	got, _ = DecodeOperateResult(EncodeOperateResult(ok))
	if got.Status != OperateStatusOK || got.FailedOp != 0 {
		t.Fatal(got)
	}
	if _, err := DecodeOperateResult([]byte{OperateStatusOK, 0xFF, 0xFF}); err == nil {
		t.Fatal("hostile nRet accepted")
	}
}

func FuzzDecodeOperateArgs(f *testing.F) {
	b, _ := EncodeOperateArgs(sampleArgs())
	f.Add(b)
	f.Add([]byte{})
	f.Add([]byte{0, 1})
	f.Fuzz(func(t *testing.T, b []byte) {
		a, err := DecodeOperateArgs(b)
		if err != nil {
			return
		}
		b2, err := EncodeOperateArgs(a)
		if err != nil {
			t.Fatal("decoded args do not re-encode:", err)
		}
		a2, err := DecodeOperateArgs(b2)
		if err != nil || !reflect.DeepEqual(a, a2) {
			t.Fatal("not stable")
		}
	})
}
