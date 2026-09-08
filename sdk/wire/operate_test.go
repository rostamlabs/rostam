// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
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
	enc, err := EncodeOperateResult(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeOperateResult(enc)
	if err != nil || !reflect.DeepEqual(got, r) {
		t.Fatalf("%+v %v", got, err)
	}
	ok := &OperateResult{Status: OperateStatusOK, Values: [][]byte{}}
	enc, err = EncodeOperateResult(ok)
	if err != nil {
		t.Fatal(err)
	}
	got, _ = DecodeOperateResult(enc)
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

// TestEncodeOperateArgsNegativeTTL covers a negative duration, which has no
// wire encoding: ttlMs is an unsigned 64-bit millisecond count, so
// converting one would wrap into a ~584-million-year TTL rather than the
// expiry the caller asked for.
func TestEncodeOperateArgsNegativeTTL(t *testing.T) {
	a := sampleArgs()
	a.TTL = -time.Second
	if _, err := EncodeOperateArgs(a); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("err = %v, want ErrOperateArgs", err)
	}
	a.TTL = 0
	if _, err := EncodeOperateArgs(a); err != nil {
		t.Fatalf("a zero TTL must still encode: %v", err)
	}
	// A sub-millisecond positive duration truncates to zero rather than
	// erroring — it is representable, just rounded.
	a.TTL = time.Microsecond
	if _, err := EncodeOperateArgs(a); err != nil {
		t.Fatalf("a sub-millisecond TTL must still encode: %v", err)
	}
}

// TestEncodeOperateResultLimits covers the two frames the result encoder
// cannot honestly represent: more values than nRet's uint16 (and
// OperateMaxRet) can count, and a status byte DecodeOperateResult would not
// accept back.
func TestEncodeOperateResultLimits(t *testing.T) {
	over := &OperateResult{Status: OperateStatusOK, Values: make([][]byte, OperateMaxRet+1)}
	if _, err := EncodeOperateResult(over); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("err = %v, want ErrOperateCap", err)
	}
	at := &OperateResult{Status: OperateStatusOK, Values: make([][]byte, OperateMaxRet)}
	b, err := EncodeOperateResult(at)
	if err != nil {
		t.Fatalf("exactly OperateMaxRet values must encode: %v", err)
	}
	got, err := DecodeOperateResult(b)
	if err != nil || len(got.Values) != OperateMaxRet {
		t.Fatalf("round trip at the cap: %+v %v", got, err)
	}
	if _, err := EncodeOperateResult(&OperateResult{Status: 7}); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("unknown status encoded: err = %v, want ErrOperateArgs", err)
	}
}

// TestDecodeOperateResultUnknownStatus covers the decode side of the same
// rule. The status byte selects the frame's shape — only
// OperateStatusCheckFailed carries failedOp — so an unknown one cannot be
// parsed past, and accepting it would hand the caller a status it has no
// branch for.
func TestDecodeOperateResultUnknownStatus(t *testing.T) {
	for _, status := range []byte{2, 3, 0xFF} {
		if _, err := DecodeOperateResult([]byte{status, 0, 0}); !errors.Is(err, ErrOperateArgs) {
			t.Fatalf("status %d: err = %v, want ErrOperateArgs", status, err)
		}
	}
	for _, status := range []uint8{OperateStatusOK, OperateStatusCheckFailed} {
		b, err := EncodeOperateResult(&OperateResult{Status: status})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeOperateResult(b); err != nil {
			t.Fatalf("status %d rejected: %v", status, err)
		}
	}
}
