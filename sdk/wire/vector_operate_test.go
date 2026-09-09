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

// vecOpArgs returns a minimal, valid vector_operate inner call: create=DYNAMIC,
// one ADD op against a by-name field. Key/TTL/TTLMode are left at their zero
// values, which is the only shape EncodeVectorOperateArgs accepts.
func vecOpArgs() *OperateArgs {
	return &OperateArgs{
		Create: OperateCreateDynamic,
		Ops: []OperateOp{
			{Opcode: OperateOpADD, Type: OperateTypeU32, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{ByName: true, Name: "rc"}}, A: 1},
		},
	}
}

// buildRawVectorOperateArgsFrame hand-builds a vector_operate args frame
// around an arbitrary inner blob, bypassing EncodeVectorOperateArgs's own
// guards entirely. Used to prove the decoder enforces its rules independently
// of the encoder (a hostile peer never calls the encoder).
func buildRawVectorOperateArgsFrame(collection string, id uint64, payloadKey string, hasExpected bool, expectedVersion uint64, inner []byte) []byte {
	buf := []byte{byte(len(collection))} //nolint:gosec // test helper, collection length controlled by the caller
	buf = append(buf, collection...)
	buf = binary.BigEndian.AppendUint64(buf, id)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(payloadKey))) //nolint:gosec // test helper
	buf = append(buf, payloadKey...)
	if hasExpected {
		buf = append(buf, 1)
		buf = binary.BigEndian.AppendUint64(buf, expectedVersion)
	} else {
		buf = append(buf, 0)
	}
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(inner))) //nolint:gosec // test helper
	buf = append(buf, inner...)
	return buf
}

func TestVectorOperateArgsRoundtrip(t *testing.T) {
	twoRets := vecOpArgs()
	twoRets.Rets = []OperateRet{
		{Mode: OperateRetValue, Path: OperatePath{Kind: OperatePathRecord}},
		{Mode: OperateRetCount, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 3}}},
	}
	schemaCall := &OperateArgs{
		Create: OperateCreateSchema,
		Schema: sessionSchema().Encode(),
		Ops: []OperateOp{
			{Opcode: OperateOpSET, Type: OperateTypeFromSchema, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 0}}, A: 5},
		},
	}

	cases := []struct {
		name            string
		collection      string
		id              uint64
		payloadKey      string
		args            *OperateArgs
		expectedVersion uint64
		hasExpected     bool
	}{
		{"no CAS", "docs", 42, "body", vecOpArgs(), 0, false},
		{"CAS present", "docs", 42, "body", vecOpArgs(), 7, true},
		{"two rets", "docs", 99, "meta", twoRets, 0, false},
		{"schema call", "docs", 5, "rec", schemaCall, 3, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := EncodeVectorOperateArgs(tc.collection, tc.id, tc.payloadKey, tc.args, tc.expectedVersion, tc.hasExpected)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			gotCol, gotID, gotPK, gotArgs, gotVer, gotHas, err := DecodeVectorOperateArgs(enc)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if gotCol != tc.collection || gotID != tc.id || gotPK != tc.payloadKey || gotVer != tc.expectedVersion || gotHas != tc.hasExpected {
				t.Fatalf("got (%q,%d,%q,%d,%v) want (%q,%d,%q,%d,%v)",
					gotCol, gotID, gotPK, gotVer, gotHas, tc.collection, tc.id, tc.payloadKey, tc.expectedVersion, tc.hasExpected)
			}
			if !reflect.DeepEqual(gotArgs, tc.args) {
				t.Fatalf("args mismatch:\n got %+v\nwant %+v", gotArgs, tc.args)
			}
		})
	}
}

func TestVectorOperateArgsRejectsInnerKey(t *testing.T) {
	a := vecOpArgs()
	a.Key = []byte("k")
	if _, err := EncodeVectorOperateArgs("docs", 1, "body", a, 0, false); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("encoder err = %v, want ErrOperateArgs", err)
	}

	inner, err := EncodeOperateArgs(&OperateArgs{Key: []byte("k"), Create: OperateCreateDynamic})
	if err != nil {
		t.Fatal(err)
	}
	frame := buildRawVectorOperateArgsFrame("docs", 1, "body", false, 0, inner)
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(frame); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("decoder err = %v, want ErrOperateArgs", err)
	}
}

func TestVectorOperateArgsRejectsTTLMode(t *testing.T) {
	byMode := vecOpArgs()
	byMode.TTLMode = OperateTTLSet
	if _, err := EncodeVectorOperateArgs("docs", 1, "body", byMode, 0, false); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("encoder err (TTLMode) = %v, want ErrOperateArgs", err)
	}

	byTTL := vecOpArgs()
	byTTL.TTL = time.Second
	if _, err := EncodeVectorOperateArgs("docs", 1, "body", byTTL, 0, false); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("encoder err (TTL) = %v, want ErrOperateArgs", err)
	}

	innerMode, err := EncodeOperateArgs(&OperateArgs{TTLMode: OperateTTLSet, Create: OperateCreateDynamic})
	if err != nil {
		t.Fatal(err)
	}
	frameMode := buildRawVectorOperateArgsFrame("docs", 1, "body", false, 0, innerMode)
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(frameMode); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("decoder err (TTLMode) = %v, want ErrOperateArgs", err)
	}

	innerTTL, err := EncodeOperateArgs(&OperateArgs{TTL: time.Second, Create: OperateCreateDynamic})
	if err != nil {
		t.Fatal(err)
	}
	frameTTL := buildRawVectorOperateArgsFrame("docs", 1, "body", false, 0, innerTTL)
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(frameTTL); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("decoder err (TTL) = %v, want ErrOperateArgs", err)
	}
}

func TestVectorOperateArgsRejectsEmptyNames(t *testing.T) {
	a := vecOpArgs()
	if _, err := EncodeVectorOperateArgs("", 1, "body", a, 0, false); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("empty collection: err = %v, want ErrOperateCap", err)
	}
	if _, err := EncodeVectorOperateArgs(strings.Repeat("x", 256), 1, "body", a, 0, false); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("256-byte collection: err = %v, want ErrOperateCap", err)
	}
	if _, err := EncodeVectorOperateArgs("docs", 1, "", a, 0, false); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("empty payload key: err = %v, want ErrOperateCap", err)
	}
}

// TestVectorOperateArgsHostileFlags covers the decoder-only branches a
// well-behaved encoder never reaches: a wire frame declaring an empty
// collection, an empty payload key, or a casPresent byte outside {0,1}. Each
// is necessarily a hostile or corrupt frame at decode time, mirroring
// TestDecodeOperateArgsHostileCounts's discipline for the inner codec.
func TestVectorOperateArgsHostileFlags(t *testing.T) {
	inner, err := EncodeOperateArgs(vecOpArgs())
	if err != nil {
		t.Fatal(err)
	}

	if _, _, _, _, _, _, err := DecodeVectorOperateArgs([]byte{0}); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Fatalf("colLen=0: err = %v, want ErrVectorArgsTruncated", err)
	}

	pkZero := buildRawVectorOperateArgsFrame("docs", 1, "", false, 0, inner)
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(pkZero); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("pkLen=0: err = %v, want ErrOperateArgs", err)
	}

	badCAS := buildRawVectorOperateArgsFrame("docs", 1, "body", false, 0, inner)
	// casPresent sits right after payloadKey: colLen(1)+col+id(8)+pkLen(2)+payloadKey.
	off := 1 + len("docs") + 8 + 2 + len("body")
	badCAS[off] = 2
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(badCAS); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("casPresent=2: err = %v, want ErrOperateArgs", err)
	}
}

func TestVectorOperateArgsTruncation(t *testing.T) {
	good, err := EncodeVectorOperateArgs("docs", 42, "body", vecOpArgs(), 7, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(good); i++ {
		if _, _, _, _, _, _, err := DecodeVectorOperateArgs(good[:i]); err == nil {
			t.Fatalf("prefix %d accepted", i)
		}
	}
}

func TestVectorOperateArgsTrailingBytes(t *testing.T) {
	good, err := EncodeVectorOperateArgs("docs", 42, "body", vecOpArgs(), 0, false)
	if err != nil {
		t.Fatal(err)
	}
	bad := append(good[:len(good):len(good)], 0)
	if _, _, _, _, _, _, err := DecodeVectorOperateArgs(bad); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("err = %v, want ErrOperateArgs", err)
	}
}

// TestVectorOperateArgsLyingArgsLen covers the 386 widening guard: argsLen
// overwritten with a value that would go negative under a 32-bit int
// conversion (0xFFFFFFFF, 0x80000000) must be rejected by a raw-uint32
// comparison against the remaining length, before any int conversion or
// allocation happens.
func TestVectorOperateArgsLyingArgsLen(t *testing.T) {
	good, err := EncodeVectorOperateArgs("docs", 42, "body", vecOpArgs(), 0, false)
	if err != nil {
		t.Fatal(err)
	}
	// argsLen sits right after casPresent (hasExpected=false, so no version
	// block): colLen(1)+col+id(8)+pkLen(2)+payloadKey+casPresent(1).
	off := 1 + len("docs") + 8 + 2 + len("body") + 1

	for _, lying := range []uint32{0xFFFFFFFF, 0x80000000} {
		frame := append([]byte(nil), good...)
		binary.BigEndian.PutUint32(frame[off:], lying)
		if _, _, _, _, _, _, err := DecodeVectorOperateArgs(frame); !errors.Is(err, ErrVectorArgsTruncated) {
			t.Fatalf("argsLen=%#x: err = %v, want ErrVectorArgsTruncated", lying, err)
		}
		if n := testing.AllocsPerRun(20, func() { _, _, _, _, _, _, _ = DecodeVectorOperateArgs(frame) }); n > 4 {
			t.Fatalf("argsLen=%#x allocated %v times; the length must be validated before any allocation", lying, n)
		}
	}
}

func TestVectorOperateResultRoundtrip(t *testing.T) {
	enc, err := EncodeVectorOperateResult(false, nil)
	if err != nil {
		t.Fatal(err)
	}
	found, r, err := DecodeVectorOperateResult(enc)
	if err != nil || found || r != nil {
		t.Fatalf("found=false: got (%v,%+v,%v)", found, r, err)
	}

	rOK := &OperateResult{Status: OperateStatusOK, Values: [][]byte{AppendTaggedCell(nil, Cell{Type: OperateTypeU8, U: 9}), {OperateTypeUnset}}}
	enc, err = EncodeVectorOperateResult(true, rOK)
	if err != nil {
		t.Fatal(err)
	}
	found, got, err := DecodeVectorOperateResult(enc)
	if err != nil || !found || !reflect.DeepEqual(got, rOK) {
		t.Fatalf("found=true OK: got (%v,%+v,%v) want %+v", found, got, err, rOK)
	}

	rFail := &OperateResult{Status: OperateStatusCheckFailed, FailedOp: 3}
	enc, err = EncodeVectorOperateResult(true, rFail)
	if err != nil {
		t.Fatal(err)
	}
	found, got, err = DecodeVectorOperateResult(enc)
	if err != nil || !found || !reflect.DeepEqual(got, rFail) {
		t.Fatalf("found=true CheckFailed: got (%v,%+v,%v) want %+v", found, got, err, rFail)
	}

	if _, err := EncodeVectorOperateResult(true, nil); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("found=true r=nil: err = %v, want ErrOperateArgs", err)
	}
}
