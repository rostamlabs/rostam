// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"math"
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

// TestDecodeOperateArgsErrorIdentity pins which error each rejection
// reports. The client turns these into a status the caller branches on, so
// "over a cap" and "the frame is short" must not collapse into one another:
//
//   - a path position past the uint32 an OperateSeg addresses with is
//     ErrOperateArgs (the frame is complete, its content is out of range),
//   - nOps/nRet over OperateMaxOps/OperateMaxRet is ErrOperateCap,
//   - nOps/nRet the remaining bytes cannot hold is still ErrShortArgs.
func TestDecodeOperateArgsErrorIdentity(t *testing.T) {
	// One op whose field seg is a by-position segment holding 2^32, which is
	// one past the largest addressable position.
	frame := func(pos uint64) []byte {
		b := binary.BigEndian.AppendUint16(nil, 1) // keyLen
		b = append(b, 'k')
		b = binary.BigEndian.AppendUint64(b, 0)          // ttlMs
		b = append(b, OperateTTLKeep, OperateCreateNone) // ttlMode, create
		b = binary.BigEndian.AppendUint16(b, 0)          // schemaLen
		b = binary.BigEndian.AppendUint16(b, 1)          // nOps
		b = append(b, OperateOpADD, OperateTypeU64, 0)   // opcode, type, aux
		b = append(b, OperatePathField, 0)               // path kind, seg kind = by position
		b = binary.AppendUvarint(b, pos)
		b = binary.BigEndian.AppendUint64(b, 1)    // a
		b = binary.BigEndian.AppendUint64(b, 0)    // b
		b = binary.BigEndian.AppendUint16(b, 0)    // blen
		return binary.BigEndian.AppendUint16(b, 0) // nRet
	}
	if _, err := DecodeOperateArgs(frame(math.MaxUint32 + 1)); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("out-of-range seg position: err = %v, want ErrOperateArgs", err)
	}
	// The largest position that IS addressable decodes, so the case above is
	// failing on the range and not on the frame.
	got, err := DecodeOperateArgs(frame(math.MaxUint32))
	if err != nil {
		t.Fatalf("seg position at MaxUint32: %v", err)
	}
	if got.Ops[0].Path.Field.Pos != math.MaxUint32 {
		t.Fatalf("pos = %d", got.Ops[0].Path.Field.Pos)
	}

	// nOps and nRet over their caps, each in a frame long enough that
	// CountFitsIn would otherwise be satisfied. An over-cap count is
	// unencodable — EncodeOperateArgs refuses it — so it only ever arrives in
	// hostile input, which is exactly why the error has to say "cap".
	base, err := EncodeOperateArgs(&OperateArgs{Key: []byte("k"), Create: OperateCreateDynamic})
	if err != nil {
		t.Fatal(err)
	}
	nOpsOff := 2 + 1 + 8 + 1 + 1 + 2
	over := append(append([]byte(nil), base...), make([]byte, (OperateMaxOps+1)*minOpBytes)...)
	binary.BigEndian.PutUint16(over[nOpsOff:], OperateMaxOps+1)
	if _, err := DecodeOperateArgs(over); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("nOps over the cap: err = %v, want ErrOperateCap", err)
	}
	overRet := append(append([]byte(nil), base...), make([]byte, (OperateMaxRet+1)*minRetBytes)...)
	binary.BigEndian.PutUint16(overRet[nOpsOff+2:], OperateMaxRet+1) // nRet, with nOps = 0 before it
	if _, err := DecodeOperateArgs(overRet); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("nRet over the cap: err = %v, want ErrOperateCap", err)
	}

	// A count under the cap that the remaining bytes cannot hold is still a
	// short frame, not a cap hit.
	short := append([]byte(nil), base...)
	binary.BigEndian.PutUint16(short[nOpsOff:], OperateMaxOps)
	if _, err := DecodeOperateArgs(short); !errors.Is(err, ErrShortArgs) {
		t.Fatalf("nOps within the cap but past the byte budget: err = %v, want ErrShortArgs", err)
	}
	shortRet := append([]byte(nil), base...)
	binary.BigEndian.PutUint16(shortRet[nOpsOff+2:], OperateMaxRet)
	if _, err := DecodeOperateArgs(shortRet); !errors.Is(err, ErrShortArgs) {
		t.Fatalf("nRet within the cap but past the byte budget: err = %v, want ErrShortArgs", err)
	}
}

// TestOperateRetModeValidated covers the return-spec mode byte at both ends
// of the codec. The apply engine has no branch for a mode outside
// OperateRet*, so encoding one would ship a frame that can only be rejected
// server side, and accepting one on decode would hand the caller a spec it
// cannot act on — and break the decode/re-encode identity, since the encoder
// now refuses what the decoder produced.
func TestOperateRetModeValidated(t *testing.T) {
	a := sampleArgs()
	a.Rets = []OperateRet{{Mode: OperateRetCount + 1, Path: OperatePath{Kind: OperatePathRecord}}}
	if _, err := EncodeOperateArgs(a); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("encode: err = %v, want ErrOperateArgs", err)
	}
	for _, mode := range []uint8{OperateRetValue, OperateRetCount} {
		a.Rets[0].Mode = mode
		if _, err := EncodeOperateArgs(a); err != nil {
			t.Fatalf("encode mode %d: %v", mode, err)
		}
	}

	// Decode: patch the ret mode byte of a well-formed frame in place. The
	// rets are last, and a record-path ret is two bytes, so the mode byte is
	// the second-to-last.
	a.Rets[0].Mode = OperateRetCount
	b, err := EncodeOperateArgs(a)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeOperateArgs(b); err != nil {
		t.Fatalf("the unpatched frame must decode: %v", err)
	}
	b[len(b)-2] = OperateRetCount + 1
	if _, err := DecodeOperateArgs(b); !errors.Is(err, ErrOperateArgs) {
		t.Fatalf("decode: err = %v, want ErrOperateArgs", err)
	}
}

// TestDecodeOperateResultCountErrors covers the result frame's declared
// count, split the same way DecodeOperateArgs splits its own: over the cap is
// a frame declaring more values than a call may return, over the byte budget
// is a truncated or lying frame.
func TestDecodeOperateResultCountErrors(t *testing.T) {
	over := []byte{OperateStatusOK, 0, 0}
	binary.BigEndian.PutUint16(over[1:], OperateMaxRet+1)
	over = append(over, make([]byte, (OperateMaxRet+1)*4)...)
	if _, err := DecodeOperateResult(over); !errors.Is(err, ErrOperateCap) {
		t.Fatalf("nRet over the cap: err = %v, want ErrOperateCap", err)
	}

	short := []byte{OperateStatusOK, 0, 0}
	binary.BigEndian.PutUint16(short[1:], OperateMaxRet)
	if _, err := DecodeOperateResult(short); !errors.Is(err, ErrShortArgs) {
		t.Fatalf("nRet within the cap but past the byte budget: err = %v, want ErrShortArgs", err)
	}
}
