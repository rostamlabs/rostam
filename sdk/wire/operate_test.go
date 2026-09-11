// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
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

// AppendOperateArgs must be byte-identical to EncodeOperateArgs whether it
// allocates its own buffer or reuses one, or a pooling caller would put
// different bytes on the wire than a plain one.
func TestAppendOperateArgsMatchesEncode(t *testing.T) {
	a := sampleArgs()
	want, err := EncodeOperateArgs(a)
	if err != nil {
		t.Fatal(err)
	}

	nilDst, err := AppendOperateArgs(nil, a)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(nilDst, want) {
		t.Errorf("nil dst:\n got %x\nwant %x", nilDst, want)
	}

	// A buffer too small to hold even the header, and one large enough for the
	// whole frame, must both produce the same bytes.
	for _, size := range []int{0, 4, len(want), 4 * len(want)} {
		got, aErr := AppendOperateArgs(make([]byte, 0, size), a)
		if aErr != nil {
			t.Fatal(aErr)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("dst cap %d:\n got %x\nwant %x", size, got, want)
		}
	}

	// Dirty buffers must be overwritten, not appended to.
	dirty := make([]byte, 8*len(want))
	for i := range dirty {
		dirty[i] = 0xAA
	}
	got, err := AppendOperateArgs(dirty, a)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("dirty dst:\n got %x\nwant %x", got, want)
	}
}

// The reason the function exists: a pooled buffer must make the encode
// allocation-free once it has grown.
func TestAppendOperateArgsZeroAllocOnWarmBuffer(t *testing.T) {
	a := sampleArgs()
	buf, err := AppendOperateArgs(nil, a)
	if err != nil {
		t.Fatal(err)
	}
	if n := testing.AllocsPerRun(100, func() {
		buf, _ = AppendOperateArgs(buf[:0], a)
	}); n != 0 {
		t.Errorf("AppendOperateArgs on a warm buffer allocated %v times, want 0", n)
	}
}

// A pooled dst must decode byte-identically to a fresh one, and must carry
// nothing from its previous use - a stale Key or Schema would be read as the
// current call's.
func TestDecodeOperateArgsIntoMatchesDecode(t *testing.T) {
	full, err := EncodeOperateArgs(sampleArgs())
	if err != nil {
		t.Fatal(err)
	}
	// A deliberately different second frame: no ops, no rets, no schema.
	bare, err := EncodeOperateArgs(&OperateArgs{Key: []byte("k"), Create: OperateCreateDynamic})
	if err != nil {
		t.Fatal(err)
	}

	for _, order := range [][][]byte{{full, bare}, {bare, full}, {full, full}, {bare, bare}} {
		dst := new(OperateArgs)
		for i, frame := range order {
			want, dErr := DecodeOperateArgs(frame)
			if dErr != nil {
				t.Fatal(dErr)
			}
			if iErr := DecodeOperateArgsInto(dst, frame); iErr != nil {
				t.Fatal(iErr)
			}
			// Compared field by field rather than with DeepEqual: a reused
			// dst keeps its slice capacity, so an op-less frame decodes to
			// an EMPTY Ops rather than the nil a fresh decode produces.
			// Semantically identical, and len() is what every caller uses.
			if !bytes.Equal(dst.Key, want.Key) || dst.TTL != want.TTL ||
				dst.TTLMode != want.TTLMode || dst.Create != want.Create ||
				!bytes.Equal(dst.Schema, want.Schema) ||
				!reflect.DeepEqual(dst.Ops, want.Ops) && (len(dst.Ops) != 0 || len(want.Ops) != 0) ||
				!reflect.DeepEqual(dst.Rets, want.Rets) && (len(dst.Rets) != 0 || len(want.Rets) != 0) {
				t.Fatalf("reuse %d:\n got %+v\nwant %+v", i, dst, want)
			}
		}
	}
}

// Reuse must not allocate once the op slice has grown. Uses schema-POSITION
// paths only: a path addressed by NAME decodes its name with string(b[...]),
// which allocates per segment no matter how the op slice is managed. Schema
// mode - the hot path this pool exists for - always addresses by position.
func TestDecodeOperateArgsIntoZeroAllocOnWarmDst(t *testing.T) {
	positional := &OperateArgs{
		Key: []byte("session:42"), Create: OperateCreateSchema,
		Schema: sessionSchema().Encode(),
		Ops: []OperateOp{
			{Opcode: OperateOpADD, Type: OperateTypeFromSchema, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 0}}, A: 1},
			{Opcode: OperateOpSHL, Type: OperateTypeFromSchema, Path: OperatePath{Kind: OperatePathCol, Field: OperateSeg{Pos: 3}, Key: []byte{1, 0, 0, 0, 0, 0, 0, 0}, Col: OperateSeg{Pos: 0}}, A: 2},
		},
		Rets: []OperateRet{{Mode: OperateRetCount, Path: OperatePath{Kind: OperatePathField, Field: OperateSeg{Pos: 3}}}},
	}
	b, err := EncodeOperateArgs(positional)
	if err != nil {
		t.Fatal(err)
	}
	dst := new(OperateArgs)
	if err = DecodeOperateArgsInto(dst, b); err != nil {
		t.Fatal(err)
	}
	if n := testing.AllocsPerRun(100, func() {
		_ = DecodeOperateArgsInto(dst, b)
	}); n != 0 {
		t.Errorf("DecodeOperateArgsInto on a warm dst allocated %v times, want 0", n)
	}
}

// A failed decode must leave dst EMPTY, not holding the previous call's data.
// The decoder appends into dst's backing array as it goes, so a call that
// fails partway has already overwritten elements the old length still covers -
// a reusing caller that trusted the old contents would read a mix of two calls.
func TestDecodeOperateArgsIntoResetsOnError(t *testing.T) {
	good, err := EncodeOperateArgs(manyRowArgs(8))
	if err != nil {
		t.Fatal(err)
	}
	dst := new(OperateArgs)
	if err = DecodeOperateArgsInto(dst, good); err != nil {
		t.Fatal(err)
	}
	if len(dst.Ops) == 0 {
		t.Fatal("seed decode produced no ops; the test proves nothing")
	}

	for name, bad := range map[string][]byte{
		"trailing bytes": append(append([]byte(nil), good...), 0xFF),
		"truncated":      good[:len(good)-1],
		"truncated hard": good[:len(good)/2],
	} {
		if err = DecodeOperateArgsInto(dst, bad); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
		if len(dst.Ops) != 0 || len(dst.Rets) != 0 || dst.Key != nil || dst.Schema != nil {
			t.Errorf("%s: dst not reset after a failed decode: %+v", name, dst)
		}
		// Re-seed so the next case starts from a populated dst again.
		if err = DecodeOperateArgsInto(dst, good); err != nil {
			t.Fatal(err)
		}
	}
}

// Shrinking a reused dst must not leave the previous call's ops reachable past
// the new length - they alias that call's request buffer and would pin it.
func TestDecodeOperateArgsIntoClearsStaleTail(t *testing.T) {
	big, err := EncodeOperateArgs(manyRowArgs(16))
	if err != nil {
		t.Fatal(err)
	}
	small, err := EncodeOperateArgs(manyRowArgs(1))
	if err != nil {
		t.Fatal(err)
	}

	dst := new(OperateArgs)
	if err = DecodeOperateArgsInto(dst, big); err != nil {
		t.Fatal(err)
	}
	bigOps, bigRets := len(dst.Ops), len(dst.Rets)
	if err = DecodeOperateArgsInto(dst, small); err != nil {
		t.Fatal(err)
	}
	if len(dst.Ops) >= bigOps {
		t.Fatalf("small frame decoded to %d ops, want fewer than %d", len(dst.Ops), bigOps)
	}

	opTail := dst.Ops[:cap(dst.Ops)][len(dst.Ops):bigOps]
	for i := range opTail {
		if !reflect.DeepEqual(opTail[i], OperateOp{}) {
			t.Fatalf("stale op at tail index %d still set: %+v", i, opTail[i])
		}
	}
	if len(dst.Rets) >= bigRets {
		t.Fatalf("small frame decoded to %d rets, want fewer than %d", len(dst.Rets), bigRets)
	}
	retTail := dst.Rets[:cap(dst.Rets)][len(dst.Rets):bigRets]
	for i := range retTail {
		if !reflect.DeepEqual(retTail[i], OperateRet{}) {
			t.Fatalf("stale ret at tail index %d still set: %+v", i, retTail[i])
		}
	}
}

// manyRowArgs is a wide hot-path call: one record touched for nRows keys with
// ~4 position-addressed ops each, plus one return per key.
func manyRowArgs(nRows int) *OperateArgs {
	a := &OperateArgs{
		Key: []byte("session:42"), Create: OperateCreateSchema,
		TTL: 2 * time.Hour, TTLMode: OperateTTLSet,
		Schema: sessionSchema().Encode(),
	}
	for i := range nRows {
		row := []byte{byte(i), byte(i >> 8), 0, 0, 0, 0, 0, 0}
		col := func(c uint32) OperatePath {
			return OperatePath{Kind: OperatePathCol, Field: OperateSeg{Pos: 3}, Key: row, Col: OperateSeg{Pos: c}}
		}
		a.Ops = append(a.Ops,
			OperateOp{Opcode: OperateOpADD, Type: OperateTypeFromSchema, Path: col(0), A: 1},
			OperateOp{Opcode: OperateOpSHL, Type: OperateTypeFromSchema, Path: col(1), A: 2},
			OperateOp{Opcode: OperateOpOR, Type: OperateTypeFromSchema, Path: col(1), A: 3},
			OperateOp{Opcode: OperateOpADD, Type: OperateTypeFromSchema, Path: col(2), A: 1},
		)
		// One return per key, so Rets scales with nRows too - a fixed-size
		// Rets would leave the rets shrink-clear path untested.
		a.Rets = append(a.Rets, OperateRet{
			Mode: OperateRetValue,
			Path: OperatePath{Kind: OperatePathRow, Field: OperateSeg{Pos: 3}, Key: row},
		})
	}
	return a
}

func BenchmarkDecodeOperateArgs(b *testing.B) {
	buf, err := EncodeOperateArgs(manyRowArgs(16))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := DecodeOperateArgs(buf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeOperateArgsInto(b *testing.B) {
	buf, err := EncodeOperateArgs(manyRowArgs(16))
	if err != nil {
		b.Fatal(err)
	}
	dst := new(OperateArgs)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := DecodeOperateArgsInto(dst, buf); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodeOperateResult measures the reply frame with return values in
// it — the case the capacity reservation has to get right. With no values the
// frame is 5 bytes and the tiny allocator handles it; with values, an
// under-reserved buffer makes append grow the array.
func BenchmarkEncodeOperateResult(b *testing.B) {
	for _, n := range []int{1, 4, 16} {
		r := &OperateResult{Status: OperateStatusOK, Values: make([][]byte, n)}
		for i := range r.Values {
			r.Values[i] = make([]byte, 32) // a scalar-ish return
		}
		b.Run(fmt.Sprintf("values=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := EncodeOperateResult(r); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// The reservation must be EXACT: append never grows the array, so the frame
// comes back with cap == len. A formula that under-counts (the length prefixes
// without the value bytes, say) regrows and fails this; one that over-counts
// wastes the slack and fails it too. Covers both statuses, since failedOp is
// written only for OperateStatusCheckFailed.
func TestEncodeOperateResultReservesExactly(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status uint8
		vals   [][]byte
	}{
		{"ok/no values", OperateStatusOK, nil},
		{"ok/one empty value", OperateStatusOK, [][]byte{{}}},
		{"ok/mixed sizes", OperateStatusOK, [][]byte{make([]byte, 1), make([]byte, 300), make([]byte, 8)}},
		{"check failed/no values", OperateStatusCheckFailed, nil},
		{"check failed/with values", OperateStatusCheckFailed, [][]byte{make([]byte, 64), make([]byte, 7)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := EncodeOperateResult(&OperateResult{Status: tc.status, FailedOp: 3, Values: tc.vals})
			if err != nil {
				t.Fatal(err)
			}
			if cap(b) != len(b) {
				t.Errorf("frame is %d bytes in a %d-byte array; the reservation is not exact", len(b), cap(b))
			}
			// And it must still decode back to what went in.
			got, derr := DecodeOperateResult(b)
			if derr != nil {
				t.Fatalf("DecodeOperateResult: %v", derr)
			}
			if got.Status != tc.status || len(got.Values) != len(tc.vals) {
				t.Errorf("round trip: status=%d values=%d, want status=%d values=%d",
					got.Status, len(got.Values), tc.status, len(tc.vals))
			}
		})
	}
}
