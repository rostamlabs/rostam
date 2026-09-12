// SPDX-License-Identifier: Apache-2.0

package ops

// The byte budget on an operate call's returns (wire.OperateMaxRetBytes).
//
// wire.OperateMaxRet bounds the return-spec COUNT, which says nothing about
// how much those specs produce: a record-kind VALUE copies the whole record,
// so a return list at the count cap asks for OperateMaxRet full copies held
// live at once. Against the 16 MiB record cap that is 64 GiB from a request
// of a few kilobytes, and none of it could ever be framed back — a reply that
// large is far past server.MaxFrameSize. These tests pin the refusal, its
// boundary, and the fact that the refusal happens while the values are being
// built rather than after they all exist.

import (
	"bytes"
	"errors"
	"runtime"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// retBudgetRecord builds a dynamic-mode record holding one BYTES field at the
// per-value cap (~64 KiB) and returns its stored bytes together with the size
// one record-kind VALUE produces for it — which is the whole record, so the
// two are the same length. The size is measured rather than computed: the
// budget arithmetic below derives its spec counts from it, so it must be what
// the engine really emits.
func retBudgetRecord(t *testing.T) ([]byte, int) {
	t.Helper()
	return retBudgetRecordWithBlob(t, wire.OperateMaxBytesLen)
}

// retBudgetRecordOfValueLen builds the same shape of record sized so that one
// record-kind VALUE is EXACTLY want bytes. The blob length is solved for by
// measuring, not computed from the encoding: the record framing around the
// blob (mode byte, field count, name, type tag, length prefix) is the encoder's
// business, and a test that re-derived it would pass while disagreeing with it.
func retBudgetRecordOfValueLen(t *testing.T, want int) ([]byte, int) {
	t.Helper()
	blobLen := want
	for range 8 {
		rec, valLen := retBudgetRecordWithBlob(t, blobLen)
		if valLen == want {
			return rec, valLen
		}
		// The framing overhead is a handful of bytes and varies only where the
		// blob's uvarint length prefix changes width, so correcting by the
		// shortfall converges immediately.
		blobLen += want - valLen
		if blobLen < 1 || blobLen > wire.OperateMaxBytesLen {
			t.Fatalf("cannot size a record to a %d-byte VALUE: blob would have to be %d bytes", want, blobLen)
		}
	}
	t.Fatalf("record sizing did not converge on a %d-byte VALUE", want)
	return nil, 0
}

func retBudgetRecordWithBlob(t *testing.T, blobLen int) ([]byte, int) {
	t.Helper()
	blob := bytes.Repeat([]byte{0xAB}, blobLen)
	rec, deleted, res, err := applyRecordBytes(nil, &wire.OperateArgs{
		Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpSET, Type: wire.OperateTypeBytes, Path: namePath("blob"), Bytes: blob},
		},
		Rets: []wire.OperateRet{{Mode: wire.OperateRetValue, Path: recPath()}},
	}, 0)
	if err != nil || deleted || rec == nil {
		t.Fatalf("seed: err=%v deleted=%v rec=%v", err, deleted, rec == nil)
	}
	if res.Status != wire.OperateStatusOK || len(res.Values) != 1 {
		t.Fatalf("seed result: status=%d values=%d", res.Status, len(res.Values))
	}
	if !bytes.Equal(res.Values[0], rec) {
		t.Fatalf("a record VALUE should be the stored record: %d bytes vs %d", len(res.Values[0]), len(rec))
	}
	return rec, len(res.Values[0])
}

// recordValueRets is n return specs, each asking for the whole record.
func recordValueRets(n int) []wire.OperateRet {
	rets := make([]wire.OperateRet, n)
	for i := range rets {
		rets[i] = wire.OperateRet{Mode: wire.OperateRetValue, Path: recPath()}
	}
	return rets
}

// The headline: a return list within the COUNT cap but far past the byte
// budget is refused. Before the budget existed this call returned 256 MiB of
// values — a 32,701x amplification of the ~8 KiB request that asked for it.
func TestOperateRetBytesBudgetRefusesAmplifiedReturns(t *testing.T) {
	rec, _ := retBudgetRecord(t)
	_, _, _, err := applyRecordBytes(rec, &wire.OperateArgs{
		Create: wire.OperateCreateNone,
		Rets:   recordValueRets(wire.OperateMaxRet),
	}, 0)
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("%d record-VALUE rets against a %d-byte record: err=%v, want wire.ErrOperateCap",
			wire.OperateMaxRet, len(rec), err)
	}
}

// The boundary, tested at EQUALITY rather than near it: a return list whose
// bytes come to exactly wire.OperateMaxRetBytes is accepted with every value
// intact, and one more identical spec is refused.
//
// Equality is the case worth pinning. A floored count would leave the accepted
// call short of the cap, and an off-by-one that wrongly rejected an exactly-full
// budget would sail through. So the record is sized so its VALUE divides the
// budget exactly (4096 bytes into 16,711,680 = 4080 specs, inside the 4096
// count cap) and the test FAILS LOUDLY if that stops holding, rather than
// quietly falling back to a floor.
func TestOperateRetBytesBudgetBoundary(t *testing.T) {
	const valueSize = 4096
	if wire.OperateMaxRetBytes%valueSize != 0 {
		t.Fatalf("budget %d is no longer a multiple of %d, so this test can no longer hit the cap exactly — pick a value size that divides it",
			wire.OperateMaxRetBytes, valueSize)
	}
	fits := wire.OperateMaxRetBytes / valueSize
	if fits < 2 || fits > wire.OperateMaxRet {
		t.Fatalf("budget %d / value %d = %d specs, which is not inside the count cap %d — the test needs a different value size",
			wire.OperateMaxRetBytes, valueSize, fits, wire.OperateMaxRet)
	}
	rec, valLen := retBudgetRecordOfValueLen(t, valueSize)

	_, _, res, err := applyRecordBytes(rec, &wire.OperateArgs{
		Create: wire.OperateCreateNone, Rets: recordValueRets(fits)}, 0)
	if err != nil {
		t.Fatalf("%d rets totalling exactly the budget (%d bytes): %v", fits, fits*valLen, err)
	}
	if fits*valLen != wire.OperateMaxRetBytes {
		t.Fatalf("the accepted call produced %d bytes, not the budget's %d — this is meant to be the exactly-full case",
			fits*valLen, wire.OperateMaxRetBytes)
	}
	if res.Status != wire.OperateStatusOK || len(res.Values) != fits {
		t.Fatalf("status=%d values=%d, want OK and %d", res.Status, len(res.Values), fits)
	}
	for i, v := range res.Values {
		if len(v) != valLen {
			t.Fatalf("value %d is %d bytes, want the whole %d-byte record", i, len(v), valLen)
		}
		if !bytes.Equal(v, res.Values[0]) {
			t.Fatalf("value %d differs from value 0 — the values under the budget must be intact", i)
		}
	}

	_, _, _, err = applyRecordBytes(rec, &wire.OperateArgs{
		Create: wire.OperateCreateNone, Rets: recordValueRets(fits + 1)}, 0)
	if !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("%d rets (%d bytes, budget %d): err=%v, want wire.ErrOperateCap",
			fits+1, (fits+1)*valLen, wire.OperateMaxRetBytes, err)
	}
}

// The refusal must happen INSIDE the return loop, so the peak a refused call
// reaches is the budget plus the single value that crossed it — not the whole
// list. A pre-pass that predicted sizes, or a check after the fact, would let
// the 256 MiB be built and then thrown away, which is the regression this
// guards: the bound is deliberately loose (its job is to separate "roughly
// the budget" from "everything materialised"), not a precise measurement.
func TestOperateRetBytesBudgetBoundsPeakAllocation(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector's own bookkeeping inflates allocation, and the engine pools this leans on drop items under -race; see race_detect_test.go")
	}
	rec, _ := retBudgetRecord(t)
	a := &wire.OperateArgs{
		Create: wire.OperateCreateNone,
		Rets:   recordValueRets(wire.OperateMaxRet),
	}
	// Warm the engine pools so their fill is not charged to the measurement.
	if _, _, _, err := applyRecordBytes(rec, a, 0); !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("warm: err=%v, want wire.ErrOperateCap", err)
	}

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, _, _, err := applyRecordBytes(rec, a, 0); !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("measured call: err=%v, want wire.ErrOperateCap", err)
	}
	runtime.ReadMemStats(&after)

	// TotalAlloc is cumulative, so it measures what the call allocated
	// regardless of what the collector did meanwhile.
	got := after.TotalAlloc - before.TotalAlloc
	limit := uint64(4 * (wire.OperateMaxRetBytes + maxOperateRecordBytes))
	if got > limit {
		t.Errorf("a refused %d-ret call allocated %d bytes; want at most %d (the budget plus a record, times four)",
			wire.OperateMaxRet, got, limit)
	}
}

// The CHECK_FAILED path evaluates the same return specs against the pre-call
// record, through a second engine opened on those bytes (retsBefore). It is a
// separate call into evalRets, so it needs the same bound: an aborted call is
// exactly as good a lever for the amplification as a successful one.
func TestOperateRetBytesBudgetOnCheckFailed(t *testing.T) {
	rec, _ := retBudgetRecord(t)
	a := &wire.OperateArgs{
		Create: wire.OperateCreateNone,
		Ops: []wire.OperateOp{
			// "blob" exists, so ABSENT is false and the whole list aborts.
			{Opcode: wire.OperateOpCHECK, Aux: wire.OperateCmpAbsent, Path: namePath("blob")},
		},
		Rets: recordValueRets(wire.OperateMaxRet),
	}
	if _, _, _, err := applyRecordBytes(rec, a, 0); !errors.Is(err, wire.ErrOperateCap) {
		t.Fatalf("CHECK_FAILED with over-budget rets: err=%v, want wire.ErrOperateCap", err)
	}

	// The same aborted call with one ret is still a plain CHECK_FAILED, so the
	// refusal above is the budget and not the abort.
	a.Rets = recordValueRets(1)
	_, _, res, err := applyRecordBytes(rec, a, 0)
	if err != nil {
		t.Fatalf("CHECK_FAILED with one ret: %v", err)
	}
	if res.Status != wire.OperateStatusCheckFailed || len(res.Values) != 1 {
		t.Fatalf("status=%d values=%d, want CHECK_FAILED and 1 value", res.Status, len(res.Values))
	}
}
