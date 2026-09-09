// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"encoding/binary"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// encodeErrorMsgFrame mirrors client/protocol.go's decodeErrorMsg wire shape
// ([2-byte length][message]) so a fake server can hand back a StatusError
// payload the client decodes the same way a real server's does.
func encodeErrorMsgFrame(msg string) []byte {
	out := make([]byte, 2+len(msg))
	binary.BigEndian.PutUint16(out[0:2], uint16(len(msg))) //nolint:gosec // test-only
	copy(out[2:], msg)
	return out
}

// TestCollectionOperateIncrements is the brief's canonical builder usage, run
// twice against a fake server that plays back an incrementing counter — the
// same round trip client.Operate performs for a KV key, but addressed at a
// point's record via Collection.Operate.
func TestCollectionOperateIncrements(t *testing.T) {
	var calls, counter int32
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		atomic.AddInt32(&calls, 1)
		opName, argsBytes := decodeOpFrame(t, body)
		if opName != "vector_operate" {
			t.Fatalf("op = %q, want vector_operate", opName)
		}
		collection, id, payloadKey, _, _, hasExpected, err := wire.DecodeVectorOperateArgs(argsBytes)
		if err != nil {
			t.Fatalf("DecodeVectorOperateArgs: %v", err)
		}
		if collection != "posts" || id != 1 || payloadKey != "session" || hasExpected {
			t.Fatalf("routing fields = (%q, %d, %q, hasExpected=%v), want (posts, 1, session, false)", collection, id, payloadKey, hasExpected)
		}
		n := atomic.AddInt32(&counter, 1)
		payload, err := wire.EncodeVectorOperateResult(true, &wire.OperateResult{
			Status: wire.OperateStatusOK,
			Values: [][]byte{wire.AppendTaggedCell(nil, wire.Cell{Type: wire.OperateTypeU32, U: uint64(n)})},
		}, uint64(n))
		if err != nil {
			t.Fatalf("EncodeVectorOperateResult: %v", err)
		}
		return StatusOK, payload
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")
	ctx := context.Background()

	for _, want := range []uint64{1, 2} {
		a, err := NewOperate(nil).Dynamic().
			AddT(F("rc"), wire.OperateTypeU32, 1).
			Return(F("rc")).
			Args()
		if err != nil {
			t.Fatalf("build operate args: %v", err)
		}
		found, res, version, err := col.Operate(ctx, OperateRequest{ID: 1, PayloadKey: "session", Args: a})
		if err != nil {
			t.Fatalf("Operate: %v", err)
		}
		if !found {
			t.Fatal("found = false, want true")
		}
		// The applied version rides back in the same frame: the fake server stamps
		// it with the same counter it uses for the returned rc, so a wrong or
		// dropped field shows up as a mismatch rather than as a plausible zero.
		if version != uint64(want) {
			t.Fatalf("version = %d, want %d", version, want)
		}
		cell, err := DecodeOperateValue(res.Values[0])
		if err != nil {
			t.Fatalf("DecodeOperateValue: %v", err)
		}
		if cell.U != want {
			t.Fatalf("rc = %d, want %d", cell.U, want)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("server saw %d calls, want 2", got)
	}
}

// TestCollectionOperateBuilderKeyAndTTLRejected checks the two shapes
// EncodeVectorOperateArgs rejects — a non-empty builder Key and a TTL/TTLMode
// carried on the inner OperateArgs — are caught by Collection.Operate BEFORE
// any round trip: the target is named once by (collection, ID, PayloadKey),
// and a point's TTL is the point's, never a record's.
func TestCollectionOperateBuilderKeyAndTTLRejected(t *testing.T) {
	var calls int32
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		atomic.AddInt32(&calls, 1)
		return StatusOK, []byte{0}
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")
	ctx := context.Background()

	// A non-empty builder key.
	keyed, err := NewOperate([]byte("k")).Dynamic().
		AddT(F("rc"), wire.OperateTypeU32, 1).Args()
	if err != nil {
		t.Fatalf("build operate args (keyed): %v", err)
	}
	_, res, _, err := col.Operate(ctx, OperateRequest{ID: 1, PayloadKey: "session", Args: keyed})
	if !errors.Is(err, wire.ErrOperateArgs) {
		t.Fatalf("err = %v, want wire.ErrOperateArgs", err)
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}

	// A TTL carried on the inner args.
	ttled, err := NewOperate(nil).Dynamic().
		AddT(F("rc"), wire.OperateTypeU32, 1).
		TTL(time.Second, wire.OperateTTLSet).Args()
	if err != nil {
		t.Fatalf("build operate args (ttled): %v", err)
	}
	_, res, _, err = col.Operate(ctx, OperateRequest{ID: 1, PayloadKey: "session", Args: ttled})
	if !errors.Is(err, wire.ErrOperateArgs) {
		t.Fatalf("err = %v, want wire.ErrOperateArgs", err)
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}

	if got := atomic.LoadInt32(&calls); got != 0 {
		t.Fatalf("server saw %d calls, want 0 — the rejection must happen before any round trip", got)
	}
}

// TestCollectionOperateMissingPoint checks the found=false, res=nil, err=nil
// convention: a missing/tombstoned/expired point is a flag, not an error,
// exactly like every other point write.
func TestCollectionOperateMissingPoint(t *testing.T) {
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		payload, err := wire.EncodeVectorOperateResult(false, nil, 0)
		if err != nil {
			t.Fatalf("EncodeVectorOperateResult: %v", err)
		}
		return StatusOK, payload
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")

	a, err := NewOperate(nil).Dynamic().AddT(F("rc"), wire.OperateTypeU32, 1).Args()
	if err != nil {
		t.Fatalf("build operate args: %v", err)
	}
	found, res, _, err := col.Operate(context.Background(), OperateRequest{ID: 1, PayloadKey: "session", Args: a})
	if err != nil {
		t.Fatalf("Operate: %v", err)
	}
	if found {
		t.Fatal("found = true, want false")
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}
}

// TestCollectionOperateRecordAbsentIsErrNotFound checks the OTHER absent
// shape a payload-key mutation can hit: the POINT exists, but create=NONE
// against a payload key holding no record is ops.ErrVectorRecordAbsent, which
// the server maps to StatusNotFound (server.TestMapResultVectorRecordAbsentIsNotFound).
// Unlike a missing/expired POINT (TestCollectionOperateMissingPoint's
// found=false/nil/nil convention), this is a StatusNotFound wire response —
// Client.Call's ordinary not-found path — so Operate surfaces it as
// client.ErrNotFound rather than a silent false.
func TestCollectionOperateRecordAbsentIsErrNotFound(t *testing.T) {
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		return StatusNotFound, nil
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")

	a, err := NewOperate(nil).AddT(F("rc"), wire.OperateTypeU32, 1).Args()
	if err != nil {
		t.Fatalf("build operate args: %v", err)
	}
	found, res, _, err := col.Operate(context.Background(), OperateRequest{ID: 1, PayloadKey: "session", Args: a})
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
	if found {
		t.Fatal("found = true, want false")
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}
}

// TestCollectionOperateCheckFailedResult checks that a CHECK-failed result
// decodes as an ordinary (non-error) OperateResult — found=true, err=nil,
// Status=OperateStatusCheckFailed, FailedOp naming the CHECK that aborted the
// list — exactly like Client.Operate's KV counterpart. A failed CHECK is a
// normal outcome of a well-formed call, not a transport or CAS error.
func TestCollectionOperateCheckFailedResult(t *testing.T) {
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		// A failed CHECK still carries the point's CURRENT, unbumped version, so a
		// caller can retry the whole read-modify-write without a re-read.
		payload, err := wire.EncodeVectorOperateResult(true, &wire.OperateResult{
			Status:   wire.OperateStatusCheckFailed,
			FailedOp: 2,
		}, 41)
		if err != nil {
			t.Fatalf("EncodeVectorOperateResult: %v", err)
		}
		return StatusOK, payload
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")

	a, err := NewOperate(nil).
		Check(F("rc"), wire.OperateCmpGE, 999).
		AddT(F("rc"), wire.OperateTypeU32, 1).
		Args()
	if err != nil {
		t.Fatalf("build operate args: %v", err)
	}
	found, res, _, err := col.Operate(context.Background(), OperateRequest{ID: 1, PayloadKey: "session", Args: a})
	if err != nil {
		t.Fatalf("Operate: %v", err)
	}
	if !found {
		t.Fatal("found = false, want true")
	}
	if res.Status != wire.OperateStatusCheckFailed || res.FailedOp != 2 {
		t.Fatalf("res = %+v, want Status=CheckFailed FailedOp=2", res)
	}
}

// TestCollectionOperateCASConflict checks that a wrong ExpectedVersion maps to
// client.ErrVersionConflict through mapWriteErr, exactly like Upsert/Delete.
func TestCollectionOperateCASConflict(t *testing.T) {
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		_, argsBytes := decodeOpFrame(t, body)
		_, _, _, _, expectedVersion, hasExpected, err := wire.DecodeVectorOperateArgs(argsBytes)
		if err != nil {
			t.Fatalf("DecodeVectorOperateArgs: %v", err)
		}
		if !hasExpected || expectedVersion != 999 {
			t.Fatalf("hasExpected=%v expectedVersion=%d, want true/999", hasExpected, expectedVersion)
		}
		return StatusError, encodeErrorMsgFrame("vector: expected_version conflict: mismatch")
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")

	a, err := NewOperate(nil).Dynamic().AddT(F("rc"), wire.OperateTypeU32, 1).Args()
	if err != nil {
		t.Fatalf("build operate args: %v", err)
	}
	_, res, _, err := col.Operate(context.Background(), OperateRequest{
		ID: 1, PayloadKey: "session", Args: a,
		ExpectedVersion: 999, HasExpectedVersion: true,
	})
	if err != ErrVersionConflict {
		t.Fatalf("err = %v, want ErrVersionConflict", err)
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}
}

// TestCollectionOperateNilArgs checks the nil-args shape a caller lands on by
// ignoring OperateBuilder.Args's error: wire.ErrOperateArgs, no panic, no
// network call (the address is a closed port a dial would fail loudly on).
func TestCollectionOperateNilArgs(t *testing.T) {
	c, err := New(Config{Servers: []string{"127.0.0.1:1"}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	col := c.Collection("posts")

	found, res, _, err := col.Operate(context.Background(), OperateRequest{ID: 1, PayloadKey: "session"})
	if !errors.Is(err, wire.ErrOperateArgs) {
		t.Fatalf("err = %v, want wire.ErrOperateArgs", err)
	}
	if found {
		t.Fatal("found = true, want false")
	}
	if res != nil {
		t.Fatalf("res = %+v, want nil", res)
	}
}

// TestVectorOperateIsNonReplayable checks that all three vector_operate op
// names are registered as non-replayable: a blind replay after an ambiguous
// post-commit transport failure would apply every ADD twice, or re-evaluate a
// CHECK against a record the first attempt already changed.
func TestVectorOperateIsNonReplayable(t *testing.T) {
	for _, op := range []string{"vector_operate", "vector_named_operate", "vector_mv_operate"} {
		if !nonReplayableOp(op) {
			t.Errorf("nonReplayableOp(%q) = false, want true", op)
		}
	}
}
