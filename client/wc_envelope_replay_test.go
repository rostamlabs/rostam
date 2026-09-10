// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"encoding/binary"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// encodeWCEnvelopeFrame builds a write-consistency envelope the way the root
// store's wcWire does (ops.EncodeWCEnvelope):
//
//	[wcf u8][wait u8][nameLen u8][name...][argsLen u32][args...]
//
// It is hand-built because ops lives in the root module and this one cannot
// import it. The two are pinned against each other by
// TestWCEnvelopeInnerOpMatchesTheStoreWire in the root module, which encodes
// with the real producer and reads the name back with wire.WCEnvelopeInnerOp.
func encodeWCEnvelopeFrame(wcf, wait uint8, innerName string, innerArgs []byte) []byte {
	out := make([]byte, 0, 3+len(innerName)+4+len(innerArgs))
	out = append(out, wcf, wait, byte(len(innerName)))
	out = append(out, innerName...)
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(innerArgs))) //nolint:gosec // test-only
	out = append(out, hdr[:]...)
	return append(out, innerArgs...)
}

// TestWCEnvelopedNonReplayableOpNotReplayed is the regression for the envelope
// bypass: a write issued with write-consistency options travels under the name
// wire.WCEnvelopeOp, so the retry guard used to answer "replayable" for every
// one of them. Server A takes the request and drops the connection (the
// committed-but-reply-lost shape → ambiguous EOF); server B must never be
// contacted, because replaying a committed vector_operate ADD double-counts.
func TestWCEnvelopedNonReplayableOpNotReplayed(t *testing.T) {
	inner := []byte("inner-args")
	for _, op := range []string{"vector_operate", "vector_named_operate", "vector_mv_operate", "operate", "cas"} {
		t.Run(op, func(t *testing.T) {
			var callA, callB atomic.Int32
			addrA, stopA := startDropAfterReadServer(t, func() { callA.Add(1) })
			defer stopA()
			addrB, stopB := startFakeServer(t, func(_ []byte) (uint8, []byte) {
				callB.Add(1)
				return StatusOK, []byte{0}
			})
			defer stopB()

			c, err := New(Config{Servers: []string{addrA, addrB}})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = c.Close() }()

			_, err = c.Call(context.Background(), wire.WCEnvelopeOp, encodeWCEnvelopeFrame(2, 1, op, inner))
			if err == nil {
				t.Fatalf("%s under the envelope returned nil; want an ambiguous transport error", op)
			}
			if !isAmbiguous(err) {
				t.Fatalf("%s under the envelope: err = %v; want an ambiguous (post-transmission) error", op, err)
			}
			if n := callA.Load(); n != 1 {
				t.Fatalf("%s: server A calls = %d, want 1", op, n)
			}
			if n := callB.Load(); n != 0 {
				t.Fatalf("%s: server B calls = %d, want 0 (an enveloped non-replayable write must not replay)", op, n)
			}
		})
	}
}

// TestWCEnvelopedReplayableOpStillReplays is the negative control: the guard
// must look through the envelope, not blanket-refuse everything wearing it. An
// idempotent inner op still fails over to server B.
func TestWCEnvelopedReplayableOpStillReplays(t *testing.T) {
	var callB atomic.Int32
	addrA, stopA := startDropAfterReadServer(t, nil)
	defer stopA()
	addrB, stopB := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		callB.Add(1)
		return StatusOK, []byte("ok")
	})
	defer stopB()

	c, err := New(Config{Servers: []string{addrA, addrB}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	res, err := c.Call(context.Background(), wire.WCEnvelopeOp, encodeWCEnvelopeFrame(2, 1, "vector_insert", []byte("args")))
	if err != nil {
		t.Fatalf("enveloped vector_insert: %v; want transparent replay onto B", err)
	}
	if string(res) != "ok" {
		t.Fatalf("result = %q, want ok", res)
	}
	if n := callB.Load(); n != 1 {
		t.Fatalf("server B calls = %d, want 1 (idempotent inner op must still replay)", n)
	}
}

// TestNonReplayableCallThroughEnvelope is the unit table for the guard itself,
// including the frames wire.WCEnvelopeInnerOp cannot read. An undecodable
// envelope is treated as non-replayable: this client never builds one, so
// declining costs at most a retry while guessing costs a double-applied write.
func TestNonReplayableCallThroughEnvelope(t *testing.T) {
	cases := []struct {
		name string
		op   string
		args []byte
		want bool
	}{
		{"plain non-replayable", "vector_operate", nil, true},
		{"plain replayable", "vector_get", nil, false},
		{"envelope wrapping vector_operate", wire.WCEnvelopeOp, encodeWCEnvelopeFrame(1, 1, "vector_operate", nil), true},
		{"envelope wrapping vector_named_operate", wire.WCEnvelopeOp, encodeWCEnvelopeFrame(1, 1, "vector_named_operate", nil), true},
		{"envelope wrapping vector_mv_operate", wire.WCEnvelopeOp, encodeWCEnvelopeFrame(1, 1, "vector_mv_operate", nil), true},
		{"envelope wrapping operate", wire.WCEnvelopeOp, encodeWCEnvelopeFrame(1, 1, "operate", nil), true},
		{"envelope wrapping an idempotent op", wire.WCEnvelopeOp, encodeWCEnvelopeFrame(1, 1, "vector_insert", nil), false},
		// A NESTED envelope: the fanout dispatcher unwraps and dispatches what
		// is inside, so the frame still reaches a real write. This client never
		// builds one, so it is declined rather than followed.
		{"envelope wrapping another envelope around vector_operate", wire.WCEnvelopeOp,
			encodeWCEnvelopeFrame(1, 1, wire.WCEnvelopeOp,
				encodeWCEnvelopeFrame(1, 1, "vector_operate", nil)), true},
		{"envelope wrapping another envelope around an idempotent op", wire.WCEnvelopeOp,
			encodeWCEnvelopeFrame(1, 1, wire.WCEnvelopeOp,
				encodeWCEnvelopeFrame(1, 1, "vector_insert", nil)), true},
		{"envelope too short for a header", wire.WCEnvelopeOp, []byte{1, 1}, true},
		{"envelope truncated inside the name", wire.WCEnvelopeOp, []byte{1, 1, 8, 'v', 'e', 'c'}, true},
		{"empty args under the envelope name", wire.WCEnvelopeOp, nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := nonReplayableCall(tc.op, tc.args); got != tc.want {
				t.Fatalf("nonReplayableCall(%q) = %v, want %v", tc.op, got, tc.want)
			}
		})
	}
}

// TestNestedWCEnvelopeNotReplayed is the end-to-end half of the nested case: the
// server's fanout dispatcher unwraps an envelope and dispatches what is inside,
// so a doubly-wrapped vector_operate still reaches the real write. Server A takes
// the request and drops the connection; server B must never be contacted.
func TestNestedWCEnvelopeNotReplayed(t *testing.T) {
	var callB atomic.Int32
	addrA, stopA := startDropAfterReadServer(t, nil)
	defer stopA()
	addrB, stopB := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		callB.Add(1)
		return StatusOK, []byte{0}
	})
	defer stopB()

	c, err := New(Config{Servers: []string{addrA, addrB}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	nested := encodeWCEnvelopeFrame(2, 1, wire.WCEnvelopeOp,
		encodeWCEnvelopeFrame(2, 1, "vector_operate", []byte("inner-args")))
	if _, err := c.Call(context.Background(), wire.WCEnvelopeOp, nested); err == nil {
		t.Fatal("a doubly-wrapped vector_operate returned nil; want an ambiguous transport error")
	} else if !isAmbiguous(err) {
		t.Fatalf("err = %v; want an ambiguous (post-transmission) error", err)
	}
	if n := callB.Load(); n != 0 {
		t.Fatalf("server B calls = %d, want 0 (a nested envelope must not replay)", n)
	}
}
