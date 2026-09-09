// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// TestMapResultRedactsInternalError covers finding 039: an internal (non-sentinel,
// non-classified) error carries identifiable topology detail; mapResult must NOT
// return it verbatim over the wire — it logs server-side and returns the generic
// "internal error" payload, mirroring the HTTP edge's writeDispatchError redaction.
func TestMapResultRedactsInternalError(t *testing.T) {
	disp := &fakeDispatcher{}
	const secret = "open /var/lib/rostam/shard-7/partition.idx: no such file; leader 10.0.0.9:7001"
	status, payload := mapResult(disp, nil, errors.New(secret), "")
	if status != StatusError {
		t.Fatalf("status = %d, want StatusError", status)
	}
	msg, err := DecodeErrorPayload(payload)
	if err != nil {
		t.Fatalf("DecodeErrorPayload: %v", err)
	}
	if strings.Contains(msg, "shard-7") || strings.Contains(msg, "10.0.0.9") || strings.Contains(msg, "/var/lib") {
		t.Fatalf("internal error payload leaked identifiable detail: %q", msg)
	}
	if msg != "internal error" {
		t.Errorf("redacted payload = %q, want %q", msg, "internal error")
	}
}

// TestMapResultKeepsClientFacingErrors covers the companion contract of finding 039:
// classified client-facing signals keep their descriptive payload. The two existing
// sentinels (NotFound / NotLeader) map to their own status codes, and validation /
// routing signals (e.g. a dimension mismatch, a stringified "not leader") stay
// descriptive rather than being redacted.
func TestMapResultKeepsClientFacingErrors(t *testing.T) {
	disp := &fakeDispatcher{leader: "10.0.0.2:7001"}

	// cache.ErrNotFound → its own status, unchanged.
	if status, _ := mapResult(disp, nil, cache.ErrNotFound, ""); status != StatusNotFound {
		t.Errorf("cache.ErrNotFound status = %d, want StatusNotFound", status)
	}
	// shard.ErrNotLeader → its own status, unchanged.
	if status, _ := mapResult(disp, nil, shard.ErrNotLeader, ""); status != StatusNotLeader {
		t.Errorf("shard.ErrNotLeader status = %d, want StatusNotLeader", status)
	}

	// A validation mistake is a client signal → descriptive text preserved.
	_, payload := mapResult(disp, nil, vector.ErrDimMismatch, "")
	msg, err := DecodeErrorPayload(payload)
	if err != nil {
		t.Fatalf("DecodeErrorPayload: %v", err)
	}
	if msg == "internal error" || !strings.Contains(msg, "does not match") {
		t.Errorf("validation error redacted: got %q, want the descriptive dim-mismatch text", msg)
	}

	// A stringified leadership transient (the clustered path) stays descriptive so
	// the caller can see it is retryable rather than an opaque internal fault.
	_, payload = mapResult(disp, nil, errors.New("shard: not leader for partition 3"), "")
	msg, _ = DecodeErrorPayload(payload)
	if msg == "internal error" || !strings.Contains(msg, "not leader") {
		t.Errorf("leadership transient redacted: got %q", msg)
	}
}

// TestMapResultKeepsUnregisteredOpVisible pins the classification of an
// unregistered op name over the TCP path. cluster.ErrUnknownOp ("cluster: op
// not registered") and shard.ErrOpNotRegistered ("shard: op not registered")
// used to match nothing in clientFacingErr and fell into the catch-all, so the
// caller got the opaque "internal error". That hid the real cause in
// diagnostics AND stopped the client from rotating: a plain StatusError that is
// neither NotLeader nor a transport error is returned immediately, so a server
// that has not yet applied a dynamic WASM registration ended the call instead of
// deferring to a peer that had.
//
// The message is safe to disclose — it only echoes back the op name the caller
// already sent, with no path, shard id, or host address in it.
func TestMapResultKeepsUnregisteredOpVisible(t *testing.T) {
	disp := &fakeDispatcher{}
	for _, err := range []error{
		shard.ErrOpNotRegistered,
		// cluster.ErrUnknownOp's text; the cluster package cannot be imported
		// here, and the clustered path stringifies it across the Raft boundary
		// anyway.
		errors.New("cluster: op not registered"),
		// cluster.ErrWASMOpNotInThisGroup: the op EXISTS here, but this node will
		// not propose an invocation into the target shard group's log until it
		// knows that log carries the registration. Transient and retryable, so it
		// must reach the client with its text intact — redacting it would leave
		// the caller unable to tell a wait-and-retry from a server fault.
		errors.New(`cluster: op not registered in this shard group yet: op "wasm_incr", shard group 2`),
	} {
		status, payload := mapResult(disp, nil, err, "")
		if status != StatusError {
			t.Fatalf("%v: status = %d, want StatusError", err, status)
		}
		msg, decErr := DecodeErrorPayload(payload)
		if decErr != nil {
			t.Fatalf("DecodeErrorPayload: %v", decErr)
		}
		if msg == "internal error" {
			t.Errorf("%v: redacted to the generic payload; the caller cannot tell an unknown op from a server fault", err)
		}
		if !strings.Contains(msg, "op not registered") {
			t.Errorf("%v: payload = %q, want the descriptive text", err, msg)
		}
	}
}

// TestMapResultKeepsWASMUpdateRefusalVisible pins cluster.ErrWASMUpdateUnsupported
// over the TCP path. Updating a live WASM module is an unsupported operation, and
// the refusal carries the one thing the caller can act on — register the new
// module under a new op name. Redacted to "internal error" it would read as a
// server fault and the caller would retry the same unsupported call forever.
//
// The message discloses only the op name the caller already sent, the node id it
// addressed, and the installed epoch: no path, no shard id, no host address.
func TestMapResultKeepsWASMUpdateRefusalVisible(t *testing.T) {
	disp := &fakeDispatcher{}
	// cluster.ErrWASMUpdateUnsupported's text, rebuilt from the const the sentinel
	// itself is built from: the cluster package cannot be imported here, and the
	// clustered path stringifies the error across the Raft boundary anyway, so the
	// const is the only compile-time link between the refusal and this classifier.
	err := errors.New(`cluster: ` + ops.WASMUpdateUnsupportedMsg + `: op "wasm_incr" is already registered on node n1 (installed epoch 1) and this registration differs from it; register the new module under a NEW op name instead`)
	status, payload := mapResult(disp, nil, err, "")
	if status != StatusError {
		t.Fatalf("status = %d, want StatusError", status)
	}
	msg, decErr := DecodeErrorPayload(payload)
	if decErr != nil {
		t.Fatalf("DecodeErrorPayload: %v", decErr)
	}
	if msg == "internal error" {
		t.Fatalf("redacted to the generic payload; the caller cannot tell an unsupported operation from a server fault")
	}
	if !strings.Contains(msg, "NEW op name") {
		t.Errorf("payload = %q, want the remedy to survive", msg)
	}
}

// TestMapResultKeepsRecordTooLargeVisible pins vector.ErrRecordTooLarge as a
// client-facing signal on the binary transport, in every shape it can arrive
// in. The sentinel arm was already classified; the message-shape arm is the
// one that recognizes the STRINGIFIED forms — shard.decodePBResult rebuilds
// an op error with errors.New(string(payload)) across replication, so
// errors.Is stops matching and the caller got the redacted "internal error"
// for a mistake they could have fixed.
//
// The message-shape arm is vector.IsRecordTooLargeMessage, an exact-shape
// matcher, NOT a bare strings.Contains — the negative controls below assert
// that distinction: an unrelated internal error whose message merely contains
// the sentinel text must still be redacted.
func TestMapResultKeepsRecordTooLargeVisible(t *testing.T) {
	disp := &fakeDispatcher{}
	single := fmt.Errorf("%w: payload key %q holds a 20000000-byte record, the cap is 16777216 bytes",
		vector.ErrRecordTooLarge, "session")
	bulk := fmt.Errorf("payload %d: %w", 3, single)
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", vector.ErrRecordTooLarge},
		{"single-payload form (%w wrapped)", single},
		{"bulk form (%w wrapped, real shape checkRecordValuesAll produces)", bulk},
		{"single-payload form, stringified across replication", errors.New(single.Error())},
		{"bulk form, stringified across replication", errors.New(bulk.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, payload := mapResult(disp, nil, tc.err, "")
			msg, err := DecodeErrorPayload(payload)
			if err != nil {
				t.Fatalf("DecodeErrorPayload: %v", err)
			}
			if msg == "internal error" || !strings.Contains(msg, "exceeds the storage cap") {
				t.Errorf("record-too-large redacted: got %q", msg)
			}
		})
	}
	// Negative controls: an unrelated fault is still redacted, including one
	// that merely CONTAINS the sentinel text inside an unrelated wrapper — the
	// exact bug a bare strings.Contains classifier would reintroduce.
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"unrelated fault, no sentinel text", errors.New("open /var/lib/rostam/shard-7: no such file")},
		{"unrelated fault wrapping the sentinel (%v)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordTooLarge)},
		{"unrelated fault stringified across replication",
			errors.New(fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordTooLarge).Error())},
	} {
		t.Run("negative/"+tc.name, func(t *testing.T) {
			_, payload := mapResult(disp, nil, tc.err, "")
			if msg, _ := DecodeErrorPayload(payload); msg != "internal error" {
				t.Errorf("unrelated fault = %q, want the redacted message", msg)
			}
		})
	}
}

// TestClientFacingErrMalformedRecord pins the TCP transport's half of the same
// classification the HTTP edge makes (httpapi.TestStatusForErrorMalformedRecord).
// vector.ErrRecordMalformed is a caller mistake — bytes the caller sent that no
// operate engine can open — so its message must reach the client verbatim rather
// than being redacted to "internal error", which would leave a client unable to
// tell a bad payload from a server fault.
func TestClientFacingErrMalformedRecord(t *testing.T) {
	err := fmt.Errorf("%w: payload key %q: bad", vector.ErrRecordMalformed, "session")
	if !clientFacingErr(err) {
		t.Error("clientFacingErr(ErrRecordMalformed) = false, want true")
	}
	if !clientFacingErr(vector.ErrRecordTooLarge) {
		t.Error("clientFacingErr(ErrRecordTooLarge) = false, want true (the sibling bound)")
	}
}

// TestClientFacingErrPayloadKeyNotRecord pins the TCP transport's half of
// httpapi.TestStatusForErrorPayloadKeyNotRecord. vector_operate refuses a
// payload key that holds a plain value rather than a record — replacing it would
// destroy data the caller can still read — and that refusal names the caller's
// own key, so it must reach the client verbatim instead of being redacted to
// "internal error", which would read as a server fault for a caller mistake.
func TestClientFacingErrPayloadKeyNotRecord(t *testing.T) {
	err := fmt.Errorf("%w: payload key %q holds a value of kind %d", vector.ErrPayloadKeyNotRecord, "country", 2)
	if !clientFacingErr(err) {
		t.Error("clientFacingErr(ErrPayloadKeyNotRecord) = false, want true")
	}
}

// TestMapResultVectorRecordAbsentIsNotFound pins vector_operate's create=NONE
// refusal as the SAME wire status KV operate answers with. handleOperate maps
// its create=NONE-against-an-absent-key case to cache.ErrNotFound, which
// mapResult already answers StatusNotFound for (TestOperateCreateNoneOnAbsent
// pins the KV half); ops.ErrVectorRecordAbsent is that signal for a record held
// in a point's payload, so the two transports must not disagree about it. Left
// unclassified it fell through to the redacted StatusError bucket, telling a
// caller "internal error" for a record that simply is not there.
//
// The substring arm covers the clustered path, where the sentinel is stringified
// across the Raft boundary and errors.Is stops matching — the same reason
// clientFacingErr carries string fallbacks.
func TestMapResultVectorRecordAbsentIsNotFound(t *testing.T) {
	disp := &fakeDispatcher{}
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", ops.ErrVectorRecordAbsent},
		{"wrapped", fmt.Errorf("shard 3: %w", ops.ErrVectorRecordAbsent)},
		{"stringified across Raft", errors.New("apply: " + ops.ErrVectorRecordAbsent.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, payload := mapResult(disp, nil, tc.err, "")
			if status != StatusNotFound {
				t.Fatalf("status = %d, want StatusNotFound (%d)", status, StatusNotFound)
			}
			if payload != nil {
				t.Fatalf("payload = %q, want nil (the not-found status carries no body, as for cache.ErrNotFound)", payload)
			}
		})
	}
}
