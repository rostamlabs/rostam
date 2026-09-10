// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
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

// errRealMalformedSingle, errRealMalformedBulk and errRealNotRecordDetailed reproduce
// the EXACT shapes checkRecordValues / checkRecordValuesAll (ErrRecordMalformed)
// and currentRecordValue (ErrPayloadKeyNotRecord) actually produce in
// production — see vector.IsRecordMalformedMessage / IsPayloadKeyNotRecordMessage
// for the format strings these mirror. server cannot call those unexported
// vector-package producers directly, so the shapes are reproduced by hand;
// vector's own TestIsRecordMalformedMessage / TestIsPayloadKeyNotRecordMessage
// pin them against the real producers.
var (
	errRealMalformedSingle   = fmt.Errorf("%w: payload key %q: %s", vector.ErrRecordMalformed, "session", "record: malformed record: wire: args too short")
	errRealMalformedBulk     = fmt.Errorf("payload %d: %w", 7, errRealMalformedSingle)
	errRealNotRecordDetailed = fmt.Errorf("%w: payload key %q holds a value of kind %d", vector.ErrPayloadKeyNotRecord, "country", 2)
)

// TestClientFacingErrMalformedRecord pins the TCP transport's half of the same
// classification the HTTP edge makes (httpapi.TestStatusForErrorMalformedRecord).
// vector.ErrRecordMalformed is a caller mistake — bytes the caller sent that no
// operate engine can open — so its message must reach the client verbatim rather
// than being redacted to "internal error", which would leave a client unable to
// tell a bad payload from a server fault.
//
// Per shape: the bare sentinel, the real %w-wrapped single/bulk shapes (matched
// by errors.Is regardless of exact text), and those shapes stringified across
// the Raft boundary (matched only by vector.IsRecordMalformedMessage — a
// clustered apply loses errors.Is identity, which is why clientFacingErr also
// carries a message-shape fallback). A negative control asserts an unrelated
// error that merely wraps the sentinel with a foreign prefix stays redacted —
// the exact bug a bare strings.Contains classifier would reintroduce.
func TestClientFacingErrMalformedRecord(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", vector.ErrRecordMalformed},
		{"wrapped, real single-payload shape", errRealMalformedSingle},
		{"wrapped, real bulk shape", errRealMalformedBulk},
		{"stringified across Raft, real single-payload shape", errors.New(errRealMalformedSingle.Error())},
		{"stringified across Raft, real bulk shape", errors.New(errRealMalformedBulk.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !clientFacingErr(tc.err) {
				t.Error("clientFacingErr(ErrRecordMalformed) = false, want true")
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix (%v, no %w identity)", func(t *testing.T) {
		if clientFacingErr(fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordMalformed)) {
			t.Error("clientFacingErr(wrapped ErrRecordMalformed) = true, want false (redacted)")
		}
	})
}

// TestClientFacingErrRecordTooLarge is ErrRecordMalformed's sibling bound: a
// record value above the storage cap is the same caller-fixable-mistake bucket.
// Per shape: bare sentinel, real %w-wrapped shape, and that shape stringified
// across the Raft boundary, plus a negative control for an unrelated wrap.
func TestClientFacingErrRecordTooLarge(t *testing.T) {
	wrapped := fmt.Errorf("%w: payload key %q holds a 17000000-byte record, the cap is 16777216 bytes", vector.ErrRecordTooLarge, "session")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", vector.ErrRecordTooLarge},
		{"wrapped, real shape", wrapped},
		{"stringified across Raft, real shape", errors.New(wrapped.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !clientFacingErr(tc.err) {
				t.Error("clientFacingErr(ErrRecordTooLarge) = false, want true")
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix (%v, no %w identity)", func(t *testing.T) {
		if clientFacingErr(fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordTooLarge)) {
			t.Error("clientFacingErr(wrapped ErrRecordTooLarge) = true, want false (redacted)")
		}
	})
}

// TestClientFacingErrPayloadKeyNotRecord pins the TCP transport's half of
// httpapi.TestStatusForErrorPayloadKeyNotRecord. vector_operate refuses a
// payload key that holds a plain value rather than a record — replacing it would
// destroy data the caller can still read — and that refusal names the caller's
// own key, so it must reach the client verbatim instead of being redacted to
// "internal error", which would read as a server fault for a caller mistake.
//
// Per shape: bare sentinel, real %w-wrapped detailed shape, and that shape
// stringified across the Raft boundary, plus a negative control for an
// unrelated wrap.
func TestClientFacingErrPayloadKeyNotRecord(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", vector.ErrPayloadKeyNotRecord},
		{"wrapped, real detailed shape", errRealNotRecordDetailed},
		{"stringified across Raft, real detailed shape", errors.New(errRealNotRecordDetailed.Error())},
		{"stringified across Raft, bare shape", errors.New(vector.ErrPayloadKeyNotRecord.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !clientFacingErr(tc.err) {
				t.Error("clientFacingErr(ErrPayloadKeyNotRecord) = false, want true")
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix (%v, no %w identity)", func(t *testing.T) {
		if clientFacingErr(fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrPayloadKeyNotRecord)) {
			t.Error("clientFacingErr(wrapped ErrPayloadKeyNotRecord) = true, want false (redacted)")
		}
	})
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
// ErrVectorRecordAbsent has exactly ONE production shape — bare, never wrapped
// (see ops.IsVectorRecordAbsentMessage's doc). "wrapped" here is a synthetic
// %w wrap exercising errors.Is identity, not a real production shape;
// "stringified" is the bare form re-created with errors.New, the real
// clustered-apply shape. A negative control asserts an unrelated error that
// wraps the sentinel with a foreign prefix (the message-shape matcher's whole
// reason to exist — a bare strings.Contains classifier used to accept this)
// stays redacted.
func TestMapResultVectorRecordAbsentIsNotFound(t *testing.T) {
	disp := &fakeDispatcher{}
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", ops.ErrVectorRecordAbsent},
		{"wrapped (synthetic %w, exercises errors.Is identity — production never wraps this sentinel)",
			fmt.Errorf("shard 3: %w", ops.ErrVectorRecordAbsent)},
		{"stringified across Raft, real bare shape", errors.New(ops.ErrVectorRecordAbsent.Error())},
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
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		status, payload := mapResult(disp, nil, errors.New("apply: "+ops.ErrVectorRecordAbsent.Error()), "")
		if status != StatusError {
			t.Fatalf("status = %d, want StatusError (%d), i.e. redacted, not StatusNotFound", status, StatusError)
		}
		if msg, _ := DecodeErrorPayload(payload); msg != "internal error" {
			t.Fatalf("payload = %q, want the redacted message", msg)
		}
	})
}

// TestMapResultOperateDuringReshardIsClientFacing pins the reshard refusal as a
// client-facing signal on the binary transport. A vector_operate against a
// collection a reshard is dual-writing is REFUSED — the op-list is not
// idempotent, so applying it to both generations would double-count — and the
// whole design rests on the caller learning it should retry after cutover.
// Unclassified, the refusal fell to the redacted internal-error bucket and a
// retryable transient read as a server fault.
//
// StatusError with the verbatim message is the same answer this transport gives
// the other retryable conditions ("no reachable owner" and friends); StatusNotLeader
// is specifically a leader hint and this is not a leadership condition.
//
// A negative control asserts an unrelated fault that merely wraps the refusal
// text with a foreign prefix stays redacted — the exact class a bare
// strings.Contains arm would leak.
func TestMapResultOperateDuringReshardIsClientFacing(t *testing.T) {
	disp := &fakeDispatcher{}
	detailed := ops.OperateDuringReshardErr("docs")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", ops.ErrOperateDuringReshard},
		{"production detailed form (names the collection)", detailed},
		{"wrapped (%w, exercises errors.Is identity)", fmt.Errorf("shard 3: %w", ops.ErrOperateDuringReshard)},
		{"stringified across Raft, bare shape", errors.New(ops.ErrOperateDuringReshard.Error())},
		{"stringified across Raft, detailed shape", errors.New(detailed.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, payload := mapResult(disp, nil, tc.err, "")
			if status != StatusError {
				t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
			}
			msg, _ := DecodeErrorPayload(payload)
			if msg == "internal error" {
				t.Fatalf("the refusal was redacted to %q — the caller is never told to retry", msg)
			}
			if msg != tc.err.Error() {
				t.Fatalf("payload = %q, want the verbatim refusal %q", msg, tc.err.Error())
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the refusal with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + detailed.Error())
		status, payload := mapResult(disp, nil, err, "")
		if status != StatusError {
			t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
		}
		if msg, _ := DecodeErrorPayload(payload); msg != "internal error" {
			t.Fatalf("payload = %q, want the redacted message", msg)
		}
	})
}

// TestMapResultMalformedOperateFrameIsClientFacing pins a malformed operate
// frame as the caller's mistake on the binary transport. wire.ErrOperateArgs is
// what DecodeVectorOperateArgs, ops.checkVectorOperateArgs and the KV
// DecodeOperateArgs raise for args that do not decode, that name a second target,
// or that carry a TTL the op does not own. Unclassified, a client protocol
// mistake read as a server fault and the message was redacted, leaving the caller
// nothing to act on.
//
// KV operate raises the same sentinel from the same decoder, so this one arm
// covers both ops.
func TestMapResultMalformedOperateFrameIsClientFacing(t *testing.T) {
	disp := &fakeDispatcher{}
	// A real production shape, not a hand-written sentinel: a structurally
	// complete frame declaring an EMPTY payload key, which
	// DecodeVectorOperateArgs rejects with ErrOperateArgs.
	frame := append([]byte{4, 'd', 'o', 'c', 's'}, make([]byte, 8)...) // colLen, "docs", id=0
	frame = append(frame, 0, 0)                                        // pkLen = 0
	_, _, _, _, _, _, decErr := wire.DecodeVectorOperateArgs(frame)
	if !errors.Is(decErr, wire.ErrOperateArgs) {
		t.Fatalf("decoder fixture drifted: err = %v, want wire.ErrOperateArgs", decErr)
	}
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", wire.ErrOperateArgs},
		{"wrapped (%w, exercises errors.Is identity)", fmt.Errorf("vector_operate: %w", wire.ErrOperateArgs)},
		{"as the decoder actually returns it", decErr},
		// The clustered shape, and the one the sentinel arm cannot reach: an
		// operate handler decodes inside the FSM apply, so shard.decodePBResult
		// hands the error back rebuilt with errors.New and identity is gone.
		{"stringified across Raft, real bare shape", errors.New(wire.ErrOperateArgs.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, payload := mapResult(disp, nil, tc.err, "")
			if status != StatusError {
				t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
			}
			msg, _ := DecodeErrorPayload(payload)
			if msg == "internal error" {
				t.Fatalf("a malformed frame was redacted to %q — a client protocol mistake reads as a server fault", msg)
			}
			if msg != tc.err.Error() {
				t.Fatalf("payload = %q, want the verbatim message %q", msg, tc.err.Error())
			}
		})
	}
	// Negative control: an unrelated internal fault that merely mentions the
	// sentinel text must STAY redacted. This is what an exact-equality matcher
	// buys over a strings.Contains arm.
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + wire.ErrOperateArgs.Error())
		status, payload := mapResult(disp, nil, err, "")
		if status != StatusError {
			t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
		}
		if msg, _ := DecodeErrorPayload(payload); msg != "internal error" {
			t.Fatalf("payload = %q, want the redacted message", msg)
		}
	})
}

// TestMapResultTruncatedOperateFrameIsClientFacing is the parity guard for
// wire.ErrVectorArgsTruncated, the sentinel DecodeVectorOperateArgs raises for a
// frame SHORTER than the fields it declares — right beside the ErrOperateArgs it
// raises for a complete-but-invalid one.
//
// Both are the caller's framing mistake and both say nothing but "the arguments
// do not decode", so both belong in the same bucket. Left unclassified, a
// truncated frame was redacted to "internal error" and the caller was told a
// server fault had happened.
func TestMapResultTruncatedOperateFrameIsClientFacing(t *testing.T) {
	disp := &fakeDispatcher{}
	// A real production shape: a frame that declares a 4-byte collection name
	// and then simply stops.
	frame := []byte{4, 'd', 'o'}
	_, _, _, _, _, _, decErr := wire.DecodeVectorOperateArgs(frame)
	if !errors.Is(decErr, wire.ErrVectorArgsTruncated) {
		t.Fatalf("decoder fixture drifted: err = %v, want wire.ErrVectorArgsTruncated", decErr)
	}
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", wire.ErrVectorArgsTruncated},
		{"wrapped (%w, exercises errors.Is identity)", fmt.Errorf("vector_operate: %w", wire.ErrVectorArgsTruncated)},
		{"as the decoder actually returns it", decErr},
		{"stringified across Raft, real bare shape", errors.New(wire.ErrVectorArgsTruncated.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, payload := mapResult(disp, nil, tc.err, "")
			if status != StatusError {
				t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
			}
			msg, _ := DecodeErrorPayload(payload)
			if msg == "internal error" {
				t.Fatalf("a truncated frame was redacted to %q — a client protocol mistake reads as a server fault", msg)
			}
			if msg != tc.err.Error() {
				t.Fatalf("payload = %q, want the verbatim message %q", msg, tc.err.Error())
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + wire.ErrVectorArgsTruncated.Error())
		status, payload := mapResult(disp, nil, err, "")
		if status != StatusError {
			t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
		}
		if msg, _ := DecodeErrorPayload(payload); msg != "internal error" {
			t.Fatalf("payload = %q, want the redacted message", msg)
		}
	})
}

// The kv_query refusals must cross the TCP edge unredacted.
//
// This edge is not only the client's: a kv_query fans out to every shard group,
// and a group a node does not host is answered by a PEER through
// __kv_query_shard__ — so a peer's reply passes through here on its way to the
// coordinator that has to classify it. Redacted, a permanent client mistake and
// a retryable "that group is still installing the index" arrive as the same
// opaque string.
func TestClientFacingErrKVQueryFamily(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"invalid filter", ops.ErrKVQueryFilter},
		{"scan not consented", ops.ErrKVQueryScanRequired},
		{"scan budget", ops.ErrKVQueryScanBudget},
		{"no index on this dispatcher", ops.ErrKVIndexUnavailable},
		{"shard unavailable", ops.ErrKVQueryUnavailable},
		{"no such index", kvindex.ErrNoSuchIndex},
		{"index building", kvindex.ErrIndexBuilding},
		{"index changed", kvindex.ErrIndexChanged},
		{"candidate budget", kvindex.ErrCandidateBudget},
		{"bad args", wire.ErrKVQueryArgs},
		{"bad result", wire.ErrKVQueryResult},
		{"truncated args", wire.ErrKVQueryArgsTruncated},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !clientFacingErr(tc.err) {
				t.Errorf("clientFacingErr(%v) = false, want true", tc.err)
			}
			// Wrapped the way the leaf and the coordinator wrap them.
			if !clientFacingErr(fmt.Errorf("shard group 3: %w", tc.err)) {
				t.Errorf("clientFacingErr(wrapped %v) = false, want true", tc.err)
			}
		})
	}
	// And the bucket has not swallowed everything: an internal fault that merely
	// mentions kv_query is still redacted.
	if clientFacingErr(errors.New("wal append failed at /var/lib/rostam/x during kv_query")) {
		t.Error("an internal fault mentioning kv_query was classified client-facing")
	}
}
