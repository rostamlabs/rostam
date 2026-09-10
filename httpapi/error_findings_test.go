// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// TestStatusForErrorConflictAndBackpressure covers finding 019: the four
// collection-level outcomes that previously fell through to the 500 default.
// ErrCollectionExists / ErrDuplicateID are routine create-conflicts (409); the
// quota/rate-limit refusals are backpressure (429). Both the sentinel path and the
// clustered/stringified fallback path are exercised.
func TestStatusForErrorConflictAndBackpressure(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"exists-sentinel", vector.ErrCollectionExists, http.StatusConflict},
		{"dupid-sentinel", vector.ErrDuplicateID, http.StatusConflict},
		{"ratelimited-sentinel", vector.ErrCollectionRateLimited, http.StatusTooManyRequests},
		{"full-sentinel", vector.ErrCollectionFull, http.StatusTooManyRequests},
		// Clustered path: the sentinel is stringified across the Raft boundary, so
		// only the message survives — the string fallbacks must still classify it.
		{"exists-string", errors.New("rostam: collection already exists"), http.StatusConflict},
		{"dupid-string", errors.New("rostam: id already present (delete first)"), http.StatusConflict},
		{"ratelimited-string", errors.New("rostam: collection insert rate limited"), http.StatusTooManyRequests},
		{"full-string", errors.New("rostam: collection full (quota exceeded)"), http.StatusTooManyRequests},
	}
	for _, tc := range cases {
		if got := statusForError(tc.err); got != tc.want {
			t.Errorf("%s: statusForError = %d, want %d", tc.name, got, tc.want)
		}
	}
	// The mapped statuses are non-500, so clientError must pass the descriptive
	// message through verbatim (it is a client signal, not a topology leak) — never
	// the redacted "internal error".
	for _, tc := range cases {
		status, msg := clientError("vector_insert", tc.err)
		if status == http.StatusInternalServerError || msg == "internal error" {
			t.Errorf("%s: clientError redacted a client-facing error: status=%d msg=%q", tc.name, status, msg)
		}
	}
}

// TestStatusForErrorNoShardOwner covers finding 021 (HTTP side): cluster's
// no-reachable-owner-for-shard is an ownership transient, retryable → 503, not the
// 500 default.
func TestStatusForErrorNoShardOwner(t *testing.T) {
	if got := statusForError(errors.New("cluster: no reachable owner for shard")); got != http.StatusServiceUnavailable {
		t.Fatalf("statusForError(no reachable owner) = %d, want 503", got)
	}
}

// TestHTTPDoubleCreateAndDuplicateInsertConflict covers finding 019 end-to-end: a
// second create of a live collection and a default (upsert=false) insert of a live
// id both surface as 409 Conflict with a DESCRIPTIVE body (no longer a redacted
// 500).
func TestHTTPDoubleCreateAndDuplicateInsertConflict(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	create := `{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`
	if rec := do(t, h, "POST", "/v1/collections", create, nil); rec.Code != http.StatusCreated {
		t.Fatalf("first create = %d (%s)", rec.Code, rec.Body)
	}
	rec := do(t, h, "POST", "/v1/collections", create, nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("double create = %d, want 409 (%s)", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "already exists") {
		t.Errorf("double-create body redacted/unexpected: %s", rec.Body)
	}

	// Default insert of id 1, then a second default insert of the SAME id → 409.
	pt := `{"id":1,"vector":[1,2,3]}`
	if rec := do(t, h, "POST", "/v1/collections/docs/points", pt, nil); rec.Code != http.StatusOK {
		t.Fatalf("first insert = %d (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points", pt, nil)
	if rec.Code != http.StatusConflict {
		t.Fatalf("duplicate insert = %d, want 409 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPBatchRejectsWriteConsistency covers finding 012: the batch route must
// reject (400) a body carrying a per-point write_consistency_factor>0 or wait=false
// rather than silently accepting it and downgrading the durability to Raft majority.
func TestHTTPBatchRejectsWriteConsistency(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	if rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	// Per-point write_consistency_factor > 0 → 400 (not a silent 200).
	wcf := `{"points":[{"id":1,"vector":[1,2,3]},{"id":2,"vector":[1,2,3],"write_consistency_factor":3}]}`
	rec := do(t, h, "POST", "/v1/collections/docs/points/batch", wcf, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("batch with wcf = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "write_consistency_factor") {
		t.Errorf("400 body should name the offending field: %s", rec.Body)
	}
	// wait=false → 400 too.
	nowait := `{"points":[{"id":1,"vector":[1,2,3],"wait":false}]}`
	if rec := do(t, h, "POST", "/v1/collections/docs/points/batch", nowait, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("batch with wait=false = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// Sanity: a plain batch (no WCF fields) still succeeds.
	ok := `{"points":[{"id":1,"vector":[1,2,3]},{"id":2,"vector":[1,2,3]}]}`
	if rec := do(t, h, "POST", "/v1/collections/docs/points/batch", ok, nil); rec.Code != http.StatusOK {
		t.Fatalf("plain batch = %d, want 200 (%s)", rec.Code, rec.Body)
	}
}

// failingBatchDispatcher fails the Call at index failAt with failErr; all other
// calls succeed with an empty body. It drives putPointsBatch's mid-batch failure
// paths deterministically without needing to provoke a real internal fault.
type failingBatchDispatcher struct {
	calls   int
	failAt  int
	failErr error
}

func (d *failingBatchDispatcher) Call(string, []byte) ([]byte, error) {
	i := d.calls
	d.calls++
	if i == d.failAt {
		return nil, d.failErr
	}
	return nil, nil
}
func (d *failingBatchDispatcher) LeaderAddr() string { return "" }

// TestPutPointsBatchRedactsInternalError covers finding 033: a mid-batch error that
// maps to 500 must be redacted to the generic "internal error" (logged server-side)
// while still reporting how many points committed — matching the single-point path
// (writeDispatchError) — whereas a 4xx keeps its descriptive message.
func TestPutPointsBatchRedactsInternalError(t *testing.T) {
	const secret = "boom at /var/lib/rostam/shard-7 leader 10.0.0.9:7001"
	body := `{"points":[{"id":1,"vector":[1,2,3]},{"id":2,"vector":[1,2,3]},{"id":3,"vector":[1,2,3]}]}`

	// Internal error mid-batch (fails on the 2nd point, index 1) → 500 redacted.
	disp := &failingBatchDispatcher{failAt: 1, failErr: errors.New(secret)}
	h := Handler(disp, Options{})
	var out map[string]any
	rec := do(t, h, "POST", "/v1/collections/docs/points/batch", body, &out)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("internal mid-batch = %d, want 500 (%s)", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), "shard-7") || strings.Contains(rec.Body.String(), "10.0.0.9") {
		t.Fatalf("500 body leaked internal detail: %s", rec.Body)
	}
	if out["error"] != "internal error" {
		t.Errorf("500 error = %v, want %q", out["error"], "internal error")
	}
	if got, ok := out["committed"].(float64); !ok || got != 1 {
		t.Errorf("committed = %v, want 1", out["committed"])
	}

	// 4xx mid-batch (ErrDimMismatch) → descriptive message preserved, still reports
	// committed.
	disp = &failingBatchDispatcher{failAt: 1, failErr: vector.ErrDimMismatch}
	h = Handler(disp, Options{})
	out = nil
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch", body, &out)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("4xx mid-batch = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if msg, _ := out["error"].(string); !strings.Contains(msg, "does not match") {
		t.Errorf("400 error = %q, want the descriptive dim-mismatch message", msg)
	}
	if got, ok := out["committed"].(float64); !ok || got != 1 {
		t.Errorf("committed = %v, want 1", out["committed"])
	}
}

// TestStatusForErrorUnregisteredOp keeps this classifier in sync with
// server.clientFacingErr (the contract the comment above statusForError
// demands): an op name the server does not have registered is a client mistake
// — or a dynamic WASM registration this node has not applied yet — not a server
// fault. It maps to 404 with the message intact, rather than falling through to
// the 500 default where clientError redacts it to "internal error". The text is
// safe to disclose: it echoes only the op name the caller already sent.
func TestStatusForErrorUnregisteredOp(t *testing.T) {
	for _, err := range []error{
		errors.New("cluster: op not registered"), // cluster.ErrUnknownOp
		errors.New("shard: op not registered"),   // shard.ErrOpNotRegistered
	} {
		if got := statusForError(err); got != http.StatusNotFound {
			t.Errorf("statusForError(%v) = %d, want 404", err, got)
		}
		status, msg := clientError("wasm_incr", err)
		if status == http.StatusInternalServerError || msg == "internal error" {
			t.Errorf("clientError(%v) redacted an unknown-op signal: status=%d msg=%q", err, status, msg)
		}
		if !strings.Contains(msg, "op not registered") {
			t.Errorf("clientError(%v) msg = %q, want the descriptive text", err, msg)
		}
	}
}

// TestStatusForErrorWASMOpNotInThisGroup pins the distinction between "this
// server does not have that op" (404) and "this server has it but declines to
// propose an invocation into the target shard group's log until it knows that
// log carries the registration" (503, cluster.ErrWASMOpNotInThisGroup).
//
// The second is transient by construction — the registration is still fanning
// out to the remaining groups — so it has to be retryable. Its message CONTAINS
// the "op not registered" substring the 404 case matches, so the ordering of the
// two cases in statusForError is what makes this work; the test fails if they are
// ever reordered.
func TestStatusForErrorWASMOpNotInThisGroup(t *testing.T) {
	err := errors.New(`cluster: op not registered in this shard group yet: op "wasm_incr", shard group 2 on node n1`)
	if got := statusForError(err); got != http.StatusServiceUnavailable {
		t.Errorf("statusForError = %d, want 503 (retryable), not %d", got, http.StatusNotFound)
	}
	status, msg := clientError("wasm_incr", err)
	if status == http.StatusInternalServerError || msg == "internal error" {
		t.Errorf("clientError redacted the route-gate signal: status=%d msg=%q", status, msg)
	}
}

// TestStatusForErrorWASMUpdateUnsupported pins the third member of this family,
// and it is the one that must NOT look retryable. cluster.ErrWASMUpdateUnsupported
// refuses a __register_wasm__ that would change a live module in place: updating
// a WASM module is an unsupported operation, so the caller's remedy is to register
// under a new name, not to wait. 400, not the 503 the route-gate transient gets
// and not the 500 default that would redact the remedy away.
func TestStatusForErrorWASMUpdateUnsupported(t *testing.T) {
	// Built from the same const cluster.ErrWASMUpdateUnsupported is built from, so
	// a rewording breaks the build rather than silently re-redacting the refusal.
	err := errors.New(`cluster: ` + ops.WASMUpdateUnsupportedMsg + `: op "wasm_incr" is already registered on node n1 (installed epoch 1) and this registration differs from it; register the new module under a NEW op name instead`)
	if got := statusForError(err); got != http.StatusBadRequest {
		t.Errorf("statusForError = %d, want 400 (client mistake with a remedy, not a transient)", got)
	}
	status, msg := clientError("wasm_incr", err)
	if status == http.StatusInternalServerError || msg == "internal error" {
		t.Errorf("clientError redacted the refusal: status=%d msg=%q", status, msg)
	}
	if !strings.Contains(msg, "NEW op name") {
		t.Errorf("clientError msg = %q, want the remedy to survive", msg)
	}
}

// TestStatusForErrorAlreadyPartitioned pins that re-creating a PARTITIONED
// collection is a 409, not a 500.
//
// "already exists" was already mapped to 409, but the partitioned-create path
// reports "is already partitioned" instead (embedded.go), which matched nothing
// and fell through to the 500 default. That default also REDACTS the message, so
// the caller got an opaque {"error":"internal error"} while the real reason
// appeared only in the server log — indistinguishable from a genuine fault.
//
// It is the same routine create-conflict as any other re-create, and it is what
// a caller hits by re-running a quickstart script. It cost two invalidated
// benchmark runs before being diagnosed, because a client cannot tell this
// apart from a real server error.
func TestStatusForErrorAlreadyPartitioned(t *testing.T) {
	err := errors.New(`vector: collection "bench" is already partitioned`)
	if got := statusForError(err); got != http.StatusConflict {
		t.Fatalf("statusForError(already partitioned) = %d, want %d (409 Conflict)",
			got, http.StatusConflict)
	}
}

// TestStatusForErrorRecordTooLarge pins the HTTP status for an oversize record
// payload at 400 in every shape the sentinel can arrive in. The sentinel arm
// was already classified; the message-shape arm (vector.IsRecordTooLargeMessage)
// is what recognizes the STRINGIFIED forms — the shape a clustered write
// produces (shard.decodePBResult rebuilds an op error with errors.New across
// replication, so errors.Is stops matching) — and unclassified it fell through
// to the redacted 500 bucket, a server fault for a mistake the caller can fix
// by sending a smaller record.
//
// The message-shape arm is an exact-shape matcher, not a bare strings.Contains:
// the negative controls assert that an unrelated internal error whose message
// merely contains the sentinel text is still redacted as a 500.
func TestStatusForErrorRecordTooLarge(t *testing.T) {
	single := fmt.Errorf("%w: payload key %q holds a 20000000-byte record, the cap is 16777216 bytes",
		vector.ErrRecordTooLarge, "session")
	bulk := fmt.Errorf("payload %d: %w", 3, single)
	cases := []struct {
		name string
		err  error
	}{
		{"sentinel", vector.ErrRecordTooLarge},
		{"single-payload form (%w wrapped)", single},
		{"bulk form (%w wrapped, real shape checkRecordValuesAll produces)", bulk},
		{"single-payload form, stringified across replication", errors.New(single.Error())},
		{"bulk form, stringified across replication", errors.New(bulk.Error())},
	}
	for _, tc := range cases {
		if got := statusForError(tc.err); got != http.StatusBadRequest {
			t.Errorf("%s: statusForError = %d, want 400", tc.name, got)
		}
		// A 4xx must keep its descriptive message: it names only the caller's own
		// payload key and the two sizes.
		status, msg := clientError("vector_insert", tc.err)
		if status == http.StatusInternalServerError || msg == "internal error" {
			t.Errorf("%s: clientError redacted a client-facing error: status=%d msg=%q", tc.name, status, msg)
		}
		if !strings.Contains(msg, "exceeds the storage cap") {
			t.Errorf("%s: message lost the cap text: %q", tc.name, msg)
		}
	}
	// Negative controls: an unrelated fault is still a redacted 500, including
	// one that merely CONTAINS the sentinel text inside an unrelated wrapper —
	// the exact bug a bare strings.Contains classifier would reintroduce.
	negatives := []struct {
		name string
		err  error
	}{
		{"unrelated fault, no sentinel text", errors.New("open /var/lib/rostam/shard-7: no such file")},
		{"unrelated fault wrapping the sentinel (%v)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordTooLarge)},
		{"unrelated fault stringified across replication",
			errors.New(fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordTooLarge).Error())},
	}
	for _, tc := range negatives {
		if got := statusForError(tc.err); got != http.StatusInternalServerError {
			t.Errorf("%s: statusForError = %d, want 500", tc.name, got)
		}
	}
}

// errRealMalformedSingle, errRealMalformedBulk and errRealNotRecordDetailed reproduce
// the EXACT shapes checkRecordValues / checkRecordValuesAll (ErrRecordMalformed)
// and currentRecordValue (ErrPayloadKeyNotRecord) actually produce in
// production — see vector.IsRecordMalformedMessage / IsPayloadKeyNotRecordMessage
// for the format strings these mirror. httpapi cannot call those unexported
// vector-package producers directly, so the shapes are reproduced by hand;
// vector's own TestIsRecordMalformedMessage / TestIsPayloadKeyNotRecordMessage
// pin them against the real producers.
var (
	errRealMalformedSingle   = fmt.Errorf("%w: payload key %q: %s", vector.ErrRecordMalformed, "session", "record: malformed record: wire: args too short")
	errRealMalformedBulk     = fmt.Errorf("payload %d: %w", 7, errRealMalformedSingle)
	errRealNotRecordDetailed = fmt.Errorf("%w: payload key %q holds a value of kind %d", vector.ErrPayloadKeyNotRecord, "country", 2)
)

// TestStatusForErrorMalformedRecord pins the ingest gate's shape rejection as a
// 400. vector.ErrRecordMalformed is what every wire-reachable entry returns for
// record bytes no operate engine can open, and it arrived with the phase-2
// ingest validation — a sentinel that matched nothing here would have surfaced a
// caller's own bad bytes as an opaque 500 with the message redacted, so this
// keeps it in the same bucket as ErrRecordTooLarge and in sync with
// server.clientFacingErr.
//
// Per shape: the bare sentinel, the real %w-wrapped single/bulk shapes (matched
// by errors.Is regardless of exact text), and those shapes stringified across
// the Raft boundary (matched only by vector.IsRecordMalformedMessage, since
// shard.decodePBResult rebuilds a replicated op error with errors.New and
// errors.Is stops matching there). A negative control asserts an unrelated
// error that merely wraps the sentinel with a foreign prefix stays a redacted
// 500 — the exact bug a bare strings.Contains classifier would reintroduce.
func TestStatusForErrorMalformedRecord(t *testing.T) {
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
			if got := statusForError(tc.err); got != http.StatusBadRequest {
				t.Errorf("statusForError(ErrRecordMalformed) = %d, want 400", got)
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix (%v, no %w identity)", func(t *testing.T) {
		err := fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordMalformed)
		if got := statusForError(err); got != http.StatusInternalServerError {
			t.Errorf("statusForError(%v) = %d, want 500", err, got)
		}
	})
}

// TestStatusForErrorPayloadKeyNotRecord pins vector_operate's "that key does not
// hold a record" refusal as a 400. The op deliberately refuses rather than
// overwriting a plain value with a record, so the caller has a key to fix; an
// unmatched sentinel here would have surfaced that as an opaque, redacted 500.
// Kept in sync with server.clientFacingErr.
//
// Per shape: the bare sentinel, the real %w-wrapped detailed shape, and that
// shape stringified across the Raft boundary. A negative control asserts an
// unrelated error that merely wraps the sentinel with a foreign prefix stays a
// redacted 500.
func TestStatusForErrorPayloadKeyNotRecord(t *testing.T) {
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
			if got := statusForError(tc.err); got != http.StatusBadRequest {
				t.Errorf("statusForError(ErrPayloadKeyNotRecord) = %d, want 400", got)
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix (%v, no %w identity)", func(t *testing.T) {
		err := fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrPayloadKeyNotRecord)
		if got := statusForError(err); got != http.StatusInternalServerError {
			t.Errorf("statusForError(%v) = %d, want 500", err, got)
		}
	})
}

// TestStatusForErrorVectorRecordAbsent pins vector_operate's create=NONE refusal
// as a 404, the HTTP rendering of the not-found status the binary transport
// answers for the same signal (server.TestMapResultVectorRecordAbsentIsNotFound).
// The caller declined to create a record and there was none: the thing they
// named does not exist, which is what 404 means — not the redacted 500 an
// unclassified sentinel would have produced.
//
// ErrVectorRecordAbsent has exactly ONE production shape — bare, never wrapped
// (see ops.IsVectorRecordAbsentMessage's doc). "wrapped" here is a synthetic
// %w wrap exercising errors.Is identity, not a real production shape;
// "stringified" is the bare form re-created with errors.New, the real
// clustered-apply shape. A negative control asserts an unrelated error that
// wraps the sentinel with a foreign prefix (the OLD test's "stringified" case,
// which a bare strings.Contains classifier used to accept) stays a redacted 500.
func TestStatusForErrorVectorRecordAbsent(t *testing.T) {
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
			if got := statusForError(tc.err); got != http.StatusNotFound {
				t.Errorf("statusForError = %d, want 404", got)
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + ops.ErrVectorRecordAbsent.Error())
		if got := statusForError(err); got != http.StatusInternalServerError {
			t.Errorf("statusForError(%v) = %d, want 500", err, got)
		}
	})
}

// TestStatusForErrorOperateDuringReshard pins the reshard refusal as a 503, the
// HTTP rendering of the retryable answer the binary transport gives for the same
// signal (server.TestMapResultOperateDuringReshardIsClientFacing). A
// vector_operate against a collection a reshard is dual-writing is refused
// because the op-list is not idempotent; the refusal lasts the minutes the
// reshard runs and the caller retries after cutover. Unclassified it was a
// redacted 500, so the retryability docs/vector/filtering.md promises never
// reached the caller. No Retry-After: none of this transport's other retryable
// buckets set one.
//
// A negative control asserts an unrelated fault that merely wraps the refusal
// text with a foreign prefix stays a redacted 500.
func TestStatusForErrorOperateDuringReshard(t *testing.T) {
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
			if got := statusForError(tc.err); got != http.StatusServiceUnavailable {
				t.Errorf("statusForError = %d, want 503", got)
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the refusal with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + detailed.Error())
		if got := statusForError(err); got != http.StatusInternalServerError {
			t.Errorf("statusForError(%v) = %d, want 500", err, got)
		}
	})
}

// TestStatusForErrorMalformedOperateFrame pins a malformed operate frame as a
// 400, matching the binary transport's client-facing answer for the same
// sentinel (server.TestMapResultMalformedOperateFrameIsClientFacing). The caller
// built the frame; unclassified it was a redacted 500 that read as a server
// fault. KV operate raises the same sentinel from the same decoder, so this arm
// covers both ops.
func TestStatusForErrorMalformedOperateFrame(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", wire.ErrOperateArgs},
		{"wrapped (%w, exercises errors.Is identity)", fmt.Errorf("vector_operate: %w", wire.ErrOperateArgs)},
		// The clustered shape, and the one the sentinel arm cannot reach: an
		// operate handler decodes inside the FSM apply, so shard.decodePBResult
		// hands the error back rebuilt with errors.New and identity is gone.
		{"stringified across Raft, real bare shape", errors.New(wire.ErrOperateArgs.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := statusForError(tc.err); got != http.StatusBadRequest {
				t.Errorf("statusForError = %d, want 400", got)
			}
		})
	}
	// Negative control: an unrelated internal fault that merely mentions the
	// sentinel text must stay a redacted 500 — what exact equality buys over a
	// strings.Contains arm.
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + wire.ErrOperateArgs.Error())
		if got := statusForError(err); got != http.StatusInternalServerError {
			t.Errorf("statusForError(%v) = %d, want 500", err, got)
		}
	})
}

// TestStatusForErrorTruncatedOperateFrame is the parity guard for
// wire.ErrVectorArgsTruncated: the sentinel DecodeVectorOperateArgs raises for a
// frame shorter than the fields it declares, beside the ErrOperateArgs it raises
// for a complete-but-invalid one. Both are the caller's framing mistake, so both
// answer 400 here, matching the binary transport
// (server.TestMapResultTruncatedOperateFrameIsClientFacing) and gRPC's
// InvalidArgument.
func TestStatusForErrorTruncatedOperateFrame(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"sentinel", wire.ErrVectorArgsTruncated},
		{"wrapped (%w, exercises errors.Is identity)", fmt.Errorf("vector_operate: %w", wire.ErrVectorArgsTruncated)},
		{"stringified across Raft, real bare shape", errors.New(wire.ErrVectorArgsTruncated.Error())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := statusForError(tc.err); got != http.StatusBadRequest {
				t.Errorf("statusForError = %d, want 400", got)
			}
		})
	}
	t.Run("negative/unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + wire.ErrVectorArgsTruncated.Error())
		if got := statusForError(err); got != http.StatusInternalServerError {
			t.Errorf("statusForError(%v) = %d, want 500", err, got)
		}
	})
}
