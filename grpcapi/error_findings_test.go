// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestGrpcErrorConflictAndBackpressure covers finding 019 (gRPC side): the four
// collection-level outcomes that previously fell through to the Internal default.
// Create-conflicts map to AlreadyExists; quota/rate-limit refusals to
// ResourceExhausted. Both the sentinel path and the clustered/stringified fallback
// are exercised, mirroring the HTTP 409/429 mapping.
func TestGrpcErrorConflictAndBackpressure(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want codes.Code
	}{
		{"exists-sentinel", vector.ErrCollectionExists, codes.AlreadyExists},
		{"dupid-sentinel", vector.ErrDuplicateID, codes.AlreadyExists},
		{"ratelimited-sentinel", vector.ErrCollectionRateLimited, codes.ResourceExhausted},
		{"full-sentinel", vector.ErrCollectionFull, codes.ResourceExhausted},
		{"exists-string", errors.New("rostam: collection already exists"), codes.AlreadyExists},
		{"dupid-string", errors.New("rostam: id already present (delete first)"), codes.AlreadyExists},
		{"ratelimited-string", errors.New("rostam: collection insert rate limited"), codes.ResourceExhausted},
		{"full-string", errors.New("rostam: collection full (quota exceeded)"), codes.ResourceExhausted},
	}
	for _, tc := range cases {
		if got := status.Code(grpcError(tc.err)); got != tc.want {
			t.Errorf("%s: grpcError = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestGrpcErrorLeaderlessIsRetryable covers finding 021: a leaderless/ownership
// transient must map to the RETRYABLE codes.Unavailable (mirroring HTTP's 503), not
// the non-retryable codes.Internal that standard gRPC retry policies and service
// meshes ignore. The reachable divergent case is client.ErrNoLeaderKnown ("client:
// no leader known after retries") which contains "no leader", not "not leader"; the
// secondary case is cluster.ErrNoShardOwner.
func TestGrpcErrorLeaderlessIsRetryable(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"no-leader-known", errors.New("client: no leader known after retries")},
		{"not-leader", errors.New("shard: not leader for partition 3")},
		{"no-reachable-owner", errors.New("cluster: no reachable owner for shard")},
	}
	for _, tc := range cases {
		if got := status.Code(grpcError(tc.err)); got != codes.Unavailable {
			t.Errorf("%s: grpcError = %v, want Unavailable (retryable)", tc.name, got)
		}
	}
}

// TestGrpcErrorRecordTooLarge pins vector.ErrRecordTooLarge as a CLIENT error on
// the gRPC front end. It was classified by neither errIs list, so an oversize
// record payload came back as codes.Internal — a server fault, which standard
// gRPC retry policies retry and which tells the caller nothing about the one
// thing they can fix.
//
// The sentinel does not survive every path it can take, so this covers both
// the in-process shapes (single-payload and bulk, %w-wrapped so errors.Is
// still finds the sentinel) and the stringified ones a clustered apply
// produces, where shard.decodePBResult rebuilds the error with
// errors.New(string(payload)) and errors.Is stops matching — recognized
// instead by vector.IsRecordTooLargeMessage, an exact-shape matcher.
func TestGrpcErrorRecordTooLarge(t *testing.T) {
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
		if got := status.Code(grpcError(tc.err)); got != codes.InvalidArgument {
			t.Errorf("%s: grpcError = %v, want InvalidArgument", tc.name, got)
		}
	}
	// Negative controls: an unrelated internal fault must still be Internal,
	// including one that merely CONTAINS the sentinel text inside an unrelated
	// wrapper — the exact bug a bare strings.Contains classifier would
	// reintroduce (it would leak the WAL path below to the caller).
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
		if got := status.Code(grpcError(tc.err)); got != codes.Internal {
			t.Errorf("%s: grpcError = %v, want Internal", tc.name, got)
		}
	}
}

// errRealMalformedSingle, errRealMalformedBulk and errRealNotRecordDetailed reproduce
// the EXACT shapes checkRecordValues / checkRecordValuesAll (ErrRecordMalformed)
// and currentRecordValue (ErrPayloadKeyNotRecord) actually produce in
// production — see vector.IsRecordMalformedMessage / IsPayloadKeyNotRecordMessage
// for the format strings these mirror. grpcapi cannot call those unexported
// vector-package producers directly, so the shapes are reproduced by hand;
// vector's own TestIsRecordMalformedMessage / TestIsPayloadKeyNotRecordMessage
// pin them against the real producers.
var (
	errRealMalformedSingle   = fmt.Errorf("%w: payload key %q: %s", vector.ErrRecordMalformed, "session", "record: malformed record: wire: args too short")
	errRealMalformedBulk     = fmt.Errorf("payload %d: %w", 7, errRealMalformedSingle)
	errRealNotRecordDetailed = fmt.Errorf("%w: payload key %q holds a value of kind %d", vector.ErrPayloadKeyNotRecord, "country", 2)
)

// TestGrpcErrorVectorRecordMalformedAndPayloadKeyNotRecord pins the gRPC
// transport's classification of vector_operate's two ingest/shape refusals —
// vector.ErrRecordMalformed and vector.ErrPayloadKeyNotRecord — as
// InvalidArgument, mirroring server.clientFacingErr / httpapi.statusForError's
// 400 bucket for the same sentinels (PR #102 found these unclassified here,
// falling through to codes.Internal). vector.ErrRecordTooLarge shares the
// bucket and is pinned by TestGrpcErrorRecordTooLarge above.
//
// Per error: the bare sentinel (errIs), the real %w-wrapped shape (matched by
// errIs regardless of exact text — identity survives an in-process %w wrap),
// and that same shape stringified across the Raft boundary (matched only by
// the message-shape matcher, since shard.decodePBResult rebuilds a replicated
// op error with errors.New and errors.Is stops matching there).
func TestGrpcErrorVectorRecordMalformedAndPayloadKeyNotRecord(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"malformed-sentinel", vector.ErrRecordMalformed},
		{"malformed-wrapped, real single-payload shape", errRealMalformedSingle},
		{"malformed-wrapped, real bulk shape", errRealMalformedBulk},
		{"malformed-stringified across replication, real single-payload shape", errors.New(errRealMalformedSingle.Error())},
		{"malformed-stringified across replication, real bulk shape", errors.New(errRealMalformedBulk.Error())},
		{"not-record-sentinel", vector.ErrPayloadKeyNotRecord},
		{"not-record-wrapped, real detailed shape", errRealNotRecordDetailed},
		{"not-record-stringified across replication, real detailed shape", errors.New(errRealNotRecordDetailed.Error())},
		{"not-record-stringified across replication, bare shape", errors.New(vector.ErrPayloadKeyNotRecord.Error())},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != codes.InvalidArgument {
				t.Errorf("grpcError(%v) = %v, want InvalidArgument", tc.err, got)
			}
		})
	}
	// Negative controls: an unrelated internal fault must still be Internal,
	// including one that merely CONTAINS the sentinel text inside an unrelated
	// wrapper — the exact redaction bypass a bare strings.Contains classifier
	// reintroduces (it would leak the WAL path below to the caller). Before this
	// fix these two cases were misclassified InvalidArgument.
	negatives := []struct {
		name string
		err  error
	}{
		{"unrelated fault wrapping ErrRecordMalformed (%v, no %w identity)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrRecordMalformed)},
		{"unrelated fault wrapping ErrPayloadKeyNotRecord (%v, no %w identity)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", vector.ErrPayloadKeyNotRecord)},
	}
	for _, tc := range negatives {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != codes.Internal {
				t.Errorf("grpcError(%v) = %v, want Internal", tc.err, got)
			}
		})
	}
}

// TestGrpcErrorVectorRecordAbsentIsNotFound pins vector_operate's create=NONE
// refusal (ops.ErrVectorRecordAbsent) as NotFound over gRPC, the same status
// server.mapResult (StatusNotFound) and httpapi.statusForError (404) answer
// for the identical signal — every transport tells the caller the same thing
// for a record that simply is not there.
//
// ErrVectorRecordAbsent, unlike the vector-package sentinels above, has
// exactly ONE production shape — bare, never wrapped (see
// ops.IsVectorRecordAbsentMessage's doc: every caller propagates it verbatim).
// "wrapped" here is therefore a synthetic %w wrap that exercises errIs's
// identity match, not a real production shape; "stringified" is the bare form
// re-created with errors.New, the real clustered-apply shape.
func TestGrpcErrorVectorRecordAbsentIsNotFound(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"sentinel", ops.ErrVectorRecordAbsent},
		{"wrapped (synthetic %w, exercises errIs identity — production never wraps this sentinel)",
			fmt.Errorf("shard 3: %w", ops.ErrVectorRecordAbsent)},
		{"stringified across replication, real bare shape", errors.New(ops.ErrVectorRecordAbsent.Error())},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != codes.NotFound {
				t.Errorf("grpcError(%v) = %v, want NotFound", tc.err, got)
			}
		})
	}
	// Negative control: an unrelated internal fault that merely wraps the
	// sentinel with its OWN prefix text must stay Internal — a bare
	// strings.Contains classifier would leak it as NotFound instead (the exact
	// redaction bypass this replaces "apply: "+err.Error() with an exact-form
	// matcher to close).
	t.Run("unrelated fault wrapping the sentinel with a foreign prefix", func(t *testing.T) {
		err := errors.New("apply: " + ops.ErrVectorRecordAbsent.Error())
		if got := status.Code(grpcError(err)); got != codes.Internal {
			t.Errorf("grpcError(%v) = %v, want Internal", err, got)
		}
	})
}
