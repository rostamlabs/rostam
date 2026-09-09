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

// TestGrpcErrorVectorRecordMalformedAndPayloadKeyNotRecord pins the gRPC
// transport's classification of vector_operate's two ingest/shape refusals —
// vector.ErrRecordMalformed and vector.ErrPayloadKeyNotRecord — as
// InvalidArgument, mirroring server.clientFacingErr / httpapi.statusForError's
// 400 bucket for the same sentinels (PR #102 found these unclassified here,
// falling through to codes.Internal). vector.ErrRecordTooLarge shares the
// bucket and is pinned by TestGrpcErrorRecordTooLarge above.
//
// Three arms per error, like server.TestClientFacingErrMalformedRecord /
// httpapi.TestStatusForErrorMalformedRecord: sentinel, wrapped, and
// stringified across the Raft boundary — shard.decodePBResult rebuilds op
// errors with errors.New across replication, so errors.Is alone loses the
// sentinel there and grpcError's string fallback is what keeps this
// InvalidArgument instead of a redacted Internal.
func TestGrpcErrorVectorRecordMalformedAndPayloadKeyNotRecord(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"malformed-sentinel", vector.ErrRecordMalformed},
		{"malformed-wrapped", fmt.Errorf("%w: payload key %q: bad", vector.ErrRecordMalformed, "session")},
		{"malformed-stringified", errors.New("apply: " + vector.ErrRecordMalformed.Error())},
		{"not-record-sentinel", vector.ErrPayloadKeyNotRecord},
		{"not-record-wrapped", fmt.Errorf("%w: payload key %q holds a value of kind %d", vector.ErrPayloadKeyNotRecord, "country", 2)},
		{"not-record-stringified", errors.New("apply: " + vector.ErrPayloadKeyNotRecord.Error())},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != codes.InvalidArgument {
				t.Errorf("grpcError(%v) = %v, want InvalidArgument", tc.err, got)
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
// Three arms, for the same clustered-apply stringification reason as above.
func TestGrpcErrorVectorRecordAbsentIsNotFound(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"sentinel", ops.ErrVectorRecordAbsent},
		{"wrapped", fmt.Errorf("shard 3: %w", ops.ErrVectorRecordAbsent)},
		{"stringified", errors.New("apply: " + ops.ErrVectorRecordAbsent.Error())},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != codes.NotFound {
				t.Errorf("grpcError(%v) = %v, want NotFound", tc.err, got)
			}
		})
	}
}
