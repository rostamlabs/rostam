// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

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
