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
// Three arms, because the sentinel does not survive every path it can take:
// the sentinel itself, a wrapped one (%w through the op layer), and the
// stringified one a clustered apply produces, where shard.decodePBResult
// rebuilds the error with errors.New(string(payload)) and errors.Is stops
// matching. The stringified arm is built from the sentinel's own .Error() so it
// cannot drift from the classifier's substring.
func TestGrpcErrorRecordTooLarge(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"sentinel", vector.ErrRecordTooLarge},
		{"wrapped", fmt.Errorf("vector_insert: %w: payload key %q holds a 20000000-byte record, the cap is 16777216 bytes", vector.ErrRecordTooLarge, "session")},
		{"stringified", errors.New(vector.ErrRecordTooLarge.Error() + ": payload key \"session\" holds a 20000000-byte record, the cap is 16777216 bytes")},
	}
	for _, tc := range cases {
		if got := status.Code(grpcError(tc.err)); got != codes.InvalidArgument {
			t.Errorf("%s: grpcError = %v, want InvalidArgument", tc.name, got)
		}
	}
	// The negative control: an unrelated internal fault must still be Internal,
	// so the new arm is not classifying by accident.
	if got := status.Code(grpcError(errors.New("open /var/lib/rostam/shard-7: no such file"))); got != codes.Internal {
		t.Errorf("unrelated fault = %v, want Internal", got)
	}
}
