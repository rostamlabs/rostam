// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestGRPCKVQueryErrorCodes pins the kv_query family onto its gRPC codes.
//
// The three transports classify the SAME set, and the whole point of doing it
// three times is that a caller switching from REST to gRPC keeps the same
// retry behaviour. Unclassified, every one of these fell to codes.Internal —
// which standard retry policies and service meshes either hammer or hard-fail,
// so a create-then-query would have become a permanent failure over gRPC while
// it merely needed a retry over HTTP.
func TestGRPCKVQueryErrorCodes(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want codes.Code
	}{
		// PERMANENT: facts about the query.
		{"bad filter", ops.ErrKVQueryFilter, codes.InvalidArgument},
		{"scan consent missing", ops.ErrKVQueryScanRequired, codes.InvalidArgument},
		{"scan budget", ops.ErrKVQueryScanBudget, codes.InvalidArgument},
		{"no KV index on this dispatcher", ops.ErrKVIndexUnavailable, codes.InvalidArgument},
		{"cursor cap", ops.ErrKVQueryCursorCap, codes.InvalidArgument},
		{"candidate budget", kvindex.ErrCandidateBudget, codes.InvalidArgument},
		{"filter budget", wire.ErrKVFilterBudget, codes.InvalidArgument},
		{"malformed args", wire.ErrKVQueryArgs, codes.InvalidArgument},
		{"truncated args", wire.ErrKVQueryArgsTruncated, codes.InvalidArgument},
		{"malformed result", wire.ErrKVQueryResult, codes.InvalidArgument},
		// The caller named something that does not exist.
		{"unknown index", kvindex.ErrNoSuchIndex, codes.NotFound},
		{
			name: "unknown index, as the leaf renders it",
			err:  fmt.Errorf("%w: %q on shard group 3", kvindex.ErrNoSuchIndex, "by_age"),
			want: codes.NotFound,
		},
		// RETRYABLE.
		{"index still building", kvindex.ErrIndexBuilding, codes.Unavailable},
		{
			name: "the coordinator's rewrite of a group-level no-such-index",
			err: fmt.Errorf("%w: %q is in the meta catalog but not yet installed on shard group %d (retry)",
				kvindex.ErrIndexBuilding, "by_age", 3),
			want: codes.Unavailable,
		},
		{"index changed under the query", kvindex.ErrIndexChanged, codes.Unavailable},
		{"shard unavailable mid-scan", ops.ErrKVQueryUnavailable, codes.Unavailable},
		{"store draining", errors.New(ops.StoreClosedMsg), codes.Unavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != tc.want {
				t.Fatalf("grpcError(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// TestGRPCStoreClosedIsUnavailable pins shard.ErrStoreClosed across the wrappers
// it actually arrives in, and pins the negative controls.
//
// grpcapi cannot import shard, so the refusal is matched by message. That makes
// the negative controls the load-bearing half: an unrelated fault must stay
// Internal (and stay redacted at the transports that redact), and a PERMANENT
// filter refusal that merely quotes the refusal text inside a caller-chosen
// field must stay InvalidArgument. The second is the smuggling hole a
// strings.Contains classifier would open — a client naming a filter field after
// the sentinel could make its own permanent error retry forever, one full
// cluster fan-out per attempt.
func TestGRPCStoreClosedIsUnavailable(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"bare", errors.New(ops.StoreClosedMsg)},
		{"wrapped by the fan-out", errors.New("cluster: kv_query: shard group 3: " + ops.StoreClosedMsg)},
		{
			name: "wrapped by the fan-out and the peer client",
			err:  errors.New(`cluster: kv_query: shard group 3: client: server error on op "__kv_query_shard__": ` + ops.StoreClosedMsg),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != codes.Unavailable {
				t.Fatalf("grpcError(%v) = %v, want Unavailable", tc.err, got)
			}
		})
	}

	for _, tc := range []struct {
		name string
		err  error
		want codes.Code
	}{
		{"unrelated fault", errors.New("open /var/lib/rostam/shard-7: no such file"), codes.Internal},
		{
			name: "a fault that merely mentions the refusal mid-message",
			err:  errors.New(ops.StoreClosedMsg + " while writing /var/lib/rostam/wal-3"),
			want: codes.Internal,
		},
		{
			name: "a filter refusal quoting the sentinel in a caller-chosen field",
			err:  fmt.Errorf("%w: field %q is not a path", ops.ErrKVQueryFilter, ops.StoreClosedMsg),
			want: codes.InvalidArgument,
		},
		{
			name: "a filter refusal whose quoted field ENDS with the sentinel text",
			err:  fmt.Errorf("%w: field %q: bad path", ops.ErrKVQueryFilter, "x: "+ops.StoreClosedMsg),
			want: codes.InvalidArgument,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != tc.want {
				t.Fatalf("grpcError(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// kvQuerySteeringFields are strings that appear in the SUBSTRING arms further
// down grpcError's switch. Each is a legal filter field name, and a kv_query
// filter refusal quotes the caller's field verbatim, so each is a string a
// client can plant inside its own error message.
var kvQuerySteeringFields = []string{
	"rate limited",
	"collection full",
	"unknown collection",
	"no collection",
	"already exists",
	"already present",
	"version conflict",
	"not leader",
	"no leader",
	"no reachable owner",
	"cluster: write ",
	"rostam: alias ",
	"rostam: online key-admin unavailable",
}

// TestGRPCKVQueryFilterTextCannotSteerTheCode is a regression test for a hole
// that was REAL before the kv_query arms were moved to the front of the switch.
//
// A filter refusal renders the caller's field name with %q, and the arms below
// match by substring, so a client asking for a field named "rate limited" got
// ResourceExhausted for its own permanent mistake — a code standard retry
// policies and service meshes retry, turning a bad query into an unbounded
// retry loop at one full cluster fan-out per attempt. Every string here used to
// produce a different code; all must now be InvalidArgument.
func TestGRPCKVQueryFilterTextCannotSteerTheCode(t *testing.T) {
	for _, field := range kvQuerySteeringFields {
		err := fmt.Errorf("%w: field %q is not a path", ops.ErrKVQueryFilter, field)
		if got := status.Code(grpcError(err)); got != codes.InvalidArgument {
			t.Errorf("a filter field named %q classified as %v, want InvalidArgument", field, got)
		}
	}
}

// TestGRPCKVQueryIndexNameCannotSteerTheCode is the same hole through the OTHER
// caller-controlled string in this family: the index name, which the leaf quotes
// into its no-such-index refusal. The name charset is narrow, but it admits
// spaces-free forms of several of the substrings above, so the property is
// asserted rather than assumed.
func TestGRPCKVQueryIndexNameCannotSteerTheCode(t *testing.T) {
	for _, name := range []string{"no-leader", "not-leader", "already-exists", "rate-limited"} {
		err := fmt.Errorf("%w: %q on shard group 3", kvindex.ErrNoSuchIndex, name)
		if got := status.Code(grpcError(err)); got != codes.NotFound {
			t.Errorf("an index named %q classified as %v, want NotFound", name, got)
		}
	}
}

// TestGRPCNonKVQueryClassificationUnchanged is the control for the reordering:
// moving the kv_query arms to the front must not capture anything that is not
// kv_query. These are the codes the arms below still own.
func TestGRPCNonKVQueryClassificationUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want codes.Code
	}{
		{"rate limited", errors.New("collection rate limited"), codes.ResourceExhausted},
		{"collection full", errors.New("collection full"), codes.ResourceExhausted},
		{"unknown collection", errors.New("unknown collection \"docs\""), codes.NotFound},
		{"already exists", errors.New("collection already exists"), codes.AlreadyExists},
		{"version conflict", errors.New("version conflict"), codes.FailedPrecondition},
		{"not leader", errors.New("not leader"), codes.Unavailable},
		{"write consistency", errors.New("cluster: write consistency not met"), codes.FailedPrecondition},
		{"unrelated fault", errors.New("open /var/lib/rostam/shard-7: no such file"), codes.Internal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := status.Code(grpcError(tc.err)); got != tc.want {
				t.Fatalf("grpcError(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}
