// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/vector"
)

func TestReadOnlyCollectionIsFailedPrecondition(t *testing.T) {
	err := grpcError(fmt.Errorf("%w: default/mcp_memory: restart", vector.ErrCollectionReadOnly))
	if got := status.Code(err); got != codes.FailedPrecondition {
		t.Fatalf("code = %v, want FailedPrecondition", got)
	}
}

func TestWALRecordTooLargeIsInvalidArgument(t *testing.T) {
	err := grpcError(fmt.Errorf("%w: the document's record takes 70000000 bytes, the limit is 67108864", vector.ErrWALRecordTooLarge))
	if got := status.Code(err); got != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", got)
	}
}
