// SPDX-License-Identifier: Apache-2.0

package server

import (
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// The read-only refusal crosses the wire with its message, so an mcp -connect
// client can tell its user what happened and what to do.
func TestReadOnlyCollectionErrorIsClientFacing(t *testing.T) {
	err := fmt.Errorf("mcp: remember: %w", fmt.Errorf("%w: default/mcp_memory: fix the disk problem and restart", vector.ErrCollectionReadOnly))
	if !clientFacingErr(err) {
		t.Fatalf("clientFacingErr(%v) = false, want the message passed to the client", err)
	}
}
