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

// A multi-vector document too large for one log record is refused before it
// applies: the caller's own mistake, so its message reaches the client.
func TestWALRecordTooLargeIsClientFacing(t *testing.T) {
	err := fmt.Errorf("%w: the document's record takes 70000000 bytes, the limit is 67108864", vector.ErrWALRecordTooLarge)
	if !clientFacingErr(err) {
		t.Fatalf("clientFacingErr(%v) = false, want the message passed to the client", err)
	}
}

// A poisoned log is the server's durability fault, including an applied write
// whose record was too large to append: redacted, not the caller's mistake.
func TestPoisonedWALIsNotClientFacing(t *testing.T) {
	err := fmt.Errorf("%w: an applied write's log record takes 70000000 bytes, the limit is 67108864", vector.ErrWALPoisoned)
	if clientFacingErr(err) {
		t.Fatalf("clientFacingErr(%v) = true, want a redacted server fault", err)
	}
}
