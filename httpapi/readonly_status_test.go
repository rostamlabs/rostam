// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// A busy eviction and a read-only collection are conditions the caller acts
// on, not server faults: 409 with the message, never a redacted 500.
func TestStatusForReadOnlyAndBusyEviction(t *testing.T) {
	for _, err := range []error{
		fmt.Errorf("vector: evict %q: %w", "default/docs", vector.ErrEvictBusy),
		fmt.Errorf("%w: default/mcp_memory: restart", vector.ErrCollectionReadOnly),
	} {
		if got := statusForError(err); got != http.StatusConflict {
			t.Fatalf("statusForError(%v) = %d, want 409", err, got)
		}
	}
}
