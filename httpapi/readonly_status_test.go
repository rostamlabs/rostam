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
		got, msg := clientError("op", err)
		if got != http.StatusConflict {
			t.Fatalf("clientError(%v) status = %d, want 409", err, got)
		}
		if msg != err.Error() {
			t.Fatalf("clientError(%v) message = %q, want the error's own message", err, msg)
		}
	}
}

// A multi-vector document too large for one log record is refused before it
// applies: a 400 carrying the message, not a redacted 500.
func TestStatusForWALRecordTooLarge(t *testing.T) {
	err := fmt.Errorf("%w: the document's record takes 70000000 bytes, the limit is 67108864", vector.ErrWALRecordTooLarge)
	got, msg := clientError("op", err)
	if got != http.StatusBadRequest || msg != err.Error() {
		t.Fatalf("clientError(%v) = (%d, %q), want 400 with the message", err, got, msg)
	}
}
