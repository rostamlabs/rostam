// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// readOnlyStore refuses every change to the memory collection the way the
// engine does for a collection served read-only, and serves reads normally.
type readOnlyStore struct{ rostam.Store }

func readOnlyErr() error {
	return fmt.Errorf("%w: default/%s: its upgraded write-ahead log form did not load; fix the disk problem and restart",
		vector.ErrCollectionReadOnly, memCollection)
}

func (s readOnlyStore) VectorUpsert(context.Context, string, uint64, []float32, string, rostam.VectorInsertOpts) error {
	return readOnlyErr()
}

func (s readOnlyStore) VectorDelete(context.Context, string, uint64, ...rostam.WriteOpts) (bool, error) {
	return false, readOnlyErr()
}

func (s readOnlyStore) VectorDeleteByFilter(context.Context, string, rostam.VectorFilter) (int, error) {
	return 0, readOnlyErr()
}

// TestMemoryReadOnlyCollection: remember and forget report the read-only
// refusal in words the user can act on (remember as a tool error, forget in its
// per-item errors, as for any other failed delete), and recall still works.
func TestMemoryReadOnlyCollection(t *testing.T) {
	st := newHeapStore(t)
	w := startServer(t, Config{Store: st})
	w.initialize()
	var r struct {
		ID uint64 `json:"id"`
	}
	w.callTool("remember", map[string]any{"content": "the deploy key lives in the vault", "key": "k1"}, &r, false)

	c := startServer(t, Config{Store: readOnlyStore{st}})
	c.initialize()
	readable := func(what, msg string) {
		t.Helper()
		if !strings.Contains(msg, "read-only") || !strings.Contains(msg, "restart") {
			t.Fatalf("%s on a read-only memory collection: %q, want the read-only refusal and what to do", what, msg)
		}
	}
	// remember fails the call.
	readable("remember", c.callTool("remember", map[string]any{"content": "a new fact"}, nil, true))
	// forget reports per item: nothing deleted, the refusal in its errors.
	for _, args := range []map[string]any{{"ids": []uint64{r.ID}}, {"keys": []string{"k1"}}} {
		var fg struct {
			Deleted []uint64 `json:"deleted"`
			Errors  []string `json:"errors"`
		}
		c.callTool("forget", args, &fg, false)
		if len(fg.Deleted) != 0 || len(fg.Errors) != 1 {
			t.Fatalf("forget %v on a read-only memory collection = %+v, want nothing deleted and one error", args, fg)
		}
		readable("forget", fg.Errors[0])
	}
	var rec struct {
		Hits []memoryHit `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy key vault", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 || rec.Hits[0].ID != r.ID {
		t.Fatalf("recall on a read-only memory collection = %+v", rec.Hits)
	}
}
