// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

func TestRememberRecallBM25(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var r1 struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "the deploy password is stored in vault under ops/deploy"}, &r1, false)
	c.callTool("remember", map[string]any{"content": "the coffee machine is on floor two"}, nil, false)

	var rec struct {
		Hits []struct {
			ID      uint64 `json:"id"`
			Content string `json:"content"`
		} `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy password vault", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 || !strings.Contains(rec.Hits[0].Content, "vault") {
		t.Fatalf("BM25 recall missed: %+v", rec.Hits)
	}
	if rec.Hits[0].ID != r1.ID {
		t.Fatalf("id mismatch: %d vs %d", rec.Hits[0].ID, r1.ID)
	}
}

func TestRememberDedupesSameContent(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var a, b struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "same fact"}, &a, false)
	c.callTool("remember", map[string]any{"content": "same fact"}, &b, false)
	if a.ID != b.ID {
		t.Fatalf("same content should dedupe: %d vs %d", a.ID, b.ID)
	}
}

func TestNamespaceIsolation(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "alpha secret", "namespace": "projA"}, nil, false)
	var rec struct {
		Hits []struct{ Content string } `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "alpha secret", "namespace": "projB"}, &rec, false)
	if len(rec.Hits) != 0 {
		t.Fatalf("cross-namespace leak: %+v", rec.Hits)
	}
}

func TestRememberRejectsReservedMetadata(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	msg := c.callTool("remember", map[string]any{"content": "x", "metadata": map[string]any{"__ns": "evil"}}, nil, true)
	if !strings.Contains(msg, "__ns") {
		t.Fatalf("error should name the reserved key, got %q", msg)
	}
}

// fixed-vector embedder: "a"-prefixed texts embed near each other, "b" far.
type fakeEmbedder struct{}

func (fakeEmbedder) Model() string { return "fake" }
func (fakeEmbedder) Dim() int      { return 4 }
func (fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, tx := range texts {
		if strings.HasPrefix(tx, "a") {
			out[i] = []float32{1, 0, 0, 0}
		} else {
			out[i] = []float32{0, 1, 0, 0}
		}
	}
	return out, nil
}

func TestRecallHybridUsesEmbedder(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "a: dense-close fact"}, nil, false)
	c.callTool("remember", map[string]any{"content": "b: dense-far fact"}, nil, false)
	var rec struct {
		Hits []struct {
			Content string `json:"content"`
		} `json:"hits"`
	}
	// query "a..." shares NO BM25 tokens with either doc; only the dense side ranks it.
	c.callTool("recall", map[string]any{"query": "a unrelated words", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 || !strings.HasPrefix(rec.Hits[0].Content, "a:") {
		t.Fatalf("hybrid recall did not use dense side: %+v", rec.Hits)
	}
}

// TestRecallResponseHasNoDistanceKey guards memoryHit's locked wire shape
// {id, content, score, metadata}: recall's BM25 path has no notion of
// nearest-neighbor distance, so a stray "distance" key would be a
// meaningless signal to callers. Decoded into map[string]any so an
// unexpected key is visible instead of silently dropped by a typed struct.
func TestRecallResponseHasNoDistanceKey(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "the deploy password is stored in vault"}, nil, false)

	var rec struct {
		Hits []map[string]any `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy password vault"}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("expected 1 hit, got %+v", rec.Hits)
	}
	if _, ok := rec.Hits[0]["distance"]; ok {
		t.Fatalf("recall hit should not carry a distance key: %+v", rec.Hits[0])
	}
}

// TestRecallHybridResponseHasNoDistanceKey is the hybrid-path counterpart:
// recallHybrid goes through the shared hybridDocs helper (mcp/db.go), which
// does carry Distance for the generic search tool, so this guards that
// narrowing the result back down to memoryHit actually drops it.
func TestRecallHybridResponseHasNoDistanceKey(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "a: dense-close fact"}, nil, false)

	var rec struct {
		Hits []map[string]any `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "a unrelated words", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("expected 1 hit, got %+v", rec.Hits)
	}
	if _, ok := rec.Hits[0]["distance"]; ok {
		t.Fatalf("hybrid recall hit should not carry a distance key: %+v", rec.Hits[0])
	}
}

// TestListMemoriesResponseHasNoDistanceKey: list_memories is a scroll, which
// has no query to rank against, so it must not carry a distance key either.
func TestListMemoriesResponseHasNoDistanceKey(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "fact one", "namespace": "page"}, nil, false)

	var page struct {
		Memories []map[string]any `json:"memories"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "page"}, &page, false)
	if len(page.Memories) != 1 {
		t.Fatalf("expected 1 memory, got %+v", page.Memories)
	}
	if _, ok := page.Memories[0]["distance"]; ok {
		t.Fatalf("list_memories entry should not carry a distance key: %+v", page.Memories[0])
	}
}

// memoryHitOut mirrors memoryHit's wire shape for decoding recall/list
// responses in tests.
type memoryHitOut struct {
	ID       uint64         `json:"id"`
	Content  string         `json:"content"`
	Score    float32        `json:"score"`
	Key      string         `json:"key"`
	Created  int64          `json:"created"`
	Updated  int64          `json:"updated"`
	Metadata map[string]any `json:"metadata"`
}

// assertKeyAndFreshness checks the surfaced Key/Created/Updated columns and
// that none of the reserved metadata fields leaked into Metadata.
func assertKeyAndFreshness(t *testing.T, label string, h memoryHitOut, wantKey string) {
	t.Helper()
	if h.Key != wantKey {
		t.Fatalf("%s: Key = %q, want %q: %+v", label, h.Key, wantKey, h)
	}
	if h.Created == 0 {
		t.Fatalf("%s: Created not surfaced: %+v", label, h)
	}
	if h.Updated == 0 {
		t.Fatalf("%s: Updated not surfaced: %+v", label, h)
	}
	for _, reserved := range []string{nsField, createdField, updatedField, keyField} {
		if _, ok := h.Metadata[reserved]; ok {
			t.Fatalf("%s: reserved field %q leaked into metadata: %+v", label, reserved, h.Metadata)
		}
	}
}

// TestRecallAndListSurfaceKeyAndFreshness covers list_memories and recall's
// BM25 path (no embedder configured). See
// TestRecallHybridSurfacesKeyAndFreshness for the dense+BM25 fusion path,
// which goes through a separate code path (recallHybrid/hybridDocs) with a
// different metadata shape at the call site.
func TestRecallAndListSurfaceKeyAndFreshness(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "epoll loop count", "namespace": "proj", "key": "note1"}, nil, false)

	var page struct {
		Memories []memoryHitOut `json:"memories"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "proj"}, &page, false)
	if len(page.Memories) != 1 {
		t.Fatalf("list must return 1 memory: %+v", page.Memories)
	}
	assertKeyAndFreshness(t, "list_memories", page.Memories[0], "note1")

	var rec struct {
		Hits []memoryHitOut `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "epoll", "namespace": "proj"}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("recall must return 1 hit: %+v", rec.Hits)
	}
	assertKeyAndFreshness(t, "recall (BM25)", rec.Hits[0], "note1")
}

// TestRecallUnkeyedCarriesFreshness locks the contract that created/updated are
// surfaced on EVERY hit, keyed or not — only `key` is omitted for an unkeyed
// remember. (The docs previously claimed created/updated were also omitted for
// unkeyed hits, which was wrong: both fields are always set at write time.)
func TestRecallUnkeyedCarriesFreshness(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "raft shards ~= cores", "namespace": "proj"}, nil, false)

	var rec struct {
		Hits []memoryHitOut `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "raft shards", "namespace": "proj"}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("recall must return 1 hit: %+v", rec.Hits)
	}
	// wantKey "" — an unkeyed hit omits key but must still carry created/updated.
	assertKeyAndFreshness(t, "recall (unkeyed)", rec.Hits[0], "")
}

// TestRecallHybridSurfacesKeyAndFreshness is
// TestRecallAndListSurfaceKeyAndFreshness's counterpart for the hybrid
// (dense+BM25 fusion) recall path: recallHybrid goes through the shared
// hybridDocs helper (db.go), whose docs already carry JSON-converted
// metadata (map[string]any) rather than rostam.VectorMetadata, so it is
// exercised separately from the BM25 path above.
func TestRecallHybridSurfacesKeyAndFreshness(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "a: epoll loop count", "namespace": "proj", "key": "note1"}, nil, false)

	var rec struct {
		Hits []memoryHitOut `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "a unrelated words", "namespace": "proj", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("hybrid recall must return 1 hit: %+v", rec.Hits)
	}
	assertKeyAndFreshness(t, "recall (hybrid)", rec.Hits[0], "note1")
}

// TestForgetDeletesAndEmptiedNamespaceDisappears: list_namespaces reports the
// namespaces the live memories carry, so deleting a namespace's last memory is
// the whole of "removing" it. There is no registry to fall out of step.
func TestForgetDeletesAndEmptiedNamespaceDisappears(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var a1, a2, b1 struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "a fact one", "namespace": "a"}, &a1, false)
	c.callTool("remember", map[string]any{"content": "a fact two", "namespace": "a"}, &a2, false)
	c.callTool("remember", map[string]any{"content": "b fact one", "namespace": "b"}, &b1, false)

	var nsBefore struct {
		Namespaces []string `json:"namespaces"`
	}
	c.callTool("list_namespaces", map[string]any{}, &nsBefore, false)
	if !containsStr(nsBefore.Namespaces, "a") || !containsStr(nsBefore.Namespaces, "b") {
		t.Fatalf("expected both namespaces before forget: %+v", nsBefore.Namespaces)
	}

	const unknownID = uint64(999999999)
	var fg struct {
		Deleted []uint64 `json:"deleted"`
		Missing []uint64 `json:"missing"`
	}
	payload := c.callTool("forget", map[string]any{"ids": []uint64{b1.ID, unknownID}}, &fg, false)
	// A fully successful forget keeps the pre-errors-field shape exactly: an
	// always-present "errors":[] would break every caller reading it as a signal.
	if strings.Contains(payload, "errors") {
		t.Fatalf("full-success forget must omit the errors field: %s", payload)
	}
	if len(fg.Deleted) != 1 || fg.Deleted[0] != b1.ID {
		t.Fatalf("expected deleted=[%d], got %+v", b1.ID, fg.Deleted)
	}
	if len(fg.Missing) != 1 || fg.Missing[0] != unknownID {
		t.Fatalf("expected missing=[%d], got %+v", unknownID, fg.Missing)
	}

	var rec struct {
		Hits []struct{ ID uint64 } `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "b fact one", "namespace": "b"}, &rec, false)
	if len(rec.Hits) != 0 {
		t.Fatalf("forgotten memory still recallable: %+v", rec.Hits)
	}

	var nsAfter struct {
		Namespaces []string `json:"namespaces"`
	}
	c.callTool("list_namespaces", map[string]any{}, &nsAfter, false)
	if containsStr(nsAfter.Namespaces, "b") {
		t.Fatalf("emptied namespace \"b\" should be gone: %+v", nsAfter.Namespaces)
	}
	if !containsStr(nsAfter.Namespaces, "a") {
		t.Fatalf("namespace \"a\" should remain: %+v", nsAfter.Namespaces)
	}
}

func containsStr(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// failDeleteStore wraps a rostam.Store, forcing VectorDelete to fail for one
// specific id. It exists to deterministically provoke forget's
// partial-failure path (some ids delete, one doesn't) without relying on any
// real engine fault, which the public Store interface has no way to inject.
type failDeleteStore struct {
	rostam.Store
	failID uint64
}

func (f *failDeleteStore) VectorDelete(ctx context.Context, collection string, id uint64, opts ...rostam.WriteOpts) (bool, error) {
	if id == f.failID {
		return false, fmt.Errorf("injected delete failure for id %d", id)
	}
	return f.Store.VectorDelete(ctx, collection, id, opts...)
}

// TestForgetReportsPartialFailure: one id's delete fails, the rest still go
// through and the result says exactly which did what. list_namespaces then
// reflects the live data — the namespace whose only memory was deleted is
// gone, the one whose memory survived the failure is still there — with no
// bookkeeping step in between that could disagree with either.
func TestForgetReportsPartialFailure(t *testing.T) {
	failing := &failDeleteStore{Store: newHeapStore(t)}
	c := startServer(t, Config{Store: failing})
	c.initialize()

	var a1, a2, b1 struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "a fact one", "namespace": "a"}, &a1, false)
	c.callTool("remember", map[string]any{"content": "a fact two", "namespace": "a"}, &a2, false)
	c.callTool("remember", map[string]any{"content": "b fact one", "namespace": "b"}, &b1, false)

	// a2's delete will fail; b1's is the only memory in namespace "b" and
	// should still delete, emptying "b", even though the overall call reports
	// an error for a2.
	failing.failID = a2.ID

	// The call succeeds and reports the partial outcome (matching delete's
	// contract): b1 in deleted, a2's failure named in errors. Surfacing this as a
	// tool-level error instead would throw away the fact that b1 did delete.
	var fr struct {
		Deleted []uint64 `json:"deleted"`
		Missing []uint64 `json:"missing"`
		Errors  []string `json:"errors"`
	}
	c.callTool("forget", map[string]any{"ids": []uint64{a2.ID, b1.ID}}, &fr, false)
	if len(fr.Deleted) != 1 || fr.Deleted[0] != b1.ID {
		t.Fatalf("deleted = %v, want just b1 (%d)", fr.Deleted, b1.ID)
	}
	if len(fr.Errors) != 1 || !strings.Contains(fr.Errors[0], fmt.Sprint(a2.ID)) {
		t.Fatalf("errors should name the failing id %d: %+v", a2.ID, fr.Errors)
	}

	var ns struct {
		Namespaces []string `json:"namespaces"`
	}
	c.callTool("list_namespaces", map[string]any{}, &ns, false)
	if containsStr(ns.Namespaces, "b") {
		t.Fatalf("namespace \"b\" is empty and should be gone despite a2's delete failure: %+v", ns.Namespaces)
	}
	if !containsStr(ns.Namespaces, "a") {
		t.Fatalf("namespace \"a\" should remain (a2 is still present): %+v", ns.Namespaces)
	}

	var recB struct {
		Hits []struct{ ID uint64 } `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "b fact one", "namespace": "b"}, &recB, false)
	if len(recB.Hits) != 0 {
		t.Fatalf("b1 should have been deleted: %+v", recB.Hits)
	}

	var recA struct {
		Hits []struct{ ID uint64 } `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "a fact two", "namespace": "a", "k": 10}, &recA, false)
	found := false
	for _, h := range recA.Hits {
		if h.ID == a2.ID {
			found = true
		}
	}
	if !found {
		t.Fatalf("a2's delete should have failed, leaving it recallable: %+v", recA.Hits)
	}
}

// TestListNamespacesSeesOutOfBandWrites is the property the KV registry could
// not have: list_namespaces reflects whatever is actually in the collection,
// including a memory this Server never wrote. That stands in for the other
// remote session a registry would have missed — under the old code the
// namespace below would have been invisible until (and unless) some session
// happened to append it to the shared key.
func TestListNamespacesSeesOutOfBandWrites(t *testing.T) {
	st := newHeapStore(t)
	s, err := NewServer(context.Background(), Config{Store: st})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	ctx := context.Background()
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatalf("ensureMemory: %v", err)
	}

	vecs, err := s.emb.Embed(ctx, []string{"written by another session"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	md := rostam.VectorMetadata{nsField: vector.NewString("elsewhere")}
	if err := st.VectorUpsert(ctx, memCollection, 1, vecs[0], "written by another session", rostam.VectorInsertOpts{Metadata: md}); err != nil {
		t.Fatalf("VectorUpsert: %v", err)
	}

	res, err := s.handleListNamespaces(ctx, nil)
	if err != nil {
		t.Fatalf("list_namespaces: %v", err)
	}
	got := res.(map[string]any)["namespaces"].([]string)
	if !containsStr(got, "elsewhere") {
		t.Fatalf("a memory written outside this Server must still name its namespace: %+v", got)
	}
}

// TestListNamespacesPagesPastOnePage guards the scroll loop: with more
// memories than one page holds, a namespace whose only memory sits beyond the
// first page must still be reported.
func TestListNamespacesPagesPastOnePage(t *testing.T) {
	st := newHeapStore(t)
	s, err := NewServer(context.Background(), Config{Store: st})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	ctx := context.Background()
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatalf("ensureMemory: %v", err)
	}
	vecs, err := s.emb.Embed(ctx, []string{"filler"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	// One more than a full page, with the last id in its own namespace.
	total := nsScanPage + 1
	for i := range total {
		ns := "bulk"
		if i == total-1 {
			ns = "tail"
		}
		md := rostam.VectorMetadata{nsField: vector.NewString(ns)}
		if err := st.VectorUpsert(ctx, memCollection, uint64(i+1), vecs[0], "filler", rostam.VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorUpsert(%d): %v", i, err)
		}
	}

	res, err := s.handleListNamespaces(ctx, nil)
	if err != nil {
		t.Fatalf("list_namespaces: %v", err)
	}
	got := res.(map[string]any)["namespaces"].([]string)
	if !containsStr(got, "bulk") || !containsStr(got, "tail") {
		t.Fatalf("both namespaces should survive pagination, got %+v", got)
	}
}

func TestListMemoriesPaginates(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var facts [3]struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "fact one", "namespace": "page"}, &facts[0], false)
	c.callTool("remember", map[string]any{"content": "fact two", "namespace": "page"}, &facts[1], false)
	c.callTool("remember", map[string]any{"content": "fact three", "namespace": "page"}, &facts[2], false)

	var page1 struct {
		Memories []struct {
			ID uint64 `json:"id"`
		} `json:"memories"`
		NextCursor string `json:"next_cursor"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "page", "limit": 2}, &page1, false)
	if len(page1.Memories) != 2 {
		t.Fatalf("expected 2 memories on page 1, got %d", len(page1.Memories))
	}
	if page1.NextCursor == "" {
		t.Fatalf("expected non-empty cursor after a partial page")
	}

	var page2 struct {
		Memories []struct {
			ID uint64 `json:"id"`
		} `json:"memories"`
		NextCursor string `json:"next_cursor"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "page", "limit": 2, "cursor": page1.NextCursor}, &page2, false)
	if len(page2.Memories) != 1 {
		t.Fatalf("expected 1 memory on page 2, got %d", len(page2.Memories))
	}
	if page2.NextCursor != "" {
		t.Fatalf("expected exhausted cursor on page 2, got %q", page2.NextCursor)
	}

	seen := map[uint64]bool{}
	for _, m := range page1.Memories {
		seen[m.ID] = true
	}
	for _, m := range page2.Memories {
		seen[m.ID] = true
	}
	if len(seen) != 3 {
		t.Fatalf("expected union of 3 distinct ids across pages, got %d: %+v", len(seen), seen)
	}
	for _, f := range facts {
		if !seen[f.ID] {
			t.Fatalf("id %d missing from paginated results", f.ID)
		}
	}
}

// shortEmbedder returns fewer vectors than it was asked for, with a nil error
// — what an OpenAI-compatible endpoint answering with a short or empty `data`
// array decodes to. Nothing about the Embedder interface forbids it, and the
// call sites here all pass exactly one string, so this is the shape that used
// to panic the dispatch goroutine on vecs[0].
type shortEmbedder struct{ empty bool } // empty: one vector, but zero-length

func (shortEmbedder) Model() string { return "short" }
func (shortEmbedder) Dim() int      { return 4 }
func (s shortEmbedder) Embed(_ context.Context, _ []string) ([][]float32, error) {
	if s.empty {
		return [][]float32{{}}, nil
	}
	return [][]float32{}, nil
}

// TestMisbehavingEmbedderIsAnErrorNotAPanic covers both memory paths (remember
// embeds content, recall embeds the query) against both bad shapes. A panic
// here would take the whole session down, so each has to come back as an
// ordinary tool error with the session still answering afterwards.
func TestMisbehavingEmbedderIsAnErrorNotAPanic(t *testing.T) {
	for _, tc := range []struct {
		name  string
		empty bool
	}{
		{"no vectors", false},
		{"empty vector", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := startServer(t, Config{Store: newHeapStore(t), Embedder: shortEmbedder{empty: tc.empty}})
			c.initialize()

			msg := c.callTool("remember", map[string]any{"content": "anything"}, nil, true)
			if !strings.Contains(msg, "embed") {
				t.Fatalf("remember error should name the embed step, got %q", msg)
			}
			msg = c.callTool("recall", map[string]any{"query": "anything"}, nil, true)
			if !strings.Contains(msg, "embed") {
				t.Fatalf("recall error should name the embed step, got %q", msg)
			}
			// Still alive: the failure was a tool error, not a dead session.
			c.rpc("ping", nil, false)
		})
	}
}

// TestRecallClampsNonPositiveK: "k": -1 must not reach the search call. The
// old check only replaced an exact 0.
func TestRecallClampsNonPositiveK(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "the deploy password is in vault"}, nil, false)

	for _, k := range []int{-1, 0} {
		var rec struct {
			Hits []struct{ Content string } `json:"hits"`
		}
		c.callTool("recall", map[string]any{"query": "deploy password vault", "k": k}, &rec, false)
		if len(rec.Hits) != 1 {
			t.Fatalf("k=%d should fall back to the default, got %+v", k, rec.Hits)
		}
	}
}

// flakyCreateStore fails CreateCollection while fail is set. It stands in for
// the transient failures a real bootstrap hits — a canceled context, a
// not-leader reply from a remote store, a transport blip — none of which the
// public Store interface offers a way to inject.
type flakyCreateStore struct {
	rostam.Store
	fail atomic.Bool
}

func (f *flakyCreateStore) CreateCollection(ctx context.Context, name string, cfg rostam.VectorConfig) error {
	if f.fail.Load() {
		return errors.New("injected transient bootstrap failure")
	}
	return f.Store.CreateCollection(ctx, name, cfg)
}

// TestEnsureMemoryRetriesAfterTransientFailure is the whole point of dropping
// sync.Once: a bootstrap that failed once must not poison every later memory
// tool call in the process. Under Once, the second ensureMemory below returned
// the cached first error and the server never recovered.
func TestEnsureMemoryRetriesAfterTransientFailure(t *testing.T) {
	flaky := &flakyCreateStore{Store: newHeapStore(t)}
	flaky.fail.Store(true)

	s, err := NewServer(context.Background(), Config{Store: flaky})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	ctx := context.Background()
	if err := s.ensureMemory(ctx); err == nil {
		t.Fatal("first ensureMemory should have failed: CreateCollection is faulted")
	}

	flaky.fail.Store(false)
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatalf("ensureMemory after the fault cleared: %v", err)
	}
	// And the latch closes on success: a subsequent failure injection must not
	// matter, because the collection is already there.
	flaky.fail.Store(true)
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatalf("ensureMemory after a success should be a no-op: %v", err)
	}
}

// TestEnsureMemoryRetriesEndToEnd is the same contract through the tool
// surface: a remember that failed on a faulted bootstrap must succeed once the
// fault clears, in the same session.
func TestEnsureMemoryRetriesEndToEnd(t *testing.T) {
	flaky := &flakyCreateStore{Store: newHeapStore(t)}
	flaky.fail.Store(true)
	c := startServer(t, Config{Store: flaky})
	c.initialize()

	c.callTool("remember", map[string]any{"content": "first try fails"}, nil, true)

	flaky.fail.Store(false)
	c.callTool("remember", map[string]any{"content": "the deploy key lives in vault"}, nil, false)
	var rec struct {
		Hits []struct {
			Content string `json:"content"`
		} `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy key vault"}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("the session should have recovered after the transient failure: %+v", rec.Hits)
	}
}

// hiddenIdentityStore answers the first `remaining` reads of the embedder
// identity key as "not found" while the underlying store already holds it.
// That is exactly the window two concurrent bootstrappers race in: both see no
// identity, both decide to create, and the loser's CreateCollection comes back
// "already exists".
type hiddenIdentityStore struct {
	rostam.Store
	remaining atomic.Int64
}

func (h *hiddenIdentityStore) Get(ctx context.Context, key []byte) ([]byte, error) {
	if string(key) == kvEmbedder && h.remaining.Add(-1) >= 0 {
		return nil, rostam.ErrNotFound
	}
	return h.Store.Get(ctx, key)
}

// TestBootstrapToleratesConcurrentCreator: a session that loses the create
// race joins the winner instead of failing. Under the old code the
// ErrCollectionExists was cached in memOnce and every memory tool in that
// session failed permanently.
func TestBootstrapToleratesConcurrentCreator(t *testing.T) {
	st := newHeapStore(t)
	// The "other session": bootstrap the collection and its identity for real.
	first, err := NewServer(context.Background(), Config{Store: st})
	if err != nil {
		t.Fatalf("NewServer (first): %v", err)
	}
	if err := first.ensureMemory(context.Background()); err != nil {
		t.Fatalf("first ensureMemory: %v", err)
	}

	// This session is blind to the identity key for its startup check and its
	// bootstrap existence check, so it tries to create and loses.
	hidden := &hiddenIdentityStore{Store: st}
	hidden.remaining.Store(2)
	second, err := NewServer(context.Background(), Config{Store: hidden})
	if err != nil {
		t.Fatalf("NewServer (second) should not see a conflict it cannot know about: %v", err)
	}
	if err := second.ensureMemory(context.Background()); err != nil {
		t.Fatalf("losing the create race must not fail the session: %v", err)
	}
	// And the joined session is fully usable.
	if err := second.ensureMemory(context.Background()); err != nil {
		t.Fatalf("second ensureMemory: %v", err)
	}
}

// TestBootstrapVerifiesWhenIdentityStaysMissing is the other side of that
// tolerance: when none of the retries sees an identity, the session gives up
// waiting and verifies its embedder against the collection instead of hanging.
// An embedder that does not fit the collection is still refused.
func TestBootstrapVerifiesWhenIdentityStaysMissing(t *testing.T) {
	st := newHeapStore(t)
	first, err := NewServer(context.Background(), Config{Store: st})
	if err != nil {
		t.Fatalf("NewServer (first): %v", err)
	}
	if err := first.ensureMemory(context.Background()); err != nil {
		t.Fatalf("first ensureMemory: %v", err)
	}

	// Hidden from the startup check, the bootstrap check and every retry.
	const blind = 2 + identityReadAttempts
	hidden := &hiddenIdentityStore{Store: st}
	hidden.remaining.Store(blind)
	second, err := NewServer(context.Background(), Config{Store: hidden})
	if err != nil {
		t.Fatalf("NewServer (second): %v", err)
	}
	if err := second.ensureMemory(context.Background()); err != nil {
		t.Fatalf("a matching embedder must not be locked out: %v", err)
	}

	hidden.remaining.Store(blind)
	other, err := NewServer(context.Background(), Config{Store: hidden, Embedder: fakeEmbedder{}})
	if err != nil {
		t.Fatalf("NewServer (other): %v", err)
	}
	if err := other.ensureMemory(context.Background()); !errors.Is(err, ErrEmbedderMismatch) {
		t.Fatalf("an embedder that does not fit the collection: err = %v, want ErrEmbedderMismatch", err)
	}
}

// TestRememberKeyedUpsertReplacesContent: two remembers sharing a key land on
// the same id and leave exactly one memory behind, carrying the latest
// content — the whole point of a keyed remember over the content-hash id.
func TestRememberKeyedUpsertReplacesContent(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var first, second struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "state A", "namespace": "proj", "key": "pr-status"}, &first, false)
	c.callTool("remember", map[string]any{"content": "state B", "namespace": "proj", "key": "pr-status"}, &second, false)
	if first.ID != second.ID {
		t.Fatalf("same key must yield same id: %d vs %d", first.ID, second.ID)
	}

	var page struct {
		Memories []struct {
			ID      uint64 `json:"id"`
			Content string `json:"content"`
		} `json:"memories"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "proj"}, &page, false)
	if len(page.Memories) != 1 {
		t.Fatalf("keyed upsert must not accumulate: got %d memories: %+v", len(page.Memories), page.Memories)
	}
	if page.Memories[0].Content != "state B" {
		t.Fatalf("content = %q, want latest %q", page.Memories[0].Content, "state B")
	}
}

// TestRememberWithoutKeyUnchanged guards backward compat: remembers with no
// key keep deriving their id from (namespace, content), so two different
// facts stay two independent memories.
func TestRememberWithoutKeyUnchanged(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "fact one", "namespace": "proj"}, nil, false)
	c.callTool("remember", map[string]any{"content": "fact two", "namespace": "proj"}, nil, false)

	var page struct {
		Memories []struct{ ID uint64 } `json:"memories"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "proj"}, &page, false)
	if len(page.Memories) != 2 {
		t.Fatalf("non-keyed remembers should be independent: got %d", len(page.Memories))
	}
}

// TestRememberRejectsReservedKeyAndUpdatedInMetadata extends the existing
// reserved-metadata guard (TestRememberRejectsReservedMetadata) to the two
// fields this feature adds: a caller cannot smuggle a fake __key or
// __updated_unix into user metadata.
func TestRememberRejectsReservedKeyAndUpdatedInMetadata(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	for _, bad := range []string{"__key", "__updated_unix"} {
		msg := c.callTool("remember", map[string]any{"content": "x", "metadata": map[string]any{bad: "nope"}}, nil, true)
		if !strings.Contains(msg, bad) {
			t.Fatalf("error should name the reserved key %q, got %q", bad, msg)
		}
	}
}

// TestRememberKeyedPreservesCreatedTimestamp: a keyed re-remember must keep
// the ORIGINAL __created_unix (this is still the same logical memory, first
// created earlier) while __updated_unix moves to reflect the new write. The
// existing point is planted directly via VectorUpsert with a deliberately
// old created timestamp so the test doesn't depend on wall-clock timing.
func TestRememberKeyedPreservesCreatedTimestamp(t *testing.T) {
	s, err := NewServer(context.Background(), Config{Store: newHeapStore(t)})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	ctx := context.Background()
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatalf("ensureMemory: %v", err)
	}

	const ns, key = "proj", "pr-status"
	id := memoryKeyID(ns, key)
	const oldCreated = int64(1000)

	vecs, err := s.emb.Embed(ctx, []string{"state A"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	seedMD := rostam.VectorMetadata{
		nsField:      vector.NewString(ns),
		keyField:     vector.NewString(key),
		createdField: vector.NewInt(oldCreated),
		updatedField: vector.NewInt(oldCreated),
	}
	if err := s.store.VectorUpsert(ctx, memCollection, id, vecs[0], "state A", rostam.VectorInsertOpts{Metadata: seedMD}); err != nil {
		t.Fatalf("seed VectorUpsert: %v", err)
	}

	raw, err := json.Marshal(map[string]any{"content": "state B", "namespace": ns, "key": key})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, err := s.handleRemember(ctx, raw); err != nil {
		t.Fatalf("handleRemember: %v", err)
	}

	found, _, md, _, _, err := s.store.VectorGet(ctx, memCollection, id, false, true)
	if err != nil {
		t.Fatalf("VectorGet: %v", err)
	}
	if !found {
		t.Fatalf("expected point %d to exist after keyed remember", id)
	}
	created, ok := md[createdField]
	if !ok || created.Kind != vector.ValueInt {
		t.Fatalf("expected an int __created_unix field, got %+v", md[createdField])
	}
	if created.Int != oldCreated {
		t.Fatalf("created should be preserved across a keyed re-remember: got %d, want %d", created.Int, oldCreated)
	}
	updated, ok := md[updatedField]
	if !ok || updated.Kind != vector.ValueInt {
		t.Fatalf("expected an int __updated_unix field, got %+v", md[updatedField])
	}
	if updated.Int == oldCreated {
		t.Fatalf("updated should have moved past the seeded old timestamp, still %d", updated.Int)
	}
}

// TestForgetByKeyDeletesKeyedMemory: forget accepts keys as an alternative to
// ids, resolving each (namespace, key) pair to the point the keyed remember
// landed on (memoryKeyID) and deleting it — a caller holding a stable key
// doesn't have to remember or look up its numeric id first.
func TestForgetByKeyDeletesKeyedMemory(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "state", "namespace": "proj", "key": "pr-status"}, nil, false)

	var fg struct {
		Deleted []uint64 `json:"deleted"`
	}
	c.callTool("forget", map[string]any{"namespace": "proj", "keys": []string{"pr-status"}}, &fg, false)
	if len(fg.Deleted) != 1 {
		t.Fatalf("forget by key should delete 1, got %+v", fg.Deleted)
	}

	var page struct {
		Memories []struct{ ID uint64 } `json:"memories"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "proj"}, &page, false)
	if len(page.Memories) != 0 {
		t.Fatalf("memory not gone after forget-by-key: %d remain", len(page.Memories))
	}
}

// TestForgetByIDStillWorks guards that adding key-based forgetting didn't
// disturb the existing id-only path.
func TestForgetByIDStillWorks(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var r struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "a fact", "namespace": "proj"}, &r, false)

	var fg struct {
		Deleted []uint64 `json:"deleted"`
	}
	c.callTool("forget", map[string]any{"ids": []uint64{r.ID}}, &fg, false)
	if len(fg.Deleted) != 1 || fg.Deleted[0] != r.ID {
		t.Fatalf("expected deleted=[%d], got %+v", r.ID, fg.Deleted)
	}
}

// TestForgetRequiresIdsOrKeys: the old "ids is required" check must now allow
// keys alone, but still reject a call carrying neither.
func TestForgetRequiresIdsOrKeys(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	msg := c.callTool("forget", map[string]any{}, nil, true)
	if !strings.Contains(msg, "ids") || !strings.Contains(msg, "keys") {
		t.Fatalf("error should mention both ids and keys, got %q", msg)
	}
}

func TestEmbedderIdentityMismatchFailsStartup(t *testing.T) {
	st := newHeapStore(t)
	c := startServer(t, Config{Store: st}) // BM25-only creates the collection
	c.initialize()
	c.callTool("remember", map[string]any{"content": "seed"}, nil, false)

	// Reopening the same store with a real embedder must fail loudly.
	_, err := NewServer(context.Background(), Config{Store: st, Embedder: fakeEmbedder{}})
	if err == nil {
		t.Fatal("expected embedder-identity mismatch error")
	}
	// Callers (mcp -autostart) add mode-specific advice, so the error must be
	// recognisable, and its own advice must name the env that selects it.
	if !errors.Is(err, ErrEmbedderMismatch) || !strings.Contains(err.Error(), "ROSTAM_EMBED_") {
		t.Fatalf("mismatch error must wrap ErrEmbedderMismatch and name ROSTAM_EMBED_*: %v", err)
	}
}
