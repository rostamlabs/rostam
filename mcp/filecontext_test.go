// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// seedMemory writes one linked memory with an explicit Updated stamp so the
// ordering tests do not depend on wall-clock seconds.
func seedMemory(t *testing.T, s *Server, ns, path, content string, updated int64) {
	t.Helper()
	ctx := context.Background()
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatal(err)
	}
	vecs, err := s.emb.Embed(ctx, []string{content})
	if err != nil {
		t.Fatal(err)
	}
	md := rostam.VectorMetadata{
		nsField:      vector.NewString(ns),
		createdField: vector.NewInt(updated),
		updatedField: vector.NewInt(updated),
	}
	if path != "" {
		md[pathField] = vector.NewString(path)
		md[fileHashField] = vector.NewString("")
	}
	if err := s.store.VectorUpsert(ctx, memCollection, memoryID(ns, content), vecs[0], content, rostam.VectorInsertOpts{Metadata: md}); err != nil {
		t.Fatal(err)
	}
}

func newTestServer(t *testing.T, cfg Config) *Server {
	t.Helper()
	if cfg.Store == nil {
		cfg.Store = newHeapStore(t)
	}
	s, err := NewServer(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestFileMemoriesExactPathOnly(t *testing.T) {
	s := newTestServer(t, Config{})
	seedMemory(t, s, "p", "a.go", "about a", 10)
	seedMemory(t, s, "p", "a.go.bak", "about bak", 11)
	seedMemory(t, s, "p", "dir/a.go", "about dir a", 12)
	seedMemory(t, s, "q", "a.go", "other ns", 13)
	got, err := s.FileMemories(context.Background(), "p", "a.go", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Content != "about a" || got[0].Path != "a.go" {
		t.Fatalf("got %+v", got)
	}
}

// Skipped ids are dropped before the k cap, so the cap still yields k unseen
// memories when the newest ones were already seen.
func TestMemoriesExceptSkipsBeforeCap(t *testing.T) {
	s := newTestServer(t, Config{})
	seedMemory(t, s, "p", "a.go", "old", 1)
	seedMemory(t, s, "p", "a.go", "newest", 30)
	seedMemory(t, s, "p", "a.go", "mid", 20)
	all, err := s.FileMemories(context.Background(), "p", "a.go", 0)
	if err != nil || len(all) != 3 {
		t.Fatalf("all = %+v, %v", all, err)
	}
	seen := map[uint64]bool{all[0].ID: true, all[1].ID: true} // newest, mid
	skip := func(id uint64) bool { return seen[id] }
	got, err := s.FileMemoriesExcept(context.Background(), "p", "a.go", 2, skip)
	if err != nil || len(got) != 1 || got[0].Content != "old" {
		t.Fatalf("FileMemoriesExcept = %+v, %v; want just the unseen \"old\"", got, err)
	}
	got, err = s.RecentMemoriesExcept(context.Background(), "p", 1, skip)
	if err != nil || len(got) != 1 || got[0].Content != "old" {
		t.Fatalf("RecentMemoriesExcept = %+v, %v; want just the unseen \"old\"", got, err)
	}
}

func TestFileMemoriesNewestFirstAndK(t *testing.T) {
	s := newTestServer(t, Config{})
	seedMemory(t, s, "p", "a.go", "old", 1)
	seedMemory(t, s, "p", "a.go", "newest", 30)
	seedMemory(t, s, "p", "a.go", "mid", 20)
	got, err := s.FileMemories(context.Background(), "p", "a.go", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Content != "newest" || got[1].Content != "mid" {
		t.Fatalf("got %+v", got)
	}
}

func TestRecentMemoriesOrder(t *testing.T) {
	s := newTestServer(t, Config{})
	seedMemory(t, s, "p", "", "unlinked newest", 50)
	seedMemory(t, s, "p", "a.go", "linked", 40)
	seedMemory(t, s, "p", "", "oldest", 5)
	seedMemory(t, s, "q", "", "elsewhere", 99)
	got, err := s.RecentMemories(context.Background(), "p", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Content != "unlinked newest" || got[1].Content != "linked" {
		t.Fatalf("got %+v", got)
	}
}

func TestRecallForFileTool(t *testing.T) {
	c, root := linkEnv(t)
	writeFile(t, root, "a.go", "package a")
	writeFile(t, root, "b.go", "package b")
	c.callTool("remember", map[string]any{"content": "a is fragile", "namespace": "p", "path": "a.go"}, nil, false)
	c.callTool("remember", map[string]any{"content": "b is fine", "namespace": "p", "path": "b.go"}, nil, false)
	writeFile(t, root, "a.go", "package a // edited")

	var out struct {
		Memories []linkHit `json:"memories"`
	}
	// An absolute path inside the root is normalised.
	c.callTool("recall_for_file", map[string]any{"path": root + "/a.go", "namespace": "p"}, &out, false)
	if len(out.Memories) != 1 || out.Memories[0].Content != "a is fragile" || out.Memories[0].Stale != true {
		t.Fatalf("got %+v", out.Memories)
	}
	c.callTool("recall_for_file", map[string]any{"path": "../x", "namespace": "p"}, nil, true)
}

func TestReadOnlyNoCollection(t *testing.T) {
	st := newHeapStore(t)
	s := newTestServer(t, Config{Store: st, ReadOnly: true})
	if _, err := s.RecentMemories(context.Background(), "p", 5); !errors.Is(err, ErrNoMemory) {
		t.Fatalf("RecentMemories err = %v, want ErrNoMemory", err)
	}
	if _, err := s.FileMemories(context.Background(), "p", "a.go", 5); !errors.Is(err, ErrNoMemory) {
		t.Fatalf("FileMemories err = %v, want ErrNoMemory", err)
	}
	// Not created as a side effect: creating it ourselves must succeed.
	cfg := vector.DefaultConfig()
	cfg.Dim, cfg.Metric, cfg.FullText = 4, vector.Cosine, &vector.FullTextConfig{}
	if err := st.CreateCollection(context.Background(), memCollection, cfg); err != nil {
		t.Fatalf("read-only mode created the collection: %v", err)
	}
}

func TestReadOnlyIgnoresEmbedIdentity(t *testing.T) {
	st := newHeapStore(t)
	w := newTestServer(t, Config{Store: st, Embedder: fakeEmbedder{}})
	seedMemory(t, w, "p", "a.go", "hello", 1)
	r, err := NewServer(context.Background(), Config{Store: st, ReadOnly: true})
	if err != nil {
		t.Fatalf("NewServer read-only: %v", err)
	}
	got, err := r.FileMemories(context.Background(), "p", "a.go", 5)
	if err != nil || len(got) != 1 {
		t.Fatalf("got %+v, %v", got, err)
	}
}

// A read-only server that WAS given an embedder still has to match the store.
func TestReadOnlyWithEmbedderChecksIdentity(t *testing.T) {
	st := newHeapStore(t)
	w := newTestServer(t, Config{Store: st}) // bootstrapped with the stub embedder
	seedMemory(t, w, "p", "a.go", "hello", 1)
	if _, err := NewServer(context.Background(), Config{Store: st, ReadOnly: true, Embedder: fakeEmbedder{}}); !errors.Is(err, ErrEmbedderMismatch) {
		t.Fatalf("read-only NewServer with a mismatched embedder: err = %v, want ErrEmbedderMismatch", err)
	}
}

func TestReadOnlyRejectsWrites(t *testing.T) {
	st := newHeapStore(t)
	w := newTestServer(t, Config{Store: st})
	seedMemory(t, w, "p", "", "x", 1)
	c := startServer(t, Config{Store: st, ReadOnly: true})
	c.initialize()
	if msg := c.callTool("remember", map[string]any{"content": "nope", "namespace": "p"}, nil, true); !strings.Contains(msg, "read-only") {
		t.Errorf("remember error = %q, want the read-only refusal", msg)
	}
	got, err := w.RecentMemories(context.Background(), "p", 10)
	if err != nil || len(got) != 1 || got[0].Content != "x" {
		t.Fatalf("after a refused remember the namespace holds %+v (%v); want only the seeded memory", got, err)
	}
}

// failingScroll is a store whose VectorScroll always fails with err.
type failingScroll struct {
	rostam.Store
	err error
}

func (f failingScroll) VectorScroll(context.Context, string, rostam.VectorFilter, int, rostam.VectorScrollOpts) ([]rostam.VectorDocument, rostam.FanMeta, string, error) {
	return nil, rostam.FanMeta{}, "", f.err
}

func TestProbeMemoryMissingSpellings(t *testing.T) {
	for _, msg := range []string{"unknown collection mcp_memory", "no collection \"mcp_memory\"", "rpc: No Collection mcp_memory",
		// verbatim from a real remote (memory_e2e_test.go)
		`client: server error on op "vector_scroll": ops: unknown collection "mcp_memory"`} {
		s := newTestServer(t, Config{Store: failingScroll{newHeapStore(t), errors.New(msg)}, ReadOnly: true})
		if _, err := s.RecentMemories(context.Background(), "p", 5); !errors.Is(err, ErrNoMemory) {
			t.Errorf("%q: err = %v, want ErrNoMemory", msg, err)
		}
	}
}

func TestProbeMemoryTransientErrorNotNoMemory(t *testing.T) {
	s := newTestServer(t, Config{Store: failingScroll{newHeapStore(t), errors.New("connection reset by peer")}, ReadOnly: true})
	_, err := s.RecentMemories(context.Background(), "p", 5)
	if err == nil || errors.Is(err, ErrNoMemory) {
		t.Fatalf("err = %v, want a non-ErrNoMemory error", err)
	}
}
