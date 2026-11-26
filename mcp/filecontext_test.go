// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

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
		md[linkField] = linkMarker()
	}
	if err := s.store.VectorUpsert(ctx, memCollection, memoryID(ns, content), vecs[0], content, rostam.VectorInsertOpts{Metadata: md}); err != nil {
		t.Fatal(err)
	}
}

// A memory written before code links existed may hold __path, __symbol or
// __file_hash as the user's own metadata. With no link marker it is not a
// link: the fields stay visible, nothing is flagged stale, and it is never
// returned as linked to that file.
func TestLegacyLinkNamedMetadataStaysUserData(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.go", "package a\n")
	s := newTestServer(t, Config{ProjectRoot: root})
	ctx := context.Background()
	if err := s.ensureMemory(ctx); err != nil {
		t.Fatal(err)
	}
	vecs, err := s.emb.Embed(ctx, []string{"legacy fact"})
	if err != nil {
		t.Fatal(err)
	}
	md := rostam.VectorMetadata{
		nsField:       vector.NewString("p"),
		createdField:  vector.NewInt(1),
		updatedField:  vector.NewInt(1),
		pathField:     vector.NewString("a.go"),
		symbolField:   vector.NewString("Mine"),
		fileHashField: vector.NewString("not-a-hash"),
	}
	if err := s.store.VectorUpsert(ctx, memCollection, memoryID("p", "legacy fact"), vecs[0], "legacy fact", rostam.VectorInsertOpts{Metadata: md}); err != nil {
		t.Fatal(err)
	}
	got, err := s.FileMemories(ctx, "p", "a.go", 5)
	if err != nil || len(got) != 0 {
		t.Fatalf("FileMemories = %+v, %v; a legacy __path is not a link", got, err)
	}
	all, err := s.RecentMemories(ctx, "p", 5)
	if err != nil || len(all) != 1 {
		t.Fatalf("RecentMemories = %+v, %v", all, err)
	}
	h := all[0]
	if h.Path != "" || h.Symbol != "" || h.Stale != nil {
		t.Fatalf("legacy memory read as a link: path %q symbol %q stale %v", h.Path, h.Symbol, h.Stale)
	}
	if h.Metadata[pathField] != "a.go" || h.Metadata[symbolField] != "Mine" || h.Metadata[fileHashField] != "not-a-hash" {
		t.Fatalf("legacy user metadata hidden: %v", h.Metadata)
	}
	// The same through recall, BM25 and the JSON (hybrid) conversion.
	c := startServer(t, Config{Store: s.store, ProjectRoot: root})
	c.initialize()
	var out struct {
		Hits []memoryHit `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "legacy fact", "namespace": "p"}, &out, false)
	if len(out.Hits) != 1 || out.Hits[0].Path != "" || out.Hits[0].Metadata[pathField] != "a.go" {
		t.Fatalf("recall = %+v", out.Hits)
	}
	hj := toMemoryHitJSON(1, "x", 0, metadataToJSON(md))
	if hj.Path != "" || hj.Metadata[pathField] != "a.go" {
		t.Fatalf("toMemoryHitJSON read a legacy __path as a link: %+v", hj)
	}
	md[linkField] = linkMarker()
	if hj := toMemoryHitJSON(1, "x", 0, metadataToJSON(md)); hj.Path != "a.go" || hj.Metadata[pathField] != nil || hj.Metadata[linkField] != nil {
		t.Fatalf("toMemoryHitJSON with the marker: %+v", hj)
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

// Memories written within one second still come back newest first, by the
// nanosecond write time remember stores alongside the seconds field; and a
// user field of that name that does not match the stored second stays data.
func TestMemoriesSubSecondOrder(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.go", "package a\n")
	c := startServer(t, Config{Store: newHeapStore(t), ProjectRoot: root})
	c.initialize()
	const n = 8
	for i := 0; i < n; i++ {
		c.callTool("remember", map[string]any{"content": fmt.Sprintf("burst-%d", i), "namespace": "p", "path": "a.go"}, nil, false)
		time.Sleep(time.Millisecond) // distinct nanosecond stamps even on coarse clocks
	}
	var out struct {
		Memories []memoryHit `json:"memories"`
	}
	c.callTool("recall_for_file", map[string]any{"path": "a.go", "namespace": "p", "k": 5}, &out, false)
	if len(out.Memories) != 5 {
		t.Fatalf("got %d memories", len(out.Memories))
	}
	for i, h := range out.Memories {
		if want := fmt.Sprintf("burst-%d", n-1-i); h.Content != want {
			t.Fatalf("position %d = %q, want %q (newest first)", i, h.Content, want)
		}
		if _, ok := h.Metadata[updatedNsField]; ok {
			t.Fatalf("own %s leaked into metadata: %v", updatedNsField, h.Metadata)
		}
	}
	hj := toMemoryHitJSON(1, "x", 0, map[string]any{updatedField: int64(100), updatedNsField: int64(5)})
	if hj.updatedNs != 0 || hj.Metadata[updatedNsField] != int64(5) {
		t.Fatalf("a mismatching %s was taken as ours: %+v", updatedNsField, hj)
	}
}

// One effective key for timestamped and older memories: a strict total order
// (the old two-level comparison formed a cycle across a shared second).
func TestNewerThanIsTotalOrder(t *testing.T) {
	const s = int64(1_700_000_000)
	hits := []memoryHit{
		{ID: 3, Updated: s, updatedNs: s*1e9 + 900},
		{ID: 1, Updated: s, updatedNs: s*1e9 + 100},
		{ID: 2, Updated: s}, // written before nanosecond times existed
	}
	for _, a := range hits {
		for _, b := range hits {
			if newerThan(a, b) && newerThan(b, a) {
				t.Fatalf("newerThan(%d,%d) and newerThan(%d,%d) both true", a.ID, b.ID, b.ID, a.ID)
			}
		}
	}
	// transitivity over every ordered triple
	for _, a := range hits {
		for _, b := range hits {
			for _, c := range hits {
				if newerThan(a, b) && newerThan(b, c) && !newerThan(a, c) {
					t.Fatalf("not transitive: %d > %d > %d but not %d > %d", a.ID, b.ID, c.ID, a.ID, c.ID)
				}
			}
		}
	}
	sort.Slice(hits, func(i, j int) bool { return newerThan(hits[i], hits[j]) })
	if hits[0].ID != 3 || hits[1].ID != 1 || hits[2].ID != 2 {
		t.Fatalf("order = %d,%d,%d; want 3,1,2", hits[0].ID, hits[1].ID, hits[2].ID)
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
