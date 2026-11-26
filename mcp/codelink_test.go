// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type linkHit struct {
	Content  string         `json:"content"`
	Path     string         `json:"path"`
	Symbol   string         `json:"symbol"`
	Stale    any            `json:"stale"`
	Metadata map[string]any `json:"metadata"`
}

// linkEnv starts a BM25-mode server rooted at a temp dir.
func linkEnv(t *testing.T) (*testClient, string) {
	t.Helper()
	root := t.TempDir()
	c := startServer(t, Config{Store: newHeapStore(t), ProjectRoot: root})
	c.initialize()
	return c, root
}

func writeFile(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func recallLinked(t *testing.T, c *testClient) linkHit {
	t.Helper()
	var out struct {
		Hits []linkHit `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "widget", "namespace": "p"}, &out, false)
	if len(out.Hits) != 1 {
		t.Fatalf("want 1 hit, got %d", len(out.Hits))
	}
	return out.Hits[0]
}

func TestNormalizeLinkPath(t *testing.T) {
	root := t.TempDir()
	ok := map[string]string{
		"a/b.go":                         "a/b.go",
		filepath.Join(root, "a", "b.go"): "a/b.go",
		"./a/../a/b.go":                  "a/b.go",
	}
	for in, want := range ok {
		got, err := normalizeLinkPath(root, in)
		if err != nil || got != want {
			t.Errorf("normalizeLinkPath(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, in := range []string{"../x", "/etc/passwd", "", "a/../../x", "."} {
		if got, err := normalizeLinkPath(root, in); err == nil {
			t.Errorf("normalizeLinkPath(%q) = %q, want error", in, got)
		}
	}
}

func TestRememberWithPathStoresHash(t *testing.T) {
	c, root := linkEnv(t)
	writeFile(t, root, "a.go", "package a")
	c.callTool("remember", map[string]any{"content": "the widget is fragile", "namespace": "p", "path": "a.go", "symbol": "Widget"}, nil, false)
	h := recallLinked(t, c)
	if h.Path != "a.go" || h.Symbol != "Widget" {
		t.Fatalf("path/symbol = %q/%q", h.Path, h.Symbol)
	}
	if h.Stale != nil {
		t.Fatalf("stale = %v, want none", h.Stale)
	}
	for _, k := range []string{"__path", "__symbol", "__file_hash", "__link"} {
		if _, ok := h.Metadata[k]; ok {
			t.Errorf("metadata leaks %s", k)
		}
	}
}

func TestRecallMarksStaleAfterEdit(t *testing.T) {
	c, root := linkEnv(t)
	writeFile(t, root, "a.go", "package a")
	c.callTool("remember", map[string]any{"content": "the widget is fragile", "namespace": "p", "path": "a.go"}, nil, false)
	writeFile(t, root, "a.go", "package a // edited")
	if h := recallLinked(t, c); h.Stale != true {
		t.Fatalf("stale = %v, want true", h.Stale)
	}
}

func TestRecallMarksFileMissing(t *testing.T) {
	c, root := linkEnv(t)
	writeFile(t, root, "a.go", "package a")
	c.callTool("remember", map[string]any{"content": "the widget is fragile", "namespace": "p", "path": "a.go"}, nil, false)
	if err := os.Remove(filepath.Join(root, "a.go")); err != nil {
		t.Fatal(err)
	}
	if h := recallLinked(t, c); h.Stale != "file missing" {
		t.Fatalf("stale = %v, want \"file missing\"", h.Stale)
	}
}

func TestRememberMissingFileStillSucceeds(t *testing.T) {
	c, root := linkEnv(t)
	c.callTool("remember", map[string]any{"content": "the widget is fragile", "namespace": "p", "path": "nope.go"}, nil, false)
	h := recallLinked(t, c)
	if h.Path != "nope.go" || h.Stale != nil {
		t.Fatalf("path=%q stale=%v", h.Path, h.Stale)
	}
	writeFile(t, root, "nope.go", "package a")
	if h := recallLinked(t, c); h.Stale != true {
		t.Fatalf("stale after file appears = %v, want true", h.Stale)
	}
}

func TestRememberRejectsEscapingPath(t *testing.T) {
	c, _ := linkEnv(t)
	msg := c.callTool("remember", map[string]any{"content": "x", "namespace": "p", "path": "../x"}, nil, true)
	if !strings.Contains(msg, "outside the project root") {
		t.Errorf("error %q does not name the escaping path", msg)
	}
}

func TestRememberPathReservedKeysRejectedInMetadata(t *testing.T) {
	c, _ := linkEnv(t)
	for _, k := range []string{"__path", "__symbol", "__file_hash", "__link"} {
		msg := c.callTool("remember", map[string]any{"content": "x", "namespace": "p", "metadata": map[string]any{k: "v"}}, nil, true)
		if !strings.Contains(msg, "reserved") {
			t.Errorf("%s: error %q", k, msg)
		}
	}
}

func TestRememberRejectsEmptyOrPaddedPath(t *testing.T) {
	c, _ := linkEnv(t)
	for _, p := range []string{"", " a.go", "a.go ", "  "} {
		msg := c.callTool("remember", map[string]any{"content": "x", "namespace": "p", "path": p}, nil, true)
		if !strings.Contains(msg, "non-empty") || !strings.Contains(msg, "whitespace") {
			t.Errorf("path %q: error %q is not the padded-path rule", p, msg)
		}
	}
}

func TestRecallHybridKeepsLink(t *testing.T) {
	root := t.TempDir()
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}, ProjectRoot: root})
	c.initialize()
	writeFile(t, root, "a.go", "package a")
	c.callTool("remember", map[string]any{"content": "a: widget is fragile", "namespace": "p", "path": "a.go", "symbol": "Widget"}, nil, false)
	writeFile(t, root, "a.go", "package a // edited")
	h := recallLinked(t, c)
	if h.Path != "a.go" || h.Symbol != "Widget" || h.Stale != true {
		t.Fatalf("hybrid hit = %+v", h)
	}
	for _, k := range []string{"__path", "__symbol", "__file_hash", "__link"} {
		if _, ok := h.Metadata[k]; ok {
			t.Errorf("metadata leaked %s", k)
		}
	}
}

func TestInstructionsMentionPath(t *testing.T) {
	if !strings.Contains(instructionsFor(""), "path") || !strings.Contains(instructionsFor(""), "recall_for_file") {
		t.Fatalf("instructions must teach the path argument and recall_for_file:\n%s", instructions)
	}
}

func TestMarkStaleHashesEachPathOnce(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.go", "package a\n")
	writeFile(t, root, "b.go", "package b\n")
	s := newTestServer(t, Config{ProjectRoot: root})
	for i := 0; i < 12; i++ {
		p := "a.go"
		if i%4 == 3 {
			p = "b.go"
		}
		seedMemory(t, s, "p", p, "fact "+strings.Repeat("x", i), int64(100+i))
	}
	calls := map[string]int{}
	orig := hashFileFn
	hashFileFn = func(root, rel string) (string, fileState) {
		calls[rel]++
		return orig(root, rel)
	}
	t.Cleanup(func() { hashFileFn = orig })

	hits, err := s.RecentMemories(context.Background(), "p", 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 12 {
		t.Fatalf("want 12 hits, got %d", len(hits))
	}
	if calls["a.go"] != 1 || calls["b.go"] != 1 {
		t.Fatalf("each path must be hashed once per call, got %v", calls)
	}
	for _, h := range hits {
		if h.Stale != true { // seeded with an empty stored hash: the file exists, so it changed
			t.Fatalf("hit %q: stale = %v, want true", h.Content, h.Stale)
		}
	}
}
