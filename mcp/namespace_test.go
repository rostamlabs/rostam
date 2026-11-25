// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"encoding/json"
	"strings"
	"testing"
)

// nsEnv starts a server with the given default namespace and one file to link.
func nsEnv(t *testing.T, defNS string) *testClient {
	t.Helper()
	root := t.TempDir()
	writeFile(t, root, "a.go", "package a\n")
	c := startServer(t, Config{Store: newHeapStore(t), ProjectRoot: root, DefaultNamespace: defNS})
	c.initialize()
	return c
}

func TestDefaultNamespaceUsedWhenOmitted(t *testing.T) {
	c := nsEnv(t, "proj")
	var rem struct {
		Namespace string `json:"namespace"`
	}
	c.callTool("remember", map[string]any{"content": "the widget cache is per tenant", "path": "a.go", "key": "k1"}, &rem, false)
	if rem.Namespace != "proj" {
		t.Fatalf("remember with no namespace landed in %q, want proj", rem.Namespace)
	}

	var rec struct {
		Hits []json.RawMessage `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "widget cache"}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("recall with no namespace: %d hits, want 1", len(rec.Hits))
	}
	c.callTool("recall", map[string]any{"query": "widget cache", "namespace": "default"}, &rec, false)
	if len(rec.Hits) != 0 {
		t.Fatalf("recall in \"default\" must not see the project's memory, got %d hits", len(rec.Hits))
	}

	var file struct {
		Memories []json.RawMessage `json:"memories"`
	}
	c.callTool("recall_for_file", map[string]any{"path": "a.go"}, &file, false)
	if len(file.Memories) != 1 {
		t.Fatalf("recall_for_file with no namespace: %d memories, want 1", len(file.Memories))
	}

	var list struct {
		Memories []json.RawMessage `json:"memories"`
	}
	c.callTool("list_memories", map[string]any{}, &list, false)
	if len(list.Memories) != 1 {
		t.Fatalf("list_memories with no namespace: %d, want 1", len(list.Memories))
	}

	var fg struct {
		Deleted []json.RawMessage `json:"deleted"`
	}
	c.callTool("forget", map[string]any{"keys": []string{"k1"}}, &fg, false)
	if len(fg.Deleted) != 1 {
		t.Fatalf("forget by key with no namespace deleted %d, want 1", len(fg.Deleted))
	}
}

func TestDefaultNamespaceUnsetIsDefault(t *testing.T) {
	c := nsEnv(t, "")
	var rem struct {
		Namespace string `json:"namespace"`
	}
	c.callTool("remember", map[string]any{"content": "x"}, &rem, false)
	if rem.Namespace != "default" {
		t.Fatalf("namespace = %q, want default", rem.Namespace)
	}
}

func TestInstructionsNameDefaultNamespace(t *testing.T) {
	get := func(defNS string) string {
		c := startServer(t, Config{Store: newHeapStore(t), DefaultNamespace: defNS})
		var init struct {
			Instructions string `json:"instructions"`
		}
		res := c.rpc("initialize", map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{}, "clientInfo": map[string]any{"name": "t", "version": "0"}}, false)
		if err := json.Unmarshal(res, &init); err != nil {
			t.Fatal(err)
		}
		return init.Instructions
	}
	set := get("myrepo")
	if !strings.Contains(set, `"myrepo"`) || !strings.Contains(strings.ToLower(set), "omit namespace") {
		t.Errorf("instructions with a default namespace must name it and say to omit namespace:\n%s", set)
	}
	for name, s := range map[string]string{"set": set, "unset": get("")} {
		if !strings.Contains(s, "worktree") || !strings.Contains(s, "main checkout") {
			t.Errorf("%s: instructions must say to use the main checkout's name, not a worktree's:\n%s", name, s)
		}
	}
}
