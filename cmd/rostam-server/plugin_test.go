// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// pluginDir is the Claude Code plugin shipped in this repo.
const pluginDir = "../../integrations/claude-code-plugin"

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s is not valid JSON: %v", path, err)
	}
}

func TestPluginFilesValid(t *testing.T) {
	var manifest struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	}
	readJSON(t, filepath.Join(pluginDir, ".claude-plugin", "plugin.json"), &manifest)
	if manifest.Name != "rostam-memory" || manifest.Version == "" {
		t.Fatalf("plugin.json name/version = %q/%q", manifest.Name, manifest.Version)
	}

	var mcp struct {
		Servers map[string]struct {
			Command string            `json:"command"`
			Args    []string          `json:"args"`
			Env     map[string]string `json:"env"`
		} `json:"mcpServers"`
	}
	readJSON(t, filepath.Join(pluginDir, ".mcp.json"), &mcp)
	srv, ok := mcp.Servers["rostam"]
	if !ok || srv.Command != "rostam-server" || !reflect.DeepEqual(srv.Args, []string{"mcp", "-autostart"}) {
		t.Fatalf(".mcp.json rostam server = %+v", srv)
	}
	// The embedder env is passed through with an empty default, and an empty
	// value must behave exactly like an unset one (BM25-only, no error).
	empty := map[string]string{}
	for k, v := range srv.Env {
		if !strings.HasPrefix(k, "ROSTAM_EMBED_") || v != "${"+k+":-}" {
			t.Errorf(".mcp.json env %s=%q, want ${%s:-}", k, v, k)
		}
		empty[k] = ""
	}
	if _, ok := srv.Env["ROSTAM_EMBED_LOCAL"]; !ok {
		t.Error(".mcp.json must pass ROSTAM_EMBED_LOCAL through")
	}
	if emb, err := embedderFromEnv(func(k string) (string, bool) { v, ok := empty[k]; return v, ok }); emb != nil || err != nil {
		t.Errorf("all-empty embedder env = %v, %v; want BM25-only", emb, err)
	}

	var hooks struct {
		Hooks map[string][]struct {
			Matcher string `json:"matcher"`
			Hooks   []struct {
				Type    string `json:"type"`
				Command string `json:"command"`
			} `json:"hooks"`
		} `json:"hooks"`
	}
	readJSON(t, filepath.Join(pluginDir, "hooks", "hooks.json"), &hooks)
	want := map[string]struct{ matcher, command string }{
		"SessionStart": {"startup|resume|clear|compact", "rostam-server hook session-start"},
		"PreToolUse":   {"Read|Edit|Write|MultiEdit", "rostam-server hook pre-tool-use"},
	}
	for event, w := range want {
		groups := hooks.Hooks[event]
		if len(groups) != 1 || len(groups[0].Hooks) != 1 {
			t.Fatalf("%s: want one group with one hook, got %+v", event, groups)
		}
		g := groups[0]
		if g.Matcher != w.matcher {
			t.Errorf("%s matcher = %q, want %q", event, g.Matcher, w.matcher)
		}
		if g.Hooks[0].Type != "command" || g.Hooks[0].Command != w.command {
			t.Errorf("%s hook = %+v, want command %q", event, g.Hooks[0], w.command)
		}
	}

	var market struct {
		Name    string `json:"name"`
		Plugins []struct {
			Name   string `json:"name"`
			Source string `json:"source"`
		} `json:"plugins"`
	}
	readJSON(t, "../../.claude-plugin/marketplace.json", &market)
	if market.Name == "" || len(market.Plugins) != 1 || market.Plugins[0].Name != manifest.Name {
		t.Fatalf("marketplace.json = %+v", market)
	}
	src := filepath.Join("../..", market.Plugins[0].Source, ".claude-plugin", "plugin.json")
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("marketplace source does not point at the plugin: %v", err)
	}

	skill, err := os.ReadFile(filepath.Join(pluginDir, "skills", "rostam-memory", "SKILL.md"))
	if err != nil || len(skill) == 0 {
		t.Fatalf("skill missing: %v", err)
	}
}
