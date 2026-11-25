// SPDX-License-Identifier: Apache-2.0
//go:build linux

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/mcp"
	"github.com/rostamlabs/rostam/ops"
)

// e2eEnv is the sandbox every child process of TestMemoryE2E runs in: a temp
// HOME (so ~/.rostam is never the developer's), BM25-only embedding, and none
// of the ambient ROSTAM_* / CLAUDE_* configuration.
type e2eEnv struct {
	t    *testing.T
	bin  string
	home string
	repo string // temp git checkout used as the project
	ns   string // memory namespace == basename of repo
	data string // <home>/.rostam/memory
}

func (e *e2eEnv) environ(extra ...string) []string {
	var out []string
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "ROSTAM_") || strings.HasPrefix(kv, "CLAUDE_") || strings.HasPrefix(kv, "HOME=") {
			continue
		}
		out = append(out, kv)
	}
	out = append(out, "HOME="+e.home, "ROSTAM_EMBED_ENDPOINT=")
	return append(out, extra...)
}

// hook runs `rostam-server hook <event>` as a real process and returns its
// stdout, exit code and wall time.
func (e *e2eEnv) hook(event string, payload map[string]any) (string, int, time.Duration) {
	e.t.Helper()
	b, _ := json.Marshal(payload)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, e.bin, "hook", event)
	cmd.Env = e.environ()
	cmd.Stdin = strings.NewReader(string(b))
	var stdout, stderr syncBuffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	start := time.Now()
	err := cmd.Run()
	dur := time.Since(start)
	code := 0
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			e.t.Fatalf("running hook %s: %v", event, err)
		}
		code = ee.ExitCode()
	}
	return stdout.String(), code, dur
}

func (e *e2eEnv) preToolUse(session string) map[string]any {
	return map[string]any{
		"session_id":      session,
		"cwd":             e.repo,
		"hook_event_name": "PreToolUse",
		"tool_name":       "Read",
		"tool_input":      map[string]any{"file_path": filepath.Join(e.repo, "main.go")},
	}
}

func (e *e2eEnv) sessionStart(session, source string) map[string]any {
	return map[string]any{
		"session_id":      session,
		"cwd":             e.repo,
		"hook_event_name": "SessionStart",
		"source":          source,
	}
}

// wantSilent asserts the hook printed nothing and exited 0.
func (e *e2eEnv) wantSilent(what, out string, code int) {
	e.t.Helper()
	if out != "" || code != 0 {
		e.t.Fatalf("%s: want empty stdout and exit 0, got exit %d stdout %q", what, code, out)
	}
}

// wantInjected asserts out is a valid hookSpecificOutput JSON object whose
// additionalContext contains every want substring, and returns the context.
func (e *e2eEnv) wantInjected(what, out string, code int, event string, want ...string) string {
	e.t.Helper()
	if code != 0 {
		e.t.Fatalf("%s: exit %d", what, code)
	}
	var o struct {
		H struct {
			Event string `json:"hookEventName"`
			Ctx   string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal([]byte(out), &o); err != nil {
		e.t.Fatalf("%s: stdout is not valid JSON: %v\n%q", what, err, out)
	}
	if o.H.Event != event {
		e.t.Fatalf("%s: hookEventName %q, want %q", what, o.H.Event, event)
	}
	for _, w := range want {
		if !strings.Contains(o.H.Ctx, w) {
			e.t.Fatalf("%s: additionalContext lacks %q:\n%s", what, w, o.H.Ctx)
		}
	}
	return o.H.Ctx
}

// startMcpAutostart launches `mcp -autostart -project-root <repo>`.
func (e *e2eEnv) startMcpAutostart() *mcpProc {
	e.t.Helper()
	return e.startMcpAutostartIn(e.repo)
}

// startMcpAutostartIn launches `mcp -autostart` as Claude Code would for a
// session opened in dir: no -project-root, CLAUDE_PROJECT_DIR=dir.
func (e *e2eEnv) startMcpAutostartIn(dir string) *mcpProc {
	e.t.Helper()
	args := []string{"mcp", "-autostart"}
	if dir == e.repo {
		args = append(args, "-project-root", e.repo)
	}
	cmd := exec.Command(e.bin, args...)
	cmd.Env = e.environ("CLAUDE_PROJECT_DIR=" + dir)
	in, err := cmd.StdinPipe()
	if err != nil {
		e.t.Fatal(err)
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		e.t.Fatal(err)
	}
	logs := &syncBuffer{}
	cmd.Stderr = logs
	if err := cmd.Start(); err != nil {
		e.t.Fatalf("starting mcp -autostart: %v", err)
	}
	p := &mcpProc{t: e.t, cmd: cmd, in: in, out: bufio.NewScanner(out), logs: logs}
	p.out.Buffer(make([]byte, 64<<10), 16<<20)
	e.t.Cleanup(func() {
		_ = in.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	return p
}

// procs lists the pids of every live process whose argv[0] is this test's
// private binary, with their argv.
func (e *e2eEnv) procs() map[int][]string {
	ents, _ := os.ReadDir("/proc")
	out := map[int][]string{}
	for _, de := range ents {
		pid, err := strconv.Atoi(de.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile("/proc/" + de.Name() + "/cmdline")
		if err != nil || len(b) == 0 {
			continue
		}
		args := strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
		if args[0] == e.bin {
			out[pid] = args
		}
	}
	return out
}

func (e *e2eEnv) daemons() (supervisors, servers []int) {
	for pid, args := range e.procs() {
		switch {
		case len(args) > 1 && args[1] == "memory-daemon":
			supervisors = append(supervisors, pid)
		case containsArg(args, "-config") && containsArg(args, filepath.Join(e.data, daemonConfigName)):
			servers = append(servers, pid)
		}
	}
	return
}

func containsArg(args []string, a string) bool {
	for _, x := range args {
		if x == a {
			return true
		}
	}
	return false
}

func (e *e2eEnv) stopDaemon() error {
	cmd := exec.Command(e.bin, "memory-daemon", "-stop")
	cmd.Env = e.environ()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, out)
	}
	return nil
}

// waitFor polls cond until it holds or the deadline passes.
func waitFor(t *testing.T, what string, d time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s", d, what)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func portFree(addr string) bool {
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return false
	}
	_ = l.Close()
	return true
}

func (e *e2eEnv) daemonUp() bool {
	c, err := net.DialTimeout("tcp", memoryDefaultAddr, 200*time.Millisecond)
	if err != nil {
		return false
	}
	_ = c.Close()
	return true
}

// openStore connects to the running daemon with its token, as the hook does.
func (e *e2eEnv) openStore() rostam.Store {
	e.t.Helper()
	tok, err := os.ReadFile(filepath.Join(e.data, memoryTokenName))
	if err != nil {
		e.t.Fatalf("reading daemon token: %v", err)
	}
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		e.t.Fatal(err)
	}
	st, err := connectStore(storeFlags{connect: memoryDefaultAddr, authToken: strings.TrimSpace(string(tok))}, reg,
		func(string) (string, bool) { return "", false })
	if err != nil {
		e.t.Fatalf("connecting to daemon: %v", err)
	}
	e.t.Cleanup(func() { _ = st.Close() })
	return st
}

func TestMemoryE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("builds rostam-server and runs a real memory daemon")
	}
	// -autostart has no address flag: the daemon always listens on the
	// well-known loopback port, so this test needs it free.
	if !portFree(memoryDefaultAddr) {
		t.Skipf("%s is in use (a real memory daemon?); the e2e needs the well-known port", memoryDefaultAddr)
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	bin := buildServer(t)
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "widgets")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	e := &e2eEnv{t: t, bin: bin, home: home, repo: repo, ns: "widgets", data: filepath.Join(home, ".rostam", "memory")}
	if out, err := exec.Command("git", "-C", repo, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	mainGo := filepath.Join(repo, "main.go")
	if err := os.WriteFile(mainGo, []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A commit, so a linked worktree can be added for the namespace case.
	run(t, repo, "git", "add", "main.go")
	run(t, repo, "git", "commit", "-q", "-m", "init")

	// Cleanup order (LIFO): this runs after every subtest and after the
	// per-session cleanups. It stops the daemon cleanly, then kills anything
	// still running our private binary, and fails loudly if one survives.
	t.Cleanup(func() {
		if t.Failed() {
			if b, err := os.ReadFile(filepath.Join(e.data, daemonLogName)); err == nil {
				t.Logf("daemon.log:\n%s", b)
			}
		}
		_ = e.stopDaemon()
		for pid := range e.procs() {
			_ = syscall.Kill(pid, syscall.SIGKILL)
		}
		deadline := time.Now().Add(5 * time.Second)
		for len(e.procs()) > 0 && time.Now().Before(deadline) {
			time.Sleep(25 * time.Millisecond)
		}
		if left := e.procs(); len(left) > 0 {
			t.Errorf("processes left running after the test: %v", left)
		}
	})

	t.Run("a_no_daemon_and_missing_collection", func(t *testing.T) {
		out, code, _ := e.hook("pre-tool-use", e.preToolUse("s-a1"))
		e.wantSilent("hook with no daemon", out, code)
		out, code, _ = e.hook("session-start", e.sessionStart("s-a1", "startup"))
		e.wantSilent("session-start with no daemon", out, code)

		// Start the daemon by hand: an `mcp -autostart` session would create
		// the memory collection, which is exactly what must not exist yet.
		logf, err := os.OpenFile(filepath.Join(t.TempDir(), "daemon-out.log"), os.O_CREATE|os.O_WRONLY, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(bin, "memory-daemon")
		cmd.Env = e.environ()
		cmd.Stdout, cmd.Stderr = logf, logf
		if err := cmd.Start(); err != nil {
			t.Fatalf("starting daemon: %v", err)
		}
		t.Cleanup(func() { _ = logf.Close() })
		go func() { _ = cmd.Wait() }()
		waitFor(t, "daemon to listen", 20*time.Second, e.daemonUp)

		out, code, _ = e.hook("pre-tool-use", e.preToolUse("s-a2"))
		e.wantSilent("pre-tool-use against a daemon with no collection", out, code)
		out, code, _ = e.hook("session-start", e.sessionStart("s-a2", "startup"))
		e.wantSilent("session-start against a daemon with no collection", out, code)

		// The real remote's error for the missing collection must read as
		// ErrNoMemory (not a generic failure), and the read must not have
		// created the collection.
		st := e.openStore()
		_, _, _, serr := st.VectorScroll(context.Background(), "mcp_memory", rostam.VectorFilter{}, 1, rostam.VectorScrollOpts{})
		if serr == nil {
			t.Fatal("mcp_memory exists before any memory was stored")
		}
		t.Logf("real remote missing-collection error: %q", serr.Error())
		srv, err := mcp.NewServer(context.Background(), mcp.Config{Store: st, ReadOnly: true, ProjectRoot: repo})
		if err != nil {
			t.Fatalf("read-only NewServer against empty daemon: %v", err)
		}
		if _, err := srv.RecentMemories(context.Background(), e.ns, 5); !errors.Is(err, mcp.ErrNoMemory) {
			t.Fatalf("RecentMemories on a missing collection: got %v, want mcp.ErrNoMemory", err)
		}
		if _, _, _, serr2 := st.VectorScroll(context.Background(), "mcp_memory", rostam.VectorFilter{}, 1, rostam.VectorScrollOpts{}); serr2 == nil {
			t.Fatal("a read-only session created the memory collection")
		}
	})

	var alpha, beta *mcpProc
	t.Run("b_two_sessions_share_memory", func(t *testing.T) {
		alpha, beta = e.startMcpAutostart(), e.startMcpAutostart()
		alpha.initialize()
		beta.initialize()
		alpha.callTool("remember", map[string]any{
			"content": "alpha rule: parse input before validate", "namespace": e.ns, "path": "main.go"})
		beta.callTool("remember", map[string]any{
			"content": "beta rule: gizmo handlers must be idempotent", "namespace": e.ns, "path": "main.go"})
		if got := alpha.callTool("recall", map[string]any{"query": "gizmo idempotent", "namespace": e.ns}); !strings.Contains(got, "gizmo") {
			t.Fatalf("session A cannot see session B's memory: %s", got)
		}
		if got := beta.callTool("recall", map[string]any{"query": "parse validate", "namespace": e.ns}); !strings.Contains(got, "alpha rule") {
			t.Fatalf("session B cannot see session A's memory: %s", got)
		}
	})

	t.Run("b2_worktree_memory_without_namespace_reaches_main_checkout", func(t *testing.T) {
		wt := filepath.Join(t.TempDir(), "widgets-feature")
		run(t, repo, "git", "worktree", "add", "-q", wt)
		p := e.startMcpAutostartIn(wt)
		p.initialize()
		// No namespace: the session resolves the main checkout's ("widgets").
		if got := p.callTool("remember", map[string]any{"content": "worktree rule: widgets flush on close", "path": "main.go"}); !strings.Contains(got, `"namespace":"widgets"`) {
			t.Fatalf("worktree session remembered into the wrong namespace: %s", got)
		}
		_ = p.in.Close() // drop its daemon connection before f stops the daemon
		out, code, _ := e.hook("pre-tool-use", e.preToolUse("s-b2"))
		e.wantInjected("main-checkout hook after a worktree remember", out, code, "PreToolUse", "worktree rule")
	})

	t.Run("c_pre_tool_use_injects_and_dedupes", func(t *testing.T) {
		out, code, _ := e.hook("pre-tool-use", e.preToolUse("s-c1"))
		ctx := e.wantInjected("first pre-tool-use", out, code, "PreToolUse", "alpha rule", "beta rule")
		if strings.Contains(ctx, "[stale") {
			t.Fatalf("fresh memories flagged stale:\n%s", ctx)
		}
		out, code, _ = e.hook("pre-tool-use", e.preToolUse("s-c1"))
		e.wantSilent("same session again (dedupe)", out, code)
		out, code, _ = e.hook("pre-tool-use", e.preToolUse("s-c2"))
		e.wantInjected("different session", out, code, "PreToolUse", "alpha rule", "beta rule")
	})

	t.Run("d_stale_after_edit_and_delete", func(t *testing.T) {
		if err := os.WriteFile(mainGo, []byte("package main\n\nfunc main() { println(1) }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		out, code, _ := e.hook("pre-tool-use", e.preToolUse("s-d1"))
		e.wantInjected("after edit", out, code, "PreToolUse", "[stale] ", "alpha rule")
		if err := os.Remove(mainGo); err != nil {
			t.Fatal(err)
		}
		out, code, _ = e.hook("pre-tool-use", e.preToolUse("s-d2"))
		e.wantInjected("after delete", out, code, "PreToolUse", "[stale: file missing]", "alpha rule")
	})

	t.Run("e_session_start_compact_resets_resume_does_not", func(t *testing.T) {
		out, code, _ := e.hook("session-start", e.sessionStart("s-e", "startup"))
		e.wantInjected("startup", out, code, "SessionStart", "rule")
		out, code, _ = e.hook("session-start", e.sessionStart("s-e", "startup"))
		e.wantSilent("startup again (dedupe)", out, code)
		out, code, _ = e.hook("session-start", e.sessionStart("s-e", "resume"))
		e.wantSilent("resume", out, code)
		out, code, _ = e.hook("session-start", e.sessionStart("s-e", "compact"))
		e.wantInjected("compact", out, code, "SessionStart", "rule")
	})

	t.Run("f_daemon_killed_hook_is_silent_and_fast", func(t *testing.T) {
		// Sessions hold open connections to the daemon; close them first so
		// the stop is a clean one.
		if alpha == nil || beta == nil {
			t.Skip("sessions from b did not start")
		}
		_ = alpha.in.Close()
		_ = beta.in.Close()
		if err := e.stopDaemon(); err != nil {
			t.Fatalf("memory-daemon -stop: %v", err)
		}
		waitFor(t, "daemon port to close", 10*time.Second, func() bool { return !e.daemonUp() })
		out, code, dur := e.hook("pre-tool-use", e.preToolUse("s-f"))
		e.wantSilent("hook with the daemon gone", out, code)
		if dur > 1500*time.Millisecond {
			t.Fatalf("hook took %s with the daemon gone, want ~1s at most", dur)
		}
	})

	t.Run("g_autostart_race_spawns_one_daemon", func(t *testing.T) {
		if sup, srv := e.daemons(); len(sup)+len(srv) != 0 {
			t.Fatalf("daemon processes still alive before the race: %v %v", sup, srv)
		}
		if !portFree(memoryDefaultAddr) {
			t.Fatalf("%s still bound before the race", memoryDefaultAddr)
		}
		const n = 5
		sessions := make([]*mcpProc, n)
		for i := range sessions {
			sessions[i] = e.startMcpAutostart() // all spawn before any is initialized
		}
		for i, s := range sessions {
			s.initialize()
			if i == 0 {
				// The store outlived the daemon restart.
				if got := s.callTool("recall", map[string]any{"query": "gizmo idempotent", "namespace": e.ns}); !strings.Contains(got, "gizmo") {
					t.Fatalf("memory lost across daemon restart: %s", got)
				}
			}
		}
		sup, srv := e.daemons()
		if len(sup) != 1 || len(srv) != 1 {
			t.Fatalf("want exactly 1 supervisor and 1 server, got supervisors=%v servers=%v\nall: %v", sup, srv, e.procs())
		}
		pid, err := os.ReadFile(filepath.Join(e.data, daemonPidName))
		if err != nil {
			t.Fatalf("pid file: %v", err)
		}
		if f := strings.Fields(string(pid)); len(f) != 2 || f[0] != strconv.Itoa(sup[0]) || f[1] != strconv.Itoa(srv[0]) {
			t.Fatalf("pid file %q does not match live daemon pids %v/%v", pid, sup, srv)
		}
	})
}
