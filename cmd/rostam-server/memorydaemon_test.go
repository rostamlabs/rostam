// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"flag"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
)

func TestEnsureTokenCreatesAndReuses(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "memory")
	tok, err := ensureToken(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(tok) != 64 {
		t.Fatalf("token = %q, want 64 hex chars (32 random bytes)", tok)
	}
	again, err := ensureToken(dir)
	if err != nil || again != tok {
		t.Fatalf("second call = %q, %v; want the same token", again, err)
	}
	if runtime.GOOS != "windows" {
		fi, err := os.Stat(filepath.Join(dir, "token"))
		if err != nil {
			t.Fatal(err)
		}
		if fi.Mode().Perm() != 0o600 {
			t.Fatalf("token mode = %v, want 0600", fi.Mode().Perm())
		}
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.Name() != "token" {
			t.Fatalf("stray file %q left behind by token creation", e.Name())
		}
	}
}

func TestEnsureTokenConcurrentFirstRunsAgree(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "memory")
	const n = 12
	got := make([]string, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tok, err := ensureToken(dir)
			if err != nil {
				t.Error(err)
			}
			got[i] = tok
		}()
	}
	wg.Wait()
	for _, g := range got {
		if g == "" || g != got[0] {
			t.Fatalf("concurrent first runs produced different tokens: %v", got)
		}
	}
}

func TestEnsureTokenRepairsLoosePermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("no POSIX modes")
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "token")
	if err := os.WriteFile(p, []byte("abc\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := ensureToken(dir)
	if err != nil || tok != "abc" {
		t.Fatalf("got %q, %v", tok, err)
	}
	if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %v, want 0600", fi.Mode().Perm())
	}
}

func TestDaemonRefusesNonLoopback(t *testing.T) {
	for _, addr := range []string{
		"127.0.0.1:7480", "[::1]:7480", "127.0.0.2:1",
	} {
		if err := validateLoopbackAddr(addr); err != nil {
			t.Errorf("%q should be accepted: %v", addr, err)
		}
	}
	for _, addr := range []string{
		"", ":7480", "0.0.0.0:7480", "[::]:7480", "localhost:7480", "example.com:7480",
		"10.0.0.5:7480", "192.168.1.2:7480", "127.0.0.1", "127.0.0.1:notaport", "[::ffff:10.0.0.1]:1",
	} {
		if err := validateLoopbackAddr(addr); err == nil {
			t.Errorf("%q must be refused", addr)
		}
	}
}

// fakeDaemon is a loopback listener that only starts answering once start() is
// called, standing in for a spawned daemon. A client that sends the right
// token line gets "ok".
type fakeDaemon struct {
	ln    net.Listener
	token string
	once  sync.Once
}

func newFakeDaemon(t *testing.T, token string) *fakeDaemon {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	return &fakeDaemon{ln: ln, token: token}
}

func (f *fakeDaemon) addr() string { return f.ln.Addr().String() }

func (f *fakeDaemon) start() {
	f.once.Do(func() {
		go func() {
			for {
				c, err := f.ln.Accept()
				if err != nil {
					return
				}
				go func() {
					defer c.Close()
					line, _ := bufio.NewReader(c).ReadString('\n')
					if strings.TrimSpace(line) == f.token {
						_, _ = c.Write([]byte("ok\n"))
					}
				}()
			}
		}()
	})
}

func fakeProbe(ctx context.Context, addr, token string) error {
	d := net.Dialer{Timeout: 200 * time.Millisecond}
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return err
	}
	defer c.Close()
	_ = c.SetDeadline(time.Now().Add(200 * time.Millisecond))
	if _, err := c.Write([]byte(token + "\n")); err != nil {
		return err
	}
	line, err := bufio.NewReader(c).ReadString('\n')
	if err != nil {
		return err
	}
	if strings.TrimSpace(line) != "ok" {
		return os.ErrInvalid
	}
	return nil
}

func TestEnsureDaemonSpawnsOnce(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	tok, err := ensureToken(data)
	if err != nil {
		t.Fatal(err)
	}
	fd := newFakeDaemon(t, tok)
	var spawns atomic.Int32
	spawn := func() error {
		spawns.Add(1)
		fd.start()
		return nil
	}

	const n = 5
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = ensureDaemonWith(context.Background(), data, fd.addr(), spawn, fakeProbe, 5*time.Second)
		}()
	}
	wg.Wait()
	for i, e := range errs {
		if e != nil {
			t.Errorf("session %d: %v", i, e)
		}
	}
	if got := spawns.Load(); got != 1 {
		t.Fatalf("spawn called %d times, want exactly 1", got)
	}
}

func TestEnsureDaemonAlreadyUpDoesNotSpawn(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	tok, _ := ensureToken(data)
	fd := newFakeDaemon(t, tok)
	fd.start()
	spawn := func() error { t.Error("spawn must not be called"); return nil }
	if err := ensureDaemonWith(context.Background(), data, fd.addr(), spawn, fakeProbe, time.Second); err != nil {
		t.Fatal(err)
	}
}

func TestEnsureDaemonFailureNamesLog(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	fd := newFakeDaemon(t, "never-started") // nothing accepts
	err := ensureDaemonWith(context.Background(), data, fd.addr(),
		func() error { return nil }, fakeProbe, 300*time.Millisecond)
	if err == nil {
		t.Fatal("expected failure")
	}
	if !strings.Contains(err.Error(), filepath.Join(data, "daemon.log")) {
		t.Fatalf("error %q does not name the log file", err)
	}
	// A spawn failure names it too.
	err = ensureDaemonWith(context.Background(), data, fd.addr(),
		func() error { return os.ErrPermission }, fakeProbe, 300*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "daemon.log") {
		t.Fatalf("spawn failure error = %v", err)
	}
}

// Exercises the real `mcp` flag set: -autostart combined with any flag that
// selects another store or credentials is refused.
func TestMcpAutostartFlagConflicts(t *testing.T) {
	parse := func(args ...string) error {
		fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		var fl mcpFlags
		defineMcpFlags(fs, &fl)
		if err := fs.Parse(args); err != nil {
			t.Fatalf("parse %v: %v", args, err)
		}
		return checkAutostartConflicts(visitedFlags(fs))
	}
	if err := parse("-autostart"); err != nil {
		t.Errorf("plain -autostart: %v", err)
	}
	if err := parse("-autostart", "-project-root", "/x", "-enable-destructive"); err != nil {
		t.Errorf("unrelated flags: %v", err)
	}
	for _, args := range [][]string{
		{"-autostart", "-data", "/d"},
		{"-autostart", "-data", ""},
		{"-autostart", "-connect", "h:1"},
		{"-autostart", "-auth-token", "t"},
		{"-autostart", "-tls-ca", "ca.pem"},
		{"-autostart", "-tls-cert", "c.pem"},
		{"-autostart", "-tls-key", "k.pem"},
		{"-autostart", "-tls-server-name", "n"},
	} {
		err := parse(args...)
		if err == nil || !strings.Contains(err.Error(), args[1]) {
			t.Errorf("%v: err = %v, want a refusal naming %s", args, err, args[1])
		}
	}
}

// A listener that is up but wants a different token (token file deleted and
// regenerated while the daemon runs) must be reported as such, at once, with
// no spawn attempt, not as a 5 s startup timeout.
func TestEnsureDaemonRejectedTokenFailsFast(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	tok, err := ensureToken(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, serverKey string
		wantErr         bool
	}{
		{"wrong token", "some-other-key", true},
		{"right token", tok, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			addr := ln.Addr().String()
			_ = ln.Close()
			reg := ops.NewRegistry()
			if err := ops.RegisterBuiltins(reg); err != nil {
				t.Fatal(err)
			}
			srv, err := rostam.NewServer(withAuth(rostam.ServerConfig{
				TCPAddr:      addr,
				DirectConfig: rostam.DirectConfig{Ops: reg},
			}, staticKeyAuthenticator(tc.serverKey)))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = srv.Close() })

			var spawns atomic.Int32
			start := time.Now()
			err = ensureDaemon(context.Background(), data, addr, func() error { spawns.Add(1); return nil })
			if el := time.Since(start); el > 3*time.Second {
				t.Errorf("took %v, want well under the 5s startup wait", el)
			}
			if spawns.Load() != 0 {
				t.Errorf("spawn called %d times; a listening server must never trigger a spawn", spawns.Load())
			}
			switch {
			case tc.wantErr && (err == nil || !strings.Contains(err.Error(), "rejected the token") || !strings.Contains(err.Error(), filepath.Join(data, "token"))):
				t.Errorf("err = %v, want the rejected-token error naming the token file", err)
			case !tc.wantErr && err != nil:
				t.Errorf("right token: %v", err)
			}
		})
	}
}

func TestAutostartSharesHookDefaults(t *testing.T) {
	if memoryDefaultAddr != "127.0.0.1:7480" {
		t.Fatalf("default addr = %s", memoryDefaultAddr)
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	dir, err := defaultMemoryDir()
	if err != nil || dir != filepath.Join(home, ".rostam", "memory") {
		t.Fatalf("defaultMemoryDir = %q, %v", dir, err)
	}
}

func TestProjectRootDefaults(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	repo, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "init", "-q", repo).CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, out)
	}
	sub := filepath.Join(repo, "pkg", "x")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	noCwd := func() (string, error) { t.Fatal("cwd must not be consulted"); return "", nil }

	// CLAUDE_PROJECT_DIR wins and is resolved to the git top-level.
	env := func(k string) string {
		if k == "CLAUDE_PROJECT_DIR" {
			return sub
		}
		return ""
	}
	if got := defaultProjectRoot("", env, noCwd); got != repo {
		t.Errorf("env default = %q, want %q", got, repo)
	}
	// No env: cwd, also resolved to the top-level.
	cwd := func() (string, error) { return sub, nil }
	if got := defaultProjectRoot("", func(string) string { return "" }, cwd); got != repo {
		t.Errorf("cwd default = %q, want %q", got, repo)
	}
	// Outside any repo: the directory itself.
	plain := t.TempDir()
	plainReal, _ := filepath.EvalSymlinks(plain)
	if got := defaultProjectRoot("", func(string) string { return "" },
		func() (string, error) { return plain, nil }); got != plain && got != plainReal {
		t.Errorf("non-repo default = %q, want %q", got, plain)
	}
	// An explicit flag is used as given (made absolute), not re-resolved.
	if got := defaultProjectRoot(sub, env, noCwd); got != sub {
		t.Errorf("explicit = %q, want %q", got, sub)
	}
}

func TestIsDaemonArgs(t *testing.T) {
	cases := []struct {
		args []string
		want bool
	}{
		{[]string{"/usr/bin/rostam-server", "memory-daemon", "-data", "/x"}, true},
		{[]string{"rostam-server", "memory-daemon"}, true},
		{[]string{"/usr/bin/rostam-server", "mcp"}, false},
		{[]string{"/usr/bin/vim", "memory-daemon"}, false},
		{[]string{"rostam-server"}, false},
		{nil, false},
	}
	for _, tc := range cases {
		if got := isDaemonArgs(tc.args, "rostam-server"); got != tc.want {
			t.Errorf("isDaemonArgs(%v) = %v, want %v", tc.args, got, tc.want)
		}
	}
}

func withAuth(c rostam.ServerConfig, a authz.Authenticator) rostam.ServerConfig {
	c.Authenticator = a
	return c
}
