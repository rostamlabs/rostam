// SPDX-License-Identifier: Apache-2.0
//go:build linux

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// These tests hold agent memory to its crash contract through the built binary
// only: what remember and forget acknowledged is there after the process that
// holds the store is killed with SIGKILL. Every child runs with a temporary
// HOME and a temporary data directory, and only processes started here are
// signalled.

const (
	keyedOld  = "deploy status: waiting for review"
	keyedNew  = "deploy status: merged and shipped"
	forgotten = "the temporary password is on the whiteboard"
)

// killSandbox is the environment of every child: a temporary HOME, none of the
// ambient ROSTAM_* or CLAUDE_* configuration, and BM25-only embedding.
type killSandbox struct {
	bin  string
	home string
}

func newKillSandbox(t *testing.T) killSandbox {
	t.Helper()
	if testing.Short() {
		t.Skip("compiles the rostam-server binary and kills it as a subprocess")
	}
	return killSandbox{bin: buildServer(t), home: t.TempDir()}
}

func (s killSandbox) environ(extra ...string) []string {
	return (&e2eEnv{home: s.home}).environ(extra...)
}

// session starts `rostam-server mcp <args>` and initializes it.
func (s killSandbox) session(t *testing.T, env []string, args ...string) *mcpProc {
	t.Helper()
	cmd := exec.Command(s.bin, append([]string{"mcp"}, args...)...)
	cmd.Env, cmd.Dir = env, s.home
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("stdin pipe: %v", err)
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	logs := &syncBuffer{}
	cmd.Stderr = logs
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting mcp: %v", err)
	}
	p := &mcpProc{t: t, cmd: cmd, in: in, out: bufio.NewScanner(out), logs: logs}
	p.out.Buffer(make([]byte, 64<<10), 16<<20)
	t.Cleanup(func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})
	p.initialize()
	return p
}

// killSession ends an mcp session with SIGKILL and checks that is what ended it.
func killSession(t *testing.T, p *mcpProc) {
	t.Helper()
	if err := p.cmd.Process.Kill(); err != nil {
		t.Fatalf("SIGKILL: %v\nserver log:\n%s", err, p.logs)
	}
	err := p.cmd.Wait()
	var exit *exec.ExitError
	if !errors.As(err, &exit) {
		t.Fatalf("want the session ended by SIGKILL, got %v\nserver log:\n%s", err, p.logs)
	}
	if ws, ok := exit.Sys().(syscall.WaitStatus); !ok || !ws.Signaled() || ws.Signal() != syscall.SIGKILL {
		t.Fatalf("want the session ended by SIGKILL, got %v\nserver log:\n%s", err, p.logs)
	}
}

// closeSession ends an mcp session the clean way: EOF on stdin, exit 0.
func closeSession(t *testing.T, p *mcpProc) {
	t.Helper()
	if err := p.in.Close(); err != nil {
		t.Fatalf("closing stdin: %v", err)
	}
	done := make(chan error, 1)
	go func() { done <- p.cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("want a clean exit, got %v\nserver log:\n%s", err, p.logs)
		}
	case <-time.After(60 * time.Second):
		_ = p.cmd.Process.Kill()
		<-done
		t.Fatalf("the session did not exit within 60s of EOF\nserver log:\n%s", p.logs)
	}
}

func rememberID(t *testing.T, p *mcpProc, args map[string]any) uint64 {
	t.Helper()
	out := p.callTool("remember", args)
	var res struct {
		ID uint64 `json:"id"`
	}
	if err := json.Unmarshal([]byte(out), &res); err != nil || res.ID == 0 {
		t.Fatalf("remember result %q: %v", out, err)
	}
	return res.ID
}

// changeMemories makes one of each acknowledged change: a plain remember, a
// keyed memory written and then overwritten, and a memory remembered and then
// forgotten. What must remain of it is plain and keyedNew.
func changeMemories(t *testing.T, p *mcpProc, plain string) {
	t.Helper()
	rememberID(t, p, map[string]any{"content": plain})
	first := rememberID(t, p, map[string]any{"content": keyedOld, "key": "deploy"})
	if second := rememberID(t, p, map[string]any{"content": keyedNew, "key": "deploy"}); second != first {
		t.Fatalf("the keyed overwrite made a new memory: ids %d and %d", first, second)
	}
	gone := rememberID(t, p, map[string]any{"content": forgotten})
	out := p.callTool("forget", map[string]any{"ids": []uint64{gone}})
	var res struct {
		Deleted []uint64 `json:"deleted"`
		Errors  []string `json:"errors"`
	}
	if err := json.Unmarshal([]byte(out), &res); err != nil || !slices.Equal(res.Deleted, []uint64{gone}) || len(res.Errors) != 0 {
		t.Fatalf("forget result %q: %v", out, err)
	}
}

// wantMemories checks that list_memories holds exactly want, so a lost memory,
// a surviving old version and a returned forgotten one all fail, and that
// full-text recall finds the overwritten memory.
func wantMemories(t *testing.T, p *mcpProc, want ...string) {
	t.Helper()
	contents := func(tool, field string, args map[string]any) []string {
		out := p.callTool(tool, args)
		var res map[string]json.RawMessage
		var items []struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal([]byte(out), &res); err != nil {
			t.Fatalf("%s result %q: %v", tool, out, err)
		}
		if err := json.Unmarshal(res[field], &items); err != nil {
			t.Fatalf("%s result %q: %v", tool, out, err)
		}
		got := make([]string, len(items))
		for i, it := range items {
			got[i] = it.Content
		}
		return got
	}
	got := contents("list_memories", "memories", map[string]any{})
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("memories after the restart:\n got %q\nwant %q\nserver log:\n%s", got, want, p.logs)
	}
	if hits := contents("recall", "hits", map[string]any{"query": "merged shipped", "k": 1}); !slices.Equal(hits, []string{keyedNew}) {
		t.Fatalf("recall after the restart: got %q, want %q\nserver log:\n%s", hits, keyedNew, p.logs)
	}
}

// A store that has never been stopped cleanly keeps everything acknowledged.
func TestMCPMemorySurvivesKillBeforeFirstCleanStop(t *testing.T) {
	s := newKillSandbox(t)
	data := filepath.Join(t.TempDir(), "memory")

	p := s.session(t, s.environ(), "-data", data)
	changeMemories(t, p, "one")
	killSession(t, p)

	p = s.session(t, s.environ(), "-data", data)
	wantMemories(t, p, "one", keyedNew)
	closeSession(t, p)
}

// A store with a checkpoint from an earlier clean stop keeps everything
// acknowledged since.
func TestMCPMemorySurvivesKillAfterCleanStop(t *testing.T) {
	s := newKillSandbox(t)
	data := filepath.Join(t.TempDir(), "memory")

	p := s.session(t, s.environ(), "-data", data)
	rememberID(t, p, map[string]any{"content": "zero"})
	closeSession(t, p)

	p = s.session(t, s.environ(), "-data", data)
	changeMemories(t, p, "two")
	killSession(t, p)

	p = s.session(t, s.environ(), "-data", data)
	wantMemories(t, p, "zero", "two", keyedNew)
	closeSession(t, p)
}

// freeLoopbackAddr is a loopback address nothing listens on, never the
// daemon's well-known one.
func freeLoopbackAddr(t *testing.T) string {
	t.Helper()
	for {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		addr := l.Addr().String()
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}
		if addr != memoryDefaultAddr {
			return addr
		}
	}
}

func answers(addr string) bool {
	c, err := net.DialTimeout("tcp", addr, 200*time.Millisecond)
	if err != nil {
		return false
	}
	_ = c.Close()
	return true
}

// killDaemon is a `memory-daemon` supervisor and its server, in a process group
// of their own.
type killDaemon struct {
	t     *testing.T
	cmd   *exec.Cmd
	data  string
	addr  string
	token string
	done  chan struct{} // closed once the supervisor has been reaped
	err   error         // its Wait result, valid after done
	gone  bool          // the whole group is known to be gone
}

// daemon starts `memory-daemon` on data and addr and waits until it answers an
// authenticated request.
func (s killSandbox) daemon(t *testing.T, data, addr string) *killDaemon {
	t.Helper()
	logPath := filepath.Join(t.TempDir(), "daemon.log")
	logf, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = logf.Close() }() // the daemon holds its own descriptor
	cmd := exec.Command(s.bin, "memory-daemon", "-data", data, "-addr", addr)
	cmd.Env, cmd.Dir = s.environ(), s.home
	cmd.Stdout, cmd.Stderr = logf, logf
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting memory-daemon: %v", err)
	}
	d := &killDaemon{t: t, cmd: cmd, data: data, addr: addr, done: make(chan struct{})}
	go func() {
		d.err = cmd.Wait()
		close(d.done)
	}()
	t.Cleanup(func() {
		if !d.gone {
			d.signalGroup()
			<-d.done
		}
		if t.Failed() {
			if b, err := os.ReadFile(logPath); err == nil {
				t.Logf("daemon log:\n%s", b)
			}
		}
	})

	waitFor(t, "the daemon to answer", 30*time.Second, func() bool {
		select {
		case <-d.done:
			t.Fatalf("memory-daemon exited while starting: %v", d.err)
		default:
		}
		tok, err := readToken(filepath.Join(data, memoryTokenName))
		if err != nil {
			return false
		}
		d.token = tok
		return probeDaemon(context.Background(), addr, tok) == nil
	})

	// Killing the group must reach the server, the process holding the store.
	pids, err := os.ReadFile(filepath.Join(data, daemonPidName))
	if err != nil {
		t.Fatalf("pid file: %v", err)
	}
	f := strings.Fields(string(pids))
	if len(f) != 2 || f[0] != strconv.Itoa(cmd.Process.Pid) {
		t.Fatalf("pid file %q, want supervisor %d and its server", pids, cmd.Process.Pid)
	}
	server, err := strconv.Atoi(f[1])
	if err != nil {
		t.Fatalf("pid file %q: %v", pids, err)
	}
	if pgid, err := syscall.Getpgid(server); err != nil || pgid != cmd.Process.Pid {
		t.Fatalf("server %d is in group %d (%v), want the daemon's own group %d", server, pgid, err, cmd.Process.Pid)
	}
	return d
}

// signalGroup sends SIGKILL to the daemon's process group, and to nothing else.
func (d *killDaemon) signalGroup() {
	pgid := d.cmd.Process.Pid
	if pgid <= 1 || pgid == syscall.Getpgrp() {
		d.t.Fatalf("refusing to signal process group %d", pgid)
	}
	if err := syscall.Kill(-pgid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		d.t.Errorf("killing process group %d: %v", pgid, err)
	}
}

// kill kills supervisor and server together and waits until both are gone.
func (d *killDaemon) kill() {
	d.t.Helper()
	d.signalGroup()
	d.waitGone()
}

// stop stops the daemon cleanly, by the signal the supervisor forwards.
func (d *killDaemon) stop() {
	d.t.Helper()
	if err := d.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		d.t.Fatalf("SIGTERM: %v", err)
	}
	d.waitGone()
	if d.err != nil {
		d.t.Fatalf("want a clean daemon stop, got %v", d.err)
	}
}

// waitGone waits for the supervisor to be reaped, for the port to stop
// answering and for the server to release the data directory.
func (d *killDaemon) waitGone() {
	d.t.Helper()
	select {
	case <-d.done:
	case <-time.After(60 * time.Second):
		d.t.Fatal("the daemon did not exit within 60s")
	}
	waitFor(d.t, "the daemon's port to stop answering", 20*time.Second, func() bool { return !answers(d.addr) })
	waitFor(d.t, "the data directory to be released", 20*time.Second, func() bool {
		unlock, err := lockDataDir(d.data)
		if err != nil {
			return false
		}
		_ = unlock()
		return true
	})
	d.gone = true
}

func (s killSandbox) connect(t *testing.T, d *killDaemon) *mcpProc {
	t.Helper()
	return s.session(t, s.environ("ROSTAM_AUTH_TOKEN="+d.token), "-connect", d.addr)
}

// The shared daemon's memory survives its server being killed, before its first
// clean stop and after one.
func TestDaemonMemorySurvivesServerKill(t *testing.T) {
	s := newKillSandbox(t)

	t.Run("before_first_clean_stop", func(t *testing.T) {
		data, addr := filepath.Join(t.TempDir(), "memory"), freeLoopbackAddr(t)
		d := s.daemon(t, data, addr)
		p := s.connect(t, d)
		changeMemories(t, p, "one")
		d.kill()
		closeSession(t, p)

		d = s.daemon(t, data, addr)
		p = s.connect(t, d)
		wantMemories(t, p, "one", keyedNew)
		closeSession(t, p)
		d.stop()
	})

	t.Run("after_clean_stop", func(t *testing.T) {
		data, addr := filepath.Join(t.TempDir(), "memory"), freeLoopbackAddr(t)
		d := s.daemon(t, data, addr)
		p := s.connect(t, d)
		rememberID(t, p, map[string]any{"content": "zero"})
		closeSession(t, p)
		d.stop()

		d = s.daemon(t, data, addr)
		p = s.connect(t, d)
		changeMemories(t, p, "two")
		d.kill()
		closeSession(t, p)

		d = s.daemon(t, data, addr)
		p = s.connect(t, d)
		wantMemories(t, p, "zero", "two", keyedNew)
		closeSession(t, p)
		d.stop()
	})
}
