// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"context"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/mcp"
)

// fakeOldDaemonEnv makes this test binary act as a previous release's memory
// daemon supervisor: it holds the data-dir lock, writes the pid file and exits
// on SIGTERM. It is handled in init, before the testing package parses argv,
// so the process can carry the real daemon's command line
// (`<exe> memory-daemon -data <dir> ...`), which is what -stop verifies.
const fakeOldDaemonEnv = "ROSTAM_TEST_FAKE_OLD_DAEMON"

func init() {
	if os.Getenv(fakeOldDaemonEnv) == "" {
		return
	}
	data := ""
	for i, a := range os.Args {
		if a == "-data" && i+1 < len(os.Args) {
			data = os.Args[i+1]
		}
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM)
	unlock, err := lockDataDir(data)
	if err != nil {
		os.Exit(3)
	}
	if err := os.WriteFile(filepath.Join(data, daemonPidName), []byte(strconv.Itoa(os.Getpid())+"\n"), 0o600); err != nil {
		os.Exit(4)
	}
	select {
	case <-sig:
	case <-time.After(time.Minute):
	}
	_ = os.Remove(filepath.Join(data, daemonPidName))
	_ = unlock()
	os.Exit(0)
}

// startFakeOldDaemon runs a copy of this test binary installed as binDir/name
// with a previous release's daemon argv, and waits for its pid file.
func startFakeOldDaemon(t *testing.T, binDir, name, data string) (*exec.Cmd, string) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(binDir, name)
	copyFile(t, self, bin, 0o755)
	if err := os.MkdirAll(data, 0o700); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin, "memory-daemon", "-data", data, "-addr", noDaemonAddr)
	cmd.Env = append(os.Environ(), fakeOldDaemonEnv+"=1", "HOME="+t.TempDir())
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	deadline := time.Now().Add(10 * time.Second)
	for !daemonPidFileExists(data) {
		if time.Now().After(deadline) {
			t.Fatal("fake old daemon never wrote its pid file")
		}
		time.Sleep(20 * time.Millisecond)
	}
	return cmd, bin
}

func copyFile(t *testing.T, src, dst string, mode os.FileMode) {
	t.Helper()
	in, err := os.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
}

// An upgrade installs the new binary at the old one's path while the old
// daemon keeps running from the replaced file. -stop identifies the daemon by
// its argv (unchanged by the replacement), so it still stops it.
func TestStopDaemonAfterBinaryReplaced(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the daemon is Linux-only")
	}
	self, _ := os.Executable()
	binDir := t.TempDir()
	data := filepath.Join(t.TempDir(), "memory")
	cmd, bin := startFakeOldDaemon(t, binDir, filepath.Base(self), data)

	// Install "the new release" over it the way an installer does: write a new
	// file beside it and rename it into place.
	next := bin + ".new"
	if err := os.WriteFile(next, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(next, bin); err != nil {
		t.Fatal(err)
	}
	if exe, err := os.Readlink("/proc/" + strconv.Itoa(cmd.Process.Pid) + "/exe"); err != nil || !strings.HasSuffix(exe, "(deleted)") {
		t.Fatalf("premise: the old daemon should run a replaced file, /proc exe = %q, %v", exe, err)
	}

	if err := stopDaemon(data, 10*time.Second); err != nil {
		t.Fatalf("stopDaemon against a daemon whose binary was replaced: %v", err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("old daemon exited with %v, want a clean exit on SIGTERM", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("old daemon still running after stopDaemon returned")
	}
}

// An old daemon run from a binary with another name is not verifiable as ours:
// -stop refuses and signals nothing, and retire warns and leaves it running.
func TestRetireLeavesUnverifiableOldDaemon(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the daemon is Linux-only")
	}
	data := filepath.Join(t.TempDir(), "memory")
	cmd, _ := startFakeOldDaemon(t, t.TempDir(), "rostam-server-previous", data)

	var w warnings
	answering := func(context.Context, string, string) error { return nil }
	err := retireNonDurableDaemon(t.Context(), data, noDaemonAddr, answering,
		func() error { return stopDaemon(data, time.Second) }, w.warn)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(w.all(), "memory-daemon -stop") {
		t.Fatalf("warning %q must name `rostam-server memory-daemon -stop`", w.all())
	}
	time.Sleep(50 * time.Millisecond)
	var ws syscall.WaitStatus
	if n, _ := syscall.Wait4(cmd.Process.Pid, &ws, syscall.WNOHANG, nil); n == cmd.Process.Pid {
		t.Fatalf("an unverifiable daemon was signalled and exited: %v", ws)
	}
}

// The server child holds the durable marker for exactly as long as it lives:
// from before it starts, through a SIGKILL of its supervisor alone, until it
// exits.
func TestSupervisorHoldsDurableMarker(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	sup, childPid := startHelperSupervisor(t, data)
	if ok, err := daemonIsDurable(data); !ok || err != nil {
		t.Fatalf("while the child runs: daemonIsDurable = %v, %v; want true", ok, err)
	}

	if err := sup.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = sup.Wait()
	if !pidAlive(childPid) {
		t.Fatal("child died with its supervisor; test premise broken")
	}
	if ok, err := daemonIsDurable(data); !ok || err != nil {
		t.Fatalf("after the supervisor alone was killed: daemonIsDurable = %v, %v; want true (held by the child)", ok, err)
	}

	if err := syscall.Kill(childPid, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		ok, err := daemonIsDurable(data)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("marker still held after the child exited")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestDaemonChildArgs(t *testing.T) {
	data := "/h/.rostam/memory"
	cfg := filepath.Join(data, daemonConfigName)
	args := daemonServerArgs(data, "127.0.0.1:7480", cfg)
	found := false
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "-durable-collections" && args[i+1] == mcp.MemoryCollection {
			found = true
		}
	}
	if !found || mcp.MemoryCollection != "mcp_memory" {
		t.Fatalf("daemon server args %q lack -durable-collections mcp_memory", args)
	}
	// -stop must still recognise the server it launched (an orphan fallback).
	if !isDaemonServerArgs(append([]string{"/bin/rostam-server"}, args...), "rostam-server", data) {
		t.Fatalf("isDaemonServerArgs rejects the daemon's own server argv %q", args)
	}
}

// A session's daemonIsDurable check holds the marker for an instant; a
// supervisor starting at that moment waits it out instead of failing.
func TestLockDurableMarkerWaitsOutACheck(t *testing.T) {
	data := t.TempDir()
	unlock := holdDurableMarker(t, data)
	go func() {
		time.Sleep(100 * time.Millisecond)
		_ = unlock()
	}()
	f, err := lockDurableMarker(data)
	if err != nil {
		t.Fatalf("lockDurableMarker with a passing check: %v", err)
	}
	if ok, err := daemonIsDurable(data); !ok || err != nil {
		t.Fatalf("after lockDurableMarker: daemonIsDurable = %v, %v; want true", ok, err)
	}
	_ = f.Close()
}
