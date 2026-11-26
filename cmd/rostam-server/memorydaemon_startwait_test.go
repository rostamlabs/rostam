// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A first start that upgrades a large store can take longer than
// daemonStartWait. While a daemon process demonstrably holds the data dir and
// is starting, -autostart keeps waiting, up to daemonStartMaxWait; a daemon
// that is not starting still fails at daemonStartWait.

func setDaemonStarting(t *testing.T, fn func(string) bool, maxWait time.Duration) {
	t.Helper()
	prevFn, prevMax := daemonStartingFn, daemonStartMaxWait
	daemonStartingFn, daemonStartMaxWait = fn, maxWait
	t.Cleanup(func() { daemonStartingFn, daemonStartMaxWait = prevFn, prevMax })
}

func TestEnsureDaemonKeepsWaitingForAStartingDaemon(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	tok, err := ensureToken(data)
	if err != nil {
		t.Fatal(err)
	}
	fd := newFakeDaemon(t, tok)
	setDaemonStarting(t, func(string) bool { return true }, 10*time.Second)
	spawn := func() error {
		time.AfterFunc(800*time.Millisecond, fd.start) // ready well after the short wait
		return nil
	}
	if err := ensureDaemonWith(context.Background(), data, fd.addr(), spawn, fakeProbe, 200*time.Millisecond); err != nil {
		t.Fatalf("a daemon that was still starting: %v", err)
	}
}

func TestEnsureDaemonStartingWaitIsBounded(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	fd := newFakeDaemon(t, "never-started")
	setDaemonStarting(t, func(string) bool { return true }, 600*time.Millisecond)
	start := time.Now()
	err := ensureDaemonWith(context.Background(), data, fd.addr(), func() error { return nil }, fakeProbe, 100*time.Millisecond)
	if err == nil {
		t.Fatal("a daemon that never became ready was accepted")
	}
	if el := time.Since(start); el < 500*time.Millisecond || el > 5*time.Second {
		t.Fatalf("gave up after %v, want about the 600ms overall bound", el)
	}
	if !strings.Contains(err.Error(), "daemon.log") {
		t.Fatalf("error %q does not name the log file", err)
	}
}

func TestEnsureDaemonNotStartingFailsAtTheShortWait(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	fd := newFakeDaemon(t, "never-started")
	setDaemonStarting(t, func(string) bool { return false }, time.Minute)
	start := time.Now()
	err := ensureDaemonWith(context.Background(), data, fd.addr(), func() error { return nil }, fakeProbe, 300*time.Millisecond)
	if err == nil {
		t.Fatal("expected failure")
	}
	if el := time.Since(start); el > 5*time.Second {
		t.Fatalf("waited %v for a daemon that is not starting", el)
	}
}

// The real check never counts an embedded holder (the lock, no pid file) or a
// pid file naming a process that is not a memory daemon as a starting daemon.
func TestDaemonStartingRefusesNonDaemonHolders(t *testing.T) {
	if !daemonSupported {
		t.Skip("no memory daemon on this platform")
	}
	data := filepath.Join(t.TempDir(), "memory")
	if err := os.MkdirAll(data, 0o700); err != nil {
		t.Fatal(err)
	}
	if daemonStarting(data) {
		t.Fatal("an empty data dir counts as a starting daemon")
	}
	unlock, err := lockDataDir(data) // an embedded mcp
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	if daemonStarting(data) {
		t.Fatal("an embedded holder counts as a starting daemon")
	}
	self := strconv.Itoa(os.Getpid())
	if err := os.WriteFile(filepath.Join(data, daemonPidName), []byte(self+"\n"+self+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if daemonStarting(data) {
		t.Fatal("a pid file naming a process that is not a memory daemon counts as a starting daemon")
	}
}
