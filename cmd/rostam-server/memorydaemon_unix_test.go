// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestProcessIsDaemonRejectsUnrelatedPids(t *testing.T) {
	if processIsDaemon(os.Getpid()) {
		t.Fatal("the test process is not a memory daemon")
	}
	if processIsDaemon(0) || processIsDaemon(-1) || processIsDaemon(1<<30) {
		t.Fatal("invalid/nonexistent pids must not verify")
	}
}

// -stop must never signal a process it cannot verify as our daemon, even if the
// pid file names a live process and the data dir lock is held.
func TestStopRefusesUnverifiedPid(t *testing.T) {
	data := t.TempDir()
	victim := exec.Command("sleep", "30")
	if err := victim.Start(); err != nil {
		t.Skip("no sleep binary")
	}
	t.Cleanup(func() { _ = victim.Process.Kill(); _ = victim.Wait() })
	if err := os.WriteFile(filepath.Join(data, daemonPidName), []byte(strconv.Itoa(victim.Process.Pid)), 0o600); err != nil {
		t.Fatal(err)
	}
	unlock, err := claimDataDir(data) // pretend a daemon holds the lock
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()

	err = stopDaemon(data, time.Second)
	if err == nil || !strings.Contains(err.Error(), "not") {
		t.Fatalf("stopDaemon = %v, want a refusal", err)
	}
	time.Sleep(50 * time.Millisecond)
	if err := syscall.Kill(victim.Process.Pid, 0); err != nil {
		t.Fatalf("victim is gone (signalled?): %v", err)
	}
}

func TestStopWithNoDaemonIsClean(t *testing.T) {
	data := t.TempDir()
	if err := stopDaemon(data, time.Second); err == nil || !strings.Contains(err.Error(), "not running") {
		t.Fatalf("got %v, want not-running error", err)
	}
}
