// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"context"
	"errors"
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
	if processIsDaemon(os.Getpid(), "/x") {
		t.Fatal("the test process is not a memory daemon")
	}
	if processIsDaemon(0, "/x") || processIsDaemon(-1, "/x") || processIsDaemon(1<<30, "/x") {
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
	// The victim is our unreaped child: once signalled it would be a zombie,
	// which kill(pid, 0) still reports alive. A non-blocking reap tells.
	var ws syscall.WaitStatus
	if n, _ := syscall.Wait4(victim.Process.Pid, &ws, syscall.WNOHANG, nil); n == victim.Process.Pid {
		t.Fatalf("victim was signalled and exited: %v", ws)
	}
}

func TestStopWithNoDaemonIsClean(t *testing.T) {
	data := t.TempDir()
	if err := stopDaemon(data, time.Second); err == nil || !strings.Contains(err.Error(), "not running") {
		t.Fatalf("got %v, want not-running error", err)
	}
}

// Helper-process entry points (re-exec of the test binary).
func TestHelperSupervisor(t *testing.T) {
	data := os.Getenv("ROSTAM_TEST_SUPERVISOR_DATA")
	if data == "" {
		t.Skip("helper process only")
	}
	exe, _ := os.Executable()
	err := superviseDaemonWith(data, func(token, cfgPath string) *exec.Cmd {
		// A fake server whose argv looks like the real one.
		cmd := exec.Command(exe, "-test.run=^TestHelperChild$", "--", "-data", data, "-config", cfgPath)
		cmd.Env = append(os.Environ(), "ROSTAM_TEST_CHILD=1")
		return cmd
	}, func(context.Context, string) error {
		// The fake server "answers" once the test creates this file.
		_, err := os.Stat(filepath.Join(data, helperAnswersName))
		return err
	})
	t.Fatalf("supervisor returned: %v", err)
}

// helperAnswersName is the file whose presence makes the helper supervisor's
// readiness probe succeed, standing in for its server answering.
const helperAnswersName = "helper-answers"

func TestHelperChild(t *testing.T) {
	if os.Getenv("ROSTAM_TEST_CHILD") == "" {
		t.Skip("helper process only")
	}
	time.Sleep(time.Minute)
}

func pidAlive(pid int) bool { return syscall.Kill(pid, 0) == nil }

// startHelperSupervisor runs TestHelperSupervisor (a real supervisor over a
// fake server) on data, waits for its pid file, and returns the supervisor and
// the server's pid. Both are killed when the test ends.
func startHelperSupervisor(t *testing.T, data string) (*exec.Cmd, int) {
	t.Helper()
	exe, _ := os.Executable()
	sup := exec.Command(exe, "-test.run=^TestHelperSupervisor$")
	sup.Env = append(os.Environ(), "ROSTAM_TEST_SUPERVISOR_DATA="+data, "HOME="+t.TempDir())
	if err := sup.Start(); err != nil {
		t.Fatal(err)
	}
	var childPid int
	t.Cleanup(func() {
		_ = sup.Process.Kill()
		_ = sup.Wait()
		if childPid > 1 {
			_ = syscall.Kill(childPid, syscall.SIGKILL)
		}
	})
	deadline := time.Now().Add(10 * time.Second)
	for {
		raw, _ := os.ReadFile(filepath.Join(data, daemonPidName))
		if f := strings.Fields(string(raw)); len(f) == 2 {
			childPid, _ = strconv.Atoi(f[1])
			return sup, childPid
		}
		if time.Now().After(deadline) {
			t.Fatal("supervisor never wrote its pid file")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// The data-dir lock must stay held by the server process when its supervisor
// dies, and -stop must still stop that orphan.
func TestDaemonLockSurvivesSupervisorKill(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	sup, childPid := startHelperSupervisor(t, data)
	if _, err := lockDataDir(data); !errors.Is(err, errDataDirBusy) {
		t.Fatalf("lock while running: %v, want busy", err)
	}

	// Kill the supervisor only.
	if err := sup.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = sup.Wait()
	if !pidAlive(childPid) {
		t.Fatal("child died with its supervisor; test premise broken")
	}
	if _, err := lockDataDir(data); !errors.Is(err, errDataDirBusy) {
		t.Fatalf("lock after supervisor death: %v, want still busy (held by the child)", err)
	}
	if err := superviseDaemonWith(data, nil, nil); !errors.Is(err, errDataDirBusy) {
		t.Fatalf("second supervisor: %v, want refusal on the busy lock", err)
	}

	if err := stopDaemon(data, 10*time.Second); err != nil {
		t.Fatalf("stopDaemon against the orphan: %v", err)
	}
	if unlock, err := lockDataDir(data); err != nil {
		t.Fatalf("lock after stop: %v", err)
	} else {
		_ = unlock()
	}
}

func TestIsDaemonServerArgs(t *testing.T) {
	d := "/h/.rostam/memory"
	good := []string{"/bin/rostam-server", "-tcp", "127.0.0.1:7480", "-http", "", "-data", d, "-shards", "8", "-config", d + "/daemon-config.json"}
	if !isDaemonServerArgs(good, "rostam-server", d) {
		t.Error("real server argv rejected")
	}
	if isDaemonServerArgs(good, "other", d) || isDaemonServerArgs(good, "rostam-server", "/else") {
		t.Error("mismatched exe/data accepted")
	}
	// Long enough to pass the length guard, so the missing -config decides.
	if isDaemonServerArgs([]string{"/bin/rostam-server", "-tcp", "127.0.0.1:7480", "-data", d, "-shards", "8"}, "rostam-server", d) {
		t.Error("argv without -config accepted")
	}
}

// -stop given a relative -data stops the daemon started with the absolute one.
func TestStopDaemonRelativeData(t *testing.T) {
	parent := t.TempDir()
	data := filepath.Join(parent, "memory")
	startHelperSupervisor(t, data)
	t.Chdir(parent)
	if err := stopDaemon("memory", 10*time.Second); err != nil {
		t.Fatalf("stopDaemon with a relative -data: %v", err)
	}
}
