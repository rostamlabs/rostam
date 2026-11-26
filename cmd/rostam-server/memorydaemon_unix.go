// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/rostamlabs/rostam/mcp"
)

// daemonSupported: see errDaemonUnsupported (persistence is Linux-only among
// the unixes).
const daemonSupported = runtime.GOOS == "linux"

// runMemoryDaemonCmd implements `rostam-server memory-daemon`. It is a thin
// supervisor, not a second server: it re-executes this binary as the regular
// single-node server on a loopback address with the token as its API key, and
// forwards termination signals so the server flushes (instant-restart
// sidecars) before the process exits. Diagnostics go to stderr, which a
// spawned daemon has pointed at <data>/daemon.log.
func runMemoryDaemonCmd(args []string) {
	if !daemonSupported {
		fatal("memory-daemon: " + errDaemonUnsupported.Error())
	}
	fs := flag.NewFlagSet("memory-daemon", flag.ExitOnError)
	defData, _ := defaultMemoryDir()
	data := fs.String("data", defData, "daemon data directory (default ~/.rostam/memory)")
	addr := fs.String("addr", memoryDefaultAddr, "loopback listen address (host must be a loopback IP)")
	stop := fs.Bool("stop", false, "stop the running daemon for -data, then exit")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if *data == "" {
		fatal("memory-daemon: no -data directory and no home directory to default to")
	}
	// Absolute from here on: the server is started with it, and -stop matches
	// it against running daemons' argv, so "-data ./mem" and "-data /abs/mem"
	// must name the same daemon.
	if abs, err := filepath.Abs(*data); err == nil {
		*data = abs
	}
	if err := validateLoopbackAddr(*addr); err != nil {
		fatal("memory-daemon: " + err.Error())
	}
	if *stop {
		if err := stopDaemon(*data, daemonStopWait); err != nil {
			fatal("memory-daemon: " + err.Error())
		}
		fmt.Fprintln(os.Stderr, "memory daemon stopped")
		return
	}
	if err := superviseDaemon(*data, *addr); err != nil {
		fatal("memory-daemon: " + err.Error())
	}
}

func superviseDaemon(data, addr string) error {
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locating own executable: %w", err)
	}
	return superviseDaemonWith(data, func(token, cfgPath string) *exec.Cmd {
		cmd := exec.Command(exe, daemonServerArgs(data, addr, cfgPath)...) //nolint:gosec // G204: exe is os.Executable(), argv only, no shell
		// The token travels in the environment, never argv. Inherited ROSTAM_*
		// variables are dropped: every server flag reads one, and the user's
		// shell must not reconfigure (or re-expose) the daemon.
		cmd.Env = daemonEnv(os.Environ(), token)
		return cmd
	})
}

// daemonServerArgs is the argv (after the executable) of the server the
// supervisor runs for data. isDaemonServerArgs recognises it, so -stop can
// still reach that server when its supervisor is gone.
func daemonServerArgs(data, addr, cfgPath string) []string {
	return []string{
		"-tcp", addr,
		"-http", "", // the flag default is :8080 on every interface
		"-data", data,
		"-shards", strconv.Itoa(daemonShards),
		"-config", cfgPath,
		"-durable-collections", mcp.MemoryCollection,
	}
}

// markerLockWait bounds how long a starting supervisor retries the durable
// marker while a session's daemonIsDurable check holds it for an instant.
const markerLockWait = 2 * time.Second

// lockDurableMarker creates <data>/daemon-durable and locks it. It is called
// with the data-dir lock held, so no other daemon's server can be holding the
// marker: a busy lock is a session checking it, and is retried.
func lockDurableMarker(data string) (*os.File, error) {
	dir := filepath.Join(data, daemonDurableDir)
	if err := os.MkdirAll(dir, 0o700); err != nil { //nolint:gosec // G703: fixed name inside the data dir
		return nil, fmt.Errorf("creating %s: %w", dir, err)
	}
	deadline := time.Now().Add(markerLockWait)
	for {
		f, err := lockDataDirFile(dir)
		if !errors.Is(err, errDataDirBusy) {
			return f, err
		}
		if !time.Now().Before(deadline) {
			return nil, fmt.Errorf("the durable-memory marker %s stayed locked for %s: %w", dir, markerLockWait, err)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// superviseDaemonWith runs the child built by mkChild under the data-dir lock.
// The locked file is passed to the child as an extra fd: flock belongs to the
// open file description, so the lock stays held for as long as the child (the
// process that actually has the store open) lives, even if this supervisor is
// killed. The durable marker (daemonDurableDir) is locked and handed over the
// same way, before the child starts, so a server that answers always holds it.
func superviseDaemonWith(data string, mkChild func(token, cfgPath string) *exec.Cmd) error {
	if err := os.MkdirAll(data, 0o700); err != nil { //nolint:gosec // G703: data is the user's own -data flag or the home default
		return fmt.Errorf("creating -data directory %s: %w", data, err)
	}
	// The regular server takes no lock on -data, so the supervisor takes the
	// single-writer lock for it: one daemon per data dir, and an embedded
	// `mcp -data` on the same directory is refused rather than corrupting it.
	lockFile, err := lockDataDirFile(data)
	switch {
	case errors.Is(err, errDataDirBusy):
		return fmt.Errorf("another process (a running memory daemon or an embedded mcp) already holds %s: %w", data, err)
	case err != nil:
		return err
	}
	defer func() { _ = lockFile.Close() }()

	marker, err := lockDurableMarker(data)
	if err != nil {
		return err
	}
	defer func() { _ = marker.Close() }()

	token, err := ensureToken(data)
	if err != nil {
		return err
	}
	// A personal memory store must not size itself like a server (the default
	// budget is a fraction of host RAM, mmap-backed). -config is the server's
	// only cache-budget knob. Rewritten on every start.
	cfgPath := filepath.Join(data, daemonConfigName)
	cfg := `{"cache":{"max_memory":"` + daemonCacheBudget + `"}}` + "\n"
	if err := os.WriteFile(cfgPath, []byte(cfg), 0o600); err != nil { //nolint:gosec // G703: fixed name inside the data dir
		return fmt.Errorf("writing %s: %w", cfgPath, err)
	}

	cmd := mkChild(token, cfgPath)
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	cmd.ExtraFiles = append(cmd.ExtraFiles, lockFile, marker) // child inherits both locks

	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting server: %w", err)
	}
	// Line 1: supervisor pid; line 2: server pid (what -stop falls back to if
	// the supervisor was killed and the server orphaned).
	pidPath := filepath.Join(data, daemonPidName)
	pids := strconv.Itoa(os.Getpid()) + "\n" + strconv.Itoa(cmd.Process.Pid) + "\n"
	if err := os.WriteFile(pidPath, []byte(pids), 0o600); err != nil { //nolint:gosec // G703: fixed name inside the data dir
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return fmt.Errorf("writing %s: %w", pidPath, err)
	}
	defer func() { _ = os.Remove(pidPath) }() //nolint:gosec // G703: fixed name inside the data dir

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return fmt.Errorf("server exited: %v", err)
	case <-sigCh:
		_ = cmd.Process.Signal(syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(daemonStopWait):
			_ = cmd.Process.Kill()
			<-done
		}
		return nil
	}
}

// daemonEnv is env minus every ROSTAM_* variable, plus the API key.
func daemonEnv(env []string, token string) []string {
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		if !strings.HasPrefix(kv, "ROSTAM_") {
			out = append(out, kv)
		}
	}
	return append(out, "ROSTAM_API_KEY="+token)
}

// newDaemonSpawner returns the spawn func for ensureDaemon: it starts a
// detached `memory-daemon` (own session, stdin closed, output to
// <data>/daemon.log mode 0600) that outlives this process.
func newDaemonSpawner(data, addr string) func() error {
	return func() error {
		if !daemonSupported {
			return errDaemonUnsupported
		}
		exe, err := os.Executable()
		if err != nil {
			return err
		}
		if err := os.MkdirAll(data, 0o700); err != nil {
			return err
		}
		logf, err := os.OpenFile(filepath.Join(data, daemonLogName), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
		if err != nil {
			return err
		}
		defer func() { _ = logf.Close() }() // the child holds its own copy of the fd
		_ = logf.Chmod(0o600)
		cmd := exec.Command(exe, "memory-daemon", "-data", data, "-addr", addr) //nolint:gosec // G204: exe is os.Executable(), argv only, no shell
		cmd.Stdin, cmd.Stdout, cmd.Stderr = nil, logf, logf                     // nil stdin = /dev/null
		cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
		if err := cmd.Start(); err != nil {
			return err
		}
		go func() { _ = cmd.Wait() }() // reap it if it dies while we are alive
		return nil
	}
}

// stopDaemon SIGTERMs the daemon for data and waits for it to release the
// data-dir lock. It signals only a pid it has verified: the pid file must name
// a live process whose command line is `<this binary> memory-daemon ...`, and
// the data-dir lock must be held (a stale pid file whose number was recycled
// is therefore never signalled).
func stopDaemon(data string, wait time.Duration) error {
	if abs, err := filepath.Abs(data); err == nil {
		data = abs // argv of a daemon we started holds the absolute path
	}
	pidPath := filepath.Join(data, daemonPidName)
	raw, err := os.ReadFile(pidPath)
	if errors.Is(err, os.ErrNotExist) {
		return errors.New("memory daemon is not running (no pid file)")
	} else if err != nil {
		return err
	}
	if unlock, lerr := lockDataDir(data); lerr == nil {
		_ = unlock()
		_ = os.Remove(pidPath)
		return errors.New("memory daemon is not running (removed a stale pid file)")
	} else if !errors.Is(lerr, errDataDirBusy) {
		return lerr
	}
	// Only a pid verified as ours is signalled: the supervisor (it forwards
	// SIGTERM to the server), else the server itself when the supervisor was
	// killed and left it orphaned still holding the lock.
	var target int
	for i, field := range strings.Fields(string(raw)) {
		pid, perr := strconv.Atoi(field)
		if perr != nil || pid <= 1 {
			continue
		}
		if (i == 0 && processIsDaemon(pid, data)) || (i == 1 && processIsDaemonServer(pid, data)) {
			target = pid
			break
		}
	}
	if target == 0 {
		return fmt.Errorf("refusing to signal: the pids in %s are not a rostam-server memory daemon", pidPath)
	}
	pid := target
	if err := syscall.Kill(pid, syscall.SIGTERM); err != nil {
		return fmt.Errorf("signalling pid %d: %w", pid, err)
	}
	deadline := time.Now().Add(wait)
	for time.Now().Before(deadline) {
		if unlock, lerr := lockDataDir(data); lerr == nil {
			_ = unlock()
			return nil
		}
		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("pid %d did not exit within %s", pid, wait)
}

// isDaemonServerArgs reports whether args is the server the supervisor
// launched for data: `<exe> ... -data <data> ... -config <data>/daemon-config.json`.
func isDaemonServerArgs(args []string, exeBase, data string) bool {
	if len(args) < 5 || filepath.Base(args[0]) != exeBase {
		return false
	}
	hasPair := func(flag, val string) bool {
		for i := 1; i+1 < len(args); i++ {
			if args[i] == flag && args[i+1] == val {
				return true
			}
		}
		return false
	}
	return hasPair("-data", data) && hasPair("-config", filepath.Join(data, daemonConfigName))
}

// processArgs returns pid's argv: /proc where it exists (Linux), else the
// platform's own source (platformProcArgs).
func processArgs(pid int) []string {
	if b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline"); err == nil {
		return strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
	}
	return platformProcArgs(pid)
}

// processIsDaemonServer reports whether pid is the server a supervisor started
// for data.
func processIsDaemonServer(pid int, data string) bool {
	exe, err := os.Executable()
	if pid <= 1 || err != nil {
		return false
	}
	return isDaemonServerArgs(processArgs(pid), filepath.Base(exe), data)
}

// processIsDaemon reports whether pid is a live `<this binary> memory-daemon`
// process for data.
func processIsDaemon(pid int, data string) bool {
	if pid <= 1 {
		return false
	}
	exe, err := os.Executable()
	if err != nil {
		return false
	}
	defData, _ := defaultMemoryDir()
	return isDaemonArgs(processArgs(pid), filepath.Base(exe), data, defData)
}
