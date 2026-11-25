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
	"strconv"
	"strings"
	"syscall"
	"time"
)

const daemonSupported = true

const daemonStopWait = 30 * time.Second

// runMemoryDaemonCmd implements `rostam-server memory-daemon`. It is a thin
// supervisor, not a second server: it re-executes this binary as the regular
// single-node server on a loopback address with the token as its API key, and
// forwards termination signals so the server flushes (instant-restart
// sidecars) before the process exits. Diagnostics go to stderr, which a
// spawned daemon has pointed at <data>/daemon.log.
func runMemoryDaemonCmd(args []string) {
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
	// The regular server takes no lock on -data, so the supervisor holds the
	// single-writer lock for it: one daemon per data dir, and an embedded
	// `mcp -data` on the same directory is refused rather than corrupting it.
	unlock, err := claimDataDir(data)
	switch {
	case errors.Is(err, errDataDirBusy):
		return fmt.Errorf("another process (a running memory daemon or an embedded mcp) already holds %s: %w", data, err)
	case err != nil:
		return err
	}
	defer func() { _ = unlock() }()

	token, err := ensureToken(data)
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locating own executable: %w", err)
	}
	// A personal memory store must not size itself like a server (the default
	// budget is a fraction of host RAM, mmap-backed). -config is the server's
	// only cache-budget knob.
	cfgPath := filepath.Join(data, daemonConfigName)
	cfg := `{"cache":{"max_memory":"` + daemonCacheBudget + `"}}` + "\n"
	if err := os.WriteFile(cfgPath, []byte(cfg), 0o600); err != nil {
		return fmt.Errorf("writing %s: %w", cfgPath, err)
	}

	cmd := exec.Command(exe,
		"-tcp", addr,
		"-http", "", // the flag default is :8080 on every interface
		"-data", data,
		"-shards", strconv.Itoa(daemonShards),
		"-config", cfgPath)
	// The token travels in the environment, never argv. Inherited ROSTAM_*
	// variables are dropped: every server flag reads one, and the user's shell
	// must not reconfigure (or re-expose) the daemon.
	cmd.Env = daemonEnv(os.Environ(), token)
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr

	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting server: %w", err)
	}
	pidPath := filepath.Join(data, daemonPidName)
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o600); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return fmt.Errorf("writing %s: %w", pidPath, err)
	}
	defer os.Remove(pidPath)

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
		defer logf.Close()
		_ = logf.Chmod(0o600)
		cmd := exec.Command(exe, "memory-daemon", "-data", data, "-addr", addr)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = nil, logf, logf // nil stdin = /dev/null
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
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || pid <= 1 {
		return fmt.Errorf("refusing to signal: %s does not hold a valid pid", pidPath)
	}
	if !processIsDaemon(pid) {
		return fmt.Errorf("refusing to signal pid %d: it is not a rostam-server memory-daemon process", pid)
	}
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

// processIsDaemon reports whether pid is a live `<this binary> memory-daemon`
// process.
func processIsDaemon(pid int) bool {
	if pid <= 1 {
		return false
	}
	exe, err := os.Executable()
	if err != nil {
		return false
	}
	var args []string
	if b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline"); err == nil {
		args = strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
	} else if out, err := exec.Command("ps", "-o", "args=", "-p", strconv.Itoa(pid)).Output(); err == nil {
		args = strings.Fields(string(out))
	}
	return isDaemonArgs(args, filepath.Base(exe))
}
