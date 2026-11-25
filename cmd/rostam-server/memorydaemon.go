// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
)

// Defaults shared by the daemon, `mcp -autostart` and `hook`: all three must
// agree on where the daemon listens and where its token lives, or a hook would
// silently talk to nothing.
const (
	memoryDefaultAddr = "127.0.0.1:7480"
	memoryTokenName   = "token"
	daemonLogName     = "daemon.log"
	daemonPidName     = "daemon.pid"
	daemonConfigName  = "daemon-config.json"
	spawnLockDirName  = "spawn.lock"

	// daemonStartWait bounds how long a session waits for a freshly spawned
	// daemon to accept an authenticated connection.
	daemonStartWait = 5 * time.Second

	// daemonCacheBudget is the cache bound handed to the daemon's server: a
	// personal memory store, not a server. Same figure mcpSetup uses embedded.
	daemonCacheBudget = "256MiB"
	daemonShards      = 8
)

// defaultMemoryDir is ~/.rostam/memory, the daemon's data directory.
func defaultMemoryDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if home == "" {
		return "", errors.New("empty home directory")
	}
	return filepath.Join(home, ".rostam", "memory"), nil
}

// ensureToken returns the daemon's bearer token from <dir>/token, creating it
// on first run: 32 bytes from crypto/rand, hex, mode 0600. Creation is atomic
// (write a private temp file, then hard-link it into place, which fails if the
// name exists), so concurrent first runs all end up reading the single winner's
// token and no reader ever sees a partial file.
func ensureToken(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", fmt.Errorf("creating %s: %w", dir, err)
	}
	path := filepath.Join(dir, memoryTokenName)
	if tok, err := readToken(path); err == nil {
		return tok, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}

	var raw [32]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", fmt.Errorf("generating token: %w", err)
	}
	tok := hex.EncodeToString(raw[:])

	tmp, err := os.CreateTemp(dir, ".token-*") // CreateTemp opens O_EXCL, mode 0600
	if err != nil {
		return "", fmt.Errorf("creating token: %w", err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(tok + "\n"); err != nil {
		_ = tmp.Close()
		return "", fmt.Errorf("writing token: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("writing token: %w", err)
	}
	if err := os.Link(tmp.Name(), path); err != nil {
		// Lost the race (or no hard links): whoever is there now wins.
		if t, rerr := readToken(path); rerr == nil {
			return t, nil
		}
		return "", fmt.Errorf("installing token: %w", err)
	}
	return tok, nil
}

// readToken reads a token file, tightening a loose mode: a token that other
// local users can read is no token.
func readToken(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	tok := strings.TrimSpace(string(b))
	if tok == "" {
		return "", fmt.Errorf("token file %s is empty; delete it to generate a new one", path)
	}
	if fi, err := os.Stat(path); err == nil && fi.Mode().Perm()&0o077 != 0 {
		_ = os.Chmod(path, 0o600)
	}
	return tok, nil
}

// validateLoopbackAddr accepts only host:port where host is a literal loopback
// IP. Hostnames (even "localhost", which resolution can redirect), wildcard
// binds and the empty host are refused: the daemon authenticates with a static
// token over plaintext, which is only acceptable on the loopback interface.
func validateLoopbackAddr(addr string) error {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("invalid -addr %q: %w", addr, err)
	}
	if n, err := strconv.Atoi(port); err != nil || n < 1 || n > 65535 {
		return fmt.Errorf("invalid -addr %q: bad port %q", addr, port)
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("-addr %q refused: the memory daemon binds loopback only (an IP such as 127.0.0.1 or ::1, not a hostname or wildcard)", addr)
	}
	return nil
}

// errTokenRejected means something is listening at the address but refused
// our token: spawning another daemon cannot help.
var errTokenRejected = errors.New("daemon rejected the token")

// probeFunc reports nil when an authenticated daemon answers at addr.
type probeFunc func(ctx context.Context, addr, token string) error

// probeDaemon connects with the token and performs one real read, so a wrong
// token or an unrelated listener on the port counts as "not our daemon".
func probeDaemon(ctx context.Context, addr, token string) error {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	// Cheap reachability check first: while the daemon is still starting this
	// fails fast and avoids building a full client per poll.
	var d net.Dialer
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return err
	}
	_ = c.Close()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		return err
	}
	store, err := connectStore(storeFlags{connect: addr, authToken: token}, reg,
		func(string) (string, bool) { return "", false })
	if err != nil {
		return err
	}
	defer store.Close()
	if _, err := store.Get(ctx, []byte("__rostam_probe__")); err != nil && !errors.Is(err, rostam.ErrNotFound) {
		if errors.Is(err, client.ErrUnauthorized) {
			return errTokenRejected
		}
		return err
	}
	return nil
}

// ensureDaemon makes sure an authenticated memory daemon answers at addr,
// calling spawn at most once across all concurrent callers.
func ensureDaemon(ctx context.Context, data, addr string, spawn func() error) error {
	return ensureDaemonWith(ctx, data, addr, spawn, probeDaemon, daemonStartWait)
}

func ensureDaemonWith(ctx context.Context, data, addr string, spawn func() error, probe probeFunc, wait time.Duration) error {
	token, err := ensureToken(data)
	if err != nil {
		return err
	}
	logPath := filepath.Join(data, daemonLogName)
	// up reports a ready daemon; a non-nil error is final (something answers
	// but refuses our token, so spawning another daemon cannot help).
	up := func() (bool, error) {
		perr := probe(ctx, addr, token)
		if errors.Is(perr, errTokenRejected) {
			return false, fmt.Errorf("a server on %s rejected the token in %s; run \"rostam-server memory-daemon -stop\" and retry",
				addr, filepath.Join(data, memoryTokenName))
		}
		return perr == nil, nil
	}
	if ok, err := up(); ok || err != nil {
		return err
	}

	notReady := fmt.Errorf("memory daemon did not become ready on %s within %s; see %s", addr, wait, logPath)
	deadline := time.Now().Add(wait)
	lockDir := filepath.Join(filepath.Dir(data), spawnLockDirName)

	for {
		unlock, lerr := claimDataDir(lockDir)
		if lerr == nil {
			defer unlock()
			// Another session may have finished spawning while we waited.
			if ok, err := up(); ok || err != nil {
				return err
			}
			if err := spawn(); err != nil {
				return fmt.Errorf("starting memory daemon: %w (see %s)", err, logPath)
			}
			for time.Now().Before(deadline) {
				if ok, err := up(); ok || err != nil {
					return err
				}
				if err := sleepCtx(ctx, 50*time.Millisecond); err != nil {
					return err
				}
			}
			return notReady
		}
		if !errors.Is(lerr, errDataDirBusy) {
			return fmt.Errorf("taking daemon spawn lock: %w", lerr)
		}
		// A sibling session is spawning: wait for it to come up.
		if err := sleepCtx(ctx, 25*time.Millisecond); err != nil {
			return err
		}
		if ok, err := up(); ok || err != nil {
			return err
		}
		if !time.Now().Before(deadline) {
			return notReady
		}
	}
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// autostartConflictFlags select a different store or credentials; -autostart
// always uses its own daemon and token.
var autostartConflictFlags = []string{"data", "connect", "auth-token", "tls-ca", "tls-cert", "tls-key", "tls-server-name"}

// checkAutostartConflicts rejects -autostart combined with any flag in
// autostartConflictFlags. visited is the set of flags given explicitly.
func checkAutostartConflicts(visited map[string]bool) error {
	for _, name := range autostartConflictFlags {
		if visited[name] {
			return fmt.Errorf("mcp: -autostart cannot be combined with -%s (it connects to its own daemon with its own token)", name)
		}
	}
	return nil
}

// defaultProjectRoot picks the MCP project root. An explicit flag is used as
// given (made absolute); otherwise CLAUDE_PROJECT_DIR, else the working
// directory, resolved to the enclosing git top-level.
func defaultProjectRoot(flagVal string, getenv func(string) string, getwd func() (string, error)) string {
	if flagVal != "" {
		if abs, err := filepath.Abs(flagVal); err == nil {
			return abs
		}
		return flagVal
	}
	dir := getenv("CLAUDE_PROJECT_DIR")
	if dir == "" {
		wd, err := getwd()
		if err != nil {
			return ""
		}
		dir = wd
	}
	root, _ := resolveProject(dir)
	return root
}

// isDaemonArgs reports whether a process command line is `<exe> memory-daemon ...`
// for our executable name.
func isDaemonArgs(args []string, exeBase string) bool {
	return len(args) >= 2 && filepath.Base(args[0]) == exeBase && args[1] == "memory-daemon"
}
