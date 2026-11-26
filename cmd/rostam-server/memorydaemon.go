// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
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
	"github.com/rostamlabs/rostam/mcp"
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
	if err := os.MkdirAll(dir, 0o700); err != nil { //nolint:gosec // G703: dir is the daemon data dir (home default or the user's own -data flag)
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
	defer func() { _ = os.Remove(tmp.Name()) }() //nolint:gosec // G703: CreateTemp name inside the data dir; fails harmlessly once linked
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
// local users can read is no token, so one whose mode cannot be checked or
// tightened is refused rather than used.
//
// path is <data dir>/token, where the data dir is the home default or the
// user's own -data flag; the G703 findings below are that trusted path.
func readToken(path string) (string, error) {
	b, err := os.ReadFile(path) //nolint:gosec // G703: fixed name inside the daemon data dir
	if err != nil {
		return "", err
	}
	tok := strings.TrimSpace(string(b))
	if tok == "" {
		return "", fmt.Errorf("token file %s is empty; delete it to generate a new one", path)
	}
	fi, err := os.Stat(path) //nolint:gosec // G703: as above
	if err != nil {
		return "", fmt.Errorf("checking token file %s: %w", path, err)
	}
	if fi.Mode().Perm()&0o077 != 0 {
		if err := os.Chmod(path, 0o600); err != nil { //nolint:gosec // G703: as above
			return "", fmt.Errorf("token file %s is readable by other users and its mode cannot be tightened to 0600: %w", path, err)
		}
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
	// Never hand the token to another user's process squatting on the port.
	if err := listenerOwnerCheck(addr); err != nil {
		return err
	}
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		return err
	}
	store, err := connectStore(storeFlags{connect: addr, authToken: token, beforeConnect: ownerCheckBeforeConnect, afterConnect: ownerCheckAfterConnect}, reg,
		func(string) (string, bool) { return "", false })
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
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
		var owner *portOwnerError
		if errors.As(perr, &owner) {
			return false, owner // spawning cannot help: our daemon could never bind the port
		}
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
			defer func() { _ = unlock() }()
			// Another session may have finished spawning while we waited.
			if ok, err := up(); ok || err != nil {
				return err
			}
			if err := embeddedHolderErr(data); err != nil {
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

// defaultProjectRoot picks the MCP project root and its namespace. An explicit
// flag is used as given (made absolute); otherwise CLAUDE_PROJECT_DIR, else the
// working directory, resolved to the enclosing git top-level. The namespace is
// resolveProject's: the MAIN checkout's directory name, so a linked worktree
// shares its repository's namespace, exactly as the hook computes it.
func defaultProjectRoot(flagVal string, getenv func(string) string, getwd func() (string, error)) (root, namespace string) {
	if flagVal != "" {
		abs, err := filepath.Abs(flagVal)
		if err != nil {
			abs = flagVal
		}
		_, namespace = resolveProject(abs)
		return abs, namespace
	}
	dir := getenv("CLAUDE_PROJECT_DIR")
	if dir == "" {
		wd, err := getwd()
		if err != nil {
			return "", ""
		}
		dir = wd
	}
	return resolveProject(dir)
}

// mcpDefaultNamespace is the namespace `mcp` uses when a tool call omits one
// ("" = the historical "default"). An explicit -namespace always wins. Without
// it, the resolved project namespace is used only when the session was started
// FOR a project: -autostart (the Claude Code plugin) or an explicit
// -project-root. A plain embedded `mcp` keeps "default", so existing setups
// that never passed a namespace still see their memories.
func mcpDefaultNamespace(resolved, flagNS string, nsGiven, autostart, rootGiven bool) string {
	switch {
	case nsGiven:
		return flagNS
	case autostart || rootGiven:
		return resolved
	}
	return ""
}

// isDaemonArgs reports whether a process command line is `<exe> memory-daemon ...`
// for our executable name serving data: its -data flag names data, or it has
// none and data is defData (the flag's default). A recycled pid that is some
// OTHER data dir's daemon is therefore not ours.
func isDaemonArgs(args []string, exeBase, data, defData string) bool {
	if len(args) < 2 || filepath.Base(args[0]) != exeBase || args[1] != "memory-daemon" {
		return false
	}
	got, found := defData, false
	for i := 2; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-data" || a == "--data":
			if i+1 < len(args) {
				got, found = args[i+1], true
			}
		case strings.HasPrefix(a, "-data="), strings.HasPrefix(a, "--data="):
			_, got, _ = strings.Cut(a, "=")
			found = true
		}
	}
	if !found && defData == "" {
		return false
	}
	return got != "" && filepath.Clean(got) == filepath.Clean(data)
}

// parseProcArgs2 decodes a macOS KERN_PROCARGS2 buffer: a native-endian int32
// argc, the executable path, NUL padding, then argc NUL-terminated argv
// strings. Unlike `ps -o args=` it keeps argv boundaries, so a path with a
// space in it still matches. nil when the buffer is malformed.
func parseProcArgs2(buf []byte) []string {
	if len(buf) < 4 {
		return nil
	}
	argc := int(int32(binary.NativeEndian.Uint32(buf[:4]))) //nolint:gosec // G115: kernel-reported count, bounded below
	rest := buf[4:]
	i := bytes.IndexByte(rest, 0) // end of the executable path
	if argc <= 0 || i < 0 {
		return nil
	}
	rest = rest[i:]
	for len(rest) > 0 && rest[0] == 0 {
		rest = rest[1:]
	}
	args := make([]string, 0, argc)
	for len(args) < argc {
		j := bytes.IndexByte(rest, 0)
		if j < 0 {
			return nil
		}
		args = append(args, string(rest[:j]))
		rest = rest[j+1:]
	}
	return args
}

// mcpInitErr adds -autostart's remedies to an embedder-identity mismatch: the
// store there is the shared daemon's directory, which -data cannot change.
func mcpInitErr(err error, autostart bool, dir string) error {
	if !autostart || !errors.Is(err, mcp.ErrEmbedderMismatch) {
		return err
	}
	return fmt.Errorf("%w. Under -autostart the store is the shared daemon's %s: either set the same ROSTAM_EMBED_* variables in the environment your MCP client (Claude Code) is started from, "+
		"or stop the daemon (rostam-server memory-daemon -stop) and move %s aside to start a fresh store", err, dir, dir)
}

// daemonPidFileExists reports whether dir holds a memory daemon's pid file:
// the supervisor writes it once its server is started and removes it on exit.
func daemonPidFileExists(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, daemonPidName))
	return err == nil
}

// dataDirBusyMsg is the embedded `mcp` error for a data dir whose lock is held,
// naming the daemon when it is the likely holder.
func dataDirBusyMsg(dir string) string {
	if daemonPidFileExists(dir) {
		return "mcp: the shared memory daemon is using this data directory; use -autostart to connect to it, " +
			"or stop it first with \"rostam-server memory-daemon -stop\""
	}
	return "mcp: another rostam-server mcp process is using this data directory; " +
		"a data dir has one writer, so concurrent clients must share one server over -connect " +
		"(or -autostart, the shared local daemon) instead of each embedding their own"
}

// embeddedHolderErr reports an embedded `rostam-server mcp` holding data: the
// data-dir lock is taken and no daemon pid file exists. Spawning a daemon then
// cannot succeed (it would lose the lock), so ensureDaemon says so instead of
// timing out.
func embeddedHolderErr(data string) error {
	unlock, err := lockDataDir(data)
	if err == nil {
		_ = unlock()
		return nil
	}
	if !errors.Is(err, errDataDirBusy) || daemonPidFileExists(data) {
		return nil
	}
	return fmt.Errorf("an embedded `rostam-server mcp` is using %s; stop it or remove the old MCP server entry that runs it, then retry", data)
}
