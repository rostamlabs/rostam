// SPDX-License-Identifier: Apache-2.0
//go:build unix

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// dataDirLockName is the lock file an embedded session (mcp today; llm-proxy
// will too) holds inside its data directory for the lifetime of the process.
// The name is kept as ".mcp.lock" for compatibility: existing user data dirs
// already contain a lock file with this name.
const dataDirLockName = ".mcp.lock"

// lockDataDir takes an exclusive advisory lock on dir, and returns the closer
// that releases it.
//
// An embedded data directory is a single-writer store: two rostam-server mcp
// processes opening the same -data dir both map the same cache files and both
// believe they own them, which corrupts the store rather than failing. There is
// no coordination between them to fix that after the fact, so the second one is
// refused up front.
//
// The lock is flock(2)-based, which means it is advisory (only processes that
// also take it are held off) and released automatically by the kernel if this
// process dies without closing the file — a crash must not leave a data dir
// permanently unopenable. It is also per-open-file-description, so it does NOT
// protect against the same process opening the dir twice; that isn't a case
// runMcpCmd can produce.
func lockDataDir(dir string) (func() error, error) {
	f, err := lockDataDirFile(dir)
	if err != nil {
		return nil, err
	}
	// Closing the file drops the flock. The lock file itself is left behind on
	// purpose: unlinking it would let a second process create a fresh one and
	// lock that instead, which is the same as having no lock at all.
	return f.Close, nil
}

// lockDataDirFile is lockDataDir returning the locked file, so a caller can
// hand it to a child process (exec.Cmd.ExtraFiles): flock belongs to the open
// file description, which the child then shares.
func lockDataDirFile(dir string) (*os.File, error) {
	path := filepath.Join(dir, dataDirLockName)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("opening lock file %s: %w", path, err)
	}
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		_ = f.Close()
		return nil, flockErr(path, err)
	}
	return f, nil
}

func flockErr(path string, err error) error {
	if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
		return fmt.Errorf("%w: %w", errDataDirBusy, err)
	}
	return fmt.Errorf("locking %s: %w", path, err)
}
