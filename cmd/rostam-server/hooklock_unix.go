// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"context"
	"errors"
	"os"
	"time"

	"golang.org/x/sys/unix"
)

// lockSession takes an exclusive advisory flock on the session's lock file so
// concurrent hooks for one session serialise their read-seen -> render ->
// save sequence. It polls a non-blocking flock so the wait honours ctx; on
// timeout or any error it reports false and the caller injects nothing.
//
// A lock file may be unlinked by pruning (removeLockIfFree) while a hook is
// waiting on it. Holding the flock of an unlinked file excludes nobody, so
// after acquiring it the path is checked to still name the locked file, and
// the lock is retaken on the current file if not.
func lockSession(ctx context.Context, dir, sessionID string) (func(), bool) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, false
	}
	path := stateFile(dir, sessionID) + ".lock"
	for {
		f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G703: hashed name inside the hook state dir
		if err != nil {
			return nil, false
		}
		if flockWait(ctx, f) != nil {
			_ = f.Close()
			return nil, false
		}
		if sameFileAsPath(f, path) {
			return func() { _ = f.Close() }, true // closing drops the flock
		}
		_ = f.Close() // pruned while we waited: lock the file that is there now
	}
}

// lockContendedHook, when set (tests only), is called the first time a
// flockWait finds its file locked by someone else.
var lockContendedHook func()

// flockWait takes an exclusive flock on f, waiting out contention until ctx
// ends (errLockBusy). Only contention is worth waiting out; any other failure
// (ENOLCK, ENOTSUP on a filesystem without flock) will not clear within the
// budget and is returned at once.
func flockWait(ctx context.Context, f *os.File) error {
	for first := true; ; first = false {
		err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
		if err == nil {
			return nil
		}
		if !errors.Is(err, unix.EWOULDBLOCK) && !errors.Is(err, unix.EAGAIN) {
			return err
		}
		if first && lockContendedHook != nil {
			lockContendedHook()
		}
		select {
		case <-ctx.Done():
			return errLockBusy
		case <-time.After(2 * time.Millisecond):
		}
	}
}

func sameFileAsPath(f *os.File, path string) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	pi, err := os.Stat(path) //nolint:gosec // G703: hashed name inside the hook state dir
	return err == nil && os.SameFile(fi, pi)
}

// lockFileCtx takes an exclusive flock on path (created 0600), waiting out
// contention until ctx ends (errLockBusy). Any other flock failure is
// returned as is. The file is never unlinked.
func lockFileCtx(ctx context.Context, path string) (func(), error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G703: fixed name in the daemon data dir
	if err != nil {
		return nil, err
	}
	if err := flockWait(ctx, f); err != nil {
		_ = f.Close()
		return nil, err
	}
	return func() { _ = f.Close() }, nil
}

// registryLocked: the namespace registry is serialised across processes.
const registryLocked = true

// removeLockIfFree unlinks a session lock file only while holding its flock,
// so no hook is inside its critical section; a hook already waiting on the
// unlinked file notices (lockSession's path check) and relocks the new one.
// A held lock is left alone.
func removeLockIfFree(path string) {
	f, err := os.OpenFile(path, os.O_RDWR, 0) //nolint:gosec // G703: name matched inside the hook state dir
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()
	if unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB) != nil || !sameFileAsPath(f, path) {
		return
	}
	_ = os.Remove(path) //nolint:gosec // G703: as above
}
