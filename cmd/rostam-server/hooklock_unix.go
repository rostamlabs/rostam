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
func lockSession(ctx context.Context, dir, sessionID string) (func(), bool) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, false
	}
	f, err := os.OpenFile(stateFile(dir, sessionID)+".lock", os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G703: hashed name inside the hook state dir
	if err != nil {
		return nil, false
	}
	for {
		err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
		if err == nil {
			return func() { _ = f.Close() }, true // closing drops the flock
		}
		// Only contention is worth waiting out; any other failure (ENOLCK,
		// ENOTSUP on a filesystem without flock) will not clear within the
		// budget.
		if !errors.Is(err, unix.EWOULDBLOCK) && !errors.Is(err, unix.EAGAIN) {
			_ = f.Close()
			return nil, false
		}
		select {
		case <-ctx.Done():
			_ = f.Close()
			return nil, false
		case <-time.After(2 * time.Millisecond):
		}
	}
}
