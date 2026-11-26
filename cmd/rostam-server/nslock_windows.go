// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"os"
	"time"

	"golang.org/x/sys/windows"
)

// lockFileCtx takes an exclusive LockFileEx lock on path (created 0600),
// retrying while another process holds it until ctx ends (errLockBusy), like
// the unix flock version. Any other failure is returned as is. The registry
// lives in ~/.rostam/memory, shared by every `mcp -project-root` process, so
// it must be serialised here too.
func lockFileCtx(ctx context.Context, path string) (func(), error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	for {
		var ol windows.Overlapped
		err := windows.LockFileEx(windows.Handle(f.Fd()),
			lockfileExclusiveLock|lockfileFailImmediately, 0, 1, 0, &ol)
		if err == nil {
			return func() { _ = f.Close() }, nil
		}
		if !errors.Is(err, windows.ERROR_LOCK_VIOLATION) && !errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
			_ = f.Close()
			return nil, err
		}
		select {
		case <-ctx.Done():
			_ = f.Close()
			return nil, errLockBusy
		case <-time.After(2 * time.Millisecond):
		}
	}
}

// registryLocked: the namespace registry is serialised across processes.
const registryLocked = true
