// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

import (
	"context"
	"errors"
)

// lockSession degrades to no locking where flock is unavailable: concurrent
// hooks may then occasionally inject a memory twice, which beats failing.
func lockSession(context.Context, string, string) (func(), bool) {
	return func() {}, true
}

// removeLockIfFree is a no-op: without flock no lock files are created.
func removeLockIfFree(string) {}

// errLockBusy: see the unix build; never returned here.
var errLockBusy = errors.New("lock is busy")

// lockFileCtx degrades to no locking where flock is unavailable. There is no
// memory daemon on these platforms, so only an embedded `mcp -project-root`
// reaches the namespace registry, one session at a time per data dir.
func lockFileCtx(context.Context, string) (func(), error) { return func() {}, nil }
