// SPDX-License-Identifier: Apache-2.0

//go:build !unix && !windows

package main

import "context"

// lockFileCtx is a no-op on the remaining platforms (no flock, no LockFileEx):
// concurrent first sessions there could race on the namespace registry.
func lockFileCtx(context.Context, string) (func(), error) { return func() {}, nil }

// registryLocked: the namespace registry is NOT serialised on this platform.
const registryLocked = false
