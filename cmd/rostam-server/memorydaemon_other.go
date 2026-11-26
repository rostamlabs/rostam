// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

import "time"

// daemonSupported gates `mcp -autostart` before it touches the filesystem.
const daemonSupported = false

func runMemoryDaemonCmd([]string) { fatal("memory-daemon: " + errDaemonUnsupported.Error()) }

func newDaemonSpawner(string, string) func() error {
	return func() error { return errDaemonUnsupported }
}

// stopDaemon is the daemon's -stop; there is no daemon to stop here.
func stopDaemon(string, time.Duration) error { return errDaemonUnsupported }

// processIsDaemon: unix-only; the Windows build has no daemon.
