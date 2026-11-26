// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

// daemonSupported gates `mcp -autostart` before it touches the filesystem.
const daemonSupported = false

func runMemoryDaemonCmd([]string) { fatal("memory-daemon: " + errDaemonUnsupported.Error()) }

func newDaemonSpawner(string, string) func() error {
	return func() error { return errDaemonUnsupported }
}

// stopDaemon, processIsDaemon: unix-only; the Windows build has no daemon.
