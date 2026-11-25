// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

import "errors"

var errDaemonUnsupported = errors.New("the shared memory daemon (memory-daemon, mcp -autostart) is not supported on Windows in this release; use `mcp` with -data for a single embedded session or -connect to a running rostam-server")

// daemonSupported gates `mcp -autostart` before it touches the filesystem.
const daemonSupported = false

func runMemoryDaemonCmd([]string) { fatal("memory-daemon: " + errDaemonUnsupported.Error()) }

func newDaemonSpawner(string, string) func() error {
	return func() error { return errDaemonUnsupported }
}

// stopDaemon, processIsDaemon: unix-only; the Windows build has no daemon.
