// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import "os"

// checkListenerOwner reads /proc/net/tcp{,6}, which exists on Linux only;
// elsewhere both reads fail with ErrNotExist and the check passes unchecked.
func checkListenerOwner(addr string) error {
	return listenerOwnerErr(addr, os.ReadFile, os.Getuid())
}

// checkConnOwner: as checkListenerOwner, after the dial (see connOwnerErr).
func checkConnOwner(addr string) error {
	return connOwnerErr(addr, os.ReadFile, os.Getuid())
}
