// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"net"
	"os"
)

// checkListenerOwner reads /proc/net/tcp{,6}, which exists on Linux only;
// elsewhere both reads fail with ErrNotExist and the check passes unchecked.
func checkListenerOwner(addr string) error {
	return listenerOwnerErr(addr, os.ReadFile, os.Getuid())
}

// checkConnOwner: as checkListenerOwner, for an established connection.
func checkConnOwner(local, remote net.Addr) error {
	return connOwnerErr(local, remote, os.ReadFile, os.Getuid())
}
