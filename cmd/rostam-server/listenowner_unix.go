// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import "os"

// checkListenerOwner reads /proc/net/tcp{,6}, which exists on Linux only;
// elsewhere both reads fail and the check passes (best effort).
func checkListenerOwner(addr string) error {
	return listenerOwnerErr(addr, os.ReadFile, os.Getuid())
}
