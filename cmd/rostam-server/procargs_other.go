// SPDX-License-Identifier: Apache-2.0

//go:build unix && !darwin

package main

import (
	"os/exec"
	"strconv"
	"strings"
)

// platformProcArgs is the last resort on a unix with neither /proc nor
// KERN_PROCARGS2: ps prints the command line joined by spaces, so argv is
// split on whitespace and a path containing a space will not match (stop
// then refuses to signal rather than signalling the wrong process).
func platformProcArgs(pid int) []string {
	out, err := exec.Command("ps", "-o", "args=", "-p", strconv.Itoa(pid)).Output() //nolint:gosec // G204: fixed "ps" binary, numeric pid, no shell
	if err != nil {
		return nil
	}
	return strings.Fields(string(out))
}
