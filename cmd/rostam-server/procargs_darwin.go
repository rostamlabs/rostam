// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import "golang.org/x/sys/unix"

// platformProcArgs reads pid's exact argv from the kernel (KERN_PROCARGS2),
// so a data dir or executable path containing spaces still matches.
func platformProcArgs(pid int) []string {
	buf, err := unix.SysctlRaw("kern.procargs2", pid)
	if err != nil {
		return nil
	}
	return parseProcArgs2(buf)
}
