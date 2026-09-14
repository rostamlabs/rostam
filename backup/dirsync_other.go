// SPDX-License-Identifier: Apache-2.0
//go:build !windows

package backup

// dirSyncSupported reports whether the platform can be asked to make a
// directory entry durable. See syncDir in fs.go.
const dirSyncSupported = true
