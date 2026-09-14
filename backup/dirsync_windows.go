// SPDX-License-Identifier: Apache-2.0
//go:build windows

package backup

// dirSyncSupported is false because Windows exposes no directory fsync: a
// directory handle opened through os.Open cannot be flushed (FlushFileBuffers
// answers ERROR_ACCESS_DENIED), and NTFS carries the rename or link through its
// own metadata log rather than leaving it for the caller to force. Mirrors
// vector/dirsync_windows.go. See syncDir in fs.go.
const dirSyncSupported = false
