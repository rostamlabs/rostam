// SPDX-License-Identifier: Apache-2.0

//go:build unix

package mcp

import (
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestFileHashNonRegularDoesNotBlock(t *testing.T) {
	root := t.TempDir()
	fifo := filepath.Join(root, "pipe")
	if err := syscall.Mkfifo(fifo, 0o600); err != nil {
		t.Skipf("mkfifo: %v", err)
	}
	done := make(chan fileState, 1)
	go func() { _, st := fileHash(root, "pipe"); done <- st }()
	select {
	case st := <-done:
		if st != fileNoRoot {
			t.Fatalf("state = %v, want fileNoRoot", st)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("fileHash blocked on a FIFO")
	}
}
