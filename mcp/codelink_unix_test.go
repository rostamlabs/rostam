// SPDX-License-Identifier: Apache-2.0

//go:build unix

package mcp

import (
	"os"
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

// symlinkLayout builds base/real/repo (the real checkout, with an in-repo
// symlinked dir docs -> site/docs) and base/link -> base/real, so base/link/repo
// is the same checkout reached through a symlinked prefix.
func symlinkLayout(t *testing.T) (realRoot, linkRoot string) {
	t.Helper()
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	realRoot = filepath.Join(base, "real", "repo")
	writeFile(t, realRoot, "a.go", "package a\n")
	writeFile(t, realRoot, "site/docs/x.md", "x\n")
	if err := os.Symlink(filepath.Join("site", "docs"), filepath.Join(realRoot, "docs")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	if err := os.Symlink(filepath.Join(base, "real"), filepath.Join(base, "link")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	return realRoot, filepath.Join(base, "link", "repo")
}

func TestNormalizeLinkPathSymlinks(t *testing.T) {
	realRoot, linkRoot := symlinkLayout(t)
	for _, root := range []string{realRoot, linkRoot} {
		ok := map[string]string{
			filepath.Join(realRoot, "a.go"):             "a.go",
			filepath.Join(linkRoot, "a.go"):             "a.go",
			filepath.Join(realRoot, "docs", "x.md"):     "docs/x.md", // in-repo symlink kept as written
			filepath.Join(linkRoot, "docs", "x.md"):     "docs/x.md",
			filepath.Join(linkRoot, "newdir", "new.go"): "newdir/new.go", // does not exist yet
			filepath.Join(realRoot, "newdir", "new.go"): "newdir/new.go",
			"docs/x.md": "docs/x.md",
		}
		for in, want := range ok {
			got, err := NormalizeLinkPath(root, in)
			if err != nil || got != want {
				t.Errorf("root %s: NormalizeLinkPath(%q) = %q, %v; want %q", root, in, got, err, want)
			}
		}
		for _, in := range []string{
			filepath.Join(linkRoot, "..", "x"),
			filepath.Join(realRoot, "..", "..", "etc", "passwd"),
			filepath.Dir(linkRoot),
			"../x",
			"/etc/passwd",
		} {
			if got, err := NormalizeLinkPath(root, in); err == nil {
				t.Errorf("root %s: NormalizeLinkPath(%q) = %q, want error", root, in, got)
			}
		}
	}
}
