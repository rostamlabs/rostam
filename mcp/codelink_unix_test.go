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
		if st != fileUnreadable {
			t.Fatalf("state = %v, want fileUnreadable", st)
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

// No symlink may carry hashing outside the project: not a symlinked file, and
// not a symlinked ancestor directory (Lstat on the final component alone does
// not catch the latter). Symlinks that stay inside the project are followed.
func TestFileHashSymlinkConfinement(t *testing.T) {
	outside := t.TempDir()
	writeFile(t, outside, "secret.go", "package secret\n")
	realRoot, _ := symlinkLayout(t)
	for name, target := range map[string]string{
		"out.go":  filepath.Join(outside, "secret.go"), // file symlink out of the root
		"vendor":  outside,                             // dir symlink out of the root
		"self.go": "a.go",                              // file symlink inside the root
	} {
		if err := os.Symlink(target, filepath.Join(realRoot, name)); err != nil {
			t.Skipf("symlink: %v", err)
		}
	}
	for _, rel := range []string{"out.go", "vendor/secret.go"} {
		if h, st := fileHash(realRoot, rel); st != fileUnreadable || h != "" {
			t.Errorf("fileHash(%q) = %q, %v; want \"\", fileUnreadable", rel, h, st)
		}
	}
	want, st := fileHash(realRoot, "a.go")
	if st != fileOK {
		t.Fatalf("a.go: %v", st)
	}
	if h, st := fileHash(realRoot, "self.go"); st != fileOK || h != want {
		t.Errorf("in-root file symlink = %q, %v; want a.go's hash", h, st)
	}
	if _, st := fileHash(realRoot, "docs/x.md"); st != fileOK {
		t.Errorf("in-root dir symlink docs/x.md = %v, want fileOK", st)
	}
}

// A linked file that was hashed and is now unreadable (here, replaced by a
// directory) is stale; one that never had a hash stays unflagged.
func TestStaleWhenFileBecomesUnreadable(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.go", "package a\n")
	stored, st := fileHash(root, "a.go")
	if st != fileOK {
		t.Fatal(st)
	}
	if err := os.Remove(filepath.Join(root, "a.go")); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "a.go"), 0o700); err != nil {
		t.Fatal(err)
	}
	cur, st := fileHash(root, "a.go")
	if st != fileUnreadable {
		t.Fatalf("directory at a.go: state %v, want fileUnreadable", st)
	}
	if got := staleState("a.go", stored, cur, st); got != true {
		t.Errorf("hashed then unreadable: stale = %v, want true", got)
	}
	if got := staleState("a.go", "", cur, st); got != nil {
		t.Errorf("never hashed: stale = %v, want nil", got)
	}
	if got := staleState("a.go", stored, "", fileNoRoot); got != nil {
		t.Errorf("no project root: stale = %v, want nil", got)
	}
}
