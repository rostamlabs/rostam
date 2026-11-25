// SPDX-License-Identifier: Apache-2.0

//go:build unix

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/mcp"
)

// TestRelInRootMatchesMCPNormalisation pins the hook and the MCP server to one
// path rule: a file reached through a symlinked prefix, an in-repo symlinked
// dir, or a new file under a new dir maps to the same repo-relative path on
// both sides, and escapes are rejected on both.
func TestRelInRootMatchesMCPNormalisation(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	realRoot := filepath.Join(base, "real", "repo")
	if err := os.MkdirAll(filepath.Join(realRoot, "site", "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"a.go", "site/docs/x.md"} {
		if err := os.WriteFile(filepath.Join(realRoot, f), []byte("x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(filepath.Join("site", "docs"), filepath.Join(realRoot, "docs")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	if err := os.Symlink(filepath.Join(base, "real"), filepath.Join(base, "link")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	linkRoot := filepath.Join(base, "link", "repo")

	type tc struct{ cwd, file, want string } // want "" = rejected
	cases := []tc{
		{linkRoot, "a.go", "a.go"},                          // symlinked cwd, relative file
		{linkRoot, "docs/x.md", "docs/x.md"},                // in-repo symlinked dir kept as written
		{realRoot, "docs/x.md", "docs/x.md"},                //
		{realRoot, filepath.Join(linkRoot, "a.go"), "a.go"}, // absolute via symlinked prefix
		{linkRoot, "newdir/new.go", "newdir/new.go"},        // new file under a new dir
		{filepath.Join(linkRoot, "docs"), "../a.go", "a.go"},
		{linkRoot, "../x", ""},
		{linkRoot, "../../../etc/passwd", ""},
		{realRoot, "/etc/passwd", ""},
	}
	for _, root := range []string{realRoot, linkRoot} {
		for _, c := range cases {
			got, ok := relInRoot(root, c.cwd, c.file)
			if c.want == "" {
				if ok {
					t.Errorf("root %s cwd %s file %s: relInRoot = %q, want rejected", root, c.cwd, c.file, got)
				}
				continue
			}
			if !ok || got != c.want {
				t.Errorf("root %s cwd %s file %s: relInRoot = %q, %v; want %q", root, c.cwd, c.file, got, ok, c.want)
			}
			abs := c.file
			if !filepath.IsAbs(abs) {
				abs = filepath.Join(c.cwd, abs)
			}
			if m, err := mcp.NormalizeLinkPath(root, abs); err != nil || m != got {
				t.Errorf("root %s: MCP side maps %s to %q, %v; hook side %q", root, abs, m, err, got)
			}
		}
	}
}

func TestHookLockWaitBoundedByContext(t *testing.T) {
	dir := t.TempDir()
	// unix only: the !unix lockSession is a no-op that always succeeds.
	release, ok := lockSession(context.Background(), dir, "s")
	if !ok {
		t.Fatal("first lock not acquired")
	}
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, ok := lockSession(ctx, dir, "s"); ok {
		t.Fatal("second lock acquired while held")
	}
	if time.Since(start) > time.Second {
		t.Fatal("lock wait not bounded by ctx")
	}
}

const flockSupported = true

// Pruning never removes a lock file somebody holds.
func TestPruneKeepsHeldLock(t *testing.T) {
	dir := t.TempDir()
	release, ok := lockSession(context.Background(), dir, "held")
	if !ok {
		t.Fatal("lock not acquired")
	}
	defer release()
	lock := stateFile(dir, "held") + ".lock"
	ancient := time.Now().Add(-2 * hookStateMaxAge)
	if err := os.Chtimes(lock, ancient, ancient); err != nil {
		t.Fatal(err)
	}
	pruneHookState(dir, hookStateMaxAge, time.Now())
	if _, err := os.Stat(lock); err != nil {
		t.Fatalf("held lock file was removed: %v", err)
	}
}

// A hook waiting on a lock file that is unlinked under it must not end up
// "holding" the dead file: it relocks the file now at the path, so it still
// excludes a newcomer.
func TestLockSessionRelocksAfterUnlink(t *testing.T) {
	dir := t.TempDir()
	release, ok := lockSession(context.Background(), dir, "s")
	if !ok {
		t.Fatal("first lock not acquired")
	}
	got := make(chan func(), 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		r, ok := lockSession(ctx, dir, "s")
		if !ok {
			got <- nil
			return
		}
		got <- r
	}()
	time.Sleep(50 * time.Millisecond) // the waiter is now polling the old inode
	if err := os.Remove(stateFile(dir, "s") + ".lock"); err != nil {
		t.Fatal(err)
	}
	release()
	r := <-got
	if r == nil {
		t.Fatal("waiter never got the lock")
	}
	defer r()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if r2, ok := lockSession(ctx, dir, "s"); ok {
		r2()
		t.Fatal("a newcomer took the lock while the waiter holds it: the waiter held an unlinked file")
	}
}
