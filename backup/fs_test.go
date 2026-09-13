// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestFSObjectStorePutRoundTrip verifies that a value written via Put is
// durably persisted and reads back byte-for-byte, and that the temp file used
// for the atomic write is renamed away (never left behind) on success. The
// durability discipline the fix adds — tmp.Sync() before rename plus a
// parent-directory fsync — must not break this happy path.
func TestFSObjectStorePutRoundTrip(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}

	const key = "tenant/coll/2026-07-08T00-00-00Z.snap"
	want := []byte("durable snapshot payload")
	if err := store.Put(context.Background(), key, strings.NewReader(string(want)), int64(len(want))); err != nil {
		t.Fatalf("Put: %v", err)
	}

	rc, err := store.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer rc.Close()
	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("round-trip mismatch: got %q, want %q", got, want)
	}

	// After a successful Put the atomic temp file must be gone: it was either
	// renamed onto dst or removed on an error path. A lingering putTempPrefix
	// file would signal a broken rename step.
	if tmps := listTempFiles(t, filepath.Join(root, "tenant", "coll")); len(tmps) != 0 {
		t.Fatalf("temp files left behind after Put: %v", tmps)
	}
}

// TestFSObjectStorePutCopyErrorNoDest verifies the error path still removes the
// temp file and leaves no destination behind when the reader fails mid-copy —
// the fix's added Sync steps must not regress this cleanup.
func TestFSObjectStorePutCopyErrorNoDest(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}

	const key = "tenant/coll/bad.snap"
	if err := store.Put(context.Background(), key, errReader{}, 0); err == nil {
		t.Fatal("Put: expected error from failing reader, got nil")
	}

	if _, err := os.Stat(filepath.Join(root, "tenant", "coll", "bad.snap")); !os.IsNotExist(err) {
		t.Fatalf("destination should not exist after failed Put, stat err = %v", err)
	}
	if tmps := listTempFiles(t, filepath.Join(root, "tenant", "coll")); len(tmps) != 0 {
		t.Fatalf("temp files left behind after failed Put: %v", tmps)
	}
}

// errReader always fails, exercising Put's io.Copy error branch.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

// listTempFiles returns the names of any Put staging temp files (putTempPrefix…)
// under dir. A missing dir yields none.
func listTempFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatalf("ReadDir %q: %v", dir, err)
	}
	var tmps []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), putTempPrefix) {
			tmps = append(tmps, e.Name())
		}
	}
	return tmps
}

// TestFSObjectStoreListPrefixScopedWalk pins that List, which starts its walk
// at the directory the prefix implies instead of at root, returns exactly what
// a full-root walk filtered by key prefix would — including a trailing partial
// segment ("acme/col/2024-"), a single exact key, a prefix naming a file as a
// directory, a NUL byte, a ".." prefix, and a prefix whose directory does not
// exist (nil, no error). Symlinks are the one deliberate difference: a start
// path through one — escaping root or not — is an error rather than silence.
func TestFSObjectStoreListPrefixScopedWalk(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	fsStore, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	keys := []string{
		"acme/col/2024-01-01T00:00:00Z.snap",
		"acme/col/2024-01-01T00:00:00Z.cfg.json",
		"acme/col/2025-01-01T00:00:00Z.snap",
		"acme/other/2024-01-01T00:00:00Z.snap",
		"beta/col/2024-01-01T00:00:00Z.snap",
		"top.snap",
	}
	if runtime.GOOS != "windows" {
		// A literal backslash in a name: scoping must stop before that segment
		// and let the lexical filter decide, so such keys still list — and a
		// prefix with a backslash in a complete segment still matches nothing
		// that a full-root walk would not have matched.
		keys = append(keys, "bs/x\\y.snap")
	}
	for _, k := range keys {
		if err := fsStore.Put(ctx, k, strings.NewReader(k), int64(len(k))); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}
	// A file outside root that a ".." prefix must never reach.
	outside := filepath.Join(filepath.Dir(root), "outside.snap")
	if err := os.WriteFile(outside, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(outside) })

	listKeys := func(prefix string) []string {
		t.Helper()
		infos, err := fsStore.List(ctx, prefix)
		if err != nil {
			t.Fatalf("list %q: %v", prefix, err)
		}
		out := make([]string, 0, len(infos))
		for _, in := range infos {
			out = append(out, in.Key)
		}
		return out
	}
	// Reference: full-root walk filtered by key prefix.
	fullWalk := func(prefix string) []string {
		var out []string
		for _, k := range keys {
			if strings.HasPrefix(k, prefix) {
				out = append(out, k)
			}
		}
		sortStrings(out)
		return out
	}
	for _, prefix := range []string{
		"", "acme/", "acme/col/", "acme/col/2024-", "acme/col/2024-01-01T00:00:00Z.snap",
		"acme/co", "top", "nope/", "acme/nope/2024-", "../", "../outside", "/acme/col/",
		"top.snap/", "top.snap/x/y", "acme/col/2024-01-01T00:00:00Z.snap/x/",
		"a\x00b/", "acme/\x00/", "bs/x\\", "bs\\x/", "bs\\x/y", "acme\\col/",
		"./acme/col/", "acme/./col/", "acme/../acme/col/", "acme//col/", "/",
	} {
		got, want := listKeys(prefix), fullWalk(prefix)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("prefix %q: got %v, want %v", prefix, got, want)
		}
	}
}

// TestFSObjectStoreListRefusesSymlinkStart pins the symlink rule: a start path
// with a symlink in any component is refused by os.Root — whether the link
// escapes root (the case that let retention delete files outside the backup
// root) or stays inside it — and List reports that as an error, not as "no
// snapshots". The link itself is still reported as a plain entry by a walk
// that merely passes it, exactly as the full-root walk did.
func TestFSObjectStoreListRefusesSymlinkStart(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	fsStore, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := fsStore.Put(ctx, "acme/col/2024-01-01T00:00:00Z.snap", strings.NewReader("a"), 1); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.MkdirAll(filepath.Join(outside, "col"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "col", "escaped.snap"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "link")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	if err := os.Symlink(filepath.Join(root, "acme"), filepath.Join(root, "inlink")); err != nil {
		t.Fatal(err)
	}
	// RELATIVE in-root links are the ones os.Root itself would follow (it only
	// refuses absolute and root-escaping targets), so they are the real test of
	// the per-component Lstat: one at the top, one below the tenant directory.
	if err := os.Symlink("acme", filepath.Join(root, "rel")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("col", filepath.Join(root, "acme", "relcol")); err != nil {
		t.Fatal(err)
	}

	for _, prefix := range []string{"link/", "link/col/", "inlink/", "inlink/col/", "rel/", "rel/col/", "acme/relcol/"} {
		infos, err := fsStore.List(ctx, prefix)
		if err == nil {
			t.Errorf("list %q through a symlink: want an error, got %d keys", prefix, len(infos))
		}
		for _, in := range infos {
			if strings.HasPrefix(in.Key, "link/") || strings.HasPrefix(in.Key, "rel/") || strings.HasPrefix(in.Key, "acme/relcol/") {
				t.Errorf("list %q resolved a symlink: %q", prefix, in.Key)
			}
		}
	}
	// A non-canonical prefix that would CLEAN to a path through a link matches no
	// key (keys are clean), so it must yield nothing — not the link error.
	for _, prefix := range []string{"../link/col/", "acme/../rel/col/", "/rel/col/", "./acme/relcol/"} {
		infos, err := fsStore.List(ctx, prefix)
		if err != nil || len(infos) != 0 {
			t.Errorf("list %q (non-canonical): got %d keys, err=%v; want none, nil", prefix, len(infos), err)
		}
	}
	// Passing the links during a root walk still just reports them as entries.
	infos, err := fsStore.List(ctx, "")
	if err != nil {
		t.Fatalf("root list: %v", err)
	}
	var got []string
	for _, in := range infos {
		got = append(got, in.Key)
	}
	if want := "acme/col/2024-01-01T00:00:00Z.snap,acme/relcol,inlink,link,rel"; strings.Join(got, ",") != want {
		t.Errorf("root list = %v, want %s", got, want)
	}
	// The file behind the escaping link was never touched.
	if _, err := os.Stat(filepath.Join(outside, "col", "escaped.snap")); err != nil {
		t.Errorf("file outside root disturbed: %v", err)
	}
}

// TestFSObjectStoreListIsScoped pins that the walk really starts at the
// prefix's directory and not at root: an unreadable directory in an UNRELATED
// subtree fails a root walk (WalkDir reports the ReadDir error) but must not be
// visited — and so must not fail — by a walk scoped to another prefix. Remove
// the scoping and this test fails; the equivalence test alone would not.
func TestFSObjectStoreListIsScoped(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("needs a directory the process cannot read")
	}
	ctx := context.Background()
	root := t.TempDir()
	fsStore, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := fsStore.Put(ctx, "acme/col/2024-01-01T00:00:00Z.snap", strings.NewReader("a"), 1); err != nil {
		t.Fatal(err)
	}
	locked := filepath.Join(root, "beta", "locked")
	if err := os.MkdirAll(locked, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o750) })

	if _, err := fsStore.List(ctx, ""); err == nil {
		t.Fatal("root walk must fail on the unreadable directory (fixture check)")
	}
	infos, err := fsStore.List(ctx, "acme/col/")
	if err != nil {
		t.Fatalf("scoped walk visited an unrelated subtree: %v", err)
	}
	if len(infos) != 1 {
		t.Fatalf("scoped walk = %d keys, want 1", len(infos))
	}
}

// TestFSObjectStoreListMissingRootIsError pins that a root that has gone away
// (an unmounted backup volume) is an error from List — for any prefix — and not
// an empty result that LatestKey would turn into "no snapshots".
func TestFSObjectStoreListMissingRootIsError(t *testing.T) {
	ctx := context.Background()
	root := filepath.Join(t.TempDir(), "backups")
	fsStore, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(root); err != nil {
		t.Fatal(err)
	}
	for _, prefix := range []string{"", "acme/col/"} {
		if _, err := fsStore.List(ctx, prefix); err == nil {
			t.Errorf("list %q with a missing root: want an error, got nil", prefix)
		}
	}
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
