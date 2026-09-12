// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
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

// TestFSObjectStorePutReclaimsStaleTemps verifies Put's best-effort reclamation
// of staging files abandoned by a process killed mid-Put. Such a leftover is not
// inert — the same Put stages the snapshot itself, so it can be a partial
// snapshot of arbitrary size — and nothing else ever removes it. A temp older
// than staleTempAge is swept; a temp with a CURRENT modtime is left alone,
// because it may belong to a concurrent writer.
func TestFSObjectStorePutReclaimsStaleTemps(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}

	dir := filepath.Join(root, "tenant", "coll")
	if err := os.MkdirAll(dir, 0o750); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	stale := filepath.Join(dir, putTempPrefix+"stale.tmp")
	if err := os.WriteFile(stale, []byte("abandoned partial snapshot"), 0o600); err != nil {
		t.Fatalf("write stale temp: %v", err)
	}
	old := time.Now().Add(-2 * staleTempAge)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	fresh := filepath.Join(dir, putTempPrefix+"fresh.tmp")
	if err := os.WriteFile(fresh, []byte("a concurrent writer's staging file"), 0o600); err != nil {
		t.Fatalf("write fresh temp: %v", err)
	}

	const key = "tenant/coll/2026-07-08T00:00:00Z.snap"
	want := []byte("payload")
	if err := store.Put(context.Background(), key, strings.NewReader(string(want)), int64(len(want))); err != nil {
		t.Fatalf("Put: %v", err)
	}

	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("stale staging file not reclaimed, stat err = %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Errorf("a current-modtime staging file must be left alone: %v", err)
	}
	rc, err := store.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("Get after Put: %v", err)
	}
	defer rc.Close()
	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("object = %q, want %q", got, want)
	}
}

// TestFSObjectStoreRejectsReservedStagingKey asserts the putTempPrefix name
// space is reserved. Nothing stops a caller from asking to publish an object
// named like a staging file, and such an object would be silently deleted by
// reclaimStaleTemps an hour later — so keyToPath refuses the key outright. The
// refusal lives in keyToPath, so it covers Get and Delete as well as Put: a key
// that can never be written must not be readable or deletable either.
func TestFSObjectStoreRejectsReservedStagingKey(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}
	ctx := context.Background()

	for _, key := range []string{
		"tenant/coll/" + putTempPrefix + "sneaky" + putTempSuffix,
		"tenant/coll/" + putTempPrefix + "sneaky.snap",
		putTempPrefix + "toplevel",
	} {
		if err := store.Put(ctx, key, strings.NewReader("payload"), 7); err == nil {
			t.Errorf("Put(%q): expected the reserved name space to be refused, got nil", key)
		}
		if _, err := store.Get(ctx, key); err == nil {
			t.Errorf("Get(%q): expected the reserved name space to be refused, got nil", key)
		} else if errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Get(%q): refused as not-found, not as an invalid key: %v", key, err)
		}
		if err := store.Delete(ctx, key); err == nil {
			t.Errorf("Delete(%q): expected the reserved name space to be refused, got nil", key)
		} else if errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Delete(%q): refused as not-found, not as an invalid key: %v", key, err)
		}
	}

	// The refused Put must not have written anything — not even the directory
	// tree, since keyToPath fails before MkdirAll.
	if _, err := os.Stat(filepath.Join(root, "tenant")); !os.IsNotExist(err) {
		t.Errorf("refused Put created files under root, stat err = %v", err)
	}

	// A key whose non-final segment shares the prefix is fine: the sweep skips
	// directories and only ever inspects one directory's own entries.
	ok := putTempPrefix + "dir/real.snap"
	if err := store.Put(ctx, ok, strings.NewReader("payload"), 7); err != nil {
		t.Errorf("Put(%q): a prefixed DIRECTORY segment must stay legal: %v", ok, err)
	}
}

// TestFSObjectStoreSweepRequiresTempSuffix asserts the sweep matches on BOTH
// halves of the staging name, not the prefix alone. A file carrying the prefix
// but not putTempSuffix was never created by this Put (keyToPath reserves the
// prefix, so it cannot be a published object either) — it is something else on
// the volume, and the reclaimer must leave it alone however old it is.
func TestFSObjectStoreSweepRequiresTempSuffix(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}

	dir := filepath.Join(root, "tenant", "coll")
	if err := os.MkdirAll(dir, 0o750); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// Both are ancient; only the one with the full staging name may be swept.
	suffixed := filepath.Join(dir, putTempPrefix+"abandoned"+putTempSuffix)
	bare := filepath.Join(dir, putTempPrefix+"not-a-staging-file")
	old := time.Now().Add(-2 * staleTempAge)
	for _, p := range []string{suffixed, bare} {
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatalf("write %q: %v", p, err)
		}
		if err := os.Chtimes(p, old, old); err != nil {
			t.Fatalf("chtimes %q: %v", p, err)
		}
	}

	const key = "tenant/coll/2026-07-08T00:00:00Z.snap"
	if err := store.Put(context.Background(), key, strings.NewReader("payload"), 7); err != nil {
		t.Fatalf("Put: %v", err)
	}

	if _, err := os.Stat(suffixed); !os.IsNotExist(err) {
		t.Errorf("a full staging name should have been reclaimed, stat err = %v", err)
	}
	if _, err := os.Stat(bare); err != nil {
		t.Errorf("a file without %q must be left alone: %v", putTempSuffix, err)
	}
}

// TestFSObjectStoreRejectsBackslashKeys pins the cross-platform hole in the
// staging-name reservation. A backslash is an ordinary character in an object key
// and to path.Base, but a SEPARATOR to filepath on Windows — so
// "coll\.rostam-put-x.tmp" has a final segment of "coll\.rostam-put-x.tmp" by
// the key's own rules and ".rostam-put-x.tmp" by the filesystem's, which let it
// publish into the reserved name space and be swept an hour later. Keys carrying
// the character are refused outright, on every platform, so one key always means
// one path.
func TestFSObjectStoreRejectsBackslashKeys(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}
	for _, key := range []string{
		`tenant/coll\` + putTempPrefix + `sneaky` + putTempSuffix, // the bypass itself
		`tenant\coll/ts.snap`, // a plain separator swap
	} {
		if err := store.Put(context.Background(), key, strings.NewReader("x"), 1); err == nil {
			t.Errorf("Put(%q): expected a backslash key to be refused, got nil", key)
		}
		if _, err := store.Get(context.Background(), key); err == nil {
			t.Errorf("Get(%q): expected a backslash key to be refused, got nil", key)
		} else if errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Get(%q): refused as not-found, not as an invalid key: %v", key, err)
		}
		// Delete too: keyToPath gates all three, and on Windows an unguarded
		// Delete of a backslash key resolves into the reserved staging name space
		// — where it could unlink a live Put's staging file.
		if err := store.Delete(context.Background(), key); err == nil {
			t.Errorf("Delete(%q): expected a backslash key to be refused, got nil", key)
		} else if errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Delete(%q): refused as not-found, not as an invalid key: %v", key, err)
		}
	}
	// Nothing may have been created: keyToPath fails before MkdirAll.
	if entries, rerr := os.ReadDir(root); rerr != nil || len(entries) != 0 {
		t.Errorf("refused keys created %d entries under root (err %v), want none", len(entries), rerr)
	}
}

// TestTempHeartbeatKeepsTempFresh asserts a running Put's staging file is kept
// young, and stops being kept young once the Put is done. Without the heartbeat
// a copy that stalls for longer than staleTempAge is indistinguishable from a
// temp orphaned by a killed process, so a concurrent Put would unlink it and the
// stalled Put's rename would fail. tempHeartbeat is shortened so the test is a
// few milliseconds rather than a quarter of an hour.
func TestTempHeartbeatKeepsTempFresh(t *testing.T) {
	prev := tempHeartbeat
	tempHeartbeat = 2 * time.Millisecond
	t.Cleanup(func() { tempHeartbeat = prev })

	path := filepath.Join(t.TempDir(), putTempPrefix+"inflight"+putTempSuffix)
	if err := os.WriteFile(path, []byte("in-flight snapshot"), 0o600); err != nil {
		t.Fatalf("write temp: %v", err)
	}
	backdate := func() {
		old := time.Now().Add(-2 * staleTempAge)
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatalf("chtimes: %v", err)
		}
	}
	// fresh reports whether the file would survive a sweep right now.
	fresh := func() bool {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat: %v", err)
		}
		return !info.ModTime().Before(time.Now().Add(-staleTempAge))
	}

	backdate()
	stop := startTempHeartbeat(path)
	deadline := time.Now().Add(2 * time.Second)
	for !fresh() {
		if time.Now().After(deadline) {
			stop()
			t.Fatal("heartbeat never refreshed the staging file's mtime")
		}
		time.Sleep(time.Millisecond)
	}

	// After stop the refreshes must cease, or the file could never age out and an
	// abandoned temp would leak forever.
	stop()
	backdate()
	time.Sleep(20 * tempHeartbeat)
	if fresh() {
		t.Error("mtime still being refreshed after stop")
	}
}
