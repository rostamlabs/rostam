// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// onDiskNames returns every name under root, relative and slash-separated.
func onDiskNames(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, _ fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p == root {
			return nil
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return rerr
		}
		out = append(out, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		t.Fatalf("walk %q: %v", root, err)
	}
	return out
}

func listKeysOf(t *testing.T, obj objstore.ObjectStore, prefix string) []string {
	t.Helper()
	infos, err := obj.List(context.Background(), prefix)
	if err != nil {
		t.Fatalf("list %q: %v", prefix, err)
	}
	out := make([]string, 0, len(infos))
	for _, in := range infos {
		out = append(out, in.Key)
	}
	return out
}

// writeLiteral places content at key's LITERAL on-disk path, the way a build
// from before the colon escape wrote every object.
func writeLiteral(t *testing.T, root, key, content string) string {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(key))
	if err := os.MkdirAll(filepath.Dir(p), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestFSObjectStoreColonKeyRoundTrip pins the key-to-name mapping: no object
// name the store creates contains a ':' (illegal in a Windows file name), and
// every key comes back from List exactly as it was written — including keys
// whose PathEscape'd collection segment already carries %XX sequences, a '%'
// sitting right before an escaped ':', and a ':' in a directory segment. It runs
// on every platform, Windows included.
func TestFSObjectStoreColonKeyRoundTrip(t *testing.T) {
	ctx := context.Background()
	ts := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	type objKeys struct{ prefix, snap string }
	var cases []objKeys
	for _, tenant := range []string{"acme", ""} {
		for _, col := range []string{
			"default/c",       // "/" -> %2F
			"default/100%",    // '%' -> %25
			"default/a%3Ab",   // a literal "%3A" in the name -> %253A
			"default/x:y",     // ':' in the directory segment
			"default/%2F:%3A", // escapes and ':' together
			"default/%:",      // '%' immediately before ':'
		} {
			cases = append(cases, objKeys{collectionKeyPrefix(tenant, col), snapshotKey(tenant, col, ts)})
		}
	}
	for _, raw := range []string{"a:b/c:d/e:f", "t/:", "t/::", "t/%:3A", "t/%3:", "t/%:A", "t/3A:%"} {
		cases = append(cases, objKeys{raw[:strings.LastIndex(raw, "/")+1], raw})
	}

	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		cfg := cfgKeyFor(c.snap)
		if err := putIfAbsentString(ctx, store, cfg, "cfg:"+c.snap); err != nil {
			t.Fatalf("PutIfAbsent %q: %v", cfg, err)
		}
		if err := store.Put(ctx, c.snap, strings.NewReader("snap:"+c.snap), 0); err != nil {
			t.Fatalf("Put %q: %v", c.snap, err)
		}
	}
	for _, name := range onDiskNames(t, root) {
		if strings.ContainsRune(name, ':') {
			t.Errorf("on-disk name %q contains ':'", name)
		}
	}
	for _, c := range cases {
		cfg := cfgKeyFor(c.snap)
		if got := mustGet(t, store, c.snap); got != "snap:"+c.snap {
			t.Errorf("Get %q = %q", c.snap, got)
		}
		if got := mustGet(t, store, cfg); got != "cfg:"+c.snap {
			t.Errorf("Get %q = %q", cfg, got)
		}
		// Each key exactly once, spelled as written. The prefix list may also hold
		// other cases' keys where prefixes nest, so filter to this stem.
		var got []string
		for _, k := range listKeysOf(t, store, c.prefix) {
			if k == c.snap || k == cfg {
				got = append(got, k)
			}
		}
		want := []string{c.snap, cfg}
		sortStrings(want)
		if strings.Join(got, "\n") != strings.Join(want, "\n") {
			t.Errorf("List %q = %q, want %q", c.prefix, got, want)
		}
		// Listing the exact key: longer keys it prefixes may follow, but the key
		// itself appears once.
		if got := listKeysOf(t, store, c.snap); len(got) == 0 || got[0] != c.snap || (len(got) > 1 && got[1] == c.snap) {
			t.Errorf("List of the exact key %q = %q", c.snap, got)
		}
		// A partial prefix ending right after a ':' still matches.
		if i := strings.LastIndex(c.snap, ":"); i >= 0 {
			if got := listKeysOf(t, store, c.snap[:i+1]); len(got) == 0 {
				t.Errorf("List of the partial prefix %q found nothing", c.snap[:i+1])
			}
		}
	}
	for _, c := range cases {
		if err := store.Delete(ctx, c.snap); err != nil {
			t.Fatalf("Delete %q: %v", c.snap, err)
		}
		if _, err := store.Get(ctx, c.snap); !errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Get after Delete %q = %v, want ErrNotFound", c.snap, err)
		}
	}
}

// TestFSObjectStoreColonEscapeIsReserved pins why the mapping is reversible: a
// key that already contains the escape sequence would come back from List as a
// different key, and on a case-insensitive filesystem its lower-case spelling
// would name the same file as an escaped ':'. Such keys are refused by every
// operation, as invalid keys rather than as missing ones.
func TestFSObjectStoreColonEscapeIsReserved(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"t/c/%3A", "t/c/x%3ay", "t%3A/c/k.snap"} {
		if err := store.Put(ctx, key, strings.NewReader("v"), 1); err == nil {
			t.Errorf("Put(%q): want the reserved escape refused", key)
		}
		if err := putIfAbsentString(ctx, store, key, "v"); err == nil || errors.Is(err, objstore.ErrExists) {
			t.Errorf("PutIfAbsent(%q) = %v, want an invalid-key error", key, err)
		}
		if _, err := store.Get(ctx, key); err == nil || errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Get(%q) = %v, want an invalid-key error", key, err)
		}
		if err := store.Delete(ctx, key); err == nil || errors.Is(err, objstore.ErrNotFound) {
			t.Errorf("Delete(%q) = %v, want an invalid-key error", key, err)
		}
	}
	if names := onDiskNames(t, root); len(names) != 0 {
		t.Errorf("refused keys wrote %v", names)
	}
}

// TestFSObjectStoreColonLegacyLiteralNames seeds a backup directory the way a
// build from before the escape left it — every ':' literal in the file names —
// and checks it stays fully usable: readable, listed once per key, restorable,
// and pruned correctly by retention when a new backup lands beside it.
func TestFSObjectStoreColonLegacyLiteralNames(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a ':' cannot appear in a Windows file name, so no such directory exists there")
	}
	ctx := context.Background()
	store := newStore(t)
	mustCreate(t, store, "docs", 1, 2, 3)

	// Produce real backup objects, then lay them out under literal names.
	mem := objstore.NewMemStore()
	stamps := []time.Time{
		time.Date(2026, 6, 23, 1, 0, 0, 0, time.UTC),
		time.Date(2026, 6, 23, 2, 0, 0, 0, time.UTC),
		time.Date(2026, 6, 23, 3, 0, 0, 0, time.UTC),
	}
	for _, ts := range stamps {
		if _, err := Backup(ctx, store, mem, BackupOpts{Tenant: "acme", Timestamp: ts}); err != nil {
			t.Fatal(err)
		}
	}
	root := t.TempDir()
	prefix := collectionKeyPrefix("acme", "default/docs")
	memKeys := listKeysOf(t, mem, prefix)
	for _, k := range memKeys {
		writeLiteral(t, root, k, mustGet(t, mem, k))
	}
	fsStore, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}

	if got := listKeysOf(t, fsStore, prefix); strings.Join(got, "\n") != strings.Join(memKeys, "\n") {
		t.Fatalf("List over literal names = %q, want %q", got, memKeys)
	}
	for _, k := range memKeys {
		if got, want := mustGet(t, fsStore, k), mustGet(t, mem, k); got != want {
			t.Fatalf("Get %q over its literal name differs from what was stored", k)
		}
	}
	latest, err := LatestKey(ctx, fsStore, "acme", "default/docs")
	if err != nil {
		t.Fatal(err)
	}
	if want := snapshotKey("acme", "default/docs", stamps[2]); latest != want {
		t.Fatalf("LatestKey = %q, want %q", latest, want)
	}
	target := newStore(t)
	if err := RestoreLatest(ctx, target, fsStore, "acme", "default/docs"); err != nil {
		t.Fatalf("RestoreLatest from literal names: %v", err)
	}
	c, ok := target.Acquire("default/docs")
	if !ok {
		t.Fatal("restored collection is missing")
	}
	c.Release()

	// A new backup lands under escaped names; retention must rank it among the
	// literal ones by time and delete the two oldest literal snapshots and their
	// configs.
	newTS := time.Date(2026, 6, 23, 4, 0, 0, 0, time.UTC)
	res, err := Backup(ctx, store, fsStore, BackupOpts{Tenant: "acme", Timestamp: newTS, Retention: 2})
	if err != nil {
		t.Fatalf("backup beside literal names: %v", err)
	}
	var wantSnaps []string
	for _, ts := range []time.Time{stamps[2], newTS} {
		k := snapshotKey("acme", "default/docs", ts)
		wantSnaps = append(wantSnaps, k, cfgKeyFor(k))
	}
	sortStrings(wantSnaps)
	if got := listKeysOf(t, fsStore, prefix); strings.Join(got, "\n") != strings.Join(wantSnaps, "\n") {
		t.Fatalf("after retention List = %q, want %q", got, wantSnaps)
	}
	if res[0].Key != snapshotKey("acme", "default/docs", newTS) {
		t.Fatalf("new backup key = %q", res[0].Key)
	}
	for _, ts := range stamps[:2] {
		k := snapshotKey("acme", "default/docs", ts)
		for _, kk := range []string{k, cfgKeyFor(k)} {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(kk))); !os.IsNotExist(err) {
				t.Errorf("pruned literal file %q still on disk (stat err = %v)", kk, err)
			}
		}
	}
}

// TestFSObjectStoreColonBothFormsListedOnce covers a key present under both its
// literal and its escaped name (a key a pre-escape build wrote and a later Put
// overwrote): List reports it once, reads return the escaped — newer — copy, and
// Delete removes both, so the old bytes cannot resurface.
func TestFSObjectStoreColonBothFormsListedOnce(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a ':' cannot appear in a Windows file name")
	}
	ctx := context.Background()
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	const key = "t/c:d/2024-01-01T00:00:00Z.snap"
	lit := writeLiteral(t, root, key, "old")
	if err := store.Put(ctx, key, strings.NewReader("newer"), 5); err != nil {
		t.Fatal(err)
	}
	for _, prefix := range []string{"", "t/", "t/c:d/", key} {
		infos, err := store.List(ctx, prefix)
		if err != nil {
			t.Fatal(err)
		}
		if len(infos) != 1 || infos[0].Key != key || infos[0].Size != 5 {
			t.Errorf("List %q = %+v, want %q once with the escaped copy's size", prefix, infos, key)
		}
	}
	if got := mustGet(t, store, key); got != "newer" {
		t.Errorf("Get = %q, want the escaped copy", got)
	}
	if err := store.Delete(ctx, key); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(lit); !os.IsNotExist(err) {
		t.Errorf("Delete left the literal copy behind (stat err = %v)", err)
	}
	if _, err := store.Get(ctx, key); !errors.Is(err, objstore.ErrNotFound) {
		t.Errorf("Get after Delete = %v, want ErrNotFound", err)
	}
	if err := store.Delete(ctx, key); !errors.Is(err, objstore.ErrNotFound) {
		t.Errorf("second Delete = %v, want ErrNotFound", err)
	}

	// Delete of a key present ONLY under its literal name removes it.
	writeLiteral(t, root, key, "old")
	if err := store.Delete(ctx, key); err != nil {
		t.Fatalf("Delete of a literal-only key: %v", err)
	}
	if _, err := os.Stat(lit); !os.IsNotExist(err) {
		t.Errorf("literal-only key survived Delete (stat err = %v)", err)
	}
}

// TestFSObjectStoreColonPutIfAbsentSeesLiteralName pins write-once across the two
// on-disk spellings. A key that exists under its literal name exists: publishing
// the escaped name beside it would "succeed" and leave two objects for one key.
func TestFSObjectStoreColonPutIfAbsentSeesLiteralName(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a ':' cannot appear in a Windows file name")
	}
	ctx := context.Background()
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"t/c/2024-01-01T00:00:00Z.cfg.json", "t/c:d/2024-01-01T00:00:00Z.snap"} {
		writeLiteral(t, root, key, "original")
		if err := putIfAbsentString(ctx, store, key, "duplicate"); !errors.Is(err, objstore.ErrExists) {
			t.Fatalf("PutIfAbsent over a literal-name key %q = %v, want ErrExists", key, err)
		}
		if got := mustGet(t, store, key); got != "original" {
			t.Fatalf("Get %q = %q, want the original", key, got)
		}
		if got := listKeysOf(t, store, key); len(got) != 1 {
			t.Fatalf("List %q = %q, want one key", key, got)
		}
	}
	for _, name := range onDiskNames(t, root) {
		if strings.Contains(name, "%3A") {
			if info, err := os.Stat(filepath.Join(root, filepath.FromSlash(name))); err == nil && !info.IsDir() {
				t.Errorf("PutIfAbsent published an escaped duplicate %q", name)
			}
		}
		if strings.Contains(filepath.Base(name), putTempPrefix) {
			t.Errorf("staging file left behind: %q", name)
		}
	}

	// The same through a backup: a pre-escape config-less snapshot holds the key,
	// so the run is refused and releases its claimed config.
	vs := newStore(t)
	mustCreate(t, vs, "c", 1, 2)
	opts := BackupOpts{Tenant: "acme", Timestamp: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)}
	snap := snapshotKey(opts.Tenant, "default/c", opts.Timestamp)
	writeLiteral(t, root, snap, "config-less legacy snapshot")
	if _, err := Backup(ctx, vs, store, opts); !errors.Is(err, ErrSnapshotExists) {
		t.Fatalf("Backup over a literal-name snapshot = %v, want ErrSnapshotExists", err)
	}
	if got := listKeysOf(t, store, collectionKeyPrefix(opts.Tenant, "default/c")); len(got) != 1 || got[0] != snap {
		t.Fatalf("after the refused backup List = %q, want only the legacy snapshot", got)
	}
	if got := mustGet(t, store, snap); got != "config-less legacy snapshot" {
		t.Fatalf("legacy snapshot became %q", got)
	}
}

// TestFSObjectStoreColonReclaimsLiteralDirTemps checks the stale-staging sweep
// still runs in a directory whose name carries a literal ':' from a pre-escape
// build, even though new staging files go to the escaped directory.
func TestFSObjectStoreColonReclaimsLiteralDirTemps(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a ':' cannot appear in a Windows file name")
	}
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	stale := writeLiteral(t, root, "t/c:d/"+putTempPrefix+"stale"+putTempSuffix, "abandoned")
	old := time.Now().Add(-2 * staleTempAge)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatal(err)
	}
	if err := store.Put(context.Background(), "t/c:d/2024-01-01T00:00:00Z.snap", strings.NewReader("v"), 1); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("stale staging file in the literal directory not reclaimed (stat err = %v)", err)
	}
}
