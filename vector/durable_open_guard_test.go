// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"runtime"
	"strings"
	"testing"
)

// TestDurableRefusedOpenLeavesManifestUnchanged: a listed collection whose
// manifest names an unsynced log, with its earlier form's sidecar still there
// and no snapshot, fails the open with an error saying nothing was changed. The
// manifest used to be rewritten to a synced log before that refusal.
func TestDurableRefusedOpenLeavesManifestUnchanged(t *testing.T) {
	dir := t.TempDir()
	writeLegacyStore(t, dir)
	cfg := manifestOf(t, dir, upgradeCol)
	cfg.Persistent, cfg.WAL, cfg.WALNoSync = false, true, true
	if err := writeConfig(colFile(dir, ".json"), cfg); err != nil {
		t.Fatal(err)
	}
	_ = os.Remove(colFile(dir, ".snap"))
	before := fileHash(t, colFile(dir, ".json"))
	cs, err := OpenCollectionStoreWith(dir, StoreOptions{DurableCollections: []string{upgradeCol}})
	if err == nil {
		_ = cs.Close()
		t.Fatal("open succeeded with the snapshot missing beside the sidecar")
	}
	if !strings.Contains(err.Error(), "nothing was changed") {
		t.Fatalf("err = %v, want the refusal that changes nothing", err)
	}
	if after := fileHash(t, colFile(dir, ".json")); after != before {
		t.Fatal("a refused open rewrote the manifest")
	}
}

// TestDurableListedHeapCollectionWarns: a listed collection already on disk
// with neither Persistent nor a log keeps its mode, and the open says so
// instead of leaving it unprotected silently.
func TestDurableListedHeapCollectionWarns(t *testing.T) {
	dir := t.TempDir()
	cs := openDurable(t, dir)
	heap := durablePersistentCfg()
	heap.Persistent = false
	if err := cs.CreateCollection("docs", heap); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert("docs", 1, durableVec(1), "kept", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Flush("docs"); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	logs := captureLogs(t)
	re := openDurable(t, dir, "docs")
	defer func() { _ = re.Close() }()
	if !strings.Contains(logs.String(), "not crash-safe") {
		t.Fatalf("log = %q, want a not crash-safe warning for the listed collection", logs.String())
	}
	assertPoints(t, pointsOf(t, re, "docs"), map[uint64]string{1: "kept"})
}

// TestFailedStoreOpenClosesLoadedCollections: when one collection fails to
// load, the ones registered before it are closed, not left holding their files.
func TestFailedStoreOpenClosesLoadedCollections(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("counts open descriptors through /proc/self/fd")
	}
	dir := t.TempDir()
	cs := openDurable(t, dir, "a")
	if err := cs.CreateCollection("a", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if err := cs.Upsert("a", 1, durableVec(1), "x", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("z", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()
	// "z" loads after "a" and fails: its manifest is torn.
	if err := os.WriteFile(dir+"/vectors/"+DefaultTenant+"/z.json", []byte("{torn"), 0o600); err != nil {
		t.Fatal(err)
	}
	fds := func() int {
		ents, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Fatal(err)
		}
		return len(ents)
	}
	before := fds()
	for range 3 {
		if s, err := OpenCollectionStoreWith(dir, StoreOptions{DurableCollections: []string{"a"}}); err == nil {
			_ = s.Close()
			t.Fatal("open succeeded over a torn manifest")
		}
	}
	if after := fds(); after > before {
		t.Fatalf("open descriptors grew from %d to %d across failed opens", before, after)
	}
}
