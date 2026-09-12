// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"io"
	"os"
	"path/filepath"
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
