// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// failingReader yields n bytes, then fails, as a copy hitting a full disk does.
type failingReader struct{ n int }

func (r *failingReader) Read(p []byte) (int, error) {
	if r.n == 0 {
		return 0, errors.New("injected: no space left on device")
	}
	k := min(len(p), r.n)
	for i := range p[:k] {
		p[i] = 'x'
	}
	r.n -= k
	return k, nil
}

// TestWALTornTailCopyRemovedWhenSaveFails: a copy of a torn tail that cannot
// be completed is removed, so repeated failed opens do not pile up partial
// copies on a disk that is already short of space.
func TestWALTornTailCopyRemovedWhenSaveFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.wal.torn-1")
	err := saveTornBytes(&failingReader{n: 4096}, path)
	if err == nil {
		t.Fatal("a failed copy reported success")
	}
	if _, serr := os.Stat(path); !os.IsNotExist(serr) {
		t.Fatalf("partial copy left at %s (stat err %v)", path, serr)
	}
	// A copy that completes is kept.
	if err := saveTornBytes(io.LimitReader(strings.NewReader("tail"), 4), path); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != "tail" { //nolint:gosec // test temp dir
		t.Fatalf("saved copy = %q", got)
	}
}
