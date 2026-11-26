// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// These tests pin the replay rule for a log that cannot be read to its end: a
// half-written tail (nothing complete follows the bad bytes) is cut back before
// anything is appended, so later writes stay replayable; anything else refuses
// the open with ErrLogDamaged and leaves the file untouched.

const tailCol = "m"

func tailDenseCfg() Config {
	return Config{
		Dim: 4, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1,
		Quant: QuantNone, WAL: true,
	}
}

func tailVec(id uint64) []float32 { return []float32{float32(id), 1, 0, 0} }

// tailWALPath returns the log path of collection "m" for the given extension
// (".wal" dense, ".nwal" named, ".mvwal" multi-vector).
func tailWALPath(dir, ext string) string {
	return filepath.Join(dir, "vectors", "default", tailCol+ext)
}

// denseFixture creates a dense WAL collection, upserts ids 1, 2, 3, closes the
// store without a flush and returns the log path.
func denseFixture(t *testing.T) (dir, walPath string) {
	t.Helper()
	dir = t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection(tailCol, tailDenseCfg()); err != nil {
		t.Fatal(err)
	}
	c, _ := cs.Get(tailCol)
	for id := uint64(1); id <= 3; id++ {
		if err := c.Upsert(id, tailVec(id), "", 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	return dir, tailWALPath(dir, ".wal")
}

func denseLiveIDs(t *testing.T, cs *CollectionStore) []uint64 {
	t.Helper()
	c, ok := cs.Get(tailCol)
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	var ids []uint64
	for id := uint64(1); id <= 5; id++ {
		if _, _, _, _, _, ok := c.Get(id); ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func tailRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path) //nolint:gosec // test temp path
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func tailWrite(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func tailCut(t *testing.T, path string, n int) {
	t.Helper()
	b := tailRead(t, path)
	if len(b) <= n {
		t.Fatalf("log is %d bytes, cannot cut %d", len(b), n)
	}
	tailWrite(t, path, b[:len(b)-n])
}

func tailFlip(t *testing.T, path string, off int) {
	t.Helper()
	b := tailRead(t, path)
	b[off] ^= 0xFF
	tailWrite(t, path, b)
}

func tailAppend(t *testing.T, path string, extra []byte) {
	t.Helper()
	tailWrite(t, path, append(tailRead(t, path), extra...))
}

func tailTornFiles(t *testing.T, walPath string) []string {
	t.Helper()
	m, err := filepath.Glob(walPath + ".torn-*")
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// assertOneNonEmptyTornFile checks the cut bytes were saved next to the log.
func assertOneNonEmptyTornFile(t *testing.T, walPath string) {
	t.Helper()
	torn := tailTornFiles(t, walPath)
	if len(torn) != 1 {
		t.Fatalf("torn files = %v, want exactly one", torn)
	}
	if len(tailRead(t, torn[0])) == 0 {
		t.Fatalf("torn file %s is empty", torn[0])
	}
}

// assertDamaged checks the open failed with ErrLogDamaged naming the log and the
// offset, and that nothing was modified or saved.
func assertDamaged(t *testing.T, err error, walPath string, before []byte, offset string) {
	t.Helper()
	if !errors.Is(err, ErrLogDamaged) {
		t.Fatalf("open err = %v, want ErrLogDamaged", err)
	}
	if !strings.Contains(err.Error(), walPath) {
		t.Errorf("err %q does not name the log %s", err, walPath)
	}
	if offset != "" && !strings.Contains(err.Error(), offset) {
		t.Errorf("err %q does not name %q", err, offset)
	}
	if !bytes.Equal(tailRead(t, walPath), before) {
		t.Error("damaged log was modified")
	}
	if torn := tailTornFiles(t, walPath); len(torn) != 0 {
		t.Errorf("damaged log produced torn files %v", torn)
	}
}

// tailFrame builds one correctly checksummed log frame around payload.
func tailFrame(payload []byte) []byte {
	out := make([]byte, 8, 8+len(payload))
	binary.BigEndian.PutUint32(out[0:4], uint32(len(payload))) //nolint:gosec // test payloads are tiny
	binary.BigEndian.PutUint32(out[4:8], crc32.ChecksumIEEE(payload))
	return append(out, payload...)
}

func TestWALTornTailCutThenAppendSurvives(t *testing.T) {
	dir, walPath := denseFixture(t)
	tailCut(t, walPath, 7)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("open with torn tail: %v", err)
	}
	c, _ := cs.Get(tailCol)
	if err := c.Upsert(4, tailVec(4), "", 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	if got, want := denseLiveIDs(t, cs2), []uint64{1, 2, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("live ids = %v, want %v", got, want)
	}
	assertOneNonEmptyTornFile(t, walPath)
}

func TestWALTrailingZerosAreATornTail(t *testing.T) {
	dir, walPath := denseFixture(t)
	size := len(tailRead(t, walPath))
	tailAppend(t, walPath, make([]byte, 4096))

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("open with trailing zeros: %v", err)
	}
	defer func() { _ = cs.Close() }()
	if got, want := denseLiveIDs(t, cs), []uint64{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("live ids = %v, want %v", got, want)
	}
	if got := len(tailRead(t, walPath)); got != size {
		t.Fatalf("log size = %d, want %d (zeros cut)", got, size)
	}
}

func TestWALDamageBeforeGoodRecordsRefusesOpen(t *testing.T) {
	dir, walPath := denseFixture(t)
	tailFlip(t, walPath, 8)
	before := tailRead(t, walPath)

	_, err := OpenCollectionStore(dir)
	assertDamaged(t, err, walPath, before, "offset 0")
}

func TestWALDamagedLengthBeforeGoodRecordsRefusesOpen(t *testing.T) {
	dir, walPath := denseFixture(t)
	b := tailRead(t, walPath)
	binary.BigEndian.PutUint32(b[0:4], 0x00FFFFFF)
	tailWrite(t, walPath, b)

	_, err := OpenCollectionStore(dir)
	assertDamaged(t, err, walPath, b, "offset 0")
}

func TestWALUndecodableRecordRefusesOpen(t *testing.T) {
	dir, walPath := denseFixture(t)
	tailAppend(t, walPath, tailFrame([]byte{byte(walInsert)}))
	before := tailRead(t, walPath)

	_, err := OpenCollectionStore(dir)
	assertDamaged(t, err, walPath, before, "")
}

// namedFixture creates a named WAL collection, inserts ids 1, 2, 3 and closes the
// store without a flush.
func namedFixture(t *testing.T) (dir, walPath string) {
	t.Helper()
	dir = t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection(tailCol, namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, _ := cs.GetNamed(tailCol)
	for id := uint64(1); id <= 3; id++ {
		if err := nc.Insert(id, map[string][]float32{"title": tailVec(id)}, nil, 0); err != nil {
			t.Fatal(err)
		}
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	return dir, tailWALPath(dir, ".nwal")
}

func namedLiveIDs(t *testing.T, cs *CollectionStore) []uint64 {
	t.Helper()
	nc, ok := cs.GetNamed(tailCol)
	if !ok {
		t.Fatal("named collection missing after reopen")
	}
	var ids []uint64
	for id := uint64(1); id <= 5; id++ {
		if _, _, _, _, ok := nc.Get(id); ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func TestNamedWALTornTailCutThenAppendSurvives(t *testing.T) {
	dir, walPath := namedFixture(t)
	tailCut(t, walPath, 7)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("open with torn tail: %v", err)
	}
	nc, _ := cs.GetNamed(tailCol)
	if err := nc.Insert(4, map[string][]float32{"title": tailVec(4)}, nil, 0); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	if got, want := namedLiveIDs(t, cs2), []uint64{1, 2, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("live ids = %v, want %v", got, want)
	}
	assertOneNonEmptyTornFile(t, walPath)
}

func TestNamedWALDamageRefusesOpen(t *testing.T) {
	dir, walPath := namedFixture(t)
	tailFlip(t, walPath, 8)
	before := tailRead(t, walPath)

	_, err := OpenCollectionStore(dir)
	assertDamaged(t, err, walPath, before, "offset 0")
}

// mvFixture creates a multi-vector WAL collection, adds ids 1, 2, 3 and closes
// the store without a flush.
func mvFixture(t *testing.T) (dir, walPath string) {
	t.Helper()
	dir = t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector(tailCol, mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, _ := cs.GetMultiVector(tailCol)
	for id := uint64(1); id <= 3; id++ {
		if err := idx.Add(id, [][]float32{tailVec(id)}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}
	return dir, tailWALPath(dir, ".mvwal")
}

func mvLiveIDs(t *testing.T, cs *CollectionStore) []uint64 {
	t.Helper()
	idx, ok := cs.GetMultiVector(tailCol)
	if !ok {
		t.Fatal("multi-vector collection missing after reopen")
	}
	var ids []uint64
	for id := uint64(1); id <= 5; id++ {
		if _, _, _, ok := idx.Get(id); ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func TestMVWALTornTailCutThenAppendSurvives(t *testing.T) {
	dir, walPath := mvFixture(t)
	tailCut(t, walPath, 7)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("open with torn tail: %v", err)
	}
	idx, _ := cs.GetMultiVector(tailCol)
	if err := idx.Add(4, [][]float32{tailVec(4)}, nil); err != nil {
		t.Fatal(err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	if got, want := mvLiveIDs(t, cs2), []uint64{1, 2, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("live ids = %v, want %v", got, want)
	}
	assertOneNonEmptyTornFile(t, walPath)
}

func TestMVWALDamageRefusesOpen(t *testing.T) {
	dir, walPath := mvFixture(t)
	tailFlip(t, walPath, 8)
	before := tailRead(t, walPath)

	_, err := OpenCollectionStore(dir)
	assertDamaged(t, err, walPath, before, "offset 0")
}

// TestLaterFrameExistsAcrossWindows drives the bounded-window scan directly: a
// frame after a zero run longer than one window, a frame whose header straddles a
// window boundary, a frame whose payload runs past its window, and garbage with
// no frame at all.
func TestLaterFrameExistsAcrossWindows(t *testing.T) {
	good := tailFrame([]byte("a complete later record"))
	big := tailFrame(bytes.Repeat([]byte{7}, 4096))
	cases := []struct {
		name string
		data []byte
		want bool
	}{
		{"only zeros", make([]byte, 3*tornScanWindow), false},
		{"frame after long zeros", append(make([]byte, 2*tornScanWindow+3), good...), true},
		{"header straddles window", append(make([]byte, tornScanWindow-3), good...), true},
		{"payload past window", append(make([]byte, tornScanWindow-100), big...), true},
		{"bad crc only", func() []byte {
			b := append(make([]byte, 10), good...)
			b[len(b)-1] ^= 0xFF
			return b
		}(), false},
		{"length overruns file", good[:len(good)-1], false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "log")
			tailWrite(t, path, tc.data)
			f, err := os.Open(path) //nolint:gosec // test temp path
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = f.Close() }()
			got, err := laterFrameExists(f, 0, int64(len(tc.data)))
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("laterFrameExists = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestWALFailedWriteNeverAcksBehindPartialFrame injects a write that lands only
// part of a frame (ENOSPC after the header, mid-header, or mid-payload). The
// failed append must not be acknowledged, no later append may be acknowledged
// behind the partial bytes, and reopening must succeed with every acknowledged
// record. The "cut fails" case closes the file inside the failed write, so the
// partial bytes stay on disk and the next open must treat them as a torn tail.
func TestWALFailedWriteNeverAcksBehindPartialFrame(t *testing.T) {
	recA := []byte{byte(walDelete), 0, 0, 0, 0, 0, 0, 0, 1}
	recB := []byte{byte(walDelete), 0, 0, 0, 0, 0, 0, 0, 2}
	recC := []byte{byte(walDelete), 0, 0, 0, 0, 0, 0, 0, 3}
	cases := []struct {
		name     string
		keep     int // bytes of the failing frame that reach the file
		closeFil bool
	}{
		{"mid header", 3, false},
		{"header only", 8, false},
		{"mid payload", 12, false},
		{"mid payload, cut fails", 12, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "x.wal")
			w, err := openWAL(path, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := w.appendFramed(recA); err != nil {
				t.Fatalf("append A: %v", err)
			}

			left := tc.keep // bytes the failing frame may still write
			w.writeHook = func(f *os.File, p []byte) (int, error) {
				n := len(p)
				if n > left {
					n = left
				}
				left -= n
				m, err := f.Write(p[:n])
				if err != nil {
					return m, err
				}
				if m < len(p) {
					if tc.closeFil {
						_ = f.Close()
					}
					return m, errors.New("injected: no space left on device")
				}
				return m, nil
			}
			seqB, err := w.appendFramedStaged(recB)
			if err == nil {
				if cerr := w.commitWaitStaged(seqB); cerr == nil {
					t.Fatal("append B acknowledged despite a failed write")
				}
			}
			w.writeHook = nil

			seqC, err := w.appendFramedStaged(recC)
			if err == nil {
				if cerr := w.commitWaitStaged(seqC); cerr == nil {
					t.Fatal("append C acknowledged behind a failed write")
				}
			}
			if err := w.appendFramed(recC); err == nil {
				t.Fatal("blocking append acknowledged behind a failed write")
			}
			_ = w.close()

			var got [][]byte
			if err := replayFramed(path, func(rec []byte) error {
				got = append(got, append([]byte(nil), rec...))
				return nil
			}); err != nil {
				t.Fatalf("replay after failed write: %v", err)
			}
			if len(got) != 1 || !bytes.Equal(got[0], recA) {
				t.Fatalf("replayed %v, want only record A", got)
			}

			w2, err := openWAL(path, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := w2.appendFramed(recC); err != nil {
				t.Fatalf("append after reopen: %v", err)
			}
			_ = w2.close()
			got = nil
			if err := replayFramed(path, func(rec []byte) error {
				got = append(got, append([]byte(nil), rec...))
				return nil
			}); err != nil {
				t.Fatalf("second replay: %v", err)
			}
			if len(got) != 2 || !bytes.Equal(got[0], recA) || !bytes.Equal(got[1], recC) {
				t.Fatalf("replayed %v, want A then C", got)
			}
		})
	}
}
