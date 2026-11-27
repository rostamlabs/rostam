// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestWALTornTailScanCostIsBounded: every even offset of this tail reads as a
// 1 MiB length that fits in the file, so a scan that checksums each candidate
// payload in full does about 1.5 TB of work (minutes). Checksumming from
// running CRCs takes a few kilobytes per candidate (about a second).
func TestWALTornTailScanCostIsBounded(t *testing.T) {
	tail := make([]byte, 4<<20)
	for i := 0; i+1 < len(tail); i += 2 {
		tail[i], tail[i+1] = 0x00, 0x10
	}
	path := filepath.Join(t.TempDir(), "c.wal")
	if err := os.WriteFile(path, tail, 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path) //nolint:gosec // test temp dir
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	start := time.Now()
	found, err := laterFrameExists(f, 0, int64(len(tail)))
	if err != nil {
		t.Fatal(err)
	}
	if found {
		t.Fatal("found a frame in a tail that holds none")
	}
	if took := time.Since(start); took > time.Minute {
		t.Fatalf("scanning a %d-byte tail took %v", len(tail), took)
	}
}

// TestWALCRCZerosCombines: the CRC of a payload derived from running CRCs
// equals the CRC computed directly.
func TestWALCRCZerosCombines(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	data := make([]byte, 3*tailCRCStride+123)
	for i := range data {
		data[i] = byte(rng.Uint32())
	}
	for range 2000 {
		a := rng.IntN(len(data))
		b := a + rng.IntN(len(data)-a+1)
		want := crc32.ChecksumIEEE(data[a:b])
		got := crc32.ChecksumIEEE(data[:b]) ^ crcZeros(crc32.ChecksumIEEE(data[:a]), int64(b-a))
		if got != want {
			t.Fatalf("[%d,%d): derived %08x, direct %08x", a, b, got, want)
		}
	}
}

// bruteLaterFrame is the definition laterFrameExists implements: any offset
// whose length fits and whose payload's CRC matches.
func bruteLaterFrame(b []byte, from int) bool {
	for i := from; i+8 <= len(b); i++ {
		plen := int(binary.BigEndian.Uint32(b[i : i+4]))
		if plen == 0 || plen > maxWALRecord || plen > len(b)-i-8 {
			continue
		}
		if crc32.ChecksumIEEE(b[i+8:i+8+plen]) == binary.BigEndian.Uint32(b[i+4:i+8]) {
			return true
		}
	}
	return false
}

// TestWALLaterFrameMatchesBruteForce: on random tails, some holding a planted
// frame (long ones included, so the derived CRC is exercised), the scan agrees
// with checksumming every candidate in full.
func TestWALLaterFrameMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	dir := t.TempDir()
	for trial := range 60 {
		size := 64 + rng.IntN(5*tailCRCStride)
		b := make([]byte, size)
		for i := range b {
			// Small bytes make many plausible lengths.
			b[i] = byte(rng.IntN(4))
		}
		if trial%2 == 0 {
			plen := 1 + rng.IntN(size-9)
			at := rng.IntN(size - 8 - plen + 1)
			binary.BigEndian.PutUint32(b[at:], uint32(plen)) //nolint:gosec // test sizes
			binary.BigEndian.PutUint32(b[at+4:], crc32.ChecksumIEEE(b[at+8:at+8+plen]))
		}
		path := filepath.Join(dir, fmt.Sprintf("t%d.wal", trial))
		if err := os.WriteFile(path, b, 0o600); err != nil {
			t.Fatal(err)
		}
		f, err := os.Open(path) //nolint:gosec // test temp dir
		if err != nil {
			t.Fatal(err)
		}
		from := rng.IntN(8)
		got, err := laterFrameExists(f, int64(from), int64(size))
		_ = f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if want := bruteLaterFrame(b, from); got != want {
			t.Fatalf("trial %d: laterFrameExists = %v, brute force = %v", trial, got, want)
		}
	}
}
