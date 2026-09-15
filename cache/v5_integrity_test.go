// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// TestRecoveryResyncDiscardsOnlyTheDamagedEntry pins the core issue-#135 metric: a
// single corrupted entry mid-page costs exactly that entry's bytes, and the entries
// before AND after it survive. A v4 build truncated the whole page tail here.
func TestRecoveryResyncDiscardsOnlyTheDamagedEntry(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0
	cfg.DisableColdCompaction = true

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	offs := writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
		{[]byte("k1"), []byte("BBBB")},
		{[]byte("k2"), []byte("CCCC")},
	})

	// Flip one byte of entry 1's payload so its MAC fails at recovery.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte{0x00}, int64(headerSize+pageHdrSize+offs[1]+entryHeaderSize)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	if v, err := s.Get([]byte("k0")); err != nil || !bytes.Equal(v, []byte("AAAA")) {
		t.Errorf("Get(k0) = %q,%v; want AAAA (before the tear)", v, err)
	}
	if _, err := s.Get([]byte("k1")); err != ErrNotFound {
		t.Errorf("Get(k1) = %v; want ErrNotFound (the damaged entry)", err)
	}
	if v, err := s.Get([]byte("k2")); err != nil || !bytes.Equal(v, []byte("CCCC")) {
		t.Errorf("Get(k2) = %q,%v; want CCCC (recovered by resync)", v, err)
	}
	st := s.snapshot()
	if st.CorruptionErrors != 1 {
		t.Errorf("CorruptionErrors = %d, want 1", st.CorruptionErrors)
	}
	// Exactly one entry's worth of bytes discarded — from the tear to the start of k2.
	if want := uint64(offs[2] - offs[1]); st.CorruptionBytesDiscarded != want {
		t.Errorf("CorruptionBytesDiscarded = %d, want %d (just entry k1, not the tail)", st.CorruptionBytesDiscarded, want)
	}
}

// TestResyncForwardIsBudgeted pins the availability guard on the resync: its total
// MAC work is capped by a per-call byte budget, so it cannot be driven quadratic by a
// page that presents a valid frame length at many offsets. A genuine frame within the
// budget's reach is still found (the common case is unaffected); a genuine frame only
// reachable past the budget is NOT adopted — the scan gives up and the caller
// truncates. The beyond-budget assertion fails PRE-FIX: an unbudgeted scan runs the
// whole page and finds the far frame.
func TestResyncForwardIsBudgeted(t *testing.T) {
	const n = 1 << 16 // a modest page so the test is fast
	const nonce = uint64(0)

	// A zero-filled buffer with one genuine frame at offset g. Every zero offset is a
	// tiny in-bounds candidate (keyLen=valLen=0) that costs one header's worth of MAC,
	// so the budget (2*n) is reached after ~2*n/entryHeaderSize offsets.
	build := func(g int) []byte {
		buf := make([]byte, n)
		frame := make([]byte, entrySpanExact(len("real"), len("value")))
		if _, err := encodeEntry(frame, []byte("real"), []byte("value"), 0, makeMeta(1, false), testFramingKey, nonce, uint32(g)); err != nil {
			t.Fatal(err)
		}
		copy(buf[g:], frame)
		return buf
	}

	// WITHIN budget: a genuine frame a short way in is found and adopted.
	if at, found := resyncForward(build(300), 0, n, testFramingKey, nonce); !found || at != 300 {
		t.Errorf("within-budget: resyncForward = (%d,%v), want (300,true)", at, found)
	}

	// BEYOND budget: the same frame near the end is not reached — the budget is spent
	// first and the scan truncates. Pre-fix (no budget) this returns (near-end, true).
	if at, found := resyncForward(build(n-100), 0, n, testFramingKey, nonce); found || at != n {
		t.Errorf("beyond-budget: resyncForward = (%d,%v), want (%d,false)", at, found, n)
	}

	// LARGE candidates: a single maximal in-bounds frame claimed at offset 0 alone
	// charges ~a whole page of MAC input — the shape that makes an unbudgeted scan
	// quadratic. With a genuine frame beyond, the budget still gives up.
	buf := build(n - 100)
	binary.LittleEndian.PutUint16(buf[0:2], 0)                         // keyLen = 0
	binary.LittleEndian.PutUint32(buf[2:6], uint32(n-entryHeaderSize)) // valLen fills the page
	if at, found := resyncForward(buf, 0, n, testFramingKey, nonce); found || at != n {
		t.Errorf("large-candidate beyond-budget: resyncForward = (%d,%v), want (%d,false)", at, found, n)
	}
}

// TestNonceRotatesAndRejectsReverseSkew pins the reverse-skew guard: when an mmap
// extent is handed back to the write path, zeroDurableBoundsForReuseLocked rotates the
// page nonce and persists it, so any old-life bytes still lying in the extent can no
// longer verify — closing the window where a torn bounds writeback could resurrect a
// previous life's frame (or let a planted one pass).
func TestNonceRotatesAndRejectsReverseSkew(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0

	s, err := newShard(cfg, t.TempDir(), nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	p := s.pages[0]
	if p.nonce != 0 {
		t.Fatalf("fresh page nonce = %d, want 0", p.nonce)
	}
	off, _, err := p.Write([]byte("old"), []byte("life"), 0, makeMeta(1, false))
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	ent := p.entries()
	// Verifies under the nonce it was written with.
	if _, _, _, _, err := decodeEntry(ent[off:], s.framingKey, 0, off); err != nil {
		t.Fatalf("decode under original nonce = %v, want nil", err)
	}

	// Hand the extent back to the write path: rotate the nonce + zero the bounds.
	s.mu.Lock()
	s.zeroDurableBoundsForReuseLocked(0)
	s.mu.Unlock()

	newNonce := p.nonce
	if newNonce == 0 {
		t.Fatal("nonce did not rotate on reuse")
	}
	if p.durableNonce() != newNonce {
		t.Errorf("durable nonce = %d, want the rotated %d (must be persisted before reuse)", p.durableNonce(), newNonce)
	}
	// The old-life bytes still physically present in the extent must NOT verify under
	// the new nonce — the reverse-skew rejection.
	if _, _, _, _, err := decodeEntry(ent[off:], s.framingKey, newNonce, off); err != errMACMismatch {
		t.Errorf("old-life frame under the rotated nonce = %v, want errMACMismatch", err)
	}
}

// TestRecoveryRejectsReplayedGenuineFrame pins the OFFSET binding against replay: a
// byte-perfect copy of a GENUINE frame (encoded under this very file's framing key)
// planted at a different offset inside a value is NOT adopted by the resync, because
// its MAC was bound to the offset it was originally written at. Unkeyed CRCs and even
// a keyed MAC that omitted the offset would accept the replay.
func TestRecoveryRejectsReplayedGenuineFrame(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0
	cfg.DisableColdCompaction = true

	const replaySeq = uint64(1) << 50
	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")

	// Build the file manually so the replayed frame can be encoded under the file's
	// OWN framing key (a genuine frame, not a client guess) — isolating the offset
	// binding from the key binding.
	buf := make([]byte, headerSize+cfg.PageSize)
	if err := writeHeader(buf, uint32(cfg.PageSize), 1, 0); err != nil {
		t.Fatal(err)
	}
	fk, ok := readFramingKey(buf)
	if !ok {
		t.Fatal("no framing key")
	}
	// A genuine "replay" frame encoded as if it were the first entry (offset 0).
	replayFrame := make([]byte, entrySpanExact(len("replay"), len("R")))
	if _, err := encodeEntry(replayFrame, []byte("replay"), []byte("R"), 0, makeMeta(replaySeq, false), fk, 0, 0); err != nil {
		t.Fatal(err)
	}
	// Carrier value = padding then the genuine replay frame, so the frame lands at a
	// NON-zero offset inside the page.
	carrierVal := append(bytes.Repeat([]byte("c"), 200), replayFrame...)

	base := headerSize + pageHdrSize
	cursor := 0
	writeReal := func(key, val []byte, seq uint64) int {
		n, err := encodeEntry(buf[base+cursor:], key, val, 0, makeMeta(seq, false), fk, 0, uint32(cursor))
		if err != nil {
			t.Fatal(err)
		}
		at := cursor
		cursor += n
		return at
	}
	writeReal([]byte("anchor"), []byte("A"), 1)
	carrierOff := writeReal([]byte("carrier"), carrierVal, 2)
	writeReal([]byte("k2"), []byte("CCCC"), 3)
	// page 0 head/tail
	binary.LittleEndian.PutUint32(buf[headerSize:headerSize+4], 0)
	binary.LittleEndian.PutUint32(buf[headerSize+4:headerSize+8], uint32(cursor))
	// Tear the carrier's payload so recovery must resync past it (and thus scan across
	// the embedded replay frame).
	buf[base+carrierOff+entryHeaderSize] ^= 0xFF
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}

	s, err := newShard(cfg, dir, nil)
	if err != nil {
		t.Fatalf("newShard: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Get([]byte("replay")); err != ErrNotFound {
		t.Errorf("Get(replay) = %v; a genuine frame replayed at a different offset was adopted", err)
	}
	if s.writeSeq >= replaySeq {
		t.Errorf("writeSeq = %d; lifted to the replayed frame's sequence", s.writeSeq)
	}
	if v, err := s.Get([]byte("anchor")); err != nil || !bytes.Equal(v, []byte("A")) {
		t.Errorf("Get(anchor) = %q,%v; want A", v, err)
	}
	if v, err := s.Get([]byte("k2")); err != nil || !bytes.Equal(v, []byte("CCCC")) {
		t.Errorf("Get(k2) = %q,%v; want CCCC (resync resumed at the genuine successor)", v, err)
	}
}
