// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// testFramingKey is a fixed 16-byte key for unit tests that encode AND decode a frame
// in the same process (production keys are random per file). Tests that build a whole
// pages file read the file's own generated key back instead.
var testFramingKey = []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

func TestEntryEncodeDecode(t *testing.T) {
	buf := make([]byte, 256)
	key := []byte("user:42")
	val := []byte(`{"coins":100}`)
	expiry := uint64(1717000000000) // arbitrary ms
	meta := makeMeta(987654321, false)
	const nonce = uint64(0x1122334455667788)
	const offset = uint32(48)

	n, err := encodeEntry(buf, key, val, expiry, meta, testFramingKey, nonce, offset)
	if err != nil {
		t.Fatalf("encodeEntry: %v", err)
	}
	if n != entryHeaderSize+len(key)+len(val) {
		t.Fatalf("encoded size = %d, want %d", n, entryHeaderSize+len(key)+len(val))
	}

	dk, dv, dexp, dmeta, derr := decodeEntry(buf[:n], testFramingKey, nonce, offset)
	if derr != nil {
		t.Fatalf("decodeEntry: %v", derr)
	}
	if !bytes.Equal(dk, key) {
		t.Errorf("key roundtrip: got %q want %q", dk, key)
	}
	if !bytes.Equal(dv, val) {
		t.Errorf("val roundtrip: got %q want %q", dv, val)
	}
	if dexp != expiry {
		t.Errorf("expiry roundtrip: got %d want %d", dexp, expiry)
	}
	if dmeta != meta {
		t.Errorf("meta roundtrip: got %#x want %#x", dmeta, meta)
	}
	// entryMetaAt is the recovery-path accessor; it must agree with the full decode
	// so the hot decode never has to widen its signature to carry meta.
	if got := entryMetaAt(buf[:n]); got != meta {
		t.Errorf("entryMetaAt = %#x, want %#x", got, meta)
	}
}

// TestEntryMACBindsNonceAndOffset is the property the whole v5 resync rests on: the
// SAME bytes verify ONLY at the (nonce, offset) they were written under. A frame
// decoded at a different offset, or under a rotated nonce, must be rejected — which is
// what stops a genuine frame being replayed into a reused extent or copied elsewhere,
// and what lets recovery trust "the next frame that verifies". A pre-fix (CRC or
// unkeyed) implementation cannot fail these because its integrity field depends on
// neither.
func TestEntryMACBindsNonceAndOffset(t *testing.T) {
	buf := make([]byte, 128)
	key, val := []byte("k"), []byte("v")
	const nonce = uint64(7)
	const offset = uint32(100)
	n, err := encodeEntry(buf, key, val, 0, makeMeta(1, false), testFramingKey, nonce, offset)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := decodeEntry(buf[:n], testFramingKey, nonce, offset); err != nil {
		t.Fatalf("decode at written (nonce,offset) = %v, want nil", err)
	}
	if _, _, _, _, err := decodeEntry(buf[:n], testFramingKey, nonce, offset+1); err != errMACMismatch {
		t.Errorf("decode at a different offset = %v, want errMACMismatch", err)
	}
	if _, _, _, _, err := decodeEntry(buf[:n], testFramingKey, nonce+1, offset); err != errMACMismatch {
		t.Errorf("decode under a rotated nonce = %v, want errMACMismatch", err)
	}
	other := []byte{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
	if _, _, _, _, err := decodeEntry(buf[:n], other, nonce, offset); err != errMACMismatch {
		t.Errorf("decode under a different framing key = %v, want errMACMismatch", err)
	}
}

// TestEntryMetaPacking pins the meta word's field split: 56 bits of sequence, one
// tombstone flag above them, and no interference in either direction.
func TestEntryMetaPacking(t *testing.T) {
	const maxSeq = entrySeqMask // 2^56 - 1
	for _, tc := range []struct {
		seq  uint64
		tomb bool
	}{
		{0, false}, {0, true}, {1, false}, {1, true},
		{maxSeq, false}, {maxSeq, true}, {1 << 55, true},
	} {
		m := makeMeta(tc.seq, tc.tomb)
		if got := metaSeq(m); got != tc.seq {
			t.Errorf("makeMeta(%d,%v): seq = %d, want %d", tc.seq, tc.tomb, got, tc.seq)
		}
		if got := metaIsTombstone(m); got != tc.tomb {
			t.Errorf("makeMeta(%d,%v): tombstone = %v, want %v", tc.seq, tc.tomb, got, tc.tomb)
		}
	}
	// A tombstone must not be forgeable by a large sequence alone.
	if metaIsTombstone(makeMeta(entrySeqMask, false)) {
		t.Fatal("the maximum sequence set the tombstone flag — the mask is wrong")
	}
}

// TestEntryGoldenBytes pins the exact on-disk serialization of a known entry. The
// ring-buffer entry codec is a persisted format: any drift (endianness, field
// order/width, MAC input) silently breaks reopening a DataDir. The fixed header and
// payload bytes are pinned literally; the MAC slot is pinned to the value entryMAC
// produces for a fixed (key, nonce, offset), and TestSipHashKnownAnswers separately
// locks entryMAC's algorithm — together they make any codec change fail the suite.
//
// It also pins the ORDERING the format rests on: meta sits between the expiry and the
// MAC (BEFORE the key), so the key still starts at the fixed entryHeaderSize offset
// and decodeEntryFast needs no extra arithmetic.
func TestEntryGoldenBytes(t *testing.T) {
	key := []byte("user:1")
	val := []byte("hi")
	expiry := uint64(0x0102030405060708)
	meta := makeMeta(0x0102030405, true) // both the flag and the sequence are non-trivial
	const nonce = uint64(0)
	const offset = uint32(0)

	// Header + payload, little-endian: keyLen(2) valLen(4) expiry(8) meta(8) mac(8) key value.
	wantHead := []byte{
		0x06, 0x00, // keyLen = 6
		0x02, 0x00, 0x00, 0x00, // valLen = 2
		0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // expiry
		0x05, 0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x01, // meta = tombstone | seq 0x0102030405
	}

	buf := make([]byte, entryHeaderSize+len(key)+len(val))
	n, err := encodeEntry(buf, key, val, expiry, meta, testFramingKey, nonce, offset)
	if err != nil {
		t.Fatalf("encodeEntry: %v", err)
	}
	if n != len(buf) {
		t.Fatalf("encoded length = %d, want %d", n, len(buf))
	}
	if !bytes.Equal(buf[0:entryMACOff], wantHead) {
		t.Fatalf("header golden mismatch:\n got %x\nwant %x", buf[0:entryMACOff], wantHead)
	}
	if !bytes.Equal(buf[entryHeaderSize:], append([]byte("user:1"), "hi"...)) {
		t.Fatalf("payload golden mismatch: got %x", buf[entryHeaderSize:])
	}
	wantMAC := entryMAC(testFramingKey, nonce, offset, buf[0:entryMACOff], buf[entryHeaderSize:n])
	if got := binary.LittleEndian.Uint64(buf[entryMACOff:entryHeaderSize]); got != wantMAC {
		t.Fatalf("MAC slot = %#016x, want %#016x", got, wantMAC)
	}
}

// TestEntryMACCoversMeta: the meta word is inside the MAC's covered range, so a
// flipped sequence or tombstone bit is a detected corruption, not a silently accepted
// change of recency.
func TestEntryMACCoversMeta(t *testing.T) {
	buf := make([]byte, 64)
	n, err := encodeEntry(buf, []byte("k"), []byte("v"), 0, makeMeta(7, false), testFramingKey, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	buf[entryMetaOff] ^= 0x01
	if _, _, _, _, err := decodeEntry(buf[:n], testFramingKey, 0, 0); err != errMACMismatch {
		t.Fatalf("decodeEntry after flipping a meta bit = %v, want errMACMismatch", err)
	}
}

func TestEntryDecodeCorruption(t *testing.T) {
	buf := make([]byte, 64)
	n, _ := encodeEntry(buf, []byte("k"), []byte("v"), 0, 0, testFramingKey, 0, 0)
	// flip a payload byte after the header
	buf[entryHeaderSize] ^= 0xFF
	if _, _, _, _, err := decodeEntry(buf[:n], testFramingKey, 0, 0); err == nil {
		t.Fatal("expected MAC error after payload mutation, got nil")
	}
}

func TestEntryEncodeBufferTooSmall(t *testing.T) {
	buf := make([]byte, 4)
	_, err := encodeEntry(buf, []byte("longer-than-buffer"), []byte("v"), 0, 0, testFramingKey, 0, 0)
	if err == nil {
		t.Fatal("expected errBufferTooSmall")
	}
}

func TestEntryKeyTooLong(t *testing.T) {
	buf := make([]byte, 1<<16)
	key := make([]byte, 1<<16) // exceeds uint16 max
	_, err := encodeEntry(buf, key, []byte("v"), 0, 0, testFramingKey, 0, 0)
	if err == nil {
		t.Fatal("expected key-too-long error")
	}
}
