// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"testing"
)

// valLens the entry framing must refuse on EVERY platform: each runs far past any
// page, and together they straddle the point where entryHeaderSize+keyLen+valLen
// stops fitting a 32-bit int. On 386 the widened valLen stays POSITIVE for all of
// them, so a `valLen < 0` guard alone lets them through, and summing them into a
// total wraps it negative — a negative slice bound (a panic) in the decoders and a
// negative head in EvictFront. On 64-bit none of this wraps and they are simply too
// long, which is the answer 386 must give as well.
var overflowingValLens = []uint32{
	0x7FFFFFFF,
	0x80000000 - entryHeaderSize, // with keyLen 0 the total lands exactly on the wrap
	0x7FFFFFF0,
	0x7FFF0000,
	0x7FF00000,
}

// framedWithLens is a 4 KiB buffer whose first entry header claims keyLen/valLen.
func framedWithLens(keyLen uint16, valLen uint32) []byte {
	src := make([]byte, 4096)
	binary.LittleEndian.PutUint16(src[0:2], keyLen)
	binary.LittleEndian.PutUint32(src[2:6], valLen)
	return src
}

// TestDecodersRejectValLenThatOverflowsTheTotal pins that neither decoder sums a
// crash-exposed length before bounding it. rebuildIndexFromPages feeds decodeEntry
// raw page bytes, so on a 32-bit build a single rotted valLen used to panic the
// warm-restart walk — on every restart, since the bytes are still there.
func TestDecodersRejectValLenThatOverflowsTheTotal(t *testing.T) {
	for _, keyLen := range []uint16{0, 1, 255, maxKeyLen} {
		for _, valLen := range overflowingValLens {
			src := framedWithLens(keyLen, valLen)
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("decodeEntry(keyLen=%d, valLen=%#x) panicked: %v", keyLen, valLen, r)
					}
				}()
				if _, _, _, _, err := decodeEntry(src); err != errEntryTruncated {
					t.Errorf("decodeEntry(keyLen=%d, valLen=%#x) = %v, want errEntryTruncated", keyLen, valLen, err)
				}
			}()
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("decodeEntryFast(keyLen=%d, valLen=%#x) panicked: %v", keyLen, valLen, r)
					}
				}()
				if _, _, _, err := decodeEntryFast(src); err != errEntryTruncated {
					t.Errorf("decodeEntryFast(keyLen=%d, valLen=%#x) = %v, want errEntryTruncated", keyLen, valLen, err)
				}
			}()
		}
	}
}

// TestDecodersAcceptAnEntryEndingExactlyAtTheSlice is the other edge of the same
// bound: rewriting the length check must not start refusing an entry that fills its
// slice to the last byte.
func TestDecodersAcceptAnEntryEndingExactlyAtTheSlice(t *testing.T) {
	key, val := []byte("key"), []byte("value")
	src := make([]byte, entrySpanExact(len(key), len(val)))
	if _, err := encodeEntry(src, key, val, 0, makeMeta(1, false)); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := decodeEntry(src); err != nil {
		t.Errorf("decodeEntry on an exactly-sized slice = %v, want nil", err)
	}
	if _, _, _, err := decodeEntryFast(src); err != nil {
		t.Errorf("decodeEntryFast on an exactly-sized slice = %v, want nil", err)
	}
	if _, _, _, err := decodeEntryFast(src[:len(src)-1]); err != errEntryTruncated {
		t.Errorf("decodeEntryFast one byte short = %v, want errEntryTruncated", err)
	}
}

// TestEvictFrontRejectsValLenThatOverflowsTheHead is EvictFront's half: it computed
// newHead = keyEnd + valLen before comparing it with tail, so on 386 the sum wrapped
// negative, passed `newHead > tail`, and was stored as the page head — silently, with
// no error for drainPageLocked to count.
func TestEvictFrontRejectsValLenThatOverflowsTheHead(t *testing.T) {
	for _, valLen := range overflowingValLens {
		p := newTestPage(t, 1<<20)
		for range 4 {
			if _, _, err := p.Write([]byte("k"), make([]byte, 256), 0, 0); err != nil {
				t.Fatalf("Write: %v", err)
			}
		}
		headBefore, tailBefore := p.head(), p.tail()
		binary.LittleEndian.PutUint32(p.entries()[headBefore+2:headBefore+6], valLen)

		if _, _, err := p.EvictFront(); err != errEntryTruncated {
			t.Errorf("EvictFront(valLen=%#x) = %v, want errEntryTruncated", valLen, err)
		}
		if p.head() != headBefore || p.tail() != tailBefore {
			t.Errorf("EvictFront(valLen=%#x) moved the framing to head=%d tail=%d (was %d/%d)",
				valLen, p.head(), p.tail(), headBefore, tailBefore)
		}
	}
}
