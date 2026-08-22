// SPDX-License-Identifier: Apache-2.0
package wire

import (
	"bytes"
	"encoding/binary"
	"testing"
	"time"
)

// TestAppendKeyArgsByteIdentical checks AppendKeyArgs against an INDEPENDENT
// hand-built encoding (not the delegating EncodeKeyArgs), across nil and a
// pre-grown (reused-cap) dst.
func TestAppendKeyArgsByteIdentical(t *testing.T) {
	key := []byte("some/key-0123456789")

	oracle := make([]byte, 2+len(key))
	binary.BigEndian.PutUint16(oracle[0:2], uint16(len(key)))
	copy(oracle[2:], key)

	if got := AppendKeyArgs(nil, key); !bytes.Equal(got, oracle) {
		t.Fatalf("dst=nil: got %x want %x", got, oracle)
	}
	dirty := bytes.Repeat([]byte{0xEE}, len(oracle)+64)
	got := AppendKeyArgs(dirty[:0], key)
	if !bytes.Equal(got, oracle) || len(got) != len(oracle) {
		t.Fatalf("reused-cap: got %x (len %d) want %x (len %d)", got, len(got), oracle, len(oracle))
	}
}

// TestAppendPutArgsByteIdentical checks AppendPutArgs against an INDEPENDENT
// hand-built encoding, across nil and a pre-grown (reused-cap) dst.
func TestAppendPutArgsByteIdentical(t *testing.T) {
	key := []byte("k1")
	val := []byte("value-payload-here")
	ttl := 5 * time.Second

	n := 2 + len(key) + 4 + len(val) + 8
	oracle := make([]byte, n)
	binary.BigEndian.PutUint16(oracle[0:2], uint16(len(key)))
	copy(oracle[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint32(oracle[off:off+4], uint32(len(val)))
	copy(oracle[off+4:], val)
	off += 4 + len(val)
	binary.BigEndian.PutUint64(oracle[off:off+8], uint64(ttl/time.Millisecond))

	if got := AppendPutArgs(nil, key, val, ttl); !bytes.Equal(got, oracle) {
		t.Fatalf("dst=nil: got %x want %x", got, oracle)
	}
	dirty := bytes.Repeat([]byte{0xEE}, n+64)
	got := AppendPutArgs(dirty[:0], key, val, ttl)
	if !bytes.Equal(got, oracle) || len(got) != len(oracle) {
		t.Fatalf("reused-cap: got %x (len %d) want %x (len %d)", got, len(got), oracle, len(oracle))
	}
}
