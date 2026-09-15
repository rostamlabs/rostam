// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"math/bits"
)

// SipHash-2-4, vendored.
//
// WHY VENDORED, AND WHY THIS ALGORITHM. The persistent cache frames every entry
// with a keyed MAC (see cache/ringbuf.go): the tag over a frame is computed with a
// per-file secret so a client VALUE — which is attacker-chosen bytes — cannot carry
// a frame that recovery's resync would adopt (the GHSA-m63m-rp87-w4rf forge class).
// That requires a PSEUDORANDOM FUNCTION keyed by a secret, and it requires the SAME
// bytes to hash to the SAME tag on every process, build and architecture that ever
// opens the file, forever. Neither property is available from the standard library:
//
//   - hash/maphash is keyed, but its algorithm is DELIBERATELY unspecified and free
//     to change between Go versions, so a tag written by one build could stop
//     verifying under the next — silent, whole-page data loss on upgrade.
//   - crc32 / fnv / xxhash are stable but UNKEYED: a client can compute the tag over
//     bytes it controls, which is exactly the forge the MAC exists to stop.
//
// SipHash-2-4 is a fixed, published PRF (Aumasson & Bernstein) with a frozen spec,
// so a vendored implementation is stable across Go versions and GOARCH by
// construction, and it is keyed, so the tag is unforgeable without the per-file
// secret. It is a MAC here, not a hash-table hash; 64 bits of tag is the format's
// per-entry integrity field. TestSipHashKnownAnswers locks this implementation to
// the reference vectors so a future edit that drifts from the spec fails the suite
// rather than silently rejecting every entry an older build wrote.
//
// The streaming shape (init/write/sum) exists so the MAC can be computed over the
// logical concatenation nonce‖offset‖header‖key‖value WITHOUT allocating a
// contiguous buffer to hold it — the write path calls this once per stored entry.

// SipHash-2-4 round constants (the ASCII of "somepseudorandomlygeneratedbytes").
const (
	sipC0 = 0x736f6d6570736575
	sipC1 = 0x646f72616e646f6d
	sipC2 = 0x6c7967656e657261
	sipC3 = 0x7465646279746573
)

// sipHasher is a streaming SipHash-2-4 state. It absorbs an arbitrary byte stream
// in 8-byte little-endian words, buffering the partial trailing word across write
// calls so a caller can feed non-contiguous pieces (nonce, offset, header, key,
// value) with no intermediate allocation. Zero value is NOT valid; construct with
// newSipHasher.
type sipHasher struct {
	v0, v1, v2, v3 uint64
	buf            [8]byte // partial (<8-byte) trailing word not yet compressed
	n              int     // bytes currently buffered in buf (0..7)
	total          uint64  // total bytes absorbed (feeds the finalization length byte)
}

// newSipHasher initializes the state from a 16-byte key split into two
// little-endian 64-bit halves (k0 = key[0:8], k1 = key[8:16]), exactly as the
// SipHash reference does.
func newSipHasher(k0, k1 uint64) sipHasher {
	return sipHasher{
		v0: sipC0 ^ k0,
		v1: sipC1 ^ k1,
		v2: sipC2 ^ k0,
		v3: sipC3 ^ k1,
	}
}

// sipRound is one SIPROUND applied to the state.
func (h *sipHasher) sipRound() {
	h.v0 += h.v1
	h.v1 = bits.RotateLeft64(h.v1, 13)
	h.v1 ^= h.v0
	h.v0 = bits.RotateLeft64(h.v0, 32)
	h.v2 += h.v3
	h.v3 = bits.RotateLeft64(h.v3, 16)
	h.v3 ^= h.v2
	h.v0 += h.v3
	h.v3 = bits.RotateLeft64(h.v3, 21)
	h.v3 ^= h.v0
	h.v2 += h.v1
	h.v1 = bits.RotateLeft64(h.v1, 17)
	h.v1 ^= h.v2
	h.v2 = bits.RotateLeft64(h.v2, 32)
}

// compress absorbs one full 8-byte little-endian word m (the c=2 compression
// rounds of SipHash-2-4).
func (h *sipHasher) compress(m uint64) {
	h.v3 ^= m
	h.sipRound()
	h.sipRound()
	h.v0 ^= m
}

// write absorbs p into the state.
func (h *sipHasher) write(p []byte) {
	h.total += uint64(len(p))
	// Top off any partial word first.
	if h.n > 0 {
		for len(p) > 0 && h.n < 8 {
			h.buf[h.n] = p[0]
			h.n++
			p = p[1:]
		}
		if h.n == 8 {
			h.compress(binary.LittleEndian.Uint64(h.buf[:]))
			h.n = 0
		}
	}
	for len(p) >= 8 {
		h.compress(binary.LittleEndian.Uint64(p[:8]))
		p = p[8:]
	}
	for i := 0; i < len(p); i++ {
		h.buf[h.n] = p[i]
		h.n++
	}
}

// sum finalizes and returns the 64-bit tag. The last block carries the remaining
// buffered bytes in its low positions and the total input length (mod 256) in its
// most-significant byte, per the spec; the finalization applies c=2 then the d=4
// output rounds.
func (h *sipHasher) sum() uint64 {
	// b = (len mod 256) << 56 | (trailing bytes, little-endian). The shift discards
	// all but the low 8 bits of total, so this is exactly (total & 0xff) << 56.
	b := h.total << 56
	for i := 0; i < h.n; i++ {
		b |= uint64(h.buf[i]) << (8 * uint(i)) //nolint:gosec // i < 8
	}
	h.v3 ^= b
	h.sipRound()
	h.sipRound()
	h.v0 ^= b
	h.v2 ^= 0xff
	h.sipRound()
	h.sipRound()
	h.sipRound()
	h.sipRound()
	return h.v0 ^ h.v1 ^ h.v2 ^ h.v3
}

// sipKeyHalves splits a 16-byte framing key into the two little-endian 64-bit
// halves SipHash is keyed by. Panics if key is not exactly 16 bytes — a
// programming error (the framing key is always framingKeyLen), never reachable
// from a decoded file.
func sipKeyHalves(key []byte) (k0, k1 uint64) {
	if len(key) != framingKeyLen {
		panic("cache: siphash key must be 16 bytes")
	}
	return binary.LittleEndian.Uint64(key[0:8]), binary.LittleEndian.Uint64(key[8:16])
}

// sipHash24 computes SipHash-2-4 of msg under a 16-byte key. A convenience
// one-shot over a contiguous message; the entry MAC uses the streaming form
// directly to avoid materializing its concatenated input.
func sipHash24(key, msg []byte) uint64 {
	k0, k1 := sipKeyHalves(key)
	h := newSipHasher(k0, k1)
	h.write(msg)
	return h.sum()
}
