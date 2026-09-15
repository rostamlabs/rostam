// SPDX-License-Identifier: Apache-2.0

package cache

import "testing"

// TestSipHashKnownAnswers locks the vendored SipHash-2-4 to the reference test
// vectors published with the algorithm (Aumasson & Bernstein). The key is the
// canonical 00 01 02 … 0f, and message i is the i-byte prefix 00 01 … (i-1); the
// expected tags are vectors_sip64[0..15] from the reference implementation.
//
// This is the correctness anchor for the whole entry-integrity format: the MAC is a
// persisted field, so an implementation that drifts from the spec would silently
// reject every entry a correct build ever wrote (whole-page loss on upgrade). These
// vectors cover every trailing-block residue (message lengths 0..7) and one and two
// full compression blocks (lengths 8..15), so any divergence in the compression
// rounds, the finalization, the length byte, or the key split fails here.
func TestSipHashKnownAnswers(t *testing.T) {
	key := make([]byte, 16)
	for i := range key {
		key[i] = byte(i)
	}
	want := []uint64{
		0x726fdb47dd0e0e31, 0x74f839c593dc67fd, 0x0d6c8009d9a94f5a, 0x85676696d7fb7e2d,
		0xcf2794e0277187b7, 0x18765564cd99a68d, 0xcbc9466e58fee3ce, 0xab0200f58b01d137,
		0x93f5f5799a932462, 0x9e0082df0ba9e4b0, 0x7a5dbbc594ddb9f3, 0xf4b32f46226bada7,
		0x751e8fbc860ee5fb, 0x14ea5627c0843d90, 0xf723ca908e7af2ee, 0xa129ca6149be45e5,
	}
	for i, w := range want {
		msg := make([]byte, i)
		for j := range msg {
			msg[j] = byte(j)
		}
		if got := sipHash24(key, msg); got != w {
			t.Errorf("sipHash24(len=%d) = %#016x, want %#016x", i, got, w)
		}
	}
}

// TestSipHashStreamingMatchesOneShot proves the streaming write path (which the
// entry MAC uses to hash nonce‖offset‖header‖key‖value without a contiguous buffer)
// produces the same tag as a single write, across every split of the input — the
// property that lets the MAC be computed piecewise over a frame's fields.
func TestSipHashStreamingMatchesOneShot(t *testing.T) {
	key := make([]byte, 16)
	for i := range key {
		key[i] = byte(0x40 + i)
	}
	msg := make([]byte, 37)
	for i := range msg {
		msg[i] = byte(i * 7)
	}
	k0, k1 := sipKeyHalves(key)
	oneShot := sipHash24(key, msg)
	for split1 := 0; split1 <= len(msg); split1++ {
		for split2 := split1; split2 <= len(msg); split2++ {
			h := newSipHasher(k0, k1)
			h.write(msg[:split1])
			h.write(msg[split1:split2])
			h.write(msg[split2:])
			if got := h.sum(); got != oneShot {
				t.Fatalf("streaming split (%d,%d) = %#016x, want %#016x", split1, split2, got, oneShot)
			}
		}
	}
}
