// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"math"
	"testing"
)

// maxEncodableValueLen is the largest value length this platform can even
// represent in an int. maxValueLen is typed int64 precisely because it does not
// fit an int on a 32-bit build, and the 32-bit lane runs these tests — so the
// clamp goes through an int64 VARIABLE. int(maxValueLen) would be a constant
// conversion and would refuse to compile on 386 rather than clamping.
func maxEncodableValueLen() int {
	maxLen := maxValueLen
	if maxLen > int64(math.MaxInt) {
		maxLen = int64(math.MaxInt)
	}
	return int(maxLen)
}

func TestEntrySpanExactIsHeaderPlusKeyPlusValue(t *testing.T) {
	for _, tc := range []struct{ keyLen, valLen int }{
		{0, 0}, {1, 0}, {0, 1}, {3, 7}, {16, 64}, {maxKeyLen, 1 << 20},
	} {
		want := entryHeaderSize + tc.keyLen + tc.valLen
		if got := entrySpanExact(tc.keyLen, tc.valLen); got != want {
			t.Errorf("entrySpanExact(%d, %d) = %d, want %d", tc.keyLen, tc.valLen, got, want)
		}
	}
}

// TestEntrySpanEqualsExactInBothModes is the whole no-behaviour-change claim of
// the entrySpan split: with classOf an identity there are no size classes yet, so
// the padded and unpadded spans are the exact framing and each other. Whatever
// decision a call site recorded, it got the number entrySize used to give it.
func TestEntrySpanEqualsExactInBothModes(t *testing.T) {
	for _, tc := range []struct{ keyLen, valLen int }{
		{0, 0}, {1, 0}, {0, 1}, {3, 7}, {16, 64}, {255, 4095}, {maxKeyLen, 1 << 20},
	} {
		want := entrySpanExact(tc.keyLen, tc.valLen)
		if got := entrySpan(tc.keyLen, tc.valLen, false); got != want {
			t.Errorf("entrySpan(%d, %d, false) = %d, want %d", tc.keyLen, tc.valLen, got, want)
		}
		if got := entrySpan(tc.keyLen, tc.valLen, true); got != want {
			t.Errorf("entrySpan(%d, %d, true) = %d, want %d", tc.keyLen, tc.valLen, got, want)
		}
	}
}

// TestPageWriteReservesTheOccupancySpan anchors the capacity/write rule stated on
// entrySpan. page.Write is the only place that turns a span into reserved bytes,
// so every capacity check in the package is checked against THIS: a page with
// exactly the occupancy span free must accept the entry, and one byte less must
// refuse it. A capacity site that asks for anything smaller than what this pins
// would say yes to a page Write then rejects.
func TestPageWriteReservesTheOccupancySpan(t *testing.T) {
	key, val := []byte("key"), []byte("a-value-of-some-length")
	span := entrySpan(len(key), len(val), true)

	exact := newHeapPage(span)
	if _, _, err := exact.Write(key, val, 0, 0); err != nil {
		t.Errorf("Write into exactly entrySpan(...,true)=%d bytes: %v, want success", span, err)
	}

	short := newHeapPage(span - 1)
	if _, _, err := short.Write(key, val, 0, 0); err != errPageFull {
		t.Errorf("Write into %d bytes: err = %v, want %v", span-1, err, errPageFull)
	}
}

// classOfTestLens spans the lengths a class function has to behave at: the
// degenerate small ones, powers of two and their neighbours (where any rounding
// scheme puts its boundaries), and the top of the representable range.
func classOfTestLens() []int {
	lens := []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65}
	for shift := 7; shift < 22; shift++ {
		n := 1 << shift
		lens = append(lens, n-1, n, n+1)
	}
	maxLen := maxEncodableValueLen()
	lens = append(lens, maxLen/2, maxLen-1, maxLen)
	return lens
}

// TestClassOfIsIdentityForNow pins the CURRENT implementation, which is what
// makes entrySpan(padded=true) provably equal to entrySpanExact. It is expected
// to be deleted by whichever change introduces real size classes; the three
// contract tests below are the ones that must survive that change.
func TestClassOfIsIdentityForNow(t *testing.T) {
	for _, n := range classOfTestLens() {
		if got := classOf(n); got != n {
			t.Errorf("classOf(%d) = %d, want %d (identity)", n, got, n)
		}
	}
}

// TestClassOfNeverShrinks is the safety half of the contract: a class must be
// able to HOLD the value handed to it. A class smaller than its value would
// frame an entry over bytes it does not own.
func TestClassOfNeverShrinks(t *testing.T) {
	for _, n := range classOfTestLens() {
		if got := classOf(n); got < n {
			t.Errorf("classOf(%d) = %d, want >= %d", n, got, n)
		}
	}
}

// TestClassOfIsMonotonic is the ordering half: a longer value may never land in
// a SMALLER class. Without it, "does the stored class still hold the new value"
// could not be answered by comparing classes at all.
func TestClassOfIsMonotonic(t *testing.T) {
	lens := classOfTestLens()
	for i := range lens {
		for j := range lens {
			if lens[i] > lens[j] {
				continue
			}
			if ci, cj := classOf(lens[i]), classOf(lens[j]); ci > cj {
				t.Errorf("classOf(%d) = %d > classOf(%d) = %d; not monotonic",
					lens[i], ci, lens[j], cj)
			}
		}
	}
}

// TestClassOfDoesNotOverflowNearMaxValueLen is the guard a rounding-up class
// function most easily fails: at the top of the range there is nothing left to
// round up INTO. Rounding past the maximum encodable length would either wrap the
// result negative or hand back a class encodeEntry must reject, and a negative
// span would send every band and room check the wrong way.
func TestClassOfDoesNotOverflowNearMaxValueLen(t *testing.T) {
	maxLen := maxEncodableValueLen()
	// int64 candidates, filtered to what an int holds here: 1<<31 is not an int
	// constant on 386 and would fail to compile inside an []int literal.
	candidates := []int64{int64(maxLen), int64(maxLen) - 1, int64(maxLen) / 2, int64(maxLen)/2 + 1, 1 << 31, 1 << 30}
	for _, c := range candidates {
		if c < 0 || c > int64(maxLen) {
			continue // not representable on this platform
		}
		n := int(c)
		got := classOf(n)
		if got < 0 {
			t.Errorf("classOf(%d) = %d, overflowed negative", n, got)
		}
		if got < n {
			t.Errorf("classOf(%d) = %d, want >= %d", n, got, n)
		}
		if int64(got) > maxValueLen {
			t.Errorf("classOf(%d) = %d, want <= maxValueLen (%d)", n, got, maxValueLen)
		}
	}
}
