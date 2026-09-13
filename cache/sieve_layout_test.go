// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"testing"
	"unsafe"
)

// TestSieveFieldShareLineWithIsMmap pins a LAYOUT property the read path depends
// on for its cost, not for its correctness.
//
// indexTable.get tests s.sieve on every hit. getCore has already read s.isMmap by
// then (via seqlockReads -> inPlaceEligible), so if the two fields share a cache
// line the test is free; if they do not, every hit pulls an extra line for one
// bool. That is a real cost on a path whose whole budget is a few nanoseconds, and
// it is invisible in the source — a field added at the wrong place in a large
// struct looks identical to one added at the right place.
func TestSieveFieldSharesLineWithIsMmap(t *testing.T) {
	var s shard
	sieveOff := unsafe.Offsetof(s.sieve)
	mmapOff := unsafe.Offsetof(s.isMmap)
	t.Logf("isMmap at %d, sieve at %d (lines %d and %d)", mmapOff, sieveOff, mmapOff/64, sieveOff/64)
	if sieveOff/64 != mmapOff/64 {
		t.Errorf("s.sieve is on cache line %d but s.isMmap is on line %d; "+
			"the read path loads isMmap already, so sieve must ride the same line",
			sieveOff/64, mmapOff/64)
	}
}
