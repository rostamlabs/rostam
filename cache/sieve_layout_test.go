// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"testing"
	"unsafe"
)

// sieveFlagMaxOffsetGap is how far s.sieve may sit from s.isMmap, in bytes.
//
// It is deliberately a small ADJACENCY bound rather than a cache-line calculation.
// Line size is not 64 everywhere — 128 is normal on arm64, including Apple silicon
// and several server parts — and Go does not align a struct to a line anyway, so
// `offsetof(a)/64 == offsetof(b)/64` neither holds portably nor implies the two
// fields share a line at run time. Adjacency does imply it, at every line size and
// whatever the struct's base alignment: two fields within 8 bytes of each other
// straddle a boundary only in the one case where the boundary falls between them,
// and can never be a whole line apart.
const sieveFlagMaxOffsetGap = 8

// TestSieveFlagSitsBesideIsMmap pins a LAYOUT property the read path depends on for
// its COST, not for its correctness — which is why it is asserted here rather than
// left to be rediscovered as an unexplained regression.
//
// indexTable.get tests s.sieve on every hit. getCore has already read s.isMmap by
// then (via seqlockReads -> inPlaceEligible), so while the two fields sit together
// the test is free; separate them and every hit pulls an additional line to read one
// bool. Nothing in the source makes that visible: a field added at the wrong place in
// a large struct looks exactly like a field added at the right place, and the cost
// shows up only as a benchmark moving for no stated reason.
func TestSieveFlagSitsBesideIsMmap(t *testing.T) {
	var s shard
	sieveOff := unsafe.Offsetof(s.sieve)
	mmapOff := unsafe.Offsetof(s.isMmap)
	gap := sieveOff - mmapOff
	if sieveOff < mmapOff {
		gap = mmapOff - sieveOff
	}
	t.Logf("isMmap at offset %d, sieve at offset %d, gap %d bytes", mmapOff, sieveOff, gap)
	if gap > sieveFlagMaxOffsetGap {
		t.Errorf("s.sieve is %d bytes from s.isMmap (allowed %d): the read path reads "+
			"isMmap on every hit and tests sieve on every hit, so they must share a line; "+
			"move the field back beside it rather than relaxing this bound",
			gap, sieveFlagMaxOffsetGap)
	}
}
