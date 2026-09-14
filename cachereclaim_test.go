// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
)

// The knobs must arrive on cache.Config. Asserting that loadFileConfig parses
// them would pass with no wiring at all, which is the failure this guards.
func TestApplyRingbufReclaimCarriesEveryKnob(t *testing.T) {
	cc := cache.DefaultConfig()
	applyRingbufReclaim(&cc, CacheConfig{
		RelocatingEviction:        true,
		SieveVisitedBit:           true,
		InPlaceSameSizeUpdate:     true,
		InPlaceSeqlockReads:       true,
		RelocateReserveIntervalMs: 17,
	})
	for _, c := range []struct {
		name string
		got  bool
	}{
		{"RelocatingEviction", cc.RelocatingEviction},
		{"SieveVisitedBit", cc.SieveVisitedBit},
		{"InPlaceSameSizeUpdate", cc.InPlaceSameSizeUpdate},
		{"InPlaceSeqlockReads", cc.InPlaceSeqlockReads},
	} {
		if !c.got {
			t.Errorf("cache.Config.%s = false, want true", c.name)
		}
	}
	if cc.RelocateReserveIntervalMs != 17 {
		t.Errorf("RelocateReserveIntervalMs = %d, want 17", cc.RelocateReserveIntervalMs)
	}
}

// A zero CacheConfig must leave every feature off and the reserve cadence on
// the cache's own default, so an operator who sets none of this keeps today's
// behaviour exactly.
func TestApplyRingbufReclaimZeroValueChangesNothing(t *testing.T) {
	def := cache.DefaultConfig()
	cc := cache.DefaultConfig()
	applyRingbufReclaim(&cc, CacheConfig{})

	if cc.RelocatingEviction || cc.SieveVisitedBit || cc.InPlaceSameSizeUpdate || cc.InPlaceSeqlockReads {
		t.Errorf("zero CacheConfig enabled a feature: %+v", cc)
	}
	if cc.RelocateReserveIntervalMs != def.RelocateReserveIntervalMs {
		t.Errorf("RelocateReserveIntervalMs = %d, want the cache default %d",
			cc.RelocateReserveIntervalMs, def.RelocateReserveIntervalMs)
	}
}

// Negative means "run no reserve ticker", matching TTLSweepIntervalMs. Without
// the sentinel there is no way to express it: 0 already means "use the default",
// and the default is non-zero.
func TestApplyRingbufReclaimNegativeReserveDisablesTicker(t *testing.T) {
	cc := cache.DefaultConfig()
	if cc.RelocateReserveIntervalMs == 0 {
		t.Fatal("cache default RelocateReserveIntervalMs is 0; this test cannot distinguish disabled from default")
	}
	applyRingbufReclaim(&cc, CacheConfig{RelocateReserveIntervalMs: -1})
	if cc.RelocateReserveIntervalMs != 0 {
		t.Errorf("RelocateReserveIntervalMs = %d, want 0 (no ticker)", cc.RelocateReserveIntervalMs)
	}
}
