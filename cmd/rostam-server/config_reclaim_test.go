// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestLoadFileConfigRingbufReclaimKnobs(t *testing.T) {
	p := writeConfig(t, `{"cache":{"max_memory":"1GiB",
		"relocating_eviction":true,
		"relocate_reserve_interval_ms":25,
		"sieve_visited_bit":true,
		"inplace_same_size_update":true,
		"inplace_seqlock_reads":true}}`)
	fc, err := loadFileConfig(p)
	if err != nil {
		t.Fatalf("loadFileConfig: %v", err)
	}
	if !fc.Cache.RelocatingEviction || !fc.Cache.SieveVisitedBit ||
		!fc.Cache.InPlaceSameSizeUpdate || !fc.Cache.InPlaceSeqlockReads {
		t.Errorf("a flag did not decode: %+v", fc.Cache)
	}
	if fc.Cache.RelocateReserveIntervalMs != 25 {
		t.Errorf("RelocateReserveIntervalMs = %d, want 25", fc.Cache.RelocateReserveIntervalMs)
	}
}

// Absent keys must decode false, so an existing config file keeps its behaviour.
func TestLoadFileConfigReclaimAbsentIsOff(t *testing.T) {
	fc, err := loadFileConfig(writeConfig(t, `{"cache":{"max_memory":"1GiB"}}`))
	if err != nil {
		t.Fatalf("loadFileConfig: %v", err)
	}
	if fc.Cache.RelocatingEviction || fc.Cache.SieveVisitedBit ||
		fc.Cache.InPlaceSameSizeUpdate || fc.Cache.InPlaceSeqlockReads ||
		fc.Cache.RelocateReserveIntervalMs != 0 {
		t.Errorf("absent keys did not decode to off: %+v", fc.Cache)
	}
}

// The misspelling an operator actually makes must fail loudly rather than run
// with the feature silently off.
func TestLoadFileConfigRejectsMisspelledReclaimKnob(t *testing.T) {
	for _, body := range []string{
		`{"cache":{"relocating_evictions":true}}`,
		`{"cache":{"in_place_same_size_update":true}}`,
		`{"cache":{"sieve_visited":true}}`,
	} {
		if _, err := loadFileConfig(writeConfig(t, body)); err == nil {
			t.Errorf("loadFileConfig(%s) = nil error, want a rejection", body)
		}
	}
}
