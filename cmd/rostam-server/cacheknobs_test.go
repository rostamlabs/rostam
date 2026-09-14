// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"strings"
	"testing"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cache"
)

// resolveCacheKnobs runs the same sequence main does — parse, fill from the
// environment, apply — against a fresh flag set.
func resolveCacheKnobs(t *testing.T, args []string, vars map[string]string) (rostam.CacheConfig, error) {
	t.Helper()
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	k := registerCacheKnobFlags(fs)
	if err := fs.Parse(args); err != nil {
		t.Fatalf("parse %v: %v", args, err)
	}
	if err := applyEnvDefaults(fs, env(vars)); err != nil {
		t.Fatalf("env %v: %v", vars, err)
	}
	var cc rostam.CacheConfig
	err := k.apply(&cc, env(vars))
	return cc, err
}

func TestCacheKnobFlagsDefaultOff(t *testing.T) {
	cc, err := resolveCacheKnobs(t, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if cc != (rostam.CacheConfig{}) {
		t.Errorf("no flags set, but the cache config is not zero: %+v", cc)
	}
}

func TestCacheKnobFlagsLandInConfig(t *testing.T) {
	want := rostam.CacheConfig{
		RelocatingEviction:        true,
		RelocateReserveIntervalMs: 200,
		InPlaceSameSizeUpdate:     true,
		InPlaceSeqlockReads:       true,
		SieveVisitedBit:           true,
	}
	t.Run("flags", func(t *testing.T) {
		cc, err := resolveCacheKnobs(t, []string{
			"-relocating-eviction",
			"-relocate-reserve-interval", "200ms",
			"-in-place-same-size-update",
			"-in-place-seqlock-reads",
			"-sieve-visited-bit",
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if cc != want {
			t.Errorf("got %+v, want %+v", cc, want)
		}
	})
	t.Run("environment", func(t *testing.T) {
		cc, err := resolveCacheKnobs(t, nil, map[string]string{
			"ROSTAM_RELOCATING_EVICTION":       "true",
			"ROSTAM_RELOCATE_RESERVE_INTERVAL": "200ms",
			"ROSTAM_IN_PLACE_SAME_SIZE_UPDATE": "true",
			"ROSTAM_IN_PLACE_SEQLOCK_READS":    "true",
			"ROSTAM_SIEVE_VISITED_BIT":         "true",
		})
		if err != nil {
			t.Fatal(err)
		}
		if cc != want {
			t.Errorf("got %+v, want %+v", cc, want)
		}
	})
	// Each flag on its own sets its own field and no other.
	for _, tc := range []struct {
		arg  string
		want rostam.CacheConfig
	}{
		{"-relocating-eviction", rostam.CacheConfig{RelocatingEviction: true}},
		{"-in-place-same-size-update", rostam.CacheConfig{InPlaceSameSizeUpdate: true}},
		{"-in-place-seqlock-reads", rostam.CacheConfig{InPlaceSeqlockReads: true}},
		{"-sieve-visited-bit", rostam.CacheConfig{SieveVisitedBit: true}},
		{"-relocate-reserve-interval=75ms", rostam.CacheConfig{RelocateReserveIntervalMs: 75}},
	} {
		t.Run(tc.arg, func(t *testing.T) {
			cc, err := resolveCacheKnobs(t, []string{tc.arg}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if cc != tc.want {
				t.Errorf("got %+v, want %+v", cc, tc.want)
			}
		})
	}
}

// The interval follows -ttl-sweep-interval: 0 turns it off (the public config's -1),
// a sub-millisecond value floors to 1ms rather than collapsing to "library default",
// a negative one is refused, and an unset flag passes nothing at all.
func TestRelocateReserveIntervalFlag(t *testing.T) {
	for _, tc := range []struct {
		val  string
		want int
	}{
		{"0", -1},
		{"500us", 1},
		{"2s", 2000},
	} {
		cc, err := resolveCacheKnobs(t, []string{"-relocate-reserve-interval", tc.val}, nil)
		if err != nil {
			t.Fatalf("%s: %v", tc.val, err)
		}
		if cc.RelocateReserveIntervalMs != tc.want {
			t.Errorf("-relocate-reserve-interval %s gave %d, want %d", tc.val, cc.RelocateReserveIntervalMs, tc.want)
		}
	}
	if _, err := resolveCacheKnobs(t, []string{"-relocate-reserve-interval", "-1s"}, nil); err == nil {
		t.Error("a negative -relocate-reserve-interval was accepted")
	}
	cc, err := resolveCacheKnobs(t, nil, map[string]string{"ROSTAM_RELOCATE_RESERVE_INTERVAL": "0"})
	if err != nil {
		t.Fatal(err)
	}
	if cc.RelocateReserveIntervalMs != -1 {
		t.Errorf("ROSTAM_RELOCATE_RESERVE_INTERVAL=0 gave %d, want -1", cc.RelocateReserveIntervalMs)
	}
}

// The help shows the library's own default rather than a copy of it that could drift.
func TestRelocateReserveIntervalFlagDefaultIsTheLibraryDefault(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	registerCacheKnobFlags(fs)
	want := time.Duration(cache.DefaultConfig().RelocateReserveIntervalMs) * time.Millisecond
	if got := fs.Lookup(relocateReserveIntervalFlag).DefValue; got != want.String() {
		t.Errorf("flag default = %s, want %s", got, want)
	}
}

// The race-detection caveat must survive the grouped -h listing, which keeps only
// each description's first sentence.
func TestSeqlockReadsHelpLeadsWithTheRaceCaveat(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	registerCacheKnobFlags(fs)
	summary := summarize(fs.Lookup("in-place-seqlock-reads").Usage, 96)
	if !strings.Contains(summary, "race detection") {
		t.Errorf("grouped help for -in-place-seqlock-reads drops the race caveat: %q", summary)
	}
}
