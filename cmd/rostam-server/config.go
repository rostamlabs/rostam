// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// fileConfig mirrors the -config JSON document. JSON (not YAML) to match the
// existing -keys-file precedent and to avoid a new dependency.
//
// The cache stanza carries knobs that do not belong on a ~40-flag command line
// and that need to be set together. It deliberately does NOT duplicate any
// existing flag, so there is no flag-vs-file precedence to reason about: a
// field lives in exactly one place.
type fileConfig struct {
	Cache cacheFileConfig `json:"cache"`
}

type cacheFileConfig struct {
	// MaxMemory bounds TOTAL cache memory for this node across every shard,
	// as a size string ("8GiB", "512MiB", "2GB") or a plain byte count.
	// Empty/absent = derive from host RAM.
	//
	// This is the bound that stops a write-heavy node eating the box: Put is
	// append-only, so memory climbs toward this cap even when the live key set
	// is small, and recycling only begins once it is reached.
	MaxMemory string `json:"max_memory"`

	// RelocatingEviction carries live records off a page before it drains, so a
	// record is not dropped merely for sharing a page with other keys' superseded
	// versions. Ringbuf shards only; default false.
	RelocatingEviction bool `json:"relocating_eviction"`

	// RelocateReserveIntervalMs is the free-page reserve cadence in ms, which moves
	// most of relocation's copying off the write path. Inert unless
	// relocating_eviction is set. Absent/0 keeps the engine default; negative runs
	// no reserve.
	RelocateReserveIntervalMs int `json:"relocate_reserve_interval_ms"`

	// SieveVisitedBit rescues records that have been read or rewritten since the
	// last drain passed them, rather than whichever the walk meets first. No effect
	// unless relocating_eviction is set.
	SieveVisitedBit bool `json:"sieve_visited_bit"`

	// InPlaceSameSizeUpdate writes a same-size rewrite over the copy already
	// stored instead of appending a new one. Default false. Read
	// cache.Config.InPlaceSameSizeUpdate before enabling: it trades the recency
	// a ring buffer gets for free, and matters on a shard whose live set exceeds
	// its budget.
	InPlaceSameSizeUpdate bool `json:"inplace_same_size_update"`

	// InPlaceSeqlockReads keeps reads lock-free under inplace_same_size_update by
	// validating each against a version counter, instead of taking the read lock.
	InPlaceSeqlockReads bool `json:"inplace_seqlock_reads"`
}

// loadFileConfig reads and validates the -config document. Unknown fields are
// rejected: a typo'd knob in a config file that silently does nothing is how a
// node ends up running an unintended memory bound.
func loadFileConfig(path string) (fileConfig, error) {
	var fc fileConfig
	b, err := os.ReadFile(path)
	if err != nil {
		return fc, fmt.Errorf("rostam-server: -config: %w", err)
	}
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&fc); err != nil {
		return fc, fmt.Errorf("rostam-server: -config %s: %w", path, err)
	}
	return fc, nil
}

// cacheMaxMemoryBytes resolves the cache stanza's MaxMemory to bytes.
// Zero means "unset" — the engine then derives a budget from host RAM.
func (fc fileConfig) cacheMaxMemoryBytes() (int64, error) {
	if strings.TrimSpace(fc.Cache.MaxMemory) == "" {
		return 0, nil
	}
	n, err := parseSize(fc.Cache.MaxMemory)
	if err != nil {
		return 0, fmt.Errorf("rostam-server: -config cache.max_memory: %w", err)
	}
	return n, nil
}

// sizeUnits maps a suffix to its multiplier. Both IEC (KiB=1024) and the
// decimal SI spellings (KB=1000) are accepted and mean what they say, so an
// operator who writes 8GB gets 8e9 rather than a silent 7.45 GiB.
var sizeUnits = []struct {
	suffix string
	mult   int64
}{
	{"KIB", 1 << 10}, {"MIB", 1 << 20}, {"GIB", 1 << 30}, {"TIB", 1 << 40},
	{"KB", 1e3}, {"MB", 1e6}, {"GB", 1e9}, {"TB", 1e12},
	{"K", 1 << 10}, {"M", 1 << 20}, {"G", 1 << 30}, {"T", 1 << 40},
	{"B", 1},
}

// parseSize parses a byte size: a plain integer, or an integer with an IEC
// ("512MiB") or SI ("2GB") suffix. Case-insensitive; surrounding space ignored.
func parseSize(s string) (int64, error) {
	t := strings.ToUpper(strings.TrimSpace(s))
	if t == "" {
		return 0, fmt.Errorf("empty size")
	}
	for _, u := range sizeUnits {
		if !strings.HasSuffix(t, u.suffix) {
			continue
		}
		num := strings.TrimSpace(strings.TrimSuffix(t, u.suffix))
		if num == "" {
			return 0, fmt.Errorf("%q: missing number before %q", s, u.suffix)
		}
		n, err := strconv.ParseInt(num, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("%q: %w", s, err)
		}
		if n <= 0 {
			return 0, fmt.Errorf("%q: must be > 0", s)
		}
		// Reject overflow rather than wrap into a nonsense bound.
		if n > (1<<62)/u.mult {
			return 0, fmt.Errorf("%q: overflows int64", s)
		}
		return n * u.mult, nil
	}
	n, err := strconv.ParseInt(t, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%q: not a byte size (want e.g. 8GiB, 512MiB, 2GB, or a byte count)", s)
	}
	if n <= 0 {
		return 0, fmt.Errorf("%q: must be > 0", s)
	}
	return n, nil
}
