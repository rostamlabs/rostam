// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseSize(t *testing.T) {
	tests := []struct {
		in   string
		want int64
	}{
		{"1024", 1024},
		{"512MiB", 512 << 20},
		{"8GiB", 8 << 30},
		{"2TiB", 2 << 40},
		{"64KiB", 64 << 10},
		// SI spellings mean decimal, not a silently-rounded IEC value.
		{"2GB", 2e9},
		{"500MB", 500e6},
		// Case and surrounding space are not significant.
		{"  8gib ", 8 << 30},
		{"8GIB", 8 << 30},
		// Bare single-letter suffixes are IEC by convention.
		{"8G", 8 << 30},
		{"512M", 512 << 20},
		{"4096B", 4096},
	}
	for _, tc := range tests {
		got, err := parseSize(tc.in)
		if err != nil {
			t.Errorf("parseSize(%q): unexpected error %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseSize(%q) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestParseSizeRejects(t *testing.T) {
	for _, in := range []string{
		"", "   ", "abc", "GiB", "-5GiB", "0", "0GiB", "8 GiB extra", "1.5GiB", "9223372036854775807GiB",
	} {
		if got, err := parseSize(in); err == nil {
			t.Errorf("parseSize(%q) = %d, want an error", in, got)
		}
	}
}

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadFileConfigCacheMaxMemory(t *testing.T) {
	p := writeConfig(t, `{"cache":{"max_memory":"8GiB"}}`)
	fc, err := loadFileConfig(p)
	if err != nil {
		t.Fatalf("loadFileConfig: %v", err)
	}
	got, err := fc.cacheMaxMemoryBytes()
	if err != nil {
		t.Fatalf("cacheMaxMemoryBytes: %v", err)
	}
	if want := int64(8 << 30); got != want {
		t.Errorf("cacheMaxMemoryBytes = %d, want %d", got, want)
	}
}

// Absent stanza => 0 => the engine derives a budget from the host. This is the
// default path for every existing deployment, so it must not error.
func TestLoadFileConfigEmptyMeansDerive(t *testing.T) {
	for _, body := range []string{`{}`, `{"cache":{}}`, `{"cache":{"max_memory":""}}`} {
		fc, err := loadFileConfig(writeConfig(t, body))
		if err != nil {
			t.Fatalf("loadFileConfig(%s): %v", body, err)
		}
		got, err := fc.cacheMaxMemoryBytes()
		if err != nil {
			t.Fatalf("cacheMaxMemoryBytes(%s): %v", body, err)
		}
		if got != 0 {
			t.Errorf("cacheMaxMemoryBytes(%s) = %d, want 0 (derive from host)", body, got)
		}
	}
}

// A typo'd knob must fail loudly. Silently ignoring it would leave the node on
// a memory bound the operator did not intend — the exact failure this config
// stanza exists to prevent.
// The cache eviction knobs are flags and environment variables only, and the docs
// tell operators a config file naming one fails to load. Pin that, so a future file
// key cannot quietly give one setting two homes.
func TestLoadFileConfigRejectsCacheEvictionKnobs(t *testing.T) {
	for _, key := range []string{
		"relocating_eviction", "relocate_reserve_interval", "in_place_same_size_update",
		"in_place_seqlock_reads", "sieve_visited_bit",
		"RelocatingEviction", "RelocateReserveIntervalMs", "InPlaceSameSizeUpdate",
		"InPlaceSeqlockReads", "SieveVisitedBit",
	} {
		body := `{"cache":{"` + key + `":true}}`
		if _, err := loadFileConfig(writeConfig(t, body)); err == nil {
			t.Errorf("loadFileConfig(%s) = nil error, want a rejection", body)
		}
	}
}

func TestLoadFileConfigRejectsUnknownFields(t *testing.T) {
	for _, body := range []string{
		`{"cache":{"max_memmory":"8GiB"}}`,
		`{"cashe":{"max_memory":"8GiB"}}`,
		`{"cache":{"max_memory":"8GiB"},"bogus":1}`,
	} {
		if _, err := loadFileConfig(writeConfig(t, body)); err == nil {
			t.Errorf("loadFileConfig(%s) = nil error, want a rejection", body)
		}
	}
}

func TestLoadFileConfigErrors(t *testing.T) {
	if _, err := loadFileConfig(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("loadFileConfig(missing file) = nil error, want a rejection")
	}
	if _, err := loadFileConfig(writeConfig(t, `{not json`)); err == nil {
		t.Error("loadFileConfig(malformed) = nil error, want a rejection")
	}
	fc, err := loadFileConfig(writeConfig(t, `{"cache":{"max_memory":"banana"}}`))
	if err != nil {
		t.Fatalf("loadFileConfig: %v", err)
	}
	if _, err := fc.cacheMaxMemoryBytes(); err == nil {
		t.Error("cacheMaxMemoryBytes(banana) = nil error, want a rejection")
	}
}
