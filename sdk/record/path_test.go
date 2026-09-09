// SPDX-License-Identifier: Apache-2.0

package record

import (
	"errors"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestParsePathAccepted(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want Path
	}{
		{
			name: "bare name field",
			in:   "rc",
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "rc"},
			}},
		},
		{
			name: "positional field",
			in:   "#2",
			want: Path{Segs: []Segment{
				{Kind: SegField, ByPos: true, Pos: 2},
			}},
		},
		{
			name: "field row col, decimal row key",
			in:   "b/42/t",
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "b"},
				{Kind: SegRow, KeyText: "42"},
				{Kind: SegCol, Name: "t"},
			}},
		},
		{
			name: "field row col, quoted row key",
			in:   `b/"DE"/hits`,
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "b"},
				{Kind: SegRow, Key: []byte("DE"), KeyText: `"DE"`, KeyQuoted: true},
				{Kind: SegCol, Name: "hits"},
			}},
		},
		{
			name: "field count",
			in:   "b#count",
			want: Path{Segs: []Segment{
				{Kind: SegCount, Name: "b"},
			}},
		},
		{
			name: "field and row, no column",
			in:   "b/42",
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "b"},
				{Kind: SegRow, KeyText: "42"},
			}},
		},
		{
			name: "quoted row key with escapes",
			in:   `a/"a\"b"/c`,
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "a"},
				{Kind: SegRow, Key: []byte(`a"b`), KeyText: `"a\"b"`, KeyQuoted: true},
				{Kind: SegCol, Name: "c"},
			}},
		},
		{
			name: "positional count",
			in:   "#5#count",
			want: Path{Segs: []Segment{
				{Kind: SegCount, ByPos: true, Pos: 5},
			}},
		},
		{
			name: "max position value",
			in:   "#65535",
			want: Path{Segs: []Segment{
				{Kind: SegField, ByPos: true, Pos: 65535},
			}},
		},
		{
			// "#0" is the one position that legally starts with '0': a
			// single digit is canonical by definition.
			name: "zero position",
			in:   "#0",
			want: Path{Segs: []Segment{
				{Kind: SegField, ByPos: true, Pos: 0},
			}},
		},
		{
			name: "canonical positional column",
			in:   "a/1/#12345",
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "a"},
				{Kind: SegRow, KeyText: "1"},
				{Kind: SegCol, ByPos: true, Pos: 12345},
			}},
		},
		{
			name: "max-length decimal row key (20 digits, fits uint64)",
			in:   "a/18446744073709551615/c",
			want: Path{Segs: []Segment{
				{Kind: SegField, Name: "a"},
				{Kind: SegRow, KeyText: "18446744073709551615"},
				{Kind: SegCol, Name: "c"},
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParsePath(tc.in)
			if err != nil {
				t.Fatalf("ParsePath(%q) unexpected error: %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ParsePath(%q) = %+v, want %+v", tc.in, got, tc.want)
			}
		})
	}
}

func TestParsePathRejected(t *testing.T) {
	longName := strings.Repeat("x", 256)

	cases := map[string]string{
		"empty path":                         "",
		"trailing slash":                     "a/",
		"leading slash":                      "/a",
		"too many segments":                  "a/b/c/d",
		"count not sole segment":             "a#count/x",
		"unterminated quote":                 `a/"unterminated`,
		"row key neither decimal nor quoted": "a/x/y",
		"position used as row key":           "a/#3",
		"position out of range":              "#70000",
		// Non-canonical spellings of positions that are in range. The value
		// they denote is legal; the SPELLING is not, so that ParsePath's
		// pre-split length bound and the "#N" grammar cannot disagree about
		// the same string (a long enough run of leading zeros exceeded
		// maxPathBytes while parseNameOrPos still accepted the digits).
		"position with a leading zero":      "#01",
		"position padded with zeros":        "#000001",
		"position with six digits":          "#123456",
		"positional column with a zero pad": "a/1/#01",
		"256-byte name":                     longName,
		"21-digit row key":                  "a/123456789012345678901/c",
		"row key overflows uint64":          "a/18446744073709551616/c",
		"name containing a quote":           `a"b`,
	}

	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParsePath(in)
			if err == nil {
				t.Fatalf("ParsePath(%q) expected error, got nil", in)
			}
			if !errors.Is(err, ErrPath) {
				t.Fatalf("ParsePath(%q) error %v does not wrap ErrPath", in, err)
			}
		})
	}
}

func TestSplitField(t *testing.T) {
	cases := []struct {
		name           string
		in             string
		wantPayloadKey string
		wantPath       string
		wantOK         bool
	}{
		{name: "field with path", in: "session/rc", wantPayloadKey: "session", wantPath: "rc", wantOK: true},
		{name: "no slash", in: "plain", wantPayloadKey: "", wantPath: "", wantOK: false},
		{name: "leading slash yields empty payload key", in: "/x", wantPayloadKey: "", wantPath: "x", wantOK: true},
		{name: "empty string", in: "", wantPayloadKey: "", wantPath: "", wantOK: false},
		{name: "splits at first slash only", in: "a/b/c", wantPayloadKey: "a", wantPath: "b/c", wantOK: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			payloadKey, path, ok := SplitField(tc.in)
			if payloadKey != tc.wantPayloadKey || path != tc.wantPath || ok != tc.wantOK {
				t.Fatalf("SplitField(%q) = (%q, %q, %v), want (%q, %q, %v)",
					tc.in, payloadKey, path, ok, tc.wantPayloadKey, tc.wantPath, tc.wantOK)
			}
		})
	}
}

// TestParsePathQuotedRowKeyLengthBound pins the bound on a quoted row key.
//
// WHY IT IS A BOUND AND NOT JUST A VALIDATION. unescapeQuoted allocates the raw
// inner length, and a filter's path is parsed against the record of every point
// in a scan, so an unbounded quoted key is a per-row allocation the caller sizes:
// a field of session/"<30 MiB of x>"/col cost ~30 MB per point. The raw length is
// therefore checked BEFORE the unescape (every escape is two bytes producing one,
// so 2*OperateMaxKeyLen raw is the widest input that could still be legal), and
// the unescaped length is checked after.
//
// The bound is semantically free: a row key wider than OperateMaxKeyLen cannot
// exist in any encodable record — schema mode answers Absent on a key-width
// mismatch and a dynamic-mode key length is a u8 — so rejecting means exactly
// what resolving would have meant.
func TestParsePathQuotedRowKeyLengthBound(t *testing.T) {
	quoted := func(n int) string { return `a/"` + strings.Repeat("k", n) + `"/c` }

	t.Run("255 accepted", func(t *testing.T) {
		p, err := ParsePath(quoted(255))
		if err != nil {
			t.Fatalf("ParsePath with a 255-byte quoted row key: %v, want nil", err)
		}
		if len(p.Segs) != 3 || p.Segs[1].Kind != SegRow || len(p.Segs[1].Key) != 255 {
			t.Fatalf("unexpected parse: %+v", p)
		}
	})

	t.Run("256 rejected", func(t *testing.T) {
		_, err := ParsePath(quoted(256))
		if err == nil || !errors.Is(err, ErrPath) {
			t.Fatalf("ParsePath with a 256-byte quoted row key: err = %v, want an ErrPath", err)
		}
	})

	// The hostile case: rejected on the RAW length, so nothing near this size is
	// ever copied. A test that only asserted the error would pass even if the
	// unescape still ran, so this one is the reason the raw check exists.
	t.Run("1 MiB rejected", func(t *testing.T) {
		_, err := ParsePath(quoted(1 << 20))
		if err == nil || !errors.Is(err, ErrPath) {
			t.Fatalf("ParsePath with a 1 MiB quoted row key: err = %v, want an ErrPath", err)
		}
	})

	// Escapes count against the UNESCAPED length: 255 escaped pairs are 510 raw
	// bytes and a legal 255-byte key, while 256 pairs are not. This is the case a
	// raw-length-only bound would get wrong in both directions.
	t.Run("escapes count unescaped", func(t *testing.T) {
		esc := func(n int) string { return `a/"` + strings.Repeat(`\"`, n) + `"/c` }
		p, err := ParsePath(esc(255))
		if err != nil {
			t.Fatalf("255 escaped pairs: %v, want nil", err)
		}
		if len(p.Segs[1].Key) != 255 {
			t.Fatalf("unescaped key length = %d, want 255", len(p.Segs[1].Key))
		}
		if _, err := ParsePath(esc(256)); err == nil || !errors.Is(err, ErrPath) {
			t.Fatalf("256 escaped pairs: err = %v, want an ErrPath", err)
		}
	})
}

// parsePathSink keeps the ParsePath results below from being optimized away.
var parsePathSink error

// TestParsePathLengthBoundBeforeSplit pins the bound ParsePath applies AHEAD of
// strings.Split.
//
// Split allocates one []string header per segment and only then can the
// segment-count check reject the input, so before this bound an input carrying
// N slashes cost roughly 16N bytes of transient headers before rejection — once
// per filter leaf per request, under a 32 MiB route body cap. The segment-level
// bounds (validateName's 255, parseRowSeg's quoted-key cap) cannot help: both
// live in helpers that run after the split.
func TestParsePathLengthBoundBeforeSplit(t *testing.T) {
	// The longest path the grammar allows, spelled out exactly: a 255-byte
	// field name, a quoted row key at its raw cap (255 escaped pairs, 510 bytes
	// inside the quotes, unescaping to a legal 255-byte key), and a 255-byte
	// column name. This is the input a bound set one byte too low would break.
	longest := strings.Repeat("f", maxNameLen) + `/"` + strings.Repeat(`\"`, 255) + `"/` + strings.Repeat("c", maxNameLen)
	if len(longest) != maxPathBytes {
		t.Fatalf("the longest legal path is %d bytes but maxPathBytes is %d — the derivation and the constant disagree",
			len(longest), maxPathBytes)
	}
	p, err := ParsePath(longest)
	if err != nil {
		t.Fatalf("the longest legal path was rejected: %v", err)
	}
	if len(p.Segs) != 3 || len(p.Segs[1].Key) != 255 {
		t.Fatalf("unexpected parse of the longest legal path: %+v", p)
	}

	// One byte over is rejected, and with an ErrPath like every other rejection.
	if _, err := ParsePath(longest + "x"); err == nil || !errors.Is(err, ErrPath) {
		t.Fatalf("maxPathBytes+1: err = %v, want an ErrPath", err)
	}

	// The hostile input: a 1 MiB field that is almost entirely '/'.
	hostile := strings.Repeat("a/", 512<<10)
	if _, err := ParsePath(hostile); err == nil || !errors.Is(err, ErrPath) {
		t.Fatalf("a 1 MiB field of slashes: err = %v, want an ErrPath", err)
	}

	// The point of the fix, measured in BYTES rather than allocation count:
	// strings.Split makes one big allocation for the whole []string, so the
	// count barely moves while the bytes move by 16x the slash count. A 1 MiB
	// field of slashes carries 512Ki segments, which is 8 MiB of headers.
	small := strings.Repeat("a/", 32<<10) // 64 KiB, 32Ki slashes
	smallBytes := bytesPerParse(small)
	bigBytes := bytesPerParse(hostile)
	const byteCeiling = 4 << 10 // the error value and its formatting; nothing per segment
	if bigBytes > byteCeiling {
		t.Errorf("rejecting a 1 MiB field allocated %d bytes, want at most %d — the split still runs first",
			bigBytes, byteCeiling)
	}
	// And it does not scale: a 16x larger input must not cost 16x the bytes.
	if bigBytes > smallBytes+byteCeiling {
		t.Errorf("rejection bytes scale with input: %d for 64 KiB vs %d for 1 MiB", smallBytes, bigBytes)
	}

	// Allocation COUNT is asserted too, since a future rewrite could split into
	// many small allocations instead of one big one.
	const allocCeiling = 8 // the error value and its formatting
	if got := testing.AllocsPerRun(50, func() { _, parsePathSink = ParsePath(hostile) }); got > allocCeiling {
		t.Errorf("rejecting a 1 MiB field allocated %.0f times, want at most %d", got, allocCeiling)
	}
}

// bytesPerParse reports the average heap bytes one rejected ParsePath call
// allocates. runtime.MemStats rather than AllocsPerRun because the cost this
// test is about is BYTES: strings.Split allocates the whole []string at once,
// so a per-segment cost is invisible in the allocation count.
func bytesPerParse(s string) uint64 {
	const runs = 20
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	for i := 0; i < runs; i++ {
		_, parsePathSink = ParsePath(s)
	}
	runtime.ReadMemStats(&after)
	return (after.TotalAlloc - before.TotalAlloc) / runs
}
