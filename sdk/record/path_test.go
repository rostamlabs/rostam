// SPDX-License-Identifier: Apache-2.0

package record

import (
	"errors"
	"reflect"
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
		"256-byte name":                      longName,
		"21-digit row key":                   "a/123456789012345678901/c",
		"row key overflows uint64":           "a/18446744073709551616/c",
		"name containing a quote":            `a"b`,
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
