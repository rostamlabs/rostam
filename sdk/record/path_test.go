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
