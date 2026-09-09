// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestKVIndexDefRoundtrip(t *testing.T) {
	def := KVIndexDef{Name: "by_rc", KeyPrefix: []byte("sess:"), PayloadPath: "rc", Kind: KVIndexKindScalar, Enabled: true}
	b := AppendKVIndexDef(nil, def)
	got, n, err := DecodeKVIndexDef(b)
	if err != nil {
		t.Fatalf("DecodeKVIndexDef: %v", err)
	}
	if n != len(b) {
		t.Fatalf("n = %d, want %d (len(b))", n, len(b))
	}
	if !reflect.DeepEqual(got, def) {
		t.Fatalf("got %+v, want %+v", got, def)
	}

	// Empty prefix (indexes the whole keyspace) and a #count path round-trip
	// too.
	def2 := KVIndexDef{Name: "sessions", PayloadPath: "hits#count", Kind: KVIndexKindCount, Enabled: false}
	b2 := AppendKVIndexDef(nil, def2)
	got2, n2, err := DecodeKVIndexDef(b2)
	if err != nil {
		t.Fatalf("DecodeKVIndexDef (empty prefix): %v", err)
	}
	if n2 != len(b2) {
		t.Fatalf("n2 = %d, want %d", n2, len(b2))
	}
	if !reflect.DeepEqual(got2, def2) {
		t.Fatalf("got %+v, want %+v", got2, def2)
	}
}

func TestKVIndexDefValidate(t *testing.T) {
	base := func() KVIndexDef {
		return KVIndexDef{Name: "idx", KeyPrefix: []byte("k:"), PayloadPath: "rc", Kind: KVIndexKindScalar}
	}

	cases := []struct {
		name    string
		mut     func(d *KVIndexDef)
		wantErr bool
	}{
		{"empty name", func(d *KVIndexDef) { d.Name = "" }, true},
		{"65-byte name", func(d *KVIndexDef) { d.Name = strings.Repeat("a", 65) }, true},
		{"64-byte name ok", func(d *KVIndexDef) { d.Name = strings.Repeat("a", 64) }, false},
		{"scalar path with #count suffix", func(d *KVIndexDef) {
			d.PayloadPath, d.Kind = "b#count", KVIndexKindScalar
		}, true},
		{"count path with #count suffix", func(d *KVIndexDef) {
			d.PayloadPath, d.Kind = "b#count", KVIndexKindCount
		}, false},
		{"row path rejected", func(d *KVIndexDef) { d.PayloadPath = "b/42/hi" }, true},
		{"256-byte prefix", func(d *KVIndexDef) { d.KeyPrefix = make([]byte, 256) }, true},
		{"255-byte prefix ok", func(d *KVIndexDef) { d.KeyPrefix = make([]byte, 255) }, false},
		{"empty prefix ok (whole keyspace)", func(d *KVIndexDef) { d.KeyPrefix = nil }, false},
		{"empty path", func(d *KVIndexDef) { d.PayloadPath = "" }, true},
		{"unknown kind", func(d *KVIndexDef) { d.Kind = 2 }, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := base()
			tc.mut(&d)
			err := d.Validate()
			if tc.wantErr && err == nil {
				t.Fatalf("%+v: expected error, got nil", d)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("%+v: unexpected error: %v", d, err)
			}
			if tc.wantErr && err != nil && !errors.Is(err, ErrKVIndexDef) {
				t.Fatalf("err = %v, want wrapping ErrKVIndexDef", err)
			}
		})
	}

	// The row-path rejection names the exact reason, since it is the one a
	// caller is most likely to hit by accident (passing a table-cell path
	// where only a top-level field or #count is allowed).
	d := base()
	d.PayloadPath = "b/42/hi"
	err := d.Validate()
	if err == nil || !strings.Contains(err.Error(), "an index path must be a top-level field or a #count") {
		t.Fatalf("err = %v, want the row-path message", err)
	}
}

func TestKVIndexSetArgsRoundtrip(t *testing.T) {
	def := KVIndexDef{Name: "by_status", KeyPrefix: []byte("order:"), PayloadPath: "status", Kind: KVIndexKindScalar, Enabled: true}
	b := EncodeKVIndexSetArgs(def)
	got, err := DecodeKVIndexSetArgs(b)
	if err != nil {
		t.Fatalf("DecodeKVIndexSetArgs: %v", err)
	}
	if !reflect.DeepEqual(got, def) {
		t.Fatalf("got %+v, want %+v", got, def)
	}
	// Trailing bytes are rejected — the frame is exactly one KVIndexDef.
	if _, err := DecodeKVIndexSetArgs(append(b, 0)); err == nil {
		t.Fatal("trailing byte accepted")
	}
}

func TestKVIndexListRoundtrip(t *testing.T) {
	defs := []KVIndexDef{
		{Name: "by_rc", KeyPrefix: []byte("sess:"), PayloadPath: "rc", Kind: KVIndexKindScalar, Enabled: true},
		{Name: "by_hits", PayloadPath: "hits#count", Kind: KVIndexKindCount, Enabled: false},
	}
	ready := []bool{true, false}
	b := EncodeKVIndexList(defs, ready)
	gotDefs, gotReady, err := DecodeKVIndexList(b)
	if err != nil {
		t.Fatalf("DecodeKVIndexList: %v", err)
	}
	if !reflect.DeepEqual(gotDefs, defs) {
		t.Fatalf("defs = %+v, want %+v", gotDefs, defs)
	}
	if !reflect.DeepEqual(gotReady, ready) {
		t.Fatalf("ready = %+v, want %+v", gotReady, ready)
	}

	// Empty list round-trips too.
	b0 := EncodeKVIndexList(nil, nil)
	d0, r0, err := DecodeKVIndexList(b0)
	if err != nil || len(d0) != 0 || len(r0) != 0 {
		t.Fatalf("empty list: defs=%v ready=%v err=%v", d0, r0, err)
	}

	// Trailing bytes and an over-cap declared count are both rejected.
	if _, _, err := DecodeKVIndexList(append(b, 0)); err == nil {
		t.Fatal("trailing byte accepted")
	}
	over := make([]byte, 2)
	binary.BigEndian.PutUint16(over, KVIndexMaxDefs+1)
	if _, _, err := DecodeKVIndexList(over); err == nil {
		t.Fatal("over-cap declared count accepted")
	}
}

func TestKVIndexDefTruncation(t *testing.T) {
	def := KVIndexDef{Name: "by_rc", KeyPrefix: []byte("sess:"), PayloadPath: "rc", Kind: KVIndexKindScalar, Enabled: true}
	b := AppendKVIndexDef(nil, def)
	for i := 0; i < len(b); i++ {
		if _, _, err := DecodeKVIndexDef(b[:i]); err == nil {
			t.Fatalf("prefix %d accepted", i)
		}
	}
}
