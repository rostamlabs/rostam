// SPDX-License-Identifier: Apache-2.0

package vtypes

import (
	"encoding/json"
	"testing"
)

// TestValueRecordJSONRoundtrip pins the exact JSON shape a ValueRecord
// produces: {"kind":"record","rec":"AQID"} — Rec's raw bytes marshal to
// base64 automatically via encoding/json's []byte handling, and the kind
// name comes from ValueKind.MarshalText.
func TestValueRecordJSONRoundtrip(t *testing.T) {
	v := Value{Kind: ValueRecord, Rec: []byte{1, 2, 3}}

	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	const want = `{"kind":"record","rec":"AQID"}`
	if string(b) != want {
		t.Fatalf("Marshal = %s, want %s", b, want)
	}

	var got Value
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Kind != ValueRecord {
		t.Fatalf("Kind = %v, want ValueRecord", got.Kind)
	}
	if string(got.Rec) != string(v.Rec) {
		t.Fatalf("Rec = %v, want %v", got.Rec, v.Rec)
	}
}

// TestValueRecordEqual exercises the three cases the bug class in Equal's doc
// comment warns about: without an explicit case, ANY two ValueRecords would
// compare equal via the default `return true`.
func TestValueRecordEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b Value
		want bool
	}{
		{"equal bytes", NewRecord([]byte{1, 2, 3}), NewRecord([]byte{1, 2, 3}), true},
		{"different bytes", NewRecord([]byte{1, 2, 3}), NewRecord([]byte{1, 2, 4}), false},
		{"different length", NewRecord([]byte{1, 2, 3}), NewRecord([]byte{1, 2}), false},
		{"nil vs empty", NewRecord(nil), NewRecord([]byte{}), true},
		{"nil vs nil", NewRecord(nil), NewRecord(nil), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.a.Equal(tt.b); got != tt.want {
				t.Errorf("Equal = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestValueKindNamesAppendOnly pins ValueRecord's numeric value at 9
// (ValueGeo+1). The kind is encoded as a raw u8 on disk (snapshot/WAL) and
// over the wire, so renumbering it would silently corrupt every existing
// snapshot and break replay — this test exists to catch an accidental
// reorder of the const block, not to validate any behavior.
func TestValueKindNamesAppendOnly(t *testing.T) {
	if ValueRecord != 9 {
		t.Fatalf("ValueRecord = %d, want 9 (ValueGeo+1)", ValueRecord)
	}
	if ValueGeo != 8 {
		t.Fatalf("ValueGeo = %d, want 8", ValueGeo)
	}
}

func TestValueRecordKindName(t *testing.T) {
	name, err := ValueRecord.MarshalText()
	if err != nil {
		t.Fatalf("MarshalText: %v", err)
	}
	if string(name) != "record" {
		t.Fatalf("MarshalText = %q, want %q", name, "record")
	}

	var k ValueKind
	if err := k.UnmarshalText([]byte("record")); err != nil {
		t.Fatalf("UnmarshalText: %v", err)
	}
	if k != ValueRecord {
		t.Fatalf("UnmarshalText = %v, want ValueRecord", k)
	}
}
