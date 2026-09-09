// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestIsPayloadKeyNotRecordMessage pins IsPayloadKeyNotRecordMessage against the
// exact shapes production code produces for ErrPayloadKeyNotRecord (the bare
// guard-clause form and currentRecordValue's detailed form), and against the
// class of input it exists to REJECT: an unrelated error whose message merely
// contains the sentinel text. A bare strings.Contains would pass every case in
// the reject table below, which is exactly the bug this replaces.
func TestIsPayloadKeyNotRecordMessage(t *testing.T) {
	_, _, detailed := currentRecordValue(Metadata{"k": {Kind: ValueInt, Int: 5}}, "k", false)
	if detailed == nil {
		t.Fatal("currentRecordValue fixture returned a nil error")
	}
	detailedMsg := detailed.Error()
	wantPrefix := ErrPayloadKeyNotRecord.Error() + `: payload key "k" holds a value of kind `
	if !strings.HasPrefix(detailedMsg, wantPrefix) {
		t.Fatalf("detailed-form fixture drifted from currentRecordValue: %q", detailedMsg)
	}

	_, _, detailedFloat := currentRecordValue(Metadata{"widerKind": {Kind: ValueFloat, Flt: 1.5}}, "widerKind", false)
	detailedFloatMsg := detailedFloat.Error()

	accept := []struct {
		name string
		msg  string
	}{
		{"bare sentinel text (guard-clause form)", ErrPayloadKeyNotRecord.Error()},
		{"detailed form, short key", detailedMsg},
		{"detailed form, different kind digit", detailedFloatMsg},
		{"bare form re-stringified (simulates decodePBResult)", errors.New(ErrPayloadKeyNotRecord.Error()).Error()},
		{"detailed form re-stringified (simulates decodePBResult)", errors.New(detailedMsg).Error()},
	}
	for _, tc := range accept {
		t.Run("accept/"+tc.name, func(t *testing.T) {
			if !IsPayloadKeyNotRecordMessage(tc.msg) {
				t.Errorf("IsPayloadKeyNotRecordMessage(%q) = false, want true", tc.msg)
			}
		})
	}

	reject := []struct {
		name string
		msg  string
	}{
		{"empty string", ""},
		{"unrelated error containing the sentinel text (%v wrap)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", ErrPayloadKeyNotRecord).Error()},
		{"unrelated error containing the sentinel text (%w wrap, op-name prefixed)",
			fmt.Errorf("vector_operate: %w", ErrPayloadKeyNotRecord).Error()},
		{"sentinel text embedded mid-message with unrelated trailing content",
			"apply failed: " + ErrPayloadKeyNotRecord.Error() + " (retrying on shard 3)"},
		{"detailed form with a byte cut off the front",
			strings.TrimPrefix(detailedMsg, "v")},
		{"detailed form with trailing garbage appended after the digits",
			detailedMsg + " extra"},
		{"detailed form with no digits after the marker",
			ErrPayloadKeyNotRecord.Error() + `: payload key "k" holds a value of kind `},
		{"looks like the marker but missing a space",
			ErrPayloadKeyNotRecord.Error() + `: payload key "k" holds a value of kind5`},
		{"oversize input beyond the length bound",
			ErrPayloadKeyNotRecord.Error() + `: payload key "` + strings.Repeat("k", 5000) + `" holds a value of kind 2`},
	}
	for _, tc := range reject {
		t.Run("reject/"+tc.name, func(t *testing.T) {
			if IsPayloadKeyNotRecordMessage(tc.msg) {
				t.Errorf("IsPayloadKeyNotRecordMessage(%q) = true, want false", tc.msg)
			}
		})
	}
}
