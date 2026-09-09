// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"testing"
)

// TestIsVectorRecordAbsentMessage pins IsVectorRecordAbsentMessage against
// ErrVectorRecordAbsent's one production shape (bare, no wrapping — see the
// function's doc comment for why), and against the class of input it exists to
// REJECT: an unrelated error whose message merely contains the sentinel text. A
// bare strings.Contains would pass every case in the reject table below, which
// is exactly the bug this replaces.
func TestIsVectorRecordAbsentMessage(t *testing.T) {
	accept := []struct {
		name string
		msg  string
	}{
		{"bare sentinel text", ErrVectorRecordAbsent.Error()},
		{"re-stringified (simulates decodePBResult)", errors.New(ErrVectorRecordAbsent.Error()).Error()},
	}
	for _, tc := range accept {
		t.Run("accept/"+tc.name, func(t *testing.T) {
			if !IsVectorRecordAbsentMessage(tc.msg) {
				t.Errorf("IsVectorRecordAbsentMessage(%q) = false, want true", tc.msg)
			}
		})
	}

	reject := []struct {
		name string
		msg  string
	}{
		{"empty string", ""},
		{"unrelated error containing the sentinel text (%v wrap)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", ErrVectorRecordAbsent).Error()},
		{"unrelated error containing the sentinel text (%w wrap, op-name prefixed)",
			fmt.Errorf("vector_operate: %w", ErrVectorRecordAbsent).Error()},
		{"sentinel text embedded mid-message with unrelated trailing content",
			"apply failed: " + ErrVectorRecordAbsent.Error() + " (retrying on shard 3)"},
		{"sentinel text with a byte cut off the front",
			ErrVectorRecordAbsent.Error()[1:]},
		{"sentinel text with trailing garbage appended",
			ErrVectorRecordAbsent.Error() + " extra"},
	}
	for _, tc := range reject {
		t.Run("reject/"+tc.name, func(t *testing.T) {
			if IsVectorRecordAbsentMessage(tc.msg) {
				t.Errorf("IsVectorRecordAbsentMessage(%q) = true, want false", tc.msg)
			}
		})
	}
}
