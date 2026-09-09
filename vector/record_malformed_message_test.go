// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestIsRecordMalformedMessage pins IsRecordMalformedMessage against the exact
// shapes checkRecordValues / checkRecordValuesAll produce for ErrRecordMalformed,
// and against the class of input it exists to REJECT: an unrelated error whose
// message merely contains the sentinel text. A bare strings.Contains would pass
// every case in the reject table below, which is exactly the bug (a WAL/path/IO
// error that wraps ErrRecordMalformed becoming client-facing verbatim,
// unredacted).
func TestIsRecordMalformedMessage(t *testing.T) {
	single := checkRecordValues(malformedRecord()).Error()
	if !strings.HasPrefix(single, ErrRecordMalformed.Error()+": payload key \"session\": record: malformed record: ") {
		t.Fatalf("single-form fixture drifted from checkRecordValues: %q", single)
	}

	bulkRow0 := checkRecordValuesAll([]Metadata{malformedRecord()}).Error()
	bulkRow12 := checkRecordValuesAll([]Metadata{
		{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
		malformedRecord(),
	}).Error()

	longKey := strings.Repeat("k", 200)
	singleLongKey := checkRecordValues(Metadata{longKey: NewRecord([]byte{0x01, 0xFF})}).Error()

	accept := []struct {
		name string
		msg  string
	}{
		{"bare sentinel text", ErrRecordMalformed.Error()},
		{"single-payload form", single},
		{"single-payload form, long/clipped key", singleLongKey},
		{"bulk form, row 0", bulkRow0},
		{"bulk form, row 12 (multi-digit index)", bulkRow12},
		{"single form re-stringified (simulates decodePBResult)", errors.New(single).Error()},
		{"bulk form re-stringified (simulates decodePBResult)", errors.New(bulkRow12).Error()},
		{"trailing content appended after the detail (record.Validate's own detail has no closing anchor, so this is a real accepted shape, not a near-miss)",
			single + ": extra wrapped context"},
	}
	for _, tc := range accept {
		t.Run("accept/"+tc.name, func(t *testing.T) {
			if !IsRecordMalformedMessage(tc.msg) {
				t.Errorf("IsRecordMalformedMessage(%q) = false, want true", tc.msg)
			}
		})
	}

	reject := []struct {
		name string
		msg  string
	}{
		{"empty string", ""},
		{"unrelated error containing the sentinel text (%v wrap)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", ErrRecordMalformed).Error()},
		{"unrelated error containing the sentinel text (%w wrap, op-name prefixed)",
			fmt.Errorf("vector_insert: %w", ErrRecordMalformed).Error()},
		{"sentinel text embedded mid-message with unrelated trailing content",
			"apply failed: " + ErrRecordMalformed.Error() + " (retrying on shard 3)"},
		{"single-payload form with a byte cut off the front",
			strings.TrimPrefix(single, "v")},
		{"bulk form with the row index missing",
			"payload : " + single},
		{"looks like the sentinel but missing the record.ErrRecord marker entirely",
			ErrRecordMalformed.Error() + ": payload key \"session\": wire: args too short"},
		{"marker present but nothing follows it (empty detail)",
			ErrRecordMalformed.Error() + ": payload key \"session\": record: malformed record: "},
		{"oversize input beyond the length bound",
			single + strings.Repeat(" more detail", 300)},
	}
	for _, tc := range reject {
		t.Run("reject/"+tc.name, func(t *testing.T) {
			if IsRecordMalformedMessage(tc.msg) {
				t.Errorf("IsRecordMalformedMessage(%q) = true, want false", tc.msg)
			}
		})
	}
}
