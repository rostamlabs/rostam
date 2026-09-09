// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestIsRecordTooLargeMessage pins IsRecordTooLargeMessage against the exact
// shapes checkRecordValues / checkRecordValuesAll produce, and against the
// class of input it exists to REJECT: an unrelated error whose message merely
// contains the sentinel text. A bare strings.Contains would pass every case in
// the reject table below, which is exactly the bug (a WAL/path/IO error that
// wraps ErrRecordTooLarge becoming client-facing verbatim, unredacted).
func TestIsRecordTooLargeMessage(t *testing.T) {
	single := checkRecordValues(Metadata{"session": NewRecord(make([]byte, maxRecordValueBytes+1))}).Error()
	if !strings.HasPrefix(single, ErrRecordTooLarge.Error()+": payload key \"session\" holds a ") {
		t.Fatalf("single-form fixture drifted from checkRecordValues: %q", single)
	}

	longKey := string(make([]byte, 200))
	for i := range longKey {
		// A printable filler so clipField's quoting/prefix branch is exercised
		// without embedding NUL bytes in the fixture.
		longKey = longKey[:i] + "k" + longKey[i+1:]
	}
	singleLongKey := checkRecordValues(Metadata{longKey: NewRecord(make([]byte, maxRecordValueBytes+1))}).Error()

	bulkRow0 := checkRecordValuesAll([]Metadata{
		{"session": NewRecord(make([]byte, maxRecordValueBytes+1))},
	}).Error()
	bulkRow12 := checkRecordValuesAll([]Metadata{
		{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
		{"session": NewRecord(make([]byte, maxRecordValueBytes+1))},
	}).Error()

	// The POST-MUTATION form: not an ingest gate at all, but the bound the four
	// mutatePayloadRecord bodies apply to the bytes the operate engine just
	// produced. It used to be a SECOND shape ("payload key %q would hold a ...")
	// that this matcher declined, which redacted a fixable client mistake to
	// "internal error" on every clustered apply — see recordTooLargeErr. Built by
	// driving a real mutation so a re-divergence fails here rather than in
	// production.
	mutated := func(key string) string {
		t.Helper()
		h, err := newHNSW(mutateRecordCfg())
		if err != nil {
			t.Fatalf("newHNSW: %v", err)
		}
		seedMutatePoint(t, h, 1)
		_, _, _, _, merr := h.MutatePayloadRecord(1, key, func(_ []byte, _ bool) ([]byte, RecordMutation, error) {
			return make([]byte, maxRecordValueBytes+1), RecordStore, nil
		}, CASCond{})
		if !errors.Is(merr, ErrRecordTooLarge) {
			t.Fatalf("post-mutation fixture: err = %v, want ErrRecordTooLarge", merr)
		}
		return merr.Error()
	}
	postMutation := mutated("session")
	postMutationLongKey := mutated(longKey)

	accept := []struct {
		name string
		msg  string
	}{
		{"bare sentinel text", ErrRecordTooLarge.Error()},
		{"post-mutation form, short key", postMutation},
		{"post-mutation form, long/clipped key", postMutationLongKey},
		{"post-mutation form re-stringified (simulates decodePBResult)", errors.New(postMutation).Error()},
		{"single-payload form, short key", single},
		{"single-payload form, long/clipped key", singleLongKey},
		{"bulk form, row 0", bulkRow0},
		{"bulk form, row 12 (multi-digit index)", bulkRow12},
		{"single form re-stringified (simulates decodePBResult)", errors.New(single).Error()},
		{"bulk form re-stringified (simulates decodePBResult)", errors.New(bulkRow12).Error()},
	}
	for _, tc := range accept {
		t.Run("accept/"+tc.name, func(t *testing.T) {
			if !IsRecordTooLargeMessage(tc.msg) {
				t.Errorf("IsRecordTooLargeMessage(%q) = false, want true", tc.msg)
			}
		})
	}

	reject := []struct {
		name string
		msg  string
	}{
		{"empty string", ""},
		{"unrelated error containing the sentinel text (%v wrap)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", ErrRecordTooLarge).Error()},
		{"unrelated error containing the sentinel text (%w wrap, op-name prefixed)",
			fmt.Errorf("vector_insert: %w", ErrRecordTooLarge).Error()},
		{"sentinel text embedded mid-message with unrelated trailing content",
			"apply failed: " + ErrRecordTooLarge.Error() + " (retrying on shard 3)"},
		{"single-payload form with a byte cut off the front",
			strings.TrimPrefix(single, "v")},
		{"single-payload form with trailing garbage appended",
			single + " extra"},
		{"bulk form with the row index missing",
			"payload : " + single},
		{"looks like the bulk prefix but the sentinel text is misquoted",
			"payload 0: vector: record payload value exceeds the storage cap (no detail)"},
		{"oversize input beyond the length bound",
			strings.Repeat("vector: record payload value exceeds the storage cap: payload key \"k\" holds a 1-byte record, the cap is 16777216 bytes ", 64)},
	}
	for _, tc := range reject {
		t.Run("reject/"+tc.name, func(t *testing.T) {
			if IsRecordTooLargeMessage(tc.msg) {
				t.Errorf("IsRecordTooLargeMessage(%q) = true, want false", tc.msg)
			}
		})
	}
}
