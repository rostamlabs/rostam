// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestIsOperateDuringReshardMessage pins IsOperateDuringReshardMessage against
// the shapes OperateDuringReshardErr actually produces, and against the class it
// exists to REJECT: an unrelated internal fault whose message merely contains the
// refusal text. A bare strings.Contains would accept every case in the reject
// table, which is the redaction bypass the exact-form matchers were introduced to
// close.
func TestIsOperateDuringReshardMessage(t *testing.T) {
	short := OperateDuringReshardErr("docs").Error()
	if !strings.HasPrefix(short, ErrOperateDuringReshard.Error()+": collection \"docs\"") {
		t.Fatalf("fixture drifted from OperateDuringReshardErr: %q", short)
	}
	longName := strings.Repeat("c", 200)
	long := OperateDuringReshardErr(longName).Error()
	if !strings.HasSuffix(long, " bytes)") {
		t.Fatalf("long-name fixture did not take clipOperateName's truncated branch: %q", long)
	}
	// A name needing escapes, so the quoted rendering is not a plain identifier.
	quoted := OperateDuringReshardErr("a\"b\nc").Error()

	accept := []struct {
		name string
		msg  string
	}{
		{"bare sentinel text", ErrOperateDuringReshard.Error()},
		{"detailed form, short name", short},
		{"detailed form, long/clipped name", long},
		{"detailed form, name needing escapes", quoted},
		{"detailed form re-stringified (simulates decodePBResult)", errors.New(short).Error()},
		{"clipped form re-stringified (simulates decodePBResult)", errors.New(long).Error()},
	}
	for _, tc := range accept {
		t.Run("accept/"+tc.name, func(t *testing.T) {
			if !IsOperateDuringReshardMessage(tc.msg) {
				t.Errorf("IsOperateDuringReshardMessage(%q) = false, want true", tc.msg)
			}
		})
	}

	reject := []struct {
		name string
		msg  string
	}{
		{"empty string", ""},
		{"unrelated fault containing the refusal text (%v wrap)",
			fmt.Errorf("wal append failed at /var/lib/rostam/x: %v", ErrOperateDuringReshard).Error()},
		{"unrelated fault containing the refusal text (%w wrap, op-name prefixed)",
			fmt.Errorf("vector_operate: %w", ErrOperateDuringReshard).Error()},
		{"refusal text with unrelated trailing content",
			short + " (retrying on shard 3)"},
		{"refusal text embedded mid-message",
			"apply failed: " + ErrOperateDuringReshard.Error() + " on shard 3"},
		{"detailed form with a byte cut off the front", strings.TrimPrefix(short, "r")},
		{"collection name not quoted at all",
			ErrOperateDuringReshard.Error() + ": collection docs"},
		{"collection name with a stray byte after the closing quote",
			ErrOperateDuringReshard.Error() + ": collection \"docs\"!"},
		{"clipped form with the byte count missing",
			ErrOperateDuringReshard.Error() + ": collection \"docs\"… ( bytes)"},
		{"oversize input beyond the length bound",
			strings.Repeat(short+" ", 64)},
	}
	for _, tc := range reject {
		t.Run("reject/"+tc.name, func(t *testing.T) {
			if IsOperateDuringReshardMessage(tc.msg) {
				t.Errorf("IsOperateDuringReshardMessage(%q) = true, want false", tc.msg)
			}
		})
	}
}
