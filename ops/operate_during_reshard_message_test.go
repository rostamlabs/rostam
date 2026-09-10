// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// TestIsOperateDuringReshardMessage pins IsOperateDuringReshardMessage against
// the shapes OperateDuringReshardErr actually produces, and against the class it
// exists to REJECT.
//
// Two kinds of rejection are in the table, and they are rejected for different
// reasons. The first kind is the redaction bypass the exact-form matchers were
// introduced to close: a message that DOES contain the refusal text — an
// unrelated internal fault that wrapped it, or the refusal with a fault's own
// trailing context — which a bare strings.Contains would wave through and this
// matcher declines because the text is not anchored where the producer puts it.
// The second kind never contains the sentinel text at all (the empty string, a
// form with a byte cut off the front) and is rejected by the length bound or the
// prefix anchor; those rows guard the matcher's edges rather than demonstrating
// anything about Contains.
//
// The rows about the clipped rendering pin clipOperateName's INVARIANTS, not
// just its punctuation: at most 64 decoded bytes when there is no length tail,
// and exactly 64 with a count greater than 64 when there is. A quoted name is
// also required to be CANONICAL — the rendering strconv.Quote itself produces —
// so a message carrying an equivalent-but-different escape is a shape no
// producer in this tree can emit.
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

	// The two boundary names: exactly maxClippedNameBytes takes the untruncated
	// branch, one more takes the truncated one.
	atBound := OperateDuringReshardErr(strings.Repeat("c", maxClippedNameBytes)).Error()
	overBound := OperateDuringReshardErr(strings.Repeat("c", maxClippedNameBytes+1)).Error()
	prefix := ErrOperateDuringReshard.Error() + ": collection "

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
		{"name of exactly the shown-byte bound", atBound},
		{"name one byte over the bound (shortest truncated form)", overBound},
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
		// clipOperateName quotes a name this long only WITH a length tail, so an
		// untruncated form carrying one is a shape it cannot produce.
		{"quoted name one byte over the bound with no length tail",
			prefix + strconv.Quote(strings.Repeat("c", maxClippedNameBytes+1))},
		// The truncated branch always shows exactly the bound's worth of bytes.
		{"clipped form whose shown prefix is one byte short of the bound",
			prefix + strconv.Quote(strings.Repeat("c", maxClippedNameBytes-1)) + "… (200 bytes)"},
		// It is only reached when the full name is LONGER than what it shows.
		{"clipped form whose byte count equals the bound",
			prefix + strconv.Quote(strings.Repeat("c", maxClippedNameBytes)) +
				fmt.Sprintf("… (%d bytes)", maxClippedNameBytes)},
		// clipOperateName formats the count with %d, which never pads, so a
		// zero-padded count is a shape it cannot emit even though the digits
		// parse to a legal value.
		{"clipped form with a zero-padded byte count",
			prefix + strconv.Quote(strings.Repeat("c", maxClippedNameBytes)) + "… (0200 bytes)"},
		{"clipped form with a leading zero on a single-digit-padded count",
			prefix + strconv.Quote(strings.Repeat("c", maxClippedNameBytes)) + "… (0" +
				strconv.Itoa(maxClippedNameBytes+1) + " bytes)"},
		// strconv.Quote renders "A" as "A", never as an escape.
		{"noncanonical escape in the quoted name", prefix + `"\x41"`},
		{"noncanonical escape in a clipped name's prefix",
			prefix + `"\x41` + strings.Repeat("c", maxClippedNameBytes-1) + `"… (200 bytes)`},
	}
	for _, tc := range reject {
		t.Run("reject/"+tc.name, func(t *testing.T) {
			if IsOperateDuringReshardMessage(tc.msg) {
				t.Errorf("IsOperateDuringReshardMessage(%q) = true, want false", tc.msg)
			}
		})
	}
}
