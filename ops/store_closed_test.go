// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"fmt"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops/kvindex"
)

// TestIsStoreClosedMessage pins the matcher's exact form.
//
// The negative half is the load-bearing half. This predicate makes an error
// RETRYABLE, so every case a caller can influence must be false: the cases
// below build the two messages that actually quote caller input — the filter
// refusal's field name and the no-such-index refusal's index name — with that
// input ending in the sentinel, which is precisely what defeated the SUFFIX
// matcher this replaced.
func TestIsStoreClosedMessage(t *testing.T) {
	const inner = StoreClosedMsg
	fanout := func(group int, s string) string {
		return fmt.Sprintf("cluster: kv_query: shard group %d: %s", group, s)
	}
	remote := func(op, s string) string {
		return fmt.Sprintf("client: server error on op %q: %s", op, s)
	}

	for _, tc := range []struct {
		name string
		msg  string
		want bool
	}{
		// --- the real forms -------------------------------------------------
		{"bare", inner, true},
		{"behind the fan-out wrapper", fanout(3, inner), true},
		{"behind the fan-out wrapper, group 0", fanout(0, inner), true},
		{"behind a multi-digit group", fanout(127, inner), true},
		{"behind the peer client wrapper", remote("__kv_query_shard__", inner), true},
		{"behind both, as a remote leg arrives", fanout(3, remote("__kv_query_shard__", inner)), true},
		{"behind both, kv_query op name", fanout(11, remote("kv_query", inner)), true},

		// --- caller-controlled text ending in the sentinel -------------------
		// Each of these was TRUE under the suffix matcher.
		{
			name: "a filter refusal whose field name ends in the sentinel",
			msg:  fmt.Errorf("%w: field %q is not a path", ErrKVQueryFilter, "x: "+inner).Error(),
			want: false,
		},
		{
			name: "a filter refusal ending in the sentinel, behind the fan-out wrapper",
			msg:  fanout(3, fmt.Errorf("%w: field %q", ErrKVQueryFilter, "x: "+inner).Error()),
			want: false,
		},
		{
			name: "a no-such-index refusal whose index name ends in the sentinel",
			msg:  fmt.Errorf(kvQueryNoSuchIndexFmt, kvindex.ErrNoSuchIndex, "idx: "+inner, 3).Error(),
			want: false,
		},
		{
			name: "a scan-budget refusal ending in the sentinel",
			msg:  ErrKVQueryScanBudget.Error() + ": " + inner,
			want: false,
		},

		// --- unrelated faults ------------------------------------------------
		{"an unrelated fault merely ending in the sentinel", "wal append failed: " + inner, false},
		{"the sentinel with a tail", inner + " while writing /var/lib/rostam/wal-3", false},
		{"an unrelated fault", "open /var/lib/rostam/shard-7: no such file", false},
		{"empty", "", false},

		// --- malformed wrappers ---------------------------------------------
		{"fan-out wrapper with no group ordinal", "cluster: kv_query: shard group : " + inner, false},
		{"fan-out wrapper with a non-numeric group", "cluster: kv_query: shard group abc: " + inner, false},
		{"peer wrapper with an empty op name", `client: server error on op "": ` + inner, false},
		{
			name: "peer wrapper whose op name is arbitrary caller text",
			msg:  `client: server error on op "a b": ` + inner,
			want: false,
		},
		{"a lookalike prefix that is not a wrapper", "cluster: kv_query: " + inner, false},

		// --- bounds -----------------------------------------------------------
		{"over the length cap", strings.Repeat("a", maxStoreClosedMessageLen) + inner, false},
		{
			name: "more nested wrappers than production can produce",
			msg:  fanout(1, fanout(2, fanout(3, fanout(4, fanout(5, inner))))),
			want: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsStoreClosedMessage(tc.msg); got != tc.want {
				t.Fatalf("IsStoreClosedMessage(%q) = %v, want %v", tc.msg, got, tc.want)
			}
		})
	}
}

// TestIsStoreClosedMessageRejectsEveryFamilyMemberButItsOwn walks the canonical
// family and asserts the matcher claims exactly one member.
//
// The matcher's job is to recognise ONE refusal; every other member of the
// family reaches the same classifiers, several of them PERMANENT, and a matcher
// that claimed one of those would turn a client mistake into an unbounded retry
// loop. Driving it from the shared list means a refusal added later is checked
// here without anyone remembering to.
func TestIsStoreClosedMessageRejectsEveryFamilyMemberButItsOwn(t *testing.T) {
	claimed := 0
	for _, spec := range KVQueryErrorFamily() {
		got := IsStoreClosedMessage(spec.Err.Error())
		if spec.Err.Error() == StoreClosedMsg {
			if !got {
				t.Errorf("%s is the store-closed entry and must be claimed", spec.Name)
			}
			claimed++
			continue
		}
		if got {
			t.Errorf("%s (%s) was claimed as the store-closed refusal", spec.Name, spec.Class)
		}
	}
	if claimed != 1 {
		t.Fatalf("the family carries %d store-closed entries, want exactly 1", claimed)
	}
}
