// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// THE EMPTY KEY IS A REAL KV KEY. cache.Put and wire.DecodePutArgs both accept a
// zero-length key, so it can be stored, read back and walked like any other. It
// is also the SMALLEST key, so it is the first row of the first page of any
// ascending walk — which is exactly where a page can end on it.
//
// That made len(After) an unsafe stand-in for "is this request resuming". A page
// whose last emitted row is the empty key hands back a continuation whose After
// is legitimately empty; reading "no bytes" as "no cursor" makes the next page
// re-examine and re-emit that key. With a limit of 1 the query never advances at
// all. These tests page with limit 1 over a keyspace whose smallest key is the
// empty one, and fail by not terminating.

func TestKVQueryScanPagesPastTheEmptyKey(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	// The empty key sorts first, so a limit-1 page ends on it and the NEXT page
	// is the one that has to move.
	seedKV(t, tx, "", kvRec(7, "gold"))
	seedKV(t, tx, "a", kvRec(7, "gold"))
	seedKV(t, tx, "b", kvRec(7, "gold"))

	got, pages := pageAll(t, tx, wire.KVQueryArgs{
		Scan:   true,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  1,
	}, 12)
	assertAscendingUnique(t, got)
	want := []string{"", "a", "b"}
	if len(got) != len(want) {
		t.Fatalf("scan paging over the empty key returned %d keys in %d pages, want %v", len(got), pages, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scan paging returned %q, want %q", got, want)
		}
	}
}

// TestKVQueryScanEmptyKeyAloneTerminates is the sharpest form: the empty key is
// the ONLY key. The page emits it, the continuation is empty, and a build that
// reads that as "no cursor" re-emits it on every following page forever. pageAll
// fails the test when the page count runs away.
func TestKVQueryScanEmptyKeyAloneTerminates(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "", kvRec(7, "gold"))

	got, _ := pageAll(t, tx, wire.KVQueryArgs{
		Scan:   true,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  1,
	}, 8)
	if len(got) != 1 || got[0] != "" {
		t.Fatalf("paging over a single empty key returned %q, want exactly one empty key", got)
	}
}

// TestVerifyPageResumptionIsPresenceNotLength pins the leaf's rule directly: with
// resumed=true and an empty After, the empty key has ALREADY been returned and
// must be skipped; with resumed=false the same empty After means "first page"
// and the same key must be emitted.
func TestVerifyPageResumptionIsPresenceNotLength(t *testing.T) {
	tx, _, _ := newIndexedTx(t, "rc")
	seedKV(t, tx, "", kvRec(7, "gold"))
	seedKV(t, tx, "a", kvRec(7, "gold"))
	keys := [][]byte{{}, []byte("a")}

	first, err := verifyPage(tx, nil, keys, nil, false, nil, wire.KVQueryArgs{Limit: 10}, 0, false)
	if err != nil {
		t.Fatalf("verifyPage(first page): %v", err)
	}
	res, err := wire.DecodeKVQueryResult(first)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult: %v", err)
	}
	if len(res.Rows) != 2 {
		t.Fatalf("an unresumed page returned %d rows, want both keys including the empty one", len(res.Rows))
	}

	resumedPage, err := verifyPage(tx, nil, keys, []byte{}, true, nil, wire.KVQueryArgs{Limit: 10}, 0, false)
	if err != nil {
		t.Fatalf("verifyPage(resumed at the empty key): %v", err)
	}
	res, err = wire.DecodeKVQueryResult(resumedPage)
	if err != nil {
		t.Fatalf("DecodeKVQueryResult: %v", err)
	}
	if len(res.Rows) != 1 || string(res.Rows[0].Key) != "a" {
		t.Fatalf("a page resumed AT the empty key returned %d rows (first %q), want only %q",
			len(res.Rows), res.Rows[0].Key, "a")
	}
}

// TestKVQueryPageOverheadCoversTheWholeCursorBlock pins the reservation against
// the encoder rather than against a comment. It builds the worst page shape the
// leaf can produce — one continuation carrying a key at the encoder's 64 KiB cap
// — and asserts the reserved overhead is at least what that frame actually
// costs outside the rows. Reserving one byte too few (the cursor block's
// prefixLen) let a page filled exactly to the budget be rejected by
// EncodeKVQueryResult with no continuation, which wedges the query.
func TestKVQueryPageOverheadCoversTheWholeCursorBlock(t *testing.T) {
	key := make([]byte, 0xFFFF)
	for i := range key {
		key[i] = 'k'
	}
	// No rows: everything the frame costs here IS the overhead the constant
	// reserves — the row count plus the whole cursor block.
	frame, err := wire.EncodeKVQueryResult(wire.KVQueryResult{
		Cursor: []wire.KVQueryCont{{Group: 0, After: key, More: true}},
	})
	if err != nil {
		t.Fatalf("EncodeKVQueryResult(worst-case cursor): %v", err)
	}
	if len(frame) > kvQueryPageOverhead {
		t.Fatalf("a rowless page with a %d-byte continuation encodes to %d bytes, but only %d are reserved: a full page can exceed the cap and come back with no continuation",
			len(key), len(frame), kvQueryPageOverhead)
	}
}
