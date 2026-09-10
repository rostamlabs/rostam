// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- fixtures -------------------------------------------------------------

// kvMergeRows builds key-only rows (the projection every ordering test uses).
func kvMergeRows(keys ...string) []wire.KVQueryRow {
	rows := make([]wire.KVQueryRow, 0, len(keys))
	for _, k := range keys {
		rows = append(rows, wire.KVQueryRow{Key: []byte(k)})
	}
	return rows
}

// kvMergeKeys reads a merged page's keys back as strings.
func kvMergeKeys(r wire.KVQueryResult) []string {
	out := make([]string, 0, len(r.Rows))
	for _, row := range r.Rows {
		out = append(out, string(row.Key))
	}
	return out
}

// kvMergeCont finds group's continuation in a merged cursor.
func kvMergeCont(t *testing.T, r wire.KVQueryResult, group int) (wire.KVQueryCont, bool) {
	t.Helper()
	for _, c := range r.Cursor {
		if c.Group == uint32(group) {
			return c, true
		}
	}
	return wire.KVQueryCont{}, false
}

// kvMergeGroups asserts the merged cursor names exactly these groups, in the
// strictly increasing order the codec requires.
func kvMergeGroups(t *testing.T, r wire.KVQueryResult, want ...int) {
	t.Helper()
	got := make([]int, 0, len(r.Cursor))
	for _, c := range r.Cursor {
		got = append(got, int(c.Group))
	}
	if len(got) != len(want) {
		t.Fatalf("cursor names groups %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cursor names groups %v, want %v", got, want)
		}
	}
	for i := 1; i < len(r.Cursor); i++ {
		if r.Cursor[i].Group <= r.Cursor[i-1].Group {
			t.Fatalf("cursor groups are not strictly increasing: %v", got)
		}
	}
}

// --- ordering and truncation ----------------------------------------------

func TestMergeKVQueryOrdersAcrossGroups(t *testing.T) {
	parts := []wire.KVQueryResult{
		{Rows: kvMergeRows("a", "d", "g")},
		{Rows: kvMergeRows("b", "e")},
		{Rows: kvMergeRows("c", "f")},
	}
	got := mergeKVQuery(parts, nil, 10, wire.KVQueryMaxPageBytes)
	want := []string{"a", "b", "c", "d", "e", "f", "g"}
	if keys := kvMergeKeys(got); fmt.Sprint(keys) != fmt.Sprint(want) {
		t.Fatalf("merged keys = %v, want %v", keys, want)
	}
	// Every group answered in full and returned fewer rows than the limit, so
	// paging is over and the page carries NO cursor.
	if len(got.Cursor) != 0 {
		t.Fatalf("a complete answer carried a cursor: %+v", got.Cursor)
	}
}

func TestMergeKVQueryTruncates(t *testing.T) {
	parts := []wire.KVQueryResult{
		{Rows: kvMergeRows("a", "d"), Cursor: []wire.KVQueryCont{{Group: 0, After: []byte("d"), More: true}}},
		{Rows: kvMergeRows("b", "e"), Cursor: []wire.KVQueryCont{{Group: 1, After: []byte("e"), More: true}}},
	}
	got := mergeKVQuery(parts, nil, 3, wire.KVQueryMaxPageBytes)
	if keys := kvMergeKeys(got); fmt.Sprint(keys) != fmt.Sprint([]string{"a", "b", "d"}) {
		t.Fatalf("merged keys = %v, want [a b d]", keys)
	}
	kvMergeGroups(t, got, 0, 1)
	// Group 0 lost nothing to the truncation, so it keeps its OWN continuation.
	c0, _ := kvMergeCont(t, got, 0)
	if string(c0.After) != "d" || !c0.More {
		t.Fatalf("group 0 cont = %q more=%v, want \"d\" true (its own leaf continuation)", c0.After, c0.More)
	}
	// Group 1's "e" was dropped by the limit, so its continuation is rolled BACK
	// to the last key of its own that the merged page actually emitted.
	c1, _ := kvMergeCont(t, got, 1)
	if string(c1.After) != "b" || !c1.More {
		t.Fatalf("group 1 cont = %q more=%v, want \"b\" true (rolled back to its last emitted key)", c1.After, c1.More)
	}
}

// A group whose rows were ALL cut by the limit must keep the cursor it came in
// with: advancing it to the leaf's own continuation would step over rows this
// page never delivered, and no later page would ever go back for them.
func TestMergeKVQueryKeepsTruncatedGroupInCursor(t *testing.T) {
	in := []wire.KVQueryCont{{Group: 1, After: []byte("k00"), More: true}}
	parts := []wire.KVQueryResult{
		{Rows: kvMergeRows("k01", "k02"), Cursor: []wire.KVQueryCont{{Group: 0, After: []byte("k02"), More: true}}},
		// Every row here sorts after the limit's cut.
		{Rows: kvMergeRows("k80", "k81"), Cursor: []wire.KVQueryCont{{Group: 1, After: []byte("k81"), More: true}}},
	}
	got := mergeKVQuery(parts, in, 2, wire.KVQueryMaxPageBytes)
	if keys := kvMergeKeys(got); fmt.Sprint(keys) != fmt.Sprint([]string{"k01", "k02"}) {
		t.Fatalf("merged keys = %v, want [k01 k02]", keys)
	}
	// THE RULE 3 / RULE 4 ORDERING: group 1 emitted nothing, so rule 3 marks it
	// More=true, and rule 4 — which reads what rule 3 wrote — must therefore KEEP
	// it. Dropping it here loses k80 and k81 forever.
	c1, ok := kvMergeCont(t, got, 1)
	if !ok {
		t.Fatalf("group 1 was dropped from the cursor; its rows k80,k81 are now unreachable: %+v", got.Cursor)
	}
	if !c1.More {
		t.Fatalf("group 1 cont more=false, want true")
	}
	if string(c1.After) != "k00" {
		t.Fatalf("group 1 cont = %q, want \"k00\" — the cursor it came in with, not the leaf's %q", c1.After, "k81")
	}
}

func TestMergeKVQueryDropsExhaustedGroups(t *testing.T) {
	parts := []wire.KVQueryResult{
		// Exhausted: fewer rows than the limit and no continuation of its own.
		{Rows: kvMergeRows("a")},
		// Still has more.
		{Rows: kvMergeRows("b", "c", "d"), Cursor: []wire.KVQueryCont{{Group: 1, After: []byte("d"), More: true}}},
		// Answered nothing at all and is done.
		{},
	}
	got := mergeKVQuery(parts, nil, 4, wire.KVQueryMaxPageBytes)
	kvMergeGroups(t, got, 1)

	// And when the only group left is exhausted, the page carries no cursor at
	// all — that is how the caller learns paging is over.
	done := mergeKVQuery([]wire.KVQueryResult{{Rows: kvMergeRows("a")}, {}}, nil, 4, wire.KVQueryMaxPageBytes)
	if len(done.Cursor) != 0 {
		t.Fatalf("an exhausted fan-out returned a cursor: %+v", done.Cursor)
	}
}

// A group that returned EXACTLY the limit is kept even with More=false: it is
// indistinguishable from one that has more, and one extra empty round trip is
// cheaper than a lost row.
func TestMergeKVQueryKeepsFullGroupWithoutMore(t *testing.T) {
	parts := []wire.KVQueryResult{{Rows: kvMergeRows("a", "b")}}
	got := mergeKVQuery(parts, nil, 2, wire.KVQueryMaxPageBytes)
	c, ok := kvMergeCont(t, got, 0)
	if !ok {
		t.Fatalf("a group that filled the page exactly was dropped: %+v", got.Cursor)
	}
	if string(c.After) != "b" {
		t.Fatalf("cont = %q, want \"b\" (its last emitted key)", c.After)
	}
}

func TestMergeKVQueryEmpty(t *testing.T) {
	got := mergeKVQuery(nil, nil, 10, wire.KVQueryMaxPageBytes)
	if len(got.Rows) != 0 || len(got.Cursor) != 0 {
		t.Fatalf("merge of nothing = %+v, want an empty page with no cursor", got)
	}
	got = mergeKVQuery([]wire.KVQueryResult{{}, {}, {}}, nil, 10, wire.KVQueryMaxPageBytes)
	if len(got.Rows) != 0 || len(got.Cursor) != 0 {
		t.Fatalf("merge of three empty parts = %+v, want an empty page with no cursor", got)
	}
}

// The merge does NOT dedup, and this pins the reason: a key lives in exactly one
// group, so two groups can never offer the same key. The assertion is on the
// INPUT, so a future routing change that broke the property fails here rather
// than silently doubling rows.
func TestMergeKVQueryKeysAreGloballyUnique(t *testing.T) {
	const groups, keys = 6, 40
	seen := make(map[string]int, groups*keys)
	parts := make([]wire.KVQueryResult, groups)
	for i := 0; i < groups*keys; i++ {
		k := fmt.Sprintf("k%05d", i)
		g := shardOf([]byte(k), groups)
		if prev, dup := seen[k]; dup {
			t.Fatalf("key %q hashes into groups %d and %d", k, prev, g)
		}
		seen[k] = g
		parts[g].Rows = append(parts[g].Rows, wire.KVQueryRow{Key: []byte(k)})
	}
	for g := range parts {
		sort.Slice(parts[g].Rows, func(i, j int) bool {
			return string(parts[g].Rows[i].Key) < string(parts[g].Rows[j].Key)
		})
	}
	got := mergeKVQuery(parts, nil, wire.KVQueryMaxLimit, wire.KVQueryMaxPageBytes)
	if len(got.Rows) != groups*keys {
		t.Fatalf("merged %d rows, want %d — a duplicate or a dropped row", len(got.Rows), groups*keys)
	}
	for i := 1; i < len(got.Rows); i++ {
		if string(got.Rows[i-1].Key) >= string(got.Rows[i].Key) {
			t.Fatalf("merged rows are not strictly ascending at %d: %q then %q", i, got.Rows[i-1].Key, got.Rows[i].Key)
		}
	}
}

// --- the byte budget ------------------------------------------------------

func TestMergeKVQueryByteBudget(t *testing.T) {
	// Four 1 KiB values across two groups, against a budget that fits two of
	// them plus the frame overhead.
	val := make([]byte, 1024)
	row := func(k string) wire.KVQueryRow { return wire.KVQueryRow{Key: []byte(k), Value: val} }
	parts := []wire.KVQueryResult{
		{Rows: []wire.KVQueryRow{row("a"), row("c")}, Cursor: []wire.KVQueryCont{{Group: 0, After: []byte("c"), More: true}}},
		{Rows: []wire.KVQueryRow{row("b"), row("d")}, Cursor: []wire.KVQueryCont{{Group: 1, After: []byte("d"), More: true}}},
	}
	const maxBytes = 4 + 2 + 2*(7+1) + 2*(2+1+1+4+1024) + 8
	got := mergeKVQuery(parts, nil, 100, maxBytes)
	if keys := kvMergeKeys(got); fmt.Sprint(keys) != fmt.Sprint([]string{"a", "b"}) {
		t.Fatalf("merged keys = %v, want [a b] — the byte budget must cut the page", keys)
	}
	kvMergeGroups(t, got, 0, 1)
	c0, _ := kvMergeCont(t, got, 0)
	if string(c0.After) != "a" || !c0.More {
		t.Fatalf("group 0 cont = %q more=%v, want \"a\" true", c0.After, c0.More)
	}
	c1, _ := kvMergeCont(t, got, 1)
	if string(c1.After) != "b" || !c1.More {
		t.Fatalf("group 1 cont = %q more=%v, want \"b\" true", c1.After, c1.More)
	}

	// The merged page must ENCODE: the reserve for the cursor is what makes
	// "the rows fit" imply "the frame fits".
	enc, err := wire.EncodeKVQueryResult(got)
	if err != nil {
		t.Fatalf("the merged page does not encode: %v", err)
	}
	if len(enc) > maxBytes {
		t.Fatalf("merged frame is %d bytes, over the %d-byte budget", len(enc), maxBytes)
	}

	// A single row larger than the whole budget is still emitted: a page must
	// always advance or paging never terminates.
	huge := wire.KVQueryRow{Key: []byte("h"), Value: make([]byte, 4096)}
	one := mergeKVQuery([]wire.KVQueryResult{{Rows: []wire.KVQueryRow{huge}}}, nil, 10, 64)
	if len(one.Rows) != 1 {
		t.Fatalf("a page that cannot fit its first row emitted %d rows, want 1", len(one.Rows))
	}
}

// --- value nil-ness -------------------------------------------------------

// nil means "the value was omitted"; a zero-length value means "present and
// empty". The merge re-encodes rows it did not produce, so it must not collapse
// the two.
func TestMergeKVQueryPreservesValueNilness(t *testing.T) {
	parts := []wire.KVQueryResult{{Rows: []wire.KVQueryRow{
		{Key: []byte("a"), Value: nil},
		{Key: []byte("b"), Value: []byte{}},
		{Key: []byte("c"), Value: []byte("v")},
	}}}
	got := mergeKVQuery(parts, nil, 10, wire.KVQueryMaxPageBytes)
	if len(got.Rows) != 3 {
		t.Fatalf("merged %d rows, want 3", len(got.Rows))
	}
	if got.Rows[0].Value != nil {
		t.Errorf("an omitted value came back as %q, want nil", got.Rows[0].Value)
	}
	if got.Rows[1].Value == nil || len(got.Rows[1].Value) != 0 {
		t.Errorf("a present-and-empty value came back as %v, want a non-nil zero-length slice", got.Rows[1].Value)
	}
	// And it survives the round trip the coordinator actually performs.
	enc, err := wire.EncodeKVQueryResult(got)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	back, err := wire.DecodeKVQueryResult(enc)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if back.Rows[0].Value != nil {
		t.Errorf("after the round trip the omitted value is %q, want nil", back.Rows[0].Value)
	}
	if back.Rows[1].Value == nil || len(back.Rows[1].Value) != 0 {
		t.Errorf("after the round trip the empty value is %v, want non-nil and empty", back.Rows[1].Value)
	}
}

// A row at or below the cursor the caller sent is a broken leaf answer. It is
// DROPPED rather than emitted (the caller has already seen it), and the group
// keeps a continuation so nothing above it is lost.
func TestMergeKVQueryDropsRowsAtOrBelowTheCursor(t *testing.T) {
	in := []wire.KVQueryCont{{Group: 0, After: []byte("m"), More: true}}
	parts := []wire.KVQueryResult{{Rows: kvMergeRows("a", "m", "n")}}
	got := mergeKVQuery(parts, in, 10, wire.KVQueryMaxPageBytes)
	if keys := kvMergeKeys(got); fmt.Sprint(keys) != fmt.Sprint([]string{"n"}) {
		t.Fatalf("merged keys = %v, want [n] — rows at or below the cursor must be dropped", keys)
	}
}

// A group's continuation is a PEER's word. One that points below the cursor the
// caller sent would make the next page re-request what it just received, drop
// every row as already-seen, and rewrite the same cursor — paging that never
// terminates and never advances.
func TestMergeKVQueryCursorNeverMovesBackwards(t *testing.T) {
	in := []wire.KVQueryCont{{Group: 0, After: []byte("m"), More: true}}
	parts := []wire.KVQueryResult{{
		Rows:   kvMergeRows("n"),
		Cursor: []wire.KVQueryCont{{Group: 0, After: []byte("a"), More: true}}, // below the caller's cursor
	}}
	got := mergeKVQuery(parts, in, 10, wire.KVQueryMaxPageBytes)
	c, ok := kvMergeCont(t, got, 0)
	if !ok {
		t.Fatalf("group 0 left the cursor: %+v", got.Cursor)
	}
	if string(c.After) < "m" {
		t.Fatalf("cont = %q, want at least %q — the continuation moved backwards", c.After, "m")
	}
}

// --- the paging property --------------------------------------------------

// kvFakeGroup is one shard group's leaf: it answers with the smallest
// min(limit, remaining) keys strictly above the cursor, and continues from the
// last one it returned — the leaf's contract (ops.verifyPage), reduced to keys.
type kvFakeGroup struct {
	group int
	keys  []string // sorted, unique
}

func (g kvFakeGroup) page(after string, limit int) wire.KVQueryResult {
	i := 0
	for i < len(g.keys) && g.keys[i] <= after {
		i++
	}
	end := i + limit
	if end > len(g.keys) {
		end = len(g.keys)
	}
	res := wire.KVQueryResult{Rows: kvMergeRows(g.keys[i:end]...)}
	if end < len(g.keys) {
		res.Cursor = []wire.KVQueryCont{{Group: uint32(g.group), After: []byte(g.keys[end-1]), More: true}}
	}
	return res
}

// Threading the composite cursor to exhaustion must deliver EVERY key exactly
// once, in ascending order, over a bounded number of pages — for any split of
// the keyspace across groups and any page size.
func TestMergeKVQueryPagingIsExhaustive(t *testing.T) {
	const groups = 6
	rng := rand.New(rand.NewSource(20260910)) //nolint:gosec // deterministic test input, not cryptography

	for trial := 0; trial < 60; trial++ {
		limit := 1 + rng.Intn(9)
		fakes := make([]kvFakeGroup, groups)
		want := make([]string, 0, groups*40)
		for g := 0; g < groups; g++ {
			fakes[g] = kvFakeGroup{group: g}
			for i := 0; i < rng.Intn(41); i++ {
				k := fmt.Sprintf("k%06d", rng.Intn(1_000_000))
				fakes[g].keys = append(fakes[g].keys, k)
			}
			sort.Strings(fakes[g].keys)
			fakes[g].keys = kvDedup(fakes[g].keys)
			want = append(want, fakes[g].keys...)
		}
		// A key lives in exactly one group, so make the split disjoint the way
		// routing does.
		want = kvDedup(kvSorted(want))
		owned := make(map[string]int, len(want))
		for g := range fakes {
			kept := fakes[g].keys[:0]
			for _, k := range fakes[g].keys {
				if _, taken := owned[k]; taken {
					continue
				}
				owned[k] = g
				kept = append(kept, k)
			}
			fakes[g].keys = kept
		}

		var in []wire.KVQueryCont
		got := make([]string, 0, len(want))
		for page := 0; ; page++ {
			if page > 10*len(want)+50 {
				t.Fatalf("trial %d (limit %d): paging did not terminate after %d pages", trial, limit, page)
			}
			parts := make([]wire.KVQueryResult, groups)
			for g := 0; g < groups; g++ {
				if page > 0 && !kvCursorNames(in, g) {
					continue // exhausted on an earlier page: never sent again
				}
				after := ""
				for _, c := range in {
					if int(c.Group) == g {
						after = string(c.After)
					}
				}
				parts[g] = fakes[g].page(after, limit)
			}
			merged := mergeKVQuery(parts, in, limit, wire.KVQueryMaxPageBytes)
			if len(merged.Rows) > limit {
				t.Fatalf("trial %d: page %d carries %d rows, over the limit %d", trial, page, len(merged.Rows), limit)
			}
			got = append(got, kvMergeKeys(merged)...)
			if len(merged.Cursor) == 0 {
				break
			}
			if len(merged.Rows) == 0 && page > 0 && fmt.Sprint(in) == fmt.Sprint(merged.Cursor) {
				t.Fatalf("trial %d: page %d made no progress and returned the same cursor", trial, page)
			}
			in = merged.Cursor
		}

		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Fatalf("trial %d (limit %d): paged %d keys, want %d\n got=%v\nwant=%v",
				trial, limit, len(got), len(want), got, want)
		}
		for i := 1; i < len(got); i++ {
			if got[i-1] >= got[i] {
				t.Fatalf("trial %d: pages are not globally ascending at %d: %q then %q", trial, i, got[i-1], got[i])
			}
		}
	}
}

func kvCursorNames(conts []wire.KVQueryCont, group int) bool {
	for _, c := range conts {
		if int(c.Group) == group {
			return true
		}
	}
	return false
}

func kvSorted(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func kvDedup(sorted []string) []string {
	out := sorted[:0]
	for i, s := range sorted {
		if i == 0 || sorted[i-1] != s {
			out = append(out, s)
		}
	}
	return out
}
