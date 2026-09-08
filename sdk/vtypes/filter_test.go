// SPDX-License-Identifier: Apache-2.0

package vtypes

import "testing"

// TestFilterOpRowPresenceTextRoundtrip pins FilterRowExists/FilterRowAbsent
// to their wire values and their JSON names ("row_exists"/"row_absent"), and
// checks MarshalText/UnmarshalText agree in both directions.
//
// NOTE on the values: the plan docs (contract.md/constraints.md) state 21/22
// "following FilterGeoPolygon", but FilterGeoPolygon is ALREADY 21 in the
// existing enum (FilterAnd..FilterGeoPolygon is 22 values, 0-21 — verified by
// enumerating every FilterOp's MarshalText). Using literal 21 for
// FilterRowExists would silently collide with FilterGeoPolygon=21. The
// append-only/never-renumber rule (constraints.md) takes priority over the
// docs' arithmetic slip, so these are appended after the last existing op:
// FilterRowExists=22, FilterRowAbsent=23. Flagged to the team lead to correct
// the docs; no other code was found depending on the literal 21/22.
func TestFilterOpRowPresenceTextRoundtrip(t *testing.T) {
	cases := []struct {
		op   FilterOp
		val  FilterOp
		name string
	}{
		{FilterRowExists, 22, "row_exists"},
		{FilterRowAbsent, 23, "row_absent"},
	}
	for _, c := range cases {
		if c.op != c.val {
			t.Errorf("%s: FilterOp value = %d, want %d", c.name, c.op, c.val)
		}
		text, err := c.op.MarshalText()
		if err != nil {
			t.Fatalf("%s: MarshalText: %v", c.name, err)
		}
		if string(text) != c.name {
			t.Errorf("MarshalText(%d) = %q, want %q", c.op, text, c.name)
		}
		var got FilterOp
		if err := got.UnmarshalText([]byte(c.name)); err != nil {
			t.Fatalf("UnmarshalText(%q): %v", c.name, err)
		}
		if got != c.op {
			t.Errorf("UnmarshalText(%q) = %d, want %d", c.name, got, c.op)
		}
	}
}

// TestFilterOpRowPresenceRejectsUnknownName is a control: a near-miss name
// must not silently parse as one of the two new ops.
func TestFilterOpRowPresenceRejectsUnknownName(t *testing.T) {
	var got FilterOp
	if err := got.UnmarshalText([]byte("row_present")); err == nil {
		t.Errorf("UnmarshalText(row_present) succeeded as %d, want a hard error", got)
	}
}
