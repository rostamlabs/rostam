// SPDX-License-Identifier: Apache-2.0

package record

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestRowAbsentProvenMatchesOracle holds RowAbsentProven to the tree oracle on
// well-formed records: for every field/row path the generators produce, it must
// answer true exactly when the decoded tree says the field is a table AND the
// row is not in it. Anything else — a present row, a scalar or missing field, a
// path that cannot apply — is false.
//
// The stronger half of the property is the error class: on a record the
// generators built and DecodeRecord accepted, the proof walk may never report
// ErrRecord. A "prove it" operation that called well-formed records malformed
// would answer false for rows that really are absent, which is a wrong answer in
// the opposite direction from the one it exists to prevent.
func TestRowAbsentProvenMatchesOracle(t *testing.T) {
	records, paths := 1500, 24
	if testing.Short() {
		records, paths = 200, 10
	}
	rng := rand.New(rand.NewSource(20260909))
	r := NewResolver(64)
	seen := map[string]int{}
	checked := 0
	for i := 0; i < records; i++ {
		var enc []byte
		var dec *wire.Record
		var err error
		if i%2 == 0 {
			enc, dec, err = randomSchemaRecord(rng)
		} else {
			enc, dec, err = randomDynamicRecord(rng)
		}
		if err != nil {
			t.Fatalf("generator: %v", err)
		}
		for j := 0; j < paths; j++ {
			p, perr := ParsePath(randomPath(rng, dec))
			if perr != nil || len(p.Segs) != 2 || p.Segs[1].Kind != SegRow {
				continue // only field/row paths are in this operation's domain
			}
			checked++
			got, gerr := r.RowAbsentProven(enc, p)
			if gerr != nil && errors.Is(gerr, ErrRecord) {
				t.Fatalf("record %d: proof called a well-formed record malformed: %v", i, gerr)
			}
			// The oracle: the row is provably absent iff the tree resolves the
			// field to a table and the row to Absent.
			res, oerr := resolveTree(dec, p)
			fres, ferr := resolveTree(dec, Path{Segs: p.Segs[:1:1]})
			want := oerr == nil && ferr == nil && res.Kind == Absent && fres.Kind == Table
			if got != want {
				t.Fatalf("record %d path %+v: RowAbsentProven = (%v, %v), oracle wants %v (row %+v/%v, field %+v/%v)",
					i, p.Segs, got, gerr, want, res, oerr, fres, ferr)
			}
			switch {
			case want:
				seen["proven-absent"]++
			case oerr == nil && res.Kind == RowPresent:
				seen["row-present"]++
			case ferr == nil && fres.Kind == Table:
				seen["table-but-not-absent"]++
			default:
				seen["not-a-table"]++
			}
		}
	}
	t.Logf("%d field/row paths checked: %v", checked, seen)
	// A property test that only ever saw one outcome would prove nothing.
	for _, k := range []string{"proven-absent", "row-present", "not-a-table"} {
		if seen[k] == 0 {
			t.Errorf("outcome %q never occurred; the generator stopped producing it", k)
		}
	}
}

// TestRowAbsentProvenRejectsUnorderedRows is the finding itself: a table whose
// rows are NOT in ascending key order can answer Absent for a row that is
// stored, because a schema-mode lookup is a binary search and a dynamic-mode
// one stops at the first key greater than the target. Resolve documents that
// leniency and keeps it — a filter leaf reads Absent as "no match", which is
// safe. RowAbsentProven must refuse instead, because it turns Absent into a
// positive claim about the point.
//
// Both modes are patched at the BYTE level, since the encoder sorts rows and
// DecodeRecord rejects the result: this is a record only a hostile writer
// produces, which is exactly the input the operation has to be right about.
func TestRowAbsentProvenRejectsUnorderedRows(t *testing.T) {
	t.Run("schema", func(t *testing.T) {
		r := NewResolver(8)
		enc, _ := sessionRecord(t, true)
		// Precondition: with the rows in order, row 42 is present and row 7 is
		// provably absent.
		mustResolveKind(t, r, enc, "b/42", RowPresent)
		if ok, err := r.RowAbsentProven(enc, mustPath(t, "b/7")); !ok || err != nil {
			t.Fatalf("precondition: RowAbsentProven(b/7) = (%v, %v), want (true, nil)", ok, err)
		}

		swapSchemaRows(t, r, enc, 5, 0, 1)

		// The documented leniency: the binary search now misses a row that is
		// still there.
		mustResolveKind(t, r, enc, "b/42", Absent)
		for _, path := range []string{"b/42", "b/99", "b/7"} {
			ok, err := r.RowAbsentProven(enc, mustPath(t, path))
			if ok {
				t.Errorf("RowAbsentProven(%s) = true on an out-of-order table, want false", path)
			}
			if !errors.Is(err, ErrRecord) {
				t.Errorf("RowAbsentProven(%s) error = %v, want one wrapping ErrRecord", path, err)
			}
		}
	})

	t.Run("dynamic", func(t *testing.T) {
		r := NewResolver(8)
		enc, _ := dynamicRecord(t)
		mustResolveKind(t, r, enc, "t/42", RowPresent)
		if ok, err := r.RowAbsentProven(enc, mustPath(t, "t/7")); !ok || err != nil {
			t.Fatalf("precondition: RowAbsentProven(t/7) = (%v, %v), want (true, nil)", ok, err)
		}

		swapDynamicRows(t, enc, "t", 0, 1)

		mustResolveKind(t, r, enc, "t/42", Absent)
		// The row that moved to the FRONT still matches on the first
		// comparison, so it resolves RowPresent and needs no proof: row
		// presence is proof of itself even on a damaged table.
		mustResolveKind(t, r, enc, `t/"DE"`, RowPresent)
		if ok, err := r.RowAbsentProven(enc, mustPath(t, `t/"DE"`)); ok || err != nil {
			t.Errorf(`RowAbsentProven(t/"DE") = (%v, %v), want (false, nil)`, ok, err)
		}
		// Everything that resolved Absent must now be refused: the walk sees
		// the disorder that made the early stop wrong.
		for _, path := range []string{"t/42", "t/7"} {
			ok, err := r.RowAbsentProven(enc, mustPath(t, path))
			if ok {
				t.Errorf("RowAbsentProven(%s) = true on an out-of-order table, want false", path)
			}
			if !errors.Is(err, ErrRecord) {
				t.Errorf("RowAbsentProven(%s) error = %v, want one wrapping ErrRecord", path, err)
			}
		}
	})
}

// TestRowAbsentProvenNonAbsentOutcomes pins the answers that need no walk: a
// present row, a field that is not a table, a field the record does not have,
// and a path of the wrong shape.
func TestRowAbsentProvenNonAbsentOutcomes(t *testing.T) {
	r := NewResolver(8)
	enc, _ := sessionRecord(t, true)
	for _, c := range []struct {
		path    string
		want    bool
		wantErr error
	}{
		{path: "b/42", want: false},                   // present
		{path: "b/7", want: true},                     // genuinely absent
		{path: "rc/1", want: false, wantErr: ErrPath}, // a scalar, not a table
		{path: "zzz/1", want: false},                  // no such field
		{path: "#9/1", want: false},                   // position past the schema
	} {
		got, err := r.RowAbsentProven(enc, mustPath(t, c.path))
		if got != c.want {
			t.Errorf("RowAbsentProven(%s) = %v, want %v (err %v)", c.path, got, c.want, err)
		}
		if c.wantErr != nil && !errors.Is(err, c.wantErr) {
			t.Errorf("RowAbsentProven(%s) error = %v, want one wrapping %v", c.path, err, c.wantErr)
		}
		if c.wantErr == nil && err != nil {
			t.Errorf("RowAbsentProven(%s) unexpected error: %v", c.path, err)
		}
	}
	// A path that is not field/row is a caller mistake, not a record fact.
	if _, err := r.RowAbsentProven(enc, mustPath(t, "b/42/hi")); !errors.Is(err, ErrPath) {
		t.Errorf("RowAbsentProven on a three-segment path error = %v, want one wrapping ErrPath", err)
	}
}

func mustResolveKind(t *testing.T, r *Resolver, rec []byte, path string, want Kind) {
	t.Helper()
	got, err := r.Resolve(rec, mustPath(t, path))
	if err != nil {
		t.Fatalf("Resolve(%q): %v", path, err)
	}
	if got.Kind != want {
		t.Fatalf("Resolve(%q).Kind = %v, want %v", path, got.Kind, want)
	}
}

// swapSchemaRows swaps two fixed-width rows of a schema-mode table IN PLACE,
// producing the out-of-order table DecodeRecord would refuse. It locates the
// rows through the resolver's own offset arithmetic rather than hard-coded
// offsets, so it keeps working if the fixture changes.
func swapSchemaRows(t *testing.T, r *Resolver, rec []byte, fieldPos, i, j int) {
	t.Helper()
	e, err := r.schemaFor(rec)
	if err != nil {
		t.Fatalf("schemaFor: %v", err)
	}
	tl := e.layout.Tables[fieldPos]
	off, err := schemaFieldOffset(rec, 1+e.blobLen, e.schema, e.layout, fieldPos)
	if err != nil {
		t.Fatalf("schemaFieldOffset: %v", err)
	}
	nRows, rowsOff, err := schemaTableHeader(rec, off, tl)
	if err != nil {
		t.Fatalf("schemaTableHeader: %v", err)
	}
	if i >= int(nRows) || j >= int(nRows) {
		t.Fatalf("table has %d rows, cannot swap %d and %d", nRows, i, j)
	}
	a := rec[rowsOff+i*tl.RowWidth:][:tl.RowWidth]
	b := rec[rowsOff+j*tl.RowWidth:][:tl.RowWidth]
	tmp := append([]byte(nil), a...)
	copy(a, b)
	copy(b, tmp)
}

// swapDynamicRows swaps two whole [rowLen][row] units of a dynamic-mode table
// IN PLACE. Swapping complete units keeps the table's byte length identical, so
// no length prefix upstream has to be rewritten.
func swapDynamicRows(t *testing.T, rec []byte, field string, i, j int) {
	t.Helper()
	inner, rowsOff := dynamicTableFor(t, rec, field)
	spans := dynamicRowSpans(t, inner, rowsOff)
	if i >= len(spans) || j >= len(spans) {
		t.Fatalf("table has %d rows, cannot swap %d and %d", len(spans), i, j)
	}
	var out []byte
	for k := range spans {
		src := spans[k]
		switch k {
		case i:
			src = spans[j]
		case j:
			src = spans[i]
		}
		out = append(out, inner[src[0]:src[1]]...)
	}
	if got, want := len(out), len(inner)-rowsOff; got != want {
		t.Fatalf("rebuilt rows are %d bytes, want %d", got, want)
	}
	copy(inner[rowsOff:], out)
}

// dynamicTableFor returns the named dynamic field's table body (aliasing rec,
// so writes through it patch the record) and the offset of its first row.
func dynamicTableFor(t *testing.T, rec []byte, field string) ([]byte, int) {
	t.Helper()
	nFields, m, err := readCanonicalUvarint(rec, 1)
	if err != nil {
		t.Fatalf("field count: %v", err)
	}
	off := 1 + m
	for i := uint64(0); i < nFields; i++ {
		name, next, err := readName(rec, off)
		if err != nil {
			t.Fatalf("name: %v", err)
		}
		off = next
		typ, n, valOff, err := readTypeTag(rec, off)
		if err != nil {
			t.Fatalf("type tag: %v", err)
		}
		if string(name) == field {
			if typ != wire.OperateTypeTable {
				t.Fatalf("field %q is not a table", field)
			}
			inner, _, err := dynamicTableInner(rec, valOff)
			if err != nil {
				t.Fatalf("table body: %v", err)
			}
			_, rowsOff, err := dynamicTableRows(inner)
			if err != nil {
				t.Fatalf("table rows: %v", err)
			}
			return inner, rowsOff
		}
		if off, err = skipDynamicValue(rec, valOff, typ, n); err != nil {
			t.Fatalf("skip: %v", err)
		}
	}
	t.Fatalf("no field %q in the record", field)
	return nil, 0
}

// dynamicRowSpans returns each row unit's [start,end) inside inner, the unit
// being its length prefix plus its body.
func dynamicRowSpans(t *testing.T, inner []byte, rowsOff int) [][2]int {
	t.Helper()
	nRows, _, err := dynamicTableRows(inner)
	if err != nil {
		t.Fatalf("table rows: %v", err)
	}
	spans := make([][2]int, 0, nRows)
	off := rowsOff
	for i := uint64(0); i < nRows; i++ {
		start := off
		rowLen, m, err := readCanonicalUvarint(inner, off)
		if err != nil {
			t.Fatalf("row length: %v", err)
		}
		off += m + int(rowLen)
		spans = append(spans, [2]int{start, off})
	}
	return spans
}
