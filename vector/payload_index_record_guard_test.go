// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// TestRecordPathIndexNarrowableGuard pins the shape+op guard itself: exactly
// the shapes reindex posts (a named top-level field, and a named field's
// "#count") are narrowable, and only for the ops whose posting set is exact.
func TestRecordPathIndexNarrowableGuard(t *testing.T) {
	narrowOps := []FilterOp{FilterEq, FilterIn, FilterGt, FilterGte, FilterLt, FilterLte,
		FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte}
	wideOps := []FilterOp{FilterContains, FilterMatch, FilterRegex, FilterIsEmpty, FilterIsNull,
		FilterRowExists, FilterRowAbsent, FilterNe}

	indexedShapes := []string{"session/rc", "session/b#count", "session/rc#count", "a/b"}
	declinedShapes := []string{"session/#0", "session/#0#count", "session/b/42", "session/b/42/hi"}

	for _, field := range indexedShapes {
		for _, op := range narrowOps {
			if !indexNarrowable(field, op) {
				t.Errorf("indexNarrowable(%q, %v) = false, want true (an indexed shape under an exact op)", field, op)
			}
		}
		for _, op := range wideOps {
			if indexNarrowable(field, op) {
				t.Errorf("indexNarrowable(%q, %v) = true, want false (no postings exist for this op)", field, op)
			}
		}
	}
	for _, field := range declinedShapes {
		for _, op := range append(append([]FilterOp{}, narrowOps...), wideOps...) {
			if indexNarrowable(field, op) {
				t.Errorf("indexNarrowable(%q, %v) = true, want false (this shape is never indexed)", field, op)
			}
		}
	}
	// Unchanged for everything that is not a record path.
	for _, op := range append(append([]FilterOp{}, narrowOps...), wideOps...) {
		if !indexNarrowable("country", op) {
			t.Errorf("indexNarrowable(country, %v) = false, want true", op)
		}
		if indexNarrowable(contentField, op) {
			t.Errorf("indexNarrowable($content, %v) = true, want false", op)
		}
	}
}
