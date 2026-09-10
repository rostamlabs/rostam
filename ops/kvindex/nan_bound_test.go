// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// A NaN RANGE BOUND NARROWS TO THE EMPTY SET, AND MUST COST NOTHING TO SAY SO.
//
// IEEE-754 leaves every comparison against NaN unordered, so orderingHoldsFloat
// admits no posting whatsoever — the answer is empty before any value is looked
// at. Without a short circuit the range arm still walks every distinct value to
// discover that, charging one budget unit per examination, so on any index with
// more distinct values than the candidate budget a query that matches nothing
// came back as ErrCandidateBudget instead.
//
// It short-circuits to EMPTY rather than declining to drive the index:
// declining falls through to the leaf's scan-required refusal (or a full walk
// with scan consent), and "a NaN bound narrows to the empty set" is the
// contract docs/kv/querying-records.md states.
func TestCandidatesNaNBoundIsEmptyNotOverBudget(t *testing.T) {
	d := mustDef(t, "ix", "", "rc", wire.KVIndexKindScalar)
	s := readySet(d)
	const distinct = 500
	for i := 0; i < distinct; i++ {
		s.Reindex([]byte(fmt.Sprintf("k%04d", i)), intRec("rc", int64(i)))
	}

	nan := vtypes.NewFloat(math.NaN())
	for _, op := range []vtypes.FilterOp{vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte} {
		sel := Selector{Def: d, Op: op, Values: []vtypes.Value{nan}}
		// A budget far below the distinct-value count: reaching the walk at all
		// is what this asserts against.
		got, err := s.Candidates(sel, nil, false, 1)
		if errors.Is(err, ErrCandidateBudget) {
			t.Fatalf("op %v against a NaN bound over %d distinct values reported ErrCandidateBudget; it admits nothing and must answer empty without walking",
				op, distinct)
		}
		if err != nil {
			t.Fatalf("op %v against a NaN bound: %v", op, err)
		}
		if len(got) != 0 {
			t.Fatalf("op %v against a NaN bound returned %d candidates, want none", op, len(got))
		}
	}

	// The control: a real bound over the same index still walks and is still
	// charged, so the short circuit above is about NaN and nothing else.
	ordinary := Selector{Def: d, Op: vtypes.FilterGte, Values: []vtypes.Value{vtypes.NewInt(0)}}
	if _, err := s.Candidates(ordinary, nil, false, 1); !errors.Is(err, ErrCandidateBudget) {
		t.Fatalf("an ordinary range bound with budget 1 must still be refused, got %v", err)
	}
}
