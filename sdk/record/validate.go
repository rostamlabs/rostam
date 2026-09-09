// SPDX-License-Identifier: Apache-2.0

package record

import (
	"fmt"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// Validate reports whether rec is a record the operate engines can open: its
// acceptance set is EXACTLY wire.DecodeRecord's, wrapped so callers outside the
// sdk module's wire package (the vector engine) do not need to import it. It
// returns nil, or an error wrapping ErrRecord.
//
// It is the INGEST gate. Every wire-reachable entry that stores a ValueRecord
// runs it before any state changes, so from this phase onward no stored record
// can be one the operate engine, IndexEntries or DecodeRecord would refuse.
//
// It is STRICTER than Resolve, deliberately. Resolve reads only the bytes on
// its own path and is documented as answering Absent for shapes DecodeRecord
// rejects (an unsorted row block, for one), and IndexEntries — which walks
// every field — rejects records Resolve still answers from. That asymmetry is
// what poisons a payload key: a record that indexes nothing while it still
// resolves makes every empty-set inference under that key a lie, so phase 1 had
// to fail closed on it. Validating at ingest makes the divergence unreachable
// for anything stored from here on, WITHOUT changing Resolve, which must stay
// lenient for bytes written before it.
//
// WHY IT DECODES RATHER THAN WALKING. A validate-only pass that did not build
// the tree would be a SECOND acceptance set to hold in step with DecodeRecord's
// — and two readers disagreeing about one record is the exact failure this gate
// exists to prevent, so the cheaper walk would reintroduce the class it closes.
// wire exposes no such walk today (sdk/wire/operate_record.go: DecodeRecord is
// the only entry) and adding one is not worth that risk.
//
// COST. DecodeRecord builds the full tree, so this is O(len(rec)) with
// allocations proportional to the record's rows and fields, paid once per
// ingested record VALUE on the write path — not per point and not per read.
// Records are capped at 16 MiB (vector's maxRecordValueBytes, checked first so
// an oversize value is refused without being decoded) and the caller is already
// copying them. BenchmarkValidate measures both ends of the range.
func Validate(rec []byte) error {
	if _, err := wire.DecodeRecord(rec); err != nil {
		return fmt.Errorf("%w: %w", ErrRecord, err)
	}
	return nil
}
