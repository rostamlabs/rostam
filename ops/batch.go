// SPDX-License-Identifier: Apache-2.0

package ops

import "github.com/rostamlabs/rostam/sdk/wire"

// handlePutBatch applies every put in the batch within one TxContext — one Raft
// log entry / fsync / round-trip / apply for the whole batch.
//
// It decodes the ENTIRE buffer first, so a malformed (truncated) batch applies
// nothing (structural errors are atomic). It then applies each entry
// independently, CONTINUING past a per-entry tx.Put failure and returning the
// first such error. This makes a put_batch state-equivalent to applying the same
// keys as N sequential single puts: a capacity/quota rejection on one entry
// affects only that entry (as it would for a lone put), never the rest of the
// batch — so put_batch cannot amplify a replica-local capacity decision into a
// whole-tail divergence. tx.Put normally only errors on validation/quota.
func handlePutBatch(tx *TxContext, args []byte) ([]byte, error) {
	entries, err := wire.DecodePutBatchArgs(args)
	if err != nil {
		return nil, err
	}
	var firstErr error
	for _, e := range entries {
		if perr := tx.Put(e.Key, e.Val, e.TTL); perr != nil {
			if firstErr == nil {
				firstErr = perr
			}
			// This entry stored nothing, so it posts nothing — the same
			// "N independent puts" equivalence the doc above states.
			continue
		}
		// Per entry, and AFTER its own Put: an evicting Put can fire onRemove
		// for the very key it is writing, so a posting made first would be
		// dropped by that eviction.
		tx.reindexKV(e.Key, e.Val)
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return wire.EncodePutBatchResult(len(entries)), nil
}
