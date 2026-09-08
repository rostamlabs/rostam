// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// handleOperate is the server-side handler for the "operate" op (design doc
// §3.5): one atomic, multi-field read/check/write call against a single
// stored record, addressed and typed by the call's schema (or, in dynamic
// mode, the stored field set). Every semantic rule — path resolution, op
// dispatch, the record-size backstop, and what a call returns — lives in
// applyRecordBytes (ops/operate_apply.go, ops/operate_engine.go); this
// handler only wraps that pure function with the store's key lookup, the
// TTL rules below, and the reply frame.
//
// All-or-nothing: applyRecordBytes either returns a complete replacement for
// the stored bytes (out) or signals that nothing is to be stored (an error,
// a failed CHECK, or a cap hit never partially apply — the record is left
// exactly as it was). This handler performs exactly one store mutation per
// call — a single Put/PutAbs, or a single Del on deletion — never more than
// one, and never on an error or a failed CHECK.
//
// Determinism: the only clock this handler (and, through stampMs,
// applyRecordBytes) ever consults is tx.applyStamp() — the leader-stamped
// apply clock on a replicated apply, or the unstamped zero value otherwise —
// never a wall clock. Two replicas applying the same call against the same
// prior record, with the same stamp, therefore always produce identical
// stored bytes and identical expiries.
//
// cur aliasing: tx.GetWithExpiry's returned value aliases the cache's page.
// applyRecordBytes copies it before making any change (openRecord ->
// copyRecord) and returns its own, freshly allocated out; this handler never
// writes through cur and never touches it again once applyRecordBytes has
// been called.
func handleOperate(tx *TxContext, args []byte) ([]byte, error) {
	a, err := wire.DecodeOperateArgs(args)
	if err != nil {
		return nil, err
	}

	cur, expiryMs, err := tx.GetWithExpiry(a.Key)
	absent := errors.Is(err, cache.ErrNotFound)
	if err != nil && !absent {
		return nil, err
	}
	if absent {
		cur = nil
	}

	stampMs, _ := tx.applyStamp()
	out, deleted, res, err := applyRecordBytes(cur, a, stampMs)
	if err != nil {
		if errors.Is(err, errOperateAbsent) {
			// design doc §3.5: create = NONE against an absent key is the
			// store's own not-found error, not an operate-specific one.
			return nil, cache.ErrNotFound
		}
		return nil, err
	}

	switch {
	case res.Status == wire.OperateStatusCheckFailed:
		// A no-op (design doc §3.3): the record is unchanged and its TTL is
		// untouched. out is nil here (applyWithEngine never sets it on a
		// failed CHECK), so there is nothing to write even if this case were
		// skipped — but this way that invariant does not have to hold for
		// correctness.
	case deleted:
		// design doc §2.5. Deleting a key that was already absent is a
		// no-op: DEL () against nothing does not manufacture a tombstone.
		if !absent {
			if _, err := tx.Del(a.Key); err != nil {
				return nil, err
			}
		}
	default:
		switch a.TTLMode {
		case wire.OperateTTLSet:
			// Refreshed on every call, create or update alike.
			err = tx.Put(a.Key, out, a.TTL)
		case wire.OperateTTLCreateOnly:
			if absent {
				err = tx.Put(a.Key, out, a.TTL)
			} else {
				// Preserve the existing deadline verbatim (PutAbs takes no
				// stamp branch, so this is deterministic across replicas).
				err = tx.PutAbs(a.Key, out, expiryMs)
			}
		default: // wire.OperateTTLKeep
			if absent {
				err = tx.Put(a.Key, out, 0)
			} else {
				err = tx.PutAbs(a.Key, out, expiryMs)
			}
		}
		if err != nil {
			return nil, err
		}
	}

	return wire.EncodeOperateResult(res)
}
