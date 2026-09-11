// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"sync"

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
// cur ownership: tx.GetWithExpiryInto copies the stored record into a pooled
// buffer, so cur is owned by this call rather than aliasing the cache's page.
// It still must not outlive the handler - the buffer goes back to the pool on
// return - and nothing here needs it to: applyRecordBytes copies it again
// before making any change (openRecord -> copyRecord) and returns its own,
// freshly allocated out, and this handler never writes through cur nor touches
// it once applyRecordBytes has been called.
// operateArgsPool recycles the decoded call. The op slice is the single
// largest allocation on this path - one operate request carries up to
// OperateMaxOps ops, and a caller that touches many keys in one call sends a
// few ops per key - so decoding a fresh one per request dominated the server's
// allocation volume. Same pattern as schemaEnginePool/dynamicEnginePool.
//
// The decoded args alias the request buffer and applyRecordBytes only reads
// them during the call, so the struct is safe to return to the pool as soon
// as handleOperate returns.
var operateArgsPool = sync.Pool{New: func() any { return new(wire.OperateArgs) }}

// operateReadBufPool backs the pooled record read in handleOperate. Buffers
// grown past maxPooledReadBuf are dropped rather than pooled, to bound retained
// memory: a record may reach maxOperateRecordBytes, and a value written by a
// plain put can be larger still - GetWithExpiryInto grows the buffer to hold it
// before copyRecord rejects it - so pooling unconditionally would let a few
// outsized keys pin a large array per pool slot. Same guard as the client's
// kvArgsPool.
const maxPooledReadBuf = 64 << 10

var operateReadBufPool = sync.Pool{New: func() any { b := make([]byte, 0, 512); return &b }}

func putOperateReadBuf(bp *[]byte) {
	if cap(*bp) <= maxPooledReadBuf {
		*bp = (*bp)[:0]
		operateReadBufPool.Put(bp)
	}
}

func handleOperate(tx *TxContext, args []byte) ([]byte, error) {
	a, _ := operateArgsPool.Get().(*wire.OperateArgs)
	defer func() {
		// Clear before recycling: the ops hold Bytes/Name/path-Key slices into
		// the request buffer, and a pooled entry holding them would pin that
		// buffer until its next use.
		//
		// Clearing the live prefix is enough, by induction on this reset
		// rather than on anything the decoder does: an entry is handed back
		// with length 0 and a fully cleared array, so the next decode can only
		// dirty [0:len) and this clear puts it back. (The decoder's own
		// shrink-clear never fires here - it triggers on a SHRINK, and a
		// pooled entry always arrives at length 0. It is there for callers
		// that reuse a dst without a pool.)
		clear(a.Ops)
		clear(a.Rets)
		*a = wire.OperateArgs{Ops: a.Ops[:0], Rets: a.Rets[:0]}
		operateArgsPool.Put(a)
	}()
	if err := wire.DecodeOperateArgsInto(a, args); err != nil {
		return nil, err
	}

	// Pooled read buffer: the stored record is only read here and by
	// applyRecordBytes, which copies whatever it needs, so the bytes never
	// outlive the call. GetWithExpiryInto copies into this buffer instead of
	// allocating a fresh one per request - the read was the largest remaining
	// allocation on this path after the args pool.
	rb, _ := operateReadBufPool.Get().(*[]byte)
	defer func() { putOperateReadBuf(rb) }()
	cur, expiryMs, err := tx.GetWithExpiryInto((*rb)[:0], a.Key)
	*rb = cur
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
