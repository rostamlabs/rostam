// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"

	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// ErrVectorRecordAbsent is the vector twin of handleOperate's cache.ErrNotFound
// mapping: the point exists, but its payload key holds no record and the call
// asked for create = NONE. A caller who declined to create a record and found
// none is told so, rather than silently getting an OK for a call that changed
// nothing.
var ErrVectorRecordAbsent = errors.New("ops: vector_operate: create=NONE and no record under the payload key")

// IsVectorRecordAbsentMessage reports whether s is the EXACT serialised form
// of ErrVectorRecordAbsent — anchored the same way vector.IsRecordTooLargeMessage
// and its Phase 2 twins are for their own sentinels, and for the same reason:
// shard.decodePBResult rebuilds a replicated op error with errors.New, losing
// errors.Is identity, and a bare strings.Contains fallback would make any
// error that merely mentions the sentinel text client-facing.
//
// Unlike the vector-package sentinels this one has exactly ONE shape: bare
// equality, no detail suffix and no bulk wrapper. vectorOperateMutator returns
// it verbatim (return nil, vector.RecordUnchanged, ErrVectorRecordAbsent), and
// every one of its three callers — handleVectorOperate, handleNamedVectorOperate,
// handleMVVectorOperate — propagates whatever the engine returned unwrapped
// ("return nil, err"), so no call site ever attaches a key, a kind, or a bulk
// index to it.
func IsVectorRecordAbsentMessage(s string) bool {
	return s == ErrVectorRecordAbsent.Error()
}

// vectorOperateMutator builds the RecordMutator the three handlers hand to the
// engine, plus the pointer the decoded result lands in.
//
// The mutator is the ONLY place any operate semantics happen, and all of it is
// applyRecordBytes — the same pure function the KV handler drives, with the same
// leader stamp as its only clock. What is left here is the translation between
// applyRecordBytes's (out, deleted, res, err) and vector's three-outcome
// RecordMutation contract:
//
//	failed CHECK → RecordUnchanged   (the point must stay byte-identical AND
//	                                  unbumped, which "store these bytes" cannot
//	                                  express)
//	deleted      → RecordDelete      (the op list emptied the record; the payload
//	                                  key goes away entirely)
//	otherwise    → RecordStore
//
// errOperateAbsent — create = NONE against an absent record — is mapped at this
// boundary to ErrVectorRecordAbsent, exactly as handleOperate maps it to
// cache.ErrNotFound. Every other error is returned verbatim so the engine can
// leave the point exactly as it was and the caller sees the real reason.
//
// Both error returns pass vector.RecordUnchanged explicitly. The engine tests
// err != nil first, so the action is never read on an error — but a bare zero
// would silently read as RecordStore if that order ever changed.
func vectorOperateMutator(a *wire.OperateArgs, stampMs int64, res *wire.OperateResult) vector.RecordMutator {
	return func(old []byte, exists bool) ([]byte, vector.RecordMutation, error) {
		var cur []byte
		if exists {
			cur = old
		}
		out, deleted, r, aerr := applyRecordBytes(cur, a, stampMs)
		if aerr != nil {
			if errors.Is(aerr, errOperateAbsent) {
				return nil, vector.RecordUnchanged, ErrVectorRecordAbsent
			}
			return nil, vector.RecordUnchanged, aerr
		}
		*res = r
		switch {
		case r.Status == wire.OperateStatusCheckFailed:
			return nil, vector.RecordUnchanged, nil
		case deleted:
			return nil, vector.RecordDelete, nil
		default:
			return out, vector.RecordStore, nil
		}
	}
}

// handleVectorOperate applies one operate op-list to the record held in a
// point's payload under payloadKey. It is handleOperate's vector twin: the SAME
// applyRecordBytes decides everything semantic, and this handler only supplies
// the record bytes, the leader-stamped clock, and the reply frame.
//
// The op-list runs INSIDE the collection's write lock, as the mutator the engine
// calls. That is what makes the read-modify-write atomic, and it is bounded by
// the same caps a KV operate is (OperateMaxOps ops, a bounded record), so the
// lock is held for a bounded, caller-declared amount of work.
//
// A missing/dead point is the not-found FLAG (found=0), never an op error, so a
// fan-out treats it the way it treats a set_payload against a missing point
// (payloadAppliedV). A point that exists but whose payload key holds no record
// and create = NONE is ErrVectorRecordAbsent — a real error, mirroring
// handleOperate's cache.ErrNotFound mapping.
//
// Determinism: the stamp is the ONLY clock, exactly as in handleOperate —
// unstamped (single-node) calls pass 0, which is what a KV operate does too, and
// route to the wall-clock engine variant exactly as SetPayload does beside
// SetPayloadAt.
//
// The decoder already rejects an inner Key and any ttlMode other than KEEP (a
// point's TTL is the point's), but the handler defends anyway: a future decoder
// change, or a direct call, must not be able to smuggle a second target or a
// silently-ignored TTL past this boundary.
func handleVectorOperate(tx *TxContext, args []byte) ([]byte, error) {
	return vectorOperateBody(tx, args, applyDenseRecordMutation)
}

// recordMutateApply is a family's ENGINE ENTRY, and the only thing the three
// vector_operate handlers do not share. It picks the stamped or unstamped
// variant, which is the same decision in every family: only a stamped apply has
// a clock every replica agrees on (see vector.recordKeyPastDeadline).
//
// The three implementations are package-level functions, not closures, so
// passing one costs no allocation on the apply path.
type recordMutateApply func(v *vector.CollectionStore, name string, id uint64, pk string,
	fn vector.RecordMutator, cas vector.CASCond, stampMs int64, stamped bool) (applied bool, version uint64, err error)

func applyDenseRecordMutation(v *vector.CollectionStore, name string, id uint64, pk string,
	fn vector.RecordMutator, cas vector.CASCond, stampMs int64, stamped bool,
) (bool, uint64, error) {
	if stamped {
		return v.MutatePayloadRecordCASAt(name, id, pk, fn, cas, stampMs)
	}
	return v.MutatePayloadRecordCAS(name, id, pk, fn, cas)
}

func applyNamedRecordMutation(v *vector.CollectionStore, name string, id uint64, pk string,
	fn vector.RecordMutator, cas vector.CASCond, stampMs int64, stamped bool,
) (bool, uint64, error) {
	if stamped {
		return v.NamedMutatePayloadRecordCASAt(name, id, pk, fn, cas, stampMs)
	}
	return v.NamedMutatePayloadRecordCAS(name, id, pk, fn, cas)
}

func applyMVRecordMutation(v *vector.CollectionStore, name string, docID uint64, pk string,
	fn vector.RecordMutator, cas vector.CASCond, stampMs int64, stamped bool,
) (bool, uint64, error) {
	if stamped {
		return v.MVMutatePayloadRecordCASAt(name, docID, pk, fn, cas, stampMs)
	}
	return v.MVMutatePayloadRecordCAS(name, docID, pk, fn, cas)
}

// vectorOperateBody is the whole of a vector_operate handler except its engine
// entry: the decode, the argument re-assertion, the CAS threading, the mutator
// construction and the result frame.
//
// It exists as ONE copy because the three families' handlers were three copies
// of it, and a validation or result-format fix landing in only two of them is a
// silent divergence between ops that are documented as identical. This is the
// same reasoning the Store layer already follows, where one shared body serves
// nine methods — that duplication is how set_payload lost its CAS precondition
// on three of its four paths.
func vectorOperateBody(tx *TxContext, args []byte, apply recordMutateApply) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, id, pk, a, expected, hasExpected, err := wire.DecodeVectorOperateArgs(args)
	if err != nil {
		return nil, err
	}
	if err := checkVectorOperateArgs(a); err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	stampMs, stamped := tx.applyStamp()
	var res wire.OperateResult
	fn := vectorOperateMutator(a, stampMs, &res)

	applied, version, err := apply(tx.vectors, name, id, pk, fn, cas, stampMs, stamped)
	if err != nil {
		return nil, err // incl. ErrVersionConflict, ErrPayloadKeyNotRecord, ErrRecordTooLarge
	}
	// version is the point's version AFTER the call — bumped when the op-list
	// applied, and the CURRENT unbumped one when it was a deliberate no-op (a
	// failed CHECK). Returning it lets a CAS loop feed the next attempt straight
	// from this result instead of re-reading the point, which is both a round trip
	// and a race.
	return wire.EncodeVectorOperateResult(applied, &res, version)
}

// handleNamedVectorOperate is handleVectorOperate against a named-vector
// collection's shared payload. See handleVectorOperate for the whole contract;
// only the engine entry point differs.
func handleNamedVectorOperate(tx *TxContext, args []byte) ([]byte, error) {
	return vectorOperateBody(tx, args, applyNamedRecordMutation)
}

// handleMVVectorOperate is handleVectorOperate against a multi-vector
// collection's document payload. See handleVectorOperate for the whole contract;
// only the engine entry point differs.
func handleMVVectorOperate(tx *TxContext, args []byte) ([]byte, error) {
	return vectorOperateBody(tx, args, applyMVRecordMutation)
}

// checkVectorOperateArgs re-asserts, at the handler boundary, the two rules the
// vector_operate codec already enforces on both sides of the wire: the target is
// named exactly once by (collection, id, payloadKey) — so the inner Key must be
// empty — and a point's TTL is the point's, so ttlMode must be KEEP with no ttl.
//
// This is defence in depth, not belt-and-braces duplication. Rejecting beats
// ignoring for both rules (a caller who set either expecting it to work would
// otherwise get silence), and the decoder is not the only way an OperateArgs can
// reach these handlers — a future codec revision or an in-process caller must
// fail loudly rather than quietly acquire a second source of truth for the
// target key, or a TTL that is silently dropped.
func checkVectorOperateArgs(a *wire.OperateArgs) error {
	if a == nil {
		return wire.ErrOperateArgs
	}
	if len(a.Key) != 0 || a.TTLMode != wire.OperateTTLKeep || a.TTL != 0 {
		return wire.ErrOperateArgs
	}
	return nil
}
