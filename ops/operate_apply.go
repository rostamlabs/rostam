// SPDX-License-Identifier: Apache-2.0

package ops

// applyRecordBytes is the pure core of the operate op: stored record bytes in,
// stored record bytes out, with no store, no clock, and no lock of its own.
// The handler (design doc §3.5) wraps it with the key lookup, the TTL rules
// and the reply frame; everything below is a deterministic function of
// (cur, args, stampMs), which is what makes an operate call replayable on
// every replica.

import (
	"sync"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// schemaEnginePool keeps schema engines (and, with them, their tail index and
// cell scratch) alive between calls: the design doc's §2.6 budget for the
// in-place path is one allocation for the record copy, which leaves no room
// for a fresh engine per call on a hot key.
var schemaEnginePool = sync.Pool{New: func() any { return new(schemaEngine) }}

// applyRecordBytes applies one operate call to a stored record.
//
// cur is the record's stored bytes, or nil when the key is absent; it is never
// mutated — the engine works on a private copy.
//
// out is what the caller should store, and it is nil whenever the call stores
// nothing: on a failed CHECK (a no-op, whose result still carries
// OperateStatusCheckFailed and the return specs evaluated against the pre-call
// record, design doc §3.3), on any error, and when deleted is true — which
// means "delete the key" rather than "store nothing" (design doc §2.5). out is
// never an alias of cur: the caller can write it back without thinking about
// the store's own page.
func applyRecordBytes(cur []byte, a *wire.OperateArgs, stampMs int64) ([]byte, bool, *wire.OperateResult, error) {
	if len(a.Ops) > wire.OperateMaxOps || len(a.Rets) > wire.OperateMaxRet {
		return nil, false, nil, wire.ErrOperateCap
	}
	se := schemaEnginePool.Get().(*schemaEngine) //nolint:errcheck,forcetypeassert // the pool's New returns exactly this
	out, deleted, res, err := applyWithEngine(se, cur, a, stampMs)
	se.clear()
	schemaEnginePool.Put(se)
	return out, deleted, res, err
}

func applyWithEngine(se *schemaEngine, cur []byte, a *wire.OperateArgs, stampMs int64) ([]byte, bool, *wire.OperateResult, error) {
	e, err := openRecord(se, cur, a)
	if err != nil {
		return nil, false, nil, err
	}

	status, failedOp, err := applyOps(e, a.Ops, stampMs)
	if err != nil {
		return nil, false, nil, err
	}
	if status == wire.OperateStatusCheckFailed {
		// The op list aborts with the record unchanged, and the returns are
		// evaluated against that unchanged record.
		vals, verr := retsBefore(se, cur, a.Rets)
		if verr != nil {
			return nil, false, nil, verr
		}
		return nil, false, &wire.OperateResult{Status: status, FailedOp: failedOp, Values: vals}, nil
	}

	if e.empty() {
		// The record is gone, so every return is absent (oracle ruling): there
		// is nothing left to resolve a path against.
		vals, verr := absentRets(a.Rets)
		if verr != nil {
			return nil, false, nil, verr
		}
		return nil, true, &wire.OperateResult{Status: wire.OperateStatusOK, Values: vals}, nil
	}

	out := e.bytes()
	// §2.7's backstop, checked on the finished record: a call that would store
	// an over-large record fails with the record unchanged.
	if len(out) > maxOperateRecordBytes {
		return nil, false, nil, wire.ErrOperateCap
	}
	// The returns see the record as it now is, so a record path is present
	// even when this call created it — ref.present is "existed before the
	// call" only for the op list's own EXISTS/ABSENT (the oracle evaluates
	// its success-path returns with existed = true for the same reason).
	se.existed = true
	vals, verr := evalRets(e, a.Rets)
	if verr != nil {
		return nil, false, nil, verr
	}
	return out, false, &wire.OperateResult{Status: wire.OperateStatusOK, Values: vals}, nil
}

// openRecord resolves the call's `create` parameter against the stored record
// (design doc §3.5) and returns an engine over a private, patchable copy of
// the bytes to apply the ops to.
func openRecord(se *schemaEngine, cur []byte, a *wire.OperateArgs) (engine, error) {
	if cur == nil {
		return createRecord(se, a)
	}
	if len(cur) < 1 {
		return nil, wire.ErrOperateRecord
	}
	mode := cur[0]

	var callSchema *wire.Schema
	switch a.Create {
	case wire.OperateCreateNone:
		// Ruling (the oracle's): a create = NONE call carries no schema blob,
		// so one that arrives anyway is ignored rather than checked.
	case wire.OperateCreateSchema:
		if len(a.Schema) == 0 {
			return nil, wire.ErrOperateSchema
		}
		s, _, err := operateSchemas.get(a.Schema)
		if err != nil {
			return nil, err
		}
		callSchema = s
		if mode != wire.OperateModeSchema {
			return nil, wire.ErrOperateMode
		}
	case wire.OperateCreateDynamic:
		if mode != wire.OperateModeDynamic {
			return nil, wire.ErrOperateMode
		}
	default:
		return nil, wire.ErrOperateArgs
	}

	e, err := openMode(se, copyRecord(cur), true)
	if err != nil {
		return nil, err
	}
	if callSchema != nil {
		// Ruling (the oracle's): §2.8 compares versions, not blobs. The stored
		// schema stays authoritative for the record; the call's blob only has
		// to agree on the version, so a differently encoded blob of the same
		// version is accepted and changes nothing.
		if se.schema.Version != callSchema.Version {
			return nil, wire.ErrOperateSchemaVersion
		}
	}
	return e, nil
}

// createRecord builds the record a call creates when the key is absent
// (design doc §3.5).
func createRecord(se *schemaEngine, a *wire.OperateArgs) (engine, error) {
	switch a.Create {
	case wire.OperateCreateNone:
		return nil, errOperateAbsent
	case wire.OperateCreateSchema:
		buf, err := newSchemaRecord(a.Schema, operateSchemas)
		if err != nil {
			return nil, err
		}
		return openMode(se, buf, false)
	case wire.OperateCreateDynamic:
		return openMode(se, []byte{wire.OperateModeDynamic, 0}, false)
	default:
		return nil, wire.ErrOperateArgs
	}
}

// openMode builds the engine for a record's stored mode byte. The dynamic
// engine lands in a later task; until then a dynamic record is a mode this
// build cannot apply to.
func openMode(se *schemaEngine, buf []byte, existed bool) (engine, error) {
	if len(buf) < 1 {
		return nil, wire.ErrOperateRecord
	}
	switch buf[0] {
	case wire.OperateModeSchema:
		if err := se.reset(buf, operateSchemas); err != nil {
			return nil, err
		}
		se.existed = existed
		return se, nil
	case wire.OperateModeDynamic:
		return nil, wire.ErrOperateMode
	default:
		return nil, wire.ErrOperateRecord
	}
}

// copyRecord copies cur into a buffer the engine owns, with slack so a single
// row insert usually does not reallocate. The stored value aliases the
// store's cache page (design doc §2.6): it must never be written through.
func copyRecord(cur []byte) []byte {
	buf := make([]byte, len(cur), len(cur)+64)
	copy(buf, cur)
	return buf
}

// retsBefore evaluates the return specs against the pre-call record after a
// failed CHECK. Against a record that did not exist before the call every
// return is absent, without resolving anything (oracle ruling).
func retsBefore(se *schemaEngine, cur []byte, rets []wire.OperateRet) ([][]byte, error) {
	if len(rets) == 0 {
		return nil, nil
	}
	if cur == nil {
		return absentRets(rets)
	}
	e, err := openMode(se, copyRecord(cur), true)
	if err != nil {
		return nil, err
	}
	return evalRets(e, rets)
}

// absentRets is what every return spec evaluates to when there is no record
// to resolve against: UNSET for a VALUE, a tagged zero for a COUNT (design
// doc §2.4/§3.4).
func absentRets(rets []wire.OperateRet) ([][]byte, error) {
	if len(rets) == 0 {
		return nil, nil
	}
	out := make([][]byte, 0, len(rets))
	for i := range rets {
		switch rets[i].Mode {
		case wire.OperateRetCount:
			out = append(out, appendCountValue(0))
		case wire.OperateRetValue:
			out = append(out, []byte{wire.OperateTypeUnset})
		default:
			return nil, wire.ErrOperateArgs
		}
	}
	return out, nil
}
