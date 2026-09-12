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

// dynamicEnginePool does the same for dynamic-mode records (design doc §2.9),
// whose engine carries a field index and an encode scratch buffer worth
// keeping between calls for the same reason.
var dynamicEnginePool = sync.Pool{New: func() any { return new(dynamicEngine) }}

// modeEngine is the engine of one stored record plus the one thing
// applyWithEngine needs that the engine interface does not carry: the
// "did this record exist before the call" flag a record-path EXISTS/ABSENT
// compares (design doc §2.4), which the returns see as true once the call
// has stored something.
type modeEngine interface {
	engine
	setExisted(v bool)
}

// setExisted implements modeEngine for the schema engine.
func (e *schemaEngine) setExisted(v bool) { e.existed = v }

// engines is the pair of pooled engines one call borrows: which of the two
// runs is decided by the stored record's mode byte, and the other costs
// nothing but a pool round trip.
type engines struct {
	se *schemaEngine
	de *dynamicEngine
}

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
func applyRecordBytes(cur []byte, a *wire.OperateArgs, stampMs int64) ([]byte, bool, wire.OperateResult, error) {
	return applyRecordBytesInto(nil, cur, a, stampMs)
}

// applyRecordBytesInto is applyRecordBytes with a caller-owned scratch buffer
// for the private record copy — the one allocation the in-place path otherwise
// costs (design doc §2.6).
//
// scratch may be nil, in which case this is exactly applyRecordBytes.
//
// The returned out ALIASES scratch whenever scratch's array had room — which
// includes a record that GREW, since insertGap extends in place while capacity
// allows (`cap(buf)-old >= n`) and only allocates a replacement when it does
// not. So growth alone does not tell a caller whether out is a new array;
// capacity does, and the caller cannot see it.
//
// A caller recycling the scratch should recycle what out POINTS AT, but that is
// a CAPACITY contract, not a safety one: when growth allocated a replacement the
// two arrays no longer alias, so keeping the original is harmless — it just
// discards the larger array and makes the next call grow again.
//
// The safety constraint is ordering: out stays valid only until the caller
// reuses the scratch, so everything that reads out must happen first. tx.Put
// copies the bytes into the shard's page arena (encodeEntry), so the store
// itself never holds onto it.
func applyRecordBytesInto(scratch, cur []byte, a *wire.OperateArgs, stampMs int64) ([]byte, bool, wire.OperateResult, error) {
	if len(a.Ops) > wire.OperateMaxOps || len(a.Rets) > wire.OperateMaxRet {
		return nil, false, wire.OperateResult{}, wire.ErrOperateCap
	}
	es := engines{
		se: schemaEnginePool.Get().(*schemaEngine),   //nolint:errcheck,forcetypeassert // the pool's New returns exactly this
		de: dynamicEnginePool.Get().(*dynamicEngine), //nolint:errcheck,forcetypeassert // the pool's New returns exactly this
	}
	out, deleted, res, err := applyWithEngine(es, scratch, cur, a, stampMs)
	es.se.clear()
	es.de.clear()
	schemaEnginePool.Put(es.se)
	dynamicEnginePool.Put(es.de)
	return out, deleted, res, err
}

func applyWithEngine(es engines, scratch, cur []byte, a *wire.OperateArgs, stampMs int64) ([]byte, bool, wire.OperateResult, error) {
	e, err := openRecord(es, scratch, cur, a)
	if err != nil {
		return nil, false, wire.OperateResult{}, err
	}

	status, failedOp, err := applyOps(e, a.Ops, stampMs)
	if err != nil {
		return nil, false, wire.OperateResult{}, err
	}
	if status == wire.OperateStatusCheckFailed {
		// The op list aborts with the record unchanged, and the returns are
		// evaluated against that unchanged record.
		vals, verr := retsBefore(es, cur, a.Rets)
		if verr != nil {
			return nil, false, wire.OperateResult{}, verr
		}
		return nil, false, wire.OperateResult{Status: status, FailedOp: failedOp, Values: vals}, nil
	}

	if e.empty() {
		// The record is gone, so every return is absent (oracle ruling): there
		// is nothing left to resolve a path against.
		vals, verr := absentRets(a.Rets)
		if verr != nil {
			return nil, false, wire.OperateResult{}, verr
		}
		return nil, true, wire.OperateResult{Status: wire.OperateStatusOK, Values: vals}, nil
	}

	out := e.bytes()
	// §2.7's backstop, checked on the finished record: a call that would store
	// an over-large record fails with the record unchanged.
	if len(out) > maxOperateRecordBytes {
		return nil, false, wire.OperateResult{}, wire.ErrOperateCap
	}
	// The returns see the record as it now is, so a record path is present
	// even when this call created it — ref.present is "existed before the
	// call" only for the op list's own EXISTS/ABSENT (the oracle evaluates
	// its success-path returns with existed = true for the same reason).
	e.setExisted(true)
	vals, verr := evalRets(e, a.Rets)
	if verr != nil {
		return nil, false, wire.OperateResult{}, verr
	}
	return out, false, wire.OperateResult{Status: wire.OperateStatusOK, Values: vals}, nil
}

// openRecord resolves the call's `create` parameter against the stored record
// (design doc §3.5) and returns an engine over a private, patchable copy of
// the bytes to apply the ops to.
func openRecord(es engines, scratch, cur []byte, a *wire.OperateArgs) (modeEngine, error) {
	if cur == nil {
		return createRecord(es, scratch, a)
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

	buf, err := copyRecord(scratch, cur)
	if err != nil {
		return nil, err
	}
	e, err := openMode(es, buf, true)
	if err != nil {
		return nil, err
	}
	if callSchema != nil {
		// Ruling (the oracle's): §2.8 compares versions, not blobs. The stored
		// schema stays authoritative for the record; the call's blob only has
		// to agree on the version, so a differently encoded blob of the same
		// version is accepted and changes nothing.
		if es.se.schema.Version != callSchema.Version {
			return nil, wire.ErrOperateSchemaVersion
		}
	}
	return e, nil
}

// createRecord builds the record a call creates when the key is absent
// (design doc §3.5).
// createRecord takes the caller's scratch for the same reason openRecord does:
// a first write to a key allocated a fresh record every time, even though the
// handler already holds a pooled buffer that is about to be idle.
func createRecord(es engines, scratch []byte, a *wire.OperateArgs) (modeEngine, error) {
	switch a.Create {
	case wire.OperateCreateNone:
		return nil, errOperateAbsent
	case wire.OperateCreateSchema:
		buf, err := newSchemaRecordInto(scratch, a.Schema, operateSchemas)
		if err != nil {
			return nil, err
		}
		return openMode(es, buf, false)
	case wire.OperateCreateDynamic:
		// An empty dynamic record: the mode byte and a field count of zero
		// (design doc §2.9). The slack is what a first field's splice grows
		// into without reallocating.
		buf := scratch[:0]
		if cap(buf) < 64 {
			buf = make([]byte, 0, 64)
		}
		return openMode(es, append(buf, wire.OperateModeDynamic, 0), false)
	default:
		return nil, wire.ErrOperateArgs
	}
}

// openMode builds the engine for a record's stored mode byte: the stored
// bytes, not the call, decide which of the two engines runs (design doc
// §2.9, "the mode byte on the record and the create byte on the call are the
// whole contract").
func openMode(es engines, buf []byte, existed bool) (modeEngine, error) {
	if len(buf) < 1 {
		return nil, wire.ErrOperateRecord
	}
	switch buf[0] {
	case wire.OperateModeSchema:
		if err := es.se.reset(buf, operateSchemas); err != nil {
			return nil, err
		}
		es.se.existed = existed
		return es.se, nil
	case wire.OperateModeDynamic:
		if err := es.de.reset(buf, operateSchemas); err != nil {
			return nil, err
		}
		es.de.existed = existed
		return es.de, nil
	default:
		return nil, wire.ErrOperateRecord
	}
}

// copyRecord copies cur into a buffer the engine owns, with slack so a single
// row insert usually does not reallocate. The stored value aliases the
// store's cache page (design doc §2.6): it must never be written through.
//
// The §2.7 record-size cap is checked BEFORE the copy, not after: the value
// under the key is whatever a plain `put` left there, so it can be far larger
// than any operate record may be, and copying it first would let one call
// allocate an arbitrary multiple of the cap before the engine ever looked at
// the bytes. A stored value that big cannot be a valid operate record, so it
// is wire.ErrOperateRecord — a malformed stored record — rather than
// wire.ErrOperateCap, which means "this call would grow the record too far".
//
// FRESHNESS IS LOAD-BEARING FOR A CALLER OUTSIDE THIS PACKAGE. vector's
// RecordMutator contract hands ownership of the returned bytes to the engine,
// which stores them BY REFERENCE in the arena. That is sound only while the
// bytes are freshly allocated, so vector must keep calling applyRecordBytes —
// the nil-scratch path, where this function allocates per call and createRecord
// allocates fresh. Do not hand the vector path a scratch buffer.
//
// The KV path may reuse one (applyRecordBytesInto, from handleOperate's pool)
// for the opposite reason: tx.Put COPIES the record into the shard's page arena
// (encodeEntry), so the store holds no reference once Put returns.
// copyRecordNeed is the buffer size copyRecord wants for a record of n bytes:
// the record, slack to grow into, and a floor. Exported to the package so the
// allocation-budget tests size their scratch from the SAME expression rather
// than re-encoding it — a test that guesses "big enough" stops pinning the
// formula the moment the formula changes.
//
// Capped so the result stays POOLABLE: putOperateWriteBuf drops anything over
// maxPooledReadBuf, so a record just under that ceiling must not be given slack
// that pushes it over, or its buffer is dropped after every single update and
// the pooling silently stops working for exactly the largest records.
func copyRecordNeed(n int) int {
	need := n + n/4 + 64
	if need > maxPooledReadBuf && n <= maxPooledReadBuf {
		return maxPooledReadBuf
	}
	return need
}

// copyRecord copies cur into dst, reusing dst's array when it is already large
// enough and allocating only when it is not. The +64 of slack is what a first
// splice or row insert grows into without reallocating (see insertGap).
//
// dst is the CALLER's buffer: applyRecordBytesInto threads handleOperate's
// pooled scratch down to here, so a warm pool makes the in-place apply path
// allocate nothing at all. Passing nil is the allocating path every other
// caller takes.
func copyRecord(dst, cur []byte) ([]byte, error) {
	if len(cur) > maxOperateRecordBytes {
		return nil, wire.ErrOperateRecord
	}
	// Slack scales with the record, not a flat +64. A record holding tables
	// grows a row at a time, and a single insert routinely exceeds 64 bytes, so
	// a flat reservation leaves insertGap no room and it reallocates. See
	// copyRecordNeed for the size and why it is capped.
	if need := copyRecordNeed(len(cur)); cap(dst) < need {
		dst = make([]byte, 0, need)
	}
	return append(dst[:0], cur...), nil
}

// retsBefore evaluates the return specs against the pre-call record after a
// failed CHECK. Against a record that did not exist before the call every
// return is absent, without resolving anything (oracle ruling).
func retsBefore(es engines, cur []byte, rets []wire.OperateRet) ([][]byte, error) {
	if len(rets) == 0 {
		return nil, nil
	}
	if cur == nil {
		return absentRets(rets)
	}
	// Re-opening the pre-call bytes reuses the same pooled engines: the
	// working copy they were pointing at is discarded whole by a failed
	// CHECK, so nothing in it is still needed.
	//
	// It does NOT reuse the apply scratch, deliberately. This is the aborted
	// path, not the hot one, and the values below alias the buffer they are
	// read out of — sharing the scratch here would tie their lifetime to a
	// buffer the handler is about to recycle, for no gain on any real call.
	buf, err := copyRecord(nil, cur)
	if err != nil {
		return nil, err
	}
	e, err := openMode(es, buf, true)
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
