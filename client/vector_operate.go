// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// OperateRequest is the payload for Collection.Operate.
type OperateRequest struct {
	ID         uint64
	PayloadKey string
	Args       *wire.OperateArgs
	// ExpectedVersion enables optimistic concurrency; HasExpectedVersion must be
	// true for it to apply.
	ExpectedVersion    uint64
	HasExpectedVersion bool
}

// Operate applies one atomic operate op-list to the record stored under
// req.PayloadKey in the point at req.ID (design doc for operate v2, §3-§3.5;
// record search brief §3). Build req.Args with NewOperate(nil) — the call names
// its target with Collection/ID/PayloadKey, so the builder's KEY stays empty and
// the builder's TTL must stay unset: a point's TTL belongs to the point, and
// EncodeVectorOperateArgs rejects a call that carries either.
//
// found is false when the point is absent, tombstoned or expired — a FLAG, not
// an error, exactly like every other point write. res is nil then.
//
// version is the point's version AFTER the call: bumped when the op-list
// applied, and the CURRENT unbumped one when it was a deliberate no-op (a failed
// CHECK). It is 0 when found is false. Feed it straight into the next call's
// ExpectedVersion to run a CAS loop without re-reading the point:
//
//	found, res, v, err := col.Operate(ctx, OperateRequest{ID: id, PayloadKey: k, Args: a})
//	// ... on ErrVersionConflict, retry with ExpectedVersion: v, HasExpectedVersion: true
//
// The re-read that would otherwise be needed is both a round trip and a race,
// since another writer can land between it and the retry.
//
// The op is NOT replayable (nonReplayableOp): an ADD applied twice after an
// ambiguous post-commit transport failure double-counts, so an ambiguous error
// surfaces to the caller instead of being retried.
func (col *Collection) Operate(ctx context.Context, req OperateRequest) (found bool, res *wire.OperateResult, version uint64, err error) {
	if req.Args == nil {
		return false, nil, 0, wire.ErrOperateArgs
	}
	args, err := wire.EncodeVectorOperateArgs(
		col.name, req.ID, req.PayloadKey, req.Args, req.ExpectedVersion, req.HasExpectedVersion)
	if err != nil {
		return false, nil, 0, err
	}
	body, err := col.c.Call(ctx, "vector_operate", args)
	if err != nil {
		return false, nil, 0, mapWriteErr(err)
	}
	return wire.DecodeVectorOperateResult(body)
}
