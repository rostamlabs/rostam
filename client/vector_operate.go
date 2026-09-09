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
// The op is NOT replayable (nonReplayableOp): an ADD applied twice after an
// ambiguous post-commit transport failure double-counts, so an ambiguous error
// surfaces to the caller instead of being retried.
func (col *Collection) Operate(ctx context.Context, req OperateRequest) (bool, *wire.OperateResult, error) {
	if req.Args == nil {
		return false, nil, wire.ErrOperateArgs
	}
	args, err := wire.EncodeVectorOperateArgs(
		col.name, req.ID, req.PayloadKey, req.Args, req.ExpectedVersion, req.HasExpectedVersion)
	if err != nil {
		return false, nil, err
	}
	body, err := col.c.Call(ctx, "vector_operate", args)
	if err != nil {
		return false, nil, mapWriteErr(err)
	}
	return wire.DecodeVectorOperateResult(body)
}
