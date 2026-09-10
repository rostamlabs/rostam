// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"errors"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// mapErr translates internal package errors to the public Store sentinels.
// Unknown errors are returned as-is.
func mapErr(err error) error {
	if err == nil {
		return nil
	}
	if matchesNotFound(err) {
		return ErrNotFound
	}
	if matchesNotLeader(err) {
		return ErrNotLeader
	}
	return err
}

// matchesNotFound returns true for cache.ErrNotFound or client.ErrNotFound.
func matchesNotFound(err error) bool {
	return err == cache.ErrNotFound || err == client.ErrNotFound
}

// matchesNotLeader returns true for shard.NotLeaderError (embedded path)
// or client.ErrNoLeaderKnown (networked path).
func matchesNotLeader(err error) bool {
	if err == client.ErrNoLeaderKnown {
		return true
	}
	var nle *shard.NotLeaderError
	return errors.As(err, &nle)
}

// mapVectorOperateErr normalizes the record-absent outcome of a vector_operate
// to the ONE sentinel Store.VectorOperate documents, ops.ErrVectorRecordAbsent,
// so `errors.Is(err, ops.ErrVectorRecordAbsent)` answers the same on every
// backend. It is applied by all three Store implementations' shared operate
// body and by nothing else.
//
// WHAT IT NORMALIZES, AND WHY THE THREE BACKENDS DISAGREED WITHOUT IT. The
// engine raises the sentinel in one place (ops.vectorOperateMutator), but only
// the direct path hands it back intact:
//
//   - DIRECT: the handler's error is returned in-process, identity intact —
//     already the sentinel, and this function leaves it alone.
//   - EMBEDDED, REPLICATED: an operate handler runs INSIDE the FSM apply, so
//     shard.decodePBResult rebuilds the error with errors.New across the Raft
//     boundary and errors.Is stops matching. The exact-form matcher
//     ops.IsVectorRecordAbsentMessage is what recognises it — never a
//     strings.Contains, which would also match an internal fault that merely
//     mentions the sentinel text.
//   - NETWORKED: server.mapResult answers StatusNotFound for it (deliberately:
//     it is the vector twin of the KV operate's not-found, and the status is
//     what carries the meaning), which the client turns into client.ErrNotFound
//     and mapErr into the root ErrNotFound. Not-found is the ONLY thing that
//     status can mean for these three ops — an absent POINT is the found=false
//     FLAG, not an error — so mapping it back is unambiguous.
//
// The caller-visible contract is therefore the sentinel on every transport, and
// the root ErrNotFound is no longer part of vector_operate's surface.
func mapVectorOperateErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ops.ErrVectorRecordAbsent) || ops.IsVectorRecordAbsentMessage(err.Error()) {
		return ops.ErrVectorRecordAbsent
	}
	if errors.Is(err, ErrNotFound) || errors.Is(err, client.ErrNotFound) || errors.Is(err, cache.ErrNotFound) {
		return ops.ErrVectorRecordAbsent
	}
	return err
}
