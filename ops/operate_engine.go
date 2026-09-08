// SPDX-License-Identifier: Apache-2.0

package ops

import "errors"

// maxOperateRecordBytes is the design doc §2.7 backstop on a stored operate
// record: after a call's ops apply, the encoded record must still fit, or the
// call fails with wire.ErrOperateCap and the record is left unchanged (a cap
// hit is never an implicit eviction). It is a var rather than a const so a
// test can lower it and exercise the bound without building 16 MiB of record.
var maxOperateRecordBytes = 16 << 20

var (
	// errOperateAbsent is the "create = NONE and the key does not exist"
	// error of design doc §3.5. The handler maps it to the store's
	// not-found error.
	errOperateAbsent = errors.New("ops: operate record absent")
	// errOperateIfRange marks an IF whose skip count is negative or reaches
	// past the end of the op list (design doc §3.3, "n bounded by the
	// remaining list"). The bound is checked whenever the IF executes, not
	// only when the branch is taken, so a malformed op list is rejected the
	// same way regardless of the data it runs against.
	errOperateIfRange = errors.New("ops: operate IF skip count out of range")
)
