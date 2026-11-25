// SPDX-License-Identifier: Apache-2.0

//go:build !unix

package main

import "context"

// lockSession degrades to no locking where flock is unavailable: concurrent
// hooks may then occasionally inject a memory twice, which beats failing.
func lockSession(context.Context, string, string) (func(), bool) {
	return func() {}, true
}
