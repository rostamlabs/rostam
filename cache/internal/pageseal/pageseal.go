// SPDX-License-Identifier: Apache-2.0

// Package pageseal provides a MONOTONE mutable→sealed latch.
//
// A latch is minted mutable by New and may be sealed exactly once, by Seal.
// Nothing in this package can return a sealed latch to mutable, and a latch's
// state is not settable from outside it — which is the whole reason the type
// lives in its own package rather than as a bare bool on the caller's struct.
// Monotonicity is the property the caller depends on: a reader that observes
// "sealed" may conclude the answer can never change back, with no lock and no
// further coordination.
//
// Re-mutating something a latch guards therefore costs a NEW latch, which in
// practice means constructing a new owner object for it — deliberate and
// reviewable, rather than one field assignment that quietly breaks every
// conclusion a reader drew from an earlier observation.
//
// A nil *Latch is a valid, permanently-sealed latch. That is the state of
// anything that was never admitted to the mutable set, so callers need no
// separate "not a member" flag and no nil checks of their own.
package pageseal

import "sync/atomic"

// Latch is a one-way mutable→sealed flag. The zero value is SEALED; use New to
// mint a mutable one. Safe for concurrent use; must not be copied after first
// use (it holds an atomic).
type Latch struct {
	mutable atomic.Bool
}

// New returns a latch in the MUTABLE state. It is the only way to obtain one:
// there is no exported way to unseal, so every mutable latch is demonstrably
// fresh.
func New() *Latch {
	l := &Latch{}
	l.mutable.Store(true)
	return l
}

// Mutable reports whether the latch is still mutable. A nil latch is sealed.
func (l *Latch) Mutable() bool {
	return l != nil && l.mutable.Load()
}

// Seal moves the latch to sealed, permanently. It returns true only for the call
// that actually performed the transition, so callers may count transitions
// without double-counting a repeated or racing seal. Sealing a nil (already
// sealed) latch reports false.
func (l *Latch) Seal() bool {
	if l == nil {
		return false
	}
	return l.mutable.CompareAndSwap(true, false)
}
