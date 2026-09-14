// SPDX-License-Identifier: Apache-2.0

package rostam

import "github.com/rostamlabs/rostam/cache"

// applyRingbufReclaim copies the ringbuf reclamation knobs from the public
// CacheConfig onto the cache config a store is about to be built with. Both
// constructors call it so the two cannot drift: a field wired in one and
// forgotten in the other compiles cleanly and silently runs with the feature
// off, which is indistinguishable from an operator not asking for it.
//
// Every knob here defaults off and the cache gates each one again on the shard
// actually being ringbuf, so this only ever carries an explicit opt-in through.
func applyRingbufReclaim(cc *cache.Config, cfg CacheConfig) {
	cc.RelocatingEviction = cfg.RelocatingEviction
	cc.SieveVisitedBit = cfg.SieveVisitedBit
	cc.InPlaceSameSizeUpdate = cfg.InPlaceSameSizeUpdate
	cc.InPlaceSeqlockReads = cfg.InPlaceSeqlockReads

	// Matches the TTLSweepIntervalMs convention: 0 keeps the cache default,
	// negative means "no ticker", positive is the interval.
	switch {
	case cfg.RelocateReserveIntervalMs < 0:
		cc.RelocateReserveIntervalMs = 0
	case cfg.RelocateReserveIntervalMs > 0:
		cc.RelocateReserveIntervalMs = cfg.RelocateReserveIntervalMs
	}
}
