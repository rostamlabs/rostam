// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cache"
)

// cacheKnobFlags are the opt-in cache eviction knobs. All are off by default and
// stay off unless set. Most do nothing on most topologies, so the help text says
// where each one acts, and the store logs a startup warning naming any that is set
// where it cannot (rostam.NewDirect / rostam.NewEmbedded).
//
// Like every flag here, each also reads ROSTAM_<NAME> (applyEnvDefaults). They are
// deliberately NOT in the -config file: that file carries only knobs that have no
// flag, so a setting lives in exactly one place.
type cacheKnobFlags struct {
	fs                      *flag.FlagSet
	relocatingEviction      *bool
	relocateReserveInterval *time.Duration
	inPlaceSameSizeUpdate   *bool
	inPlaceSeqlockReads     *bool
	sieveVisitedBit         *bool
}

const relocateReserveIntervalFlag = "relocate-reserve-interval"

func registerCacheKnobFlags(fs *flag.FlagSet) *cacheKnobFlags {
	defaultReserve := time.Duration(cache.DefaultConfig().RelocateReserveIntervalMs) * time.Millisecond
	return &cacheKnobFlags{
		fs: fs,
		relocatingEviction: fs.Bool("relocating-eviction", false,
			"opt-in, single-node only (no effect with -cluster): rescue live records from evicted pages. "+
				"When a shard at capacity evicts a page, the records on it that are still live are copied forward instead of being dropped with the dead versions sharing the page. "+
				"It costs extra copying on the eviction path, which in-memory shards move to a background reserve (-relocate-reserve-interval). "+
				"Takes effect on a single-node server with or without -data. NO EFFECT with -cluster: cluster shards refuse writes at capacity instead of evicting, so there is nothing to relocate (a startup warning says so)"),
		relocateReserveInterval: fs.Duration(relocateReserveIntervalFlag, defaultReserve,
			"cadence of the -relocating-eviction background reserve; single-node without -data only. "+
				"Each shard tops up a small reserve of free pages on this interval, so a write at capacity finds room instead of evicting inline. "+
				"0 turns the reserve off, leaving relocation on the write path only. "+
				"Every shard runs its own ticker, so a shorter interval costs proportionally more on a node with many shards. "+
				"Takes effect only with -relocating-eviction on a single-node server WITHOUT -data; no effect with -data or -cluster"),
		inPlaceSameSizeUpdate: fs.Bool("in-place-same-size-update", false,
			"opt-in, single-node without -data only: rewrite same-length values in place. "+
				"A rewrite of a key whose new value has the same length overwrites the stored copy instead of appending a new one, so steady same-size rewrites stop filling the cache with dead versions. "+
				"Two trade-offs: a rewritten key no longer moves to the newest page, so on a cache run over capacity a frequently rewritten key is evicted on the same schedule as a rarely rewritten one; "+
				"and reads on the shard take its read lock (unless -in-place-seqlock-reads). "+
				"Takes effect only on a single-node server WITHOUT -data; no effect with -data (an interrupted overwrite of a file-backed page would lose other keys at recovery) or -cluster"),
		inPlaceSeqlockReads: fs.Bool("in-place-seqlock-reads", false,
			"opt-in performance trade whose read protocol is NOT covered by race detection. "+
				"Keeps reads lock-free under -in-place-same-size-update by reading page bytes a writer may be rewriting at that moment and discarding the read if a version counter moved. "+
				"That read is a deliberate data race: Go's race detector reports it and its concurrent tests are skipped under -race, so a green race-enabled test run says nothing about it. "+
				"Worth it only when reads far outnumber writes. "+
				"Takes effect only together with -in-place-same-size-update, on a single-node server WITHOUT -data"),
		sieveVisitedBit: fs.Bool("sieve-visited-bit", false,
			"opt-in, with -relocating-eviction only (no effect with -cluster): rescue records by recent use. "+
				"Relocating eviction then rescues only records read or rewritten since eviction last passed them, instead of whichever live records it meets first, so a stream of write-once keys cannot push the working set out. "+
				"It costs a compare on every index probe and an atomic store on the first read of a record after eviction clears its mark. "+
				"Takes effect only with -relocating-eviction on a single-node server, with or without -data; no effect with -cluster"),
	}
}

// apply copies the knobs onto cc. It must run after flag parsing and after
// applyEnvDefaults, and lookup must be the environment those used.
//
// The reserve interval is passed through ONLY when the operator set it (on the
// command line or through its variable). Left alone, cc keeps zero — the library
// default — so an unset flag is never mistaken for a request and never draws an
// inert-knob warning on a topology where the reserve does not run.
func (k *cacheKnobFlags) apply(cc *rostam.CacheConfig, lookup func(string) (string, bool)) error {
	cc.RelocatingEviction = *k.relocatingEviction
	cc.InPlaceSameSizeUpdate = *k.inPlaceSameSizeUpdate
	cc.InPlaceSeqlockReads = *k.inPlaceSeqlockReads
	cc.SieveVisitedBit = *k.sieveVisitedBit

	explicit := false
	k.fs.Visit(func(f *flag.Flag) { explicit = explicit || f.Name == relocateReserveIntervalFlag })
	if _, ok := lookup(envNameFor(relocateReserveIntervalFlag)); ok {
		explicit = true
	}
	if !explicit {
		return nil
	}
	switch d := *k.relocateReserveInterval; {
	case d < 0:
		return errors.New("-relocate-reserve-interval must not be negative (use 0 to turn the reserve off)")
	case d == 0:
		cc.RelocateReserveIntervalMs = -1
	case d.Milliseconds() > int64(math.MaxInt):
		return fmt.Errorf("-relocate-reserve-interval %s is too large for this platform", d)
	default:
		cc.RelocateReserveIntervalMs = max(int(d.Milliseconds()), 1)
	}
	return nil
}
