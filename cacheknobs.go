// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"log/slog"

	"github.com/rostamlabs/rostam/cache"
)

// applyEvictionKnobs copies CacheConfig's opt-in eviction knobs onto cc. A zero
// CacheConfig leaves cache.DefaultConfig's choices untouched: the four switches off,
// and a reserve interval that is non-zero but inert without RelocatingEviction. The
// interval follows TTLSweepIntervalMs (0 keeps the default, negative turns the
// ticker off, positive sets it).
func applyEvictionKnobs(cc *cache.Config, c CacheConfig) {
	cc.RelocatingEviction = c.RelocatingEviction
	switch {
	case c.RelocateReserveIntervalMs < 0:
		cc.RelocateReserveIntervalMs = 0
	case c.RelocateReserveIntervalMs > 0:
		cc.RelocateReserveIntervalMs = c.RelocateReserveIntervalMs
	}
	cc.InPlaceSameSizeUpdate = c.InPlaceSameSizeUpdate
	cc.InPlaceSeqlockReads = c.InPlaceSeqlockReads
	cc.SieveVisitedBit = c.SieveVisitedBit
}

// cacheDeployment is the storage shape a store's cache shards end up in, which is
// what decides whether an eviction knob can act at all.
type cacheDeployment uint8

const (
	// deploySingleNodeHeap: NewDirect with no DataDir — in-memory shards that evict
	// at capacity.
	deploySingleNodeHeap cacheDeployment = iota
	// deploySingleNodeDataDir: NewDirect with a DataDir, or NewEmbedded with no
	// Peers — file-backed shards that evict at capacity.
	deploySingleNodeDataDir
	// deployCluster: NewEmbedded with Peers (every rostam-server -cluster node) —
	// file-backed, replicated shards that REFUSE writes at capacity and never evict.
	deployCluster
)

func (d cacheDeployment) String() string {
	switch d {
	case deploySingleNodeHeap:
		return "single-node in-memory"
	case deploySingleNodeDataDir:
		return "single-node with a data directory"
	default:
		return "cluster"
	}
}

// inertCacheKnob is one eviction knob that was set but cannot act on the store
// being built.
type inertCacheKnob struct {
	Knob   string // the CacheConfig field
	Flag   string // the rostam-server flag that sets it
	Reason string
}

// Reasons are phrased for an operator reading a startup log, so they name the
// mechanism rather than the code path. Each is derived from the gate that makes the
// knob a no-op in cache/: AtCapPolicy (replication forces PolicyRejectWrites in
// shard.New), the heap-only guard in inPlaceEligible, the heap-only
// reserveRelocationEligible, and the RelocatingEviction term in sieveVisited.
const (
	reasonClusterNoEviction = "cluster shards are replicated, and replication makes every shard refuse writes at capacity instead of evicting (eviction would let replicas diverge), so no eviction ever runs"
	reasonInPlaceHeapOnly   = "in-place updates apply only to in-memory shards that evict at capacity; on file-backed shards a rewrite always appends, because an interrupted overwrite would lose the rest of its page at recovery"
	reasonReserveHeapOnly   = "the background free-page reserve runs only on in-memory shards; on file-backed shards relocating eviction runs on the write path alone"
	reasonSeqlockFileBacked = "it changes only how reads cope with in-place updates, and in-place updates never take effect on file-backed shards"
	reasonSeqlockNeedsInPl  = "in-place same-size updates are off, and this knob only changes how reads cope with them"
	reasonNeedsRelocating   = "relocating eviction is off, and this knob only tunes relocating eviction"
)

// inertCacheKnobs lists the eviction knobs set in c that have no effect on a store
// of deployment d, in CacheConfig field order. It judges only what the caller SET:
// a knob left at its zero value is never reported, even where it would be inert.
//
// The order of checks per knob matters only for which reason is given: the
// deployment reason wins over a pairing reason, because fixing the pairing would
// not make the knob act.
func inertCacheKnobs(c CacheConfig, d cacheDeployment) []inertCacheKnob {
	var out []inertCacheKnob
	add := func(knob, flag, reason string) {
		out = append(out, inertCacheKnob{Knob: knob, Flag: flag, Reason: reason})
	}
	cluster := d == deployCluster
	fileBacked := d != deploySingleNodeHeap

	if c.RelocatingEviction && cluster {
		add("RelocatingEviction", "-relocating-eviction", reasonClusterNoEviction)
	}
	if c.RelocateReserveIntervalMs != 0 {
		switch {
		case cluster:
			add("RelocateReserveIntervalMs", "-relocate-reserve-interval", reasonClusterNoEviction)
		case fileBacked:
			add("RelocateReserveIntervalMs", "-relocate-reserve-interval", reasonReserveHeapOnly)
		case !c.RelocatingEviction:
			add("RelocateReserveIntervalMs", "-relocate-reserve-interval", reasonNeedsRelocating)
		}
	}
	if c.InPlaceSameSizeUpdate && fileBacked {
		add("InPlaceSameSizeUpdate", "-in-place-same-size-update", reasonInPlaceHeapOnly)
	}
	if c.InPlaceSeqlockReads {
		switch {
		case fileBacked:
			add("InPlaceSeqlockReads", "-in-place-seqlock-reads", reasonSeqlockFileBacked)
		case !c.InPlaceSameSizeUpdate:
			add("InPlaceSeqlockReads", "-in-place-seqlock-reads", reasonSeqlockNeedsInPl)
		}
	}
	if c.SieveVisitedBit {
		switch {
		case cluster:
			add("SieveVisitedBit", "-sieve-visited-bit", reasonClusterNoEviction)
		case !c.RelocatingEviction:
			add("SieveVisitedBit", "-sieve-visited-bit", reasonNeedsRelocating)
		}
	}
	return out
}

// warnInertCacheKnobs logs one startup warning per knob inertCacheKnobs reports.
// A warning rather than an error on purpose: an over-specified config — the same
// flags or environment shared by a cluster and a single-node box — is not a broken
// one, and refusing to start would turn a harmless no-op into an outage.
func warnInertCacheKnobs(c CacheConfig, d cacheDeployment) {
	for _, k := range inertCacheKnobs(c, d) {
		slog.Warn("cache option is set but has no effect on this deployment",
			"component", "cache", "option", k.Knob, "flag", k.Flag,
			"deployment", d.String(), "reason", k.Reason)
	}
}
