// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"fmt"
	"math"
	"math/bits"

	"github.com/rostamlabs/rostam/cache"
)

// Cache memory is bounded by NumShards * MaxMemoryPerShard. Before this file
// existed, every construction path took cache.DefaultConfig()'s per-shard cap
// (256 MiB) verbatim and only NumShards was settable, so the TOTAL bound was a
// side effect of a concurrency knob: the single-node default (256 shards) capped
// at 64 GiB and -shards 1024 silently meant 256 GiB. Since Put is append-only
// (the lock-free read path freezes retired pages — see cache/shard.go
// retirePageLocked), a write-heavy server climbs toward that cap even when the
// live set is tiny, so on any host below the cap the process dies before the
// ring-buffer eviction it relies on ever engages.
//
// These helpers turn the bound into a TOTAL budget the caller states directly,
// and derive the per-shard geometry from it.

const (
	// minPageSize / maxPageSize mirror cache.Config.Validate's bounds. Kept
	// here so geometry errors name a budget the caller can act on rather than
	// surfacing as an opaque validation failure from cache.New.
	minPageSize = 1 << 20 // 1 MiB
	maxPageSize = 1 << 30 // 1 GiB

	// targetPagesPerShard is the pages-per-shard ratio the geometry aims for.
	// 16 reproduces cache.DefaultConfig() exactly (256 MiB / 16 MiB) so a
	// budget equal to the historical 64 GiB derives byte-identical settings.
	targetPagesPerShard = 16

	// minPagesPerShard is the floor. A shard with one page has no ring
	// headroom: retiring it drops the shard's whole contents at once. Two is
	// the minimum that lets the ring wrap rather than flush.
	minPagesPerShard = 2

	// fallbackBudget is used when total system memory cannot be detected
	// (non-Linux, or a failed syscall). Deliberately modest: a wrong guess that
	// is too small degrades hit rate, one that is too large kills the host.
	fallbackBudget = 1 << 30 // 1 GiB

	// budgetFractionPercent is the share of detected system memory used when the
	// caller states no budget.
	//
	// NOTE the budget bounds CACHE PAGES, not process RSS. Cache pages are the
	// live heap, and Go lets the heap reach (1 + GOGC/100) x live before
	// collecting, so:
	//
	//	RSS ~= budget * (1 + GOGC/100) + ~40 MB
	//
	// Measured against a 512 MiB budget (RSS flat over 56M writes):
	//
	//	GOGC=40  -> 754 MB (1.47x)    GOGC=200 -> 1586 MB (3.10x)
	//	GOGC=100 -> 1057 MB (2.07x)   GOGC=400 -> 2624 MB (5.13x)
	//
	// The multiplier is Go's GC headroom, NOT engine overhead — the ~40 MB
	// residual is the only structural cost. At Go's default GOGC=100 this
	// fraction therefore lands near 50% of the host in RSS, which is the reason
	// it is not set higher. Deployments that want RSS bounded independently of
	// GOGC should set GOMEMLIMIT.
	budgetFractionPercent = 25
)

// defaultCacheBudget returns the total cache budget to use when the caller
// states none: a fraction of detected system memory, or fallbackBudget when the
// host cannot be probed. The result is always >= fallbackBudget's floor logic
// only insofar as a detected tiny host still yields a valid geometry; callers
// pass it through cacheGeometry, which reports an unusable budget as an error
// rather than silently producing a broken cache.
func defaultCacheBudget() int64 {
	total := systemMemoryBytes()
	if total <= 0 {
		return fallbackBudget
	}
	return total * budgetFractionPercent / 100
}

// floorPow2 returns the largest power of two <= n, for n >= 1. Used to keep the
// derived page size aligned and predictable; callers clamp to the validator's
// [minPageSize, maxPageSize] range before calling, and both bounds are already
// powers of two, so rounding down cannot fall below the floor.
func floorPow2(n int64) int64 {
	if n < 1 {
		return 1
	}
	return int64(1) << (bits.Len64(uint64(n)) - 1)
}

// cacheGeometry derives (MaxMemoryPerShard, PageSize) from a TOTAL byte budget
// spread over numShards. It aims for targetPagesPerShard pages per shard and
// clamps PageSize to cache.Config.Validate's [1 MiB, 1 GiB] range.
//
// A budget equal to the historical default (256 shards * 256 MiB = 64 GiB)
// derives exactly cache.DefaultConfig()'s geometry, so stating that budget is a
// no-op. Smaller budgets hit the 1 MiB page floor and therefore get fewer, not
// smaller, pages: at 512 MiB over 256 shards a shard holds 2 MiB = 2 pages, so
// eviction retires half a shard at a time. That coarseness is inherent to the
// page floor at small budgets, not a tuning choice.
func cacheGeometry(totalBytes int64, numShards int) (maxMemPerShard, pageSize int, err error) {
	if numShards <= 0 {
		return 0, 0, fmt.Errorf("rostam: cache geometry: NumShards=%d must be > 0", numShards)
	}
	if totalBytes <= 0 {
		return 0, 0, fmt.Errorf("rostam: cache geometry: budget=%d must be > 0", totalBytes)
	}

	perShard := totalBytes / int64(numShards)

	// Round the page size DOWN to a power of two. Dividing raw host RAM by
	// targetPagesPerShard otherwise yields ragged sizes (a 25%-of-31.7 GiB
	// budget lands on 2029312-byte pages), so the derived default would differ
	// in shape from every explicitly-stated budget — all of which already fall
	// on powers of two. Rounding down never exceeds the budget; it only trades
	// a sliver of the cap for a predictable, aligned geometry.
	ps := floorPow2(min(max(perShard/targetPagesPerShard, minPageSize), maxPageSize))

	if perShard < int64(minPagesPerShard)*ps {
		return 0, 0, fmt.Errorf(
			"rostam: cache budget %d bytes over %d shards leaves %d bytes per shard, "+
				"below the %d-byte minimum (%d pages x %d-byte minimum page): "+
				"raise the budget to at least %d bytes or lower NumShards",
			totalBytes, numShards, perShard, int64(minPagesPerShard)*ps,
			minPagesPerShard, ps, int64(minPagesPerShard)*ps*int64(numShards))
	}

	// perShard is an int64 but the cache config field is an int, so on a 32-bit
	// build a budget whose per-shard share exceeds MaxInt would wrap silently —
	// 32 GiB truncates to 0, which Validate then rejects with an error naming the
	// wrong cause. Refuse it here, where the real reason can be stated.
	if perShard > math.MaxInt {
		return 0, 0, fmt.Errorf(
			"rostam: cache budget %d bytes over %d shards leaves %d bytes per shard, "+
				"which exceeds the %d-byte maximum addressable on this platform: "+
				"lower the budget or raise NumShards",
			totalBytes, numShards, perShard, int64(math.MaxInt))
	}

	return int(perShard), int(ps), nil
}

// applyCacheBudget sets cc.MaxMemoryPerShard and cc.PageSize so the node's
// TOTAL cache memory stays within budget. A budget of zero means "derive from
// the host" (defaultCacheBudget).
//
// shardsTotal is the number of cache shards the budget is divided across on
// this node, which is NOT always cc.NumShards:
//   - Direct: one cache with cc.NumShards shards => shardsTotal = cc.NumShards.
//   - Embedded: one cache per Raft shard, each pinned to cc.NumShards = 1, so
//     the node holds NumShards of them => shardsTotal = the Raft shard count.
//
// Passing cc.NumShards blindly would under-count the embedded case by 64x and
// hand every Raft shard the whole budget, so the caller states it explicitly.
func applyCacheBudget(cc *cache.Config, budget int64, shardsTotal int) error {
	if budget <= 0 {
		budget = defaultCacheBudget()
	}
	perShard, pageSize, err := cacheGeometry(budget, shardsTotal)
	if err != nil {
		return err
	}
	cc.MaxMemoryPerShard = perShard
	cc.PageSize = pageSize
	return nil
}
