// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
)

// The historical geometry: 256 shards x 256 MiB per shard = 64 GiB total, with
// 16 MiB pages. Stating that budget must reproduce it byte-for-byte, otherwise
// this change silently re-tunes every existing deployment.
func TestCacheGeometryReproducesHistoricalDefault(t *testing.T) {
	const historicalTotal = int64(256) * 256 << 20 // 64 GiB

	perShard, pageSize, err := cacheGeometry(historicalTotal, 256)
	if err != nil {
		t.Fatalf("cacheGeometry: %v", err)
	}

	def := cache.DefaultConfig()
	if perShard != def.MaxMemoryPerShard {
		t.Errorf("MaxMemoryPerShard = %d, want %d (cache.DefaultConfig)", perShard, def.MaxMemoryPerShard)
	}
	if pageSize != def.PageSize {
		t.Errorf("PageSize = %d, want %d (cache.DefaultConfig)", pageSize, def.PageSize)
	}
}

func TestCacheGeometry(t *testing.T) {
	const MiB = 1 << 20
	const GiB = 1 << 30

	tests := []struct {
		name         string
		total        int64
		shards       int
		wantPerShard int64
		wantPageSize int
		wantPages    int
	}{
		{
			// The bench-suite budget. The 1 MiB page floor binds here, so the
			// shard gets 2 pages rather than 16 — half a shard per eviction.
			name:  "512MiB/256shards hits the page floor",
			total: 512 * MiB, shards: 256,
			wantPerShard: 2 * MiB, wantPageSize: 1 * MiB, wantPages: 2,
		},
		{
			// Roughly this box's derived default (25% of 32 GiB).
			name:  "8GiB/256shards keeps the 16-page ratio",
			total: 8 * GiB, shards: 256,
			wantPerShard: 32 * MiB, wantPageSize: 2 * MiB, wantPages: 16,
		},
		{
			name:  "64GiB/256shards == historical default",
			total: 64 * GiB, shards: 256,
			wantPerShard: 256 * MiB, wantPageSize: 16 * MiB, wantPages: 16,
		},
		{
			// Fewer shards => more memory each => same ratio, bigger pages.
			name:  "8GiB/64shards",
			total: 8 * GiB, shards: 64,
			wantPerShard: 128 * MiB, wantPageSize: 8 * MiB, wantPages: 16,
		},
		{
			// The page ceiling binds: perShard/16 would exceed 1 GiB.
			name:  "512GiB/16shards hits the page ceiling",
			total: 512 * GiB, shards: 16,
			wantPerShard: 32 * GiB, wantPageSize: 1 * GiB, wantPages: 32,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			perShard, pageSize, err := cacheGeometry(tc.total, tc.shards)
			// A per-shard share that does not fit an int is unrepresentable in the
			// cache config, so cacheGeometry owes a refusal rather than a truncation.
			// Never taken on a 64-bit build, where every case here fits.
			if tc.wantPerShard > int64(math.MaxInt) {
				if err == nil {
					t.Fatalf("cacheGeometry(%d, %d) = %d bytes per shard, want a refusal: %d does not fit an int on this platform",
						tc.total, tc.shards, perShard, tc.wantPerShard)
				}
				// Any non-nil error would pass a bare != nil check, including the
				// unrelated headroom rejection — and a regression that truncated
				// perShard BEFORE the conversion would produce exactly that. Pin
				// the identity, so only this rejection satisfies the case.
				if !errors.Is(err, ErrBudgetExceedsPlatformInt) {
					t.Fatalf("cacheGeometry(%d, %d) error = %v, want %v",
						tc.total, tc.shards, err, ErrBudgetExceedsPlatformInt)
				}
				// And pin that it names the real cause: the share that did not fit
				// and the limit it exceeded. Without these the message could regress
				// to the misleading downstream wording this guard exists to replace.
				for _, want := range []string{
					strconv.FormatInt(tc.total, 10),
					strconv.Itoa(tc.shards),
					strconv.FormatInt(tc.wantPerShard, 10),
					strconv.FormatInt(int64(math.MaxInt), 10),
				} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not name %s", err, want)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("cacheGeometry(%d, %d): %v", tc.total, tc.shards, err)
			}
			if int64(perShard) != tc.wantPerShard {
				t.Errorf("perShard = %d, want %d", perShard, tc.wantPerShard)
			}
			if pageSize != tc.wantPageSize {
				t.Errorf("pageSize = %d, want %d", pageSize, tc.wantPageSize)
			}
			if got := perShard / pageSize; got != tc.wantPages {
				t.Errorf("pages/shard = %d, want %d", got, tc.wantPages)
			}

			// Whatever we derive must satisfy the cache's own validator —
			// that is the contract cache.New enforces.
			cc := cache.DefaultConfig()
			cc.NumShards = tc.shards
			cc.MaxMemoryPerShard = perShard
			cc.PageSize = pageSize
			if err := cc.Validate(); err != nil {
				t.Errorf("derived geometry fails cache.Config.Validate: %v", err)
			}

			// The budget must be honoured, never silently exceeded.
			if got := int64(perShard) * int64(tc.shards); got > tc.total {
				t.Errorf("derived total = %d exceeds budget %d", got, tc.total)
			}
		})
	}
}

// A budget too small for the shard count must fail loudly with an actionable
// message rather than produce a 1-page shard (which drops everything on each
// retire) or an opaque cache.New validation error.
func TestCacheGeometryRejectsUnusableBudget(t *testing.T) {
	const MiB = 1 << 20
	tests := []struct {
		name   string
		total  int64
		shards int
	}{
		{"1MiB per shard leaves no ring headroom", 256 * MiB, 256},
		{"below one page per shard", 64 * MiB, 256},
		{"zero budget", 0, 256},
		{"zero shards", 1 << 30, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := cacheGeometry(tc.total, tc.shards); err == nil {
				t.Fatalf("cacheGeometry(%d, %d) = nil error, want a rejection", tc.total, tc.shards)
			}
		})
	}
}

// A zero budget means "derive from the host" and must yield a usable geometry.
func TestApplyCacheBudgetZeroDerivesFromHost(t *testing.T) {
	cc := cache.DefaultConfig()
	if err := applyCacheBudget(&cc, 0, cc.NumShards); err != nil {
		t.Fatalf("applyCacheBudget(0): %v", err)
	}
	if err := cc.Validate(); err != nil {
		t.Fatalf("derived config invalid: %v", err)
	}
	if cc.MaxMemoryPerShard/cc.PageSize < minPagesPerShard {
		t.Errorf("derived %d pages/shard, want >= %d",
			cc.MaxMemoryPerShard/cc.PageSize, minPagesPerShard)
	}

	// The whole point: the derived default must not exceed the historical
	// 64 GiB cap on any host we would actually run on.
	total := int64(cc.MaxMemoryPerShard) * int64(cc.NumShards)
	if budget := defaultCacheBudget(); total > budget {
		t.Errorf("derived total %d exceeds derived budget %d", total, budget)
	}
	t.Logf("host-derived budget: %.2f GiB total (%d shards x %d bytes, %d-byte pages)",
		float64(total)/(1<<30), cc.NumShards, cc.MaxMemoryPerShard, cc.PageSize)
}

// The derived page size must always be an aligned power of two, whatever
// ragged budget the host produces (25% of RAM is rarely a round number).
func TestCacheGeometryPageSizeIsPowerOfTwo(t *testing.T) {
	// Deliberately ragged budgets, incl. this box's ~25%-of-31.7 GiB shape
	// which previously derived 2029312-byte pages.
	for _, total := range []int64{
		8312064000, 7_000_000_000, 33248256000 / 4, 12345678901, 1 << 30, 999 << 20,
	} {
		for _, shards := range []int{1, 16, 64, 256} {
			perShard, pageSize, err := cacheGeometry(total, shards)
			if err != nil {
				continue // budget genuinely too small for this shard count
			}
			if pageSize&(pageSize-1) != 0 {
				t.Errorf("cacheGeometry(%d, %d): pageSize=%d is not a power of two", total, shards, pageSize)
			}
			if pageSize < minPageSize || pageSize > maxPageSize {
				t.Errorf("cacheGeometry(%d, %d): pageSize=%d outside [%d,%d]",
					total, shards, pageSize, minPageSize, maxPageSize)
			}
			if int64(perShard)*int64(shards) > total {
				t.Errorf("cacheGeometry(%d, %d): derived total exceeds budget", total, shards)
			}
		}
	}
}

func TestFloorPow2(t *testing.T) {
	tests := map[int64]int64{
		1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 1023: 512, 1024: 1024, 1025: 1024,
		2029312: 1 << 20, 1 << 30: 1 << 30, (1 << 30) + 1: 1 << 30,
	}
	for in, want := range tests {
		if got := floorPow2(in); got != want {
			t.Errorf("floorPow2(%d) = %d, want %d", in, got, want)
		}
	}
}

func TestDefaultCacheBudgetIsSane(t *testing.T) {
	got := defaultCacheBudget()
	if got <= 0 {
		t.Fatalf("defaultCacheBudget() = %d, want > 0", got)
	}
	if sys := systemMemoryBytes(); sys > 0 && got >= sys {
		t.Errorf("defaultCacheBudget() = %d, must be below total system memory %d", got, sys)
	}
	t.Logf("system memory=%d bytes, default budget=%.2f GiB",
		systemMemoryBytes(), float64(got)/(1<<30))
}
