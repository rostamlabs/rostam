// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

// EvictionsLive must count records displaced because the cache ran out of
// room, and must NOT count records that simply reached their TTL - that is the
// whole point of having it beside Evictions.
func TestEvictionsLiveCountsCapacityNotTTL(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.MaxMemoryPerShard = 8 << 20
	cfg.PageSize = 1 << 20
	cfg.AtCapPolicy = PolicyRingbufEvict
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Write well past the budget with no TTL, so nothing can expire and the
	// only way an entry leaves is capacity.
	val := make([]byte, 4<<10)
	for i := range 4000 {
		k := []byte("capacity-key-" + strings.Repeat("x", i%8) + string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('a'+(i/676)%26)))
		if err := c.Put(k, val, 0); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}

	st := c.Stats()
	if st.EvictionsLive == 0 {
		t.Fatalf("no live evictions after overfilling the budget: %+v", st)
	}
	if st.Expirations != 0 {
		t.Errorf("nothing had a TTL, so Expirations must be 0, got %d", st.Expirations)
	}
	if st.EvictionsLive > st.Evictions {
		t.Errorf("EvictionsLive (%d) cannot exceed Evictions (%d)", st.EvictionsLive, st.Evictions)
	}
}

func TestStatsWritePrometheus(t *testing.T) {
	var buf bytes.Buffer
	s := Stats{Gets: 7, Hits: 5, Evictions: 9, EvictionsLive: 4, BytesUsed: 123}
	if err := s.WritePrometheus(&buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{
		"# TYPE rostam_kv_gets_total counter\nrostam_kv_gets_total 7\n",
		"# TYPE rostam_kv_evictions_live_total counter\nrostam_kv_evictions_live_total 4\n",
		"# TYPE rostam_kv_bytes_used gauge\nrostam_kv_bytes_used 123\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("exposition missing:\n%q\ngot:\n%s", want, out)
		}
	}
}

// Rewriting ONE key fills the ring with superseded versions: every entry the
// eviction sweep walks is stale except the newest, so Evictions climbs while
// EvictionsLive stays near zero. This is what separates the two counters -
// without the cur == ref guard they would move together.
func TestEvictionsLiveExcludesSupersededEntries(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.MaxMemoryPerShard = 8 << 20
	cfg.PageSize = 1 << 20
	cfg.AtCapPolicy = PolicyRingbufEvict
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	key := []byte("one-hot-key")
	val := make([]byte, 64<<10)
	for range 400 {
		if err := c.Put(key, val, 0); err != nil {
			t.Fatal(err)
		}
	}

	st := c.Stats()
	if st.Evictions == 0 {
		t.Fatalf("expected the ring to evict while rewriting one key: %+v", st)
	}
	// Only the newest version is ever the live record, so essentially every
	// eviction here displaced a superseded entry.
	if st.EvictionsLive > st.Evictions/10 {
		t.Errorf("EvictionsLive=%d is too close to Evictions=%d; the cur == ref guard is not filtering superseded entries",
			st.EvictionsLive, st.Evictions)
	}
}

// An entry that expired but has not been swept yet is still index-current, so
// the cur == ref guard alone would count its eviction as capacity pressure.
// It is TTL turnover, and a correctly-sized cache is full of exactly this.
func TestEvictionsLiveExcludesExpiredEntries(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.MaxMemoryPerShard = 8 << 20
	cfg.PageSize = 1 << 20
	cfg.AtCapPolicy = PolicyRingbufEvict
	// Disable the background sweeper: this test needs expired-but-unswept pages
	// to still be there when eviction reaches them, which is the whole scenario.
	// Left at DefaultConfig's 1000ms the sweeper can reclaim them first and the
	// test flakes. Same convention as b3b_sweeper_test.go.
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Fill with entries that expire almost immediately, each under its own key
	// so none supersedes another.
	val := make([]byte, 4<<10)
	for i := range 1200 {
		if err := c.Put(fmt.Appendf(nil, "ttl-key-%06d", i), val, time.Millisecond); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	time.Sleep(30 * time.Millisecond) // everything written so far is now expired

	// Now drive eviction with fresh, long-lived entries. The pages being
	// reclaimed are full of expired-but-unswept records.
	before := c.Stats().EvictionsLive
	for i := range 1200 {
		if err := c.Put(fmt.Appendf(nil, "live-key-%06d", i), val, time.Hour); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}

	st := c.Stats()
	if st.Evictions == 0 {
		t.Fatalf("expected evictions while overfilling: %+v", st)
	}
	// The expired entries must not be booked as capacity loss. Some genuinely
	// live ones may be evicted late in the run, so allow a small margin rather
	// than demanding exactly zero.
	if grew := st.EvictionsLive - before; grew > st.Evictions/4 {
		t.Errorf("EvictionsLive grew by %d of %d evictions; expired-but-unswept entries are being counted as capacity loss",
			grew, st.Evictions)
	}
}

// StatsNoWalk must never trigger the reclaimable-bytes recomputation, because
// the cluster scrape calls it while holding the node's shard mutex. It agrees
// with Stats on every counter; only ReclaimableBytes may lag.
func TestStatsNoWalkMatchesStatsOnCounters(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 2
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	for i := range 200 {
		k := fmt.Appendf(nil, "k-%04d", i)
		if err := c.Put(k, []byte("v"), 0); err != nil {
			t.Fatal(err)
		}
		if _, err := c.Get(k); err != nil {
			t.Fatal(err)
		}
	}

	full, lean := c.Stats(), c.StatsNoWalk()
	full.ReclaimableBytes, lean.ReclaimableBytes = 0, 0
	if full != lean {
		t.Errorf("StatsNoWalk disagrees with Stats outside ReclaimableBytes:\n full %+v\n lean %+v", full, lean)
	}
	if lean.Gets == 0 || lean.Puts == 0 {
		t.Errorf("counters did not come through: %+v", lean)
	}
}
