// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"slices"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// allEvictionKnobs sets every eviction knob CacheConfig carries to a non-default
// value, with an interval no default could produce by accident.
func allEvictionKnobs() CacheConfig {
	return CacheConfig{
		RelocatingEviction:        true,
		RelocateReserveIntervalMs: 123,
		InPlaceSameSizeUpdate:     true,
		InPlaceSeqlockReads:       true,
		SieveVisitedBit:           true,
	}
}

func assertEvictionKnobsReached(t *testing.T, cc cache.Config) {
	t.Helper()
	if !cc.RelocatingEviction {
		t.Error("RelocatingEviction did not reach cache.Config")
	}
	if cc.RelocateReserveIntervalMs != 123 {
		t.Errorf("RelocateReserveIntervalMs = %d in cache.Config, want 123", cc.RelocateReserveIntervalMs)
	}
	if !cc.InPlaceSameSizeUpdate {
		t.Error("InPlaceSameSizeUpdate did not reach cache.Config")
	}
	if !cc.InPlaceSeqlockReads {
		t.Error("InPlaceSeqlockReads did not reach cache.Config")
	}
	if !cc.SieveVisitedBit {
		t.Error("SieveVisitedBit did not reach cache.Config")
	}
	if err := cc.Validate(); err != nil {
		t.Errorf("cache.Config with every eviction knob set does not validate: %v", err)
	}
}

// assertEvictionDefaults is the other half of the contract: a CacheConfig that says
// nothing about eviction must leave cache.DefaultConfig's choices exactly as they are.
func assertEvictionDefaults(t *testing.T, cc cache.Config) {
	t.Helper()
	def := cache.DefaultConfig()
	if cc.RelocatingEviction || cc.InPlaceSameSizeUpdate || cc.InPlaceSeqlockReads || cc.SieveVisitedBit {
		t.Errorf("a zero CacheConfig turned an eviction knob on: %+v", cc)
	}
	if cc.RelocateReserveIntervalMs != def.RelocateReserveIntervalMs {
		t.Errorf("zero RelocateReserveIntervalMs gave %d, want the library default %d",
			cc.RelocateReserveIntervalMs, def.RelocateReserveIntervalMs)
	}
}

// TestCacheConfigEvictionKnobsReachDirectCache is the regression this plumbing
// exists for: the knobs were cache.Config fields that no public config could set.
func TestCacheConfigEvictionKnobsReachDirectCache(t *testing.T) {
	cfg := DirectConfig{Ops: ops.NewRegistry(), Cache: allEvictionKnobs()}
	cfg.Cache.MaxMemoryBytes = 64 << 20
	cfg.Cache.NumShardsPerNode = 4
	cc, err := directCacheConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	assertEvictionKnobsReached(t, cc)

	cc, err = directCacheConfig(DirectConfig{Ops: ops.NewRegistry(), Cache: CacheConfig{NumShardsPerNode: 4, MaxMemoryBytes: 64 << 20}})
	if err != nil {
		t.Fatal(err)
	}
	assertEvictionDefaults(t, cc)
}

func TestCacheConfigEvictionKnobsReachEmbeddedCache(t *testing.T) {
	cfg := EmbeddedConfig{Ops: ops.NewRegistry(), Cache: allEvictionKnobs()}
	cfg.Cache.MaxMemoryBytes = 64 << 20
	cc, err := embeddedCacheConfig(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	assertEvictionKnobsReached(t, cc)

	cc, err = embeddedCacheConfig(EmbeddedConfig{Ops: ops.NewRegistry(), Cache: CacheConfig{MaxMemoryBytes: 64 << 20}}, 4)
	if err != nil {
		t.Fatal(err)
	}
	assertEvictionDefaults(t, cc)
}

// A negative interval follows TTLSweepIntervalMs: it turns the reserve ticker off,
// which cache.Config spells as zero.
func TestRelocateReserveIntervalNegativeDisables(t *testing.T) {
	cfg := DirectConfig{Ops: ops.NewRegistry(), Cache: CacheConfig{
		NumShardsPerNode: 4, MaxMemoryBytes: 64 << 20, RelocatingEviction: true, RelocateReserveIntervalMs: -1,
	}}
	cc, err := directCacheConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if cc.RelocateReserveIntervalMs != 0 {
		t.Errorf("RelocateReserveIntervalMs = %d, want 0 (reserve ticker off)", cc.RelocateReserveIntervalMs)
	}
}

func TestInertCacheKnobWarnings(t *testing.T) {
	const (
		reloc  = "RelocatingEviction"
		ivl    = "RelocateReserveIntervalMs"
		inpl   = "InPlaceSameSizeUpdate"
		seq    = "InPlaceSeqlockReads"
		sieve  = "SieveVisitedBit"
		heap   = deploySingleNodeHeap
		data   = deploySingleNodeDataDir
		clustr = deployCluster
	)
	cases := []struct {
		name string
		d    cacheDeployment
		c    CacheConfig
		want []string
	}{
		{"heap/nothing set", heap, CacheConfig{}, nil},
		{"heap/all five take effect", heap, allEvictionKnobs(), nil},
		{"heap/relocating", heap, CacheConfig{RelocatingEviction: true}, nil},
		{"heap/interval with relocating", heap, CacheConfig{RelocatingEviction: true, RelocateReserveIntervalMs: 50}, nil},
		{"heap/interval without relocating", heap, CacheConfig{RelocateReserveIntervalMs: 50}, []string{ivl}},
		{"heap/interval disabled without relocating", heap, CacheConfig{RelocateReserveIntervalMs: -1}, []string{ivl}},
		{"heap/in-place", heap, CacheConfig{InPlaceSameSizeUpdate: true}, nil},
		{"heap/seqlock with in-place", heap, CacheConfig{InPlaceSameSizeUpdate: true, InPlaceSeqlockReads: true}, nil},
		{"heap/seqlock without in-place", heap, CacheConfig{InPlaceSeqlockReads: true}, []string{seq}},
		{"heap/sieve with relocating", heap, CacheConfig{RelocatingEviction: true, SieveVisitedBit: true}, nil},
		{"heap/sieve without relocating", heap, CacheConfig{SieveVisitedBit: true}, []string{sieve}},

		{"data/nothing set", data, CacheConfig{}, nil},
		{"data/relocating", data, CacheConfig{RelocatingEviction: true}, nil},
		{"data/sieve with relocating", data, CacheConfig{RelocatingEviction: true, SieveVisitedBit: true}, nil},
		{"data/sieve without relocating", data, CacheConfig{SieveVisitedBit: true}, []string{sieve}},
		{"data/interval with relocating", data, CacheConfig{RelocatingEviction: true, RelocateReserveIntervalMs: 50}, []string{ivl}},
		{"data/in-place", data, CacheConfig{InPlaceSameSizeUpdate: true}, []string{inpl}},
		{"data/seqlock alone", data, CacheConfig{InPlaceSeqlockReads: true}, []string{seq}},
		{"data/in-place and seqlock", data, CacheConfig{InPlaceSameSizeUpdate: true, InPlaceSeqlockReads: true}, []string{inpl, seq}},
		{"data/all five", data, allEvictionKnobs(), []string{ivl, inpl, seq}},

		{"cluster/nothing set", clustr, CacheConfig{}, nil},
		{"cluster/all five", clustr, allEvictionKnobs(), []string{reloc, ivl, inpl, seq, sieve}},
		{"cluster/relocating", clustr, CacheConfig{RelocatingEviction: true}, []string{reloc}},
		{"cluster/interval", clustr, CacheConfig{RelocateReserveIntervalMs: 50}, []string{ivl}},
		{"cluster/in-place", clustr, CacheConfig{InPlaceSameSizeUpdate: true}, []string{inpl}},
		{"cluster/seqlock", clustr, CacheConfig{InPlaceSeqlockReads: true}, []string{seq}},
		{"cluster/sieve", clustr, CacheConfig{SieveVisitedBit: true}, []string{sieve}},
		// Unrelated cache settings are not eviction knobs and never warn.
		{"cluster/other settings", clustr, CacheConfig{Durable: true, TTLSweepIntervalMs: 5, MaxMemoryBytes: 1 << 30}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := inertCacheKnobs(tc.c, tc.d)
			var names []string
			for _, w := range got {
				names = append(names, w.Knob)
				if w.Flag == "" || w.Reason == "" {
					t.Errorf("warning for %s lacks a flag or a reason: %+v", w.Knob, w)
				}
			}
			if !slices.Equal(names, tc.want) {
				t.Errorf("inert knobs = %v, want %v", names, tc.want)
			}
		})
	}
}

// The deployment reason wins over a pairing reason: telling an operator on a cluster
// to turn on RelocatingEviction would send them after a fix that changes nothing.
func TestInertCacheKnobReasonNamesTheRealCause(t *testing.T) {
	for _, tc := range []struct {
		d    cacheDeployment
		c    CacheConfig
		want string
	}{
		{deployCluster, CacheConfig{SieveVisitedBit: true}, reasonClusterNoEviction},
		{deploySingleNodeHeap, CacheConfig{SieveVisitedBit: true}, reasonNeedsRelocating},
		{deploySingleNodeDataDir, CacheConfig{RelocateReserveIntervalMs: 50}, reasonReserveHeapOnly},
		{deploySingleNodeHeap, CacheConfig{RelocateReserveIntervalMs: 50}, reasonNeedsRelocating},
		{deploySingleNodeDataDir, CacheConfig{InPlaceSeqlockReads: true}, reasonSeqlockFileBacked},
		{deploySingleNodeHeap, CacheConfig{InPlaceSeqlockReads: true}, reasonSeqlockNeedsInPl},
	} {
		got := inertCacheKnobs(tc.c, tc.d)
		if len(got) != 1 || got[0].Reason != tc.want {
			t.Errorf("%v %+v: got %+v, want one warning with reason %q", tc.d, tc.c, got, tc.want)
		}
	}
}

// TestInPlaceWarningMatchesCacheBehaviour ties the matrix to the engine rather than
// to a reading of it: on the deployment the warning calls inert, same-size rewrites
// really do not happen in place, and where no warning fires they really do.
func TestInPlaceWarningMatchesCacheBehaviour(t *testing.T) {
	for _, tc := range []struct {
		name    string
		dataDir string
		d       cacheDeployment
	}{
		{"heap", "", deploySingleNodeHeap},
		{"data dir", t.TempDir(), deploySingleNodeDataDir},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DirectConfig{DataDir: tc.dataDir, Ops: ops.NewRegistry(), Cache: CacheConfig{
				NumShardsPerNode: 1, MaxMemoryBytes: 16 << 20, InPlaceSameSizeUpdate: true,
			}}
			if err := ops.RegisterBuiltins(cfg.Ops); err != nil {
				t.Fatal(err)
			}
			st, err := NewDirect(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = st.Close() }()
			c := st.(*directStore).cache
			for i := range 8 {
				if err := c.Put([]byte("k"), []byte{byte(i), 1, 2, 3}, 0); err != nil {
					t.Fatal(err)
				}
			}
			inPlace := c.Stats().InPlaceUpdates > 0
			warned := slices.ContainsFunc(inertCacheKnobs(cfg.Cache, tc.d), func(w inertCacheKnob) bool {
				return w.Knob == "InPlaceSameSizeUpdate"
			})
			if inPlace == warned {
				t.Errorf("in-place updates happened=%v but inert warning fired=%v; they must disagree", inPlace, warned)
			}
		})
	}
}

// A NewEmbedded node is a cluster for these purposes only when it has Peers: without
// them cluster.New takes its single-node path, gives each shard no Raft transport,
// and shard.New leaves ring-buffer eviction in place (shard/b2_reject_writes_test.go
// pins that half). rostam-server -cluster always passes at least its own peer.
func TestEmbeddedDeploymentFollowsPeers(t *testing.T) {
	if got := embeddedDeployment(EmbeddedConfig{}); got != deploySingleNodeDataDir {
		t.Errorf("no peers: deployment = %v, want %v", got, deploySingleNodeDataDir)
	}
	self := EmbeddedConfig{Peers: []Peer{{NodeID: "n1", RaftAddr: "127.0.0.1:7400"}}}
	if got := embeddedDeployment(self); got != deployCluster {
		t.Errorf("one self peer: deployment = %v, want %v", got, deployCluster)
	}
}
