// SPDX-License-Identifier: Apache-2.0

// Package cache benchmarks measure Rostam's cache on representative workloads.
// Cross-engine comparison benchmarks (vs freecache / ristretto / bigcache /
// fastcache / otter) live in the separate rostam-bench repo, so the engine
// module stays dependency-light (no competitor libraries in go.mod).
package cache

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

const (
	benchN     = 1 << 16 // 65 536 keys
	benchValSz = 256
)

func buildKeys(n int) [][]byte {
	r := rand.New(rand.NewSource(1)) //nolint:gosec // deterministic seed intentional for reproducible benchmark keys
	keys := make([][]byte, n)
	for i := range keys {
		k := make([]byte, 16)
		r.Read(k) //nolint:gosec // benchmark key generation; not security-sensitive
		keys[i] = k
	}
	return keys
}

func benchValue() []byte {
	v := make([]byte, benchValSz)
	for i := range v {
		v[i] = byte(i)
	}
	return v
}

// --- Rostam benchmarks ---

func BenchmarkRostamGetHit(b *testing.B) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	keys := buildKeys(benchN)
	val := benchValue()
	for _, k := range keys {
		_ = c.Put(k, val, 0)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		r := rand.New(rand.NewSource(time.Now().UnixNano())) //nolint:gosec // benchmark RNG; not used for security
		for pb.Next() {
			k := keys[r.Intn(len(keys))]
			_, _ = c.Get(k)
		}
	})
}

func BenchmarkRostamPut(b *testing.B) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	keys := buildKeys(benchN)
	val := benchValue()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		r := rand.New(rand.NewSource(time.Now().UnixNano())) //nolint:gosec // benchmark RNG; not used for security
		for pb.Next() {
			k := keys[r.Intn(len(keys))]
			_ = c.Put(k, val, 0)
		}
	})
}

// --- Single-shard low-level micro-benchmark to measure index-only overhead ---

func BenchmarkShardGetHit(b *testing.B) {
	s, err := newShard(DefaultConfig(), "", nil)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	keys := buildKeys(benchN)
	val := benchValue()
	for _, k := range keys {
		_ = s.Put(k, val, 0)
	}
	b.ResetTimer()
	i := 0
	for b.Loop() {
		_, _ = s.Get(keys[i%len(keys)])
		i++
	}
}

// --- Eyeball: how many pages get allocated for a realistic workload? ---

func BenchmarkPageGrowthReport(b *testing.B) {
	cfg := DefaultConfig()
	cfg.NumShards = 1
	c, _ := New(cfg)
	defer func() { _ = c.Close() }()
	val := benchValue()
	for i := range 200_000 {
		k := fmt.Appendf(nil, "k%07d", i)
		_ = c.Put(k, val, 0)
	}
	st := c.Stats()
	b.ReportMetric(float64(st.PagesAllocated), "pages")
	b.ReportMetric(float64(st.BytesUsed)/float64(st.BytesAllocated)*100, "%used")
}

// BenchmarkRostamGetWithExpiry and its Into twin measure the read an apply-path
// handler makes: the allocating form returns a fresh copy per hit, the Into form
// copies into a reused buffer. Both COPY - the difference is the allocation.
func BenchmarkRostamGetWithExpiry(b *testing.B) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	keys := buildKeys(benchN)
	val := benchValue()
	for _, k := range keys {
		_ = c.Put(k, val, 0)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := c.GetWithExpiry(keys[i%len(keys)]); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRostamGetWithExpiryInto(b *testing.B) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	keys := buildKeys(benchN)
	val := benchValue()
	for _, k := range keys {
		_ = c.Put(k, val, 0)
	}

	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _, err := c.GetWithExpiryInto(buf[:0], keys[i%len(keys)])
		if err != nil {
			b.Fatal(err)
		}
		buf = out
	}
}

// BenchmarkInPlaceCandidateRate reports what share of a write stream could have
// overwritten its predecessor where it lay instead of appending a fresh copy —
// Stats.InPlaceCandidates / Stats.Puts, in percent — across write shapes that
// bracket the interesting range. It is a REPORT, not a throughput measurement:
// the timed loop is the same append path in every case.
//
// The shapes:
//
//	fresh-keys       every write is a new key. Nothing to overwrite; the floor.
//	rewrite-samesize a bounded key set rewritten at a constant record size —
//	                 the shape the optimisation exists for.
//	rewrite-jitter   the same key set, but each rewrite picks a value length
//	                 from a small spread, so most rewrites reframe the entry.
//
// The shard is driven to capacity first so the rate reported is the steady-state
// one, eviction included: a rewrite whose key was evicted in the meantime is not
// a candidate, and that loss belongs in the figure rather than outside it.
func BenchmarkInPlaceCandidateRate(b *testing.B) {
	const (
		pages   = 8
		keySpan = 20_000
	)
	shapes := []struct {
		name string
		// value returns the bytes written on iteration i, and key the key.
		key   func(i int) []byte
		value func(r *rand.Rand, base []byte) []byte
	}{
		{
			name:  "fresh-keys",
			key:   func(i int) []byte { return fmt.Appendf(nil, "k%012d", i) },
			value: func(_ *rand.Rand, base []byte) []byte { return base },
		},
		{
			name:  "rewrite-samesize",
			key:   func(i int) []byte { return fmt.Appendf(nil, "k%012d", i%keySpan) },
			value: func(_ *rand.Rand, base []byte) []byte { return base },
		},
		{
			name: "rewrite-jitter",
			key:  func(i int) []byte { return fmt.Appendf(nil, "k%012d", i%keySpan) },
			value: func(r *rand.Rand, base []byte) []byte {
				return base[:len(base)-r.Intn(16)]
			},
		},
	}
	for _, sh := range shapes {
		b.Run(sh.name, func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.NumShards = 1
			cfg.PageSize = 1 << 20
			cfg.MaxMemoryPerShard = pages << 20
			cfg.TTLSweepIntervalMs = 0
			s, err := newShard(cfg, "", nil)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = s.Close() }()
			base := benchValue()
			r := rand.New(rand.NewSource(1)) //nolint:gosec // deterministic shape, not security
			// Warm to steady state: fill every page and start evicting, so the
			// measured window sees the rate a running shard actually has.
			for i := 0; s.evictions.Load() == 0 && i < 1<<22; i++ {
				_ = s.Put(sh.key(i), sh.value(r, base), 0)
			}
			before := s.snapshot()
			b.ReportAllocs()
			b.ResetTimer()
			i := 0
			for b.Loop() {
				_ = s.Put(sh.key(i), sh.value(r, base), 0)
				i++
			}
			b.StopTimer()
			after := s.snapshot()
			puts := after.Puts - before.Puts
			cand := after.InPlaceCandidates - before.InPlaceCandidates
			if puts == 0 {
				b.Fatal("no writes measured")
			}
			b.ReportMetric(float64(cand)/float64(puts)*100, "%candidates")
			b.ReportMetric(float64(after.BytesUsed)/float64(max(after.Entries, 1)), "B/livekey")
			b.ReportMetric(float64(after.Evictions-before.Evictions)/float64(puts), "evict/put")
		})
	}
}
