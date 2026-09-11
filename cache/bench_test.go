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
	s, err := newShard(DefaultConfig(), "")
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
