// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// BenchmarkIndexMemory reports what one definition's postings cost in heap:
// bytes per indexed key, and how much of that a distinct value adds.
//
// WHY A BENCHMARK AND NOT A TEST. There is no threshold here worth failing on —
// the number moves with the Go map implementation, the key length and the
// value distribution, and pinning it would break on a toolchain bump for no
// correctness reason. What it is for is the docs: the cost model on
// docs/kv/querying-records.md quotes figures an operator sizes a definition
// with, and a figure nobody can reproduce is a figure nobody should trust.
// Running this is how those numbers are re-derived.
//
//	go test ./ops/kvindex/ -run xxx -bench IndexMemory -benchtime 1x
//
//	BenchmarkIndexMemory/100k_keys_1k_distinct-20     1  ...  147.4 B/key
//	BenchmarkIndexMemory/100k_keys_100k_distinct-20   1  ...  450.8 B/key
//
// The two rows are the whole model: the first is the per-key cost at
// negligible cardinality, and their difference re-spread over the extra
// distinct values — (450.8-147.4) x 100 000 / 99 000, about 306 bytes — is
// what one more posting SET costs, which is a small Go map plus its scalar
// key rather than anything per key. Both are measured with the key length
// below, and the key bytes are stored once per posting, so a longer key moves
// the first number by roughly its own length. The second number is a ceiling
// on cardinality's cost, not a rate: at one key per value every set is a
// map holding one entry, which is the worst ratio there is.
//
// It measures with ReadMemStats around a GC rather than with the allocation
// counters testing itself reports, because the question is RETAINED heap — what
// the index still holds afterwards — not what was allocated on the way there.
func BenchmarkIndexMemory(b *testing.B) {
	for _, tc := range []struct {
		name           string
		keys, distinct int
	}{
		{"100k_keys_1k_distinct", 100_000, 1_000},
		{"100k_keys_100k_distinct", 100_000, 100_000},
	} {
		b.Run(tc.name, func(b *testing.B) {
			var perKey float64
			for i := 0; i < b.N; i++ {
				perKey = indexHeapPerKey(b, tc.keys, tc.distinct)
			}
			b.ReportMetric(perKey, "B/key")
		})
	}
}

// benchKeyFor is the key shape the figures are measured with: 14 bytes,
// "user:" plus nine digits. Stated explicitly because the per-key number
// includes the key's own bytes.
func benchKeyFor(i int) []byte { return fmt.Appendf(nil, "user:%09d", i) }

// indexHeapPerKey builds one definition's postings over `keys` keys spread
// across `distinct` values and returns the retained heap per key.
//
// The Set is kept alive across the second reading (runtime.KeepAlive), which is
// the whole measurement: without it the compiler is free to consider it dead
// before the GC that precedes the reading, and the answer would be zero.
func indexHeapPerKey(b *testing.B, keys, distinct int) float64 {
	b.Helper()
	d, err := DefFrom(wire.KVIndexDef{
		Name: "m", PayloadPath: "rc", Kind: wire.KVIndexKindScalar, Enabled: true,
	}, 1)
	if err != nil {
		b.Fatal(err)
	}
	s := New(0)
	s.Install([]Def{d})

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	for i := 0; i < keys; i++ {
		s.Reindex(benchKeyFor(i), intRec("rc", int64(i%distinct)))
	}

	runtime.GC()
	runtime.ReadMemStats(&after)
	// Reported so a run that drifted from its intended shape is visible rather
	// than silently averaged into the figure.
	if got, gotDistinct := s.Stats("m"); got != keys || gotDistinct != distinct {
		b.Fatalf("built %d keys over %d distinct values, want %d over %d", got, gotDistinct, keys, distinct)
	}
	heap := float64(after.HeapAlloc) - float64(before.HeapAlloc)
	runtime.KeepAlive(s)
	return heap / float64(keys)
}
