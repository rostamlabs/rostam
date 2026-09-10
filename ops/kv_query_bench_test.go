// SPDX-License-Identifier: Apache-2.0

package ops

// Benchmarks for the two shapes of a kv_query page over the SAME 100 000-key
// shard: the indexed one, whose cost is the size of the answer, and the scanned
// one, whose cost is the size of the keyspace. The gap between them is the
// whole reason the index exists, and the reason a filter that cannot drive one
// is refused rather than silently scanned.

import (
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

const kvQueryBenchKeys = 100_000

// newKVQueryBenchTx builds an indexed store holding kvQueryBenchKeys records
// under the definition's SINGLE key prefix. One key in a thousand carries the
// value the benchmarks select on, so an indexed page reads about a hundred
// candidates out of a hundred thousand keys.
//
// The keyspace is one logical prefix; the cache underneath it is the ordinary
// 8-shard one, so the walk the scan benchmarks pay for is the walk a real
// deployment pays for.
func newKVQueryBenchTx(b *testing.B) *TxContext {
	b.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 8
	c, err := cache.New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = c.Close() })

	idx := NewKVIndexFor(c)
	def, err := kvindex.DefFrom(wire.KVIndexDef{
		Name:        wiringIndexName,
		KeyPrefix:   []byte("u:"),
		PayloadPath: "rc",
		Kind:        wire.KVIndexKindScalar,
		Enabled:     true,
	}, 1)
	if err != nil {
		b.Fatal(err)
	}
	idx.Install([]kvindex.Def{def})
	idx.MarkReady(wiringIndexName)

	tx := NewTxContextWithIndex(c, nil, idx)
	for i := 0; i < kvQueryBenchKeys; i++ {
		rc := int64(1)
		if i%1000 == 0 {
			rc = 7
		}
		if err := tx.PutIndexed([]byte(fmt.Sprintf("u:%06d", i)), kvRec(rc, "gold"), 0); err != nil {
			b.Fatal(err)
		}
	}
	return tx
}

func benchKVQuery(b *testing.B, tx *TxContext, a wire.KVQueryArgs) {
	b.Helper()
	args, err := wire.EncodeKVQueryArgs(a)
	if err != nil {
		b.Fatal(err)
	}
	// One untimed call, both to fault the pages in and to prove the benchmark
	// is measuring an answer rather than an error.
	out, err := handleKVQuery(tx, args)
	if err != nil {
		b.Fatal(err)
	}
	res, err := wire.DecodeKVQueryResult(out)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportMetric(float64(len(res.Rows)), "rows/page")

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := handleKVQuery(tx, args); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkKVQueryIndexedEq measures a page whose candidates come from the
// index: one map probe, then one live re-read and one predicate evaluation per
// candidate.
func BenchmarkKVQueryIndexedEq(b *testing.B) {
	tx := newKVQueryBenchTx(b)
	benchKVQuery(b, tx, wire.KVQueryArgs{
		Index:  wiringIndexName,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  100,
	})
}

// BenchmarkKVQueryScanPage measures the same answer without an index: one full
// walk of the keyspace into the bounded chunk heap, then the same verify step
// over the chunk.
func BenchmarkKVQueryScanPage(b *testing.B) {
	tx := newKVQueryBenchTx(b)
	benchKVQuery(b, tx, wire.KVQueryArgs{
		Scan:   true,
		Filter: eqFilter("rc", vtypes.NewInt(7)),
		Limit:  100,
	})
}
