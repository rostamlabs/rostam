// SPDX-License-Identifier: Apache-2.0

//go:build !cacheregionfr

package cache

// Region-residency measurement, COMPILED OUT. regionfr_on.go holds the real
// implementation and the explanation of what is being measured; this file is
// what every ordinary build gets.
//
// The hooks sit on the read hit path, the write path and page allocation, so
// they are gated behind a build tag rather than a runtime flag: a runtime flag
// still costs a load and a branch on every hit, to answer a question only a
// benchmark ever asks. Each stub below has an empty body and is inlined away, so
// an untagged build carries no instruction for any of them and the hit path is
// byte-for-byte what it was.
//
// Build the measurement with -tags cacheregionfr.

// regionFRInstrumented reports whether the counters are real. Benchmarks use it
// to decide whether reporting a residency metric would be reporting a zero.
const regionFRInstrumented = false

// regionFRMaxK is the largest region size, in pages, the histograms resolve.
const regionFRMaxK = 16

func regionNoteGen(*shard, uint16) {}

func regionNoteHit(*shard, *page) {}

func regionNoteInPlace(*shard, *page) {}

// regionFRCounts is the histogram pair a measurement window produces. See
// regionfr_on.go for the meaning of each field.
type regionFRCounts struct {
	Hits      uint64
	InPlace   uint64
	HitDist   [regionFRMaxK]uint64
	PlaceDist [regionFRMaxK]uint64
}

func regionFRAttach(*shard) func() { return func() {} }

func regionFRSnapshot() regionFRCounts { return regionFRCounts{} }
