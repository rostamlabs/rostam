// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"math/rand"
	"sort"
	"testing"
)

// TestLatHistBucketsAreMonotoneAndBounded checks the two properties every quantile this
// histogram reports rests on: the mapping never goes backwards, so a larger latency can
// never land in an earlier bucket and be reported as faster; and a bucket's lower bound is
// never above the value that fell into it, so a reported quantile is always a latency some
// write actually met or exceeded rather than an optimistic guess.
func TestLatHistBucketsAreMonotoneAndBounded(t *testing.T) {
	prev := -1
	for _, ns := range latHistProbeValues() {
		idx := latIndex(ns)
		if idx < prev {
			t.Fatalf("latIndex(%d) = %d went backwards from %d", ns, idx, prev)
		}
		prev = idx
		if lo := latValue(idx); lo > ns {
			t.Fatalf("latIndex(%d) = %d, whose lower bound %d is ABOVE the value", ns, idx, lo)
		}
		// And the bucket is fine enough for the ratios this benchmark reports: worst case
		// one part in latSub, so never more than ~13%% wide.
		if ns >= latSub {
			if lo := latValue(idx); ns-lo > lo/latSub+1 {
				t.Fatalf("bucket %d (lower bound %d) is too wide to hold %d", idx, lo, ns)
			}
		}
	}
}

// latHistProbeValues walks the interesting region densely and the rest by octave: every
// value below one microsecond, then bucket edges and their neighbours out to a second.
func latHistProbeValues() []uint64 {
	var out []uint64
	for ns := uint64(0); ns < 1024; ns++ {
		out = append(out, ns)
	}
	for shift := uint(10); shift < 30; shift++ {
		base := uint64(1) << shift
		for sub := range uint64(latSub) {
			step := base / latSub
			v := base + sub*step
			out = append(out, v, v+1)
			if v > 0 {
				out = append(out, v-1)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// TestLatHistQuantilesTrackTheTruth is the end-to-end check: against a known sample the
// histogram's quantiles must land within a bucket of the exact answer. A tail benchmark
// whose quantiles are quietly wrong is worse than no tail benchmark, because its numbers
// still look plausible.
func TestLatHistQuantilesTrackTheTruth(t *testing.T) {
	rng := rand.New(rand.NewSource(7)) //nolint:gosec // test RNG
	// A shape like the one being measured: almost everything fast, a thin slow tail.
	raw := make([]uint64, 0, 200_000)
	var h latHist
	for range 200_000 {
		ns := uint64(200 + rng.Intn(120))
		if rng.Intn(1000) == 0 {
			ns = uint64(50_000 + rng.Intn(400_000)) // the eviction-shaped tail
		}
		raw = append(raw, ns)
		h.add(ns)
	}
	sort.Slice(raw, func(i, j int) bool { return raw[i] < raw[j] })

	for _, q := range []float64{0.5, 0.9, 0.99, 0.999} {
		want := raw[int(q*float64(len(raw)))]
		got := h.quantile(q)
		// The histogram reports a bucket lower bound, so it may sit up to one bucket width
		// below the exact value and never above it.
		if got > want || want-got > want/latSub+1 {
			t.Fatalf("q%.3f: histogram says %d, exact is %d", q, got, want)
		}
	}
	if h.max != raw[len(raw)-1] {
		t.Fatalf("max = %d, exact is %d", h.max, raw[len(raw)-1])
	}
	if h.n != uint64(len(raw)) {
		t.Fatalf("n = %d, want %d", h.n, len(raw))
	}
}

// TestLatHistMergeIsAdditive pins the merge the workers rely on: per-worker histograms
// combined after the timed loop must say exactly what one histogram fed every sample would
// have said.
func TestLatHistMergeIsAdditive(t *testing.T) {
	rng := rand.New(rand.NewSource(11)) //nolint:gosec // test RNG
	var whole, a, bh latHist
	for i := range 50_000 {
		ns := uint64(1 + rng.Intn(1_000_000))
		whole.add(ns)
		if i%2 == 0 {
			a.add(ns)
		} else {
			bh.add(ns)
		}
	}
	var merged latHist
	merged.merge(&a)
	merged.merge(&bh)
	if merged.n != whole.n || merged.max != whole.max {
		t.Fatalf("merged (n=%d max=%d) != whole (n=%d max=%d)", merged.n, merged.max, whole.n, whole.max)
	}
	for i := range whole.b {
		if merged.b[i] != whole.b[i] {
			t.Fatalf("bucket %d: merged %d, whole %d", i, merged.b[i], whole.b[i])
		}
	}
}
