// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// TestMetaDropOfAnAbsentKVIndexDoesNotTouchState.
//
// A drop is a create with Enabled=false, and dropping a name that was never
// there is a clean no-op. It has to be a no-op in the STATE too: materialising
// a nil catalog into an empty map makes a logically empty apply change the FSM
// and, with it, the bytes of every snapshot taken afterwards.
func TestMetaDropOfAnAbsentKVIndexDoesNotTouchState(t *testing.T) {
	f := NewMetaFSM()
	if f.State().KVIndexes != nil {
		t.Fatal("fixture: a fresh meta FSM already has a KV index catalog")
	}

	drop := wire.KVIndexDef{Name: "never-existed", PayloadPath: "-", Kind: wire.KVIndexKindScalar}
	if resp := applyKVIndexEntry(t, f, drop, 1); resp != nil {
		t.Fatalf("dropping an absent index returned %v, want a clean no-op", resp)
	}
	if got := f.State().KVIndexes; got != nil {
		t.Fatalf("a no-op drop materialised the catalog as %v; it must leave the state untouched", got)
	}

	// The control: an enabled definition does create the catalog.
	create := kvIndexDefFixture("real", "u:")
	if resp := applyKVIndexEntry(t, f, create, 2); resp != nil {
		t.Fatalf("creating an index returned %v", resp)
	}
	if _, ok := f.State().KVIndexes["real"]; !ok {
		t.Fatal("an enabled definition did not reach the catalog")
	}
}

// TestKVIndexRejectWarningIsLoggedOnChangeNotPerPass.
//
// The observer runs on every meta-index advance, and PB liveness advances it on
// its own beacon cadence, so a definition this build cannot parse used to print
// its warning about once a second per node for as long as it sat in the
// catalog — a client-authored definition with a write handle on the operator's
// log. The condition is a STATE: worth saying when it starts and when it
// clears, worth nothing repeated.
//
// The counters are deliberately NOT changed by this (Rejects still counts one
// event per pass, RejectedDefs still gauges the current set); that half is
// pinned by TestKVIndexRejectsCounterAndGauge.
func TestKVIndexRejectWarningIsLoggedOnChangeNotPerPass(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)

	warns := countingLogHandler(t)

	// "#count" with no field in front of it passes wire.KVIndexDef.Validate and
	// fails record.ParsePath in kvindex.DefFrom, so it has to be planted past the
	// admission check — exactly as TestKVIndexRejectsCounterAndGauge does.
	bad := wire.KVIndexDef{Name: "headless", PayloadPath: "#count", Kind: wire.KVIndexKindCount, Enabled: true}
	commitUnvalidatedKVIndex(t, n, bad)

	const passes = 5
	for i := 0; i < passes; i++ {
		n.applyKVIndexDefs()
	}
	if got := warns.Load(); got != 1 {
		t.Fatalf("%d passes over an unchanged rejected definition logged %d warnings, want exactly 1", passes, got)
	}
	if got := n.Stats().KVIndex.RejectedDefs; got != 1 {
		t.Fatalf("RejectedDefs = %d, want the gauge unaffected by the log change", got)
	}
	if got := n.Stats().KVIndex.Rejects; got != passes {
		t.Fatalf("Rejects = %d after %d passes, want the event counter unaffected by the log change", got, passes)
	}
	if !n.kvIndexDefRejected("headless") {
		t.Fatal("the rejected name is no longer published; classifyKVQueryErr would call the definition retryable forever")
	}
}

// countingLogHandler installs a slog handler that counts WARN records for the
// duration of one test and restores the previous default afterwards.
func countingLogHandler(t *testing.T) *atomic.Int64 {
	t.Helper()
	var n atomic.Int64
	prev := slog.Default()
	slog.SetDefault(slog.New(&kvWarnCounter{n: &n}))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &n
}

type kvWarnCounter struct{ n *atomic.Int64 }

func (h *kvWarnCounter) Enabled(context.Context, slog.Level) bool { return true }
func (h *kvWarnCounter) Handle(_ context.Context, r slog.Record) error {
	// Only THIS line: the cluster fixture emits raft warnings of its own, and
	// counting those would make the assertion a lottery.
	if r.Level >= slog.LevelWarn && strings.HasPrefix(r.Message, "kv index definition is in the meta catalog") {
		h.n.Add(1)
	}
	return nil
}
func (h *kvWarnCounter) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *kvWarnCounter) WithGroup(string) slog.Handler      { return h }
