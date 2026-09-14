// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// newHaltRecordingMetaFSM returns a MetaFSM whose fail-closed halt records the
// error and returns instead of exiting the test binary.
func newHaltRecordingMetaFSM() (*MetaFSM, *[]error) {
	var halts []error
	m := NewMetaFSM()
	m.onFatalApply = func(err error) { halts = append(halts, err) }
	return m, &halts
}

// applyRecovering applies one entry and survives a panic from Apply, so a test
// can still inspect the frontier afterwards. A halt implemented as a panic is
// exactly the fix this file must reject: the deferred frontier advance runs
// during the unwind.
func applyRecovering(m *MetaFSM, idx uint64, data []byte) (resp any, panicked bool) {
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	return m.Apply(&raft.Log{Index: idx, Type: raft.LogCommand, Data: data}), false
}

func mustEncodeLogEntry(t *testing.T, e LogEntry) []byte {
	t.Helper()
	b, err := encodeLogEntry(e)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestMetaFSMUnknownOpFailsClosed is the gate for issue #113. A committed meta
// entry whose op this binary does not recognise was proposed by a binary that
// does, so every peer on that binary APPLIES it. This node must stop — and must
// not record the entry as applied, or a restart would resume past it.
func TestMetaFSMUnknownOpFailsClosed(t *testing.T) {
	m, halts := newHaltRecordingMetaFSM()
	known := mustEncodeLogEntry(t, LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8})
	if resp := m.Apply(&raft.Log{Index: 5, Type: raft.LogCommand, Data: known}); resp != nil {
		t.Fatalf("known apply: %v", resp)
	}
	before := m.State()

	poison := mustEncodeLogEntry(t, LogEntry{Op: metaTestUnknownOp, Collection: "docs", Partitions: 99})
	_, panicked := applyRecovering(m, 6, poison)

	if got := m.AppliedIndex(); got != 5 {
		t.Fatalf("applied frontier = %d after an unknown op at index 6, want 5 (panicked=%v): the entry must NOT be recorded as applied", got, panicked)
	}
	if len(*halts) != 1 {
		t.Fatalf("halts = %d, want exactly 1: an unknown op must fail closed, not return an error raft ignores", len(*halts))
	}
	if !errors.Is((*halts)[0], errMetaUnknownOp) {
		t.Fatalf("halt error = %v, want errMetaUnknownOp", (*halts)[0])
	}
	if !equalState(before, m.State()) {
		t.Fatal("state mutated by an unknown op")
	}

	// Restart: raft restores the latest snapshot and redelivers every entry after
	// it. The snapshot a halted node could have taken stamps the known frontier,
	// so the replay meets the SAME entry — and must stop on it again.
	snap, err := m.SnapshotBytes()
	if err != nil {
		t.Fatal(err)
	}
	restarted, rhalts := newHaltRecordingMetaFSM()
	if err := restarted.Restore(io.NopCloser(bytes.NewReader(snap))); err != nil {
		t.Fatal(err)
	}
	if got := restarted.AppliedIndex(); got != 5 {
		t.Fatalf("restored frontier = %d, want 5: a snapshot after the halt must not cover the unknown entry", got)
	}
	_, panicked = applyRecovering(restarted, 6, poison)
	if len(*rhalts) != 1 {
		t.Fatalf("after restart: halts = %d, want 1 — the node skipped the unknown entry instead of stopping on it again (panicked=%v)", len(*rhalts), panicked)
	}
	if got := restarted.AppliedIndex(); got != 5 {
		t.Fatalf("after restart: applied frontier = %d, want 5", got)
	}
}

// metaOpRegistry is every op this binary applies. It is deliberately a literal:
// adding an op to MetaFSM.Apply without adding it here fails the table below, so
// no op becomes "known" — or stops being known — by accident.
var metaOpRegistry = map[Op]string{
	OpSetMembers:        "OpSetMembers",
	OpSetPlacement:      "OpSetPlacement",
	OpSetCatalogEntry:   "OpSetCatalogEntry",
	OpSetCatalogReshard: "OpSetCatalogReshard",
	OpSetAliasBatch:     "OpSetAliasBatch",
	OpSetShardEpoch:     "OpSetShardEpoch",
	OpSetShardISR:       "OpSetShardISR",
	OpShardLeaseRenew:   "OpShardLeaseRenew",
	OpSetShardFormer:    "OpSetShardFormer",
	OpSetKVIndex:        "OpSetKVIndex",
}

// fullLogEntry is one body every known op accepts as well-formed, so the table
// can drive all of them through the same entry and differ only in the op byte.
func fullLogEntry(op Op) LogEntry {
	return LogEntry{
		Op:                op,
		Members:           []Peer{{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"}},
		NumShards:         4,
		ReplicationFactor: 1,
		ShardID:           0,
		Owners:            []string{"n1"},
		Collection:        "default/docs",
		Partitions:        8,
		ReshardStatus:     1,
		ReshardTargetP:    16,
		ReshardTargetGen:  1,
		AliasBatch:        []AliasAction{{Alias: "default/a", Canonical: "default/docs"}},
		Epoch:             1,
		Primary:           "n1",
		ISR:               []string{"n1"},
		Node:              "n1",
		LeaseRenew:        []ShardEpochPair{{ShardID: 0, Epoch: 1}},
		KVIndex:           wire.KVIndexDef{Name: "age", PayloadPath: "age", Kind: wire.KVIndexKindScalar, Enabled: true},
	}
}

// TestMetaFSMOpRegistryFailClosedTable walks EVERY op byte. Each registered op
// applies normally and advances the frontier; every other value — the zero
// OpUnknown, the never-implemented reserved 2 and 3, and everything past the
// registry — halts without advancing. There is no retired op: none has ever
// been removed, so none is known-and-ignored.
func TestMetaFSMOpRegistryFailClosedTable(t *testing.T) {
	const idx = 10
	for v := 0; v <= 255; v++ {
		op := Op(v)
		m, halts := newHaltRecordingMetaFSM()
		resp, panicked := applyRecovering(m, idx, mustEncodeLogEntry(t, fullLogEntry(op)))
		if panicked {
			t.Fatalf("op %d: Apply panicked", v)
		}
		if name, known := metaOpRegistry[op]; known {
			if len(*halts) != 0 {
				t.Errorf("%s (%d): halted on a registered op: %v", name, v, (*halts)[0])
			}
			if err, isErr := resp.(error); isErr {
				t.Errorf("%s (%d): apply error %v", name, v, err)
			}
			if got := m.AppliedIndex(); got != idx {
				t.Errorf("%s (%d): frontier = %d, want %d", name, v, got, idx)
			}
			continue
		}
		if len(*halts) != 1 || !errors.Is((*halts)[0], errMetaUnknownOp) {
			t.Errorf("op %d: halts = %v, want one errMetaUnknownOp", v, *halts)
		}
		if got := m.AppliedIndex(); got != 0 {
			t.Errorf("op %d: frontier = %d, want 0", v, got)
		}
	}
}

// TestMetaFSMRestoreRefusesUnrecognisedState is the snapshot half of the same
// hazard. A node that halts on an unknown op can still be handed the RESULT of
// that op: once the leader compacts past the entry it sends an InstallSnapshot
// instead. gob silently drops fields the reader does not have, so without this
// check the node would adopt the snapshot, lose the new state, and move on — the
// skip, one step removed.
func TestMetaFSMRestoreRefusesUnrecognisedState(t *testing.T) {
	// futureState is State as a newer binary might write it: one more top-level
	// field, carrying data.
	type futureState struct {
		NumShards       int
		Catalog         map[string]uint32
		FutureCatalog   map[string]string
		LastIndex       uint64
		PopulatedFields []string
	}
	encode := func(t *testing.T, v any) []byte {
		t.Helper()
		var buf bytes.Buffer
		if err := gobEncodeForTest(&buf, v); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}

	t.Run("populated unknown field is refused", func(t *testing.T) {
		blob := encode(t, futureState{
			NumShards:       4,
			Catalog:         map[string]uint32{"docs": 8},
			FutureCatalog:   map[string]string{"x": "y"},
			LastIndex:       99,
			PopulatedFields: []string{"Catalog", "FutureCatalog", "LastIndex", "NumShards"},
		})
		m := NewMetaFSM()
		err := m.Restore(io.NopCloser(bytes.NewReader(blob)))
		if err == nil || !strings.Contains(err.Error(), "FutureCatalog") {
			t.Fatalf("restore err = %v, want a refusal naming FutureCatalog", err)
		}
		if got := m.AppliedIndex(); got != 0 {
			t.Fatalf("frontier = %d after a refused restore, want 0", got)
		}
		if _, ok := m.CatalogLookup("docs"); ok {
			t.Fatal("a refused snapshot must not install any state")
		}
	})

	t.Run("populated unknown nested field is refused", func(t *testing.T) {
		blob := encode(t, futureState{
			NumShards:       4,
			LastIndex:       99,
			PopulatedFields: []string{"CatalogReshard", "CatalogReshard[].FutureField", "LastIndex", "NumShards"},
		})
		err := NewMetaFSM().Restore(io.NopCloser(bytes.NewReader(blob)))
		if err == nil || !strings.Contains(err.Error(), "CatalogReshard[].FutureField") {
			t.Fatalf("restore err = %v, want a refusal naming CatalogReshard[].FutureField", err)
		}
	})

	t.Run("empty unknown field is accepted", func(t *testing.T) {
		// Nothing would be lost: the reader's implicit value for a field it does
		// not have is the same zero value the writer holds.
		blob := encode(t, futureState{
			NumShards:       4,
			Catalog:         map[string]uint32{"docs": 8},
			LastIndex:       99,
			PopulatedFields: []string{"Catalog", "LastIndex", "NumShards"},
		})
		m := NewMetaFSM()
		if err := m.Restore(io.NopCloser(bytes.NewReader(blob))); err != nil {
			t.Fatalf("restore: %v", err)
		}
		if p, ok := m.CatalogLookup("docs"); !ok || p != 8 {
			t.Fatalf("catalog docs = (%d,%v), want (8,true)", p, ok)
		}
	})

	t.Run("every field this binary writes round-trips", func(t *testing.T) {
		// No false refusals: apply every registered op so every State field and
		// nested field is populated, then snapshot and restore on this binary.
		src, halts := newHaltRecordingMetaFSM()
		i := uint64(1)
		for op := range metaOpRegistry {
			src.Apply(&raft.Log{Index: i, Type: raft.LogCommand, Data: mustEncodeLogEntry(t, fullLogEntry(op))})
			i++
		}
		if len(*halts) != 0 {
			t.Fatalf("registered op halted: %v", (*halts)[0])
		}
		snap, err := src.SnapshotBytes()
		if err != nil {
			t.Fatal(err)
		}
		// The stamp must really describe the snapshot, nested fields included —
		// otherwise the refusal above could never fire on a real one.
		stamped, err := decodeState(snap)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"Members", "Members[].NodeID", "CatalogReshard[].TargetP", "KVIndexes[].Def.Name", "LastIndex"} {
			found := false
			for _, p := range stamped.PopulatedFields {
				found = found || p == want
			}
			if !found {
				t.Errorf("snapshot PopulatedFields = %v, missing %q", stamped.PopulatedFields, want)
			}
		}
		dst := NewMetaFSM()
		if err := dst.Restore(io.NopCloser(bytes.NewReader(snap))); err != nil {
			t.Fatalf("restore of this binary's own snapshot: %v", err)
		}
		if !equalState(src.State(), dst.State()) {
			t.Fatal("state mismatch after round trip")
		}
		if dst.State().PopulatedFields != nil {
			t.Fatal("PopulatedFields is snapshot metadata and must not leak into live state")
		}
	})
}
