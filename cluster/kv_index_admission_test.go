// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// --- admission: the shallow check is not the whole check ------------------

// kvIndexUninstallablePaths are payload paths wire.KVIndexDef.Validate ACCEPTS
// and ops/kvindex.DefFrom REFUSES. Validate cannot parse a path (sdk/record
// imports sdk/wire), so it only spells the "#count" suffix and the '/' ban;
// the grammar underneath is the parser's.
//
// Each one is an uninstallable definition: committed, it lands in the meta
// catalog, every node's observer refuses it, and every query naming it comes
// back as RETRYABLE "still building" for an index that will never build. Kind
// tracks the "#count" suffix here only because Validate requires it to — the
// paths fail for the path's sake, not the kind's.
var kvIndexUninstallablePaths = []struct {
	name string
	path string
	kind uint8
}{
	{"count_of_no_field", "#count", wire.KVIndexKindCount},
	{"non_canonical_position", "#00001", wire.KVIndexKindScalar},
	{"invalid_character", `"x"`, wire.KVIndexKindScalar},
	{"count_twice", "a#count#count", wire.KVIndexKindCount},
}

// uninstallableDef builds the definition for one such path.
func uninstallableDef(name, path string, kind uint8) wire.KVIndexDef {
	return wire.KVIndexDef{
		Name: name, KeyPrefix: []byte("u:"), PayloadPath: path, Kind: kind, Enabled: true,
	}
}

// commitUnvalidatedKVIndex puts d on the meta log WITHOUT the admission check,
// which is the only way an unbuildable definition can get there now.
//
// It is exactly VERSION SKEW in one process: a node running a build whose path
// grammar admits d proposes it, and the meta FSM — whose check must be identical
// on every node at every version, so it is the shallow one — accepts it. Tests
// that need a definition this build cannot parse must plant it this way;
// Node.SetKVIndex and MetaRaft.ApplySetKVIndex now refuse it, which is the point
// of the admission check.
//
// The node must be the meta leader (every caller uses a single-node cluster).
// Raft.Apply's future carries the FSM's own return value, so by the time this
// returns the local FSM has applied the entry and no polling is needed.
func commitUnvalidatedKVIndex(t *testing.T, n *Node, d wire.KVIndexDef) {
	t.Helper()
	entry, err := encodeLogEntry(LogEntry{Op: OpSetKVIndex, KVIndex: d})
	if err != nil {
		t.Fatalf("encodeLogEntry: %v", err)
	}
	f := n.meta.Raft.Apply(entry, 5*time.Second)
	if err := f.Error(); err != nil {
		t.Fatalf("meta Apply(%q): %v", d.Name, err)
	}
	if respErr, isErr := f.Response().(error); isErr {
		t.Fatalf("the meta FSM refused %q: %v", d.Name, respErr)
	}
}

// TestValidateKVIndexDefParsesThePath is the unit-level statement of the rule:
// the admission check is Validate AND the parse, and a drop is exempt from the
// parse.
func TestValidateKVIndexDefParsesThePath(t *testing.T) {
	for _, tc := range kvIndexUninstallablePaths {
		t.Run(tc.name, func(t *testing.T) {
			d := uninstallableDef(tc.name, tc.path, tc.kind)
			// The premise: the shallow check really does pass. Without this the
			// test would still pass if Validate started catching these, and would
			// no longer be testing admission at all.
			if err := d.Validate(); err != nil {
				t.Fatalf("premise broken: Validate now refuses %q by itself: %v", tc.path, err)
			}
			err := validateKVIndexDef(d)
			if err == nil {
				t.Fatalf("validateKVIndexDef accepted the uninstallable path %q", tc.path)
			}
			// It must be the PERMANENT sentinel: classified anything else, the
			// client is told to retry a definition that can never be installed.
			if !errors.Is(err, wire.ErrKVIndexDef) {
				t.Fatalf("err = %v, want a wire.ErrKVIndexDef", err)
			}
		})
	}

	// A DROP carries a placeholder path nothing ever reads (the FSM deletes the
	// entry rather than storing it), so the parse must not run on it — otherwise
	// a definition a newer build wrote could not be dropped by this one.
	drop := uninstallableDef("by_age", "#count", wire.KVIndexKindCount)
	drop.Enabled = false
	if err := validateKVIndexDef(drop); err != nil {
		t.Fatalf("a drop was refused over its placeholder path: %v", err)
	}
	// But a drop still has to satisfy the SHAPE check: the codec writes u8 length
	// prefixes without checking them.
	drop.Name = "not a legal name!"
	if err := validateKVIndexDef(drop); !errors.Is(err, wire.ErrKVIndexDef) {
		t.Fatalf("a drop naming an illegal index = %v, want a wire.ErrKVIndexDef", err)
	}

	// The control: a definition that parses is admitted.
	if err := validateKVIndexDef(kvIndexDefFixture("by_age", "u:")); err != nil {
		t.Fatalf("a valid definition was refused: %v", err)
	}
}

// waitMetaLogQuiescent returns the meta log's last index once it has stopped
// moving, so a "nothing was committed" assertion is not racing the shard
// formation and catalog writes a fresh cluster makes on its own.
func waitMetaLogQuiescent(t *testing.T, n *Node) uint64 {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	last := n.meta.Raft.LastIndex()
	for time.Now().Before(deadline) {
		time.Sleep(300 * time.Millisecond)
		if cur := n.meta.Raft.LastIndex(); cur == last {
			return last
		} else {
			last = cur
		}
	}
	t.Fatalf("the meta log never went quiet (last index %d)", last)
	return 0
}

// TestKVIndexAdmissionRefusesUninstallableDefinitions walks every entry point a
// definition can reach the meta log through and pins that an uninstallable one
// is refused at all of them, with the log untouched.
//
// The refusal has to happen HERE. Past the commit the definition is cluster-wide
// truth: every node's observer rejects it (RejectedDefs=1 everywhere) while the
// name IS in the catalog, so classifyKVQueryErr turns each group's permanent
// "no such index" into the retryable ErrIndexBuilding and a correct client
// retries forever, one full fan-out per attempt.
func TestKVIndexAdmissionRefusesUninstallableDefinitions(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	before := waitMetaLogQuiescent(t, n)

	for _, p := range kvIndexUninstallablePaths {
		t.Run(p.name, func(t *testing.T) {
			d := uninstallableDef(p.name, p.path, p.kind)

			// 1. The public entry point.
			if err := n.SetKVIndex(d, 5*time.Second); !errors.Is(err, wire.ErrKVIndexDef) {
				t.Fatalf("SetKVIndex = %v, want a wire.ErrKVIndexDef", err)
			}
			// 2. The FORWARDED admin op, which is what a peer and every client
			//    transport actually call.
			if _, err := n.Call(opKVIndexSetName, wire.EncodeKVIndexSetArgs(d)); !errors.Is(err, wire.ErrKVIndexDef) {
				t.Fatalf("%s = %v, want a wire.ErrKVIndexDef", opKVIndexSetName, err)
			}
			// 3. The direct proposer, one layer under both.
			if err := n.meta.ApplySetKVIndex(d, 5*time.Second); !errors.Is(err, wire.ErrKVIndexDef) {
				t.Fatalf("ApplySetKVIndex = %v, want a wire.ErrKVIndexDef", err)
			}
			if _, in := n.meta.FSM.KVIndexes()[p.name]; in {
				t.Fatal("a refused definition entered the meta catalog")
			}
		})
	}

	if after := n.meta.Raft.LastIndex(); after != before {
		t.Fatalf("the meta log grew from %d to %d: a refused definition was proposed", before, after)
	}
	// And it is invisible to the catalog listing, which is what a client polls
	// for readiness — a rejected definition sitting there "not ready" forever is
	// the shape this refusal exists to prevent.
	defs, _, err := n.ListKVIndexes(5 * time.Second)
	if err != nil {
		t.Fatalf("ListKVIndexes: %v", err)
	}
	for _, d := range defs {
		for _, p := range kvIndexUninstallablePaths {
			if d.Name == p.name {
				t.Fatalf("%q is in __kv_index_list__", d.Name)
			}
		}
	}
	// The control: a definition that parses still goes all the way through.
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("a valid definition was refused after the bad ones: %v", err)
	}
	if _, in := n.meta.FSM.KVIndexes()["by_age"]; !in {
		t.Fatal("the valid definition did not reach the catalog")
	}
}

// TestHandleSetKVIndexRefusesUninstallableDefinitions covers the FORWARDED
// handler directly. It is the route every remote caller takes: REST and the
// native Go client both send __kv_index_set__, and a non-leader node forwards
// the same op to the meta leader, so this handler is where most admissions
// actually happen.
//
// It asserts the two things a peer-facing refusal has to get right — the error
// reaches the caller in the PERMANENT class, and nothing was committed — rather
// than only that an error came back.
func TestHandleSetKVIndexRefusesUninstallableDefinitions(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]

	for _, p := range kvIndexUninstallablePaths {
		t.Run(p.name, func(t *testing.T) {
			d := uninstallableDef(p.name, p.path, p.kind)
			body, err := n.handleSetKVIndex(wire.EncodeKVIndexSetArgs(d))
			if err == nil {
				t.Fatalf("handleSetKVIndex accepted the uninstallable path %q", p.path)
			}
			if body != nil {
				t.Fatalf("a refused write returned a body: %q", body)
			}
			if class, found := kvQueryClassOf(err); !found || class != ops.KVQueryErrPermanent {
				t.Fatalf("err %v classifies as %v (in the family: %v), want permanent", err, class, found)
			}
			if _, in := n.meta.FSM.KVIndexes()[p.name]; in {
				t.Fatal("a refused definition entered the meta catalog")
			}
		})
	}

	// The control: the same handler still commits a definition that parses, so
	// the refusals above are about the path and not about the handler.
	good := kvIndexDefFixture("by_age", "u:")
	if _, err := n.handleSetKVIndex(wire.EncodeKVIndexSetArgs(good)); err != nil {
		t.Fatalf("handleSetKVIndex refused a valid definition: %v", err)
	}
	if _, in := n.meta.FSM.KVIndexes()["by_age"]; !in {
		t.Fatal("the valid definition did not reach the catalog")
	}
}

// kvQueryClassOf reports how the shared error family classifies err, which is
// what all three transports spell in their own vocabulary. Asserting through the
// family rather than against a sentinel means this test fails if the sentinel is
// ever moved between classes.
func kvQueryClassOf(err error) (ops.KVQueryErrorClass, bool) {
	for _, spec := range ops.KVQueryErrorFamily() {
		if errors.Is(err, spec.Err) {
			return spec.Class, true
		}
	}
	return 0, false
}

// TestRejectedDefinitionIsPermanentNotBuilding covers the residue admission
// cannot cover: VERSION SKEW. A newer node admits a path grammar this build does
// not have, so the definition is in the catalog and this node still cannot build
// it. Reported as "still building", that is the retry-forever loop again, for an
// index that will never install here.
//
// The skew is simulated the only honest way available in one build: the entry is
// proposed to the meta log WITHOUT the admission check (commitUnvalidatedKVIndex),
// which is exactly what a peer running a different binary does.
func TestRejectedDefinitionIsPermanentNotBuilding(t *testing.T) {
	tc := newTestCluster(t, 1, 2)
	n := tc.nodes[0]
	freezeKVIndexObserver(t, n)

	commitUnvalidatedKVIndex(t, n, uninstallableDef("from_the_future", "#count", wire.KVIndexKindCount))
	if err := n.SetKVIndex(kvIndexDefFixture("by_age", "u:"), 5*time.Second); err != nil {
		t.Fatalf("SetKVIndex(good): %v", err)
	}
	// One observer pass by hand: it is the pass that decides which definitions
	// this node can build.
	n.applyKVIndexDefs()
	if n.kvIndexStats().RejectedDefs != 1 {
		t.Fatalf("RejectedDefs = %d, want 1", n.kvIndexStats().RejectedDefs)
	}

	leaf := func(name string) error {
		return fmt.Errorf("%w: %q on shard group 0", kvindex.ErrNoSuchIndex, name)
	}
	// The skewed definition: PERMANENT, and not the retryable rewrite.
	got := n.classifyKVQueryErr("from_the_future", 0, leaf("from_the_future"))
	if !errors.Is(got, wire.ErrKVIndexDef) {
		t.Fatalf("a rejected definition classified as %v, want a wire.ErrKVIndexDef", got)
	}
	if errors.Is(got, kvindex.ErrIndexBuilding) {
		t.Fatal("a definition this node can never build was reported as still building")
	}
	// The control: a buildable definition the group has not installed yet is
	// still the retryable rewrite. Losing that would break create-then-query.
	if got := n.classifyKVQueryErr("by_age", 0, leaf("by_age")); !errors.Is(got, kvindex.ErrIndexBuilding) {
		t.Fatalf("a buildable definition classified as %v, want kvindex.ErrIndexBuilding", got)
	}
	// And a name in no catalog at all stays exactly what the leaf said.
	if got := n.classifyKVQueryErr("never_created", 0, leaf("never_created")); !errors.Is(got, kvindex.ErrNoSuchIndex) {
		t.Fatalf("an unknown index classified as %v, want kvindex.ErrNoSuchIndex", got)
	}
}
