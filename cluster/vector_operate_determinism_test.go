// SPDX-License-Identifier: Apache-2.0

package cluster

// Replica agreement for vector_operate.
//
// The op-list is applied INSIDE each replica's FSM, so every replica runs the
// arithmetic itself rather than receiving the resulting bytes. That makes the
// clock the whole risk: a STAMP op resolved from a wall clock would resolve to a
// different millisecond on each node and the replicas would diverge silently,
// with no error anywhere and nothing but a later byte comparison to reveal it.
//
// This test drives a STAMP-carrying op-list plus two ADDs through the replicated
// apply path on a 3-node RF=3 cluster, then reads each node's OWN local replica
// (CallPhysical with leaderOnly=false, the pattern TestClusterLinearizable* uses
// for an AnyReplica read) and asserts the stored record bytes are IDENTICAL on
// all three — the STAMPed field included.
//
// On the stamp's VALUE, see the note at the end of the test: shard's
// EnableApplyStamp rollout flag is off cluster-wide, so the entries reach every
// replica unstamped and STAMP resolves to 0 on all three. That does not weaken
// what is being tested — a wall clock read inside the apply would still differ
// per node and break byte identity — it only means this test cannot also assert
// that the stamp is a real timestamp.

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// operateFieldPath addresses a dynamic-mode record field by name.
func operateFieldPath(name string) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: name}}
}

// getFlagsPayloadOnly is the vector_get projection byte for payload-only (bit 1);
// it mirrors ops' getFlagWithPayload.
const getFlagsPayloadOnly uint8 = 1 << 1

// readRecordFromNode reads the point's payload from n's OWN replica (no leader
// routing) and returns the record bytes under key plus the point's version.
func readRecordFromNode(t *testing.T, n *Node, physCol string, id uint64, key string) ([]byte, uint64) {
	t.Helper()
	body, err := n.CallPhysical(physCol, "vector_get", ops.EncodeVectorGetArgs(physCol, id, getFlagsPayloadOnly), false)
	if err != nil {
		t.Fatalf("CallPhysical vector_get: %v", err)
	}
	found, _, meta, _, _, version, err := ops.DecodeVectorGetResultV(body)
	if err != nil {
		t.Fatalf("DecodeVectorGetResultV: %v", err)
	}
	if !found {
		return nil, 0
	}
	v, ok := meta[key]
	if !ok {
		return nil, version
	}
	return append([]byte(nil), v.Rec...), version
}

// TestVectorOperateReplicasAgreeByteForByte is the determinism proof at the
// cluster level: the record every replica stores after a STAMP-carrying operate
// must be byte-identical. A wall-clock leak in any replica's apply makes the
// stamped field differ and fails this test.
func TestVectorOperateReplicasAgreeByteForByte(t *testing.T) {
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 3) // RF=3: every node replicates every shard
	ctx := context.Background()

	// Single-partition physical name ⇒ one Raft group, deterministic routing.
	const coll = "operate/docs"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	if _, err := tc.client.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	const id = uint64(7)
	if _, err := tc.client.Call(ctx, "vector_upsert",
		ops.EncodeVectorUpsertArgs(physCol, id, []float32{1, 0, 0, 0}, "", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	// A STAMP plus two ADDs: the STAMP is the wall-clock trap, the ADDs prove the
	// arithmetic itself replays identically.
	opList := &wire.OperateArgs{
		Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpSTAMP, Type: wire.OperateTypeU64, Aux: wire.OperateStampMs, Path: operateFieldPath("t")},
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: operateFieldPath("rc"), A: 1},
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: operateFieldPath("bc"), A: 5},
		},
	}
	args, err := wire.EncodeVectorOperateArgs(physCol, id, "session", opList, 0, false)
	if err != nil {
		t.Fatalf("EncodeVectorOperateArgs: %v", err)
	}
	body, err := tc.client.Call(ctx, "vector_operate", args)
	if err != nil {
		t.Fatalf("vector_operate: %v", err)
	}
	found, res, err := wire.DecodeVectorOperateResult(body)
	if err != nil {
		t.Fatalf("DecodeVectorOperateResult: %v", err)
	}
	if !found || res == nil || res.Status != wire.OperateStatusOK {
		t.Fatalf("vector_operate: found=%v res=%+v", found, res)
	}

	// Gate on the DATA, not on raft's applied_index: that counter advances at
	// enqueue, so it is only a proxy for the apply having run. Poll every node's
	// own replica until it carries a record for the point.
	nodes := make([]*Node, 0, len(tc.nodes))
	for _, n := range tc.nodes {
		if n != nil {
			nodes = append(nodes, n)
		}
	}
	if len(nodes) != 3 {
		t.Fatalf("expected 3 live nodes, got %d", len(nodes))
	}

	deadline := time.Now().Add(20 * time.Second)
	var recs [][]byte
	var versions []uint64
	for {
		recs = recs[:0]
		versions = versions[:0]
		ready := true
		for _, n := range nodes {
			rec, ver := readRecordFromNode(t, n, physCol, id, "session")
			if len(rec) == 0 {
				ready = false
				break
			}
			recs = append(recs, rec)
			versions = append(versions, ver)
		}
		if ready {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for every replica to apply the operate")
		}
		time.Sleep(20 * time.Millisecond)
	}

	for i := 1; i < len(recs); i++ {
		if versions[i] != versions[0] {
			t.Fatalf("replica %d version = %d, replica 0 = %d — replicas disagree on the CAS version",
				i, versions[i], versions[0])
		}
		if !bytes.Equal(recs[i], recs[0]) {
			t.Fatalf("replica %d stored DIFFERENT record bytes than replica 0 — the apply is not deterministic "+
				"(a wall clock leaked into STAMP?)\n replica0=%x\n replica%d=%x", i, recs[0], i, recs[i])
		}
	}

	// Positive control: the bytes compared above are a REAL record carrying all
	// three ops' results, not two empty payloads that trivially agree.
	rec, err := wire.DecodeRecord(recs[0])
	if err != nil {
		t.Fatalf("DecodeRecord: %v", err)
	}
	// Dynamic-mode fields are stored sorted by name, so look each up by name.
	fields := make(map[string]wire.Cell, len(rec.Fields))
	for _, f := range rec.Fields {
		fields[f.Name] = f.Cell
	}
	for _, want := range []struct {
		name string
		val  uint64
	}{{"rc", 1}, {"bc", 5}} {
		c, ok := fields[want.name]
		if !ok {
			t.Fatalf("record carries no field %q: %+v", want.name, rec.Fields)
		}
		if c.U != want.val {
			t.Fatalf("field %q = %d, want %d — the op-list did not apply as written", want.name, c.U, want.val)
		}
	}
	// The STAMPed field must EXIST — STAMP ran on every replica and its value is
	// part of the byte-identity assertion above.
	//
	// Its value is 0 here, and that is the harness, not a bug: shard.Config's
	// EnableApplyStamp is the two-phase rollout flag for the leader-stamped apply
	// clock and it is OFF everywhere outside shard's own tests, so the leader emits
	// unstamped log entries and every replica applies with stampMs = 0. The
	// determinism property under test is unaffected — a wall clock consulted inside
	// the apply would still resolve differently on each node and break the
	// byte-identity assertion — but a non-zero assertion here would be asserting the
	// rollout flag, not the op. The handler's own stamp threading is pinned by
	// ops.TestVectorOperateIsDeterministicUnderTheStamp, which drives the stamp
	// directly.
	if _, ok := fields["t"]; !ok {
		t.Fatalf("record carries no STAMPed field %q: %+v", "t", rec.Fields)
	}
}
