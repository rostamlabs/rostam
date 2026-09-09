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
	"sync/atomic"
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

// setNodeVectorClock pins n's TTL wall clock on every shard store it hosts. The
// clock reaches only the NON-apply expiry sites plus the wall-clock branch of the
// write paths (vector.CollectionStore.SetNowFunc's contract), which is exactly
// what an unstamped replicated apply reads — so skewing it per node is how a
// replica's local clock is put either side of a per-key deadline.
func setNodeVectorClock(t *testing.T, n *Node, fn func() int64) {
	t.Helper()
	for _, s := range n.shards {
		if s == nil {
			continue
		}
		if vs := s.VectorStore(); vs != nil {
			vs.SetNowFunc(fn)
		}
	}
}

// TestVectorOperateReplicasAgreeWithAPerKeyDeadline is the byte-agreement proof
// for the case the STAMP test cannot reach: the record's payload key carries a
// per-key DEADLINE, and the replicas' wall clocks straddle it.
//
// With the leader apply stamp off (it is off cluster-wide today) every replica
// applies with its own wall clock. If the mutator were allowed to judge the key's
// deadline against that clock, a replica past the deadline would read the record
// as ABSENT and create a fresh one while a replica short of it would increment
// the stored one — both commit, both bump the version, neither errors, and the
// two replicas hold different bytes forever. The engines therefore refuse to
// consult the clock for the key's deadline unless the apply is stamped (see
// vector.recordKeyPastDeadline), which is what this test pins end to end.
//
// The clocks are pinned through CollectionStore.SetNowFunc on each node's shard
// stores, including for the final read: a node whose clock is past the deadline
// would otherwise hide the key from its own vector_get and the comparison would
// read two empty payloads that trivially agree.
func TestVectorOperateReplicasAgreeWithAPerKeyDeadline(t *testing.T) {
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 3) // RF=3: every node replicates every shard
	ctx := context.Background()

	const coll = "operate/ttl"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	if _, err := tc.client.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	nodes := make([]*Node, 0, len(tc.nodes))
	for _, n := range tc.nodes {
		if n != nil {
			nodes = append(nodes, n)
		}
	}
	if len(nodes) != 3 {
		t.Fatalf("expected 3 live nodes, got %d", len(nodes))
	}

	// One pinned clock per node. base is a fixed instant so nothing below depends
	// on what time.Now happened to return.
	const base int64 = 1_700_000_000_000
	const keyTTLMs int64 = 50_000
	const deadline = base + keyTTLMs
	clocks := make([]*atomic.Int64, len(nodes))
	for i, n := range nodes {
		clk := &atomic.Int64{}
		clk.Store(base)
		clocks[i] = clk
		setNodeVectorClock(t, n, func() int64 { return clk.Load() })
	}
	// Re-pin after the collection exists so a store that built it lazily still
	// inherits the override (SetNowFunc propagates to resident indexes).
	pinAll := func(ms int64) {
		for i, n := range nodes {
			clocks[i].Store(ms)
			setNodeVectorClock(t, n, func() int64 { return clocks[i].Load() })
		}
	}
	pinAll(base)

	const id = uint64(7)
	if _, err := tc.client.Call(ctx, "vector_upsert",
		ops.EncodeVectorUpsertArgs(physCol, id, []float32{1, 0, 0, 0}, "", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	addRC := &wire.OperateArgs{
		Create: wire.OperateCreateDynamic,
		Ops:    []wire.OperateOp{{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: operateFieldPath("rc"), A: 1}},
	}
	operate := func(t *testing.T) {
		t.Helper()
		args, err := wire.EncodeVectorOperateArgs(physCol, id, "session", addRC, 0, false)
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
	}

	// waitForVersion gates on the DATA reaching every replica: raft's applied_index
	// advances at ENQUEUE, so it is only a proxy for the apply having run. The
	// version is read rather than the record because a replica whose clock is past
	// the deadline hides the key from its own read.
	waitForVersion := func(t *testing.T, want uint64) {
		t.Helper()
		deadlineAt := time.Now().Add(20 * time.Second)
		for {
			ready := true
			for _, n := range nodes {
				if _, v := readRecordFromNode(t, n, physCol, id, "session"); v != want {
					ready = false
					break
				}
			}
			if ready {
				return
			}
			if time.Now().After(deadlineAt) {
				t.Fatalf("timed out waiting for every replica to reach version %d", want)
			}
			time.Sleep(20 * time.Millisecond)
		}
	}

	// 1. Create the record (rc=1). No deadline yet, so this step is clock-free.
	operate(t)
	waitForVersion(t, 2) // upsert=1, operate=2

	// 2. Put a per-key deadline on the record key. An EMPTY patch with a keyTTL
	//    only re-deadlines a key already in the payload (setPayloadBody), and with
	//    every clock still at base every replica computes the SAME deadline.
	if _, err := tc.client.Call(ctx, "vector_set_payload",
		wire.EncodeSetPayloadArgsOpts(physCol, id, nil, map[string]int64{"session": keyTTLMs})); err != nil {
		t.Fatalf("vector_set_payload: %v", err)
	}
	waitForVersion(t, 3)

	// 3. Skew the replicas' clocks ACROSS the deadline: two past it, one short of
	//    it. This is the state the divergence needs.
	clocks[0].Store(deadline + 10_000)
	clocks[1].Store(base + 10_000)
	clocks[2].Store(deadline + 30_000)

	// 4. The operate every replica now applies with its own clock.
	operate(t)
	waitForVersion(t, 4)

	// 5. Read every replica at ONE common clock, short of the deadline, so the key
	//    is visible everywhere and the comparison is of real stored bytes.
	pinAll(base)
	recs := make([][]byte, 0, len(nodes))
	for _, n := range nodes {
		rec, _ := readRecordFromNode(t, n, physCol, id, "session")
		if len(rec) == 0 {
			t.Fatalf("a replica holds no record under the payload key at the common probe clock — "+
				"the mutation dropped the key or its deadline (node %p)", n)
		}
		recs = append(recs, rec)
	}
	for i := 1; i < len(recs); i++ {
		if !bytes.Equal(recs[i], recs[0]) {
			t.Fatalf("replica %d stored DIFFERENT record bytes than replica 0 — an unstamped apply judged the "+
				"payload key's deadline against its OWN wall clock, so one replica recreated the record while "+
				"another incremented it\n replica0=%x\n replica%d=%x", i, recs[0], i, recs[i])
		}
	}

	// Positive control: the agreed bytes are the INCREMENTED record (rc=2), not a
	// fresh rc=1 that every replica recreated identically because every clock
	// happened to be past the deadline.
	rec, err := wire.DecodeRecord(recs[0])
	if err != nil {
		t.Fatalf("DecodeRecord: %v", err)
	}
	var rc *wire.Cell
	for i := range rec.Fields {
		if rec.Fields[i].Name == "rc" {
			rc = &rec.Fields[i].Cell
		}
	}
	if rc == nil {
		t.Fatalf("record carries no field %q: %+v", "rc", rec.Fields)
	}
	if rc.U != 2 {
		t.Fatalf("rc = %d, want 2 — the second operate did not increment the EXISTING record, so the "+
			"deadline-passed key was read as absent", rc.U)
	}
}
