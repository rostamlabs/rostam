// SPDX-License-Identifier: Apache-2.0

package rostam

// Cluster-plumbing tests for vector_operate: the three Store implementations
// (embedded, networked, direct), the partitioned fan-out, and the reshard
// refusal.
//
// What these tests pin that no lower layer can:
//
//   - the CAS precondition survives EVERY Store path. The vector_set_payload
//     siblings drop it on three of their four paths (networkedStore encodes with
//     the non-CAS encoder, directStore takes `_ ...WriteOpts`, fanSetPayload
//     decodes with the non-CAS decoder), so "mirror set_payload" is exactly the
//     wrong instinct here and each path gets its own regression guard.
//   - an operate on a partitioned collection reaches the OWNING partition and
//     only it.
//   - an operate is REFUSED while a reshard is dual-writing: applying a
//     non-idempotent op-list to both generations would double-count every ADD.

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// operateFieldPath addresses a dynamic-mode record field by name.
func operateFieldPath(name string) wire.OperatePath {
	return wire.OperatePath{Kind: wire.OperatePathField, Field: wire.OperateSeg{ByName: true, Name: name}}
}

// bumpRC is the canonical op-list these tests drive: ADD 1 to the by-name U64
// field "rc", creating a dynamic-mode record when the payload key is empty, and
// return the resulting value. Two calls therefore leave rc == 2.
func bumpRC() *wire.OperateArgs {
	return &wire.OperateArgs{
		Create: wire.OperateCreateDynamic,
		Ops: []wire.OperateOp{
			{Opcode: wire.OperateOpADD, Type: wire.OperateTypeU64, Path: operateFieldPath("rc"), A: 1},
		},
		Rets: []wire.OperateRet{{Mode: wire.OperateRetValue, Path: operateFieldPath("rc")}},
	}
}

// retRC decodes the single returned cell of a bumpRC result as a uint64.
func retRC(t *testing.T, res *wire.OperateResult) uint64 {
	t.Helper()
	if res == nil {
		t.Fatal("operate result is nil")
	}
	if res.Status != wire.OperateStatusOK {
		t.Fatalf("operate status = %d, want OK", res.Status)
	}
	if len(res.Values) != 1 {
		t.Fatalf("operate returned %d values, want 1", len(res.Values))
	}
	c, _, err := wire.DecodeTaggedCell(res.Values[0])
	if err != nil {
		t.Fatalf("DecodeTaggedCell: %v", err)
	}
	return c.U
}

// pointVersion reads a point's current CAS version through the Store API (the
// batch-get row carries it on every backend), so a CAS test can supply the
// RIGHT expected version rather than guessing at the engine's bump schedule.
func pointVersion(t *testing.T, s Store, coll string, id uint64) uint64 {
	t.Helper()
	pts, _, err := s.VectorGetBatch(context.Background(), coll, []uint64{id}, false, false)
	if err != nil {
		t.Fatalf("VectorGetBatch %s/%d: %v", coll, id, err)
	}
	if len(pts) != 1 {
		t.Fatalf("VectorGetBatch %s/%d returned %d points, want 1", coll, id, len(pts))
	}
	return pts[0].Version
}

// isCASConflict matches vector.ErrVersionConflict by identity AND by text: the
// sentinel's identity does not survive the networked transport (the server
// stringifies it onto the wire), so the text arm is what covers that path.
func isCASConflict(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, vector.ErrVersionConflict) {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "version") && (strings.Contains(s, "conflict") || strings.Contains(s, "mismatch"))
}

// physRecordAbsent reports whether the payload key holds no record on the given
// PHYSICAL partition (the reshard test's "nothing was written" assertion).
func physRecordAbsent(t *testing.T, ee *embedded, phys string, id uint64, key string) bool {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_get", ops.EncodeVectorGetArgs(phys, id, getFlags(false, true)))
	if err != nil {
		t.Fatalf("vector_get %q id=%d: %v", phys, id, err)
	}
	found, _, meta, _, _, err := ops.DecodeVectorGetResult(body)
	if err != nil {
		t.Fatalf("decode vector_get %q: %v", phys, err)
	}
	if !found {
		return true
	}
	_, ok := meta[key]
	return !ok
}

// seedOperateCollection creates a dense collection (P partitions; P<=1 =
// unpartitioned) and inserts ids 1..n so every operate has a live point.
func seedOperateCollection(t *testing.T, s Store, coll string, P, n int) {
	t.Helper()
	ctx := context.Background()
	must(t, s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Partitions: P,
	}))
	for id := uint64(1); id <= uint64(n); id++ {
		must(t, s.VectorInsert(ctx, coll, id, []float32{float32(id), 0, 0, 0}))
	}
}

// TestVectorOperateUnpartitionedPassThrough is the baseline: on an
// UNPARTITIONED embedded collection two operate calls through Store.VectorOperate
// create the record and increment it, returning 1 then 2.
func TestVectorOperateUnpartitionedPassThrough(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	seedOperateCollection(t, s, "docs", 0, 1)

	found, res, _, err := s.VectorOperate(ctx, "docs", 1, "session", bumpRC())
	must(t, err)
	if !found {
		t.Fatal("first operate: found=false, want true (the point exists)")
	}
	if got := retRC(t, res); got != 1 {
		t.Fatalf("first operate rc = %d, want 1", got)
	}

	found, res, _, err = s.VectorOperate(ctx, "docs", 1, "session", bumpRC())
	must(t, err)
	if !found {
		t.Fatal("second operate: found=false")
	}
	if got := retRC(t, res); got != 2 {
		t.Fatalf("second operate rc = %d, want 2", got)
	}
}

// TestVectorOperatePartitionedRoutesToOwningPartition drives 40 points on a P=4
// collection: each gets a record, each is incremented twice, each reads back
// rc == 2, and each point lives on ONLY its owning physical partition (proving
// the operate reached the partition the id routes to, not some other one).
func TestVectorOperatePartitionedRoutesToOwningPartition(t *testing.T) {
	const (
		coll = "parted"
		P    = 4
		N    = 40
	)
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ee := s.(*embedded)
	ctx := context.Background()
	seedOperateCollection(t, s, coll, P, N)

	for id := uint64(1); id <= N; id++ {
		for round := 1; round <= 2; round++ {
			found, res, _, err := s.VectorOperate(ctx, coll, id, "session", bumpRC())
			if err != nil {
				t.Fatalf("operate id=%d round=%d: %v", id, round, err)
			}
			if !found {
				t.Fatalf("operate id=%d round=%d: found=false", id, round)
			}
			if got := retRC(t, res); got != uint64(round) { //nolint:gosec // round is 1 or 2
				t.Fatalf("operate id=%d round=%d: rc = %d, want %d", id, round, got, round)
			}
		}
	}

	// Every point sits on exactly one physical partition: its owning one.
	for id := uint64(1); id <= N; id++ {
		own := ops.PartitionOf(id, P)
		for p := 0; p < P; p++ {
			phys := string(ops.PartitionKeyGen(coll, 0, p))
			got := physVecExists(t, ee, phys, id)
			if want := p == own; got != want {
				t.Fatalf("id=%d present in %q = %v, want %v (owning partition is %d)", id, phys, got, want, own)
			}
		}
		// And its record is on the owning partition, holding rc == 2.
		phys := string(ops.PartitionKeyGen(coll, 0, own))
		if physRecordAbsent(t, ee, phys, id, "session") {
			t.Fatalf("id=%d: no record under %q on the owning partition %q", id, "session", phys)
		}
	}
}

// TestVectorOperateCASOverTheNetworkedPath is the networkedStore regression
// guard: networkedStore.VectorSetPayload encodes with the NON-CAS encoder and
// silently drops the precondition. A vector_operate must not — the wrong version
// must conflict and the right version must apply.
func TestVectorOperateCASOverTheNetworkedPath(t *testing.T) {
	reg := ops.NewRegistry()
	must(t, ops.RegisterBuiltins(reg))
	srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{Ops: reg})
	must(t, err)
	t.Cleanup(func() { _ = srv.Close() })

	s, err := NewClient(ClientConfig{Servers: []string{srv.Addr()}, Ops: reg})
	must(t, err)
	t.Cleanup(func() { _ = s.Close() })

	seedOperateCollection(t, s, "docs", 0, 1)
	assertOperateCAS(t, s, "docs", 1)
}

// TestVectorOperateCASOverTheDirectPath is the directStore regression guard:
// directStore.VectorSetPayload takes `_ ...WriteOpts` and throws the CAS
// precondition away outright. On a single-node direct store there is no other
// guard on a write, so dropping it is the whole failure.
func TestVectorOperateCASOverTheDirectPath(t *testing.T) {
	s := newSingleDirect(t)
	seedOperateCollection(t, s, "docs", 0, 1)
	assertOperateCAS(t, s, "docs", 1)
}

// assertOperateCAS drives the shared CAS contract against any Store: a wrong
// expected version conflicts and changes nothing; the right one applies.
func assertOperateCAS(t *testing.T, s Store, coll string, id uint64) {
	t.Helper()
	ctx := context.Background()

	// Seed a record so the CAS attempts have something to fail against.
	found, res, _, err := s.VectorOperate(ctx, coll, id, "session", bumpRC())
	must(t, err)
	if !found || retRC(t, res) != 1 {
		t.Fatalf("seed operate: found=%v res=%+v", found, res)
	}

	ver := pointVersion(t, s, coll, id)
	if ver == 0 {
		t.Fatal("point version is 0 — this backend does not carry versions, the CAS test cannot run")
	}

	// Wrong version ⇒ conflict, nothing applied.
	wrong := ver + 999
	if _, _, _, err := s.VectorOperate(ctx, coll, id, "session", bumpRC(), WriteOpts{ExpectedVersion: &wrong}); !isCASConflict(err) {
		t.Fatalf("operate with the WRONG expected version = %v, want a version conflict (the CAS guard was dropped)", err)
	}
	if _, res, _, err := s.VectorOperate(ctx, coll, id, "session", bumpRC()); err != nil || retRC(t, res) != 2 {
		t.Fatalf("after the refused CAS the counter should read 2: err=%v res=%+v", err, res)
	}

	// Right version ⇒ applied.
	right := pointVersion(t, s, coll, id)
	found, res, _, err = s.VectorOperate(ctx, coll, id, "session", bumpRC(), WriteOpts{ExpectedVersion: &right})
	must(t, err)
	if !found || retRC(t, res) != 3 {
		t.Fatalf("operate with the RIGHT expected version: found=%v res=%+v, want rc=3", found, res)
	}
}

// TestVectorOperateCASSurvivesFanOut is the fan-out regression guard:
// fanSetPayload decodes with the NON-CAS decoder, so a partitioned set_payload
// loses the guard entirely. fanOperate must thread it through.
func TestVectorOperateCASSurvivesFanOut(t *testing.T) {
	const (
		coll = "parted"
		P    = 4
		id   = uint64(7)
	)
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	seedOperateCollection(t, s, coll, P, 10)

	call := func(t *testing.T, exp uint64, hasExp bool) (bool, *wire.OperateResult, error) {
		t.Helper()
		args, err := wire.EncodeVectorOperateArgs(coll, id, "session", bumpRC(), exp, hasExp)
		if err != nil {
			t.Fatalf("EncodeVectorOperateArgs: %v", err)
		}
		body, cerr := fan.Call("vector_operate", args)
		if cerr != nil {
			return false, nil, cerr
		}
		found, res, _, derr := ops.DecodeVectorOperateResult(body)
		return found, res, derr
	}

	// Seed the record through the fan-out itself.
	found, res, err := call(t, 0, false)
	must(t, err)
	if !found || retRC(t, res) != 1 {
		t.Fatalf("seed through fan-out: found=%v res=%+v", found, res)
	}

	ver := pointVersion(t, s, coll, id)
	if _, _, err := call(t, ver+999, true); !isCASConflict(err) {
		t.Fatalf("fan-out operate with the WRONG expected version = %v, want a version conflict "+
			"(fanOperate dropped the CAS guard, exactly like fanSetPayload)", err)
	}
	found, res, err = call(t, ver, true)
	must(t, err)
	if !found || retRC(t, res) != 2 {
		t.Fatalf("fan-out operate with the RIGHT expected version: found=%v res=%+v, want rc=2", found, res)
	}
}

// TestVectorOperateRefusedDuringReshard pins the ruling: an operate is refused
// while a reshard is dual-writing, because applying a non-idempotent op-list to
// both generations double-counts every ADD and the disagreement survives
// cutover. The refusal must write NOTHING to either generation, and clearing the
// reshard state must make the identical call succeed.
func TestVectorOperateRefusedDuringReshard(t *testing.T) {
	const (
		coll       = "dw"
		oldP, newP = 4, 8
		id         = uint64(7)
	)
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	// The insert dual-writes, so the point exists in BOTH generations.
	must(t, ee.VectorInsert(ctx, coll, id, []float32{1, 0, 0, 0}))

	_, _, _, err := ee.VectorOperate(ctx, coll, id, "session", bumpRC())
	if !errors.Is(err, ErrOperateDuringReshard) {
		t.Fatalf("operate during a reshard = %v, want ErrOperateDuringReshard", err)
	}
	for _, phys := range []string{oldPhys, newPhys} {
		if !physRecordAbsent(t, ee, phys, id, "session") {
			t.Fatalf("the refused operate wrote a record to %q", phys)
		}
	}

	// Clearing the reshard state makes the identical call succeed.
	must(t, ee.catalog.SetReshardState(coll, ReshardState{Status: 0}))
	found, res, _, err := ee.VectorOperate(ctx, coll, id, "session", bumpRC())
	must(t, err)
	if !found || retRC(t, res) != 1 {
		t.Fatalf("operate after the reshard cleared: found=%v res=%+v, want rc=1", found, res)
	}
}

// TestVectorOperateMissingPointFlagThroughFanOut: an absent point is the
// not-found FLAG, never an error — the same contract set_payload has.
func TestVectorOperateMissingPointFlagThroughFanOut(t *testing.T) {
	const (
		coll = "parted"
		P    = 4
	)
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	seedOperateCollection(t, s, coll, P, 4)

	args, err := wire.EncodeVectorOperateArgs(coll, 9999, "session", bumpRC(), 0, false)
	must(t, err)
	body, err := fan.Call("vector_operate", args)
	if err != nil {
		t.Fatalf("operate on an absent point returned an error: %v (want the found=false FLAG)", err)
	}
	found, _, _, err := ops.DecodeVectorOperateResult(body)
	must(t, err)
	if found {
		t.Fatal("operate on an absent point: found=true, want false")
	}

	// The Store path agrees.
	found, _, _, err = s.VectorOperate(context.Background(), coll, 9999, "session", bumpRC())
	must(t, err)
	if found {
		t.Fatal("Store.VectorOperate on an absent point: found=true, want false")
	}
}

// TestVectorOperateAliasResolves drives the operate family through an ALIAS on a
// partitioned collection — the dataPlaneAliasOps entry. Without it the alias name
// is never rewritten, the route reports unpartitioned, and the op hits the empty
// logical collection instead of the owning partition.
func TestVectorOperateAliasResolves(t *testing.T) {
	const (
		real = "real"
		prod = "prod"
		P    = 4
		id   = uint64(5)
	)
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	ctx := context.Background()
	seedOperateCollection(t, s, real, P, 10)
	must(t, emb.CreateAlias(ctx, prod, real))

	args, err := wire.EncodeVectorOperateArgs(prod, id, "session", bumpRC(), 0, false)
	must(t, err)
	body, err := fan.Call("vector_operate", args)
	must(t, err)
	found, res, _, err := ops.DecodeVectorOperateResult(body)
	must(t, err)
	if !found || retRC(t, res) != 1 {
		t.Fatalf("operate via alias through the dispatcher: found=%v res=%+v, want rc=1", found, res)
	}

	// The record landed on the REAL collection's owning partition.
	phys := string(ops.PartitionKeyGen(real, 0, ops.PartitionOf(id, P)))
	if physRecordAbsent(t, emb, phys, id, "session") {
		t.Fatalf("operate via alias did not write to the real collection's owning partition %q", phys)
	}

	// The Store path resolves the alias too, and sees the SAME record.
	found, res, _, err = s.VectorOperate(ctx, prod, id, "session", bumpRC())
	must(t, err)
	if !found || retRC(t, res) != 2 {
		t.Fatalf("Store.VectorOperate via alias: found=%v res=%+v, want rc=2 (same record)", found, res)
	}
}

// TestVectorOperateNamedAndMVThroughStoreAndFanOut covers the other two
// families' Store methods and fan handlers on PARTITIONED collections, so the
// three fan-out rows are each exercised rather than only the dense one.
func TestVectorOperateNamedAndMVThroughStoreAndFanOut(t *testing.T) {
	const (
		P  = 4
		id = uint64(3)
	)
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	ctx := context.Background()

	must(t, s.VectorNamedCreateCollection(ctx, "named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
	}, P))
	must(t, s.VectorNamedInsert(ctx, "named", id, map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0))

	must(t, s.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Partitions: P,
	}))
	must(t, s.VectorMVAdd(ctx, "mv", id, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, nil))

	for _, fam := range []struct {
		op    string
		coll  string
		store func(ctx context.Context, a *wire.OperateArgs, opts ...WriteOpts) (bool, *wire.OperateResult, uint64, error)
	}{
		{
			op: "vector_named_operate", coll: "named",
			store: func(ctx context.Context, a *wire.OperateArgs, opts ...WriteOpts) (bool, *wire.OperateResult, uint64, error) {
				return s.VectorNamedOperate(ctx, "named", id, "session", a, opts...)
			},
		},
		{
			op: "vector_mv_operate", coll: "mv",
			store: func(ctx context.Context, a *wire.OperateArgs, opts ...WriteOpts) (bool, *wire.OperateResult, uint64, error) {
				return s.VectorMVOperate(ctx, "mv", id, "session", a, opts...)
			},
		},
	} {
		t.Run(fam.op, func(t *testing.T) {
			// Through the Store method.
			found, res, _, err := fam.store(ctx, bumpRC())
			must(t, err)
			if !found || retRC(t, res) != 1 {
				t.Fatalf("Store %s: found=%v res=%+v, want rc=1", fam.op, found, res)
			}
			// Through the fan-out dispatcher, onto the SAME record.
			args, err := wire.EncodeVectorOperateArgs(fam.coll, id, "session", bumpRC(), 0, false)
			must(t, err)
			body, err := fan.Call(fam.op, args)
			must(t, err)
			found, res, _, err = ops.DecodeVectorOperateResult(body)
			must(t, err)
			if !found || retRC(t, res) != 2 {
				t.Fatalf("fan %s: found=%v res=%+v, want rc=2", fam.op, found, res)
			}
			// An absent id is the not-found flag on both families too.
			missing, err := wire.EncodeVectorOperateArgs(fam.coll, 9999, "session", bumpRC(), 0, false)
			must(t, err)
			body, err = fan.Call(fam.op, missing)
			must(t, err)
			found, _, _, err = ops.DecodeVectorOperateResult(body)
			must(t, err)
			if found {
				t.Fatalf("fan %s on an absent id: found=true, want false", fam.op)
			}
		})
	}
}
