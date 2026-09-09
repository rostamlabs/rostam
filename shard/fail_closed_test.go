// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"fmt"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// newRejectCache builds a single-shard, single-page cache under PolicyRejectWrites
// so a small number of puts drives it to capacity and any further put returns
// cache.ErrFull — the canonical NON-DETERMINISTIC apply error this Phase-A gate
// must fail closed on.
func newRejectCache(t *testing.T) *cache.Cache {
	t.Helper()
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.PageSize = 1 << 20          // 1 MiB (minimum) — one page
	cc.MaxMemoryPerShard = 1 << 20 // exactly one page ⇒ MaxPagesPerShard=1
	cc.AtCapPolicy = cache.PolicyRejectWrites
	c, err := cache.New(cc)
	if err != nil {
		t.Fatalf("new reject cache: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// fillToCap puts distinct keys until the cache rejects with cache.ErrFull.
func fillToCap(t *testing.T, c *cache.Cache) {
	t.Helper()
	val := make([]byte, 4096)
	for i := 0; i < 1_000_000; i++ {
		if err := c.Put([]byte(fmt.Sprintf("fill-%d", i)), val, 0); err != nil {
			if errors.Is(err, cache.ErrFull) {
				return
			}
			t.Fatalf("fill put %d: %v", i, err)
		}
	}
	t.Fatal("cache never reached capacity")
}

// newGatedFSM builds an fsm over cache c with the fail-closed gate wired: replicated
// controls f.isReplicated and onFatal is the injected halt hook (so tests observe
// the fail-closed decision without os.Exit'ing the test binary).
func newGatedFSM(t *testing.T, c *cache.Cache, replicated bool, onFatal func(error)) *fsm {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	f := newFSM(c, reg, false, nil, nil)
	f.isReplicated = func() bool { return replicated }
	f.onFatalApply = onFatal
	return f
}

// bigPut encodes a put whose value (8 KiB) is at least as large as fillToCap's, so
// a shard already at capacity for that size has no leftover gap that could admit
// it — the write reliably hits cache.ErrFull rather than sneaking into slack.
func bigPut(key string) []byte {
	return EncodeLogEntry("put", ops.EncodePutArgs([]byte(key), make([]byte, 8192), 0))
}

// TestClassifyApplyErr pins the INCLUSION-list classification: only the known
// non-deterministic cache sentinels are fatal; everything else (nil, deterministic
// op/infra errors, wrapped forms) advances.
func TestClassifyApplyErr(t *testing.T) {
	fatal := []struct {
		name string
		err  error
	}{
		{"ErrFull", cache.ErrFull},
		{"ErrCannotEvict", cache.ErrCannotEvict},
		{"wrapped ErrFull", fmt.Errorf("apply at index 9: %w", cache.ErrFull)},
		{"wrapped ErrCannotEvict", fmt.Errorf("x: %w", cache.ErrCannotEvict)},
		// Version-skew is fatal even though it rides inside errPBApplyDecode (a peer
		// on a newer binary decodes+applies the same entry this node cannot parse).
		{"ErrLogEntryVersion", ErrLogEntryVersion},
		{"multi-%w decode+version (as applyEntryData wraps it)", fmt.Errorf("%w: %w", errPBApplyDecode, ErrLogEntryVersion)},
		// An unregistered op at APPLY time means this replica's ops registry differs
		// from the proposer's (cluster.Node.Call Looks the op up BEFORE proposing, so
		// the entry could not exist otherwise). Peers that have the op execute it
		// while this one would skip — the divergence condition. Ops are dynamically
		// registrable (__register_wasm__), so "same binary ⇒ same registry" is false.
		{"ErrOpNotRegistered", ErrOpNotRegistered},
		{"wrapped ErrOpNotRegistered", fmt.Errorf("apply at index 4: %w", ErrOpNotRegistered)},
		// A read-only op at APPLY time means the same thing one layer along: the
		// proposer's registry said the op was writable (Store.Call never proposes a
		// read-only op) and this replica's says it is not. Kind is a wire-controlled
		// field of a WASM registration installed into the node-wide registry, so
		// "the op's kind is a structural constant" is false for exactly the ops
		// ErrOpNotRegistered's premise died on.
		{"errPBApplyReadOnly", errPBApplyReadOnly},
		{"wrapped errPBApplyReadOnly (as applyEntryData wraps it)", fmt.Errorf("%w: %q", errPBApplyReadOnly, "wasm_op")},
		// A WASM op with no version bound to THIS shard group, on a committed entry
		// from that group's own log. The propose-time route gate puts a registration
		// below every invocation in the same group's log, and both catch-up routes
		// reconstruct the binding from it, so a miss means this replica's state
		// disagrees with the proposer's: peers hold the binding and WILL execute the
		// entry. Advancing over it is the same execute-on-peers / skip-here
		// divergence as ErrOpNotRegistered.
		{"ops.ErrWASMNoGroupBinding", ops.ErrWASMNoGroupBinding},
		{"wrapped ops.ErrWASMNoGroupBinding (as the resolver wraps it)", fmt.Errorf("%w: op %q shard group %d", ops.ErrWASMNoGroupBinding, "udf", 3)},
		// A flush whose durability watermark (sidecar) could not be persisted on THIS
		// node. A local disk fault a peer need not share, so advancing over it would
		// resurrect pre-flush keys on a later failover — fatal, like ErrFull. The
		// wrapped case mirrors how cache.shard.flush wraps the sidecar I/O cause.
		{"cache.ErrFlushNotDurable", cache.ErrFlushNotDurable},
		{"wrapped cache.ErrFlushNotDurable (as cache.flush wraps the I/O cause)", fmt.Errorf("%w: %w", cache.ErrFlushNotDurable, errors.New("fsync flush sidecar dir: read-only file system"))},
	}
	for _, tc := range fatal {
		if got := classifyApplyErr(tc.err); got != classFatal {
			t.Errorf("classifyApplyErr(%s) = %v, want classFatal", tc.name, got)
		}
	}
	advance := []struct {
		name string
		err  error
	}{
		{"nil", nil},
		{"ErrShortArgs (deterministic op error)", ops.ErrShortArgs},
		{"decode", fmt.Errorf("%w: bad", errPBApplyDecode)},
		{"cache.ErrNotFound", cache.ErrNotFound},
		{"arbitrary", errors.New("some other error")},
	}
	for _, tc := range advance {
		if got := classifyApplyErr(tc.err); got != classAdvance {
			t.Errorf("classifyApplyErr(%s) = %v, want classAdvance", tc.name, got)
		}
	}
}

// TestFSMApplyErrFullReplicatedFailsClosed proves the core Phase-A fix: on a
// replicated shard, an ErrFull apply fires onFatalApply and does NOT advance the
// applied index. The companion roomy replica applies the SAME committed entry
// successfully — demonstrating exactly the non-deterministic divergence the gate
// prevents (advancing on the full node would silently disagree with the roomy one).
func TestFSMApplyErrFullReplicatedFailsClosed(t *testing.T) {
	full := newRejectCache(t)
	fillToCap(t, full)

	var fired error
	ffull := newGatedFSM(t, full, true /* replicated */, func(err error) { fired = err })

	entry := bigPut("new-key")
	res := ffull.Apply(&hraft.Log{Index: 7, Type: hraft.LogCommand, Data: entry})
	resp, ok := res.(*ApplyResponse)
	if !ok {
		t.Fatalf("Apply returned %T", res)
	}
	if !errors.Is(resp.Err, cache.ErrFull) {
		t.Fatalf("apply err = %v, want ErrFull", resp.Err)
	}
	if fired == nil {
		t.Fatal("onFatalApply did not fire on a replicated ErrFull")
	}
	if !errors.Is(fired, cache.ErrFull) {
		t.Fatalf("fatal err must wrap ErrFull, got %v", fired)
	}
	if idx := ffull.AppliedIndex(); idx != 0 {
		t.Fatalf("applied index advanced to %d on a fatal apply; must stay 0 (fail-closed)", idx)
	}

	// The SAME committed entry applies cleanly on a roomy replica — the divergence
	// the gate exists to catch.
	roomy := newRejectCache(t)
	froomy := newGatedFSM(t, roomy, true, func(err error) { t.Fatalf("roomy replica must not fire fatal: %v", err) })
	rres := froomy.Apply(&hraft.Log{Index: 7, Type: hraft.LogCommand, Data: entry})
	if rresp := rres.(*ApplyResponse); rresp.Err != nil {
		t.Fatalf("roomy replica apply err = %v, want nil", rresp.Err)
	}
	if idx := froomy.AppliedIndex(); idx != 7 {
		t.Fatalf("roomy replica applied index = %d, want 7", idx)
	}
}

// TestFSMApplyErrFullSingleNodeAdvances is the single-node regression guard: with
// isReplicated()==false there are no peers to diverge from, so ErrFull must advance
// the index and surface normally — never halt (that would be a pure availability
// regression).
func TestFSMApplyErrFullSingleNodeAdvances(t *testing.T) {
	full := newRejectCache(t)
	fillToCap(t, full)

	fatalCalled := false
	f := newGatedFSM(t, full, false /* single node */, func(err error) { fatalCalled = true })

	res := f.Apply(&hraft.Log{Index: 3, Type: hraft.LogCommand, Data: bigPut("k")})
	resp := res.(*ApplyResponse)
	if !errors.Is(resp.Err, cache.ErrFull) {
		t.Fatalf("apply err = %v, want ErrFull surfaced", resp.Err)
	}
	if fatalCalled {
		t.Fatal("single-node ErrFull must NOT halt")
	}
	if idx := f.AppliedIndex(); idx != 3 {
		t.Fatalf("single-node applied index = %d, want 3 (advances past a deterministic-on-one-node error)", idx)
	}
}

// TestFSMApplyDeterministicErrorReplicatedAdvances proves a DETERMINISTIC error on
// a replicated shard still advances and surfaces — no halt — since it fails
// identically on every replica and cannot cause divergence.
//
// The probe is a truncated argument buffer handed to a compile-time builtin
// WRITE op: "put" is registered by this binary, its handler is a constant of
// this binary, and ops.ErrShortArgs is a pure function of the entry bytes, so
// every replica rejects the same committed entry the same way.
//
// It has now replaced TWO earlier probes, both for the same underlying reason —
// the thing they tested turned out to be per-node MUTABLE state rather than a
// constant of the binary. First an unregistered op name (ErrOpNotRegistered
// became fatal once ops became dynamically registrable), then a read-only op
// driven through the write path (errPBApplyReadOnly became fatal once a WASM
// registration could change an op's Kind at runtime — see
// TestFSMApplyReadOnlyKindReplicatedFailsClosed). What is left is a genuinely
// compile-time property: a builtin handler's own argument validation.
func TestFSMApplyDeterministicErrorReplicatedAdvances(t *testing.T) {
	c := newRejectCache(t) // roomy; the error is the arg length, not capacity

	fatalCalled := false
	f := newGatedFSM(t, c, true /* replicated */, func(err error) { fatalCalled = true })

	// A 1-byte args buffer cannot carry even the u16 key length "put" requires.
	entry := EncodeLogEntry("put", []byte{0x00})
	res := f.Apply(&hraft.Log{Index: 5, Type: hraft.LogCommand, Data: entry})
	resp := res.(*ApplyResponse)
	if !errors.Is(resp.Err, ops.ErrShortArgs) {
		t.Fatalf("apply err = %v, want ops.ErrShortArgs", resp.Err)
	}
	if fatalCalled {
		t.Fatal("deterministic error must NOT halt on a replicated shard")
	}
	if idx := f.AppliedIndex(); idx != 5 {
		t.Fatalf("applied index = %d, want 5 (deterministic error advances)", idx)
	}
}

// TestFSMApplyReadOnlyKindReplicatedFailsClosed is the gate for the SECOND
// registry-disagreement channel: a committed entry naming an op whose kind THIS
// replica records as OpReadOnly must HALT, not skip.
//
// It is the same divergence as the unknown-op case, one field along.
// shard.Store.Call never proposes a read-only op (it serves it locally), so the
// entry can only exist because the PROPOSER's registry classified the op as
// writable. A replica that classifies it as read-only therefore disagrees with
// the node that produced the entry — and with every peer that shares the
// proposer's view, all of which EXECUTE it.
//
// The disagreement is reachable without any version skew of the BINARY: Kind is
// a wire-controlled field of a WASM registration, a registration is broadcast to
// shard groups that commit at independent times, and cluster's
// RegisterOrReplaceModule installs the new Kind into the node-wide registry on
// whichever node applies it first. Register X as OpReadWrite, then re-register it
// at a higher Epoch as OpReadOnly, and every node that has not yet applied the
// second registration keeps proposing invocations that the nodes which HAVE
// applied it would, under classAdvance, silently skip.
//
// The applied index must NOT move: the entry replays once the registries agree.
func TestFSMApplyReadOnlyKindReplicatedFailsClosed(t *testing.T) {
	c := newRejectCache(t) // roomy; the error is the op kind, not capacity

	var fatalErr error
	f := newGatedFSM(t, c, true /* replicated */, func(err error) { fatalErr = err })

	// "get" stands in for the post-update registry state: an op present in this
	// replica's registry as OpReadOnly, reached through the write path.
	res := f.Apply(&hraft.Log{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("get", ops.EncodeKeyArgs([]byte("k")))})
	resp := res.(*ApplyResponse)
	if !errors.Is(resp.Err, errPBApplyReadOnly) {
		t.Fatalf("apply err = %v, want errPBApplyReadOnly", resp.Err)
	}
	if fatalErr == nil {
		t.Fatal("a read-only-kind op on a REPLICATED shard must fail closed (onFatalApply), not skip-advance: peers whose registry calls it writable execute this entry")
	}
	if !errors.Is(fatalErr, errPBApplyReadOnly) {
		t.Errorf("fatal error = %v, want it to wrap errPBApplyReadOnly", fatalErr)
	}
	if idx := f.AppliedIndex(); idx != 0 {
		t.Fatalf("applied index = %d, want 0: the fatal entry must NOT be marked applied (it has to replay)", idx)
	}
}

// TestFSMApplyUnknownOpReplicatedFailsClosed is the gate for the core fix: a
// committed entry naming an op this replica's registry does not hold must HALT,
// not skip.
//
// Why that is divergence and not a benign no-op: cluster.Node.Call performs a
// registry Lookup and returns ErrUnknownOp BEFORE proposing, so the entry could
// only have entered the log because the PROPOSER held the op. Every replica whose
// registry has it therefore EXECUTES this entry. Skip-advancing here would leave
// this replica silently missing that mutation forever while still reporting the
// index as applied. Ops are dynamically registrable (cluster's __register_wasm__
// mutates the node-wide ops.Registry at apply time), so registries genuinely do
// differ across in-sync replicas.
//
// The applied index must NOT move: the entry has to replay once the registry
// catches up.
func TestFSMApplyUnknownOpReplicatedFailsClosed(t *testing.T) {
	c := newRejectCache(t) // roomy; the error is the unknown op, not capacity

	var fatalErr error
	f := newGatedFSM(t, c, true /* replicated */, func(err error) { fatalErr = err })

	res := f.Apply(&hraft.Log{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("nonexistent-op", nil)})
	resp := res.(*ApplyResponse)
	if !errors.Is(resp.Err, ErrOpNotRegistered) {
		t.Fatalf("apply err = %v, want ErrOpNotRegistered", resp.Err)
	}
	if fatalErr == nil {
		t.Fatal("an unregistered op on a REPLICATED shard must fail closed (onFatalApply), not skip-advance")
	}
	if !errors.Is(fatalErr, ErrOpNotRegistered) {
		t.Errorf("fatal error = %v, want it to wrap ErrOpNotRegistered", fatalErr)
	}
	if idx := f.AppliedIndex(); idx != 0 {
		t.Fatalf("applied index = %d, want 0: the fatal entry must NOT be marked applied (it has to replay)", idx)
	}
}

// TestFSMApplyUnknownOpSingleNodeAdvances pins the other half of the trade: a
// genuine single-node store has no peer to diverge FROM, so an unregistered op
// stays a plain surfaced error and the index advances. Halting there would be a
// pure availability regression with no correctness benefit.
func TestFSMApplyUnknownOpSingleNodeAdvances(t *testing.T) {
	c := newRejectCache(t)

	fatalCalled := false
	f := newGatedFSM(t, c, false /* single node */, func(err error) { fatalCalled = true })

	res := f.Apply(&hraft.Log{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("nonexistent-op", nil)})
	resp := res.(*ApplyResponse)
	if !errors.Is(resp.Err, ErrOpNotRegistered) {
		t.Fatalf("apply err = %v, want ErrOpNotRegistered", resp.Err)
	}
	if fatalCalled {
		t.Fatal("single-node unregistered op must NOT halt")
	}
	if idx := f.AppliedIndex(); idx != 5 {
		t.Fatalf("applied index = %d, want 5", idx)
	}
}

// futureVersionEntry builds a valid stamped put entry, then bumps its version byte
// past what this binary supports so DecodeLogEntry returns ErrLogEntryVersion —
// simulating a stamped entry produced by a NEWER binary (a version-skewed fleet).
func futureVersionEntry(key, val string) []byte {
	e := EncodeLogEntryStamped("put", ops.EncodePutArgs([]byte(key), []byte(val), 0), 12345)
	e[1] = logStampVersion + 1 // an unrecognized future version
	return e
}

// TestFSMApplyVersionSkewReplicatedFailsClosed is the coverage gap that let the
// silent-skip bug ship: a version-skewed stamped entry (one a peer on a newer
// binary decodes and applies) MUST fail closed on a replicated shard — fire
// onFatalApply and NOT advance the applied index — instead of the pre-fix behavior
// of classifying it as a generic decode error, advancing, and silently skipping
// the write while peers applied it. It also pins that the wrapped error preserves
// BOTH sentinels through the multi-%w wrap.
func TestFSMApplyVersionSkewReplicatedFailsClosed(t *testing.T) {
	c := newRejectCache(t) // roomy; the error is the version, not capacity

	var fired error
	f := newGatedFSM(t, c, true /* replicated */, func(err error) { fired = err })

	res := f.Apply(&hraft.Log{Index: 8, Type: hraft.LogCommand, Data: futureVersionEntry("k", "v")})
	resp, ok := res.(*ApplyResponse)
	if !ok {
		t.Fatalf("Apply returned %T", res)
	}
	// The wrap must carry BOTH sentinels: ErrLogEntryVersion (→ classFatal, halt on
	// the Raft path) AND errPBApplyDecode (→ isInfraError, abort/NACK on the PB path).
	if !errors.Is(resp.Err, ErrLogEntryVersion) {
		t.Fatalf("apply err = %v, want it to wrap ErrLogEntryVersion", resp.Err)
	}
	if !errors.Is(resp.Err, errPBApplyDecode) {
		t.Fatalf("apply err = %v, want it to ALSO wrap errPBApplyDecode (PB abort path)", resp.Err)
	}
	if classifyApplyErr(resp.Err) != classFatal {
		t.Fatal("version-skew apply error must classify classFatal")
	}
	if fired == nil {
		t.Fatal("onFatalApply did not fire on a replicated version-skew entry — this is the silent-skip regression")
	}
	if !errors.Is(fired, ErrLogEntryVersion) {
		t.Fatalf("fatal err must wrap ErrLogEntryVersion, got %v", fired)
	}
	if idx := f.AppliedIndex(); idx != 0 {
		t.Fatalf("applied index advanced to %d on a version-skew apply; must stay 0 (fail-closed, not skip)", idx)
	}
}

// TestFSMApplyVersionSkewSingleNodeAdvances is the single-node regression guard:
// with no peers there is nobody to diverge from, so even a version-skew entry
// advances and surfaces normally rather than halting (a version bump on a genuine
// single node is a local, deterministic decode failure).
func TestFSMApplyVersionSkewSingleNodeAdvances(t *testing.T) {
	c := newRejectCache(t)

	fatalCalled := false
	f := newGatedFSM(t, c, false /* single node */, func(err error) { fatalCalled = true })

	res := f.Apply(&hraft.Log{Index: 4, Type: hraft.LogCommand, Data: futureVersionEntry("k", "v")})
	resp := res.(*ApplyResponse)
	if !errors.Is(resp.Err, ErrLogEntryVersion) {
		t.Fatalf("apply err = %v, want ErrLogEntryVersion surfaced", resp.Err)
	}
	if fatalCalled {
		t.Fatal("single-node version-skew must NOT halt")
	}
	if idx := f.AppliedIndex(); idx != 4 {
		t.Fatalf("single-node applied index = %d, want 4", idx)
	}
}

// TestFSMApplyBatchFatalHaltsNoBookkeeping proves the batch path fails closed too:
// a fatal entry fires onFatalApply BEFORE the post-loop index bookkeeping, so NO
// index is recorded and the whole batch replays on restart.
func TestFSMApplyBatchFatalHaltsNoBookkeeping(t *testing.T) {
	full := newRejectCache(t)
	fillToCap(t, full)

	var fired error
	f := newGatedFSM(t, full, true /* replicated */, func(err error) { fired = err })

	logs := []*hraft.Log{
		{Index: 10, Type: hraft.LogCommand, Data: bigPut("a")},
		{Index: 11, Type: hraft.LogCommand, Data: bigPut("b")},
	}
	f.ApplyBatch(logs)
	if fired == nil {
		t.Fatal("batch fatal apply did not fire onFatalApply")
	}
	if idx := f.AppliedIndex(); idx != 0 {
		t.Fatalf("batch applied index = %d on a fatal entry; must stay 0 (whole batch replays)", idx)
	}
}

// TestRaftReplicatedFn pins the fail-closed live-membership gate logic: only a
// positively-observed single-server group disables the halt; an unreadable or
// not-yet-learned (0-server) configuration counts as replicated.
func TestRaftReplicatedFn(t *testing.T) {
	cases := []struct {
		n    int
		ok   bool
		want bool
		why  string
	}{
		{0, false, true, "unreadable configuration -> fail closed"},
		{0, true, true, "fresh resharding joiner, not yet learned membership -> fail closed"},
		{1, true, false, "genuine single-node cluster -> gate off"},
		{2, true, true, "RF=2 group -> gate on"},
		{3, true, true, "RF=3 group -> gate on"},
	}
	for _, tc := range cases {
		fn := raftReplicatedFn(func() (int, bool) { return tc.n, tc.ok })
		if got := fn(); got != tc.want {
			t.Errorf("raftReplicatedFn(n=%d, ok=%v) = %v, want %v (%s)", tc.n, tc.ok, got, tc.want, tc.why)
		}
	}
}

// TestStoreSingleNodeNotReplicated exercises the REAL wiring: a genuine single-node
// bootstrap cluster has a 1-server configuration, so the fail-closed gate is OFF —
// a single node must never halt on a classFatal apply (no peers to diverge from).
func TestStoreSingleNodeNotReplicated(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "solo", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && !s.IsLeader() {
		time.Sleep(20 * time.Millisecond)
	}
	if s.fsm.replicated() {
		t.Fatal("single-node bootstrap cluster must be NOT replicated (gate off, no needless halt)")
	}
}

// TestStoreJoinModeIsReplicatedFailClosed exercises the REAL wiring for the online
// -resharding JOIN path: cluster AddShardOwner builds the store with owners=nil ⇒
// empty RaftPeers ⇒ a node that has not yet learned any configuration. Before the
// live-membership fix this read len(RaftPeers)>1 == false and DISABLED the gate on
// exactly the node about to be AddVoter'd into a live RF>1 group — reintroducing
// the silent-divergence bug. The gate must FAIL CLOSED (report replicated) here.
// This test fails against construction-time-peer wiring and passes with live
// membership.
func TestStoreJoinModeIsReplicatedFailClosed(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "joiner", reg)
	cfg.Bootstrap = false // resharding join mode (buildShardConfig(shardID, nil, false))
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	if !s.fsm.replicated() {
		t.Fatal("join-mode shard (empty/0-server config, destined for RF>1) must be treated as REPLICATED (fatal gate ON)")
	}
}

// TestFSMApplyBatchFatalRecordsSuccessfulPrefix proves the mixed-batch fix: on a
// classFatal entry mid-batch, the successfully-applied PREFIX index is DURABLY
// recorded (so replay skips it — non-idempotent ops like incr must not double
// -apply) while the fatal entry itself is NOT advanced (it replays). Uses a durable
// (mmap) cache so cache.AppliedIndex() reflects the persisted header.
func TestFSMApplyBatchFatalRecordsSuccessfulPrefix(t *testing.T) {
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.PageSize = 1 << 20
	cc.MaxMemoryPerShard = 1 << 20
	cc.AtCapPolicy = cache.PolicyRejectWrites
	cc.DataDir = t.TempDir() // durable header live
	c, err := cache.New(cc)
	if err != nil {
		t.Skipf("durable cache unavailable on this platform: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	// Now a put ErrFulls (classFatal), while a builtin's own short-args rejection
	// still "applies" (classAdvance — a pure function of the entry bytes and of a
	// handler that is a constant of this binary). The prefix probe has twice had
	// to move as the fatal set widened: an unregistered op name is fatal now, and
	// so is a read-only op driven through the write path; either would have halted
	// the batch at the prefix and defeated what this test measures.
	fillToCap(t, c)

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	f := newFSM(c, reg, true /* durable */, nil, nil)
	f.isReplicated = func() bool { return true }
	var fired error
	f.onFatalApply = func(e error) { fired = e }

	if c.AppliedIndex() != 0 {
		t.Fatalf("precondition: durable applied header = %d, want 0", c.AppliedIndex())
	}
	logs := []*hraft.Log{
		// classAdvance (applied prefix): a builtin's own short-args rejection.
		{Index: 5, Type: hraft.LogCommand, Data: EncodeLogEntry("put", []byte{0x00})},
		{Index: 6, Type: hraft.LogCommand, Data: bigPut("x")}, // classFatal ErrFull
	}
	f.ApplyBatch(logs)

	if fired == nil {
		t.Fatal("fatal apply did not fire onFatalApply")
	}
	if got := c.AppliedIndex(); got != 5 {
		t.Fatalf("durable applied header = %d, want 5 (successful prefix recorded; fatal entry 6 excluded so it replays)", got)
	}
	if got := f.AppliedIndex(); got != 5 {
		t.Fatalf("fsm applied index = %d, want 5", got)
	}
}
