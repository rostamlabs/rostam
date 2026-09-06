# Dual-Mode Replication: Raft + Primary-Backup (Cluster-Level)

**Status:** Design approved 2026-07-21.
**Related:** `shard/pbisr/DESIGN.md`, `shard/replicator.go`

## 1. Motivation

Rostam replicates every write through a per-shard hashicorp/raft group: an
append to a replicated log, a durability decision, and a quorum-commit hop per
write. This gives strong consistency and self-contained fault tolerance, but the
per-write consensus log is the structural reason Rostam is ~2.4× slower than
Aerospike on replicated writes. Aerospike has no per-write consensus log — it
uses primary-backup replication (master forwards synchronously to replicas,
acks, done) with a separate consensus layer (Paxos) that runs only on cluster
membership changes. Redpanda keeps Raft but replaces the entire engine (Seastar
thread-per-core), which is not reproducible in Go.

The two levers that actually move replicated-write latency are therefore
(a) change the replication model, or (b) change the engine. Lever (b) is a
6–12 month rewrite, which is not planned. This spec
pursues lever (a): offer **primary-backup/ISR replication as a second,
cluster-level replication mode alongside Raft**, selected by a launch flag. A
deployment picks the mode that fits its workload:

- **Raft** — strong consistency, majority-quorum commit (tolerates one slow/down
  replica with no write stall), self-contained failover. Slower writes.
- **Primary-backup (PB)** — strong consistency (linearizable, no acked-write
  loss), full-ISR commit, faster happy-path writes. Failover is driven by the
  control plane; a slow backup slows every write.

**Scope decision — cluster-level, not per-collection.** An earlier draft made
the mode selectable per collection. Two facts killed that:

1. **Collections are a vector-only concept.** `CreateCollection` exists only on
   the vector store (`tx.vectors.CreateCollection`); the KV/cache path has no
   namespace or collection at all (keys are hashed raw). The replicated-write
   workload this whole effort targets — KV writes vs Aerospike — has no
   collections to attach a mode to. Per-collection mode is inapplicable to the
   exact path we care about.
2. **Mixed needs are better served by two services.** If a deployment genuinely
   wants both a strong tier and a fast tier, running two single-mode
   clusters/services is simpler and stronger than in-cluster mixing: zero new
   routing code, zero mode-aware `shardOf`, zero collection metadata, zero
   client-protocol changes, and clean physical isolation (the fast tier can't
   destabilize the safe tier). The only thing given up is "one endpoint for
   both" — an operational nicety, not a capability.

So the mode is chosen **once, at cluster launch**, and applies to the whole
cluster. Deployments that need both tiers run two services.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Granularity | **Cluster-level** — one mode per cluster, set at launch |
| 2 | Selection | Launch flag `-replication-mode=raft\|pb` (default `raft`) |
| 3 | PB guarantee | **Linearizable, no acked-write loss** (lease-fenced primary + full-ISR commit) |
| 4 | Mixed workloads | **Run two services** (one all-Raft, one all-PB) — not in-cluster mixing |
| 5 | Structural cost | **Zero** — every shard already runs one `replicator`; the flag only decides which implementation |

Non-goals: per-collection mode; mode-partitioned shard pools; live Raft↔PB
migration; relaxed/tunable-consistency reads; replica reads.

## 3. Current architecture (what we build on)

- **Global shard pool.** `cluster.Node` holds `shards []*shard.Store` of length
  `cfg.NumShards`, created once at startup (`cluster/node.go`). Routing is
  `shardOf(key) = xxhash.Sum64(key) % NumShards` (`cluster/router.go`).
- **The replicator seam.** `shard/replicator.go` defines `replicator`, the
  data-plane interface a `Store` writes/reads through. `*raft.Node` satisfies it
  directly (compile-time assertion), so the Raft path is a zero-behavior-change
  extraction. Method set: `ApplyIndexed`, `IsLeader`, `LeaderAddr`,
  `CommitIndex`, `AppliedIndex`, `LastIndex`, `VerifyLeader`, `Barrier`,
  `AddVoter`, `RemoveServer`, `LeadershipTransferToServer`, `Stats`, `Shutdown`.
  **This seam is the entire enabling mechanism — the flag just decides which
  implementation each `Store` constructs.**
- **The PB engine skeleton.** `shard/pbisr/` has `engine.go` (lease fence,
  full-ISR commit), `types.go` (`ReplicateMsg`/`AckMsg` with
  `(Epoch, Seq)` totally ordering writes in place of Raft's `(term, index)`),
  `inmem_transport.go`, and `DESIGN.md` (OH1/OH2/OH3 hazards + fixes).
- **The control plane.** `MetaRaft` (`cluster/meta_fsm.go`) already carries
  `ShardEpoch`/`ShardPrimary`/`ShardISR` accessors and the monotonic
  `OpSetShardEpoch` / epoch-guarded `OpSetShardISR` apply cases, plus the
  alloc-free `ServerAddrForRaftAddr` lookup.

Because the mode is cluster-wide, every shard runs the *same* engine — there is
no mixed-mode shard, no routing change, and the FSM state is never split. The
all-Raft cluster is byte-identical to today.

## 4. Design

At cluster construction, each `shard.Store` is given a `replicator` chosen by
`cfg.ReplicationMode`:

- `raft` (default): `*raft.Node`, exactly as today. No behavior change.
- `pb`: a `pbReplicator` wrapping `pbisr.Engine`, satisfying the same
  `replicator` interface.

The Raft path is untouched. All new work is behind the `pb` branch. The flag is
validated at startup; an unknown value is a hard config error.

## 5. The PB data plane (linearizable, no acked-write loss)

All three properties come from the control plane owning `(epoch, primary, ISR)`
and the primary committing only on the full ISR. See `shard/pbisr/DESIGN.md` for
the hazard proofs; this section is the contract.

### 5.1 Write path
1. Client routes to the shard's **primary** (from topology).
2. Primary assigns the next `(epoch, seq)` — `epoch` = its leadership
   generation from MetaRaft, `seq` = per-epoch monotonic counter.
3. Primary applies locally and ships `ReplicateMsg{Epoch, Seq, PrevSeq, Data}`
   to **every** current ISR backup.
4. A backup that has adopted a higher epoch rejects the message (H1/H5 fencing).
   A backup whose applied seq ≠ `PrevSeq` signals a gap → catch-up.
5. Primary acks the client only when **all** current ISR backups have returned
   `AckMsg{Epoch, Seq, OK:true}` for that exact `(epoch, seq)` — the H6
   durability floor. A liveness signal or an ack for a different seq never
   counts. This full-ISR commit is what makes acked writes survive any failover.

**Full-ISR commit vs the min-ISR floor.** "Commit on the full ISR" means all
members *currently* in the ISR, not all replicas ever assigned — a down/slow
backup is removed from the ISR by the control plane (§5.3), after which writes
proceed on the smaller set. A **minimum ISR size** (`minISR`, config) gates
this: if removing a backup would drop the ISR below `minISR`, writes **stall**
rather than commit on too few copies. This is the concrete form of the
availability tradeoff vs Raft: Raft commits on a majority and tolerates one down
replica with no stall, whereas no-loss PB must keep the full ISR acking or
shrink it through a control-plane round, and stalls once it hits `minISR`.

### 5.2 Read path (linearizable)
- Served from the **primary only** (no replica reads).
- Gated by a lease/epoch barrier: the primary self-fences on a monotonic clock
  (the OH1 lease fix) and confirms it still holds the current epoch before
  serving. Maps onto the seam's `VerifyLeader`/`Barrier` methods.

### 5.3 Failover (no acked-write loss)
1. MetaRaft detects primary loss (heartbeat/lease expiry), bumps the shard epoch
   (`OpSetShardEpoch`, monotonic), and reassigns the primary **from within the
   current ISR** (a member guaranteed to hold every acked write, because acks
   required full ISR).
2. The old primary self-fences when its lease expires: it cannot ack after
   losing leadership, closing the OH1 acked-write-loss window.
3. ISR membership changes (shrink on a down backup, grow on catch-up) go through
   the epoch-guarded `OpSetShardISR` so a stale primary can't rewrite ISR.

### 5.4 Seam mapping (`pbReplicator` implements `replicator`)
| Method | PB semantics |
|--------|--------------|
| `ApplyIndexed` | assign `(epoch,seq)`, apply, replicate to full ISR, ack on floor; returns `seq` as index |
| `IsLeader` / `LeaderAddr` | am I the current primary? / who is |
| `VerifyLeader` / `Barrier` | linearizable read barrier — confirm still-primary for this epoch + catch up |
| `CommitIndex` / `AppliedIndex` / `LastIndex` | committed/applied/last `seq` |
| `AddVoter` / `RemoveServer` / `LeadershipTransferToServer` | ISR / primary changes, driven by MetaRaft |
| `Stats` / `Shutdown` | engine stats / graceful stop |

## 6. Deployment model

- **One mode per cluster.** Choose at launch with `-replication-mode`. All
  shards, all data in that cluster use that mode.
- **Need both tiers?** Run two services: an all-Raft cluster for
  strong-consistency workloads and an all-PB cluster for throughput workloads.
  They are independent deployments with independent endpoints; the client points
  at whichever it needs.
- **KV vs vectors.** Both KV writes and vector writes flow through the same
  per-shard `replicator`, so both benefit from the chosen mode. Collections
  (vector-only) are irrelevant to mode selection.

## 7. Implementation

This is now a **single feature**, not a staged multi-plan effort, because the
per-collection surface is gone.

1. Finish `pbisr.Engine`: write path (§5.1), read barrier (§5.2), and
   control-plane failover (§5.3) against a real transport (fabric frame codec).
2. Add `pbReplicator` satisfying `replicator`; wire mode selection into shard
   construction via `cfg.ReplicationMode` + the `-replication-mode` flag.
3. Stand up a real 3-node cluster in `pb` mode; run the A/B benchmark vs Raft.

**Go/no-go gate:** the linearizability + no-acked-loss failover tests pass **and**
PB measurably beats Raft on the A/B benchmark. If PB does not beat Raft, stop —
the feature's premise is false and the flag ships as experimental/off.

> **Outcome (2026-09-06):** gate passed at **RF=2** (PB ~1.7× raft, real separate
> machines); PB **RF=3** full-ISR loses to raft's majority. `-replication-mode=pb`
> is promoted out of experimental with RF=2 as the recommended config; raft stays
> the default. See `BENCHMARK.md` and `DESIGN.md`.

## 8. Verification

- **Linearizability.** Porcupine/Jepsen-style history checking under concurrent
  writes interleaved with failovers, for the PB path.
- **No-acked-loss failover.** Kill the primary mid-flight under load; assert
  every client-acked write is present on the new primary. Repeat across ISR
  shrink/grow.
- **Hazard regressions.** The OH1 (co-partitioned stale-quorum), OH2, OH3 tests
  from `pbisr/DESIGN.md`, kept green.
- **Fencing.** Old primary cannot ack after lease expiry; stale primary cannot
  mutate ISR (epoch guard).
- **A/B benchmark (the gate).** PB vs Raft on the same 3-node cluster,
  `GOGC=40`, fabric transport, volatile-log where comparable. Report throughput
  @128/256 conns and p50/p99. PB must beat Raft to justify shipping the mode on.
- **Regression.** All existing multinode/replication/leader-kill/partition
  suites pass in `raft` mode (default = unchanged behavior).

## 9. Risks

- **PB may not beat Raft on this hardware.** The path is latency/coordination-
  bound and the laptop is latency-bound; the win is expected to show structurally,
  not on the laptop. §7's gate is explicit — measure before shipping the mode on.
  Prefer a dedicated high-clock box for the decisive number.
- **Control-plane dependency.** PB failover leans entirely on MetaRaft
  correctness; Raft is self-contained. MetaRaft is already the cluster's trust
  root, so this concentrates rather than adds risk.
- **A second consistency model to maintain.** Mitigated by keeping PB strong (no
  relaxed variant) and reusing the documented OH1/OH2/OH3 analysis, and by the
  mode being off by default until the gate is met.
