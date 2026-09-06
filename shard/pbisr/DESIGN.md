# Option B: primary-backup / ISR data-plane replication (flag-gated)

Replace the per-shard hashicorp/raft DATA group with a lighter primary-backup
protocol, keeping Raft for the CONTROL plane (MetaRaft: membership, per-shard
epoch, ISR set, placement, catalog, resharding). Flag-gated behind
`ReplicationMode` ∈ {`raft` (default, unchanged), `pb-isr`}. The per-shard Raft
path stays byte-identical when not selected.

Reference: raft/fabric/DESIGN.md (transport), the architect risk analysis, and
Kafka KRaft+ISR as the model. This doc pins the CORRECTNESS decisions; the win
is bounded (profile is RTT/latency-bound) so correctness, not speed, is the bar.

## The seam

`shard.Store` today holds a concrete `raft *raft.Node` (shard/store.go:30) and
routes every mutation and leadership check through it: `ApplyIndexed`,
`VerifyLeader`, `CommitIndex`, `Barrier`, `IsLeader`, `LeaderAddr`, `Shutdown`.

Introduce `type replicator interface` with exactly those methods (plus a
linearizable-read barrier). Two impls:
- `raftReplicator` — thin wrapper over `*raft.Node` (today's behavior, default).
- `pbReplicator` — the new primary-backup engine (this package).

`Store` holds a `replicator`; `shard.Config.ReplicationMode` selects it at
construction. NOTHING else in Store changes shape — the read/write call sites
already funnel through the interface methods.

## Model

Each shard has, in MetaRaft (authoritative, the control plane):
- `epoch uint64` — leadership generation, bumped by MetaRaft on every primary
  election. Monotonic, durable via the meta log.
- `primary nodeID` — who holds `epoch`.
- `isr []nodeID` — the in-sync set (includes primary). Grown/shrunk ONLY by a
  MetaRaft-committed op; never by a primary's local belief.
- `minISR int` — durability floor (default: majority of RF, e.g. 2 for RF=3).

Per write, the primary assigns a monotonic `seq uint64`. The pair `(epoch, seq)`
totally orders writes and replaces Raft's `(term, index)`.

## Write path (nosync tier — the only tier Option B helps; see §Durability)

1. Client write → shard primary (routing unchanged, cluster/node.go).
2. Primary checks it holds the current `epoch` (leased from MetaRaft, cached +
   revalidated — see Read barrier). If not → `NotLeaderError`.
3. Primary assigns `seq = ++lastSeq`, applies to its in-memory FSM (cache +
   vectors — NOTE: vectors ride the same FSM, shard/fsm.go:99), and appends
   `(epoch, seq, op)` to an in-memory ordered ring (the catch-up backlog).
4. Primary ships `(epoch, seq, prevSeq, op)` to each ISR backup over `fabric`
   (new frame kinds; reuses raft/fabric framing + pipelining + snapshot conn).
5. Backup: `epoch >= local watermark` (FENCE) AND `prevSeq == local lastApplied`
   (gap-free order) → apply → ack `(epoch, seq)`. Else reject (triggers catch-up).
6. Primary counts acks **per this write's seq** from ISR members; once the count
   reaches `max(minISR, |required|)` (including itself) → ack the client.

## Correctness: the six hazards and their defenses

- **H1 stale-primary writes** (old primary keeps writing after a new epoch):
  backup-side epoch fencing rejects epoch-N appends *when the backup has learned
  the new epoch*. This is NECESSARY BUT NOT SUFFICIENT — see "Open hazards / OH1"
  below: a co-partitioned old primary + old backup can still commit an old-epoch
  write that a disjoint new-epoch election never saw, losing an ACKED write on
  heal. The fence alone does not close H1; a quorum-intersection or lease
  mechanism is required and is NOT yet implemented. Treat H1 as OPEN.
- **H2 stale-primary reads** (THE linchpin): a fenced-but-unaware primary must
  not serve a linearizable read missing a newly-acked write. Defense: a
  linearizable read confirms the primary still holds `epoch` **as of the read**.
  DECISION: use an **ISR confirmation round-trip** (a lightweight "still epoch E?"
  to a quorum of ISR), NOT a clock-lease. Rationale: Raft's VerifyLeader already
  costs an RTT; matching it keeps the SAME safety class (no new clock-skew
  assumption). A lease is a future optimization, explicitly out of scope for v1.
  UPDATE: the OH1 write-path fix introduced a real primary lease
  (`GrantLease` / self-fence on `e.now() >= leaseExpiry`). H2 reads will SHARE
  that same lease — a fenced primary whose lease has lapsed must not serve a
  stale linearizable read either — folding the read barrier into the lease once
  the MetaRaft lease-grant/honor half lands.
- **H3 unclean election on ISR shrink** (Kafka min.insync trap): when ISR would
  drop below `minISR`, the primary **rejects writes** (chooses unavailability).
  ISR shrink is only effective once committed to MetaRaft; a stale backup below
  the high-water `seq` can never be promoted because MetaRaft only elects a new
  primary from a member whose acked `seq` ≥ the last committed high-water.
- **H4 acked-backup-then-fails**: acks reflect **materialized** state (applied;
  for sync tier, fsynced) — never a promise. `minISR ≥ 2` for RF ≥ 2 ⇒ every
  acked write is on ≥2 nodes ⇒ survives one failure. Same as Raft majority.
- **H5 split-brain**: MetaRaft gives one primary per epoch; backup epoch fencing
  (H1) blocks the loser's writes; H2 blocks its reads. Same mechanism.
- **H6 lagging backup counted in-sync**: acks are matched **per-seq** — a backup
  counts toward write W iff it acked W's exact seq, never "is alive". A liveness
  signal is never a durability signal. Enforced at the ack-matching layer.

The proof closes ONLY with: (a) per-`(epoch,seq)` acks, (b) MetaRaft-committed
`minISR` floor with write-rejection below it + promotion only of a
caught-up member, (c) the H2 read confirmation. All three are mandatory for v1.

## Open hazards (MUST be closed before pb-isr is trusted with data)

> **STATUS 2026-09-06 — all three CLOSED in code.** OH1 (incl. the MetaRaft
> half), OH2, and OH3 are closed with named guards and passing tests; the
> per-hazard "STILL PENDING" notes below are historical. See the STATUS block at
> the top of this repo's pbisr work and `cluster/pb_failover.go` /
> `cluster/config.go:394-401` (the OH1 honor-rule construction guard) +
> `cluster/pb_partition_test.go` (the partition e2e). `-replication-mode=pb` is
> no longer experimental (raft stays the default for RF=3 performance reasons).
> Remaining hardening, not correctness holes: a PB-mode linearizable
> stale-primary-read e2e, and a long nosync bake.

An adversarial review (2026-07-20) found the backup-side epoch fence is NOT a
sufficient defense on its own. These were tracked as gating non-experimental use
(now closed — see the status note above):

- **OH1 — co-partitioned stale quorum loses an acked write (CRITICAL). —
  WRITE PATH CLOSED IN-ENGINE (2026-07-20); MetaRaft half PENDING.**
  RF=3, minISR=2, epoch 1, ISR {P,B1,B2}. Partition {P,B1} from {B2 + MetaRaft
  leader}. MetaRaft elects B2 as epoch 2 (its ISR resets to {B2}, so epoch 2 is
  write-unavailable). On the stale side P reads its own lagging MetaFSM
  (epoch 1) → the OLD Propose fence passed; P ships epoch-1 writes to B1, which is
  ALSO stale (never adopted epoch 2) → B1 acks → P reaches minISR=2 and ACKS the
  client. On heal, epoch 2 is authoritative and never saw the write → it is
  lost. Root cause: no guaranteed intersection between the set that COMMITS a
  write in epoch E and the set that ELECTS epoch E+1. Raft avoids this because
  commit and election are the same majority over the same membership. The
  original code's fence `primary != nodeID || epoch != ctrlEpoch` was the bug:
  `epoch` (cached) and `ctrlEpoch` (a LOCAL stale MetaFSM read) are BOTH stale on
  a partitioned node, so they agreed and the fence passed — a self-consistency
  check on stale state, not an authority check.

  Fix originally required ONE of: (i) MetaRaft election guarantees the new-epoch
  electable set intersects EVERY possible old-epoch commit quorum; (ii) extend
  the H2 ISR/control-plane epoch confirmation to the WRITE path; (iii) a real
  primary lease with expiry MetaRaft honors before granting the next epoch.

  **Implemented (engine.go):** a combination of (iii) + full-ISR commit.
  - **Lease self-fence.** Propose no longer reads `ctrl.Epoch`. The fence is now
    `leaseEpoch == epoch && now() < leaseExpiry` on the engine's OWN injected
    monotonic clock (`e.now()` — never `time.Now` in the decision path). A
    partitioned primary that cannot renew its lease fails the fence purely on its
    local clock, observing NO new control-plane state → `ErrLeaseExpired`.
    `GrantLease(epoch, expiryMonoNs)` models MetaRaft granting/renewing the lease.
  - **Full-ISR commit.** A write commits only when EVERY current ISR member acks
    its exact `(epoch,seq)` — not a minISR subset. The H3 floor is retained as an
    ISR-SIZE guard (`len(isr) < minISR` → `ErrBelowMinISR`, choose unavailability);
    above the floor, ALL current-ISR members must ack. Because every current-ISR
    member then holds every committed write, the commit set always intersects any
    single member MetaRaft can promote. Stragglers are removed by a MetaRaft ISR
    shrink (out of scope here). See `TestOH1StalePrimarySelfFencesOnLeaseExpiry`,
    `TestLeaseRenewalReenablesPropose`, `TestFullISRCommitRequiresEveryMember`.

  **MetaRaft half — CLOSED (2026-09-06; this text below is the original
  requirement, now implemented).** Enforced by construction in
  `cluster/config.go:394-401` (`failoverTimeout > pbLeaseTTL +
  metaContactStaleness + renewInterval + failoverTick`) + the failover ticker in
  `cluster/pb_failover.go` (`decidePBPromotions` gates every epoch bump on
  `nowNs - lastRenewNs > failoverTimeout`), verified by
  `TestPBFailoverPartitionNoDoublePrimary` (P's last ack strictly precedes Q's
  promotion). IMPORTANT SCOPE: the lease-lapse precondition lives in that
  promotion PATH, not inside `ApplySetShardEpoch` itself (the raw op only enforces
  epoch monotonicity) — `pb_failover.go:410` is the sole promotion caller today,
  and any future promotion path MUST reuse the same timing gate or the intersection
  argument breaks. The engine fix is
  necessary but NOT sufficient on its own. Promotion-completeness is enforced at
  MetaRaft, not here: MetaRaft MUST NOT grant epoch E+1 to any node until the
  epoch-E lease has PROVABLY lapsed (grant the next lease strictly after the prior
  lease's max expiry + clock-skew bound). Without that guarantee, a new primary
  could be elected while the old lease is still (from the old primary's clock)
  valid, and the intersection argument breaks. This half introduces the
  clock-skew assumption option (iii) trades for; it is a separate, later step.
  NOTE: H2 linearizable reads will share this SAME lease (a fenced primary whose
  lease lapsed must not serve a stale linearizable read either).

- **OH2 — apply-before-quorum uncommitted tail.** The primary applies locally
  before quorum; a quorum-timeout write is applied but not committed. Safety
  provisos, now partially enforced in code: `committed` (the min-ISR
  high-watermark) is tracked separately from `lastSeq`/`lastApplied`; P3
  linearizable reads and P4 failover election MUST key off `committed` only. An
  in-memory FSM cannot truncate an uncommitted tail, so a demoted ex-primary
  MUST be snapshot-reloaded, never ring-delta rejoined (P4 constraint).

- **OH3 — epoch-blind gap check.** `lastApplied` is a bare seq; the gap check is
  numeric. On an epoch increase a snapshot/rebase is required rather than
  trusting a numeric prevSeq match (P4). Track `(epoch,seq)` for lastApplied.

## Read barrier

- AnyReplica / LeaderOnly (default paths): unchanged, cheap (no consensus).
- Linearizable: `pbReplicator.verifyPrimaryAndCatchUp(deadline)` — confirm
  current epoch with an ISR quorum, then ensure local FSM has applied through
  the high-water `seq` the quorum reports. Mirrors the existing
  `verifyLeaderAndCatchUp` (shard/store.go:389) one-for-one.

## Durability tiers

- **nosync**: no per-write disk. Durability = "applied on ≥ minISR nodes in
  memory." The catch-up ring + periodic FSM snapshots are the only persistence.
  THIS is the tier that maps to Aerospike in-memory RF=2 and where the win lives.
- **sync**: primary AND every acking backup must fsync `(epoch,seq,op)` to a
  local log before acking. That fsync is exactly the cost Option B set out to
  avoid, so **Option B offers no win in the sync tier** — it just reuses the
  existing WAL there. v1 may simply fall back to `raft` mode when sync is
  requested, or reuse logstore.WAL for the pb log. DECISION: v1 is nosync-only;
  sync requests keep `raft` mode.

## Recovery / rejoin

- Backup within the primary's retained ring (bounded by expected lag) → ship the
  seq delta.
- Backup outside the ring / fresh → full FSM snapshot (reuse
  shard/fsm.go serializeSnapshot — KV + vectors) over the fabric dedicated
  snapshot conn, then tail-replay from the ring. Reuses shard/snapshot.go.
- Primary retains: the op ring (sized to cover lag) + snapshot capability —
  i.e. a bespoke in-memory version of the log we removed. Accepted tradeoff.

## Reuse (per the risk analysis)

- **MetaRaft** (cluster/meta.go, meta_fsm.go): add `ShardEpoch map[int]uint64`,
  `ShardPrimary map[int]string`, `ShardISR map[int][]string` to MetaState;
  new ops `OpSetShardEpoch`, `OpSetISR`. Follows the existing OpSetPlacement
  shard-by-shard pattern (meta_fsm.go:226).
- **write_consistency.go** `appliedReplicas`/`leaderAppliedIndex` (lines 80-219)
  — already ~80% of the "who reached seq" ack primitive; adapt to per-seq push.
- **fabric** — carries the new backup-replication frames + snapshot conn.
- **migration** (cluster/migrate.go:65-135) — gain-then-release choreography
  ports to "add backup / wait seq catch-up / commit ISR / remove backup".

## Phasing (each phase flag-gated, reviewed, tested before the next)

- **P1 control plane**: MetaState epoch/primary/ISR + ops + minISR floor; epoch
  bump on primary election; ISR grow/shrink ratified by MetaRaft. Unit tests.
- **P2 data plane core**: `replicator` interface + `raftReplicator` (extract
  today's behavior, prove byte-identical) + `pbReplicator` write path (assign
  seq, apply, ship, per-seq ack). Backup receive (fence + gap + apply + ack).
- **P3 read barrier**: linearizable read via ISR epoch confirmation (H2).
- **P4 recovery**: ring delta + snapshot catch-up; primary failover (MetaRaft
  elects caught-up member, bumps epoch).
- **P5 migration/resharding** port to ISR add/remove.
- **P6 adversarial tests**: partition, leader-kill-under-load, ISR-shrink,
  stale-primary-read, lagging-backup — the H1–H6 acceptance gate. Must match the
  existing shard/linearizable_*_test.go and cluster/partition_test.go rigor.

Default stays `raft` (for RF=3 performance — PB's full-ISR commit loses to raft's
majority there; PB's win is at RF=2). PB is promoted out of experimental as of
2026-09-06: P6 is green except a PB-mode linearizable stale-primary-read e2e
(remaining hardening), and a long nosync bake is still recommended before
relying on PB for critical data. No in-place hot switch on a live shard.
