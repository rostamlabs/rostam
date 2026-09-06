# Clustering

Cluster mode runs each node as a Raft participant. The keyspace is split into
independent per-shard Raft groups, so consensus throughput scales with shard
count; vector collections can additionally be partitioned across the cluster.

## Concepts

- **Node** — one `rostam-server -cluster` process, identified by `-node-id`.
- **Shard** — an independent Raft group owning a slice of the keyspace
  (`-shards`, default 64). Each shard runs its own goroutine set, so
  over-sharding burns CPU on scheduler churn — the rule of thumb is shards ≈
  the node's core count, not far above it.
- **Replication factor** — replicas per shard. `0` (or ≥ node count) = every
  node holds every shard (full replication); smaller values partition shards
  across nodes.
- **Peer string** — `id@raftAddr@serverAddr`, e.g.
  `n1@10.0.0.1:7400@10.0.0.1:7000`. Raft traffic and client traffic use
  separate ports. PB mode adds a fourth field — see
  [Replication engine](#replication-engine).

## Bootstrapping a cluster

Three nodes, replication factor 3 (full replication):

```sh
PEERS="n1@10.0.0.1:7400@10.0.0.1:7000,n2@10.0.0.2:7400@10.0.0.2:7000,n3@10.0.0.3:7400@10.0.0.3:7000"

# On every node: the same API key and the same inter-node token. These binds
# are non-loopback, so the server refuses to start without authentication
# (or an explicit -insecure); and a cluster running with auth requires the
# internal token so inter-node forwarded ops carry a trusted identity.
export ROSTAM_API_KEY=...        # or -keys-file for RBAC; see Security
export ROSTAM_INTERNAL_TOKEN=...

# node 1 — bootstrap ONCE on first start of a fresh cluster
rostam-server -cluster -bootstrap -node-id n1 -raft-addr 10.0.0.1:7400 \
  -tcp 10.0.0.1:7000 -http 10.0.0.1:8080 -data /var/lib/rostam/n1 -peers "$PEERS"

# nodes 2 and 3 — same flags, no -bootstrap
rostam-server -cluster -node-id n2 -raft-addr 10.0.0.2:7400 \
  -tcp 10.0.0.2:7000 -http 10.0.0.2:8080 -data /var/lib/rostam/n2 -peers "$PEERS"
rostam-server -cluster -node-id n3 -raft-addr 10.0.0.3:7400 \
  -tcp 10.0.0.3:7000 -http 10.0.0.3:8080 -data /var/lib/rostam/n3 -peers "$PEERS"
```

`-bootstrap` initializes a fresh cluster from `-peers` — set it on exactly one
node, on first start only. With auth enabled, every node also needs the same
`-internal-token` ([Security](security.md#inter-node-auth)); with TLS, see
[inter-node TLS](security.md#tls).

Clients connect to any node's `-tcp`/`-http`/`-grpc` address: the Go smart
client discovers topology and routes writes to shard leaders; HTTP/gRPC
requests landing on a non-leader are forwarded internally.

## Writes, reads, durability

- **Writes** serialize through the owning shard's Raft log; the default
  acknowledgment is Raft majority. Point writes can request more with
  `write_consistency_factor` + `wait`.
- **Reads** default to any-replica; stricter levels (leader-only,
  linearizable, bounded staleness) are per-request — see
  [Read consistency](../concepts/deployment-modes.md#read-consistency).
- **Vector memory**: `-persistent-vectors` mmap-backs collections off-heap on
  every node; Raft remains the durability authority.

## The durability ladder

The default posture — fsync on every Raft log write — is the strongest, and
the right one until measurement says the disk is the bottleneck. Two flags
step down from it; each rung buys throughput by giving something up, so choose
by what you can afford to lose, not by the benchmark number.

| Rung | Flag | What you give up |
|---|---|---|
| 1 | *(default)* | Nothing. Every Raft log write is fsynced. |
| 2 | `-nosync` | Crash-durability of the last few milliseconds of writes. Log writes skip fsync but still reach the OS page cache; replication still holds, because a majority has every acked write in memory. Use it when durability comes from replication rather than local disk — the same posture as Redis/Valkey with appendonly off, or an in-memory Aerospike namespace. |
| 3 | `-volatile-log` | Local durability of data shards entirely. Their Raft logs live fully in memory — no `write()` syscall on the replication hot path — and durability comes only from replication. The meta group stays durable. |

!!! danger "`-volatile-log` nodes must rejoin fresh"

    A `-volatile-log` node that crashes **must rejoin as a fresh member** and
    catch up from a leader snapshot — never resume in place, or its lost vote
    state can break Raft safety. This is a correctness requirement, not a
    tuning consideration: rungs 1 and 2 tolerate an operator restarting a
    crashed node in place; rung 3 does not.

## Replication engine

`-replication-mode` selects the data-plane replication engine:

- **`raft`** (default) — per-shard Raft groups, exactly as described above.
- **`pb`** — primary-backup / in-sync-replica (ISR) replication for every
  shard, with automatic failover on by default. Its blocking correctness hazards
  are closed: a lease-fenced primary plus full-ISR commit means no acked-write
  loss across a failover. **Recommended at replication-factor 2**, where it beats
  raft on throughput and tail latency; at RF=3 its full-ISR commit (wait for the
  slowest of every replica) is slower than raft's majority (fastest of a
  majority), so raft stays the default. Two caveats: PB's no-acked-loss guarantee
  assumes a bounded cross-node clock rate, and the mode is newer than the raft
  path — validate it on your workload before relying on it. Requires `-min-isr`
  and `-pb-addr`; see the measured comparison in `shard/pbisr/BENCHMARK.md`.

PB mode requires two extra pieces of configuration:

- **`-min-isr`** — minimum in-sync-replica count per shard; must be ≥ 1 in pb
  mode. **`-min-isr=1` provides no no-acked-loss guarantee across failover,
  whatever the replication factor**: every promotion resets the shard's
  in-sync set to the new primary alone, and with a floor of 1 that primary is
  permitted to acknowledge writes held on no other node until the grow driver
  re-admits the backups (seconds). Those writes are durable only on that one
  node — if it then fails, the shard has no in-sync survivor to promote and
  stays DOWN until it returns. Set `-min-isr=2` or higher (requires at least
  that many replicas) to keep every acknowledged write on a second node at
  all times.
- **`-pb-addr`** — this node's PB transport listen endpoint, e.g.
  `10.0.0.1:7200`. Other peers' PB addresses come from a fourth `@`-field in
  their `-peers` entry: `id@raftAddr@serverAddr@pbAddr`.

**Commit point** — by default (`-pb-commit-primary=false`) a write is
acknowledged only after the full ISR has it, so no acked write is lost while
any ISR member survives. `-pb-commit-primary` instead commits on local
primary apply and replicates to backups asynchronously — the Aerospike
commit-master posture. Lower per-write latency, no throughput change on a
pipelined path.

!!! warning "`-pb-commit-primary` is a durability downgrade"

    With it set, an acked write can be lost if the primary dies before a
    backup received it. Leave it at the default unless per-write latency
    matters more than never losing an acknowledged write.

**Failover** — with `-pb-auto-failover` (default on), each primary commits a
periodic liveness beacon; when one goes silent past the failover timeout, the
meta leader promotes an ISR survivor — only from the ISR, and only one whose
applied high-water is verified, so no acked write is lost — and the ISR
shrink/grow drivers un-wedge a shard stalled on a dead backup, re-opening it
once a survivor catches up. `-pb-auto-failover=false` gives a static cluster:
a PB shard whose primary dies stays DOWN until an operator intervenes.

## Raft transport

`-raft-transport` selects how Raft traffic moves between nodes:

- **`mux`** (default) — a per-group NetworkTransport over one shared TCP
  listener.
- **`fabric`** — a multiplexed batching transport that carries every Raft
  group's traffic to a peer over a single connection (fewer syscalls,
  zero-reflection codec). **Experimental**; `mux` stays the default path.

## Changing the cluster: reconfigure

Membership and replication-factor changes are an online rebalance, driven by a
one-shot operator command:

```sh
# grow to 4 nodes / rf=2: start n4 first, then
rostam-server -reconfigure -replication-factor 2 \
  -peers "n1@...@...,n2@...@...,n3@...@...,n4@...@..."
```

or programmatically:

```go
result, err := rostam.Reconfigure(ctx, serverAddrs, targetPeers, rf)
// RebalanceResult{Moves, Done, Failed}
```

The target `-peers` list is the **desired end state** — include new nodes, omit
decommissioned ones. Decommissioned nodes keep running and forwarding until
their shards re-home, then can be shut down. The call blocks until the
rebalance completes; size the context deadline to your data volume.

This redistributes the fixed `-shards` groups (key-value **and** vector shards
alike) across the new membership — a shard gains the new node as a Raft voter,
waits for it to catch up, then drops departing owners, so a caught-up owner is
retained at every step. Reconfiguration requires no planned downtime, though a
write can still hit a transient retryable error if a departing owner was the
shard leader (see [Failure behavior](#failure-behavior)). It does **not** change
the shard count: `-shards` is fixed for the life of the cluster, so choose it
with headroom (shards ≫ nodes) if you expect to grow. Changing the *partition
count* is a per-collection operation (below) and applies to vector collections
only — there is no key-value equivalent.

## Resharding collections online

Partitioned vector collections (not the key-value store) can change partition
count without downtime:

```
POST /v1/collections/{name}/reshard        {"new_partitions": 8}
POST /v1/collections/{name}/reshard/abort  # pre-cutover only
```

Online reshard dual-writes to old and new partition sets while a background
copy catches up, then performs an atomic cutover; the process is resumable. The
offline alternative (`.../resplit`) is faster but requires the caller to quiesce
writes, with `.../resplit/cleanup` to drop orphaned partitions afterwards.

## Failure behavior

- A follower failure is invisible (majority intact).
- During a leader election, writes to that shard briefly fail with retryable
  errors — the Go client retries automatically (`MaxNotLeaderHops`).
- Cross-shard reads during a partition outage return partial results with a
  degradation flag by default (`on_partition_unavailable=0`), or fail hard with
  `1`.
