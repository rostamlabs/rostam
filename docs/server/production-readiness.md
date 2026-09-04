# Production readiness

Rostam is self-hosted — you own running it. This is a pre-flight checklist for
putting it in front of production traffic: the operational subset that matters
most, in one place. Nothing here is new mechanism; each item links to the page
with the full detail. Walk it once before go-live, and again when you change
topology.

## Security & access

- [ ] **Authentication is on.** Never run open on a network-reachable bind — the
  server refuses to start on a non-loopback address without auth unless you pass
  `-insecure`. Prefer a `-keys-file` registry (per-key RBAC) over a single
  `-api-key`, and pass the secret via `ROSTAM_API_KEY` (env), not a flag, so it
  isn't visible in `/proc`. [Authentication modes](security.md#authentication-modes)
- [ ] **Keys are least-privilege.** Scope each key `read:`/`write:`/`admin:` with
  a collection pattern; reserve `*:*` superuser for setup, not for services.
  [RBAC keys and scopes](security.md#rbac-keys-and-scopes)
- [ ] **Multi-tenant isolation is enforced** (if applicable). Bind keys to a
  tenant and run `-tenant-isolation`; the static `-api-key`/`ROSTAM_API_KEY` is a
  superuser that sees every tenant. [Tenant isolation](security.md#tenant-isolation)
- [ ] **TLS on client listeners.** `-tls-cert`/`-tls-key` cover HTTP, gRPC, and
  TCP (≥ 1.2); misconfiguration fails at startup rather than falling back to
  plaintext. For mTLS, add `-tls-ca` — and to make a client certificate the
  *required* identity (not merely an accepted one: `-tls-ca` alone still lets a
  certless client fall back to token auth), also set `-tls-require-client-cert`. [TLS](security.md#tls)
- [ ] **Inter-node auth set** (cluster). `-internal-token` (prefer
  `ROSTAM_INTERNAL_TOKEN`) must be the same on every node — an authenticated
  cluster cannot function without it. Consider inter-node TLS
  (`-tls-node-cert`, `-node-cn-allowlist`). [Inter-node auth](security.md#inter-node-auth)
- [ ] **Audit trail shipped.** `-audit-log` emits a JSON record per authorization
  decision to stderr — **only under `-keys-file` RBAC** (it is a no-op with a
  single `-api-key`, so don't tick this box while running the static key); route
  stderr to your log pipeline. [Audit log](monitoring.md#audit-log)

## Durability & replication

- [ ] **Durability rung chosen deliberately.** The default (fsync every Raft log
  write) is the strongest; `-nosync` and `-volatile-log` step down for
  throughput. Pick by what you can afford to lose, not by the benchmark number.
  [The durability ladder](clustering.md#the-durability-ladder)
- [ ] **`-volatile-log` nodes rejoin fresh.** A crashed rung-3 node must rejoin as
  a fresh member and catch up from a snapshot — never resume in place (a
  correctness requirement). Rungs 1–2 tolerate in-place restart.
  [Durability ladder](clustering.md#the-durability-ladder)
- [ ] **Replication factor ≥ 2 for HA.** With a majority intact, a follower
  failure is invisible; a leader election briefly fails that shard's writes with
  retryable errors. [Failure behavior](clustering.md#failure-behavior)
- [ ] **PB mode configured for no-acked-loss** (if using `-replication-mode=pb`,
  which is **experimental**). Set `-min-isr ≥ 2` — `=1` can lose acknowledged
  writes across failover — and leave `-pb-commit-primary` at its default unless
  per-write latency matters more than durability. Review `shard/pbisr/BENCHMARK.md`
  first. [Replication engine](clustering.md#replication-engine)
- [ ] **Shard count has headroom.** `-shards` is fixed for the life of the
  cluster; choose shards ≫ nodes if you expect to grow (membership/RF changes
  redistribute the fixed shards, they don't add more).
  [Reconfigure](clustering.md#changing-the-cluster-reconfigure)

## Backups & disaster recovery

- [ ] **Periodic backups configured.** `-backup-dir` (filesystem) or
  `-backup-bucket` (S3/MinIO/R2, stdlib SigV4 — no AWS SDK), with
  `-backup-interval` and `-backup-retention`. [Backups](backups.md)
- [ ] **Restore rehearsed, not assumed.** Cluster restore requires the **same
  topology** (shard count + node IDs) as the backup, and a shard with no artifact
  fails loud unless you pass `-allow-missing-shards`. Practise a restore before
  you need one. [Cluster backups & restore](backups.md#cluster-backups-restore)
- [ ] **Backup destination is in-boundary** if data residency matters — point the
  S3 client at an in-network store (e.g. MinIO). Rostam initiates no egress
  otherwise. [S3-compatible backups](backups.md#s3-compatible-backups)

## Observability & alerting

- [ ] **Metrics scraped.** Prometheus against `/metrics` with a read-scoped token
  (the endpoint is scope-gated when auth is on). [Monitoring](monitoring.md#prometheus-endpoint)
- [ ] **Probes wired to the right question.** `/v1/health` is **liveness** (a
  restart decision — it stays green without quorum, so it's wrong for routing);
  `/v1/ready` is **readiness** — use it for load-balancer membership and as the
  Kubernetes `readinessProbe`. [Liveness and readiness](monitoring.md#liveness-and-readiness)
- [ ] **Alerts on the signals that matter:** quota/rate-limit rejections rising,
  the search-latency histogram shifting right, `degraded`/`missing` fields in
  search responses (partitions unreachable), and 503s on writes (transient
  elections). [Operational signals](monitoring.md#operational-signals-worth-alerting-on)
- [ ] **Memory alerts on RSS, not VIRT.** Vector collections reserve address
  space far beyond what they use, so a virtual-size threshold fires spuriously.
  [Alert on resident memory](monitoring.md#operational-signals-worth-alerting-on)

## Capacity & sizing

- [ ] **Per-collection quotas set.** `MaxVectors` / `MaxBytes` /
  `MaxInsertsPerSecond` so a runaway workload fails cleanly (`ErrCollectionFull` /
  `ErrCollectionRateLimited`) instead of exhausting the host — and `MaxVectors`
  also sizes the memory reservation. [Collection limits](../concepts/collections.md)
- [ ] **KV at-capacity policy chosen.** `PolicyRingbufEvict` (bounded cache,
  overwrites oldest) vs `PolicyRejectWrites` (returns `cache.ErrFull`) — pick per
  workload. [Cache configuration](../kv/cache.md#configuration) ·
  [Sizing guidance](../kv/cache.md#sizing-guidance)
- [ ] **Persistence on disk you control.** `-data` (and `-persistent-vectors` to
  mmap-back vectors off-heap); review RAM-vs-disk sizing before load.
  [Persistence & warm restart](../kv/cache.md#persistence-warm-restart)

## Rollout & day-2 operations

- [ ] **Rolling restarts respect the durability rung** — rungs 1–2 tolerate an
  in-place restart of a node; rung 3 (`-volatile-log`) does not (see above).
- [ ] **Membership / RF changes run online.** `-reconfigure` with the desired
  end-state `-peers`; size the context deadline to your data volume. A departing
  leader can cause a transient retryable write error mid-rebalance.
  [Reconfigure](clustering.md#changing-the-cluster-reconfigure)
- [ ] **Partition-count changes use online reshard** (vector collections): a
  dual-write + atomic cutover that is resumable.
  [Resharding online](clustering.md#resharding-collections-online)
- [ ] **Clients retry transient errors.** The Go client retries not-leader/503s
  automatically; ensure any other client retries with backoff.
  [Failure behavior](clustering.md#failure-behavior)

## Before you publish an incident channel

- [ ] **The team knows the private disclosure path** for security issues —
  `security@rostamlabs.com` or the advisory form, never a public issue.
  [Reporting vulnerabilities](security.md#reporting-vulnerabilities)

---

See also: [Running the server](running.md) · [Security](security.md) ·
[Clustering](clustering.md) · [Backups & cold tier](backups.md) ·
[Monitoring](monitoring.md) · [Deployment modes](../concepts/deployment-modes.md).
