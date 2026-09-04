# Monitoring

## Prometheus endpoint

```
GET /metrics       # also /v1/metrics
```

Serves the Prometheus text exposition format. When authentication is enabled
the endpoint is scope-gated like any other read — scrape with a token that has
read access.

```yaml
# prometheus.yml
scrape_configs:
  - job_name: rostam
    static_configs: [{ targets: ["10.0.0.1:8080"] }]
    authorization:
      credentials: <token with read scope>
```

## What's exported

Metrics are collected per dense collection (via the internal `__metrics__` op
fanned across collections) and include the engine's live counters:

- **Traffic**: search ops, insert ops
- **Latency**: search and insert latency histograms
- **Health of the index**: live size, tombstoned count, average search depth
- **Pressure signals**: TTL expirations, quota rejections
  (`MaxVectors`/`MaxBytes`), rate-limit rejections, filter rejections
- **Lanes**: sparse vector counts

The same numbers are available programmatically as `Collection.Stats()` in the
library.

## Liveness and readiness

Two auth-exempt probes, and they answer different questions — an infrastructure
probe carries no token, so neither requires auth.

`GET /v1/health` is **liveness**: 200 whenever the process is serving. Use it to
decide whether to restart a container. It stays green on a node that has lost
quorum, so it is the wrong signal for routing.

`GET /v1/ready` is **readiness**: 200 only when every shard this node hosts can
actually serve. A hosted shard makes the node un-ready when it has no usable
leader (the node is neither leader nor knows an address to forward to — quorum
lost, an election in flight, or no valid primary lease under PB replication), or
— in PB mode — when its in-sync replica set has fallen below the configured
min-ISR floor, since the primary refuses to ack writes below it. The response
names the offending shard ids. Use this one for load-balancer membership and as
a Kubernetes `readinessProbe`.

In single-node and embedded deployments there are no hosted shard groups, so
readiness always answers ready.

## Audit log

With `-audit-log`, every authorization decision is emitted to stderr as a
structured JSON record — principal redacted to a token fingerprint, decision,
op, and resource. Ship stderr to your log pipeline to get a security audit
trail. This applies only under `-keys-file` RBAC; with a single `-api-key` the
flag is a no-op. See [Security](security.md).

## Operational signals worth alerting on

| Signal | Meaning |
|---|---|
| quota/rate-limit rejections rising | a collection hit `MaxVectors`/`MaxBytes`/`MaxInsertsPerSecond` |
| search latency histogram shifting right | index growth, cold-tier restores, or CPU saturation |
| `degraded`/`missing` fields appearing in search responses | partitions unreachable during fan-out — check node health |
| 503 responses on writes | Raft leader election in progress; should clear in seconds |

!!! warning "Alert on resident memory, not virtual"

    Large vector collections reserve address space well beyond what they use, so
    `VIRT`/`VSZ` can read tens of gigabytes per process while actual memory use
    is unchanged. Reserved-but-unused address space consumes no memory and no
    swap. Alert on **RSS**; a threshold on virtual size will fire spuriously.
    See [Growing a collection doesn't stall queries](../performance.md#growing-a-collection-doesnt-stall-queries).
