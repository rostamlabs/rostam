# Backups & cold tier

Two related mechanisms use collection snapshots and an object store: **periodic
backups** (durability) and the **cold tier** (evict idle collections from RAM,
restore lazily on access). Backups work in both single-node and `-cluster`
mode — in a cluster each node backs up the shards it owns, plus the cluster
catalog. The cold tier is single-node only (`-cold-tier-after` is rejected in
cluster mode: vectors are partitioned across shards).

## Filesystem backups

```sh
rostam-server -http 127.0.0.1:8080 -data ./data \
  -backup-dir /var/backups/rostam \
  -backup-interval 30m \
  -backup-retention 24
```

Every `-backup-interval`, each collection is snapshotted into `-backup-dir`
(keyed under `-backup-prefix`, default `default`); only the newest
`-backup-retention` snapshots per collection are kept (default 24).

File names follow the object layout below, except that each `:` (from the
timestamp) is written as `%3A`, since Windows file names cannot contain `:`.
Directories written by earlier builds, with a literal `:` in their file names,
are still read, restored from and pruned on Linux and macOS.

## S3-compatible backups

Point the same machinery at S3, MinIO, Cloudflare R2, or any SigV4-compatible
store — the client is stdlib-only, no AWS SDK:

```sh
rostam-server ... \
  -backup-bucket rostam-backups \
  -backup-region us-east-1 \
  [-backup-endpoint https://minio.internal:9000] \
  [-backup-tenant default] \
  [-s3-path-style=true]        # default true; false for AWS virtual-host style
```

Credentials come from the standard AWS environment variables. Object layout:

```
<tenant>/<escaped-collection-name>/<RFC3339-timestamp>.snap
<tenant>/<escaped-collection-name>/<RFC3339-timestamp>.cfg.json
```

**Single-node** collection snapshot keys (the layout above) are write-once: a
backup never overwrites an existing one, and of backups that start in the same
second — an on-demand backup during a periodic one, or two servers sharing a
prefix — exactly one publishes and the others report that the snapshot exists.
That is enforced by the store itself, so it needs **conditional writes**
(`If-None-Match: *` on PUT), which AWS S3 supports; check that your
S3-compatible store does. Some, older MinIO releases among them, ignore the
header and would overwrite instead; Rostam checks this once per process with a
throwaway object next to the first backup key and, on such a store, **fails
every single-node backup** with "backend does not support conditional create"
rather than overwriting silently. A filesystem `-backup-dir` gets the same
guarantee from hard links, so it must be on a filesystem that has them (not
FAT32 or exFAT).

**Cluster backups are not write-once yet.** Shard snapshots, their sidecars and
the catalog are still written with an ordinary overwrite, so two cluster
backups that land on the same timestamp can replace each other's objects. Avoid
running two backups of one cluster into the same prefix at once.

## On-demand backup & restore

With an object store configured (these return **412 Precondition Failed**
otherwise):

```
POST /v1/admin/backup                       # snapshot all collections now
GET  /v1/admin/backups                      # list available snapshots
POST /v1/collections/{name}/restore         # restore a collection
```

## Cluster backups & restore

The same flags in `-cluster` mode take periodic point-in-time backups of each
node's owned shards (cache + vectors, via the shard FSM snapshot) **and** the
MetaRaft catalog. Restoring is a one-shot disaster-recovery flag:

```sh
rostam-server -cluster ... -restore -backup-dir /var/backups/rostam
```

Run once on every node of a fresh cluster: after bring-up, each node restores
the shards it owns and the catalog from `-backup-dir` (or `-backup-bucket`),
then continues serving. The cluster must have the **same topology** (shard
count and node IDs) as the backup — a mismatch fails loud. A shard with no
backup artifact also fails loud by default, so an incomplete backup cannot
silently lose a shard's keys; `-allow-missing-shards` is the explicit operator
override that brings such shards up empty (logged per shard).

## Cold tier

For long-tail workloads (many collections, few hot), the cold tier evicts
collections idle longer than a threshold to the object store, leaving a stub
that restores lazily on the next access:

```sh
rostam-server ... -backup-bucket rostam-cold -backup-region us-east-1 \
  -cold-tier-after 1h
```

Manual control:

```
POST /v1/collections/{name}/evict           # evict to object store now
POST /v1/collections/{name}/restore         # bring it back
```

The first request against an evicted collection pays the restore latency;
subsequent requests are served from RAM as usual. Library-level APIs
(`EvictCollection`, `SweepCold`) are described in
[Persistence](../vector/persistence.md#cold-tier-offload-idle-collections-to-object-storage).
