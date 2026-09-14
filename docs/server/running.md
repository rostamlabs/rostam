# Running the server

`rostam-server` exposes the engine over three transports — REST, gRPC, and a
compact binary TCP protocol — all dispatching into the same store, so semantics
are identical regardless of how you connect.

```sh
# from a repo clone
go run ./cmd/rostam-server -http 127.0.0.1:8080 -data ./data

# or build a binary
go build -o rostam-server ./cmd/rostam-server
./rostam-server -http 127.0.0.1:8080 -grpc 127.0.0.1:9090 -tcp 127.0.0.1:7000 -data ./data
```

!!! warning "A bare `:8080` will not start without authentication"

    With no authenticator configured, every request is served unauthenticated —
    so the server **refuses to bind a reachable address** rather than silently
    exposing an open datastore. A bare `:8080`, `0.0.0.0`, `::` or any
    non-loopback host counts as reachable. To listen beyond loopback, give it
    auth (`-api-key` or `-keys-file`) or pass `-insecure` to run open
    deliberately. Loopback binds are unaffected, which is why the examples above
    spell out `127.0.0.1`.

## Transports & storage

| Flag | Default | Meaning |
|---|---|---|
| `-http` | `:8080` | REST/JSON listen address; `""` disables |
| `-grpc` | disabled | gRPC listen address |
| `-tcp` | disabled | binary TCP protocol listen address (what `rostam.NewClient` speaks) |
| `-data` | in-memory | persistence directory; empty = nothing survives restart |
| `-shards` | auto | cache shards (single node) or Raft shards (cluster) |
| `-ttl-sweep-interval` | `30s` | how often each shard reaps expired TTL keys to reclaim memory; `0` disables active reaping (lazy-on-read expiry still applies) |

Any subset of transports can be enabled. `GET /v1/health` (auth-exempt) is the
liveness probe and `GET /v1/ready` the readiness probe — use the latter for
load-balancer membership, since health stays green on a node that has lost
quorum. `GET /metrics` serves Prometheus ([Monitoring](monitoring.md)).

## Configuring by environment

Every flag has an environment variable: uppercase the name and replace dashes
with underscores, prefixed with `ROSTAM_`.

| Flag | Variable |
|---|---|
| `-http` | `ROSTAM_HTTP` |
| `-api-key` | `ROSTAM_API_KEY` |
| `-pb-auto-failover` | `ROSTAM_PB_AUTO_FAILOVER` |
| `-wasm-blob-retention` | `ROSTAM_WASM_BLOB_RETENTION` |

This is what makes a container or a Kubernetes ConfigMap workable — otherwise
59 flags have to be assembled into one command line.

```sh
docker run -p 8080:8080 \
  -e ROSTAM_API_KEY=secret \
  -e ROSTAM_SHARDS=16 \
  rostam-server
```

**Precedence is `flag` > `ROSTAM_*` > default.** An explicit flag always wins,
so a command line can override a baked-in environment without editing it.

Three details worth knowing:

- **Empty is a value, not an absence.** `ROSTAM_GRPC=""` disables the gRPC
  listener exactly as `-grpc ""` does. A variable that is set-but-empty is
  honoured; only an unset variable falls through to the default.
- **A bad value is fatal.** `ROSTAM_SHARDS=sixteen` exits with
  `ROSTAM_SHARDS="sixteen" is not a valid value for -shards`, naming the
  variable, the value and the flag. It does not silently keep the default —
  that is how a node ends up running a configuration nobody chose.
- **Secrets are the deliberate exception.** For `-api-key` and
  `-internal-token` the *environment* wins over the flag, because a secret on
  the command line is visible to other local users through `/proc` and lands in
  shell history.

## Flag reference by area

**Authentication** — `-api-key` (single superuser key; prefer the
`ROSTAM_API_KEY` env var so the key doesn't show in `/proc`), `-keys-file`
(RBAC key registry), `-internal-token` (inter-node credential; prefer
`ROSTAM_INTERNAL_TOKEN`), `-audit-log`, `-tenant-isolation`,
`-jwt-public-key` / `-jwt-issuer` / `-jwt-audience`. With **no** authentication
configured, the server refuses to start on a non-loopback bind unless you pass
`-insecure` — an open datastore on the network must be a deliberate choice.
→ [Security](security.md)

**TLS / mTLS** — `-tls-cert`, `-tls-key`, `-tls-ca`,
`-tls-require-client-cert`, `-tls-node-cert`, `-tls-node-key`,
`-node-cn-allowlist`. One certificate config covers all three transports.
→ [Security](security.md#tls)

**Storage** — `-data`, `-shards`, `-config` (carries the cache `max_memory`
stanza), `-ttl-sweep-interval`, `-disable-cold-compaction`. The
`-disable-cold-compaction` flag is an escape hatch, not a tuning knob: a
persistent shard rewrites its pages file live-only at open, and that rewrite is
the only thing that reclaims the bytes left behind by overwritten and expired
keys — without it a shard under TTL churn eventually refuses writes. Turn it off
only to work around a problem with the rewrite itself.

`-ttl-sweep-interval` (default `30s`) sets how often each shard actively reaps
expired TTL keys to reclaim memory, independent of whether they are read again —
an expired key is *always* returned as not-found regardless, so this is a
memory-reclaim-latency vs CPU-churn tradeoff, not a correctness knob. `0`
disables active reaping, leaving only lazy-on-read expiry (and, on persistent
shards, cold compaction at the next restart). Lower it below the default if a
write-heavy replicated node is climbing toward the cache cap between sweeps;
raise it to cut background CPU on a cluster with many shards and slow-churning
TTLs.

The cache eviction knobs (`-relocating-eviction`, `-relocate-reserve-interval`,
`-in-place-same-size-update`, `-in-place-seqlock-reads`, `-sieve-visited-bit`)
are also Storage flags, but most of them do nothing on most topologies — see
[Cache eviction knobs](#cache-eviction-knobs) before enabling one.

**Clustering** — `-cluster`, `-node-id`, `-raft-addr`, `-bootstrap`, `-peers`,
`-replication-factor`, `-persistent-vectors`, `-reconfigure`; durability
posture: `-nosync`, `-volatile-log`; replication engine: `-replication-mode`
(`raft` | `pb`, with `-min-isr`, `-pb-addr`, `-pb-commit-primary`,
`-pb-auto-failover`); Raft transport: `-raft-transport`
(`mux` | experimental `fabric`).
→ [Clustering](clustering.md)

**Backups & cold tier** — `-backup-dir`, `-backup-interval`,
`-backup-prefix`, `-backup-retention`, `-backup-bucket`, `-backup-endpoint`,
`-backup-region`, `-backup-tenant`, `-cold-tier-after`, `-s3-path-style`;
`-restore` + `-allow-missing-shards` for one-shot cluster disaster recovery.
→ [Backups & cold tier](backups.md)

**Help & version** — `-version` prints the build identity and exits: a release
tag for a released binary, otherwise the module pseudo-version Go records, with
a `+dirty` marker when the tree had uncommitted changes. `-h` prints the flags
grouped by the areas on this page; `-help-all` prints every description in full.

**Logging** — `-log-format` (`text` | `json`), `-log-level`
(`debug`–`error`), `-access-log` (one structured line per request on every
transport, principal redacted).

## Cache eviction knobs

Five opt-in flags change what a shard does when its cache reaches
`max_memory`. All are **off by default**, and setting them never turns anything
else on. Each also reads its `ROSTAM_*` variable
(`-in-place-same-size-update` → `ROSTAM_IN_PLACE_SAME_SIZE_UPDATE`).

| Flag | What it does |
|---|---|
| `-relocating-eviction` | When a full shard evicts a page, copy the records on it that are still live forward instead of dropping them with the dead versions that share the page. Costs extra copying at eviction time. |
| `-relocate-reserve-interval` | How often each shard tops up a small reserve of free pages for `-relocating-eviction`, so a write at capacity finds room instead of evicting inline (default `50ms`; `0` turns the reserve off). Every shard runs its own ticker, so a shorter interval costs more on a node with many shards. |
| `-in-place-same-size-update` | A rewrite of a key whose new value has the same length overwrites the stored copy instead of appending a new one, so steady same-size rewrites stop filling the cache with dead versions. Reads on the shard then take a read lock (unless `-in-place-seqlock-reads`), and a rewritten key no longer moves to the newest page: on a cache run over capacity, a frequently rewritten key is evicted on the same schedule as a rarely rewritten one. |
| `-in-place-seqlock-reads` | Keeps reads lock-free under `-in-place-same-size-update` by validating each read against a version counter after the fact. See the warning below. |
| `-sieve-visited-bit` | `-relocating-eviction` rescues only records that were read or rewritten since eviction last passed them, instead of whichever live records it meets first, so a stream of write-once keys cannot push the working set out. |

### Where each knob takes effect

Check this before enabling a knob: on the wrong topology it is accepted and
does nothing.

| Flag | Single node, in-memory (no `-data`) | Single node with `-data` | `-cluster` |
|---|---|---|---|
| `-relocating-eviction` | **takes effect** (at eviction and in the background reserve) | **takes effect** (at eviction only) | no effect |
| `-relocate-reserve-interval` | **takes effect** with `-relocating-eviction` | no effect | no effect |
| `-in-place-same-size-update` | **takes effect** | no effect | no effect |
| `-in-place-seqlock-reads` | **takes effect** with `-in-place-same-size-update` | no effect | no effect |
| `-sieve-visited-bit` | **takes effect** with `-relocating-eviction` | **takes effect** with `-relocating-eviction` | no effect |

Why:

- **`-cluster`: none of them.** Every one acts only when a shard evicts to make
  room. A cluster shard never evicts: replicas evicting independently would
  drop different keys and diverge, so replication makes every shard refuse
  writes at capacity instead. (Every cluster shard is also file-backed, since a
  cluster requires `-data`.)
- **`-data`: no in-place updates.** A file-backed page is the durable copy. An
  overwrite interrupted by a crash is found at recovery as a corrupt entry
  mid-page, and recovery discards the rest of that page with it — other keys,
  written long before. An interrupted append normally costs only the write in
  flight, and the key's previous version is still on disk. So a file-backed
  shard always appends, and `-in-place-seqlock-reads`, which exists
  only to serve in-place updates, has nothing to do either.
- **`-data`: no background reserve.** On a file-backed shard the reserve's
  copies and the page it frees could reach disk out of order, so a crash could
  lose a record that would otherwise have survived. Relocation there runs only
  at eviction time.
- **The background reserve also needs at least four pages per shard.** A small
  `max_memory` spread over many `-shards` can leave fewer, and such a shard
  keeps no reserve; `-relocating-eviction` still works, at eviction time only.

The server does not refuse a knob that has no effect — a config file shared by
a cluster and a single-node box is over-specified, not broken — but it logs one
warning per such knob at startup, naming the flag, the deployment and the
reason:

```text
level=WARN msg="cache option is set but has no effect on this deployment" component=cache option=InPlaceSameSizeUpdate flag=-in-place-same-size-update deployment="single-node with a data directory" reason="..."
```

The same warning fires for a pairing that does nothing: `-sieve-visited-bit` or
`-relocate-reserve-interval` without `-relocating-eviction`, and
`-in-place-seqlock-reads` without `-in-place-same-size-update`.

!!! danger "`-in-place-seqlock-reads` is not covered by race detection"

    This is an opt-in performance trade, not a free speed-up. Its reads are a
    **deliberate data race**: a reader copies page bytes that a writer may be
    rewriting at that moment, then discards the copy if a version counter
    moved. Go's race detector reports that access by design, and the engine's
    concurrent tests for this read protocol are skipped under `-race` — so a
    green race-enabled test run says nothing about it. The protocol's
    correctness argument rests on how Go compiles its atomic operations rather
    than on a guarantee of the language's memory model (the full argument and
    its limits are in `cache/seqlock.go`). It pays only when reads far
    outnumber writes; otherwise leave it off and let reads take the lock.

Embedding the engine directly, the same knobs are fields of
`rostam.CacheConfig` with the same names in Go form, and `NewDirect` /
`NewEmbedded` log the same warnings. A `NewEmbedded` store with no `Peers` is
not replicated and behaves as the "single node with `-data`" column.

## Recipes

**Dev server, everything open, in-memory:**

```sh
rostam-server -http 127.0.0.1:8080
```

A loopback bind may run open; a network-reachable bind (e.g. `-http :8080`)
with no authentication is refused at startup unless you pass `-insecure`.

**Single node with persistence and a single API key:**

```sh
ROSTAM_API_KEY=$(openssl rand -hex 32) rostam-server \
  -http :8080 -tcp :7000 -data /var/lib/rostam
```

**TLS on all transports:**

```sh
ROSTAM_API_KEY=$(openssl rand -hex 32) rostam-server \
  -http :8443 -grpc :9443 -tcp :7443 -data /var/lib/rostam \
  -tls-cert server.pem -tls-key server-key.pem
```

(TLS does not substitute for authentication — a non-loopback bind still needs
an API key, a keys file, or an explicit `-insecure`.)

**Three-node replicated cluster:** see [Clustering](clustering.md#bootstrapping-a-cluster).

## Choosing a transport

| Transport | Wire cost | Use when |
|---|---|---|
| HTTP | JSON, easiest | curl, Python client, browsers, most integrations |
| gRPC | protobuf | typed clients, streaming-friendly infrastructure |
| TCP | compact binary, lowest latency | the Go smart client (`rostam.NewClient`), latency-sensitive services |

### The TCP event-loop transport (`-epoll`)

`-tcp` can be served two ways: a goroutine per connection, or an epoll event
loop (`-epoll`, with `-epoll-loops` setting the loop count; `0` = `GOMAXPROCS`).

| Mode | `-epoll` default | Why |
|---|---|---|
| Single node | on | It beats goroutine-per-connection under core pressure — up to ~1.4x at low concurrency on an 8-core co-located box — and is within noise on dedicated cores. |
| `-cluster` | off unless `-epoll` is passed explicitly | The event loop executes dispatch **inline**, so a replicated write blocks the whole loop for a full replication round trip. Write throughput is then capped at ~(loops / RTT) regardless of connection count — measured 2.1x slower than the goroutine server on a real-network 3-node PB RF=2 cluster (61k vs 127k ops/s at 128 connections). |

The event-loop transport is **plaintext only**: with TLS configured the TCP
transport silently falls back to the goroutine server.
