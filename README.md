<img src="docs/assets/logo.svg" alt="" width="64" height="64">

# Rostam

[![CI](https://github.com/rostamlabs/rostam/actions/workflows/test.yml/badge.svg)](https://github.com/rostamlabs/rostam/actions/workflows/test.yml)
[![Go 1.26+](https://img.shields.io/badge/go-1.26%2B-00ADD8)](go.mod)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

**A high-performance vector database and sub-microsecond key-value store in a
single Go engine — run it as a standalone server, replicate it across a Raft
cluster, or embed it directly in your binary with no server at all.**

At matched recall it serves **~2× the queries of Milvus and pgvector** and
**~4× Qdrant**, with the **fastest load in the set** (1M × 768d in 282 s) and
the highest recall measured. → [How it compares](#how-it-compares)

**Three ways to run it, one engine:**

| | |
|---|---|
| **Standalone server** | REST, gRPC and a binary TCP protocol. Talk to it from Python or any language. |
| **Replicated cluster** | Per-shard Raft, online resharding, backups to S3, RBAC/JWT/mTLS. |
| **Embedded library** | Import it into a Go binary. No server, no cgo required. |

## Agent memory over MCP

`rostam-server mcp` runs Rostam as an [MCP](https://modelcontextprotocol.io/)
stdio server, giving Claude Code, Claude Desktop, Cursor, or any MCP client
persistent agent memory and vector-DB tools. No daemon, no cloud account, and
no embedding API key required — memory works out of the box on built-in
BM25, and pointing it at an OpenAI-compatible endpoint upgrades recall to
hybrid dense+BM25.

```sh
claude mcp add rostam -- rostam-server mcp
```

→ [MCP server](https://docs.rostamlabs.com/server/mcp/) for the full tool
reference, client config snippets, and embedder configuration.

## LLM response caching

`rostam-server llm-proxy` is an OpenAI-compatible caching reverse proxy:
change one line (`base_url="http://localhost:8484/v1"`) and an eligible
repeated chat-completions request — cacheable shape, matching scope, and a
prompt that hits the cache — is answered from a local semantic cache instead
of hitting the upstream API. Exact byte-identical matching works out of the
box with no embedding key; pointing it at an OpenAI-compatible embedder
upgrades it to matching near-duplicate prompts too.

```sh
rostam-server llm-proxy
```

→ [LLM caching proxy](https://docs.rostamlabs.com/server/llm-proxy/) for
flags, cache-scoping rules, and what's never cached.

## RAG in a box

`rostam-server rag` chunks and indexes your documents, then answers
questions from them with cited sources — no separate vector-store setup or
ingestion pipeline required:

```sh
rostam-server rag ingest ./docs
rostam-server rag ask "How does the LLM proxy decide what's cacheable?"
```

Retrieval works out of the box on BM25 full-text search; pointing it at an
OpenAI-compatible embedder upgrades it to dense+BM25 hybrid fusion by default
(`-no-hybrid` selects pure dense search). `rag query` does retrieval alone,
with no LLM required.

→ [RAG CLI](https://docs.rostamlabs.com/server/rag/) for flags,
embedded-vs-remote mode, and supported file types.

And two engines, usable together or entirely on their own:

- a **vector search engine** ([`vector/`](./vector)) — HNSW, IVF and Vamana
  indexes, SQ8/binary/PQ quantization, hybrid dense+sparse fusion, BM25 full
  text, and metadata filtering with an exact filter-first query planner that
  does not fall off a recall cliff. Depends on nothing else in the repo.
- a **key-value store** — an in-process session cache, a replicated store, or a
  TCP server, all behind one `Store` interface, with server-side custom ops and
  sandboxed WASM stored procedures.

Both are built for the latency-sensitive end of the spectrum: reads are
zero-copy and lock-light, the hot paths are allocation-free, and the distance
kernels are hand-vectorized.

> **Status:** Beta. Actively developed and tested (race-clean, benchmarked).
> APIs may still change ahead of a 1.0 release.

📚 **[docs.rostamlabs.com](https://docs.rostamlabs.com/)** — quickstart, concepts,
guides, and full API references (HTTP, gRPC, Go, Python). The site is built from
the [`docs/`](./docs) tree in this repository ([`mkdocs.yml`](./mkdocs.yml)), so
it can also be read here or served locally.

---

## How it compares

Under [VectorDBBench](https://github.com/zilliztech/VectorDBBench) — a neutral,
third-party harness — on Cohere-1M (768d, cosine). Six engines, 27 cases, one
continuous session, with HNSW `m=16` / `ef_construction=200` / `k=100` pinned
identically on every engine and `ef` swept.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench/pareto-dark.svg">
  <img alt="Recall versus queries per second for Rostam, Milvus, pgvector, Weaviate, Qdrant and Redis. Rostam's curve sits above every other engine across the full recall range and reaches the highest recall measured, 0.9978." src="docs/assets/bench/pareto-light.svg" width="100%">
</picture>

Read at **matched recall**, which is the only like-for-like comparison — engines
land at different recall for the same `ef`, and a segmented engine runs that `ef`
against *every* segment, so equal `ef` is not equal work:

| Matched recall | Rostam | vs Milvus | vs pgvector | vs Weaviate | vs Qdrant | vs Redis |
|---|--:|--:|--:|--:|--:|--:|
| 0.95 | **3,161 QPS** | 1.84× | 1.83× | 3.25× | — | 6.54× |
| 0.97 | **2,642 QPS** | 2.02× | 2.18× | 3.03× | 4.16× | 7.53× |
| 0.98 | **2,088 QPS** | 1.90× | 2.28× | 2.67× | 3.98× | 7.90× |
| 0.99 | **1,471 QPS** | 1.78× | — | — | 3.55× | — |

The lead is stable across the whole curve, so it does not depend on picking a
favourable operating point. Rostam also **loads fastest** (282 s for 1M × 768d)
while spending the **least total CPU** of any multi-core engine in the set, and
reaches the **highest recall measured** (0.9978).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench/load-cpu-dark.svg">
  <img alt="Scatter of load wall-clock against CPU seconds spent. Rostam is alone in the fast-and-cheap corner; Redis uses a single core and takes 4.8 times as long." src="docs/assets/bench/load-cpu-light.svg" width="100%">
</picture>

Two caveats that bound all of this, and they are not footnotes: every number is
**same-session**, because this hardware drifts up to **42%** between sessions on
unchanged code; and the benchmark client shares the box, which penalises the
*fastest* engine hardest — so the throughput figures are **floors, not ceilings**.
Floors in one more way: the sweep runs over Rostam's JSON wire, and at the
`ef=300` point its binary query framing measures **+16.8% QPS and −19%
single-client p99** on the same box and corpus. That is one point on the curve,
not a uniform uplift — the wire saves a fixed cost per query, so its share
shrinks as the search itself gets more expensive.

Full per-ef curves, per-engine CPU accounting, the filter case, four paired A/B
controls and the complete methodology (including where the data cuts *against*
Rostam) are in
[**`rostam-bench/vectordbbench`**](https://github.com/rostamlabs/rostam-bench/tree/main/vectordbbench#results).

### And the key-value side

Rostam is two engines, so the KV half gets measured the same way — seven engines
over the wire, one machine, one session:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench/local-get-fast-dark.svg">
  <img alt="GET throughput against concurrency for Rostam, Memcached, Dragonfly and Aerospike. Memcached leads at 8 connections; Rostam pulls ahead from 64 onward and holds roughly 730k operations per second." src="docs/assets/bench/local-get-fast-light.svg" width="100%">
</picture>

| GET ops/s | **Rostam** | Memcached | Dragonfly | Aerospike | Valkey | Redis | KeyDB |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 64 conns | **726.1k** | 682.9k | 532.0k | 510.6k | 239.1k | 228.3k | 206.3k |
| 256 conns | **732.4k** | 681.7k | 551.6k | 516.9k | 227.9k | 219.6k | 208.8k |

p99 at 512 connections: Rostam **2.02 ms**, best of the seven (Memcached 2.32,
Aerospike 2.75, Redis 4.69). Memcached wins at 8 connections; Rostam pulls ahead
as concurrency rises.

**Replicated writes (RF=2)** against Aerospike CE 8.1 at **matched commit
semantics** — both acking only after the replica has the write:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench/rf2-commit-all-dark.svg">
  <img alt="Replicated write throughput against concurrency at RF=2 with replica-ack. Aerospike is slightly ahead at 8 connections; Rostam overtakes at 32 and reaches 113.6k against Aerospike's 85.9k at 128, while Redis, Valkey and KeyDB flatten together near 43k." src="docs/assets/bench/rf2-commit-all-light.svg" width="100%">
</picture>

| RF=2, replica-ack | 8 conns | 32 conns | 128 conns | p50 @128 |
|---|--:|--:|--:|--:|
| **Rostam PB** | 35.6k | **80.9k** | **113.6k** | **0.99 ms** |
| Aerospike CE | **38.0k** | 57.0k | 85.9k | 1.35 ms |
| Redis 7 (`WAIT 1`) | 28.7k | 43.2k | 42.9k | 2.95 ms |
| Valkey 8 (`WAIT 1`) | 28.0k | 42.2k | 45.9k | 2.72 ms |

**Rostam's lead is a scaling property.** Aerospike wins at 8 connections — and
has the better tail there — while Rostam overtakes at 32 and leads **+42% / +32%**
at 32/128, with the better median from 32 upward. The Redis-protocol engines
flatten above 32 connections, which is a per-instance ceiling. Relaxing Rostam
and Aerospike to ack-at-master keeps the shape (157.4k against 125.7k at 128).

Measured twice on the same box four days apart; Rostam reproduced within ~1–4% at
the anchor points.

Over a real network on separate nodes it tightens to near-parity, with Rostam's
tail still ahead (p99 ~8 ms vs 17–22 ms). Both runs, the full posture matrix, and
why a single load generator invalidates this comparison entirely are in
[**`rostam-bench/netkv`**](https://github.com/rostamlabs/rostam-bench/tree/main/netkv).

---

**Two ways to use it.** Run the server and talk to it over REST, gRPC or a
binary TCP protocol from any language — or embed the Go library directly, with
no server in the picture. Start below; the Go embedding path is
[further down](#quick-start--vector-search-go-library).

## Install

```sh
# Prebuilt binary — verifies the release checksum before installing
curl -fsSL https://raw.githubusercontent.com/rostamlabs/rostam/main/install.sh | sh

# ...to a system path instead (prompts for sudo; ROSTAM_NO_SUDO refuses)
curl -fsSL https://raw.githubusercontent.com/rostamlabs/rostam/main/install.sh \
  | ROSTAM_INSTALL_DIR=/usr/local/bin sh

# Container (amd64 + arm64)
docker run -p 8080:8080 -e ROSTAM_API_KEY=secret ghcr.io/rostamlabs/rostam:latest

# Go toolchain
go install github.com/rostamlabs/rostam/cmd/rostam-server@latest   # the server
go get github.com/rostamlabs/rostam                                # the library

# Python client
pip install rostam-client
```

Prefer not to pipe a script into a shell? Download the archive and
`checksums.txt` from the [latest release](https://github.com/rostamlabs/rostam/releases/latest),
then verify before extracting — that is all the script does, plus picking the
right file for your platform:

```sh
sha256sum -c checksums.txt --ignore-missing        # Linux
shasum -a 256 -c checksums.txt --ignore-missing    # macOS
```

The release binaries are **pure-Go builds**: everything works except WASM stored
procedures, which return `wasm: stored procedures require a cgo build`. The
container image always includes them. `go install` includes them only when it
builds with cgo — a native build (cgo defaults off when cross-compiling) on a
machine with a C compiler.

Local in-process embeddings (`-tags localembed`) are a cgo feature too, and need
an ONNX Runtime shared library at runtime, so they are **not** in the pure-Go
binaries either. Two supported ways to get them:

```sh
# Opt-in container image (linux/amd64) — bundles ONNX Runtime, self-contained
docker run -p 8080:8080 -e ROSTAM_API_KEY=secret \
  -e ROSTAM_EMBED_LOCAL=minilm-l6-v2 -v rostam-models:/models \
  ghcr.io/rostamlabs/rostam:localembed

# Or build from source with the tag (needs ONNX Runtime >= 1.29.0 installed)
CGO_ENABLED=1 go install -tags localembed \
  github.com/rostamlabs/rostam/cmd/rostam-server@latest
```

See [Local embeddings](docs/server/mcp.md) for the model catalog and configuration.

## Quick start — run the server

```sh
go run ./cmd/rostam-server -http 127.0.0.1:8080 -data ./data
```

With no authenticator configured the server **refuses to bind a reachable
address** rather than serve an open datastore to the network — so a bare
`:8080` will not start. To listen beyond loopback, give it auth (`-api-key` or
`-keys-file`), or pass `-insecure` to run open deliberately.

```sh
curl -s localhost:8080/v1/collections \
  -d '{"name":"docs","config":{"dim":4,"metric":"cosine"}}'
curl -s localhost:8080/v1/collections/docs/points \
  -d '{"id":1,"vector":[0.1,0.2,0.3,0.4],"content":"hello rostam","upsert":true}'
curl -s localhost:8080/v1/collections/docs/points/search \
  -d '{"query":[0.1,0.2,0.3,0.4],"k":3}'
```

### Or in a container

The image binds `0.0.0.0`, so it **requires authentication** — pass the token by
environment variable, which keeps it out of the process table and out of
`docker inspect`:

```sh
docker build -f cmd/rostam-server/Dockerfile -t rostam-server .
docker run -p 8080:8080 -e ROSTAM_API_KEY=secret rostam-server

curl -s localhost:8080/v1/collections -H 'Authorization: Bearer secret' \
  -d '{"name":"docs","config":{"dim":4,"metric":"cosine"}}'
```

`GET /v1/health` and `/v1/ready` stay auth-exempt, so orchestrator probes work
without the token. For a deliberately open dev container, append `-insecure`.

Clustering (per-shard Raft, online resharding), auth (RBAC/JWT/mTLS), TLS,
backups to S3, and Prometheus metrics are flag-driven.
→ [Running the server](https://docs.rostamlabs.com/server/running/) ·
[Clustering](https://docs.rostamlabs.com/server/clustering/) ·
[Security](https://docs.rostamlabs.com/server/security/) ·
[HTTP API reference](https://docs.rostamlabs.com/api/http/)

## Quick start — Python

```sh
pip install rostam-client   # stdlib-only; optional [langchain] [llamaindex] [haystack]
```

One class, `Rostam(target)`, picks the transport from the target string: an
`http(s)://` URL speaks REST, a `tcp://host:port` (or bare `host:port`)
speaks the native binary protocol. Vector methods are identical either way:

```python
from rostam import Rostam, filters as f

c = Rostam("http://localhost:8080")   # REST, port 8080
c.create_collection("docs", dim=768, full_text=True)
vec = [0.0] * 768  # replace with a real embedding
c.upsert("docs", 1, vec, content="hello", metadata={"tenant": "acme"})

# Dense kNN, hybrid dense+sparse, or BM25 full text — one call each.
hits = c.hybrid_text("docs", dense=vec, text="hello", k=5)

# Metadata filtering goes through the exact filter-first planner, so a
# selective filter does not degrade recall.
hits = c.search("docs", vec, k=10, filter=f.eq("tenant", "acme"))
```

Point the same class at the `-tcp` port instead for the native binary
protocol — same flat vector API, plus `r.kv.*` for the key-value store, which
has no REST surface:

```python
r = Rostam("tcp://localhost:7000")    # or Rostam("localhost:7000") — bare host:port defaults to TCP

r.create_collection("docs", dim=768, full_text=True)
vec = [0.0] * 768  # replace with a real embedding
r.upsert("docs", 1, vec, content="hello", metadata={"tenant": "acme"})
hits = r.hybrid_text("docs", dense=vec, text="hello", k=5)

r.kv.put("user:42", b'{"coins":100}', ttl_ms=300_000)   # TCP-only
r.kv.incr("views:42", 1)                                  # atomic; missing key counts as 0
```

Working against one collection? `r.collection(name)` binds it so you stop
repeating the name (mirrors the Go client's `client.Collection`):

```python
docs = r.collection("docs")
docs.upsert(1, vec, content="hello")
hits = docs.hybrid_text(dense=vec, text="hello", k=5)
```

Bulk ingest (`bulk_stage` + `bulk_build`, HTTP-only) ships vectors as raw f32
over a binary wire rather than JSON text, which is what makes large loads
fast — a 1M × 768d load runs in **282 s**.
→ [Python client docs](https://docs.rostamlabs.com/api/python/)

Other languages talk to the same server over
[REST](https://docs.rostamlabs.com/api/http/) or [gRPC](https://docs.rostamlabs.com/api/grpc/).

---

## Quick start — vector search (Go library)

Embedding the engine in a Go process: no server, no cgo, no other dependencies:

```go
import "github.com/rostamlabs/rostam/vector"

col, _ := vector.NewCollection("docs", vector.Config{
    Dim:    768,
    Metric: vector.Cosine,
    Quant:  vector.QuantSQ8, // 4× smaller, ~98% recall retained
})
defer col.Close()

// Insert is create-only (ErrDuplicateID on a live id); Upsert replaces.
_ = col.Insert(1, embedding, 0, vector.Metadata{
    "tenant": vector.NewString("acme"),
}, nil)

// Exact, fast filtered search — the payload index narrows to tenant=acme.
hits, _ := col.SearchFiltered(query, 10, vector.Filter{
    Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme"),
})

// Diversified retrieval for RAG:
diverse, _ := col.SearchMMR(query, 10, vector.MMROpts{Lambda: 0.5})
```

Beyond kNN: hybrid dense+sparse fusion (RRF/weighted/DBSF), BM25 full-text,
recommendation (± examples), discovery, group-by, scroll with order-by, IVF and
Vamana indexes, binary/PQ quantization with mmap-resident codes, per-collection
quotas and TTL. → [Vector engine docs](https://docs.rostamlabs.com/vector/collections-and-indexes/)

## Quick start — key-value store (Go library)

```go
import (
    "github.com/rostamlabs/rostam"
    "github.com/rostamlabs/rostam/ops"
)

reg := ops.NewRegistry()
_ = ops.RegisterBuiltins(reg) // get / put / del / incr / expire / vector ops (Ops is required)
store, err := rostam.NewDirect(rostam.DirectConfig{Ops: reg /* DataDir: "..." */})
if err != nil {
    log.Fatal(err)
}
defer store.Close()

_ = store.Put(ctx, []byte("user:42"), []byte(`{"coins":100}`), 5*time.Minute)
v, _ := store.Get(ctx, []byte("user:42"))

// Atomic, server-side updates — one round trip, serialized per shard:
_, _ = store.Call(ctx, "incr", ops.EncodeIncrArgs([]byte("views:42"), 1))
_, _ = store.Call(ctx, "expire", ops.EncodeExpireArgs([]byte("user:42"), time.Hour))
```

Swap the constructor for a different backend — same `Store` interface:

| Constructor | Backend | When |
|---|---|---|
| `rostam.NewDirect` | in-process, no Raft | single node, library, fastest |
| `rostam.NewEmbedded` | in-process + per-shard Raft | replicated / multi-node durability |
| `rostam.NewClient` | TCP client to a remote cluster | talk to a running server |

You can also register **your own Go functions as server-side ops** (atomic
read-modify-write under the shard lock, no CAS loops) or ship sandboxed,
fuel-capped **WASM procedures** over the wire to a running cluster.
→ [KV docs](https://docs.rostamlabs.com/kv/overview/) · [Custom ops](https://docs.rostamlabs.com/kv/custom-ops/) ·
[WASM](https://docs.rostamlabs.com/kv/wasm/)

---

## Architecture

```
        vector/   ← standalone vector engine (usable on its own)
        HNSW · IVF · Vamana · GPU · quantization (SQ8/BQ1/PQ/PRQ, mmap) ·
        hybrid sparse+dense · BM25 · payload index + planner ·
        MMR/recommend/discover · tenants/auth/quotas

        rostam.Store  (Direct │ Embedded │ Client)      ← unified KV facade
              │
   ┌──────────┼───────────────┬──────────────────┐
 cache/     shard/+raft/     server/+client/     wasm/
 slab pool  per-shard        TCP protocol +      wasmtime UDF
 TTL, mmap  Raft + ops       smart routing       runtime
 warm-start registry
```

`vector/`'s only in-repo dependencies are the stdlib-only `objstore/` package
and its own `analysis` subpackage — vendor those three as a pure vector
library. → [Architecture](https://docs.rostamlabs.com/concepts/architecture/)

## Micro-benchmarks

The head-to-head comparisons are [at the top](#how-it-compares). These are
in-process numbers for both engines, plus accuracy properties that hold
independently of the machine.

**Accuracy properties** — these are properties of the algorithms, not of the
machine, so they hold anywhere:

| | Result |
|---|---|
| SQ8 quantization (4× smaller) | recall@10 ≈ 0.98 of exact |
| Binary quantization (32× smaller) | recall@10 ≈ 0.96 of exact (rescored) |
| Selective filtered search | exact filter-first path — no recall cliff |
| int8 (SQ8) distance kernel, AVX2 vs scalar | ~2.9–3.3× (768d–1536d) |
| int8 distance kernel, VNNI vs AVX2 | ~1.3× |
| float32 distance kernels, AVX2 vs scalar | ~12–14× |

**Latencies** (`make bench`, in-process unless noted):

| | Result |
|---|---|
| KV Get (hit), `Direct` | **~29 ns/op** |
| KV Put, `Direct` | ~240 ns/op |
| KV Get/Put, `Embedded` (Raft, no-sync) | ~222 ns / ~12.7 µs |
| Search, SIFT-1M, recall@10 0.968 | 173 µs p50 / 60.4k QPS |
| KV Get/Put over TCP loopback (`Direct` server) | ~1.7 µs / ~1.8 µs |

> Measured on the same 12-core AMD EPYC Genoa the VectorDBBench comparison above
> runs on, so the two are directly comparable. **The KV figures are parallel
> benchmarks at `GOMAXPROCS=12`** — per-op cost scales with core count, so a
> wider machine reports a lower number for identical code. The loopback row is
> the one exception: that box's NIC has a single queue and would measure the
> adapter rather than the storage path, so it comes from a multi-queue host.
> Re-run `make bench` on your own machine before sizing anything.

Reproducible head-to-head comparisons against Redis, Aerospike, Qdrant, Milvus,
pgvector, and in-memory Go caches live in the separate
[**`rostam-bench`**](https://github.com/rostamlabs/rostam-bench) repo, so this
module stays dependency-light. → [Performance](https://docs.rostamlabs.com/performance/)

## Examples

- [`examples/semantic-search`](./examples/semantic-search) — end-to-end RAG
  pipeline: OpenAI embeddings → upsert → dense vs hybrid search.
- [`examples/filtered-recall-cliff`](./examples/filtered-recall-cliff) — why
  the filter-first planner exists, measured.

## Development

Requires Go 1.26+. mmap persistence and the AVX2 kernels are Linux/amd64;
everything has a portable fallback. The full build needs cgo (wasmtime); the
vector/cache/ops packages build with `CGO_ENABLED=0`.

Two optional build tags add cgo-only functionality, compiled out by default:

| Tag | Enables | Requires |
|---|---|---|
| `-tags cuda` | GPU exact-KNN index | cgo + a CUDA toolchain |
| `-tags localembed` | In-process ONNX embeddings for the MCP server's memory and generic vector-DB tools, no cloud API | cgo + ONNX Runtime 1.29.0+ ([docs](https://docs.rostamlabs.com/server/mcp/#local-embeddings-tags-localembed)) |

```sh
make test    # tests           make race   # race detector
make bench   # benchmarks      make all    # lint + tests + race + bench
```

→ [Development](https://docs.rostamlabs.com/development/) · [CONTRIBUTING.md](./CONTRIBUTING.md) ·
[SECURITY.md](./SECURITY.md)

## License

[Apache License 2.0](./LICENSE) (`Apache-2.0`), © 2026 RostamLabs.

Rostam is open source. Use it, embed it, modify it, run it in production, offer
it as a service. Apache-2.0 also grants a patent licence explicitly — one that
terminates if you sue someone claiming Rostam infringes a patent.

Redistributing it asks a little more than attribution: ship a copy of the
licence and [`NOTICE`](./NOTICE), keep the notices already in the files, and
mark the files you changed. [`LICENSE`](./LICENSE) is the authoritative text —
the paragraph above is a summary, not terms.

"Rostam" and the Rostam logo are trademarks of RostamLabs; the licence grants no
trademark rights.
