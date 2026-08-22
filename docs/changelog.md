# Changelog

Notable user-visible changes. Entries that alter existing behaviour are marked
**Breaking** and say what to do about it.

## Unreleased

- **The Python client (`rostam-client` on PyPI) unifies on a single `Rostam`
  class — v0.2.0 (Breaking).** The Python client is versioned and released
  independently of the server/project (this changelog's `v0.3.0` entry
  below) — `0.2.0` is the client's own version, not this project's.
  `RostamClient` (HTTP) and the native-TCP `Rostam`/`RostamKV` pair used to be
  two classes with two different vocabularies for the same server — one flat,
  one nesting vector ops under `.vector`. Both collapse into one
  `Rostam(target)`: the transport is chosen from the target string —
  `http://`/`https://` for REST, `tcp://host:port` or a bare `host:port` for
  the native binary protocol — and the vector API is flat (`r.search`,
  `r.upsert`, `r.hybrid_text`, ...) on both transports. Key-value operations
  move to `r.kv.*` and stay TCP-only: on an HTTP-connected client, any `r.kv`
  operation (e.g. `r.kv.get(...)`) raises `TransportError` — reading `r.kv`
  itself does not raise — as does any other op with no equivalent on the
  connected transport (the general `query()` is HTTP-only; TCP callers use
  `recommend()` instead). `RostamClient` and `RostamKV` are removed, not
  deprecated — importing either now raises `ImportError`. Note `r.get` is
  REPURPOSED, not preserved: on the old native `Rostam` class it was a KV
  read; on the unified client it means vector point-get
  (`r.get(collection, id, ...)`). KV reads move to `r.kv.get(...)`.

  | Before | After |
  | --- | --- |
  | `RostamClient("http://host:8080")` | `Rostam("http://host:8080")` |
  | `Rostam("host", 7000)` (native) | `Rostam("tcp://host:7000")` or `Rostam("host:7000")` |
  | `r.vector.hybrid_text(...)` | `r.hybrid_text(...)` (flat) |
  | `r.get("k")` (KV read) / `RostamKV` | `r.kv.get("k")` — `r.get` now means `r.get(collection, id, ...)` (vector point-get) |
  | `query(prefetch=...)` on the native client | HTTP-only — raises `TransportError` on TCP; TCP callers use `recommend()` |

- **Local ONNX embeddings (opt-in `-tags localembed`).** Rostam can now generate
  semantic embeddings in-process from a catalog of downloadable models
  (`minilm-l6-v2`, `bge-small-en-v1.5`, `gte-small`) with no cloud API. Select
  one with `ROSTAM_EMBED_LOCAL=<name>`; weights download to `~/.rostam/models`
  on first run and are checksum-verified. Requires ONNX Runtime installed; the
  default build is unchanged. See `docs/server/mcp.md`.
- **768-dim "base" tier for local embeddings.** The `-tags localembed` catalog
  adds three higher-quality 768-dim models: `bge-base-en-v1.5`, `gte-base`, and
  `all-mpnet-base-v2`. Select one with `ROSTAM_EMBED_LOCAL=<name>`.
- **Prebuilt local-embedding image.** Releases now publish an opt-in
  `ghcr.io/rostamlabs/rostam:localembed` image (linux/amd64) that bundles ONNX
  Runtime, so local embeddings work without building from source or installing
  ONNX Runtime yourself. The default image stays lean and ONNX-free.

## v0.3.0 — 2026-08-16

### The Python native client reaches vector parity over TCP

`Rostam(...).vector` now speaks the full set of point, search, hybrid, and
recommend ops over the native TCP protocol, not just the basics. Added:
`get_batch`, `scroll`, `search_docs`, `search_groups`, `hybrid_search`,
`hybrid_text`, `recommend`, `query`, and `upsert_batch` — so hybrid (BM25 +
dense) search and recommendations no longer require the HTTP client. Every
request is byte-identical to the Go encoder (verified by a cross-stack golden
oracle) and every method is exercised against a live server. `recommend`/`query`
currently encode the recommend-shaped query spec; a general multi-leaf query
encoder is future work.

### A RAG CLI with no pipeline to assemble

`rostam-server rag` turns the engine into a retrieval-augmented-generation tool
you can point at a directory:

```sh
rostam-server rag ingest ./docs
rostam-server rag query "How does the LLM proxy decide what's cacheable?"
```

`ingest` chunks and indexes every recognized file into a local corpus
(`./.rostam-rag` by default) — recognized meaning a known text extension whose
contents are valid UTF-8; anything else is skipped and reported, and a source
that *becomes* invalid has its stale chunks purged rather than left to be
returned by searches. `query` returns the matching chunks with a
`source#chunk-index` and score, and `ask` sends them to an LLM and prints an
answer that cites the chunks it used. There is no separate ingestion service
and no vector store to stand up first.

As with the MCP server, it does something useful before you configure anything:
retrieval runs on BM25 with no embedder, no API key and no network. Pointing it
at an OpenAI-compatible embedding endpoint upgrades retrieval in place, without
changing the commands. Full flag and file-type reference:
[RAG CLI](server/rag.md).

### Retrieval fuses dense and BM25 by default

With an embedder configured, `rag query` and `rag ask` now run **both** a dense
KNN search and a BM25 search and fuse the two ranked lists, rather than
searching dense alone. Fusion is reciprocal-rank by default, with a weighted
rank-blend available; chunks are deduplicated on `source#index`, and ties break
in a fixed first-seen order so the same corpus and query always produce the same
output.

This matters because the two lanes fail differently: dense retrieval misses
exact identifiers, error strings and flag names that a user copies verbatim into
a question, and BM25 misses paraphrase. Fusing them covers both without asking
the user to know which kind of question they are asking.

`-no-hybrid` restores pure dense KNN. Note the returned `Score` is the fusion
score, not the original per-lane score — they are not comparable across modes.

### A native-protocol Python client for KV and the vector database

Two things reachable from Go's remote client but not Python have both landed:
the key-value store (never on REST — it lives only on the binary TCP protocol,
built for sub-microsecond ops an HTTP round trip would defeat) and the same
transport for vector operations. The new `Rostam` client speaks that protocol
directly, standard library only:

```python
from rostam import Rostam, filters

r = Rostam("127.0.0.1", 7000)          # the server's -tcp port

# key-value
r.put("user:42", b'{"coins":100}', ttl_ms=300_000)
r.incr("views:42", 1)                  # atomic; missing key counts as 0

# vector database, same connection
r.vector.create_collection("docs", dim=768, metric="cosine")
r.vector.upsert("docs", 1, embedding, content="hello", metadata={"tenant": "acme"})
r.vector.search("docs", query, k=10, filter=filters.eq("tenant", "acme"))
```

KV covers `get`/`put`/`delete`/`incr`/`expire` (plus a `ping` heartbeat);
`.vector` covers `create_collection`, `upsert`, `insert`, `search`, `get`,
`delete` and `exists`. `auth_token=` rides the protocol-v2 frame when the server
requires one.
`RostamKV` remains as an alias for the KV-focused name.

The vector arg layouts — including create_collection's fragile config trailer —
are differential-tested byte-for-byte against the Go encoders (a Go oracle emits
golden hex the Python encoders must reproduce); the JSON-carrying parts
(metadata, filter, content) round-trip through a real server. Custom ops and
WASM procedures stay Go-only (per-op encoders). Pairs with the container serving
the TCP port by default.


### The container serves the Go client out of the box

The published image bound only REST (`:8080`), but the Go remote client
(`rostam.NewClient`) speaks the binary TCP protocol and nothing else — so a Go
service could not reach the container without overriding its command. The
default now binds REST **and** TCP (`:7000`) and documents both with `EXPOSE`.

Nothing new is published to the host by default: a bound port is not a
*published* one, so `docker run -p 8080:8080` still maps only REST to the host,
and a Go deployment adds `-p 7000:7000`. The token guards every transport identically — the startup
gate refuses an unauthenticated non-loopback bind on any of them — so
publishing `7000` is no less safe than `8080`. gRPC stays opt-in
(`-grpc 0.0.0.0:9090`).

### The Python client reaches the Query API

`rostam-client` gains `query()`, `recommend()` and `discover()`. Recommend and
discover have no standalone route on any transport — they are leaves of the
composable Query API — so they were previously unreachable from Python without
hand-building the request. `recommend(collection, positive=[...], k=...)` scores
toward example ids (and away from `negative` ones); `discover(...)` guides a
search with `(positive, negative)` context pairs and an optional anchor; both
take an optional metadata `filter`. `query()` is the general form for
multi-lane fusion or rerank.

### The Go library defaults HNSW parameters

`vector.NewCollection` used to reject a config that left `M`, `EfConstruction`
or `EfSearch` at zero, even though `Config.M` documents "default 16" and the
HTTP and Python layers fill exactly those. The same create therefore succeeded
over the wire and failed from the Go library — including the quickstart's
embedded example. The engine now fills the standard 16 / 200 / 64 when they are
left zero (or any non-positive value), matching the other entry points. Only
the HNSW (default) index type is affected; IVF still requires and validates its
parameters, and `Config.Validate()` is unchanged.

### The MCP server explains itself to the agent

Tool schemas say what a tool does; they do not say *when* to reach for it. So an
agent with memory available would still re-read a file it had already
summarised. The server now returns MCP's top-level `instructions` from
`initialize` — clients inject it into the model's context — and the `remember`
and `recall` descriptions carry the same guidance at the point of use.

The doctrine is short and deliberate: recall at the start of a task before
re-exploring a codebase, namespace per project rather than `default`, one
self-contained durable fact per `remember`, and never store secrets.

### Memory keeps one entry per key for live state

`remember` now takes an optional `key`. Without one, nothing changes — a memory
is still identified by its `(namespace, content)`, so re-remembering an edited
fact leaves the old version behind. With a `key`, the memory is identified by
`(namespace, key)` instead, so re-remembering the same key **replaces** the
prior entry in place. That makes it the right shape for live, in-flight state —
a PR's status, what you're mid-task on — which otherwise piles up as stale
snapshots that a later agent can't tell apart from the current one.

`recall` and `list_memories` now surface each memory's `key` and its `created`
and `updated` times, so a reader can spot a keyed live-state entry and judge its
freshness at a glance; a keyed memory keeps its original `created` across
updates while `updated` moves to now. `forget` can delete by `key` as well as by
id (pass `namespace` + `keys`). The MCP `instructions` and the `remember`
description now teach the pattern: use a key for live state, omit it for durable
facts.

### Fixed: the MCP server reported the wrong version

`initialize` answered with a hardcoded `0.1.0`, so a v0.2.0 binary introduced
itself to Claude Code, Cursor and every other client as 0.1.0 — and every bug
report filed through an MCP client named the wrong release. It now reports the
binary's real version, derived from the same source `-version` uses.

## v0.2.0 — 2026-08-15

### The MCP server: agent memory with nothing to set up

`rostam-server mcp` runs Rostam as a [Model Context
Protocol](https://modelcontextprotocol.io/) server over stdio, giving Claude
Code, Claude Desktop, Cursor or any MCP client persistent agent memory and
generic vector-DB tools:

```sh
claude mcp add rostam -- rostam-server mcp
```

There is no daemon and no account. The process embeds the engine directly and
persists to `~/.rostam/memory`, so the agent has durable memory seconds after
the command above.

**It works with no embedder at all.** `remember` and `recall` run on the
built-in BM25 index, so there is no API key and no external service in the
default path — the usual reason an agent-memory tool cannot be tried in one
command. Pointing it at any OpenAI-compatible `/embeddings` endpoint (OpenAI,
Azure, Ollama, LM Studio, TEI, LiteLLM) upgrades recall to hybrid dense+BM25:
set `ROSTAM_EMBED_ENDPOINT`, `ROSTAM_EMBED_MODEL` and `ROSTAM_EMBED_DIM`
together — the endpoint alone is a configuration error and refuses to start,
naming the variable it is missing. The tools and their call shapes do not
change; only how well results rank.

Five memory tools (`remember`, `recall`, `forget`, `list_memories`,
`list_namespaces`) and four collection tools (`create_collection`, `upsert`,
`search`, `get`) are always registered. `delete` and `delete_by_filter` are
**absent from `tools/list`** unless `-enable-destructive` is passed, rather
than present and refusing — a model cannot call a tool it cannot see.

`-connect host:port` runs the tools against a remote `rostam-server` over the
binary TCP protocol instead of embedding the engine, with the usual
token/mTLS options. Full tool reference and client config: [MCP
server](server/mcp.md).

### The LLM caching proxy

`rostam-server llm-proxy` is an OpenAI-compatible caching reverse proxy: point
an existing OpenAI SDK client at it instead of `api.openai.com`, and a chat
completion that **hits** the cache is answered locally, at no generation cost
and without the round trip. A cacheable request that misses is still forwarded
and still costs what it always did — eligibility decides whether the cache is
consulted, not whether you pay.

Anything the cache cannot answer is forwarded upstream verbatim: every other
`/v1/*` route, and any chat request whose shape or scope makes it uncacheable.
The two exceptions are handled locally rather than passed on — a chat body over
the size limit gets a `413`, and one that will not parse gets a `400`.

Like the MCP server it is useful before you configure anything: with no
embedder, byte-identical prompts are served from cache. With one, it upgrades
to semantic matching, so a reworded or differently-whitespaced prompt *may* hit
as well — subject to the same eligibility and cache-identity rules, and to the
configured similarity threshold.

Chat responses carry an `x-rostam-cache` header (`hit`, `miss`, `uncacheable`,
`bypass`) so it is visible which requests are still costing generation tokens.
Generic passthrough routes such as `/v1/models` deliberately omit it — they
were never cache candidates, and a verdict there would be noise. See [LLM
caching proxy](server/llm-proxy.md).

### Breaking: NaN no longer matches range filters

A point whose numeric payload field is `NaN` previously matched `gte` and `lte`
against **every** bound, because the comparison classified "neither less nor
greater" as equal. It now follows IEEE 754: a NaN operand makes the comparison
unordered, so `gt`, `gte`, `lt` and `lte` are all false — for a NaN field value
and for a NaN bound alike.

`eq`, `ne` and `in` are unchanged (they compare with `==`, under which NaN was
never equal to anything), and `is_null`/`is_empty` are unchanged (a NaN is a
present, non-null value).

**Who is affected:** anyone whose payloads contain NaN, which in practice comes
from a division by zero or a failed numeric parse upstream. Those points will
stop appearing in `gte`/`lte` results.

**What to do:** write a real sentinel value rather than NaN if such points should
remain matchable.

**Why:** the payload index never agreed with the old behaviour — it excluded NaN
from every range posting list — so a filtered search could already return
different rows depending on which query path the planner chose. Making NaN
unordered is what lets the index and the predicate answer the same question, and
it matches Go, Rust, Milvus and Qdrant. See
[Metadata filtering](vector/filtering.md#nan-and-range-comparisons).

### Breaking: three misconfigurations now refuse to start instead of serving

All three used to come up and serve traffic (one printed a warning):

- **An open non-loopback bind with no authenticator.** With neither `-keys-file`
  nor `-api-key` set, binding a non-loopback address is refused at startup,
  because every request would be served unauthenticated. Loopback-only binds —
  the dev default — are unaffected. Configure auth, bind `127.0.0.1`, or pass
  the new `-insecure` flag to run open deliberately (dev/trusted networks only).
- **Inter-node TLS without a CA.** In cluster mode, `-tls-cert` without
  `-tls-ca` encrypted the replication listener while authenticating nobody: any
  peer that completed a TLS handshake could replicate writes. `-tls-ca` is now
  required whenever `-tls-cert` is set in cluster mode.
- **Tenant-scoped keys with `-tenant-isolation` off.** A key carrying a real
  tenant while isolation is off silently crosses tenant boundaries through glob
  scopes; the old startup warning is now a refusal. Enable `-tenant-isolation`,
  or mark deliberately cross-tenant keys with `Tenant: "*"`.

See [Security](server/security.md) for the full model.

### Breaking: WASM registration sends the module out of band

`Store.RegisterWASM` now takes the module bytes as a separate argument and
returns a push report:

```go
pushReport, err := store.RegisterWASM(ctx, reg, moduleBytes)
```

`WASMRegistration` no longer carries the bytes inline — the `Bytes` field is
replaced by a content address computed for you from the module argument — and
`KeyExtractorHandle` is gone: every WASM op uses the standard extractor, and the
field's other value could make replicas silently diverge, so it cannot exist.
Pass the bytes as the new argument and delete any `KeyExtractorHandle`
assignment; the compiler will point at both sites.

Two behaviour changes ride along. Updating a live module's **bytes** is now
supported on replicated stores — the version a shard group executes is bound to
that group's own log, which is what made updates unsafe before — while changing
a registered op's `Kind` is still refused; use a new op name for that. And the
push report is part of the result, not diagnostics: when non-empty it names the
cluster members that did not acknowledge the module bytes and will have to
fetch them on demand.

### Range filters are no longer a throughput cliff

A range predicate over a high-cardinality field (`id >= N` and friends) used to
cost a large multiple of the same search unfiltered, most of it spent building a
candidate set the planner then discarded. Filtered throughput is now close to
unfiltered for high-pass-rate ranges, with no configuration change and no API
change. Two new counters expose which acceleration a filtered search used:
`filter_column_gates_total` and `filter_complement_gates_total`, alongside the
existing `filter_gates_total`.

Numeric range filters may now build a **column sidecar** — one `float64` per slot
per range-queried field, at most eight fields at a time, least-recently-used
evicted beyond that. It is built lazily on the first range query over a field.

The sidecar counts against `MaxBytes`, and **writes always win**: an insert that
would otherwise be refused reclaims the sidecar and proceeds, so query traffic
can never make a collection permanently reject writes. `ErrCollectionFull`
continues to mean what it always meant — the collection's own durable data has
filled its budget. The `filter_column_drops_total` counter reports how often a
write had to reclaim columns; a value that climbs steadily means reads and writes
are fighting over the same bytes and `MaxBytes` wants raising. In the other
direction, a search on a collection with no byte headroom simply skips the
sidecar and uses the slower path.

On low-dimensional collections the sidecar is a meaningful fraction of memory —
a full set of columns is a quarter of the vector data at 64 dimensions and twice
it at 8 — so size `MaxBytes` with that in mind.

### Filtered searches spend less time re-checking the filter

When a filtered search cannot brute-force the candidate set the payload index
narrowed, it falls back to graph traversal — and that traversal used to re-derive
each candidate's filter membership from scratch, a fact the index had already
established. It now folds the same narrowing plan into a membership bitset and
consults one bit per candidate instead. No API change, no configuration, and no
change to which points a filter matches: it is the same plan, read a cheaper way.

### Binary query wire

`/points/search` and `/points/search/docs` now accept a dense binary body,
selected by `Content-Type: application/octet-stream` — the ingest wire's
counterpart on the read side. The query vector travels as raw `f32` instead of
base-10 text, which it turns out is a large share of a small request: at dim=768,
k=10 on a kept-alive connection, building the JSON body measured **0.258 ms of a
0.845 ms search**, against 0.011 ms to write the same vector as bytes. The
server's matching decode of those literals goes with it.

That is the mechanism; the effect on a real corpus was measured separately, on a
dedicated 12-core EPYC Genoa server against Cohere-1M at `k=100`, `m=16`,
`ef_construction=200`, `ef_search=300`. Three same-session pairs, each loading
the corpus once and then running both wires against that one index, gave
**2,718 / 2,739 / 2,777 QPS over JSON against 3,163 / 3,178 / 3,277 over the
binary wire — +16.8%** — with single-client p99 falling from 7.8–7.9 ms to
6.2–6.5 ms. One of the three pairs reverses which arm loads the corpus and which
inherits the warm index, and the advantage does not move, which is what
distinguishes a transport effect from a page-cache one. Recall is identical
within every pair to four decimals: the binary path returns the same answers,
sooner.

Adds no semantics, on the same terms as the bulk wire: a binary body decodes into
exactly the request its JSON body produces, then runs the identical validation
and dispatch, and any other content type takes the JSON path unchanged. The
framing, its limits (`dim` ≤ 65,536, filter ≤ 1 MiB, `NaN`/`Inf` refused) and its
error behaviour are in [Binary query body](api/http.md#binary-query-body).

Clients need no flag day. A server that predates this answers a binary body with
`400 invalid JSON body: ...`, which is specific enough to fall back on; the
Python client does so automatically and permanently.

### The Python client reuses connections

`rostam-client` opened a new TCP connection per request. It now keeps a small
pool, and is still safe to share between threads. Combined with the binary query
wire, repeated searches measured **724–990/s before and 1911–3179/s after** at
dim=768 against the same server — non-overlapping ranges, on a laptop.

That figure is the two changes together and cannot be divided between them; it
is also a laptop measuring a small index, where per-request overhead dominates
in a way it does not at scale. For the query wire on its own, on server hardware
and a 1M corpus, see [Binary query wire](#binary-query-wire) above.

Reads are retried once when a pooled connection turns out to have been closed
while idle; writes are not, because that failure surfaces after the request is on
the wire and a replayed insert is worse than an error. `close()` releases the
pool. Note that `http.client` does not consult `HTTP_PROXY`/`HTTPS_PROXY` the way
the previous `urllib` call did.

### Binary bulk-ingest wire

`/points/bulk` and `/points/batch` now accept a dense binary body, selected by
`Content-Type: application/octet-stream`, carrying vectors as raw `f32` rather
than as base-10 JSON text. On a large initial load the JSON encode/decode — not
the index build — was the larger cost: a 1M × 768d load measured locally at
roughly **1043 s dropped to roughly 508 s**.

This adds no semantics. Each route decodes a binary body into exactly the request
its JSON body produces and then runs the identical code, so results are
unaffected, and any other content type takes the JSON path unchanged. The framing,
its limits, and its error behaviour are in
[Binary bulk ingest](api/http.md#binary-bulk-ingest).

### Bulk ingest carries payloads, so filtered workloads get the fast path

A load that needed metadata on each point previously had exactly one route:
`/points/batch`, one indexed insert per point. The staging path carried ids and
vectors only, so every filtered workload was pushed off the multi-core bulk build
and onto the slow inline route. `/points/bulk` now accepts per-point payloads and
stages them for the same concurrent build. Measured locally at 100k × 768d with a
single scalar payload per point, ingest went from **760 to 4915 vectors/s
(6.46×)**; carrying the payloads costs nothing measurable against a vectors-only
staged load, because they ride a pass that was already visiting every slot.

Content and sparse vectors still have no bulk representation. The staging route
now **rejects** a point carrying `content`, `sparse`, `ttl_ms`, `key_ttl_ms` or
`expected_version` with a 400 rather than staging it with those fields quietly
dropped.

### Queries no longer stall while a collection grows

A collection's per-slot arrays used to be reallocated and copied as it grew, and
a query arriving at a growth boundary waited for the copy. They now grow in place
on a reserved address range, so nothing moves and nothing waits. The effect on
worst-case latency, the virtual-versus-resident memory consequence, and the
platforms that fall back to the old behaviour are covered in
[Growing a collection doesn't stall queries](performance.md#growing-a-collection-doesnt-stall-queries).

### Cluster backups and one-shot restore

`-backup-dir`/`-backup-interval` now cover `-cluster` deployments: each node
streams per-shard artifacts plus the meta catalog, and a fresh same-topology
cluster is restored by starting every node once with `-restore`. A restore
fails loud on a topology mismatch or a missing shard artifact rather than
bringing a shard up empty. See [Backups](server/backups.md).

### A readiness probe that reflects shard leadership

`GET /v1/ready` now answers from cluster state — shard leadership backed by
per-shard replication-lag and ISR-health metrics — rather than from mere
process liveness, so a load balancer stops routing to a node that cannot
actually serve. `/v1/health` remains the liveness check. Both endpoints are
auth-exempt, since an infrastructure probe carries no token.

### Structured logs, request ids, and an opt-in access log

`-log-format json` switches server logs to one JSON object per line (`text`,
the default, keeps the historical stderr format); `-log-level` sets the floor.
`-access-log` emits one structured line per request on every transport — HTTP,
gRPC and TCP — with a request id (an inbound `X-Request-Id` is reused), the op,
status, latency, bytes, and a redacted principal; raw tokens are never logged.
Both new logs are off by default and cost nothing on the hot path when off.

### Experimental: primary-backup replication mode

`-replication-mode=pb` selects a primary-backup/ISR replication engine for
every shard in place of per-shard Raft; the default (`raft`) is byte-identical
to previous behaviour. PB mode requires `-min-isr` and a per-node PB address,
promotes a verified ISR survivor automatically when a primary dies
(`-pb-auto-failover`, default on within pb mode), and offers
`-pb-commit-primary` to trade acked-write durability for lower write latency.
Explicitly experimental — see `shard/pbisr/BENCHMARK.md` for the measured comparison
it must clear.

### Fixed: a point stored under id 0 was never returned by search

A point inserted with id `0` was stored, live, returned by `Get`, and counted by
the live-count metric — but no search of any kind returned it, on any index type.
A collection holding exactly one point, id 0, answered its own vector with an
empty result set. Search now returns it like any other point.

The same absent-versus-zero confusion reached two `/query` discover cases, both
also fixed. A discover query with `"target": 0` silently lost its anchor and was
answered from the mean of its context positives instead of erroring; it now
anchors on point 0 as written. And a context pair specifying one side as a vector
and the other as an id — for example `{"positive_vec": [...], "negative": 5}` —
used to be accepted and searched against an anchor the caller never named. A pair
must now be wholly the id form or wholly the vector form; a mixed pair is
rejected with a 400. If you were sending a half-specified pair, it was not doing
what it appeared to do, and it now fails loudly.

On CUDA builds (`-tags cuda`), the same slot guard ran in the GPU exact-scan
lanes with a second consequence: a slot freed by a delete keeps a stale id,
which the old check waved through, so a **deleted** point could resurface in
GPU search results. The same fix covers both, and is verified on real CUDA
hardware.

### Fixed: a filter on `$content` returned no rows

Any filter naming the `$content` field returned zero results, even when every
point matched — for example `And(Eq("tag", "a"), Match("$content", "quick"))`
over a set in which all points satisfy both clauses. `$content` is deliberately
not indexed, and the query planner read "no index entries" as "no matching
points" and returned early without ever evaluating the predicate. The predicate
is now consulted, and such filters return the rows they always should have. This
affected every filter operator on `$content`, not just text match.

### Fixed: points could become permanently unsearchable under several workload shapes

Under each of these shapes, a handful of points ended up in the collection —
byte-correct, live, returned by `Get` and by scans — yet reachable by no search
at any `ef_search`:

- **The default multi-worker bulk build, and upserts overlapping a reshard**,
  could link a point against a neighbour that had no usable edges yet, leaving
  it with no path back from the entry point.
- **An insert arriving when everything reachable was dead** — every candidate
  the graph traversal could see being deleted or expired, a reshard's ordinary
  lazy-delete band and a natural state of any delete-heavy collection — wrote
  an *empty* edge set for the new point. Worse than orphaning that one point:
  if it drew a taller level it promoted itself to entry point, severing the
  whole live graph behind it. Linking now falls back to structural neighbours
  (still-traversable tombstoned slots) rather than accepting emptiness.
- **Repeated upsert sweeps at low graph degree** (`m` of about 4–6) fed each
  reused slot self-loops and one-way upper edges and discarded the edges the
  slot inherited, thinning the graph below its configured degree with every
  sweep until points fell off it.

All are fixed, and a standing invariant test now requires every live point to
be findable by its own vector.

If you loaded a large collection through the bulk build, ran upsert churn, or
deleted heavily before these fixes, a small number of points may still be
missing from your search results. The stored data is intact — `Get` and scans
return those points — but only a re-load on a fixed build puts them back in the
graph.

### Fixed: a malformed bulk-ingest body could reserve memory before being rejected

The bulk-ingest routes sized their allocations from counts the request itself
declared, before the bytes backing them arrived — and on the JSON path the
encoder ran ahead of the authorization check, so an anonymous caller could
trigger the allocation and only then be told 401. Worse, the JSON path sized
its buffer from the point count and the first vector's length, two numbers the
body chooses independently: a measured 289 KB anonymous request allocated
1.61 GB, and the same shape scaled within the body cap reached an allocation
no process survives. Requests are now authorized before anything is sized by
the body, every declared count and length is bounded before it sizes anything,
and each section of a binary body is read in windows that grow only as bytes
actually arrive. A body that over-declares its size is refused having consumed
only what it really sent.

Two related input errors are now rejected at the edge rather than deeper in:
a bulk-stage batch whose vectors are not all the same length, and a bulk body
whose payload section ends early. Neither was merely an error before. A ragged
batch whose short vector sat mid-batch shifted every following row during
encoding, so ids were reconstructed out of vector bytes — points stored under
ids nobody sent, with no error anywhere. And the early-ending payload section
mattered most on a cluster, where such a body could be replicated before it
was decoded and then bring down every node that applied it.

### Fixed: an upsert into an over-budget collection could resurrect the point it replaced

An upsert reclaims the dead slot of the id it replaces — free it, then
immediately reuse it — and both quota checks sat between those two steps. An
upsert refused for being over `MaxVectors`/`MaxBytes` had therefore already
freed the slot and then abandoned it, with the old id, vector and payload
still in place and still reachable through the graph's in-edges: a search
could return the **deleted** point, scored against its stale vector and
carrying its old payload. A collection over its budget is reachable in the
first place because the bulk load does not consult the quotas.

The quota verdict now precedes the reclaim, judged against the accounting the
reclaim is about to produce — so an upsert into a merely *full* collection is
a replace and still succeeds, while a rejected upsert mutates nothing at all.

### Fixed: loading a snapshot silently dropped most of the collection's config

The snapshot header carries six config fields (dimension, metric, `m`,
`ef_construction`, `ef_search`, seed). Reading a snapshot rebuilt the live
config from those six and zeroed everything else, while `Config()` kept
reporting the original values — so nothing looked wrong. Every path that loads
a snapshot was affected: a restart with persistence enabled, a backup restore,
and a Raft snapshot install. After any of them, `MaxVectors`/`MaxBytes` quotas
were silently lifted, every subsequent insert wrote half the intended level-0
graph degree, a quantized build silently switched to exact-float navigation,
and the `max_ef_search` request ceiling vanished. The stored data was never
harmed; the config now survives the round trip, and only the six
header-carried fields are taken from the stream.

### Fixed: TTL deadlines could differ between replicas

In a cluster, each replica applying a write stamped TTL deadlines from its
**own** clock, so replicas at skewed clocks stored different absolute
deadlines for the same committed write — and could then disagree about whether
a point was live: different search results, and a different insert-if-absent
outcome, depending on the replica asked. Every clock-dependent decision in the
apply path — point TTLs, per-key payload TTLs, expiry-aware liveness for CAS
and reclaim — now uses a single leader-assigned stamp carried with the entry,
across dense, multi-vector and named collections. Single-node behaviour is
unchanged.

### Fixed: a deleted key could come back after a warm restart

In the KV cache, `Del` tombstoned only the in-memory index; nothing was
recorded on the page, so a warm restart rebuilt the index straight off the
bytes and the deleted key returned — and in a cluster the restarted node then
disagreed with its peers forever, because nothing re-applies a delete below
the applied index. Deletes now append a tombstone entry that outranks every
earlier copy of the key in the rebuild.

### Fixed: a handler panic no longer takes down the whole process

There was no `recover()` anywhere in the op path, so a panic in any request
handler — reachable from a malformed frame through the argument decoders —
crashed the entire node: every shard, every connection. Every transport
(HTTP, gRPC, TCP) now recovers handler panics into an `internal error`
response plus a server-side log with the stack; the panic detail never
reaches the client. The same hardening pass made the `-api-key` comparison
constant-time and put body-size, idle and write timeouts on the HTTP server
(the long-running bulk-build route is exempted from the write deadline, which
otherwise cut it off mid-build on any large corpus).
