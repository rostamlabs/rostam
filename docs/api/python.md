# Python client

A dependency-free (stdlib-only) REST client for Rostam, with optional
LangChain, LlamaIndex, and Haystack adapters. Requires Python ≥ 3.9.

```sh
pip install rostam-client

# with an adapter
pip install "rostam-client[langchain]"   # or [llamaindex], [haystack]
```

```python
from rostam import RostamClient

c = RostamClient("http://localhost:8080", api_key=None, timeout=30.0)
c.health()  # -> bool
```

All errors raise `RostamError(message, status)` carrying the HTTP status.
Metadata is plain Python dicts — the client converts to/from the server's
tagged value encoding automatically.

!!! info "Two clients, two ports"
    This package ships **two** clients:

    - `RostamClient` (below) speaks **REST over HTTP**, against the server's
      HTTP port (`8080` above).
    - `Rostam` (see [Native TCP client](#native-tcp-client-rostam)) speaks the
      **binary protocol over TCP**, against the server's `-tcp` port (`7000`
      in the examples). It carries both the KV store (which is TCP-only) and
      the full vector API as `r.vector.*` — the same collections, the same
      points, just a different wire.

    Reach for the native client when you want the lower-overhead binary
    protocol or need the KV store; reach for `RostamClient` for a
    dependency-free REST surface (and the LangChain/LlamaIndex/Haystack
    adapters, which are HTTP-only).

## Connections

The client keeps a small pool of connections and reuses them, so a sequence of
calls does not pay a TCP handshake each time. It is safe to share one client
across threads: a connection is held by one thread for the length of a request.

```python
with RostamClient("http://localhost:8080") as c:   # or call c.close()
    c.search("docs", q, k=10)

RostamClient(url, pool_maxsize=8)    # idle connections kept per client
```

`close()` releases pooled connections; the client stays usable and reconnects on
the next call. A server that closes the connection (HTTP/1.0, or
`Connection: close`) is handled — the client simply does not pool those.

A pooled connection can still die while idle, and that is only discovered on the
next request — after its bytes have been sent. **Reads are retried once on a
fresh connection; writes are not**, and surface as `RostamError` with
`status == 0`, because replaying a write could insert twice. Retry those in the
caller, where the intent is known.

Searches are sent in the [binary query framing](http.md#binary-query-body),
which skips encoding the query vector as text; at dim=768 that was 31% of the
request. Against a server too old to understand it, the client detects the
refusal and falls back to the JSON body for the rest of its life. Pass
`binary_search=False` to send JSON from the start.

!!! note "Proxies"
    Connection reuse is built on `http.client`, which — unlike the `urllib`
    call it replaced — does not consult `HTTP_PROXY`/`HTTPS_PROXY`. A client
    behind an egress proxy should point `base_url` at the proxy or at a local
    forwarder.

## Collections

```python
c.create_collection(
    "docs", dim=768,
    metric="cosine",          # "cosine" | "l2" | "dot"
    quant="sq8",              # "" | "sq8" | "bq1" | "pq" | "sq" | "prq"
    index_type="hnsw",        # "" | "hnsw" | "ivf" | "vamana"
    m=0, ef_construction=0, ef_search=0,   # 0 = engine defaults
    persistent=False, rescore_factor=0,
    full_text=True,           # True or {"analyzer": "english", "k1": 1.2, "b": 0.75}
)
c.drop_collection("docs")
```

Advanced index knobs (`sq_bits`, `prq_layers`, `pq_nbits`, `vamana_r/l/alpha`,
`anisotropic_eta`, `soar`, `soar_lambda`, `seed`) pass straight through to the
[collection config](http.md#collections-dense).

## Writing points

```python
c.upsert("docs", 1, vec, content="document text",
         metadata={"tenant": "acme", "year": 2026},
         ttl_ms=0, sparse=None)              # insert-or-replace
c.insert("docs", 2, vec)                     # create-only: duplicate id raises
c.delete("docs", 2)                          # -> bool (existed)
c.delete_by_filter("docs", {"op": "eq", "field": "tenant", "value": "acme"})  # -> count
```

## Bulk loading (binary wire)

Three methods cover large loads. They ship vectors over the binary bulk wire
(`Content-Type: application/octet-stream`, raw `f32` instead of JSON text) and
automatically split the load across requests to stay under the server's
per-request caps (256 MiB, 262,144 points) — a million vectors in one call is
fine:

```python
# Initial load of an empty collection: cheap parallel staging, then one
# multi-core index build. Metadata rides along, so filtered workloads get
# the fast path too (measured ~6x faster to searchable than inline indexing).
c.bulk_stage("docs", ids, vectors, metadatas=metas)   # metas: one dict/None per id
c.bulk_build("docs", workers=0)                       # 0 = all cores; blocks

# Writes into a collection that is already built: each point indexed inline.
c.batch_upsert("docs", ids, vectors, metadatas=metas, upsert=True)  # -> count
```

Prefer `bulk_stage` + `bulk_build` for initial loads; `batch_upsert` is for
incremental writes, or for points that need content, sparse vectors, TTLs, or
CAS — none of which the staging wire carries.

## Reading & listing

```python
points = c.get_batch("docs", [1, 2, 3], with_vector=True, with_payload=True)
# -> [Point(id, vector, content, metadata)], content lifted out of $content

page = c.scroll("docs", filter=None, limit=100)      # ScrollPage: list-like
while page.next_cursor:
    page = c.scroll("docs", limit=100, cursor=page.next_cursor)
```

## Search

```python
c.search("docs", query_vec, k=10, filter=None)
# -> [SearchResult(id, distance, score)]

c.search_docs("docs", query_vec, k=10)
# -> [Document(id, distance, content, score, metadata)]

c.search_groups("docs", query_vec, k=5, group_by="doc_id", group_size=2)
# -> [Group(key, hits)]

c.hybrid_search("docs", dense=query_vec, k=10,
                sparse={"indices": [3, 17], "values": [0.4, 0.9]},
                method="rrf", alpha=0.0)      # method: "rrf" | "weighted"

c.search_text("docs", "how do i rotate api keys", k=10, global_idf=False)
# BM25 — requires full_text=True at creation

c.hybrid_text("docs", vector=query_vec, text="rotate api keys", k=10,
              method="rrf")                   # "rrf" | "weighted" | "dbsf"
```

Filters are plain dicts in the server's filter JSON
([operators](../vector/filtering.md#operators)):

```python
f = {"op": "and", "and": [
    {"op": "eq",  "field": "tenant", "value": "acme"},
    {"op": "gte", "field": "year",   "value": 2020},
]}
c.search("docs", query_vec, k=10, filter=f)
```

The `rostam.filters` module builds the same trees with less ceremony:

```python
from rostam import filters as f

c.search("docs", query_vec, k=10,
         filter=f.and_(f.eq("tenant", "acme"), f.gte("year", 2020)))
```

## Text-first helpers

The core client is deliberately vector-only. `rostam.TextStore` wraps a client
plus an embedder for a text-in, text-out surface — `FunctionEmbedder` adapts
any callable (e.g. a sentence-transformers model's `.encode`), and
`OpenAIEmbedder` calls any OpenAI-compatible `/embeddings` endpoint over the
standard library:

```python
from rostam import RostamClient, TextStore, OpenAIEmbedder

store = TextStore(RostamClient("http://localhost:8080"), "docs", OpenAIEmbedder())
store.create_collection()          # dim inferred from the embedder
store.add(["first chunk", "second chunk"], metadatas=[{"doc_id": 1}, {"doc_id": 1}])
hits = store.search("a question", k=4)
```

## Multi-vector (late interaction)

```python
c.mv_create_collection("docs-colbert", dim=128)
c.mv_add("docs-colbert", doc_id=1, tokens=token_vectors, metadata={"src": "faq"})
c.mv_search("docs-colbert", query_tokens, k=10)   # -> [MultiResult(id, score, metadata)]
c.mv_delete("docs-colbert", 1)
```

## Native TCP client (Rostam)

`Rostam` (alias `RostamKV`) is a dependency-free, stdlib-only client for the
binary TCP protocol — the same protocol the [Go smart client](go-client.md)
speaks. It connects to the server's `-tcp` port (**`7000`** in the examples
throughout these docs), not the HTTP port:

```python
from rostam import Rostam

r = Rostam("127.0.0.1", 7000, auth_token=None, timeout=30.0, pool_maxsize=8)
r.put("user:42", b'{"coins":100}')       # KV ops are TCP-only — see kv/overview.md
r.vector.create_collection("posts", dim=768, metric="cosine")
```

Key-value operations (`get`/`put`/`delete`/`incr`/`expire`/`ping`) are methods
on `r` directly — see [Key-value store](../kv/overview.md). Vector operations
live under `r.vector`, sharing the same connection pool and auth token.

### Vector operations

`r.vector` covers the full vector surface over TCP, including the batch/scroll/
RAG-shaped search/hybrid/recommend methods added alongside the Go typed
[`Collection` client](go-client.md#typed-collection-client):

```python
r.vector.create_collection("posts", dim=768, metric="cosine", full_text=True)

r.vector.upsert("posts", 1, vec, content="rotating api keys safely",
                metadata={"tenant": "acme"})
r.vector.insert("posts", 2, vec)                    # create-only: errors if id exists
r.vector.upsert_batch("posts", [
    {"id": 3, "vector": v3, "content": "third post"},
    {"id": 4, "vector": v4, "content": "fourth post"},
])                                                    # N sequential upserts, not pipelined

point = r.vector.get("posts", 1, with_vector=True, with_payload=True)
# -> {vector, metadata, ttl_ms, sparse, content} or None if absent

rows = r.vector.get_batch("posts", [1, 2, 3], with_vector=True, with_payload=True)
# -> [{id, found, vector, metadata, ttl_ms, sparse, version, content}, ...]

docs, next_cursor = r.vector.scroll("posts", limit=100)
while next_cursor:
    docs, next_cursor = r.vector.scroll("posts", limit=100, cursor=next_cursor)

r.vector.delete("posts", 2)                          # -> bool (existed)
r.vector.exists("posts", 1)                           # -> bool
```

Search-family methods return `SearchResults` (or `GroupResults` for
`search_groups`) — a plain `list` subclass that also carries `.degraded` and
`.missing`, reporting whether the read was partial (e.g. during a cluster
outage). Iteration, indexing, `len()`, and equality against a bare list all
work as normal; `.degraded`/`.missing` just ride along, mirroring the Go typed
client's `SearchResponse{Results, Degraded, Missing}`:

```python
hits = r.vector.search("posts", query_vec, k=10, filter=None)
hits.degraded, hits.missing         # False, [] on a healthy single-node server

docs = r.vector.search_docs("posts", query_vec, k=10, filter=None)
# -> SearchResults of {id, distance, score, content, metadata}

groups = r.vector.search_groups("posts", query_vec, k=5, group_by="doc_id",
                                group_size=2, fetch_k=0, filter=None)
# -> GroupResults of {key, hits}

hits = r.vector.hybrid_search("posts", dense=query_vec, k=10,
                              sparse={"indices": [3, 17], "values": [0.4, 0.9]},
                              method="rrf", alpha=0.0)

hits = r.vector.hybrid_text("posts", dense=query_vec, text="rotate api keys",
                            k=10, method="rrf")
# requires the collection to have been created with full_text=...

hits = r.vector.recommend("posts", positive=[1, 2], negative=[9], k=10,
                          strategy="average_vector")   # or "best_score"
```

`r.vector.query(...)` has the same signature as `recommend` — it exists only
so callers reaching for the "unified Query API" name find the one shape this
client speaks. This client is **recommend-shaped only**: it does not build the
general fusion/rerank/prefetch-tree `QuerySpec` the Go SDK's `Query` supports.

### Worked example: recommend the next post (native TCP)

```python
from rostam import Rostam

r = Rostam("127.0.0.1", 7000)
r.vector.create_collection("posts", dim=768, metric="cosine", full_text=True)

for id, text in [(1, "rotating api keys safely"),
                  (2, "kubernetes autoscaling basics"),
                  (3, "zero-downtime key rotation at scale")]:
    r.vector.upsert("posts", id, embed(text), content=text,
                    metadata={"tenant": "acme"})

# Fuse a dense query with BM25 full-text.
hits = r.vector.hybrid_text("posts", dense=embed("how do i rotate api keys"),
                            text="how do i rotate api keys", k=5, method="rrf")

# Or: the reader just finished post 1 — recommend what's similar.
next_up = r.vector.recommend("posts", positive=[1], k=5)
```

## Framework adapters

Installing an extra enables the corresponding integration package for
LangChain (`langchain-core ≥ 0.2`), LlamaIndex (`llama-index-core ≥ 0.10`), or
Haystack (`haystack-ai ≥ 2.0`) — use Rostam as the vector store behind your
existing pipeline.

## Return types

| Class | Fields |
|---|---|
| `SearchResult` | `id, distance, score` |
| `Document` | `id, distance, content, score, metadata` |
| `Point` | `id, vector, content, metadata` |
| `Group` | `key, hits` |
| `MultiResult` | `id, score, metadata` |
| `ScrollPage` | list-like of `Document` + `next_cursor` |
