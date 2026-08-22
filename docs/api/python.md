# Python client

A dependency-free (stdlib-only) client for Rostam, with optional LangChain,
LlamaIndex, and Haystack adapters. Requires Python ≥ 3.9.

```sh
pip install rostam-client

# with an adapter
pip install "rostam-client[langchain]"   # or [llamaindex], [haystack]
```

One class, `Rostam`, speaks both of the server's non-gRPC transports — REST
over `http.client` and the native binary protocol over `socket` — and picks
between them from the `target` string you pass in:

```python
from rostam import Rostam

c = Rostam("http://localhost:8080", api_key=None, timeout=30.0)   # REST, port 8080
r = Rostam("tcp://localhost:7000")                                # native TCP, port 7000
r = Rostam("localhost:7000")                                      # bare host:port also -> TCP
```

`http://`/`https://` selects REST; `tcp://host:port` or a bare `host:port`
(no scheme) selects the native binary protocol. A bare host with no port is a
hard error — the two transports' default ports differ, so guessing would be
unsafe. `api_key`/`auth_token` (either name works; `auth_token` wins if both
are given) is a bearer token on HTTP and rides the TCP protocol-v2 frame on
every request.

All errors raise `RostamError(message, status)`; the HTTP backend carries the
response status, `TransportError` (a `RostamError` subclass, see
[Transport-specific methods](#transport-specific-methods)) carries none.
Metadata is plain Python dicts — the client converts to/from the server's
tagged value encoding automatically.

## One client, flat vector API, either transport

The vector methods are identical in name and signature on both transports —
`create_collection, drop_collection, upsert, insert, upsert_batch, delete,
get, get_batch, scroll, search, search_docs, search_groups, hybrid_search,
hybrid_text, recommend, exists` — so code written against one target works
unchanged against the other, and switching from a loopback TCP dev server to
an HTTP-fronted cluster is a one-line change:

```python
from rostam import Rostam, filters as f

c = Rostam("http://localhost:8080", api_key="optional-bearer-token")

c.create_collection("docs", dim=384, metric="cosine")   # metric: cosine|l2|dot

# Upsert points. Metadata is plain Python — the client encodes it to Rostam's
# tagged wire form for you (and decodes it back on the way out).
c.upsert("docs", 1, embedding, content="the chunk text", metadata={"doc_id": 7, "lang": "en"})

# k-NN with content + metadata, filtered.
hits = c.search_docs("docs", embedding, k=5, filter=f.eq("doc_id", 7))
for d in hits:
    print(d.id, d.distance, d.content, d.metadata)

# Group-by-document: the top-k distinct documents, best chunk(s) each.
for g in c.search_groups("docs", embedding, k=5, group_by="doc_id", group_size=2):
    print(g.key, [h.content for h in g.hits])

# Compound filters.
hits = c.search_docs("docs", embedding, k=5,
                     filter=f.and_(f.gte("price", 10.0), f.eq("in_stock", True)))

c.delete("docs", 1)
c.insert("docs", 2, embedding)                # create-only: duplicate id raises
c.drop_collection("docs")
```

`insert` has no `content` parameter (it's create-only, and the wire op it
sends carries no content field) — use `upsert` when a point needs stored text.

## Collection handle

When most of your calls target one collection, `r.collection(name)` returns a
handle that binds the name so it stops being the first argument every time
(mirrors the Go client's `client.Collection`). Construction does no I/O, and
each method forwards to the identically-named flat method — so transport rules
still apply (`query`, `delete_by_filter` raise `TransportError` on TCP just as
`r.query` / `r.delete_by_filter` do).

```python
docs = c.collection("docs")   # a handle; does no I/O
docs.create(dim=384, metric="cosine")
docs.upsert(1, embedding, content="the chunk text")
hits = docs.search_docs(embedding, k=5, filter=f.eq("doc_id", 7))
docs.delete(1)
docs.drop()          # -> c.drop_collection("docs")
```

Handle methods: `create`, `drop`, `upsert`, `insert`, `upsert_batch`, `delete`,
`delete_by_filter`, `get`, `get_batch`, `scroll`, `exists`, `search`,
`search_docs`, `search_groups`, `hybrid_search`, `hybrid_text`, `recommend`,
`query`.

## Connections

The client keeps a small pool of connections and reuses them, so a sequence
of calls does not pay a handshake each time. It is safe to share one client
across threads.

```python
with Rostam("http://localhost:8080") as c:   # or call c.close()
    c.search("docs", q, k=10)

Rostam(url, pool_maxsize=8)    # idle connections kept per client
```

`close()` releases pooled connections; the client stays usable and reconnects
on the next call.

On HTTP, connection reuse is built on `http.client`, which — unlike the
`urllib` call it replaced — does not consult `HTTP_PROXY`/`HTTPS_PROXY`. A
client behind an egress proxy should point `target` at the proxy or at a
local forwarder. Searches are sent in Rostam's binary query framing, which
skips encoding the query vector as text; at dim=768 that was 31% of the
request. Against a server too old to understand it, the client detects the
refusal and falls back to JSON for the rest of its life.

## Search

```python
c.search("docs", query_vec, k=10, filter=None)
# -> SearchResults (SearchResult(id, distance, score))

c.search_docs("docs", query_vec, k=10)
# -> SearchResults (Document(id, distance, content, score, metadata))

c.search_groups("docs", query_vec, k=5, group_by="doc_id", group_size=2)
# -> GroupResults (Group(key, hits))

c.hybrid_search("docs", dense=query_vec, k=10,
                sparse={"indices": [3, 17], "values": [0.4, 0.9]},
                method="rrf", alpha=0.0)      # method: "rrf" | "weighted" | "dbsf"

c.hybrid_text("docs", dense=query_vec, text="rotate api keys", k=10,
              method="rrf")                   # dense + BM25 fusion; needs full_text=True at creation

c.recommend("docs", positive=[1, 2], k=5)     # score toward example ids (± negative)
```

`SearchResults`/`GroupResults` are list-like and carry `.degraded`/`.missing`,
reporting whether a partition was unreachable and which one — see
[Cross-shard reads](../vector/search.md#cross-shard-reads).

Filters are plain dicts in the server's filter JSON
([operators](../vector/filtering.md#operators)), or build them with less
ceremony via `rostam.filters`:

```python
from rostam import filters as f

c.search("docs", query_vec, k=10,
         filter=f.and_(f.eq("tenant", "acme"), f.gte("year", 2020)))
```

## Reading & listing

```python
pt = c.get("docs", 1)                                # -> Point | None
points = c.get_batch("docs", [1, 2, 3], with_vector=True, with_payload=True)
# -> [Point(id, vector, content, metadata)], one per PRESENT id (absent ids omitted)

page = c.scroll("docs", filter=None, limit=100)      # ScrollPage: list-like
while page.next_cursor:
    page = c.scroll("docs", limit=100, cursor=page.next_cursor)

c.exists("docs", 1)                                   # -> bool
```

`Point.id`/`.vector`/`.content`/`.metadata` are cross-transport; the wire also
carries `ttl_ms`/`sparse`/`version`, which this unified `Point` shape does not
expose on either backend.

## Key-value (`r.kv`, TCP only)

The KV store has no REST surface — it lives only on the native binary
protocol, because it is built for sub-microsecond operations an HTTP round
trip would defeat. Reach it as `r.kv.*` on a `tcp://` client:

```python
r = Rostam("tcp://localhost:7000")

r.kv.put("user:42", b'{"coins":100}', ttl_ms=300_000)   # ttl_ms=0: no expiry
r.kv.get("user:42")                                      # -> bytes | None
r.kv.delete("user:42")                                    # -> bool (existed)
r.kv.incr("views:42", 1)                                  # atomic; missing key counts as 0
r.kv.expire("user:42", 3_600_000)                          # refresh TTL, ms
r.kv.ping()                                                # -> bool
```

Keys and values may be `str` (encoded UTF-8) or `bytes`; reads always return
`bytes` (or `None`). On an HTTP-connected client, `c.kv` is a sentinel object
— any attribute access raises `TransportError`, not `AttributeError`, so the
failure names the fix (connect with `tcp://host:7000`) instead of looking
like a typo. See [KV overview](../kv/overview.md) for the full op set and
`Store` backend comparison.

## Transport-specific methods

Some methods only make sense on one transport. Calling them on the other
raises `rostam.TransportError` (a `RostamError` subclass) naming the fix,
rather than failing silently or with `AttributeError`:

- **HTTP-only**: `health`, `delete_by_filter`, `bulk_stage`, `bulk_build`,
  `batch_upsert` (not to be confused with the shared `upsert_batch` above —
  see [Bulk loading](#bulk-loading-http-binary-wire) for how they differ),
  `search_text`, `discover`, `mv_create_collection`, `mv_drop_collection`,
  `mv_add`, `mv_search`, `mv_delete`, and the general composable `query`.
- **TCP-only**: `r.kv.*` (see above).

```python
c = Rostam("http://localhost:8080")
c.delete_by_filter("docs", f.eq("doc_id", 7))   # purge a whole document

r = Rostam("tcp://localhost:7000")
r.delete_by_filter("docs", f.eq("doc_id", 7))   # raises TransportError: HTTP-only
```

### Bulk loading (HTTP, binary wire)

Three methods cover large loads. They ship vectors over the binary bulk wire
(`Content-Type: application/octet-stream`, raw `f32` instead of JSON text)
and automatically split the load across requests to stay under the server's
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

`batch_upsert` (this section, HTTP-only, parallel `ids`/`vectors`/`metadatas`
arrays over the binary bulk wire) is a different method from the shared
`upsert_batch` (works on both transports, one dict per point — see
[One client, flat vector API, either transport](#one-client-flat-vector-api-either-transport)).
The names are easy to conflate; pick the one whose call shape and transport
match what you need.

### Full-text and multi-vector (HTTP-only)

```python
c.search_text("docs", "how do i rotate api keys", k=10, global_idf=False)
# BM25 — requires full_text=True at creation

c.mv_create_collection("docs-colbert", dim=128)
c.mv_add("docs-colbert", doc_id=1, tokens=token_vectors, metadata={"src": "faq"})
c.mv_search("docs-colbert", query_tokens, k=10)   # -> [MultiResult(id, score, metadata)]
c.mv_delete("docs-colbert", 1)
```

## Text-first helpers

The core client is deliberately vector-only. `rostam.TextStore` wraps a
`Rostam` client plus an embedder for a text-in, text-out surface —
`FunctionEmbedder` adapts any callable (e.g. a sentence-transformers model's
`.encode`), and `OpenAIEmbedder` calls any OpenAI-compatible `/embeddings`
endpoint over the standard library:

```python
from rostam import Rostam, TextStore, OpenAIEmbedder

store = TextStore(Rostam("http://localhost:8080"), "docs", OpenAIEmbedder())
store.create_collection()          # dim inferred from the embedder
store.add(["first chunk", "second chunk"], metadatas=[{"doc_id": 1}, {"doc_id": 1}])
hits = store.search("a question", k=4)
```

`TextStore` works over either transport — it just calls the flat vector API
underneath, so a `tcp://`-backed `Rostam` client works here too.

## Framework adapters

Installing an extra enables the corresponding integration package for
LangChain (`langchain-core ≥ 0.2`), LlamaIndex (`llama-index-core ≥ 0.10`),
or Haystack (`haystack-ai ≥ 2.0`) — use Rostam as the vector store behind
your existing pipeline. Every adapter wraps a `Rostam` instance you construct
and hand in, so it works against either transport the same way the core
client does:

```python
from langchain_openai import OpenAIEmbeddings
from rostam import Rostam
from rostam.langchain import RostamVectorStore

client = Rostam("http://localhost:8080")
client.create_collection("docs", dim=1536, metric="cosine")

store = RostamVectorStore.from_texts(
    texts=["first chunk", "second chunk"],
    embedding=OpenAIEmbeddings(),
    metadatas=[{"doc_id": 1}, {"doc_id": 1}],
    client=client,
    collection="docs",
)

docs = store.similarity_search("a question", k=4, filter={"doc_id": 1})
```

See the [client README](https://github.com/rostamlabs/rostam/tree/main/clients/python)
for the full adapter walkthroughs — LangChain hybrid search and MMR,
LlamaIndex hybrid mode, and Haystack's `RostamDocumentStore`.

## Return types

| Class | Fields |
|---|---|
| `SearchResult` | `id, distance, score` |
| `Document` | `id, distance, content, score, metadata` |
| `Point` | `id, vector, content, metadata` |
| `Group` | `key, hits` |
| `MultiResult` | `id, score, metadata` (HTTP-only) |
| `ScrollPage` | list-like of `Document` + `next_cursor` |
| `SearchResults` / `GroupResults` | list-like, plus `.degraded` / `.missing` |

## Notes

- **IDs.** Rostam point ids are `uint64`. The client takes integers directly.
  The LangChain adapter accepts string ids: a purely-numeric string is used
  verbatim, anything else is hashed (BLAKE2b) to a stable 64-bit id, so
  repeated upserts/deletes of the same external id address the same point.
- **Metadata kinds.** Supported value types: `int`, `float`, `str`, `bool`,
  and lists of `int`/`float`/`str`.
- **Ports.** The server's default REST port is `8080`; the native TCP port is
  `7000` — pick one target string, not both, per `Rostam(...)` instance.
