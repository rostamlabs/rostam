# rostam-client

A dependency-free Python client and LangChain adapter for the
[Rostam](https://github.com/rostamlabs/rostam) vector store.

The core client uses **only the Python standard library** — no `requests`, no
gRPC, nothing to pull in. It speaks both of the server's non-gRPC transports:
REST over `http.client` and the native binary protocol over `socket`. The
optional LangChain adapter requires `langchain-core`.

## Install

```bash
pip install rostam-client            # core client only (zero dependencies)
pip install rostam-client[langchain] # + the LangChain VectorStore adapter
```

## Run a server

Point the client at either the server's `-http` listener (REST, default port
`8080`) or its `-tcp` listener (the native binary protocol, default port
`7000`) — pick one when you start the server, or bind both. (The server also
offers gRPC; this client does not speak it.)

Neither of these needs a Go toolchain or a checkout:

```bash
# Container. Auth is required because it binds 0.0.0.0 inside the container.
docker run --rm -p 127.0.0.1:8080:8080 -p 127.0.0.1:7000:7000 \
  -e ROSTAM_API_KEY=dev-token -v rostam-data:/data \
  ghcr.io/rostamlabs/rostam:latest -http 0.0.0.0:8080 -tcp 0.0.0.0:7000 -data /data

# Prebuilt binary. Verifies the release checksum before installing.
curl -fsSL https://raw.githubusercontent.com/rostamlabs/rostam/main/install.sh | sh
export PATH="$PATH:$HOME/.local/bin"    # where the installer puts it
rostam-server -http 127.0.0.1:8080 -tcp 127.0.0.1:7000 -data ./data
```

Then point the client at it. `Rostam(target)` picks the transport from the
target string — `http(s)://` speaks REST, `tcp://host:port` (or a bare
`host:port`) speaks the native binary protocol:

```python
from rostam import Rostam

c = Rostam("http://localhost:8080", api_key="dev-token")  # container, REST
c = Rostam("http://localhost:8080")                       # loopback binary, REST
```

Without `-data` the store is memory-only and everything is lost when the process
exits — that is the container's default, which is why the command above mounts a
volume.

With no authenticator configured the server **refuses to bind a reachable
address** rather than serve an open datastore to the network. That is why the
container needs `ROSTAM_API_KEY` while a loopback binary does not, and why a
bare `-http :8080` will not start. To listen beyond loopback, give it auth
(`-api-key`/`ROSTAM_API_KEY` or `-keys-file`), or pass `-insecure` to run open
deliberately.

Building from source stays available for contributors:
`go build -o rostam-server ./cmd/rostam-server`.

## Quickstart

### HTTP (`http://host:8080`)

```python
from rostam import Rostam
from rostam import filters as f

c = Rostam("http://localhost:8080", api_key="optional-bearer-token")

c.create_collection("docs", dim=384, metric="cosine")   # metric: cosine|l2|dot

# Upsert points. Metadata is plain Python — the client encodes it to Rostam's
# tagged wire form for you (and decodes it back on the way out).
c.upsert("docs", 1, embedding, content="the chunk text", metadata={"doc_id": 7, "lang": "en"})

# k-NN with content + metadata, filtered.
hits = c.search_docs("docs", query_embedding, k=5, filter=f.eq("doc_id", 7))
for d in hits:
    print(d.id, d.distance, d.content, d.metadata)

# Group-by-document: the top-k distinct documents, best chunk(s) each.
for g in c.search_groups("docs", query_embedding, k=5, group_by="doc_id", group_size=2):
    print(g.key, [h.content for h in g.hits])

# Compound filters.
hits = c.search_docs("docs", query_embedding, k=5,
                     filter=f.and_(f.gte("price", 10.0), f.eq("in_stock", True)))

c.delete("docs", 1)
c.delete_by_filter("docs", f.eq("doc_id", 7))   # purge a whole document; HTTP-only
```

The client also exposes `insert` (rejects duplicate ids), `hybrid_search`
(dense + sparse fusion), `drop_collection`, and `health`.

### Native TCP (`tcp://host:7000`)

Same vector API, flat, over the binary protocol — plus `r.kv.*`, which has no
HTTP equivalent:

```python
from rostam import Rostam
from rostam import filters as f

r = Rostam("tcp://localhost:7000")   # or Rostam("localhost:7000") — bare host:port defaults to TCP

r.create_collection("docs", dim=384, metric="cosine")
r.upsert("docs", 1, embedding, content="the chunk text", metadata={"doc_id": 7, "lang": "en"})

# Dense + BM25 fusion in one call (collection needs full-text indexing enabled).
hits = r.hybrid_text("docs", embedding, "apple pie", k=5, filter=f.eq("doc_id", 7))

# Recommend: score toward example ids (and away from `negative` ones).
hits = r.recommend("docs", positive=[1, 2], k=5)

# Key-value, same connection. TCP-only: r.kv raises TransportError on an
# HTTP-connected client.
r.kv.put("user:42", b'{"coins":100}', ttl_ms=300_000)
r.kv.get("user:42")
r.kv.incr("views:42", 1)   # atomic; missing key counts as 0
```

`r.query(...)` (the general composable Query API) and a few HTTP-only extras
(`health`, `delete_by_filter`, `bulk_build`, `mv_*`, `search_text`,
`discover`) raise `TransportError` on a TCP-connected client; TCP callers use
`recommend()`/`hybrid_text()` in their place.

## Embeddings (work in text, not vectors)

The core client takes vectors. `TextStore` adds the text-first ergonomics —
embedding happens client-side, so no model dependency touches Rostam's engine.

```python
from rostam import Rostam, TextStore, OpenAIEmbedder

store = TextStore(Rostam("http://localhost:8080"), "docs", OpenAIEmbedder())
store.create_collection()                       # dim inferred from the embedder
store.add(["first chunk", "second chunk"], metadatas=[{"doc_id": 1}, {"doc_id": 1}])

docs = store.search("a question", k=4)                       # embeds the query for you
groups = store.search_groups("a question", k=4, group_by="doc_id")
```

Embedder options:

- **`OpenAIEmbedder`** — calls any OpenAI-compatible `/embeddings` endpoint using
  only the standard library (no `openai` package). Works with OpenAI, Azure
  OpenAI, and local servers (Ollama, LM Studio, text-embeddings-inference) via
  `base_url`. Reads `OPENAI_API_KEY` by default.
- **`FunctionEmbedder`** — wraps any callable, e.g. a local model:
  ```python
  from sentence_transformers import SentenceTransformer
  m = SentenceTransformer("all-MiniLM-L6-v2")
  embedder = FunctionEmbedder(lambda ts: m.encode(ts).tolist())
  ```

Embedders implement the same interface as LangChain's `Embeddings`, so the same
object feeds both `TextStore` and `RostamVectorStore`.

## Multi-vector / late interaction (ColBERT MaxSim)

For late-interaction retrieval, a document is represented by *many* token vectors
and scored by MaxSim (`Σ_q max_d cos(q,d)`) rather than a single pooled vector.
Multi-vector collections are in-memory.

```python
# quant ("sq8"/"bq1") quantizes the first-stage graph; persistent=True keeps the
# float32 token vectors off-heap in an mmap file and survives restart.
c.mv_create_collection("docs", dim=128, quant="sq8", persistent=True)
c.mv_add("docs", 1, doc_token_vectors, metadata={"doc_id": 1})   # token matrix
c.mv_add("docs", 2, other_token_vectors)

hits = c.mv_search("docs", query_token_vectors, k=5)             # MaxSim ranking
for h in hits:
    print(h.id, h.score, h.metadata)

c.mv_delete("docs", 1)
```

You supply token vectors yourself (e.g. from a ColBERT/late-interaction model);
the client handles the wire encoding and decodes results (including native
metadata). Persistent collections are flushed server-side (embedded
`CollectionStore.FlushMultiVector`).

## LangChain

`RostamVectorStore` implements the standard LangChain `VectorStore` interface, so
it drops into existing retrieval chains. You bring the embeddings; Rostam stores
the vector, the chunk text, and metadata.

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
docs_scored = store.similarity_search_with_score("a question", k=4)

# Rostam-specific extension: retrieve the top-k distinct documents.
groups = store.search_grouped("a question", k=4, group_by="doc_id", group_size=2)
```

`filter` accepts either a native Rostam filter (`rostam.filters`) or a simple
`{field: value}` map (translated to an AND of equalities). Relevance scores map
Rostam's distance to a `0..1` range.

### Hybrid retrieval

Fuse dense KNN with BM25 full-text search by enabling `full_text` on the store
(the collection must have the full-text index enabled):

```python
store = RostamVectorStore(
    client, "docs", embedding, full_text=True, auto_create=True
)
# Dense + server-side BM25 over the raw query string (default):
docs = store.hybrid_search("apple pie", k=4)

# Dense + SPLADE-style sparse (pass a callable that returns a sparse vector):
store = RostamVectorStore(
    client, "docs", embedding, full_text=True, sparse_embedding=my_splade_fn
)
docs = store.hybrid_search("apple pie", k=4)
```

`hybrid_search` signature: `hybrid_search(query, k=4, *, filter=None, method="rrf", alpha=0.0)`.
`method` and `alpha` are forwarded to Rostam's fusion endpoint unchanged.

### Maximal Marginal Relevance (MMR)

Retrieve diverse results by trading off relevance against redundancy:

```python
docs = store.max_marginal_relevance_search(
    "apple pie", k=4, fetch_k=20, lambda_mult=0.5
)
```

`fetch_k` candidates are fetched first; MMR re-ranks them to the final `k`.
`lambda_mult=1.0` is pure relevance; `0.0` is pure diversity.
The async variant is `await store.amax_marginal_relevance_search(...)`.

### Fetch by id

Retrieve documents by their original string ids (missing ids are silently omitted):

```python
docs = store.get_by_ids(["id-1", "id-2"])
# Async:
docs = await store.aget_by_ids(["id-1", "id-2"])
```

### Async methods

Every retrieval and write method has an `a`-prefixed async counterpart. They
offload to a thread pool over the synchronous client — no extra dependency is
required:

```python
docs = await store.asimilarity_search("a question", k=4)
docs_scored = await store.asimilarity_search_with_score("a question", k=4)
ids = await store.aadd_texts(["chunk one", "chunk two"])
ok = await store.adelete(["id-1"])
```

### Auto-create

By default (`auto_create=True`) the collection is created on the first write.
Dimensionality is inferred from the first batch of embeddings. If the store is
configured with `full_text=True` the collection is created with the full-text
index enabled (required for hybrid search):

```python
# Collection created automatically on first add_texts / from_texts call:
store = RostamVectorStore(client, "docs", embedding, auto_create=True, full_text=True)
store.add_texts(["chunk one"])   # collection created here

# Manage the collection yourself:
client.create_collection("docs", dim=1536, metric="cosine")
store = RostamVectorStore(client, "docs", embedding, auto_create=False)
```

`from_texts` forwards `auto_create`, `metric`, and `full_text` to the
constructor, so the class method works the same way.

## LlamaIndex

`rostam.llamaindex.RostamVectorStore` implements the LlamaIndex `VectorStore`
interface (`pip install rostam-client[llamaindex]`).

```python
from rostam import Rostam
from rostam.llamaindex import RostamVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

client = Rostam("http://localhost:8080")
client.create_collection("docs", dim=1536, metric="cosine")
store = RostamVectorStore(client=client, collection="docs")
index = VectorStoreIndex.from_documents(
    documents, storage_context=StorageContext.from_defaults(vector_store=store)
)
results = index.as_retriever().retrieve("a question")
```

Nodes are serialized with LlamaIndex's own metadata utils; `delete(ref_doc_id)`
purges every node of a document via a metadata filter; metadata filters on the
query translate to Rostam filters.

### Hybrid mode

Pass `mode=VectorStoreQueryMode.HYBRID` and set `query_str` to enable hybrid
retrieval. The collection must have the full-text index enabled (`full_text=True`).
Note that `query_embedding` is always required — the hybrid path uses the dense
vector unconditionally and only fuses it with BM25/sparse when `query_str` is also set:

```python
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

store = RostamVectorStore(client=client, collection="docs", full_text=True)
q = VectorStoreQuery(
    query_embedding=embedding,
    query_str="apple pie",          # required for hybrid
    mode=VectorStoreQueryMode.HYBRID,
    similarity_top_k=4,
)
result = store.query(q)
```

With a `sparse_embedding` callable, dense + sparse (SPLADE-style) fusion is used
instead of dense + BM25:

```python
store = RostamVectorStore(
    client=client, collection="docs", full_text=True, sparse_embedding=my_splade_fn
)
```

Hybrid results are ranked by the server's fusion score; similarities in the
returned `VectorStoreQueryResult` are rank-based (`1/(1+rank)`).

### Async methods

`async_add`, `aquery`, and `adelete` are available. They offload to a thread
pool over the sync client — no extra dependency is required:

```python
ids = await store.async_add(nodes)
result = await store.aquery(query)
await store.adelete(ref_doc_id)
```

### Auto-create

By default (`auto_create=True`) the collection is created on the first `add`
call. Dimensionality is inferred from the first node's embedding. If the store
is configured with `full_text=True` the collection is created with the full-text
index enabled (required for hybrid mode):

```python
# Collection created automatically on first add:
store = RostamVectorStore(client=client, collection="docs", full_text=True)
index = VectorStoreIndex.from_documents(
    documents, storage_context=StorageContext.from_defaults(vector_store=store)
)

# Manage the collection yourself:
client.create_collection("docs", dim=1536, metric="cosine")
store = RostamVectorStore(client=client, collection="docs", auto_create=False)
```

## Haystack

`rostam.haystack` provides a `RostamDocumentStore` (Haystack 2.x `DocumentStore`)
and a `RostamEmbeddingRetriever` component (`pip install rostam-client[haystack]`).

```python
from haystack import Document
from rostam import Rostam
from rostam.haystack import RostamDocumentStore, RostamEmbeddingRetriever

Rostam("http://localhost:8080").create_collection("docs", dim=384, metric="cosine")
store = RostamDocumentStore(url="http://localhost:8080", collection="docs")
store.write_documents([Document(content="hello", embedding=[...], meta={"src": "a"})])

retriever = RostamEmbeddingRetriever(document_store=store, top_k=5)
docs = retriever.run(query_embedding=[...])["documents"]
```

`count_documents` / `filter_documents` are served by Rostam's `scroll` listing.
Documents must carry embeddings; writes use overwrite semantics.

## Notes

- **Connections.** The client pools and reuses connections, so a sequence of
  calls does not pay a TCP handshake each time, and it is safe to share one
  client across threads. Use it as a context manager, or call `close()`, to
  release the pool; the client stays usable afterwards. Connection reuse is
  built on `http.client`, which does not consult `HTTP_PROXY`/`HTTPS_PROXY` —
  behind an egress proxy, point `base_url` at the proxy.
- **Search encoding.** Searches go out in Rostam's binary query framing rather
  than as JSON text, which at dim=768 was 31% of the request. Against a server
  too old to understand it the client notices and falls back to JSON for the
  rest of its life; this is automatic and not currently a constructor option
  on `Rostam(...)`.
- **IDs.** Rostam point ids are `uint64`. The client takes integers directly. The
  LangChain adapter accepts string ids: a purely-numeric string is used verbatim,
  anything else is hashed (BLAKE2b) to a stable 64-bit id, so repeated
  upserts/deletes of the same external id address the same point.
- **Metadata kinds.** Supported value types: `int`, `float`, `str`, `bool`, and
  lists of `int`/`float`/`str`.
