# Migrating from Pinecone to Rostam

This guide maps Pinecone concepts and API calls to Rostam so you can move an
existing workload over. Rostam is open source (Apache-2.0) and **self-hosted** —
one Go binary you run in your own environment — so the main reasons teams make
this move are cost control and keeping vector data inside a boundary you own
(data residency, on-prem, air-gapped). There is no per-project minimum, and the
database itself makes no outbound calls: nothing Rostam stores leaves your
network.

!!! note "One caveat about egress"
    Rostam has no egress, but *embedding* is a separate step. If you generate
    embeddings with a hosted API (OpenAI, Cohere, …), your text leaves your
    network at that step regardless of where the vectors are stored. For a fully
    in-boundary pipeline, embed with a local/self-hosted model.

If you just want the API side by side, jump to [Code, side by side](#code-side-by-side).

## Concept mapping

| Pinecone | Rostam | Notes |
|---|---|---|
| Account / project | Your Rostam server (or cluster) | You run it; no hosted control plane. |
| Index | **Collection** | `create_collection(name, dim, metric)`. |
| Metric `cosine` / `euclidean` / `dotproduct` | `metric="cosine"` / `"l2"` / `"dot"` | Set per collection at creation. |
| Vector (`id`, `values`, `metadata`) | Point (`id`, embedding, `metadata`, optional `content`) | Rostam ids are integers; see [IDs](#ids-string-vs-integer). |
| Namespace | **Tenant** (`<tenant>/<collection>`) | A real primitive — and an optional security boundary. See [Namespaces](#namespaces-tenants). |
| Metadata filter (`$eq`, `$in`, …) | `filters` helpers (`f.eq`, `f.in_`, …) | Full translation table [below](#translating-metadata-filters). |
| Serverless / pods | The single binary, or a Raft cluster | Scale by running more shards/nodes; see [Clustering](../server/clustering.md). |
| Managed, multi-tenant SaaS | You operate it | See [Security](../server/security.md) and [Backups](../server/backups.md). |

## Run Rostam

```bash
# one static binary; data stays on disk you control.
# pass the token via the environment, NOT -api-key (a flag secret leaks via /proc and shell history):
export ROSTAM_API_KEY="a-strong-token"
rostam-server -http :8080 -data ./data
```

The server refuses to bind a reachable address without auth. See
[Running the server](../server/running.md) for TLS, clustering, and Docker.

Install the Python client:

```bash
pip install rostam-client
```

## Code, side by side

### Connect

```python
# Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("docs")

# Rostam
from rostam import Rostam, filters as f
c = Rostam("http://localhost:8080", api_key="a-strong-token")
```

### Create the index / collection

```python
# Pinecone
pc.create_index(name="docs", dimension=384, metric="cosine", spec=...)

# Rostam
c.create_collection("docs", dim=384, metric="cosine")   # metric: cosine | l2 | dot
```

### Upsert

```python
# Pinecone
index.upsert(vectors=[
    {"id": "1", "values": embedding, "metadata": {"doc_id": 7, "lang": "en"}},
])

# Rostam  (content is an optional stored text payload returned with hits)
c.upsert("docs", 1, embedding, content="the chunk text",
         metadata={"doc_id": 7, "lang": "en"})
```

### Query

```python
# Pinecone
res = index.query(vector=embedding, top_k=5,
                  filter={"doc_id": {"$eq": 7}}, include_metadata=True)
for m in res["matches"]:
    print(m["id"], m["score"], m["metadata"])

# Rostam  (hits carry .id / .content / .metadata / .distance — smaller distance = closer)
hits = c.search_docs("docs", embedding, k=5, filter=f.eq("doc_id", 7))
for h in hits:
    print(h.id, h.distance, h.metadata)
```

### Fetch and delete

```python
# Pinecone
index.fetch(ids=["1"])
index.delete(ids=["1"])

# Rostam
c.get("docs", 1)      # -> Point | None
c.delete("docs", 1)
```

## Translating metadata filters

Pinecone filters are dicts of operators; Rostam uses the `filters` helpers
(`from rostam import filters as f`). They compose the same way.

| Pinecone | Rostam |
|---|---|
| `{"g": {"$eq": "x"}}` | `f.eq("g", "x")` |
| `{"g": {"$ne": "x"}}` | `f.ne("g", "x")` |
| `{"n": {"$gt": 3}}` | `f.gt("n", 3)` |
| `{"n": {"$gte": 3}}` | `f.gte("n", 3)` |
| `{"n": {"$lt": 3}}` | `f.lt("n", 3)` |
| `{"n": {"$lte": 3}}` | `f.lte("n", 3)` |
| `{"g": {"$in": ["a","b"]}}` | `f.in_("g", ["a", "b"])` |
| `{"g": {"$nin": ["a","b"]}}` | `f.not_(f.in_("g", ["a", "b"]))` |
| `{"$and": [A, B]}` | `f.and_(A, B)` |
| `{"$or": [A, B]}` | `f.or_(A, B)` |
| implicit `{"a": 1, "b": 2}` (AND) | `f.and_(f.eq("a", 1), f.eq("b", 2))` |

Pinecone's **`$exists`** has no direct equivalent — Rostam filters match on values,
not key presence. Emulate it by storing an explicit sentinel (e.g. a boolean
`has_x` field) at write time and filtering on that.

Rostam runs filters through an **exact, filter-first path** — a selective filter
does not degrade recall. See [Filtering](../vector/filtering.md).

## Namespaces → tenants

Pinecone namespaces partition one index. Rostam's equivalent is a **tenant**:
collection names can be written `<tenant>/<collection>` (a bare name lands in the
default tenant). Unlike a Pinecone namespace — which is only a partition — a Rostam
tenant can also be an **authoritative security boundary**: bind an API key to a
tenant and run the server with `-tenant-isolation`, and that key can only see its
own tenant's collections.

```python
# Pinecone: index.query(vector=v, top_k=5, namespace="user-42")
# Rostam:   the namespace becomes the tenant prefix
c.search_docs("user-42/docs", v, k=5)
```

Alternatives when you do **not** need isolation: store the namespace as a metadata
field and add `f.eq("ns", "user-42")` to every query, or use a separate collection
per namespace. See [Collections, tenants & aliases](../concepts/collections.md).

## IDs: string vs. integer

Pinecone ids are strings; Rostam point ids are integers (`uint64`). Map strings
deterministically with the client's helper:

```python
from rostam._ids import to_uint64   # stable str -> uint64; used by all the framework adapters
c.upsert("docs", to_uint64("abc-123"), embedding, metadata={"pinecone_id": "abc-123"})
```

`to_uint64` is **one-way**. Neither the raw client nor `TextStore` preserves the
original string for you — if you need it back on read, store it in metadata
yourself (as above) and read it from `hit.metadata["pinecone_id"]`. Pick a key
your own metadata doesn't already use.

## Migrating your data

There is no import tool — you re-upsert. If you still have the source embeddings,
upsert them straight into Rostam. If not, read them out of Pinecone in pages and
write them across, keeping the original id in metadata:

```python
from rostam._ids import to_uint64

for ids in paginate_pinecone_ids(index):          # your paging over index.list()
    fetched = index.fetch(ids=ids)["vectors"]
    for pid, v in fetched.items():
        meta = dict(v.get("metadata") or {})
        meta["pinecone_id"] = pid                  # preserve the original id for lookup
        c.upsert("docs", to_uint64(pid), v["values"], metadata=meta)
```

Re-embedding from source documents is often cleaner than exporting vectors, and it
lets you switch embedding models at the same time. Rostam can also embed for you —
see `TextStore` and the built-in embedders in the [Python client](../api/python.md)
(note that a hosted embedder sends text out at the embedding step; see the egress
caveat above).

## What is different (read before you commit)

- **You operate it.** No managed control plane — you run the server, back it up
  ([Backups](../server/backups.md)), and secure it ([Security](../server/security.md)).
  That is the point (your data, your boundary), but it is real work.
- **Integer ids** — map strings with `to_uint64`, keep the original in metadata (see above).
- **Distance, not score** — hits carry `distance` (smaller = closer), not a Pinecone-style
  similarity `score`. Convert if your app expects scores.
- **No `$exists` filter** and no server-side embedding — see the sections above.

## Next steps

- [Quickstart](../quickstart.md)
- [Search](../vector/search.md) · [Filtering](../vector/filtering.md) · [Hybrid & full-text](../vector/hybrid-search.md)
- [Clustering](../server/clustering.md) for HA, [Security](../server/security.md) for auth/TLS/RBAC
- Benchmarks vs. Pinecone and others: <https://rostamlabs.com/compare>
