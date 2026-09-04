# Migrating from Pinecone to Rostam

This guide maps Pinecone concepts and API calls to Rostam so you can move an
existing workload over. Rostam is open source (Apache-2.0) and **self-hosted** —
one Go binary you run in your own environment — so the main reasons teams make
this move are cost control, and keeping vector data inside a boundary you own
(data residency, on-prem, air-gapped). Nothing leaves your network, and there is
no per-project minimum.

If you just want to see the API side by side, jump to
[Code, side by side](#code-side-by-side).

## Concept mapping

| Pinecone | Rostam | Notes |
|---|---|---|
| Account / project | Your Rostam server (or cluster) | You run it; no hosted control plane. |
| Index | **Collection** | `create_collection(name, dim, metric)`. |
| Metric `cosine` / `euclidean` / `dotproduct` | `metric="cosine"` / `"l2"` / `"dot"` | Set per collection at creation. |
| Vector (`id`, `values`, `metadata`) | Point (`id`, embedding, `metadata`, optional `content`) | Rostam ids are integers; see [IDs](#ids-string-vs-integer). |
| Namespace | A metadata field you filter on, or a separate collection | Rostam has no namespace primitive; both patterns work — see [Namespaces](#namespaces). |
| Metadata filter (`$eq`, `$in`, …) | `filters` helpers (`f.eq`, `f.in_`, …) | Full translation table [below](#translating-metadata-filters). |
| Serverless / pods | The single binary, or a Raft cluster | Scale by running more shards/nodes; see [Clustering](../server/clustering.md). |
| Managed, multi-tenant SaaS | You operate it | See [Security](../server/security.md) and [Backups](../server/backups.md). |

## Run Rostam

```bash
# one static binary; data stays on disk you control
rostam-server -http :8080 -data ./data -api-key "$ROSTAM_TOKEN"
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
c = Rostam("http://localhost:8080", api_key="optional-bearer-token")
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

# Rostam  (hits carry .id / .content / .metadata / .distance)
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
| `{"$and": [A, B]}` | `f.and_(A, B)` |
| `{"$or": [A, B]}` | `f.or_(A, B)` |
| implicit `{"a": 1, "b": 2}` (AND) | `f.and_(f.eq("a", 1), f.eq("b", 2))` |

Rostam runs filters through an **exact, filter-first path** — a selective filter
does not degrade recall. See [Filtering](../vector/filtering.md).

## Namespaces

Pinecone namespaces partition a single index. Rostam has no namespace primitive;
two patterns cover the same need:

1. **A metadata field.** Store the namespace as metadata (`metadata={"ns": "user-42"}`)
   and add `f.eq("ns", "user-42")` to every query. Simplest; one collection.
2. **A collection per namespace.** `create_collection("docs__user-42", ...)`. Use
   this when namespaces need independent lifecycles (drop one without touching others).

For true multi-tenant isolation with quotas, see
[Collections, tenants & aliases](../concepts/collections.md).

## IDs: string vs. integer

Pinecone ids are strings; Rostam point ids are integers. If your ids are already
numeric, use them directly. If they are strings, keep the original in metadata and
map to an integer id — the Python client's `TextStore`/framework adapters do this
for you (hash the string to a `uint64`, store the original under a reserved key,
strip it on return). For a hand-rolled migration, keep a `{"ext_id": "abc"}`
metadata field so you can look the original back up.

## Migrating your data

There is no import tool — you re-upsert. If you still have the source embeddings,
upsert them straight into Rostam. If not, read them out of Pinecone in pages and
write them across:

```python
# pull from Pinecone (list + fetch), push into Rostam
for ids in paginate_pinecone_ids(index):          # your paging over index.list()
    fetched = index.fetch(ids=ids)["vectors"]
    for pid, v in fetched.items():
        c.upsert("docs", int_id(pid), v["values"],
                 metadata={**v.get("metadata", {}), "ext_id": pid})
```

Re-embedding from source documents is often cleaner than exporting vectors, and it
lets you switch embedding models at the same time. Rostam can also do the embedding
for you — see `TextStore` and the built-in embedders in the
[Python client](../api/python.md).

## What is different (read before you commit)

- **You operate it.** No managed control plane — you run the server, back it up
  ([Backups](../server/backups.md)), and secure it ([Security](../server/security.md)).
  That is the point (your data, your boundary), but it is real work.
- **Integer ids** (see above).
- **No namespace primitive** (see above).
- Rostam returns **`distance`**, not a similarity `score`; smaller is closer. Convert
  if your app expects Pinecone-style scores.

## Next steps

- [Quickstart](../quickstart.md)
- [Search](../vector/search.md) · [Filtering](../vector/filtering.md) · [Hybrid & full-text](../vector/hybrid-search.md)
- [Clustering](../server/clustering.md) for HA, [Security](../server/security.md) for auth/TLS/RBAC
- Benchmarks vs. Pinecone and others: <https://rostamlabs.com/compare>
