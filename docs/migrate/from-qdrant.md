# Migrating from Qdrant to Rostam

This guide maps Qdrant concepts and API calls to Rostam so you can move an
existing workload over. Both are self-hostable, so the usual reasons for the
switch are operational: one Go binary with **no external dependencies** and a
second built-in engine (a sub-µs key-value store) in the same process, plus
throughput ahead of Qdrant at matched recall
([benchmarks](https://rostamlabs.com/compare)).

!!! note "Egress — both paths are under your control"
    Rostam makes no outbound calls on its own. Data leaves only through egress you
    configure: S3 backups / a cold tier ([Backups](../server/backups.md)) — point
    it at an in-boundary object store to stay fully local — and, separately, a
    **hosted embedder** (OpenAI, Cohere, …), which sends your text out at the
    embedding step regardless of where vectors live. Embed with a local model for a
    fully in-boundary pipeline.

Jump to [Code, side by side](#code-side-by-side).

## Concept mapping

| Qdrant | Rostam | Notes |
|---|---|---|
| `QdrantClient(url=…)` | `Rostam(url, api_key=…)` | REST on `:8080`, native TCP on `:7000`. |
| Collection | **Collection** | `create_collection(name, dim, metric)`. |
| `VectorParams(size, distance)` | `dim=…, metric=…` | Distance mapping [below](#distance-metrics). |
| `PointStruct(id, vector, payload)` | Point (`id`, embedding, `metadata`, optional `content`) | `payload` → `metadata`; ids are integers, see [IDs](#ids). |
| `Filter` / `FieldCondition` | `filters` helpers (`f.eq`, …) | Full table [below](#translating-filters). |
| Multitenancy via a payload field | **Tenant** (`<tenant>/<collection>`) | A real primitive + optional auth boundary; see [Multitenancy](#multitenancy-tenants). |
| `ScoredPoint.score` | `hit.distance` (and `hit.score`) | Rostam's `distance` is unambiguous (smaller = closer); see [Scores](#scores-vs-distances). |
| Cluster / sharding | Raft cluster, online resharding | [Clustering](../server/clustering.md). |

## Run Rostam

```bash
# one static binary; no separate services to run. Pass the token via env, not a
# flag (a -api-key flag secret leaks via /proc and shell history):
export ROSTAM_API_KEY="a-strong-token"
rostam-server -http :8080 -data ./data
```

The server refuses to bind a reachable address without auth. See
[Running the server](../server/running.md) for TLS, clustering, and Docker.

```bash
pip install rostam-client
```

## Code, side by side

### Connect

```python
# Qdrant
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")

# Rostam
from rostam import Rostam, filters as f
c = Rostam("http://localhost:8080", api_key="a-strong-token")
```

### Create the collection

```python
# Qdrant
client.create_collection(
    "docs",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# Rostam
c.create_collection("docs", dim=384, metric="cosine")   # metric: cosine | l2 | dot
```

### Upsert

```python
# Qdrant  (payload holds your metadata)
client.upsert("docs", points=[
    models.PointStruct(id=1, vector=embedding, payload={"doc_id": 7, "lang": "en"}),
])

# Rostam  (payload -> metadata; content is an optional stored text payload)
c.upsert("docs", 1, embedding, content="the chunk text",
         metadata={"doc_id": 7, "lang": "en"})
```

### Query

```python
# Qdrant  (query_points; older clients use .search)
res = client.query_points(
    "docs", query=embedding, limit=5,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=7))]),
).points
for p in res:
    print(p.id, p.score, p.payload)

# Rostam  (hits carry .id / .content / .metadata / .distance)
hits = c.search_docs("docs", embedding, k=5, filter=f.eq("doc_id", 7))
for h in hits:
    print(h.id, h.distance, h.metadata)
```

### Retrieve and delete

```python
# Qdrant
client.retrieve("docs", ids=[1])
client.delete("docs", points_selector=[1])

# Rostam
c.get("docs", 1)      # -> Point | None
c.delete("docs", 1)
```

## Distance metrics

| Qdrant | Rostam |
|---|---|
| `Distance.COSINE` | `metric="cosine"` |
| `Distance.EUCLID` | `metric="l2"` |
| `Distance.DOT` | `metric="dot"` |
| `Distance.MANHATTAN` | no direct equivalent — use `l2`, or file an issue if you need L1 |

## Translating filters

Qdrant builds filters from `Filter` / `FieldCondition`; Rostam uses the `filters`
helpers (`from rostam import filters as f`).

| Qdrant | Rostam |
|---|---|
| `FieldCondition(key="g", match=MatchValue(value="x"))` | `f.eq("g", "x")` |
| `FieldCondition(key="g", match=MatchAny(any=["a","b"]))` | `f.in_("g", ["a", "b"])` |
| `FieldCondition(key="g", match=MatchExcept(**{"except": ["a"]}))` | `f.not_(f.in_("g", ["a"]))` |
| `FieldCondition(key="n", range=Range(gte=3, lte=9))` | `f.and_(f.gte("n", 3), f.lte("n", 9))` |
| `Filter(must=[A, B])` | `f.and_(A, B)` |
| `Filter(should=[A, B])` | `f.or_(A, B)` |
| `Filter(must_not=[A])` | `f.not_(A)` |

Qdrant's `IsEmptyCondition` / `IsNullCondition` (matching on key presence) have no
direct Rostam equivalent — Rostam filters match on values. Emulate by writing an
explicit sentinel field (e.g. a boolean `has_x`) and filtering on it.

Rostam runs filters through an **exact, filter-first path**, so a selective filter
does not degrade recall. See [Filtering](../vector/filtering.md).

## Multitenancy → tenants

Qdrant's recommended multitenancy is a single collection with a tenant payload
field and a payload index. Rostam offers that pattern too (store the tenant in
`metadata` and add `f.eq("tenant", "user-42")` to every query), but it also has a
first-class **tenant** primitive: collection names can be written
`<tenant>/<collection>`, and with `-tenant-isolation` a tenant becomes an
**authoritative security boundary** — an API key bound to a tenant can only see
that tenant's collections.

```python
c.create_collection("user-42/docs", dim=384, metric="cosine")
c.search_docs("user-42/docs", v, k=5)
```

See [Collections, tenants & aliases](../concepts/collections.md).

## Scores vs. distances

Qdrant returns a similarity `score` (for cosine, higher is closer). Rostam hits
carry `distance` (smaller is closer) — the unambiguous field to sort or threshold
on — and also expose a `score` field. Because score scaling can differ, prefer
`distance` when porting ranking/threshold logic, or re-tune thresholds against
Rostam's values.

## IDs

Qdrant point ids are unsigned integers or UUIDs; Rostam ids are integers
(`uint64`). Integer ids carry over directly. Map UUID/string ids deterministically
with the client helper and keep the original in metadata so you can read it back:

```python
from rostam._ids import to_uint64   # stable str -> uint64; the framework adapters use it
c.upsert("docs", to_uint64("2f1c…"), embedding, metadata={"qdrant_id": "2f1c…"})
```

`to_uint64` is one-way; neither the client nor `TextStore` preserves the original
string for you, so store it yourself (pick a metadata key your data doesn't use).

## Migrating your data

There is no import tool — you re-upsert. Page through the source with Qdrant's
`scroll` and write each point across, preserving the id:

```python
from rostam._ids import to_uint64

next_page = None
while True:
    points, next_page = client.scroll("docs", limit=256, offset=next_page,
                                       with_vectors=True, with_payload=True)
    for p in points:
        meta = dict(p.payload or {})
        pid = p.id if isinstance(p.id, int) else to_uint64(str(p.id))
        if not isinstance(p.id, int):
            meta["qdrant_id"] = str(p.id)
        c.upsert("docs", pid, p.vector, metadata=meta)
    if next_page is None:
        break
```

Re-embedding from source documents is often cleaner than exporting vectors, and it
lets you change embedding models at the same time — Rostam can embed for you via
`TextStore` (see the [Python client](../api/python.md); note the egress caveat for
hosted embedders).

## What is different (read before you commit)

- **You operate it** — same as self-hosted Qdrant: run it, back it up
  ([Backups](../server/backups.md)), secure it ([Security](../server/security.md)).
- **`distance`, not `score`**, as the primary ranking field (see above).
- **Integer ids** — map UUIDs with `to_uint64`, keep the original in metadata.
- **No `is-empty`/`is-null` filter** and no server-side embedding (see above).
- **Named/sparse vectors:** Rostam does hybrid dense+sparse and full-text (BM25) —
  see [Hybrid & full-text](../vector/hybrid-search.md) — but the API shape differs
  from Qdrant's named vectors; check that doc before porting a multi-vector schema.

## Next steps

- [Quickstart](../quickstart.md)
- [Search](../vector/search.md) · [Filtering](../vector/filtering.md) · [Hybrid & full-text](../vector/hybrid-search.md) · [Quantization](../vector/quantization.md)
- [Clustering](../server/clustering.md) · [Security](../server/security.md)
- Benchmarks vs. Qdrant and others: <https://rostamlabs.com/compare>
