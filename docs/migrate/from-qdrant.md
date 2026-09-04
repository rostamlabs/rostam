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
| `QdrantClient(url=…)` | `Rostam(url, api_key=…)` | REST on `:8080`; native TCP on `:7000` is opt-in (start with `-tcp :7000`). |
| Collection | **Collection** | `create_collection(name, dim, metric)`. |
| `VectorParams(size, distance)` | `dim=…, metric=…` | Distance mapping [below](#distance-metrics). |
| `PointStruct(id, vector, payload)` | Point (`id`, embedding, `metadata`, optional `content`) | `payload` → `metadata` (scalars + homogeneous scalar lists); ids are integers, see [IDs](#ids). |
| `Filter` / `FieldCondition` | `filters` helpers (`f.eq`, …) | Full table [below](#translating-filters). |
| Multitenancy via a payload field | **Tenant** (`<tenant>/<collection>`) | A real primitive + optional auth boundary; see [Multitenancy](#multitenancy). |
| `ScoredPoint.score` | `hit.distance` (and `hit.score`) | Rostam's `distance` is unambiguous (smaller = closer); see [Scores](#scores-vs-distances). |
| Cluster / sharding | Raft cluster, online resharding | [Clustering](../server/clustering.md). |

## Run Rostam

```bash
# one static binary; no separate services to run. Pass the token via env, not a
# flag (a -api-key flag secret leaks via /proc and shell history):
export ROSTAM_API_KEY="a-strong-token"
rostam-server -http :8080 -data ./data          # add -tcp :7000 to also serve the native TCP protocol
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

# Rostam  (payload -> metadata: scalars or homogeneous scalar lists; content is an optional stored text payload)
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
| `MatchAny(any=[…])` on a **scalar** field | `f.in_("g", [...])` |
| `MatchAny(any=[…])` on an **array** field | `f.or_(f.contains("g", a), f.contains("g", b), …)` |
| `MatchExcept(except=[…])` (scalar field) | `f.not_(f.in_("g", [...]))` |
| `range=Range(gte=3, lte=9)` | `f.and_(f.gte("n", 3), f.lte("n", 9))` |
| `Filter(must=[A, B])` | `f.and_(A, B)` |
| `Filter(should=[A, B])` | `f.or_(A, B)` |
| `Filter(must_not=[A])` | `f.not_(A)` |

Two mapping caveats worth checking against your data:

- **`MatchAny` semantics depend on the field.** On a scalar field it means "value is
  one of these" → `f.in_`. On an **array** payload field it means "the array contains
  any of these" → use `f.contains` (which tests array membership), OR'd across the
  values. Rostam's `f.in_` is for scalar fields.
- **`is-empty` / `is-null`** *are* supported by the engine — the Python `filters`
  helpers just don't have a dedicated builder for them. A filter is a plain dict, so
  pass the raw predicate: Qdrant's `IsEmpty(key="x")` → `{"op": "is_empty", "field": "x"}`
  (field absent, null, `""`, or empty array) and `IsNull(key="x")` →
  `{"op": "is_null", "field": "x"}` (present and explicitly null). These compose with
  the `f.*` helpers (`f.and_`, `f.not_`, …) like any other predicate.

Rostam runs filters through an **exact, filter-first path**, so a selective filter
does not degrade recall. See [Filtering](../vector/filtering.md).

## Multitenancy

Qdrant's recommended multitenancy is a single collection with a tenant payload
field and a payload index. Rostam supports that pattern (store the tenant in
`metadata` and add `f.eq("tenant", "user-42")` to every query), and it also has a
first-class **tenant** primitive: collection names can be written
`<tenant>/<collection>`.

```python
c.create_collection("user-42/docs", dim=384, metric="cosine")
c.search_docs("user-42/docs", v, k=5)
```

For an **enforced** isolation boundary (a key that can only see its own tenant),
the `<tenant>/<collection>` naming alone is not enough — you need **tenant-bound
keys**. Issue keys via `-keys-file` (per-key RBAC, each key carrying a tenant) and
run with `-tenant-isolation`; then that boundary is enforced after scope checks.
The single static `-api-key`/`ROSTAM_API_KEY` used elsewhere in this guide is a
**superuser** and sees every tenant, so use it for setup, not for tenant isolation.
See [Security](../server/security.md#tenant-isolation) and
[Collections, tenants & aliases](../concepts/collections.md).

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

!!! warning "Mixed integer and UUID ids in one collection"
    If a source collection mixes plain integer ids with UUIDs, writing ints verbatim
    while hashing UUIDs puts both in the same `uint64` space, where a hashed UUID
    could collide with a verbatim int. For a mixed collection, hash **every** id
    (`to_uint64(str(p.id))`) so they share one consistent mapping, and keep the
    original in metadata.

## Migrating your data

There is no import tool — you re-upsert. Page through the source with Qdrant's
`scroll` and write each point across. Rostam metadata values are scalars
(str / int / float / bool) **or homogeneous scalar lists** (all-int, all-float, or
all-str — these stay filterable with `f.contains`). What it can't store is a nested
object, a mixed-type or empty list, or an explicit null, so coerce only those before
upserting:

```python
import json
from rostam._ids import to_uint64

def _scalar_list(v):
    """True if v is a non-empty, homogeneous int/float/str list Rostam can store."""
    v = list(v)
    if not v or all(isinstance(x, bool) for x in v):   # no bool-list kind
        return False
    return (all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            or all(isinstance(x, float) for x in v)
            or all(isinstance(x, str) for x in v))

def scalarize(payload):
    out = {}
    for k, v in (payload or {}).items():
        if isinstance(v, (bool, int, float, str)):
            out[k] = v                                 # scalar -> as-is
        elif isinstance(v, (list, tuple)) and _scalar_list(v):
            out[k] = list(v)                           # keep the array; f.contains still works
        else:
            out[k] = json.dumps(v)                     # nested / mixed / empty / null -> string
    return out

next_page = None
while True:
    points, next_page = client.scroll("docs", limit=256, offset=next_page,
                                       with_vectors=True, with_payload=True)
    for p in points:
        meta = scalarize(p.payload)
        meta["qdrant_id"] = str(p.id)       # preserve the original id
        c.upsert("docs", to_uint64(str(p.id)), p.vector, metadata=meta)
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
- **Scalar or homogeneous-scalar-list metadata** — coerce only nested/mixed/null
  payload values (see above).
- **No server-side embedding** (`is-empty`/`is-null` *are* supported — see Filters).
- **Named/sparse vectors:** Rostam does hybrid dense+sparse and full-text (BM25) —
  see [Hybrid & full-text](../vector/hybrid-search.md) — but the API shape differs
  from Qdrant's named vectors; check that doc before porting a multi-vector schema.

## Next steps

- [Quickstart](../quickstart.md)
- [Search](../vector/search.md) · [Filtering](../vector/filtering.md) · [Hybrid & full-text](../vector/hybrid-search.md) · [Quantization](../vector/quantization.md)
- [Clustering](../server/clustering.md) · [Security](../server/security.md)
- Benchmarks vs. Qdrant and others: <https://rostamlabs.com/compare>
