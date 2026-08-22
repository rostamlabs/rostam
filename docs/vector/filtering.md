# Metadata filtering

Every search variant accepts a filter over point payloads. Filters are
structured trees, not query strings:

```go
f := vector.Filter{
	Op: vector.FilterAnd,
	And: []vector.Filter{
		{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")},
		{Op: vector.FilterGte, Field: "year", Value: vector.NewInt(2020)},
	},
}
hits, err := col.SearchFiltered(query, 10, f)
```

Over HTTP the same filter is JSON, with operators as lowercase names:

```json
{"op":"and","and":[
  {"op":"eq","field":"tenant","value":{"kind":"string","str":"acme"}},
  {"op":"gte","field":"year","value":{"kind":"int","int":2020}}
]}
```

## Operators

| Group | Operators |
|---|---|
| Composite | `and`, `or`, `not` |
| Comparison | `eq`, `ne`, `gt`, `gte`, `lt`, `lte` |
| Membership | `in` (value in array), `contains` (array field contains value) |
| Text | `match` (lightweight full-text), `regex` (RE2) |
| Presence | `is_empty`, `is_null` |
| Datetime | `dt_gt`, `dt_gte`, `dt_lt`, `dt_lte` (RFC 3339 bounds) |
| Geo | `geo_radius`, `geo_bounding_box`, `geo_polygon` |

Geo filters take a `geo` condition object instead of `value`:
`geo_radius` → `{center_lat, center_lon, radius_m}`; `geo_bounding_box` →
`{min_lat, min_lon, max_lat, max_lon}`; `geo_polygon` → `{polygon: [lat, lon, …]}`
(flat exterior ring).

## Payload values

Payload values are a tagged union. In Go, build them with constructors
(`NewString`, `NewInt`, `NewFloat`, `NewBool`, `NewStrings`, `NewInts`,
`NewFloats`, `NewGeo(lat, lon)`). Over raw HTTP, spell out the tag:

| kind | Type | JSON |
|---|---|---|
| `string` | string | `{"kind":"string","str":"acme"}` |
| `int` | int | `{"kind":"int","int":2020}` |
| `float` | float | `{"kind":"float","flt":0.5}` |
| `bool` | bool | `{"kind":"bool","bool":true}` |
| `strings` | []string | `{"kind":"strings","strs":["a","b"]}` |
| `ints` | []int | `{"kind":"ints","ints":[1,2]}` |
| `floats` | []float | `{"kind":"floats","flts":[1.5]}` |
| `geo` | geo point | `{"kind":"geo","lat":52.5,"lon":13.4}` |

The [Python client](../api/python.md) converts plain dicts to and from this
encoding automatically.

### NaN and range comparisons

A `float` payload value can be NaN, and NaN has no position in an ordering.
Rostam follows IEEE 754: **a NaN operand makes a range comparison unordered, so
`gt`, `gte`, `lt` and `lte` are all false.** This holds whether the NaN is the
stored field value or the filter's bound.

```
payload {"score": NaN}

score >= 0     → no match
score <= 0     → no match
score > 1e308  → no match
score < 1e308  → no match
```

Concretely: a point whose numeric field is NaN matches **no** range filter, and a
filter with a NaN bound matches **no** point. `is_null` and `is_empty` are
unaffected (a NaN is a present, non-null value), and so are `eq`, `ne` and `in`,
which compare with `==` — under which NaN was never equal to anything, including
itself.

This is the same rule Go, Rust, Milvus and Qdrant apply. It is *not* the
PostgreSQL/Lucene rule, where a total order sorts NaN above `+Inf` so that
`'NaN'::float8 > 1` is true; Rostam deliberately does not do that, because a
range index and a range predicate can only be made to agree on a value that has
an ordering, and inventing one leaves `x >= 3 AND x <= 2` matching a NaN row.

!!! warning "Changed in the m5 filter release"

    Before this change, a NaN field value was treated as *equal to every bound*,
    so `score >= b` and `score <= b` both **matched** it for any `b`. If your
    payloads contain NaN — most often from a division by zero or a failed
    numeric parse upstream — those points will stop appearing in `gte`/`lte`
    results. The payload index never agreed with the old behaviour (it excluded
    NaN from every range posting list), so filtered searches could already return
    different rows depending on which query path ran; the new rule is what makes
    both paths answer the same question. To keep such points matchable, write a
    real sentinel value instead of NaN.

## Building filters from Python

`rostam.filters` has helpers for the operators you reach for most:

```python
from rostam import Rostam, filters as f

c = Rostam("http://localhost:8080")
query = [0.1, 0.2, 0.3, 0.4]   # your embedding model's output

c.search_docs("docs", query, k=5, filter=f.eq("tenant", "acme"))
c.search_docs("docs", query, k=5, filter=f.in_("tenant", ["acme", "beta"]))
c.search_docs("docs", query, k=5,
              filter=f.and_(f.gte("year", 2021), f.eq("tenant", "beta")))
```

Helpers exist for `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in_`, `contains`,
`and_`, `or_` and `not_` — note the trailing underscore on the three that would
otherwise collide with Python keywords.

**The rest of the operator table has no helper**, including `match`, `regex`,
`is_empty`, `is_null`, the datetime bounds and the geo predicates. They are not
out of reach: a filter is just a dict, so spell out the JSON form and pass it
directly.

```python
# Continues from the client and `query` above.
# Full-text match — no helper, so write the wire form.
c.search_docs("docs", query, k=5, filter={
    "op": "match", "field": "$content",
    "value": {"kind": "string", "str": "gamma"},
})
```

Note that raw dicts take the **tagged** value encoding shown above; only the
helpers accept plain Python values.

## The filter-first planner (why filtered recall doesn't collapse)

Filtered ANN has a classic failure mode: **post-filtering** (search the graph,
then discard non-matching hits) collapses recall as filters get selective —
with a 0.1 % filter, a k=10 search needs ~10,000 graph hits to find 10 matches.
Filter-aware graph traversal keeps recall but latency explodes on selective
filters.

Rostam takes a third path. Index-narrowable filters — `eq`, `in`, `contains`,
the numeric ranges `gt`/`gte`/`lt`/`lte` and their `dt_*` datetime forms — are
backed by a **payload index**; at query time the planner estimates the filter's
match-set:

- **Selective filter** (match-set below the threshold): take the **filter-first
  path** — materialize the exact match-set from the payload index and score it
  by brute force. The result is *exact*, and small match-sets make it fast.
- **Broad filter**: use graph traversal with filter checks, where recall is not
  under threat.

The broad path does not re-evaluate the filter from scratch per candidate. The
planner folds the same narrowing plan into a per-query **admission bitset** and
consults one bit per candidate; for a high-pass-rate filter the bitset is built
from the filter's cheaper *complement* side. Numeric range predicates go one
step further: a **column sidecar** (one `float64` per point per range-queried
field, built lazily on the first range query, at most eight fields with LRU
eviction) answers the comparison from a single array read. The sidecar counts
against `MaxBytes`, and writes always win — an insert that needs the bytes
reclaims it. The `filter_gates_total`, `filter_complement_gates_total`,
`filter_column_gates_total` and `filter_column_drops_total` counters show which
acceleration a filtered search used.

Tuning (per collection):

| Config | Default | Meaning |
|---|---|---|
| `FilterFirstThreshold` | 10,000 | absolute match-set size below which filter-first engages |
| `FilterFirstRelativeBP` | 0 (off) | relative gate in basis points of live size; effective limit = max(absolute, min(BP·live/10000, 1M)) |

The reserved `$content` field (document text) is excluded from the payload
index. The TTL sweeper keeps the index consistent as points expire.

Run [`examples/filtered-recall-cliff`](https://github.com/rostamlabs/rostam/tree/main/examples/filtered-recall-cliff)
to see the effect measured: at 0.1 % selectivity the filter-first path is both
exact and orders of magnitude faster than filter-aware graph traversal.
