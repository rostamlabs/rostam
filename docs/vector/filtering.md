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
| Record | `row_exists`, `row_absent` (table-row presence inside a [record payload value](#record-paths)) |

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

## Record paths

`operate` (the KV store's server-side atomic multi-field update op, see
[the KV overview](../kv/overview.md#atomic-multi-field-updates-operate))
writes a **record**: a struct of scalar fields where a field may be a table
of keyed rows of columns. A vector payload can carry one of these records
verbatim as a payload value of kind `record`:

```json
{"kind": "record", "rec": "<base64 of the operate record bytes>"}
```

`rec` is exactly the bytes `operate` stores for the key — copy them in as-is
(e.g. from a KV `get`'s raw value, or the Go client's typed operate result),
never hand-built. A record value is never `is_empty`/`is_null` unless it is
literally absent or zero-length; a present, non-empty record is neither.

Over HTTP, gRPC or the binary protocol the record you read back is a copy the
codec wrote, and it is yours. In the **in-process Go API** it is not: a payload
returned by a read shares its bytes with the stored record, exactly as a
list-valued payload shares its slice. Read it, never write through it — mutating
it changes the live record while the index still describes the old bytes. Copy
the slice if you need to keep or edit it.

Every write that stores a record value is **validated at ingest** — insert,
insert-if-absent, upsert, `set_payload`, `overwrite_payload` and the bulk
paths alike. Bytes no operate engine could open are refused with a 400 before
anything changes (`vector: record payload value is not a decodable operate
record`), and so is a record above the 16 MiB storage cap. A mis-encoded `rec`
is therefore a rejected request, not a stored value that misbehaves later.

### Path grammar

A filter's `field` string is tried as an exact payload key first. Only when
there is no exact match, and the field contains a `/`, is it split at the
**first** `/` into a payload key and a path into that key's record (a literal
payload key that itself contains `/` — including one that happens to look
like a path, e.g. `"a/b"` — always wins over this splitting). The path is
1-3 `/`-separated segments:

| Segment | Form | Names |
|---|---|---|
| field | `name` or `#N` | a top-level record field, by name or position (`N` ≤ 65535, written canonically: at most 5 digits, no leading zeros except `#0`) |
| row key | decimal digits, or `"…"` | a table row: decimal for an integer key type, quoted (`\"`/`\\` escapes) for a `FIXED` key type |
| column | `name` or `#N` | one column of the row named by the previous segment |

A field segment may instead end in `#count` (e.g. `session/b#count`), which
must be the only segment — it reports a table field's row count and does not
compose with a row or column segment.

For a **dynamic**-mode record, table row keys have no declared width: a
decimal segment denotes the 8-byte little-endian encoding of the number (what
the Go client's `client.KeyU64` produces), and a quoted segment denotes its
unescaped bytes verbatim. For a **schema**-mode record, the row key's type and
width come from the schema, and a decimal segment denotes that width's
little-endian encoding.

Which spelling applies follows the record: a **schema**-mode record always
accepts positions (`session/#0`, `session/#5/42/#1`), and accepts names only
when its schema stores them — a names-less schema is addressable by position
alone. A **dynamic**-mode record is the mirror image: names only, never a
position, since it has no schema to number its fields against.

A path that cannot apply — an unknown field, an absent row or column, a
row/column segment against a scalar field, a *name* against a schema that
stores no names, a *position* against a dynamic record, or a payload key that
holds something other than a record — evaluates as "no such field": no match,
never a filter error. `row_exists`
and `row_absent` are the one exception: their `field` string's path **shape**
is validated once, at filter-compile time (see below), because that shape
depends only on the string, never on any point's data.

### Example

```
payload {"session": <record>}
  #0 rc   U8    = 7
  #1 bc   U8    = 3
  #2 hist U32   = 1234
  #3 bal  I32   = -5
  #4 tag  BYTES = "de"
  #5 b    TABLE(key U64; hi U32, lo U32) = {42: (hi=500, lo=1), 99: (hi=7, lo=2)}
```

```json
{"op": "gt", "field": "session/rc", "value": {"kind": "int", "int": 5}}
{"op": "eq", "field": "session/b/42/hi", "value": {"kind": "int", "int": 500}}
{"op": "gte", "field": "session/b#count", "value": {"kind": "int", "int": 1}}
{"op": "row_exists", "field": "session/b/42"}
{"op": "row_absent", "field": "session/b/7"}
```

### `row_exists` / `row_absent`

Both take a `field` whose path names exactly a table row — a field segment
then a row segment (`table/rowKey`), nothing shorter or longer; `CompileFilter`
rejects any other shape (a bare field, a `#count`, or a row **and** column) as
a compile error, since that shape never depends on the data. `row_exists` is
true iff the row is present. `row_absent` is **not** simply "not
`row_exists`": it is true only when the row is affirmatively missing from a
real table on that point (the payload key holds a record and the named field
really is a table). All of the following make **both** ops false:

- the field string matches an **exact payload key** on that point — a literal
  key wins over splitting, exactly as it does for every other op;
- the payload key is missing, or holds something other than a record;
- that point's record has **no such field at all**: an unknown field name, or
  a position past the end of its schema;
- the path cannot apply to that point's record shape, e.g. the field holds a
  scalar, not a table;
- the record is **malformed** where it matters — for `row_absent`, a table
  whose rows are not in the ascending key order the format requires. Absence is
  proved by walking the table, not inferred from an ordering the bytes merely
  claim, so a damaged table asserts nothing. (`row_exists` needs no such check:
  a matched row is proof of itself.)

"The row is absent" presupposes a table to search, and a point that cannot
even be asked the question is not in a position to assert that.

### Value mapping

A resolved cell (or a `#count`) maps to a payload value the same comparators
`eq`/`ne`/`gt`/`gte`/`lt`/`lte`/`in` use everywhere else:

| Stored type | Payload kind |
|---|---|
| `U8`/`U16`/`U32`/`I8`/`I16`/`I32`/`I64`/`IVARINT`, and `U64`/`UVARINT` up to `MaxInt64` | `int` |
| `U64`/`UVARINT` above `MaxInt64` | `float` (lossy — the only wider payload kind) |
| `F32`/`F64` | `float` |
| `BYTES`/`FIXED` | `string` (the raw bytes) |
| `#count` (a table's row count) | `int` |

`contains`, `match`, and `regex` read a record path's resolved **string**
cell (`BYTES`/`FIXED`) the same way they read any string payload field, via
the live record — they have no index of their own over record content (see
below).

### What's indexed

A collection auto-indexes, for every point whose payload carries a record —
no configuration needed — each top-level **scalar** field by name and each
table field's row count, as synthetic payload fields `<payloadKey>/<field>`
and `<payloadKey>/<table>#count`, with the same `eq`/`in`/range acceleration
a literal payload field gets.

**Not indexed** — always evaluated against the live record instead, still
correct, just not accelerated:

- a positional field (`session/#0`) — even when the record's schema would
  resolve it, since one collection can mix a names-carrying schema (indexed
  by name) and a names-less one (indexed positionally), and a field string
  alone can't say which a given point uses;
- a table row, a table column, or the table field itself (`session/b`,
  `session/b/42`, `session/b/42/hi`);
- `contains`, `match`, `regex`, `row_exists`, and `row_absent` on any record
  path — a record's scalars post whole-value **equality** keys only, never
  tokens, contains-elements, or geo cells.

### Fail-closed indexing

If any live point stores a **malformed** record under a payload key —
bytes that fail to enumerate in full, even though some of its fields would
still resolve individually — every path under that payload key stops using
the index for **every point**, for as long as the malformed record survives:
filtered search and delete/scroll selection fall back to evaluating the live
record, which stays correct. The key regains acceleration automatically once
the malformed point's payload is repaired, cleared, or the point itself is
reclaimed; it is a tracked state, not a one-way trip.

Ingest validation (above) means a write can no longer put the collection into
this state: every entry that stores a record value refuses one that cannot be
decoded. The fail-closed machinery remains as the backstop for records written
before that gate existed and for anything restored from an older snapshot or
log, which replay must never refuse. Produce records only through `operate` or
its SDK encoder, never by hand.

### Known limits

- A literal payload key whose name merely **looks like** a path (e.g. a
  string field literally named `"a/b"`) still resolves and indexes exactly —
  but the planner decides whether to accelerate a filter purely from the
  field *name*'s shape, which cannot tell a literal key from a record path
  apart. How much acceleration such a key loses depends on the shape its
  tail parses as:
    - **One indexed segment** (`a/b`, `a/b#count`) — keeps `eq`, `in` and
      range acceleration, loses `match`/`contains`/geo.
    - **Anything else that still parses as a path** — a multi-segment tail
      (`metrics/q1/42`, a row or a cell) or a positional segment
      (`a/#0`) — loses index acceleration for **every** operator, `eq` and
      range included, because those shapes are not posted at all.
    - **A tail that is not a valid path** (e.g. `metrics/2024/q1`, whose
      middle segment is not a legal row key) is treated as an ordinary
      literal key and keeps full acceleration.
  This is a deliberate, accepted phase-1 cost. In every case the filter
  still answers **correctly**; it just falls back to graph traversal instead
  of the payload index for the operators it cannot narrow.
- A positional field (`#N`) is always evaluated live, regardless of
  selectivity, per the indexing rule above.

## Updating a record in place

`vector_operate` runs the same
[`operate` op-list](../kv/overview.md#atomic-multi-field-updates-operate) the
KV store uses, but against the record held in one point's payload instead of
a KV key: one atomic op-list, applied to one point's record, under the
collection's write lock, in one apply. The op list, the paths into a record,
the scalar types and the returns are **exactly** `operate`'s — this page does
not repeat that grammar; see the KV page for the ops, types, `CHECK`/`SET`/
`ADD`/table semantics, and the returned `OperateResult`.

A call is named by four things: the **collection**, the point's **ID**, the
**payload key** the record lives under, and the **op-list** (`*wire.OperateArgs`)
to run. The Go typed client's [`Collection.Operate`](../api/go-client.md)
also accepts an optional CAS precondition on the point's version, same as
`SetPayload`.

Two fields `operate` normally carries do not apply here and are **rejected**,
not ignored:

- The op-list's inner **key** must be empty. The target is already named once,
  by `(collection, id, payloadKey)`; a caller who set it too would be a second,
  silently-ignored source of truth for what the call touches.
- **`ttlMode`/`ttl`** must stay at their zero values (`OperateTTLKeep`, `0`). A
  point's TTL belongs to the point, set by `Upsert`/`Insert`/`Expire` — not to
  a record living inside one of its payload values — so a caller who set a TTL
  here expecting it to take effect would otherwise get silence.

`create` still applies as it does for `operate`: `vector_operate` may create
the record if the payload key is absent. `create = NONE` against a payload key
that holds no record is an **error**, not a silent no-op — the caller declined
to create one and there was none. A payload key that holds something other
than a record (a plain string, number, or the reserved document-content key)
is also an error: silently replacing a value the caller can still read through
`get`/search would destroy data on a call the caller believes only touches a
record.

A stored record `vector_operate` cannot open fails the call and leaves the
point exactly as it was — no partial mutation, no version bump. **From this
release, every write that stores a record value is validated at ingest** (see
[Record paths](#record-paths) above), so a malformed stored record can now
only be one written by an earlier version, before that gate existed. The
mutation path itself does not re-validate the bytes it produces on a
successful op-list: they come from the same `operate` engine that already
holds them well-formed by construction, and re-decoding them on every call
would double the cost of a plain counter increment for a case that cannot
arise from a well-formed input. A failed `CHECK` — the op-list's own
conditional — leaves the record untouched: it bumps no version and writes no
WAL record, exactly like a `CHECK`-failed `operate` against a KV key.

A record's top-level fields are indexed the same way [Record paths](#record-paths)
describes for any record-valued payload: a mutation reindexes the point in the
same write-lock critical section that applies it, so a filtered search sees
the new field values immediately — there is no separate reindex step to wait
on. The record is still bound by the same 16 MiB storage cap every other
record-writing entry point enforces.

### Clocks and per-key deadlines

The **only** clock any op consults is the applying transaction's stamp, exactly
as for a KV [`operate`](../kv/overview.md#atomic-multi-field-updates-operate):
never a wall clock. On an unreplicated or otherwise unstamped apply that stamp
is `0`, so an unstamped `STAMP` writes `0` — the same caveat the KV page
records, and with the same consequence for a `MIN_COL`/`MAX_COL` "LRU" column
built on it.

The payload key's own **per-key deadline** (set by `set_payload`'s `keyTTLMs`)
is judged on the same terms. On a **stamped** apply a key whose deadline has
passed reads as absent: the op-list starts a fresh record and the stale
deadline is dropped. On an **unstamped** apply the deadline is not consulted at
all — the record is mutated in place and its deadline is passed through
unchanged.

That asymmetry is deliberate. Every replica runs the op-list itself, so if an
unstamped apply judged the deadline against its own wall clock, a replica past
the deadline would create a fresh record while a replica short of it
incremented the stored one, and the two would hold different bytes forever. A
wall-clock read is only safe where it can move a deadline *value* (as
`set_payload` does, at a bounded millisecond skew), not where it decides
whether a record exists. The key remains invisible to reads until its deadline
is refreshed either way.

`vector_operate` is **refused while the collection is resharding** (a
partitioned collection whose shard count is changing dual-writes every op to
both the old and new shard generation; `set_payload`-style dual-writing is
safe because storing the same payload twice is idempotent, but an op-list's
`ADD`/`MUL`/`SHL` are not — the two copies could disagree by an arbitrary
number of increments and the disagreement would survive cutover). The refusal
is retryable after the reshard completes. It is a check against the local
catalog, not a fence: a reshard that begins in the single dispatch between
that check and the call's apply is not caught, so the refusal covers a
reshard already in progress and races the instant one begins by one dispatch.

```go
schema := &wire.Schema{Version: 1, StoreNames: true, Fields: []wire.FieldDef{
    {Name: "rc", Type: wire.OperateTypeU8},
    {Name: "bc", Type: wire.OperateTypeU8},
    {Name: "hist", Type: wire.OperateTypeU32},
    {Name: "bal", Type: wire.OperateTypeI32},
    {Name: "tag", Type: wire.OperateTypeBytes},
    {Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{
        KeyType: wire.OperateTypeU64,
        Cols: []wire.ColumnDef{
            {Name: "hi", Type: wire.OperateTypeU32},
            {Name: "lo", Type: wire.OperateTypeU32},
        },
    }},
}}

args, err := client.NewOperate(nil).WithSchema(schema).
    Add(client.F("rc"), 1).
    Return(client.F("rc")).
    Count(client.F("b")).
    Args()
if err != nil {
    return err
}
found, res, err := posts.Operate(ctx, client.OperateRequest{
    ID: 1, PayloadKey: "session", Args: args,
})
// found: false if the point is absent, tombstoned or expired (res is nil then)
// res.Values[0]: the new "rc", res.Values[1]: the "b" table's row count
```

This mirrors `TestVectorOperateAgainstSchemaRecord` (`ops/vector_operate_test.go`):
an `ADD` on a schema-mode `rc` field survives beside an untouched `b` table,
and both are read back in the same round trip.

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
