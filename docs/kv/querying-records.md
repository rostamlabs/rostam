# Querying records in the KV store

`get` answers "what is stored under this key". `kv_query` answers the other
question: **"which keys hold a record that matches this filter"** — using the
same JSON filter grammar the vector side uses, over the records
[`operate`](overview.md#atomic-multi-field-updates-operate) writes.

It is a lookup and reporting path, not a hot-key path. A query is a
scatter-gather over every shard group plus one verification read per candidate;
a `get` is one hash lookup. Reach for `kv_query` when you need "the sessions
whose `rc` is over 5", not when you already know the key.

To make it cheap you define a **KV index**: a named, cluster-wide definition
that says "for keys under this prefix, remember the value at this record path".

## Index definitions

A definition names four things:

| Field | Meaning | Cap |
|---|---|---|
| `name` | how a query names the index | 1–64 bytes from `[A-Za-z0-9_.:-]` |
| `key_prefix` | the keys this index covers; empty means the whole keyspace | 255 bytes |
| `payload_path` | the record path to remember, **one top-level field** (`rc`) or that field's row count (`b#count`) | 255 bytes |
| `kind` | `scalar` or `count`, and it must agree with the path | — |

A cluster holds at most **64** definitions. An index path is a single segment
by design: `rc` and `b#count` are indexable, `b/42/hi` is not. Row, column and
positional paths are still perfectly usable **in a filter** — they are just
evaluated against the live record rather than driving the candidate set. See
[record paths in filters](../vector/filtering.md#record-paths) for the full path
grammar, which `kv_query` shares.

Definitions are **cluster-wide state on the meta log**, like a collection's
catalog entry. You create one against any node; it is committed through
meta-Raft and every node picks it up. Creating and dropping are the *same*
write — a drop is a create with `enabled: false` — and both sit behind an
**admin** key, not the ordinary write bar, because dropping an index makes every
query naming it start failing at once and re-creating it costs a full cache walk
on every node.

### `building`, and the retryable error

A create returns once the definition is **committed**, not once it is usable.
Each node polls the meta catalog about once a second, installs the definition
into every shard group it hosts, and then walks that group's cache to fill the
postings. Until that walk finishes, the group's postings are a proper *subset*
of the truth, so answering from them would silently lose rows. The group refuses
instead:

```
kvindex: index is still building: "by_rc" on shard group 3 (backfill in progress; retry)
```

That is a **retryable** error (HTTP 503, gRPC `Unavailable`,
`client.ErrKVIndexBuilding`). A create-then-query lands here, and so does a
query issued while a definition is being redefined. Retry, or poll the list op
until the index reports ready.

**"Ready" means one replica per group said ready.** The list op asks each shard
group once — the local copy when this node hosts the group, otherwise one owner
— and ANDs the answers. It fails closed: a group still backfilling, a group that
never installed the definition, and a group that could not be reached all make
the bit false. A definition is never dropped from the list because nothing could
be said about it, since a name vanishing reads as "the create failed". Note the
consequence for `consistency: "any"`: a ready bit gathered from one replica per
group does not promise that *every* replica has finished, so a stale read may
still meet a backfilling replica and get the retryable building error.

## What the answer actually is

**An indexed query answers "the keys under this definition's `key_prefix` that
match the filter" — never "the keys that match the filter".** Keys outside the
prefix are outside the answer. They are not missing rows; they were never in
scope. Only `scan: true` sees them.

This scoping is the whole reason the answer can be exact while reading a
best-effort structure. Postings are **hints**: every candidate key is re-read
from the cache and the full filter is re-evaluated on the live value before a row
is returned. A stale posting therefore costs one wasted lookup and can never
produce a wrong row. A *missing* posting is the one failure that would lose a
row, and two rules prevent it — the readiness gate above, and the rule about
which filter leaf may narrow the search:

> The candidate set is driven only by a **positive** leaf (`eq`, `in`, `gt`,
> `gte`, `lt`, `lte`) on the index's **own path**, taken from the filter root or
> from a **top-level `and`**. Never from under an `or`, never from under a `not`,
> never from a nested `and`.

An `and` is false unless all its children are true, so a conjunct can narrow
soundly. A disjunct cannot: a record could satisfy the `or` through its other
branch, have no posting under this leaf's value, and be lost.

### Accelerated versus evaluated live

| Filter shape | Effect |
|---|---|
| `eq` on the index's path, with a scalar value | one map probe; the fastest case |
| `in` on the index's path | one probe per listed value |
| `gt` / `gte` / `lt` / `lte` on the index's path | walks the index's **distinct values** and unions the matching sets |
| the same leaf inside a top-level `and`, alongside anything else | drives the candidates; the rest is evaluated live |
| a leaf on any **other** path (including deeper paths like `b/42/hi`) | evaluated live on every candidate |
| every **negation** and absence test — `ne`, `not`, `is_empty`, `is_null` | evaluated live; never narrows |
| `contains`, `match`, `regex`, the `dt_*` and `geo_*` families, `row_exists`, `row_absent` | evaluated live; never narrows |
| a bound that is not a scalar — a list under `eq` or a range, a geo point, a record, an absent value | **declines to narrow**: the index is not consulted, and with no other leaf to drive it the query is refused with `filter needs an index or scan:true` |
| a `NaN` bound, or an ordering (`gt`/`gte`/`lt`/`lte`) against a **bool** | still drives the index, and the candidate set comes back **empty** — so the page is empty, which is what the predicate would have decided anyway |
| `in` whose value is not a list | a permanent invalid-filter refusal; the predicate itself will not compile |

When `eq` and a range are both available on the index's path, `eq` wins: one
probe beats a walk over every distinct value.

If **no** leaf can drive the named index, the query is refused rather than
quietly turned into a full walk:

```
ops: kv_query: filter needs an index or scan:true
```

### Filter grammar

The filter is the ordinary vector filter tree, with one difference: **fields are
bare record paths**, because a KV value *is* one record and there is no payload
map to key into. Write `rc`, `b#count`, `b/42/hi` — not `payload/rc`.

Internally the leaf presents the live value under the reserved name **`$rec`**
and rewrites every field to `$rec/<path>`, which is how the existing evaluator is
reused unchanged. The whole `$`-prefixed namespace is therefore reserved: a field
starting with `$` is a filter error, not a field that never matches.

A path the record cannot answer — an unknown field, a row that is not there, a
column against a scalar — is "no such field": no match, never an error. A field
that is not a legal path at all *is* an error, deliberately, so a typo like
`rc/` fails loudly instead of quietly answering "no matches" to a question it
never asked.

Filter caps: **64 KiB** of filter JSON, **256** nodes, **32** levels of nesting.

## Scanning without an index

`scan: true` is explicit consent to walk the whole keyspace of every shard
group. Use it for the ad-hoc question an index does not cover.

It is expensive in a specific, non-obvious way. The cache has no ordered index
and its iterator takes no start key, so **every page walks the entire shard**. A
page keeps the smallest `ScanChunk` keys above the cursor in a bounded heap,
verifies those in order, and continues from the largest key it retained. Paging a
keyspace of *n* keys to completion therefore costs about `ceil(n / ScanChunk)`
full walks — quadratic, and honestly so. With the default chunk of 10 000, a
million-key shard is a hundred full walks.

Above the per-page scan budget the answer is a refusal, not a short page:

```
ops: kv_query: scan budget exceeded; use an index
```

Refusing is the only sound option. A truncated *unordered* walk yields no
continuation, because the keys it did not reach are indistinguishable from the
keys that did not match.

## Pages: the row limit and the byte budget

Every page is bounded twice.

- **`limit`** is 1–1000 rows, defaulting to 100. There is no unbounded mode: a
  page is what bounds a fan-out read's memory on every node it touches.
- **The page byte budget** is 8 MiB, for one group's page and for the merged
  page alike (about 64 KiB of that is reserved for the frame's own header and
  continuation).

Either bound sets `More` and returns a cursor. The first row of a page is always
emitted, so a page always advances and paging always terminates.

**A value larger than a whole page comes back key-only.** A cache value may be up
to the 16 MiB page size while a query page caps at 8 MiB, so this is reachable
with ordinary data. Rather than dropping a true match or wedging the query, the
row is returned with its key and no value; fetch it with `get`. In `values` and
`records` mode a present-but-empty value still encodes as present, so "no value"
is unambiguous. The count of such rows is exported as a process counter
(`ops.KVQueryOversizeRows`); a rising rate means callers are being handed keys
they must fetch separately.

## Return modes

| `return` | What comes back |
|---|---|
| `keys` (default) | the matching keys, and nothing else — the cheap answer |
| `values` | each key with its raw stored bytes |
| `records` | the same raw bytes, with the server having **validated** that each one decodes as a record |

`records` does not ship a decoded tree. **The decode happens on the client**, by
design: the server has already established the bytes are a record, and building a
typed tree is work that does not belong on a node serving everyone else's queries
too. The native Go client does it for you and hands back `page.Records`,
row-aligned with `page.Rows`. Over REST, the bytes arrive in `value_b64` and
decoding them is yours to do — there is deliberately no decoded-record JSON
field, because that would commit the REST surface to a record-to-JSON shape
nobody has designed yet.

Over REST a row carries up to four fields, the same convention
`GET /v1/kv/{key}` already follows:

| Field | When it appears |
|---|---|
| `key_b64` | always — a key is arbitrary bytes |
| `key_utf8` | only when the key bytes are valid UTF-8 |
| `value_b64` | only under `values` or `records`, and only when the value fit the page |
| `value_utf8` | only when `value_b64` is present **and** those bytes are valid UTF-8 |

The `_utf8` fields are a convenience, never the authority: they are omitted
rather than lossily transcoded, so a client that always reads the `_b64` field
is always correct. A record's bytes are usually binary and `value_utf8` will
simply be absent; a small dynamic-mode record of ASCII values may happen to
qualify.

A value under `records` that is *not* a record is skipped, not an error: the KV
keyspace is shared, and one unrelated value must not fail a query over the
records around it.

## Paging with the composite cursor

The cursor is **one continuation per shard group**, and it is opaque: echo it
back verbatim. An empty cursor means the result set is exhausted, and that is the
only termination signal.

A group the previous page finished is dropped from the cursor and is simply not
asked again — on a long paging run that is most of the round trips saved.

What the cursor guarantees, and what it does not:

- **Rows are never lost or duplicated across pages.** A group whose rows were all
  cut by the limit resumes from exactly where it started.
- **Ordering is ascending by key bytes within a page.** It is *not* guaranteed
  across pages. A group whose own page was cut short — by its byte budget, or by
  a chunk of non-matching keys — can contribute keys on a later page that sort
  below keys another group already returned.
- **A page is not a snapshot.** Groups are read concurrently and each group's
  page is read at its own moment, so a write that lands mid-query may or may not
  be reflected. A key deleted after its group's page was built still appears in
  that page; a key written after a group passed it is picked up only if the write
  sorts above that group's continuation.
- **A continuation is the last key *examined*, not the last key returned.** A
  page can therefore come back with few rows (or none) and a cursor that has
  advanced a long way, because the keys in between were examined and did not
  match. That is normal; keep paging until the cursor is empty.

**The cursor has a ceiling.** It is capped at 4 MiB on the way back in, so the
coordinator refuses to build one it could not accept:

```
ops: kv_query: continuation exceeds the cursor cap: the continuation for N shard
groups needs X bytes, over the 4194304-byte cursor cap; ...
```

The codec stores the continuations' shared prefix once, so each group really
spends 7 bytes plus whatever its key holds *beyond* that prefix — about
`(4 MiB − 3)/groups − 7` bytes each. At 128 groups that is roughly 32 KiB of
distinguishing suffix per group against a 64 KiB maximum key: reachable only by
keys that are both enormous and share almost nothing after the index's prefix.
The remedy is a narrower filter or shorter keys; the message carries the
arithmetic.

## Read consistency

| `consistency` | Behaviour |
|---|---|
| `"leader"` (default) | each group's read is served by its Raft leader — best-effort, the ordinary routed-read semantic |
| `"any"` | any replica answers, including a local one; the cheapest read |
| `"linearizable"` | the serving leader runs a `VerifyLeader` barrier before answering |

The default is deliberately **not** the cheapest one. A `kv_query` page claims to
be a complete answer over the whole keyspace, and a stale replica silently
omitting matching rows is the one failure this feature may not have. Ask for
`"any"` by name when you want it.

One consequence on the native Go client: `wire.ConsistencyAnyReplica` *is* the
zero value, so `KVQuery` reads a zero `Consistency` as "the caller did not
choose" and promotes it to leader-only. To issue an any-replica read from Go,
encode the args and call `Call("kv_query", …)` directly.

There is a join window that makes `"any"` sharper than "slightly stale": a
replica that has just joined a group publishes its store before the snapshot
lands, so its backfill walks an empty cache and grants the definition *ready*
over nothing — and an `"any"` read routed there can come back as an exhausted
empty page rather than as a page that is merely behind. The leader-only default
avoids it, which is the second reason it is the default.

**A partial answer is a hard error.** If one group's leg fails, the whole query
fails with that group named. An under-complete page is a *wrong* answer, not a
partial effect: the caller cannot tell it from a complete one.

## Example over REST

Define an index over the `rc` field of every key under `session:`:

```bash
curl -s localhost:8080/v1/kv/indexes \
  -d '{"name":"by_rc","key_prefix_b64":"c2Vzc2lvbjo=","payload_path":"rc","kind":"scalar"}'
```

`key_prefix_b64` is base64 because a key prefix is arbitrary bytes;
`c2Vzc2lvbjo=` is `session:`. On a server with auth enabled this endpoint (and
the `DELETE` below) needs an **admin** key, not a write key — add
`-H 'Authorization: Bearer <admin token>'`. Poll until it is ready:

```bash
curl -s localhost:8080/v1/kv/indexes
```

```json
{"indexes":[{"name":"by_rc","key_prefix_b64":"c2Vzc2lvbjo=",
             "payload_path":"rc","kind":"scalar","ready":true}]}
```

Write some records the ordinary way (`operate`, or any `put` of record bytes),
then query:

```bash
curl -s localhost:8080/v1/kv/query -d '{
  "index": "by_rc",
  "filter": {"op":"and","and":[
    {"op":"gt","field":"rc","value":{"kind":"int","int":5}},
    {"op":"gte","field":"b#count","value":{"kind":"int","int":1}}
  ]},
  "limit": 100,
  "return": "keys"
}'
```

The `rc` leaf drives the index; the `b#count` leaf is checked live on each
candidate. The answer:

```json
{"rows":[{"key_b64":"c2Vzc2lvbjo0Mg==","key_utf8":"session:42"}],
 "cursor":""}
```

`key_utf8` rides along only when the key really is valid UTF-8, so a client never
mistakes lossy bytes for a string. An empty `cursor` means there is nothing more.
When it is non-empty, send it back untouched:

```bash
curl -s localhost:8080/v1/kv/query -d '{
  "index":"by_rc","limit":100,
  "filter":{"op":"gt","field":"rc","value":{"kind":"int","int":5}},
  "cursor":"<the cursor string from the previous page, verbatim>"
}'
```

Drop the index when you are done with it:

```bash
curl -s -X DELETE localhost:8080/v1/kv/indexes/by_rc
```

## Example from Go

```go
import (
	"errors"
	"time"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

c, err := client.NewRouted(client.Config{Servers: []string{"127.0.0.1:7000"}})
if err != nil { ... }
defer c.Close()

// Cluster-wide, committed through meta-Raft. Admin-scoped.
err = c.CreateKVIndex(ctx, wire.KVIndexDef{
	Name:        "by_rc",
	KeyPrefix:   []byte("session:"),
	PayloadPath: "rc",
	Kind:        wire.KVIndexKindScalar,
})

// Poll until every shard group has backfilled it.
defs, ready, err := c.ListKVIndexes(ctx)

args := wire.KVQueryArgs{
	Index:  "by_rc",
	Filter: vtypes.Filter{Op: vtypes.FilterGt, Field: "rc", Value: vtypes.NewInt(5)},
	Limit:  100,
	Return: wire.KVQueryReturnRecords,
}
for {
	page, err := c.KVQuery(ctx, args)
	if errors.Is(err, client.ErrKVIndexBuilding) {
		time.Sleep(200 * time.Millisecond)
		continue // the backfill has not reached every group yet
	}
	if err != nil {
		return err
	}
	for i, row := range page.Rows {
		rec := page.Records[i] // nil when the row came back key-only
		_ = row.Key
		_ = rec
	}
	if len(page.Cursor) == 0 {
		break
	}
	args.Cursor = page.Cursor
}
```

`Limit` defaults to 100 and `Consistency` to leader-only. `Records` is populated
only for `KVQueryReturnRecords`, is row-aligned with `Rows`, and holds `nil`
where a row came back key-only.

## Errors, and which ones to retry

Every transport classifies the same set of refusals the same way, so a caller
that moves from REST to gRPC does not discover that its retry loop has become a
hard failure.

| Refusal | Class | HTTP | gRPC | Binary / Go client |
|---|---|---|---|---|
| `kvindex: no such index` | not found | 404 | `NotFound` | `client.ErrKVIndexNotFound` |
| `kvindex: index is still building` | retryable | 503 | `Unavailable` | `client.ErrKVIndexBuilding` |
| `kvindex: index definition changed under the query` | retryable | 503 | `Unavailable` | `client.ErrKVIndexBuilding` |
| `ops: kv_query: shard is unavailable; retry` | retryable | 503 | `Unavailable` | `client.ErrKVQueryUnavailable` |
| `shard: store is closed` | retryable | 503 | `Unavailable` | `client.ErrKVQueryUnavailable` |
| `ops: kv_query: invalid filter` | permanent | 400 | `InvalidArgument` | `client.ErrKVQueryFilter` |
| `ops: kv_query: filter needs an index or scan:true` | permanent | 400 | `InvalidArgument` | `client.ErrKVQueryFilter` |
| `ops: kv_query: scan budget exceeded; use an index` | permanent | 400 | `InvalidArgument` | `client.ErrKVQueryFilter` |
| `ops: kv_query: continuation exceeds the cursor cap` | permanent | 400 | `InvalidArgument` | `client.ErrKVQueryFilter` |
| `kvindex: candidate budget exceeded` | permanent | 400 | `InvalidArgument` | `client.ErrKVQueryFilter` |
| `ops: kv_query: no KV index on this dispatcher` | permanent | 400 | `InvalidArgument` | `client.ErrKVQueryFilter` |
| a malformed args, filter or result frame | permanent | 400 | `InvalidArgument` | the codec's own error |

Two of these deserve a note.

**"No such index" can become retryable.** A shard group knows only its own
installed definitions, so a name it has never seen is permanent *to it*. The
coordinator knows better: if the name is in the meta catalog, it rewrites the
refusal as `index is still building`. That is what makes create-then-query a
retry rather than a hard failure.

**`no KV index on this dispatcher`** means this deployment has no KV index layer
wired at all — every embedder predating the feature. It is permanent, but it is a
fact about the deployment, not about your query.

**A definition counted in `RejectedDefs`** is in the catalog but unbuildable on
this node, so its postings do not exist here. Creating one is refused before the
commit, so the only way to reach this state is build skew: a node whose payload-
path grammar is wider wrote a definition an older node cannot parse. Querying it
on a node that rejected it is a **permanent** error naming the definition
(`invalid kv index definition`), not the retryable "still building" — an index
that cannot be built here will never finish building, and a client told to retry
would retry forever at one full fan-out per attempt. The coordinator answers for
*itself*: if a remote group's older binary is the one that rejected it while the
coordinator can build it, the query is still reported as retryable, because a
peer's reject state is not carried back in the leaf reply. Upgrade the lagging
node, or drop the definition.

During a **rolling upgrade** that means the answer depends on which node you ask.
A coordinator that cannot build the definition refuses permanently, and it does
so even when it hosts the group and an upgraded replica of that same group could
have served the read: a hosted copy is preferred locally, `consistency: "any"`
included, so the upgraded replica is never consulted. Issue the query at an
upgraded node, or wait for the upgrade to finish.

## Operating it

`Stats.KVIndex` makes the index observable, which matters because it is derived
state: a definition can be committed cluster-wide and still be doing nothing on a
given node.

| Field | Meaning |
|---|---|
| `Definitions` | definitions installed on this node; lags the catalog by up to one observe interval |
| `Ready` | how many are ready on **every** group this node hosts |
| `Backfills` | completed definition walks since start |
| `BackfillKeys` | cache entries visited by walks, completed or not — rising while `Backfills` is flat means walks are being abandoned |
| `Rejects` | monotonic count of reject *events*: definitions the meta FSM accepted that this node cannot build, one per definition per pass |
| `RejectedDefs` | gauge: definitions in the catalog right now that this node cannot build. Non-zero is a standing misconfiguration or a version skew — **this is the one to alert on** |
| `VerifyMisses` | candidates whose live re-read missed. A cost, never a wrong answer; rising means keys are leaving the cache by a path that does not reach the index |
| `ReconcileDrops` | postings the reconciler removed because their key is no longer live |

`VerifyMisses` counts **key** staleness only. A candidate whose key is live but
whose value no longer satisfies the filter is discarded by the predicate and
counted nowhere, so a zero here is not evidence that the postings agree with the
data.

### The reconcile pass

Each store runs one **reconcile** ticker. A tick takes a **random sample of at
most 10 000 posted keys of one definition** — definitions are taken in
rotation, and one still building is skipped — re-reads each of those keys
through the cache, and drops only the postings whose key no longer reads back
live. Nothing else about a posting is inspected: an abandoned slot can decode
cleanly as some other key, so the re-read is the only sound liveness test.

**Coverage is probabilistic, and that is the design.** The sample is a
truncated range over the reverse map, and Go randomises where a range starts,
so successive ticks look at different keys. Every key is reached in
expectation — roughly `n·ln(n) / 10 000` ticks to have touched all *n* of them
at least once — with no guarantee for any particular key on any particular
tick. An ordered cursor would give exact coverage at the price of an
O(live keys) scan under the index lock every tick, which also starves under
monotonically increasing keys.

That trade is only available because **correctness never depends on this
pass**. Verify-on-read already makes a dangling posting harmless: it costs one
wasted lookup and can never produce a wrong row. What the pass bounds is
memory, against the residue the cache's removal hook cannot reach — a corrupt
slot with no key to name, a torn page's abandoned slots, or a rebuild whose
walk straddled a flush.

**Two knobs, and their zero values mean opposite things.** Both are named
`KVIndexReconcileIntervalMs` and both are in milliseconds:

| Where | `0` means | Negative means |
|---|---|---|
| `shard.Config` (the replicated/embedded store) | **disables** the pass — the 60 000 default is filled in by `shard.DefaultConfig` | rejected as a configuration error |
| `rostam.DirectConfig` (the single-node store) | keeps the **default**, 60 s — matching its own `Cache.TTLSweepIntervalMs` | **disables** the pass |

Each side follows its own struct's established convention, so porting a config
across by copying the field silently flips the pass on or off. Set it
deliberately on each side: to disable it under `DirectConfig`, write `-1`.
Disabling it is safe and never changes an answer.

### What each thing costs

**Memory.** One posting plus one reverse entry per indexed key per definition,
plus one set per distinct value. `BenchmarkIndexMemory` in `ops/kvindex`
measures it, so the figures below can be re-derived rather than taken on
trust:

```
go test ./ops/kvindex/ -run xxx -bench IndexMemory -benchtime 1x

BenchmarkIndexMemory/100k_keys_1k_distinct-20     1  ...  147.4 B/key
BenchmarkIndexMemory/100k_keys_100k_distinct-20   1  ...  450.8 B/key
```

With a 14-byte key that is about **147 bytes per indexed key** at negligible
cardinality — 100 000 keys for roughly 15 MB — and the difference between the
two rows, re-spread over the extra distinct values, puts one more posting set
at about **306 bytes**. That second figure is a ceiling rather than a rate: at
one key per value every set is a map holding a single entry, the worst ratio
there is. Both numbers move with the key length, since the key bytes are
stored once per posting. High cardinality is paid for twice — in memory, and
in every range query.

**Lookups per query, per shard group.** One map probe per `eq` value; for a range,
one examination per **distinct value** in the index, plus the union of the
matching sets. Then **one cache read per candidate key**, whatever the selector
was. The candidate budget (default 250 000) charges both the keys copied and the
distinct values examined.

**A range refuses on cardinality, not on result size.** Because a range walks the
distinct values, an index holding more distinct values than the candidate budget
is refused with `kvindex: candidate budget exceeded` *even if the range would
have matched nothing*. The walk is the cost. The remedy is a different index, or
an `eq`/`in` leaf.

**The scan cost cliff.** The scan chunk's heap is bounded by **bytes** (32 MiB)
as well as by count, and the byte cap binds first when keys are large: at the
64 KiB maximum key size a page retains about 512 keys instead of `ScanChunk`'s
10 000, and each retained key still costs a full walk of the shard. Large keys
therefore make a scan roughly twenty times more expensive to page than the chunk
size suggests.

**Listing indexes costs a fan-out.** `GET /v1/kv/indexes` (`__kv_index_list__`)
gathers one readiness leg per shard group per call. It is read-scoped, bounded
per call (at most 64 definitions, one leg per group, each under a 10 s timeout)
and **unmetered** — so a client polling it in a tight loop is a cluster-wide
fan-out per poll. Poll it at human intervals, not per request.

**A backfill briefly contends with writers.** The walk releases each shard's read
lock every 4 096 index slots rather than holding it for the whole pass, so a
writer waits at most one chunk — never for a whole shard's walk. Creating a
definition on a large live keyspace is a visible but bounded latency bump on that
node, once, and again on every restart (the index is rebuilt from the cache every
time).

### Node budgets

Budgets are **node configuration, not call arguments**: `kv_query` is read-only
and never applied, so a per-node value cannot diverge committed state, and a
caller cannot raise them.

| Budget | Default | Bounds |
|---|---|---|
| `Candidates` | 250 000 | posting keys a selector may union, plus distinct values a range examines |
| `Scan` | 5 000 000 | keys one scan page may visit, matching or not |
| `ScanChunk` | 10 000 | keys one scan page carries forward |

A candidate or scan budget that cannot be paid is a **typed error, never a
truncated page** — a caller cannot tell a truncated answer from a complete one.
The page *byte* budget is the exception, and it truncates safely because it
truncates with a continuation.

## Non-goals

These are deliberate omissions, not gaps waiting to be filled:

- **No table-column indexes.** A definition names one top-level field or one
  table's row count. Column and row paths are filterable, never indexable.
- **No cross-index planning.** A query names exactly one index and one leaf
  drives it. There is no intersection of two indexes and no cost-based choice
  between them.
- **No ordering other than ascending key bytes.** There is no `order by` on an
  indexed field, and no descending mode.
- **No aggregation.** `kv_query` returns rows, not counts or groupings.
- **No MCP surface.** `kv_query` mirrors `flush`: the native client and one HTTP
  endpoint, and it stops there.
