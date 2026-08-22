# Go client (TCP smart client)

Two ways to reach Rostam from Go, both over the compact binary TCP protocol to
a running server or cluster (`rostam-server -tcp ...`):

- **Typed `Collection` client** (`client.NewRouted` + `c.Collection(name)`) —
  the recommended API for vector work: request/response structs, typed errors,
  shard-aware routing on by default.
- **Low-level client** (`rostam.NewClient`) — the same `rostam.Store`
  interface as the embedded backends, plus `Call` as an escape hatch for
  custom ops and WASM procedures.

## Typed Collection client

`client.NewRouted` builds a `*client.Client` with Rostam's builtin routing
registry wired in, so each request is dispatched to the shard that owns its
key instead of round-robin:

```go
import (
	"os"

	"github.com/rostamlabs/rostam/client"
)

c, err := client.NewRouted(client.Config{
	Servers:   []string{"127.0.0.1:7000"}, // binary TCP port, bootstrap list
	AuthToken: os.Getenv("ROSTAM_TOKEN"),
})
if err != nil { ... }
defer c.Close()

posts := c.Collection("posts")
```

`client.New(cfg)` builds the same client without the routing registry (nil
`Ops` — no self-routing). It exists for internal/verbatim-dialing callers that
need to address a specific, already-resolved node (e.g. a leader-pinned read);
application code should use `NewRouted`.

### Collection handle

`c.Collection(name)` returns a `*client.Collection` bound to that name. It
performs no I/O and does not require the collection to exist yet.

```go
type CreateRequest struct {
	Dim            int
	Metric         vector.Metric // zero value = vector.Cosine (also L2, DotProduct)
	M              int
	EfConstruction int
	EfSearch       int
	Persistent     bool
	FullText       *vector.FullTextConfig // set to enable BM25 (required for HybridText)
}

err := posts.Create(ctx, client.CreateRequest{Dim: 768, Metric: vector.Cosine})
err  = posts.Drop(ctx) // idempotent — a never-created name is a no-op, not an error
```

### Writing

```go
type WriteRequest struct {
	ID       uint64
	Vector   []float32
	Content  string              // raw text tokenized into the BM25 lane (Upsert only)
	Metadata vector.Metadata     // build with vector.NewString/NewInt/NewFloat/NewBool/...
	Sparse   vector.SparseVector // optional client sparse vector
	TTL      time.Duration       // 0 = no expiry
	// CAS: HasExpectedVersion must be true for ExpectedVersion to apply.
	ExpectedVersion    uint64
	HasExpectedVersion bool
	KeyTTLMs           map[string]int64
}

err := posts.Upsert(ctx, client.WriteRequest{ID: 1, Vector: vec, Content: "..."})
err  = posts.Insert(ctx, client.WriteRequest{ID: 2, Vector: vec}) // fails if id exists
err  = posts.Delete(ctx, client.DeleteRequest{ID: 2})
```

`Insert` rejects a non-empty `Content` (the insert wire op carries no content
field) — use `Upsert` when you need the BM25 content lane.

`UpsertBatch` issues one `Upsert` per point over the (optionally pipelined)
connection — there is no single batch op on the wire — and returns the
per-point failures, if any:

```go
errs := posts.UpsertBatch(ctx, []client.PointInput{
	{ID: 1, Vector: v1, Content: "first post"},
	{ID: 2, Vector: v2, Content: "second post"},
})
for _, e := range errs { // nil/empty slice means everything succeeded
	log.Printf("point %d: %v", e.ID, e.Err)
}
```

### Reading

```go
pt, err := posts.Get(ctx, client.GetRequest{ID: 1, WithVector: true, WithPayload: true})
// pt: Point{ID, Vector, Metadata, TTLMs, Version}
// err is client.ErrNotFound if the point is absent, client.ErrCollectionNotFound
// if the collection itself doesn't exist.

batch, err := posts.GetBatch(ctx, client.GetBatchRequest{IDs: []uint64{1, 2, 3}})
// batch.Points + batch.Missing (requested ids that weren't found — not an error)

page, err := posts.Scroll(ctx, client.ScrollRequest{Limit: 100})
for page.NextCursor != "" {
	page, err = posts.Scroll(ctx, client.ScrollRequest{Limit: 100, Cursor: page.NextCursor})
}
```

### Searching

Every search method returns a `SearchResponse` (or a doc/group variant of it)
carrying explicit degraded-partition info:

```go
type SearchResponse struct {
	Results  []vector.Result // {ID, Distance, Score}
	Degraded bool            // true if a partition was unavailable when this answered
	Missing  []uint16        // the unavailable partition indices
}
```

```go
res, err := posts.Search(ctx, client.SearchRequest{Query: qvec, K: 10, Filter: f})

docs, err := posts.SearchDocs(ctx, client.SearchDocsRequest{Query: qvec, K: 10})
// docs.Documents: []client.Document{ID, Distance, Score, Content, Metadata}

groups, err := posts.SearchGroups(ctx, client.GroupSearchRequest{
	Query: qvec, K: 5, GroupBy: "doc_id", GroupSize: 2,
})
// groups.Groups: []client.Group{Key, Hits []Document}

res, err = posts.HybridSearch(ctx, client.HybridSearchRequest{
	Dense: qvec, Sparse: sparseVec, K: 10, Method: vector.FusionRRF,
})

res, err = posts.HybridText(ctx, client.HybridTextRequest{
	Dense: qvec, Text: "how do i rotate api keys", K: 10, Method: vector.FusionRRF,
})
// HybridText requires the collection to have been Created with FullText set.
```

`Search`/`SearchDocs`/`SearchGroups`/`HybridSearch`/`HybridText` all accept a
`Consistency` field (`client.ConsistencyDefault` is the strong/leader-read
server default) and return `client.ErrCollectionNotFound` when the collection
doesn't exist.

### Recommend / raw Query

```go
res, err := posts.Recommend(ctx, client.RecommendRequest{
	Positive: []uint64{1, 2},  // point ids the caller liked
	Negative: []uint64{9},     // optional: steer away from these
	K:        10,
})
```

`Recommend` resolves the example ids to their stored vectors server-side and
excludes them from the results. It rides `Query` under the hood, sending a
`ModeFusion` spec with a single `RECOMMEND` prefetch lane; that shape only
fuses correctly behind a routed/multi-node (coordinator) deployment — against
a single coordinator-less node the fused result cannot be decoded. `Query` is
the power-user escape hatch for any `vector.QuerySpec` (multi-leaf fusion,
rerank, discover), decoded into the same `SearchResponse` shape.

### Typed errors

| Error | Meaning |
|---|---|
| `client.ErrNotFound` | point/key absent |
| `client.ErrVersionConflict` | CAS `ExpectedVersion` mismatch |
| `client.ErrCollectionExists` | `Create` on a name that already exists |
| `client.ErrCollectionNotFound` | the collection itself doesn't exist |

```go
if errors.Is(err, client.ErrCollectionExists) { ... }
```

### Worked example: recommend the next post

```go
posts := c.Collection("posts")
_ = posts.Create(ctx, client.CreateRequest{
	Dim: 768, Metric: vector.Cosine,
	FullText: &vector.FullTextConfig{}, // defaults: english analyzer, k1=1.2, b=0.75
})

for _, p := range []struct {
	id      uint64
	vec     []float32
	content string
} {
	{1, embed("rotating api keys safely"), "rotating api keys safely"},
	{2, embed("kubernetes autoscaling basics"), "kubernetes autoscaling basics"},
	{3, embed("zero-downtime key rotation at scale"), "zero-downtime key rotation at scale"},
} {
	_ = posts.Upsert(ctx, client.WriteRequest{
		ID: p.id, Vector: p.vec, Content: p.content,
		Metadata: vector.Metadata{"tenant": vector.NewString("acme")},
	})
}

// Fuse a dense query with BM25 full-text.
hits, err := posts.HybridText(ctx, client.HybridTextRequest{
	Dense: embed("how do i rotate api keys"),
	Text:  "how do i rotate api keys",
	K:     5, Method: vector.FusionRRF,
})

// Or: the reader just finished post 1 — recommend what's similar.
next, err := posts.Recommend(ctx, client.RecommendRequest{Positive: []uint64{1}, K: 5})
```

## Low-level client

`rostam.NewClient` returns the same `rostam.Store` interface as the embedded
backends, speaking the compact binary TCP protocol to a running server or
cluster (`rostam-server -tcp ...`).

```go
import (
	"os"

	"github.com/rostamlabs/rostam"
)

store, err := rostam.NewClient(rostam.ClientConfig{
	Servers:   []string{"10.0.0.1:7000", "10.0.0.2:7000"}, // bootstrap list
	AuthToken: os.Getenv("ROSTAM_TOKEN"),
})
if err != nil { ... }
defer store.Close()

// Same interface as NewDirect/NewEmbedded:
_ = store.Put(ctx, []byte("k"), []byte("v"), 0)
hits, meta, err := store.VectorSearchExt(ctx, "docs", query, 10, opts)
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `Servers` | — (required) | initial `host:port` bootstrap list; topology is discovered from any live entry |
| `AuthToken` | "" | bearer token sent on every RPC (protocol-v2 frame; 255-byte limit — registry tokens, not JWTs) |
| `TLSConfig` | nil (plaintext) | build with `tlsutil.ClientTLS(caFile, certFile, keyFile, serverName)` for TLS/mTLS |
| `Ops` | nil | registry mirror used to route **custom ops** by key; without it, custom-op calls fall back to round-robin routing |
| `MaxConnsPerServer` | 8 | connection pool per server |
| `MaxNotLeaderHops` | 5 | retry budget when topology is stale |
| `TopologyRefreshInterval` | 5 s | how often cluster membership/leadership is re-polled |

## What "smart" means

- **Topology awareness** — the client polls cluster membership and per-shard
  leadership, so requests go to the right node the first time.
- **Leader routing** — writes route to the owning shard's Raft leader; reads
  can be served by any replica (subject to the per-request read-consistency
  level).
- **Bounded retries** — a write that lands on a stale leader is retried toward
  the new one up to `MaxNotLeaderHops` times before surfacing
  `rostam.ErrNotLeader`.
- **Connection pooling** — up to `MaxConnsPerServer` multiplexed connections
  per node.

## Calling custom ops remotely

Registered [custom ops](../kv/custom-ops.md) and
[WASM procedures](../kv/wasm.md) are invoked by name exactly as in-process:

```go
res, err := store.Call(ctx, "incr", ops.EncodeIncrArgs([]byte("views:42"), 1))
```

Mirror your op registry into `ClientConfig.Ops` so the client can extract the
routing key and send the call directly to the owning shard's leader.

`RegisterWASM` also works over the TCP client — the registration is forwarded
to the leader, and the returned push report names any cluster members that did
not yet receive the module bytes:

```go
pushReport, err := store.RegisterWASM(ctx, reg, moduleBytes)
```

## Errors

| Error | Meaning |
|---|---|
| `rostam.ErrNotFound` | key/point absent or expired |
| `rostam.ErrNotLeader` | could not reach the shard leader within the retry budget (election in progress, stale topology) |
| `vector.ErrDuplicateID` | dense `VectorInsert` on a live id |

The TCP token limit (255 bytes) means JWT auth is HTTP/gRPC-only; use registry
tokens for TCP clients.
