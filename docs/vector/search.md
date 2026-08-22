# Search

All search entry points live on the collection (library) and under
`/v1/collections/{name}/points/...` (HTTP). Results are `(id, distance)` pairs;
fusion-based searches also carry a `score`.

**Not every mode is reachable from every entry point.** Check here before
designing around one:

| Mode | Go library | HTTP | gRPC | Binary TCP | Python |
|---|---|---|---|---|---|
| [kNN](#k-nearest-neighbors) | `Search` | `points/search` | `Search` | `vector_search` | `search()` |
| kNN + content | `SearchDocs` | `points/search/docs` | `SearchDocs` | `vector_search_docs` | `search_docs()` |
| [Grouping](#grouping-top-k-per-group) | `SearchGroups` | `points/search/groups` | `SearchGroups` | `vector_search_groups` | `search_groups()` |
| [Scroll](#scroll-filtered-listing-with-pagination) | `ScrollDocs` | `points/scroll` | `Scroll` | `vector_scroll` | `scroll()` |
| [Recommend](#recommendation-positivenegative-examples) | `Recommend` | via Query API | via `VectorQuery` | via `vector_query` | `recommend()` |
| [Discover](#discovery-context-pairs) | `Discover` | via Query API | via `VectorQuery` | via `vector_query` | — |
| [MMR](#mmr-diversified-retrieval) | `SearchMMR` | — | — | — | client-side † |

Two different situations hide behind the gaps, and they are worth telling apart.

**Recommend and Discover are not missing remotely** — they are leaves of the
[unified Query API](#unified-query-api-multi-stage-fusion-rerank) rather than
routes of their own. One composable endpoint carries them on all three
transports, which is why there is no `points/recommend`.

**MMR really is Go-only.** It appears in no transport layer at all, because it
needs the candidate *vectors* to score pairwise diversity and the search wire
returns ids and distances, not vectors. Nothing makes that impossible to add —
it would need an op that returns vectors, or the diversity computed server-side
— so treat it as a gap rather than a rule.

† Over the wire, MMR is done by fetching candidates with their vectors and
re-ranking locally. The Python LangChain integration already implements exactly
that (`max_marginal_relevance_search`), so reach for it before writing your own.

## k-nearest neighbors

=== "Go"

    ```go
    hits, err := col.Search(query, 10)                       // plain kNN
    hits, err = col.SearchFiltered(query, 10, filter)        // kNN + metadata filter
    hits, err = col.SearchInto(dst, query, 10, filter)       // allocation-light: reuses dst
    docs, err := col.SearchDocs(query, 10, filter)           // hits + stored content + metadata
    ```

=== "Python"

    ```python
    from rostam import Rostam, filters as f

    c = Rostam("http://localhost:8080")
    query = [0.1, 0.2, 0.3, 0.4]   # your embedding model's output

    hits = c.search("docs", query, k=10)                       # ids, distances, scores
    hits = c.search("docs", query, k=10, filter=f.eq("tenant", "acme"))
    docs = c.search_docs("docs", query, k=10)                  # + content and metadata
    ```

=== "curl"

    ```sh
    curl -s localhost:8080/v1/collections/docs/points/search \
      -d '{"query":[0.1,0.2,0.3,0.4],"k":10}'

    curl -s localhost:8080/v1/collections/docs/points/search/docs \
      -d '{"query":[0.1,0.2,0.3,0.4],"k":10}'
    ```

`SearchDocs` returns `Document{ID, Distance, Score, Content, Metadata}` — the
RAG-friendly shape. Filtering semantics and the filter-first planner are covered
in [Filtering](filtering.md).

Recall is tuned with the collection's `EfSearch`
([Collections & indexes](collections-and-indexes.md#core-parameters)). Note the
effective beam width is `max(EfSearch, k)`: an `EfSearch` below `k` has no
effect, so exploring low-ef behaviour means lowering `k` too.

## MMR — diversified retrieval

Maximal Marginal Relevance re-ranks a candidate pool to balance relevance
against diversity — useful when the top-k would otherwise be near-duplicates.

!!! info "Go library only"

    MMR has no HTTP route and no Python method. Over the wire, fetch a wider
    `k` and re-rank client-side.

```go
hits, err := col.SearchMMR(query, 10, vector.MMROpts{
	Lambda: 0.5, // 1.0 = pure relevance … 0.0 = pure diversity (default 0.5)
	FetchK: 0,   // candidate pool; default 4·k (min 50)
	Filter: filter,
})
```

## Recommendation — positive/negative examples

Search by example ids instead of a raw query vector:

```go
hits, err := col.Recommend(10, vector.RecommendOpts{
	Positive: []uint64{12, 96},  // required
	Negative: []uint64{40},      // optional
	Filter:   filter,
})
```

`RecommendVecs` is the same with caller-resolved vectors. No positive examples →
`ErrNoRecommendExamples`.

## Discovery — context pairs

Guide the search with (positive, negative) context pairs and an optional target
anchor — useful for "more like this, but away from that" exploration:

```go
hits, err := col.Discover(10, vector.DiscoverOpts{
	Target:  queryVec,                       // optional anchor
	Context: []vector.ContextPair{{Positive: p, Negative: n}}, // required
	Filter:  filter,
})
```

## Grouping — top-k per group

Collapse hits by a payload field (e.g. one best chunk per source document):

=== "Go"

    ```go
    groups, err := col.SearchGroups(query, 5, vector.GroupOpts{
    	GroupBy:   "doc_id", // required
    	GroupSize: 2,        // hits kept per group (default 1)
    })
    // Group{Key, Hits []Document}
    ```

=== "Python"

    ```python
    query = [0.1, 0.2, 0.3, 0.4]

    groups = c.search_groups("docs", query, k=5,
                             group_by="doc_id", group_size=2)
    for g in groups:
        print(g.key, [d.id for d in g.hits])
    ```

## Scroll — filtered listing with pagination

Deterministic id-ascending listing of live points, with cursor pagination:

=== "Go"

    ```go
    docs, err := col.ScrollDocs(filter, 100)                                  // first page
    docs, next, more, err := col.ScrollDocsPage(filter, afterID, true, 100)   // continue after id
    ```

=== "Python"

    ```python
    page = c.scroll("docs", limit=100)                     # first page
    while page.next_cursor:                                # pass the cursor back verbatim
        page = c.scroll("docs", limit=100, cursor=page.next_cursor)
    ```

`ScrollDocsPageOrder` adds order-by on a payload key (numeric, datetime, or
string; multi-key via `Tail`), ascending or descending, with resumable cursors.
Over HTTP: `POST .../points/scroll` with `{filter, limit, cursor}` — pass back
the returned `next_cursor` verbatim.

## Unified Query API — multi-stage fusion & rerank

The Query API (HTTP `POST /v1/collections/{name}/query`, Go `VectorQuery` on the
store facade) composes multiple search lanes into one request: run `prefetch`
lanes (dense, sparse, full-text — each with its own k and filter), then either
**fuse** them (RRF / weighted / DBSF) or **rerank** the union with the root
query. Grouped variants return top-k per group. This is the wire-level
counterpart of hybrid search generalized to arbitrary lane trees — including
nested fusion nodes.

## Cross-shard reads

On partitioned collections, searches fan out across partitions and merge.
`FanMeta` (the `degraded`/`missing` fields over HTTP) reports partitions that
could not be reached; `on_partition_unavailable` chooses between partial results
(default) and failing the request. Read consistency levels are described in
[Deployment modes](../concepts/deployment-modes.md#read-consistency).
