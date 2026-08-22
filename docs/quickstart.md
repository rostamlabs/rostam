# Quickstart

Rostam can run three ways: as a **single-node server** speaking HTTP/gRPC/TCP,
as a **replicated Raft cluster**, or as a **Go library** embedded in your binary
(no server at all). This page gets you from zero to a working search in each
mode — the server and Python paths first, the Go embedding paths after.

## Requirements

**Nothing, to run the server.** It ships as a single static binary with no
runtime dependencies.

- **Go 1.26+** only if you embed the library or build from source.
- The default full-module build requires **cgo** (the WASM stored-procedure
  backend uses `wasmtime-go`). The vector engine and cache packages are pure Go —
  see [Development](development.md#building) for `CGO_ENABLED=0` builds.
- mmap persistence and the AVX2 kernels are Linux/amd64; everything has a
  portable fallback.

## Install the server

=== "Install script"

    ```sh
    curl -fsSL https://rostamlabs.com/install.sh | sh
    ```

    Detects your OS and architecture, downloads the matching release binary and
    puts it in `~/.local/bin`. Set `ROSTAM_INSTALL_DIR` to place it elsewhere.

=== "Docker"

    ```sh
    docker run -p 8080:8080 -e ROSTAM_API_KEY=secret ghcr.io/rostamlabs/rostam
    ```

    Multi-arch (amd64/arm64). The image binds `0.0.0.0`, so it **requires** a
    token. An environment variable keeps it off the command line and out of
    `ps`, but it is *not* hidden: `docker inspect` shows it under `Config.Env`.
    For anything beyond local use, mount a secret or use your orchestrator's
    secret store.

    The default command serves **both** REST (`8080`) and the binary TCP
    protocol (`7000`) — publish only the ports you use. The command above maps
    REST; add TCP for the Go client, which speaks only the binary protocol:

    ```sh
    docker run -p 8080:8080 -p 7000:7000 -e ROSTAM_API_KEY=secret \
      ghcr.io/rostamlabs/rostam
    ```

    An unpublished listener is not reachable from the host (though other
    containers on the same network still can reach it, and host networking
    publishes everything). Either way the token guards every transport
    identically — so exposing `7000` is no less safe than `8080`. For gRPC, add
    `-grpc 0.0.0.0:9090` and `-p 9090:9090`.

=== "Go"

    ```sh
    go install github.com/rostamlabs/rostam/cmd/rostam-server@latest
    ```

=== "From source"

    ```sh
    git clone https://github.com/rostamlabs/rostam
    cd rostam
    go build -o rostam-server ./cmd/rostam-server
    ```

## Run it

`rostam-server` exposes the same engine over three transports: REST (`-http`),
gRPC (`-grpc`), and a compact binary TCP protocol (`-tcp`). Start it with REST
and TCP on loopback, persisting to `./data`:

```sh
rostam-server -http 127.0.0.1:8080 -tcp 127.0.0.1:7000 -data ./data
```

!!! note "Non-loopback binds require authentication"

    The server refuses to start with no authentication on a network-reachable
    address (e.g. `-http :8080`). Bind loopback for local development, set
    `ROSTAM_API_KEY`, or pass `-insecure` to run open deliberately — see
    [Security](server/security.md).

`GET /v1/health` and `/v1/ready` stay auth-exempt so orchestrator probes work
without the token.

## Your first search

Create a collection, add a point, and search it. These target the **loopback**
server from [Run it](#run-it), which needs no token.

Against the container, authenticate every call: add
`-H 'Authorization: Bearer secret'` to each curl, construct the Python client as
`Rostam("http://localhost:8080", api_key="secret")`, and set
`ClientConfig.AuthToken` in Go.

=== "curl"

    ```sh
    # Create a 4-dimensional cosine collection
    curl -s localhost:8080/v1/collections \
      -d '{"name":"docs","config":{"dim":4,"metric":"cosine"}}'

    # Upsert a point. Metadata values use a tagged encoding — see Filtering.
    curl -s localhost:8080/v1/collections/docs/points \
      -d '{"id":1,"vector":[0.1,0.2,0.3,0.4],"content":"hello rostam",
           "metadata":{"tenant":{"kind":"string","str":"acme"}},"upsert":true}'

    # Search
    curl -s localhost:8080/v1/collections/docs/points/search \
      -d '{"query":[0.1,0.2,0.3,0.4],"k":3}'
    ```

=== "Python"

    ```sh
    pip install rostam-client
    ```

    ```python
    from rostam import Rostam, filters as f

    c = Rostam("http://localhost:8080")

    c.create_collection("docs", dim=4, metric="cosine")
    c.upsert("docs", 1, [0.1, 0.2, 0.3, 0.4],
             content="hello rostam",
             metadata={"tenant": "acme"})

    # search() returns ids, distances and scores.
    hits = c.search("docs", [0.1, 0.2, 0.3, 0.4], k=3)

    # search_docs() returns the stored content too, and takes a filter.
    docs = c.search_docs("docs", [0.1, 0.2, 0.3, 0.4], k=3,
                         filter=f.eq("tenant", "acme"))
    print([(d.id, d.content) for d in docs])
    ```

    The client sends plain Python values — it applies the tagged metadata
    encoding for you. Full reference: [Python client](api/python.md).

=== "Go"

    ```sh
    go mod init example.com/myapp    # modern Go needs a module first
    go get github.com/rostamlabs/rostam
    ```

    ```go
    package main

    import (
    	"context"
    	"fmt"
    	"log"

    	"github.com/rostamlabs/rostam"
    	"github.com/rostamlabs/rostam/vector"
    )

    func main() {
    	ctx := context.Background()

    	// The -tcp port, not -http: the Go remote client speaks the binary
    	// protocol only. This targets the loopback server from "Run it" (no
    	// token). Against the container, publish the port (-p 7000:7000) and set
    	// AuthToken to the same value as ROSTAM_API_KEY.
    	store, err := rostam.NewClient(rostam.ClientConfig{
    		Servers: []string{"127.0.0.1:7000"},
    		// AuthToken: "secret",   // required when the server has -api-key set
    	})
    	if err != nil {
    		log.Fatal(err)
    	}
    	defer store.Close()

    	// M / EfConstruction / EfSearch default to 16 / 200 / 64 when omitted,
    	// as they do over REST and Python. Set here to keep the tuning visible.
    	if err := store.CreateCollection(ctx, "docs", rostam.VectorConfig{
    		Dim: 4, Metric: vector.Cosine,
    		M: 16, EfConstruction: 200, EfSearch: 64,
    	}); err != nil {
    		log.Fatal(err)
    	}

    	if err := store.VectorUpsert(ctx, "docs", 1,
    		[]float32{0.1, 0.2, 0.3, 0.4}, "hello rostam",
    		rostam.VectorInsertOpts{}); err != nil {
    		log.Fatal(err)
    	}

    	hits, err := store.VectorSearch(ctx, "docs", []float32{0.1, 0.2, 0.3, 0.4}, 3)
    	if err != nil {
    		log.Fatal(err)
    	}
    	fmt.Println(hits)
    }
    ```

    Full reference: [Go client](api/go-client.md).

The full endpoint inventory is in the [HTTP API reference](api/http.md). To add
authentication and TLS, see [Security](server/security.md); for multi-node
clusters, see [Clustering](server/clustering.md).

## Large initial loads

For a first bulk import, `bulk_stage(...)` + `bulk_build(...)` ship vectors over
a binary wire and build the index on all cores — far faster than upserting point
by point:

```python
c.bulk_stage("docs", ids, vectors)
c.bulk_build("docs")          # concurrent build, all cores
```

Full method reference: [Python client](api/python.md).

## Embedded vector search (Go library)

```sh
go get github.com/rostamlabs/rostam
```

The vector engine is a standalone package — no server, no cgo, no other Rostam
dependencies:

```go
package main

import (
	"fmt"

	"github.com/rostamlabs/rostam/vector"
)

func main() {
	col, err := vector.NewCollection("docs", vector.Config{
		Dim:    768,
		Metric: vector.Cosine,
		Quant:  vector.QuantSQ8, // int8 codes: 4× smaller, ~98% recall retained

		// Optional — these default to 16 / 200 / 64. Shown so the recall/latency
		// dials are visible in the example you copy from.
		M:              16,
		EfConstruction: 200,
		EfSearch:       64,
	})
	if err != nil {
		panic(err)
	}
	defer col.Close()

	embedding := make([]float32, 768) // your embedding model's output
	query := make([]float32, 768)

	// Insert is create-only (ErrDuplicateID on a live id); use Upsert to replace.
	_ = col.Insert(1, embedding, 0, vector.Metadata{
		"tenant": vector.NewString("acme"),
	}, nil)

	// Exact, fast filtered search — the payload index narrows to tenant=acme.
	hits, _ := col.SearchFiltered(query, 10, vector.Filter{
		Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme"),
	})
	fmt.Println(hits)

	// Diversified retrieval for RAG:
	diverse, _ := col.SearchMMR(query, 10, vector.MMROpts{Lambda: 0.5})
	_ = diverse
}
```

Where to go next: [Search APIs](vector/search.md), [Filtering](vector/filtering.md),
[Quantization](vector/quantization.md).

## Embedded key-value store (Go library)

The `rostam.Store` facade gives you KV *and* vector operations behind one
interface. `NewDirect` is the single-node, no-Raft backend — the fastest path:

```go
package main

import (
	"context"
	"log"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

func main() {
	ctx := context.Background()

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil { // get/put/del/incr/expire + vector ops
		log.Fatal(err)
	}
	store, err := rostam.NewDirect(rostam.DirectConfig{
		Ops: reg, // required
		// DataDir: "./data", // enable mmap persistence + warm restart
	})
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	_ = store.Put(ctx, []byte("user:42"), []byte(`{"coins":100}`), 5*time.Minute)
	v, _ := store.Get(ctx, []byte("user:42"))
	_ = v

	// Atomic server-side read-modify-write, serialized per shard:
	_, _ = store.Call(ctx, "incr", ops.EncodeIncrArgs([]byte("views:42"), 1))
	_, _ = store.Call(ctx, "expire", ops.EncodeExpireArgs([]byte("user:42"), time.Hour))
}
```

Swap the constructor to change the backend — the `Store` interface is identical:

| Constructor | Backend | When |
|---|---|---|
| `rostam.NewDirect` | in-process, no Raft | single node, library use, fastest |
| `rostam.NewEmbedded` | in-process + per-shard Raft | replicated / multi-node durability |
| `rostam.NewClient` | TCP client to a remote cluster | talk to a running server |

Details: [KV overview](kv/overview.md), [Deployment modes](concepts/deployment-modes.md).

## Worked examples

- [`examples/semantic-search`](https://github.com/rostamlabs/rostam/tree/main/examples/semantic-search)
  — end-to-end RAG-style pipeline: OpenAI embeddings → upsert → dense vs hybrid
  search over the TCP client.
- [`examples/filtered-recall-cliff`](https://github.com/rostamlabs/rostam/tree/main/examples/filtered-recall-cliff)
  — a runnable demonstration of why the filter-first query planner exists
  (post-filtering recall collapse vs exact filter-first).
