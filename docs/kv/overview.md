# Key-value store

The KV engine is a sharded, in-memory store with a lazy slab-pool allocator,
TTL, optional mmap persistence, and optional per-shard Raft replication. You use
it through the `rostam.Store` facade (any backend) or, for a standalone
in-process cache, through `cache.Cache` directly.

Unlike the vector API, KV is **not on the REST endpoint** — it lives only on the
binary TCP protocol, because it is built for sub-microsecond operations an HTTP
round trip would defeat. In Go that is the `Store` facade below; from Python it
is `r.kv` on a `tcp://`-connected `Rostam` client, which speaks the same
protocol over a socket.

## Core operations

=== "Go"

    ```go
    payload := []byte(`{"coins":100}`)
    _ = store.Put(ctx, []byte("user:42"), payload, 5*time.Minute) // ttl 0 = no expiry
    v, err := store.Get(ctx, []byte("user:42"))                   // rostam.ErrNotFound on miss/expiry
    existed, err := store.Del(ctx, []byte("user:42"))
    ```

=== "Python"

    ```python
    from rostam import Rostam

    r = Rostam("tcp://127.0.0.1:7000")           # the server's -tcp port
    r.kv.put("user:42", b'{"coins":100}', ttl_ms=300_000)   # ttl_ms 0 = no expiry
    r.kv.get("user:42")                            # bytes, or None on miss/expiry
    r.kv.delete("user:42")                         # -> bool (existed)
    ```

    Keys and values may be `str` (encoded UTF-8) or `bytes`; reads always return
    `bytes` (or `None`). Pass `auth_token=` when the server requires one — it
    rides the protocol-v2 frame on every request.

    The same client speaks the vector database over the same connection — the
    flat API directly on `r` (`r.create_collection / upsert / search / get /
    delete`) — see [the vector docs](../vector/collections-and-indexes.md).
    `r.kv.<operation>` raises `TransportError` on an HTTP-connected client (`Rostam("http://...")`);
    KV has no REST surface.

Beyond get/put/del, two built-in atomic ops run server-side — no
read-modify-write race, no extra round trips:

=== "Go"

    ```go
    // counter += 1, returns the new value as big-endian int64
    res, err := store.Call(ctx, "incr", ops.EncodeIncrArgs([]byte("views:42"), 1))
    n, _ := ops.DecodeIncrResult(res)

    // refresh a TTL without rewriting the value
    _, err = store.Call(ctx, "expire", ops.EncodeExpireArgs([]byte("user:42"), time.Hour))
    ```

=== "Python"

    ```python
    r.kv.incr("views:42", 1)          # atomic add, returns the new int (missing = 0)
    r.kv.expire("user:42", 3_600_000) # refresh the TTL without rewriting the value
    ```

The Python client covers the five built-in ops (`get`, `put`, `delete`, `incr`,
`expire`). [Custom ops](custom-ops.md) and [WASM procedures](wasm.md) are
dispatched through Go's `Call(ctx, name, args)` with op-specific argument
encoders, and are Go-only for now.

`Call(ctx, name, args)` dispatches any registered op by name. Read-only ops
execute locally on the routed shard; read-write ops serialize through the
shard's Raft log on `Embedded` (and under the shard lock on `Direct`). This is
the extension point for [custom ops](custom-ops.md) and
[WASM procedures](wasm.md).

## TTL semantics

TTLs are absolute deadlines computed at write time. Expiry is enforced lazily on
read plus by a background sweeper (`cache.Config.TTLSweepIntervalMs`, default
1000 ms; 0 disables the sweeper, lazy expiry still applies).

## Leadership helpers

On replicated backends, writes must reach the shard leader. The facade exposes
`IsLeader(key)` and `LeaderAddr(key)`; the smart client uses the same topology
data to route automatically. A write landing on a non-leader returns
`rostam.ErrNotLeader`.

## Standalone cache: allocation-free reads

The `rostam.Store` facade wraps a cache internally but does not expose it. When
you need an allocation-free hot loop, use `cache.New` directly as a standalone
session cache:

```go
c, err := cache.New(cache.Config{})
_ = c.Put([]byte("user:42"), []byte(`{"coins":100}`), 5*time.Minute)

buf := make([]byte, 0, 256)
buf, err = c.GetInto(buf[:0], []byte("user:42")) // 0 allocs on a hit
```

What `Get` returns depends on the eviction policy. Under `PolicyRejectWrites` it
is a slice aliasing the backing arena — zero-copy, but don't retain it across
writes. Under the default `PolicyRingbufEvict` it is a freshly allocated copy you
own and may retain freely, because eviction can overwrite live page bytes and an
alias would be unsafe; that costs one allocation per hit. `GetInto` copies into
your reusable buffer on either policy, is always the safe form to retain, and
stays allocation-free. Configuration knobs (shards, page size, eviction policy, mmap
durability) are covered in [Cache tuning](cache.md).

## Backends at a glance

| | `Direct` | `Embedded` | `Client` |
|---|---|---|---|
| Process | in-process | in-process | remote (TCP) |
| Consensus | none | per-shard Raft | server-side |
| Get / Put (measured) | ~29 ns / ~240 ns | ~222 ns / ~12.7 µs (no-sync) | ~1.7 µs / ~1.8 µs (loopback, Direct server) |
| Durability | optional mmap | Raft log + mmap warm-start | server's |

The backends and their configs are documented in
[Deployment modes](../concepts/deployment-modes.md); performance methodology in
[Performance](../performance.md).
