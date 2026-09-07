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

## Atomic multi-field updates: `operate`

The built-in **`operate`** op applies a *list* of pure-integer ops to one record
atomically, server-side, in one round-trip under the shard lock — so a **hot key
updated by many callers in parallel loses no update** (what a client CAS-retry
loop can't guarantee). The op-list is data, so the update logic lives in your app
and adding a counter never rebuilds Rostam. (For arbitrary Go logic, a
[custom op](custom-ops.md) is the escape hatch; `operate` is the ready-made
generic form.)

One record decodes as a small **globals** array plus a capped map of **entries**
(`u64` → sub-record, each with a leader-stamped `touchMs` for deterministic LRU
eviction). Fields are **typed** for memory efficiency — each field is stored
self-describing as `[type u8][data]`, so a `u8` counter costs 2 bytes instead of 8
(a typed record is ~50% smaller than an all-i64 one). The app assigns each field
index a type; the type rides the op so a field is created on first touch, and the
**stored type wins** for a field that already exists.

Field types (`wire.OperateType*`): `U8 U16 U32 U64 I8 I16 I32 I64 F32 F64 UVARINT
IVARINT`. Fixed-width data is native little-endian; `UVARINT`/`IVARINT` are LEB128
(IVARINT zigzag), a compact growable counter. Opcodes:

| opcode | effect |
|---|---|
| `INCR(f, n)` / `INCRF(f, n)` | `f += n`; fixed-width ints **saturate** at the type's min/max (never wrap), floats use IEEE add (operand `n` rides `Arg` as the IEEE bit pattern) |
| `SETMAX(f, v)` | `f = max(f, v)` in the field's domain |
| `SHIFTOR(f, b, v)` | `f = ((f << b) \| v)` **masked to the field's bit width** (rolling window; fixed-width int only; `b` ≥ 0) |
| `HALVE_GRP(f, thr, len)` | if `f ≥ thr`, halve the contiguous fields `[f, f+len)` per each field's type (overflow guard) |

A target is a **global** field or a **(entryKey, field)** inside the map;
referencing an absent entry upserts it (subject to `maxEntries` — `0` = unbounded —
evicting the smallest `touchMs`, ties by smallest key). Return specs read fields
back after the ops apply (the returned i64 is the field's raw bits — interpret per
its type), so one call can update *and* read the record.

```go
import "github.com/rostamlabs/rostam/sdk/wire"

ops := []wire.OperateOp{
    {Target: wire.OperateTargetGlobal, FieldIdx: 0, Type: wire.OperateTypeU8, Opcode: wire.OperateOpHALVEGRP, Arg: 255, Arg2: 2}, // guard
    {Target: wire.OperateTargetGlobal, FieldIdx: 0, Type: wire.OperateTypeU8, Opcode: wire.OperateOpINCR, Arg: 1},                 // request count (saturates at 255)
    {Target: wire.OperateTargetEntry, EntryKey: bidderID, FieldIdx: 0, Type: wire.OperateTypeU16, Opcode: wire.OperateOpINCR, Arg: 1},
}
ret := []wire.OperateRet{{Target: wire.OperateTargetGlobal, FieldIdx: 0}}
vals, err := c.Operate(ctx, []byte("session:42"), 90*time.Second, 1024, ops, ret) // vals[i] ← ret[i]
```

Wire (args): `[keyLen u16][key][ttlMs u64][maxEntries u16][nOps u32]` then per op
`[tgt u8][entryKey u64 iff tgt=1][fieldIdx u16][opcode u8][ftype u8][arg i64][arg2 i64]`,
then `[nReturn u16]` and per return `[tgt u8][entryKey u64 iff tgt=1][fieldIdx u16]`;
the result is the requested field values as i64 big-endian in request order. Every
mutation is pure integer/float arithmetic stamped only by the leader clock (and
IEEE float encodings are deterministic), so a replicated apply is byte-identical
across the group. The whole op-list is all-or-nothing: an invalid op (unknown
opcode/type, negative shift, SHIFTOR on a non-fixed-int field) aborts it with the
record unchanged.

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
