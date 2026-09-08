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

`operate` is a built-in op for updating several fields of one record
atomically, in one round trip, against a hot key — the way you'd reach for a
Redis hash command or an Aerospike `Operate` call, without writing a custom Go
op and without a client-side get-modify-put CAS loop. The op list you send is
data: a new counter, a new guard, or a new table is a different call, not a
server rebuild. It runs under the same shard lock as every other read-write
op (design doc `operate-v2-design.md`, §1).

### The record and table model

A record is a struct of scalar fields; a field may be a **table** of keyed
rows of fixed-width columns. The shape is exactly record → table → row →
column — no arbitrary nesting, and no table inside a row. Every record is
stored in one of two encodings, chosen when the record is created:

| Mode | Trade-off | Pick it when |
|---|---|---|
| **Schema** | Client-defined, versioned layout stored once; values packed with nothing per value; rows are fixed-width and binary-searchable. Smallest memory, fastest apply. | The shape is known and the record is hot or numerous (counters, per-key sub-records, rate limits). |
| **Dynamic** | No schema: every field and column carries its own `[name][type]`; any name/type can be added by any op, at any time. Redis-hash ergonomics, ~40-60% more bytes per value, a scan instead of a binary search. | The shape is still changing, or you want `HINCRBY`/`HSET`-style ergonomics without declaring anything up front. |

Both modes share the same paths, ops, returns, caps, and determinism — one
apply engine over two byte layouts. A dynamic record can later be frozen into
a schema mode record with `MIGRATE` (see below).

A path into a record is one of:

- `()` — the record itself (valid for `CHECK EXISTS`/`ABSENT`, `DEL`, and a whole-record return).
- `(field)` — a record field.
- `(field, rowKey)` — a whole row of the table at `field`.
- `(field, rowKey, col)` — one column of one row.

In schema mode a `field`/`col` is addressed by its **schema position**;
dynamic mode addresses both by **name**. Writing to a path vivifies it (a
missing row is created, evicting first if the table is full; a missing
dynamic field/column is created with the op's declared type); reads never
vivify, and an absent or `UNSET` scalar reads as zero for arithmetic and
comparisons.

### Scalar types

| Tag | Type | Width | Allowed |
|---|---|---|---|
| 0-7 | `U8` `U16` `U32` `U64` `I8` `I16` `I32` `I64` | 1/2/4/8 B, little-endian; arithmetic saturates | record field or table column |
| 8-9 | `F32` `F64` | 4/8 B, IEEE 754, canonical quiet NaN | record field or table column |
| 10-11 | `UVARINT` `IVARINT` | 1-10 B, LEB128 (`IVARINT` zigzagged) | **record fields only**, never a table column |
| 12 | `BYTES` | `[len uvarint][data]`, ≤ 65535 B | **record fields only** |
| 13 | `FIXED(n)` | `n` raw bytes, 1-255 | record field, table column, or table row key |
| 14 | `TABLE` | — | **record fields only** — a table cannot nest inside a row |
| 15 | `UNSET` | 0 bytes | not a storable type — how an absent, declared slot reads back |

A table's row key must be an unsigned fixed-width int (`U8`..`U64`, compared
numerically) or `FIXED(n)` (compared bytewise); signed ints, floats, and every
variable-length type are rejected as a key.

### Schema lifecycle

A schema (`wire.Schema`: a `Version` and an ordered `[]FieldDef`) is defined
**client-side**, chosen by the app, and rides the call:

- **Create.** A call may carry the schema blob (`WithSchema` in the Go
  client). If the record is absent, it's created with that schema. If it
  exists, the call's version must equal the stored version or the call fails
  with `wire.ErrOperateSchemaVersion` before any op runs — only the version is
  compared, not the blob's bytes.
- **Evolve.** A `MIGRATE` op at the head of the list moves a record from its
  stored version to a new one in the same atomic call: the new schema must be
  an **append-only extension** (`Schema.Extends`) — every existing field and
  table column keeps its position, type, and width (a stored name may not
  change, though an unnamed one may gain a name); new fields/columns may only
  be appended; a table's `Cap`/`Policy`/`ByCol` may change freely. Anything
  else is rejected. New fields/columns zero-fill on migrate; a lowered `Cap`
  evicts down to it during the same rewrite.
- **Freeze.** `MIGRATE` with `from = wire.OperateMigrateFromDynamic` converts
  a dynamic record into schema mode in one atomic rewrite: each schema field
  and column is filled from the dynamic value of the same name (the target
  schema must store names — a freeze matches by name), domains must match
  (widths saturate), and a dynamic field the schema doesn't declare is an
  error unless the migration's `dropExtra` flag is set. The reverse (thaw) is
  not supported.
- **Cache.** The server keeps a small cache keyed by schema bytes holding the
  decoded layout (field offsets, row width, column offsets), so a repeat
  schema resolves every path by arithmetic with no re-parsing.

### Ops

Every op is `(opcode, type, aux, path, a, b, bytes)`. `type` is the
create/check type byte — in dynamic mode it's the type a new field/column is
created with; in schema mode it must match the schema's type or be
`wire.OperateTypeFromSchema` (`0xFF`, what the typed Go client sends). `a`
carries an int operand, or the float64 bit pattern (`math.Float64bits`) of a
float operand — `F32` fields included, storage rounds. `bytes` carries a
`BYTES`/`FIXED` operand, a `CONFIG`/`TRIM` column name, or a `MIGRATE` schema
blob.

**Scalar ops** (record field or table column):

| Op | Operand | Effect | Field types |
|---|---|---|---|
| `SET` | `a` or `bytes` | write `v` | all scalars |
| `ADD` | `a` | `x += a`, saturating (int) / IEEE (float) | int, float |
| `MUL` | `a` | `x *= a`, saturating (int) / IEEE (float) | int, float |
| `MIN` / `MAX` | `a` or `bytes` | `x = min/max(x, v)` | int, float, bytes/fixed (bytewise) |
| `AND` / `OR` / `XOR` | `a` | bitwise, masked to the field width | fixed-width int only |
| `SHL` | `a` (0..64) | shift left, masked to the field width | fixed-width int only |
| `SHR` | `a` (0..64) | shift right; arithmetic for signed, logical for unsigned | fixed-width int only |
| `STAMP` | `aux` = unit (`wire.OperateStampMs`/`OperateStampS`) | `x = tx.applyStamp()` in that unit, saturating | int only |
| `DEL` | — | see below | all |

`DEL`'s effect depends on the path it's given: a record field becomes
`UNSET` (schema mode) or is removed outright (dynamic mode); a row is
removed from its table; a table field is emptied (all rows dropped); the
record path (`()`) deletes the whole record. **`DEL ()` is terminal**: the
op list stops there and every return sees an absent record, rather than
running further ops against nothing. In dynamic mode, **deleting a record's
last field deletes the record** (the `HDEL`-of-the-last-field precedent); a
schema-mode record is never deleted implicitly — an all-zero record is a
valid state.

**Table ops** (path = the table field itself):

| Op | Operand | Effect |
|---|---|---|
| `MIGRATE` | `a` = from version (or `wire.OperateMigrateFromDynamic`), `aux` = flags, `bytes` = new schema blob | must be the first op in the list; append-only evolution or a dynamic freeze |
| `CONFIG` | `a` = cap, `aux` = policy, `bytes` = byCol name | **dynamic mode only** — sets the table's eviction triple, creating an empty table if absent; schema mode rejects it (`wire.ErrOperateOpcode`) since eviction there is part of the schema, changed via `MIGRATE` |
| `TRIM` | `a` = keep, `aux` = policy, `bytes`/`b` = byCol | one-off shrink of the table to `keep` rows by the given policy, without changing the table's stored policy; a no-op against an absent table |

**Control ops:**

| Op | Operand | Effect |
|---|---|---|
| `IF` | `aux` = cmp, `a`/`bytes`, `b` = n | if `node(path) cmp v` is false, skip the next `n` ops |
| `CHECK` | `aux` = cmp, `a`/`bytes` | if false, **abort the whole list** — record unchanged, result status `CHECK_FAILED` with this op's index |

Comparators: `EQ NE LT LE GT GE EXISTS ABSENT`. A scalar compares in its
domain (an absent numeric reads as 0; bytes compare bytewise); **a table
compares by its row count**; **a row has no value of its own and compares as
1 (present) / 0 (absent)**, making `EXISTS`/`ABSENT` the only comparators
that mean anything on it. `IF`'s skip count `n` is bounded by the remaining
op list and is validated whenever the `IF` runs — not only when the skip
branch is taken — so a malformed op list is rejected regardless of what the
record holds. On `CHECK_FAILED`, return specs are still evaluated, against
the record as it was *before* the call, so the caller can see why it failed.

### Returns

Each return spec is `(mode, path)`, evaluated after every op has applied (or,
on `CHECK_FAILED`, against the original record):

- **`VALUE`** — the node's self-describing encoding: a scalar as
  `[type u8][data]`; a row as `[mode][nCols]{[type][data]}*` in schema order
  (schema mode) or `[mode][nCols]{[nlen][name][type][data]}*` (dynamic mode);
  the record as its full stored encoding, mode byte and schema included, so
  it's decodable stand-alone (`wire.DecodeRecord` reads a plain `get` of an
  operate key the same way); a **table as `[mode][nRows]{rows}` exactly as
  stored, without a schema blob** — the caller is expected to already hold
  the schema. An absent path returns `[wire.OperateTypeUnset]`.
- **`COUNT`** — a tagged `U64`: row count for a table, field count for the
  record, `1`/`0` for a present/absent scalar or row. Absent is `0`.

### TTL modes

| Mode | Behavior |
|---|---|
| `KEEP` (default) | leave the existing expiry alone; no expiry on create |
| `SET` | refresh to the call's TTL on every call, create or update alike; a TTL of `0` **clears** the expiry |
| `CREATE_ONLY` | set the TTL only when this call creates the record (`incr_ex` semantics); an update preserves the existing deadline verbatim |

### Eviction policies

A table's `cap` (0 = unbounded, the default) and `policy` decide what's
evicted when an insert would grow the table past its cap:

| Policy | Victim |
|---|---|
| `MIN_KEY` | smallest row key |
| `MAX_KEY` | largest row key |
| `MIN_COL` | smallest value in the eviction column (`byCol`); ties go to the smallest key |
| `MAX_COL` | largest value in the eviction column; ties go to the smallest key |
| `NONE` | with `cap > 0` this still needs a deterministic victim — it evicts by `MIN_KEY` |

Recipes:

- **LRU** — `MIN_COL` on a column the app refreshes with `STAMP` on every
  touch; the stalest row is evicted.
- **LFU** — `MIN_COL` on a hit counter the app `ADD`s to on every touch; the
  least-used row is evicted.
- **Top-N by score** — `MIN_COL` on the score column: inserting a new, higher
  score evicts the current lowest.
- **Bounded FIFO / recent-N** — `MIN_KEY` (or `MAX_KEY`) with the app's own
  sequence number or timestamp as the row key and `cap = N`; there's no
  server-assigned auto-key, so the app supplies one it can reproduce
  deterministically.

The victim is a pure function of the row bytes, so every replica evicts the
same row. In schema mode the eviction triple lives in the schema and changes
via `MIGRATE`; in dynamic mode it's set per record with `CONFIG`. `TRIM` does
a one-off shrink to a row count without touching the stored policy.

### Determinism

Stored bytes are a pure function of the record: rows sorted by key, fields in
schema order (schema mode) or name order (dynamic mode) — Go map/slice
iteration never leaks into the bytes. Integer math saturates rather than
wraps; floats are canonicalized (`F32` rounded, NaN stored as the canonical
quiet NaN). The **only** clock any op ever consults is the applying
transaction's stamp (`tx.applyStamp()`), via `STAMP` — never a wall clock. On
an unreplicated or otherwise unstamped apply that stamp is `0`, so an
unstamped `STAMP` writes `0`: a `MIN_COL`/`MAX_COL` "LRU" table built on that
column then degrades to evicting by ascending key, since every row ties on
the stamp and `*_COL` ties resolve to the smallest key. A follower re-applying
the same call against the same prior record, with the same stamp, always
produces byte-identical stored bytes.

### Caps and all-or-nothing

| Cap | Value | Bounds |
|---|---|---|
| `maxFields`, `maxCols` | 65535 | fields per schema, columns per table |
| `maxSchemaBytes` | 4096 B | encoded schema size |
| `maxRows` | 1<<20 | rows per table |
| `maxRowWidth` | 4096 B | key + columns of one row |
| `maxKeyLen`, `maxNameLen` | 255 | row keys and (dynamic) field/column names |
| `maxBytesLen` | 65535 B | a `BYTES` field's payload |
| `maxNameBytes` | 4096 B | all names stored in one record |
| `maxRecordBytes` | 16 MiB | the record's total encoded size after apply |
| `maxOps`, `maxRet` | 4096 each | ops and return specs per call |

Hitting any cap is an **error with the record left unchanged** — never an
implicit eviction. Row and column growth is charged against these budgets
before anything is allocated, and the stored-bytes decoder is hardened the
same way a hostile `put` value would need to be handled: every count is
bounded before it sizes an allocation, and every read is truncation-checked.

### Example: schema mode

The v1-session shape from the design doc — a per-key bidder counter, a bid
history bitmask, and a capped, LRU-evicted table of recent bidders — built
through the Go client's `OperateBuilder`:

```go
schema := &wire.Schema{Version: 1, Fields: []wire.FieldDef{
    {Name: "rc", Type: wire.OperateTypeU8},
    {Name: "bc", Type: wire.OperateTypeU8},
    {Name: "hist", Type: wire.OperateTypeU32},
    {Name: "b", Type: wire.OperateTypeTable, Table: &wire.TableDef{
        KeyType: wire.OperateTypeU64,
        Cols: []wire.ColumnDef{
            {Name: "c", Type: wire.OperateTypeU16},
            {Name: "hi", Type: wire.OperateTypeI64},
            {Name: "t", Type: wire.OperateTypeU32},
        },
        Cap: 1024, Policy: wire.OperatePolicyMinCol, ByCol: 2, // LRU on "t"
    }},
}}

rowKey := client.KeyU64(bidderID)
args, err := client.NewOperate(sessionKey).WithSchema(schema).
    TTL(90*time.Second, wire.OperateTTLCreateOnly).
    If(client.F("rc"), wire.OperateCmpGE, 255, 2). // guard: halve before overflow
    Shr(client.F("rc"), 1).
    Shr(client.F("bc"), 1).
    Add(client.F("rc"), 1).
    Add(client.Col("b", rowKey, "c"), 1).       // this bidder's bid count
    Max(client.Col("b", rowKey, "hi"), bidAmount).
    Stamp(client.Col("b", rowKey, "t"), wire.OperateStampS). // keeps the row fresh for MIN_COL
    Shl(client.F("hist"), 1).
    Or(client.F("hist"), won).
    Return(client.F("rc")).
    Return(client.Row("b", rowKey)).
    Args()
if err != nil {
    return err
}
res, err := c.Operate(ctx, args)
```

One round trip: guards the overflow, updates three fields and one table row
(creating and, if the table is full, evicting a row as needed), and reads
back the new bidder count and the touched row — no client-side read, no CAS
retry.

### Example: dynamic mode

A `HINCRBY`/`HSET`-style counter with no schema declared up front:

```go
args, err := client.NewOperate([]byte("user:42")).Dynamic().
    AddT(client.F("hits"), wire.OperateTypeI64, 1).
    Return(client.F("hits")).
    Args()
if err != nil {
    return err
}
res, err := c.Operate(ctx, args)
hits, err := client.DecodeOperateValue(res.Values[0]) // wire.Cell{Type: I64, U: ...}
```

The first call creates the record in dynamic mode and the `hits` field with
it; later calls (`Add`, without the `T` suffix) resolve it as an existing
`I64` field. `SetBytesT(client.F("name"), wire.OperateTypeBytes, v)` adds a
second, differently-typed field the same way. A plain `get` on the key
decodes with `wire.DecodeRecord`.

### Wire summary

A call encodes as (`sdk/wire/operate.go`):

```
[keyLen u16][key][ttlMs u64][ttlMode u8][create u8][schemaLen u16][schema]?
[nOps u16]{op}*[nRet u16]{ret}*
```

- **op**: `[opcode u8][type u8][aux u8][path][a i64][b i64][blen u16][bytes]`
- **path**: `[kind u8]` then, per kind, `[fseg]`, `[fseg][kseg]`, or
  `[fseg][kseg][cseg]`, where `fseg`/`cseg` is `[0][pos uvarint]` (schema
  position) or `[1][len u8][name]` (name segment), and `kseg` is `[len
  u8][key bytes]`
- **ret**: `[mode u8][path]`
- **result**: `[status u8][failedOp u16 iff CHECK_FAILED][nRet u16]{[vlen
  u32][value]}*`

Every multi-byte field in the call and result frames (lengths, `a`/`b`,
`ttlMs`, `failedOp`, `vlen`) is **big-endian**, matching every other builtin's
wire args. The stored **record and cell data themselves are little-endian**
(fixed-width ints, table row keys) — the wire args format and the on-disk
record format are independent conventions.

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
