# Cache tuning

`cache.Cache` is the storage core under every KV backend, and a capable
standalone in-process cache in its own right. All knobs live on `cache.Config`
(reachable as `DirectConfig.Cache` / `EmbeddedConfig.Cache`, or passed straight
to `cache.New`).

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `NumShards` | 256 | independent shards; must be a power of two. More shards = less lock contention. |
| `PageSize` | 16 MiB | slab page size (1 MiB – 1 GiB) |
| `MaxMemoryPerShard` | 256 MiB | per-shard cap (≥ `PageSize`); total budget ≈ `NumShards × MaxMemoryPerShard` |
| `InitialPagesPerShard` | 0 (lazy) | pre-allocate pages up front instead of on demand |
| `AtCapPolicy` | `PolicyRingbufEvict` | at capacity: overwrite oldest entries (`PolicyRingbufEvict`) or reject writes with `cache.ErrFull` (`PolicyRejectWrites`) |
| `TTLSweepIntervalMs` | 1000 | background expiry sweep; 0 disables (lazy expiry on read still applies) |
| `DataDir` | "" (heap) | directory for per-shard mmap files; enables persistence + warm restart. Linux and Windows only. |
| `Durable` | false | `msync` on commit boundaries (bounded by `MsyncIntervalMs`, default 100 ms); false = opportunistic OS flushing |
| `Mlock` | false | pin the mmap into RAM (needs `ulimit -l` headroom; failure logs and continues) |

The opt-in eviction knobs — `RelocatingEviction`, `RelocateReserveIntervalMs`,
`InPlaceSameSizeUpdate`, `InPlaceSeqlockReads` and `SieveVisitedBit`, all off by
default and also settable as `rostam.CacheConfig` fields — act only on shards
that evict at capacity, and some only on heap shards. Check
[where each takes effect](../server/running.md#where-each-knob-takes-effect)
before enabling one, and read the warning there on `InPlaceSeqlockReads`, whose
read protocol is not covered by race detection.

## Read semantics

- `Get(key)` — **policy-dependent**. Under `PolicyRejectWrites` it is
  **zero-copy**: the returned slice aliases the page arena, valid until the next
  write, so copy it if you retain it. Under the default `PolicyRingbufEvict` it
  is a freshly allocated copy you own and may retain — one allocation per hit,
  and the reason is safety rather than oversight: ring-buffer eviction overwrites
  live page bytes, so an alias could be pulled out from under you.
- `GetInto(dst, key)` — copies into your reusable buffer and returns it. 0
  allocations on a hit when capacity suffices, and the copy is safe to retain
  even under the evicting ring-buffer policy. This is the hot-loop read.

```go
buf := make([]byte, 0, 256)
for _, k := range keys {
	buf, err = c.GetInto(buf[:0], k)
	// use buf before the next iteration reuses it
}
```

## Persistence & warm restart

With `DataDir` set, each shard's pages live in a memory-mapped file. On
restart the index is rebuilt from the pages — a warm start that skips
re-population. Under `Embedded`, the cache also tracks the Raft
`AppliedIndex`, so recovery replays only Raft entries newer than the persisted
state.

Durability levels:

1. **Heap mode** (`DataDir: ""`) — fastest, nothing survives restart.
2. **mmap, opportunistic** (`DataDir` set, `Durable: false`) — survives process
   restart; OS decides flush timing, so a power loss can lose recent writes.
3. **mmap, durable** (`Durable: true`) — `msync` at commit boundaries, at most
   `MsyncIntervalMs` behind.

On `Embedded`, the Raft log is the source of truth regardless — mmap
persistence is an optimization for restart speed, not the correctness
mechanism.

## Sizing guidance

- Size `MaxMemoryPerShard × NumShards` to your working set plus headroom;
  under `PolicyRingbufEvict` the store behaves as a bounded cache, evicting
  oldest-first per shard once full.
- Prefer more shards over bigger shards for write-heavy concurrent loads.
- Values are stored in slab pages; very large values fragment less with a
  larger `PageSize`.
