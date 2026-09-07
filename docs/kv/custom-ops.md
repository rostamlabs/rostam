# Custom ops (native Go stored procedures)

You build the op registry, so you can register your own Go functions as
server-side ops — no WASM, no module to ship. A handler is just:

```go
func(tx *ops.TxContext, args []byte) ([]byte, error)
```

It runs inside the shard lock, so an `OpReadWrite` op is an **atomic
read-modify-write of a whole record in one round trip** — no client-side
`Get`-modify-`Put`, no CAS retry loop.

## A worked example

One `match` op folds a game result into a player record, updating four fields
atomically — an increment, a monotonic high-water mark, a running sum, and a
rolling bitmask:

```go
type playerStats struct {
	Games      int64  `json:"games"`
	BestStreak int64  `json:"bestStreak"`
	Score      int64  `json:"score"`
	Recent     uint64 `json:"recent"` // rolling win/loss mask, newest result in bit 0
}

// "match" — one match result → one atomic record update. Runs under the shard
// lock; on Embedded it replicates as a single Raft entry, so concurrent updates
// to the same player are linearized. Keep it deterministic (no clock/RNG — pass
// such values via args) so every replica's FSM computes the identical record.
reg.RegisterRoutable("match", ops.OpReadWrite,
	func(tx *ops.TxContext, args []byte) ([]byte, error) {
		key, streak, points, won := decodeMatch(args) // std [keyLen u16][key] prefix + payload

		var s playerStats
		if cur, err := tx.Get(key); err == nil {
			if err := json.Unmarshal(cur, &s); err != nil {
				return nil, err
			}
		} else if err != cache.ErrNotFound {
			return nil, err // a real error, not just a first-seen key
		}

		s.Games++
		if streak > s.BestStreak {
			s.BestStreak = streak
		}
		s.Score += points
		s.Recent = (s.Recent << 1) | won

		out, err := json.Marshal(&s)
		if err != nil {
			return nil, err
		}
		return out, tx.Put(key, out, 0) // single write; returns the new record
	},
	ops.KeyExtractorByHandle("std")) // route on the leading [keyLen u16][key]
```

Wire it in once at startup, then call it by name:

```go
reg := ops.NewRegistry()
_ = ops.RegisterBuiltins(reg) // get / put / del / incr / expire / vector ops
_ = registerMatchOp(reg)      // the op above
store, err := rostam.NewDirect(rostam.DirectConfig{Ops: reg})
// ...
// res, err := store.Call(ctx, "match", encodeMatch([]byte("player:42"), streak, points, won))
```

`decodeMatch`/`encodeMatch` are trivial byte helpers. For a JSON-free hot path,
store the record as a fixed packed layout and use `binary.BigEndian` — the op
logic is identical.

## The rules

| Rule | Why |
|---|---|
| Args for `"std"`-routed ops must start with `[keyLen u16][key]` | that prefix is how the extractor picks the destination shard |
| Op names are ≤ 255 bytes and never exactly 2 bytes | a 2-byte name collides with the protocol-v2 version byte |
| `OpReadWrite` handlers must be deterministic — no clock, no RNG, no map iteration order dependence; pass such values in `args` | on Embedded the op replays on every replica's FSM; divergence corrupts the cluster |
| `OpReadOnly` handlers must not mutate | they bypass Raft and may serve from any replica |
| Register the same ops on every node | replicas look ops up by name at apply time |

## OpKind and routing

- `ops.OpReadOnly` — executes locally, never enters the Raft log.
- `ops.OpReadWrite` — serialized through Raft (Embedded) / the shard lock (Direct).
- `Register(name, kind, fn)` — shardless op (cluster-level, e.g. health checks).
- `RegisterRoutable(name, kind, fn, keyExtractor)` — routed to the shard owning
  the extracted key. `ops.KeyExtractorByHandle("std")` reads the standard
  `[keyLen u16][key]` prefix.
- `RegisterRoutableCrossShard(...)` — for handlers that touch keys beyond their
  routing shard; `Direct` serializes these behind a global barrier instead of a
  single shard lock.

## TxContext

Inside a handler, `tx` gives you the shard-local store:

| Method | Semantics |
|---|---|
| `tx.Get(key)` | read; result aliases the page store — copy if you retain it |
| `tx.Put(key, val, ttl)` | insert/replace |
| `tx.Del(key)` | delete, reports existence |
| `tx.Expire(key, ttl)` | update TTL; `cache.ErrNotFound` if absent |
| `tx.Cache()` | escape hatch to the underlying cache (stats, iteration) |
| `tx.Vectors()` | the vector `CollectionStore` (nil if the dispatcher has none) |

## Don't need custom Go? Use the built-in `operate`

If your update is just atomic counter/bitfield math (like the `match` example
above), you don't need to compile a handler at all — the built-in **`operate`** op
is a ready-made generic version: the client sends a *list* of typed integer ops and
the server applies them to one record atomically under the shard lock. The op-list
is data, so the logic stays in your app. See
[Atomic multi-field updates: `operate`](overview.md#atomic-multi-field-updates-operate).
Reach for a custom Go op only when the math isn't expressible as that op-list.

## Native Go vs WASM

A native Go op lives in the server process, so it works on `Direct` and
`Embedded` — but it can't travel to a remote cluster you don't compile. To ship
logic over the wire to a running cluster, use a sandboxed
[WASM procedure](wasm.md) instead.
