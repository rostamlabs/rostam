# Roadmap

Planned and in-flight work. This is a statement of direction, not a commitment to
dates or ordering — items land as they're built, reviewed, and proven. For what
has already shipped, see [docs/changelog.md](docs/changelog.md).

## Key-value store

The KV store already covers `get`, `put`, `del`, `expire`, `incr`, and batched
writes (`put_batch`). The atomic conditional-write family — `set_nx` (set if
absent), `cas` (compare-and-swap), and compare-and-delete (`cad`, safe lock
release) — is **in review** and turns Rostam into a correct backend for
distributed locks and once-only writes.

The theme for what's next is *completing the primitives* for the three workloads
people reach for a fast KV store to do: **caching, distributed locks, and rate
limiting.** Each item below is a small atomic handler on the same execution model
the conditional-write ops already use (the handler runs under the shard lock with
a leader-stamped clock, so a read-decide-write sequence is atomic and
replica-deterministic with no extra machinery).

### Next — completes the lock / rate-limit / cache trifecta

- **`mget` — batch read.** The read counterpart to `put_batch`: fetch N keys in
  one round-trip, fanned out by shard. Removes the current asymmetry (writes can
  batch, reads can't) and powers bulk cache reads (e.g. `Cache::many`), feature
  flags, and session loads.
- **`incr_ex` — increment, set TTL only on create.** The correct rate-limiter
  primitive. The usual "`incr` then `expire`" idiom races: if the caller dies
  between the two, the counter never expires and the window never resets. Doing
  both atomically — setting the TTL only when the key is newly created — closes
  it (the same class of race the conditional-write ops close for locks).
- **Compare-and-expire — lock renewal.** Refresh a key's TTL *only if* its value
  still equals an expected token. The third leg of the lock story: `set_nx`
  acquires, `cad` releases, and this lets a long-running holder safely renew its
  lease without over-provisioning the initial TTL or clobbering a lock another
  worker has since re-acquired.

### Then — common ergonomics (each ~one small handler)

- **`getdel` — atomic get-and-delete.** Read-and-consume in one op: one-time
  tokens (magic links, password resets), "claim once," a simple queue pop.
- **`ttl` and `persist`.** Read a key's remaining TTL, and remove a TTL to make a
  key permanent. Today TTLs can be set but not introspected or cleared — locks
  and sessions want both ("how long left?", "pin this").
- **`exists`.** A cheap presence check that doesn't transfer the value — for
  idempotency-key checks and feature flags.
- **`getset` / `swap`.** Atomic get-then-set returning the old value. Largely
  covered by `cas`; primarily useful for atomic counter-window rollover.

### Larger design item

- **Prefix `scan` — key enumeration.** High user value (cache tags, session GC,
  targeted flush) but architecturally heavier: the store is a hash index, not an
  ordered one, so a prefix scan is a full per-shard walk (the same shape as the
  generational flush). This needs its own design pass on cost, cursoring, and
  consistency before it's scheduled — it is not a drop-in like the items above.

## Platform & operations

- **Windows persistence.** File-mapping support so `DataDir` persistence works on
  Windows (today only heap mode runs there) — in review.

---

*Have a use case that needs an op not listed here? Open an issue — the KV op set
is deliberately small and grows from real workloads.*
