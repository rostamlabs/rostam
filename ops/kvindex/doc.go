// SPDX-License-Identifier: Apache-2.0

// Package kvindex holds a shard's derived posting index over KV records: the
// equality postings a kv_query narrows its candidate set with, plus the
// definitions they are built from.
//
// # It is derived state, never a durability unit
//
// Nothing here is snapshotted, written to the WAL, or carried in a Raft log.
// A restart, a restore, and a definition install all rebuild the postings by
// walking the live cache. That is what lets a definition be activated
// node-locally without risking apply divergence: Reindex and Drop never
// return an error, never touch the cache, and never change what a handler
// stores. A record that fails to resolve simply has no posting, and that is
// the whole of the consequence.
//
// # Postings are hints, and the guarantee is SUPERSETNESS
//
// Candidates returns keys the caller then re-reads, re-resolving the full
// compiled predicate against the LIVE value before any row is returned. So a
// stale posting costs one wasted lookup and can never produce a wrong row,
// while a MISSING posting loses a row silently. The contract is therefore
// asymmetric and one-directional:
//
//	Candidates(sel, after, budget) ⊇ { live keys under sel.Def.Prefix, above
//	  `after`, whose live value satisfies sel }        (never a proper subset)
//	Candidates(sel, after, budget) ⊆ { keys this Set has a posting for }
//
// The scope is the definition's KeyPrefix, always: an indexed kv_query
// answers "the keys under this prefix that match", never "the keys that
// match". Keys outside the prefix are outside the answer, and only a scan
// sees them.
//
// Supersetness holds because a selector is only ever a POSITIVE leaf (eq, in,
// gt, gte, lt, lte) on the named definition's own path: any record under the
// prefix satisfying such a leaf resolves to a value at that path, therefore
// has a posting, therefore is a candidate. Negations, absence tests and
// unindexed paths are re-checked by the predicate on the live value and never
// narrow the candidate set.
//
// # Lock order is cache-lock → index-lock, one way, always
//
// The cache's onRemove hook fires with a shard's WRITE lock held and calls
// Drop, so Drop takes only Set.mu and does bounded work. The rule that makes
// that safe is the symmetric one: NO method here holds Set.mu across a call
// that could take a cache lock. Candidates copies the keys it found under
// RLock and returns them with the lock released; Rebuild and Backfill call
// the caller's walk function with Set.mu NOT held, one short-locked Reindex
// per visited entry. Holding Set.mu across a cache read self-deadlocks
// single-threaded, with no concurrency required: cache.Get of an expired key
// on a non-replicated shard drops it, which fires onRemove, which wants
// Set.mu. The bounded reconcile pass is the third consumer of that rule and
// the one whose whole shape comes from it — see reconcile.go.
//
// # Determinism
//
// Reindex is a pure function of (key, value, definitions) — the resolver's
// schema cache is keyed by the schema's exact bytes, so a hit and a miss
// answer identically. Map iteration order is never observable in a result:
// Candidates sorts its output by key, and the candidate budget is charged
// against a sum, which no ordering changes.
package kvindex
