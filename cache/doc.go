// SPDX-License-Identifier: Apache-2.0

// Package cache provides a sharded in-memory key-value store with lazy
// slab pool allocation and per-shard TTL.
//
// Each shard owns an independent set of pages (fixed-size byte slabs) and
// an index from hashed keys to (page, offset, size) tuples. Pages allocate
// lazily on first write up to a configured cap; when the cap is reached,
// behavior is controlled by Config.AtCapPolicy.
//
// Concurrency: all exported operations are safe for concurrent use. Writes
// (including TTL expiration and eviction) hold the shard's write lock and are
// serialized. The per-shard index is a single-writer / multi-reader
// open-addressing table of atomic slots (see indextable.go), so readers probe
// it without taking a lock.
//
// The read path's locking depends on Config.AtCapPolicy:
//
//   - PolicyRejectWrites never overwrites live page bytes in place (an at-cap
//     shard rejects new writes), so the value bytes a reader observes are
//     immutable for the entry's lifetime. Reads are fully lock-free and return a
//     zero-copy alias into the page backing store.
//
//   - PolicyRingbufEvict evicts by reclaiming whole pages, and its read path
//     depends on the page backing:
//
//   - Heap mode reads lock-free. Eviction never overwrites a live heap page in
//     place; it RETIRES the page by swapping in a fresh, empty page object with a
//     new generation and abandoning the old object, which is then frozen (never
//     mutated again) until the GC reclaims it. A reader resolves the page OBJECT
//     pointer atomically (pageSlots), so it either keeps the old frozen object
//     GC-alive and reads its immutable bytes, or loads the fresh object and finds
//     its generation != the slabRef's generation and misses WITHOUT reading the
//     bytes a writer may be appending. Frozen objects replace the epoch/RCU
//     machinery a byte-recycling ring would need. Reads still return an owned
//     copy (callers may mutate it).
//
//     Config.InPlaceSameSizeUpdate gives that up deliberately. It lets a write
//     overwrite the entry already stored for its key when the new one is framed
//     identically, which stops updates from stranding a dead copy apiece — but
//     live bytes then change under readers, so those shards move to the
//     read-locked path below. The flag is off by default and the lock-free path
//     is what a heap ringbuf shard uses without it.
//
//   - Mmap mode — and any shard with in-place same-size updates enabled — takes
//     the shard read lock. An mmap page object wraps a fixed
//     region of the persisted file and cannot be swapped for a fresh allocation,
//     so eviction overwrites its bytes in place; a lock-free read would risk a
//     torn value and race the writer's overwrite at the byte level (which the
//     race detector flags regardless of any post-hoc torn-value check such as a
//     seqlock). Its reads take the read lock for the probe and the value copy,
//     exactly excluding the writer's overwrite.
//
// # The mutable region
//
// Each heap shard keeps an explicit FIFO of the pages it considers MUTABLE. A heap
// page is born mutable — freshHeapPageLocked is the one place a heap page is
// constructed, so the initial allocation, both retirement paths and the free-page
// reserve's handoff all mint the flag in the same place — and is SEALED when the
// region's bound pushes it off the front of the FIFO. The flag is monotone: there
// is no operation that returns a sealed page to mutable, so an observer that sees
// "sealed" may conclude it will stay sealed for that page object's life. Mutability
// comes back only with a NEW page object, which is what retirement already builds.
//
// Membership is a FIFO of page pointers rather than a comparison on page.gen:
// generations wrap, and a FIFO says exactly what is wanted with no wrap-aware
// arithmetic. Mmap pages are never members (their objects are reused in place, so a
// flag on one could not stay monotone) and always read as not mutable, which is
// also the true answer for them — their bytes may never be overwritten where they
// lie.
//
// Sealing forbids OVERWRITING a page's live bytes; appending into its tail stays
// legal. Nothing in the cache consults the flag today and no shard the cache
// constructs bounds its region, so the whole mechanism is bookkeeping: it cannot
// change which writes go in place, where a record lands, or what a read observes.
//
// This is the foundational component of Rostam. Higher-level features
// (Raft replication, transactions, the network server, the migration
// shim) are layered on top — see docs/concepts/architecture.md.
package cache
