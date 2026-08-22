// SPDX-License-Identifier: Apache-2.0

// Package rostam is the public library entry point. It exposes a Store
// interface backed by either an in-process cluster.Node (Embedded mode)
// or a networked client.Client (Client mode). Callers code against the
// interface and switch backends with one constructor.
package rostam

import (
	"context"
	"errors"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Store is the unified interface implemented by both backends. All
// methods are safe for concurrent use.
type Store interface {
	// Get reads a value. Local read; never goes through Raft. Returns
	// ErrNotFound if absent or expired.
	Get(ctx context.Context, key []byte) ([]byte, error)

	// GetInto is the allocation-light variant of Get: the value is copied into
	// dst (reusing its capacity when large enough) and the resulting slice is
	// returned. With a reused dst the Networked path is a zero-allocation read
	// (pooled request args, no defensive response copy). Same ErrNotFound
	// semantics as Get. The returned slice may alias dst; do not retain the
	// argument after the call.
	//
	// Note: on the Networked backend GetInto goes through the request-response
	// path even when PipelineDepth > 0 (pipelining applies to Get, not the
	// zero-copy CallFunc path GetInto uses) — trade pipelining throughput for
	// the per-call allocation win accordingly.
	GetInto(ctx context.Context, key, dst []byte) ([]byte, error)

	// Put writes through Raft with the given TTL. Returns ErrNotLeader
	// if the backing node (Embedded) or the client's exhausted retry
	// budget (Networked) cannot reach the shard leader.
	Put(ctx context.Context, key, value []byte, ttl time.Duration) error

	// PutBatch writes many key/value pairs, batching per shard so each shard
	// takes one Raft log entry (fsync/round-trip/apply) per chunk instead of one
	// per key — the bulk-insert fast path. Same durability/consistency as Put;
	// ErrNotLeader semantics propagate as with Put.
	PutBatch(ctx context.Context, entries []ops.PutEntry) error

	// Del removes a key. Returns (true, nil) if the entry existed and
	// was deleted, (false, nil) if absent, (false, err) on failure.
	Del(ctx context.Context, key []byte) (bool, error)

	// Call invokes a registered op by name with caller-encoded args.
	// Read-only ops execute locally; read-write ops go through Raft.
	Call(ctx context.Context, op string, args []byte) ([]byte, error)

	// IsLeader reports whether the backing node owns leadership for the
	// shard hashing to key. Embedded: authoritative. Networked:
	// best-effort from the topology cache.
	IsLeader(key []byte) bool

	// LeaderAddr returns the current leader's address for key's shard,
	// or "" if unknown.
	LeaderAddr(key []byte) string

	// Close releases all resources. Idempotent.
	Close() error

	// RegisterWASM compiles and registers a WASM module as a named op.
	// On Embedded stores this round-trips through Raft, broadcasting the
	// registration into every shard group's log. On Direct stores the
	// module is registered in-process only (no replication). On networked
	// stores the call is forwarded to the leader via __register_wasm__.
	//
	// On the replicated paths a successful return means every shard group
	// COMMITTED the registration, not that the op is invocable everywhere yet:
	// each node opens its route gate per shard group as it applies the entry,
	// so an invocation may briefly return a transient, retryable error. See
	// client.Client.RegisterWASM for the full contract.
	//
	// UPDATING A LIVE MODULE'S BYTES IS SUPPORTED on the replicated paths, since
	// per-group version binding shipped. The version a group executes is bound to
	// that group's own log, so an update cannot leave two replicas of one group
	// executing the same committed entry with different bytes — which is what made
	// it unsafe before, when the executing version was node-wide while
	// registrations commit per shard group.
	//
	// ONLY THE BYTES MAY CHANGE. Kind is frozen at first registration and a change
	// to it is still refused with cluster.ErrWASMUpdateUnsupported — register the
	// new contract under a NEW op name. The freeze is not a simplification: Kind is
	// read on the propose side to decide whether an invocation replicates at all,
	// so a per-group Kind cannot be resolved before the group is known. The key
	// extractor is the other field that cannot be per group — it is what COMPUTES
	// the group index — and it is CONSTANT rather than frozen: every WASM op uses
	// the same one, so there is nothing to change (ops.WASMKeyExtractorHandle).
	// An identical re-registration remains accepted — that is the idempotent retry
	// the partial-broadcast recovery path depends on. Direct stores reject a
	// re-registration for an unrelated reason (a duplicate op name); see
	// directStore.RegisterWASM.
	//
	// pushReport IS PART OF THE RESULT, NOT DIAGNOSTICS. On the replicated paths,
	// before the registration enters any log the receiving node pushes the
	// module's BYTES to every member it can reach and requires a compile verdict
	// from each one that answers; a member that refuses fails this call. The
	// report is empty when every member acked (and always on Direct stores, which
	// have no peers). When it is not empty it NAMES the members that rendered no
	// verdict — unreachable, or on a build that does not know the push op — and
	// those are exactly the members that do not hold the bytes and will have to
	// fetch them on demand. Ignoring it is choosing not to know which nodes are one
	// step from being unable to run the op.
	RegisterWASM(ctx context.Context, r WASMRegistration, module []byte) (pushReport string, err error)

	// VectorInsert adds a vector with id under the named collection. CREATE-ONLY:
	// a live id is rejected with vector.ErrDuplicateID (the id must be deleted
	// first, or use VectorUpsert to replace / VectorInsertIfAbsent to no-op).
	// Note: unlike the dense family, VectorNamedInsert and VectorMVAdd REPLACE an
	// existing id rather than erroring.
	VectorInsert(ctx context.Context, collection string, id uint64, vec []float32, opts ...WriteOpts) error

	// VectorSearch returns the k nearest neighbors of query in the named collection.
	// Convenience form; use VectorSearchExt to observe FanMeta.
	VectorSearch(ctx context.Context, collection string, query []float32, k int) ([]VectorResult, error)

	// VectorSearchInto is the allocation-light variant of VectorSearch: results
	// are appended into dst (reused when its capacity allows), returning the
	// populated slice. Pass a reused dst from a hot query loop to avoid the
	// per-search result-slice allocation. Direct (in-process) stores hit the
	// engine's zero-alloc SearchInto directly; networked stores decode the wire
	// response straight into dst with no defensive copy. Plain kNN (no filter).
	// Convenience form; use VectorSearchExt to observe FanMeta.
	VectorSearchInto(ctx context.Context, collection string, query []float32, k int, dst []VectorResult) ([]VectorResult, error)

	// VectorDelete tombstones the vector with id in the named collection. Returns
	// true if the vector was present.
	VectorDelete(ctx context.Context, collection string, id uint64, opts ...WriteOpts) (bool, error)

	// VectorInsertIfAbsent inserts vec under id ONLY if id is not currently live,
	// reporting whether it inserted (false = no-op because id was already live).
	// ATOMIC: the liveness check and the insert run in a single op (one engine
	// write-lock critical section, serialized on the partition's Raft log) — there
	// is no check-then-act gap, so when it races a concurrent upsert on the same id
	// the live value always wins (it never clobbers). LIVENESS: an id is live iff
	// present and neither tombstoned (deleted) nor TTL-expired; a dead slot counts
	// as absent and is resurrected with the new value. The online-copy primitive
	// that closes Race A (value clobber); the copy MUST use this, never a plain upsert.
	VectorInsertIfAbsent(ctx context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) (bool, error)

	// VectorExists reports whether id is currently live in the named collection,
	// using the same liveness definition as search admission: tombstoned (deleted)
	// and TTL-expired ids are NOT live, and a never-inserted id is absent. O(1)
	// idMap probe (no scan). The cheap liveness probe the online-copy resurrection
	// guard uses to re-check the source generation after an insert-if-absent (Race B).
	VectorExists(ctx context.Context, collection string, id uint64) (bool, error)

	// CreateCollection registers a new vector collection with the given configuration.
	CreateCollection(ctx context.Context, name string, cfg VectorConfig) error

	// VectorInsertExt adds a vector with optional TTL and metadata. CREATE-ONLY,
	// like VectorInsert: a live id is rejected with vector.ErrDuplicateID (delete
	// first, or use VectorUpsert / VectorInsertIfAbsent).
	VectorInsertExt(ctx context.Context, collection string, id uint64, vec []float32, opts VectorInsertOpts) error

	// VectorSearchExt returns the k nearest neighbors matching the optional filter.
	// The returned FanMeta reports cross-shard fan-out completeness (degraded when
	// a partition was unreachable in Partial mode); it is the zero value on a
	// single-partition or non-clustered backend.
	VectorSearchExt(ctx context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorResult, FanMeta, error)

	// VectorHybridSearch fuses a dense KNN lane and a sparse lane into the top-k.
	// An empty opts.Sparse degrades to pure dense; a nil dense query is pure sparse.
	// The returned FanMeta reports cross-shard fan-out completeness (see VectorSearchExt).
	VectorHybridSearch(ctx context.Context, collection string, dense []float32, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error)

	// VectorQuery runs the unified Query API (vector_query): a root leaf plus N
	// single-level prefetch leaves combined by FUSION (RRF/Weighted/DBSF over the
	// prefetch lanes) or RERANK (the root re-scores the union of the prefetch
	// candidates). specBytes is the marshaled pb.QuerySpec carried on the wire;
	// spec is the decoded engine spec the coordinator uses for the fan-out
	// fusion/rerank merge. Returns the final top-k + FanMeta (cross-shard
	// completeness; see VectorSearchExt). The dense-family Qdrant-parity query op.
	VectorQuery(ctx context.Context, collection string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error)

	// VectorQueryGrouped runs the GROUPED Query API (vector_query with a non-empty
	// spec.GroupBy): the SAME root leaf + N prefetch leaves combined by FUSION/RERANK,
	// but the final ordered candidate pool is collapsed by the GroupBy metadata field
	// into the top-k GROUPS (best member first) with up to spec.GroupSize hits each —
	// the Query API generalization of VectorSearchGroups. Grouping is a deterministic
	// post-process over the EXACT global ordered pool (the same merge VectorQuery uses),
	// so P>1==P1 for both modes. Dense-only in v1 (named/MV grouped query fails loud).
	VectorQueryGrouped(ctx context.Context, collection string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorGroup, FanMeta, error)

	// VectorUpsert inserts or replaces a record (vector + document content +
	// metadata) by id — the RAG-store write path. opts carries optional
	// TTL/metadata/sparse. Replacing an id updates its vector and content.
	VectorUpsert(ctx context.Context, collection string, id uint64, vec []float32, content string, opts VectorInsertOpts) error

	// VectorSearchDocs runs a filtered KNN search and returns each hit with its
	// stored content and metadata in one call. The returned FanMeta reports
	// cross-shard fan-out completeness (see VectorSearchExt).
	VectorSearchDocs(ctx context.Context, collection string, query []float32, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error)

	// VectorSearchText runs a BM25 full-text search over the collection's indexed
	// $content and returns each hit enriched with content + metadata (like
	// VectorSearchDocs). The query is RAW text — the server tokenizes + BM25-scores
	// it (the caller ships no tokens). Requires a collection created with FullText
	// (else ErrFullTextDisabled). Under partitioning IDF is per-shard-local, so the
	// scores are APPROXIMATE (query_then_fetch); see VectorSearchExt for FanMeta.
	VectorSearchText(ctx context.Context, collection string, query string, k int, opts VectorSearchOpts) ([]VectorDocument, FanMeta, error)

	// VectorHybridText fuses a dense KNN lane with a BM25 full-text lane into the
	// top-k. The text lane is RAW query text analyzed server-side (no sparse query
	// rides the wire). opts mirrors VectorHybridOpts (fusion method/alpha/per-lane
	// pools/filter). Requires a FullText collection (else ErrFullTextDisabled). The
	// returned FanMeta reports cross-shard completeness (see VectorSearchExt). Under
	// partitioning the text lane's IDF is per-shard-local (approximate).
	VectorHybridText(ctx context.Context, collection string, dense []float32, query string, k int, opts VectorHybridOpts) ([]VectorResult, FanMeta, error)

	// VectorDeleteByFilter deletes all records matching filter (e.g. every chunk
	// of a document) and returns the count removed. A zero filter is rejected.
	VectorDeleteByFilter(ctx context.Context, collection string, filter VectorFilter) (int, error)

	// VectorSearchGroups runs a group-by-document search: it collapses KNN hits
	// sharing the opts.GroupBy metadata value into groups, returning the top-k
	// groups (best member first) with up to opts.GroupSize hits each. The RAG
	// "top-k distinct documents" retrieval primitive. The returned FanMeta reports
	// cross-shard fan-out completeness (see VectorSearchExt).
	VectorSearchGroups(ctx context.Context, collection string, query []float32, k int, opts VectorGroupOpts) ([]VectorGroup, FanMeta, error)

	// VectorMVCreateCollection registers a late-interaction (multi-vector /
	// ColBERT MaxSim) collection. Multi-vector collections are in-memory only.
	VectorMVCreateCollection(ctx context.Context, name string, cfg MultiVectorConfig) error

	// VectorMVDropCollection removes a multi-vector collection.
	VectorMVDropCollection(ctx context.Context, name string) error

	// VectorMVAdd inserts or replaces a document's token vectors (each length
	// cfg.Dim) in a multi-vector collection. meta is optional.
	VectorMVAdd(ctx context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata, opts ...WriteOpts) error

	// VectorMVSearch runs a MaxSim late-interaction search, returning the top-k
	// documents ranked by descending score. The returned FanMeta reports
	// cross-shard fan-out completeness (see VectorSearchExt).
	VectorMVSearch(ctx context.Context, name string, query [][]float32, k int, opts MultiSearchOpts) ([]MultiResult, FanMeta, error)

	// VectorMVHybridSearch fuses an MV collection's MaxSim (late-interaction dense)
	// lane and its per-doc sparse lane into the top-k (cross-modality hybrid). query
	// is the MV token query matrix (empty ⇒ sparse-only); sparseQ is the doc-sparse
	// query (zero ⇒ MaxSim-only). opts carries the fusion method/params, the optional
	// shared-payload filter (applied to BOTH lanes), and read-consistency / partition
	// opts. A Linearizable read arms the meta + per-shard barriers. Returns the fused
	// top-k (id + fusion score). The MV analogue of VectorNamedHybridSearch.
	VectorMVHybridSearch(ctx context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, error)

	// VectorMVHybridSearchExt is VectorMVHybridSearch with the cross-partition
	// fan-out completeness exposed as FanMeta (mirroring VectorHybridSearch). Under
	// the default OnPartitionUnavailable=Partial an unreachable partition yields
	// FanMeta{Degraded:true, Missing:...} instead of a silently incomplete top-k; a
	// single-node/unpartitioned backend never fans out and reports a zero FanMeta.
	VectorMVHybridSearchExt(ctx context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, FanMeta, error)

	// VectorMVQuery runs the unified Query API (vector_mv_query) against a
	// MULTI-VECTOR collection: a root leaf plus N single-level prefetch leaves where
	// every leaf is an MV node (a MaxSim late-interaction lane and/or the doc-level
	// sparse field), combined by FUSION (RRF/Weighted/DBSF over the prefetch lanes) or
	// RERANK (the root re-scores the union of the prefetch candidates). Both MV lanes
	// are score-descending, so the coordinator folds via FuseScoreLanes (the
	// orientation-aware merge). specBytes is the marshaled pb.QuerySpec carried on the
	// wire; spec is the decoded engine spec the coordinator uses for the fan-out
	// fusion/rerank merge. Returns the final top-k + FanMeta (cross-shard
	// completeness; see VectorSearchExt). The MV-family analogue of VectorNamedQuery.
	VectorMVQuery(ctx context.Context, name string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error)

	// VectorMVDelete removes a document from a multi-vector collection, returning
	// whether it existed.
	VectorMVDelete(ctx context.Context, name string, docID uint64, opts ...WriteOpts) (bool, error)

	// VectorMVAddIfAbsent adds a document ONLY if docID is not already present,
	// reporting whether it inserted (false = no-op because docID was live). ATOMIC
	// (single op, Raft-serialized, no check-then-act gap) so a copy's add-if-absent
	// racing a concurrent replace-Add never clobbers the live document. LIVENESS:
	// the MV index has no tombstones/TTL, so docID is live iff it has a token-set
	// entry; a deleted docID counts as absent. The MV online-copy primitive that
	// closes Race A. Mirrors VectorInsertIfAbsent for the multi-vector path.
	VectorMVAddIfAbsent(ctx context.Context, name string, docID uint64, tokens [][]float32, meta VectorMetadata) (bool, error)

	// VectorMVExists reports whether docID is currently live in a multi-vector
	// collection (O(1) map probe). The MV resurrection-guard liveness check (Race B).
	VectorMVExists(ctx context.Context, name string, docID uint64) (bool, error)

	// VectorScroll lists live documents matching filter (zero filter = all),
	// enriched with content + metadata, up to limit (0 = no cap). A query-less
	// listing primitive (used by framework adapters for enumerate/count/filter).
	// The returned FanMeta reports cross-shard fan-out completeness (see VectorSearchExt).
	//
	// Cursor pagination (resume-after-id): opts.Cursor is the opaque token from the
	// previous page (empty = first page); the returned nextCursor is the token for
	// the next page (empty = exhausted, no more pages). Scroll is deterministic
	// id-ASCENDING globally; a no-cursor limit-capped scroll returns the smallest-id
	// `limit` documents. A malformed cursor surfaces ops.ErrBadScrollCursor.
	VectorScroll(ctx context.Context, collection string, filter VectorFilter, limit int, opts VectorScrollOpts) (docs []VectorDocument, meta FanMeta, nextCursor string, err error)

	// VectorResplit changes a partitioned collection's partition count by building a
	// NEW generation of physical partitions, streaming every vector into it re-hashed
	// by PartitionOf(id, newP), atomically flipping the catalog to {newP, gen+1}, then
	// dropping the old generation. OFFLINE: the caller MUST quiesce writes first — a
	// concurrent write during resplit may land in the old generation and be lost.
	VectorResplit(ctx context.Context, collection string, newP int) error

	// VectorResplitCleanup drops physical partitions left behind by a failed resplit:
	// every partition whose generation is not the collection's current live generation.
	// Safe to call any time; idempotent; returns the number of partitions dropped.
	// Best-effort discovery via bounded probe (the system has no collection enumeration),
	// so an orphan generation with more than the probe bound of partitions, or a wide
	// internal gap from a partial drop, may leave a tail (benign storage leak; re-runnable).
	VectorResplitCleanup(ctx context.Context, collection string) (int, error)

	// VectorMVResplit changes a partitioned multi-vector collection's partition count
	// by building a NEW generation of physical partitions, streaming every document
	// into it re-hashed by PartitionOf(id, newP), atomically flipping the catalog to
	// {newP, gen+1}, then dropping the old generation. OFFLINE: the caller MUST quiesce
	// writes first — a concurrent write during resplit may land in the old generation
	// and be lost. Mirrors VectorResplit for the dense path.
	VectorMVResplit(ctx context.Context, collection string, newP int) error

	// VectorMVResplitCleanup drops physical partitions left behind by a failed MV
	// resplit: every partition whose generation is not the collection's current live
	// generation. Safe to call any time; idempotent; returns the number of partitions
	// dropped. Best-effort discovery via bounded probe (the system has no collection
	// enumeration), so an orphan generation with more than the probe bound of
	// partitions, or a wide internal gap from a partial drop, may leave a tail (benign
	// storage leak; re-runnable). Mirrors VectorResplitCleanup for the dense path.
	VectorMVResplitCleanup(ctx context.Context, collection string) (int, error)

	// VectorReshard repartitions a dense collection LIVE (online) — both reads AND
	// writes stay up for the entire operation; there is no quiesce and no recreate.
	// It builds a NEW generation (gen+1) of physical partitions and runs a
	// dual-write + background-copy state machine:
	//
	//   - While resharding, every user point write (insert/upsert/delete) is
	//     dual-written to BOTH the old (read source-of-truth) gen and the new gen,
	//     so the new gen converges to live data without losing concurrent writes.
	//   - The copy uses atomic insert-IF-ABSENT (never plain upsert) so it can never
	//     clobber a newer concurrent write (value-clobber race), and a per-record
	//     resurrection guard (re-check old, delete-from-new if gone) so it can never
	//     resurrect a concurrently-deleted id (delete-resurrection race).
	//   - CUTOVER is a single atomic catalog flip to {newP, gen+1}; reads move to
	//     the new gen there. That flip is the single POINT OF NO RETURN.
	//
	// Resumable: if the coordinator dies mid-reshard the collection is left in the
	// Resharding state (reads still served from the old gen, dual-write still on,
	// status durable). Re-invoking VectorReshard with the SAME newP resumes and
	// converges (the copy is idempotent); a different in-flight target is refused.
	//
	// Abort window: VectorReshardAbort is valid only BEFORE cutover (see below).
	//
	// Cost: dual-write doubles write amplification for the reshard's duration only.
	// The background copy streams (bounded memory) and is throttleable so it cannot
	// saturate the cluster. newP must be in [2, 65536] and != the current P; the
	// collection must already be partitioned (P>1). The offline VectorResplit
	// (quiesced bulk path) remains available and is unaffected.
	VectorReshard(ctx context.Context, collection string, newP int) error

	// VectorReshardAbort cancels an in-progress online reshard and restores the
	// collection to its old generation. It is valid ONLY before cutover — while the
	// live generation is still the old one. It clears the Resharding status (turning
	// off dual-write) and drops the new-gen partitions; the collection is fully
	// intact on the old gen because reads never left it and dual-writes to it were
	// the source of truth. After cutover the reshard is committed and abort returns
	// an error (run a new reshard to revert). Errors if no reshard is in progress.
	VectorReshardAbort(ctx context.Context, collection string) error

	// VectorMVReshard repartitions a multi-vector collection LIVE (online) — the
	// multi-vector mirror of VectorReshard with identical semantics. Both reads AND
	// writes stay up for the whole operation (no quiesce, no recreate); it builds a
	// NEW generation (gen+1) and runs the same dual-write + background-copy state
	// machine:
	//
	//   - While resharding, every MV point write (add/delete) is dual-written to
	//     BOTH the old (read source-of-truth) gen and the new gen, so the new gen
	//     converges to live data without losing concurrent writes.
	//   - The copy uses atomic mv-add-IF-ABSENT (never plain add) so it can never
	//     clobber a newer concurrent write (value-clobber race), threading the FULL
	//     token matrix + metadata; a per-doc resurrection guard (re-check old,
	//     delete-from-new if gone) prevents resurrecting a concurrently-deleted doc.
	//   - CUTOVER is a single atomic catalog flip to {newP, gen+1}; that flip is the
	//     single POINT OF NO RETURN.
	//
	// Resumable: re-invoking with the SAME newP resumes a crashed reshard and
	// converges (the copy is idempotent); a different in-flight target is refused.
	// Abort window: VectorMVReshardAbort is valid only BEFORE cutover. Cost:
	// dual-write doubles write amplification for the reshard's duration only; the
	// copy streams (bounded memory) and is throttleable. newP must be in [2, 65536]
	// and != the current P; the collection must already be partitioned (P>1). The
	// offline VectorMVResplit (quiesced bulk path) remains available and unaffected.
	VectorMVReshard(ctx context.Context, collection string, newP int) error

	// VectorMVReshardAbort cancels an in-progress multi-vector online reshard and
	// restores the collection to its old generation — the MV mirror of
	// VectorReshardAbort. Valid ONLY before cutover (while the live generation is
	// still the old one): it clears the Resharding status (turning off dual-write)
	// and drops the new-gen partitions; the collection is fully intact on the old
	// gen because reads never left it and dual-writes to it were the source of truth.
	// After cutover the reshard is committed and abort returns an error (run a new
	// reshard to revert). Errors if no reshard is in progress.
	VectorMVReshardAbort(ctx context.Context, collection string) error

	// CreateAlias creates (or overwrites — upsert) an alias name that resolves to a
	// real (canonical) collection, so data-plane ops on the alias transparently
	// route to the target. Validation: the target collection must EXIST; the alias
	// name must not shadow an existing real collection nor contain reserved
	// '#'/'@' characters; the target must not itself be an alias (one level only).
	// Alias management is a coordinator op (meta-Raft metadata, NOT shard-routed).
	CreateAlias(ctx context.Context, alias, collection string) error

	// DeleteAlias removes an alias. An absent alias is a no-op.
	DeleteAlias(ctx context.Context, alias string) error

	// AliasBatch atomically applies a batch of alias mutations (create/delete) in
	// ONE meta-Raft log entry — an atomic swap {delete prod, create prod→v2}
	// repoints with no undefined window. The WHOLE batch is validated before
	// commit; any invalid create rejects the entire batch (nothing applied).
	AliasBatch(ctx context.Context, actions []AliasAction) error

	// ListAliases returns the alias→collection map (a local read, no consensus).
	// When collection != "" the result is filtered to aliases targeting it.
	ListAliases(ctx context.Context, collection string) (map[string]string, error)

	// VectorNamedCreateCollection registers a named-vector (Qdrant-style
	// multi-vector-space) collection: a MAP of named dense vector spaces, each its
	// own HNSW index, all sharing ONE per-point payload + point-id namespace. The
	// config maps each space name to its per-space index params. At least one space
	// is required; names must be non-empty and reserved-char-free. Named collections
	// are in-memory only (durable via Raft snapshot, not WAL). partitions is the
	// collection-level partition count (0 or 1 = single-partition; >1 splits the
	// collection across shards via cross-shard fan-out, like dense/MV — every
	// physical partition is a named collection with the same spaces config and a
	// point's id maps to exactly one partition).
	VectorNamedCreateCollection(ctx context.Context, name string, cfg map[string]NamedVectorParams, partitions int) error

	// VectorNamedDropCollection removes a named-vector collection (all sub-indexes
	// + the shared per-point store).
	VectorNamedDropCollection(ctx context.Context, name string) error

	// VectorNamedInsert upserts point id into a named-vector collection: a map of
	// named vectors (each name must be a configured space; each vec's length must
	// equal that space's Dim — fail loud on either), a SHARED per-point payload, and
	// a point-level ttl. A point may omit some configured spaces; re-inserting an id
	// replaces the vectors for the provided spaces + the payload + ttl.
	VectorNamedInsert(ctx context.Context, name string, id uint64, vectors map[string][]float32, payload VectorMetadata, ttl time.Duration, opts ...WriteOpts) error

	// VectorNamedSearch runs a filtered KNN search against the named space
	// (vectorName must be a configured space — fail loud). The optional filter is
	// predicate-evaluated against the SHARED per-point payload. Returns the top-k
	// point ids + distances. Back-compat convenience for VectorNamedSearchExt with
	// default (AnyReplica) consistency.
	VectorNamedSearch(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorResult, error)

	// VectorNamedSearchExt is VectorNamedSearch with read-consistency / partition
	// opts (opts.Filter carries the optional payload predicate). A Linearizable
	// read arms the meta + per-shard barriers; AnyReplica/LeaderOnly stay
	// zero-overhead.
	VectorNamedSearchExt(ctx context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorResult, error)

	// VectorNamedSearchDocs is VectorNamedSearch returning each hit enriched with
	// the SHARED per-point payload (the named spaces store no per-arena content).
	// Back-compat convenience for VectorNamedSearchDocsExt.
	VectorNamedSearchDocs(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorDocument, error)

	// VectorNamedSearchDocsExt is VectorNamedSearchDocs with read-consistency /
	// partition opts (opts.Filter carries the optional payload predicate).
	VectorNamedSearchDocsExt(ctx context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorDocument, error)

	// VectorNamedSparseSearch runs a sparse-dot-product top-k search against a
	// SPARSE named space (space must be a configured SPARSE space — fail loud;
	// ErrSpaceModalityMismatch for a dense space). The optional filter is
	// predicate-evaluated against the SHARED per-point payload. Returns the top-k
	// point ids + scores (descending by sparse dot product). Back-compat convenience
	// for VectorNamedSparseSearchExt with default (AnyReplica) consistency.
	VectorNamedSparseSearch(ctx context.Context, name, space string, query VectorSparse, k int, filter VectorFilter) ([]VectorResult, error)

	// VectorNamedSparseSearchExt is VectorNamedSparseSearch with read-consistency /
	// partition opts (opts.Filter carries the optional payload predicate). A
	// Linearizable read arms the meta + per-shard barriers.
	VectorNamedSparseSearchExt(ctx context.Context, name, space string, query VectorSparse, k int, opts NamedSearchOpts) ([]VectorResult, error)

	// VectorNamedHybridSearch fuses a DENSE named space and a SPARSE named space
	// into the top-k (cross-space hybrid). denseSpace must be a configured dense
	// space and sparseSpace a configured sparse space (fail loud:
	// ErrSpaceModalityMismatch / ErrUnknownVectorName). An empty dense query degrades
	// to the sparse lane only; an empty sparse query degrades to the dense lane only
	// (mirror the dense hybrid). opts carries the fusion method/params, the optional
	// shared-payload filter (applied to BOTH lanes), and read-consistency / partition
	// opts. A Linearizable read arms the meta + per-shard barriers. Returns the fused
	// top-k (id + dense distance + fusion score).
	VectorNamedHybridSearch(ctx context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, error)

	// VectorNamedHybridSearchExt is VectorNamedHybridSearch with the cross-partition
	// fan-out completeness exposed as FanMeta (mirroring VectorHybridSearch). Under
	// the default OnPartitionUnavailable=Partial an unreachable partition yields
	// FanMeta{Degraded:true, Missing:...} instead of a silently incomplete top-k; a
	// single-node/unpartitioned backend never fans out and reports a zero FanMeta.
	VectorNamedHybridSearchExt(ctx context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, FanMeta, error)

	// VectorNamedQuery runs the unified Query API (vector_named_query) against a
	// NAMED collection: a root leaf plus N single-level prefetch leaves where EVERY
	// leaf targets a configured named SPACE (dense or sparse), combined by FUSION
	// (RRF/Weighted/DBSF over the prefetch lanes — N>2 multi-space fusion is the
	// distinctive named-family value) or RERANK (the root re-scores the union of the
	// prefetch candidates). specBytes is the marshaled pb.QuerySpec carried on the
	// wire; spec is the decoded engine spec the coordinator uses for the fan-out
	// fusion/rerank merge. Returns the final top-k + FanMeta (cross-shard
	// completeness; see VectorSearchExt). The named-family analogue of VectorQuery.
	VectorNamedQuery(ctx context.Context, name string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error)

	// VectorNamedDelete removes point id from EVERY named space + the shared
	// payload/ttl, returning whether it existed.
	VectorNamedDelete(ctx context.Context, name string, id uint64, opts ...WriteOpts) (bool, error)

	// VectorNamedScroll lists live points (+ shared payload) matching filter (zero
	// filter = all), up to limit (0 = no cap). Payload-only (no vectors).
	//
	// Cursor pagination (resume-after-id): cursor is the opaque token from the
	// previous page (empty = first page); the returned nextCursor is the token for
	// the next page (empty = exhausted). Scroll is deterministic id-ASCENDING
	// globally. A malformed cursor surfaces ops.ErrBadScrollCursor. Back-compat
	// convenience for VectorNamedScrollExt with default consistency.
	VectorNamedScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) (docs []VectorDocument, nextCursor string, err error)

	// VectorNamedScrollExt is VectorNamedScroll with read-consistency / partition
	// opts. The cursor stays its own parameter.
	VectorNamedScrollExt(ctx context.Context, name string, filter VectorFilter, limit int, cursor string, opts NamedScrollOpts) (docs []VectorDocument, nextCursor string, err error)

	// VectorNamedGetConfig returns the configured named spaces of a named-vector
	// collection (the introspection accessor).
	VectorNamedGetConfig(ctx context.Context, name string) (map[string]NamedVectorParams, error)

	// VectorNamedGetConfigExt is VectorNamedGetConfig with read-consistency opts. A
	// Linearizable read arms the meta-catalog read barrier (resolveCollectionForRead)
	// so the returned config reflects a just-created / just-reconfigured collection,
	// and routes the catalog read to the owning shard leader. The zero ReadOpts is
	// AnyReplica — behaviour-identical to VectorNamedGetConfig.
	VectorNamedGetConfigExt(ctx context.Context, name string, opts ReadOpts) (map[string]NamedVectorParams, error)

	// VectorGet retrieves a dense point by id: its (cosine-normalized, if the
	// metric is cosine) vector, payload, remaining TTL, and sparse lane. withVector
	// / withPayload gate the vector and the payload+sparse projections (pass both
	// true for the common "fetch everything" case). found is false for an
	// absent/tombstoned/TTL-expired point — a not-found FLAG, NOT an error, so a
	// point-op routed to one partition treats "not here" as expected.
	VectorGet(ctx context.Context, collection string, id uint64, withVector, withPayload bool) (found bool, vec []float32, meta VectorMetadata, ttl time.Duration, sparse *VectorSparse, err error)

	// VectorGetExt is VectorGet with read-consistency opts. A Linearizable read
	// (opts.ReadConsistency == 2) routes the single-id point-get to the owning
	// partition's Raft leader and arms the shard readIndex barrier (read-your-writes).
	// The zero ReadOpts is AnyReplica — byte/behaviour-identical to VectorGet.
	VectorGetExt(ctx context.Context, collection string, id uint64, withVector, withPayload bool, opts ReadOpts) (found bool, vec []float32, meta VectorMetadata, ttl time.Duration, sparse *VectorSparse, err error)

	// VectorGetBatch retrieves MANY dense points by id in ONE op. It returns the
	// PRESENT points (each carrying its id + the with_vector / with_payload
	// projection) plus the missing ids (absent / tombstoned / TTL-expired). A
	// partial miss is NORMAL, never an error — mirroring single VectorGet's
	// not-found FLAG. On a partitioned collection the ids are grouped by their
	// owning partition and each partition is asked ONLY for its owned subset
	// (concurrently), then the results are merged. Duplicate ids are deduped (a
	// repeated id is fetched once and appears once). points and missing are both
	// sorted ascending by id (deterministic). An empty ids list yields empty
	// points + empty missing. Like VectorGet this is AnyReplica (no read
	// consistency). An unreachable partition fails the whole batch (fail-loud).
	VectorGetBatch(ctx context.Context, collection string, ids []uint64, withVector, withPayload bool) (points []BatchGetPoint, missing []uint64, err error)

	// VectorSetPayload merges patch into the point's existing payload (patch keys
	// overwrite/add, other keys retained), reindexing the dense payload index and
	// WAL-logging the result. Does NOT change the vector or TTL. applied is false
	// (NOT an error) when the point is absent/tombstoned/expired. keyTTLMs is an
	// optional per-key payload TTL map (key -> RELATIVE ms; the engine computes the
	// absolute deadline); nil/empty = no per-key TTL.
	VectorSetPayload(ctx context.Context, collection string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (applied bool, err error)

	// VectorOverwritePayload replaces the point's entire payload with meta (nil =
	// clear). applied is false (not an error) for an absent point. keyTTLMs sets the
	// per-key payload TTL for the new payload (relative ms; engine computes the
	// absolute deadline); nil/empty = no per-key TTL.
	VectorOverwritePayload(ctx context.Context, collection string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (applied bool, err error)

	// VectorDeletePayloadKeys removes the listed keys from the point's payload
	// (absent keys = no-op). applied is false (not an error) for an absent point.
	VectorDeletePayloadKeys(ctx context.Context, collection string, id uint64, keys []string, opts ...WriteOpts) (applied bool, err error)

	// VectorClearPayload removes the point's entire payload. applied is false (not
	// an error) for an absent point.
	VectorClearPayload(ctx context.Context, collection string, id uint64, opts ...WriteOpts) (applied bool, err error)

	// VectorNamedGet retrieves a named-vector point by id: its per-space vectors
	// (map[name][]float32; omitted spaces absent), shared payload, and remaining
	// TTL. found is false (not an error) for an absent/expired point. See VectorGet.
	VectorNamedGet(ctx context.Context, name string, id uint64, withVector, withPayload bool) (found bool, vectors map[string][]float32, payload VectorMetadata, ttl time.Duration, err error)

	// VectorNamedGetExt is VectorNamedGet with read-consistency opts. A
	// Linearizable read routes to the owning partition's leader and arms the shard
	// readIndex barrier. The zero ReadOpts is byte/behaviour-identical to
	// VectorNamedGet.
	VectorNamedGetExt(ctx context.Context, name string, id uint64, withVector, withPayload bool, opts ReadOpts) (found bool, vectors map[string][]float32, payload VectorMetadata, ttl time.Duration, err error)

	// VectorNamedGetBatch retrieves MANY named-vector points by id in ONE op. It
	// returns the PRESENT points (each carrying its id + the per-space vectors map +
	// shared payload + remaining TTL, gated by with_vector / with_payload) plus the
	// missing ids (absent / expired). A partial miss is NORMAL, never an error —
	// mirroring single VectorNamedGet's not-found FLAG. On a partitioned collection
	// the ids are grouped by their owning partition and each partition is asked ONLY
	// for its owned subset (concurrently), then merged. Duplicate ids are deduped.
	// points and missing are both sorted ascending by id (deterministic). An empty
	// ids list yields empty points + missing. Like VectorNamedGet this is AnyReplica
	// (no read consistency). An unreachable partition fails the batch (fail-loud).
	// The named clone of VectorGetBatch.
	VectorNamedGetBatch(ctx context.Context, collection string, ids []uint64, withVector, withPayload bool) (points []NamedBatchGetPoint, missing []uint64, err error)

	// VectorNamedSetPayload merges patch into id's SHARED payload (no reindex — the
	// named family has no payload index). keyTTLMs is an optional per-key payload
	// TTL map (key -> RELATIVE ms; the engine computes the absolute deadline, stored
	// in the named snapshot); nil/empty = no per-key TTL. applied false (not an
	// error) for absent.
	VectorNamedSetPayload(ctx context.Context, name string, id uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (applied bool, err error)

	// VectorNamedOverwritePayload replaces id's entire shared payload. keyTTLMs sets
	// the per-key payload TTL for the new payload (relative ms; engine computes the
	// absolute deadline); nil/empty = no per-key TTL. applied false (not an error)
	// for absent.
	VectorNamedOverwritePayload(ctx context.Context, name string, id uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (applied bool, err error)

	// VectorNamedDeletePayloadKeys removes the listed keys from id's shared payload.
	// applied false (not an error) for absent.
	VectorNamedDeletePayloadKeys(ctx context.Context, name string, id uint64, keys []string, opts ...WriteOpts) (applied bool, err error)

	// VectorNamedClearPayload removes id's entire shared payload. applied false (not
	// an error) for absent.
	VectorNamedClearPayload(ctx context.Context, name string, id uint64, opts ...WriteOpts) (applied bool, err error)

	// VectorMVGet retrieves a multi-vector document by id: its token matrix
	// ([][]float32) and payload. found is false (not an error) for an absent
	// document (the MV index has no tombstones/TTL). See VectorGet.
	VectorMVGet(ctx context.Context, name string, docID uint64, withVector, withPayload bool) (found bool, tokens [][]float32, payload VectorMetadata, err error)

	// VectorMVGetExt is VectorMVGet with read-consistency opts. A Linearizable read
	// routes to the owning partition's leader and arms the shard readIndex barrier.
	// The zero ReadOpts is byte/behaviour-identical to VectorMVGet.
	VectorMVGetExt(ctx context.Context, name string, docID uint64, withVector, withPayload bool, opts ReadOpts) (found bool, tokens [][]float32, payload VectorMetadata, err error)

	// VectorMVGetBatch retrieves MANY multi-vector documents by id in ONE op. It
	// returns the PRESENT points (each carrying its id + the token matrix +
	// payload, gated by with_vector / with_payload) plus the missing ids (absent).
	// A partial miss is NORMAL, never an error — mirroring single VectorMVGet's
	// not-found FLAG. On a partitioned collection the ids are grouped by their
	// owning partition and each partition is asked ONLY for its owned subset
	// (concurrently), then merged. Duplicate ids are deduped. points and missing
	// are both sorted ascending by id (deterministic). An empty ids list yields
	// empty points + missing. Like VectorMVGet this is AnyReplica (no read
	// consistency). An unreachable partition fails the batch (fail-loud). MV has NO
	// ttl. The MV clone of VectorNamedGetBatch.
	VectorMVGetBatch(ctx context.Context, collection string, ids []uint64, withVector, withPayload bool) (points []MVBatchGetPoint, missing []uint64, err error)

	// VectorMVScroll lists live multi-vector documents (id + payload, no token
	// vectors) matching filter (zero filter = all), up to limit (0 = no cap). A
	// query-less listing primitive for the MV family (the dense/named scroll
	// mirror; vector_mv_get is the path for token matrices). The returned FanMeta
	// reports cross-shard fan-out completeness (see VectorScroll).
	//
	// Cursor pagination (resume-after-id): cursor is the opaque token from the
	// previous page (empty = first page); the returned nextCursor is the token for
	// the next page (empty = exhausted). Scroll is deterministic id-ASCENDING
	// globally; a no-cursor limit-capped scroll returns the smallest-id `limit`
	// documents. A malformed cursor surfaces ops.ErrBadScrollCursor. Back-compat
	// convenience for VectorMVScrollExt with default consistency.
	VectorMVScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) (docs []VectorDocument, meta FanMeta, nextCursor string, err error)

	// VectorMVScrollExt is VectorMVScroll with read-consistency / partition opts. A
	// Linearizable scroll arms the meta readIndex barrier on the coordinator and
	// the per-shard data barrier on every partition; rc rides every per-partition
	// arg. The cursor stays its own parameter.
	VectorMVScrollExt(ctx context.Context, name string, filter VectorFilter, limit int, cursor string, opts MVScrollOpts) (docs []VectorDocument, meta FanMeta, nextCursor string, err error)

	// VectorMVSetPayload merges patch into docID's payload (no reindex). keyTTLMs is
	// an optional per-key payload TTL map (key -> RELATIVE ms; the engine computes
	// the absolute deadline, stored in the MV snapshot); nil/empty = no per-key TTL.
	// applied false (not an error) for an absent document.
	VectorMVSetPayload(ctx context.Context, name string, docID uint64, patch VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (applied bool, err error)

	// VectorMVOverwritePayload replaces docID's entire payload. keyTTLMs sets the
	// per-key payload TTL for the new payload (relative ms; engine computes the
	// absolute deadline); nil/empty = no per-key TTL. applied false (not an error)
	// for an absent document.
	VectorMVOverwritePayload(ctx context.Context, name string, docID uint64, meta VectorMetadata, keyTTLMs map[string]int64, opts ...WriteOpts) (applied bool, err error)

	// VectorMVDeletePayloadKeys removes the listed keys from docID's payload. applied
	// false (not an error) for an absent document.
	VectorMVDeletePayloadKeys(ctx context.Context, name string, docID uint64, keys []string, opts ...WriteOpts) (applied bool, err error)

	// VectorMVClearPayload removes docID's entire payload. applied false (not an
	// error) for an absent document.
	VectorMVClearPayload(ctx context.Context, name string, docID uint64, opts ...WriteOpts) (applied bool, err error)
}

// getFlags builds the get-op projection flags byte from the with_vector /
// with_payload booleans. The bit layout MIRRORS ops' getFlagWithVector (bit0) /
// getFlagWithPayload (bit1); the ops package decodes it with the same bits. Both
// false yields a found-only probe (no projections).
func getFlags(withVector, withPayload bool) uint8 {
	var f uint8
	if withVector {
		f |= 1 << 0
	}
	if withPayload {
		f |= 1 << 1
	}
	return f
}

// FanMeta reports cross-shard fan-out completeness for a partitioned read. On a
// single-partition or non-clustered backend it is the zero value (not degraded).
type FanMeta struct {
	Degraded bool  // true when some partitions were unreachable in Partial mode — results are incomplete
	Missing  []int // partition indices skipped (sorted); nil if none
}

// ErrNotFound is returned when a key is absent or expired.
var ErrNotFound = errors.New("rostam: not found")

// ErrNotLeader is returned when a write reaches a non-leader and
// (Networked mode only) the retry budget is exhausted.
var ErrNotLeader = errors.New("rostam: not leader")

// ErrPartitionedUnsupported is the loud-fail sentinel for an operation that is
// not wired for cross-shard fan-out being invoked on a partitioned collection
// (Partitions>1): on such a collection the data lives in the physical partition
// collections and the logical name is empty, so routing the op by the logical
// name would silently return zero results (or mutate nothing). No op currently
// returns this — every vector op now fans out across partitions — so it is
// reserved for future partition-unsupported ops. Unpartitioned collections
// (Partitions<=1) are unaffected.
var ErrPartitionedUnsupported = errors.New("rostam: operation not supported on a partitioned collection (Partitions>1)")

// Peer describes one node in the cluster membership list. Exported so
// callers do not have to import the internal cluster package.
type Peer struct {
	NodeID     string // unique identifier in the Raft cluster
	RaftAddr   string // host:port for inter-node Raft transport
	ServerAddr string // host:port for the TCP server (used by Client mode)

	// PBAddr is this node's primary-backup (pbisr) NetTransport listen
	// endpoint, e.g. "10.0.0.1:7200". Required only when
	// EmbeddedConfig.ReplicationMode is "pb"; left empty in "raft" mode
	// (unused there). Mirrors cluster.Peer.PBAddr.
	PBAddr string
}

// WASMRegistration is a re-export of ops.WASMRegistration so callers do not
// have to import the internal ops package.
type WASMRegistration = ops.WASMRegistration

// VectorResult is one entry in a VectorSearch result list. Alias of vector.Result.
type VectorResult = vector.Result

// BatchGetPoint is one present point returned by VectorGetBatch: its id plus the
// SAME projected fields a single VectorGet carries (vector, payload, remaining
// TTL, sparse lane). Vec/Meta/Sparse follow the with_vector / with_payload
// projection requested at fetch time. Unlike VectorDocument (a search-hit alias
// with Distance/Score/Content) this carries the raw point projection and its id —
// a batch caller must know which id each point belongs to. Absent ids are NOT
// represented here; they appear in VectorGetBatch's separate missing slice.
type BatchGetPoint struct {
	ID      uint64
	Vec     []float32
	Meta    VectorMetadata
	TTL     time.Duration
	Sparse  *VectorSparse
	Version uint64 // per-point CAS version (>=1); 0 on a backend that does not carry it
}

// NamedBatchGetPoint is one present point returned by VectorNamedGetBatch: its id
// plus the SAME projected fields a single VectorNamedGet carries (the per-space
// vectors map, shared payload, remaining TTL). Vectors/Meta follow the with_vector
// / with_payload projection requested at fetch time. Absent ids are NOT
// represented here; they appear in VectorNamedGetBatch's separate missing slice.
// The named clone of BatchGetPoint (a vectors MAP + ttl, no sparse lane).
type NamedBatchGetPoint struct {
	ID      uint64
	Vectors map[string][]float32
	Meta    VectorMetadata
	TTL     time.Duration
	Version uint64 // per-point CAS version (>=1); 0 on a backend that does not carry it
}

// MVBatchGetPoint is one present point returned by VectorMVGetBatch: its id plus
// the SAME projected fields a single VectorMVGet carries (the token matrix +
// payload). Tokens/Meta follow the with_vector / with_payload projection
// requested at fetch time. Absent ids are NOT represented here; they appear in
// VectorMVGetBatch's separate missing slice. The MV clone of NamedBatchGetPoint
// (a token matrix, NO ttl and no sparse lane).
type MVBatchGetPoint struct {
	ID      uint64
	Tokens  [][]float32
	Meta    VectorMetadata
	Version uint64 // per-document CAS version (>=1); 0 on a backend that does not carry it
}

// VectorDocument is a SearchDocs hit: a result enriched with its stored content
// and metadata. Alias of vector.Document.
type VectorDocument = vector.Document

// VectorGroup is one group of a group-by-document search: a shared key and its
// best hits. Alias of vector.Group.
type VectorGroup = vector.Group

// VectorGroupOpts configures a group-by-document search. Alias of vector.GroupOpts.
type VectorGroupOpts = vector.GroupOpts

// MultiVectorConfig configures a late-interaction collection. Alias of
// vector.MultiVectorConfig.
type MultiVectorConfig = vector.MultiVectorConfig

// MultiSearchOpts tunes a multi-vector search. Alias of vector.MultiSearchOpts.
type MultiSearchOpts = vector.MultiSearchOpts

// MultiResult is one scored document from a multi-vector search. Alias of
// vector.MultiResult.
type MultiResult = vector.MultiResult

// NamedVectorParams is the per-named-space index configuration of a named-vector
// collection. Alias of vector.NamedVectorParams.
type NamedVectorParams = vector.NamedVectorParams

// VectorConfig configures a vector collection. Alias of vector.Config.
type VectorConfig = vector.Config

// VectorMetadata is per-vector attribute data. Alias of vector.Metadata.
type VectorMetadata = vector.Metadata

// VectorFilter is a metadata predicate tree for filtered search. Alias of vector.Filter.
type VectorFilter = vector.Filter

// VectorSparse is a sparse vector for hybrid search. Alias of vector.SparseVector.
type VectorSparse = vector.SparseVector

// FusionMethod selects dense/sparse fusion in a hybrid search. Alias of vector.FusionMethod.
type FusionMethod = vector.FusionMethod

// Fusion method constants, re-exported from the vector package.
const (
	FusionRRF      = vector.FusionRRF
	FusionWeighted = vector.FusionWeighted
	FusionDBSF     = vector.FusionDBSF
)

// WriteOpts carries the tunable write-consistency knobs shared by every
// data-plane write (insert/upsert, delete-by-id, payload mutations, etc.). It is
// embedded into the per-op opts structs so the zero value (WriteConsistencyFactor==0,
// Wait==nil) means "today's behavior" — no __wc__ envelope is ever built and no
// barrier engages (see wcActive).
//
// WriteConsistencyFactor is the number of replicas of the target shard's Raft
// group that must have applied the write before it returns success. It is
// clamped downstream to [1, RF]; 0 (unset) means "majority" — the Raft floor —
// i.e. byte-for-byte today's behavior with no barrier.
//
// Wait is a tri-state bool: nil = default true (block until the factor is met);
// non-nil false = explicit wait=false (return at majority, skip the >majority
// barrier — a latency knob, NOT fire-and-forget, since Raft majority is the
// durability floor); non-nil true = explicit wait=true.
type WriteOpts struct {
	WriteConsistencyFactor uint8 // 0 = majority (default, no barrier); >0 = explicit factor (clamped to [1,RF])
	Wait                   *bool // nil = default true; &false = explicit no-barrier; &true = explicit wait

	// ExpectedVersion is the optimistic-CAS precondition: the per-point version the
	// caller expects the point to currently have. When non-nil the write applies
	// ONLY when it matches (0 = expect the point to be absent/new); a mismatch
	// returns vector.ErrVersionConflict with no mutation. nil (the default) = an
	// unconditional write (byte-identical to the pre-CAS wire). Honored by the
	// delete + payload-mutation write paths (insert/upsert carry it via
	// VectorInsertOpts.ExpectedVersion).
	ExpectedVersion *uint64

	// KeyTTLMs is an OPTIONAL per-key payload TTL map (payload key -> RELATIVE ms)
	// for the named-insert / MV-add write paths (dense insert/upsert carry it via
	// VectorInsertOpts.KeyTTLMs instead). At insert/add the engine computes the
	// ABSOLUTE deadline now+ttl for each key (mirroring set_payload) and lazily drops
	// the key once its deadline passes, while the point/document lives on. nil/empty =
	// no per-key TTL (the zero-overhead, byte-identical wire path).
	KeyTTLMs map[string]int64

	// Sparse is an OPTIONAL doc-level sparse vector for the MV-add write path: each
	// multi-vector document MAY carry one doc-level sparse vector alongside its dense
	// token matrix (the MV analogue of a named point's sparse space; consumed by the
	// MV hybrid search in a later task). nil/zero = dense-only (the zero-overhead,
	// byte-identical wire path — no add-wire trailer, no persist block). Ignored by
	// the other write paths.
	Sparse *VectorSparse
}

// expectedVersion resolves the CAS precondition to (value, has): nil → (0, false)
// (an unconditional write); non-nil → (*ExpectedVersion, true).
func (w WriteOpts) expectedVersion() (uint64, bool) {
	if w.ExpectedVersion == nil {
		return 0, false
	}
	return *w.ExpectedVersion, true
}

// wcActive reports whether these opts request anything beyond the default
// behavior. When false, no __wc__ envelope is built and the barrier never
// engages, so the write path is byte-for-byte unchanged. Unexported: only the
// root package consumes it today. If a later task needs it from another
// package, promote to an exported WCActive.
func (w WriteOpts) wcActive() bool {
	return w.WriteConsistencyFactor > 0 || (w.Wait != nil && !*w.Wait)
}

// waitValue resolves the tri-state Wait to a concrete bool: nil defaults to true.
func (w WriteOpts) waitValue() bool {
	if w.Wait == nil {
		return true
	}
	return *w.Wait
}

// VectorInsertOpts carries optional per-insert settings.
type VectorInsertOpts struct {
	TTL      time.Duration  // 0 = no expiry
	Metadata VectorMetadata // nil = no metadata
	Sparse   VectorSparse   // zero = no sparse lane

	// KeyTTLMs is an OPTIONAL per-key payload TTL map (payload key -> RELATIVE
	// ms). At insert/upsert the engine computes the ABSOLUTE deadline now+ttl for
	// each key (mirroring set_payload) and lazily drops the key once its deadline
	// passes, while the point itself lives on. nil/empty = no per-key TTL (the
	// zero-overhead, byte-identical wire path).
	KeyTTLMs map[string]int64

	// WriteOpts carries the write-consistency knobs (WriteConsistencyFactor,
	// Wait). Embedded so the zero value preserves today's behavior. The same
	// knobs will be added to the delete/payload write paths in a later task;
	// those methods do not yet take an opts struct, so there is nothing else to
	// embed into today.
	WriteOpts
}

// ReadOpts carries read-consistency settings for the point-get + get_config
// reads (VectorGetExt / VectorNamedGetExt / VectorMVGetExt and the *GetConfigExt
// variants). It mirrors the ReadConsistency / OnPartitionUnavailable knobs the
// search/scroll opts carry, so a Linearizable point-get arms the shard readIndex
// barrier (and a Linearizable get_config also arms the meta-catalog barrier),
// symmetric with search/scroll. The zero value is AnyReplica / Partial — the
// legacy behaviour, so the non-Ext methods delegate with a zero ReadOpts and stay
// byte/behaviour-identical.
type ReadOpts struct {
	// ReadConsistency controls which replicas may serve the read.
	// 0 = AnyReplica (default, fastest); 1 = LeaderOnly; 2 = Linearizable
	// (readIndex barrier — read-your-writes).
	ReadConsistency uint8

	// OnPartitionUnavailable controls cross-shard fan-out behaviour when a shard
	// partition is unreachable. 0 = Partial (default); 1 = Fail. A single-id get
	// routes to ONE partition, so this matters only on the get_config catalog read
	// fan-out; it is threaded for symmetry with the search opts.
	OnPartitionUnavailable uint8

	// MaxStaleness is the max raft-entry lag the serving replica may have behind
	// the leader's committed frontier, in effect ONLY when ReadConsistency==3
	// (BoundedStaleness). Zero is a valid bound (serve only a fully-caught-up
	// replica). Ignored for every other ReadConsistency level.
	MaxStaleness uint64
}

// VectorSearchOpts carries optional per-search settings.
type VectorSearchOpts struct {
	Filter VectorFilter // zero = no filter

	// ReadConsistency controls which replicas may serve the query.
	// 0 = AnyReplica (default, fastest); 1 = LeaderOnly (best-effort, no barrier);
	// 2 = Linearizable (readIndex barrier — read-your-writes); 3 = BoundedStaleness
	// (any-replica read within MaxStaleness raft entries of the leader).
	// Applies only to the clustered backend when Partitions > 1; ignored otherwise.
	ReadConsistency uint8

	// OnPartitionUnavailable controls what happens when a shard partition is
	// unreachable during a cross-shard fan-out.
	// 0 = Partial (default, return results from available shards);
	// 1 = Fail (return an error if any partition is unavailable).
	// Applies only to the clustered backend when Partitions > 1; ignored otherwise.
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64

	// GlobalIDF opts into the BM25 global-DF (dfs_query_then_fetch) two-phase text
	// search: a partitioned (P>1) VectorSearchText first gathers + sums per-shard
	// corpus stats (n/df/avgdl) into GLOBAL stats, then re-scores each shard with the
	// SAME IDF so the merged top-k is the EXACT global ranking (bit-identical to a
	// single-node corpus). Costs one extra round-trip. Default false ⇒ the per-shard-
	// local fast path (today's behavior), byte-identical wire. Ignored for an
	// unpartitioned collection (local corpus IS global) and outside the clustered
	// backend.
	GlobalIDF bool
}

// NamedSearchOpts carries optional per-search settings for a named-vector KNN
// search (VectorNamedSearchExt / VectorNamedSearchDocsExt). Mirrors
// VectorSearchOpts / MultiSearchOpts.
type NamedSearchOpts struct {
	Filter VectorFilter // zero = no filter (predicate over the shared payload)

	// ReadConsistency controls which replicas may serve the query.
	// 0 = AnyReplica (default, fastest); 1 = LeaderOnly; 2 = Linearizable.
	// Linearizable arms the meta readIndex barrier on the coordinator and the
	// per-shard data barrier on every partition. Ignored for a single-node
	// engine / unpartitioned collection.
	ReadConsistency uint8

	// OnPartitionUnavailable controls what happens when a shard partition is
	// unreachable during a cross-shard fan-out.
	// 0 = Partial (default, return results from available shards);
	// 1 = Fail (return an error if any partition is unavailable).
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64
}

// NamedHybridOpts carries the settings for a named cross-space hybrid search
// (VectorNamedHybridSearch): one dense named space fused with one sparse named
// space. It is the named-family analogue of VectorHybridOpts (the dense single-
// vector hybrid), minus the sparse query (which is a top-level arg here, alongside
// the dense query and the two space names).
type NamedHybridOpts struct {
	Filter  VectorFilter // shared-payload predicate applied to BOTH lanes; zero = no filter
	Method  FusionMethod // FusionRRF (default), FusionWeighted, or FusionDBSF
	Alpha   float64      // weighted only: dense weight in [0,1] (0 → 0.5 default)
	RRFK    int          // RRF constant; 0 = default 60
	DenseK  int          // dense-lane candidate pool; 0 = max(k, 50)
	SparseK int          // sparse-lane candidate pool; 0 = max(k, 50)

	// ReadConsistency / OnPartitionUnavailable mirror NamedSearchOpts: 0 = AnyReplica
	// / Partial (default). Linearizable arms the meta + per-shard barriers; Fail errors
	// if any partition is unreachable during fan-out.
	ReadConsistency        uint8
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64
}

// toNamedHybridVectorOpts maps the public NamedHybridOpts onto the internal
// vector.HybridOpts the ops codec / engine expect (the rc/opa pair rides
// separately on the wire, like the dense hybrid).
func toNamedHybridVectorOpts(o NamedHybridOpts) vector.HybridOpts {
	return vector.HybridOpts{
		Filter:  o.Filter,
		Method:  o.Method,
		Alpha:   o.Alpha,
		RRFK:    o.RRFK,
		DenseK:  o.DenseK,
		SparseK: o.SparseK,
	}
}

// MVHybridOpts carries the settings for an MV cross-modality hybrid search
// (VectorMVHybridSearch): the per-doc MaxSim (late-interaction dense) lane fused
// with the per-doc sparse lane. The MV-family analogue of NamedHybridOpts; the MV
// token query matrix and the sparse query ride as top-level args. DenseK sizes the
// MaxSim lane, SparseK the sparse lane.
type MVHybridOpts struct {
	Filter  VectorFilter // shared-payload predicate applied to BOTH lanes; zero = no filter
	Method  FusionMethod // FusionRRF (default), FusionWeighted, or FusionDBSF
	Alpha   float64      // weighted only: MaxSim-lane weight in [0,1] (0 → 0.5 default)
	RRFK    int          // RRF constant; 0 = default 60
	DenseK  int          // MaxSim-lane candidate pool; 0 = max(k, 50)
	SparseK int          // sparse-lane candidate pool; 0 = max(k, 50)

	// ReadConsistency / OnPartitionUnavailable mirror MultiSearchOpts: 0 = AnyReplica
	// / Partial (default). Linearizable arms the meta + per-shard barriers; Fail errors
	// if any partition is unreachable during fan-out.
	ReadConsistency        uint8
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64
}

// toMVHybridVectorOpts maps the public MVHybridOpts onto the internal
// vector.HybridOpts the ops codec / engine expect (the rc/opa pair rides separately
// on the wire, like the named/dense hybrid).
func toMVHybridVectorOpts(o MVHybridOpts) vector.HybridOpts {
	return vector.HybridOpts{
		Filter:  o.Filter,
		Method:  o.Method,
		Alpha:   o.Alpha,
		RRFK:    o.RRFK,
		DenseK:  o.DenseK,
		SparseK: o.SparseK,
	}
}

// NamedScrollOpts carries optional per-scroll settings for a named-vector scroll
// (VectorNamedScrollExt). The cursor stays its own VectorNamedScrollExt
// parameter (mirrors the scroll signature); only the consistency knobs live
// here.
type NamedScrollOpts struct {
	// ReadConsistency controls which replicas may serve the scroll.
	// 0 = AnyReplica (default); 1 = LeaderOnly; 2 = Linearizable.
	ReadConsistency uint8

	// OnPartitionUnavailable controls cross-shard fan-out behaviour when a
	// partition is unreachable. 0 = Partial (default); 1 = Fail.
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64

	// OrderBy, when non-nil, paginates the named scroll by an arbitrary NUMERIC or
	// DATETIME payload field (Qdrant-style order_by) instead of the default
	// id-ascending order: the result set is globally ordered by the field's
	// (value, id) total order (ASC or DESC), missing/non-numeric-field points are
	// EXCLUDED, and the cursor is then a v2 (value, id) resume token. nil = the
	// id-ascending scroll (zero-overhead, v1 cursor). Mirrors VectorScrollOpts.OrderBy.
	OrderBy *vector.OrderBy
}

// MVScrollOpts carries optional per-scroll settings for a multi-vector scroll
// (VectorMVScrollExt). The cursor stays its own VectorMVScrollExt parameter
// (mirrors the scroll signature); only the consistency knobs live here. Mirrors
// NamedScrollOpts / VectorScrollOpts.
type MVScrollOpts struct {
	// ReadConsistency controls which replicas may serve the scroll.
	// 0 = AnyReplica (default); 1 = LeaderOnly; 2 = Linearizable.
	ReadConsistency uint8

	// OnPartitionUnavailable controls cross-shard fan-out behaviour when a
	// partition is unreachable. 0 = Partial (default); 1 = Fail.
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64

	// OrderBy, when non-nil, paginates the MV scroll by an arbitrary NUMERIC or
	// DATETIME payload field (Qdrant-style order_by) instead of the default
	// id-ascending order: the result set is globally ordered by the field's
	// (value, id) total order (ASC or DESC), missing/non-numeric-field points are
	// EXCLUDED, and the cursor is then a v2 (value, id) resume token. nil = the
	// id-ascending scroll (zero-overhead, v1 cursor). Mirrors VectorScrollOpts.OrderBy.
	OrderBy *vector.OrderBy
}

// VectorHybridOpts carries the settings for a hybrid (dense + sparse) search.
type VectorHybridOpts struct {
	Sparse  VectorSparse // query sparse vector; zero = dense-only
	Filter  VectorFilter // metadata predicate; zero = no filter
	Method  FusionMethod // FusionRRF (default), FusionWeighted, or FusionDBSF
	Alpha   float64      // weighted only: dense weight in [0,1]
	RRFK    int          // RRF constant; 0 = default 60
	DenseK  int          // dense-lane candidate pool; 0 = max(k, 50)
	SparseK int          // sparse-lane candidate pool; 0 = max(k, 50)

	// ReadConsistency controls which replicas may serve the query.
	// 0 = AnyReplica (default, fastest); 1 = LeaderOnly (best-effort, no barrier);
	// 2 = Linearizable (readIndex barrier — read-your-writes); 3 = BoundedStaleness
	// (any-replica read within MaxStaleness raft entries of the leader).
	// Applies only to the clustered backend when Partitions > 1; ignored otherwise.
	ReadConsistency uint8

	// OnPartitionUnavailable controls what happens when a shard partition is
	// unreachable during a cross-shard fan-out.
	// 0 = Partial (default, return results from available shards);
	// 1 = Fail (return an error if any partition is unavailable).
	// Applies only to the clustered backend when Partitions > 1; ignored otherwise.
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64

	// GlobalIDF opts into the BM25 global-DF (dfs_query_then_fetch) two-phase text
	// lane: a partitioned (P>1) VectorHybridText gathers + sums per-shard corpus
	// stats into GLOBAL stats and re-scores the BM25 text lane with the SAME IDF so
	// the fused result matches a single-node corpus. The DENSE lane is unaffected.
	// Default false ⇒ the per-shard-local text lane (today's behavior), byte-
	// identical wire. Ignored for an unpartitioned collection.
	GlobalIDF bool
}

// toVectorHybridOpts maps the public VectorHybridOpts onto the internal
// vector.HybridOpts the ops codec expects.
func toVectorHybridOpts(o VectorHybridOpts) vector.HybridOpts {
	return vector.HybridOpts{
		Filter:  o.Filter,
		Method:  o.Method,
		Alpha:   o.Alpha,
		RRFK:    o.RRFK,
		DenseK:  o.DenseK,
		SparseK: o.SparseK,
	}
}

// VectorScrollOpts carries optional per-scroll settings. Scroll has no query
// tuning of its own; these are the cross-shard routing knobs.
type VectorScrollOpts struct {
	// Cursor is the opaque resume-after-id pagination token from the previous
	// page (ops.EncodeScrollCursor). Empty = the first page (no lower bound,
	// id 0 included). The next page returns ids strictly greater than the cursor's
	// id, globally id-ascending. A malformed cursor fails loud (ops.ErrBadScrollCursor).
	Cursor string

	// ReadConsistency controls which replicas may serve the query.
	// 0 = AnyReplica (default, fastest); 1 = LeaderOnly (best-effort, no barrier);
	// 2 = Linearizable (readIndex barrier — read-your-writes); 3 = BoundedStaleness
	// (any-replica read within MaxStaleness raft entries of the leader).
	// Applies only to the clustered backend when Partitions > 1; ignored otherwise.
	ReadConsistency uint8

	// OnPartitionUnavailable controls what happens when a shard partition is
	// unreachable during a cross-shard fan-out.
	// 0 = Partial (default, return results from available shards);
	// 1 = Fail (return an error if any partition is unavailable).
	// Applies only to the clustered backend when Partitions > 1; ignored otherwise.
	OnPartitionUnavailable uint8

	// MaxStaleness bounds replica lag (raft entries) behind the leader's
	// committed frontier; in effect ONLY when ReadConsistency==3 (BoundedStaleness).
	MaxStaleness uint64

	// OrderBy, when non-nil, paginates the scroll by an arbitrary NUMERIC or
	// DATETIME payload field (Qdrant-style order_by) instead of the default
	// id-ascending order: the result set is globally ordered by the field's
	// (value, id) total order (ASC or DESC), points whose order field is
	// missing/non-numeric are EXCLUDED, and Cursor is then a v2 (value, id) resume
	// token (ops.EncodeScrollCursorOrder). nil = today's id-ascending scroll
	// (zero-overhead, v1 cursor). A v2 cursor with no OrderBy — or a v1/mismatched
	// cursor with an OrderBy — is rejected loud (ops.ErrCursorOrderMismatch).
	OrderBy *vector.OrderBy
}

// CacheConfig mirrors the cache-layer knobs callers care about. Its
// fields map one-to-one onto cache.Config; the indirection keeps the
// public API stable when cache.Config grows internal fields.
type CacheConfig struct {
	// NumShardsPerNode controls how many independent Raft groups this
	// node hosts. Defaults to 64 when zero (matches cluster.Config).
	NumShardsPerNode int

	// Durable, when true, runs an msync ticker so dirty mmap
	// pages flush every MsyncIntervalMs. Requires NumShardsPerNode > 0
	// and a non-empty DataDir on EmbeddedConfig.
	Durable bool

	// Mlock pins the mmap region into RAM. Requires ulimit -l to cover
	// the total size; failure logs and continues without mlock.
	Mlock bool

	// MsyncIntervalMs is the flush interval for the Durable ticker.
	// Defaults to 100 when zero. Ignored when Durable is false.
	MsyncIntervalMs int

	// DisableColdCompaction turns OFF the live-only rewrite of each persistent
	// shard's pages file at open. Default false (compaction ON), which is what a
	// persistent shard needs: it is the only thing that reclaims the ghost page
	// bytes left behind by overwritten and expired keys. This is the operational
	// escape hatch if that rewrite ever misbehaves — see cache.Config's field.
	DisableColdCompaction bool

	// MaxMemoryBytes bounds TOTAL cache memory for this node across every
	// shard. Zero means derive it from the host (a fraction of system RAM);
	// see cachebudget.go. The per-shard cap and page size are derived from
	// this, so the bound no longer moves when NumShardsPerNode changes.
	//
	// This is an upper bound, not a reservation: pages are allocated lazily.
	// It matters because Put is append-only — the lock-free read path freezes
	// retired pages — so a write-heavy node climbs toward this cap even when
	// the live key set is small, and only starts recycling once it arrives.
	// Set it below what the host can spare, or the process dies before the
	// ring-buffer eviction it depends on ever runs.
	//
	// It bounds CACHE PAGES, not process RSS. Pages are the live heap and Go
	// lets the heap reach (1 + GOGC/100) x live before collecting, so plan for
	// RSS ~= MaxMemoryBytes * (1 + GOGC/100) + ~40 MB: ~2x at Go's default
	// GOGC=100, ~1.5x at GOGC=40 (both measured, flat over 56M writes). The
	// multiplier is GC headroom rather than engine overhead; set GOMEMLIMIT to
	// bound RSS independently of it.
	MaxMemoryBytes int64
}
