// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"time"

	"github.com/rostamlabs/rostam/shard"
)

// Stats is a cluster-level snapshot aggregating cache and Raft stats
// across every shard.
type Stats struct {
	// NumShards is the configured shard count.
	NumShards int

	// PerShard holds the underlying shard.Stats indexed by shard ID.
	PerShard []shard.Stats

	// WASMGate reports the route gate's state and its refusal counter.
	WASMGate WASMGateStats

	// WASMBlobPush reports the pre-registration module-blob push.
	WASMBlobPush WASMBlobPushStats

	// WASMBlock reports shard groups currently PARKED waiting for module bytes
	// (see WASMBlockStats). It is a sibling of WASMGate rather than a field of it
	// because the two answer different questions about different mechanisms: the
	// gate refuses to PROPOSE into a group whose log lacks a registration, while a
	// block is a group that cannot APPLY an entry its log already carries.
	WASMBlock WASMBlockStats

	// KVIndex reports this node's KV record index: what is installed, what is
	// usable, and what the derivation has cost.
	KVIndex KVIndexStats

	// WASMBlobRetire reports the blob retirement sweeper (see
	// WASMBlobRetireStats). Retention == 0 means retirement is OFF, which is the
	// default and the only configuration in which nothing can ever be removed.
	WASMBlobRetire WASMBlobRetireStats
}

// WASMBlobRetireStats makes WASM blob retirement observable — including, and
// especially, the fact that it is switched off.
//
// Retirement is the one WASM mechanism that DELETES something, and the failure
// it can cause (a replica that needed a retired version blocks until someone
// supplies the bytes) surfaces somewhere else entirely, in WASMBlockStats. So
// the two numbers an operator correlating those needs are how many files this
// node has removed and whether the sweeper is even running — neither of which is
// inferable from anything else.
type WASMBlobRetireStats struct {
	// Retention echoes Config.WASMBlobRetention. ZERO MEANS OFF: no sweeper
	// goroutine exists, and nothing has been or can be removed.
	Retention time.Duration

	// Sweeps counts retirement passes since process start. It is what separates
	// "off" from "on and finding nothing to do" — with Retention non-zero and
	// Sweeps flat, the sweeper is not running.
	Sweeps uint64

	// Retired counts blob FILES removed since process start. Correlate a rise
	// here with a later WASMBlock entry naming a fingerprint: that pairing is
	// what a retention window set too short looks like.
	Retired uint64

	// Pending is how many unreferenced blobs are currently waiting out their
	// window — the sweeper's backlog, and an upper bound on what the next few
	// sweeps can remove.
	Pending int64
}

// WASMBlobPushStats makes the pre-registration blob push observable.
//
// The push tolerates an unreachable member by design — refusing would let any
// node being restarted stop the cluster from registering a module — so a member
// that is persistently unreachable produces no error anywhere: every registration
// succeeds. The reply payload names it, but only to the caller of that one call;
// Skips is what makes the STANDING state visible from the server side, to an
// operator who was not holding the return value of any particular registration.
type WASMBlobPushStats struct {
	// Acks counts per-member push legs that were delivered and acked (the peer
	// verified the hash and compiled the module) since process start.
	Acks uint64

	// Skips counts per-member push legs skipped because the member could not be
	// reached. A steadily climbing Skips means some member is missing every
	// module's bytes and is relying entirely on fetching them on demand.
	Skips uint64
}

// WASMGateStats makes the WASM route gate observable.
//
// The gate deliberately trades a shard-wide halt for a client-visible retryable
// error (ErrWASMOpNotInThisGroup), which means a gate that never opens is a
// silent, permanent refusal of exactly the keys that route to the unproven
// group — indistinguishable from a client bug unless something reports it. That
// makes the error's visibility part of the design, not a nice-to-have.
type WASMGateStats struct {
	// Refusals counts Calls the gate has declined to propose since process
	// start, across all ops and groups.
	Refusals uint64

	// ProvenGroups maps each gated (replicated) op name to the sorted shard
	// groups whose Raft log this node knows carries its registration. An op
	// present here with a group MISSING is a wedged (op, group) pair — that is
	// the diagnostic. Freshly allocated per call.
	ProvenGroups map[string][]int
}

// KVIndexStats makes the KV record index observable.
//
// The index is DERIVED state — never snapshotted, never logged, never
// replicated — installed from the meta catalog by a per-node polling observer
// and filled by walking the local cache. That makes almost everything about it
// invisible from the outside: a definition can be committed cluster-wide and
// still be doing nothing on this node, and the two reasons for that (still
// backfilling, or rejected at install) look identical to a client, which just
// sees queries that do not use the index.
type KVIndexStats struct {
	// Definitions is how many distinct index definitions are installed on this
	// node. It can lag the meta catalog by up to one observe interval, and sits
	// BELOW it whenever Rejects is climbing.
	Definitions int

	// Ready is how many of those are ready on EVERY shard group this node hosts.
	// A definition ready on some groups and building on others is not ready:
	// answering from it would silently return a proper subset of the matches.
	Ready int

	// Backfills counts COMPLETED definition walks since process start.
	// BackfillKeys counts cache entries visited by walks, completed or not — a
	// walk aborted because its shard is being removed still contributes what it
	// read before it stopped, because the cost was paid. So the two do not move
	// together: a rise in BackfillKeys with Backfills flat is walks being
	// abandoned, which is worth seeing rather than hiding. Both climb on a restart
	// (the index is rebuilt from the cache every time) and on every new
	// definition.
	Backfills    uint64
	BackfillKeys uint64

	// Rejects counts reject EVENTS since process start: definitions the meta FSM
	// ACCEPTED that this node cannot build, one per offending definition per
	// observer pass. Monotonic, so it never goes backwards and a rate can be taken
	// from it — but it keeps climbing while a bad definition simply sits in the
	// catalog, which is why it is not the number to alert on.
	Rejects uint64

	// RejectedDefs is how many definitions the catalog holds RIGHT NOW that this
	// node cannot build — a gauge, recomputed on every pass. A non-zero value is a
	// standing misconfiguration (or a version skew): the definition exists
	// cluster-wide and does nothing here. It returns to zero when the definition is
	// fixed or removed, which is the signal an operator acts on.
	RejectedDefs int

	// VerifyMisses counts candidates whose live re-read MISSED — a posting for a
	// key the cache no longer holds. Postings are hints, so this is a cost (one
	// wasted lookup) and never a wrong answer; a rising rate means keys are
	// leaving the cache by a path that does not reach kvindex.Set.Drop.
	//
	// IT IS KEY STALENESS ONLY, not value staleness. A candidate whose key is
	// still live but whose VALUE no longer satisfies the filter is discarded by
	// the predicate and counted nowhere — so a write path that stopped
	// reindexing is invisible here, and this counter staying at zero is not
	// evidence that the postings agree with the data. A separate counter for
	// that is a follow-up.
	//
	// Monotonic across shard removal: RemoveShardOwner folds a departing group's
	// count into the node total before its index is dropped.
	VerifyMisses uint64

	// ReconcileDrops counts postings the reconciler removed because their key is
	// no longer live.
	ReconcileDrops uint64
}
