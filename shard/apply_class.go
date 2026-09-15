// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// applyClass classifies the outcome of a committed op-apply so the FSM (and the
// primary-backup Applier) can decide whether advancing the applied index is
// SAFE. This is a correctness/trust boundary: a Raft log entry (or a PB seq) is
// already committed by the time apply runs, so every replica applies the SAME
// bytes in the SAME order. The only question is whether a non-nil apply error
// means "every replica failed identically" (safe to advance — state stays in
// agreement) or "this replica failed for a reason peers may not share" (fatal —
// advancing would durably diverge this replica's state from the others while
// still reporting the entry as applied).
//
// Why default-advance is the right bias (INCLUSION list, not exclusion):
// Deterministic apply errors are a pure function of the committed entry bytes and
// the op HANDLER, which are identical on every replica running the same binary:
//
//   - GENERIC decode failure (errPBApplyDecode without a version mismatch): the
//     entry bytes are malformed — they are malformed on every replica running the
//     same binary. (The version-skew SUB-case is the exception — see below.)
//   - a COMPILE-TIME handler's own fixed-limit / too-large / bad-argument
//     business error (e.g. ops.ErrShortArgs): a pure function of the entry and of
//     a handler that is a constant of the binary, so it is identical everywhere.
//     See the DYNAMIC-HANDLER CAVEAT below — this premise holds only for the
//     builtins, and error classification cannot rescue the dynamic case at all.
//
// Each of those fails on ALL replicas or NONE, so advancing the index keeps every
// replica in the same (failed-that-entry) state — no divergence. Treating them as
// fatal would needlessly halt an otherwise-healthy node, regressing availability.
// Hence the DEFAULT is classAdvance and we never widen the fatal set casually.
//
// Non-deterministic apply errors depend on per-node RUNTIME state that legitimately
// differs across replicas that are nonetheless in-sync on the log:
//
//   - ErrOpNotRegistered (the op name is absent from THIS node's registry). This
//     used to sit in the advance list above, justified by "unregistered on every
//     replica (same binary / same registry)". That premise died when ops became
//     DYNAMICALLY registrable: cluster's __register_wasm__ loads a module and adds
//     its op to the node-wide ops.Registry at APPLY time, so the registry is
//     per-node mutable state, not a compile-time constant.
//
//     Why it is fatal, precisely: cluster.Node.Call performs a registry Lookup and
//     returns ErrUnknownOp BEFORE proposing (cluster/node.go), so an entry can only
//     enter a Raft log at all if the PROPOSER's registry held the op. Seeing
//     ErrOpNotRegistered at APPLY time therefore means this replica's registry
//     DIFFERS from the proposer's — which is exactly the divergence condition: the
//     proposer (and every peer that has the op) executes the entry while this node
//     would skip it. Advancing is the silent divergence this whole file exists to
//     prevent.
//
//   - errPBApplyReadOnly (a committed entry names an op whose kind THIS node's
//     registry records as ops.OpReadOnly). This used to sit in the advance list
//     above, justified by "a structural property of the op, identical
//     everywhere". That premise died for exactly the same reason
//     ErrOpNotRegistered's did, and at exactly the same place: fsm.applyEntryData
//     reads the kind out of f.registry.Lookup, which is the NODE-WIDE MUTABLE
//     ops.Registry, and Kind is a WIRE-CONTROLLED field of ops.WASMRegistration
//     that cluster's RegisterOrReplaceModule installs on whichever node applies
//     the registration first.
//
//     Concretely: register X at Epoch 1 as OpReadWrite, then X at Epoch 2 as
//     OpReadOnly (a non-writing module, so wasm's moduleWritesState guard lets it
//     through). A registration is broadcast to every shard group, so during the
//     broadcast window a node that hosts the group the update reached first holds
//     X as read-only while a node that does not still holds it as read-write. An
//     invocation of X committed in a group both replicate: the second node
//     EXECUTES it, the first returns errPBApplyReadOnly. Under classAdvance the
//     first SKIPS an entry its peer executed — verbatim the divergence condition
//     this file exists to prevent.
//
//     WHY FATAL AND NOT ONE OF THE ALTERNATIVES. "Forbid a registration from
//     changing Kind" cannot be enforced deterministically: the check is
//     state-dependent, so a replica that already holds X rejects the update while
//     a replica that holds nothing accepts it — the same trap documented under
//     "WHY NOT reject a re-registration" in cluster/wasm_load.go. And the
//     cluster's route gate does NOT cover this: the gate runs at PROPOSE time and
//     constrains only which group an entry may ENTER, whereas this is an
//     APPLY-time property of the applying node's own registry. In the example
//     above the proposing node's gate is legitimately open (its own version is
//     proven in that group); nothing it can check locally reveals that a peer
//     holds a different Kind.
//
//   - cache.ErrFull (PolicyRejectWrites, shard at MaxPagesPerShard): whether a
//     write is rejected depends on THIS node's page occupancy, which is a function
//     of local eviction timing / warm-restart history, not of the committed log.
//
//   - cache.ErrCannotEvict (PolicyRingbufEvict, nothing left to evict + still no
//     room): same — a function of local occupancy, not the log.
//
//   - ErrLogEntryVersion (a stamped 0x00 entry whose version THIS binary cannot
//     decode — a version-skewed / prematurely-enabled fleet): the SAME committed
//     entry is decoded and applied by peers running a newer binary while this node
//     cannot parse it. Advancing here would skip a write peers applied → silent
//     divergence. It is fatal DESPITE arriving as a decode error, so it is matched
//     BEFORE the generic errPBApplyDecode default (which stays classAdvance). This
//     is the #4 Phase B / B1 rollout net for a POST-decoder version bump. The
//     pre-decoder case (an old binary reads the 0x00 marker as opNameLen=0 →
//     opName="" → ErrOpNotRegistered) is now ALSO caught, because
//     ErrOpNotRegistered is fatal — it halts instead of skipping. It no longer
//     relies on rollout discipline alone.
//
// A node hitting one of these while a peer stored/applied the write would, if it
// advanced, ACK the entry as applied while holding DIFFERENT state — a silent,
// permanent replica divergence. So these are classFatal: the replicated apply path
// must fail-closed (halt in Raft mode; abort/NACK in PB mode) instead of advancing.
//
// ROLLING-UPGRADE CONSEQUENCE (deliberate, not an oversight). Making
// ErrOpNotRegistered fatal means a node on an OLDER binary that does not know a
// newly-introduced op now HALTS on the first such committed entry instead of
// silently skipping it. That is the intended fail-closed posture: a halt is
// visible, bounded and recoverable (upgrade the binary, the entry replays);
// silent divergence is none of those. Deploy order (upgrade every replica before
// the new op is first invoked) remains the operator's job — it is now enforced by
// a loud stop rather than trusted.
//
// The SNAPSHOT format carries a second, independent rolling-upgrade constraint
// that points the other way: serializeSnapshot always writes the current version
// and readers reject anything newer, so an upgraded leader that snapshots and
// compacts hands an old follower an InstallSnapshot it can never accept. See the
// ONE-WAY UPGRADE note in snapshot.go — same deploy discipline, different
// failure surface (a stalled catch-up rather than a halt).
//
// DYNAMIC-HANDLER CAVEAT — LARGELY CLOSED, and the residue is worth stating
// exactly. It used to read: for a dynamically-registered WASM op the registry
// entry AND the module behind it are node-wide MUTABLE state, a registration is
// broadcast to shard groups that commit at independent times, so two replicas of
// the SAME group can run two different versions of one op name; error
// classification catches only the sub-cases that SURFACE as an error (a missing
// op, a changed Kind) and CANNOT catch the general one, because v1 and v2 may
// both apply the same committed entry successfully and simply write different
// values — no error to classify, just divergence.
//
// The general case is now closed at its source rather than by classification.
// The module version used to execute a committed entry in group g is resolved
// from g's OWN per-group binding table (wasm.Runtime.resolveModuleForInvoke),
// which is a pure function of g's ORDERED log prefix — so every replica of g,
// which by definition has applied that same sequence, picks the same version for
// the same entry regardless of which other groups it hosts. (Ordered, not
// unordered: the per-group fold is a maximum composed with a contract freeze and
// is order-DEPENDENT — see cluster.installedWASM.groups. Prefix determinism is
// what this argument needs and all it needs.) There is no longer a window in
// which two replicas of one group run different bytes for one entry, so in-place
// BYTES updates to a live WASM op are supported. The lookup's own failure mode
// (ops.ErrWASMNoGroupBinding) is classFatal above.
//
// WHAT REMAINS, AND IT FAILS CLOSED. Kind is still read node-wide on the PROPOSE
// side (it decides whether an entry is replicated at all), so it cannot be
// resolved per group and is FROZEN at first registration: cluster refuses a
// change at propose time, and refuses it again at apply time against the group's
// own binding, which is a pure function of that group's ordered prefix and so
// identical on every replica of it.
//
// What is NOT covered is a race between two FIRST-TIME registrations of one name
// declaring DIFFERENT KINDS, landing in different groups: no group has a prior
// binding to refuse against. The node-wide value is a maximum over the set of
// registrations a node RECEIVED, and cluster/wasm_gate.go's forwarded-leg gate
// makes that set differ between nodes on purpose — once peer P holds B it refuses
// a contract-differing A leg for every group P leads, so A never enters those
// groups' logs at all (see "STATE THE REST OF THAT COST" there). A node hosting
// only a group that got A therefore ends on A's Kind and one hosting only a group
// that got B ends on B's.
//
// AND THAT RESIDUE HALTS RATHER THAN DIVERGING. The node whose registry records
// the op read-only returns errPBApplyReadOnly on a committed entry its peer
// executes, and errPBApplyReadOnly is classFatal here — it HALTS instead of
// skipping. Visible, bounded, recoverable, which is the whole posture of this
// file.
//
// THE OTHER HALF OF THIS RESIDUE IS GONE, and it is worth saying what it was
// because it was the one hole in this file with no backstop at all. The key
// extractor COMPUTES the group index, so two nodes that ended on different
// extractors routed INV(X) to DIFFERENT shard groups: different replica sets
// applied it, every apply SUCCEEDED, and there was no error for classification to
// reach — it re-routed, SILENTLY. It was closed at its source rather than here:
// ops.WASMKeyExtractorHandle collapsed the legal set to ONE value, so "two
// registrations of one name declaring different extractors" is not a
// representable state and there is nothing left to classify. Closing it by
// classification was never available, which is exactly why it had to be closed
// structurally.
//
// AVAILABILITY CAVEAT for dynamically-registered ops. Because a registration is
// itself a replicated entry, a replica can be legitimately BEHIND on the log that
// carries the registration while already applying an invocation from another
// group's log. cluster/wasm_broadcast.go replicates every registration into every
// shard group specifically to shrink that window, and cluster/wasm_load.go carries
// registrations through snapshot/restore so a snapshot-installed replica is not
// permanently missing them. Those two make the halt rare; they do not make it
// impossible, and that residual is accepted on purpose — see the honest statement
// of what the broadcast does and does not establish in wasm_broadcast.go.
//
// TODO(#4 phase-B / vector-audit): the vector-engine capacity error surface
// (vector.CollectionStore inserts, index capacity, dimension/quota rejections) is
// NOT yet audited and NOT classified here — any non-deterministic error it can
// return on a committed entry would currently fall through to classAdvance.
// RESOLVED (#4 vector TTL determinism): vector TTL is no longer a wall-clock
// committed-state expiry site. Every replicated vector WRITE op — point TTL,
// per-key payload TTL, insert-if-absent liveness, CAS/reclaim liveness, and
// delete-by-filter selection, across all collection types (dense HNSW/IVF, the
// multi-vector and named-vector families) — now stamps its deadlines and judges
// its liveness against the leader apply stamp via the engine ...At variants
// (mirror cache.PutAt), so skewed replicas store byte-identical vector state. The
// wall-clock vector sweeper is disabled on replicated collections (Config/MV/named
// SuppressSweep), so expired vectors are filtered lazily at read time only, never
// physically removed on a per-node clock. CAVEAT (B3b analog, deferred): with the
// sweeper off, reclamation of expired-vector space on a replicated collection
// awaits a future deterministic stamped reclaimer. Separately, the DEFAULT cache
// policy (PolicyRingbufEvict) and
// wall-clock TTL can diverge replicas WITHOUT ever surfacing an apply error (a
// ring evicts a different victim, or a TTL boundary lands differently per node);
// that error-FREE divergence class cannot be caught by error classification at
// all and is handled by Phase B (deterministic eviction + logical TTL). This file
// closes only the error-SURFACED non-deterministic divergences (Phase A).
type applyClass uint8

const (
	// classAdvance: safe to advance the applied index. Either the apply
	// succeeded, or it failed DETERMINISTICALLY (identically on every replica).
	// This is the default — availability is never regressed for an error that
	// cannot cause divergence.
	classAdvance applyClass = iota
	// classFatal: a NON-DETERMINISTIC apply error whose occurrence depends on
	// per-node runtime state. Advancing the index here would silently diverge
	// this replica from its peers, so the replicated apply path must fail closed.
	classFatal
	// classRetry: DO NOT ADVANCE, DO NOT HALT — WAIT AND RE-RUN THIS ENTRY.
	//
	// ################ WHY A THIRD CLASS AND NOT ONE OF THE TWO ###############
	//
	// The two classes above between them assume that an apply error is a verdict:
	// either every replica reaches it (advance, they stay in agreement) or this
	// replica reached it alone (halt, refuse to diverge). ops.ErrWASMModuleNotResident
	// is neither. It says the entry cannot be judged HERE YET — this node holds a
	// correct binding to a module version whose bytes have not arrived, because a
	// thin registration marker names its module instead of carrying it.
	//
	// It cannot be classAdvance. Peers that hold the bytes EXECUTE the entry; this
	// node would skip it and record it as applied. That is verbatim the silent
	// permanent divergence this file exists to prevent.
	//
	// It cannot be classFatal either, and this is the less obvious half. Four
	// independent reasons, each sufficient:
	//
	//   - A HALT IS PROCESS-GLOBAL FOR A GROUP-LOCAL CONDITION. defaultFatalApply
	//     is os.Exit(1). One group missing one blob would take down every other
	//     group this node hosts, all of which are perfectly healthy.
	//   - FAILOVER CANNOT HELP. Every replica of the group applies the same log in
	//     the same order, so every one of them meets this entry; the difference
	//     between them is only WHETHER THEY HAVE THE BYTES, which leadership does
	//     not change. A halt sheds this node's load onto peers that may be in
	//     exactly the same state.
	//   - IT CRASH-LOOPS. The condition does not clear by restarting: the entry is
	//     committed, so it replays into the same missing blob, exits again, and the
	//     node never comes up. A halt is supposed to be bounded and recoverable
	//     (upgrade the binary, add capacity, the entry replays); this one is
	//     neither.
	//   - THERE IS NO DIVERGENCE FOR IT TO PREVENT. A halt earns its cost by
	//     stopping a node before it records state its peers do not have. A retry
	//     records nothing and mutates nothing — the handler never ran — so there is
	//     nothing to stop.
	//
	// And unlike either of them, the condition SELF-HEALS: the blob arrives (pushed
	// at registration, fetched on demand, or handed over by an operator with
	// __wasm_blob_put__) and the same entry applies normally. So the right answer
	// is to make waiting an explicit, observable FSM state rather than an outcome.
	//
	// WHAT IT COSTS, and it is not free — this is the second-order price of the
	// unbounded wait, stated here because this is where the decision lives.
	// hashicorp/raft calls Snapshot and Restore on the SAME goroutine as Apply, so
	// a group parked in a retry CANNOT SNAPSHOT and therefore cannot compact its
	// Raft log, and cannot accept an InstallSnapshot either. Its log grows for as
	// long as it waits. That is why the observability for this class alerts on the
	// DURATION of the longest current block and not merely on a count: a short
	// block is invisible and harmless, and a long one is a disk-consumption
	// incident before it is anything else. See cluster.WASMBlockStats.
	classRetry
)

// classifyApplyErr maps an ApplyResponse.Err to an applyClass. It uses an
// INCLUSION list: the result is classAdvance unless err matches one of the known
// non-deterministic sentinels, so a newly-introduced error defaults to the safe,
// availability-preserving behavior (advance) rather than an unexpected halt. Add
// an error here ONLY after confirming it can differ across in-sync replicas.
//
// A nil err classifies as classAdvance (a successful apply always advances).
//
// classRetry is NOT part of the inclusion list's default reasoning: it is a
// single, explicitly-matched sentinel meaning "this entry has not been judged
// yet". Nothing defaults into it, and nothing should — a wrongly-retried entry
// stalls a group forever, which is the one failure mode here with no timeout.
func classifyApplyErr(err error) applyClass {
	// ErrLogEntryVersion is checked FIRST and independently of the errPBApplyDecode
	// wrapper it rides inside: a version-skew entry is fatal (peers may decode+apply
	// it) even though it surfaces as a decode error, whereas a generic decode
	// failure is deterministic and stays classAdvance. The apply path wraps with
	// multi-%w so errors.Is still sees this sentinel through the wrap.
	// ops.ErrWASMModuleNotResident is checked FIRST, before every fatal sentinel,
	// because it is the one outcome that is not a verdict at all: the entry has not
	// been judged yet. Nothing below can be true of it — the op IS registered, the
	// group IS bound, the entry DID decode — so the order is not load-bearing for
	// correctness today. It is first anyway so that a future sentinel added to the
	// fatal list can never accidentally swallow a wait and turn it into a halt,
	// which is the one misclassification here that crash-loops a node.
	if errors.Is(err, ops.ErrWASMModuleNotResident) {
		return classRetry
	}
	if errors.Is(err, ErrLogEntryVersion) {
		return classFatal
	}
	// ErrOpNotRegistered: this replica's op registry differs from the proposer's
	// (the proposer had to Look the op up to propose at all). Skipping would
	// execute-on-peers / skip-here — the divergence condition. See the doc block.
	if errors.Is(err, ErrOpNotRegistered) {
		return classFatal
	}
	// errPBApplyReadOnly: this replica's registry records the op as read-only
	// while the proposer's recorded it as writable (shard.Store.Call never
	// proposes a read-only op). Same registry-disagreement divergence condition as
	// ErrOpNotRegistered — skipping would execute-on-peers / skip-here. Checked
	// independently of the errPBApplyDecode default for the same reason
	// ErrLogEntryVersion is. See the doc block.
	if errors.Is(err, errPBApplyReadOnly) {
		return classFatal
	}
	// ops.ErrWASMNoGroupBinding: a committed entry in THIS group's log invokes a
	// replicated WASM op for which this replica holds no per-group version binding,
	// while the proposer necessarily held one (its route gate had to be open to
	// propose at all). Peers execute the entry; this node cannot. Same
	// registry-disagreement divergence condition as ErrOpNotRegistered. See the
	// sentinel's own doc for the full argument.
	if errors.Is(err, ops.ErrWASMNoGroupBinding) {
		return classFatal
	}
	if errors.Is(err, cache.ErrFull) || errors.Is(err, cache.ErrCannotEvict) {
		return classFatal
	}
	// cache.ErrFlushNotDurable: this replica could not make the flush watermark
	// (flushed.seq sidecar) durable — a LOCAL disk fault that a peer need not share.
	// It is fatal for the same reason ErrFull is: advancing the applied index past a
	// flush this node could not durably record would, on a later failover, resurrect
	// every pre-flush key while peers whose sidecar landed hold an empty keyspace —
	// silent, permanent divergence. Fail-closed instead (halt in Raft mode; NACK in
	// PB mode). On a single-node (non-replicated) shard classFatal does not halt (see
	// fsm.go), which is correct: there is no peer to diverge from, and shard.flush
	// writes the sidecar BEFORE the index swap, so the keyspace stays intact anyway.
	if errors.Is(err, cache.ErrFlushNotDurable) {
		return classFatal
	}
	// cache.ErrReuseBarrier: this replica could not make a reused mmap extent's cleared
	// header durable before refilling it — a LOCAL disk/entropy fault (msync or the
	// crypto/rand nonce rotation) that a peer need not share. It surfaces from a write
	// whose eviction of a victim page hit the barrier. Fatal for the same reason ErrFull
	// is: the peer that did not hit the fault stores/applies the write while this node
	// fails it, so advancing the applied index here would ACK an entry this replica did
	// not apply — silent divergence. Fail-closed instead (halt in Raft mode; NACK in PB
	// mode); the fault is transient, so a restart or failover replays the entry and it
	// then lands in a durably-reset extent.
	if errors.Is(err, cache.ErrReuseBarrier) {
		return classFatal
	}
	return classAdvance
}
