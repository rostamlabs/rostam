// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sync/atomic"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/vector"
)

// ApplyResponse is what Apply returns. Wrapping result + error in a struct
// lets Raft callers distinguish "no error, no result" (nil + nil) from
// "error" without ambiguity in the `any` return.
type ApplyResponse struct {
	Result []byte
	Err    error
}

// fsm implements hraft.FSM. It dispatches every Raft log entry to a handler
// in the ops.Registry that runs inside a TxContext.
type fsm struct {
	cache    *cache.Cache
	registry *ops.Registry
	tx       *ops.TxContext // reused across every Apply — Raft serializes Apply per-fsm anyway
	vectors  *vector.CollectionStore
	durable  bool

	// wasmSnapshot / wasmRestore carry dynamic op registrations through a
	// snapshot install (see Config.WASMSnapshot). nil on a bare Store.
	wasmSnapshot func() []byte
	wasmRestore  func([]byte) error

	// isReplicated reports whether this shard currently runs in an RF>1 Raft group
	// (a real peer set), as opposed to a genuine single-node store. It gates the
	// fail-closed halt in Apply/ApplyBatch: a fatal non-deterministic apply error
	// (classFatal) can only cause SILENT DIVERGENCE when there are peers to diverge
	// from, so on a single node we advance and surface the error normally rather
	// than halting (a pure availability regression with no correctness benefit).
	//
	// It reads LIVE Raft group membership at apply time (shard.New wires it to the
	// raft node's GetConfiguration), NOT construction-time peers. This is required
	// for online resharding: a shard joined via cluster AddShardOwner is built with
	// an EMPTY peer list (owners=nil), then AddVoter'd into a live RF>1 group — a
	// construction-time snapshot would read len==0 and wrongly disable the gate on
	// exactly that node, reintroducing the divergence bug. The wiring FAILS CLOSED:
	// an unreadable or not-yet-learned (0-server) configuration counts as replicated,
	// so only a positively-observed single-server group disables the halt. nil is
	// treated as false (see replicated()) — a bare fsm (no raft) never halts.
	isReplicated func() bool

	// onFatalApply is invoked when a replicated shard hits a classFatal apply
	// error. Its contract is to HALT the process WITHOUT letting the applied index
	// advance. The production default (defaultFatalApply) uses log.Fatalf → os.Exit,
	// which bypasses both any caller defer AND hraft's per-Apply panic recovery, so
	// the index provably does not advance and the entry replays on restart. It is
	// injectable so tests can assert the fail-closed decision without os.Exit'ing
	// the test binary. nil is treated as the default halt (see fireFatalApply).
	onFatalApply func(err error)

	// onApplyRetry / onApplyRetryCleared observe a classRetry block: an apply that
	// cannot be judged yet because the module a committed entry names is not
	// resident on this node (see classRetry). onApplyRetry fires before EVERY wait,
	// onApplyRetryCleared once when the entry finally applies.
	//
	// ############ NEITHER MAY BLOCK, AND NEITHER MAY TAKE AN APPLY LOCK #########
	//
	// They run ON the FSM apply goroutine, so anything they wait for stalls this
	// group. The cluster-side implementation is the one that matters: it resolves
	// the missing fingerprint under wasmApplyMu, RELEASES IT, and kicks a
	// fire-and-forget fetch. It must never fetch inline and must never still hold
	// wasmApplyMu when it returns — cluster's snapshotWASMState and restoreWASMState
	// take that same mutex, so a block holding it would stall EVERY group's snapshot
	// on this node instead of only the one that is waiting.
	//
	// nil is fine: the FSM still retries, just silently (a bare Store with no
	// cluster behind it has no fetcher to kick and nothing to report to).
	onApplyRetry        func(err error, attempt int, blockedFor time.Duration)
	onApplyRetryCleared func(err error, attempts int, blockedFor time.Duration)

	// applyRetryWait is the base backoff between re-runs of a blocked entry. It
	// doubles up to applyRetryWaitCap. Zero means defaultApplyRetryWait.
	applyRetryWait time.Duration

	// stop is closed when the owning Store is closing. It is the ONLY exit from an
	// otherwise unbounded block, and it exists because the block is unbounded: a
	// group parked in a retry holds hashicorp/raft's single FSM goroutine, so
	// raft.Shutdown would never observe its shutdown channel and Store.Close would
	// hang forever. Abandoning the retry on shutdown is safe for exactly the reason
	// the halt path is safe — the applied index is NOT advanced, the entry is
	// preserved, and it replays on the next start.
	//
	// nil (a bare fsm with no Store) means "never stop", which is the correct
	// default: there is no Close to race.
	stop <-chan struct{}

	// abandoned records that a classRetry block was given up on mid-entry because
	// the owning Store started closing (see stop). It is set on the two abandon
	// paths and never cleared: the fsm it is set on belongs to a process that is
	// on its way down.
	//
	// ######## IT EXISTS BECAUSE RAFT'S OWN INDEX BOOKKEEPING IS NOW A LIE ########
	//
	// This is not about our accounting — ours is right, the abandoned entry is not
	// recorded. It is about hashicorp/raft's. runFSM (v1.7.3 fsm.go) computes
	// `lastBatchIndex` over EVERY log in the batch BEFORE calling ApplyBatch, and
	// assigns `lastIndex = lastBatchIndex` UNCONDITIONALLY once it returns; the
	// single-entry applySingle does the same after Apply. Raft has no way to be
	// told "I applied a prefix", because until this branch no path could return
	// from Apply/ApplyBatch in a live process without having applied the entry —
	// classFatal reaches os.Exit(1) and only a test hook returns. The classRetry
	// abandon is the first, so raft now believes indexes we never applied are in
	// state, and two of its consumers act on that belief:
	//
	//   - fsm.go's snapshot() stamps the request with `lastIndex`, which
	//     snapshot.go hands to snapshots.Create. We would serialize state as of
	//     appliedPrefix under a label of lastBatchIndex; on restart raft sets
	//     lastApplied from the snapshot metadata and NEVER redelivers the entries
	//     in (appliedPrefix, lastBatchIndex]. Silent, permanent, unrecoverable.
	//   - runFSM's own loop, which is the likelier of the two and needs no race at
	//     all. Store.Close closes stop BEFORE raft.Shutdown closes shutdownCh, so
	//     when the abandon unwinds, the FSM goroutine's select sees only the 128-
	//     deep fsmMutateCh — full of batches committed while we were parked. Those
	//     apply normally, and the post-loop SetAppliedIndex writes a durable
	//     watermark ABOVE the entries we skipped, so warm restart drops them.
	//
	// The flag closes both: Snapshot refuses, and Apply/ApplyBatch refuse to apply
	// anything further. Refusing is free — this only ever happens during Close.
	abandoned atomic.Bool

	// appliedIndex is the TRUE FSM-applied index: the highest Raft log index for
	// which Apply has fully run (handler + state mutation), advanced at the END of
	// each Apply. Unlike cache.AppliedIndex() (which is 0 in heap mode) and unlike
	// raft.AppliedIndex() (which only reflects what raft DISPATCHED to this
	// goroutine), this is the exact signal the linearizable readIndex barrier polls:
	// once appliedIndex >= the committed index captured after VerifyLeader, the
	// local state reflects everything committed as of confirmed leadership. Atomic
	// because the barrier reads it concurrently from the read path.
	appliedIndex atomic.Uint64
}

// AppliedIndex returns the highest Raft log index fully applied to this FSM.
func (f *fsm) AppliedIndex() uint64 { return f.appliedIndex.Load() }

// replicated reports whether this shard runs with a configured RF>1. nil-safe:
// an unwired isReplicated (e.g. a bare test fsm) counts as single-node, so the
// fail-closed halt never fires unless replication was explicitly configured.
func (f *fsm) replicated() bool {
	return f.isReplicated != nil && f.isReplicated()
}

// fireFatalApply halts on a classFatal non-deterministic apply error. When no
// hook is injected it uses the production default (log.Fatalf → os.Exit), which
// halts BEFORE the caller advances the applied index and BEFORE hraft can recover
// — the whole point is that this replica stops rather than ACKing an entry whose
// effects diverge from its peers. Tests inject onFatalApply to observe the
// decision (and may return, in which case callers must NOT advance the index).
func (f *fsm) fireFatalApply(err error) {
	if f.onFatalApply != nil {
		f.onFatalApply(err)
		return
	}
	defaultFatalApply(err)
}

// defaultFatalApply is the production halt. log.Fatalf calls os.Exit(1), which
// runs no deferred functions and is not recoverable — guaranteeing the applied
// index is not advanced. This is a DELIBERATE consistency-over-availability halt:
// the node stops rather than durably diverge from its peers. The offending entry is
// preserved (not skipped) so it replays once state can accept it.
//
// A RESTART IS NOW A REAL REMEDY for the cache.ErrFull case, which it was not when
// this comment first claimed capacity "does not auto-correct". Cold compaction at
// shard open (cache/compact.go) rewrites the pages file live-only, and since
// deletes are persisted as tombstones (cache/shard.go delH) a deleted key's ENTIRE
// byte history — the tombstone and every older copy — is reclaimed in that pass.
// So a shard whose occupancy is ghost bytes recovers by restarting. A shard that is
// genuinely full of LIVE data still does not: compaction has nothing to reclaim,
// the replay hits the same entry, and clearing the condition (more capacity, a
// larger per-shard memory budget) remains the operator's job.
func defaultFatalApply(err error) {
	// slog.Error + os.Exit(1) is the slog analog of log.Fatalf: it emits the halt
	// reason in the configured log format, then exits WITHOUT running deferred
	// functions (so the applied index is not advanced — the whole point of the
	// halt), exactly as log.Fatalf did.
	slog.Error("FATAL non-deterministic apply error — halting to preserve replica "+
		"consistency (deliberate fail-closed; needs operator remediation if the "+
		"capacity condition persists — see runbook)", "component", "shard", "err", err)
	os.Exit(1)
}

// defaultApplyRetryWait and applyRetryWaitCap bound the re-run cadence of a
// blocked entry (see classRetry).
//
// The first re-run comes fast because the overwhelmingly common block is a race
// that is already over: the marker applied a few milliseconds before the blob
// finished being written by a push that was in flight, or a prefetch kicked at
// marker-apply is one round trip from landing. Backing off geometrically from
// there keeps a genuinely long block — a member the push could not reach, waiting
// on a fetch — from spinning: at the cap, a blocked group costs one map lookup
// and one hook call per second, which is nothing next to the Raft log it is
// failing to compact.
//
// The cap is what makes the escalating log schedule expressible at all: the
// cluster-side hook is what decides to WARN at 5s and ERROR every 30s, and it can
// only do that if it is called often enough to notice those thresholds passing.
const (
	defaultApplyRetryWait = 50 * time.Millisecond
	applyRetryWaitCap     = time.Second
)

// newFSM builds the apply dispatcher for one shard group.
//
// kvIdx is THE KV record index of c — the same pointer the owning Store hands
// its read-only TxContext, created once next to the cache by ops.NewKVIndexFor.
// A second, private Set here would compile and then silently answer every
// kv_query empty: writes would maintain the FSM's index while reads are served
// from the Store's. nil is valid and means this FSM maintains no index (bare
// FSMs in tests, embedders that never define one).
func newFSM(c *cache.Cache, r *ops.Registry, durable bool, vectors *vector.CollectionStore, kvIdx *kvindex.Set) *fsm {
	return &fsm{
		cache:    c,
		registry: r,
		tx:       ops.NewTxContextWithIndex(c, vectors, kvIdx),
		durable:  durable,
		vectors:  vectors,
	}
}

// Apply decodes the log entry, looks up the handler, and runs it, then advances
// the applied index — but ONLY when advancing is safe. The index advances on a
// successful apply and on a DETERMINISTIC error (one that fails identically on
// every replica, so all replicas stay in agreement — see classifyApplyErr). On a
// NON-DETERMINISTIC (classFatal) error in a REPLICATED shard, advancing would
// silently diverge this replica from its peers, so we fail closed via
// onFatalApply (default: halt the process) and do NOT advance.
//
// Panic semantics (hashicorp/raft v1.7.3): runFSM does NOT recover an FSM panic —
// a handler panic crashes the process. The advance is therefore kept in a defer so
// it runs DURING the panic unwind (skipAdvance is false on that path), persisting
// the durable applied-index header exactly as the pre-#4 code did. That is load-
// bearing: without it a durable-mode restart would replay the poison entry, panic
// again, and never make progress — a permanent shard-wide crash loop. The defer's
// advance is suppressed (skipAdvance=true) on the deliberate classFatal fail-
// closed path, where not-advancing is the whole point, and on the classRetry path
// abandoned by shutdown, where the entry has not been applied at all.
//
// THE RETRY LOOP NEEDS NO SUPPRESSION OF ITS OWN, and it is worth saying why
// rather than leaving it to be rediscovered. A deferred function runs when the
// FUNCTION returns, not when a loop inside it iterates, so the advance below is
// structurally unreachable while applyWithRetry is still re-running the entry.
// "Nothing happened, so nothing may be recorded" is enforced by the shape of the
// code, not by a flag; the only way to break it is to move the advance out of the
// defer and into the loop.
func (f *fsm) Apply(l *hraft.Log) any {
	// Nothing may be applied after an abandon — see fsm.abandoned. Raft's
	// lastIndex already covers the entry we walked away from, so applying a LATER
	// one would write a watermark over a hole.
	if f.abandoned.Load() {
		return &ApplyResponse{Err: errApplyAbandoned}
	}
	// Warm-restart skip: when the mmap-backed cache header records an
	// applied index >= this log entry, the entry was already applied in
	// a prior process lifetime. Returning early avoids redundant work
	// and prevents double-applying an idempotent op like Put.
	if l.Index <= f.cache.AppliedIndex() {
		// Already applied in a prior process lifetime (mmap warm restart). The
		// entry is durably in state, so advance the in-memory FSM applied index
		// too — the barrier and any catch-up poll must see it as applied.
		f.advanceApplied(l.Index)
		return &ApplyResponse{}
	}

	// Advance the TRUE FSM applied index AFTER the handler has run and mutated
	// state. Deferred so it ALSO fires during a panic unwind (see the panic note
	// above): a crashing handler still records the index, so warm restart SKIPS the
	// poison entry instead of re-panicking on it. SetAppliedIndex (durable mmap
	// header) runs BEFORE advanceApplied (barrier signal) so a linearizable read
	// never observes an applied index the header has not yet recorded. skipAdvance
	// gates the ONE case where we must not advance: the classFatal halt below.
	skipAdvance := false
	defer func() {
		if skipAdvance {
			return
		}
		f.cache.SetAppliedIndex(l.Index, f.durable)
		f.advanceApplied(l.Index)
	}()

	// The single-entry path has no prefix to record before waiting: this entry IS
	// the batch, and nothing before it ran. Hence a nil beforeFirstWait.
	resp, applied := f.applyWithRetry(l.Data, nil)
	if !applied {
		// Still blocked, abandoned because the Store is closing. Nothing ran, so
		// nothing may be recorded — the entry is committed and replays next start.
		// Raft records l.Index as applied anyway (see fsm.abandoned), so poison the
		// snapshotter and every later apply before returning into its loop.
		skipAdvance = true
		f.abandoned.Store(true)
		return resp
	}

	// Fail-closed gate: a non-deterministic apply error on a replicated shard must
	// NOT advance the index (doing so would ACK an entry whose local effects differ
	// from peers — silent divergence). The default onFatalApply halts via os.Exit,
	// which bypasses this deferred advance entirely; a test-injected hook may return,
	// so skipAdvance keeps the deferred advance from running in that case too.
	if resp.Err != nil && f.replicated() && classifyApplyErr(resp.Err) == classFatal {
		skipAdvance = true
		f.fireFatalApply(fmt.Errorf("shard: fatal non-deterministic apply at index %d: %w", l.Index, resp.Err))
		return resp
	}

	return resp
}

// applyWithRetry runs one entry until it produces a VERDICT — success, a
// deterministic error, or a fatal one — re-running it verbatim for as long as it
// classifies classRetry.
//
// ##################### THE RETRY MUTATES AND RECORDS NOTHING #################
//
// That is the whole invariant, and every caller depends on it: a blocked apply
// never reached its handler, so no state changed, no index may advance, and
// re-running the SAME entry later is not a re-application — it is the first one.
// This function therefore does no bookkeeping at all; callers own the index, and
// the two of them differ only in what they must record BEFORE the first wait.
//
// beforeFirstWait runs exactly once, immediately before the first sleep, and is
// how ApplyBatch durably records the prefix of already-applied entries. It runs
// only when a block actually happens, so the unblocked path pays nothing.
//
// THE WAIT IS UNBOUNDED except for shutdown, and reports which of the two it was.
// ok=false means the Store is closing and the entry is still blocked: the caller
// must NOT advance, exactly as on the classFatal path. The entry is committed and
// preserved, so it replays on the next start; by then the blob has very likely
// arrived, and if it has not, the node blocks again and says so.
func (f *fsm) applyWithRetry(data []byte, beforeFirstWait func()) (resp *ApplyResponse, ok bool) {
	var (
		attempts int
		start    time.Time
		// blockErr is the error that CAUSED the block, kept because the cleared
		// hook needs it and the successful re-run does not carry it. The observer
		// keys its live-block record on the coordinates inside this error (op,
		// group), so handing it the FINAL response — which is nil on success — would
		// leave the record it created permanently un-retired: a phantom entry on the
		// gauge operators are told to alert on, which is worse than no reporting at
		// all because it is a false alarm that never ends.
		blockErr error
	)
	for {
		resp = f.applyEntryData(data)
		if resp.Err == nil || classifyApplyErr(resp.Err) != classRetry {
			if attempts > 0 {
				f.fireApplyRetryCleared(blockErr, attempts, time.Since(start))
			}
			return resp, true
		}
		blockErr = resp.Err
		if attempts == 0 {
			start = time.Now()
			if beforeFirstWait != nil {
				beforeFirstWait()
			}
		}
		attempts++
		// The hook is called BEFORE the wait and OUTSIDE any lock this FSM holds:
		// it is what kicks the fetch that will eventually unblock us, so calling it
		// after the sleep would add a full backoff interval to every block for no
		// reason. It must not block — see fsm.onApplyRetry.
		f.fireApplyRetry(resp.Err, attempts, time.Since(start))
		if !f.waitApplyRetry(attempts) {
			return resp, false
		}
	}
}

// waitApplyRetry sleeps the backoff for the given attempt, or returns false the
// moment the owning Store starts closing.
//
// A plain time.Sleep would be simpler and is wrong: hashicorp/raft runs Apply,
// ApplyBatch, Snapshot and Restore on ONE goroutine, so a group sleeping here is
// a group that cannot observe raft's shutdown channel. With an unbounded block
// that turns Store.Close into a hang rather than a shutdown.
func (f *fsm) waitApplyRetry(attempt int) bool {
	wait := f.applyRetryWait
	if wait <= 0 {
		wait = defaultApplyRetryWait
	}
	// Geometric backoff, capped. attempt is 1-based, so the first wait is `wait`.
	for i := 1; i < attempt && wait < applyRetryWaitCap; i++ {
		wait *= 2
	}
	if wait > applyRetryWaitCap {
		wait = applyRetryWaitCap
	}
	t := time.NewTimer(wait)
	defer t.Stop()
	select {
	case <-f.stop:
		return false
	case <-t.C:
		return true
	}
}

// fireApplyRetry / fireApplyRetryCleared are nil-safe hook calls. A bare fsm
// (no cluster behind it) retries silently.
func (f *fsm) fireApplyRetry(err error, attempt int, blockedFor time.Duration) {
	if f.onApplyRetry != nil {
		f.onApplyRetry(err, attempt, blockedFor)
	}
}

func (f *fsm) fireApplyRetryCleared(err error, attempts int, blockedFor time.Duration) {
	if f.onApplyRetryCleared != nil {
		f.onApplyRetryCleared(err, attempts, blockedFor)
	}
}

// applyEntryData decodes an encoded log entry (EncodeLogEntry), looks up its op
// handler, and runs it inside the shared TxContext. It is the single place both
// the Raft FSM path (applyCommand) and the primary-backup engine's Applier go
// through, so an op behaves identically under either replication mode. No
// applied-index bookkeeping happens here.
func (f *fsm) applyEntryData(data []byte) *ApplyResponse {
	opName, args, stampMs, stamped, err := DecodeLogEntry(data)
	if err != nil {
		// Wrap with MULTI-%w so BOTH sentinels survive in the chain: errPBApplyDecode
		// (so the PB path's isInfraError aborts/NACKs) AND, for a version-skew entry,
		// ErrLogEntryVersion (so classifyApplyErr returns classFatal and the replicated
		// Raft FSM HALTS instead of silently skip-advancing). A plain %v on the inner
		// error would discard the ErrLogEntryVersion identity and defeat the halt.
		//
		// The halt covers the POST-decoder version-bump / premature-enable case: this
		// node cannot parse a 0x00 entry a PEER may decode and apply, so advancing here
		// would diverge — it must fail closed. A GENERIC malformed-bytes decode error
		// (not a version mismatch) stays classAdvance: it fails identically on every
		// replica, so skip-advance keeps them in agreement. The PRE-decoder case (an
		// OLD binary with no 0x00 support reads the marker as opNameLen=0 → opName="" →
		// ErrOpNotRegistered → classAdvance → skip) is NOT caught by classification and
		// relies on rollout discipline — see EncodeLogEntryStamped's two-phase note.
		return &ApplyResponse{Err: fmt.Errorf("%w: %w", errPBApplyDecode, err)}
	}
	handler, kind, _, ok := f.registry.Lookup(opName)
	if !ok {
		return &ApplyResponse{Err: ErrOpNotRegistered}
	}
	if kind == ops.OpReadOnly {
		return &ApplyResponse{Err: fmt.Errorf("%w: %q", errPBApplyReadOnly, opName)}
	}
	// Thread the leader-stamped clock into the handler's TxContext so its KV
	// Get/Put/Expire evaluate and stamp expiry against the SAME clock on every
	// replica (#4 Phase B / B1). `stamped` — not stampMs != 0 — drives the choice:
	// a stamped entry ALWAYS uses the At-path (deterministic even for a 0 stamp),
	// and a legacy/unstamped entry uses the wall clock (byte-identical to pre-B1).
	// Reset after the handler so a subsequent apply (or any reuse of this shared tx)
	// never inherits a stale stamp.
	f.tx.SetApplyStamp(stampMs, stamped)
	result, herr := handler(f.tx, args)
	f.tx.SetApplyStamp(0, false)
	return &ApplyResponse{Result: result, Err: herr}
}

// ApplyBatch implements hraft.BatchingFSM. hashicorp/raft already coalesces
// concurrently-submitted writes into one commit batch; with a plain FSM it still
// dispatches them one entry at a time. Implementing ApplyBatch lets Raft hand us
// the whole committed batch at once, so we pay the applied-index bookkeeping
// (and, in durable mode, the mmap-header write via SetAppliedIndex) ONCE per
// batch instead of once per entry — the win on the concurrent-write hot path.
//
// Contract (see hashicorp/raft runFSM): exactly one response per input log, in
// order; the batch may include LogConfiguration entries (which plain Apply never
// sees — Raft applies those itself), so anything that is not a LogCommand gets a
// benign empty response and skips the op handler.
//
// Applied-index bookkeeping is done AFTER the loop (not deferred). On a handler
// PANIC mid-batch the process crashes WITHOUT recording the batch's index (v1.7.3
// runFSM does not recover), so warm restart replays the whole batch. Batch ops are
// re-applied on that replay, so an op that is NOT idempotent (e.g. incr, a
// Get-then-Put) could double-apply — but a panic is an unexpected-bug path, not a
// steady-state one, and the alternative (recording an index for a batch whose tail
// never ran) would skip un-applied entries, which is worse.
//
// The DELIBERATE classFatal fail-closed path below is different: it is a normal,
// reachable outcome, so it must NOT rely on idempotency. Before halting at entry i
// it durably records the watermark for the successfully-applied PREFIX (entries
// strictly before i), so a replay skips that prefix and re-applies ONLY the fatal
// entry (and anything after it). This is what keeps a non-idempotent op in the
// prefix (incr@5 committed, put@6 ErrFull) from double-counting on restart.
//
// The classRetry BLOCK path has the identical requirement and satisfies it the
// identical way — see the beforeFirstWait callback below. It is arguably the more
// pressing of the two: a halt exits the process immediately, so the window in
// which an operator can intervene is nil, whereas a block is a long-lived state
// whose whole visible symptom is "this node stopped making progress", which is
// the single most likely thing to be answered with a restart.
func (f *fsm) ApplyBatch(logs []*hraft.Log) []any {
	resps := make([]any, len(logs))
	// Nothing may be applied after an abandon — see fsm.abandoned. This is the arm
	// that needs no race to fire: Store.Close closes stop before raft.Shutdown
	// closes shutdownCh, so the batches already queued in raft's 128-deep
	// fsmMutateCh are dispatched here the moment the abandon unwinds, and the
	// post-loop durable SetAppliedIndex would jump the watermark over the hole.
	if f.abandoned.Load() {
		for i := range resps {
			resps[i] = &ApplyResponse{Err: errApplyAbandoned}
		}
		return resps
	}
	cacheApplied := f.cache.AppliedIndex()
	var maxIdx uint64
	// appliedPrefix tracks the highest log index that is safely applied so far —
	// the durable watermark to record if we must fail-closed mid-batch. It starts
	// at cacheApplied and rises only as entries are processed (applied, warm-skipped,
	// or benign), so at a fatal entry it holds the max index of the prefix BEFORE it.
	appliedPrefix := cacheApplied
	for i, l := range logs {
		if l.Index > maxIdx {
			maxIdx = l.Index
		}
		// Non-command entries (e.g. LogConfiguration) are not ops: respond
		// benignly and let the index advance cover them.
		if l.Type != hraft.LogCommand {
			resps[i] = &ApplyResponse{}
			if l.Index > appliedPrefix {
				appliedPrefix = l.Index
			}
			continue
		}
		// Warm-restart skip: already applied in a prior process lifetime.
		if l.Index <= cacheApplied {
			resps[i] = &ApplyResponse{}
			if l.Index > appliedPrefix {
				appliedPrefix = l.Index
			}
			continue
		}
		// A classRetry block at entry i must RECORD THE PREFIX DURABLY BEFORE IT
		// WAITS, and this callback is the whole of that requirement.
		//
		// ############ WHY THE WAIT CANNOT BE ENTERED UNRECORDED ############
		//
		// The block is unbounded, so "eventually the entry applies and the post-loop
		// write covers everything" is not a complete story: an operator restarting a
		// blocked node — which is exactly what an operator does to a node that has
		// stopped making progress — kills the process mid-wait. With nothing
		// recorded, raft replays the WHOLE batch, and entries 0..i-1 apply a SECOND
		// time. For an idempotent op that is merely wasted work; for a
		// non-idempotent one (incr, a Get-then-Put) it is silent corruption of
		// committed state, and it is the same hazard this file already documents for
		// the panic path — except that this path is a normal, reachable, deliberately
		// long-lived state rather than an unexpected bug.
		//
		// It uses the SAME code the classFatal branch below uses, for the same
		// reason and with the same guard: appliedPrefix has NOT yet been raised to
		// this (blocked) entry, so its index is intentionally excluded, and
		// SetAppliedIndex stores unconditionally so it must not be allowed to regress
		// the persisted watermark below cacheApplied.
		//
		// It runs ONCE, before the FIRST wait, not on every re-run: the prefix cannot
		// change while we are parked, and rewriting the mmap header once per backoff
		// tick would be pure I/O for no information.
		resp, applied := f.applyWithRetry(l.Data, func() {
			if appliedPrefix > cacheApplied {
				f.cache.SetAppliedIndex(appliedPrefix, f.durable)
				f.advanceApplied(appliedPrefix)
			}
		})
		resps[i] = resp
		if !applied {
			// Still blocked, abandoned because the Store is closing. The prefix is
			// already durable (above); this entry and everything after it replay.
			// Raft, however, has already decided the WHOLE batch applied (see
			// fsm.abandoned), so poison the snapshotter and every later apply on the
			// way out — otherwise its bookkeeping, not ours, loses the entries.
			f.abandoned.Store(true)
			return resps
		}
		// Fail-closed per entry: a non-deterministic apply error on a replicated
		// shard must halt. Before halting, durably record the successfully-applied
		// PREFIX (appliedPrefix = max index of entries strictly before this one) so
		// replay skips it and only the fatal entry replays — non-idempotent prefix
		// ops (incr) must not double-apply. appliedPrefix has NOT yet been raised to
		// this (fatal) entry, so its index is intentionally excluded. The guard
		// mirrors the post-loop one: SetAppliedIndex stores unconditionally and must
		// not regress the persisted watermark below cacheApplied. The default
		// onFatalApply halts via os.Exit (after the durable write); a test-injected
		// hook may return, and the prefix watermark is already recorded.
		if resp.Err != nil && f.replicated() && classifyApplyErr(resp.Err) == classFatal {
			if appliedPrefix > cacheApplied {
				f.cache.SetAppliedIndex(appliedPrefix, f.durable)
				f.advanceApplied(appliedPrefix)
			}
			f.fireFatalApply(fmt.Errorf("shard: fatal non-deterministic apply at index %d (batch): %w", l.Index, resp.Err))
			return resps
		}
		if l.Index > appliedPrefix {
			appliedPrefix = l.Index
		}
	}
	if maxIdx > 0 {
		// Durable header first, then the barrier signal — matching Apply's
		// LIFO-defer order (SetAppliedIndex before advanceApplied) so a
		// linearizable read never observes an applied index the mmap header has
		// not yet recorded. The durable write is guarded on maxIdx > cacheApplied:
		// unlike the monotonic advanceApplied (CAS max-guard), SetAppliedIndex
		// stores UNCONDITIONALLY and can lower the watermark. During a durable
		// warm-restart replay raft re-delivers already-applied entries in
		// MaxAppendEntries-sized batches (all warm-skipped above), and an
		// unguarded write on such a batch (maxIdx < persisted header) would
		// REGRESS the persisted watermark, defeating the warm-restart skip.
		if maxIdx > cacheApplied {
			f.cache.SetAppliedIndex(maxIdx, f.durable)
		}
		f.advanceApplied(maxIdx)
	}
	return resps
}

// advanceApplied monotonically raises the FSM applied index to idx. Raft applies
// log entries in index order on a single goroutine, so this is effectively a
// store; the max-guard is belt-and-suspenders against an out-of-order or stale
// idx (e.g. a Restore path) ever lowering it.
func (f *fsm) advanceApplied(idx uint64) {
	for {
		cur := f.appliedIndex.Load()
		if idx <= cur {
			return
		}
		if f.appliedIndex.CompareAndSwap(cur, idx) {
			return
		}
	}
}

// Snapshot freezes a consistent point-in-time copy of the FSM state.
//
// hashicorp/raft guarantees Snapshot is called on the SAME goroutine as Apply
// (and never concurrently with it), but FSMSnapshot.Persist runs on a separate
// goroutine CONCURRENTLY with later Apply calls. Serializing the cache + vector
// state HERE — synchronously on the Apply goroutine — captures everything as of
// a single log index; Persist then merely flushes the frozen bytes and can race
// Apply harmlessly. Reading live cache/vector pointers from Persist instead
// would walk shards and the vector store while Apply mutates them, producing a
// torn, cross-subsystem-inconsistent snapshot that corresponds to no single
// index. This mirrors cluster/meta_fsm.go's MetaFSM.Snapshot, which encodes the
// frozen state under its read lock so metaSnapshot.Persist only flushes bytes.
// The WASM registration blob is captured here too, on the same goroutine and for
// the same reason: it must correspond to the same log index as the cache and
// vector state, or a restored replica could hold state produced by an op it did
// not receive.
func (f *fsm) Snapshot() (hraft.FSMSnapshot, error) {
	// ###### A SNAPSHOT AFTER AN ABANDONED APPLY WOULD BE MISLABELLED ######
	//
	// The reason is entirely non-local: it is about raft's index bookkeeping, not
	// ours. runFSM sets its `lastIndex` from the LAST log of the batch it handed
	// us, unconditionally, and snapshot() stamps the request with that index —
	// which snapshot.go passes straight to snapshots.Create. After a classRetry
	// abandon (fsm.abandoned) state here reflects only the applied PREFIX, so a
	// snapshot taken now is durably labelled with an index whose tail never ran.
	// On restart raft sets lastApplied from the snapshot metadata and never
	// redelivers those entries: they are gone from this replica for good, with no
	// error anywhere to say so.
	//
	// Returning an error is the correct shape and costs nothing. runFSM's
	// snapshot() does req.respond(err) and takeSnapshot surfaces it as a logged
	// failure — the same handling ErrNothingNewToSnapshot gets — and this can only
	// happen while the Store is closing, where declining to snapshot is free.
	if f.abandoned.Load() {
		return nil, errors.New("shard: refusing to snapshot: an apply was abandoned mid-block, " +
			"so raft's index is ahead of this FSM's and the snapshot would be labelled with entries it does not contain")
	}
	var wasmBlob []byte
	if f.wasmSnapshot != nil {
		wasmBlob = f.wasmSnapshot()
	}
	data, err := serializeSnapshot(f.cache, f.vectors, f.AppliedIndex(), wasmBlob)
	if err != nil {
		return nil, err
	}
	return &fsmSnapshot{data: data}, nil
}

// Restore replaces the FSM state with the contents of the reader and advances
// the applied-index tracker to the index recorded in the snapshot. Without this
// a snapshot-restored follower (whose appliedIndex would otherwise stay 0, since
// heap-mode cache.AppliedIndex() is always 0) under-reports its frontier to the
// linearizable barrier — a latency/efficiency regression and a footgun. Mirrors
// cluster/meta_fsm.go's MetaFSM.Restore (advanceApplied(s.LastIndex)).
func (f *fsm) Restore(rc io.ReadCloser) error {
	appliedIndex, err := restoreSnapshot(f.cache, f.vectors, f.wasmRestore, rc)
	if err != nil {
		return err
	}
	// advanceApplied is monotonic-max, so a stale/old (pre-v3, index 0) snapshot
	// can never lower an already-higher frontier. SetAppliedIndex persists it into
	// the mmap header in durable mode (no-op for heap shards), keeping warm-restart
	// skip consistent with the restored state.
	f.advanceApplied(appliedIndex)
	f.cache.SetAppliedIndex(appliedIndex, f.durable)

	// The KV record index is NOT a durability unit — it is in no snapshot, no
	// WAL and no log — so a restore leaves it describing the keyspace that was
	// just replaced. restoreSnapshot REFILLS THE EXISTING cache (it does not
	// install a new one), so the Set to rebuild is the one already wired to
	// this cache's onRemove hook: rebuild it in place, never re-create it, or
	// the Store's read-only dispatcher would keep pointing at the old one.
	//
	// Rebuild empties every posting, replays the restored keyspace through it
	// and only then republishes readiness, so no query can be served from a
	// half-filled index. It walks nothing when no definition is installed,
	// which is the normal case for a restore (definitions arrive from the meta
	// log). Safe here for locking too: we hold no cache lock, and Rebuild does
	// not hold the index lock across the walk.
	if err := ops.RebuildKVIndex(f.tx.KVIndex(), f.cache); err != nil {
		// The walk was cut short (the cache closed underneath it). kvindex
		// published nothing, so the index stays building and queries get a
		// retryable refusal rather than a silently short answer. Not fatal to the
		// restore, which is about the cache, not the derived index.
		slog.Warn("kv index rebuild after a snapshot restore did not finish; the index stays building until it is walked again",
			"component", "shard", "err", err)
	}
	return nil
}
