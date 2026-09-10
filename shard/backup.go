// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft"
)

// ErrNothingToBackup reports that this shard has no state to snapshot yet (an
// empty Raft shard that has applied nothing). The cluster backup layer treats it
// as "skip this shard", never as a failure.
var ErrNothingToBackup = errors.New("shard: nothing to back up (empty shard)")

// restoreTimeout bounds a Raft-mode disaster-recovery Restore (the leader
// install + the no-op that confirms followers faulted-and-installed).
const restoreTimeout = 30 * time.Second

// BackupSnapshot returns a CONSISTENT, point-in-time serialization of this
// shard's full FSM state — the cache KV set (key/value/absolute-expiry) plus all
// vector collections — as a single RSST blob (the exact wire format
// restoreSnapshot / RestoreSnapshot consume), together with the applied index
// stamped into it. This is the per-shard DR unit: it is strictly MORE complete
// than the single-node vector-only backup (it also captures the KV cache and the
// applied index), and it is internally point-in-time (cache and vectors at the
// same logical index, no torn read).
//
// The consistency guarantee is mode-specific — both routes serialize where no
// Apply can race:
//
//   - RAFT: route through hashicorp/raft's snapshot machinery (see
//     raft.Node.BackupSnapshot). fsm.Snapshot() runs serializeSnapshot on the FSM
//     apply goroutine, which raft never runs concurrently with Apply, so the blob
//     is torn-free. An empty shard (nothing applied) reports ErrNothingToBackup.
//
//   - PB: there is no raft goroutine to quiesce, so serialize under the pbisr
//     engine's RunExclusive, which holds BOTH engine locks (writeMu excludes the
//     primary Propose apply path, e.mu excludes the backup Receive apply path —
//     the only two Applier.Apply sites), proving no torn snapshot under concurrent
//     PB applies. See pbisr.Engine.RunExclusive for the full argument.
//
// SWEEPER vs SNAPSHOT invariant (why the TTL sweepers do not break the crux):
// serializeSnapshot's cache.Iterate and vector SnapshotAll take the store-internal
// READ locks that the background TTL sweepers write-lock against (cache sweeper;
// vector/hnsw.go:78 "RLock for reads, Lock for inserts/deletes/tombstone reclaim"),
// so a sweep and the serialization walk never physically interleave. And a sweeper
// only reclaims already-EXPIRED (logically-absent) entries, so removing them does
// not change the logical point-in-time state a snapshot captures (a snapshot taken
// just before vs just after a sweep is logically identical — see #4 Phase B's
// replicated-expiry determinism). CAVEAT: this rests on sweepers reclaiming ONLY
// expired entries under the store lock. A FUTURE non-expiry background reclaimer
// (e.g. graph compaction beyond tombstone/expiry reclaim) MUST hold the same
// collection write lock the snapshot read path locks against, or it would break
// this invariant — the quiesce (writeMu+e.mu / raft goroutine) fences the APPLY
// path, not an independent background mutator.
//
// ONE-WAY UPGRADE. The archive is a serializeSnapshot blob, which always carries
// the CURRENT snapshotVersion, and a reader rejects any version above its own.
// A backup is therefore restorable only by a binary at least as new as the one
// that wrote it — there is no downgrade and no negotiation. Retaining archives
// across a format bump does not retain the ability to restore them on the old
// binary. See the ONE-WAY UPGRADE note in snapshot.go.
func (s *Store) BackupSnapshot(ctx context.Context) (data []byte, appliedIndex uint64, err error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if s.cfg.ReplicationMode == ReplicationModePB {
		pr, ok := s.raft.(*pbReplicator)
		if !ok {
			return nil, 0, fmt.Errorf("shard: BackupSnapshot: PB mode but replicator is %T", s.raft)
		}
		var serErr error
		pr.e.RunExclusive(func(frontier uint64) {
			appliedIndex = frontier
			// Carry dynamic WASM registrations too: a disaster-recovery restore
			// into a fresh shard must reproduce the op registry the backed-up
			// state was produced by, or the restored replicas fail closed on the
			// first invocation of a dynamically-registered op.
			var wasmBlob []byte
			if s.fsm != nil && s.fsm.wasmSnapshot != nil {
				wasmBlob = s.fsm.wasmSnapshot()
			}
			data, serErr = serializeSnapshot(s.cache, s.vectors, frontier, wasmBlob)
		})
		if serErr != nil {
			return nil, 0, fmt.Errorf("shard: BackupSnapshot (pb): %w", serErr)
		}
		return data, appliedIndex, nil
	}

	// Raft mode (default). The replicator is a *raft.Node.
	rn, ok := s.raft.(*raft.Node)
	if !ok {
		return nil, 0, fmt.Errorf("shard: BackupSnapshot: raft mode but replicator is %T", s.raft)
	}
	data, appliedIndex, err = rn.BackupSnapshot()
	if errors.Is(err, raft.ErrNoSnapshot) {
		return nil, 0, ErrNothingToBackup
	}
	if err != nil {
		return nil, 0, fmt.Errorf("shard: BackupSnapshot (raft): %w", err)
	}
	return data, appliedIndex, nil
}

// RestoreSnapshot installs a BackupSnapshot blob into this shard as a
// disaster-recovery bootstrap snapshot. It is the inverse of BackupSnapshot and
// must be called during restore into a FRESH (empty) shard.
//
//   - RAFT: the blob is installed on the shard LEADER via hashicorp/raft's
//     Restore primitive, which faults-and-installs it onto every follower — so a
//     same-topology restore only needs to run here on whichever owner won the
//     fresh election. A non-leader returns raft.ErrNotLeader (the caller skips it;
//     replication delivers the data).
//
//   - PB: there is no raft to stream, so every owner installs the blob directly
//     into its own cache+vectors (under the engine quiesce for safety). The
//     control plane re-seeds leases/ISR at a higher epoch afterward; the applied
//     index in the blob is only a floor.
func (s *Store) RestoreSnapshot(ctx context.Context, data []byte, appliedIndex uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.cfg.ReplicationMode == ReplicationModePB {
		pr, ok := s.raft.(*pbReplicator)
		if !ok {
			return fmt.Errorf("shard: RestoreSnapshot: PB mode but replicator is %T", s.raft)
		}
		var rErr error
		pr.e.RunExclusive(func(_ uint64) {
			rc := io.NopCloser(bytes.NewReader(data))
			var wasmRestore func([]byte) error
			if s.fsm != nil {
				wasmRestore = s.fsm.wasmRestore
			}
			_, rErr = restoreSnapshot(s.cache, s.vectors, wasmRestore, rc)
			if rErr == nil {
				// The KV record index is derived state — in no snapshot, no WAL
				// and no log — so this install has just replaced the keyspace it
				// describes, leaving it with postings for keys that are gone and
				// NONE for the keys that arrived. Rebuild it against the same Set
				// (restoreSnapshot refills the EXISTING cache, so the Set already
				// wired to that cache's onRemove hook is the one to refill),
				// inside the exclusive section so no write lands between the
				// refill and the rebuild. It walks nothing when no definition is
				// installed. The raft branch below reaches fsm.Restore, which
				// does the same thing.
				// Inside RunExclusive: a Close cannot interleave, so the walk
				// cannot be cut short. If it ever were, kvindex publishes nothing
				// and the index stays building — a retryable refusal, not a short
				// answer.
				if rebuildErr := ops.RebuildKVIndex(s.kvIdx, s.cache); rebuildErr != nil {
					slog.Warn("kv index rebuild after a PB snapshot install did not finish; the index stays building until it is walked again",
						"component", "shard", "err", rebuildErr)
				}
			}
		})
		if rErr != nil {
			return fmt.Errorf("shard: RestoreSnapshot (pb): %w", rErr)
		}
		return nil
	}

	rn, ok := s.raft.(*raft.Node)
	if !ok {
		return fmt.Errorf("shard: RestoreSnapshot: raft mode but replicator is %T", s.raft)
	}
	if err := rn.Restore(data, appliedIndex, restoreTimeout); err != nil {
		return err // raft.ErrNotLeader passes through so the caller can skip a follower
	}
	return nil
}
