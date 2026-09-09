// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"io"
	"sync/atomic"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard/pbisr"
	"github.com/rostamlabs/rostam/vector"
)

// pbSnapshotStore is the production pbisr.SnapshotStore: it wires the
// engine's snapshot transfer to this shard's real FSM — the cache KV set, the
// vector CollectionStore, and the dynamic WASM registrations — plus the durable
// PB frontier and the poison fence.
//
// IT REUSES THE EXISTING PRIMITIVES RATHER THAN REINVENTING THEM. serializeSnapshot
// and restoreSnapshot are the same functions BackupSnapshot/RestoreSnapshot use,
// and restoreSnapshot ALREADY does exactly the thing this stage needs and nothing
// more: it WIPES the pre-restore key set with DURABLE TOMBSTONES before replaying.
// That is the "discard the divergent tail, do not merge" primitive, already
// written and already reasoned about for warm-restart durability (without the
// tombstones a pre-restore ghost could survive the rebuild and even out-rank the
// restored copy).
//
// WHAT IT ADDS is the two things a transfer needs that a disaster-recovery restore
// did not: the poison fence around the non-atomic wipe+replay, and the durable
// frontier write that makes the target's reported identity match the installed
// state — including resetting the amortised stamper, which would otherwise flush
// the PRE-install pending pair straight over the value we just wrote.
type pbSnapshotStore struct {
	cache   *cache.Cache
	vectors *vector.CollectionStore
	fsm     *fsm
	stamp   *pbFrontierStamper
	fence   *pbRestoreFence

	// installs counts completed installs (test/introspection).
	installs atomic.Uint64
}

var _ pbisr.SnapshotStore = (*pbSnapshotStore)(nil)

func newPBSnapshotStore(c *cache.Cache, vectors *vector.CollectionStore, f *fsm, stamp *pbFrontierStamper, fence *pbRestoreFence) *pbSnapshotStore {
	return &pbSnapshotStore{cache: c, vectors: vectors, fsm: f, stamp: stamp, fence: fence}
}

// SnapshotFSM serializes the whole FSM into one RSST blob — byte-for-byte the
// format BackupSnapshot produces and RestoreSnapshot consumes, so a transfer blob
// and a DR archive are the same artifact.
//
// The engine calls this holding writeMu+e.mu, which excludes BOTH Applier.Apply
// sites, so the cache walk and the vector SnapshotAll observe the same logical
// point (the identical argument BackupSnapshot's PB branch rests on). The dynamic
// WASM registrations ride along for the same reason a DR restore carries them: a
// target that installs a snapshot never applies the __register_wasm__ entries the
// snapshot replaced, and would fail closed on the first invocation its peers
// execute normally.
func (p *pbSnapshotStore) SnapshotFSM(appliedIndex uint64) ([]byte, error) {
	var wasmBlob []byte
	if p.fsm != nil && p.fsm.wasmSnapshot != nil {
		wasmBlob = p.fsm.wasmSnapshot()
	}
	return serializeSnapshot(p.cache, p.vectors, appliedIndex, wasmBlob)
}

// InstallFSM discards local state and installs blob. Called with writeMu+e.mu
// held, INSIDE the poison fence — which is what makes its documented
// non-atomicity safe rather than merely acknowledged.
//
// The trailer's appliedIndex is cross-checked against the frontier the wire
// declared. They are written by the same primary in the same critical section, so
// a mismatch means the blob and its header disagree about which state this is —
// cheap to check, and refusing is strictly better than installing a state under
// the wrong identity (which is the one error the log-matching check cannot see).
func (p *pbSnapshotStore) InstallFSM(blob []byte) error {
	var wasmRestore func([]byte) error
	if p.fsm != nil {
		wasmRestore = p.fsm.wasmRestore
	}
	rc := io.NopCloser(bytes.NewReader(blob))
	_, err := restoreSnapshot(p.cache, p.vectors, wasmRestore, rc)
	if err != nil {
		return err
	}
	// The KV record index is derived state, carried in no snapshot, so this
	// install just replaced the keyspace it describes: stale postings for keys
	// that are gone, and NONE for the keys that arrived. Rebuild it against the
	// same Set — restoreSnapshot refills the EXISTING cache, so the Set already
	// wired to that cache's onRemove hook is the one to refill.
	//
	// Here rather than in CommitInstall because this is the call that holds
	// writeMu+e.mu, so the refill and the rebuild are one critical section with
	// no apply able to slip between them. Lock order is unchanged (cache, then
	// index; Rebuild does not hold the index lock across its walk), and with no
	// definition installed — the common case — it walks nothing.
	if p.fsm != nil {
		ops.RebuildKVIndex(p.fsm.tx.KVIndex(), p.cache)
	}
	p.installs.Add(1)
	return nil
}

// BeginInstall raises the durable poison fence. Called OFF the engine locks (it
// is an fsync).
func (p *pbSnapshotStore) BeginInstall(seq, epoch uint64) error {
	return p.fence.raise(seq, epoch)
}

// CommitInstall persists the frontier and THEN lowers the fence, in that order.
//
// The order is the whole safety property and it is asymmetric on purpose. A crash
// between the two re-poisons a node whose state is actually fine, costing one more
// transfer. The reverse order would un-poison a node still carrying its
// PRE-install watermark over POST-install state — and for the diverged target this
// path exists to repair, that stale watermark is HIGHER than the installed
// frontier, i.e. an OVER-report, which is precisely what makes log matching
// certify a divergent append.
//
// installFrontier (not the plain SetPBFrontier) is required: the amortised stamper
// still holds the PRE-install pending pair, and its next tick would flush that
// straight back over the value written here. installFrontier rebases the stamper
// and does the crash-ordered write under the stamper's own flush lock, so no
// in-flight flush can land behind it.
func (p *pbSnapshotStore) CommitInstall(seq, epoch uint64) error {
	if p.stamp != nil {
		p.stamp.installFrontier(seq, epoch)
	} else {
		p.cache.SetPBFrontier(seq, epoch)
	}
	return p.fence.lower()
}

// AbortInstall lowers the fence on the one path where the FSM is provably
// untouched (the engine observed a newer epoch between BeginInstall and taking
// the install locks). Every other abort leaves the fence raised on purpose.
func (p *pbSnapshotStore) AbortInstall() error { return p.fence.lower() }

// InstallPending reports a raised fence — the signal that this node booted (or
// aborted) mid-install and must refuse to serve until re-snapshotted.
func (p *pbSnapshotStore) InstallPending() bool {
	_, _, up := p.fence.raised()
	return up
}
