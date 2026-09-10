// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// Snapshot wire format:
//
//   [magic 'RSST' 4B][version u8][entries...][?vectorSection][?wasmSection][trailer]
//
// Each entry: [keyLen u16][key][valLen u32][val][expiryMs u64]
//
// Vector section (version >= 2 only): [vecLen u32][vecBlob] — the
// CollectionStore snapshot of all vector collections, so replicated vector state
// survives Raft log truncation / new-node catch-up. The trailer's entryCount
// bounds the cache entries, so the vector section is unambiguously the bytes
// following the last cache entry. In v2..v4 it is OMITTED entirely when the
// shard has no vector store; from v5 it is always present (vecLen 0 when there
// is none), which is what lets a further section follow it unambiguously.
//
// WASM section (version >= 5 only): [wasmLen u32][wasmBlob] — the node's
// DYNAMIC op registrations, opaque to this package (see Config.WASMSnapshot).
// It exists because the ops registry is mutable at runtime: a replica that
// installs a snapshot never applies the __register_wasm__ entries the snapshot
// replaced, so without carrying them it would be permanently missing those ops
// and would fail closed (classFatal ErrOpNotRegistered) on the first invocation
// its peers execute normally. That is deterministic for every
// AddShardOwner-joined replica, not a race.
//
// The blob carries the module BYTES, not a reference to them. A snapshot is the
// only channel an InstallSnapshot receiver has, so a reference would have to be
// resolved out-of-band, and a peer that is slow or gone would leave the replica
// unable to apply entries it has already accepted — strictly worse than the
// storage cost. The cost is also not new: the same bytes are already in this
// group's Raft LOG (the registration is replicated to every group), and a
// snapshot REPLACES log, so carrying them lets that log be truncated. The
// residual duplication across groups is bounded by capping the module size
// accepted for a dynamic registration well below the frame limit — see
// cluster.maxDynamicWASMBytes.
//
// Trailer (v1/v2): [magic 'TRLR' 4B][entryCount u64][crc32 over body u32]
// Trailer (v3):    [magic 'TRLR' 4B][entryCount u64][appliedIndex u64][crc32 over body u32]
// Trailer (v4+):   [magic 'TRLR' 4B][entryCount u64][appliedIndex u64][stampMs u64][crc32 over body u32]
//
// v3 appends the FSM applied index into the trailer so a snapshot-restored
// follower advances its applied-index tracker to the snapshot index instead of
// under-reporting 0 (mirrors cluster/meta_fsm.go's State.LastIndex). The
// appliedIndex sits BEFORE the CRC field and is itself covered by the CRC.
//
// v4 appends the cache's LOGICAL CLOCK (cache.LastAppliedStampMs) for the same
// class of reason, one layer down. The clock only advances on the STAMPED apply
// path, and a snapshot installs through PutAbs, which does not advance it — so a
// node whose committed state arrives entirely by snapshot used to hold clock 0.
// Both deterministic TTL reclaimers (the B3b logical sweeper and cold compaction
// at shard open, cache/compact.go) are safe only because a future committed
// write's stamp dominates every replica's persisted clock, and that in turn holds
// only because a leader clamps new stamps to >= its OWN clock. A leader at 0
// would stamp at bare wall time and could land below a peer's persisted clock,
// retroactively invalidating reclamation that peer already performed. Carrying
// the clock and folding it in on restore (cache.AdvanceAppliedStamp) closes it.
// Like appliedIndex, stampMs sits before the CRC field and is covered by it.
//
// ONE-WAY UPGRADE (a deployment constraint, not a bug). This format is
// backward-compatible for READS only: parseSnapshot accepts any version <=
// snapshotVersion, but serializeSnapshot ALWAYS writes the current
// snapshotVersion, and a reader rejects version > snapshotVersion outright. So a
// mixed-version fleet is safe in exactly one direction. Once an UPGRADED node
// leads a group, takes a snapshot and compacts its log, an OLD follower that
// needs catch-up receives an InstallSnapshot it cannot parse and rejects it —
// permanently, since the log entries it would otherwise replay are gone. There
// is no negotiation and no downgrade path.
//
// Consequences for operators: upgrade every replica of a group before the new
// leader is allowed to snapshot+compact, and treat a version bump here as a
// no-rollback step. The same applies to object-store BACKUPS (see backup.go):
// an archive written by a new binary cannot be restored by an old one.
// This is the snapshot-side twin of the rolling-upgrade note on
// ErrOpNotRegistered in apply_class.go — that one halts loudly on a committed
// entry, this one fails the catch-up path instead.

var snapshotMagic = []byte{'R', 'S', 'S', 'T'}
var trailerMagic = []byte{'T', 'R', 'L', 'R'}

// v1 = cache only; v2 adds the vector section; v3 adds appliedIndex to the
// trailer; v4 adds the cache's logical clock (lastAppliedStampMs); v5 makes the
// vector section unconditional and appends the WASM registration section.
const snapshotVersion = 5

// trailerLen returns the on-wire trailer size for a given snapshot version.
// v1/v2: magic(4)+count(8)+crc(4); v3: +appliedIndex(8); v4+: +stampMs(8).
func trailerLen(version byte) int {
	switch {
	case version >= 4:
		return 4 + 8 + 8 + 8 + 4
	case version == 3:
		return 4 + 8 + 8 + 4
	default:
		return 4 + 8 + 4
	}
}

var errBadSnapshot = errors.New("shard: bad snapshot header")
var errSnapshotCRC = errors.New("shard: snapshot CRC mismatch")

// fsmSnapshot is the raft.FSMSnapshot. It holds a FROZEN, point-in-time
// serialization of the cache + vector state produced synchronously on the
// Raft FSM goroutine (see serializeSnapshot / fsm.Snapshot). Persist merely
// flushes these bytes, so it can run concurrently with Apply without ever
// observing a torn, cross-subsystem-inconsistent state (mirrors the
// cluster/meta_fsm.go metaSnapshot pattern).
type fsmSnapshot struct {
	data []byte
}

// serializeSnapshot freezes a consistent point-in-time copy of the cache and
// (v2+) vector state into a single buffer, including the trailer. It MUST be
// called on the Raft FSM goroutine (from fsm.Snapshot), which hashicorp/raft
// guarantees never runs concurrently with Apply — so the cache walk and the
// vector SnapshotAll observe the same log index. appliedIndex is stamped into
// the v3 trailer so a restored follower can advance its applied-index tracker.
// wasmBlob (may be nil) is the opaque dynamic-registration payload; see the
// WASM section note on the wire format above.
func serializeSnapshot(c *cache.Cache, vectors *vector.CollectionStore, appliedIndex uint64, wasmBlob []byte) ([]byte, error) {
	var buf bytes.Buffer
	crc := crc32.NewIEEE()
	w := io.MultiWriter(&buf, crc)

	if _, err := w.Write(snapshotMagic); err != nil {
		return nil, fmt.Errorf("snapshot magic: %w", err)
	}
	if _, err := w.Write([]byte{snapshotVersion}); err != nil {
		return nil, fmt.Errorf("snapshot version: %w", err)
	}

	var count uint64
	var entryBuf [2 + 4 + 8]byte
	var iterErr error
	c.Iterate(func(key, value []byte, expiryMs uint64) bool {
		binary.BigEndian.PutUint16(entryBuf[0:2], uint16(len(key)))   //nolint:gosec // key bounded by upstream limits
		binary.BigEndian.PutUint32(entryBuf[2:6], uint32(len(value))) //nolint:gosec // value bounded by upstream limits
		binary.BigEndian.PutUint64(entryBuf[6:14], expiryMs)
		if _, err := w.Write(entryBuf[0:2]); err != nil {
			iterErr = err
			return false
		}
		if _, err := w.Write(key); err != nil {
			iterErr = err
			return false
		}
		if _, err := w.Write(entryBuf[2:6]); err != nil {
			iterErr = err
			return false
		}
		if _, err := w.Write(value); err != nil {
			iterErr = err
			return false
		}
		if _, err := w.Write(entryBuf[6:14]); err != nil {
			iterErr = err
			return false
		}
		count++
		return true
	})
	if iterErr != nil {
		return nil, fmt.Errorf("snapshot body: %w", iterErr)
	}

	// Vector section: the full CollectionStore snapshot, length-prefixed, appended
	// to the CRC'd body after the cache entries. From v5 the length prefix is
	// ALWAYS written (0 when the shard has no vector store) so the WASM section
	// that follows has an unambiguous start — in v2..v4 the section's presence
	// was implied by "are there bytes left", which cannot carry two sections.
	var vbuf bytes.Buffer
	if vectors != nil {
		if err := vectors.SnapshotAll(&vbuf); err != nil {
			return nil, fmt.Errorf("snapshot vectors: %w", err)
		}
	}
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(vbuf.Len())) //nolint:gosec
	if _, err := w.Write(lenBuf[:]); err != nil {
		return nil, fmt.Errorf("snapshot vec len: %w", err)
	}
	if _, err := w.Write(vbuf.Bytes()); err != nil {
		return nil, fmt.Errorf("snapshot vec body: %w", err)
	}

	// WASM section (v5+): always present, length-prefixed, possibly empty.
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(wasmBlob))) //nolint:gosec
	if _, err := w.Write(lenBuf[:]); err != nil {
		return nil, fmt.Errorf("snapshot wasm len: %w", err)
	}
	if _, err := w.Write(wasmBlob); err != nil {
		return nil, fmt.Errorf("snapshot wasm body: %w", err)
	}

	// Trailer: magic, entryCount, appliedIndex (v3), stampMs (v4), then crc over
	// everything written so far (which includes both). Write appliedIndex and
	// stampMs through the CRC, then append magic+crc raw.
	//
	// stampMs is the cache's logical clock at this instant. serializeSnapshot runs
	// on the FSM goroutine, which never runs concurrently with Apply, so the clock
	// it reads is exactly the one that produced the entries above.
	var trailerBuf [4 + 8 + 8 + 8 + 4]byte
	copy(trailerBuf[0:4], trailerMagic)
	binary.BigEndian.PutUint64(trailerBuf[4:12], count)
	binary.BigEndian.PutUint64(trailerBuf[12:20], appliedIndex)
	binary.BigEndian.PutUint64(trailerBuf[20:28], c.LastAppliedStampMs())
	// appliedIndex + stampMs are CRC-covered; fold them in before sealing.
	if _, err := crc.Write(trailerBuf[12:28]); err != nil {
		return nil, fmt.Errorf("snapshot trailer crc: %w", err)
	}
	bodyCRC := crc.Sum32()
	binary.BigEndian.PutUint32(trailerBuf[28:32], bodyCRC)
	buf.Write(trailerBuf[:])
	return buf.Bytes(), nil
}

func (s *fsmSnapshot) Persist(sink hraft.SnapshotSink) error {
	if _, err := sink.Write(s.data); err != nil {
		_ = sink.Cancel()
		return fmt.Errorf("snapshot persist: %w", err)
	}
	return sink.Close()
}

func (s *fsmSnapshot) Release() {}

// restoreSnapshot wipes the cache (and vectors) and replays the snapshot stream.
// It returns the FSM applied index recorded in the snapshot trailer (0 for
// pre-v3 snapshots that did not record one) so the caller can advance its
// applied-index tracker to the snapshot index.
// wasmRestore (may be nil) receives the v5 WASM section; see Config.WASMRestore.
func restoreSnapshot(c *cache.Cache, vectors *vector.CollectionStore, wasmRestore func([]byte) error, rc io.ReadCloser) (uint64, error) {
	defer func() { _ = rc.Close() }()

	all, err := io.ReadAll(rc)
	if err != nil {
		return 0, fmt.Errorf("snapshot read: %w", err)
	}
	if len(all) < 5+4+8+4 {
		return 0, errBadSnapshot
	}
	if string(all[0:4]) != string(snapshotMagic) {
		return 0, errBadSnapshot
	}
	version := all[4]
	if version < 1 || version > snapshotVersion {
		return 0, fmt.Errorf("shard: snapshot version %d unsupported", version)
	}
	tlen := trailerLen(version)
	if len(all) < 5+tlen {
		return 0, errBadSnapshot
	}
	trailerStart := len(all) - tlen
	if string(all[trailerStart:trailerStart+4]) != string(trailerMagic) {
		return 0, errBadSnapshot
	}
	expectedCount := binary.BigEndian.Uint64(all[trailerStart+4 : trailerStart+12])
	crcStart := trailerStart + tlen - 4
	expectedCRC := binary.BigEndian.Uint32(all[crcStart : crcStart+4])

	// CRC covers the body (everything before the trailer) plus, for v3+, the
	// appliedIndex field and, for v4+, the logical clock — matching
	// serializeSnapshot, which folds both into the running CRC but leaves the
	// trailer magic + entryCount uncovered.
	crc := crc32.NewIEEE()
	_, _ = crc.Write(all[0:trailerStart])
	var appliedIndex, stampMs uint64
	if version >= 3 {
		appliedIndex = binary.BigEndian.Uint64(all[trailerStart+12 : trailerStart+20])
		_, _ = crc.Write(all[trailerStart+12 : trailerStart+20])
	}
	if version >= 4 {
		stampMs = binary.BigEndian.Uint64(all[trailerStart+20 : trailerStart+28])
		_, _ = crc.Write(all[trailerStart+20 : trailerStart+28])
	}
	if crc.Sum32() != expectedCRC {
		return 0, errSnapshotCRC
	}

	keysToDelete := [][]byte{}
	c.Iterate(func(key, _ []byte, _ uint64) bool {
		k := make([]byte, len(key))
		copy(k, key)
		keysToDelete = append(keysToDelete, k)
		return true
	})
	// Wipe the pre-restore key set. On a PERSISTENT shard each Del also appends a
	// durable tombstone, which is what makes the wipe survive a later warm restart:
	// without one, restore left the pre-restore entries physically on the page and
	// the next rebuild re-indexed them, so a snapshot install was not durable
	// (a pre-restore ghost could even out-rank the restored copy). Appending needs
	// room, so the error is propagated rather than swallowed — a half-wiped restore
	// must not be reported as a success.
	for _, k := range keysToDelete {
		if _, err := c.Del(k); err != nil {
			return 0, fmt.Errorf("snapshot restore: clearing key: %w", err)
		}
	}

	var count uint64
	body := all[5:trailerStart]
	off := 0
	for count < expectedCount {
		if off+2 > len(body) {
			return 0, errBadSnapshot
		}
		klen := int(binary.BigEndian.Uint16(body[off : off+2]))
		off += 2
		if off+klen > len(body) {
			return 0, errBadSnapshot
		}
		key := body[off : off+klen]
		off += klen
		if off+4 > len(body) {
			return 0, errBadSnapshot
		}
		vlen := int(binary.BigEndian.Uint32(body[off : off+4]))
		off += 4
		if off+vlen > len(body) {
			return 0, errBadSnapshot
		}
		val := body[off : off+vlen]
		off += vlen
		if off+8 > len(body) {
			return 0, errBadSnapshot
		}
		expiry := binary.BigEndian.Uint64(body[off : off+8])
		off += 8

		// #4 Phase B / B1: install the ABSOLUTE expiry verbatim via PutAbs, and do
		// NOT skip logically-expired entries by the installer's wall clock. The old
		// restore recomputed ttl = expiry - now and Put it (re-stamping against this
		// node's wall clock) and dropped entries with expiry <= now — BOTH
		// reintroduced nondeterminism: two followers installing the SAME snapshot at
		// different instants would produce different state (different re-derived
		// expiries, or a different set of surviving keys). Installing the recorded
		// absolute expiry unchanged makes the post-install state logically
		// byte-identical (identical key/value/exp set) on every follower. A logically-expired key that survives is filtered on read
		// and reclaimed by a later committed write or the next snapshot compaction —
		// consistent with the B3a replicated read/sweeper policy.
		if err := c.PutAbs(key, val, expiry); err != nil {
			return 0, fmt.Errorf("snapshot replay put: %w", err)
		}
		count++
	}

	// #4 Phase B: adopt the snapshot's LOGICAL CLOCK. PutAbs above installs
	// absolute expiries without advancing the clock (correctly — an absolute
	// expiry says nothing about when it was applied), so a node whose committed
	// state arrives entirely by snapshot would otherwise sit at 0 while its peers
	// hold the leader's clock. That gap is what makes a subsequently-elected
	// leader stamp below a peer's persisted clock and retroactively invalidate the
	// reclamation (B3b sweep, cold compaction) that peer already performed. The
	// fold is a MAX, so it can only move the clock forward, and a pre-v4 snapshot
	// carries 0 and is a no-op — the old behavior.
	c.AdvanceAppliedStamp(stampMs)

	// Cross-check the trailer's entryCount against where the cache entries
	// actually end. The cache/vector split is implied solely by entryCount
	// (there is no structural section delimiter), so a CRC-consistent but
	// malformed snapshot whose entryCount under-counts the real entries would
	// otherwise let leftover cache bytes be reinterpreted as the vector section
	// (a bogus vlen handed to RestoreAll). For v2+ the remainder must be EITHER
	// empty (no vectors) or a well-formed [vecLen u32][vecBlob] that consumes
	// exactly to the end of the body. v1 has no vector section, so off must land
	// exactly at the body end.
	if version < 2 {
		if off != len(body) {
			return 0, errBadSnapshot
		}
		return appliedIndex, nil
	}

	if version < 5 {
		// v2..v4: the vector section is present only when there are bytes left,
		// and must consume the remainder exactly. Any leftover (e.g. a present
		// section with vectors==nil) means entryCount and the body disagree on
		// the section boundary.
		if off < len(body) && vectors != nil {
			if off+4 > len(body) {
				return 0, errBadSnapshot
			}
			vlen := int(binary.BigEndian.Uint32(body[off : off+4]))
			off += 4
			if off+vlen != len(body) {
				return 0, errBadSnapshot
			}
			if err := vectors.RestoreAll(bytes.NewReader(body[off : off+vlen])); err != nil {
				return 0, fmt.Errorf("snapshot vector restore: %w", err)
			}
			off += vlen
		}
		if off != len(body) {
			return 0, errBadSnapshot
		}
		return appliedIndex, nil
	}

	// v5: both sections are always present and length-delimited, so each is
	// bounded by its own prefix rather than by "the rest of the body".
	if off+4 > len(body) {
		return 0, errBadSnapshot
	}
	vlen := int(binary.BigEndian.Uint32(body[off : off+4]))
	off += 4
	if vlen < 0 || off+vlen > len(body) {
		return 0, errBadSnapshot
	}
	if vlen > 0 {
		if vectors == nil {
			// The snapshot carries vector state this Store cannot install. Failing
			// is the only honest outcome: silently dropping it would leave this
			// replica missing committed data while reporting the snapshot applied.
			return 0, errBadSnapshot
		}
		if err := vectors.RestoreAll(bytes.NewReader(body[off : off+vlen])); err != nil {
			return 0, fmt.Errorf("snapshot vector restore: %w", err)
		}
	}
	off += vlen

	if off+4 > len(body) {
		return 0, errBadSnapshot
	}
	wlen := int(binary.BigEndian.Uint32(body[off : off+4]))
	off += 4
	if wlen < 0 || off+wlen > len(body) {
		return 0, errBadSnapshot
	}
	if wlen > 0 {
		if wasmRestore == nil {
			// Same reasoning as the vector case: the snapshot says this group's
			// history registered ops that this Store has no way to install, and a
			// missing op is a fail-closed halt at the first invocation. Refuse the
			// restore rather than install a knowingly incomplete state.
			return 0, errBadSnapshot
		}
		if err := wasmRestore(body[off : off+wlen]); err != nil {
			return 0, fmt.Errorf("snapshot wasm restore: %w", err)
		}
	}
	off += wlen

	// Any unconsumed bytes indicate a count/body mismatch.
	if off != len(body) {
		return 0, errBadSnapshot
	}
	return appliedIndex, nil
}
