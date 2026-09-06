// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Write-ahead log for crash durability between checkpoints.
//
// A Persistent collection is durable only as of its last Flush (which writes the
// SavePersist sidecar + msyncs the mmap files). Ops applied since then live only
// in (un-synced) mmap pages and would be lost on a hard crash. With a WAL, each
// successful Insert/Delete is appended and fsync'd before the call returns, so
// on reopen the index is recovered by openPersist(checkpoint) + replaying the
// WAL tail. Flush rotates the WAL (truncates it) once the checkpoint subsumes it.
//
// Records are length+CRC framed so a torn final record from a crash mid-append
// is detected and ignored on replay — the last fsync'd record is the durability
// boundary. The log is logical (it stores the Insert/Delete operation, not graph
// edges); replay re-applies via the normal index paths, yielding a valid index
// containing every committed vector.

type walRecType uint8

const (
	walInsert walRecType = 1
	walDelete walRecType = 2
	// walSetPayload logs an in-place payload mutation as the RESULTING full
	// payload (op-agnostic: set-merge/overwrite/delete-keys/clear all reduce to
	// "this is the new payload"). Replay = SetMetadata + payloadIdx.reindex.
	//
	// BACKWARD-COMPAT CAVEAT: this tag is ADDITIVE (3, never reused). An OLD
	// replayer that predates it hits the default case in replayRecord and treats
	// the unknown tag as a torn/malformed record, STOPPING replay there — so any
	// insert/delete records written AFTER a payload-update in a dense WAL-mode
	// collection are lost on a downgraded binary. This only affects dense
	// WAL-mode collections (the cluster-Raft path is unaffected — Raft is the
	// durability authority there). Upgrade all binaries before writing payload
	// updates to a WAL-mode collection.
	walSetPayload walRecType = 3
)

// maxWALRecord bounds a single record's payload, rejecting a garbage/torn length
// during replay rather than allocating wildly.
const maxWALRecord = 64 << 20

// wal is an append-only operation log. The file WRITE (framing + os.Write) is
// serialized by mu (the file offset / append position is not concurrency-safe
// otherwise). The fsync, however, is GROUP-COMMITTED outside mu (see appendFramed)
// so concurrent appends share one fsync and a write can overlap the prior fsync.
//
// CONCURRENCY MODEL (Step-0 finding): today every appendFramed caller holds the
// per-collection opMu across {engine apply + WAL append} (dense Collection,
// NamedCollection, MultiVectorIndex), and each collection owns a DISTINCT wal
// (one openWAL per collection, never shared). So a single wal instance currently
// sees only ONE appender at a time — the group-commit machinery below is correct
// and ready, but degenerates to per-op fsync until a caller releases opMu before
// the commit-wait (the throughput follow-up). The machinery is written so it is
// arrival-safe the moment concurrency reaches one wal: no writer is ever woken by
// a Sync whose target was captured BEFORE that writer's bytes hit the file.
//
// The framing/fsync/truncate/replay-loop (appendFramed, replayFramed, truncate)
// is the family-AGNOSTIC core: it knows only opaque framed payloads. Per-family
// record codecs (dense's appendInsertStaged/.../replayWAL below; named/MV elsewhere)
// build their own record bytes and layer on this shared core.
//
// GROUP-COMMIT (batch-on-contention, leader-fsync), durability invariant: an ack
// MUST mean fsynced. Two phases:
//
//   - WRITE phase (under mu): append [len][crc][payload], then bump writeSeq (a
//     monotonic counter of "bytes durably positioned in the file, awaiting a
//     covering sync"). The writer records its own seq = the post-write writeSeq.
//   - COMMIT-WAIT phase (under syncMu, mu released so writes overlap the fsync):
//     the writer blocks until syncedSeq >= its seq, i.e. a Sync() whose target
//     was captured AT OR AFTER its bytes were written has completed. The first
//     waiter with no sync in flight becomes the SYNCER: under syncMu it captures
//     target = writeSeq (a snapshot of all bytes written so far — the leader's
//     fsync covers exactly these), releases syncMu, runs f.Sync() (NOT holding mu
//     OR syncMu, so other writers can keep appending), then re-takes syncMu, sets
//     syncedSeq = max(syncedSeq, target), and broadcasts. Followers whose seq <=
//     target are now satisfied (their bytes were written before target was
//     captured, so this Sync covered them) and return; the rest pick a new syncer
//     for the NEXT flight (their bytes arrived after this target was captured).
//
// ARRIVAL-SAFE RULE (the crux, mirrors shard/readindex_coalesce batch-then-capture):
// target is captured (under syncMu) as the writeSeq AT THAT MOMENT; the syncer's
// Sync covers every byte written before the capture. A writer's seq was assigned
// when ITS bytes were written (under mu). So syncedSeq >= seq  ⟺  the satisfying
// Sync's target was captured at-or-after this writer's bytes existed in the file
// ⟺ that Sync covered them. We NEVER wake a writer with a Sync captured before its
// write. truncate() (Flush rotation) also advances syncedSeq: after a checkpoint
// that subsumes the truncated log, every already-written op is durable via the
// checkpoint, so its commit-wait is satisfiable by the truncate's own Sync.
type wal struct {
	mu     sync.Mutex // serializes the file WRITE (append) only — NOT held across fsync
	f      *os.File
	noSync bool

	// Group-commit state. syncMu guards writeSeq/syncedSeq/syncing; cond signals
	// waiters when a Sync completes. A separate mutex from mu so a writer can hold
	// mu for the (fast) write and then contend only on syncMu for the commit-wait,
	// while the leader's f.Sync() runs holding NEITHER.
	syncMu    sync.Mutex
	cond      *sync.Cond
	writeSeq  uint64 // monotonic count of records written (bytes positioned in file)
	syncedSeq uint64 // highest writeSeq covered by a completed Sync (or truncate)
	syncing   bool   // a leader is currently running f.Sync()

	// poisoned is a TERMINAL latch tripped on the FIRST f.Sync() failure (in
	// commitWait or truncate). It exists because of fsyncgate: on a failed
	// fsync, the kernel may DISCARD the sync's dirty pages, and a SUBSEQUENT
	// f.Sync() on the same file can then return SUCCESS having synced nothing
	// (the post-4.13 Linux behavior; the PostgreSQL PANIC-on-fsync-failure
	// case). If the next writer became fsync leader and retried Sync() after a
	// failure, it could be acked on bytes the kernel already dropped — silently
	// breaking the ack-implies-fsynced invariant this file's doc promises.
	//
	// So the WAL fails CLOSED: once poisoned, every append/commit-wait entry
	// point returns ErrWALPoisoned rather than a false success, and no writer is
	// ever allowed to retry the failed Sync. It is cleared ONLY by a fresh
	// reopen (openWAL constructs a new struct → zero-value false), because only
	// a new file handle re-establishes a known-good durability path; a
	// truncate() that happens to Sync() successfully does NOT clear it (the
	// underlying device may still be bad — see truncate). Set once under the
	// leader's post-Sync syncMu; read lock-free via atomic so the hot append
	// path gates cheaply. Mirrors arena.poisoned's terminal-latch discipline.
	poisoned atomic.Bool

	// Test hooks (nil in production). onSync runs under syncMu just before the
	// post-Sync broadcast (counts completed Sync/truncate flushes — group-commit
	// observability). beforeSync runs in the leader AFTER target is captured and
	// syncMu is released but BEFORE f.Sync(); a test can block here to deterministically
	// park followers behind one in-flight flight (proving batch-on-contention).
	onSync     func()
	beforeSync func()
}

// ErrWALPoisoned is returned by every append/commit-wait entry point once the
// WAL has been poisoned by a prior f.Sync() failure (fsyncgate — see the
// poisoned field). The WAL accepts no further writes or acks until it is
// reopened; the caller must surface this so a delete/insert is reported failed
// rather than falsely acked on bytes that may never have reached disk.
var ErrWALPoisoned = errors.New("vector: wal poisoned by a prior fsync failure")

// openWAL opens (creating if absent) the WAL at path for appending. Replay the
// existing contents with replayWAL BEFORE calling this if recovering. The
// returned wal starts UN-poisoned (zero-value latch) — a fresh open is the only
// way a poisoned WAL recovers (see the poisoned field).
func openWAL(path string, noSync bool) (*wal, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o600) //nolint:gosec // caller-supplied path
	if err != nil {
		return nil, err
	}
	w := &wal{f: f, noSync: noSync}
	w.cond = sync.NewCond(&w.syncMu)
	return w, nil
}

// appendInsertStaged logs a successful Insert. ttl is stored as milliseconds;
// replayed TTLs are measured from recovery time (a bounded inaccuracy after a crash).
//
// version is the point's resulting version, appended as a TRAILING optional block
// AFTER the sparse vector so the encoding is byte-identical to the pre-CAS format
// when version is 0 (writeOptVersion writes a single 0 flag byte) and
// FORWARD/BACKWARD safe: a replayer predating the block stops after the sparse
// vector (per-record framing isolates trailing bytes); a NEW replayer reading an
// OLD record (no trailing block) treats the missing block as version 0 (replay
// then defaults a fresh insert to 1).
// keyExpires carries the resulting per-key ABSOLUTE unix-millis payload deadlines
// (key -> deadline) the engine computed at insert. It is appended as a TRAILING
// optional block AFTER the version block (so an OLD record with only the version
// block replays unchanged, and an OLD replayer stops after the version per the
// per-record framing) — byte-identical to the pre-per-key-TTL encoding when the
// map is empty (writeOptKeyExpires writes a single 0 flag byte, like
// writeOptVersion). Replay restores the absolute map VERBATIM (NOT recomputed) so
// pending key deadlines survive a crash time-stable.
//
// WRITE phase only (see appendFramedStaged): it returns the assigned commit
// sequence instead of blocking on the fsync, so the caller can release opMu before
// waiting and group-commit with concurrent writers. The caller MUST
// commitWaitStaged(seq). There is deliberately no blocking wrapper — every insert
// record goes through the staged pair (see appendSetPayloadStaged for the rule).
func (w *wal) appendInsertStaged(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(walInsert))
	_ = writeU64(&buf, id)
	_ = writeU64(&buf, uint64(ttl.Milliseconds())) //nolint:gosec
	_ = writeU32(&buf, uint32(len(vec)))           //nolint:gosec
	for _, f := range vec {
		_ = writeF32(&buf, f)
	}
	writeOptMeta(&buf, meta)
	if sparse == nil {
		buf.WriteByte(0)
	} else {
		buf.WriteByte(1)
		_ = writeU32(&buf, uint32(len(sparse.Indices))) //nolint:gosec
		for i, d := range sparse.Indices {
			_ = writeU32(&buf, d)
			_ = writeF32(&buf, sparse.Values[i])
		}
	}
	writeOptVersion(&buf, version)
	writeOptKeyExpires(&buf, keyExpires)
	return w.appendFramedStaged(buf.Bytes())
}

// writeOptVersion writes the optional per-point-version block: a 1-byte present
// flag, then (when present) a u64 version. A 0 version writes only the 0 flag
// (byte-identical to the pre-CAS format). readOptVersion is its inverse and
// tolerates EOF (old records have no block) by returning (0, true).
func writeOptVersion(buf *bytes.Buffer, version uint64) {
	if version == 0 {
		buf.WriteByte(0)
		return
	}
	buf.WriteByte(1)
	_ = writeU64(buf, version)
}

// readOptVersion is the inverse of writeOptVersion. It tolerates EOF (an OLD
// record predating the version block) by returning (0, true) — replay then
// defaults the version (a fresh insert → 1, a payload restore → leave as-is). A
// torn block (flag present but truncated body) returns (0, false) so replay stops.
func readOptVersion(r io.Reader) (uint64, bool) {
	var flag [1]byte
	if _, err := io.ReadFull(r, flag[:]); err != nil {
		return 0, true // EOF: old record without the trailing block
	}
	if flag[0] == 0 {
		return 0, true
	}
	v, err := readU64(r)
	if err != nil {
		return 0, false
	}
	return v, true
}

// writeOptMeta writes the optional-metadata encoding shared by appendInsertStaged
// and appendSetPayloadStaged: a 1-byte present flag, then (when present) a u32 key count
// followed by each [string key][Value]. readOptMeta is its exact inverse.
func writeOptMeta(buf *bytes.Buffer, meta Metadata) {
	if meta == nil {
		buf.WriteByte(0)
		return
	}
	buf.WriteByte(1)
	_ = writeU32(buf, uint32(len(meta))) //nolint:gosec
	for k, v := range meta {
		_ = writeString(buf, k)
		_ = writeValue(buf, v)
	}
}

// appendSetPayloadStaged logs an in-place payload mutation as the resulting full
// payload under one op-agnostic record (walSetPayload). The meta encoding is
// IDENTICAL to appendInsertStaged's (writeOptMeta), so replay reuses readOptMeta. A
// nil meta encodes a cleared payload.
//
// keyExpires carries the resulting per-key ABSOLUTE unix-millis deadlines (key
// -> deadline). It is appended as a TRAILING optional block AFTER the meta so
// the encoding is byte-identical to the pre-TTL format when the map is empty
// (writeOptKeyExpires writes a single 0 flag byte) and FORWARD/BACKWARD safe: a
// replayer that predates the block stops after the meta (trailing bytes unread,
// per-record framing isolates them); a NEW replayer reading an OLD record (no
// trailing block) treats the missing block as "no per-key TTL" (readOptKeyExpires
// returns nil,true on EOF).
//
// WRITE phase only (see appendFramedStaged): it returns the assigned commit
// sequence instead of blocking on the fsync, so payloadOpCAS can release opMu
// before waiting and group-commit with concurrent writers. The caller MUST
// commitWaitStaged(seq).
//
// THE RULE: a per-family blocking wrapper exists iff production calls it. No
// family has one any more — every record type (insert, delete, setPayload, named
// insert, MV add) goes through the staged pair, so NO production code path holds
// an ordering lock across an fsync. appendInsert/appendNamedInsert/appendMVAdd
// were deleted when the insert-family restore/reshard paths converted. Only the
// family-agnostic core appendFramed survives, and only as the synchronous
// composition the group-commit tests drive directly (see wal_group_commit_test.go);
// production reaches the log exclusively through appendFramedStaged.
func (w *wal) appendSetPayloadStaged(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(walSetPayload))
	_ = writeU64(&buf, id)
	writeOptMeta(&buf, meta)
	writeOptKeyExpires(&buf, keyExpires)
	writeOptVersion(&buf, version) // trailing version block (byte-identical when 0)
	return w.appendFramedStaged(buf.Bytes())
}

// writeOptKeyExpires writes the optional per-key-deadline block: a 1-byte present
// flag, then (when present) a u32 count followed by each [string key][u64
// absolute-deadline]. An empty map writes only the 0 flag (byte-identical to the
// pre-TTL format). readOptKeyExpires is its inverse and tolerates EOF (old
// records have no block) by returning nil.
func writeOptKeyExpires(buf *bytes.Buffer, ke map[string]uint64) {
	if len(ke) == 0 {
		buf.WriteByte(0)
		return
	}
	buf.WriteByte(1)
	_ = writeU32(buf, uint32(len(ke))) //nolint:gosec
	for k, dl := range ke {
		_ = writeString(buf, k)
		_ = writeU64(buf, dl)
	}
}

// appendDeleteStaged logs a successful Delete, WRITE phase only (see
// appendFramedStaged): it returns the assigned commit sequence and the caller
// waits on it outside opMu.
//
// Two callers, two ways of satisfying the staged-write contract. DeleteCAS /
// DeleteCASAt commitWaitStaged the returned seq directly. An upsert's
// replace-delete does NOT wait separately: its durability is subsumed by the
// commit-wait on the FOLLOWING insert's sequence (the delete's bytes are written
// first under the same opMu, so any Sync covering the insert's seq also covers
// this earlier seq).
func (w *wal) appendDeleteStaged(id uint64) (uint64, error) {
	var buf bytes.Buffer
	buf.WriteByte(byte(walDelete))
	_ = writeU64(&buf, id)
	return w.appendFramedStaged(buf.Bytes())
}

// appendFramed is the family-AGNOSTIC append core in its BLOCKING form: it
// serializes appends on w.mu and writes [payloadLen u32][crc32 u32][payload],
// fsyncing (unless noSync) before returning so the record is durable at the
// durability boundary. payload is an opaque, already-encoded record body.
//
// NO PRODUCTION CALLER: every family now goes through appendFramedStaged +
// commitWaitStaged (see appendSetPayloadStaged's rule note), so no ordering lock
// is ever held across an fsync. This blocking composition is retained solely as
// the group-commit tests' driver — they need a single call that appends AND waits
// in order to exercise leader-fsync batching (wal_group_commit_test.go). Do NOT
// call it from production code; use the staged pair.
func (w *wal) appendFramed(payload []byte) error {
	if w.poisoned.Load() {
		return ErrWALPoisoned // fast fail-closed: a prior fsync failure poisoned the log.
	}
	// WRITE phase: under mu, append the framed record and assign this writer's
	// commit sequence (the post-write writeSeq). mu is released BEFORE the fsync so
	// a concurrent appender can overlap this writer's commit-wait, and a leader's
	// f.Sync() never holds mu.
	w.mu.Lock()
	// AUTHORITATIVE fail-closed re-check under w.mu (see appendFramedStaged): a
	// leader may have poisoned the WAL after the early check above; never write a
	// new record into a poisoned log.
	if w.poisoned.Load() {
		w.mu.Unlock()
		return ErrWALPoisoned
	}
	var hdr [8]byte
	binary.BigEndian.PutUint32(hdr[0:4], uint32(len(payload))) //nolint:gosec
	binary.BigEndian.PutUint32(hdr[4:8], crc32.ChecksumIEEE(payload))
	if _, err := w.f.Write(hdr[:]); err != nil {
		w.mu.Unlock()
		return err
	}
	if _, err := w.f.Write(payload); err != nil {
		w.mu.Unlock()
		return err
	}
	// Assign the commit sequence under syncMu (ordered against syncedSeq updates)
	// while still holding mu (so writeSeq increments in the same order bytes were
	// appended). The bytes are now positioned in the file; this record is durable
	// once a Sync whose target was captured at-or-after this point completes.
	w.syncMu.Lock()
	w.writeSeq++
	mySeq := w.writeSeq
	w.syncMu.Unlock()
	w.mu.Unlock()

	if w.noSync {
		return nil // noSync path unchanged: write only, no fsync, no commit-wait.
	}
	return w.commitWait(mySeq)
}

// appendFramedStaged is appendFramed's WRITE phase only: identical bytes,
// identical ordering under w.mu, but it returns the assigned commit sequence
// instead of blocking on the durability wait. The caller MUST eventually call
// commitWaitStaged(seq) (directly or via a later staged write whose own wait
// covers this seq, e.g. an upsert's staged delete folded into its insert's
// wait) so every staged write is either waited on or provably subsumed by a
// later one on the SAME wal — the durability contract from appendFramed still
// applies, just decoupled from the caller's lock hold time. This is the
// STAGED half of the split-commit optimization described in the wal type doc:
// callers hold their ordering lock (opMu) across {apply + this WRITE phase}
// only, then release it and call commitWaitStaged outside the lock so
// concurrent writers' commit-waits overlap and the leader-fsync actually
// batches them.
func (w *wal) appendFramedStaged(payload []byte) (uint64, error) {
	if w.poisoned.Load() {
		return 0, ErrWALPoisoned // fast fail-closed: a prior fsync failure poisoned the log.
	}
	w.mu.Lock()
	// AUTHORITATIVE fail-closed check, under w.mu: a sync leader can poison the WAL
	// (in commitWait) between the early check above and here. Re-check before
	// writing ANY bytes so a poisoned WAL never appends a new, un-ackable record
	// (which would replay after reopen despite never being acked).
	if w.poisoned.Load() {
		w.mu.Unlock()
		return 0, ErrWALPoisoned
	}
	var hdr [8]byte
	binary.BigEndian.PutUint32(hdr[0:4], uint32(len(payload))) //nolint:gosec
	binary.BigEndian.PutUint32(hdr[4:8], crc32.ChecksumIEEE(payload))
	if _, err := w.f.Write(hdr[:]); err != nil {
		w.mu.Unlock()
		return 0, err
	}
	if _, err := w.f.Write(payload); err != nil {
		w.mu.Unlock()
		return 0, err
	}
	w.syncMu.Lock()
	w.writeSeq++
	mySeq := w.writeSeq
	w.syncMu.Unlock()
	w.mu.Unlock()
	return mySeq, nil
}

// commitWaitStaged is commitWait honoring the noSync bypass: in noSync mode
// appendFramedStaged never arranged a covering Sync (mirroring appendFramed's
// own noSync short-circuit), so waiting on syncedSeq would block forever.
// Staged callers use this instead of calling commitWait directly.
func (w *wal) commitWaitStaged(seq uint64) error {
	if seq == 0 {
		return nil // nothing was staged (e.g. a no-op delete) — a success, even poisoned.
	}
	if w.noSync {
		return nil
	}
	return w.commitWait(seq)
}

// commitWait blocks until a Sync (or a log-subsuming truncate) covering this
// writer's bytes (seq) has completed, then returns its result. The first arriver
// with no sync in flight becomes the SYNCER (leader-fsync); the rest fold into its
// flight (those whose bytes it covers) or wait for the next flight. f.Sync() runs
// holding NEITHER mu NOR syncMu so writes overlap the in-flight fsync.
func (w *wal) commitWait(seq uint64) error {
	if seq == 0 {
		// A no-op write (nothing staged) touches no durability, so it stays a
		// success even on a poisoned WAL — this fast path MUST precede the poison
		// gate so a no-op delete of an absent id is not turned into ErrWALPoisoned.
		return nil
	}
	w.syncMu.Lock()
	for {
		if w.poisoned.Load() {
			// Fail closed: a prior fsync failed and may have dropped its dirty
			// pages. Do NOT retry Sync() (a retry can falsely succeed) and do NOT
			// ack. A follower woken by the poisoning leader's broadcast lands here.
			w.syncMu.Unlock()
			return ErrWALPoisoned
		}
		if w.syncedSeq >= seq {
			w.syncMu.Unlock() // a completed Sync already covered our bytes.
			return nil
		}
		if w.syncing {
			// A flight is in progress. Either it will cover us (its target >= seq, set
			// when it started) or we wake to find syncedSeq still < seq and become the
			// next leader. Wait for the broadcast.
			w.cond.Wait()
			continue
		}
		// No flight running ⇒ we are the leader. Capture the target (all bytes written
		// so far) BEFORE releasing syncMu — batch-then-capture: every writer already
		// blocked here with seq <= target arrived before this capture, so our Sync
		// covers them. Writers arriving after the capture get a later seq > target and
		// fall through to the NEXT flight.
		w.syncing = true
		target := w.writeSeq
		w.syncMu.Unlock()

		if w.beforeSync != nil {
			w.beforeSync() // test seam: park followers behind this in-flight flight.
		}
		err := w.f.Sync() // covers every byte written before `target` was captured.

		w.syncMu.Lock()
		w.syncing = false
		if err != nil {
			// fsyncgate: poison BEFORE returning (and before the broadcast, so every
			// woken waiter observes the latch) — a failed fsync may have dropped its
			// dirty pages, and a follower retrying Sync() could then be acked on bytes
			// that are gone. Fail closed; only a reopen recovers.
			w.poisoned.Store(true)
		} else if target > w.syncedSeq {
			w.syncedSeq = target
		}
		if w.onSync != nil {
			w.onSync()
		}
		w.cond.Broadcast() // wake folded followers (seq <= target) + next-flight leader.
		if err != nil {
			w.syncMu.Unlock()
			return err // our own bytes are not durable; surface the fsync error.
		}
		// Loop: re-check syncedSeq >= seq (now true, since target >= seq).
	}
}

// truncate empties the log (called by Flush once the checkpoint subsumes it).
// Flush holds the caller's opMu across {snapshot + truncate}, so truncate only
// ever runs once every prior op is captured by the checkpoint. truncate's Sync
// therefore makes those ops durable-via-checkpoint; we advance syncedSeq to the
// current writeSeq so any append still in commit-wait (its bytes just truncated
// but its state in the checkpoint) is satisfied — no ack ever precedes a covering
// Sync, and a writer is never left waiting on bytes the rotation removed.
func (w *wal) truncate() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	// A failure at ANY step of a rotation (truncate/seek/sync) leaves the log's
	// durability state unknown, and — like commitWait — a later writer must never
	// be allowed to retry an fsync that could falsely succeed after the kernel
	// dropped a prior failure's dirty pages (fsyncgate). Fail closed on any error.
	if err := w.f.Truncate(0); err != nil {
		w.poisoned.Store(true)
		return err
	}
	if _, err := w.f.Seek(0, io.SeekStart); err != nil {
		w.poisoned.Store(true)
		return err
	}
	if err := w.f.Sync(); err != nil {
		w.poisoned.Store(true)
		return err
	}
	// NOTE: a SUCCESSFUL truncate+Sync does NOT clear a pre-existing poison. A
	// clean rotation only proves this one Sync returned success — on post-4.13
	// Linux that can happen after the kernel already discarded a prior failure's
	// dirty pages, so "truncate succeeded" is not evidence the device is healthy.
	// Poison is therefore cleared ONLY by a fresh reopen (openWAL), which
	// re-establishes a new file handle and a known durability path.
	w.syncMu.Lock()
	if w.writeSeq > w.syncedSeq {
		w.syncedSeq = w.writeSeq
	}
	if w.onSync != nil {
		w.onSync()
	}
	w.cond.Broadcast()
	w.syncMu.Unlock()
	return nil
}

// close closes the underlying file. Idempotent.
func (w *wal) close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := w.f.Close()
	w.f = nil
	return err
}

// errUnknownWALRecord is returned by replayFramed when an INTACT (CRC-valid)
// record carries a record tag this binary does not recognize. Unlike a torn tail
// (which is the legitimate durability boundary and stops replay silently), a
// CRC-valid frame is a committed record: an unknown tag means the WAL was written
// by a NEWER binary (a downgrade) or is corrupt. Failing loud here prevents
// silently truncating replay and discarding the committed records that follow.
var errUnknownWALRecord = errors.New("vector: WAL contains an unrecognized record tag (newer binary or corruption); refusing to truncate replay")

// replayFramed is the family-AGNOSTIC replay core: it reads path and invokes
// apply(recordPayload) for each intact framed record, in order. It stops
// (without error) at EOF, a torn header, a garbage/torn length, or a CRC mismatch
// (torn/corrupt tail) — the legitimate durability boundary. apply receives the
// opaque record-payload bytes; a per-family decoder (dense's replayRecord, or
// named/MV elsewhere) interprets them and returns an error to halt replay:
//   - errStopReplay (or any apply-returned error wrapping it): a malformed body of
//     a KNOWN tag, i.e. a torn record — replay stops WITHOUT surfacing an error
//     (the historical quiet-stop boundary).
//   - errUnknownWALRecord: a CRC-valid frame with an UNRECOGNIZED tag — replay
//     fails loud (the error propagates) so a downgraded/corrupt WAL is not
//     silently truncated, dropping committed records.
//
// A missing file replays nothing.
func replayFramed(path string, apply func(rec []byte) error) error {
	f, err := os.Open(path) //nolint:gosec // caller-supplied path
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer func() { _ = f.Close() }()
	br := bufio.NewReaderSize(f, 1<<16)
	for {
		var hdr [8]byte
		if _, err := io.ReadFull(br, hdr[:]); err != nil {
			return nil // EOF or torn header
		}
		plen := binary.BigEndian.Uint32(hdr[0:4])
		crc := binary.BigEndian.Uint32(hdr[4:8])
		if plen == 0 || plen > maxWALRecord {
			return nil // garbage length -> stop
		}
		payload := make([]byte, plen)
		if _, err := io.ReadFull(br, payload); err != nil {
			return nil // torn tail
		}
		if crc32.ChecksumIEEE(payload) != crc {
			return nil // corrupt/torn -> stop at the durability boundary
		}
		if aerr := apply(payload); aerr != nil {
			if errors.Is(aerr, errStopReplay) {
				return nil // malformed body of a known tag (torn record) -> quiet stop
			}
			return aerr // unknown tag / unrecoverable -> fail loud
		}
	}
}

// errStopReplay signals a quiet replay stop at the durability boundary (a
// malformed/truncated body of a KNOWN record tag, i.e. a torn final record).
// replayFramed swallows it and stops without surfacing an error, matching the
// historical torn-tail behavior. An apply func returns errUnknownWALRecord
// instead when the frame is intact but the tag is unrecognized (fail loud).
var errStopReplay = errors.New("vector: WAL replay stop (torn record)")

// replayWAL reads path and invokes onInsert/onDelete/onSetPayload for each intact
// DENSE record, in order. It is a thin wrapper over the shared replayFramed core:
// the per-record apply closure runs the dense replayRecord switch over the 3
// dense record types, returning false on an unknown tag or malformed body so
// replay stops at the durability boundary exactly as before. A missing file
// replays nothing.
func replayWAL(
	path string,
	onInsert func(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64),
	onDelete func(id uint64),
	onSetPayload func(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64),
) error {
	return replayFramed(path, func(rec []byte) error {
		return replayRecord(rec, onInsert, onDelete, onSetPayload)
	})
}

// replayRecord decodes one record payload and dispatches it. Returns nil on a
// successfully applied record, errStopReplay on a malformed body of a KNOWN tag
// (torn record → quiet stop), or errUnknownWALRecord on an unrecognized tag (an
// intact frame from a newer/downgraded binary → fail loud, not silent truncation).
func replayRecord(
	payload []byte,
	onInsert func(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64),
	onDelete func(id uint64),
	onSetPayload func(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64),
) error {
	r := bytes.NewReader(payload)
	t, err := r.ReadByte()
	if err != nil {
		return errStopReplay
	}
	switch walRecType(t) {
	case walDelete:
		id, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		onDelete(id)
		return nil
	case walSetPayload:
		id, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		meta, ok := readOptMeta(r)
		if !ok {
			return errStopReplay
		}
		ke, ok := readOptKeyExpires(r)
		if !ok {
			return errStopReplay
		}
		version, ok := readOptVersion(r)
		if !ok {
			return errStopReplay
		}
		onSetPayload(id, meta, ke, version)
		return nil
	case walInsert:
		id, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		ttlMs, err := readU64(r)
		if err != nil {
			return errStopReplay
		}
		dim, err := readU32(r)
		if err != nil || dim > uint32(maxWALRecord/4) {
			return errStopReplay
		}
		vec := make([]float32, dim)
		for i := range vec {
			if vec[i], err = readF32(r); err != nil {
				return errStopReplay
			}
		}
		meta, ok := readOptMeta(r)
		if !ok {
			return errStopReplay
		}
		sparse, ok := readOptSparse(r)
		if !ok {
			return errStopReplay
		}
		version, ok := readOptVersion(r)
		if !ok {
			return errStopReplay
		}
		// Trailing optional per-key-deadline block (absolute unix-millis). An OLD
		// record (no block) reads (nil, true) at EOF — no per-key TTL.
		keyExpires, ok := readOptKeyExpires(r)
		if !ok {
			return errStopReplay
		}
		onInsert(id, vec, time.Duration(ttlMs)*time.Millisecond, meta, sparse, keyExpires, version) //nolint:gosec
		return nil
	default:
		// CRC-valid frame with an unrecognized tag: written by a NEWER binary (a
		// downgrade) or corrupt. Fail loud rather than silently truncating replay
		// and discarding the committed records that follow this one.
		return errUnknownWALRecord
	}
}

func readOptMeta(r io.Reader) (Metadata, bool) {
	var flag [1]byte
	if _, err := io.ReadFull(r, flag[:]); err != nil {
		return nil, false
	}
	if flag[0] == 0 {
		return nil, true
	}
	n, err := readU32(r)
	if err != nil {
		return nil, false
	}
	m := make(Metadata, n)
	for i := uint32(0); i < n; i++ {
		key, err := readString(r)
		if err != nil {
			return nil, false
		}
		val, err := readValue(r)
		if err != nil {
			return nil, false
		}
		m[key] = val
	}
	return m, true
}

// readOptKeyExpires is the inverse of writeOptKeyExpires. It tolerates EOF (an
// OLD walSetPayload record predating the per-key-TTL block) by returning
// (nil, true) — no per-key TTL, replay continues. A torn block (flag present but
// truncated body) returns (nil, false) so replay stops at the durability boundary.
func readOptKeyExpires(r io.Reader) (map[string]uint64, bool) {
	var flag [1]byte
	if _, err := io.ReadFull(r, flag[:]); err != nil {
		return nil, true // EOF: old record without the trailing block — treat as none
	}
	if flag[0] == 0 {
		return nil, true
	}
	n, err := readU32(r)
	if err != nil {
		return nil, false
	}
	ke := make(map[string]uint64, n)
	for i := uint32(0); i < n; i++ {
		key, err := readString(r)
		if err != nil {
			return nil, false
		}
		dl, err := readU64(r)
		if err != nil {
			return nil, false
		}
		ke[key] = dl
	}
	return ke, true
}

func readOptSparse(r io.Reader) (*SparseVector, bool) {
	var flag [1]byte
	if _, err := io.ReadFull(r, flag[:]); err != nil {
		return nil, false
	}
	if flag[0] == 0 {
		return nil, true
	}
	nnz, err := readU32(r)
	if err != nil {
		return nil, false
	}
	sv := &SparseVector{Indices: make([]uint32, nnz), Values: make([]float32, nnz)}
	for i := uint32(0); i < nnz; i++ {
		if sv.Indices[i], err = readU32(r); err != nil {
			return nil, false
		}
		if sv.Values[i], err = readF32(r); err != nil {
			return nil, false
		}
	}
	return sv, true
}
