// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
)

// Flush wipes the ENTIRE KV keyspace of this cache in O(shards) — not O(keys) —
// by swapping every shard's index table for a fresh empty one and recording a
// per-shard durability watermark. After it returns, every Get/GetInto/Iterate on
// a pre-flush key misses; post-flush writes proceed normally. Returns the first
// shard error, if any.
//
// DURABILITY ORDERING CONTRACT (the reason the sidecar, not the index swap, is the
// durable proof). The index swap is in-memory only; the crash-safe record of the
// flush is each mmap shard's sidecar (dataDir/flushed.seq). shard.flush writes and
// fsyncs that sidecar BEFORE it performs the swap (and thus before this method
// returns), so a sidecar-write FAILURE aborts the flush with the keyspace intact —
// never an ACKed-but-non-durable wipe (see shard.flush for why an errored handler
// still advances the applied index, which makes that ordering load-bearing). On the
// success path a replicated Flush OP applies through shard/fsm.go's Apply: the
// handler runs to completion (syncing every sidecar) and only THEN does the deferred
// f.cache.SetAppliedIndex(l.Index, …) persist the applied index (shard/fsm.go:287),
// so the sidecar is always durable before the applied index can advance past the
// flush op. That ordering is what prevents a crash from resurrecting flushed keys:
// were the applied index persisted first, a crash could leave the log unable to
// replay the flush (index already past it) while the pages still framed the wiped
// entries — and the next rebuild would re-index them. Because the sidecar lands
// first, the rebuild always sees the watermark and skips them.
func (c *Cache) Flush() error {
	var firstErr error
	for _, s := range c.shards {
		if err := s.flush(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// flush wipes one shard's keyspace under the full write lock (like a resize).
//
// The mechanism is a single atomic index swap — the same O(1) idiom resize
// (shard.go) and cold compaction (compact.go) use — so lock-free readers pick up
// the empty table on their next tab.Load(); a reader still holding a pre-swap table
// snapshot may resolve a pre-flush key until it reloads, which is the ordinary
// lock-free read contract, not a flush-specific hazard.
//
// FIELDS RESET, AND WHY ONLY THESE. The only in-memory "live" accounting a shard
// keeps is the indexTable's live/tomb slot counts, and the swap to newIndexTable(0)
// replaces the whole table — so those reset for free with the swap and there is
// nothing else to zero. Deliberately NOT reset:
//   - writeSeq — a monotonic high-water; resetting it would hand out sequences the
//     rebuild's max-seq contest could collide with. We CAPTURE it as the floor and
//     leave it advancing.
//   - the cumulative stat counters (gets/misses/puts/dels/expirations/evictions/
//     rejects/pagesAlloc/compactions/…) — Stats reports them as lifetime totals, not
//     live gauges; zeroing them would corrupt those semantics.
//   - the page bytes and their persisted head/tail offsets — bytesUsed derives the
//     live-byte figure from (tail-head), but the pages OWN that allocation and a
//     lock-free reject-writes reader may still alias those bytes, so we must not
//     overwrite or reuse them online. The physical bytes are reclaimed lazily by
//     cold compaction at the next open (which drops every now-unindexed entry), the
//     same recovery path an mmap shard already relies on. So a running shard's
//     BytesUsed stays non-zero after flush; its LIVE keyset is nonetheless empty
//     (every Get misses, Iterate yields nothing).
//
// KNOWN LIMITATION — flush frees the KEYSPACE but not necessarily WRITE CAPACITY.
// Because the page head/tail offsets above are deliberately left intact (the lock-
// free reject-writes reader contract), a shard that was FULL stays full after flush:
//   - Under PolicyRingbufEvict (the DEFAULT) this is invisible — the ring evicts to
//     make room, so the next write reclaims the flushed space and succeeds.
//   - Under PolicyRejectWrites it is visible: the next write can still return ErrFull
//     even though the keyspace is now logically EMPTY. The space is reclaimed only by
//     the next COLD COMPACTION — which runs for an mmap shard at open (a restart), and
//     NEVER online for a heap shard (heap has no cold-compaction path), so a full heap
//     reject-writes shard does not regain capacity until the process restarts.
//
// Online reclaim under reject-writes is a separate, deeper change (it would have to
// rendezvous with any lock-free reader still aliasing the pages) and is intentionally
// NOT done here.
func (s *shard) flush() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Capture the floor: the current per-shard high-water sequence. Every entry now
	// on the pages carries seq <= floor, so recording floor is exactly "wipe
	// everything written so far". writeSeq itself is left untouched (monotonic).
	floor := s.writeSeq

	// Make the wipe DURABLE BEFORE it takes effect. An mmap shard rebuilds its index
	// from page bytes at open, so the durable proof of a flush is the sidecar, not
	// the in-memory swap. Writing (and fsyncing) the sidecar FIRST means a
	// sidecar-write error returns here with the keyspace still intact and no swap
	// performed — the flush op then fails and the client retries the idempotent
	// flush — rather than ACKing an empty in-memory keyspace whose durable watermark
	// never landed. That distinction matters because fsm.Apply advances the applied
	// index even when the handler returns a (non-fatal) error, so a swap that
	// preceded a failed sidecar write would leave the log unable to replay the flush
	// on restart while the pages still framed the "wiped" entries: a resurrection on
	// a single node, and a silent divergence from replicas whose sidecar write
	// succeeded. Ordering the durable step first loses nothing on the success path —
	// the swap below is infallible. And a crash BETWEEN this sidecar write and the
	// swap is safe: the swap is RAM-only (lost on crash regardless) and, because the
	// handler never returned, the applied index did not advance, so the log replays
	// the flush — which the now-durable sidecar already makes the rebuild honor.
	//
	// Heap shards (!isMmap) have no rebuild path (their index is authoritative and
	// never reconstructed from pages), so nothing a restart could resurrect and
	// nothing to persist — skip the sidecar entirely.
	if s.isMmap {
		if err := s.writeFlushSidecar(floor); err != nil {
			// Wrap so the REPLICATED apply path can errors.Is this as ErrFlushNotDurable
			// and classify it classFatal (shard/apply_class.go): a follower that could
			// not durably record the watermark must HALT rather than advance its applied
			// index past a flush it did not durably apply — advancing would resurrect
			// pre-flush keys on a later failover and diverge it from peers whose sidecar
			// landed. The multi-%w keeps both the sentinel and the underlying I/O cause
			// visible through errors.Is. On the leader this still returns BEFORE the swap
			// below, so the keyspace stays intact either way.
			return fmt.Errorf("%w: %w", ErrFlushNotDurable, err)
		}
	}

	// NO onRemove NOTIFICATIONS HERE, deliberately. The wipe is an O(1) table swap
	// with no per-entry walk, so there are no keys to report even in principle —
	// and a derived secondary index does not need them: the flush handler resets
	// the whole index instead, because an empty cache IS an exact empty index.
	// See cache/onremove.go.
	//
	// Durable now (or heap): perform the O(1) logical wipe. Swap in a fresh empty
	// table — lock-free readers see it on their next load, and the old table (with
	// its live/tomb counts) is dropped whole. The in-memory floor mirrors the
	// sidecar for any same-process rebuild (none at runtime today; free and
	// defensive).
	s.flushedThroughSeq = floor
	s.tab.Store(newIndexTable(0))
	return nil
}

// Flush sidecar (dataDir/flushed.seq) — the per-shard durable watermark recording
// the write-sequence floor a Cache.Flush() wiped through. Fixed 17-byte record,
// CRC'd field-for-field the way cache/file.go guards its header fields.
//
//	0..3   magic uint32 (little-endian)
//	4      version byte
//	5..12  flushedThroughSeq uint64 (little-endian)
//	13..16 CRC32-IEEE of bytes 0..12
const (
	flushSidecarName      = "flushed.seq"
	flushSidecarTmpSuffix = ".tmp"
	// flushSidecarMagic is "RSFL" in little-endian bytes — distinct from the pages
	// file magic so a truncated/misnamed pages header can never be mistaken for a
	// sidecar and vice versa.
	flushSidecarMagic   uint32 = 0x4C465352
	flushSidecarVersion byte   = 1
	flushSidecarSize           = 17
	flushSidecarCRCOff         = 13
)

// encodeFlushSidecar writes the fixed record into dst[:flushSidecarSize].
func encodeFlushSidecar(dst []byte, floor uint64) {
	binary.LittleEndian.PutUint32(dst[0:4], flushSidecarMagic)
	dst[4] = flushSidecarVersion
	binary.LittleEndian.PutUint64(dst[5:13], floor)
	crc := crc32.ChecksumIEEE(dst[0:flushSidecarCRCOff])
	binary.LittleEndian.PutUint32(dst[flushSidecarCRCOff:flushSidecarSize], crc)
}

// decodeFlushSidecar validates a sidecar record and returns its floor. ok=false on
// any malformation — wrong length, bad magic/version, or CRC mismatch — so a torn
// or foreign file collapses to "no watermark" (floor 0) rather than an over-report.
func decodeFlushSidecar(b []byte) (floor uint64, ok bool) {
	if len(b) != flushSidecarSize {
		return 0, false
	}
	if binary.LittleEndian.Uint32(b[0:4]) != flushSidecarMagic {
		return 0, false
	}
	if b[4] != flushSidecarVersion {
		return 0, false
	}
	stored := binary.LittleEndian.Uint32(b[flushSidecarCRCOff:flushSidecarSize])
	if stored != crc32.ChecksumIEEE(b[0:flushSidecarCRCOff]) {
		return 0, false
	}
	return binary.LittleEndian.Uint64(b[5:13]), true
}

// writeFlushSidecar durably records `floor` into dataDir/flushed.seq using the same
// atomic temp+fsync+rename+syncDir sequence cold compaction uses (compact.go): stage
// into flushed.seq.tmp, fsync the temp so its bytes are on disk, rename over the real
// path (atomic on POSIX), then fsync the directory so the rename survives a crash. A
// crash-orphaned temp is harmless — it is never read and the next flush O_TRUNCs it.
// Called with s.mu held.
func (s *shard) writeFlushSidecar(floor uint64) error {
	var rec [flushSidecarSize]byte
	encodeFlushSidecar(rec[:], floor)

	path := filepath.Join(s.dataDir, flushSidecarName)
	tmp := path + flushSidecarTmpSuffix

	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640) //nolint:gosec // dataDir is caller-configured
	if err != nil {
		return fmt.Errorf("cache: create flush sidecar %s: %w", tmp, err)
	}
	if _, werr := f.Write(rec[:]); werr != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("cache: write flush sidecar %s: %w", tmp, werr)
	}
	// fsync the bytes before anything points at them — this is the durability the
	// flush's crash-safety rests on.
	if serr := f.Sync(); serr != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("cache: fsync flush sidecar %s: %w", tmp, serr)
	}
	if cerr := f.Close(); cerr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("cache: close flush sidecar %s: %w", tmp, cerr)
	}
	if rerr := os.Rename(tmp, path); rerr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("cache: publish flush sidecar %s: %w", path, rerr)
	}
	// Make the rename itself durable. The bytes and the rename's TARGET inode are
	// already fsynced; syncDir makes the rename's DIRECTORY ENTRY durable. If it
	// fails we cannot know whether the rename survives a crash, so we must NOT
	// acknowledge the flush as durable — return the error rather than warn-and-
	// succeed. Because shard.flush writes this sidecar BEFORE it swaps the index, a
	// failure here returns with the keyspace still live and the flush op fails, so
	// the client retries. The residual is one-directional and NEVER resurrection: if
	// the rename happened to survive, a crash before a successful retry applies the
	// (already-durable) flush on restart — the "more flushed" direction — while the
	// running node still held the keys and the errored flush told the caller the
	// outcome was uncertain; a lost rename simply leaves nothing flushed, which the
	// retry redoes.
	if derr := syncDir(s.dataDir); derr != nil {
		return fmt.Errorf("cache: fsync flush sidecar dir %s: %w", s.dataDir, derr)
	}
	return nil
}

// readFlushSidecar restores the flush watermark for a shard opening from dataDir.
// A genuinely ABSENT sidecar returns (0, nil) — today's behaviour exactly, "nothing
// was ever flushed", and backward-compatible with a shard that predates the feature.
//
// A PRESENT-but-invalid sidecar (a read error other than not-exist, a torn write, a
// CRC mismatch, or a foreign file) returns (0, error) and FAILS the shard open. This
// deliberately does NOT mirror readAppliedStamp/readPBFrontier's 0-on-corruption:
// their 0 triggers a REBUILD/recovery that reconverges, whereas a flush watermark of
// 0 re-indexes every pre-flush entry PERMANENTLY. The applied index is already past
// the flush op, so a warm restart never replays it (shard/fsm.go skips already-
// applied indices), and this replica would silently diverge from peers whose sidecar
// is intact. A corrupt watermark is a durability fault the operator must see, not a
// silent resurrection — so we refuse to open rather than fabricate floor 0.
func readFlushSidecar(dataDir string) (uint64, error) {
	path := filepath.Join(dataDir, flushSidecarName)
	b, err := os.ReadFile(path) //nolint:gosec // dataDir is caller-configured
	if err != nil {
		if os.IsNotExist(err) {
			// Never flushed (or a pre-feature shard) — the sole backward-compatible
			// "open normally at floor 0" path.
			return 0, nil
		}
		// A present sidecar we cannot READ (permissions, I/O error) is as ambiguous as
		// a corrupt one: we cannot prove nothing was flushed, so fail rather than
		// under-report the watermark.
		return 0, fmt.Errorf("cache: read flush sidecar %s: %w", path, err)
	}
	floor, ok := decodeFlushSidecar(b)
	if !ok {
		return 0, fmt.Errorf("cache: flush sidecar %s failed validation (corrupt watermark; refusing to open lest pre-flush data resurrect and diverge this replica)", path)
	}
	return floor, nil
}
