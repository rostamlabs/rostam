// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
)

// errFutureVersion is returned by validateHeader for a pages file whose on-disk
// version is NEWER than this build writes (version > cacheVersion). It is a
// distinguishable sentinel because newShard must handle it differently from every
// other validation failure: a newer-format file is refused (the open fails) rather
// than rotated aside, since rotating a file a future build wrote would be silent
// data loss. Every other failure (bad magic, too-old version, CRC mismatch) stays
// an ordinary rotatable error.
var errFutureVersion = errors.New("unsupported future cache version")

// Header layout (64 bytes total):
//
//	0..7    magic uint64 (little-endian)
//	8..11   version uint32
//	12..15  pageSize uint32
//	16..19  numPages uint32
//	20..27  appliedIndex uint64
//	28..31  headerCRC uint32 (CRC32-IEEE of bytes 0..27)
//	32..39  lastAppliedStampMs uint64      (version 3+)
//	40..43  stampCRC uint32 (CRC32-IEEE of bytes 32..39)  (version 3+)
//	44..51  pbFrontierSeq uint64
//	52..59  pbFrontierEpoch uint64
//	60..63  pbFrontierCRC uint32 (CRC32-IEEE of bytes 44..59)
//
// The stamp fields carry their OWN CRC rather than extending headerCRC's range,
// so the bytes covered by headerCRC (0..27) are byte-identical between version 2
// and version 3. That keeps the two formats mutually intelligible for the fields
// they share: a v2 file opened by this build validates its core header normally
// and simply restores stamp=0 (the stamp CRC over eight zero bytes does not
// match), and a v3 file's applied index is readable by any build that ignores
// bytes 32..63.
//
// The PB FRONTIER (44..63) follows that precedent EXACTLY and therefore carries
// NO version bump: it consumes previously reserved (zero-filled) bytes, it is
// guarded by its own CRC rather than by headerCRC, and nothing in readHeader /
// validateHeader looks at it. A file written by a build that predates the field
// has zeros there, whose CRC over sixteen zero bytes does not match zero, so it
// restores as (0, 0) — the SAFE under-reporting answer (see readPBFrontier). A
// file written WITH the field is byte-identical to one without it for every byte
// any older build reads. Bumping cacheVersion would have been strictly worse:
// minReadableCacheVersion == cacheVersion today, so a bump would rotate every
// existing pages file aside and throw away the very data whose frontier this
// field exists to describe.
const (
	cacheMagic uint64 = 0x4843414D54534552 // "RSTMCACH" little-endian
	// cacheVersion is the format this build WRITES. History:
	//
	//	v1 → v2: the on-disk ring-buffer entry codec (keyLen / valLen / expiry /
	//	  CRC) flipped from big-endian to little-endian. The 64-byte header and
	//	  per-page head/tail offsets were always little-endian, so a version-1 file
	//	  would pass every other header check; only this version gate distinguishes
	//	  the two entry codecs. Version-1 files are rejected loudly (rotated aside)
	//	  instead of being decoded with the little-endian reader, which would
	//	  silently drop every persisted key. A DataDir written by a version-1 build
	//	  must be reformatted (or migrated) before use.
	//
	//	v2 → v3: the header now persists the shard's LOGICAL clock
	//	  (lastAppliedStampMs) alongside the applied index, so cold compaction at
	//	  open can judge TTL expiry against the deterministic replicated clock
	//	  instead of the wall clock (see cache/compact.go).
	//
	//	v3 → v4: the per-entry codec grew an 8-byte META word (write sequence +
	//	  tombstone flag) between the expiry and the CRC — see cache/ringbuf.go. It
	//	  is what makes a WARM RESTART correct: the rebuild resolves each key to the
	//	  copy with the highest sequence instead of the last one the page walk
	//	  reaches (#12A), and a persisted tombstone keeps a deleted key deleted
	//	  (#12B). A v3 entry decoded with the v4 reader would frame garbage, so the
	//	  two codecs are mutually unintelligible and the version gate below is the
	//	  only thing separating them.
	cacheVersion uint32 = 4
	// minReadableCacheVersion is the oldest on-disk version this build opens in
	// place instead of rotating aside.
	//
	// It equals cacheVersion: v4 changed the ENTRY codec, and there is no
	// deployed persistent state to migrate, so a pre-v4 pages file is rotated
	// aside (renamed .bad-<timestamp>) by the same tested path that handles a bad
	// magic or a CRC failure, and the shard starts empty. On a replicated node the
	// committed state is then rebuilt from the log — replay, or an InstallSnapshot
	// from a peer. Nothing is lost that the cluster does not still hold, and the
	// runtime never has to carry a second entry codec or a version branch on any
	// path, hot or cold.
	//
	// The reverse direction has never been supported either: a v4 file opened by an
	// older build trips that build's version gate and is rotated aside in exactly
	// the same way. Downgrades reformat the DataDir.
	minReadableCacheVersion uint32 = 4
	headerSize              int    = 64
	pageHdrSize             int    = 8 // head u32 + tail u32 stored at start of each mmap-backed page

	// hdrStampOff / hdrStampCRCOff locate the version-3 persisted logical clock.
	hdrStampOff    = 32
	hdrStampCRCOff = 40

	// hdrPBSeqOff / hdrPBEpochOff / hdrPBFrontierCRCOff locate the persisted
	// primary-backup applied frontier — the (seq, epoch) IDENTITY of the newest
	// PB write materialized into these pages. Unlike appliedIndex (a Raft log
	// index) a PB position is meaningless without its epoch, because Promote
	// continues seq assignment from the promoted node's high-water and so REUSES
	// seqs across epochs; the pair is stored and CRC'd as one unit for that
	// reason. Occupies the last 16+4 previously-reserved header bytes.
	hdrPBSeqOff         = 44
	hdrPBEpochOff       = 52
	hdrPBFrontierCRCOff = 60
)

// readHeader parses the 64-byte header at the start of region. Returns
// the decoded fields and an error if the CRC is invalid or region is
// too small.
func readHeader(region []byte) (magic uint64, version, pageSize, numPages uint32, appliedIdx uint64, err error) {
	if len(region) < headerSize {
		return 0, 0, 0, 0, 0, errors.New("cache: region too small for header")
	}
	magic = binary.LittleEndian.Uint64(region[0:8])
	version = binary.LittleEndian.Uint32(region[8:12])
	pageSize = binary.LittleEndian.Uint32(region[12:16])
	numPages = binary.LittleEndian.Uint32(region[16:20])
	appliedIdx = binary.LittleEndian.Uint64(region[20:28])
	storedCRC := binary.LittleEndian.Uint32(region[28:32])
	wantCRC := crc32.ChecksumIEEE(region[0:28])
	if storedCRC != wantCRC {
		return 0, 0, 0, 0, 0, fmt.Errorf("cache: header CRC mismatch (got %x, want %x)", storedCRC, wantCRC)
	}
	return magic, version, pageSize, numPages, appliedIdx, nil
}

// writeHeader writes a fresh header to region. Caller must ensure
// region is at least headerSize bytes. Zero-fills bytes 32..63.
func writeHeader(region []byte, pageSize, numPages uint32, appliedIdx uint64) {
	binary.LittleEndian.PutUint64(region[0:8], cacheMagic)
	binary.LittleEndian.PutUint32(region[8:12], cacheVersion)
	binary.LittleEndian.PutUint32(region[12:16], pageSize)
	binary.LittleEndian.PutUint32(region[16:20], numPages)
	binary.LittleEndian.PutUint64(region[20:28], appliedIdx)
	// Reserved bytes 32..63 stay zero.
	for i := 32; i < headerSize; i++ {
		region[i] = 0
	}
	crc := crc32.ChecksumIEEE(region[0:28])
	binary.LittleEndian.PutUint32(region[28:32], crc)
	// Stamp the v3 logical-clock slot explicitly (with its own CRC) so a freshly
	// written header always carries a VALID stamp field rather than an
	// indistinguishable-from-v2 zero blob.
	setAppliedStamp(region, 0)
	// Same for the PB frontier: a fresh header states "genesis, (0,0)" with a
	// valid checksum rather than an unreadable zero blob that merely DECODES to
	// the same answer.
	setPBFrontier(region, 0, 0)
}

// setPBFrontier writes the persisted PB applied frontier (seq, epoch) and its
// CRC. Caller is responsible for msync ORDERING if durability is required — see
// Cache.SetPBFrontier, which is the only production caller and which flushes the
// page data before stamping, precisely so the watermark can never name a write
// whose data is not yet on disk.
func setPBFrontier(region []byte, seq, epoch uint64) {
	binary.LittleEndian.PutUint64(region[hdrPBSeqOff:hdrPBSeqOff+8], seq)
	binary.LittleEndian.PutUint64(region[hdrPBEpochOff:hdrPBEpochOff+8], epoch)
	crc := crc32.ChecksumIEEE(region[hdrPBSeqOff : hdrPBSeqOff+16])
	binary.LittleEndian.PutUint32(region[hdrPBFrontierCRCOff:hdrPBFrontierCRCOff+4], crc)
}

// readPBFrontier returns the persisted PB applied frontier, or (0, 0) if the
// field is absent or unreadable. Like readAppliedStamp it is guarded by its OWN
// CRC, so three cases collapse to the same answer:
//
//   - a pages file written before this field existed (reserved zero bytes);
//   - a torn frontier write (crash between the 16-byte store and its CRC);
//   - any future format that repurposes the slot without the checksum.
//
// (0, 0) is the SAFE answer in all three, and safe in exactly one direction: it
// is the genesis frontier, i.e. the maximal UNDER-report. A PB engine restored
// to it claims to hold nothing, so a primary re-ships from the start of its
// retained ring and the log-matching check (pbisr receiveLocked) either accepts a
// true prefix or rejects cleanly. The catastrophic direction is the other one —
// a watermark that OVER-reports names a prefix the node does not hold, and log
// matching, which compares an incoming frame against this very number, would then
// certify a divergent append. Every design decision around this field exists to
// make over-reporting unreachable.
func readPBFrontier(region []byte) (seq, epoch uint64) {
	if len(region) < headerSize {
		return 0, 0
	}
	stored := binary.LittleEndian.Uint32(region[hdrPBFrontierCRCOff : hdrPBFrontierCRCOff+4])
	if stored != crc32.ChecksumIEEE(region[hdrPBSeqOff:hdrPBSeqOff+16]) {
		return 0, 0
	}
	return binary.LittleEndian.Uint64(region[hdrPBSeqOff : hdrPBSeqOff+8]),
		binary.LittleEndian.Uint64(region[hdrPBEpochOff : hdrPBEpochOff+8])
}

// setAppliedStamp writes the persisted logical clock (lastAppliedStampMs) and
// its CRC. Caller is responsible for msync if durability is required. See
// readAppliedStamp for why the field carries its own checksum.
func setAppliedStamp(region []byte, stampMs uint64) {
	binary.LittleEndian.PutUint64(region[hdrStampOff:hdrStampOff+8], stampMs)
	crc := crc32.ChecksumIEEE(region[hdrStampOff : hdrStampOff+8])
	binary.LittleEndian.PutUint32(region[hdrStampCRCOff:hdrStampCRCOff+4], crc)
}

// readAppliedStamp returns the persisted logical clock, or 0 if the field is
// absent or unreadable. Its own CRC (not the core header CRC) guards it, so
// three cases collapse to the same SAFE answer of 0:
//
//   - a pre-v3 file, whose bytes 32..43 are zero-filled reserved space (no longer
//     reachable now that minReadableCacheVersion is 4, but the guard is free);
//   - a torn stamp write (crash between the 8-byte store and its CRC);
//   - any future format that repurposes the slot without the checksum.
//
// Zero is safe in every one of them: the logical clock only ever gates how much
// a compaction/sweep may reclaim, and 0 reclaims nothing by expiry (isExpired(e,
// 0) is false for every e). Under-restoring the clock costs efficiency; it can
// never drop an entry that should have lived.
func readAppliedStamp(region []byte) uint64 {
	if len(region) < headerSize {
		return 0
	}
	stamp := binary.LittleEndian.Uint64(region[hdrStampOff : hdrStampOff+8])
	stored := binary.LittleEndian.Uint32(region[hdrStampCRCOff : hdrStampCRCOff+4])
	if stored != crc32.ChecksumIEEE(region[hdrStampOff:hdrStampOff+8]) {
		return 0
	}
	return stamp
}

// setAppliedIndex updates only the appliedIndex field + CRC. Caller is
// responsible for msync if durability is required.
func setAppliedIndex(region []byte, appliedIdx uint64) {
	binary.LittleEndian.PutUint64(region[20:28], appliedIdx)
	crc := crc32.ChecksumIEEE(region[0:28])
	binary.LittleEndian.PutUint32(region[28:32], crc)
}

// validateHeader checks the header against caller-expected pageSize and
// numPages. Returns (appliedIdx, fresh=true if region is all-zero, err).
// An error means the file is unusable (bad magic, version, size, or CRC).
func validateHeader(region []byte, expectedPageSize, expectedNumPages uint32) (appliedIdx uint64, fresh bool, err error) {
	if len(region) < headerSize {
		return 0, false, errors.New("cache: region too small")
	}
	if isZeroPrefix(region[:headerSize]) {
		return 0, true, nil
	}
	magic, version, pageSize, numPages, idx, herr := readHeader(region)
	if herr != nil {
		return 0, false, herr
	}
	if magic != cacheMagic {
		return 0, false, fmt.Errorf("cache: bad magic %x (want %x)", magic, cacheMagic)
	}
	// A NEWER-than-known format is reported with a DISTINGUISHABLE sentinel. It is
	// categorically different from every other validation failure: a bad magic, a
	// too-old version, or a CRC mismatch names a file this build cannot read AND has
	// no reason to preserve, so newShard rotates it aside and starts empty. A file
	// from a FUTURE build is one this (older) build cannot read but MUST NOT destroy —
	// rotating it aside would be silent data loss of a format a newer build wrote
	// deliberately. newShard uses errors.Is(err, errFutureVersion) to refuse the open
	// instead of rotating. See minReadableCacheVersion for the too-old direction.
	if version > cacheVersion {
		return 0, false, fmt.Errorf("cache: %w %d (this build writes %d)", errFutureVersion, version, cacheVersion)
	}
	if version < minReadableCacheVersion {
		return 0, false, fmt.Errorf("cache: unsupported version %d (want %d..%d)", version, minReadableCacheVersion, cacheVersion)
	}
	if pageSize != expectedPageSize {
		return 0, false, fmt.Errorf("cache: pageSize mismatch (file %d, config %d)", pageSize, expectedPageSize)
	}
	if numPages != expectedNumPages {
		return 0, false, fmt.Errorf("cache: numPages mismatch (file %d, config %d)", numPages, expectedNumPages)
	}
	return idx, false, nil
}

func isZeroPrefix(b []byte) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}
