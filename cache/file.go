// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"crypto/rand"
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

// Header layout (128 bytes total, version 5):
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
//	64..79  framingKey [16]byte — per-file random MAC secret   (version 5+)
//	80..83  framingKeyCRC uint32 (CRC32-IEEE of bytes 64..79)   (version 5+)
//	84..127 reserved (zero)
//
// The stamp fields carry their OWN CRC rather than extending headerCRC's range,
// so the bytes covered by headerCRC (0..27) are byte-identical between version 2
// and version 3. That keeps the two formats mutually intelligible for the fields
// they share: a v2 file opened by this build validates its core header normally
// and simply restores stamp=0 (the stamp CRC over eight zero bytes does not
// match), and a v3 file's applied index is readable by any build that ignores
// bytes 32..63.
//
// The PB FRONTIER (44..63) followed that precedent and carried NO version bump: it
// consumed previously reserved (zero-filled) bytes, guarded by its own CRC.
//
// THE FRAMING KEY (64..79) IS DIFFERENT AND FORCES v4→v5. Unlike the stamp and the
// PB frontier — new HEADER fields that repurposed reserved space and left the ENTRY
// codec alone — the framing key exists to key the per-entry MAC that REPLACED the
// per-entry CRC (see cache/ringbuf.go). The entry codec changed, so v4 and v5 frames
// are mutually unintelligible and the version gate must separate them; and in a v4
// header bytes 64..79 are not reserved header space at all — the v4 header is only 64
// bytes, so those bytes are the first of v4 page 0's data. The key is therefore read
// (and its CRC checked) ONLY for a v5 header; a v4 file is upgraded to v5 on open
// (cache/migrate.go), which writes a fresh random key into the new 128-byte header.
// The key's own CRC guards a torn key write: a v5 file with an unreadable key cannot
// verify any entry, so it is treated as a corrupt file (rotated aside) rather than
// silently recovering nothing.
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
	//
	//	v4 → v5: the per-entry integrity field flipped from a 4-byte CRC32 to an
	//	  8-byte keyed MAC (SipHash-2-4 over the page nonce, the entry's in-page
	//	  offset, the frame header, key and value, under a per-file random secret),
	//	  the entry header grew 26→30, the per-page header grew 8→16 (a per-page-life
	//	  nonce joined head/tail), and the file header grew 64→128 (the framing key
	//	  and its CRC). This is what makes recovery and eviction able to RESYNC past a
	//	  damaged entry instead of discarding the rest of the page (issue #135): a
	//	  keyed, position-bound MAC cannot be forged inside a client value nor replayed
	//	  to another offset, so a forward scan can trust "the next frame that verifies"
	//	  where a CRC never could. A v4 entry decoded with the v5 reader frames garbage,
	//	  so a v4 file is UPGRADED to v5 on open (cache/migrate.go) rather than read in
	//	  place — see minReadableCacheVersion.
	cacheVersion uint32 = 5
	// minReadableCacheVersion is the oldest on-disk version this build opens WITHOUT
	// rotating it aside.
	//
	// It stays 4 across the v5 bump: v4 is deployed persistent state (v0.7.0-beta*),
	// so a v4 pages file must NOT be thrown away. newShard detects a v4 header and
	// MIGRATES it to v5 in place — read via the v4 decoder, rewritten as a v5 file
	// through the same crash-safe temp+rename+dir-fsync swap compaction uses, then
	// mapped and served as v5 for the rest of its life (the hot path never branches on
	// version). A pre-v4 file (v1/v2/v3) is still rotated aside: those changed the
	// header/entry codec with no deployed state to preserve, so the same tested path
	// that handles a bad magic or CRC renames it .bad-<timestamp> and the shard starts
	// empty, to be rebuilt from the cluster log or a peer snapshot.
	//
	// The reverse direction is unchanged: a v5 file opened by an OLDER build trips
	// that build's version gate (v5 > its cacheVersion) and is refused non-destructively
	// via errFutureVersion. Downgrades below v4 reformat the DataDir.
	minReadableCacheVersion uint32 = 4
	headerSize              int    = 128
	pageHdrSize             int    = 16 // head u32 + tail u32 + pageNonce u64, at the start of each mmap page

	// hdrStampOff / hdrStampCRCOff locate the version-3 persisted logical clock.
	hdrStampOff    = 32
	hdrStampCRCOff = 40

	// hdrPBSeqOff / hdrPBEpochOff / hdrPBFrontierCRCOff locate the persisted
	// primary-backup applied frontier — the (seq, epoch) IDENTITY of the newest
	// PB write materialized into these pages. Unlike appliedIndex (a Raft log
	// index) a PB position is meaningless without its epoch, because Promote
	// continues seq assignment from the promoted node's high-water and so REUSES
	// seqs across epochs; the pair is stored and CRC'd as one unit for that
	// reason.
	hdrPBSeqOff         = 44
	hdrPBEpochOff       = 52
	hdrPBFrontierCRCOff = 60

	// framingKeyOff / framingKeyLen / framingKeyCRCOff locate the version-5 per-file
	// MAC secret and its guard CRC. framingKeyLen is 16 bytes — the SipHash key
	// width. The key is written once, with crypto/rand, by writeFramingKey when a
	// fresh (or migrated) file's header is created, and never changes afterwards, so
	// the mapped bytes are a stable secret every page's MAC is keyed by.
	framingKeyOff    = 64
	framingKeyLen    = 16
	framingKeyCRCOff = 80
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

// writeHeader writes a fresh v5 header to region, INCLUDING a freshly generated
// per-file framing key. Caller must ensure region is at least headerSize bytes.
// Zero-fills bytes 32..127 before stamping the sub-fields. Returns an error only if
// the framing key could not be generated (crypto/rand failure); a header with no
// valid framing key must never be published, so the caller aborts the open.
func writeHeader(region []byte, pageSize, numPages uint32, appliedIdx uint64) error {
	binary.LittleEndian.PutUint64(region[0:8], cacheMagic)
	binary.LittleEndian.PutUint32(region[8:12], cacheVersion)
	binary.LittleEndian.PutUint32(region[12:16], pageSize)
	binary.LittleEndian.PutUint32(region[16:20], numPages)
	binary.LittleEndian.PutUint64(region[20:28], appliedIdx)
	// Reserved bytes 32..127 start zero; the sub-fields below overwrite their slots.
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
	// The v5 per-file MAC secret. Its own CRC guards a torn write.
	return writeFramingKey(region)
}

// writeFramingKey generates a fresh random 16-byte framing key into the v5 header
// and stamps its guard CRC. Called once per file, when a fresh (or migrated)
// header is created. A crypto/rand failure is returned rather than swallowed: a
// zero or partial key is a broken secret, so the file must not be published.
func writeFramingKey(region []byte) error {
	if _, err := rand.Read(region[framingKeyOff : framingKeyOff+framingKeyLen]); err != nil {
		return fmt.Errorf("cache: generate framing key: %w", err)
	}
	crc := crc32.ChecksumIEEE(region[framingKeyOff : framingKeyOff+framingKeyLen])
	binary.LittleEndian.PutUint32(region[framingKeyCRCOff:framingKeyCRCOff+4], crc)
	return nil
}

// randomNonce draws a fresh 8-byte per-page-life nonce. Used at every extent-reuse
// point to rotate the page nonce (see zeroDurableBoundsForReuseLocked).
func randomNonce() (uint64, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, fmt.Errorf("cache: generate page nonce: %w", err)
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

// readFramingKey returns the per-file MAC secret and true, or (nil, false) if the
// key's guard CRC does not match (a torn key write, or a pre-v5 header whose bytes
// 64..79 are not a key at all). The returned slice ALIASES region; callers that
// must outlive the mapping (the shard, which survives a compaction remap) copy it.
// Only meaningful on a v5 header — validateHeader gates the call on version.
func readFramingKey(region []byte) ([]byte, bool) {
	if len(region) < headerSize {
		return nil, false
	}
	stored := binary.LittleEndian.Uint32(region[framingKeyCRCOff : framingKeyCRCOff+4])
	if stored != crc32.ChecksumIEEE(region[framingKeyOff:framingKeyOff+framingKeyLen]) {
		return nil, false
	}
	return region[framingKeyOff : framingKeyOff+framingKeyLen], true
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
	// A CURRENT-version (v5) header must carry a readable framing key: every entry's
	// MAC is keyed by it, so an unreadable key makes the whole file unverifiable.
	// Treat a torn key as an ordinary rotatable corruption (like a bad entry CRC would
	// have been), NOT as errFutureVersion. The check is gated on version because in a
	// v4 header bytes 64..79 are page data, not a key — a v4 file validates here and is
	// then migrated (newShard), never read in place.
	if version == cacheVersion {
		if _, ok := readFramingKey(region); !ok {
			return 0, false, errors.New("cache: framing key CRC mismatch (torn or missing per-file MAC secret)")
		}
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
