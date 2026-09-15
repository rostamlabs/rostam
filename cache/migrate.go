// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"log/slog"
	"os"
)

// v4 → v5 migrate-on-open.
//
// v4 is deployed persistent state (v0.7.0-beta*), so a v4 pages file must be UPGRADED
// to v5, never rotated aside. The two formats are mutually unintelligible — v4 frames
// entries with a 4-byte CRC after a 26-byte header and an 8-byte per-page header, v5
// with an 8-byte keyed MAC after a 30-byte header and a 16-byte per-page header — so
// this code reads v4 with its OWN decoder (the ONLY place the v4 codec survives) and
// rewrites the live set as a fresh v5 file, which is then served for the rest of the
// shard's life. Nothing on the hot path ever branches on version.
//
// FAIL-CLOSED. If the rewrite cannot be staged or made durable (disk full, I/O
// error, crypto/rand failure), the open FAILS and the intact v4 file is left exactly
// where it is — it is never rotated aside, because rotating committed state a cluster
// may not still hold elsewhere is data loss. The rewrite goes through the same
// crash-safe temp+rename+directory-fsync swap cold compaction uses (compactAtOpen), so
// a crash at any point leaves either the intact v4 file or the complete v5 file.
//
// NOTE the v4 file has already been truncate-EXTENDED to the v5 size by newShard's
// mmap (the v5 header is 64 bytes larger). That appends trailing zeros past the last
// v4 page and does not touch any v4 byte, so the mapping still reads the v4 layout
// correctly and an older v4 build reopening the file (mapping only its own smaller
// size) still sees intact data. If migration keeps failing, each open re-attempts it.

const (
	v4HeaderSize      = 64
	v4PageHdrSize     = 8 // v4 per-page header: head u32 + tail u32 (no nonce)
	v4EntryHeaderSize = 2 + 4 + 8 + 8 + 4
	v4EntryMetaOff    = 14
	v4EntryCRCOff     = 22
)

// v4CRCTable is the polynomial the v4 entry codec used (CRC32-IEEE), kept local to
// the migration path now that the live codec is a keyed MAC.
var v4CRCTable = crc32.IEEETable

// errV4CRCMismatch marks a v4 entry whose stored CRC does not match — a torn write or
// bit rot. Recovery of a v4 file has no forge-proof framing to resync on (that is the
// whole reason for v5), so a v4 tear simply ends that page's walk, exactly as the v4
// build's own recovery truncated the page tail.
var errV4CRCMismatch = errors.New("cache: v4 entry CRC mismatch")

// decodeV4Entry decodes one v4 frame (CRC-verified). Returns the key/value (aliasing
// src), expiry and meta. Bounds are guarded exactly as the v4 decoder did, including
// the 32-bit valLen-overflow guards, because these bytes come straight off disk.
func decodeV4Entry(src []byte) (key, value []byte, expiryMs, meta uint64, err error) {
	if len(src) < v4EntryHeaderSize {
		return nil, nil, 0, 0, errEntryTruncated
	}
	keyLen := int(binary.LittleEndian.Uint16(src[0:2]))
	valLen := int(binary.LittleEndian.Uint32(src[2:6]))
	if valLen < 0 {
		return nil, nil, 0, 0, errEntryTruncated
	}
	if valLen > len(src)-v4EntryHeaderSize-keyLen {
		return nil, nil, 0, 0, errEntryTruncated
	}
	expiryMs = binary.LittleEndian.Uint64(src[6:14])
	meta = binary.LittleEndian.Uint64(src[v4EntryMetaOff:v4EntryCRCOff])
	storedCRC := binary.LittleEndian.Uint32(src[v4EntryCRCOff:v4EntryHeaderSize])
	total := v4EntryHeaderSize + keyLen + valLen
	crc := crc32.Checksum(src[0:v4EntryCRCOff], v4CRCTable)
	crc = crc32.Update(crc, v4CRCTable, src[v4EntryHeaderSize:total])
	if crc != storedCRC {
		return nil, nil, 0, 0, errV4CRCMismatch
	}
	key = src[v4EntryHeaderSize : v4EntryHeaderSize+keyLen]
	value = src[v4EntryHeaderSize+keyLen : total]
	return key, value, expiryMs, meta, nil
}

// v4Winner records the max-seq physical copy of a key across the v4 pages — the copy
// a warm restart would have resolved the key's index slot to.
type v4Winner struct {
	seq       uint64
	pageIdx   int
	off       int
	tombstone bool
}

// walkV4Pages visits every decodable v4 frame in page order, then offset order —
// the SAME order rebuildIndexFromPages walks, so a first-seen copy wins a seq tie. A
// page whose durable bounds are out of range is skipped whole (as v4 recovery reset
// it); a torn frame ends that page's walk (v4 had no safe resync). visit receives the
// physical position so the pack pass can recognise the winner.
func (s *shard) walkV4Pages(v4region []byte, visit func(pageIdx, off int, key, value []byte, exp, meta uint64)) {
	maxPages := s.cfg.MaxPagesPerShard()
	pageSize := s.cfg.PageSize
	for i := 0; i < maxPages; i++ {
		base := v4HeaderSize + i*pageSize
		head := int(binary.LittleEndian.Uint32(v4region[base : base+4]))
		tail := int(binary.LittleEndian.Uint32(v4region[base+4 : base+8]))
		ent := v4region[base+v4PageHdrSize : base+pageSize]
		if head < 0 || tail < head || tail > len(ent) {
			continue
		}
		for cursor := head; cursor < tail; {
			key, value, exp, meta, derr := decodeV4Entry(ent[cursor:tail])
			if derr != nil {
				break
			}
			visit(i, cursor, key, value, exp, meta)
			cursor += v4EntryHeaderSize + len(key) + len(value)
		}
	}
}

// migrateV4ToV5 rewrites the v4 file mapped at (v4file, v4region) as a v5 file and
// remaps it as the shard's backing store, leaving the shard ready for the normal v5
// rebuild in newShard. Consumes the v4 mapping (unmaps it on every path). On any
// failure before the atomic rename the intact v4 file is kept and an error returned;
// after the rename the swap has already published the v5 file. Construction-time only
// (the shard is not yet shared), so no lock is held.
//
// LIVENESS. It packs every index-current, NON-tombstone copy — the max-seq winner of
// each key that is not a delete record. It deliberately does NOT apply TTL expiry or
// the flush floor here: the v5 file it writes is then opened by the normal path, whose
// rebuild + stripDeadSlots apply expiry (mode-appropriately) and skip flushed
// sequences, exactly as they would for any freshly written v5 file. So a migrated
// shard ends in the same logical state a live v5 shard with the same history would —
// and on a replicated shard nothing is physically dropped beyond deletes/supersedes,
// which are node-local and already differ between replicas.
func (s *shard) migrateV4ToV5(dataDir, pagesPath string, v4file *os.File, v4region []byte, size int64, appliedIdx uint64) error {
	maxPages := s.cfg.MaxPagesPerShard()
	pageSize := s.cfg.PageSize

	v4Closed := false
	closeV4 := func() {
		if !v4Closed {
			v4Closed = true
			_ = munmapAndClose(v4file, v4region)
		}
	}
	defer closeV4()

	// PASS 1: resolve each key to its max-seq copy.
	winners := make(map[string]v4Winner)
	s.walkV4Pages(v4region, func(pageIdx, off int, key, _ []byte, _, meta uint64) {
		seq := metaSeq(meta)
		k := string(key)
		if w, ok := winners[k]; ok && w.seq >= seq {
			return // an earlier-walked copy already holds >= seq (first-seen wins a tie)
		}
		winners[k] = v4Winner{seq: seq, pageIdx: pageIdx, off: off, tombstone: metaIsTombstone(meta)}
	})

	// STAGE the v5 rewrite into a sibling temp file, fully reserved so a full disk
	// fails here as an ordinary error (not a SIGBUS mid-write) — see mmapFileAlloc.
	tmpPath := compactTmpPath(pagesPath)
	tmpFile, tmpRegion, aerr := mmapFileAlloc(tmpPath, size, size, false)
	if aerr != nil {
		return fmt.Errorf("cache: migrate v4→v5: cannot stage %s (v4 file left intact): %w", tmpPath, aerr)
	}
	cleanupTmp := func() {
		_ = munmapAndClose(tmpFile, tmpRegion)
		_ = os.Remove(tmpPath)
	}
	//nolint:gosec // PageSize and MaxPagesPerShard are validated positive
	if herr := writeHeader(tmpRegion, uint32(pageSize), uint32(maxPages), appliedIdx); herr != nil {
		cleanupTmp()
		return fmt.Errorf("cache: migrate v4→v5: header (v4 file left intact): %w", herr)
	}
	// Carry the logical clock and PB frontier across; their header offsets are the
	// same in v4 and v5, and the migrated file materialises the same live writes, so
	// the values that described the v4 file still describe the v5 one.
	setAppliedStamp(tmpRegion, readAppliedStamp(v4region))
	pbSeq, pbEpoch := readPBFrontier(v4region)
	setPBFrontier(tmpRegion, pbSeq, pbEpoch)
	stageKey, ok := readFramingKey(tmpRegion)
	if !ok {
		cleanupTmp()
		return errors.New("cache: migrate v4→v5: staged framing key unreadable (v4 file left intact)")
	}
	stageKey = append([]byte(nil), stageKey...)

	// PACK the live winners into v5 dst pages, re-encoding each through page.Write so it
	// lands with a fresh MAC under the staging key at its new offset (nonce 0, a fresh
	// page's first life). Destination pages are materialised lazily, one at a time.
	var dstPages []*page
	di := 0
	dstPage := func(i int) *page {
		off := headerSize + i*pageSize
		p := newMmapPage(tmpRegion[off : off+pageSize : off+pageSize])
		p.framingKey = stageKey
		p.nonce = 0
		p.Reset()
		dstPages = append(dstPages, p)
		return p
	}
	cur := dstPage(0)
	fits := true
	s.walkV4Pages(v4region, func(pageIdx, off int, key, value []byte, exp, meta uint64) {
		if !fits {
			return
		}
		w := winners[string(key)]
		if w.pageIdx != pageIdx || w.off != off || w.tombstone {
			return // not the winning copy for its key, or a delete record — dropped
		}
		for {
			if _, _, werr := cur.Write(key, value, exp, makeMeta(w.seq, false)); werr == nil {
				return
			}
			di++
			if di >= maxPages {
				fits = false
				return
			}
			cur = dstPage(di)
		}
	})
	if !fits {
		cleanupTmp()
		return fmt.Errorf("cache: migrate v4→v5: live set does not fit in %d pages (v4 file left intact)", maxPages)
	}
	// Project the packed runtime bounds into each staging page header (page.Write only
	// advanced the runtime bounds). Safe by construction: the staging file is msync'd +
	// fsync'd and atomically renamed before it is ever mapped as backing store, so the
	// bounds and the entries they name reach disk together.
	for _, p := range dstPages {
		p.projectBounds(p.head(), p.tail())
	}
	stageErr := msync(tmpFile, tmpRegion)
	if stageErr == nil {
		stageErr = tmpFile.Sync() // data + metadata durable before anything points here
	}
	if cerr := munmapAndClose(tmpFile, tmpRegion); cerr != nil && stageErr == nil {
		stageErr = cerr
	}
	if stageErr != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("cache: migrate v4→v5: staging durability failed (v4 file left intact): %w", stageErr)
	}

	// SWAP: drop the v4 mapping, then publish the v5 file atomically. A crash before
	// the rename leaves the intact v4 file plus a stale temp the next open discards; a
	// crash during it resolves to old or new; a failed rename keeps the v4 file.
	closeV4()
	if rerr := os.Rename(tmpPath, pagesPath); rerr != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("cache: migrate v4→v5: publish rename (v4 file left intact): %w", rerr)
	}
	// The directory fsync makes the rename durable. Unlike cold compaction — where a
	// lost rename merely reappears as the intact original and losing it costs only the
	// compaction — this is a MIGRATION of committed v4 data, so we must fail closed: if
	// the fsync fails the rename may not survive a crash. Return the error and abort the
	// open. On the next open the shard finds either the intact v4 file (rename lost, it
	// re-migrates) or the complete v5 file (rename held) — both are whole files, so
	// failing this open loses nothing.
	if derr := syncDir(dataDir); derr != nil {
		return fmt.Errorf("cache: migrate v4→v5: directory fsync after publish (retry the open): %w", derr)
	}

	// MAP + ATTACH the v5 file (no rebuild here — newShard rebuilds it next, on the
	// same path any v5 file takes).
	file, region, merr := mmapFile(pagesPath, size, s.cfg.Mlock)
	if merr != nil {
		return fmt.Errorf("cache: migrate v4→v5: remap %s: %w", pagesPath, merr)
	}
	//nolint:gosec // PageSize and MaxPagesPerShard are validated positive
	_, freshV5, verr := validateHeader(region, uint32(pageSize), uint32(maxPages))
	if verr != nil || freshV5 {
		_ = munmapAndClose(file, region)
		if verr == nil {
			verr = errors.New("migrated file is all-zero")
		}
		return fmt.Errorf("cache: migrate v4→v5: migrated file failed validation: %w", verr)
	}
	s.framingKey = stageKey
	s.attachMmapRegion(file, region)
	slog.Info("upgraded persistent cache file from v4 to v5",
		"component", "cache", "path", pagesPath, "live_keys", len(winners))
	return nil
}
