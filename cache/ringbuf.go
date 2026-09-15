// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"errors"
)

// Entry wire layout (within a page), format version 5:
//
//	[keyLen:2][valueLen:4][expiryMs:8][meta:8][mac:8][key][value]
//	└──────────────────entryHeaderSize (30)──────────────────┘
//
// Little-endian throughout, matching the memory layout on amd64/arm64 and
// consistent with the page head/tail offsets in page.go, which are also
// little-endian.
//
// THE INTEGRITY FIELD IS A KEYED MAC, NOT A CRC (this is the v4→v5 change). The
// 8-byte slot at [22:30] holds SipHash-2-4 over
//
//	framingKey :: ( pageNonce ‖ uint32LE(offset) ‖ frame[0:22] ‖ key ‖ value )
//
// where framingKey is a per-file 16-byte secret (cache/file.go), pageNonce is the
// per-page-life nonce from the page header (cache/page.go), and offset is the
// entry's byte offset within the page's entry region. See the whole argument in
// cache/siphash.go and rebuildIndexFromPages (cache/shard.go): a CRC bounds a
// RANDOM false match but not a CHOSEN one, so recovery cannot use it to resynchronise
// past a torn entry — a client VALUE is attacker-chosen bytes and can contain a
// complete, correctly-CRC'd frame with a chosen sequence (the GHSA-m63m-rp87-w4rf
// forge class). A keyed MAC a client cannot compute makes a forged frame verify with
// probability 2^-64, and binding the nonce+offset in makes a GENUINE frame's bytes
// unverifiable if copied to a different position or into a reused extent — so
// recovery and eviction can safely resync forward to the next frame that verifies
// and lose only the damaged entry, not the rest of the page.
//
// v3→v4 grew the META word (write sequence + tombstone) between the expiry and the
// integrity slot. v4→v5 replaces the 4-byte CRC with the 8-byte MAC (the slot moved
// from [22:26] to [22:30]) and widened the header from 26 to 30 bytes; nothing else
// in the field order changed. A v4 entry decoded with the v5 reader frames garbage,
// so the two codecs are mutually unintelligible and the per-FILE version gate is the
// only thing separating them — a v4 file is upgraded to v5 on open (cache/migrate.go)
// rather than read in place.
//
// WHY meta STILL SITS BEFORE THE KEY. It is the only placement that leaves the hot
// lock-free read path untouched. decodeEntryFast reads src[0:2], src[2:6],
// src[6:14] and then slices the key/value at entryHeaderSize; the MAC at [22:30]
// changed exactly one compile-time CONSTANT in that function and nothing else — same
// loads, same order, no branch, and the read NEVER touches meta OR the MAC. The hot
// path pays zero MAC cost: it resolves an entry through the index, which was
// populated only after a MAC-verified decode, so re-verifying on read would be
// redundant work on the latency-critical path. decodeEntryFast's signature is
// deliberately left unchanged for the same reason: no read-path caller may acquire a
// way to ask for the sequence number OR to verify the MAC (which would need the
// framing key + nonce + offset it has no business threading in).
//
// meta = flags<<56 | seq:
//
//	bits 0..55  seq — the shard's per-entry MONOTONIC write sequence. It is the
//	            persisted write-recency signal that makes warm restart correct:
//	            page order is NOT write order (findOrMakePageLocked revisits lower
//	            pages through firstPageWithRoomLocked), so rebuildIndexFromPages
//	            resolves a key to the copy with the HIGHEST seq rather than to the
//	            last one the page walk happens to reach (#12A).
//	bit 56      entryFlagTombstone — this entry RECORDS A DELETE. Del on a
//	            persistent shard appends one so the removal is part of the page
//	            bytes and survives the rebuild (#12B); the rebuild strips the slot
//	            afterwards and compaction drops the tombstone and every older copy
//	            of its key in the same pass.
//	bits 57..63 reserved (must be zero).
//
// 56 bits of sequence is 7.2e16 writes per shard — at ten million writes a second
// that is 228 years, so wrap is not a condition the code needs to handle.
const (
	entryHeaderSize = 2 + 4 + 8 + 8 + 8
	// entryMetaOff / entryMACOff locate the meta word and the MAC slot. The MAC's
	// covered fixed range is frame[0:entryMACOff] (keyLen, valLen, expiry, meta),
	// plus the key and value bytes — everything except the MAC slot itself.
	entryMetaOff = 14
	entryMACOff  = 22
	maxKeyLen    = 1<<16 - 1 // uint16 max
	// maxValueLen is typed int64 (not the untyped-int default) because it is the
	// uint32 max, 4294967295, which overflows the 32-bit `int` of a 386/arm/
	// windows-386 build. int64 holds it on every platform; the len(value)
	// comparison below widens to int64 to match, which costs nothing on 64-bit
	// (where int already covers the full range) and is simply always-false on
	// 32-bit (where a slice can never reach 2^32-1 elements) rather than a
	// truncated, lower limit.
	maxValueLen int64 = 1<<32 - 1 // uint32 max (4 GiB - 1)

	// entryFlagTombstone marks an entry as a delete record. entrySeqMask isolates
	// the sequence number from the flag bits.
	entryFlagTombstone uint64 = 1 << 56
	entrySeqMask       uint64 = entryFlagTombstone - 1
)

var (
	// errBufferTooSmall indicates the destination slice is too small for the entry.
	errBufferTooSmall = errors.New("ringbuf: buffer too small")
	// errKeyTooLong indicates the key exceeds maxKeyLen.
	errKeyTooLong = errors.New("ringbuf: key too long")
	// errValueTooLong indicates the value exceeds maxValueLen.
	errValueTooLong = errors.New("ringbuf: value too long")
	// errEntryTruncated indicates the source slice is shorter than the entry header
	// claims, or the header itself is missing.
	errEntryTruncated = errors.New("ringbuf: entry truncated")
	// errMACMismatch indicates the stored MAC does not match the computed MAC — the
	// frame is corrupt, was written under a different nonce/offset (a reused extent
	// or a moved frame), or is a forgery a client planted in a value. Recovery and
	// eviction treat all three the same: the frame is not this entry, resync past it.
	errMACMismatch = errors.New("ringbuf: MAC mismatch")
)

// makeMeta packs a write sequence and the tombstone flag into an entry's meta word.
//
// The mask is a WRAP, not a truncation. seq comes from shard.writeSeq, a full
// uint64 counter, so at 2^56 the stored sequence returns to 0 and NEWER entries
// start carrying LOWER sequences than older ones — which the rebuild reads as
// "the older copy is more recent" and silently resolves the key to stale data
// (#12A, reintroduced). It is not a saturating clamp and there is no wrap
// handling anywhere; the 56-bit budget is what makes that unnecessary, at ten
// million writes a second per shard it is 228 years.
func makeMeta(seq uint64, tombstone bool) uint64 {
	m := seq & entrySeqMask
	if tombstone {
		m |= entryFlagTombstone
	}
	return m
}

// metaSeq returns the write sequence carried by a meta word.
func metaSeq(meta uint64) uint64 { return meta & entrySeqMask }

// metaIsTombstone reports whether a meta word marks a delete record.
func metaIsTombstone(meta uint64) bool { return meta&entryFlagTombstone != 0 }

// entryMetaAt reads the meta word of the entry framed at src[0:]. Callers that
// already decoded the entry (so len(src) >= entryHeaderSize is established) use
// this instead of a wider decode signature, which keeps the meta word off the hot
// read path entirely. Returns 0 for a slice too short to hold a header.
func entryMetaAt(src []byte) uint64 {
	if len(src) < entryHeaderSize {
		return 0
	}
	return binary.LittleEndian.Uint64(src[entryMetaOff:entryMACOff])
}

// entryMAC computes the keyed frame MAC. header22 is the fixed frame prefix
// frame[0:entryMACOff] (keyLen, valLen, expiry, meta) and payload is the contiguous
// key‖value bytes. The nonce and offset are folded in AHEAD of the frame bytes so
// the same key/value/header verify ONLY at the (page-life, position) they were
// written at — copying a genuine frame to another offset, or into an extent whose
// nonce has since rotated, changes the MAC input and the copy fails to verify. This
// is what makes forward resync safe: a frame only verifies where it genuinely
// belongs. framingKey is the 16-byte per-file secret.
//
// Streamed through sipHasher so no contiguous nonce‖offset‖header‖payload buffer is
// allocated on the write path (one MAC per stored entry).
func entryMAC(framingKey []byte, nonce uint64, offset uint32, header22, payload []byte) uint64 {
	k0, k1 := sipKeyHalves(framingKey)
	h := newSipHasher(k0, k1)
	var pre [12]byte
	binary.LittleEndian.PutUint64(pre[0:8], nonce)
	binary.LittleEndian.PutUint32(pre[8:12], offset)
	h.write(pre[:])
	h.write(header22)
	h.write(payload)
	return h.sum()
}

// encodeEntry writes an entry into dst starting at index 0 and stamps its keyed
// MAC, computed at (nonce, offset) under framingKey. Returns the number of bytes
// written. This is the mmap/durable write path (reached through page.Write, which
// threads its page nonce and the entry's in-page offset in); heap-mode shards use
// [encodeEntryNoCRC], which lays down no integrity field because a heap page is
// never persisted and never participates in recovery.
func encodeEntry(dst, key, value []byte, expiryMs, meta uint64, framingKey []byte, nonce uint64, offset uint32) (int, error) {
	n, err := encodeEntryHeader(dst, key, value, expiryMs, meta)
	if err != nil {
		return 0, err
	}
	// The MAC covers the fixed header prefix [0:entryMACOff) and the key+value bytes
	// [entryHeaderSize:n) — everything except the MAC slot itself. encodeEntryHeader
	// returned without error, which guarantees len(dst) >= n and n >= entryHeaderSize,
	// so all three fixed offsets are provably within dst.
	mac := entryMAC(framingKey, nonce, offset, dst[0:entryMACOff], dst[entryHeaderSize:n]) //nolint:gosec // bounds invariant after encodeEntryHeader
	binary.LittleEndian.PutUint64(dst[entryMACOff:entryHeaderSize], mac)                   //nolint:gosec // len(dst) >= entryHeaderSize after encodeEntryHeader
	return n, nil
}

// encodeEntryNoCRC writes an entry WITHOUT an integrity field. Safe for heap-mode
// shards: the index is fully in memory, page bytes can't be corrupted by anything
// external (no mmap, no disk), and the only path that consumes the MAC —
// rebuildIndexFromPages / eviction resync — never runs for heap-backed pages. The
// MAC slot is left untouched (whatever bytes the previous occupant left there);
// decodeEntryFast doesn't read it. (The name is kept from the v4 CRC era; the field
// it once skipped is now the MAC, and a heap page has neither.)
func encodeEntryNoCRC(dst, key, value []byte, expiryMs, meta uint64) (int, error) {
	return encodeEntryHeader(dst, key, value, expiryMs, meta)
}

// encodeEntryHeader writes the keyLen / valLen / expiry / meta / key / value
// fields, returning the total bytes written. The MAC slot at
// dst[entryMACOff:entryHeaderSize] is left for the caller to fill (or skip).
func encodeEntryHeader(dst, key, value []byte, expiryMs, meta uint64) (int, error) {
	if len(key) > maxKeyLen {
		return 0, errKeyTooLong
	}
	if int64(len(value)) > maxValueLen {
		return 0, errValueTooLong
	}
	total := entryHeaderSize + len(key) + len(value)
	if len(dst) < total {
		return 0, errBufferTooSmall
	}
	binary.LittleEndian.PutUint16(dst[0:2], uint16(len(key)))   //nolint:gosec // len(key) <= maxKeyLen
	binary.LittleEndian.PutUint32(dst[2:6], uint32(len(value))) //nolint:gosec // len(value) <= maxValueLen
	binary.LittleEndian.PutUint64(dst[6:14], expiryMs)
	binary.LittleEndian.PutUint64(dst[entryMetaOff:entryMACOff], meta)
	copy(dst[entryHeaderSize:entryHeaderSize+len(key)], key)
	copy(dst[entryHeaderSize+len(key):total], value)
	return total, nil
}

// decodeEntry reads an entry from src and returns its key, value, expiry and meta
// (key/value reference into src — zero-copy), VERIFYING the keyed MAC at (nonce,
// offset) under framingKey. It is the recovery/eviction decoder: rebuildIndexFromPages
// and the drain resync it forward frame by frame, and a frame that does not verify at
// the position it is being read from is rejected with errMACMismatch. Use
// decodeEntryFast on hot paths where the slabRef has already vouched for the entry
// (see [decodeEntryFast]).
func decodeEntry(src, framingKey []byte, nonce uint64, offset uint32) (key, value []byte, expiryMs, meta uint64, err error) {
	if len(src) < entryHeaderSize {
		return nil, nil, 0, 0, errEntryTruncated
	}
	keyLen := int(binary.LittleEndian.Uint16(src[0:2]))
	valLen := int(binary.LittleEndian.Uint32(src[2:6]))
	// On a 32-bit platform int is 32 bits, so a corrupted/torn valLen near the
	// uint32 max widens NEGATIVE instead of huge. Reject that before it reaches
	// the arithmetic below: a negative valLen would shrink total, potentially
	// passing the length check with a bogus (or negative, panicking) slice
	// bound. On 64-bit this is always false — valLen tops out at maxValueLen,
	// which fits comfortably positive — so the check costs nothing there.
	if valLen < 0 {
		return nil, nil, 0, 0, errEntryTruncated
	}
	// BOUND valLen BEFORE SUMMING IT. On 32-bit, a POSITIVE valLen within keyLen+30 of
	// the int32 max still wraps the total negative, which then passes a
	// `len(src) < total` test and panics on the slice below — at warm restart, where
	// these bytes come straight off disk, on every restart. The subtraction cannot wrap:
	// len(src) >= entryHeaderSize and keyLen <= maxKeyLen. Identical on 64-bit to
	// comparing the sum.
	if valLen > len(src)-entryHeaderSize-keyLen {
		return nil, nil, 0, 0, errEntryTruncated
	}
	expiryMs = binary.LittleEndian.Uint64(src[6:14])
	meta = binary.LittleEndian.Uint64(src[entryMetaOff:entryMACOff])
	storedMAC := binary.LittleEndian.Uint64(src[entryMACOff:entryHeaderSize])

	total := entryHeaderSize + keyLen + valLen

	if entryMAC(framingKey, nonce, offset, src[0:entryMACOff], src[entryHeaderSize:total]) != storedMAC {
		return nil, nil, 0, 0, errMACMismatch
	}

	key = src[entryHeaderSize : entryHeaderSize+keyLen]
	value = src[entryHeaderSize+keyLen : total]
	return key, value, expiryMs, meta, nil
}

// decodeEntryFast reads an entry from src without verifying the MAC. Use on the hot
// Get/Del/sweep paths and on the write-path relocation walks: those reach an entry
// via the shard's in-memory index, which itself was populated only after a
// MAC-verified decode (at startup in rebuildIndexFromPages, or at write time after
// encodeEntry laid down a fresh MAC). The MAC slot is still written on encode so a
// future cold rebuild / eviction resync revalidates the frame.
//
// Its signature deliberately does NOT expose the meta word OR take a framing key /
// nonce / offset. Nothing on a read path may consult the write sequence or re-verify
// the MAC — a read resolves an entry through the index, which already encodes recency
// and integrity — so the hot decode stays exactly the three header loads plus two
// slices it has always been. Recovery-time callers that DO need the meta take it from
// [entryMetaAt] separately, and the ones that must VERIFY use [decodeEntry].
func decodeEntryFast(src []byte) (key, value []byte, expiryMs uint64, err error) {
	if len(src) < entryHeaderSize {
		return nil, nil, 0, errEntryTruncated
	}
	keyLen := int(binary.LittleEndian.Uint16(src[0:2]))
	valLen := int(binary.LittleEndian.Uint32(src[2:6]))
	// See the identical guard in decodeEntry: on a 32-bit platform a valLen near
	// the uint32 max widens NEGATIVE through int(), which would undershoot total
	// below and pass the length check with a bogus slice bound. No-op on 64-bit.
	if valLen < 0 {
		return nil, nil, 0, errEntryTruncated
	}
	// Bounded before it is summed, for the reason given in decodeEntry.
	if valLen > len(src)-entryHeaderSize-keyLen {
		return nil, nil, 0, errEntryTruncated
	}
	expiryMs = binary.LittleEndian.Uint64(src[6:14])

	total := entryHeaderSize + keyLen + valLen

	key = src[entryHeaderSize : entryHeaderSize+keyLen]
	value = src[entryHeaderSize+keyLen : total]
	return key, value, expiryMs, nil
}

// entrySpanExact returns the EXACT byte size an entry's framing occupies: the
// header, the key and the value, and nothing else. It is the encoder's length —
// precisely what encodeEntry / encodeEntryNoCRC will write — so it is the right
// figure wherever the question is "how many bytes does this framing consume" and
// the wrong one wherever the question is "how much of the page does this entry
// OCCUPY".
//
// Today those two questions have the same answer everywhere, which is why this
// used to be one function (entrySize) and why every caller of it was right by
// accident. Splitting them is deliberate: a call site that has recorded WHICH of
// the two it means is a call site nobody has to re-derive later.
//
// FOUR PLACES COMPUTE THIS SUM INLINE INSTEAD OF CALLING IT, and none of them is
// an oversight: encodeEntryHeader, decodeEntry and decodeEntryFast DEFINE the
// framing rather than predicting it (their `total` is the layout itself, taken
// from the header's own fields), and page.EvictFront reads those fields out of
// crash-exposed bytes behind its own overflow guards. Note that the rename that
// introduced this function could not force a visit to any of the four; they were
// found by grep, not by the compiler.
func entrySpanExact(keyLen, valueLen int) int {
	return entryHeaderSize + keyLen + valueLen
}

// entrySpan returns the bytes an entry OCCUPIES on a page. padded says which
// question the caller is asking:
//
//   - padded=false — the exact framing, byte for byte. For a caller that is
//     measuring or reproducing an encode.
//   - padded=true — the occupancy: the room an append reserves, the amount a page
//     walk steps by, the extent a band check must contain. For a caller that is
//     reasoning about page space rather than about encoder output.
//
// BOTH RETURN THE SAME NUMBER TODAY, because classOf is an identity and there are
// therefore no size classes for an entry to be padded into. The distinction is
// carried anyway so that it is recorded at each site while the two are still
// provably equal, rather than being reconstructed from scratch by whoever first
// makes them differ.
//
// # The capacity/write rule
//
// WHEREVER A CAPACITY DECISION AND THE WRITE IT AUTHORISES DERIVE THEIR SIZE
// INDEPENDENTLY, BOTH MUST USE THE OCCUPANCY SPAN.
//
// This is the one failure this whole split exists to prevent, and it is a single
// shape wearing several costumes. A capacity check that measures LESS than the
// write reserves says yes and then hands the write an errPageFull its caller
// treats as unreachable — every one of those call sites comments the error as
// "the budget already proved the room". The check does not have to be wrong in
// isolation; it only has to disagree with page.Write, which reserves the
// occupancy. So the two numbers must come from the same question, not merely be
// equal by coincidence.
//
// Every such pair in the package, all of them occupancy on both sides:
//
//	page.Write               FreeTail check      / its own tail advance
//	putAtExpLocked           findOrMakePageLocked / the page.Write after it
//	delH                     findOrMakePageLocked / the tombstone page.Write
//	relocateIntoFreedPageLocked  the eviction budget  / dst.Write
//	reserveMoveVictim        reserveDestinationLocked / dst.Write
//	tryRelocatePageLocked    findRelocDestLocked  / the destination page.Write
//	packPagesNeeded          the next-fit frontier / packLiveInto's page.Write
//
// The room helpers themselves (findOrMakePageLocked, firstPageWithRoomLocked,
// evictUntilFitsLocked, findRelocDestLocked, reserveDestinationLocked) take a
// `need` and compare it against FreeTail, so they are correct for whatever their
// callers pass and the rule binds the CALLERS.
//
// page.Write is the anchor the rest are checked against: it is the only place
// that turns a span into reserved bytes, and TestPageWriteReservesTheOccupancySpan
// pins it to entrySpan(..., true) exactly.
func entrySpan(keyLen, valueLen int, padded bool) int {
	if padded {
		return entrySpanExact(keyLen, classOf(valueLen))
	}
	return entrySpanExact(keyLen, valueLen)
}

// classOf maps a value length to the length of the size CLASS that holds it — the
// value length an entry framed for that value would actually reserve.
//
// IT IS THE IDENTITY FUNCTION. There are no size classes: every entry reserves
// exactly its own value's length, which is what the whole package does today and
// what makes entrySpan(padded=true) equal to entrySpanExact. It exists now, ahead
// of any class scheme, so that the contract its tests pin —
//
//	classOf(v) >= v          a class must be able to hold its value
//	v <= w  ⇒  classOf(v) <= classOf(w)     classes are ordered like lengths
//	classOf(v) <= maxValueLen and never negative, right up to the top of the range
//
// — is already tested against a working implementation, and any later class
// function is a drop-in that either satisfies it or fails an existing test.
func classOf(valueLen int) int {
	return valueLen
}
