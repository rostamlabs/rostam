// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"errors"
	"time"
)

// errPageFull is returned when a page has no remaining contiguous capacity
// at its tail. Callers respond by evicting from the front (cache-mode) or
// by allocating a new page (slab pool).
var errPageFull = errors.New("page: full")

// pageBacking tags how page storage is allocated.
type pageBacking uint8

const (
	backingHeap pageBacking = iota
	backingMmap
)

// page is a fixed-size byte slab used for ring-buffer-style storage of
// variable-length entries. Pages are owned by a single shard; concurrency
// control lives in the shard, not here.
//
// In heap mode: data is a freshly-allocated byte slice; head/tail are
// kept in the heapHead/heapTail fields. data is entirely entry bytes.
//
// In mmap mode: data is a slice of an mmap'd region. The first
// pageHdrSize (8) bytes hold head (uint32 LE) + tail (uint32 LE).
// Entries occupy data[pageHdrSize:]. Storing head/tail in the file lets
// them survive process restart.
//
// Layout invariant: 0 <= head() <= tail() <= len(entries()).
// When head() == tail(), the page holds no live entries.
type page struct {
	data     []byte
	backing  pageBacking
	heapHead int
	heapTail int
	// gen is the page object's immutable generation, assigned by the shard at
	// construction (see shard.nextGen). Every slabRef minted for an entry in this
	// page carries this value; the lock-free heap ringbuf read path compares them
	// to detect a retired page without reading its bytes. Never mutated after
	// construction.
	gen uint16

	// retired marks an mmap page whose live entries were all relocated OUT by online
	// relocating compaction (cache/compact_online.go): no index slot addresses it any
	// more, so the write path must stop handing out its stale-framed tail
	// (firstPageWithRoomLocked skips it). Its bytes stay mapped and IMMUTABLE — Stage 1
	// never resets, reuses, unmaps, or punches them, so any in-flight lock-free reader
	// alias into the extent stays valid; the extent is stranded until a recycle stage
	// gives the slot a fresh extent. WRITE-PATH-ONLY state, guarded by shard.mu (the
	// lock-free read path never consults it — it resolves pages through pageSlots + the
	// generation gate). Always false on heap / single-node / ringbuf shards, so those
	// paths are byte-for-byte unchanged.
	retired bool

	// retiredAt is the wall-clock instant this page was marked retired (set only
	// when retired flips true). It starts the alias-drain QUARANTINE: the page's
	// stranded extent may not be RECYCLED (reset + handed back to the write path)
	// until at least Config.AliasQuarantine has elapsed since retiredAt, because a
	// lock-free reader that resolved a slabRef into the old content just before
	// retirement can still alias those bytes for up to the transport's write-deadline
	// bound. WALL time (real elapsed seconds), never the injected/logical cache clock:
	// the alias-hold bound is a real network-write duration. Zero on a non-retired
	// page and on every heap / single-node / ringbuf shard. Guarded by shard.mu.
	retiredAt time.Time
}

// newHeapPage allocates a heap-backed page of size bytes.
func newHeapPage(size int) *page {
	return &page{data: make([]byte, size), backing: backingHeap}
}

// newMmapPage wraps a slice of an mmap'd region. The slice must be at
// least pageHdrSize+1 bytes; entries occupy region[pageHdrSize:].
// Head/tail are read from / written to region[0:8].
func newMmapPage(region []byte) *page {
	return &page{data: region, backing: backingMmap}
}

func (p *page) head() int {
	if p.backing == backingMmap {
		return int(binary.LittleEndian.Uint32(p.data[0:4]))
	}
	return p.heapHead
}

func (p *page) tail() int {
	if p.backing == backingMmap {
		return int(binary.LittleEndian.Uint32(p.data[4:8]))
	}
	return p.heapTail
}

func (p *page) setHead(v int) {
	if p.backing == backingMmap {
		binary.LittleEndian.PutUint32(p.data[0:4], uint32(v)) //nolint:gosec // v is bounded by entries length
		return
	}
	p.heapHead = v
}

func (p *page) setTail(v int) {
	if p.backing == backingMmap {
		binary.LittleEndian.PutUint32(p.data[4:8], uint32(v)) //nolint:gosec // v is bounded by entries length
		return
	}
	p.heapTail = v
}

// entries returns the slice of data that holds entry bytes (skipping
// the mmap header if present).
func (p *page) entries() []byte {
	if p.backing == backingMmap {
		return p.data[pageHdrSize:]
	}
	return p.data
}

// FreeTail returns the contiguous bytes available at the tail of the page.
func (p *page) FreeTail() int {
	return len(p.entries()) - p.tail()
}

// Empty reports whether the page contains zero live entries.
func (p *page) Empty() bool { return p.head() == p.tail() }

// Write appends an entry. Returns the entry's offset and on-disk size.
// Returns errPageFull when there isn't enough contiguous tail room.
// Heap-backed pages skip the per-entry CRC32 — they can't be corrupted
// by anything outside the Go runtime and never participate in the
// rebuild path that would consume the CRC.
// meta is the entry's packed write sequence + flags (see cache/ringbuf.go). It is
// passed through VERBATIM: cold compaction re-writes surviving entries through
// this same method and must not alter their recency or tombstone bit.
func (p *page) Write(key, value []byte, expiryMs, meta uint64) (offset uint32, size uint32, err error) {
	need := entrySize(len(key), len(value))
	if p.FreeTail() < need {
		return 0, 0, errPageFull
	}
	tail := p.tail()
	var n int
	if p.backing == backingMmap {
		n, err = encodeEntry(p.entries()[tail:], key, value, expiryMs, meta)
	} else {
		n, err = encodeEntryNoCRC(p.entries()[tail:], key, value, expiryMs, meta)
	}
	if err != nil {
		return 0, 0, err
	}
	off := uint32(tail) //nolint:gosec // tail < PageSize which is validated ≤ MaxInt32
	p.setTail(tail + n)
	return off, uint32(n), nil //nolint:gosec // n = entrySize which fits in uint32
}

// Read returns key, value, and expiry for the entry at the given offset.
// The returned key and value alias into the page's backing slice and remain
// valid until the entry is evicted. Skips per-entry CRC verification — see
// [decodeEntryFast] for why this is safe on the hot path. The entry's size
// is recovered from its header by [decodeEntryFast], so the caller doesn't
// have to remember it (slabRef no longer carries it).
func (p *page) Read(offset uint32) ([]byte, []byte, uint64, error) {
	entries := p.entries()
	if int(offset) >= len(entries) {
		return nil, nil, 0, errEntryTruncated
	}
	return decodeEntryFast(entries[offset:])
}

// MetaAt returns the meta word (write sequence + flags) of the entry at offset,
// and false if offset cannot hold an entry header. Kept SEPARATE from Read so the
// hot read path never loads it: the callers are the warm-restart rebuild, the
// tombstone filters, and cold compaction, none of which is latency-critical.
func (p *page) MetaAt(offset uint32) (uint64, bool) {
	entries := p.entries()
	if int(offset)+entryHeaderSize > len(entries) {
		return 0, false
	}
	return entryMetaAt(entries[offset:]), true
}

// EvictFront removes the oldest live entry from the page. Returns its key
// (so callers can drop it from an external index) and the entry size in bytes.
// The returned key ALIASES the page's backing slice — it stays valid only until
// the freed region is reused by a later Write, so the caller must consume it
// (e.g. hash it) immediately. The sole caller (evictUntilFitsLocked) does. If
// the page is empty, returns errEntryTruncated.
func (p *page) EvictFront() ([]byte, uint32, error) {
	if p.Empty() {
		return nil, 0, errEntryTruncated
	}
	head := p.head()
	tail := p.tail()
	entries := p.entries()
	if head+entryHeaderSize > len(entries) {
		return nil, 0, errEntryTruncated
	}
	keyLen := int(binary.LittleEndian.Uint16(entries[head : head+2]))
	valLen := int(binary.LittleEndian.Uint32(entries[head+2 : head+6]))
	// On a 32-bit platform int is 32 bits, so a crash-torn/garbage valLen near
	// the uint32 max widens NEGATIVE instead of huge, which would make newHead
	// below UNDERSHOOT tail and slip past the corruption check it's guarding.
	// Reject it here instead. On 64-bit valLen never goes negative (it tops out
	// at maxValueLen, comfortably positive), so this is a no-op there.
	if valLen < 0 {
		return nil, 0, errEntryTruncated
	}

	keyStart := head + entryHeaderSize
	keyEnd := keyStart + keyLen
	// Validate the WHOLE entry (key + value) lies within [head, tail), mirroring
	// decodeEntry's total-length check. A crash-torn or stale header carries a
	// garbage valLen; without this guard EvictFront would advance head past tail,
	// breaking the 0 <= head <= tail invariant and wedging the shard. newHead is
	// computed in full-width int (keyLen/valLen sum to at most maxKeyLen +
	// maxValueLen, which fits an int on 64-bit) so a huge valLen can't wrap a
	// uint32 into a small, deceptively in-bounds size. The caller
	// (drainPageLocked) treats the error as page corruption and Resets the page.
	newHead := keyEnd + valLen
	if keyEnd > tail || newHead > tail {
		return nil, 0, errEntryTruncated
	}
	// newHead <= tail <= len(entries) <= PageSize, so the entry size fits uint32.
	size := uint32(entryHeaderSize + keyLen + valLen) //nolint:gosec // bounded by tail-head <= PageSize
	// Alias the key into the page (no copy): advancing head below doesn't
	// overwrite these bytes, and the caller hashes them before any Write reuses
	// the region. See the doc comment.
	out := entries[keyStart:keyEnd]

	if newHead == tail {
		// Empty: reset both so subsequent writes start at offset 0.
		p.setHead(0)
		p.setTail(0)
	} else {
		p.setHead(newHead)
	}
	return out, size, nil
}

// Reset clears the page in-place so it can be reused.
func (p *page) Reset() {
	p.setHead(0)
	p.setTail(0)
}
