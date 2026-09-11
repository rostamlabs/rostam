// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"math"
	"time"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// ErrShortArgs indicates the args byte slice is shorter than expected.
var ErrShortArgs = errors.New("wire: args too short")

// ErrTTLOutOfRange indicates a wire ttlMs so large it would overflow the
// time.Duration it decodes into.
var ErrTTLOutOfRange = errors.New("wire: ttlMs out of range")

// maxTTLMs is the largest ttl (in ms) that fits in a time.Duration (int64
// nanoseconds) — ~292 years. Above it, time.Duration(ms)*time.Millisecond
// overflows to a NEGATIVE (or wrapped) duration, which the cache reads as the
// "no expiry" sentinel — silently making a bounded key permanent. The new
// TTL-carrying decoders reject anything above it via ttlFromMs.
const maxTTLMs = math.MaxInt64 / int64(time.Millisecond)

// ttlFromMs converts a wire ttlMs (u64) to a time.Duration, rejecting a value so
// large the multiply would overflow. It is the shared guard the incr_ex / caex
// decoders apply BEFORE the multiply. (put/expire share this latent overflow but
// predate the guard; left alone here to keep this change scoped to the new ops.)
func ttlFromMs(ms uint64) (time.Duration, error) {
	if ms > uint64(maxTTLMs) {
		return 0, ErrTTLOutOfRange
	}
	return time.Duration(ms) * time.Millisecond, nil
}

// StdKeyExtractor reads [keyLen u16][key] from the start of args.
// It is the canonical extractor for all five built-in routable ops
// (get/put/del/expire/incr), whose args always start with this layout.
func StdKeyExtractor(args []byte) ([]byte, bool) {
	if len(args) < 2 {
		return nil, false
	}
	n := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+n {
		return nil, false
	}
	return args[2 : 2+n], true
}

// ReadyOp is the shardless readiness-probe op name (see the __ready__
// registration). A nil error means ready; a non-nil error means not ready.
const ReadyOp = "__ready__"

// MetricsOp renders the Prometheus text exposition for all dense collections on
// this node. The result bytes are the exposition body (text/plain), served as-is
// by the HTTP /metrics handler.
const MetricsOp = "__metrics__"

// ReplMetricsOp is the shardless replication-observability op name (see the
// __repl_metrics__ registration). Its result is a JSON body describing the
// per-hosted-shard replication state (mode / primary / ISR / min-ISR / lag).
const ReplMetricsOp = "__repl_metrics__"

// KVMetricsOp is a shardless READ-ONLY op rendering the node's KV cache stats
// in the Prometheus text format. __metrics__ covers the VECTOR side and emits
// nothing on a KV-only node, so without this a cache-only Rostam reports
// nothing about itself: hit rate, eviction pressure and occupancy were all
// invisible from outside the process.
const KVMetricsOp = "__kv_metrics__"

// CollectionsOp is the shardless op that enumerates the local node's dense
// collections. Its result is the name list (EncodeCollectionsResult); the HTTP
// /v1/collections handler renders it as JSON. Like __metrics__ it reads the SAME
// CollectionStore.CollectionNames() source, so the two never disagree about which
// collections exist.
const CollectionsOp = "__collections__"

// EncodeCollectionsResult serializes a collection-name list as
// "{count u32}({nameLen u16}{name})*". Mirrors the MGet wire framing so the
// decoder can bound the declared count before allocating.
func EncodeCollectionsResult(names []string) []byte {
	total := 4
	for _, n := range names {
		total += 2 + len(n)
	}
	buf := make([]byte, 4, total)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(names))) //nolint:gosec // count bounded by the store's collection count
	var nl [2]byte
	for _, n := range names {
		binary.BigEndian.PutUint16(nl[:], uint16(len(n))) //nolint:gosec // bounded by MaxCollectionNameWire
		buf = append(buf, nl[:]...)
		buf = append(buf, n...)
	}
	return buf
}

// DecodeCollectionsResult reads a result produced by EncodeCollectionsResult. The
// declared count is CountFitsIn-bounded (each name costs >= its 2-byte length
// prefix) before any allocation, and every nameLen is checked in the 32-bit-safe
// form. Names are copied out so they outlive the response buffer.
func DecodeCollectionsResult(b []byte) ([]string, error) {
	if len(b) < 4 {
		return nil, ErrShortArgs
	}
	n := int(binary.BigEndian.Uint32(b[0:4]))
	off := 4
	if !CountFitsIn(n, len(b)-off, 2) {
		return nil, ErrShortArgs
	}
	names := make([]string, 0, n)
	for range n {
		if len(b)-off < 2 {
			return nil, ErrShortArgs
		}
		nl := int(binary.BigEndian.Uint16(b[off : off+2]))
		off += 2
		if len(b)-off < nl {
			return nil, ErrShortArgs
		}
		names = append(names, string(b[off:off+nl]))
		off += nl
	}
	return names, nil
}

// EncodeKeyArgs encodes "{keyLen u16}{key}" used by get and del.
func EncodeKeyArgs(key []byte) []byte {
	return AppendKeyArgs(nil, key)
}

// AppendKeyArgs is EncodeKeyArgs appending into dst (reusing its capacity when
// large enough), for a hot-loop caller that pools the buffer. Passing dst=nil
// reproduces EncodeKeyArgs's bytes exactly. The returned slice may alias dst.
func AppendKeyArgs(dst, key []byte) []byte {
	n := 2 + len(key)
	var buf []byte
	if cap(dst) >= n {
		buf = dst[:n]
	} else {
		buf = make([]byte, n)
	}
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	return buf
}

// DecodeKeyArgs reads args produced by EncodeKeyArgs.
func DecodeKeyArgs(args []byte) ([]byte, error) {
	if len(args) < 2 {
		return nil, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen {
		return nil, ErrShortArgs
	}
	return args[2 : 2+klen], nil
}

// EncodePutArgs encodes "{keyLen u16}{key}{valLen u32}{val}{ttlMs u64}".
func EncodePutArgs(key, val []byte, ttl time.Duration) []byte {
	return AppendPutArgs(nil, key, val, ttl)
}

// AppendPutArgs is EncodePutArgs appending into dst (reusing its capacity when
// large enough), for a hot-loop caller that pools the buffer. Passing dst=nil
// reproduces EncodePutArgs's bytes exactly. The returned slice may alias dst.
func AppendPutArgs(dst, key, val []byte, ttl time.Duration) []byte {
	n := 2 + len(key) + 4 + len(val) + 8
	var buf []byte
	if cap(dst) >= n {
		buf = dst[:n]
	} else {
		buf = make([]byte, n)
	}
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(val))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[off+4:], val)
	off += 4 + len(val)
	binary.BigEndian.PutUint64(buf[off:off+8], uint64(ttl/time.Millisecond)) //nolint:gosec // safe: duration to milliseconds always positive
	return buf
}

// DecodePutArgs reads args produced by EncodePutArgs.
func DecodePutArgs(args []byte) (key, val []byte, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen+4 {
		return nil, nil, 0, ErrShortArgs
	}
	key = args[2 : 2+klen]
	off := 2 + klen
	vlen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	// Overflow-safe: never add the untrusted vlen into the comparison. On 32-bit a
	// vlen above MaxInt32 widens negative and slips an additive `off+vlen+8` check,
	// then panics the slice. Check the value bytes and the trailing ttl separately.
	if vlen < 0 || len(args)-off < vlen {
		return nil, nil, 0, ErrShortArgs
	}
	val = args[off : off+vlen]
	off += vlen
	if len(args)-off < 8 {
		return nil, nil, 0, ErrShortArgs
	}
	ttl = time.Duration(binary.BigEndian.Uint64(args[off:off+8])) * time.Millisecond //nolint:gosec // safe: u64 from wire is milliseconds, always positive
	return key, val, ttl, nil
}

// EncodeExpireArgs encodes "{keyLen u16}{key}{ttlMs u64}".
func EncodeExpireArgs(key []byte, ttl time.Duration) []byte {
	buf := make([]byte, 2+len(key)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	binary.BigEndian.PutUint64(buf[2+len(key):], uint64(ttl/time.Millisecond)) //nolint:gosec // safe: duration to milliseconds always positive
	return buf
}

// DecodeExpireArgs reads args produced by EncodeExpireArgs.
func DecodeExpireArgs(args []byte) (key []byte, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen+8 {
		return nil, 0, ErrShortArgs
	}
	key = args[2 : 2+klen]
	ttl = time.Duration(binary.BigEndian.Uint64(args[2+klen:2+klen+8])) * time.Millisecond //nolint:gosec // safe: u64 from wire is milliseconds, always positive
	return key, ttl, nil
}

// EncodeIncrArgs encodes "{keyLen u16}{key}{delta i64}".
func EncodeIncrArgs(key []byte, delta int64) []byte {
	buf := make([]byte, 2+len(key)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	binary.BigEndian.PutUint64(buf[2+len(key):], uint64(delta)) //nolint:gosec // safe: reinterpret i64 as u64 for binary write
	return buf
}

// DecodeIncrArgs reads args produced by EncodeIncrArgs.
func DecodeIncrArgs(args []byte) (key []byte, delta int64, err error) {
	if len(args) < 2 {
		return nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	if len(args) < 2+klen+8 {
		return nil, 0, ErrShortArgs
	}
	key = args[2 : 2+klen]
	delta = int64(binary.BigEndian.Uint64(args[2+klen : 2+klen+8])) //nolint:gosec // safe: reinterpret stored u64 as i64 for binary read
	return key, delta, nil
}

// EncodeIncrResult encodes the int64 result of an incr op.
func EncodeIncrResult(v int64) []byte {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(v)) //nolint:gosec // safe: reinterpret i64 as u64 for binary write
	return buf
}

// DecodeIncrResult parses an incr result back into int64.
func DecodeIncrResult(b []byte) (int64, error) {
	if len(b) != 8 {
		return 0, ErrShortArgs
	}
	return int64(binary.BigEndian.Uint64(b)), nil //nolint:gosec // safe: reinterpret stored u64 as i64 for binary read
}

// EncodeSetNXArgs encodes the args for set_nx. Its wire layout is IDENTICAL to
// put's ("{keyLen u16}{key}{valLen u32}{val}{ttlMs u64}"), so it delegates to
// EncodePutArgs; the server decodes it with DecodePutArgs.
func EncodeSetNXArgs(key, val []byte, ttl time.Duration) []byte {
	return EncodePutArgs(key, val, ttl)
}

// EncodeCASArgs encodes "{keyLen u16}{key}{valLen u32}{val}{hasExpected u8}
// {expLen u32}{expected}{ttlMs u64}". hasExpected=false means "expect the key to
// be absent"; the expected bytes are dropped (zero length) in that case.
func EncodeCASArgs(key, val []byte, hasExpected bool, expected []byte, ttl time.Duration) []byte {
	if !hasExpected {
		expected = nil
	}
	buf := make([]byte, 2+len(key)+4+len(val)+1+4+len(expected)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(val))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[off+4:], val)
	off += 4 + len(val)
	if hasExpected {
		buf[off] = 1
	}
	off++
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(expected))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[off+4:], expected)
	off += 4 + len(expected)
	binary.BigEndian.PutUint64(buf[off:off+8], uint64(ttl/time.Millisecond)) //nolint:gosec // safe: duration to milliseconds always positive
	return buf
}

// DecodeCASArgs reads args produced by EncodeCASArgs. Every variable-length field
// is bounds-checked in the 32-bit-safe form `len(args)-off < need` with an
// explicit `< 0` guard on each u32 length, so a hostile length that widens
// NEGATIVE on GOARCH=386 (int is 32-bit there) is rejected instead of slipping
// past an additive `off+need` check that could itself overflow. See the hostile
// decoder sweep (ops.TestNoDecoderPanicsOnHostileBytes).
func DecodeCASArgs(args []byte) (key, val []byte, hasExpected bool, expected []byte, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, nil, false, nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	off := 2
	if len(args)-off < klen+4 { // key + valLen(4)
		return nil, nil, false, nil, 0, ErrShortArgs
	}
	key = args[off : off+klen]
	off += klen
	vlen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	if vlen < 0 || len(args)-off < vlen {
		return nil, nil, false, nil, 0, ErrShortArgs
	}
	val = args[off : off+vlen]
	off += vlen
	if len(args)-off < 1+4 { // hasExpected(1) + expLen(4)
		return nil, nil, false, nil, 0, ErrShortArgs
	}
	hasExpected = args[off] != 0
	off++
	elen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	// Check expected and ttlMs separately: never add the untrusted elen to the
	// need (elen+8 can overflow int to negative on 32-bit and slip past the
	// guard, then panic the slice below — the exact contract TestNoDecoderPanics
	// forbids). len(args)-off can't overflow (both bounded by real slice length).
	if elen < 0 || len(args)-off < elen { // expected
		return nil, nil, false, nil, 0, ErrShortArgs
	}
	expected = args[off : off+elen]
	off += elen
	if len(args)-off < 8 { // ttlMs(8)
		return nil, nil, false, nil, 0, ErrShortArgs
	}
	ttl = time.Duration(binary.BigEndian.Uint64(args[off:off+8])) * time.Millisecond //nolint:gosec // safe: u64 from wire is milliseconds, always positive
	if !hasExpected {
		expected = nil
	}
	return key, val, hasExpected, expected, ttl, nil
}

// EncodeCADArgs encodes "{keyLen u16}{key}{expLen u32}{expected}" for cad
// (compare-and-delete).
func EncodeCADArgs(key, expected []byte) []byte {
	buf := make([]byte, 2+len(key)+4+len(expected))
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(expected))) //nolint:gosec // bounded by upstream key/value length limits
	copy(buf[off+4:], expected)
	return buf
}

// DecodeCADArgs reads args produced by EncodeCADArgs. Bounds-checked in the same
// 32-bit-safe form as DecodeCASArgs.
func DecodeCADArgs(args []byte) (key, expected []byte, err error) {
	if len(args) < 2 {
		return nil, nil, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	off := 2
	if len(args)-off < klen+4 { // key + expLen(4)
		return nil, nil, ErrShortArgs
	}
	key = args[off : off+klen]
	off += klen
	elen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	if elen < 0 || len(args)-off < elen {
		return nil, nil, ErrShortArgs
	}
	expected = args[off : off+elen]
	return key, expected, nil
}

// DecodeCASResult parses the 1-byte 0/1 result shared by set_nx, cas, and cad:
// 1 = the write happened (stored / deleted), 0 = the condition failed (key
// present / value mismatch / absent). exists / persist / caex reuse it too:
// their 1-byte result has the identical present/absent-1/0 shape.
func DecodeCASResult(b []byte) (bool, error) {
	if len(b) != 1 {
		return false, ErrShortArgs
	}
	return b[0] != 0, nil
}

// --- KV roadmap ops (exists / getdel / getset / persist / ttl / incr_ex /
// caex / mget) -----------------------------------------------------------------
//
// exists / persist reuse EncodeKeyArgs+DecodeKeyArgs (args) and DecodeCASResult
// (1-byte result). getset reuses EncodePutArgs+DecodePutArgs (args, same layout
// as put) and the getdel result codec below. caex reuses DecodeCASResult.

// EncodeGetDelResult encodes the getdel / getset result: [found u8] then, when
// found, [valLen u32][val]. A found=0 result carries no value bytes. getset
// reuses it for its OLD value with the identical found/not-found shape.
func EncodeGetDelResult(val []byte, found bool) []byte {
	if !found {
		return []byte{0}
	}
	buf := make([]byte, 1+4+len(val))
	buf[0] = 1
	binary.BigEndian.PutUint32(buf[1:5], uint32(len(val))) //nolint:gosec // bounded by upstream value length limits
	copy(buf[5:], val)
	return buf
}

// DecodeGetDelResult reads a result produced by EncodeGetDelResult. found=false
// means the key was absent (no value); found=true returns the value bytes (a
// zero-length but non-nil slice when the stored value was empty). Bounds-checked
// in the 32-bit-safe `len-off < n` form with an explicit `< 0` guard, like the
// cas/cad decoders.
func DecodeGetDelResult(b []byte) (val []byte, found bool, err error) {
	if len(b) < 1 {
		return nil, false, ErrShortArgs
	}
	if b[0] == 0 {
		return nil, false, nil
	}
	off := 1
	if len(b)-off < 4 {
		return nil, false, ErrShortArgs
	}
	vlen := int(binary.BigEndian.Uint32(b[off : off+4]))
	off += 4
	if vlen < 0 || len(b)-off < vlen {
		return nil, false, ErrShortArgs
	}
	return b[off : off+vlen], true, nil
}

// EncodeTTLResult encodes the ttl op's i64 result (Redis convention: -2 absent,
// -1 present-but-no-expiry, else remaining ms >= 0).
func EncodeTTLResult(ms int64) []byte {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(ms)) //nolint:gosec // reinterpret i64 as u64 for binary write
	return buf
}

// DecodeTTLResult parses an EncodeTTLResult i64 back.
func DecodeTTLResult(b []byte) (int64, error) {
	if len(b) != 8 {
		return 0, ErrShortArgs
	}
	return int64(binary.BigEndian.Uint64(b)), nil //nolint:gosec // reinterpret stored u64 as i64 for binary read
}

// EncodeIncrExArgs encodes "{keyLen u16}{key}{delta i64}{ttlMs u64}" for incr_ex:
// incr with a TTL applied only when the key is newly created.
func EncodeIncrExArgs(key []byte, delta int64, ttl time.Duration) []byte {
	buf := make([]byte, 2+len(key)+8+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key length limits
	copy(buf[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint64(buf[off:off+8], uint64(delta)) //nolint:gosec // reinterpret i64 as u64 for binary write
	off += 8
	binary.BigEndian.PutUint64(buf[off:off+8], uint64(ttl/time.Millisecond)) //nolint:gosec // duration to milliseconds always positive
	return buf
}

// DecodeIncrExArgs reads args produced by EncodeIncrExArgs.
func DecodeIncrExArgs(args []byte) (key []byte, delta int64, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, 0, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	off := 2
	if len(args)-off < klen+8+8 { // key + delta(8) + ttlMs(8)
		return nil, 0, 0, ErrShortArgs
	}
	key = args[off : off+klen]
	off += klen
	delta = int64(binary.BigEndian.Uint64(args[off : off+8])) //nolint:gosec // reinterpret stored u64 as i64 for binary read
	off += 8
	ttl, err = ttlFromMs(binary.BigEndian.Uint64(args[off : off+8]))
	if err != nil {
		return nil, 0, 0, err
	}
	return key, delta, ttl, nil
}

// EncodeCAEXArgs encodes "{keyLen u16}{key}{expLen u32}{expected}{ttlMs u64}" for
// caex (compare-and-expire) — the cas layout without a new value.
func EncodeCAEXArgs(key, expected []byte, ttl time.Duration) []byte {
	buf := make([]byte, 2+len(key)+4+len(expected)+8)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(key))) //nolint:gosec // bounded by upstream key length limits
	copy(buf[2:], key)
	off := 2 + len(key)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(expected))) //nolint:gosec // bounded by upstream value length limits
	copy(buf[off+4:], expected)
	off += 4 + len(expected)
	binary.BigEndian.PutUint64(buf[off:off+8], uint64(ttl/time.Millisecond)) //nolint:gosec // duration to milliseconds always positive
	return buf
}

// DecodeCAEXArgs reads args produced by EncodeCAEXArgs. Bounds-checked in the
// 32-bit-safe `len-off < n` form with an explicit `< 0` guard on the u32 length,
// never the additive `off+n` that can overflow on GOARCH=386 (see DecodeCASArgs).
func DecodeCAEXArgs(args []byte) (key, expected []byte, ttl time.Duration, err error) {
	if len(args) < 2 {
		return nil, nil, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(args[0:2]))
	off := 2
	if len(args)-off < klen+4 { // key + expLen(4)
		return nil, nil, 0, ErrShortArgs
	}
	key = args[off : off+klen]
	off += klen
	elen := int(binary.BigEndian.Uint32(args[off : off+4]))
	off += 4
	if elen < 0 || len(args)-off < elen { // expected
		return nil, nil, 0, ErrShortArgs
	}
	expected = args[off : off+elen]
	off += elen
	if len(args)-off < 8 { // ttlMs(8)
		return nil, nil, 0, ErrShortArgs
	}
	ttl, err = ttlFromMs(binary.BigEndian.Uint64(args[off : off+8]))
	if err != nil {
		return nil, nil, 0, err
	}
	return key, expected, ttl, nil
}

// EncodeMGetArgs encodes "{count u16}" then, per key, "{keyLen u16}{key}" — a
// same-shard batch read. The caller guarantees every key hashes to one shard;
// mgetKeyExtractor routes the batch by its FIRST key, like put_batch.
func EncodeMGetArgs(keys [][]byte) []byte {
	total := 2
	for _, k := range keys {
		total += 2 + len(k)
	}
	buf := make([]byte, 2, total)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(keys))) //nolint:gosec // caller caps batch to <= 65535 keys
	var kl [2]byte
	for _, k := range keys {
		binary.BigEndian.PutUint16(kl[:], uint16(len(k))) //nolint:gosec // bounded by upstream key length limits
		buf = append(buf, kl[:]...)
		buf = append(buf, k...)
	}
	return buf
}

// DecodeMGetArgs reads args produced by EncodeMGetArgs. The declared count is
// bounded by CountFitsIn (each key costs >= its 2-byte length prefix) before any
// allocation, and every keyLen is checked in the 32-bit-safe form.
func DecodeMGetArgs(args []byte) ([][]byte, error) {
	if len(args) < 2 {
		return nil, ErrShortArgs
	}
	n := int(binary.BigEndian.Uint16(args[0:2]))
	off := 2
	if !CountFitsIn(n, len(args)-off, 2) {
		return nil, ErrShortArgs
	}
	keys := make([][]byte, 0, n)
	for range n {
		if len(args)-off < 2 {
			return nil, ErrShortArgs
		}
		kl := int(binary.BigEndian.Uint16(args[off : off+2]))
		off += 2
		if len(args)-off < kl {
			return nil, ErrShortArgs
		}
		keys = append(keys, args[off:off+kl])
		off += kl
	}
	return keys, nil
}

// EncodeMGetResult encodes "{count u16}" then, per key, "[found u8]" and, when
// found, "[valLen u32][val]". vals and found are parallel and same-length; a
// found[i]=false entry emits only the flag byte.
func EncodeMGetResult(vals [][]byte, found []bool) []byte {
	buf := make([]byte, 2)
	binary.BigEndian.PutUint16(buf[0:2], uint16(len(found))) //nolint:gosec // count bounded by the request's key count (<= 65535)
	var vl [4]byte
	for i := range found {
		if !found[i] {
			buf = append(buf, 0)
			continue
		}
		buf = append(buf, 1)
		binary.BigEndian.PutUint32(vl[:], uint32(len(vals[i]))) //nolint:gosec // bounded by upstream value length limits
		buf = append(buf, vl[:]...)
		buf = append(buf, vals[i]...)
	}
	return buf
}

// DecodeMGetResult reads a result produced by EncodeMGetResult into a slice
// aligned to the requested keys: a missing key is a nil entry, a found key a
// non-nil (possibly empty) copy of its value. The count is CountFitsIn-bounded
// (each entry costs >= its 1-byte flag) and every valLen is 32-bit-safe checked.
// Values are copied out so the result outlives the response buffer.
func DecodeMGetResult(b []byte) ([][]byte, error) {
	if len(b) < 2 {
		return nil, ErrShortArgs
	}
	n := int(binary.BigEndian.Uint16(b[0:2]))
	off := 2
	if !CountFitsIn(n, len(b)-off, 1) {
		return nil, ErrShortArgs
	}
	out := make([][]byte, 0, n)
	for range n {
		if len(b)-off < 1 {
			return nil, ErrShortArgs
		}
		f := b[off]
		off++
		if f == 0 {
			out = append(out, nil)
			continue
		}
		if len(b)-off < 4 {
			return nil, ErrShortArgs
		}
		vlen := int(binary.BigEndian.Uint32(b[off : off+4]))
		off += 4
		if vlen < 0 || len(b)-off < vlen {
			return nil, ErrShortArgs
		}
		v := make([]byte, vlen)
		copy(v, b[off:off+vlen])
		out = append(out, v)
		off += vlen
	}
	return out, nil
}

// mgetKeyExtractor returns the FIRST key as the mget batch's routing key. The
// args lead with a u16 count, so the first "{keyLen u16}{key}" entry starts at
// offset 2 — StdKeyExtractor reads it from there. Mirrors putBatchKeyExtractor.
func mgetKeyExtractor(args []byte) ([]byte, bool) {
	if len(args) < 2 {
		return nil, false
	}
	return StdKeyExtractor(args[2:])
}

// ScrollOrderToOrderBy maps the wire args ScrollOrder onto vector.OrderBy, including the
// MULTI-KEY Tail (the secondary key specs) and the v4 resume TUPLE (ResumeKeys). A nil
// or single-key ScrollOrder maps to the byte/behaviour-identical single-key vector.OrderBy
// (empty Tail / no ResumeKeys); a multi-key ScrollOrder fills OrderBy.Tail + ResumeKeys so
// the engine's tuple comparator + v4 seek run. Shared by the leaf engine and the
// coordinator fan-out (rostam.scrollOrderByFromOps) so they agree on the order.
func ScrollOrderToOrderBy(o *ScrollOrder) *vtypes.OrderBy {
	if o == nil {
		return nil
	}
	ob := &vtypes.OrderBy{
		Key:          o.Key,
		Desc:         o.Desc,
		IsDatetime:   o.IsDatetime,
		Kind:         o.Kind,
		StartFrom:    o.StartFrom,
		HasStart:     o.HasStart,
		ResumeStr:    o.ResumeStr,
		HasResumeStr: o.HasResumeStr,
	}
	if len(o.Tail) > 0 {
		ob.Tail = make([]vtypes.OrderBy, len(o.Tail))
		for i, tk := range o.Tail {
			ob.Tail[i] = vtypes.OrderBy{Key: tk.Key, Desc: tk.Desc, IsDatetime: tk.IsDatetime, Kind: tk.Kind}
		}
		if o.HasResumeKeys {
			ob.ResumeKeys = make([]vtypes.OrderVal, len(o.ResumeKeys))
			for i, rv := range o.ResumeKeys {
				ob.ResumeKeys[i] = vtypes.OrderVal{Num: rv.Num, Str: rv.Str, Kind: rv.Kind}
			}
			ob.HasResumeKeys = true
		}
	}
	return ob
}

// OrderByToScrollOrderTail maps a vector.OrderBy's MULTI-KEY Tail onto the wire args
// ScrollOrder.Tail (the per-key specs) — the inverse direction of ScrollOrderToOrderBy
// for the Tail only. The primary fields (Key/Desc/Kind/Start/Resume) are set by the
// caller (each transport builds the primary + resume per its cursor path); this fills the
// Tail so every transport's ScrollOrder construction shares ONE multi-key mapping. A
// single-key OrderBy (empty Tail) yields an empty Tail (byte-identical single-key path).
func OrderByToScrollOrderTail(ob *vtypes.OrderBy) []ScrollOrderKey {
	if ob == nil || len(ob.Tail) == 0 {
		return nil
	}
	tail := make([]ScrollOrderKey, len(ob.Tail))
	for i, tk := range ob.Tail {
		tail[i] = ScrollOrderKey{Key: tk.Key, Desc: tk.Desc, IsDatetime: tk.IsDatetime, Kind: tk.Kind}
	}
	return tail
}
