// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// ErrShortArgs indicates the args byte slice is shorter than expected.
var ErrShortArgs = errors.New("wire: args too short")

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
	if len(args) < off+vlen+8 {
		return nil, nil, 0, ErrShortArgs
	}
	val = args[off : off+vlen]
	off += vlen
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

// ScrollOrderToOrderBy maps the wire args ScrollOrder onto vector.OrderBy, including the
// MULTI-KEY Tail (the secondary key specs) and the v4 resume TUPLE (ResumeKeys). A nil
// or single-key ScrollOrder maps to the byte/behaviour-identical single-key vector.OrderBy
// (empty Tail / no ResumeKeys); a multi-key ScrollOrder fills OrderBy.Tail + ResumeKeys so
// the engine's tuple comparator + v4 seek run. Shared by the leaf engine and the
// coordinator fan-out (rostam.scrollOrderByFromOps) so they agree on the order.
func ScrollOrderToOrderBy(o *ScrollOrder) *vector.OrderBy {
	if o == nil {
		return nil
	}
	ob := &vector.OrderBy{
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
		ob.Tail = make([]vector.OrderBy, len(o.Tail))
		for i, tk := range o.Tail {
			ob.Tail[i] = vector.OrderBy{Key: tk.Key, Desc: tk.Desc, IsDatetime: tk.IsDatetime, Kind: tk.Kind}
		}
		if o.HasResumeKeys {
			ob.ResumeKeys = make([]vector.OrderVal, len(o.ResumeKeys))
			for i, rv := range o.ResumeKeys {
				ob.ResumeKeys[i] = vector.OrderVal{Num: rv.Num, Str: rv.Str, Kind: rv.Kind}
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
func OrderByToScrollOrderTail(ob *vector.OrderBy) []ScrollOrderKey {
	if ob == nil || len(ob.Tail) == 0 {
		return nil
	}
	tail := make([]ScrollOrderKey, len(ob.Tail))
	for i, tk := range ob.Tail {
		tail[i] = ScrollOrderKey{Key: tk.Key, Desc: tk.Desc, IsDatetime: tk.IsDatetime, Kind: tk.Kind}
	}
	return tail
}
