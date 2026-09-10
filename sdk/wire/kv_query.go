// SPDX-License-Identifier: Apache-2.0

package wire

// Every multi-byte length/count field in this file's frames (kv_query args,
// the cursor sub-frame, and the kv_query result) is big-endian, like every
// other frame in this package. The phase-3 plan's task-1 brief says
// "little-endian counts" — that line was wrong (this package is ~780
// BigEndian call sites against 39 LittleEndian ones, the latter confined to
// the operate record cell layout); this codec follows the package
// convention, not the brief's typo.

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// kv_query return projections: what a matching row carries back. Records is
// Values plus a server-side wire.DecodeRecord validation pass (a non-record
// value under Records is skipped, not an error) — see the phase-3 plan's
// decisions §6.
const (
	KVQueryReturnKeys    uint8 = 0
	KVQueryReturnValues  uint8 = 1
	KVQueryReturnRecords uint8 = 2
)

// kv_query budgets. These are wire-frame caps enforced by the codec itself
// (never a silent truncation of an oversized frame — a typed error); the
// SEPARATE candidate/scan budgets that bound how much work a shard does to
// answer a query are node config (ops.KVQueryBudget, a later task), because
// kv_query is read-only and never applied, so per-node values there cannot
// diverge committed state.
const (
	KVQueryMaxLimit       = 1000
	KVQueryMaxFilterBytes = 64 << 10
	KVQueryMaxCursorBytes = 4 << 20
	KVQueryMaxCursorConts = 4096
	KVQueryMaxFilterNodes = 256
	KVQueryMaxFilterDepth = 32
	KVQueryMaxPageBytes   = 8 << 20 // one group's page, and the merged page
)

// ErrKVQueryArgsTruncated marks a kv_query args/cursor/result frame as
// shorter than its declared lengths require.
var ErrKVQueryArgsTruncated = errors.New("wire: kv_query args truncated")

// ErrKVQueryArgs marks a kv_query args frame as structurally decodable but
// semantically invalid: an out-of-range limit/return/consistency byte, a
// scan-less query with no index named, an index name outside its charset, a
// filter or cursor blob over its byte cap, or a cursor whose continuations
// are not strictly increasing by group / exceed KVQueryMaxCursorConts.
var ErrKVQueryArgs = errors.New("wire: kv_query args invalid")

// ErrKVQueryResult marks a kv_query result frame as structurally decodable
// but semantically invalid (an oversized page, a malformed cursor), or marks
// an EncodeKVQueryResult call as unable to represent its input (a key over
// 64KiB, or a cursor violating the same invariants as the args cursor).
var ErrKVQueryResult = errors.New("wire: kv_query result invalid")

// ErrKVFilterBudget marks a decoded vtypes.Filter tree as exceeding
// CheckFilterBudget's node-count or nesting-depth cap.
var ErrKVFilterBudget = errors.New("wire: filter exceeds query budget")

// kv_query args flags (args[0]).
const (
	kvQueryFlagFilter uint8 = 1 << 0 // filter JSON block present
	kvQueryFlagCursor uint8 = 1 << 1 // cursor block present
	kvQueryFlagScan   uint8 = 1 << 2 // unindexed full-keyspace scan requested
)

// KVQueryCont is one shard group's paging position within a kv_query: the
// exclusive key to resume after, and whether that group has more matching
// keys beyond it. A kv_query cursor is a slice of these, one per group that
// had more rows than fit the page, kept in STRICTLY INCREASING Group order
// (both as a client-supplied resume cursor and as a server-returned
// continuation — the same type serves both directions unchanged).
type KVQueryCont struct {
	Group uint32
	After []byte // exclusive: the next page of this group starts at keys > After
	More  bool
}

// KVQueryRow is one matching key (plus its value, when the query asked for
// one) in a kv_query result.
type KVQueryRow struct {
	Key []byte
	// Value is nil when the query's Return == KVQueryReturnKeys, and also when
	// Return asked for a value that was too large to fit the page — see
	// EncodeKVQueryResult for the full reading of an absent value.
	Value []byte
}

// KVQueryArgs is the decoded form of a kv_query call.
type KVQueryArgs struct {
	// Index names the KVIndexDef to query by. Required unless Scan is true
	// (an unindexed query answers ops.ErrKVQueryScanRequired instead).
	Index       string
	Filter      vtypes.Filter
	Limit       uint16
	Return      uint8
	Consistency uint8 // wire.Consistency{AnyReplica,LeaderOnly,Linearizable}
	Scan        bool
	Cursor      []KVQueryCont
}

// KVQueryResult is the decoded form of a kv_query call's answer: the page of
// matching rows plus a continuation cursor (empty when every group answered
// in full).
type KVQueryResult struct {
	Rows   []KVQueryRow
	Cursor []KVQueryCont
}

// validConsistency reports whether rc is one of the three read-consistency
// levels a kv_query frame may carry. ConsistencyBoundedStaleness (3) is
// deliberately excluded: it needs the 8-byte raft-entry bound trailer this
// frame does not carry (see the phase-3 plan's decisions §8).
func validKVQueryConsistency(rc uint8) bool {
	switch rc {
	case ConsistencyAnyReplica, ConsistencyLeaderOnly, ConsistencyLinearizable:
		return true
	default:
		return false
	}
}

// checkKVQueryCursorShape validates the invariants a cursor (args-supplied or
// server-returned) must satisfy regardless of direction: at most
// KVQueryMaxCursorConts entries, and Group strictly increasing.
func checkKVQueryCursorShape(conts []KVQueryCont) error {
	if len(conts) > KVQueryMaxCursorConts {
		return fmt.Errorf("%w: %d cursor continuations exceeds cap %d", ErrKVQueryArgs, len(conts), KVQueryMaxCursorConts)
	}
	for i := 1; i < len(conts); i++ {
		if conts[i].Group <= conts[i-1].Group {
			return fmt.Errorf("%w: cursor groups not strictly increasing", ErrKVQueryArgs)
		}
	}
	return nil
}

// kv_query cursor flags, carried in each continuation's flag byte.
const (
	kvQueryContFlagMore   uint8 = 1 << 0 // this group has more matching keys beyond After
	kvQueryContFlagPrefix uint8 = 1 << 1 // After is the block's shared prefix followed by this entry's suffix
)

// KVQueryCursorMaxPrefixLen caps the shared prefix a cursor block may factor
// out. It is KVIndexMaxPrefixLen because that IS what the prefix is in
// practice: every key an indexed query can return starts with the definition's
// KeyPrefix, whose own cap is this. Capping it also bounds decode expansion —
// the prefix is stored once and restored n times, so an uncapped one would let
// a small block decode into KVQueryMaxCursorConts copies of a 64 KiB key.
const KVQueryCursorMaxPrefixLen = KVIndexMaxPrefixLen

// kvQueryCursorPrefix picks the block's shared prefix: the longest common
// prefix of the non-empty After keys, capped at KVQueryCursorMaxPrefixLen.
//
// An indexed kv_query is scoped to its definition's KeyPrefix, so EVERY key it
// can return starts with that prefix — which makes the longest common prefix at
// least the KeyPrefix, without this package needing a catalog to look one up.
// Deriving it from the keys rather than from the definition also keeps the block
// SELF-DESCRIBING: a cursor decodes to the same keys it encoded from even if the
// definition is redefined between two pages, which a name-based lookup could not
// promise.
func kvQueryCursorPrefix(conts []KVQueryCont) []byte {
	var lcp []byte
	seen := false
	for _, c := range conts {
		if len(c.After) == 0 {
			continue // a group that has emitted nothing shares no prefix
		}
		if !seen {
			lcp = c.After
			if len(lcp) > KVQueryCursorMaxPrefixLen {
				lcp = lcp[:KVQueryCursorMaxPrefixLen]
			}
			seen = true
			continue
		}
		i := 0
		for i < len(lcp) && i < len(c.After) && lcp[i] == c.After[i] {
			i++
		}
		lcp = lcp[:i]
		if len(lcp) == 0 {
			return nil
		}
	}
	return lcp
}

// appendKVQueryCursor appends a cursor block's BODY (no length prefix — the
// caller frames it, since args and result wrap it differently):
//
//	[n u16][prefixLen u8][prefix]{ [group u32][flags u8][suffixLen u16][suffix] }
//
// THE SHARED PREFIX IS STORED ONCE. A composite cursor carries one continuation
// per shard group that still has rows, and on an indexed query every one of
// those keys begins with the definition's KeyPrefix — so without factoring it
// out the cursor pays for the same bytes NumShards times, and the cap below
// becomes a per-key length limit that shrinks as an operator adds shards. An
// entry whose After does not start with the block prefix (an empty After, or a
// scan cursor over an unrelated part of the keyspace) simply carries the whole
// key and says so with kvQueryContFlagPrefix clear.
func appendKVQueryCursor(dst []byte, conts []KVQueryCont) []byte {
	prefix := kvQueryCursorPrefix(conts)
	dst = binary.BigEndian.AppendUint16(dst, uint16(len(conts))) //nolint:gosec // bounded by KVQueryMaxCursorConts by the caller
	dst = append(dst, byte(len(prefix)))                         //nolint:gosec // bounded by KVQueryCursorMaxPrefixLen above
	dst = append(dst, prefix...)
	for _, c := range conts {
		dst = binary.BigEndian.AppendUint32(dst, c.Group)
		var flags uint8
		if c.More {
			flags |= kvQueryContFlagMore
		}
		suffix := c.After
		if len(prefix) > 0 && bytes.HasPrefix(c.After, prefix) {
			flags |= kvQueryContFlagPrefix
			suffix = c.After[len(prefix):]
		}
		dst = append(dst, flags)
		dst = binary.BigEndian.AppendUint16(dst, uint16(len(suffix))) //nolint:gosec // bounded by the 64 KiB key cap
		dst = append(dst, suffix...)
	}
	return dst
}

// KVQueryCursorBytes reports exactly how many bytes appendKVQueryCursor will
// write for conts — the number that has to fit KVQueryMaxCursorBytes when the
// client sends this cursor back. Exported so the coordinator can refuse a cursor
// it cannot represent at the moment it builds one, rather than letting the query
// die on the next request with nothing to say about why.
func KVQueryCursorBytes(conts []KVQueryCont) int {
	prefix := kvQueryCursorPrefix(conts)
	n := 2 + 1 + len(prefix)
	for _, c := range conts {
		n += 4 + 1 + 2 + len(c.After)
		if len(prefix) > 0 && bytes.HasPrefix(c.After, prefix) {
			n -= len(prefix)
		}
	}
	return n
}

// kvQueryContMinBytes is the fewest bytes one KVQueryCont can possibly
// occupy on the wire: group(4) + more(1) + afterLen(2), After empty.
const kvQueryContMinBytes = 7

// decodeKVQueryCursor reads a cursor block's body (as appendKVQueryCursor
// writes it) from the front of b, returning the bytes consumed. It enforces
// checkKVQueryCursorShape as it decodes (strictly increasing Group, at most
// KVQueryMaxCursorConts entries) so a hostile or corrupt cursor is rejected
// uniformly whether it arrives as a client-supplied args cursor or a
// (supposedly server-produced) result continuation.
func decodeKVQueryCursor(b []byte) (conts []KVQueryCont, n int, err error) {
	if len(b) < 3 {
		return nil, 0, ErrKVQueryArgsTruncated
	}
	count := binary.BigEndian.Uint16(b)
	off := 2
	// The shared prefix comes before the entries and is bounded by its own u8
	// length, so the expansion it can drive is at most
	// KVQueryMaxCursorConts x 255 — checked against the remaining input before
	// anything is sized from it.
	plen := int(b[off])
	off++
	if plen > len(b)-off {
		return nil, 0, ErrKVQueryArgsTruncated
	}
	prefix := b[off : off+plen]
	off += plen

	if !CountFitsIn(int(count), len(b)-off, kvQueryContMinBytes) {
		return nil, 0, ErrKVQueryArgsTruncated
	}
	if int(count) > KVQueryMaxCursorConts {
		return nil, 0, fmt.Errorf("%w: %d cursor continuations exceeds cap %d", ErrKVQueryArgs, count, KVQueryMaxCursorConts)
	}
	conts = make([]KVQueryCont, 0, count)
	var prevGroup uint32
	for i := 0; i < int(count); i++ {
		if len(b)-off < 4+1+2 {
			return nil, 0, ErrKVQueryArgsTruncated
		}
		group := binary.BigEndian.Uint32(b[off:])
		off += 4
		flags := b[off]
		off++
		slen := binary.BigEndian.Uint16(b[off:])
		off += 2
		if int(slen) > len(b)-off {
			return nil, 0, ErrKVQueryArgsTruncated
		}
		suffix := b[off : off+int(slen)]
		off += int(slen)

		// An entry that claims the shared prefix is rebuilt into its OWN
		// allocation: the two halves are not adjacent in the input, and callers
		// hold After well past this frame. An entry without the flag keeps the
		// sub-slice behaviour the frame has always had.
		var after []byte
		switch {
		case flags&kvQueryContFlagPrefix != 0:
			if plen == 0 {
				return nil, 0, fmt.Errorf("%w: cursor entry claims a shared prefix but the block declares none", ErrKVQueryArgs)
			}
			after = make([]byte, 0, plen+int(slen))
			after = append(after, prefix...)
			after = append(after, suffix...)
		case slen > 0:
			after = suffix
		}
		if i > 0 && group <= prevGroup {
			return nil, 0, fmt.Errorf("%w: cursor groups not strictly increasing", ErrKVQueryArgs)
		}
		prevGroup = group
		conts = append(conts, KVQueryCont{Group: group, More: flags&kvQueryContFlagMore != 0, After: after})
	}
	return conts, off, nil
}

// EncodeKVQueryArgs serializes a kv_query call:
//
//	[flags u8][indexLen u8][index][limit u16][ret u8][rc u8]
//	[?filterLen u32][?filterJSON]
//	[?cursorLen u32][?cursor]
//
// It validates a IDENTICALLY to DecodeKVQueryArgs (see that function's
// checks) and returns ErrKVQueryArgs / ErrKVFilterBudget for anything it
// cannot represent, so a caller never round-trips an args value the
// decoder would then reject.
func EncodeKVQueryArgs(a KVQueryArgs) ([]byte, error) {
	if a.Limit == 0 || a.Limit > KVQueryMaxLimit {
		return nil, fmt.Errorf("%w: limit %d out of range (1..%d)", ErrKVQueryArgs, a.Limit, KVQueryMaxLimit)
	}
	if a.Return > KVQueryReturnRecords {
		return nil, fmt.Errorf("%w: return %d out of range", ErrKVQueryArgs, a.Return)
	}
	if !validKVQueryConsistency(a.Consistency) {
		return nil, fmt.Errorf("%w: consistency %d not supported", ErrKVQueryArgs, a.Consistency)
	}
	if a.Index == "" && !a.Scan {
		return nil, fmt.Errorf("%w: filter needs an index or scan:true", ErrKVQueryArgs)
	}
	if a.Index != "" && !validKVName(a.Index) {
		return nil, fmt.Errorf("%w: index name %q must match [A-Za-z0-9_.:-]{1,%d}", ErrKVQueryArgs, a.Index, KVIndexMaxNameLen)
	}
	if err := checkKVQueryCursorShape(a.Cursor); err != nil {
		return nil, err
	}

	// NOT a.Filter.IsZero(): that helper only checks Op/Field/And/Or/Not/Geo
	// and treats Value.IsZero() (Kind==ValueNone) as "no value" REGARDLESS of
	// the other Value fields — a JSON filter blob whose "value" object sets
	// e.g. "int" without a matching "kind" decodes to exactly such a
	// Filter (Kind stays its zero value, Int is set directly, unmarshal does
	// not gate one field on the other). IsZero() would then call that
	// non-empty Filter "absent" and silently drop it on re-encode — found by
	// FuzzDecodeKVQueryArgs. A full structural comparison against the
	// literal zero value has no such blind spot.
	hasFilter := !reflect.DeepEqual(a.Filter, vtypes.Filter{})
	var filterJSON []byte
	if hasFilter {
		if err := CheckFilterBudget(a.Filter, KVQueryMaxFilterNodes, KVQueryMaxFilterDepth); err != nil {
			return nil, err
		}
		var err error
		filterJSON, err = json.Marshal(a.Filter)
		if err != nil {
			return nil, fmt.Errorf("%w: marshal filter: %v", ErrKVQueryArgs, err)
		}
		if len(filterJSON) > KVQueryMaxFilterBytes {
			return nil, fmt.Errorf("%w: filter %d bytes exceeds cap %d", ErrKVQueryArgs, len(filterJSON), KVQueryMaxFilterBytes)
		}
	}

	hasCursor := len(a.Cursor) > 0
	var cursorBlob []byte
	if hasCursor {
		cursorBlob = appendKVQueryCursor(nil, a.Cursor)
		if len(cursorBlob) > KVQueryMaxCursorBytes {
			return nil, fmt.Errorf("%w: cursor %d bytes exceeds cap %d", ErrKVQueryArgs, len(cursorBlob), KVQueryMaxCursorBytes)
		}
	}

	n := 1 + 1 + len(a.Index) + 2 + 1 + 1
	if hasFilter {
		n += 4 + len(filterJSON)
	}
	if hasCursor {
		n += 4 + len(cursorBlob)
	}
	buf := make([]byte, 0, n)
	var flags uint8
	if hasFilter {
		flags |= kvQueryFlagFilter
	}
	if hasCursor {
		flags |= kvQueryFlagCursor
	}
	if a.Scan {
		flags |= kvQueryFlagScan
	}
	buf = append(buf, flags)
	buf = append(buf, byte(len(a.Index))) //nolint:gosec // bounded by KVIndexMaxNameLen above
	buf = append(buf, a.Index...)
	buf = binary.BigEndian.AppendUint16(buf, a.Limit)
	buf = append(buf, a.Return, a.Consistency)
	if hasFilter {
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(filterJSON))) //nolint:gosec // bounded by KVQueryMaxFilterBytes above
		buf = append(buf, filterJSON...)
	}
	if hasCursor {
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(cursorBlob))) //nolint:gosec // bounded by KVQueryMaxCursorBytes above
		buf = append(buf, cursorBlob...)
	}
	return buf, nil
}

// DecodeKVQueryArgs reads a kv_query args frame produced by EncodeKVQueryArgs
// (see its layout comment), rejecting anything a hostile or corrupt peer
// might send: every declared length is a raw uint32/uint16 checked against
// its own cap and against the remaining bytes BEFORE any int conversion, so
// a lying length (e.g. filterLen == 0xFFFFFFFF) is rejected without ever
// sizing an allocation or a slice from it — the cap comparison against a
// small constant runs first, in unsigned space, so no int() widening (the
// GOARCH=386 hazard this package guards throughout) has a chance to matter.
func DecodeKVQueryArgs(args []byte) (KVQueryArgs, error) {
	if len(args) < 1 {
		return KVQueryArgs{}, ErrKVQueryArgsTruncated
	}
	flags := args[0]
	off := 1

	if len(args)-off < 1 {
		return KVQueryArgs{}, ErrKVQueryArgsTruncated
	}
	idxLen := int(args[off])
	off++
	if len(args)-off < idxLen {
		return KVQueryArgs{}, ErrKVQueryArgsTruncated
	}
	index := string(args[off : off+idxLen])
	off += idxLen

	if len(args)-off < 2+1+1 {
		return KVQueryArgs{}, ErrKVQueryArgsTruncated
	}
	limit := binary.BigEndian.Uint16(args[off:])
	off += 2
	ret := args[off]
	off++
	rc := args[off]
	off++

	if limit == 0 || limit > KVQueryMaxLimit {
		return KVQueryArgs{}, fmt.Errorf("%w: limit %d out of range (1..%d)", ErrKVQueryArgs, limit, KVQueryMaxLimit)
	}
	if ret > KVQueryReturnRecords {
		return KVQueryArgs{}, fmt.Errorf("%w: return %d out of range", ErrKVQueryArgs, ret)
	}
	if !validKVQueryConsistency(rc) {
		return KVQueryArgs{}, fmt.Errorf("%w: consistency %d not supported", ErrKVQueryArgs, rc)
	}
	scan := flags&kvQueryFlagScan != 0
	if index == "" && !scan {
		return KVQueryArgs{}, fmt.Errorf("%w: filter needs an index or scan:true", ErrKVQueryArgs)
	}
	if index != "" && !validKVName(index) {
		return KVQueryArgs{}, fmt.Errorf("%w: index name %q must match [A-Za-z0-9_.:-]{1,%d}", ErrKVQueryArgs, index, KVIndexMaxNameLen)
	}

	var filter vtypes.Filter
	if flags&kvQueryFlagFilter != 0 {
		if len(args)-off < 4 {
			return KVQueryArgs{}, ErrKVQueryArgsTruncated
		}
		flen := binary.BigEndian.Uint32(args[off:])
		off += 4
		// flen is compared against the small constant cap in uint32 space
		// FIRST — a hostile 0xFFFFFFFF is rejected right here, before any
		// int conversion or slicing is attempted.
		if flen > KVQueryMaxFilterBytes {
			return KVQueryArgs{}, fmt.Errorf("%w: filter %d bytes exceeds cap %d", ErrKVQueryArgs, flen, KVQueryMaxFilterBytes)
		}
		if int(flen) > len(args)-off {
			return KVQueryArgs{}, ErrKVQueryArgsTruncated
		}
		fb := args[off : off+int(flen)]
		off += int(flen)
		if err := json.Unmarshal(fb, &filter); err != nil {
			return KVQueryArgs{}, fmt.Errorf("%w: decode filter: %v", ErrKVQueryArgs, err)
		}
		if err := CheckFilterBudget(filter, KVQueryMaxFilterNodes, KVQueryMaxFilterDepth); err != nil {
			return KVQueryArgs{}, err
		}
	}

	var cursor []KVQueryCont
	if flags&kvQueryFlagCursor != 0 {
		if len(args)-off < 4 {
			return KVQueryArgs{}, ErrKVQueryArgsTruncated
		}
		clen := binary.BigEndian.Uint32(args[off:])
		off += 4
		if clen > KVQueryMaxCursorBytes {
			return KVQueryArgs{}, fmt.Errorf("%w: cursor %d bytes exceeds cap %d", ErrKVQueryArgs, clen, KVQueryMaxCursorBytes)
		}
		if int(clen) > len(args)-off {
			return KVQueryArgs{}, ErrKVQueryArgsTruncated
		}
		cb := args[off : off+int(clen)]
		off += int(clen)
		var cn int
		var err error
		cursor, cn, err = decodeKVQueryCursor(cb)
		if err != nil {
			return KVQueryArgs{}, err
		}
		if cn != len(cb) {
			return KVQueryArgs{}, fmt.Errorf("%w: trailing bytes in cursor block", ErrKVQueryArgs)
		}
	}

	if off != len(args) {
		return KVQueryArgs{}, fmt.Errorf("%w: trailing bytes", ErrKVQueryArgs)
	}

	return KVQueryArgs{
		Index:       index,
		Filter:      filter,
		Limit:       limit,
		Return:      ret,
		Consistency: rc,
		Scan:        scan,
		Cursor:      cursor,
	}, nil
}

// EncodeKVQueryResult serializes a kv_query answer:
//
//	[nRows u32]{ [keyLen u16][key][hasVal u8](+[valLen u32][val] if hasVal) }
//	[nCont u16]{ [group u32][more u8][afterLen u16][after] }
//
// The per-row hasVal byte (rather than a single whole-result flag derived
// from the query's Return, which this type does not carry) makes each row
// self-delimiting on its own — decoding never needs to look outside the
// frame to know whether a value follows the key, and Row.Value's nil-ness
// round-trips exactly (nil in, nil out; a legitimate zero-length value in,
// a non-nil empty slice out).
//
// WHAT hasVal = 0 MEANS TO A CLIENT depends on the query's Return, and the
// two readings do not overlap:
//
//   - Return == keys: no row carries a value. Nothing is being said.
//   - Return == values/records: the value was OMITTED because encoding it
//     would push the page past KVQueryMaxPageBytes — a cache value may be up
//     to the 16 MiB cache page size, while a page caps at 8 MiB. The key
//     matched the filter and the row is a true result; fetch the value with
//     get. This reading is unambiguous because an ordinary row in these modes
//     always carries a non-nil value: a zero-length STORED value encodes as
//     present-and-empty (hasVal = 1, len 0), never as absent.
//
// A per-row reason byte was considered and rejected: the distinction above is
// already carried by Return, which the caller always has, and adding a field
// would change the frame for every row to describe a case that is rare.
func EncodeKVQueryResult(r KVQueryResult) ([]byte, error) {
	if err := checkKVQueryCursorShape(r.Cursor); err != nil {
		return nil, err
	}
	buf := make([]byte, 0, 64+16*len(r.Rows))
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(r.Rows))) //nolint:gosec // row count is caller-controlled but bounded by KVQueryMaxPageBytes below
	for _, row := range r.Rows {
		if len(row.Key) > 0xFFFF {
			return nil, fmt.Errorf("%w: key %d bytes exceeds 65535", ErrKVQueryResult, len(row.Key))
		}
		buf = binary.BigEndian.AppendUint16(buf, uint16(len(row.Key))) //nolint:gosec // bounded above
		buf = append(buf, row.Key...)
		if row.Value != nil {
			if uint64(len(row.Value)) > math.MaxUint32 {
				return nil, fmt.Errorf("%w: value too large", ErrKVQueryResult)
			}
			buf = append(buf, 1)
			buf = binary.BigEndian.AppendUint32(buf, uint32(len(row.Value))) //nolint:gosec // bounded above
			buf = append(buf, row.Value...)
		} else {
			buf = append(buf, 0)
		}
	}
	buf = appendKVQueryCursor(buf, r.Cursor)
	if len(buf) > KVQueryMaxPageBytes {
		return nil, fmt.Errorf("%w: page %d bytes exceeds cap %d", ErrKVQueryResult, len(buf), KVQueryMaxPageBytes)
	}
	return buf, nil
}

// kvQueryRowMinBytes is the fewest bytes one KVQueryRow can possibly occupy:
// keyLen(2) + hasVal(1), key and value both empty/absent.
const kvQueryRowMinBytes = 3

// DecodeKVQueryResult reads a kv_query result frame produced by
// EncodeKVQueryResult. It rejects a frame over KVQueryMaxPageBytes outright
// (the coordinator's merge budget, re-checked here so a decoded part or a
// merged page can never silently exceed it) before doing any further work.
func DecodeKVQueryResult(b []byte) (KVQueryResult, error) {
	if len(b) > KVQueryMaxPageBytes {
		return KVQueryResult{}, fmt.Errorf("%w: page %d bytes exceeds cap %d", ErrKVQueryResult, len(b), KVQueryMaxPageBytes)
	}
	if len(b) < 4 {
		return KVQueryResult{}, ErrKVQueryArgsTruncated
	}
	nRows := binary.BigEndian.Uint32(b)
	off := 4
	if !CountFitsIn(int(nRows), len(b)-off, kvQueryRowMinBytes) {
		return KVQueryResult{}, ErrKVQueryArgsTruncated
	}
	rows := make([]KVQueryRow, 0, nRows)
	for i := 0; i < int(nRows); i++ {
		if len(b)-off < 2 {
			return KVQueryResult{}, ErrKVQueryArgsTruncated
		}
		klen := binary.BigEndian.Uint16(b[off:])
		off += 2
		if int(klen) > len(b)-off {
			return KVQueryResult{}, ErrKVQueryArgsTruncated
		}
		key := b[off : off+int(klen)]
		off += int(klen)

		if len(b)-off < 1 {
			return KVQueryResult{}, ErrKVQueryArgsTruncated
		}
		hasVal := b[off] != 0
		off++
		var val []byte
		if hasVal {
			if len(b)-off < 4 {
				return KVQueryResult{}, ErrKVQueryArgsTruncated
			}
			vlen := binary.BigEndian.Uint32(b[off:])
			off += 4
			if vlen > KVQueryMaxPageBytes {
				return KVQueryResult{}, fmt.Errorf("%w: value %d bytes exceeds page cap %d", ErrKVQueryResult, vlen, KVQueryMaxPageBytes)
			}
			if int(vlen) > len(b)-off {
				return KVQueryResult{}, ErrKVQueryArgsTruncated
			}
			val = b[off : off+int(vlen)]
			off += int(vlen)
		}
		rows = append(rows, KVQueryRow{Key: key, Value: val})
	}

	cursor, cn, err := decodeKVQueryCursor(b[off:])
	if err != nil {
		return KVQueryResult{}, err
	}
	off += cn
	if off != len(b) {
		return KVQueryResult{}, fmt.Errorf("%w: trailing bytes", ErrKVQueryResult)
	}
	return KVQueryResult{Rows: rows, Cursor: cursor}, nil
}

// CheckFilterBudget walks f once and rejects a tree with more than maxNodes
// nodes (composite and leaf alike) or nesting deeper than maxDepth levels
// (the root counts as depth 1). It stops at the first violation rather than
// finishing the walk, so a pathological tree (e.g. a few thousand nested
// "not" wrappers, which a 64KiB filter blob can easily encode) cannot make
// this function do more than maxNodes-ish units of work before rejecting it.
func CheckFilterBudget(f vtypes.Filter, maxNodes, maxDepth int) error {
	nodes := 0
	var walk func(f vtypes.Filter, depth int) error
	walk = func(f vtypes.Filter, depth int) error {
		nodes++
		if nodes > maxNodes {
			return fmt.Errorf("%w: more than %d nodes", ErrKVFilterBudget, maxNodes)
		}
		if depth > maxDepth {
			return fmt.Errorf("%w: nesting deeper than %d levels", ErrKVFilterBudget, maxDepth)
		}
		for i := range f.And {
			if err := walk(f.And[i], depth+1); err != nil {
				return err
			}
		}
		for i := range f.Or {
			if err := walk(f.Or[i], depth+1); err != nil {
				return err
			}
		}
		if f.Not != nil {
			if err := walk(*f.Not, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	return walk(f, 1)
}
