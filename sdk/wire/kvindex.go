// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
)

// KVIndexDef.Kind: whether a definition indexes a top-level scalar field or a
// table field's row count (a "#count" path). Carried on the wire rather than
// dispatched on because record.Resolve already reports which the parsed path
// names — the definition's Kind must AGREE with it (see Validate), so a
// mismatch is caught once, at creation, instead of silently mis-indexing every
// write thereafter.
const (
	KVIndexKindScalar uint8 = 0
	KVIndexKindCount  uint8 = 1
)

// Caps on a KVIndexDef's fields and on the number of definitions a cluster may
// carry. All three length caps fit in the u8 the frame spends on them (255),
// so no 386-widening guard is needed decoding them — a u8 count can never
// overflow int on any platform this builds for.
const (
	KVIndexMaxNameLen   = 64
	KVIndexMaxPrefixLen = 255
	KVIndexMaxPathLen   = 255
	KVIndexMaxDefs      = 64
)

// kvIndexCountSuffix marks a payload path as addressing a table field's row
// count rather than a scalar value — the "#count" path form record.ParsePath
// also recognizes (SegCount). Checked here, without importing record (which
// itself imports this package), by a plain suffix test: a wire-level
// definition only needs to agree with the SHAPE of a "#count" path, not parse
// its full grammar — record.ParsePath (ops/kvindex.DefFrom, a later task) does
// the full parse and is the actual authority on whether the path resolves.
const kvIndexCountSuffix = "#count"

// ErrKVIndexDef marks a KVIndexDef (or a __kv_index_set__/__kv_index_list__
// frame built from one) as structurally or semantically invalid: a name
// outside its charset/length cap, an oversized prefix or path, a path shaped
// as a row path (it must be a top-level field or a "#count" path), a
// path/Kind disagreement, or a frame carrying more definitions than
// KVIndexMaxDefs. Returned by both KVIndexDef.Validate and the decoders.
var ErrKVIndexDef = errors.New("wire: invalid kv index definition")

// KVIndexDef is one cluster-wide KV index definition: an equality/range
// posting index over KeyPrefix-scoped keys' PayloadPath value. It is meta-log
// state (see the phase-3 plan's decisions §1) — never a durability unit on
// its own — so this type and its codec carry only what the meta FSM stores
// and what a node needs to derive its local postings from it.
type KVIndexDef struct {
	// Name identifies the index; it is what kv_query's Index field names and
	// what __kv_index_set__ upserts by. Charset [A-Za-z0-9_.:-]{1,64}.
	Name string
	// KeyPrefix scopes the index to keys sharing this byte prefix. Empty
	// indexes the whole keyspace. At most KVIndexMaxPrefixLen bytes.
	KeyPrefix []byte
	// PayloadPath names the record field this index posts on: a bare
	// top-level field name, or that field's "#count" row-count path. Never a
	// row/column (table-cell) path — see Validate.
	PayloadPath string
	// Kind is KVIndexKindScalar or KVIndexKindCount, and must agree with
	// whether PayloadPath ends in "#count" (see Validate).
	Kind uint8
	// Enabled is carried for completeness (a disabled definition still
	// occupies a name/meta-log slot without being installed); nothing in
	// this package branches on it.
	Enabled bool
}

// validKVName reports whether s is a legal KVIndexDef.Name or kv_query
// Index: 1-64 bytes drawn from [A-Za-z0-9_.:-]. A hand-written byte loop
// rather than regexp: the charset is a fixed class with no backtracking risk
// either way, but every other name/charset check in this package is a plain
// loop (see record.validateName's sibling in the record package), and a
// decode-path check should not introduce a dependency this package doesn't
// already have.
func validKVName(s string) bool {
	if len(s) == 0 || len(s) > KVIndexMaxNameLen {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z':
		case c >= 'a' && c <= 'z':
		case c >= '0' && c <= '9':
		case c == '_' || c == '.' || c == ':' || c == '-':
		default:
			return false
		}
	}
	return true
}

// Validate checks d's SHAPE: name charset/length, prefix/path length caps,
// that PayloadPath is a top-level field or a "#count" path (never a row
// path — a path containing '/' addresses a table row or column, which no
// KV index may target), and that Kind agrees with the "#count" suffix. It
// does NOT parse PayloadPath as a full record.Path (that would need
// record, which imports this package) — a later stage (ops/kvindex.DefFrom)
// does the full parse and is the authority on whether the path actually
// resolves against a schema.
func (d KVIndexDef) Validate() error {
	if !validKVName(d.Name) {
		return fmt.Errorf("%w: name %q must match [A-Za-z0-9_.:-]{1,%d}", ErrKVIndexDef, d.Name, KVIndexMaxNameLen)
	}
	if len(d.KeyPrefix) > KVIndexMaxPrefixLen {
		return fmt.Errorf("%w: key prefix too long (%d bytes, cap %d)", ErrKVIndexDef, len(d.KeyPrefix), KVIndexMaxPrefixLen)
	}
	if d.PayloadPath == "" {
		return fmt.Errorf("%w: empty payload path", ErrKVIndexDef)
	}
	if len(d.PayloadPath) > KVIndexMaxPathLen {
		return fmt.Errorf("%w: payload path too long (%d bytes, cap %d)", ErrKVIndexDef, len(d.PayloadPath), KVIndexMaxPathLen)
	}
	if strings.ContainsRune(d.PayloadPath, '/') {
		return fmt.Errorf("%w: an index path must be a top-level field or a #count", ErrKVIndexDef)
	}
	isCount := strings.HasSuffix(d.PayloadPath, kvIndexCountSuffix)
	switch d.Kind {
	case KVIndexKindScalar:
		if isCount {
			return fmt.Errorf("%w: path %q ends in #count but Kind is scalar", ErrKVIndexDef, d.PayloadPath)
		}
	case KVIndexKindCount:
		if !isCount {
			return fmt.Errorf("%w: Kind is count but path %q has no #count suffix", ErrKVIndexDef, d.PayloadPath)
		}
	default:
		return fmt.Errorf("%w: unknown kind %d", ErrKVIndexDef, d.Kind)
	}
	return nil
}

// AppendKVIndexDef appends d's wire frame to dst:
//
//	[nameLen u8][name][prefixLen u8][prefix][pathLen u8][path][kind u8][enabled u8]
//
// It assumes d is well-formed (Validate passed, or the caller otherwise knows
// every length fits a u8) — like every Append* in this package, Encode is the
// trusting side and Decode is the one hardened against hostile bytes.
func AppendKVIndexDef(dst []byte, d KVIndexDef) []byte {
	dst = append(dst, byte(len(d.Name))) //nolint:gosec // bounded by KVIndexMaxNameLen on a validated def
	dst = append(dst, d.Name...)
	dst = append(dst, byte(len(d.KeyPrefix))) //nolint:gosec // bounded by KVIndexMaxPrefixLen on a validated def
	dst = append(dst, d.KeyPrefix...)
	dst = append(dst, byte(len(d.PayloadPath))) //nolint:gosec // bounded by KVIndexMaxPathLen on a validated def
	dst = append(dst, d.PayloadPath...)
	dst = append(dst, d.Kind)
	if d.Enabled {
		dst = append(dst, 1)
	} else {
		dst = append(dst, 0)
	}
	return dst
}

// DecodeKVIndexDef reads one KVIndexDef frame from the front of b, returning
// the bytes consumed (n) so a caller reading several back-to-back (as
// __kv_index_list__ does) can advance past it. Every length is a single byte
// (max 255), so it is read and bounds-checked directly — there is no 32-bit
// widening hazard to guard with CountFitsIn (a u8 count cannot overflow int
// on any platform this builds for). It does NOT call Validate — this is the
// purely structural half of the codec; callers that need a semantically
// valid definition call Validate themselves (see DecodeKVIndexSetArgs's
// caller in cluster, a later task).
func DecodeKVIndexDef(b []byte) (d KVIndexDef, n int, err error) {
	if len(b) < 1 {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	nameLen := int(b[0])
	off := 1
	if len(b)-off < nameLen {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	name := string(b[off : off+nameLen])
	off += nameLen

	if len(b)-off < 1 {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	prefixLen := int(b[off])
	off++
	if len(b)-off < prefixLen {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	var prefix []byte
	if prefixLen > 0 {
		prefix = b[off : off+prefixLen]
	}
	off += prefixLen

	if len(b)-off < 1 {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	pathLen := int(b[off])
	off++
	if len(b)-off < pathLen {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	path := string(b[off : off+pathLen])
	off += pathLen

	if len(b)-off < 2 {
		return KVIndexDef{}, 0, ErrShortArgs
	}
	kind := b[off]
	enabled := b[off+1] != 0
	off += 2

	return KVIndexDef{Name: name, KeyPrefix: prefix, PayloadPath: path, Kind: kind, Enabled: enabled}, off, nil
}

// EncodeKVIndexSetArgs encodes the __kv_index_set__ admin op's args: exactly
// one KVIndexDef frame, with no trailer.
func EncodeKVIndexSetArgs(d KVIndexDef) []byte {
	return AppendKVIndexDef(nil, d)
}

// DecodeKVIndexSetArgs reads the __kv_index_set__ args frame, rejecting any
// trailing bytes (the frame is exactly one KVIndexDef, self-delimiting, so
// anything left over is corruption, not a forward-compatible extension).
func DecodeKVIndexSetArgs(args []byte) (KVIndexDef, error) {
	d, n, err := DecodeKVIndexDef(args)
	if err != nil {
		return KVIndexDef{}, err
	}
	if n != len(args) {
		return KVIndexDef{}, fmt.Errorf("%w: trailing bytes", ErrKVIndexDef)
	}
	return d, nil
}

// EncodeKVIndexList encodes the __kv_index_list__ result:
//
//	[n u16]{ KVIndexDef [ready u8] }
//
// ready[i] reports whether defs[i] is ready on EVERY hosted group (see
// kvindex.Set.IsReady in the ops/kvindex package, a later task); ready may be
// shorter than defs (a missing entry encodes as not-ready) but is never
// longer in a well-formed call.
func EncodeKVIndexList(defs []KVIndexDef, ready []bool) []byte {
	buf := make([]byte, 0, 2+16*len(defs))
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(defs))) //nolint:gosec // bounded by KVIndexMaxDefs at the call site
	for i, d := range defs {
		buf = AppendKVIndexDef(buf, d)
		if i < len(ready) && ready[i] {
			buf = append(buf, 1)
		} else {
			buf = append(buf, 0)
		}
	}
	return buf
}

// kvIndexDefMinBytes is the fewest bytes one KVIndexDef+ready entry in a
// __kv_index_list__ frame can possibly occupy: 3 one-byte length prefixes
// (name/prefix/path, each legally zero — an empty prefix indexes the whole
// keyspace, though an empty name is invalid, Validate is not this decoder's
// job) + kind(1) + enabled(1) + ready(1). Used to bound the declared entry
// count against the remaining bytes before any allocation sized by it.
const kvIndexDefMinBytes = 6

// DecodeKVIndexList reads the __kv_index_list__ result produced by
// EncodeKVIndexList.
func DecodeKVIndexList(b []byte) (defs []KVIndexDef, ready []bool, err error) {
	if len(b) < 2 {
		return nil, nil, ErrShortArgs
	}
	n := binary.BigEndian.Uint16(b)
	off := 2
	if !CountFitsIn(int(n), len(b)-off, kvIndexDefMinBytes) {
		return nil, nil, ErrShortArgs
	}
	if int(n) > KVIndexMaxDefs {
		return nil, nil, fmt.Errorf("%w: %d definitions exceeds cap %d", ErrKVIndexDef, n, KVIndexMaxDefs)
	}
	defs = make([]KVIndexDef, 0, n)
	ready = make([]bool, 0, n)
	for i := 0; i < int(n); i++ {
		d, dn, derr := DecodeKVIndexDef(b[off:])
		if derr != nil {
			return nil, nil, derr
		}
		off += dn
		if len(b)-off < 1 {
			return nil, nil, ErrShortArgs
		}
		defs = append(defs, d)
		ready = append(ready, b[off] != 0)
		off++
	}
	if off != len(b) {
		return nil, nil, fmt.Errorf("%w: trailing bytes", ErrKVIndexDef)
	}
	return defs, ready, nil
}
