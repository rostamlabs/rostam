// SPDX-License-Identifier: Apache-2.0

package record

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// ErrPath is wrapped by every error ParsePath and its helpers return for a
// path string that does not fit the grammar documented in the package
// comment. Use errors.Is(err, ErrPath) to test for it.
var ErrPath = errors.New("record: bad path")

// ErrRecord is wrapped by every error a Resolver returns for record bytes
// that cannot be decoded. Use errors.Is(err, ErrRecord) to test for it.
var ErrRecord = errors.New("record: malformed record")

// SegKind identifies the role a Segment plays in a Path.
type SegKind uint8

const (
	// SegField is a Path's first segment: the record's top-level field,
	// by name or by position.
	SegField SegKind = iota
	// SegRow is a Path's second segment: a table row key, by decimal
	// value or by quoted FIXED text.
	SegRow
	// SegCol is a Path's third segment: a table column, by name or by
	// position.
	SegCol
	// SegCount marks a Path whose sole segment is a field's "#count"
	// suffix — the field's table row count.
	SegCount
)

// Segment is one parsed component of a Path. Which fields are meaningful
// depends on Kind:
//
//   - SegField, SegCount: Name (if !ByPos) or Pos (if ByPos) identifies the
//     field.
//   - SegRow: KeyText always holds the segment's original text. For a
//     decimal key, KeyQuoted is false and Key is nil (a later stage encodes
//     the decimal text to the key's on-disk width). For a quoted key,
//     KeyQuoted is true and Key holds the unescaped bytes.
//   - SegCol: Name (if !ByPos) or Pos (if ByPos) identifies the column.
type Segment struct {
	Kind      SegKind
	Name      string
	Pos       uint32
	ByPos     bool
	Key       []byte
	KeyText   string
	KeyQuoted bool
}

// Path is a parsed record path: 1-3 segments as documented in the package
// comment. When Segs[0].Kind == SegCount it is the only segment.
type Path struct {
	Segs []Segment
}

// maxPos is the largest position a "#N" segment may name.
const maxPos = 65535

// maxRowKeyDigits is the longest decimal row key ParsePath accepts (a
// uint64's maximum decimal representation is 20 digits).
const maxRowKeyDigits = 20

// countSuffix is the literal suffix that turns a field segment into a
// SegCount path.
const countSuffix = "#count"

// maxNameLen is the longest field, column, or unquoted row-key name
// validateName accepts.
const maxNameLen = 255

// maxPathBytes is the longest path string ParsePath can possibly accept, and
// the bound it applies BEFORE strings.Split.
//
// WHY THE BOUND IS AHEAD OF THE SPLIT. strings.Split allocates one []string
// header per segment (about 16 bytes each) and only then can the segment-count
// check reject the input, so an unbounded field of '/' costs ~16x its own
// length in transient headers before the rejection fires. The only outer limit
// is the route body cap (httpapi's maxJSONBody, 32 MiB), which makes a
// worst-case field cost roughly 512 MiB of headers — once per filter leaf per
// request, multiplied by request concurrency. Segment-level bounds cannot help:
// validateName's 255 and parseRowSeg's quoted-key cap both live in helpers that
// run AFTER the split.
//
// DERIVATION. A path is at most three segments joined by two '/':
//
//	field:  a name of at most maxNameLen bytes (validateName), or "#65535";
//	        the "#count" form is a name plus 6 bytes but must be the ONLY
//	        segment, so 261 bytes total — well under the three-segment sum.
//	row:    a quoted key of at most 2+2*wire.OperateMaxKeyLen RAW bytes (the
//	        pre-unescape bound in parseRowSeg: every escape is two bytes
//	        producing one), or at most maxRowKeyDigits decimal digits.
//	column: a name of at most maxNameLen bytes.
//
// Nothing legal can exceed the sum, so the check rejects only inputs a later
// stage would reject anyway — just without paying for the split first.
const maxPathBytes = maxNameLen + 1 + (2 + 2*wire.OperateMaxKeyLen) + 1 + maxNameLen

// ParsePath parses a record path — the part of a filter or index "field"
// string that names a location inside a record's decoded fields, as
// documented in the package comment. It never panics: every rejection
// returns an error wrapping ErrPath.
func ParsePath(s string) (Path, error) {
	if s == "" {
		return Path{}, fmt.Errorf("%w: empty path", ErrPath)
	}
	// Ahead of the split, so a pathological field costs one length comparison
	// instead of one []string header per '/'. See maxPathBytes. The message
	// carries the length, never the input, so rejecting stays allocation-flat
	// in the size of s.
	if len(s) > maxPathBytes {
		return Path{}, fmt.Errorf("%w: path too long (%d bytes, cap is %d)", ErrPath, len(s), maxPathBytes)
	}

	parts := strings.Split(s, "/")
	if len(parts) > 3 {
		return Path{}, fmt.Errorf("%w: too many segments in %q", ErrPath, s)
	}
	for _, p := range parts {
		if p == "" {
			return Path{}, fmt.Errorf("%w: empty segment in %q", ErrPath, s)
		}
	}

	fieldSeg, isCount, err := parseFieldSeg(parts[0], len(parts) == 1)
	if err != nil {
		return Path{}, err
	}
	if isCount {
		return Path{Segs: []Segment{fieldSeg}}, nil
	}

	segs := make([]Segment, 1, len(parts))
	segs[0] = fieldSeg

	if len(parts) >= 2 {
		rowSeg, err := parseRowSeg(parts[1])
		if err != nil {
			return Path{}, err
		}
		segs = append(segs, rowSeg)
	}
	if len(parts) == 3 {
		colSeg, err := parseColSeg(parts[2])
		if err != nil {
			return Path{}, err
		}
		segs = append(segs, colSeg)
	}

	return Path{Segs: segs}, nil
}

// SplitField splits field at its first '/' into a payload key and a record
// path. ok is false when field contains no '/' (payloadKey and path are
// both "" in that case; the caller should treat field itself as an exact
// payload key). A leading '/' (e.g. "/x") yields an EMPTY payloadKey with
// ok == true, and that is a legal outcome, not one the caller must reject:
// an empty payload key resolves against the metadata entry whose name is the
// empty string, exactly as any other key resolves against its own entry.
// vector.lookupPath relies on this — it looks up m[""] like any other key and
// finds the record only if a point actually stored one under "" — and the
// index side agrees, so the two never disagree about a "/x" field. Callers
// that want to forbid an empty payload key must say so themselves.
func SplitField(field string) (payloadKey, path string, ok bool) {
	idx := strings.IndexByte(field, '/')
	if idx < 0 {
		return "", "", false
	}
	return field[:idx], field[idx+1:], true
}

// parseFieldSeg parses a Path's first segment. onlySeg reports whether this
// is the only segment in the path (required for a "#count" suffix to be
// valid). isCount reports whether the segment was a "#count" form, in which
// case the returned Segment already has Kind == SegCount.
func parseFieldSeg(s string, onlySeg bool) (seg Segment, isCount bool, err error) {
	if strings.HasSuffix(s, countSuffix) {
		base := s[:len(s)-len(countSuffix)]
		if base == "" {
			return Segment{}, false, fmt.Errorf("%w: %q has no field before #count", ErrPath, s)
		}
		if !onlySeg {
			return Segment{}, false, fmt.Errorf("%w: #count must be the only segment in %q", ErrPath, s)
		}
		name, pos, byPos, err := parseNameOrPos(base)
		if err != nil {
			return Segment{}, false, err
		}
		return Segment{Kind: SegCount, Name: name, Pos: pos, ByPos: byPos}, true, nil
	}

	name, pos, byPos, err := parseNameOrPos(s)
	if err != nil {
		return Segment{}, false, err
	}
	return Segment{Kind: SegField, Name: name, Pos: pos, ByPos: byPos}, false, nil
}

// parseColSeg parses a Path's third segment (a column: name or "#N").
func parseColSeg(s string) (Segment, error) {
	name, pos, byPos, err := parseNameOrPos(s)
	if err != nil {
		return Segment{}, err
	}
	return Segment{Kind: SegCol, Name: name, Pos: pos, ByPos: byPos}, nil
}

// parseNameOrPos parses a field or column segment: either a bare name, or
// "#N" naming a position (N <= maxPos).
func parseNameOrPos(s string) (name string, pos uint32, byPos bool, err error) {
	if s == "" {
		return "", 0, false, fmt.Errorf("%w: empty field or column segment", ErrPath)
	}
	if s[0] == '#' {
		rest := s[1:]
		if rest == "" || !isAllDigits(rest) {
			return "", 0, false, fmt.Errorf("%w: malformed position %q", ErrPath, s)
		}
		v, perr := strconv.ParseUint(rest, 10, 32)
		if perr != nil || v > maxPos {
			return "", 0, false, fmt.Errorf("%w: position out of range %q", ErrPath, s)
		}
		return "", uint32(v), true, nil
	}
	if err := validateName(s); err != nil {
		return "", 0, false, err
	}
	return s, 0, false, nil
}

// parseRowSeg parses a Path's second segment: a decimal row key or a
// quoted FIXED row key.
func parseRowSeg(s string) (Segment, error) {
	if s[0] == '"' {
		// Bound the RAW inner length BEFORE unescaping. unescapeQuoted allocates
		// len(inner) bytes, and this parse runs per point for a filter leaf that
		// is not compiled (and once per compile for one that is), so an unbounded
		// quoted key is a per-row allocation the caller controls: a filter field
		// of session/"<30 MiB of x>"/col would allocate 30 MB for every point in
		// the scan. Every escape is two bytes producing one, so an inner longer
		// than 2*OperateMaxKeyLen cannot unescape to a legal key — checking the
		// raw length first caps the unescape allocation at ~2*255 bytes, and the
		// exact check below then holds the UNESCAPED key to the wire cap.
		//
		// The bound is semantically free: a key longer than OperateMaxKeyLen
		// cannot name a row in any record. In schema mode findSchemaRow answers
		// Absent on any key-width mismatch, and in dynamic mode a row key's
		// length prefix is a u8 (sdk/wire/operate_record.go), so no encodable
		// record has a longer key. Rejecting is therefore identical in meaning to
		// resolving to Absent, and strictly cheaper.
		if len(s) > 2+2*wire.OperateMaxKeyLen {
			return Segment{}, fmt.Errorf("%w: quoted row key too long (%d raw bytes, cap is %d unescaped) %.64q",
				ErrPath, len(s)-2, wire.OperateMaxKeyLen, s)
		}
		key, ok := unescapeQuoted(s)
		if !ok {
			return Segment{}, fmt.Errorf("%w: malformed quoted row key %q", ErrPath, s)
		}
		if len(key) > wire.OperateMaxKeyLen {
			return Segment{}, fmt.Errorf("%w: quoted row key too long (%d bytes, cap is %d) %q",
				ErrPath, len(key), wire.OperateMaxKeyLen, s)
		}
		return Segment{Kind: SegRow, Key: key, KeyText: s, KeyQuoted: true}, nil
	}
	if s[0] == '#' {
		return Segment{}, fmt.Errorf("%w: a position is not a valid row key %q", ErrPath, s)
	}
	if !isAllDigits(s) {
		return Segment{}, fmt.Errorf("%w: row key is neither decimal nor quoted %q", ErrPath, s)
	}
	if len(s) > maxRowKeyDigits {
		return Segment{}, fmt.Errorf("%w: row key too long %q", ErrPath, s)
	}
	if _, perr := strconv.ParseUint(s, 10, 64); perr != nil {
		return Segment{}, fmt.Errorf("%w: row key does not fit uint64 %q", ErrPath, s)
	}
	return Segment{Kind: SegRow, KeyText: s}, nil
}

// validateName reports whether s is a valid field, row(-quoted excluded),
// or column name: 1-255 bytes, containing none of '/', '#', '"'.
func validateName(s string) error {
	if len(s) < 1 || len(s) > maxNameLen {
		return fmt.Errorf("%w: name length out of range (1-%d bytes) %q", ErrPath, maxNameLen, s)
	}
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '/', '#', '"':
			return fmt.Errorf("%w: name contains an invalid character %q", ErrPath, s)
		}
	}
	return nil
}

// isAllDigits reports whether s is non-empty and consists only of ASCII
// digits.
func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// unescapeQuoted parses a quoted row-key segment (including its
// surrounding '"' characters) and returns its unescaped byte content. Only
// \" and \\ are recognized escapes; any other backslash sequence, an
// unescaped '"' before the closing quote, or a missing closing quote is
// rejected.
func unescapeQuoted(s string) (key []byte, ok bool) {
	if len(s) < 2 || s[0] != '"' || s[len(s)-1] != '"' {
		return nil, false
	}
	inner := s[1 : len(s)-1]
	buf := make([]byte, 0, len(inner))
	for i := 0; i < len(inner); {
		c := inner[i]
		switch c {
		case '"':
			// An unescaped quote before the segment's closing quote is
			// malformed (it would have terminated the string early).
			return nil, false
		case '\\':
			if i+1 >= len(inner) {
				return nil, false
			}
			next := inner[i+1]
			if next != '"' && next != '\\' {
				return nil, false
			}
			buf = append(buf, next)
			i += 2
		default:
			buf = append(buf, c)
			i++
		}
	}
	return buf, true
}
