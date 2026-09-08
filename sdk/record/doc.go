// SPDX-License-Identifier: Apache-2.0

// Package record implements record paths and byte-level resolution for
// `operate` records (design doc §2.2/§2.9), shared by the vector store's
// record-typed payload cells and the KV store's operate records. It has no
// dependency on either engine: it operates purely on decoded record trees
// (sdk/wire) and produces payload-shaped values (sdk/vtypes) so filter and
// index code on both sides can treat a resolved record cell like any other
// payload value.
//
// # Path grammar
//
// A filter or index `field` string is first tried as an exact payload key
// (a key that itself contains "/" keeps working); otherwise SplitField
// divides it at the first "/" into a payload key and a record path, and the
// payload key must hold a record. ParsePath then parses the record path
// into 1-3 "/"-separated segments:
//
//	seg1        a field: a name, or "#N" for a position (N <= 65535)
//	seg2        a row key: decimal digits ([0-9]+, <= 20 digits, must fit
//	            uint64) for an integer key type, or a quoted string "..."
//	            (with \" and \\ escapes) for a FIXED key type
//	seg3        a column: a name, or "#N" for a position
//
// A field segment may end with "#count" (e.g. "b#count"), in which case it
// must be the only segment in the path — "#count" reports a table field's
// row count and does not compose with a row or column segment.
//
// Names are 1-255 bytes and may not contain '/', '#', or '"'. Any path that
// does not fit this grammar is rejected with an error wrapping ErrPath.
//
// # Determinism
//
// ParsePath and SplitField are pure functions of their input string: same
// bytes in, same result out, on every platform and Go version this module
// supports. Nothing in this package consults the clock, environment,
// filesystem, or any other ambient state. A later resolver built on top of
// this grammar (Task 3) must hold to the same rule: it is a pure function of
// the record bytes and the parsed Path, and any schema cache it keeps must
// be keyed by the schema blob's bytes so a cache hit can never change the
// answer.
//
// # Hardening
//
// A stored record's `set_payload` bytes are attacker-influenced: a client
// can store arbitrary bytes under any key, including bytes shaped to look
// like a record but crafted to defeat a careless decoder. ParsePath and
// SplitField only ever see the path string (never record bytes) and are
// therefore straightforward to keep panic-free: every length is checked
// before slicing, every numeric parse is bounds-checked before the value is
// trusted (position <= 65535, row-key digit count <= 20 and must fit
// uint64), and no segment is treated as well-formed until it has been fully
// validated. A resolver built on this grammar (Task 3, mirroring
// sdk/wire/operate_record.go) must extend the same discipline to the record
// bytes themselves: every offset bounds-checked before use, every count
// bounded before allocation, and only canonical uvarints accepted.
package record
