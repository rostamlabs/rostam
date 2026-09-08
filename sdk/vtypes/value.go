// SPDX-License-Identifier: Apache-2.0

package vtypes

import (
	"bytes"
	"fmt"
)

// ValueKind tags a Value's concrete type. The closed enumeration keeps the
// filter compiler's dispatch table small and lets the wire codec encode
// values without reflection.
type ValueKind uint8

const (
	ValueNone ValueKind = iota
	ValueString
	ValueInt
	ValueFloat
	ValueBool
	ValueStrings
	ValueInts
	ValueFloats
	// ValueGeo carries a WGS84 lat/lon point (degrees) in the Lat/Lon fields.
	// APPEND-ONLY: it must stay 8 so existing snapshots/WAL records (which encode
	// the kind as a raw u8) keep decoding correctly. Never renumber the kinds.
	ValueGeo
	// ValueRecord carries an opaque record-encoded byte blob (see sdk/record) in
	// the Rec field. APPEND-ONLY: it must stay 9 (ValueGeo+1) for the same reason
	// ValueGeo must stay 8 — the kind is a raw u8 on disk and on the wire. Never
	// renumber the kinds.
	ValueRecord
)

// Value is a tagged union holding one metadata attribute. Only the field
// matching Kind is meaningful; the others are zero. Built this way so a
// Value is a fixed-size struct (no boxing into interface{}) and the filter
// evaluator can do allocation-free comparisons in the hot path.
type Value struct {
	Kind ValueKind `json:"kind"`
	Str  string    `json:"str,omitempty"`
	Int  int64     `json:"int,omitempty"`
	Flt  float64   `json:"flt,omitempty"`
	Bool bool      `json:"bool,omitempty"`
	Strs []string  `json:"strs,omitempty"`
	Ints []int64   `json:"ints,omitempty"`
	Flts []float64 `json:"flts,omitempty"`
	// Lat/Lon hold a WGS84 geographic point (degrees) when Kind == ValueGeo.
	// Fixed-size scalars (no pointers/GC) keep the hot path allocation-free.
	// omitempty is intentional: it keeps non-geo Values' JSON clean (no spurious
	// "lat":0,"lon":0). Round-tripping the valid point (0,0) (Gulf of Guinea) is
	// safe because "kind":"geo" is always present and unmarshal zero-fills Lat/Lon
	// — i.e. omitempty here means "field absent", never "coordinate absent".
	Lat float64 `json:"lat,omitempty"`
	Lon float64 `json:"lon,omitempty"`
	// Rec holds an opaque record-encoded byte blob when Kind == ValueRecord.
	// Raw bytes marshal to base64 automatically via encoding/json's []byte
	// handling. omitempty is safe here (unlike Lat/Lon): an empty/nil record is
	// not a distinct meaningful value the way (0,0) is a real geo point.
	Rec []byte `json:"rec,omitempty"`
}

// IsZero reports whether the Value is the zero value (Kind == ValueNone).
func (v Value) IsZero() bool { return v.Kind == ValueNone }

// Equal reports whether two Values are kind-equal and value-equal. Used by
// the filter compiler's FilterEq predicate.
func (v Value) Equal(o Value) bool {
	if v.Kind != o.Kind {
		return false
	}
	switch v.Kind {
	case ValueString:
		return v.Str == o.Str
	case ValueInt:
		return v.Int == o.Int
	case ValueFloat:
		return v.Flt == o.Flt
	case ValueBool:
		return v.Bool == o.Bool
	case ValueStrings:
		return stringSliceEqual(v.Strs, o.Strs)
	case ValueInts:
		return int64SliceEqual(v.Ints, o.Ints)
	case ValueFloats:
		return float64SliceEqual(v.Flts, o.Flts)
	case ValueGeo:
		// CRITICAL: without this case geo values fall to the default `return true`
		// below, so any two geo points would compare equal — silently making
		// FilterEq always match and FilterNe never match on a geo field.
		return v.Lat == o.Lat && v.Lon == o.Lon
	case ValueRecord:
		// CRITICAL: same bug class as ValueGeo above — without this case every
		// pair of records would compare equal via the default `return true`.
		return bytes.Equal(v.Rec, o.Rec)
	}
	return true
}

// Metadata is the per-vector attribute map. nil signals "no metadata" —
// that's the common case so it's the cheap one.
type Metadata = map[string]Value

// NewString returns a Value carrying a string.
func NewString(s string) Value { return Value{Kind: ValueString, Str: s} }

// NewInt returns a Value carrying a signed integer.
func NewInt(i int64) Value { return Value{Kind: ValueInt, Int: i} }

// NewFloat returns a Value carrying a 64-bit float.
func NewFloat(f float64) Value { return Value{Kind: ValueFloat, Flt: f} }

// NewBool returns a Value carrying a boolean.
func NewBool(b bool) Value { return Value{Kind: ValueBool, Bool: b} }

// NewStrings returns a Value carrying a list of strings. The slice is
// stored by reference; callers must not mutate it after handing it off.
func NewStrings(s []string) Value { return Value{Kind: ValueStrings, Strs: s} }

// NewInts returns a Value carrying a list of int64.
func NewInts(i []int64) Value { return Value{Kind: ValueInts, Ints: i} }

// NewFloats returns a Value carrying a list of float64.
func NewFloats(f []float64) Value { return Value{Kind: ValueFloats, Flts: f} }

// NewGeo returns a Value carrying a WGS84 geographic point (lat/lon in degrees).
func NewGeo(lat, lon float64) Value { return Value{Kind: ValueGeo, Lat: lat, Lon: lon} }

// NewRecord returns a Value carrying an opaque record-encoded byte blob. rec
// is stored by reference; callers must not mutate it after handing it off.
func NewRecord(rec []byte) Value { return Value{Kind: ValueRecord, Rec: rec} }

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func int64SliceEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func float64SliceEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

var valueKindNames = map[ValueKind]string{
	ValueNone:    "none",
	ValueString:  "string",
	ValueInt:     "int",
	ValueFloat:   "float",
	ValueBool:    "bool",
	ValueStrings: "strings",
	ValueInts:    "ints",
	ValueFloats:  "floats",
	ValueGeo:     "geo",
	ValueRecord:  "record",
}

var valueKindByName = func() map[string]ValueKind {
	m := make(map[string]ValueKind, len(valueKindNames))
	for k, v := range valueKindNames {
		m[v] = k
	}
	return m
}()

// MarshalText renders the ValueKind as its lowercase name, so a Value
// serializes as {"kind":"string",...} in JSON.
func (k ValueKind) MarshalText() ([]byte, error) {
	name, ok := valueKindNames[k]
	if !ok {
		return nil, fmt.Errorf("vector: unknown ValueKind %d", k)
	}
	return []byte(name), nil
}

// UnmarshalText parses a ValueKind name produced by MarshalText.
func (k *ValueKind) UnmarshalText(text []byte) error {
	v, ok := valueKindByName[string(text)]
	if !ok {
		return fmt.Errorf("vector: unknown ValueKind %q", text)
	}
	*k = v
	return nil
}
