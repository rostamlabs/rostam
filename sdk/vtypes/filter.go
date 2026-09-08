// SPDX-License-Identifier: Apache-2.0

package vtypes

import (
	"fmt"
)

// FilterOp identifies a node type in a Filter tree. Composite ops (And, Or,
// Not) combine child filters; leaf ops compare a metadata field against a
// Value.
type FilterOp uint8

const (
	FilterAnd FilterOp = iota
	FilterOr
	FilterNot
	FilterEq
	FilterNe
	FilterGt
	FilterGte
	FilterLt
	FilterLte
	FilterIn       // value is an array; field's scalar must be a member
	FilterContains // field is an array; value scalar must be a member

	// Rich payload operators. Appended after FilterContains (=10) so existing
	// wire-encoded op numbers are never renumbered.
	FilterMatch   // full-text-lite: field tokens ⊇ query tokens (case-insensitive)
	FilterRegex   // RE2 pattern match against a string field / any string element
	FilterIsEmpty // field absent, ValueNone, "" or empty array
	FilterIsNull  // field present AND kind == ValueNone (explicit null)
	FilterDtGt    // datetime field (int64 unix-ms) > RFC3339 literal
	FilterDtGte   // datetime field >= RFC3339 literal
	FilterDtLt    // datetime field < RFC3339 literal
	FilterDtLte   // datetime field <= RFC3339 literal

	// Geo operators. Appended after FilterDtLte so existing wire-encoded op
	// numbers are never renumbered. Each carries its region in the Filter.Geo
	// pointer (a GeoCondition), not in Value, because a geo region doesn't fit
	// the scalar Value union.
	FilterGeoRadius  // geo field within RadiusM meters (haversine) of center
	FilterGeoBox     // geo field inside an SW->NE bounding box (inclusive)
	FilterGeoPolygon // geo field inside a polygon exterior ring (ray-casting)

	// Record row-presence operators. Appended after FilterGeoPolygon so
	// existing wire-encoded op numbers are never renumbered. Both address a
	// path ending in a table row segment (see sdk/record); they take no
	// Value. FilterRowExists is true iff the row is present; FilterRowAbsent
	// is its asymmetric negation (see vector/filter.go's compileRowPresence
	// doc comment for the asymmetry: it is NOT simply "not exists").
	FilterRowExists
	FilterRowAbsent
)

var filterOpNames = map[FilterOp]string{
	FilterAnd:        "and",
	FilterOr:         "or",
	FilterNot:        "not",
	FilterEq:         "eq",
	FilterNe:         "ne",
	FilterGt:         "gt",
	FilterGte:        "gte",
	FilterLt:         "lt",
	FilterLte:        "lte",
	FilterIn:         "in",
	FilterContains:   "contains",
	FilterMatch:      "match",
	FilterRegex:      "regex",
	FilterIsEmpty:    "is_empty",
	FilterIsNull:     "is_null",
	FilterDtGt:       "dt_gt",
	FilterDtGte:      "dt_gte",
	FilterDtLt:       "dt_lt",
	FilterDtLte:      "dt_lte",
	FilterGeoRadius:  "geo_radius",
	FilterGeoBox:     "geo_bounding_box",
	FilterGeoPolygon: "geo_polygon",
	FilterRowExists:  "row_exists",
	FilterRowAbsent:  "row_absent",
}

var filterOpByName = func() map[string]FilterOp {
	m := make(map[string]FilterOp, len(filterOpNames))
	for k, v := range filterOpNames {
		m[v] = k
	}
	return m
}()

// MarshalText renders the op as its lowercase name for JSON.
func (o FilterOp) MarshalText() ([]byte, error) {
	name, ok := filterOpNames[o]
	if !ok {
		return nil, fmt.Errorf("vector: unknown FilterOp %d", o)
	}
	return []byte(name), nil
}

// UnmarshalText parses an op name produced by MarshalText.
func (o *FilterOp) UnmarshalText(text []byte) error {
	v, ok := filterOpByName[string(text)]
	if !ok {
		return fmt.Errorf("vector: unknown FilterOp %q", text)
	}
	*o = v
	return nil
}

// GeoCondition carries the geographic region for a geo filter op. Which fields
// are meaningful depends on the op:
//
//	FilterGeoRadius  -> CenterLat, CenterLon, RadiusM
//	FilterGeoBox     -> MinLat, MinLon (SW) .. MaxLat, MaxLon (NE)
//	FilterGeoPolygon -> Polygon: flat lat,lon,lat,lon,... exterior ring
//
// All coordinates are WGS84 degrees (lat in [-90,90], lon in [-180,180]),
// validated once at compile time. omitempty keeps the JSON clean and makes the
// field fully backward-compatible (legacy filters omit "geo"). Antimeridian /
// pole-crossing regions and polygon holes are documented non-goals.
type GeoCondition struct {
	CenterLat float64   `json:"center_lat,omitempty"`
	CenterLon float64   `json:"center_lon,omitempty"`
	RadiusM   float64   `json:"radius_m,omitempty"`
	MinLat    float64   `json:"min_lat,omitempty"`
	MinLon    float64   `json:"min_lon,omitempty"`
	MaxLat    float64   `json:"max_lat,omitempty"`
	MaxLon    float64   `json:"max_lon,omitempty"`
	Polygon   []float64 `json:"polygon,omitempty"` // flat lat,lon,lat,lon,... exterior ring
}

// Filter is a predicate tree over vector metadata. It is JSON-serializable
// (Pinecone-style dialect) and compiles to an allocation-free Predicate.
//
// Composite nodes use And/Or/Not; leaf nodes use Field + Value. The zero
// Filter (Op==FilterAnd with no children) means "match all".
type Filter struct {
	Op    FilterOp `json:"op"`
	Field string   `json:"field,omitempty"`
	Value Value    `json:"value,omitempty"`
	And   []Filter `json:"and,omitempty"`
	Or    []Filter `json:"or,omitempty"`
	Not   *Filter  `json:"not,omitempty"`
	// Geo carries the region for a geo op (radius/box/polygon). Pointer +
	// omitempty mirrors Not: additive and backward-compatible (legacy filters
	// omit it). Used only by the FilterGeo* ops.
	Geo *GeoCondition `json:"geo,omitempty"`
}

// IsZero reports whether the Filter is the empty "match all" filter.
func (f Filter) IsZero() bool {
	return f.Op == FilterAnd && f.Field == "" && f.Value.IsZero() &&
		len(f.And) == 0 && len(f.Or) == 0 && f.Not == nil && f.Geo == nil
}
