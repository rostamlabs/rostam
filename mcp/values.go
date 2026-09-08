// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// jsonID is a point id as it arrives on the tool wire: a JSON number OR a
// decimal string. It is a decode-only type — a generic tool's response
// never marshals a jsonID, it emits ids through idOut instead, since the
// safe wire form differs by direction (see idOut).
//
// The engine's ids are full-width uint64, but a JSON number in a JavaScript
// MCP client — which is most of them — is an IEEE-754 double, exact only up to
// 2^53-1. An id above that is rounded the moment it is parsed, so a client
// that reads an id out of a search result and hands it to get, upsert, or
// delete would silently be naming a different point than the one it saw.
// Accepting the decimal string form gives those clients a way to say the exact
// id; ids the MCP server generates itself are kept inside the safe range (see
// memoryID) so the common path never needs it.
type jsonID uint64

func (id *jsonID) UnmarshalJSON(b []byte) error {
	s := string(bytes.TrimSpace(b))
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		var str string
		if err := json.Unmarshal(b, &str); err != nil {
			return fmt.Errorf("mcp: bad point id %s: %w", b, err)
		}
		s = strings.TrimSpace(str)
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return fmt.Errorf("mcp: %s is not a valid point id: want a JSON number or a decimal string, both within uint64", b)
	}
	*id = jsonID(v)
	return nil
}

// idsToUint64 converts decoded wire ids to the engine's type.
func idsToUint64(ids []jsonID) []uint64 {
	out := make([]uint64, len(ids))
	for i, id := range ids {
		out[i] = uint64(id)
	}
	return out
}

// idOut converts an engine id to the form a generic tool result puts on the
// wire: a plain JSON number when it round-trips exactly through a float64
// (id <= 2^53-1, jsSafeIDMask — see its doc comment), and a decimal string
// above that boundary. Without this, a big id comes back from a generic tool
// (upsert/get/delete/search) as a JSON number, a JavaScript client rounds it
// parsing the response, and every later call naming that id addresses the
// wrong point.
//
// Memory-tool ids never need this: memoryID masks every generated id into
// the safe range itself, so they are always plain numbers.
func idOut(id uint64) any {
	if id > jsSafeIDMask {
		return strconv.FormatUint(id, 10)
	}
	return id
}

// idsOut is idOut applied to a slice, for tool results that return several
// ids (delete's deleted/missing, get's missing).
func idsOut(ids []uint64) []any {
	out := make([]any, len(ids))
	for i, id := range ids {
		out[i] = idOut(id)
	}
	return out
}

// idSchema is the JSON Schema fragment for one point id, and idsSchema for an
// array of them. Both advertise the string alternative so a client knows the
// escape hatch exists without having to read the docs.
func idSchema(description string) map[string]any {
	return map[string]any{
		"type":        []any{"integer", "string"},
		"description": description + `; a JSON number, or a decimal string for ids above 2^53-1 (which a JavaScript client cannot represent exactly as a number)`,
	}
}

func idsSchema(description string) map[string]any {
	return map[string]any{
		"type":        "array",
		"items":       map[string]any{"type": []any{"integer", "string"}},
		"description": description + `; each id is a JSON number, or a decimal string for ids above 2^53-1`,
	}
}

// jsonToMetadata converts a JSON object (already split into per-key raw
// messages, as received in tool call params) into engine metadata. Each
// value must be a flat scalar or a homogeneous array of scalars — the
// tagged union vector.Value has no representation for nested objects or
// arrays, so those are rejected with an error naming the offending key.
func jsonToMetadata(raw map[string]json.RawMessage) (rostam.VectorMetadata, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	m := make(rostam.VectorMetadata, len(raw))
	for key, v := range raw {
		val, err := jsonToValue(v)
		if err != nil {
			return nil, fmt.Errorf("metadata key %q: %w", key, err)
		}
		m[key] = val
	}
	return m, nil
}

// jsonToValue converts a single JSON value into a vector.Value.
func jsonToValue(raw json.RawMessage) (vector.Value, error) {
	trimmed := trimSpace(raw)
	if len(trimmed) == 0 {
		return vector.Value{}, fmt.Errorf("empty value")
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return vector.Value{}, err
		}
		return vector.NewString(s), nil
	case 't', 'f':
		var b bool
		if err := json.Unmarshal(trimmed, &b); err != nil {
			return vector.Value{}, err
		}
		return vector.NewBool(b), nil
	case '[':
		return jsonArrayToValue(trimmed)
	case 'n':
		return vector.Value{}, fmt.Errorf("null is not a supported metadata value")
	case '{':
		return vector.Value{}, fmt.Errorf("nested objects are not supported metadata values")
	default:
		return jsonNumberToValue(trimmed)
	}
}

// jsonNumberToValue decodes a bare JSON number, preferring an exact int64
// and falling back to float64 when the literal has a fraction/exponent or
// overflows int64 range.
func jsonNumberToValue(raw json.RawMessage) (vector.Value, error) {
	var n json.Number
	if err := json.Unmarshal(raw, &n); err != nil {
		return vector.Value{}, fmt.Errorf("not a valid number: %w", err)
	}
	if i, err := n.Int64(); err == nil {
		return vector.NewInt(i), nil
	}
	f, err := n.Float64()
	if err != nil {
		return vector.Value{}, fmt.Errorf("not a valid number: %w", err)
	}
	return vector.NewFloat(f), nil
}

// jsonArrayToValue converts a JSON array into one of the vector.Value array
// kinds. The array must be homogeneous: all strings, all integers, or all
// numbers with at least one float (which widens the whole array to floats).
// Nested arrays/objects/nulls/mixed string+number arrays are rejected.
func jsonArrayToValue(raw json.RawMessage) (vector.Value, error) {
	var elems []json.RawMessage
	if err := json.Unmarshal(raw, &elems); err != nil {
		return vector.Value{}, fmt.Errorf("not a valid array: %w", err)
	}
	if len(elems) == 0 {
		return vector.Value{}, fmt.Errorf("empty arrays are not supported metadata values")
	}

	first := trimSpace(elems[0])
	if len(first) == 0 {
		return vector.Value{}, fmt.Errorf("empty array element")
	}

	switch first[0] {
	case '"':
		strs := make([]string, len(elems))
		for i, e := range elems {
			t := trimSpace(e)
			if len(t) == 0 || t[0] != '"' {
				return vector.Value{}, fmt.Errorf("array element %d: mixed types are not supported", i)
			}
			if err := json.Unmarshal(e, &strs[i]); err != nil {
				return vector.Value{}, fmt.Errorf("array element %d: %w", i, err)
			}
		}
		return vector.NewStrings(strs), nil
	case '[', '{':
		return vector.Value{}, fmt.Errorf("nested arrays/objects are not supported metadata values")
	case 't', 'f':
		return vector.Value{}, fmt.Errorf("arrays of booleans are not supported metadata values")
	case 'n':
		return vector.Value{}, fmt.Errorf("null is not a supported metadata value")
	default:
		return jsonNumberArrayToValue(elems)
	}
}

// jsonNumberArrayToValue converts a homogeneous array of JSON numbers. If
// every element parses as an int64, the result is ValueInts; a single float
// (fraction/exponent, or int64 overflow) widens the whole array to
// ValueFloats, matching the brief's "mixed int/float -> NewFloats" rule.
func jsonNumberArrayToValue(elems []json.RawMessage) (vector.Value, error) {
	ints := make([]int64, len(elems))
	isFloat := false
	for i, e := range elems {
		t := trimSpace(e)
		if len(t) == 0 {
			return vector.Value{}, fmt.Errorf("array element %d: empty", i)
		}
		switch t[0] {
		case '[', '{', '"', 't', 'f', 'n':
			return vector.Value{}, fmt.Errorf("array element %d: mixed types are not supported", i)
		}
		var n json.Number
		if err := json.Unmarshal(t, &n); err != nil {
			return vector.Value{}, fmt.Errorf("array element %d: not a valid number: %w", i, err)
		}
		v, err := n.Int64()
		if err != nil {
			isFloat = true
			continue
		}
		ints[i] = v
	}
	if !isFloat {
		return vector.NewInts(ints), nil
	}

	floats := make([]float64, len(elems))
	for i, e := range elems {
		var n json.Number
		if err := json.Unmarshal(trimSpace(e), &n); err != nil {
			return vector.Value{}, fmt.Errorf("array element %d: not a valid number: %w", i, err)
		}
		f, err := n.Float64()
		if err != nil {
			return vector.Value{}, fmt.Errorf("array element %d: not a valid number: %w", i, err)
		}
		floats[i] = f
	}
	return vector.NewFloats(floats), nil
}

// trimSpace strips leading/trailing JSON whitespace (space, tab, CR, LF) so
// callers can inspect the first byte to classify a raw value without a full
// parse.
func trimSpace(raw json.RawMessage) json.RawMessage {
	i, j := 0, len(raw)
	for i < j && isJSONSpace(raw[i]) {
		i++
	}
	for j > i && isJSONSpace(raw[j-1]) {
		j--
	}
	return raw[i:j]
}

func isJSONSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\r' || b == '\n'
}

// metadataToJSON converts engine metadata to plain JSON-friendly values for
// a tool result. $content is the engine's reserved content field (set via
// VectorUpsert's content arg) and is always stripped from user-visible
// metadata.
func metadataToJSON(m rostam.VectorMetadata) map[string]any {
	out := make(map[string]any, len(m))
	for key, v := range m {
		if key == "$content" {
			continue
		}
		switch v.Kind {
		case vector.ValueString:
			out[key] = v.Str
		case vector.ValueInt:
			out[key] = v.Int
		case vector.ValueFloat:
			out[key] = v.Flt
		case vector.ValueBool:
			out[key] = v.Bool
		case vector.ValueStrings:
			out[key] = v.Strs
		case vector.ValueInts:
			out[key] = v.Ints
		case vector.ValueFloats:
			out[key] = v.Flts
		case vector.ValueGeo:
			out[key] = map[string]float64{"lat": v.Lat, "lon": v.Lon}
		case vector.ValueRecord:
			// Pass-through, not a new MCP feature: v.Rec is raw bytes, which
			// encoding/json already renders as base64 for any []byte value — the
			// same representation vtypes.Value's own JSON tag produces. Without
			// this case the key would silently vanish from tool output instead.
			out[key] = v.Rec
		}
	}
	return out
}

// parseFilter decodes a tool call's optional filter argument into an engine
// filter. An absent or empty raw message yields the zero filter (match
// all); otherwise raw is unmarshaled directly into rostam.VectorFilter,
// which is JSON-round-trippable including its op names ("eq", "and", ...).
// Leaf values use vector.Value's tagged form (e.g. {"kind":"int","int":10})
// since Value has no custom UnmarshalJSON for bare scalars.
func parseFilter(raw json.RawMessage) (rostam.VectorFilter, error) {
	if len(trimSpace(raw)) == 0 {
		return rostam.VectorFilter{}, nil
	}
	var f rostam.VectorFilter
	if err := json.Unmarshal(raw, &f); err != nil {
		return rostam.VectorFilter{}, fmt.Errorf("invalid filter: %w", err)
	}
	return f, nil
}
