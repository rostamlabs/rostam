// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"unicode/utf8"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// The op names this file dispatches. They are the CLUSTER's spellings
// (cluster/kv_index_admin.go, cluster/kv_query_broadcast.go) — one name per op
// end to end, the same three the native client repeats — restated here because
// this package cannot import cluster. The internal per-group legs
// (__kv_query_shard__, __kv_index_ready__) are deliberately absent: they are
// peer-to-peer, admin-scoped, and a REST caller that could name one would be
// asking a single shard group for its slice of an answer.
const (
	opKVIndexSet  = "__kv_index_set__"
	opKVIndexList = "__kv_index_list__"
	opKVQuery     = "kv_query"
)

// --- POST /v1/kv/query ----------------------------------------------------

// kvQueryReq is the POST /v1/kv/query body.
//
// Limit is a POINTER because JSON cannot otherwise tell "limit": 0 from an
// absent field, and the two must differ: the wire codec refuses limit 0 outright
// (there is no unbounded kv_query — a page is what bounds a fan-out read's
// memory on every node it touches), so an absent limit has to become a default
// while an explicit 0 stays the client mistake it is.
//
// Return and Consistency are STRINGS rather than the wire's numbers. A REST
// caller writing "records" is telling the reader of its code what it asked for;
// a caller writing 2 is not, and a caller writing 3 by mistake would be asking
// for a projection that does not exist in a field the codec would then reject
// with a number.
//
// FILTER IS RAW, and for the same reason the cursor is a string: it must be
// SIZED BEFORE IT IS EXPANDED. Decoded straight into a vtypes.Filter, the body
// cap (maxJSONBody, 32 MiB) is the only bound on the work, and 32 MiB of
// `{"and":[{"and":[...` expands into a tree of allocated nodes costing orders of
// magnitude more heap than the bytes that asked for it — paid before the filter
// budget runs and before the request is authorized, so anonymously. Held as raw
// bytes the decoder only scans past the value, and the byte cap is checked on
// what the caller actually sent. See toWire.
type kvQueryReq struct {
	Index       string          `json:"index"`
	Filter      json.RawMessage `json:"filter"`
	Limit       *int            `json:"limit"`
	Return      string          `json:"return"`
	Consistency string          `json:"consistency"`
	Scan        bool            `json:"scan"`
	Cursor      string          `json:"cursor"`
}

// kvQueryRow is one row of the answer. KeyB64 is always present, with KeyUTF8
// beside it only when the key really is text, so a client never mistakes lossy
// bytes for a string.
//
// ValueB64 carries the raw value bytes when the query asked for a value AND that
// value fit the page. A row whose value was omitted for size comes back
// key-only, which is the leaf's documented convention and NOT an error — fetch
// those keys with GET /v1/kv/{key}.
//
// THERE IS NO DECODED-RECORD FIELD, and its absence is a decision. return:
// "records" means the SERVER validated that each value decodes as a record and
// skipped the ones that do not; the bytes still arrive in value_b64 and decoding
// them is the caller's job (the same split the native client makes — it is the
// client that calls wire.DecodeRecord). Rendering the record here would mean
// committing this REST surface to a record-to-JSON shape, and the only one
// available today is Go's default marshalling of wire.Record, whose field names
// are internal and carry no compatibility promise. A deliberate rendering is
// worth having; inheriting one by accident is not.
type kvQueryRow struct {
	KeyB64    string  `json:"key_b64"`
	KeyUTF8   *string `json:"key_utf8,omitempty"`
	ValueB64  *string `json:"value_b64,omitempty"`
	ValueUTF8 *string `json:"value_utf8,omitempty"`
}

// kvQueryResponse is the POST /v1/kv/query body. Cursor is an OPAQUE base64
// blob: a client echoes it back verbatim in the next request's "cursor" and
// never parses it. An empty cursor means the result set is exhausted — which is
// the ONLY termination signal, so it is always present as a field (empty string)
// rather than omitted.
type kvQueryResponse struct {
	Rows   []kvQueryRow `json:"rows"`
	Cursor string       `json:"cursor"`
}

// kvQueryReturns maps the REST projection names onto the wire bytes. A name
// outside this table is a 400 rather than a silent fall back to keys: a caller
// who asked for "record" (singular) and got keys would read an empty page as
// "no values stored".
var kvQueryReturns = map[string]uint8{
	"keys":    wire.KVQueryReturnKeys,
	"values":  wire.KVQueryReturnValues,
	"records": wire.KVQueryReturnRecords,
}

// kvQueryConsistencies maps the REST read-consistency names onto the wire bytes.
// The default (absent) is LEADER-ONLY, not any-replica: a kv_query's whole
// contract is that the page is a complete answer over the whole keyspace, and a
// stale replica silently omitting rows is the one failure this feature may not
// have. A caller that wants the cheaper read asks for it by name.
var kvQueryConsistencies = map[string]uint8{
	"any":          wire.ConsistencyAnyReplica,
	"leader":       wire.ConsistencyLeaderOnly,
	"linearizable": wire.ConsistencyLinearizable,
}

// kvQueryDefaultLimit is the page size an absent "limit" means. It matches the
// native client's KVQueryDefaultLimit so the two front doors page identically.
const kvQueryDefaultLimit = 100

// kvQuery answers POST /v1/kv/query: one page of a filtered read over the KV
// keyspace, narrowed by a named index or (with explicit consent) by a scan.
//
// It goes through the plain READ path — a.call, not callWrite — because kv_query
// is registered OpReadOnly and the authorizer classifies it as a read from the
// registry. That is deliberate and is the difference from the index endpoints
// below: reading rows is a read, defining an index is cluster-wide schema.
//
// Everything the wire codec would refuse is refused HERE, before dispatch, so a
// malformed request costs no cluster fan-out and the caller gets a message about
// the field it got wrong rather than a codec's view of a byte offset.
func (a *api) kvQuery(w http.ResponseWriter, r *http.Request) {
	var req kvQueryReq
	if !decodeBody(w, r, &req) {
		return
	}
	args, ok := req.toWire(w)
	if !ok {
		return
	}
	encoded, err := wire.EncodeKVQueryArgs(args)
	if err != nil {
		// The codec's own refusals (an index name outside the charset, a filter
		// over its byte cap, a cursor whose groups are not increasing) are client
		// mistakes, and its messages name the field and the cap.
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	body, ok := a.call(w, r, opKVQuery, encoded)
	if !ok {
		return
	}
	res, derr := wire.DecodeKVQueryResult(body)
	if derr != nil {
		writeInternalError(w, opKVQuery+" decode", derr)
		return
	}
	resp, rerr := kvQueryResponseFrom(res)
	if rerr != nil {
		writeInternalError(w, opKVQuery+" render", rerr)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// toWire validates the request and converts it into wire args, writing the 400
// and returning ok=false on any client mistake.
func (req *kvQueryReq) toWire(w http.ResponseWriter) (wire.KVQueryArgs, bool) {
	limit := kvQueryDefaultLimit
	if req.Limit != nil {
		limit = *req.Limit
	}
	if limit < 1 || limit > wire.KVQueryMaxLimit {
		writeError(w, http.StatusBadRequest, "limit must be between 1 and "+strconv.Itoa(wire.KVQueryMaxLimit))
		return wire.KVQueryArgs{}, false
	}
	ret, ok := kvQueryReturns[orDefault(req.Return, "keys")]
	if !ok {
		writeError(w, http.StatusBadRequest, `return must be "keys", "values" or "records"`)
		return wire.KVQueryArgs{}, false
	}
	rc, ok := kvQueryConsistencies[orDefault(req.Consistency, "leader")]
	if !ok {
		writeError(w, http.StatusBadRequest, `consistency must be "any", "leader" or "linearizable"`)
		return wire.KVQueryArgs{}, false
	}
	// The filter is SIZED, then expanded, then budgeted — in that order, and the
	// order is the point, exactly as it is for the cursor below.
	//
	// The cap is compared against the RAW bytes the caller sent. That is
	// marginally stricter than the codec's cap, which applies to the compacted
	// JSON EncodeKVQueryArgs produces, so a filter padded with tens of kilobytes
	// of whitespace can be refused here while its compact form would have fit.
	// That is the right side to err on: the alternative is to unmarshal first to
	// find out how big it really is, which is the allocation this check exists to
	// prevent, and the remedy (send compact JSON) is under the caller's control.
	//
	// Unmarshalled with DisallowUnknownFields, matching decodeBody: holding the
	// filter raw must not quietly turn a misspelled operator into a silently
	// ignored one.
	var filter vtypes.Filter
	if len(req.Filter) > 0 && !bytes.Equal(bytes.TrimSpace(req.Filter), []byte("null")) {
		if len(req.Filter) > wire.KVQueryMaxFilterBytes {
			writeError(w, http.StatusBadRequest,
				"filter is too large: "+strconv.Itoa(len(req.Filter))+" bytes exceeds the cap of "+strconv.Itoa(wire.KVQueryMaxFilterBytes))
			return wire.KVQueryArgs{}, false
		}
		dec := json.NewDecoder(bytes.NewReader(req.Filter))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&filter); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
			return wire.KVQueryArgs{}, false
		}
		// The node/depth budget, run here rather than left to the codec so the
		// message names the filter and no fan-out is dispatched.
		if err := wire.CheckFilterBudget(filter, wire.KVQueryMaxFilterNodes, wire.KVQueryMaxFilterDepth); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return wire.KVQueryArgs{}, false
		}
	}
	// The cursor is opaque to the caller, so a malformed one is a client mistake
	// with an obvious remedy (send back what the last page returned, unedited) —
	// never a 500. Three checks, in this order, and the ORDER IS THE POINT.
	//
	// The length is checked FIRST, on the ENCODED string. DecodeKVQueryCursor
	// enforces the 4 MiB cap, but only on bytes it has already been handed, and
	// base64.DecodeString allocates the whole decoded blob before returning — so
	// checking the cap after the decode means a caller can make this process
	// materialize an arbitrarily large buffer (bounded only by maxJSONBody, 32
	// MiB) for a cursor that is then rejected for being too big. Comparing
	// against EncodedLen(cap) refuses it before a byte is allocated.
	var conts []wire.KVQueryCont
	if req.Cursor != "" {
		if maxEncoded := base64.StdEncoding.EncodedLen(wire.KVQueryMaxCursorBytes); len(req.Cursor) > maxEncoded {
			writeError(w, http.StatusBadRequest,
				"cursor is too large: "+strconv.Itoa(len(req.Cursor))+" encoded bytes exceeds the cap of "+strconv.Itoa(maxEncoded))
			return wire.KVQueryArgs{}, false
		}
		blob, berr := base64.StdEncoding.DecodeString(req.Cursor)
		if berr != nil {
			writeError(w, http.StatusBadRequest, "cursor is not valid base64: "+berr.Error())
			return wire.KVQueryArgs{}, false
		}
		conts, berr = wire.DecodeKVQueryCursor(blob)
		if berr != nil {
			writeError(w, http.StatusBadRequest, "cursor is malformed: "+berr.Error())
			return wire.KVQueryArgs{}, false
		}
	}
	return wire.KVQueryArgs{
		Index:       req.Index,
		Filter:      filter,
		Limit:       uint16(limit), //nolint:gosec // bounded by KVQueryMaxLimit above
		Return:      ret,
		Consistency: rc,
		Scan:        req.Scan,
		Cursor:      conts,
	}, true
}

// kvQueryResponseFrom renders a decoded result as the REST body.
//
// Rows is always a non-nil slice so an exhausted query marshals as "rows":[]
// rather than "rows":null — a client iterating the field should not have to
// special-case an empty page.
func kvQueryResponseFrom(res wire.KVQueryResult) (kvQueryResponse, error) {
	resp := kvQueryResponse{Rows: make([]kvQueryRow, 0, len(res.Rows))}
	for _, row := range res.Rows {
		out := kvQueryRow{KeyB64: base64.StdEncoding.EncodeToString(row.Key)}
		if utf8.Valid(row.Key) {
			s := string(row.Key)
			out.KeyUTF8 = &s
		}
		if row.Value != nil {
			b64 := base64.StdEncoding.EncodeToString(row.Value)
			out.ValueB64 = &b64
			if utf8.Valid(row.Value) {
				s := string(row.Value)
				out.ValueUTF8 = &s
			}
		}
		resp.Rows = append(resp.Rows, out)
	}
	blob, err := wire.EncodeKVQueryCursor(res.Cursor)
	if err != nil {
		return kvQueryResponse{}, err
	}
	resp.Cursor = base64.StdEncoding.EncodeToString(blob)
	return resp, nil
}

// --- /v1/kv/indexes -------------------------------------------------------

// kvIndexReq is the POST /v1/kv/indexes body. KeyPrefixB64 is base64 because a
// key prefix is arbitrary bytes; an absent prefix indexes the whole keyspace.
// Kind is a string for the same reason kvQueryReq.Return is.
type kvIndexReq struct {
	Name         string `json:"name"`
	KeyPrefixB64 string `json:"key_prefix_b64"`
	PayloadPath  string `json:"payload_path"`
	Kind         string `json:"kind"`
}

// kvIndexInfo is one definition in the GET /v1/kv/indexes answer.
//
// Ready holds ONLY when every shard group reports the index ready, and it fails
// closed — a group still backfilling, one that never installed the definition,
// and one that could not be reached all make it false. It is the field a caller
// polls after a create, and the definition is never dropped from the list
// because nothing could be said about its progress: a name vanishing from the
// list reads as "the create failed".
type kvIndexInfo struct {
	Name         string `json:"name"`
	KeyPrefixB64 string `json:"key_prefix_b64"`
	PayloadPath  string `json:"payload_path"`
	Kind         string `json:"kind"`
	Ready        bool   `json:"ready"`
}

// kvIndexListResponse is the GET /v1/kv/indexes body.
//
// There is no rejected_defs field here, and that is a decision rather than an
// omission: __kv_index_list__ returns definitions and readiness bits only, and
// the RejectedDefs gauge (definitions the meta FSM accepted that a node cannot
// parse right now) is per-NODE derived state that already rides the stats
// surface as Stats.KVIndex.RejectedDefs. Duplicating it into a cluster-wide
// catalog listing would mean inventing a cluster-wide aggregate the list op does
// not gather.
type kvIndexListResponse struct {
	Indexes []kvIndexInfo `json:"indexes"`
}

// kvIndexKinds maps the REST kind names onto the wire bytes, and back.
var (
	kvIndexKinds = map[string]uint8{
		"scalar": wire.KVIndexKindScalar,
		"count":  wire.KVIndexKindCount,
	}
	kvIndexKindNames = map[uint8]string{
		wire.KVIndexKindScalar: "scalar",
		wire.KVIndexKindCount:  "count",
	}
)

// kvIndexCreate answers POST /v1/kv/indexes: define (or redefine) one KV record
// index cluster-wide.
//
// It is dispatched through callWrite, like kvFlush, so it takes the write path's
// shape (and its optional write-consistency envelope) rather than the read
// path's. The AUTHORIZATION BAR IS HIGHER THAN THAT PATH SUGGESTS, and
// deliberately so: authz.actionFor consults adminOps FIRST and __kv_index_set__
// is enumerated there, so this endpoint requires an ADMIN key — not merely the
// global-write bar flush settles for. Dropping an index makes every query naming
// it start failing at once and re-creating it costs a full cache walk on every
// node, which is schema-shaped rather than data-shaped. Pinned by
// authz.TestActionForKVIndexOpsIsAdmin.
//
// The definition is validated HERE, before dispatch. wire.AppendKVIndexDef is
// the trusting side of the codec — it writes u8 length prefixes without checking
// them — so an over-long name would otherwise be silently truncated onto the
// meta log.
func (a *api) kvIndexCreate(w http.ResponseWriter, r *http.Request) {
	var req kvIndexReq
	if !decodeBody(w, r, &req) {
		return
	}
	prefix, ok := kvIndexPrefix(w, req.KeyPrefixB64)
	if !ok {
		return
	}
	kind, ok := kvIndexKinds[orDefault(req.Kind, "scalar")]
	if !ok {
		writeError(w, http.StatusBadRequest, `kind must be "scalar" or "count"`)
		return
	}
	def := wire.KVIndexDef{
		Name: req.Name, KeyPrefix: prefix, PayloadPath: req.PayloadPath,
		Kind: kind, Enabled: true,
	}
	if err := def.Validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if _, ok := a.callWrite(w, r, opKVIndexSet, wire.EncodeKVIndexSetArgs(def), 0, true); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// kvIndexDrop answers DELETE /v1/kv/indexes/{name}: remove one index
// cluster-wide.
//
// A drop is the SAME meta-log write as a create with Enabled=false — there is no
// separate delete op — which is why it must still build a definition the codec
// accepts. PayloadPath carries a placeholder: the FSM keys the entry on Name
// alone and deletes on Enabled=false, so nothing reads it, but Validate requires
// a non-empty path that agrees with the kind.
//
// Dropping is not undoable in any cheap sense: every query naming the index
// starts failing at once, and re-creating it costs a full cache walk on every
// node. That is why it sits behind the same bar as create — and that bar is
// ADMIN, not the global-write one callWrite's name suggests: both endpoints
// dispatch __kv_index_set__, and authz.actionFor consults adminOps first,
// where that op is enumerated. Pinned by
// authz.TestActionForKVIndexOpsIsAdmin, the same test kvIndexCreate cites.
func (a *api) kvIndexDrop(w http.ResponseWriter, r *http.Request) {
	def := wire.KVIndexDef{
		Name: r.PathValue("name"), PayloadPath: "-",
		Kind: wire.KVIndexKindScalar, Enabled: false,
	}
	if err := def.Validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if _, ok := a.callWrite(w, r, opKVIndexSet, wire.EncodeKVIndexSetArgs(def), 0, true); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// kvIndexList answers GET /v1/kv/indexes with the catalog and its readiness
// bits. It goes through the plain read path: listing the catalog is a READ, and
// authz classifies __kv_index_list__ in its read set for exactly that reason.
func (a *api) kvIndexList(w http.ResponseWriter, r *http.Request) {
	body, ok := a.call(w, r, opKVIndexList, nil)
	if !ok {
		return
	}
	resp := kvIndexListResponse{Indexes: []kvIndexInfo{}}
	if len(body) == 0 {
		// An empty catalog is an empty payload, not an error.
		writeJSON(w, http.StatusOK, resp)
		return
	}
	defs, ready, err := wire.DecodeKVIndexList(body)
	if err != nil {
		writeInternalError(w, opKVIndexList+" decode", err)
		return
	}
	for i, d := range defs {
		resp.Indexes = append(resp.Indexes, kvIndexInfo{
			Name:         d.Name,
			KeyPrefixB64: base64.StdEncoding.EncodeToString(d.KeyPrefix),
			PayloadPath:  d.PayloadPath,
			Kind:         kvIndexKindNames[d.Kind],
			Ready:        i < len(ready) && ready[i],
		})
	}
	writeJSON(w, http.StatusOK, resp)
}

// kvIndexPrefix decodes the base64 key prefix, writing the 400 on failure. An
// absent prefix is a nil one (index the whole keyspace), not an error.
//
// THE LENGTH IS CHECKED BEFORE THE DECODE. A prefix over
// wire.KVIndexMaxPrefixLen is refused by the definition's own Validate anyway,
// so decoding one first only buys an allocation sized by the request body. The
// encoded form of the largest legal prefix is base64.EncodedLen of the cap;
// anything longer cannot decode to a legal prefix whatever its contents.
func kvIndexPrefix(w http.ResponseWriter, b64 string) ([]byte, bool) {
	if b64 == "" {
		return nil, true
	}
	if len(b64) > base64.StdEncoding.EncodedLen(wire.KVIndexMaxPrefixLen) {
		writeError(w, http.StatusBadRequest, "key_prefix_b64 decodes to more than the "+strconv.Itoa(wire.KVIndexMaxPrefixLen)+"-byte key prefix cap")
		return nil, false
	}
	prefix, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "key_prefix_b64 is not valid base64: "+err.Error())
		return nil, false
	}
	return prefix, true
}

// orDefault returns s, or def when s is empty. JSON's zero value for a string
// field is "", which for every enum-ish field here means "the caller did not
// choose" rather than "the caller chose the empty name".
func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
