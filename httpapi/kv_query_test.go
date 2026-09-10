// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

// --- fixtures -------------------------------------------------------------

// kvIndexTestField is the record field every fixture in this file indexes.
const kvIndexTestField = "age"

// kvQueryRecord builds the dynamic-mode record a seeded key holds.
func kvQueryRecord(t *testing.T, v int64) []byte {
	t.Helper()
	rec := &wire.Record{
		Mode:   wire.OperateModeDynamic,
		Fields: []wire.Field{{Name: kvIndexTestField, Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}},
	}
	return rec.Encode()
}

// kvIndexDispatcher is a testDispatcher that also answers the two CLUSTER admin
// ops the index endpoints send. They are not in the ops registry (they are
// intercepted by cluster.Node before routing), so a registry-only dispatcher
// answers "op not registered" and the endpoints could not be exercised at all.
// The catalog it keeps is deliberately dumb: this file is testing the HTTP
// surface, not the meta log.
type kvIndexDispatcher struct {
	*testDispatcher
	defs  []wire.KVIndexDef
	ready []bool
}

func (d *kvIndexDispatcher) Call(name string, args []byte) ([]byte, error) {
	switch name {
	case opKVIndexSet:
		def, err := wire.DecodeKVIndexSetArgs(args)
		if err != nil {
			return nil, err
		}
		for i := range d.defs {
			if d.defs[i].Name != def.Name {
				continue
			}
			if !def.Enabled {
				d.defs = append(d.defs[:i], d.defs[i+1:]...)
				d.ready = append(d.ready[:i], d.ready[i+1:]...)
				return nil, nil
			}
			d.defs[i] = def
			return nil, nil
		}
		if !def.Enabled {
			return nil, nil // dropping an unknown index is a no-op, as in the FSM
		}
		d.defs = append(d.defs, def)
		d.ready = append(d.ready, true)
		return nil, nil
	case opKVIndexList:
		if len(args) != 0 {
			return nil, fmt.Errorf("%s takes no arguments", opKVIndexList)
		}
		if len(d.defs) == 0 {
			return nil, nil
		}
		return wire.EncodeKVIndexList(d.defs, d.ready), nil
	}
	return d.testDispatcher.Call(name, args)
}

// newKVQueryTestAPI builds a handler over a dispatcher with a REAL KV record
// index (the default newTestAPI has none, so every kv_query there would answer
// ErrKVIndexUnavailable), seeds n records under "u:", and installs+backfills one
// scalar definition over kvIndexTestField so indexed queries are ready.
func newKVQueryTestAPI(t *testing.T, n int) (http.Handler, *kvIndexDispatcher, func()) {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	vstore, verr := vector.OpenCollectionStore(t.TempDir())
	if verr != nil {
		t.Fatal(verr)
	}
	idx := ops.NewKVIndexFor(c)
	def, derr := kvindex.DefFrom(wire.KVIndexDef{
		Name: "by_age", KeyPrefix: []byte("u:"), PayloadPath: kvIndexTestField,
		Kind: wire.KVIndexKindScalar, Enabled: true,
	}, 1)
	if derr != nil {
		t.Fatal(derr)
	}
	idx.Install([]kvindex.Def{def})
	for i := 0; i < n; i++ {
		key := fmt.Appendf(nil, "u:%03d", i)
		if perr := c.Put(key, kvQueryRecord(t, int64(i)), 0); perr != nil {
			t.Fatal(perr)
		}
	}
	// Backfill AFTER the seed so the definition is ready over the whole set.
	if berr := idx.Backfill("by_age", ops.CacheWalker(c)); berr != nil {
		t.Fatal(berr)
	}
	disp := &kvIndexDispatcher{testDispatcher: &testDispatcher{reg: reg, tx: ops.NewTxContextWithIndex(c, vstore, idx)}}
	return Handler(disp, Options{}), disp, func() { _ = vstore.Close(); c.Close() }
}

// --- POST /v1/kv/query ----------------------------------------------------

// TestHTTPKVQuery pages an indexed query to exhaustion over REST: the rows come
// back base64 (key_b64, with key_utf8 beside it when the key is text) and the
// cursor is an opaque blob the client echoes back verbatim.
//
// It asserts NO decoded record, because no projection carries one. `records`
// means the SERVER validated that each value decodes as a record; the bytes
// still arrive in value_b64 (and value_utf8 when they are valid UTF-8) for the
// caller to decode — see kvQueryRow for why this surface deliberately renders
// no record JSON. TestHTTPKVQueryProjections covers the three projections.
func TestHTTPKVQuery(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 6)
	defer cleanup()

	body := `{"index":"by_age","limit":4,"filter":{"op":"gte","field":"age","value":{"kind":"int","int":0}}}`
	var out kvQueryResponse
	rec := do(t, h, "POST", "/v1/kv/query", body, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(out.Rows) != 4 {
		t.Fatalf("rows = %d, want the 4 the limit asked for (%s)", len(out.Rows), rec.Body)
	}
	if out.Cursor == "" {
		t.Fatal("a page cut by the limit must return a cursor")
	}
	got := map[string]bool{}
	for _, row := range out.Rows {
		key, derr := base64.StdEncoding.DecodeString(row.KeyB64)
		if derr != nil {
			t.Fatalf("key_b64 %q: %v", row.KeyB64, derr)
		}
		got[string(key)] = true
	}

	// Page two, resuming from the opaque cursor exactly as returned.
	var out2 kvQueryResponse
	body2 := fmt.Sprintf(`{"index":"by_age","limit":4,"cursor":%q,"filter":{"op":"gte","field":"age","value":{"kind":"int","int":0}}}`, out.Cursor)
	rec = do(t, h, "POST", "/v1/kv/query", body2, &out2)
	if rec.Code != http.StatusOK {
		t.Fatalf("page two = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	for _, row := range out2.Rows {
		key, _ := base64.StdEncoding.DecodeString(row.KeyB64)
		got[string(key)] = true
	}
	if len(got) != 6 {
		t.Fatalf("paged %d distinct keys, want all 6: %v", len(got), got)
	}
}

// TestHTTPKVQueryProjections covers all three: keys carries no value at all,
// values carries the raw bytes base64, and records carries the same bytes — the
// difference being that the SERVER has validated each one decodes as a record.
// Decoding stays the caller's job at this transport too, so there is no
// decoded-record field to assert on; see kvQueryRow for why.
func TestHTTPKVQueryProjections(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 2)
	defer cleanup()

	const filter = `"filter":{"op":"eq","field":"age","value":{"kind":"int","int":1}}`
	for _, tc := range []struct {
		name     string
		ret      string
		wantVal  bool
		wantUTF8 bool
	}{
		// value_utf8 rides along whenever the bytes pass utf8.Valid, exactly as
		// GET /v1/kv/{key} already behaves. A dynamic-mode record of small
		// integers happens to be all-ASCII, so it qualifies — accidentally, but
		// consistently with the rule the rest of the KV surface follows.
		{"keys", `"return":"keys",`, false, false},
		{"values", `"return":"values",`, true, true},
		{"records", `"return":"records",`, true, true},
		{"absent return defaults to keys", "", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out kvQueryResponse
			rec := do(t, h, "POST", "/v1/kv/query", `{"index":"by_age",`+tc.ret+filter+`}`, &out)
			if rec.Code != http.StatusOK || len(out.Rows) != 1 {
				t.Fatalf("status=%d rows=%d (%s)", rec.Code, len(out.Rows), rec.Body)
			}
			if got := out.Rows[0].ValueB64 != nil; got != tc.wantVal {
				t.Fatalf("value_b64 present = %v, want %v (%s)", got, tc.wantVal, rec.Body)
			}
			if got := out.Rows[0].ValueUTF8 != nil; got != tc.wantUTF8 {
				t.Fatalf("value_utf8 present = %v, want %v (%s)", got, tc.wantUTF8, rec.Body)
			}
			// The key IS text, so its UTF-8 twin must be there.
			if out.Rows[0].KeyUTF8 == nil || *out.Rows[0].KeyUTF8 != "u:001" {
				t.Fatalf("key_utf8 = %v, want u:001", out.Rows[0].KeyUTF8)
			}
		})
	}
}

// TestHTTPKVQueryBadRequests pins the 400 bucket: the edge rejects what the wire
// codec would reject anyway, but BEFORE dispatch, so a malformed request never
// costs a cluster fan-out.
func TestHTTPKVQueryBadRequests(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 1)
	defer cleanup()

	oversized := strings.Repeat("x", wire.KVQueryMaxFilterBytes+1)
	tests := []struct {
		name string
		body string
	}{
		{"limit below zero", `{"index":"by_age","limit":-1}`},
		{"limit over the cap", fmt.Sprintf(`{"index":"by_age","limit":%d}`, wire.KVQueryMaxLimit+1)},
		{"no index and no scan consent", `{}`},
		{"malformed cursor", `{"index":"by_age","cursor":"!!!not base64!!!"}`},
		{"cursor that is base64 but not a cursor", `{"index":"by_age","cursor":"AAAA"}`},
		{"unknown return projection", `{"index":"by_age","return":"everything"}`},
		{"oversized filter", fmt.Sprintf(`{"index":"by_age","filter":{"op":"eq","field":%q,"value":{"kind":"int","int":1}}}`, oversized)},
		{"index name outside the charset", `{"index":"not a name!"}`},
		{"unknown consistency", `{"index":"by_age","consistency":"eventually"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, "POST", "/v1/kv/query", tc.body, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
			}
		})
	}
}

// TestHTTPKVQueryLimitZeroIsTheDefault pins that an ABSENT limit is the REST
// default rather than an error: JSON has no way to tell "0" from "not set" on a
// plain int, and the codec refuses limit 0, so a body that simply omits the
// field would otherwise be a 400 for asking the obvious question.
func TestHTTPKVQueryLimitZeroIsTheDefault(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 3)
	defer cleanup()

	var out kvQueryResponse
	body := `{"index":"by_age","filter":{"op":"gte","field":"age","value":{"kind":"int","int":0}}}`
	rec := do(t, h, "POST", "/v1/kv/query", body, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(out.Rows) != 3 {
		t.Fatalf("rows = %d, want 3", len(out.Rows))
	}
}

// TestHTTPKVQueryUnknownIndexIs404 pins the missing-definition answer. It is the
// one kv_query refusal that is NOT a 400: the caller named a thing that does not
// exist, which is what 404 says, and it tells a client library to create the
// index rather than to fix its filter.
func TestHTTPKVQueryUnknownIndexIs404(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 1)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/kv/query", `{"index":"nope"}`, nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPKVQueryFamilyParity walks the ONE canonical family list and asserts
// this transport's spelling of every member's class.
//
// It replaces three hand-maintained per-transport lists that had already
// drifted: the Task 7 review found server classifying kvindex.ErrNoSuchIndex,
// kvindex.ErrCandidateBudget and wire.ErrKVQueryResult while this file omitted
// all three, with both classifiers' comments claiming they were in sync. Adding
// a refusal to ops.KVQueryErrorFamily now makes all three transports' tests
// cover it at once, and whichever transport missed it fails here.
func TestHTTPKVQueryFamilyParity(t *testing.T) {
	want := map[ops.KVQueryErrorClass]int{
		ops.KVQueryErrPermanent: http.StatusBadRequest,
		ops.KVQueryErrNotFound:  http.StatusNotFound,
		ops.KVQueryErrRetryable: http.StatusServiceUnavailable,
	}
	for _, spec := range ops.KVQueryErrorFamily() {
		t.Run(spec.Name, func(t *testing.T) {
			if got := statusForError(spec.Err); got != want[spec.Class] {
				t.Fatalf("%s (%s) = %d, want %d", spec.Name, spec.Class, got, want[spec.Class])
			}
			// And again through the wrappers the fan-out and the peer client put
			// in front of a refusal, which is how most of these actually arrive.
			wrapped := fmt.Errorf("cluster: kv_query: shard group 3: %w", spec.Err)
			if got := statusForError(wrapped); got != want[spec.Class] {
				t.Fatalf("wrapped %s (%s) = %d, want %d", spec.Name, spec.Class, got, want[spec.Class])
			}
		})
	}
}

// TestHTTPStoreClosedTextParity covers the family member this transport can only
// see as TEXT. httpapi cannot import shard, so ops.KVQueryErrorFamily carries the
// message form; these are the wrapper shapes it really arrives in.
func TestHTTPStoreClosedTextParity(t *testing.T) {
	for _, msg := range []string{
		ops.StoreClosedMsg,
		"cluster: kv_query: shard group 3: " + ops.StoreClosedMsg,
		`cluster: kv_query: shard group 3: client: server error on op "__kv_query_shard__": ` + ops.StoreClosedMsg,
	} {
		if got := statusForError(errors.New(msg)); got != http.StatusServiceUnavailable {
			t.Errorf("statusForError(%q) = %d, want 503", msg, got)
		}
	}
	// Negative controls: a fault that merely MENTIONS the refusal mid-message is
	// not the refusal, and must stay a redacted 500.
	if got := statusForError(errors.New(ops.StoreClosedMsg + " while writing /var/lib/rostam/wal-3")); got != http.StatusInternalServerError {
		t.Errorf("a fault mentioning the refusal mid-message = %d, want 500", got)
	}
}

// --- /v1/kv/indexes -------------------------------------------------------

// TestHTTPKVIndexCRUD walks create → list → drop → list over REST.
func TestHTTPKVIndexCRUD(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 0)
	defer cleanup()

	create := `{"name":"by_age","key_prefix_b64":"` + base64.StdEncoding.EncodeToString([]byte("u:")) + `","payload_path":"age","kind":"scalar"}`
	rec := do(t, h, "POST", "/v1/kv/indexes", create, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("create = %d, want 200 (%s)", rec.Code, rec.Body)
	}

	var list kvIndexListResponse
	rec = do(t, h, "GET", "/v1/kv/indexes", "", &list)
	if rec.Code != http.StatusOK {
		t.Fatalf("list = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(list.Indexes) != 1 {
		t.Fatalf("indexes = %d, want 1 (%s)", len(list.Indexes), rec.Body)
	}
	got := list.Indexes[0]
	if got.Name != "by_age" || got.PayloadPath != "age" || got.Kind != "scalar" || !got.Ready {
		t.Fatalf("listed index = %+v, want the created one, ready", got)
	}
	if got.KeyPrefixB64 != base64.StdEncoding.EncodeToString([]byte("u:")) {
		t.Fatalf("key_prefix_b64 = %q, want the created prefix", got.KeyPrefixB64)
	}

	rec = do(t, h, "DELETE", "/v1/kv/indexes/by_age", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("drop = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	var after kvIndexListResponse
	rec = do(t, h, "GET", "/v1/kv/indexes", "", &after)
	if rec.Code != http.StatusOK {
		t.Fatalf("list after drop = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(after.Indexes) != 0 {
		t.Fatalf("after drop: %d indexes, want none (%s)", len(after.Indexes), rec.Body)
	}
	// An empty catalog must be an empty ARRAY, not JSON null: a client iterating
	// the field should not have to special-case "no indexes yet".
	if !strings.Contains(rec.Body.String(), `"indexes":[]`) {
		t.Fatalf("empty catalog rendered as %s, want an empty array", rec.Body)
	}
}

// TestHTTPKVIndexCreateBadRequests pins that a definition the wire codec cannot
// represent is refused at the edge, with no dispatch — AppendKVIndexDef is the
// trusting side of the codec and would silently truncate an over-long name.
func TestHTTPKVIndexCreateBadRequests(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 0)
	defer cleanup()

	for _, tc := range []struct{ name, body string }{
		{"illegal name", `{"name":"not a name!","payload_path":"age","kind":"scalar"}`},
		{"empty path", `{"name":"by_age","payload_path":"","kind":"scalar"}`},
		{"row path", `{"name":"by_age","payload_path":"b/42/c","kind":"scalar"}`},
		{"kind disagrees with the path", `{"name":"by_age","payload_path":"tags#count","kind":"scalar"}`},
		{"unknown kind", `{"name":"by_age","payload_path":"age","kind":"vector"}`},
		{"key prefix not base64", `{"name":"by_age","key_prefix_b64":"!!!","payload_path":"age","kind":"scalar"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, "POST", "/v1/kv/indexes", tc.body, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
			}
		})
	}
}

// TestHTTPKVIndexDropBadName pins that a name the codec cannot carry is a 400 at
// the edge rather than a dispatch of a truncated definition.
func TestHTTPKVIndexDropBadName(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 0)
	defer cleanup()

	rec := do(t, h, "DELETE", "/v1/kv/indexes/"+strings.Repeat("x", wire.KVIndexMaxNameLen+1), "", nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPKVIndexWriteIsAuthorizedAsAGlobalWrite pins the authz classification
// the index endpoints inherit from callWrite: they are dispatched under the op
// name the authorizer classifies, so a READ-scoped key cannot create or drop an
// index while it CAN list them (the catalog is a read).
func TestHTTPKVIndexWriteIsAuthorizedAsAGlobalWrite(t *testing.T) {
	var seen []string
	h, _, cleanup := newKVQueryTestAPIAuth(t, func(op string) bool {
		seen = append(seen, op)
		return op != opKVIndexSet
	})
	defer cleanup()

	rec := do(t, h, "POST", "/v1/kv/indexes", `{"name":"by_age","payload_path":"age","kind":"scalar"}`, nil)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("create with a key that lacks the scope = %d, want 401 (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "DELETE", "/v1/kv/indexes/by_age", "", nil)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("drop with a key that lacks the scope = %d, want 401 (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "GET", "/v1/kv/indexes", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("list = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	// Each endpoint must authorize under ITS OWN op name, never under a generic
	// one — that is what lets an operator scope a key to reads of the catalog.
	want := []string{opKVIndexSet, opKVIndexSet, opKVIndexList}
	if strings.Join(seen, ",") != strings.Join(want, ",") {
		t.Fatalf("authorized ops = %v, want %v", seen, want)
	}
}

// TestHTTPKVQueryIsAuthorizedAsARead pins that the query endpoint authorizes
// under the plain kv_query op name (an OpReadOnly the authorizer classifies as a
// read), not under the internal per-group wrapper.
func TestHTTPKVQueryIsAuthorizedAsARead(t *testing.T) {
	var seen []string
	h, _, cleanup := newKVQueryTestAPIAuth(t, func(op string) bool {
		seen = append(seen, op)
		return true
	})
	defer cleanup()

	body := `{"index":"by_age","filter":{"op":"gte","field":"age","value":{"kind":"int","int":0}}}`
	if rec := do(t, h, "POST", "/v1/kv/query", body, nil); rec.Code != http.StatusOK {
		t.Fatalf("query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(seen) != 1 || seen[0] != "kv_query" {
		t.Fatalf("authorized ops = %v, want [kv_query]", seen)
	}
}

// newKVQueryTestAPIAuth is newKVQueryTestAPI with an authenticator that consults
// allow(op) — enough to assert WHICH op name each endpoint authorizes under.
func newKVQueryTestAPIAuth(t *testing.T, allow func(op string) bool) (http.Handler, *kvIndexDispatcher, func()) {
	t.Helper()
	_, disp, cleanup := newKVQueryTestAPI(t, 1)
	opts := Options{Authenticator: func(r authz.AuthRequest) bool { return allow(r.Op) }}
	return Handler(disp, opts), disp, cleanup
}

// TestHTTPKVIndexRoutesShadowOnlyTheGet pins the exact extent of the route
// collision documented at the registration site, in both directions.
//
// "/v1/kv/{key}" matches three segments, so the four-segment drop route never
// competed with it; only the literal "GET /v1/kv/indexes" shadows the wildcard.
// Asserting the NEGATIVE half matters more than the positive: a later refactor
// that registered PUT or DELETE on /v1/kv/indexes would take a KV key away from
// callers silently, and nothing else would notice.
func TestHTTPKVIndexRoutesShadowOnlyTheGet(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 0)
	defer cleanup()

	// PUT and DELETE on the three-segment path still address the KV key.
	if rec := do(t, h, "PUT", "/v1/kv/indexes", `{"value":"hi"}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("PUT /v1/kv/indexes = %d, want 200 — the {key} route must still own it (%s)", rec.Code, rec.Body)
	}
	var del struct {
		Deleted bool `json:"deleted"`
	}
	rec := do(t, h, "DELETE", "/v1/kv/indexes", "", &del)
	if rec.Code != http.StatusOK || !del.Deleted {
		t.Fatalf("DELETE /v1/kv/indexes = %d deleted=%v, want 200/true (%s)", rec.Code, del.Deleted, rec.Body)
	}

	// GET is the one that moved: it now answers the catalog, not the key.
	var list kvIndexListResponse
	rec = do(t, h, "GET", "/v1/kv/indexes", "", &list)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/kv/indexes = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), `"indexes"`) {
		t.Fatalf("GET /v1/kv/indexes returned %s, want the catalog listing", rec.Body)
	}
}

// httpKVQuerySteeringFields are strings that appear in the SUBSTRING arms
// further down statusForError's switch. Each is a legal filter field name, and
// a kv_query filter refusal quotes the caller's field verbatim, so each is a
// string a client can plant inside its own error message.
var httpKVQuerySteeringFields = []string{
	"rate limited",
	"collection full",
	"not leader",
	"no leader",
	"no reachable owner",
	"cluster: write ",
}

// TestHTTPKVQueryFilterTextCannotSteerTheStatus is a regression test for a hole
// that was REAL before the kv_query arms were moved to the front of the switch.
//
// A filter refusal renders the caller's field name with %q, and the arms below
// match by substring, so a client asking for a field named "rate limited" got a
// 429 for its own permanent mistake — a status clients back off and RETRY,
// turning a bad query into an unbounded retry loop at one full cluster fan-out
// per attempt. Every string here used to produce a different status; all must
// now be 400.
func TestHTTPKVQueryFilterTextCannotSteerTheStatus(t *testing.T) {
	for _, field := range httpKVQuerySteeringFields {
		err := fmt.Errorf("%w: field %q is not a path", ops.ErrKVQueryFilter, field)
		if got := statusForError(err); got != http.StatusBadRequest {
			t.Errorf("a filter field named %q classified as %d, want 400", field, got)
		}
	}
}

// TestHTTPNonKVQueryStatusesUnchanged is the control for the reordering: moving
// the kv_query arms to the front must not capture anything that is not
// kv_query.
func TestHTTPNonKVQueryStatusesUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want int
	}{
		{"rate limited", errors.New("collection rate limited"), http.StatusTooManyRequests},
		{"collection full", errors.New("collection full"), http.StatusTooManyRequests},
		{"not leader", errors.New("not leader"), http.StatusServiceUnavailable},
		{"no reachable owner", errors.New("no reachable owner for shard 3"), http.StatusServiceUnavailable},
		{"write consistency", errors.New("cluster: write consistency not met"), http.StatusGatewayTimeout},
		{"unrelated fault", errors.New("open /var/lib/rostam/shard-7: no such file"), http.StatusInternalServerError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := statusForError(tc.err); got != tc.want {
				t.Fatalf("statusForError(%v) = %d, want %d", tc.err, got, tc.want)
			}
		})
	}
}

// TestHTTPKVQueryCursorCapIsCheckedBeforeDecoding pins the ORDER of the cursor
// checks, which is the whole substance of the cap at this edge.
//
// wire.DecodeKVQueryCursor enforces the 4 MiB cap, but only on bytes it has
// already been handed, and base64.DecodeString materializes the entire decoded
// blob before returning. Checking the cap only after the decode therefore lets a
// caller make this process allocate a multi-megabyte buffer for a cursor it is
// about to reject — bounded solely by maxJSONBody. The 400 must come from the
// ENCODED length, before a byte is allocated.
func TestHTTPKVQueryCursorCapIsCheckedBeforeDecoding(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 1)
	defer cleanup()

	// One base64 character past what the cap can encode. The content is
	// irrelevant: it must be refused on length alone, never decoded.
	oversized := strings.Repeat("A", base64.StdEncoding.EncodedLen(wire.KVQueryMaxCursorBytes)+1)
	body := fmt.Sprintf(`{"index":"by_age","cursor":%q}`, oversized)
	rec := do(t, h, "POST", "/v1/kv/query", body, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// The message must name the LENGTH, which is what proves the refusal came
	// from the pre-decode check rather than from the codec after the fact.
	if !strings.Contains(rec.Body.String(), "cursor is too large") {
		t.Fatalf("body = %s, want the pre-decode length refusal", rec.Body)
	}
}

// TestHTTPKVQueryCursorAtTheCapIsNotRefusedOnLength is the boundary control: a
// cursor exactly at the encoded cap must get PAST the length gate and be judged
// on its content, so the guard cannot be tightened into rejecting legal cursors.
func TestHTTPKVQueryCursorAtTheCapIsNotRefusedOnLength(t *testing.T) {
	h, _, cleanup := newKVQueryTestAPI(t, 1)
	defer cleanup()

	atCap := strings.Repeat("A", base64.StdEncoding.EncodedLen(wire.KVQueryMaxCursorBytes))
	body := fmt.Sprintf(`{"index":"by_age","cursor":%q}`, atCap)
	rec := do(t, h, "POST", "/v1/kv/query", body, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), "cursor is too large") {
		t.Fatalf("a cursor AT the cap was refused on length: %s", rec.Body)
	}
}
