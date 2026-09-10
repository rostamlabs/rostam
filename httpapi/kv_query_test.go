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
// back base64, the cursor is an opaque blob the client echoes verbatim, and the
// records projection carries the decoded record JSON.
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

// TestHTTPKVQueryRetryableIs503 pins the retryable bucket for every typed
// refusal that means "come back in a moment", including the store-draining one
// that reaches this transport as TEXT because httpapi cannot import shard.
func TestHTTPKVQueryRetryableIs503(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"index still building", fmt.Errorf("%w: on shard group 0", kvindex.ErrIndexBuilding)},
		{"index changed under the query", kvindex.ErrIndexChanged},
		{"shard unavailable mid-scan", ops.ErrKVQueryUnavailable},
		{"store draining, in process", errors.New(ops.StoreClosedMsg)},
		{"store draining, wrapped by the fan-out", errors.New("cluster: kv_query: shard group 3: " + ops.StoreClosedMsg)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := statusForError(tc.err); got != http.StatusServiceUnavailable {
				t.Fatalf("statusForError(%v) = %d, want 503", tc.err, got)
			}
		})
	}
}

// TestHTTPKVQueryPermanentIs4xx pins the permanent bucket against the same error
// set, and pins the ONE thing that must not move: a filter refusal that quotes
// the store-draining text inside a caller-chosen field stays permanent. Under a
// substring test it would become a 503 the caller retries forever.
func TestHTTPKVQueryPermanentIs4xx(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want int
	}{
		{"bad filter", ops.ErrKVQueryFilter, http.StatusBadRequest},
		{"scan consent missing", ops.ErrKVQueryScanRequired, http.StatusBadRequest},
		{"scan budget", ops.ErrKVQueryScanBudget, http.StatusBadRequest},
		{"no index on this dispatcher", ops.ErrKVIndexUnavailable, http.StatusBadRequest},
		{"candidate budget", kvindex.ErrCandidateBudget, http.StatusBadRequest},
		{"cursor cap", ops.ErrKVQueryCursorCap, http.StatusBadRequest},
		{"filter budget", wire.ErrKVFilterBudget, http.StatusBadRequest},
		{"malformed result frame", wire.ErrKVQueryResult, http.StatusBadRequest},
		{"unknown index", kvindex.ErrNoSuchIndex, http.StatusNotFound},
		{
			name: "a filter quoting the draining text stays permanent",
			err:  fmt.Errorf("%w: field %q is not a path", ops.ErrKVQueryFilter, ops.StoreClosedMsg),
			want: http.StatusBadRequest,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := statusForError(tc.err); got != tc.want {
				t.Fatalf("statusForError(%v) = %d, want %d", tc.err, got, tc.want)
			}
		})
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
