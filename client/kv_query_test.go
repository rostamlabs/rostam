// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// kvRecordBytes builds a dynamic-mode record carrying one i64 field, the shape
// every kv_query fixture in this file stores as a row value. Dynamic mode keeps
// the fixture self-describing: no schema blob has to travel with it.
func kvRecordBytes(t *testing.T, field string, v int64) []byte {
	t.Helper()
	rec := &wire.Record{
		Mode:   wire.OperateModeDynamic,
		Fields: []wire.Field{{Name: field, Cell: wire.Cell{Type: wire.OperateTypeI64, U: uint64(v)}}},
	}
	return rec.Encode()
}

// kvQueryFakeServer answers every kv_query frame with fn's result, and fails the
// test on any other op. It returns the args of the LAST kv_query it saw, so a
// test can assert on what the client actually put on the wire.
func kvQueryFakeServer(t *testing.T, fn func(a wire.KVQueryArgs) (wire.KVQueryResult, error)) (addr string, last *wire.KVQueryArgs, stop func()) {
	t.Helper()
	var seen wire.KVQueryArgs
	addr, stop = startFakeServer(t, func(body []byte) (uint8, []byte) {
		opName, argsBytes := decodeOpFrame(t, body)
		if opName != OpKVQuery {
			t.Errorf("op = %q, want %q", opName, OpKVQuery)
			return StatusError, encodeErrorMsgFrame("unexpected op")
		}
		a, err := wire.DecodeKVQueryArgs(argsBytes)
		if err != nil {
			t.Errorf("DecodeKVQueryArgs: %v", err)
			return StatusError, encodeErrorMsgFrame(err.Error())
		}
		seen = a
		res, herr := fn(a)
		if herr != nil {
			return StatusError, encodeErrorMsgFrame(herr.Error())
		}
		payload, eerr := wire.EncodeKVQueryResult(res)
		if eerr != nil {
			t.Errorf("EncodeKVQueryResult: %v", eerr)
			return StatusError, encodeErrorMsgFrame(eerr.Error())
		}
		return StatusOK, payload
	})
	return addr, &seen, stop
}

// TestClientKVQueryRoundtrip pages a two-page query to exhaustion through the
// typed client: the rows arrive verbatim, the continuation cursor the first page
// returned is what the second page sends back, and the record projection is
// decoded CLIENT-side (decision 6) so the caller gets *wire.Record, not bytes.
func TestClientKVQueryRoundtrip(t *testing.T) {
	const field = "age"
	page1 := wire.KVQueryResult{
		Rows: []wire.KVQueryRow{
			{Key: []byte("u:0000001"), Value: kvRecordBytes(t, field, 1)},
			{Key: []byte("u:0000002"), Value: kvRecordBytes(t, field, 2)},
		},
		Cursor: []wire.KVQueryCont{{Group: 0, After: []byte("u:0000002"), More: true}},
	}
	page2 := wire.KVQueryResult{
		Rows: []wire.KVQueryRow{{Key: []byte("u:0000003"), Value: kvRecordBytes(t, field, 3)}},
	}

	addr, last, stop := kvQueryFakeServer(t, func(a wire.KVQueryArgs) (wire.KVQueryResult, error) {
		if len(a.Cursor) == 0 {
			return page1, nil
		}
		return page2, nil
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	args := wire.KVQueryArgs{Index: "by_age", Return: wire.KVQueryReturnRecords}
	var gotKeys []string
	var gotAges []int64
	for page := 0; page < 4; page++ {
		p, qerr := c.KVQuery(ctx, args)
		if qerr != nil {
			t.Fatalf("page %d: KVQuery: %v", page, qerr)
		}
		if len(p.Records) != len(p.Rows) {
			t.Fatalf("page %d: %d records for %d rows — Records must be row-aligned", page, len(p.Records), len(p.Rows))
		}
		for i, row := range p.Rows {
			gotKeys = append(gotKeys, string(row.Key))
			rec := p.Records[i]
			if rec == nil || len(rec.Fields) != 1 || rec.Fields[0].Name != field {
				t.Fatalf("page %d row %d: record = %+v, want one %q field", page, i, rec, field)
			}
			gotAges = append(gotAges, int64(rec.Fields[0].Cell.U))
		}
		if len(p.Cursor) == 0 {
			break
		}
		args.Cursor = p.Cursor
	}

	wantKeys := []string{"u:0000001", "u:0000002", "u:0000003"}
	if !reflect.DeepEqual(gotKeys, wantKeys) {
		t.Fatalf("keys = %v, want %v", gotKeys, wantKeys)
	}
	if !reflect.DeepEqual(gotAges, []int64{1, 2, 3}) {
		t.Fatalf("decoded ages = %v, want [1 2 3]", gotAges)
	}
	// The SECOND request must have carried page one's continuation verbatim.
	if len(last.Cursor) != 1 || !bytes.Equal(last.Cursor[0].After, []byte("u:0000002")) || !last.Cursor[0].More {
		t.Fatalf("resumed cursor = %+v, want page one's continuation", last.Cursor)
	}
}

// TestClientKVQueryKeysProjectionDecodesNoRecords pins that Records stays nil
// for a non-record projection: decoding is gated on the Return the caller asked
// for, not attempted on whatever bytes came back.
func TestClientKVQueryKeysProjectionDecodesNoRecords(t *testing.T) {
	addr, _, stop := kvQueryFakeServer(t, func(wire.KVQueryArgs) (wire.KVQueryResult, error) {
		return wire.KVQueryResult{Rows: []wire.KVQueryRow{{Key: []byte("k1")}}}, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	p, err := c.KVQuery(context.Background(), wire.KVQueryArgs{Index: "by_age"})
	if err != nil {
		t.Fatalf("KVQuery: %v", err)
	}
	if len(p.Rows) != 1 || p.Records != nil {
		t.Fatalf("rows=%d records=%v, want 1 row and no decoded records", len(p.Rows), p.Records)
	}
}

// TestClientKVQueryDefaults asserts on the ENCODED args: a zero Limit becomes
// KVQueryDefaultLimit and a zero Consistency becomes LeaderOnly, so the default
// call is a bounded page read off the leader rather than an args-encoding error
// (EncodeKVQueryArgs rejects limit 0) or an any-replica read.
func TestClientKVQueryDefaults(t *testing.T) {
	addr, last, stop := kvQueryFakeServer(t, func(wire.KVQueryArgs) (wire.KVQueryResult, error) {
		return wire.KVQueryResult{}, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	if _, err := c.KVQuery(context.Background(), wire.KVQueryArgs{Index: "by_age"}); err != nil {
		t.Fatalf("KVQuery: %v", err)
	}
	if last.Limit != KVQueryDefaultLimit {
		t.Fatalf("limit = %d, want the %d default", last.Limit, KVQueryDefaultLimit)
	}
	if last.Consistency != wire.ConsistencyLeaderOnly {
		t.Fatalf("consistency = %d, want LeaderOnly (%d)", last.Consistency, wire.ConsistencyLeaderOnly)
	}
}

// TestClientKVQueryKeepsAnExplicitConsistency pins that the defaulting only
// fills a ZERO field: an explicit AnyReplica read must survive, which a naive
// "always overwrite" default would silently upgrade to LeaderOnly.
func TestClientKVQueryKeepsAnExplicitConsistency(t *testing.T) {
	addr, last, stop := kvQueryFakeServer(t, func(wire.KVQueryArgs) (wire.KVQueryResult, error) {
		return wire.KVQueryResult{}, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	in := wire.KVQueryArgs{Index: "by_age", Limit: 7, Consistency: wire.ConsistencyLinearizable}
	if _, err := c.KVQuery(context.Background(), in); err != nil {
		t.Fatalf("KVQuery: %v", err)
	}
	if last.Limit != 7 || last.Consistency != wire.ConsistencyLinearizable {
		t.Fatalf("limit/consistency = %d/%d, want 7/%d", last.Limit, last.Consistency, wire.ConsistencyLinearizable)
	}
}

// TestClientKVQueryDoesNotMutateCallerArgs pins that the defaulting happens on a
// COPY. KVQuery takes its args by value, but a caller threading a cursor in a
// loop reuses one variable, and a defaulted field written back through a pointer
// would silently change the caller's own request.
func TestClientKVQueryDoesNotMutateCallerArgs(t *testing.T) {
	addr, _, stop := kvQueryFakeServer(t, func(wire.KVQueryArgs) (wire.KVQueryResult, error) {
		return wire.KVQueryResult{}, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	in := wire.KVQueryArgs{Index: "by_age"}
	if _, err := c.KVQuery(context.Background(), in); err != nil {
		t.Fatalf("KVQuery: %v", err)
	}
	if in.Limit != 0 || in.Consistency != 0 {
		t.Fatalf("caller args mutated: %+v", in)
	}
}

// TestClientCreateAndListKVIndexes drives the three catalog calls against a fake
// that plays the meta catalog: create sends Enabled=true, drop sends the same op
// with Enabled=false (that is what a drop IS on the wire), and list decodes the
// definitions with their per-name readiness bits.
func TestClientCreateAndListKVIndexes(t *testing.T) {
	def := wire.KVIndexDef{Name: "by_age", KeyPrefix: []byte("u:"), PayloadPath: "age", Kind: wire.KVIndexKindScalar, Enabled: true}

	catalog := map[string]wire.KVIndexDef{}
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		opName, argsBytes := decodeOpFrame(t, body)
		switch opName {
		case OpKVIndexSet:
			d, err := wire.DecodeKVIndexSetArgs(argsBytes)
			if err != nil {
				t.Errorf("DecodeKVIndexSetArgs: %v", err)
				return StatusError, encodeErrorMsgFrame(err.Error())
			}
			if d.Enabled {
				catalog[d.Name] = d
			} else {
				delete(catalog, d.Name)
			}
			return StatusOK, nil
		case OpKVIndexList:
			if len(argsBytes) != 0 {
				t.Errorf("%s args = %d bytes, want none", OpKVIndexList, len(argsBytes))
			}
			defs := make([]wire.KVIndexDef, 0, len(catalog))
			for _, d := range catalog {
				defs = append(defs, d)
			}
			ready := make([]bool, len(defs))
			for i := range ready {
				ready[i] = true
			}
			return StatusOK, wire.EncodeKVIndexList(defs, ready)
		default:
			t.Errorf("unexpected op %q", opName)
			return StatusError, encodeErrorMsgFrame("unexpected op")
		}
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	ctx := context.Background()

	if err := c.CreateKVIndex(ctx, def); err != nil {
		t.Fatalf("CreateKVIndex: %v", err)
	}
	defs, ready, err := c.ListKVIndexes(ctx)
	if err != nil {
		t.Fatalf("ListKVIndexes: %v", err)
	}
	if len(defs) != 1 || defs[0].Name != "by_age" || defs[0].PayloadPath != "age" ||
		!bytes.Equal(defs[0].KeyPrefix, []byte("u:")) || !defs[0].Enabled {
		t.Fatalf("defs = %+v, want the created definition", defs)
	}
	if len(ready) != 1 || !ready[0] {
		t.Fatalf("ready = %v, want [true]", ready)
	}

	if err := c.DropKVIndex(ctx, "by_age"); err != nil {
		t.Fatalf("DropKVIndex: %v", err)
	}
	defs, ready, err = c.ListKVIndexes(ctx)
	if err != nil {
		t.Fatalf("ListKVIndexes after drop: %v", err)
	}
	if len(defs) != 0 || len(ready) != 0 {
		t.Fatalf("after drop defs=%+v ready=%v, want empty", defs, ready)
	}
}

// TestClientCreateKVIndexValidatesBeforeSending pins that a definition the wire
// codec cannot represent fails LOCALLY, with no round trip: the encoder is
// panic-free but silently truncating for an over-long name, and a caller that
// only learns about it from a peer's error has already spent a network hop.
func TestClientCreateKVIndexValidatesBeforeSending(t *testing.T) {
	addr, stop := startFakeServer(t, func([]byte) (uint8, []byte) {
		t.Error("a definition that fails Validate must not reach the server")
		return StatusOK, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	bad := wire.KVIndexDef{Name: "not a legal name!", PayloadPath: "age", Enabled: true}
	if err := c.CreateKVIndex(context.Background(), bad); !errors.Is(err, wire.ErrKVIndexDef) {
		t.Fatalf("CreateKVIndex(bad) = %v, want a wire.ErrKVIndexDef", err)
	}
	if err := c.DropKVIndex(context.Background(), "not a legal name!"); !errors.Is(err, wire.ErrKVIndexDef) {
		t.Fatalf("DropKVIndex(bad) = %v, want a wire.ErrKVIndexDef", err)
	}
}

// TestClientKVIndexOpNames pins the three op-name constants to their literal
// wire strings. The cluster twins are unexported, so the cross-module equality
// is pinned from the cluster side (cluster/kv_index_client_names_test.go); this
// end fixes the literals so the two tests together mean "one name per op".
func TestClientKVIndexOpNames(t *testing.T) {
	for _, tc := range []struct{ got, want string }{
		{OpKVIndexSet, "__kv_index_set__"},
		{OpKVIndexList, "__kv_index_list__"},
		{OpKVQuery, "kv_query"},
	} {
		if tc.got != tc.want {
			t.Errorf("op name = %q, want %q", tc.got, tc.want)
		}
	}
}

// TestClientKVQueryMapsTypedErrors is the phase-2 mapWriteErr precedent applied
// to the kv_query family: the server's refusals cross the wire as TEXT (wrapped
// in client.ServerError), so the client re-types the ones a caller acts on —
// "retry, the index is still coming up" versus "fix your query".
func TestClientKVQueryMapsTypedErrors(t *testing.T) {
	tests := []struct {
		name string
		msg  string
		want error
	}{
		{
			name: "leaf no-such-index",
			msg:  `kvindex: no such index: "by_age" on shard group 3`,
			want: ErrKVIndexNotFound,
		},
		{
			name: "coordinator rewrite to still-building",
			msg:  `kvindex: index is still building: "by_age" is in the meta catalog but not yet installed on shard group 3 (retry)`,
			want: ErrKVIndexBuilding,
		},
		{
			name: "definition changed under the query",
			msg:  "kvindex: index definition changed under the query",
			want: ErrKVIndexBuilding,
		},
		{
			name: "shard unavailable mid-scan",
			msg:  "ops: kv_query: shard is unavailable; retry",
			want: ErrKVQueryUnavailable,
		},
		{
			name: "store draining",
			msg:  "shard: store is closed",
			want: ErrKVQueryUnavailable,
		},
		{
			name: "bad filter",
			msg:  `ops: kv_query: invalid filter: field "$rec" addresses the reserved "$" namespace`,
			want: ErrKVQueryFilter,
		},
		{
			name: "scan budget",
			msg:  "ops: kv_query: scan budget exceeded; use an index",
			want: ErrKVQueryFilter,
		},
		{
			name: "scan consent missing",
			msg:  "ops: kv_query: filter needs an index or scan:true",
			want: ErrKVQueryFilter,
		},
		{
			// The smuggling case cluster.isKVQueryNoSuchIndex closes, restated at
			// this end: a client-chosen filter FIELD is quoted verbatim into the
			// invalid-filter message, so a field named after the no-such-index
			// sentinel must not turn a permanent error into a retryable one.
			name: "filter text quoting the no-such-index sentinel stays permanent",
			msg:  `ops: kv_query: invalid filter: field "kvindex: no such index: \"x\" on shard group 1" is not a path`,
			want: ErrKVQueryFilter,
		},
		{
			name: "an unrelated fault is left alone",
			msg:  "open /var/lib/rostam/shard-7: no such file",
			want: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			addr, stop := startFakeServer(t, func([]byte) (uint8, []byte) {
				return StatusError, encodeErrorMsgFrame(tc.msg)
			})
			defer stop()
			c, err := New(Config{Servers: []string{addr}})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer func() { _ = c.Close() }()

			_, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{Index: "by_age"})
			if qerr == nil {
				t.Fatal("KVQuery returned no error")
			}
			if tc.want == nil {
				for _, sentinel := range []error{ErrKVIndexNotFound, ErrKVIndexBuilding, ErrKVQueryUnavailable, ErrKVQueryFilter} {
					if errors.Is(qerr, sentinel) {
						t.Fatalf("%v was typed as %v; an unrelated fault must be left alone", qerr, sentinel)
					}
				}
				return
			}
			if !errors.Is(qerr, tc.want) {
				t.Fatalf("KVQuery error %v is not %v", qerr, tc.want)
			}
			// The original text is never thrown away: a caller logging the error
			// must still see which index and which shard group refused.
			if !bytes.Contains([]byte(qerr.Error()), []byte(tc.msg)) {
				t.Fatalf("mapped error %q dropped the server's message %q", qerr, tc.msg)
			}
		})
	}
}

// TestClientKVQueryRejectsAnUnrepresentableArgsLocally pins that an args value
// the codec cannot encode fails before the round trip, the shape Operate has:
// the caller gets the codec's typed error, not a peer's decode complaint.
func TestClientKVQueryRejectsAnUnrepresentableArgsLocally(t *testing.T) {
	addr, stop := startFakeServer(t, func([]byte) (uint8, []byte) {
		t.Error("unencodable args must not reach the server")
		return StatusOK, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	// No index and no scan consent: EncodeKVQueryArgs refuses it outright.
	_, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{})
	if !errors.Is(qerr, wire.ErrKVQueryArgs) {
		t.Fatalf("KVQuery(no index, no scan) = %v, want wire.ErrKVQueryArgs", qerr)
	}
}

// TestClientKVQueryRecordDecodeFailureIsReported pins that a value the server
// labelled a record but which does not decode is an ERROR, not a nil hole in a
// row-aligned slice a caller would dereference.
func TestClientKVQueryRecordDecodeFailureIsReported(t *testing.T) {
	addr, _, stop := kvQueryFakeServer(t, func(wire.KVQueryArgs) (wire.KVQueryResult, error) {
		return wire.KVQueryResult{Rows: []wire.KVQueryRow{{Key: []byte("k1"), Value: []byte{0xff, 0xff}}}}, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	_, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{Index: "by_age", Return: wire.KVQueryReturnRecords})
	if qerr == nil {
		t.Fatal("a row whose value is not a record must not decode silently")
	}
	if !bytes.Contains([]byte(qerr.Error()), []byte("k1")) {
		t.Fatalf("decode error %q does not name the offending key", qerr)
	}
}

// TestClientKVQueryOversizeRowKeepsANilRecord pins the leaf's oversize-row
// convention end to end: a records query whose value did not fit the page comes
// back key-only (Value nil), and the client must leave that record slot nil
// rather than fail the whole page — the caller fetches those keys with get.
func TestClientKVQueryOversizeRowKeepsANilRecord(t *testing.T) {
	addr, _, stop := kvQueryFakeServer(t, func(wire.KVQueryArgs) (wire.KVQueryResult, error) {
		return wire.KVQueryResult{Rows: []wire.KVQueryRow{
			{Key: []byte("k1"), Value: kvRecordBytes(t, "age", 1)},
			{Key: []byte("k2")}, // value omitted: it did not fit the page
		}}, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	p, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{Index: "by_age", Return: wire.KVQueryReturnRecords})
	if qerr != nil {
		t.Fatalf("KVQuery: %v", qerr)
	}
	if len(p.Records) != 2 || p.Records[0] == nil || p.Records[1] != nil {
		t.Fatalf("records = %v, want [decoded, nil] for the omitted value", p.Records)
	}
}

// TestClientListKVIndexesEmptyCatalog pins that an empty catalog is an empty
// answer rather than an error: __kv_index_list__ returns a zero-length payload
// when nothing is defined.
func TestClientListKVIndexesEmptyCatalog(t *testing.T) {
	addr, stop := startFakeServer(t, func(body []byte) (uint8, []byte) {
		if opName, _ := decodeOpFrame(t, body); opName != OpKVIndexList {
			t.Errorf("op = %q, want %q", opName, OpKVIndexList)
		}
		return StatusOK, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	defs, ready, lerr := c.ListKVIndexes(context.Background())
	if lerr != nil {
		t.Fatalf("ListKVIndexes: %v", lerr)
	}
	if len(defs) != 0 || len(ready) != 0 {
		t.Fatalf("defs=%v ready=%v, want both empty", defs, ready)
	}
}

// TestClientKVQueryOverLimit pins that a limit above the wire cap is refused by
// the codec rather than silently clamped — a caller that asked for 5000 rows and
// got 1000 back with no cursor would read the page as complete.
func TestClientKVQueryOverLimit(t *testing.T) {
	addr, stop := startFakeServer(t, func([]byte) (uint8, []byte) {
		t.Error("an over-cap limit must not reach the server")
		return StatusOK, nil
	})
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	_, qerr := c.KVQuery(context.Background(), wire.KVQueryArgs{Index: "by_age", Limit: wire.KVQueryMaxLimit + 1})
	if !errors.Is(qerr, wire.ErrKVQueryArgs) {
		t.Fatalf("KVQuery(limit %d) = %v, want wire.ErrKVQueryArgs", wire.KVQueryMaxLimit+1, qerr)
	}
}
