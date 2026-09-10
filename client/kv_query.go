// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// The KV record-search op names, spelled ONCE per op for the whole system. The
// cluster twins (cluster/kv_index_admin.go, cluster/kv_query_broadcast.go) are
// unexported, so the two spellings are pinned equal from the cluster side
// (cluster/kv_index_client_names_test.go) and pinned to their literals from this
// side (TestClientKVIndexOpNames). A rename that touches only one end fails
// both.
//
// There is no constant for the INTERNAL per-group wrapper __kv_query_shard__ or
// for the readiness leaf __kv_index_ready__: both are peer-to-peer legs of a
// fan-out, authorised at admin, and a client that could name them would be
// addressing one shard group's slice of an answer rather than the answer.
const (
	// OpKVIndexSet upserts one index definition into the meta catalog. A
	// definition with Enabled=false DROPS the named index cluster-wide — the
	// same op, which is why CreateKVIndex and DropKVIndex send the same name.
	OpKVIndexSet = "__kv_index_set__"
	// OpKVIndexList returns the whole catalog with a per-definition ready bit.
	OpKVIndexList = "__kv_index_list__"
	// OpKVQuery is the client-facing filtered read over the KV keyspace.
	OpKVQuery = "kv_query"
)

// KVQueryDefaultLimit is the page size KVQuery sends when the caller left Limit
// zero. It is a DEFAULT rather than a "no limit": kv_query has no unbounded
// mode (wire.EncodeKVQueryArgs rejects limit 0 outright), because a page is what
// bounds a fan-out read's memory on every node it touches. 100 is small enough
// that a first page is cheap on a wide cluster and large enough that paging a
// modest result set is not dominated by round trips.
const KVQueryDefaultLimit uint16 = 100

// The kv_query refusals a caller ACTS on, re-typed at this end.
//
// WHY THE CLIENT RE-TYPES THEM. These errors are raised inside a shard's query
// leaf, possibly on another node, and reach this process as TEXT wrapped in a
// RemoteError — errors.Is against the server-side sentinels is impossible here
// (they live in ops/ops-kvindex, which this module deliberately cannot import;
// see TestClientIsEngineFree). Without re-typing, the one distinction a caller
// must make — "wait and retry" versus "fix the query" — would be a substring
// search in every calling application. Same job mapWriteErr does for the CAS
// conflict, one layer up.
//
// The original message is always wrapped, never replaced: it names the index and
// the shard group that refused, which is what an operator reading a log needs.
var (
	// ErrKVIndexNotFound: no definition by that name is installed, and the
	// coordinator found none in the meta catalog either. PERMANENT — create the
	// index (or fix the name); retrying cannot change it.
	ErrKVIndexNotFound = errors.New("client: kv_query: no such index")
	// ErrKVIndexBuilding: the definition exists but this group has not finished
	// installing or backfilling it, or it changed under the query. RETRYABLE —
	// the very next page may succeed, and a create-then-query lands here.
	ErrKVIndexBuilding = errors.New("client: kv_query: index is still building; retry")
	// ErrKVQueryUnavailable: the shard could not serve the read — it is being
	// removed from the node mid-scan, or the store is draining for close.
	// RETRYABLE against another replica.
	ErrKVQueryUnavailable = errors.New("client: kv_query: shard unavailable; retry")
	// ErrKVQueryFilter: a fact about the QUERY, not about the cluster — an
	// uncompilable filter, a scan with no consent, or a scan over budget.
	// PERMANENT; the remedy is a different query.
	ErrKVQueryFilter = errors.New("client: kv_query: invalid query")
)

// KVQueryPage is one page of a kv_query answer.
type KVQueryPage struct {
	// Rows are the matching keys, each with its value when the query asked for
	// one. A row whose value did not fit the page comes back key-only.
	Rows []wire.KVQueryRow
	// Records is populated ONLY when the query's Return was
	// wire.KVQueryReturnRecords, and is then ROW-ALIGNED with Rows: entry i is
	// row i's decoded record, or nil when that row came back key-only.
	Records []*wire.Record
	// Cursor resumes the query. Empty means every group answered in full and
	// the result set is exhausted; otherwise feed it back as Args.Cursor.
	Cursor []wire.KVQueryCont
}

// CreateKVIndex durably records d in the cluster's KV index catalog. The write
// goes through meta-Raft, so it can be issued against any node; it returns once
// the definition is committed, NOT once every shard group has backfilled it —
// poll ListKVIndexes for the ready bit, or expect ErrKVIndexBuilding from a
// query issued in between.
//
// d is validated locally first, so a definition the wire codec cannot represent
// (an over-long name, a Kind disagreeing with a "#count" path) costs no round
// trip. AppendKVIndexDef is the trusting side of the codec — it writes u8 length
// prefixes without checking them — so skipping Validate here would put a
// silently truncated definition on the wire.
func (c *Client) CreateKVIndex(ctx context.Context, d wire.KVIndexDef) error {
	d.Enabled = true
	if err := d.Validate(); err != nil {
		return err
	}
	_, err := c.Call(ctx, OpKVIndexSet, wire.EncodeKVIndexSetArgs(d))
	return err
}

// DropKVIndex removes the named index cluster-wide. A drop is the SAME meta-log
// write as a create with Enabled=false — there is no separate delete op — so the
// definition it carries must still satisfy the codec's shape rules, which is why
// it names a placeholder path rather than an empty one.
//
// Every query naming the index starts failing immediately afterwards, and
// re-creating it costs a full cache walk on every node. That is why the op is
// admin-scoped.
func (c *Client) DropKVIndex(ctx context.Context, name string) error {
	// PayloadPath must be non-empty and Kind must agree with it for Validate to
	// pass; the meta FSM keys the entry on Name alone and deletes on
	// Enabled=false, so neither field is read. "-" is chosen because it is in the
	// legal path charset and can never be mistaken for a real field.
	d := wire.KVIndexDef{Name: name, PayloadPath: "-", Kind: wire.KVIndexKindScalar, Enabled: false}
	if err := d.Validate(); err != nil {
		return err
	}
	_, err := c.Call(ctx, OpKVIndexSet, wire.EncodeKVIndexSetArgs(d))
	return err
}

// ListKVIndexes returns every definition in the catalog, sorted by name, paired
// with a readiness bit per definition.
//
// ready[i] HOLDS ONLY WHEN EVERY SHARD GROUP REPORTS THE INDEX READY, and it
// fails closed: a group still backfilling, a group that never installed the
// definition, and a group that could not be reached all make the bit false. A
// definition is never dropped from the answer because nothing could be said
// about its progress — a caller polling for readiness would read the name
// vanishing as "the create failed".
//
// The two slices are always the same length. An empty catalog is an empty answer
// (nil, nil, nil), not an error.
func (c *Client) ListKVIndexes(ctx context.Context) ([]wire.KVIndexDef, []bool, error) {
	body, err := c.Call(ctx, OpKVIndexList, nil)
	if err != nil {
		return nil, nil, err
	}
	if len(body) == 0 {
		return nil, nil, nil
	}
	return wire.DecodeKVIndexList(body)
}

// KVQuery runs one page of a filtered read over the KV keyspace and returns it.
//
// a is taken BY VALUE and defaulted on the copy: a zero Limit becomes
// KVQueryDefaultLimit and a zero Consistency becomes wire.ConsistencyLeaderOnly.
// A caller paging in a loop reuses one args variable, so writing the defaults
// back would quietly rewrite its request; and LeaderOnly is the right default
// for a read whose whole point is that the page is a complete answer.
//
// When a.Return is wire.KVQueryReturnRecords each row's value is decoded HERE,
// on the client: the server has already validated that the bytes are a record,
// and decoding them into a tree is work that does not belong on a node serving
// every other caller's queries too. A row that came back key-only (its value did
// not fit the page) gets a nil record rather than an error — fetch those keys
// with Get.
//
// Paging: feed the returned Cursor straight back as a.Cursor. An empty Cursor
// means the result set is exhausted.
func (c *Client) KVQuery(ctx context.Context, a wire.KVQueryArgs) (*KVQueryPage, error) {
	if a.Limit == 0 {
		a.Limit = KVQueryDefaultLimit
	}
	if a.Consistency == 0 {
		a.Consistency = wire.ConsistencyLeaderOnly
	}
	args, err := wire.EncodeKVQueryArgs(a)
	if err != nil {
		return nil, err
	}
	body, err := c.Call(ctx, OpKVQuery, args)
	if err != nil {
		return nil, mapKVQueryErr(err)
	}
	res, err := wire.DecodeKVQueryResult(body)
	if err != nil {
		return nil, err
	}
	page := &KVQueryPage{Rows: res.Rows, Cursor: res.Cursor}
	if a.Return != wire.KVQueryReturnRecords || len(res.Rows) == 0 {
		return page, nil
	}
	page.Records = make([]*wire.Record, len(res.Rows))
	for i, row := range res.Rows {
		if row.Value == nil {
			continue // key-only row: the value did not fit the page
		}
		rec, derr := wire.DecodeRecord(row.Value)
		if derr != nil {
			return nil, fmt.Errorf("client: kv_query: decode record for key %q: %w", row.Key, derr)
		}
		page.Records[i] = rec
	}
	return page, nil
}

// The exact server-side sentinel texts the mapping below anchors on. They are
// the Error() strings of ops.ErrKVQueryFilter and friends, restated here because
// this module cannot import those packages. Each is matched as an ANCHORED
// PREFIX of the peer's message, never searched for with strings.Contains: the
// filter refusal quotes the caller's own FIELD NAME verbatim, so a field named
// after the no-such-index sentinel would, under a substring test, turn a
// permanent error into one the caller retries forever. Same hole
// cluster.isKVQueryNoSuchIndex closes at the coordinator.
const (
	kvErrFilter        = "ops: kv_query: invalid filter"
	kvErrScanRequired  = "ops: kv_query: filter needs an index or scan:true"
	kvErrScanBudget    = "ops: kv_query: scan budget exceeded; use an index"
	kvErrIndexMissing  = "ops: kv_query: no KV index on this dispatcher"
	kvErrUnavailable   = "ops: kv_query: shard is unavailable; retry"
	kvErrStoreClosed   = "shard: store is closed"
	kvErrNoSuchIndex   = "kvindex: no such index"
	kvErrIndexBuilding = "kvindex: index is still building"
	kvErrIndexChanged  = "kvindex: index definition changed"
	kvErrCandidateCap  = "kvindex: candidate budget exceeded"
)

// mapKVQueryErr re-types a kv_query failure into one of this package's
// sentinels, wrapping (never replacing) the server's own message.
//
// ORDER IS LOAD-BEARING. The permanent filter refusal is tested FIRST, so a
// message that embeds a caller-chosen field name cannot be re-read as one of the
// retryable conditions further down. Everything else is decided by an anchored
// prefix on the peer's message with the transport wrapper removed, so a
// classification depends on how the message STARTS — which only the raising
// handler controls — rather than on what it happens to contain.
func mapKVQueryErr(err error) error {
	if err == nil {
		return nil
	}
	sentinel := ClassifyKVQueryMessage(kvQueryRemoteMsg(err))
	if sentinel == nil {
		return err
	}
	return fmt.Errorf("%w: %w", sentinel, err)
}

// ClassifyKVQueryMessage reports which of this package's kv_query sentinels a
// server's message belongs to, or nil for a message that is none of them.
//
// It is exported so the ENGINE side can pin the classification against the
// sentinels that actually produce these messages
// (cluster/kv_index_client_names_test.go). The engine cannot be imported here
// and the texts are therefore copied; a test that renders the real errors and
// runs them through this function is what stops the copies from drifting.
func ClassifyKVQueryMessage(msg string) error {
	switch {
	case strings.HasPrefix(msg, kvErrFilter),
		strings.HasPrefix(msg, kvErrScanRequired),
		strings.HasPrefix(msg, kvErrScanBudget),
		strings.HasPrefix(msg, kvErrIndexMissing),
		strings.HasPrefix(msg, kvErrCandidateCap):
		return ErrKVQueryFilter
	case strings.HasPrefix(msg, kvErrIndexBuilding),
		strings.HasPrefix(msg, kvErrIndexChanged):
		return ErrKVIndexBuilding
	case strings.HasPrefix(msg, kvErrNoSuchIndex):
		return ErrKVIndexNotFound
	case strings.HasPrefix(msg, kvErrUnavailable),
		strings.HasPrefix(msg, kvErrStoreClosed):
		return ErrKVQueryUnavailable
	}
	return nil
}

// kvQueryRemoteMsg peels the transport wrapper off err so the prefixes above
// anchor on what the SERVER wrote. A *RemoteError carries the peer's message as
// a field, which is exact; anything else is used as-is, which for an in-process
// error (the Direct path) is already the raw message.
func kvQueryRemoteMsg(err error) string {
	var re *RemoteError
	if errors.As(err, &re) {
		return re.Msg
	}
	return err.Error()
}
