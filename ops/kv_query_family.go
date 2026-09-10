// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// KVQueryErrorClass is what a transport must DO with one of the kv_query
// refusals. It is deliberately not an HTTP status or a gRPC code: each transport
// spells the same three decisions in its own vocabulary, and naming the decision
// rather than the spelling is what lets one table serve all of them.
type KVQueryErrorClass uint8

const (
	// KVQueryErrPermanent: a fact about the query or the deployment that a
	// retry cannot change. HTTP 400, gRPC InvalidArgument.
	KVQueryErrPermanent KVQueryErrorClass = iota
	// KVQueryErrNotFound: the caller named something that does not exist. HTTP
	// 404, gRPC NotFound. Separate from Permanent because the remedy differs —
	// create the index, rather than fix the query.
	KVQueryErrNotFound
	// KVQueryErrRetryable: about this replica's readiness, not about the query.
	// HTTP 503, gRPC Unavailable. A caller told this SHOULD retry.
	KVQueryErrRetryable
)

// String renders a class for a test failure message.
func (c KVQueryErrorClass) String() string {
	switch c {
	case KVQueryErrPermanent:
		return "permanent"
	case KVQueryErrNotFound:
		return "not-found"
	case KVQueryErrRetryable:
		return "retryable"
	}
	return "unknown"
}

// KVQueryErrorSpec is one member of the kv_query error family: the sentinel, a
// name for failure messages, and the class every transport must give it.
type KVQueryErrorSpec struct {
	Name  string
	Err   error
	Class KVQueryErrorClass
}

// KVQueryErrorFamily is THE canonical list of the kv_query refusals — plus the
// one index-admin refusal that rides the same classifiers — that every transport
// must classify, and the class each one gets.
//
// WHY IT LIVES IN PRODUCTION CODE. The three classifiers —
// server.clientFacingErr, httpapi.statusForError and grpcapi.grpcError — are
// unexported, in three packages, with no common importer among their tests; a
// _test.go table cannot be shared across them. They HAVE drifted: the Task 7
// review found server classifying kvindex.ErrNoSuchIndex,
// kvindex.ErrCandidateBudget and wire.ErrKVQueryResult while httpapi silently
// omitted all three, with both files' comments claiming they were in sync. A
// comment cannot hold three tables together; an importable list can. Each
// transport's test walks this slice and maps the class to its own spelling, so
// adding a member here makes all three tests cover it at once and any transport
// that missed it fails.
//
// THE ONE MEMBER THAT IS NOT A SENTINEL is shard.ErrStoreClosed. This package
// cannot import shard (shard imports it), so the entry carries the message form
// — errors.New(StoreClosedMsg) — which is exactly what httpapi and grpcapi
// actually receive, since neither can import shard either and both match it by
// message. server's own test additionally asserts the real sentinel, closing the
// gap between the stand-in here and the value production raises.
//
// Adding a refusal to the kv_query family means adding it HERE first.
func KVQueryErrorFamily() []KVQueryErrorSpec {
	return []KVQueryErrorSpec{
		// Facts about the query the caller sent.
		{"ops.ErrKVQueryFilter", ErrKVQueryFilter, KVQueryErrPermanent},
		{"ops.ErrKVQueryScanRequired", ErrKVQueryScanRequired, KVQueryErrPermanent},
		{"ops.ErrKVQueryScanBudget", ErrKVQueryScanBudget, KVQueryErrPermanent},
		{"ops.ErrKVQueryCursorCap", ErrKVQueryCursorCap, KVQueryErrPermanent},
		{"kvindex.ErrCandidateBudget", kvindex.ErrCandidateBudget, KVQueryErrPermanent},
		// A fact about the deployment: no KV index is wired at all. Not
		// retryable either, so it shares the permanent bucket.
		{"ops.ErrKVIndexUnavailable", ErrKVIndexUnavailable, KVQueryErrPermanent},
		// Malformed frames, in both directions. The result frame is here
		// because a peer that hands this node an unusable page is answering a
		// question the caller asked, and the caller is the one who can stop
		// asking it.
		{"wire.ErrKVFilterBudget", wire.ErrKVFilterBudget, KVQueryErrPermanent},
		{"wire.ErrKVQueryArgs", wire.ErrKVQueryArgs, KVQueryErrPermanent},
		{"wire.ErrKVQueryArgsTruncated", wire.ErrKVQueryArgsTruncated, KVQueryErrPermanent},
		{"wire.ErrKVQueryResult", wire.ErrKVQueryResult, KVQueryErrPermanent},

		// THE ONE MEMBER THAT IS NOT A kv_query REFUSAL. wire.ErrKVIndexDef is
		// raised by the index-ADMIN op (__kv_index_set__) when a definition fails
		// the full admission check — the shape check plus the path parse
		// cluster.validateKVIndexDef performs before the meta commit.
		//
		// It belongs in THIS table because it is classified by exactly the same
		// three functions, and because getting it wrong reopens the kv_query
		// failure this whole family exists to prevent. Unclassified it falls to
		// the redaction/500/Internal bucket, which tells a client nothing and
		// invites the retry a permanent 400 stops. It is PERMANENT: a definition
		// this build cannot parse is a fact about the definition the caller sent,
		// and no amount of waiting makes it parse.
		{"wire.ErrKVIndexDef", wire.ErrKVIndexDef, KVQueryErrPermanent},

		// The caller named a definition that does not exist.
		{"kvindex.ErrNoSuchIndex", kvindex.ErrNoSuchIndex, KVQueryErrNotFound},

		// Readiness, not the query.
		{"kvindex.ErrIndexBuilding", kvindex.ErrIndexBuilding, KVQueryErrRetryable},
		{"kvindex.ErrIndexChanged", kvindex.ErrIndexChanged, KVQueryErrRetryable},
		{"ops.ErrKVQueryUnavailable", ErrKVQueryUnavailable, KVQueryErrRetryable},
		{"shard.ErrStoreClosed (message form)", errors.New(StoreClosedMsg), KVQueryErrRetryable},
	}
}
