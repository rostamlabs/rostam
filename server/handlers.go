// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"log/slog"
	"runtime/debug"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/rlog"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// Authenticator gates each request when protocol-v2 framing is in play. It is
// the unified RBAC authorizer (authz.Authenticator): a transport builds an
// authz.AuthRequest{Token, Op, Args} from the decoded frame and the authorizer
// derives the (action, resource) and matches it against the principal's scopes.
// Returns true to allow the request; false to fail with StatusUnauthorized.
// A nil Authenticator accepts every request (legacy / no-auth mode). A v1 frame
// carries no token, so a non-nil authenticator denies it (deny-by-default).
type Authenticator = authz.Authenticator

// dispatch parses one request body and calls Dispatcher.Call. Returns the
// wire status code and payload slice ready for writeResponse — splitting
// these out (instead of returning a pre-encoded []byte) lets the server
// loop write directly to the bufio.Writer without the per-response
// EncodeResponse allocation.
//
// Protocol detection: byte 0 of the body is the version. v2 (0x02) carries
// a [tokenLen:1][token] prefix before the v1 body. Anything else is treated
// as v1 (the byte 0 IS the opNameLen, guaranteed >=3 by the registry).
//
// clientCN is the VERIFIED mTLS client-cert CommonName for this connection, or
// "" when the connection is plaintext or the peer presented no verified cert. It
// is supplied by the connection-handling loop (server.go), which reads it from
// the *tls.Conn's ConnectionState().VerifiedChains exactly once after the
// handshake — NEVER from a spoofable in-frame field. The authorizer uses it as
// the cert principal only when the request carries no bearer token (token wins).
func dispatch(disp Dispatcher, frame []byte, auth Authenticator, clientCN string, alog *rlog.AccessLog) (status uint8, payload []byte) {
	return dispatchInto(disp, frame, auth, clientCN, alog, nil)
}

// dispatchInto is dispatch with a transport-owned reply buffer. When dst is
// non-nil and disp implements AppendDispatcher, the payload is appended to dst
// and allocates nothing; otherwise this is exactly dispatch. See
// AppendDispatcher for the lifetime the caller must honour.
func dispatchInto(disp Dispatcher, frame []byte, auth Authenticator, clientCN string, alog *rlog.AccessLog, dst []byte) (status uint8, payload []byte) {
	// Access log (OPT-IN). When -access-log is off, alog is nil: no request id is
	// generated, no timing is taken, and this path is byte-identical to the
	// pre-access-log dispatch. When on, we generate a per-request id (the TCP
	// protocol carries none inbound), stamp start, and emit one line on return via
	// the deferred closure below — which reads opName/token/reqID hoisted here so
	// they are populated by the time it runs.
	var (
		opName string
		token  string
		reqID  string
	)
	if alog.Enabled() {
		reqID = rlog.NewID()
		start := time.Now()
		defer func() {
			alog.Log(rlog.Entry{
				RequestID: reqID,
				Transport: "tcp",
				Op:        opName,
				Status:    statusName(status),
				Latency:   time.Since(start),
				Principal: rlog.Principal(token, clientCN),
				Bytes:     len(payload),
			})
		}()
	}

	// Contain a panic to THIS request. Without this a single index-out-of-range
	// or nil-map in any op handler (reachable from a crafted frame through the
	// arg decoders) would crash the whole process — every shard and every
	// connection — because an unrecovered panic in any goroutine takes the
	// process down. Recover, return a generic error to the client (never the
	// panic detail — it can leak internals), and log server-side with a stack.
	defer func() {
		if r := recover(); r != nil {
			slog.Error("recovered panic dispatching request",
				"transport", "tcp", "request_id", reqID, "panic", r, "stack", string(debug.Stack()))
			status, payload = StatusError, EncodeErrorPayload("internal error")
		}
	}()

	body := frame
	if len(frame) > 0 && frame[0] == ProtocolV2 {
		var err error
		token, body, err = DecodeRequestV2(frame)
		if err != nil {
			return StatusError, EncodeErrorPayload(err.Error())
		}
	}
	on, args, err := DecodeRequest(body)
	if err != nil {
		return StatusError, EncodeErrorPayload(err.Error())
	}
	opName = on
	// Authorize against the decoded (token, op, args). The fan-out dispatcher
	// unwraps the __wc__ envelope AFTER dispatch, so a TCP client that wraps a
	// write in __wc__ would be authorized here against "__wc__" (classified
	// admin → fail-closed). The HTTP/gRPC edges unwrap before auth; the TCP
	// client never builds __wc__ frames itself (it is built server-side by the
	// HTTP/gRPC callWrite paths), so a raw __wc__ over TCP requiring admin is the
	// correct conservative default.
	if auth != nil && !auth(authz.AuthRequest{Token: token, ClientCN: clientCN, Op: opName, Args: args}) {
		return StatusUnauthorized, nil
	}
	var result []byte
	var callErr error
	if ad, ok := disp.(AppendDispatcher); ok && dst != nil {
		result, callErr = ad.CallAppend(opName, args, dst)
	} else {
		result, callErr = disp.Call(opName, args)
	}
	return mapResult(disp, result, callErr, reqID)
}

// statusName renders a wire status code as a short label for the access log, so
// an operator reads "ok"/"not_found"/"error" rather than a bare integer.
func statusName(s uint8) string {
	switch s {
	case StatusOK:
		return "ok"
	case StatusNotFound:
		return "not_found"
	case StatusNotLeader:
		return "not_leader"
	case StatusError:
		return "error"
	case StatusUnauthorized:
		return "unauthorized"
	default:
		return "unknown"
	}
}

// mapResult converts an op result + error into a wire status code and
// payload. For NotLeader errors (errors.Is(err, shard.ErrNotLeader)),
// the per-shard hint from *shard.NotLeaderError.LeaderAddr is preferred;
// disp.LeaderAddr() is the fallback when the hint is empty. reqID (may be "")
// correlates the redacted internal-error log with the request's access line.
func mapResult(disp Dispatcher, result []byte, err error, reqID string) (uint8, []byte) {
	switch {
	case err == nil:
		return StatusOK, result
	case errors.Is(err, cache.ErrNotFound),
		// ops.ErrVectorRecordAbsent: vector_operate with create = NONE against a
		// payload key that holds no record. It is the vector twin of the signal
		// handleOperate raises for the same call against an absent KV key —
		// handleOperate maps that one to cache.ErrNotFound (ops.handleOperate's
		// doc comment; TestOperateCreateNoneOnAbsent pins it), which this very
		// case already answers StatusNotFound for. Answering the same status for
		// the vector twin is what keeps KV and vector operate telling a caller the
		// same thing; classified nowhere, it fell through to the redacted
		// StatusError bucket and read as a server fault.
		//
		// It is classified HERE rather than in clientFacingErr because the status
		// is what carries the meaning: not-found is its own wire status, not a
		// client-facing error message.
		errors.Is(err, ops.ErrVectorRecordAbsent),
		// The clustered path stringifies the sentinel across the Raft boundary, so
		// errors.Is stops matching — the same reason clientFacingErr carries a
		// message-shape fallback. ops.IsVectorRecordAbsentMessage, not
		// strings.Contains: a bare substring check would also match an unrelated
		// internal error that merely wraps the sentinel, leaking it unredacted.
		ops.IsVectorRecordAbsentMessage(err.Error()):
		return StatusNotFound, nil
	case errors.Is(err, shard.ErrNotLeader):
		var nle *shard.NotLeaderError
		hint := ""
		if errors.As(err, &nle) {
			hint = nle.LeaderAddr
		}
		if hint == "" {
			hint = disp.LeaderAddr()
		}
		return StatusNotLeader, EncodeLeaderAddrPayload(hint)
	default:
		if clientFacingErr(err) {
			// A classified client-facing signal (validation mistake, create/CAS
			// conflict, quota refusal, unknown collection, routing/leadership
			// transient): the message is a signal the caller needs, safe to return.
			return StatusError, EncodeErrorPayload(err.Error())
		}
		// Catch-all internal fault: the raw text can wrap internal filesystem paths,
		// partition/shard identifiers, leader addresses, or low-level index faults.
		// An authenticated but low-privilege client must not read that topology/
		// implementation detail off the wire, so log it server-side and return a
		// generic payload — mirroring the HTTP edge's writeDispatchError, which
		// redacts the identical 500 bucket to "internal error".
		slog.Error("internal error dispatching op",
			"transport", "tcp", "request_id", reqID, "err", err)
		return StatusError, EncodeErrorPayload("internal error")
	}
}

// clientFacingErr reports whether err is a classified client-facing signal whose
// message is safe to return verbatim to the caller: a validation mistake (bad
// dim, empty filter, bad create-config, ...), a create/CAS conflict, a quota or
// rate-limit refusal, an unknown collection, or a routing/leadership transient.
// Anything else is treated as an internal fault and redacted by mapResult, so
// this MUST fail closed (return false) for unrecognized errors.
//
// It intentionally mirrors the client-facing (4xx / 503) buckets of the HTTP
// edge's statusForError (httpapi.statusForError) so the two transports agree on
// what is safe to disclose. That classifier cannot be imported here without a
// layering cycle (httpapi is a higher-level package), so the recognized set is
// duplicated; keep the two in sync when either grows. String fallbacks cover the
// clustered path where a sentinel is stringified across the Raft boundary.
func clientFacingErr(err error) bool {
	switch {
	case errors.Is(err, vector.ErrDimMismatch),
		// vector.ErrRecordTooLarge: a payload carrying a record value above the
		// storage cap. A caller mistake with a clear remedy (send a smaller
		// record), and the message names the key and both sizes — all of which
		// the caller sent — so it is safe to return verbatim, like a bad dim.
		//
		// Matched by sentinel AND by exact message shape: shard.decodePBResult
		// rebuilds an op error with errors.New across replication, so a
		// clustered apply loses errors.Is identity and the error would fall
		// through to the redacted internal-fault bucket. The message-shape arms
		// use vector.IsRecordTooLargeMessage, NOT strings.Contains — a bare
		// substring check would also match an unrelated internal error that
		// merely wraps the sentinel (e.g. a WAL/path error), leaking it to the
		// caller unredacted.
		errors.Is(err, vector.ErrRecordTooLarge),
		vector.IsRecordTooLargeMessage(err.Error()),
		// vector.ErrRecordMalformed: a payload carrying record bytes no operate
		// engine can open. Same bucket and same reasoning as the cap above — the
		// caller sent those bytes, the remedy is to send a well-formed record,
		// and the message names only the payload key the caller chose. Same
		// sentinel-plus-message-shape matching, for the same clustered-apply
		// reason.
		errors.Is(err, vector.ErrRecordMalformed),
		vector.IsRecordMalformedMessage(err.Error()),
		// vector.ErrPayloadKeyNotRecord: a vector_operate aimed at a payload key
		// that holds a plain value (or the reserved content key) rather than a
		// record. Same bucket and same reasoning as the two above — the caller
		// chose the key, the remedy is to name a record key, and the message
		// discloses only that key and the kind stored under it. Same
		// sentinel-plus-message-shape matching, for the same clustered-apply
		// reason.
		errors.Is(err, vector.ErrPayloadKeyNotRecord),
		vector.IsPayloadKeyNotRecordMessage(err.Error()),
		errors.Is(err, vector.ErrEmptyFilter),
		errors.Is(err, vector.ErrEmptyGroupBy),
		errors.Is(err, vector.ErrSparseMismatch),
		errors.Is(err, vector.ErrSparseUnsorted),
		errors.Is(err, vector.ErrSpaceModalityMismatch),
		errors.Is(err, vector.ErrUnknownVectorName),
		errors.Is(err, vector.ErrEmptyNamedVectors),
		errors.Is(err, vector.ErrReservedVectorName),
		errors.Is(err, vector.ErrEmptyVectorName),
		errors.Is(err, vector.ErrFullTextDisabled),
		// ops.ErrMalformedPayloadJSON: a bulk-staged per-point payload that framed
		// correctly but is not a metadata object. Kept in sync with httpapi's
		// statusForError, which answers 400 for it — a caller mistake, not a server
		// fault, whichever transport carried it.
		errors.Is(err, ops.ErrMalformedPayloadJSON),
		// wire.ErrOperateArgs: a malformed operate frame — a vector_operate or a KV
		// operate whose args do not decode, or which names a second target or a
		// TTL the op does not carry (DecodeVectorOperateArgs /
		// ops.checkVectorOperateArgs / DecodeOperateArgs). The caller built the
		// frame, so it is their mistake to fix; unclassified it read as a server
		// fault. The message names no key, no path and no size — only that the
		// arguments are invalid — so it is safe verbatim.
		//
		// Sentinel AND exact message shape, both arms load-bearing. The sentinel
		// arm covers the direct path, where the decoder the handler itself calls
		// hands the error back with its identity intact.
		// The message-shape arm covers the clustered path, which stringifies the
		// sentinel across the Raft boundary:
		// an operate handler decodes its frame INSIDE the FSM apply, so
		// shard.decodePBResult rebuilds the error with errors.New and errors.Is
		// stops matching — which left a malformed frame from a clustered caller
		// redacted as a server fault, the one case the sentinel arm above cannot
		// reach. wire.IsOperateArgsMessage, not strings.Contains: a bare
		// substring check would also match an unrelated internal error that
		// merely wraps the sentinel, leaking it unredacted.
		//
		// wire.ErrVectorArgsTruncated rides in the same bucket, by both arms, for
		// the same reasons: DecodeVectorOperateArgs raises it for a frame shorter
		// than the fields it declares, right beside the ErrOperateArgs it raises
		// for a complete-but-invalid one. Both are the caller's framing mistake
		// and both name nothing but "the arguments do not decode"; left
		// unclassified, a truncated frame read as a server fault.
		wire.IsOperateArgsMessage(err.Error()),
		errors.Is(err, wire.ErrOperateArgs),
		wire.IsVectorArgsTruncatedMessage(err.Error()),
		errors.Is(err, wire.ErrVectorArgsTruncated),
		// wire.ErrOperateCap: an operate call refused for exceeding one of the
		// design doc §2.7 caps — too many ops, too many return specs, returns
		// that would produce more bytes than OperateMaxRetBytes, or a record
		// the call would grow past the size backstop. Every one of those is a
		// fact about the request the caller itself built, with an obvious
		// remedy (ask for less), and the message names only "a cap was
		// exceeded" — no key, no path, no size. Same bucket and same reasoning
		// as the malformed frame above; unclassified, a caller asking for too
		// much read back as a server fault.
		//
		// Sentinel AND exact message shape, both arms load-bearing, for exactly
		// the reason spelled out above — and MORE so than for ErrOperateArgs. A
		// KV operate does not merely decode inside the FSM apply, it APPLIES
		// there, so every cap the engine enforces (not just the frame-level
		// ones) is raised behind shard.decodePBResult and reaches this
		// classifier rebuilt with errors.New. On a cluster the sentinel arm
		// covers almost none of them. wire.IsOperateCapMessage, not
		// strings.Contains, for the usual reason.
		errors.Is(err, wire.ErrOperateCap),
		wire.IsOperateCapMessage(err.Error()):
		return true
	case errors.Is(err, ops.ErrOperateDuringReshard),
		// ops.ErrOperateDuringReshard: a vector_operate against a collection a
		// reshard is dual-writing. operate is not idempotent, so the store
		// REFUSES rather than sending the op-list to both generations — and the
		// whole design rests on the caller being told to retry after cutover.
		// Unclassified it fell to the redacted internal-error bucket, so the
		// retryability never reached the client and a transient read as a server
		// fault. Same bucket as the leadership/ownership transients below (this
		// transport carries no separate retryable status: StatusNotLeader is
		// specifically a leader hint, and this is not a leadership condition).
		//
		// Sentinel AND exact message shape, for the usual clustered-apply reason;
		// ops.IsOperateDuringReshardMessage, not strings.Contains, so an internal
		// fault that merely mentions the refusal is not leaked.
		ops.IsOperateDuringReshardMessage(err.Error()):
		return true
	case errors.Is(err, vector.ErrInvalidDim),
		errors.Is(err, vector.ErrInvalidMetric),
		errors.Is(err, vector.ErrInvalidM),
		errors.Is(err, vector.ErrInvalidQuant),
		errors.Is(err, vector.ErrInvalidIVFPQ),
		errors.Is(err, vector.ErrInvalidIVFPQM),
		errors.Is(err, vector.ErrInvalidQuantPQM),
		errors.Is(err, vector.ErrInvalidOPQ),
		errors.Is(err, vector.ErrInvalidOPQIters),
		errors.Is(err, vector.ErrInvalidPQDropVecs),
		errors.Is(err, vector.ErrInvalidIVFTrainThreshold),
		errors.Is(err, vector.ErrInvalidIVFDriftFactor),
		errors.Is(err, vector.ErrInvalidAnisotropicEta),
		errors.Is(err, vector.ErrInvalidSOAR),
		errors.Is(err, vector.ErrInvalidSOARLambda),
		errors.Is(err, vector.ErrInvalidPQNBits):
		return true
	case errors.Is(err, vector.ErrVersionConflict),
		errors.Is(err, vector.ErrCollectionExists),
		errors.Is(err, vector.ErrDuplicateID),
		errors.Is(err, vector.ErrCollectionFull),
		errors.Is(err, vector.ErrCollectionRateLimited),
		errors.Is(err, vector.ErrNoNamed),
		errors.Is(err, vector.ErrAPIKeyExists),
		errors.Is(err, vector.ErrAPIKeyNotFound):
		return true
	// The kv_query refusals. Each is a fact about the CALLER'S query or about
	// this replica's readiness, and each names only things the caller already
	// sent — the index name, the filter field it chose — plus a shard-group
	// ordinal it can read out of __topology__ anyway.
	//
	// THEY MUST CROSS THE WIRE UNREDACTED, and not merely for a better message.
	// A kv_query fans out to every shard group, and a group this node does not
	// host is answered by a PEER through __kv_query_shard__ — so the peer's edge
	// is this classifier. Redacted to "internal error", a peer's refusal reaches
	// the coordinator with nothing left to classify: a permanent client mistake
	// and a retryable "that group is still installing the index" become the same
	// opaque string, and cluster.classifyKVQueryErr — whose whole job is telling
	// a create-then-query apart from a bad query — can then only do it for
	// locally hosted groups. Sentinel matching only, no message-shape arms:
	// these errors are raised by the handler this edge just called, in-process,
	// so their identity is intact here.
	case errors.Is(err, ops.ErrKVQueryFilter),
		errors.Is(err, ops.ErrKVQueryScanRequired),
		errors.Is(err, ops.ErrKVQueryScanBudget),
		errors.Is(err, ops.ErrKVIndexUnavailable),
		errors.Is(err, ops.ErrKVQueryUnavailable),
		errors.Is(err, kvindex.ErrNoSuchIndex),
		errors.Is(err, kvindex.ErrIndexBuilding),
		errors.Is(err, kvindex.ErrIndexChanged),
		errors.Is(err, kvindex.ErrCandidateBudget),
		errors.Is(err, wire.ErrKVQueryArgs),
		errors.Is(err, wire.ErrKVQueryResult),
		errors.Is(err, wire.ErrKVQueryArgsTruncated),
		// ops.ErrKVQueryCursorCap: the coordinator built a continuation larger
		// than the cap that applies on the way back in. Its message is the
		// per-group arithmetic the caller acts on, and redacted it says nothing.
		errors.Is(err, ops.ErrKVQueryCursorCap),
		// wire.ErrKVFilterBudget: the filter tree is over the node/depth cap.
		// A fact about the query the caller sent, and the cap is in the message.
		// Found missing by the shared-family parity test — httpapi and grpcapi
		// classified it while this edge redacted it, which is the exact drift
		// that test exists to catch.
		errors.Is(err, wire.ErrKVFilterBudget),
		// wire.ErrKVIndexDef: the index-ADMIN op refused a definition that fails
		// the full admission check (shape + path parse). Not a kv_query refusal,
		// but classified here because redaction would turn "your payload path is
		// not a legal path" into "internal error" — and because admitting such a
		// definition is what makes a kv_query retry forever. See
		// cluster.validateKVIndexDef. Its message names only the definition the
		// caller just sent.
		errors.Is(err, wire.ErrKVIndexDef):
		return true
	// shard.ErrStoreClosed: a Call refused because this store is draining for
	// close. It is a REFUSAL, not a fault — the op never ran, and in a cluster
	// the group's other replicas can serve it — so the caller must be told to
	// retry rather than handed the opaque "internal error" redaction gives an
	// unclassified sentinel. It names nothing but the condition itself.
	//
	// It matters most on the __kv_query_shard__ leg, where a peer's edge is THIS
	// classifier: redacted, a coordinator draining one replica mid-fan-out
	// cannot tell that refusal from a real fault, and a retryable page becomes a
	// hard error on every multi-node cluster. Matched by IDENTITY because this
	// package already imports shard.
	//
	// THE MESSAGE ARM IS NOT REDUNDANT, and identity alone was a bug: the case
	// this classification exists for is a PEER's refusal, which arrives here
	// stringified with its type gone (shard.decodePBResult and the peer client
	// both rebuild errors from text). Matched only by identity, a peer's
	// store-closed refusal was still redacted to "internal error" — leaving the
	// coordinator nothing to classify, which is the whole failure this arm was
	// added to prevent. Found by the shared-family parity test.
	//
	// ops.IsStoreClosedMessage is the same anchored matcher httpapi and grpcapi
	// use, so all three agree on what counts as this refusal.
	case errors.Is(err, shard.ErrStoreClosed),
		ops.IsStoreClosedMessage(err.Error()):
		return true
	}
	// Cross-boundary / cluster / routing signals matched by string so the clustered
	// (stringified-across-Raft) path is covered too. These carry no host-identifying
	// topology detail beyond the transient condition itself.
	// shard.ErrOpNotRegistered is matched by IDENTITY, not by substring: this
	// package already imports shard (see the shard.ErrNotLeader use above), so
	// there is no reason to depend on the sentinel's message text surviving a
	// rewording. httpapi genuinely cannot import shard/cluster and keeps its own
	// substring match. The substring below still runs, and is still needed: it
	// covers cluster.ErrUnknownOp and the CLUSTERED path, where the error has been
	// stringified across a Raft/RPC boundary and no longer carries any identity.
	if errors.Is(err, shard.ErrOpNotRegistered) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "not leader") ||
		strings.Contains(msg, "no leader") ||
		strings.Contains(msg, "no reachable owner") ||
		// cluster.ErrUnknownOp ("cluster: op not registered") and the stringified
		// forms of shard.ErrOpNotRegistered ("shard: op not registered") — one
		// substring covers both. The caller asked for an op name this server does
		// not have; the message only echoes back the name the caller already sent
		// and discloses no topology, path or host detail. Redacting it to "internal
		// error" hid the real cause in diagnostics, which is what this fixes.
		//
		// It does NOT fix retry behaviour, and an earlier version of this comment
		// wrongly claimed it did. Both branches of mapResult return StatusError —
		// only the payload text differs — and Client.Call rotates to another server
		// only on errNotLeader or isTransportError (client/client.go), so this error
		// is terminal at the first server either way. client/wasm.go documents it as
		// transient and safe to retry, so the caller currently carries that burden.
		// Making the client rotate needs a structured signal (a distinct status code
		// or an error-code field), not substring matching on the message.
		strings.Contains(msg, "op not registered") ||
		// cluster.ErrWASMUpdateUnsupported. A refusal of a __register_wasm__ that
		// would change a live op's CONTRACT — its Kind or its key extractor, the two
		// fields that are read before any shard group is known and therefore cannot
		// be bound to a group's log prefix. (Changing the MODULE in place is
		// supported and never reaches here.) A client mistake with a remedy the
		// caller can act on (register under a new name),
		// and the message discloses only the op name the caller already sent.
		// Redacting it to "internal error" would hide the one thing the caller
		// needs to know and make an unsupported operation look like a server fault.
		// The substring is a CONST, so rewording the refusal is a compile error
		// here rather than a silent regression to "internal error".
		strings.Contains(msg, ops.WASMUpdateUnsupportedMsg) ||
		// ops.ErrWASMOpNameUnsafe. A __register_wasm__ whose Name is not usable as
		// a bare filename (a path separator, "..", NUL). Same reasoning: a caller
		// mistake with an obvious remedy, and the message echoes back only the name
		// the caller sent.
		strings.Contains(msg, ops.WASMOpNameUnsafeMsg) ||
		// cluster.ErrWASMRegistrationRefused. A propose-time refusal of the
		// __register_wasm__ PAYLOAD: an encoded frame over the cap, a frame that does
		// not decode, a module over the cap, or a Kind byte outside {0,1}. Every one
		// is a caller mistake the caller can fix, and the message discloses only
		// sizes and the op name the caller already sent. Same CONST coupling as the
		// two above.
		strings.Contains(msg, ops.WASMRegistrationRefusedMsg) ||
		strings.Contains(msg, "version conflict") ||
		strings.Contains(msg, "unknown collection") ||
		strings.Contains(msg, "no collection") ||
		strings.Contains(msg, "already exists") ||
		strings.Contains(msg, "already present") ||
		strings.Contains(msg, "collection full") ||
		strings.Contains(msg, "rate limited") ||
		strings.Contains(msg, "rostam: alias ") ||
		strings.Contains(msg, "cluster: write ")
}
