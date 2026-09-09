// SPDX-License-Identifier: Apache-2.0

// Package grpcapi exposes Rostam's vector/RAG operations over gRPC. Like the
// httpapi package it is a thin transport over the same op dispatcher the binary
// TCP server uses: each RPC translates its request into the existing ops binary
// codec, calls Dispatcher.Call, and maps the binary result back into a protobuf
// response. No engine logic lives here. The recursive metadata filter and the
// tagged Value union ride as JSON strings (filter_json / metadata_json /
// key_json), reusing the JSON contract the wire codecs already use.
package grpcapi

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/grpcapi/grpcsvc"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// Dispatcher is the seam onto a backing store — the same interface the TCP and
// HTTP front ends use.
type Dispatcher interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}

// Authenticator authorizes a request. It is the unified RBAC authorizer
// (authz.Authenticator): each RPC builds an authz.AuthRequest{Token, Op, Args}
// (token from the gRPC "authorization" metadata, op + the binary op args it is
// about to dispatch) and the authorizer derives the (action, resource) and
// matches the principal's scopes. nil accepts every request.
type Authenticator = authz.Authenticator

// Server implements grpcsvc.VectorServiceServer over a Dispatcher.
type Server struct {
	grpcsvc.UnimplementedVectorServiceServer
	disp Dispatcher
	auth Authenticator
}

// NewServer builds a gRPC service over disp. A nil auth accepts every request.
func NewServer(disp Dispatcher, auth Authenticator) *Server {
	return &Server{disp: disp, auth: auth}
}

// Register attaches the service to a grpc.Server.
func (s *Server) Register(gs *grpc.Server) { grpcsvc.RegisterVectorServiceServer(gs, s) }

// authorize runs the optional Authenticator for (opName, args) using the bearer
// token from the request metadata, returning an Unauthenticated status on
// rejection. args is the binary op payload the RPC is about to dispatch; the
// authorizer derives the target collection from it for the per-collection scope
// check. The uniform "unauthorized" message never leaks token-unknown vs
// lacks-scope.
func (s *Server) authorize(ctx context.Context, opName string, args []byte) error {
	if s.auth == nil {
		return nil
	}
	if s.auth(authz.AuthRequest{Token: bearerToken(ctx), ClientCN: clientCN(ctx), Op: opName, Args: args}) {
		return nil
	}
	return status.Error(codes.Unauthenticated, "unauthorized")
}

// clientCN extracts the VERIFIED mTLS client-cert CommonName from the gRPC peer,
// or "" when the RPC is plaintext / presented no verified cert.
//
// SECURITY: we read tlsInfo.State.VerifiedChains — which crypto/tls populates
// ONLY after a chain is successfully verified against the server's ClientCAs —
// NOT tlsInfo.State.PeerCertificates, which carries the raw presented cert even
// when it was NOT verified. Using PeerCertificates would let a client present a
// self-signed cert with an arbitrary CN and have it accepted as a principal; the
// verified chain guarantees the CN belongs to a CA-signed identity. With
// ClientAuth=RequireAndVerifyClientCert the handshake already rejects an
// unverifiable cert; with VerifyClientCertIfGiven a no-cert client yields empty
// VerifiedChains → CN "" → token-or-deny fallback.
func clientCN(ctx context.Context) string {
	p, ok := peer.FromContext(ctx)
	if !ok || p.AuthInfo == nil {
		return ""
	}
	tlsInfo, ok := p.AuthInfo.(credentials.TLSInfo)
	if !ok {
		return ""
	}
	chains := tlsInfo.State.VerifiedChains
	if len(chains) == 0 || len(chains[0]) == 0 {
		return ""
	}
	return chains[0][0].Subject.CommonName
}

func bearerToken(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	vals := md.Get("authorization")
	if len(vals) == 0 {
		return ""
	}
	if after, ok := strings.CutPrefix(vals[0], "Bearer "); ok {
		return after
	}
	return vals[0]
}

// call authorizes then dispatches, mapping a dispatch error to a gRPC status.
func (s *Server) call(ctx context.Context, opName string, args []byte) ([]byte, error) {
	if err := s.authorize(ctx, opName, args); err != nil {
		return nil, err
	}
	res, err := s.disp.Call(opName, args)
	if err != nil {
		return nil, grpcError(err)
	}
	return res, nil
}

// boolToU8 maps a Go bool onto the __wc__ envelope wait byte (1 = wait, 0 =
// no-wait), matching the fanout dispatcher's `wait != 0` interpretation.
func boolToU8(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

// callWrite is the write-path variant of call: it engages tunable write
// consistency only when requested (wcf>0 OR wait=false), wrapping the inner op in
// the __wc__ envelope so the fanout dispatcher unwraps it, dispatches the inner
// write through normal routing/Raft, then runs the post-commit barrier. When
// neither knob is set (wcf=0, wait=true) it dispatches the plain op exactly as
// before — the proto-zero request is byte-identical to the pre-feature path.
// Authorization is always against the INNER op name (never "__wc__").
func (s *Server) callWrite(ctx context.Context, opName string, args []byte, wcf uint32, wait bool) ([]byte, error) {
	// Authorize against the INNER op name + INNER args (never the "__wc__"
	// envelope, which is built only after this passes), so a per-op+collection
	// scope behaves identically with or without write consistency.
	if err := s.authorize(ctx, opName, args); err != nil {
		return nil, err
	}
	name, payload := opName, args
	if wcf > 0 || !wait {
		name = ops.WCEnvelopeOp
		// Saturate to 255 before narrowing: a bare uint8(wcf) would WRAP (256→0),
		// silently turning an "all replicas" request into the no-barrier default.
		// Saturating keeps it the strongest factor (then clamped to RF at barrier).
		if wcf > 255 {
			wcf = 255
		}
		payload = ops.EncodeWCEnvelope(uint8(wcf), boolToU8(wait), opName, args) //nolint:gosec // saturated to 255 above; clamped to [1,RF] at the barrier
	}
	res, err := s.disp.Call(name, payload)
	if err != nil {
		return nil, grpcError(err)
	}
	return res, nil
}

// wcArgs is the common shape of the two tunable-write-consistency proto fields on
// every WRITE request. The field is no_wait (proto3 bool default false ⇒
// wait=true, the safe default); wait() inverts it. A uint32 factor cannot be
// negative, so no edge validation is needed (over-RF clamps at the barrier).
type wcArgs interface {
	GetWriteConsistencyFactor() uint32
	GetNoWait() bool
}

// wcOf reads (factor, wait) from any write request carrying the WC fields.
func wcOf(r wcArgs) (uint32, bool) { return r.GetWriteConsistencyFactor(), !r.GetNoWait() }

// grpcError classifies a dispatch error onto a gRPC status code, mirroring the
// HTTP front end's 400/404/503/500 mapping.
func grpcError(err error) error {
	switch {
	case errIs(err, vector.ErrDimMismatch, vector.ErrEmptyFilter, vector.ErrEmptyGroupBy,
		vector.ErrSparseMismatch, vector.ErrSparseUnsorted, vector.ErrSpaceModalityMismatch,
		vector.ErrUnknownVectorName, vector.ErrEmptyNamedVectors,
		vector.ErrReservedVectorName, vector.ErrEmptyVectorName,
		// TextSearch/HybridTextSearch on a collection without FullText: a usage error.
		vector.ErrFullTextDisabled):
		return status.Error(codes.InvalidArgument, err.Error())
	case errIs(err, vector.ErrInvalidDim, vector.ErrInvalidMetric, vector.ErrInvalidM,
		vector.ErrInvalidQuant, vector.ErrInvalidIVFPQ, vector.ErrInvalidIVFPQM,
		vector.ErrInvalidQuantPQM, vector.ErrInvalidOPQ, vector.ErrInvalidOPQIters,
		vector.ErrInvalidPQDropVecs,
		vector.ErrInvalidIVFTrainThreshold, vector.ErrInvalidIVFDriftFactor):
		// Create-collection config validation (Config.Validate): a bad dim/M/quant, or
		// a PQ-HNSW (quant=="pq") / IVF-PQ config the engine rejects at create (e.g.
		// quant=="pq" on an IVF index, or dim not divisible by quant_pq_m) is a client
		// mistake -> InvalidArgument, not Internal.
		return status.Error(codes.InvalidArgument, err.Error())
	case strings.Contains(err.Error(), "rostam: alias "):
		// Alias-management validation errors (target missing / shadow / reserved
		// char / target-is-alias) all carry the "rostam: alias " prefix. They live
		// in the root rostam package (un-importable here without a cycle), so match
		// the message prefix rather than the sentinel.
		return status.Error(codes.InvalidArgument, err.Error())
	case errIs(err, vector.ErrRecordTooLarge),
		// vector.ErrRecordTooLarge: a payload carrying a record value above the
		// storage cap. A caller mistake with a clear remedy (send a smaller
		// record) whose message names only the payload key and the two sizes,
		// all of which the caller sent — so InvalidArgument, mirroring
		// server.clientFacingErr and httpapi.statusForError's 400 bucket for the
		// same sentinel. Unclassified it fell to codes.Internal, which reads as a
		// server fault and which standard gRPC retry policies hammer.
		//
		// Matched by sentinel AND by exact message shape: shard.decodePBResult
		// rebuilds an op error with errors.New(string(payload)) across
		// replication, so a clustered apply loses errors.Is identity — the same
		// reason the other two classifiers carry a message-shape fallback. The
		// fallback uses vector.IsRecordTooLargeMessage, NOT strings.Contains — a
		// bare substring check would also match an unrelated internal error that
		// merely wraps the sentinel, leaking it to the caller unredacted.
		vector.IsRecordTooLargeMessage(err.Error()):
		return status.Error(codes.InvalidArgument, err.Error())
	case errIs(err, vector.ErrAPIKeyExists):
		// Online key-admin: KeysAdd of an already-registered token. AlreadyExists
		// (the etcd/gRPC create-conflict code) — the caller revokes first or picks a
		// new token.
		return status.Error(codes.AlreadyExists, err.Error())
	case errIs(err, vector.ErrAPIKeyNotFound):
		// Online key-admin: KeysRevoke of an unknown token. NotFound.
		return status.Error(codes.NotFound, err.Error())
	case strings.Contains(err.Error(), "rostam: online key-admin unavailable"):
		// Online key-admin requested but no *vector.KeyRegistry is wired (open/dev
		// mode or the static -api-key authenticator). ErrKeyAdminUnavailable lives in
		// the root rostam package (un-importable here without a cycle), so match the
		// message prefix. FailedPrecondition: the server is mis-/under-configured for
		// this op (start with -keys-file), not a transient outage or a bad request.
		return status.Error(codes.FailedPrecondition, err.Error())
	case errIs(err, vector.ErrNoNamed),
		strings.Contains(err.Error(), "unknown collection"),
		strings.Contains(err.Error(), "no collection"):
		return status.Error(codes.NotFound, err.Error())
	case errIs(err, vector.ErrVersionConflict), strings.Contains(err.Error(), "version conflict"):
		// Optimistic-CAS precondition miss: the point's current version did not match
		// the request's expected_version. FailedPrecondition (the standard etcd/gRPC
		// CAS code) — the caller re-reads the current version and retries. The string
		// fallback covers the clustered path where the sentinel is stringified across
		// the Raft FSM-Apply boundary (mirrors the "cluster: write " precedent).
		return status.Error(codes.FailedPrecondition, err.Error())
	case errIs(err, vector.ErrCollectionExists, vector.ErrDuplicateID),
		strings.Contains(err.Error(), "already exists"),
		strings.Contains(err.Error(), "already present"):
		// Routine create-conflicts, not server faults: a second CreateCollection for a
		// live name (ErrCollectionExists), or a default insert (upsert=false) of an id
		// that is already live (ErrDuplicateID). AlreadyExists is the standard
		// create-conflict code. String fallbacks cover the clustered path where the
		// sentinel is stringified across the Raft boundary (mirrors "version conflict").
		return status.Error(codes.AlreadyExists, err.Error())
	case errIs(err, vector.ErrCollectionRateLimited, vector.ErrCollectionFull),
		strings.Contains(err.Error(), "rate limited"),
		strings.Contains(err.Error(), "collection full"):
		// Quota/rate-limit backpressure, not a server fault: the write hit the
		// collection's MaxInsertsPerSecond token bucket (ErrCollectionRateLimited) or
		// its MaxVectors/MaxBytes cap (ErrCollectionFull). ResourceExhausted (the gRPC
		// backpressure code) tells the caller to back off and retry, distinct from
		// Internal (which retry policies hammer). String fallbacks cover the
		// clustered/stringified path.
		return status.Error(codes.ResourceExhausted, err.Error())
	case strings.Contains(err.Error(), "not leader"),
		strings.Contains(err.Error(), "no leader"),
		strings.Contains(err.Error(), "no reachable owner"):
		// Leadership/ownership-transient and RETRYABLE -> Unavailable, mirroring HTTP's
		// 503. Either this node is not the leader for the target shard ("not leader"),
		// the client-forwarding layer could not resolve a leader during an election
		// window ("no leader known after retries", client.ErrNoLeaderKnown), or no owner
		// for the target shard was reachable (cluster.ErrNoShardOwner, "no reachable
		// owner for shard"). Mapping these to Unavailable (not the default Internal, which
		// standard gRPC retry policies and service meshes will NOT retry) keeps the two
		// transports consistent so a write that transparently retries over REST does not
		// hard-fail over gRPC during the same transient window.
		return status.Error(codes.Unavailable, err.Error())
	case strings.Contains(err.Error(), "cluster: write "):
		// Write-consistency barrier miss (*cluster.ErrWriteConsistency, carried as a
		// "cluster: write " message prefix — the typed error lives in the cluster pkg,
		// matched by prefix like the alias precedent). The write IS durably committed
		// at Raft quorum; only the requested write_consistency_factor was not reached
		// within the timeout. FailedPrecondition (not DataLoss — the write is durable;
		// not Unavailable — the call succeeded at majority and a blind retry won't
		// necessarily reach the factor) signals the >majority precondition was unmet,
		// so the caller decides to accept majority or retry deliberately. Placed after
		// "not leader" so the more specific prefix wins and stays out of the
		// Unavailable/Internal buckets.
		return status.Error(codes.FailedPrecondition, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}

func errIs(err error, targets ...error) bool {
	for _, t := range targets {
		if errors.Is(err, t) {
			return true
		}
	}
	return false
}

// u16ToU32 widens a missing-partition slice for the protobuf wire (proto3 has no
// uint16). A nil/empty input yields nil so non-degraded responses stay empty.
func u16ToU32(in []uint16) []uint32 {
	if len(in) == 0 {
		return nil
	}
	out := make([]uint32, len(in))
	for i, v := range in {
		out[i] = uint32(v)
	}
	return out
}

// validConsistency rejects out-of-range consistency knobs at the edge: the
// enums only define 0 and 1, so any larger value is an InvalidArgument before
// dispatch. Returns nil when both are in range.
func validConsistency(rc, opa uint32) error {
	// read_consistency: 0=AnyReplica, 1=LeaderOnly, 2=Linearizable, 3=BoundedStaleness.
	// on_partition_unavailable: 0=Partial, 1=Fail.
	if rc > 3 {
		return status.Error(codes.InvalidArgument, "read_consistency must be 0 (any), 1 (leader), 2 (linearizable) or 3 (bounded-staleness)")
	}
	if opa > 1 {
		return status.Error(codes.InvalidArgument, "on_partition_unavailable must be 0 or 1")
	}
	return nil
}

// ---- request/response bridging ----

// Health is a DELIBERATE, documented auth exemption: it dispatches __ping__
// directly WITHOUT calling authorize, so an unauthenticated liveness/readiness
// probe (k8s, load balancer) succeeds even when RBAC is enabled. It returns only
// a static "ok" liveness signal and touches no collection data, so exempting it
// leaks nothing. This is the ONLY client-facing gRPC path that skips authorize;
// every other RPC goes through call/callWrite. (Note: __ping__ maps to the
// "read" action in the authorizer, so an authenticated reader could also reach
// it via a future generic ping RPC; the exemption here is specifically to allow
// an UNAUTHENTICATED probe.)
func (s *Server) Health(ctx context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	if _, err := s.disp.Call("__ping__", nil); err != nil {
		return nil, grpcError(err)
	}
	return &pb.HealthResponse{Status: "ok"}, nil
}

func (s *Server) CreateCollection(ctx context.Context, req *pb.CreateCollectionRequest) (*pb.CreateCollectionResponse, error) {
	if strings.ContainsAny(req.GetName(), "#@") {
		return nil, status.Errorf(codes.InvalidArgument, "vector: collection name %q must not contain reserved characters '#' or '@'", req.GetName())
	}
	cfg, err := toConfig(req.GetConfig())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	if _, err := s.call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(req.GetName(), cfg)); err != nil {
		return nil, err
	}
	return &pb.CreateCollectionResponse{Name: req.GetName()}, nil
}

func (s *Server) DropCollection(ctx context.Context, req *pb.DropCollectionRequest) (*pb.DropCollectionResponse, error) {
	if _, err := s.call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(req.GetName())); err != nil {
		return nil, err
	}
	return &pb.DropCollectionResponse{Name: req.GetName()}, nil
}

// Resplit re-partitions a dense collection. It is synchronous and offline: the
// caller must quiesce writes for the duration and should set a long deadline,
// since the whole collection is re-indexed before the RPC returns. The
// new_partitions < 0 reject here is defense-in-depth at the edge — the embedded
// engine's range guard ([2, 65536]) is authoritative — so a clearly-typed
// InvalidArgument is returned instead of a wrapped overflow error.
func (s *Server) Resplit(ctx context.Context, req *pb.ResplitRequest) (*pb.ResplitResponse, error) {
	if req.GetNewPartitions() < 0 {
		return nil, status.Error(codes.InvalidArgument, "new_partitions must be non-negative")
	}
	if _, err := s.call(ctx, "vector_resplit", ops.EncodeResplitArgs(req.GetName(), int(req.GetNewPartitions()))); err != nil {
		return nil, err
	}
	return &pb.ResplitResponse{Name: req.GetName(), NewPartitions: req.GetNewPartitions()}, nil
}

// ResplitCleanup drops the orphaned pre-resplit partitions left behind by a
// completed Resplit, returning the number dropped. Run it after a resplit has
// been verified healthy.
func (s *Server) ResplitCleanup(ctx context.Context, req *pb.ResplitCleanupRequest) (*pb.ResplitCleanupResponse, error) {
	body, err := s.call(ctx, "vector_resplit_cleanup", ops.EncodeResplitCleanupArgs(req.GetName()))
	if err != nil {
		return nil, err
	}
	dropped, err := ops.DecodeResplitCleanupResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.ResplitCleanupResponse{Name: req.GetName(), Dropped: int32(dropped)}, nil //nolint:gosec // count fits int32
}

// Reshard re-partitions a dense collection ONLINE. Unlike Resplit, reads AND
// writes stay live for the duration (the orchestrator dual-writes during a
// streamed if-absent copy, then flips the catalog at cutover); the call still
// blocks until cutover so the caller should set a long deadline. The
// new_partitions < 0 reject here is defense-in-depth at the edge — the embedded
// engine's range guard ([2, 65536]) is authoritative.
func (s *Server) Reshard(ctx context.Context, req *pb.ReshardRequest) (*pb.ReshardResponse, error) {
	if req.GetNewPartitions() < 0 {
		return nil, status.Error(codes.InvalidArgument, "new_partitions must be non-negative")
	}
	if _, err := s.call(ctx, "vector_reshard", ops.EncodeReshardArgs(req.GetName(), int(req.GetNewPartitions()))); err != nil {
		return nil, err
	}
	return &pb.ReshardResponse{Name: req.GetName(), NewPartitions: req.GetNewPartitions()}, nil
}

// ReshardAbort aborts an in-flight dense reshard, restoring the old generation
// and dropping the new-gen partitions. Pre-cutover only — it errors if the
// reshard has already flipped.
func (s *Server) ReshardAbort(ctx context.Context, req *pb.ReshardAbortRequest) (*pb.ReshardAbortResponse, error) {
	if _, err := s.call(ctx, "vector_reshard_abort", ops.EncodeReshardAbortArgs(req.GetName())); err != nil {
		return nil, err
	}
	return &pb.ReshardAbortResponse{Name: req.GetName()}, nil
}

func (s *Server) Upsert(ctx context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	meta, err := parseMetadata(req.GetMetadataJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "metadata_json: "+err.Error())
	}
	ttl := time.Duration(req.GetTtlMs()) * time.Millisecond
	sparse := vector.SparseVector{Indices: req.GetSparseIndices(), Values: req.GetSparseValues()}
	wcf, wait := wcOf(req)
	exp, hasExp := req.GetExpectedVersion(), req.GetHasExpectedVersion()
	keyTTL := req.GetKeyTtlMs() // per-key payload TTL (relative ms); empty = none
	if req.GetUpsert() {
		args := ops.EncodeVectorUpsertArgsCASKeyTTL(req.GetCollection(), req.GetId(), req.GetVector(), req.GetContent(), ttl, meta, sparse, exp, hasExp, keyTTL)
		if _, err := s.callWrite(ctx, "vector_upsert", args, wcf, wait); err != nil {
			return nil, err
		}
	} else {
		args := ops.EncodeVectorInsertArgsCASKeyTTL(req.GetCollection(), req.GetId(), req.GetVector(), ttl, vector.WithContent(meta, req.GetContent()), sparse, exp, hasExp, keyTTL)
		if _, err := s.callWrite(ctx, "vector_insert", args, wcf, wait); err != nil {
			return nil, err
		}
	}
	return &pb.UpsertResponse{Id: req.GetId()}, nil
}

func (s *Server) Delete(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteResponse, error) {
	wcf, wait := wcOf(req)
	body, err := s.callWrite(ctx, "vector_delete", ops.EncodeVectorDeleteArgsCAS(req.GetCollection(), req.GetId(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), wcf, wait)
	if err != nil {
		return nil, err
	}
	return &pb.DeleteResponse{Deleted: len(body) > 0 && body[0] == 1}, nil
}

// getFlags maps the two with_* projection booleans onto the ops get-flags byte.
// Both default to ON: a caller leaving the bools at their proto-zero (false)
// still gets both projections, matching the HTTP edge default. To request only
// one projection a caller sets exactly that bool true and the other false; if
// BOTH are false (the zero request) we treat it as "fetch everything".
func getFlags(withVector, withPayload bool) uint8 {
	if !withVector && !withPayload {
		return ops.GetFlagsBoth
	}
	var f uint8
	if withVector {
		f |= ops.GetFlagWithVector
	}
	if withPayload {
		f |= ops.GetFlagWithPayload
	}
	return f
}

func (s *Server) Get(ctx context.Context, req *pb.GetRequest) (*pb.GetResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	flags := getFlags(req.GetWithVector(), req.GetWithPayload())
	body, err := s.call(ctx, "vector_get", ops.EncodeVectorGetArgsOpts(req.GetCollection(), req.GetId(), flags, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	found, vec, meta, ttl, _, version, err := ops.DecodeVectorGetResultV(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	if !found {
		return nil, status.Errorf(codes.NotFound, "point %d not found in collection %q", req.GetId(), req.GetCollection())
	}
	return &pb.GetResponse{
		Found: true, Vector: vec,
		PayloadJson: metadataJSON(meta), TtlMs: ttl.Milliseconds(),
		Version: version,
	}, nil
}

// GetBatch retrieves MANY dense points by id in ONE op. Mirrors the MVScroll
// fan-edge pattern: encode the vector_get_batch args + s.call (the fanout
// dispatcher's fanGetBatch runs the scatter-by-partition + merge), then decode
// the unified rows and split them into found points + the missing ids. Unlike
// single Get, a partial miss is NEVER a NotFound error — absent ids are returned
// in the missing list (200/ok). Empty ids → empty response. The per-point rows
// carry their id so the caller knows which point is which.
func (s *Server) GetBatch(ctx context.Context, req *pb.GetBatchRequest) (*pb.GetBatchResponse, error) {
	flags := getFlags(req.GetWithVector(), req.GetWithPayload())
	body, err := s.call(ctx, "vector_get_batch", ops.EncodeVectorGetBatchArgs(req.GetCollection(), req.GetIds(), flags))
	if err != nil {
		return nil, err
	}
	rows, err := ops.DecodeVectorGetBatchResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	resp := &pb.GetBatchResponse{}
	for i := range rows {
		row := &rows[i]
		if !row.Found {
			resp.Missing = append(resp.Missing, row.ID)
			continue
		}
		resp.Points = append(resp.Points, &pb.BatchGetPoint{
			Id:          row.ID,
			Vector:      row.Vec,
			PayloadJson: metadataJSON(row.Meta),
			TtlMs:       int64(row.TTLMs), //nolint:gosec // TTL ms >= 0
			Version:     row.Version,
		})
	}
	return resp, nil
}

// applyPayload dispatches a payload-mutation op and maps the applied flag: an
// absent point (applied=false) becomes NotFound (mirroring the HTTP 404). The op
// itself never errors on not-found — applied is a flag, never an op error.
func (s *Server) applyPayload(ctx context.Context, opName, collection string, id uint64, args []byte, wcf uint32, wait bool) (*pb.PayloadResponse, error) {
	body, err := s.callWrite(ctx, opName, args, wcf, wait)
	if err != nil {
		return nil, err
	}
	applied, err := ops.DecodePayloadResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	if !applied {
		return nil, status.Errorf(codes.NotFound, "point %d not found in collection %q", id, collection)
	}
	return &pb.PayloadResponse{Applied: true}, nil
}

func (s *Server) SetPayload(ctx context.Context, req *pb.SetPayloadRequest) (*pb.PayloadResponse, error) {
	meta, err := parseMetadata(req.GetPayloadJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "payload_json: "+err.Error())
	}
	return s.applyPayload(ctx, "vector_set_payload", req.GetCollection(), req.GetId(), ops.EncodeSetPayloadArgsCAS(req.GetCollection(), req.GetId(), meta, req.GetKeyTtlMs(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) OverwritePayload(ctx context.Context, req *pb.SetPayloadRequest) (*pb.PayloadResponse, error) {
	meta, err := parseMetadata(req.GetPayloadJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "payload_json: "+err.Error())
	}
	return s.applyPayload(ctx, "vector_overwrite_payload", req.GetCollection(), req.GetId(), ops.EncodeSetPayloadArgsCAS(req.GetCollection(), req.GetId(), meta, req.GetKeyTtlMs(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) DeletePayloadKeys(ctx context.Context, req *pb.DeletePayloadKeysRequest) (*pb.PayloadResponse, error) {
	return s.applyPayload(ctx, "vector_delete_payload_keys", req.GetCollection(), req.GetId(), ops.EncodeDeletePayloadKeysArgsCAS(req.GetCollection(), req.GetId(), req.GetKeys(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) ClearPayload(ctx context.Context, req *pb.ClearPayloadRequest) (*pb.PayloadResponse, error) {
	return s.applyPayload(ctx, "vector_clear_payload", req.GetCollection(), req.GetId(), ops.EncodeClearPayloadArgsCAS(req.GetCollection(), req.GetId(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	body, err := s.call(ctx, "vector_search", ops.EncodeVectorSearchArgsOpts(req.GetCollection(), int(req.GetK()), req.GetQuery(), filter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	results, degraded, missing, err := ops.DecodeVectorSearchResultsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchResponse{Results: toPBResults(results), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

func (s *Server) SearchDocs(ctx context.Context, req *pb.SearchRequest) (*pb.SearchDocsResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	body, err := s.call(ctx, "vector_search_docs", ops.EncodeVectorSearchArgsOpts(req.GetCollection(), int(req.GetK()), req.GetQuery(), filter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchDocsResponse{Documents: toPBDocs(docs), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

func (s *Server) SearchGroups(ctx context.Context, req *pb.SearchGroupsRequest) (*pb.SearchGroupsResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	opts := vector.GroupOpts{
		GroupBy: req.GetGroupBy(), GroupSize: int(req.GetGroupSize()),
		FetchK: int(req.GetFetchK()), Filter: filter,
	}
	body, err := s.call(ctx, "vector_search_groups", ops.EncodeGroupSearchArgsOpts(req.GetCollection(), int(req.GetK()), req.GetQuery(), opts, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	groups, degraded, missing, err := ops.DecodeGroupsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.Group, len(groups))
	for i, g := range groups {
		key, _ := json.Marshal(g.Key)
		out[i] = &pb.Group{KeyJson: string(key), Hits: toPBDocs(g.Hits)}
	}
	return &pb.SearchGroupsResponse{Groups: out, Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

func (s *Server) HybridSearch(ctx context.Context, req *pb.HybridRequest) (*pb.SearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	method, err := parseFusion(req.GetMethod())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	sparse := vector.SparseVector{Indices: req.GetSparseIndices(), Values: req.GetSparseValues()}
	opts := vector.HybridOpts{
		Filter: filter, Method: method, Alpha: req.GetAlpha(),
		RRFK: int(req.GetRrfK()), DenseK: int(req.GetDenseK()), SparseK: int(req.GetSparseK()),
	}
	body, err := s.call(ctx, "vector_hybrid_search", ops.EncodeHybridSearchArgsOpts(req.GetCollection(), req.GetDense(), int(req.GetK()), sparse, opts, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	results, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchResponse{Results: toPBResults(results), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// TextSearch runs a BM25 full-text search (vector_search_text): the server
// tokenizes + scores the raw query text and returns enriched documents. Mirrors
// SearchDocs; requires a FullText collection (else InvalidArgument via the
// ErrFullTextDisabled mapping in errToStatus).
func (s *Server) TextSearch(ctx context.Context, req *pb.TextSearchRequest) (*pb.SearchDocsResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	body, err := s.call(ctx, "vector_search_text", ops.EncodeSearchTextArgsGlobal(req.GetCollection(), req.GetText(), int(req.GetK()), filter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness(), req.GetGlobalIdf(), nil))
	if err != nil {
		return nil, err
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchDocsResponse{Documents: toPBDocs(docs), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// HybridTextSearch fuses a dense KNN lane with a BM25 full-text lane
// (vector_hybrid_text). The text lane is raw query text analyzed server-side.
// Mirrors HybridSearch.
func (s *Server) HybridTextSearch(ctx context.Context, req *pb.HybridTextRequest) (*pb.SearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	method, err := parseFusion(req.GetMethod())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	opts := vector.HybridOpts{
		Filter: filter, Method: method, Alpha: req.GetAlpha(),
		RRFK: int(req.GetRrfK()), DenseK: int(req.GetDenseK()), SparseK: int(req.GetSparseK()),
	}
	body, err := s.call(ctx, "vector_hybrid_text", ops.EncodeHybridTextArgsGlobal(req.GetCollection(), req.GetDense(), req.GetText(), int(req.GetK()), opts, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness(), req.GetGlobalIdf(), nil))
	if err != nil {
		return nil, err
	}
	results, degraded, missing, err := ops.DecodeHybridResultsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchResponse{Results: toPBResults(results), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// VectorQuery runs the unified Query API (Qdrant-parity): a QuerySpec (root +
// prefetch leaves + FUSION/RERANK mode + fusion config) over the dense family.
// It validates the spec at the edge (fusion method incl. dbsf, mode, per-leaf
// filter JSON — fail-loud → InvalidArgument), marshals it into the vector_query
// op's spec blob, dispatches, and decodes the coordinator's flat fused/reranked
// top-k WITH the degraded/missing trailer (mirroring HybridSearch). For a
// partitioned collection the fan-out dispatcher fans vector_query to every
// partition and re-encodes the merged top-k + FanMeta; for P=1 the per-shard
// handler returns the same flat shape.
func (s *Server) VectorQuery(ctx context.Context, req *pb.VectorQueryRequest) (*pb.SearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	specBytes, err := ops.ValidateAndMarshalQuerySpec(req.GetSpec())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	body, err := s.call(ctx, "vector_query", ops.EncodeQueryArgs(req.GetCollection(), specBytes, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	// GROUPED query (spec.group_by set): the dispatcher returns the grouped result
	// (decoded via DecodeGroupsDegraded into SearchResponse.groups), mirroring
	// SearchGroups; the flat results path is unchanged when group_by is empty.
	if req.GetSpec().GetGroupBy() != "" {
		groups, degraded, missing, gerr := ops.DecodeGroupsDegraded(body)
		if gerr != nil {
			return nil, status.Error(codes.Internal, gerr.Error())
		}
		out := make([]*pb.Group, len(groups))
		for i, g := range groups {
			key, _ := json.Marshal(g.Key)
			out[i] = &pb.Group{KeyJson: string(key), Hits: toPBDocs(g.Hits)}
		}
		return &pb.SearchResponse{Groups: out, Degraded: degraded, Missing: u16ToU32(missing)}, nil
	}
	results, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchResponse{Results: toPBResults(results), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// NamedVectorQuery runs the unified Query API over a NAMED collection: the same
// QuerySpec shape as VectorQuery, but EVERY leaf targets a named vector space
// (NamedDenseLeaf / NamedSparseLeaf) so the N prefetch lanes fuse across >2 named
// spaces (the distinctive multi-space value). It validates the spec at the edge
// (fusion method incl. dbsf, mode, per-leaf filter JSON, and — fail-loud — that
// EVERY leaf carries a non-empty space → InvalidArgument), marshals it into the
// vector_named_query op's spec blob, dispatches, and decodes the coordinator's
// flat fused/reranked top-k WITH the degraded/missing trailer (mirroring
// VectorQuery / NamedHybridSearch). For a partitioned named collection the fan-out
// dispatcher (fanNamedQuery → namedQueryFanOut) fans vector_named_query to every
// partition and re-encodes the merged top-k + FanMeta; for P=1 the local merge
// returns the same flat shape.
func (s *Server) NamedVectorQuery(ctx context.Context, req *pb.NamedVectorQueryRequest) (*pb.SearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	// Fail-loud at the edge: a named query requires EVERY leaf (root included when it
	// carries a payload) to target a named space — a Space-less leaf is a malformed
	// named request, never silently routed to a default space. (The engine enforces
	// the same invariant; checking here yields InvalidArgument directly instead of a
	// generic Internal from the op call.)
	if err := validateNamedSpaces(req.GetSpec()); err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	specBytes, err := ops.ValidateAndMarshalQuerySpec(req.GetSpec())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	body, err := s.call(ctx, "vector_named_query", ops.EncodeQueryArgs(req.GetCollection(), specBytes, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	results, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchResponse{Results: toPBResults(results), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// MVVectorQuery runs the unified Query API over a MULTI-VECTOR (late-interaction)
// collection: the same QuerySpec shape as VectorQuery / NamedVectorQuery, but the
// leaves carry MV-family payloads — a MaxSim leaf (the token query matrix, via the
// mv_maxsim oneof arm) and/or the doc-level sparse field (the sparse arm, no
// space). Both MV lanes are score-descending. It validates the spec at the edge
// (fusion method incl. dbsf, mode, per-leaf filter JSON, and the MV leaf payloads —
// ValidateAndMarshalQuerySpec handles the mv_maxsim arm since the MV proto leaf was
// marshals it into the vector_mv_query op's spec blob, dispatches,
// and decodes the coordinator's flat fused/reranked top-k WITH the degraded/missing
// trailer (mirroring VectorQuery / NamedVectorQuery / MVHybridSearch). For a
// partitioned MV collection the fan-out dispatcher (fanMVQuery → mvQueryFanOut) fans
// vector_mv_query to every partition and re-encodes the merged top-k + FanMeta; for
// P=1 the local merge returns the same flat shape.
func (s *Server) MVVectorQuery(ctx context.Context, req *pb.MVVectorQueryRequest) (*pb.SearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	specBytes, err := ops.ValidateAndMarshalQuerySpec(req.GetSpec())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	body, err := s.call(ctx, "vector_mv_query", ops.EncodeQueryArgs(req.GetCollection(), specBytes, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	results, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.SearchResponse{Results: toPBResults(results), Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// validateNamedSpaces fails loud if any leaf of a named QuerySpec lacks a target
// space. A named-family leaf is encoded as a NamedDense/NamedSparse oneof arm (the
// dense-family Dense/Sparse arms carry NO space and are therefore invalid here),
// so a leaf that is not a named arm — or is a named arm with an empty space — is
// rejected. The root is checked only when it carries a payload (a FUSION spec's
// empty root needs no space, matching (*NamedCollection).Query).
func validateNamedSpaces(spec *pb.QuerySpec) error {
	if spec == nil {
		return errors.New("named query: nil spec")
	}
	for _, leaf := range spec.GetPrefetch() {
		if err := leafHasNamedSpace(leaf); err != nil {
			return err
		}
	}
	if root := spec.GetRoot(); rootHasPayload(root) {
		if err := leafHasNamedSpace(root); err != nil {
			return err
		}
	}
	return nil
}

// rootHasPayload reports whether a root leaf carries an actual query vector (so a
// FUSION spec's empty root is treated as "no root" and exempt from the space
// check, mirroring the engine).
func rootHasPayload(leaf *pb.QueryLeaf) bool {
	switch l := leaf.GetLeaf().(type) {
	case *pb.QueryLeaf_Dense:
		return len(l.Dense.GetDense()) > 0
	case *pb.QueryLeaf_Sparse:
		return len(l.Sparse.GetIndices()) > 0 || len(l.Sparse.GetValues()) > 0
	case *pb.QueryLeaf_NamedDense:
		return len(l.NamedDense.GetDense()) > 0
	case *pb.QueryLeaf_NamedSparse:
		return len(l.NamedSparse.GetIndices()) > 0 || len(l.NamedSparse.GetValues()) > 0
	default:
		return false
	}
}

// leafHasNamedSpace requires a leaf to be a named-family arm with a non-empty
// space; a dense-family arm (or an empty space) is a malformed named query.
func leafHasNamedSpace(leaf *pb.QueryLeaf) error {
	switch l := leaf.GetLeaf().(type) {
	case *pb.QueryLeaf_NamedDense:
		if l.NamedDense.GetSpace() == "" {
			return errors.New("named query: dense leaf is missing its space")
		}
	case *pb.QueryLeaf_NamedSparse:
		if l.NamedSparse.GetSpace() == "" {
			return errors.New("named query: sparse leaf is missing its space")
		}
	default:
		return errors.New("named query: every leaf must target a named space")
	}
	return nil
}

func (s *Server) DeleteByFilter(ctx context.Context, req *pb.DeleteByFilterRequest) (*pb.DeleteByFilterResponse, error) {
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	wcf, wait := wcOf(req)
	body, err := s.callWrite(ctx, "vector_delete_by_filter", ops.EncodeDeleteByFilterArgs(req.GetCollection(), filter), wcf, wait)
	if err != nil {
		return nil, err
	}
	n, err := ops.DecodeDeleteByFilterResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.DeleteByFilterResponse{Deleted: int32(n)}, nil //nolint:gosec // count fits int32
}

// ---- late interaction (multi-vector / MaxSim) ----

func tokensToMatrix(rows []*pb.TokenVector) [][]float32 {
	out := make([][]float32, len(rows))
	for i, r := range rows {
		out[i] = r.GetValues()
	}
	return out
}

func (s *Server) MVCreateCollection(ctx context.Context, req *pb.MVCreateRequest) (*pb.MVCreateResponse, error) {
	if strings.ContainsAny(req.GetName(), "#@") {
		return nil, status.Errorf(codes.InvalidArgument, "vector: collection name %q must not contain reserved characters '#' or '@'", req.GetName())
	}
	c := req.GetConfig()
	quant, err := parseQuant(c.GetQuant())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	if c.GetPartitions() < 0 {
		return nil, status.Error(codes.InvalidArgument, "partitions must be non-negative")
	}
	indexType, err := parseIndexType(c.GetIndexType())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	if c.GetIvfNlist() < 0 || c.GetIvfNprobe() < 0 {
		return nil, status.Error(codes.InvalidArgument, "ivf_nlist and ivf_nprobe must be non-negative")
	}
	if c.GetIvfDriftGrowthFactor() != 0 && c.GetIvfDriftGrowthFactor() <= 1.0 {
		return nil, status.Error(codes.InvalidArgument, "ivf_drift_growth_factor must be > 1.0")
	}
	if c.GetIvfDriftFactor() != 0 && c.GetIvfDriftFactor() <= 1.0 {
		return nil, status.Error(codes.InvalidArgument, "ivf_drift_factor must be > 1.0")
	}
	if c.GetFilterFirstRelativeBp() < 0 || c.GetFilterFirstRelativeBp() > 10000 {
		return nil, status.Error(codes.InvalidArgument, "filter_first_relative_bp must be in [0, 10000]")
	}
	if c.GetOpqIters() < 0 || c.GetOpqIters() > 20 {
		return nil, status.Error(codes.InvalidArgument, "opq_iters must be in [0, 20]")
	}
	cfg := vector.MultiVectorConfig{
		Dim: int(c.GetDim()), M: int(c.GetM()), EfConstruction: int(c.GetEfConstruction()),
		EfSearch: int(c.GetEfSearch()), Seed: c.GetSeed(),
		Quant: quant, RescoreFactor: int(c.GetRescoreFactor()), Persistent: c.GetPersistent(),
		Partitions:            int(c.GetPartitions()),
		IndexType:             indexType,
		IVFNlist:              int(c.GetIvfNlist()),
		IVFNprobe:             int(c.GetIvfNprobe()),
		IVFPQ:                 c.GetIvfPq(),
		IVFPQM:                int(c.GetIvfPqM()),
		IVFRerank:             c.GetIvfRerank(),
		OPQ:                   c.GetOpq(),
		OPQIters:              int(c.GetOpqIters()),
		IVFTrainThreshold:     int(c.GetIvfTrainThreshold()),
		PQDropVecs:            c.GetPqDropVecs(),
		IVFDriftRetrain:       c.GetIvfDriftRetrain(),
		IVFDriftGrowthFactor:  c.GetIvfDriftGrowthFactor(),
		IVFDriftFactor:        c.GetIvfDriftFactor(),
		FilterFirstRelativeBP: int(c.GetFilterFirstRelativeBp()),
	}
	if _, err := s.call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(req.GetName(), cfg)); err != nil {
		return nil, err
	}
	return &pb.MVCreateResponse{Name: req.GetName()}, nil
}

func (s *Server) MVDropCollection(ctx context.Context, req *pb.MVDropRequest) (*pb.MVDropResponse, error) {
	if _, err := s.call(ctx, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(req.GetName(), 0)); err != nil {
		return nil, err
	}
	return &pb.MVDropResponse{Name: req.GetName()}, nil
}

func (s *Server) MVAdd(ctx context.Context, req *pb.MVAddRequest) (*pb.MVAddResponse, error) {
	meta, err := parseMetadata(req.GetMetadataJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "metadata_json: "+err.Error())
	}
	// key_ttl_ms (per-key payload TTL, relative ms); empty = no per-key TTL. The
	// OPTIONAL doc-level sparse vector (empty indices/values ⇒ dense-only, no add
	// trailer) rides last on the wire.
	var sparse *vector.SparseVector
	if len(req.GetSparseIndices()) > 0 || len(req.GetSparseValues()) > 0 {
		sparse = &vector.SparseVector{Indices: req.GetSparseIndices(), Values: req.GetSparseValues()}
	}
	args := ops.EncodeMVAddArgsCASKeyTTLSparse(req.GetName(), req.GetId(), tokensToMatrix(req.GetTokens()), meta, req.GetExpectedVersion(), req.GetHasExpectedVersion(), req.GetKeyTtlMs(), sparse)
	wcf, wait := wcOf(req)
	if _, err := s.callWrite(ctx, "vector_mv_add", args, wcf, wait); err != nil {
		return nil, err
	}
	return &pb.MVAddResponse{Id: req.GetId()}, nil
}

func (s *Server) MVSearch(ctx context.Context, req *pb.MVSearchRequest) (*pb.MVSearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	args := ops.EncodeMVSearchArgsOptsFilter(req.GetName(), tokensToMatrix(req.GetQuery()), int(req.GetK()), int(req.GetCandidatesPerToken()), uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), filter, req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_mv_search", args)
	if err != nil {
		return nil, err
	}
	results, degraded, missing, err := ops.DecodeMVResultsDegraded(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.MVResult, len(results))
	for i, r := range results {
		out[i] = &pb.MVResult{Id: r.ID, Score: r.Score, MetadataJson: metadataJSON(r.Metadata)}
	}
	return &pb.MVSearchResponse{Results: out, Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// MVHybridSearch fuses an MV collection's MaxSim lane (the token query matrix) and
// its per-doc sparse lane (sparse_indices/values) into the top-k. Mirrors
// NamedHybridSearch (fused id+distance+score in NamedSearchResponse), with the query
// as an MV token matrix instead of a single dense vector + a sparse named space.
func (s *Server) MVHybridSearch(ctx context.Context, req *pb.MVHybridRequest) (*pb.NamedSearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	method, err := parseFusion(req.GetMethod())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	sparse := vector.SparseVector{Indices: req.GetSparseIndices(), Values: req.GetSparseValues()}
	opts := vector.HybridOpts{
		Filter: filter, Method: method, Alpha: req.GetAlpha(),
		RRFK: int(req.GetRrfK()), DenseK: int(req.GetDenseK()), SparseK: int(req.GetSparseK()),
	}
	args := ops.EncodeMVHybridArgs(req.GetName(), tokensToMatrix(req.GetQuery()), sparse, int(req.GetK()), opts, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_mv_hybrid_search", args)
	if err != nil {
		return nil, err
	}
	results, err := ops.DecodeHybridResults(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.NamedResult, len(results))
	for i, r := range results {
		out[i] = &pb.NamedResult{Id: r.ID, Distance: r.Distance, Score: r.Score}
	}
	return &pb.NamedSearchResponse{Results: out}, nil
}

func (s *Server) MVDelete(ctx context.Context, req *pb.MVDeleteRequest) (*pb.MVDeleteResponse, error) {
	wcf, wait := wcOf(req)
	body, err := s.callWrite(ctx, "vector_mv_delete", ops.EncodeMVDeleteArgsCAS(req.GetName(), req.GetId(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), wcf, wait)
	if err != nil {
		return nil, err
	}
	return &pb.MVDeleteResponse{Deleted: len(body) > 0 && body[0] == 1}, nil
}

func (s *Server) MVGet(ctx context.Context, req *pb.MVGetRequest) (*pb.MVGetResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	flags := getFlags(req.GetWithVector(), req.GetWithPayload())
	body, err := s.call(ctx, "vector_mv_get", ops.EncodeVectorGetArgsOpts(req.GetCollection(), req.GetId(), flags, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	found, tokens, payload, version, err := ops.DecodeMVGetResultV(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	if !found {
		return nil, status.Errorf(codes.NotFound, "document %d not found in collection %q", req.GetId(), req.GetCollection())
	}
	out := make([]*pb.TokenVector, len(tokens))
	for i, tok := range tokens {
		out[i] = &pb.TokenVector{Values: tok}
	}
	return &pb.MVGetResponse{Found: true, Tokens: out, PayloadJson: metadataJSON(payload), Version: version}, nil
}

// MVGetBatch retrieves MANY multi-vector documents by id in ONE op. Like
// NamedGetBatch (and unlike single MVGet) a partial miss is NEVER a NotFound
// error — absent ids are returned in the missing list (ok). Each present point
// carries its token matrix + payload (MV has NO ttl). The MV clone of
// NamedGetBatch.
func (s *Server) MVGetBatch(ctx context.Context, req *pb.MVGetBatchRequest) (*pb.MVGetBatchResponse, error) {
	flags := getFlags(req.GetWithVector(), req.GetWithPayload())
	body, err := s.call(ctx, "vector_mv_get_batch", ops.EncodeVectorGetBatchArgs(req.GetCollection(), req.GetIds(), flags))
	if err != nil {
		return nil, err
	}
	rows, err := ops.DecodeMVGetBatchResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	resp := &pb.MVGetBatchResponse{}
	for i := range rows {
		row := &rows[i]
		if !row.Found {
			resp.Missing = append(resp.Missing, row.ID)
			continue
		}
		out := make([]*pb.TokenVector, len(row.Tokens))
		for j, tok := range row.Tokens {
			out[j] = &pb.TokenVector{Values: tok}
		}
		resp.Points = append(resp.Points, &pb.MVBatchGetPoint{
			Id:          row.ID,
			Tokens:      out,
			PayloadJson: metadataJSON(row.Meta),
			Version:     row.Version,
		})
	}
	return resp, nil
}

func (s *Server) MVSetPayload(ctx context.Context, req *pb.SetPayloadRequest) (*pb.PayloadResponse, error) {
	meta, err := parseMetadata(req.GetPayloadJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "payload_json: "+err.Error())
	}
	return s.applyPayload(ctx, "vector_mv_set_payload", req.GetCollection(), req.GetId(), ops.EncodeSetPayloadArgsCAS(req.GetCollection(), req.GetId(), meta, req.GetKeyTtlMs(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) MVOverwritePayload(ctx context.Context, req *pb.SetPayloadRequest) (*pb.PayloadResponse, error) {
	meta, err := parseMetadata(req.GetPayloadJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "payload_json: "+err.Error())
	}
	return s.applyPayload(ctx, "vector_mv_overwrite_payload", req.GetCollection(), req.GetId(), ops.EncodeSetPayloadArgsCAS(req.GetCollection(), req.GetId(), meta, req.GetKeyTtlMs(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) MVDeletePayloadKeys(ctx context.Context, req *pb.DeletePayloadKeysRequest) (*pb.PayloadResponse, error) {
	return s.applyPayload(ctx, "vector_mv_delete_payload_keys", req.GetCollection(), req.GetId(), ops.EncodeDeletePayloadKeysArgsCAS(req.GetCollection(), req.GetId(), req.GetKeys(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) MVClearPayload(ctx context.Context, req *pb.ClearPayloadRequest) (*pb.PayloadResponse, error) {
	return s.applyPayload(ctx, "vector_mv_clear_payload", req.GetCollection(), req.GetId(), ops.EncodeClearPayloadArgsCAS(req.GetCollection(), req.GetId(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

// MVResplit re-partitions a multi-vector collection. Like Resplit it is
// synchronous and offline: quiesce writes and use a long deadline. The
// new_partitions < 0 reject is edge defense-in-depth over the engine guard.
func (s *Server) MVResplit(ctx context.Context, req *pb.ResplitRequest) (*pb.ResplitResponse, error) {
	if req.GetNewPartitions() < 0 {
		return nil, status.Error(codes.InvalidArgument, "new_partitions must be non-negative")
	}
	if _, err := s.call(ctx, "vector_mv_resplit", ops.EncodeResplitArgs(req.GetName(), int(req.GetNewPartitions()))); err != nil {
		return nil, err
	}
	return &pb.ResplitResponse{Name: req.GetName(), NewPartitions: req.GetNewPartitions()}, nil
}

// MVResplitCleanup drops the orphaned pre-resplit partitions of a multi-vector
// collection after a verified MVResplit, returning the number dropped.
func (s *Server) MVResplitCleanup(ctx context.Context, req *pb.ResplitCleanupRequest) (*pb.ResplitCleanupResponse, error) {
	body, err := s.call(ctx, "vector_mv_resplit_cleanup", ops.EncodeResplitCleanupArgs(req.GetName()))
	if err != nil {
		return nil, err
	}
	dropped, err := ops.DecodeResplitCleanupResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.ResplitCleanupResponse{Name: req.GetName(), Dropped: int32(dropped)}, nil //nolint:gosec // count fits int32
}

// MVReshard re-partitions a multi-vector collection ONLINE. Same semantics as
// Reshard: live (dual-write), synchronous, use a long deadline.
func (s *Server) MVReshard(ctx context.Context, req *pb.ReshardRequest) (*pb.ReshardResponse, error) {
	if req.GetNewPartitions() < 0 {
		return nil, status.Error(codes.InvalidArgument, "new_partitions must be non-negative")
	}
	if _, err := s.call(ctx, "vector_mv_reshard", ops.EncodeReshardArgs(req.GetName(), int(req.GetNewPartitions()))); err != nil {
		return nil, err
	}
	return &pb.ReshardResponse{Name: req.GetName(), NewPartitions: req.GetNewPartitions()}, nil
}

// MVReshardAbort aborts an in-flight multi-vector reshard; see ReshardAbort.
// Pre-cutover only.
func (s *Server) MVReshardAbort(ctx context.Context, req *pb.ReshardAbortRequest) (*pb.ReshardAbortResponse, error) {
	if _, err := s.call(ctx, "vector_mv_reshard_abort", ops.EncodeReshardAbortArgs(req.GetName())); err != nil {
		return nil, err
	}
	return &pb.ReshardAbortResponse{Name: req.GetName()}, nil
}

// ---- named vectors (Qdrant-style per-point multi-vector-space) ----

// namedVectorsFromPB flattens the request's map<string, NamedVectorList> into a
// plain map[string][]float32 for the wire codec.
// namedVectorsFromPB splits the per-space upsert map into its DENSE and SPARSE
// lanes by entry shape: an entry carrying sparse_indices/sparse_values is a sparse
// value (-> sparseVectors), otherwise it is a dense value (-> dense). A space entry
// is dense XOR sparse; the engine validates the modality against the configured
// space (ErrSpaceModalityMismatch). An empty entry (no values and no sparse arrays)
// is treated as dense-empty, which the engine rejects with a dim mismatch for a
// dense space (fail loud), matching the binary wire path.
func namedVectorsFromPB(in map[string]*pb.NamedVectorList) (dense map[string][]float32, sparse map[string]*vector.SparseVector) {
	if len(in) == 0 {
		return nil, nil
	}
	for name, v := range in {
		if len(v.GetSparseIndices()) > 0 || len(v.GetSparseValues()) > 0 {
			if sparse == nil {
				sparse = make(map[string]*vector.SparseVector, len(in))
			}
			sparse[name] = &vector.SparseVector{Indices: v.GetSparseIndices(), Values: v.GetSparseValues()}
			continue
		}
		if dense == nil {
			dense = make(map[string][]float32, len(in))
		}
		dense[name] = v.GetValues()
	}
	return dense, sparse
}

// parseNamedConfig decodes a NamedCreate config_json into the
// map[string]NamedVectorParams the wire codec expects. An empty string yields a
// nil map (the engine rejects it as ErrEmptyNamedVectors → InvalidArgument).
func parseNamedConfig(s string) (map[string]vector.NamedVectorParams, error) {
	if s == "" {
		return nil, nil
	}
	var cfg map[string]vector.NamedVectorParams
	if err := json.Unmarshal([]byte(s), &cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (s *Server) NamedCreate(ctx context.Context, req *pb.NamedCreateRequest) (*pb.NamedCreateResponse, error) {
	if strings.ContainsAny(req.GetName(), "#@") {
		return nil, status.Errorf(codes.InvalidArgument, "vector: collection name %q must not contain reserved characters '#' or '@'", req.GetName())
	}
	cfg, err := parseNamedConfig(req.GetConfigJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "config_json: "+err.Error())
	}
	if req.GetPartitions() < 0 {
		return nil, status.Error(codes.InvalidArgument, "partitions must be >= 0")
	}
	if _, err := s.call(ctx, "vector_named_create_collection", ops.EncodeNamedCreateArgs(req.GetName(), cfg, int(req.GetPartitions()))); err != nil {
		return nil, err
	}
	return &pb.NamedCreateResponse{Name: req.GetName()}, nil
}

func (s *Server) NamedDrop(ctx context.Context, req *pb.NamedDropRequest) (*pb.NamedDropResponse, error) {
	if _, err := s.call(ctx, "vector_named_drop_collection", ops.EncodeNamedNameArgs(req.GetName())); err != nil {
		return nil, err
	}
	return &pb.NamedDropResponse{Name: req.GetName()}, nil
}

func (s *Server) NamedUpsert(ctx context.Context, req *pb.NamedUpsertRequest) (*pb.NamedUpsertResponse, error) {
	meta, err := parseMetadata(req.GetMetadataJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "metadata_json: "+err.Error())
	}
	ttl := time.Duration(req.GetTtlMs()) * time.Millisecond
	dense, sparse := namedVectorsFromPB(req.GetVectors())
	// key_ttl_ms (per-key payload TTL, relative ms); empty = no per-key TTL. The
	// sparse lane rides the additive sparse sub-block (byte-identical when empty).
	args := ops.EncodeNamedInsertArgsSparseCASKeyTTL(req.GetName(), req.GetId(), dense, sparse, meta, ttl, req.GetExpectedVersion(), req.GetHasExpectedVersion(), req.GetKeyTtlMs())
	wcf, wait := wcOf(req)
	if _, err := s.callWrite(ctx, "vector_named_insert", args, wcf, wait); err != nil {
		return nil, err
	}
	return &pb.NamedUpsertResponse{Id: req.GetId()}, nil
}

func (s *Server) NamedSearch(ctx context.Context, req *pb.NamedSearchRequest) (*pb.NamedSearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	args := ops.EncodeNamedSearchArgsOpts(req.GetName(), req.GetVectorName(), req.GetQuery(), int(req.GetK()), filter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_named_search", args)
	if err != nil {
		return nil, err
	}
	results, err := ops.DecodeVectorSearchResults(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.NamedResult, len(results))
	for i, r := range results {
		out[i] = &pb.NamedResult{Id: r.ID, Distance: r.Distance, Score: r.Score}
	}
	return &pb.NamedSearchResponse{Results: out}, nil
}

func (s *Server) NamedSparseSearch(ctx context.Context, req *pb.NamedSparseSearchRequest) (*pb.NamedSearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	query := vector.SparseVector{Indices: req.GetSparseIndices(), Values: req.GetSparseValues()}
	args := ops.EncodeNamedSparseSearchArgsOpts(req.GetName(), req.GetVectorName(), query, int(req.GetK()), filter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_named_sparse_search", args)
	if err != nil {
		return nil, err
	}
	results, err := ops.DecodeHybridResults(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.NamedResult, len(results))
	for i, r := range results {
		out[i] = &pb.NamedResult{Id: r.ID, Distance: r.Distance, Score: r.Score}
	}
	return &pb.NamedSearchResponse{Results: out}, nil
}

// NamedHybridSearch fuses a DENSE named space and a SPARSE named space of one named
// collection into the top-k (cross-space hybrid), mirroring HybridSearch but across
// two named spaces. An unknown space / modality mismatch surfaces as InvalidArgument
// (mapped from the engine's ErrUnknownVectorName / ErrSpaceModalityMismatch); the
// fused score rides NamedResult.score.
func (s *Server) NamedHybridSearch(ctx context.Context, req *pb.NamedHybridRequest) (*pb.NamedSearchResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	method, err := parseFusion(req.GetMethod())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	sparse := vector.SparseVector{Indices: req.GetSparseIndices(), Values: req.GetSparseValues()}
	opts := vector.HybridOpts{
		Filter: filter, Method: method, Alpha: req.GetAlpha(),
		RRFK: int(req.GetRrfK()), DenseK: int(req.GetDenseK()), SparseK: int(req.GetSparseK()),
	}
	args := ops.EncodeNamedHybridArgs(req.GetName(), req.GetDenseSpace(), req.GetDense(), req.GetSparseSpace(), sparse, int(req.GetK()), opts, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_named_hybrid_search", args)
	if err != nil {
		return nil, err
	}
	results, err := ops.DecodeHybridResults(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.NamedResult, len(results))
	for i, r := range results {
		out[i] = &pb.NamedResult{Id: r.ID, Distance: r.Distance, Score: r.Score}
	}
	return &pb.NamedSearchResponse{Results: out}, nil
}

func (s *Server) NamedSearchDocs(ctx context.Context, req *pb.NamedSearchRequest) (*pb.NamedSearchDocsResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	args := ops.EncodeNamedSearchArgsOpts(req.GetName(), req.GetVectorName(), req.GetQuery(), int(req.GetK()), filter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_named_search_docs", args)
	if err != nil {
		return nil, err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.NamedSearchDocsResponse{Documents: toPBDocs(docs)}, nil
}

func (s *Server) NamedDelete(ctx context.Context, req *pb.NamedDeleteRequest) (*pb.NamedDeleteResponse, error) {
	wcf, wait := wcOf(req)
	body, err := s.callWrite(ctx, "vector_named_delete", ops.EncodeNamedDeleteArgsCAS(req.GetName(), req.GetId(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), wcf, wait)
	if err != nil {
		return nil, err
	}
	return &pb.NamedDeleteResponse{Deleted: len(body) > 0 && body[0] == 1}, nil
}

func (s *Server) NamedGet(ctx context.Context, req *pb.NamedGetRequest) (*pb.NamedGetResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	flags := getFlags(req.GetWithVector(), req.GetWithPayload())
	body, err := s.call(ctx, "vector_named_get", ops.EncodeVectorGetArgsOpts(req.GetCollection(), req.GetId(), flags, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	found, vectors, payload, ttl, version, err := ops.DecodeNamedGetResultV(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	if !found {
		return nil, status.Errorf(codes.NotFound, "point %d not found in collection %q", req.GetId(), req.GetCollection())
	}
	out := make(map[string]*pb.NamedVectorList, len(vectors))
	for name, v := range vectors {
		out[name] = &pb.NamedVectorList{Values: v}
	}
	return &pb.NamedGetResponse{
		Found: true, Vectors: out,
		PayloadJson: metadataJSON(payload), TtlMs: ttl.Milliseconds(),
		Version: version,
	}, nil
}

// NamedGetBatch retrieves MANY named-vector points by id in ONE op. Like GetBatch
// (and unlike single NamedGet) a partial miss is NEVER a NotFound error — absent
// ids are returned in the missing list (ok). Each present point carries its
// per-space vectors map + payload + ttl_ms. The named clone of GetBatch.
func (s *Server) NamedGetBatch(ctx context.Context, req *pb.NamedGetBatchRequest) (*pb.NamedGetBatchResponse, error) {
	flags := getFlags(req.GetWithVector(), req.GetWithPayload())
	body, err := s.call(ctx, "vector_named_get_batch", ops.EncodeVectorGetBatchArgs(req.GetCollection(), req.GetIds(), flags))
	if err != nil {
		return nil, err
	}
	rows, err := ops.DecodeNamedGetBatchResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	resp := &pb.NamedGetBatchResponse{}
	for i := range rows {
		row := &rows[i]
		if !row.Found {
			resp.Missing = append(resp.Missing, row.ID)
			continue
		}
		out := make(map[string]*pb.NamedVectorList, len(row.Vectors))
		for name, v := range row.Vectors {
			out[name] = &pb.NamedVectorList{Values: v}
		}
		resp.Points = append(resp.Points, &pb.NamedBatchGetPoint{
			Id:          row.ID,
			Vectors:     out,
			PayloadJson: metadataJSON(row.Meta),
			TtlMs:       int64(row.TTLMs), //nolint:gosec // TTL ms >= 0
			Version:     row.Version,
		})
	}
	return resp, nil
}

func (s *Server) NamedSetPayload(ctx context.Context, req *pb.SetPayloadRequest) (*pb.PayloadResponse, error) {
	meta, err := parseMetadata(req.GetPayloadJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "payload_json: "+err.Error())
	}
	return s.applyPayload(ctx, "vector_named_set_payload", req.GetCollection(), req.GetId(), ops.EncodeSetPayloadArgsCAS(req.GetCollection(), req.GetId(), meta, req.GetKeyTtlMs(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) NamedOverwritePayload(ctx context.Context, req *pb.SetPayloadRequest) (*pb.PayloadResponse, error) {
	meta, err := parseMetadata(req.GetPayloadJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "payload_json: "+err.Error())
	}
	return s.applyPayload(ctx, "vector_named_overwrite_payload", req.GetCollection(), req.GetId(), ops.EncodeSetPayloadArgsCAS(req.GetCollection(), req.GetId(), meta, req.GetKeyTtlMs(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) NamedDeletePayloadKeys(ctx context.Context, req *pb.DeletePayloadKeysRequest) (*pb.PayloadResponse, error) {
	return s.applyPayload(ctx, "vector_named_delete_payload_keys", req.GetCollection(), req.GetId(), ops.EncodeDeletePayloadKeysArgsCAS(req.GetCollection(), req.GetId(), req.GetKeys(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) NamedClearPayload(ctx context.Context, req *pb.ClearPayloadRequest) (*pb.PayloadResponse, error) {
	return s.applyPayload(ctx, "vector_named_clear_payload", req.GetCollection(), req.GetId(), ops.EncodeClearPayloadArgsCAS(req.GetCollection(), req.GetId(), req.GetExpectedVersion(), req.GetHasExpectedVersion()), req.GetWriteConsistencyFactor(), !req.GetNoWait())
}

func (s *Server) NamedScroll(ctx context.Context, req *pb.NamedScrollRequest) (*pb.NamedScrollResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	order, err := parsePBOrderBy(req.GetOrderBy())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	// A malformed cursor — or a cursor whose version disagrees with order_by — is a
	// client error: reject with InvalidArgument BEFORE dispatch.
	afterID, hasAfter, scrollOrder, err := scrollCursorAndOrder(req.GetCursor(), order)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	args := ops.EncodeNamedScrollArgsOrderBounded(req.GetName(), filter, int(req.GetLimit()), afterID, hasAfter, uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), scrollOrder, req.GetMaxStaleness())
	body, err := s.call(ctx, "vector_named_scroll", args)
	if err != nil {
		return nil, err
	}
	docs, _, _, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.NamedScrollResponse{Documents: toPBDocs(docs), NextCursor: nextCursor}, nil
}

// Scroll is the dense cursor-paginated scroll RPC (parity with the HTTP dense
// scroll route). Mirrors NamedScroll for the dense op family: decode the cursor
// (malformed ⇒ InvalidArgument), dispatch vector_scroll, return docs +
// next_cursor + degraded/missing fan-out health.
func (s *Server) Scroll(ctx context.Context, req *pb.ScrollRequest) (*pb.ScrollResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	order, err := parsePBOrderBy(req.GetOrderBy())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	afterID, hasAfter, scrollOrder, err := scrollCursorAndOrder(req.GetCursor(), order)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	body, err := s.call(ctx, "vector_scroll", ops.EncodeScrollArgsOrderBounded(req.GetCollection(), filter, int(req.GetLimit()), uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), afterID, hasAfter, scrollOrder, req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.ScrollResponse{Documents: toPBDocs(docs), NextCursor: nextCursor, Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

// parsePBOrderBy converts a proto OrderBy into a validated *vector.OrderBy (nil ⇒ no
// order_by). An empty key with an order_by message present ⇒ InvalidArgument (via
// vector.ErrEmptyOrderKey); a bad datetime start_from ⇒ InvalidArgument. start_from
// arrives as a double (numeric) OR start_from_datetime as an RFC3339 string (lowered
// to int-ms); at most one is set.
func parsePBOrderBy(pb *pb.OrderBy) (*vector.OrderBy, error) {
	if pb == nil {
		return nil, nil
	}
	var startNum *float64
	if pb.StartFrom != nil {
		v := pb.GetStartFrom()
		startNum = &v
	}
	var startDt *string
	if pb.StartFromDatetime != nil {
		v := pb.GetStartFromDatetime()
		startDt = &v
	}
	ob, err := vector.ParseOrderBy(pb.GetKey(), pb.GetDesc(), pb.GetIsDatetime(), pb.GetIsString(), startNum, startDt)
	if err != nil {
		return nil, err
	}
	// MULTI-KEY: build the secondary/tertiary keys from tail_keys. Each tail key is a
	// plain OrderBy (key/desc/is_datetime/is_string); start_from is primary-only (a tail
	// key with a start_from is rejected by ParseOrderBy as ErrBadOrderKind for a string,
	// else simply unused — the engine ignores tail StartFrom). A tail key's own tail_keys
	// are ignored (the list is flat, one level deep).
	if len(pb.GetTailKeys()) > 0 {
		ob.Tail = make([]vector.OrderBy, 0, len(pb.GetTailKeys()))
		for _, tk := range pb.GetTailKeys() {
			tob, terr := vector.ParseOrderBy(tk.GetKey(), tk.GetDesc(), tk.GetIsDatetime(), tk.GetIsString(), nil, nil)
			if terr != nil {
				return nil, terr
			}
			ob.Tail = append(ob.Tail, vector.OrderBy{Key: tob.Key, Desc: tob.Desc, IsDatetime: tob.IsDatetime, Kind: tob.Kind})
		}
	}
	return ob, nil
}

// scrollCursorAndOrder decodes the scroll cursor TYPED and reconciles it with an
// optional order_by, producing the (afterID, hasAfter, *ops.ScrollOrder) the args
// codec needs. It enforces the cursor⇄order_by agreement at the edge (loud
// InvalidArgument), mirroring the coordinator's guard:
//
//   - order_by present: validate a v2 cursor's direction/key (or reject a v1 cursor);
//     the resume (value, id) rides the order block + afterID.
//   - order_by absent: only a v1 (id-only) cursor is valid; a v2 cursor is rejected.
func scrollCursorAndOrder(cursor string, order *vector.OrderBy) (afterID uint64, hasAfter bool, scrollOrder *ops.ScrollOrder, err error) {
	dec, derr := ops.DecodeScrollCursorTyped(cursor)
	if derr != nil {
		return 0, false, nil, derr
	}
	if order == nil {
		if dec.Present && dec.Version != 1 {
			return 0, false, nil, ops.ErrCursorOrderMismatch
		}
		return dec.LastID, dec.Present, nil, nil
	}
	if len(order.Tail) > 0 {
		return scrollCursorAndOrderMulti(dec, order)
	}
	keyHash := vector.OrderKeyHash(order.Key)
	if order.Kind == vector.OrderString {
		if verr := ops.ValidateOrderCursorString(dec, order.Desc, keyHash); verr != nil {
			return 0, false, nil, verr
		}
		so := &ops.ScrollOrder{Key: order.Key, Desc: order.Desc, Kind: vector.OrderString}
		if dec.Present {
			afterID, hasAfter = dec.LastID, true
			so.ResumeStr, so.HasResumeStr = dec.StrValue, true
		}
		return afterID, hasAfter, so, nil
	}
	if verr := ops.ValidateOrderCursor(dec, order.Desc, keyHash); verr != nil {
		return 0, false, nil, verr
	}
	so := &ops.ScrollOrder{Key: order.Key, Desc: order.Desc, IsDatetime: order.IsDatetime, Kind: order.Kind, StartFrom: order.StartFrom, HasStart: order.HasStart}
	if dec.Present {
		afterID, hasAfter = dec.LastID, true
		so.ResumeKey, so.HasResume = dec.Value, true
	}
	return afterID, hasAfter, so, nil
}

// scrollCursorAndOrderMulti is the MULTI-KEY branch of scrollCursorAndOrder: it validates
// a v4 (k1,…,kN, id) tuple cursor against the request's primary direction + key-list hash
// + arity (a v1/v2/v3 cursor, or a wrong-arity v4, is a loud mismatch) and threads the
// resume TUPLE onto ScrollOrder.ResumeKeys + the args afterID. Mirrors the coordinator's
// buildScrollOrder multi-key path so the edge guard and the dispatcher agree.
func scrollCursorAndOrderMulti(dec ops.DecodedScrollCursor, order *vector.OrderBy) (afterID uint64, hasAfter bool, scrollOrder *ops.ScrollOrder, err error) {
	keys := vector.OrderKeyList(order)
	keyHash := vector.OrderKeyListHash(keys)
	if verr := ops.ValidateOrderCursorTuple(dec, order.Desc, keyHash, len(keys)); verr != nil {
		return 0, false, nil, verr
	}
	so := &ops.ScrollOrder{Key: order.Key, Desc: order.Desc, IsDatetime: order.IsDatetime, Kind: order.Kind, Tail: ops.OrderByToScrollOrderTail(order)}
	if dec.Present {
		afterID, hasAfter = dec.LastID, true
		so.ResumeKeys = make([]ops.ScrollOrderVal, len(dec.Tuple))
		for i, kv := range dec.Tuple {
			so.ResumeKeys[i] = ops.ScrollOrderVal{Num: kv.Num, Str: kv.Str, Kind: vector.OrderKind(kv.Kind)}
		}
		so.HasResumeKeys = true
	}
	return afterID, hasAfter, so, nil
}

// MVScroll is the multi-vector cursor-paginated scroll RPC (parity with the HTTP
// MV scroll route). Mirrors Scroll for the MV op family: validate consistency,
// decode the cursor (malformed ⇒ InvalidArgument), dispatch vector_mv_scroll,
// return docs + next_cursor + degraded/missing fan-out health.
func (s *Server) MVScroll(ctx context.Context, req *pb.MVScrollRequest) (*pb.MVScrollResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	filter, err := parseFilter(req.GetFilterJson())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "filter_json: "+err.Error())
	}
	order, err := parsePBOrderBy(req.GetOrderBy())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	afterID, hasAfter, scrollOrder, err := scrollCursorAndOrder(req.GetCursor(), order)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	body, err := s.call(ctx, "vector_mv_scroll", ops.EncodeMVScrollArgsOrderBounded(req.GetCollection(), filter, int(req.GetLimit()), uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), afterID, hasAfter, scrollOrder, req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &pb.MVScrollResponse{Documents: toPBDocs(docs), NextCursor: nextCursor, Degraded: degraded, Missing: u16ToU32(missing)}, nil
}

func (s *Server) NamedGetConfig(ctx context.Context, req *pb.NamedGetConfigRequest) (*pb.NamedGetConfigResponse, error) {
	if err := validConsistency(req.GetReadConsistency(), req.GetOnPartitionUnavailable()); err != nil {
		return nil, err
	}
	body, err := s.call(ctx, "vector_named_get_config", ops.EncodeNamedNameArgsOpts(req.GetName(), uint8(req.GetReadConsistency()), uint8(req.GetOnPartitionUnavailable()), req.GetMaxStaleness()))
	if err != nil {
		return nil, err
	}
	cfg, err := ops.DecodeNamedConfigResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	cfgJSON, _ := json.Marshal(cfg)
	return &pb.NamedGetConfigResponse{Name: req.GetName(), ConfigJson: string(cfgJSON)}, nil
}

// ---- conversions ----

func toPBResults(rs []vector.Result) []*pb.Result {
	out := make([]*pb.Result, len(rs))
	for i, r := range rs {
		out[i] = &pb.Result{Id: r.ID, Distance: r.Distance, Score: r.Score}
	}
	return out
}

func toPBDocs(ds []vector.Document) []*pb.Document {
	out := make([]*pb.Document, len(ds))
	for i, d := range ds {
		out[i] = &pb.Document{
			Id: d.ID, Distance: d.Distance, Score: d.Score,
			Content: d.Content, MetadataJson: metadataJSON(d.Metadata),
		}
	}
	return out
}

func metadataJSON(m vector.Metadata) string {
	if len(m) == 0 {
		return ""
	}
	b, _ := json.Marshal(m)
	return string(b)
}

func parseMetadata(s string) (vector.Metadata, error) {
	if s == "" {
		return nil, nil
	}
	var m vector.Metadata
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, err
	}
	return m, nil
}

// parseFilter decodes filter_json and validate-at-edge compiles it: a bad RE2
// regex or non-RFC3339 datetime literal fails here (as a Compile error) so the
// caller can map it to InvalidArgument BEFORE dispatch — a malformed filter must
// never reach the engine (notably: never trigger an over-broad delete_by_filter).
func parseFilter(s string) (vector.Filter, error) {
	if s == "" {
		return vector.Filter{}, nil
	}
	var f vector.Filter
	if err := json.Unmarshal([]byte(s), &f); err != nil {
		return vector.Filter{}, err
	}
	if _, err := vector.CompileFilter(f); err != nil {
		return vector.Filter{}, err
	}
	return f, nil
}

// ---- collection aliases ----

// CreateAlias points an alias at a real collection (upsert: an existing alias of
// the same name is overwritten). It lowers to a single-action alias_batch so the
// create/delete/swap paths share one atomic coordinator op. Validation errors
// (target missing / shadow / reserved char / target-is-alias) carry the "rostam:
// alias " prefix and map to InvalidArgument via grpcError.
func (s *Server) CreateAlias(ctx context.Context, req *pb.CreateAliasRequest) (*pb.CreateAliasResponse, error) {
	if req.GetAlias() == "" || req.GetCollection() == "" {
		return nil, status.Error(codes.InvalidArgument, "alias and collection are required")
	}
	if _, err := s.call(ctx, "alias_batch", ops.EncodeAliasCreateArgs(req.GetAlias(), req.GetCollection())); err != nil {
		return nil, err
	}
	return &pb.CreateAliasResponse{Alias: req.GetAlias(), Collection: req.GetCollection()}, nil
}

// DeleteAlias removes an alias (a single delete-action alias_batch). An absent
// alias is a no-op (idempotent).
func (s *Server) DeleteAlias(ctx context.Context, req *pb.DeleteAliasRequest) (*pb.DeleteAliasResponse, error) {
	if req.GetAlias() == "" {
		return nil, status.Error(codes.InvalidArgument, "alias is required")
	}
	if _, err := s.call(ctx, "alias_batch", ops.EncodeAliasDeleteArgs(req.GetAlias())); err != nil {
		return nil, err
	}
	return &pb.DeleteAliasResponse{Alias: req.GetAlias()}, nil
}

// ListAliases returns all aliases (optionally filtered to those whose target ==
// collection). It is a local FSM read (alias_list) — no Raft.
func (s *Server) ListAliases(ctx context.Context, req *pb.ListAliasesRequest) (*pb.ListAliasesResponse, error) {
	body, err := s.call(ctx, "alias_list", ops.EncodeAliasListArgs(req.GetCollection()))
	if err != nil {
		return nil, err
	}
	entries, err := ops.DecodeAliasListResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.AliasEntry, 0, len(entries))
	for _, e := range entries {
		out = append(out, &pb.AliasEntry{Alias: e.Alias, Collection: e.Collection})
	}
	return &pb.ListAliasesResponse{Aliases: out}, nil
}

// AliasBatch applies a list of create/delete actions ATOMICALLY in one
// alias_batch coordinator op (the zero-downtime swap: {delete prod, create
// prod->v2} lands in one meta-Raft entry, so a concurrent reader never sees an
// undefined intermediate state). Each action must set exactly one of create
// (collection non-empty) / delete (delete=true).
func (s *Server) AliasBatch(ctx context.Context, req *pb.AliasBatchRequest) (*pb.AliasBatchResponse, error) {
	actions := make([]ops.AliasAction, 0, len(req.GetActions()))
	for _, a := range req.GetActions() {
		if a.GetAlias() == "" {
			return nil, status.Error(codes.InvalidArgument, "each action requires an alias")
		}
		if a.GetDelete() {
			actions = append(actions, ops.AliasAction{Alias: a.GetAlias(), Delete: true})
			continue
		}
		if a.GetCollection() == "" {
			return nil, status.Error(codes.InvalidArgument, "create action requires a collection")
		}
		actions = append(actions, ops.AliasAction{Alias: a.GetAlias(), Canonical: a.GetCollection()})
	}
	if _, err := s.call(ctx, "alias_batch", ops.EncodeAliasBatchArgs(actions)); err != nil {
		return nil, err
	}
	return &pb.AliasBatchResponse{Applied: int32(len(actions))}, nil //nolint:gosec // action count bounded
}

// KeysAdd registers a new API key on the live registry (the __keys_add__
// coordinator op). The op name is admin-classified, so s.call's authorize gate
// denies a non-admin caller (codes.Unauthenticated) before the registry is
// touched. The raw token is consumed by the registry and NEVER echoed: the ack is
// empty. A dup token → AlreadyExists, no registry wired → FailedPrecondition (see
// grpcError); the registry's own non-empty-token/tenant + known-perm validation
// surfaces as the underlying error.
func (s *Server) KeysAdd(ctx context.Context, req *pb.KeysAddRequest) (*pb.KeysAck, error) {
	if req.GetToken() == "" || req.GetTenant() == "" {
		return nil, status.Error(codes.InvalidArgument, "token and tenant are required")
	}
	args := ops.EncodeKeysAddArgs(ops.KeysAddArgs{
		Token:  req.GetToken(),
		Tenant: req.GetTenant(),
		Scopes: req.GetScopes(),
		CertCN: req.GetCertCn(),
	})
	if _, err := s.call(ctx, ops.OpKeysAdd, args); err != nil {
		return nil, err
	}
	return &pb.KeysAck{}, nil
}

// KeysRevoke removes an API key by its raw token (the __keys_revoke__ coordinator
// op). Admin-gated via s.call. An unknown token → NotFound (see grpcError). The
// ack is empty (the token is never echoed).
func (s *Server) KeysRevoke(ctx context.Context, req *pb.KeysRevokeRequest) (*pb.KeysAck, error) {
	if req.GetToken() == "" {
		return nil, status.Error(codes.InvalidArgument, "token is required")
	}
	if _, err := s.call(ctx, ops.OpKeysRevoke, ops.EncodeKeysRevokeArgs(req.GetToken())); err != nil {
		return nil, err
	}
	return &pb.KeysAck{}, nil
}

// KeysList returns the REDACTED registry snapshot (the __keys_list__ coordinator
// op). Admin-gated via s.call. SECURITY: the op result codec has NO token field
// (the registry replaces each raw token with its fingerprint at the snapshot
// boundary), and this handler maps only fingerprint + tenant + scopes + cert_cn
// onto the proto — so no raw token can reach the wire by construction.
func (s *Server) KeysList(ctx context.Context, _ *pb.KeysListRequest) (*pb.KeysListResponse, error) {
	body, err := s.call(ctx, ops.OpKeysList, ops.EncodeKeysListArgs())
	if err != nil {
		return nil, err
	}
	entries, err := ops.DecodeKeysListResult(body)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	out := make([]*pb.RedactedKey, 0, len(entries))
	for _, e := range entries {
		out = append(out, &pb.RedactedKey{
			Fingerprint: e.Fingerprint,
			Tenant:      e.Tenant,
			Scopes:      e.Scopes,
			CertCn:      e.CertCN,
		})
	}
	return &pb.KeysListResponse{Keys: out}, nil
}
