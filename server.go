// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"runtime/debug"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/grpcapi"
	"github.com/rostamlabs/rostam/httpapi"
	"github.com/rostamlabs/rostam/rlog"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
)

// ServerConfig configures a unified multi-transport server: one store exposed
// simultaneously over any combination of REST/JSON (HTTPAddr), gRPC (GRPCAddr),
// and the binary TCP protocol (TCPAddr). Leave an address empty to disable that
// transport; at least one must be set. All enabled transports dispatch into the
// same store — three front doors onto one store, not separate stores.
//
// By default the store is single-node (Direct). Set Cluster to run a replicated
// Raft-backed node instead; writes replicate across the cluster and vectors are
// partitioned across shards by collection name.
type ServerConfig struct {
	DirectConfig                 // single-node store config (used when Cluster == nil)
	Cluster      *EmbeddedConfig // when set, run a replicated (Raft) node instead of Direct
	HTTPAddr     string          // REST/JSON listen address; "" = disabled
	GRPCAddr     string          // gRPC listen address; "" = disabled
	TCPAddr      string          // binary TCP listen address; "" = disabled

	// EpollTCP selects the epoll/event-loop TCP transport (server.EpollServer)
	// instead of goroutine-per-connection. PLAINTEXT ONLY — ignored (falls back
	// to the goroutine server) when TLSConfig is set, so it is always safe to
	// enable. EpollLoops sets the event-loop count (0 = GOMAXPROCS default).
	//
	// rostam-server passes true by DEFAULT (-epoll=false opts out). The win is
	// conditional on CORE PRESSURE, not universal: when the server competes for
	// cores (a small or shared box, or the load generator co-located on it),
	// goroutine-per-connection scheduler churn costs throughput and epoll avoids
	// it — ~1.4x at 8 connections on an 8-core co-located box. When the server
	// has its own dedicated cores, the churn is absorbed and epoll is within
	// noise of the goroutine server at every concurrency (measured: 16 dedicated
	// cores, epoll and goroutine both ~720k at 64+ conns). It is never slower, so
	// it is a safe default that helps constrained deployments for free. Both
	// transports share the same 5-minute idle-connection timeout, so enabling it
	// does not change connection lifetime semantics.
	//
	// The zero value is false, so EMBEDDERS still opt in explicitly — flipping a
	// zero-valued bool would silently switch every existing caller's transport.
	EpollTCP   bool
	EpollLoops int

	// EnableOnlineCompaction opts every REPLICATED mmap reject-writes shard into
	// ONLINE relocating compaction with quarantine-then-reset recycle
	// (cache/compact_online.go): while the process runs, the TTL sweeper relocates
	// the live entries out of fragmented pages, RETIRES the emptied source extents,
	// and — once the alias-drain fence has elapsed — RESETS those extents back into
	// writable space, so a persistent replicated shard reclaims ghost page bytes
	// WHILE running instead of only at restart (cold compaction). Off by default.
	//
	// SAFETY CONTRACT — READ BEFORE ENABLING. Recycling overwrites retired mmap page
	// bytes after AliasQuarantine (= 2*WriteTimeout) has elapsed. It is ONLY
	// memory-safe when every read of the shard is released within that window — i.e.
	// all reads flow through the SERVER TRANSPORT, whose WriteTimeout bounds the
	// zero-copy response alias. A reject-writes/mmap Get returns a []byte that ALIASES
	// live mmap page bytes; the server response writer's write deadline is the only
	// thing that bounds how long that alias can escape. Do NOT enable this if ANY
	// in-process caller retains a raw cache alias past AliasQuarantine — e.g. an
	// embedder that holds a `Store.Get` / `Node.Call` result (both return the alias
	// verbatim) beyond the fence; such readers MUST copy the value out promptly. When
	// unset the whole feature is a no-op (nothing relocates or recycles), exactly as
	// before this flag existed. Threaded down to every replicated shard's cache
	// (AliasQuarantine derived from WriteTimeout, enforced fail-closed in shard.New).
	EnableOnlineCompaction bool

	// WriteTimeout bounds how long a single TCP response write+flush may block on a
	// stalled client before the connection is aborted (server.Config.WriteTimeout).
	// 0 selects the server's default (30s). It is a CORRECTNESS bound, not just
	// slow-loris hygiene: a reject-writes/mmap response payload is a zero-copy alias
	// into live mmap page bytes, and the online page-recycle fence
	// (cache.AliasQuarantine) must outlast the maximum alias hold. This one value is
	// the SINGLE SOURCE OF TRUTH: NewServer passes it to the TCP transport AND threads
	// it down to every replicated shard's cache (AliasQuarantine = 2*WriteTimeout,
	// enforced fail-closed in shard.New), so the deadline and the fence can never
	// drift apart. Applies to the goroutine-per-connection TCP server; the epoll
	// transport copies each payload synchronously in its event loop (no queued hold).
	WriteTimeout time.Duration

	// TLSConfig, when non-nil, enables TLS on ALL THREE client-facing transports
	// (HTTP, gRPC, TCP) using a single pre-built *tls.Config — typically built once
	// via tlsutil.ServerTLS(cert, key, ca, requireClientCert). nil ⇒ every
	// transport serves PLAINTEXT exactly as before (the default; byte-identical to
	// the pre-TLS path).
	//
	// When the config sets ClientAuth=RequireAndVerifyClientCert (mTLS), a client
	// presenting no cert or a cert not chaining to the config's ClientCAs is
	// rejected at the TLS handshake, before any app logic. The verified client-cert
	// CN is then threaded into authz.AuthRequest.ClientCN so a cert-only client
	// authorizes via its CN's registry scopes.
	//
	// SCOPE: this covers only the client-facing transports. The cluster's
	// inter-node peerClient and the Raft mux stay PLAINTEXT this round (a
	// documented follow-up); they are not derived from this field.
	TLSConfig *tls.Config

	// InterNodeTLS is the TLS CLIENT config for the cluster's inter-node forwarding
	// dial (peerClient). When TLSConfig wraps the client-facing transports of a
	// multi-node cluster, the TCP port that inter-node forwarding dials is itself
	// TLS-wrapped, so the inter-node dial must be TLS too or forwarded ops EOF at the
	// peer's handshake. Set this (typically tlsutil.ClientTLS(ca, cert, key, "") with
	// the node's own cert/key as the client cert; ServerName is set per-peer by
	// peerClient) so the inter-node dial verifies each peer's server cert against the
	// CA and presents a node client cert when the peer requires mTLS.
	//
	// nil ⇒ plaintext inter-node dial (the default; zero cost when client TLS is
	// off). AUTH is still the internal token; this only encrypts the transport.
	// Threaded into cfg.Cluster.InterNodeTLS in cluster mode (ignored when Cluster is
	// nil — single-node has no inter-node dial).
	InterNodeTLS *tls.Config

	// InterNodeServerTLS is the TLS SERVER config that wraps this cluster's
	// inter-node REPLICATION listeners (Raft mux/fabric + PB), threaded into
	// cfg.Cluster.InterNodeServerTLS in cluster mode (ignored when Cluster is nil).
	// It is the server-side counterpart of InterNodeTLS: without it the replication
	// ports stay plaintext even when TLSConfig wraps the client-facing ports, so a
	// hardened deployment should set BOTH. nil ⇒ plaintext replication listeners
	// (default; byte-identical to today). See cluster.Config.InterNodeServerTLS.
	InterNodeServerTLS *tls.Config

	// NodeCNAllowlist is the OPT-IN per-node mTLS identity allowlist, threaded into
	// cfg.Cluster.NodeCNAllowlist in cluster mode (ignored when Cluster is nil).
	// Empty/nil = OFF = byte-identical. See cluster.Config.NodeCNAllowlist.
	NodeCNAllowlist map[string]bool

	// KeyRegistry, when non-nil, is the SAME *vector.KeyRegistry the Authenticator
	// reads — wired into the dispatcher so the online key-admin coordinator
	// virtual-ops (__keys_add__/__keys_revoke__/__keys_list__) mutate/list it
	// directly. Sharing the one instance means concurrent auth-reads and admin
	// writes serialize on the registry's RWMutex, and an add/revoke takes effect on
	// the serving registry immediately (no restart) and persists via the registry's
	// atomic keys-file flush.
	//
	// nil ⇒ the keys ops are still reachable on every transport but fail loud with
	// ErrKeyAdminUnavailable (open/dev mode, or the static -api-key authenticator,
	// which has no mutable registry). The ops are admin-gated by the normal
	// authorize path (authz classifies the three op names as admin).
	//
	// v1 is per-node-local: in a multi-node cluster a keys mutation applies only to
	// the receiving node's registry (documented; cluster-wide meta-Raft propagation
	// is a v2 follow-up).
	KeyRegistry *vector.KeyRegistry

	// Admin, when non-nil, backs the OPT-IN object-storage admin REST endpoints
	// (backup-now / list-backups / evict / restore). It is built by the cmd layer
	// over the single-node CollectionStore + the shared objstore client, and is
	// admin-scope-gated by the normal authorize path. nil ⇒ those routes return 412
	// (object storage not configured) after the auth check. Ignored when HTTP is
	// disabled. Passed through verbatim to httpapi.Handler.
	Admin httpapi.AdminBackend

	// AccessLog, when non-nil (enabled), turns on the OPT-IN per-request access log
	// on EVERY transport: HTTP wraps the handler with a request-id + access-log
	// middleware, gRPC chains an access-log unary interceptor, and the TCP servers
	// emit one line per dispatched frame. Each line carries a request-id, the op,
	// status, latency, a REDACTED principal (token fingerprint or cert CN, never
	// the raw token), and bytes. nil ⇒ off (the default): no middleware/interceptor
	// is installed and the dispatch hot path is byte-identical to before, so the
	// feature costs nothing when unused. Built by the cmd layer from -access-log.
	AccessLog *rlog.AccessLog
}

// grpcRecoveryInterceptor contains a handler panic to the RPC (returns
// codes.Internal, never the panic detail) and logs it with a stack — so one bad
// request cannot crash the whole process. Mirrors net/http's built-in per-
// request recovery and the TCP transport's dispatch() recover.
func grpcRecoveryInterceptor(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("recovered panic in grpc handler",
				"transport", "grpc", "method", info.FullMethod, "request_id", rlog.RequestID(ctx),
				"panic", r, "stack", string(debug.Stack()))
			err = status.Errorf(codes.Internal, "internal error")
		}
	}()
	return handler(ctx, req)
}

// Server is a single store (Direct or replicated) fronted by one or more network
// transports (see ServerConfig). Close shuts every transport down and closes the
// underlying store once.
type Server struct {
	store    io.Closer
	httpSrv  *http.Server
	httpLn   net.Listener
	grpcSrv  *grpc.Server
	grpcLn   net.Listener
	tcpSrv   *server.Server
	epollSrv *server.EpollServer // set instead of tcpSrv when EpollTCP is on

	// The per-transport dispatchers, retained after wiring. HTTP and gRPC MUST
	// share the leader-following chain in cluster mode (see NewServer): a write
	// addressed to a node that hosts the shard but is not its leader is redirected
	// server-side rather than refused. Only TCP keeps the bare chain, because a
	// TCP client follows the hint itself.
	//
	// They are fields rather than locals so that property is TESTABLE. It was
	// established for HTTP by a cluster-level test and merely assumed for gRPC,
	// on the strength of one shared assignment — exactly the kind of thing a
	// later refactor splits without noticing, and whose failure mode (gRPC-only
	// write failures against a backup) no HTTP test can see.
	httpDisp server.Dispatcher
	grpcDisp server.Dispatcher
}

// NewServer builds the store (Direct, or replicated when cfg.Cluster is set) and
// serves it over every transport whose address is set in cfg. Use "127.0.0.1:0"
// for an OS-assigned port and read the bound address back via
// HTTPAddr/GRPCAddr/TCPAddr. cfg.Authenticator, if set, gates every transport
// with the same func(token, op) bool.
func NewServer(cfg ServerConfig) (*Server, error) {
	if cfg.HTTPAddr == "" && cfg.GRPCAddr == "" && cfg.TCPAddr == "" {
		return nil, errors.New("rostam: NewServer requires at least one of HTTPAddr/GRPCAddr/TCPAddr")
	}
	// A negative WriteTimeout is a misconfiguration; fold it to 0 (== unset) so it flows
	// through the SAME "<=0 means default" path on BOTH sides — the TCP transport
	// (server.Config.applyDefaults → 30s) and the replicated shards' alias-drain fence
	// (shard.New's 30s fallback). Without this, a negative value threaded raw into
	// clusterCfg.WriteTimeout below would derive a negative AliasQuarantine and fail shard
	// startup with a fence error that names the wrong knob (0 keeps the two sides matched).
	if cfg.WriteTimeout < 0 {
		cfg.WriteTimeout = 0
	}

	// Build the backing store and a dispatcher onto it, shared by every
	// transport. Cluster mode (Raft replication) when cfg.Cluster is set, else
	// single-node Direct.
	var disp server.Dispatcher
	// httpDisp/grpcDisp default to disp but, in cluster mode, follow a hosted-shard
	// NotLeader result to the leader (server-side redirect) — HTTP/gRPC clients
	// can't act on the binary NotLeader hint the way the TCP client does, so
	// without this a write to a follower of a hosted shard would 503/Unavailable.
	var httpDisp, grpcDisp server.Dispatcher
	srv := &Server{}
	if cfg.Cluster != nil {
		// Thread the inter-node TLS dial config into the cluster node. When client
		// TLS wraps this cluster's client-facing ports, the inter-node forward must
		// dial those wrapped ports over TLS too (see ServerConfig.InterNodeTLS). A
		// copy is taken so the caller's EmbeddedConfig is not mutated; nil leaves the
		// inter-node dial plaintext (default).
		clusterCfg := *cfg.Cluster
		clusterCfg.InterNodeTLS = cfg.InterNodeTLS
		clusterCfg.InterNodeServerTLS = cfg.InterNodeServerTLS
		clusterCfg.NodeCNAllowlist = cfg.NodeCNAllowlist
		// Thread the effective TCP WriteTimeout down to the replicated shards so their
		// online-compaction alias-drain fence (AliasQuarantine = 2*WriteTimeout) is
		// derived from the SAME value the TCP transport below arms as its write
		// deadline — the two can never drift apart. 0 flows through and both sides apply
		// their (matching) default.
		clusterCfg.WriteTimeout = cfg.WriteTimeout
		// Opt-in online compaction (off by default). Only when set does a replicated
		// shard recycle retired mmap extents; the fence that makes that safe is derived
		// from WriteTimeout above (see ServerConfig.EnableOnlineCompaction's contract).
		clusterCfg.EnableOnlineCompaction = cfg.EnableOnlineCompaction
		s, err := NewEmbedded(clusterCfg)
		if err != nil {
			return nil, err
		}
		emb, ok := s.(*embedded)
		if !ok {
			_ = s.Close()
			return nil, errors.New("rostam: NewEmbedded returned unexpected type")
		}
		// Wrap each cluster dispatcher in the partition-aware fan-out decorator:
		// for partitioned collections it routes through the embedded backend's
		// cross-shard coordinators; unpartitioned collections pass through
		// byte-identically. TCP uses the bare-node-wrapping decorator; HTTP/gRPC
		// use the leader-following-wrapping decorator.
		disp = newFanoutDispatcher(emb, emb.node)
		lf := emb.node.LeaderFollowingDispatcher()
		fan := newFanoutDispatcher(emb, lf)
		httpDisp, grpcDisp = fan, fan
		srv.store = s
	} else {
		s, err := NewDirect(cfg.DirectConfig)
		if err != nil {
			return nil, err
		}
		store, ok := s.(*directStore)
		if !ok {
			_ = s.Close()
			return nil, errors.New("rostam: NewDirect returned unexpected type")
		}
		disp = store.AsDispatcher()
		httpDisp, grpcDisp = disp, disp
		srv.store = s
	}

	// Wrap every transport's dispatcher in the online key-admin decorator so the
	// __keys_add__/__keys_revoke__/__keys_list__ coordinator virtual-ops hit the
	// SAME *vector.KeyRegistry the Authenticator reads (cfg.KeyRegistry). It is
	// always installed — non-keys ops pass through byte-identically; the keys ops
	// fail loud with ErrKeyAdminUnavailable when no registry is wired (open/dev or
	// static -api-key mode). Admin-gating is handled upstream by the authorize gate
	// (authz classifies the three op names as admin), so this decorator only runs
	// after the caller has passed the admin-scope check.
	disp = newKeysDispatcher(disp, cfg.KeyRegistry)
	httpDisp = newKeysDispatcher(httpDisp, cfg.KeyRegistry)
	grpcDisp = newKeysDispatcher(grpcDisp, cfg.KeyRegistry)
	srv.httpDisp, srv.grpcDisp = httpDisp, grpcDisp

	if cfg.TCPAddr != "" {
		if cfg.EpollTCP && cfg.TLSConfig == nil {
			// Experimental epoll transport (plaintext only). Reuses the same
			// Dispatcher/Authenticator; gnet.Run blocks, so serve it in a goroutine.
			es := server.NewEpollServer(disp, cfg.Authenticator, cfg.AccessLog, cfg.EpollLoops, 5*time.Minute)
			// Start (not fire-and-forget Run) so a bind failure fails NewServer
			// loudly instead of leaving a live process with no TCP transport.
			if err := es.Start(cfg.TCPAddr); err != nil {
				_ = srv.Close()
				return nil, fmt.Errorf("rostam: epoll tcp: %w", err)
			}
			srv.epollSrv = es
		} else {
			tcp, err := server.New(server.Config{
				Addr:          cfg.TCPAddr,
				Dispatcher:    disp,
				Authenticator: cfg.Authenticator,
				AccessLog:     cfg.AccessLog,
				// Same value threaded into the replicated shards' AliasQuarantine above
				// (clusterCfg.WriteTimeout): the write deadline and the recycle fence share
				// one source. 0 ⇒ server.Config.applyDefaults uses 30s, matching shard.New's
				// fallback, so an unset value keeps both sides consistent.
				WriteTimeout: cfg.WriteTimeout,
				// Same flag threaded into the replicated shards' caches above
				// (clusterCfg.EnableOnlineCompaction → cache.OnlineCompaction): recycle and
				// the pipelined copy that fences it share ONE source of truth, so the
				// transport copies a zero-copy alias out before queueing it EXACTLY when a
				// shard can recycle the extent behind it, and never otherwise (default off ⇒
				// zero-copy pipeline, no per-conn amplification). See server.Config's field doc.
				EnableOnlineCompaction: cfg.EnableOnlineCompaction,
				// nil TLSConfig ⇒ the TCP server keeps its plaintext net.Listener; a
				// non-nil one wraps the listener in tls.NewListener (the v1/v2 framing
				// rides unchanged over the TLS conn). The verified client-cert CN is
				// threaded into AuthRequest.ClientCN inside the server pkg.
				TLSConfig: cfg.TLSConfig,
			})
			if err != nil {
				_ = srv.Close()
				return nil, fmt.Errorf("rostam: tcp: %w", err)
			}
			srv.tcpSrv = tcp
			go func() { _ = tcp.Serve() }()
		}
	}

	if cfg.HTTPAddr != "" {
		ln, err := net.Listen("tcp", cfg.HTTPAddr)
		if err != nil {
			_ = srv.Close()
			return nil, fmt.Errorf("rostam: http listen: %w", err)
		}
		// When TLS is configured, wrap the listener so http.Server.Serve speaks TLS
		// over the existing net.Listener (cleanest: keeps Serve(ln), no ServeTLS file
		// reload). Go's http server only populates r.TLS.PeerCertificates after a
		// successful handshake per the config's ClientAuth, so with
		// RequireAndVerifyClientCert the leaf at PeerCertificates[0] is CA-verified.
		if cfg.TLSConfig != nil {
			ln = tls.NewListener(ln, cfg.TLSConfig)
		}
		srv.httpLn = ln
		// server.Authenticator, httpapi.Authenticator, and grpcapi.Authenticator are
		// all aliases of authz.Authenticator, so cfg.Authenticator passes through to
		// every transport unchanged (a nil stays nil = open/no-auth mode).
		srv.httpSrv = &http.Server{
			// AccessLog.Middleware is a no-op wrap when the access log is disabled
			// (returns the handler unchanged), so the default build adds nothing.
			Handler:           cfg.AccessLog.Middleware(httpapi.Handler(httpDisp, httpapi.Options{Authenticator: cfg.Authenticator, Admin: cfg.Admin})),
			ReadHeaderTimeout: 10 * time.Second,
			// ReadTimeout/IdleTimeout close the body-phase and keep-alive
			// slowloris windows that ReadHeaderTimeout alone leaves open (a client
			// dribbling a request body, or holding an idle keep-alive conn, under
			// the per-route MaxBytesReader cap). WriteTimeout bounds a slow reader.
			// Generous enough for large vector uploads over the per-route body cap.
			ReadTimeout:    60 * time.Second,
			WriteTimeout:   120 * time.Second,
			IdleTimeout:    120 * time.Second,
			MaxHeaderBytes: 1 << 20, // 1 MiB header cap (body caps are enforced per-route via MaxBytesReader)
		}
		// Capture the server in a local, the way the TCP and gRPC goroutines below
		// already do. Reading srv.httpSrv inside the goroutine races Close, which
		// sets that field to nil — so a server closed before this goroutine was
		// first scheduled dereferenced nil and took the process down with it. A
		// panic in a goroutine cannot be recovered by the caller, so the blast
		// radius was the whole program, and the window is widest exactly where it
		// hurts: short-lived servers in tests, and start-then-fail-to-start paths.
		hs := srv.httpSrv
		go func() { _ = hs.Serve(ln) }()
	}

	if cfg.GRPCAddr != "" {
		ln, err := net.Listen("tcp", cfg.GRPCAddr)
		if err != nil {
			_ = srv.Close()
			return nil, fmt.Errorf("rostam: grpc listen: %w", err)
		}
		srv.grpcLn = ln
		// grpc.Creds(credentials.NewTLS(cfg)) terminates TLS inside the gRPC server
		// (so peer.FromContext exposes the verified TLSInfo); nil TLSConfig ⇒ a plain
		// grpc.NewServer() (plaintext, unchanged).
		// Recovery interceptor: grpc-go does NOT recover handler panics (unlike
		// net/http), so without this a panic in any op handler reached over gRPC
		// crashes the whole process. Contain it to the RPC.
		// Interceptor order: the access-log interceptor (when enabled) is OUTERMOST,
		// so it assigns the request-id into the context BEFORE the recovery
		// interceptor runs — a panic recovered inside then logs with that same id,
		// and the access line records the final status/latency. When -access-log is
		// off, UnaryInterceptor() returns nil and only the recovery interceptor is
		// chained (byte-identical to before).
		interceptors := make([]grpc.UnaryServerInterceptor, 0, 2)
		if ai := cfg.AccessLog.UnaryInterceptor(); ai != nil {
			interceptors = append(interceptors, ai)
		}
		interceptors = append(interceptors, grpcRecoveryInterceptor)
		var gs *grpc.Server
		if cfg.TLSConfig != nil {
			gs = grpc.NewServer(grpc.Creds(credentials.NewTLS(cfg.TLSConfig)), grpc.ChainUnaryInterceptor(interceptors...))
		} else {
			gs = grpc.NewServer(grpc.ChainUnaryInterceptor(interceptors...))
		}
		grpcapi.NewServer(grpcDisp, cfg.Authenticator).Register(gs)
		srv.grpcSrv = gs
		go func() { _ = gs.Serve(ln) }()
	}

	return srv, nil
}

// HTTPAddr returns the bound REST address, or "" if HTTP is disabled.
func (s *Server) HTTPAddr() string {
	if s.httpLn == nil {
		return ""
	}
	return s.httpLn.Addr().String()
}

// GRPCAddr returns the bound gRPC address, or "" if gRPC is disabled.
func (s *Server) GRPCAddr() string {
	if s.grpcLn == nil {
		return ""
	}
	return s.grpcLn.Addr().String()
}

// TCPAddr returns the bound binary-TCP address, or "" if TCP is disabled.
//
// It must consult BOTH transports. The epoll server is stored in epollSrv
// instead of tcpSrv, and epoll is the default for single-node — so checking only
// tcpSrv reported "no TCP transport" for the configuration most servers run.
// The startup log iterates these accessors, which meant `-tcp` came up, accepted
// connections, and was never announced: an operator reading the log had no way
// to confirm the listener existed.
func (s *Server) TCPAddr() string {
	if s.epollSrv != nil {
		return s.epollSrv.Addr()
	}
	if s.tcpSrv == nil {
		return ""
	}
	return s.tcpSrv.Addr().String()
}

// VectorStore returns the single-node vector CollectionStore backing this
// server, or nil when the backend has no directly-reachable store. It is the
// handle the out-of-band backup driver (cmd/rostam-server) uses to enumerate +
// snapshot collections without going through a transport.
//
// Only the single-node Direct backend exposes its store this way (its
// *directStore.vectors). In cluster mode the vectors are partitioned across
// Raft shards and Raft is the durability authority, so there is no single
// store to back up here — that returns nil and cluster backup is a documented
// follow-up (drive it from the Raft FSM snapshot, not this accessor).
func (s *Server) VectorStore() *vector.CollectionStore {
	ds, ok := s.store.(*directStore)
	if !ok {
		return nil
	}
	return ds.vectors
}

// ClusterNode returns the cluster.Node backing this server, or nil when the
// backend is not a cluster (single-node Direct). It is the cluster analog of
// VectorStore(): the handle the out-of-band per-node backup/restore driver
// (cmd/rostam-server) uses to snapshot every owned shard (cache + vectors via the
// shard FSM snapshot) and the MetaRaft catalog — the state a cluster deployment
// partitions across shards, which VectorStore() cannot reach.
func (s *Server) ClusterNode() *cluster.Node {
	emb, ok := s.store.(*embedded)
	if !ok {
		return nil
	}
	return emb.node
}

// Close stops every enabled transport and closes the underlying store once.
// Idempotent.
func (s *Server) Close() error {
	var errs []error
	if s.httpSrv != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := s.httpSrv.Shutdown(ctx); err != nil {
			errs = append(errs, err)
		}
		cancel()
		s.httpSrv = nil
	}
	if s.grpcSrv != nil {
		s.grpcSrv.GracefulStop()
		s.grpcSrv = nil
	}
	if s.tcpSrv != nil {
		if err := s.tcpSrv.Close(); err != nil {
			errs = append(errs, err)
		}
		s.tcpSrv = nil
	}
	if s.epollSrv != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := s.epollSrv.Stop(ctx); err != nil {
			errs = append(errs, err)
		}
		cancel()
		s.epollSrv = nil
	}
	if s.store != nil {
		if err := s.store.Close(); err != nil {
			errs = append(errs, err)
		}
		s.store = nil
	}
	return errors.Join(errs...)
}
