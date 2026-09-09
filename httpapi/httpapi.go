// SPDX-License-Identifier: Apache-2.0

// Package httpapi exposes Rostam's vector/RAG operations over a REST/JSON HTTP
// surface. It is a thin transport over the same op dispatcher the binary TCP
// server uses: each handler translates a JSON request into the existing ops
// binary codec, calls Dispatcher.Call, and renders the binary result as JSON.
// No engine logic lives here — it is a second front door onto the same store,
// reachable from any language (curl, Python, a LangChain/LlamaIndex adapter).
package httpapi

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/dashboard"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Dispatcher is the seam onto a backing store: run an op by name with binary
// args and get the encoded result (or an error). *rostam.directStore's
// dispatcher, *shard.Store, and *cluster.Node all satisfy it — the same
// interface the TCP server uses.
type Dispatcher interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}

// Authenticator authorizes a request. It is the unified RBAC authorizer
// (authz.Authenticator): each handler builds an authz.AuthRequest{Token, Op,
// Args} (token from the Bearer header, op + the binary op args it is about to
// dispatch) and the authorizer derives the (action, resource) and matches the
// principal's scopes. A nil Authenticator accepts every request (no-auth mode),
// matching the TCP server's behavior.
type Authenticator = authz.Authenticator

// Options configures a Handler.
type Options struct {
	Authenticator Authenticator
	// Admin, when non-nil, backs the OPT-IN object-storage admin endpoints
	// (POST /v1/admin/backup, GET /v1/admin/backups, POST
	// /v1/collections/{name}/evict, POST /v1/collections/{name}/restore). nil ⇒
	// those routes are still registered but return 412 (object storage not
	// configured) after the admin-scope auth check, so an admin call against an
	// un-tiered server fails loud rather than 404-ing.
	Admin AdminBackend
}

type api struct {
	disp  Dispatcher
	auth  Authenticator
	admin AdminBackend
}

// Handler builds the REST router over disp. The routes (all under /v1):
//
//	GET    /health                              liveness: the process is serving
//	GET    /ready                               readiness: every hosted shard can serve
//	POST   /collections                         create  {name, config}
//	DELETE /collections/{name}                  drop
//	POST   /collections/{name}/points           insert/upsert  {id, vector, ...}
//	POST   /collections/{name}/points/batch     bulk insert/upsert  {upsert, points:[...]}
//	DELETE /collections/{name}/points/{id}      delete
//	POST   /collections/{name}/points/batch-get batch get-by-id  {ids:[...], with_vector, with_payload} → {points:[{id,vector,payload,ttl_ms}], missing:[...]} (partial miss = 200, NOT 404)
//	GET    /collections/{name}/points/{id}      get-by-id  {found, vector, payload, ttl_ms}; ?with_vector/?with_payload
//	POST   /collections/{name}/points/{id}/payload          merge payload  {...payload...}
//	POST   /collections/{name}/points/{id}/payload/overwrite replace payload (also PUT .../payload)
//	POST   /collections/{name}/points/{id}/payload/delete    remove keys    {keys:[...]}
//	POST   /collections/{name}/points/{id}/payload/clear     empty payload
//	POST   /collections/{name}/points/search        knn        {query, k, filter}
//	POST   /collections/{name}/points/search/docs   knn+content
//	POST   /collections/{name}/points/search/groups group-by-document
//	POST   /collections/{name}/points/search/hybrid dense+sparse fusion
//	POST   /collections/{name}/query                unified Query API {root, prefetch:[...], mode:"fusion"|"rerank", method, alpha, rrf_k, k}
//	POST   /collections/{name}/points/delete        delete-by-filter {filter}
//	POST   /collections/{name}/resplit             offline resplit {new_partitions}
//	POST   /collections/{name}/resplit/cleanup     drop orphaned old partitions
//	POST   /collections/{name}/reshard             online reshard (dual-write, live) {new_partitions}
//	POST   /collections/{name}/reshard/abort       abort in-flight reshard (pre-cutover)
//	POST   /multivector/{name}/scroll               MV list live docs {filter, limit, cursor}
//	POST   /multivector/{name}/resplit             MV offline resplit {new_partitions}
//	POST   /multivector/{name}/resplit/cleanup     MV drop orphaned old partitions
//	POST   /multivector/{name}/reshard             MV online reshard (dual-write, live) {new_partitions}
//	POST   /multivector/{name}/reshard/abort       MV abort in-flight reshard (pre-cutover)
//	POST   /multivector/{name}/hybrid-search      MV cross-modality hybrid: fuse the MaxSim lane (query token matrix) + the doc sparse lane {query, sparse:{indices,values}, k, method, alpha, filter}
//	POST   /multivector/{name}/query              MV Query API: MaxSim + sparse prefetch + fusion/rerank {root:{maxsim:[[...]]|sparse:{...},...}, prefetch:[{maxsim|sparse,...}], mode:"fusion"|"rerank", method, alpha, rrf_k, k}
//	GET    /multivector/{name}/points/{id}         MV get-by-id (tokens+payload)
//	POST   /multivector/{name}/points/batch-get    MV batch get-by-id {ids:[...], with_vector, with_payload} → {points:[{id,tokens,payload}], missing:[...]} (partial miss = 200, NOT 404)
//	POST   /multivector/{name}/points/{id}/payload          MV merge payload
//	POST   /multivector/{name}/points/{id}/payload/overwrite MV replace payload (also PUT)
//	POST   /multivector/{name}/points/{id}/payload/delete    MV remove keys {keys}
//	POST   /multivector/{name}/points/{id}/payload/clear     MV empty payload
//	POST   /named/{name}                           create named-vector collection {named_vectors}
//	DELETE /named/{name}                           drop
//	GET    /named/{name}/config                    configured named spaces
//	POST   /named/{name}/points                    upsert  {id, vectors, metadata, ttl_ms}
//	DELETE /named/{name}/points/{id}               delete by id (path)
//	POST   /named/{name}/points/delete             delete by id (body) {id}
//	POST   /named/{name}/search                    knn over a named space {vector_name, query, k, filter}
//	POST   /named/{name}/sparse-search             sparse-dot-product top-k over a SPARSE named space {vector_name, query:{indices,values}, k, filter}
//	POST   /named/{name}/hybrid-search             cross-space hybrid: fuse a dense + a sparse named space {dense_space, dense, sparse_space, sparse:{indices,values}, k, method, alpha, filter}
//	POST   /named/{name}/query                     named Query API: multi-space prefetch + fusion/rerank {root:{space,dense|sparse,...}, prefetch:[{space,...}], mode:"fusion"|"rerank", method, alpha, rrf_k, k}
//	POST   /named/{name}/search/docs               knn+payload
//	POST   /named/{name}/scroll                    list live points {filter, limit}
//	GET    /named/{name}/points/{id}               named get-by-id (per-space vectors+payload)
//	POST   /named/{name}/points/batch-get          named batch get-by-id {ids:[...], with_vector, with_payload} → {points:[{id,vectors,payload,ttl_ms}], missing:[...]} (partial miss = 200, NOT 404)
//	POST   /named/{name}/points/{id}/payload                named merge payload
//	POST   /named/{name}/points/{id}/payload/overwrite      named replace payload (also PUT)
//	POST   /named/{name}/points/{id}/payload/delete         named remove keys {keys}
//	POST   /named/{name}/points/{id}/payload/clear          named empty payload
//	POST   /aliases                              create alias  {alias, collection} (upsert)
//	DELETE /aliases/{alias}                       delete alias (idempotent)
//	GET    /aliases                              list aliases  ?collection=docs filter
//	POST   /aliases/batch                        atomic batch  {actions:[{create:{alias,collection}}|{delete:{alias}}]}
//	POST   /admin/keys                           add API key   {token,tenant,scopes,cert_cn} (admin)
//	DELETE /admin/keys                           revoke key    {token} in BODY, never the path (admin)
//	GET    /admin/keys                           list keys     redacted {keys:[{fingerprint,tenant,scopes,cert_cn}]} (admin)
func Handler(disp Dispatcher, opts Options) http.Handler {
	a := &api{disp: disp, auth: opts.Authenticator, admin: opts.Admin}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/health", a.health)
	// Readiness (distinct from health/liveness): 200 only when this node can
	// actually serve its shards; 503 when a hosted shard has no leader. Wire a
	// load balancer / k8s readiness probe here, not /v1/health. Auth-exempt like
	// health (an infra probe carries no token).
	mux.HandleFunc("GET /v1/ready", a.ready)
	// Prometheus scrape surface. Unversioned /metrics is the scraper convention;
	// /v1/metrics is the versioned alias. Both render every dense collection's
	// stats via the __metrics__ read op.
	mux.HandleFunc("GET /metrics", a.metrics)
	mux.HandleFunc("GET /v1/metrics", a.metrics)
	// Replication observability (#6): per-hosted-shard mode / primary / ISR vs
	// min-ISR / per-backup lag as JSON via the __repl_metrics__ read op.
	// Auth-exempt like /v1/ready — an ops/infra probe carries no token.
	mux.HandleFunc("GET /v1/replication", a.replication)
	// Dashboard read surface: the cluster routing map, the dense-collection list,
	// and a single collection's config. All are scope-gated reads (via a.call), the
	// data plane the embedded web dashboard renders.
	mux.HandleFunc("GET /v1/topology", a.topology)
	mux.HandleFunc("GET /v1/collections", a.collections)
	mux.HandleFunc("GET /v1/collections/{name}", a.collectionConfigView)
	// KV data plane over REST (no KV HTTP routes existed before). The {key} is a
	// URL-encoded UTF-8 path segment (arbitrary bytes on the wire); the 512-byte
	// cap is enforced in-handler since nameLenGuard does not cover /v1/kv/.
	mux.HandleFunc("GET /v1/kv/{key}", a.kvGet)
	mux.HandleFunc("PUT /v1/kv/{key}", a.kvPut)
	mux.HandleFunc("DELETE /v1/kv/{key}", a.kvDelete)
	// Wipe the whole KV keyspace. A fixed, keyless route — no {key} — so it never
	// collides with the /v1/kv/{key} patterns above (those are GET/PUT/DELETE only).
	mux.HandleFunc("POST /v1/kv/flush", a.kvFlush)
	mux.HandleFunc("POST /v1/collections", a.createCollection)
	mux.HandleFunc("DELETE /v1/collections/{name}", a.dropCollection)
	mux.HandleFunc("POST /v1/collections/{name}/points", a.putPoint)
	mux.HandleFunc("POST /v1/collections/{name}/points/batch", a.putPointsBatch)
	mux.HandleFunc("POST /v1/collections/{name}/points/bulk", a.putPointsBulk)
	mux.HandleFunc("POST /v1/collections/{name}/points/bulk/build", a.buildBulk)
	mux.HandleFunc("POST /v1/collections/{name}/points/batch-get", a.getPointsBatch)
	mux.HandleFunc("DELETE /v1/collections/{name}/points/{id}", a.deletePoint)
	mux.HandleFunc("GET /v1/collections/{name}/points/{id}", a.getPoint)
	mux.HandleFunc("POST /v1/collections/{name}/points/{id}/payload", a.setPayload)
	mux.HandleFunc("POST /v1/collections/{name}/points/{id}/payload/overwrite", a.overwritePayload)
	mux.HandleFunc("PUT /v1/collections/{name}/points/{id}/payload", a.overwritePayload)
	mux.HandleFunc("POST /v1/collections/{name}/points/{id}/payload/delete", a.deletePayloadKeys)
	mux.HandleFunc("POST /v1/collections/{name}/points/{id}/payload/clear", a.clearPayload)
	mux.HandleFunc("POST /v1/collections/{name}/points/search", a.search)
	mux.HandleFunc("POST /v1/collections/{name}/points/search/docs", a.searchDocs)
	mux.HandleFunc("POST /v1/collections/{name}/points/search/groups", a.searchGroups)
	mux.HandleFunc("POST /v1/collections/{name}/points/search/hybrid", a.hybrid)
	mux.HandleFunc("POST /v1/collections/{name}/points/search/text", a.searchText)
	mux.HandleFunc("POST /v1/collections/{name}/points/search/hybrid-text", a.hybridText)
	mux.HandleFunc("POST /v1/collections/{name}/query", a.query)
	mux.HandleFunc("POST /v1/collections/{name}/points/delete", a.deleteByFilter)
	mux.HandleFunc("POST /v1/collections/{name}/points/scroll", a.scroll)
	mux.HandleFunc("POST /v1/collections/{name}/resplit", a.resplit)
	mux.HandleFunc("POST /v1/collections/{name}/resplit/cleanup", a.resplitCleanup)
	mux.HandleFunc("POST /v1/collections/{name}/reshard", a.reshard)
	mux.HandleFunc("POST /v1/collections/{name}/reshard/abort", a.reshardAbort)
	// Late-interaction (multi-vector / MaxSim) collections.
	mux.HandleFunc("POST /v1/multivector/{name}", a.mvCreate)
	mux.HandleFunc("DELETE /v1/multivector/{name}", a.mvDrop)
	mux.HandleFunc("POST /v1/multivector/{name}/docs", a.mvAdd)
	mux.HandleFunc("DELETE /v1/multivector/{name}/docs/{id}", a.mvDelete)
	mux.HandleFunc("GET /v1/multivector/{name}/points/{id}", a.mvGet)
	mux.HandleFunc("POST /v1/multivector/{name}/points/batch-get", a.mvGetBatch)
	mux.HandleFunc("POST /v1/multivector/{name}/points/{id}/payload", a.mvSetPayload)
	mux.HandleFunc("POST /v1/multivector/{name}/points/{id}/payload/overwrite", a.mvOverwritePayload)
	mux.HandleFunc("PUT /v1/multivector/{name}/points/{id}/payload", a.mvOverwritePayload)
	mux.HandleFunc("POST /v1/multivector/{name}/points/{id}/payload/delete", a.mvDeletePayloadKeys)
	mux.HandleFunc("POST /v1/multivector/{name}/points/{id}/payload/clear", a.mvClearPayload)
	mux.HandleFunc("POST /v1/multivector/{name}/search", a.mvSearch)
	mux.HandleFunc("POST /v1/multivector/{name}/hybrid-search", a.mvHybrid)
	mux.HandleFunc("POST /v1/multivector/{name}/query", a.mvQuery)
	mux.HandleFunc("POST /v1/multivector/{name}/scroll", a.mvScroll)
	mux.HandleFunc("POST /v1/multivector/{name}/resplit", a.mvResplit)
	mux.HandleFunc("POST /v1/multivector/{name}/resplit/cleanup", a.mvResplitCleanup)
	mux.HandleFunc("POST /v1/multivector/{name}/reshard", a.mvReshard)
	mux.HandleFunc("POST /v1/multivector/{name}/reshard/abort", a.mvReshardAbort)
	// Named-vector (Qdrant-style per-point multi-vector-space) collections.
	mux.HandleFunc("POST /v1/named/{name}", a.namedCreate)
	mux.HandleFunc("DELETE /v1/named/{name}", a.namedDrop)
	mux.HandleFunc("GET /v1/named/{name}/config", a.namedGetConfig)
	mux.HandleFunc("POST /v1/named/{name}/points", a.namedUpsert)
	mux.HandleFunc("DELETE /v1/named/{name}/points/{id}", a.namedDelete)
	mux.HandleFunc("POST /v1/named/{name}/points/delete", a.namedDeleteByID)
	mux.HandleFunc("GET /v1/named/{name}/points/{id}", a.namedGet)
	mux.HandleFunc("POST /v1/named/{name}/points/batch-get", a.namedGetBatch)
	mux.HandleFunc("POST /v1/named/{name}/points/{id}/payload", a.namedSetPayload)
	mux.HandleFunc("POST /v1/named/{name}/points/{id}/payload/overwrite", a.namedOverwritePayload)
	mux.HandleFunc("PUT /v1/named/{name}/points/{id}/payload", a.namedOverwritePayload)
	mux.HandleFunc("POST /v1/named/{name}/points/{id}/payload/delete", a.namedDeletePayloadKeys)
	mux.HandleFunc("POST /v1/named/{name}/points/{id}/payload/clear", a.namedClearPayload)
	mux.HandleFunc("POST /v1/named/{name}/search", a.namedSearch)
	mux.HandleFunc("POST /v1/named/{name}/sparse-search", a.namedSparseSearch)
	mux.HandleFunc("POST /v1/named/{name}/hybrid-search", a.namedHybrid)
	mux.HandleFunc("POST /v1/named/{name}/query", a.namedQuery)
	mux.HandleFunc("POST /v1/named/{name}/search/docs", a.namedSearchDocs)
	mux.HandleFunc("POST /v1/named/{name}/scroll", a.namedScroll)
	// Collection aliases (alias -> collection resolution + atomic swap). The
	// data-plane routes (/v1/collections/{alias}/...) resolve through the engine;
	// these are the alias-management endpoints.
	mux.HandleFunc("POST /v1/aliases", a.createAlias)
	mux.HandleFunc("POST /v1/aliases/batch", a.aliasBatch)
	mux.HandleFunc("GET /v1/aliases", a.listAliases)
	mux.HandleFunc("DELETE /v1/aliases/{alias}", a.deleteAlias)

	// Online key-admin (add/revoke/list API keys at runtime, admin-scope-gated).
	// DELETE takes the token in the request BODY (not the path) so the secret is
	// never logged in access logs; GET returns the redacted view (never the token).
	mux.HandleFunc("POST /v1/admin/keys", a.keysAdd)
	mux.HandleFunc("DELETE /v1/admin/keys", a.keysRevoke)
	mux.HandleFunc("GET /v1/admin/keys", a.keysList)

	// OPT-IN object-storage admin surface (backup/cold-tier control). Admin-scoped
	// (the op names classify as admin / fail-closed via authz.actionFor). When no
	// objstore is configured (opts.Admin nil) these return 412 after the auth check.
	mux.HandleFunc("POST /v1/admin/backup", a.backupNow)
	mux.HandleFunc("GET /v1/admin/backups", a.listBackups)
	mux.HandleFunc("POST /v1/collections/{name}/evict", a.evictCollection)
	mux.HandleFunc("POST /v1/collections/{name}/restore", a.restoreCollection)

	// Embedded web dashboard (static SPA). Served UNDER /dashboard/ with no auth —
	// the assets are inert HTML/JS/CSS and every data read goes through the authed
	// /v1 endpoints above. The bare /dashboard 301-redirects to /dashboard/ so the
	// SPA's relative asset URLs resolve. The dashboard package imports nothing from
	// rostam, so mounting it here forms no import cycle.
	mux.Handle("GET /dashboard/", http.StripPrefix("/dashboard", dashboard.Handler()))
	mux.HandleFunc("GET /dashboard", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/dashboard/", http.StatusMovedPermanently)
	})
	return nameLenGuard(mux)
}

// nameLenGuard rejects requests whose collection/alias path segment exceeds
// maxCollectionNameLen BEFORE they reach a handler. The op wire codecs encode a
// collection name's length in a single byte, so a >=256-byte path segment would
// wrap modulo 256 and silently mis-decode or mis-route (see validName). Handlers
// whose name arrives in the JSON body (createCollection, createAlias, aliasBatch)
// additionally call validName after decoding. Routes without a name segment pass
// through unchanged.
func nameLenGuard(next http.Handler) http.Handler {
	// Path prefixes that carry a collection/alias name as their next segment.
	prefixes := []string{
		"/v1/collections/",
		"/v1/multivector/",
		"/v1/named/",
		"/v1/aliases/",
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, p := range prefixes {
			if rest, ok := strings.CutPrefix(r.URL.Path, p); ok {
				seg := rest
				if i := strings.IndexByte(seg, '/'); i >= 0 {
					seg = seg[:i]
				}
				if len(seg) > maxCollectionNameLen {
					writeError(w, http.StatusBadRequest, "collection name too long")
					return
				}
				break
			}
		}
		next.ServeHTTP(w, r)
	})
}

// errorResponse is the JSON body returned for any non-2xx response.
type errorResponse struct {
	Error string `json:"error"`
}

func bearer(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if after, ok := strings.CutPrefix(h, "Bearer "); ok {
		return after
	}
	return h
}

// clientCN returns the VERIFIED mTLS client-cert CommonName for the request, or
// "" when the request arrived over plaintext or the peer presented no verified
// cert.
//
// SECURITY: Go's net/http populates r.TLS.PeerCertificates (and VerifiedChains)
// ONLY after a handshake that satisfied the server tls.Config's ClientAuth. With
// ClientAuth=RequireAndVerifyClientCert the handshake itself rejects any client
// cert that does not chain to the configured ClientCAs, so when a request reaches
// a handler PeerCertificates[0] is the CA-VERIFIED leaf — never an unverified
// presented cert and never a spoofable header. We additionally require a
// non-empty VerifiedChains so that under VerifyClientCertIfGiven an unverified
// cert yields no principal (CN "" → token-or-deny fallback). This CN is the ONLY
// source of the cert principal; no header (e.g. X-Client-CN) is ever trusted.
func clientCN(r *http.Request) string {
	if r.TLS == nil {
		return ""
	}
	if len(r.TLS.VerifiedChains) == 0 || len(r.TLS.PeerCertificates) == 0 {
		return ""
	}
	return r.TLS.PeerCertificates[0].Subject.CommonName
}

// authorize runs the optional Authenticator for (opName, args), writing 401 and
// returning false when it rejects. args is the binary op payload the handler is
// about to dispatch; the authorizer derives the target collection from it, so a
// per-collection scope can be enforced. The uniform "unauthorized" message never
// leaks whether the token is unknown vs merely lacks the scope.
func (a *api) authorize(w http.ResponseWriter, r *http.Request, opName string, args []byte) bool {
	if a.auth == nil || a.auth(authz.AuthRequest{Token: bearer(r), ClientCN: clientCN(r), Op: opName, Args: args}) {
		return true
	}
	writeError(w, http.StatusUnauthorized, "unauthorized")
	return false
}

// call dispatches an op after auth, writing an error response and returning
// ok=false on failure. On success it returns the raw result bytes.
func (a *api) call(w http.ResponseWriter, r *http.Request, opName string, args []byte) ([]byte, bool) {
	if !a.authorize(w, r, opName, args) {
		return nil, false
	}
	res, err := a.disp.Call(opName, args)
	if err != nil {
		writeDispatchError(w, opName, err)
		return nil, false
	}
	return res, true
}

// boolToU8 maps a Go bool onto the envelope wait byte (1 = wait, 0 = no-wait),
// matching the fanout dispatcher's `wait != 0` interpretation.
func boolToU8(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

// callWrite is the write-path variant of call: it engages tunable write
// consistency only when requested (wcf>0 OR wait=false), and is otherwise
// byte-identical to the plain call path. When write consistency is active it
// wraps the inner op in the __wc__ envelope so the fanout dispatcher unwraps it,
// dispatches the inner write through normal routing/Raft, then runs the
// post-commit barrier; otherwise it dispatches the plain op exactly as today.
// The default request (no WCF fields in the body → wcf=0, wait=true) NEVER
// builds an envelope, so the default write path is unchanged.
func (a *api) callWrite(w http.ResponseWriter, r *http.Request, opName string, args []byte, wcf int, wait bool) ([]byte, bool) {
	if wcf > 0 || !wait {
		// Authorize against the INNER write op name + INNER args (not the
		// "__wc__" envelope), so an authenticator scoped per op+collection behaves
		// identically with or without WCF. The envelope is built only AFTER this
		// passes.
		if !a.authorize(w, r, opName, args) {
			return nil, false
		}
		// Saturate to 255 before narrowing: a factor > RF clamps to RF at the
		// barrier, but a bare uint8(wcf) would WRAP (256→0), silently turning an
		// "all replicas" request into the no-barrier default. Saturating keeps an
		// over-large request as the strongest factor (then clamped to RF).
		if wcf > 255 {
			wcf = 255
		}
		env := ops.EncodeWCEnvelope(uint8(wcf), boolToU8(wait), opName, args) //nolint:gosec // saturated to 255 above; clamped to [1,RF] at the barrier
		res, err := a.disp.Call(ops.WCEnvelopeOp, env)
		if err != nil {
			writeDispatchError(w, opName, err)
			return nil, false
		}
		return res, true
	}
	return a.call(w, r, opName, args)
}

// Request-body size caps. Unlike the TCP transport (which hard-bounds every
// frame at MaxFrameSize) and gRPC (which inherits the 4 MiB default recv limit),
// the REST front door had no ceiling: a single client could POST a multi-GB JSON
// body that the handler then fully materializes on the heap, OOM-killing the
// process. http.MaxBytesReader caps the body at the read layer so an over-large
// request is rejected (413) before it is buffered.
//
//   - maxJSONBody:     the default cap for single-document handlers.
//   - maxBulkJSONBody: a larger cap for the bulk/batch routes (insert/get/upsert)
//     that legitimately carry many points in one request.
const (
	maxJSONBody     = 32 << 20  // 32 MiB
	maxBulkJSONBody = 256 << 20 // 256 MiB
)

// decodeBody reads the JSON request body into dst, writing 400 on failure. The
// body is capped at maxJSONBody.
func decodeBody(w http.ResponseWriter, r *http.Request, dst any) bool {
	return decodeBodyLimit(w, r, dst, maxJSONBody)
}

// decodeBodyBulk is decodeBody with the larger maxBulkJSONBody cap, for the
// bulk/batch routes that legitimately carry many points per request.
func decodeBodyBulk(w http.ResponseWriter, r *http.Request, dst any) bool {
	return decodeBodyLimit(w, r, dst, maxBulkJSONBody)
}

// decodeBodyLimit caps r.Body at max bytes before decoding so an over-large body
// is rejected rather than buffered into the heap. A body that exceeds the cap
// surfaces as a decode error → 413; any other malformed JSON stays a 400.
func decodeBodyLimit(w http.ResponseWriter, r *http.Request, dst any, max int64) bool {
	r.Body = http.MaxBytesReader(w, r.Body, max)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return false
		}
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg})
}

// writeDispatchError renders a dispatch error to the client. Errors that
// statusForError deliberately classifies as a client-facing 4xx (and the 503/504
// cluster-state cases) keep their descriptive message — those are validation /
// routing signals the caller needs. The catch-all 500 bucket, however, can wrap
// internal paths, partition/shard identifiers, leader addresses, or low-level
// failures; returning that verbatim leaks topology/implementation detail. So for
// a 500 we log the detail server-side and return a generic message to the client.
func writeDispatchError(w http.ResponseWriter, opName string, err error) {
	status, msg := clientError(opName, err)
	writeError(w, status, msg)
}

// clientError maps a dispatch error to the (status, client-facing message) pair
// the caller should see: statusForError's status, and either the verbatim error
// text or — for the catch-all 500 bucket — the redacted generic "internal error"
// (logged server-side). It is the single source of truth for the status-map +
// redaction contract so every write path (single-point writeDispatchError AND the
// batch loops in putPointsBatch) stays consistent and cannot drift (DRY).
func clientError(opName string, err error) (int, string) {
	status := statusForError(err)
	if status == http.StatusInternalServerError {
		slog.Error("internal error dispatching op", "transport", "http", "op", opName, "err", err)
		return status, "internal error"
	}
	return status, err.Error()
}

// writeInternalError logs an internal failure server-side and returns a generic
// 500 to the client, so result-decode and other internal errors never leak
// implementation detail. ctx is a short server-side tag (e.g. the op name).
func writeInternalError(w http.ResponseWriter, ctx string, err error) {
	slog.Error("internal error", "transport", "http", "ctx", ctx, "err", err)
	writeError(w, http.StatusInternalServerError, "internal error")
}

// statusForError maps a dispatch error onto an HTTP status: client mistakes
// (bad dim, empty filter/group field, unknown collection) become 4xx; a missing
// leader becomes 503; everything else is 500.
func statusForError(err error) int {
	switch {
	case errors.Is(err, vector.ErrDimMismatch),
		// A payload carrying a record value above the storage cap: 400, not 500.
		// The cap is the snapshot/WAL codec's, so accepting the write would mean
		// acking something that can never be made durable; refusing it is a
		// client-fixable mistake and the message is the caller's own data.
		// Keeps this classifier in sync with server.clientFacingErr.
		//
		// Matched by sentinel AND by exact message shape: a clustered apply
		// rebuilds the op error with errors.New across the replication boundary
		// (shard.decodePBResult), so errors.Is alone loses it there and the error
		// would fall through to the redacted 500 bucket. The message-shape arms
		// use vector.IsRecordTooLargeMessage, NOT strings.Contains — a bare
		// substring check would also match an unrelated internal error that
		// merely wraps the sentinel, leaking it to the caller unredacted.
		errors.Is(err, vector.ErrRecordTooLarge),
		vector.IsRecordTooLargeMessage(err.Error()),
		// Record bytes no operate engine can open: 400, not 500. Storing them
		// would poison every accelerated filter under that payload key for the
		// whole collection, so refusing at the door is the client-fixable
		// mistake. Keeps this classifier in sync with server.clientFacingErr.
		// Same sentinel-plus-message-shape matching, for the same
		// clustered-apply reason.
		errors.Is(err, vector.ErrRecordMalformed),
		vector.IsRecordMalformedMessage(err.Error()),
		// A vector_operate against a payload key holding a non-record value: 400,
		// not 500. Silently replacing that value would destroy data the caller can
		// still read, so the op refuses — a client-fixable mistake. Keeps this
		// classifier in sync with server.clientFacingErr.
		// Same sentinel-plus-message-shape matching, for the same
		// clustered-apply reason.
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
		// A search/text or hybrid-text request against a collection that was not
		// created with FullText: a usage/config error, not a server fault → 400.
		errors.Is(err, vector.ErrFullTextDisabled),
		// ops.ErrMalformedPayloadJSON: a bulk-staged per-point payload that framed
		// correctly but is not a metadata object. The staging route streams the
		// payload section through without unmarshalling it (that is most of the
		// speed), so the op decoder is the first thing to look inside a blob —
		// which means this arrives as a DISPATCH error rather than a transport one.
		// Same client mistake the batch route answers 400 for; the two routes must
		// not disagree about the status just because they validate in different
		// places. Message unredacted: it names the payload's index and the JSON
		// error, both of which the caller sent.
		errors.Is(err, ops.ErrMalformedPayloadJSON),
		// Alias-management validation errors (target missing / shadow / reserved
		// char / target-is-alias) all carry the "rostam: alias " prefix. They live
		// in the root rostam package (un-importable here without a cycle), so match
		// the message prefix rather than the sentinel.
		strings.Contains(err.Error(), "rostam: alias "),
		// cluster.ErrWASMUpdateUnsupported: a __register_wasm__ naming an op that is
		// already registered, and CHANGING ITS CONTRACT — its Kind or its key
		// extractor. Those two are read before any shard group is known, so unlike
		// the module itself they cannot be bound to a group's log prefix and are
		// frozen at first registration. A client mistake with a remedy — register
		// under a new name — not a transient or a server fault. (A module-only
		// update is supported and never reaches here.)
		// 400, and the message stays unredacted; it only echoes the op name
		// the caller already sent. Keeps this classifier in sync with
		// server.clientFacingErr as its contract requires — and the substring is a
		// CONST both packages import, so "in sync" is now enforced by the compiler
		// rather than by this comment.
		strings.Contains(err.Error(), ops.WASMUpdateUnsupportedMsg),
		// ops.ErrWASMOpNameUnsafe: a __register_wasm__ whose Name is not usable as
		// a bare filename (a path separator, "..", NUL). A caller mistake with an
		// obvious remedy → 400, message unredacted.
		strings.Contains(err.Error(), ops.WASMOpNameUnsafeMsg),
		// cluster.ErrWASMRegistrationRefused: a __register_wasm__ whose PAYLOAD is
		// refused at propose time — an encoded frame over the cap, a frame that does
		// not decode, a module over the cap, or a Kind byte outside {0,1}. Caller
		// mistakes with obvious remedies → 400, message unredacted (it names only
		// sizes and the op the caller sent). Kept in sync with
		// server.clientFacingErr by the shared const.
		strings.Contains(err.Error(), ops.WASMRegistrationRefusedMsg):
		return http.StatusBadRequest
	case errors.Is(err, ops.ErrVectorRecordAbsent),
		// The clustered path stringifies the sentinel across the Raft boundary, so
		// errors.Is stops matching there. ops.IsVectorRecordAbsentMessage, not
		// strings.Contains: a bare substring check would also match an unrelated
		// internal error that merely wraps the sentinel, leaking it unredacted.
		ops.IsVectorRecordAbsentMessage(err.Error()):
		// vector_operate with create = NONE against a payload key that holds no
		// record: the caller declined to create one and there was none, so the
		// thing they named does not exist → 404. This is the HTTP rendering of the
		// StatusNotFound the binary transport answers for the same signal (see
		// server.mapResult), which in turn matches what KV operate answers for
		// create = NONE against an absent key. Unclassified it was a redacted 500,
		// which reads as a server fault for an ordinary miss.
		return http.StatusNotFound
	case errors.Is(err, vector.ErrAPIKeyExists):
		// Online key-admin: POST /v1/admin/keys with an already-registered token.
		// 409 Conflict (the standard create-conflict code).
		return http.StatusConflict
	case errors.Is(err, vector.ErrAPIKeyNotFound):
		// Online key-admin: DELETE /v1/admin/keys for an unknown token → 404.
		return http.StatusNotFound
	case strings.Contains(err.Error(), "rostam: online key-admin unavailable"):
		// Online key-admin requested but no *vector.KeyRegistry is wired (open/dev
		// mode or the static -api-key authenticator). ErrKeyAdminUnavailable lives in
		// the root rostam package (un-importable here without a cycle), so match the
		// message prefix. 412 Precondition Failed: the server is mis-/under-configured
		// for this op (start with -keys-file), not a transient outage or a bad request.
		return http.StatusPreconditionFailed
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
		// ScaNN create knobs (Config.Validate): anisotropic_eta < 0 / NaN, soar on a
		// non-IVF index, soar_lambda < 0 / NaN, or pq_nbits not in {0,4,8} are all
		// client config mistakes -> 400, not 500.
		errors.Is(err, vector.ErrInvalidAnisotropicEta),
		errors.Is(err, vector.ErrInvalidSOAR),
		errors.Is(err, vector.ErrInvalidSOARLambda),
		errors.Is(err, vector.ErrInvalidPQNBits):
		// Create-collection config validation (Config.Validate): a bad dim/M/quant, or
		// a PQ-HNSW (quant=="pq") / IVF-PQ config the engine rejects at create (e.g.
		// quant=="pq" on an IVF index, or dim not divisible by quant_pq_m) is a client
		// mistake -> 400, not 500.
		return http.StatusBadRequest
	case strings.Contains(err.Error(), "op not registered in this shard group yet"):
		// cluster.ErrWASMOpNotInThisGroup. The op EXISTS on this server; what is
		// missing is proof that the target shard group's Raft log carries its
		// registration, so the node declines to propose an invocation into that log
		// (cluster.checkWASMRouteGate). That is transient in the normal case — the
		// registration is still fanning out to the remaining groups — so it is 503
		// and retryable, NOT the 404 the plainer "op not registered" substring below
		// would give it. This case must stay ABOVE that one: the message contains it.
		return http.StatusServiceUnavailable
	case errors.Is(err, vector.ErrNoNamed),
		strings.Contains(err.Error(), "unknown collection"),
		strings.Contains(err.Error(), "no collection"),
		// cluster.ErrUnknownOp ("cluster: op not registered") and
		// shard.ErrOpNotRegistered ("shard: op not registered") — one substring
		// covers both sentinels, which live in packages this one cannot import
		// without a cycle. The requested op does not exist on this server: a
		// client mistake (or a not-yet-replicated dynamic WASM registration),
		// not a server fault → 404, and the message stays unredacted since it
		// only echoes the op name the caller already sent. Keeps this classifier
		// in sync with server.clientFacingErr as its contract requires.
		strings.Contains(err.Error(), "op not registered"):
		return http.StatusNotFound
	case errors.Is(err, vector.ErrVersionConflict),
		strings.Contains(err.Error(), "version conflict"):
		// Optimistic-CAS precondition miss: the point's current version did not match
		// the request's expected_version. 409 Conflict (the standard CAS code) — the
		// caller re-reads the current version and retries. The string fallback covers
		// the clustered path where the sentinel is stringified across the Raft boundary.
		return http.StatusConflict
	case errors.Is(err, vector.ErrCollectionExists),
		errors.Is(err, vector.ErrDuplicateID),
		strings.Contains(err.Error(), "already exists"),
		// A re-create of a PARTITIONED collection reports "is already partitioned"
		// rather than "already exists" (embedded.go's partitioned-create path), so
		// it matched nothing here and fell through to the 500 default — which also
		// REDACTS the reason, leaving the caller an opaque "internal error" while
		// the real cause appeared only in the server log. It is the same routine
		// create-conflict as the line above: same 409, same unredacted message.
		// Re-running a quickstart script is the common way to hit it.
		strings.Contains(err.Error(), "is already partitioned"),
		strings.Contains(err.Error(), "already present"):
		// Routine create-conflicts, not server faults: a second CreateCollection for a
		// live name (ErrCollectionExists), or a default insert (upsert=false) of an id
		// that is already live (ErrDuplicateID). 409 Conflict is the standard
		// create-conflict code. String fallbacks cover the clustered path where the
		// sentinel is stringified across the Raft boundary (mirrors "version conflict").
		return http.StatusConflict
	case errors.Is(err, vector.ErrCollectionRateLimited),
		errors.Is(err, vector.ErrCollectionFull),
		strings.Contains(err.Error(), "rate limited"),
		strings.Contains(err.Error(), "collection full"):
		// Quota/rate-limit backpressure, not a server fault: the write hit the
		// collection's MaxInsertsPerSecond token bucket (ErrCollectionRateLimited) or
		// its MaxVectors/MaxBytes cap (ErrCollectionFull). 429 Too Many Requests tells
		// the caller to back off and retry, distinct from a 5xx (which standard retry
		// policies hammer, and which pages operators on client-side throttling). String
		// fallbacks cover the clustered/stringified path.
		return http.StatusTooManyRequests
	case errors.Is(err, ops.ErrOperateDuringReshard),
		// ops.ErrOperateDuringReshard: a vector_operate against a collection a
		// reshard is dual-writing. operate is not idempotent, so the store refuses
		// rather than double-applying the op-list, and the refusal is transient —
		// it lasts the minutes a reshard runs and the caller retries after
		// cutover. 503, the same bucket as the leadership/ownership transients
		// below and the same one the other retryable conditions on this transport
		// use (no Retry-After: none of them set one). Unclassified it was a
		// redacted 500, so the retryability docs/vector/filtering.md promises
		// never reached the caller.
		//
		// Sentinel AND exact message shape, for the usual clustered-apply reason;
		// ops.IsOperateDuringReshardMessage, not strings.Contains, so an internal
		// fault that merely mentions the refusal is not leaked.
		ops.IsOperateDuringReshardMessage(err.Error()):
		return http.StatusServiceUnavailable
	case strings.Contains(err.Error(), "not leader"),
		strings.Contains(err.Error(), "no leader"),
		strings.Contains(err.Error(), "no reachable owner"):
		// Leadership/ownership-transient and retryable -> 503. Either this node is not
		// the leader for the target shard ("not leader"), the client-forwarding layer
		// could not resolve a leader during an election window ("no leader known after
		// retries"), or no owner for the target shard was reachable
		// (cluster.ErrNoShardOwner, "no reachable owner for shard"). All must be 503
		// (not 500) so callers retry through the transient window; this also matters now
		// that 500 bodies are redacted to a generic message (the detailed text is no
		// longer leaked for callers to match).
		return http.StatusServiceUnavailable
	case strings.Contains(err.Error(), "cluster: write "):
		// Write-consistency barrier miss (*cluster.ErrWriteConsistency). The write
		// IS durably committed at Raft quorum but the requested
		// write_consistency_factor was not reached within the timeout — a distinct,
		// documented outcome, NOT a failure. 504 Gateway Timeout conveys "committed
		// at quorum, consistency factor not met"; the caller may safely retry
		// (inserts are idempotent by id) or accept the majority-durable write. Placed
		// after the "not leader" case so the more specific prefix wins and does not
		// collide with the 503/500 buckets.
		return http.StatusGatewayTimeout
	default:
		return http.StatusInternalServerError
	}
}
