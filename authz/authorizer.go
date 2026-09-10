// SPDX-License-Identifier: Apache-2.0

// Package authz is the RBAC authorization core for Rostam's data/control plane.
// It maps an incoming op (+ its args + the caller's identity) to a required
// (action, resource) pair and grants access iff the caller's API-key scopes
// cover it. The engine is the security boundary, so it is FAIL-CLOSED: every
// path whose action or resource cannot be determined, and every key with no
// matching scope, is DENIED.
//
// Import-graph note: this package imports BOTH ops and vector. `ops` already
// imports `vector` (ops -> vector); `vector` imports neither. So authz -> ops ->
// vector and authz -> vector are acyclic, and nothing in ops/vector imports
// authz. The transports (httpapi, grpcapi, server) can therefore import authz
// without creating a cycle. Putting this engine in `vector` is impossible: it
// needs `ops.Registry`, and `ops` already imports `vector`, so vector importing
// ops would form vector <-> ops.
package authz

import (
	"crypto/sha256"
	"crypto/subtle"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// AuthRequest is the unified authorization input shared by all transports
// (HTTP/gRPC/TCP). Token is the bearer credential (empty if none); ClientCN is
// the verified mTLS client-cert CommonName (empty if not mTLS-authenticated or
// unverified — a transport MUST only set this from a VERIFIED peer cert, never a
// spoofable header); Op is the inner op name (the __wc__ envelope is unwrapped
// before auth, so this is always the real op); Args is the inner op's wire args
// (used to derive the target collection).
type AuthRequest struct {
	Token    string
	ClientCN string
	Op       string
	Args     []byte
}

// Authenticator decides whether a request is allowed. true = ALLOW, false =
// DENY. This is the shared transport-facing type. A nil Authenticator means "no
// auth configured" (dev/open mode) and is interpreted by the transports, not
// here — this engine itself never returns true on an indeterminate input.
type Authenticator func(AuthRequest) bool

// Action constants — the three privilege tiers a scope's action field may carry.
const (
	ActionRead  = "read"
	ActionWrite = "write"
	ActionAdmin = "admin"
)

// scopeGrants reports whether a single scope string grants (action, resource).
//
// A scope is "<scopeAction>:<pattern>", split on the FIRST ':'. A malformed
// scope with no ':' grants nothing (deny). The scopeAction must equal action OR
// be "*" (wildcard action). The pattern matches resource by:
//   - exact equality (pattern == resource); OR
//   - pattern "*" (matches any resource, INCLUDING the empty/cluster resource); OR
//   - prefix-glob: pattern ends with '*' and resource has the pattern's prefix
//     (e.g. "default/docs*" matches "default/docs" and "default/docsX";
//     "default/*" matches "default/anything").
//
// Empty resource (cluster/no-collection ops) is matched ONLY by pattern "*"
// (or "*:*"); a specific-collection or namespace-glob pattern never matches it.
// This is enforced naturally: a non-"*" glob like "default/*" has a non-empty
// prefix "default/", which "" does not carry; an exact pattern can only equal ""
// if the pattern itself is "" (a malformed/empty pattern), which we reject below.
func scopeGrants(scope, action, resource string) bool {
	i := strings.IndexByte(scope, ':')
	if i < 0 {
		return false // malformed scope: no action:pattern separator -> deny
	}
	scopeAction := scope[:i]
	pattern := scope[i+1:]

	// Action must match exactly or be the wildcard action.
	if scopeAction != action && scopeAction != "*" {
		return false
	}

	// An empty pattern grants nothing (defensive: e.g. "read:" -> deny).
	if pattern == "" {
		return false
	}
	// Bare "*" matches any resource, including the empty cluster resource.
	if pattern == "*" {
		return true
	}
	// Prefix-glob: "<prefix>*" matches any resource carrying <prefix>. The
	// prefix is non-empty here (pattern != "*"), so "" never matches a glob.
	if strings.HasSuffix(pattern, "*") {
		prefix := pattern[:len(pattern)-1]
		return strings.HasPrefix(resource, prefix)
	}
	// Exact match. (resource=="" only matches pattern=="" which is rejected
	// above, so the empty resource never matches a specific pattern.)
	return pattern == resource
}

// keyGrants reports whether ANY of a key's scopes grants (action, resource).
// Empty scopes -> false (deny-by-default: a key with no scopes grants nothing).
func keyGrants(scopes []string, action, resource string) bool {
	for _, s := range scopes {
		if scopeGrants(s, action, resource) {
			return true
		}
	}
	return false
}

// adminOps is the hardcoded set of coordinator/admin op names that require the
// "admin" action. These ops are NOT routed by op-kind alone: some live in the
// ops registry as OpReadWrite (create/drop collection) but must require admin,
// and the resplit/reshard/alias/__rb_* coordinator ops are intercepted by the
// fan-out dispatcher and are NOT in the ops registry at all — so they must be
// enumerated here to be classified. Sources (verified by grep):
//
//   - vector_create_collection / vector_drop_collection: ops/builtin.go
//     (registered OpReadWrite, but collection lifecycle = admin).
//   - vector_mv_create_collection / vector_mv_drop_collection: ops/builtin.go.
//   - vector_named_create_collection / vector_named_drop_collection: ops/builtin.go.
//   - vector_resplit / vector_mv_resplit / vector_resplit_cleanup /
//     vector_mv_resplit_cleanup: fanout_dispatcher.go dispatch switch.
//   - vector_reshard / vector_mv_reshard / vector_reshard_abort /
//     vector_mv_reshard_abort: fanout_dispatcher.go dispatch switch.
//   - alias_batch / alias_list: fanout_dispatcher.go dispatch switch.
//   - __keys_add__ / __keys_revoke__ / __keys_list__: keysDispatcher
//     (keys_dispatcher.go) — online key-admin coordinator virtual-ops; admin so
//     only an admin-scoped key may mutate/list the API-key registry.
//   - __rb_*: cluster/admin_ops.go (rebalancing coordinator ops:
//     __rb_add_owner__, __rb_remove_owner__, __rb_add_voter__,
//     __rb_remove_voter__, __rb_transfer__, __rb_set_placement__,
//     __rb_status__, __rb_placement__). Matched by the "__rb_" prefix below so
//     any future __rb_* op is admin-gated by default (fail-closed).
var adminOps = map[string]struct{}{
	"vector_create_collection":       {},
	"vector_drop_collection":         {},
	"vector_mv_create_collection":    {},
	"vector_mv_drop_collection":      {},
	"vector_named_create_collection": {},
	"vector_named_drop_collection":   {},
	"vector_resplit":                 {},
	"vector_mv_resplit":              {},
	"vector_resplit_cleanup":         {},
	"vector_mv_resplit_cleanup":      {},
	"vector_reshard":                 {},
	"vector_mv_reshard":              {},
	"vector_reshard_abort":           {},
	"vector_mv_reshard_abort":        {},
	"alias_batch":                    {},
	"alias_list":                     {},
	// Online key-admin coordinator virtual-ops (intercepted in the keys
	// dispatcher, NOT in the ops registry). Admin-gated so only an admin-scoped
	// key may add/revoke/list API keys; a read/write/tenant-scoped key is denied
	// at this gate (action=admin, resource="" cluster -> only "*:*"/"admin:*"
	// covers it). See ops/keys.go for the op names.
	"__keys_add__":    {},
	"__keys_revoke__": {},
	"__keys_list__":   {},
	// WASM module registration (ops/wasm_register.go). Registered OpReadWrite in
	// the ops registry, but loading an attacker-supplied module that runs on
	// every replica's FSM Apply path (with host functions that address ANY key)
	// is strictly more privileged than a collection write — it is code loading.
	// Admin-gate it so a broad but non-admin write:* scope cannot register a
	// module; only an admin/superuser key (resource="" cluster -> "*:*"/"admin:*")
	// passes. Checked before the OpReadWrite registry fallthrough in actionFor.
	"__register_wasm__": {},
	// The INTERNAL shard-scoped leg of the registration broadcast
	// (cluster/wasm_broadcast.go). It carries the same attacker-supplied module
	// as __register_wasm__ — it is that op, addressed to one group — so it needs
	// the same privilege.
	//
	// It is admin TODAY without this entry, but only by accident: it is not in
	// the ops registry, so actionFor falls through to the deny-by-default "admin"
	// return. That is unpinned. __register_wasm__ ITSELF is registered
	// OpReadWrite, so the obvious future refactor — registering the wrapper
	// alongside it — would silently demote this one from admin to write and hand
	// module loading to any write:* key. Enumerating it makes the classification
	// independent of whether it is ever added to the registry.
	"__register_wasm_shard__": {},
	// The WASM BLOB TRANSPORT pair (cluster/wasm_blob_transport.go). Both are
	// enumerated here from the start, rather than relying on absence from the ops
	// registry, for the reason spelled out above __register_wasm_shard__: absence
	// is not a classification, it is a coincidence that the next refactor can
	// remove. See authz/wasm_blob_authz_test.go.
	//
	// __wasm_blob_put__ carries attacker-supplied module bytes and writes them
	// into the node's code store. It is __register_wasm__'s payload without the
	// registration, so it needs __register_wasm__'s privilege; anything less hands
	// a write:* key the ability to plant code on every node and wait for a
	// fingerprint to point at it.
	//
	// __wasm_blob_get__ RETURNS module bytes, and admin is the deliberate answer
	// to a genuinely arguable case. The bytes are not secret BETWEEN CLUSTER
	// MEMBERS — every member is entitled to them, and the only legitimate caller
	// is a peer carrying the internal service token, which is superuser and never
	// consults this map. But "a member already has them" is not the boundary this
	// engine draws; it draws one over KEY PRIVILEGE, and an external read:* client
	// is not a member. Three things decide it:
	//
	//   - it is an unauthenticated-read primitive over the blob store if the
	//     gating is wrong. What confines it to <dataDir>/wasm/blobs/<hex>.wasm is
	//     one validator (cluster.wasmBlobPath). Admin means a regression there
	//     needs an admin key to exploit rather than any read scope in the cluster;
	//   - the bytes are the operator's UDF logic. A key scoped to one collection's
	//     reads should not be able to enumerate and exfiltrate every module the
	//     cluster runs;
	//   - splitting the pair's privilege is how the pair gets misread later. put
	//     and get are one mechanism, and a reviewer who finds get at "read" will
	//     reasonably infer the bytes are public.
	//
	// The cost of admin here is zero: no non-peer caller has any reason to issue
	// it, and peers are exempt by the internal-token grant.
	"__wasm_blob_put__": {},
	"__wasm_blob_get__": {},
	// The INTERNAL shard-scoped leg of the KV flush broadcast
	// (cluster/flush_broadcast.go). It drives a flush into ONE group and is
	// reachable only over the internal-token peer path in normal operation, but is
	// enumerated here — rather than relying on absence from the ops registry — for
	// the reason spelled out above __register_wasm_shard__: absence is a
	// coincidence a future refactor can remove, not a classification. `flush` ITSELF
	// is a global-WRITE op (registered OpReadWrite, empty resource); the wrapper
	// pins the shard-scoped leg at admin so it can never be silently demoted to
	// write and handed to any write:* key.
	"__flush_shard__": {},
	// The KV record-index catalog ops (cluster/kv_index_admin.go). Enumerated for
	// the reason spelled out above __flush_shard__: none of the three is in the
	// ops registry today, so all three would be admin by actionFor's
	// deny-by-default fallthrough — by coincidence, not by decision, and one
	// natural refactor (registering them so they dispatch through the registry
	// rather than through n.adminOps) removes it silently.
	//
	// __kv_index_set__ CREATES AND DROPS INDEXES CLUSTER-WIDE through the meta
	// log. Dropping one is what makes it admin rather than write: every query
	// naming that index starts failing, and on a large keyspace re-creating it
	// costs a full-cache walk on every node. A schema-shaped operation, not a
	// data-shaped one.
	//
	// __kv_index_ready__ is the internal per-group readiness leaf. Its only
	// legitimate caller is a peer, which carries the internal service token and is
	// granted before this map is consulted, so admin costs the gather nothing.
	//
	// __kv_index_list__ is a READ of the catalog and is pinned at admin only
	// because that is what it is today; demoting it to the read set is a
	// deliberate decision for the client work, not something a refactor should
	// make by accident.
	"__kv_index_set__":   {},
	"__kv_index_list__":  {},
	"__kv_index_ready__": {},
	// The INTERNAL shard-scoped leg of the kv_query fan-out
	// (cluster/kv_query_broadcast.go), enumerated for the same reason as the three
	// above. `kv_query` ITSELF is an ordinary OpReadOnly and is classified as a
	// read from the registry; the WRAPPER is pinned at admin so it can never be
	// demoted to read and become a way for a read:* key to address ONE shard group
	// directly — bypassing the coordinator, which is the only thing that makes a
	// page a complete answer rather than one group's slice of it.
	"__kv_query_shard__": {},
}

// readOps is the small set of cluster-introspection ops that are explicitly
// "read": __ping__ (liveness), __topology__ (cluster map) and __collections__
// (the dashboard's dense-collection list). All are also registered OpReadOnly,
// but enumerating them keeps the classification explicit and independent of
// registration order.
var readOps = map[string]struct{}{
	"__ping__":        {},
	"__topology__":    {},
	"__collections__": {},
}

// actionFor returns the required action ("read"|"write"|"admin") for op.
//
// Order (deny-by-default / highest-privilege-first):
//  1. If op is in the admin set (or carries the "__rb_" coordinator prefix) ->
//     "admin". Checked FIRST because create/drop are registered OpReadWrite in
//     the registry but must require admin.
//  2. Else if op is __ping__/__topology__ -> "read".
//  3. Else look it up in the ops registry: OpReadWrite -> "write", OpReadOnly ->
//     "read".
//  4. Else (unknown op, not in any set, not in the registry) -> "admin": an op
//     we cannot classify requires the highest privilege so an unmapped/new op is
//     never silently granted to a low-privilege key.
func actionFor(op string, reg *ops.Registry) string {
	if _, ok := adminOps[op]; ok {
		return ActionAdmin
	}
	if strings.HasPrefix(op, "__rb_") {
		return ActionAdmin
	}
	if _, ok := readOps[op]; ok {
		return ActionRead
	}
	if reg != nil {
		if _, kind, _, ok := reg.Lookup(op); ok {
			if kind == ops.OpReadWrite {
				return ActionWrite
			}
			return ActionRead
		}
	}
	// Unknown / unmapped op (or nil registry) -> highest privilege. Fail-closed.
	return ActionAdmin
}

// resourceFor returns the LOGICAL canonical collection an op targets, or ""
// (the cluster/no-collection resource) when the op is not collection-keyed.
//
// ops.CollectionNameFor yields the canonical name (e.g. "default/docs"), or a
// PHYSICAL partitioned name ("default/docs#3", "default/docs@1#0") for a
// physical route. The partition/generation suffix is stripped (cut at the first
// '#' or '@') so a scope on the logical collection "default/docs" covers all its
// partitions but never a different collection. This mirrors the fan-out
// dispatcher's strings.ContainsAny(name, "#@") physical-name short-circuit.
//
// SECURITY: this strip is safe because '#'/'@' are REJECTED in user collection
// names at every create edge (embedded.go / client.go / httpapi guards), so a
// caller cannot craft "default/evil#docs" to alias onto another collection's
// scope — the cut is at the FIRST such char, yielding "default/evil" regardless.
func resourceFor(op string, args []byte) string {
	name, ok := ops.CollectionNameFor(op, args)
	if !ok {
		return "" // KV / cluster-admin op without a collection -> cluster resource
	}
	if i := strings.IndexAny(name, "#@"); i >= 0 {
		return name[:i]
	}
	return name
}

// resourceTenant returns the tenant a canonical resource belongs to (the
// substring before the first '/'), or "" for the empty/cluster resource or a
// resource with no tenant prefix. It is a thin alias for vector.TenantOf so the
// tenant guard's parse is the SAME single source of truth as the storage-layout
// tenant parse (splitTenant) — the guard can never disagree with how a
// collection is actually keyed, so a malformed resource name cannot fool it.
func resourceTenant(resource string) string {
	return vector.TenantOf(resource)
}

// RBACOptions carries the optional, opt-in knobs of the RBAC authenticator.
// Its zero value is the historical default (no audit, no tenant isolation), so
// the un-optioned constructors below stay byte/behaviour-identical to before.
type RBACOptions struct {
	// Audit, if non-nil, receives one redacted AuditRecord per decision. nil =
	// auditing disabled (zero-cost fast path; the default).
	Audit AuditLogger
	// TenantIsolation, when true, turns APIKey.Tenant into an AUTHORITATIVE
	// resource boundary: after the scope-match GRANTS, the request is allowed
	// only if the requested resource's tenant == the key's Tenant (or the key is
	// the cross-tenant "*" marker). This guard ONLY ever removes access a scope
	// already granted — it NEVER adds. false (the default) skips the guard
	// entirely: the verdict is byte-identical to the scope-only engine.
	TenantIsolation bool
	// JWTVerifier, if non-nil, enables the OPT-IN stateless JWT bearer path: when
	// a bearer token is NOT found in the registry AND it looks like a JWT, it is
	// verified against this verifier (signature + alg-pin + exp/nbf/iss/aud +
	// required tenant/scopes). A verified JWT yields a synthetic principal
	// (Tenant + Scopes from its claims) that runs through the EXACT SAME RBAC
	// scope-match + tenant-isolation guard + audit as a registry key. A verify
	// FAILURE is fail-closed: deny + audit (Reason="jwt-invalid"), NEVER a
	// fallthrough to the cert-CN path. nil (the default) = JWT-off: a JWT-looking
	// token simply fails the registry lookup and is denied, byte-identical to the
	// pre-JWT engine. JWT rides the existing Authorization: Bearer credential, so
	// it is HTTP/gRPC only (the TCP 255B token cap cannot carry a JWT).
	JWTVerifier *JWTVerifier

	// NodeCNAllowlist is the OPT-IN per-node mTLS identity allowlist for the
	// SERVER-SIDE gate on the internal-token superuser grant (defense-in-depth
	// mirroring the inter-node CLIENT verify in cluster/peerClient). When
	// non-empty, the internal-token grant ALSO requires the caller's VERIFIED
	// ClientCN to be in this set — so a LEAKED internal token alone no longer
	// authenticates: the caller must ALSO present an allowlisted per-node client
	// cert. A non-allowlisted/absent ClientCN is DENIED (audit Reason
	// "peer-cn-unlisted"). Empty/nil (the default) = OFF = BYTE-IDENTICAL: the
	// internal-token grant returns true exactly as before, and ALL non-internal
	// paths are untouched in every case.
	NodeCNAllowlist map[string]bool
}

// NewRBACAuthenticator builds the RBAC Authenticator closure. reg resolves
// principals (by token, or by verified cert CN); opsReg classifies op actions;
// internalToken (if non-empty) is the inter-node service principal that is
// treated as superuser so forwarded ops carry a trusted identity.
//
// A nil reg makes the closure DENY every non-internal request (fail-closed: no
// principal store = nobody can be authorized). nil opsReg is tolerated:
// actionFor then classifies every registry-only op as admin (fail-closed).
//
// Deny-by-default at EVERY branch — see the walk-through in the package tests.
//
// This is the zero-options form: it delegates to NewRBACAuthenticatorOpts with
// the zero RBACOptions, so existing callers are unchanged and pay no added cost.
func NewRBACAuthenticator(reg *vector.KeyRegistry, opsReg *ops.Registry, internalToken string) Authenticator {
	return NewRBACAuthenticatorOpts(reg, opsReg, internalToken, RBACOptions{})
}

// NewRBACAuthenticatorWithAudit is NewRBACAuthenticator plus an optional
// AuditLogger. It delegates to NewRBACAuthenticatorOpts with only Audit set, so
// the audit-only behaviour (and every existing caller/test) is unchanged.
func NewRBACAuthenticatorWithAudit(reg *vector.KeyRegistry, opsReg *ops.Registry, internalToken string, logger AuditLogger) Authenticator {
	return NewRBACAuthenticatorOpts(reg, opsReg, internalToken, RBACOptions{Audit: logger})
}

// NewRBACAuthenticatorOpts is the full constructor: NewRBACAuthenticator plus
// the opt-in RBACOptions (audit + tenant isolation). The RBAC SCOPE verdict is
// computed EXACTLY as in the un-optioned form — audit is pure observation and
// the tenant guard only ever DOWNGRADES a scope-grant to a deny (never the
// reverse). When opts is the zero value the closure is byte/behaviour-identical
// to the historical engine (audit fast path + guard skipped entirely).
//
// Audit (opts.Audit != nil): the closure emits exactly ONE AuditRecord per
// request at the point the FINAL verdict is known, naming the deciding branch
// via Reason (internal-token / token-not-found / cert-not-found / no-principal /
// scope-miss / scope-match / tenant-mismatch). The principal is REDACTED —
// never the raw token: it is the key's CertCN, else "token:"+fingerprint, else
// the tenant, with a separate non-reversible TokenFP. The internal-token grant
// records principal="internal" and TokenFP="" (never fingerprinted). No lock is
// held across the verdict — only the logger's own mutex guards its write.
//
// Tenant isolation (opts.TenantIsolation): see RBACOptions.TenantIsolation. The
// guard runs ONLY after a scope-match grant, and only for a real (non-internal)
// principal — the internal-token early-return is untouched (already exempt).
func NewRBACAuthenticatorOpts(reg *vector.KeyRegistry, opsReg *ops.Registry, internalToken string, opts RBACOptions) Authenticator {
	logger := opts.Audit
	tenantIsolation := opts.TenantIsolation
	jwtVerifier := opts.JWTVerifier
	nodeCNAllowlist := opts.NodeCNAllowlist
	return func(req AuthRequest) bool {
		// 1. Internal service principal (inter-node forward identity) -> superuser.
		// The internal token is the highest-value secret in the system (it grants
		// unconditional superuser across every tenant/collection), so it is compared
		// in CONSTANT TIME to remove the byte-by-byte timing oracle a plain == leaks
		// over the (this round) plaintext inter-node transport. We compare the
		// fixed-size SHA-256 digests of both sides so even the token LENGTH is not
		// leaked (the digest is always 32 bytes regardless of input length).
		if internalToken != "" && constantTimeTokenEqual(req.Token, internalToken) {
			// OPT-IN per-node identity (SERVER-SIDE gate; defense-in-depth). When an
			// allowlist is configured, the internal token alone is NOT sufficient: the
			// caller must ALSO present a VERIFIED, allowlisted client-cert CN (so a
			// leaked token + a non-allowlisted/absent cert is denied). Empty allowlist
			// => this block is skipped => the grant is BYTE-IDENTICAL to before.
			if len(nodeCNAllowlist) > 0 && (req.ClientCN == "" || !nodeCNAllowlist[req.ClientCN]) {
				if logger != nil {
					// Principal names the (rejected) peer CN, NOT the token: the cert CN
					// is not a secret; the internal token is NEVER logged/fingerprinted.
					principal := "internal:unlisted"
					if req.ClientCN != "" {
						principal = "internal:" + req.ClientCN
					}
					logger.Audit(AuditRecord{
						Time:      time.Now(),
						Principal: principal,
						Action:    actionFor(req.Op, opsReg),
						Resource:  resourceFor(req.Op, req.Args),
						Op:        req.Op,
						Decision:  "deny",
						Reason:    "peer-cn-unlisted",
					})
				}
				return false
			}
			if logger != nil {
				// principal="internal"; do NOT fingerprint the internal secret.
				logger.Audit(AuditRecord{
					Time:      time.Now(),
					Principal: "internal",
					Action:    actionFor(req.Op, opsReg),
					Resource:  resourceFor(req.Op, req.Args),
					Op:        req.Op,
					Decision:  "allow",
					Reason:    "internal-token",
				})
			}
			return true
		}

		// 2. Resolve the principal. Token wins; else a verified cert CN; else deny.
		if reg == nil {
			// No principal store -> deny. Redact the (unknown) token to a
			// fingerprint so probes correlate without leaking the secret.
			if logger != nil {
				logger.Audit(AuditRecord{
					Time:      time.Now(),
					Principal: "unknown",
					TokenFP:   tokenFingerprint(req.Token),
					Action:    actionFor(req.Op, opsReg),
					Resource:  resourceFor(req.Op, req.Args),
					Op:        req.Op,
					Decision:  "deny",
					Reason:    "no-principal",
				})
			}
			return false
		}
		var key vector.APIKey
		switch {
		case req.Token != "":
			k, ok := reg.Lookup(req.Token)
			if !ok {
				// OPT-IN JWT path: the bearer is NOT a registry token. If a JWT
				// verifier is configured AND the token looks like a JWT, verify it
				// and (on success) run the synthesized principal through the SAME
				// grant tail. This is tried ONLY on a registry MISS, so the
				// internal-token, registry-hit, and cert-CN paths are untouched;
				// when no verifier is configured this whole block is skipped and a
				// JWT-looking token falls through to the token-not-found deny below
				// — byte-identical to the pre-JWT engine.
				if jwtVerifier != nil && looksLikeJWT(req.Token) {
					fp := tokenFingerprint(req.Token)
					claims, err := jwtVerifier.VerifyAndExtract(req.Token)
					if err != nil {
						// FAIL-CLOSED: a JWT-verify failure is a DENY. Do NOT fall
						// through to the cert-CN path (a JWT is a non-empty token,
						// so it has no cert-CN business). The raw JWT NEVER appears
						// in the record — only its fingerprint.
						if logger != nil {
							logger.Audit(AuditRecord{
								Time:      time.Now(),
								Principal: "jwt:" + fp,
								TokenFP:   fp,
								Action:    actionFor(req.Op, opsReg),
								Resource:  resourceFor(req.Op, req.Args),
								Op:        req.Op,
								Decision:  "deny",
								Reason:    "jwt-invalid",
							})
						}
						return false
					}
					// Verified JWT -> synthesize a principal carrying the claim's
					// tenant + scopes (Token is a non-secret placeholder; the raw
					// JWT is never stored on the key). Principal label = sub claim
					// if present, else "jwt:"+fp. Runs the EXACT SAME grant tail
					// (scope-match + tenant guard + audit) as a registry key.
					principal := claims.Sub
					if principal == "" {
						principal = "jwt:" + fp
					}
					jwtKey := vector.APIKey{Token: "(jwt)", Tenant: claims.Tenant, Scopes: claims.Scopes, CertCN: ""}
					return grantTail(logger, opsReg, tenantIsolation, req, jwtKey, principal, fp, "jwt-")
				}
				if logger != nil {
					logger.Audit(AuditRecord{
						Time:      time.Now(),
						Principal: "unknown",
						TokenFP:   tokenFingerprint(req.Token),
						Action:    actionFor(req.Op, opsReg),
						Resource:  resourceFor(req.Op, req.Args),
						Op:        req.Op,
						Decision:  "deny",
						Reason:    "token-not-found",
					})
				}
				return false // unknown token -> deny
			}
			key = k
		case req.ClientCN != "":
			k, ok := reg.LookupByCN(req.ClientCN)
			if !ok {
				if logger != nil {
					// CN-auth: no token, so no fingerprint.
					logger.Audit(AuditRecord{
						Time:      time.Now(),
						Principal: "unknown",
						Action:    actionFor(req.Op, opsReg),
						Resource:  resourceFor(req.Op, req.Args),
						Op:        req.Op,
						Decision:  "deny",
						Reason:    "cert-not-found",
					})
				}
				return false // unknown/unbound cert CN -> deny
			}
			key = k
		default:
			if logger != nil {
				logger.Audit(AuditRecord{
					Time:      time.Now(),
					Principal: "unknown",
					Action:    actionFor(req.Op, opsReg),
					Resource:  resourceFor(req.Op, req.Args),
					Op:        req.Op,
					Decision:  "deny",
					Reason:    "no-principal",
				})
			}
			return false // no token and no cert CN -> deny
		}

		// 3-5. Run the resolved registry key through the shared grant tail:
		// scope-match + tenant-isolation guard + audit. The registry path keeps
		// its exact principal (cert CN, else "token:"+fp, else tenant), TokenFP
		// (the bearer fingerprint), and reason vocabulary ("scope-match" /
		// "scope-miss" / "tenant-mismatch") — byte-identical to the prior inline
		// tail.
		fp := tokenFingerprint(req.Token)
		principal := key.CertCN
		if principal == "" {
			if fp != "" {
				principal = "token:" + fp
			} else {
				principal = key.Tenant
			}
		}
		return grantTail(logger, opsReg, tenantIsolation, req, key, principal, fp, "")
	}
}

// resourceIndeterminate reports whether op is a VECTOR collection op (a "vector_"
// op registered routable, so it carries a non-nil KeyExtractor) yet its args did
// NOT yield a collection name — the "op is collection-keyed but args unparseable"
// case, where the engine could not determine the op's true target so the empty
// resource resourceFor returned must NOT be trusted as a genuine cluster op.
//
// It is deliberately limited to the "vector_" data-plane namespace: the cluster
// KV ops (get/put/del/...) are ALSO registered routable (non-nil KeyExtractor)
// but operate on the cluster KV namespace with a genuine empty resource, so they
// must keep their real resource — flagging them would wrongly demand admin for an
// ordinary write:* KV write. A nil registry, an unknown op, a non-vector op, or a
// shardless op all return false.
func resourceIndeterminate(op string, args []byte, reg *ops.Registry) bool {
	if reg == nil {
		return false
	}
	if !strings.HasPrefix(op, "vector_") {
		return false
	}
	_, _, ke, ok := reg.Lookup(op)
	if !ok || ke == nil {
		// Unknown op (already admin via actionFor) or a shardless op with no
		// collection — its empty resource is genuine, not indeterminate.
		return false
	}
	// The op is a routable vector op. It is indeterminate iff the collection name
	// could not be extracted from these args.
	_, named := ops.CollectionNameFor(op, args)
	return !named
}

// constantTimeTokenEqual reports whether two tokens are equal in constant time.
// It hashes each side to a fixed-size SHA-256 digest first and compares the
// digests with subtle.ConstantTimeCompare, so the comparison leaks neither a
// prefix-match oracle (the byte-by-byte short-circuit of ==) NOR the token
// length (the digest is always 32 bytes). Used only for the highest-privilege
// internal-token superuser grant; the registry path is a map lookup of
// high-entropy opaque tokens and does not need this.
func constantTimeTokenEqual(a, b string) bool {
	ha := sha256.Sum256([]byte(a))
	hb := sha256.Sum256([]byte(b))
	return subtle.ConstantTimeCompare(ha[:], hb[:]) == 1
}

// SecureTokenEqual compares two secret tokens in constant time (SHA-256 then
// subtle.ConstantTimeCompare) — leaking neither a prefix-match timing oracle
// nor the token length. Exported for callers outside this package that compare
// a static secret against a request token (e.g. the -api-key authenticator),
// which must NOT use a plain == (a byte-by-byte timing oracle on the key).
func SecureTokenEqual(a, b string) bool { return constantTimeTokenEqual(a, b) }

// grantTail is the SHARED authorization tail invoked by BOTH the registry-key
// path and the JWT path once a principal (a resolved or synthesized vector.APIKey)
// is in hand. It computes the required action + target resource, applies the
// scope-match grant, applies the opt-in tenant-isolation guard (which can only
// DOWNGRADE a grant to a deny, never the reverse), and emits exactly ONE audit
// record naming the deciding branch. It returns the final verdict.
//
// reasonPrefix lets the JWT path tag its audit reasons ("jwt-scope-match" /
// "jwt-scope-miss" / "jwt-tenant-mismatch") while the registry path passes "" to
// keep its historical reason vocabulary. principal and tokenFP are computed by
// the caller (the registry path uses cert CN / "token:"+fp / tenant; the JWT path
// uses the sub claim or "jwt:"+fp) — the raw token is NEVER passed here.
func grantTail(logger AuditLogger, opsReg *ops.Registry, tenantIsolation bool, req AuthRequest, key vector.APIKey, principal, tokenFP, reasonPrefix string) bool {
	action := actionFor(req.Op, opsReg)
	resource := resourceFor(req.Op, req.Args)

	// Indeterminate-resource fail-closed: a data-plane collection op (vector_*, a
	// read/write action) whose args could not be parsed into a collection name has
	// resourceFor collapse to "" (the cluster resource). Without this guard a
	// tenant-wide wildcard scope (write:* / *:*) would match the empty resource via
	// the bare-"*" rule and AUTHORIZE an op whose true target the engine FAILED to
	// determine — violating the package's deny-by-default-on-indeterminate invariant
	// regardless of the tenant-isolation flag. Upgrade the required action to admin so
	// a malformed collection-keyed op can no longer be authorized by a tenant
	// wildcard; only a genuine admin/superuser key passes. Scoped to the read/write
	// data plane: a genuine cluster op (already admin, or a non-vector KV op) keeps
	// its real empty resource and is untouched.
	if resource == "" && (action == ActionRead || action == ActionWrite) &&
		resourceIndeterminate(req.Op, req.Args, opsReg) {
		action = ActionAdmin
	}

	// Grant iff a scope covers (action, resource). Empty scopes -> deny.
	allowed := keyGrants(key.Scopes, action, resource)

	// Default Reason mirrors the scope verdict (scope-only engine).
	reason := reasonPrefix + "scope-miss"
	if allowed {
		reason = reasonPrefix + "scope-match"
	}

	// Tenant-isolation guard (opt-in; default OFF skips this entirely so the
	// verdict is byte-identical to the scope-only engine). The guard ONLY ever
	// turns a scope-GRANT into a deny — it never grants what the scope denied. It
	// runs only when a scope already granted; a scope-miss is already a deny and
	// the guard cannot change that.
	if tenantIsolation && allowed {
		switch {
		case key.Tenant == "*":
			// Cross-tenant/admin marker: exempt — no tenant restriction.
		case resource == "":
			// Cluster/no-collection resource (KV / cluster-admin op). A
			// tenant-scoped key has no cluster-admin business; only the "*"
			// cross-tenant key (handled above) may reach a cluster resource
			// under isolation. Fail-closed: downgrade to deny.
			allowed = false
			reason = reasonPrefix + "tenant-mismatch"
		case resourceTenant(resource) != key.Tenant:
			// Scope granted, but the resource belongs to a different tenant
			// than this key's Tenant -> the scope escaped its tenant boundary.
			// Fail-closed: downgrade to deny.
			allowed = false
			reason = reasonPrefix + "tenant-mismatch"
		}
	}

	if logger != nil {
		decision := "deny"
		if allowed {
			decision = "allow"
		}
		// Exactly ONE record per request: the guard mutated `allowed`/`reason`
		// in place above, so the final verdict (incl. a tenant-mismatch deny)
		// is emitted here once — never an allow followed by a deny.
		logger.Audit(AuditRecord{
			Time:      time.Now(),
			Principal: principal,
			TokenFP:   tokenFP,
			Tenant:    key.Tenant,
			Action:    action,
			Resource:  resource,
			Op:        req.Op,
			Decision:  decision,
			Reason:    reason,
		})
	}
	return allowed
}
