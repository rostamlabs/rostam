# Security

Authentication and authorization gate all three transports through a single
chokepoint, and evaluation is **fail-closed**: an indeterminate decision is a
denial.

## Authentication modes

Precedence, highest first:

1. **`-keys-file`** — a JSON key registry with per-key RBAC scopes. The
   recommended mode.
2. **`-api-key`** — a single static superuser key (all actions, all
   collections). Prefer the `ROSTAM_API_KEY` env var over the flag so the
   secret isn't visible in `/proc`.
3. **None** — open mode, for development only. The server enforces this
   framing: with neither `-keys-file` nor `-api-key` set, it **refuses to start
   on a non-loopback bind** unless you pass `-insecure`. Loopback-only binds
   run open without ceremony.

Clients present the credential as a bearer token: `Authorization: Bearer
<token>` on HTTP, `authorization` metadata on gRPC, and the protocol-v2 frame
token on TCP (`ClientConfig.AuthToken`; 255-byte limit, so JWTs don't fit the
TCP transport).

## RBAC keys and scopes

Each key in the registry carries a token, an optional tenant, an optional mTLS
certificate CN binding, and a list of scopes:

```
<action>:<pattern>
```

- Actions: `read`, `write`, `admin`, or `*`.
- Patterns: an exact collection name, a prefix glob (`logs-*`), or `*`.
- Cluster-level operations (no collection resource) match only the bare `*`
  pattern.

Examples: `read:*` (read everything), `write:tenantA/*` (write tenant A's
collections), `*:*` (superuser).

Keys can be administered at runtime through admin-scoped endpoints — no restart:

```
POST   /v1/admin/keys    {"token":"...","tenant":"acme","scopes":["read:acme/*"]}
DELETE /v1/admin/keys    {"token":"..."}      # token in body, never in the path
GET    /v1/admin/keys                          # redacted: fingerprints only
```

`-audit-log` emits a structured JSON record to stderr for every authorization
decision (principals redacted to token fingerprints). It applies only under
`-keys-file` RBAC — with a single `-api-key` there is no per-decision record to
emit, so the flag is a no-op.

## Tenant isolation

Keys may carry a tenant. With `-tenant-isolation` set, that tenant becomes an
authoritative boundary enforced after scope checks — a key bound to `acme`
cannot touch `other-tenant/...` regardless of its scope patterns. See
[Collections, tenants & aliases](../concepts/collections.md#multi-tenancy).

## JWT bearer tokens

For HTTP and gRPC, the server can accept stateless JWTs instead of registry
tokens: `-jwt-public-key` (PEM; the algorithm is pinned by key type — RSA →
RS256, ECDSA-P256 → ES256, so `alg` confusion is off the table), with optional
`-jwt-issuer` and `-jwt-audience` claim validation. The JWT carries tenant and
scopes claims.

## TLS

One certificate configuration covers HTTP, gRPC, and TCP:

```sh
rostam-server ... -tls-cert server.pem -tls-key server-key.pem \
  [-tls-ca clients-ca.pem] [-tls-require-client-cert]
```

- `-tls-cert`/`-tls-key` enable TLS (≥ 1.2) on all client-facing listeners.
  Misconfiguration is an error at startup — never a silent plaintext fallback.
- `-tls-ca` verifies client certificates (mTLS). Without
  `-tls-require-client-cert`, a verified cert is accepted but a missing cert
  falls back to token auth; with it, the handshake requires a valid client
  cert.
- **mTLS as identity**: a key registry entry with `cert_cn` binds a key to a
  client certificate CN, authenticating the verified CN without a bearer token.

Go clients build their side with
`tlsutil.ClientTLS(caFile, certFile, keyFile, serverName)`.

**Inter-node TLS**: nodes dial peers using `-tls-node-cert`/`-tls-node-key`
(defaulting to the server cert), and `-node-cn-allowlist` optionally pins the
set of acceptable peer certificate CNs.

## Inter-node auth

In a cluster with auth enabled, nodes authenticate to each other with
`-internal-token` (same value on every node; prefer `ROSTAM_INTERNAL_TOKEN`).
Forwarded requests and admin/replication traffic present it — without it, an
authenticated cluster cannot function.

## Reporting vulnerabilities

Do not open public issues for security problems — use the
[private advisory form](https://github.com/rostamlabs/rostam/security/advisories/new)
or email **security@rostamlabs.com**. See
[`SECURITY.md`](https://github.com/rostamlabs/rostam/blob/main/SECURITY.md) for
scope and the disclosure process.
