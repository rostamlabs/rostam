// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"errors"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// ErrKeyAdminUnavailable is returned by the keys coordinator virtual-ops when no
// *vector.KeyRegistry is wired into the server (open/dev mode, or the static
// -api-key authenticator, which has no mutable registry). It is fail-loud: the
// keys ops never silently no-op when there is nothing to mutate.
var ErrKeyAdminUnavailable = errors.New("rostam: online key-admin unavailable (no keys registry configured; start with -keys-file)")

// keysDispatcher decorates an inner dispatcher to intercept the online key-admin
// coordinator virtual-ops (__keys_add__/__keys_revoke__/__keys_list__) and apply
// them DIRECTLY to the wired *vector.KeyRegistry — the SAME registry instance the
// authenticator reads, so concurrent auth-reads and admin-writes share the
// registry's RWMutex. Every other op passes through byte-identically to inner.
//
// This mirrors the alias_batch/alias_list coordinator-virtual-op pattern: the
// keys ops are NOT shard-routed, NOT forwarded to a leader, and NOT in the ops
// registry. v1 is per-node-local: a keys mutation takes effect (and persists via
// the registry's atomic keys-file flush) on the RECEIVING node only. Cluster-wide
// propagation via meta-Raft is a documented v2 follow-up.
//
// Admin-gating is handled upstream by the authorize gate (authz classifies the
// three op names as admin), so by the time Call runs here the caller has already
// passed the admin-scope check. When reg is nil the ops fail loud with
// ErrKeyAdminUnavailable.
type keysDispatcher struct {
	inner interface {
		Call(name string, args []byte) ([]byte, error)
		LeaderAddr() string
	}
	reg *vector.KeyRegistry
}

// newKeysDispatcher wraps inner so the three __keys_*__ ops hit reg. reg may be
// nil (open/dev or static-key mode): the keys ops then return
// ErrKeyAdminUnavailable while every other op still passes through. The wrapper
// is always installed (zero behaviour change for non-keys ops), so the keys ops
// are reachable over every transport regardless of registry presence.
func newKeysDispatcher(inner interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}, reg *vector.KeyRegistry) *keysDispatcher {
	return &keysDispatcher{inner: inner, reg: reg}
}

// LeaderAddr delegates to the wrapped dispatcher.
func (k *keysDispatcher) LeaderAddr() string { return k.inner.LeaderAddr() }

// Call intercepts the three keys virtual-ops and passes everything else through.
func (k *keysDispatcher) Call(name string, args []byte) ([]byte, error) {
	switch name {
	case ops.OpKeysAdd:
		return k.handleAdd(args)
	case ops.OpKeysRevoke:
		return k.handleRevoke(args)
	case ops.OpKeysList:
		return k.handleList(args)
	default:
		return k.inner.Call(name, args)
	}
}

// appendCaller is the optional allocation-free path a dispatcher may offer
// (the shape of server.AppendDispatcher, spelled here to avoid importing the
// transport into this file).
type appendCaller interface {
	CallAppend(name string, args, dst []byte) ([]byte, error)
}

// keysAppendDispatcher is keysDispatcher PLUS that optional path, and exists as a
// separate type because a Go method set is static. If keysDispatcher itself
// carried CallAppend it would advertise the append path even when wrapping a
// dispatcher that cannot append — the cluster fanout — and the transport would
// then allocate a per-connection reply buffer AND copy the whole reply into it,
// strictly worse than the plain Call path it replaced. wrapKeysDispatcher picks
// the type that matches what the inner dispatcher can actually do.
type keysAppendDispatcher struct {
	*keysDispatcher
	inner appendCaller
}

// CallAppend serves the three keys ops locally — they build their own frames and
// are administrative, not hot — and forwards everything else to the inner
// dispatcher's append path.
func (k *keysAppendDispatcher) CallAppend(name string, args, dst []byte) ([]byte, error) {
	switch name {
	case ops.OpKeysAdd, ops.OpKeysRevoke, ops.OpKeysList:
		out, err := k.Call(name, args)
		if err != nil {
			return nil, err
		}
		return append(dst, out...), nil
	}
	return k.inner.CallAppend(name, args, dst)
}

// wrapKeysDispatcher installs the keys decorator while PRESERVING whether the
// inner dispatcher offers the append path. NewServer always installs this
// decorator before handing the dispatcher to the epoll transport, so a wrapper
// that dropped the capability would kill the append path (single-node), and one
// that faked it would add a reply copy to every response (cluster).
func wrapKeysDispatcher(inner interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}, reg *vector.KeyRegistry) interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
} {
	k := newKeysDispatcher(inner, reg)
	if ac, ok := inner.(appendCaller); ok {
		return &keysAppendDispatcher{keysDispatcher: k, inner: ac}
	}
	return k
}

// handleAdd decodes the request and registers the key on the live registry. The
// raw token is consumed by AddKey and never echoed back: the ack is empty.
// AddKey validates (non-empty token+tenant, no dup, known perms) and flushes the
// keys file atomically before returning, so a success means the key is both live
// (the authenticator will resolve it immediately) and durable.
func (k *keysDispatcher) handleAdd(args []byte) ([]byte, error) {
	if k.reg == nil {
		return nil, ErrKeyAdminUnavailable
	}
	a, err := ops.DecodeKeysAddArgs(args)
	if err != nil {
		return nil, err
	}
	if err := k.reg.AddKey(vector.APIKey{
		Token:  a.Token,
		Tenant: a.Tenant,
		Scopes: a.Scopes,
		CertCN: a.CertCN,
	}); err != nil {
		return nil, err
	}
	return nil, nil
}

// handleRevoke decodes the token and removes it from the live registry (and the
// keys file, atomically). After this returns the token no longer authenticates.
func (k *keysDispatcher) handleRevoke(args []byte) ([]byte, error) {
	if k.reg == nil {
		return nil, ErrKeyAdminUnavailable
	}
	token, err := ops.DecodeKeysRevokeArgs(args)
	if err != nil {
		return nil, err
	}
	if err := k.reg.RevokeKey(token); err != nil {
		return nil, err
	}
	return nil, nil
}

// handleList returns the REDACTED key snapshot. The raw token never leaves the
// registry: ListRedacted replaces it with TokenFingerprint, and the result codec
// has no token field, so the marshaled result cannot carry the secret.
func (k *keysDispatcher) handleList(_ []byte) ([]byte, error) {
	if k.reg == nil {
		return nil, ErrKeyAdminUnavailable
	}
	redacted := k.reg.ListRedacted()
	entries := make([]ops.RedactedKeyEntry, len(redacted))
	for i, rk := range redacted {
		entries[i] = ops.RedactedKeyEntry{
			Fingerprint: rk.Fingerprint,
			Tenant:      rk.Tenant,
			Scopes:      rk.Scopes,
			CertCN:      rk.CertCN,
		}
	}
	return ops.EncodeKeysListResult(entries), nil
}
