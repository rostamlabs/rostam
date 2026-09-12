// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
)

// passThroughInner is a minimal innerDispatcher: it records the last non-keys op
// it saw and returns an empty ack, proving the keys decorator passes non-keys
// ops straight through.
type passThroughInner struct {
	lastOp string
}

func (p *passThroughInner) Call(name string, _ []byte) ([]byte, error) {
	p.lastOp = name
	return nil, nil
}
func (p *passThroughInner) LeaderAddr() string { return "" }

func newTestRegistry(t *testing.T) (*vector.KeyRegistry, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "keys.json")
	reg, err := vector.OpenKeyRegistry(path)
	if err != nil {
		t.Fatalf("OpenKeyRegistry: %v", err)
	}
	return reg, path
}

// TestKeysDispatcherLiveAddAuthenticatesThenRevokeDenies is the headline teeth
// test: an __keys_add__ makes the new token authenticate against the LIVE
// registry with no restart, and an __keys_revoke__ then makes that token fail
// auth — all through the SAME registry instance the authenticator reads.
func TestKeysDispatcherLiveAddAuthenticatesThenRevokeDenies(t *testing.T) {
	reg, _ := newTestRegistry(t)
	opsReg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(opsReg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	auth := authz.NewRBACAuthenticator(reg, opsReg, "")
	disp := newKeysDispatcher(&passThroughInner{}, reg)

	const tok = "LIVE-TOKEN"
	// Before add: the token does not authenticate.
	if auth(authz.AuthRequest{Token: tok, Op: "vector_search", Args: searchArgs("default/docs")}) {
		t.Fatal("token must not authenticate before it is added")
	}

	// Add via the dispatcher op (the same path a real admin request takes).
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{
		Token:  tok,
		Tenant: "acme",
		Scopes: []string{"read:default/docs"},
	})); err != nil {
		t.Fatalf("__keys_add__: %v", err)
	}

	// After add: the token authenticates immediately (live registry, no restart).
	if !auth(authz.AuthRequest{Token: tok, Op: "vector_search", Args: searchArgs("default/docs")}) {
		t.Fatal("token must authenticate immediately after __keys_add__")
	}

	// Revoke via the dispatcher op.
	if _, err := disp.Call(ops.OpKeysRevoke, ops.EncodeKeysRevokeArgs(tok)); err != nil {
		t.Fatalf("__keys_revoke__: %v", err)
	}

	// After revoke: the token fails auth.
	if auth(authz.AuthRequest{Token: tok, Op: "vector_search", Args: searchArgs("default/docs")}) {
		t.Fatal("token must fail auth after __keys_revoke__")
	}
}

// searchArgs builds the vector_search At2 wire layout [flags:u8][colLen:u8][col].
func searchArgs(col string) []byte {
	out := []byte{0x00, byte(len(col))}
	return append(out, col...)
}

// TestKeysDispatcherListRedactedNoRawToken is the redaction teeth test: the
// __keys_list__ result, fully marshaled, must NOT contain the raw token but MUST
// contain its fingerprint.
func TestKeysDispatcherListRedactedNoRawToken(t *testing.T) {
	reg, _ := newTestRegistry(t)
	disp := newKeysDispatcher(&passThroughInner{}, reg)

	const sentinel = "SENTINEL-RAW-TOKEN-XYZ"
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{
		Token:  sentinel,
		Tenant: "acme",
		Scopes: []string{"*:*"},
		CertCN: "svc.acme",
	})); err != nil {
		t.Fatalf("__keys_add__: %v", err)
	}

	frame, err := disp.Call(ops.OpKeysList, ops.EncodeKeysListArgs())
	if err != nil {
		t.Fatalf("__keys_list__: %v", err)
	}

	// Raw token absent from the marshaled result.
	if bytes.Contains(frame, []byte(sentinel)) {
		t.Fatal("SECURITY: __keys_list__ result must NEVER contain the raw token")
	}

	// Fingerprint present.
	wantFP := vector.TokenFingerprint(sentinel)
	if wantFP == "" {
		t.Fatal("fingerprint must be non-empty")
	}
	if !bytes.Contains(frame, []byte(wantFP)) {
		t.Fatal("__keys_list__ result must contain the token fingerprint")
	}

	// Decode and assert the descriptive fields survived (and there is no token).
	entries, err := ops.DecodeKeysListResult(frame)
	if err != nil {
		t.Fatalf("DecodeKeysListResult: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("want 1 entry, got %d", len(entries))
	}
	e := entries[0]
	if e.Fingerprint != wantFP || e.Tenant != "acme" || e.CertCN != "svc.acme" {
		t.Fatalf("unexpected redacted entry: %+v", e)
	}
	if len(e.Scopes) != 1 || e.Scopes[0] != "*:*" {
		t.Fatalf("unexpected scopes: %+v", e.Scopes)
	}
}

// TestKeysDispatcherValidationFailsLoud: empty token/tenant and a duplicate token
// fail loud (no silent success).
func TestKeysDispatcherValidationFailsLoud(t *testing.T) {
	reg, _ := newTestRegistry(t)
	disp := newKeysDispatcher(&passThroughInner{}, reg)

	// Empty token.
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{Tenant: "acme"})); err == nil {
		t.Error("add with empty token must fail")
	}
	// Empty tenant.
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{Token: "t"})); err == nil {
		t.Error("add with empty tenant must fail")
	}
	// First add ok.
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{Token: "dup", Tenant: "acme", Scopes: []string{"*:*"}})); err != nil {
		t.Fatalf("first add: %v", err)
	}
	// Duplicate token.
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{Token: "dup", Tenant: "acme", Scopes: []string{"*:*"}})); !errors.Is(err, vector.ErrAPIKeyExists) {
		t.Errorf("duplicate add must return ErrAPIKeyExists, got %v", err)
	}
	// Revoke unknown token.
	if _, err := disp.Call(ops.OpKeysRevoke, ops.EncodeKeysRevokeArgs("missing")); !errors.Is(err, vector.ErrAPIKeyNotFound) {
		t.Errorf("revoke unknown must return ErrAPIKeyNotFound, got %v", err)
	}
}

// TestKeysDispatcherNoRegistryFailsLoud: with a nil registry the keys ops fail
// loud (ErrKeyAdminUnavailable), while non-keys ops still pass through.
func TestKeysDispatcherNoRegistryFailsLoud(t *testing.T) {
	inner := &passThroughInner{}
	disp := newKeysDispatcher(inner, nil)

	for _, op := range []string{ops.OpKeysAdd, ops.OpKeysRevoke, ops.OpKeysList} {
		if _, err := disp.Call(op, nil); !errors.Is(err, ErrKeyAdminUnavailable) {
			t.Errorf("op %q with nil registry must return ErrKeyAdminUnavailable, got %v", op, err)
		}
	}

	// A non-keys op passes through unchanged.
	if _, err := disp.Call("vector_search", searchArgs("default/docs")); err != nil {
		t.Fatalf("pass-through op errored: %v", err)
	}
	if inner.lastOp != "vector_search" {
		t.Fatalf("non-keys op must reach inner dispatcher, got lastOp=%q", inner.lastOp)
	}
}

// TestKeysDispatcherDurability: a keys mutation persists to the keys file —
// reopening the registry from disk shows the change.
func TestKeysDispatcherDurability(t *testing.T) {
	reg, path := newTestRegistry(t)
	disp := newKeysDispatcher(&passThroughInner{}, reg)

	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{
		Token: "persisted", Tenant: "acme", Scopes: []string{"*:*"},
	})); err != nil {
		t.Fatalf("__keys_add__: %v", err)
	}

	// Reopen from disk: the added key is present.
	reopened, err := vector.OpenKeyRegistry(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if _, ok := reopened.Lookup("persisted"); !ok {
		t.Fatal("added key must persist across a registry reopen")
	}

	// Revoke + reopen: the key is gone.
	if _, err := disp.Call(ops.OpKeysRevoke, ops.EncodeKeysRevokeArgs("persisted")); err != nil {
		t.Fatalf("__keys_revoke__: %v", err)
	}
	reopened2, err := vector.OpenKeyRegistry(path)
	if err != nil {
		t.Fatalf("reopen after revoke: %v", err)
	}
	if _, ok := reopened2.Lookup("persisted"); ok {
		t.Fatal("revoked key must be absent after a registry reopen")
	}
}

// TestKeysDispatcherConcurrentAuthReadsAndAddWrite exercises the registry RWMutex
// under -race: concurrent auth Lookup reads while __keys_add__ writes. Run with
// `dgo test -race -run TestKeysDispatcherConcurrent`.
func TestKeysDispatcherConcurrentAuthReadsAndAddWrite(t *testing.T) {
	reg, _ := newTestRegistry(t)
	opsReg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(opsReg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	auth := authz.NewRBACAuthenticator(reg, opsReg, "")
	disp := newKeysDispatcher(&passThroughInner{}, reg)

	// Seed a key the readers look up.
	if _, err := disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{
		Token: "seed", Tenant: "acme", Scopes: []string{"read:default/docs"},
	})); err != nil {
		t.Fatalf("seed add: %v", err)
	}

	var wg sync.WaitGroup
	const readers = 8
	stop := make(chan struct{})
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					auth(authz.AuthRequest{Token: "seed", Op: "vector_search", Args: searchArgs("default/docs")})
					_, _ = disp.Call(ops.OpKeysList, ops.EncodeKeysListArgs())
				}
			}
		}()
	}

	// Concurrent writes.
	for i := 0; i < 64; i++ {
		tok := "w" + string(rune('A'+i%26)) + string(rune('0'+i/26))
		_, _ = disp.Call(ops.OpKeysAdd, ops.EncodeKeysAddArgs(ops.KeysAddArgs{
			Token: tok, Tenant: "acme", Scopes: []string{"read:*"},
		}))
	}
	close(stop)
	wg.Wait()
}

// appendingInner is a passThroughInner that also offers the optional
// allocation-free path, so the wrapper's forwarding can be observed.
type appendingInner struct {
	passThroughInner
	appendCalled bool
}

func (a *appendingInner) CallAppend(name string, _, dst []byte) ([]byte, error) {
	a.appendCalled = true
	a.lastOp = name
	return append(dst, "INNER"...), nil
}

// The epoll transport takes the allocation-free path only when the dispatcher it
// is handed implements server.AppendDispatcher — and NewServer ALWAYS wraps the
// store in the keys decorator before handing it over. The decorator therefore has
// to PRESERVE the inner dispatcher's capability in both directions, and both
// directions have bitten:
//
//   - dropping it killed the append path in production (single-node) while every
//     handler-level benchmark still looked green;
//   - faking it made the cluster path allocate a reply buffer AND copy the whole
//     reply into it, which is worse than the plain Call path it replaced.
func TestWrapKeysDispatcherPreservesAppendCapability(t *testing.T) {
	t.Run("inner can append", func(t *testing.T) {
		inner := &appendingInner{}
		w := wrapKeysDispatcher(inner, nil)

		ad, ok := w.(server.AppendDispatcher)
		if !ok {
			t.Fatal("wrapper dropped the inner dispatcher's append path; every get and operate would allocate its reply")
		}
		got, err := ad.CallAppend("get", []byte("args"), []byte("DST"))
		if err != nil {
			t.Fatal(err)
		}
		if !inner.appendCalled {
			t.Error("wrapper served the op itself instead of forwarding CallAppend")
		}
		if string(got) != "DSTINNER" {
			t.Errorf("payload = %q, want the inner reply appended to dst", got)
		}
	})

	t.Run("inner cannot append", func(t *testing.T) {
		w := wrapKeysDispatcher(&passThroughInner{}, nil)
		if _, ok := w.(server.AppendDispatcher); ok {
			t.Error("wrapper advertises an append path its inner cannot serve; the transport would allocate a buffer and copy every reply into it for nothing")
		}
	})
}

// The intercepted keys ops are administrative: served by the wrapper itself,
// never forwarded to the inner dispatcher. A LIVE registry is required to see
// that — with a nil one every keys op fails with ErrKeyAdminUnavailable before
// producing a frame, so the assertion would hold vacuously.
func TestKeysAppendDispatcherKeepsKeysOpsLocal(t *testing.T) {
	reg, _ := newTestRegistry(t)
	inner := &appendingInner{}
	ad, ok := wrapKeysDispatcher(inner, reg).(server.AppendDispatcher)
	if !ok {
		t.Fatal("expected an append-capable wrapper")
	}

	got, err := ad.CallAppend(ops.OpKeysList, nil, []byte("DST"))
	if err != nil {
		t.Fatalf("keys list: %v", err)
	}
	if inner.appendCalled {
		t.Error("a keys op was forwarded to the inner dispatcher instead of being served locally")
	}
	if len(got) == 0 {
		t.Error("keys list produced no frame")
	}
	// It is served locally, so the frame is its own rather than an append onto
	// dst — the payload is explicitly allowed not to alias.
	if bytes.HasPrefix(got, []byte("DST")) {
		t.Error("the keys frame was copied into dst; it should be returned as is")
	}
}
