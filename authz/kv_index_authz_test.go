// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// The KV record-index catalog ops (cluster/kv_index_admin.go). __kv_index_set__
// creates and DROPS indexes cluster-wide through the meta log; __kv_index_list__
// reads the catalog; __kv_index_ready__ is the internal per-group readiness leaf.
//
// THESE TESTS EXIST BECAUSE THE CLASSIFICATION IS OTHERWISE UNPINNED, the same
// defect the __wasm_blob_* and __register_wasm_shard__ tests were written for.
// None of the three is in the ops registry, so actionFor would fall through to
// the deny-by-default "admin" return and all three would BE admin with no
// adminOps entry at all — by coincidence, not by decision. Registering them so
// they dispatch through the registry (the obvious refactor) would silently demote
// them to "write" and hand index DROP to any write:* key. Every test below
// therefore registers the op OpReadWrite FIRST, so what is covered is the
// dangerous configuration rather than the fallthrough.

// kvIndexOps is the KV record-search admin surface: the catalog CRUD, the
// per-group readiness leaf the list handler gathers, and the shard-scoped leg
// of the kv_query fan-out.
//
// __kv_query_shard__ is here for a reason worth stating: `kv_query` ITSELF is an
// ordinary OpReadOnly and is authorised as a read, but the WRAPPER addresses ONE
// shard group directly, bypassing the coordinator that is the only thing making
// a page a complete answer rather than one group's slice of it. A read:* key
// that could call it would be able to read a partial answer and never know.
var kvIndexOps = []string{"__kv_index_set__", "__kv_index_list__", "__kv_index_ready__", "__kv_query_shard__"}

func newOpsRegWithKVIndexOps(t *testing.T, names ...string) *ops.Registry {
	t.Helper()
	r := ops.NewRegistry()
	if err := ops.RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	for _, name := range names {
		if err := r.Register(name, ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
			return nil, nil
		}); err != nil {
			t.Fatalf("register %s: %v", name, err)
		}
		if _, kind, _, ok := r.Lookup(name); !ok || kind != ops.OpReadWrite {
			t.Fatalf("precondition: %s must be registered OpReadWrite (ok=%v kind=%v)", name, ok, kind)
		}
	}
	return r
}

func TestActionForKVIndexOpsIsAdmin(t *testing.T) {
	for _, op := range kvIndexOps {
		if got := actionFor(op, nil); got != ActionAdmin {
			t.Errorf("actionFor(%s,nil)=%q want %q", op, got, ActionAdmin)
		}
		reg := newOpsRegWithKVIndexOps(t, op)
		if got := actionFor(op, reg); got != ActionAdmin {
			t.Errorf("actionFor(%s) after OpReadWrite registration = %q want %q", op, got, ActionAdmin)
		}
	}
}

func TestRBACKVIndexOpsRequireAdmin(t *testing.T) {
	opsReg := newOpsRegWithKVIndexOps(t, kvIndexOps...)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:*"}},
		vector.APIKey{Token: "writer", Tenant: "acme", Scopes: []string{"write:*"}},
		vector.APIKey{Token: "admin", Tenant: "acme", Scopes: []string{"admin:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL-SVC-TOKEN")

	for _, op := range kvIndexOps {
		for _, c := range []struct {
			token string
			want  bool
			desc  string
		}{
			{"reader", false, "read:* may not touch the index catalog"},
			{"writer", false, "cluster-wide write:* is NOT enough: dropping an index breaks every query naming it"},
			{"admin", true, "admin:* may"},
			{"super", true, "*:* may"},
		} {
			if got := auth(AuthRequest{Token: c.token, Op: op, Args: nil}); got != c.want {
				t.Errorf("authorize(%s, %s)=%v want %v — %s", c.token, op, got, c.want, c.desc)
			}
		}
		// The readiness leaf's only legitimate caller is a peer, which is granted
		// before the adminOps map is consulted, so admin costs the gather nothing.
		if !auth(AuthRequest{Token: "INTERNAL-SVC-TOKEN", Op: op}) {
			t.Errorf("the internal service principal must be able to call %s", op)
		}
	}
}
