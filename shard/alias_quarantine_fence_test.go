// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"strings"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops"
)

// TestNewFailsClosedOnUnsafeAliasQuarantine proves Fix 2: on a REPLICATED shard the
// online-compaction alias-drain fence is derived from the effective server
// WriteTimeout (Cache.ServerWriteTimeout) as AliasQuarantine = 2*WriteTimeout, and an
// explicitly-set AliasQuarantine below that floor makes New() refuse to start rather
// than run an unsafe fence. This is the corruption invariant a comment alone cannot
// enforce: a fence shorter than the worst-case inline alias hold could recycle a
// retired mmap page while a response writer still aliases its bytes (torn read).
func TestNewFailsClosedOnUnsafeAliasQuarantine(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	// mk builds a cluster shard (external inmem Raft transport ⇒ replicatedCacheShard)
	// with the given effective WriteTimeout and explicit AliasQuarantine (0 = derive).
	mk := func(writeTimeout, aliasQuarantine time.Duration) Config {
		cfg := DefaultConfig(t.TempDir(), "rep", reg)
		cfg.Bootstrap = true
		cfg.RaftHeartbeatMs = 50
		cfg.RaftElectionMs = 100
		cfg.NoSync = true
		_, inmem := hraft.NewInmemTransport("")
		cfg.RaftTransport = inmem
		cfg.Cache.ServerWriteTimeout = writeTimeout
		cfg.Cache.AliasQuarantine = aliasQuarantine
		return cfg
	}

	// UNSAFE cases: an explicit AliasQuarantine below 2*WriteTimeout must fail closed.
	// The fence assertion runs before cache/raft construction, so these return cheaply
	// with no resources to clean up.
	unsafe := []struct {
		name   string
		wt, aq time.Duration
	}{
		{"half the floor", 30 * time.Second, 10 * time.Second},
		{"just below the floor", 60 * time.Second, 119 * time.Second},
	}
	for _, c := range unsafe {
		t.Run("unsafe/"+c.name, func(t *testing.T) {
			s, err := New(mk(c.wt, c.aq))
			if err == nil {
				_ = s.Close()
				t.Fatalf("New succeeded with AliasQuarantine %s < 2*WriteTimeout %s — fence not enforced", c.aq, 2*c.wt)
			}
			if !strings.Contains(err.Error(), "AliasQuarantine") {
				t.Fatalf("error %q does not name the AliasQuarantine fence invariant", err)
			}
		})
	}

	// SAFE cases: exactly at the floor, and the default (0 ⇒ derive 2*WriteTimeout),
	// both start. The default case also proves the derivation itself satisfies the
	// assertion (2*WT is never < 2*WT).
	safe := []struct {
		name   string
		wt, aq time.Duration
	}{
		{"exactly at the floor", 60 * time.Second, 120 * time.Second},
		{"derived default", 60 * time.Second, 0},
	}
	for _, c := range safe {
		t.Run("safe/"+c.name, func(t *testing.T) {
			s, err := New(mk(c.wt, c.aq))
			if err != nil {
				t.Fatalf("New failed for a safe fence (WriteTimeout=%s, AliasQuarantine=%s): %v", c.wt, c.aq, err)
			}
			t.Cleanup(func() { _ = s.Close() })
		})
	}
}
