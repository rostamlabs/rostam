// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// Config governs a Client and its per-server pools.
type Config struct {
	// Servers is the list of "host:port" addresses to use for Calls.
	// On NotLeader, the client may temporarily prefer a hinted server.
	Servers []string

	// MaxConnsPerServer caps the puddle pool per server. Default = max(4, NumCPU).
	MaxConnsPerServer int32

	// MinConnsPerServer is the warm-pool floor per server. Default 0 (lazy).
	MinConnsPerServer int32

	// MaxConnLifetime closes connections older than this. Default 1 hour.
	MaxConnLifetime time.Duration

	// MaxConnLifetimeJitter adds up to this much randomness to MaxConnLifetime
	// per connection to avoid simultaneous reconnect storms. Default 0.
	MaxConnLifetimeJitter time.Duration

	// MaxConnIdleTime closes idle connections after this. Default 30 minutes.
	MaxConnIdleTime time.Duration

	// HealthCheckPeriod is how often the background loop scans for expired
	// or idle conns. Default 1 minute.
	HealthCheckPeriod time.Duration

	// DialTimeout caps how long a single TCP dial can take. Default 5s.
	DialTimeout time.Duration

	// CallTimeout caps how long an individual Call can take (when ctx has
	// no deadline). Default 5s.
	CallTimeout time.Duration

	// MaxNotLeaderHops bounds how many NotLeader hints the client follows
	// after the initial attempt. Total attempts = MaxNotLeaderHops + 1.
	// Default 3 (one initial + three redirects).
	MaxNotLeaderHops int

	// PipelineDepth, when > 0, enables PIPELINED Calls: up to this many requests
	// in flight per connection instead of one-at-a-time. It removes the
	// "throughput = connections / per-op latency" ceiling — the big win on a
	// latency-bound (e.g. replicated) write workload where a request-response
	// client would otherwise be connection-limited. 0 (default) keeps the strict
	// request-response pool, byte-identical to before. SEMANTICS: pipelined
	// requests are concurrently in flight, so two writes to the SAME key on one
	// connection may apply in either order — do not pipeline order-dependent
	// writes (see client/pipeline.go). NotLeader auto-retry still applies. A
	// reasonable value is 32–64. Both Call and CallFunc take the pipelined path.
	PipelineDepth int32

	// PipelineConns is the number of pipelined connections per server (round-
	// robin) when PipelineDepth > 0. Default 4. More conns spread concurrent
	// Calls and cap head-of-line blocking behind a slow response.
	//
	// It cuts the other way too, and the trade is easy to get backwards. Writes
	// from callers that are in flight AT THE SAME INSTANT on the SAME connection
	// leave in one syscall; spreading the same rate over more connections drops
	// the concurrency on each, and below roughly one concurrent caller per
	// connection there is nothing left to combine. Pipelining is what lets a
	// client serve its load from FEWER connections -- keeping the pooled count
	// while enabling it gives up that saving for no benefit.
	PipelineConns int32

	// Ops is the routing-only op registry for smart routing. If nil, the
	// client falls back to round-robin + NotLeader retry. It is a
	// *wire.Registry (routing metadata only, no handlers); the server uses a
	// separate *ops.Registry (routing + handlers). They are DIFFERENT types
	// and are never the same object — the client cannot import the server's
	// cache-bound Handler/TxContext. What must match is the routing metadata:
	// populate the client with wire.RegisterRoutableBuiltins (or use
	// NewRouted, which does it) while the server uses ops.RegisterBuiltins —
	// both draw from the shared wire.BuiltinOps table, so the mappings agree.
	// To route from an already-built server registry, adapt it with
	// (*ops.Registry).ExportRouting.
	Ops *wire.Registry

	// TopologyRefreshInterval is how often the background goroutine
	// polls __topology__. Default 5s. Ignored when Ops == nil.
	TopologyRefreshInterval time.Duration

	// AuthToken, when non-empty, is sent on every RPC via a protocol-v2
	// frame prefix. The server must validate it through its Authenticator
	// hook; otherwise the token is harmlessly ignored (legacy server).
	// Empty AuthToken keeps the v1 frame layout (no per-request overhead).
	//
	// WARNING: if TLSConfig is nil the token is transmitted in cleartext over
	// a plaintext TCP connection. Set TLSConfig to encrypt the connection and
	// protect the credential in transit.
	AuthToken string

	// TLSConfig, when non-nil, makes the client dial each server over TLS
	// (tls.Client over the raw TCP conn) instead of plaintext. Build it via
	// tlsutil.ClientTLS(caFile, certFile, keyFile, serverName): set RootCAs to
	// verify the server's cert, and (for mTLS) a client cert/key the server
	// verifies — the cert's CN then becomes the client's principal on a server
	// with no bearer token. nil ⇒ plaintext (the default; unchanged hot path).
	//
	// NOTE: the inter-node cluster peerClient sets this (a per-peer clone with
	// ServerName) ONLY when the cluster is configured with InterNodeTLS — i.e. when
	// client-facing TLS is on, so the inter-node dial matches the TLS-wrapped peer
	// port. A plaintext cluster leaves it nil (plaintext inter-node, unchanged).
	TLSConfig *tls.Config

	// Optional hooks (mirror chpool). Any may be nil.
	BeforeConnect func(ctx context.Context, addr string) error
	AfterConnect  func(ctx context.Context, c *Conn) error
	BeforeAcquire func(ctx context.Context, c *Conn) bool
	AfterRelease  func(c *Conn) bool
	BeforeClose   func(c *Conn)
}

func (c *Config) applyDefaults() {
	if c.MaxConnsPerServer == 0 {
		c.MaxConnsPerServer = 4
		if cpu := int32(runtime.NumCPU()); cpu > c.MaxConnsPerServer { //nolint:gosec // NumCPU fits int32
			c.MaxConnsPerServer = cpu
		}
	}
	if c.MaxConnLifetime == 0 {
		c.MaxConnLifetime = time.Hour
	}
	if c.MaxConnIdleTime == 0 {
		c.MaxConnIdleTime = 30 * time.Minute
	}
	if c.HealthCheckPeriod == 0 {
		c.HealthCheckPeriod = time.Minute
	}
	if c.DialTimeout == 0 {
		c.DialTimeout = 5 * time.Second
	}
	if c.CallTimeout == 0 {
		c.CallTimeout = 5 * time.Second
	}
	if c.MaxNotLeaderHops == 0 {
		c.MaxNotLeaderHops = 3
	}
	if c.TopologyRefreshInterval == 0 {
		c.TopologyRefreshInterval = 5 * time.Second
	}
	if c.PipelineDepth > 0 && c.PipelineConns == 0 {
		c.PipelineConns = 4
	}
}

// Validate enforces invariants. Returns error on misconfiguration.
func (c Config) Validate() error {
	if len(c.Servers) == 0 {
		return errors.New("client.Config: at least one server required")
	}
	for i, s := range c.Servers {
		if s == "" {
			return fmt.Errorf("client.Config: Servers[%d] empty", i)
		}
	}
	if c.MaxConnsPerServer < 0 {
		return errors.New("client.Config: MaxConnsPerServer must be >= 0")
	}
	if c.MinConnsPerServer < 0 {
		return errors.New("client.Config: MinConnsPerServer must be >= 0")
	}
	if c.MaxConnsPerServer > 0 && c.MinConnsPerServer > c.MaxConnsPerServer {
		return fmt.Errorf("client.Config: MinConnsPerServer=%d > MaxConnsPerServer=%d",
			c.MinConnsPerServer, c.MaxConnsPerServer)
	}
	if c.MaxNotLeaderHops < 0 {
		return errors.New("client.Config: MaxNotLeaderHops must be >= 0")
	}
	if c.Ops != nil && c.TopologyRefreshInterval < 1*time.Second {
		return errors.New("client: TopologyRefreshInterval must be >= 1s when Ops is set")
	}
	if c.AuthToken != "" && c.TLSConfig == nil {
		// Not a hard error: plaintext is a valid deployment choice (e.g. loopback,
		// trusted private network, mTLS handled by a sidecar). The caller is
		// responsible for ensuring the connection is adequately protected.
		// To suppress this at build time, set AllowInsecureAuth = true (reserved
		// for a future field; for now callers may ignore the log line if intentional).
		fmt.Fprintln(os.Stderr, "client WARNING: AuthToken is set but TLSConfig is nil — "+
			"the bearer token will be transmitted in cleartext")
	}
	return nil
}
