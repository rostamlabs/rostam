// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// Errors surfaced to callers.
var (
	// ErrNotFound is returned when the server responded with StatusNotFound.
	ErrNotFound = errors.New("client: not found")
	// ErrNoLeaderKnown indicates retries were exhausted without a successful
	// leader response.
	ErrNoLeaderKnown = errors.New("client: no leader known after retries")
	// ErrClientClosed indicates Call was made after Close.
	ErrClientClosed = errors.New("client: closed")
	// ErrUnauthorized indicates the server rejected the request because the
	// supplied auth token was missing or invalid.
	ErrUnauthorized = errors.New("client: unauthorized")
)

// RemoteError represents a StatusError response from the server.
type RemoteError struct {
	Op  string
	Msg string
}

func (e *RemoteError) Error() string {
	return fmt.Sprintf("client: server error on op %q: %s", e.Op, e.Msg)
}

// Client manages one pool per server address. Safe for concurrent use.
type Client struct {
	cfg     *Config
	pools   map[string]*perServerPool
	poolsMu sync.RWMutex

	// pipeSets holds the opt-in pipelined connection sets per server
	// (Config.PipelineDepth > 0). nil/empty when pipelining is off.
	pipeSets  map[string]*pipeSet
	pipeMu    sync.Mutex
	pipeDial  *net.Dialer
	closeOnce sync.Once
	closed    chan struct{}
	wg        sync.WaitGroup

	topology topologyCache

	rr atomic.Uint64 // round-robin counter for nextServer
}

// errPipeDead marks a pipelined connection that has failed (I/O error / closed);
// the owning set redials on the next pick.
var errPipeDead = errors.New("client: pipelined connection dead")

// pipelining reports whether the opt-in pipelined Call path is enabled.
func (c *Client) pipelining() bool { return c.cfg.PipelineDepth > 0 }

// getPipeSet returns (lazily creating) the pipelined connection set for addr.
func (c *Client) getPipeSet(addr string) *pipeSet {
	c.pipeMu.Lock()
	defer c.pipeMu.Unlock()
	if s, ok := c.pipeSets[addr]; ok {
		return s
	}
	if c.pipeSets == nil {
		c.pipeSets = make(map[string]*pipeSet)
	}
	s := &pipeSet{
		addr:      addr,
		authToken: c.cfg.AuthToken,
		depth:     int(c.cfg.PipelineDepth),
		callT:     c.cfg.CallTimeout,
		dialer:    c.pipeDial,
		tlsCfg:    c.cfg.TLSConfig,
		conns:     make([]*pipeConn, c.cfg.PipelineConns),
	}
	c.pipeSets[addr] = s
	return s
}

// callAddrPipelined is the pipelined counterpart of callAddr: it routes op+args
// over a pipelined connection (many in flight per conn) and maps the response.
func (c *Client) callAddrPipelined(ctx context.Context, op string, args []byte, addr string) ([]byte, error) {
	pc, err := c.getPipeSet(addr).pick(ctx)
	if err != nil {
		return nil, err
	}
	status, payload, callErr := pc.call(ctx, op, args)
	if callErr != nil {
		return nil, callErr
	}
	return mapCallStatus(op, status, payload)
}

// mapCallStatus turns a response (status, payload) into the Call result or a
// typed error — shared by the pooled and pipelined paths. The StatusOK payload
// is copied (it may alias a connection buffer the caller does not own).
func mapCallStatus(op string, status uint8, payload []byte) ([]byte, error) {
	switch status {
	case StatusOK:
		return append([]byte(nil), payload...), nil
	case StatusNotFound:
		return nil, ErrNotFound
	case StatusNotLeader:
		hint, _ := decodeLeaderAddr(payload)
		return nil, &errNotLeader{leaderAddr: hint}
	case StatusError:
		msg, _ := decodeErrorMsg(payload)
		return nil, &RemoteError{Op: op, Msg: msg}
	case StatusUnauthorized:
		return nil, ErrUnauthorized
	default:
		return nil, fmt.Errorf("client: unknown response status %d", status)
	}
}

// New constructs a Client. Pools are created lazily on first reference,
// except those for servers that need warmup (MinConnsPerServer > 0), which
// are created eagerly.
func New(cfg Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	cfg.applyDefaults()
	c := &Client{
		cfg:      &cfg,
		pools:    make(map[string]*perServerPool, len(cfg.Servers)),
		pipeDial: &net.Dialer{Timeout: cfg.DialTimeout},
		closed:   make(chan struct{}),
	}
	if cfg.MinConnsPerServer > 0 {
		for _, addr := range cfg.Servers {
			if _, err := c.getOrCreatePool(addr); err != nil {
				_ = c.Close()
				return nil, err
			}
		}
	}
	if cfg.Ops != nil {
		// Initial best-effort topology bootstrap (synchronous).
		c.refreshTopology(context.Background())
		// Background refresh loop.
		c.wg.Add(1)
		go c.refreshLoop()
	}
	return c, nil
}

// Call sends one request, following NotLeader hints up to cfg.MaxNotLeaderHops.
// getOrCreatePool handles hinted addrs that are not in cfg.Servers.
// Transport errors (dial refused, EOF, etc.) are retried against another
// server within the same hop budget.
func (c *Client) Call(ctx context.Context, op string, args []byte) ([]byte, error) {
	select {
	case <-c.closed:
		return nil, ErrClientClosed
	default:
	}

	target := c.pickInitialTarget(op, args)
	maxHops := c.cfg.MaxNotLeaderHops

	for hop := 0; hop <= maxHops; hop++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		result, err := c.callAddr(ctx, op, args, target)
		if err == nil {
			return result, nil
		}
		var nle *errNotLeader
		if errors.As(err, &nle) {
			// NotLeader: refresh topology, then pick a new target.
			c.refreshTopology(ctx)
			switch {
			case nle.leaderAddr != "":
				target = nle.leaderAddr
			default:
				next := c.pickInitialTarget(op, args)
				if next == target {
					return nil, ErrNoLeaderKnown
				}
				target = next
			}
			continue
		}
		// Non-NotLeader error: if it's a transport error, rotate to
		// another server and retry within the hop budget.
		if isTransportError(err) && hop < maxHops {
			next := c.nextServer()
			if next != "" && next != target {
				target = next
				continue
			}
		}
		return nil, err
	}
	return nil, fmt.Errorf("client: NotLeader exceeded %d hops", maxHops)
}

// SetNX sets key to value only if the key is currently absent or expired — the
// atomic set-if-absent primitive. Returns true if the value was stored, false if
// the key was already present. ttl > 0 sets an expiry on the stored value.
func (c *Client) SetNX(ctx context.Context, key, value []byte, ttl time.Duration) (bool, error) {
	res, err := c.Call(ctx, "set_nx", wire.EncodeSetNXArgs(key, value, ttl))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// CAS is compare-and-swap: it sets key to value only if the key's current value
// equals expected. Returns true if the value was stored, false on a mismatch (or
// if the key is absent). ttl > 0 sets an expiry on the stored value. To store only
// when the key is absent, use SetNX.
func (c *Client) CAS(ctx context.Context, key, value, expected []byte, ttl time.Duration) (bool, error) {
	res, err := c.Call(ctx, "cas", wire.EncodeCASArgs(key, value, true, expected, ttl))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// CompareAndDelete deletes key only if its current value equals expected — the
// safe-unlock primitive (release a lock only if you still hold it). Returns true
// if the key was deleted, false on a value mismatch or an absent key.
func (c *Client) CompareAndDelete(ctx context.Context, key, expected []byte) (bool, error) {
	res, err := c.Call(ctx, "cad", wire.EncodeCADArgs(key, expected))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// PutBatch writes many key/value pairs as bulk put_batch ops — the ~10x
// bulk-insert fast path. It groups entries by their owning shard (the same hash
// as pickInitialTarget) so each shard receives ONE Raft log entry per chunk, and
// chunks each group to wire.MaxPutBatchSize. Each Call follows NotLeader hints like
// a single put. When the topology is unknown (no cache / NumShards==0) it cannot
// group, so it falls back to one put per entry (the correctness floor). If a stale
// topology (mid-reshard) makes the server reject a group as cross-shard, it
// refreshes the topology once and retries that group re-grouped.
func (c *Client) PutBatch(ctx context.Context, entries []wire.PutEntry) error {
	if len(entries) == 0 {
		return nil
	}
	t := c.topology.get()
	if t == nil || t.NumShards == 0 {
		return c.putBatchFallback(ctx, entries)
	}
	for _, g := range groupEntriesByShard(entries, t.NumShards) {
		if err := c.sendSameShardBatch(ctx, g); err != nil {
			if !strings.Contains(err.Error(), "span multiple shards") {
				return err
			}
			// Stale topology misgrouped this batch; refresh once and retry the
			// group re-grouped against the fresh topology (one retry only).
			c.refreshTopology(ctx)
			nt := c.topology.get()
			if nt == nil || nt.NumShards == 0 {
				if ferr := c.putBatchFallback(ctx, g); ferr != nil {
					return ferr
				}
				continue
			}
			for _, rg := range groupEntriesByShard(g, nt.NumShards) {
				if rerr := c.sendSameShardBatch(ctx, rg); rerr != nil {
					return rerr
				}
			}
		}
	}
	return nil
}

// putBatchFallback issues one put per entry — the correctness floor used when the
// topology is unknown, so entries cannot be grouped by shard for a put_batch.
func (c *Client) putBatchFallback(ctx context.Context, entries []wire.PutEntry) error {
	for _, e := range entries {
		if _, err := c.Call(ctx, "put", wire.EncodePutArgs(e.Key, e.Val, e.TTL)); err != nil {
			return err
		}
	}
	return nil
}

// sendSameShardBatch chunks a single-shard entry group to wire.MaxPutBatchSize and
// dispatches each chunk as a put_batch (Call follows NotLeader hints).
func (c *Client) sendSameShardBatch(ctx context.Context, g []wire.PutEntry) error {
	for len(g) > 0 {
		chunk := g
		if len(chunk) > wire.MaxPutBatchSize {
			chunk = g[:wire.MaxPutBatchSize]
		}
		if _, err := c.Call(ctx, "put_batch", wire.EncodePutBatchArgs(chunk)); err != nil {
			return err
		}
		g = g[len(chunk):]
	}
	return nil
}

// groupEntriesByShard buckets entries by shard index using the same hash formula
// as pickInitialTarget. Each bucket is dispatched independently, so map iteration
// order does not matter.
func groupEntriesByShard(entries []wire.PutEntry, numShards int) map[int][]wire.PutEntry {
	groups := make(map[int][]wire.PutEntry)
	for _, e := range entries {
		shard := int(xxhash.Sum64(e.Key) % uint64(numShards)) //nolint:gosec // numShards bounded by Config validation
		groups[shard] = append(groups[shard], e)
	}
	return groups
}

// isTransportError reports whether err is a network-level transport
// error (dial refused, connection reset, EOF, etc.) that is safe to
// retry against a different server. It deliberately does NOT match
// application-level errors (StatusError, StatusNotFound, etc.).
func isTransportError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed) {
		return true
	}
	var netErr *net.OpError
	return errors.As(err, &netErr)
}

// nextServer returns the next server in a round-robin sequence over
// c.cfg.Servers. Used as the fallback target when smart routing isn't
// available (topology not cached, op is shardless, or the target shard
// has no known leader).
func (c *Client) nextServer() string {
	if len(c.cfg.Servers) == 0 {
		return ""
	}
	i := c.rr.Add(1) - 1
	return c.cfg.Servers[int(i%uint64(len(c.cfg.Servers)))] //nolint:gosec // result bounded by len(Servers), fits in int
}

// pickInitialTarget picks the first server to dial for op+args (client-side
// sharding). When Ops is configured, a topology is cached, and the op has a
// KeyExtractor that yields a key, it routes to the key's shard:
//
//  1. the shard's leader, if known (best — no NotLeader hop); else
//  2. an owner of the shard from Placement (the fast path under partitioned
//     placement, where a node only knows leaders for shards it hosts). An owner
//     that is not the leader returns a NotLeader hint the caller then follows.
//
// Otherwise it falls back to round-robin over configured servers (which the
// server forwards to an owner — the correctness floor for dumb routing).
func (c *Client) pickInitialTarget(op string, args []byte) string {
	if c.cfg.Ops != nil {
		if _, ke, layout, ok := c.cfg.Ops.LookupRouting(op); ok {
			// maxRouteKeyLen bounds "default/" (8) + a u8-length collection name
			// (255). Keeping scratch on the stack lets RouteKeyInto extract the
			// routing key with no heap allocation on the hot path; a DIRECT call
			// to RouteKeyInto (not through the ke func value) is what lets escape
			// analysis keep scratch stack-bound — see ops/wire.RouteKeyInto.
			const maxRouteKeyLen = 8 + 255
			var scratch [maxRouteKeyLen]byte
			var key []byte
			var hasKey bool
			switch {
			case layout != wire.RouteLayoutNone:
				key = wire.RouteKeyInto(layout, args, scratch[:0])
				hasKey = key != nil
			case ke != nil:
				// RouteLayoutNone ops route through ke: KV builtins (the
				// subslicing stdKeyExtractor, already alloc-free) and
				// dynamic/WASM ops (no allocation-free layout).
				key, hasKey = ke(args)
			}
			if hasKey {
				if t := c.topology.get(); t != nil && t.NumShards > 0 {
					shardID := int(xxhash.Sum64(key) % uint64(t.NumShards)) //nolint:gosec // NumShards bounded by Config validation
					if len(t.Leaders) == t.NumShards && t.Leaders[shardID] != "" {
						return t.Leaders[shardID]
					}
					if addr := t.OwnerAddr(shardID); addr != "" {
						return addr
					}
				}
			}
		}
	}
	return c.nextServer()
}

// refreshTopology calls __topology__ against each configured server in
// turn until one succeeds. Best-effort: on full failure, the existing
// cache is left in place. No-op when Ops is not configured.
//
// Each server attempt is capped at 500 ms so a dead Servers[0] does not
// stall the entire refresh budget.
func (c *Client) refreshTopology(ctx context.Context) {
	if c.cfg.Ops == nil {
		return
	}
	// Total budget = 500ms × number of servers; bail early if parent cancelled.
	totalTimeout := 500 * time.Millisecond * time.Duration(len(c.cfg.Servers))
	refreshCtx, cancel := context.WithTimeout(ctx, totalTimeout)
	defer cancel()

	for _, addr := range c.cfg.Servers {
		if refreshCtx.Err() != nil {
			return
		}
		attemptCtx, attemptCancel := context.WithTimeout(refreshCtx, 500*time.Millisecond)
		result, err := c.callAddr(attemptCtx, "__topology__", nil, addr)
		attemptCancel()
		if err != nil {
			continue
		}
		t, derr := wire.DecodeTopology(result)
		if derr != nil {
			continue
		}
		c.topology.set(t)
		return
	}
}

// refreshLoop periodically refreshes the topology cache. Exits when
// closed is closed. Only started when Ops != nil.
func (c *Client) refreshLoop() {
	defer c.wg.Done()
	t := time.NewTicker(c.cfg.TopologyRefreshInterval)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			c.refreshTopology(context.Background())
		case <-c.closed:
			return
		}
	}
}

// errNotLeader is an internal sentinel returned by callAddr when the server
// responds with StatusNotLeader. It carries the hinted leader address.
type errNotLeader struct {
	leaderAddr string
}

func (e *errNotLeader) Error() string {
	return fmt.Sprintf("client: not leader (hint: %q)", e.leaderAddr)
}

// CallFunc is like Call but invokes fn with the response payload while
// the connection is still held — payload aliases the connection's read
// buffer and is valid ONLY for the duration of fn. Use this to skip the
// per-Call defensive copy when the response is consumed in place
// (e.g., parsed, written to an io.Writer, or copied into a caller-owned
// buffer). fn is not invoked on non-OK statuses; the appropriate error
// is returned instead. fn may be nil to discard the payload.
func (c *Client) CallFunc(ctx context.Context, op string, args []byte, fn func(payload []byte) error) error {
	select {
	case <-c.closed:
		return ErrClientClosed
	default:
	}

	target := c.pickInitialTarget(op, args)
	maxHops := c.cfg.MaxNotLeaderHops

	for hop := 0; hop <= maxHops; hop++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		err := c.callAddrFunc(ctx, op, args, target, fn)
		if err == nil {
			return nil
		}
		var nle *errNotLeader
		if errors.As(err, &nle) {
			c.refreshTopology(ctx)
			switch {
			case nle.leaderAddr != "":
				target = nle.leaderAddr
			default:
				next := c.pickInitialTarget(op, args)
				if next == target {
					return ErrNoLeaderKnown
				}
				target = next
			}
			continue
		}
		if isTransportError(err) && hop < maxHops {
			next := c.nextServer()
			if next != "" && next != target {
				target = next
				continue
			}
		}
		return err
	}
	return fmt.Errorf("client: NotLeader exceeded %d hops", maxHops)
}

// callAddr acquires a pooled connection for addr, sends the request, and
// returns the decoded result. getOrCreatePool dynamically adds hinted addrs
// that are not in cfg.Servers. Returns *errNotLeader on StatusNotLeader.
//
// Kept separate from callAddrFunc to avoid a capturing-closure alloc on
// the Call hot path — the closure variant pays one extra heap alloc per
// call for the captured `result` slice.
func (c *Client) callAddr(ctx context.Context, op string, args []byte, addr string) ([]byte, error) {
	if c.pipelining() {
		return c.callAddrPipelined(ctx, op, args, addr)
	}
	pool, err := c.getOrCreatePool(addr)
	if err != nil {
		return nil, err
	}
	res, err := pool.acquire(ctx)
	if err != nil {
		return nil, err
	}
	conn := res.Value()
	status, payload, callErr := conn.doCall(ctx, op, args, c.cfg.CallTimeout)
	if callErr != nil {
		conn.poisoned = true
		pool.release(res)
		return nil, callErr
	}
	switch status {
	case StatusOK:
		result := make([]byte, len(payload))
		copy(result, payload)
		pool.release(res)
		return result, nil
	case StatusNotFound:
		pool.release(res)
		return nil, ErrNotFound
	case StatusNotLeader:
		hint, _ := decodeLeaderAddr(payload)
		pool.release(res)
		return nil, &errNotLeader{leaderAddr: hint}
	case StatusError:
		msg, _ := decodeErrorMsg(payload)
		pool.release(res)
		return nil, &RemoteError{Op: op, Msg: msg}
	case StatusUnauthorized:
		pool.release(res)
		return nil, ErrUnauthorized
	default:
		pool.release(res)
		return nil, fmt.Errorf("client: unknown response status %d", status)
	}
}

// callAddrFunc is the CallFunc core. It acquires a pooled conn, sends the
// request, and on StatusOK invokes fn while still holding the conn so fn
// can read payload zero-copy.
func (c *Client) callAddrFunc(ctx context.Context, op string, args []byte, addr string, fn func([]byte) error) error {
	pool, err := c.getOrCreatePool(addr)
	if err != nil {
		return err
	}
	res, err := pool.acquire(ctx)
	if err != nil {
		return err
	}
	conn := res.Value()
	status, payload, callErr := conn.doCall(ctx, op, args, c.cfg.CallTimeout)
	if callErr != nil {
		conn.poisoned = true
		pool.release(res)
		return callErr
	}
	switch status {
	case StatusOK:
		var fnErr error
		if fn != nil {
			fnErr = fn(payload)
		}
		pool.release(res)
		return fnErr
	case StatusNotFound:
		pool.release(res)
		return ErrNotFound
	case StatusNotLeader:
		hint, _ := decodeLeaderAddr(payload)
		pool.release(res)
		return &errNotLeader{leaderAddr: hint}
	case StatusError:
		msg, _ := decodeErrorMsg(payload)
		pool.release(res)
		return &RemoteError{Op: op, Msg: msg}
	case StatusUnauthorized:
		pool.release(res)
		return ErrUnauthorized
	default:
		pool.release(res)
		return fmt.Errorf("client: unknown response status %d", status)
	}
}

// Topology returns the most recent cluster topology snapshot, or nil
// if the refresh loop has not yet populated it. Callers must not mutate
// the returned value — it is shared with concurrent readers.
func (c *Client) Topology() *wire.Topology {
	return c.topology.get()
}

// Stat returns a snapshot per server address.
func (c *Client) Stat() map[string]Stat {
	c.poolsMu.RLock()
	defer c.poolsMu.RUnlock()
	out := make(map[string]Stat, len(c.pools))
	for addr, p := range c.pools {
		out[addr] = p.stat()
	}
	return out
}

// Reset closes all connections in all pools (puddle keeps the pool object).
// Use after a known network blip or server restart.
func (c *Client) Reset() {
	c.poolsMu.RLock()
	defer c.poolsMu.RUnlock()
	for _, p := range c.pools {
		p.reset()
	}
}

// Close stops the background refresh goroutine (if running) and all pools.
// Idempotent.
func (c *Client) Close() error {
	c.closeOnce.Do(func() {
		close(c.closed)
		c.wg.Wait()
		c.pipeMu.Lock()
		for _, s := range c.pipeSets {
			s.closeAll()
		}
		c.pipeMu.Unlock()
		c.poolsMu.Lock()
		defer c.poolsMu.Unlock()
		for _, p := range c.pools {
			p.close()
		}
	})
	return nil
}

func (c *Client) getOrCreatePool(addr string) (*perServerPool, error) {
	c.poolsMu.RLock()
	p, ok := c.pools[addr]
	c.poolsMu.RUnlock()
	if ok {
		return p, nil
	}
	c.poolsMu.Lock()
	defer c.poolsMu.Unlock()
	if p, ok := c.pools[addr]; ok {
		return p, nil
	}
	np, err := newPerServerPool(addr, c.cfg)
	if err != nil {
		return nil, err
	}
	c.pools[addr] = np
	return np, nil
}
