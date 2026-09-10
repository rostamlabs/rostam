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
		// Post-transmission: the pipelined request may have committed before
		// its response was lost. Mark it ambiguous so Call won't replay a
		// conditional write. The pick() error above is pre-transmission and
		// stays unwrapped.
		return nil, &ambiguousError{err: callErr}
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
// server within the same hop budget — EXCEPT that an outcome-returning
// conditional write (set_nx/cas/cad) is NOT replayed across an AMBIGUOUS
// failure (one that happened after the request was transmitted): the first
// server may have committed before the response was lost, so a replay would
// report a wrong result. Such a failure is returned to the caller instead.
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
		// another server and retry within the hop budget — but never
		// replay a non-replayable conditional write across an ambiguous
		// (post-transmission) failure, which could report a wrong outcome.
		if isTransportError(err) && hop < maxHops && !(nonReplayableCall(op, args) && isAmbiguous(err)) {
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
//
// On an ambiguous transport failure (the request may have committed before the
// response was lost) it returns a non-nil error rather than a possibly-wrong
// false — the write is NOT transparently replayed against another server. A
// caller that gets a transport error must re-check state itself.
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
//
// Like SetNX, an ambiguous transport failure returns a non-nil error rather
// than a possibly-wrong false; the write is not replayed against another server.
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
//
// Like SetNX/CAS, an ambiguous transport failure returns a non-nil error rather
// than a possibly-wrong false; the delete is not replayed against another server.
func (c *Client) CompareAndDelete(ctx context.Context, key, expected []byte) (bool, error) {
	res, err := c.Call(ctx, "cad", wire.EncodeCADArgs(key, expected))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// Exists reports whether key is currently present (and live). An expired key
// reads as absent.
func (c *Client) Exists(ctx context.Context, key []byte) (bool, error) {
	res, err := c.Call(ctx, "exists", wire.EncodeKeyArgs(key))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// Flush wipes the ENTIRE KV keyspace. A single call fans out SERVER-SIDE: flush is
// keyless, so it falls to a round-robin server in pickInitialTarget, and the
// receiving node's Call intercept broadcasts it into every shard group's Raft log
// (see cluster.broadcastFlush). flush IS a nonReplayableOp: it is idempotent only
// while nothing else writes between attempts, so a blind replay after an AMBIGUOUS
// (post-transmission) transport failure could re-wipe data another client wrote in
// the interval. On an ambiguous failure Call therefore surfaces the error instead
// of silently re-flushing; the caller decides whether re-issuing is safe.
func (c *Client) Flush(ctx context.Context) error {
	_, err := c.Call(ctx, "flush", nil)
	return err
}

// GetDel atomically returns key's value and deletes it. found is false (and value
// nil) when the key was absent; a found empty value comes back as a non-nil
// zero-length slice.
func (c *Client) GetDel(ctx context.Context, key []byte) (value []byte, found bool, err error) {
	res, err := c.Call(ctx, "getdel", wire.EncodeKeyArgs(key))
	if err != nil {
		return nil, false, err
	}
	return wire.DecodeGetDelResult(res)
}

// GetSet atomically replaces key's value (with ttl > 0 setting an expiry) and
// returns the OLD value. found is false when the key had no prior value.
func (c *Client) GetSet(ctx context.Context, key, value []byte, ttl time.Duration) (old []byte, found bool, err error) {
	res, err := c.Call(ctx, "getset", wire.EncodePutArgs(key, value, ttl))
	if err != nil {
		return nil, false, err
	}
	return wire.DecodeGetDelResult(res)
}

// Persist removes key's TTL so it never expires. Returns true if a TTL was
// removed, false if the key is absent or already had no expiry.
func (c *Client) Persist(ctx context.Context, key []byte) (bool, error) {
	res, err := c.Call(ctx, "persist", wire.EncodeKeyArgs(key))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// TTL returns key's remaining time-to-live in milliseconds, following the Redis
// convention: -2 if the key is absent, -1 if it exists but has no expiry, else
// the remaining ms (>= 0).
func (c *Client) TTL(ctx context.Context, key []byte) (int64, error) {
	res, err := c.Call(ctx, "ttl", wire.EncodeKeyArgs(key))
	if err != nil {
		return 0, err
	}
	return wire.DecodeTTLResult(res)
}

// IncrEx atomically adds delta and returns the new value, setting the TTL only
// when the key is newly created — on an existing key the increment leaves the
// deadline untouched. This is the fixed-window rate-limit primitive: the first
// hit in a window creates the counter with the window's TTL, and subsequent hits
// increment it without extending the window.
func (c *Client) IncrEx(ctx context.Context, key []byte, delta int64, ttl time.Duration) (int64, error) {
	res, err := c.Call(ctx, "incr_ex", wire.EncodeIncrExArgs(key, delta, ttl))
	if err != nil {
		return 0, err
	}
	return wire.DecodeIncrResult(res)
}

// CompareAndExpire refreshes key's TTL only if its current value equals expected
// — the lock-renewal primitive (extend a lease only while you still hold the
// token). Returns true if the TTL was refreshed, false on a value mismatch or an
// absent key.
//
// Like the other conditional writes, an ambiguous transport failure returns a
// non-nil error rather than a possibly-wrong false; the write is not replayed.
func (c *Client) CompareAndExpire(ctx context.Context, key, expected []byte, ttl time.Duration) (bool, error) {
	res, err := c.Call(ctx, "caex", wire.EncodeCAEXArgs(key, expected, ttl))
	if err != nil {
		return false, err
	}
	return wire.DecodeCASResult(res)
}

// maxMGetKeys caps how many keys ride in one mget call — the u16 count field's
// range. MGet chunks each shard's key group to this size.
const maxMGetKeys = 65535

// MGet reads many keys in one round trip per shard, returning a slice aligned to
// keys where a missing key is a nil entry (a stored empty value is a non-nil
// zero-length slice). It groups keys by their owning shard — the same hash as
// PutBatch — issues one mget per shard (chunked to maxMGetKeys), and stitches the
// results back into the original key order. When the topology is unknown (no
// cache / NumShards==0) it cannot group, so it falls back to one get per key (the
// correctness floor), like PutBatch.
func (c *Client) MGet(ctx context.Context, keys [][]byte) ([][]byte, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	out := make([][]byte, len(keys))
	t := c.topology.get()
	if t == nil || t.NumShards == 0 {
		for i, k := range keys {
			v, err := c.Call(ctx, "get", wire.EncodeKeyArgs(k))
			if errors.Is(err, ErrNotFound) {
				continue // out[i] stays nil
			}
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	}
	// Group key INDICES by owning shard so each shard is read with one mget and
	// each returned value lands back at its original position.
	groups := make(map[int][]int)
	for i, k := range keys {
		shard := int(xxhash.Sum64(k) % uint64(t.NumShards)) //nolint:gosec // NumShards bounded by Config validation
		groups[shard] = append(groups[shard], i)
	}
	for _, idxs := range groups {
		for start := 0; start < len(idxs); start += maxMGetKeys {
			end := start + maxMGetKeys
			if end > len(idxs) {
				end = len(idxs)
			}
			chunk := idxs[start:end]
			batchKeys := make([][]byte, len(chunk))
			for j, ki := range chunk {
				batchKeys[j] = keys[ki]
			}
			res, err := c.Call(ctx, "mget", wire.EncodeMGetArgs(batchKeys))
			if err != nil {
				return nil, err
			}
			vals, err := wire.DecodeMGetResult(res)
			if err != nil {
				return nil, err
			}
			if len(vals) != len(chunk) {
				return nil, fmt.Errorf("client: mget returned %d values for %d keys", len(vals), len(chunk))
			}
			for j, ki := range chunk {
				out[ki] = vals[j]
			}
		}
	}
	return out, nil
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

// ambiguousError marks a transport failure that occurred AFTER a live
// connection was obtained — the request may have reached the server and
// committed before the response was lost. Replaying an outcome-returning
// conditional write across it can report a wrong result, so Call refuses to.
// It Unwraps to the underlying error so errors.Is/As (and isTransportError)
// see straight through the wrapper.
type ambiguousError struct{ err error }

func (e *ambiguousError) Error() string { return e.err.Error() }
func (e *ambiguousError) Unwrap() error { return e.err }

// nonReplayableOp reports whether replaying op after an ambiguous transport
// failure can corrupt its result. These mutations are outcome-returning or
// non-idempotent: if the first server committed but the response was lost, a
// replay sees the key already changed and returns the wrong result — a replayed
// incr_ex double-increments, getdel/getset return the retry's view not the
// original, and persist/caex report the wrong bit or extend a lease the caller
// can no longer be sure it still holds. Read-only / idempotent ops (get, exists,
// ttl, mget) are deliberately absent: replaying them cannot corrupt a result.
//
// "flush" and its INTERNAL per-group wrapper "__flush_shard__" (cluster's
// opFlushShardName, forwarded between nodes over a peer Client) are both here: a
// flush is idempotent ONLY while nothing writes between attempts, so replaying one
// whose commit succeeded but whose response was lost would silently re-wipe writes
// made in the interval. The wrapper must carry the same guard as "flush" or the
// double-apply hole reopens one layer down — cluster.proposeFlush relies on this to
// surface an ambiguous per-group flush instead of the peer Client re-sending it.
//
// "operate" mutates counters non-idempotently (ADD/MUL/SHL/…, and CHECK's
// server-side CAS), so a blind replay after an ambiguous post-commit failure
// would apply every increment twice; it surfaces the ambiguous error instead,
// exactly like "incr_ex".
// "vector_operate" and its named / multi-vector twins are the SAME op-list
// applied to a record held inside a POINT's payload instead of under a KV key,
// so they inherit "operate"'s reasoning verbatim: a replayed ADD double-counts,
// and a replayed CHECK is evaluated against a record the first attempt already
// changed. They are listed individually rather than matched by a "vector_"
// prefix so a future read-only vector op cannot be swept in by accident.
//
// "kv_query" and its INTERNAL per-group wrapper "__kv_query_shard__" are
// deliberately ABSENT, and the omission is a decision rather than an oversight.
// kv_query is a READ: registered OpReadOnly, never proposed into any Raft log,
// and with nothing a replay could double-apply — a replayed page is answered
// from the same cursor and returns the same rows. It sits beside "flush" in the
// cluster layer (both are keyless and both fan out to every shard group), and
// the resemblance ends exactly here: flush is on this list because re-wiping is
// a SECOND mutation, and a read has no such thing to repeat.
func nonReplayableOp(op string) bool {
	switch op {
	case "set_nx", "cas", "cad", "getdel", "getset", "incr_ex", "caex", "persist", "flush", "__flush_shard__", "operate",
		"vector_operate", "vector_named_operate", "vector_mv_operate":
		return true
	}
	return false
}

// nonReplayableCall is nonReplayableOp seen THROUGH the write-consistency
// envelope, and it is what the two retry decisions call — never nonReplayableOp
// directly.
//
// WHY THE ENVELOPE HAD TO BE LOOKED THROUGH. A write issued with active
// write-consistency options is not sent under its own name: the store wraps it
// (wcWire, root package) so the server's fanout dispatcher can run the
// post-commit barrier, and the name on the wire becomes wire.WCEnvelopeOp. A
// guard keyed on that name alone answers false for EVERY op, so exactly the
// conditional writes the guard exists for — a vector_operate ADD, an incr_ex, a
// cas — were retried across an ambiguous post-transmission failure and could
// apply twice. The envelope changes how a write is COMMITTED, never whether
// replaying it is safe, so the classification has to follow the inner op.
//
// An envelope whose head does not decode is treated as non-replayable. This
// client never builds such a frame, so it is either corruption or a caller
// hand-rolling one; declining to retry it costs at most one retry, while
// guessing "replayable" costs a double-applied write.
func nonReplayableCall(op string, args []byte) bool {
	if nonReplayableOp(op) {
		return true
	}
	if op != wire.WCEnvelopeOp {
		return false
	}
	inner, ok := wire.WCEnvelopeInnerOp(args)
	if !ok {
		return true
	}
	// A NESTED envelope is non-replayable outright. The server's fanout
	// dispatcher unwraps the envelope and dispatches whatever is inside, so a
	// frame wrapping a second envelope eventually reaches some real write, and
	// this guard cannot see which one without decoding an unbounded chain. This
	// client never builds a nested frame — wcWire wraps exactly once — so
	// declining is free, while recursing would let a hand-built frame choose how
	// much work the classification does.
	if inner == wire.WCEnvelopeOp {
		return true
	}
	return nonReplayableOp(inner)
}

// isAmbiguous reports whether err is (or wraps) an ambiguousError — a
// post-transmission transport failure that must not be blindly replayed for a
// non-replayable op.
func isAmbiguous(err error) bool {
	var a *ambiguousError
	return errors.As(err, &a)
}

// IsAmbiguous reports whether err (returned by Client.Call) is a post-transmission
// transport failure: the request may have committed on the server before the failure,
// so its outcome is UNKNOWN. Exposed for callers that forward a non-replayable op
// across peers and must not try another target after an ambiguous send — notably
// cluster.proposeFlush, whose per-group flush leg must surface such an error rather
// than re-issue the wipe against the next owner. A PRE-transmission failure (dial
// refused, no leader yet) is not ambiguous and is safe to retry elsewhere.
func IsAmbiguous(err error) bool { return isAmbiguous(err) }

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
		// See Call: a non-replayable conditional write is not retried across
		// an ambiguous (post-transmission) transport failure.
		if isTransportError(err) && hop < maxHops && !(nonReplayableCall(op, args) && isAmbiguous(err)) {
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
		// The request may have been transmitted and committed before the
		// response was lost — ambiguous, so Call won't replay a conditional
		// write across it. getOrCreatePool/acquire failures above are left
		// unwrapped: those happen before any bytes are sent (pre-transmission).
		return nil, &ambiguousError{err: callErr}
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
		// Post-transmission: mark ambiguous so CallFunc's retry guard refuses to
		// replay a conditional write (parity with callAddr — see Call).
		return &ambiguousError{err: callErr}
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
