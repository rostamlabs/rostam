// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/shard"
)

// KV index CRUD, following the __set_catalog__ precedent exactly. All three ops
// are shardless and node-local: Node.Call dispatches them off n.adminOps BEFORE
// any key routing, because an admin op targets the node it was sent to, not a
// key's shard owner. The client speaks these same strings, so there is exactly
// one name per op end to end.
//
// None of them gets a wire.BuiltinOps row — __set_catalog__ has none either.
// Only kv_query needs one, because only kv_query must also work through the ops
// registry on the Direct (single-store) path.
const (
	// opKVIndexSetName upserts one definition into the meta catalog (an entry
	// with Enabled=false drops it). Proposed here when this node is the meta
	// leader, forwarded to the leader otherwise.
	opKVIndexSetName = "__kv_index_set__"
	// opKVIndexListName returns every definition in the meta catalog paired with
	// a ready bit that holds only when EVERY shard group reports ready.
	opKVIndexListName = "__kv_index_list__"
	// opKVIndexReadyName is the per-group readiness LEAF the list handler
	// gathers. It answers for ONE shard group from that group's local index Set
	// and never fans out, so the list's fan-out stays flat (the shape
	// __flush_shard__ gives the flush broadcast).
	opKVIndexReadyName = "__kv_index_ready__"
)

// kvIndexSetTimeout bounds the meta-Raft commit a forwarded __kv_index_set__
// performs on the leader, matching the other forwarded meta handlers' 5s
// internal deadline (handleSetCatalog, handlePBSetISR).
const kvIndexSetTimeout = 5 * time.Second

// kvIndexReadyGroupTimeout bounds ONE group's leg of the readiness gather,
// mirroring flushBroadcastGroupTimeout. The legs run concurrently, so a list
// costs one slow group's latency rather than their sum, and a group that never
// answers makes its bit false instead of hanging the call.
const kvIndexReadyGroupTimeout = 10 * time.Second

// validateKVIndexDef is the FULL admission check a definition must pass before
// it reaches the meta log: the wire shape check AND the path parse the observer
// performs after the commit.
//
// THE TWO MUST RUN TOGETHER, HERE, because they disagree about what is legal.
// wire.KVIndexDef.Validate is a shallow string check — sdk/record imports
// sdk/wire, so the path grammar cannot live there — and it accepts paths
// kvindex.DefFrom later refuses: "#count" (a count of no field), "#00001" (a
// non-canonical position), a name carrying a character the grammar forbids,
// "a#count#count". Checked only shallowly, such a definition COMMITS; every
// node then rejects it at install time (Stats().KVIndex.RejectedDefs = 1 on all
// of them); and a query naming it is worse than a hard failure, because the
// name IS in the meta catalog, so classifyKVQueryErr rewrites each group's
// PERMANENT "no such index" into the RETRYABLE ErrIndexBuilding — and a
// well-behaved client then retries forever, at one full cluster fan-out per
// attempt, for an index that can never install anywhere.
//
// ONLY AN ENABLED DEFINITION IS PARSED. Enabled=false is a DELETE keyed on Name
// alone: its PayloadPath is a placeholder ("-") nothing ever reads, and the FSM
// removes the entry instead of storing it, so no observer will ever build it.
// Parsing it would fail a drop over a field that does not matter — and would
// make a definition written by a NEWER build undroppable by this one.
//
// IT DOES NOT MOVE INTO THE META FSM. An FSM check must decide identically on
// every node at every version or the state machines diverge, and the path
// grammar is precisely the part that may widen between builds. So the FSM keeps
// the version-stable shape check and admission does the parse: a definition this
// build cannot build is refused at the door, and one a newer build admits is
// still applied identically everywhere.
func validateKVIndexDef(d wire.KVIndexDef) error {
	if err := d.Validate(); err != nil {
		return err
	}
	if !d.Enabled {
		return nil
	}
	// The error wraps wire.ErrKVIndexDef, which every transport classifies as
	// PERMANENT (ops.KVQueryErrorFamily) — the one classification that stops the
	// retry loop described above.
	_, err := kvindex.DefFrom(d, 0)
	return err
}

// handleSetKVIndex applies a FORWARDED index-definition write. Like
// handleSetCatalog it runs on the meta-Raft leader — the sender selected this
// node as the leader — so it proposes the entry LOCALLY and never re-enters the
// forwarding path.
//
// It must not call Node.SetKVIndex, whose own leadership check would forward
// AGAIN. Under a leadership flap that is a chain: A forwards to B, B has just
// lost leadership and forwards to C, and a stale view can point C back at A —
// each hop a goroutine parked on a 5 s call, for a write that should have failed
// fast. Proposing directly makes a mis-addressed forward a clean, immediate
// hraft.ErrNotLeader that the CALLER re-resolves, exactly as __set_catalog__
// behaves. It also drops the redundant read-your-writes wait: the leader's Raft
// Apply has already returned, so its own FSM has applied the entry by
// construction.
func (n *Node) handleSetKVIndex(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errNoMeta
	}
	d, err := wire.DecodeKVIndexSetArgs(args)
	if err != nil {
		return nil, fmt.Errorf("cluster: %s decode: %w", opKVIndexSetName, err)
	}
	err = n.meta.ApplySetKVIndex(d, kvIndexSetTimeout)
	if errors.Is(err, hraft.ErrNotLeader) {
		// A REFUSAL THE CALLER CAN ACT ON, not a bare error. Unlike the other
		// forwarded meta writes, this op has a client-facing entry point: the
		// native client's CreateKVIndex and DropKVIndex send __kv_index_set__
		// straight to whichever node they are connected to, so a follower is a
		// perfectly ordinary destination for it — and CreateKVIndex's own doc
		// promises the write "can be issued against any node".
		//
		// hraft.ErrNotLeader is not shard.ErrNotLeader, so server.mapResult never
		// answered StatusNotLeader for it and the client's own not-leader hop loop
		// could not fire: the caller saw a flat failure whose remedy (reconnect to
		// the meta leader) it had no way to learn. Answering with the shard
		// package's typed error, carrying the meta leader's CLIENT-FACING address,
		// makes the existing hop do the work.
		//
		// Still not a forward. Re-entering Node.SetKVIndex would forward again,
		// and under a leadership flap that is a chain of 5 s parked calls (see this
		// function's doc); one hop decided by the CALLER, which re-resolves the
		// topology as it goes, is what __set_catalog__'s design intends and what
		// this now delivers.
		//
		// BOTH identities are kept. The shard error is what the transport reads
		// (errors.Is for the status, errors.As for the hint); hraft.ErrNotLeader
		// stays wrapped so a cluster-internal caller that checks the meta layer's
		// own sentinel still matches.
		return nil, fmt.Errorf("%w: %w", &shard.NotLeaderError{LeaderAddr: n.metaLeaderServerAddr()}, hraft.ErrNotLeader)
	}
	return nil, err
}

// SetKVIndex durably records one KV index definition in the meta-Raft catalog.
// Like SetCollectionPartitions it can be called on any node: a non-leader
// forwards the write to the meta-Raft leader via the __kv_index_set__ admin op,
// so the write reaches consensus regardless of which node issued it. Returns
// errNoMeta in single-node mode.
//
// A definition with Enabled=false DELETES the named index cluster-wide.
//
// On return THIS node's local FSM already reflects the write (read-your-writes),
// so a create-then-query on the same connection does not race its own observer
// into a spurious "no such index". The wait polls the STATE, never a Raft index:
// raft advances applied_index at ENQUEUE, not at apply, so an index barrier is a
// proxy that can pass before the catalog has actually changed.
func (n *Node) SetKVIndex(d wire.KVIndexDef, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	if err := validateKVIndexDef(d); err != nil {
		return fmt.Errorf("cluster: SetKVIndex: %w", err)
	}
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return fmt.Errorf("cluster: SetKVIndex: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if _, err = cl.Call(ctx, opKVIndexSetName, wire.EncodeKVIndexSetArgs(d)); err != nil {
			return err
		}
	} else {
		if err := n.meta.ApplySetKVIndex(d, timeout); err != nil {
			return err
		}
	}
	return n.waitLocalKVIndex(d, timeout)
}

// waitLocalKVIndex blocks until this node's own meta-FSM reflects the definition
// just written, or timeout elapses. A disable is satisfied by the name's
// ABSENCE, which is how the FSM stores it.
func (n *Node) waitLocalKVIndex(d wire.KVIndexDef, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		e, ok := n.meta.FSM.KVIndexLookup(d.Name)
		if !d.Enabled {
			if !ok {
				return nil
			}
		} else if ok && kvIndexDefEqual(e.Def, d) {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("cluster: kv index write for %q not locally applied within %s", d.Name, timeout)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// kvIndexDefEqual compares two definitions by value. wire.KVIndexDef holds a
// byte slice, so == would not compile and reflect.DeepEqual would be both slower
// and quietly sensitive to nil-vs-empty; an empty prefix and a nil prefix mean
// the same thing (index the whole keyspace) and must compare equal.
func kvIndexDefEqual(a, b wire.KVIndexDef) bool {
	return a.Name == b.Name &&
		a.PayloadPath == b.PayloadPath &&
		a.Kind == b.Kind &&
		a.Enabled == b.Enabled &&
		string(a.KeyPrefix) == string(b.KeyPrefix)
}

// handleListKVIndexes answers __kv_index_list__ with every definition in the
// meta catalog and, per definition, a bit that is true only when EVERY shard
// group reports the index ready.
func (n *Node) handleListKVIndexes(args []byte) ([]byte, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("cluster: %s takes no arguments, got %d bytes", opKVIndexListName, len(args))
	}
	defs, ready, err := n.ListKVIndexes(kvIndexReadyGroupTimeout)
	if err != nil {
		return nil, err
	}
	return wire.EncodeKVIndexList(defs, ready), nil
}

// ListKVIndexes returns the KV index catalog from this node's local meta-FSM,
// sorted by name, paired with a per-definition readiness bit.
//
// READINESS IS AN AGGREGATE OVER GROUPS, and it fails closed. Each shard group
// carries its own index Set, so a definition is only usable cluster-wide when
// every group has finished backfilling it; a group that is still building, that
// never installed the definition, or that cannot be reached at all makes the bit
// FALSE. It is never dropped from the answer — a definition that exists must
// appear in the list even when nothing can be said about its progress, or a
// caller polling for readiness would see the name vanish and conclude the
// create failed.
func (n *Node) ListKVIndexes(timeout time.Duration) ([]wire.KVIndexDef, []bool, error) {
	if n.meta == nil {
		return nil, nil, errNoMeta
	}
	cat := n.meta.FSM.KVIndexes()
	if len(cat) == 0 {
		return nil, nil, nil
	}
	names := make([]string, 0, len(cat))
	for name := range cat {
		names = append(names, name)
	}
	sort.Strings(names)
	defs := make([]wire.KVIndexDef, len(names))
	for i, name := range names {
		defs[i] = cat[name].Def
	}

	results, errs := n.forEachGroup(timeout, func(ctx context.Context, group int) ([]byte, error) {
		return n.kvIndexReadyFromGroup(ctx, group, names, timeout)
	})
	return defs, aggregateKVIndexReady(names, results, errs), nil
}

// aggregateKVIndexReady ANDs the per-group readiness replies into one bit per
// name. A leg that errored, or whose reply cannot be decoded, contributes false
// for EVERY name: the two are the same fact ("this group could not tell us"),
// and the only safe reading of that is not-ready. The returned slice always has
// one entry per name.
func aggregateKVIndexReady(names []string, results [][]byte, errs []error) []bool {
	ready := make([]bool, len(names))
	for i := range ready {
		ready[i] = true
	}
	for g := range results {
		var bits []bool
		if errs[g] == nil {
			var derr error
			bits, derr = decodeKVIndexReadyReply(results[g], len(names))
			if derr != nil {
				bits = nil
			}
		}
		for i := range ready {
			ready[i] = ready[i] && i < len(bits) && bits[i]
		}
	}
	return ready
}

// kvIndexReadyFromGroup asks ONE shard group whether each named index is ready
// there: locally when this node hosts the group, otherwise by forwarding the
// leaf to an owner. Readiness is node-local derived state with no Raft entry
// behind it, so a hosted group is answered straight from its Set.
func (n *Node) kvIndexReadyFromGroup(ctx context.Context, group int, names []string, timeout time.Duration) ([]byte, error) {
	if s := n.getShard(group); s != nil {
		return encodeKVIndexReadyReply(kvIndexReadyBits(s.KVIndex(), names)), nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return n.forwardTimeout(group, opKVIndexReadyName, encodeKVIndexReadyReq(group, names), timeout)
}

// handleKVIndexReady is the node-local leaf for opKVIndexReadyName. It answers
// for the ONE group named in the payload and never forwards, so the list's
// gather stays flat. ErrNoShardOwner when this node does not host the group lets
// the sender's forward() loop move on to the next owner.
func (n *Node) handleKVIndexReady(args []byte) ([]byte, error) {
	group, names, err := decodeKVIndexReadyReq(args)
	if err != nil {
		return nil, err
	}
	if group < 0 || group >= n.cfg.NumShards {
		return nil, fmt.Errorf("cluster: %s: shard %d out of range [0,%d)", opKVIndexReadyName, group, n.cfg.NumShards)
	}
	s := n.getShard(group)
	if s == nil {
		return nil, ErrNoShardOwner
	}
	return encodeKVIndexReadyReply(kvIndexReadyBits(s.KVIndex(), names)), nil
}

// kvIndexReadyBits reads one Set's readiness for each name. A nil Set (a store
// built without an index) and an uninstalled name both read as not ready. The
// parameter is the concrete *kvindex.Set rather than an interface on purpose: a
// nil *Set stored in an interface is not == nil, so the guard below would not
// fire and a shard with no index would panic instead of answering "not ready".
func kvIndexReadyBits(idx *kvindex.Set, names []string) []bool {
	bits := make([]bool, len(names))
	if idx == nil {
		return bits
	}
	for i, name := range names {
		bits[i] = idx.IsReady(name)
	}
	return bits
}

// __kv_index_ready__ request wire form:
//
//	[group u32 BE][n u16 BE]{ [nameLen u8][name] }
//
// The names travel with the request so ONE round trip per group answers for the
// whole catalog rather than one per (group, name) pair. Every count is bounded
// against the remaining input before it drives an allocation, and the frame is
// self-delimiting, so trailing bytes are corruption rather than a
// forward-compatible extension.
func encodeKVIndexReadyReq(group int, names []string) []byte {
	total := 6
	for _, name := range names {
		total += 1 + len(name)
	}
	buf := make([]byte, 0, total)
	buf = binary.BigEndian.AppendUint32(buf, uint32(group)) //nolint:gosec // bounded by NumShards at every call site
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(names)))
	for _, name := range names {
		buf = append(buf, byte(len(name)))
		buf = append(buf, name...)
	}
	return buf
}

// kvIndexReadyNameMinBytes is the fewest bytes one name entry can occupy: its
// 1-byte length prefix. Used to bound the declared count against the remaining
// input before any allocation sized by it.
const kvIndexReadyNameMinBytes = 1

func decodeKVIndexReadyReq(b []byte) (group int, names []string, err error) {
	if len(b) < 6 {
		return 0, nil, fmt.Errorf("cluster: %s: %w", opKVIndexReadyName, wire.ErrShortArgs)
	}
	group = int(binary.BigEndian.Uint32(b[0:4]))
	count := int(binary.BigEndian.Uint16(b[4:6]))
	off := 6
	if !wire.CountFitsIn(count, len(b)-off, kvIndexReadyNameMinBytes) {
		return 0, nil, fmt.Errorf("cluster: %s: %w", opKVIndexReadyName, wire.ErrShortArgs)
	}
	if count > wire.KVIndexMaxDefs {
		return 0, nil, fmt.Errorf("cluster: %s: %d names exceeds the %d-definition cap", opKVIndexReadyName, count, wire.KVIndexMaxDefs)
	}
	names = make([]string, 0, count)
	for i := 0; i < count; i++ {
		if len(b)-off < 1 {
			return 0, nil, fmt.Errorf("cluster: %s: %w", opKVIndexReadyName, wire.ErrShortArgs)
		}
		nameLen := int(b[off])
		off++
		if len(b)-off < nameLen {
			return 0, nil, fmt.Errorf("cluster: %s: %w", opKVIndexReadyName, wire.ErrShortArgs)
		}
		names = append(names, string(b[off:off+nameLen]))
		off += nameLen
	}
	if off != len(b) {
		return 0, nil, fmt.Errorf("cluster: %s: trailing bytes", opKVIndexReadyName)
	}
	return group, names, nil
}

// __kv_index_ready__ reply wire form: [n u16 BE][ready u8 × n], positionally
// aligned with the request's names.
func encodeKVIndexReadyReply(bits []bool) []byte {
	buf := make([]byte, 0, 2+len(bits))
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(bits))) //nolint:gosec // bounded by KVIndexMaxDefs at every call site
	for _, b := range bits {
		if b {
			buf = append(buf, 1)
		} else {
			buf = append(buf, 0)
		}
	}
	return buf
}

// decodeKVIndexReadyReply reads a reply and requires it to answer for exactly
// want names. A reply of a different length is a disagreement about the
// question, not a partial answer, so it is an error the caller turns into
// "not ready" rather than silently aligning bits to the wrong names.
func decodeKVIndexReadyReply(b []byte, want int) ([]bool, error) {
	if len(b) < 2 {
		return nil, fmt.Errorf("cluster: %s reply: %w", opKVIndexReadyName, wire.ErrShortArgs)
	}
	count := int(binary.BigEndian.Uint16(b[0:2]))
	if len(b)-2 != count {
		return nil, fmt.Errorf("cluster: %s reply: declares %d bits, carries %d bytes", opKVIndexReadyName, count, len(b)-2)
	}
	if count != want {
		return nil, fmt.Errorf("cluster: %s reply: answers for %d names, asked about %d", opKVIndexReadyName, count, want)
	}
	bits := make([]bool, count)
	for i := 0; i < count; i++ {
		bits[i] = b[2+i] != 0
	}
	return bits, nil
}

// forEachGroup runs fn once for EVERY shard group, CONCURRENTLY, each leg bound
// by its own timeout, and returns the results and errors indexed BY GROUP.
//
// Two properties make it the shared shape for every per-group fan-out (the
// readiness gather here, the kv_query fan-out that reads across all groups):
//
//   - Latency is the SLOWEST group, not the sum over groups. A sequential loop
//     over NumShards remote calls is a per-group RTT multiplied by the shard
//     count, which for a read is the difference between a fast answer and a
//     timeout.
//   - One hung group cannot hang the call. The bound is enforced OUTSIDE fn —
//     a leg that never looks at its ctx still gives up its slot at the deadline
//     and reports a DeadlineExceeded error — so a group whose owner accepts the
//     connection and then goes silent costs one timeout, not the whole call.
//     The abandoned leg finishes into a buffered channel and is collected; it
//     writes to nothing the caller can observe.
//
// Results are positional: results[g]/errs[g] always describe group g, whatever
// order the legs completed in, and both slices are always NumShards long. A
// timeout <= 0 means no bound.
func (n *Node) forEachGroup(timeout time.Duration, fn func(ctx context.Context, group int) ([]byte, error)) ([][]byte, []error) {
	return n.forEachGroupIn(nil, timeout, fn)
}

// forEachGroupIn is forEachGroup restricted to the groups `visit` marks true —
// nil meaning every group, which is forEachGroup itself.
//
// It exists because the kv_query fan-out stops asking a group once that group
// has answered in full, and on a long paging run most groups are finished: a
// pass that spawned a goroutine, a context and a timer for each of them, only
// for its fn to return immediately, would pay per page for work it has already
// decided not to do. Results stay positional and NumShards long whatever is
// visited, so a skipped group reads as the zero value — which is what "asked
// nothing, heard nothing" should look like.
func (n *Node) forEachGroupIn(visit []bool, timeout time.Duration, fn func(ctx context.Context, group int) ([]byte, error)) ([][]byte, []error) {
	results := make([][]byte, n.cfg.NumShards)
	errs := make([]error, n.cfg.NumShards)
	var wg sync.WaitGroup
	for g := 0; g < n.cfg.NumShards; g++ {
		if visit != nil && (g >= len(visit) || !visit[g]) {
			continue
		}
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			ctx := context.Background()
			if timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
			}
			type outcome struct {
				body []byte
				err  error
			}
			// Buffered, so an abandoned leg's send never blocks and its
			// goroutine still exits once fn returns.
			ch := make(chan outcome, 1)
			go func() {
				body, err := fn(ctx, g)
				ch <- outcome{body, err}
			}()
			select {
			case o := <-ch:
				results[g], errs[g] = o.body, o.err
			case <-ctx.Done():
				errs[g] = fmt.Errorf("cluster: shard group %d: %w", g, ctx.Err())
			}
		}(g)
	}
	wg.Wait()
	return results, errs
}
