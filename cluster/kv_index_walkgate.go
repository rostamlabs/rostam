// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/shard"
)

// A backfill walk aliases a LIVE MMAP. cache.IterateChunked hands the index the
// key and value bytes straight out of the shard's mapped pages, and
// shard.Store.Close (via cache.Close) munmaps that region — so a walk still
// running when its store closes does not read a stale value, it reads unmapped
// memory and the process dies.
//
// The fix is the one the cache already uses one level up for its own
// goroutines: a DRAIN, not a flag. cache.shard.Close closes stopSweeper and
// waits on sweepWG before it unmaps; cache.Close closes closeCh and waits on
// c.wg. Same shape here, one level higher: every backfill walk registers against
// the shard group it walks, and RemoveShardOwner signals those walks and waits
// for them to return BEFORE it calls Store.Close. The store is therefore never
// closed while a walk is inside it, and no check on the read path is needed —
// which matters, because a flag checked between chunks would not actually be
// sound unless Close held the shard write lock across the munmap, and that is a
// repo-wide read-path change.
//
// The gate is per shard group, and closing it does two things that must happen
// together: it stops NEW walks from registering, and it signals the ones already
// running. Without the first, a pass that had already snapshotted the store
// pointer could start a walk in the window between the drain and the Close.
type kvIndexWalkGate struct {
	mu sync.Mutex
	// closed is set by drain and cleared by reopen. While set, begin refuses, so
	// a pass holding a stale *shard.Store pointer cannot start walking a store
	// that is about to be closed.
	closed bool
	// stop is closed by drain to signal every in-flight walk. Replaced by reopen,
	// so a group that is removed and later re-added gets a fresh signal.
	stop chan struct{}
	// wg tracks in-flight walks. drain waits on it OUTSIDE mu, so a long walk
	// blocks the removal (which is the point) and not every other group's gate.
	wg sync.WaitGroup
}

// kvIndexWalkProbe, when set, is called at every abort check of every KV index
// walk — backfill and scan alike, at the same stride the abort check runs on.
//
// TEST SEAM, nil in production. The two tests that race a real removal against a
// real walk previously gave the walk a 2 ms head start and hoped: if the walk
// finished first the removal exercised the drain's WAIT and never its ABORT, and
// the tests failed with "the removal never raced one" on a fast or lightly
// loaded box. The probe lets a test PARK a walk at a known point inside the
// store, so the overlap is a fact rather than a race the test hopes to win.
//
// An atomic pointer rather than a plain variable so installing and clearing it
// is not itself a data race with the walk that reads it. The cost in production
// is one atomic load per kvIndexAbortCheckEvery keys.
var kvIndexWalkProbe atomic.Pointer[func(group int)]

// fireKVIndexWalkProbe runs the test probe if one is installed.
func fireKVIndexWalkProbe(group int) {
	if p := kvIndexWalkProbe.Load(); p != nil {
		(*p)(group)
	}
}

// kvIndexWalkGateFor returns group's gate, creating it on first use.
//
// A GATE CREATED AFTER THE NODE-WIDE SHUTDOWN LATCH IS BORN CLOSED. Gates are
// lazy, so before this the FIRST scan on a hosted group could create its gate
// AFTER drainAllKVIndexWalks had already snapshotted the map: the drain would
// find nothing to wait for, Node.Close would go on to unmap the stores, and the
// scan would be walking pages that no longer exist. The latch closes that by
// construction — once shutdown has begun, every gate this hands out, existing or
// brand new, refuses to admit a walk.
func (n *Node) kvIndexWalkGateFor(group int) *kvIndexWalkGate {
	n.kvIndexWalkMu.Lock()
	defer n.kvIndexWalkMu.Unlock()
	if n.kvIndexWalkGates == nil {
		n.kvIndexWalkGates = make(map[int]*kvIndexWalkGate)
	}
	g := n.kvIndexWalkGates[group]
	if g == nil {
		g = &kvIndexWalkGate{stop: make(chan struct{})}
		if n.kvIndexWalkAllClosed {
			g.closed = true
			close(g.stop)
		}
		n.kvIndexWalkGates[group] = g
	}
	return g
}

// beginKVIndexWalk registers a walk on group. It reports ok=false when the group
// is being (or has been) removed, in which case the caller must not walk that
// store at all. On ok the caller MUST call done when the walk returns, and
// should stop the walk when stop is closed.
func (n *Node) beginKVIndexWalk(group int) (stop <-chan struct{}, done func(), ok bool) {
	g := n.kvIndexWalkGateFor(group)
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil, func() {}, false
	}
	g.wg.Add(1)
	return g.stop, g.wg.Done, true
}

// closeKVIndexWalkGate shuts group's gate: no new walk can register, and every
// walk already running is signalled. It does NOT wait — see waitKVIndexWalks.
//
// THE SPLIT EXISTS SO THE SHUT CAN HAPPEN UNDER n.shardMu. Shutting and re-arming
// are the gate's half of the store swap, and doing them outside that lock let
// AddShardOwner and RemoveShardOwner interleave the wrong way round: the add
// could install the replacement store and re-arm the gate, and the concurrent
// remove could then shut it — leaving a live store behind a permanently closed
// gate, where every scan answers ErrWalkAborted and the observer can never
// backfill. Both are now decided in the same critical section that swaps
// n.shards, so the gate always ends in the state the store slot ended in. The
// WAIT stays outside, because it can block for a whole chunk.
func (n *Node) closeKVIndexWalkGate(group int) {
	g := n.kvIndexWalkGateFor(group)
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.closed {
		g.closed = true
		close(g.stop)
	}
}

// waitKVIndexWalks waits for every walk already running on group. The caller
// must have shut the gate first, or a new walk can start while it waits.
//
// It is called with no lock held and may block for as long as one walk takes to
// notice the signal — bounded by the walk's abort check, not by the size of the
// keyspace.
func (n *Node) waitKVIndexWalks(group int) {
	n.kvIndexWalkGateFor(group).wg.Wait()
}

// drainKVIndexWalks closes group's gate and waits for every walk already running
// on it. After it returns, no walk is inside that group's store and none can
// start, so the store is safe to close.
func (n *Node) drainKVIndexWalks(group int) {
	n.closeKVIndexWalkGate(group)
	n.waitKVIndexWalks(group)
}

// reopenKVIndexWalks re-arms group's gate after the group is hosted again
// (AddShardOwner). The stop channel is REPLACED rather than reused: the old one
// is closed, and a walk registered against it would abort immediately.
func (n *Node) reopenKVIndexWalks(group int) {
	g := n.kvIndexWalkGateFor(group)
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.closed {
		return
	}
	g.closed = false
	g.stop = make(chan struct{})
}

// installKVQueryScanGate puts group's SCAN walk behind the same per-group gate
// the backfill walk registers with, by installing a gated walker on the store's
// read-only dispatcher (shard.Store.SetKVWalker → ops.TxContext.SetWalker).
//
// WHY A SCAN NEEDS THE GATE AT LEAST AS MUCH AS A BACKFILL. Both walk the whole
// keyspace and both alias the store's live mmap, but a scan is CLIENT-DRIVEN:
// any reader can start one at any moment, and a 5M-key page is a longer window
// onto pages Close is about to unmap than a backfill that runs once per
// definition. Without this the leaf walks through ops.CacheWalker, which cannot
// fail — so a scan racing RemoveShardOwner reads unmapped memory and takes the
// process down, and ops.ErrKVQueryUnavailable is unreachable.
//
// It must be installed for EVERY hosted store, wherever one comes into
// existence: both node constructors and AddShardOwner.
func (n *Node) installKVQueryScanGate(group int, s *shard.Store) {
	plain := s.CacheWalker()
	s.SetKVWalker(func(yield func(key, value []byte) bool) error {
		return n.gatedKVWalk(group, plain, yield)
	})
}

// gatedKVWalk runs walk under group's gate: it registers the walk (so a removal
// waits for it), aborts at a chunk boundary once the gate is shut, and reports
// kvindex.ErrWalkAborted rather than a short walk that returned normally.
//
// The abort is by RETURNING FALSE from the yield, which unwinds the cache's own
// early-stop path with every lock released — never by abandoning the walk. The
// check runs on the same stride as the backfill's (kvIndexAbortCheckEvery),
// which is what bounds how long RemoveShardOwner waits to a few thousand entries
// instead of a whole keyspace.
//
// A walk that could not register (ok=false) means the group is already being
// removed. It must not touch that store at all, and the caller must be told:
// a scan that visited nothing looks exactly like a scan that matched nothing.
func (n *Node) gatedKVWalk(group int, walk kvindex.Walker, yield func(key, value []byte) bool) error {
	stop, done, ok := n.beginKVIndexWalk(group)
	if !ok {
		return fmt.Errorf("%w: shard group %d is being removed from this node", kvindex.ErrWalkAborted, group)
	}
	defer done()

	var visited uint64
	aborted := false
	werr := walk(func(key, value []byte) bool {
		visited++
		if visited%kvIndexAbortCheckEvery == 0 {
			fireKVIndexWalkProbe(group)
			select {
			case <-stop:
				aborted = true
				return false
			default:
			}
		}
		return yield(key, value)
	})
	if werr != nil {
		return werr
	}
	if aborted {
		return fmt.Errorf("%w: shard group %d was removed from this node mid-walk", kvindex.ErrWalkAborted, group)
	}
	return nil
}

// drainAllKVIndexWalks shuts every group's gate and waits for every in-flight
// walk. Node.Close uses it so shutdown is bounded by a walk's abort check rather
// than by the length of a full-keyspace walk: stopKVIndexObserver alone WOULD be
// safe (a pass runs on the observer goroutine, so waiting for that goroutine
// waits for its walk), but it would wait for the whole walk to finish.
func (n *Node) drainAllKVIndexWalks() {
	n.kvIndexWalkMu.Lock()
	// THE LATCH GOES UP BEFORE THE SNAPSHOT, and that ordering is the whole
	// point. Gates are created lazily, so a group that has never been walked has
	// none — and a first scan arriving between the snapshot and the unmap would
	// otherwise create a fresh, OPEN gate and walk a store Close is about to
	// unmap. With the latch set first, any gate created from here on is born
	// closed, so the snapshot below does not have to be complete to be safe.
	n.kvIndexWalkAllClosed = true
	groups := make([]int, 0, len(n.kvIndexWalkGates))
	for group := range n.kvIndexWalkGates {
		groups = append(groups, group)
	}
	n.kvIndexWalkMu.Unlock()
	for _, group := range groups {
		n.drainKVIndexWalks(group)
	}
}
