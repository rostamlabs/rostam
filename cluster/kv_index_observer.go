// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/shard"
)

// kvIndexObserveInterval is how often the node re-reads the meta catalog. It is
// a POLL, not a callback, because the meta FSM has no non-leaf apply hook and an
// O(live-keys) backfill must never run under its write lock.
//
// The only existing hook in MetaFSM.Apply is the test-only leaseRenewObserver,
// which fires while the FSM write lock is held and is documented as a LEAF: it
// may take one mutex and must not re-enter the FSM. Installing definitions and
// walking a shard's whole keyspace is the opposite of a leaf, so it belongs on a
// goroutine of its own — and a poll comparing the applied index costs one atomic
// load per second when nothing has changed.
const kvIndexObserveInterval = time.Second

// startKVIndexObserver derives this node's local KV index state from the meta
// catalog and keeps it in step with it.
//
// THE INSTALL IS SYNCHRONOUS, THE BACKFILL IS NOT, and the split is the whole
// point of this function's shape. Installing is cheap and has to happen before
// the node serves anything: from the moment a definition is installed every
// apply reindexes through it, so a write that lands in the first second of the
// process's life is recorded rather than lost. The WALK is O(live keys) — on a
// restart with a large keyspace and definitions already in the catalog, running
// it here would hold the node in its constructor for the length of a full-cache
// walk before it could accept a single request. It runs on the observer
// goroutine instead, and until it finishes the index is not ready, so a query
// naming it gets the retryable ErrIndexBuilding: the designed answer, not a
// silently short one.
//
// The goroutine then re-derives whenever the meta FSM's applied index moves —
// any meta write at all, which deliberately includes the OpSetPlacement a
// rebalance commits when this node starts hosting a new shard group whose index
// Set is empty.
func (n *Node) startKVIndexObserver() {
	if n.meta == nil {
		return // single-node mode: no meta catalog, so nothing to derive from
	}
	n.kvIndexStop = make(chan struct{})
	stop := n.kvIndexStop
	n.installKVIndexDefs() // definitions in place before the first apply
	n.kvIndexWg.Add(1)
	go func() {
		defer n.kvIndexWg.Done()
		n.applyKVIndexDefs() // the first backfill, off the construction path
		t := time.NewTicker(kvIndexObserveInterval)
		defer t.Stop()
		var seen uint64
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				if idx := n.meta.FSM.AppliedIndex(); idx != seen {
					seen = idx
					n.applyKVIndexDefs()
				}
			}
		}
	}()
}

// stopKVIndexObserver ends the polling goroutine and waits for it to exit.
// Idempotent (Close calls it, and a test may have called it first), and safe on
// a node whose observer was never started.
//
// The wait is only bounded because a pass CHECKS the stop signal: see
// applyKVIndexDefs. Without that, Close on a node with a large keyspace and
// several hosted groups would block for the length of every remaining backfill.
func (n *Node) stopKVIndexObserver() {
	n.kvIndexStopOnce.Do(func() {
		if n.kvIndexStop != nil {
			close(n.kvIndexStop)
		}
	})
	n.kvIndexWg.Wait()
}

// kvIndexStopped reports whether the observer has been asked to exit. Reading
// n.kvIndexStop is race-free: it is assigned once in startKVIndexObserver before
// the goroutine exists and never reassigned, and observing a channel close needs
// no further synchronisation.
func (n *Node) kvIndexStopped() bool {
	if n.kvIndexStop == nil {
		return false
	}
	select {
	case <-n.kvIndexStop:
		return true
	default:
		return false
	}
}

// applyKVIndexDefs brings every hosted shard's index Set into line with the meta
// catalog: install the definitions, then backfill the ones that are not ready.
//
// ENABLE THEN BACKFILL, in that order, and the order is the whole correctness
// argument. From the moment a definition is installed, every subsequent write
// reindexes through it; the walk that follows can therefore only ever ADD
// postings the write path might also have added. A posting can be STALE — one
// wasted lookup, since every candidate is re-read and re-checked against the
// live value — but never MISSING. Reversing the two would drop every write that
// landed during the walk, silently, and produce a proper subset of the truth.
//
// A FLUSH IS NOT A REASON TO RE-BACKFILL. kvindex.Set.Reset clears every posting
// AND marks every definition ready, because an empty posting set is exactly
// right over an empty keyspace. The `!IsReady` gate below is what makes that
// hold: a flushed definition stays ready, so this loop walks nothing.
//
// The Set is taken unguarded and installed into while applies are running. That
// is the enable-then-backfill shape, not an oversight: the Set's own lock
// serialises Install against every concurrent Reindex, and a write that
// interleaves is precisely the write the ordering above is designed to keep.
func (n *Node) applyKVIndexDefs() { n.kvIndexPass(false) }

// installKVIndexDefs is applyKVIndexDefs' first half alone: put the definitions
// in place on every hosted group and walk nothing. Used at node start, where the
// install must precede the first apply but the walk must not precede the node
// serving. See startKVIndexObserver.
func (n *Node) installKVIndexDefs() { n.kvIndexPass(true) }

// kvIndexPass is the body of both. installOnly skips every walk.
//
// IT CHECKS THE STOP SIGNAL BETWEEN SHARDS AND BETWEEN DEFINITIONS, and
// deliberately NOT inside a walk. Close waits on this pass, so a node hosting
// several groups over a large keyspace would otherwise take the sum of every
// remaining backfill to shut down. Aborting MID-walk would be worse than slow:
// the walk would return normally, having visited a prefix of the keyspace, and
// kvindex would publish that subset as a complete index. A walk either finishes
// or reports an error; it never stops early and calls itself done.
//
// The whole pass is serialised by kvIndexApplyMu so the observer's pass and a
// direct call cannot interleave two walks over one definition.
func (n *Node) kvIndexPass(installOnly bool) {
	if n.meta == nil {
		return
	}
	n.kvIndexApplyMu.Lock()
	defer n.kvIndexApplyMu.Unlock()
	n.kvIndexPasses.Add(1)

	defs := n.kvIndexDefsFromCatalog()
	shards := n.snapshotShards()

	// THE INSTALL IS SKIPPED WHEN NOTHING IT DEPENDS ON MOVED. The poll fires on
	// ANY meta write, and in PB mode the primary-liveness beacons alone move the
	// applied index every interval — so without this, every hosted group rebuilt
	// its definition and posting maps once a second forever, for a catalog that
	// had not changed. The readiness scan below still runs on every pass: it is
	// one map read per definition, and skipping it would strand a definition
	// whose backfill was cut short.
	fingerprint := kvIndexPassFingerprint(defs, shards)
	install := fingerprint != n.kvIndexInstalled
	if install {
		n.kvIndexInstalls.Add(1)
	}

	for group, s := range shards {
		if n.kvIndexStopped() {
			return // aborted: kvIndexInstalled is NOT advanced, so the next pass re-installs
		}
		if s == nil {
			continue // not hosted here (partitioned cluster)
		}
		idx := s.KVIndex()
		if idx == nil {
			continue // a store built without an index
		}
		// The fingerprint decides globally; this decides per group. A Set with the
		// wrong number of definitions installed has not seen this pass's set —
		// which is what a store that was just created looks like — so install into
		// it whatever the fingerprint said. It makes the correctness of the skip
		// independent of the fingerprint's ability to tell two stores apart, and
		// therefore independent of the allocator ever reusing an address.
		if install || len(idx.Defs()) != len(defs) {
			idx.Install(defs)
		}
		if installOnly {
			continue
		}
		for _, d := range defs {
			if n.kvIndexStopped() {
				return
			}
			if idx.IsReady(d.Name) {
				continue
			}
			if !n.backfillKVIndex(group, s, idx, d.Name) {
				// The group is being removed from this node. Nothing was published,
				// and the remaining definitions on it would abort the same way; the
				// next pass reads a hosted-shard set that no longer includes it.
				break
			}
		}
	}
	// Only after every hosted group has been visited: a pass that returned early
	// left some group uninstalled, and recording the fingerprint would tell the
	// next pass there was nothing to do.
	n.kvIndexInstalled = fingerprint
}

// kvIndexPassFingerprint identifies what an install depends on: the definitions
// (identity AND shape, so a redefinition under the same name is a change) and
// which shard groups this node hosts (so a group gained by a rebalance forces an
// install into its empty Set). Anything else moving in the meta log is not a
// reason to touch a single Set.
func kvIndexPassFingerprint(defs []kvindex.Def, shards []*shard.Store) string {
	var b strings.Builder
	for _, d := range defs {
		fmt.Fprintf(&b, "%s\x00%s\x00%s\x00%d\x00%d\x01", d.Name, d.Prefix, d.PathText, d.Kind, d.MetaIndex)
	}
	b.WriteByte('|')
	for i, s := range shards {
		if s != nil {
			// The STORE, not just the index it sits at. A group removed and
			// re-added at the same index is a DIFFERENT store with a brand-new,
			// empty index Set; keyed on the index alone the two passes fingerprint
			// identically, the install is skipped, and that Set never receives the
			// definitions — after which Backfill has no posting to fill and no-ops
			// forever. Fail-closed (the index reports not-ready, so it is never
			// queried from) but silent, which is the worst shape for a bug.
			fmt.Fprintf(&b, "%d:%p,", i, s)
		}
	}
	return b.String()
}

// kvIndexDefsFromCatalog turns the meta catalog into the Defs this build can
// actually post on, sorted by name so every node installs the same set in the
// same order.
//
// A definition the meta FSM ACCEPTED but this build cannot parse is counted and
// SKIPPED. wire.KVIndexDef.Validate is a shape check that cannot parse the
// payload path (sdk/record imports sdk/wire), so the full grammar is only
// enforced here, at install time — and a definition that fails it must not be
// installed as an index that posts nothing, which would answer queries with an
// empty set and call it exact.
//
// It maintains BOTH reject numbers, and they answer different questions.
// Rejects is a monotonic counter of reject EVENTS, so it never goes backwards
// and a rate can be taken from it. RejectedDefs is a gauge recomputed here on
// every pass: how many definitions the catalog holds right now that this node
// cannot build. The counter alone reads as a worsening problem while nothing
// changes (it climbs once per pass for as long as a bad definition sits there);
// the gauge alone loses the history. An operator watching for "is anything
// broken, and did it get fixed" reads the gauge.
func (n *Node) kvIndexDefsFromCatalog() []kvindex.Def {
	cat := n.meta.FSM.KVIndexes()
	defs := make([]kvindex.Def, 0, len(cat))
	var rejects uint64
	for name, e := range cat {
		d, err := kvindex.DefFrom(e.Def, e.MetaIndex)
		if err != nil {
			rejects++
			slog.Warn("kv index definition is in the meta catalog but this node cannot build it; it is NOT installed here and queries naming it will report no such index",
				"component", "cluster", "node", n.cfg.NodeID, "index", name, "path", e.Def.PayloadPath, "err", err)
			continue
		}
		defs = append(defs, d)
	}
	n.kvIndexRejects.Add(rejects)
	n.kvIndexRejectedDefs.Store(rejects)
	sort.Slice(defs, func(i, j int) bool { return defs[i].Name < defs[j].Name })
	return defs
}

// backfillKVIndex walks one group's whole keyspace through one definition and
// records the size of what it did.
//
// The walk is chunked (cache.IterateChunked releases each cache shard's read
// lock every few thousand index slots), so writers interleave with it instead of
// queueing behind a whole shard's walk. It is NOT free for writers, and the log
// line says so with a number: each chunk holds the read lock, so a writer waits
// at most one chunk.
// kvIndexAbortCheckEvery is how often the walk looks at its abort signal, in
// entries. A non-blocking channel receive is cheap but not free, and a walk
// visits every live key; checking on a power-of-two stride keeps it off the hot
// path while still bounding how long RemoveShardOwner waits to a few thousand
// entries rather than a whole keyspace.
const kvIndexAbortCheckEvery = 256

// It reports false when the walk did not finish — the shard is being removed
// from this node, so the remaining definitions on it are pointless too.
//
// THE WALK IS REGISTERED AGAINST THE SHARD GROUP, and that registration is what
// makes it safe. The walk aliases the store's live mmap, so the store must not
// close underneath it; RemoveShardOwner drains this registration before calling
// Store.Close, so the walk has already returned by the time anything is
// unmapped. Not hosting the walk in the gate at all (ok=false) means the group
// is already being removed and this walk must never start.
//
// An aborted walk publishes NOTHING: kvindex.Backfill grants readiness only when
// the walker returns nil, and the wrapper reports ErrWalkAborted when it stopped
// early. A half-read keyspace presented as an exact index answers queries with
// silently missing rows, which is the one failure this index may not have.
func (n *Node) backfillKVIndex(group int, s *shard.Store, idx *kvindex.Set, name string) bool {
	stop, done, ok := n.beginKVIndexWalk(group)
	if !ok {
		return false // the group is being removed; do not touch its store
	}
	defer done()

	var visited uint64
	walk := s.CacheWalker()
	began := time.Now()
	aborted := false
	err := idx.Backfill(name, func(yield func(key, value []byte) bool) error {
		werr := walk(func(key, value []byte) bool {
			visited++
			if visited%kvIndexAbortCheckEvery == 0 {
				select {
				case <-stop:
					aborted = true
					// Stop by RETURNING FALSE, which unwinds the cache's walk
					// through its own early-stop path (locks released, no state
					// left behind). Never by abandoning it.
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
			return kvindex.ErrWalkAborted
		}
		return nil
	})
	n.kvBackfillKeys.Add(visited)
	if err != nil {
		slog.Info("kv index backfill abandoned; nothing was published as ready",
			"component", "cluster", "node", n.cfg.NodeID, "shard", group, "index", name,
			"keys", visited, "err", err)
		return false
	}
	n.kvBackfills.Add(1)
	slog.Info("kv index backfill complete",
		"component", "cluster", "node", n.cfg.NodeID, "shard", group, "index", name,
		"keys", visited, "took", time.Since(began), "ready", idx.IsReady(name))
	return true
}

// kvIndexStats snapshots this node's KV index state for Stats().
//
// Definitions counts distinct names installed on this node; Ready counts the
// names that are ready on EVERY hosted group, which is the only readiness that
// means anything to a query — a group still building answers from a proper
// subset, so one lagging group makes the whole index not ready here.
func (n *Node) kvIndexStats() KVIndexStats {
	st := KVIndexStats{
		Backfills:      n.kvBackfills.Load(),
		BackfillKeys:   n.kvBackfillKeys.Load(),
		Rejects:        n.kvIndexRejects.Load(),
		RejectedDefs:   int(n.kvIndexRejectedDefs.Load()), //nolint:gosec // bounded by KVIndexMaxDefs
		VerifyMisses:   n.kvIndexVerifyMisses.Load(),
		ReconcileDrops: n.kvIndexReconcileDrops.Load(),
	}
	readyOn := make(map[string]int)
	hosted := 0
	for _, s := range n.snapshotShards() {
		if s == nil {
			continue
		}
		idx := s.KVIndex()
		if idx == nil {
			continue
		}
		hosted++
		for _, d := range idx.Defs() {
			if _, seen := readyOn[d.Name]; !seen {
				readyOn[d.Name] = 0
			}
			if idx.IsReady(d.Name) {
				readyOn[d.Name]++
			}
		}
	}
	st.Definitions = len(readyOn)
	if hosted == 0 {
		return st
	}
	for _, groups := range readyOn {
		if groups == hosted {
			st.Ready++
		}
	}
	return st
}
