// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// BuildConcurrent bulk-loads an EMPTY index from ids/vecs using `workers`
// goroutines, exploiting multiple cores for the otherwise single-threaded HNSW
// build. It produces a graph equivalent (within the usual non-determinism of
// parallel edge selection) to inserting the same vectors serially.
//
// A pure VECTOR bulk load — see BuildConcurrentMeta for the payload-bearing form,
// which is where the constraints and the mechanism are documented.
func (h *hnsw) BuildConcurrent(ids []uint64, vecs [][]float32, workers int) error {
	return h.BuildConcurrentMeta(ids, vecs, nil, workers)
}

// BuildConcurrentMeta is BuildConcurrent with an OPTIONAL per-point payload:
// metas is either nil/empty (a pure vector load, byte-identical to
// BuildConcurrent) or exactly len(ids) long, with a nil entry for each point that
// carries no payload.
//
// PRECONDITIONS. The index must be EMPTY ON ENTRY — no vectors, no graph, and
// nothing in the payload index — and entries carry no TTL and no sparse vector.
// ids should be unique. Works with heap or mmap (QuantMmap) storage: the mmap
// region is pre-reserved to N up front so the parallel phase only reads it. After
// it returns, the index is an ordinary serial index: Search/Insert/Delete/Snapshot
// all work normally.
//
// Mechanism: the arena is pre-reserved to N (no realloc → stable Vec slices),
// all nodes are created single-threaded (levels assigned up front), then workers
// link nodes in parallel. Each node's neighbor list is guarded by a per-slot
// lock (h.linkLocks); the serial search/insert paths check `linkLocks == nil`
// and pay nothing once the build is done.
//
// WHERE THE PAYLOADS GO. Into the SINGLE-THREADED placement loop, alongside the
// vector — not into a pass of their own. The loop already visits every slot
// exactly once with the write lock held and the graph not yet built, which is the
// only moment at which the payload index, the BM25 corpus and the arena can be
// filled with no locking, no id→slot lookup and no window in which the collection
// is searchable but not yet filterable. A separate "apply payloads after the
// build" op would have all three, and the third is the dangerous one: a filtered
// search landing between build and apply returns a WRONG (empty-ish) answer
// rather than an error.
func (h *hnsw) BuildConcurrentMeta(ids []uint64, vecs [][]float32, metas []Metadata, workers int) error {
	if err := checkRecordValuesAll(metas); err != nil {
		return err
	}
	if len(metas) != 0 && len(metas) != len(ids) {
		return ErrBuildMetaLenMismatch
	}
	// IndexVamana: the single-layer two-pass α-RobustPrune build (medoid entry
	// point, randomized order, α=1 then α=VamanaAlpha). Shares the placement/arena
	// + concurrent linkOneNode machinery; differs only in the entry point, the order,
	// and the two-pass α schedule. HNSW/IVF fall through to the multi-level build.
	if h.vamana {
		return h.buildVamana(ids, vecs, metas, workers)
	}
	start := time.Now()
	defer func() { h.insertLat.observe(time.Since(start)) }()

	if len(ids) != len(vecs) {
		return ErrBuildLenMismatch
	}
	for _, v := range vecs {
		if len(v) != h.cfg.Dim {
			return ErrDimMismatch
		}
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// The payload index is part of "empty", not an afterthought, and it is checked
	// HERE — before a single slot is written — because everything the index derives
	// (posting counts, the numeric column sidecar) is computed against a slot space
	// this build is about to claim from zero. A pre-populated index would leave
	// columns holding pre-build values for slots the build just reused, which is
	// the one way this file could produce a WRONG ROW rather than a slow one.
	//
	// NOTE ON THE RESTATEMENT. This check used to be justified by "BuildConcurrent
	// takes no metadata, so there is nothing to index" — the placement loop wrote
	// slots without calling payloadIdx.reindex at all. That is no longer true:
	// metas is indexed BY the placement loop (applyBulkMeta, below). The
	// precondition is unchanged and is NOT weaker — the index must still be empty
	// when this function is entered — but its justification is now the ordinary
	// one: a bulk build owns the whole slot space, so it must start from nothing.
	// Payload state created by this call is not a violation of the precondition,
	// it is this call's own output.
	if h.arena.Size() != 0 || h.maxLevel >= 0 || !h.payloadIdx.isEmpty() {
		return ErrBuildNonEmpty
	}
	n := len(ids)
	if n == 0 {
		return nil
	}
	// Quota, before a single byte is reserved. See bulkQuotaErr: this is
	// insertLocked's own bound applied once to the whole load, and it is checked
	// AFTER ErrBuildNonEmpty so a caller that violates the precondition still
	// learns that rather than being told the collection is full.
	if err := bulkQuotaErr(h.cfg, n); err != nil {
		h.quotaRejects.Add(1)
		return err
	}
	if err := h.arena.Reserve(n); err != nil {
		return err
	}

	// ---- single-threaded setup: place every vector and node up front ----
	// Pre-size the node table and level-0 slab to N so the parallel phase only
	// reads/writes existing storage (no reallocation race). presizeGraphSlab
	// handles both heap and mmap backing.
	h.nodes = make([]*node, n) // dense slots 0..n-1
	if err := h.presizeGraphSlab(n); err != nil {
		return err
	}
	levels := make([]int, n)
	// PQ-HNSW: collect the placed (normalized-for-cosine) vectors as the codebook
	// training sample. trainPQ runs below, after placement and BEFORE the link
	// phase, so the link phase already has codes (it still navigates on exact
	// floats — see buildScorer/quantizedBuild). nil for every non-PQ build.
	var pqSample [][]float32
	if _, isPQ := h.quant.(*pqQuantizer); isPQ {
		pqSample = make([][]float32, n)
	}
	// HNSW-SQ: collect the placed vectors as the trained-SQ range sample, exactly
	// as the PQ path collects its codebook sample. trainAndEncodeSQ runs below
	// (after placement, before link), so the link phase already has SQ codes. nil
	// for every non-SQ build (the fixed-scale sq8/bq1/none need no sample).
	var sqSample [][]float32
	if _, isSQ := h.quant.(*trainedSQ); isSQ {
		sqSample = make([][]float32, n)
	}
	// HNSW-PRQ: collect the placed vectors as the L-layer residual-codebook training
	// sample, exactly as the PQ path collects its codebook sample. trainAndEncodePRQ
	// runs below (after placement, before link), so the link phase already has PRQ
	// codes. nil for every non-PRQ build.
	var prqSample [][]float32
	if _, isPRQ := h.quant.(*prqQuantizer); isPRQ {
		prqSample = make([][]float32, n)
	}
	for i := range ids {
		slot := uint32(i) //nolint:gosec // i < n, bounded by 2^32 vectors per arena
		v := vecs[i]
		if h.cfg.Metric == Cosine {
			v = append([]float32(nil), v...)
			normalize(v)
		}
		h.arena.PutAt(slot, ids[i], v) // codes left zero for untrained PQ (Encode no-op)
		h.arena.idMap[ids[i]] = slot
		// len(metas), not metas != nil: the length is what the precondition above
		// validated, and an EMPTY non-nil slice is a legal "no payloads" — indexing
		// it would panic rather than fall through to the vectors-only path.
		if len(metas) != 0 {
			h.applyBulkMeta(slot, metas[i])
		}
		if pqSample != nil {
			pqSample[i] = v
		}
		if sqSample != nil {
			sqSample[i] = v
		}
		if prqSample != nil {
			prqSample[i] = v
		}
		lvl := h.assignLevel()
		levels[i] = lvl
		nd := &node{slot: slot, level: lvl, upper: makeUpper(lvl)}
		// PLACED-BUT-NOT-YET-LINKED, AND SAY SO. Every node is placed here, up
		// front, and only linked later by a worker — so for most of the build
		// h.nodes is full of nodes carrying their final level and no edges. That
		// is EXACTLY the Option B window node.unlinked exists for, and this path
		// used to leave the flag false, so the bulk build ran the whole parallel
		// phase with that guard disarmed.
		//
		// It matters here for a reason peculiar to a MULTI-LEVEL node: linkNode
		// writes its levels top-down, and each level's forward write is followed
		// immediately by its back-edges, so the node becomes REACHABLE at level
		// lc while its level lc-1..0 lists are still empty. Another worker's
		// width-1 descent can then land on it at an upper level, carry it down as
		// the sole level-0 frontier, and expand a node with zero level-0 edges —
		// a dead end. The level-0 searchLayer returns that one node, the new node
		// links to it alone, and its single back-edge is later pruned away when
		// the frontier node writes its own 2M level-0 edges. In-degree zero,
		// forever.
		//
		// Measured at n=20,000 / M=16 / EfConstruction=200 with 8 workers: EVERY
		// orphan had ncand=1, a one-element level-0 frontier whose degree at
		// level 0 was 0 at that instant, and that frontier node was always a
		// level-1-or-2 node that had already published its upper-level edges.
		//
		// SLOT 0 IS THE EXCEPTION AND MUST STAY UNFLAGGED: it is the seed entry
		// point, it is never handed to a worker (indices start at 1), so a flag
		// set here would never be cleared — and a flagged entry point expands to
		// nothing and is not admitted, which starves every candidate set in the
		// build and orphans the whole index. Its empty edge list at the very
		// start is the ordinary HNSW bootstrap, the same one the serial path
		// runs: the first nodes link to it, their back-edges fill it, and it is
		// well-connected long before its degree cap makes a back-edge droppable.
		//
		// WHY FLAGGING CANNOT STARVE A CANDIDATE SET. `cur` only ever holds
		// nodes that a searchLayer ADMITTED (or the entry point, which is slot 0
		// or a node that cleared its flag before promoting itself), and the flag
		// only ever goes true→false. So every frontier node is fully linked and
		// carries a complete list at the level about to be searched, and the
		// entry point is always admitted — `cand` is never empty, and the
		// structuralLayer fallback below stays dormant on this path. What the
		// flag removes is exactly the PARTIAL node, which is the only thing that
		// was ever a dead end.
		//
		// buildVamana deliberately does NOT do this. It is single-layer, so a
		// node has no upper levels to publish ahead of level 0 and the mechanism
		// cannot arise (measured: 9 runs at 20k, workers 1/8/16, 0 orphans). It
		// would also be actively wrong there: its entry point is the medoid,
		// which IS linked by a worker, so flagging would empty the candidate set
		// during the medoid's own link window.
		if slot != 0 {
			nd.unlinked.Store(true)
		}
		h.nodes[slot] = nd
	}
	// Train the PQ codebooks on the placed vectors, swap the trained codec into the
	// quantizer, and encode every slot's code into the arena. After this the codec
	// is trained, so search navigates on ADC + exact-rescores; the graph below is
	// still LINKED on exact float32 distances (quantizedBuild() is false for PQ).
	if pqSample != nil {
		if err := h.trainAndEncodePQ(pqSample, workers); err != nil {
			return err
		}
	}
	// HNSW-SQ: learn the per-dimension ranges on the placed vectors, swap them in,
	// and encode every slot. After this trained-SQ.trained() is true, so the link
	// phase below navigates on SQ codes when QuantizedBuild is set (else exact
	// floats) and search rescores. nil sqSample for every non-SQ build (no-op).
	if sqSample != nil {
		if err := h.trainAndEncodeSQ(sqSample); err != nil {
			return err
		}
	}
	// HNSW-PRQ: train the L-layer residual codebooks on the placed vectors, swap the
	// trained codec in, and encode every slot. After this prqQuantizer.trained() is
	// true, so the link phase navigates on exact floats (quantizedBuild() is false
	// for PRQ, like PQ) and search uses the summed-LUT ADC + exact rescore. nil
	// prqSample for every non-PRQ build (no-op).
	if prqSample != nil {
		if err := h.trainAndEncodePRQ(prqSample, workers); err != nil {
			return err
		}
	}
	h.bytesUsed += estimateInsertBytes(h.cfg.Dim, h.cfg.M) * int64(n)
	h.insertOps.Add(uint64(n)) //nolint:gosec // n >= 0
	h.idSetVersion++           // bulk-populated the id set: invalidate any scroll snapshot
	h.bumpData()               // bulk id-set change also invalidates the order_by snapshot
	// Seed: node 0 is the initial entry point; workers promote a taller node.
	h.entryPoint = 0
	h.maxLevel = levels[0]

	// ---- concurrent graph phase ----
	// Gate ON for the duration of the build: the workers below mutate neighbor
	// lists in parallel, so every neighbor READ (theirs, via neighborsAt) has to
	// go through the link stripes. Raised here under the write lock this whole
	// function holds, which is exactly the publication rule the gate relies on.
	h.ensureLinkStripes()
	h.linkers.Add(1)
	defer h.linkers.Add(-1) // gate OFF: index reverts to unsynchronized reads

	var next atomic.Int64 // hands out indices 1..n-1 (index 0 is the seed)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := getLayerScratch()
			defer layerScratchPool.Put(s)
			for {
				i := int(next.Add(1))
				if i >= n {
					return
				}
				h.linkOneNode(s, uint32(i), levels[i]) //nolint:gosec // i < n
			}
		}()
	}
	wg.Wait()

	// PQDropVecs: the graph is now fully LINKED on EXACT floats (the link phase
	// above read arena.Vec), the PQ codebooks are trained, and every slot's code is
	// in the arena. Release the resident float32 vectors — only the M-byte codes
	// stay in RAM (maximum compression). Gated on a TRAINED PQ codec (pqSample !=
	// nil only for a pqQuantizer, and trainAndEncodePQ populated it), so vecsDropped
	// is never set for a non-PQ index. After this, search is ADC-only (rescore is
	// skipped), the exact paths reconstruct via vecFor, and Insert returns
	// ErrPQDropVecsReadOnly. Mirror ivf.go's drop-after-train. n>0 here (returned
	// early above for n==0). PQDropVecs=false ⇒ this is a no-op (byte-identical).
	if h.cfg.PQDropVecs && !h.pqUntrained() {
		h.arena.dropVecs()
	}
	return nil
}

// applyBulkMeta attaches one bulk-placed slot's payload: the stored metadata, its
// payload-index postings, and — when the payload carries the reserved $content
// key — its BM25 postings. It is the bulk analogue of the metadata half of
// placeLockedAt, and is deliberately written to do the SAME THREE THINGS IN THE
// SAME ORDER, because the equivalence this whole path is judged by is "a
// payload-bearing bulk load produces the state an inline load would have".
//
// Caller must hold h.mu (write) and must already have placed the slot. Only ever
// called from the single-threaded placement pass of a bulk build, which is why
// there is no drop-the-previous-occupant step: the index is empty on entry, so
// every slot is fresh and reindex has nothing to remove. (The inline path calls
// reindex unconditionally precisely because ITS slots can be reused; that is the
// difference between the two, and the only one.)
//
// Sparse vectors and TTLs are NOT handled here: the staging wire does not carry
// them, and a bulk build's precondition still excludes them. If that ever
// changes, this is the function that has to grow, not the caller.
func (h *hnsw) applyBulkMeta(slot uint32, meta Metadata) {
	if len(meta) == 0 {
		return
	}
	h.arena.SetMetadata(slot, meta)
	h.payloadIdx.reindex(slot, meta)
	if h.bm25 != nil {
		// Mirrors placeLockedAt: the reserved $content field rides the metadata map,
		// so a payload that carries it enters the BM25 corpus exactly as an inline
		// insert's would. Missing/empty content adds no postings.
		if content := contentOf(meta); content != "" {
			h.bm25.add(slot, h.analyzeCounts(content))
		}
	}
}

// linkOneNode wires the already-placed node at slot into the graph, reading its
// stored vector from the arena and its clock from the wall — the shape the bulk
// builders (BuildConcurrent, buildVamana) want, where there is no per-op stamp
// to thread. The incremental Insert path calls linkNode directly so it can pass
// the op's single clock snapshot.
func (h *hnsw) linkOneNode(s *layerScratch, slot uint32, level int) {
	// One wall-clock snapshot for this node's link traversals (see the search path):
	// the admission gate is loop-invariant in time.
	h.linkNode(s, h.nodes[slot], h.arena.Vec(slot), level, uint64(h.now()))
}

// linkNode is the ONE copy of the HNSW link phase, shared by the incremental
// Insert path and both bulk builders so they cannot drift into constructing
// different graphs. It wires an already-PLACED node (its arena slot, its entry
// in h.nodes, and its level are all committed) into the graph: greedy descent to
// its top level, then per-level neighbor selection plus bidirectional
// back-edges.
//
// LOCKING. The caller must hold h.mu — read OR write — for the whole call, and
// must have raised h.linkers under the WRITE lock beforehand (see
// link_stripes.go). Every mutation below is confined to the target node's or a
// neighbor's link stripe, and the entry-point promotion to globalMu, so this is
// safe to run concurrently with queries and with other linkers.
//
// `stored` is the node's vector, already cosine-normalized if the metric
// requires it; `now` is the op's single clock snapshot, which the admission gate
// during the build traversal reads — threading it (rather than re-reading the
// wall clock here) is what keeps a replicated insert's EDGES deterministic
// across replicas when an expiring point is in flight.
func (h *hnsw) linkNode(s *layerScratch, nd *node, stored []float32, level int, now uint64) {
	score := h.buildScorer(s, stored)
	ep, maxL := h.entryState()
	slot := nd.slot

	// Greedy descent from the entry point down to one above this node's level.
	// linkLayer, not searchLayer: a node reached from its OWN inherited in-edges
	// (the upsert/reused-slot case) is at distance 0 and would otherwise collapse
	// the width-1 frontier onto itself. See linkLayer.
	cur := append(s.cur[:0], ep)
	for lc := maxL; lc > level; lc-- {
		near := h.linkLayer(s, score, cur, 1, lc, now, slot)
		if len(near) > 0 {
			cur = append(cur[:0], near[0].slot)
		}
	}

	connectTop := level
	if maxL < connectTop {
		connectTop = maxL
	}
	for lc := connectTop; lc >= 0; lc-- {
		cand := h.linkLayer(s, score, cur, h.cfg.EfConstruction, lc, now, slot)
		// AN EMPTY CANDIDATE SET IS NOT "NO NEIGHBOURS", IT IS AN ORPHAN.
		//
		// searchLayer admits only LIVE points, and it has no early exit while its
		// result set is empty — so it walks the ENTIRE component reachable from the
		// frontier before returning nothing. Empty therefore means something exact:
		// every node reachable from the entry point at this level is tombstoned,
		// TTL-expired, or still inside its own link window.
		//
		// The old behaviour was to accept that and write an empty forward list. The
		// node then has no out-edges AND — since back-edges are only added to the
		// neighbours just chosen — no in-edges, so it is unreachable from the entry
		// point forever: present in the arena, byte-correct, returned by Get and by
		// scans, and permanently absent from vector search. Worse, it clears
		// node.unlinked on the way out and so passes every guard that exists for
		// exactly this shape (the placement-window flag, the stale-edge marker),
		// and if it drew a taller level it promotes ITSELF to entry point — an
		// edgeless entry point, which orphans the whole index at a stroke.
		//
		// The reshard produces this precisely: the copy backfills ids in ascending
		// order while the delete worker tombstones the low band it has just copied,
		// so there is a window in which every live-reachable point is tombstoned.
		// Measured on the boundary repro, a link writing zero level-0 edges happened
		// in 3 of 3 failing runs (1, 1 and 6 times) and in 0 of 797 clean ones.
		//
		// So fall back to the STRUCTURAL candidate set — the same traversal without
		// the liveness half of the gate. Tombstoned vertices are still vertices, and
		// linking to them is what canonical lazy-deletion HNSW does; Reclaim prunes
		// those edges when it finally removes them, by which point the ordinary
		// back-edges from later inserts have arrived. This fires ONLY where the
		// alternative is a guaranteed orphan, so no run that has any live candidate
		// changes shape and recall is untouched.
		if len(cand) == 0 {
			cand = h.structuralLayer(s, score, cur, h.cfg.EfConstruction, lc, slot)
		}
		neighbors := h.pickNeighbors(s, stored, cand, h.forwardM(lc), lc, slot)
		maxM := h.cfg.M
		if lc == 0 {
			maxM = h.maxM0()
		}

		// Write this node's own forward edges under its own lock, PRESERVING any
		// back-edges other workers appended before this write. A multi-level node
		// that has finished linking level lc+1 is already reachable there (its
		// back-edges exist), so another worker can carry it down in its candidate
		// frontier and use it as a DIRECT searchLayer entry point at lc — where
		// this node has not yet written its forward list — and append a back-edge
		// to it. Blindly overwriting would destroy that edge (the appender's
		// in-edge), which is how the concurrent build orphaned nodes the serial
		// build never does. Re-applying the pre-existing entries through
		// addBackEdge after the forward write reproduces the serial order "this
		// node linked first, the appender arrived after" exactly (append, prune
		// via the heuristic when full).
		//
		// Under the Vamana two-pass build the reset IS intentional: pass 2 re-runs
		// over nodes that already carry pass-1 edges and RobustPrune recomputes
		// Nout(p) from scratch each pass, so the merge is gated off (and the
		// carry-down channel needs >= 2 levels, which single-layer Vamana lacks).
		// The slot's link stripe serializes the whole read-write-merge against
		// concurrent back-edge appends, so there is no data race.
		//
		// On the incremental path `pre` has exactly two sources, and level 0 is
		// where they meet: a raced-in back-edge, and — on a REUSED slot — the
		// predecessor's own edge list, which placement deliberately leaves in the
		// slab (see node.staleLevel0). Both are handled the same way, for the same
		// reason: whatever is already in this list is somebody's live in-edge, and
		// the forward write is about to overwrite the region it sits in. Above
		// level 0 only the raced-in source exists, since node.upper is freshly
		// allocated per node.
		lk := h.stripe(slot)
		lk.Lock()
		// Preserve any raced-in back-edges in the per-worker scratch (reused across
		// writes — no per-write allocation; level 0 aliases the slab, so the append
		// copy is required before writeNbrsFromDist overwrites it). pre is nil on the
		// common (uncontended) path, so the replay loop is a no-op.
		var pre []uint32
		// staleLevel0: this slot's level-0 edges belong to the PREDECESSOR that
		// occupied it, kept readable so the slot stays traversable during the
		// placement/link window (see placeLockedAt). They are MERGED into this
		// node's list — offered to the ordinary diversity heuristic, judged against
		// THIS node's vector — rather than discarded wholesale.
		//
		// AN UPSERT MUST REVISE A SLOT'S ADJACENCY, NOT REPLACE IT. This forward
		// write is the only place in the index where a node's level-0 out-edges are
		// destroyed en bloc; every other write appends and lets the diversity
		// heuristic decide what to drop. Discarding them is invisible for the
		// node's own reachability (it is about to get a full set of fresh picks)
		// and destructive for its NEIGHBOURS': every discarded edge is an in-edge
		// somebody else just lost. A full upsert sweep demolishes the entire
		// level-0 edge population exactly once, so each point is left with only the
		// back-edges it happens to be granted, and at low degree that is not enough
		// — points fall out of the reachable graph. Measured by ABLATION — this
		// same code with the merge declined, which is the one-line change of
		// `stale` to a constant false — dim 8, L2, EfConstruction=200, three seeds
		// x three successive sweeps, range of points UNREACHABLE from the entry
		// point:
		//
		//	     n=192                    n=1000
		//	M    declined    merged       declined    merged
		//	2    5-17        0-2          68-97       8-13
		//	3    1-5         0             9-27       0-2
		//	4    0-3         0             2-8        0
		//	5    0-2         0             0-4        0
		//	6    0           0             0-2        0
		//
		// Nothing is kept on sentiment: a stale edge survives only if reselect
		// prefers it to the alternatives at the NEW vector, so an upsert that moves
		// the point far away drops them all, which recovers the discard behaviour
		// exactly where the discard was right.
		//
		// Vamana keeps the reset: its pass 2 re-runs RobustPrune over nodes that
		// already carry pass-1 edges and recomputes Nout(p) from scratch, which is
		// the same reason the raced-back-edge merge below is gated off for it.
		//
		// ONE SELECTION, NOT A REPLAY. Offering the inherited edges to addBackEdge
		// one at a time — the way a raced-in back-edge has to be offered, since that
		// path has to reproduce "the appender arrived after" — costs a full reselect
		// PER EDGE once the list is at its cap. Re-selecting once over the union is
		// a single prune of the same candidates. The replay stays for the raced
		// case, where the ordering is the point.
		//
		// WHAT THIS COSTS, MEASURED, because it is not free on either path.
		// n=4000, dim 64, L2, EfConstruction=200, EfSearch=128, one full upsert
		// sweep. Upsert ns/op is the median of 3 runs; everything else is exact.
		// "discard" is the previous behaviour, "union" is this code:
		//
		//	          upsert ns/op   query    recall  mean level-0   unreachable
		//	          discard union  us/query  @10    out-deg (cap)   points
		//	M=8       161k -> 186k   86 -> 104 .880 -> .909  11.16 -> 16.00 (16)  44 -> 2
		//	M=16      226k -> 302k  118 -> 123 .978 -> .988  22.87 -> 32.00 (32)   2 -> 0
		//	M=32      475k -> 780k  145 -> 158 .999 -> 1.00  45.08 -> 64.00 (64)   0 -> 0
		//
		// (The one-at-a-time replay, for reference, was 198k/347k/975k, so batching
		// the selection takes back about a third of the added upsert cost.)
		//
		// THE OUT-DEGREE COLUMN IS THE WHOLE STORY, and it is why the remaining cost
		// is the right trade rather than a regression to argue about. The discard
		// was cheap because it was THINNING the graph: a swept index ended below the
		// out-degree its own FRESH build had (11.16 against 12.00 at M=8) and kept
		// falling with every sweep, so an index under churn slowly became a
		// different, sparser index than the one the user configured. Keeping the
		// edges holds it at its configured cap, and a graph at full degree costs
		// what full degree costs — on both paths. What it buys is the reachability
		// this change exists for, plus recall that moves the right way at every M.
		// A deployment that wants the old latency can ask for the old graph by
		// lowering M, which is the knob for exactly that; it could not previously
		// ask for a degree that churn would not erode.
		stale := lc == 0 && nd.staleLevel0
		if stale {
			nd.staleLevel0 = false
		}
		if !h.vamana && h.nbrLen(nd, lc) > 0 {
			s.preScratch = append(s.preScratch[:0], h.nbrsAt(nd, lc)...)
			pre = s.preScratch
		}
		switch {
		case stale && len(pre) > 0:
			// Union of the fresh picks and the inherited edges, deduped, minus the
			// node itself (a self-loop inherited from a graph built before linkLayer
			// existed — a restored snapshot — must not be re-admitted: at distance 0
			// it survives every subsequent prune, see linkLayer). reselect then
			// applies the ordinary heuristic at the level cap, so an inherited edge
			// survives only if it is still a good neighbour of the NEW vector.
			u := s.pruneSlots[:0]
			for _, n := range neighbors {
				if n.slot != slot {
					u = append(u, n.slot)
				}
			}
			for _, p := range pre {
				if p != slot && !slices.Contains(u, p) {
					u = append(u, p)
				}
			}
			s.pruneSlots = u
			// `u` is a private copy, which reselect requires at level 0 (its
			// writeNbrs overwrites the slab region the live list aliases).
			h.reselect(s, nd, lc, u, maxM)
		default:
			h.writeNbrsFromDist(nd, lc, neighbors)
			for _, p := range pre {
				if p != slot && !slices.Contains(h.nbrsAt(nd, lc), p) {
					h.addBackEdge(s, nd, lc, p, maxM)
				}
			}
		}
		lk.Unlock()

		// Bidirectional back-edges: one neighbor lock at a time (never nested),
		// so no deadlock is possible.
		for _, nb := range neighbors {
			other := h.nodeAt(nb.slot)
			if other == nil || lc > other.level {
				continue
			}
			olk := h.stripe(other.slot)
			olk.Lock()
			h.addBackEdge(s, other, lc, slot, maxM)
			olk.Unlock()
		}

		// Next level down starts from this level's candidate frontier.
		cur = cur[:0]
		for _, c := range cand {
			cur = append(cur, c.slot)
		}
	}

	// The node now has its edges, so it is part of the graph: clear the flag that
	// hid it from traversal and from the entry-point elections (see node.unlinked;
	// a no-op for the bulk builders, which never set it).
	//
	// THIS MUST PRECEDE THE PROMOTION BELOW. Clearing it afterwards — or, as it
	// was, back in linkRead once linkNode had returned — leaves a window in which
	// the node is the ENTRY POINT while still flagged, and a flagged entry point
	// is worse than a flagged interior node: every traversal starts there, cannot
	// expand it, and gets nothing. A query returns empty; a concurrent linker
	// descends from it, finds no candidates at any level, and writes a node with
	// ZERO edges. That is the same orphan the flag exists to prevent, reintroduced
	// at the one node through which every traversal passes.
	nd.unlinked.Store(false)

	if level > maxL {
		h.globalMu.Lock()
		if level > h.maxLevel {
			h.entryPoint = slot
			h.maxLevel = level
		}
		h.globalMu.Unlock()
	}
}
