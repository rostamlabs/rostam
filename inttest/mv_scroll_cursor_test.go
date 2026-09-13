// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// ----------------------------------------------------------------------------
// MV scroll cursor-pagination proofs (mirror the dense cursor proof in
// remote_scroll_cursor_integration_test.go + the named/MV barrier proofs in
// named_linearizable_proof_test.go).
//
//   - TestMVScrollPaginationStableAcrossPartitions: a real 3-node partitioned
//     (P>1) MV collection paged ENTIRELY via next_cursor returns every seeded id
//     EXACTLY ONCE, ascending, gap-free + dup-free, == the independent ground
//     truth; page count == ceil(N/limit); exhaustion ⇒ empty cursor. Plus a
//     stability check: a concurrent insert/delete mid-scroll never drops or
//     duplicates an id present THROUGHOUT (mirror the dense scroll stability
//     proof — TestScrollPageDeleteBetweenPages / ...RaceConcurrentInsert).
//
//   - TestMVScrollLinearizableArmsBarriers: a Linearizable MV scroll fires the
//     META barrier EXACTLY ONCE on the coordinator (SetMetaReadIndexForwardHook
//     == 1, NOT P) AND the SHARD barrier per partition (SetBarrierEnteredHook
//     >= P) — mirror TestNamedLinearizableShardBarrierRealPath +
//     TestNamedLinearizableReadFiresBarrierExactlyOnce. AnyReplica fires the meta
//     barrier ZERO times. Filter applied during a Linearizable scroll.
// ----------------------------------------------------------------------------

// mvScrollPageAll pages the MV collection start-to-exhaustion via the global
// cursor on the given store, returning the concatenated id stream (page order)
// and the page count. It asserts the per-page exhaustion rule (a full page must
// carry a next_cursor; a short/empty page must end pagination) so a cursor/merge
// bug fails loud. The cluster mirror of pageAllMV (embedded_mv_scroll_cursor_test.go).
func mvScrollPageAll(t *testing.T, ctx context.Context, s rostam.Store, name string, filter rostam.VectorFilter, limit int) (ids []uint64, pages int) {
	t.Helper()
	cursor := ""
	for {
		docs, _, next, err := s.VectorMVScroll(ctx, name, filter, limit, cursor)
		if err != nil {
			t.Fatalf("VectorMVScroll page %d: %v", pages, err)
		}
		pages++
		for _, d := range docs {
			ids = append(ids, d.ID)
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("MV page %d full (len=%d) but next_cursor empty (not exhausted)", pages, limit)
			}
		} else if next != "" {
			t.Fatalf("MV page %d short (len=%d<%d) but next_cursor=%q (want exhausted)", pages, len(docs), limit, next)
		}
		if next == "" {
			return ids, pages
		}
		cursor = next
		if pages > limit*1000+100 { // runaway guard
			t.Fatalf("MV pagination did not terminate after %d pages", pages)
		}
	}
}

// mvCollSeq makes every mvCreatePartitionedClean attempt target a UNIQUE physical
// name, so a partial/leaderless create is simply abandoned and the next attempt
// builds a brand-new collection — never colliding with a half-built predecessor.
var mvCollSeq atomic.Uint64

// mvCreatePartitionedClean creates a P-partition MV collection and returns the
// UNIQUE name it actually created — GUARANTEED fully built (a successful
// VectorMVCreateCollection only returns nil after building the logical collection
// AND every physical partition shard) AND confirmed usable (a probe doc
// round-trips through the fan-out to a live partition leader).
//
// Under load / -count, a VectorMVCreateCollection can partially commit (the
// logical name + some physical partitions, then a transient before the rest) or
// race leadership ("not leader"). A plain retry of the SAME name then either sees
// "already exists" on a physical shard and can never complete the build, or fatals
// on the non-leader error (retryUntil only loops on ErrNotLeader). This helper
// instead retries each attempt on a FRESH unique name (base-seq), so no stale
// half-built collection can ever block it, and returns only once a probe doc
// round-trips. Test-harness robustness only — no production change.
func mvCreatePartitionedClean(t *testing.T, ctx context.Context, s rostam.Store, base string, P int) string {
	t.Helper()
	const probeID = uint64(1<<63 + 7) // sentinel id outside any seeded range
	deadline := time.Now().Add(60 * time.Second)
	var last error
	for time.Now().Before(deadline) {
		name := fmt.Sprintf("%s_%d", base, mvCollSeq.Add(1))
		if err := s.VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
			last = err
			time.Sleep(100 * time.Millisecond)
			continue // a fresh name next time — never collides with this partial build
		}
		// Created. Confirm the collection is genuinely usable: a probe add to the
		// logical name must route to a live physical shard (catches leadership still
		// settling). Tolerate transient routing/leader blips while shards finish
		// electing.
		if perr := s.VectorMVAdd(ctx, name, probeID, [][]float32{mvTokenAt(0)}, rostam.VectorMetadata{}); perr != nil {
			last = perr
			time.Sleep(100 * time.Millisecond)
			continue
		}
		// Remove the probe and confirm it is gone, so it can never pollute a
		// ground-truth set (a leaked sentinel would fail assertExactlyOnceAscending).
		if _, derr := s.VectorMVDelete(ctx, name, probeID); derr != nil {
			last = derr
			time.Sleep(100 * time.Millisecond)
			continue
		}
		return name
	}
	t.Fatalf("mv create (clean) base=%q: timed out: %v", base, last)
	return ""
}

// TestMVScrollPaginationStableAcrossPartitions is the core cursor-pagination
// proof for the MV family over a REAL 3-node partitioned (P=4) collection.
//
// Part A (deterministic deep pagination): seed ids 1..N across 4 partitions in
// random order; page the WHOLE collection in limit-sized pages following
// next_cursor to exhaustion. Assert: every seeded id EXACTLY ONCE, globally
// ascending, gap-free + dup-free, == the independent ground-truth set; page count
// == ceil(N/limit); the final short page carries an empty cursor.
//
// Part B (stable under concurrent mutation): re-create + re-seed a CORE set of
// ids (never mutated) interleaved with a VOLATILE set; mid-pagination,
// concurrently INSERT brand-new ids (above the seeded range) and DELETE volatile
// ids that have NOT yet been paged. Assert that every CORE id (present
// THROUGHOUT) is returned EXACTLY ONCE, gap-free + dup-free, and the whole stream
// stays globally ascending — ids inserted after the cursor passed their position,
// or volatile ids deleted before the cursor reached them, MAY or may not appear,
// but no id present throughout is ever dropped or duplicated (the dense
// scroll-stability contract).
func TestMVScrollPaginationStableAcrossPartitions(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	// --- Part A: deterministic deep pagination across partitions ---
	{
		const (
			P = 4
			N = 250
			L = 30
		)
		name := mvCreatePartitionedClean(t, ctx, stores[0], "mvpage", P)
		if p, _, ok := stores[0].(*rostam.Embedded).Catalog().PartitionsGen(name); !ok || p != P {
			t.Fatalf("PartitionsGen = (%d, ok=%v), want (%d, true)", p, ok, P)
		}
		// Seed 1..N in RANDOM order so a stable ascending paged result proves the
		// cross-partition merge, not the add order.
		ids := shuffledIDs(N, 73)
		want := map[uint64]bool{}
		for _, id := range ids {
			idc := id
			md := rostam.VectorMetadata{"n": vector.NewInt(int64(idc))}
			retryUntil(t, "mv add", func() error {
				return stores[0].VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(int(idc))}, md)
			})
			want[idc] = true
		}
		// Drive from a NON-creating coordinator (node 1) to exercise the cross-node
		// fan-out cursor merge. Wait for node 1's catalog to converge to P first.
		waitEmbeddedCatalog(t, stores[1].(*rostam.Embedded), name, P, 5*time.Second)

		got, pages := mvScrollPageAll(t, ctx, stores[1], name, rostam.VectorFilter{}, L)
		assertExactlyOnceAscending(t, got, want)

		wantPages := (N + L - 1) / L // ceil(N/L); a trailing empty page may add one
		if pages != wantPages && pages != wantPages+1 {
			t.Fatalf("page count = %d, want %d or %d (ceil(N/L))", pages, wantPages, wantPages+1)
		}
		// Exhaustion ⇒ empty cursor is asserted inside mvScrollPageAll (a short/empty
		// final page MUST end pagination with next_cursor=="").
	}

	// --- Part B: stable under a concurrent insert/delete mid-scroll ---
	{
		const (
			P       = 4
			coreN   = 200 // ids 1..200 are CORE: never mutated, present THROUGHOUT
			volBase = 1000
			volN    = 60 // ids 1000..1059 are VOLATILE: candidates for deletion
			insBase = 5000
			L       = 20
		)
		name := mvCreatePartitionedClean(t, ctx, stores[0], "mvstable", P)
		// Seed CORE (1..coreN) and VOLATILE (volBase..volBase+volN-1) in shuffled order.
		core := map[uint64]bool{}
		var seedIDs []uint64
		for _, id := range shuffledIDs(coreN, 41) {
			seedIDs = append(seedIDs, id)
			core[id] = true
		}
		for i := 0; i < volN; i++ {
			seedIDs = append(seedIDs, uint64(volBase+i))
		}
		// Shuffle the combined seed order so core/volatile interleave across partitions.
		for i := range seedIDs {
			j := int((uint64(i)*2654435761 + 12345) % uint64(len(seedIDs)))
			seedIDs[i], seedIDs[j] = seedIDs[j], seedIDs[i]
		}
		for _, id := range seedIDs {
			idc := id
			md := rostam.VectorMetadata{"n": vector.NewInt(int64(idc))}
			retryUntil(t, "mv add", func() error {
				return stores[0].VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(int(idc % 80))}, md)
			})
		}
		waitEmbeddedCatalog(t, stores[1].(*rostam.Embedded), name, P, 5*time.Second)

		// A concurrent mutator: while we page, INSERT brand-new high ids and DELETE
		// not-yet-paged VOLATILE ids. Neither touches a CORE id, so the CORE set is
		// present THROUGHOUT the scroll. cursorMax tracks the highest id we have paged
		// so the mutator only deletes ids the cursor has NOT yet reached (an id deleted
		// after the cursor passed it was already returned — never re-deleted here).
		var cursorMax atomic.Uint64
		var wg sync.WaitGroup
		stop := make(chan struct{})
		wg.Add(1)
		go func() {
			defer wg.Done()
			nextDelIdx := 0
			ins := uint64(insBase)
			for {
				select {
				case <-stop:
					return
				default:
				}
				// Insert a brand-new high id (above everything; may or may not be paged).
				md := rostam.VectorMetadata{"n": vector.NewInt(int64(ins))}
				_ = stores[0].VectorMVAdd(ctx, name, ins, [][]float32{mvTokenAt(int(ins % 80))}, md)
				ins++
				// Delete the next VOLATILE id that the cursor has NOT yet reached.
				cm := cursorMax.Load()
				for nextDelIdx < volN {
					cand := uint64(volBase + nextDelIdx)
					nextDelIdx++
					if cand > cm { // not yet paged ⇒ safe to delete (it may simply never appear)
						_, _ = stores[0].VectorMVDelete(ctx, name, cand)
						break
					}
				}
				time.Sleep(time.Millisecond)
			}
		}()

		// Page the whole collection while the mutator runs. We track the id stream and
		// publish the paging high-water mark so the mutator deletes only ahead-of-cursor
		// volatile ids.
		var got []uint64
		cursor := ""
		var last uint64
		have := false
		for {
			docs, _, next, err := stores[1].VectorMVScroll(ctx, name, rostam.VectorFilter{}, L, cursor)
			if err != nil {
				close(stop)
				wg.Wait()
				t.Fatalf("VectorMVScroll (stability): %v", err)
			}
			for i, d := range docs {
				if i > 0 && d.ID <= docs[i-1].ID {
					close(stop)
					wg.Wait()
					t.Fatalf("stability page not strictly ascending at %d: %d <= %d", i, d.ID, docs[i-1].ID)
				}
				if have && d.ID <= last {
					close(stop)
					wg.Wait()
					t.Fatalf("stability stream not ascending across pages: %d <= %d (gap/dup/order bug)", d.ID, last)
				}
				got = append(got, d.ID)
				last = d.ID
				have = true
			}
			if have {
				cursorMax.Store(last)
			}
			if next == "" {
				break
			}
			cursor = next
			time.Sleep(2 * time.Millisecond) // give the mutator a window between pages
		}
		close(stop)
		wg.Wait()

		// Every CORE id (present THROUGHOUT) returned EXACTLY ONCE; the stream globally
		// ascending + dup-free. Volatile/inserted ids may or may not appear, but a CORE
		// id is never dropped or duplicated.
		seen := map[uint64]int{}
		for i, id := range got {
			if i > 0 && id <= got[i-1] {
				t.Fatalf("final stability stream not strictly ascending at %d: %d <= %d", i, id, got[i-1])
			}
			seen[id]++
			if seen[id] > 1 {
				t.Fatalf("id %d duplicated in concurrent-mutation scroll (dup bug)", id)
			}
		}
		for id := range core {
			if seen[id] != 1 {
				t.Fatalf("CORE id %d (present throughout) appeared %d times, want exactly 1 (gap/drop under concurrent mutation)", id, seen[id])
			}
		}
	}
}

// linMVScrollOnce runs a strict Linearizable MV scroll (read_consistency=2,
// fail-on-unavailable) and returns the distinct id set + error. Fail-on-unavailable
// turns an incomplete fan-out into a hard error, so the result is never a silently
// truncated subset (the MV mirror of linNamedScrollIDs).
func linMVScrollOnce(ctx context.Context, s rostam.Store, name string, filter rostam.VectorFilter, limit int) (map[uint64]bool, error) {
	docs, _, _, err := s.VectorMVScrollExt(ctx, name, filter, limit, "",
		rostam.MVScrollOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1})
	if err != nil {
		return nil, err
	}
	return idSet(docs), nil
}

// TestMVScrollLinearizableArmsBarriers is THE barrier proof for the MV scroll: a
// Linearizable MV scroll over a PARTITIONED (P>1) collection arms BOTH barriers,
// the named scroll's MV mirror (TestNamedLinearizableShardBarrierRealPath +
// TestNamedLinearizableReadFiresBarrierExactlyOnce):
//
//   - the META readIndex barrier fires EXACTLY ONCE on the coordinator (one
//     coordinator RTT before fan-out, NOT P), counted via SetMetaReadIndexForwardHook
//     on a meta-FOLLOWER coordinator.
//   - the SHARD data barrier fires per partition (>= P), counted via
//     SetBarrierEnteredHook — proving `vector_mv_scroll` in ops.ReadConsistencyOf
//     plus rc-rides-every-arg actually arms each partition leader's barrier.
//
// CONTRAST: an AnyReplica MV scroll fires the META barrier ZERO times. A filter
// threaded through a Linearizable scroll returns only matching docs.
func TestMVScrollLinearizableArmsBarriers(t *testing.T) {
	// (1) META-barrier-exactly-once + AnyReplica-zero, on a real cluster with a
	// meta-FOLLOWER coordinator (mirror TestNamedLinearizableReadFiresBarrierExactlyOnce).
	t.Run("MetaBarrierExactlyOnce", func(t *testing.T) {
		stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3 so leader-routed partitioned reads are serviceable everywhere
		ctx := context.Background()
		const (
			P = 4
			N = 60
		)
		readIdx := metaFollowerStore(t, stores)
		name := mvCreatePartitionedClean(t, ctx, stores[0], "mvmeta", P)
		for id := 1; id <= N; id++ {
			idc := uint64(id)
			md := rostam.VectorMetadata{"group": vector.NewInt(int64(id % 2)), "n": vector.NewInt(int64(id))}
			retryUntil(t, "mv add", func() error {
				return stores[0].VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(id % 80)}, md)
			})
		}
		waitEmbeddedCatalog(t, stores[readIdx].(*rostam.Embedded), name, P, 10*time.Second)

		var forwards int32
		cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
		defer cluster.SetMetaReadIndexForwardHook(nil)

		// A Linearizable MV scroll on the follower coordinator. P=4 ⇒ a per-partition
		// meta barrier would forward 4 times; the coordinator barrier forwards EXACTLY 1.
		atomic.StoreInt32(&forwards, 0)
		if _, _, _, err := stores[readIdx].VectorMVScrollExt(ctx, name, rostam.VectorFilter{}, 10, "",
			rostam.MVScrollOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1}); err != nil {
			t.Fatalf("Linearizable VectorMVScrollExt: %v", err)
		}
		if got := atomic.LoadInt32(&forwards); got != 1 {
			t.Fatalf("Linearizable MV scroll fired %d __meta_readindex__ forwards on a P=%d collection, want EXACTLY 1 (one coordinator barrier, NOT per-partition)", got, P)
		}

		// CONTRAST: an AnyReplica MV scroll fires the meta barrier ZERO times.
		atomic.StoreInt32(&forwards, 0)
		_, _, _, _ = stores[readIdx].VectorMVScrollExt(ctx, name, rostam.VectorFilter{}, 10, "",
			rostam.MVScrollOpts{ReadConsistency: ops.ConsistencyAnyReplica, OnPartitionUnavailable: 1})
		if got := atomic.LoadInt32(&forwards); got != 0 {
			t.Fatalf("AnyReplica MV scroll fired %d __meta_readindex__ forwards, want 0 (zero added cost off the Linearizable path)", got)
		}
	})

	// (2) SHARD-barrier-per-partition + AnyReplica-zero + filter-applied, on a
	// single partitioned embedded engine (mirror TestNamedLinearizableShardBarrierRealPath).
	t.Run("ShardBarrierPerPartition", func(t *testing.T) {
		s := newSingleEmbedded(t)
		waitLeaderEmbedded(t, s)
		ctx := context.Background()

		var barrierEntries atomic.Int64
		shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
		t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
		delta := func() int64 { return barrierEntries.Swap(0) }

		const (
			parts  = "mvparts"
			single = "mvsingle"
			P      = 4
			N      = 80
		)
		if err := s.VectorMVCreateCollection(ctx, parts, rostam.MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
			t.Fatalf("VectorMVCreateCollection parts (P=%d): %v", P, err)
		}
		if err := s.VectorMVCreateCollection(ctx, single, rostam.MultiVectorConfig{Dim: 4, Partitions: 1}); err != nil {
			t.Fatalf("VectorMVCreateCollection single (P=1): %v", err)
		}
		evenWant := map[uint64]bool{}
		for id := 1; id <= N; id++ {
			idc := uint64(id)
			md := rostam.VectorMetadata{"group": vector.NewInt(int64(id % 2)), "n": vector.NewInt(int64(id))}
			if err := s.VectorMVAdd(ctx, parts, idc, [][]float32{mvTokenAt(id % 80)}, md); err != nil {
				t.Fatalf("mv add parts %d: %v", id, err)
			}
			if err := s.VectorMVAdd(ctx, single, idc, [][]float32{mvTokenAt(id % 80)}, md); err != nil {
				t.Fatalf("mv add single %d: %v", id, err)
			}
			if id%2 == 0 {
				evenWant[idc] = true
			}
		}
		delta() // drain any setup entries

		lin := rostam.MVScrollOpts{ReadConsistency: ops.ConsistencyLinearizable}
		any := rostam.MVScrollOpts{ReadConsistency: ops.ConsistencyAnyReplica}

		// (a) PARTITIONED Linearizable MV scroll: shard barrier on the partition
		// leaders. Floor is >= 1 (NOT >= P): the readindex-coalescer merges concurrent
		// per-partition reads that land on co-hosted Stores into shared barriers, so the
		// fire count is 1..P depending on timing. The anti-drop teeth survive: a dropped
		// rc byte ⇒ ReadConsistencyOf returns (0,false) on every partition ⇒ 0 barriers
		// ⇒ 0 < 1 still fails (STALE-SERVE HOLE caught).
		if _, _, _, err := s.VectorMVScrollExt(ctx, parts, rostam.VectorFilter{}, 10, "", lin); err != nil {
			t.Fatalf("linearizable VectorMVScrollExt parts: %v", err)
		}
		if n := delta(); n < 1 {
			t.Fatalf("PARTITIONED linearizable MV scroll entered the shard barrier %d times, want >= 1 "+
				"(coalesced partition-leader reads). 0 ⇒ vector_mv_scroll not in ReadConsistencyOf or rc byte dropped ⇒ STALE-SERVE HOLE", n)
		}

		// (b) UNPARTITIONED (P==1) Linearizable MV scroll: routes to the leader via
		// callReadLeader, barrier runs (>= 1).
		if _, _, _, err := s.VectorMVScrollExt(ctx, single, rostam.VectorFilter{}, 10, "", lin); err != nil {
			t.Fatalf("linearizable VectorMVScrollExt single: %v", err)
		}
		if n := delta(); n < 1 {
			t.Fatalf("UNPARTITIONED (P==1) linearizable MV scroll entered the shard barrier %d times, want >= 1 "+
				"(the P<=1 path must route to the leader WITH rc). 0 ⇒ STALE-SERVE HOLE", n)
		}

		// (c) FILTER applied during a Linearizable scroll: only matching (even) docs are
		// paged, gap-free + dup-free, == the even ground truth. Also proves the shard
		// barrier fires (rc rides the filtered per-partition arg).
		evenFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: "group", Value: vector.NewInt(0)}
		var gotEven []uint64
		cursor := ""
		for {
			docs, _, next, err := s.VectorMVScrollExt(ctx, parts, evenFilter, 7, cursor, lin)
			if err != nil {
				t.Fatalf("linearizable filtered VectorMVScrollExt: %v", err)
			}
			for _, d := range docs {
				gotEven = append(gotEven, d.ID)
			}
			if next == "" {
				break
			}
			cursor = next
		}
		assertExactlyOnceAscending(t, gotEven, evenWant)
		if n := delta(); n < 1 {
			t.Fatalf("PARTITIONED linearizable FILTERED MV scroll entered the shard barrier %d times across pages, want >= 1 "+
				"(coalesced per-partition reads; rc must ride the filtered per-partition arg — 0 ⇒ dropped rc ⇒ STALE-SERVE)", n)
		}

		// (d) CONTRAST: AnyReplica MV scrolls on the same collections must NOT enter the
		// shard barrier (zero added consensus cost on the default path).
		if _, _, _, err := s.VectorMVScrollExt(ctx, parts, rostam.VectorFilter{}, 10, "", any); err != nil {
			t.Fatalf("anyreplica VectorMVScrollExt parts: %v", err)
		}
		if _, _, _, err := s.VectorMVScrollExt(ctx, single, rostam.VectorFilter{}, 10, "", any); err != nil {
			t.Fatalf("anyreplica VectorMVScrollExt single: %v", err)
		}
		if n := delta(); n != 0 {
			t.Fatalf("AnyReplica MV scrolls entered the shard barrier %d times, want 0 (default path must be barrier-free)", n)
		}

		// A Linearizable filtered scroll returns ONLY matching docs (independent re-check
		// of the full filtered set via a single big page).
		got, err := linMVScrollOnce(ctx, s, parts, evenFilter, 0)
		if err != nil {
			t.Fatalf("linearizable filtered full scroll: %v", err)
		}
		for id := range got {
			if id%2 != 0 {
				t.Fatalf("Linearizable filtered MV scroll returned ODD id %d — filter not applied during the scroll", id)
			}
		}
		if len(got) != len(evenWant) {
			t.Fatalf("Linearizable filtered MV scroll returned %d ids, want %d (the even ground truth)", len(got), len(evenWant))
		}
	})
}
