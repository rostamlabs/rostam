// SPDX-License-Identifier: Apache-2.0

package vector

// The bitset admission gate.
//
// WHAT IT IS. When the filter-first planner DECLINES to brute-force an
// index-narrowed candidate set (the set is bigger than the crossover, or bigger
// than the materialization limit), the query falls back to filtered graph
// traversal — and until now that traversal threw the planner's work away and
// re-derived membership per candidate from scratch: liveMeta(slot) → a
// string-keyed map lookup → Value.Equal, once per traversed node, for a fact the
// payload index already knew. Profiling put that at ~13% of a filtered query.
//
// The gate keeps the plan — literally the same one. searchIntoWith now collects
// the narrowing plan once (collectNarrowSets) and hands it to both consumers:
// the planner intersects it into a candidate list, and when that is too big to
// brute-force, the gate folds the SAME posting sets into a slot bitset over the
// arena's capacity. The admission gate then consults one bit instead of a map.
// Sharing the plan is not just tidiness: it means the gate can never materialize
// a posting set the planner declined to, so arming one cannot add work that the
// query was not already doing.
//
// THREE GRADES, one per posting set (narrowClass) and then per query:
//
//   - EXACT — the index set IS the predicate's match set, so the bit test
//     REPLACES the predicate entirely. No liveMeta, no map, no Value.Equal.
//   - SUPERSET — the index set merely CONTAINS the match set, so a clear bit is a
//     proven reject (taken immediately, without touching metadata) and a set bit
//     falls through to the full predicate. Strictly fewer predicate evaluations
//     than today; never a wider result.
//   - UNPROVEN — the superset property is not established, so the set is not used
//     by the gate at all (only by the planner, which re-checks everything). No op
//     produces this grade since m5 closed the NaN hole that put the ordering
//     family here; see narrowClass for the history and why the grade stays.
//
// A filter that is not index-narrowable at all gets no gate and the exact path
// that exists today.
//
// TWO SIDES (m5). Everything above prices the gate by the ACCEPTED posting mass,
// which is the right model for a selective filter and the wrong one for a filter
// that passes nearly everything: `id >= N` over a million points enumerates
// 990k postings to accelerate a traversal of ~10k candidates, so the gate
// correctly declines and the query pays a metadata lookup per candidate — which
// is exactly the VectorDBBench filter case Rostam was losing. The same filter's
// REJECTION side is 1% of the corpus, and a bitset of rejections answers the
// identical question with the sense inverted: mark what fails, admit what is
// still set. buildComplementGate is that second attempt, tried whenever the
// first produced nothing, and it makes profitability depend on
// min(accept mass, reject mass) rather than on the accept side alone.
//
// The two builds have DIFFERENT safety obligations and must not be reasoned
// about as one. The positive build may always be widened (dropping a set is
// safe); the complement build may always be NARROWED (marking fewer rejects is
// safe). Getting that backwards in either direction loses rows. See
// collectComplementSets for the complement's argument in full.
//
// SCOPE. This is the dense HNSW path (searchIntoWith → searchDenseLockedWith)
// only. Deliberately NOT covered, and each for a reason rather than an oversight:
//
//   - The external-provider path (metaOf != nil — named vectors and multivector,
//     whose payloads live outside the sub-arena in an ID-keyed index). Admission
//     there runs against a different key space, and the sub-arena's own payload
//     index is empty, so the whole planner is already skipped on that path. A
//     gate for payloadIndexID needs an id→slot mapping decision of its own; it is
//     a follow-up, not a copy of this file.
//   - HybridSearch's dense lane (buildLanes → searchDenseLocked). It has a
//     predicate but no Filter in scope, so arming a gate would mean threading the
//     uncompiled filter down a second call chain for a lane that also runs a
//     sparse pass. Follow-up.
//
// WHAT IT DOES NOT CHANGE. Tombstone/TTL admission runs BEFORE the gate, exactly
// as before, so a dead slot is still charged to admitDead and never to
// filterRejects — the batched-vs-per-pair reject-tally equality
// (batched_traversal_test.go) is load-bearing and both traversal paths share one
// admit closure, hence one gate. A gate reject IS counted as a filter reject: it
// is a rejection by the filter, decided by a faster oracle, and hiding it would
// make Stats().FilterRejects mean something different depending on which oracle
// answered.
//
// MUTATION SAFETY, under the Option B locking model. The gate is built and
// consulted inside ONE h.mu.RLock section (searchIntoWith holds it across both
// the build and the whole traversal), so the question is what else can be
// running concurrently with that read lock — and the answer changed when the
// insert path split (link_stripes.go): a LINKER now runs under the read lock
// too, alongside queries. The gate is still safe, for two separate reasons that
// are worth keeping apart.
//
//   - THE POSTING SETS CANNOT MOVE. Every payload-index write — reindex from
//     Insert/SetPayload/OverwritePayload/DeletePayloadKeys/ClearPayload, drop
//     from Delete/Reclaim, rebuild from restore — runs under h.mu.Lock. Option B
//     did not relax that: reindex (along with the sparse and BM25 index updates)
//     stayed in placeLockedAt, the exclusive PLACEMENT half. What moved to the
//     read lock is linkRead, and all it mutates is graph topology — level0 /
//     level0Len / node.upper under the stripes, entryPoint / maxLevel under
//     globalMu. It never touches payloadIdx. So a concurrent linker cannot make
//     the bitset disagree with the metadata the predicate would have read, and
//     the gate needs no version stamp: it cannot outlive the lock that pins the
//     index it was built from.
//
//   - THE BITSET CANNOT BE INDEXED OUT OF RANGE. build sizes it to
//     arena.Capacity() and test indexes words directly. A concurrent linker can
//     publish back-edges pointing at a slot this traversal has not seen, but not
//     at a slot beyond that capacity: placement — the only thing that allocates
//     a slot or grows the slab — needs the write lock this reader is holding
//     off, and a linker already in flight had its slot allocated before the
//     reader's Capacity() read. This is exactly the invariant s.reset's visited
//     set already relies on (searchLayerCore sizes it from the same call), so
//     the gate is neither more nor less exposed than the visited set is; the
//     grow-race coverage that pins one pins the other.

// admitGate is one query's slot bitset plus its strength. The zero value is
// disabled, which is the correct state for every path that never builds one.
type admitGate struct {
	// words is the slot bitset: bit (slot%64) of words[slot/64] is set iff slot
	// is in the intersection of the plan's posting sets. Sized to the arena's
	// capacity in words and REUSED across queries via the pooled layerScratch.
	words []uint64
	on    bool
	// exact marks that the bitset equals the predicate's match set, so a set bit
	// admits without evaluating the predicate. See filterIndexExact.
	exact bool
	// cols is the COLUMN oracle (payload_column.go), and when it is non-empty it
	// REPLACES the bitset rather than supplementing it: admission is the
	// conjunction of the terms evaluated at the slot, always exact, never a
	// metadata read. The two are mutually exclusive by construction — the search
	// path tries the column first and only builds a bitset if the filter is not
	// column-expressible — so `words` is untouched (and unsized) on this path.
	cols []columnTerm
}

func (g *admitGate) active() bool { return g.on }

// disable returns the gate to its zero (unused) state, RETAINING the backing
// array for the next query on this scratch. Every path that leaves a scratch
// must go through here or through getLayerScratch: a gate that outlived its
// query would answer a different filter's question.
func (g *admitGate) disable() {
	g.on = false
	g.exact = false
	// Truncate rather than nil: the term slice is pooled with the scratch, and
	// retaining its capacity is the point. But the elements must be cleared over
	// the CAPACITY, not the length, and that distinction is the whole reason this
	// is three lines instead of one.
	//
	// A columnTerm holds a []float64 that is the field's WHOLE COLUMN — eight
	// bytes per slot, tens of megabytes on a large index. buildColumnGate's
	// decline path leaves partial terms in the slice and truncates to zero length
	// (a filter like And(a>=x, b>=y, c==z) resolves two columns before the Eq leaf
	// refuses the whole thing), so by the time disable runs there is nothing in
	// [0,len) to clear and everything worth clearing sits in [len,cap). A
	// length-bounded clear here retains those column pointers inside an IDLE
	// POOLED SCRATCH — invisible, unbounded by anything the query did, and immune
	// to the one thing that is supposed to release a column, since Reclaim only
	// nils payloadIndex.cols and cannot reach a sync.Pool.
	clear(g.cols[:cap(g.cols)])
	g.cols = g.cols[:0]
}

// test reports whether slot survives the gate. Indexes words directly: the gate
// is sized to arena.Capacity() under the same read lock the traversal holds, and
// every traversed slot is < capacity (the visited set makes the identical
// assumption), so an out-of-range slot is a bug we want loud, not silently
// admitted or silently dropped.
func (g *admitGate) test(slot uint32) bool {
	return g.words[slot>>6]&(1<<(slot&63)) != 0
}

// columnar reports whether this gate answers from columns rather than a bitset.
func (g *admitGate) columnar() bool { return len(g.cols) > 0 }

// testCols evaluates the column terms at slot. An And, so the first failure
// answers — which on a selective filter is usually the first term.
func (g *admitGate) testCols(slot uint32) bool {
	for i := range g.cols {
		if !g.cols[i].holds(slot) {
			return false
		}
	}
	return true
}

// armColumns arms the gate on the column oracle. Always EXACT: every term is
// exact (see collectColumnTerms) and collection is all-or-nothing, so a slot
// that satisfies every term satisfies the filter, full stop. The slice is the
// scratch's own (collectColumnTerms appends into it), so arming allocates
// nothing on a warm pool.
func (g *admitGate) armColumns(terms []columnTerm) {
	g.cols = terms
	g.on = true
	g.exact = true
}

// build fills the gate with the intersection of sets, over `capacity` slot bits.
//
// CLEARING: memclr of capacity/64 words, not an epoch stamp. The epoch trick
// visitedSet uses buys an O(1) reset but needs a whole stamp word (or at least a
// byte) per slot, which is 8-64x the cache footprint — and shrinking that
// footprint on the hot test path is the entire point of a bitset. The clear is
// also paid ONCE per query against a traversal of thousands of candidates, where
// visitedSet pays its reset once per LEVEL: at one million slots the bitset is
// 15625 words (~122 KiB) and clears in about 2 µs (BenchmarkAdmitGateUnits), and
// gateProfitable prices even that explicitly (gateWordClearDNS) so a query too
// small to amortize it never builds a gate at all.
func (g *admitGate) build(sets []narrowSet, capacity int, exact bool) {
	nw := (capacity + 63) / 64
	if cap(g.words) < nw {
		g.words = make([]uint64, nw) // make zero-fills; no separate clear
	} else {
		g.words = g.words[:nw]
		clear(g.words)
	}
	g.on = true
	g.exact = exact
	if len(sets) == 0 {
		return // no sets ⇒ no bits ⇒ reject everything (caller must not do this)
	}
	// Scan the smallest set and probe the rest — the same strategy (and the same
	// cost) intersectSlotSets uses, so gateProfitable's estimate matches reality.
	smallest := 0
	for i := 1; i < len(sets); i++ {
		if len(sets[i].set) < len(sets[smallest].set) {
			smallest = i
		}
	}
	for slot := range sets[smallest].set {
		inAll := true
		for i, s := range sets {
			if i == smallest {
				continue
			}
			if _, ok := s.set[slot]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			g.words[slot>>6] |= 1 << (slot & 63)
		}
	}
}

// buildComplement fills the gate from REJECTION sets: every bit starts SET
// (admit) and each posting in each set CLEARS one. Admission is then the same
// single bit test the ordinary gate uses — the inversion is paid once, at build
// time, so the traversal's hot path is byte-identical and `exact` keeps its
// meaning.
//
// It is always EXACT, and that is not an optimization but the reason it exists.
// collectComplementSets refuses to produce a partial union (see its safety
// argument), so by the time we are here R(f) ⊆ U ⊆ R(f): a clear bit is a proven
// reject and a set bit is a proven admit. A complement gate that had to re-check
// the predicate on a set bit would save nothing at the pass rates it is built
// for — at 99% pass, almost every candidate takes the set-bit branch.
//
// COMBINATION IS UNION, not intersection. The gate's ordinary build intersects
// (a slot must be in every set to be admitted); rejection sets compose the other
// way — failing ANY conjunct is failing the And — so each set independently
// clears bits. That also makes the loop cheaper than intersection's probe-the-
// rest inner loop, which matters because this path is chosen precisely when the
// posting mass is small.
//
// Bits at or beyond `capacity` are left SET by the all-ones fill and never
// examined: test() is only ever called with a slot the traversal reached, and
// that is < capacity for the same reason the visited set assumes it (see the
// file header's range argument).
func (g *admitGate) buildComplement(rejects []map[uint32]struct{}, capacity int) {
	nw := (capacity + 63) / 64
	if cap(g.words) < nw {
		g.words = make([]uint64, nw)
	} else {
		g.words = g.words[:nw]
	}
	for i := range g.words {
		g.words[i] = ^uint64(0)
	}
	g.on = true
	g.exact = true
	for _, set := range rejects {
		for slot := range set {
			g.words[slot>>6] &^= 1 << (slot & 63)
		}
	}
}

// Gate cost model, in TENTHS OF A NANOSECOND per unit of work. These constants
// are the only tuning surface, and they are MEASURED, not guessed —
// BenchmarkAdmitGateUnits reports each one directly in dns so the numbers below
// can be checked against a run rather than argued about:
//
//   - gateWordClearDNS — zeroing one 64-bit word of the bitset. Measured 1.3
//     dns/word (16 KiB memclr); rounded up.
//   - gatePostingDNS — visiting one posting entry while filling the bitset: a Go
//     map iteration step plus a scattered bit write (and, for a multi-set plan,
//     one probe per other set). Measured 77.6 dns/posting over a 100k-entry set.
//     This is the term that bounds the whole feature — it is ~35x the word
//     clear, so posting mass, not arena size, is what decides.
//   - gateVisitSaveDNS — what the gate saves on ONE traversed candidate it
//     rejects: liveMeta + a string-hashed map lookup + Value.Equal + the closure
//     call, minus the bit test that replaces them. Measured 233 dns/eval for an
//     Eq predicate over a warm two-key map; held at 200 because the benchmark's
//     map is cache-resident and its key comparison is short, while the saving
//     must stay a lower bound for the decision to be conservative.
//
// Ratio sanity check, which is really all the model needs: a posting costs about
// a third of the predicate call it might save, so the gate is worth building
// while posting mass stays under ~2.5x the traversal it accelerates.
//   - gateRangeKeyDNS — reaching one DISTINCT KEY while walking a range out of
//     the payload index: a scalarKey hash probe into the field's key map (a
//     40-byte struct key, so a string hash) plus a fresh map iterator over that
//     key's posting set. BenchmarkAdmitGateUnits/rangekey measures 1452 dns for
//     a key carrying ONE posting; unionposting puts a posting at 494 when it is
//     merged into a result map, leaving ~960 for the key itself. Held at 1000.
//
// gateRangeKeyDNS is the m5 addition, and it changes the SHAPE of the model
// rather than its arithmetic. It is 12x the per-posting cost and 5x the
// predicate evaluation the gate replaces, so a range whose posting mass is one
// slot per distinct value costs ~5 predicate calls PER SLOT to index — the
// opposite conclusion from the one a mass-only model reaches. That is not a
// corner case, it is the VectorDBBench shape (`id >= N` over a unique id), where
// a mass-only model prices a 10k-slot complement at 800k dns and is wrong by an
// order of magnitude. Two different costs, two different terms.
//
// The unionposting measurement is also why collectComplementSets hands back the
// index's own per-key posting maps instead of merging them: at 494 dns a merged
// posting costs 6x the 80 it costs to clear a bit straight from the stored map,
// and the merge buys the gate nothing, because clearing a bit twice is clearing
// a bit. gatePostingDNS prices the path that is actually taken.
const (
	gateWordClearDNS = 2
	gatePostingDNS   = 80
	gateRangeKeyDNS  = 1000
	gateVisitSaveDNS = 200
)

// gateProfitable is the build/skip decision, and it is the honest one: building
// the bitset costs O(posting mass) while the traversal it accelerates costs
// O(ef · 2M) — INDEPENDENT of corpus size — so the gate is profitable only in a
// band, and the band must be checked rather than assumed.
//
//	build  ≈ words·gateWordClearDNS + minSize·nsets·gatePostingDNS
//	saving ≈ visits · rejectFraction · gateVisitSaveDNS
//
// The band's LOWER edge is the filter-first crossover itself: below it the
// planner already brute-forced the set and we are never called. Its UPPER edge
// is posting mass ≈ 2.5·visits, from the constants above — and that upper edge is
// exactly why the gate is not a universal win. It widens with ef·M (a selective
// filter that drives the ef-doubling ladder to MaxEfSearch has a huge traversal
// to amortize against) and narrows as the corpus grows (posting mass scales with
// N, the traversal does not). The high-selectivity case the audit cared about —
// a filter matching a fraction of a percent of a large corpus, too big for the
// materialization limit but rejecting nearly every traversed node — sits in the
// middle of the band and is where the win is largest.
//
// Both bounds are ESTIMATES used only to choose a strategy; either answer is
// correct, so a mis-estimate costs throughput and never results.
//
// minSize is the size of the plan's smallest posting set — an upper bound on the
// intersection, which makes 1 - minSize/n a LOWER bound on the fraction of
// traversed candidates the gate will reject. Erring toward under-counting the
// saving keeps the decision conservative.
func (h *hnsw) gateProfitable(minSize, nsets, k int) bool {
	n := h.arena.Size()
	if n <= 0 {
		return false
	}
	words := (h.arena.Capacity() + 63) / 64
	cost := int64(words)*gateWordClearDNS + int64(minSize)*int64(nsets)*gatePostingDNS

	// Reuse the planner's OWN traversal-size model (the efEff·2M term
	// preferFilterFirst weighs filter-first against) so the gate and the planner
	// cannot disagree about how big the graph walk is.
	visits := h.graphVisitEstimate(minSize, k)
	sel := float64(minSize) / float64(n)
	if sel > 1 {
		sel = 1 // superset (tombstones) → assume nothing is rejected
	}
	saving := int64(visits * (1 - sel) * gateVisitSaveDNS)
	return cost <= saving
}

// filterIndexExact reports whether the payload index answers this filter
// EXACTLY — i.e. the plan's posting-set intersection equals the set of live
// slots the compiled predicate accepts, so a set bit may admit without
// evaluating the predicate at all.
//
// PER-OP CLASSIFICATION. Exactness is a property of the posting set's
// construction versus the predicate's own test, so it is decided op by op. The
// EXACT ops here, and why:
//
//   - FilterEq (scalar value). reindex posts each slot under scalarKeyOf(value),
//     which carries the value's KIND; the predicate is lookupPath(m,f) ok &&
//     got.Equal(want), and Value.Equal likewise requires equal kinds first. The
//     two agree case by case: string/int/float/bool compare identically, ±0.0
//     compares equal under both Go map keys and ==, and a NaN want is
//     unretrievable as a map key exactly as NaN == NaN is false. A non-scalar
//     want (slices/geo) is declined by scalarKeyOf, and collectEqTerms then
//     never narrows on it — so an Eq that reaches this classifier is one the eq
//     index answers exactly. A field with no eq postings is exact too: any value
//     the predicate could accept would be eq-indexable and therefore present.
//   - FilterContains (scalar want). elementKeysOf posts one key per DISTINCT
//     array element under the SAME kind-carrying scalarKey; compileContains
//     kind-checks the array and scans it for ==. containsSet's own doc comment
//     already claims this equality; the classifier is where it is finally
//     RELIED on, so payload_contains_index_test.go's element-kind cases become
//     load-bearing.
//   - FilterIn (array want). inSet unions the eq postings of each member under
//     the member's kind; compileIn kind-checks the field then scans the members
//     for ==. Same argument as Eq, member by member. An empty array is exact
//     (both sides match nothing).
//   - FilterGt/Gte/Lt/Lte, and the FilterDt* family that lowers onto them
//     (m5). This entry USED to be in the "everything else" list below, and the
//     thing that moved it is orderingHoldsFloat: with NaN unordered for range
//     ops and unindexable as a key, the predicate and orderingSet finally answer
//     the same question, case for case. Numeric want: reindex posts each slot
//     under scalarKeyOf(value), ensureSorted lifts the int and float keys into
//     one `<`-ordered list, and numRange's binary search selects exactly the keys
//     the predicate's compareFloat would accept — int/float interoperate on both
//     sides, ±0.0 compares equal on both sides, ±Inf orders on both sides. A
//     slot whose value is a STRING, bool, geo, array or absent carries no
//     numeric key, and the predicate rejects it too (numericValue fails, then
//     the string path needs a ValueString want). String want: the mirror, over
//     sc.str and compareString, and a numeric-valued slot is rejected by both.
//     A field with no postings at all is exact-empty: nothing can match a field
//     nothing carries. An overflowed `limit` never reaches here — orderingSet
//     returns ok=false rather than a truncated set — and a want whose kind can
//     drive neither comparison is declined below, mirroring orderingSet.
//   - FilterAnd of exact conjuncts. The intersection of exact sets is exact,
//     PROVIDED every conjunct contributed a set — which the caller verifies
//     separately by counting (len(sets) == filterLeafCount), because
//     collectNarrowSets legitimately SKIPS conjuncts it cannot narrow.
//
// Everything else is SUPERSET (pre-filter only), and deliberately so:
//
//   - FilterMatch. Token postings are PER-DOCUMENT: a ValueStrings field posts
//     the tokens of ALL its elements, while compileMatch requires ONE element to
//     carry every query token. ["red car","blue sky"] is posted under red/car/
//     blue/sky and so survives a "red blue" query the predicate rejects. Strict
//     superset — never exact.
//   - Geo. geoSet covers the region with a geohash-cell union of a bounding box
//     that is definitionally larger than the region. Superset by construction.
//   - Or / Not / Ne / Regex / IsNull / IsEmpty. Never narrowed at all, so they
//     cannot appear in an exact plan; an And containing one is a superset plan
//     (the un-narrowed conjunct is re-checked by the predicate).
//
// contentField and every record path (isRecordPath) are declined here too,
// but neither is really THIS function's problem any more: indexNarrowable
// declines both at every posting-set lookup upstream (collectEqTerms and
// friends), so neither ever reaches a plan for these checks to grade. They
// stay as defence in depth — this function derives its own answer per op
// straight from f.Field, not from what collectNarrowSets happened to plan,
// so it must independently agree with indexNarrowable rather than trust that
// whatever reached it was already filtered.
//
// The original note here read "only the EXACT classification could turn that
// into wrong results — a superset plan re-checks, and an empty gate just rejects
// what filter-first would also have rejected". That was WRONG, and wrong in the
// way that hid a real bug: what filter-first "would also have rejected" was the
// entire result set. An empty posting set made the planner brute-force ZERO
// candidates and return nothing, for a filter every row satisfied. The gate was
// never the exposure; the planner was. Left as a warning against reasoning about
// an empty set as though it were merely a narrow one.
//
// NOT COVERED HERE: per-key TTL. A key past its deadline is hidden by liveMeta
// but is still in the posting set until a mutation or the sweeper reindexes the
// slot, which would let an exact gate admit a slot the predicate rejects. The
// caller gates exactness on the arena's per-key-deadline count being zero
// (buildAdmitGate); this function is about op semantics only.
func filterIndexExact(f Filter) bool {
	switch f.Op {
	case FilterEq, FilterContains:
		if !indexNarrowable(f.Field, f.Op) {
			return false
		}
		_, ok := scalarKeyOf(f.Value)
		return ok
	case FilterIn:
		if !indexNarrowable(f.Field, f.Op) {
			return false
		}
		switch f.Value.Kind {
		case ValueStrings, ValueInts, ValueFloats:
			return true
		default:
			// ValueRecord considered: falls here, correctly declining — a record
			// is not an array, so FilterIn's index-exactness claim doesn't apply.
			return false
		}
	case FilterGt, FilterGte, FilterLt, FilterLte:
		if !indexNarrowable(f.Field, f.Op) {
			return false
		}
		// Mirror orderingSet's own kind test: a want that can drive neither the
		// numeric nor the string comparison produces no posting set at all, so
		// claiming exactness for it would be claiming it about a set that does not
		// exist. numericValue also rejects a NaN want here — scalarKeyOf declines
		// the key, orderingHoldsFloat rejects every row, and the two agree that
		// nothing matches, but the leaf never reaches a plan either way.
		if wv, ok := numericValue(f.Value); ok {
			return wv == wv // a NaN bound narrows nothing; see orderingHoldsFloat
		}
		return f.Value.Kind == ValueString
	case FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte:
		if !indexNarrowable(f.Field, f.Op) {
			return false
		}
		// datetimeBound is the SHARED lowering (compileDatetime calls it too), so
		// an ordering leaf and its datetime spelling are the same leaf by the time
		// either side compares anything.
		_, ok := datetimeBound(f.Value)
		return ok
	case FilterAnd:
		if len(f.And) == 0 {
			return false // an empty And matches everything; the index narrows nothing
		}
		for i := range f.And {
			if !filterIndexExact(f.And[i]) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

// filterLeafCount counts the filter's leaves, recursing through And exactly as
// collectNarrowSets' helpers do. Compared against len(sets) it answers "did the
// plan narrow on EVERY conjunct, or did it skip one?" — the second half of the
// exactness test (see collectNarrowSets' one-set-per-covered-leaf invariant).
// Non-And nodes (including Or/Not, which are never narrowable) count as one leaf
// so a plan that skipped them can never reach len(sets) == filterLeafCount.
func filterLeafCount(f Filter) int {
	if f.Op != FilterAnd {
		return 1
	}
	n := 0
	for i := range f.And {
		n += filterLeafCount(f.And[i])
	}
	return n
}

// gateMode selects the gate's arming policy. It exists for the same reason
// batchedExpand does: an A/B over the SAME index, so a differential test is not
// secretly comparing two different graphs.
type gateMode uint8

const (
	gateAuto  gateMode = iota // production: gateProfitable decides
	gateOff                   // never arm — the pre-feature path, byte for byte
	gateForce                 // arm whenever the filter is narrowable, cost be damned
	// gateForceComplement suppresses the POSITIVE gate and forces the complement
	// one. Without it the complement path would be nearly untestable: for every
	// filter the complement can handle, the positive gate can handle it too and
	// gateForce arms that one first, so a differential run under gateForce would
	// compare the positive gate against gate-off and report a green that says
	// nothing about the rejection side.
	gateForceComplement
)

// admitGateMode is the arming policy. gateForce is what makes the equivalence
// tests meaningful: on a test-sized corpus the cost model correctly declines
// almost every gate, so an auto-only test would assert equivalence between two
// identical code paths and pass forever.
var admitGateMode = gateAuto

// admitGateForceExact promotes an armed gate to EXACT regardless of what the
// classification concluded. It is a MUTATION-TESTING SEAM and nothing else.
//
// The exactness decision is deliberately over-determined — the plan's per-set
// narrowClass, the leaf count, the per-op filterIndexExact table, and the
// per-key-TTL check must ALL agree — which is good for safety and bad for
// testability: mutating any single one of them leaves the other three to veto,
// so the mutant is silently harmless and the test that "caught" nothing looks
// like it passed. This flag mutates the CONCLUSION instead of one of its
// premises, which is the question actually worth answering: if a superset op
// were ever classified exact, by any route, would the differential harness
// notice? See TestAdmitGateExactMutantsAreCaught.
var admitGateForceExact bool

// admitGateSkipTotality drops the complement gate's TOTALITY precondition — the
// requirement that every live slot carries a comparable value for the field
// (fieldTotalNumeric). It is a MUTATION-TESTING SEAM, and it mutates the one
// premise whose failure is silent: without totality, a slot with no value for
// the field is rejected by the predicate but appears in no rejection posting
// set, so the complement gate leaves its bit SET and admits a point the filter
// excludes. Nothing else in the system notices — the result set is simply wrong
// by a row. TestComplementGateTotalityMutantIsCaught turns this on over a corpus
// with an OPTIONAL field and requires the differential harness to fail.
var admitGateSkipTotality bool

// buildAdmitGate arms s.gate for this query, or leaves it disabled. Called by
// searchIntoWith AFTER the filter-first planner has declined, under the read
// lock the traversal will hold for its whole life, with the plan the planner
// already collected (collectNarrowSets) and this query's k.
//
// Taking the plan rather than re-deriving it is what keeps the gate free to
// decline: every bail below costs a survey of maps that are already in hand, so
// the price of saying "no" — paid by every filtered graph query, most of which
// the gate cannot help — is a loop over a handful of set lengths. It also caps
// the gate by construction at what the planner was willing to materialize.
//
// Leaving the gate disabled is ALWAYS correct — it is exactly today's path — so
// every bail here is a throughput decision, not a correctness one.
func (h *hnsw) buildAdmitGate(s *layerScratch, f Filter, plan []narrowSet, k int) {
	if admitGateMode == gateOff || admitGateMode == gateForceComplement || len(plan) == 0 {
		return
	}
	// Survey the plan BEFORE committing to anything. The sets whose superset
	// property is not proven (the ordering family — see narrowClass) must be
	// DROPPED: the planner may intersect them because it re-checks every
	// candidate with the predicate, but the gate may not, because a set that came
	// back NARROWER than the predicate would make the gate pre-reject a matching
	// point and silently lose it. Dropping a set only widens the gate, which is
	// always safe.
	//
	// The survey allocates nothing and touches only set lengths. exactPlan tracks
	// whether every KEPT set is exact; combined with the leaf count below it is
	// what promotes the gate from pre-filter to replacement.
	nKept, nDropped, minSize := 0, 0, 0
	exactPlan := true
	for _, ns := range plan {
		if ns.class == narrowUnproven {
			nDropped++
			continue
		}
		if ns.class != narrowExact {
			exactPlan = false
		}
		if nKept == 0 || len(ns.set) < minSize {
			minSize = len(ns.set)
		}
		nKept++
	}
	if nKept == 0 {
		return // the whole plan was unproven: no gate
	}
	if admitGateMode != gateForce && !h.gateProfitable(minSize, nKept, k) {
		return
	}
	// Only now materialize the kept plan — and only when something was actually
	// dropped, since the common case keeps everything and can build in place.
	kept := plan
	if nDropped > 0 {
		kept = make([]narrowSet, 0, nKept)
		for _, ns := range plan {
			if ns.class != narrowUnproven {
				kept = append(kept, ns)
			}
		}
	}
	// EXACT needs all four: every kept set exact, every leaf narrowed (nothing
	// skipped or dropped), every op exactly answerable by its posting set, and no
	// per-key TTL anywhere in the arena (a hidden key makes liveMeta narrower
	// than what the index was built from). Any doubt degrades to SUPERSET, which
	// re-checks the predicate.
	exact := exactPlan &&
		len(kept) == filterLeafCount(f) &&
		filterIndexExact(f) &&
		h.arena.KeyDeadlineSlots() == 0
	s.gate.build(kept, h.arena.Capacity(), exact || admitGateForceExact)
	h.filterGates.Add(1)
}

// buildColumnGate arms the COLUMN oracle if the whole filter can be expressed
// as numeric range comparisons over columnised fields, and reports whether it
// did. It is tried before either bitset because its per-query cost is zero — the
// only work here is compiling bounds and looking up (or, once per field,
// building) the columns.
//
// NO COST MODEL, deliberately. The other two gates have to be priced because
// they trade a per-query BUILD against a per-candidate saving, and the build can
// dwarf the traversal. A column has no per-query build: the first range query on
// a field pays O(postings) once, every query after it pays nothing, and the
// per-candidate cost (one array index) is strictly below the metadata lookup it
// replaces. There is no regime in which arming this loses, so there is no
// decision to model — and a model that cannot say no is worse than none, because
// it invites tuning that has no effect.
//
// The per-key TTL veto is the one exactness precondition, and it is the same one
// the other two apply: liveMeta can hide a key the column still carries.
//
// THE QUOTA IS CHECKED HERE BECAUSE THIS IS THE ONLY PLACE THAT CAN. A column is
// allocated on the READ path, which is the one path Config.MaxBytes never used
// to constrain — estimateInsertBytes prices vectors and graph edges, and a
// collection sized against it could be pushed maxNumColumns·capacity·8 bytes
// past its ceiling by query traffic alone, without a single insert being
// rejected. The fix is two-sided and both halves are needed: this function
// refuses to BUILD a column that would cross the line, and insertLocked adds
// payloadIdx.columnBytes() to its own check so the columns already resident are
// charged against the same budget. Refusing to build only costs the
// acceleration; the predicate still answers.
func (h *hnsw) buildColumnGate(s *layerScratch, f Filter) bool {
	if admitGateMode == gateOff || admitGateMode == gateForceComplement {
		return false
	}
	if h.arena.KeyDeadlineSlots() != 0 {
		return false
	}
	// Negative = unlimited, which is what an unconfigured MaxBytes means.
	budget := int64(-1)
	if h.cfg.MaxBytes > 0 {
		budget = h.cfg.MaxBytes - h.bytesUsed - h.payloadIdx.columnBytes()
		if budget < 0 {
			budget = 0
		}
	}
	// Accumulate straight into the pooled term slice; a declined filter leaves
	// partial terms behind, which disable() clears on the way out — over the
	// slice's CAPACITY, because that is where they are (see disable).
	terms, ok := h.payloadIdx.collectColumnTerms(f, h.arena.Capacity(), budget, s.gate.cols[:0])
	if !ok || len(terms) == 0 {
		s.gate.cols = terms[:0]
		return false
	}
	s.gate.armColumns(terms)
	h.filterGates.Add(1)
	h.columnGates.Add(1)
	return true
}

// buildComplementGate is the m5 second attempt: the gate built from what the
// filter REJECTS rather than what it accepts. Called only when the positive path
// produced nothing — either the filter was not narrowable at all, or
// buildAdmitGate looked at the plan and declined.
//
// WHY A SECOND SHAPE EXISTS. The m4 gate's cost is O(accepted posting mass) and
// its saving is O(traversal), so it wins on SELECTIVE filters and must lose as
// the filter loosens. A 99%-pass range is the far end of that: 990k postings to
// index a traversal of ~10k candidates. But the same filter's rejection side is
// 1% of the corpus, and a bitset of rejections answers the identical question —
// admission is "bit still set". Profitability is therefore governed by
// min(accept mass, reject mass), and the m4 model only ever looked at one of
// them.
//
// It is an ALL-OR-NOTHING EXACT gate; see collectComplementSets for why a
// partial rejection union is useless here even though it would be sound.
//
// BUDGETS RATHER THAN AN ESTIMATE. The positive path can price itself because
// the planner already materialized its sets; the complement's sets do not exist
// until we build them, and building them to discover they were too expensive is
// exactly the waste the m5 key-count pre-check removed elsewhere. So we invert
// the model: compute what the saving is WORTH, turn it into a mass budget and a
// key budget, and hand those to orderingSetCapped as hard caps. The key budget
// is what makes the VectorDBBench shape cheap to REFUSE — 10k distinct ids
// against a budget of ~1k keys is proved over budget by one subtraction, with no
// map touched.
func (h *hnsw) buildComplementGate(s *layerScratch, f Filter, k int) {
	if admitGateMode == gateOff {
		return
	}
	n := h.arena.Size()
	if n <= 0 {
		return
	}
	// A per-key TTL can hide a key liveMeta still reads out of the arena, which
	// would make the predicate reject a slot the index says nothing about — i.e.
	// break R(f) ⊆ U, the half the EXACT admission rests on. Same veto, same
	// reason, as the positive path's exactness check.
	if h.arena.KeyDeadlineSlots() != 0 {
		return
	}
	// The traversal this would accelerate. ncand = n because the complement gate
	// is only ever reached for a filter the planner found non-selective (or could
	// not narrow at all), which is the regime where the ef-doubling ladder does
	// not fire and efEff sits at the base.
	visits := h.graphVisitEstimate(n, k)
	words := (h.arena.Capacity() + 63) / 64
	// EVERY visited candidate saves a predicate evaluation, not just the rejected
	// ones: an exact gate answers admit and reject alike from one bit. That is the
	// second way this model differs from the m4 one, whose saving term carries a
	// (1 - selectivity) factor because a superset gate only short-circuits
	// rejections.
	budget := int64(visits*gateVisitSaveDNS) - int64(words)*gateWordClearDNS
	if budget <= 0 {
		return
	}
	massLimit := int(budget / gatePostingDNS)
	keyLimit := int(budget / gateRangeKeyDNS)
	if admitGateMode == gateForce || admitGateMode == gateForceComplement {
		// "Cost be damned" — but still bounded by the arena, so a forced gate in a
		// test cannot wander into an unbounded union.
		massLimit, keyLimit = n+1, n+1
	}
	if keyLimit <= 0 {
		return
	}
	// THE TWO BUDGETS ARE ENFORCED DIFFERENTLY, and only one of them is divided.
	//
	//   - massLimit is enforced against a RUNNING TOTAL that appendComplementSets
	//     threads across every leaf of the filter, so it already accounts for an
	//     And paying for all of its conjuncts. Passing massLimit/leaves here would
	//     charge that division twice and cap the query at a `leaves`-fold tighter
	//     budget than the one just priced — which is safe (declining always is)
	//     but silently withholds the gate from the multi-conjunct ranges it is
	//     most useful for. It is passed WHOLE.
	//   - keyLimit is enforced PER LEAF, inside orderingPostings, as the O(1)
	//     `hi-lo > keyLimit` proof over that leaf's own key range. There is no
	//     running total to compare against — the check has to be answerable from
	//     two binary-search indices, which is the entire point of it — so the
	//     division is what bounds the sum across leaves. It is passed DIVIDED.
	leaves := filterLeafCount(f)
	if leaves <= 0 {
		return
	}
	rejects, ok := h.payloadIdx.collectComplementSets(f, n, massLimit, keyLimit/leaves)
	if !ok {
		return
	}
	// An EMPTY rejection list is armed, not skipped. It means the index found no
	// in-range key on the negated side — nothing is rejected — and over a field
	// the totality check just proved TOTAL that is a proof, not an absence of
	// information: every live slot carries a comparable value and none of them
	// falls on the reject side, so the filter matches everything and the whole
	// traversal can skip the predicate. (Without totality the same emptiness
	// would mean the opposite — a field nothing carries rejects everything —
	// which is why complementLeaf checks totality before it looks at any key, and
	// why this is the branch admitGateSkipTotality's mutant turns into visibly
	// wrong results.)
	s.gate.buildComplement(rejects, h.arena.Capacity())
	h.filterGates.Add(1)
	h.complementGates.Add(1)
}
