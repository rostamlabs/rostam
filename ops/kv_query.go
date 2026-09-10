// SPDX-License-Identifier: Apache-2.0

package ops

// The per-shard kv_query leaf.
//
// WHAT IT ANSWERS. Given a filter over record paths, one page of the keys this
// shard group holds that match it, in ascending key order, plus a continuation.
// With an index named, the answer is scoped to that definition's KeyPrefix (see
// the supersetness argument on SelectorFor); with scan: true it is the whole
// keyspace of this group.
//
// HOW IT IS EXACT DESPITE READING A HINT. The index narrows, it never decides.
// Postings are a SUPERSET of the true matches within the prefix, so every
// candidate is re-read from the cache and re-checked with the full compiled
// predicate on the LIVE value. A stale posting therefore costs one lookup and
// can never produce a wrong row; a MISSING posting is the only failure mode
// that would lose a row, and that is what SelectorFor's positive-leaf rule and
// the readiness gate exist to prevent.
//
// WHAT IT REFUSES RATHER THAN GUESSES. A candidate or scan budget it cannot pay
// is a typed error, never a truncated page: a caller cannot tell a truncated
// answer from a complete one, so silently shortening it would turn a budget
// into a correctness bug. The PAGE BYTE budget is different and does truncate —
// but it truncates with a continuation, which is a complete answer delivered in
// pieces.

import (
	"bytes"
	"container/heap"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops/kvindex"
	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

var (
	// ErrKVQueryScanRequired means no leaf of the filter can drive the named
	// index, and the caller did not pass scan: true. It is a REFUSAL, not a
	// fallback: a full-keyspace walk is orders of magnitude more expensive than
	// what the caller asked for, so consenting to it is the caller's decision.
	ErrKVQueryScanRequired = errors.New("ops: kv_query: filter needs an index or scan:true")
	// ErrKVQueryScanBudget means one page's walk visited more keys than the
	// node's scan budget allows. It refuses instead of truncating because an
	// UNORDERED walk cut short yields no sound continuation — the keys it did
	// not reach are indistinguishable from keys that did not match, so the next
	// page could not know where to resume. Use an index.
	ErrKVQueryScanBudget = errors.New("ops: kv_query: scan budget exceeded; use an index")
	// ErrKVIndexUnavailable means this dispatcher was built without a KV record
	// index. Every embedder that predates the index has none, and kv_query is
	// the one op that cannot work without one.
	ErrKVIndexUnavailable = errors.New("ops: kv_query: no KV index on this dispatcher")
	// ErrKVQueryUnavailable marks a refusal that is RETRYABLE and about this
	// replica rather than about the query: a scan's walk was cut short, which
	// today means kvindex.ErrWalkAborted — the walk gate stopping the walk
	// because this shard is being removed from the node. The coordinator
	// classifies it like kvindex.ErrIndexBuilding — try again, elsewhere or
	// later.
	//
	// The GATING IS SUPPLIED BY THE STORE, not by this package: the leaf walks
	// through TxContext.Walker(), and the store/cluster layer installs the
	// gated walker there (shard.Store.SetKVWalker). A dispatcher with no gate
	// installed walks its cache directly and this error is simply never
	// produced.
	//
	// It is NEVER a short page. The keys an aborted walk did not reach are
	// indistinguishable from keys that did not match, so a page built from a
	// partial walk would be a silently wrong answer rather than a slow one.
	//
	// EVERY non-nil walk error lands here, not just the ones named above: a
	// walker that fails is a walker whose result cannot be trusted, whatever it
	// failed with.
	ErrKVQueryUnavailable = errors.New("ops: kv_query: shard is unavailable; retry")
)

// KVQueryBudget bounds the work ONE page of a kv_query may do on one shard.
//
// These are NODE CONFIG, not call arguments, and deliberately so: kv_query is
// read-only and never applied, so a per-node value cannot diverge committed
// state. A caller cannot raise them.
//
//   - Candidates caps the posting keys a selector may union (kvindex charges
//     every key it copies and, for a range, every distinct value it examines).
//   - Scan caps the keys ONE scan page may visit, matching or not.
//   - ScanChunk is how many keys one scan page carries forward: the size of the
//     bounded heap that makes an unordered walk pageable.
type KVQueryBudget struct{ Candidates, Scan, ScanChunk int }

// defaultKVQueryBudget is sized so the common answer is served and the
// pathological one is refused: 250k candidate keys is a large answer and a
// small allocation, 5M visited keys is about a second of walking, and a 10k
// chunk pages a million-key shard in a hundred pages.
var defaultKVQueryBudget = KVQueryBudget{Candidates: 250_000, Scan: 5_000_000, ScanChunk: 10_000}

var (
	kvQueryBudgetMu sync.RWMutex
	kvQueryBudgetV  = defaultKVQueryBudget
)

// SetKVQueryBudget installs the node's kv_query budget. A non-positive field
// falls back to its default: a zero budget is a misconfiguration, and honouring
// it literally would refuse every query (or, for ScanChunk, page forever
// without advancing), which is a worse answer than the default.
func SetKVQueryBudget(b KVQueryBudget) {
	if b.Candidates <= 0 {
		b.Candidates = defaultKVQueryBudget.Candidates
	}
	if b.Scan <= 0 {
		b.Scan = defaultKVQueryBudget.Scan
	}
	if b.ScanChunk <= 0 {
		b.ScanChunk = defaultKVQueryBudget.ScanChunk
	}
	kvQueryBudgetMu.Lock()
	kvQueryBudgetV = b
	kvQueryBudgetMu.Unlock()
}

func kvQueryBudget() KVQueryBudget {
	kvQueryBudgetMu.RLock()
	defer kvQueryBudgetMu.RUnlock()
	return kvQueryBudgetV
}

// kvQueryPageOverhead is the room reserved inside wire.KVQueryMaxPageBytes for
// everything a page carries that is not row payload: the row count, the
// continuation block, and one 64 KiB key inside it (the encoder's own key cap).
// Reserving it is what makes "the rows fit" imply "the frame encodes".
const kvQueryPageOverhead = 4 + 2 + (4 + 1 + 2 + 0xFFFF)

// kvQueryScanHeapMaxBytes bounds the KEY BYTES one scan page's chunk heap
// holds, independently of ScanChunk. ScanChunk bounds the key COUNT, which is
// only a memory bound if keys are small: at the default chunk of 10 000 and the
// 64 KiB the encoder allows a key, a count bound alone permits 625 MiB.
const kvQueryScanHeapMaxBytes = 32 << 20

// kvQueryOversizeRows counts the rows this process has emitted WITHOUT their
// value because the value alone exceeded the page cap (see verifyPage's oversize
// branch).
//
// A COUNTER RATHER THAN A LOG LINE, and the difference is who controls the
// volume. kv_query is a client-driven read: a caller querying a keyspace with
// large values can produce this condition on every page of every query, from
// every connection, for as long as it likes — a log line there is a client with
// a write handle on the operator's disk. The number is the part an operator
// acts on ("values are being omitted; fetch those keys with get"), and it is
// exactly as informative when read once a minute as when printed thousands of
// times a second.
var kvQueryOversizeRows atomic.Uint64

// KVQueryOversizeRows reports how many kv_query rows this process has returned
// with the value omitted because it did not fit a page. Monotonic since start;
// a rising rate means callers are being handed keys they must fetch with get.
func KVQueryOversizeRows() uint64 { return kvQueryOversizeRows.Load() }

// handleKVQuery answers one shard group's page of a kv_query.
func handleKVQuery(tx *TxContext, args []byte) ([]byte, error) {
	a, err := wire.DecodeKVQueryArgs(args)
	if err != nil {
		return nil, err
	}
	idx := tx.KVIndex()
	if idx == nil {
		return nil, ErrKVIndexUnavailable
	}
	group := kvQueryGroup(tx)
	after := contFor(a.Cursor, group)

	// Compiled BEFORE any candidate work: a filter this leaf will not accept is
	// a query error, and finding that out after walking the keyspace would be
	// the same answer at a much higher price.
	pred, err := BuildKVPredicate(a.Filter)
	if err != nil {
		return nil, err
	}
	b := kvQueryBudget()

	if a.Index == "" {
		// The decoder already rejected index == "" && !scan, so this is a
		// consented scan and nothing else.
		return scanPage(tx, tx.Walker(), after, pred, a, group, b)
	}

	cands, err := kvQueryCandidates(idx, a, after, group, b)
	switch {
	case errors.Is(err, errKVQuerySelectorless):
		// The index exists and is ready, but no leaf of this filter can drive
		// it. Only now — after the readiness gate — may a scan stand in, and
		// only with the caller's consent.
		if !a.Scan {
			return nil, ErrKVQueryScanRequired
		}
		return scanPage(tx, tx.Walker(), after, pred, a, group, b)
	case err != nil:
		return nil, err
	}
	return verifyPage(tx, idx, cands, after, pred, a, group, false)
}

// errKVQuerySelectorless is internal: it reports "the index is usable but this
// filter cannot drive it", which is the one case the caller answers with a scan
// (when the caller consented) rather than with the error itself.
var errKVQuerySelectorless = errors.New("ops: kv_query: no candidate selector for this index")

// kvQueryCandidates resolves the definition, gates on readiness, extracts the
// selector and reads the candidate keys — retrying ONCE if the definition moved
// under the query.
//
// WHY THE RETRY. kvindex.ErrIndexChanged means the installed definition's shape
// no longer matches the one the selector was built from, so those postings
// answer a DIFFERENT question and are not a superset of this predicate. It is
// handled exactly like ErrIndexBuilding: re-read the definition once and try
// again, and if it is still moving, return the retryable refusal. What it must
// never do is fall through to a scan — that would silently turn a cheap query
// into a full walk the caller never consented to — or leak ErrIndexChanged,
// which says nothing actionable to a client.
func kvQueryCandidates(idx *kvindex.Set, a wire.KVQueryArgs, after []byte, group uint32, b KVQueryBudget) ([][]byte, error) {
	for attempt := 0; attempt < 2; attempt++ {
		def, ok := idx.Lookup(a.Index)
		if !ok {
			// PERMANENT here on purpose: this group has no such definition. The
			// COORDINATOR is the layer that knows whether the name exists in the
			// meta catalog and, if it does, converts this into the retryable
			// ErrIndexBuilding.
			return nil, fmt.Errorf("%w: %q on shard group %d", kvindex.ErrNoSuchIndex, a.Index, group)
		}
		// PER REPLICA, not per group: readiness is answered by ONE replica, so a
		// stale read landing on a replica still backfilling must refuse rather
		// than answer from a proper subset of the postings.
		if !idx.IsReady(a.Index) {
			return nil, fmt.Errorf("%w: %q on shard group %d (backfill in progress; retry)", kvindex.ErrIndexBuilding, a.Index, group)
		}
		sel, ok := SelectorFor(a.Filter, def)
		if !ok {
			return nil, errKVQuerySelectorless
		}
		cands, err := idx.Candidates(sel, after, b.Candidates)
		if err == nil {
			// SORTED HERE, not merely assumed. Candidates already returns a
			// sorted set, but ascending key order IS this leaf's contract, so it
			// does not rest on a neighbour's postcondition; on an already-sorted
			// slice this is one linear scan.
			sort.Slice(cands, func(i, j int) bool { return bytes.Compare(cands[i], cands[j]) < 0 })
			return cands, nil
		}
		if !errors.Is(err, kvindex.ErrIndexChanged) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("%w: %q on shard group %d (definition changed under the query; retry)",
		kvindex.ErrIndexBuilding, a.Index, group)
}

// kvQueryGroup reports the shard group this dispatcher serves, as the cursor
// keys on it.
//
// Direct never calls SetShardIndex, so ShardIndex() is 0 there — and its single
// cache IS group 0, so that is correct rather than a fallback. NoShardIndex (-1)
// only ever comes from a nil receiver, which a dispatched handler cannot have;
// it is mapped to 0 here rather than converted, because a negative int widened
// to uint32 would name group 4294967295 and quietly mismatch every cursor.
func kvQueryGroup(tx *TxContext) uint32 {
	i := tx.ShardIndex()
	if i < 0 {
		return 0
	}
	return uint32(i) //nolint:gosec // non-negative, and bounded by the node's shard count
}

// contFor returns the exclusive resume key this group's cursor carries, or nil
// when the cursor does not mention this group (a first page, or a group that
// finished on an earlier one).
//
// The cursor is in strictly increasing Group order — DecodeKVQueryArgs enforces
// it — so this is a binary search rather than a scan over up to 4096 entries.
//
// More is deliberately IGNORED. It describes what the SERVER said last time,
// and reading it as "this group is finished" would make the page's contents
// depend on a flag the caller can set freely. After is the only thing that
// changes what this page contains.
func contFor(conts []wire.KVQueryCont, group uint32) []byte {
	i := sort.Search(len(conts), func(i int) bool { return conts[i].Group >= group })
	if i < len(conts) && conts[i].Group == group {
		return conts[i].After
	}
	return nil
}

// verifyPage is the shared verify-and-emit step: it re-reads each candidate in
// ascending key order, re-checks the full predicate on the LIVE value, and
// emits rows until the row limit or the page byte budget stops it.
//
// IT HOLDS NO INDEX LOCK, and that is not a style preference. tx.Get of an
// expired key on a non-replicated shard calls dropExpiredLocked under the cache
// shard's WRITE lock, which fires onRemove -> kvindex.Set.Drop -> Set.mu. A
// caller holding any Set lock across this loop self-deadlocks single-threaded,
// with no concurrency required. Candidates has already copied what it needed
// and released the lock; nothing here re-acquires it.
//
// The clock is the WALL clock, always: kv_query is OpReadOnly and never runs
// under an apply, so there is no leader stamp to judge expiry against and
// tx.Get's unstamped branch is the only one reachable.
//
// idx is the Set to charge verify misses to, and is nil on the scan path: a key
// that vanished between the walk and the re-read is not a stale posting, and
// counting it as one would make the staleness signal meaningless.
//
// truncated says the CALLER already dropped keys it could not carry (the scan
// chunk overflowed), so the page is incomplete however few rows match.
//
// A row whose value is too large for any page is emitted WITHOUT the value
// rather than dropped (see the oversize branch): a kv_query page caps at 8 MiB
// while a cache value may be up to the 16 MiB page size.
func verifyPage(tx *TxContext, idx *kvindex.Set, keys [][]byte, after []byte, pred vector.Predicate, a wire.KVQueryArgs, group uint32, truncated bool) ([]byte, error) {
	// DecodeKVQueryArgs already refuses a limit outside 1..KVQueryMaxLimit, so
	// this clamp is defence in depth against a future caller that builds args
	// by hand: a limit of 0 emits no row, so the continuation never advances
	// and the caller pages forever.
	limit := int(a.Limit)
	if limit < 1 {
		limit = 1
	}
	byteBudget := wire.KVQueryMaxPageBytes - kvQueryPageOverhead

	res := wire.KVQueryResult{}
	used, misses, oversize := 0, 0, 0
	// The continuation starts at the incoming cursor: a page that examines
	// nothing has made no progress and must not claim any.
	cont := append([]byte(nil), after...)

	// ONE metadata map for the whole page, its single entry rewritten per
	// candidate. A fresh map per candidate is a map allocation per key, which
	// on the scan path is one per key in the CHUNK rather than one per row —
	// the dominant allocation of a page that matches nothing. Predicates read
	// the map synchronously and never retain it (vector.Predicate), so reusing
	// it is safe.
	var meta vector.Metadata
	if pred != nil {
		meta = make(vector.Metadata, 1)
	}

	i := 0
	for ; i < len(keys); i++ {
		if len(res.Rows) >= limit {
			break
		}
		k := keys[i]
		// DEFENSIVE, and cheap: both producers promise keys strictly ABOVE the
		// cursor (Candidates filters on it, the scan chunk's walk filters on it).
		// Trusting that silently is what makes a bug in either one unbounded — a
		// key at or below the cursor would set `cont` BACKWARDS, and the caller
		// would then re-request a page it has already seen, forever. Skipping it
		// without touching `cont` leaves the continuation monotonic whatever the
		// producer did.
		if len(after) > 0 && bytes.Compare(k, after) <= 0 {
			continue
		}
		v, err := tx.Get(k)
		if errors.Is(err, cache.ErrNotFound) {
			// A posting for a key the cache no longer holds. Harmless — this is
			// exactly the staleness the re-read exists to absorb — but counted,
			// because a rising rate means keys are leaving by a path that does
			// not reach Drop.
			misses++
			cont = k
			continue
		}
		if err != nil {
			return nil, err
		}
		// v ALIASES THE CACHE'S PAGE BACKING STORE and stays valid only until
		// the next call into the cache — which is the next iteration's Get.
		// Everything between here and the copy below reads it synchronously
		// (the predicate resolves the record and returns; DecodeRecord's result
		// is discarded), so the copy is deferred to the row that is actually
		// EMITTED. Nothing aliasing v survives this iteration, which is the
		// invariant handleGetDel states as "copy before the next call".
		//
		// The predicate NEVER sees the raw record as a value it can compare:
		// it sees a map whose single entry is the reserved alias, and every
		// leaf's field addresses it through a "$rec/..." path
		// (BuildKVPredicate refuses a field that would name the alias itself).
		if pred != nil {
			meta[KVRecordAlias] = vtypes.Value{Kind: vtypes.ValueRecord, Rec: v}
			if !pred(meta) {
				cont = k
				continue
			}
		}
		if a.Return == wire.KVQueryReturnRecords {
			if _, derr := wire.DecodeRecord(v); derr != nil {
				// A non-record value under `records` is SKIPPED, not an error:
				// the KV keyspace is shared, and one unrelated value must not
				// fail a query over the records around it.
				cont = k
				continue
			}
		}

		cost := 2 + len(k) + 1
		withValue := a.Return != wire.KVQueryReturnKeys
		if withValue {
			cost += 4 + len(v)
			if cost > byteBudget {
				// OVERSIZE: this row cannot fit a page even alone. A cache value
				// may be up to the 16 MiB page size while a kv_query page caps
				// at 8 MiB, so this is reachable with ordinary data.
				//
				// The row is emitted WITHOUT its value rather than dropped or
				// refused, because both alternatives are worse than useless:
				// dropping it loses a true match silently, and refusing the page
				// wedges the query — EncodeKVQueryResult would reject the frame,
				// the handler would return an error with NO continuation, and
				// every retry would re-examine the same key, so every key
				// sorting after it becomes unreachable forever.
				//
				// hasVal = false is unambiguous here: in values/records mode
				// every ordinary row carries a non-nil value (a zero-length
				// stored value encodes as a present, empty one), so a row with
				// no value means exactly "omitted, larger than the page cap;
				// fetch it with get". See EncodeKVQueryResult's doc.
				withValue = false
				cost = 2 + len(k) + 1
				oversize++
			}
		}
		// Any OTHER row that does not fit is deferred to the next page, where it
		// will be first and will fit. The first row of a page is therefore always
		// emitted, so a page always advances and paging always terminates.
		if len(res.Rows) > 0 && used+cost > byteBudget {
			break
		}
		row := wire.KVQueryRow{Key: k}
		if withValue {
			// The copy, taken only now that the row is certain to be emitted,
			// and before the next Get can overwrite, free or unmap the page.
			row.Value = make([]byte, len(v))
			copy(row.Value, v)
		}
		res.Rows = append(res.Rows, row)
		used += cost
		// Set only now: a key whose row did not fit has NOT been accounted for
		// and must be re-examined on the next page.
		cont = k
	}

	if idx != nil {
		idx.NoteVerifyMisses(misses)
	}
	if oversize > 0 {
		kvQueryOversizeRows.Add(uint64(oversize)) //nolint:gosec // a count of rows within one page, never negative
	}
	// More when the caller already dropped keys, or when this loop stopped
	// early. i == len(keys) with truncated false is the only complete answer.
	if truncated || i < len(keys) {
		res.Cursor = []wire.KVQueryCont{{Group: group, After: cont, More: true}}
	}
	return wire.EncodeKVQueryResult(res)
}

// scanPage answers one page WITHOUT an index, by making an unordered walk
// pageable.
//
// THE PROBLEM. Cache.Iterate takes no start key and the hash table has no
// order, so a scan page has to walk the whole shard whatever the cursor says.
// A walk cut short cannot yield a sound continuation either, because the keys
// it did not reach are indistinguishable from keys that did not match.
//
// THE SHAPE. One pass keeps, in a bounded MAX-heap of ScanChunk keys, the
// SMALLEST ScanChunk keys strictly greater than `after`, charging every visited
// key — matching or not, above the cursor or not — against the scan budget. The
// chunk is then drained ascending into verifyPage. When the heap overflowed the
// page is marked More even if fewer than `limit` rows matched, and the
// continuation is the largest key the chunk retained, so the next page resumes
// there. A keyspace of any size below the scan budget therefore pages to
// completion in about ceil(n / ScanChunk) pages, each costing one full walk —
// quadratic, and honestly so. Use an index.
func scanPage(tx *TxContext, walk kvindex.Walker, after []byte, pred vector.Predicate, a wire.KVQueryArgs, group uint32, b KVQueryBudget) ([]byte, error) {
	h := &kvKeyHeap{max: b.ScanChunk, maxBytes: kvQueryScanHeapMaxBytes}
	visited := 0
	overBudget := false
	// dropped records that at least one key above the cursor did not fit the
	// chunk, which is what makes this page's More true regardless of how many
	// rows matched.
	dropped := false

	werr := walk(func(key, _ []byte) bool {
		visited++
		if visited > b.Scan {
			overBudget = true
			return false
		}
		if len(after) > 0 && bytes.Compare(key, after) <= 0 {
			return true
		}
		if !h.offer(key) {
			dropped = true
		}
		return true
	})
	// The walk error comes FIRST: a cache closing under the walk makes the
	// visited count meaningless, and a page built from a partial walk would be
	// a silently short answer.
	if werr != nil {
		return nil, fmt.Errorf("%w: scan walk: %w", ErrKVQueryUnavailable, werr)
	}
	if overBudget {
		return nil, fmt.Errorf("%w: visited more than %d keys on shard group %d", ErrKVQueryScanBudget, b.Scan, group)
	}
	// idx is nil: a key that vanished between the walk and the re-read is not a
	// stale posting, so it is not charged to the index's staleness counter.
	return verifyPage(tx, nil, h.ascending(), after, pred, a, group, dropped)
}

// kvKeyHeap is the bounded MAX-heap scanPage carries a chunk in: it holds the
// smallest keys offered to it, so the largest is the one to evict, which is why
// the ordering is inverted.
//
// THE INVARIANT, and everything depends on it: whatever the heap holds is
// always the SMALLEST m keys of everything offered so far, for the current m.
// scanPage's continuation is the largest key retained, so every key at or below
// it must have been examined — and that is true exactly when the retained set
// is a prefix of the sorted offer set. It is maintained by only ever discarding
// the current MAXIMUM (or a key that does not sort below it).
//
// This is why a byte cap cannot simply refuse an offer. Refusing a key that
// sorts BELOW the retained maximum would leave a hole under the continuation,
// and that key would never be visited again on any later page — a lost row, not
// a slow one. So when a key belongs in the chunk, it goes in, and the cap is
// then restored by evicting from the TOP.
//
// It bounds both the key count (max) and the key bytes (maxBytes), and always
// keeps at least one key, so a page always advances and paging terminates.
type kvKeyHeap struct {
	keys     [][]byte
	max      int
	maxBytes int
	bytes    int
}

func (h *kvKeyHeap) Len() int           { return len(h.keys) }
func (h *kvKeyHeap) Less(i, j int) bool { return bytes.Compare(h.keys[i], h.keys[j]) > 0 }
func (h *kvKeyHeap) Swap(i, j int)      { h.keys[i], h.keys[j] = h.keys[j], h.keys[i] }

func (h *kvKeyHeap) Push(x any) {
	k, _ := x.([]byte)
	h.keys = append(h.keys, k)
}

func (h *kvKeyHeap) Pop() any {
	last := len(h.keys) - 1
	k := h.keys[last]
	h.keys = h.keys[:last]
	return k
}

// offer adds key to the chunk if it belongs there, reporting whether nothing
// was dropped. A false result means a key above the cursor was discarded, which
// is what the page's More is derived from.
//
// The key is COPIED: the walk's slice aliases the cache's page backing store
// and is valid only for the duration of the callback.
func (h *kvKeyHeap) offer(key []byte) bool {
	// Room by COUNT: take it, then restore the byte cap from the top. The
	// len == 0 arm also makes a heap with a nonsensical max of 0 hold one key
	// rather than index an empty slice below.
	if len(h.keys) == 0 || len(h.keys) < h.max {
		h.pushCopy(key)
		return h.shrinkToCap()
	}
	// Full by count. The root is the largest key held, so a key that does not
	// sort below it is itself the one dropped.
	if bytes.Compare(key, h.keys[0]) >= 0 {
		return false
	}
	// It sorts below the maximum, so it BELONGS in the chunk (see the type's
	// invariant): the maximum makes way for it, whatever that does to the byte
	// total, which shrinkToCap then restores by evicting from the top.
	h.bytes -= len(h.keys[0])
	h.keys[0] = append([]byte(nil), key...)
	h.bytes += len(h.keys[0])
	heap.Fix(h, 0)
	h.shrinkToCap()
	return false // the displaced maximum was dropped
}

func (h *kvKeyHeap) pushCopy(key []byte) {
	cp := append([]byte(nil), key...)
	h.bytes += len(cp)
	heap.Push(h, cp)
}

// shrinkToCap evicts the LARGEST keys until the byte cap holds again, keeping
// at least one whatever its size (a single key is bounded by the encoder's
// 64 KiB key cap, and a chunk of none would page forever without advancing).
// Evicting from the top is what preserves the smallest-m-keys invariant.
//
// It reports whether nothing was evicted, so the caller can mark the page.
func (h *kvKeyHeap) shrinkToCap() bool {
	kept := true
	for h.bytes > h.maxBytes && len(h.keys) > 1 {
		k, _ := heap.Pop(h).([]byte)
		h.bytes -= len(k)
		kept = false
	}
	return kept
}

// ascending drains the chunk into a sorted slice, which is the order verifyPage
// requires and the order the page is emitted in.
func (h *kvKeyHeap) ascending() [][]byte {
	out := make([][]byte, len(h.keys))
	for i := len(out) - 1; i >= 0; i-- {
		k, _ := heap.Pop(h).([]byte)
		out[i] = k
	}
	return out
}
