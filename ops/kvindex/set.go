// SPDX-License-Identifier: Apache-2.0

package kvindex

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/rostamlabs/rostam/sdk/record"
	"github.com/rostamlabs/rostam/sdk/vtypes"
)

var (
	// ErrNoSuchIndex means the named definition is not installed on this node.
	ErrNoSuchIndex = errors.New("kvindex: no such index")
	// ErrIndexBuilding means the definition is installed but its backfill has
	// not finished. Its postings are a proper SUBSET of the truth until it
	// does, so serving a query from them would lose rows — refusing is the
	// only sound answer.
	ErrIndexBuilding = errors.New("kvindex: index is still building")
	// ErrCandidateBudget means the selector's candidate set is larger than the
	// node's budget. It is a refusal, never a truncated page: a truncated
	// candidate set is indistinguishable from a complete one at the caller.
	ErrCandidateBudget = errors.New("kvindex: candidate budget exceeded")
)

// keySet is the set of cache keys posted under one scalar value. Keys are
// held as strings so the map owns its bytes: the []byte a caller hands
// Reindex belongs to the cache (or to a decoder's buffer) and may be reused
// the moment the call returns.
type keySet = map[string]struct{}

// posting is one definition's index: the forward map (value -> keys) queries
// read, the reverse map (key -> value) that makes a drop O(1), and the
// readiness flag.
//
// The reverse map is what keeps Drop cheap, and Drop's cost is not
// negotiable: it runs under a cache shard's WRITE lock (the onRemove hook),
// so a scan of the forward map there would block a shard for as long as the
// index is large.
type posting struct {
	vals  map[scalarKey]keySet
	keys  map[string]scalarKey
	ready bool
}

func newPosting() *posting {
	return &posting{vals: make(map[scalarKey]keySet), keys: make(map[string]scalarKey)}
}

// set posts key under sk, removing whatever it was posted under before.
//
// The key stays a []byte until a copy is genuinely needed. Every read here is
// a m[string(b)] / delete(m, string(b)) form, which the compiler lowers
// WITHOUT allocating the string, so the two commonest write outcomes — a
// rewrite that does not move the field, and a key that has no posting to drop
// — cost zero allocations on the KV write path. Only an actual insert
// allocates, and it allocates ONCE for both maps.
func (p *posting) set(key []byte, sk scalarKey) {
	if old, ok := p.keys[string(key)]; ok {
		if old == sk {
			return
		}
		p.removeFrom(key, old)
	}
	s := p.vals[sk]
	if s == nil {
		s = make(keySet)
		p.vals[sk] = s
	}
	ks := string(key) // the one copy the maps own
	s[ks] = struct{}{}
	p.keys[ks] = sk
}

// drop removes key's posting, if it has one.
func (p *posting) drop(key []byte) bool {
	old, ok := p.keys[string(key)]
	if !ok {
		return false
	}
	delete(p.keys, string(key))
	p.removeFrom(key, old)
	return true
}

// removeFrom takes key out of sk's set and deletes the set when it empties.
// Deleting the empty set is not tidiness: a range walk is O(distinct values),
// so leftover empty entries would make every range query slower for as long
// as the process lives, and would keep the vacated string alive with them.
func (p *posting) removeFrom(key []byte, sk scalarKey) {
	s := p.vals[sk]
	if s == nil {
		return
	}
	delete(s, string(key))
	if len(s) == 0 {
		delete(p.vals, sk)
	}
}

func (p *posting) reset() {
	p.vals = make(map[scalarKey]keySet)
	p.keys = make(map[string]scalarKey)
}

// Set is a shard's whole KV index: every installed definition and its
// postings. Safe for concurrent use.
//
// LOCK ORDER, stated once and obeyed everywhere in this file: cache lock ->
// Set.mu, one way. Drop is called with a cache shard's write lock held, so no
// method here may hold s.mu across anything that could take a cache lock. See
// the package doc.
type Set struct {
	mu    sync.RWMutex
	defs  []Def
	posts map[string]*posting
	res   *record.Resolver
}

// New returns an empty Set whose resolver caches at most resolverCache
// decoded schemas. A shard sees few distinct schemas and reuses them on every
// write, so the cache turns a schema-mode resolve into offset arithmetic; a
// value below 1 disables it (every resolve decodes the schema afresh, which
// is slower and answers identically).
func New(resolverCache int) *Set {
	return &Set{posts: make(map[string]*posting), res: record.NewResolver(resolverCache)}
}

// Install replaces the definition set. A definition whose Name, Prefix,
// PathText and Kind are all unchanged KEEPS its postings and its readiness —
// nothing about what it posts changed, so a backfill would rebuild exactly
// what is already there. Any other definition (new, or one whose path or
// prefix moved) starts empty and NOT ready, because its old postings describe
// a different question.
//
// A duplicate name is ignored (the first wins), so posts and defs cannot
// disagree about which definition a name means.
func (s *Set) Install(defs []Def) {
	s.mu.Lock()
	defer s.mu.Unlock()

	old := make(map[string]Def, len(s.defs))
	for _, d := range s.defs {
		old[d.Name] = d
	}
	next := make([]Def, 0, len(defs))
	posts := make(map[string]*posting, len(defs))
	for _, d := range defs {
		if _, dup := posts[d.Name]; dup {
			continue
		}
		p := s.posts[d.Name]
		if p == nil || !sameShape(old[d.Name], d) {
			p = newPosting()
		}
		posts[d.Name] = p
		next = append(next, d)
	}
	s.defs, s.posts = next, posts
}

// Defs returns the installed definitions in install order. The slice is a
// copy; the Defs in it are immutable and must be treated as such (their
// Prefix and Path are shared with the Set).
func (s *Set) Defs() []Def {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]Def(nil), s.defs...)
}

// Lookup returns the definition named name.
func (s *Set) Lookup(name string) (Def, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := range s.defs {
		if s.defs[i].Name == name {
			return s.defs[i], true
		}
	}
	return Def{}, false
}

// IsReady reports whether name's backfill has completed. An index that is not
// installed is not ready.
func (s *Set) IsReady(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p := s.posts[name]
	return p != nil && p.ready
}

// MarkReady records that name's postings now cover every live key.
func (s *Set) MarkReady(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if p := s.posts[name]; p != nil {
		p.ready = true
	}
}

// Reindex brings every definition's posting for key into line with value.
//
// It never errors and never touches the cache: a value that is not a record,
// a record this build cannot decode, and a path that names nothing all mean
// exactly one thing — no posting for this key under this definition — and the
// old posting is dropped so a rewrite can never leave a key posted under a
// value it no longer holds.
//
// Call it AFTER the store write returns. An evicting Put can fire onRemove
// for the very key being written (the cache's retirePageLocked, cur == ref),
// so reindexing first would post a key the eviction then drops.
//
// It posts regardless of readiness, on purpose: activation is
// enable-then-backfill, so a write racing a backfill must be recorded, and a
// posting can then only be STALE (harmless — one wasted lookup), never
// missing (a lost row).
func (s *Set) Reindex(key, value []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.defs {
		d := &s.defs[i]
		if !bytes.HasPrefix(key, d.Prefix) {
			// Out of scope for this definition. A key's prefix membership never
			// changes, so there is nothing here to drop either.
			continue
		}
		p := s.posts[d.Name]
		if p == nil {
			continue
		}
		s.reindexOne(p, d, key, value)
	}
}

// reindexOne posts (or unposts) one key under one definition. Caller holds
// s.mu for writing and has already checked the prefix.
func (s *Set) reindexOne(p *posting, d *Def, key, value []byte) {
	v, ok := s.resolveValue(d, value)
	if !ok {
		p.drop(key)
		return
	}
	sk, ok := scalarKeyOf(v)
	if !ok {
		// Not equality-indexable (NaN, a list kind, a record blob): nothing can
		// query it, so nothing posts it. See scalarKeyOf.
		p.drop(key)
		return
	}
	p.set(key, sk)
}

// resolveValue resolves d's path against value and maps the result to the
// payload-shaped Value a filter compares against — the SAME mapping the
// predicate side uses (record.ResultValue), which is what makes the postings
// agree with the predicate they narrow.
func (s *Set) resolveValue(d *Def, value []byte) (vtypes.Value, bool) {
	res, err := s.res.Resolve(value, d.Path)
	if err != nil {
		// Not a record, a truncated/corrupt one, a #count on a non-table field,
		// a by-name path against a schema that stores no names: every one of
		// these is "no posting", never a failure the caller must handle.
		return vtypes.Value{}, false
	}
	// Absent, Table and RowPresent carry no comparable value; ResultValue says
	// so for all three, so the Kind switch lives there and not here.
	return record.ResultValue(res)
}

// Drop removes key from every definition's postings.
//
// It is the cache's onRemove hook, so it runs UNDER a cache shard's write
// lock: it takes only s.mu, never calls back into the cache, and does O(number
// of definitions) map deletes and nothing else.
func (s *Set) Drop(key []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.posts {
		p.drop(key)
	}
}

// Reset drops every posting and KEEPS readiness.
//
// Reset is what a flush calls, and a flush empties the cache: an empty
// posting set over an empty keyspace is EXACT, not partial. Clearing
// readiness here would strand every definition in `building` until an
// unrelated meta write happened to move the observer, turning every kv_query
// into ErrIndexBuilding for as long as that took.
func (s *Set) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.posts {
		p.reset()
	}
}

// Rebuild clears every posting and replays the whole keyspace through walk,
// then marks every definition ready. Used on a warm start, after a restore,
// and whenever a definition set is installed from scratch.
//
// s.mu IS NOT HELD ACROSS walk. walk takes cache locks, and a Set method
// holding s.mu while a cache lock is taken closes the lock cycle Drop opens
// from the other side. Each visited entry pays one short-locked Reindex
// instead. The consequence is that the walk is not atomic — a concurrent
// write may be seen by both the walk and its own Reindex — which is harmless,
// because Reindex is idempotent in (key, value). The cache's chunked walk can
// visit a key twice for the same reason (it restarts on a table swap).
func (s *Set) Rebuild(walk func(func(key, value []byte) bool)) {
	s.mu.Lock()
	started := make(map[string]*posting, len(s.posts))
	for name, p := range s.posts {
		p.reset()
		// NOT ready until the walk finishes: a query served from a half-filled
		// posting set is a proper subset of the truth, which is the one answer
		// this index may never give.
		p.ready = false
		started[name] = p
	}
	s.mu.Unlock()

	if walk != nil {
		walk(func(key, value []byte) bool {
			s.Reindex(key, value)
			return true
		})
	}

	s.mu.Lock()
	for name, p := range started {
		// Only the postings this walk actually filled are marked ready. An
		// Install during the walk swaps in a FRESH posting for any definition
		// whose shape moved, and that one has seen only the tail of the
		// keyspace: marking it ready would publish a proper subset. It stays
		// building until its own backfill runs.
		if s.posts[name] == p {
			p.ready = true
		}
	}
	s.mu.Unlock()
}

// Backfill is Rebuild for a single definition: the one a meta write just
// added. Unknown names are a no-op. The other definitions keep their postings
// and their readiness throughout.
func (s *Set) Backfill(name string, walk func(func(key, value []byte) bool)) {
	s.mu.Lock()
	p := s.posts[name]
	if p == nil {
		s.mu.Unlock()
		return
	}
	p.reset()
	p.ready = false
	s.mu.Unlock()

	if walk != nil {
		walk(func(key, value []byte) bool {
			s.reindexFor(name, p, key, value)
			return true
		})
	}

	s.mu.Lock()
	// Same guard as Rebuild's: if an Install replaced this definition while the
	// walk ran, the posting we filled is no longer the installed one, and the
	// one that IS installed has not been backfilled.
	if s.posts[name] == p {
		p.ready = true
	}
	s.mu.Unlock()
}

// reindexFor is Reindex scoped to one definition — the backfill's inner step.
// Reindexing every definition there would be correct but would pay for
// resolving paths whose postings are already complete.
func (s *Set) reindexFor(name string, want *posting, key, value []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if p := s.posts[name]; p == nil || p != want {
		// Replaced (or removed) mid-backfill: this walk is stale and must not
		// write into the definition that took its place.
		return
	}
	for i := range s.defs {
		d := &s.defs[i]
		if d.Name != name {
			continue
		}
		if !bytes.HasPrefix(key, d.Prefix) {
			return
		}
		s.reindexOne(want, d, key, value)
		return
	}
}

// Stats reports how many keys the named index posts and how many distinct
// values they are spread over. Both are 0 for an index that is not installed.
func (s *Set) Stats(name string) (keys, distinct int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p := s.posts[name]
	if p == nil {
		return 0, 0
	}
	return len(p.keys), len(p.vals)
}

// Selector is one positive filter leaf turned into an index probe: the
// definition to read, the comparison, and the value(s) to compare against
// (one for eq and for the ordering family, N for in).
//
// Only POSITIVE leaves are selectors. That is the whole of the supersetness
// argument: a record satisfying a positive leaf on the indexed path resolves
// to a value there, so it has a posting, so it is a candidate. A negation
// would be the other way round and is left to the predicate.
type Selector struct {
	Def    Def
	Op     vtypes.FilterOp
	Values []vtypes.Value
}

// Candidates returns the keys the selector may match: a SUPERSET of the true
// matches under Def.Prefix above `after`, and a subset of this index's posted
// keys. The caller re-reads each one and re-evaluates the full predicate on
// the live value, which is what turns the superset back into an exact answer.
//
// `after` is EXCLUSIVE; an empty `after` means no cursor. The result is
// sorted by key and free of duplicates, so a caller can page over it.
//
// Every posting key the selector unions counts against budget, cursor or no
// cursor: a set too large to examine is refused on every page rather than
// silently on some. Exceeding it is ErrCandidateBudget and NO keys — never a
// truncated set, which a caller cannot tell from a complete one.
//
// The keys are copies taken under RLock and returned with the lock released.
// The caller reads the cache next, and holding s.mu across a cache read
// self-deadlocks (package doc, lock order).
func (s *Set) Candidates(sel Selector, after []byte, budget int) ([][]byte, error) {
	if budget < 0 {
		budget = 0
	}
	switch sel.Op {
	case vtypes.FilterEq, vtypes.FilterIn:
		if len(sel.Values) == 0 {
			return nil, fmt.Errorf("kvindex: selector for index %q carries no values", sel.Def.Name)
		}
	case vtypes.FilterGt, vtypes.FilterGte, vtypes.FilterLt, vtypes.FilterLte:
		if len(sel.Values) != 1 {
			return nil, fmt.Errorf("kvindex: range selector for index %q carries %d values, want 1", sel.Def.Name, len(sel.Values))
		}
	default:
		// A negation or an absence test narrows nothing: its posting set is not
		// a superset of anything, so accepting one here would lose rows.
		return nil, fmt.Errorf("kvindex: filter op %d is not a candidate selector", sel.Op)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	p := s.posts[sel.Def.Name]
	if p == nil {
		return nil, ErrNoSuchIndex
	}
	if !p.ready {
		return nil, ErrIndexBuilding
	}

	sets, total, err := p.selectSets(sel, budget)
	if err != nil {
		return nil, err
	}
	// total is bounded by budget, so the one allocation this makes is bounded
	// by node config and not by anything the caller sent.
	out := make([][]byte, 0, total)
	afterStr := string(after)
	for _, st := range sets {
		for k := range st {
			if len(after) > 0 && k <= afterStr {
				continue
			}
			out = append(out, []byte(k))
		}
	}
	// Map iteration order must never be observable. Sorting also makes the
	// page a caller cuts off deterministic.
	sort.Slice(out, func(i, j int) bool { return bytes.Compare(out[i], out[j]) < 0 })
	return out, nil
}

// selectSets collects the posting sets the selector unions, charging each
// one's size against budget BEFORE anything is copied. Caller holds the read
// lock.
//
// The sets are disjoint by construction: the reverse map gives every key
// exactly one scalar key per definition, so no key can appear under two
// distinct values and the union needs no deduplication.
func (p *posting) selectSets(sel Selector, budget int) ([]keySet, int, error) {
	var sets []keySet
	used := 0
	// Subtraction form, not addition: used <= budget always holds, so
	// budget-used cannot overflow, and no sum can wrap on a 32-bit int.
	add := func(st keySet) bool {
		if len(st) == 0 {
			return true
		}
		if len(st) > budget-used {
			return false
		}
		used += len(st)
		sets = append(sets, st)
		return true
	}

	switch sel.Op {
	case vtypes.FilterEq, vtypes.FilterIn:
		var seen map[scalarKey]struct{}
		if len(sel.Values) > 1 {
			seen = make(map[scalarKey]struct{}, len(sel.Values))
		}
		for _, v := range sel.Values {
			sk, ok := scalarKeyOf(v)
			if !ok {
				// A NaN or non-scalar bound matches nothing (vtypes.Value.Equal is
				// kind-strict and NaN != NaN), so it contributes no candidates.
				continue
			}
			if seen != nil {
				if _, dup := seen[sk]; dup {
					continue
				}
				seen[sk] = struct{}{}
			}
			if !add(p.vals[sk]) {
				return nil, 0, ErrCandidateBudget
			}
		}
	default: // the ordering family, checked by the caller
		bound := sel.Values[0]
		if bf, ok := numericBound(bound); ok {
			// Cross-kind numeric, matching compileOrdering: an int posting and a
			// float bound compare as float64. A NaN bound admits nothing
			// (orderingHoldsFloat), which is right — nothing satisfies it.
			for sk, st := range p.vals {
				kf, ok := numericKey(sk)
				if !ok || !orderingHoldsFloat(sel.Op, kf, bf) {
					continue
				}
				if !add(st) {
					return nil, 0, ErrCandidateBudget
				}
			}
		} else if bound.Kind == vtypes.ValueString {
			for sk, st := range p.vals {
				if sk.kind != vtypes.ValueString || !orderingHolds(sel.Op, strings.Compare(sk.str, bound.Str)) {
					continue
				}
				if !add(st) {
					return nil, 0, ErrCandidateBudget
				}
			}
		}
		// Any other bound kind (bool, a list, geo, a record) admits nothing,
		// because compileOrdering rejects every field against it.
	}
	return sets, used, nil
}
