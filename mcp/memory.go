// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Reserved names for the memory subsystem's collection, KV bookkeeping keys,
// and metadata fields. Callers of remember/recall never see these directly.
const (
	memCollection = "mcp_memory"     // the vector collection backing remember/recall
	nsField       = "__ns"           // metadata field: the memory's namespace
	createdField  = "__created_unix" // metadata field: unix-seconds creation time
	keyField      = "__key"          // metadata field: caller's stable key (keyed memories only)
	updatedField  = "__updated_unix" // metadata field: unix-seconds last-write time
	// updatedNsField: unix-nanoseconds last-write time, for newest-first order
	// within one second. Additive: `updated` stays seconds. An older version
	// could have stored a user field of this name, so it is read (and hidden)
	// as ours only when it agrees with __updated_unix to the second; anything
	// else stays user metadata. Memories written before it existed sort by
	// seconds, as before.
	updatedNsField = "__updated_ns"
	defaultNS      = "default"        // namespace used when the caller omits one
	kvEmbedder     = "__mcp/embedder" // KV key: the embedIdentity this store was bootstrapped with
)

// MemoryCollection is the name of the vector collection backing remember/recall.
const MemoryCollection = memCollection

// embedIdentity fingerprints the embedder a data dir's mcp_memory collection
// was created with. It is stored once at bootstrap and checked on every
// subsequent NewServer so a config change (a different model, dimension, or
// BM25-only <-> hybrid switch) fails loudly instead of silently corrupting
// search results or crashing on a dimension mismatch deep in the vector index.
type embedIdentity struct {
	Model  string `json:"model"`
	Dim    int    `json:"dim"`
	Hybrid bool   `json:"hybrid"`
}

// identityReadAttempts / identityReadBackoff bound the wait for a concurrent
// bootstrapper to finish writing the embedder identity key. The gap between
// its CreateCollection and that write is one round trip, so a couple of retries
// is generous; a record still missing after them is rebuilt (repairIdentity).
const (
	identityReadAttempts = 3
	identityReadBackoff  = 50 * time.Millisecond
)

// identitySampleSize is how many stored memories repairIdentity re-embeds, and
// identityMinCosine how close each new vector must be to the stored one to
// count as a match. The sample is sought in at most identityScanPages pages of
// identityScanPage points.
const (
	identitySampleSize = 3
	identityMinCosine  = 0.999
	identityScanPage   = 100
	identityScanPages  = 10
)

// ErrEmbedderMismatch is wrapped by the error NewServer or a memory tool
// returns when the store's memory was created with a different embedder than
// this run's, or this run's cannot be verified against it.
var ErrEmbedderMismatch = errors.New("mcp: embedder mismatch")

// embedderAdvice ends every mismatch error: what to do about it.
const embedderAdvice = `set the same ROSTAM_EMBED_* environment the store was created with (none at all for model "stub"), or use a different store`

// checkEmbedderIdentity compares the configured embedder against the one
// recorded in KV, if any. No record yet means the memory collection has
// never been bootstrapped, so there is nothing to conflict with.
func (s *Server) checkEmbedderIdentity(ctx context.Context) error {
	raw, err := s.store.Get(ctx, []byte(kvEmbedder))
	if err != nil {
		if errors.Is(err, rostam.ErrNotFound) {
			return nil
		}
		return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
	}
	return s.matchStoredIdentity(raw)
}

// matchStoredIdentity decodes a recorded identity and checks it against this
// run's embedder. Shared by the startup check and the concurrent-bootstrap
// path, which must apply exactly the same rule to the identity it reads back.
func (s *Server) matchStoredIdentity(raw []byte) error {
	var stored embedIdentity
	if err := json.Unmarshal(raw, &stored); err != nil {
		return fmt.Errorf("mcp: decoding stored embedder identity: %w", err)
	}
	want := s.identity()
	if stored != want {
		return fmt.Errorf("%w: this store's memory was bootstrapped with model=%q dim=%d hybrid=%v, but this run is configured with model=%q dim=%d hybrid=%v; %s",
			ErrEmbedderMismatch, stored.Model, stored.Dim, stored.Hybrid, want.Model, want.Dim, want.Hybrid, embedderAdvice)
	}
	return nil
}

// identity is this run's embedder identity.
func (s *Server) identity() embedIdentity {
	return embedIdentity{Model: s.emb.Model(), Dim: s.emb.Dim(), Hybrid: s.hybrid}
}

// claimIdentity records this run's embedder as the memory collection's, unless
// a record is already there: that one is never overwritten, and this run is
// held to it instead.
func (s *Server) claimIdentity(ctx context.Context) error {
	b, err := json.Marshal(s.identity())
	if err != nil {
		return fmt.Errorf("mcp: encoding embedder identity: %w", err)
	}
	res, err := s.store.Call(ctx, "set_nx", ops.EncodeSetNXArgs([]byte(kvEmbedder), b, 0))
	if err != nil {
		return fmt.Errorf("mcp: storing embedder identity: %w", err)
	}
	stored, err := ops.DecodeCASResult(res)
	if err != nil {
		return fmt.Errorf("mcp: storing embedder identity: %w", err)
	}
	if stored {
		return nil
	}
	raw, err := s.store.Get(ctx, []byte(kvEmbedder))
	if err != nil {
		return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
	}
	return s.matchStoredIdentity(raw)
}

// ensureMemory lazily bootstraps the mcp_memory collection on first use:
// creates it and records the embedder identity that owns it.
//
// Only SUCCESS is latched. sync.Once used to guard this and cached whatever
// the first attempt produced, error included — so one canceled context, one
// not-leader reply, or one transport blip on the very first memory tool call
// left every later remember/recall/forget in the process failing with that
// stale error forever, long after the store recovered. A failed attempt now
// leaves the latch closed and the next caller tries again.
func (s *Server) ensureMemory(ctx context.Context) error {
	s.memMu.Lock()
	defer s.memMu.Unlock()
	if s.memReady {
		return nil
	}
	if s.readOnly {
		if err := s.probeMemory(ctx); err != nil {
			return err
		}
		s.memReady = true
		return nil
	}
	if err := s.bootstrapMemory(ctx); err != nil {
		return err
	}
	s.memReady = true
	return nil
}

// probeMemory is ensureMemory's read-only form: it only checks that the memory
// collection exists, by scrolling one document. ErrNoMemory is returned (and
// not latched) when it does not.
func (s *Server) probeMemory(ctx context.Context) error {
	_, _, _, err := s.store.VectorScroll(ctx, memCollection, rostam.VectorFilter{}, 1, rostam.VectorScrollOpts{})
	switch {
	case err == nil:
		return nil
	case collectionMissing(err):
		return ErrNoMemory
	default:
		return fmt.Errorf("mcp: probing memory collection: %w", err)
	}
}

// collectionMissing reports whether err is the server saying the collection
// does not exist. It mirrors the client package's unexported
// isCollectionNotFound (both spellings, case-insensitive); no importable form
// exists, and any other error (a timeout, a reset) must NOT read as "missing".
func collectionMissing(err error) bool {
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "no collection") || strings.Contains(s, "unknown collection")
}

// bootstrapMemory is one bootstrap attempt. The caller holds s.memMu.
func (s *Server) bootstrapMemory(ctx context.Context) error {
	raw, err := s.store.Get(ctx, []byte(kvEmbedder))
	if err == nil {
		// Already bootstrapped. Compared here too: the record may not have been
		// there when NewServer looked.
		return s.matchStoredIdentity(raw)
	}
	if !errors.Is(err, rostam.ErrNotFound) {
		return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
	}
	// vector.DefaultConfig fills in the HNSW build knobs (M/EfConstruction/
	// EfSearch) that Config.Validate requires but the memory collection has
	// no opinion on; only Dim/Metric/FullText are memory-specific.
	cfg := vector.DefaultConfig()
	cfg.Dim, cfg.Metric, cfg.FullText = s.emb.Dim(), vector.Cosine, &vector.FullTextConfig{}
	applyPersistence(&cfg)
	switch err := s.store.CreateCollection(ctx, memCollection, cfg); {
	case err == nil:
	case isCollectionExists(err):
		// A second session bootstrapped between our existence check and this
		// call — routine in remote mode, where several `mcp -connect` processes
		// share one store. The collection this server needs now exists, so the
		// race is won by joining it, not by failing.
		return s.awaitStoredIdentity(ctx)
	default:
		return fmt.Errorf("mcp: creating memory collection: %w", err)
	}
	return s.claimIdentity(ctx)
}

// isCollectionExists reports whether err is "that collection is already
// there". Embedded mode returns vector.ErrCollectionExists directly; over
// -connect the same failure arrives as a reconstructed error whose chain does
// not survive the wire, so the message is matched as well.
func isCollectionExists(err error) bool {
	return errors.Is(err, vector.ErrCollectionExists) ||
		strings.Contains(err.Error(), "collection already exists")
}

// awaitStoredIdentity reads back the embedder identity a concurrent creator
// recorded and verifies it matches this run's configuration.
//
// The other session writes the identity key one round trip AFTER creating the
// collection, so reading it immediately can legitimately miss it; a few
// bounded retries cover that window. A record still missing after them is
// rebuilt, which first checks this run's embedder against the collection.
func (s *Server) awaitStoredIdentity(ctx context.Context) error {
	for attempt := range identityReadAttempts {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(identityReadBackoff):
			}
		}
		raw, err := s.store.Get(ctx, []byte(kvEmbedder))
		switch {
		case err == nil:
			return s.matchStoredIdentity(raw)
		case errors.Is(err, rostam.ErrNotFound):
			continue
		default:
			return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
		}
	}
	return s.repairIdentity(ctx)
}

// repairIdentity records this run's embedder for a memory collection that has
// no identity record, after verifying it against what the collection holds: the
// dimension must match, and re-embedding a sample of memories must reproduce
// the stored vectors of at least half of them (and at least one), so that one
// bad stored vector does not refuse the right embedder. An empty collection
// offers only its dimension to check; a non-empty one with no content to
// re-embed is refused. A refusal wraps ErrEmbedderMismatch and records
// nothing; no memory is ever changed.
func (s *Server) repairIdentity(ctx context.Context) error {
	probe := make([]float32, s.emb.Dim())
	if len(probe) > 0 {
		probe[0] = 1
	}
	if _, err := s.store.VectorSearch(ctx, memCollection, probe, 1); err != nil {
		// Matched by message: over -connect the error chain does not survive
		// the wire.
		if strings.Contains(err.Error(), vector.ErrDimMismatch.Error()) {
			return fmt.Errorf("%w: the %q collection does not hold vectors of this run's dim=%d; %s",
				ErrEmbedderMismatch, memCollection, s.emb.Dim(), embedderAdvice)
		}
		return err
	}
	sample, empty, err := s.identitySample(ctx)
	if err != nil {
		return err
	}
	if empty {
		return s.claimIdentity(ctx)
	}
	if len(sample) == 0 {
		return fmt.Errorf("%w: cannot verify this run's embedder: the %q collection is not empty, but no point with content to re-embed was found in it",
			ErrEmbedderMismatch, memCollection)
	}
	matched := 0
	for _, d := range sample {
		ok, err := s.reproduces(ctx, d)
		if err != nil {
			return err
		}
		if ok {
			matched++
		}
	}
	if matched == 0 || 2*matched < len(sample) {
		return fmt.Errorf("%w: model=%q reproduces the stored vector of only %d of %d sampled memories; %s",
			ErrEmbedderMismatch, s.emb.Model(), matched, len(sample), embedderAdvice)
	}
	return s.claimIdentity(ctx)
}

// identitySample returns up to identitySampleSize points of the memory
// collection that have content, skipping those without, and whether the
// collection holds no point at all. The next page's cursor is built from the
// last id: a server reached over -connect may return a page without one.
func (s *Server) identitySample(ctx context.Context) (sample []rostam.VectorDocument, empty bool, err error) {
	empty = true
	var cursor string
	for range identityScanPages {
		docs, _, _, err := s.store.VectorScroll(ctx, memCollection, rostam.VectorFilter{}, identityScanPage, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			return nil, false, err
		}
		for _, d := range docs {
			empty = false
			if d.Content == "" {
				continue
			}
			sample = append(sample, d)
			if len(sample) == identitySampleSize {
				return sample, empty, nil
			}
		}
		if len(docs) < identityScanPage {
			break
		}
		cursor = ops.EncodeScrollCursor(docs[len(docs)-1].ID)
	}
	return sample, empty, nil
}

// reproduces reports whether embedding d's content yields its stored vector.
func (s *Server) reproduces(ctx context.Context, d rostam.VectorDocument) (bool, error) {
	found, stored, _, _, _, err := s.store.VectorGet(ctx, memCollection, d.ID, true, false)
	if err != nil {
		return false, err
	}
	if !found {
		return false, fmt.Errorf("mcp: memory %d disappeared while the embedder was being verified against it", d.ID)
	}
	vecs, err := s.emb.Embed(ctx, []string{d.Content})
	if err != nil {
		return false, err
	}
	vec, err := firstEmbedding(vecs)
	if err != nil {
		return false, err
	}
	return cosine(stored, vec) >= identityMinCosine, nil // false for NaN
}

// cosine is the cosine similarity of a and b; 0 when their lengths differ or
// either has no magnitude.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}

// firstEmbedding pulls the single vector an Embed of one text should have
// produced.
//
// Every embed call in this package passes exactly one string, so it is
// tempting to index [0] straight away — but the embedder is not necessarily
// ours. semcache's OpenAI-compatible embedder is an HTTP client pointed at
// whatever ROSTAM_EMBED_ENDPOINT names, and a response with a short or empty
// data array decodes into a short slice with a nil error. Indexing that panics
// the dispatch goroutine and takes the session down, so a result that is not
// exactly one usable vector is reported as the bad response it is.
func firstEmbedding(vecs [][]float32) ([]float32, error) {
	if len(vecs) == 0 {
		return nil, errors.New("mcp: the embedder returned no vectors for one input")
	}
	if len(vecs[0]) == 0 {
		return nil, errors.New("mcp: the embedder returned an empty vector")
	}
	return vecs[0], nil
}

// jsSafeIDMask keeps a generated id inside the range JSON numbers represent
// exactly (2^53-1). See memoryID.
const jsSafeIDMask = uint64(1)<<53 - 1

// memoryID derives a memory's point id from its namespace and content, so
// remembering the same fact twice (in the same namespace) upserts the same
// point instead of accumulating duplicates.
//
// The hash is masked down to 53 bits because the id is handed straight back to
// the client and comes straight back to forget. Most MCP clients are
// JavaScript, where a JSON number is an IEEE-754 double: a full-width uint64
// id is rounded on parse, so the id a client echoes into forget would not be
// the id it was given, and the memory would come back "missing". 53 bits is
// where that stops being possible.
//
// The narrower space costs nothing real here: 2^53 is about 9x10^15 ids, so
// even a memory store of a hundred million facts sits far below the birthday
// threshold, and a collision only ever affects a (namespace, content) pair
// that would have deduped anyway.
func memoryID(ns, content string) uint64 {
	return xxhash.Sum64String(ns+"\x00"+content) & jsSafeIDMask
}

// memoryKeyID derives a memory's point id from (namespace, key) so
// re-remembering the same key upserts the same point regardless of content.
//
// The preimage inserts a "__key\x00" marker after the namespace separator, so
// in practice a keyed id and a content id never coincide. This is NOT a hard
// domain separation: memoryID(ns, content) hashes ns+"\x00"+content, so a
// crafted content of exactly "__key\x00"+key would reproduce this preimage and
// collide. That requires an embedded NUL byte in content, the same namespace,
// and the same (single-tenant) agent's own store — no trust boundary is
// crossed, and normal facts never carry a "__key\x00" prefix. memoryID is left
// unchanged deliberately: altering it would change every existing unkeyed
// memory's id (breaking the store) and violate the no-key backward-compat
// guarantee. If true domain separation is ever needed, tag BOTH functions with
// a leading byte before the namespace — which is a store-migrating change.
func memoryKeyID(ns, key string) uint64 {
	return xxhash.Sum64String(ns+"\x00__key\x00"+key) & jsSafeIDMask
}

// memoryHit is one recalled (or listed, Task 5) memory: its id, content,
// relevance score, and user metadata with the reserved fields stripped. No
// Distance field: it's meaningless for BM25-only recall and for a scroll
// listing (no query), and this shape is locked by Task 4's tests. The
// generic search tool (Task 6) has its own searchHit type in db.go for
// exactly this reason — see hybridDocs and recallHybrid.
type memoryHit struct {
	ID       uint64         `json:"id"`
	Content  string         `json:"content"`
	Score    float32        `json:"score"`
	Key      string         `json:"key,omitempty"`
	Created  int64          `json:"created,omitempty"`
	Updated  int64          `json:"updated,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`

	// Code link (set only for memories remembered with a path). Stale is true
	// when the file changed since the memory was written, "file missing" when
	// it is gone; fileHash is the stored hash, kept for markStale.
	Path     string `json:"path,omitempty"`
	Symbol   string `json:"symbol,omitempty"`
	Stale    any    `json:"stale,omitempty"`
	fileHash string
	// updatedNs is the nanosecond write time when stored (0 for older
	// memories): ordering only, never on the wire.
	updatedNs int64
}

// registerMemoryTools registers remember, recall, forget, list_memories, and
// list_namespaces.
func (s *Server) registerMemoryTools() {
	s.register(toolDef{
		Name:        "remember",
		Description: `Store a fact in persistent memory for later recall. Store ONE self-contained, durable fact per call (a decision, gotcha, command, or finding), phrased so it stands alone when recalled later — not transient chatter, and never secrets. Omit namespace to use this server's default (the project's, when it was started for one); when passing one, use the project's main checkout directory name, not a worktree's. Pass a stable key for live/in-flight state (e.g. a PR's status) so re-remembering the same key REPLACES the prior entry instead of piling up snapshots; omit key for durable one-off facts.`,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"content":   map[string]any{"type": "string", "description": "the fact to store"},
				"namespace": map[string]any{"type": "string", "description": s.nsDesc("isolation namespace")},
				"metadata":  map[string]any{"type": "object", "description": "optional extra metadata to attach"},
				"key":       map[string]any{"type": "string", "description": "optional stable key; re-remembering the same key REPLACES the prior content (one canonical entry). Use for live/in-flight state; omit for durable one-off facts."},
				"path":      map[string]any{"type": "string", "description": "optional repo-relative file this fact is about; recall then flags the memory stale if the file changes"},
				"symbol":    map[string]any{"type": "string", "description": "optional symbol within path (requires path)"},
			},
			"required": []any{"content"},
		},
		Handler: s.handleRemember,
	})
	s.register(toolDef{
		Name:        "recall",
		Description: `Search stored memories by relevance to a query, scoped to a namespace. Call this at the START of a task, before re-reading large files or re-exploring a codebase — it's cheaper than rediscovering what you already know. Omit namespace to search this server's default namespace.`,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":     map[string]any{"type": "string", "description": "search text"},
				"namespace": map[string]any{"type": "string", "description": s.nsDesc("restrict the search to this namespace")},
				"k":         map[string]any{"type": "integer", "description": "max hits to return (default 5)"},
				"filter":    map[string]any{"type": "object", "description": "optional metadata filter, ANDed with the namespace"},
			},
			"required": []any{"query"},
		},
		Handler: s.handleRecall,
	})
	s.register(toolDef{
		Name:        "forget",
		Description: `Delete stored memories by id and/or by their stable key (namespace + key resolves to the memory a keyed remember landed on). ids and keys may be combined in one call. Returns {"deleted":[...],"missing":[...],"errors":[...]}; a per-id delete failure does not abort the rest of the batch.`,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"ids": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "integer"},
					"description": "ids of the memories to delete",
				},
				"namespace": map[string]any{"type": "string", "description": s.nsDesc("namespace the keys below belong to")},
				"keys": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "stable keys of keyed memories to delete, resolved within namespace",
				},
			},
		},
		Handler: s.handleForget,
	})
	s.register(toolDef{
		Name:        "list_memories",
		Description: "List stored memories in a namespace, paginated.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"namespace": map[string]any{"type": "string", "description": s.nsDesc("namespace to list")},
				"limit":     map[string]any{"type": "integer", "description": "max memories per page (default 50, max 500)"},
				"cursor":    map[string]any{"type": "string", "description": "resume-after cursor from a previous call's next_cursor"},
			},
		},
		Handler: s.handleListMemories,
	})
	s.registerFileContextTools()
	s.register(toolDef{
		Name:        "list_namespaces",
		Description: "List all known memory namespaces.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
		Handler: s.handleListNamespaces,
	})
}

// rememberArgs is the remember tool's decoded input.
type rememberArgs struct {
	Content   string                     `json:"content"`
	Namespace string                     `json:"namespace"`
	Metadata  map[string]json.RawMessage `json:"metadata"`
	Key       string                     `json:"key"`
	Path      *string                    `json:"path"` // pointer: absent and "" differ
	Symbol    string                     `json:"symbol"`
}

func (s *Server) handleRemember(ctx context.Context, raw json.RawMessage) (any, error) {
	if s.readOnly {
		return nil, errReadOnly
	}
	var args rememberArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad remember args: %w", err)
	}
	if strings.TrimSpace(args.Content) == "" {
		return nil, fmt.Errorf("mcp: remember: content is required")
	}
	ns := s.namespaceOr(args.Namespace)
	for _, reserved := range [...]string{nsField, createdField, keyField, updatedField, updatedNsField, pathField, symbolField, fileHashField, linkField, "$content"} {
		if _, ok := args.Metadata[reserved]; ok {
			return nil, fmt.Errorf("mcp: remember: metadata key %q is reserved", reserved)
		}
	}

	var relPath string
	if args.Path != nil {
		if *args.Path == "" || *args.Path != strings.TrimSpace(*args.Path) {
			return nil, fmt.Errorf("mcp: remember: path must be non-empty with no leading or trailing whitespace")
		}
		var err error
		if relPath, err = normalizeLinkPath(s.projectRoot, *args.Path); err != nil {
			return nil, fmt.Errorf("mcp: remember: %w", err)
		}
	} else if args.Symbol != "" {
		return nil, fmt.Errorf("mcp: remember: symbol requires path")
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	md, err := jsonToMetadata(args.Metadata)
	if err != nil {
		return nil, err
	}
	if md == nil {
		md = make(rostam.VectorMetadata, 2)
	}
	wrote := time.Now()
	now := wrote.Unix()
	md[nsField] = vector.NewString(ns)
	md[updatedField] = vector.NewInt(now)
	md[updatedNsField] = vector.NewInt(wrote.UnixNano())
	if relPath != "" {
		hash, _ := fileHash(s.projectRoot, relPath) // "" when unreadable: link is still stored
		md[pathField] = vector.NewString(relPath)
		md[fileHashField] = vector.NewString(hash)
		md[linkField] = linkMarker()
		if args.Symbol != "" {
			md[symbolField] = vector.NewString(args.Symbol)
		}
	}

	// A keyed remember always lands on the same point (memoryKeyID ignores
	// content), so a re-remember under the same key is an update to one
	// canonical memory, not a new fact: its __created_unix should keep
	// naming when that memory first came into being, not when it was last
	// touched. The lookup is best-effort — a fresh key has no prior point,
	// and that's not an error, just "created now".
	var id uint64
	if args.Key != "" {
		id = memoryKeyID(ns, args.Key)
		md[keyField] = vector.NewString(args.Key)
		created := now
		if found, _, existing, _, _, err := s.store.VectorGet(ctx, memCollection, id, false, true); err != nil {
			return nil, fmt.Errorf("mcp: remember: checking existing keyed memory: %w", err)
		} else if found {
			if v, ok := existing[createdField]; ok && v.Kind == vector.ValueInt {
				created = v.Int
			}
		}
		md[createdField] = vector.NewInt(created)
	} else {
		id = memoryID(ns, args.Content)
		md[createdField] = vector.NewInt(now)
	}

	vecs, err := s.emb.Embed(ctx, []string{args.Content})
	if err != nil {
		return nil, fmt.Errorf("mcp: embed content: %w", err)
	}
	vec, err := firstEmbedding(vecs)
	if err != nil {
		return nil, fmt.Errorf("mcp: embed content: %w", err)
	}

	if err := s.store.VectorUpsert(ctx, memCollection, id, vec, args.Content, rostam.VectorInsertOpts{Metadata: md}); err != nil {
		return nil, fmt.Errorf("mcp: remember: %w", err)
	}
	result := map[string]any{"id": id, "namespace": ns}
	if args.Key != "" {
		result["key"] = args.Key
	}
	return result, nil
}

// recallArgs is the recall tool's decoded input.
type recallArgs struct {
	Query     string          `json:"query"`
	Namespace string          `json:"namespace"`
	K         int             `json:"k"`
	Filter    json.RawMessage `json:"filter"`
}

func (s *Server) handleRecall(ctx context.Context, raw json.RawMessage) (any, error) {
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad recall args: %w", err)
	}
	if strings.TrimSpace(args.Query) == "" {
		return nil, fmt.Errorf("mcp: recall: query is required")
	}
	ns := s.namespaceOr(args.Namespace)
	// <= 0, not == 0: a client can send "k": -1, and a negative k reaching the
	// search call is the same nonsense as a zero one. Matches list_memories.
	k := args.K
	if k <= 0 {
		k = 5
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	userFilter, err := parseFilter(args.Filter)
	if err != nil {
		return nil, err
	}
	nsFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: nsField, Value: vector.NewString(ns)}
	f := nsFilter
	if !userFilter.IsZero() {
		f = rostam.VectorFilter{Op: vector.FilterAnd, And: []rostam.VectorFilter{nsFilter, userFilter}}
	}

	if !s.hybrid {
		return s.recallBM25(ctx, args.Query, k, f)
	}
	return s.recallHybrid(ctx, args.Query, k, f)
}

// recallBM25 runs the text-only search path (no embedder configured): a
// VectorDocument already carries content and metadata, so no batch fetch is
// needed.
func (s *Server) recallBM25(ctx context.Context, query string, k int, f rostam.VectorFilter) (any, error) {
	docs, _, err := s.store.VectorSearchText(ctx, memCollection, query, k, rostam.VectorSearchOpts{Filter: f})
	if err != nil {
		return nil, fmt.Errorf("mcp: recall: %w", err)
	}
	hits := make([]memoryHit, len(docs))
	for i, d := range docs {
		hits[i] = toMemoryHit(d.ID, d.Content, d.Score, d.Metadata)
	}
	s.markStale(hits)
	return map[string]any{"hits": hits}, nil
}

// recallHybrid runs the dense+BM25 fusion path via the shared hybridDocs
// helper (db.go), then narrows each hit down to memoryHit's locked shape:
// dropping Distance (meaningless for a memory recall) and stripping the
// memory subsystem's reserved metadata fields. hybridDocs itself is
// collection-agnostic and leaves metadata/distance untouched, since a
// generic search caller (Task 6) wants both.
func (s *Server) recallHybrid(ctx context.Context, query string, k int, f rostam.VectorFilter) (any, error) {
	dense, err := s.emb.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("mcp: embed query: %w", err)
	}
	vec, err := firstEmbedding(dense)
	if err != nil {
		return nil, fmt.Errorf("mcp: embed query: %w", err)
	}
	docs, err := s.hybridDocs(ctx, memCollection, vec, query, k, f)
	if err != nil {
		return nil, fmt.Errorf("mcp: recall: %w", err)
	}
	hits := make([]memoryHit, len(docs))
	for i, d := range docs {
		hits[i] = toMemoryHitJSON(d.ID, d.Content, d.Score, d.Metadata)
	}
	s.markStale(hits)
	return map[string]any{"hits": hits}, nil
}

// reservedMetadataFields lists every metadata field the memory subsystem
// owns for its own bookkeeping. Callers of remember/recall never see these
// directly; toMemoryHit/toMemoryHitJSON strip them from returned metadata
// after lifting key/created/updated into memoryHit's typed columns.
var reservedMetadataFields = [...]string{nsField, createdField, keyField, updatedField}

// stripReservedMetadata removes the memory subsystem's own bookkeeping
// fields (namespace, key, creation/update time, and the code-link fields when
// link is set) before metadata is handed back to a tool caller. $content is
// stripped separately by metadataToJSON.
func stripReservedMetadata(md rostam.VectorMetadata, link bool) rostam.VectorMetadata {
	if len(md) == 0 {
		return md
	}
	out := make(rostam.VectorMetadata, len(md))
	for k, v := range md {
		if isReservedField(k, link) {
			continue
		}
		out[k] = v
	}
	return out
}

func isReservedField(k string, link bool) bool {
	for _, r := range reservedMetadataFields {
		if k == r {
			return true
		}
	}
	if link {
		for _, r := range linkFields {
			if k == r {
				return true
			}
		}
	}
	return false
}

// toMemoryHit builds a memoryHit from a raw doc, lifting the reserved key /
// created / updated fields into typed columns and stripping ALL reserved
// fields (__ns, __created_unix, __updated_unix, __key) from the returned
// metadata. Used by recallBM25 and handleListMemories, whose docs carry
// metadata as rostam.VectorMetadata straight from the engine.
func toMemoryHit(id uint64, content string, score float32, md rostam.VectorMetadata) memoryHit {
	hit := memoryHit{ID: id, Content: content, Score: score}
	if v, ok := md[keyField]; ok && v.Kind == vector.ValueString {
		hit.Key = v.Str
	}
	if v, ok := md[createdField]; ok && v.Kind == vector.ValueInt {
		hit.Created = v.Int
	}
	if v, ok := md[updatedField]; ok && v.Kind == vector.ValueInt {
		hit.Updated = v.Int
	}
	mv, ok := md[linkField]
	link := isLinkMarker(mv, ok)
	if link {
		if v, ok := md[pathField]; ok && v.Kind == vector.ValueString {
			hit.Path = v.Str
		}
		if v, ok := md[symbolField]; ok && v.Kind == vector.ValueString {
			hit.Symbol = v.Str
		}
		if v, ok := md[fileHashField]; ok && v.Kind == vector.ValueString {
			hit.fileHash = v.Str
		}
	}
	sm := stripReservedMetadata(md, link)
	if v, ok := md[updatedNsField]; ok && v.Kind == vector.ValueInt && ownUpdatedNs(v.Int, hit.Updated) {
		hit.updatedNs = v.Int
		delete(sm, updatedNsField)
	}
	hit.Metadata = metadataToJSON(sm)
	return hit
}

// ownUpdatedNs reports whether ns is the write time remember stored: it must
// fall within the stored second. See updatedNsField.
func ownUpdatedNs(ns, updated int64) bool {
	return updated != 0 && ns >= updated*1e9 && ns < (updated+1)*1e9
}

// orderKey is a memory's one effective write time in nanoseconds: the stored
// nanosecond time, else (older memories) the start of its stored second.
func orderKey(h memoryHit) int64 {
	if h.updatedNs != 0 {
		return h.updatedNs
	}
	return h.Updated * 1e9
}

// newerThan orders memories newest first by orderKey, the id breaking ties:
// a strict total order, so sorting is well-defined even when timestamped and
// older memories share a second.
func newerThan(a, b memoryHit) bool {
	if ka, kb := orderKey(a), orderKey(b); ka != kb {
		return ka > kb
	}
	return a.ID < b.ID
}

// toMemoryHitJSON is toMemoryHit's counterpart for recallHybrid, whose docs
// arrive through the shared hybridDocs helper (db.go) with metadata already
// converted to JSON-friendly map[string]any (via metadataToJSON) rather than
// rostam.VectorMetadata — hybridDocs is collection-agnostic and shared with
// the generic search tool, which wants that shape. It applies the same
// key/created/updated extraction and reserved-field strip to that shape.
func toMemoryHitJSON(id uint64, content string, score float32, md map[string]any) memoryHit {
	hit := memoryHit{ID: id, Content: content, Score: score}
	if v, ok := md[keyField].(string); ok {
		hit.Key = v
	}
	if v, ok := md[createdField].(int64); ok {
		hit.Created = v
	}
	if v, ok := md[updatedField].(int64); ok {
		hit.Updated = v
	}
	// metadataToJSON renders the geo marker as {"lat","lon"}.
	if _, link := md[linkField].(map[string]float64); link {
		hit.Path, _ = md[pathField].(string)
		hit.Symbol, _ = md[symbolField].(string)
		hit.fileHash, _ = md[fileHashField].(string)
		for _, r := range linkFields {
			delete(md, r)
		}
	}
	if v, ok := md[updatedNsField].(int64); ok && ownUpdatedNs(v, hit.Updated) {
		hit.updatedNs = v
		delete(md, updatedNsField)
	}
	for _, r := range reservedMetadataFields {
		delete(md, r)
	}
	hit.Metadata = md
	return hit
}

// forgetArgs is the forget tool's decoded input.
type forgetArgs struct {
	IDs       []uint64 `json:"ids"`
	Namespace string   `json:"namespace"`
	Keys      []string `json:"keys"`
}

// handleForget deletes memories by id and/or by stable key. Each key in Keys
// is resolved to the id a keyed remember would have landed on
// (memoryKeyID(ns, key)) and folded into the same id set as IDs, so from here
// on a key-derived id and a caller-supplied id are handled identically. Ids
// not found in the collection are reported back as "missing" rather than
// erroring, so a caller can forget a batch without first checking which ids
// still exist; per-id failures land in "errors" for the same reason (see
// forgetResult).
//
// There is no namespace bookkeeping to do afterwards: a namespace is defined
// by the memories carrying it (see handleListNamespaces), so deleting the last
// one is all that "removing a namespace" means.
func (s *Server) handleForget(ctx context.Context, raw json.RawMessage) (any, error) {
	if s.readOnly {
		return nil, errReadOnly
	}
	var args forgetArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad forget args: %w", err)
	}
	if len(args.IDs) == 0 && len(args.Keys) == 0 {
		return nil, fmt.Errorf("mcp: forget: ids or keys is required")
	}

	ids := args.IDs
	if len(args.Keys) > 0 {
		ns := s.namespaceOr(args.Namespace)
		ids = make([]uint64, 0, len(args.IDs)+len(args.Keys))
		ids = append(ids, args.IDs...)
		for _, key := range args.Keys {
			ids = append(ids, memoryKeyID(ns, key))
		}
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	points, missing, err := s.store.VectorGetBatch(ctx, memCollection, ids, false, true)
	if err != nil {
		return nil, fmt.Errorf("mcp: forget: %w", err)
	}

	// Every point's delete is attempted, even after an earlier one fails: a
	// batch with a mix of good and bad ids must still delete whatever it
	// legitimately can, instead of aborting the whole call and leaving the
	// caller unable to tell "nothing happened" from "some of this went
	// through".
	deleted := make([]uint64, 0, len(points))
	var delErrs []string
	for _, p := range points {
		if _, err := s.store.VectorDelete(ctx, memCollection, p.ID); err != nil {
			delErrs = append(delErrs, fmt.Sprintf("id %d: %s", p.ID, err))
			continue
		}
		deleted = append(deleted, p.ID)
	}

	if missing == nil {
		missing = []uint64{}
	}
	return forgetResult{Deleted: deleted, Missing: missing, Errors: delErrs}, nil
}

// forgetResult is the forget tool's response shape, deliberately identical to
// delete's (db.go): the two tools do the same thing to different collections,
// so a caller should not have to learn two contracts. Errors is omitted on full
// success, which keeps the {deleted, missing} shape callers already depend on;
// when present, a partial batch still reports what it managed to delete rather
// than discarding that outcome behind a tool-level error.
type forgetResult struct {
	Deleted []uint64 `json:"deleted"`
	Missing []uint64 `json:"missing"`
	Errors  []string `json:"errors,omitempty"`
}

// listMemoriesArgs is the list_memories tool's decoded input.
type listMemoriesArgs struct {
	Namespace string `json:"namespace"`
	Limit     int    `json:"limit"`
	Cursor    string `json:"cursor"`
}

// handleListMemories pages through a namespace's memories in id order.
// Score is left at its zero value on every hit: scroll has no query to rank
// results against.
func (s *Server) handleListMemories(ctx context.Context, raw json.RawMessage) (any, error) {
	var args listMemoriesArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad list_memories args: %w", err)
	}
	ns := s.namespaceOr(args.Namespace)
	limit := args.Limit
	switch {
	case limit <= 0:
		limit = 50
	case limit > 500:
		limit = 500
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	nsFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: nsField, Value: vector.NewString(ns)}
	docs, _, cursor, err := s.store.VectorScroll(ctx, memCollection, nsFilter, limit, rostam.VectorScrollOpts{Cursor: args.Cursor})
	if err != nil {
		return nil, fmt.Errorf("mcp: list_memories: %w", err)
	}
	memories := make([]memoryHit, len(docs))
	for i, d := range docs {
		memories[i] = toMemoryHit(d.ID, d.Content, d.Score, d.Metadata)
	}
	s.markStale(memories)
	return map[string]any{"memories": memories, "next_cursor": cursor}, nil
}

// nsScanPage is how many memories one list_namespaces page pulls. Only each
// doc's __ns field is kept, so the page size trades round trips against
// per-page memory, and memory sets are small.
const nsScanPage = 500

// handleListNamespaces reports every namespace currently holding at least one
// memory, by scrolling the collection and collecting the distinct __ns values.
//
// This used to read a KV registry that remember appended to and forget pruned.
// That registry was a read-modify-write on one key with no CAS, correct only
// under the assumption of a single session — which does not hold in remote
// mode, where several `mcp -connect` processes share one store and each has
// its own Server. Two of them interleaving could drop a namespace from the
// list permanently while its memories sat there, and no later operation would
// ever notice.
//
// A scan costs more than the registry's single Get, but it cannot be wrong:
// the memories ARE the namespace list. It also deletes the whole class of
// bookkeeping bugs the registry needed (seeding it at bootstrap, appending on
// remember, pruning on forget, and keeping all three consistent when a batch
// half-failed).
func (s *Server) handleListNamespaces(ctx context.Context, _ json.RawMessage) (any, error) {
	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}
	seen := make(map[string]struct{})
	var cursor string
	for {
		// A zero filter is match-all: every memory, every namespace.
		docs, _, next, err := s.store.VectorScroll(ctx, memCollection, rostam.VectorFilter{}, nsScanPage, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("mcp: list_namespaces: %w", err)
		}
		for _, d := range docs {
			if v, ok := d.Metadata[nsField]; ok && v.Kind == vector.ValueString {
				seen[v.Str] = struct{}{}
			}
		}
		// A cursor that does not advance would loop forever; treat it as
		// exhausted rather than trusting the backend to always terminate.
		if next == "" || next == cursor {
			break
		}
		cursor = next
	}
	out := make([]string, 0, len(seen))
	for n := range seen {
		out = append(out, n)
	}
	sort.Strings(out)
	return map[string]any{"namespaces": out}, nil
}
