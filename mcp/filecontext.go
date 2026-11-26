// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// MemoryHit is the wire shape of one recalled memory, exported for callers
// outside this package (the hook command).
type MemoryHit = memoryHit

// ErrNoMemory is returned in ReadOnly mode when the memory collection does not
// exist yet. Nothing has been remembered, so there is nothing to read.
var ErrNoMemory = errors.New("mcp: no memory collection yet")

var errReadOnly = errors.New("mcp: this memory store is open read-only")

const defaultFileK = 5

// FileMemories returns the memories in namespace ns linked to exactly the
// repo-relative path rel, newest first, at most k.
func (s *Server) FileMemories(ctx context.Context, ns, rel string, k int) ([]MemoryHit, error) {
	return s.FileMemoriesExcept(ctx, ns, rel, k, nil)
}

// FileMemoriesExcept is FileMemories leaving out every memory whose id skip
// reports true, BEFORE the k cap: memories a caller has already seen do not
// use up the k it asked for. A nil skip leaves nothing out.
func (s *Server) FileMemoriesExcept(ctx context.Context, ns, rel string, k int, skip func(id uint64) bool) ([]MemoryHit, error) {
	f := rostam.VectorFilter{Op: vector.FilterAnd, And: []rostam.VectorFilter{
		{Op: vector.FilterEq, Field: nsField, Value: vector.NewString(ns)},
		{Op: vector.FilterEq, Field: pathField, Value: vector.NewString(rel)},
	}}
	// A legacy memory whose USER metadata happens to hold __path is not a
	// link (no marker, so toMemoryHit leaves Path empty): drop it.
	notLink := func(h *MemoryHit) bool { return h.Path == "" }
	return s.newestMemories(ctx, f, k, skip, notLink)
}

// RecentMemories returns the newest k memories in namespace ns.
func (s *Server) RecentMemories(ctx context.Context, ns string, k int) ([]MemoryHit, error) {
	return s.RecentMemoriesExcept(ctx, ns, k, nil)
}

// RecentMemoriesExcept is RecentMemories with skip applied before the k cap;
// see FileMemoriesExcept.
func (s *Server) RecentMemoriesExcept(ctx context.Context, ns string, k int, skip func(id uint64) bool) ([]MemoryHit, error) {
	f := rostam.VectorFilter{Op: vector.FilterEq, Field: nsField, Value: vector.NewString(ns)}
	return s.newestMemories(ctx, f, k, skip, nil)
}

// newestMemories scrolls every match of f (memory sets are small), drops the
// ids skip reports, sorts by Updated descending (id ascending breaks ties so
// output is stable), keeps k, and marks staleness on the survivors only.
func (s *Server) newestMemories(ctx context.Context, f rostam.VectorFilter, k int, skip func(uint64) bool, drop func(*MemoryHit) bool) ([]MemoryHit, error) {
	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}
	var hits []MemoryHit
	var cursor string
	for {
		docs, _, next, err := s.store.VectorScroll(ctx, memCollection, f, nsScanPage, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("mcp: listing memories: %w", err)
		}
		for _, d := range docs {
			if skip != nil && skip(d.ID) {
				continue
			}
			h := toMemoryHit(d.ID, d.Content, d.Score, d.Metadata)
			if drop != nil && drop(&h) {
				continue
			}
			hits = append(hits, h)
		}
		if next == "" || next == cursor {
			break
		}
		cursor = next
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Updated != hits[j].Updated {
			return hits[i].Updated > hits[j].Updated
		}
		return hits[i].ID < hits[j].ID
	})
	if k > 0 && len(hits) > k {
		hits = hits[:k]
	}
	s.markStale(hits)
	return hits, nil
}

func (s *Server) registerFileContextTools() {
	s.register(toolDef{
		Name:        "recall_for_file",
		Description: `List the memories linked to a file (remembered with that path), newest first, each flagged stale if the file has changed since. Call this before editing a file.`,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":      map[string]any{"type": "string", "description": "repo-relative (or absolute, inside the project) file path"},
				"namespace": map[string]any{"type": "string", "description": s.nsDesc("namespace to search")},
				"k":         map[string]any{"type": "integer", "description": "max memories to return (default 5)"},
			},
			"required": []any{"path"},
		},
		Handler: s.handleRecallForFile,
	})
}

func (s *Server) handleRecallForFile(ctx context.Context, raw json.RawMessage) (any, error) {
	var args struct {
		Path      string `json:"path"`
		Namespace string `json:"namespace"`
		K         int    `json:"k"`
	}
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad recall_for_file args: %w", err)
	}
	rel, err := normalizeLinkPath(s.projectRoot, args.Path)
	if err != nil {
		return nil, fmt.Errorf("mcp: recall_for_file: %w", err)
	}
	ns := s.namespaceOr(args.Namespace)
	k := args.K
	if k <= 0 {
		k = defaultFileK
	}
	hits, err := s.FileMemories(ctx, ns, rel, k)
	if err != nil {
		return nil, err
	}
	if hits == nil {
		hits = []MemoryHit{}
	}
	return map[string]any{"memories": hits}, nil
}
