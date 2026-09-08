// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// SearchResponse carries ranked results plus explicit degraded-partition info.
type SearchResponse struct {
	Results  []vtypes.Result
	Degraded bool
	Missing  []uint16
}

// SearchRequest is a plain dense-vector KNN search.
type SearchRequest struct {
	Query       []float32
	K           int
	Filter      vtypes.Filter
	Consistency Consistency
}

// Search runs a dense-vector KNN search, returning the top-k results ranked
// by distance/score. Returns ErrCollectionNotFound when the collection does
// not exist.
func (col *Collection) Search(ctx context.Context, req SearchRequest) (SearchResponse, error) {
	body, err := col.c.Call(ctx, "vector_search",
		wire.EncodeVectorSearchArgsOpts(col.name, req.K, req.Query, req.Filter, uint8(req.Consistency), 0, 0))
	if err != nil {
		return SearchResponse{}, mapCollErr(err)
	}
	results, degraded, missing, err := wire.DecodeVectorSearchResultsDegraded(body)
	if err != nil {
		return SearchResponse{}, err
	}
	return SearchResponse{Results: results, Degraded: degraded, Missing: missing}, nil
}

// HybridTextRequest fuses a dense KNN lane with a BM25 full-text lane, the
// text lane analyzed server-side from raw query text.
type HybridTextRequest struct {
	Dense       []float32
	Text        string
	K           int
	Filter      vtypes.Filter
	Method      vtypes.FusionMethod // zero = FusionRRF
	Alpha       float64
	RRFK        int
	DenseK      int
	SparseK     int
	GlobalIDF   bool
	Consistency Consistency
}

// HybridText runs a dense+BM25 hybrid search, fusing both lanes server-side.
// Returns ErrCollectionNotFound when the collection itself does not exist.
func (col *Collection) HybridText(ctx context.Context, req HybridTextRequest) (SearchResponse, error) {
	opts := vtypes.HybridOpts{
		Filter: req.Filter, Method: req.Method, Alpha: req.Alpha,
		RRFK: req.RRFK, DenseK: req.DenseK, SparseK: req.SparseK,
	}
	body, err := col.c.Call(ctx, "vector_hybrid_text",
		wire.EncodeHybridTextArgsGlobal(col.name, req.Dense, req.Text, req.K, opts,
			uint8(req.Consistency), 0, 0, req.GlobalIDF, nil))
	if err != nil {
		return SearchResponse{}, mapCollErr(err)
	}
	results, degraded, missing, err := wire.DecodeHybridResultsDegraded(body)
	if err != nil {
		return SearchResponse{}, err
	}
	return SearchResponse{Results: results, Degraded: degraded, Missing: missing}, nil
}

// HybridSearchRequest fuses a dense KNN lane with an explicit sparse-vector
// lane.
type HybridSearchRequest struct {
	Dense       []float32
	Sparse      vtypes.SparseVector
	K           int
	Filter      vtypes.Filter
	Method      vtypes.FusionMethod
	Alpha       float64
	RRFK        int
	DenseK      int
	SparseK     int
	Consistency Consistency
}

// HybridSearch runs a dense+sparse hybrid search, fusing both lanes
// server-side. Returns ErrCollectionNotFound when the collection itself does
// not exist.
func (col *Collection) HybridSearch(ctx context.Context, req HybridSearchRequest) (SearchResponse, error) {
	opts := vtypes.HybridOpts{
		Filter: req.Filter, Method: req.Method, Alpha: req.Alpha,
		RRFK: req.RRFK, DenseK: req.DenseK, SparseK: req.SparseK,
	}
	body, err := col.c.Call(ctx, "vector_hybrid_search",
		wire.EncodeHybridSearchArgsOpts(col.name, req.Dense, req.K, req.Sparse, opts,
			uint8(req.Consistency), 0, 0))
	if err != nil {
		return SearchResponse{}, mapCollErr(err)
	}
	results, degraded, missing, err := wire.DecodeHybridResultsDegraded(body)
	if err != nil {
		return SearchResponse{}, err
	}
	return SearchResponse{Results: results, Degraded: degraded, Missing: missing}, nil
}

// SearchDocsRequest is a dense-vector KNN search that returns content-bearing
// Documents instead of bare Results.
type SearchDocsRequest struct {
	Query       []float32
	K           int
	Filter      vtypes.Filter
	Consistency Consistency
}

// DocsResponse carries content-bearing documents plus degraded-partition info.
type DocsResponse struct {
	Documents []Document
	Degraded  bool
	Missing   []uint16
}

// SearchDocs runs a dense-vector KNN search, returning the top-k points as
// content-bearing Documents (reusing the Document type from Scroll/reads).
// Returns ErrCollectionNotFound when the collection itself does not exist.
func (col *Collection) SearchDocs(ctx context.Context, req SearchDocsRequest) (DocsResponse, error) {
	body, err := col.c.Call(ctx, "vector_search_docs",
		wire.EncodeVectorSearchArgsOpts(col.name, req.K, req.Query, req.Filter, uint8(req.Consistency), 0, 0))
	if err != nil {
		return DocsResponse{}, mapCollErr(err)
	}
	docs, degraded, missing, err := wire.DecodeVectorDocsDegradedRaw(body)
	if err != nil {
		return DocsResponse{}, err
	}
	out := DocsResponse{Degraded: degraded, Missing: missing}
	for _, d := range docs {
		doc, err := rawDocumentToDocument(d)
		if err != nil {
			return DocsResponse{}, err
		}
		out.Documents = append(out.Documents, doc)
	}
	return out, nil
}

// GroupSearchRequest collapses KNN hits sharing a metadata field value into
// groups, returning the top-k groups ranked by their best member.
type GroupSearchRequest struct {
	Query       []float32
	K           int
	GroupBy     string
	GroupSize   int
	FetchK      int
	Filter      vtypes.Filter
	Consistency Consistency
}

// Group is one group-by-document result: the grouping key plus its best
// hits, best-first.
type Group struct {
	Key  string
	Hits []Document
}

// GroupResponse carries grouped results plus degraded-partition info.
type GroupResponse struct {
	Groups   []Group
	Degraded bool
	Missing  []uint16
}

// SearchGroups runs a group-by-document search, returning the top-k groups
// (ranked by best member) each with up to GroupSize best hits. Returns
// ErrCollectionNotFound when the collection itself does not exist.
func (col *Collection) SearchGroups(ctx context.Context, req GroupSearchRequest) (GroupResponse, error) {
	opts := vtypes.GroupOpts{GroupBy: req.GroupBy, GroupSize: req.GroupSize, FetchK: req.FetchK, Filter: req.Filter}
	body, err := col.c.Call(ctx, "vector_search_groups",
		wire.EncodeGroupSearchArgsOpts(col.name, req.K, req.Query, opts, uint8(req.Consistency), 0, 0))
	if err != nil {
		return GroupResponse{}, mapCollErr(err)
	}
	groups, degraded, missing, err := wire.DecodeGroupsDegradedRaw(body)
	if err != nil {
		return GroupResponse{}, err
	}
	out := GroupResponse{Degraded: degraded, Missing: missing}
	for _, g := range groups {
		key, err := rawGroupKeyToString(g.Key)
		if err != nil {
			return GroupResponse{}, err
		}
		grp := Group{Key: key}
		for _, d := range g.Hits {
			doc, err := rawDocumentToDocument(d)
			if err != nil {
				return GroupResponse{}, err
			}
			grp.Hits = append(grp.Hits, doc)
		}
		out.Groups = append(out.Groups, grp)
	}
	return out, nil
}

// rawDocumentToDocument decodes a RawDocument's aliasing metadata bytes into
// a typed, standalone Document (mirrors Scroll's conversion).
func rawDocumentToDocument(d vtypes.RawDocument) (Document, error) {
	var meta vtypes.Metadata
	if len(d.Metadata) > 0 {
		meta = make(vtypes.Metadata)
		if err := json.Unmarshal(d.Metadata, &meta); err != nil {
			return Document{}, err
		}
	}
	return Document{ID: d.ID, Distance: d.Distance, Score: d.Score, Content: d.Content, Metadata: meta}, nil
}

// rawGroupKeyToString renders a group key (the wire's json.Marshal of a
// vector.Value) as a plain string for the typed Group.
func rawGroupKeyToString(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var v vtypes.Value
	if err := json.Unmarshal(raw, &v); err != nil {
		return "", err
	}
	switch v.Kind {
	case vtypes.ValueString:
		return v.Str, nil
	case vtypes.ValueInt:
		return strconv.FormatInt(v.Int, 10), nil
	case vtypes.ValueFloat:
		return strconv.FormatFloat(v.Flt, 'g', -1, 64), nil
	case vtypes.ValueBool:
		return strconv.FormatBool(v.Bool), nil
	default:
		// ValueRecord considered: falls here too — vector/group.go's groupKeyString
		// already declines records upstream (a byte blob has no single-string
		// bucket identity), so a record can never actually arrive as a group key
		// wire value. The raw-JSON fallback is kept generic rather than special-
		// cased so it degrades safely for any future kind, record included.
		return string(raw), nil
	}
}
