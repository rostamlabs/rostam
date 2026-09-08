// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"sort"
	"testing"
)

// namedSparseWALConfig is a WAL-mode collection-level config with a mixed
// dense+sparse named layout.
func namedSparseWALConfig() Config {
	return Config{WAL: true, NamedVectors: namedSparseTestConfig()}
}

// namedSparseTestConfig is a mixed dense+sparse config: "title" dim4 cosine
// (dense), "image" dim3 dot (dense), and "terms" sparse.
func namedSparseTestConfig() map[string]NamedVectorParams {
	return map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: Cosine},
		"image": {Dim: 3, Metric: DotProduct},
		"terms": {Sparse: true},
	}
}

func sv(idx []uint32, val []float32) *SparseVector { return &SparseVector{Indices: idx, Values: val} }

// bruteForceSparseTopK ranks vecs by sparseDot(query, v) descending (ties: lower
// id), admit-gated, returning the top-k ids — the ground truth for searchTopK.
func bruteForceSparseTopK(vecs map[uint64]*SparseVector, query *SparseVector, k int, admit func(uint64) bool) []uint64 {
	type sc struct {
		id    uint64
		score float32
	}
	var all []sc
	for id, v := range vecs {
		if admit != nil && !admit(id) {
			continue
		}
		d := sparseDot(*query, *v)
		if d == 0 {
			continue
		}
		all = append(all, sc{id, d})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].id < all[j].id
	})
	if len(all) > k {
		all = all[:k]
	}
	out := make([]uint64, len(all))
	for i, s := range all {
		out[i] = s.id
	}
	return out
}

// TestSparseIndexIDSearchMatchesBruteForce builds an id-keyed sparse index and
// checks searchTopK == a brute-force sparseDot top-k over the same vectors, and
// that the admit gate excludes rejected ids.
func TestSparseIndexIDSearchMatchesBruteForce(t *testing.T) {
	vecs := map[uint64]*SparseVector{
		1: sv([]uint32{0, 2, 5}, []float32{1, 2, 3}),
		2: sv([]uint32{2, 3}, []float32{4, 1}),
		3: sv([]uint32{0, 5, 9}, []float32{2, 2, 5}),
		4: sv([]uint32{7}, []float32{9}),
	}
	si := newSparseIndexID()
	for id, v := range vecs {
		si.add(id, v)
	}
	query := sv([]uint32{0, 2, 5}, []float32{1, 1, 1})

	got := resultIDs(si.searchTopK(query, 10, nil))
	want := bruteForceSparseTopK(vecs, query, 10, nil)
	if !eqUint64(got, want) {
		t.Fatalf("searchTopK=%v want=%v", got, want)
	}

	// Top-k truncation matches.
	got2 := resultIDs(si.searchTopK(query, 2, nil))
	want2 := bruteForceSparseTopK(vecs, query, 2, nil)
	if !eqUint64(got2, want2) {
		t.Fatalf("top2 searchTopK=%v want=%v", got2, want2)
	}

	// admit gate excludes id 3.
	admit := func(id uint64) bool { return id != 3 }
	gotA := resultIDs(si.searchTopK(query, 10, admit))
	wantA := bruteForceSparseTopK(vecs, query, 10, admit)
	if !eqUint64(gotA, wantA) {
		t.Fatalf("admit searchTopK=%v want=%v", gotA, wantA)
	}
	for _, id := range gotA {
		if id == 3 {
			t.Fatal("admit gate did not exclude id 3")
		}
	}
}

// TestSparseIndexIDRemoveAndRebuild checks remove drops all of an id's postings
// and rebuild reconstructs the index from a vecs map.
func TestSparseIndexIDRemoveAndRebuild(t *testing.T) {
	vecs := map[uint64]*SparseVector{
		1: sv([]uint32{0, 1}, []float32{1, 1}),
		2: sv([]uint32{0, 2}, []float32{2, 2}),
	}
	si := newSparseIndexID()
	si.add(1, vecs[1])
	si.add(2, vecs[2])
	si.remove(1)
	delete(vecs, 1)

	query := sv([]uint32{0, 1, 2}, []float32{1, 1, 1})
	got := resultIDs(si.searchTopK(query, 10, nil))
	if !eqUint64(got, []uint64{2}) {
		t.Fatalf("after remove searchTopK=%v want [2]", got)
	}
	// Removed id 0-dim postings should be gone for id 1.
	for _, p := range si.postings[0] {
		if p.id == 1 {
			t.Fatal("remove left a stale posting for id 1")
		}
	}

	// rebuild from a fresh vecs map.
	rebuilt := map[uint64]*SparseVector{
		5: sv([]uint32{0, 7}, []float32{3, 3}),
		6: sv([]uint32{7}, []float32{1}),
	}
	si.rebuild(rebuilt)
	got2 := resultIDs(si.searchTopK(sv([]uint32{0, 7}, []float32{1, 1}), 10, nil))
	want2 := bruteForceSparseTopK(rebuilt, sv([]uint32{0, 7}, []float32{1, 1}), 10, nil)
	if !eqUint64(got2, want2) {
		t.Fatalf("after rebuild searchTopK=%v want=%v", got2, want2)
	}
}

// TestNamedSparseSpaceInsertStores inserts sparse vectors into a sparse named
// space and verifies the engine stores them (vecs + the inverted index), with
// dense+sparse spaces coexisting in one collection.
func TestNamedSparseSpaceInsertStores(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := nc.sparseSpaces["terms"]; !ok {
		t.Fatal("sparse space 'terms' not built")
	}
	if _, ok := nc.indexes["terms"]; ok {
		t.Fatal("sparse space 'terms' must NOT have an hnsw")
	}

	// Mixed insert: dense title + sparse terms.
	if err := nc.InsertSparse(1,
		map[string][]float32{"title": {1, 0, 0, 0}},
		map[string]*SparseVector{"terms": sv([]uint32{0, 3}, []float32{1, 2})},
		Metadata{"k": NewString("a")}, 0); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if err := nc.InsertSparse(2, nil,
		map[string]*SparseVector{"terms": sv([]uint32{3}, []float32{5})},
		nil, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}

	sp := nc.sparseSpaces["terms"]
	if len(sp.vecs) != 2 {
		t.Fatalf("vecs len=%d want 2", len(sp.vecs))
	}
	q := sv([]uint32{3}, []float32{1})
	got := resultIDs(sp.idx.searchTopK(q, 10, nil))
	want := bruteForceSparseTopK(sp.vecs, q, 10, nil)
	if !eqUint64(got, want) {
		t.Fatalf("sparse space search=%v want=%v", got, want)
	}

	// Dense space still works.
	res, err := nc.SearchNamed("title", []float32{1, 0, 0, 0}, 1, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("dense search=%v want id1", res)
	}
}

// TestNamedSparseModalityMismatch checks a sparse value for a dense space (and
// vice versa) fails loud.
func TestNamedSparseModalityMismatch(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	// sparse value for the dense "title" space.
	err = nc.InsertSparse(1, nil, map[string]*SparseVector{"title": sv([]uint32{0}, []float32{1})}, nil, 0)
	if !errors.Is(err, ErrSpaceModalityMismatch) {
		t.Fatalf("sparse-into-dense err=%v want ErrSpaceModalityMismatch", err)
	}
	// dense value for the sparse "terms" space.
	err = nc.InsertSparse(1, map[string][]float32{"terms": {1, 2}}, nil, nil, 0)
	if !errors.Is(err, ErrSpaceModalityMismatch) {
		t.Fatalf("dense-into-sparse err=%v want ErrSpaceModalityMismatch", err)
	}
}

// TestNamedSparseDeleteAndUpsert checks delete drops the id from the sparse space
// and upsert replaces the sparse vector (old postings dropped).
func TestNamedSparseDeleteAndUpsert(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	mustInsert := func(id uint64, idx []uint32, val []float32) {
		if err := nc.InsertSparse(id, nil, map[string]*SparseVector{"terms": sv(idx, val)}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	mustInsert(1, []uint32{0, 1}, []float32{1, 1})
	mustInsert(2, []uint32{0}, []float32{1})

	// Upsert id 1 with a disjoint sparse vector — old dim-1 postings must be gone.
	mustInsert(1, []uint32{5}, []float32{9})
	sp := nc.sparseSpaces["terms"]
	// dim 1 should no longer reference id 1.
	for _, p := range sp.idx.postings[1] {
		if p.id == 1 {
			t.Fatal("upsert left a stale dim-1 posting for id 1")
		}
	}
	got := resultIDs(sp.idx.searchTopK(sv([]uint32{5}, []float32{1}), 10, nil))
	if !eqUint64(got, []uint64{1}) {
		t.Fatalf("after upsert search=%v want [1]", got)
	}

	// Delete id 2.
	if _, err := nc.Delete(2); err != nil {
		t.Fatal(err)
	}
	if _, had := sp.vecs[2]; had {
		t.Fatal("delete left id 2 in sparse vecs")
	}
	got2 := resultIDs(sp.idx.searchTopK(sv([]uint32{0}, []float32{1}), 10, nil))
	for _, id := range got2 {
		if id == 2 {
			t.Fatal("delete left id 2 in sparse index")
		}
	}
}

// TestNamedSparseSnapshotV4RoundTrip checks a collection with a sparse space
// snapshots at v4 and restores the sparse vecs + rebuilt index verbatim.
func TestNamedSparseSnapshotV4RoundTrip(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := nc.InsertSparse(1, map[string][]float32{"title": {1, 0, 0, 0}},
		map[string]*SparseVector{"terms": sv([]uint32{0, 3}, []float32{1, 2})}, nil, 0); err != nil {
		t.Fatal(err)
	}
	if err := nc.InsertSparse(2, nil, map[string]*SparseVector{"terms": sv([]uint32{3, 9}, []float32{4, 1})}, nil, 0); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	// Version byte (after the 4-byte magic) must be v4.
	if buf.Bytes()[4] != 4 {
		t.Fatalf("snapshot version byte=%d want 4 (sparse present)", buf.Bytes()[4])
	}

	nc2, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := nc2.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	sp := nc2.sparseSpaces["terms"]
	if len(sp.vecs) != 2 {
		t.Fatalf("restored vecs len=%d want 2", len(sp.vecs))
	}
	q := sv([]uint32{3}, []float32{1})
	got := resultIDs(sp.idx.searchTopK(q, 10, nil))
	want := bruteForceSparseTopK(sp.vecs, q, 10, nil)
	if !eqUint64(got, want) {
		t.Fatalf("restored search=%v want=%v", got, want)
	}
	// Verify the rebuilt index returns the SAME vectors (id 1 dim 0 value 1).
	v1 := sp.vecs[1]
	if v1 == nil || len(v1.Indices) != 2 || v1.Indices[0] != 0 || v1.Values[1] != 2 {
		t.Fatalf("restored vec id1=%+v", v1)
	}
}

// TestNamedDenseOnlySnapshotByteIdentical proves a dense-only collection's
// snapshot is byte-identical before and after this change: it writes version 3
// and NO sparse block.
func TestNamedDenseOnlySnapshotByteIdentical(t *testing.T) {
	nc, err := NewNamedCollection("c", namedTestConfig()) // dense-only
	if err != nil {
		t.Fatal(err)
	}
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}},
		Metadata{"k": NewString("a")}, 0); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	if buf.Bytes()[4] != 3 {
		t.Fatalf("dense-only snapshot version byte=%d want 3 (byte-identical)", buf.Bytes()[4])
	}
	// Restore into a fresh dense-only collection: identical state.
	nc2, err := NewNamedCollection("c", namedTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := nc2.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("dense-only v3 restore: %v", err)
	}
	res, err := nc2.SearchNamed("title", []float32{1, 0, 0, 0}, 1, Filter{})
	if err != nil || len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("dense-only restore search=%v err=%v", res, err)
	}
	if len(nc2.sparseSpaces) != 0 {
		t.Fatalf("dense-only restore created %d sparse spaces", len(nc2.sparseSpaces))
	}
}

// TestNamedSparseWALReplay inserts sparse vectors on a WAL-mode collection, crashes
// (no Flush), reopens, and verifies the sparse space is recovered from the replayed
// WAL tail.
func TestNamedSparseWALReplay(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("c", namedSparseWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, ok := cs.GetNamed("c")
	if !ok {
		t.Fatal("named collection missing")
	}
	if nc.wal == nil {
		t.Fatal("WAL-mode named collection has nil wal")
	}
	if err := nc.InsertSparse(1, map[string][]float32{"title": {1, 0, 0, 0}},
		map[string]*SparseVector{"terms": sv([]uint32{0, 3}, []float32{1, 2})}, nil, 0); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if err := nc.InsertSparse(2, nil, map[string]*SparseVector{"terms": sv([]uint32{3}, []float32{5})}, nil, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}

	_ = cs.Close() // crash: no Flush, only WAL on disk

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	nc2, ok := cs2.GetNamed("c")
	if !ok {
		t.Fatal("named collection missing after reopen")
	}
	sp := nc2.sparseSpaces["terms"]
	if sp == nil {
		t.Fatal("sparse space missing after reopen")
	}
	if len(sp.vecs) != 2 {
		t.Fatalf("recovered sparse vecs len=%d want 2", len(sp.vecs))
	}
	q := sv([]uint32{3}, []float32{1})
	got := resultIDs(sp.idx.searchTopK(q, 10, nil))
	want := bruteForceSparseTopK(sp.vecs, q, 10, nil)
	if !eqUint64(got, want) {
		t.Fatalf("recovered sparse search=%v want=%v", got, want)
	}
	v1 := sp.vecs[1]
	if v1 == nil || len(v1.Indices) != 2 || v1.Indices[1] != 3 || v1.Values[1] != 2 {
		t.Fatalf("recovered sparse vec id1=%+v", v1)
	}
}

// TestNamedDenseOnlyWALByteIdentical proves a dense-only named WAL record is
// byte-identical: the appendNamedInsertStaged encoding with nil sparseVectors writes NO
// trailing sparse block (compare to the encoding the pre-sparse code produced — a
// record ending right after the version block).
func TestNamedDenseOnlyWALByteIdentical(t *testing.T) {
	vectors := map[string][]float32{"title": {1, 2, 3, 4}}
	payload := Metadata{"k": NewString("v")}

	// Build the record body the OLD (pre-sparse) way: everything through the version
	// block, with NO trailing sparse bytes.
	var oldBuf bytes.Buffer
	oldBuf.WriteByte(byte(namedInsertRec))
	_ = writeU64(&oldBuf, 7)
	_ = writeU64(&oldBuf, 0) // ttlMs
	_ = writeU32(&oldBuf, uint32(len(vectors)))
	for name, vec := range vectors {
		_ = writeString(&oldBuf, name)
		_ = writeU32(&oldBuf, uint32(len(vec)))
		for _, f := range vec {
			_ = writeF32(&oldBuf, f)
		}
	}
	_ = writeOptMeta(&oldBuf, payload)
	writeOptKeyExpires(&oldBuf, nil)
	writeOptVersion(&oldBuf, 1)

	// The new encoder with nil sparseVectors must produce the SAME bytes.
	var newBuf bytes.Buffer
	newBuf.WriteByte(byte(namedInsertRec))
	_ = writeU64(&newBuf, 7)
	_ = writeU64(&newBuf, 0)
	_ = writeU32(&newBuf, uint32(len(vectors)))
	for name, vec := range vectors {
		_ = writeString(&newBuf, name)
		_ = writeU32(&newBuf, uint32(len(vec)))
		for _, f := range vec {
			_ = writeF32(&newBuf, f)
		}
	}
	_ = writeOptMeta(&newBuf, payload)
	writeOptKeyExpires(&newBuf, nil)
	writeOptVersion(&newBuf, 1)
	writeNamedSparseVectors(&newBuf, nil) // dense-only: must write nothing

	if !bytes.Equal(oldBuf.Bytes(), newBuf.Bytes()) {
		t.Fatalf("dense-only WAL record changed: old=%d bytes new=%d bytes", oldBuf.Len(), newBuf.Len())
	}
}
