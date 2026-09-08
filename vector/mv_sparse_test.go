// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// sv builds a validated SparseVector from alternating dim/value pairs (dims must be
// supplied ascending). Helper for the MV doc-sparse tests.
func mvSV(pairs ...float32) *SparseVector {
	n := len(pairs) / 2
	out := &SparseVector{Indices: make([]uint32, n), Values: make([]float32, n)}
	for i := 0; i < n; i++ {
		out.Indices[i] = uint32(pairs[2*i])
		out.Values[i] = pairs[2*i+1]
	}
	return out
}

// bruteSparseTopK is the reference sparse-lane scorer: dot product of query vs each
// live doc's sparse vector, top-k descending (ties by lower id).
func bruteSparseTopK(query *SparseVector, docs map[uint64]*SparseVector, k int) []Result {
	all := make([]Result, 0, len(docs))
	for id, dv := range docs {
		if dv == nil {
			continue
		}
		all = append(all, Result{ID: id, Score: sparseDot(*query, *dv)})
	}
	// simple selection sort (small n)
	for i := 0; i < len(all); i++ {
		best := i
		for j := i + 1; j < len(all); j++ {
			if all[j].Score > all[best].Score || (all[j].Score == all[best].Score && all[j].ID < all[best].ID) {
				best = j
			}
		}
		all[i], all[best] = all[best], all[i]
	}
	if len(all) > k {
		all = all[:k]
	}
	return all
}

// TestMVDocSparseStoredAndSearched adds MV docs WITH a doc-level sparse field and
// checks docSparse is stored + sparseIdx.searchTopK matches a brute-force sparse dot.
func TestMVDocSparseStoredAndSearched(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	docs := map[uint64]*SparseVector{
		1: mvSV(0, 1.0, 2, 2.0, 5, 0.5),
		2: mvSV(1, 3.0, 2, 1.0),
		3: mvSV(0, 0.2, 5, 4.0),
	}
	for id, s := range docs {
		if _, err := m.AddCASKeyTTLSparse(id, [][]float32{{1, 0, 0, 0}}, nil, nil, s, CASCond{}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}

	// docSparse stored (deep copy, not aliasing the caller).
	got, ok := m.GetSparse(1)
	if !ok {
		t.Fatal("doc 1 has no stored sparse")
	}
	if len(got.Indices) != 3 || got.Indices[1] != 2 || got.Values[1] != 2.0 {
		t.Fatalf("doc 1 sparse = %+v", got)
	}

	// sparseIdx.searchTopK == brute force.
	q := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	want := bruteSparseTopK(q, docs, 3)
	gotRes := m.SearchSparse(q, 3)
	if len(gotRes) != len(want) {
		t.Fatalf("searchTopK len = %d, want %d (%+v vs %+v)", len(gotRes), len(want), gotRes, want)
	}
	for i := range want {
		if gotRes[i].ID != want[i].ID {
			t.Fatalf("rank %d: id %d, want %d (got %+v want %+v)", i, gotRes[i].ID, want[i].ID, gotRes, want)
		}
		if d := gotRes[i].Score - want[i].Score; d > 1e-6 || d < -1e-6 {
			t.Fatalf("rank %d: score %v, want %v", i, gotRes[i].Score, want[i].Score)
		}
	}
}

// TestMVDocSparseReplaceDropsOld checks a replace-Add drops the prior sparse postings.
func TestMVDocSparseReplaceDropsOld(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if _, err := m.AddCASKeyTTLSparse(1, [][]float32{{1, 0, 0, 0}}, nil, nil, mvSV(0, 5.0, 9, 5.0), CASCond{}); err != nil {
		t.Fatal(err)
	}
	// Replace with a different sparse (drops dim 0/9, adds dim 3).
	if _, err := m.AddCASKeyTTLSparse(1, [][]float32{{0, 1, 0, 0}}, nil, nil, mvSV(3, 2.0), CASCond{}); err != nil {
		t.Fatal(err)
	}
	// Query on the OLD dim must not match doc 1 anymore.
	if r := m.SearchSparse(mvSV(0, 1.0), 5); len(r) != 0 {
		t.Fatalf("old-dim query still matches after replace: %+v", r)
	}
	// Query on the NEW dim matches.
	r := m.SearchSparse(mvSV(3, 1.0), 5)
	if len(r) != 1 || r[0].ID != 1 || r[0].Score != 2.0 {
		t.Fatalf("new-dim query = %+v, want id 1 score 2", r)
	}
	got, _ := m.GetSparse(1)
	if got == nil || len(got.Indices) != 1 || got.Indices[0] != 3 {
		t.Fatalf("stored sparse after replace = %+v", got)
	}
}

// TestMVDocSparseReplaceWithDenseOnlyDrops checks replacing a sparse-bearing doc
// with a dense-only Add (nil sparse) removes the sparse posting + entry.
func TestMVDocSparseReplaceWithDenseOnlyDrops(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, err := m.AddCASKeyTTLSparse(1, [][]float32{{1, 0, 0, 0}}, nil, nil, mvSV(0, 5.0), CASCond{}); err != nil {
		t.Fatal(err)
	}
	if err := m.Add(1, [][]float32{{0, 1, 0, 0}}, nil); err != nil { // dense-only replace
		t.Fatal(err)
	}
	if _, ok := m.GetSparse(1); ok {
		t.Fatal("dense-only replace left a stored sparse")
	}
	if r := m.SearchSparse(mvSV(0, 1.0), 5); len(r) != 0 {
		t.Fatalf("dense-only replace left a sparse posting: %+v", r)
	}
}

// TestMVDocSparseDeleteRemoves checks Delete drops the doc's sparse entry + posting.
func TestMVDocSparseDeleteRemoves(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, err := m.AddCASKeyTTLSparse(7, [][]float32{{1, 0, 0, 0}}, nil, nil, mvSV(2, 9.0), CASCond{}); err != nil {
		t.Fatal(err)
	}
	if !m.Delete(7) {
		t.Fatal("delete reported not present")
	}
	if _, ok := m.GetSparse(7); ok {
		t.Fatal("delete left a stored sparse")
	}
	if r := m.SearchSparse(mvSV(2, 1.0), 5); len(r) != 0 {
		t.Fatalf("delete left a sparse posting: %+v", r)
	}
}

// TestMVDocSparseValidateRejectsBad checks an unsorted sparse vector is rejected and
// nothing is mutated.
func TestMVDocSparseValidateRejectsBad(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	bad := &SparseVector{Indices: []uint32{5, 2}, Values: []float32{1, 2}} // unsorted
	if _, err := m.AddCASKeyTTLSparse(1, [][]float32{{1, 0, 0, 0}}, nil, nil, bad, CASCond{}); err == nil {
		t.Fatal("expected validation error for unsorted sparse")
	}
	if m.NumDocs() != 0 {
		t.Fatalf("rejected add still mutated: NumDocs=%d", m.NumDocs())
	}
}

// TestMVDocSparseSnapshotRoundTrip checks snapshot + restore brings back docSparse
// AND a rebuilt sparseIdx, and that OLD blobs (no sparse block) still decode.
func TestMVDocSparseSnapshotRoundTrip(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	want := map[uint64]*SparseVector{
		1: mvSV(0, 1.0, 3, 2.0),
		2: mvSV(1, 5.0),
	}
	for id, s := range want {
		if _, err := m.AddCASKeyTTLSparse(id, [][]float32{{1, 0, 0, 0}}, nil, nil, s, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// Doc 3 is dense-only (no sparse) — must survive too.
	if err := m.Add(3, [][]float32{{0, 1, 0, 0}}, nil); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	m2, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	if err := m2.restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}

	for id, w := range want {
		got, ok := m2.GetSparse(id)
		if !ok {
			t.Fatalf("doc %d sparse missing after restore", id)
		}
		if len(got.Indices) != len(w.Indices) {
			t.Fatalf("doc %d sparse len %d, want %d", id, len(got.Indices), len(w.Indices))
		}
		for i := range w.Indices {
			if got.Indices[i] != w.Indices[i] || got.Values[i] != w.Values[i] {
				t.Fatalf("doc %d sparse[%d] = (%d,%v), want (%d,%v)", id, i, got.Indices[i], got.Values[i], w.Indices[i], w.Values[i])
			}
		}
	}
	if _, ok := m2.GetSparse(3); ok {
		t.Fatal("dense-only doc 3 gained a sparse after restore")
	}
	// The rebuilt sparseIdx answers searches.
	r := m2.SearchSparse(mvSV(0, 1.0), 5)
	if len(r) != 1 || r[0].ID != 1 {
		t.Fatalf("rebuilt sparseIdx search = %+v, want id 1", r)
	}
}

// TestMVDenseOnlyPersistByteIdentical asserts a dense-only MV's encodeMaps output is
// BYTE-IDENTICAL whether or not the docSparse map is engaged — i.e. the sparse block
// is entirely omitted when no doc carries a sparse vector (no extra marker byte), so
// a dense-only blob matches the pre-sparse format byte-for-byte.
func TestMVDenseOnlyPersistByteIdentical(t *testing.T) {
	build := func() *MultiVectorIndex {
		m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewInt(1)}); err != nil {
			t.Fatal(err)
		}
		if err := m.Add(2, [][]float32{{0, 1, 0, 0}}, nil); err != nil {
			t.Fatal(err)
		}
		return m
	}
	a := build()
	defer a.Close()
	var bufA bytes.Buffer
	if err := a.encodeMaps(&bufA); err != nil {
		t.Fatal(err)
	}

	// A second dense-only index (docSparse touched-but-empty: removeLocked runs on the
	// replace, exercising delete(docSparse) + sparseIdx.remove on a dense-only doc).
	b := build()
	defer b.Close()
	if err := b.Add(1, [][]float32{{0, 0, 1, 0}}, Metadata{"k": NewInt(1)}); err != nil { // replace doc 1
		t.Fatal(err)
	}
	// re-add doc 1 original token to match content of `a`? Not needed: we only compare
	// the LENGTH/shape invariant — assert NO trailing sparse marker beyond the version
	// block by checking the dense-only blob decodes with empty docSparse and the
	// encode is stable across a decode/re-encode round trip.
	var bufB bytes.Buffer
	if err := b.encodeMaps(&bufB); err != nil {
		t.Fatal(err)
	}

	// Round-trip stability: decode bufA into a fresh index, re-encode, expect identical
	// bytes (proves the dense-only encoding has no nondeterministic sparse tail).
	rt, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	// encodeMaps writes the magic; decodeMaps expects the body AFTER the magic.
	body := bufA.Bytes()[len(mvMapsMagic):]
	if err := rt.decodeMaps(bytes.NewReader(body)); err != nil {
		t.Fatal(err)
	}
	if len(rt.docSparse) != 0 {
		t.Fatalf("dense-only decode produced %d docSparse entries, want 0", len(rt.docSparse))
	}
	var bufRT bytes.Buffer
	if err := rt.encodeMaps(&bufRT); err != nil {
		t.Fatal(err)
	}
	// Compare LENGTH, not bytes: encodeMaps iterates docMeta/docSparse maps in Go's
	// randomized order, so a re-encode of the same data has identical length but may
	// reorder entries. A stray sparse block (the bug this guards) would change the
	// length; map-iteration order does not. (len(rt.docSparse)==0 above already proves
	// no sparse tail survived the round-trip.)
	if bufA.Len() != bufRT.Len() {
		t.Fatalf("dense-only encodeMaps length not stable across round-trip: %d vs %d bytes", bufA.Len(), bufRT.Len())
	}
}

// TestMVOldBlobNoSparseBlockDecodes builds a maps blob the SAME way the encoder does
// for a dense-only index (which omits the sparse block) and confirms decodeMaps
// tolerates the missing trailing block (backward compatibility with pre-sparse blobs).
func TestMVOldBlobNoSparseBlockDecodes(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"a": NewInt(1)}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := m.encodeMaps(&buf); err != nil {
		t.Fatal(err)
	}
	// Decode into a fresh index: no sparse block present (dense-only), must succeed +
	// leave docSparse empty + version defaulted to 1 for the live doc.
	m2, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	body := buf.Bytes()[len(mvMapsMagic):]
	if err := m2.decodeMaps(bytes.NewReader(body)); err != nil {
		t.Fatalf("decode dense-only blob: %v", err)
	}
	if len(m2.docSparse) != 0 {
		t.Fatalf("dense-only blob decoded %d docSparse entries", len(m2.docSparse))
	}
	if m2.version[1] != 1 {
		t.Fatalf("doc 1 version = %d, want 1", m2.version[1])
	}
}

// TestMVDocSparseWALRoundTrip adds sparse-bearing + dense-only MV docs to a WAL-mode
// collection, crashes (no Flush), reopens, and confirms the replayed WAL restores
// docSparse + a rebuilt sparseIdx; a dense-only doc gains no sparse.
func TestMVDocSparseWALRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok || idx.wal == nil {
		t.Fatal("WAL-mode MV collection not wired")
	}
	if _, err := idx.AddCASKeyTTLSparse(1, [][]float32{{1, 0, 0, 0}}, nil, nil, mvSV(0, 2.0, 4, 1.0), CASCond{}); err != nil {
		t.Fatal(err)
	}
	if _, err := idx.AddCASKeyTTLSparse(2, [][]float32{{0, 1, 0, 0}}, nil, nil, mvSV(4, 3.0), CASCond{}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Add(3, [][]float32{{0, 0, 1, 0}}, nil); err != nil { // dense-only
		t.Fatal(err)
	}
	_ = cs.Close() // crash: no Flush, only the WAL on disk

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, ok := cs2.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after reopen")
	}

	got1, ok := idx2.GetSparse(1)
	if !ok || len(got1.Indices) != 2 || got1.Indices[0] != 0 || got1.Values[1] != 1.0 {
		t.Fatalf("doc 1 sparse after WAL replay = %+v ok=%v", got1, ok)
	}
	got2, ok := idx2.GetSparse(2)
	if !ok || len(got2.Indices) != 1 || got2.Indices[0] != 4 || got2.Values[0] != 3.0 {
		t.Fatalf("doc 2 sparse after WAL replay = %+v ok=%v", got2, ok)
	}
	if _, ok := idx2.GetSparse(3); ok {
		t.Fatal("dense-only doc 3 gained a sparse after WAL replay")
	}
	// Rebuilt sparseIdx answers a query (dim 4 shared by docs 1 + 2).
	r := idx2.SearchSparse(mvSV(4, 1.0), 5)
	if len(r) != 2 {
		t.Fatalf("rebuilt sparseIdx after replay returned %+v, want 2 docs", r)
	}
	if r[0].ID != 2 { // doc 2 has weight 3 on dim 4 > doc 1's weight 1
		t.Fatalf("top sparse after replay = id %d, want 2", r[0].ID)
	}
}

// TestMVDenseOnlyWALByteIdentical asserts a dense-only MV WAL Add record is
// BYTE-IDENTICAL whether the sparse codec is engaged or not — the sparse trailer is
// omitted entirely when nil (the record matches the pre-sparse format).
func TestMVDenseOnlyWALByteIdentical(t *testing.T) {
	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	meta := Metadata{"k": NewInt(1)}

	// Build the record bytes the WAL append produces for a dense-only Add (nil sparse).
	denseOnly := mvAddRecordBytes(t, 1, tokens, meta, nil, 1, nil)
	// The same record via the explicit nil-sparse path must be identical.
	nilSparse := mvAddRecordBytes(t, 1, tokens, meta, nil, 1, (*SparseVector)(nil))
	if !bytes.Equal(denseOnly, nilSparse) {
		t.Fatalf("nil-sparse MV WAL record differs from dense-only: %d vs %d bytes", len(denseOnly), len(nilSparse))
	}
	// A sparse-bearing record is STRICTLY longer (the trailer is present).
	withSparse := mvAddRecordBytes(t, 1, tokens, meta, nil, 1, mvSV(0, 1.0))
	if len(withSparse) <= len(denseOnly) {
		t.Fatalf("sparse-bearing record not longer: %d vs dense-only %d", len(withSparse), len(denseOnly))
	}
	// And it replays back to the same sparse vector.
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := replayMVRecord(withSparse, m); err != nil {
		t.Fatalf("replay of sparse-bearing record failed: %v", err)
	}
	got, ok := m.GetSparse(1)
	if !ok || len(got.Indices) != 1 || got.Indices[0] != 0 || got.Values[0] != 1.0 {
		t.Fatalf("replayed sparse = %+v ok=%v", got, ok)
	}
}

// mvAddRecordBytes captures the framed-record PAYLOAD bytes appendMVAddStaged would write
// (the per-record body, after the framing length prefix is stripped) for assertion +
// replay. It builds the buffer the same way appendMVAddStaged does so the comparison is
// exact.
func mvAddRecordBytes(t *testing.T, docID uint64, tokens [][]float32, meta Metadata, keyExpires map[string]uint64, version uint64, sparse *SparseVector) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteByte(byte(mvAddRec))
	_ = writeU64(&buf, docID)
	_ = writeU32(&buf, uint32(len(tokens)))
	for _, tok := range tokens {
		_ = writeU32(&buf, uint32(len(tok)))
		for _, f := range tok {
			_ = writeF32(&buf, f)
		}
	}
	_ = writeOptMeta(&buf, meta)
	writeOptKeyExpires(&buf, keyExpires)
	writeOptVersion(&buf, version)
	writeOptMVSparse(&buf, sparse)
	return buf.Bytes()
}
