// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"encoding/binary"
	"errors"
	"runtime"
	"testing"
	"time"
)

func TestSnapshotWritesMagicAndHeader(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 7, Metric: L2})
	_, _, _ = h.Insert(1, []float32{1, 2, 3, 4}, 0, nil, nil, nil, CASCond{})

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if buf.Len() < 8 {
		t.Fatalf("buffer too short: %d", buf.Len())
	}
	if string(buf.Bytes()[:8]) != snapshotMagic {
		t.Errorf("magic = %q, want %q", buf.Bytes()[:8], snapshotMagic)
	}
	gotVersion := binary.BigEndian.Uint32(buf.Bytes()[8:12])
	// This is a non-PQ (QuantNone) index, so it stays at snapshotVersionNoPQ (v6)
	// and emits no PQ block — byte-identical to pre-Task-2. Only a QuantPQ index
	// bumps to snapshotVersion (v7) and appends the codebook block.
	if gotVersion != snapshotVersionNoPQ {
		t.Errorf("version = %d, want %d", gotVersion, snapshotVersionNoPQ)
	}
}

func TestSnapshotRestoreRoundtrip(t *testing.T) {
	src, _ := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 7, Metric: L2})
	for i := uint64(0); i < 20; i++ {
		_, _, _ = src.Insert(i+1, []float32{float32(i), 0, 0, float32(i % 3)}, 0, nil, nil, nil, CASCond{})
	}

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	dst, _ := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 7, Metric: L2})
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("restore: %v", err)
	}

	if dst.maxLevel != src.maxLevel {
		t.Errorf("maxLevel: src=%d dst=%d", src.maxLevel, dst.maxLevel)
	}
	if dst.entryPoint != src.entryPoint {
		t.Errorf("entryPoint: src=%d dst=%d", src.entryPoint, dst.entryPoint)
	}
	countNodes := func(ns []*node) int {
		c := 0
		for _, n := range ns {
			if n != nil {
				c++
			}
		}
		return c
	}
	if countNodes(dst.nodes) != countNodes(src.nodes) {
		t.Errorf("node count: src=%d dst=%d", countNodes(src.nodes), countNodes(dst.nodes))
	}
	if dst.arena.Size() != src.arena.Size() {
		t.Errorf("arena Size: src=%d dst=%d", src.arena.Size(), dst.arena.Size())
	}

	query := []float32{5, 0, 0, 1}
	srcResults, _ := src.Search(query, 5)
	dstResults, _ := dst.Search(query, 5)
	if len(srcResults) != len(dstResults) {
		t.Fatalf("result count: src=%d dst=%d", len(srcResults), len(dstResults))
	}
	for i := range srcResults {
		if srcResults[i].ID != dstResults[i].ID {
			t.Errorf("rank %d: src=%d dst=%d", i, srcResults[i].ID, dstResults[i].ID)
		}
	}
}

func TestRestoreRejectsBadMagic(t *testing.T) {
	dst, _ := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 7, Metric: L2})
	buf := bytes.NewReader([]byte("NOTMAGIC" + string(make([]byte, 100))))
	err := dst.Restore(buf)
	if err == nil || !errors.Is(err, ErrSnapshotFormat) {
		t.Errorf("bad magic: got %v, want ErrSnapshotFormat", err)
	}
}

func TestRestoreRejectsBadVersion(t *testing.T) {
	dst, _ := newHNSW(Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 7, Metric: L2})
	var buf bytes.Buffer
	buf.WriteString(snapshotMagic)
	var v [4]byte
	binary.BigEndian.PutUint32(v[:], 99)
	buf.Write(v[:])
	err := dst.Restore(&buf)
	if err == nil || !errors.Is(err, ErrSnapshotFormat) {
		t.Errorf("bad version: got %v, want ErrSnapshotFormat", err)
	}
}

func TestSnapshotV2ExpiresRoundtrip(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	h.now = func() int64 { return 0 } // freeze clock so deadlines are deterministic

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 500*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	h2, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 2})
	if err != nil {
		t.Fatalf("newHNSW restore: %v", err)
	}
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	slot1, ok := h2.arena.Slot(1)
	if !ok {
		t.Fatalf("id 1 missing after restore")
	}
	if got, want := h2.arena.ExpiresAt(slot1), uint64(500); got != want {
		t.Fatalf("ExpiresAt(slot1)=%d, want %d", got, want)
	}
	slot2, ok := h2.arena.Slot(2)
	if !ok {
		t.Fatalf("id 2 missing after restore")
	}
	if got := h2.arena.ExpiresAt(slot2); got != 0 {
		t.Fatalf("ExpiresAt(slot2)=%d, want 0", got)
	}
}

func TestSnapshotPreservesTombstones(t *testing.T) {
	src, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	for i := uint64(1); i <= 5; i++ {
		_, _, _ = src.Insert(i, []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{})
	}
	src.Delete(3, CASCond{})

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	dst, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	if err := dst.Restore(&buf); err != nil {
		t.Fatal(err)
	}

	// Tombstoned id 3 must not appear in search results.
	results, _ := dst.Search([]float32{3, 0}, 5)
	for _, r := range results {
		if r.ID == 3 {
			t.Errorf("tombstoned id 3 leaked into post-restore search results: %+v", results)
		}
	}

	// Stats must report exactly one tombstone.
	st := dst.Stats()
	if st.Tombstoned != 1 {
		t.Errorf("Tombstoned = %d, want 1", st.Tombstoned)
	}
}

func TestSnapshotV3MetadataRoundtrip(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	meta1 := Metadata{
		"tenant": NewString("acme"),
		"score":  NewInt(95),
		"ratio":  NewFloat(0.5),
		"active": NewBool(true),
		"tags":   NewStrings([]string{"prod", "v2"}),
		"perms":  NewInts([]int64{4, 8}),
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, meta1, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	h2, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 2})
	if err != nil {
		t.Fatalf("newHNSW restore: %v", err)
	}
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	slot1, ok := h2.arena.Slot(1)
	if !ok {
		t.Fatal("id 1 missing after restore")
	}
	got := h2.arena.Metadata(slot1)
	for k, want := range meta1 {
		if !got[k].Equal(want) {
			t.Errorf("restored meta[%q] = %+v, want %+v", k, got[k], want)
		}
	}

	slot2, ok := h2.arena.Slot(2)
	if !ok {
		t.Fatal("id 2 missing after restore")
	}
	if m := h2.arena.Metadata(slot2); m != nil {
		t.Errorf("restored meta for id 2 = %+v, want nil", m)
	}

	// Filtered search over the restored index must honor metadata.
	res, err := h2.SearchFiltered([]float32{1, 0, 0, 0}, 10, Filter{Op: FilterEq, Field: "tenant", Value: NewString("acme")})
	if err != nil {
		t.Fatalf("SearchFiltered after restore: %v", err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Errorf("filtered search after restore = %+v, want [id=1]", res)
	}
}

// TestSnapshotGeoMetadataRoundtrip proves a ValueGeo metadata field survives a
// Snapshot->Restore cycle with EXACT lat/lon preservation. This is the
// persistence landmine: writeValue/readValue (shared by snapshot + WAL +
// persist) must carry the geo lat/lon, else a geo-bearing collection cannot be
// snapshotted/restored.
func TestSnapshotGeoMetadataRoundtrip(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	const lat, lon = 48.8566, 2.3522
	meta1 := Metadata{
		"loc":    NewGeo(lat, lon),
		"tenant": NewString("acme"),
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, meta1, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	h2, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 2})
	if err != nil {
		t.Fatalf("newHNSW restore: %v", err)
	}
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	slot1, ok := h2.arena.Slot(1)
	if !ok {
		t.Fatal("id 1 missing after restore")
	}
	got := h2.arena.Metadata(slot1)
	gv := got["loc"]
	if gv.Kind != ValueGeo {
		t.Fatalf("restored loc kind = %d, want ValueGeo", gv.Kind)
	}
	if gv.Lat != lat || gv.Lon != lon {
		t.Errorf("restored geo = (%v, %v), want (%v, %v)", gv.Lat, gv.Lon, lat, lon)
	}
	if !gv.Equal(NewGeo(lat, lon)) {
		t.Errorf("restored geo not Equal to original: %+v", gv)
	}
}

func TestSnapshotV3NoMetadataRestoresNil(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 1; i <= 5; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	h2, _ := newHNSW(Config{Dim: 2, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 2})
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if len(h2.arena.metadata) != h2.arena.Capacity() {
		t.Errorf("metadata len = %d, want capacity %d", len(h2.arena.metadata), h2.arena.Capacity())
	}
	for slot, m := range h2.arena.metadata {
		if m != nil {
			t.Errorf("slot %d metadata = %+v, want nil", slot, m)
		}
	}
	res, err := h2.Search([]float32{1, 0}, 3)
	if err != nil || len(res) == 0 {
		t.Fatalf("Search after restore: res=%v err=%v", res, err)
	}
}

func TestSnapshotV4SparseRoundtrip(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	sv := &SparseVector{Indices: []uint32{2, 5, 9}, Values: []float32{1.5, 2.5, 3.5}}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, sv, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	h2, _ := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 2})
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	slot1, _ := h2.arena.Slot(1)
	got := h2.arena.Sparse(slot1)
	if got == nil || len(got.Indices) != 3 || got.Indices[2] != 9 || got.Values[1] != 2.5 {
		t.Errorf("restored sparse(slot1) = %+v", got)
	}
	slot2, _ := h2.arena.Slot(2)
	if h2.arena.Sparse(slot2) != nil {
		t.Errorf("restored sparse(slot2) = %+v, want nil", h2.arena.Sparse(slot2))
	}

	// The inverted index must be rebuilt: a sparse search over the restored
	// index should score slot1.
	scores := h2.sparseIdx.search(SparseVector{Indices: []uint32{5}, Values: []float32{2.0}}, nil)
	if scores[slot1] != 5.0 { // 2.5 * 2.0
		t.Errorf("rebuilt index score = %v, want 5.0", scores[slot1])
	}

	// And a hybrid search over the restored index works end-to-end.
	res, err := h2.HybridSearch([]float32{1, 0, 0, 0}, SparseVector{Indices: []uint32{9}, Values: []float32{1.0}}, 5, HybridOpts{})
	if err != nil {
		t.Fatalf("HybridSearch after restore: %v", err)
	}
	found := false
	for _, r := range res {
		if r.ID == 1 {
			found = true
		}
	}
	if !found {
		t.Errorf("hybrid search after restore missing id 1: %+v", res)
	}
}

func TestSnapshotV4NoSparseRestoresNil(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1})
	for i := uint64(1); i <= 5; i++ {
		if _, _, err := h.Insert(i, []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	h2, _ := newHNSW(Config{Dim: 2, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 2})
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if len(h2.arena.sparse) != h2.arena.Capacity() {
		t.Errorf("sparse len = %d, want capacity %d", len(h2.arena.sparse), h2.arena.Capacity())
	}
	for slot, sv := range h2.arena.sparse {
		if sv != nil {
			t.Errorf("slot %d sparse = %+v, want nil", slot, sv)
		}
	}
	if got := h2.Stats().SparseVectors; got != 0 {
		t.Errorf("Stats.SparseVectors = %d, want 0", got)
	}
}

func TestStatsSparseVectors(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	for i := uint64(1); i <= 6; i++ {
		var sv *SparseVector
		if i%2 == 0 {
			sv = &SparseVector{Indices: []uint32{1}, Values: []float32{1.0}}
		}
		if _, _, err := h.Insert(i, []float32{float32(i), 0, 0, 0}, 0, nil, sv, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if got := h.Stats().SparseVectors; got != 3 {
		t.Errorf("Stats.SparseVectors = %d, want 3", got)
	}
}

// TestValueCodecRecordRoundtrip covers the [len:u32][bytes] ValueRecord case
// added to writeValue/readValue — the single codec shared by snapshot, WAL,
// and persist, so this one round-trip test exercises all three durability
// paths.
func TestValueCodecRecordRoundtrip(t *testing.T) {
	t.Run("roundtrip", func(t *testing.T) {
		want := Value{Kind: ValueRecord, Rec: []byte{0xDE, 0xAD, 0xBE, 0xEF}}
		var buf bytes.Buffer
		if err := writeValue(&buf, want); err != nil {
			t.Fatalf("writeValue: %v", err)
		}
		got, err := readValue(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("readValue: %v", err)
		}
		if got.Kind != ValueRecord {
			t.Fatalf("Kind = %v, want ValueRecord", got.Kind)
		}
		if !bytes.Equal(got.Rec, want.Rec) {
			t.Fatalf("Rec = %v, want %v", got.Rec, want.Rec)
		}
	})

	t.Run("empty record roundtrip", func(t *testing.T) {
		want := Value{Kind: ValueRecord, Rec: []byte{}}
		var buf bytes.Buffer
		if err := writeValue(&buf, want); err != nil {
			t.Fatalf("writeValue: %v", err)
		}
		got, err := readValue(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("readValue: %v", err)
		}
		if len(got.Rec) != 0 {
			t.Fatalf("Rec = %v, want empty", got.Rec)
		}
	})

	t.Run("truncated frame errors", func(t *testing.T) {
		want := Value{Kind: ValueRecord, Rec: []byte{1, 2, 3, 4, 5}}
		var buf bytes.Buffer
		if err := writeValue(&buf, want); err != nil {
			t.Fatalf("writeValue: %v", err)
		}
		// Chop off the last two payload bytes: the length prefix still claims 5
		// bytes, but only 3 are actually present.
		truncated := buf.Bytes()[:buf.Len()-2]
		if _, err := readValue(bytes.NewReader(truncated)); err == nil {
			t.Fatal("readValue on truncated frame: got nil error, want one")
		}
	})

	t.Run("oversized length errors without allocating", func(t *testing.T) {
		// kind byte + a length prefix claiming ~4 GiB, with NO payload bytes
		// behind it. bytes.Reader implements Len(), so boundedRecordBytes must
		// reject this from the length prefix alone rather than attempting
		// make([]byte, 0xFFFFFFF0) — which would otherwise blow the test's
		// memory budget or the process's.
		var frame bytes.Buffer
		frame.WriteByte(byte(ValueRecord))
		if err := writeU32(&frame, 0xFFFFFFF0); err != nil {
			t.Fatalf("writeU32: %v", err)
		}
		frameBytes := frame.Bytes()

		// testing.AllocsPerRun counts allocation EVENTS, not bytes — a single
		// make([]byte, 4<<30) would show up as "1 alloc", not as a red flag. Measure
		// heap bytes via runtime.MemStats instead, so a multi-GiB buffer can't hide.
		var before, after runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)
		if _, err := readValue(bytes.NewReader(frameBytes)); err == nil {
			t.Fatal("readValue on oversized length: got nil error, want one")
		}
		runtime.ReadMemStats(&after)

		const budget = 1 << 20 // 1 MiB — generous headroom over the few bytes this path should touch, nowhere near the ~4 GiB it would need to actually allocate the claimed length
		if got := after.TotalAlloc - before.TotalAlloc; got > budget {
			t.Fatalf("readValue allocated %d bytes for an oversized length, want < %d (no ~4 GiB buffer)", got, budget)
		}
	})
}
