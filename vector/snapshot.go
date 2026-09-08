// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	// snapshotMagic identifies a vector-index snapshot.
	snapshotMagic = "rstmhsw1"

	// snapshotVersion is the codec version. v1 = original (no expires). v2 adds
	// arena.expires after arena.vecs. v3 adds an arena.metadata block after
	// expires. v4 adds an arena.sparse block after metadata. v5 adds an
	// arena.keyExpires (per-key payload TTL) block after sparse. v6 adds an
	// arena.versions (per-point CAS version) block after keyExpires. v7 adds a PQ
	// codebook block immediately after arena.vecs (before expires): a presence byte
	// (1 only when the index is a trained QuantPQ index, else 0), then M and the
	// codebooks. It lets a restored PQ-HNSW index re-encode codes from the TRAINED
	// codebooks (ADC navigation survives restart) instead of degrading to exact
	// float. Older snapshots load with the missing blocks zero/nil-filled (no
	// per-key TTL for v<5); a v<6 snapshot defaults every live point to version 1
	// (a sane existing-point version — see readSnapshot); a v<7 snapshot has no PQ
	// block (a non-PQ index, or a PQ index re-encodes from vecs as before). A
	// non-PQ index NEVER emits v7 — it stays at v6 (snapshotVersionNoPQ) and writes
	// no PQ block at all, so a non-PQ (sq8/bq1/none) snapshot is BYTE-IDENTICAL to
	// pre-Task-2. Only a QuantPQ index writes v7 + the block. v8 appends the OPQ
	// rotation R (dim×dim floats) to the PQ block; it is stamped ONLY when the
	// trained codec has a non-nil rotation, so a PQ index WITHOUT OPQ stays at v7
	// and is BYTE-IDENTICAL to pre-OPQ (no R bytes). v7 reads back with rotation
	// nil (back-compat); v8 restores R VERBATIM so the codec rotates identically.
	snapshotVersion uint32 = 8

	// snapshotVersionPQNoOPQ is the version stamped for a trained QuantPQ index
	// WITHOUT OPQ (rotation nil): the PQ block carries codebooks but no R, byte-
	// identical to the pre-OPQ v7 layout.
	snapshotVersionPQNoOPQ uint32 = 7

	// snapshotVersionNoPQ is the version stamped for a non-PQ index (the historical
	// v6). Keeping it distinct from snapshotVersion lets a non-PQ snapshot stay
	// byte-identical to pre-Task-2 (no version bump, no PQ block).
	snapshotVersionNoPQ uint32 = 6

	// snapshotVersionPQDrop is the version stamped ONLY for a float-drop PQ-HNSW
	// index (PQDropVecs + the post-build drop fired ⇒ vecsDropped() true). It is the
	// HNSW analogue of the IVF hasVecs byte (ivf.go writeArena): a dropped index has
	// NO resident float32 vectors to serialize, so v9 OMITS the vecs float block and
	// instead writes the PQ codes VERBATIM (a codes block after the PQ codebooks),
	// exactly like the IVF snapshot serializes residual codes that can't be
	// re-derived. The PQ block (v9 always carries OPQ R when present, like v8) is
	// restored BEFORE the dropped flag is set, then the codes are loaded verbatim and
	// the arena is marked dropped WITHOUT re-encoding (there are no floats to encode
	// from). v9 is stamped ONLY when dropped — a keep-floats PQ-HNSW stays at v7/v8
	// and a non-PQ index at v6, so every existing on-disk format is BYTE-IDENTICAL.
	snapshotVersionPQDrop uint32 = 9

	// snapshotVersionSQ is the version stamped ONLY for a QuantSQ index (the
	// trained metric-agnostic scalar quantizer). It carries an SQ-ranges block
	// (presence byte, bits, per-dimension min[]/max[]) right after the PQ codebook
	// block (which a non-PQ index writes as a single 0 byte). On restore the ranges
	// rebuild the trainedSQ BEFORE the re-encode-from-vecs, so codes are identical.
	// Stamped ONLY for QuantSQ — every QuantNone/SQ8/BQ1/PQ snapshot stays at its
	// existing version (6/7/8/9) and is BYTE-IDENTICAL. A reader at v10 expects the
	// SQ block; v<10 has none.
	snapshotVersionSQ uint32 = 10

	// snapshotVersionPRQ is the version stamped ONLY for a QuantPRQ index
	// (product-residual quantization). It carries a PRQ codebook block (presence
	// byte, layer count L, then per-layer PQ codebooks, then the OPQ rotation R once
	// when present) where the PQ codebook block would be. On restore the L layers'
	// codebooks (+R) rebuild the prq codec BEFORE the re-encode-from-vecs, so codes
	// are identical. Stamped ONLY for QuantPRQ — every QuantNone/SQ8/BQ1/PQ/SQ
	// snapshot stays at its existing version (6/7/8/9/10) and is BYTE-IDENTICAL. A
	// 1-layer PRQ does NOT reuse the PQ v7/v8 layout (it is a distinct enum + a
	// distinct version), so the existing PQ snapshot bytes are untouched.
	snapshotVersionPRQ uint32 = 11
)

// errBadMagic is returned by restore when the magic bytes don't match.
var errBadMagic = fmt.Errorf("%w: bad magic", ErrSnapshotFormat)

// writeSnapshot serializes hnsw state to w. The format is positional:
//
//	[magic:8][version:u32]
//	[dim:u32][metric:u8][M:u32][efConstruction:u32][efSearch:u32][seed:i64]
//	[size:u32]                                        // live (idMap count)
//	[arenaCapacity:u32]                               // total slots, including tombstoned
//	[arenaVecs: float32×dim×arenaCapacity]            // raw arena.vecs
//	[arenaFreeCount:u32][arenaFreeList: u32×count]    // arena.free
//	[idMapCount:u32][idMap: (u64,u32)×count]          // arena.idMap as id->slot pairs
//	[tombstoneCount:u32][tombstones: u32×count]
//	[entryPoint:u32][maxLevel:i32]
//	[nodeCount:u32]
//	for each node:
//	    [slot:u32][level:i32][neighborsLevels:u32]
//	    for each level 0..levelsCount-1:
//	        [neighborCount:u32][neighbors: u32×count]
//
// Caller must hold h.mu for reading.
func (h *hnsw) writeSnapshot(w io.Writer) error {
	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(snapshotMagic); err != nil {
		return err
	}
	// The PQ codebook block (v7) is emitted ONLY for a QuantPQ index. A non-PQ
	// index (sq8/bq1/none) stays at v6 and writes NO PQ block at all, so its
	// snapshot is BYTE-IDENTICAL to pre-Task-2 (the version field is unchanged and
	// no extra bytes are appended). This preserves the hard back-compat invariant.
	pqIndex := h.cfg.Quant == QuantPQ
	sqIndex := h.cfg.Quant == QuantSQ
	prqIndex := h.cfg.Quant == QuantPRQ
	hasR := pqIndex && pqRotation(h.quant) != nil
	dropped := h.arena.vecsDropped // float-drop PQ-HNSW (implies pqIndex + trained)
	ver := snapshotVersionNoPQ
	switch {
	case dropped:
		ver = snapshotVersionPQDrop // v9: no vecs block, verbatim codes + dropped byte
	case prqIndex:
		ver = snapshotVersionPRQ // v11: PRQ codebook block (L layers + R once)
	case sqIndex:
		ver = snapshotVersionSQ // v10: SQ-ranges block (trained scalar quantizer)
	case hasR:
		ver = snapshotVersion // v8: PQ block + OPQ rotation R
	case pqIndex:
		ver = snapshotVersionPQNoOPQ // v7: PQ block, no R (byte-identical to pre-OPQ)
	}
	if err := writeU32(bw, ver); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(h.cfg.Dim)); err != nil {
		return err
	}
	if err := bw.WriteByte(byte(h.cfg.Metric)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(h.cfg.M)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(h.cfg.EfConstruction)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(h.cfg.EfSearch)); err != nil {
		return err
	}
	if err := writeI64(bw, h.cfg.Seed); err != nil {
		return err
	}

	if err := writeU32(bw, uint32(h.arena.Size())); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(h.arena.Capacity())); err != nil {
		return err
	}
	// v9 (float-drop): the resident floats are gone — OMIT the vecs block entirely
	// (mirrors the IVF hasVecs=0 path). v6/v7/v8 write the full vecs block here,
	// byte-identically to before.
	if !dropped {
		if err := writeF32s(bw, h.arena.vecs); err != nil {
			return err
		}
	}
	// v7: PQ codebook block — written ONLY for a QuantPQ index (a non-PQ index is
	// v6 and emits nothing here, staying byte-identical to pre-Task-2). The presence
	// byte inside the block is 1 for a trained codec, 0 for an untrained one (a PQ
	// index snapshotted before BuildConcurrent — its codes re-encode from vecs as
	// before, the exact-float path). For v9 the codebooks are written WITHOUT R here;
	// R + the verbatim codes follow in the dropped trailer below so the reader can
	// load codebooks+R and the codes verbatim WITHOUT re-encoding (no floats exist).
	if pqIndex {
		if err := writePQCodebooks(bw, h.quant, hasR && !dropped); err != nil {
			return err
		}
	}
	// v10: SQ-ranges block — written ONLY for a QuantSQ index (a non-SQ index emits
	// nothing here, staying byte-identical). The presence byte is 1 for a trained
	// quantizer, 0 for an untrained one (a QuantSQ index snapshotted before the
	// auto-train threshold — its codes re-encode from vecs as before). On restore
	// the ranges rebuild the trainedSQ BEFORE the re-encode below, so the re-encode
	// yields identical codes.
	if sqIndex {
		if err := writeSQRanges(bw, h.quant); err != nil {
			return err
		}
	}
	// v11: PRQ codebook block — written ONLY for a QuantPRQ index (every other mode
	// emits nothing here, staying byte-identical). The presence byte is 1 for a
	// trained codec, 0 for an untrained one (a QuantPRQ index snapshotted before the
	// train threshold — its codes re-encode from vecs as before). On restore the L
	// layers' codebooks (+R once) rebuild the prq codec BEFORE the re-encode below,
	// so the re-encode yields identical codes.
	if prqIndex {
		if err := writePRQCodebooks(bw, h.quant); err != nil {
			return err
		}
	}
	// v9 dropped trailer: [hasR byte][R floats?][codeLen u32][len(codes) u32][codes].
	// The codes are written VERBATIM — they are the ONLY source of truth after the
	// drop (no floats to re-encode from), exactly like the IVF snapshot serializes
	// residual codes. R is carried here (not in the codebook block) so the reader can
	// loadCodebooks(cb, R) in one shot before marking the arena dropped.
	if dropped {
		rb := byte(0)
		if hasR {
			rb = 1
		}
		if err := writeByte(bw, rb); err != nil {
			return err
		}
		if hasR {
			for _, v := range pqRotation(h.quant) {
				if err := writeF32(bw, v); err != nil {
					return err
				}
			}
		}
		if err := writeU32(bw, uint32(h.arena.codeLen)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(h.arena.codes))); err != nil {
			return err
		}
		if _, err := bw.Write(h.arena.codes); err != nil {
			return err
		}
	}
	// v2: arena.expires (one uint64 per slot; len == arena.Capacity()).
	for _, e := range h.arena.expires {
		if err := writeU64(bw, e); err != nil {
			return err
		}
	}
	// v3: arena.metadata. Only slots WITH metadata are written, prefixed by a
	// count, to avoid emitting nils for the common no-metadata case.
	withMeta := 0
	for _, m := range h.arena.metadata {
		if m != nil {
			withMeta++
		}
	}
	if err := writeU32(bw, uint32(withMeta)); err != nil {
		return err
	}
	for slot, m := range h.arena.metadata {
		if m == nil {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(m))); err != nil {
			return err
		}
		for key, val := range m {
			if err := writeString(bw, key); err != nil {
				return err
			}
			if err := writeValue(bw, val); err != nil {
				return err
			}
		}
	}
	// v4: arena.sparse. Only slots WITH a sparse vector are written. The
	// inverted index is NOT serialized — it is rebuilt from these on restore.
	withSparse := 0
	for _, sv := range h.arena.sparse {
		if sv != nil {
			withSparse++
		}
	}
	if err := writeU32(bw, uint32(withSparse)); err != nil {
		return err
	}
	for slot, sv := range h.arena.sparse {
		if sv == nil {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(sv.Indices))); err != nil {
			return err
		}
		for i, dim := range sv.Indices {
			if err := writeU32(bw, dim); err != nil {
				return err
			}
			if err := writeF32(bw, sv.Values[i]); err != nil {
				return err
			}
		}
	}
	// v5: arena.keyExpires (per-key payload TTL). Only slots WITH a non-empty
	// per-key deadline map are written, prefixed by a count — absolute unix-ms
	// deadlines preserved verbatim so restore is time-stable. Old (v<5) snapshots
	// have no block; restore zero-fills keyExpires (no per-key TTL).
	withKeyExp := 0
	for _, ke := range h.arena.keyExpires {
		if len(ke) > 0 {
			withKeyExp++
		}
	}
	if err := writeU32(bw, uint32(withKeyExp)); err != nil {
		return err
	}
	for slot, ke := range h.arena.keyExpires {
		if len(ke) == 0 {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(len(ke))); err != nil {
			return err
		}
		for key, dl := range ke {
			if err := writeString(bw, key); err != nil {
				return err
			}
			if err := writeU64(bw, dl); err != nil {
				return err
			}
		}
	}
	// v6: arena.versions (per-point CAS version). Present-only [slot u32, version
	// u64] pairs, so a no-version index pays just a zero count. Versions are
	// written verbatim and restored verbatim (NOT re-bumped) so a point's version
	// survives restart time-stable.
	withVer := 0
	for _, v := range h.arena.versions {
		if v != 0 {
			withVer++
		}
	}
	if err := writeU32(bw, uint32(withVer)); err != nil {
		return err
	}
	for slot, v := range h.arena.versions {
		if v == 0 {
			continue
		}
		if err := writeU32(bw, uint32(slot)); err != nil {
			return err
		}
		if err := writeU64(bw, v); err != nil {
			return err
		}
	}
	if err := writeU32(bw, uint32(len(h.arena.free))); err != nil {
		return err
	}
	for _, s := range h.arena.free {
		if err := writeU32(bw, s); err != nil {
			return err
		}
	}
	if err := writeU32(bw, uint32(len(h.arena.idMap))); err != nil {
		return err
	}
	for id, slot := range h.arena.idMap {
		if err := writeU64(bw, id); err != nil {
			return err
		}
		if err := writeU32(bw, slot); err != nil {
			return err
		}
	}

	if err := writeU32(bw, uint32(len(h.tombstoned))); err != nil {
		return err
	}
	for s := range h.tombstoned {
		if err := writeU32(bw, s); err != nil {
			return err
		}
	}

	if err := writeU32(bw, h.entryPoint); err != nil {
		return err
	}
	if err := writeI32(bw, int32(h.maxLevel)); err != nil {
		return err
	}
	nodeCount := 0
	for _, n := range h.nodes {
		if n != nil {
			nodeCount++
		}
	}
	if err := writeU32(bw, uint32(nodeCount)); err != nil { //nolint:gosec // bounded by arena capacity
		return err
	}
	for _, n := range h.nodes {
		if n == nil {
			continue
		}
		if err := writeU32(bw, n.slot); err != nil {
			return err
		}
		if err := writeI32(bw, int32(n.level)); err != nil {
			return err
		}
		if err := writeU32(bw, uint32(n.level+1)); err != nil {
			return err
		}
		for lc := 0; lc <= n.level; lc++ {
			lvl := h.nbrsAt(n, lc)
			if err := writeU32(bw, uint32(len(lvl))); err != nil {
				return err
			}
			for _, nb := range lvl {
				if err := writeU32(bw, nb); err != nil {
					return err
				}
			}
		}
	}
	return bw.Flush()
}

// Snapshot writes the index's state to w under the read lock, plus the
// exclusive side of the link barrier.
//
// The read lock alone is no longer enough. An insert's link phase runs under the
// READ lock too (see link_stripes.go), so without the barrier a snapshot could
// serialize a node whose edges were still being written — a point that is fully
// present in the arena but has an incomplete, possibly empty, adjacency list,
// which restores as an ORPHAN: retrievable by id, unreachable by search, forever.
// linkMu makes every serialized graph a fully-linked one. It costs a snapshot
// only the tail of at most one in-flight link phase.
// The barrier is taken BEFORE h.mu, which is the global order (see
// link_stripes.go) and is not interchangeable: an insert holds the barrier
// across the lock-free gap between its placement and its link phase, so a
// Snapshot that took h.mu first would be holding a read lock while waiting for
// a linker that may itself be waiting to re-enter h.mu behind a queued writer.
func (h *hnsw) Snapshot(w io.Writer) error {
	h.linkMu.Lock()
	defer h.linkMu.Unlock()
	h.mu.RLock()
	defer h.mu.RUnlock()
	// A failed mmap slab growth freed the backing region; refuse to serialize
	// rather than emit a silently truncated snapshot (Capacity()/vecs are 0 after
	// the poison). This is called from the raft FSM snapshot and backup paths,
	// both in goroutines with no panic recovery, so a broken checkpoint here would
	// otherwise be produced silently. Checked under h.mu (closes the TOCTOU with a
	// concurrent grow) — see arena.poisoned / ErrIndexPoisoned.
	if h.arena.poisoned.Load() {
		return ErrIndexPoisoned
	}
	return h.writeSnapshot(w)
}

// writePQCodebooks serializes the PQ codebooks of a TRAINED QuantPQ index so a
// restored PQ-HNSW index can re-encode codes from the trained codebooks (ADC
// navigation survives restart) instead of degrading to exact-float. The format
// mirrors the IVF writePQTrailer: a presence byte, then M, then per-subspace
// [count u32, count×dsub floats] (each subspace length-prefixed because small-n
// training can yield < 256 sub-centroids). A non-PQ quantizer, or an UNTRAINED PQ
// quantizer, writes a single 0 byte (no block) — its snapshot stays
// behaviour-identical (codes re-encode from vecs as before). Shared by the
// snapshot stream and the persist sidecar.
// pqRotation returns the OPQ rotation R of a trained pqQuantizer, or nil (non-PQ,
// untrained, or OPQ off). Used by the snapshot/sidecar writers to decide the
// version stamp (R present ⇒ v8 / sidecar v8).
func pqRotation(quant quantizer) []float32 {
	pq, ok := quant.(*pqQuantizer)
	if !ok || !pq.trained() {
		return nil
	}
	return pq.rotation()
}

// writePQCodebooks serializes the PQ codebooks; when writeR is true it appends
// the OPQ rotation R (dim×dim floats, VERBATIM) after the codebooks. writeR is
// gated by the caller's version stamp (v8 / sidecar v8) and is true ONLY when the
// codec has a non-nil rotation, so a PQ index WITHOUT OPQ writes the exact pre-OPQ
// bytes (writeR=false ⇒ no R). The reader is told to expect R by the version.
func writePQCodebooks(w io.Writer, quant quantizer, writeR bool) error {
	pq, ok := quant.(*pqQuantizer)
	if !ok || !pq.trained() {
		return writeByte(w, 0)
	}
	if err := writeByte(w, 1); err != nil {
		return err
	}
	cb := pq.codebooks()
	if err := writeU32(w, uint32(len(cb))); err != nil { //nolint:gosec // m is small
		return err
	}
	for s := range cb {
		if err := writeU32(w, uint32(len(cb[s]))); err != nil { //nolint:gosec // <= 256
			return err
		}
		for _, sub := range cb[s] {
			for _, v := range sub {
				if err := writeF32(w, v); err != nil {
					return err
				}
			}
		}
	}
	if writeR {
		for _, v := range pq.rotation() {
			if err := writeF32(w, v); err != nil {
				return err
			}
		}
	}
	return nil
}

// readPQCodebooks reads the block written by writePQCodebooks and, when present,
// loads the codebooks into quant (a *pqQuantizer) so its codec is TRAINED before
// the caller re-encodes codes from the restored vectors. A leading 0 byte means
// no PQ block (non-PQ index, or an untrained PQ index at snapshot time): a no-op.
// A present block with a non-PQ quant is a format error (the index this restore
// targets must be QuantPQ). Shared by the snapshot stream and the persist sidecar.
// hasR tells the reader whether the PQ block is followed by an OPQ rotation R
// (v8 / sidecar v8). The codebooks and R are restored together so the codec
// rotates/un-rotates VERBATIM.
func readPQCodebooks(r io.Reader, quant quantizer, hasR bool) error {
	pq, _ := quant.(*pqQuantizer)
	dim := 0
	if pq != nil {
		dim = pq.dim
	}
	cb, rot, err := readPQCodebooksRaw(r, dim, hasR)
	if err != nil {
		return err
	}
	if cb == nil {
		return nil // no block
	}
	if pq == nil {
		return fmt.Errorf("%w: PQ codebook block but index is not QuantPQ", ErrSnapshotFormat)
	}
	pq.loadCodebooks(cb, rot)
	return nil
}

// readPQCodebooksRaw reads the presence-gated PQ block and returns the codebooks
// ([m][≤256][dsub]), or nil when the presence byte is 0 (no block). dim is the
// index dimension (used to derive dsub = dim/m and validate the block). The
// persist sidecar uses this directly because its quantizer shell does not exist
// yet when the block is read; the snapshot path goes through readPQCodebooks.
// hasR, when true (v8 / sidecar v8), means dim×dim rotation floats follow the
// codebooks; they are returned VERBATIM as the second result (nil when hasR is
// false or the block is absent).
func readPQCodebooksRaw(r io.Reader, dim int, hasR bool) ([][][]float32, []float32, error) {
	on, err := readByte(r)
	if err != nil {
		return nil, nil, err
	}
	if on == 0 {
		return nil, nil, nil
	}
	mVal, err := readU32(r)
	if err != nil {
		return nil, nil, err
	}
	m := int(mVal)
	if m <= 0 || dim <= 0 || dim%m != 0 {
		return nil, nil, ErrSnapshotFormat
	}
	dsub := dim / m
	cb := make([][][]float32, m)
	for s := 0; s < m; s++ {
		nc, err := readU32(r)
		if err != nil {
			return nil, nil, err
		}
		// Bound the per-subspace centroid count against the codec maximum
		// (8-bit PQ ⇒ pqCodebookSize=256; 4-bit ⇒ 16, a subset). A corrupt nc
		// would otherwise drive an enormous allocation, or — read as int on a
		// 32-bit platform — a negative length that panics in make().
		if nc == 0 || nc > pqCodebookSize {
			return nil, nil, ErrSnapshotFormat
		}
		sub := make([][]float32, int(nc))
		for c := range sub {
			vec := make([]float32, dsub)
			for i := range vec {
				f, ferr := readF32(r)
				if ferr != nil {
					return nil, nil, ferr
				}
				vec[i] = f
			}
			sub[c] = vec
		}
		cb[s] = sub
	}
	var rot []float32
	if hasR {
		rot = make([]float32, dim*dim)
		for i := range rot {
			f, ferr := readF32(r)
			if ferr != nil {
				return nil, nil, ferr
			}
			rot[i] = f
		}
	}
	return cb, rot, nil
}

// writeSQRanges serializes the learned per-dimension ranges of a TRAINED QuantSQ
// index so a restored HNSW-SQ index re-encodes codes from the SAME ranges
// (identical codes) instead of degrading to exact-float. Format: a presence byte
// (1 only for a trained trainedSQ, else 0), then bits (u32), then dim min floats
// and dim max floats. A non-SQ quantizer, or an UNTRAINED one, writes a single 0
// byte (no block). Shared by the snapshot stream and the persist sidecar.
func writeSQRanges(w io.Writer, quant quantizer) error {
	sq, ok := quant.(*trainedSQ)
	if !ok || !sq.trained() {
		return writeByte(w, 0)
	}
	if err := writeByte(w, 1); err != nil {
		return err
	}
	if err := writeU32(w, uint32(sq.bits)); err != nil { //nolint:gosec // bits is small
		return err
	}
	for _, v := range sq.min {
		if err := writeF32(w, v); err != nil {
			return err
		}
	}
	for _, v := range sq.max {
		if err := writeF32(w, v); err != nil {
			return err
		}
	}
	return nil
}

// readSQRanges reads the block written by writeSQRanges and, when present, loads
// the ranges into quant (a *trainedSQ) so its encoder is TRAINED before the
// caller re-encodes codes from the restored vectors. A leading 0 byte means no
// block (non-SQ index, or an untrained SQ index at snapshot time): a no-op. A
// present block with a non-SQ quant is a format error. dim is the index
// dimension (each of min/max is dim floats). Shared by snapshot + persist.
func readSQRanges(r io.Reader, quant quantizer, dim int) error {
	min, max, bits, err := readSQRangesRaw(r, dim)
	if err != nil {
		return err
	}
	if min == nil {
		return nil // no block
	}
	sq, ok := quant.(*trainedSQ)
	if !ok {
		return fmt.Errorf("%w: SQ ranges block but index is not QuantSQ", ErrSnapshotFormat)
	}
	// Apply the serialized bit-depth (defense-in-depth, MN2): the bits written
	// alongside the ranges come from the SAME trained quantizer, so a 4/6-bit SQ
	// index decodes its bit-packed codes correctly even if the rebuilt config's
	// SQBits were wrong. On a correctly-configured restore this AGREES with the
	// factory bits (both derive from the collection's SQBits).
	sq.loadRangesBits(min, max, bits)
	return nil
}

// readSQRangesRaw reads the presence-gated SQ-ranges block and returns the
// per-dimension min[]/max[] (each dim long) and bits, or nil min when the
// presence byte is 0 (no block). The persist sidecar uses this directly; the
// snapshot path goes through readSQRanges.
func readSQRangesRaw(r io.Reader, dim int) (min, max []float32, bits int, err error) {
	on, err := readByte(r)
	if err != nil {
		return nil, nil, 0, err
	}
	if on == 0 {
		return nil, nil, 0, nil
	}
	b, err := readU32(r)
	if err != nil {
		return nil, nil, 0, err
	}
	if dim <= 0 {
		return nil, nil, 0, ErrSnapshotFormat
	}
	min = make([]float32, dim)
	for i := range min {
		f, ferr := readF32(r)
		if ferr != nil {
			return nil, nil, 0, ferr
		}
		min[i] = f
	}
	max = make([]float32, dim)
	for i := range max {
		f, ferr := readF32(r)
		if ferr != nil {
			return nil, nil, 0, ferr
		}
		max[i] = f
	}
	return min, max, int(b), nil
}

// writePRQCodebooks serializes the L-layer codebooks of a TRAINED QuantPRQ index so
// a restored PRQ-HNSW index re-encodes codes from the trained codebooks (summed-LUT
// ADC navigation survives restart) instead of degrading to exact-float. Format: a
// presence byte (1 for a trained codec, else 0), then the layer count L (u32), then
// an R-present byte; then for each layer the per-subspace codebook block (M u32,
// then per-subspace [count u32, count×dsub floats]); then, when the R-present byte
// is 1, the OPQ rotation R (dim×dim floats, VERBATIM) ONCE at the end (it is layer
// 0's rotation, applied once). A non-PRQ quantizer, or an UNTRAINED one, writes a
// single 0 byte (no block). Shared by the snapshot stream and the persist sidecar.
func writePRQCodebooks(w io.Writer, quant quantizer) error {
	prq, ok := quant.(*prqQuantizer)
	if !ok || !prq.trained() {
		return writeByte(w, 0)
	}
	if err := writeByte(w, 1); err != nil {
		return err
	}
	cbAll := prq.codebooks()                                // [L][m][≤256][dsub]
	if err := writeU32(w, uint32(len(cbAll))); err != nil { //nolint:gosec // L is small
		return err
	}
	r := prq.rotation()
	rb := byte(0)
	if r != nil {
		rb = 1
	}
	if err := writeByte(w, rb); err != nil {
		return err
	}
	for _, cb := range cbAll {
		if err := writeU32(w, uint32(len(cb))); err != nil { //nolint:gosec // m is small
			return err
		}
		for s := range cb {
			if err := writeU32(w, uint32(len(cb[s]))); err != nil { //nolint:gosec // <= 256
				return err
			}
			for _, sub := range cb[s] {
				for _, v := range sub {
					if err := writeF32(w, v); err != nil {
						return err
					}
				}
			}
		}
	}
	if rb == 1 {
		for _, v := range r {
			if err := writeF32(w, v); err != nil {
				return err
			}
		}
	}
	return nil
}

// readPRQCodebooks reads the block written by writePRQCodebooks and, when present,
// loads the L layers' codebooks (+R) into quant (a *prqQuantizer) so its codec is
// TRAINED before the caller re-encodes codes from the restored vectors. A leading 0
// byte means no block (non-PRQ index, or an untrained PRQ index at snapshot time): a
// no-op. A present block with a non-PRQ quant is a format error. Shared by the
// snapshot stream and the persist sidecar.
func readPRQCodebooks(r io.Reader, quant quantizer, dim int) error {
	cb, rot, err := readPRQCodebooksRaw(r, dim)
	if err != nil {
		return err
	}
	if cb == nil {
		return nil // no block
	}
	prq, ok := quant.(*prqQuantizer)
	if !ok {
		return fmt.Errorf("%w: PRQ codebook block but index is not QuantPRQ", ErrSnapshotFormat)
	}
	prq.loadCodebooks(cb, rot)
	return nil
}

// readPRQCodebooksRaw reads the presence-gated PRQ block and returns the L-layer
// codebooks ([L][m][≤256][dsub]) and the OPQ rotation R (nil when absent), or nil cb
// when the presence byte is 0 (no block). dim is the index dimension (dsub = dim/m
// per layer; R is dim×dim). The persist sidecar uses this directly; the snapshot
// path goes through readPRQCodebooks.
func readPRQCodebooksRaw(r io.Reader, dim int) ([][][][]float32, []float32, error) {
	on, err := readByte(r)
	if err != nil {
		return nil, nil, err
	}
	if on == 0 {
		return nil, nil, nil
	}
	lVal, err := readU32(r)
	if err != nil {
		return nil, nil, err
	}
	l := int(lVal)
	if l <= 0 || dim <= 0 {
		return nil, nil, ErrSnapshotFormat
	}
	rb, err := readByte(r)
	if err != nil {
		return nil, nil, err
	}
	cbAll := make([][][][]float32, l)
	for layer := 0; layer < l; layer++ {
		mVal, merr := readU32(r)
		if merr != nil {
			return nil, nil, merr
		}
		m := int(mVal)
		if m <= 0 || dim%m != 0 {
			return nil, nil, ErrSnapshotFormat
		}
		dsub := dim / m
		cb := make([][][]float32, m)
		for s := 0; s < m; s++ {
			nc, ncerr := readU32(r)
			if ncerr != nil {
				return nil, nil, ncerr
			}
			// Bound the per-subspace centroid count against the codec maximum
			// (PRQ layers use the 8-bit PQ codec ⇒ pqCodebookSize=256); reject a
			// corrupt nc before the huge/negative-length allocation.
			if nc == 0 || nc > pqCodebookSize {
				return nil, nil, ErrSnapshotFormat
			}
			sub := make([][]float32, int(nc))
			for c := range sub {
				vec := make([]float32, dsub)
				for i := range vec {
					f, ferr := readF32(r)
					if ferr != nil {
						return nil, nil, ferr
					}
					vec[i] = f
				}
				sub[c] = vec
			}
			cb[s] = sub
		}
		cbAll[layer] = cb
	}
	var rot []float32
	if rb == 1 {
		rot = make([]float32, dim*dim)
		for i := range rot {
			f, ferr := readF32(r)
			if ferr != nil {
				return nil, nil, ferr
			}
			rot[i] = f
		}
	}
	return cbAll, rot, nil
}

// writeU32 writes v as 4 big-endian bytes. When w is an io.ByteWriter (the
// snapshot stream and persist sidecar both wrap their destination in a
// bufio.Writer), it appends byte-by-byte into the buffer — NO per-call slice
// allocation. The snapshot writes millions of scalars (one writeF32 per vector
// element, one writeU32 per graph edge), so the old `var b [4]byte; w.Write(b[:])`
// form (b escapes through the io.Writer interface ⇒ one heap alloc per call) cost
// ~3.2M allocs per snapshot. Output bytes are identical (MSB-first big-endian).
func writeU32(w io.Writer, v uint32) error {
	if bw, ok := w.(io.ByteWriter); ok {
		if err := bw.WriteByte(byte(v >> 24)); err != nil {
			return err
		}
		if err := bw.WriteByte(byte(v >> 16)); err != nil {
			return err
		}
		if err := bw.WriteByte(byte(v >> 8)); err != nil {
			return err
		}
		return bw.WriteByte(byte(v))
	}
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func writeU64(w io.Writer, v uint64) error {
	if bw, ok := w.(io.ByteWriter); ok {
		for s := 56; s >= 0; s -= 8 {
			if err := bw.WriteByte(byte(v >> uint(s))); err != nil {
				return err
			}
		}
		return nil
	}
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func writeI32(w io.Writer, v int32) error { return writeU32(w, uint32(v)) }
func writeI64(w io.Writer, v int64) error { return writeU64(w, uint64(v)) }

func writeF32(w io.Writer, v float32) error { return writeU32(w, math.Float32bits(v)) }

// writeF32s writes len(vals) big-endian float32s, byte-identical to a writeF32
// loop, but encodes through a single reused stack scratch and writes in 4 KiB
// chunks — so the snapshot's largest array (the flat arena vec block, millions of
// floats) costs ONE small heap alloc (the scratch escaping through w.Write) rather
// than a per-scalar WriteByte chain. The scalar writeF32/ByteWriter path stays for
// the small/scattered floats; this is only worth it for the bulk vec write.
func writeF32s(w io.Writer, vals []float32) error {
	var scratch [4096]byte // 1024 floats per chunk
	n := 0
	for _, v := range vals {
		binary.BigEndian.PutUint32(scratch[n:], math.Float32bits(v))
		n += 4
		if n == len(scratch) {
			if _, err := w.Write(scratch[:]); err != nil {
				return err
			}
			n = 0
		}
	}
	if n > 0 {
		_, err := w.Write(scratch[:n])
		return err
	}
	return nil
}

// writeByte writes a single byte to any io.Writer (the PQ presence flag is shared
// by the bufio-backed snapshot stream and the persist sidecar, both io.Writer).
func writeByte(w io.Writer, b byte) error {
	if bw, ok := w.(io.ByteWriter); ok {
		return bw.WriteByte(b)
	}
	_, err := w.Write([]byte{b})
	return err
}

// readByte reads a single byte from any io.Reader (PQ presence flag).
func readByte(r io.Reader) (byte, error) {
	if br, ok := r.(io.ByteReader); ok {
		return br.ReadByte()
	}
	var b [1]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return b[0], nil
}

func writeF64(w io.Writer, v float64) error {
	return writeU64(w, math.Float64bits(v))
}

func readF64(r io.Reader) (float64, error) {
	bits, err := readU64(r)
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(bits), nil
}

func writeString(w io.Writer, s string) error {
	if err := writeU32(w, uint32(len(s))); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

func readString(r io.Reader) (string, error) {
	n, err := readU32(r)
	if err != nil {
		return "", err
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// writeValue serializes a Value via its closed tagged union:
//
//	[kind:u8]
//	string:  [len:u32][bytes]
//	int:     [i64]
//	float:   [f64]
//	bool:    [u8]
//	strings: [count:u32] then each [len:u32][bytes]
//	ints:    [count:u32] then each [i64]
//	floats:  [count:u32] then each [f64]
//	geo:     [lat:f64][lon:f64]
//	record:  [len:u32][bytes]
func writeValue(w io.Writer, v Value) error {
	if _, err := w.Write([]byte{byte(v.Kind)}); err != nil {
		return err
	}
	switch v.Kind {
	case ValueString:
		return writeString(w, v.Str)
	case ValueInt:
		return writeU64(w, uint64(v.Int))
	case ValueFloat:
		return writeF64(w, v.Flt)
	case ValueBool:
		b := byte(0)
		if v.Bool {
			b = 1
		}
		_, err := w.Write([]byte{b})
		return err
	case ValueStrings:
		if err := writeU32(w, uint32(len(v.Strs))); err != nil {
			return err
		}
		for _, s := range v.Strs {
			if err := writeString(w, s); err != nil {
				return err
			}
		}
		return nil
	case ValueInts:
		if err := writeU32(w, uint32(len(v.Ints))); err != nil {
			return err
		}
		for _, n := range v.Ints {
			if err := writeU64(w, uint64(n)); err != nil {
				return err
			}
		}
		return nil
	case ValueFloats:
		if err := writeU32(w, uint32(len(v.Flts))); err != nil {
			return err
		}
		for _, f := range v.Flts {
			if err := writeF64(w, f); err != nil {
				return err
			}
		}
		return nil
	case ValueGeo:
		// lat then lon, each a big-endian float64 (mirrors ValueFloat's encoding).
		// Shared by snapshot + WAL + persist, so this single case makes a
		// geo-bearing collection durable on every path.
		if err := writeF64(w, v.Lat); err != nil {
			return err
		}
		return writeF64(w, v.Lon)
	case ValueRecord:
		// [len:u32][bytes], mirroring ValueString above. This one function pair
		// serves WAL, snapshot, and persist, so this case makes a record-bearing
		// collection durable on every path.
		if err := writeU32(w, uint32(len(v.Rec))); err != nil {
			return err
		}
		_, err := w.Write(v.Rec)
		return err
	case ValueNone:
		return nil
	default:
		return fmt.Errorf("%w: unknown value kind %d", ErrSnapshotFormat, v.Kind)
	}
}

// sizedReader is implemented by readers that can report how many bytes remain
// unread (e.g. *bytes.Reader). readValue's ValueRecord case uses it to reject
// an oversized length prefix BEFORE allocating — a hostile or corrupt frame
// otherwise drives an attacker/corruption-controlled make([]byte, n) up to
// 4 GiB before io.ReadFull ever gets a chance to fail on the short read. Only
// readers that expose this — untrusted callers should pass one — get the
// protection; a plain io.Reader (e.g. bufio.Reader) falls back to the same
// allocate-then-fail behavior every other case here already has.
type sizedReader interface{ Len() int }

func boundedRecordBytes(r io.Reader, n uint32) ([]byte, error) {
	if sr, ok := r.(sizedReader); ok && uint64(n) > uint64(sr.Len()) {
		return nil, fmt.Errorf("%w: record length %d exceeds %d remaining bytes", ErrSnapshotFormat, n, sr.Len())
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func readValue(r io.Reader) (Value, error) {
	var kb [1]byte
	if _, err := io.ReadFull(r, kb[:]); err != nil {
		return Value{}, err
	}
	kind := ValueKind(kb[0])
	v := Value{Kind: kind}
	switch kind {
	case ValueString:
		s, err := readString(r)
		if err != nil {
			return Value{}, err
		}
		v.Str = s
	case ValueInt:
		n, err := readU64(r)
		if err != nil {
			return Value{}, err
		}
		v.Int = int64(n)
	case ValueFloat:
		f, err := readF64(r)
		if err != nil {
			return Value{}, err
		}
		v.Flt = f
	case ValueBool:
		var bb [1]byte
		if _, err := io.ReadFull(r, bb[:]); err != nil {
			return Value{}, err
		}
		v.Bool = bb[0] == 1
	case ValueStrings:
		n, err := readU32(r)
		if err != nil {
			return Value{}, err
		}
		v.Strs = make([]string, n)
		for i := range v.Strs {
			s, err := readString(r)
			if err != nil {
				return Value{}, err
			}
			v.Strs[i] = s
		}
	case ValueInts:
		n, err := readU32(r)
		if err != nil {
			return Value{}, err
		}
		v.Ints = make([]int64, n)
		for i := range v.Ints {
			u, err := readU64(r)
			if err != nil {
				return Value{}, err
			}
			v.Ints[i] = int64(u)
		}
	case ValueFloats:
		n, err := readU32(r)
		if err != nil {
			return Value{}, err
		}
		v.Flts = make([]float64, n)
		for i := range v.Flts {
			f, err := readF64(r)
			if err != nil {
				return Value{}, err
			}
			v.Flts[i] = f
		}
	case ValueGeo:
		lat, err := readF64(r)
		if err != nil {
			return Value{}, err
		}
		lon, err := readF64(r)
		if err != nil {
			return Value{}, err
		}
		v.Lat = lat
		v.Lon = lon
	case ValueRecord:
		n, err := readU32(r)
		if err != nil {
			return Value{}, err
		}
		buf, err := boundedRecordBytes(r, n)
		if err != nil {
			return Value{}, err
		}
		v.Rec = buf
	case ValueNone:
		// no payload
	default:
		return Value{}, fmt.Errorf("%w: unknown value kind %d", ErrSnapshotFormat, kind)
	}
	return v, nil
}

// readSnapshot deserializes the bytes from r into h. Caller must hold
// h.mu for writing. Wraps malformed input with ErrSnapshotFormat.
func (h *hnsw) readSnapshot(r io.Reader) error {
	br := bufio.NewReader(r)

	var magic [8]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return fmt.Errorf("%w: %v", ErrSnapshotFormat, err)
	}
	if string(magic[:]) != snapshotMagic {
		return errBadMagic
	}
	version, err := readU32(br)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrSnapshotFormat, err)
	}
	if version < 1 || version > snapshotVersionPRQ {
		return fmt.Errorf("%w: version %d", ErrSnapshotFormat, version)
	}

	dim, err := readU32(br)
	if err != nil {
		return err
	}
	metric, err := br.ReadByte()
	if err != nil {
		return err
	}
	mVal, err := readU32(br)
	if err != nil {
		return err
	}
	efC, err := readU32(br)
	if err != nil {
		return err
	}
	efS, err := readU32(br)
	if err != nil {
		return err
	}
	seed, err := readI64(br)
	if err != nil {
		return err
	}
	// Start from the TARGET index's config and overwrite ONLY the fields the
	// stream carries. The snapshot header is exactly six fields wide — Dim,
	// Metric, M, EfConstruction, EfSearch, Seed (writeSnapshot writes these and
	// nothing else) — and EVERY other Config field describes the index this
	// Restore targets, not the snapshot.
	//
	// This used to build a fresh Config{} from the six and hand-list the ones
	// worth back-filling, which silently zeroed everything nobody remembered to
	// list. Fields read LIVE off h.cfg on the write path were the casualties:
	// Level0FullDegree (forwardM picked M instead of 2*M at level 0 for every
	// post-restore insert), MaxVectors/MaxBytes (quota silently lifted, while
	// Collection.Config — a separate copy — kept reporting it), QuantizedBuild,
	// ExtendCandidates/ExtendCandidatesMax, MaxEfSearch, PQDropVecs,
	// FilterFirstThreshold/FilterFirstRelativeBP. Copy-then-overwrite inverts the
	// default to PRESERVED, so a Config field added later cannot regress this by
	// omission — the only fields that need thought are the six below.
	//
	// Load-bearing preservations that used to be explicit and are now covered by
	// the copy: the quantizer shape (Quant/RescoreFactor/SQBits/PRQLayers/
	// QuantPQM/OPQ) — quantization is not in the stream, and Validate plus the
	// code re-encode below both need it — and IndexType + Vamana geometry, which
	// drive the slab stride: effectiveM0(cfg) returns VamanaR for IndexVamana vs
	// 2*M for HNSW, so losing them would re-presize the graph with the WRONG
	// stride and corrupt every neighbor list. (mL stays 0 for Vamana's
	// single-layer pinning — see below; pruneAlpha/vamana are set at construction
	// and untouched here.)
	cfg := h.cfg
	cfg.Dim = int(dim)
	cfg.Metric = Metric(metric)
	cfg.M = int(mVal)
	cfg.EfConstruction = int(efC)
	cfg.EfSearch = int(efS)
	cfg.Seed = seed
	if err := ValidateConfig(cfg); err != nil {
		return fmt.Errorf("%w: %v", ErrSnapshotFormat, err)
	}
	h.cfg = cfg
	// Single-layer Vamana pins mL=0 (every node level 0); HNSW uses 1/ln(M). Keep the
	// construction-time mL for Vamana so the restored graph stays single-layer.
	if cfg.IndexType == IndexVamana {
		h.mL = 0
	} else {
		h.mL = 1.0 / math.Log(float64(cfg.M))
	}
	// Re-resolve the build-time pair-distance kernel: it is derived from
	// cfg.Metric/cfg.Dim and cached, so swapping cfg above invalidates it. This
	// is NOT optional — RestoreCollection constructs the target from a Cosine
	// dim=1 placeholder and relies entirely on this assignment for the real
	// geometry, so without this every post-restore Insert would run its
	// diversity heuristic on cosine distances while comparing them against the
	// restored metric's candidate distances. Silent recall collapse, no error.
	h.initPairFns()

	size, err := readU32(br)
	if err != nil {
		return err
	}
	capacity, err := readU32(br)
	if err != nil {
		return err
	}
	// Close the outgoing arena before dropping the reference. A heap-backed arena
	// used to be reclaimed by simply becoming unreachable, but its big slabs can
	// now be off-heap address-space reservations (reserve.go), which the GC does
	// not see and no finalizer releases — so replacing the arena without this is an
	// unbounded leak of mapped memory (and, for QuantMmap, of the fd) once per
	// restore. Close is idempotent and a no-op for a small heap arena, so it is
	// safe here even though every reachable caller today hands Restore a fresh
	// index: NamedCollection.Restore and MultiVectorIndex.restore both document
	// restoring into PRE-BUILT sub-indexes, which is one caller away.
	if h.arena != nil {
		_ = h.arena.Close()
	}
	h.arena = newArena(int(dim), int(size))
	h.arena.maxVectorsHint = h.cfg.MaxVectors // as newHNSW does: the cap sizes the slab reservation
	if h.quant != nil {
		h.arena.setQuant(h.quant)
	}
	dropped := version == snapshotVersionPQDrop
	// v9 (float-drop) OMITS the vecs block — the floats are gone. v6/v7/v8 carry the
	// full block, read verbatim into the arena.
	if !dropped {
		h.arena.vecs = make([]float32, int(capacity)*int(dim))
		for i := range h.arena.vecs {
			f, err := readF32(br)
			if err != nil {
				return err
			}
			h.arena.vecs[i] = f
		}
	}
	// v7: PQ codebook block (immediately after vecs, before expires). Only a
	// QuantPQ index ever emits v7, so version == 7 ⇒ a PQ block follows. For a
	// trained codec this loads the TRAINED codebooks into the pqQuantizer BEFORE the
	// re-encode below, so the re-encode produces real PQ codes (ADC navigation
	// survives restart). A 0 presence byte (a v7 snapshot of an untrained PQ index)
	// is a no-op. v6 snapshots (non-PQ, or a PQ index snapshotted pre-Task-2) have
	// no block at all (read nothing); a PQ v6 snapshot re-encodes with an untrained
	// codec → exact-float degrade, the pre-Task-2 behaviour, preserved for back-compat.
	// v10 (QuantSQ): an SQ-ranges block sits where the PQ block would be (a QuantSQ
	// index writes NO PQ block — pqIndex is false at save time). Read the ranges
	// into the trainedSQ BEFORE the re-encode below so codes are identical. v<10 and
	// non-SQ v10 (impossible — v10 is stamped only for QuantSQ) skip this.
	if version == snapshotVersionPRQ {
		// v11 (QuantPRQ): a PRQ codebook block (L layers + R once) sits where the PQ
		// block would be. Load the L layers' codebooks (+R) into the prqQuantizer
		// BEFORE the re-encode below so codes are identical (a restored PRQ-HNSW index
		// navigates on the summed-LUT ADC, not the untrained exact-float degrade).
		if err := readPRQCodebooks(br, h.quant, int(dim)); err != nil {
			return err
		}
	} else if version == snapshotVersionSQ {
		if err := readSQRanges(br, h.quant, int(dim)); err != nil {
			return err
		}
	} else if version >= 7 {
		// v8 ⇒ the codebook block is followed by the OPQ rotation R (restored VERBATIM
		// so the codec rotates identically). v9 writes R (when present) in the dropped
		// trailer below, NOT in the codebook block, so the codebook block here is
		// R-less for v9; the trailer loadCodebooks supplies R + sets dropped state.
		if dropped {
			cb, _, cberr := readPQCodebooksRaw(br, int(dim), false)
			if cberr != nil {
				return cberr
			}
			if cb == nil {
				return fmt.Errorf("%w: float-drop snapshot missing PQ codebooks", ErrSnapshotFormat)
			}
			// Dropped trailer: [hasR byte][R floats?][codeLen u32][len u32][codes].
			rb, rerr := readByte(br)
			if rerr != nil {
				return rerr
			}
			var rot []float32
			if rb == 1 {
				rot = make([]float32, int(dim)*int(dim))
				for i := range rot {
					f, ferr := readF32(br)
					if ferr != nil {
						return ferr
					}
					rot[i] = f
				}
			}
			pq, ok := h.quant.(*pqQuantizer)
			if !ok {
				return fmt.Errorf("%w: float-drop snapshot but index is not QuantPQ", ErrSnapshotFormat)
			}
			// Load codebooks (+R) so ADC/reconstruct work, THEN restore codes verbatim
			// and mark the arena dropped WITHOUT re-encoding (no floats to encode from).
			pq.loadCodebooks(cb, rot)
			cl, clerr := readU32(br)
			if clerr != nil {
				return clerr
			}
			h.arena.codeLen = int(cl)
			n, nerr := readU32(br)
			if nerr != nil {
				return nerr
			}
			h.arena.codes = make([]byte, int(n))
			if _, rderr := io.ReadFull(br, h.arena.codes); rderr != nil {
				return rderr
			}
			h.arena.releaseVecsBacking()
			h.arena.vecs = nil
			h.arena.vecsDropped = true
			h.arena.nslots = int(capacity)
			// Validate the verbatim-restored codes against the loaded codec before
			// any slot is scored: codeLen must match the codec, the buffer must
			// cover exactly nslots slots, and every sub-code index must be in range.
			// A corrupt code would otherwise OOB-index a codebook (arena.Code /
			// reconstruct / adc) and panic at query time. Convert to a format error.
			if h.arena.codeLen != pq.CodeLen() || h.arena.codeLen <= 0 ||
				len(h.arena.codes) != h.arena.nslots*h.arena.codeLen {
				return fmt.Errorf("%w: float-drop codes length %d inconsistent with codeLen %d × nslots %d",
					ErrSnapshotFormat, len(h.arena.codes), h.arena.codeLen, h.arena.nslots)
			}
			if verr := pq.validateCodes(h.arena.codes, h.arena.codeLen); verr != nil {
				return verr
			}
		} else if err := readPQCodebooks(br, h.quant, version >= snapshotVersion); err != nil {
			return err
		}
	}
	// Re-encode the quantization codes from the restored vectors (codes are
	// derived state, never serialized) — SKIPPED for v9 (float-drop), whose codes
	// were already restored VERBATIM above (there are no floats to re-encode from).
	// For a trained QuantPQ index the codec was loaded just above, so this re-encode
	// is deterministic and yields the same codes as before the snapshot.
	if !dropped && h.arena.quant != nil {
		h.arena.codes = make([]byte, int(capacity)*h.arena.codeLen)
		for slot := 0; slot < int(capacity); slot++ {
			h.arena.quant.Encode(
				h.arena.codes[slot*h.arena.codeLen:(slot+1)*h.arena.codeLen],
				h.arena.Vec(uint32(slot)), //nolint:gosec
			)
		}
	}
	if version >= 2 {
		h.arena.expires = make([]uint64, int(capacity))
		for i := range h.arena.expires {
			e, err := readU64(br)
			if err != nil {
				return err
			}
			h.arena.expires[i] = e
		}
	} else {
		// v1 snapshot: no expires field on disk. Zero-fill so loaded vectors
		// have no expiry (subsystem-1 behavior).
		h.arena.expires = make([]uint64, int(capacity))
	}
	// arena.metadata is allocated to capacity regardless of version so slot
	// indexing is always valid. v3 fills present slots; v1/v2 leave all nil.
	h.arena.metadata = make([]Metadata, int(capacity))
	if version >= 3 {
		metaCount, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < metaCount; i++ {
			slot, err := readU32(br)
			if err != nil {
				return err
			}
			entries, err := readU32(br)
			if err != nil {
				return err
			}
			m := make(Metadata, entries)
			for j := uint32(0); j < entries; j++ {
				key, err := readString(br)
				if err != nil {
					return err
				}
				val, err := readValue(br)
				if err != nil {
					return err
				}
				m[key] = val
			}
			if int(slot) < len(h.arena.metadata) {
				h.arena.metadata[slot] = m
			}
		}
	}
	// arena.sparse allocated to capacity regardless of version. v4 fills present
	// slots; v1/v2/v3 leave all nil. The inverted index is rebuilt below.
	h.arena.sparse = make([]*SparseVector, int(capacity))
	if version >= 4 {
		sparseCount, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < sparseCount; i++ {
			slot, err := readU32(br)
			if err != nil {
				return err
			}
			nnz, err := readU32(br)
			if err != nil {
				return err
			}
			sv := &SparseVector{
				Indices: make([]uint32, nnz),
				Values:  make([]float32, nnz),
			}
			for j := uint32(0); j < nnz; j++ {
				dim, err := readU32(br)
				if err != nil {
					return err
				}
				val, err := readF32(br)
				if err != nil {
					return err
				}
				sv.Indices[j] = dim
				sv.Values[j] = val
			}
			if int(slot) < len(h.arena.sparse) {
				h.arena.sparse[slot] = sv
			}
		}
	}
	// arena.keyExpires (per-key payload TTL) allocated to capacity regardless of
	// version. v5 fills present slots with absolute unix-ms deadlines; v<5 leave
	// all nil (no per-key TTL — backward-compatible). Deadlines are restored
	// verbatim (NOT recomputed) so a pending key TTL survives restart time-stable.
	h.arena.keyExpires = make([]map[string]uint64, int(capacity))
	if version >= 5 {
		keyExpCount, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < keyExpCount; i++ {
			slot, err := readU32(br)
			if err != nil {
				return err
			}
			entries, err := readU32(br)
			if err != nil {
				return err
			}
			ke := make(map[string]uint64, entries)
			for j := uint32(0); j < entries; j++ {
				key, err := readString(br)
				if err != nil {
					return err
				}
				dl, err := readU64(br)
				if err != nil {
					return err
				}
				ke[key] = dl
			}
			if int(slot) < len(h.arena.keyExpires) && len(ke) > 0 {
				h.arena.keyExpires[slot] = ke
			}
		}
	}
	// arena.versions (per-point CAS version) allocated to capacity regardless of
	// version. v6 fills present slots verbatim; v<6 leave all 0 here and default
	// live points to version 1 AFTER the idMap is read below (a sane existing-point
	// version — backward-compatible across the CAS upgrade).
	h.arena.versions = make([]uint64, int(capacity))
	if version >= 6 {
		verCount, err := readU32(br)
		if err != nil {
			return err
		}
		for i := uint32(0); i < verCount; i++ {
			slot, err := readU32(br)
			if err != nil {
				return err
			}
			v, err := readU64(br)
			if err != nil {
				return err
			}
			if int(slot) < len(h.arena.versions) {
				h.arena.versions[slot] = v
			}
		}
	}
	freeCount, err := readU32(br)
	if err != nil {
		return err
	}
	h.arena.free = make([]uint32, 0, freeCount)
	for i := uint32(0); i < freeCount; i++ {
		s, err := readU32(br)
		if err != nil {
			return err
		}
		h.arena.free = append(h.arena.free, s)
	}
	idCount, err := readU32(br)
	if err != nil {
		return err
	}
	h.arena.idMap = make(map[uint64]uint32, idCount)
	// Reverse map (slot -> id), allocated to capacity and filled from idMap so
	// slotID stays O(1) after restore. Freed slots keep a zero id (harmless —
	// they are not reachable from the graph).
	h.arena.ids = make([]uint64, int(capacity))
	for i := uint32(0); i < idCount; i++ {
		id, err := readU64(br)
		if err != nil {
			return err
		}
		slot, err := readU32(br)
		if err != nil {
			return err
		}
		h.arena.idMap[id] = slot
		if int(slot) < len(h.arena.ids) {
			h.arena.ids[slot] = id
		}
	}
	// Backward-compat: a v<6 snapshot has no versions block, so every live point
	// defaults to version 1 (a sane existing-point version). v6+ already filled the
	// versions verbatim above.
	if version < 6 {
		for _, slot := range h.arena.idMap {
			if int(slot) < len(h.arena.versions) {
				h.arena.versions[slot] = 1
			}
		}
	}

	tCount, err := readU32(br)
	if err != nil {
		return err
	}
	h.tombstoned = make(map[uint32]bool, tCount)
	for i := uint32(0); i < tCount; i++ {
		s, err := readU32(br)
		if err != nil {
			return err
		}
		h.tombstoned[s] = true
	}

	ep, err := readU32(br)
	if err != nil {
		return err
	}
	h.entryPoint = ep
	mlvl, err := readI32(br)
	if err != nil {
		return err
	}
	h.maxLevel = int(mlvl)
	nodeCount, err := readU32(br)
	if err != nil {
		return err
	}
	// Pre-size the node table and level-0 slab to arena capacity so each node's
	// edges land in their slab region. m0 derives from the restored config via
	// effectiveM0 (Vamana ⇒ VamanaR, HNSW ⇒ 2*M) so the restored slab stride matches
	// how the graph was built; presizeGraphSlab handles both heap and mmap backing.
	h.m0 = effectiveM0(h.cfg)
	cap0 := h.arena.Capacity()
	h.nodes = make([]*node, cap0)
	if err := h.presizeGraphSlab(cap0); err != nil {
		return err
	}
	for i := uint32(0); i < nodeCount; i++ {
		slot, err := readU32(br)
		if err != nil {
			return err
		}
		level, err := readI32(br)
		if err != nil {
			return err
		}
		levelsCount, err := readU32(br)
		if err != nil {
			return err
		}
		nd := &node{slot: slot, level: int(level), upper: makeUpper(int(level))}
		if err := h.setNode(slot, nd); err != nil {
			return err
		}
		for lvl := uint32(0); lvl < levelsCount; lvl++ {
			nc, err := readU32(br)
			if err != nil {
				return err
			}
			row := make([]uint32, nc)
			for j := uint32(0); j < nc; j++ {
				v, err := readU32(br)
				if err != nil {
					return err
				}
				// Validate each neighbor slot against arena capacity before it is
				// written into the graph slab. An out-of-range slot from a corrupt
				// snapshot would otherwise be stored verbatim and only surface as a
				// deferred out-of-bounds panic at query time.
				if int(v) >= cap0 {
					return fmt.Errorf("%w: neighbor slot %d out of range (capacity %d)", ErrSnapshotFormat, v, cap0)
				}
				row[j] = v
			}
			h.writeNbrs(nd, int(lvl), row)
		}
	}

	// Rebuild the sparse inverted index from the restored arena.sparse. The
	// index is never serialized — always reconstructed. No-op for v1-v3
	// snapshots (no sparse vectors loaded).
	if h.sparseIdx == nil {
		h.sparseIdx = newSparseIndex()
	}
	h.sparseIdx.rebuild(h.arena, h.tombstoned)

	// Rebuild the payload (equality) index from the restored metadata; like the
	// sparse index it is never serialized, always reconstructed.
	if h.payloadIdx == nil {
		h.payloadIdx = newPayloadIndex()
	}
	h.payloadIdx.rebuild(h.arena)

	// Rebuild the BM25 full-text index from each restored slot's $content. Like
	// sparseIdx/payloadIdx it is NEVER serialized — it is re-derived from the
	// persisted reserved metadata. h.bm25/h.az survive readSnapshot (it never
	// resets them; they were allocated at newHNSW from cfg.FullText, which the
	// store-side config round-trip preserves), and readSnapshot overwrote h.cfg
	// without FullText, so the rebuild keys off the live h.bm25/h.az, not h.cfg.
	// nil bm25 (full-text disabled) is a no-op → byte/behavior-identical. This one
	// site covers snapshot Restore, WAL replay (Restore-then-tail), and Raft
	// RestoreAll (NewCollection allocates the lane, then Restore rebuilds it).
	if h.bm25 != nil {
		h.bm25.rebuild(h.arena, h.tombstoned, h.az)
	}
	return nil
}

// Restore loads the index from r, replacing all current state.
func (h *hnsw) Restore(r io.Reader) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	err := h.readSnapshot(r)
	// readSnapshot wrote h.arena.expires/keyExpires directly, bypassing
	// SetExpires/SetKeyExpires's incremental deadline-count maintenance (the TTL
	// sweep's fast-path gate) — recompute once, in bulk.
	h.arena.RecomputeDeadlineCounts()
	// readSnapshot rebuilt idMap/tombstoned from the blob — the live id set is
	// wholesale replaced. Invalidate any cached scroll snapshot.
	h.idSetVersion++
	h.bumpData() // wholesale state replacement also invalidates the order_by snapshot
	return err
}

// unexpectedIfEOF maps a mid-value io.EOF to io.ErrUnexpectedEOF, matching
// io.ReadFull's contract (EOF only when ZERO bytes were read; a short read after
// some bytes is ErrUnexpectedEOF). Used by the ByteReader fast paths so their
// error behaviour on a truncated stream is identical to the old io.ReadFull form.
func unexpectedIfEOF(err error) error {
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return err
}

// readU32 reads 4 big-endian bytes. When r is an io.ByteReader (the snapshot
// stream and persist sidecar wrap their source in a bufio.Reader) it reads
// byte-by-byte — NO per-call slice allocation, the read-side analogue of writeU32
// (the snapshot reads millions of scalars). Bytes and error semantics match the
// old io.ReadFull form (a clean end before the value returns io.EOF; a partial
// value returns io.ErrUnexpectedEOF).
func readU32(r io.Reader) (uint32, error) {
	if br, ok := r.(io.ByteReader); ok {
		b0, err := br.ReadByte()
		if err != nil {
			return 0, err
		}
		b1, err := br.ReadByte()
		if err != nil {
			return 0, unexpectedIfEOF(err)
		}
		b2, err := br.ReadByte()
		if err != nil {
			return 0, unexpectedIfEOF(err)
		}
		b3, err := br.ReadByte()
		if err != nil {
			return 0, unexpectedIfEOF(err)
		}
		return uint32(b0)<<24 | uint32(b1)<<16 | uint32(b2)<<8 | uint32(b3), nil
	}
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(b[:]), nil
}

func readU64(r io.Reader) (uint64, error) {
	if br, ok := r.(io.ByteReader); ok {
		var v uint64
		for i := 0; i < 8; i++ {
			b, err := br.ReadByte()
			if err != nil {
				if i > 0 {
					err = unexpectedIfEOF(err)
				}
				return 0, err
			}
			v = v<<8 | uint64(b)
		}
		return v, nil
	}
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint64(b[:]), nil
}

func readI32(r io.Reader) (int32, error) {
	v, err := readU32(r)
	return int32(v), err
}

func readI64(r io.Reader) (int64, error) {
	v, err := readU64(r)
	return int64(v), err
}

func readF32(r io.Reader) (float32, error) {
	v, err := readU32(r)
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}
