// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// CollectionStore snapshot: serialize every collection (single- and multi-
// vector) into one stream so a replicated deployment can capture vector state in
// a Raft FSM snapshot (durable across log truncation / new-node catch-up).
//
// Collections are serialized heap-backed (no mmap paths persisted): in a
// replicated cluster the Raft log + this snapshot are the durability layer, not
// per-node mmap sidecars. Each entry carries a relocatable config marker plus the
// collection's own snapshot blob.

var storeSnapshotMagic = []byte{'R', 'V', 'S', 'A'}

const storeSnapshotVersion = 1

// snapColCfg is the relocatable config persisted per collection (heap form).
type snapColCfg struct {
	Dim                  int       `json:"dim"`
	Metric               Metric    `json:"metric"`
	M                    int       `json:"m"`
	EfConstruction       int       `json:"ef_construction"`
	EfSearch             int       `json:"ef_search"`
	MaxEfSearch          int       `json:"max_ef_search"`
	Seed                 int64     `json:"seed"`
	Quant                QuantMode `json:"quant"`
	RescoreFactor        int       `json:"rescore_factor"`
	FilterFirstThreshold int       `json:"filter_first_threshold"`

	// SQBits / PRQLayers / QuantPQM / OPQ persist the SQ/PRQ quantizer geometry so
	// the cluster restore path (RestoreAll→NewCollection→sc.toConfig) rebuilds the
	// quantizer with the SAME bit-depth / layer count / sub-quantizer count /
	// rotation it was created with. Without them an SQ4/SQ6 collection rebuilds as
	// 8-bit (SQBits 0⇒8) and mis-decodes its bit-packed codes, and a PRQ collection
	// with non-default QuantPQM/PRQLayers gets a wrong arena.codeLen. omitempty keeps
	// a non-SQ/PRQ store snapshot's JSON byte-identical to the pre-feature format
	// (all zero/false, so every key is omitted).
	SQBits    int  `json:"sq_bits,omitempty"`
	PRQLayers int  `json:"prq_layers,omitempty"`
	QuantPQM  int  `json:"quant_pq_m,omitempty"`
	OPQ       bool `json:"opq,omitempty"`
	// PQNBits persists the 4-bit-vs-8-bit PQ code width so the cluster restore path
	// (RestoreAll→NewCollection→sc.toConfig) rebuilds the codec at the SAME width it
	// was created with. CodeLen depends on it (ceil(m/2) for 4-bit vs m for 8-bit), so
	// a wrong width mis-sizes the arena codes side-array and mis-unpacks the nibble
	// codes — the exact geometry-on-restore class as SQBits/PRQLayers/VamanaR. The
	// collection's own snapshot blob carries the trained 4-bit codebooks + packed codes
	// verbatim; this flag ensures a post-restore INSERT re-encodes at 4-bit too.
	// omitempty keeps a non-4-bit snapshot's JSON byte-identical (0 ⇒ key omitted).
	PQNBits int `json:"pq_nbits,omitempty"`
	// FilterFirstRelativeBP rides the snapshot so the opt-in relative selectivity gate
	// survives a snapshot/restore round-trip. omitempty keeps a default (0/off)
	// snapshot's JSON byte-identical to the pre-feature format.
	FilterFirstRelativeBP int `json:"filter_first_relative_bp,omitempty"`

	// IndexType + IVF params persist the backing-index kind so the restore path
	// reconstructs an IVF collection AS IVF (NewCollection→newIndex dispatches on
	// IndexType). omitempty keeps an HNSW snapshot's JSON byte-identical to the
	// pre-IVF format (IndexHNSW=0, IVFNlist/IVFNprobe=0 are all omitted).
	IndexType IndexType `json:"index_type,omitempty"`
	IVFNlist  int       `json:"ivf_nlist,omitempty"`
	IVFNprobe int       `json:"ivf_nprobe,omitempty"`

	// IVFPQ / IVFPQM / IVFRerank persist the IVF-PQ residual-quantization knobs so the
	// cluster restore path (RestoreAll→NewCollection→sc.toConfig) rebuilds the
	// collection AS the SAME IVF-PQ variant. The snapshot blob's PQ trailer carries the
	// trained codebooks + the float-dropped state verbatim, so a snapshot restores its
	// EXISTING codes either way; but without these flags the restored cfg is IVF-Flat-
	// shaped — IVFRerank=false silently DROPS the exact-rescore stage (so the restored
	// ADC ranking diverges from the source's rescored ranking), and IVFPQ=false would
	// mis-shape a post-restore insert / re-validation. This is the geometry-on-restore
	// class — same lesson as SQBits / PRQLayers / SOAR / PQNBits. omitempty keeps a
	// non-IVF-PQ snapshot's JSON byte-identical (false/0 ⇒ all three keys omitted).
	IVFPQ     bool `json:"ivf_pq,omitempty"`
	IVFPQM    int  `json:"ivf_pq_m,omitempty"`
	IVFRerank bool `json:"ivf_rerank,omitempty"`

	// SOAR / SOARLambda persist the IVF multi-assignment knobs so the cluster restore
	// path (RestoreAll→NewCollection→sc.toConfig) rebuilds the collection AS a SOAR
	// index. The collection's own snapshot blob carries the EXISTING cellOf2/code2
	// verbatim (so the multi-assignment list membership survives), but without these
	// flags an INSERT after RestoreAll would single-assign and the index would silently
	// drift toward non-SOAR (the geometry-on-restore class — same lesson as SQBits /
	// PRQLayers / VamanaR). omitempty keeps a non-SOAR snapshot's JSON byte-identical to
	// the pre-feature format (false/0 ⇒ both keys omitted).
	SOAR       bool    `json:"soar,omitempty"`
	SOARLambda float32 `json:"soar_lambda,omitempty"`

	// VamanaR / VamanaL / VamanaAlpha persist the Vamana graph geometry so the
	// cluster restore path (RestoreAll→NewCollection→sc.toConfig→newVamana) rebuilds
	// the index with the SAME out-degree / beam / prune-α it was created with.
	// VamanaR is the level-0 adjacency slab stride (m0); a wrong R re-presizes every
	// neighbor list at the wrong stride → silent graph corruption (the exact class of
	// bug that the SQ/PRQ SQBits/PRQLayers fields close). omitempty keeps a non-Vamana
	// snapshot's JSON byte-identical to the pre-feature format (all zero ⇒ omitted).
	VamanaR     int     `json:"vamana_r,omitempty"`
	VamanaL     int     `json:"vamana_l,omitempty"`
	VamanaAlpha float32 `json:"vamana_alpha,omitempty"`

	// FullText persists the BM25 full-text config (analyzer name + k1/b) so the
	// cluster restore path reconstructs the text lane: RestoreAll→NewCollection
	// allocates the bm25Index + resolves the analyzer from this config, then
	// Collection.Restore re-derives the postings from each slot's persisted
	// $content. omitempty keeps a non-full-text snapshot's JSON byte-identical to
	// the pre-feature format (a nil pointer is omitted entirely).
	FullText *FullTextConfig `json:"full_text,omitempty"`
}

func toSnapColCfg(c Config) snapColCfg {
	return snapColCfg{
		Dim: c.Dim, Metric: c.Metric, M: c.M, EfConstruction: c.EfConstruction,
		EfSearch: c.EfSearch, MaxEfSearch: c.MaxEfSearch, Seed: c.Seed,
		Quant: c.Quant, RescoreFactor: c.RescoreFactor, FilterFirstThreshold: c.FilterFirstThreshold,
		FilterFirstRelativeBP: c.FilterFirstRelativeBP,
		IndexType:             c.IndexType, IVFNlist: c.IVFNlist, IVFNprobe: c.IVFNprobe,
		VamanaR: c.VamanaR, VamanaL: c.VamanaL, VamanaAlpha: c.VamanaAlpha,
		FullText: c.FullText,
		SQBits:   c.SQBits, PRQLayers: c.PRQLayers, QuantPQM: c.QuantPQM, OPQ: c.OPQ,
		PQNBits: c.PQNBits,
		IVFPQ:   c.IVFPQ, IVFPQM: c.IVFPQM, IVFRerank: c.IVFRerank,
		SOAR: c.SOAR, SOARLambda: c.SOARLambda,
	}
}

func (sc snapColCfg) toConfig() Config {
	return Config{
		Dim: sc.Dim, Metric: sc.Metric, M: sc.M, EfConstruction: sc.EfConstruction,
		EfSearch: sc.EfSearch, MaxEfSearch: sc.MaxEfSearch, Seed: sc.Seed,
		Quant: sc.Quant, RescoreFactor: sc.RescoreFactor, FilterFirstThreshold: sc.FilterFirstThreshold,
		FilterFirstRelativeBP: sc.FilterFirstRelativeBP,
		IndexType:             sc.IndexType, IVFNlist: sc.IVFNlist, IVFNprobe: sc.IVFNprobe,
		VamanaR: sc.VamanaR, VamanaL: sc.VamanaL, VamanaAlpha: sc.VamanaAlpha,
		FullText: sc.FullText,
		SQBits:   sc.SQBits, PRQLayers: sc.PRQLayers, QuantPQM: sc.QuantPQM, OPQ: sc.OPQ,
		PQNBits: sc.PQNBits,
		IVFPQ:   sc.IVFPQ, IVFPQM: sc.IVFPQM, IVFRerank: sc.IVFRerank,
		SOAR: sc.SOAR, SOARLambda: sc.SOARLambda,
	}
}

// snapMVCfg is the relocatable multi-vector config (heap form).
type snapMVCfg struct {
	Dim            int       `json:"dim"`
	M              int       `json:"m"`
	EfConstruction int       `json:"ef_construction"`
	EfSearch       int       `json:"ef_search"`
	Seed           int64     `json:"seed"`
	Quant          QuantMode `json:"quant"`
	RescoreFactor  int       `json:"rescore_factor"`
	// FilterFirstRelativeBP rides the MV snapshot (omitempty keeps a default-off
	// snapshot byte-identical to the pre-feature format).
	FilterFirstRelativeBP int `json:"filter_first_relative_bp,omitempty"`
}

// SnapshotAll writes every collection (single- and multi-vector) to w. Safe to
// call concurrently with reads; not with structural changes (create/drop).
func (s *CollectionStore) SnapshotAll(w io.Writer) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	bw := bufio.NewWriter(w)
	if _, err := bw.Write(storeSnapshotMagic); err != nil {
		return err
	}
	if err := bw.WriteByte(storeSnapshotVersion); err != nil {
		return err
	}

	if err := writeU32(bw, uint32(len(s.collections))); err != nil { //nolint:gosec
		return err
	}
	for name, c := range s.collections {
		cfg, _ := json.Marshal(toSnapColCfg(c.cfg))
		var blob bytes.Buffer
		if err := c.Snapshot(&blob); err != nil {
			return fmt.Errorf("vector: snapshot %q: %w", name, err)
		}
		if err := writeFramed(bw, []byte(name), cfg, blob.Bytes()); err != nil {
			return err
		}
	}

	if err := writeU32(bw, uint32(len(s.multi))); err != nil { //nolint:gosec
		return err
	}
	for name, mv := range s.multi {
		cfg, _ := json.Marshal(snapMVCfg{
			Dim: mv.cfg.Dim, M: mv.cfg.M, EfConstruction: mv.cfg.EfConstruction,
			EfSearch: mv.cfg.EfSearch, Seed: mv.cfg.Seed, Quant: mv.cfg.Quant, RescoreFactor: mv.cfg.RescoreFactor,
			FilterFirstRelativeBP: mv.cfg.FilterFirstRelativeBP,
		})
		var blob bytes.Buffer
		if err := mv.snapshot(&blob); err != nil {
			return fmt.Errorf("vector: snapshot multi %q: %w", name, err)
		}
		if err := writeFramed(bw, []byte(name), cfg, blob.Bytes()); err != nil {
			return err
		}
	}

	// Third section (additive): named-vector collections. Appended LAST so an old
	// reader (which stops after the MV section) still restores dense+MV cleanly —
	// it just drops the named collections (opt-in; upgrade-all-first). The store
	// snapshot version is NOT bumped: a snapshot with zero named collections is
	// byte-identical to the old format up to here, and the new reader tolerates the
	// section's absence (EOF) for backward-compat — see RestoreAll.
	if err := writeU32(bw, uint32(len(s.named))); err != nil { //nolint:gosec
		return err
	}
	for name, nc := range s.named {
		cfg, _ := json.Marshal(nc.Config())
		var blob bytes.Buffer
		if err := nc.Snapshot(&blob); err != nil {
			return fmt.Errorf("vector: snapshot named %q: %w", name, err)
		}
		if err := writeFramed(bw, []byte(name), cfg, blob.Bytes()); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// RestoreAll replaces the store's collections with those read from r. Existing
// collections are closed first.
func (s *CollectionStore) RestoreAll(r io.Reader) error {
	br := bufio.NewReader(r)
	magic := make([]byte, len(storeSnapshotMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return err
	}
	if string(magic) != string(storeSnapshotMagic) {
		return fmt.Errorf("vector: bad store snapshot magic")
	}
	ver, err := br.ReadByte()
	if err != nil {
		return err
	}
	if ver != storeSnapshotVersion {
		return fmt.Errorf("vector: store snapshot version %d unsupported", ver)
	}

	// Cluster-persistent restores build a FRESH generation of mmap files, disjoint
	// from the generation in-flight readers still map, so the build→swap→retire
	// sequence never truncates a file under a reader (a fatal SIGBUS).
	gen := uint64(0)
	if s.persistentCluster {
		gen = s.gen.Add(1)
	}

	single := make(map[string]*Collection)
	nSingle, err := readU32(br)
	if err != nil {
		return err
	}
	for i := uint32(0); i < nSingle; i++ {
		name, cfgJSON, blob, ferr := readFramed(br)
		if ferr != nil {
			return ferr
		}
		var sc snapColCfg
		if err := json.Unmarshal(cfgJSON, &sc); err != nil {
			return err
		}
		colCfg := sc.toConfig()
		if s.persistentCluster {
			colCfg = s.effectiveClusterConfig(name, gen, colCfg)
			// A catching-up node has no tenant dir yet; the mmap files open at
			// construction, so create it first (GraphMmapPath is always set).
			if err := os.MkdirAll(filepath.Dir(colCfg.GraphMmapPath), 0o750); err != nil {
				return fmt.Errorf("vector: restore mkdir %q: %w", name, err)
			}
		}
		c, cerr := NewCollection(name, colCfg)
		if cerr != nil {
			return fmt.Errorf("vector: restore %q: %w", name, cerr)
		}
		if rerr := c.Restore(bytes.NewReader(blob)); rerr != nil {
			_ = c.Close()
			return fmt.Errorf("vector: restore %q: %w", name, rerr)
		}
		single[name] = c
	}

	multi := make(map[string]*MultiVectorIndex)
	nMulti, err := readU32(br)
	if err != nil {
		return err
	}
	for i := uint32(0); i < nMulti; i++ {
		name, cfgJSON, blob, ferr := readFramed(br)
		if ferr != nil {
			return ferr
		}
		var mc snapMVCfg
		if err := json.Unmarshal(cfgJSON, &mc); err != nil {
			return err
		}
		mvCfg := MultiVectorConfig{
			Dim: mc.Dim, M: mc.M, EfConstruction: mc.EfConstruction, EfSearch: mc.EfSearch,
			Seed: mc.Seed, Quant: mc.Quant, RescoreFactor: mc.RescoreFactor,
			FilterFirstRelativeBP: mc.FilterFirstRelativeBP,
		}
		if s.persistentCluster {
			mvCfg = s.effectiveClusterMVConfig(name, gen, mvCfg)
			// mmap files open at construction; ensure the tenant dir exists on a
			// catching-up node (Persistent only when quantized — GraphMmapPath is
			// always set in cluster mode).
			if err := os.MkdirAll(filepath.Dir(mvCfg.GraphMmapPath), 0o750); err != nil {
				return fmt.Errorf("vector: restore multi mkdir %q: %w", name, err)
			}
		}
		mv, merr := NewMultiVectorIndex(mvCfg)
		if merr != nil {
			return fmt.Errorf("vector: restore multi %q: %w", name, merr)
		}
		if rerr := mv.restore(bytes.NewReader(blob)); rerr != nil {
			_ = mv.Close()
			return fmt.Errorf("vector: restore multi %q: %w", name, rerr)
		}
		multi[name] = mv
	}

	// Third section (additive): named-vector collections. BACKWARD-COMPAT: an old
	// snapshot (written before this section existed) ends right after the MV
	// section, so the nNamed read hits a CLEAN io.EOF (ReadFull at a section
	// boundary reads 0 bytes) — treat that as zero named collections and restore
	// cleanly. Only io.EOF is the old-format signal: io.ErrUnexpectedEOF (1-3
	// stray bytes) means a physically truncated/corrupt stream, NOT the old
	// format, and must surface as an error. A new snapshot ALWAYS writes this
	// section (>=0), so a genuine truncation surfaces as a non-EOF error path.
	named := make(map[string]*NamedCollection)
	nNamed, nerr := readU32(br)
	if nerr != nil && !errors.Is(nerr, io.EOF) {
		return nerr
	}
	if nerr == nil {
		for i := uint32(0); i < nNamed; i++ {
			name, cfgJSON, blob, ferr := readFramed(br)
			if ferr != nil {
				return ferr
			}
			var ncfg map[string]NamedVectorParams
			if err := json.Unmarshal(cfgJSON, &ncfg); err != nil {
				return err
			}
			nc, ncerr := NewNamedCollection(name, ncfg)
			if ncerr != nil {
				return fmt.Errorf("vector: restore named %q: %w", name, ncerr)
			}
			if rerr := nc.Restore(bytes.NewReader(blob)); rerr != nil {
				_ = nc.Close()
				return fmt.Errorf("vector: restore named %q: %w", name, rerr)
			}
			named[name] = nc
		}
	}

	// Write side of catalogMu: the swap waits for any create-or-replace restore
	// that is mid-publication (it has dropped a collection and not yet registered
	// its replacement), so it lands wholly before or wholly after that publication.
	// No name lock is taken here, before or while holding catalogMu.
	s.catalogMu.Lock()
	s.mu.Lock()
	old, oldMV, oldNamed := s.collections, s.multi, s.named
	s.collections, s.multi, s.named = single, multi, named
	s.mu.Unlock()
	s.catalogMu.Unlock()
	// Retire old collections: drain in-flight readers before unmapping, then
	// delete the previous generation's mmap files. Safe (and immediate) for heap
	// collections too — their cleanup is a no-op and there's nothing to unmap.
	for _, c := range old {
		col := c
		col.retire(func() { removeClusterMmapFiles(col.cfg) })
	}
	for _, mv := range oldMV {
		idx := mv
		idx.retire(func() { removeClusterMVFiles(idx.cfg) })
	}
	// Named collections are heap-only in v1 (no mmap sidecar — see the named-
	// vectors plan non-goals), so retirement just drains readers + closes; there
	// are no per-generation files to remove.
	for _, nc := range oldNamed {
		nc.retire(nil)
	}
	return nil
}

// writeFramed writes [nameLen u16][name][cfgLen u32][cfg][blobLen u32][blob].
func writeFramed(w io.Writer, name, cfg, blob []byte) error {
	var hdr [2]byte
	binary.BigEndian.PutUint16(hdr[:], uint16(len(name))) //nolint:gosec
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := w.Write(name); err != nil {
		return err
	}
	if err := writeBytes32(w, cfg); err != nil {
		return err
	}
	return writeBytes32(w, blob)
}

func readFramed(r *bufio.Reader) (name string, cfg, blob []byte, err error) {
	var hdr [2]byte
	if _, err = io.ReadFull(r, hdr[:]); err != nil {
		return "", nil, nil, err
	}
	nb := make([]byte, binary.BigEndian.Uint16(hdr[:]))
	if _, err = io.ReadFull(r, nb); err != nil {
		return "", nil, nil, err
	}
	if cfg, err = readBytes32(r); err != nil {
		return "", nil, nil, err
	}
	if blob, err = readBytes32(r); err != nil {
		return "", nil, nil, err
	}
	return string(nb), cfg, blob, nil
}

func writeBytes32(w io.Writer, b []byte) error {
	if err := writeU32(w, uint32(len(b))); err != nil { //nolint:gosec
		return err
	}
	_, err := w.Write(b)
	return err
}

func readBytes32(r io.Reader) ([]byte, error) {
	n, err := readU32(r)
	if err != nil {
		return nil, err
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return nil, err
	}
	return b, nil
}
