// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// ErrCollectionExists is returned by CreateCollection when name is taken.
var ErrCollectionExists = errors.New("vector: collection already exists")

// ErrNoMultiVector is the sentinel wrapped by multi-vector operations that
// target an absent collection. Callers use errors.Is to distinguish "absent"
// from real failures (e.g. to make drop idempotent, mirroring dense
// DropCollection's no-op-on-missing).
var ErrNoMultiVector = errors.New("vector: no multi-vector collection")

// CollectionStore is the persistent registry of named Collections.
//
// Disk layout under DataDir/vectors/:
//
//	<name>.json   — config (encoding/json)
//	<name>.snap   — index snapshot (binary)
//
// Atomic writes: tmp file + rename so a crash mid-write leaves the previous
// good snapshot intact.
type CollectionStore struct {
	dir string

	// ephemeralDir is set when the caller configured no directory. The store
	// then owns a private temp dir and removes it on Close.
	//
	// Without this, an empty dir made every path below RELATIVE: the store wrote
	// ./vectors/<tenant>/ into whatever working directory the process happened to
	// have, and read it back on the next run — so a "fresh" store returned
	// "collection already exists" for a name the caller had never used, and two
	// independent stores in one process shared a namespace. Both contradict the
	// documented contract that an empty DataDir means heap mode.
	ephemeralDir bool

	// persistentCluster makes every collection mmap-backed (off-heap) per node,
	// for a replicated deployment where Raft (log + FSM snapshot) is the
	// durability authority and the mmap is a non-authoritative memory layout.
	// Cluster collections use generation-suffixed mmap files (gen) so a Restore
	// can build a fresh generation without colliding with the one in-flight
	// readers still map; on-disk files from a prior process are wiped at open
	// (Raft repopulates). false = single-node behavior (heap or per-collection
	// Persistent instant-restart), unchanged.
	persistentCluster bool
	gen               atomic.Uint64

	mu          sync.RWMutex
	collections map[string]*Collection
	// multi holds late-interaction (multi-vector) indexes by canonical name.
	// In-memory only — these are not persisted across restart.
	multi map[string]*MultiVectorIndex
	// named holds Qdrant-style named-vector collections by canonical name. A NEW
	// family parallel to collections/multi; a name belongs to exactly one family.
	// In-memory only (snapshot/restore is separate).
	named map[string]*NamedCollection

	// cold holds the lightweight stubs of collections evicted to an object store
	// (single-node cold tiering — see coldtier.go). A canonical name is in EITHER
	// collections (hot) XOR cold (evicted). nil/empty unless cold tiering is used,
	// so the hot path's only cost is a len()==0 check in Acquire.
	cold map[string]*coldEntry
	// nowFn is the INJECTED clock used to stamp lastAccess on resolve (Acquire) and
	// to seed it on promote. nil (the default) means the engine reads no clock at
	// all — determinism — and lastAccess is driven only by the explicit timestamps
	// passed to EvictCollection/SweepCold. The cmd-layer driver and tests inject a
	// real/fake clock via SetClock. Guarded by mu.
	nowFn func() time.Time

	// nowFnMs is a TEST/advanced override of the per-index TTL wall clock (unix
	// millis), propagated to every collection/multi/named index via SetNowFunc and
	// applied to any built later. nil (production) leaves each index on time.Now, so
	// the default path is byte-identical. It skews ONLY the non-apply expiry sites
	// (sweeper + read filter + the wall-clock branch of writes); the stamped apply
	// path (InsertAt et al.) is unaffected. Guarded by mu. See SetNowFunc.
	nowFnMs func() int64
}

// OpenCollectionStore loads any pre-existing collection configs and
// snapshots from dir/vectors/, returning a ready-to-use store.
//
// Disk layout:
//
//	<dir>/vectors/<tenant>/<collection>.json
//	<dir>/vectors/<tenant>/<collection>.snap
//
// Legacy top-level files from subsystem 1 (no tenant subdir) are picked up
// and remapped to the "default" tenant on load.
func OpenCollectionStore(dir string) (*CollectionStore, error) {
	return OpenCollectionStorePersistent(dir, false)
}

// OpenCollectionStorePersistent is OpenCollectionStore with the per-node
// persistent-cluster policy. When persistentCluster is true, every collection is
// mmap-backed (vectors off-heap) and the durability authority is Raft, not the
// per-collection sidecar: stale on-disk vector data files are wiped at open and
// repopulated from the Raft snapshot/log. Used by shard.New in a replicated
// deployment (shard.Config.PersistentVectors).
func OpenCollectionStorePersistent(dir string, persistentCluster bool) (store *CollectionStore, retErr error) {
	// No directory configured means heap mode. Give the store a private temp dir
	// rather than letting every path resolve relative to the process's working
	// directory: the on-disk layout below is an implementation detail of the
	// running store, not something a caller who configured nothing asked to have
	// written next to their binary — and reading it back on the next run made a
	// fresh store inherit a previous one's collections.
	ephemeral := false
	if dir == "" {
		td, err := os.MkdirTemp("", "rostam-heap-")
		if err != nil {
			return nil, fmt.Errorf("vector: create heap-mode scratch dir: %w", err)
		}
		dir, ephemeral = td, true
		// Every failure path below this point returns without a store, so nothing
		// will ever call Close to reclaim the directory just created. Deferring on
		// the named error covers them all — including the ones added later by
		// someone who never reads this comment.
		defer func() {
			if retErr != nil {
				_ = os.RemoveAll(td)
			}
		}()
	}
	vd := filepath.Join(dir, "vectors")
	if err := os.MkdirAll(vd, 0o750); err != nil {
		return nil, fmt.Errorf("vector: mkdir %s: %w", vd, err)
	}
	s := &CollectionStore{dir: dir, ephemeralDir: ephemeral, persistentCluster: persistentCluster, collections: make(map[string]*Collection), multi: make(map[string]*MultiVectorIndex), named: make(map[string]*NamedCollection)}

	// Cluster-persistent: the mmap files are a non-authoritative cache. Wipe any
	// left by a prior process so Raft (FSM.Restore + log replay) is the single
	// source of truth — never an instant-restart from a possibly-torn prior life.
	if persistentCluster {
		if err := wipeVectorDataFiles(vd); err != nil {
			return nil, fmt.Errorf("vector: wipe stale data: %w", err)
		}
	}

	// Pass 1: legacy flat layout (subsystem-1) — top-level *.json files.
	top, err := os.ReadDir(vd)
	if err != nil {
		return nil, fmt.Errorf("vector: read %s: %w", vd, err)
	}
	for _, e := range top {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		base := e.Name()[:len(e.Name())-len(".json")]
		canonical := DefaultTenant + "/" + base
		if err := s.loadCollection(canonical, filepath.Join(vd, e.Name()), filepath.Join(vd, base+".snap")); err != nil {
			return nil, err
		}
	}

	// Pass 2: tenant-prefixed layout — vectors/<tenant>/<collection>.json.
	for _, e := range top {
		if !e.IsDir() {
			continue
		}
		tenant := e.Name()
		tenantDir := filepath.Join(vd, tenant)
		colEntries, err := os.ReadDir(tenantDir)
		if err != nil {
			return nil, fmt.Errorf("vector: read %s: %w", tenantDir, err)
		}
		for _, ce := range colEntries {
			if ce.IsDir() {
				continue
			}
			switch filepath.Ext(ce.Name()) {
			case ".json":
				base := ce.Name()[:len(ce.Name())-len(".json")]
				if err := s.loadCollection(tenant+"/"+base, filepath.Join(tenantDir, ce.Name()), filepath.Join(tenantDir, base+".snap")); err != nil {
					return nil, err
				}
			case ".mvcfg":
				base := ce.Name()[:len(ce.Name())-len(".mvcfg")]
				if err := s.loadMultiVector(tenant+"/"+base, filepath.Join(tenantDir, ce.Name())); err != nil {
					return nil, err
				}
			case ".ncfg":
				base := ce.Name()[:len(ce.Name())-len(".ncfg")]
				if err := s.loadNamed(tenant+"/"+base, filepath.Join(tenantDir, ce.Name())); err != nil {
					return nil, err
				}
			}
		}
	}
	return s, nil
}

// loadCollection reads cfg + snapshot for one collection and registers it
// under its canonical name. Used by OpenCollectionStore's two passes.
func (s *CollectionStore) loadCollection(canonical, cfgPath, snapPath string) error {
	cfg, err := readConfig(cfgPath)
	if err != nil {
		return fmt.Errorf("vector: load %s: %w", canonical, err)
	}

	// Cluster-persistent: data files were wiped at open; build an empty
	// mmap-backed collection (current generation) and let Raft (FSM.Restore / log
	// replay) repopulate it. No sidecar instant-restart, no WAL — Raft is the
	// durability authority.
	if s.persistentCluster {
		eff := s.effectiveClusterConfig(canonical, s.gen.Load(), cfg)
		c, nerr := NewCollection(canonical, eff)
		if nerr != nil {
			return fmt.Errorf("vector: cluster open %s: %w", canonical, nerr)
		}
		s.collections[canonical] = c
		return nil
	}

	// WAL collections recover from a CONSISTENT checkpoint (Snapshot) plus the
	// WAL tail — not the mmap sidecar. Instant-restart's zero-copy mmap base is
	// only valid when nothing mutates after Flush; with a WAL, post-checkpoint
	// inserts mutate the mmap in place (incl. pruning earlier nodes' edges), so
	// the live mmap diverges from any sidecar. Restore rebuilds a consistent
	// graph (into a fresh mmap when Persistent), then replay applies the tail.
	if cfg.WAL {
		eff := s.effectiveConfig(canonical, cfg)
		c, nerr := NewCollection(canonical, eff)
		if nerr != nil {
			return fmt.Errorf("vector: rebuild %s: %w", canonical, nerr)
		}
		if f, oerr := os.Open(snapPath); oerr == nil {
			rerr := c.Restore(f)
			_ = f.Close()
			if rerr != nil {
				_ = c.Close()
				return fmt.Errorf("vector: restore %s: %w", canonical, rerr)
			}
		}
		_, _, _, walPath := s.persistPaths(canonical)
		if rerr := c.replayWAL(walPath); rerr != nil {
			_ = c.Close()
			return fmt.Errorf("vector: wal replay %s: %w", canonical, rerr)
		}
		w, werr := openWAL(walPath, cfg.WALNoSync)
		if werr != nil {
			_ = c.Close()
			return fmt.Errorf("vector: wal open %s: %w", canonical, werr)
		}
		c.wal = w
		s.collections[canonical] = c
		return nil
	}

	// Persistent collections instant-restart: map their vector + graph files and
	// the sidecar, no graph rebuild. A collection created but never flushed has
	// no sidecar yet — open it as a fresh empty (mmap-backed) index.
	if cfg.Persistent {
		eff := s.effectiveConfig(canonical, cfg)
		_, _, metaPath, _ := s.persistPaths(canonical)
		if _, statErr := os.Stat(metaPath); statErr == nil {
			c, oerr := openPersistentCollection(canonical, eff, metaPath)
			if oerr != nil {
				return fmt.Errorf("vector: instant-open %s: %w", canonical, oerr)
			}
			s.collections[canonical] = c
			return nil
		}
		c, nerr := NewCollection(canonical, eff)
		if nerr != nil {
			return fmt.Errorf("vector: rebuild %s: %w", canonical, nerr)
		}
		s.collections[canonical] = c
		return nil
	}

	c, err := NewCollection(canonical, cfg)
	if err != nil {
		return fmt.Errorf("vector: rebuild %s: %w", canonical, err)
	}
	if f, err := os.Open(snapPath); err == nil {
		rerr := c.Restore(f)
		_ = f.Close()
		if rerr != nil {
			return fmt.Errorf("vector: restore %s: %w", canonical, rerr)
		}
	}
	s.collections[canonical] = c
	return nil
}

// collectionPath returns the on-disk paths (cfg, snap) for a canonical name.
// Caller must already canonicalize.
func (s *CollectionStore) collectionPath(canonical string) (cfgPath, snapPath string) {
	tenant, col, _ := splitTenant(canonical) // canonical is always valid
	base := filepath.Join(s.dir, "vectors", tenant)
	return filepath.Join(base, col+".json"), filepath.Join(base, col+".snap")
}

// persistPaths returns the mmap + sidecar + WAL paths the store manages for a
// Persistent collection (vectors, level-0 graph, instant-restart sidecar, and
// the write-ahead log).
func (s *CollectionStore) persistPaths(canonical string) (vecsPath, graphPath, metaPath, walPath string) {
	tenant, col, _ := splitTenant(canonical)
	base := filepath.Join(s.dir, "vectors", tenant)
	return filepath.Join(base, col+".vecs"), filepath.Join(base, col+".graph"),
		filepath.Join(base, col+".meta"), filepath.Join(base, col+".wal")
}

// effectiveConfig augments a user Config for a Persistent collection with the
// store-managed mmap backing (QuantStorage + file paths). Non-persistent configs
// pass through unchanged. The user-facing Config (Persistent flag, quantizer) is
// what gets serialized to <col>.json; paths are re-derived here on every open so
// the store stays relocatable.
func (s *CollectionStore) effectiveConfig(canonical string, cfg Config) Config {
	if !cfg.Persistent {
		return cfg
	}
	vecs, graph, _, _ := s.persistPaths(canonical)
	// A Persistent IVF instant-restarts via its own mmap sidecar (SavePersist /
	// openPersistIVF): the float vectors live in the .vecs mmap file, externalized
	// into the sidecar; IVF has no level-0 graph slab, so no GraphMmapPath. We set
	// ONLY MmapPath here — NOT QuantStorage=QuantMmap: an IVF-Flat collection is
	// QuantNone, and the QuantStorage==QuantMmap gate (index.go) requires a
	// quantizer (Quant != QuantNone), so forcing it would reject IVF-Flat. newIVF
	// keys off cfg.Persistent && cfg.MmapPath (NOT QuantStorage) to back the float
	// arena with the mmap file, so MmapPath alone is sufficient for every IVF mode.
	if cfg.IndexType == IndexIVF {
		cfg.MmapPath = vecs
		return cfg
	}
	cfg.QuantStorage = QuantMmap
	cfg.MmapPath = vecs
	cfg.GraphMmapPath = graph
	return cfg
}

// clusterMmapPaths returns the generation-suffixed mmap file paths for a
// cluster-persistent collection. The generation lets a Restore build a fresh
// backing without colliding with the files in-flight readers still map.
func (s *CollectionStore) clusterMmapPaths(canonical string, gen uint64) (vecs, graph string) {
	tenant, col, _ := splitTenant(canonical)
	base := filepath.Join(s.dir, "vectors", tenant)
	suffix := fmt.Sprintf(".g%d", gen)
	return filepath.Join(base, col+suffix+".vecs"), filepath.Join(base, col+suffix+".graph")
}

// effectiveClusterConfig backs a collection with generation-suffixed mmap files
// for the persistent-cluster policy. It does NOT set the Persistent flag or a
// WAL: durability is Raft, not the per-collection sidecar, so there is no
// instant-restart sidecar and no WAL — only the off-heap layout. The level-0
// graph slab always mmaps; the vector store mmaps only for a quantized
// collection (QuantMmap holds quantized codes — full-precision float32 vectors
// have no mmap storage mode and stay heap-resident).
func (s *CollectionStore) effectiveClusterConfig(canonical string, gen uint64, cfg Config) Config {
	vecs, graph := s.clusterMmapPaths(canonical, gen)
	cfg.Persistent = false
	cfg.WAL = false
	// Raft is the durability + determinism authority: disable the background
	// wall-clock TTL sweeper so its physical removal never diverges committed state
	// across replicas at skewed clocks. Expired entries are still filtered lazily at
	// read time (client staleness only). See Config.SuppressSweep (#4 B3a analog).
	cfg.SuppressSweep = true
	cfg.GraphMmapPath = graph
	if cfg.Quant != QuantNone {
		cfg.QuantStorage = QuantMmap
		cfg.MmapPath = vecs
	}
	return cfg
}

// removeClusterMmapFiles deletes a cluster-persistent collection's generation
// files (run from retire, after the index is closed/unmapped).
func removeClusterMmapFiles(cfg Config) {
	if cfg.MmapPath != "" {
		_ = os.Remove(cfg.MmapPath)
	}
	if cfg.GraphMmapPath != "" {
		_ = os.Remove(cfg.GraphMmapPath)
	}
}

// wipeVectorDataFiles deletes per-collection data files (mmap vecs/graph,
// instant-restart sidecar, WAL, heap snapshot) under the vectors dir, keeping
// the .json/.mvcfg config registry. Used at cluster-persistent open so Raft is
// the unambiguous source of vector state.
func wipeVectorDataFiles(vd string) error {
	return filepath.WalkDir(vd, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		switch filepath.Ext(d.Name()) {
		case ".vecs", ".graph", ".meta", ".wal", ".snap", ".maps", ".nsnap", ".nwal", ".mvwsnap", ".mvwal":
			if rerr := os.Remove(path); rerr != nil && !os.IsNotExist(rerr) { //nolint:gosec // local data dir under our control, not attacker-supplied paths
				return rerr
			}
		}
		return nil
	})
}

// Acquire returns the named collection with a reference held, so a concurrent
// RestoreAll/DropCollection won't unmap it mid-operation (a fatal SIGBUS for
// mmap-backed cluster collections). The caller MUST pair it with exactly one
// (*Collection).Release, typically via defer. It is the guarded counterpart of
// Get for the read/write hot paths.
func (s *CollectionStore) Acquire(name string) (*Collection, bool) {
	canonical, err := canonicalName(name)
	if err != nil {
		return nil, false
	}
	s.mu.RLock()
	c, ok := s.collections[canonical]
	if ok {
		c.inuse.Add(1)
	}
	// Cold-tier resolve hook: only consult the cold catalog when it is non-empty,
	// so a store with no evicted collections pays just this len()==0 branch (the
	// hot path is byte-identical to before cold tiering). When the name is a cold
	// stub, fall through to the (slow, single-flight) promote path below.
	var cold *coldEntry
	if !ok && len(s.cold) > 0 {
		cold = s.cold[canonical]
	}
	// Stamp last-access for the idle sweeper, but only when a clock was injected —
	// the engine never reads time.Now itself (determinism), and nowFn is set only
	// when cold tiering is enabled, so a store without cold tiering pays nothing
	// here. The stamp is a lock-free atomic store on the collection itself: two
	// concurrent Acquire calls both hold the shared read lock, so a shared map
	// write here would be a fatal concurrent-map-write — the per-collection atomic
	// removes that hazard (and the shared map) from the hot path entirely. The
	// write-locked sweeper reads the same atomic.
	if ok && s.nowFn != nil {
		c.lastAccess.Store(s.nowFn().UnixNano())
	}
	s.mu.RUnlock()
	if ok {
		return c, true
	}
	if cold == nil {
		return nil, false
	}
	// Lazily promote the cold collection (single-flight inside promoteCold), then
	// retry the live lookup. A failed promote leaves the stub intact and Acquire
	// reports the collection as absent (the op layer surfaces "no collection"),
	// which is recoverable on a later access.
	if perr := s.promoteCold(canonical, cold); perr != nil {
		return nil, false
	}
	s.mu.RLock()
	c, ok = s.collections[canonical]
	if ok {
		c.inuse.Add(1)
	}
	s.mu.RUnlock()
	return c, ok
}

// CreateCollection registers a new collection, writes its config to disk,
// and returns ErrCollectionExists if name is taken. Name may be a bare
// collection (e.g., "docs", which maps to "default/docs") or a tenant-
// qualified path (e.g., "acme/docs").
func (s *CollectionStore) CreateCollection(name string, cfg Config) error {
	// Named-vector collections are a separate family: route to CreateNamedConfig
	// (which validates the spaces, guards cross-family name collisions, and
	// registers the NamedCollection). The collection-level WAL flags thread through
	// the carrier so a single-node vector.Config{NamedVectors, WAL} gets the named
	// WAL lifecycle (cluster mode forces WAL off inside CreateNamedConfig — Raft is
	// the authority there). The dense path below is untouched.
	if len(cfg.NamedVectors) > 0 {
		return s.CreateNamedConfig(name, NamedConfig{
			Spaces:    cfg.NamedVectors,
			WAL:       cfg.WAL,
			WALNoSync: cfg.WALNoSync,
		})
	}
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.collections[canonical]; ok {
		return ErrCollectionExists
	}
	if _, ok := s.multi[canonical]; ok {
		return ErrCollectionExists
	}
	if _, ok := s.named[canonical]; ok {
		return ErrCollectionExists
	}
	c, err := s.buildCollection(canonical, cfg)
	if err != nil {
		// The config marker is the load-time source of truth: OpenCollectionStore
		// rebuilds the catalog from it. buildCollection can fail AFTER the marker's
		// rename has landed (the durable publish also fsyncs the parent directory,
		// which can fail on its own), and it fails after writing the marker whenever
		// the WAL cannot be opened. Either way the caller is told creation failed and
		// nothing is registered in memory, so leaving the marker behind would let a
		// restart resurrect the collection as an empty one. Best-effort remove it.
		// Done here rather than inside buildCollection because the cold-tier promote
		// path also builds through it, for a collection whose marker is a live
		// catalog entry that must survive a failed promote.
		cfgPath, _ := s.collectionPath(canonical)
		_ = os.Remove(cfgPath)
		return err
	}
	s.collections[canonical] = c
	return nil
}

// buildCollection constructs a fully-initialized *Collection for canonical+cfg
// (effective-config mmap backing, on-disk config sidecar, WAL open) WITHOUT
// registering it in any catalog map. CreateCollection registers the result under
// the store lock; the cold-tier promote path restores into it off-catalog and
// then publishes it atomically (so a concurrent Acquire never serves a
// half-restored index). Callers that register must hold s.mu.
func (s *CollectionStore) buildCollection(canonical string, cfg Config) (*Collection, error) {
	cfgPath, _ := s.collectionPath(canonical)
	// Create the tenant dir first: a Persistent collection opens its mmap files
	// at construction, so the directory must already exist.
	if err := os.MkdirAll(filepath.Dir(cfgPath), 0o750); err != nil {
		return nil, err
	}
	// Cluster-persistent: back the collection with generation-suffixed mmap files
	// (off-heap) regardless of the user's Persistent flag — the node-level policy
	// decides, since the snapshot wire format can't carry per-collection backing.
	effective := s.effectiveConfig(canonical, cfg)
	if s.persistentCluster {
		effective = s.effectiveClusterConfig(canonical, s.gen.Load(), cfg)
	}
	c, err := NewCollection(canonical, effective)
	if err != nil {
		return nil, err
	}
	// Persist the user-facing config (Persistent flag + quantizer); the mmap
	// paths are re-derived on open, so the store stays relocatable.
	if err := writeConfig(cfgPath, cfg); err != nil {
		_ = c.Close()
		return nil, err
	}
	if cfg.WAL && !s.persistentCluster {
		_, _, _, walPath := s.persistPaths(canonical)
		w, werr := openWAL(walPath, cfg.WALNoSync)
		if werr != nil {
			_ = c.Close()
			return nil, werr
		}
		c.wal = w
	}
	// Inherit any store-level TTL clock override (test/advanced; nil in production).
	if s.nowFnMs != nil {
		c.SetNowFunc(s.nowFnMs)
	}
	return c, nil
}

// SetNowFunc installs a TEST/advanced override of the per-index TTL wall clock
// (unix millis) and propagates it to every currently-resident collection, multi-
// vector index, and named collection; a dense collection built later also inherits
// it (buildCollection re-applies s.nowFnMs), while a named/multi collection created
// afterward should be followed by another SetNowFunc call to pick up the override.
// nil restores the real clock everywhere. Production never calls it, so the default
// path is byte-identical to time.Now. It skews ONLY the non-apply expiry sites
// (background sweeper + client read/query filter + the wall-clock branch of the
// write paths) — the stamped replicated apply path (InsertAt et al.) takes an
// explicit leader stamp and is unaffected. Mirrors cache.Cache.SetNowFunc so the
// shard-level determinism tests can skew replicas' wall clocks.
func (s *CollectionStore) SetNowFunc(fn func() int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nowFnMs = fn
	for _, c := range s.collections {
		c.SetNowFunc(fn)
	}
	for _, m := range s.multi {
		m.SetNowFunc(fn)
	}
	for _, nc := range s.named {
		nc.SetNowFunc(fn)
	}
}

// DropCollection removes a collection and its on-disk files. Drop of an
// unknown name is a no-op (returns nil).
func (s *CollectionStore) DropCollection(name string) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	s.mu.Lock()
	c, ok := s.collections[canonical]
	if !ok {
		// A cold (evicted) stub: drop the catalog entry. The snapshot object in the
		// store is left as-is (it is a backup the caller may still restore explicitly).
		// No retire — a stub holds no index. But the on-disk sidecar files written at
		// create/restore time (the .cfg.json, and the mmap/WAL files for a persistent
		// collection) survive eviction (EvictCollection deliberately keeps them), so
		// removing only the catalog identity here would LEAK them. Mirror the hot-path
		// cleanup below and delete them, so a dropped cold collection leaves nothing on
		// disk — exactly like a dropped hot one.
		if coldCfg, cold := s.cold[canonical]; cold {
			delete(s.cold, canonical)
			s.mu.Unlock()
			cfgPath, snapPath := s.collectionPath(canonical)
			vecsPath, graphPath, metaPath, walPath := s.persistPaths(canonical)
			_ = os.Remove(cfgPath)
			_ = os.Remove(snapPath)
			_ = os.Remove(vecsPath)
			_ = os.Remove(graphPath)
			_ = os.Remove(metaPath)
			_ = os.Remove(walPath)
			removeClusterMmapFiles(coldCfg.cfg)
			return nil
		}
		// Dispatch to the named family: a name can be dense XOR named, so a
		// DropCollection on a named collection drops it (mirrors how the dense
		// path frees its collection). MV has its own DropMultiVector entry point.
		if nc, nok := s.named[canonical]; nok {
			delete(s.named, canonical)
			s.mu.Unlock()
			nc.retire(nil)
			return nil
		}
		s.mu.Unlock()
		return nil
	}
	delete(s.collections, canonical)
	s.mu.Unlock()
	// Out of the lock: drain in-flight users (Acquire holders) before closing, so
	// an mmap-backed collection is never unmapped under a reader, then delete the
	// on-disk files. Removed from the map above, so no new Acquire can find it.
	cfgPath, snapPath := s.collectionPath(canonical)
	vecsPath, graphPath, metaPath, walPath := s.persistPaths(canonical)
	colCfg := c.cfg
	c.retire(func() {
		_ = os.Remove(cfgPath)
		_ = os.Remove(snapPath)
		// Single-node persistent files + cluster generation files (no-ops if absent).
		_ = os.Remove(vecsPath)
		_ = os.Remove(graphPath)
		_ = os.Remove(metaPath)
		_ = os.Remove(walPath)
		removeClusterMmapFiles(colCfg)
	})
	return nil
}

// RestoreCollection (re)creates the named collection from a snapshot reader,
// the inverse of writeSnapshotFile's offload: it is how a collection is
// reconstructed from a snapshot stream OUTSIDE the on-disk startup load path
// (e.g. by the backup package pulling a snapshot back from object storage).
//
// It is create-or-replace: any existing collection of this name is Dropped
// first (its on-disk files removed), then a fresh collection is constructed and
// Collection.Restore is applied on top. The dense HNSW snapshot stream carries
// the core index config (Dim/Metric/M/EfConstruction/EfSearch/Seed) and
// readSnapshot overwrites the target index's cfg with it, so a minimal
// placeholder config suffices here — the restored collection ends up with the
// geometry the snapshot was taken at. (Quantization, IndexType and Vamana
// geometry are NOT in the stream; a snapshot of such a collection restores as a
// plain HNSW of the recorded geometry. Restoring those advanced configs
// faithfully requires re-creating the collection with its original config and
// using Restore — see the backup package — and is out of scope here.)
func (s *CollectionStore) RestoreCollection(name string, r io.Reader) error {
	// Placeholder config: Dim/M/Ef* must pass Config.Validate at construction;
	// readSnapshot replaces them wholesale from the snapshot's header. Note the
	// zero-valued Metric here is Cosine, and Dim is 1 — i.e. this placeholder is
	// WRONG for most snapshots on purpose, and correctness depends on
	// readSnapshot re-deriving EVERY cfg-derived cached value after its
	// `h.cfg = cfg` (today: mL and the pair-distance kernel). Anything cached
	// off cfg that readSnapshot forgets to refresh silently keeps the
	// placeholder's cosine/dim-1 geometry on the restored index.
	//
	// This is the config-LESS path — advanced quant/IndexType/Vamana geometry
	// (NOT in the stream) restores as a plain HNSW of the recorded geometry. Use
	// RestoreCollectionWithConfig when the original Config is available (e.g. the
	// backup package's sibling .cfg.json) to restore those faithfully.
	placeholder := Config{Dim: 1, M: 4, EfConstruction: 1, EfSearch: 1}
	return s.RestoreCollectionWithConfig(name, placeholder, r)
}

// RestoreCollectionWithConfig is RestoreCollection that re-creates the target
// collection with an EXACT, caller-supplied Config before applying
// Collection.Restore. This is the config-FAITHFUL restore: because the dense
// snapshot stream does NOT carry quantization, IndexType, or Vamana geometry
// (Quant/QuantPQM/SQBits/IndexType/VamanaR/L/Alpha …), restoring such a
// collection on a fresh store requires re-creating it with the original Config so
// the index is built with the right quantizer/index family BEFORE Restore loads
// its codes/graph on top. The backup package supplies this Config from the
// sibling .cfg.json object written alongside each snapshot.
//
// It is create-or-replace: any existing collection of this name is Dropped first
// (its on-disk files removed), then a fresh collection is constructed from cfg and
// Collection.Restore is applied.
func (s *CollectionStore) RestoreCollectionWithConfig(name string, cfg Config, r io.Reader) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	if err := s.DropCollection(canonical); err != nil {
		return err
	}
	if err := s.CreateCollection(canonical, cfg); err != nil {
		return err
	}
	c, ok := s.Acquire(canonical)
	if !ok {
		return fmt.Errorf("vector: restore %q: collection vanished after create", canonical)
	}
	defer c.Release()
	if err := c.Restore(r); err != nil {
		return fmt.Errorf("vector: restore %q: %w", canonical, err)
	}
	return nil
}

// Get returns the named collection (canonicalized for tenant lookup).
func (s *CollectionStore) Get(name string) (*Collection, bool) {
	canonical, err := canonicalName(name)
	if err != nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	c, ok := s.collections[canonical]
	return c, ok
}

// CollectionNames returns a snapshot of the canonical names of every dense
// collection currently registered, taken under the store lock so a concurrent
// Create/Drop cannot tear the slice. The order is unspecified (Go map order);
// callers that need determinism (e.g. backup key layout) sort it themselves.
// Only the dense family is reported — the multi-vector and named-vector families
// are intentionally excluded, mirroring the backup driver's current scope.
func (s *CollectionStore) CollectionNames() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	names := make([]string, 0, len(s.collections)+len(s.cold))
	for name := range s.collections {
		names = append(names, name)
	}
	// Cold-tiered (evicted) collections are still part of the catalog — their
	// identity and config survive the eviction — so they remain listed. A name is
	// in collections XOR cold, so this never double-counts.
	for name := range s.cold {
		names = append(names, name)
	}
	return names
}

// WritePrometheusAll renders the Prometheus text exposition for every dense
// collection currently registered, concatenated. The collections are snapshotted
// under the store lock AND each is pinned with an inuse reference while still
// holding the lock; the actual rendering (one Stats read + format per collection)
// then happens OUTSIDE the lock so a slow io.Writer cannot stall concurrent
// Create/Drop/insert traffic. Pinning is essential: without the ref a concurrent
// Drop's retire() (which spins only while inuse>0) would close/unmap the index
// out from under the render, a use-after-free (SIGBUS on mmap-backed indexes).
// The deferred release runs on every exit path so a render error can never wedge
// a later Drop. Each series is already labeled collection="<tenant>/<name>" by
// Collection.WritePrometheus.
//
// Only the dense family is exported — the multi-vector and named-vector indexes
// expose no Stats/WritePrometheus surface yet (follow-up).
func (s *CollectionStore) WritePrometheusAll(w io.Writer) error {
	s.mu.RLock()
	handles := make([]*Collection, 0, len(s.collections))
	for _, c := range s.collections {
		c.inuse.Add(1) // pin under the lock; Drop's map-delete needs the write lock, so retire cannot interleave
		handles = append(handles, c)
	}
	s.mu.RUnlock()
	defer func() {
		for _, c := range handles {
			c.Release()
		}
	}()
	for _, c := range handles {
		if err := c.WritePrometheus(w); err != nil {
			return err
		}
	}
	return nil
}

// Flush writes the named collection's current index state to disk atomically.
func (s *CollectionStore) Flush(name string) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}
	c, ok := s.Acquire(name)
	if !ok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()

	_, snapPath := s.collectionPath(canonical)

	// WAL collections checkpoint with a CONSISTENT snapshot (not the mmap
	// sidecar) and rotate the log — atomically w.r.t. Insert/Delete via opMu, so
	// the checkpoint always subsumes the truncated log.
	//
	// This DOES stall every writer for the length of the serialization, and the
	// obvious fix — capture the WAL position under opMu, release it, serialize
	// under the index read lock, re-take opMu to truncate the prefix — was
	// evaluated and REJECTED. It buys nothing: Snapshot holds h.mu.RLock across
	// the ENTIRE serialization (vector/snapshot.go:427), and every mutator needs
	// h.mu.Lock (e.g. vector/hnsw.go:1525), so writers block for exactly as long
	// on a different lock. It also costs a lot: the "checkpoint subsumes the
	// truncated log" invariant above stops being free (wal.truncate is
	// Truncate(0); a prefix truncate would need new crash-safe machinery), and
	// dense replay is not a general replay-over-a-newer-snapshot primitive —
	// an insert record for an already-live id is silently dropped
	// (vector/arena.go:376, error discarded at collection.go's replayWAL).
	// See vector/flush_snapshot_lock_test.go, which pins both facts and flips to
	// failing if snapshot serialization ever stops excluding writers.
	if c.cfg.WAL {
		c.opMu.Lock()
		defer c.opMu.Unlock()
		if err := s.writeSnapshotFile(c, snapPath); err != nil {
			return err
		}
		return c.wal.truncate()
	}

	// Persistent (no WAL) collections flush via the native sidecar (instant
	// restart); the mmap-backed vectors + graph are synced in place by SavePersist.
	if c.cfg.Persistent {
		_, _, metaPath, _ := s.persistPaths(canonical)
		if err := os.MkdirAll(filepath.Dir(metaPath), 0o750); err != nil {
			return err
		}
		return c.SavePersist(metaPath)
	}

	return s.writeSnapshotFile(c, snapPath)
}

// writeSnapshotFile atomically writes c's snapshot to path (tmp + fsync + rename).
func (s *CollectionStore) writeSnapshotFile(c *Collection, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if err := c.Snapshot(f); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	// renameDurable (rename + parent-dir fsync), NOT a bare os.Rename: Flush
	// truncates the WAL on the strength of this checkpoint. Without the dir
	// fsync the truncate (itself fsync'd) can reach disk BEFORE the rename's
	// directory entry — a crash in that window reopens on the OLD checkpoint
	// with an EMPTY log, silently dropping the whole inter-checkpoint delta.
	return renameDurable(tmp, path)
}

// Insert inserts a vector with the given TTL, metadata, and sparse vector into
// the named collection. ttl=0 means no expiry; meta=nil means no metadata;
// sparse=nil means no sparse lane. Returns an error if the collection does not
// exist. Name may be bare (resolves to "default/<name>") or tenant-qualified.
func (s *CollectionStore) Insert(name string, id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) error {
	c, ok := s.Acquire(name)
	if !ok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.Insert(id, vec, ttl, meta, sparse)
}

// InsertCAS inserts/upserts with an optimistic-CAS precondition, returning the
// resulting version. A mismatch returns ErrVersionConflict (no mutation). See
// Collection.InsertCAS.
func (s *CollectionStore) InsertCAS(name string, id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, cas CASCond) (uint64, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.InsertCAS(id, vec, ttl, meta, sparse, cas)
}

// UpsertCAS inserts-or-replaces with an optimistic-CAS precondition, returning the
// resulting version. See Collection.UpsertCAS.
func (s *CollectionStore) UpsertCAS(name string, id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector, cas CASCond) (uint64, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.UpsertCAS(id, vec, content, ttl, meta, sparse, cas)
}

// DeleteCAS deletes id with an optimistic-CAS precondition. removed reports
// whether id was live; a mismatch returns ErrVersionConflict. See
// Collection.DeleteCAS.
func (s *CollectionStore) DeleteCAS(name string, id uint64, cas CASCond) (bool, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.DeleteCAS(id, cas)
}

// GetPoint retrieves a live point by id from the named collection: its
// deep-copied vector + payload + sparse + remaining TTL. ok is false for an
// absent/tombstoned/expired point. A pure read. See Collection.Get. (Named
// GetPoint to avoid colliding with Get(name)→(*Collection,bool).)
func (s *CollectionStore) GetPoint(name string, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, ok bool, err error) {
	c, cok := s.Acquire(name)
	if !cok {
		return nil, nil, 0, nil, false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	vec, meta, ttl, sparse, _, ok = c.Get(id) // version-less form; see GetPointVersion
	return vec, meta, ttl, sparse, ok, nil
}

// GetPointInto is GetPoint that appends the dense vector into the caller-owned
// scratch dst (passed as dst[:0]) instead of allocating a fresh []float32 each
// call — the dense-read analogue of cache.GetInto. A hot-loop caller reusing one
// buffer pays zero allocations for the vector copy. See Collection.GetInto for
// the aliasing and PQDropVecs caveats.
func (s *CollectionStore) GetPointInto(dst []float32, name string, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, ok bool, err error) {
	c, cok := s.Acquire(name)
	if !cok {
		return nil, nil, 0, nil, false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	vec, meta, ttl, sparse, _, ok = c.GetInto(dst, id) // version-less form; see GetPointVersionInto
	return vec, meta, ttl, sparse, ok, nil
}

// GetPointVersion is GetPoint plus the point's per-point CAS version (>=1 for a
// live point; 0 for an absent/dead point). The wire layer surfaces it on the
// get-result codec so a caller can read-then-CAS-write.
func (s *CollectionStore) GetPointVersion(name string, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool, err error) {
	c, cok := s.Acquire(name)
	if !cok {
		return nil, nil, 0, nil, 0, false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	vec, meta, ttl, sparse, version, ok = c.Get(id)
	return vec, meta, ttl, sparse, version, ok, nil
}

// GetPointVersionInto is GetPointVersion that appends the dense vector into the
// caller-owned scratch dst (passed as dst[:0]) instead of allocating a fresh
// []float32 — the version-returning Into form the single-get handler uses to pool
// the dense buffer. See Collection.GetInto for the aliasing/PQDropVecs caveats.
func (s *CollectionStore) GetPointVersionInto(dst []float32, name string, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool, err error) {
	c, cok := s.Acquire(name)
	if !cok {
		return nil, nil, 0, nil, 0, false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	vec, meta, ttl, sparse, version, ok = c.GetInto(dst, id)
	return vec, meta, ttl, sparse, version, ok, nil
}

// GetPointsProjected fetches a batch of ids from one collection in a SINGLE
// Acquire/Release, invoking fn once per id in the given order. withVec/withPayload
// gate the dense-vector copy and the meta+sparse clones respectively, so the
// projections a caller will discard are never copied (Collection.GetProjected). This
// is the batch-read fast path: the previous per-id GetPointVersion paid one
// Acquire/Release AND a full dense+meta+sparse deep copy per id regardless of
// projection; a with_vector=false batch now does one Acquire and allocates nothing
// per point. The vec/meta/sparse passed to fn are owned by fn (safe to retain);
// there is no shared scratch across calls. Returns an error only when the collection
// is absent.
func (s *CollectionStore) GetPointsProjected(name string, ids []uint64, withVec, withPayload bool, fn func(id uint64, vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool)) error {
	c, cok := s.Acquire(name)
	if !cok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	for _, id := range ids {
		vec, meta, ttl, sparse, version, ok := c.GetProjected(id, withVec, withPayload)
		fn(id, vec, meta, ttl, sparse, version, ok)
	}
	return nil
}

// SetPayload merges patch into id's payload (reindexing + WAL-logging on a dense
// WAL-mode collection). keyTTLMs sets per-key relative TTLs (key -> ms; ttl<=0
// clears a key's deadline). Returns applied=false (NOT an error) for an
// absent/tombstoned/expired point, so a point-op fan-out treats the not-found as
// expected; a real failure propagates as err. See Collection.SetPayload.
func (s *CollectionStore) SetPayload(name string, id uint64, patch Metadata, keyTTLMs map[string]int64) (applied bool, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadApplied(c.SetPayload(id, patch, keyTTLMs))
}

// OverwritePayload replaces id's entire payload with meta. keyTTLMs sets the
// per-key relative TTLs. applied=false for a dead point (not an error). See
// Collection.OverwritePayload.
func (s *CollectionStore) OverwritePayload(name string, id uint64, meta Metadata, keyTTLMs map[string]int64) (applied bool, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadApplied(c.OverwritePayload(id, meta, keyTTLMs))
}

// DeletePayloadKeys removes the listed keys from id's payload. applied=false for a
// dead point (not an error). See Collection.DeletePayloadKeys.
func (s *CollectionStore) DeletePayloadKeys(name string, id uint64, keys []string) (applied bool, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadApplied(c.DeletePayloadKeys(id, keys))
}

// ClearPayload removes all of id's payload. applied=false for a dead point (not an
// error). See Collection.ClearPayload.
func (s *CollectionStore) ClearPayload(name string, id uint64) (applied bool, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadApplied(c.ClearPayload(id))
}

// SetPayloadCAS merges patch into id's payload with an optimistic-CAS
// precondition. Returns applied (false for an absent point — a FLAG, not an
// error), the resulting version, and err (ErrVersionConflict on a mismatch). See
// Collection.SetPayloadCAS.
func (s *CollectionStore) SetPayloadCAS(name string, id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.SetPayloadCAS(id, patch, keyTTLMs, cas))
}

// OverwritePayloadCAS replaces id's payload with an optimistic-CAS precondition.
func (s *CollectionStore) OverwritePayloadCAS(name string, id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.OverwritePayloadCAS(id, meta, keyTTLMs, cas))
}

// DeletePayloadKeysCAS removes keys from id's payload with an optimistic-CAS
// precondition.
func (s *CollectionStore) DeletePayloadKeysCAS(name string, id uint64, keys []string, cas CASCond) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.DeletePayloadKeysCAS(id, keys, cas))
}

// ClearPayloadCAS clears id's payload with an optimistic-CAS precondition.
func (s *CollectionStore) ClearPayloadCAS(name string, id uint64, cas CASCond) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.ClearPayloadCAS(id, cas))
}

// The ...At store variants judge the per-key deadline computation (Set/Overwrite)
// AND the dead-point liveness gate against the EXPLICIT leader apply stamp nowMs —
// the replicated-apply path the vector payload handlers take under a stamp (#4
// vector TTL determinism). The non-At forms use the wall clock (byte-identical).
func (s *CollectionStore) SetPayloadCASAt(name string, id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.SetPayloadCASAt(id, patch, keyTTLMs, cas, nowMs))
}

func (s *CollectionStore) OverwritePayloadCASAt(name string, id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.OverwritePayloadCASAt(id, meta, keyTTLMs, cas, nowMs))
}

func (s *CollectionStore) DeletePayloadKeysCASAt(name string, id uint64, keys []string, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.DeletePayloadKeysCASAt(id, keys, cas, nowMs))
}

func (s *CollectionStore) ClearPayloadCASAt(name string, id uint64, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.ClearPayloadCASAt(id, cas, nowMs))
}

// MutatePayloadRecordCAS applies fn to the operate record under key in the named
// collection's point id. See Collection.MutatePayloadRecordCAS. An absent point
// is the not-found FLAG (applied=false, err=nil), like every other payload
// dispatcher here.
func (s *CollectionStore) MutatePayloadRecordCAS(name string, id uint64, key string, fn RecordMutator, cas CASCond) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.MutatePayloadRecordCAS(id, key, fn, cas))
}

// MutatePayloadRecordCASAt is MutatePayloadRecordCAS under the leader apply
// stamp nowMs. See Collection.MutatePayloadRecordCASAt.
func (s *CollectionStore) MutatePayloadRecordCASAt(name string, id uint64, key string, fn RecordMutator, cas CASCond, nowMs int64) (applied bool, version uint64, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return false, 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return payloadAppliedV(c.MutatePayloadRecordCASAt(id, key, fn, cas, nowMs))
}

// payloadAppliedV is payloadApplied for the CAS variants that also return the
// resulting version: ErrIDNotFound → (false, 0, nil) (absent point is a FLAG);
// any other error (incl. ErrVersionConflict) → (false, 0, err); nil → (true,
// version, nil).
func payloadAppliedV(version uint64, err error) (bool, uint64, error) {
	if errors.Is(err, ErrIDNotFound) {
		return false, 0, nil
	}
	if err != nil {
		return false, 0, err
	}
	return true, version, nil
}

// payloadApplied maps an engine payload-mutation error onto the (applied, err)
// not-found-flag contract: ErrIDNotFound → (false, nil) (the point was absent —
// a FLAG, never an op error); any other error → (false, err); nil → (true, nil).
// Shared by the dense/named/MV CollectionStore payload dispatchers.
func payloadApplied(err error) (bool, error) {
	if errors.Is(err, ErrIDNotFound) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// Upsert inserts or replaces a record (vector + document content + metadata) in
// the named collection — the RAG-store write path. See Collection.Upsert.
func (s *CollectionStore) Upsert(name string, id uint64, vec []float32, content string, ttl time.Duration, meta Metadata, sparse *SparseVector) error {
	c, ok := s.Acquire(name)
	if !ok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.Upsert(id, vec, content, ttl, meta, sparse)
}

// SearchDocs runs a filtered KNN search against the named collection and returns
// hits enriched with stored content + metadata. See Collection.SearchDocs.
func (s *CollectionStore) SearchDocs(name string, query []float32, k int, filter Filter) ([]Document, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.SearchDocs(query, k, filter)
}

// SearchText runs a BM25 full-text search against the named collection and
// returns documents enriched with content + metadata. See Collection.SearchText.
func (s *CollectionStore) SearchText(name string, query string, k int, filter Filter) ([]Document, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.SearchText(query, k, filter)
}

// HybridText fuses a dense KNN lane with a BM25 full-text lane in the named
// collection. See Collection.HybridText.
func (s *CollectionStore) HybridText(name string, dense []float32, query string, k int, opts HybridOpts) ([]Result, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.HybridText(dense, query, k, opts)
}

// CorpusStats returns the named collection's corpus-wide BM25 stats for query's
// terms — phase 0 of the global-DF (dfs_query_then_fetch) fan-out. Zero/nil when
// full text is disabled (a non-full-text partition adds nothing to the global sum).
// See Collection.CorpusStats.
func (s *CollectionStore) CorpusStats(name string, query string) (n int, tokenTotal uint64, df map[uint32]int, err error) {
	c, ok := s.Acquire(name)
	if !ok {
		return 0, 0, nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	n, tokenTotal, df = c.CorpusStats(query)
	return n, tokenTotal, df, nil
}

// SearchTextGlobal runs a BM25 full-text search against the named collection scored
// with coordinator-supplied GLOBAL stats g — phase 1 of the global-DF fan-out. See
// Collection.SearchTextGlobal.
func (s *CollectionStore) SearchTextGlobal(name string, query string, k int, filter Filter, g BM25GlobalStats) ([]Result, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.SearchTextGlobal(query, k, filter, g)
}

// HybridTextLanesGlobal returns the named collection's dense + BM25-text candidate
// lanes UNFUSED, the text lane scored with the coordinator-supplied GLOBAL stats g
// (phase 1 of the global-DF fan-out). See Collection.HybridTextLanesGlobal.
func (s *CollectionStore) HybridTextLanesGlobal(name string, dense []float32, query string, k int, opts HybridOpts, g BM25GlobalStats) ([]Result, []Result, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.HybridTextLanesGlobal(dense, query, k, opts, g)
}

// SearchGroups runs a group-by-document search against the named collection,
// returning the top-k groups (best member first) with up to opts.GroupSize hits
// each. See Collection.SearchGroups.
func (s *CollectionStore) SearchGroups(name string, query []float32, k int, opts GroupOpts) ([]Group, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.SearchGroups(query, k, opts)
}

// ScrollDocs lists live documents in the named collection matching filter (zero
// filter = all), up to limit. See Collection.ScrollDocs.
func (s *CollectionStore) ScrollDocs(name string, filter Filter, limit int) ([]Document, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.ScrollDocs(filter, limit)
}

// DeleteByFilter deletes all records in the named collection matching filter
// (e.g. every chunk of a document), returning the count removed.
func (s *CollectionStore) DeleteByFilter(name string, filter Filter) (int, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return 0, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.DeleteByFilter(filter)
}

// SearchFiltered runs a filtered KNN search against the named collection.
// Name may be bare (resolves to "default/<name>") or tenant-qualified.
func (s *CollectionStore) SearchFiltered(name string, query []float32, k int, filter Filter) ([]Result, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.SearchFiltered(query, k, filter)
}

// StageBulk appends (id, vec) pairs to the named collection's bulk-load staging
// buffer (see Collection.StageBulk). Concurrency-safe. Returns ErrDimMismatch —
// staging nothing — when a vector's length is not the collection's Dim.
func (s *CollectionStore) StageBulk(name string, ids []uint64, vecs [][]float32) error {
	c, ok := s.Acquire(name)
	if !ok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.StageBulk(ids, vecs)
}

// StageBulkPayloads appends (id, vec, payload) triples to the named collection's
// bulk-load staging buffer (see Collection.StageBulkPayloads). Concurrency-safe.
func (s *CollectionStore) StageBulkPayloads(name string, ids []uint64, vecs [][]float32, metas []Metadata) error {
	c, ok := s.Acquire(name)
	if !ok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.StageBulkPayloads(ids, vecs, metas)
}

// BuildStaged builds the named collection's staged vectors in one concurrent
// pass (see Collection.BuildStaged). The collection must be empty.
func (s *CollectionStore) BuildStaged(name string, workers int) error {
	c, ok := s.Acquire(name)
	if !ok {
		return fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.BuildStaged(workers)
}

// HybridSearch runs a fused dense + sparse search against the named collection.
func (s *CollectionStore) HybridSearch(name string, dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, error) {
	c, ok := s.Acquire(name)
	if !ok {
		return nil, fmt.Errorf("vector: no collection %q", name)
	}
	defer c.Release()
	return c.HybridSearch(dense, sparse, k, opts)
}

// FlushPersistent flushes every Persistent collection's instant-restart
// sidecar — dense AND multi-vector — and reports the joined failures rather
// than the first.
//
// A Persistent collection keeps its vectors and level-0 graph in mmap files, so
// those bytes already survive process exit — but the sidecar that makes them
// *readable* again (per-slot ids, node levels, edge lengths, entry point) is
// only ever written by Flush. Without one, reopening the collection yields a
// fresh empty index, so an embedded caller that never flushes gets the mmap
// files' cost and none of their benefit. This is the shutdown hook a caller
// needs to honour the "callers control persistence explicitly via Flush"
// contract for a whole store at once, without reaching into each collection's
// config to find out which ones care.
//
// Multi-vector collections are covered because they have exactly the same
// hazard: CreateMultiVector with Persistent set mmaps the inner index and
// writes its meta + maps sidecars only from FlushMultiVector, so one skipped
// here reopens empty after a perfectly clean Close. directStore.Close relies on
// this method alone, and an embedded caller can create a persistent MV
// collection through the ordinary API.
//
// What is deliberately skipped, and why:
//
//   - Non-persistent (heap) collections. Their Flush writes a full snapshot,
//     which is a caller's explicit choice, not something a shutdown should do
//     behind their back.
//   - WAL collections, dense or multi-vector. Every op is already logged, so
//     replay reconstructs them without a checkpoint. (WAL and Persistent are
//     mutually exclusive by construction in both flavours.)
//   - Named collections. They have no mmap-Persistent mode at all — only heap
//     or WAL — so both exclusions above already cover them and FlushNamed has
//     nothing to contribute here.
func (s *CollectionStore) FlushPersistent() error {
	// A heap-mode store's scratch dir goes away with it at Close, so every byte
	// written here would be deleted moments later.
	if s.ephemeralDir {
		return nil
	}
	// Names are snapshotted under the lock and flushed outside it: the flushes
	// themselves take the store lock via Acquire/AcquireMulti, and serializing a
	// large index can be slow enough that holding the write lock across it would
	// stall every reader.
	s.mu.RLock()
	names := make([]string, 0, len(s.collections))
	for name, c := range s.collections {
		if c.cfg.Persistent && !c.cfg.WAL {
			names = append(names, name)
		}
	}
	multi := make([]string, 0, len(s.multi))
	for name, mv := range s.multi {
		if mv.persistent {
			multi = append(multi, name)
		}
	}
	s.mu.RUnlock()

	var errs []error
	for _, name := range names {
		if err := s.Flush(name); err != nil {
			errs = append(errs, fmt.Errorf("vector: flushing persistent collection %q: %w", name, err))
		}
	}
	for _, name := range multi {
		if err := s.FlushMultiVector(name); err != nil {
			errs = append(errs, fmt.Errorf("vector: flushing persistent multi-vector collection %q: %w", name, err))
		}
	}
	return errors.Join(errs...)
}

// Close releases the in-memory collections (callers control persistence
// explicitly via Flush — or FlushPersistent — before Close).
func (s *CollectionStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.collections {
		c.Stop()
		_ = c.Close()
	}
	for _, mv := range s.multi {
		mv.Stop()
		_ = mv.Close()
	}
	for _, nc := range s.named {
		nc.Stop()
		_ = nc.Close()
	}
	s.collections = nil
	s.multi = nil
	s.named = nil
	// Cold stubs hold no index/goroutine — just drop the catalog maps. Per-collection
	// lastAccess lives on the Collection objects, which are dropped with s.collections.
	s.cold = nil
	// A heap-mode store owns its scratch dir, so it takes it with it — and says so
	// if it cannot. Reporting success over a directory that is still on disk would
	// make the ownership contract unobservable exactly when it has been broken,
	// and "the OS will reclaim it" is not a guarantee any platform actually makes.
	if s.ephemeralDir && s.dir != "" {
		if err := os.RemoveAll(s.dir); err != nil {
			return fmt.Errorf("vector: remove heap-mode scratch dir %s: %w", s.dir, err)
		}
	}
	return nil
}

// MaybeReclaim triggers reclaim on the named collection only if its
// tombstone ratio is above threshold. Returns (count, ran) — count is the
// number of slots reclaimed if ran is true.
func (s *CollectionStore) MaybeReclaim(name string, threshold float64) (int, bool) {
	c, ok := s.Acquire(name)
	if !ok {
		return 0, false
	}
	defer c.Release()
	if c.TombstoneRatio() < threshold {
		return 0, false
	}
	return c.Reclaim(), true
}

func readConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func writeConfig(path string, cfg Config) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return atomicWriteFile(path, data)
}
