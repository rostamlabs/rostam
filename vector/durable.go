// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// StoreOptions configures OpenCollectionStoreWith.
type StoreOptions struct {
	// PersistentCluster selects the per-node persistent-cluster policy (see
	// CollectionStore.persistentCluster).
	PersistentCluster bool

	// DurableCollections names the dense collections that always run with the
	// write-ahead log, fsynced before every acknowledgement, whatever their
	// create or restore request asked for. Bare names resolve to the default
	// tenant. Ignored in cluster mode, where Raft is the durability authority.
	DurableCollections []string
}

// durableCheckpointBytes is the log size above which a durable collection is
// checkpointed while the store opens, so replay time stays bounded. A variable
// so tests can lower it.
var durableCheckpointBytes int64 = 16 << 20

// canonicalDurable canonicalises the configured durable names. An invalid name
// fails the open rather than silently leaving a collection unprotected.
func canonicalDurable(names []string) (map[string]struct{}, error) {
	if len(names) == 0 {
		return nil, nil
	}
	out := make(map[string]struct{}, len(names))
	for _, n := range names {
		c, err := canonicalName(n)
		if err != nil {
			return nil, fmt.Errorf("vector: durable collection %q: %w", n, err)
		}
		out[c] = struct{}{}
	}
	return out, nil
}

// isDurable reports whether canonical is a durable collection. Always false in
// cluster mode.
func (s *CollectionStore) isDurable(canonical string) bool {
	if s.persistentCluster {
		return false
	}
	_, ok := s.durable[canonical]
	return ok
}

// warnNotDurable warns, when canonical is listed as durable, that this
// collection does not get the synced write-ahead log: kind says what it is (a
// named-vector or multi-vector collection, or a dense one on a store with no
// data directory, whose files are removed at Close). Called at create and at
// open, so a listed name is never left unprotected silently.
func (s *CollectionStore) warnNotDurable(canonical, kind string) {
	if !s.isDurable(canonical) {
		return
	}
	slog.Warn("vector: durable collection is not crash-safe: only a dense collection in a store with a data directory runs with the synced write-ahead log",
		"collection", canonical, "kind", kind)
}

// durableConfig returns cfg with the durable mode applied when canonical is a
// listed dense collection: write-ahead log on and fsynced, no mmap
// instant-restart (the two are mutually exclusive). Applied before the config is
// validated and before the manifest is written.
func (s *CollectionStore) durableConfig(canonical string, cfg Config) Config {
	if !s.isDurable(canonical) || len(cfg.NamedVectors) > 0 {
		return cfg
	}
	cfg.WAL = true
	cfg.Persistent = false
	cfg.WALNoSync = false
	return cfg
}

// checkpoint writes c's snapshot and then truncates its log, holding opMu so no
// write lands between the two. A stop between them leaves the new snapshot next
// to the whole old log, which replays over it to the same state.
func (s *CollectionStore) checkpoint(canonical string, c *Collection) error {
	c.opMu.Lock()
	defer c.opMu.Unlock()
	_, snapPath := s.collectionPath(canonical)
	if err := s.writeSnapshotFile(c, snapPath); err != nil {
		return err
	}
	return c.wal.truncate()
}

// checkpointLargeLog checkpoints a durable collection just loaded at open when
// its log has outgrown durableCheckpointBytes. A failure only warns: the
// collection is intact and the log still holds every write.
func (s *CollectionStore) checkpointLargeLog(canonical, walPath string, c *Collection) {
	st, err := os.Stat(walPath)
	if err != nil || st.Size() <= durableCheckpointBytes {
		return
	}
	if err := s.checkpoint(canonical, c); err != nil {
		slog.Warn("vector: checkpointing a durable collection at open failed; its log is kept",
			"collection", canonical, "log", walPath, "bytes", st.Size(), "err", err)
	}
}

// CheckpointDurable snapshots every durable collection that runs with the
// write-ahead log and truncates its log. Errors are joined; one collection's
// failure does not stop the others. A heap-mode store, whose files are deleted
// at Close, and a cluster store write nothing.
func (s *CollectionStore) CheckpointDurable() error {
	if s.ephemeralDir || s.persistentCluster || len(s.durable) == 0 {
		return nil
	}
	s.mu.RLock()
	names := make([]string, 0, len(s.durable))
	for name := range s.durable {
		if c, ok := s.collections[name]; ok && c.cfg.WAL {
			names = append(names, name)
		}
	}
	s.mu.RUnlock()

	var errs []error
	for _, name := range names {
		// A collection dropped since the listing is simply skipped; one replaced
		// since is checkpointed in its new form.
		c, ok := s.Acquire(name)
		if !ok {
			continue
		}
		if c.cfg.WAL && c.wal != nil {
			if err := s.checkpoint(name, c); err != nil {
				errs = append(errs, fmt.Errorf("vector: checkpointing durable collection %q: %w", name, err))
			}
		}
		c.Release()
	}
	return errors.Join(errs...)
}

// syncDurableWAL is run at open for a listed collection whose manifest already
// says WAL. A manifest that asked for an unsynced log is rewritten to a synced
// one; a failed rewrite only warns, because the returned config, which the log
// is opened with, syncs regardless.
func (s *CollectionStore) syncDurableWAL(canonical string, cfg Config, cfgPath string) Config {
	if cfg.WALNoSync {
		cfg.WALNoSync = false
		if err := writeConfig(cfgPath, cfg); err != nil {
			slog.Warn("vector: rewriting a durable collection's manifest to a synced log failed; its log syncs anyway",
				"collection", canonical, "manifest", cfgPath, "err", err)
		}
	}
	return cfg
}

// removeMappedLeftovers removes, best effort, the mapped files and sidecar of a
// listed collection's earlier Persistent form. It runs only once the log form
// has loaded, so a log form that fails to load still has them beside it. Files
// that cannot be removed (Windows keeps a file that is still mapped) are left for
// the next open.
func (s *CollectionStore) removeMappedLeftovers(canonical string) {
	vecsPath, graphPath, metaPath, _ := s.persistPaths(canonical)
	for _, p := range []string{vecsPath, graphPath, metaPath} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			slog.Warn("vector: removing a file left by an earlier form of a durable collection failed; the next open retries",
				"collection", canonical, "file", p, "err", err)
		}
	}
}

// upgradeStepHook, when non-nil, is called after each step of upgradeToDurable
// with the step's name: "loaded", "snapshot", "log-removed", "manifest". An
// error it returns is that step's failure. Tests only; nil in production.
var upgradeStepHook func(step string) error

func upgradeStep(step string) error {
	if upgradeStepHook != nil {
		return upgradeStepHook(step)
	}
	return nil
}

// upgradeToDurable converts the listed Persistent collection canonical, just
// opened and registered in its old form, to a write-ahead log collection. cfg is
// its manifest, read from cfgPath; snapPath is where its snapshot goes.
//
// The live points are copied into a fresh in-memory collection, never the
// mapped graph, which after an unclean stop can name slots that do not exist.
// That collection is written as a synced snapshot and proven to restore, any
// stale log is removed, and then the manifest is replaced: the commit point.
// Until then nothing the old form reads has been touched. The new form is then
// loaded beside the old one, and only once it has loaded is the old one closed
// and replaced. If it does not load, the old manifest is put back.
//
// No failure here fails the open: every one leaves the old form registered and
// working, warns that the collection is not crash-safe, and the next open tries
// again. The return value is always nil; it is kept for the caller's shape.
func (s *CollectionStore) upgradeToDurable(canonical string, cfg Config, cfgPath, snapPath string) error {
	old := s.collections[canonical]
	walCfg := s.durableConfig(canonical, cfg)
	if !s.keepOrphanSnapshot(canonical, walCfg, snapPath) {
		return nil
	}
	res := s.convertToDurable(canonical, old, cfg, walCfg, cfgPath, snapPath)
	if !res.committed {
		if res.oldManifestCertain {
			_ = os.Remove(snapPath)
			slog.Warn("vector: durable collection is not crash-safe: upgrading it to the write-ahead log failed, so it stays in its old form until a later open succeeds",
				"collection", canonical, "err", res.err)
		} else {
			// The snapshot is kept: the manifest may name the log form, and then
			// the snapshot is the collection.
			slog.Error("vector: durable collection is not crash-safe, and its manifest could not be read back or put back after a failed upgrade: it stays in its old form for now, but the next open may load the write-ahead log form instead and miss what is written until then",
				"collection", canonical, "manifest", cfgPath, "err", res.err)
		}
		return nil
	}

	nc, err := s.buildWALCollection(canonical, walCfg, cfgPath, snapPath)
	if err != nil {
		s.rollbackUpgrade(canonical, cfg, cfgPath, snapPath, err)
		return nil
	}
	s.collections[canonical] = nc
	if cerr := old.Close(); cerr != nil {
		slog.Warn("vector: closing the old form of an upgraded collection failed",
			"collection", canonical, "err", cerr)
	}
	if res.manifestDurable {
		s.removeMappedLeftovers(canonical)
	} else {
		slog.Warn("vector: the upgraded manifest of a durable collection may not be on disk yet, so the files of its old form are kept until a later open",
			"collection", canonical, "manifest", cfgPath)
	}
	return nil
}

// rollbackUpgrade runs when the log form of a just-committed upgrade does not
// load. The old manifest is put back, atomically, and the snapshot and the log
// the load may have created are removed; the old form, still open, stays.
func (s *CollectionStore) rollbackUpgrade(canonical string, cfg Config, cfgPath, snapPath string, loadErr error) {
	if werr := writeConfig(cfgPath, cfg); werr != nil {
		slog.Error("vector: durable collection is not crash-safe: its upgraded form did not load and the old manifest could not be put back; it stays in its old form for now, but the next open loads the write-ahead log form and may fail",
			"collection", canonical, "manifest", cfgPath, "err", loadErr, "rollback_err", werr)
		return
	}
	_, _, _, walPath := s.persistPaths(canonical)
	_ = os.Remove(snapPath)
	_ = os.Remove(walPath)
	slog.Warn("vector: durable collection is not crash-safe: its upgraded form did not load, so the old manifest was put back and it stays in its old form until a later open succeeds",
		"collection", canonical, "err", loadErr)
}

// keepOrphanSnapshot handles a listed collection whose Persistent manifest has
// no sidecar beside it but a snapshot. The upgrade never leaves that state with
// points in the snapshot (it removes the sidecar only after the log manifest is
// durable), but if it is met the snapshot may hold the collection, and the
// upgrade of the empty old form would overwrite it. A snapshot holding points,
// or one that does not restore, is moved aside under a name the warning gives;
// an empty one, as an interrupted upgrade of a never-flushed collection leaves,
// is simply overwritten. It reports false, skipping the upgrade, when the
// snapshot cannot be moved.
func (s *CollectionStore) keepOrphanSnapshot(canonical string, walCfg Config, snapPath string) bool {
	_, _, metaPath, _ := s.persistPaths(canonical)
	if _, err := os.Stat(metaPath); err == nil {
		return true
	}
	if _, err := os.Stat(snapPath); err != nil {
		return true
	}
	if n, err := s.snapshotPoints(canonical, walCfg, snapPath); err == nil && n == 0 {
		return true
	}
	kept := fmt.Sprintf("%s.kept-%d", snapPath, time.Now().UnixNano())
	// Not renameDurable: it removes its source when the rename fails, and the
	// source here is the snapshot that may hold the collection.
	err := os.Rename(snapPath, kept)
	if err == nil {
		err = syncDirf(filepath.Dir(kept))
	}
	if err != nil {
		slog.Warn("vector: durable collection is not crash-safe: it has a snapshot but no sidecar, and the snapshot could not be moved aside, so it is not upgraded",
			"collection", canonical, "snapshot", snapPath, "err", err)
		return false
	}
	slog.Warn("vector: a durable collection had a snapshot but no sidecar; the snapshot was kept aside before upgrading",
		"collection", canonical, "kept", kept)
	return true
}

// upgradeResult is the outcome of convertToDurable.
type upgradeResult struct {
	// committed: the manifest now names the log form.
	committed bool
	// manifestDurable: when committed, the manifest's rename is known to be
	// on disk, so the old files may go.
	manifestDurable bool
	// oldManifestCertain: when not committed, the manifest is known to name the
	// old form, so the snapshot may go.
	oldManifestCertain bool
	err                error
}

// convertToDurable runs the upgrade up to and including the commit point.
func (s *CollectionStore) convertToDurable(canonical string, old *Collection, cfg, walCfg Config, cfgPath, snapPath string) upgradeResult {
	fail := func(err error) upgradeResult { return upgradeResult{oldManifestCertain: true, err: err} }
	h, ok := old.idx.(*hnsw)
	if cfg.IndexType != IndexHNSW || !ok {
		return fail(fmt.Errorf("vector: only an HNSW collection can be upgraded (index type %v)", cfg.IndexType))
	}
	fresh, err := NewCollection(canonical, walCfg)
	if err != nil {
		return fail(err)
	}
	// The fresh collection is heap-backed and has no log; once its snapshot is
	// written the ordinary load rebuilds it from that, so it is always dropped.
	defer func() { _ = fresh.Close() }()
	if err := copyLivePoints(h, fresh); err != nil {
		return fail(err)
	}
	if err := upgradeStep("loaded"); err != nil {
		return fail(err)
	}
	if err := s.writeSnapshotFile(fresh, snapPath); err != nil {
		return fail(fmt.Errorf("vector: writing the snapshot: %w", err))
	}
	if err := upgradeStep("snapshot"); err != nil {
		return fail(err)
	}
	if err := s.verifySnapshot(canonical, walCfg, snapPath, fresh.Stats().Size); err != nil {
		return fail(err)
	}
	// A log left from an earlier life would replay over the snapshot. It is
	// moved aside (or, when empty, removed), durably, before the manifest can
	// name the log path again.
	_, _, _, walPath := s.persistPaths(canonical)
	if err := setStaleLogAside(canonical, walPath); err != nil {
		return fail(err)
	}
	if err := syncDirf(filepath.Dir(walPath)); err != nil {
		return fail(fmt.Errorf("vector: fsync dir %s: %w", filepath.Dir(walPath), err))
	}
	if err := upgradeStep("log-removed"); err != nil {
		return fail(err)
	}
	werr := writeConfig(cfgPath, walCfg)
	if werr == nil {
		werr = upgradeStep("manifest")
	}
	if werr == nil {
		return upgradeResult{committed: true, manifestDurable: true}
	}
	// writeConfig can fail after its rename has landed (the directory fsync), so
	// what the manifest says is read back.
	got, rerr := readConfig(cfgPath)
	switch {
	case rerr == nil && got.WAL:
		// Committed: keeping the old form would accept writes the next open,
		// which takes the log path, cannot see. The failed directory sync may be
		// the one that made the rename durable, so it is retried; until it
		// succeeds the old files stay, since removing them could reach disk
		// before the rename does.
		slog.Warn("vector: the upgraded manifest reported an error but is in place; continuing with the write-ahead log",
			"collection", canonical, "err", werr)
		derr := syncDirf(filepath.Dir(cfgPath))
		return upgradeResult{committed: true, manifestDurable: derr == nil}
	case rerr == nil:
		return fail(fmt.Errorf("vector: replacing the manifest: %w", werr))
	}
	// Unreadable: write the old manifest again so the state is certain.
	if perr := writeConfig(cfgPath, cfg); perr != nil {
		return upgradeResult{err: fmt.Errorf("vector: replacing the manifest: %w; reading it back: %v; putting the old one back: %v", werr, rerr, perr)}
	}
	return fail(fmt.Errorf("vector: replacing the manifest: %w; reading it back: %v", werr, rerr))
}

// setStaleLogAside clears the log path of a collection being upgraded. A log
// there is normally left from an earlier life, but it can hold acknowledged
// writes: a log-form manifest whose directory sync failed, then a power loss,
// leaves the old manifest in force over a log that was written to. So a
// non-empty one is never deleted: it is renamed to <log>.stale-<unixnano>, which
// no open reads, and a warning names it. An empty one is removed. The caller
// syncs the directory.
func setStaleLogAside(canonical, walPath string) error {
	st, err := os.Lstat(walPath)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("vector: checking for a stale log: %w", err)
	}
	if st.Size() == 0 {
		if err := os.Remove(walPath); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("vector: removing an empty stale log: %w", err)
		}
		return nil
	}
	kept := fmt.Sprintf("%s.stale-%d", walPath, time.Now().UnixNano())
	if err := os.Rename(walPath, kept); err != nil {
		return fmt.Errorf("vector: moving a stale log aside: %w", err)
	}
	slog.Warn("vector: a durable collection being upgraded had a log beside its old manifest; it was moved aside, not replayed, and may hold writes made before an earlier interrupted upgrade",
		"collection", canonical, "kept", kept, "bytes", st.Size())
	return nil
}

// verifySnapshot proves the snapshot just written restores, with the config the
// log form will load with, to want points.
func (s *CollectionStore) verifySnapshot(canonical string, walCfg Config, snapPath string, want int) error {
	got, err := s.snapshotPoints(canonical, walCfg, snapPath)
	if err != nil {
		return fmt.Errorf("vector: the written snapshot does not restore: %w", err)
	}
	if got != want {
		return fmt.Errorf("vector: the written snapshot restores %d points, want %d", got, want)
	}
	return nil
}

// snapshotPoints restores the snapshot at snapPath into a scratch collection
// built with the log form's config and returns how many points it holds.
func (s *CollectionStore) snapshotPoints(canonical string, walCfg Config, snapPath string) (int, error) {
	scratch, err := NewCollection(canonical, s.effectiveConfig(canonical, walCfg))
	if err != nil {
		return 0, err
	}
	defer func() { _ = scratch.Close() }()
	f, err := os.Open(snapPath) //nolint:gosec // store-managed path
	if err != nil {
		return 0, err
	}
	rerr := scratch.Restore(f)
	_ = f.Close()
	if rerr != nil {
		return 0, rerr
	}
	return scratch.Stats().Size, nil
}

// copyLivePoints copies every live point of src into dst with its version and
// per-key deadlines verbatim. The stored data of a store once stopped uncleanly
// can be inconsistent; a panic while reading it is returned as an error, so the
// upgrade fails and the store still opens.
func copyLivePoints(src *hnsw, dst *Collection) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("vector: reading the stored points: %v", r)
		}
	}()
	return src.forEachLive(func(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error {
		if err := dst.RestoreInsert(id, vec, ttl, meta, sparse, keyExpires, version); err != nil {
			return fmt.Errorf("vector: copying point %d: %w", id, err)
		}
		return nil
	})
}

// forEachLive calls fn, in slot order, for every point that is neither deleted
// nor expired, with copies the caller owns: the stored vector, the remaining
// point TTL (0 = none), the metadata, the sparse vector, the per-key deadlines
// and the version. It never reads graph edges. It stops at fn's first error. A
// point whose slot lies outside the stored data, or whose vector cannot be read,
// is an error.
func (h *hnsw) forEachLive(fn func(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.arena.poisoned.Load() {
		return ErrIndexPoisoned
	}
	type point struct {
		id   uint64
		slot uint32
	}
	points := make([]point, 0, len(h.arena.idMap))
	for id, slot := range h.arena.idMap {
		points = append(points, point{id, slot})
	}
	sort.Slice(points, func(i, j int) bool { return points[i].slot < points[j].slot })
	slots := h.arena.Capacity()
	now := uint64(h.now()) //nolint:gosec // unix millis are non-negative
	for _, p := range points {
		if int(p.slot) >= slots {
			return fmt.Errorf("vector: point %d names slot %d of %d", p.id, p.slot, slots)
		}
		if h.tombstoned[p.slot] || h.isExpiredAt(p.slot, now) {
			continue
		}
		stored := h.vecFor(p.slot)
		if len(stored) != h.cfg.Dim {
			return fmt.Errorf("vector: point %d has no readable vector", p.id)
		}
		var ttl time.Duration
		if exp := h.arena.ExpiresAt(p.slot); exp != 0 {
			ttl = time.Duration(exp-now) * time.Millisecond // exp > now: expired points were skipped
		}
		var sparse *SparseVector
		if sv := h.arena.Sparse(p.slot); sv != nil {
			sparse = sv.Clone()
		}
		if err := fn(p.id, append([]float32(nil), stored...), ttl, cloneMeta(h.arena.Metadata(p.slot)),
			sparse, maps.Clone(h.arena.KeyExpires(p.slot)), h.arena.Version(p.slot)); err != nil {
			return err
		}
	}
	return nil
}
