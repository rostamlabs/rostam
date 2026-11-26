// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
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
// That collection is written as a synced snapshot, any stale log is removed, and
// then the manifest is replaced: the commit point. Until then nothing the old
// form reads has been touched, so any failure leaves it registered and working,
// warns that the collection is not crash-safe, and returns nil; the next open
// tries again. After the commit the old collection is closed and the collection
// loads down the ordinary log path, whose error, if any, is returned.
func (s *CollectionStore) upgradeToDurable(canonical string, cfg Config, cfgPath, snapPath string) error {
	old := s.collections[canonical]
	walCfg := s.durableConfig(canonical, cfg)
	committed, manifestTried, err := s.convertToDurable(canonical, old, cfg, walCfg, cfgPath, snapPath)
	if !committed {
		// The snapshot is only removed when the manifest was never attempted: a
		// manifest that could not be read back may still have been replaced, and
		// then the snapshot is the collection.
		if !manifestTried {
			_ = os.Remove(snapPath)
		}
		slog.Warn("vector: durable collection is not crash-safe: upgrading it to the write-ahead log failed, so it stays in its old form until a later open succeeds",
			"collection", canonical, "err", err)
		return nil
	}
	delete(s.collections, canonical)
	if cerr := old.Close(); cerr != nil {
		slog.Warn("vector: closing the old form of an upgraded collection failed",
			"collection", canonical, "err", cerr)
	}
	return s.loadWALCollection(canonical, walCfg, cfgPath, snapPath)
}

// convertToDurable runs the upgrade up to and including the commit point.
// committed reports whether the manifest now says WAL; manifestTried whether
// replacing it was attempted.
func (s *CollectionStore) convertToDurable(canonical string, old *Collection, cfg, walCfg Config, cfgPath, snapPath string) (committed, manifestTried bool, err error) {
	h, ok := old.idx.(*hnsw)
	if cfg.IndexType != IndexHNSW || !ok {
		return false, false, fmt.Errorf("vector: only an HNSW collection can be upgraded (index type %v)", cfg.IndexType)
	}
	fresh, err := NewCollection(canonical, walCfg)
	if err != nil {
		return false, false, err
	}
	// The fresh collection is heap-backed and has no log; once its snapshot is
	// written the ordinary load rebuilds it from that, so it is always dropped.
	defer func() { _ = fresh.Close() }()
	if err := copyLivePoints(h, fresh); err != nil {
		return false, false, err
	}
	if err := upgradeStep("loaded"); err != nil {
		return false, false, err
	}
	if err := s.writeSnapshotFile(fresh, snapPath); err != nil {
		return false, false, fmt.Errorf("vector: writing the snapshot: %w", err)
	}
	if err := upgradeStep("snapshot"); err != nil {
		return false, false, err
	}
	// A log left from an earlier life would replay over the snapshot. Its removal
	// is made durable before the manifest can name the log path again.
	_, _, _, walPath := s.persistPaths(canonical)
	if err := os.Remove(walPath); err != nil && !os.IsNotExist(err) {
		return false, false, fmt.Errorf("vector: removing a stale log: %w", err)
	}
	if err := syncDirf(filepath.Dir(walPath)); err != nil {
		return false, false, fmt.Errorf("vector: fsync dir %s: %w", filepath.Dir(walPath), err)
	}
	if err := upgradeStep("log-removed"); err != nil {
		return false, false, err
	}
	werr := writeConfig(cfgPath, walCfg)
	if werr == nil {
		werr = upgradeStep("manifest")
	}
	if werr != nil {
		// writeConfig can fail after its rename has landed (the directory fsync).
		// Then the upgrade has committed, and keeping the old form would accept
		// writes the next open, which takes the log path, cannot see.
		got, rerr := readConfig(cfgPath)
		if rerr != nil || !got.WAL {
			return false, true, fmt.Errorf("vector: replacing the manifest: %w", werr)
		}
		slog.Warn("vector: the upgraded manifest reported an error but is in place; continuing with the write-ahead log",
			"collection", canonical, "err", werr)
	}
	return true, true, nil
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
