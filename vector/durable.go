// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
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
