// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path"
	"sort"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// Cold-tier object key layout mirrors the backup package
// (<tenant>/<escaped-canonical>/<ts RFC3339>.snap + sibling .cfg.json) so a cold
// collection's object IS a normal backup object and either path can restore it.
// Kept self-contained here to avoid a vector→backup import cycle.
const (
	coldSnapExt = ".snap"
	coldCfgExt  = ".cfg.json"
	coldTsLayut = time.RFC3339
)

// coldSnapshotKey is <tenant>/<escaped-canonical>/<ts>.snap.
func coldSnapshotKey(tenant, canonical string, ts time.Time) string {
	return path.Join(tenant, url.PathEscape(canonical)) + "/" + ts.UTC().Format(coldTsLayut) + coldSnapExt
}

// coldCfgKey maps a snapshot key to its sibling config key (.snap → .cfg.json).
func coldCfgKey(snapKey string) string {
	return snapKey[:len(snapKey)-len(coldSnapExt)] + coldCfgExt
}

// marshalConfigJSON serializes a Config the same way the on-disk sidecar and the
// backup package do (encoding/json), so all three round-trip identically.
func marshalConfigJSON(cfg Config) ([]byte, error) { return json.Marshal(cfg) }

func sortStrings(s []string) { sort.Strings(s) }

func joinErrs(errs []error) error { return errors.Join(errs...) }

// Cold tiering (single-node): a CollectionStore can EVICT an idle collection's
// resident data to an object store, keeping only a lightweight STUB in the
// catalog (its name + Config + snapshot key + cold flag), then LAZILY RESTORE it
// on the next access. This trades object-store latency on the first touch of a
// cold collection for the resident RAM/mmap of every idle collection — the
// many-collections-mostly-idle single-node profile.
//
// Zero-overhead when OFF: a store with no cold entries and no lastAccess tracking
// requested behaves byte-identically to one without this file. The hot path adds
// only a single map-empty check (s.cold == nil || len == 0) inside Acquire; the
// lastAccess map is touched ONLY once a collection has ever been evicted (the maps
// are lazily allocated on the first EvictCollection). Nothing here reads
// time.Now(): evict takes a caller-supplied timestamp and the idle sweeper takes a
// caller-supplied `now`.

// coldEntry is the catalog stub for an evicted collection. It holds everything
// needed to (a) keep the collection listed and config-introspectable while cold
// and (b) lazily restore it: the original Config, the object store + snapshot key
// to pull from, and a per-entry single-flight guard so N concurrent accesses
// fetch the snapshot exactly ONCE.
type coldEntry struct {
	cfg    Config
	obj    objstore.ObjectStore
	key    string
	tenant string

	// promoteMu serializes promotion of THIS entry: the first accessor restores
	// the collection while the rest block, then all observe it live (single-flight).
	promoteMu sync.Mutex
}

// SetClock injects the clock the store uses to stamp per-collection lastAccess on
// resolve (Acquire) and on promote. The engine itself NEVER calls time.Now
// (determinism); pass a real clock from the cmd layer (func() time.Time {
// return time.Now() }) or a fake from tests. Passing nil disables on-resolve
// lastAccess updates entirely (the idle sweeper is then driven purely by the
// timestamps handed to EvictCollection/SweepCold).
func (s *CollectionStore) SetClock(now func() time.Time) {
	s.mu.Lock()
	s.nowFn = now
	s.mu.Unlock()
}

// EvictCollection backs the named collection up to obj (snapshot + sibling
// config) and then releases its resident in-memory/mmap data, leaving a
// lightweight cold STUB in the catalog so ListCollections/CollectionNames still
// report it and its Config survives. A later access lazily restores it (see the
// promote hook in Acquire).
//
// ts is the caller-supplied backup timestamp (the package never reads time.Now);
// it versions the snapshot object key, exactly like backup.BackupOpts.Timestamp.
// tenant is the object-key namespace prefix.
//
// Evicting an already-cold collection is a no-op (nil). Evicting an unknown
// collection returns an error. The backup is taken and verified BEFORE any
// resident data is dropped, so a backup failure leaves the collection hot and
// intact (recoverable).
func (s *CollectionStore) EvictCollection(ctx context.Context, name string, obj objstore.ObjectStore, tenant string, ts time.Time) error {
	canonical, err := canonicalName(name)
	if err != nil {
		return err
	}

	s.mu.RLock()
	_, isCold := s.cold[canonical]
	_, isHot := s.collections[canonical]
	s.mu.RUnlock()
	if isCold {
		return nil // already cold — no-op
	}
	if !isHot {
		// Match the wording the rest of collections.go uses for a missing collection
		// ("vector: no collection %q"), so httpapi's statusForError maps it to 404
		// (its "no collection" bucket) rather than falling through to 500.
		return fmt.Errorf("vector: evict %q: no collection %q", canonical, canonical)
	}

	// Pin the collection and snapshot its config + data to the object store BEFORE
	// dropping anything. We mirror backup.backupOne's key layout (so a cold
	// collection's object is indistinguishable from a normal backup and can be
	// restored by either path) but keep this self-contained to avoid a vector→backup
	// import cycle.
	c, ok := s.Acquire(canonical)
	if !ok {
		return fmt.Errorf("vector: evict %q: collection vanished", canonical)
	}
	cfg := c.Config()
	key := coldSnapshotKey(tenant, canonical, ts)
	if err := s.putColdSnapshot(ctx, c, obj, key, cfg); err != nil {
		c.Release()
		return fmt.Errorf("vector: evict %q: %w", canonical, err)
	}
	c.Release()

	if evictCommitHook != nil {
		evictCommitHook()
	}

	// Register the stub and remove the live collection under the store lock, then
	// retire the index out of the lock (drain in-flight users, close/unmap — the
	// REAL memory release). A racing access between Acquire above and the lock here
	// is harmless: it sees the collection still hot and serves normally; the next
	// access sees the stub and lazily restores.
	//
	// The commit holds the name lock, so it cannot interleave with a drop, create,
	// promotion or restore of this name.
	defer s.lockName(canonical)()
	s.mu.Lock()
	live, stillHot := s.collections[canonical]
	if !stillHot || live != c {
		// Dropped out from under us between the Acquire and here — or dropped and
		// replaced (a restore, or a drop and create). The snapshot above is of a
		// collection that is no longer this name's, so committing it would put a
		// stub for the OLD data in place of the current collection. Nothing to
		// evict; the snapshot we wrote is just an extra backup. Not an error.
		s.mu.Unlock()
		return nil
	}
	if s.cold == nil {
		s.cold = make(map[string]*coldEntry)
	}
	s.cold[canonical] = &coldEntry{cfg: cfg, obj: obj, key: key, tenant: tenant}
	delete(s.collections, canonical)
	// No lastAccess cleanup needed: it lives on the evicted Collection object (now
	// removed from s.collections, which the sweeper scans), and a promote rebuilds a
	// fresh Collection and re-seeds its lastAccess.
	s.mu.Unlock()

	// Out of the lock: drain Acquire holders, stop the sweeper, close/unmap the
	// index. This is the actual resident-memory release — the stub holds no
	// arena/graph. We do NOT delete the on-disk files: the snapshot in the object
	// store is authoritative, and a promote rebuilds the index from it.
	live.retire(nil)
	return nil
}

// evictCommitHook, when non-nil, runs in EvictCollection between releasing the
// collection it snapshotted and committing the eviction — the gap in which the
// name can be dropped and replaced. Tests only; nil in production.
var evictCommitHook func()

// putColdSnapshot streams c's snapshot to obj under key and writes the sibling
// config object, identical in layout to the backup package (a cold collection's
// object IS a backup). It snapshots straight to the object store via a pipe so no
// temp file or full-buffer is needed for the engine path; MemStore buffers in
// memory anyway, and the S3 client streams the pipe.
func (s *CollectionStore) putColdSnapshot(ctx context.Context, c *Collection, obj objstore.ObjectStore, key string, cfg Config) error {
	// Snapshot into a buffer: Put needs the exact size for Content-Length and the
	// snapshot stream isn't seekable, so buffer it. Cold eviction is an
	// idle-collection, off-hot-path operation — the transient buffer is acceptable;
	// a streaming/temp-file variant is a follow-up.
	var buf bytes.Buffer
	if err := c.Snapshot(&buf); err != nil {
		return fmt.Errorf("snapshot: %w", err)
	}
	if err := obj.Put(ctx, key, bytes.NewReader(buf.Bytes()), int64(buf.Len())); err != nil {
		return fmt.Errorf("put %q: %w", key, err)
	}
	cfgData, err := marshalConfigJSON(cfg)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	if err := obj.Put(ctx, coldCfgKey(key), bytes.NewReader(cfgData), int64(len(cfgData))); err != nil {
		return fmt.Errorf("put %q: %w", coldCfgKey(key), err)
	}
	return nil
}

// promoteCold restores a cold entry's collection from its snapshot object and
// marks it hot, under a per-entry single-flight so N concurrent accesses fetch
// the snapshot exactly ONCE (the first restores; the rest block on promoteMu and
// then observe it already hot). A failed restore returns a clear error and leaves
// the stub intact, so the collection stays cold-but-recoverable.
//
// It seeds the freshly-hot collection's lastAccess from the injected clock (zero
// time if none — no time.Now in the engine).
func (s *CollectionStore) promoteCold(canonical string, e *coldEntry) error {
	// The whole promotion holds the name lock: it writes the collection's files and,
	// if a drop won the race, deletes them by path again — which must not reach
	// files a later operation on the name has published. Taken before promoteMu,
	// which nothing else takes.
	defer s.lockName(canonical)()
	e.promoteMu.Lock()
	defer e.promoteMu.Unlock()

	// Re-check under the single-flight lock: a prior holder may have already
	// promoted it (single-flight collapse) — then there is nothing to do.
	s.mu.RLock()
	_, stillCold := s.cold[canonical]
	s.mu.RUnlock()
	if !stillCold {
		return nil
	}

	rc, err := e.obj.Get(context.Background(), e.key)
	if err != nil {
		return fmt.Errorf("vector: promote %q: get %q: %w", canonical, e.key, err)
	}
	defer func() { _ = rc.Close() }()

	// Build and fully restore the collection OFF-CATALOG (it is not in
	// s.collections yet), so a concurrent Acquire — which finds only the cold stub
	// and blocks on this same promoteMu — can never observe a half-restored index
	// (that would be a read/write data race against this Restore). Only after the
	// restore completes do we publish it atomically (swap stub→live under s.mu).
	c, err := s.buildCollection(canonical, e.cfg)
	if err != nil {
		return fmt.Errorf("vector: promote %q: build: %w", canonical, err)
	}
	if rerr := c.Restore(rc); rerr != nil {
		// Discard the half-built collection; the stub stays in place, so the
		// collection remains cold-but-recoverable on a later access.
		c.Stop()
		_ = c.Close()
		return fmt.Errorf("vector: promote %q: restore: %w", canonical, rerr)
	}

	// Publish: the collection is fully restored. Re-check the stub is STILL present
	// under the write lock before swapping it in. A concurrent DropCollection deletes
	// the cold stub under s.mu and returns success; if it interleaved between the
	// RLock check above and this Lock, the collection has been dropped and publishing
	// our rebuilt copy here would RESURRECT it (and leak the .cfg.json/mmap files
	// buildCollection just wrote). If the stub is gone, discard the rebuilt
	// collection: tear it down exactly as DropCollection does (retire drains, stops,
	// and closes/unmaps) and remove the on-disk files buildCollection created, so no
	// index or files leak.
	s.mu.Lock()
	if _, stillCold := s.cold[canonical]; !stillCold {
		s.mu.Unlock()
		// A concurrent Drop (or another promoter) won. Discard the rebuilt collection
		// and clean up its on-disk files, mirroring DropCollection's teardown.
		cfgPath, snapPath := s.collectionPath(canonical)
		vecsPath, graphPath, metaPath, walPath := s.persistPaths(canonical)
		colCfg := c.cfg
		c.retire(func() {
			_ = os.Remove(cfgPath)
			_ = os.Remove(snapPath)
			_ = os.Remove(vecsPath)
			_ = os.Remove(graphPath)
			_ = os.Remove(metaPath)
			_ = os.Remove(walPath)
			removeClusterMmapFiles(colCfg)
		})
		return nil
	}
	// Swap stub→live atomically and seed lastAccess from the injected clock (zero
	// time if none — no time.Now here). The store is a lock-free atomic on the newly
	// built Collection; the sweeper reads the same field.
	s.collections[canonical] = c
	delete(s.cold, canonical)
	var now time.Time
	if s.nowFn != nil {
		now = s.nowFn()
	}
	c.lastAccess.Store(now.UnixNano())
	s.mu.Unlock()
	return nil
}

// SweepCold evicts every hot collection whose last access is strictly older than
// now-idle, using obj/tenant for the backup. It is the idle-eviction policy: the
// cmd-layer driver calls it on a ticker with the wall clock; here it is a
// pure function of the injected `now` (no time.Now in the engine). idle <= 0
// disables sweeping (returns immediately). A collection that has never been
// accessed (no lastAccess entry) is treated as accessed "now" on first sight and
// NOT evicted, so a freshly-created collection isn't swept before it is ever used.
//
// It returns the canonical names it evicted (sorted) and a joined error of any
// per-collection eviction failures (one failure does not abort the rest).
func (s *CollectionStore) SweepCold(now time.Time, idle time.Duration, obj objstore.ObjectStore, tenant string) ([]string, error) {
	if idle <= 0 {
		return nil, nil
	}
	cutoff := now.Add(-idle)

	// Snapshot the candidate set under the lock: hot collections whose lastAccess
	// is older than the cutoff. Seed lastAccess for any hot collection we have never
	// seen (lastAccess == 0), so it gets a full idle window before becoming eligible.
	s.mu.Lock()
	var candidates []string
	for name, c := range s.collections {
		last := c.lastAccess.Load()
		if last == 0 {
			c.lastAccess.Store(now.UnixNano()) // first sight: start its idle clock now
			continue
		}
		if time.Unix(0, last).Before(cutoff) {
			candidates = append(candidates, name)
		}
	}
	s.mu.Unlock()

	sortStrings(candidates)
	var evicted []string
	var errs []error
	for _, name := range candidates {
		if err := s.EvictCollection(context.Background(), name, obj, tenant, now); err != nil {
			errs = append(errs, err)
			continue
		}
		evicted = append(evicted, name)
	}
	return evicted, joinErrs(errs)
}

// IsCold reports whether the named collection is currently an evicted stub. Used
// by tests and the admin/ops surface.
func (s *CollectionStore) IsCold(name string) bool {
	canonical, err := canonicalName(name)
	if err != nil {
		return false
	}
	s.mu.RLock()
	_, ok := s.cold[canonical]
	s.mu.RUnlock()
	return ok
}

// ColdNames returns the canonical names of every collection currently evicted
// (cold stub), sorted. Empty when cold tiering is unused.
func (s *CollectionStore) ColdNames() []string {
	s.mu.RLock()
	names := make([]string, 0, len(s.cold))
	for name := range s.cold {
		names = append(names, name)
	}
	s.mu.RUnlock()
	sortStrings(names)
	return names
}
