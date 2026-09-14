// SPDX-License-Identifier: Apache-2.0

// Package backup streams each live collection's versioned snapshot to a
// pluggable object store and restores them back, with optional per-collection
// retention.
//
// It talks ONLY to the objstore.ObjectStore interface (Put/Get/List/Delete over
// string keys), so the same Backup/Restore logic drives the in-memory MemStore
// fake (tests), a local-filesystem FSObjectStore (demoable today), or the
// zero-dep SigV4 S3 client — no cloud SDK is a dependency of this package.
//
// Determinism: the package NEVER calls time.Now(). The per-run timestamp that
// versions every object key is supplied by the caller via BackupOpts.Timestamp,
// so a test can pin distinct stamps and the cmd-layer driver passes the
// real wall clock. The key layout — and therefore retention pruning — is fully
// reproducible: the same Timestamp always yields the same key.
//
// Key layout: each backup writes one object per collection at
//
//	<Tenant>/<escaped-collection>/<ts.UTC().Format(RFC3339)>.snap
//
// The collection name is canonical ("tenant/name") and so contains a "/"; it is
// url.PathEscape'd into a single key segment so it never spawns spurious key
// levels (and is reversible). All of one collection's snapshots share the prefix
// <Tenant>/<escaped-collection>/, which List+lexical-sort makes the retention
// unit — RFC3339 stamps sort chronologically, so a descending key sort is a
// newest-first ordering.
package backup

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/vector"
)

// snapExt is the extension every snapshot object key ends with. It is purely
// cosmetic (human-identifiable bucket listings) and mirrors the on-disk
// <name>.snap convention CollectionStore uses.
const snapExt = ".snap"

// cfgExt is the extension of the sibling config object written next to every
// snapshot: <ts>.cfg.json holds the collection's vector.Config serialized as
// JSON. It lets Restore re-create the collection with its EXACT original config
// (quantization, IndexType, Vamana geometry) — settings the bare snapshot stream
// does NOT carry — so a restore onto a fresh store is config-faithful.
const cfgExt = ".cfg.json"

// tsLayout is the timestamp format embedded in every object key. RFC3339 is
// lexicographically sortable in time order (given UTC, fixed-width fields), so
// retention can keep the newest N by a plain descending key sort.
const tsLayout = time.RFC3339

// ErrSnapshotExists is reported (wrapped, in BackupResult.Err) when a run's
// snapshot key already exists in the store, or another run — concurrent, or one
// that died mid-backup — has already claimed its timestamp. Snapshot keys are
// WRITE-ONCE: a backup never overwrites an existing backup, and of concurrent
// runs at one timestamp exactly one publishes. See backupOne for why.
var ErrSnapshotExists = errors.New("backup: snapshot key already exists")

// BackupOpts configures a single Backup run.
type BackupOpts struct {
	// Tenant is the top-level key prefix every object is written under (e.g.
	// "default" or "acme-prod"). It is the BACKUP namespace, independent of a
	// collection's own canonical tenant. Empty means no prefix.
	Tenant string
	// Timestamp versions this run's objects: each key embeds
	// Timestamp.UTC().Format(RFC3339). It is supplied by the caller (the package
	// never calls time.Now) so the layout is deterministic and testable — the
	// SAME Timestamp always produces the SAME key.
	Timestamp time.Time
	// Retention keeps only the newest N snapshots per collection, deleting older
	// ones AFTER the current run's object is written. 0 (the default) keeps all
	// snapshots — no pruning. The just-written newest object is NEVER deleted.
	Retention int
}

// BackupResult reports the outcome of backing up one collection.
type BackupResult struct {
	// Collection is the (canonical) collection name this result is for.
	Collection string
	// Key is the object key the snapshot was written to. Empty if Err != nil and
	// the failure occurred before/at the Put.
	Key string
	// Size is the snapshot's byte size as written. 0 on failure.
	Size int64
	// Err is non-nil if this collection's backup (snapshot, put, or prune)
	// failed. A failure here does NOT abort the other collections.
	Err error
}

// validateTenant rejects a backup-namespace tenant that could break out of the
// key prefix. The collection segment is PathEscape'd in collectionKeyPrefix, but
// the tenant is joined raw, so an unvalidated tenant like "../../evil" or "/abs"
// would place every object outside the configured prefix (and, combined with a
// filesystem store, outside the root) and mis-scope the retention List/prune. We
// reject empty, absolute, "."/".." and any tenant containing a path separator or
// a NUL byte. A normal tenant ("default", "acme-prod") passes unchanged, so the
// key layout for every valid tenant is byte-identical to before.
func validateTenant(tenant string) error {
	// Empty is the documented "no prefix" case (BackupOpts.Tenant) — allowed.
	if tenant == "" {
		return nil
	}
	if tenant == "." || tenant == ".." {
		return fmt.Errorf("backup: invalid tenant %q", tenant)
	}
	if strings.HasPrefix(tenant, "/") || strings.HasPrefix(tenant, ".") {
		return fmt.Errorf("backup: invalid tenant %q (must not start with '/' or '.')", tenant)
	}
	if strings.ContainsAny(tenant, "/\\\x00") {
		return fmt.Errorf("backup: invalid tenant %q (must not contain a path separator or NUL)", tenant)
	}
	return nil
}

// collectionKeyPrefix returns the per-collection key prefix
// <tenant>/<escaped-collection>/ that scopes both the snapshot key and the
// retention List. The collection name is PathEscape'd so a canonical
// "tenant/name" stays a single key segment (and a sibling collection whose name
// is a byte-prefix of another can never be swept by retention). The tenant is
// validated by the exported entry points (Backup/Restore/LatestKey) before this
// is reached, so it is a single safe key segment here.
func collectionKeyPrefix(tenant, collection string) string {
	return path.Join(tenant, url.PathEscape(collection)) + "/"
}

// snapshotKey is collectionKeyPrefix + "<rfc3339>.snap".
func snapshotKey(tenant, collection string, ts time.Time) string {
	return collectionKeyPrefix(tenant, collection) + ts.UTC().Format(tsLayout) + snapExt
}

// cfgKeyFor maps a snapshot key to its sibling config-object key by swapping the
// .snap suffix for .cfg.json. The two share the same <tenant>/<col>/<ts> stem so
// a restore can derive one from the other with no extra bookkeeping.
func cfgKeyFor(snapKey string) string {
	return strings.TrimSuffix(snapKey, snapExt) + cfgExt
}

// Backup streams every live dense collection in store to obj, one object per
// collection at <Tenant>/<escaped-collection>/<ts>.snap, then applies retention.
//
// For each collection it snapshots to a temp file (reusing CollectionStore's
// temp+fsync+remove discipline — the snapshot's exact size is then known for the
// object store's Content-Length), Puts the file, removes the temp, and (if
// Retention > 0) prunes older snapshots beyond the newest N.
//
// Errors are isolated per collection: a failure on one (snapshot, put, or prune)
// is captured in that collection's BackupResult.Err and the run CONTINUES with
// the rest, so one bad collection never silently skips the others. The full
// per-collection slice is always returned; the second return value is the joined
// error (nil iff every collection succeeded).
func Backup(ctx context.Context, store *vector.CollectionStore, obj objstore.ObjectStore, opts BackupOpts) ([]BackupResult, error) {
	if err := validateTenant(opts.Tenant); err != nil {
		return nil, err
	}
	names := store.CollectionNames()
	sort.Strings(names) // deterministic result ordering
	results := make([]BackupResult, 0, len(names))
	var errs []error
	for _, name := range names {
		res := backupOne(ctx, store, obj, name, opts)
		results = append(results, res)
		if res.Err != nil {
			errs = append(errs, res.Err)
		}
	}
	return results, errors.Join(errs...)
}

// backupOne snapshots a single collection to a temp file and Puts it under its
// timestamped key, then prunes per retention. All failures are returned in the
// BackupResult (never panics, never aborts the caller's loop).
func backupOne(ctx context.Context, store *vector.CollectionStore, obj objstore.ObjectStore, name string, opts BackupOpts) (res BackupResult) {
	res = BackupResult{Collection: name}

	c, ok := store.Acquire(name)
	if !ok {
		// Raced with a Drop between CollectionNames and Acquire — skip silently;
		// the next run reflects the new set. Not an error.
		return res
	}
	defer c.Release()

	key := snapshotKey(opts.Tenant, name, opts.Timestamp)
	cfgKey := cfgKeyFor(key)

	// Snapshot keys are WRITE-ONCE. The two objects of a backup (config, snapshot)
	// are independent writes, and no ordering of two independent writes can keep
	// an EXISTING pair consistent through a re-run at the same timestamp: whichever
	// write fails second leaves the surviving old object paired with a new one.
	// Refusing to touch an already-claimed timestamp removes the overwrite case
	// entirely, so the config-first ordering below is a TOTAL pairing guarantee,
	// not just a fresh-key one. It also turns an accidental same-second collision
	// (tsLayout is second-granularity) into a clean error instead of a silently
	// overwritten backup.
	//
	// The refusal must be ATOMIC, not a probe followed by writes: two runs at one
	// timestamp — an admin backup landing in the ticker's second, or two processes
	// backing up the same prefix — would both see the key absent and both publish.
	// So the claim is the first write itself, made with PutIfAbsent on the CONFIG
	// key: exactly one run creates it, and every other run fails there, before it
	// has written anything. The snapshot then goes second, also with PutIfAbsent.
	//
	// Sibling config object FIRST, snapshot SECOND. The ordering is load-bearing:
	// LatestKey/prune key off the <ts>.snap object, so writing the config before
	// the snapshot means every snapshot that can ever be selected as "latest"
	// already has its sibling config present. The snapshot stream does NOT carry
	// quantization / IndexType / Vamana geometry / FullText, so a config-less
	// restore of such a collection is silently degraded (wrong metric/geometry, or
	// an empty BM25 index) or fails after already dropping the target. Were the
	// snapshot written first, an interruption between the two writes would leave a
	// selectable snapshot with no config; this order can only ever leave an orphan
	// config with no snapshot, which LatestKey/prune ignore (they filter on .snap).
	//
	// Because the config is the claim, an orphan config HOLDS its timestamp: a
	// re-run at that exact timestamp is refused, since from outside it cannot be
	// told apart from a run still writing its snapshot. That is why the claim is
	// released (the config deleted) on every failure that certainly left no
	// snapshot of ours behind; only a crash, or a snapshot write whose outcome is
	// unknown, leaves an orphan. Later runs use a distinct timestamp key, so an
	// orphan never blocks the periodic backup; it persists until a manual sweep
	// (prune only deletes configs paired with a pruned snapshot).
	//
	// We reuse the same JSON marshal the store uses for its on-disk <col>.json
	// sidecar.
	cfgData, err := json.Marshal(c.Config())
	if err != nil {
		res.Err = fmt.Errorf("backup %q: marshal config: %w", name, err)
		return res
	}
	if err := obj.PutIfAbsent(ctx, cfgKey, strings.NewReader(string(cfgData)), int64(len(cfgData))); err != nil {
		if errors.Is(err, objstore.ErrExists) {
			res.Err = fmt.Errorf("backup %q: %q is already claimed by %q: %w", name, key, cfgKey, ErrSnapshotExists)
		} else {
			res.Err = fmt.Errorf("backup %q: put %q: %w", name, cfgKey, err)
		}
		return res
	}
	// claimed stays true only while the config may be paired with a snapshot of
	// ours; any return with it still set means the snapshot was certainly not
	// written, so the claim is released. The release runs under cleanupContext:
	// detached, so a cancelled run still releases, and bounded, so a stalled
	// store cannot hold the run (and a shutdown waiting on it) indefinitely.
	//
	// A release that fails is reported, not dropped. The config it leaves is at
	// best an orphan holding its timestamp and at worst — when the run stopped
	// because a snapshot with no config of its own already held the key — taken
	// by Restore as that snapshot's config, which it does not describe. Either
	// way it needs removing by hand, and only the error can say so.
	claimed := true
	defer func() {
		if !claimed || res.Err == nil {
			return
		}
		cctx, cancel := cleanupContext(ctx)
		defer cancel()
		if err := obj.Delete(cctx, cfgKey); err != nil && !errors.Is(err, objstore.ErrNotFound) {
			res.Err = errors.Join(res.Err, fmt.Errorf("backup %q: could not release the claimed config %q, remove it by hand: %w", name, cfgKey, err))
		}
	}()

	// Snapshot to a temp file so we know the exact byte size for Put's
	// Content-Length (the object store streams the file; it does not buffer the
	// whole snapshot in memory).
	tmp, err := os.CreateTemp("", "rostam-backup-*.snap")
	if err != nil {
		res.Err = fmt.Errorf("backup %q: temp file: %w", name, err)
		return res
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()

	if err := c.Snapshot(tmp); err != nil {
		_ = tmp.Close()
		res.Err = fmt.Errorf("backup %q: snapshot: %w", name, err)
		return res
	}
	size, err := tmp.Seek(0, 1) // io.SeekCurrent: bytes written == current offset
	if err != nil {
		_ = tmp.Close()
		res.Err = fmt.Errorf("backup %q: size: %w", name, err)
		return res
	}
	if _, err := tmp.Seek(0, 0); err != nil { // rewind for the Put read
		_ = tmp.Close()
		res.Err = fmt.Errorf("backup %q: rewind: %w", name, err)
		return res
	}

	if err := obj.PutIfAbsent(ctx, key, tmp, size); err != nil {
		_ = tmp.Close()
		switch {
		case errors.Is(err, objstore.ErrExists):
			// A snapshot with no config of its own already holds this key (our
			// config claim succeeded, so no run of this version wrote it). Leaving
			// our config would pair it with that snapshot; the deferred release
			// removes it.
			res.Err = fmt.Errorf("backup %q: %q: %w", name, key, ErrSnapshotExists)
		case errors.Is(err, objstore.ErrConditionalWriteUnsupported):
			res.Err = fmt.Errorf("backup %q: put %q: %w", name, key, err)
		default:
			// The write may have landed (a timeout after the store committed), so
			// the config must stay: deleting it could leave a selectable snapshot
			// with no config.
			claimed = false
			res.Err = fmt.Errorf("backup %q: put %q: %w", name, key, err)
		}
		return res
	}
	claimed = false
	_ = tmp.Close()
	res.Key = key
	res.Size = size

	if opts.Retention > 0 {
		if err := prune(ctx, obj, collectionKeyPrefix(opts.Tenant, name), key, opts.Retention); err != nil {
			res.Err = fmt.Errorf("backup %q: prune: %w", name, err)
			return res
		}
	}
	return res
}

// prune enforces retention for one collection: it lists the collection's key
// prefix, sorts DESCENDING (newest first, since RFC3339 keys sort
// chronologically), keeps the newest retention, and deletes the rest. The
// just-written newest key is guaranteed kept: it is the lexicographically
// greatest for this run's timestamp and is never beyond index retention. As an
// extra safety belt it is explicitly skipped if ever encountered.
func prune(ctx context.Context, obj objstore.ObjectStore, prefix, newest string, retention int) error {
	infos, err := obj.List(ctx, prefix)
	if err != nil {
		return fmt.Errorf("list %q: %w", prefix, err)
	}
	// Retention is counted in SNAPSHOTS, not raw objects: every backup writes two
	// objects (a <ts>.snap and its <ts>.cfg.json sibling), so prune over .snap keys
	// only and delete each pruned snapshot's sibling config alongside it. (A .cfg.json
	// alone is never retained — it is bound to its snapshot's lifetime.)
	snaps := make([]string, 0, len(infos))
	for _, in := range infos {
		if strings.HasSuffix(in.Key, snapExt) {
			snaps = append(snaps, in.Key)
		}
	}
	if len(snaps) <= retention {
		return nil
	}
	sort.Sort(sort.Reverse(sort.StringSlice(snaps))) // newest first
	for _, key := range snaps[retention:] {
		if key == newest {
			// Never delete the snapshot we just wrote (defensive; it should be
			// within the kept window).
			continue
		}
		if err := obj.Delete(ctx, key); err != nil && !errors.Is(err, objstore.ErrNotFound) {
			return fmt.Errorf("delete %q: %w", key, err)
		}
		// Delete the sibling config object too (ErrNotFound tolerated — a backup
		// from before the .cfg.json era, or a partial run, has no sibling).
		cfgKey := cfgKeyFor(key)
		if err := obj.Delete(ctx, cfgKey); err != nil && !errors.Is(err, objstore.ErrNotFound) {
			return fmt.Errorf("delete %q: %w", cfgKey, err)
		}
	}
	return nil
}

// Restore pulls the object at key from obj and (re)creates the named collection
// from it via CollectionStore.RestoreCollection (create-or-replace). It is the
// inverse of Backup for a single, explicit key. tenant is accepted for symmetry
// with the key layout and future scoping; the key is authoritative.
func Restore(ctx context.Context, store *vector.CollectionStore, obj objstore.ObjectStore, tenant, collection, key string) error {
	if err := validateTenant(tenant); err != nil {
		return err
	}
	// Config-faithful restore: read the sibling <ts>.cfg.json (if present) and
	// re-create the collection with that EXACT Config before loading the snapshot,
	// so quantization / IndexType / Vamana geometry — none of which the bare
	// snapshot stream carries — are reconstructed correctly. A snapshot written
	// before the .cfg.json era (or whose sibling is missing) falls back to the
	// plain placeholder restore (a config-less HNSW of the recorded geometry).
	cfg, haveCfg, err := readConfigObject(ctx, obj, cfgKeyFor(key))
	if err != nil {
		return fmt.Errorf("restore %q: config %q: %w", collection, cfgKeyFor(key), err)
	}

	rc, err := obj.Get(ctx, key)
	if err != nil {
		return fmt.Errorf("restore %q: get %q: %w", collection, key, err)
	}
	defer func() { _ = rc.Close() }()

	if haveCfg {
		if err := store.RestoreCollectionWithConfig(collection, cfg, rc); err != nil {
			return fmt.Errorf("restore %q from %q: %w", collection, key, err)
		}
		return nil
	}
	if err := store.RestoreCollection(collection, rc); err != nil {
		return fmt.Errorf("restore %q from %q: %w", collection, key, err)
	}
	return nil
}

// readConfigObject fetches and decodes a sibling .cfg.json config object. It
// returns (cfg, true, nil) when the object exists and decodes, (Config{}, false,
// nil) when the object is absent (the pre-.cfg.json fallback), and a non-nil error
// only on a real Get/decode failure.
func readConfigObject(ctx context.Context, obj objstore.ObjectStore, cfgKey string) (vector.Config, bool, error) {
	rc, err := obj.Get(ctx, cfgKey)
	if err != nil {
		if errors.Is(err, objstore.ErrNotFound) {
			return vector.Config{}, false, nil
		}
		return vector.Config{}, false, err
	}
	defer func() { _ = rc.Close() }()
	data, err := io.ReadAll(rc)
	if err != nil {
		return vector.Config{}, false, err
	}
	var cfg vector.Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return vector.Config{}, false, fmt.Errorf("decode config: %w", err)
	}
	return cfg, true, nil
}

// LatestKey returns the newest snapshot key for a collection under tenant, or
// objstore.ErrNotFound if the collection has no snapshots. "Newest" is the
// lexicographically greatest key, which equals the most-recent RFC3339 timestamp.
func LatestKey(ctx context.Context, obj objstore.ObjectStore, tenant, collection string) (string, error) {
	if err := validateTenant(tenant); err != nil {
		return "", err
	}
	prefix := collectionKeyPrefix(tenant, collection)
	infos, err := obj.List(ctx, prefix)
	if err != nil {
		return "", fmt.Errorf("latest %q: list %q: %w", collection, prefix, err)
	}
	// Consider ONLY snapshot keys: the prefix also contains sibling .cfg.json
	// objects, and a newer run's .cfg.json must never be mistaken for the latest
	// snapshot. The greatest .snap key is the most-recent RFC3339 timestamp.
	latest := ""
	for _, in := range infos {
		if !strings.HasSuffix(in.Key, snapExt) {
			continue
		}
		if in.Key > latest {
			latest = in.Key
		}
	}
	if latest == "" {
		return "", objstore.ErrNotFound
	}
	return latest, nil
}

// RestoreLatest restores a collection from its newest snapshot under tenant. It
// is LatestKey + Restore, and returns objstore.ErrNotFound (wrapped) if no
// snapshot exists for the collection.
func RestoreLatest(ctx context.Context, store *vector.CollectionStore, obj objstore.ObjectStore, tenant, collection string) error {
	key, err := LatestKey(ctx, obj, tenant, collection)
	if err != nil {
		return err
	}
	return Restore(ctx, store, obj, tenant, collection, key)
}

// cleanupTimeout bounds cleanup that has to run even when the caller's context
// has ended — releasing a backup's timestamp claim. It is a var, not a const, purely so a test can shorten it;
// nothing in production assigns to it.
var cleanupTimeout = 30 * time.Second

// cleanupContext returns the context for such cleanup. It is detached from
// ctx, because cleanup exists precisely for the runs that were cancelled or ran
// out of time; and it carries a deadline of its own, because detaching also
// drops ctx's deadline, and without one a stalled store (or an HTTP client with
// no timeout) would hold the caller indefinitely.
func cleanupContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
}
