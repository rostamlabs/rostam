// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// FSObjectStore is a filesystem-backed objstore.ObjectStore rooted at a local
// directory. Each object key maps to a file at <root>/<key>, with key's "/"
// separators becoming directory levels (parent dirs are created on Put). It is
// fully demoable today with no cloud credentials and lets the backup driver run
// against local disk before any S3 bucket is configured.
//
// It implements the SAME objstore.ObjectStore interface as the SigV4 S3 client,
// so swapping FSObjectStore for an S3Store in the server's backup driver is a
// one-line change and the Backup/Restore/retention logic is untouched.
//
// Writes are atomic per object: Put streams into a temp file in the destination
// directory and renames it over the final path, so a crash mid-write never
// leaves a torn snapshot (mirroring CollectionStore's tmp+rename discipline).
type FSObjectStore struct {
	root string
}

// compile-time assertion that FSObjectStore satisfies objstore.ObjectStore.
var _ objstore.ObjectStore = (*FSObjectStore)(nil)

// putTempPrefix is the fixed name prefix of every Put staging file. It is kept
// OUT of the ".snap"/".cfg.json" key space (List/LatestKey/prune filter on
// ".snap") so an in-flight staging file is never seen by retention as a
// published object — and it is the marker that makes reclaiming leftovers safe
// (see reclaimStaleTemps). That only holds because the name space is RESERVED:
// keyToPath rejects any key whose final segment starts with this prefix, so no
// published object can ever wear a name the sweep is entitled to delete.
//
// A staging file left behind by a process killed mid-Put is NOT a small, inert
// file: the very same Put stages the SNAPSHOT, so the leftover can be a partial
// snapshot of arbitrary size — gigabytes of dead space on the backup volume,
// leaked once per kill and never reclaimed on its own.
const putTempPrefix = ".rostam-put-"

// putTempSuffix is the fixed name suffix of every Put staging file, the second
// half of the CreateTemp pattern below. The sweep requires BOTH this suffix and
// putTempPrefix, so the two filters are declared here together and cannot drift
// apart from the name Put actually creates.
const putTempSuffix = ".tmp"

// staleTempAge is how old a staging file must be before a Put reclaims it. It is
// the outer safety margin of the sweep: a CONCURRENT Put created its staging
// file moments ago, and a Put whose copy is still running keeps refreshing the
// mtime (see tempHeartbeat), so nothing an hour old can belong to a live
// writer — only to a process that died mid-Put.
const staleTempAge = time.Hour

// tempHeartbeat is how often a running Put refreshes its staging file's mtime.
// It must be well inside staleTempAge so a single missed tick cannot expose a
// live Put to the sweep. It is a var, not a const, purely so a test can shorten
// it; nothing in production assigns to it.
var tempHeartbeat = staleTempAge / 4

// NewFSObjectStore returns an FSObjectStore rooted at root, creating root if it
// does not exist.
func NewFSObjectStore(root string) (*FSObjectStore, error) {
	if err := os.MkdirAll(root, 0o750); err != nil {
		return nil, fmt.Errorf("fsstore: mkdir root %q: %w", root, err)
	}
	return &FSObjectStore{root: root}, nil
}

// keyToPath resolves a forward-slash object key to an on-disk path under root,
// using the OS path separator. It REJECTS any key that would escape root: the
// key is path.Clean'd as an absolute path (collapsing ".."/"."), then the joined
// destination is verified to stay within root. A key containing "../" segments,
// a NUL byte, or one that resolves to root itself is an error — so Put cannot
// overwrite, Get cannot read, and Delete cannot remove a file outside root. It
// also rejects a key landing in the reserved staging name space (see below).
func (f *FSObjectStore) keyToPath(key string) (string, error) {
	if strings.ContainsRune(key, '\x00') {
		return "", fmt.Errorf("fsstore: invalid key %q", key)
	}
	// Object keys are slash-separated by the ObjectStore contract, so a backslash
	// is an ordinary character in a key — but NOT to filepath on Windows, where it
	// is a separator. A key like "coll\.rostam-put-x.tmp" therefore has one final
	// segment by the key's own rules and a DIFFERENT one by the filesystem's, which
	// is how such a key could slip past the reservation below and publish a file the
	// sweep would later delete. Refusing the character keeps one key meaning one
	// path on every platform; the backup package's own keys never carry it
	// (url.PathEscape encodes it).
	if strings.ContainsRune(key, '\\') {
		return "", fmt.Errorf("fsstore: invalid key %q (backslash is not a key separator)", key)
	}
	// path.Clean("/"+key) forces an absolute path and resolves away ".."/"." so a
	// traversal key like "../../etc/passwd" cleans to "/etc/passwd" (a single
	// rooted segment list) rather than escaping.
	clean := path.Clean("/" + key)
	if clean == "/" {
		return "", fmt.Errorf("fsstore: invalid key %q (empty after clean)", key)
	}
	// The putTempPrefix name space is RESERVED for Put's staging files, which
	// reclaimStaleTemps is entitled to delete once they age past staleTempAge. A
	// key whose final segment lands in that name space would be published as a
	// real object and then silently swept an hour later, so refuse it here.
	//
	// This sits in keyToPath, so the refusal applies uniformly to Get and Delete
	// as well as Put. That is deliberate, not collateral: a key that can never be
	// written must not be readable or deletable either, or the store would answer
	// for names whose meaning it does not control. Only the FINAL segment is
	// checked — a directory so named is harmless, since the sweep skips
	// directories and only ever looks at one directory's own entries.
	// filepath.Base over FromSlash, not path.Base: the check must see the final
	// segment the FILESYSTEM will see, which on Windows means splitting on the
	// separator filepath uses. The backslash rejection above already makes the two
	// agree, so this is defence in depth against a future key shape that does not.
	if strings.HasPrefix(filepath.Base(filepath.FromSlash(clean)), putTempPrefix) {
		return "", fmt.Errorf("fsstore: invalid key %q (the %q name space is reserved for staging files)", key, putTempPrefix)
	}
	dst := filepath.Join(f.root, filepath.FromSlash(clean))
	rootAbs, err := filepath.Abs(f.root)
	if err != nil {
		return "", fmt.Errorf("fsstore: resolve root: %w", err)
	}
	dstAbs, err := filepath.Abs(dst)
	if err != nil {
		return "", fmt.Errorf("fsstore: resolve key %q: %w", key, err)
	}
	if dstAbs != rootAbs && !strings.HasPrefix(dstAbs, rootAbs+string(os.PathSeparator)) {
		return "", fmt.Errorf("fsstore: key %q escapes root", key)
	}
	return dst, nil
}

// Put writes r's full contents to <root>/<key> atomically (temp file + rename),
// creating parent directories as needed and overwriting any prior value. size is
// accepted for interface parity (Content-Length on the wire); the bytes actually
// written are whatever r yields.
func (f *FSObjectStore) Put(_ context.Context, key string, r io.Reader, size int64) error {
	dst, err := f.keyToPath(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
		return fmt.Errorf("fsstore: mkdir for %q: %w", key, err)
	}
	// Reclaim staging files abandoned by a process killed mid-Put before staging
	// ours. Doing it here rather than at open is what makes it race-free on a root
	// shared by several processes; it is best-effort and never fails the Put.
	reclaimStaleTemps(filepath.Dir(dst))
	// Staging name deliberately does NOT end in snapExt: List/LatestKey/prune
	// filter on the ".snap" suffix, so a temp named "*.snap" would be visible to
	// them mid-write — a concurrent prune could delete an in-flight Put's staging
	// file (or the temp could inflate a retention count). A ".tmp" suffix keeps it
	// invisible to those filters until the atomic rename publishes the real key.
	tmp, err := os.CreateTemp(filepath.Dir(dst), putTempPrefix+"*"+putTempSuffix)
	if err != nil {
		return fmt.Errorf("fsstore: temp for %q: %w", key, err)
	}
	tmpName := tmp.Name()
	// Keep this staging file's mtime fresh for as long as the copy runs, so a slow
	// or wedged reader is not mistaken for an abandoned Put by a concurrent sweep
	// (see startTempHeartbeat). Stopped explicitly before the rename; the deferred
	// stop covers every error return in between.
	stopHeartbeat := startTempHeartbeat(tmpName)
	defer stopHeartbeat()
	if _, err := io.Copy(tmp, r); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: write %q: %w", key, err)
	}
	// fsync the temp file before the rename so the data blocks reach stable
	// storage first (mirroring CollectionStore.writeSnapshotFile). Without this,
	// ext4/xfs may make the rename's metadata durable before the file data,
	// leaving a torn/empty snapshot at dst after power loss even though Put
	// already returned nil.
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: sync temp for %q: %w", key, err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: close temp for %q: %w", key, err)
	}
	// The staging file is fully written and closed; nothing is left to protect from
	// the sweep, and the path is about to stop existing under this name.
	stopHeartbeat()
	if err := os.Rename(tmpName, dst); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: rename for %q: %w", key, err)
	}
	// fsync the destination's parent directory so the rename itself is durable;
	// otherwise the directory entry for dst may be lost on power loss.
	if err := syncDir(filepath.Dir(dst)); err != nil {
		return fmt.Errorf("fsstore: sync dir for %q: %w", key, err)
	}
	return nil
}

// startTempHeartbeat refreshes path's mtime every tempHeartbeat until the
// returned stop function is called. stop is idempotent and WAITS for the
// refreshing goroutine to exit, so no stray Chtimes can trail a later rename.
//
// Why a heartbeat and not an inter-process lock: the sweep tells a live writer
// from a dead one by age alone, and an mtime set once at create time cannot
// express "alive but making no progress" — a Put whose reader stalls for longer
// than staleTempAge looks exactly like a temp orphaned by a killed process, so a
// concurrent Put would unlink it and the stalled Put's rename would then fail. A
// lock is the wrong instrument: it would have to be held across a copy of
// unbounded duration and released correctly by a process that may be killed at
// any instant — which is the very failure the sweep exists to clean up after, so
// the lock would need a staleness rule of its own and we would be back here. A
// ticker dies with its process for free: while the Put lives its staging file
// keeps getting younger, and the moment the process is killed the refreshes stop
// and the file starts ageing toward the bound. That is exactly the signal the
// sweep needs, held in the file itself with no extra state to reconcile.
func startTempHeartbeat(path string) func() {
	done := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		tick := time.NewTicker(tempHeartbeat)
		defer tick.Stop()
		for {
			select {
			case <-done:
				return
			case now := <-tick.C:
				// Ignored: the file may already be renamed away (a refresh racing
				// stop), and a missed refresh only risks the temp being swept — it
				// can never corrupt anything.
				_ = os.Chtimes(path, now, now)
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() {
			close(done)
			<-exited
		})
	}
}

// reclaimStaleTemps removes abandoned Put staging files from dir. It is called
// from Put — on the NEXT write to that directory — rather than from a sweep at
// open: there is then no startup instant to reason about on a shared root, only
// an age bound, and three filters keep it safe:
//
//   - the putTempPrefix name prefix AND the putTempSuffix suffix, both of which
//     Put's CreateTemp pattern gives every staging file. Requiring both, rather
//     than the prefix alone, means an unrelated file that merely happens to share
//     the prefix is not swept; and no PUBLISHED object can wear either name,
//     because keyToPath reserves the prefix outright.
//   - the staleTempAge modtime bound. A concurrent Put created its staging file
//     moments ago and a long-running Put keeps refreshing its mtime (see
//     startTempHeartbeat), so a live writer's temp is never old enough to be
//     swept; only a temp orphaned by a dead process ages past the bound.
//
// Every error is ignored, including the removal's: reclaiming dead space must
// never fail a backup.
func reclaimStaleTemps(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-staleTempAge)
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), putTempPrefix) || !strings.HasSuffix(e.Name(), putTempSuffix) {
			continue
		}
		info, ierr := e.Info()
		if ierr != nil {
			continue
		}
		if !info.ModTime().Before(cutoff) {
			continue
		}
		_ = os.Remove(filepath.Join(dir, e.Name()))
	}
}

// syncDir fsyncs the directory at dir so a preceding rename/create within it is
// made durable. Both a failure to open the directory and a Sync error are
// surfaced to the caller, since either means the rename cannot be guaranteed
// durable across a crash.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err := d.Sync(); err != nil {
		_ = d.Close()
		return err
	}
	return d.Close()
}

// Get opens <root>/<key> for reading. The caller MUST Close the returned reader.
// A missing key returns objstore.ErrNotFound.
func (f *FSObjectStore) Get(_ context.Context, key string) (io.ReadCloser, error) {
	dst, err := f.keyToPath(key)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(dst)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, objstore.ErrNotFound
		}
		return nil, fmt.Errorf("fsstore: open %q: %w", key, err)
	}
	return file, nil
}

// List returns every object whose key begins with prefix (the path relative to
// root, in forward-slash form), sorted by key. Directories are not reported.
//
// The walk starts at the deepest directory the prefix fully names (for
// "acme/col/2024-" that is acme/col), not at root: retention lists one
// collection prefix per collection and per run, and an existence probe lists a
// single key, so walking the whole root would make each backup run cost
// O(collections × all stored files). The key-prefix filter is still applied to
// every entry, so the result is what a full-root walk filtered by prefix would
// return.
//
// Symlinks are never followed. A full-root walk never descended into one, so
// nothing under a symlink was ever listable; a scoped walk must not resolve
// one on its way to the start directory either, or it would list — and let
// retention delete — files under keys the full walk never produced. Two layers
// enforce that. Every component between root and the start directory is
// Lstat'd and a symlink is an ERROR (where the full walk was silent — that is
// deliberate: a tenant directory an operator symlinked onto another disk
// otherwise surfaces as "no snapshots" from LatestKey/RestoreLatest and as a
// silent no-op from prune, when the truth is that List cannot see it). Behind
// that, the walk runs inside os.Root, so even a link created between the Lstat
// and the walk cannot leave root: os.Root refuses absolute and root-escaping
// link targets outright, and a relative in-root link — which os.Root itself
// WOULD follow — can at worst alias in-root files. path.Clean on the prefix is
// the only lexical step and it is not relied on for containment.
//
// Error semantics, matching MemStore.List (a pure prefix filter, which cannot
// fail on the SHAPE of a prefix): a prefix whose directory does not exist yet,
// or whose directory component names a plain file, or that contains a NUL
// byte (no key can — keyToPath rejects it) yields no objects (nil), not an
// error. A missing or unreadable ROOT is an error, as it always was: an
// unmounted backup volume must not read as "no snapshots".
//
// On a case-insensitive filesystem (darwin, Windows) a prefix that differs
// from the on-disk name only by case walks the directory and yields keys in the
// prefix's case, where a full-root walk yielded the on-disk case and matched
// nothing. No caller re-cases a prefix (tenants are validated, collections
// escaped), so this is documented rather than defended.
//
// NOTE: like the S3 store, List BUFFERS every matching key in memory (one
// ObjectInfo per matching file). This is bounded for the backup/retention use
// (a handful of snapshots per collection prefix) but a prefix spanning millions
// of files will materialize them all at once; scope the prefix narrowly.
func (f *FSObjectStore) List(_ context.Context, prefix string) ([]objstore.ObjectInfo, error) {
	if strings.ContainsRune(prefix, '\x00') {
		return nil, nil
	}
	r, err := os.OpenRoot(f.root)
	if err != nil {
		return nil, fmt.Errorf("fsstore: list %q: open root: %w", prefix, err)
	}
	defer func() { _ = r.Close() }()
	start, ok, err := listStart(r, prefix)
	if err != nil {
		return nil, fmt.Errorf("fsstore: list %q: %w", prefix, err)
	}
	if !ok {
		return nil, nil // no such prefix: nothing to list
	}
	var out []objstore.ObjectInfo
	err = fs.WalkDir(r.FS(), start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		// fs.FS paths are already slash-separated and root-relative: p IS the key.
		if !strings.HasPrefix(p, prefix) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return ierr
		}
		out = append(out, objstore.ObjectInfo{
			Key:          p,
			Size:         info.Size(),
			LastModified: info.ModTime(),
		})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("fsstore: list %q: %w", prefix, err)
	}
	sortInfosByKey(out)
	return out, nil
}

// listStart returns the fs.FS path (unrooted, slash-separated, "." for root)
// List should walk for prefix: every COMPLETE "/"-separated segment of prefix
// that exists as a real directory. The trailing partial segment, if any, is a
// name filter, not a directory. ok=false means the walk would yield nothing
// (a segment is missing or is a plain file); a symlink segment is an error
// (see List). Scoping stops at the first segment containing a backslash and
// walks from its parent instead: on Windows os.Root would treat "\\" as a
// separator while the key filter treats it as a literal, so a scoped walk
// there could return keys that do not carry the requested prefix — walking
// from the parent lets the lexical filter decide, identically everywhere.
// A prefix whose directory part is not already canonical (a leading "/", a
// "." or ".." segment, a doubled "/") can match no key at all — every key is
// a clean relative path — so it yields nothing without touching disk, rather
// than being scoped to the directory its CLEANED form names (which the raw
// prefix would never match, and which could be a symlink the raw prefix would
// never have reached). Containment itself is enforced by os.Root, not here.
func listStart(r *os.Root, prefix string) (start string, ok bool, err error) {
	start = "."
	i := strings.LastIndex(prefix, "/")
	if i < 0 {
		return start, true, nil
	}
	rel := strings.TrimPrefix(path.Clean("/"+prefix[:i]), "/")
	if rel != prefix[:i] {
		return "", false, nil // non-canonical directory part: no key can match
	}
	if rel == "" {
		return start, true, nil
	}
	for _, seg := range strings.Split(rel, "/") {
		if strings.ContainsRune(seg, '\\') {
			break
		}
		next := path.Join(start, seg)
		info, lerr := r.Lstat(next)
		if lerr != nil {
			if errors.Is(lerr, fs.ErrNotExist) || errors.Is(lerr, syscall.ENOTDIR) {
				return "", false, nil
			}
			return "", false, lerr
		}
		if info.Mode()&fs.ModeSymlink != 0 {
			return "", false, fmt.Errorf("%q is a symlink: refusing to list through it", next)
		}
		if !info.IsDir() {
			return "", false, nil // a plain file cannot hold keys beneath it
		}
		start = next
	}
	return start, true, nil
}

// Delete removes <root>/<key>. Deleting a missing key returns
// objstore.ErrNotFound, matching the S3 client and the MemStore fake.
func (f *FSObjectStore) Delete(_ context.Context, key string) error {
	dst, err := f.keyToPath(key)
	if err != nil {
		return err
	}
	if err := os.Remove(dst); err != nil {
		if os.IsNotExist(err) {
			return objstore.ErrNotFound
		}
		return fmt.Errorf("fsstore: delete %q: %w", key, err)
	}
	return nil
}

// sortInfosByKey sorts object infos ascending by key, matching the
// objstore.ObjectStore.List contract (sorted by key).
func sortInfosByKey(in []objstore.ObjectInfo) {
	for i := 1; i < len(in); i++ {
		for j := i; j > 0 && in[j-1].Key > in[j].Key; j-- {
			in[j-1], in[j] = in[j], in[j-1]
		}
	}
}
