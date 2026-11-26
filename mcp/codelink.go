// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/rostamlabs/rostam/vector"
)

// Metadata fields that link a memory to a file. Reserved like nsField: stored
// in metadata, lifted into memoryHit's typed fields, never echoed back.
const (
	pathField     = "__path"      // repo-relative, slash-separated file path
	symbolField   = "__symbol"    // optional symbol within the file
	fileHashField = "__file_hash" // hex SHA-256 of the file at write time ("" if unreadable)
	linkField     = "__link"      // marker: the three fields above are a code link
)

// The three link field names were ordinary user metadata keys before code
// links existed, so a memory written by an older version may carry them as
// the user's own data. Only a memory that also carries linkField as a geo
// point is a code link: a geo value is the one kind remember's JSON metadata
// can never produce (objects are rejected), now or in any earlier version,
// so no legacy field can be mistaken for the marker. Without it, the fields
// stay visible user metadata and the memory is not linked to any file.
func linkMarker() vector.Value { return vector.NewGeo(0, 0) }

func isLinkMarker(v vector.Value, ok bool) bool { return ok && v.Kind == vector.ValueGeo }

// linkFields are the fields owned by a code link (stripped only from a
// memory that is one).
var linkFields = [...]string{pathField, symbolField, fileHashField, linkField}

type fileState int

const (
	fileOK         fileState = iota // file read; hash is valid
	fileMissing                     // file does not exist
	fileNoRoot                      // no project root configured: nothing can be checked
	fileUnreadable                  // exists but is not a readable regular file inside the root
)

// NormalizeLinkPath is normalizeLinkPath for callers outside this package (the
// hook), so both sides map a file to the same repo-relative path.
func NormalizeLinkPath(root, p string) (string, error) { return normalizeLinkPath(root, p) }

// normalizeLinkPath returns p as a clean repo-relative slash path. An absolute
// p must lie inside root. Empty paths, "." and anything that escapes root
// (via ".." or an absolute path elsewhere) are rejected.
//
// Symlinks are handled so that every way of naming the same checkout gives the
// same answer: the root is accepted both as given and with its symlinks
// resolved, and an absolute p reached through a symlinked prefix of the root
// (a symlinked cwd, say) is matched by resolving p's ancestors until one IS the
// root. Symlinks inside the repo are never resolved: a symlinked dir
// docs -> site/docs keeps the path as written, docs/x.md.
func normalizeLinkPath(root, p string) (string, error) {
	if strings.TrimSpace(p) == "" {
		return "", errors.New("path is empty")
	}
	if filepath.IsAbs(p) {
		if root == "" {
			return "", fmt.Errorf("absolute path %q needs a project root", p)
		}
		rel, ok := relUnderRoot(root, filepath.Clean(p))
		if !ok {
			return "", fmt.Errorf("path %q is outside the project root", p)
		}
		p = rel
	}
	rel := path.Clean(filepath.ToSlash(p))
	if rel == "." || rel == ".." || strings.HasPrefix(rel, "../") || path.IsAbs(rel) {
		return "", fmt.Errorf("path %q is outside the project root", p)
	}
	return rel, nil
}

// relUnderRoot maps the clean absolute path p to a path relative to root, or
// reports false when p is not inside it. See normalizeLinkPath.
func relUnderRoot(root, p string) (string, bool) {
	roots := []string{filepath.Clean(root)}
	realRoot := roots[0]
	if r, err := filepath.EvalSymlinks(root); err == nil && r != realRoot {
		realRoot = r
		roots = append(roots, r)
	}
	for _, r := range roots {
		if rel, ok := lexicalRel(r, p); ok {
			return rel, true
		}
	}
	// p names the checkout through a symlinked prefix: find the deepest
	// existing ancestor that resolves to the root and keep the rest lexical, so
	// in-repo symlinks and not-yet-created files are left as written.
	for a := filepath.Dir(p); ; a = filepath.Dir(a) {
		if r, err := filepath.EvalSymlinks(a); err == nil && r == realRoot {
			return lexicalRel(a, p)
		}
		if filepath.Dir(a) == a {
			return "", false
		}
	}
}

// lexicalRel is filepath.Rel(base, p) restricted to paths strictly inside base.
func lexicalRel(base, p string) (string, bool) {
	rel, err := filepath.Rel(base, p)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", false
	}
	return rel, true
}

// hashFileFn is fileHash behind a seam so tests can count hashing calls.
var hashFileFn = fileHash

// fileHash streams the file at root/rel through SHA-256. The hash is "" unless
// the state is fileOK. fileNoRoot means only that no root is configured; a
// configured root that is gone or unreadable makes its files missing or
// unreadable, so their memories are still flagged.
//
// Every access goes through an os.Root, so no symlink — in any path
// component, not just the last — can reach outside the project; symlinks that
// stay inside it are followed. os.Root refuses ABSOLUTE symlink targets even
// when they point back inside the project (alias.go -> /repo/a.go), so on
// such a refusal the path is resolved once with EvalSymlinks and, only if the
// result lies under the root's real path, opened again through an os.Root on
// that real root (still confined, so a link swapped in meanwhile cannot
// escape). Only regular files are hashed: the type is checked before opening
// (a FIFO or device must not be opened at all), the open is non-blocking (a
// file swapped for a FIFO in between cannot hang it), and the opened
// descriptor is checked again before reading.
func fileHash(root, rel string) (string, fileState) {
	if root == "" {
		return "", fileNoRoot
	}
	name := filepath.FromSlash(rel)
	h, st, err := hashInRoot(root, name)
	if err == nil || errors.Is(err, os.ErrNotExist) {
		return h, st
	}
	realRoot, rerr := filepath.EvalSymlinks(root)
	real, ferr := filepath.EvalSymlinks(filepath.Join(root, name))
	if errors.Is(ferr, os.ErrNotExist) {
		return "", fileMissing // e.g. an absolute in-repo symlink whose target is gone
	}
	if rerr != nil || ferr != nil {
		return h, st
	}
	inside, ok := lexicalRel(realRoot, real)
	if !ok {
		return h, st // resolves outside the project: never read
	}
	h2, st2, _ := hashInRoot(realRoot, inside)
	return h2, st2
}

// hashInRoot is one confined attempt; err is the access error, if any, so the
// caller can tell a refused symlink from a missing file.
func hashInRoot(root, name string) (string, fileState, error) {
	classify := func(err error) fileState {
		if errors.Is(err, os.ErrNotExist) {
			return fileMissing
		}
		return fileUnreadable
	}
	r, err := os.OpenRoot(root)
	if err != nil {
		return "", classify(err), err
	}
	defer func() { _ = r.Close() }()
	fi, err := r.Stat(name)
	if err != nil {
		return "", classify(err), err
	}
	if !fi.Mode().IsRegular() {
		return "", fileUnreadable, nil
	}
	f, err := r.OpenFile(name, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return "", classify(err), err
	}
	defer func() { _ = f.Close() }()
	if ofi, err := f.Stat(); err != nil || !ofi.Mode().IsRegular() {
		return "", fileUnreadable, nil
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fileUnreadable, nil
	}
	return hex.EncodeToString(h.Sum(nil)), fileOK, nil
}

// staleState compares a memory's stored hash against cur/st, the file's
// current state: true (changed), "file missing", or nil (fresh, unlinked, or
// not checkable).
func staleState(p, stored, cur string, st fileState) any {
	if p == "" {
		return nil
	}
	switch st {
	case fileMissing:
		if stored != "" {
			return "file missing"
		}
	case fileUnreadable:
		// It was hashed when linked, and now it is a directory, a symlink out
		// of the project, or unreadable: it is no longer the file remembered.
		if stored != "" {
			return true
		}
	case fileOK:
		// No stored hash means the file did not exist at write time, so
		// whatever is there now is a change.
		if cur != stored {
			return true
		}
	}
	return nil
}

// markStale fills Stale on every linked hit. Each distinct path is hashed once
// per call: a page of hits is often many memories about the same file.
func (s *Server) markStale(hits []memoryHit) {
	type state struct {
		hash string
		st   fileState
	}
	seen := map[string]state{}
	for i := range hits {
		p := hits[i].Path
		if p == "" {
			hits[i].Stale = nil
			continue
		}
		cur, ok := seen[p]
		if !ok {
			cur.hash, cur.st = hashFileFn(s.projectRoot, p)
			seen[p] = cur
		}
		hits[i].Stale = staleState(p, hits[i].fileHash, cur.hash, cur.st)
	}
}
