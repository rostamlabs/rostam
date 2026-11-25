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
)

// Metadata fields that link a memory to a file. Reserved like nsField: stored
// in metadata, lifted into memoryHit's typed fields, never echoed back.
const (
	pathField     = "__path"      // repo-relative, slash-separated file path
	symbolField   = "__symbol"    // optional symbol within the file
	fileHashField = "__file_hash" // hex SHA-256 of the file at write time ("" if unreadable)
)

type fileState int

const (
	fileOK      fileState = iota // file read; hash is valid
	fileMissing                  // file does not exist
	fileNoRoot                   // no project root, or the file is unreadable (e.g. a directory)
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
// the state is fileOK.
func fileHash(root, rel string) (string, fileState) {
	if root == "" {
		return "", fileNoRoot
	}
	full := filepath.Join(root, filepath.FromSlash(rel))
	// Lstat first: opening a FIFO or device would block, and a symlink could
	// point outside the project. Only regular files are hashed.
	fi, err := os.Lstat(full)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fileMissing
		}
		return "", fileNoRoot
	}
	if !fi.Mode().IsRegular() {
		return "", fileNoRoot
	}
	f, err := os.Open(full)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fileMissing
		}
		return "", fileNoRoot
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fileNoRoot
	}
	return hex.EncodeToString(h.Sum(nil)), fileOK
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
