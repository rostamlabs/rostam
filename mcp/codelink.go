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

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
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

// normalizeLinkPath returns p as a clean repo-relative slash path. An absolute
// p must lie inside root. Empty paths, "." and anything that escapes root
// (via ".." or an absolute path elsewhere) are rejected. Symlinks are not
// resolved.
func normalizeLinkPath(root, p string) (string, error) {
	if strings.TrimSpace(p) == "" {
		return "", errors.New("path is empty")
	}
	if filepath.IsAbs(p) {
		if root == "" {
			return "", fmt.Errorf("absolute path %q needs a project root", p)
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
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

// staleFor reports whether a linked memory's file has changed since it was
// written: true, "file missing", or nil (fresh, unlinked, or not checkable).
func (s *Server) staleFor(md rostam.VectorMetadata) any {
	var p, stored string
	if v, ok := md[pathField]; ok && v.Kind == vector.ValueString {
		p = v.Str
	}
	if v, ok := md[fileHashField]; ok && v.Kind == vector.ValueString {
		stored = v.Str
	}
	return s.staleState(p, stored)
}

func (s *Server) staleState(p, stored string) any {
	if p == "" {
		return nil
	}
	cur, st := fileHash(s.projectRoot, p)
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

// markStale fills Stale on every linked hit.
func (s *Server) markStale(hits []memoryHit) {
	for i := range hits {
		hits[i].Stale = s.staleState(hits[i].Path, hits[i].fileHash)
	}
}
