// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// The namespace registry: which project owns which memory namespace on this
// machine. A namespace used to be just the project's directory name, so two
// unrelated checkouts called "api" shared (and overwrote) one memory. Now each
// project is identified by its canonical main-repository path, and the
// registry, a small file beside the daemon's data, maps that identity to the
// namespace assigned to it:
//
//   - the first identity seen with a given directory name gets the plain name
//     (so memories saved before the registry existed keep working);
//   - a different identity with the same name gets "<name>-<8 hex of its
//     identity's hash>";
//   - an assignment is sticky: an identity that has a name keeps it;
//   - an identity with no assignment may take over the plain name only when
//     the recorded owner's path no longer exists (ENOENT, nothing else).
//
// When the registry cannot be used (unreadable, corrupt, or its lock fails)
// the hashed name is used and the file is never rewritten: the plain name is
// never handed out by guesswork. A lock that is merely busy past the caller's
// deadline is errRegistryBusy, with no name at all.
//
// The hook and `mcp` both get their namespace from projectNamespace and
// nothing else, so they always agree.

const (
	nsRegistryName     = "namespaces.json"
	nsRegistryLockName = "namespaces.lock"
)

// nsRegistryDir is where the registry lives: the daemon's data directory.
// A variable so tests use a temporary one.
var nsRegistryDir = defaultMemoryDir

// errLockBusy: a lock was held by someone else until the caller's ctx ended.
var errLockBusy = errors.New("lock is busy")

// errRegistryBusy means another process held the registry lock until the
// caller's deadline: no namespace was decided. The hook then stays silent.
var errRegistryBusy = errors.New("namespace registry is busy")

// errIdentifyTimeout means git did not answer before the caller's deadline, so
// the project could not be identified and nothing was registered.
var errIdentifyTimeout = errors.New("timed out identifying the project (git)")

// namespaceUndecided reports the errors that leave no namespace decided: the
// hook then injects nothing, and mcp refuses to start rather than pin a name
// the hooks would not use.
func namespaceUndecided(err error) bool {
	return errors.Is(err, errRegistryBusy) || errors.Is(err, errIdentifyTimeout)
}

// projectID identifies a project for the registry.
type projectID struct {
	identity string // canonical main-repository path (see identifyProject)
	base     string // the directory name the plain namespace is (as before the registry)
	// stable names the hashed namespaces: base, except for a separate-git-dir
	// repository, whose worktrees have different directory names but must
	// still agree on the hashed (fallback) name.
	stable string
}

type nsRegistryFile struct {
	Version  int               `json:"version"`
	Projects map[string]string `json:"projects"` // identity -> namespace
}

// hashedNamespace is the name for an identity that cannot have the plain one,
// and the fallback when the registry cannot be used: <stable>-<8 hex>.
func hashedNamespace(id projectID) string { return hashedNamespaceN(id, 4) }

// hashedNamespaceN uses n bytes of the identity's hash (n <= 32).
func hashedNamespaceN(id projectID, n int) string {
	sum := sha256.Sum256([]byte(id.identity))
	stable := id.stable
	if stable == "" {
		stable = id.base
	}
	return stable + "-" + hex.EncodeToString(sum[:n])
}

// freeHashedNamespace is hashedNamespace unless another identity already holds
// that name (a 32-bit collision, or a project whose directory is literally
// called that): then a longer hash, then a counter, until the name is unused.
func freeHashedNamespace(id projectID, reg nsRegistryFile) string {
	taken := make(map[string]bool, len(reg.Projects))
	for owner, name := range reg.Projects {
		if owner != id.identity {
			taken[name] = true
		}
	}
	for _, n := range []int{4, 8, 16, 32} {
		if ns := hashedNamespaceN(id, n); !taken[ns] {
			return ns
		}
	}
	for i := 2; ; i++ {
		if ns := fmt.Sprintf("%s-%d", hashedNamespaceN(id, 32), i); !taken[ns] {
			return ns
		}
	}
}

// projectNamespace returns id's namespace from the registry in dir, assigning
// one if id has none. A non-nil error other than errRegistryBusy is a warning:
// the returned (hashed) name is still the one to use.
func projectNamespace(ctx context.Context, dir string, id projectID) (string, error) {
	if dir == "" {
		return hashedNamespace(id), errors.New("namespace registry: no directory (no home directory?)")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil { //nolint:gosec // G703: the daemon data dir
		return hashedNamespace(id), fmt.Errorf("namespace registry: %w", err)
	}
	unlock, err := lockFileCtx(ctx, filepath.Join(dir, nsRegistryLockName))
	if errors.Is(err, errLockBusy) {
		return "", errRegistryBusy
	}
	if err != nil {
		return hashedNamespace(id), fmt.Errorf("namespace registry: locking: %w", err)
	}
	defer unlock()

	path := filepath.Join(dir, nsRegistryName)
	reg := nsRegistryFile{Version: 1, Projects: map[string]string{}}
	switch b, err := os.ReadFile(path); { //nolint:gosec // G703: fixed name in the daemon data dir
	case errors.Is(err, fs.ErrNotExist):
	case err != nil:
		return hashedNamespace(id), fmt.Errorf("namespace registry %s unreadable: %w", path, err)
	default:
		// A fresh value: "null" or "{}" must read as corrupt, not as the
		// empty registry it would leave a pre-filled one.
		var got nsRegistryFile
		if err := json.Unmarshal(b, &got); err != nil || got.Projects == nil {
			return hashedNamespace(id), fmt.Errorf("namespace registry %s is corrupt (left as is; move it aside to rebuild): %v", path, err)
		}
		reg = got
	}
	if ns, ok := reg.Projects[id.identity]; ok {
		return ns, nil
	}
	ns := id.base
	for owner, name := range reg.Projects {
		if name != id.base {
			continue
		}
		if _, err := os.Stat(owner); errors.Is(err, fs.ErrNotExist) { //nolint:gosec // G703: a recorded project path, only stat'ed
			delete(reg.Projects, owner) // the owner is gone: its name is free
		} else {
			ns = freeHashedNamespace(id, reg) // owned (or unknown): never share it
		}
		break
	}
	reg.Projects[id.identity] = ns
	if err := writeRegistry(dir, path, reg); err != nil {
		return hashedNamespace(id), fmt.Errorf("namespace registry: saving: %w", err)
	}
	return ns, nil
}

// writeRegistry replaces the registry atomically (temp file, 0600, rename).
func writeRegistry(dir, path string, reg nsRegistryFile) error {
	b, err := json.MarshalIndent(reg, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".namespaces-*") // 0600
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }() //nolint:gosec // G703: CreateTemp name in the data dir
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path) //nolint:gosec // G703: as above
}

// identifyProject returns the project root for dir (its git top-level, else
// dir) and its registry identity:
//
//   - in a git repository whose common dir is "<main>/.git" (a normal clone,
//     and every linked worktree of it): the canonical <main>, named after
//     <main>'s directory, so worktrees share one identity and one name;
//   - with a separate git dir (common dir not named ".git"): the canonical
//     common dir, so unrelated repositories never collapse together and its
//     worktrees never split; named after the checkout's directory;
//   - outside git: the canonical project root.
//
// "Canonical" means symlinks resolved, so two paths to one checkout agree.
// The name part matches what the namespace was before the registry existed.
//
// ok is false when ctx ran out during the git calls: the result may be a
// subdirectory instead of the repository, so it must not be registered.
func identifyProject(ctx context.Context, dir string) (root string, id projectID, ok bool) {
	root = dir
	if out, ok := gitOut(ctx, dir, "rev-parse", "--show-toplevel"); ok {
		root = out
	}
	id = projectID{identity: canonicalPath(root), base: filepath.Base(root)}
	if out, ok := gitOut(ctx, dir, "rev-parse", "--git-common-dir"); ok {
		if !filepath.IsAbs(out) {
			out = filepath.Join(dir, out)
		}
		out = filepath.Clean(out)
		if filepath.Base(out) == ".git" {
			main := filepath.Dir(out)
			id = projectID{identity: canonicalPath(main), base: filepath.Base(main)}
		} else {
			id.identity = canonicalPath(out)
			if s := strings.TrimSuffix(filepath.Base(id.identity), ".git"); s != "" {
				id.stable = s
			}
		}
	}
	if id.stable == "" {
		id.stable = id.base
	}
	return root, id, ctx.Err() == nil
}

func canonicalPath(p string) string {
	if r, err := filepath.EvalSymlinks(p); err == nil {
		p = r
	}
	if a, err := filepath.Abs(p); err == nil {
		p = a
	}
	return filepath.Clean(p)
}

// resolveNamespace is THE project-namespace function: the hook and `mcp` both
// call it, so they always agree. It returns the project root for dir and the
// namespace the registry assigns to dir's project. err is either a warning
// (the returned hashed name is still the one to use, and the other side gets
// the same one) or, when namespaceUndecided(err), a sign that no name was
// decided and the caller must not use one.
func resolveNamespace(ctx context.Context, dir string) (root, ns string, err error) {
	root, id, ok := identifyProject(ctx, dir)
	if !ok {
		return root, hashedNamespace(id), errIdentifyTimeout
	}
	regDir, derr := nsRegistryDir()
	if derr != nil {
		regDir = ""
	}
	ns, err = projectNamespace(ctx, regDir, id)
	return root, ns, err
}

// mcpRegistryWait bounds how long `mcp` waits for git and the registry lock at
// startup; the hook waits only as long as its own budget allows.
// A variable so tests need not wait it out.
var mcpRegistryWait = 30 * time.Second
