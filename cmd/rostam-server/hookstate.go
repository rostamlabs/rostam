// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Hooks are separate processes, so "already injected in this session" lives in
// a small file per session id under the hook state directory. An empty session
// id means no dedupe: nothing is read and nothing is written.

// stateFile hashes the session id so an attacker-influenced id can never
// escape the state directory or collide on a sanitised form.
func stateFile(dir, sessionID string) string {
	sum := sha256.Sum256([]byte(sessionID))
	return filepath.Join(dir, hex.EncodeToString(sum[:16])+".json")
}

// loadInjected returns the memory ids already injected for sessionID. Any
// problem reads as "none injected": a duplicate injection is better than none.
func loadInjected(dir, sessionID string) map[uint64]bool {
	seen := map[uint64]bool{}
	if dir == "" || sessionID == "" {
		return seen
	}
	b, err := os.ReadFile(stateFile(dir, sessionID))
	if err != nil {
		return seen
	}
	var ids []uint64
	if json.Unmarshal(b, &ids) != nil {
		return seen
	}
	for _, id := range ids {
		seen[id] = true
	}
	return seen
}

// saveInjected records ids for sessionID (mode 0600, replaced atomically). It
// is best effort: failures are ignored because the hook must stay silent.
func saveInjected(dir, sessionID string, ids map[uint64]bool) {
	if dir == "" || sessionID == "" {
		return
	}
	list := make([]uint64, 0, len(ids))
	for id := range ids {
		list = append(list, id)
	}
	b, err := json.Marshal(list)
	if err != nil {
		return
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*") // CreateTemp creates 0600
	if err != nil {
		return
	}
	_, werr := tmp.Write(b)
	cerr := tmp.Close()
	if werr != nil || cerr != nil || os.Rename(tmp.Name(), stateFile(dir, sessionID)) != nil {
		_ = os.Remove(tmp.Name())
	}
}

// hookStateMaxAge is how long an untouched session's state files are kept.
// Claude Code gives no "session ended" signal, so finished sessions' files
// would otherwise pile up forever; a session idle this long that comes back
// at worst sees its memories injected once more.
const hookStateMaxAge = 30 * 24 * time.Hour

// pruneHookState removes session state and lock files under dir not modified
// within maxAge of now. Best effort and silent, like everything in the hook.
// A lock file's mtime never moves (it is only opened), so it goes only when
// its session's state file is gone or stale too: a long-lived session that
// is still injecting keeps the lock its concurrent hooks share.
func pruneHookState(dir string, maxAge time.Duration, now time.Time) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	old := func(name string) bool {
		fi, err := os.Stat(filepath.Join(dir, name))
		return err == nil && now.Sub(fi.ModTime()) > maxAge
	}
	for _, e := range entries {
		name := e.Name()
		switch {
		case e.IsDir():
			continue
		case strings.HasSuffix(name, ".json.lock"):
			state := strings.TrimSuffix(name, ".lock")
			if _, err := os.Stat(filepath.Join(dir, state)); err == nil && !old(state) {
				continue
			}
		case strings.HasSuffix(name, ".json"), strings.HasPrefix(name, ".tmp-"):
		default:
			continue
		}
		if old(name) {
			_ = os.Remove(filepath.Join(dir, name))
		}
	}
}

// resetInjected forgets everything injected for sessionID.
func resetInjected(dir, sessionID string) {
	if dir == "" || sessionID == "" {
		return
	}
	_ = os.Remove(stateFile(dir, sessionID))
}
