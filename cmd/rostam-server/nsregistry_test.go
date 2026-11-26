// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func regID(identity, base string) projectID { return projectID{identity: identity, base: base} }

func mustNS(t *testing.T, dir string, id projectID) string {
	t.Helper()
	ns, err := projectNamespace(context.Background(), dir, id)
	if err != nil {
		t.Fatalf("projectNamespace(%s): %v", id.identity, err)
	}
	return ns
}

// First comer gets the plain name, a different identity with the same name
// gets the hashed one, and both are sticky.
func TestRegistryAssignsAndSticks(t *testing.T) {
	dir := t.TempDir()
	a := regID(filepath.Join(t.TempDir(), "api"), "api")
	b := regID(filepath.Join(t.TempDir(), "api"), "api")
	if err := os.MkdirAll(a.identity, 0o700); err != nil {
		t.Fatal(err)
	}
	if ns := mustNS(t, dir, a); ns != "api" {
		t.Fatalf("first project = %q, want api", ns)
	}
	nb := mustNS(t, dir, b)
	if nb != hashedNamespace(b) || !strings.HasPrefix(nb, "api-") || len(nb) != len("api-")+8 {
		t.Fatalf("second same-named project = %q, want %q", nb, hashedNamespace(b))
	}
	if ns := mustNS(t, dir, a); ns != "api" {
		t.Fatalf("first project again = %q, want api (sticky)", ns)
	}
	if ns := mustNS(t, dir, b); ns != nb {
		t.Fatalf("second project again = %q, want %q (sticky)", ns, nb)
	}
	fi, err := os.Stat(filepath.Join(dir, nsRegistryName))
	if err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("registry file: %v, mode %v; want 0600", err, fi.Mode().Perm())
	}
	// Sticky even after the owner of "api" disappears: b keeps its name.
	if err := os.RemoveAll(a.identity); err != nil {
		t.Fatal(err)
	}
	if ns := mustNS(t, dir, b); ns != nb {
		t.Fatalf("assigned project after the owner vanished = %q, want %q", ns, nb)
	}
}

// An identity with no assignment takes the plain name over only when the
// recorded owner's path is gone (ENOENT), not on any other stat error.
func TestRegistryTakeover(t *testing.T) {
	dir := t.TempDir()
	parent := t.TempDir()
	owner := regID(filepath.Join(parent, "repo", "api"), "api")
	if err := os.MkdirAll(owner.identity, 0o700); err != nil {
		t.Fatal(err)
	}
	mustNS(t, dir, owner)

	// Owner unreachable (EACCES), not gone: no takeover.
	if os.Getuid() != 0 {
		if err := os.Chmod(filepath.Join(parent, "repo"), 0); err != nil {
			t.Fatal(err)
		}
		c := regID(filepath.Join(t.TempDir(), "api"), "api")
		if ns := mustNS(t, dir, c); ns != hashedNamespace(c) {
			t.Fatalf("owner unreadable: newcomer got %q, want the hashed name", ns)
		}
		if err := os.Chmod(filepath.Join(parent, "repo"), 0o700); err != nil {
			t.Fatal(err)
		}
	}

	// Owner deleted: the next unassigned identity takes the plain name.
	if err := os.RemoveAll(owner.identity); err != nil {
		t.Fatal(err)
	}
	d := regID(filepath.Join(t.TempDir(), "api"), "api")
	if err := os.MkdirAll(d.identity, 0o700); err != nil {
		t.Fatal(err)
	}
	if ns := mustNS(t, dir, d); ns != "api" {
		t.Fatalf("owner gone: newcomer got %q, want api", ns)
	}
	// The old owner, if it comes back, no longer has the plain name.
	if err := os.MkdirAll(owner.identity, 0o700); err != nil {
		t.Fatal(err)
	}
	if ns := mustNS(t, dir, owner); ns != hashedNamespace(owner) {
		t.Fatalf("returning former owner got %q, want the hashed name", ns)
	}
}

// Many first sessions of same-named projects at once: exactly one gets the
// plain name, and every identity ends up with one stable name.
func TestRegistryConcurrentFirstSessions(t *testing.T) {
	dir := t.TempDir()
	const n = 16
	ids := make([]projectID, n)
	for i := range ids {
		ids[i] = regID(filepath.Join(t.TempDir(), "api"), "api")
		if err := os.MkdirAll(ids[i].identity, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	got := make([]string, n)
	var wg sync.WaitGroup
	for i := range ids {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got[i] = mustNS(t, dir, ids[i])
		}(i)
	}
	wg.Wait()
	plain := 0
	for i, ns := range got {
		if ns == "api" {
			plain++
		} else if ns != hashedNamespace(ids[i]) {
			t.Errorf("identity %d got %q", i, ns)
		}
		if again := mustNS(t, dir, ids[i]); again != ns {
			t.Errorf("identity %d: %q then %q", i, ns, again)
		}
	}
	if plain != 1 {
		t.Fatalf("%d projects got the plain name, want exactly 1", plain)
	}
}

// A corrupt or unreadable registry never hands out the plain name and is
// never overwritten.
func TestRegistryUnusableFallsBackToHashed(t *testing.T) {
	id := regID(filepath.Join(t.TempDir(), "api"), "api")

	dir := t.TempDir()
	path := filepath.Join(dir, nsRegistryName)
	corrupt := []byte("{not json")
	if err := os.WriteFile(path, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	ns, err := projectNamespace(context.Background(), dir, id)
	if err == nil || ns != hashedNamespace(id) || !strings.Contains(err.Error(), "corrupt") {
		t.Fatalf("corrupt registry: %q, %v; want the hashed name and a corrupt warning", ns, err)
	}
	if b, _ := os.ReadFile(path); string(b) != string(corrupt) {
		t.Fatalf("corrupt registry was rewritten: %q", b)
	}

	if os.Getuid() != 0 {
		dir2 := t.TempDir()
		p2 := filepath.Join(dir2, nsRegistryName)
		if err := os.WriteFile(p2, []byte(`{"version":1,"projects":{}}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(p2, 0); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(p2, 0o600) })
		if ns, err := projectNamespace(context.Background(), dir2, id); err == nil || ns != hashedNamespace(id) {
			t.Fatalf("unreadable registry: %q, %v; want the hashed name and a warning", ns, err)
		}
	}

	if ns, err := projectNamespace(context.Background(), "", id); err == nil || ns != hashedNamespace(id) {
		t.Fatalf("no registry dir: %q, %v; want the hashed name and a warning", ns, err)
	}
}

// A registry held past the deadline decides nothing: errRegistryBusy.
func TestRegistryBusy(t *testing.T) {
	if !flockSupported {
		t.Skip("no flock on this platform")
	}
	dir := t.TempDir()
	unlock, err := lockFileCtx(context.Background(), filepath.Join(dir, nsRegistryLockName))
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 50e6)
	defer cancel()
	if ns, err := projectNamespace(ctx, dir, regID("/x/api", "api")); !errors.Is(err, errRegistryBusy) || ns != "" {
		t.Fatalf("busy registry: %q, %v; want errRegistryBusy", ns, err)
	}
}

// Identity: symlinked paths and linked worktrees resolve to one identity; two
// repositories with separate git dirs and the same directory name stay
// distinct, and each one's worktrees share its identity.
func TestIdentifyProject(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	ctx := context.Background()
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(base, "one", "api")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	run(t, repo, "git", "init", "-q")
	run(t, repo, "git", "commit", "-q", "--allow-empty", "-m", "init")
	link := filepath.Join(base, "link")
	if err := os.Symlink(filepath.Join(base, "one"), link); err != nil {
		t.Skipf("symlink: %v", err)
	}
	wt := filepath.Join(base, "api-feature")
	run(t, repo, "git", "worktree", "add", "-q", wt)

	_, want := identifyProject(ctx, repo)
	if want.identity != repo || want.base != "api" {
		t.Fatalf("repo identity = %+v", want)
	}
	for _, d := range []string{filepath.Join(link, "api"), wt} {
		if _, got := identifyProject(ctx, d); got != want {
			t.Errorf("identifyProject(%s) = %+v, want %+v", d, got, want)
		}
	}

	// Separate git dirs: same checkout name, different repositories.
	var seps []projectID
	for _, n := range []string{"two", "three"} {
		co := filepath.Join(base, n, "api")
		if err := os.MkdirAll(co, 0o755); err != nil {
			t.Fatal(err)
		}
		run(t, co, "git", "init", "-q", "--separate-git-dir", filepath.Join(base, n+"-gitdir"))
		run(t, co, "git", "commit", "-q", "--allow-empty", "-m", "init")
		_, id := identifyProject(ctx, co)
		if id.base != "api" || id.identity != filepath.Join(base, n+"-gitdir") {
			t.Fatalf("separate git dir identity = %+v", id)
		}
		w := filepath.Join(base, n+"-wt")
		run(t, co, "git", "worktree", "add", "-q", w)
		if _, wid := identifyProject(ctx, w); wid.identity != id.identity {
			t.Errorf("worktree of a separate-git-dir repo split: %+v vs %+v", wid, id)
		}
		seps = append(seps, id)
	}
	if seps[0].identity == seps[1].identity || seps[0].identity == want.identity {
		t.Fatalf("distinct repositories collapsed: %+v", seps)
	}

	// Outside git: the canonical directory.
	plain := filepath.Join(base, "plain")
	if err := os.MkdirAll(plain, 0o755); err != nil {
		t.Fatal(err)
	}
	if root, id := identifyProject(ctx, plain); root != plain || id.identity != plain || id.base != "plain" {
		t.Fatalf("non-git: root %s id %+v", root, id)
	}
}

// withRegistryDir points the registry at dir for one test.
func withRegistryDir(t *testing.T, dir string) {
	orig := nsRegistryDir
	t.Cleanup(func() { nsRegistryDir = orig })
	nsRegistryDir = func() (string, error) { return dir, nil }
}

// gitRepoNamed makes a git checkout <tmp>/<name> with a.go in it.
func gitRepoNamed(t *testing.T, name string) string {
	t.Helper()
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(base, name)
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	run(t, root, "git", "init", "-q")
	if err := os.WriteFile(filepath.Join(root, "a.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// Two unrelated checkouts named "api": mcp and the hook agree on each one's
// namespace, and a memory saved in one is never injected in the other.
func TestSameNamedProjectsDoNotShareMemory(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	withRegistryDir(t, t.TempDir())
	a, b := gitRepoNamed(t, "api"), gitRepoNamed(t, "api")
	noEnv := func(string) string { return "" }
	nsFor := func(dir string) string {
		_, nsDir := defaultProjectRoot("", noEnv, func() (string, error) { return dir, nil })
		return mcpProjectNamespace(nsDir, func(m string) { t.Error(m) })
	}
	nsA, nsB := nsFor(a), nsFor(b)
	if nsA != "api" || nsB == "api" || !strings.HasPrefix(nsB, "api-") {
		t.Fatalf("namespaces: a=%q b=%q; want a=api and b hashed", nsA, nsB)
	}

	st := hookHeapStore(t)
	hookRemember(t, st, a, map[string]any{"content": "a-only fact", "namespace": nsA, "path": "a.go"})
	for _, c := range []struct {
		root, ns string
		want     bool
	}{{a, nsA, true}, {b, nsB, false}} {
		e := hookEnv{st: st, root: c.root, state: t.TempDir()}
		d := e.deps("pre-tool-use")
		if _, ns, ok := hookProject(context.Background(), d, c.root); !ok || ns != c.ns {
			t.Fatalf("hook namespace for %s = %q (ok %v), mcp's is %q", c.root, ns, ok, c.ns)
		}
		out := runHookStr(t, d, e.payload("s", "Read", filepath.Join(c.root, "a.go")))
		if got := strings.Contains(out, "a-only fact"); got != c.want {
			t.Fatalf("hook in %s injected a's memory = %v, want %v (%q)", c.root, got, c.want, out)
		}
	}
}

// While the registry is held past the hook's budget, the hook stays silent
// (it does not guess); mcp falls back to the hashed name and warns.
func TestRegistryBusyHookSilentMcpWarns(t *testing.T) {
	if !flockSupported {
		t.Skip("no flock on this platform")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	dir := t.TempDir()
	withRegistryDir(t, dir)
	root := gitRepoNamed(t, "busy")
	unlock, err := lockFileCtx(context.Background(), filepath.Join(dir, nsRegistryLockName))
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()

	e := hookEnv{st: hookHeapStore(t), root: root, state: t.TempDir()}
	d := e.deps("pre-tool-use")
	d.budget = 200e6
	if out := runHookStr(t, d, e.payload("s", "Read", filepath.Join(root, "a.go"))); out != "" {
		t.Fatalf("hook with a busy registry wrote %q", out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100e6)
	defer cancel()
	_, ns, err := resolveNamespace(ctx, root)
	if !errors.Is(err, errRegistryBusy) || ns != hashedNamespace(projectID{identity: root, base: "busy"}) {
		t.Fatalf("resolveNamespace busy = %q, %v; want the hashed fallback and errRegistryBusy", ns, err)
	}
}

// A corrupt registry: mcp warns, and mcp and the hook agree on the hashed name.
func TestCorruptRegistryMcpAndHookAgree(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	dir := t.TempDir()
	withRegistryDir(t, dir)
	if err := os.WriteFile(filepath.Join(dir, nsRegistryName), []byte("garbage"), 0o600); err != nil {
		t.Fatal(err)
	}
	root := gitRepoNamed(t, "proj")
	var warned string
	ns := mcpProjectNamespace(root, func(m string) { warned = m })
	if !strings.Contains(warned, "corrupt") || ns == "proj" {
		t.Fatalf("mcp: ns %q, warning %q; want a hashed name and a corrupt warning", ns, warned)
	}
	e := hookEnv{st: hookHeapStore(t), root: root, state: t.TempDir()}
	if _, hns, ok := hookProject(context.Background(), e.deps("pre-tool-use"), root); !ok || hns != ns {
		t.Fatalf("hook namespace %q (ok %v), mcp's %q", hns, ok, ns)
	}
}
