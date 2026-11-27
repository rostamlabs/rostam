// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strings"
	"testing"
)

// A listed durable name only gets the fsynced write-ahead log as a dense
// collection on a store with a data directory. Anything else is not silently
// left unprotected: creating it, and opening it, logs a warning naming it. A
// named or multi-vector collection is told the option does not apply (it keeps
// its own log settings, which may sync); the others are told they are not
// crash-safe.

func assertDurableNotAppliedWarning(t *testing.T, logs *syncBuffer, name, when string) {
	t.Helper()
	out := logs.String()
	if !strings.Contains(out, "does not apply") || !strings.Contains(out, name) {
		t.Fatalf("%s: no does-not-apply warning naming %q:\n%s", when, name, out)
	}
}

func assertNotCrashSafeWarning(t *testing.T, logs *syncBuffer, name, when string) {
	t.Helper()
	out := logs.String()
	if !strings.Contains(out, "not crash-safe") || !strings.Contains(out, name) {
		t.Fatalf("%s: no not crash-safe warning naming %q:\n%s", when, name, out)
	}
}

func TestListedNamedOrMultiVectorWarnsAtCreateAndOpen(t *testing.T) {
	for _, kind := range []string{"named", "multi-vector"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			logs := captureLogs(t)
			cs := openDurable(t, dir, "listed")
			var err error
			if kind == "named" {
				err = cs.CreateCollection("listed", namedWALConfig())
			} else {
				err = cs.CreateMultiVector("listed", mvWALConfig())
			}
			if err != nil {
				t.Fatal(err)
			}
			assertDurableNotAppliedWarning(t, logs, "listed", "create")
			if err := cs.Close(); err != nil {
				t.Fatal(err)
			}

			logs = captureLogs(t)
			re := openDurable(t, dir, "listed")
			defer func() { _ = re.Close() }()
			assertDurableNotAppliedWarning(t, logs, "listed", "open")
		})
	}
}

func TestListedDenseInHeapModeStoreWarnsAtCreate(t *testing.T) {
	logs := captureLogs(t)
	cs, err := OpenCollectionStoreWith("", StoreOptions{DurableCollections: []string{"listed"}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("listed", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	assertNotCrashSafeWarning(t, logs, "listed", "create")
}

func TestListedDenseOnDiskDoesNotWarn(t *testing.T) {
	logs := captureLogs(t)
	cs := openDurable(t, t.TempDir(), "listed")
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("listed", durablePersistentCfg()); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(logs.String(), "not crash-safe") {
		t.Fatalf("a listed dense collection on disk warned:\n%s", logs.String())
	}
}
