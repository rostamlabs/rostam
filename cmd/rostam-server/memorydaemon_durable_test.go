// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam/mcp"
)

// noDaemonAddr is never dialled: every test below injects its probe.
const noDaemonAddr = "127.0.0.1:1"

// holdDurableMarker takes the marker lock in data as a durable daemon's server
// would, and returns its release.
func holdDurableMarker(t *testing.T, data string) func() error {
	t.Helper()
	dir := filepath.Join(data, daemonDurableDir)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	unlock, err := lockDataDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	return unlock
}

func TestDaemonIsDurable(t *testing.T) {
	data := t.TempDir()
	if ok, err := daemonIsDurable(data); ok || err != nil {
		t.Fatalf("no marker dir: daemonIsDurable = %v, %v; want false, nil", ok, err)
	}
	if err := os.MkdirAll(filepath.Join(data, daemonDurableDir), 0o700); err != nil {
		t.Fatal(err)
	}
	if ok, err := daemonIsDurable(data); ok || err != nil {
		t.Fatalf("empty marker dir: daemonIsDurable = %v, %v; want false, nil", ok, err)
	}

	unlock := holdDurableMarker(t, data)
	if ok, err := daemonIsDurable(data); !ok || err != nil {
		t.Fatalf("marker held: daemonIsDurable = %v, %v; want true, nil", ok, err)
	}
	if err := unlock(); err != nil {
		t.Fatal(err)
	}
	if ok, err := daemonIsDurable(data); ok || err != nil {
		t.Fatalf("marker released: daemonIsDurable = %v, %v; want false, nil", ok, err)
	}
	// The check itself must not keep the lock.
	again, err := lockDataDir(filepath.Join(data, daemonDurableDir))
	if err != nil {
		t.Fatalf("lock after the checks: %v", err)
	}
	_ = again()
}

// oldDaemon is a daemon that answers until it is stopped.
type oldDaemon struct {
	stopped atomic.Bool
	stops   atomic.Int32
	stopErr error
}

func (d *oldDaemon) probe(context.Context, string, string) error {
	if d.stopped.Load() {
		return errors.New("connection refused")
	}
	return nil
}

func (d *oldDaemon) stop() error {
	d.stops.Add(1)
	if d.stopErr != nil {
		return d.stopErr
	}
	d.stopped.Store(true)
	return nil
}

type warnings struct {
	mu   sync.Mutex
	msgs []string
}

func (w *warnings) warn(s string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.msgs = append(w.msgs, s)
}

func (w *warnings) all() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return strings.Join(w.msgs, "\n")
}

func TestRetireStopsOldDaemonOnce(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	d := &oldDaemon{}
	var w warnings

	const n = 8
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = retireNonDurableDaemon(context.Background(), data, noDaemonAddr, d.probe, d.stop, w.warn)
		}()
	}
	wg.Wait()
	for i, e := range errs {
		if e != nil {
			t.Errorf("session %d: %v", i, e)
		}
	}
	if got := d.stops.Load(); got != 1 {
		t.Fatalf("stop called %d times, want exactly 1", got)
	}
	if strings.Contains(w.all(), "memory-daemon -stop") {
		t.Fatalf("a clean stop warned as if it failed: %q", w.all())
	}
}

func TestRetireLeavesDurableDaemon(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	unlock := holdDurableMarker(t, data)
	defer func() { _ = unlock() }()
	d := &oldDaemon{}
	var w warnings
	if err := retireNonDurableDaemon(context.Background(), data, noDaemonAddr, d.probe, d.stop, w.warn); err != nil {
		t.Fatal(err)
	}
	if got := d.stops.Load(); got != 0 {
		t.Fatalf("stop called %d times on a durable daemon, want 0", got)
	}
}

func TestRetireNoDaemon(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	d := &oldDaemon{}
	d.stopped.Store(true) // nothing answers
	var w warnings
	if err := retireNonDurableDaemon(context.Background(), data, noDaemonAddr, d.probe, d.stop, w.warn); err != nil {
		t.Fatal(err)
	}
	if got := d.stops.Load(); got != 0 {
		t.Fatalf("stop called %d times with no daemon, want 0", got)
	}
}

func TestRetireWarnsWhenStopFails(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	d := &oldDaemon{stopErr: errors.New("refusing to signal: the pids are not a rostam-server memory daemon")}
	var w warnings
	if err := retireNonDurableDaemon(context.Background(), data, noDaemonAddr, d.probe, d.stop, w.warn); err != nil {
		t.Fatalf("a failed stop must not fail the session: %v", err)
	}
	if got := d.stops.Load(); got != 1 {
		t.Fatalf("stop called %d times, want 1", got)
	}
	if msg := w.all(); !strings.Contains(msg, "memory-daemon -stop") || !strings.Contains(msg, "refusing to signal") {
		t.Fatalf("warning %q must name `rostam-server memory-daemon -stop` and the stop error", msg)
	}
}

// Only a context error ends the session; a cancelled one stops nothing.
func TestRetireReturnsContextError(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	// A sibling session holds the spawn lock, so retire has to wait for it.
	if err := os.MkdirAll(data, 0o700); err != nil {
		t.Fatal(err)
	}
	unlock, err := claimDataDir(filepath.Join(filepath.Dir(data), spawnLockDirName))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = unlock() }()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	d := &oldDaemon{}
	var w warnings
	if err := retireNonDurableDaemon(ctx, data, noDaemonAddr, d.probe, d.stop, w.warn); !errors.Is(err, context.Canceled) {
		t.Fatalf("retire = %v, want context.Canceled", err)
	}
	if got := d.stops.Load(); got != 0 {
		t.Fatalf("stop called %d times after cancellation, want 0", got)
	}
}

func TestEmbeddedDurableCollections(t *testing.T) {
	if got := embeddedDurableCollections("/some/data"); len(got) != 1 || got[0] != mcp.MemoryCollection {
		t.Fatalf("persistent embedded store: durable collections %q, want [%s]", got, mcp.MemoryCollection)
	}
	if got := embeddedDurableCollections(""); got != nil {
		t.Fatalf("heap store: durable collections %q, want none", got)
	}
}

// A cancelled context stops nothing even when no sibling holds the spawn lock
// and the old daemon answers.
func TestRetireCancelledUncontendedStopsNothing(t *testing.T) {
	data := filepath.Join(t.TempDir(), "memory")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	d := &oldDaemon{}
	var w warnings
	if err := retireNonDurableDaemon(ctx, data, noDaemonAddr, d.probe, d.stop, w.warn); !errors.Is(err, context.Canceled) {
		t.Fatalf("retire = %v, want context.Canceled", err)
	}
	if got := d.stops.Load(); got != 0 {
		t.Fatalf("stop called %d times after cancellation, want 0", got)
	}
}
