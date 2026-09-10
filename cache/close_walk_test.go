// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// walkTestCache builds an mmap-backed cache — mmap is the mode where the race
// this file is about is fatal rather than merely wrong, because Close unmaps the
// region a walk is reading out of.
func walkTestCache(t *testing.T, shards int) *Cache {
	t.Helper()
	cfg := DefaultConfig()
	cfg.NumShards = shards
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 16 << 20
	cfg.TTLSweepIntervalMs = 0
	cfg.MsyncIntervalMs = 1000
	// The temp dir is allocated BEFORE any cleanup that could tear the cache
	// down, so the directory outlives every goroutine still touching it.
	cfg.DataDir = t.TempDir()
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func fillWalkTestCache(t *testing.T, c *Cache, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if err := c.Put(fmt.Appendf(nil, "k%07d", i), fmt.Appendf(nil, "v%07d", i), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
}

// A walk that starts after Close must refuse rather than read the unmapped
// region. This is the deterministic half of the guarantee; the racing half is
// below.
func TestWalkAfterCloseReportsErrClosed(t *testing.T) {
	c := walkTestCache(t, 2)
	fillWalkTestCache(t, c, 200)
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	visited := 0
	if err := c.IterateChunked(16, func(_, _ []byte) bool { visited++; return true }); !errors.Is(err, ErrClosed) {
		t.Fatalf("IterateChunked after Close: err = %v, want ErrClosed", err)
	}
	if visited != 0 {
		t.Fatalf("IterateChunked after Close read %d entries out of an unmapped region", visited)
	}
	if err := c.Iterate(func(_, _ []byte, _ uint64) bool { return true }); !errors.Is(err, ErrClosed) {
		t.Fatalf("Iterate after Close: err = %v, want ErrClosed", err)
	}
}

// The real shape: a chunked walk is in flight, and the cache is closed while it
// is between chunks. It must come back with ErrClosed and no segfault. Run under
// -race with repeats — the window is the gap between one chunk's RUnlock and the
// next chunk's RLock, which a small batch makes frequent.
func TestCloseDuringChunkedWalkIsNotAUseAfterUnmap(t *testing.T) {
	for attempt := 0; attempt < 16; attempt++ {
		c := walkTestCache(t, 4)
		fillWalkTestCache(t, c, 4000)

		started := make(chan struct{})
		var once sync.Once
		var walkErr error
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			walkErr = c.IterateChunked(1, func(_, _ []byte) bool {
				once.Do(func() { close(started) })
				return true
			})
		}()

		<-started
		if err := c.Close(); err != nil {
			t.Fatalf("attempt %d: Close: %v", attempt, err)
		}
		wg.Wait()

		// Either the walk finished before the close landed (nil) or it was cut
		// short by it (ErrClosed). Any other error, or a crash, is the bug.
		if walkErr != nil && !errors.Is(walkErr, ErrClosed) {
			t.Fatalf("attempt %d: walk error = %v, want nil or ErrClosed", attempt, walkErr)
		}
	}
}

// Close must not unmap while a reader is inside the mapping. The walk here holds
// the read lock (it is inside fn) when Close is called, so Close has to wait for
// it — proving the unmap is serialised against readers rather than merely
// flagged.
func TestCloseWaitsForAWalkInsideTheMapping(t *testing.T) {
	c := walkTestCache(t, 1)
	fillWalkTestCache(t, c, 500)

	inside := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = c.IterateChunked(1024, func(_, _ []byte) bool {
			once.Do(func() { close(inside); <-release })
			return true
		})
	}()

	<-inside
	closed := make(chan struct{})
	go func() { defer close(closed); _ = c.Close() }()

	select {
	case <-closed:
		t.Fatal("Close unmapped the region while a walk was inside it")
	case <-time.After(100 * time.Millisecond):
	}
	close(release)
	<-done
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not complete after the walk let go")
	}
}
