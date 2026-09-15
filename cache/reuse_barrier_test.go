// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// errInjectedBarrier is the fault a test forces at the durable reuse/recovery barrier
// (the header msync or the crypto/rand nonce rotation) to prove the fail-closed
// propagation added for issue #154. It is distinct from any production error so a test
// can errors.Is the surfaced error back to exactly the fault it injected.
var errInjectedBarrier = errors.New("cache_test: injected reuse-barrier fault")

// TestReuseBarrierDrainMsyncFailsClosed proves the drain/eviction reuse path fails
// CLOSED: when the barrier msync that makes a drained extent's cleared header durable
// fails, the write that triggered the eviction returns ErrReuseBarrier rather than
// silently landing in a non-durable extent (the pre-#154 behavior logged a durability
// warning and CONTINUED, dropping the pending durable-write obligation). It also
// asserts the failed write is absent — the drained extent was not returned to the
// writable set under a failed barrier.
func TestReuseBarrierDrainMsyncFailsClosed(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	dir := t.TempDir()
	c, err := New(relocMmapConfig(dir, 3, false))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	// Drive to the first eviction with the barrier WORKING, so the fault we inject next
	// lands on a genuine drain-to-empty reuse and not on shard warm-up.
	for n := 0; c.Stats().Evictions == 0; n++ {
		if n > 10_000 {
			t.Fatal("shard never reached its first eviction")
		}
		mustPut(t, c, fmt.Appendf(nil, "warm%06d", n), relocValue(n%256))
	}

	// Inject an msync failure at the reuse/recovery barrier (routed through the
	// msyncTestHook seam by msyncPageHeaderLocked).
	prev := msyncTestHook
	msyncTestHook = func(_ *os.File, _ []byte) error { return errInjectedBarrier }
	defer func() { msyncTestHook = prev }()

	// Keep writing distinct full-size keys until one needs a fresh page and its eviction
	// drains a victim, hitting the faulted barrier. That Put must FAIL CLOSED.
	var failKey []byte
	var putErr error
	for i := 0; i < 10_000; i++ {
		k := fmt.Appendf(nil, "post%06d", i)
		if putErr = c.Put(k, relocValue(i%256), 0); putErr != nil {
			failKey = k
			break
		}
	}
	if !errors.Is(putErr, ErrReuseBarrier) {
		t.Fatalf("Put after a faulted reuse barrier = %v; want it to wrap ErrReuseBarrier "+
			"(pre-fix it logged and continued, returning nil)", putErr)
	}
	if !errors.Is(putErr, errInjectedBarrier) {
		t.Fatalf("Put error does not wrap the injected fault: %v", putErr)
	}
	// The write did not silently succeed: findOrMakePageLocked returned the barrier
	// error before the entry was written, so the key is absent.
	if _, err := c.Get(failKey); err != ErrNotFound {
		t.Fatalf("Get(%q) = %v; want ErrNotFound (a fail-closed write must not have landed)", failKey, err)
	}
}

// TestReuseBarrierPoisonedExtentNotReusedUntilFlushSucceeds closes the drain-path
// reuse residual. A drained extent has its RUNTIME bounds reset before the barrier
// runs, so on a barrier failure the extent shows full FreeTail; without the poison
// flag a LATER independent write would select it (its durable header reset not on
// disk) — the exact window #154 Part 2 must close. Failing only the triggering write
// is not enough.
//
// With the fault still active, the next write must NOT land in the poisoned extent
// (the shard is otherwise full, so it fails closed). Then, once the fault clears, the
// extent must come back into service via the retry-flush — no permanent capacity leak
// on a transient failure.
func TestReuseBarrierPoisonedExtentNotReusedUntilFlushSucceeds(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	dir := t.TempDir()
	c, err := New(relocMmapConfig(dir, 3, false))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	s := c.shards[0]

	// Drive to the first eviction with the barrier working.
	for n := 0; c.Stats().Evictions == 0; n++ {
		if n > 10_000 {
			t.Fatal("shard never reached its first eviction")
		}
		mustPut(t, c, fmt.Appendf(nil, "warm%06d", n), relocValue(n%256))
	}

	// Fault the barrier and force an eviction whose drain-to-empty barrier fails,
	// poisoning the drained (now-empty) victim extent.
	prev := msyncTestHook
	msyncTestHook = func(_ *os.File, _ []byte) error { return errInjectedBarrier }
	var putErr error
	for i := 0; i < 10_000; i++ {
		if putErr = c.Put(fmt.Appendf(nil, "a%06d", i), relocValue(i%256), 0); putErr != nil {
			break
		}
	}
	if !errors.Is(putErr, ErrReuseBarrier) {
		msyncTestHook = prev
		t.Fatalf("Put during barrier fault = %v; want ErrReuseBarrier", putErr)
	}

	// A page must now be poisoned (drained-empty, non-durably-reset).
	s.mu.Lock()
	var poisoned []int
	for i := range s.pages {
		if s.pages[i].reuseBarrierFailed {
			poisoned = append(poisoned, i)
		}
	}
	s.mu.Unlock()
	if len(poisoned) == 0 {
		msyncTestHook = prev
		t.Fatal("no page was poisoned after the faulted drain — the residual guard is not armed")
	}

	// THE RESIDUAL. With the fault STILL active, the next write must not silently land in
	// the poisoned empty extent. The shard is otherwise full, so a correct impl skips the
	// poisoned page and fails closed; the pre-fix impl (no poison flag) writes straight
	// into it and SUCCEEDS.
	residualKey := []byte("residual-key")
	rErr := c.Put(residualKey, relocValue(7), 0)
	if rErr == nil {
		msyncTestHook = prev
		t.Fatal("Put landed in the poisoned extent (its durable reset is not on disk); " +
			"it must be kept out of the writable set until the reset is durable")
	}
	if !errors.Is(rErr, ErrReuseBarrier) {
		msyncTestHook = prev
		t.Fatalf("residual Put = %v; want ErrReuseBarrier (poisoned extent unusable while the fault persists)", rErr)
	}
	if _, gErr := c.Get(residualKey); gErr != ErrNotFound {
		msyncTestHook = prev
		t.Fatalf("Get(residual) = %v; want ErrNotFound (must not have landed anywhere)", gErr)
	}

	// Capture every extent poisoned across the fault window (the residual write forces a
	// second faulted drain), so we can prove each one comes back.
	s.mu.Lock()
	poisoned = poisoned[:0]
	for i := range s.pages {
		if s.pages[i].reuseBarrierFailed {
			poisoned = append(poisoned, i)
		}
	}
	s.mu.Unlock()

	// Clear the fault: the poisoned extents must self-heal via a successful retry-flush
	// the next time a selector wants them, so capacity returns and none stays stranded.
	msyncTestHook = prev
	allCleared := func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, i := range poisoned {
			if s.pages[i].reuseBarrierFailed {
				return false
			}
		}
		return true
	}
	healed := false
	for i := 0; i < 10_000 && !healed; i++ {
		_ = c.Put(fmt.Appendf(nil, "b%06d", i), relocValue(i%256), 0)
		healed = allCleared()
	}
	if !healed {
		t.Fatal("a poisoned extent was never reused after the fault cleared — capacity leak on a transient failure")
	}
}

// TestReuseBarrierNonceRotationFailsClosed proves the SAME fail-closed propagation for
// the crypto/rand half of the barrier: if the per-page nonce cannot be rotated on
// reuse, the extent must stay unavailable and the op must error rather than proceed on
// the OLD nonce (which would leave the torn-writeback forge window open).
func TestReuseBarrierNonceRotationFailsClosed(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	dir := t.TempDir()
	c, err := New(relocMmapConfig(dir, 3, false))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })

	for n := 0; c.Stats().Evictions == 0; n++ {
		if n > 10_000 {
			t.Fatal("shard never reached its first eviction")
		}
		mustPut(t, c, fmt.Appendf(nil, "warm%06d", n), relocValue(n%256))
	}

	// Fault the nonce draw (not the msync) so the barrier fails BEFORE it can rotate.
	prev := randomNonceTestHook
	randomNonceTestHook = func() (uint64, error) { return 0, errInjectedBarrier }
	defer func() { randomNonceTestHook = prev }()

	var failKey []byte
	var putErr error
	for i := 0; i < 10_000; i++ {
		k := fmt.Appendf(nil, "post%06d", i)
		if putErr = c.Put(k, relocValue(i%256), 0); putErr != nil {
			failKey = k
			break
		}
	}
	if !errors.Is(putErr, ErrReuseBarrier) {
		t.Fatalf("Put after a faulted nonce rotation = %v; want it to wrap ErrReuseBarrier", putErr)
	}
	if !errors.Is(putErr, errInjectedBarrier) {
		t.Fatalf("Put error does not wrap the injected nonce fault: %v", putErr)
	}
	if _, err := c.Get(failKey); err != ErrNotFound {
		t.Fatalf("Get(%q) = %v; want ErrNotFound (nonce failure must not proceed into a reuse)", failKey, err)
	}
}

// TestReuseBarrierRecoveryFlushFailsOpen proves the recovery counterpart: when
// rebuildIndexFromPages corrects a page's durable bound (here a corrupt head/tail
// pair) it must make that correction durable, and if the flush fails newShard must
// REFUSE TO OPEN rather than serve a shard whose corrected bound is not on disk (which
// a later crash could revert, resurrecting the stale bytes the correction excluded).
// This is the intentional availability-for-integrity trade.
func TestReuseBarrierRecoveryFlushFailsOpen(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0
	cfg.DisableColdCompaction = true

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
		{[]byte("k1"), []byte("BBBB")},
	})

	// Corrupt page 0's durable TAIL (the second uint32 of its 8-byte header) to an
	// out-of-range value, so recovery rejects the head/tail pair, resets the page, and
	// must FLUSH the (0,0) correction — the flush we fault below.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], 0xFFFFFFF0)
	if _, err := f.WriteAt(b[:], int64(headerSize+4)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	prev := msyncTestHook
	msyncTestHook = func(_ *os.File, _ []byte) error { return errInjectedBarrier }
	defer func() { msyncTestHook = prev }()

	s, err := newShard(cfg, dir, nil)
	if err == nil {
		_ = s.Close()
		t.Fatal("newShard succeeded despite a non-durable recovery correction; want a fail-closed open error")
	}
	if !errors.Is(err, ErrReuseBarrier) {
		t.Fatalf("newShard err = %v; want it to wrap ErrReuseBarrier", err)
	}
}

// TestReuseBarrierRecycleSkipsPublishOnFault proves the online-recycle background tick
// fails closed WITHOUT halting: on a barrier failure it must NOT publish the recycled
// page (leaving it retired and unavailable for a later tick to retry), and it must not
// panic. Log-and-continue is acceptable here precisely because nothing is published.
func TestReuseBarrierRecycleSkipsPublishOnFault(t *testing.T) {
	const S = uint64(1_000_000)
	const quarantine = 80 * time.Millisecond
	c := eligibleOnlineCacheQ(t, 1, 1<<20, 8<<20, quarantine) // 8 pages; skips if !linux
	s := c.shards[0]

	// Fragment then relocate so pages retire (barrier working throughout).
	const nKeys = 12
	keyFor := func(i int) []byte { return fmt.Appendf(nil, "live%03d", i) }
	origFor := func(i int) []byte { return bytes.Repeat([]byte{byte('A' + i)}, 180_000) }
	liveFor := func(i int) []byte { return bytes.Repeat([]byte{byte('a' + i)}, 180_000) }
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt(keyFor(i), origFor(i), 0, S); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	for i := 0; i < nKeys; i++ {
		if err := c.PutAt(keyFor(i), liveFor(i), 0, S); err != nil {
			t.Fatalf("overwrite %d: %v", i, err)
		}
	}
	advanceLogicalClock(c, S)
	relocateToRetirement(t, c, S)
	if c.Stats().OnlinePagesRetired == 0 {
		t.Fatal("no pages retired — cannot exercise recycle")
	}

	// Record which pages are retired so we can prove they STAY retired after a faulted
	// recycle pass.
	s.mu.Lock()
	var retiredIdx []int
	for i, p := range s.pages {
		if p.retired {
			retiredIdx = append(retiredIdx, i)
		}
	}
	s.mu.Unlock()
	if len(retiredIdx) == 0 {
		t.Fatal("expected retired page objects present")
	}

	// Fault the barrier and run a recycle pass PAST the quarantine (so every retired
	// page is otherwise eligible). Nothing must be published.
	prev := msyncTestHook
	msyncTestHook = func(_ *os.File, _ []byte) error { return errInjectedBarrier }
	future := time.Now().Add(2 * quarantine)
	s.mu.Lock()
	recycled := s.compactRecycleRetiredLocked(future, quarantine)
	stillRetired := true
	for _, i := range retiredIdx {
		if !s.pages[i].retired {
			stillRetired = false
		}
	}
	s.mu.Unlock()
	msyncTestHook = prev

	if recycled != 0 {
		t.Fatalf("recycled %d pages under a faulted barrier; want 0 (nothing published on failure)", recycled)
	}
	if got := c.Stats().OnlinePagesRecycled; got != 0 {
		t.Fatalf("OnlinePagesRecycled = %d; want 0 (a faulted recycle must not count a publish)", got)
	}
	if !stillRetired {
		t.Fatal("a retired page was published/cleared despite the barrier failure; it must stay retired for a later tick")
	}
}

// TestReuseBarrierRecoveryFlushFailsOpenWithColdCompaction is the cold-compaction-
// ENABLED counterpart of TestReuseBarrierRecoveryFlushFailsOpen (which disables it).
// With cold compaction enabled, newShard runs rebuildIndexFromPages and then
// compactAtOpen -> remapPagesFile, both of which now propagate a recovery-flush error.
// A corrupt page bound is corrected (and its flush faulted) by the FIRST rebuild, so
// the open must still fail closed with ErrReuseBarrier and, crucially, must NOT rotate
// the file aside (a barrier failure is a transient durability fault, not a bad file):
// a .bad-* sibling would discard recoverable data, and the intact file must remain for
// a retry once the fault clears.
func TestReuseBarrierRecoveryFlushFailsOpenWithColdCompaction(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20
	cfg.TTLSweepIntervalMs = 0
	cfg.DisableColdCompaction = false // the point of this variant: exercise the open path with compaction on

	dir := t.TempDir()
	path := filepath.Join(dir, "pages.dat")
	writeCurrentFileWithEntries(t, path, cfg.PageSize, 1, [][2][]byte{
		{[]byte("k0"), []byte("AAAA")},
		{[]byte("k1"), []byte("BBBB")},
	})

	// Corrupt page 0's durable TAIL to an out-of-range value so recovery rejects the
	// head/tail pair, resets the page, and must FLUSH the (0,0) correction — the flush
	// we fault.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], 0xFFFFFFF0)
	if _, err := f.WriteAt(b[:], int64(headerSize+4)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	prev := msyncTestHook
	msyncTestHook = func(_ *os.File, _ []byte) error { return errInjectedBarrier }
	defer func() { msyncTestHook = prev }()

	s, err := newShard(cfg, dir, nil)
	if err == nil {
		_ = s.Close()
		t.Fatal("newShard succeeded with cold compaction enabled despite a non-durable recovery correction; want fail-closed")
	}
	if !errors.Is(err, ErrReuseBarrier) {
		t.Fatalf("newShard err = %v; want it to wrap ErrReuseBarrier", err)
	}
	// The file must be left intact — a durability fault is not a bad file, so it is not
	// rotated aside; a retry after the fault clears must find it.
	bad, _ := filepath.Glob(filepath.Join(dir, "pages.dat.bad-*"))
	if len(bad) != 0 {
		t.Fatalf("file was rotated aside on a barrier fault (%v); a transient durability fault must leave it intact", bad)
	}
	if _, statErr := os.Stat(path); statErr != nil {
		t.Fatalf("pages.dat missing after a failed open: %v", statErr)
	}
}
