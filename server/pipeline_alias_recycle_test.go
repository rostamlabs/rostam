// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// aliasRecycleDispatcher models the exact hazard the Critical fix addresses: like
// handleGet on a reject-writes/mmap shard, its Call returns a []byte that ALIASES
// caller-owned backing bytes (here backings[i], one per request in dispatch order)
// verbatim — no copy. The test later OVERWRITES those backing bytes to simulate the
// online compactor recycling the retired page's extent while a response is still in
// flight. If the server's writer holds the raw alias across the (stalled) flush, the
// client receives the overwritten bytes: a torn read.
type aliasRecycleDispatcher struct {
	mu         sync.Mutex
	backings   [][]byte
	next       atomic.Int64
	dispatched chan struct{} // one send per Call, buffered to len(backings)
}

func (d *aliasRecycleDispatcher) Call(_ string, args []byte) ([]byte, error) {
	// Select the backing by the request's key byte, NOT a dispatch counter: pipelined
	// requests dispatch CONCURRENTLY, so a counter would not match request (== response)
	// order. args is the decoded key []byte{byte(i)}.
	i := int(args[0])
	d.next.Add(1)
	d.mu.Lock()
	b := d.backings[i] // ALIAS returned verbatim, exactly like the cache's zero-copy Get
	d.mu.Unlock()
	d.dispatched <- struct{}{}
	return b, nil
}

func (d *aliasRecycleDispatcher) LeaderAddr() string { return "" }

// TestPipelinedSlowReaderRecycleIntegrity reproduces the Critical scenario and proves
// Fix 1: a client pipelines several large gets and then reads SLOWLY, so the ordered
// writer stalls with responses still queued; meanwhile the backing bytes those
// responses alias are recycled (overwritten). Every value the client eventually
// receives must still match what was dispatched.
//
// WHY IT IS DETERMINISTIC. The requests are tiny and sent in a SINGLE write, so they
// all arrive buffered together and every one takes the PIPELINED path (the reader sees
// r.Buffered() > 0 / outstanding > 0). The responses are far larger than any socket +
// bufio buffer, and the client does not read until after the recycle, so the writer
// cannot deliver them and blocks in Flush with the later responses still queued. Fix 1
// copies each payload out of the alias at dispatch time — BEFORE the connPipeResp is
// enqueued — so once all Calls have returned (we wait for the dispatch signals, plus a
// margin) every queued response references an OWNED copy and the subsequent overwrite
// cannot reach it. WITHOUT Fix 1 the queued responses still point at the backing
// bytes, so the overwrite turns them into the 0xFF sentinel and the integrity check
// fails. Confirmed to FAIL on the pre-fix code (payload enqueued as the raw alias).
func TestPipelinedSlowReaderRecycleIntegrity(t *testing.T) {
	if raceEnabled {
		// This test DELIBERATELY collapses the alias-drain fence: it overwrites the
		// backing bytes while a response is mid-flight to prove Fix 1's copy defuses the
		// hazard. That is an unsynchronized reader/writer overlap by construction — the
		// production recycle is fenced from readers by AliasQuarantine (a WALL-CLOCK
		// window), not a lock, so -race flags the intentional overlap (the copy reading
		// the backing vs the test overwriting it) even though the copy provably precedes
		// the overwrite. The compaction/recycle path itself has proper -race coverage in
		// package cache (TestOnlineRecycleHeavyRace); skip the temporal-fence reproduction
		// under the detector.
		t.Skip("temporal-fence reproduction; unsynchronized by design — see comment")
	}
	const (
		n       = 3
		valLen  = 12 << 20 // >> any socket+bufio buffer, so the writer genuinely stalls mid-flush with responses still aliasing their backings (a smaller value the loopback buffers absorb whole, and the hold never materializes)
		sentTag = 0xFF     // recycle sentinel; no original byte uses it
	)

	backings := make([][]byte, n)
	want := make([][]byte, n)
	for i := range backings {
		b := make([]byte, valLen)
		for j := range b {
			b[j] = byte(i*31 + j) // per-response varying pattern; never all-0xFF
		}
		backings[i] = b
		want[i] = append([]byte(nil), b...) // pristine copy to compare against
	}

	d := &aliasRecycleDispatcher{backings: backings, dispatched: make(chan struct{}, n)}
	// EnableOnlineCompaction ON: this is the opted-in configuration where recycle can
	// occur, so the pipelined path must copy each payload out of its alias. The default
	// (flag off) path enqueues the raw alias and would (correctly) fail this recycle
	// simulation because nothing recycles in that mode — see TestPipelinedDefaultNoCopy.
	// n*valLen = 36 MiB fits under connPipelineByteBudget (64 MiB), so all n admissions
	// proceed without byte-budget backpressure and the stall reproduction is unchanged.
	addr, stop := startPipelineTestServerCompaction(t, d, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = c.Close() }()

	// Send all n get requests in ONE write so they arrive buffered together and every
	// one takes the pipelined path. Do NOT read yet: the writer will stall.
	var out bytes.Buffer
	for i := 0; i < n; i++ {
		body := EncodeRequest("get", []byte{byte(i)})
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
		out.Write(hdr[:])
		out.Write(body)
	}
	if _, err := c.Write(out.Bytes()); err != nil {
		t.Fatalf("write requests: %v", err)
	}

	// Wait until every request has been dispatched (so every payload copy has been
	// taken, in the fixed server) before recycling. A generous margin covers the
	// dispatch closure's copy+enqueue and lets the writer reach its blocked flush.
	for i := 0; i < n; i++ {
		select {
		case <-d.dispatched:
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d/%d requests dispatched — requests did not pipeline", i, n)
		}
	}
	// The copy+enqueue happens in the server's dispatch closure just AFTER Call
	// returns (which is what signals d.dispatched), so this margin bridges that gap.
	// Copying 3×12 MiB is a few ms of memcpy; 1s is a large headroom that keeps this
	// robust on a loaded CI box (it can only ever cause a false FAIL, never a false
	// pass — pre-fix the client has not read, so the alias is provably still held).
	time.Sleep(1 * time.Second)

	// Recycle: overwrite every backing extent, simulating the compactor resetting a
	// retired page and future writes trampling its bytes.
	d.mu.Lock()
	for _, b := range backings {
		for j := range b {
			b[j] = sentTag
		}
	}
	d.mu.Unlock()

	// Now drain slowly and verify integrity. Every response must equal what was
	// dispatched — no recycled (0xFF) bytes.
	_ = c.SetReadDeadline(time.Now().Add(10 * time.Second))
	r := bufio.NewReader(c)
	for i := 0; i < n; i++ {
		status, payload := pipeRecv(t, r)
		if status != StatusOK {
			t.Fatalf("resp %d: status %d, want OK", i, status)
		}
		if !bytes.Equal(payload, want[i]) {
			// Report the first differing byte for a crisp failure.
			diff := -1
			for j := 0; j < len(payload) && j < len(want[i]); j++ {
				if payload[j] != want[i][j] {
					diff = j
					break
				}
			}
			t.Fatalf("resp %d: payload does not match dispatched value (len got=%d want=%d, first diff at %d) — "+
				"a recycled alias was shipped to the client (torn read); Fix 1 (copy pipelined payload at dispatch) is missing or broken",
				i, len(payload), len(want[i]), diff)
		}
	}
}
