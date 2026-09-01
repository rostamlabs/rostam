// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"net"
	"sync"
	"testing"
	"time"
)

// gatedSizedDispatcher records every Call and then BLOCKS it until gate is released
// (closed), returning a fixed-size payload once released. Blocking in Call lets a
// test freeze the pipeline mid-dispatch: no closure returns, so none copies or
// reconciles its byte-budget reservation — which makes the reader's admission cap
// deterministic (it depends only on the worst-case reservations taken at admission,
// never on copy timing).
type gatedSizedDispatcher struct {
	respLen    int
	calls      chan struct{} // one send per Call entry (buffered to the number sent)
	gate       chan struct{} // Calls block on this; close() releases them all
	mu         sync.Mutex
	backing    []byte // shared payload backing; copied out by the server, never mutated here
	backingSet bool
}

func (d *gatedSizedDispatcher) Call(_ string, _ []byte) ([]byte, error) {
	d.calls <- struct{}{}
	<-d.gate // blocks until close(d.gate); a closed channel yields immediately
	d.mu.Lock()
	if !d.backingSet {
		d.backing = make([]byte, d.respLen)
		for i := range d.backing {
			d.backing[i] = byte(i)
		}
		d.backingSet = true
	}
	b := d.backing
	d.mu.Unlock()
	return b, nil
}

func (d *gatedSizedDispatcher) LeaderAddr() string { return "" }

// sendGets writes count tiny "get" request frames in ONE Write so they all arrive
// buffered together and every one takes the server's pipelined path.
func sendGets(t *testing.T, c net.Conn, count int) {
	t.Helper()
	var out bytes.Buffer
	for i := 0; i < count; i++ {
		body := EncodeRequest("get", []byte{byte(i)})
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
		out.Write(hdr[:])
		out.Write(body)
	}
	if _, err := c.Write(out.Bytes()); err != nil {
		t.Fatalf("write requests: %v", err)
	}
}

// TestPipelinedByteBudgetBackpressure proves Part B: with EnableOnlineCompaction ON
// the pipelined path copies each response into an owned buffer, and a per-connection
// byte budget bounds how many such copies a slow/stalled reader can pin. The reader
// reserves one worst-case frame (MaxFrameSize) per admission and blocks once the
// reservations would exceed connPipelineByteBudget, so a client that pipelines far
// more large gets than the budget allows CANNOT balloon owned copies to the window
// ceiling (connPipelineWindow * MaxFrameSize ≈ 1 GiB) — it is capped at
// ~connPipelineByteBudget. Draining then unblocks it and every response completes,
// proving the backpressure is not a deadlock.
func TestPipelinedByteBudgetBackpressure(t *testing.T) {
	wantAdmitted := connPipelineByteBudget / MaxFrameSize // reservations that fit the budget
	const extra = 4
	sent := wantAdmitted + extra // more than the budget can admit at once

	d := &gatedSizedDispatcher{
		respLen: 4 << 10, // small: the admission cap comes from the worst-case RESERVATION, not the payload
		calls:   make(chan struct{}, sent),
		gate:    make(chan struct{}),
	}
	// Long write timeout: the writer stalls (we do not read until the assertion is
	// done), and we must not let it abort the connection before we drain.
	addr, stop := startPipelineTestServerCompaction(t, d, 30*time.Second)
	defer stop()

	// Always release the gated dispatch on the way out, even if an assertion fails
	// early: gated closures block the writer, and srv.Close (stop) waits on the handler
	// goroutines, so an unreleased gate would wedge cleanup for the whole package.
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(d.gate) }) }
	defer release()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = c.Close() }()

	sendGets(t, c, sent)

	// Exactly wantAdmitted dispatches occur: the reader admits until its worst-case
	// reservations would exceed the budget, then blocks. Because every Call is gated
	// (no closure returns), no reservation is reconciled, so the cap is exact.
	for i := 0; i < wantAdmitted; i++ {
		select {
		case <-d.calls:
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d/%d dispatched — the byte budget should admit %d worst-case frames", i, wantAdmitted, wantAdmitted)
		}
	}
	// No further dispatch while the writer stays stalled: the reader is backpressured
	// on the byte budget, NOT ballooning to all `sent` copies (the vulnerability this
	// fixes). Pre-fix, the 64-slot window alone would have admitted every one.
	select {
	case <-d.calls:
		t.Fatalf("reader admitted more than the byte budget allows (%d frames) — Part B backpressure missing; a stalled reader can still balloon owned copies toward ~1 GiB", wantAdmitted)
	case <-time.After(500 * time.Millisecond):
	}

	// Release dispatch and start reading: the writer drains, releasing budget, so the
	// reader admits the held-back requests and every response completes in order. This
	// is the no-deadlock proof — the oldest response always proceeds and frees room.
	release()
	_ = c.SetReadDeadline(time.Now().Add(30 * time.Second))
	r := bufio.NewReader(c)
	for i := 0; i < sent; i++ {
		status, payload := pipeRecv(t, r)
		if status != StatusOK {
			t.Fatalf("resp %d: status %d, want OK", i, status)
		}
		if len(payload) != d.respLen {
			t.Fatalf("resp %d: payload len %d, want %d", i, len(payload), d.respLen)
		}
	}
	// Every request was eventually dispatched — the held-back ones only after the drain
	// freed their budget — confirming backpressure released cleanly (no deadlock).
	for i := wantAdmitted; i < sent; i++ {
		select {
		case <-d.calls:
		case <-time.After(5 * time.Second):
			t.Fatalf("held-back request %d never dispatched after drain — budget backpressure deadlocked", i)
		}
	}
}

// TestPipelinedByteBudgetSingleLargeResponse proves the deadlock guard: the byte
// budget gates ADDITIONAL queued responses, never the oldest outstanding one, so a
// single large (near-max-frame) pipelined response always completes even under
// EnableOnlineCompaction. (Responses are frame-capped at MaxFrameSize, and the budget
// is a multiple of MaxFrameSize, so no legal single response can exceed the budget;
// this verifies the equivalent guarantee — the first/only response is never gated
// against itself.)
func TestPipelinedByteBudgetSingleLargeResponse(t *testing.T) {
	const respLen = MaxFrameSize - 64 // largest legal response payload (leaves room for the frame header)

	d := &gatedSizedDispatcher{
		respLen: respLen,
		calls:   make(chan struct{}, 2),
		gate:    make(chan struct{}),
	}
	close(d.gate) // never block dispatch
	addr, stop := startPipelineTestServerCompaction(t, d, 30*time.Second)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = c.Close() }()

	// Two requests in one write so both take the pipelined path; the first is admitted
	// under the empty-queue rule regardless of its (worst-case) reservation.
	sendGets(t, c, 2)
	_ = c.SetReadDeadline(time.Now().Add(30 * time.Second))
	r := bufio.NewReader(c)
	for i := 0; i < 2; i++ {
		status, payload := pipeRecv(t, r)
		if status != StatusOK {
			t.Fatalf("resp %d: status %d, want OK", i, status)
		}
		if len(payload) != respLen {
			t.Fatalf("resp %d: payload len %d, want %d", i, len(payload), respLen)
		}
	}
}

// TestPipelinedDefaultNoCopy proves Part A: with EnableOnlineCompaction OFF (the
// default) the pipelined path does NOT copy — it enqueues the raw zero-copy cache
// alias verbatim, exactly as the server did before this branch. It is the mirror of
// TestPipelinedSlowReaderRecycleIntegrity: the same slow-reader + backing-overwrite
// setup, but because no copy is taken the queued responses STILL alias the backing,
// so the overwrite reaches the client. Observing the sentinel bytes in a queued
// response is the proof that copyPipelinePayload was not invoked (no owned buffer,
// no amplification) in the default configuration.
func TestPipelinedDefaultNoCopy(t *testing.T) {
	if raceEnabled {
		// Overwrites backing bytes that the writer aliases by construction (there is no
		// copy in this mode) — an intentional unsynchronized overlap, same as the recycle
		// test. See TestPipelinedSlowReaderRecycleIntegrity's race comment.
		t.Skip("intentionally overwrites bytes aliased by the writer; unsynchronized by design")
	}
	const (
		n       = 3
		valLen  = 12 << 20 // >> any socket+bufio buffer, so the writer stalls mid-flush with responses still queued (and still aliasing their backings)
		sentTag = 0xFF
	)

	backings := make([][]byte, n)
	for i := range backings {
		b := make([]byte, valLen)
		for j := range b {
			b[j] = byte(i*31 + j) // not all-sentinel
		}
		backings[i] = b
	}

	d := &aliasRecycleDispatcher{backings: backings, dispatched: make(chan struct{}, n)}
	// DEFAULT server: EnableOnlineCompaction OFF ⇒ the pipelined path enqueues the raw
	// alias with no copy.
	addr, stop := startPipelineTestServer(t, d)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = c.Close() }()

	sendGets(t, c, n)

	for i := 0; i < n; i++ {
		select {
		case <-d.dispatched:
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d/%d dispatched — requests did not pipeline", i, n)
		}
	}
	// Let the writer reach its blocked flush on response 0 with 1..n-1 still queued.
	time.Sleep(1 * time.Second)

	// Overwrite every backing extent. In the DEFAULT (no-copy) path the queued
	// responses alias these bytes, so the overwrite must reach the client.
	d.mu.Lock()
	for _, b := range backings {
		for j := range b {
			b[j] = sentTag
		}
	}
	d.mu.Unlock()

	_ = c.SetReadDeadline(time.Now().Add(30 * time.Second))
	r := bufio.NewReader(c)
	var got [][]byte
	for i := 0; i < n; i++ {
		status, payload := pipeRecv(t, r)
		if status != StatusOK {
			t.Fatalf("resp %d: status %d, want OK", i, status)
		}
		got = append(got, append([]byte(nil), payload...))
	}

	// The LAST response was still queued behind the stalled writer when we overwrote,
	// so a no-copy server ships it as the all-sentinel bytes. If it instead matched the
	// original pattern, a copy had been made — i.e. Part A failed to skip the copy when
	// online compaction is off.
	last := got[n-1]
	if len(last) != valLen {
		t.Fatalf("last response len %d, want %d", len(last), valLen)
	}
	for j, x := range last {
		if x != sentTag {
			t.Fatalf("default (flag off) pipelined path did NOT alias the cache buffer: last response byte %d = %d, want sentinel %d — a copy was made when EnableOnlineCompaction is off (Part A regression: zero-copy pipeline not restored)", j, x, sentTag)
		}
	}
}
