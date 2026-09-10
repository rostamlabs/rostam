// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// A DIRECT STORE MUST NOT UNMAP THE CACHE UNDER A READER.
//
// directStore.Call takes no op lock for a read-only op — that is deliberate,
// since the cache's own per-shard RWMutex gives each individual read its
// atomicity. It says nothing about a WALK that spans many of them, and a
// kv_query scan is exactly that: it aliases pages d.cache.Close() unmaps, so a
// scan racing Close reads freed memory and takes the process down.
//
// shard.Store carries the identical fence (drainCalls in Close) for the
// identical reason. Direct is the third host of the hazard and had none.
//
// The test parks a handler inside Call, starts Close, and asserts that Close
// WAITS rather than proceeding to the unmap.
func TestDirectCloseDrainsInFlightCalls(t *testing.T) {
	d := newDirectForReconcile(t, -1)

	// A registered read-only op that parks until the test lets it go — standing
	// in for a scan still walking the cache.
	entered := make(chan struct{})
	release := make(chan struct{})
	if err := d.registry.Register("test_parked_read", ops.OpReadOnly,
		func(_ *ops.TxContext, _ []byte) ([]byte, error) {
			close(entered)
			<-release
			return nil, nil
		}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	callDone := make(chan error, 1)
	go func() {
		_, err := d.Call(context.Background(), "test_parked_read", nil)
		callDone <- err
	}()
	<-entered

	closeDone := make(chan error, 1)
	go func() { closeDone <- d.Close() }()

	// Close must still be inside the drain: it has not reached cache.Close, so
	// the parked handler's view of the mapping is still valid.
	select {
	case err := <-closeDone:
		t.Fatalf("Close returned (%v) while a read was still inside the cache; it would have unmapped under it", err)
	case <-time.After(200 * time.Millisecond):
	}

	// And a new call arriving during the drain is REFUSED rather than admitted
	// into a store being torn down.
	if _, err := d.Call(context.Background(), "test_parked_read", nil); !errors.Is(err, ErrDirectClosed) {
		t.Fatalf("a Call during the drain returned %v, want ErrDirectClosed", err)
	}

	close(release)
	if err := <-callDone; err != nil {
		t.Fatalf("the parked call: %v", err)
	}
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Close did not return after the in-flight call finished")
	}
}
