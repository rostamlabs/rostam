// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"runtime"
	"sync"
	"testing"
)

// parkedWalk installs kvIndexWalkProbe so the FIRST walk to reach an abort check
// stops there and stays stopped until release is called. Later checks, and every
// other walk, pass straight through.
//
// It is what makes the removal-races-a-walk tests deterministic. They used to
// give the walk a 2 ms head start and hope the removal landed inside it: when it
// did not, the test exercised the drain's WAIT and never its ABORT and failed
// with "the removal never raced one" — a false failure on a fast or lightly
// loaded box, and a silent loss of coverage whenever the sleep happened to be
// long enough. With the walk parked, the overlap is a fact.
type parkedWalk struct {
	walking chan struct{} // closed once a walk is provably inside the store
	gate    chan struct{} // closed by release, letting the parked walk continue
	once    sync.Once
}

func parkFirstKVIndexWalk(t *testing.T) *parkedWalk {
	t.Helper()
	p := &parkedWalk{walking: make(chan struct{}), gate: make(chan struct{})}
	fn := func(int) {
		p.once.Do(func() { close(p.walking) })
		<-p.gate
	}
	kvIndexWalkProbe.Store(&fn)
	t.Cleanup(func() {
		kvIndexWalkProbe.Store(nil)
		p.release()
	})
	return p
}

// waitParked blocks until a walk has reached the park.
func (p *parkedWalk) waitParked() { <-p.walking }

// release lets every parked walk continue. Idempotent.
func (p *parkedWalk) release() {
	select {
	case <-p.gate:
	default:
		close(p.gate)
	}
}

// waitKVIndexGateShut spins until group's gate has actually been shut. It waits
// on the STATE CHANGE the removal makes rather than on a duration, which is what
// lets a test release a parked walk at the exact moment the abort is guaranteed
// to be observed.
func (n *Node) waitKVIndexGateShut(group int) {
	g := n.kvIndexWalkGateFor(group)
	for {
		g.mu.Lock()
		shut := g.closed
		g.mu.Unlock()
		if shut {
			return
		}
		runtime.Gosched()
	}
}
