// SPDX-License-Identifier: Apache-2.0

package pageseal

import (
	"sync"
	"testing"
)

func TestNewIsMutableAndSealsOnce(t *testing.T) {
	l := New()
	if !l.Mutable() {
		t.Fatal("a new latch is not mutable")
	}
	if !l.Seal() {
		t.Fatal("the first Seal did not report the transition")
	}
	if l.Mutable() {
		t.Fatal("a sealed latch reports itself mutable")
	}
	if l.Seal() {
		t.Fatal("a second Seal reported a transition it did not perform")
	}
}

// TestZeroValueAndNilAreSealed: anything that never got a latch — a page that was
// never admitted to a mutable region — must read as sealed, with no nil check at
// the call site.
func TestZeroValueAndNilAreSealed(t *testing.T) {
	var zero Latch
	if zero.Mutable() {
		t.Fatal("the zero latch is mutable")
	}
	var nilLatch *Latch
	if nilLatch.Mutable() {
		t.Fatal("a nil latch is mutable")
	}
	if nilLatch.Seal() {
		t.Fatal("sealing a nil latch reported a transition")
	}
}

// TestConcurrentSealTransitionsOnce is what lets the caller count seals: only one
// of N racing sealers may claim the transition.
func TestConcurrentSealTransitionsOnce(t *testing.T) {
	const goroutines = 32
	l := New()
	var (
		wg     sync.WaitGroup
		mu     sync.Mutex
		claims int
	)
	start := make(chan struct{})
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if l.Seal() {
				mu.Lock()
				claims++
				mu.Unlock()
			}
		}()
	}
	close(start)
	wg.Wait()
	if claims != 1 {
		t.Fatalf("%d goroutines claimed the seal transition, want exactly 1", claims)
	}
	if l.Mutable() {
		t.Fatal("the latch is mutable after being sealed")
	}
}
