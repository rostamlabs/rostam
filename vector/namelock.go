// SPDX-License-Identifier: Apache-2.0

package vector

import "sync"

// nameLock serializes the structural operations on one canonical collection
// name. refs counts the holder plus every waiter, so the entry is removed from
// the store's map exactly when nobody holds or wants it.
type nameLock struct {
	mu   sync.Mutex
	refs int
}

// lockName blocks until the caller holds canonical's name lock and returns the
// function that releases it.
//
// Every operation that changes what a name refers to takes it for its whole
// duration, INCLUDING the file cleanup that runs after the catalog entry is gone:
// the creates, the drops, a cold promotion, the commit of an eviction, and a
// create-or-replace restore. Those operations delete and write files by path, and
// the path is a function of the name alone, so without mutual exclusion a
// cleanup that started before another operation can remove the files that
// operation has just published. Reads and writes on an already-resolved
// collection never take it.
//
// Lock order: a name lock is taken before catalogMu, which is taken before mu.
// No path holds mu or catalogMu while acquiring a name lock, and no path holds
// one name lock while acquiring another. Name locks are not reentrant: nothing
// that runs under one may resolve the same name through Acquire, since a cold
// collection's promotion takes that lock.
//
// The map holds an entry only while some goroutine holds or waits for that name,
// so it does not grow with the number of names ever used.
func (s *CollectionStore) lockName(canonical string) (unlock func()) {
	s.nameLocksMu.Lock()
	if s.nameLocks == nil {
		s.nameLocks = make(map[string]*nameLock)
	}
	l := s.nameLocks[canonical]
	if l == nil {
		l = &nameLock{}
		s.nameLocks[canonical] = l
	}
	l.refs++
	s.nameLocksMu.Unlock()

	l.mu.Lock()
	return func() {
		l.mu.Unlock()
		s.nameLocksMu.Lock()
		l.refs--
		if l.refs == 0 {
			delete(s.nameLocks, canonical)
		}
		s.nameLocksMu.Unlock()
	}
}
