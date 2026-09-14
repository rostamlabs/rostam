// SPDX-License-Identifier: Apache-2.0

// Package objstore provides a minimal, dependency-free object-storage
// abstraction plus a stdlib-only S3-compatible client (AWS Signature V4).
//
// The whole point of this package is to add an OPTIONAL object-storage tier
// (backup/restore + cold tiering) WITHOUT pulling in aws-sdk-go or minio-go,
// keeping rostam a single, dependency-light binary. Everything here is built
// on the Go standard library.
package objstore

import (
	"context"
	"errors"
	"io"
	"sort"
	"strings"
	"sync"
	"time"
)

// ErrNotFound is returned by Get and Delete when the requested key does not
// exist in the object store.
var ErrNotFound = errors.New("objstore: object not found")

// ErrExists is returned by PutIfAbsent when an object is already stored under
// the key. Nothing was written.
var ErrExists = errors.New("objstore: object already exists")

// ErrConditionalWriteUnsupported is returned by PutIfAbsent when the backend
// cannot guarantee a create-if-absent: for example an S3-compatible service that
// ignores the conditional header and would silently overwrite. Nothing was
// written under the key.
var ErrConditionalWriteUnsupported = errors.New("objstore: backend does not support conditional create")

// ObjectInfo describes a single stored object as returned by List.
type ObjectInfo struct {
	Key          string
	Size         int64
	LastModified time.Time
}

// ObjectStore is the abstraction the rest of the engine depends on. The S3
// client and the in-memory fake both satisfy it, so higher layers never import
// any concrete object-storage code.
type ObjectStore interface {
	// Put stores the contents of r under key. size is the exact number of
	// bytes that will be read from r (used for Content-Length).
	Put(ctx context.Context, key string, r io.Reader, size int64) error
	// PutIfAbsent stores the contents of r under key only if no object exists
	// under key, as ONE atomic step: of any number of concurrent callers for the
	// same absent key — in this process or another — exactly one succeeds. The
	// rest return ErrExists and write nothing; the stored object is the winner's
	// intact. A backend that cannot guarantee that returns
	// ErrConditionalWriteUnsupported rather than degrading to an overwrite. size
	// is as for Put.
	PutIfAbsent(ctx context.Context, key string, r io.Reader, size int64) error
	// Get returns a reader over the object stored under key. The caller must
	// Close it. Returns ErrNotFound if the key does not exist.
	Get(ctx context.Context, key string) (io.ReadCloser, error)
	// List returns all objects whose key starts with prefix, sorted by key.
	List(ctx context.Context, prefix string) ([]ObjectInfo, error)
	// Delete removes the object stored under key. Returns ErrNotFound if the
	// key does not exist.
	Delete(ctx context.Context, key string) error
}

// MemStore is an in-memory ObjectStore implementation intended for tests and
// for higher layers that want a deterministic, network-free backend.
type MemStore struct {
	mu      sync.RWMutex
	objects map[string]memObject
	// clock, if set, supplies LastModified timestamps deterministically.
	clock func() time.Time
}

type memObject struct {
	data         []byte
	lastModified time.Time
}

// NewMemStore returns an empty in-memory object store.
func NewMemStore() *MemStore {
	return &MemStore{objects: make(map[string]memObject)}
}

// SetClock overrides the timestamp source used for LastModified, enabling
// deterministic tests. If unset, time.Now is used.
func (m *MemStore) SetClock(clock func() time.Time) {
	m.mu.Lock()
	m.clock = clock
	m.mu.Unlock()
}

func (m *MemStore) now() time.Time {
	if m.clock != nil {
		return m.clock()
	}
	return time.Now()
}

// Put copies r fully into memory under key. size is ignored beyond being part
// of the interface contract; the actual stored length is whatever r yields.
func (m *MemStore) Put(ctx context.Context, key string, r io.Reader, size int64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.objects[key] = memObject{data: data, lastModified: m.now()}
	m.mu.Unlock()
	return nil
}

// PutIfAbsent copies r fully into memory, then stores it under key only if key
// is absent. The existence check and the insert happen under one hold of the
// write lock, so concurrent callers cannot both see the key absent. The body is
// read BEFORE taking the lock: a slow reader must not stall every other store
// operation, and a read error must leave the map untouched.
func (m *MemStore) PutIfAbsent(ctx context.Context, key string, r io.Reader, size int64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.objects[key]; ok {
		return ErrExists
	}
	m.objects[key] = memObject{data: data, lastModified: m.now()}
	return nil
}

// Get returns a reader over the bytes stored under key, or ErrNotFound.
func (m *MemStore) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	obj, ok := m.objects[key]
	m.mu.RUnlock()
	if !ok {
		return nil, ErrNotFound
	}
	// Copy so the caller can't mutate stored bytes.
	buf := make([]byte, len(obj.data))
	copy(buf, obj.data)
	return io.NopCloser(strings.NewReader(string(buf))), nil
}

// List returns all objects whose key has the given prefix, sorted by key.
func (m *MemStore) List(ctx context.Context, prefix string) ([]ObjectInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	var out []ObjectInfo
	for k, obj := range m.objects {
		if strings.HasPrefix(k, prefix) {
			out = append(out, ObjectInfo{
				Key:          k,
				Size:         int64(len(obj.data)),
				LastModified: obj.lastModified,
			})
		}
	}
	m.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, nil
}

// Delete removes key, or returns ErrNotFound if it is absent.
func (m *MemStore) Delete(ctx context.Context, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.objects[key]; !ok {
		return ErrNotFound
	}
	delete(m.objects, key)
	return nil
}

// compile-time assertion that MemStore satisfies ObjectStore.
var _ ObjectStore = (*MemStore)(nil)
