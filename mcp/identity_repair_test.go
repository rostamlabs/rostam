// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/semcache"
	"github.com/rostamlabs/rostam/vector"
)

// saltedEmbedder has the stub's shape but produces different vectors for the
// same text: a different model of the same dimension.
type saltedEmbedder struct{ semcache.Embedder }

func newSaltedEmbedder() saltedEmbedder {
	return saltedEmbedder{semcache.NewStubEmbedder("salted", 64)}
}

func (e saltedEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	salted := make([]string, len(texts))
	for i, t := range texts {
		salted[i] = "salt " + t
	}
	return e.Embedder.Embed(ctx, salted)
}

// driftEmbedder tilts the stub's unit vectors by a fixed amount, so the cosine
// to the stored vector is 1/sqrt(1+tilt^2).
type driftEmbedder struct {
	semcache.Embedder
	tilt float64
}

func (e driftEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	vecs, err := e.Embedder.Embed(ctx, texts)
	if err != nil {
		return nil, err
	}
	for _, v := range vecs {
		// u: the first axis made orthogonal to v, at unit length.
		u := make([]float64, len(v))
		for i := range v {
			u[i] = -float64(v[0]) * float64(v[i])
		}
		u[0]++
		var n float64
		for _, x := range u {
			n += x * x
		}
		n = math.Sqrt(n)
		for i := range v {
			v[i] += float32(e.tilt * u[i] / n)
		}
	}
	return vecs, nil
}

// removeIdentity removes the embedder identity record.
func removeIdentity(t *testing.T, st rostam.Store) {
	t.Helper()
	if ok, err := st.Del(context.Background(), []byte(kvEmbedder)); err != nil || !ok {
		t.Fatalf("deleting the identity record: removed=%v err=%v", ok, err)
	}
}

func requireNoIdentity(t *testing.T, st rostam.Store) {
	t.Helper()
	if raw, err := st.Get(context.Background(), []byte(kvEmbedder)); !errors.Is(err, rostam.ErrNotFound) {
		t.Fatalf("identity record = %q, err = %v; want it absent", raw, err)
	}
}

func requireIdentity(t *testing.T, st rostam.Store, want embedIdentity) {
	t.Helper()
	raw, err := st.Get(context.Background(), []byte(kvEmbedder))
	if err != nil {
		t.Fatalf("reading the identity record: %v", err)
	}
	var got embedIdentity
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decoding the identity record %q: %v", raw, err)
	}
	if got != want {
		t.Fatalf("identity record = %+v, want %+v", got, want)
	}
}

// createMemoryCollection creates an empty memory collection of the given
// dimension and records no identity.
func createMemoryCollection(t *testing.T, st rostam.Store, dim int) {
	t.Helper()
	cfg := vector.DefaultConfig()
	cfg.Dim, cfg.Metric, cfg.FullText = dim, vector.Cosine, &vector.FullTextConfig{}
	applyPersistence(&cfg)
	if err := st.CreateCollection(context.Background(), MemoryCollection, cfg); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
}

// storeWithoutIdentity returns a store holding memories written with emb (nil
// = BM25-only) with no identity record.
func storeWithoutIdentity(t *testing.T, st rostam.Store, emb semcache.Embedder, contents ...string) {
	t.Helper()
	w := newTestServer(t, Config{Store: st, Embedder: emb})
	for i, c := range contents {
		seedMemory(t, w, "default", "", c, int64(i+1))
	}
	removeIdentity(t, st)
}

func countMemories(t *testing.T, st rostam.Store) int {
	t.Helper()
	docs, _, _, err := st.VectorScroll(context.Background(), memCollection, rostam.VectorFilter{}, 100, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("VectorScroll: %v", err)
	}
	return len(docs)
}

var stubIdentity = embedIdentity{Model: "stub", Dim: 64}

func TestRepairIdentityEmptyCollection(t *testing.T) {
	st := newHeapStore(t)
	createMemoryCollection(t, st, 64)

	s := newTestServer(t, Config{Store: st})
	if err := s.ensureMemory(context.Background()); err != nil {
		t.Fatalf("ensureMemory: %v", err)
	}
	requireIdentity(t, st, stubIdentity)
}

func TestRepairIdentityMatchingMemories(t *testing.T) {
	cases := []struct {
		name  string
		store func(*testing.T) rostam.Store
		emb   semcache.Embedder
		want  embedIdentity
	}{
		{"bm25 only", newHeapStore, nil, stubIdentity},
		{"embedder", newHeapStore, newSaltedEmbedder(), embedIdentity{Model: "salted", Dim: 64, Hybrid: true}},
		{"remote", newRemoteStore, nil, stubIdentity},
		{"remote embedder", newRemoteStore, newSaltedEmbedder(), embedIdentity{Model: "salted", Dim: 64, Hybrid: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := tc.store(t)
			c, stop := startServerStop(t, Config{Store: st, Embedder: tc.emb})
			c.initialize()
			// A keyed, a code-linked and a plain memory: each must offer its
			// content for verification.
			c.callTool("remember", map[string]any{"content": "the deploy key lives in vault", "key": "deploy"}, nil, false)
			c.callTool("remember", map[string]any{"content": "atlas ships in march", "path": "atlas/main.go", "symbol": "Ship"}, nil, false)
			c.callTool("remember", map[string]any{"content": "the coffee machine is on floor two"}, nil, false)
			stop()
			removeIdentity(t, st)

			c2 := startServer(t, Config{Store: st, Embedder: tc.emb})
			c2.initialize()
			var rec struct {
				Hits []struct {
					Content string `json:"content"`
				} `json:"hits"`
			}
			c2.callTool("recall", map[string]any{"query": "deploy key vault"}, &rec, false)
			if len(rec.Hits) == 0 {
				t.Fatal("recall after repair found nothing")
			}
			requireIdentity(t, st, tc.want)
			if n := countMemories(t, st); n != 3 {
				t.Fatalf("memories after repair = %d, want 3", n)
			}
		})
	}
}

func TestRepairIdentityOtherEmbedderSameDim(t *testing.T) {
	for _, tc := range []struct {
		name  string
		store func(*testing.T) rostam.Store
	}{{"embedded", newHeapStore}, {"remote", newRemoteStore}} {
		t.Run(tc.name, func(t *testing.T) {
			st := tc.store(t)
			storeWithoutIdentity(t, st, nil, "fact one", "fact two")

			wrong := newTestServer(t, Config{Store: st, Embedder: newSaltedEmbedder()})
			err := wrong.ensureMemory(context.Background())
			if !errors.Is(err, ErrEmbedderMismatch) {
				t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch", err)
			}
			if !strings.Contains(err.Error(), "ROSTAM_EMBED_") {
				t.Fatalf("the error must say which environment to set: %v", err)
			}
			requireNoIdentity(t, st)

			right := startServer(t, Config{Store: st})
			right.initialize()
			var lst struct {
				Memories []struct {
					Content string `json:"content"`
				} `json:"memories"`
			}
			right.callTool("list_memories", map[string]any{}, &lst, false)
			if len(lst.Memories) != 2 {
				t.Fatalf("list_memories after a refused repair = %+v, want both", lst.Memories)
			}
			requireIdentity(t, st, stubIdentity)
		})
	}
}

// One matching memory is not enough: every sampled one has to match.
func TestRepairIdentityOneSampleMismatches(t *testing.T) {
	st := newHeapStore(t)
	ctx := context.Background()
	w := newTestServer(t, Config{Store: st})
	seedMemory(t, w, "default", "", "fact one", 1)
	seedMemory(t, w, "default", "", "fact two", 2)
	other, err := newSaltedEmbedder().Embed(ctx, []string{"fact three"})
	if err != nil {
		t.Fatal(err)
	}
	// The largest id, so it is the last one sampled.
	if err := st.VectorUpsert(ctx, memCollection, math.MaxUint64, other[0], "fact three", rostam.VectorInsertOpts{}); err != nil {
		t.Fatal(err)
	}
	removeIdentity(t, st)

	s := newTestServer(t, Config{Store: st})
	if err := s.ensureMemory(ctx); !errors.Is(err, ErrEmbedderMismatch) {
		t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch", err)
	}
	requireNoIdentity(t, st)
	if n := countMemories(t, st); n != 3 {
		t.Fatalf("memories after a refused repair = %d, want 3", n)
	}
}

func TestRepairIdentityCosineThreshold(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tilt  float64
		match bool
	}{
		{"cosine 0.9992", 0.04, true},
		{"cosine 0.9986", 0.053, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := newHeapStore(t)
			storeWithoutIdentity(t, st, nil, "fact one")
			emb := driftEmbedder{semcache.NewStubEmbedder("drift", 64), tc.tilt}

			err := newTestServer(t, Config{Store: st, Embedder: emb}).ensureMemory(context.Background())
			if tc.match {
				if err != nil {
					t.Fatalf("ensureMemory: %v", err)
				}
				requireIdentity(t, st, embedIdentity{Model: "drift", Dim: 64, Hybrid: true})
				return
			}
			if !errors.Is(err, ErrEmbedderMismatch) {
				t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch", err)
			}
			requireNoIdentity(t, st)
		})
	}
}

func TestRepairIdentityWrongDimEmpty(t *testing.T) {
	for _, tc := range []struct {
		name  string
		store func(*testing.T) rostam.Store
	}{{"embedded", newHeapStore}, {"remote", newRemoteStore}} {
		t.Run(tc.name, func(t *testing.T) {
			st := tc.store(t)
			createMemoryCollection(t, st, 32)

			s := newTestServer(t, Config{Store: st}) // stub: dim 64
			err := s.ensureMemory(context.Background())
			if !errors.Is(err, ErrEmbedderMismatch) {
				t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch", err)
			}
			if !strings.Contains(err.Error(), "ROSTAM_EMBED_") {
				t.Fatalf("the error must say which environment to set: %v", err)
			}
			requireNoIdentity(t, st)
		})
	}
}

func TestRepairIdentityReadOnlyNeverWrites(t *testing.T) {
	st := newHeapStore(t)
	storeWithoutIdentity(t, st, nil, "fact one")

	r := newTestServer(t, Config{Store: st, ReadOnly: true})
	if err := r.ensureMemory(context.Background()); err != nil {
		t.Fatalf("read-only ensureMemory: %v", err)
	}
	c := startServer(t, Config{Store: st, ReadOnly: true})
	c.initialize()
	var rec struct {
		Hits []struct {
			Content string `json:"content"`
		} `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "fact one"}, &rec, false)
	if len(rec.Hits) != 1 {
		t.Fatalf("read-only recall = %+v, want the one memory", rec.Hits)
	}
	requireNoIdentity(t, st)
}

var errInjected = errors.New("injected failure")

type failingScrollStore struct{ rostam.Store }

func (failingScrollStore) VectorScroll(context.Context, string, rostam.VectorFilter, int, rostam.VectorScrollOpts) ([]rostam.VectorDocument, rostam.FanMeta, string, error) {
	return nil, rostam.FanMeta{}, "", errInjected
}

type failingEmbedder struct{ semcache.Embedder }

func (failingEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	return nil, errInjected
}

// vanishedStore reports every sampled memory as gone by the time it is read.
type vanishedStore struct{ rostam.Store }

func (vanishedStore) VectorGet(context.Context, string, uint64, bool, bool) (bool, []float32, rostam.VectorMetadata, time.Duration, *rostam.VectorSparse, error) {
	return false, nil, nil, 0, nil, nil
}

// A repair that cannot finish records nothing, and is not reported as a
// mismatch: the next call tries again.
func TestRepairIdentityOtherFailures(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  func(rostam.Store) Config
		want error
	}{
		{"scroll fails", func(st rostam.Store) Config { return Config{Store: failingScrollStore{st}} }, errInjected},
		{"embedder fails", func(st rostam.Store) Config {
			return Config{Store: st, Embedder: failingEmbedder{semcache.NewStubEmbedder("stub", 64)}}
		}, errInjected},
		{"memory vanishes", func(st rostam.Store) Config { return Config{Store: vanishedStore{st}} }, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := newHeapStore(t)
			storeWithoutIdentity(t, st, nil, "fact one")

			err := newTestServer(t, tc.cfg(st)).ensureMemory(context.Background())
			if err == nil || errors.Is(err, ErrEmbedderMismatch) {
				t.Fatalf("ensureMemory = %v, want a failure that is not a mismatch", err)
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Fatalf("ensureMemory = %v, want %v", err, tc.want)
			}
			requireNoIdentity(t, st)
		})
	}
}
