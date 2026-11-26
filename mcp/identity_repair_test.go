// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
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
	docs, _, _, err := st.VectorScroll(context.Background(), memCollection, rostam.VectorFilter{}, 1000, rostam.VectorScrollOpts{})
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

// putPoint writes a point into the memory collection: content "" leaves it
// without content, and wrong stores a vector this run's stub embedder cannot
// reproduce.
func putPoint(t *testing.T, st rostam.Store, id uint64, content string, wrong bool) {
	t.Helper()
	ctx := context.Background()
	var emb semcache.Embedder = semcache.NewStubEmbedder("stub", 64)
	if wrong {
		emb = newSaltedEmbedder()
	}
	vecs, err := emb.Embed(ctx, []string{content})
	if err != nil {
		t.Fatal(err)
	}
	if err := st.VectorUpsert(ctx, memCollection, id, vecs[0], content, rostam.VectorInsertOpts{}); err != nil {
		t.Fatal(err)
	}
}

// At least half of the sampled memories, and at least one, must match.
func TestRepairIdentityMajority(t *testing.T) {
	const ok, bad = false, true
	for _, tc := range []struct {
		name  string
		wrong []bool // one memory each, in id order
		match bool
	}{
		{"1 of 1", []bool{ok}, true},
		{"0 of 1", []bool{bad}, false},
		{"1 of 2", []bool{bad, ok}, true},
		{"0 of 2", []bool{bad, bad}, false},
		{"2 of 3, first wrong", []bool{bad, ok, ok}, true},
		{"2 of 3, last wrong", []bool{ok, ok, bad}, true},
		{"1 of 3", []bool{ok, bad, bad}, false},
		{"0 of 3", []bool{bad, bad, bad}, false},
		{"only three are sampled", []bool{ok, ok, ok, bad, bad, bad, bad}, true},
		{"matches past the sample do not count", []bool{bad, bad, bad, ok, ok, ok, ok}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := newHeapStore(t)
			createMemoryCollection(t, st, 64)
			for i, wrong := range tc.wrong {
				putPoint(t, st, uint64(i+1), fmt.Sprintf("fact %d", i+1), wrong)
			}

			err := newTestServer(t, Config{Store: st}).ensureMemory(context.Background())
			if n := countMemories(t, st); n != len(tc.wrong) {
				t.Fatalf("memories after repair = %d, want %d", n, len(tc.wrong))
			}
			if tc.match {
				if err != nil {
					t.Fatalf("ensureMemory: %v", err)
				}
				requireIdentity(t, st, stubIdentity)
				return
			}
			if !errors.Is(err, ErrEmbedderMismatch) || !strings.Contains(err.Error(), "ROSTAM_EMBED_") {
				t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch naming the environment", err)
			}
			requireNoIdentity(t, st)
		})
	}
}

// Points without content are neither matches nor misses, and the sample is
// looked for past a first page full of them.
func TestRepairIdentitySkipsContentlessPoints(t *testing.T) {
	for _, tc := range []struct {
		name        string
		store       func(*testing.T) rostam.Store
		contentless int
	}{
		{"a few", newHeapStore, 5},
		{"more than a page", newHeapStore, identityScanPage + 20},
		{"remote, more than a page", newRemoteStore, identityScanPage + 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := tc.store(t)
			createMemoryCollection(t, st, 64)
			// Their vectors are ones the embedder cannot reproduce, so sampling
			// them would count as misses.
			for i := range tc.contentless {
				putPoint(t, st, uint64(i+1), "", true)
			}
			// One wrong of three still passes.
			next := uint64(tc.contentless + 1)
			putPoint(t, st, next, "fact one", true)
			putPoint(t, st, next+1, "fact two", false)
			putPoint(t, st, next+2, "fact three", false)

			if err := newTestServer(t, Config{Store: st}).ensureMemory(context.Background()); err != nil {
				t.Fatalf("ensureMemory: %v", err)
			}
			requireIdentity(t, st, stubIdentity)
		})
	}
}

// requireUnverifiable checks the refusal for a collection offering nothing to
// re-embed: changing the embedder environment would not help, so it must not
// advise that.
func requireUnverifiable(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, ErrEmbedderMismatch) {
		t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch", err)
	}
	if msg := err.Error(); !strings.Contains(msg, "cannot verify") || !strings.Contains(msg, "content") || strings.Contains(msg, "ROSTAM_EMBED_") {
		t.Fatalf("the error must say there is no content to verify against, without environment advice: %v", err)
	}
}

func TestRepairIdentityOnlyContentlessPoints(t *testing.T) {
	st := newHeapStore(t)
	createMemoryCollection(t, st, 64)
	for id := range uint64(4) {
		putPoint(t, st, id+1, "", false)
	}

	requireUnverifiable(t, newTestServer(t, Config{Store: st}).ensureMemory(context.Background()))
	requireNoIdentity(t, st)
	if n := countMemories(t, st); n != 4 {
		t.Fatalf("points after a refused repair = %d, want 4", n)
	}
}

// endlessContentlessStore scrolls full pages of contentless points forever.
type endlessContentlessStore struct {
	rostam.Store
	pages int
}

func (s *endlessContentlessStore) VectorScroll(_ context.Context, _ string, _ rostam.VectorFilter, limit int, _ rostam.VectorScrollOpts) ([]rostam.VectorDocument, rostam.FanMeta, string, error) {
	s.pages++
	return make([]rostam.VectorDocument, limit), rostam.FanMeta{}, "", nil
}

func TestRepairIdentitySampleSearchIsBounded(t *testing.T) {
	st := newHeapStore(t)
	createMemoryCollection(t, st, 64)
	endless := &endlessContentlessStore{Store: st}

	requireUnverifiable(t, newTestServer(t, Config{Store: endless}).ensureMemory(context.Background()))
	if endless.pages != identityScanPages {
		t.Fatalf("pages scrolled = %d, want %d", endless.pages, identityScanPages)
	}
	requireNoIdentity(t, st)
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

var backends = []struct {
	name  string
	store func(*testing.T) rostam.Store
}{{"embedded", newHeapStore}, {"remote", newRemoteStore}}

// A record that appears after a server started is compared like one that was
// there at startup.
func TestIdentityAppearingAfterStartupIsCompared(t *testing.T) {
	for _, b := range backends {
		t.Run(b.name, func(t *testing.T) {
			st := b.store(t)
			storeWithoutIdentity(t, st, nil, "fact one")
			wrong := startServer(t, Config{Store: st, Embedder: newSaltedEmbedder()})
			wrong.initialize()
			right := newTestServer(t, Config{Store: st}) // started with no record too
			if err := newTestServer(t, Config{Store: st}).ensureMemory(context.Background()); err != nil {
				t.Fatalf("repairing session: %v", err)
			}

			if msg := wrong.callTool("remember", map[string]any{"content": "incompatible fact"}, nil, true); !strings.Contains(msg, ErrEmbedderMismatch.Error()) {
				t.Fatalf("remember = %q, want an embedder mismatch", msg)
			}
			if n := countMemories(t, st); n != 1 {
				t.Fatalf("memories = %d, want only the original", n)
			}
			if err := right.ensureMemory(context.Background()); err != nil {
				t.Fatalf("matching session: %v", err)
			}
			requireIdentity(t, st, stubIdentity)
		})
	}
}

// rivalStore runs rival while a session is between looking for the identity
// record and writing it: right after it creates the collection, or while it
// verifies an existing one.
type rivalStore struct {
	rostam.Store
	rival func()
}

func (s rivalStore) CreateCollection(ctx context.Context, name string, cfg rostam.VectorConfig) error {
	err := s.Store.CreateCollection(ctx, name, cfg)
	if err == nil {
		s.rival()
	}
	return err
}

func (s rivalStore) VectorScroll(ctx context.Context, collection string, f rostam.VectorFilter, limit int, opts rostam.VectorScrollOpts) ([]rostam.VectorDocument, rostam.FanMeta, string, error) {
	s.rival()
	return s.Store.VectorScroll(ctx, collection, f, limit, opts)
}

// A record written by someone else in the meantime is never overwritten: it is
// read back and compared.
func TestIdentityWrittenConcurrentlyIsKept(t *testing.T) {
	paths := []struct {
		name    string
		prepare func(*testing.T, rostam.Store)
	}{
		{"creator", func(*testing.T, rostam.Store) {}},
		{"repair", func(t *testing.T, st rostam.Store) { storeWithoutIdentity(t, st, nil, "fact one") }},
	}
	rivals := []struct {
		name  string
		id    embedIdentity
		match bool
	}{
		{"different", embedIdentity{Model: "rival", Dim: 64, Hybrid: true}, false},
		{"identical", stubIdentity, true},
	}
	for _, b := range backends {
		for _, p := range paths {
			for _, r := range rivals {
				t.Run(b.name+"/"+p.name+"/"+r.name, func(t *testing.T) {
					st := b.store(t)
					p.prepare(t, st)
					rec, err := json.Marshal(r.id)
					if err != nil {
						t.Fatal(err)
					}
					wrote := 0
					racing := rivalStore{st, func() {
						wrote++
						if err := st.Put(context.Background(), []byte(kvEmbedder), rec, 0); err != nil {
							t.Fatalf("rival Put: %v", err)
						}
					}}

					s := newTestServer(t, Config{Store: racing})
					err = s.ensureMemory(context.Background())
					if wrote != 1 {
						t.Fatalf("rival wrote %d times, want 1", wrote)
					}
					if r.match && err != nil {
						t.Fatalf("ensureMemory: %v", err)
					}
					if !r.match && !errors.Is(err, ErrEmbedderMismatch) {
						t.Fatalf("ensureMemory = %v, want ErrEmbedderMismatch", err)
					}
					if s.memReady != r.match {
						t.Fatalf("memReady = %v, want %v", s.memReady, r.match)
					}
					requireIdentity(t, st, r.id)
				})
			}
		}
	}
}

// meetingStore holds each session's verification until all of them are in it.
type meetingStore struct {
	rostam.Store
	meet *sync.WaitGroup
}

func (s meetingStore) VectorScroll(ctx context.Context, collection string, f rostam.VectorFilter, limit int, opts rostam.VectorScrollOpts) ([]rostam.VectorDocument, rostam.FanMeta, string, error) {
	s.meet.Done()
	s.meet.Wait()
	return s.Store.VectorScroll(ctx, collection, f, limit, opts)
}

// Two embedders that both pass the check (an empty collection offers only its
// dimension) cannot both be recorded.
func TestIdentityOnlyOneConcurrentRepairerWins(t *testing.T) {
	for _, b := range backends {
		t.Run(b.name, func(t *testing.T) {
			st := b.store(t)
			createMemoryCollection(t, st, 64)
			meet := &sync.WaitGroup{}
			meet.Add(2)
			sessions := []*Server{
				newTestServer(t, Config{Store: meetingStore{st, meet}}),
				newTestServer(t, Config{Store: meetingStore{st, meet}, Embedder: newSaltedEmbedder()}),
			}
			errs := make([]error, len(sessions))
			var done sync.WaitGroup
			for i, s := range sessions {
				done.Add(1)
				go func() {
					defer done.Done()
					errs[i] = s.ensureMemory(context.Background())
				}()
			}
			done.Wait()

			winner, loser := 0, 1
			if errs[0] != nil {
				winner, loser = 1, 0
			}
			if errs[winner] != nil || !errors.Is(errs[loser], ErrEmbedderMismatch) {
				t.Fatalf("errors = %v, want one success and one ErrEmbedderMismatch", errs)
			}
			requireIdentity(t, st, sessions[winner].identity())
		})
	}
}

// refusingSetNXStore fails every set_nx and counts plain writes.
type refusingSetNXStore struct {
	rostam.Store
	puts *int
}

func (s refusingSetNXStore) Call(ctx context.Context, op string, args []byte) ([]byte, error) {
	if op == "set_nx" {
		return nil, errInjected
	}
	return s.Store.Call(ctx, op, args)
}

func (s refusingSetNXStore) Put(ctx context.Context, key, value []byte, ttl time.Duration) error {
	*s.puts++
	return s.Store.Put(ctx, key, value, ttl)
}

// A failed set_nx is reported with the server version it may be down to, and
// nothing is written another way.
func TestIdentityFailedSetNXHintsAndWritesNothing(t *testing.T) {
	for _, tc := range []struct {
		name    string
		prepare func(*testing.T, rostam.Store)
	}{
		{"creator", func(*testing.T, rostam.Store) {}},
		{"repair", func(t *testing.T, st rostam.Store) { storeWithoutIdentity(t, st, nil, "fact one") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := newHeapStore(t)
			tc.prepare(t, st)
			puts := 0
			s := newTestServer(t, Config{Store: refusingSetNXStore{st, &puts}})

			err := s.ensureMemory(context.Background())
			if !errors.Is(err, errInjected) {
				t.Fatalf("ensureMemory = %v, want it to wrap the set_nx failure", err)
			}
			if msg := err.Error(); !strings.Contains(msg, "may") || !strings.Contains(msg, "-connect") || !strings.Contains(msg, "v0.5.0 or later") {
				t.Fatalf("the error must hint at the server version: %v", err)
			}
			if puts != 0 || s.memReady {
				t.Fatalf("puts = %d, memReady = %v; want nothing written and not ready", puts, s.memReady)
			}
			requireNoIdentity(t, st)
		})
	}
}
