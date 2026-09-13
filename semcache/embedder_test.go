// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

// fakeEmbedder maps text -> a deterministic unit vector of length dim. Used by
// every library test so no network/model is required.
type fakeEmbedder struct {
	model string
	dim   int
}

func (f fakeEmbedder) Model() string { return f.model }
func (f fakeEmbedder) Dim() int      { return f.dim }

func (f fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, f.dim)
		for j := 0; j < f.dim; j++ {
			// stable per (text, j) value; normalized below
			h := uint32(2166136261)
			for _, b := range []byte(t) {
				h = (h ^ uint32(b)) * 16777619
			}
			h ^= uint32(j) * 2654435761
			v[j] = float32(int32(h)) / float32(math.MaxInt32)
		}
		normalizeVec(v)
		out[i] = v
	}
	return out, nil
}

func TestFakeEmbedderIsNormalizedAndDeterministic(t *testing.T) {
	e := fakeEmbedder{model: "fake-1", dim: 8}
	a, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := e.Embed(context.Background(), []string{"hello"})
	if a[0][0] != b[0][0] {
		t.Fatalf("embedder not deterministic")
	}
	var sum float64
	for _, x := range a[0] {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Fatalf("not unit-normalized: |v|^2=%v", sum)
	}
}

// fnv32a is the 32-bit hash the stub embedder used to seed itself with. It
// lives here, in the test, purely to manufacture the collision that made that
// choice unsafe.
func fnv32a(s string) uint32 {
	h := uint32(2166136261)
	for _, b := range []byte(s) {
		h = (h ^ uint32(b)) * 16777619
	}
	return h
}

// TestStubEmbedderSurvivesA32BitCollision finds a real pair of distinct
// strings sharing a 32-bit FNV-1a hash (a birthday search over 2^32 needs
// only ~100k candidates) and asserts the stub embedder still separates them.
// The stub used to derive every dimension from exactly that 32-bit hash, so
// such a pair produced identical vectors — cosine 1.0, an exact-mode cache
// hit, and one prompt answered with another prompt's completion.
func TestStubEmbedderSurvivesA32BitCollision(t *testing.T) {
	const maxCandidates = 500_000 // P(no collision) < 1e-6

	seen := make(map[uint32]string, maxCandidates)
	var a, b string
	for i := 0; i < maxCandidates && a == ""; i++ {
		s := "collide-" + strconv.Itoa(i)
		h := fnv32a(s)
		if prev, ok := seen[h]; ok {
			a, b = prev, s
			break
		}
		seen[h] = s
	}
	if a == "" {
		t.Skip("no 32-bit FNV-1a collision found in the candidate budget")
	}
	if a == b {
		t.Fatalf("collision search returned the same string twice: %q", a)
	}
	t.Logf("colliding pair: %q / %q (fnv32a=%d)", a, b, fnv32a(a))

	e := NewStubEmbedder("stub-1", 64)
	vecs, err := e.Embed(context.Background(), []string{a, b})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	var dot float64
	for i := range vecs[0] {
		dot += float64(vecs[0][i]) * float64(vecs[1][i])
	}
	// The exact-mode threshold is 0.999; anything at or above it would be
	// served as a hit.
	if dot >= 0.999 {
		t.Fatalf("32-bit-colliding prompts embed to cosine %v, want well below the 0.999 exact-mode floor", dot)
	}
}

func TestStubEmbedderDeterministicAndNormalized(t *testing.T) {
	e := NewStubEmbedder("stub-1", 32)
	a, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	b, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatalf("stub embedder not deterministic at dim %d: %v vs %v", i, a[0][i], b[0][i])
		}
	}
	if len(a[0]) != 32 {
		t.Fatalf("dim = %d, want 32", len(a[0]))
	}
	var sum float64
	for _, x := range a[0] {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Fatalf("not unit-normalized: |v|^2=%v", sum)
	}
}

func TestOpenAIEmbedderShapesRequestAndNormalizes(t *testing.T) {
	var gotModel string
	var gotInputs []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing/wrong auth header: %q", r.Header.Get("Authorization"))
		}
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotModel, gotInputs = req.Model, req.Input
		// Return a non-unit vector to prove the embedder normalizes it.
		resp := map[string]any{"data": []map[string]any{{"embedding": []float32{3, 4}}}}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("test-key", "text-embedding-3-small", 2)
	e.Endpoint = srv.URL
	vecs, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "text-embedding-3-small" || len(gotInputs) != 1 || gotInputs[0] != "hello" {
		t.Fatalf("bad request shape: model=%q inputs=%v", gotModel, gotInputs)
	}
	// (3,4) normalized => (0.6,0.8).
	if math.Abs(float64(vecs[0][0])-0.6) > 1e-4 || math.Abs(float64(vecs[0][1])-0.8) > 1e-4 {
		t.Fatalf("not normalized: %v", vecs[0])
	}
}
