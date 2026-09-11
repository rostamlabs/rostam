// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"strings"
	"testing"
)

// TestHTTPMetrics covers the Prometheus scrape surface: after inserting points
// into two collections, both /metrics and /v1/metrics return 200 with the
// Prometheus content type and an exposition body carrying each collection's
// labeled series.
func TestHTTPMetrics(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	for _, name := range []string{"docs", "images"} {
		rec := do(t, h, "POST", "/v1/collections",
			`{"name":"`+name+`","config":{"dim":3,"metric":"l2"}}`, nil)
		if rec.Code != http.StatusCreated {
			t.Fatalf("create %s = %d (%s)", name, rec.Code, rec.Body)
		}
		rec = do(t, h, "POST", "/v1/collections/"+name+"/points",
			`{"id":1,"vector":[1,0,0]}`, nil)
		if rec.Code != http.StatusOK && rec.Code != http.StatusCreated {
			t.Fatalf("insert into %s = %d (%s)", name, rec.Code, rec.Body)
		}
	}

	for _, path := range []string{"/metrics", "/v1/metrics"} {
		rec := do(t, h, "GET", path, "", nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s = %d (%s)", path, rec.Code, rec.Body)
		}
		if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain; version=0.0.4") {
			t.Fatalf("GET %s content-type = %q", path, ct)
		}
		body := rec.Body.String()
		for _, want := range []string{
			`rostam_vector_size{collection="default/docs"}`,
			`rostam_vector_size{collection="default/images"}`,
			"rostam_vector_insert_ops_total",
		} {
			if !strings.Contains(body, want) {
				t.Fatalf("GET %s body missing %q\n---\n%s", path, want, body)
			}
		}
	}
}

// TestHTTPKVMetrics covers the KV scrape surface: both route aliases return 200
// with the Prometheus content type and an exposition body carrying the cache
// counters — including on a node with no vector collections, which is exactly
// where /metrics returns nothing.
func TestHTTPKVMetrics(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	for _, path := range []string{"/kv-metrics", "/v1/kv-metrics"} {
		rec := do(t, h, "GET", path, "", nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s = %d (%s)", path, rec.Code, rec.Body)
		}
		if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
			t.Errorf("GET %s content-type = %q, want text/plain", path, ct)
		}
		body := rec.Body.String()
		for _, want := range []string{
			"# TYPE rostam_kv_gets_total counter",
			"# TYPE rostam_kv_evictions_live_total counter",
			"# TYPE rostam_kv_bytes_used gauge",
		} {
			if !strings.Contains(body, want) {
				t.Errorf("GET %s missing %q in:\n%s", path, want, body)
			}
		}
	}
}

// The KV metrics routes must not live under /v1/kv/: a literal segment beats a
// wildcard in ServeMux, so a route there would shadow GET /v1/kv/{key} and make
// a key of that name unreadable. This pins the behaviour rather than the path.
func TestKVMetricsDoesNotShadowKeyGet(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	const val = "stored-value-not-cache-stats"
	rec := do(t, h, "PUT", "/v1/kv/metrics", `{"value":"`+val+`"}`, nil)
	if rec.Code != http.StatusOK && rec.Code != http.StatusCreated && rec.Code != http.StatusNoContent {
		t.Fatalf("PUT /v1/kv/metrics = %d (%s)", rec.Code, rec.Body)
	}

	rec = do(t, h, "GET", "/v1/kv/metrics", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/kv/metrics = %d (%s)", rec.Code, rec.Body)
	}
	if got := rec.Body.String(); !strings.Contains(got, val) {
		t.Errorf("a key named \"metrics\" is shadowed by the metrics route; got:\n%s", got)
	}
	if strings.Contains(rec.Body.String(), "rostam_kv_gets_total") {
		t.Error("GET /v1/kv/metrics returned cache stats instead of the stored key")
	}
}
