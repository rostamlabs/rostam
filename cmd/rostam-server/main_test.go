// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"github.com/rostamlabs/rostam/authz"
)

// TestStaticKeyAuthenticator covers the -api-key fallback adapter: the matching
// token is a superuser (any op on any collection allowed), any other or empty
// token is denied. This is the legacy single-static-key mode adapted to the
// unified authz.Authenticator signature.
func TestStaticKeyAuthenticator(t *testing.T) {
	auth := staticKeyAuthenticator("secret")

	// Matching token → allowed regardless of op/collection (superuser).
	if !auth(authz.AuthRequest{Token: "secret", Op: "vector_insert", Args: []byte("\x00\x04docs")}) {
		t.Error("matching token should be allowed (superuser)")
	}
	if !auth(authz.AuthRequest{Token: "secret", Op: "vector_drop_collection"}) {
		t.Error("matching token should be allowed for admin ops too")
	}
	// Wrong token → denied.
	if auth(authz.AuthRequest{Token: "wrong", Op: "vector_search"}) {
		t.Error("wrong token must be denied")
	}
	// Empty token → denied (never matches an empty apiKey case because the flag
	// branch only builds this when apiKey != "").
	if auth(authz.AuthRequest{Token: "", Op: "vector_search"}) {
		t.Error("empty token must be denied")
	}
}

// TestExposedBind covers the open-bind fail-closed gate (#10): only a
// genuinely network-reachable address is "exposed"; disabled and loopback
// binds are not, so the dev default (loopback / disabled transports) never
// trips the no-auth refusal.
func TestExposedBind(t *testing.T) {
	cases := []struct {
		addr string
		want bool
	}{
		{"", false},                // transport disabled
		{"127.0.0.1:8080", false},  // loopback IPv4
		{"127.0.0.5:8080", false},  // loopback range
		{"[::1]:8080", false},      // loopback IPv6
		{"localhost:8080", false},  // loopback hostname
		{":8080", true},            // all interfaces (no host)
		{"0.0.0.0:8080", true},     // all interfaces IPv4
		{"[::]:8080", true},        // all interfaces IPv6
		{"10.0.0.1:8080", true},    // private but non-loopback
		{"203.0.113.7:8080", true}, // public IP
		{"db.internal:8080", true}, // hostname could resolve anywhere
		{"not-an-address", true},   // unparseable: fail safe as exposed
	}
	for _, c := range cases {
		if got := exposedBind(c.addr); got != c.want {
			t.Errorf("exposedBind(%q) = %v, want %v", c.addr, got, c.want)
		}
	}
}
