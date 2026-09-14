// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
	"testing"
)

func TestGroupForMatchesTheDocumentedAreas(t *testing.T) {
	for name, want := range map[string]string{
		"http":                   "Transports",
		"epoll-loops":            "Transports",
		"data":                   "Storage",
		"in-place-seqlock-reads": "Storage",
		"api-key":                "Authentication",
		"jwt-issuer":             "Authentication",
		"tls-node-cert":          "TLS", // TLS wins over Clustering: order matters
		"pb-auto-failover":       "Clustering",
		"nosync":                 "Clustering",
		"backup-bucket":          "Backups & cold tier",
		"cold-tier-after":        "Backups & cold tier",
		"log-format":             "Logging",
		"wasm-blob-retention":    "WASM",
		"version":                "Help",
		"some-future-flag-xyz":   "Other", // unmatched is visible, not missing
	} {
		if got := groupFor(name); got != want {
			t.Errorf("groupFor(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestFirstSentenceSkipsAbbreviations(t *testing.T) {
	cases := []struct{ in, want string }{
		// The bug this exists to prevent: "e.g." is not a sentence end, and
		// cutting there leaves a fragment ending in an open bracket.
		{"evict after it is untouched this long (e.g. 24h) and no longer. Next.",
			"evict after it is untouched this long (e.g. 24h) and no longer."},
		{"one sentence. two sentence.", "one sentence."},
		{"i.e. this stays. but not this", "i.e. this stays."},
		{"no trailing period at all", "no trailing period at all"},
	}
	for _, c := range cases {
		if got := firstSentence(c.in); got != c.want {
			t.Errorf("firstSentence(%q)\n  got  %q\n  want %q", c.in, got, c.want)
		}
	}
}

// A short description must come back untouched — no stray ellipsis. This is the
// regression the truncation bookkeeping can most easily reintroduce.
func TestSummarizeLeavesShortTextAlone(t *testing.T) {
	in := "gRPC listen address (\"\" = disabled)"
	if got := summarize(in, 96); got != in {
		t.Errorf("summarize altered short text:\n  got  %q\n  want %q", got, in)
	}
}

func TestSummarizeTruncatesOnAWordBoundaryWithEllipsis(t *testing.T) {
	in := strings.Repeat("alpha ", 40) // far over any limit, no sentence end
	got := summarize(in, 40)
	if !strings.HasSuffix(got, "…") {
		t.Errorf("no ellipsis on truncated text: %q", got)
	}
	if len(got) > 44 {
		t.Errorf("truncation exceeded the limit: %d chars in %q", len(got), got)
	}
	if strings.Contains(strings.TrimSuffix(got, "…"), "alph…") {
		t.Errorf("cut mid-word: %q", got)
	}
}

// Cutting inside a bracket must drop the bracket, not leave it hanging open.
func TestSummarizeDropsADanglingOpenBracket(t *testing.T) {
	got := summarize("idle-evict threshold for a collection (e.g", 96)
	if strings.Contains(got, "(") {
		t.Errorf("dangling open bracket survived: %q", got)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("trimmed text should be marked as truncated: %q", got)
	}
}

// A balanced bracket in the middle of a description must survive untouched.
func TestSummarizeKeepsBalancedBrackets(t *testing.T) {
	in := "use path-style addressing (<endpoint>/<bucket>/<key>)"
	if got := summarize(in, 96); got != in {
		t.Errorf("balanced brackets altered:\n  got  %q\n  want %q", got, in)
	}
}
