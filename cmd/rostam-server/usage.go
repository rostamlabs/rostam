// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
)

// Grouped help. `-h` on 59 alphabetically-sorted flags — several with
// paragraph-length descriptions explaining a tuning trade-off — is the first
// thing a new operator meets, and it is unreadable. This prints the same flags
// under the areas docs/server/running.md already uses, one line each, with
// -help-all left as the escape hatch for the full text.
//
// Flags are matched to areas by NAME, not by an explicit list. A list would
// need updating for every new flag and would silently drop the ones nobody
// remembered; name rules degrade instead — an unmatched flag lands in "Other",
// which is visible rather than missing.

type flagGroup struct {
	title string
	// match reports whether a flag belongs to this group. The first group that
	// matches wins, so order encodes precedence: -tls-node-cert is TLS, not
	// Clustering, even though a node is a clustering concept.
	match func(name string) bool
}

func hasAnyPrefix(name string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

func isAnyOf(name string, names ...string) bool {
	return slices.Contains(names, name)
}

// flagGroups mirrors the "Flag reference by area" section of the running-the-
// server docs, so the two do not tell different stories.
var flagGroups = []flagGroup{
	{"Transports", func(n string) bool {
		return isAnyOf(n, "http", "grpc", "tcp") || hasAnyPrefix(n, "epoll")
	}},
	{"Storage", func(n string) bool {
		return isAnyOf(n, "data", "shards", "config", "disable-cold-compaction",
			"relocating-eviction", "relocate-reserve-interval", "in-place-same-size-update",
			"in-place-seqlock-reads", "sieve-visited-bit")
	}},
	{"Authentication", func(n string) bool {
		return hasAnyPrefix(n, "jwt-") ||
			isAnyOf(n, "api-key", "keys-file", "internal-token", "audit-log",
				"tenant-isolation", "insecure", "node-cn-allowlist")
	}},
	{"TLS", func(n string) bool { return hasAnyPrefix(n, "tls-") }},
	{"Clustering", func(n string) bool {
		return hasAnyPrefix(n, "pb-", "raft-", "replication-") ||
			isAnyOf(n, "cluster", "node-id", "bootstrap", "peers", "min-isr",
				"persistent-vectors", "reconfigure", "nosync", "volatile-log")
	}},
	{"Backups & cold tier", func(n string) bool {
		return hasAnyPrefix(n, "backup-", "cold-tier-", "s3-") ||
			isAnyOf(n, "restore", "allow-missing-shards")
	}},
	{"Logging", func(n string) bool {
		return hasAnyPrefix(n, "log-") || isAnyOf(n, "access-log")
	}},
	{"WASM", func(n string) bool { return hasAnyPrefix(n, "wasm-") }},
	{"Help", func(n string) bool { return isAnyOf(n, "version", "help-all") }},
}

// groupFor returns the area a flag belongs to, or "Other" when no rule claims
// it. "Other" is deliberate: a new flag should be visible in the wrong place
// rather than absent from the right one.
func groupFor(name string) string {
	for _, g := range flagGroups {
		if g.match(name) {
			return g.title
		}
	}
	return "Other"
}

// summarize reduces a flag's description to its first sentence, so the grouped
// listing stays one line per flag. The long-form text is what -help-all and the
// docs are for; several descriptions here run to a paragraph because they carry
// a genuine trade-off, and that belongs in prose, not in a terminal dump.
// sentenceAbbrevs are the abbreviations whose trailing dot is NOT a sentence
// end. Without these, "evict after it is untouched this long (e.g. 24h)" gets
// cut at "e.g." and the reader is shown a fragment ending in an open bracket.
var sentenceAbbrevs = []string{"e.g", "i.e", "vs", "etc", "approx", "cf", "Fig", "no"}

// firstSentence returns s up to the first sentence-ending period, skipping the
// dots inside known abbreviations.
func firstSentence(s string) string {
	for i := 0; i+1 < len(s); i++ {
		if s[i] != '.' || s[i+1] != ' ' {
			continue
		}
		// The abbreviation occupies the len(a) bytes ENDING at the dot, so the
		// window is s[i-len(a):i] — "(e.g." with the dot at 4 must compare
		// s[1:4] == "e.g", not s[2:5] == ".g.".
		abbrev := false
		for _, a := range sentenceAbbrevs {
			if i >= len(a) && strings.EqualFold(s[i-len(a):i], a) {
				abbrev = true
				break
			}
		}
		if !abbrev {
			return s[:i+1]
		}
	}
	return s
}

// closeDanglingParen trims a fragment that ends inside an unclosed bracket.
// "... this long (e.g" reads as broken text rather than as a truncation.
func closeDanglingParen(s string) string {
	if o, c := strings.LastIndex(s, "("), strings.LastIndex(s, ")"); o > c {
		return strings.TrimRight(s[:o], " ")
	}
	return s
}

// summarize reduces a flag's description to its first sentence, so the grouped
// listing stays one line per flag. The long-form text is what -help-all and the
// docs are for; several descriptions here run to a paragraph because they carry
// a genuine trade-off, and that belongs in prose, not in a terminal dump.
func summarize(usage string, max int) string {
	s := strings.Join(strings.Fields(usage), " ") // collapse newlines/indent
	s = firstSentence(s)
	truncated := false
	if len(s) > max {
		cut := s[:max]
		if sp := strings.LastIndex(cut, " "); sp > max/2 {
			cut = cut[:sp]
		}
		s, truncated = cut, true
	}
	// Applies to both paths: a first-sentence cut can land inside a bracket too.
	if t := closeDanglingParen(s); t != s {
		s, truncated = t, true
	}
	s = strings.TrimRight(s, " ,;:")
	if truncated {
		s += "…"
	}
	return s
}

// printGroupedUsage writes the grouped listing. helpAll switches to the stock
// alphabetical dump with every description in full.
func printGroupedUsage(w io.Writer, fs *flag.FlagSet, helpAll bool) {
	// Usage output cannot act on a write error — the destination is the very
	// stream the error would be reported on — so it is swallowed in one place
	// rather than by dropping eight return values individually.
	p := func(format string, a ...any) { _, _ = fmt.Fprintf(w, format, a...) }

	p("rostam-server — a vector database and key-value store over REST, gRPC and TCP.\n\n")
	p("Usage:\n  rostam-server [flags]\n  rostam-server keys <add|revoke|list> [flags]\n  rostam-server mcp [flags]\n  rostam-server llm-proxy [flags]\n\n")

	if helpAll {
		fs.SetOutput(w)
		fs.PrintDefaults()
		return
	}

	byGroup := make(map[string][]*flag.Flag)
	for _, g := range flagGroups {
		byGroup[g.title] = nil
	}
	fs.VisitAll(func(f *flag.Flag) {
		g := groupFor(f.Name)
		byGroup[g] = append(byGroup[g], f)
	})

	order := make([]string, 0, len(flagGroups)+1)
	for _, g := range flagGroups {
		order = append(order, g.title)
	}
	order = append(order, "Other")

	for _, title := range order {
		flags := byGroup[title]
		if len(flags) == 0 {
			continue
		}
		sort.Slice(flags, func(i, j int) bool { return flags[i].Name < flags[j].Name })
		p("%s:\n", title)
		for _, f := range flags {
			name := "-" + f.Name
			// Several descriptions already spell out their default; appending
			// ours produced "(default info) (default info)".
			def := ""
			summary := summarize(f.Usage, 96)
			if f.DefValue != "" && f.DefValue != "false" && f.DefValue != "0" &&
				!strings.Contains(strings.ToLower(summary), "default") {
				def = fmt.Sprintf(" (default %s)", f.DefValue)
			}
			p("  %-26s %s%s\n", name, summary, def)
		}
		p("\n")
	}

	p("Descriptions are shortened. Full text: rostam-server -help-all\n")
	p("Every flag also reads ROSTAM_<NAME> from the environment (-pb-addr → ROSTAM_PB_ADDR).\n")
	p("Docs: https://docs.rostamlabs.com/server/running/\n")
}
