// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/mcp"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/semcache"
	"github.com/rostamlabs/rostam/semcache/localcatalog"
	"github.com/rostamlabs/rostam/tlsutil"
)

// mcpDataAuto is the -data sentinel default. runMcpCmd resolves it to
// $HOME/.rostam/memory; an explicit -data "" selects heap/ephemeral mode
// instead, and any other value is used as given. The sentinel exists so
// mcpSetup can stay pure (a plain "" default would be indistinguishable
// from an operator explicitly asking for heap mode).
const mcpDataAuto = "auto"

// errDataDirBusy marks the one lockDataDir failure that means what the lock
// exists to catch: another process already holds the data dir. Every other
// failure (a path that cannot be created, a read-only filesystem) is ordinary
// I/O and must not be reported as a conflict — a caller told to go looking for
// a second rostam-server that does not exist has been sent the wrong way.
// Declared here rather than in mcplock_unix.go so the platform variants do not
// each need their own copy.
var errDataDirBusy = errors.New("data directory is held by another process")

// storeFlags holds the flags common to any subcommand that connects to or
// embeds a rostam.Store: an embedded -data directory, or -connect plus its
// auth token and TLS options. It is shared between subcommands (mcpFlags
// embeds it today; a future llm-proxy subcommand will too) so the
// store-selection logic in connectStore only has to be written once.
type storeFlags struct {
	data, connect, authToken          string
	tlsCA, tlsCert, tlsKey, tlsServer string

	// beforeConnect, when set, vets the address before every dial (see
	// rostam.ClientConfig.BeforeConnect). Not a flag: set by -autostart and
	// the hook so the daemon token only ever goes to this user's listener.
	beforeConnect func(ctx context.Context, addr string) error
	// afterConnect vets each new connection before anything is sent on it
	// (see rostam.ClientConfig.AfterConnect).
	afterConnect func(ctx context.Context, addr string) error
}

// mcpFlags holds the parsed `mcp` subcommand flags. It is passed to mcpSetup
// as plain data so setup logic is unit-testable without touching flag.FlagSet
// or the real environment.
type mcpFlags struct {
	storeFlags
	destructive bool
}

// mcpRuntime is what mcpSetup produces: a ready store and (optionally) a
// hosted embedder. embedder is nil in BM25-only mode — mcp.NewServer falls
// back to a deterministic stub embedder in that case.
type mcpRuntime struct {
	store    rostam.Store
	embedder semcache.Embedder // nil in BM25-only mode
}

// mcpExtraFlags are the `mcp` flags that live outside mcpFlags.
type mcpExtraFlags struct {
	autostart   *bool
	projectRoot *string
	namespace   *string
	listModels  *bool
}

// defineMcpFlags registers every `mcp` flag on fs. Split from runMcpCmd so the
// real flag set can be parsed in tests.
func defineMcpFlags(fs *flag.FlagSet, fl *mcpFlags) mcpExtraFlags {
	var x mcpExtraFlags
	fs.StringVar(&fl.data, "data", mcpDataAuto,
		`data directory for embedded mode: "auto" (default) resolves to `+
			`~/.rostam/memory (created if missing); "" runs heap/ephemeral `+
			`(no persistence); any other value is used as given. Mutually `+
			`exclusive with -connect`)
	fs.StringVar(&fl.connect, "connect", "",
		"connect to a remote rostam-server instead of running embedded (host:port); mutually exclusive with -data")
	fs.StringVar(&fl.authToken, "auth-token", "",
		"bearer token for -connect (PREFER the ROSTAM_AUTH_TOKEN environment variable: a flag-passed secret is visible to other local users via /proc and shell history)")
	fs.StringVar(&fl.tlsCA, "tls-ca", "", "CA bundle PEM to verify the remote server's certificate (-connect)")
	fs.StringVar(&fl.tlsCert, "tls-cert", "", "client certificate PEM for mTLS (-connect; requires -tls-key)")
	fs.StringVar(&fl.tlsKey, "tls-key", "", "client private key PEM for mTLS (-connect; requires -tls-cert)")
	fs.StringVar(&fl.tlsServer, "tls-server-name", "", "expected server certificate name (SNI + verification) for -connect")
	fs.BoolVar(&fl.destructive, "enable-destructive", false, "register delete/delete_by_filter tools for arbitrary collections (off by default)")
	x.autostart = fs.Bool("autostart", false,
		"connect to the shared local memory daemon (127.0.0.1:7480, token from ~/.rostam/memory/token), starting it first if it is not running; cannot be combined with -data, -connect, -auth-token or the -tls-* flags")
	x.projectRoot = fs.String("project-root", "",
		"project root for code-linked memories (default: CLAUDE_PROJECT_DIR, else the working directory, resolved to its git top-level)")
	x.namespace = fs.String("namespace", "",
		`namespace the memory tools use when a call omits one (default: the project's namespace, the main checkout's directory name, under -autostart or -project-root; otherwise "default")`)
	x.listModels = fs.Bool("list-embed-models", false, "print the built-in local embedding models, then exit")
	return x
}

// visitedFlags is the set of flag names given explicitly on the command line.
func visitedFlags(fs *flag.FlagSet) map[string]bool {
	v := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { v[f.Name] = true })
	return v
}

// runMcpCmd implements `rostam-server mcp`: an MCP stdio server exposing
// memory + generic DB tools over a rostam.Store. Like runKeysCmd, it owns its
// own FlagSet and process exit. stdout is the MCP JSON-RPC wire, so every
// diagnostic here goes to stderr — never fmt.Print* to stdout.
func runMcpCmd(args []string) {
	fs := flag.NewFlagSet("mcp", flag.ExitOnError)
	var fl mcpFlags
	x := defineMcpFlags(fs, &fl)
	autostart, projectRoot, listModels := x.autostart, x.projectRoot, x.listModels
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}

	if *listModels {
		for _, n := range localcatalog.Names() {
			m, _ := localcatalog.Lookup(n)
			fmt.Printf("%-20s dim=%d license=%s  (%s)\n", m.Name, m.Dim, m.License, m.HFRepo)
		}
		return
	}

	// Was -data given explicitly on the command line, as opposed to sitting at
	// its "auto" default? mcpSetup's -data/-connect conflict check works on
	// plain non-empty-string tests, and "auto" is itself non-empty, so a
	// caller who passed ONLY -connect must not have -data resolved to a real
	// (equally non-empty) path below — that would spuriously look like both
	// flags were set once it reaches mcpSetup.
	dataExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "data" {
			dataExplicit = true
		}
	})

	var autoEmbedder *semcache.Embedder // built early by -autostart; reused below
	if *autostart {
		if err := checkAutostartConflicts(visitedFlags(fs)); err != nil {
			fatal(err.Error())
		}
		if !daemonSupported {
			fatal("mcp: -autostart: " + errDaemonUnsupported.Error())
		}
		dir, err := defaultMemoryDir()
		if err != nil {
			fatal("mcp: resolving home directory for -autostart", "err", err)
		}
		// Validate the embedder config BEFORE spawning: a typo'd ROSTAM_EMBED_*
		// must not leave a detached daemon behind on its way to failing.
		emb, err := embedderFromEnv(os.LookupEnv)
		if err != nil {
			fatal("mcp: setup failed", "err", err)
		}
		autoEmbedder = &emb
		if err := ensureDaemon(context.Background(), dir, memoryDefaultAddr, newDaemonSpawner(dir, memoryDefaultAddr)); err != nil {
			fatal("mcp: " + err.Error())
		}
		tok, err := ensureToken(dir)
		if err != nil {
			fatal("mcp: reading daemon token", "err", err)
		}
		// From here on this is exactly `-connect 127.0.0.1:7480` with the token:
		// the connect branch below skips claimDataDir.
		fl.connect, fl.authToken = memoryDefaultAddr, tok
		// This session lives as long as Claude Code does and redials whenever
		// the daemon restarts, so every new connection is vetted before the
		// token goes over it: on Linux, the listener's owner before the dial
		// and the accepting socket's owner after it. Elsewhere (no /proc
		// socket table) both checks are skipped; see listenerOwnerCheck.
		fl.beforeConnect, fl.afterConnect = ownerCheckBeforeConnect, ownerCheckAfterConnect
	}

	switch {
	case fl.connect != "" && !dataExplicit:
		// -data is still at its unset "auto" default; treat it as absent so
		// the store-selection logic below cleanly picks the -connect path.
		fl.data = ""
	case fl.data == mcpDataAuto:
		// -data "auto" is resolved to the home directory HERE, not in
		// mcpSetup: mcpSetup must stay pure (table-driven-testable with no
		// real filesystem or HOME dependency), so home-dir resolution belongs
		// in the subcommand entry point. -data "" (explicit heap mode) and any
		// other explicit value pass through unchanged. This branch is also
		// reached when -data was explicitly given as "auto" alongside -connect —
		// the conflict check below then correctly rejects it, since both are
		// non-empty.
		home, err := os.UserHomeDir()
		if err != nil {
			fatal("mcp: resolving home directory for -data", "err", err)
		}
		fl.data = filepath.Join(home, ".rostam", "memory")
	}

	// Claim the data dir before opening it. Only embedded mode with a real
	// directory needs this: heap mode (-data "") has nothing on disk to share,
	// and -connect's concurrency is the remote server's problem, not ours.
	unlock := func() error { return nil }
	if fl.connect == "" {
		var lerr error
		unlock, lerr = claimDataDir(fl.data)
		switch {
		case errors.Is(lerr, errDataDirBusy):
			fatal(dataDirBusyMsg(fl.data), "dir", fl.data, "err", lerr)
		case lerr != nil:
			// Not a conflict — a path that cannot be created, a read-only
			// filesystem, a full disk. Reporting it as "another process is
			// using this" would send the reader hunting for a process that
			// does not exist.
			fatal("mcp: claiming -data directory", "dir", fl.data, "err", lerr)
		}
	}

	var rt mcpRuntime
	var err error
	if autoEmbedder != nil {
		rt, err = mcpSetupWith(fl, *autoEmbedder, os.LookupEnv)
	} else {
		rt, err = mcpSetup(fl, os.LookupEnv)
	}
	if err != nil {
		_ = unlock()
		fatal("mcp: setup failed", "err", err)
	}

	// stdout is reserved for the MCP wire (JSON-RPC frames to the client), so
	// every log line goes to stderr via a plain text handler.
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	visited := visitedFlags(fs)
	root, nsDir := defaultProjectRoot(*projectRoot, os.Getenv, os.Getwd)
	resolvedNS := ""
	if !visited["namespace"] && (*autostart || *projectRoot != "") {
		ns, err := mcpProjectNamespace(nsDir, func(msg string) { log.Warn(msg) })
		if err != nil {
			_ = rt.store.Close()
			_ = unlock()
			fatal(err.Error())
		}
		resolvedNS = ns
	}
	srv, err := mcp.NewServer(context.Background(), mcp.Config{
		Store:            rt.store,
		Embedder:         rt.embedder,
		Destructive:      fl.destructive,
		Logger:           log,
		ProjectRoot:      root,
		DefaultNamespace: mcpDefaultNamespace(resolvedNS, *x.namespace, visited["namespace"], *autostart, *projectRoot != ""),
	})
	if err != nil {
		_ = rt.store.Close()
		_ = unlock()
		memDir, _ := defaultMemoryDir()
		fatal("mcp: server init failed", "err", mcpInitErr(err, *autostart, memDir))
	}

	// An MCP client that goes away closes stdin and Serve returns on its own.
	// A supervisor stopping the server sends SIGINT or SIGTERM instead, and the
	// default disposition for both is to kill the process immediately — before
	// the close below, which is where persistent collections write the
	// instant-restart sidecars that make their mmap files readable again. Every
	// memory since the last flush would be lost on an ordinary, deliberate
	// shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	serveDone := make(chan error, 1)
	go func() { serveDone <- srv.Serve(os.Stdin, os.Stdout) }()

	var serveErr error
	select {
	case serveErr = <-serveDone:
	case <-sigCh:
		// Shutdown waits for any tool call in flight and stops the loop from
		// starting another — the exact condition the close below needs. Serve's
		// goroutine may stay parked on a read from stdin that will never
		// complete, which is fine: nothing is running against the store, and
		// this process is about to exit. See mcp.Server.Shutdown for why
		// interrupting that read is not attempted.
		srv.Shutdown()
	}

	// Close before unlocking: the store's own shutdown writes to the data dir
	// (persistent collections flush their instant-restart sidecars), so releasing
	// the lock first would let a waiting process in mid-write.
	closeErr := rt.store.Close()
	_ = unlock()
	if serveErr != nil {
		fatal("mcp: serve failed", "err", serveErr)
	}
	if closeErr != nil {
		fatal("mcp: store close failed", "err", closeErr)
	}
}

// claimDataDir creates an embedded data directory and takes the single-writer
// lock on it, returning the closer that releases it. A "" dir is heap mode:
// nothing on disk to create or claim, so the closer is a no-op. Shared by any
// subcommand that can embed a store (mcp today; llm-proxy will too).
//
// The directory has to be created here rather than being left to the engine
// further down, because the lock file lives INSIDE it and is opened first —
// without this, a -data path that does not exist yet (an ordinary first run)
// fails on a missing directory. The resolved "auto" default and an explicit
// path both come through here, so there is one directory-creating step rather
// than two that can drift apart.
//
// It returns its errors instead of calling fatal so both paths are testable
// without a subprocess, the same reason the tiering validation is a returning
// function. Only a real lock conflict carries errDataDirBusy.
func claimDataDir(dir string) (func() error, error) {
	noop := func() error { return nil }
	if dir == "" {
		return noop, nil
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("creating -data directory %s: %w", dir, err)
	}
	return lockDataDir(dir)
}

// mcpSetup turns parsed flags + an environment lookup into a ready
// mcpRuntime. All validation (the -data/-connect conflict, the embedder env
// combination) runs before any I/O, so a misconfiguration is reported
// without ever touching disk or dialing a remote server. lookupEnv mirrors
// os.LookupEnv's signature so tests can substitute a fake map and never touch
// the real process environment.
func mcpSetup(fl mcpFlags, lookupEnv func(string) (string, bool)) (mcpRuntime, error) {
	if fl.data != "" && fl.connect != "" {
		return mcpRuntime{}, errors.New("mcp: use -data or -connect, not both")
	}

	embedder, err := embedderFromEnv(lookupEnv)
	if err != nil {
		return mcpRuntime{}, err
	}
	return mcpSetupWith(fl, embedder, lookupEnv)
}

// mcpSetupWith is mcpSetup with the embedder already built (-autostart builds
// it before spawning the daemon, so a bad embedder config never spawns one).
func mcpSetupWith(fl mcpFlags, embedder semcache.Embedder, lookupEnv func(string) (string, bool)) (mcpRuntime, error) {
	if fl.data != "" && fl.connect != "" {
		return mcpRuntime{}, errors.New("mcp: use -data or -connect, not both")
	}
	var err error
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		return mcpRuntime{}, fmt.Errorf("mcp: register ops: %w", err)
	}

	var store rostam.Store
	if fl.connect != "" {
		store, err = connectStore(fl.storeFlags, reg, lookupEnv)
	} else {
		// Size the cache for what this actually is: one person's memory store,
		// reached one tool call at a time over a pipe. The defaults are sized for a
		// server (256 shards, and a budget derived as a fraction of host RAM), which
		// here means a fresh -data dir lands at hundreds of shard directories and
		// tens of gigabytes of sparse mmap files before a single fact is stored —
		// alarming to look at, and a real cost on any filesystem without sparse-file
		// support. 8 shards still gives the op path room to spread, and 256 MiB is
		// far more than the handful of small bookkeeping keys the memory tools put
		// in KV (the memories themselves live in the vector collection).
		store, err = rostam.NewDirect(rostam.DirectConfig{
			DataDir: fl.data,
			Ops:     reg,
			Cache:   rostam.CacheConfig{NumShardsPerNode: 8, MaxMemoryBytes: 256 << 20},
		})
	}
	if err != nil {
		return mcpRuntime{}, err
	}

	return mcpRuntime{store: store, embedder: embedder}, nil
}

// embedderFromEnv reads the ROSTAM_EMBED_* variables and builds a hosted
// embedder, or nil for BM25-only mode. ROSTAM_EMBED_ENDPOINT is the trigger:
// unset (along with everything else) means BM25-only; set without a valid
// ROSTAM_EMBED_MODEL/ROSTAM_EMBED_DIM is a configuration error naming the
// exact missing/invalid variable, so a typo'd env var fails loud at startup
// rather than silently falling back to BM25-only. Shared by any subcommand
// that wants a hosted embedder (mcp today; llm-proxy will too).
func embedderFromEnv(lookupEnv func(string) (string, bool)) (semcache.Embedder, error) {
	localName, _ := lookupEnv("ROSTAM_EMBED_LOCAL")
	endpoint, _ := lookupEnv("ROSTAM_EMBED_ENDPOINT")
	if localName != "" {
		if endpoint != "" {
			return nil, errors.New("rostam-server: ROSTAM_EMBED_LOCAL and ROSTAM_EMBED_ENDPOINT are mutually exclusive")
		}
		return newLocalEmbedder(localName, lookupEnv)
	}
	if endpoint == "" {
		return nil, nil
	}
	model, _ := lookupEnv("ROSTAM_EMBED_MODEL")
	if model == "" {
		return nil, errors.New("rostam-server: ROSTAM_EMBED_ENDPOINT is set but ROSTAM_EMBED_MODEL is missing")
	}
	dimStr, _ := lookupEnv("ROSTAM_EMBED_DIM")
	if dimStr == "" {
		return nil, errors.New("rostam-server: ROSTAM_EMBED_ENDPOINT is set but ROSTAM_EMBED_DIM is missing")
	}
	dim, err := strconv.Atoi(dimStr)
	if err != nil {
		return nil, fmt.Errorf("rostam-server: ROSTAM_EMBED_DIM=%q is not a valid integer: %w", dimStr, err)
	}
	// Atoi is happy with 0 and negatives, and neither is a dimension. Caught
	// here so the documented fail-fast holds: otherwise startup succeeds and
	// the misconfiguration only surfaces later, as a collection-creation or
	// embedding failure inside the first memory tool call.
	if dim <= 0 {
		return nil, fmt.Errorf("rostam-server: ROSTAM_EMBED_DIM=%q must be a positive integer", dimStr)
	}
	apiKey, _ := lookupEnv("ROSTAM_EMBED_API_KEY")
	oe := semcache.NewOpenAIEmbedder(apiKey, model, dim)
	oe.Endpoint = endpoint
	// Without a timeout an unresponsive embedding endpoint wedges the tool call
	// forever, and an MCP client has no way to cancel it — the session just stops
	// answering. Five minutes matches objstore's HTTP client and leaves ample room
	// for a slow batch; the point is a bound, not a tight one.
	oe.HTTPClient = &http.Client{Timeout: 5 * time.Minute}
	return oe, nil
}

// connectStore builds the remote-mode Store for -connect: the auth token
// (flag, else ROSTAM_AUTH_TOKEN) and TLS config (built only when any -tls-*
// flag is set, so plaintext stays the zero-config default) feed
// rostam.NewClient. Shared by any subcommand that offers -connect (mcp
// today; llm-proxy will too).
func connectStore(fl storeFlags, reg *ops.Registry, lookupEnv func(string) (string, bool)) (rostam.Store, error) {
	token := fl.authToken
	if token == "" {
		if v, ok := lookupEnv("ROSTAM_AUTH_TOKEN"); ok {
			token = v
		}
	}

	var tlsCfg *tls.Config
	if fl.tlsCA != "" || fl.tlsCert != "" || fl.tlsKey != "" || fl.tlsServer != "" {
		cfg, err := tlsutil.ClientTLS(fl.tlsCA, fl.tlsCert, fl.tlsKey, fl.tlsServer)
		if err != nil {
			return nil, fmt.Errorf("rostam-server: -connect TLS config: %w", err)
		}
		tlsCfg = cfg
	}

	store, err := rostam.NewClient(rostam.ClientConfig{
		Servers:       []string{fl.connect},
		Ops:           reg,
		AuthToken:     token,
		TLSConfig:     tlsCfg,
		BeforeConnect: fl.beforeConnect,
		AfterConnect:  fl.afterConnect,
	})
	if err != nil {
		return nil, fmt.Errorf("rostam-server: connect to %q: %w", fl.connect, err)
	}
	return store, nil
}
