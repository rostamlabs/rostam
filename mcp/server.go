// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/internal/buildinfo"
	"github.com/rostamlabs/rostam/semcache"
)

const protocolVersion = "2025-06-18"

// instructions is the MCP top-level `instructions` string returned from
// initialize. Clients inject it into the model's context, so it is how the
// server teaches an agent WHEN to use the memory tools — the tools alone don't
// convey that. Keep it short, imperative, and client-agnostic.
const instructions = `Rostam is persistent, namespaced memory for carrying facts across sessions so you don't re-derive what you already know. Use it well:
- At the start of a nontrivial task, call recall with the task's keywords BEFORE re-reading large files or re-exploring a codebase — recall is cheaper than rediscovery.
- When you learn a durable fact (a decision, a gotcha, a build/test command, an investigation result), call remember — one self-contained fact per call, phrased so it still makes sense when recalled later.
- Prefer recall over re-reading a large file to answer "what did we already establish about X".
- Never store secrets or credentials.
- Use list_namespaces and list_memories to orient when unsure what's stored.
- For live/in-flight state (a PR's status, what you're mid-task on), pass a stable "key" to remember (e.g. "pr-status:<branch>") so each update REPLACES the prior entry instead of piling up stale snapshots; recall and list show each memory's key and its updated time. Omit key for durable one-off facts.
- When a fact is about a specific file, pass path (repo-relative, e.g. "server/protocol.go") to remember, and symbol when it narrows the fact. Linked memories are shown automatically when that file is next touched; recall_for_file {path} lists everything linked to a file. A memory marked stale was linked to a file that has since changed or disappeared: verify it before relying on it.`

// instructionsFor is instructions plus the namespace rule for a server whose
// omitted-namespace default is defNS ("" = "default").
func instructionsFor(defNS string) string {
	if defNS != "" && defNS != defaultNS {
		return instructions + fmt.Sprintf("\n- Namespace: this server is scoped to the project namespace %q. Omit namespace to use it. It is this machine's registered name for the project: normally the main checkout's directory name (the same from any git worktree, never the worktree's own name), with a short hash suffix when another project here already has that name, so never derive it yourself; when passing one explicitly for this project, pass exactly %q.", defNS, defNS)
	}
	return instructions + "\n- Namespace per project: pass namespace = the project's main checkout directory name (in a git worktree, the main repository's directory name, not the worktree's), not \"default\", so facts stay scoped and don't collide across projects."
}

// Config configures a Server.
type Config struct {
	Store       rostam.Store      // required: embedded or remote engine
	Embedder    semcache.Embedder // nil => BM25-only mode (stub vectors, text recall)
	Destructive bool              // register delete/delete_by_filter for arbitrary collections
	Logger      *slog.Logger      // stderr logger; nil => slog.Default()
	ProjectRoot string            // absolute project root for path-linked memories; "" => paths stored but never hashed
	// DefaultNamespace is the namespace the memory tools use when a call omits
	// one; "" means "default". Set it to the project's namespace (the main
	// checkout's directory name) so memories land where the hook looks.
	DefaultNamespace string
	// ReadOnly guards the MEMORY tools only: the memory collection is never
	// created, the embedder-identity check is skipped, and remember/forget are
	// rejected. The generic db tools (upsert, create_collection, ...) stay
	// registered and writable; the only ReadOnly caller is the hook, which never
	// serves tools to a client.
	ReadOnly bool
}

type toolDef struct {
	Name        string
	Description string
	InputSchema map[string]any
	Handler     func(ctx context.Context, args json.RawMessage) (any, error)
}

// Server is one stdio MCP session over a rostam.Store.
type Server struct {
	store       rostam.Store
	emb         semcache.Embedder
	hybrid      bool // real embedder configured: recall uses dense+BM25 fusion
	destructive bool
	projectRoot string
	defNS       string // namespace for calls that omit one
	readOnly    bool
	log         *slog.Logger
	tools       []toolDef
	initialized bool

	// Lazy mcp_memory bootstrap. A mutex plus a succeeded-flag rather than a
	// sync.Once: Once would latch a transient first failure for the lifetime
	// of the process, and only success may be latched here. See ensureMemory.
	memMu    sync.Mutex
	memReady bool

	// dispatchMu is held for the whole of one tool call, and by Shutdown. It
	// is what lets a caller close the Store safely while Serve is still parked
	// on its input: see Shutdown.
	dispatchMu sync.Mutex
	stopped    bool
}

func NewServer(ctx context.Context, cfg Config) (*Server, error) {
	if cfg.Store == nil {
		return nil, errors.New("mcp: Config.Store is required")
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	s := &Server{store: cfg.Store, destructive: cfg.Destructive, projectRoot: cfg.ProjectRoot, readOnly: cfg.ReadOnly, log: log, defNS: cfg.DefaultNamespace}
	if s.defNS == "" {
		s.defNS = defaultNS
	}
	if cfg.Embedder != nil {
		s.emb, s.hybrid = cfg.Embedder, true
	} else {
		s.emb = semcache.NewStubEmbedder("stub", 64)
	}
	// A read-only client with no embedder never embeds, so it skips the
	// identity check; one that was given an embedder is held to it.
	if !s.readOnly || cfg.Embedder != nil {
		if err := s.checkEmbedderIdentity(ctx); err != nil {
			return nil, err
		}
	}
	s.registerMemoryTools()
	s.registerDBTools()
	if s.destructive {
		s.registerDestructiveTools()
	}
	return s, nil
}

func (s *Server) register(t toolDef) { s.tools = append(s.tools, t) }

// namespaceOr returns ns, or the server's default namespace when ns is "".
func (s *Server) namespaceOr(ns string) string {
	if ns == "" {
		return s.defNS
	}
	return ns
}

// nsDesc is a namespace argument's schema description, naming the default.
func (s *Server) nsDesc(what string) string {
	return fmt.Sprintf("%s (default %q)", what, s.defNS)
}

func textResult(v any) map[string]any {
	b, err := json.Marshal(v)
	if err != nil { // handler values are our own structs; this is a programming error
		return errResult(fmt.Errorf("mcp: marshal result: %w", err))
	}
	return map[string]any{"content": []map[string]any{{"type": "text", "text": string(b)}}}
}

func errResult(err error) map[string]any {
	return map[string]any{
		"content": []map[string]any{{"type": "text", "text": err.Error()}},
		"isError": true,
	}
}

// Serve runs the session until EOF on r. The caller owns the Store's
// lifecycle; Serve never closes it.
func (s *Server) Serve(r io.Reader, w io.Writer) error {
	ctx := context.Background()
	c := newConn(r, w)
	for {
		req, err := c.next()
		switch {
		case errors.Is(err, io.EOF):
			return nil
		case errors.Is(err, errLine):
			if werr := c.replyError(json.RawMessage("null"), codeParse, err.Error()); werr != nil {
				return werr
			}
			continue
		case err != nil:
			return err
		}
		// Envelope validation runs BEFORE the notification check: an object
		// with a broken envelope cannot be trusted to be a notification just
		// because its id did not parse, so it is answered (with a null id
		// where none was usable) rather than silently swallowed.
		if verr := req.validate(); verr != nil {
			if werr := c.replyError(errorID(req.ID), codeInvalidRequest, "invalid request: "+verr.Error()); werr != nil {
				return werr
			}
			continue
		}
		if req.ID == nil { // notification: never answered
			continue
		}
		s.dispatchMu.Lock()
		if s.stopped {
			s.dispatchMu.Unlock()
			return nil // Shutdown: the Store may already be closed
		}
		err = s.dispatch(ctx, c, req)
		s.dispatchMu.Unlock()
		if err != nil {
			return err // write failure: the client is gone
		}
	}
}

// Shutdown stops the session from handling any further request and waits for
// the one in flight, if any, to return. It is safe to call from another
// goroutine, and safe to call more than once.
//
// It exists so a caller can close the Store on a signal. Serve reads from a
// blocking stdin, and there is no portable way to interrupt that read from
// outside — closing the file does not reliably unblock it (the fd is not
// registered with the runtime poller) and would race the in-flight read
// against fd reuse. So Shutdown does not try to make Serve return. It
// guarantees the weaker but sufficient thing: after it returns, no handler is
// running and none will start, which is exactly the condition Store.Close
// needs. Serve's goroutine may stay parked on a read that never completes; in
// the signal path the process is exiting anyway.
func (s *Server) Shutdown() {
	s.dispatchMu.Lock()
	defer s.dispatchMu.Unlock()
	s.stopped = true
}

func (s *Server) dispatch(ctx context.Context, c *conn, req *request) error {
	switch req.Method {
	case "initialize":
		s.initialized = true
		return c.reply(req.ID, map[string]any{
			"protocolVersion": protocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			// Derived, not written down: a literal here goes stale at the
			// next release and the client shows a version that never existed.
			"serverInfo": map[string]any{"name": "rostam", "version": buildinfo.Version()},
			// Teaches the agent WHEN to use the memory tools; clients inject
			// this into the model's context.
			"instructions": instructionsFor(s.defNS),
		})
	case "ping":
		return c.reply(req.ID, map[string]any{})
	}
	if !s.initialized {
		return c.replyError(req.ID, codeInvalidRequest, "server not initialized")
	}
	switch req.Method {
	case "tools/list":
		list := make([]map[string]any, len(s.tools))
		for i, t := range s.tools {
			list[i] = map[string]any{"name": t.Name, "description": t.Description, "inputSchema": t.InputSchema}
		}
		return c.reply(req.ID, map[string]any{"tools": list})
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return c.replyError(req.ID, codeInvalidParams, "bad tools/call params: "+err.Error())
		}
		for _, t := range s.tools {
			if t.Name == p.Name {
				v, err := t.Handler(ctx, p.Arguments)
				if err != nil {
					s.log.Warn("tool failed", "tool", t.Name, "err", err)
					return c.reply(req.ID, errResult(err))
				}
				return c.reply(req.ID, textResult(v))
			}
		}
		return c.replyError(req.ID, codeInvalidParams, "unknown tool: "+p.Name)
	default:
		return c.replyError(req.ID, codeMethodNotFound, "unknown method: "+req.Method)
	}
}
