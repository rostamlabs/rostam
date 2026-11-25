# Claude Code plugin: code-linked memory

The `rostam-memory` plugin gives Claude Code memory that follows your code. A
fact saved against a file comes back on its own the next time Claude reads or
edits that file, and several Claude Code sessions can use the same memory at
once.

It builds on the [MCP server](mcp.md): the plugin registers it, adds two hooks,
and ships a short skill that tells the agent when to link a fact to a file.

## Install

`rostam-server` must be on your `PATH`; the plugin calls it by name and does not
bundle a binary. Then, in Claude Code:

```text
/plugin marketplace add rostamlabs/rostam
/plugin install rostam-memory@rostam
```

The plugin is not supported on Windows in this release, because the shared
daemon below is not.

## What it does

**Linking.** `remember` accepts an optional `path` (repo-relative, for example
`server/protocol.go`) and `symbol`. The memory stores the path and a hash of the
file's contents at that moment. A path outside the project root, or one that
escapes it with `..`, is rejected. `recall_for_file {path, namespace?, k?}`
returns up to `k` memories linked to one file (`k` defaults to 5), newest first.

**Injection.** Two hooks read memories and add them to Claude's context:

| Hook | When | What it adds |
|---|---|---|
| `session-start` | A session starts, resumes, forks, clears or compacts (see below) | The project's most recently updated memories |
| `pre-tool-use` | Before `Read`, `Edit`, `Write` or `MultiEdit` | Memories linked to the file about to be touched |

After a compaction or a clear the earlier injections are gone from Claude's
context, so those two start the session's "already injected" record afresh and
everything, including file-linked memories, can be injected again. A resumed
session keeps the same transcript, so its record is kept and nothing is repeated.

Each injection is capped at 5 memories and 1,500 characters, and a given memory
is injected at most once per session. When more exist, the injection ends with a
pointer to `recall_for_file` (to `recall` for the session-start list). The text is labelled as recalled data to verify,
not as instructions.

**Stale marking.** A linked memory is still shown when its file has changed, but
it is marked `[stale]`; if the file no longer exists it is marked
`[stale: file missing]`. Both marks need a file that was readable when the
memory was linked (that is when its hash is taken): a link to a file that did
not exist yet is marked `[stale]` once the file appears, and is not marked while
it stays missing. A linked file that has turned into a directory, become
unreadable, or become a symlink leading out of the project is marked `[stale]`. The same marks appear in `recall` and `recall_for_file`
results as `stale: true` and `stale: "file missing"`. Matching is exact on the
normalised repo-relative path, so a renamed file leaves its old memories marked
`[stale: file missing]` until they are re-saved.

**Namespace.** Memories are scoped to the project's namespace: the directory
name of its main checkout. A linked git worktree uses the name of its main
checkout too, so worktrees share the project's memories. The hooks and the MCP
server compute it the same way, and under `-autostart` the memory tools use it
whenever a call leaves `namespace` out, so the agent should simply omit it (the
server's instructions name the resolved namespace). A namespace passed
explicitly must be that main-checkout name, not a worktree's; anything else is
stored but never injected. `mcp -namespace <name>` overrides the default.

Paths are matched the same way on both sides: the project root is accepted as
given and with its symlinks resolved, and a file reached through a symlinked
prefix of the root (a symlinked working directory, say) maps to the same
repo-relative path. Symlinks inside the repository are not resolved: with
`docs -> site/docs`, a memory about `docs/x.md` stays linked to `docs/x.md`.
For the hash, symlinks that stay inside the project are followed; one that leads
outside it (in any path component) is never read.

## Shared daemon

The plugin starts the MCP server with `rostam-server mcp -autostart`. Instead of
embedding the engine (which allows one session at a time), it connects to a
local daemon on `127.0.0.1:7480`, starting it if needed. Two sessions starting
together end up with one daemon. Data lives in `~/.rostam/memory`; the daemon's
bearer token is in `~/.rostam/memory/token` (mode `0600`).

```sh
rostam-server memory-daemon        # run the daemon in the foreground
rostam-server memory-daemon -stop  # stop the running daemon
```

`-autostart` cannot be combined with `-data`, `-connect`, `-auth-token` or
`-tls-*`; it always uses its own daemon. `-project-root` overrides the project
root, which otherwise comes from `CLAUDE_PROJECT_DIR` or the working directory
(resolved to its git top-level).

The embedder is chosen by the MCP server process, from the same `ROSTAM_EMBED_*`
variables as plain `mcp` (see [MCP server](mcp.md)). The plugin passes those
variables through from the environment Claude Code is started in; unset, the
store is BM25-only. A store remembers the embedder it was created with, so they
must stay the same for the life of `~/.rostam/memory`.

**One user per host.** The daemon's port is fixed, so this release supports one
user per machine. Before sending its token, a session (and the hook) checks on
Linux, via `/proc/net/tcp`, that the process listening on `127.0.0.1:7480`
belongs to the current user. If another user owns the port, the hook stays
silent and the MCP server fails with "port 7480 is owned by another user (uid
N)" instead of starting a daemon. The check is best effort: where `/proc` is
unavailable (macOS) or the socket cannot be found, the connection proceeds as
before.

## Failure behaviour

Hooks never get in the way of a session. They have a hard 800 ms budget, and on
any error, timeout or unreachable daemon they print nothing and exit 0. The hook
never starts the daemon itself, so a session whose daemon is down simply gets no
injection until the MCP server starts it.

If the daemon crashes or is stopped while sessions are open, those sessions'
memory tools return errors and their hooks go silent; they do not reconnect or
restart the daemon on their own in this release. Restart the affected Claude
Code sessions (the first one to start brings the daemon back). Automatic
recovery is planned.

## Troubleshooting

**`rostam-server` is not on `PATH`.** The plugin calls the binary by name, so
the `rostam` MCP server fails to start and every `Read`, `Edit` and `Write`
shows a hook error (exit code 127). Install `rostam-server` into a directory on
the `PATH` Claude Code is started with, or, in a local copy of the plugin,
replace `rostam-server` in `.mcp.json` and `hooks/hooks.json` with the binary's
absolute path.

**"embedder mismatch".** The store in `~/.rostam/memory` was created with a
different embedder than the one this session is configured with. Either start
Claude Code with the same `ROSTAM_EMBED_*` variables the store was created with,
or run `rostam-server memory-daemon -stop`, move `~/.rostam/memory` aside, and
let the next session create a fresh store.

**"an embedded `rostam-server mcp` is using ~/.rostam/memory".** An older MCP
entry runs plain `rostam-server mcp` on the same directory, and a data
directory has one writer. See the next section.

## Migrating from an embedded rostam MCP entry

If you registered `rostam-server mcp` yourself before installing the plugin,
that entry embeds the engine over the same `~/.rostam/memory` the daemon uses,
and whichever starts second is refused. To move to the plugin:

1. Remove the old entry (`claude mcp remove <name>`, with the name and scope you
   added it under) and close the sessions that were using it.
2. If you set `ROSTAM_EMBED_*` for it, set the same variables in the environment
   you start Claude Code from, so the plugin's server uses the same embedder.
3. Start a session. The plugin's `mcp -autostart` starts the daemon on the
   existing directory, and your memories are still there.

Memories the old entry saved with no namespace live in `"default"`; recall them
with `namespace: "default"`, or re-save the ones you want injected under the
project's namespace.

## Without the plugin

Plain `rostam-server mcp` is unchanged: it embeds the engine over
`~/.rostam/memory` and supports one session at a time. Linking works there too
(`remember` with `path`, `recall_for_file`), but nothing is injected
automatically. To get the hooks without the plugin, register `rostam-server hook session-start` and
`rostam-server hook pre-tool-use` as Claude Code hooks yourself; they read the
hook JSON on stdin and accept `-connect`, `-token-file` and `-state-dir` before
the event name. `-connect` must be an `IP:port`: the hook sends the token only
after checking that the listener there belongs to the current user, and a
hostname (even `localhost`) or an empty host cannot be checked, so the hook
stays silent.
