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
returns everything linked to one file, newest first.

**Injection.** Two hooks read memories and add them to Claude's context:

| Hook | When | What it adds |
|---|---|---|
| `session-start` | A session starts, resumes, clears or compacts | The project's most recently updated memories |
| `pre-tool-use` | Before `Read`, `Edit`, `Write` or `MultiEdit` | Memories linked to the file about to be touched |

Each injection is capped at 5 memories and 1,500 characters, and a given memory
is injected at most once per session. When more exist, the injection ends with a
pointer to `recall_for_file`. The text is labelled as recalled data to verify,
not as instructions.

**Stale marking.** A linked memory is still shown when its file has changed, but
it is marked `[stale]`; if the file no longer exists it is marked
`[stale: file missing]`. The same marks appear in `recall` and `recall_for_file`
results as `stale: true` and `stale: "file missing"`. Matching is exact on the
normalised repo-relative path, so a renamed file leaves its old memories marked
`[stale: file missing]` until they are re-saved.

**Namespace.** Memories are scoped to the project directory's name. A linked git
worktree uses the name of its main checkout, so worktrees share the project's
memories.

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

## Failure behaviour

Hooks never get in the way of a session. They have a hard 800 ms budget, and on
any error, timeout or unreachable daemon they print nothing and exit 0. The hook
never starts the daemon itself, so a session whose daemon is down simply gets no
injection until the MCP server starts it.

## Without the plugin

Plain `rostam-server mcp` is unchanged: it embeds the engine over
`~/.rostam/memory` and supports one session at a time. Linking works there too
(`remember` with `path`, `recall_for_file`), but nothing is injected
automatically. To get the hooks without the plugin, register `rostam-server hook session-start` and
`rostam-server hook pre-tool-use` as Claude Code hooks yourself; they read the
hook JSON on stdin and accept `-connect`, `-token-file` and `-state-dir` before
the event name.
