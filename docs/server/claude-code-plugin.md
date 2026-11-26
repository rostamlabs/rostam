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

The plugin runs on Linux only in this release, because the shared daemon below
does. The daemon's store is persistent, and the engine's persistent
(file-mapped) storage exists on Linux and Windows only; the daemon itself is
not ported to Windows yet. On macOS and the BSDs `mcp -autostart` and
`memory-daemon` fail at once with an error saying so.

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

**Namespace.** Memories are scoped to the project's namespace. Each project is
identified by its canonical main-repository path (symlinks resolved; a linked
git worktree resolves to its main repository; a repository with a separate git
directory is identified by that directory, so its worktrees stay together and
two such repositories never merge; outside git, the project directory), and a
per-machine registry, `~/.rostam/memory/namespaces.json`, assigns each identity
a namespace:

- the first project seen with a given directory name gets that plain name, so
  memories saved before the registry existed keep working;
- a different project with the same directory name gets the name plus a short
  hash, for example `api-1a2b3c4d`, so two unrelated checkouts called `api` no
  longer share or overwrite each other's memories (the registry checks the
  hashed name against every assignment and lengthens it if another project
  already holds it);
- an assignment never changes once made;
- a project with no assignment takes over the plain name only when the
  recorded owner's directory no longer exists.

The hooks and the MCP server use the same function and the same registry, and
under `-autostart` the memory tools use the result whenever a call leaves
`namespace` out, so the agent should simply omit it (the server's instructions
name the resolved namespace). Explicit `namespace` arguments and
`mcp -namespace <name>` are used as given.

Limits:

- After upgrading, whichever same-named project opens first gets the plain
  name, and with it every memory saved under that name before. If that was the
  wrong project, move the registry entry by hand (or the memories with
  `namespace` arguments).
- Passing `namespace: "<basename>"` explicitly from the second project still
  writes into the first project's memory. Leave `namespace` out.
- Deleting, moving or unmounting the owning project lets the next same-named
  project that has no assignment yet take the plain name (and its memories).
  The former owner, if it comes back, gets a hashed name.
- The registry is per machine: the same project can have a different name on
  another machine.
- If the registry is unreadable or corrupt, or its lock fails, the project's
  hashed name is used and the file is left untouched; `mcp` says so on stderr
  and the hooks use the same name. That fallback name is not checked against
  other projects (the registry could not be read), so in this state two
  projects could in principle share a name (a 32-bit hash collision). Memories
  saved in this state are stored under that fallback name; once the registry
  works again the project uses its registered name, so unless that is the same
  hashed name, reach them with an explicit `namespace` argument (the fallback
  name is in `mcp`'s stderr warning). Repairing
  `namespaces.json` restores the checked assignments; moving it aside instead
  rebuilds the registry from scratch, so the first same-named project to open
  then takes the plain name. A lock failure is about `namespaces.lock` or the
  `~/.rostam/memory` directory itself (permissions, a filesystem without
  locking): fix that, since moving `namespaces.json` does not help.
- If the registry stays locked, or git does not answer, past the caller's
  deadline, no name is decided and none is registered: that hook injects
  nothing (800 ms budget), and `mcp` refuses to start after 30 s with an error
  telling you to retry or pass `-namespace`, rather than running the session
  under a name the hooks would not use.

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

### Durability

The daemon runs its server with `-durable-collections mcp_memory`, so a
`remember` or `forget` that was acknowledged survives the daemon being killed,
at the price of a disk sync per change. The rules, and the one exception
(replacing a memory and being killed before the call answers), are the same as
for plain `mcp`: see [What survives a crash](mcp.md#what-survives-a-crash).

- **Existing stores** are [converted](mcp.md#upgrading-an-existing-store) when
  the new daemon first starts. If that fails, the daemon still runs, and
  `~/.rostam/memory/daemon.log` carries a `not crash-safe` warning. The
  conversion happens before the daemon answers; if a session gives up with "did
  not become ready" and the log shows no error, start another session.
- **A failed log write** (a full disk, for example) makes memory refuse further
  writes until the daemon restarts. Free the space, run
  `rostam-server memory-daemon -stop`, and start a new session.
- **A damaged log** keeps the daemon from starting. Sessions report that it did
  not become ready, and `daemon.log` names the file
  (`~/.rostam/memory/vectors/default/mcp_memory.wal`) and the offset. With no
  daemon running, follow [A damaged log](mcp.md#a-damaged-log).

### Replacing an older daemon

A daemon left running by an earlier build does not keep memory durable. When
`mcp -autostart` finds one, it stops it cleanly, the way `memory-daemon -stop`
does, which saves its data, and then starts a new one. Sessions starting
together replace it once. A daemon that already runs with durable memory is
never replaced, and the hooks never stop or start a daemon. If the old daemon
cannot be stopped safely (it cannot be identified, for example), the session
does not force it: it prints a warning and uses it, without durable memory. Run
`rostam-server memory-daemon -stop` and start a new session to replace it. An
old daemon that takes more than 30 seconds to stop is killed by its own
supervisor, as under `memory-daemon -stop`, and loses what it had not saved.

**One user per host.** The daemon's port is fixed, so this release supports one
user per machine. On Linux, the clients (the MCP session and the hook) check
the kernel socket table (`/proc/net/tcp`) before the token goes anywhere:
before each dial, that the listener on `127.0.0.1:7480` belongs to the current
user, and after it, before the first request is sent, that every connection
this user has open to that port was accepted by a socket of the current user
(the new one is always among them). If either belongs to another user, or
cannot be found or read (including `/proc` missing or hidden on Linux), the
token is not sent. The hook then stays silent.
`mcp -autostart` fails with "port 7480 is owned by another user (uid N)"
without trying to start a daemon; when the owner merely cannot be verified it
treats the daemon as not running, may try to start one (which cannot bind a
port that is in use), and fails with a "did not become ready" error. The hook's own check
needs the same socket table: pointed with `-connect` at a server on a platform
without one, it sends the token unverified.

## Failure behaviour

Hooks never get in the way of a session. They have a hard 800 ms budget, and on
any error, timeout or unreachable daemon they print nothing and exit 0. The hook
never starts the daemon itself, so a session whose daemon is down simply gets no
injection until the MCP server starts it.

If the daemon crashes or is stopped while sessions are open, those sessions'
memory tools return errors and their hooks go silent; they do not restart the
daemon on their own in this release. Start a new Claude Code session to bring it
back. The open sessions then reconnect on a later call (the first call after a
restart may still fail). Automatic restart is planned.

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

Plain `rostam-server mcp` embeds the engine over `~/.rostam/memory` and
supports one session at a time. Its memory is
[durable](mcp.md#what-survives-a-crash) too. Linking works there too
(`remember` with `path`, `recall_for_file`), but nothing is injected
automatically. To get the hooks without the plugin, register `rostam-server hook session-start` and
`rostam-server hook pre-tool-use` as Claude Code hooks yourself; they read the
hook JSON on stdin and accept `-connect`, `-token-file`, `-state-dir` and
`-project-root` before the event name. The hooks find the project the way
`mcp` does: `-project-root` as given, else `CLAUDE_PROJECT_DIR`, else the
session's working directory, resolved to its git top-level; if you pass
`mcp -project-root`, pass the same value to the hooks so file links match. On Linux the hook sends the token only after the owner check
above, which needs `-connect` to be an `IP:port`: a hostname (even
`localhost`) or an empty host cannot be checked, so the hook stays silent.
Elsewhere the check does not run and the connection proceeds unverified.
