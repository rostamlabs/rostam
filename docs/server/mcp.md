# MCP server

`rostam-server mcp` runs Rostam as a [Model Context Protocol](https://modelcontextprotocol.io/)
server over stdio, giving an MCP client (Claude Code, Claude Desktop, Cursor,
…) persistent agent memory and generic vector-DB tools. There is no daemon to
run and nothing to sign up for: the process embeds the engine directly
(`rostam.NewDirect`) and persists to a local directory, so `claude mcp add`
and the agent has durable memory a few seconds later.

Memory works with zero configuration. With no embedder set, `remember` and
`recall` run on Rostam's built-in BM25 full-text search — no embedding API
key, no external service. Pointing `ROSTAM_EMBED_ENDPOINT` at any
OpenAI-compatible `/embeddings` endpoint (OpenAI, Azure, Ollama, LM Studio,
TEI, LiteLLM) upgrades recall to hybrid dense+BM25 fusion with the same
tools and the same call shapes — nothing about how the agent uses memory
changes, only how well it ranks.

## Quickstart

**Claude Code:**

```sh
claude mcp add rostam -- rostam-server mcp
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rostam": {
      "command": "rostam-server",
      "args": ["mcp"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "rostam": {
      "command": "rostam-server",
      "args": ["mcp"]
    }
  }
}
```

All three launch the same binary the same way: no flags are required. Memory
persists to `~/.rostam/memory` by default, so it survives across sessions:
close the client, reopen it tomorrow, and `recall` still finds what you told
it. Only `-data ""` opts out, which runs entirely in memory.

A data directory has one writer. Sessions that come and go **sequentially**
all share `~/.rostam/memory` and see each other's memories, but two clients
running at the **same time** cannot both embed the engine over it — the second
one is refused at startup with an error saying so. To give concurrent clients
one shared memory, run a single `rostam-server` and point each client at it
with [`-connect`](#remote-mode).

### What survives a crash

The memory collection (`mcp_memory`) has a write-ahead log, in an embedded
session with a data directory and in the
[shared daemon](claude-code-plugin.md#durability). Every `remember` and
`forget` is written to that log and synced to disk before the tool call answers,
so a change that was acknowledged is still there after the process holding the
store is killed (`SIGKILL`, an out-of-memory kill, a crash). The price is one
disk sync per `remember` and one per forgotten memory. A killed process is what
is tested. Safety against power loss rests on that sync and is not claimed as
tested.

Replacing a memory (a `remember` with a `key` that already exists, or of
content identical to an earlier one) is written to the log as one record. If
the process is killed, or the log write fails, before that call has answered,
the store holds either the old version or the new one, never neither; once the
call has answered, it holds the new one. A replace refused before anything is
written (invalid input, a version conflict, a full collection or the insert
rate limit) leaves the old version; one that failed writing or syncing the log
may or may not have replaced it, as for any failed change below.

On a clean shutdown (the client closing the connection, `SIGINT` or `SIGTERM`)
the log is folded into a snapshot and emptied. The same happens at startup when
the log has grown past 16 MiB. The memory collection is held in RAM and loaded
from the snapshot and the log at startup.

If writing the log fails (a full disk, for example), the collection refuses
further writes until the process holding the store is restarted, and the
shutdown does not fold the log into the snapshot: the files are left as they
were, and the next start replays them. Free the space first, then restart the
session, or for the shared daemon run `rostam-server memory-daemon -stop` and
start a new session. A change that was reported as failed may or may not be
there afterwards, so check before relying on it.

Only `mcp_memory` gets this. Collections made with the generic tools keep their
own mode: a `persistent` one is saved at a clean shutdown, and a forced stop can
lose what changed since. With `-data ""` nothing is durable, by design. Over
`-connect`, durability is the server's business: see
[Remote mode](#remote-mode).

### Upgrading an existing store

A memory store written by an earlier release is converted to the log form the
first time it is opened. There is nothing to run. The live memories are copied
into a new snapshot and the old files are replaced only once that snapshot is
complete. Memories already lost to an earlier forced stop cannot be recovered.

If the conversion fails, the store still opens with the collection in its old
form, a warning says it is `not crash-safe` and why, and the next start tries
again. Until it succeeds, a forced stop can lose recent memories. A log
found beside the old form is not replayed. An empty one is removed; any other
is renamed to `<file>.stale-<number>` and a warning names it. While a large store converts,
`mcp -autostart` keeps waiting for the daemon it started. The warning
goes to stderr. A shared daemon that `mcp -autostart` started has it in
`~/.rostam/memory/daemon.log`; a `memory-daemon` you started yourself writes it
to its own stderr. If a conversion was interrupted and the new snapshot cannot
be loaded, the store fails to open instead of falling back to the old files.
It also fails to open, keeping every file, when a conversion cannot tell which
form the collection is in afterwards: the new manifest could not be written,
and the old one could not be read back or put back.

### A damaged log

At startup the log is checked as it is replayed.

**A half-written tail**, bad bytes at the end of the file with no complete
record after them, is what a stop in the middle of a write leaves. It is cut
off, the removed bytes are saved next to the log as `<file>.torn-<number>`, a
warning is logged, and the store opens. If the bytes cannot be saved (a full
disk, for example), the store does not open; free space and start again.
Nothing deletes the saved files, and one can hold the content of a memory you
later forgot, so remove them yourself once you no longer need them.

**Damage anywhere else** (a complete record follows the bad bytes, or a record
passes its checksum but cannot be decoded) makes the store refuse to open. The
file is left as it is, and the error names it and the position:

```text
vector: write-ahead log damaged: /home/you/.rostam/memory/vectors/default/mcp_memory.wal at offset 18342
```

To recover:

1. Stop every process using the data directory.
2. Keep a copy of the named file.
3. Truncate the file at the named offset, in bytes:
   `truncate -s 18342 /path/named/in/the/error.wal`.
4. Start again.

This keeps every record before the offset and loses the ones after it. One rare
power-loss case is refused the same way although nothing acknowledged is
missing: a later, unacknowledged record reached the disk and an earlier one did
not.

A log holding a record type this build does not know, as one written by a newer
release may, also refuses to open. Run the newer release instead of truncating.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `-data` | `auto` | Embedded data directory. `auto` resolves to `~/.rostam/memory` (created if missing); `""` runs heap/ephemeral mode (nothing persists past the process, and nothing is durable); any other value is used as given. Mutually exclusive with `-connect`. |
| `-connect` | disabled | Remote mode: `host:port` of a running `rostam-server` to connect to over the binary TCP protocol, instead of embedding the engine. Mutually exclusive with `-data`. |
| `-auth-token` | — | Bearer token for `-connect`. Prefer `ROSTAM_AUTH_TOKEN` — a flag-passed secret is visible to other local users via `/proc` and shell history. |
| `-tls-ca` | — | CA bundle PEM to verify the remote server's certificate (`-connect`). |
| `-tls-cert` / `-tls-key` | — | Client certificate/key PEM for mTLS (`-connect`; both required together). |
| `-tls-server-name` | — | Expected server certificate name (SNI + verification) for `-connect`. |
| `-autostart` | `false` | Connect to the shared local memory daemon on `127.0.0.1:7480` (token from `~/.rostam/memory/token`), starting it first if needed. This is what the [Claude Code plugin](claude-code-plugin.md) runs. Cannot be combined with `-data`, `-connect`, `-auth-token` or `-tls-*`. Linux only in this release (on macOS, the BSDs and Windows it fails at once: the daemon's persistent store needs the engine's file-mapped storage, which is Linux/Windows-only, and the daemon is not ported to Windows). Intended for single-user hosts: the `mcp` client refuses to send the token to a port, or a connection, that another local user owns, as read from `/proc`. |
| `-project-root` | see meaning | Project root for code-linked memories. Default: `CLAUDE_PROJECT_DIR`, else the working directory, resolved to its git top-level. |
| `-namespace` | see meaning | Namespace the memory tools use when a call omits one. Default: under `-autostart` or an explicit `-project-root`, the project's namespace from the per-machine registry `~/.rostam/memory/namespaces.json`, exactly as the hooks compute it — normally the directory name of the project's MAIN checkout (a linked git worktree uses its repository's), or that name plus a short hash when another project on the machine already has it (see [Claude Code plugin](claude-code-plugin.md#what-it-does) for the rules and limits); otherwise `"default"`, so a plain embedded `mcp` behaves as it always has. An explicit `-namespace` always wins (`-namespace ""` means `"default"`). The resolved name is stated in the server's `instructions`. Memories stored earlier under `"default"` do not appear under the project namespace; pass `-namespace default` or a `namespace` argument to reach them. |
| `-enable-destructive` | `false` | Register the `delete` and `delete_by_filter` tools for arbitrary collections. Without it, those two tools are absent from `tools/list` entirely — not present-but-refusing. |

`-data` and `-connect` are mutually exclusive: pass one or the other, never
both.

## Embedder configuration (environment only)

Unlike the flags above, the embedder is configured entirely through
environment variables — that matches how MCP clients pass configuration to a
server (the `env` block in the JSON snippets above), rather than through
command-line arguments baked into `args`.

| Variable | Required | Meaning |
|---|---|---|
| `ROSTAM_EMBED_LOCAL` | trigger | In-process, pure-Go local embedder — a catalog name (`minilm-l6-v2`, …) or any Hugging Face `org/model` id; no endpoint or API key. See [Local embeddings](#local-embeddings) below. Mutually exclusive with `ROSTAM_EMBED_ENDPOINT`. |
| `ROSTAM_EMBED_ENDPOINT` | trigger | OpenAI-compatible `/embeddings` URL (hosted embedder). |
| `ROSTAM_EMBED_MODEL` | if endpoint set | Model id sent to the endpoint. |
| `ROSTAM_EMBED_DIM` | if endpoint set | Output embedding dimension, as an integer. |
| `ROSTAM_EMBED_API_KEY` | optional | Bearer token for the endpoint (local endpoints like Ollama typically don't need one). |

With neither trigger set, the zero-config default is BM25 full-text alone:
`remember`/`recall` and the generic `search` tool still work, on keyword
matching. `ROSTAM_EMBED_LOCAL` and `ROSTAM_EMBED_ENDPOINT` are mutually
exclusive (setting both fails at startup); setting `ROSTAM_EMBED_ENDPOINT`
without `ROSTAM_EMBED_MODEL` or `ROSTAM_EMBED_DIM` fails at startup with a
message naming the exact missing variable — never a silent fall-back to
BM25-only.

Example: a local Ollama embedding model, wired through the Claude Desktop
config from above:

```json
{
  "mcpServers": {
    "rostam": {
      "command": "rostam-server",
      "args": ["mcp"],
      "env": {
        "ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
        "ROSTAM_EMBED_MODEL": "nomic-embed-text",
        "ROSTAM_EMBED_DIM": "768"
      }
    }
  }
}
```

!!! note "Changing embedders fails loudly, not silently"

    The embedder identity (model, dimension, BM25-only vs. hybrid) a data
    directory's memory was first created with is recorded and checked on
    every subsequent start. A mismatch — a different model, a different
    `ROSTAM_EMBED_DIM`, or adding/removing the embedder entirely — refuses to
    start rather than silently mixing embedding spaces or crashing deep
    inside the vector index on a dimension mismatch. The ways out: restore
    the embedder configuration the store was created with, point `-data` at a
    different directory, or move the existing data directory aside to start
    over. There is no automatic re-embed migration in this release.

    A store whose record is missing is not locked out. The first memory tool
    call checks the configured embedder against the store and records it
    again. The vector dimension must match. Then up to three stored memories
    are embedded again, and at least half of them, and at least one, must
    reproduce their stored vectors. An empty store offers only the dimension
    to check. If the check fails, or the store has entries but none with
    content to embed, the call fails with an embedder mismatch and nothing is
    recorded or changed. The hooks never write the record. Over `-connect`
    this needs a recent server: see [Remote mode](#remote-mode).

## Local embeddings

Rostam has a third embedder option — an in-process model — alongside BM25-only
and `ROSTAM_EMBED_ENDPOINT`. No cloud API, no network call per embed. It is
**pure Go** ([rembed](https://github.com/rostamlabs/rembed)): no cgo, no ONNX
Runtime, no shared library to install, and no build tag — it is compiled into
every binary and image. Nothing to build — just set the model:

```sh
ROSTAM_EMBED_LOCAL=minilm-l6-v2 rostam-server mcp -data ./data
```

### Docker

Local embeddings are in the default image (`cmd/rostam-server/Dockerfile`) —
there is no separate image. Just set the model. Weights download on first use;
mount a named volume at `/models` (and point `REMBED_CACHE` at it) to persist
them across restarts:

```sh
docker run -p 8080:8080 -e ROSTAM_API_KEY=<token> \
  -e ROSTAM_EMBED_LOCAL=minilm-l6-v2 -e REMBED_CACHE=/models -v rostam-models:/models \
  ghcr.io/rostamlabs/rostam:latest
```

Select a model with `ROSTAM_EMBED_LOCAL=<name>`. It is mutually exclusive with
`ROSTAM_EMBED_ENDPOINT` — setting both is a startup error, the same as the
mismatch cases above. `1` or `default` selects the default model,
`minilm-l6-v2`. Run `rostam-server mcp -list-embed-models` to print the full
catalog from the binary you have installed.

The catalog has two tiers: a 384-dim tier (smaller and faster) and a higher-
quality 768-dim "base" tier.

| Name | Dim | License |
|---|--:|---|
| `minilm-l6-v2` (default) | 384 | Apache-2.0 |
| `bge-small-en-v1.5` | 384 | MIT |
| `gte-small` | 384 | MIT |
| `bge-base-en-v1.5` | 768 | MIT |
| `gte-base` | 768 | MIT |
| `all-mpnet-base-v2` | 768 | Apache-2.0 |

Beyond the catalog, `ROSTAM_EMBED_LOCAL` also accepts any Hugging Face
`org/model` id directly (e.g. `BAAI/bge-large-en-v1.5`); it is passed straight
to rembed. rembed reads each model's tokenizer, pooling, and normalization from
the model itself, so the catalog only curates names, dimensions, and licenses.
The embedder id stamped into cache scope keys is `local:<value of
ROSTAM_EMBED_LOCAL>`, so the same model selected by its catalog short name
(`local:minilm-l6-v2`) and by its full Hub id
(`local:sentence-transformers/all-MiniLM-L6-v2`) build separate caches — pick
one form and keep it.

The selected model's weights download from the Hugging Face Hub on first run
into rembed's cache (`REMBED_CACHE`, default the OS user cache dir; the legacy
`ROSTAM_EMBED_MODELS_DIR` is honored when `REMBED_CACHE` is unset). Later starts
reuse the cached files with no network call.

**Known limitation:** every catalog model is used symmetrically — the same
embedding function for what you store and what you search with. Asymmetric
models such as E5, which need distinct query/passage prefixes to perform
well, aren't in the catalog yet; that needs a query-vs-passage role extension
to the embedder interface, which is future work.

**A local embedder isn't limited to memory.** Once configured, it also
satisfies the generic vector-DB tools' embedder requirement, so any
collection — not just `mcp_memory` — can be written and queried by plain
text, with no cloud API: `upsert` auto-embeds `content` when `vector` is
omitted, and `search` embeds `query_text` in `dense`/`hybrid` mode when
`vector` is omitted.

```json
{"tool": "upsert", "arguments": {"collection": "docs", "id": 1, "content": "hello rostam"}}
{"tool": "search", "arguments": {"collection": "docs", "query_text": "hello"}}
```

## Filters

Tools that accept a `filter` argument take
[`vector.Filter`](../vector/filtering.md)'s JSON form directly. Leaf values are a
**tagged union**, not a bare scalar — this is the one detail every filter
example in this page depends on:

```json
{"op": "eq", "field": "lang", "value": {"kind": "string", "str": "en"}}
```

`kind` is one of `string` (`str`), `int` (`int`), `float` (`flt`), `bool`
(`bool`), `strings` (`strs`), `ints` (`ints`), `floats` (`flts`), or `geo`
(`lat`/`lon`) — the field holding the value is named after its kind.
Composite filters nest with `and`/`or`/`not`:

```json
{
  "op": "and",
  "and": [
    {"op": "eq", "field": "lang", "value": {"kind": "string", "str": "en"}},
    {"op": "gte", "field": "year", "value": {"kind": "int", "int": 2020}}
  ]
}
```

A bare `{"lang": "en"}` is **not** a valid filter — it has to go through the
tagged form above.

## Tool reference

Every tool returns `content: [{"type": "text", "text": "<json>"}]` on
success; a tool-level failure sets `isError: true` on the same shape rather
than a protocol-level error, so a bad argument never tears down the session.

### Memory tools (always registered)

| Tool | Args | Behavior |
|---|---|---|
| `remember` | `content` (required), `namespace` (default: the server's default namespace, see `-namespace`), `metadata` (flat JSON object), `key` (optional stable key), `path` (optional repo-relative file link), `symbol` (optional, requires `path`; see [Claude Code plugin](claude-code-plugin.md)) | Embeds (or stubs) `content` and upserts it into the `mcp_memory` collection. With no `key`, re-remembering identical content in the same namespace upserts the same point — dedupe by `(namespace, content)` is free. With `key`, the point id is derived from `(namespace, key)` instead: re-remembering the same key always upserts that one canonical memory regardless of content, so it REPLACES the prior entry rather than accumulating stale snapshots — the right shape for live/in-flight state (a PR's status, what you're mid-task on). A keyed memory keeps its original `created` time across updates; `updated` always moves to now. Returns `{id, namespace}`, plus `key` when one was given. |
| `recall` | `query` (required), `namespace` (default: the server's default namespace, see `-namespace`), `k` (default 5), `filter` (optional, ANDed with the namespace) | BM25-only or hybrid dense+BM25, per the embedder mode. Returns `{hits: [{id, content, score, key, created, updated, metadata, path, symbol, stale}]}` — `created` and `updated` (unix seconds) are present on every hit written by this version; `key` is omitted on a hit with no key (a plain, unkeyed `remember`); `path` and `symbol` appear only on a memory remembered with a file link, and `stale` only when that file has changed (`true`) or is gone (`"file missing"`). |
| `forget` | `ids` (array of ids), `namespace` (default: the server's default namespace, scopes `keys`), `keys` (array of stable keys) | Deletes memories by id and/or by stable key — a key is resolved to the id a keyed `remember` landed on within `namespace`, and `ids`/`keys` may be combined in one call. At least one of `ids`/`keys` is required. A per-id failure doesn't abort the rest of the batch. Returns `{"deleted":[...],"missing":[...],"errors":[...]}` — `errors` is present only when at least one id failed to delete, so a fully successful call returns just `{deleted, missing}`. Same shape as `delete` below. |
| `list_memories` | `namespace` (default: the server's default namespace, see `-namespace`), `limit` (default 50, max 500), `cursor` (from a previous call's `next_cursor`) | Pages through a namespace's memories in id order. Returns `{memories: [{id, content, score, key, created, updated, metadata, path, symbol, stale}], next_cursor}` — same per-memory shape as `recall`'s hits, with `score` left at zero (a scroll has no query to rank against). |
| `list_namespaces` | — | Returns `{namespaces: [...]}`, sorted. Derived from the memories themselves by scanning the collection, so it always agrees with what `recall`/`list_memories` can find — a namespace exists exactly as long as at least one memory carries it, and forgetting the last one makes it disappear with no separate bookkeeping step. |
| `recall_for_file` | `path` (required, repo-relative or absolute inside the project), `namespace` (default: the server's default namespace), `k` (default 5) | Memories linked to one file, newest first, each marked `stale` when the file changed or `"file missing"` when it is gone. Returns `{memories: [...]}` with the same per-memory shape as `recall`'s hits. |

Memory hits (`remember`/`recall`/`list_memories`) never carry a `distance`
field — it has no meaning for BM25-only recall or for a plain listing.

`created`/`updated` are Unix seconds. They exist to let an agent tell a fresh
keyed memory from a stale one at a glance — e.g. "this PR-status memory was
last `updated` 3 days ago" — without parsing `content`.

### Generic DB tools (always registered)

| Tool | Args | Behavior |
|---|---|---|
| `create_collection` | `name` (required), `dim` (required), `metric` (`cosine`\|`l2`\|`dot`, default `cosine`), `full_text` (default `true`), `persistent` (default `true`) | Creates a vector collection. `persistent` collections are stored on disk and survive a clean restart (see [What survives a crash](#what-survives-a-crash)); `persistent: false` makes an in-memory collection that is gone when the process exits. A persistent collection is SQ8-quantized (disk-backed vector storage requires a quantizer): candidates are re-ranked exactly against the full-precision vectors, so scores are unaffected and only recall moves, marginally. |
| `upsert` | `collection` (required), `id` (required), `vector` (optional array), `content` (optional), `metadata` (optional) | Inserts or updates a point. Provide `vector` explicitly, or omit it and provide `content` with an embedder configured to auto-embed. Returns `{id}`. |
| `search` | `collection` (required), `mode` (`text`\|`dense`\|`hybrid`; default `text` with no embedder, `hybrid` with one), `query_text`, `vector`, `k` (default 10), `filter` | Text search, dense nearest-neighbor, or dense+BM25 hybrid fusion. In `dense`/`hybrid` mode, an omitted `vector` is derived by embedding `query_text` (requires an embedder). Returns `{hits: [{id, content, score, distance, metadata}]}`. |
| `get` | `collection` (required), `ids` (required), `with_vector` (default `false`) | Fetches points by id. Returns `{points: [{id, content, metadata, vector?}], missing: [...]}`. |

!!! warning "Point ids above 2^53 from a JavaScript client"

    Rostam's point ids are full-width `uint64`, but a JSON number in a
    JavaScript MCP client is an IEEE-754 double — exact only up to
    2^53−1 (9007199254740991). An id above that is rounded the moment the
    client parses it, so echoing an id from a `search` result back into
    `get`, `upsert`, or `delete` would silently name a *different* point.

    For those collections, pass the id as a **decimal string** instead:
    `{"collection": "docs", "ids": ["18446744073709551000"]}`. Every id
    argument accepts either form, and the round trip is safe in both
    directions: a generic tool's result (`upsert`'s `id`, `get`'s `points`
    and `missing`, `delete`'s `deleted` and `missing`, `search`'s hit `id`)
    gives back a plain number for an id within the safe range and a decimal
    string above it, so an id read out of one response can always be fed
    back into another call without rounding.

    Memory ids (`remember`/`recall`/`list_memories` → `forget`) are generated
    inside the safe range on purpose, so this never applies to them.

**Every** generic tool — `create_collection`, `upsert`, `search`, `get`,
`delete`, and `delete_by_filter` — refuses the `mcp_memory` collection. The
memory tools above are its only interface.

For the writes, the reason is corruption: `mcp_memory`'s schema, reserved
metadata fields, and embedder-identity bootstrap belong to
`remember`/`recall`/`forget`. For the reads, the reason is namespace
isolation: `recall` and `list_memories` always scope their query to one
namespace and strip the reserved fields out of what they return, while
`search` and `get` do neither — so allowing them here would let any client
read every namespace's memories at once, and see the internal field naming
which namespace each one came from.

### Destructive tools (only with `-enable-destructive`)

| Tool | Args | Behavior |
|---|---|---|
| `delete` | `collection` (required), `ids` (required) | Deletes points by id; a per-id failure doesn't abort the rest of the batch. Returns `{"deleted":[...],"missing":[...],"errors":[...]}` — `errors` is present only when at least one id failed to delete. |
| `delete_by_filter` | `collection` (required), `filter` (required) | Deletes every point matching `filter`. A match-all (empty/zero) filter is refused even with the gate open — `-enable-destructive` authorizes targeted deletes, not a blanket wipe. |

Without `-enable-destructive`, `delete` and `delete_by_filter` are absent from
`tools/list` entirely. Memory's `forget` tool is always available regardless
of this flag — it's scoped to `mcp_memory` only.

## Remote mode

By default `rostam-server mcp` embeds the engine and owns a local data
directory. Point it at an already-running server instead with `-connect`:

```sh
rostam-server mcp -connect 127.0.0.1:7000 -auth-token "$ROSTAM_AUTH_TOKEN"
```

`-connect` speaks the binary TCP protocol via `rostam.NewClient` — the same
client the Go smart client uses — and every tool works identically against
either backend; only the backend selection differs. Authentication and TLS
follow the same conventions as the rest of the client tooling:

- **Auth**: `-auth-token`, or `ROSTAM_AUTH_TOKEN` (preferred — a flag-passed
  secret is visible via `/proc` and shell history).
- **TLS**: set any of `-tls-ca`, `-tls-cert`, `-tls-key`, `-tls-server-name`
  to enable it; plaintext stays the default when none are set. `-tls-cert`
  and `-tls-key` together enable mTLS.

See [Security](security.md) for how the target server's own auth and TLS are
configured.

### Durability over `-connect`

The `mcp` process stores nothing itself in remote mode. For memory to survive
the server being killed, start a single-node server with a data directory and
list the memory collection in
[`-durable-collections`](running.md):

```sh
rostam-server -tcp 127.0.0.1:7000 -http "" -data /var/lib/rostam -durable-collections mcp_memory
```

`-http ""` turns the REST API off. It is on by default, on `:8080` on every
interface, and the server refuses to start with that unless authentication is
configured or `-insecure` is passed.

The flag also converts an `mcp_memory` collection that already exists on that
server, at startup (see [Upgrading an existing store](#upgrading-an-existing-store)),
and the notes on a [damaged log](#a-damaged-log) apply to the server. Without
the flag, a new `mcp_memory` collection on a single-node server has no
write-ahead log, and a forced stop can lose memories. Without `-data` nothing is kept across a
restart. With `-cluster` the flag is ignored: Raft is the durability authority
there.

### Server version

Creating a new memory store over `-connect`, and rebuilding a missing embedder
record, need a server at **v0.5.0 or later**. No v0.4.x server has the
operation this needs, so the call fails there with any key, whether or not
authorization is on. An existing store that has its record is not affected.
