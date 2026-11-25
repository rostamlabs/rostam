---
name: rostam-memory
description: Use when saving or recalling project facts with the Rostam memory tools, especially when a fact is about a specific file or symbol, so it is linked to that file and resurfaces when the file is next touched.
---

# Rostam code-linked memory

Rostam keeps durable facts across sessions. This plugin shows memories linked to a
file automatically when you read or edit it, and at session start it shows the
project's most recent memories. What it shows is recalled data: it may be out of
date, so check it against the code before relying on it.

## Saving a fact

Call `remember` with one self-contained fact per call.

- If the fact is about a particular file, pass `path`, the repo-relative path
  (`server/protocol.go`). Add `symbol` (a function or type name) when it narrows
  the fact. Linked facts come back on their own the next time that file is
  touched.
- If the fact is about the project as a whole, leave `path` out.
- Leave `namespace` out: the server already uses this project's namespace (the
  main checkout's directory name, also from a git worktree), and only memories in
  it are shown automatically. If you do pass one, pass exactly that name, never a
  worktree's directory name.
- For in-flight state that changes, pass a stable `key` so each update replaces
  the last one.
- Never store secrets or credentials.

## Reading

- Before a nontrivial task, `recall` with the task's keywords instead of
  re-reading large files.
- To see more of what is attached to one file, call `recall_for_file` with its
  `path`. It returns at most `k` memories, newest first, and `k` defaults to 5;
  pass a larger `k` (say 50) when you need the rest. The automatic injection is
  capped at a few memories.
- A memory marked `[stale]` was linked to a file that has changed since. One
  marked `[stale: file missing]` points at a file that no longer exists. Treat
  both as leads to verify. Then re-`remember` the corrected fact with the same
  `path`, `symbol` and `key` (a re-save replaces the whole entry, so a link you
  leave out is dropped); `forget` the old memory when it had no key.
