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
- For in-flight state that changes, pass a stable `key` so each update replaces
  the last one.
- Never store secrets or credentials.

## Reading

- Before a nontrivial task, `recall` with the task's keywords instead of
  re-reading large files.
- To see everything attached to one file, call `recall_for_file` with its `path`.
  The automatic injection is capped at a few memories; this is the way to get
  the rest.
- A memory marked `[stale]` was linked to a file that has changed since. One
  marked `[stale: file missing]` points at a file that no longer exists. Treat
  both as leads to verify, then re-`remember` the corrected fact (same `key`, or
  `forget` the old one).
