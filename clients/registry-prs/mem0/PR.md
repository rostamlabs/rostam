# PR Title

Add Rostam as a vector store provider

# PR Description

## Description

This adds [Rostam](https://github.com/rostamlabs/rostam), an open-source
vector database, as a vector store provider in mem0. After this change,
users can select Rostam the same way they select any other provider:

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "rostam",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "url": "http://localhost:8080",
        },
    },
}

m = Memory.from_config(config)
```

Rostam runs as a standalone HTTP service (self-hosted via Docker, or a remote
deployment), and this provider talks to it through the `rostam-client`
package on PyPI.

### What's added

- `mem0/vector_stores/rostam.py` — `RostamDB`, implementing `VectorStoreBase`
  (create/insert/search/delete/update/get/list/reset/col_info; `list_cols`
  raises `NotImplementedError` since Rostam's client has no
  list-collections endpoint, matching the documented contract for the
  method). Mem0 memory ids are opaque uuid4 strings, but Rostam point ids
  are unsigned 64-bit integers, so string ids are mapped to a stable point
  id (numeric strings verbatim, everything else via a BLAKE2b-64 hash) and
  the original string is kept in a reserved `_mem0_id` metadata key so it
  round-trips back out of `search`/`get`/`list`.
- `mem0/configs/vector_stores/rostam.py` — `RostamConfig` (pydantic), with
  the same `collection_name`/`embedding_model_dims` fields every other
  provider uses, plus `url`, `api_key`, `metric`.
- `mem0/utils/factory.py` — registers `"rostam"` in
  `VectorStoreFactory.provider_to_class`.
- `mem0/vector_stores/configs.py` — registers `"rostam"` in
  `VectorStoreConfig._provider_configs`.
- `pyproject.toml` — adds `rostam-client` to the `vector-stores` optional
  dependency group (so `pip install mem0ai[vector-stores]` pulls it in,
  matching how every other optional provider dependency is declared).
- `docs/components/vectordbs/dbs/rostam.mdx` + a nav entry in
  `docs/docs.json` + a card in `docs/components/vectordbs/overview.mdx` —
  documents the provider the same way Turbopuffer/Upstash/etc. are
  documented.
- `tests/vector_stores/test_rostam.py` — unit tests mirroring
  `tests/vector_stores/test_turbopuffer.py`'s structure: init (incl. the
  create-collection-is-idempotent behavior), id mapping, insert, search
  (incl. filter translation and score conversion), delete, update (both the
  vector-and-payload and the read-then-fill-in-the-missing-half paths),
  get, list_cols, delete_col, col_info, list, reset, config validation, and
  factory/config registration.

### Verified locally (not part of this diff)

Since I don't have push access to open this PR directly, I validated the
change by installing this checkout of mem0 together with `rostam-client`
in a fresh virtualenv and running:

- `pytest tests/vector_stores/test_rostam.py -q` → 35 passed
- `ruff check` / `ruff format --check` on all new/changed Python files →
  clean
- An end-to-end smoke test that resolves `"rostam"` through
  `VectorStoreConfig` → `VectorStoreFactory.create(...)` with a mocked
  Rostam client, confirming the exact `Memory.from_config({"vector_store":
  {"provider": "rostam", ...}})` path wires up correctly.

## Test plan

- [ ] `pip install -e ".[vector-stores,test]"` (adds `rostam-client`)
- [ ] `pytest tests/vector_stores/test_rostam.py -q`
- [ ] `ruff check mem0/vector_stores/rostam.py mem0/configs/vector_stores/rostam.py tests/vector_stores/test_rostam.py`
- [ ] Optional live check: run a local Rostam server (`docker run -p 8080:8080 rostamlabs/rostam`) and exercise `Memory.from_config(...)` end to end.

# Submit Instructions

The files in `files/` in this directory are already drafted and locally
verified (see "Verified locally" above) — they mirror the exact repo-relative
paths inside `mem0ai/mem0`. To open the real PR:

1. **Fork and clone.**
   ```bash
   gh repo fork mem0ai/mem0 --clone
   cd mem0
   git checkout -b feat/rostam-vector-store
   ```
   (Or fork via the GitHub UI and `git clone` your fork, then `git checkout -b feat/rostam-vector-store`.)

2. **Copy the drafted files in**, overwriting/creating at the matching paths
   (run from this scratch dir, adjusting `MEM0_CHECKOUT` to your clone):
   ```bash
   MEM0_CHECKOUT=/path/to/your/mem0
   cp -v files/mem0/vector_stores/rostam.py            "$MEM0_CHECKOUT/mem0/vector_stores/rostam.py"
   cp -v files/mem0/configs/vector_stores/rostam.py     "$MEM0_CHECKOUT/mem0/configs/vector_stores/rostam.py"
   cp -v files/tests/vector_stores/test_rostam.py       "$MEM0_CHECKOUT/tests/vector_stores/test_rostam.py"
   cp -v files/docs/components/vectordbs/dbs/rostam.mdx "$MEM0_CHECKOUT/docs/components/vectordbs/dbs/rostam.mdx"
   ```
   The remaining four files (`mem0/utils/factory.py`, `mem0/vector_stores/configs.py`,
   `pyproject.toml`, `docs/docs.json`, `docs/components/vectordbs/overview.mdx`) are
   **modified**, not new — since upstream may have moved on since this draft was
   made, don't blind-overwrite them. Instead open each drafted copy next to the
   current upstream file and apply the same small diff by hand (each is a single
   one-line/one-entry addition — a new `"rostam": ...` map entry, a new
   `rostam-client` dependency line, a new nav string, or a new `<Card>` line — so a
   manual merge takes under a minute per file even if line numbers have shifted).
   `git diff` against this `files/` copy after merging to confirm only the
   intended lines changed.

3. **Install and test.**
   ```bash
   cd "$MEM0_CHECKOUT"
   pip install -e ".[vector-stores,test]"
   pytest tests/vector_stores/test_rostam.py -q
   ruff check mem0/vector_stores/rostam.py mem0/configs/vector_stores/rostam.py tests/vector_stores/test_rostam.py
   ruff format --check mem0/vector_stores/rostam.py mem0/configs/vector_stores/rostam.py tests/vector_stores/test_rostam.py
   ```

4. **Commit and push.**
   ```bash
   git add mem0/vector_stores/rostam.py mem0/configs/vector_stores/rostam.py \
           mem0/utils/factory.py mem0/vector_stores/configs.py pyproject.toml \
           tests/vector_stores/test_rostam.py \
           docs/components/vectordbs/dbs/rostam.mdx docs/docs.json \
           docs/components/vectordbs/overview.mdx
   git commit -m "Add Rostam as a vector store provider"
   git push -u origin feat/rostam-vector-store
   ```

5. **Open the PR** using the title and description above:
   ```bash
   gh pr create --repo mem0ai/mem0 \
     --title "Add Rostam as a vector store provider" \
     --body-file PR.md
   ```
   (Trim this file down to just the "## Description" through "## Test plan"
   sections first, since `--body-file` will otherwise include this Submit
   Instructions section and the Title header too.)

6. Check the CI checks that run on the PR (lint + the provider's own test
   file) and address any upstream drift they surface.

