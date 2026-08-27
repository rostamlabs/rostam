# LangChain + LlamaIndex registry listings — publish + PR runbook

Both are blocked on ONE thing: the packages must be installable from PyPI. The
partner packages already exist on `origin/main` and both depend on
`rostam-client>=0.2` — but `rostam-client` is currently `0.1.2`. So the chain is:

## Step 0 — Publish rostam-client 0.2.0 (unblocks everything)
The unified-`Rostam` client (removed `RostamClient`) is a breaking change, so a
0.2.0 bump is the right call, and the partner packages already require `>=0.2`.

```bash
cd clients/python
# bump version 0.1.2 -> 0.2.0 in pyproject.toml AND rostam/__init__.py __version__
# (test_version.py enforces they match)
python -m build            # produces dist/rostam_client-0.2.0-*.whl + .tar.gz
python -m twine upload dist/rostam_client-0.2.0*   # needs your PyPI token
```
NOTE: merge the mem0/semantic_router/crewai/dspy adapters branch first if you
want 0.2.0 to ship all seven adapters; otherwise they land in 0.2.1.

## Step 1 — Publish the two partner packages
```bash
cd clients/langchain-rostam        && python -m build && python -m twine upload dist/*
cd clients/llama-index-vector-stores-rostam && python -m build && python -m twine upload dist/*
```

## Step 2a — LangChain listing (repo: langchain-ai/langchain)
Two edits in a single PR to your fork:
1. **`libs/packages.yml`** — append an entry (match the existing shape):
   ```yaml
   - name: langchain-rostam
     repo: rostamlabs/rostam
     path: clients/langchain-rostam
     provider_page: rostam           # optional; only if you add the provider doc
   ```
2. **Docs notebook** `docs/docs/integrations/vectorstores/rostam.ipynb` — a
   short notebook: install (`pip install langchain-rostam`), construct
   `RostamVectorStore` with a `Rostam(...)` client + an `Embeddings`, and show
   `add_texts` / `similarity_search`. Copy the structure of an existing small
   vectorstore notebook (e.g. `qdrant.ipynb`) and swap in Rostam's API.
   (Optional but recommended: `docs/docs/integrations/providers/rostam.mdx`.)
PR title: `docs: add Rostam vector store integration`

## Step 2b — LlamaIndex listing (repo: run-llama/llama_index)
LlamaIndex integrations live in the monorepo under
`llama-index-integrations/vector_stores/llama-index-vector-stores-rostam/`.
Two viable paths — pick one:
- **A (contribute the package into the monorepo):** copy the partner package into
  that path, matching a sibling (e.g. `llama-index-vector-stores-qdrant/`)
  layout + `BUILD`/`pyproject`, add a docs example under
  `docs/docs/examples/vector_stores/`, open a PR. This is how most vector stores
  are listed and is the durable route.
- **B (registry only):** if the package is on PyPI, it can also be surfaced via
  their integration registry — but path A is what gets it into the docs sidebar.
PR title: `feat: add Rostam vector store integration`

## Human-gated summary
- PyPI publish of 3 packages → **you** (needs your PyPI token).
- Version bump 0.1.2 → 0.2.0 → I can do this in the client repo on request.
- The LangChain/LlamaIndex PRs are against THEIR repos → your fork + `gh` auth.
I can draft the notebook + packages.yml entry + the LlamaIndex package layout in
full once you confirm you want path A vs B for LlamaIndex and the 0.2.0 bump.
