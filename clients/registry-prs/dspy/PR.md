# PR title

Add RostamRM retriever integration

# PR description

## What

Adds `RostamRM`, a `dspy.Module`-based retriever integration for [Rostam](https://github.com/rostamlabs/rostam), an open-source vector database. It follows the same shape as the existing `WeaviateRM` / `DatabricksRM` vendor integrations in `dspy/retrievers/`, and returns `dspy.Prediction(passages=...)`, matching what `dspy.retrievers.Embeddings` and the current RAG tutorial use.

Rostam's search API takes a query *vector*, not a query string, so `RostamRM` is constructed with an `embedder: Callable[[str], list[float]]` used to embed both indexed documents and incoming queries — the same shape `dspy.retrievers.Embeddings` already expects from its `embedder` argument.

Files added/changed:

- `dspy/retrievers/rostam_rm.py` — the `RostamRM` class.
- `tests/retrievers/test_rostam_rm.py` — unit tests (mocked `rostam.Rostam` client, no live server required; mirrors the mocking style of `tests/retrievers/test_colbertv2.py`).
- `pyproject.toml` — adds an optional `rostam` extra (`rostam = ["rostam-client>=0.2.0"]`), matching the existing `weaviate` extra.

## Usage

```python
from rostam import Rostam
from dspy.retrievers.rostam_rm import RostamRM

client = Rostam("http://localhost:8080")
retriever = RostamRM("my_collection", rostam_client=client, embedder=embed_fn, k=5)
retriever.index(["doc one text", "doc two text"])

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question):
        passages = self.retrieve(question).passages
        return self.answer(question=question, context=passages)
```

## Design notes / conventions followed

- **Location & naming**: `dspy/retrieve/<vendor>_rm.py` (the pre-3.x layout with `qdrant_rm.py`, `chromadb_rm.py`, etc.) no longer exists in the current tree — third-party vendor retrievers now live directly in `dspy/retrievers/` (currently `weaviate_rm.py`, `databricks_rm.py`). `rostam_rm.py` / `RostamRM` follows that current location and the `<vendor>_rm.py` / `<Vendor>RM` naming convention of its two siblings.
- **Return type**: `WeaviateRM.forward` and `DatabricksRM.forward` predate the current recommended pattern and return, respectively, a raw list of `dotdict(long_text=...)` and a `Prediction(docs=..., doc_ids=...)` (no `passages` key) — both meant to be used through the legacy `dspy.settings.configure(rm=...)` + `dspy.Retrieve` machinery. `RostamRM` instead follows `dspy.retrievers.Embeddings`, the pattern the current DSPy RAG tutorial teaches: a plain module you call directly, whose `forward` returns `dspy.Prediction(passages=...)`. It still works with `dspy.settings.configure(rm=...)` + `dspy.Retrieve()` too, since `dspy.Retrieve.forward` degrades gracefully — but callers are not required to go through it.
- **Not added to `dspy/retrievers/__init__.py`**: neither `WeaviateRM` nor `DatabricksRM` is exported there, presumably to avoid a hard import-time dependency on their (optional) client libraries. `RostamRM` follows that precedent — users import it explicitly: `from dspy.retrievers.rostam_rm import RostamRM`.
- **No new docs page**: `docs/docs/api/tools/` documents `ColBERTv2`, `Embeddings`, and `PythonInterpreter` via `mkdocstrings` auto-ref, but neither `WeaviateRM` nor `DatabricksRM` has a page there or a `mkdocs.yml` nav entry, despite both being fully documented in their own docstrings. `RostamRM`'s docstring follows the same Google-style/Examples format as `WeaviateRM`'s, so a docs page can be added the same way (`::: dspy.retrievers.rostam_rm.RostamRM`) if maintainers want one — I didn't add one unprompted, to stay consistent with the existing (undocumented) siblings.
- **Optional dependency via `pyproject.toml` extra**: `weaviate-client` is an extra; `databricks-sdk` is not (that RM falls back to `requests`, already a base dependency). Rostam has no such fallback, so it gets an extra the same way Weaviate does.
- **Tests**: `tests/retrievers/` currently only has tests for `test_colbertv2.py` and `test_embeddings.py` — neither `WeaviateRM` nor `DatabricksRM` has a test file, presumably because they need a live/mocked vendor client and weren't prioritized. `test_rostam_rm.py` is included anyway (mocking `rostam.Rostam`, no network/server needed) since it was cheap to write and keeps `RostamRM`'s behavior pinned; feel free to drop it if maintainers prefer parity with the untested siblings.
- **Lint/format**: verified against the repo's own `ruff` config (`ruff check` and `ruff format --check` both pass on the new files).

## Verification performed before opening this PR

- `pip install -e .` (dspy) + `pip install -e <rostam-client checkout>` in a clean venv, then `pytest tests/retrievers/test_rostam_rm.py -v` — **7 passed**.
- `ruff check dspy/retrievers/rostam_rm.py tests/retrievers/test_rostam_rm.py` — clean.
- `ruff format --check dspy/retrievers/rostam_rm.py tests/retrievers/test_rostam_rm.py` — clean.

## Checklist

- [x] New retriever class + tests added
- [x] Tests pass locally
- [x] Lint/format pass locally
- [ ] CI green (will confirm once opened)

---

# Submit steps

These files are already drafted and verified against a real clone of `stanfordnlp/dspy` plus a real install of the `rostam-client` PyPI package — this is a mechanical "copy files in, push, open PR" job, no further design decisions needed unless a maintainer asks for changes in review.

1. **Fork** `stanfordnlp/dspy` on GitHub (or use an existing fork), then clone your fork and add upstream:
   ```bash
   git clone git@github.com:<your-user>/dspy.git
   cd dspy
   git remote add upstream https://github.com/stanfordnlp/dspy.git
   git fetch upstream
   git checkout -b add-rostam-retriever upstream/main
   ```

2. **Copy the drafted files in** (paths below are relative to this `PR.md`'s directory, i.e. `files/`):
   ```bash
   FILES_DIR=/tmp/claude-1000/-home-vahid-projects-rostam/b48579fe-9b92-48d4-b4ad-2232f0db0084/scratchpad/registry-prs/dspy/files
   cp "$FILES_DIR/dspy/retrievers/rostam_rm.py" dspy/retrievers/rostam_rm.py
   cp "$FILES_DIR/tests/retrievers/test_rostam_rm.py" tests/retrievers/test_rostam_rm.py
   git apply "$FILES_DIR/pyproject.toml.patch"
   ```

3. **Install and run tests/lint** (in a virtualenv):
   ```bash
   pip install -e ".[dev]"
   pip install rostam-client
   pytest tests/retrievers/test_rostam_rm.py -v
   ruff check dspy/retrievers/rostam_rm.py tests/retrievers/test_rostam_rm.py
   ruff format --check dspy/retrievers/rostam_rm.py tests/retrievers/test_rostam_rm.py
   ```
   All of the above passed when drafting this PR; re-confirm against upstream `main` at the time you submit, since it may have moved.

4. **Commit**:
   ```bash
   git add dspy/retrievers/rostam_rm.py tests/retrievers/test_rostam_rm.py pyproject.toml
   git commit -m "Add RostamRM retriever integration"
   git push -u origin add-rostam-retriever
   ```

5. **Open the PR** against `stanfordnlp/dspy:main`:
   ```bash
   gh pr create \
     --repo stanfordnlp/dspy \
     --base main \
     --head <your-user>:add-rostam-retriever \
     --title "Add RostamRM retriever integration" \
     --body-file "$FILES_DIR/../PR.md"
   ```
   (Trim `PR.md` down to the "What" / "Usage" / "Design notes" / "Checklist" sections for the actual PR body — the "Submit steps" section below this line is for you, not for the PR description.)

6. Respond to review feedback. Likely points a maintainer may raise, with a ready answer for each:
   - *"Why not export from `__init__.py`?"* → Matches `WeaviateRM`/`DatabricksRM`, which also aren't exported there.
   - *"Why no docs page?"* → Matches the same two siblings; happy to add one via `mkdocstrings` if wanted.
   - *"Why does this depend on a whole vector DB rather than being a thin wrapper?"* → `rostam-client` is a pure-stdlib, dependency-free client (no transitive deps beyond the optional embedder), same posture as `weaviate-client`.
