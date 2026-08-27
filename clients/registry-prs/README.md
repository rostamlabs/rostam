# Registry PR drafts — get Rostam listed in framework integration registries

Ready-to-submit PR drafts for **external** framework repos, so Rostam shows up in
their integration registries/docs (the passive-discovery channel). These are NOT
part of the Rostam build — they are submission material you open from your own
fork of each upstream repo.

Each subfolder has:
- `PR.md` — the PR title, description, and step-by-step submit instructions.
- `files/` — the files to add, mirroring the **target repo's** paths.

## mem0/ — register Rostam as a Mem0 vector-store provider
Target: `github.com/mem0ai/mem0`. Adds `mem0/vector_stores/rostam.py` (`RostamDB`),
a config class, two one-line registry wires (`utils/factory.py`,
`vector_stores/configs.py`), a `vector-stores` extra, docs, and a test. Verified
against mem0ai 2.0.19 (35 tests passed). Lets users do
`Memory.from_config({"vector_store": {"provider": "rostam", ...}})`.

## dspy/ — add a Rostam retriever to DSPy
Target: `github.com/stanfordnlp/dspy`. Adds `dspy/retrievers/rostam_rm.py`
(`RostamRM`), a test, and a `rostam` extra patch. Verified against dspy 3.3.1
(7 tests passed). Follows the current `WeaviateRM`/`DatabricksRM` convention.

## langchain-llamaindex-runbook.md — LangChain + LlamaIndex listings
Not a PR draft — a runbook. Both are unblocked now that `rostam-client` is on
PyPI; they need the partner packages (`langchain-rostam`,
`llama-index-vector-stores-rostam`) published, then a PR into each framework's
own registry/docs. Full commands inside.

## ⚠️ Before submitting — reconcile with rostam-client 0.3.0
These were drafted against the adapters as of the 0.3.0 release. Two notes:
- **dspy `RostamRM` wraps `rostam-client` directly**, so it inherits the 0.3.0
  fixes (incl. the CodeRabbit round) automatically — just pin `rostam-client>=0.3.0`.
- **mem0's `RostamDB` reimplements the adapter logic locally** (per mem0's own
  convention that every provider is self-contained). Diff it against
  `rostam/mem0.py` at 0.3.0 and carry over any fixes (e.g. the metric-aware
  `_score` conversion) before opening the PR.

Submit each from a fork of the respective upstream repo — see the per-folder `PR.md`.
