"""Rostam Python client — dependency-free SDK.

One entry point, ``Rostam(target)``, whose transport is chosen from the
target: ``http(s)://host:8080`` speaks the REST API, ``tcp://host:7000`` (or a
bare ``host:7000``) speaks the native binary TCP protocol. Vector ops are flat
(``r.search``, ``r.upsert``, ``r.hybrid_text``, ...); key-value ops live under
``r.kv.*`` (TCP only). Uses only the standard library. The optional LangChain /
LlamaIndex / Haystack adapters live in ``rostam.langchain`` / ``rostam.llamaindex``
/ ``rostam.haystack`` and require their respective extras.
"""

from . import filters
from ._http import MultiResult
from ._types import (
    Document,
    Group,
    GroupResults,
    Point,
    RostamError,
    ScrollPage,
    SearchResult,
    SearchResults,
    TransportError,
)
from .embeddings import Embedder, FunctionEmbedder, OpenAIEmbedder, TextStore
from .rostam import Rostam

__all__ = [
    "Rostam",
    "RostamError",
    "TransportError",
    "SearchResult",
    "SearchResults",
    "GroupResults",
    "Document",
    "Point",
    "Group",
    "ScrollPage",
    "MultiResult",
    "filters",
    "Embedder",
    "FunctionEmbedder",
    "OpenAIEmbedder",
    "TextStore",
]

# Keep in step with [project] version in pyproject.toml. test_version.py fails
# when they drift: this constant sat at 0.1.0 through the 0.1.1 release, so
# anything asking rostam.__version__ was told the wrong release, and nothing
# noticed. It is a literal rather than an importlib.metadata lookup so that
# importing the package stays free of a dist-info read.
__version__ = "0.2.0"
