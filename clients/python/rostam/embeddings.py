"""Embedding helpers — work in text, not vectors.

The core ``Rostam`` client is deliberately vector-only (and dependency-free): you
hand it float32 vectors. This module adds the missing piece for RAG ergonomics —
turning text into vectors — without pulling a model into the core:

- ``FunctionEmbedder`` wraps any callable (e.g. a ``sentence-transformers`` model's
  ``.encode``), so local/in-process models work with no extra Rostam dependency.
- ``OpenAIEmbedder`` calls any OpenAI-compatible ``/embeddings`` endpoint over the
  standard library (no ``openai`` package needed) — OpenAI, Azure OpenAI, and
  local servers such as Ollama, LM Studio, or text-embeddings-inference.
- ``TextStore`` wraps a ``Rostam`` client + an embedder and exposes a text-first
  surface: ``add(texts=…)``, ``search("a query")``, ``search_groups(…)``.

In-server embedding (running the model inside Rostam via the WASM UDF subsystem)
is intentionally not done here: production embedding models don't fit a WASM
sandbox well, and keeping generation client-side keeps model dependencies out of
the engine's hot path.

    from rostam import Rostam, TextStore, OpenAIEmbedder

    store = TextStore(Rostam("http://localhost:8080"), "docs", OpenAIEmbedder())
    store.create_collection()                       # dim inferred from the embedder
    store.add(["first chunk", "second chunk"], metadatas=[{"doc_id": 1}, {"doc_id": 1}])
    hits = store.search("a question", k=4, filter={"op": "eq", "field": "doc_id", ...})
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from ._ids import to_uint64
from ._types import Document, Group, RostamError
from .rostam import Rostam


class Embedder(Protocol):
    """Anything that turns text into vectors.

    Compatible with LangChain's ``Embeddings`` interface, so the same object can
    feed both ``TextStore`` and ``RostamVectorStore``.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class FunctionEmbedder:
    """Wrap a callable as an Embedder.

    ``fn`` maps a list of texts to a list of vectors (e.g. a sentence-transformers
    model's ``.encode`` — pass ``lambda ts: model.encode(ts).tolist()``).
    ``query_fn`` optionally embeds a single query differently; by default a query
    is embedded with ``fn``.
    """

    def __init__(
        self,
        fn: Callable[[List[str]], Sequence[Sequence[float]]],
        query_fn: Optional[Callable[[str], Sequence[float]]] = None,
    ):
        self._fn = fn
        self._query_fn = query_fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(v) for v in self._fn(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        if self._query_fn is not None:
            return list(self._query_fn(text))
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """Embedder backed by an OpenAI-compatible ``/embeddings`` endpoint.

    Uses only the standard library (no ``openai`` package). Works with OpenAI,
    Azure OpenAI, and local OpenAI-compatible servers (Ollama, LM Studio,
    text-embeddings-inference, …) by setting ``base_url``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.timeout = timeout

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        body: Dict[str, Any] = {"model": self.model, "input": inputs}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        req = urllib.request.Request(
            self.base_url + "/embeddings", data=json.dumps(body).encode("utf-8"), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            msg = str(e)
            try:
                msg = json.loads(raw).get("error", {}).get("message", msg)
            except Exception:
                pass
            raise RostamError(f"embedding request failed: {msg}", status=e.code) from None
        except urllib.error.URLError as e:
            raise RostamError(f"embedding transport error: {e.reason}", status=0) from None
        # Sort by index so the order matches the inputs regardless of server order.
        data = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class TextStore:
    """A text-first wrapper over a Rostam collection: it embeds text for you.

    For users not on LangChain — the native ``upsert(text=…)`` ergonomics. For
    LangChain users, see ``rostam.langchain.RostamVectorStore``.
    """

    def __init__(self, client: Rostam, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def create_collection(
        self, *, dim: Optional[int] = None, metric: str = "cosine", **cfg: Any
    ) -> None:
        """Create the backing collection. If dim is omitted it is inferred by
        embedding a one-token probe — convenient when you don't know the model's
        output dimensionality offhand."""
        if dim is None:
            dim = len(self.embedder.embed_query("dimension probe"))
        self.client.create_collection(self.collection, dim, metric=metric, **cfg)

    def add(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        ttl_ms: int = 0,
    ) -> List[str]:
        """Embed and upsert texts. Returns the external ids (generated stably from
        the text when not supplied)."""
        texts = list(texts)
        if not texts:
            return []
        vectors = self.embedder.embed_documents(texts)
        if ids is None:
            ids = [_text_id(t) for t in texts]
        else:
            ids = list(ids)
        metas = list(metadatas) if metadatas is not None else [None] * len(texts)
        for text, vec, meta, ext in zip(texts, vectors, metas, ids):
            self.client.upsert(self.collection, to_uint64(ext), vec, content=text, metadata=meta, ttl_ms=ttl_ms)
        return ids

    def search(self, text: str, k: int = 4, *, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Embed the query text and return the k nearest documents (content + metadata)."""
        return self.client.search_docs(self.collection, self.embedder.embed_query(text), k, filter=filter)

    def search_groups(
        self,
        text: str,
        k: int,
        group_by: str,
        *,
        group_size: int = 1,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Group]:
        """Embed the query text and return the top-k distinct documents (group-by)."""
        return self.client.search_groups(
            self.collection, self.embedder.embed_query(text), k, group_by,
            group_size=group_size, filter=filter,
        )

    def delete(self, ids: Sequence[str]) -> int:
        """Delete points by external id; returns the number that existed."""
        n = 0
        for ext in ids:
            if self.client.delete(self.collection, to_uint64(ext)):
                n += 1
        return n


def _text_id(text: str) -> str:
    import hashlib

    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
