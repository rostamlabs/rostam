"""LangChain ``VectorStore`` adapter backed by a Rostam collection.

Requires ``langchain-core`` (install the ``langchain`` extra:
``pip install rostam-client[langchain]``). Embeddings are produced by the
LangChain ``Embeddings`` object you pass in; Rostam stores the vector, the chunk
text (as content), and metadata, and serves similarity search / group-by-document
retrieval.

    from langchain_openai import OpenAIEmbeddings
    from rostam import Rostam
    from rostam.langchain import RostamVectorStore

    store = RostamVectorStore.from_texts(
        texts, OpenAIEmbeddings(),
        client=Rostam("http://localhost:8080"),
        collection="docs",
    )
    docs = store.similarity_search("a query", k=4, filter={"doc_id": 7})
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from . import filters as f
from ._ids import to_uint64
from ._types import Document, RostamError
from .rostam import Rostam

# Rostam metrics whose distance is "smaller = closer", so a 0..1 relevance score
# is 1/(1+distance). Cosine/L2 are distances; dot is a negated similarity.
_DISTANCE_METRICS = {"cosine", "l2", "euclidean"}


# _to_id maps a LangChain string id onto a Rostam uint64 (numeric verbatim, else
# a stable BLAKE2b hash). Shared with the text-store helper via rostam._ids.
_to_id = to_uint64


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _mmr_select(query: List[float], cands: List[Tuple[int, List[float]]], k: int, lambda_mult: float) -> List[int]:
    """Maximal Marginal Relevance over candidate (id, vector) pairs. Returns the
    selected ids in MMR order. cands with empty vectors are ignored."""
    cands = [(i, v) for i, v in cands if v]
    selected: List[int] = []
    sel_vecs: List[List[float]] = []
    while cands and len(selected) < k:
        best_i, best_score = 0, None
        for idx, (cid, vec) in enumerate(cands):
            relevance = _cosine(query, vec)
            diversity = max((_cosine(vec, sv) for sv in sel_vecs), default=0.0)
            score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            if best_score is None or score > best_score:
                best_i, best_score = idx, score
        cid, vec = cands.pop(best_i)
        selected.append(cid)
        sel_vecs.append(vec)
    return selected


def _build_filter(filter: Optional[Any]) -> Optional[dict]:
    """Accept a native Rostam filter (has 'op') or a simple {field: value} map
    (built into an AND of equalities)."""
    if not filter:
        return None
    if isinstance(filter, dict) and "op" in filter:
        return filter
    if isinstance(filter, dict):
        clauses = [f.eq(k, v) for k, v in filter.items()]
        return clauses[0] if len(clauses) == 1 else f.and_(*clauses)
    raise TypeError(f"unsupported filter type: {type(filter).__name__}")


class RostamVectorStore(VectorStore):
    """A LangChain VectorStore over a single Rostam collection."""

    def __init__(
        self,
        client: Rostam,
        collection: str,
        embedding: Embeddings,
        *,
        auto_create: bool = True,
        metric: str = "cosine",
        full_text: bool = False,
        sparse_embedding: Optional[Any] = None,
    ):
        self._client = client
        self._collection = collection
        self._embedding = embedding
        self._auto_create = auto_create
        self._metric = metric
        self._full_text = full_text
        self._sparse_embedding = sparse_embedding
        self._created = False

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---- writes ----

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if not texts:
            return []
        vectors = self._embedding.embed_documents(texts)
        if vectors:
            self._ensure_collection(len(vectors[0]))
        if ids is None:
            ids = [hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest() for t in texts]
        metadatas = metadatas or [{} for _ in texts]
        for text, vec, meta, ext_id in zip(texts, vectors, metadatas, ids):
            self._client.upsert(
                self._collection, _to_id(ext_id), vec, content=text, metadata=meta or {}
            )
        return ids

    def add_documents(self, documents: List[LCDocument], **kwargs: Any) -> List[str]:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = kwargs.pop("ids", None)
        if ids is None and all(d.id is not None for d in documents):
            ids = [d.id for d in documents]
        return self.add_texts(texts, metadatas, ids=ids, **kwargs)

    # ---- reads ----

    def get_by_ids(self, ids: Sequence[str]) -> List[LCDocument]:
        """Fetch documents by their original string ids. Missing ids are omitted.
        The returned Document carries the ORIGINAL string id (we hold the inputs,
        so the uint64 round-trip is lossless)."""
        ids = list(ids)
        if not ids:
            return []
        uint_to_ext: Dict[int, str] = {}
        for ext in ids:
            uint_to_ext.setdefault(_to_id(ext), ext)
        pts = self._client.get_batch(
            self._collection, list(uint_to_ext.keys()), with_vector=False
        )
        out = []
        for p in pts:
            ext = uint_to_ext.get(p.id, str(p.id))
            out.append(LCDocument(id=ext, page_content=p.content, metadata=p.metadata))
        return out

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Any] = None, **kwargs: Any
    ) -> List[LCDocument]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Any] = None, **kwargs: Any
    ) -> List[Tuple[LCDocument, float]]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(vector, k, filter, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, filter: Optional[Any] = None, **kwargs: Any
    ) -> List[LCDocument]:
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(embedding, k, filter, **kwargs)]

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, filter: Optional[Any] = None, **kwargs: Any
    ) -> List[Tuple[LCDocument, float]]:
        hits = self._client.search_docs(self._collection, embedding, k, filter=_build_filter(filter))
        return [(self._to_lc(h), h.distance) for h in hits]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20,
        lambda_mult: float = 0.5, filter: Optional[Any] = None, **kwargs: Any,
    ) -> List[LCDocument]:
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20,
        lambda_mult: float = 0.5, filter: Optional[Any] = None, **kwargs: Any,
    ) -> List[LCDocument]:
        hits = self._client.search_docs(
            self._collection, embedding, fetch_k, filter=_build_filter(filter)
        )
        if not hits:
            return []
        by_id = {h.id: h for h in hits}
        pts = self._client.get_batch(self._collection, list(by_id.keys()), with_payload=False)
        order = _mmr_select(list(embedding), [(p.id, p.vector) for p in pts], k, lambda_mult)
        return [self._to_lc(by_id[i]) for i in order if i in by_id]

    def search_grouped(
        self, query: str, k: int, group_by: str, *, group_size: int = 1, filter: Optional[Any] = None
    ) -> List[List[LCDocument]]:
        """Group-by-document retrieval: top-k distinct documents, each as a list
        of its best chunk(s). A Rostam-specific extension beyond the base
        VectorStore interface."""
        vector = self._embedding.embed_query(query)
        groups = self._client.search_groups(
            self._collection, vector, k, group_by, group_size=group_size, filter=_build_filter(filter)
        )
        return [[self._to_lc(h) for h in g.hits] for g in groups]

    def hybrid_search(
        self, query: str, k: int = 4, *, filter: Optional[Any] = None,
        method: str = "rrf", alpha: float = 0.0,
    ) -> List[LCDocument]:
        """Hybrid retrieval. With a sparse_embedding configured, fuses dense +
        sparse (SPLADE-style); otherwise fuses dense + server-side BM25 over the
        raw query text (requires a full_text collection)."""
        dense = self._embedding.embed_query(query)
        flt = _build_filter(filter)
        if self._sparse_embedding is not None:
            sparse = self._sparse_embedding(query)
            results = self._client.hybrid_search(
                self._collection, dense, k, sparse=sparse, filter=flt, method=method, alpha=alpha
            )
        else:
            results = self._client.hybrid_text(
                self._collection, dense, query, k, filter=flt, method=method, alpha=alpha
            )
        if not results:
            return []
        order = [r.id for r in results]
        pts = {p.id: p for p in self._client.get_batch(self._collection, order, with_vector=False)}
        return [
            LCDocument(id=str(i), page_content=pts[i].content, metadata=pts[i].metadata)
            for i in order if i in pts
        ]

    # ---- deletes ----

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        ok = True
        for ext_id in ids:
            ok = self._client.delete(self._collection, _to_id(ext_id)) and ok
        return ok

    # ---- async ----

    async def aadd_texts(self, texts, metadatas=None, *, ids=None, **kwargs):
        return await asyncio.to_thread(self.add_texts, texts, metadatas, ids=ids, **kwargs)

    async def asimilarity_search(self, query, k=4, filter=None, **kwargs):
        return await asyncio.to_thread(self.similarity_search, query, k, filter, **kwargs)

    async def asimilarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        return await asyncio.to_thread(self.similarity_search_with_score, query, k, filter, **kwargs)

    async def amax_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5, filter=None, **kwargs):
        return await asyncio.to_thread(
            self.max_marginal_relevance_search, query, k, fetch_k, lambda_mult, filter, **kwargs
        )

    async def aget_by_ids(self, ids):
        return await asyncio.to_thread(self.get_by_ids, ids)

    async def adelete(self, ids=None, **kwargs):
        return await asyncio.to_thread(self.delete, ids, **kwargs)

    # ---- relevance scoring ----

    def _select_relevance_score_fn(self):
        # Rostam distances are "smaller = closer"; map to a 0..1 relevance score.
        return lambda distance: 1.0 / (1.0 + max(distance, 0.0))

    # ---- construction ----

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        client: Rostam,
        collection: str,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "RostamVectorStore":
        store = cls(
            client, collection, embedding,
            **{k: kwargs.pop(k) for k in ("auto_create", "metric", "full_text", "sparse_embedding") if k in kwargs},
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection on first write (idempotent). No-op if
        auto_create is off or we already created it this session."""
        if not self._auto_create or self._created:
            return
        try:
            self._client.create_collection(
                self._collection, dim, metric=self._metric,
                full_text=self._full_text or None,
            )
        except RostamError as e:
            # Already-exists is fine; anything else propagates.
            if "exist" not in (e.message or "").lower():
                raise
        self._created = True

    def _to_lc(self, d: Document) -> LCDocument:
        return LCDocument(id=str(d.id), page_content=d.content, metadata=d.metadata)
