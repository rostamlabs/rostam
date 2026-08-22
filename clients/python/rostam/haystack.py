"""Haystack 2.x integration backed by a Rostam collection.

Requires ``haystack-ai`` (``pip install rostam-client[haystack]``). Provides a
``RostamDocumentStore`` implementing the Haystack ``DocumentStore`` protocol
(write/delete/count/filter via Rostam's scroll listing) and a
``RostamEmbeddingRetriever`` component for embedding retrieval.

    from rostam import Rostam
    from rostam.haystack import RostamDocumentStore, RostamEmbeddingRetriever

    client = Rostam("http://localhost:8080")
    client.create_collection("docs", dim=384, metric="cosine")
    store = RostamDocumentStore(url="http://localhost:8080", collection="docs")
    retriever = RostamEmbeddingRetriever(document_store=store, top_k=5)

Documents must carry embeddings (Rostam is a vector store). Writes use
overwrite semantics regardless of DuplicatePolicy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DuplicatePolicy

from . import filters as f
from ._ids import to_uint64
from .rostam import Rostam

# Reserved metadata key preserving the original Haystack (string) document id,
# since Rostam point ids are uint64.
_HS_ID = "_hs_id"

_LEAF = {"==": f.eq, "!=": f.ne, ">": f.gt, ">=": f.gte, "<": f.lt, "<=": f.lte}


def _translate(filters: Optional[Dict[str, Any]]):
    """Translate a Haystack 2.x filter tree into a Rostam filter."""
    if not filters:
        return None
    return _node(filters)


def _node(flt: Dict[str, Any]):
    if "conditions" in flt:
        op = str(flt["operator"]).upper()
        subs = [_node(c) for c in flt["conditions"]]
        if op == "AND":
            return f.and_(*subs)
        if op == "OR":
            return f.or_(*subs)
        if op == "NOT":
            return f.not_(subs[0] if len(subs) == 1 else f.and_(*subs))
        raise ValueError(f"unsupported Haystack logical operator: {op}")
    field = flt["field"]
    if field.startswith("meta."):
        field = field[len("meta."):]
    op, val = flt["operator"], flt["value"]
    if op == "in":
        return f.in_(field, list(val))
    if op in _LEAF:
        return _LEAF[op](field, val)
    raise ValueError(f"unsupported Haystack operator: {op}")


def _scalar(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
            out[k] = list(v)
    return out


class RostamDocumentStore:
    """A Haystack DocumentStore over a Rostam collection (vector store)."""

    def __init__(self, url: str = "http://localhost:8080", collection: str = "haystack", api_key: Optional[str] = None):
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self._client = Rostam(url, api_key=api_key)

    def count_documents(self) -> int:
        return len(self._client.scroll(self.collection))

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        docs = self._client.scroll(self.collection, filter=_translate(filters))
        return [self._to_haystack(d) for d in docs]

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        n = 0
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"RostamDocumentStore requires an embedding (document {doc.id!r} has none)")
            meta = _scalar(doc.meta)
            meta[_HS_ID] = doc.id
            self._client.upsert(self.collection, to_uint64(doc.id), doc.embedding, content=doc.content or "", metadata=meta)
            n += 1
        return n

    def delete_documents(self, document_ids: List[str]) -> None:
        for did in document_ids:
            self._client.delete(self.collection, to_uint64(did))

    # Called by RostamEmbeddingRetriever.
    def _embedding_retrieval(
        self, query_embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        docs = self._client.search_docs(self.collection, query_embedding, top_k, filter=_translate(filters))
        return [self._to_haystack(d, scored=True) for d in docs]

    def _to_haystack(self, d, scored: bool = False) -> Document:
        meta = dict(d.metadata)
        hid = meta.pop(_HS_ID, str(d.id))
        return Document(
            id=hid, content=d.content, meta=meta,
            score=(1.0 / (1.0 + max(d.distance, 0.0))) if scored else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, url=self.url, collection=self.collection, api_key=self.api_key)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RostamDocumentStore":
        return default_from_dict(cls, data)


@component
class RostamEmbeddingRetriever:
    """Haystack retriever: embedding query -> top-k documents from Rostam."""

    def __init__(self, document_store: RostamDocumentStore, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
        self._store = document_store
        self._top_k = top_k
        self._filters = filters

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None):
        return {"documents": self._store._embedding_retrieval(
            query_embedding, top_k if top_k is not None else self._top_k, filters if filters is not None else self._filters,
        )}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, document_store=self._store.to_dict(), top_k=self._top_k, filters=self._filters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RostamEmbeddingRetriever":
        init = data.get("init_parameters", {})
        init["document_store"] = RostamDocumentStore.from_dict(init["document_store"])
        return default_from_dict(cls, data)
