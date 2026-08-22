"""LlamaIndex ``VectorStore`` adapter backed by a Rostam collection.

Requires ``llama-index-core`` (``pip install rostam-client[llamaindex]``). Stores
each node's embedding, text (as Rostam content), and serialized node metadata;
serves ``query`` via Rostam KNN and ``delete`` by ref-doc filter.

    from rostam import Rostam
    from rostam.llamaindex import RostamVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    client = Rostam("http://localhost:8080")
    client.create_collection("docs", dim=1536, metric="cosine")
    store = RostamVectorStore(client=client, collection="docs")
    index = VectorStoreIndex.from_documents(
        docs, storage_context=StorageContext.from_defaults(vector_store=store)
    )
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from . import filters as f
from ._ids import to_uint64
from ._types import RostamError
from .rostam import Rostam

# Reserved metadata key holding a node's ref-doc id, so delete(ref_doc_id) can
# purge every node of a document via a Rostam metadata filter.
_REF_KEY = "_rostam_ref_doc_id"

_OP = {
    "==": f.eq, "!=": f.ne, ">": f.gt, ">=": f.gte, "<": f.lt, "<=": f.lte,
}


def _translate(filters: Optional[MetadataFilters]):
    """Translate LlamaIndex MetadataFilters into a Rostam filter (best effort:
    scalar comparisons + IN, combined by the AND/OR condition)."""
    if filters is None or not getattr(filters, "filters", None):
        return None
    clauses = []
    for mf in filters.filters:
        opattr: Any = getattr(mf, "operator", "==")
        op = str(opattr.value if hasattr(opattr, "value") else opattr)
        if op == "in":
            clauses.append(f.in_(mf.key, list(mf.value)))
        elif op in _OP:
            clauses.append(_OP[op](mf.key, mf.value))
        else:
            raise ValueError(f"unsupported LlamaIndex filter operator: {op}")
    if len(clauses) == 1:
        return clauses[0]
    cond = str(getattr(filters, "condition", "and"))
    cond = cond.value if hasattr(cond, "value") else cond
    return f.or_(*clauses) if cond == "or" else f.and_(*clauses)


def _scalar_meta(meta: dict) -> dict:
    """Drop values Rostam can't store (None / nested), keeping the flat scalars
    LlamaIndex's flat_metadata serialization produces."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
            out[k] = list(v)
    return out


class RostamVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    collection: str

    _client: Rostam
    _default_k: int
    _auto_create: bool
    _metric: str
    _full_text: bool
    _created: bool
    _sparse_embedding: Any

    def __init__(
        self, client: Rostam, collection: str, *,
        default_top_k: int = 10, auto_create: bool = True,
        metric: str = "cosine", full_text: bool = False,
        sparse_embedding: Optional[Any] = None, **kwargs: Any,
    ):
        super().__init__(collection=collection, **kwargs)
        self._client = client
        self._default_k = default_top_k
        self._auto_create = auto_create
        self._metric = metric
        self._full_text = full_text
        self._created = False
        self._sparse_embedding = sparse_embedding

    def _ensure_collection(self, dim: int) -> None:
        if not self._auto_create or self._created:
            return
        try:
            self._client.create_collection(
                self.collection, dim, metric=self._metric,
                full_text=self._full_text or None,
            )
        except RostamError as e:
            if "exist" not in (e.message or "").lower():
                raise
        self._created = True

    @classmethod
    def class_name(cls) -> str:
        return "RostamVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        nodes = list(nodes)
        if nodes:
            self._ensure_collection(len(nodes[0].get_embedding()))
        ids: List[str] = []
        for node in nodes:
            meta = _scalar_meta(node_to_metadata_dict(node, remove_text=True, flat_metadata=self.flat_metadata))
            if node.ref_doc_id:
                meta[_REF_KEY] = node.ref_doc_id
            self._client.upsert(
                self.collection,
                to_uint64(node.node_id),
                node.get_embedding(),
                content=node.get_content(metadata_mode=MetadataMode.NONE) or "",
                metadata=meta,
            )
            ids.append(node.node_id)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self._client.delete_by_filter(self.collection, f.eq(_REF_KEY, ref_doc_id))

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        k = query.similarity_top_k or self._default_k
        flt = _translate(query.filters)
        dense = list(query.query_embedding)
        is_hybrid = (
            getattr(query, "mode", None) == VectorStoreQueryMode.HYBRID
            and query.query_str
        )
        if is_hybrid:
            if self._sparse_embedding is not None:
                results = self._client.hybrid_search(
                    self.collection, dense, k, sparse=self._sparse_embedding(query.query_str), filter=flt
                )
            else:
                results = self._client.hybrid_text(
                    self.collection, dense, query.query_str, k, filter=flt
                )
            order = [r.id for r in results]
            pts = {p.id: p for p in self._client.get_batch(self.collection, order, with_vector=False)}
            pt_docs = [pts[i] for i in order if i in pts]
            return self._result_from_points(pt_docs)
        docs = self._client.search_docs(self.collection, dense, k, filter=flt)
        return self._result_from_docs(docs)

    async def async_add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        return await asyncio.to_thread(self.add, nodes, **kwargs)

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return await asyncio.to_thread(self.query, query, **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        return await asyncio.to_thread(self.delete, ref_doc_id, **kwargs)

    def _result_from_docs(self, docs) -> VectorStoreQueryResult:
        nodes, sims, ids = [], [], []
        for d in docs:
            meta = dict(d.metadata)
            meta.pop(_REF_KEY, None)
            node = metadata_dict_to_node(meta, text=d.content)
            nodes.append(node)
            ids.append(node.node_id)
            sims.append(1.0 / (1.0 + max(d.distance, 0.0)))
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)

    def _result_from_points(self, pts) -> VectorStoreQueryResult:
        nodes, sims, ids = [], [], []
        for rank, p in enumerate(pts, start=1):
            meta = dict(p.metadata)
            meta.pop(_REF_KEY, None)
            node = metadata_dict_to_node(meta, text=p.content)
            nodes.append(node)
            ids.append(node.node_id)
            sims.append(1.0 / (1.0 + rank))   # rank-based; never saturates to 1.0
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)
