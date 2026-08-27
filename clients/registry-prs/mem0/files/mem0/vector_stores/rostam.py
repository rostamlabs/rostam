import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from rostam import Rostam as RostamClient
    from rostam import RostamError
    from rostam import filters as rf
except ImportError:
    raise ImportError("The 'rostam-client' library is required. Please install it using 'pip install rostam-client'.")

from pydantic import BaseModel

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict]


# Reserved metadata key preserving the original mem0 (string) memory id, since
# Rostam point ids are unsigned 64-bit integers rather than opaque strings.
_MEM0_ID = "_mem0_id"


def _to_point_id(external_id: Union[str, int]) -> int:
    """Map an external mem0 memory id (a uuid4 string) to a Rostam point id.

    A purely-numeric id is used verbatim; anything else is hashed (BLAKE2b, 8
    bytes) to a stable 64-bit id, so repeated inserts/updates/deletes of the
    same external id address the same point across calls.
    """
    s = str(external_id)
    if s.isdigit():
        v = int(s)
        if v < (1 << 64):
            return v
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def _scalar_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rostam metadata values must be scalars or homogeneous scalar lists."""
    out: Dict[str, Any] = {}
    for k, v in (payload or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif (
            isinstance(v, (list, tuple))
            and v
            and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v)
        ):
            out[k] = list(v)
    return out


def _translate_filters(filters: Optional[Dict[str, Any]]):
    """Translate mem0's flat filter dict into a Rostam filter tree. A list
    value means membership (field in [...]); anything else is equality."""
    if not filters:
        return None
    clauses = [rf.in_(k, list(v)) if isinstance(v, (list, tuple)) else rf.eq(k, v) for k, v in filters.items()]
    return clauses[0] if len(clauses) == 1 else rf.and_(*clauses)


def _query_vector(vectors: Sequence[Any]) -> List[float]:
    """mem0 calls search(query, vectors=embedding, ...) with a single query
    embedding -- usually a flat float list, occasionally wrapped as [[...]]."""
    if vectors and isinstance(vectors[0], (list, tuple)):
        return list(vectors[0])
    return list(vectors)


class RostamDB(VectorStoreBase):
    """Rostam vector store provider.

    Rostam (https://github.com/rostamlabs/rostam) is an open-source vector
    database with an HTTP API. This provider talks to a single Rostam
    collection over `rostam-client`.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        metric: str = "cosine",
    ):
        """
        Initialize the Rostam vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            url (str, optional): Rostam server URL. Defaults to "http://localhost:8080".
            api_key (str, optional): API key for Rostam. Defaults to None.
            metric (str, optional): Distance metric for vector similarity ("cosine",
                "dot", or "euclidean"). Defaults to "cosine".
        """
        self.client = RostamClient(url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.metric = metric

        self.create_col(collection_name, embedding_model_dims, metric)

    def create_col(self, name: str, vector_size: int, distance: str = "cosine"):
        """
        Create a new collection. A no-op if the collection already exists.

        Args:
            name (str): Name of the collection.
            vector_size (int): Dimensions of the embedding model.
            distance (str, optional): Distance metric. Defaults to "cosine".
        """
        try:
            self.client.create_collection(name, vector_size, metric=distance)
        except RostamError as e:
            # Already-exists is fine; anything else propagates.
            if "exist" not in (e.message or "").lower():
                raise

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Insert vectors into the collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        if ids is None:
            raise ValueError("RostamDB.insert requires ids")

        payloads = payloads or [{} for _ in vectors]
        for vector, payload, ext_id in zip(vectors, payloads, ids):
            meta = _scalar_payload(payload or {})
            meta[_MEM0_ID] = str(ext_id)
            self.client.upsert(self.collection_name, _to_point_id(ext_id), vector, metadata=meta)

    def search(
        self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """
        Search for similar vectors.

        Args:
            query (str): Query text (unused in vector search, kept for interface consistency).
            vectors (list): Query vector to search with.
            top_k (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        vec = _query_vector(vectors)
        hits = self.client.search_docs(self.collection_name, vec, top_k, filter=_translate_filters(filters))
        return [self._parse_hit(h, scored=True) for h in hits]

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        self.client.delete(self.collection_name, _to_point_id(vector_id))

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ):
        """
        Update a vector and/or its payload. Either may be omitted, in which case the
        existing value is kept (Rostam has no partial-update endpoint, so a missing
        half is filled in from a read before the upsert).

        Args:
            vector_id (str): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        point_id = _to_point_id(vector_id)
        if vector is None or payload is None:
            existing = self.client.get_batch(self.collection_name, [point_id])
            if not existing:
                raise ValueError(f"vector {vector_id!r} not found")
            point = existing[0]
            if vector is None:
                vector = point.vector
            if payload is None:
                payload = dict(point.metadata)
                payload.pop(_MEM0_ID, None)

        meta = _scalar_payload(payload)
        meta[_MEM0_ID] = str(vector_id)
        self.client.upsert(self.collection_name, point_id, vector, metadata=meta)

    def get(self, vector_id: str) -> Optional[OutputData]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            OutputData: Retrieved vector data, or None if not found.
        """
        points = self.client.get_batch(self.collection_name, [_to_point_id(vector_id)], with_vector=False)
        if not points:
            return None
        meta = dict(points[0].metadata)
        meta.pop(_MEM0_ID, None)
        return OutputData(id=str(vector_id), score=None, payload=meta)

    def list_cols(self) -> List[str]:
        """
        List all collections.

        Rostam's client has no list-collections endpoint; this provider only ever
        tracks the single collection it was constructed with.
        """
        raise NotImplementedError(
            "Rostam has no list-collections endpoint; RostamDB tracks only the "
            "single collection it was constructed with."
        )

    def delete_col(self):
        """Delete the collection."""
        self.client.drop_collection(self.collection_name)

    def col_info(self) -> Dict:
        """
        Get information about the collection.

        Returns:
            dict: Collection metadata.
        """
        return {
            "name": self.collection_name,
            "dimension": self.embedding_model_dims,
            "distance": self.metric,
        }

    def list(self, filters: Optional[Dict] = None, top_k: Optional[int] = None) -> Tuple[List[OutputData], None]:
        """
        List vectors in the collection with optional filtering.

        Args:
            filters (dict, optional): Filters to apply. Defaults to None.
            top_k (int, optional): Maximum number of vectors to return. Defaults to None (no limit).

        Returns:
            tuple: (list of OutputData, None) -- mirrors the other providers' (results, next_cursor) shape;
            Rostam's scroll has no pagination cursor to hand back.
        """
        page = self.client.scroll(self.collection_name, filter=_translate_filters(filters), limit=top_k or 0)
        return [self._parse_hit(point) for point in page], None

    def reset(self):
        """Reset the collection by deleting and recreating it."""
        self.delete_col()
        self.create_col(self.collection_name, self.embedding_model_dims, self.metric)

    def _parse_hit(self, point, scored: bool = False) -> OutputData:
        """Convert a Rostam Point/SearchResult into mem0's OutputData, converting
        distance to a [0, 1] similarity score (higher = better) when present."""
        meta = dict(point.metadata)
        ext_id = meta.pop(_MEM0_ID, str(point.id))
        score = (1.0 / (1.0 + max(point.distance, 0.0))) if scored else None
        return OutputData(id=ext_id, score=score, payload=meta)
