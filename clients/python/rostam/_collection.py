"""A collection-scoped handle over a Rostam client.

``r.collection("posts")`` returns a :class:`Collection` bound to that name, so
the collection argument stops being repeated on every call::

    posts = r.collection("posts")
    posts.upsert(1, vec, content="hello")
    hits = posts.search(query_vec, k=10)

Each method forwards to the identically-named flat method on the client with the
collection name supplied as the first argument — so transport rules still apply
(e.g. ``query`` / ``delete_by_filter`` raise ``TransportError`` on a TCP client,
exactly as the flat ``r.query`` / ``r.delete_by_filter`` do). Mirrors the Go
client's ``client.Collection`` handle.

Scope: the handle carries the collection-scoped operations — Go's ``Collection``
surface plus the extra collection-scoped ops the Python flat API has (``exists``,
``delete_by_filter``). Non-collection-scoped or HTTP-only extras (``health``,
``bulk_stage`` / ``bulk_build``, ``search_text``, ``discover``, ``mv_*``) stay on
the flat client by design; reach for them via ``r.<method>`` when you need them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle; only needed for type checkers
    from .rostam import Rostam


class Collection:
    """A handle bound to one collection name. See the module docstring."""

    __slots__ = ("_r", "name")

    def __init__(self, client: "Rostam", name: str):
        self._r = client
        #: The collection name this handle is bound to.
        self.name = name

    def __repr__(self) -> str:
        return f"Collection({self.name!r})"

    # ---- lifecycle ----------------------------------------------------
    def create(self, *args, **kwargs):
        """Create this collection. Forwards to ``client.create_collection(name, ...)``."""
        return self._r.create_collection(self.name, *args, **kwargs)

    def drop(self, *args, **kwargs):
        """Drop this collection. Forwards to ``client.drop_collection(name)``."""
        return self._r.drop_collection(self.name, *args, **kwargs)

    # ---- writes -------------------------------------------------------
    def upsert(self, *args, **kwargs):
        return self._r.upsert(self.name, *args, **kwargs)

    def insert(self, *args, **kwargs):
        return self._r.insert(self.name, *args, **kwargs)

    def upsert_batch(self, *args, **kwargs):
        return self._r.upsert_batch(self.name, *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._r.delete(self.name, *args, **kwargs)

    def delete_by_filter(self, *args, **kwargs):
        return self._r.delete_by_filter(self.name, *args, **kwargs)

    # ---- reads --------------------------------------------------------
    def get(self, *args, **kwargs):
        return self._r.get(self.name, *args, **kwargs)

    def get_batch(self, *args, **kwargs):
        return self._r.get_batch(self.name, *args, **kwargs)

    def scroll(self, *args, **kwargs):
        return self._r.scroll(self.name, *args, **kwargs)

    def exists(self, *args, **kwargs):
        return self._r.exists(self.name, *args, **kwargs)

    # ---- search -------------------------------------------------------
    def search(self, *args, **kwargs):
        return self._r.search(self.name, *args, **kwargs)

    def search_docs(self, *args, **kwargs):
        return self._r.search_docs(self.name, *args, **kwargs)

    def search_groups(self, *args, **kwargs):
        return self._r.search_groups(self.name, *args, **kwargs)

    def hybrid_search(self, *args, **kwargs):
        return self._r.hybrid_search(self.name, *args, **kwargs)

    def hybrid_text(self, *args, **kwargs):
        return self._r.hybrid_text(self.name, *args, **kwargs)

    def recommend(self, *args, **kwargs):
        return self._r.recommend(self.name, *args, **kwargs)

    def query(self, *args, **kwargs):
        return self._r.query(self.name, *args, **kwargs)
