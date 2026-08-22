"""The collection-scoped handle: r.collection(name) binds the collection so it
stops being repeated, forwarding each call to the identically-named flat method
with the name as the first positional argument."""

import pytest

from rostam import Collection, Rostam, TransportError


class _Spy:
    """Records every attribute call; any method name is accepted."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return ("result", name)
        return rec


# method on the handle -> flat method it must forward to
FORWARDS = {
    "create": "create_collection",
    "drop": "drop_collection",
    "upsert": "upsert",
    "insert": "insert",
    "upsert_batch": "upsert_batch",
    "delete": "delete",
    "delete_by_filter": "delete_by_filter",
    "get": "get",
    "get_batch": "get_batch",
    "scroll": "scroll",
    "exists": "exists",
    "search": "search",
    "search_docs": "search_docs",
    "search_groups": "search_groups",
    "hybrid_search": "hybrid_search",
    "hybrid_text": "hybrid_text",
    "recommend": "recommend",
    "query": "query",
}


def test_binds_name_and_forwards_positionally():
    spy = _Spy()
    col = Collection(spy, "posts")
    assert col.name == "posts"
    for handle_method, flat_method in FORWARDS.items():
        spy.calls.clear()
        getattr(col, handle_method)("ARG", kw=1)
        assert len(spy.calls) == 1, f"{handle_method} did not forward exactly once"
        called, args, kwargs = spy.calls[0]
        assert called == flat_method, f"{handle_method} forwarded to {called}, want {flat_method}"
        assert args == ("posts", "ARG"), f"{handle_method} must pass name first, got {args}"
        assert kwargs == {"kw": 1}


def test_factory_returns_bound_handle_without_io():
    r = Rostam("tcp://127.0.0.1:7000")  # construction does no I/O
    col = r.collection("c")
    assert isinstance(col, Collection)
    assert col.name == "c"


def test_repr():
    assert repr(Collection(_Spy(), "x")) == "Collection('x')"


def test_forwarding_preserves_transport_guard():
    # query is HTTP-only; forwarding through the handle must still raise the
    # facade's TransportError on a TCP client (before any I/O).
    r = Rostam("tcp://127.0.0.1:7000")
    with pytest.raises(TransportError):
        r.collection("c").query([])
