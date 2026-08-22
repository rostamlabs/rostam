"""Cross-transport parity tests for the unified ``Rostam`` facade.

The facade forwards shared vector methods straight to whichever backend
(``TcpTransport`` / ``HttpTransport``) answered ``target``. That is only a
safe "one API, two transports" promise if the shared methods really do have
identical signatures on both backends, and if the transport-specific methods
(the general Query API, and the HTTP-only extras) fail loudly on the wrong
transport instead of silently misbehaving or AttributeError-ing.

All three tests build clients against nothing-listening addresses:
construction does no I/O (see TcpTransport / HttpTransport docstrings), so
this needs no server.
"""

import inspect

import pytest

from rostam import TransportError
from rostam.rostam import Rostam


def test_http_only_methods_raise_on_tcp():
    r = Rostam("tcp://127.0.0.1:7000")  # no I/O at construction
    for call in (lambda: r.health(), lambda: r.delete_by_filter("c", {}),
                 lambda: r.bulk_stage("c", [], []), lambda: r.bulk_build("c"),
                 lambda: r.batch_upsert("c", [], []), lambda: r.query("c", [])):
        with pytest.raises(TransportError):
            call()


def test_shared_methods_have_identical_signatures():
    shared = ["search", "search_docs", "search_groups", "hybrid_search",
              "hybrid_text", "recommend", "upsert", "insert", "upsert_batch",
              "get", "get_batch", "scroll", "delete", "create_collection",
              "drop_collection", "exists"]
    rt = Rostam("tcp://127.0.0.1:7000")
    rh = Rostam("http://127.0.0.1:8080")
    for name in shared:
        ft, fh = getattr(rt, name, None), getattr(rh, name, None)
        assert callable(ft) and callable(fh), f"{name} missing on a transport"
        # signatures must match (the unification promise) — compare the underlying
        # backend method params, ignoring self.
        st = inspect.signature(getattr(rt._t, name))
        sh = inspect.signature(getattr(rh._t, name))
        assert list(st.parameters) == list(sh.parameters), (
            f"{name} signature differs: tcp{list(st.parameters)} vs http{list(sh.parameters)}")


def test_recommend_same_call_shape_on_both():
    # recommend(collection, positive, *, k=..., ...) must accept the same call on both.
    for target in ("tcp://127.0.0.1:7000", "http://127.0.0.1:8080"):
        r = Rostam(target)
        sig = inspect.signature(getattr(r._t, "recommend"))
        assert "positive" in sig.parameters
        assert sig.parameters["k"].default == 10   # keyword with default on both


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
