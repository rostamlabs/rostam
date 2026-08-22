"""The unified Rostam client: one entry point, transport chosen from the target."""
from typing import Optional, Tuple
from urllib.parse import urlsplit

from . import _http, _tcp
from ._kv import _KV, _KVUnavailable
from ._types import TransportError


def _validate_port(port: int, target: str) -> None:
    """Every explicit port must be a real TCP port: 1..65535. ``urlsplit``
    already rejects out-of-range/unparseable URL ports with ValueError (see
    below), but it lets 0 through and a bare ``host:port`` target never goes
    through urlsplit at all — so this is the one place both paths converge."""
    if not (1 <= port <= 65535):
        raise TransportError(f"invalid port {port} in target {target!r}; must be between 1 and 65535")


def _parse_target(target: str) -> Tuple[str, str, int, str]:
    """Return (kind, host, port, base_url). kind is 'http' or 'tcp'.
    http/https scheme -> HTTP; tcp:// or no scheme -> TCP. A bare host with no
    port is an error (the HTTP vs TCP default ports differ, so guessing is unsafe)."""
    if "://" in target:
        try:
            parts = urlsplit(target)
            port = parts.port
        except ValueError as e:
            # urlsplit(...).port raises ValueError for a malformed or
            # out-of-range (>65535) port instead of returning None — normalize
            # it to this client's error contract rather than let it leak out.
            raise TransportError(f"invalid target {target!r}: {e}") from e
        scheme = parts.scheme.lower()
        if scheme in ("http", "https"):
            if port is None:
                port = 443 if scheme == "https" else 80
            else:
                # `port or default` would silently turn an explicit :0 into
                # the scheme default; check for None instead, and validate
                # whatever was explicitly given.
                _validate_port(port, target)
            return "http", parts.hostname or "", port, target
        if scheme == "tcp":
            host = parts.hostname or ""
            if not port:
                raise TransportError("tcp:// target requires an explicit port, e.g. tcp://host:7000")
            _validate_port(port, target)
            return "tcp", host, port, ""
        raise TransportError(f"unknown target scheme {scheme!r}; use http://, https://, or tcp://")
    # bare host:port -> TCP (the native protocol is the default path)
    if ":" not in target:
        raise TransportError(f"target {target!r} needs a port (tcp) or an http:// scheme")
    host, _, port_s = target.rpartition(":")
    try:
        port = int(port_s)
    except ValueError:
        raise TransportError(f"invalid port in target {target!r}")
    _validate_port(port, target)
    return "tcp", host, port, ""


class Rostam:
    """Unified Rostam client. Transport is selected from `target`:
    `http(s)://host:8080` -> REST; `tcp://host:7000` or bare `host:7000` -> the
    binary TCP protocol. Vector ops are flat (`r.search`, `r.hybrid_text`, ...);
    KV lives under `r.kv.*` (TCP only)."""

    def __init__(self, target: str, *, api_key: Optional[str] = None,
                 auth_token: Optional[str] = None, timeout: float = 30.0,
                 pool_maxsize: int = 8):
        kind, host, port, base_url = _parse_target(target)
        self._kind = kind
        token = auth_token if auth_token is not None else api_key
        self._t = None  # set by _connect
        self._connect(kind, host, port, base_url, token, timeout, pool_maxsize)
        #: Key-value operations (r.kv.get/put/delete/incr/expire/ping). Only
        #: reachable over the native TCP protocol; on the HTTP backend this is
        #: a _KVUnavailable sentinel that raises TransportError on any attribute
        #: access, since the KV store has no REST surface.
        self.kv = _KV(self._t) if kind == "tcp" else _KVUnavailable()

    def _connect(self, kind, host, port, base_url, token, timeout, pool_maxsize):
        if kind == "tcp":
            self._t = _tcp.TcpTransport(host, port, token, timeout, pool_maxsize=pool_maxsize)
            return
        if kind == "http":
            self._t = _http.HttpTransport(base_url, token, timeout, pool_maxsize=pool_maxsize)
            return
        raise TransportError(f"unknown transport kind {kind!r}")

    # ---- flat vector API -----------------------------------------------
    #
    # SHARED methods (create_collection, upsert, insert, upsert_batch,
    # delete, get, get_batch, scroll, search, search_docs, search_groups,
    # hybrid_search, hybrid_text, recommend, drop_collection, exists) are the
    # union both transport backends serve with an IDENTICAL signature (see
    # tests/test_transport_gaps.py's test_shared_methods_have_identical_
    # signatures) — each just forwards to the active backend (self._t),
    # which returns the unified rostam._types result objects.
    #
    # TRANSPORT-SPECIFIC methods are guarded below: `query` (the general
    # composable Query API) and the HTTP-only extras (health,
    # delete_by_filter, bulk_stage, bulk_build, batch_upsert, mv_*,
    # search_text, discover) raise TransportError on a TCP client instead of
    # silently misbehaving or AttributeError-ing. `r.kv` already raises on
    # HTTP (see _KVUnavailable).

    def _require_http(self, op: str) -> None:
        if self._kind != "http":
            raise TransportError(f"{op} requires the HTTP transport; connect with http://host:8080")

    def create_collection(self, *args, **kwargs):
        """See TcpTransport.create_collection / HttpTransport.create_collection."""
        return self._t.create_collection(*args, **kwargs)

    def drop_collection(self, *args, **kwargs):
        """See TcpTransport.drop_collection / HttpTransport.drop_collection."""
        return self._t.drop_collection(*args, **kwargs)

    def upsert(self, *args, **kwargs):
        """See TcpTransport.upsert / HttpTransport.upsert."""
        return self._t.upsert(*args, **kwargs)

    def insert(self, *args, **kwargs):
        """See TcpTransport.insert / HttpTransport.insert."""
        return self._t.insert(*args, **kwargs)

    def upsert_batch(self, *args, **kwargs):
        """See TcpTransport.upsert_batch / HttpTransport.upsert_batch."""
        return self._t.upsert_batch(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """See TcpTransport.delete / HttpTransport.delete."""
        return self._t.delete(*args, **kwargs)

    def get(self, *args, **kwargs):
        """See TcpTransport.get / HttpTransport.get."""
        return self._t.get(*args, **kwargs)

    def get_batch(self, *args, **kwargs):
        """See TcpTransport.get_batch / HttpTransport.get_batch."""
        return self._t.get_batch(*args, **kwargs)

    def scroll(self, *args, **kwargs):
        """See TcpTransport.scroll / HttpTransport.scroll."""
        return self._t.scroll(*args, **kwargs)

    def search(self, *args, **kwargs):
        """See TcpTransport.search / HttpTransport.search."""
        return self._t.search(*args, **kwargs)

    def search_docs(self, *args, **kwargs):
        """See TcpTransport.search_docs / HttpTransport.search_docs."""
        return self._t.search_docs(*args, **kwargs)

    def search_groups(self, *args, **kwargs):
        """See TcpTransport.search_groups / HttpTransport.search_groups."""
        return self._t.search_groups(*args, **kwargs)

    def hybrid_search(self, *args, **kwargs):
        """See TcpTransport.hybrid_search / HttpTransport.hybrid_search."""
        return self._t.hybrid_search(*args, **kwargs)

    def hybrid_text(self, *args, **kwargs):
        """See TcpTransport.hybrid_text / HttpTransport.hybrid_text."""
        return self._t.hybrid_text(*args, **kwargs)

    def recommend(self, *args, **kwargs):
        """See TcpTransport.recommend / HttpTransport.recommend."""
        return self._t.recommend(*args, **kwargs)

    def query(self, *args, **kwargs):
        """The general composable Query API — HTTP-only (see
        HttpTransport.query). TCP cannot build a general QuerySpec, only a
        recommend-shaped one (exactly what recommend() already sends), so a
        TCP client raises TransportError here instead of silently returning
        something narrower than what was asked for."""
        if self._kind != "http":
            raise TransportError(
                "the general query API requires the HTTP transport; connect with http://host:8080  "
                "(TCP clients use recommend())"
            )
        return self._t.query(*args, **kwargs)

    def exists(self, *args, **kwargs):
        """See TcpTransport.exists / HttpTransport.exists."""
        return self._t.exists(*args, **kwargs)

    # ---- HTTP-only extras (guarded) --------------------------------------

    def health(self, *args, **kwargs):
        """See HttpTransport.health. HTTP-only."""
        self._require_http("health")
        return self._t.health(*args, **kwargs)

    def delete_by_filter(self, *args, **kwargs):
        """See HttpTransport.delete_by_filter. HTTP-only."""
        self._require_http("delete_by_filter")
        return self._t.delete_by_filter(*args, **kwargs)

    def bulk_stage(self, *args, **kwargs):
        """See HttpTransport.bulk_stage. HTTP-only."""
        self._require_http("bulk_stage")
        return self._t.bulk_stage(*args, **kwargs)

    def bulk_build(self, *args, **kwargs):
        """See HttpTransport.bulk_build. HTTP-only."""
        self._require_http("bulk_build")
        return self._t.bulk_build(*args, **kwargs)

    def batch_upsert(self, *args, **kwargs):
        """See HttpTransport.batch_upsert. HTTP-only."""
        self._require_http("batch_upsert")
        return self._t.batch_upsert(*args, **kwargs)

    def search_text(self, *args, **kwargs):
        """See HttpTransport.search_text. HTTP-only."""
        self._require_http("search_text")
        return self._t.search_text(*args, **kwargs)

    def discover(self, *args, **kwargs):
        """See HttpTransport.discover. HTTP-only."""
        self._require_http("discover")
        return self._t.discover(*args, **kwargs)

    def mv_create_collection(self, *args, **kwargs):
        """See HttpTransport.mv_create_collection. HTTP-only (no native-TCP
        multivector op)."""
        self._require_http("mv_create_collection")
        return self._t.mv_create_collection(*args, **kwargs)

    def mv_drop_collection(self, *args, **kwargs):
        """See HttpTransport.mv_drop_collection. HTTP-only."""
        self._require_http("mv_drop_collection")
        return self._t.mv_drop_collection(*args, **kwargs)

    def mv_add(self, *args, **kwargs):
        """See HttpTransport.mv_add. HTTP-only."""
        self._require_http("mv_add")
        return self._t.mv_add(*args, **kwargs)

    def mv_search(self, *args, **kwargs):
        """See HttpTransport.mv_search. HTTP-only."""
        self._require_http("mv_search")
        return self._t.mv_search(*args, **kwargs)

    def mv_delete(self, *args, **kwargs):
        """See HttpTransport.mv_delete. HTTP-only."""
        self._require_http("mv_delete")
        return self._t.mv_delete(*args, **kwargs)

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> "Rostam":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
