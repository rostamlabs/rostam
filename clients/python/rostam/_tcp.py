"""TCP transport backend: Rostam's native binary protocol, over the ``-tcp``
port.

``TcpTransport`` owns the socket pool, the wire framing (``_call``), and the
``_vecwire`` encode/decode calls for every vector op. It is the migration
target for what used to be ``kv.py``'s ``_VectorAPI`` — same op names, same
wire layouts, same auth/framing — but every method now returns one of the
unified ``rostam._types`` result types instead of ad-hoc dicts/tuples, so
callers get the same shapes regardless of which transport backend answered.

Construction (``TcpTransport(host, port, ...)``) does **no** network I/O: the
underlying ``_SocketPool`` connects lazily on the first call, exactly like the
pre-unification native client. This matters for tests that build a client
against a target with nothing listening yet.

Wire, for reference (all big-endian) — unchanged from ``kv.py``:

    frame     [len u32][body]
    body v1   [opNameLen u8][opName][argsLen u32][args]
    body v2   [0x02][tokenLen u8][token][opNameLen u8][opName][argsLen u32][args]
    response  [bodyLen u32][status u8][payloadLen u32][payload]

v2 is used when an auth token is set, v1 otherwise — mirroring the Go client.
"""

from __future__ import annotations

import socket
import struct
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _vecwire
from ._types import (
    Document,
    Group,
    GroupResults,
    Point,
    RostamError,
    SearchResult,
    SearchResults,
    ScrollPage,
)

_STATUS_OK = 0
_STATUS_NOT_FOUND = 1
_STATUS_NOT_LEADER = 2
_STATUS_ERROR = 3
_STATUS_UNAUTHORIZED = 4

_PROTOCOL_V2 = 0x02
# The server rejects a frame whose length prefix exceeds this; used only to
# fail fast with a clear message rather than send a doomed frame.
_MAX_FRAME = 64 * 1024 * 1024


class _SocketPool:
    """A tiny pool of connected sockets, safe to share across threads.

    Hands a live socket to a caller, takes it back when they are done. A
    socket that errored mid-call is discarded rather than returned, so a
    broken connection never poisons the next request. Moved verbatim from
    ``kv.py``.
    """

    def __init__(self, host: str, port: int, timeout: float, maxsize: int):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._maxsize = maxsize
        self._free: List[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        # Nagle off: these are small, latency-sensitive request/response frames.
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def acquire(self) -> Tuple[socket.socket, bool]:
        """Return (socket, reused). ``reused`` is True when the socket came
        from the free list rather than a fresh connect() — only a reused
        socket can be silently dead from a peer's idle-timeout close, so only
        that case is a candidate for the stale-connection retry in `_call`."""
        with self._lock:
            if self._closed:
                raise RostamError("client is closed")
            s = self._free.pop() if self._free else None
        if s is None:
            return self._connect(), False
        s.settimeout(self._timeout)
        return s, True

    def release(self, s: socket.socket) -> None:
        with self._lock:
            if self._closed or len(self._free) >= self._maxsize:
                drop = True
            else:
                self._free.append(s)
                drop = False
        if drop:
            _silent_close(s)

    def discard(self, s: socket.socket) -> None:
        _silent_close(s)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            socks, self._free = self._free, []
        for s in socks:
            _silent_close(s)


def _silent_close(s: socket.socket) -> None:
    try:
        s.close()
    except OSError:
        pass


class _StaleConnection(Exception):
    """Internal signal: a pooled socket failed before any response byte
    arrived. Never raised for a freshly-connected socket, and never once a
    response has started arriving — see `_call`."""


def _recv_exactly(s: socket.socket, n: int, *, stale_ok: bool = False) -> bytes:
    """Read exactly n bytes or raise — a short read means the peer went away.

    When ``stale_ok`` is set and the peer closes before delivering a single
    byte back (``got == 0`` — whether via a graceful empty read or the OS
    surfacing it as ``ConnectionResetError``/``ConnectionError``), raises
    ``_StaleConnection`` instead of ``RostamError``: that pattern is what a
    pooled connection that died while idle looks like from here, and `_call`
    may retry it. Any other short read or reset (some bytes already arrived)
    means the peer vanished mid-response, which is a real failure and always
    raises ``RostamError``/the original exception."""
    chunks = []
    got = 0
    while got < n:
        try:
            b = s.recv(n - got)
        except (BrokenPipeError, ConnectionResetError, ConnectionError) as e:
            if stale_ok and got == 0:
                raise _StaleConnection(str(e)) from e
            raise
        if not b:
            if stale_ok and got == 0:
                raise _StaleConnection("connection closed before any response arrived")
            raise RostamError("connection closed by server mid-response")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _status_message(status: int, payload: bytes) -> str:
    detail = payload.decode("utf-8", "replace") if payload else ""
    name = {
        _STATUS_NOT_LEADER: "not leader",
        _STATUS_ERROR: "server error",
        _STATUS_UNAUTHORIZED: "unauthorized (auth token missing or invalid)",
    }.get(status, f"status {status}")
    return f"{name}: {detail}" if detail else name


def _to_document(d: Dict[str, Any]) -> Document:
    return Document(id=d["id"], distance=d["distance"], score=d["score"],
                     content=d["content"], metadata=d["metadata"])


class TcpTransport:
    """Vector-database operations over Rostam's native binary TCP protocol.

    Shares one connection pool and auth token across every call. The binary
    arg layouts live in ``rostam._vecwire`` and are differential-tested
    byte-for-byte against the Go encoders; the JSON-carrying parts (metadata,
    filter, content) round-trip through a real server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        token: Optional[str] = None,
        timeout: float = 30.0,
        *,
        pool_maxsize: int = 8,
    ):
        self._token = token or ""
        if len(self._token.encode("utf-8")) > 255:
            raise ValueError("auth_token is longer than 255 bytes")
        # No network I/O here: _SocketPool connects lazily on first acquire().
        self._pool = _SocketPool(host, port, timeout, pool_maxsize)

    # ---- framing -----------------------------------------------------------

    def _encode_body(self, op: str, args: bytes) -> bytes:
        op_b = op.encode("ascii")
        if len(op_b) > 0xFF:
            raise ValueError("op name too long")
        v1 = bytes([len(op_b)]) + op_b + struct.pack(">I", len(args)) + args
        if not self._token:
            return v1
        tok = self._token.encode("utf-8")
        return bytes([_PROTOCOL_V2, len(tok)]) + tok + v1

    def _exchange(self, s: socket.socket, frame: bytes, *, stale_ok: bool) -> Tuple[int, bytes]:
        """Send one frame and parse its response into (status, payload).

        Send/read AND full parse happen here so a socket is only handed back
        to `_call` after a well-formed response — a truncated or malformed
        frame is always the caller's problem to discard, never pooled.

        When ``stale_ok``, a failure before any response byte arrives (send
        itself breaks, or the peer closes on the very first read) raises
        `_StaleConnection` instead of the usual error, so `_call` can retry
        once on a fresh connection. Any failure once bytes have started
        arriving — even with ``stale_ok`` — raises normally: the server has
        necessarily seen (and may have executed) the request by then, so a
        retry is no longer just re-probing a dead idle socket.
        """
        try:
            s.sendall(frame)
        except (BrokenPipeError, ConnectionResetError, ConnectionError) as e:
            if stale_ok:
                raise _StaleConnection(str(e)) from e
            raise
        body_len = struct.unpack(">I", _recv_exactly(s, 4, stale_ok=stale_ok))[0]
        if body_len < 5 or body_len > _MAX_FRAME:
            raise RostamError(f"invalid response frame length {body_len}")
        resp = _recv_exactly(s, body_len)
        status = resp[0]
        payload_len = struct.unpack(">I", resp[1:5])[0]
        if 5 + payload_len != body_len:
            raise RostamError("response payload length does not match frame")
        payload = resp[5:5 + payload_len]
        return status, payload

    def _call(self, op: str, args: bytes, *, idempotent: bool = False) -> Optional[bytes]:
        """Send one op, return its payload. Raises RostamError on a non-OK status.

        A miss (StatusNotFound) is NOT an error here — it returns ``None`` — so
        read ops can distinguish "absent" from "empty value" without an
        exception. Every other non-OK status raises.

        ``idempotent=True`` (read ops only — never vector_upsert/insert/
        delete or kv put/del/incr/expire) allows ONE retry on a fresh
        connection, but only when the failure happened on a *reused* pooled
        socket AND before any response byte arrived. A server or middlebox
        can close an idle pooled connection at any time; the next op to land
        on it would otherwise fail with a confusing transport error even
        though nothing about the request itself was wrong. A freshly
        connected socket, or a failure after the response has started
        arriving, is never retried.
        """
        body = self._encode_body(op, args)
        if 4 + len(body) > _MAX_FRAME:
            raise RostamError("request frame exceeds the server's frame limit")
        frame = struct.pack(">I", len(body)) + body

        retry_ok = idempotent
        force_fresh = False
        while True:
            # Acquire may connect, and a failed connect raises OSError —
            # convert it to the client's RostamError contract like every
            # other transport failure. On a stale-connection retry we must
            # open a brand-new socket rather than acquire(), which could hand
            # back another idle (possibly also-stale) pooled socket.
            try:
                if force_fresh:
                    s, reused = self._pool._connect(), False
                else:
                    s, reused = self._pool.acquire()
            except OSError as e:
                raise RostamError(f"connect failed: {e}") from e
            force_fresh = False

            try:
                status, payload = self._exchange(s, frame, stale_ok=reused and retry_ok)
            except _StaleConnection:
                self._pool.discard(s)
                if reused and retry_ok:
                    retry_ok = False  # exactly one retry, on a fresh connection
                    force_fresh = True  # bypass the idle pool for that retry
                    continue
                raise RostamError("connection closed by server mid-response")
            except (OSError, RostamError, struct.error) as e:
                self._pool.discard(s)
                if isinstance(e, RostamError):
                    raise
                raise RostamError(f"transport error: {e}") from e

            # Well-formed response: the connection is healthy even if the op
            # failed at the application level (StatusError etc.), so keep it
            # pooled.
            self._pool.release(s)

            if status == _STATUS_OK:
                return payload
            if status == _STATUS_NOT_FOUND:
                return None
            # _types.RostamError (unlike client.py's HTTP-status-carrying
            # variant) takes no `status` kwarg — fold the status into the
            # message instead.
            raise RostamError(_status_message(status, payload))

    def close(self) -> None:
        self._pool.close()

    # ---- vector ops ----------------------------------------------------
    #
    # search-family reads (search, search_docs, search_groups, hybrid_search,
    # hybrid_text, recommend, query) surface each decoder's degraded/missing
    # trailer via SearchResults/GroupResults; scroll() carries next_cursor via
    # ScrollPage. get()/get_batch() return Point(s): only id/vector/content/
    # metadata are cross-transport (the wire also carries ttl_ms/sparse/
    # version, which the unified Point type does not expose — see _types.Point).

    def create_collection(
        self,
        name: str,
        dim: int,
        *,
        metric: str = "cosine",
        m: int = 0,
        ef_construction: int = 0,
        ef_search: int = 0,
        seed: int = 0,
        quant: str = "",
        persistent: bool = False,
        rescore_factor: int = 0,
        extend_candidates: bool = False,
        extend_candidates_max: int = 0,
        level0_full_degree: bool = False,
        quantized_build: bool = False,
        partitions: int = 0,
        index_type: str = "",
        ivf_nlist: int = 0,
        ivf_nprobe: int = 0,
        ivf_pq: bool = False,
        ivf_pq_m: int = 0,
        ivf_rerank: bool = False,
        quant_pq_m: int = 0,
        opq: bool = False,
        pq_drop_vecs: bool = False,
        ivf_train_threshold: int = 0,
        ivf_drift_retrain: bool = False,
        ivf_drift_growth_factor: float = 0.0,
        ivf_drift_factor: float = 0.0,
        filter_first_relative_bp: int = 0,
        opq_iters: int = 0,
        full_text: Any = None,
        sq_bits: int = 0,
        prq_layers: int = 0,
        vamana_r: int = 0,
        vamana_l: int = 0,
        vamana_alpha: float = 0.0,
        anisotropic_eta: float = 0.0,
        soar: bool = False,
        soar_lambda: float = 0.0,
        pq_nbits: int = 0,
    ) -> None:
        """Create a vector collection. Keyword config is identical to
        HttpTransport.create_collection (see there for what each knob does) —
        every field the config trailer accepts is a named parameter here too,
        so there is no catch-all `**cfg` any more."""
        conf: Dict[str, Any] = dict(
            dim=dim, metric=metric, m=m, ef_construction=ef_construction, ef_search=ef_search,
            seed=seed, quant=quant, persistent=persistent, rescore_factor=rescore_factor,
            extend_candidates=extend_candidates, extend_candidates_max=extend_candidates_max,
            level0_full_degree=level0_full_degree, quantized_build=quantized_build,
            partitions=partitions, index_type=index_type,
            ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe, ivf_pq=ivf_pq, ivf_pq_m=ivf_pq_m,
            ivf_rerank=ivf_rerank, quant_pq_m=quant_pq_m, opq=opq, pq_drop_vecs=pq_drop_vecs,
            ivf_train_threshold=ivf_train_threshold, ivf_drift_retrain=ivf_drift_retrain,
            ivf_drift_growth_factor=ivf_drift_growth_factor, ivf_drift_factor=ivf_drift_factor,
            filter_first_relative_bp=filter_first_relative_bp, opq_iters=opq_iters,
            full_text=full_text, sq_bits=sq_bits, prq_layers=prq_layers,
            vamana_r=vamana_r, vamana_l=vamana_l, vamana_alpha=vamana_alpha,
            anisotropic_eta=anisotropic_eta, soar=soar, soar_lambda=soar_lambda, pq_nbits=pq_nbits,
        )
        self._call("vector_create_collection", _vecwire.encode_create_collection_args(name, conf))

    def drop_collection(self, name: str) -> None:
        """Delete a collection and its data."""
        self._call("vector_drop_collection", _vecwire.encode_drop_collection_args(name))

    def upsert(self, collection: str, id: int, vector: Sequence[float], *, content: str = "",
               metadata: Optional[Dict[str, Any]] = None, ttl_ms: int = 0,
               sparse: Optional[Dict[str, Sequence]] = None) -> None:
        """Insert or replace a point, optionally with stored content for RAG."""
        self._call("vector_upsert", _vecwire.encode_upsert_args(
            collection, int(id), vector, content=content, ttl_ms=ttl_ms,
            metadata=metadata, sparse=sparse))

    def insert(self, collection: str, id: int, vector: Sequence[float], *,
               metadata: Optional[Dict[str, Any]] = None, ttl_ms: int = 0,
               sparse: Optional[Dict[str, Sequence]] = None) -> None:
        """Create-only insert (errors if the id is live)."""
        self._call("vector_insert", _vecwire.encode_insert_args(
            collection, int(id), vector, ttl_ms=ttl_ms, metadata=metadata, sparse=sparse))

    def upsert_batch(self, collection: str, points: Sequence[Dict[str, Any]]) -> None:
        """N sequential vector_upsert ops over one connection, each awaited
        before the next is sent — there is no native-TCP batch-upsert wire op,
        and this does not pipeline (matching the Go client). Each point dict:
        {id, vector, content="", ttl_ms=0, metadata=None, sparse=None}."""
        for args in _vecwire.encode_upsert_batch_args(collection, points):
            self._call("vector_upsert", args)

    def delete(self, collection: str, id: int) -> bool:
        """Delete a point. Returns whether it existed."""
        payload = self._call("vector_delete", _vecwire.encode_delete_args(collection, int(id)))
        return bool(payload and payload[0])

    def exists(self, collection: str, id: int) -> bool:
        payload = self._call("vector_exists", _vecwire.encode_exists_args(collection, int(id)),
                              idempotent=True)
        return _vecwire.decode_exists_result(payload or b"\x00")

    def get(self, collection: str, id: int, *, with_vector: bool = True,
            with_payload: bool = True) -> Optional[Point]:
        """Fetch a point by id, or None if absent."""
        flags = (0x01 if with_vector else 0) | (0x02 if with_payload else 0)
        payload = self._call("vector_get", _vecwire.encode_get_args(collection, int(id), flags),
                             idempotent=True)
        if payload is None:
            return None
        got = _vecwire.decode_get_result(payload)
        if got is None:
            # A miss comes back as StatusOK with a found=0 body (not
            # StatusNotFound), so the payload is non-None but decodes to None.
            return None
        # Lift stored content out of the reserved $content key, mirroring the
        # HTTP client's Point shape.
        meta = got.get("metadata") or {}
        content = meta.pop("$content", "")
        return Point(id=int(id), vector=got.get("vector"), content=content, metadata=meta)

    def get_batch(self, collection: str, ids: Sequence[int], *, with_vector: bool = True,
                  with_payload: bool = True) -> List[Point]:
        """Fetch multiple points by id in one round trip. Returns one Point per
        PRESENT id (absent ids are omitted, matching the HTTP backend's
        get_batch contract) — never raises on a partial miss."""
        flags = (0x01 if with_vector else 0) | (0x02 if with_payload else 0)
        payload = self._call("vector_get_batch",
                              _vecwire.encode_vector_get_batch_args(collection, list(ids), flags),
                              idempotent=True)
        rows = _vecwire.decode_get_batch_result(payload or b"\x00\x00\x00\x00")
        points: List[Point] = []
        for row in rows:
            if not row.get("found"):
                continue
            meta = row.get("metadata") or {}
            content = meta.pop("$content", "")
            points.append(Point(id=row["id"], vector=row.get("vector"), content=content, metadata=meta))
        return points

    def scroll(self, collection: str, *, filter: Optional[Dict[str, Any]] = None,
               limit: int = 0, cursor: str = "") -> ScrollPage:
        """Page through a collection's points in id order. Returns a
        ScrollPage of Document — iterate/len it like a list, and read
        ``.next_cursor`` to fetch the next page (pass it back as ``cursor``);
        an empty ``next_cursor`` means the scroll is exhausted."""
        after_id, _has_after = _vecwire.decode_scroll_cursor(cursor)
        args = _vecwire.encode_scroll_args_order_bounded(collection, limit, filter=filter, after_id=after_id)
        payload = self._call("vector_scroll", args, idempotent=True)
        docs, _degraded, _missing, next_cursor = _vecwire.decode_scroll_result_raw(payload or b"\x00\x00\x00\x00")
        if not next_cursor and limit > 0 and len(docs) == limit:
            # This op's leaf handler returns a plain doc block with no wire
            # cursor on an unpartitioned/single-node server — only a clustered
            # coordinator's fan-out dispatcher supplies one. Derive it
            # client-side in that case: a FULL page may have more, so resume
            # after the last doc's id.
            next_cursor = _vecwire.encode_scroll_cursor(docs[-1]["id"])
        return ScrollPage([_to_document(d) for d in docs], next_cursor=next_cursor)

    def search(self, collection: str, query: Sequence[float], k: int, *,
               filter: Optional[Dict[str, Any]] = None) -> SearchResults:
        """k-nearest-neighbour search. Returns a SearchResults list of
        SearchResult (score defaults to 0.0 — plain kNN has no fusion score);
        .degraded/.missing report whether the read was partial."""
        payload = self._call("vector_search", _vecwire.encode_search_args(collection, k, query, filter),
                             idempotent=True)
        results, degraded, missing = _vecwire.decode_search_results_degraded(payload or b"\x00\x00\x00\x00")
        items = [SearchResult(id=r["id"], distance=r["distance"], score=0.0) for r in results]
        return SearchResults(items, degraded=degraded, missing=missing)

    def search_docs(self, collection: str, query: Sequence[float], k: int, *,
                     filter: Optional[Dict[str, Any]] = None) -> SearchResults:
        """k-nearest-neighbour search returning Document (content + metadata)
        instead of bare id/distance — the RAG-shaped counterpart of search()."""
        args = _vecwire.encode_search_docs_args_opts(collection, k, query, filter)
        payload = self._call("vector_search_docs", args, idempotent=True)
        docs, degraded, missing = _vecwire.decode_docs_degraded_raw(payload or b"\x00\x00\x00\x00")
        return SearchResults([_to_document(d) for d in docs], degraded=degraded, missing=missing)

    def search_groups(self, collection: str, query: Sequence[float], k: int, group_by: str, *,
                       group_size: int = 1, fetch_k: int = 0,
                       filter: Optional[Dict[str, Any]] = None) -> GroupResults:
        """k-nearest-neighbour search grouped by a payload field. Returns a
        GroupResults list of Group(key, hits: List[Document])."""
        opts = {"group_by": group_by, "group_size": group_size, "fetch_k": fetch_k, "filter": filter}
        args = _vecwire.encode_group_search_args_opts(collection, k, query, opts)
        payload = self._call("vector_search_groups", args, idempotent=True)
        groups, degraded, missing = _vecwire.decode_groups_degraded_raw(payload or b"\x00\x00\x00\x00")
        items = [Group(key=g["key"], hits=[_to_document(h) for h in g["hits"]]) for g in groups]
        return GroupResults(items, degraded=degraded, missing=missing)

    def hybrid_search(self, collection: str, dense: Sequence[float], k: int, *,
                       sparse: Optional[Dict[str, Sequence]] = None,
                       filter: Optional[Dict[str, Any]] = None, method: str = "rrf",
                       alpha: float = 0.0, rrf_k: int = 0, dense_k: int = 0,
                       sparse_k: int = 0) -> SearchResults:
        """Fuse a dense-KNN lane with an optional sparse lane. Returns a
        SearchResults list of SearchResult fused by `method`
        ("rrf"/"weighted"/"dbsf")."""
        opts = {"filter": filter, "method": method, "alpha": alpha, "rrf_k": rrf_k,
                "dense_k": dense_k, "sparse_k": sparse_k}
        args = _vecwire.encode_hybrid_search_args_opts(collection, dense, k, sparse, opts)
        payload = self._call("vector_hybrid_search", args, idempotent=True)
        results, degraded, missing = _vecwire.decode_hybrid_results_degraded(payload or b"\x00\x00\x00\x00")
        items = [SearchResult(id=r["id"], distance=r["distance"], score=r["score"]) for r in results]
        return SearchResults(items, degraded=degraded, missing=missing)

    def hybrid_text(self, collection: str, dense: Sequence[float], text: str, k: int, *,
                     filter: Optional[Dict[str, Any]] = None, method: str = "rrf",
                     alpha: float = 0.0, rrf_k: int = 0, dense_k: int = 0,
                     sparse_k: int = 0, global_idf: bool = False) -> SearchResults:
        """Fuse a dense-KNN lane with a server-side BM25 full-text lane (the
        collection must have been created with full_text=... for a full-text
        analyzer to exist).

        global_idf=True opts into the BM25 global-DF (dfs_query_then_fetch)
        two-phase search across partitions (default False => the per-shard-
        local-IDF fast path; single-partition collections ignore it) — same
        knob as HttpTransport.hybrid_text's."""
        opts = {"filter": filter, "method": method, "alpha": alpha, "rrf_k": rrf_k,
                "dense_k": dense_k, "sparse_k": sparse_k}
        args = _vecwire.encode_hybrid_text_args_global(collection, dense, text, k, opts,
                                                        global_idf=global_idf)
        payload = self._call("vector_hybrid_text", args, idempotent=True)
        results, degraded, missing = _vecwire.decode_hybrid_results_degraded(payload or b"\x00\x00\x00\x00")
        items = [SearchResult(id=r["id"], distance=r["distance"], score=r["score"]) for r in results]
        return SearchResults(items, degraded=degraded, missing=missing)

    def recommend(self, collection: str, positive: Sequence[int], *,
                  negative: Optional[Sequence[int]] = None, k: int = 10,
                  filter: Optional[Dict[str, Any]] = None,
                  strategy: str = "average_vector") -> SearchResults:
        """Recommend points similar to the `positive` example ids and
        dissimilar to the `negative` ones. `strategy`: "average_vector"
        (default, average the example vectors then kNN) or "best_score"
        (score by best per-example similarity)."""
        try:
            strat = _vecwire.RECOMMEND_STRATEGY[strategy]
        except KeyError:
            raise ValueError(
                f"unknown recommend strategy {strategy!r}; expected one of "
                f"{sorted(_vecwire.RECOMMEND_STRATEGY)}"
            ) from None
        args = _vecwire.encode_recommend_query(collection, positive=positive, negative=negative,
                                                k=k, filter=filter, strategy=strat)
        payload = self._call("vector_query", args, idempotent=True)
        results, degraded, missing = _vecwire.decode_query_result_degraded(payload or b"\x01\x00\x00\x00\x00")
        items = [SearchResult(id=r["id"], distance=r["distance"], score=r["score"]) for r in results]
        return SearchResults(items, degraded=degraded, missing=missing)
