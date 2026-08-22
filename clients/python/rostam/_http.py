"""HTTP transport backend: Rostam's REST API over the ``-http`` port (default
:8080).

``HttpTransport`` is the migration target for what used to be ``client.py``'s
``RostamClient`` — same connection pool, same request/response plumbing, same
binary-query (RVQ1) and binary-bulk (RVB1) framings — but every shared vector
op now returns one of the unified ``rostam._types`` result types instead of
this module's own dataclasses, so callers get the same shapes regardless of
which transport backend answered (see ``rostam._tcp.TcpTransport``).

Construction (``HttpTransport(base_url, ...)``) does **no** network I/O: the
underlying ``_ConnectionPool`` connects lazily on the first request, exactly
like the pre-unification HTTP client. This matters for tests that build a
client against a target with nothing listening yet.

Method surface:

- Shared with ``TcpTransport`` (unified return types, identical signatures —
  verified by tests/test_transport_gaps.py): create_collection,
  drop_collection, upsert, insert, upsert_batch, delete, get, get_batch,
  scroll, search, search_docs, search_groups, hybrid_search, hybrid_text,
  recommend, exists.
- HTTP-only (guarded by the ``Rostam`` facade, which raises TransportError for
  these on a TCP client): health, search_text, discover, mv_*,
  delete_by_filter, bulk_stage, bulk_build, batch_upsert, and the general
  composable ``query`` (TCP has no ``query`` of its own — a TCP client uses
  ``recommend()`` directly, which is exactly what a recommend-shaped query
  would send).
"""

from __future__ import annotations

import array
import http.client
import json
import struct
import sys
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from ._values import decode_metadata, decode_value, encode_metadata

Vector = Sequence[float]

# Reserved payload key holding a record's document content (server-side rag.go
# contentField). get()/get_batch() lift it into Point.content.
_RESERVED_CONTENT = "$content"


def _seg(s: Union[str, int]) -> str:
    """Percent-encode a single URL path segment (encodes /, ?, #, space, etc.)."""
    return urllib.parse.quote(str(s), safe="")


def _err(message: str, status: int = 0) -> RostamError:
    """Build a ``_types.RostamError`` carrying the HTTP status code (0 for a
    transport-level failure with no response, e.g. a connection error)."""
    return RostamError(message, status=status)


@dataclass
class MultiResult:
    """One hit from a late-interaction (multi-vector / ColBERT MaxSim) search.

    HTTP-only: multivector has no native-TCP counterpart, so this stays a
    local dataclass rather than moving into ``_types``."""
    id: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _sparse(sparse: Optional[Dict[str, Sequence]]) -> Dict[str, Any]:
    if not sparse:
        return {}
    return {"indices": list(sparse["indices"]), "values": list(sparse["values"])}


def _to_document(d: Dict[str, Any]) -> Document:
    return Document(
        id=d["id"],
        distance=d.get("distance", 0.0),
        score=d.get("score", 0.0),
        content=d.get("content", ""),
        metadata=decode_metadata(d.get("metadata")),
    )


# ---- binary bulk framing ("RVB1") ----
#
# The JSON body is the ingest bottleneck for a large initial load: the server has
# to parse `dim` base-10 float literals per point, which dominates the actual
# index build. This dense framing ships the same points as raw f32 instead. It is
# stdlib-only (struct + array) and the server selects it purely by Content-Type,
# so nothing about the JSON API changes.
#
#   magic  b"RVB1"
#   flags  u32   bit0 payloads present, bit1 upsert
#   count  u32
#   dim    u32
#   rows   count x [ id u64 ][ dim x f32 ]
#   pays   count x [ len u32 ][ len bytes of JSON ]   (only when bit0)
#
# All big-endian, matching Rostam's op wire — a staged row is byte-identical to
# the server's internal staging row, so the server reads the body straight into
# the op with no per-float conversion.
_BULK_MAGIC = b"RVB1"
_BULK_FLAG_PAYLOADS = 1 << 0
_BULK_FLAG_UPSERT = 1 << 1


def _encode_bulk_body(
    ids: Sequence[int],
    vectors: Sequence[Vector],
    *,
    flags: int = 0,
    payloads: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> bytearray:
    """Pack (ids, vectors[, payloads]) into an RVB1 binary bulk body.

    Returns the bytearray itself rather than ``bytes(out)``: the copy would
    momentarily double the body, which is the opposite of the point in a function
    whose whole job is to make a large load cheap. http.client accepts any
    bytes-like object as a request body.
    """
    n = len(ids)
    if len(vectors) != n:
        raise ValueError(f"ids/vectors length mismatch: {n} vs {len(vectors)}")
    dim = len(vectors[0]) if n else 0
    if payloads is not None:
        if len(payloads) != n:
            raise ValueError(f"ids/payloads length mismatch: {n} vs {len(payloads)}")
        flags |= _BULK_FLAG_PAYLOADS
    out = bytearray(struct.pack(">4sIII", _BULK_MAGIC, flags, n, dim))
    swap = sys.byteorder == "little"
    for i in range(n):
        row = array.array("f", vectors[i])
        if len(row) != dim:
            raise ValueError(f"vector {i} has dim {len(row)}, expected {dim}")
        out += struct.pack(">Q", ids[i])
        if swap:
            row.byteswap()
        out += row.tobytes()
    if payloads is not None:
        for meta in payloads:
            if not meta:
                out += b"\x00\x00\x00\x00"
                continue
            blob = json.dumps(encode_metadata(meta)).encode("utf-8")
            out += struct.pack(">I", len(blob))
            out += blob
    return out


_RVQ1_MAGIC = b"RVQ1"
_RVQ1_FLAG_FILTER = 1 << 0
# The server refuses a declared dim above this, so a longer vector goes as JSON
# rather than as a request the server is certain to reject.
_RVQ1_MAX_DIM = 1 << 16
# ...and the same for the filter blob. A filter past this is refused by the
# binary route but well within the JSON route's 32 MiB body, so exceeding it has
# to mean "send JSON", not "fail" — an `in_` over enough values gets there.
_RVQ1_MAX_FILTER = 1 << 20


def _encode_rvq1(query: Vector, k: int, filter_blob: bytes = b"") -> bytearray:
    """Encode a search request in the binary query framing.

    Same shape as _encode_bulk on the ingest side, and big-endian for the same
    reason: it lands in the server byte-identical to the op wire, so neither end
    swaps per float. ``array.byteswap`` does the whole vector in one C call —
    which is what makes this 0.011 ms where json.dumps of the same 768 floats is
    0.258 ms.

    read_consistency, on_partition_unavailable and max_staleness are written as
    their defaults: this client has never exposed them, and the framing carries
    them so that adding them later needs no second wire format.
    """
    vec = array.array("f", query)
    flags = _RVQ1_FLAG_FILTER if filter_blob else 0
    out = bytearray(struct.pack(">4sIIIBBHQ", _RVQ1_MAGIC, flags, k, len(vec), 0, 0, 0, 0))
    if sys.byteorder == "little":
        vec.byteswap()
    out += vec.tobytes()
    if filter_blob:
        out += struct.pack(">I", len(filter_blob))
        out += filter_blob
    return out


# The server caps a single binary bulk body at 256 MiB and a single request at
# 262,144 points. bulk_stage/batch_upsert therefore SPLIT a large load into
# requests instead of sending one giant body — otherwise the advertised "load a
# million vectors" use case would just 413 (at dim=768 one request tops out
# around 87k points). The target below leaves generous headroom under both caps.
_BULK_TARGET_BYTES = 64 << 20  # 64 MiB per request
_BULK_MAX_POINTS = 1 << 17     # 131,072 points per request (server ceiling is 2x)


def _points_per_request(dim: int, payload_bytes: int = 0) -> int:
    """How many points fit in one request at this dim, under the server's caps.

    payload_bytes is the per-point payload size to budget for. Ignoring it was a
    bug: batch_upsert with metadata sends 4 + len(json) extra bytes per point, so
    a chunk sized purely on the vector row overshot the server's 256 MiB cap and
    413'd (131,072 points x ~2 KB of metadata is 268 MB of payload alone, before
    a single vector).
    """
    row = 8 + max(dim, 1) * 4 + 4 + max(payload_bytes, 0)
    return max(1, min(_BULK_MAX_POINTS, _BULK_TARGET_BYTES // row))


def _payload_bytes(payloads: Optional[Sequence[Optional[Dict[str, Any]]]]) -> int:
    """Estimate the per-point encoded payload size from a sample of the batch.

    Sampling rather than encoding everything: this runs to decide a chunk size,
    and encoding the whole batch twice would cost more than it saves. The sample
    is scaled up so a batch with uneven payloads still lands under the cap.
    """
    if not payloads:
        return 0
    sample = [p for p in payloads[:64] if p]
    if not sample:
        return 0
    total = sum(len(json.dumps(encode_metadata(p)).encode("utf-8")) for p in sample)
    return (total // len(sample)) * 2  # 2x headroom for uneven payloads


def _chunks(n: int, vectors: Sequence[Vector], payloads=None):
    """Yield (lo, hi) request-sized spans over n points."""
    if n == 0:
        yield 0, 0
        return
    step = _points_per_request(len(vectors[0]), _payload_bytes(payloads))
    for lo in range(0, n, step):
        yield lo, min(lo + step, n)


class _ConnectionPool:
    """Keep-alive connections to one host, safe to share between threads.

    The client used to call ``urllib.request.urlopen`` per request, which opens
    and tears down a TCP connection every time. Reusing one costs a lock and a
    list; measured against a local server it was 1.49x the throughput on
    repeated searches, and the gap widens with network latency because a fresh
    connection pays a round-trip before the request is even sent.

    ``urlopen`` was accidentally thread-safe — nothing was shared. A kept-alive
    connection is not: two threads writing requests into one socket interleave
    and desynchronize the response stream. So connections live here, one thread
    holds one connection for the length of a request, and a connection is only
    returned to the pool once its response has been fully read.
    """

    def __init__(self, base_url: str, maxsize: int = 8):
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"base_url must be http:// or https://, got {base_url!r}")
        self._https = parts.scheme == "https"
        self._host = parts.hostname or "localhost"
        self._port = parts.port or (443 if self._https else 80)
        self._maxsize = maxsize
        self._idle: List[Any] = []
        self._lock = threading.Lock()

    def _new(self, timeout: float):
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout)

    def acquire(self, timeout: float):
        """Return (connection, reused), with `timeout` applied to it.

        The timeout has to be re-applied on every acquire, not just at connect.
        http.client fixes the socket timeout when the connection is made, so a
        pooled connection carries whatever timeout its first caller happened to
        want — and bulk_build asks for 24 hours. Inheriting an earlier 30-second
        connection would abort a long build at 30 seconds, which urllib never
        did because it applied the timeout per request.
        """
        with self._lock:
            if self._idle:
                conn = self._idle.pop()
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                return conn, True
        return self._new(timeout), False

    def release(self, conn) -> None:
        with self._lock:
            if len(self._idle) < self._maxsize:
                self._idle.append(conn)
                return
        conn.close()

    def discard(self, conn) -> None:
        try:
            conn.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for c in idle:
            self.discard(c)


class HttpTransport:
    """Vector-database operations over Rostam's REST API (see
    ``rostam.NewHTTPServer``).

    Shares one keep-alive connection pool and bearer token across every call.
    Search-family reads (search, search_docs, search_groups, hybrid_search,
    hybrid_text, recommend, query) surface the response's degraded/missing
    trailer via SearchResults/GroupResults, matching TcpTransport; scroll()
    carries next_cursor via ScrollPage. get()/get_batch() return Point(s).
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 30.0,
        *,
        binary_search: bool = True,
        pool_maxsize: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # Whether to send search queries in the binary framing. Turning it off
        # forces the JSON body; see _search for what the framing buys and for
        # how an older server is detected and fallen back to automatically.
        self.binary_search = binary_search
        self._binary_search_supported = True
        # No network I/O here: _ConnectionPool connects lazily on first acquire().
        self._pool = _ConnectionPool(self.base_url, maxsize=pool_maxsize)
        self._path_prefix = urllib.parse.urlsplit(self.base_url).path.rstrip("/")

    def close(self) -> None:
        """Close pooled connections. The client stays usable; it reconnects."""
        self._pool.close()

    def __enter__(self) -> "HttpTransport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- transport ----

    def _request(
        self, method: str, path: str, body: Optional[dict] = None, *,
        idempotent: Optional[bool] = None,
    ) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        if idempotent is None:
            idempotent = method in ("GET", "HEAD")
        return self._send(method, path, data, "application/json", idempotent=idempotent)

    def _send(
        self,
        method: str,
        path: str,
        data: Optional[Union[bytes, bytearray]],
        content_type: str,
        timeout: Optional[float] = None,
        *,
        idempotent: bool = False,
    ) -> Any:
        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        deadline = timeout or self.timeout
        url = self._path_prefix + path

        # One retry, and only for a request that is safe to send twice on a
        # connection taken from the pool.
        #
        # A server is free to close an idle keep-alive connection at any moment,
        # so a pooled connection can be dead before we use it. But the failure
        # does not reliably arrive while writing: RemoteDisconnected and
        # BadStatusLine are raised by getresponse(), by which point the request
        # bytes are on the wire and the server may already have executed them.
        # Retrying then would replay the write — a second /points insert that
        # fails on a duplicate id, or a double batch. So the caller says whether
        # the request can be repeated; reads say yes, writes say nothing and get
        # an error they can act on.
        attempts = 2 if idempotent else 1
        while True:
            attempts -= 1
            conn, reused = self._pool.acquire(deadline)
            try:
                conn.request(method, url, body=data, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                status = resp.status
            except (http.client.RemoteDisconnected, http.client.BadStatusLine,
                    ConnectionResetError, BrokenPipeError) as e:
                self._pool.discard(conn)
                if reused and attempts > 0:
                    continue
                raise _err(f"transport error: {e}", status=0) from None
            except (OSError, http.client.HTTPException) as e:
                self._pool.discard(conn)
                raise _err(f"transport error: {e}", status=0) from None

            # Only a connection the server intends to keep goes back in the pool.
            # A response that closes it (HTTP/1.0, or an explicit Connection:
            # close) leaves a dead socket that the next caller would spend a
            # failed write and a retry to discover — turning pooling into a
            # per-request penalty against exactly the servers that opted out.
            if resp.will_close:
                self._pool.discard(conn)
            else:
                self._pool.release(conn)
            if status >= 400:
                msg = f"HTTP {status}"
                try:
                    msg = json.loads(raw).get("error", msg)
                except Exception:
                    pass
                raise _err(msg, status=status)
            if not raw:
                return None
            return json.loads(raw)

    # ---- search encoding ----

    def _search(
        self, path: str, query: Vector, k: int, filter: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """POST a search, in the binary framing when the server understands it.

        The JSON body spends its time turning float32s into decimal for the
        server to parse straight back. At dim=768, k=10, that encode is 0.258 ms
        of a 0.845 ms request — 31% — against 0.011 ms to write the same vector
        as bytes, and the server's matching decode disappears with it.
        """
        # A k the framing cannot express (negative, or past u32) goes down the
        # JSON path, where the server answers 400 exactly as it always has.
        # Encoding it here would raise struct.error instead — a different
        # exception type for the same misuse, decided by which encoding the
        # client happened to pick.
        # Encoded once here rather than inside _encode_rvq1, because its SIZE is
        # part of deciding whether the binary path can carry this request at all.
        blob = json.dumps(filter).encode("utf-8") if filter else b""
        encodable = (0 <= k <= 0xFFFFFFFF and len(query) <= _RVQ1_MAX_DIM
                     and len(blob) <= _RVQ1_MAX_FILTER)
        if self.binary_search and self._binary_search_supported and encodable:
            try:
                res = self._send(
                    "POST", path, _encode_rvq1(query, k, blob),
                    "application/octet-stream", idempotent=True,
                )
                return res or {}
            except RostamError as e:
                # A server without RVQ1 support routes the body to its JSON
                # decoder, which chokes on byte one and says so. That specific
                # message is the signal to stop offering binary for the life of
                # this client and use JSON — anything else is a real error about
                # this request (a bad k, a bad filter) and must surface as one.
                if not (getattr(e, "status", None) == 400 and "invalid JSON body" in str(e)):
                    raise
                self._binary_search_supported = False

        body: Dict[str, Any] = {"query": list(query), "k": k}
        if filter:
            body["filter"] = filter
        return self._request("POST", path, body, idempotent=True) or {}

    # ---- collections ----

    def health(self) -> bool:
        """Return True if the server is reachable and healthy. HTTP-only."""
        return (self._request("GET", "/v1/health") or {}).get("status") == "ok"

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
        """Create a collection. metric: "cosine"|"l2"|"dot"; quant:
        ""|"sq8"|"bq1"|"pq"|"sq"|"prq".

        The keyword surface is identical to TcpTransport.create_collection's
        (the unification promise — signatures match byte for byte via
        inspect.signature): index_type/ivf_*/opq*/pq_drop_vecs/
        ivf_train_threshold/ivf_drift_* tune an "ivf" index; extend_candidates*/
        level0_full_degree/quantized_build are HNSW build levers; partitions
        sets the collection-level partition count. Each is sent to the server
        only when non-default, so a plain create stays byte-compatible with a
        pre-unification request.

        quant="sq" is the trained metric-agnostic scalar quantizer; sq_bits picks
        its bit-depth (4, 6, or 8; 0 = server default 8). quant="prq" is
        product-residual quantization; prq_layers is the residual layer count
        (0 = server default 2). Both numeric knobs are sent only when non-zero,
        so a non-SQ/PRQ create stays byte-compatible with the prior request.

        index_type selects the backing index: ""/"hnsw" (default), "ivf", or
        "vamana" (the DiskANN single-layer graph). For "vamana", vamana_r is the
        max out-degree (0 = server default 64), vamana_l the build/search beam
        width (0 = default 100), vamana_alpha the pass-2 RobustPrune α (0 = default
        1.2). index_type and the vamana knobs are sent only when set, so a non-
        Vamana create stays byte-compatible with the prior request.

        anisotropic_eta is the ScaNN score-aware PQ weight (η ≥ 1; 0/1 = isotropic,
        byte-compatible default). soar opts an "ivf" index into ScaNN-style multi-
        assignment (higher recall at fixed nprobe); soar_lambda tunes the SOAR
        orthogonality weight λ (0 = server default 1.5). pq_nbits is the per-subspace
        PQ code width (0/8 = 8-bit default, 4 = 4-bit LUT16 fast-scan). Each is sent
        only when set, so a non-ScaNN create stays byte-compatible with the prior
        request.

        full_text enables the server-side BM25 full-text lane (so search_text /
        hybrid_text work): pass True for the default English analyzer, or a dict
        like {"analyzer": "english", "k1": 1.2, "b": 0.75} to tune the BM25 knobs.
        None (default) leaves full-text disabled."""
        cfg = {
            "dim": dim,
            "metric": metric,
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "seed": seed,
            "quant": quant,
            "persistent": persistent,
            "rescore_factor": rescore_factor,
        }
        if extend_candidates:
            cfg["extend_candidates"] = extend_candidates
        if extend_candidates_max:
            cfg["extend_candidates_max"] = extend_candidates_max
        if level0_full_degree:
            cfg["level0_full_degree"] = level0_full_degree
        if quantized_build:
            cfg["quantized_build"] = quantized_build
        if partitions:
            cfg["partitions"] = partitions
        if index_type:
            cfg["index_type"] = index_type
        if ivf_nlist:
            cfg["ivf_nlist"] = ivf_nlist
        if ivf_nprobe:
            cfg["ivf_nprobe"] = ivf_nprobe
        if ivf_pq:
            cfg["ivf_pq"] = ivf_pq
        if ivf_pq_m:
            cfg["ivf_pq_m"] = ivf_pq_m
        if ivf_rerank:
            cfg["ivf_rerank"] = ivf_rerank
        if quant_pq_m:
            cfg["quant_pq_m"] = quant_pq_m
        if opq:
            cfg["opq"] = opq
        if pq_drop_vecs:
            cfg["pq_drop_vecs"] = pq_drop_vecs
        if ivf_train_threshold:
            cfg["ivf_train_threshold"] = ivf_train_threshold
        if ivf_drift_retrain:
            cfg["ivf_drift_retrain"] = ivf_drift_retrain
        if ivf_drift_growth_factor:
            cfg["ivf_drift_growth_factor"] = ivf_drift_growth_factor
        if ivf_drift_factor:
            cfg["ivf_drift_factor"] = ivf_drift_factor
        if filter_first_relative_bp:
            cfg["filter_first_relative_bp"] = filter_first_relative_bp
        if opq_iters:
            cfg["opq_iters"] = opq_iters
        if sq_bits:
            cfg["sq_bits"] = sq_bits
        if prq_layers:
            cfg["prq_layers"] = prq_layers
        if vamana_r:
            cfg["vamana_r"] = vamana_r
        if vamana_l:
            cfg["vamana_l"] = vamana_l
        if vamana_alpha:
            cfg["vamana_alpha"] = vamana_alpha
        if anisotropic_eta:
            cfg["anisotropic_eta"] = anisotropic_eta
        if soar:
            cfg["soar"] = soar
        if soar_lambda:
            cfg["soar_lambda"] = soar_lambda
        if pq_nbits:
            cfg["pq_nbits"] = pq_nbits
        if full_text is True:
            cfg["full_text"] = {}
        elif isinstance(full_text, dict):
            cfg["full_text"] = full_text
        self._request("POST", "/v1/collections", {"name": name, "config": cfg})

    def drop_collection(self, name: str) -> None:
        """Delete a collection and its data."""
        self._request("DELETE", "/v1/collections/" + _seg(name))

    # ---- points ----

    def upsert(
        self,
        collection: str,
        id: int,
        vector: Vector,
        *,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_ms: int = 0,
        sparse: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        """Insert or replace a point (the RAG write path; stores content)."""
        self._put_point(collection, id, vector, content, metadata, ttl_ms, sparse, upsert=True)

    def insert(
        self,
        collection: str,
        id: int,
        vector: Vector,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_ms: int = 0,
        sparse: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        """Insert a point, rejecting a duplicate id (use upsert to replace).

        No `content` kwarg (unlike upsert): content is a RAG/upsert concept —
        the engine's Insert rejects Content, and this mirrors that so the
        signature is identical to TcpTransport.insert's."""
        self._put_point(collection, id, vector, "", metadata, ttl_ms, sparse, upsert=False)

    def _put_point(self, collection, id, vector, content, metadata, ttl_ms, sparse, upsert):
        body = {
            "id": id,
            "vector": list(vector),
            "content": content,
            "ttl_ms": ttl_ms,
            "metadata": encode_metadata(metadata),
            "upsert": upsert,
        }
        sp = _sparse(sparse)
        if sp:
            body["sparse"] = sp
        self._request("POST", f"/v1/collections/{_seg(collection)}/points", body)

    def upsert_batch(self, collection: str, points: Sequence[Dict[str, Any]]) -> None:
        """Upsert many points in one request, over the JSON /points/batch route.

        Mirrors TcpTransport.upsert_batch's shape (one dict per point: {id,
        vector, content="", ttl_ms=0, metadata=None, sparse=None}), always as
        an unconditional upsert — matching the TCP backend, which sends every
        point as vector_upsert. For a large INITIAL load prefer bulk_stage() +
        bulk_build() (HTTP-only, ~6x faster to searchable); this is the
        cross-transport, indexed-inline equivalent.
        """
        body_points = []
        for p in points:
            row: Dict[str, Any] = {
                "id": p["id"],
                "vector": list(p["vector"]),
                "content": p.get("content", ""),
                "ttl_ms": p.get("ttl_ms", 0),
                "metadata": encode_metadata(p.get("metadata")),
                "upsert": True,
            }
            sp = _sparse(p.get("sparse"))
            if sp:
                row["sparse"] = sp
            body_points.append(row)
        self._request(
            "POST", f"/v1/collections/{_seg(collection)}/points/batch",
            {"upsert": True, "points": body_points},
        )

    # ---- bulk load (binary wire; HTTP-only) ----

    def bulk_stage(
        self,
        collection: str,
        ids: Sequence[int],
        vectors: Sequence[Vector],
        *,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """Stage points for a concurrent bulk build, over the binary wire.

        The initial-load fast path: staging is cheap and parallel, and the
        multi-core index build happens once in :meth:`bulk_build`. The collection
        must be empty.

        ``metadatas`` is optional and, when given, must have one entry per id
        (``None`` for a point with no payload). The payloads are applied by the
        build itself, so a load whose points need metadata to filter on gets the
        multi-core build too — measured ~6x faster to searchable than indexing the
        same corpus inline via :meth:`upsert_batch`. Prefer this method for an
        initial load even when the points carry payloads; :meth:`upsert_batch` is
        for writes into a collection that is already built, or for points that
        need content, sparse vectors, TTLs or a CAS precondition, none of which
        the staging wire carries.

        The load is SPLIT across requests to stay under the server's per-request
        caps (256 MiB, 262,144 points), so passing a million vectors in one call
        works. Returns the number of points staged. HTTP-only.
        """
        path = f"/v1/collections/{_seg(collection)}/points/bulk"
        staged = 0
        for lo, hi in _chunks(len(ids), vectors, metadatas):
            res = self._send(
                "POST",
                path,
                _encode_bulk_body(
                    ids[lo:hi], vectors[lo:hi],
                    payloads=None if metadatas is None else metadatas[lo:hi],
                ),
                "application/octet-stream",
                timeout=timeout,
            )
            staged += int((res or {}).get("staged", 0))
        return staged

    def bulk_build(self, collection: str, *, workers: int = 0, timeout: float = 24 * 3600) -> None:
        """Build everything staged by :meth:`bulk_stage` into the index in one pass.

        Blocks until the build finishes (minutes on a large corpus), so the
        default timeout is deliberately generous. workers=0 uses every core.
        HTTP-only.
        """
        self._send(
            "POST",
            f"/v1/collections/{_seg(collection)}/points/bulk/build",
            json.dumps({"workers": workers}).encode("utf-8"),
            "application/json",
            timeout=timeout,
        )

    def batch_upsert(
        self,
        collection: str,
        ids: Sequence[int],
        vectors: Sequence[Vector],
        *,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        upsert: bool = True,
        timeout: Optional[float] = None,
    ) -> int:
        """Insert/upsert many points in one request over the binary wire.

        Each point is indexed INLINE (no separate build step), which is what makes
        this the path for writing into a collection that is already built. For an
        INITIAL load prefer :meth:`bulk_stage`, which now carries metadata too and
        gets the multi-core build — measured ~6x faster to searchable on a
        payload-bearing 1M x 768d corpus. Split across requests under the server's
        per-request caps, like :meth:`bulk_stage`. Returns the number of points
        written. HTTP-only (parallel-array shape); see :meth:`upsert_batch` for
        the cross-transport, one-dict-per-point shape.
        """
        flags = _BULK_FLAG_UPSERT if upsert else 0
        path = f"/v1/collections/{_seg(collection)}/points/batch"
        written = 0
        for lo, hi in _chunks(len(ids), vectors, metadatas):
            res = self._send(
                "POST",
                path,
                _encode_bulk_body(
                    ids[lo:hi], vectors[lo:hi], flags=flags,
                    payloads=None if metadatas is None else metadatas[lo:hi],
                ),
                "application/octet-stream",
                timeout=timeout,
            )
            written += int((res or {}).get("count", 0))
        return written

    def delete(self, collection: str, id: int) -> bool:
        """Delete a point by id; returns whether it existed."""
        res = self._request("DELETE", f"/v1/collections/{_seg(collection)}/points/{_seg(id)}")
        return bool((res or {}).get("deleted"))

    def delete_by_filter(self, collection: str, filter: Dict[str, Any]) -> int:
        """Delete every point matching filter; returns the count removed. HTTP-only."""
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/delete", {"filter": filter})
        return int((res or {}).get("deleted", 0))

    def exists(self, collection: str, id: int) -> bool:
        """Whether a point id is live. No projection is requested (id-only check)."""
        try:
            self._request(
                "GET",
                f"/v1/collections/{_seg(collection)}/points/{_seg(id)}"
                "?with_vector=false&with_payload=false",
            )
            return True
        except RostamError as e:
            if getattr(e, "status", None) == 404:
                return False
            raise

    def get(
        self, collection: str, id: int, *, with_vector: bool = True, with_payload: bool = True,
    ) -> Optional[Point]:
        """Fetch a point by id, or None if absent."""
        q = (
            f"?with_vector={'true' if with_vector else 'false'}"
            f"&with_payload={'true' if with_payload else 'false'}"
        )
        try:
            res = self._request("GET", f"/v1/collections/{_seg(collection)}/points/{_seg(id)}{q}") or {}
        except RostamError as e:
            if getattr(e, "status", None) == 404:
                return None
            raise
        meta = decode_metadata(res.get("payload")) if with_payload else {}
        content = ""
        if isinstance(meta, dict) and _RESERVED_CONTENT in meta:
            cv = meta.pop(_RESERVED_CONTENT)
            content = cv if isinstance(cv, str) else ""
        vec = res.get("vector")
        return Point(id=int(id), vector=list(vec) if vec else None, content=content, metadata=meta)

    def scroll(
        self,
        collection: str,
        *,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 0,
        cursor: str = "",
    ) -> ScrollPage:
        """List live documents (content + metadata) matching filter (None = all),
        in deterministic id-ASCENDING order, up to limit (0 = no cap).

        Returns a ScrollPage: iterate/len it like a list of documents, and read
        its ``next_cursor`` to paginate. A non-empty ``next_cursor`` is the
        resume token for the following page (ids strictly greater than the last
        returned) — pass it back as ``cursor=`` to fetch it; an empty
        ``next_cursor`` means the listing is exhausted. Leaving ``cursor`` empty
        fetches the first page.
        """
        body: Dict[str, Any] = {"limit": limit}
        if filter:
            body["filter"] = filter
        if cursor:
            body["cursor"] = cursor
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/scroll", body) or {}
        docs = [_to_document(d) for d in (res.get("documents") or [])]
        return ScrollPage(docs, next_cursor=res.get("next_cursor") or "")

    def get_batch(
        self,
        collection: str,
        ids: Sequence[int],
        *,
        with_vector: bool = True,
        with_payload: bool = True,
    ) -> List[Point]:
        """Fetch points by id in one request. Returns one Point per PRESENT id
        (absent ids are omitted; never raises on partial miss). Content is lifted
        from the reserved payload field into Point.content and removed from
        Point.metadata."""
        body = {
            "ids": [int(i) for i in ids],
            "with_vector": with_vector,
            "with_payload": with_payload,
        }
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/batch-get", body) or {}
        out: List[Point] = []
        for row in (res.get("points") or []):
            meta = decode_metadata(row.get("payload")) if with_payload else {}
            content = ""
            if isinstance(meta, dict) and _RESERVED_CONTENT in meta:
                cv = meta.pop(_RESERVED_CONTENT)
                content = cv if isinstance(cv, str) else ""
            rvec = row.get("vector")
            out.append(Point(
                id=row["id"],
                vector=list(rvec) if rvec else None,
                content=content,
                metadata=meta,
            ))
        return out

    # ---- search ----

    def search(
        self, collection: str, query: Vector, k: int, *, filter: Optional[Dict[str, Any]] = None
    ) -> SearchResults:
        """k-nearest-neighbor search, returning ids + distances."""
        res = self._search(f"/v1/collections/{_seg(collection)}/points/search", query, k, filter)
        items = [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                 for r in (res.get("results") or [])]
        return SearchResults(items, degraded=res.get("degraded", False), missing=res.get("missing") or [])

    def search_docs(
        self, collection: str, query: Vector, k: int, *, filter: Optional[Dict[str, Any]] = None
    ) -> SearchResults:
        """kNN search returning each hit enriched with content + metadata."""
        res = self._search(f"/v1/collections/{_seg(collection)}/points/search/docs", query, k, filter)
        items = [_to_document(d) for d in (res.get("documents") or [])]
        return SearchResults(items, degraded=res.get("degraded", False), missing=res.get("missing") or [])

    def search_groups(
        self,
        collection: str,
        query: Vector,
        k: int,
        group_by: str,
        *,
        group_size: int = 1,
        fetch_k: int = 0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> GroupResults:
        """Group-by-document search: top-k distinct documents, best chunk(s) each."""
        body = {"query": list(query), "k": k, "group_by": group_by, "group_size": group_size, "fetch_k": fetch_k}
        if filter:
            body["filter"] = filter
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/groups", body) or {}
        groups = []
        for g in (res.get("groups") or []):
            key = decode_value(g["key"]) if isinstance(g.get("key"), dict) else g.get("key")
            groups.append(Group(key=key, hits=[_to_document(d) for d in (g.get("hits") or [])]))
        return GroupResults(groups, degraded=res.get("degraded", False), missing=res.get("missing") or [])

    def query(
        self,
        collection: str,
        prefetch: Sequence[Dict[str, Any]],
        *,
        root: Optional[Dict[str, Any]] = None,
        mode: str = "fusion",
        method: str = "",
        alpha: float = 0.0,
        rrf_k: int = 0,
        k: int = 10,
        group_by: str = "",
        group_size: int = 0,
    ) -> Any:
        """The composable Query API: run one or more ``prefetch`` lanes and fuse or
        rerank them in a single request. This is the only entry point that carries
        recommend and discover over the wire (there are no standalone routes), and
        the ``recommend()`` / ``discover()`` helpers below are thin wrappers over it.

        Each ``prefetch`` entry is a leaf dict — a dense vector, a sparse lane, a
        recommend spec, or a discover spec — optionally with its own ``k`` and a
        ``filter``::

            [{"dense": [0.1, 0.2, ...], "k": 50}]
            [{"sparse": {"indices": [...], "values": [...]}}]
            [{"recommend": {"positive": [12, 96], "negative": [40]}, "k": 50}]
            [{"discover": {"target": 7, "context": [{"positive": 1, "negative": 2}]}}]

        ``mode`` is "fusion" (combine the lanes, default) or "rerank" (``root``
        re-scores the union of the prefetch candidates — pass a ``root`` leaf for
        that). ``method`` picks the fusion: "rrf" | "weighted" | "dbsf".

        Returns ``SearchResults``; when ``group_by`` is set the response is
        grouped and this returns ``GroupResults`` instead, with ``k`` counting
        groups. HTTP-only: TCP has no general QuerySpec builder, only a
        recommend-shaped call (``recommend()``); ``recommend()``/``discover()``
        are the cross-transport-shaped entry points.
        """
        if not prefetch:
            raise ValueError("query requires at least one prefetch leaf")
        body: Dict[str, Any] = {
            "prefetch": list(prefetch),
            "mode": mode, "method": method, "alpha": alpha, "rrf_k": rrf_k, "k": k,
        }
        if root is not None:
            body["root"] = root
        if group_by:
            body["group_by"] = group_by
            body["group_size"] = group_size or 1
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/query", body) or {}
        degraded, missing = res.get("degraded", False), res.get("missing") or []
        if group_by:
            groups = []
            for g in (res.get("groups") or []):
                key = decode_value(g["key"]) if isinstance(g.get("key"), dict) else g.get("key")
                groups.append(Group(key=key, hits=[_to_document(d) for d in (g.get("hits") or [])]))
            return GroupResults(groups, degraded=degraded, missing=missing)
        items = [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                 for r in (res.get("results") or [])]
        return SearchResults(items, degraded=degraded, missing=missing)

    def recommend(
        self,
        collection: str,
        positive: Sequence[int],
        *,
        negative: Optional[Sequence[int]] = None,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        strategy: str = "average_vector",
    ) -> SearchResults:
        """Recommend by example: score toward the ``positive`` ids and away from the
        ``negative`` ids, instead of from a raw query vector. ``strategy`` is
        "average_vector" (default, mean(pos) - mean(neg) → one dense query) or
        "best_score". Signature is identical to TcpTransport.recommend's.

        Carried over the wire by the Query API — there is no ``points/recommend``
        route — so this works identically on HTTP, gRPC and the binary TCP protocol.
        """
        rec: Dict[str, Any] = {"positive": [int(i) for i in positive], "strategy": strategy}
        if negative:
            rec["negative"] = [int(i) for i in negative]
        leaf: Dict[str, Any] = {"recommend": rec, "k": k}
        if filter:
            leaf["filter"] = filter
        return self.query(collection, [leaf], k=k)

    def discover(
        self,
        collection: str,
        context: Sequence[Tuple[int, int]],
        k: int,
        *,
        target: Optional[int] = None,
        target_vec: Optional[Vector] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResults:
        """Guided "more like this, away from that" search. ``context`` is a list of
        ``(positive_id, negative_id)`` pairs; ``target`` (a point id) or
        ``target_vec`` (a raw vector) is an optional anchor to explore around.

        Like ``recommend()``, this rides the Query API and so is available on every
        transport. HTTP-only surface (no ``discover`` on TcpTransport today).
        """
        pairs = [{"positive": int(p), "negative": int(n)} for (p, n) in context]
        disc: Dict[str, Any] = {"context": pairs}
        if target is not None:
            disc["target"] = int(target)
        if target_vec is not None:
            disc["target_vec"] = list(target_vec)
        leaf: Dict[str, Any] = {"discover": disc, "k": k}
        if filter:
            leaf["filter"] = filter
        return self.query(collection, [leaf], k=k)

    def hybrid_search(
        self,
        collection: str,
        dense: Vector,
        k: int,
        *,
        sparse: Optional[Dict[str, Sequence]] = None,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "rrf",
        alpha: float = 0.0,
        rrf_k: int = 0,
        dense_k: int = 0,
        sparse_k: int = 0,
    ) -> SearchResults:
        """Fused dense + sparse search. method: "rrf"|"weighted"."""
        body: Dict[str, Any] = {
            "dense": list(dense), "k": k, "method": method, "alpha": alpha,
            "rrf_k": rrf_k, "dense_k": dense_k, "sparse_k": sparse_k,
        }
        sp = _sparse(sparse)
        if sp:
            body["sparse"] = sp
        if filter:
            body["filter"] = filter
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/hybrid", body) or {}
        items = [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                 for r in (res.get("results") or [])]
        return SearchResults(items, degraded=res.get("degraded", False), missing=res.get("missing") or [])

    def search_text(
        self, collection: str, text: str, k: int, *,
        filter: Optional[Dict[str, Any]] = None, global_idf: bool = False,
    ) -> SearchResults:
        """BM25 full-text search. The RAW query text is sent to the server, which
        tokenizes + scores it (the SDK ships no tokens). Returns each hit enriched
        with content + metadata. Requires a collection created with full_text=True.
        HTTP-only (no equivalent op on TcpTransport today).

        global_idf=True opts into the BM25 global-DF (dfs_query_then_fetch) two-phase
        search across partitions (default False ⇒ the per-shard-local-IDF fast path;
        single-partition collections ignore it)."""
        body: Dict[str, Any] = {"text": text, "k": k}
        if filter:
            body["filter"] = filter
        if global_idf:
            body["global_idf"] = True
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/text", body) or {}
        items = [_to_document(d) for d in (res.get("documents") or [])]
        return SearchResults(items, degraded=res.get("degraded", False), missing=res.get("missing") or [])

    def hybrid_text(
        self,
        collection: str,
        dense: Vector,
        text: str,
        k: int,
        *,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "rrf",
        alpha: float = 0.0,
        rrf_k: int = 0,
        dense_k: int = 0,
        sparse_k: int = 0,
        global_idf: bool = False,
    ) -> SearchResults:
        """Fuse a dense query vector plus the RAW query text; the server analyzes
        the text into the BM25 lane and fuses it with the dense lane. method:
        "rrf"|"weighted"|"dbsf". Requires a collection created with
        full_text=True. Parameter name/order is identical to
        TcpTransport.hybrid_text's (the `dense` param serializes to the wire's
        "vector" field either way).

        global_idf=True opts into the BM25 global-DF (dfs_query_then_fetch) two-phase
        text lane across partitions (default False ⇒ the per-shard-local-IDF fast
        path; affects only the BM25 text lane)."""
        body: Dict[str, Any] = {
            "vector": list(dense), "text": text, "k": k, "method": method, "alpha": alpha,
            "rrf_k": rrf_k, "dense_k": dense_k, "sparse_k": sparse_k,
        }
        if filter:
            body["filter"] = filter
        if global_idf:
            body["global_idf"] = True
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/hybrid-text", body) or {}
        items = [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                 for r in (res.get("results") or [])]
        return SearchResults(items, degraded=res.get("degraded", False), missing=res.get("missing") or [])

    # ---- late interaction (multi-vector / ColBERT MaxSim; HTTP-only) ----

    def mv_create_collection(
        self,
        name: str,
        dim: int,
        *,
        m: int = 0,
        ef_construction: int = 0,
        ef_search: int = 0,
        seed: int = 0,
        quant: str = "",
        rescore_factor: int = 0,
        persistent: bool = False,
    ) -> None:
        """Create a late-interaction collection (token vectors + MaxSim).

        quant ("sq8"|"bq1") quantizes the first-stage graph; with persistent=True
        the float32 token vectors move off-heap into an mmap file and the
        collection is durable across restart.
        """
        body = {
            "dim": dim, "m": m, "ef_construction": ef_construction, "ef_search": ef_search,
            "seed": seed, "quant": quant, "rescore_factor": rescore_factor, "persistent": persistent,
        }
        self._request("POST", "/v1/multivector/" + _seg(name), body)

    def mv_drop_collection(self, name: str) -> None:
        """Delete a late-interaction collection."""
        self._request("DELETE", "/v1/multivector/" + _seg(name))

    def mv_add(
        self,
        name: str,
        doc_id: int,
        tokens: Sequence[Vector],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or replace a document represented by its token vectors."""
        body = {"id": doc_id, "tokens": [list(t) for t in tokens], "metadata": encode_metadata(metadata)}
        self._request("POST", f"/v1/multivector/{_seg(name)}/docs", body)

    def mv_search(
        self,
        name: str,
        query: Sequence[Vector],
        k: int,
        *,
        candidates_per_token: int = 0,
    ) -> List[MultiResult]:
        """MaxSim late-interaction search: top-k documents for the multi-vector query."""
        body = {"query": [list(t) for t in query], "k": k, "candidates_per_token": candidates_per_token}
        res = self._request("POST", f"/v1/multivector/{_seg(name)}/search", body) or {}
        return [MultiResult(id=r["id"], score=r.get("score", 0.0), metadata=decode_metadata(r.get("metadata")))
                for r in (res.get("results") or [])]

    def mv_delete(self, name: str, doc_id: int) -> bool:
        """Delete a document from a late-interaction collection."""
        res = self._request("DELETE", f"/v1/multivector/{_seg(name)}/docs/{_seg(doc_id)}")
        return bool((res or {}).get("deleted"))
