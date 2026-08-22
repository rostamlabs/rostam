"""Binary encoders for the vector ops carried on the native TCP protocol.

These mirror Rostam's Go `ops.Encode*Args` byte layouts exactly. The fixed
binary parts (config trailer, ids, dims, vectors, sparse) are asserted
byte-for-byte against the Go reference in tests/test_vecwire_golden.py; the
embedded JSON parts (metadata, filter, content) are validated by round-tripping
through a real server, since the server unmarshals them and any equivalent JSON
decodes the same.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._values import decode_metadata, decode_value, encode_metadata

Vector = Sequence[float]

# --- enum maps (must match vector/index.go, vector/quant.go) -----------------
_METRIC = {"cosine": 0, "l2": 1, "dot": 2, "dotproduct": 2, "ip": 2}
_QUANT = {"": 0, "none": 0, "sq8": 1, "bq1": 2, "pq": 3, "sq": 4, "prq": 5}
_INDEX = {"": 0, "hnsw": 0, "ivf": 1, "vamana": 2, "gpu": 3}

# insert/search flag bits (vector.go)
_F_TTL = 1 << 0
_F_META = 1 << 1
_F_SPARSE = 1 << 2
_F_FILTER = 1 << 0  # search flags use bit0 for filter
_F_SEARCH_OPTS = 1 << 1  # search flags: consistency opts trailer present (vecFlagSearchOpts)

# hybrid_search flag bits (vector.go: hybridFlagFilter/hybridFlagSparse/hybridFlagOpts)
_HYBRID_F_FILTER = 1 << 0
_HYBRID_F_SPARSE = 1 << 1
_HYBRID_F_OPTS = 1 << 2

# hybrid_text flag bits (text.go: textFlagFilter/textFlagOpts/textFlagGlobalIDF).
# textFlagGlobalStats (1 << 3) is deliberately not defined here: it tags the
# coordinator-only phase-1 global-DF stats block, which this client never
# constructs (see encode_hybrid_text_args_global).
_TEXT_F_FILTER = 1 << 0
_TEXT_F_OPTS = 1 << 1
_TEXT_F_GLOBAL_IDF = 1 << 2

# fusion method (vector/fusion.go: FusionMethod)
_FUSION_METHOD = {"rrf": 0, "weighted": 1, "dbsf": 2}

# order_by kind (vector/order.go: OrderKind) — only "string" changes the wire shape;
# numeric/datetime share the float64 path and are distinguished by the is_datetime bit.
_ORDER_KIND = {"numeric": 0, "datetime": 1, "string": 2}

# read-consistency levels (ops/consistency.go)
CONSISTENCY_ANY_REPLICA = 0
CONSISTENCY_LEADER_ONLY = 1
CONSISTENCY_LINEARIZABLE = 2
CONSISTENCY_BOUNDED_STALENESS = 3


def _col(collection: str) -> bytes:
    c = collection.encode("utf-8")
    if len(c) > 0xFF:
        raise ValueError("collection name too long")
    return bytes([len(c)]) + c


def _f32be(vec: Vector) -> bytes:
    out = bytearray(struct.pack(">I", len(vec)))
    for f in vec:
        out += struct.pack(">f", f)
    return bytes(out)


def _meta_json(metadata: Optional[Dict[str, Any]]) -> bytes:
    # encode_metadata builds the tagged form; the Go Value marshals its struct
    # fields in declaration order (kind first), so the inner dicts must NOT be
    # key-sorted. The server unmarshals this, so exact byte order is not required
    # for correctness — this only aims to look like the Go output.
    tagged = encode_metadata(metadata or {})
    return json.dumps(tagged, separators=(",", ":")).encode("utf-8")


def encode_search_args(collection: str, k: int, query: Vector,
                       filter: Optional[Dict[str, Any]] = None) -> bytes:
    flags = _F_FILTER if filter else 0
    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">I", k)
    out += _f32be(query)
    if filter:
        fj = json.dumps(filter, separators=(",", ":")).encode("utf-8")
        out += struct.pack(">I", len(fj)) + fj
    return bytes(out)


def _bound_tail(read_consistency: int, bound: int) -> bytes:
    """Mirrors ops.appendBoundTail: the 8-byte BE staleness bound rides ONLY when
    read_consistency == CONSISTENCY_BOUNDED_STALENESS (3); every other level is
    byte-identical to the pre-bounded-staleness wire (no tail bytes at all)."""
    if read_consistency != CONSISTENCY_BOUNDED_STALENESS:
        return b""
    return struct.pack(">Q", bound)


def encode_search_args_opts(collection: str, k: int, query: Vector,
                            filter: Optional[Dict[str, Any]] = None, *,
                            read_consistency: int = 0, on_partition_unavailable: int = 0,
                            bound: int = 0) -> bytes:
    """Mirrors ops.EncodeVectorSearchArgsOpts. Shared by vector_search AND
    vector_search_docs — the two ops carry an IDENTICAL wire layout and differ only
    in the op string used at call time (search_docs additionally returns each hit's
    stored content, which is a server-side concern, not a wire-shape one)."""
    base = encode_search_args(collection, k, query, filter)
    if read_consistency == 0 and on_partition_unavailable == 0:
        return base  # byte-identical to the legacy/no-opts form
    out = bytearray(base)
    out[0] |= _F_SEARCH_OPTS
    out += bytes([read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    return bytes(out)


# search_docs shares vector_search's wire encoder exactly (ops.EncodeVectorSearchArgsOpts
# backs both vector_search and vector_search_docs); only the op name differs at call time.
encode_search_docs_args_opts = encode_search_args_opts


def encode_vector_get_batch_args(collection: str, ids: Sequence[int], flags: int = 0) -> bytes:
    """Mirrors ops.EncodeVectorGetBatchArgs. Wire:
    [colLen:u8][col][flags:u8][n:u32][id:u64 x n]."""
    out = bytearray(_col(collection))
    out += bytes([flags & 0xFF])
    out += struct.pack(">I", len(ids))
    for i in ids:
        out += struct.pack(">Q", i)
    return bytes(out)


def encode_scroll_cursor(last_id: int) -> str:
    """Mirrors ops.EncodeScrollCursor: base64.RawURLEncoding of
    [ver:u8=1][lastID:u64 BE]. Used CLIENT-SIDE to derive the next-page cursor
    when the server's vector_scroll response carries no wire cursor (the
    unpartitioned/single-node dispatch path — see the rostam.scrollNextCursor
    rule kv.py's scroll() replicates: a FULL page may have more)."""
    raw = bytes([1]) + struct.pack(">Q", last_id)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_scroll_cursor(token: str) -> Tuple[Optional[int], bool]:
    """Mirrors ops.DecodeScrollCursor (v1, id-only cursor): a scroll cursor is
    base64.RawURLEncoding of [ver:u8=1][lastID:u64 BE]. An empty token is the
    first page (returns (None, False)); the caller resumes at ids strictly
    greater than the decoded lastID. Raises ValueError on a malformed token,
    mirroring ops.ErrBadScrollCursor."""
    if not token:
        return None, False
    pad = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token + pad)
    except Exception as e:
        raise ValueError("malformed scroll cursor") from e
    if len(raw) != 9 or raw[0] != 1:
        raise ValueError("malformed scroll cursor")
    return struct.unpack(">Q", raw[1:9])[0], True


def _scroll_base(collection: str, limit: int, filter: Optional[Dict[str, Any]]) -> bytes:
    fj = b""
    if filter:
        fj = json.dumps(filter, separators=(",", ":")).encode("utf-8")
    out = bytearray(_col(collection))
    out += struct.pack(">I", limit)
    out += struct.pack(">I", len(fj)) + fj
    return bytes(out)


def _encode_scroll_order_block(order: Dict[str, Any]) -> bytes:
    """Mirrors ops.appendScrollOrderBlock. `order` shape:
    {key, desc=False, is_datetime=False, kind="numeric"|"datetime"|"string",
     has_start=False, start_from=0.0, has_resume=False, resume_key=0.0,
     has_resume_str=False, resume_str="",
     tail=[{key, desc=False, is_datetime=False, kind="numeric"}, ...],
     has_resume_keys=False, resume_keys=[{kind, num=0.0, str=""}, ...]}
    kind only changes wire shape for "string" (bit2 + the resume-str tail); the
    string-resume tail is written ONLY when kind=="string" (additive, matches Go).
    """
    key = order["key"].encode("utf-8")
    desc = bool(order.get("desc", False))
    is_datetime = bool(order.get("is_datetime", False))
    kind = order.get("kind", "numeric")
    tail = order.get("tail") or []

    flags = 0
    if desc:
        flags |= 1 << 0
    if is_datetime:
        flags |= 1 << 1
    if kind == "string":
        flags |= 1 << 2
    if tail:
        flags |= 1 << 3  # scrollOrderFlagMultiKey

    out = bytearray([1])  # orderPresent=1
    out += struct.pack(">I", len(key)) + key
    out += bytes([flags])

    if order.get("has_start"):
        out += bytes([1]) + struct.pack(">d", float(order["start_from"]))
    else:
        out += bytes([0])

    if order.get("has_resume"):
        out += bytes([1]) + struct.pack(">d", float(order["resume_key"]))
    else:
        out += bytes([0])

    if kind == "string":
        if order.get("has_resume_str"):
            rs = order["resume_str"].encode("utf-8")
            out += bytes([1]) + struct.pack(">I", len(rs)) + rs
        else:
            out += bytes([0])

    if tail:
        out += bytes([len(tail)])
        for tk in tail:
            tkey = tk["key"].encode("utf-8")
            out += struct.pack(">I", len(tkey)) + tkey
            tf = 0
            if tk.get("desc"):
                tf |= 1 << 0
            if tk.get("is_datetime"):
                tf |= 1 << 1
            if tk.get("kind") == "string":
                tf |= 1 << 2
            out += bytes([tf])
        if order.get("has_resume_keys"):
            out += bytes([1])
            for rv in order.get("resume_keys", []):
                rk = _ORDER_KIND[rv["kind"]]
                out += bytes([rk])
                if rv["kind"] == "string":
                    s = rv["str"].encode("utf-8")
                    out += struct.pack(">I", len(s)) + s
                else:
                    out += struct.pack(">d", float(rv["num"]))
        else:
            out += bytes([0])

    return bytes(out)


def encode_scroll_args_order_bounded(collection: str, limit: int, *,
                                     filter: Optional[Dict[str, Any]] = None,
                                     read_consistency: int = 0, on_partition_unavailable: int = 0,
                                     after_id: Optional[int] = None,
                                     order: Optional[Dict[str, Any]] = None,
                                     bound: int = 0) -> bytes:
    """Mirrors ops.EncodeScrollArgsOrderBounded. NOTE the asymmetry this replicates
    byte-for-byte: when order is None the opts+cursor trailer (and the cursor's
    cursorPresent=0 byte) is omitted ENTIRELY unless opts or a cursor are actually
    in use (EncodeScrollArgsCursorBounded); when order is given, the trailer AND an
    explicit cursorPresent byte (0 or 1) are always forced present, so the order
    block has an unambiguous, self-delimiting start position."""
    base = _scroll_base(collection, limit, filter)
    has_after = after_id is not None

    if order is None:
        if read_consistency == 0 and on_partition_unavailable == 0 and not has_after:
            return base  # byte-identical to the legacy form
        out = bytearray(base)
        out += bytes([1, read_consistency, on_partition_unavailable])
        out += _bound_tail(read_consistency, bound)
        if has_after:
            out += bytes([1]) + struct.pack(">Q", after_id)
        # else: no cursorPresent byte at all (matches EncodeScrollArgsCursorBounded)
        return bytes(out)

    out = bytearray(base)
    out += bytes([1, read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    if has_after:
        out += bytes([1]) + struct.pack(">Q", after_id)
    else:
        out += bytes([0])  # cursorPresent=0, forced present
    out += _encode_scroll_order_block(order)
    return bytes(out)


def _encode_sparse(sparse: Optional[Dict[str, Sequence]]) -> bytes:
    """Mirrors ops.writeSparse: [nnz:u32]{[dim:u32][value:f32]}."""
    idx = list(sparse["indices"]) if sparse else []
    val = list(sparse["values"]) if sparse else []
    if len(idx) != len(val):
        raise ValueError("sparse indices and values must have the same length")
    out = bytearray(struct.pack(">I", len(idx)))
    for i, v in zip(idx, val, strict=True):
        out += struct.pack(">I", i) + struct.pack(">f", v)
    return bytes(out)


def encode_group_search_args_opts(collection: str, k: int, query: Vector,
                                  opts: Optional[Dict[str, Any]] = None, *,
                                  read_consistency: int = 0, on_partition_unavailable: int = 0,
                                  bound: int = 0) -> bytes:
    """Mirrors ops.EncodeGroupSearchArgsOpts. `opts`: {group_by, group_size=0,
    fetch_k=0, filter=None}. Unlike search/hybrid, the filter block here is
    UNCONDITIONAL (no flag bit) — Go always writes [filterLen:u32][filterJSON],
    with filterJSON empty (len 0) when there is no filter."""
    opts = opts or {}
    group_by = str(opts.get("group_by", "")).encode("utf-8")
    if len(group_by) > 0xFFFF:
        raise ValueError("group_by too long")
    filt = opts.get("filter")
    fj = json.dumps(filt, separators=(",", ":")).encode("utf-8") if filt else b""

    out = bytearray(_col(collection))
    out += struct.pack(">I", k)
    out += struct.pack(">I", int(opts.get("group_size", 0)))
    out += struct.pack(">I", int(opts.get("fetch_k", 0)))
    out += struct.pack(">H", len(group_by)) + group_by
    out += _f32be(query)
    out += struct.pack(">I", len(fj)) + fj

    if read_consistency == 0 and on_partition_unavailable == 0:
        return bytes(out)
    out += bytes([1, read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    return bytes(out)


def encode_hybrid_search_args_opts(collection: str, dense: Vector, k: int,
                                   sparse: Optional[Dict[str, Sequence]] = None,
                                   opts: Optional[Dict[str, Any]] = None, *,
                                   read_consistency: int = 0, on_partition_unavailable: int = 0,
                                   bound: int = 0) -> bytes:
    """Mirrors ops.EncodeHybridSearchArgsOpts. `opts`: {filter=None, method="rrf",
    alpha=0.0, rrf_k=0, dense_k=0, sparse_k=0}."""
    opts = opts or {}
    filt = opts.get("filter")
    flags = 0
    fj = b""
    if filt:
        flags |= _HYBRID_F_FILTER
        fj = json.dumps(filt, separators=(",", ":")).encode("utf-8")
    has_sparse = bool(sparse and sparse.get("indices"))
    if has_sparse:
        flags |= _HYBRID_F_SPARSE

    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">I", k)
    out += bytes([_FUSION_METHOD[opts.get("method", "rrf")]])
    out += struct.pack(">d", float(opts.get("alpha", 0.0)))
    out += struct.pack(">I", int(opts.get("rrf_k", 0)))
    out += struct.pack(">I", int(opts.get("dense_k", 0)))
    out += struct.pack(">I", int(opts.get("sparse_k", 0)))
    out += _f32be(dense)
    if has_sparse:
        out += _encode_sparse(sparse)
    if flags & _HYBRID_F_FILTER:
        out += struct.pack(">I", len(fj)) + fj

    if read_consistency == 0 and on_partition_unavailable == 0:
        return bytes(out)
    out[0] |= _HYBRID_F_OPTS
    out += bytes([read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    return bytes(out)


def encode_hybrid_text_args_global(collection: str, dense: Vector, query: str, k: int,
                                   opts: Optional[Dict[str, Any]] = None, *,
                                   read_consistency: int = 0, on_partition_unavailable: int = 0,
                                   bound: int = 0, global_idf: bool = False) -> bytes:
    """Mirrors ops.EncodeHybridTextArgsGlobal. `opts`: {filter=None, method="rrf",
    alpha=0.0, rrf_k=0, dense_k=0, sparse_k=0}. The Go op also carries a
    coordinator-only phase-1 global-DF stats block (`g` / textFlagGlobalStats),
    but that block is populated by the coordinator itself when fanning a
    global_idf=True request out to shards — this client only ever sends the
    initial (no-`g`) request, so that block is intentionally not encoded here."""
    opts = opts or {}
    filt = opts.get("filter")
    flags = 0
    fj = b""
    if filt:
        flags |= _TEXT_F_FILTER
        fj = json.dumps(filt, separators=(",", ":")).encode("utf-8")
    has_opts = read_consistency != 0 or on_partition_unavailable != 0
    if has_opts:
        flags |= _TEXT_F_OPTS
    if global_idf:
        flags |= _TEXT_F_GLOBAL_IDF

    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">I", k)
    out += bytes([_FUSION_METHOD[opts.get("method", "rrf")]])
    out += struct.pack(">d", float(opts.get("alpha", 0.0)))
    out += struct.pack(">I", int(opts.get("rrf_k", 0)))
    out += struct.pack(">I", int(opts.get("dense_k", 0)))
    out += struct.pack(">I", int(opts.get("sparse_k", 0)))
    out += _f32be(dense)
    qb = query.encode("utf-8")
    out += struct.pack(">I", len(qb)) + qb
    if flags & _TEXT_F_FILTER:
        out += struct.pack(">I", len(fj)) + fj
    if has_opts:
        out += bytes([read_consistency, on_partition_unavailable])
        out += _bound_tail(read_consistency, bound)

    return bytes(out)


def _encode_insert_like(op_flag_prefix: bool, collection: str, id: int, vec: Vector, *,
                        content: Optional[str], ttl_ms: int,
                        metadata: Optional[Dict[str, Any]],
                        sparse: Optional[Dict[str, Sequence]]) -> bytes:
    """Shared body for insert/upsert. content!=None ⇒ upsert layout (JSON carries
    $content merged with metadata); content is None ⇒ insert layout."""
    flags = 0
    if ttl_ms > 0:
        flags |= _F_TTL
    # upsert folds content into the metadata JSON under $content
    if metadata and "$content" in metadata:
        raise ValueError("metadata key '$content' is reserved")
    meta_obj = dict(encode_metadata(metadata or {})) if metadata else {}
    if content is not None:
        meta_obj["$content"] = {"kind": "string", "str": content}
    has_meta = bool(meta_obj)
    if has_meta:
        flags |= _F_META
    if sparse:
        flags |= _F_SPARSE

    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">Q", id)
    out += _f32be(vec)
    if flags & _F_TTL:
        out += struct.pack(">Q", ttl_ms)
    if has_meta:
        mj = json.dumps(meta_obj, separators=(",", ":")).encode("utf-8")
        out += struct.pack(">I", len(mj)) + mj
    if sparse:
        idx = list(sparse["indices"])
        val = list(sparse["values"])
        if len(idx) != len(val):
            raise ValueError("sparse indices and values must have the same length")
        out += struct.pack(">I", len(idx))
        for i, v in zip(idx, val, strict=True):
            out += struct.pack(">I", i) + struct.pack(">f", v)
    return bytes(out)


def encode_insert_args(collection: str, id: int, vec: Vector, *, ttl_ms: int = 0,
                       metadata: Optional[Dict[str, Any]] = None,
                       sparse: Optional[Dict[str, Sequence]] = None) -> bytes:
    return _encode_insert_like(False, collection, id, vec, content=None,
                               ttl_ms=ttl_ms, metadata=metadata, sparse=sparse)


def encode_upsert_args(collection: str, id: int, vec: Vector, *, content: str = "",
                       ttl_ms: int = 0, metadata: Optional[Dict[str, Any]] = None,
                       sparse: Optional[Dict[str, Sequence]] = None) -> bytes:
    return _encode_insert_like(True, collection, id, vec, content=content,
                               ttl_ms=ttl_ms, metadata=metadata, sparse=sparse)


def encode_upsert_batch_args(collection: str, points: Sequence[Dict[str, Any]]) -> List[bytes]:
    """There is NO single native-TCP batch-upsert wire op. ops.EncodeVectorUpsertArgs
    is single-point only; the batch/bulk framing that does exist (RVB1, in
    httpapi/binary_bulk.go for POST /points/bulk and /points/bulk/build) is an
    HTTP-only staging protocol for the multi-core index build, not part of the
    ops.Encode* native-TCP family and not something a TCP client op can drive.
    So a native-TCP upsert_batch is just N sequential vector_upsert ops: this
    returns the list of per-point encode_upsert_args() outputs for the caller to
    send one at a time (each awaited before the next) over the same connection —
    not pipelined (matching the Go client, which also loops per-point).
    Each point dict: {id, vector, content="", ttl_ms=0, metadata=None, sparse=None}.
    """
    out = []
    for p in points:
        out.append(encode_upsert_args(
            collection, p["id"], p["vector"],
            content=p.get("content", ""),
            ttl_ms=p.get("ttl_ms", 0),
            metadata=p.get("metadata"),
            sparse=p.get("sparse"),
        ))
    return out


def encode_delete_args(collection: str, id: int) -> bytes:
    return _col(collection) + struct.pack(">Q", id)


encode_exists_args = encode_delete_args  # same layout


def encode_get_args(collection: str, id: int, flags: int = 0) -> bytes:
    return _col(collection) + struct.pack(">Q", id) + bytes([flags & 0xFF])


# ---- create_collection: the config trailer ---------------------------------
#
# Mirrors ops.EncodeCreateCollectionArgs. The trailer blocks are appended only
# when non-default, but each late block FORCES every earlier optional block
# present (with zero values) so the decoder's greedy length guards have fixed
# anchors. The forcing chain is replicated exactly; verified byte-for-byte
# against the Go encoder in the golden test.
def encode_create_collection_args(name: str, cfg: Dict[str, Any]) -> bytes:
    g = cfg.get

    dim = int(g("dim"))
    metric = _METRIC[str(g("metric", "cosine")).lower()]
    m = int(g("m", 0))
    efc = int(g("ef_construction", 0))
    efs = int(g("ef_search", 0))
    seed = int(g("seed", 0))
    quant = _QUANT[str(g("quant", "")).lower()]
    persistent = 1 if g("persistent") else 0
    rescore = int(g("rescore_factor", 0))
    extend = 1 if g("extend_candidates") else 0
    extend_max = int(g("extend_candidates_max", 0))
    l0full = 1 if g("level0_full_degree") else 0
    qbuild = 1 if g("quantized_build") else 0
    partitions = int(g("partitions", 0))
    index_type = _INDEX[str(g("index_type", "")).lower()]

    ivf_nlist = int(g("ivf_nlist", 0))
    ivf_nprobe = int(g("ivf_nprobe", 0))
    ivf_pq = bool(g("ivf_pq"))
    ivf_pq_m = int(g("ivf_pq_m", 0))
    ivf_rerank = bool(g("ivf_rerank"))
    quant_pq_m = int(g("quant_pq_m", 0))
    opq = bool(g("opq"))
    pq_drop_vecs = bool(g("pq_drop_vecs"))
    ivf_train_threshold = int(g("ivf_train_threshold", 0))
    drift_retrain = bool(g("ivf_drift_retrain"))
    drift_growth = float(g("ivf_drift_growth_factor", 0.0))
    drift_factor = float(g("ivf_drift_factor", 0.0))
    rel_bp = int(g("filter_first_relative_bp", 0))
    opq_iters = int(g("opq_iters", 0))
    full_text = g("full_text")  # None, True/analyzer dict, or falsy
    sq_bits = int(g("sq_bits", 0))
    prq_layers = int(g("prq_layers", 0))
    vamana_r = int(g("vamana_r", 0))
    vamana_l = int(g("vamana_l", 0))
    vamana_alpha = float(g("vamana_alpha", 0.0))
    anisotropic_eta = float(g("anisotropic_eta", 0.0))
    soar = bool(g("soar"))
    soar_lambda = float(g("soar_lambda", 0.0))
    pq_nbits = int(g("pq_nbits", 0)) == 4

    # Forcing chain (bottom-up), matching the Go booleans exactly.
    ft_present = full_text is not None and full_text is not False
    _soar_lambda = soar_lambda != 0 or pq_nbits
    _soar = soar or _soar_lambda
    _aniso = anisotropic_eta != 0 or _soar
    _vamana = vamana_r != 0 or vamana_l != 0 or vamana_alpha != 0 or _aniso
    _prq = prq_layers != 0 or _vamana
    _sq = sq_bits != 0 or _prq
    _ft_slot = ft_present or _sq
    _opq_iters = opq_iters != 0 or _ft_slot
    _rel_bp = rel_bp != 0 or _opq_iters
    _drift = drift_retrain or drift_growth != 0 or drift_factor != 0 or _rel_bp
    _threshold = ivf_train_threshold != 0 or _drift
    _ivfpq = ivf_pq or ivf_rerank or _threshold
    _ivf = index_type != 0 or ivf_nlist != 0 or ivf_nprobe != 0 or _ivfpq
    _quantpqm = quant_pq_m != 0 or _threshold

    nm = name.encode("utf-8")
    out = bytearray([len(nm)]) + nm
    out += struct.pack(">I", dim)
    out += bytes([metric])
    out += struct.pack(">I", m) + struct.pack(">I", efc) + struct.pack(">I", efs)
    out += struct.pack(">q", seed)
    out += bytes([quant, persistent])
    out += struct.pack(">I", rescore)
    out += bytes([extend])
    out += struct.pack(">I", extend_max)
    out += bytes([l0full, qbuild])
    out += struct.pack(">I", partitions)

    if _ivf:
        out += bytes([index_type]) + struct.pack(">I", ivf_nlist) + struct.pack(">I", ivf_nprobe)
    if _ivfpq:
        out += bytes([1 if ivf_pq else 0]) + struct.pack(">I", ivf_pq_m) + bytes([1 if ivf_rerank else 0])
    if _quantpqm:
        out += struct.pack(">I", quant_pq_m)
    if opq or pq_drop_vecs or _threshold:
        out += bytes([1 if opq else 0])
    if pq_drop_vecs or _threshold:
        out += bytes([1 if pq_drop_vecs else 0])
    if _threshold:
        out += struct.pack(">I", ivf_train_threshold)
    if _drift:
        out += bytes([1 if drift_retrain else 0]) + struct.pack(">d", drift_growth) + struct.pack(">d", drift_factor)
    if _rel_bp:
        out += struct.pack(">I", rel_bp)
    if _opq_iters:
        out += struct.pack(">I", opq_iters)
    if _ft_slot:
        # presence byte, then analyzer/k1/b only when a real FullText config
        if ft_present:
            an = (full_text.get("analyzer", "") if isinstance(full_text, dict) else "").encode("utf-8")
            k1 = float(full_text.get("k1", 0.0)) if isinstance(full_text, dict) else 0.0
            b = float(full_text.get("b", 0.0)) if isinstance(full_text, dict) else 0.0
            out += bytes([1, len(an)]) + an + struct.pack(">f", k1) + struct.pack(">f", b)
        else:
            out += bytes([0])
    if _sq:
        out += struct.pack(">I", sq_bits)
    if _prq:
        out += struct.pack(">I", prq_layers)
    # VamanaL / VamanaAlpha force VamanaR, so when any is set all three words are
    # written; each of Aniso / SOAR / SOARLambda / PQNBits is a SEPARATE word
    # appended only under its own flag (f32, not f64).
    if _vamana:
        out += struct.pack(">I", vamana_r) + struct.pack(">I", vamana_l) + struct.pack(">f", vamana_alpha)
    if _aniso:
        out += struct.pack(">f", anisotropic_eta)
    if _soar:
        out += bytes([1 if soar else 0])
    if _soar_lambda:
        out += struct.pack(">f", soar_lambda)
    if pq_nbits:
        out += struct.pack(">I", int(g("pq_nbits", 0)))
    return bytes(out)


def encode_drop_collection_args(name: str) -> bytes:
    """Mirrors ops.EncodeDropCollectionArgs: [nameLen u8][name] — the same
    length-prefixed name framing create_collection uses for its `name` field
    (see `_col`), just with no config trailer."""
    return _col(name)


# ---- vector_query: hand-rolled protobuf QuerySpec ---------------------------
#
# The vector_query op carries a proto-marshaled pb.QuerySpec as its spec blob
# (ops/query.go: EncodeQueryArgs → specBytes = proto.Marshal(querySpecToProto)).
# The stdlib-only Python client has no protobuf library, so the wire encoding is
# hand-rolled here to byte-match Go's proto.Marshal. proto3 rules replicated:
#   - key = (field_number << 3) | wire_type, itself a varint
#   - scalars are omitted when zero/empty (default); a set message oneof arm is
#     ALWAYS emitted (even an empty sub-message)
#   - fields are emitted in ASCENDING field-number order
#   - repeated scalar (varint) fields use PACKED encoding (one length-delimited
#     block of concatenated varints)
# Proto field map (from grpcapi/pb/rostam.pb.go):
#   QuerySpec: root=1(msg) prefetch=2(rep msg) mode=3(enum) fusion_method=4(str)
#              alpha=5(fixed64) rrf_k=6(varint) k=7(varint) prefetch_sources=8
#              group_by=9(str) group_size=10(varint)
#   QueryLeaf oneof: dense=1 sparse=2 named_dense=3 named_sparse=4 mv_maxsim=5
#              recommend=6(msg) discover=7
#   RecommendLeaf: positive=1(varint rep packed) negative=2(varint rep packed)
#              k=3(varint) filter_json=4(bytes) strategy=5(enum) best_pos=6
#              best_neg=7 space=8(str)
# Enum values: QueryMode FUSION=0 RERANK=1 ; RecommendStrategy AVERAGE=0 BEST=1.
# A recommend request (client Recommend) builds ModeFusion (=0 ⇒ omitted) with a
# single LeafRecommend prefetch lane and no root; fusion_method resolves to "rrf"
# (non-empty ⇒ always emitted) and spec.k == leaf.k.

_QUERY_MODE = {"fusion": 0, "rerank": 1}
_RECOMMEND_STRATEGY = {"average": 0, "average_vector": 0, "best": 1, "best_score": 1}
# Public alias: kv.py's recommend()/query() map a strategy NAME to the
# RecommendStrategy enum code through this table (0=average-vector, 1=best-score).
RECOMMEND_STRATEGY = _RECOMMEND_STRATEGY

# QueryLeaf oneof field numbers
_LEAF_RECOMMEND = 6
# QuerySpec field numbers
_SPEC_PREFETCH = 2
_SPEC_MODE = 3
_SPEC_FUSION_METHOD = 4
_SPEC_K = 7
# RecommendLeaf field numbers
_REC_POSITIVE = 1
_REC_NEGATIVE = 2
_REC_K = 3
_REC_FILTER_JSON = 4
_REC_STRATEGY = 5


def _pb_uvarint(n: int) -> bytes:
    """Base-128 unsigned varint (Go binary.PutUvarint / proto varint)."""
    if n < 0:
        n &= (1 << 64) - 1  # two's-complement into u64 (enums/ids are non-negative here)
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_key(field: int, wire: int) -> bytes:
    return _pb_uvarint((field << 3) | wire)


def _pb_len_delim(field: int, payload: bytes) -> bytes:
    """Wire type 2: [key][len:varint][payload]. Always emitted (caller decides)."""
    return _pb_key(field, 2) + _pb_uvarint(len(payload)) + payload


def _pb_varint_field(field: int, n: int) -> bytes:
    """Wire type 0 scalar, proto3 omit-zero."""
    if n == 0:
        return b""
    return _pb_key(field, 0) + _pb_uvarint(n)


def _pb_string_field(field: int, s: str) -> bytes:
    """Wire type 2 string, proto3 omit-empty."""
    b = s.encode("utf-8")
    if not b:
        return b""
    return _pb_len_delim(field, b)


def _pb_bytes_field(field: int, b: bytes) -> bytes:
    """Wire type 2 bytes, proto3 omit-empty."""
    if not b:
        return b""
    return _pb_len_delim(field, b)


def _pb_packed_varints(field: int, vals: Sequence[int]) -> bytes:
    """Repeated scalar varint, PACKED (proto3 default). Omit-empty."""
    if not vals:
        return b""
    payload = b"".join(_pb_uvarint(int(v)) for v in vals)
    return _pb_len_delim(field, payload)


def _marshal_recommend_leaf(positive: Sequence[int], negative: Sequence[int],
                            k: int, filter_json: bytes, strategy: int) -> bytes:
    """Marshal a pb.RecommendLeaf (fields in ascending number order)."""
    out = bytearray()
    out += _pb_packed_varints(_REC_POSITIVE, positive)
    out += _pb_packed_varints(_REC_NEGATIVE, negative)
    out += _pb_varint_field(_REC_K, k)
    out += _pb_bytes_field(_REC_FILTER_JSON, filter_json)
    out += _pb_varint_field(_REC_STRATEGY, strategy)
    return bytes(out)


def marshal_recommend_query_spec(*, positive: Sequence[int],
                                 negative: Optional[Sequence[int]] = None,
                                 k: int = 0,
                                 filter: Optional[Dict[str, Any]] = None,
                                 strategy: int = 0) -> bytes:
    """Hand-roll the marshaled pb.QuerySpec bytes for a RECOMMEND query, matching
    Go's proto.Marshal(querySpecToProto(recommendSpec(...))) byte-for-byte.

    Mirrors the Go client's Recommend spec shape: Mode=ModeFusion (enum 0, so the
    mode field is omitted), a single LeafRecommend prefetch lane (no root), and
    spec.K == leaf.K. fusion_method resolves to "rrf" and is always emitted.

    The per-leaf filter rides as filter_json — Go's json.Marshal(vector.Filter)
    output (struct-field order op/field/value, kind as a string). Pass `filter`
    as the already-Go-shaped tagged dict (e.g. {"op":"eq","field":...,"value":
    {"kind":"string","str":...}}); it is dumped with compact separators, matching
    Go's marshaler byte-for-byte.
    """
    positive = list(positive or [])
    negative = list(negative or [])
    fj = b""
    if filter:
        fj = json.dumps(filter, separators=(",", ":")).encode("utf-8")

    leaf_body = _marshal_recommend_leaf(positive, negative, k, fj, strategy)
    # QueryLeaf: the recommend oneof arm (field 6) is a set message — always emitted.
    query_leaf = _pb_len_delim(_LEAF_RECOMMEND, leaf_body)

    out = bytearray()
    out += _pb_len_delim(_SPEC_PREFETCH, query_leaf)  # prefetch[0] (repeated QueryLeaf)
    # mode: ModeFusion == 0 ⇒ omitted (proto3 default). RERANK would emit field 3.
    out += _pb_string_field(_SPEC_FUSION_METHOD, "rrf")
    out += _pb_varint_field(_SPEC_K, k)
    return bytes(out)


def encode_query_args(collection: str, spec_bytes: bytes, *,
                      read_consistency: int = 0, on_partition_unavailable: int = 0,
                      bound: int = 0) -> bytes:
    """Mirrors ops.EncodeQueryArgs. Wire:
    [colLen:u8][col][specLen:u32][specBytes][optsTrailer]

    optsTrailer is appendReadOptsTrailerBounded (ops/consistency.go): omitted when
    rc==0 && opa==0; else [marker][rc][opa](+[bound:u64]). The marker is
    readOptsTrailerMarker(1); for BOUNDED_STALENESS it also sets readOptsStalenessBit
    (2 ⇒ marker byte 3) and the 8-byte BE bound rides only then."""
    out = bytearray(_col(collection))
    out += struct.pack(">I", len(spec_bytes))
    out += spec_bytes
    if read_consistency == 0 and on_partition_unavailable == 0:
        return bytes(out)  # byte-identical to the no-trailer form
    if read_consistency == CONSISTENCY_BOUNDED_STALENESS:
        out += bytes([1 | 2, read_consistency, on_partition_unavailable])
        out += struct.pack(">Q", bound)
    else:
        out += bytes([1, read_consistency, on_partition_unavailable])
    return bytes(out)


def encode_recommend_query(collection: str, *, positive: Sequence[int],
                           negative: Optional[Sequence[int]] = None, k: int = 0,
                           filter: Optional[Dict[str, Any]] = None, strategy: int = 0,
                           read_consistency: int = 0, on_partition_unavailable: int = 0,
                           bound: int = 0) -> bytes:
    """Build the full vector_query op args for a RECOMMEND request: hand-roll the
    QuerySpec proto blob then frame it with EncodeQueryArgs. `strategy`: 0 =
    average-vector (default), 1 = best-score."""
    spec = marshal_recommend_query_spec(positive=positive, negative=negative, k=k,
                                        filter=filter, strategy=strategy)
    return encode_query_args(collection, spec, read_consistency=read_consistency,
                             on_partition_unavailable=on_partition_unavailable, bound=bound)


# ---- result decoders --------------------------------------------------------
def decode_search_results(body: bytes) -> List[Dict[str, Any]]:
    (count,) = struct.unpack(">I", body[:4])
    out = []
    off = 4
    for _ in range(count):
        rid = struct.unpack(">Q", body[off:off + 8])[0]
        dist = struct.unpack(">f", body[off + 8:off + 12])[0]
        out.append({"id": rid, "distance": dist})
        off += 12
    return out


def decode_search_results_degraded(body: bytes) -> Tuple[List[Dict[str, Any]], bool, List[int]]:
    """Like decode_search_results, but also reads the optional degraded trailer
    appended after the base [count:u32]{[id:u64][distance:f32]} block. Mirrors
    ops.DecodeVectorSearchResultsDegraded: the base block is read exactly
    (count*12 bytes after the count u32), then any remaining bytes are the
    degraded trailer (absent → degraded=False, missing=[]). Bounds-checked:
    a truncated body raises ValueError rather than over-reading."""
    count, off = _read_u32(body, 0, "search results count")
    out = []
    for _ in range(count):
        rid, off = _read_u64(body, off, "search result id")
        dist, off = _read_f32(body, off, "search result distance")
        out.append({"id": rid, "distance": dist})
    degraded, missing, _ = _read_degraded_trailer(body, off)
    return out, degraded, missing


def decode_exists_result(body: bytes) -> bool:
    return bool(body and body[0])


def decode_get_result(body: bytes) -> Optional[Dict[str, Any]]:
    """Decode a vector_get result. Returns None if the point is absent, else a
    dict with id-independent fields: vector, metadata (tagged→plain), ttl_ms,
    sparse. Fields the request did not ask for come back empty."""
    if not body or body[0] == 0:
        return None
    off = 1
    (dim,) = struct.unpack(">I", body[off:off + 4]); off += 4
    vec = None
    if dim:
        vec = [struct.unpack(">f", body[off + 4 * i:off + 4 * i + 4])[0] for i in range(dim)]
        off += 4 * dim
    (ttl_ms,) = struct.unpack(">Q", body[off:off + 8]); off += 8
    meta = None
    if body[off]:
        off += 1
        (mlen,) = struct.unpack(">I", body[off:off + 4]); off += 4
        meta = decode_metadata(json.loads(body[off:off + mlen])); off += mlen
    else:
        off += 1
    sparse = None
    if body[off]:
        off += 1
        (nnz,) = struct.unpack(">I", body[off:off + 4]); off += 4
        idx, val = [], []
        for _ in range(nnz):
            idx.append(struct.unpack(">I", body[off:off + 4])[0])
            val.append(struct.unpack(">f", body[off + 4:off + 8])[0])
            off += 8
        sparse = {"indices": idx, "values": val}
    return {"vector": vec, "metadata": meta or {}, "ttl_ms": ttl_ms, "sparse": sparse}


# ---- Phase C: batch/scroll/docs/groups/hybrid/query result decoders --------
#
# These mirror the Go decoders in ops/vector.go and ops/query.go: read_only the
# same layout, degraded-trailer-tolerant where the Go decoder is (a missing
# trailer decodes as degraded=False, missing=[]).
#
# Every length prefix these decoders read (count/clen/mlen/nnz/dim/...) is
# validated against the remaining buffer before it's used to slice, via the
# _need/_read_* helpers below — a truncated or hostile response raises a clear
# ValueError instead of over-reading (silently returning garbage) or
# over-allocating.

def _need(body: bytes, off: int, n: int, what: str) -> None:
    """Raise ValueError unless body[off:off + n] is fully within bounds."""
    if off < 0 or n < 0 or off + n > len(body):
        raise ValueError(
            f"corrupt/truncated {what} response: need {n} bytes at offset {off}, "
            f"body is only {len(body)} bytes")


def _read_u32(body: bytes, off: int, what: str) -> Tuple[int, int]:
    _need(body, off, 4, what)
    return struct.unpack(">I", body[off:off + 4])[0], off + 4


def _read_u64(body: bytes, off: int, what: str) -> Tuple[int, int]:
    _need(body, off, 8, what)
    return struct.unpack(">Q", body[off:off + 8])[0], off + 8


def _read_f32(body: bytes, off: int, what: str) -> Tuple[float, int]:
    _need(body, off, 4, what)
    return struct.unpack(">f", body[off:off + 4])[0], off + 4


def _read_bytes(body: bytes, off: int, n: int, what: str) -> Tuple[bytes, int]:
    _need(body, off, n, what)
    return body[off:off + n], off + n


def _read_flag(body: bytes, off: int, what: str) -> Tuple[bool, int]:
    _need(body, off, 1, what)
    return bool(body[off]), off + 1


def _read_degraded_trailer(body: bytes, off: int) -> Tuple[bool, List[int], int]:
    """Mirrors ops.readDegradedTrailerN: [degraded:u8][missingCount:u16]{partID:u16}
    at body[off:]. A fully-absent trailer (exactly zero bytes remaining — the
    single-node / no-trailer body) is the only tolerated short case and returns
    (False, [], off). Any bytes present but too few for a complete trailer are a
    truncated/corrupt frame and raise ValueError, matching this module's decoder
    contract (rather than silently returning incomplete degraded metadata)."""
    remaining = len(body) - off
    if remaining == 0:
        return False, [], off
    if remaining < 3:
        raise ValueError(
            f"corrupt/truncated degraded trailer: {remaining} byte(s) at offset "
            f"{off}, need >= 3 for the [degraded:u8][missingCount:u16] header"
        )
    degraded = bool(body[off])
    (missing_count,) = struct.unpack(">H", body[off + 1:off + 3])
    trailer_end = off + 3 + 2 * missing_count
    if len(body) < trailer_end:
        raise ValueError(
            f"corrupt/truncated degraded trailer: declares {missing_count} "
            f"missing partition id(s) but body ends at {len(body)} (need {trailer_end})"
        )
    missing = []
    p = off + 3
    for _ in range(missing_count):
        missing.append(struct.unpack(">H", body[p:p + 2])[0])
        p += 2
    return degraded, missing, trailer_end


def _decode_docs_raw(body: bytes, off: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """Mirrors ops.frameVectorDocsN + the typed unmarshal: one EncodeVectorDocs
    block at body[off:]. Wire per doc: [id:u64][distance:f32][score:f32]
    [contentLen:u32][content][metaLen:u32][metaJSON]. Returns (docs, next_off).
    Raises ValueError (not struct.error/IndexError) on a truncated body."""
    count, off = _read_u32(body, off, "docs count")
    docs = []
    for _ in range(count):
        rid, off = _read_u64(body, off, "doc id")
        dist, off = _read_f32(body, off, "doc distance")
        score, off = _read_f32(body, off, "doc score")
        clen, off = _read_u32(body, off, "doc content length")
        content_b, off = _read_bytes(body, off, clen, "doc content")
        content = content_b.decode("utf-8")
        mlen, off = _read_u32(body, off, "doc metadata length")
        meta_b, off = _read_bytes(body, off, mlen, "doc metadata")
        meta = decode_metadata(json.loads(meta_b)) if mlen else {}
        docs.append({"id": rid, "distance": dist, "score": score, "content": content, "metadata": meta})
    return docs, off


def decode_docs_degraded_raw(body: bytes) -> Tuple[List[Dict[str, Any]], bool, List[int]]:
    """Mirrors ops.DecodeVectorDocsDegradedRaw. Used by search_docs (and shares
    the doc framing scroll/groups use)."""
    docs, off = _decode_docs_raw(body)
    degraded, missing, _ = _read_degraded_trailer(body, off)
    return docs, degraded, missing


def decode_scroll_result_raw(body: bytes) -> Tuple[List[Dict[str, Any]], bool, List[int], str]:
    """Mirrors ops.DecodeScrollResultRaw: the doc block, the ALWAYS-present
    degraded trailer, then the [cursorLen:u32][cursorBytes] next_cursor tail
    (may be zero-length ⇒ ""). Tolerant of an old/short body (empty cursor)."""
    docs, off = _decode_docs_raw(body)
    degraded, missing, off = _read_degraded_trailer(body, off)
    next_cursor = ""
    # Deliberately tolerant (not _read_u32/_read_bytes, which raise): an
    # old/short body with no cursor tail at all is a valid, expected shape
    # (single-node server, see scroll()'s caller), not a corrupt response.
    if len(body) >= off + 4:
        (clen,) = struct.unpack(">I", body[off:off + 4]); off += 4
        if clen > 0 and len(body) >= off + clen:
            next_cursor = body[off:off + clen].decode("utf-8")
    return docs, degraded, missing, next_cursor


def decode_groups_degraded_raw(body: bytes) -> Tuple[List[Dict[str, Any]], bool, List[int]]:
    """Mirrors ops.DecodeGroupsDegradedRaw. Wire: [count:u32]{[keyLen:u32][keyJSON]
    [hitsLen:u32][hits: EncodeVectorDocs block]} then the degraded trailer. Each
    group's key is a single tagged Value (json.Marshal of vector.Value), decoded
    with decode_value (NOT decode_metadata, which expects a map). Raises
    ValueError on a truncated body."""
    count, off = _read_u32(body, 0, "groups count")
    groups = []
    for _ in range(count):
        klen, off = _read_u32(body, off, "group key length")
        key_b, off = _read_bytes(body, off, klen, "group key")
        key = decode_value(json.loads(key_b))
        dlen, off = _read_u32(body, off, "group hits length")
        hits_b, off = _read_bytes(body, off, dlen, "group hits")
        hits, _ = _decode_docs_raw(hits_b, 0)
        groups.append({"key": key, "hits": hits})
    degraded, missing, _ = _read_degraded_trailer(body, off)
    return groups, degraded, missing


def _decode_hybrid_results_block(body: bytes, off: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """Mirrors ops.decodeHybridResultsN: [count:u32]{[id:u64][distance:f32]
    [score:f32]}. Shared by hybrid_search/hybrid_text and the recommend/query
    flat-fused result (which carries the SAME per-row shape). Raises
    ValueError on a truncated body."""
    count, off = _read_u32(body, off, "hybrid results count")
    results = []
    for _ in range(count):
        rid, off = _read_u64(body, off, "hybrid result id")
        dist, off = _read_f32(body, off, "hybrid result distance")
        score, off = _read_f32(body, off, "hybrid result score")
        results.append({"id": rid, "distance": dist, "score": score})
    return results, off


def decode_hybrid_results_degraded(body: bytes) -> Tuple[List[Dict[str, Any]], bool, List[int]]:
    """Mirrors ops.DecodeHybridResultsDegraded. Used by hybrid_search/hybrid_text."""
    results, off = _decode_hybrid_results_block(body)
    degraded, missing, _ = _read_degraded_trailer(body, off)
    return results, degraded, missing


# queryResultModeRerank (ops/query.go): the tag a flat fused/recommend query
# result carries. recommend/query build a single-leaf ModeFusion spec whose
# server-side result the coordinator always re-encodes flat under this tag
# (see EncodeQueryResultFused) — a FUSION-tagged (unfused-lanes) body is not a
# valid result here.
_QUERY_RESULT_MODE_RERANK = 1


def decode_query_result_degraded(body: bytes) -> Tuple[List[Dict[str, Any]], bool, List[int]]:
    """Mirrors ops.DecodeQueryResultDegraded: a mode-tagged flat fused result
    (used by recommend/query) plus the degraded trailer. Fails loud if the body
    is not RERANK-tagged (mirroring the Go decoder's fail-loud contract), or is
    truncated."""
    _need(body, 0, 1, "query result mode byte")
    mode = body[0]
    if mode != _QUERY_RESULT_MODE_RERANK:
        raise ValueError(f"query result mode {mode} is not a flat fused result")
    results, off = _decode_hybrid_results_block(body, 1)
    degraded, missing, _ = _read_degraded_trailer(body, off)
    return results, degraded, missing


def _decode_get_body_after_found(body: bytes, off: int, version_framed: bool) -> Tuple[Dict[str, Any], int]:
    """Mirrors ops.decodeGetResultAtArena from just AFTER the leading [found=1]
    byte: [dim:u32][vec][ttl:u64][metaPresent:u8][?metaLen:u32][?metaJSON]
    [sparsePresent:u8][?sparse] then a trailing version block. version_framed
    selects how that block is read:
      - True (batch row): ALWAYS framed ([verPresent:u8][?version:u64]) so the
        record self-delimits and the next row's id is found unambiguously.
      - False (single get): OPTIONAL — read only when bytes remain.
    Returns ({vector, metadata, ttl_ms, sparse, version}, next_off). Raises
    ValueError on a truncated body."""
    dim, off = _read_u32(body, off, "get vector dim")
    vec = None
    if dim:
        vec_b, off = _read_bytes(body, off, 4 * dim, "get vector data")
        vec = [struct.unpack(">f", vec_b[i * 4:i * 4 + 4])[0] for i in range(dim)]
    ttl_ms, off = _read_u64(body, off, "get ttl")
    meta_present, off = _read_flag(body, off, "get metadata-present flag")
    meta = None
    if meta_present:
        mlen, off = _read_u32(body, off, "get metadata length")
        meta_b, off = _read_bytes(body, off, mlen, "get metadata")
        meta = decode_metadata(json.loads(meta_b))
    sparse_present, off = _read_flag(body, off, "get sparse-present flag")
    sparse = None
    if sparse_present:
        nnz, off = _read_u32(body, off, "get sparse nnz")
        idx, val = [], []
        for _ in range(nnz):
            entry_b, off = _read_bytes(body, off, 8, "get sparse entry")
            idx.append(struct.unpack(">I", entry_b[0:4])[0])
            val.append(struct.unpack(">f", entry_b[4:8])[0])
        sparse = {"indices": idx, "values": val}
    version = 0
    if version_framed:
        ver_present, off = _read_flag(body, off, "get batch version-present flag")
        if ver_present:
            version, off = _read_u64(body, off, "get batch version")
    elif off < len(body):
        ver_present, off = _read_flag(body, off, "get version-present flag")
        if ver_present:
            version, off = _read_u64(body, off, "get version")
    return {"vector": vec, "metadata": meta or {}, "ttl_ms": ttl_ms, "sparse": sparse, "version": version}, off


def decode_get_batch_result(body: bytes) -> List[Dict[str, Any]]:
    """Mirrors ops.DecodeVectorGetBatchResult. Wire: [n:u32] then per row
    [id:u64][found:u8] followed, when found, by the SAME record layout a single
    vector_get carries (see _decode_get_body_after_found), with an ALWAYS-framed
    trailing version block so rows self-delimit. Returns a list of
    {id, found, vector, metadata, ttl_ms, sparse, version} — a not-found row
    carries only id/found (the rest default to the not-found shape). Raises
    ValueError on a truncated body."""
    n, off = _read_u32(body, 0, "get_batch count")
    rows = []
    for _ in range(n):
        rid, off = _read_u64(body, off, "get_batch row id")
        found, off = _read_flag(body, off, "get_batch found flag")
        if not found:
            rows.append({"id": rid, "found": False, "vector": None, "metadata": {}, "ttl_ms": 0, "sparse": None, "version": 0})
            continue
        rec, off = _decode_get_body_after_found(body, off, version_framed=True)
        rec["id"] = rid
        rec["found"] = True
        rows.append(rec)
    return rows
